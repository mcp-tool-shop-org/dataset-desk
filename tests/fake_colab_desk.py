#!/usr/bin/env python3
"""Hermetic fake `colab` CLI and fake Colab VM tools for CPU-only tests.

Every invocation runs with `#SDLAB_FAKE_COLAB_CLI=1` (`SDLAB_TEST_FAKE_COLAB=1`) and isolated
`HOME` / `USERPROFILE` / `XDG_*` temp directories so real Colab CLI state or live Colab windows
on the host are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
import zipfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LAUNCH_SH = REPO_ROOT / "scripts" / "colab" / "launch_desk_draft.sh"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def to_posix(path: pathlib.Path | str) -> str:
    return pathlib.Path(path).resolve().as_posix()


def resolve_git_bash() -> str:
    candidates = [
        "C:/Program Files/Git/bin/bash.exe",
        "C:/Program Files/Git/usr/bin/bash.exe",
    ]
    if os.name == "nt":
        for cand in candidates:
            if pathlib.Path(cand).is_file():
                return cand
    found = shutil.which("bash")
    if not found:
        raise RuntimeError("bash not found on PATH")
    return found


def _append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_json(path: pathlib.Path, default: dict | None = None) -> dict:
    if not path.is_file():
        return {} if default is None else dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _write_bash_shim(path: pathlib.Path, body: str, *, include_fake_marker: bool = False) -> None:
    lines = ["#!/usr/bin/env bash"]
    if include_fake_marker:
        lines.append("#SDLAB_FAKE_COLAB_CLI=1")
    lines.append("set -euo pipefail")
    lines.append("export MSYS_NO_PATHCONV=1")
    lines.append('export MSYS2_ARG_CONV_EXCL="*"')
    lines.append(body.rstrip())
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    if os.name != "nt":
        path.chmod(0o755)


def _local_path(raw: str) -> pathlib.Path:
    norm = raw.replace("\\", "/")
    if os.name == "nt":
        m = re.match(r"^/([a-zA-Z])/(.*)$", norm)
        if m:
            return pathlib.Path(f"{m.group(1).upper()}:/{m.group(2)}")
    return pathlib.Path(raw)


def _map_vm_path(remote_path: str, vm_content: pathlib.Path) -> pathlib.Path:
    norm = remote_path.replace("\\", "/")
    if norm == "/content" or norm.lower().endswith("/git/content"):
        return vm_content
    if norm.startswith("/content/"):
        rel = norm[len("/content/") :]
        return vm_content / rel
    m = re.search(r"(?i)/git/content/(.+)$", norm)
    if m:
        return vm_content / m.group(1)
    return _local_path(remote_path)


def _rewrite_content_paths(text: str, vm_content_posix: str) -> str:
    return re.sub(r"(?<![A-Za-z0-9_.-])/content(?=/|\"|'|\s|$)", vm_content_posix, text)


def _mutate_bundle_for_fault(bundle_path: pathlib.Path, cfg: dict) -> None:
    del_rel = cfg.get("delete_output_file")
    corrupt_rel = cfg.get("corrupt_output_file")
    if not (del_rel or corrupt_rel) or not bundle_path.is_file():
        return
    stage = pathlib.Path(tempfile.mkdtemp(prefix="fake_mut_bundle_"))
    try:
        with tarfile.open(bundle_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    dst = stage / member.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    ext = tf.extractfile(member)
                    if ext is not None:
                        with dst.open("wb") as out_fh:
                            shutil.copyfileobj(ext, out_fh)
        if del_rel:
            target = stage / del_rel if (stage / del_rel).is_file() else stage / "desk_out" / del_rel
            if target.is_file():
                target.unlink()
        if corrupt_rel:
            target = stage / corrupt_rel if (stage / corrupt_rel).is_file() else stage / "desk_out" / corrupt_rel
            if target.is_file():
                target.write_bytes(target.read_bytes() + b"\ncorrupted-after-vm-manifest\n")
        with tarfile.open(bundle_path, "w:gz", compresslevel=6) as tf:
            for fp in sorted(stage.rglob("*")):
                if not fp.is_file():
                    continue
                rel_posix = fp.relative_to(stage).as_posix()
                ti = tarfile.TarInfo(name=rel_posix)
                st = fp.stat()
                ti.size = st.st_size
                ti.mtime = 0
                ti.mode = 0o644
                with fp.open("rb") as fh:
                    tf.addfile(ti, fh)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


STANDIN_HF_HUB_CODE = '''"""Offline stand-in huggingface_hub module for fake Colab VM tests."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from types import SimpleNamespace


class HfHubHttpError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = SimpleNamespace(status_code=status_code)


def _state_dir() -> pathlib.Path:
    return pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])


def _cfg() -> dict:
    p = _state_dir() / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _log(rec: dict) -> None:
    p = _state_dir() / "hf_calls.jsonl"
    with p.open("a", encoding="utf-8", newline="\\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\\n")


def _maybe_429(cfg: dict, op: str) -> int:
    limit = int(cfg.get("weight_429_count") or 0)
    cnt_file = _state_dir() / "hf_429_count.txt"
    cur = int(cnt_file.read_text(encoding="utf-8").strip()) if cnt_file.is_file() else 0
    if cur < limit:
        nxt = cur + 1
        cnt_file.write_text(f"{nxt}\\n", encoding="utf-8", newline="\\n")
        _log({"op": op, "status": 429, "attempt": nxt})
        raise HfHubHttpError(f"HTTP 429 Too Many Requests (simulated {nxt}/{limit})", 429)
    return cur + 1


def _payloads(revision: str, cfg: dict) -> dict[str, bytes]:
    commit_in_cfg = "0000000000000000000000000000000000000000" if cfg.get("config_wrong_commit_hash") else revision
    config_bytes = (
        json.dumps(
            {
                "architectures": ["Qwen3VLForConditionalGeneration"],
                "model_type": "qwen3_vl",
                "_commit_hash": commit_in_cfg,
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\\n"
    )
    preproc_bytes = b\'{"image_processor_type": "Qwen2VLImageProcessor"}\\n\'
    tok_bytes = b\'{"tokenizer_class": "Qwen2Tokenizer"}\\n\'
    weights_bytes = b"QWEN3-VL-FAKE-SAFETENSORS-HEADER:" + revision.encode("ascii") + (b"\\x00" * 512)
    files = {
        ".gitattributes": b"*.safetensors filter=lfs diff=lfs merge=lfs -text\\n",
        "README.md": b"# Qwen3-VL-8B-Instruct (fake)\\n",
        "chat_template.json": b"{}\\n",
        "config.json": config_bytes,
        "generation_config.json": b"{}\\n",
        "merges.txt": b"#version: 0.2\\n",
        "model.safetensors.index.json": b"{}\\n",
        "preprocessor_config.json": preproc_bytes,
        "tokenizer.json": b"{}\\n",
        "tokenizer_config.json": tok_bytes,
        "video_preprocessor_config.json": b"{}\\n",
        "vocab.json": b"{}\\n",
    }
    for shard in range(1, 5):
        files[f"model-0000{shard}-of-00004.safetensors"] = weights_bytes + bytes([shard])
    if cfg.get("weight_omit_vocab"):
        files.pop("vocab.json")
    if cfg.get("weight_extra_file"):
        files["extra.bin"] = b"extra"
    return files


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\\0" + data).hexdigest()


class HfApi:
    def model_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        files_metadata: bool = False,
        token: str | None = None,
    ):
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise RuntimeError("HfApi.model_info called while HF_HUB_OFFLINE=1")
        cfg = _cfg()
        attempt = _maybe_429(cfg, "model_info")
        rev = str(revision or "")
        files = _payloads(rev, cfg)
        siblings = [
            SimpleNamespace(
                rfilename=name,
                blob_id=_blob_id(data),
                lfs={"sha256": hashlib.sha256(data).hexdigest()} if name.endswith(".safetensors") else None,
            )
            for name, data in sorted(files.items())
        ]
        resolved_sha = "deadbeef" * 5 if cfg.get("weight_wrong_revision") else rev
        _log(
            {
                "attempt": attempt,
                "files_metadata": bool(files_metadata),
                "hf_env_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
                "op": "model_info",
                "repo_id": repo_id,
                "revision": rev,
                "token_passed": bool(token),
            }
        )
        return SimpleNamespace(id=repo_id, sha=resolved_sha, siblings=siblings)


def snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    local_files_only: bool = False,
) -> str:
    cfg = _cfg()
    rev = str(revision or "")
    base_cache = pathlib.Path(cache_dir or (pathlib.Path(os.environ["HF_HOME"]) / "hub"))
    safe_repo = repo_id.replace("/", "--")
    snap_dir = base_cache / f"models--{safe_repo}" / "snapshots" / rev
    if local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1":
        if not snap_dir.is_dir():
            raise FileNotFoundError(f"offline snapshot missing: {snap_dir}")
        return str(snap_dir)
    attempt = _maybe_429(cfg, "snapshot_download")
    snap_dir.mkdir(parents=True, exist_ok=True)
    files = _payloads(rev, cfg)
    for name, data in sorted(files.items()):
        out_data = data
        if cfg.get("weight_corrupt_sha") and name.endswith(".safetensors"):
            out_data = data + b"CORRUPTED"
        if cfg.get("weight_tamper_tokenizer") and name == "tokenizer.json":
            out_data = data + b" "
        (snap_dir / name).write_bytes(out_data)
    _log(
        {
            "attempt": attempt,
            "hf_env_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
            "op": "snapshot_download",
            "repo_id": repo_id,
            "revision": rev,
            "snapshot_dir": str(snap_dir),
            "token_passed": bool(token),
        }
    )
    return str(snap_dir)
'''


STANDIN_TORCH_CODE = '''"""Lightweight CPU-only stand-in torch module for fake Colab VM tests."""
from __future__ import annotations

import contextlib
import os

bfloat16 = "bfloat16"
float32 = "float32"


def set_num_threads(n: int) -> None:
    return None


def set_num_interop_threads(n: int) -> None:
    return None


def is_tensor(obj: object) -> bool:
    return False


@contextlib.contextmanager
def no_grad():
    yield


class _CudaNamespace:
    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def mem_get_info(device: object = None) -> tuple[int, int]:
        vram_mib = int(os.environ.get("FAKE_TORCH_VRAM_MIB", "23034"))
        total = vram_mib * (1 << 20)
        return total, total


cuda = _CudaNamespace()
'''


STANDIN_TRANSFORMERS_CODE = '''"""Offline stand-in transformers module for fake Colab VM tests."""
from __future__ import annotations

import json
import os
import pathlib
import time
from types import SimpleNamespace

import numpy as np


def _cfg() -> dict:
    sd = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
    if not sd:
        return {}
    p = pathlib.Path(sd) / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _log(rec: dict) -> None:
    sd = os.environ.get("FAKE_COLAB_STATE_DIR", "").strip()
    if not sd:
        return
    p = pathlib.Path(sd) / "hf_calls.jsonl"
    with p.open("a", encoding="utf-8", newline="\\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\\n")


def _resolve_snapshot(repo_id: str, revision: str | None, local_files_only: bool) -> tuple[pathlib.Path, dict]:
    if not local_files_only:
        raise RuntimeError("Expected local_files_only=True after weight prefetch")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Expected HF_HUB_OFFLINE=1 during model/config load")
    hf_home = pathlib.Path(os.environ["HF_HOME"])
    rev = str(revision or "")
    snap = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / rev
    cfg_file = snap / "config.json"
    if not cfg_file.is_file():
        raise FileNotFoundError(f"missing cached config.json at {cfg_file}")
    return snap, json.loads(cfg_file.read_text(encoding="utf-8"))


class AutoConfig:
    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None, local_files_only: bool = False, **_kw):
        _, cfg_doc = _resolve_snapshot(repo_id, revision, local_files_only)
        _log(
            {
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
                "local_files_only": bool(local_files_only),
                "op": "AutoConfig.from_pretrained",
                "repo_id": repo_id,
                "revision": str(revision or ""),
            }
        )
        return SimpleNamespace(
            model_type=cfg_doc.get("model_type", "qwen3_vl"),
            _name_or_path=repo_id,
            _commit_hash=cfg_doc.get("_commit_hash", revision),
        )


class _BatchInputs(dict):
    def to(self, _device: str):
        return self


class AutoProcessor:
    def __init__(self, repo_id: str, commit_hash: str) -> None:
        self._name_or_path = repo_id
        self._commit_hash = commit_hash
        self._last_tone = "crimson"

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None, local_files_only: bool = False, **_kw):
        _, cfg_doc = _resolve_snapshot(repo_id, revision, local_files_only)
        _log(
            {
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
                "local_files_only": bool(local_files_only),
                "op": "AutoProcessor.from_pretrained",
                "repo_id": repo_id,
                "revision": str(revision or ""),
            }
        )
        return cls(repo_id, str(cfg_doc.get("_commit_hash", revision)))

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True) -> str:
        prompt_txt = ""
        for msg in messages:
            for item in msg.get("content", []):
                if item.get("type") == "text":
                    prompt_txt = str(item.get("text", ""))
        return f"<|im_start|>user\\n{prompt_txt}<|im_end|>\\n<|im_start|>assistant\\n"

    def __call__(self, *, text=None, images=None, padding: bool = True, return_tensors: str = "pt"):
        tone = "stone"
        if images:
            im0 = images[0]
            arr = np.asarray(im0.convert("RGB").resize((8, 8)), dtype=np.float32)
            r, g, b = arr.reshape(-1, 3).mean(axis=0)
            tone = "crimson" if r > max(g, b) else ("emerald" if g > max(r, b) else "cobalt")
        tone_code = {"crimson": 11, "emerald": 12, "cobalt": 13}.get(tone, 14)
        in_ids = np.array([[101, 102, tone_code]], dtype=np.int64)
        return _BatchInputs(input_ids=in_ids)

    def batch_decode(self, ids, *, skip_special_tokens: bool = False) -> list[str]:
        arr = np.asarray(ids[0], dtype=np.int64)
        code = int(arr[-1]) if arr.size else 21
        tone = {21: "crimson", 22: "emerald", 23: "cobalt"}.get(code, "weathered")
        prose = (
            f"a weathered {tone} masonry pillar stands beside damp flagstones under low overcast courtyard light"
        )
        if skip_special_tokens:
            return [prose]
        return [f"<|im_start|>assistant\\n{prose}<|im_end|>"]


class AutoModelForImageTextToText:
    def __init__(self, repo_id: str, commit_hash: str) -> None:
        self.config = SimpleNamespace(
            model_type="qwen3_vl",
            _name_or_path=repo_id,
            _commit_hash=commit_hash,
        )
        self.device = "cpu"

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        dtype: object = None,
        **_kw,
    ):
        _, cfg_doc = _resolve_snapshot(repo_id, revision, local_files_only)
        load_sleep_s = float(_cfg().get("model_load_sleep_s") or 0.0)
        if load_sleep_s > 0.0:
            time.sleep(load_sleep_s)
        _log(
            {
                "dtype": str(dtype),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", ""),
                "local_files_only": bool(local_files_only),
                "op": "AutoModelForImageTextToText.from_pretrained",
                "repo_id": repo_id,
                "revision": str(revision or ""),
            }
        )
        return cls(repo_id, str(cfg_doc.get("_commit_hash", revision)))

    def eval(self):
        return self

    def to(self, device: str):
        self.device = device
        return self

    def generate(self, *, input_ids, max_new_tokens: int = 256, do_sample: bool = False, **_kw):
        in_arr = np.asarray(input_ids, dtype=np.int64)
        tone_in = int(in_arr[0, -1])
        out_tok = tone_in + 10
        gen = np.array([[201, out_tok]], dtype=np.int64)
        return np.concatenate([in_arr, gen], axis=1)
'''


def _venv_site_packages(venv_dir: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        sp = venv_dir / "Lib" / "site-packages"
        sp.mkdir(parents=True, exist_ok=True)
        return sp
    lib_dir = venv_dir / "lib"
    candidates = sorted(lib_dir.glob("python*/site-packages"))
    if candidates:
        return candidates[0]
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sp = lib_dir / py_ver / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    return sp


def _outer_site_dirs() -> list[str]:
    out: list[str] = []
    for p in sys.path:
        if not p:
            continue
        rp = pathlib.Path(p)
        if rp.is_dir() and ("site-packages" in rp.name or "dist-packages" in rp.name):
            out.append(str(rp.resolve()))
    return out


def _cmd_colab_cli(argv: list[str]) -> int:
    state_dir = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
    cfg = _read_json(state_dir / "config.json")
    sess_file = state_dir / "session_state.json"
    sess_state = _read_json(
        sess_file,
        {
            "active": False,
            "card": "",
            "card_class": "cpu",
            "open_count": 0,
            "session": "",
            "shape_label": "High-RAM",
            "stop_count": 0,
        },
    )
    vm_content = pathlib.Path(cfg["vm_content"])
    vm_bin = pathlib.Path(cfg["vm_bin"])

    subcmd = argv[0] if argv else ""
    leaked = any(
        (
            bool(re.search(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{8,}", tok))
            or "HF_TOKEN" in tok
            or "HUGGING_FACE_HUB_TOKEN" in tok
        )
        for tok in argv
    )
    _append_jsonl(
        state_dir / "colab_calls.jsonl",
        {
            "argv": argv,
            "subcmd": subcmd,
            "token_leaked_in_cli": leaked,
            "ts": round(time.time(), 6),
        },
    )

    if subcmd == "usage":
        bal = float(cfg.get("balance_start", 45.0))
        if sess_state.get("active"):
            if sess_state.get("card_class") == "gpu":
                rate = float(cfg.get("gpu_rate", 2.10))
            else:
                rate = float(cfg.get("cpu_rate", 0.26))
            assign = 1
        else:
            rate = 0.0
            assign = 0
        print(f"Current balance: {bal:.2f} compute units")
        print(f"Usage rate: {rate:.2f}/hr")
        print(f"Active assignments: {assign}")
        return 0

    if subcmd == "new":
        http_st = cfg.get("new_http_status")
        if http_st == 412:
            print("Error: HTTP 412 Precondition Failed: account session cap reached", file=sys.stderr)
            return 1
        if http_st == 503:
            print("Error: HTTP 503 Service Unavailable: accelerator pool empty", file=sys.stderr)
            return 1

        session = "default"
        card = "CPU-high-mem"
        card_class = "cpu"
        high_mem = False
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok == "--gpu" and i + 1 < len(argv):
                card = argv[i + 1]
                card_class = "gpu"
                i += 2
            elif tok == "-s" and i + 1 < len(argv):
                session = argv[i + 1]
                i += 2
            elif tok == "--high-mem":
                high_mem = True
                i += 1
            else:
                i += 1

        if cfg.get("session_shape_override"):
            shape_label = str(cfg["session_shape_override"])
        elif card.upper() == "L4":
            # An L4 opens as Standard even with --high-mem (Publisher fleet ruling 2026-11-09).
            shape_label = "Standard"
        elif high_mem or card.upper() in ("V5E1", "V6E1"):
            shape_label = "High-RAM"
        else:
            shape_label = "Standard"

        vm_content.mkdir(parents=True, exist_ok=True)
        if cfg.get("mount_drive"):
            (vm_content / "drive" / "MyDrive").mkdir(parents=True, exist_ok=True)

        sess_state.update(
            {
                "active": True,
                "card": card,
                "card_class": card_class,
                "open_count": int(sess_state.get("open_count", 0)) + 1,
                "session": session,
                "shape_label": shape_label,
            }
        )
        _write_json(sess_file, sess_state)
        print(f"[colab] Opened session '{session}' ({card}, Shape: {shape_label})")
        return 0

    if subcmd == "sessions":
        if sess_state.get("active"):
            s_id = sess_state["session"]
            c_name = sess_state["card"]
            sh_lbl = sess_state["shape_label"]
            print(f"[{s_id}] connected | Accelerator: {c_name} | Shape: {sh_lbl}")
        else:
            print("No active sessions.")
        return 0

    if subcmd == "upload":
        # colab upload -s <session> <src> <dst>
        pos: list[str] = []
        i = 1
        while i < len(argv):
            if argv[i] == "-s" and i + 1 < len(argv):
                i += 2
            else:
                pos.append(argv[i])
                i += 1
        if len(pos) != 2:
            print(f"Error: bad upload args: {argv}", file=sys.stderr)
            return 2
        src_str, dst_str = pos
        src_p = _local_path(src_str)
        dst_p = _map_vm_path(dst_str, vm_content)

        if cfg.get("upload_part_sleep_s") and ".part.0000" in dst_p.name:
            (state_dir / "upload_part_in_flight.txt").write_text("1\n", encoding="utf-8")
            time.sleep(float(cfg["upload_part_sleep_s"]))

        if cfg.get("upload_fail_always") and "project_input" in dst_p.name:
            print("403 Client Error: Forbidden for url: https://colab.example/upload", file=sys.stderr)
            return 1

        if cfg.get("upload_fail_500_once"):
            marker = state_dir / "upload_failed_once.txt"
            if not marker.is_file() and ("project_input" in dst_p.name or ".part.0000" in dst_p.name):
                marker.write_text("1\n", encoding="utf-8")
                print("502 Server Error: Bad Gateway for url: https://colab.example/upload", file=sys.stderr)
                return 1

        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
        if dst_p.name == "project_input.tar" and tarfile.is_tarfile(dst_p):
            with tarfile.open(dst_p, "r:*") as tf_up:
                _write_json(
                    state_dir / "uploaded_tar_members.json",
                    {"members": sorted(m.name for m in tf_up.getmembers())},
                )
        if dst_p.name == "desk_hf_token" and os.name != "nt":
            dst_p.chmod(0o600)
        if cfg.get("corrupt_wheel_on_vm") and dst_p.name.endswith(".whl"):
            dst_p.write_bytes(dst_p.read_bytes() + b"corrupt")
        if cfg.get("corrupt_lock_on_vm") and dst_p.name == "desk_draft.lock":
            dst_p.write_bytes(dst_p.read_bytes() + b"\n# tampered\n")
        print(f"[colab] Uploaded '{src_str}' to '{dst_str}'")
        return 0

    if subcmd == "download":
        pos = []
        i = 1
        while i < len(argv):
            if argv[i] == "-s" and i + 1 < len(argv):
                i += 2
            else:
                pos.append(argv[i])
                i += 1
        if len(pos) != 2:
            print(f"Error: bad download args: {argv}", file=sys.stderr)
            return 2
        src_str, dst_str = pos
        src_p = _map_vm_path(src_str, vm_content)
        dst_p = _local_path(dst_str)
        if not src_p.is_file():
            print(f"404 Client Error: File or directory not found: {src_str}", file=sys.stderr)
            return 1
        if cfg.get("download_fail_500_once"):
            marker = state_dir / "download_failed_once.txt"
            if not marker.is_file():
                marker.write_text("1\n", encoding="utf-8")
                print("503 Server Error: Service Unavailable for url: https://colab.example/download", file=sys.stderr)
                return 1
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_p, dst_p)
        print(f"[colab] Downloaded '{src_str}' to '{dst_str}'")
        return 0

    if subcmd == "exec":
        script_file = ""
        env_pairs: list[tuple[str, str]] = []
        i = 1
        while i < len(argv):
            tok = argv[i]
            if tok == "-s" and i + 1 < len(argv):
                i += 2
            elif tok == "-f" and i + 1 < len(argv):
                script_file = argv[i + 1]
                i += 2
            elif tok == "--timeout" and i + 1 < len(argv):
                i += 2
            elif tok == "--env" and i + 1 < len(argv):
                kv = argv[i + 1]
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    env_pairs.append((k, v))
                i += 2
            else:
                i += 1
        if not script_file:
            print("Error: colab exec missing -f <file>", file=sys.stderr)
            return 2

        vm_content_posix = to_posix(vm_content)
        exec_env = dict(os.environ)
        exec_env["PATH"] = str(vm_bin) + os.pathsep + exec_env.get("PATH", "")
        exec_env["FAKE_COLAB_STATE_DIR"] = str(state_dir)
        exec_env["COLAB_RELEASE_TAG"] = str(cfg.get("colab_release_tag", "release-colab-2026-11-05"))
        exec_env["DESK_VM_USR_LOCAL"] = str(cfg["vm_usr_local"])
        exec_env["DESK_VM_CGROUP_DIR"] = str(vm_content.parent / "cgroup")
        exec_env["DESK_VM_PROC_MOUNTS"] = str(cfg["vm_proc_mounts"])
        exec_env["DESK_VM_PROC_MEMINFO"] = str(cfg["vm_proc_meminfo"])
        exec_env["DESK_VM_CONTENT_DRIVE"] = str(vm_content / "drive")
        exec_env["DESK_VM_VITALS_INTERVAL_S"] = "1"
        exec_env["FAKE_TORCH_VRAM_MIB"] = str(cfg.get("gpu_vram_mib", 23034))
        if cfg.get("disk_gib") is not None:
            exec_env["DESK_VM_OVERRIDE_DISK_GIB"] = str(cfg["disk_gib"])
        if cfg.get("ram_gib") is not None:
            exec_env["DESK_VM_OVERRIDE_RAM_GIB"] = str(cfg["ram_gib"])
        if cfg.get("token_shred_fail"):
            exec_env["DESK_VM_SIMULATE_SHRED_FAIL"] = "1"
        if cfg.get("cleanup_fail"):
            exec_env["DESK_VM_SIMULATE_CLEANUP_FAIL"] = "1"

        for k, v in env_pairs:
            v_norm = v.replace("\\", "/")
            if (
                v_norm == "/content"
                or v_norm.startswith("/content/")
                or re.search(r"(?i)/git/content(?:/|$)", v_norm)
            ):
                v = to_posix(_map_vm_path(v_norm, vm_content))
            exec_env[k] = v

        remote_sh = vm_content / "remote_desk_draft.sh"
        if remote_sh.is_file():
            sh_txt = remote_sh.read_text(encoding="utf-8")
            sh_txt = _rewrite_content_paths(sh_txt, vm_content_posix)
            crash_step = cfg.get("abrupt_crash_at_step")
            if crash_step:
                # Inject an abrupt non-RefusalError crash immediately after recording the named step.
                sh_txt = sh_txt.replace(
                    f"--step {crash_step}",
                    f"--step {crash_step}\nexit 17",
                )
            if cfg.get("inject_disguised_image_in_desk_out"):
                disguised_hook = (
                    f'"{to_posix(sys.executable)}" -c '
                    f'"import pathlib; p = pathlib.Path(\'{vm_content_posix}/desk_out/logs/disguised_plate.bin\'); '
                    f'p.parent.mkdir(parents=True, exist_ok=True); '
                    f'p.write_bytes(b\'\\\\xff\\\\xd8\\\\xff\\\\xe0\\\\x00\\\\x10JFIF\\\\x00\\\\x01\\\\x01\\\\x00\\\\x00\\\\x01\\\\x00\\\\x01\\\\x00\\\\x00\\\\xff\\\\xd9\')"\n'
                )
                sh_txt = sh_txt.replace(
                    "--step bundle_outputs",
                    f"--step bundle_outputs\n{disguised_hook.strip()}",
                )
            if cfg.get("delete_output_file") or cfg.get("corrupt_output_file"):
                hook = (
                    f'\n"{to_posix(sys.executable)}" "{to_posix(__file__)}" '
                    f'mutate-bundle "{vm_content_posix}/desk_draft_bundle.tar.gz"\n'
                )
                sh_txt += hook
            remote_sh.write_bytes(sh_txt.encode("utf-8"))

        target_py = _local_path(script_file)
        temp_copy: pathlib.Path | None = None
        if target_py.name == "_vm_cleanup_exec.py":
            rewritten = _rewrite_content_paths(target_py.read_text(encoding="utf-8"), vm_content_posix)
            fd, tmp_name = tempfile.mkstemp(prefix="fake_vm_cleanup_", suffix=".py")
            os.close(fd)
            temp_copy = pathlib.Path(tmp_name)
            temp_copy.write_text(rewritten, encoding="utf-8", newline="\n")
            target_py = temp_copy

        try:
            proc = subprocess.Popen(
                [sys.executable, str(target_py)],
                env=exec_env,
                stdin=subprocess.DEVNULL,
            )
            rc = proc.wait()
            up_tar = vm_content / "project_input.tar"
            if up_tar.is_file() and not (state_dir / "uploaded_tar_members.json").is_file() and tarfile.is_tarfile(up_tar):
                with tarfile.open(up_tar, "r:*") as tf_up:
                    _write_json(
                        state_dir / "uploaded_tar_members.json",
                        {"members": sorted(m.name for m in tf_up.getmembers())},
                    )
            if target_py.name == "desk_bundle_now.py":
                _mutate_bundle_for_fault(vm_content / "desk_draft_bundle.tar.gz", cfg)
            return int(rc)
        finally:
            if temp_copy is not None:
                temp_copy.unlink(missing_ok=True)

    if subcmd == "restart-kernel":
        print("[colab] Kernel restarted")
        return 0

    if subcmd == "stop":
        sess_state["active"] = False
        sess_state["stop_count"] = int(sess_state.get("stop_count", 0)) + 1
        _write_json(sess_file, sess_state)
        print("[colab] Stopped session")
        return 0

    print(f"Error: unknown fake colab subcommand {subcmd!r}", file=sys.stderr)
    return 2


def _cmd_vm_nvidia_smi(argv: list[str]) -> int:
    state_dir = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
    cfg = _read_json(state_dir / "config.json")
    if cfg.get("gpu_missing"):
        print("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", file=sys.stderr)
        return 1
    gpu_name = str(cfg.get("gpu_name", "NVIDIA L4"))
    gpu_vram = int(cfg.get("gpu_vram_mib", 23034))
    driver = str(cfg.get("gpu_driver", "550.54.15"))
    print(f"{gpu_name}, {gpu_vram}, {driver}")
    return 0


def _cmd_vm_pip(argv: list[str]) -> int:
    state_dir = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
    _append_jsonl(
        state_dir / "install_commands.jsonl",
        {"argv": argv, "tool": "pip", "ts": round(time.time(), 6)},
    )
    if "--require-hashes" not in argv or "--only-binary" not in argv or "--no-deps" not in argv:
        print(f"floating pip install forbidden: {argv}", file=sys.stderr)
        return 2
    if "-r" in argv:
        r_idx = argv.index("-r")
        req_p = pathlib.Path(argv[r_idx + 1])
        if not req_p.is_file() or "--hash=sha256:" not in req_p.read_text(encoding="utf-8"):
            print(f"missing hash-pinned bootstrap file: {req_p}", file=sys.stderr)
            return 2
    return 0


def _cmd_vm_uv(argv: list[str]) -> int:
    state_dir = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
    _append_jsonl(
        state_dir / "install_commands.jsonl",
        {"argv": argv, "tool": "uv", "ts": round(time.time(), 6)},
    )
    if len(argv) >= 3 and argv[0] == "python" and argv[1] == "install":
        ver = argv[2]
        inst_root = pathlib.Path(os.environ.get("UV_PYTHON_INSTALL_DIR", "/tmp/uv-py"))
        (inst_root / ver).mkdir(parents=True, exist_ok=True)
        return 0

    if len(argv) >= 2 and argv[0] == "venv":
        venv_dir = pathlib.Path(argv[-1])
        venv.EnvBuilder(with_pip=False).create(venv_dir)
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            scripts_dir = venv_dir / "Scripts"
            for item in scripts_dir.iterdir():
                shutil.copy2(item, bin_dir / item.name)
        return 0

    if len(argv) >= 3 and argv[0] == "pip" and argv[1] == "install":
        py_idx = argv.index("--python")
        py_target = pathlib.Path(argv[py_idx + 1])
        venv_dir = py_target.parent.parent
        sp = _venv_site_packages(venv_dir)

        if "-r" in argv:
            if "--require-hashes" not in argv or "--only-binary" not in argv or "--no-deps" not in argv:
                print(f"uv lock install missing hash/binary/no-deps flags: {argv}", file=sys.stderr)
                return 2
            lock_p = pathlib.Path(argv[argv.index("-r") + 1])
            lock_txt = lock_p.read_text(encoding="utf-8")
            if "--hash=sha256:" not in lock_txt:
                print("lockfile missing hashes", file=sys.stderr)
                return 2
            (sp / "huggingface_hub.py").write_text(STANDIN_HF_HUB_CODE, encoding="utf-8", newline="\n")
            (sp / "torch.py").write_text(STANDIN_TORCH_CODE, encoding="utf-8", newline="\n")
            (sp / "transformers.py").write_text(STANDIN_TRANSFORMERS_CODE, encoding="utf-8", newline="\n")
            outer = _outer_site_dirs()
            if outer:
                (sp / "zzz_outer_deps.pth").write_text("\n".join(outer) + "\n", encoding="utf-8", newline="\n")
            return 0

        wheel_arg = argv[-1]
        wheel_p = pathlib.Path(wheel_arg)
        if not wheel_p.is_file() or not wheel_p.name.endswith(".whl"):
            print(f"expected local wheel file, got {wheel_arg!r}", file=sys.stderr)
            return 2
        if "--no-deps" not in argv or "--only-binary" not in argv:
            print(f"wheel install must pass --no-deps --only-binary :all:, got {argv}", file=sys.stderr)
            return 2
        with zipfile.ZipFile(wheel_p, "r") as zf:
            zf.extractall(sp)
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        py_exe = bin_dir / ("python.exe" if os.name == "nt" else "python")
        desk_script = bin_dir / "dataset-desk"
        desk_code = (
            f"#!{py_exe}\n"
            "import json, os, pathlib, sys, time\n"
            "_sd = os.environ.get('FAKE_COLAB_STATE_DIR', '').strip()\n"
            "if _sd and len(sys.argv) > 1:\n"
            "    _cp = pathlib.Path(_sd) / 'config.json'\n"
            "    _cfg = json.loads(_cp.read_text(encoding='utf-8')) if _cp.is_file() else {}\n"
            "    if _cfg.get('hang_desk_subcmd') == sys.argv[1]:\n"
            "        time.sleep(float(_cfg.get('hang_desk_sleep_s', 5.0)))\n"
            "from dataset_desk.desk import run\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(run())\n"
        )
        desk_script.write_text(desk_code, encoding="utf-8", newline="\n")
        if os.name != "nt":
            desk_script.chmod(0o755)
        return 0

    print(f"unsupported fake uv command: {argv}", file=sys.stderr)
    return 2


class FakeColabEnv:
    """Per-test sandboxed rig + fake Colab CLI + fake VM directory tree."""

    def __init__(self, base_dir: pathlib.Path, config: dict | None = None) -> None:
        self.base_dir = base_dir.resolve()
        self.home_dir = self.base_dir / "empty_home"
        self.runtime_dir = self.base_dir / "runtime_dir"
        self.rig_bin = self.base_dir / "rig_bin"
        self.vm_root = self.base_dir / "fake_vm"
        self.vm_content = self.vm_root / "content"
        self.vm_bin = self.vm_root / "bin"
        self.vm_usr_local = self.vm_root / "usr_local"
        self.vm_proc_mounts = self.vm_root / "proc_mounts"
        self.vm_proc_meminfo = self.vm_root / "proc_meminfo"
        self.state_dir = self.base_dir / "fake_state"

        for d in (
            self.home_dir,
            self.runtime_dir,
            self.rig_bin,
            self.vm_content,
            self.vm_bin,
            self.vm_usr_local / "cuda-12.6",
            self.vm_usr_local / "cuda-12",
            self.state_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.runtime_dir.chmod(stat.S_IRWXU)

        cfg = dict(config or {})
        cfg.setdefault("vm_content", str(self.vm_content))
        cfg.setdefault("vm_bin", str(self.vm_bin))
        cfg.setdefault("vm_usr_local", str(self.vm_usr_local))
        cfg.setdefault("vm_proc_mounts", str(self.vm_proc_mounts))
        cfg.setdefault("vm_proc_meminfo", str(self.vm_proc_meminfo))
        self.config = cfg
        _write_json(self.state_dir / "config.json", self.config)

        mounts_lines = ["/dev/root / ext4 rw,relatime 0 0"]
        if cfg.get("mount_drive"):
            mounts_lines.append("drive /content/drive fuse.drive rw,nosuid,nodev 0 0")
        if cfg.get("hf_mount_fstype"):
            hf_posix = to_posix(self.vm_content / "hf_home")
            mounts_lines.append(f"remote:{hf_posix} {hf_posix} {cfg['hf_mount_fstype']} rw 0 0")
        self.vm_proc_mounts.write_text("\n".join(mounts_lines) + "\n", encoding="utf-8", newline="\n")

        ram_gib = float(cfg.get("ram_gib", 50.0))
        ram_kib = int(ram_gib * 1024 * 1024)
        total_kib = max(ram_kib + 2 * 1024 * 1024, 52 * 1024 * 1024)
        mem_text = (
            f"MemTotal:       {total_kib} kB\n"
            f"MemFree:        {ram_kib} kB\n"
            f"MemAvailable:   {ram_kib} kB\n"
            "AnonPages:       1048576 kB\n"
            "Shmem:             65536 kB\n"
        )
        self.vm_proc_meminfo.write_text(mem_text, encoding="utf-8", newline="\n")

        py_posix = to_posix(sys.executable)
        mod_posix = to_posix(__file__)
        _write_bash_shim(
            self.rig_bin / "colab",
            f'exec "{py_posix}" "{mod_posix}" colab-cli "$@"',
            include_fake_marker=True,
        )
        _write_bash_shim(self.vm_bin / "python3", f'exec "{py_posix}" "$@"')
        _write_bash_shim(self.vm_bin / "nvidia-smi", f'exec "{py_posix}" "{mod_posix}" vm-nvidia-smi "$@"')
        _write_bash_shim(self.vm_bin / "pip", f'exec "{py_posix}" "{mod_posix}" vm-pip "$@"')
        _write_bash_shim(self.vm_bin / "uv", f'exec "{py_posix}" "{mod_posix}" vm-uv "$@"')

    def _sanitized_path(self) -> str:
        bash_dir = str(pathlib.Path(resolve_git_bash()).parent)
        usr_bin = str(pathlib.Path(resolve_git_bash()).parent.parent / "usr" / "bin")
        parts: list[str] = [str(self.rig_bin)]
        if os.name == "nt":
            for extra in (bash_dir, usr_bin):
                if pathlib.Path(extra).is_dir() and extra not in parts:
                    parts.append(extra)
        for entry in os.environ.get("PATH", "").split(os.pathsep):
            if not entry:
                continue
            d = pathlib.Path(entry)
            # Never keep a PATH directory containing a real `colab` binary.
            if (d / "colab").is_file() or (d / "colab.exe").is_file() or (d / "colab.cmd").is_file():
                continue
            if entry not in parts:
                parts.append(entry)
        return os.pathsep.join(parts)

    def build_env(
        self,
        *,
        extra_env: dict[str, str | None] | None = None,
        test_mode: bool = True,
        allow_any_out_drive: bool = True,
    ) -> dict[str, str]:
        env: dict[str, str] = {}
        for keep in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR", "MSYSTEM", "TERM"):
            if keep in os.environ:
                env[keep] = os.environ[keep]
        env["HOME"] = to_posix(self.home_dir)
        env["USERPROFILE"] = str(self.home_dir)
        env["XDG_CONFIG_HOME"] = to_posix(self.home_dir / ".config")
        env["XDG_CACHE_HOME"] = to_posix(self.home_dir / ".cache")
        env["XDG_DATA_HOME"] = to_posix(self.home_dir / ".local" / "share")
        env["XDG_RUNTIME_DIR"] = to_posix(self.runtime_dir)
        env["PATH"] = self._sanitized_path()
        env["COLAB_RIG_PYTHON"] = to_posix(sys.executable)
        env["COLAB_OPEN_LOCK"] = to_posix(self.base_dir / "open.lock")
        env["FAKE_COLAB_STATE_DIR"] = to_posix(self.state_dir)

        if test_mode:
            env["SDLAB_TEST_FAKE_COLAB"] = "1"
            env["SDLAB_TEST_RATE_READ_INTERVAL_S"] = "0.02"
            env["SDLAB_TEST_KILL_GRACE_S"] = "1"
            env["SDLAB_TEST_STOP_TIMEOUT_S"] = "15"
            env["SDLAB_TEST_SESSIONS_TIMEOUT_S"] = "15"
            env["SDLAB_TEST_UPLOAD_TIMEOUT_S"] = "30"
            env["SDLAB_TEST_DOWNLOAD_TIMEOUT_S"] = "30"
            env["SDLAB_TEST_SALVAGE_RESTART_TIMEOUT_S"] = "15"
            env["SDLAB_TEST_SALVAGE_EXEC_TIMEOUT_S"] = "30"
            env["DESK_TEST_BACKOFF_BASE_S"] = "0.02"
            env["DESK_TEST_HEARTBEAT_POLL_S"] = "0.03"
            if allow_any_out_drive:
                env["DESK_TEST_ALLOW_ANY_OUT_DRIVE"] = "1"

        if extra_env:
            for k, v in extra_env.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = v
        return env

    def run_launch(
        self,
        args: list[str],
        *,
        extra_env: dict[str, str | None] | None = None,
        test_mode: bool = True,
        allow_any_out_drive: bool = True,
        timeout: float = 180.0,
    ) -> subprocess.CompletedProcess[str]:
        env = self.build_env(
            extra_env=extra_env,
            test_mode=test_mode,
            allow_any_out_drive=allow_any_out_drive,
        )
        bash_bin = resolve_git_bash()
        return subprocess.run(
            [bash_bin, to_posix(LAUNCH_SH), *args],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )

    def colab_calls(self) -> list[dict]:
        p = self.state_dir / "colab_calls.jsonl"
        if not p.is_file():
            return []
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def install_commands(self) -> list[dict]:
        p = self.state_dir / "install_commands.jsonl"
        if not p.is_file():
            return []
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def hf_calls(self) -> list[dict]:
        p = self.state_dir / "hf_calls.jsonl"
        if not p.is_file():
            return []
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def session_state(self) -> dict:
        return _read_json(self.state_dir / "session_state.json")

    def uploaded_tar_members(self) -> list[str]:
        doc = _read_json(self.state_dir / "uploaded_tar_members.json")
        return list(doc.get("members") or [])


def make_synthetic_project(
    root: pathlib.Path,
    *,
    pool_tag: str | None = "rustline",
    flagged_count: int = 2,
    unflagged_count: int = 2,
    extra_pad_bytes: int = 0,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Create a synthetic desk project with gate-flagged and gate-passing records."""
    from PIL import Image, PngImagePlugin

    proj = root / "project"
    wave = proj / "outputs" / "wave-1"
    recs = proj / "records"
    wave.mkdir(parents=True, exist_ok=True)
    recs.mkdir(parents=True, exist_ok=True)

    colors = [
        (205, 48, 42),
        (42, 168, 88),
        (48, 92, 204),
        (188, 152, 44),
        (142, 64, 178),
    ]
    idx = 0
    for f_i in range(flagged_count):
        c = colors[idx % len(colors)]
        idx += 1
        img_rel = f"outputs/wave-1/flagged_{f_i:02d}.png"
        png_info = None
        if f_i == 0 and extra_pad_bytes > 0:
            png_info = PngImagePlugin.PngInfo()
            seed_hex = hashlib.sha256(b"synthetic-pad-seed").hexdigest()
            repeats = (extra_pad_bytes + len(seed_hex) - 1) // len(seed_hex)
            png_info.add_text("synthetic_pad", (seed_hex * repeats)[:extra_pad_bytes], zip=False)
        Image.new("RGB", (56, 56), c).save(proj / img_rel, pnginfo=png_info)
        rec = {
            "id": f"rec_flagged_{f_i:02d}",
            "asset_path": img_rel,
            "canon": {"subject": f"stone archway {f_i}"},
            # Short prompt-echo fails the model-free gate -> flagged for drafting!
            "judgment": {"status": "approved", "explanation": f"stone archway {f_i}"},
            "provenance": {"prompt": f"stone archway {f_i}"},
        }
        _write_json(recs / f"rec_flagged_{f_i:02d}.json", rec)

    clean_scenes = [
        (
            "lighthouse cliff",
            "coastal beacon",
            "in rustline style, a whitewashed masonry lighthouse rises from spray-wet basalt cliffs "
            "beside coiled hemp hawser under overcast coastal twilight",
        ),
        (
            "windmill field",
            "barley mill",
            "in rustline style, an octagonal timber windmill with canvas sails stands amid golden "
            "barley furrows beside a dry-stone wall under clear afternoon sun",
        ),
        (
            "forge anvil",
            "smithy interior",
            "in rustline style, a blackened iron anvil rests on a scarred oak stump beside glowing "
            "hearth tongs and stacked charcoal baskets in a smoky stone smithy",
        ),
        (
            "bridge canal",
            "archway canal",
            "in rustline style, a weathered brick footbridge arches over still green canal water "
            "between shuttered limestone warehouses under pale morning mist",
        ),
    ]
    for u_i in range(unflagged_count):
        c = colors[idx % len(colors)]
        idx += 1
        subj, prm, expl = clean_scenes[u_i % len(clean_scenes)]
        if u_i >= len(clean_scenes):
            expl = f"{expl} (variant {u_i})"
        img_rel = f"outputs/wave-1/clean_{u_i:02d}.png"
        Image.new("RGB", (56, 56), c).save(proj / img_rel)
        rec = {
            "id": f"rec_clean_{u_i:02d}",
            "asset_path": img_rel,
            "canon": {"subject": subj},
            "judgment": {
                "status": "approved",
                "explanation": expl,
            },
            "provenance": {"prompt": prm},
        }
        _write_json(recs / f"rec_clean_{u_i:02d}.json", rec)

    if extra_pad_bytes > 0:
        # Deterministic incompressible payload inside project so chunked tar upload is exercised cleanly.
        pad = hashlib.sha256(b"synthetic-pad-seed").digest()
        repeats = (extra_pad_bytes + len(pad) - 1) // len(pad)
        (wave / "plate_pad.bin").write_bytes((pad * repeats)[:extra_pad_bytes])

    cap_cfg = root / "caption-rules.json"
    cfg_doc: dict[str, object] = {
        "prefix": "in rustline style,",
        "drafter": "qwen3-vl",
        "drafter_prompt": "Describe only visible subjects, materials, setting, and lighting in plain prose.",
    }
    if pool_tag is not None:
        cfg_doc["multimodal_pool"] = pool_tag
    _write_json(cap_cfg, cfg_doc)
    return proj, cap_cfg


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return 2
    mode, rest = args[0], args[1:]
    if mode == "colab-cli":
        return _cmd_colab_cli(rest)
    if mode == "vm-nvidia-smi":
        return _cmd_vm_nvidia_smi(rest)
    if mode == "vm-pip":
        return _cmd_vm_pip(rest)
    if mode == "vm-uv":
        return _cmd_vm_uv(rest)
    if mode == "mutate-bundle":
        state_dir = pathlib.Path(os.environ["FAKE_COLAB_STATE_DIR"])
        cfg = _read_json(state_dir / "config.json")
        _mutate_bundle_for_fault(pathlib.Path(rest[0]), cfg)
        return 0
    print(f"unknown fake_colab_desk mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
