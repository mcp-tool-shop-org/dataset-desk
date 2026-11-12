#!/usr/bin/env python3
"""Rig and VM helper subcommands for scripts/colab/launch_desk_draft.sh."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile

SDLAB_VENDORED_COMMIT = "df9a1b3be2f8729a57776253e9b0cb32b7f5299c"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-8B-Instruct"
QWEN3_VL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

GPU_CARDS_ALLOWED = {"L4", "A100", "A100-40GB", "A100-80GB", "H100"}
GPU_MIN_VRAM_MIB = 22000

DEFAULT_MODEL_LOAD_STALL_S = 420.0
DEFAULT_HEARTBEAT_STALL_S = 180.0
DEFAULT_DESK_CMD_TIMEOUT_S = 300.0
DEFAULT_DESK_VM_CLEANUP_BOUND_S = 70

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}

FORBIDDEN_MOUNT_FSTYPES = {
    "fuse",
    "fuse.drive",
    "fuse.sshfs",
    "fuse.gcsfuse",
    "fuse.rclone",
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "9p",
    "drvfs",
    "ceph",
    "glusterfs",
    "lustre",
}

PRIVATE_URI_RE = re.compile(
    r"(?i)(?:^(?:https?|s3|gs|hf|gdrive|ftp|rclone)://|^rclone:|/content/drive\b|\bdrive/mydrive\b|huggingface\.co/)",
)

REQUIRED_DESK_ARTIFACTS = (
    "desk_artifacts/drafter-raw.jsonl",
    "desk_artifacts/raw_outputs.json",
    "desk_artifacts/description-review.jsonl",
    "desk_artifacts/description-worksheet.md",
    "desk_artifacts/captions-manifest.json",
    "desk_artifacts/pool-detection.json",
    "desk_artifacts/report-none.md",
)

REFUSED_LINE_RE = re.compile(r"REFUSED:\s*([A-Z0-9_]+):\s*(.+)")
STEP_MARKER_RE = re.compile(r"REMOTE_STEP_FAILED_MARKER\s+step=([^\s]+)\s+code=([A-Z0-9_]+)\s+why=(.+)")
HTTP_412_RE = re.compile(r"(?i)(?:\bHTTP[/\s]*\d*(?:\.\d+)?\s*412\b|\bstatus(?:\s+code)?\s*[:=]?\s*412\b|\b412\s+Precondition\s+Failed\b|\baccount[_\s-]*cap\b.*\b412\b)")
HTTP_503_RE = re.compile(r"(?i)(?:\bHTTP[/\s]*\d*(?:\.\d+)?\s*503\b|\bstatus(?:\s+code)?\s*[:=]?\s*503\b|\b503\s+Service\s+Unavailable\b|\bpool[_\s-]*empty\b.*\b503\b)")


class RefusalError(Exception):
    def __init__(self, code: str, why: str, *, step: str | None = None, retryable: bool = False) -> None:
        super().__init__(f"REFUSED: {code}: {why}")
        self.code = code
        self.why = why
        self.step = step
        self.retryable = retryable


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_lf(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def detect_image_magic_bytes(data: bytes) -> str | None:
    """Return MIME type if `data` starts with raster image magic bytes (PNG/JPEG/GIF/WebP/TIFF/BMP)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if len(data) >= 14 and data[:2] == b"BM" and data[6:10] == b"\x00\x00\x00\x00":
        return "image/bmp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        # ISO-BMFF image brands: the major brand, then the compatible brands the box lists.
        box_len = int.from_bytes(data[:4], "big")
        end = box_len if 16 <= box_len <= len(data) else len(data)
        brands = {data[8:12]} | {data[i : i + 4] for i in range(16, max(16, end - 3), 4)}
        if brands & {b"avif", b"avis"}:
            return "image/avif"
        if brands & {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1"}:
            return "image/heic"
    return None


def is_image_file_by_magic(path: pathlib.Path) -> bool:
    """True when the file's own first bytes are a raster image (the extension is never trusted)."""
    try:
        with pathlib.Path(path).open("rb") as fh:
            return detect_image_magic_bytes(fh.read(64)) is not None
    except OSError:
        return False


_REJECTED_83_RE = re.compile(r"^reject~\d+$")


def _is_rejected_component(part: str) -> bool:
    # Windows ignores trailing dots and spaces ("rejected." is the directory "rejected") and may show a long
    # name as an 8.3 short name ("REJECT~1"); both count as rejected.
    norm = part.rstrip(". ").lower()
    return norm == "rejected" or bool(_REJECTED_83_RE.match(norm))


def has_rejected_component(rel_posix: str) -> bool:
    """True when any component of a project-relative path is `rejected` (case-insensitive, Windows aliases too)."""
    parts = [p for p in str(rel_posix).replace("\\", "/").split("/") if p]
    return any(_is_rejected_component(p) for p in parts)


def path_is_rejected(proj_dir: pathlib.Path, rel: object) -> bool:
    """True if `rel` names a path under `rejected/`, either as written or AFTER symlinks and junctions resolve."""
    rel_s = str(rel or "").replace("\\", "/").strip("/")
    if not rel_s:
        return False
    if has_rejected_component(rel_s):
        return True
    try:
        real = (pathlib.Path(proj_dir) / rel_s).resolve()
        real_rel = os.path.relpath(str(real), str(pathlib.Path(proj_dir).resolve())).replace("\\", "/")
    except (OSError, ValueError) as exc:
        # Fail closed: a path that cannot be resolved is treated as rejected and never uploads.
        print(f"REJECTED_PATH_CHECK_FAILED {rel_s!r}: {exc.__class__.__name__}", file=sys.stderr)
        return True
    if real_rel == ".." or real_rel.startswith("../"):
        return False
    return has_rejected_component(real_rel)


# Kept-file path scrubbing (Publisher, 2026-11-12): nothing under the out dir may name a rig home, a Windows or
# WSL user directory, a drive root, or the studio config path. Explicit roots (out dir, repo, project, home, the
# user-config file) become named placeholders first; the generic patterns then catch whatever is left.
RIG_PATH_PLACEHOLDER = "<rig>"
_SEP = r"(?:\\\\|\\|/)+"
_USER = r"[^\\/\"'\r\n:;,<>|]+"
_STUDIO_CONFIG_PATTERN = re.compile(r"[^\s\"'<>]*dataset-desk[\\/]+config\.json", re.I)
_RIG_PATH_PATTERNS = (
    # \\wsl.localhost\<distro>\... and \\wsl$\<distro>\...
    re.compile(r"(?:\\{2,4}|//)wsl(?:\.localhost|\$)(?:\\{1,4}|/)+[^\\/\s\"']+(?:(?:\\{1,4}|/)+home(?:\\{1,4}|/)+[^\\/\s\"']+)?", re.I),
    # URL-encoded forms: C%3A%5CUsers%5Cname, %2Fhome%2Fname, %2Fmnt%2Fc%2FUsers%2Fname
    re.compile(r"[A-Za-z]%3A(?:%5C|%2F)+Users(?:%5C|%2F)+[^%\s\"'&/\\]+", re.I),
    re.compile(r"%2Fmnt%2F[A-Za-z]%2FUsers%2F[^%\s\"'&/\\]+", re.I),
    re.compile(r"%2F(?:home|Users)%2F[^%\s\"'&/\\]+", re.I),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:" + _SEP + r"Users" + _SEP + _USER, re.I),
    re.compile(r"/mnt/[A-Za-z]/Users/" + _USER, re.I),
    re.compile(r"/home/[^/\\\s\"']+", re.I),
    re.compile(r"/Users/" + _USER, re.I),
    re.compile(r"(?<![A-Za-z0-9])/mnt/[A-Za-z](?=/)", re.I),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?=\\|/(?!/))"),
)
_SCRUB_MAX_BYTES = 64 * 1024 * 1024


_GUARDS_MODULE: object = None


def _token_scrub(text: str) -> str:
    """Apply launch_guards.scrub_colab_tokens (sits next to this file on the rig and on the VM)."""
    global _GUARDS_MODULE
    if _GUARDS_MODULE is None:
        try:
            spec = importlib.util.spec_from_file_location("desk_launch_guards", pathlib.Path(__file__).with_name("launch_guards.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _GUARDS_MODULE = mod
        except Exception:
            _GUARDS_MODULE = False
    return _GUARDS_MODULE.scrub_colab_tokens(text) if _GUARDS_MODULE else text


def scrub_rig_paths(text: str, roots: list[tuple[str, str]] | None = None) -> str:
    """Replace Colab/HF tokens, then explicit roots (longest first), then every generic rig-path pattern, in `text`."""
    out = _token_scrub(text)
    variants: list[tuple[str, str]] = []
    for root, placeholder in roots or []:
        r = str(root or "").strip()
        if len(r) < 4:
            continue
        for form in {r, r.rstrip("/\\"), r.replace("/", "\\"), r.replace("\\", "/")}:
            if len(form) >= 4:
                variants.append((form, placeholder))
                variants.append((form.replace("\\", "\\\\"), placeholder))
    for form, placeholder in sorted(set(variants), key=lambda kv: -len(kv[0])):
        out = out.replace(form, placeholder)
    out = _STUDIO_CONFIG_PATTERN.sub("<studio-config>", out)
    for pat in _RIG_PATH_PATTERNS:
        out = pat.sub(RIG_PATH_PLACEHOLDER, out)
    return out


def _scrub_json_values(node, roots):
    if isinstance(node, str):
        return scrub_rig_paths(node, roots)
    if isinstance(node, list):
        return [_scrub_json_values(v, roots) for v in node]
    if isinstance(node, dict):
        return {scrub_rig_paths(str(k), roots): _scrub_json_values(v, roots) for k, v in node.items()}
    return node


def _json_text_ok(text: str, jsonl: bool) -> bool:
    try:
        if jsonl:
            for ln in text.splitlines():
                if ln.strip():
                    json.loads(ln)
        else:
            json.loads(text)
        return True
    except ValueError:
        return False


def scrub_text_for_file(name: str, text: str, roots: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """Scrub `text` of the file `name`. Returns (cleaned, status).

    status: ok; json_fallback (the regex scrub broke parsable JSON, so it was redone on the decoded values);
    json_error_scrubbed (the original is not valid JSON; the regex-scrubbed text is returned); json_error (the
    original parsed and neither scrub could keep it valid; the ORIGINAL text is returned)."""
    cleaned = scrub_rig_paths(text, roots)
    suffix = pathlib.PurePosixPath(name).suffix.lower()
    if suffix not in (".json", ".jsonl") or cleaned == text:
        return cleaned, "ok"
    jsonl = suffix == ".jsonl"
    if _json_text_ok(cleaned, jsonl):
        return cleaned, "ok"
    if not _json_text_ok(text, jsonl):
        # The original does not parse (a JSONL cut off by a killed writer): there is no JSON to protect, so the
        # regex-scrubbed text is written and the file is still reported.
        if jsonl:
            # Per line: a line that parses is scrubbed on its decoded values (a `\\/` writer stays valid); a line
            # that does not (a cut-off tail) is regex-scrubbed.
            out_lines = []
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    out_lines.append(json.dumps(_scrub_json_values(json.loads(ln), roots), ensure_ascii=False))
                except ValueError:
                    out_lines.append(scrub_rig_paths(ln, roots))
            return "\n".join(out_lines) + "\n", "json_error_scrubbed"
        return cleaned, "json_error_scrubbed"
    try:
        if jsonl:
            lines = [json.dumps(_scrub_json_values(json.loads(ln), roots), ensure_ascii=False) for ln in text.splitlines() if ln.strip()]
            redone = "\n".join(lines) + "\n"
        else:
            redone = json.dumps(_scrub_json_values(json.loads(text), roots), ensure_ascii=False, indent=2) + "\n"
    except ValueError:
        return text, "json_error"
    return redone, "json_fallback"


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write via an exclusively created temp file in the same directory, copy the original's mode, and rename, so a
    kill cannot leave a truncated file."""
    tmp = path.with_name(f".{path.name}.scrub_tmp_{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        try:
            fd = os.open(tmp, flags, 0o600)
        except FileExistsError:
            tmp.unlink()
            fd = os.open(tmp, flags, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _scrub_dir(out_dir: pathlib.Path, roots: list[tuple[str, str]]) -> dict:
    scrubbed = 0
    skipped: list[str] = []
    errors: list[str] = []
    for fpath in sorted(out_dir.rglob("*")):
        try:
            rel = fpath.relative_to(out_dir).as_posix()
            if fpath.is_symlink() or not fpath.is_file() or rel == "scrub_report.json" or ".scrub_tmp_" in fpath.name:
                continue
            if fpath.stat().st_size > _SCRUB_MAX_BYTES:
                skipped.append(f"{rel} (over {_SCRUB_MAX_BYTES} bytes)")
                continue
            raw = fpath.read_bytes()
            if b"\x00" in raw:
                skipped.append(f"{rel} (binary)")
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Decoding with replacement would write U+FFFD into the file for good.
                skipped.append(f"{rel} (not utf-8)")
                continue
            cleaned, status = scrub_text_for_file(rel, text, roots)
            if status == "json_error":
                errors.append(f"{rel} (scrub would break the JSON; original kept)")
                continue
            if status == "json_error_scrubbed":
                errors.append(f"{rel} (not valid JSON; regex-scrubbed)")
            if cleaned != text:
                try:
                    _atomic_write_text(fpath, cleaned)
                except OSError as exc:
                    skipped.append(f"{rel} (replace_failed: {exc.__class__.__name__})")
                    continue
                scrubbed += 1
        except OSError as exc:
            skipped.append(f"{fpath.name} (unreadable: {exc.__class__.__name__})")
    return {"scrubbed_files": scrubbed, "skipped": skipped, "scrub_errors": errors}


SCRUB_PENDING_MARKER = ".scrub_pending"


def _scrub_complete(report: dict) -> bool:
    """A scrub is complete only with no scrub_errors and nothing that could not be rewritten or read as text."""
    if report["scrub_errors"]:
        return False
    return not any(("replace_failed" in x) or ("not utf-8" in x) or ("over " in x) or ("unreadable" in x) for x in report["skipped"])


def cmd_check_out_dir_scrubbed(args: argparse.Namespace) -> int:
    """Refuse (OUT_DIR_SCRUB_PENDING) a run folder that still carries .scrub_pending: any promote, export or publish
    step that reads a desk-lane out dir must call this first."""
    out_dir = pathlib.Path(args.out_dir)
    if (out_dir / SCRUB_PENDING_MARKER).exists():
        raise RefusalError(
            "OUT_DIR_SCRUB_PENDING",
            f"{out_dir.name!r} still carries {SCRUB_PENDING_MARKER}: its kept files may hold unscrubbed rig paths; "
            "run scrub-kept-paths or the next launch's sweep, and review scrub_report.json",
        )
    print(json.dumps({"scrubbed": True}, sort_keys=True))
    return 0


def cmd_scrub_kept_paths(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_dir():
        return 0
    roots: list[tuple[str, str]] = []
    for spec in args.root or []:
        if "=" not in spec:
            continue
        raw_root, placeholder = spec.rsplit("=", 1)
        roots.append((raw_root, placeholder))
        try:
            resolved = str(pathlib.Path(raw_root).resolve())
        except OSError:
            resolved = ""
        if resolved and resolved != raw_root:
            roots.append((resolved, placeholder))
    cfg_env = os.environ.get("DATASET_DESK_USER_CONFIG", "").strip()
    if cfg_env:
        roots.append((cfg_env, "<studio-config>"))
    report = _scrub_dir(out_dir, roots)
    try:
        write_json_lf(
            out_dir / "scrub_report.json",
            {"scrub_errors": report["scrub_errors"], "scrubbed_files": report["scrubbed_files"], "skipped": report["skipped"]},
        )
    except OSError:
        pass
    if _scrub_complete(report):
        try:
            (out_dir / SCRUB_PENDING_MARKER).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    print(json.dumps(report, sort_keys=True))
    return 0


def cmd_sweep_unscrubbed(args: argparse.Namespace) -> int:
    """Scrub earlier desk-lane run folders next to this one: any folder with .scrub_pending (a run killed before its
    scrub, or an escalated run whose watcher died) or remote_desk_draft.sh. Paths and tokens are both scrubbed."""
    parent = pathlib.Path(args.parent)
    exclude = pathlib.Path(args.exclude).resolve() if args.exclude else None
    if not parent.is_dir():
        return 0
    swept = 0
    children = sorted(parent.iterdir())
    if len(children) > 200 and exclude is not None and exclude.is_dir():
        with (exclude / "guards.log").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"SWEEP_LIMIT_REACHED parent holds {len(children)} entries; only the first 200 were checked\n")
    for child in children[:200]:
        try:
            if not child.is_dir() or child.is_symlink() or (exclude is not None and child.resolve() == exclude):
                continue
            if not ((child / SCRUB_PENDING_MARKER).exists() or (child / "remote_desk_draft.sh").is_file()):
                continue
            report = _scrub_dir(child, [])
            if (child / SCRUB_PENDING_MARKER).exists() and _scrub_complete(report):
                (child / SCRUB_PENDING_MARKER).unlink()
            swept += 1
        except OSError:
            continue
    print(json.dumps({"swept_run_dirs": swept}, sort_keys=True))
    return 0


def sanitize_pool_detection(pd: object) -> object:
    """Store pool names and integer counts only; strip all `sources`, `path`, and `config_path` entries."""
    if not isinstance(pd, dict):
        return pd
    status = str(pd.get("status") or "off")
    if status != "on":
        out_off: dict[str, object] = {"status": status}
        if isinstance(pd.get("warning"), str):
            out_off["warning"] = pd["warning"]
        return out_off

    counts_raw = pd.get("counts")
    pools_raw = pd.get("pools")
    clean_counts: dict[str, int] = {}
    if isinstance(counts_raw, dict):
        for k in sorted(counts_raw):
            if isinstance(counts_raw[k], (int, float)):
                clean_counts[str(k)] = int(counts_raw[k])
    elif isinstance(pools_raw, dict):
        for k in sorted(pools_raw):
            entry = pools_raw[k]
            if isinstance(entry, dict) and isinstance(entry.get("count"), (int, float)):
                clean_counts[str(k)] = int(entry["count"])

    clean_pools = {k: {"count": v} for k, v in sorted(clean_counts.items())}
    cfg_pools = pd.get("configured_pools")
    clean_cfg = (
        sorted(str(x) for x in cfg_pools if isinstance(x, str))
        if isinstance(cfg_pools, list)
        else [k for k, v in sorted(clean_counts.items()) if v > 0]
    )
    out_on: dict[str, object] = {
        "configured_pools": clean_cfg,
        "counts": clean_counts,
        "pools": clean_pools,
        "status": "on",
    }
    if isinstance(pd.get("stem_check"), str):
        out_on["stem_check"] = pd["stem_check"]
    if isinstance(pd.get("command"), str):
        out_on["command"] = pd["command"]
    return out_on


def _sanitize_artifact_json_obj(obj: object, *, file_name: str) -> object:
    if isinstance(obj, dict):
        out: dict[str, object] = {}
        for k, v in obj.items():
            if k == "pool_detection" and isinstance(v, dict):
                out[k] = sanitize_pool_detection(v)
            elif k == "last_path" and ("heartbeat" in file_name or file_name == "vm_window.json"):
                continue
            elif k == "heartbeat_last" and isinstance(v, dict):
                out[k] = {hk: hv for hk, hv in v.items() if hk != "last_path"}
            elif k == "project" and file_name == "meta.json" and isinstance(v, str):
                norm = v.replace("\\", "/").rstrip("/")
                out[k] = norm.rsplit("/", 1)[-1] if "/" in norm else norm
            else:
                out[k] = _sanitize_artifact_json_obj(v, file_name=file_name)
        return out
    if isinstance(obj, list):
        return [_sanitize_artifact_json_obj(item, file_name=file_name) for item in obj]
    return obj


def _sanitize_json_tree(root: pathlib.Path) -> None:
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".json":
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            cleaned = _sanitize_artifact_json_obj(doc, file_name=p.name)
            if cleaned != doc:
                write_json_lf(p, cleaned)
        elif suf == ".jsonl":
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            changed = False
            out_lines: list[str] = []
            for ln in lines:
                if not ln.strip():
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    out_lines.append(ln)
                    continue
                cleaned_row = _sanitize_artifact_json_obj(row, file_name=p.name)
                if cleaned_row != row:
                    changed = True
                    out_lines.append(json.dumps(cleaned_row, sort_keys=True))
                else:
                    out_lines.append(ln)
            if changed:
                p.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8", newline="\n")


def record_vm_step(state_dir: pathlib.Path, step: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "remote_step.txt").write_text(f"{step}\n", encoding="utf-8", newline="\n")
    entry = {"step": step, "ts": round(time.time(), 6)}
    with (state_dir / "remote_steps.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"REMOTE_STEP step={step}", flush=True)


def record_vm_failure(
    state_dir: pathlib.Path,
    *,
    step: str,
    code: str,
    why: str,
    exit_code: int = 2,
    retryable: bool = False,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    clean_why = " ".join(str(why).splitlines()).strip()
    doc = {
        "step": step,
        "refusal_code": code,
        "why": clean_why,
        "exit_code": exit_code,
        "retryable": retryable,
        "ts": round(time.time(), 6),
    }
    write_json_lf(state_dir / "remote_failure.json", doc)
    msg = f"REFUSED: {code}: {clean_why}"
    print(msg, file=sys.stderr, flush=True)
    print(f"REMOTE_STEP_FAILED_MARKER step={step} code={code} why={clean_why}", file=sys.stderr, flush=True)


def _check_no_private_uri(raw: str, label: str) -> None:
    val = str(raw or "").strip()
    if PRIVATE_URI_RE.search(val):
        raise RefusalError(
            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
            f"{label} references forbidden external/Drive/bucket/HF route: {val!r}; "
            "private images may reach the VM only via colab_upload_checked",
        )


def _verify_inside_project(proj_dir: pathlib.Path, candidate: pathlib.Path, *, label: str) -> pathlib.Path:
    real = candidate.resolve()
    try:
        in_proj = os.path.commonpath([str(proj_dir), str(real)]) == str(proj_dir)
    except ValueError:
        in_proj = False
    if not in_proj:
        raise RefusalError(
            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
            f"project entry {label!r} symlink/junction resolves outside project root",
        )
    return real


def _walk_followlinks_guard_cycles(
    top: pathlib.Path,
    proj_dir: pathlib.Path,
    *,
    prune_fn=None,
    _ancestors: frozenset[str] = frozenset(),
):
    """Walk `top` following symlinks and Windows directory junctions, refusing any cycle with PROJECT_LINK_CYCLE."""
    real_path = _verify_inside_project(proj_dir, top, label=top.name or ".")
    real_key = os.path.normcase(str(real_path))
    if real_key in _ancestors:
        rel_lbl = top.relative_to(proj_dir).as_posix() if top != proj_dir else "."
        raise RefusalError(
            "PROJECT_LINK_CYCLE",
            f"followed symlink/junction cycle detected at {rel_lbl!r}",
        )
    next_ancestors = _ancestors | {real_key}
    try:
        entries = list(os.scandir(top))
    except OSError:
        return
    dirs: list[str] = []
    files: list[str] = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=True):
                dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=True):
                files.append(entry.name)
        except OSError:
            continue
    dirs.sort()
    files.sort()
    if prune_fn is not None:
        prune_fn(str(top), dirs, str(proj_dir))
    yield top, dirs, files
    for d in dirs:
        yield from _walk_followlinks_guard_cycles(
            top / d,
            proj_dir,
            prune_fn=prune_fn,
            _ancestors=next_ancestors,
        )


def _collect_compose_upload_files(
    proj_dir: pathlib.Path,
    desk_mod,
    gate_mod,
) -> tuple[list[tuple[str, pathlib.Path]], int]:
    """Build the minimal upload allowlist of files that `index`, `report`, and `compose` actually read."""
    # Guard both the pruned project walk (reusing desk.prune_project_walk_dirs) and top-level records/ against link cycles.
    for _root, _dirs, _files in _walk_followlinks_guard_cycles(
        proj_dir,
        proj_dir,
        prune_fn=desk_mod.prune_project_walk_dirs,
    ):
        pass
    rec_dir = proj_dir / "records"
    if rec_dir.exists():
        for _root, _dirs, _files in _walk_followlinks_guard_cycles(
            rec_dir,
            proj_dir,
            prune_fn=lambda _r, ds, _p: ds.__setitem__(slice(None), [d for d in ds if d not in desk_mod.PRUNED_WALK_DIRS]),
        ):
            pass

    images = desk_mod.enumerate_project_images(str(proj_dir), sidecars=True, include_record_assets=True)
    if not images:
        raise RefusalError("PROJECT_EMPTY", f"--project {proj_dir.name!r} contains zero readable images")

    selected_rels: set[str] = set()
    asset_rels: set[str] = set()
    approved_image_count = 0
    raw_recs = gate_mod.load_project_records(str(proj_dir)) if rec_dir.is_dir() else []

    if raw_recs:
        approved = [e for e in (gate_mod._extract_record_fields(r) for r in raw_recs) if e["status"] == "approved"]
        # A `rejected/` path is never an approved training image, whatever the record says (Publisher, 2026-11-12).
        approved = [
            e
            for e in approved
            if not path_is_rejected(proj_dir, e["record_file"]) and not path_is_rejected(proj_dir, e["asset_path"])
        ]
        if not approved:
            raise RefusalError("NO_APPROVED_RECORDS", f"--project {proj_dir.name!r} contains zero approved records")
        for entry in approved:
            rec_rel = str(entry["record_file"]).replace("\\", "/").strip("/")
            asset_rel = str(entry["asset_path"]).replace("\\", "/").strip("/")
            if not asset_rel:
                raise RefusalError("PROJECT_RECORD_INVALID", f"{rec_rel}: missing asset_path")
            asset_cand = proj_dir / asset_rel
            if not asset_cand.exists() or not asset_cand.is_file():
                raise RefusalError("PROJECT_FILE_MISSING", f"{rec_rel}: approved asset missing at {asset_rel!r}")
            if not is_image_file_by_magic(asset_cand):
                raise RefusalError(
                    "PROJECT_ASSET_NOT_IMAGE",
                    f"{rec_rel}: approved asset {asset_rel!r} is not an image (magic bytes); refusing to upload it",
                )
            selected_rels.add(rec_rel)
            selected_rels.add(asset_rel)
            asset_rels.add(asset_rel)
            approved_image_count += 1

            sidecar_rel = entry.get("sidecar_file")
            if isinstance(sidecar_rel, str) and sidecar_rel.strip():
                sc_norm = sidecar_rel.replace("\\", "/").strip("/")
                if (proj_dir / sc_norm).is_file():
                    selected_rels.add(sc_norm)

            parts = pathlib.PurePosixPath(asset_rel).parts
            if len(parts) >= 3 and parts[0] == "outputs":
                wave_rel = f"outputs/{parts[1]}"
                for wave_meta_name in ("generation.json", "_curation.json"):
                    meta_rel = f"{wave_rel}/{wave_meta_name}"
                    if (proj_dir / meta_rel).is_file():
                        selected_rels.add(meta_rel)
    else:
        sc_scan = gate_mod.scan_sidecar_project(str(proj_dir))
        approved_sc = [
            e
            for e in (gate_mod._extract_record_fields(r) for r in sc_scan.get("records", []))
            if e["status"] == "approved"
            and not path_is_rejected(proj_dir, e["asset_path"])
            and not path_is_rejected(proj_dir, e.get("sidecar_file"))
        ]
        if not approved_sc:
            raise RefusalError("NO_APPROVED_RECORDS", f"--project {proj_dir.name!r} contains zero paired sidecar records")
        for entry in approved_sc:
            asset_rel = str(entry["asset_path"]).replace("\\", "/").strip("/")
            asset_cand_sc = proj_dir / asset_rel
            if not asset_cand_sc.is_file() or not is_image_file_by_magic(asset_cand_sc):
                raise RefusalError(
                    "PROJECT_ASSET_NOT_IMAGE",
                    f"sidecar record asset {asset_rel!r} is not an image (magic bytes); refusing to upload it",
                )
            selected_rels.add(asset_rel)
            asset_rels.add(asset_rel)
            approved_image_count += 1
            sidecar_rel = entry.get("sidecar_file")
            if isinstance(sidecar_rel, str) and sidecar_rel.strip():
                sc_norm = sidecar_rel.replace("\\", "/").strip("/")
                if (proj_dir / sc_norm).is_file():
                    selected_rels.add(sc_norm)
            parts = pathlib.PurePosixPath(asset_rel).parts
            if len(parts) >= 3 and parts[0] == "outputs":
                wave_rel = f"outputs/{parts[1]}"
                for wave_meta_name in ("generation.json", "_curation.json"):
                    meta_rel = f"{wave_rel}/{wave_meta_name}"
                    if (proj_dir / meta_rel).is_file():
                        selected_rels.add(meta_rel)

    for cfg_name in desk_mod.PROJECT_CONFIG_FILENAMES:
        cfg_cand = proj_dir / cfg_name
        if cfg_cand.is_file():
            selected_rels.add(cfg_name)

    rel_files: list[tuple[str, pathlib.Path]] = []
    for rel_posix in sorted(selected_rels):
        norm_rel = rel_posix.replace("\\", "/")
        if norm_rel.startswith("/") or ".." in pathlib.PurePosixPath(norm_rel).parts:
            raise RefusalError("PRIVATE_IMAGE_ROUTE_FORBIDDEN", f"project path escapes root: {rel_posix!r}")
        if has_rejected_component(norm_rel):
            raise RefusalError("PRIVATE_IMAGE_ROUTE_FORBIDDEN", f"project path under rejected/ is never uploaded: {rel_posix!r}")
        cand = proj_dir / norm_rel
        real = _verify_inside_project(proj_dir, cand, label=norm_rel)
        if not real.is_file():
            raise RefusalError("PROJECT_FILE_MISSING", f"expected file missing at {norm_rel!r}")
        # A symlink or junction whose target sits under rejected/ is rejected too (checked on the resolved path).
        if path_is_rejected(proj_dir, norm_rel):
            raise RefusalError(
                "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
                f"project path {norm_rel!r} resolves into rejected/ and is never uploaded",
            )
        # Only the approved assets are images; every other uploaded file is a record, sidecar, or config.
        if norm_rel in asset_rels:
            if not is_image_file_by_magic(real):
                raise RefusalError("PROJECT_ASSET_NOT_IMAGE", f"upload asset {norm_rel!r} is not an image (magic bytes)")
            if real.stat().st_nlink > 1:
                # A hard link cannot be traced to its other names (one may sit under rejected/): copy the file instead.
                raise RefusalError(
                    "PROJECT_ASSET_HARDLINKED",
                    f"upload asset {norm_rel!r} has {real.stat().st_nlink} hard links; copy it to a plain file",
                )
        elif is_image_file_by_magic(real):
            raise RefusalError(
                "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
                f"{norm_rel!r} carries image bytes but is not an approved asset; it is not uploaded",
            )
        rel_files.append((norm_rel, real))
    return rel_files, approved_image_count


def cmd_make_stub_project(args: argparse.Namespace) -> int:
    from PIL import Image

    base = pathlib.Path(args.out_dir).resolve() / "_stub_input"
    proj = base / "project"
    wave = proj / "outputs" / "wave-1"
    recs = proj / "records"
    wave.mkdir(parents=True, exist_ok=True)
    recs.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (48, 48), (196, 52, 44)).save(wave / "plate_01.png")
    Image.new("RGB", (48, 48), (44, 156, 92)).save(wave / "plate_02.png")

    rec1 = {
        "id": "stub_flagged_01",
        "asset_path": "outputs/wave-1/plate_01.png",
        "canon": {"subject": "crimson stone pillar"},
        "judgment": {"status": "approved", "explanation": "crimson stone pillar"},
        "provenance": {"prompt": "crimson stone pillar in courtyard"},
    }
    rec2 = {
        "id": "stub_clean_02",
        "asset_path": "outputs/wave-1/plate_02.png",
        "canon": {"subject": "lantern stair"},
        "judgment": {
            "status": "approved",
            "explanation": (
                "in rustline style, a tarnished brass lantern hangs beside rain-dark basalt steps "
                "under a low timber eave with moss along the mortar"
            ),
        },
        "provenance": {"prompt": "emerald lantern hanging beside wet stair"},
    }
    write_json_lf(recs / "stub_flagged_01.json", rec1)
    write_json_lf(recs / "stub_clean_02.json", rec2)

    cap_cfg = base / "caption-rules.json"
    rules_doc = {
        "multimodal_pool": "rustline",
        "prefix": "in rustline style,",
        "drafter": "qwen3-vl",
        "drafter_prompt": "Describe the visible subject, setting, lighting, and materials in plain prose.",
    }
    write_json_lf(cap_cfg, rules_doc)
    print(json.dumps({"project_dir": str(proj), "caption_config": str(cap_cfg)}, sort_keys=True))
    return 0


def cmd_preflight_project(args: argparse.Namespace) -> int:
    repo_root = pathlib.Path(args.repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _check_no_private_uri(args.project, "--project")
    _check_no_private_uri(args.caption_config, "--caption-config")

    proj_raw = pathlib.Path(args.project)
    if not proj_raw.exists() or not proj_raw.is_dir():
        raise RefusalError("PROJECT_MISSING", f"--project directory does not exist: {args.project!r}")
    proj_dir = proj_raw.resolve()

    cap_raw = pathlib.Path(args.caption_config)
    if not cap_raw.exists() or not cap_raw.is_file():
        raise RefusalError("CAPTION_CONFIG_MISSING", f"--caption-config file does not exist: {args.caption_config!r}")
    cap_path = cap_raw.resolve()

    out_dir = pathlib.Path(args.out_dir).resolve()
    try:
        inside = os.path.commonpath([str(proj_dir), str(out_dir)]) == str(proj_dir)
    except ValueError:
        inside = False
    if inside:
        raise RefusalError("OUT_INSIDE_PROJECT", "--out is inside --project; the desk never writes into a project")

    try:
        cap_doc = json.loads(cap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefusalError("CAPTION_CONFIG_INVALID", f"cannot parse --caption-config JSON: {exc}") from exc
    if not isinstance(cap_doc, dict):
        raise RefusalError("CAPTION_CONFIG_INVALID", "--caption-config must be a JSON object")

    raw_pool = cap_doc.get("multimodal_pool")
    if raw_pool is None or (isinstance(raw_pool, str) and not raw_pool.strip()):
        raise RefusalError(
            "MULTIMODAL_POOL_TAG_MISSING",
            "--caption-config must carry an explicit 'multimodal_pool' tag",
        )

    from dataset_desk import description_gate, desk
    from dataset_desk.errors import DeskError

    if not isinstance(raw_pool, str) or raw_pool.strip().lower() not in desk.MULTIMODAL_POOL:
        raise RefusalError(
            "MULTIMODAL_POOL_TAG_INVALID",
            f"--caption-config 'multimodal_pool'={raw_pool!r} is not in {sorted(desk.MULTIMODAL_POOL)}",
        )
    pool_tag = raw_pool.strip().lower()

    for k, v in cap_doc.items():
        if isinstance(v, str):
            _check_no_private_uri(v, f"caption-config.{k}")

    try:
        description_gate.load_caption_rules(project=str(proj_dir), config_path=str(cap_path))
    except DeskError as exc:
        raise RefusalError("CAPTION_CONFIG_INVALID", exc.message) from exc

    rec_dir = proj_dir / "records"
    if rec_dir.is_dir():
        for rfile in sorted(rec_dir.glob("*.json")):
            try:
                rdoc = json.loads(rfile.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RefusalError("PROJECT_RECORD_INVALID", f"invalid record JSON {rfile.name}: {exc}") from exc
            if isinstance(rdoc, dict):
                for fld in ("asset_path", "image_url", "url", "source_url", "uri"):
                    val = rdoc.get(fld)
                    if isinstance(val, str) and val.strip():
                        _check_no_private_uri(val, f"{rfile.name}:{fld}")
                ap_val = rdoc.get("asset_path")
                if isinstance(ap_val, str) and ap_val.strip():
                    norm_ap = ap_val.replace("\\", "/")
                    if norm_ap.startswith("/") or ".." in pathlib.PurePosixPath(norm_ap).parts:
                        raise RefusalError(
                            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
                            f"{rfile.name}:asset_path escapes project root: {ap_val!r}",
                        )

    rel_files, approved_image_count = _collect_compose_upload_files(proj_dir, desk, description_gate)

    class _Qwen3VLProbe:
        name = "qwen3-vl"
        drafter_id = QWEN3_VL_REPO
        revision = QWEN3_VL_REVISION
        is_arm_model = False

    prev_user_cfg = os.environ.get("DATASET_DESK_USER_CONFIG")
    if args.isolate_user_config:
        os.environ["DATASET_DESK_USER_CONFIG"] = ""
    try:
        pool_detection = desk.check_drafter_pool(
            str(proj_dir),
            _Qwen3VLProbe(),
            allow_multimodal_pool=False,
            caption_config=str(cap_path),
        )
    except DeskError as exc:
        raise RefusalError(exc.code, scrub_rig_paths(exc.message)) from exc
    finally:
        if args.isolate_user_config:
            if prev_user_cfg is None:
                os.environ.pop("DATASET_DESK_USER_CONFIG", None)
            else:
                os.environ["DATASET_DESK_USER_CONFIG"] = prev_user_cfg

    stage_dir = out_dir / "_rig_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    tar_path = stage_dir / "project_input.tar"
    manifest_entries: list[dict] = []

    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        for rel_posix, real_path in rel_files:
            payload: bytes | None = None
            if not is_image_file_by_magic(real_path):
                # Records, sidecars, configs and wave metadata are uploaded as scrubbed copies: they can carry
                # absolute rig paths (generation.json source dirs, record provenance) that must not reach the VM.
                raw_bytes = real_path.read_bytes()
                try:
                    as_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    as_text = None
                if as_text is None:
                    raise RefusalError(
                        "UPLOAD_NON_UTF8_TEXT",
                        f"{rel_posix!r} is not an image and not valid UTF-8, so it cannot be scrubbed of rig paths",
                    )
                else:
                    cleaned_text, scrub_status = scrub_text_for_file(rel_posix, as_text)
                    if scrub_status in ("json_error", "json_error_scrubbed"):
                        raise RefusalError(
                            "UPLOAD_SCRUB_INVALID_JSON",
                            f"{rel_posix!r} is not valid JSON and a rig-path scrub cannot be verified; fix or remove it",
                        )
                    payload = cleaned_text.encode("utf-8")
            if payload is not None:
                size_up, fsha = len(payload), sha256_bytes(payload)
            else:
                size_up, fsha = real_path.stat().st_size, sha256_file(real_path)
            manifest_entries.append(
                {
                    "path": rel_posix,
                    "bytes": size_up,
                    "sha256": fsha,
                }
            )
            ti = tarfile.TarInfo(name=rel_posix)
            ti.size = size_up
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = 0
            ti.gid = 0
            ti.uname = ""
            ti.gname = ""
            if payload is not None:
                tf.addfile(ti, io.BytesIO(payload))
            else:
                with real_path.open("rb") as fh:
                    tf.addfile(ti, fh)

    staged_cap = stage_dir / "caption_config.json"
    cap_bytes = scrub_rig_paths(json.dumps(cap_doc, indent=2, sort_keys=True)).encode("utf-8") + b"\n"
    staged_cap.write_bytes(cap_bytes)

    uploaded_file_count = len(manifest_entries)
    uploaded_total_bytes = sum(int(item["bytes"]) for item in manifest_entries)
    archive_sha = sha256_file(tar_path)
    archive_bytes = tar_path.stat().st_size
    manifest_doc = {
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "caption_config_bytes": len(cap_bytes),
        "caption_config_sha256": sha256_bytes(cap_bytes),
        "delivery_method": "colab_upload_checked",
        "files": manifest_entries,
        "image_count": approved_image_count,
        "multimodal_pool": pool_tag,
        "multimodal_pool_locked": True,
        "pool_detection": sanitize_pool_detection(pool_detection),
        "uploaded_file_count": uploaded_file_count,
        "uploaded_total_bytes": uploaded_total_bytes,
    }
    manifest_path = out_dir / "project_manifest.json"
    write_json_lf(manifest_path, manifest_doc)
    print(json.dumps(manifest_doc, sort_keys=True))
    return 0


def _urlsafe_sha256_and_len(data: bytes) -> tuple[str, int]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    return f"sha256={digest}", len(data)


def cmd_build_wheel(args: argparse.Namespace) -> int:
    repo_root = pathlib.Path(args.repo_root).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    stage_dir = out_dir / "_rig_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)

    pyproj = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    proj_meta = pyproj["project"]
    raw_name = str(proj_meta["name"])
    dist_name = raw_name.replace("-", "_")
    version = str(proj_meta["version"])
    summary = str(proj_meta.get("description", ""))
    req_py = str(proj_meta.get("requires-python", ">=3.11"))
    deps = list(proj_meta.get("dependencies", []))
    scripts = dict(proj_meta.get("scripts", {}))
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    license_bytes = (repo_root / "LICENSE").read_bytes().replace(b"\r\n", b"\n")

    dist_info = f"{dist_name}-{version}.dist-info"
    wheel_name = f"{dist_name}-{version}-py3-none-any.whl"
    wheel_path = stage_dir / wheel_name

    pkg_dir = repo_root / "dataset_desk"
    entries: list[tuple[str, bytes]] = []
    for p in sorted(pkg_dir.iterdir(), key=lambda x: x.name):
        if not p.is_file():
            continue
        if p.suffix == ".py":
            entries.append((f"dataset_desk/{p.name}", p.read_bytes().replace(b"\r\n", b"\n")))
        elif p.name == "stopwords.txt":
            entries.append((f"dataset_desk/{p.name}", p.read_bytes()))

    meta_lines = [
        "Metadata-Version: 2.1",
        f"Name: {raw_name}",
        f"Version: {version}",
        f"Summary: {summary}",
        f"Requires-Python: {req_py}",
        "License: MIT",
    ]
    for dep in deps:
        meta_lines.append(f"Requires-Dist: {dep}")
    meta_lines.append("Description-Content-Type: text/markdown")
    meta_lines.append("")
    meta_lines.append(readme_text)
    metadata_bytes = "\n".join(meta_lines).encode("utf-8")

    wheel_meta_bytes = (
        b"Wheel-Version: 1.0\nGenerator: dataset-desk-colab-builder\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )

    ep_lines = ["[console_scripts]"]
    for k, v in sorted(scripts.items()):
        ep_lines.append(f"{k} = {v}")
    ep_bytes = ("\n".join(ep_lines) + "\n").encode("utf-8")

    entries.append((f"{dist_info}/LICENSE", license_bytes))
    entries.append((f"{dist_info}/METADATA", metadata_bytes))
    entries.append((f"{dist_info}/WHEEL", wheel_meta_bytes))
    entries.append((f"{dist_info}/entry_points.txt", ep_bytes))

    record_lines: list[str] = []
    for arcname, payload in entries:
        h_str, size = _urlsafe_sha256_and_len(payload)
        record_lines.append(f"{arcname},{h_str},{size}")
    record_arc = f"{dist_info}/RECORD"
    record_lines.append(f"{record_arc},,")
    record_bytes = ("\n".join(record_lines) + "\n").encode("utf-8")
    entries.append((record_arc, record_bytes))

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, payload in entries:
            zi = zipfile.ZipInfo(filename=arcname, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, payload)

    w_sha = sha256_file(wheel_path)
    w_bytes = wheel_path.stat().st_size
    doc = {
        "bytes": w_bytes,
        "git_commit": args.git_commit,
        "sha256": w_sha,
        "version": version,
        "wheel_name": wheel_name,
        "wheel_path": f"_rig_stage/{wheel_name}",
    }
    write_json_lf(out_dir / "wheel_manifest.json", doc)
    print(json.dumps(doc, sort_keys=True))
    return 0


def _collect_colab_runtime_version() -> str:
    env_tag = os.environ.get("COLAB_RELEASE_TAG", "").strip()
    if env_tag:
        return env_tag
    for candidate in ("/etc/colab-release", "/var/colab/runtime-version", "/etc/os-release"):
        p = pathlib.Path(candidate)
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return f"{p.name}:{line}"
            except OSError:
                continue
    return f"platform:{sys.platform}:{os.uname().release if hasattr(os, 'uname') else 'unknown'}"


def _collect_usr_local_cuda(usr_local: pathlib.Path) -> list[str]:
    if not usr_local.is_dir():
        return []
    out: list[str] = []
    for entry in sorted(usr_local.glob("cuda*"), key=lambda p: p.name):
        if entry.is_symlink():
            try:
                target = pathlib.Path(os.readlink(entry)).name
            except OSError:
                target = "?"
            out.append(f"{entry.name} -> {target}")
        else:
            out.append(entry.name)
    return out


def _read_mounts(mounts_path: pathlib.Path) -> list[tuple[str, str]]:
    if not mounts_path.is_file():
        return []
    entries: list[tuple[str, str]] = []
    try:
        for line in mounts_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mount_point = parts[1].replace("\\040", " ")
                fstype = parts[2].lower()
                entries.append((mount_point, fstype))
    except OSError:
        return []
    return entries


def _mount_fstype_for_path(target: pathlib.Path, mounts: list[tuple[str, str]]) -> tuple[str, str]:
    best_mp = ""
    best_fs = "local"
    target_posix = target.resolve().as_posix()
    for mp, fs in mounts:
        mp_clean = mp.rstrip("/") or "/"
        if (
            target_posix == mp_clean or mp_clean == "/" or target_posix.startswith(mp_clean + "/")
        ) and len(mp_clean) >= len(best_mp):
            best_mp = mp_clean
            best_fs = fs
    return best_mp or "/", best_fs


def _read_host_ram_avail_gib(meminfo_path: pathlib.Path) -> float:
    override = os.environ.get("DESK_VM_OVERRIDE_RAM_GIB", "").strip()
    if override:
        return float(override)
    if meminfo_path.is_file():
        avail_kib = None
        total_kib = None
        for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                avail_kib = int(line.split()[1])
            elif line.startswith("MemTotal:"):
                total_kib = int(line.split()[1])
        chosen = avail_kib if avail_kib is not None else total_kib
        if chosen is not None:
            return round(chosen / (1024.0 * 1024.0), 3)
    return 16.0


def _read_disk_free_gib(target_dir: pathlib.Path) -> float:
    override = os.environ.get("DESK_VM_OVERRIDE_DISK_GIB", "").strip()
    if override:
        return float(override)
    target_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    return round(usage.free / float(1 << 30), 3)


def _resolve_on_path(prog: str, path_val: str | None) -> str:
    p_obj = pathlib.Path(prog)
    if p_obj.parent != pathlib.Path(".") or "/" in prog or "\\" in prog:
        if p_obj.is_file():
            return str(p_obj)
        if os.name == "nt" and p_obj.with_suffix(".exe").is_file():
            return str(p_obj.with_suffix(".exe"))
        return prog
    search_path = path_val if path_val is not None else os.environ.get("PATH", "")
    for entry in search_path.split(os.pathsep):
        if not entry:
            continue
        d = pathlib.Path(entry)
        cand = d / prog
        if cand.is_file():
            return str(cand)
        if os.name == "nt" and (d / f"{prog}.exe").is_file():
            return str(d / f"{prog}.exe")
    found = shutil.which(prog, path=search_path)
    return found or prog


def _build_exec_cmd(prog: str, argv: list[str], *, env: dict[str, str] | None = None) -> list[str]:
    path_val = (env or os.environ).get("PATH")
    resolved = _resolve_on_path(prog, path_val)
    if os.name == "nt" and not resolved.lower().endswith(".exe"):
        rp = pathlib.Path(resolved)
        if rp.is_file() and rp.stat().st_size > 0:
            first = rp.read_bytes().splitlines()[0].decode("utf-8", errors="replace").strip()
            if first.startswith("#!"):
                interp = first[2:].strip()
                if "bash" in interp:
                    git_bash = "C:/Program Files/Git/bin/bash.exe"
                    bash_bin = (
                        git_bash
                        if pathlib.Path(git_bash).is_file()
                        else (shutil.which("bash", path=path_val) or "bash")
                    )
                    return [bash_bin, resolved, *argv]
                if interp:
                    return [interp, resolved, *argv]
    return [resolved, *argv]


def cmd_vm_probe_and_guards(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    record_vm_step(state_dir, "vm_probe_and_guards")

    usr_local = pathlib.Path(os.environ.get("DESK_VM_USR_LOCAL", "/usr/local"))
    mounts_file = pathlib.Path(os.environ.get("DESK_VM_PROC_MOUNTS", "/proc/mounts"))
    meminfo_file = pathlib.Path(os.environ.get("DESK_VM_PROC_MEMINFO", "/proc/meminfo"))
    drive_dir = pathlib.Path(os.environ.get("DESK_VM_CONTENT_DRIVE", "/content/drive"))

    runtime_ver = _collect_colab_runtime_version()
    cuda_listing = _collect_usr_local_cuda(usr_local)
    probe_doc: dict[str, object] = {
        "colab_runtime_version": runtime_ver,
        "guards_passed": False,
        "mode": args.mode,
        "requested_card": args.card,
        "usr_local_cuda": cuda_listing,
    }
    write_json_lf(state_dir / "vm_probe.json", probe_doc)

    mounts = _read_mounts(mounts_file)
    if drive_dir.exists() or any("drive" in mp.lower() or fs == "fuse.drive" for mp, fs in mounts):
        raise RefusalError(
            "PRIVATE_IMAGE_ROUTE_FORBIDDEN",
            "Google Drive mount detected on VM (/content/drive); private images forbid Drive mounts",
            step="vm_probe_and_guards",
        )

    gpu_name = None
    gpu_vram_mib = 0
    gpu_driver = None
    if args.mode == "gpu":
        try:
            proc = subprocess.run(
                _build_exec_cmd(
                    "nvidia-smi",
                    ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                ),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RefusalError(
                "GPU_MISSING",
                "nvidia-smi not found on GPU VM",
                step="vm_probe_and_guards",
            ) from exc
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RefusalError(
                "GPU_MISSING",
                f"nvidia-smi failed (rc={proc.returncode}): {proc.stderr.strip() or 'no GPU reported'}",
                step="vm_probe_and_guards",
            )
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) < 2:
            raise RefusalError("GPU_MISSING", f"unexpected nvidia-smi output: {first_line!r}", step="vm_probe_and_guards")
        gpu_name = parts[0]
        try:
            gpu_vram_mib = int(float(parts[1]))
        except ValueError as exc:
            raise RefusalError("GPU_MISSING", f"non-numeric GPU VRAM in {first_line!r}", step="vm_probe_and_guards") from exc
        gpu_driver = parts[2] if len(parts) >= 3 else None

        upper_name = gpu_name.upper()
        matched_allowed = any(tok in upper_name for tok in ("L4", "A100", "H100"))
        if not matched_allowed:
            probe_doc.update({"gpu_name": gpu_name, "gpu_vram_mib": gpu_vram_mib, "gpu_driver": gpu_driver})
            write_json_lf(state_dir / "vm_probe.json", probe_doc)
            raise RefusalError(
                "GPU_CARD_TOO_SMALL",
                f"VM GPU {gpu_name!r} is smaller than required L4/A100/H100 class",
                step="vm_probe_and_guards",
            )
        if gpu_vram_mib < GPU_MIN_VRAM_MIB:
            probe_doc.update({"gpu_name": gpu_name, "gpu_vram_mib": gpu_vram_mib, "gpu_driver": gpu_driver})
            write_json_lf(state_dir / "vm_probe.json", probe_doc)
            raise RefusalError(
                "GPU_VRAM_TOO_LOW",
                f"VM GPU {gpu_name!r} has {gpu_vram_mib} MiB VRAM < {GPU_MIN_VRAM_MIB} MiB floor",
                step="vm_probe_and_guards",
            )
    elif args.mode == "stub" and args.card != "CPU-high-mem":
        raise RefusalError(
            "STUB_CARD_MISMATCH",
            f"--stub requires --card CPU-high-mem, got {args.card!r}",
            step="vm_probe_and_guards",
        )

    disk_free_gib = _read_disk_free_gib(pathlib.Path(args.work_root))
    ram_avail_gib = _read_host_ram_avail_gib(meminfo_file)
    probe_doc.update(
        {
            "disk_free_gib": disk_free_gib,
            "gpu_driver": gpu_driver,
            "gpu_name": gpu_name,
            "gpu_vram_mib": gpu_vram_mib,
            "host_ram_avail_gib": ram_avail_gib,
        }
    )
    write_json_lf(state_dir / "vm_probe.json", probe_doc)

    if disk_free_gib < args.min_disk_gib:
        raise RefusalError(
            "DISK_FREE_LOW",
            f"VM free disk {disk_free_gib:.2f} GiB < {args.min_disk_gib:.2f} GiB floor",
            step="vm_probe_and_guards",
        )
    if ram_avail_gib < args.min_ram_gib:
        raise RefusalError(
            "HOST_RAM_LOW",
            f"VM available host RAM {ram_avail_gib:.2f} GiB < {args.min_ram_gib:.2f} GiB floor",
            step="vm_probe_and_guards",
        )

    probe_doc["guards_passed"] = True
    write_json_lf(state_dir / "vm_probe.json", probe_doc)
    print(json.dumps(probe_doc, sort_keys=True))
    return 0


def cmd_verify_uploaded_wheel(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    record_vm_step(state_dir, "verify_uploaded_wheel")

    wheel_path = pathlib.Path(args.wheel_path)
    lock_path = pathlib.Path(args.lock_path)
    if not wheel_path.is_file():
        raise RefusalError("WHEEL_SHA_MISMATCH", f"uploaded desk wheel missing at {str(wheel_path)!r}", step="verify_uploaded_wheel")
    actual_wheel_sha = sha256_file(wheel_path)
    if actual_wheel_sha != args.expected_wheel_sha:
        raise RefusalError(
            "WHEEL_SHA_MISMATCH",
            f"uploaded wheel sha256 {actual_wheel_sha} != expected {args.expected_wheel_sha}",
            step="verify_uploaded_wheel",
        )
    if not lock_path.is_file():
        raise RefusalError("LOCK_HASH_MISMATCH", f"uploaded lockfile missing at {str(lock_path)!r}", step="verify_uploaded_wheel")
    actual_lock_sha = sha256_file(lock_path)
    if actual_lock_sha != args.expected_lock_sha:
        raise RefusalError(
            "LOCK_HASH_MISMATCH",
            f"uploaded desk_draft.lock sha256 {actual_lock_sha} != expected {args.expected_lock_sha}",
            step="verify_uploaded_wheel",
        )
    doc = {
        "lock_sha256": actual_lock_sha,
        "ok": True,
        "wheel_name": wheel_path.name,
        "wheel_sha256": actual_wheel_sha,
    }
    write_json_lf(state_dir / "wheel_verify.json", doc)
    print(json.dumps(doc, sort_keys=True))
    return 0


def _shred_and_unlink_token(token_path: pathlib.Path, *, step: str) -> str | None:
    if not token_path.exists():
        return None
    if os.environ.get("DESK_VM_SIMULATE_SHRED_FAIL", "").strip() == "1":
        raise RefusalError("TOKEN_SHRED_FAILED", f"failed to shred and remove {str(token_path)!r}", step=step)
    try:
        st = token_path.stat()
        if os.name != "nt" and (stat.S_IMODE(st.st_mode) & 0o077) != 0:
            raise RefusalError(
                "TOKEN_FILE_PERMS_BAD",
                f"token file {str(token_path)!r} has unsafe mode {oct(stat.S_IMODE(st.st_mode))}; expected 0600",
                step=step,
            )
        raw = token_path.read_text(encoding="utf-8").strip()
        size = max(64, st.st_size)
        with token_path.open("r+b") as fh:
            fh.seek(0)
            fh.write(os.urandom(size))
            fh.flush()
            os.fsync(fh.fileno())
            fh.seek(0)
            fh.write(b"\x00" * size)
            fh.flush()
            os.fsync(fh.fileno())
        token_path.unlink()
    except RefusalError:
        raise
    except OSError as exc:
        raise RefusalError("TOKEN_SHRED_FAILED", f"failed shredding token file: {exc}", step=step) from exc
    if token_path.exists():
        raise RefusalError("TOKEN_SHRED_FAILED", f"token file still exists after unlink: {str(token_path)!r}", step=step)
    return raw or None


def _is_429_error(exc: BaseException) -> bool:
    resp = getattr(exc, "response", None)
    status_code = getattr(resp, "status_code", None)
    if status_code == 429:
        return True
    msg = str(exc)
    return bool(re.search(r"\b429\b|Too Many Requests", msg, re.IGNORECASE))


def _extract_expected_sibling_sha(sib: object) -> str | None:
    lfs = getattr(sib, "lfs", None)
    if isinstance(lfs, dict) and isinstance(lfs.get("sha256"), str):
        return lfs["sha256"].lower()
    if lfs is not None and isinstance(getattr(lfs, "sha256", None), str):
        return str(lfs.sha256).lower()
    sha_attr = getattr(sib, "blob_sha256", None) or getattr(sib, "sha256", None)
    if isinstance(sha_attr, str) and len(sha_attr) == 64:
        return sha_attr.lower()
    return None


# The 16 files of the pinned Qwen/Qwen3-VL-8B-Instruct revision: 4 LFS safetensors shards (verified by LFS sha256)
# and 12 plain git files (verified by git blob id). Any missing file refuses.
PINNED_QWEN3_VL_FILES = frozenset(
    {
        ".gitattributes",
        "README.md",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    }
)


def _extract_expected_sibling_blob_id(sib: object) -> str | None:
    blob_id = getattr(sib, "blob_id", None)
    if isinstance(blob_id, str) and re.fullmatch(r"[0-9a-fA-F]{40}", blob_id):
        return blob_id.lower()
    return None


def git_blob_sha1(path: pathlib.Path) -> str:
    """git blob id: sha1 of the blob header plus content, streamed."""
    p = pathlib.Path(path)
    h = hashlib.sha1()
    h.update(b"blob " + str(p.stat().st_size).encode("ascii") + b"\x00")
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_download_weights(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    step = "weight_download"
    record_vm_step(state_dir, step)

    hf_home = pathlib.Path(args.hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    mounts_file = pathlib.Path(os.environ.get("DESK_VM_PROC_MOUNTS", "/proc/mounts"))
    mounts = _read_mounts(mounts_file)
    mp, fstype = _mount_fstype_for_path(hf_home, mounts)
    if fstype in FORBIDDEN_MOUNT_FSTYPES or "drive" in mp.lower():
        raise RefusalError(
            "WEIGHT_MOUNT_FORBIDDEN",
            f"weight directory {str(hf_home)!r} sits on mount {mp!r} (fstype={fstype!r}); weights must live on local disk",
            step=step,
        )

    token: str | None = None
    if args.token_file:
        token = _shred_and_unlink_token(pathlib.Path(args.token_file), step=step)
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ["HF_HOME"] = str(hf_home)

    if args.revision != QWEN3_VL_REVISION or args.repo_id != QWEN3_VL_REPO:
        raise RefusalError(
            "WEIGHT_REVISION_MISMATCH",
            f"requested {args.repo_id}@{args.revision} != pinned {QWEN3_VL_REPO}@{QWEN3_VL_REVISION}",
            step=step,
        )

    from huggingface_hub import HfApi, snapshot_download

    bound_s = float(args.download_bound_s)
    base_backoff_s = float(args.backoff_base_s)
    max_attempts = max(1, int(args.max_attempts))
    t_start = time.monotonic()
    attempts = 0
    retried_429 = 0
    snapshot_dir_str: str | None = None
    info = None

    while True:
        attempts += 1
        elapsed = time.monotonic() - t_start
        if elapsed > bound_s:
            code = "DOWNLOAD_429_BOUND_EXCEEDED" if retried_429 > 0 else "DOWNLOAD_BOUND_EXCEEDED"
            raise RefusalError(
                code,
                f"weight download exceeded {bound_s:.2f}s bound after {attempts - 1} attempt(s) ({retried_429} HTTP 429)",
                step=step,
            )
        try:
            api = HfApi()
            info = api.model_info(
                args.repo_id,
                revision=args.revision,
                files_metadata=True,
                token=token,
            )
            snapshot_dir_str = snapshot_download(
                repo_id=args.repo_id,
                revision=args.revision,
                cache_dir=str(hf_home / "hub"),
                token=token,
            )
            break
        except Exception as exc:
            now_elapsed = time.monotonic() - t_start
            if _is_429_error(exc):
                retried_429 += 1
                sleep_s = base_backoff_s * (2 ** (retried_429 - 1))
                if attempts >= max_attempts or (now_elapsed + sleep_s) > bound_s:
                    raise RefusalError(
                        "DOWNLOAD_429_BOUND_EXCEEDED",
                        f"HTTP 429 rate limit exceeded download bound ({bound_s:.2f}s, attempts={attempts}, 429s={retried_429})",
                        step=step,
                        retryable=True,
                    ) from exc
                time.sleep(sleep_s)
                continue
            if now_elapsed > bound_s:
                raise RefusalError(
                    "DOWNLOAD_BOUND_EXCEEDED",
                    f"weight download exceeded {bound_s:.2f}s bound: {exc}",
                    step=step,
                ) from exc
            raise RefusalError("WEIGHT_DOWNLOAD_FAILED", f"huggingface_hub download failed: {exc}", step=step) from exc
    token = None

    total_elapsed = time.monotonic() - t_start
    if total_elapsed > bound_s:
        code = "DOWNLOAD_429_BOUND_EXCEEDED" if retried_429 > 0 else "DOWNLOAD_BOUND_EXCEEDED"
        raise RefusalError(
            code,
            f"weight download took {total_elapsed:.2f}s > {bound_s:.2f}s bound",
            step=step,
        )

    resolved_sha = str(getattr(info, "sha", "") or "")
    if resolved_sha != args.revision:
        raise RefusalError(
            "WEIGHT_REVISION_MISMATCH",
            f"model_info resolved sha {resolved_sha!r} != pinned {args.revision!r}",
            step=step,
        )

    assert snapshot_dir_str is not None
    snapshot_dir = pathlib.Path(snapshot_dir_str)
    if not snapshot_dir.is_dir():
        raise RefusalError("WEIGHT_FILE_MISSING", f"snapshot directory missing: {snapshot_dir_str!r}", step=step)

    siblings = list(getattr(info, "siblings", None) or [])
    if not siblings:
        raise RefusalError("WEIGHT_FILE_MISSING", "model_info returned zero sibling files", step=step)

    sha_verified_count = 0
    blob_verified_count = 0
    for sib in siblings:
        rfilename = getattr(sib, "rfilename", None)
        if not isinstance(rfilename, str) or not rfilename:
            continue
        local_f = snapshot_dir / rfilename
        if not local_f.is_file():
            raise RefusalError("WEIGHT_FILE_MISSING", f"expected weight file missing: {rfilename}", step=step)
        real_f = local_f.resolve()
        f_mp, f_fs = _mount_fstype_for_path(real_f, mounts)
        if f_fs in FORBIDDEN_MOUNT_FSTYPES or "drive" in f_mp.lower():
            raise RefusalError(
                "WEIGHT_MOUNT_FORBIDDEN",
                f"weight blob {rfilename!r} resolves onto forbidden mount {f_mp!r} ({f_fs!r})",
                step=step,
            )
        expected_sha = _extract_expected_sibling_sha(sib)
        if expected_sha is not None:
            actual_sha = sha256_file(real_f)
            if actual_sha != expected_sha:
                raise RefusalError(
                    "WEIGHT_SHA_MISMATCH",
                    f"weight file {rfilename!r} sha256 {actual_sha} != expected {expected_sha}",
                    step=step,
                )
            sha_verified_count += 1
        else:
            expected_blob = _extract_expected_sibling_blob_id(sib)
            if expected_blob is None:
                raise RefusalError(
                    "WEIGHT_BLOB_ID_MISSING",
                    f"weight file {rfilename!r} has neither an LFS sha256 nor a git blob id to verify against",
                    step=step,
                )
            actual_blob = git_blob_sha1(real_f)
            if actual_blob != expected_blob:
                raise RefusalError(
                    "WEIGHT_BLOB_MISMATCH",
                    f"weight file {rfilename!r} git blob id {actual_blob} != expected {expected_blob}",
                    step=step,
                )
            blob_verified_count += 1

    if args.repo_id == QWEN3_VL_REPO and args.revision == QWEN3_VL_REVISION:
        present = {getattr(sib, "rfilename", None) for sib in siblings}
        missing = sorted(PINNED_QWEN3_VL_FILES - present)
        if missing:
            raise RefusalError(
                "WEIGHT_FILE_MISSING",
                f"pinned revision lists {len(present)} files; missing {len(missing)}: {', '.join(missing)}",
                step=step,
            )
        on_disk = {
            f.relative_to(snapshot_dir).as_posix()
            for f in snapshot_dir.rglob("*")
            if (f.is_file() or f.is_symlink()) and ".cache" not in f.relative_to(snapshot_dir).parts
        }
        extra = sorted((present | on_disk) - PINNED_QWEN3_VL_FILES - {None})
        if extra:
            raise RefusalError(
                "WEIGHT_UNEXPECTED_FILE",
                f"pinned revision holds {len(extra)} file(s) beyond the 16 pinned names: {', '.join(extra)}",
                step=step,
            )

    try:
        snapshot_rel = snapshot_dir.resolve().relative_to(hf_home.resolve()).as_posix()
    except ValueError:
        snapshot_rel = snapshot_dir.name
    doc = {
        "attempts": attempts,
        "download_bound_s": bound_s,
        "elapsed_s": round(total_elapsed, 4),
        "blob_verified_count": blob_verified_count,
        "file_count": len(siblings),
        "hf_home_dir": hf_home.name,
        "local_disk_verified": True,
        "repo_id": args.repo_id,
        "retried_429_count": retried_429,
        "revision": args.revision,
        "sha_verified_count": sha_verified_count,
        "snapshot_rel_dir": snapshot_rel,
    }
    write_json_lf(state_dir / "weight_download.json", doc)
    print(json.dumps(doc, sort_keys=True))
    return 0


def cmd_config_only_load(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    step = "config_only_load"
    record_vm_step(state_dir, step)

    os.environ["HF_HOME"] = str(pathlib.Path(args.hf_home))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.pop("HF_TOKEN", None)

    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(
            args.repo_id,
            revision=args.revision,
            local_files_only=True,
        )
    except Exception as exc:
        raise RefusalError("CONFIG_LOAD_FAILED", f"offline AutoConfig load failed: {exc}", step=step) from exc

    commit_hash = getattr(cfg, "_commit_hash", None)
    if commit_hash is not None and str(commit_hash) != args.revision:
        raise RefusalError(
            "WEIGHT_REVISION_MISMATCH",
            f"AutoConfig._commit_hash {commit_hash!r} != pinned {args.revision!r}",
            step=step,
        )

    doc = {
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "model_type": getattr(cfg, "model_type", "qwen3_vl"),
        "ok": True,
        "repo_id": args.repo_id,
        "revision": args.revision,
    }
    write_json_lf(state_dir / "config_load.json", doc)
    print(json.dumps(doc, sort_keys=True))
    return 0


def _safe_extract_tar(tar_path: pathlib.Path, dest_dir: pathlib.Path, *, step: str) -> None:
    dest_resolved = dest_dir.resolve()
    dest_resolved.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            norm_name = member.name.replace("\\", "/")
            if norm_name.startswith("/") or ".." in pathlib.PurePosixPath(norm_name).parts:
                raise RefusalError("PROJECT_ARCHIVE_UNSAFE", f"unsafe tar member {member.name!r}", step=step)
            if member.issym() or member.islnk():
                raise RefusalError("PROJECT_ARCHIVE_UNSAFE", f"tar link member forbidden: {member.name!r}", step=step)
            target = (dest_resolved / norm_name).resolve()
            if os.path.commonpath([str(dest_resolved), str(target)]) != str(dest_resolved):
                raise RefusalError("PROJECT_ARCHIVE_UNSAFE", f"tar member escapes dest: {member.name!r}", step=step)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise RefusalError("PROJECT_FILE_MISSING", f"cannot extract {member.name!r}", step=step)
                with target.open("wb") as out_fh:
                    shutil.copyfileobj(extracted, out_fh)


def cmd_unpack_project(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    step = "unpack_project"
    record_vm_step(state_dir, step)

    tar_path = pathlib.Path(args.tar_path)
    manifest_path = pathlib.Path(args.manifest_path)
    cap_path = pathlib.Path(args.caption_config_path)
    dest_dir = pathlib.Path(args.dest_dir)

    if not manifest_path.is_file():
        raise RefusalError("PROJECT_MANIFEST_MISSING", f"missing {str(manifest_path)!r}", step=step)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not tar_path.is_file():
        raise RefusalError("PROJECT_ARCHIVE_SHA_MISMATCH", f"missing {str(tar_path)!r}", step=step)
    actual_tar_sha = sha256_file(tar_path)
    actual_tar_bytes = tar_path.stat().st_size
    if actual_tar_sha != manifest.get("archive_sha256") or actual_tar_bytes != manifest.get("archive_bytes"):
        raise RefusalError(
            "PROJECT_ARCHIVE_SHA_MISMATCH",
            f"uploaded project_input.tar sha/bytes ({actual_tar_sha}, {actual_tar_bytes}) != "
            f"manifest ({manifest.get('archive_sha256')}, {manifest.get('archive_bytes')})",
            step=step,
        )

    if not cap_path.is_file():
        raise RefusalError("CAPTION_CONFIG_MISSING", f"uploaded caption_config missing at {str(cap_path)!r}", step=step)
    if sha256_file(cap_path) != manifest.get("caption_config_sha256"):
        raise RefusalError("CAPTION_CONFIG_INVALID", "uploaded caption_config sha256 mismatch", step=step)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    _safe_extract_tar(tar_path, dest_dir, step=step)

    for entry in manifest.get("files", []):
        rel_p = entry["path"]
        f_full = dest_dir / rel_p
        if not f_full.is_file():
            raise RefusalError("PROJECT_FILE_MISSING", f"unpacked project missing {rel_p!r}", step=step)
        if f_full.stat().st_size != entry["bytes"] or sha256_file(f_full) != entry["sha256"]:
            raise RefusalError("PROJECT_FILE_SHA_MISMATCH", f"unpacked project sha mismatch on {rel_p!r}", step=step)

    doc = {"file_count": len(manifest.get("files", [])), "multimodal_pool": manifest.get("multimodal_pool"), "ok": True}
    write_json_lf(state_dir / "unpack_project.json", doc)
    print(json.dumps(doc, sort_keys=True))
    return 0


def _kill_process_tree(proc: subprocess.Popen) -> None:
    pid = proc.pid
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        subprocess.run(["pkill", "-KILL", "-P", str(pid)], capture_output=True, check=False)
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_desk_checked(
    desk_bin: str,
    argv: list[str],
    *,
    env: dict[str, str],
    log_path: pathlib.Path,
    step: str,
    timeout_s: float = DEFAULT_DESK_CMD_TIMEOUT_S,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_exec_cmd(desk_bin, argv, env=env)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out_txt, err_txt = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc)
        with log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"$ {' '.join(cmd)}\n[timeout after {timeout_s:.2f}s]\n")
        raise RefusalError(
            "DESK_COMMAND_TIMEOUT",
            f"dataset-desk {argv[0]} exceeded {timeout_s:.2f}s timeout",
            step=step,
        ) from exc

    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        if out_txt:
            fh.write(out_txt)
        if err_txt:
            fh.write(err_txt)
        fh.write(f"[rc={proc.returncode}]\n")
    if proc.returncode != 0:
        combined = f"{err_txt}\n{out_txt}".strip()
        m = re.search(r"error\s+([A-Z0-9_]+):\s*(.+)", combined)
        if m:
            raise RefusalError(m.group(1), m.group(2).strip(), step=step)
        raise RefusalError(
            "DESK_COMMAND_FAILED",
            f"dataset-desk {argv[0]} exited {proc.returncode}: {combined[:240]}",
            step=step,
        )


def _run_compose_with_heartbeat(
    desk_bin: str,
    argv: list[str],
    *,
    env: dict[str, str],
    hb_path: pathlib.Path,
    stall_s: float,
    model_load_stall_s: float = DEFAULT_MODEL_LOAD_STALL_S,
    poll_s: float,
    log_path: pathlib.Path,
    step: str,
) -> dict:
    if hb_path.exists():
        hb_path.unlink()
    jsonl_hb = pathlib.Path(f"{hb_path}.jsonl")
    if jsonl_hb.exists():
        jsonl_hb.unlink()

    cmd = _build_exec_cmd(desk_bin, argv, env=env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
        log_fh.write(f"$ {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh, text=True)
        last_progress_mono = time.monotonic()
        last_state: tuple[str, int] | None = None
        last_hb_doc: dict = {}

        while True:
            rc = proc.poll()
            hb_doc = None
            if jsonl_hb.is_file():
                try:
                    hb_lines = [ln for ln in jsonl_hb.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    if hb_lines:
                        hb_doc = json.loads(hb_lines[-1])
                except (OSError, json.JSONDecodeError):
                    hb_doc = None
            elif hb_path.is_file():
                try:
                    hb_doc = json.loads(hb_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    hb_doc = None
            if isinstance(hb_doc, dict):
                cur_state = (str(hb_doc.get("phase", "")), int(hb_doc.get("drafted", 0)))
                if cur_state != last_state:
                    last_state = cur_state
                    last_progress_mono = time.monotonic()
                    last_hb_doc = hb_doc
            if rc is not None:
                log_fh.write(f"[rc={rc}]\n")
                log_fh.flush()
                break
            phase_now = str(last_hb_doc.get("phase") or "startup")
            in_model_load = phase_now in ("startup", "loading_model")
            active_bound_s = model_load_stall_s if in_model_load else stall_s
            idle_s = time.monotonic() - last_progress_mono
            if idle_s > active_bound_s:
                _kill_process_tree(proc)
                drafted_now = last_hb_doc.get("drafted", 0)
                total_now = last_hb_doc.get("total", "?")
                if in_model_load:
                    raise RefusalError(
                        "MODEL_LOAD_STALLED",
                        f"compose model load stalled for {idle_s:.2f}s > {model_load_stall_s:.2f}s bound "
                        f"(phase={phase_now})",
                        step=step,
                    )
                raise RefusalError(
                    "DRAFT_HEARTBEAT_STALLED",
                    f"compose drafter stalled for {idle_s:.2f}s > {stall_s:.2f}s bound "
                    f"(phase={phase_now}, drafted={drafted_now}/{total_now})",
                    step=step,
                )
            time.sleep(poll_s)

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
        m = re.search(r"error\s+([A-Z0-9_]+):\s*(.+)", tail)
        if m:
            raise RefusalError(m.group(1), m.group(2).strip(), step=step)
        raise RefusalError("COMPOSE_FAILED", f"dataset-desk compose exited {proc.returncode}: {tail.strip()[:240]}", step=step)
    return {k: v for k, v in last_hb_doc.items() if k != "last_path"}


def cmd_run_compose_and_report(args: argparse.Namespace) -> int:
    state_dir = pathlib.Path(args.state_dir)
    out_root = pathlib.Path(args.desk_out)
    artifacts_dir = out_root / "desk_artifacts"
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    embedder_log = state_dir / "embedder_log.jsonl"
    hb_file = state_dir / "draft_heartbeat.json"
    desk_log = logs_dir / "desk_commands.log"

    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env.pop("DATASET_DESK_POOL_MANIFEST", None)
    env.pop("DATASET_DESK_POOL_DIRS", None)
    env["HF_HOME"] = str(pathlib.Path(args.hf_home))
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["DATASET_DESK_FORBID_EMBEDDERS"] = "1"
    env["DATASET_DESK_EMBEDDER_LOG_FILE"] = str(embedder_log)
    env["DATASET_DESK_DRAFT_HEARTBEAT_FILE"] = str(hb_file)
    env["DATASET_DESK_USER_CONFIG"] = ""

    desk_cmd_timeout_s = float(args.desk_cmd_timeout_s)
    record_vm_step(state_dir, "desk_index_report")
    _run_desk_checked(
        args.desk_bin,
        [
            "index",
            "--project",
            args.project_dir,
            "--caption-config",
            args.caption_config,
            "--backend",
            "none",
            "--out",
            str(artifacts_dir),
        ],
        env=env,
        log_path=desk_log,
        step="desk_index_report",
        timeout_s=desk_cmd_timeout_s,
    )
    _run_desk_checked(
        args.desk_bin,
        [
            "report",
            "--project",
            args.project_dir,
            "--caption-config",
            args.caption_config,
            "--backend",
            "none",
            "--draft",
            "none",
            "--out",
            str(artifacts_dir),
        ],
        env=env,
        log_path=desk_log,
        step="desk_index_report",
        timeout_s=desk_cmd_timeout_s,
    )

    record_vm_step(state_dir, "desk_compose")
    compose_argv = [
        "compose",
        "--project",
        args.project_dir,
        "--caption-config",
        args.caption_config,
        "--draft",
        args.drafter,
        "--out",
        str(artifacts_dir),
    ]
    if args.device != "cpu":
        compose_argv.extend(["--device", args.device, "--grant", args.grant_id, "--vram-budget", str(args.vram_budget_mib)])
    last_hb = _run_compose_with_heartbeat(
        args.desk_bin,
        compose_argv,
        env=env,
        hb_path=hb_file,
        stall_s=float(args.heartbeat_stall_s),
        model_load_stall_s=float(args.model_load_stall_s),
        poll_s=float(args.heartbeat_poll_s),
        log_path=desk_log,
        step="desk_compose",
    )

    record_vm_step(state_dir, "verify_vm_outputs")
    emb_events: list[dict] = []
    if embedder_log.is_file():
        for line in embedder_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                emb_events.append(json.loads(line))
    non_none = [ev for ev in emb_events if ev.get("backend") != "none"]
    if non_none:
        raise RefusalError(
            "EMBEDDING_BACKEND_FORBIDDEN",
            f"non-none embedding backend event recorded during run: {non_none[0]!r}",
            step="verify_vm_outputs",
        )

    review_path = artifacts_dir / "description-review.jsonl"
    if not review_path.is_file():
        raise RefusalError("OUTPUT_FILE_MISSING", "compose did not produce description-review.jsonl", step="verify_vm_outputs")
    review_rows = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    flagged_drafted = 0
    unflagged_count = 0
    for row in review_rows:
        if row.get("passed_model_free"):
            unflagged_count += 1
            if row.get("draft") or row.get("draft_offer") or row.get("drafter_raw_output"):
                raise RefusalError(
                    "UNFLAGGED_RECORD_DRAFTED",
                    f"unflagged record {row.get('id')!r} received a draft; drafts are strictly for gate-flagged records",
                    step="verify_vm_outputs",
                )
        elif row.get("draft"):
            flagged_drafted += 1

    raw_jsonl_path = artifacts_dir / "drafter-raw.jsonl"
    if not raw_jsonl_path.is_file():
        raw_jsonl_path.write_text("", encoding="utf-8", newline="\n")
    raw_entries = [json.loads(line) for line in raw_jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_outputs_doc = {
        "drafter": args.drafter,
        "entries": raw_entries,
        "flagged_drafted_count": flagged_drafted,
        "raw_outputs": {
            str(entry.get("asset_path") or entry.get("id")): entry.get("raw_output", "")
            for entry in raw_entries
        },
        "unflagged_untouched_count": unflagged_count,
    }
    write_json_lf(artifacts_dir / "raw_outputs.json", raw_outputs_doc)

    for rel_req in REQUIRED_DESK_ARTIFACTS:
        if not (out_root / rel_req).is_file():
            raise RefusalError("OUTPUT_FILE_MISSING", f"VM output missing required file {rel_req!r}", step="verify_vm_outputs")

    _sanitize_json_tree(out_root)
    _sanitize_json_tree(state_dir)

    out_files: list[dict] = []
    for fpath in sorted(out_root.rglob("*")):
        if not fpath.is_file():
            continue
        rel_posix = fpath.relative_to(out_root).as_posix()
        out_files.append(
            {
                "bytes": fpath.stat().st_size,
                "path": rel_posix,
                "sha256": sha256_file(fpath),
            }
        )
    vm_manifest = {
        "drafter": args.drafter,
        "embedding_backend_loaded": False,
        "files": out_files,
        "flagged_drafted_count": flagged_drafted,
        "unflagged_untouched_count": unflagged_count,
    }
    write_json_lf(state_dir / "vm_output_manifest.json", vm_manifest)

    vm_window = {
        "drafter": args.drafter,
        "embedder_events": emb_events,
        "embedding_backend_loaded": False,
        "flagged_drafted_count": flagged_drafted,
        "heartbeat_last": last_hb,
        "hf_hub_offline": env["HF_HUB_OFFLINE"] == "1",
        "mode": args.mode,
        "review_row_count": len(review_rows),
        "unflagged_untouched_count": unflagged_count,
    }
    write_json_lf(state_dir / "vm_window.json", vm_window)
    print(json.dumps(vm_window, sort_keys=True))
    return 0


PRIVATE_IMAGE_BUNDLE_RECORD = "private_image_bundle_refusal.json"


def _record_private_image_bundle_refusal(
    out_dir: pathlib.Path, bundle_path: pathlib.Path, member: str, why: str
) -> None:
    """Keep a sha + size + refusal record for a bundle that carried an image; the launcher's staging clean
    then deletes the bundle itself (it is the only copy of a private image left on the rig after a trip)."""
    rec = {
        "bundle_bytes": bundle_path.stat().st_size,
        "bundle_name": bundle_path.name,
        "bundle_sha256": sha256_file(bundle_path),
        "code": "PRIVATE_IMAGE_IN_BUNDLE",
        "member": member,
        "why": why,
    }
    write_json_lf(out_dir / PRIVATE_IMAGE_BUNDLE_RECORD, rec)


def cmd_extract_and_verify_outputs(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out_dir).resolve()
    bundle_path = pathlib.Path(args.bundle_path).resolve()
    if not bundle_path.is_file():
        raise RefusalError("OUTPUT_BUNDLE_MISSING", f"downloaded bundle missing at {str(bundle_path)!r}")

    stage_extract = out_dir / "_bundle_extract"
    if stage_extract.exists():
        shutil.rmtree(stage_extract)
    try:
        _safe_extract_tar(bundle_path, stage_extract, step="fetch_outputs")
        for extracted_f in sorted(stage_extract.rglob("*")):
            if not extracted_f.is_file():
                continue
            rel_member = extracted_f.relative_to(stage_extract).as_posix()
            with extracted_f.open("rb") as fh:
                magic_mime = detect_image_magic_bytes(fh.read(64))
            if magic_mime is not None:
                _record_private_image_bundle_refusal(out_dir, bundle_path, rel_member, f"raster image magic bytes ({magic_mime})")
                raise RefusalError(
                    "PRIVATE_IMAGE_IN_BUNDLE",
                    f"downloaded bundle member {rel_member!r} contains raster image magic bytes ({magic_mime})",
                )
            if extracted_f.suffix.lower() in IMAGE_EXTS:
                _record_private_image_bundle_refusal(
                    out_dir, bundle_path, rel_member, f"forbidden image extension {extracted_f.suffix.lower()!r}"
                )
                raise RefusalError(
                    "PRIVATE_IMAGE_IN_BUNDLE",
                    f"downloaded bundle member {rel_member!r} has forbidden image extension {extracted_f.suffix.lower()!r}",
                )

        outputs_dest = out_dir / "outputs"
        vm_state_dest = out_dir / "vm_state"
        if (stage_extract / "desk_out").is_dir():
            if outputs_dest.exists():
                shutil.rmtree(outputs_dest)
            shutil.copytree(stage_extract / "desk_out", outputs_dest)
        if (stage_extract / "desk_state").is_dir():
            if vm_state_dest.exists():
                shutil.rmtree(vm_state_dest)
            shutil.copytree(stage_extract / "desk_state", vm_state_dest)
    finally:
        shutil.rmtree(stage_extract, ignore_errors=True)

    if args.salvage_only:
        return 0

    vm_manifest_path = vm_state_dest / "vm_output_manifest.json"
    if not vm_manifest_path.is_file():
        raise RefusalError("OUTPUT_MANIFEST_MISSING", "VM output manifest vm_output_manifest.json missing from bundle")
    vm_manifest = json.loads(vm_manifest_path.read_text(encoding="utf-8"))
    entries = vm_manifest.get("files", [])
    by_path = {str(item["path"]): item for item in entries if isinstance(item, dict) and "path" in item}

    for req_rel in REQUIRED_DESK_ARTIFACTS:
        if req_rel not in by_path:
            raise RefusalError("OUTPUT_FILE_MISSING", f"required output {req_rel!r} missing from VM output manifest")
        local_req = outputs_dest / req_rel
        if not local_req.is_file():
            raise RefusalError("OUTPUT_FILE_MISSING", f"required output {req_rel!r} missing after fetch")

    verified_count = 0
    for rel_posix, meta in sorted(by_path.items()):
        local_f = outputs_dest / rel_posix
        if not local_f.is_file():
            raise RefusalError("OUTPUT_FILE_MISSING", f"fetched output missing file {rel_posix!r} listed in VM manifest")
        actual_bytes = local_f.stat().st_size
        actual_sha = sha256_file(local_f)
        if actual_bytes != meta.get("bytes") or actual_sha != meta.get("sha256"):
            raise RefusalError(
                "OUTPUT_MANIFEST_SHA_MISMATCH",
                f"fetched output {rel_posix!r} sha/bytes ({actual_sha}, {actual_bytes}) != "
                f"VM manifest ({meta.get('sha256')}, {meta.get('bytes')})",
            )
        verified_count += 1

    ver_doc = {
        "ok": True,
        "required_artifacts": list(REQUIRED_DESK_ARTIFACTS),
        "verified_file_count": verified_count,
    }
    write_json_lf(out_dir / "output_verification.json", ver_doc)
    print(json.dumps(ver_doc, sort_keys=True))
    return 0


def cmd_vm_cleanup_workdir(args: argparse.Namespace) -> int:
    if os.environ.get("DESK_VM_SIMULATE_CLEANUP_FAIL", "").strip() == "1":
        print("REFUSED: VM_WORKDIR_CLEANUP_FAILED: simulated VM workdir cleanup failure", file=sys.stderr)
        return 11

    content_root = pathlib.Path(args.content_root)
    targets = [
        content_root / "project_input.tar",
        content_root / "project_manifest.json",
        content_root / "caption_config.json",
        content_root / "desk_work",
        content_root / "desk_out",
        content_root / "desk_state",
        content_root / "desk_draft_bundle.tar.gz",
        content_root / "desk_hf_token",
    ]
    if content_root.is_dir():
        targets.extend(sorted(content_root.glob(".sdlab_up_*")))
        targets.extend(sorted(content_root.glob(".sdlab_dl_*")))

    errors: list[str] = []
    for t in targets:
        try:
            if t.is_symlink() or t.is_file():
                t.unlink()
            elif t.is_dir():
                shutil.rmtree(t)
        except OSError as exc:
            errors.append(f"{t.as_posix()}: {exc}")

    survivors = [t.as_posix() for t in targets if t.exists()]
    if errors or survivors:
        why = "; ".join(errors + [f"still_exists:{s}" for s in survivors])
        print(f"REFUSED: VM_WORKDIR_CLEANUP_FAILED: {why}", file=sys.stderr)
        return 12

    print(json.dumps({"cleaned": True, "survivors": []}, sort_keys=True))
    return 0


def _scan_logs_for_refusal(out_dir: pathlib.Path) -> tuple[str | None, str | None, str | None]:
    remote_fail = out_dir / "vm_state" / "remote_failure.json"
    if remote_fail.is_file():
        try:
            rf = json.loads(remote_fail.read_text(encoding="utf-8"))
            if rf.get("refusal_code") and rf.get("why"):
                return str(rf["refusal_code"]), str(rf["why"]), str(rf.get("step") or "")
        except (OSError, json.JSONDecodeError):
            pass

    for log_name in ("guards.log", "vitals.log", "run_remote.log", "setup.log", "colab.log"):
        lp = out_dir / log_name
        if not lp.is_file():
            continue
        in_proc = False
        for raw_line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
            if log_name == "vitals.log":
                if raw_line.startswith("PROC:"):
                    in_proc = True
                    continue
                if in_proc:
                    if raw_line.startswith("HARNESS:"):
                        in_proc = False
                    continue
            sm = STEP_MARKER_RE.search(raw_line)
            if sm:
                return sm.group(2).strip(), sm.group(3).strip(), sm.group(1).strip()
            rm = REFUSED_LINE_RE.search(raw_line)
            if rm:
                return rm.group(1).strip(), rm.group(2).strip(), None
    return None, None, None


def _classify_colab_open_error(out_dir: pathlib.Path) -> tuple[str, str, bool, str] | None:
    for log_name in ("colab.log", "setup.log"):
        lp = out_dir / log_name
        if not lp.is_file():
            continue
        txt = lp.read_text(encoding="utf-8", errors="replace")
        if HTTP_412_RE.search(txt):
            return (
                "COLAB_ACCOUNT_CAP_412",
                "Colab returned HTTP 412 (account/session cap reached); never retry automatically",
                False,
                "never_retry",
            )
        if HTTP_503_RE.search(txt):
            return (
                "COLAB_POOL_EMPTY_503",
                "Colab returned HTTP 503 (accelerator pool empty); retry later",
                True,
                "retry_later",
            )
    return None


def _read_optional_json(path: pathlib.Path) -> dict:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _basename_only(raw_val: object) -> str:
    return pathlib.PurePath(str(raw_val).replace("\\", "/")).name


def _sanitize_transfer_record_dict(rec: dict) -> dict:
    cleaned = dict(rec)
    for k in ("local", "src"):
        if isinstance(cleaned.get(k), str) and cleaned[k]:
            cleaned[k] = _basename_only(cleaned[k])
    return cleaned


def _sanitize_window_host_paths(out_dir: pathlib.Path, win: dict, outcome: dict) -> None:
    receipts_p = out_dir / "upload_receipts.jsonl"
    if receipts_p.is_file():
        cleaned_lines: list[str] = []
        for raw_ln in receipts_p.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw_ln.strip():
                continue
            try:
                rec = json.loads(raw_ln)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                rec = _sanitize_transfer_record_dict(rec)
            cleaned_lines.append(json.dumps(rec, sort_keys=True))
        receipts_p.write_text("\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""), encoding="utf-8", newline="\n")

    for bd_name in ("bundle_download.json", "salvage_bundle_download.json"):
        bd_p = out_dir / bd_name
        if bd_p.is_file():
            bd_doc = _read_optional_json(bd_p)
            if bd_doc:
                write_json_lf(bd_p, _sanitize_transfer_record_dict(bd_doc))

    for doc in (win, outcome):
        if isinstance(doc.get("upload_transfers"), list):
            doc["upload_transfers"] = [
                _sanitize_transfer_record_dict(u) if isinstance(u, dict) else u for u in doc["upload_transfers"]
            ]
        for bd_key in ("bundle_download", "salvage_bundle_download"):
            if isinstance(doc.get(bd_key), dict):
                doc[bd_key] = _sanitize_transfer_record_dict(doc[bd_key])
        if isinstance(doc.get("input_shas"), dict):
            doc["input_shas"] = {_basename_only(k): v for k, v in sorted(doc["input_shas"].items(), key=lambda kv: _basename_only(kv[0]))}


def cmd_finalize_window_record(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out_dir).resolve()
    win_path = out_dir / "window.json"
    outcome_path = out_dir / "outcome.draft.json"
    legacy_outcome_path = out_dir / "outcome.train.json"

    win = _read_optional_json(win_path)
    outcome = _read_optional_json(outcome_path) or _read_optional_json(legacy_outcome_path)
    _sanitize_window_host_paths(out_dir, win, outcome)

    proj_manifest = _read_optional_json(out_dir / "project_manifest.json")
    wheel_manifest = _read_optional_json(out_dir / "wheel_manifest.json")
    vm_probe = _read_optional_json(out_dir / "vm_state" / "vm_probe.json")
    weight_dl = _read_optional_json(out_dir / "vm_state" / "weight_download.json")
    cfg_load = _read_optional_json(out_dir / "vm_state" / "config_load.json")
    vm_win = _read_optional_json(out_dir / "vm_state" / "vm_window.json")
    out_ver = _read_optional_json(out_dir / "output_verification.json")
    cleanup_doc = _read_optional_json(out_dir / "vm_cleanup_status.json")

    last_step = None
    step_file = out_dir / "vm_state" / "remote_step.txt"
    if step_file.is_file():
        last_step = step_file.read_text(encoding="utf-8", errors="replace").strip() or None
    else:
        for cand_log in ("vitals.log", "run_remote.log"):
            lp = out_dir / cand_log
            if lp.is_file():
                for line in lp.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("REMOTE_STEP step="):
                        last_step = line.split("=", 1)[1].strip()

    exit_code = int(args.exit_code)
    refusal_code = win.get("refusal_code")
    why = win.get("why")
    retryable: bool | None = None
    retry_disposition: str | None = None

    http_class = _classify_colab_open_error(out_dir)
    scan_code, scan_why, scan_step = _scan_logs_for_refusal(out_dir)
    if scan_step:
        last_step = scan_step

    if http_class is not None and (
        not refusal_code or refusal_code in ("SESSION_OPEN_FAILED", "COLAB_SESSION_OPEN_FAILED")
    ):
        refusal_code, why, retryable, retry_disposition = http_class
    elif scan_code and (
        not refusal_code or refusal_code in ("SESSION_OPEN_FAILED", "COLAB_SESSION_OPEN_FAILED", "REMOTE_STEP_FAILED")
    ):
        refusal_code = scan_code
        why = scan_why

    cleanup_attempted = bool(cleanup_doc.get("attempted", False))
    cleanup_rc = cleanup_doc.get("exit_status")
    cleanup_ok = bool(cleanup_doc.get("ok", False)) if cleanup_attempted else False
    if cleanup_attempted and not cleanup_ok:
        exit_code = exit_code or 12
        if not refusal_code:
            refusal_code = "VM_WORKDIR_CLEANUP_FAILED"
            why = str(cleanup_doc.get("why") or f"VM workdir cleanup failed with exit status {cleanup_rc}")

    if exit_code != 0:
        if not refusal_code:
            refusal_code = "REMOTE_STEP_FAILED"
        if not why:
            step_lbl = last_step or "pre_vm_bootstrap"
            why = f"step {step_lbl!r} exited with status {exit_code} before emitting a refusal reason"

    status = "completed" if (exit_code == 0 and not refusal_code) else ("refused" if refusal_code else "failed")
    if status != "completed":
        assert refusal_code is not None and why is not None

    if retryable is None:
        retryable = refusal_code in ("COLAB_POOL_EMPTY_503", "DOWNLOAD_429_BOUND_EXCEEDED")
    if retry_disposition is None:
        if refusal_code == "COLAB_ACCOUNT_CAP_412":
            retry_disposition = "never_retry"
        elif retryable:
            retry_disposition = "retry_later"
        else:
            retry_disposition = "never_retry" if status != "completed" else "none"

    if isinstance(win.get("heartbeat_last"), dict):
        win["heartbeat_last"] = {k: v for k, v in win["heartbeat_last"].items() if k != "last_path"}

    stop_timeout_s = int(os.environ.get("COLAB_STOP_TIMEOUT_S", "60"))
    kill_grace_s = int(os.environ.get("COLAB_KILL_GRACE_S", "10"))
    desk_vm_cleanup_bound_s = stop_timeout_s + kill_grace_s
    env_stop_by = os.environ.get("COLAB_CLEANUP_STOP_ATTEMPTED_BY_S", "").strip()
    env_bound = os.environ.get("COLAB_CLEANUP_BOUND_S", "").strip()
    if env_stop_by.isdigit() and env_bound.isdigit():
        cleanup_stop_attempted_by_s = int(env_stop_by)
        cleanup_bound_s = int(env_bound)
    else:
        base_stop_by = int(win.get("cleanup_stop_attempted_by_s") or 1150)
        base_bound = int(win.get("cleanup_bound_s") or 1290)
        cleanup_stop_attempted_by_s = base_stop_by + desk_vm_cleanup_bound_s
        cleanup_bound_s = base_bound + desk_vm_cleanup_bound_s

    priced_rate = win.get("rate_used_for_cap_and_projection")
    if priced_rate is None and (out_dir / "max_rate.txt").is_file():
        try:
            priced_rate = float((out_dir / "max_rate.txt").read_text(encoding="utf-8").strip())
        except ValueError:
            priced_rate = None
    if priced_rate is not None and win.get("cap") is not None and win.get("margin") is not None:
        win["cleanup_reserve_units"] = round((cleanup_bound_s / 3600.0) * float(priced_rate), 6)
        win["effective_trip_units"] = round(float(win["cap"]) - float(win["margin"]) - win["cleanup_reserve_units"], 6)

    win.update(
        {
            "cleanup_bound_s": cleanup_bound_s,
            "cleanup_stop_attempted_by_s": cleanup_stop_attempted_by_s,
            "colab_runtime_version": vm_probe.get("colab_runtime_version"),
            "config_only_load": cfg_load or None,
            "desk_vm_cleanup_bound_s": desk_vm_cleanup_bound_s,
            "desk_wheel": {
                "git_commit": wheel_manifest.get("git_commit"),
                "sha256": wheel_manifest.get("sha256"),
                "wheel_name": wheel_manifest.get("wheel_name"),
            }
            if wheel_manifest
            else None,
            "drafter": args.drafter,
            "embedding_backend_loaded": bool(vm_win.get("embedding_backend_loaded", False)),
            "exit_code": exit_code,
            "lane": "draft",
            "last_remote_step": last_step,
            "mode": args.mode,
            "multimodal_pool": proj_manifest.get("multimodal_pool"),
            "multimodal_pool_locked": bool(proj_manifest.get("multimodal_pool_locked", True)),
            "output_manifest_verified": bool(out_ver.get("ok", False)),
            "private_images": {
                "archive_bytes": proj_manifest.get("archive_bytes"),
                "archive_sha256": proj_manifest.get("archive_sha256"),
                "delivery_method": "colab_upload_checked",
                "external_urls_allowed": False,
                "uploaded_file_count": proj_manifest.get("uploaded_file_count"),
                "uploaded_total_bytes": proj_manifest.get("uploaded_total_bytes"),
                "vm_workdir_cleanup_attempted": cleanup_attempted,
                "vm_workdir_cleanup_exit_status": cleanup_rc,
                "vm_workdir_deleted": cleanup_ok,
            },
            "qwen3_vl_weights": weight_dl or None,
            "refusal_code": refusal_code,
            "retry_disposition": retry_disposition,
            "retryable": retryable,
            "sdlab_vendored_commit": SDLAB_VENDORED_COMMIT,
            "status": status,
            "usr_local_cuda": vm_probe.get("usr_local_cuda"),
            "vm_probe": vm_probe or None,
            "vm_workdir_cleanup": {
                "attempted": cleanup_attempted,
                "exit_status": cleanup_rc,
                "ok": cleanup_ok,
                "why": cleanup_doc.get("why"),
            },
            "why": why,
        }
    )
    write_json_lf(win_path, win)

    outcome.update(
        {
            "exit_code": exit_code,
            "lane": "draft",
            "last_remote_step": last_step,
            "multimodal_pool": proj_manifest.get("multimodal_pool"),
            "output_manifest_verified": bool(out_ver.get("ok", False)),
            "refusal_code": refusal_code,
            "retry_disposition": retry_disposition,
            "retryable": retryable,
            "sdlab_vendored_commit": SDLAB_VENDORED_COMMIT,
            "status": status,
            "vm_workdir_cleanup_exit_status": cleanup_rc,
            "vm_workdir_deleted": cleanup_ok,
            "why": why,
        }
    )
    write_json_lf(outcome_path, outcome)
    if legacy_outcome_path.is_file():
        legacy_outcome_path.unlink(missing_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="dataset-desk Colab drafting lane helper")
    sub = ap.add_subparsers(dest="subcmd", required=True)

    p_stub = sub.add_parser("make-stub-project")
    p_stub.add_argument("--out-dir", required=True)
    p_stub.set_defaults(func=cmd_make_stub_project)

    p_pre = sub.add_parser("preflight-project")
    p_pre.add_argument("--repo-root", required=True)
    p_pre.add_argument("--project", required=True)
    p_pre.add_argument("--caption-config", required=True)
    p_pre.add_argument("--out-dir", required=True)
    p_pre.add_argument("--isolate-user-config", action="store_true")
    p_pre.set_defaults(func=cmd_preflight_project)

    p_whl = sub.add_parser("build-wheel")
    p_whl.add_argument("--repo-root", required=True)
    p_whl.add_argument("--out-dir", required=True)
    p_whl.add_argument("--git-commit", required=True)
    p_whl.set_defaults(func=cmd_build_wheel)

    p_step = sub.add_parser("record-step")
    p_step.add_argument("--state-dir", default="/content/desk_state")
    p_step.add_argument("--step", required=True)
    p_step.set_defaults(func=lambda a: (record_vm_step(pathlib.Path(a.state_dir), a.step) or 0))

    p_fail = sub.add_parser("record-failure")
    p_fail.add_argument("--state-dir", default="/content/desk_state")
    p_fail.add_argument("--step", required=True)
    p_fail.add_argument("--code", required=True)
    p_fail.add_argument("--why", required=True)
    p_fail.add_argument("--exit-code", type=int, default=2)
    p_fail.set_defaults(
        func=lambda a: (
            record_vm_failure(
                pathlib.Path(a.state_dir),
                step=a.step,
                code=a.code,
                why=a.why,
                exit_code=a.exit_code,
            )
            or 0
        )
    )

    p_probe = sub.add_parser("vm-probe-and-guards")
    p_probe.add_argument("--state-dir", default="/content/desk_state")
    p_probe.add_argument("--mode", choices=("gpu", "stub"), required=True)
    p_probe.add_argument("--card", required=True)
    p_probe.add_argument("--work-root", default="/content")
    p_probe.add_argument("--min-disk-gib", type=float, default=35.0)
    p_probe.add_argument("--min-ram-gib", type=float, default=12.0)
    p_probe.set_defaults(func=cmd_vm_probe_and_guards)

    p_vwhl = sub.add_parser("verify-uploaded-wheel")
    p_vwhl.add_argument("--state-dir", default="/content/desk_state")
    p_vwhl.add_argument("--wheel-path", required=True)
    p_vwhl.add_argument("--expected-wheel-sha", required=True)
    p_vwhl.add_argument("--lock-path", required=True)
    p_vwhl.add_argument("--expected-lock-sha", required=True)
    p_vwhl.set_defaults(func=cmd_verify_uploaded_wheel)

    p_dl = sub.add_parser("download-weights")
    p_dl.add_argument("--state-dir", default="/content/desk_state")
    p_dl.add_argument("--hf-home", default="/content/hf_home")
    p_dl.add_argument("--repo-id", default=QWEN3_VL_REPO)
    p_dl.add_argument("--revision", default=QWEN3_VL_REVISION)
    p_dl.add_argument("--token-file", default="")
    p_dl.add_argument("--download-bound-s", type=float, default=600.0)
    p_dl.add_argument("--backoff-base-s", type=float, default=1.0)
    p_dl.add_argument("--max-attempts", type=int, default=6)
    p_dl.set_defaults(func=cmd_download_weights)

    p_cfg = sub.add_parser("config-only-load")
    p_cfg.add_argument("--state-dir", default="/content/desk_state")
    p_cfg.add_argument("--hf-home", default="/content/hf_home")
    p_cfg.add_argument("--repo-id", default=QWEN3_VL_REPO)
    p_cfg.add_argument("--revision", default=QWEN3_VL_REVISION)
    p_cfg.set_defaults(func=cmd_config_only_load)

    p_unp = sub.add_parser("unpack-project")
    p_unp.add_argument("--state-dir", default="/content/desk_state")
    p_unp.add_argument("--tar-path", default="/content/project_input.tar")
    p_unp.add_argument("--manifest-path", default="/content/project_manifest.json")
    p_unp.add_argument("--caption-config-path", default="/content/caption_config.json")
    p_unp.add_argument("--dest-dir", default="/content/desk_work/project")
    p_unp.set_defaults(func=cmd_unpack_project)

    p_run = sub.add_parser("run-compose-and-report")
    p_run.add_argument("--state-dir", default="/content/desk_state")
    p_run.add_argument("--desk-out", default="/content/desk_out")
    p_run.add_argument("--hf-home", default="/content/hf_home")
    p_run.add_argument("--desk-bin", required=True)
    p_run.add_argument("--project-dir", default="/content/desk_work/project")
    p_run.add_argument("--caption-config", default="/content/caption_config.json")
    p_run.add_argument("--drafter", choices=("qwen3-vl", "fake"), required=True)
    p_run.add_argument("--mode", choices=("gpu", "stub"), required=True)
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--grant-id", default="")
    p_run.add_argument("--vram-budget-mib", type=int, default=22000)
    p_run.add_argument("--heartbeat-stall-s", type=float, default=DEFAULT_HEARTBEAT_STALL_S)
    p_run.add_argument("--model-load-stall-s", type=float, default=DEFAULT_MODEL_LOAD_STALL_S)
    p_run.add_argument("--desk-cmd-timeout-s", type=float, default=DEFAULT_DESK_CMD_TIMEOUT_S)
    p_run.add_argument("--heartbeat-poll-s", type=float, default=0.2)
    p_run.set_defaults(func=cmd_run_compose_and_report)

    p_ver = sub.add_parser("extract-and-verify-outputs")
    p_ver.add_argument("--out-dir", required=True)
    p_ver.add_argument("--bundle-path", required=True)
    p_ver.add_argument("--salvage-only", action="store_true")
    p_ver.set_defaults(func=cmd_extract_and_verify_outputs)

    p_cln = sub.add_parser("vm-cleanup-workdir")
    p_cln.add_argument("--content-root", default="/content")
    p_cln.set_defaults(func=cmd_vm_cleanup_workdir)

    p_chk = sub.add_parser("check-out-dir-scrubbed")
    p_chk.add_argument("--out-dir", required=True)
    p_chk.set_defaults(func=cmd_check_out_dir_scrubbed)

    p_sw = sub.add_parser("sweep-unscrubbed")
    p_sw.add_argument("--parent", required=True)
    p_sw.add_argument("--exclude", default="")
    p_sw.set_defaults(func=cmd_sweep_unscrubbed)

    p_scr = sub.add_parser("scrub-kept-paths")
    p_scr.add_argument("--out-dir", required=True)
    p_scr.add_argument("--root", action="append", default=[], help="<abs path>=<placeholder>; repeatable")
    p_scr.set_defaults(func=cmd_scrub_kept_paths)

    p_fin = sub.add_parser("finalize-window-record")
    p_fin.add_argument("--out-dir", required=True)
    p_fin.add_argument("--exit-code", type=int, required=True)
    p_fin.add_argument("--mode", required=True)
    p_fin.add_argument("--drafter", required=True)
    p_fin.set_defaults(func=cmd_finalize_window_record)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    state_dir_raw = getattr(args, "state_dir", None)
    try:
        return int(args.func(args))
    except RefusalError as ref:
        if state_dir_raw:
            step_name = ref.step or "helper"
            record_vm_failure(
                pathlib.Path(state_dir_raw),
                step=step_name,
                code=ref.code,
                why=ref.why,
                exit_code=2,
                retryable=ref.retryable,
            )
        else:
            print(f"REFUSED: {ref.code}: {ref.why}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
