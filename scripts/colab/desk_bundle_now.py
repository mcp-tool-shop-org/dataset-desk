#!/usr/bin/env python3
"""Bundle desk draft VM outputs and diagnostic state into /content/desk_draft_bundle.tar.gz.

Never bundles private project images (/content/project_input.tar or /content/desk_work/project),
refuses disguised images by magic-byte content inspection, and strips absolute paths and private
asset filenames from staged state/manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tarfile
import tempfile
import time

SCRUB_EXTS = {".log", ".txt", ".md", ".json", ".jsonl"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class BundleRefusalError(Exception):
    def __init__(self, code: str, why: str) -> None:
        super().__init__(f"REFUSED: {code}: {why}")
        self.code = code
        self.why = why


def detect_image_magic_bytes(data: bytes) -> str | None:
    """Return MIME type if `data` begins with a raster image header (PNG/JPEG/GIF/WebP/TIFF/BMP)."""
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
    return None


def sanitize_pool_detection(pd: object) -> object:
    """Keep pool names and integer counts only; strip all host/VM `sources`, `path`, and `config_path`."""
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


def _sanitize_json_obj(obj: object, *, file_name: str) -> object:
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
                out[k] = _sanitize_json_obj(v, file_name=file_name)
        return out
    if isinstance(obj, list):
        return [_sanitize_json_obj(item, file_name=file_name) for item in obj]
    return obj


def _sanitize_staged_json_files(root: pathlib.Path) -> None:
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".json":
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            cleaned = _sanitize_json_obj(doc, file_name=p.name)
            if cleaned != doc:
                p.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
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
                cleaned_row = _sanitize_json_obj(row, file_name=p.name)
                if cleaned_row != row:
                    changed = True
                    out_lines.append(json.dumps(cleaned_row, sort_keys=True))
                else:
                    out_lines.append(ln)
            if changed:
                p.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8", newline="\n")


def _verify_no_images_in_stage(stage_dir: pathlib.Path) -> None:
    for fpath in sorted(stage_dir.rglob("*")):
        if not fpath.is_file():
            continue
        rel_posix = fpath.relative_to(stage_dir).as_posix()
        with fpath.open("rb") as fh:
            header = fh.read(64)
        mime = detect_image_magic_bytes(header)
        if mime is not None:
            raise BundleRefusalError(
                "PRIVATE_IMAGE_IN_BUNDLE",
                f"staged bundle member {rel_posix!r} carries raster image magic bytes ({mime}); private images are forbidden in output bundles",
            )
        if fpath.suffix.lower() in IMAGE_EXTS:
            raise BundleRefusalError(
                "PRIVATE_IMAGE_IN_BUNDLE",
                f"staged bundle member {rel_posix!r} has forbidden image extension {fpath.suffix.lower()!r}",
            )


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scrub_tree(root: pathlib.Path, guards_path: pathlib.Path) -> None:
    if not guards_path.is_file():
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("launch_guards", str(guards_path))
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scrub_fn = getattr(mod, "scrub_colab_tokens", None)
    if not callable(scrub_fn):
        return
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SCRUB_EXTS:
            continue
        try:
            raw = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        cleaned = scrub_fn(raw)
        if cleaned != raw:
            p.write_text(cleaned, encoding="utf-8", newline="\n")


def build_bundle(
    *,
    desk_out: pathlib.Path,
    desk_state: pathlib.Path,
    bundle_path: pathlib.Path,
    guards_path: pathlib.Path,
) -> dict:
    stage_dir = pathlib.Path(tempfile.mkdtemp(prefix="desk_bundle_stage_"))
    try:
        if desk_out.is_dir():
            for src in sorted(desk_out.rglob("*")):
                if not src.is_file():
                    continue
                rel = src.relative_to(desk_out)
                dst = stage_dir / "desk_out" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        if desk_state.is_dir():
            for src in sorted(desk_state.rglob("*")):
                if not src.is_file():
                    continue
                rel = src.relative_to(desk_state)
                dst = stage_dir / "desk_state" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        _verify_no_images_in_stage(stage_dir)
        _sanitize_staged_json_files(stage_dir)
        _scrub_tree(stage_dir, guards_path)

        vm_manifest_file = stage_dir / "desk_state" / "vm_output_manifest.json"
        stage_desk_out = stage_dir / "desk_out"
        if vm_manifest_file.is_file() and stage_desk_out.is_dir():
            try:
                vm_manifest_doc = json.loads(vm_manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                vm_manifest_doc = None
            if isinstance(vm_manifest_doc, dict):
                out_files: list[dict] = []
                for fpath in sorted(stage_desk_out.rglob("*")):
                    if not fpath.is_file():
                        continue
                    rel_posix = fpath.relative_to(stage_desk_out).as_posix()
                    out_files.append(
                        {
                            "bytes": fpath.stat().st_size,
                            "path": rel_posix,
                            "sha256": _sha256_file(fpath),
                        }
                    )
                vm_manifest_doc["files"] = out_files
                vm_manifest_file.write_text(
                    json.dumps(vm_manifest_doc, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )

        files_meta: list[dict] = []
        for fpath in sorted(stage_dir.rglob("*")):
            if not fpath.is_file():
                continue
            rel_posix = fpath.relative_to(stage_dir).as_posix()
            files_meta.append(
                {
                    "path": rel_posix,
                    "bytes": fpath.stat().st_size,
                    "sha256": _sha256_file(fpath),
                }
            )
        manifest_doc = {"files": files_meta}
        manifest_file = stage_dir / "bundle_manifest.json"
        manifest_file.write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "w:gz", compresslevel=6) as tf:
            for fpath in sorted(stage_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                rel_posix = fpath.relative_to(stage_dir).as_posix()
                ti = tarfile.TarInfo(name=rel_posix)
                st = fpath.stat()
                ti.size = st.st_size
                ti.mtime = 0
                ti.mode = 0o644
                ti.uid = 0
                ti.gid = 0
                ti.uname = ""
                ti.gname = ""
                with fpath.open("rb") as fh:
                    tf.addfile(ti, fh)
        return {
            "bundle_bytes": bundle_path.stat().st_size,
            "bundle_name": bundle_path.name,
            "bundle_sha256": _sha256_file(bundle_path),
            "file_count": len(files_meta),
        }
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    content_root = os.environ.get("SDLAB_CONTENT", "/content")
    ap = argparse.ArgumentParser(description="Bundle desk draft outputs and state without private images")
    ap.add_argument("--desk-out", default=os.environ.get("DESK_VM_OUT_DIR", f"{content_root}/desk_out"))
    ap.add_argument("--desk-state", default=os.environ.get("DESK_VM_STATE_DIR", f"{content_root}/desk_state"))
    ap.add_argument(
        "--bundle-path",
        default=os.environ.get("DESK_VM_BUNDLE_PATH", f"{content_root}/desk_draft_bundle.tar.gz"),
    )
    ap.add_argument("--guards-path", default=os.environ.get("DESK_VM_GUARDS_PATH", f"{content_root}/launch_guards.py"))
    args = ap.parse_args(argv)
    desk_state = pathlib.Path(args.desk_state)
    bundle_path = pathlib.Path(args.bundle_path)
    try:
        info = build_bundle(
            desk_out=pathlib.Path(args.desk_out),
            desk_state=desk_state,
            bundle_path=bundle_path,
            guards_path=pathlib.Path(args.guards_path),
        )
    except BundleRefusalError as ref:
        bundle_path.unlink(missing_ok=True)
        clean_why = " ".join(ref.why.splitlines()).strip()
        if desk_state.is_dir():
            fail_doc = {
                "exit_code": 2,
                "refusal_code": ref.code,
                "retryable": False,
                "step": "bundle_outputs",
                "ts": round(time.time(), 6),
                "why": clean_why,
            }
            (desk_state / "remote_failure.json").write_text(
                json.dumps(fail_doc, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        print(f"REFUSED: {ref.code}: {clean_why}", file=sys.stderr, flush=True)
        print(f"REMOTE_STEP_FAILED_MARKER step=bundle_outputs code={ref.code} why={clean_why}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
