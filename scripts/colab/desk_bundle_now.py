#!/usr/bin/env python3
"""Bundle desk draft VM outputs and diagnostic state into /content/desk_draft_bundle.tar.gz.

Never bundles private project images (/content/project_input.tar or /content/desk_work/project).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import tarfile
import tempfile

SCRUB_EXTS = {".log", ".txt", ".md", ".json", ".jsonl"}


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
            "bundle_path": str(bundle_path),
            "bundle_bytes": bundle_path.stat().st_size,
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
    info = build_bundle(
        desk_out=pathlib.Path(args.desk_out),
        desk_state=pathlib.Path(args.desk_state),
        bundle_path=pathlib.Path(args.bundle_path),
        guards_path=pathlib.Path(args.guards_path),
    )
    print(json.dumps(info, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
