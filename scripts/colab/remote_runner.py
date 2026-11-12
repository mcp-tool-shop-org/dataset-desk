#!/usr/bin/env python3
"""Remote kernel runner executed via `colab exec -s <session> -f remote_runner.py`.

Runs the uploaded remote shell script (`SDLAB_REMOTE_SCRIPT`, e.g.
`/content/remote_lane1_stub.sh` or `/content/remote_lane3_stub.sh`) via
`subprocess.Popen`, streaming each stdout/stderr line to the IPython kernel's
stdout as it arrives so the rig's vitals and DEAD HARNESS watcher read it in
real time, and finishes by printing a `SDLAB_REMOTE_RC=<rc>` marker line.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

# Exec outputs land in google-colab-cli's history (docs/colab-cli-contract.md, section 3), so no line
# that could carry a Hugging Face token is relayed as it is.
_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{16,}\b")


def scrub_line(line: str) -> str:
    return _HF_TOKEN_RE.sub("hf_[REDACTED]", line)


def _resolve_bash(env: dict[str, str]) -> str:
    git_bash = "C:/Program Files/Git/bin/bash.exe"
    if os.name == "nt" and os.path.exists(git_bash):
        return git_bash
    found = shutil.which("bash", path=env.get("PATH"))
    return found or "bash"


def prepare_content_downloads(content_root: str | Path, chunk_bytes: int) -> list[dict[str, object]]:
    """Record whole-file sha256 (and split into `chunk_bytes` parts + manifest when larger) for any
    `*.tar.gz` bundle in `content_root` inside the single remote `colab exec` cell."""
    root = Path(content_root)
    records: list[dict[str, object]] = []
    if not root.is_dir() or chunk_bytes <= 0:
        return records
    for vm_file in sorted(root.iterdir()):
        if not vm_file.is_file() or vm_file.name.startswith(".sdlab_") or not vm_file.name.endswith(".tar.gz"):
            continue
        rel = vm_file.name
        size = vm_file.stat().st_size
        if size <= chunk_bytes:
            h = hashlib.sha256()
            with vm_file.open("rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            rec: dict[str, object] = {
                "rel": rel,
                "exists": True,
                "split": False,
                "whole_bytes": size,
                "whole_sha256": h.hexdigest(),
                "chunk_bytes": chunk_bytes,
                "part_count": 1,
            }
            print("SDLAB_DOWNLOAD_PREPARE_JSON " + json.dumps(rec), flush=True)
            records.append(rec)
            continue

        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", rel)[:48]
        prefix = f".sdlab_dl_{safe_name}"
        for stale in root.iterdir():
            if stale.name.startswith(prefix + ".") and stale.is_file():
                stale.unlink(missing_ok=True)
        whole_h = hashlib.sha256()
        parts: list[dict[str, object]] = []
        idx = 0
        with vm_file.open("rb") as fh:
            while True:
                remaining = chunk_bytes
                part_name = f"{prefix}.part.{idx:04d}"
                part_path = root / part_name
                part_h = hashlib.sha256()
                part_b = 0
                first_blk = fh.read(min(1 << 20, remaining))
                if not first_blk:
                    break
                with part_path.open("wb") as pfh:
                    pfh.write(first_blk)
                    part_h.update(first_blk)
                    whole_h.update(first_blk)
                    part_b += len(first_blk)
                    remaining -= len(first_blk)
                    while remaining > 0:
                        blk = fh.read(min(1 << 20, remaining))
                        if not blk:
                            break
                        pfh.write(blk)
                        part_h.update(blk)
                        whole_h.update(blk)
                        part_b += len(blk)
                        remaining -= len(blk)
                parts.append({
                    "index": idx,
                    "remote_path": f"/content/{part_name}",
                    "bytes": part_b,
                    "sha256": part_h.hexdigest(),
                })
                idx += 1
        whole_sha = whole_h.hexdigest()
        manifest_name = f"{prefix}.manifest.json"
        manifest_payload = {
            "remote": f"/content/{rel}",
            "prefix": prefix,
            "split": True,
            "whole_bytes": size,
            "whole_sha256": whole_sha,
            "chunk_bytes": chunk_bytes,
            "part_count": len(parts),
            "parts": parts,
        }
        (root / manifest_name).write_text(
            json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        rec = {
            "rel": rel,
            "exists": True,
            "split": True,
            "prefix": prefix,
            "manifest_remote": f"/content/{manifest_name}",
            "whole_bytes": size,
            "whole_sha256": whole_sha,
            "chunk_bytes": chunk_bytes,
            "part_count": len(parts),
        }
        print("SDLAB_DOWNLOAD_PREPARE_JSON " + json.dumps(rec), flush=True)
        records.append(rec)
    print(f"SDLAB_DOWNLOAD_PREPARE_DONE chunk_bytes={chunk_bytes}", flush=True)
    return records


def run_remote_script(
    script_path: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> int:
    effective_env = dict(os.environ if env is None else env)
    target_script = script_path or effective_env.get("SDLAB_REMOTE_SCRIPT", "").strip()
    if not target_script:
        print("REFUSED: SDLAB_REMOTE_SCRIPT is required", file=sys.stderr, flush=True)
        print("SDLAB_REMOTE_RC=2", flush=True)
        return 2

    cmd = [_resolve_bash(effective_env), target_script]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=effective_env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(scrub_line(line))
        sys.stdout.flush()
    rc = int(proc.wait())
    content_root = effective_env.get("SDLAB_CONTENT", "").strip()
    dl_chunk_raw = effective_env.get("SDLAB_DL_CHUNK_BYTES", "").strip()
    if content_root and dl_chunk_raw.isdigit() and int(dl_chunk_raw) > 0:
        prepare_content_downloads(content_root, int(dl_chunk_raw))
    print(f"SDLAB_REMOTE_RC={rc}", flush=True)
    return rc


if __name__ == "__main__":
    run_remote_script()
