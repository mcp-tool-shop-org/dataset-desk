"""dataset desk (prototype): curate a style-dataset-lab project's images with an image-text embedder.

The roles:
- EmbeddingGemma 2 (or SigLIP2) is the index;
- description_gate is the model-free description & prompt-echo gate;
- plain-sight writes captions;
- ai-eyes verifies claims.

This prototype does the index's and gate's jobs:
- exact, near and semantic duplicates;
- search by description, by picture, or by interleaved picture + text;
- outliers from the approved set (content and 30d palette+texture look channel);
- coverage of a subject list with canon exemplars;
- model-free description audit (exact/Jaccard duplicates, prompt echo, source, review worksheet);
- how well each approved record's description finds its own image (with a peeking guard on the multimodal pool);
- a Markdown report, plus a torch-free lite mode (--backend none).

It reads the project and never writes into it. It writes only its own cache, review queue and report under --out.
Models load lazily, on the CPU with capped threads at below-normal priority; a GPU needs --grant.

    dataset-desk index    --project P --backend eg2|siglip2|none [--limit N] [--images-only]
    dataset-desk dupes    --project P --backend eg2
    dataset-desk search   --project P --backend eg2 --text "a squat bog creature" [--image ref.png]
    dataset-desk coverage --project P --backend eg2 --subjects subjects.txt
    dataset-desk report   --project P --backend eg2|siglip2|none [--subjects subjects.txt] [--images-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

from dataset_desk import backends  # noqa: E402 — must run before numpy imports so BLAS thread env vars are capped

# Cap thread environment variables before numpy loads its BLAS pool.
backends.cap_threads(backends._argv_threads())

import numpy as np  # noqa: E402
from dataset_desk import description_gate  # noqa: E402
from dataset_desk.errors import EXIT_RUNTIME, DeskError  # noqa: E402
from dataset_desk import __version__  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
NEAR_BITS = 10        # dHash Hamming distance (of 256 bits) for a near-duplicate, as in the multimodal design
# The cosine at which a pair is flagged for review as a semantic duplicate. SemDeDup's band was 0.93-0.97 in CLIP
# space; it belongs to that embedder, so recalibrate it for each one from the top-1 neighbour percentiles in the
# report. Flags are for review only: nothing is removed automatically.
SEMANTIC = 0.95
DESC_CLUSTER_COSINE = 0.90

# The four projects in the EmbeddingGemma 2 vs SigLIP2 multimodal retrieval experiment.
# Embedding their descriptions or running the embedding fit check on them is refused until the counted run finishes,
# unless --images-only (or --backend none) or --allow-multimodal-pool is passed.
MULTIMODAL_POOL = frozenset({"rustline", "tallow-fen", "salt-road", "hesperia"})


def check_multimodal_pool(
    project: str,
    backend: str,
    images_only: bool = False,
    allow_multimodal_pool: bool = False,
    *,
    arm_acceptance: bool = False,
) -> None:
    """Enforce the peeking and arm-model acceptance guard on the four multimodal benchmark projects.

    On rustline, tallow-fen, salt-road and hesperia, before the experiment's counted run, no model that measures one
    of its arms (EmbeddingGemma 2 = arm G; ai-eyes / SigLIP2 = arm S) may embed descriptions or accept/reject a
    description unless --allow-multimodal-pool is passed."""
    name = os.path.basename(os.path.abspath(project)).lower()
    if name not in MULTIMODAL_POOL or allow_multimodal_pool:
        return
    if arm_acceptance:
        raise DeskError(
            "PERM_LOCKED_POOL",
            f"REFUSED: project {name!r} is in the multimodal experiment pool {sorted(MULTIMODAL_POOL)}; "
            f"arm-model acceptance step {backend!r} (EmbeddingGemma 2 = arm G, ai-eyes/SigLIP2 = arm S) is forbidden "
            "before the counted run. Only the model-free gate and human edit may accept descriptions unless "
            "--allow-multimodal-pool is passed."
        )
    if backend != "none" and not images_only:
        raise DeskError(
            "PERM_LOCKED_POOL",
            f"REFUSED: project {name!r} is in the multimodal experiment pool {sorted(MULTIMODAL_POOL)}; "
            "description embeddings and fit checks are locked until its counted run. "
            "Pass --images-only (or --backend none) to run without description embeddings, "
            "or --allow-multimodal-pool after the counted run."
        )


def ai_eyes_verify(
    project: str,
    descriptions: dict[str, str] | list[dict],
    *,
    allow_multimodal_pool: bool = False,
) -> dict:
    """Stub for ai-eyes (SigLIP2, arm S) noun verification (image_contains).

    Guarded by check_multimodal_pool(arm_acceptance=True) so SigLIP2 cannot accept or reject descriptions on
    rustline, tallow-fen, salt-road or hesperia before the counted run."""
    check_multimodal_pool(
        project,
        "ai-eyes (SigLIP2 / arm S image_contains)",
        allow_multimodal_pool=allow_multimodal_pool,
        arm_acceptance=True,
    )
    return {"checked": len(descriptions), "backend": "ai-eyes", "status": "stub"}


def eg2_fit_accept(
    project: str,
    descriptions: dict[str, str] | list[dict],
    *,
    allow_multimodal_pool: bool = False,
) -> dict:
    """Stub for any acceptance step driven by EmbeddingGemma 2 (arm G) retrieval rank."""
    check_multimodal_pool(
        project,
        "eg2 (EmbeddingGemma 2 / arm G fit check)",
        allow_multimodal_pool=allow_multimodal_pool,
        arm_acceptance=True,
    )
    return {"checked": len(descriptions), "backend": "eg2", "status": "stub"}


@dataclass
class Item:
    sha256: str
    paths: list           # project-relative, every byte-identical copy
    folders: list
    dhash: str = ""       # 256 bits as 64 hex digits
    record: dict | None = None  # id, status, explanation, judgment_note, failure_modes, prompt of the copy
    palette: list | None = None  # 30 numbers: 24 HSV colour/tone + 6 native surface/brushwork/FFT metrics

    @property
    def status(self) -> str:
        return (self.record or {}).get("status") or "unrecorded"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path: str, size: int = 16) -> str:
    """The difference hash: grayscale, 17×16, each pixel against its right neighbour. 256 bits."""
    from PIL import Image

    with Image.open(path) as im:
        return _dhash(im, size)


def _dhash(im, size: int = 16) -> str:
    from PIL import Image

    g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def _conv1d_axis0(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    r = len(k) // 2
    p = np.pad(a, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(a, dtype=np.float32)
    for i, w in enumerate(k):
        out += w * p[i:i + a.shape[0]]
    return out


def _gauss2d(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return _conv1d_axis0(_conv1d_axis0(a, k).T, k).T


def _sobel(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(a, 1, mode="reflect")
    s0 = p[:-2] + 2.0 * p[1:-1] + p[2:]
    gx = s0[:, 2:] - s0[:, :-2]
    s1 = p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]
    gy = s1[2:, :] - s1[:-2, :]
    return gx, gy


def _texture_stats(im) -> list[float]:
    """Six native-resolution surface/brushwork/frequency metrics ported from style-dataset-lab's measure_image.py
    (pure numpy, no scipy dependency), scaled to [0, 1]:
    - hf_ratio: high-frequency residual std relative to luminance std
    - flat_hf:  impasto/tooth surviving inside the flattest 40% of large-scale structure (scaled by 1/32)
    - edge_dens: fraction of strong edges (Sobel > 12)
    - edge_hard: acutance of the top 2% edges surviving a sigma-2 blur (squashed via tanh(x/3))
    - flat_coh:  structure-tensor stroke directionality inside the flat mask
    - band_hi:   radial 2D FFT power fraction above 0.30 Nyquist"""
    from PIL import Image

    rgb_im = im.convert("RGB")
    w, h = rgb_im.size
    if w < 8 or h < 8:
        return [0.0] * 6
    if max(w, h) > 1344:
        scale = 1344.0 / max(w, h)
        rgb_im = rgb_im.resize((max(8, int(round(w * scale))), max(8, int(round(h * scale)))), Image.Resampling.LANCZOS)
    a = np.asarray(rgb_im, dtype=np.float32)
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    lum_std = float(lum.std())
    if lum_std < 1e-6:
        return [0.0] * 6

    hp = lum - _gauss2d(lum, 1.5)
    struct = _gauss2d(lum, 6.0)
    sgx, sgy = _sobel(struct)
    sgrad = np.hypot(sgx, sgy)
    flat = sgrad <= np.percentile(sgrad, 40)

    ex, ey = _sobel(lum)
    egrad = np.hypot(ex, ey) / 4.0
    blur = _gauss2d(lum, 2.0)
    bx, by = _sobel(blur)
    bgrad = np.hypot(bx, by) / 4.0
    strong = egrad >= np.percentile(egrad, 98)
    raw_hard = float(egrad[strong].mean() / max(float(bgrad[strong].mean()), 1e-6)) if strong.any() else 0.0

    hx, hy = _sobel(hp)
    jxx = _gauss2d(hx * hx, 4.0)
    jyy = _gauss2d(hy * hy, 4.0)
    jxy = _gauss2d(hx * hy, 4.0)
    disc = np.sqrt(((jxx - jyy) / 2.0) ** 2 + jxy ** 2)
    tr = (jxx + jyy) / 2.0
    coh = 2.0 * disc / np.maximum(tr * 2.0, 1e-6)

    win = np.hanning(lum.shape[0])[:, None] * np.hanning(lum.shape[1])[None, :]
    spec = np.fft.fftshift(np.abs(np.fft.fft2((lum - float(lum.mean())) * win)) ** 2)
    cy, cx = (np.array(spec.shape) - 1) / 2.0
    yy, xx = np.ogrid[:spec.shape[0], :spec.shape[1]]
    rad = np.hypot((yy - cy) / max(spec.shape[0] / 2.0, 1.0), (xx - cx) / max(spec.shape[1] / 2.0, 1.0))
    tot = max(float(spec.sum()), 1e-9)

    hf_ratio = min(1.0, float(hp.std()) / max(lum_std, 1e-6))
    flat_hf = min(1.0, float(hp[flat].std()) / 32.0) if flat.any() else 0.0
    edge_dens = float((egrad > 12.0).mean())
    edge_hard = float(np.tanh(raw_hard / 3.0))
    flat_coh = float(coh[flat].mean()) if flat.any() else 0.0
    band_hi = float(spec[rad > 0.30].sum() / tot)
    return [hf_ratio, flat_hf, edge_dens, edge_hard, flat_coh, band_hi]


def _palette(im) -> list:
    """Colour, tone, and surface-texture statistics (30 bounded numbers in [0, 1]), a channel that sees the look more
    than the subject: a 12-bin hue histogram weighted by saturation, an 8-bin value histogram, the mean and spread of
    saturation and value, and 6 native-resolution surface/brushwork/FFT metrics (ported from measure_image.py).
    General image-text embedders rank mostly by subject (CSD, Somepalli et al. 2024; MegaStyle, Gao et al. 2026),
    so the desk keeps this beside them until a style descriptor is measured on our art."""
    hsv = np.asarray(im.convert("RGB").resize((64, 64)).convert("HSV"), dtype=np.float32) / 255.0
    h, sat, val = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    hue = np.histogram(h, bins=12, range=(0, 1), weights=sat)[0]
    hue = hue / max(float(hue.sum()), 1e-6)
    tone = np.histogram(val, bins=8, range=(0, 1))[0] / val.size
    tex = _texture_stats(im)
    vec = np.concatenate([hue, tone, [sat.mean(), sat.std(), val.mean(), val.std()], tex])
    return [round(float(x), 5) for x in vec]


def _trace_sampler_text(graph: dict, link) -> str | None:
    """Follow a sampler's positive or negative conditioning link back to its text-encoding node."""
    visited: set[str] = set()
    cur = link
    while isinstance(cur, list) and cur and isinstance(cur[0], (str, int)):
        nid = str(cur[0])
        if nid in visited or nid not in graph:
            break
        visited.add(nid)
        node = graph[nid]
        if not isinstance(node, dict):
            break
        inp = node.get("inputs") or {}
        for key in ("text", "prompt", "text_g", "text_l"):
            val = inp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        cur = inp.get("conditioning") or inp.get("conditioning_1") or inp.get("positive")
    return None


def _png_prompt_meta(im) -> dict:
    """Extract seed, positive prompt, negative_prompt, UNET/checkpoint, and LoRA stack from a ComfyUI PNG tEXt
    'prompt' chunk by following the sampler's positive and negative conditioning links."""
    raw = (getattr(im, "info", None) or {}).get("prompt")
    if not raw or not isinstance(raw, str):
        return {}
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(graph, dict):
        return {}
    meta: dict = {}
    loras = []
    fallback_texts: list[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type") or ""
        inp = node.get("inputs") or {}
        if cls in ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced", "RandomNoise"):
            seed = inp.get("seed", inp.get("noise_seed"))
            if isinstance(seed, int):
                meta.setdefault("seed", seed)
            pos_txt = _trace_sampler_text(graph, inp.get("positive"))
            if pos_txt:
                meta.setdefault("prompt", pos_txt)
            neg_txt = _trace_sampler_text(graph, inp.get("negative"))
            if neg_txt:
                meta.setdefault("negative_prompt", neg_txt)
        elif cls in ("UNETLoader", "CheckpointLoaderSimple"):
            name = inp.get("unet_name") or inp.get("ckpt_name")
            if isinstance(name, str):
                meta.setdefault("unet", name)
        elif cls in ("LoraLoaderModelOnly", "LoraLoader"):
            lname = inp.get("lora_name")
            if isinstance(lname, str):
                loras.append({"name": lname, "weight": inp.get("strength_model", 1.0)})
        elif cls in ("CLIPTextEncode", "TextEncodeQwenImageEditPlus"):
            txt = (inp.get("text") or inp.get("prompt") or "").strip()
            if txt:
                fallback_texts.append(txt)
    if "prompt" not in meta and fallback_texts:
        meta["prompt"] = fallback_texts[0]
    if loras:
        meta["loras"] = loras
    return meta


def image_stats(path: str) -> tuple[str, list, dict]:
    from PIL import Image

    with Image.open(path) as im:
        png_meta = _png_prompt_meta(im)
        return _dhash(im), _palette(im), png_meta


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _record_priority(rec: dict) -> tuple[int, str]:
    """Deterministic tie-breaker when byte-identical files have different records: approved beats borderline beats
    rejected beats unrecorded, then lexical id."""
    st = rec.get("status")
    rank = 0 if st == "approved" else (1 if st == "borderline" else (2 if st == "rejected" else 3))
    return (rank, str(rec.get("id") or ""))


def load_records(project: str) -> tuple[dict, list]:
    """asset_path -> the record's id, status, explanation (canon.subject, or judgment.explanation ONLY when status is
    approved), judgment_note, failure_modes, and prompt; plus the records that couldn't be read."""
    recs, problems = {}, []
    rdir = os.path.join(project, "records")
    for name in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(rdir, name), encoding="utf-8") as f:
                r = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"{name}: {e}")
            continue
        c, j, prov = r.get("canon") or {}, r.get("judgment") or {}, r.get("provenance") or {}
        status = j.get("status")
        subj = (c.get("subject") or "").strip()
        j_exp = (j.get("explanation") or "").strip()
        # A searchable description is canon.subject, or judgment.explanation only when status is approved.
        # Rejection notes stay in judgment_note for display and are never embedded as search descriptions.
        explanation = subj or (j_exp if status == "approved" else None) or None
        source = "canon.subject" if subj else ("judgment.explanation" if (status == "approved" and j_exp) else "none")
        fmodes = j.get("failure_modes")
        if isinstance(fmodes, list):
            modes = [str(m) for m in fmodes if m]
        elif j.get("failure_mode"):
            modes = [str(j["failure_mode"])]
        else:
            modes = []

        asset = (r.get("asset_path") or "").replace("\\", "/")
        entry: dict = {
            "id": r.get("id"),
            "status": status,
            "explanation": explanation,
            "judgment_note": j_exp or None,
            "canon_subject": subj or None,
            "source": source,
            "prompt": prov.get("prompt"),
            "recorded": True,
        }
        if modes:
            entry["failure_modes"] = modes
        if asset in recs and recs[asset].get("id") != entry["id"]:
            problems.append(f"{name}: duplicate record for {asset} (already recorded by {recs[asset].get('id')})")
            if _record_priority(entry) < _record_priority(recs[asset]):
                recs[asset] = entry
        else:
            recs[asset] = entry
    return recs, problems


def load_wave_metadata(project: str) -> dict[str, dict]:
    """Read outputs/<wave>/generation.json (PIN_PER_STEP receipts written by qwen_generate.py) and
    outputs/<wave>/_curation.json (multi-juror panel verdicts), indexed by project-relative path."""
    wave_meta: dict[str, dict] = {}
    out_dir = os.path.join(project, "outputs")
    if not os.path.isdir(out_dir):
        return wave_meta
    for folder in sorted(os.listdir(out_dir)):
        fdir = os.path.join(out_dir, folder)
        if not os.path.isdir(fdir):
            continue
        gen_path = os.path.join(fdir, "generation.json")
        if os.path.isfile(gen_path):
            try:
                with open(gen_path, encoding="utf-8") as f:
                    gen = json.load(f)
                base_info = {k: gen.get(k) for k in ("wave", "base", "unet", "style_prefix", "loras") if gen.get(k)}
                for item in gen.get("items") or []:
                    fname = item.get("file")
                    if not fname:
                        continue
                    rel = f"outputs/{folder}/{fname}".replace("\\", "/")
                    wave_meta[rel] = {
                        **base_info,
                        "subject_id": item.get("id"),
                        "variation": item.get("variation"),
                        "seed": item.get("seed"),
                        "prompt": item.get("prompt"),
                    }
            except (OSError, json.JSONDecodeError):
                pass
        cur_path = os.path.join(fdir, "_curation.json")
        if os.path.isfile(cur_path):
            try:
                with open(cur_path, encoding="utf-8") as f:
                    cur = json.load(f)
                for p_item in cur.get("passed") or []:
                    fname = p_item.get("file") if isinstance(p_item, dict) else p_item
                    if fname:
                        rel = f"outputs/{folder}/{fname}".replace("\\", "/")
                        entry = wave_meta.setdefault(rel, {})
                        entry["curation_status"] = "passed"
                        if isinstance(p_item, dict) and p_item.get("issues"):
                            entry["curation_issues"] = p_item["issues"]
                for f_item in cur.get("failed") or []:
                    fname = f_item.get("file") if isinstance(f_item, dict) else f_item
                    if fname:
                        rel = f"outputs/{folder}/{fname}".replace("\\", "/")
                        entry = wave_meta.setdefault(rel, {})
                        entry["curation_status"] = "failed"
                        if isinstance(f_item, dict) and f_item.get("issues"):
                            entry["curation_issues"] = f_item["issues"]
            except (OSError, json.JSONDecodeError):
                pass
    return wave_meta


def _load_scan_cache(out: str | None) -> dict[str, dict]:
    if not out:
        return {}
    path = os.path.join(out, "scan.jsonl")
    if not os.path.isfile(path):
        return {}
    cache: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if row.get("sha256") and len(row.get("palette") or []) == 30:
                        cache[row["sha256"]] = row
    except (OSError, json.JSONDecodeError):
        return {}
    return cache


def _save_scan_cache(out: str | None, rows: list[dict]) -> None:
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "scan.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def scan(project: str, strict: bool = False, out: str | None = None) -> tuple[list[Item], dict]:
    """Every image under outputs/<folder>/, merged by content. Files at the top of outputs/ (contact sheets and
    composites) are left out, as in the multimodal design. Joins records/*.json, outputs/<wave>/generation.json,
    outputs/<wave>/_curation.json, and embedded ComfyUI PNG tEXt prompt metadata. Reuses <out>/scan.jsonl across
    backends and runs so dHash and the 30d native-resolution look channel are computed once per unique sha256."""
    recs, problems = load_records(project)
    wave_meta = load_wave_metadata(project)
    scan_cache = _load_scan_cache(out)
    by_sha: dict[str, Item] = {}
    seen = set()
    out_dir = os.path.join(project, "outputs")
    for folder in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        fdir = os.path.join(out_dir, folder)
        if not os.path.isdir(fdir):
            continue
        for root, dirs, files in os.walk(fdir):
            dirs.sort()
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() not in IMAGE_EXT:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, project).replace("\\", "/")
                seen.add(rel)
                sha = sha256_file(full)
                it = by_sha.setdefault(sha, Item(sha, [], []))
                it.paths.append(rel)
                it.folders.append(folder)
                if rel in recs:
                    cand = recs[rel]
                    if it.record is None or not it.record.get("recorded"):
                        merged = dict(it.record or {})
                        merged.update({k: v for k, v in cand.items() if v is not None or k == "explanation"})
                        it.record = merged
                    elif it.record.get("id") != cand.get("id"):
                        problems.append(
                            f"{rel}: record collision on byte-identical file with record {it.record.get('id')}"
                        )
                        if _record_priority(cand) < _record_priority(it.record):
                            merged = dict(it.record)
                            for k in ("id", "status", "explanation", "judgment_note", "canon_subject", "source",
                                      "prompt", "failure_modes", "recorded"):
                                if k in cand:
                                    merged[k] = cand[k]
                                else:
                                    merged.pop(k, None)
                            it.record = merged
                if rel in wave_meta:
                    w = wave_meta[rel]
                    if it.record is None:
                        it.record = {"id": w.get("subject_id"), "status": None, "explanation": None,
                                     "prompt": w.get("prompt"), "recorded": False}
                    for k in ("wave", "base", "unet", "style_prefix", "loras", "seed", "curation_status", "curation_issues"):
                        if w.get(k) is not None and not it.record.get(k):
                            it.record[k] = w[k]
                    if w.get("prompt") and not it.record.get("prompt"):
                        it.record["prompt"] = w["prompt"]

    valid_items: list[Item] = []
    cache_rows: list[dict] = []
    look_seconds = 0.0
    look_computed_now = 0
    for it in by_sha.values():
        cached = scan_cache.get(it.sha256)
        if cached is not None:
            dh = cached["dhash"]
            pal = cached["palette"]
            png_meta = cached.get("png_meta") or {}
            dt = float(cached.get("look_seconds") or 0.0)
        else:
            t0 = time.perf_counter()
            try:
                dh, pal, png_meta = image_stats(os.path.join(project, it.paths[0]))
            except Exception as e:
                if strict:
                    raise
                problems.append(f"{it.paths[0]}: unreadable image ({e})")
                continue
            dt = time.perf_counter() - t0
            look_seconds += dt
            look_computed_now += 1
        it.dhash, it.palette = dh, pal
        if png_meta:
            if it.record is None:
                it.record = {"id": None, "status": None, "explanation": None,
                             "prompt": png_meta.get("prompt"), "recorded": False}
            for k, v in png_meta.items():
                if v is not None and not it.record.get(k):
                    it.record[k] = v
        valid_items.append(it)
        cache_rows.append({
            "sha256": it.sha256,
            "dhash": dh,
            "palette": pal,
            "png_meta": png_meta,
            "look_seconds": round(dt, 6),
        })

    _save_scan_cache(out, cache_rows)
    total_cached_look = sum(r["look_seconds"] for r in cache_rows)
    sec_per_img = (
        look_seconds / look_computed_now
        if look_computed_now
        else (total_cached_look / len(valid_items) if valid_items else 0.0)
    )
    return valid_items, {
        "files": len(seen),
        "unique": len(valid_items),
        "records": len(recs),
        "records_without_image": sorted(set(recs) - seen),
        "look_computed_now": look_computed_now,
        "look_seconds": round(look_seconds, 4),
        "look_seconds_per_image": round(sec_per_img, 6),
        "problems": problems,
    }


# ---- the cache: one folder per backend under --out ------------------------------------------------------------

def cache_dir(out: str, backend: str) -> str:
    return os.path.join(out, backend)


def save_cache(out: str, backend: str, items: list[Item], img: np.ndarray, txt: dict, meta: dict,
               cluster_txt: dict | None = None) -> None:
    d = cache_dir(out, backend)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    np.save(os.path.join(d, "images.npy"), img)
    dim = int(img.shape[1]) if img.ndim == 2 else 0
    shas = sorted(txt)
    np.save(os.path.join(d, "texts.npy"),
            np.vstack([txt[s] for s in shas]) if shas else np.zeros((0, dim), np.float32))
    cshas = sorted(cluster_txt or {})
    np.save(os.path.join(d, "cluster_texts.npy"),
            np.vstack([cluster_txt[s] for s in cshas]) if cshas else np.zeros((0, dim), np.float32))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "text_shas": shas, "cluster_text_shas": cshas}, f, indent=1)


def load_cache(out: str, backend: str) -> tuple[list[Item], np.ndarray, dict, dict]:
    d = cache_dir(out, backend)
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(d, "items.jsonl"), encoding="utf-8") as f:
            items = [Item(**json.loads(line)) for line in f]
        img = np.load(os.path.join(d, "images.npy"))
        tv = np.load(os.path.join(d, "texts.npy"))
    except OSError as err:
        raise DeskError("STATE_NO_INDEX", f"no {backend} index under {d}; run: dataset-desk index --backend {backend}",
                        cause=err) from err
    t_shas = meta.pop("text_shas", [])
    c_shas = meta.pop("cluster_text_shas", [])
    c_path = os.path.join(d, "cluster_texts.npy")
    if c_shas and os.path.isfile(c_path):
        ctv = np.load(c_path)
        meta["cluster_txt"] = dict(zip(c_shas, ctv, strict=True))
    return items, img, dict(zip(t_shas, tv, strict=True)), meta


def index(project: str, out: str, emb, limit: int | None = None, grant: str | None = None,
          strict: bool = False, images_only: bool = False, allow_multimodal_pool: bool = False,
          log=print) -> dict:
    """Embed every unique image, and every approved record's description as a search query. Reuses vectors already
    cached for the same backend, revision and vision token budget, and shares scan.jsonl across backends."""
    check_multimodal_pool(project, emb.name, images_only=images_only, allow_multimodal_pool=allow_multimodal_pool)
    items, stats = scan(project, strict=strict, out=out)
    if limit:
        items = items[:limit]
    old_img, old_txt, old_ctxt = {}, {}, {}
    vtokens = getattr(emb, "vision_tokens", 280)
    try:
        o_items, o_img, o_txt, o_meta = load_cache(out, emb.name)
        if o_meta.get("revision") == emb.revision and o_meta.get("vision_tokens", 280) == vtokens:
            old_img = dict(zip((it.sha256 for it in o_items), o_img, strict=True))
            cur_exp = {x.sha256: (x.record or {}).get("explanation") for x in items}
            old_txt = {
                it.sha256: o_txt[it.sha256]
                for it in o_items
                if it.sha256 in o_txt and (it.record or {}).get("explanation") == cur_exp.get(it.sha256)
            }
            o_ctxt = o_meta.get("cluster_txt") or {}
            old_ctxt = {
                it.sha256: o_ctxt[it.sha256]
                for it in o_items
                if it.sha256 in o_ctxt and (it.record or {}).get("explanation") == cur_exp.get(it.sha256)
            }
    except SystemExit:
        pass
    t0 = time.time()
    todo = [it for it in items if it.sha256 not in old_img]
    new = emb.images([os.path.join(project, it.paths[0]) for it in todo]) if todo else None
    if new is not None:
        for it, v in zip(todo, new, strict=True):
            old_img[it.sha256] = v
    t_img = time.time() - t0

    described = []
    if not images_only and emb.name != "none":
        described = [
            it for it in items
            if it.status == "approved" and (it.record or {}).get("explanation") and it.sha256 not in old_txt
        ]
        if described:
            texts_list = [it.record["explanation"] for it in described]
            for it, v in zip(described, emb.texts(texts_list, "query"), strict=True):
                old_txt[it.sha256] = v
            for it, cv in zip(described, emb.texts(texts_list, "clustering"), strict=True):
                old_ctxt[it.sha256] = cv

    dim = getattr(emb, "dim", 0)
    img = np.vstack([old_img[it.sha256] for it in items]) if (items and dim > 0) else np.zeros((len(items), dim), np.float32)
    keep = (
        {it.sha256 for it in items if it.status == "approved" and (it.record or {}).get("explanation")}
        if (not images_only and emb.name != "none")
        else set()
    )
    txt = {s: v for s, v in old_txt.items() if s in keep}
    cluster_txt = {s: v for s, v in old_ctxt.items() if s in keep}
    meta = {
        "backend": emb.name,
        "revision": emb.revision,
        "dim": int(img.shape[1]) if items else 0,
        "vision_tokens": vtokens,
        "dtype": getattr(emb, "dtype_name", "float32"),
        "images_only": bool(images_only or emb.name == "none"),
        "project": os.path.abspath(project),
        "grant": grant,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedded_now": len(todo),
        "texts_now": len(described),
        "image_seconds": round(t_img, 1),
        "params": getattr(emb, "params", None),
        **{k: v for k, v in stats.items() if k != "problems"},
        "problems": stats["problems"],
    }
    save_cache(out, emb.name, items, img, txt, meta, cluster_txt=cluster_txt)
    log(f"indexed {len(items)} unique images ({len(todo)} new, {t_img:.1f} s) and {len(txt)} descriptions "
        f"with {emb.name}; cache: {cache_dir(out, emb.name)}")
    return meta


# ---- the analyses: cache only, except where a query has to be embedded --------------------------------------------

def pairs_at_least(vecs: np.ndarray, thr: float, chunk: int = 1024):
    if vecs.ndim != 2 or vecs.shape[1] == 0:
        return
    for i in range(0, len(vecs), chunk):
        sims = vecs[i:i + chunk] @ vecs.T
        nz_a, nz_b = np.nonzero(sims >= thr)
        for a, b in zip(nz_a, nz_b, strict=True):
            if i + a < b:
                yield int(i + a), int(b), float(sims[a, b])


def clusters(n: int, pairs) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def dupes(items: list[Item], img: np.ndarray, near: int = NEAR_BITS, semantic: float = SEMANTIC) -> dict:
    exact = [it for it in items if len(it.paths) > 1]
    hs = [int(it.dhash, 16) for it in items]
    near_pairs = [(a, b, h) for a in range(len(hs)) for b in range(a + 1, len(hs))
                  if (h := (hs[a] ^ hs[b]).bit_count()) <= near]
    near_set = {(a, b) for a, b, _ in near_pairs}
    has_vecs = img.ndim == 2 and img.shape[1] > 0
    sem_pairs = [(a, b, s) for a, b, s in pairs_at_least(img, semantic) if (a, b) not in near_set] if has_vecs else []
    return {"exact": exact, "near": near_pairs, "semantic": sem_pairs,
            "semantic_groups": clusters(len(items), [(a, b) for a, b, _ in sem_pairs]),
            "thresholds": {"near_bits": near, "semantic_cosine": semantic},
            "top1_percentiles": top1_percentiles(img) if has_vecs else {}}


def top1_percentiles(img: np.ndarray, chunk: int = 1024) -> dict:
    """Each image's cosine to its nearest other image, as percentiles: the scale to set the semantic cut on."""
    if len(img) < 2 or img.ndim != 2 or img.shape[1] == 0:
        return {}
    best = []
    for i in range(0, len(img), chunk):
        sims = img[i:i + chunk] @ img.T
        for r in range(sims.shape[0]):
            sims[r, i + r] = -np.inf
        best.append(sims.max(axis=1))
    b = np.concatenate(best)
    return {f"p{q}": round(float(np.percentile(b, q)), 4) for q in (50, 90, 95, 99)} | {"max": round(float(b.max()), 4)}


def vendi(vecs: np.ndarray) -> float:
    """The Vendi score (Friedman and Dieng 2022): exp of the entropy of the eigenvalues of the cosine-similarity
    matrix over n. It reads as an effective number of distinct images: n for unrelated ones, 1 for copies.
    Uses the dual covariance matrix (d x d) when n > d, avoiding O(n^3) eigendecomposition at 1,000+ images."""
    if len(vecs) == 0 or vecs.ndim != 2 or vecs.shape[1] == 0:
        return 0.0
    n, d = vecs.shape
    gram = (vecs @ vecs.T) / n if n <= d else (vecs.T @ vecs) / n
    lam = np.linalg.eigvalsh(gram)
    lam = lam[lam > 1e-12]
    return float(np.exp(-(lam * np.log(lam)).sum()))


def diversity(items: list[Item], img: np.ndarray) -> list[tuple[str, int, float]]:
    """Vendi for the approved set, and for each folder's images (a folder is usually a wave), by size."""
    if img.ndim != 2 or img.shape[1] == 0:
        return []
    rows = []
    A = approved_idx(items)
    if A:
        rows.append(("approved", len(A), vendi(img[A])))
    folders = {}
    for i, it in enumerate(items):
        folders.setdefault(it.folders[0], []).append(i)
    for f, idx in sorted(folders.items(), key=lambda kv: -len(kv[1])):
        rows.append((f"outputs/{f}", len(idx), vendi(img[idx])))
    return rows


def approved_idx(items: list[Item]) -> list[int]:
    return [i for i, it in enumerate(items) if it.status == "approved"]


def outliers(items: list[Item], img: np.ndarray, k: int = 5, n: int = 15) -> dict:
    """The approved images least like the rest of the approved set (mean cosine to their k nearest), and the
    unapproved images nearest to and furthest from it. A general embedder mostly measures content, so read 'far'
    as an unusual subject or an unusual look, and check by eye."""
    A = approved_idx(items)
    if len(A) < 2 or img.ndim != 2 or img.shape[1] == 0:
        return {"approved": len(A), "least_typical": [], "nearest_canon": [], "furthest_from_canon": []}
    S = img[A] @ img[A].T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(A) - 1)
    fit = np.sort(S, axis=1)[:, -kk:].mean(axis=1)
    least = [(A[i], float(fit[i])) for i in np.argsort(fit)[:n]]
    others = [i for i in range(len(items)) if items[i].status != "approved"]
    near = img[others] @ img[A].T if others else np.zeros((0, len(A)))
    best = near.max(axis=1) if others else np.zeros(0)
    order = np.argsort(best)
    return {"approved": len(A), "k": kk, "least_typical": least,
            "nearest_canon": [(others[i], float(best[i])) for i in order[::-1][:n]],
            "furthest_from_canon": [(others[i], float(best[i])) for i in order[:n]]}


def look_fit(items: list[Item], k: int = 5) -> dict[int, float]:
    """For each approved image: the mean distance of its look vector (palette + native texture) to its k nearest
    approved neighbours, after standardizing each bounded [0, 1] statistic with a 0.02 spread floor so dead hue bins
    in monochrome/limited-palette projects do not explode. Larger means a less typical look."""
    A = [i for i in approved_idx(items) if items[i].palette]
    if len(A) < 2:
        return {}
    X = np.array([items[i].palette for i in A], dtype=np.float32)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 0.02)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    kk = min(k, len(A) - 1)
    return {A[i]: float(v) for i, v in enumerate(np.sort(D, axis=1)[:, :kk].mean(axis=1))}


def reading(items: list[Item], img: np.ndarray, k: int = 5) -> list[dict]:
    """Both channels for each approved image, as ranks within the approved set (1 = least typical). In lite mode
    (--backend none), content_rank is None and look_rank orders the approved set by the 30d look channel."""
    look = look_fit(items, k)
    look_rank = {i: r for r, (i, _) in enumerate(sorted(look.items(), key=lambda kv: -kv[1]), start=1)}
    if img.ndim != 2 or img.shape[1] == 0:
        return [{"item": i, "content_rank": None, "look_rank": r} for i, r in sorted(look_rank.items(), key=lambda kv: kv[1])]
    o = outliers(items, img, k=k, n=len(items))
    content = {i: r for r, (i, _) in enumerate(o["least_typical"], start=1)}
    rows = []
    for i in content:
        rows.append({"item": i, "content_rank": content[i], "look_rank": look_rank.get(i)})
    return rows


def mid_rank(sims: np.ndarray, own: int) -> float:
    """1 + the images scored above the own image + half the ties, so a tie never follows file order."""
    s = sims[own]
    return 1 + float((sims > s).sum()) + 0.5 * float((sims == s).sum() - 1)


def captions(items: list[Item], img: np.ndarray, txt: dict, n: int = 15) -> dict:
    """For every approved record description: the rank of its own image when the description searches the whole pool,
    plus group_rank (masking co-described siblings that share the exact same description text, as in rustline)."""
    if img.ndim != 2 or img.shape[1] == 0:
        return {"described": 0, "pool": len(items), "recall_at_1": None, "recall_at_5": None,
                "median_rank": None, "group_recall_at_1": None, "group_median_rank": None, "worst": []}
    pos = {it.sha256: i for i, it in enumerate(items)}
    norm_desc = {
        i: ((it.record or {}).get("explanation") or "").strip().lower()
        for i, it in enumerate(items)
        if it.status == "approved"
    }
    desc_groups: dict[str, list[int]] = {}
    for i, d in norm_desc.items():
        if d:
            desc_groups.setdefault(d, []).append(i)

    rows = []
    for sha, v in txt.items():
        if sha not in pos:
            continue
        i = pos[sha]
        if items[i].status != "approved" or not (items[i].record or {}).get("explanation"):
            continue
        sims = img @ v
        rk = mid_rank(sims, i)
        siblings = desc_groups.get(norm_desc.get(i, ""), [i])
        if len(siblings) > 1:
            masked = sims.copy()
            for sib in siblings:
                if sib != i:
                    masked[sib] = -np.inf
            grk = mid_rank(masked, i)
        else:
            grk = rk
        rows.append({"item": i, "rank": rk, "group_rank": grk, "siblings": len(siblings), "cosine": float(sims[i])})
    rows.sort(key=lambda r: (-r["group_rank"], -r["rank"]))
    ranks = [r["rank"] for r in rows]
    granks = [r["group_rank"] for r in rows]
    return {"described": len(rows), "pool": len(items),
            "recall_at_1": sum(r <= 1 for r in ranks) / len(ranks) if ranks else None,
            "recall_at_5": sum(r <= 5 for r in ranks) / len(ranks) if ranks else None,
            "median_rank": float(np.median(ranks)) if ranks else None,
            "group_recall_at_1": sum(r <= 1 for r in granks) / len(granks) if granks else None,
            "group_median_rank": float(np.median(granks)) if granks else None,
            "worst": rows[:n]}


def swap_check(items: list[Item], img: np.ndarray, txt: dict, seed: int = 0) -> dict | None:
    """Calibrate the caption check on approved images with known-bad pairs: give each description another approved
    image's description (enforcing that the swapped description text is not identical to the original), and measure
    how often a swapped pair ranks worse than a true one (an AUC; 0.5 is chance)."""
    if img.ndim != 2 or img.shape[1] == 0:
        return None
    pos = {it.sha256: i for i, it in enumerate(items)}
    shas = [
        sh for sh in sorted(txt)
        if sh in pos and items[pos[sh]].status == "approved" and (items[pos[sh]].record or {}).get("explanation")
    ]
    if len(shas) < 3:
        return None
    descs = [((items[pos[sh]].record or {}).get("explanation") or sh).strip().lower() for sh in shas]
    counts: dict[str, int] = {}
    for d in descs:
        counts[d] = counts.get(d, 0) + 1
    if len(counts) < 3:
        return None
    if max(counts.values()) > len(shas) // 2:
        seen_d = set()
        dedup_shas, dedup_descs = [], []
        for sh, d in zip(shas, descs, strict=True):
            if d not in seen_d:
                seen_d.add(d)
                dedup_shas.append(sh)
                dedup_descs.append(d)
        shas, descs = dedup_shas, dedup_descs
        counts = {d: 1 for d in descs}

    rng = np.random.default_rng(seed)
    n = len(shas)
    desc_arr = np.array(descs)
    perm = None
    for _ in range(64):
        cand = rng.permutation(n)
        if not np.any(cand == np.arange(n)) and not np.any(desc_arr[cand] == desc_arr):
            perm = cand
            break
    if perm is None:
        max_c = max(counts.values())
        group_keys = list(counts.keys())
        rng.shuffle(group_keys)
        g_rank = {g: idx for idx, g in enumerate(group_keys)}
        order = sorted(range(n), key=lambda i: (g_rank[descs[i]], float(rng.random())))
        shifted = np.roll(order, max_c)
        perm = np.empty(n, dtype=int)
        for src, dst in zip(order, shifted, strict=True):
            perm[src] = dst

    true = [mid_rank(img @ txt[sh], pos[sh]) for sh in shas]
    swapped = [mid_rank(img @ txt[shas[perm[k]]], pos[sh]) for k, sh in enumerate(shas)]
    t, w = np.array(true), np.array(swapped)
    auc = float(((w[:, None] > t[None, :]).sum() + 0.5 * (w[:, None] == t[None, :]).sum()) / (len(w) * len(t)))
    return {"pairs": len(shas), "auc": auc, "true_median_rank": float(np.median(t)),
            "swapped_median_rank": float(np.median(w)), "seed": seed}


def embedding_description_clusters(items: list[Item], cluster_txt: dict, thr: float = DESC_CLUSTER_COSINE) -> list[list[str]]:
    """Display-only near-duplicate description groups using clustering-prefixed text embeddings (excluding exact
    text duplicates already grouped by the model-free gate)."""
    if not cluster_txt:
        return []
    pos = {it.sha256: i for i, it in enumerate(items)}
    shas = [sh for sh in sorted(cluster_txt) if sh in pos and items[pos[sh]].status == "approved"]
    if len(shas) < 2:
        return []
    vecs = np.vstack([cluster_txt[sh] for sh in shas])
    norms = [description_gate.normalize_text((items[pos[sh]].record or {}).get("explanation")) for sh in shas]
    ids = [str((items[pos[sh]].record or {}).get("id") or items[pos[sh]].paths[0]) for sh in shas]
    pairs = []
    for a, b, _ in pairs_at_least(vecs, thr):
        if norms[a] != norms[b]:
            pairs.append((a, b))
    return [[ids[idx] for idx in g] for g in clusters(len(shas), pairs)]


def search(img: np.ndarray, q: np.ndarray, k: int = 10, among: list[int] | None = None) -> list[tuple[int, float]]:
    if img.ndim != 2 or img.shape[1] == 0:
        return []
    idx = among if among is not None else list(range(len(img)))
    if not idx:
        return []
    sims = img[idx] @ q
    return [(idx[i], float(sims[i])) for i in np.argsort(-sims)[:k]]


def coverage(items: list[Item], img: np.ndarray, subjects: list[str], qv: np.ndarray, k: int = 3) -> list[dict]:
    """For each wanted subject, the closest approved images plus canon exemplar metadata (path, cosine, look_rank,
    loras). Weakest best match comes first."""
    A = approved_idx(items)
    look = look_fit(items)
    look_rank = {i: r for r, (i, _) in enumerate(sorted(look.items(), key=lambda kv: -kv[1]), start=1)}
    rows = []
    for s, q in zip(subjects, qv, strict=True):
        top = search(img, q, k, among=A) if A else []
        exemplars = [
            {
                "item": i,
                "path": items[i].paths[0],
                "cosine": round(score, 4),
                "look_rank": look_rank.get(i),
                "loras": (items[i].record or {}).get("loras"),
            }
            for i, score in top
        ]
        rows.append({"subject": s, "top": top, "exemplars": exemplars})
    rows.sort(key=lambda r: r["top"][0][1] if r["top"] else -1.0)
    return rows


# ---- the report ------------------------------------------------------------------------------------------------

def _link(project: str, report_dir: str, it: Item) -> str:
    return f"[{it.paths[0]}]({description_gate.link_target(os.path.join(project, it.paths[0]), report_dir)})"


def _status_label(it: Item) -> str:
    st = it.status
    modes = (it.record or {}).get("failure_modes") or []
    if st in ("rejected", "borderline") and modes:
        return f"{st}: {', '.join(modes)}"
    return st


def report(project: str, out: str, items: list[Item], img: np.ndarray, txt: dict, meta: dict,
           cov: list[dict] | None = None, images_only: bool = False,
           allow_multimodal_pool: bool = False, drafter=None, token_counter=None) -> str:
    eff_images_only = bool(images_only or meta.get("images_only") or meta.get("backend") == "none")
    check_multimodal_pool(project, meta.get("backend", "eg2"), images_only=eff_images_only,
                          allow_multimodal_pool=allow_multimodal_pool)
    has_vecs = img.ndim == 2 and img.shape[1] > 0
    d = dupes(items, img)
    o = outliers(items, img) if has_vecs else {"approved": len(approved_idx(items)), "least_typical": [],
                                               "nearest_canon": [], "furthest_from_canon": []}
    c = captions(items, img, txt) if (has_vecs and not eff_images_only) else {"described": 0}
    raw_recs = description_gate.load_project_records(project)
    if raw_recs:
        desc_audit = description_gate.audit_records(raw_recs, token_counter=token_counter)
    else:
        synth = [{"id": it.record.get("id") or it.paths[0], "asset_path": it.paths[0], **it.record}
                 for it in items if it.record and it.record.get("recorded")]
        desc_audit = description_gate.audit_records(synth, token_counter=token_counter)
    if drafter is not None:
        description_gate.draft_flagged_records(project, desc_audit, drafter)
    dhashes = {p: it.dhash for it in items for p in it.paths if it.dhash}
    rev_jsonl, rev_ws = description_gate.write_review_files(project, out, desc_audit, dhashes=dhashes)

    L = [f"# Dataset desk: {os.path.basename(os.path.abspath(project))} ({meta['backend']})", "",
         f"{meta['files']} image files, {meta['unique']} unique by content, {meta['records']} records; "
         f"{len(approved_idx(items))} approved. Index: {meta['backend']} @ {meta['revision'][:7]}, "
         f"{meta['dim']} dimensions, look channel {meta.get('look_seconds_per_image', 0.0):.4f} s/image, "
         f"{meta['created']}. Every line below is a prompt to look, not a verdict.", ""]
    if meta.get("problems") or meta.get("records_without_image"):
        L += ["## Problems", ""] + [f"- {p}" for p in meta.get("problems", [])] + \
             [f"- record without an image file: {p}" for p in meta.get("records_without_image", [])] + [""]
    link = lambda it: _link(project, out, it)  # noqa: E731
    L += ["## Duplicates", "", f"- **Byte-identical copies:** {len(d['exact'])} images have more than one file."]
    L += [f"  - {', '.join(it.paths)} ({_status_label(it)})" for it in d["exact"][:20]]
    L += [f"- **Near-duplicates** (dHash within {d['thresholds']['near_bits']} of 256 bits): {len(d['near'])} pairs."]
    L += [f"  - {link(items[a])} ({_status_label(items[a])}) and {link(items[b])} ({_status_label(items[b])}): {h} bits"
          for a, b, h in d["near"][:20]]
    if has_vecs:
        L += [f"- **Semantic duplicates** (cosine at least {d['thresholds']['semantic_cosine']}, near pairs left out): "
              f"{len(d['semantic'])} pairs in {len(d['semantic_groups'])} groups."]
        L += [f"  - {', '.join(f'{link(items[i])} ({_status_label(items[i])})' for i in g[:6])}"
              for g in d["semantic_groups"][:20]]
        L += [f"- **Each image's cosine to its nearest other image:** "
              f"{', '.join(f'{k} {v}' for k, v in d['top1_percentiles'].items())}. Set the semantic cut against this "
              f"scale, then look at about 30 pairs on each side of it."]
        L += ["", "## Diversity (Vendi score: the effective number of distinct images)", "",
              "| set | images | Vendi |", "|---|---|---|"]
        L += [f"| {name} | {n} | {v:.1f} |" for name, n, v in diversity(items, img)]

    L += ["", "## Least typical of the approved set", "",
          "Ranked within the approved set (1 = least typical). **Look**: palette, tone and native surface/brushwork "
          "texture against the approved set. Content-atypical but look-typical is most likely an odd subject in the right "
          "look; look-atypical is worth checking for rendering.", "", "| image | content rank | look rank |", "|---|---|---|"]
    rd = sorted(reading(items, img), key=lambda r: min(r["content_rank"] or 10 ** 9, r["look_rank"] or 10 ** 9))
    L += [f"| {link(items[r['item']])} | {r['content_rank'] if r['content_rank'] is not None else '—'} | {r['look_rank']} |"
          for r in rd[:20]]

    if has_vecs:
        L += ["", "## Unapproved images nearest the approved set", "", "Candidates that already look like canon.", ""]
        L += [f"- {link(items[i])} ({_status_label(items[i])}): {s:.3f}" for i, s in o["nearest_canon"]]
        L += ["", "## Unapproved images furthest from the approved set", ""]
        L += [f"- {link(items[i])} ({_status_label(items[i])}): {s:.3f}" for i, s in o["furthest_from_canon"]]

    # Model-free description gate section (always runs, including in lite / --images-only mode)
    L += [
        "",
        "## Description audit (model-free gate)",
        "",
        f"{desc_audit['total_checked']} approved records checked (`stopwords.txt` sha256 `{desc_audit['stopwords_sha256'][:12]}`): "
        f"**{desc_audit['passed_model_free_count']} passed model-free** "
        f"({desc_audit['passed_count']} token-verified, {desc_audit['unchecked_count']} token-unchecked), "
        f"**{desc_audit['flagged_count']} flagged** (median prompt word share {desc_audit['median_prompt_word_share']:.2f}, "
        f"content-word share {desc_audit['median_prompt_content_word_share']:.2f}). "
        f"Review queue: `{os.path.basename(rev_jsonl)}`; re-description worksheet: `{os.path.basename(rev_ws)}`.",
    ]
    if desc_audit["exact_groups"]:
        L += ["", f"- **Exact duplicate descriptions/rulings:** {len(desc_audit['exact_groups'])} group(s)."]
        for g in desc_audit["exact_groups"][:15]:
            L += [f"  - ×{g['count']} (`{g['source']}`): \"{g['text']}\" ({', '.join(g['ids'][:6])}{'…' if len(g['ids']) > 6 else ''})"]
    if desc_audit["near_groups"]:
        L += ["", f"- **Near-duplicate descriptions** (content-word Jaccard ≥ {desc_audit['jaccard_threshold']:.2f}): "
              f"{len(desc_audit['near_groups'])} group(s)."]
        for g in desc_audit["near_groups"][:15]:
            L += [f"  - {', '.join(g[:8])}"]
    if has_vecs and not eff_images_only and meta.get("cluster_txt"):
        emb_groups = embedding_description_clusters(items, meta["cluster_txt"])
        L += ["", f"- **Embedding near-duplicate descriptions** (`clustering` prefix, cosine ≥ {DESC_CLUSTER_COSINE:.2f}; "
              f"display only, never gates): {len(emb_groups)} group(s)."]
        for g in emb_groups[:15]:
            L += [f"  - {', '.join(g[:8])}"]
    if desc_audit["flagged"]:
        L += ["", "### Flagged descriptions for review", ""]
        for r in desc_audit["flagged"][:30]:
            L += [
                f"- `{r['id']}` (`{r['source']}`, `author_kind: {r['author_kind']}`, "
                f"flags: {', '.join(r['flags'])} — {'; '.join(r['reasons'])})"
            ]

    if c.get("described"):
        L += ["", "## Do the descriptions find their own images?", "",
              f"{c['described']} approved descriptions searched the pool of {c['pool']} (display only, never gates): "
              f"recall@1 {c['recall_at_1']:.2f} (group recall@1 {c['group_recall_at_1']:.2f}), "
              f"recall@5 {c['recall_at_5']:.2f}, median rank {c['median_rank']:.1f} "
              f"(group median {c['group_median_rank']:.1f}). The worst first:", ""]
        L += [f"- rank {r['rank']:.1f} (group {r['group_rank']:.1f}), cosine {r['cosine']:.3f}: {link(items[r['item']])}: "
              f"\"{items[r['item']].record['explanation']}\"" for r in c["worst"]]
        sw = swap_check(items, img, txt)
        if sw:
            L += ["", f"**Calibration with swapped descriptions** ({sw['pairs']} pairs, seed {sw['seed']}): a swapped "
                  f"pair ranks worse than a true one with probability {sw['auc']:.2f} (0.5 is chance); median rank "
                  f"{sw['true_median_rank']:.1f} true against {sw['swapped_median_rank']:.1f} swapped. This measures gross "
                  f"mismatches only."]
    elif eff_images_only:
        reason = "`--backend none` (lite mode)" if meta.get("backend") == "none" else "`--images-only`"
        L += ["", "## Do the descriptions find their own images?", "",
              f"Embedding description fit check skipped ({reason})."]
    if cov:
        L += ["", "## Coverage of the subject list", "", "Weakest best match first; these are the likely gaps.", ""]
        for row in cov:
            tops = "; ".join(f"{link(items[i])} {s:.3f}" for i, s in row["top"])
            L += [f"- **{row['subject']}**: {tops}"]
    path = os.path.join(out, f"report-{meta['backend']}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---- the command line ---------------------------------------------------------------------------------------------

def lower_priority() -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    else:
        os.nice(10)


def _token_counter():
    """The gate's SigLIP2 token counter over the cached tokenizer, loaded once, offline. A missing tokenizer is a
    structured error, not a crash."""
    try:
        tok = description_gate.load_siglip2_tokenizer()
    except (ImportError, OSError, ValueError) as err:
        raise DeskError("DEP_TOKENIZER_MISSING",
                        f"--count-tokens needs the SigLIP2 tokenizer ({backends.SIGLIP2_REPO} at "
                        f"{backends.SIGLIP2_REVISION[:7]}) in the local Hugging Face cache", cause=err) from err
    return lambda texts: description_gate.siglip2_token_counts(texts, tokenizer=tok)


def main(argv: list[str] | None = None, make=backends.make) -> int:
    ap = argparse.ArgumentParser(
        prog="dataset-desk", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("commands:\n"
                "  index     scan the project and embed its images (and descriptions) into the cache under --out\n"
                "  dupes     exact, near and semantic duplicates, as JSON\n"
                "  search    the nearest images to --text, --image, or both\n"
                "  coverage  the best images for each subject in --subjects\n"
                "  report    the Markdown report: duplicates, outliers, descriptions, coverage\n\n"
                "exit codes: 0 ok, 1 user error or refusal, 2 unexpected error"))
    ap.add_argument("command", choices=["index", "dupes", "search", "coverage", "report"], help="what to do")
    ap.add_argument("--version", action="version", version=f"dataset-desk {__version__}")
    ap.add_argument("--project", required=True, help="a style-dataset-lab project folder (read only)")
    ap.add_argument("--backend", default="eg2", choices=["eg2", "siglip2", "none", "fake"],
                    help="the index: EmbeddingGemma 2, SigLIP2, none (model-free lite mode), or fake (tests)")
    ap.add_argument("--out", help="where the cache and report go (default: desk-out/<project> here)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (on a shared machine, half the cores at most)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or a torch device such as cuda; needs --grant")
    ap.add_argument("--grant", help="a GPU grant id from whoever schedules the card; required for any device but cpu")
    ap.add_argument("--limit", type=int, help="index only the first N unique images (a smoke test)")
    ap.add_argument("--strict", action="store_true", help="halt immediately on an unreadable image file")
    ap.add_argument("--count-tokens", action="store_true",
                    help="report: run the gate's 64-token test with the cached SigLIP2 tokenizer (offline; needs the "
                         "models extra); without it, records that pass the other tests stay 'unchecked'")
    ap.add_argument("--images-only", action="store_true",
                    help="index/report images and model-free gates only; do not embed descriptions")
    ap.add_argument("--allow-multimodal-pool", action="store_true",
                    help="permit description embeddings and arm-model checks on the locked benchmark projects "
                         f"({', '.join(sorted(MULTIMODAL_POOL))}) once its counted run is done")
    ap.add_argument("--draft", default="none", choices=["none", "fake", "plain-sight", "florence2"],
                    help="report: draft flagged descriptions blind from image pixels via plain-sight (or fake)")
    ap.add_argument("--text", help="search: a description (combine with --image on eg2/fake for multimodal search)")
    ap.add_argument("--image", help="search: an image file")
    ap.add_argument("--subjects", help="coverage/report: a file with one wanted subject per line")
    ap.add_argument("-k", type=int, default=10, help="search: how many results")
    ap.add_argument("--debug", action="store_true", help="show stack traces for unexpected errors")
    a = ap.parse_args(argv)
    project = os.path.abspath(a.project)
    out = os.path.abspath(a.out or os.path.join("desk-out", os.path.basename(project)))
    try:
        out_inside = os.path.commonpath([project, out]) == project
    except ValueError:  # different drives on Windows: --out cannot be inside the project
        out_inside = False
    if out_inside:
        raise DeskError("PERM_OUT_INSIDE_PROJECT", "REFUSED: --out is inside the project; the desk never writes into a project")
    backends.check_device(a.device, a.grant)
    backends.cap_threads(a.threads)
    if a.command in ("index", "report"):
        check_multimodal_pool(project, a.backend, images_only=a.images_only,
                              allow_multimodal_pool=a.allow_multimodal_pool)
    if a.backend == "none" and a.command in ("search", "coverage"):
        raise DeskError("INPUT_BAD_ARGUMENT",
                        f"command {a.command!r} requires an embedding backend (eg2, siglip2, or fake), not 'none'")
    needs_model = a.command in ("index", "search", "coverage") or (a.command == "report" and a.subjects)
    emb = None
    if needs_model:
        if a.backend not in ("fake", "none"):
            lower_priority()
        emb = make(a.backend, device=a.device, threads=a.threads, grant=a.grant)
    if a.command == "index":
        index(project, out, emb, limit=a.limit, grant=a.grant, strict=a.strict,
              images_only=a.images_only, allow_multimodal_pool=a.allow_multimodal_pool)
        return 0
    items, img, txt, meta = load_cache(out, a.backend)
    subjects = None
    if a.subjects:
        with open(a.subjects, encoding="utf-8") as f:
            subjects = [s.strip() for s in f if s.strip() and not s.startswith("#")]
    if a.command == "dupes":
        d = dupes(items, img)
        print(json.dumps({"exact": [it.paths for it in d["exact"]],
                          "near": [(items[x].paths[0], items[y].paths[0], h) for x, y, h in d["near"]],
                          "semantic": [(items[x].paths[0], items[y].paths[0], round(s, 4)) for x, y, s in d["semantic"]],
                          "thresholds": d["thresholds"], "top1_percentiles": d["top1_percentiles"]}, indent=1))
    elif a.command == "search":
        if not a.text and not a.image:
            raise DeskError("INPUT_BAD_ARGUMENT", "search needs --text or --image (or both on eg2/fake for multimodal search)")
        if a.text and a.image:
            if not hasattr(emb, "multimodal"):
                raise DeskError("INPUT_BAD_ARGUMENT",
                                f"backend {a.backend!r} does not support joint --text + --image search; use eg2")
            q = emb.multimodal([{"text": a.text, "images": [a.image]}], role="query")[0]
        elif a.text:
            q = emb.texts([a.text], "query")[0]
        else:
            q = emb.images([a.image])[0]
        for i, s in search(img, q, a.k):
            print(f"{s:.4f}  {items[i].status:10s}  {items[i].paths[0]}")
    elif a.command == "coverage":
        if not subjects:
            raise DeskError("INPUT_BAD_ARGUMENT", "coverage needs --subjects")
        for row in coverage(items, img, subjects, emb.texts(subjects, "query"), k=3):
            print(f"{row['subject']}: " + "; ".join(f"{items[i].paths[0]} {s:.3f}" for i, s in row["top"]))
    elif a.command == "report":
        cov = coverage(items, img, subjects, emb.texts(subjects, "query")) if (subjects and a.backend != "none") else None
        drafter = backends.make_drafter(a.draft, device=a.device, threads=a.threads, grant=a.grant)
        print(report(project, out, items, img, txt, meta, cov,
                     images_only=a.images_only, allow_multimodal_pool=a.allow_multimodal_pool, drafter=drafter,
                     token_counter=_token_counter() if a.count_tokens else None))
    return 0


def run(argv: list[str] | None = None) -> int:
    """The console entry point: structured errors, exit codes 0/1/2, and no stack trace unless --debug."""
    args = sys.argv[1:] if argv is None else argv
    debug = "--debug" in args
    try:
        return main(args)
    except DeskError as err:
        print(err.render(), file=sys.stderr)
        return err.exit_code
    except KeyboardInterrupt:
        print("error RUNTIME_INTERRUPTED: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI boundary turns any failure into the structured shape
        if debug:
            raise
        err = DeskError("RUNTIME_UNEXPECTED", "the desk stopped on an unexpected error", cause=exc,
                        exit_code=EXIT_RUNTIME)
        print(err.render(), file=sys.stderr)
        return err.exit_code


if __name__ == "__main__":
    sys.exit(run())
