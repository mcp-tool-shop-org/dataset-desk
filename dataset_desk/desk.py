"""dataset desk (prototype): duplicates, outliers and gaps in image, text and code datasets, and a description gate.

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
import concurrent.futures
import functools
import hashlib
import heapq
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass, field

from dataset_desk import backends  # noqa: E402 — must run before numpy imports so BLAS thread env vars are capped

# Cap thread environment variables before numpy loads its BLAS pool.
backends.cap_threads(backends._argv_threads())

import numpy as np  # noqa: E402
from dataset_desk import description_gate, registry  # noqa: E402
from dataset_desk.errors import EXIT_RUNTIME, DeskError  # noqa: E402
from dataset_desk import __version__  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_EXT = frozenset({".md", ".txt", ".json", ".jsonl", ".log", ".out", ".py", ".sh", ".toml", ".yml", ".yaml", ""})
CODE_EXT = frozenset({".py", ".sh", ".toml", ".yml", ".yaml"})
TEXT_CHUNK_CHARS = 6000
TEXT_MAX_CHUNKS = 8
NUL_PROBE_BYTES = 4096
SCAN_V = 2  # scan.jsonl row schema version (v2 adds dhash_flip)

# Every duplicate threshold below is provisional and named after the calibration rung that measured it.
NEAR_BITS = 26          # provisional (rung 2, recall 0.81): lite flip_hamming threshold (disagree bits of 256)
TWO_SIGNAL_BITS = 44    # provisional (rung 2, recall 0.91, 0.00 cross-project FP): DINOv2 two-signal flip_hamming ceiling
DINOV2_COS_LOW = 0.94   # provisional (rung 2, recall 0.91, 0.00 cross-project FP): DINOv2 two-signal low cosine floor
DINOV2_COS_HIGH = 0.985 # provisional (rung 2, recall 0.91, 0.00 cross-project FP): DINOv2 unconditional cosine floor
SEMANTIC = 0.95         # provisional (rung 1/2): fallback cosine threshold for non-DINOv2 or explicit --semantic runs
CODE_SEMANTIC = 0.95    # provisional (rung 2b, 17 seeds: positives >= 0.976, hard negatives <= 0.931): EG2 code dupes
TEXT_SEMANTIC = 0.98    # provisional (rung 2b, recall 0.94, FP 0.01): bge-m3 text dupes
DESC_CLUSTER_COSINE = 0.90  # provisional (rung 1): display-only description cluster cosine

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
    spec = registry.REGISTRY.get(backend)
    eff_images_only = bool(images_only or backend == "none" or (spec is not None and "text" not in spec.modalities))
    if not eff_images_only:
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
    dhash_flip: str = ""  # 256 bits as 64 hex digits of the image flipped left-to-right
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


def dhash_flip(path: str, size: int = 16) -> str:
    """The difference hash of the image flipped left to right, computed from the flipped image itself."""
    from PIL import Image

    with Image.open(path) as im:
        return _dhash_flip(im, size)


def _dhash(im, size: int = 16) -> str:
    from PIL import Image

    g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def _dhash_flip(im, size: int = 16) -> str:
    from PIL import Image

    return _dhash(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT), size)


def _conv1d_axis0(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    r = len(k) // 2
    p = np.pad(a, ((r, r), (0, 0)), mode="reflect")
    h, w = a.shape
    out = np.empty((h, w), dtype=np.float32)
    strip = 128
    for s in range(0, h, strip):
        e = min(h, s + strip)
        acc = k[0] * p[s:e]
        for i in range(1, len(k)):
            acc = acc + k[i] * p[s + i:e + i]
        out[s:e] = acc
    return out


@functools.lru_cache(maxsize=16)
def _fft_window_and_mask(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(shape[0])[:, None] * np.hanning(shape[1])[None, :]
    cy, cx = (np.array(shape) - 1) / 2.0
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    rad = np.hypot((yy - cy) / max(shape[0] / 2.0, 1.0), (xx - cx) / max(shape[1] / 2.0, 1.0))
    return win, rad > 0.30


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

    win, rad_hi = _fft_window_and_mask(lum.shape)
    spec = np.fft.fftshift(np.abs(np.fft.fft2((lum - float(lum.mean())) * win)) ** 2)
    tot = max(float(spec.sum()), 1e-9)

    hf_ratio = min(1.0, float(hp.std()) / max(lum_std, 1e-6))
    flat_hf = min(1.0, float(hp[flat].std()) / 32.0) if flat.any() else 0.0
    edge_dens = float((egrad > 12.0).mean())
    edge_hard = float(np.tanh(raw_hard / 3.0))
    flat_coh = float(coh[flat].mean()) if flat.any() else 0.0
    band_hi = float(spec[rad_hi].sum() / tot)
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


def _scan_image_stats(path: str) -> tuple[str, str, list, dict]:
    from PIL import Image

    with Image.open(path) as im:
        png_meta = _png_prompt_meta(im)
        return _dhash(im), _dhash_flip(im), _palette(im), png_meta


def _init_scan_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _scan_worker_one(full_path: str) -> tuple[tuple[str, str, list, dict] | None, str | None, float]:
    t0 = time.perf_counter()
    try:
        return _scan_image_stats(full_path), None, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return None, str(e), time.perf_counter() - t0


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def flip_hamming(a: Item, b: Item) -> int:
    """min(hamming(a.dhash, b.dhash), hamming(a.dhash, b.dhash_flip)), falling back to the plain hash when
    dhash_flip is empty on in-memory test items."""
    d0 = hamming(a.dhash, b.dhash)
    if b.dhash_flip:
        return min(d0, hamming(a.dhash, b.dhash_flip))
    if a.dhash_flip:
        return min(d0, hamming(a.dhash_flip, b.dhash))
    return d0


# ---- many-image paths: vectorised and memory-bounded, with the same results as the per-pair functions above ----

BLOCK_BYTES = 1 << 26   # about 64 MiB: the most any one block of a pairwise computation holds at once
_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")  # NumPy 2.0+; older NumPy uses the byte table


def _block_rows(n_cols: int, bytes_per_cell: int) -> int:
    """Rows per block so a (rows, n_cols) block of bytes_per_cell-byte cells stays under BLOCK_BYTES."""
    return max(1, BLOCK_BYTES // max(1, n_cols * bytes_per_cell))


def _hash_words(hexes: list[str]) -> np.ndarray:
    """(n, 4) uint64 from 256-bit hex hashes. An empty hash gives a zero row; callers mask it out."""
    out = np.zeros((len(hexes), 4), dtype=np.uint64)
    for i, h in enumerate(hexes):
        if h:
            out[i] = np.frombuffer(int(h, 16).to_bytes(32, "big"), dtype=">u8")
    return out


def _hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) Hamming distances between rows of two _hash_words arrays."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    if _HAS_BITWISE_COUNT:
        return np.bitwise_count(x).sum(axis=2, dtype=np.int32)
    return _POP8[x.view(np.uint8)].sum(axis=2, dtype=np.int32)


def near_pairs_within(items: list[Item], near: int):
    """Yield (a, b, flip_hamming) for a < b with both hashes present and flip_hamming <= near, in (a, b) order:
    the same pairs and order as a loop over flip_hamming, block by block so it scales to many images."""
    n = len(items)
    has = np.array([bool(it.dhash) for it in items])
    has_flip = np.array([bool(it.dhash_flip) for it in items])
    H = _hash_words([it.dhash for it in items])
    F = _hash_words([it.dhash_flip for it in items])
    any_flip = bool(has_flip.any())
    need_row_flip = any_flip and not bool(has_flip.all())
    rows = _block_rows(n, 4 * 8 * 3)
    for i in range(0, n, rows):
        j = min(n, i + rows)
        cols = slice(i, n)  # a < b, so a block of rows only meets columns from its own first row on
        d = _hamming_block(H[i:j], H[cols])
        if any_flip:
            d_bf = _hamming_block(H[i:j], F[cols])
            bf = has_flip[cols][None, :]
            if need_row_flip:
                d_af = _hamming_block(F[i:j], H[cols])
                af = has_flip[i:j][:, None]
                d = np.where(bf, np.minimum(d, d_bf), np.where(af, np.minimum(d, d_af), d))
            else:
                d = np.where(bf, np.minimum(d, d_bf), d)
        r = np.arange(i, j)[:, None]
        c = np.arange(i, n)[None, :]
        ok = (c > r) & has[i:j][:, None] & has[cols][None, :] & (d <= near)
        for ra, cb in zip(*np.nonzero(ok), strict=True):
            yield int(i + ra), int(i + cb), int(d[ra, cb])


_STEM_ASSET_VIEW_RE = re.compile(r"^(.+)__(y-?\d+_e-?\d+(?:_.*)?)$")


def _asset_view(item: Item) -> tuple[str | None, str | None]:
    """Return (asset, view) per field: from the record when present, else parsed from paths[0]'s file stem."""
    rec = item.record or {}
    rec_asset = rec.get("asset_id") or None
    rec_view = rec.get("render_id") or None
    stem_asset, stem_view = None, None
    if item.paths:
        stem = os.path.splitext(os.path.basename(item.paths[0]))[0]
        m = _STEM_ASSET_VIEW_RE.match(stem)
        if m:
            stem_asset, stem_view = m.group(1), m.group(2)
    return (rec_asset or stem_asset, rec_view or stem_view)


def _record_priority(rec: dict) -> tuple[int, str]:
    """Deterministic tie-breaker when byte-identical files have different records: approved beats borderline beats
    rejected beats unrecorded, then lexical id."""
    st = rec.get("status")
    rank = 0 if st == "approved" else (1 if st == "borderline" else (2 if st == "rejected" else 3))
    return (rank, str(rec.get("id") or ""))


def load_records(project: str) -> tuple[dict, list]:
    """asset_path -> the record's id, status, explanation (canon.subject, or judgment.explanation ONLY when status is
    approved), judgment_note, failure_modes, prompt, asset_id and render_id; plus the records that couldn't be read."""
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
            "asset_id": prov.get("asset_id") or None,
            "render_id": prov.get("render_id") or None,
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
                    if (
                        row.get("scan_v") == SCAN_V
                        and row.get("sha256")
                        and row.get("dhash")
                        and row.get("dhash_flip")
                        and len(row.get("palette") or []) == 30
                    ):
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


def scan(project: str, strict: bool = False, out: str | None = None, threads: int = 1) -> tuple[list[Item], dict]:
    """Every image under outputs/<folder>/, merged by content. Files at the top of outputs/ (contact sheets and
    composites) are left out, as in the multimodal design. Joins records/*.json, outputs/<wave>/generation.json,
    outputs/<wave>/_curation.json, and embedded ComfyUI PNG tEXt prompt metadata. Reuses <out>/scan.jsonl across
    backends and runs so dHash, dhash_flip and the 30d native-resolution look channel are computed once per unique
    sha256."""
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
                        merged.update({
                            k: v for k, v in cand.items()
                            if v is not None or k in ("explanation", "asset_id", "render_id")
                        })
                        it.record = merged
                    elif it.record.get("id") != cand.get("id"):
                        problems.append(
                            f"{rel}: record collision on byte-identical file with record {it.record.get('id')}"
                        )
                        if _record_priority(cand) < _record_priority(it.record):
                            merged = dict(it.record)
                            for k in ("id", "status", "explanation", "judgment_note", "canon_subject", "source",
                                      "prompt", "asset_id", "render_id", "failure_modes", "recorded"):
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
    uncached = [it for it in by_sha.values() if it.sha256 not in scan_cache]
    computed_by_sha: dict[str, tuple[tuple[str, str, list, dict] | None, str | None, float]] = {}
    if threads > 1 and len(uncached) >= 8:
        paths_to_scan = [os.path.join(project, it.paths[0]) for it in uncached]
        main_mod = sys.modules.get("__main__")
        main_file = getattr(main_mod, "__file__", None) if main_mod is not None else None
        strip_main_file = isinstance(main_file, str) and not os.path.isfile(main_file)
        if strip_main_file:
            delattr(main_mod, "__file__")
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=threads, initializer=_init_scan_worker
            ) as pool:
                for it, res in zip(uncached, pool.map(_scan_worker_one, paths_to_scan), strict=True):
                    computed_by_sha[it.sha256] = res
        finally:
            if strip_main_file and main_mod is not None and main_file is not None:
                main_mod.__file__ = main_file

    for it in by_sha.values():
        cached = scan_cache.get(it.sha256)
        if cached is not None:
            dh = cached["dhash"]
            dh_flip = cached["dhash_flip"]
            pal = cached["palette"]
            png_meta = cached.get("png_meta") or {}
            dt = float(cached.get("look_seconds") or 0.0)
        elif it.sha256 in computed_by_sha:
            stats_res, err_text, dt = computed_by_sha[it.sha256]
            if err_text is not None:
                if strict:
                    _scan_image_stats(os.path.join(project, it.paths[0]))
                    raise OSError(err_text)
                problems.append(f"{it.paths[0]}: unreadable image ({err_text})")
                continue
            assert stats_res is not None
            dh, dh_flip, pal, png_meta = stats_res
            look_seconds += dt
            look_computed_now += 1
        else:
            t0 = time.perf_counter()
            try:
                dh, dh_flip, pal, png_meta = _scan_image_stats(os.path.join(project, it.paths[0]))
            except Exception as e:
                if strict:
                    raise
                problems.append(f"{it.paths[0]}: unreadable image ({e})")
                continue
            dt = time.perf_counter() - t0
            look_seconds += dt
            look_computed_now += 1
        it.dhash, it.dhash_flip, it.palette = dh, dh_flip, pal
        if png_meta:
            if it.record is None:
                it.record = {"id": None, "status": None, "explanation": None,
                             "prompt": png_meta.get("prompt"), "recorded": False}
            for k, v in png_meta.items():
                if v is not None and not it.record.get(k):
                    it.record[k] = v
        valid_items.append(it)
        cache_rows.append({
            "scan_v": SCAN_V,
            "sha256": it.sha256,
            "dhash": dh,
            "dhash_flip": dh_flip,
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

def _default_dim_for_backend(backend: str) -> int:
    if backend in registry.REGISTRY:
        return registry.REGISTRY[backend].dim
    if backend == "fake":
        return 4
    return 0


def _default_vision_tokens_for_backend(backend: str) -> int:
    if backend == "siglip2":
        return 729
    if backend == "dinov2":
        return 256
    if backend in ("eg2", "fake"):
        return 280
    return 0


def cache_dir(
    out: str,
    backend: str,
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> str:
    parts = [backend]
    def_dim = _default_dim_for_backend(backend)
    if dim is not None and dim > 0 and dim != def_dim:
        parts.append(f"d{dim}")
    def_vt = _default_vision_tokens_for_backend(backend)
    if vision_tokens is not None and vision_tokens > 0 and vision_tokens != def_vt:
        parts.append(f"vt{vision_tokens}")
    return os.path.join(out, "-".join(parts))


def save_cache(
    out: str,
    backend: str,
    items: list[Item],
    img: np.ndarray,
    txt: dict,
    meta: dict,
    cluster_txt: dict | None = None,
    *,
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> None:
    d = cache_dir(out, backend, dim=dim, vision_tokens=vision_tokens)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    np.save(os.path.join(d, "images.npy"), img)
    vec_dim = int(img.shape[1]) if img.ndim == 2 else 0
    shas = sorted(txt)
    np.save(os.path.join(d, "texts.npy"),
            np.vstack([txt[s] for s in shas]) if shas else np.zeros((0, vec_dim), np.float32))
    cshas = sorted(cluster_txt or {})
    np.save(os.path.join(d, "cluster_texts.npy"),
            np.vstack([cluster_txt[s] for s in cshas]) if cshas else np.zeros((0, vec_dim), np.float32))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "text_shas": shas, "cluster_text_shas": cshas}, f, indent=1)


def load_cache(
    out: str,
    backend: str,
    *,
    dim: int | None = None,
    vision_tokens: int | None = None,
    project: str | None = None,
) -> tuple[list[Item], np.ndarray, dict, dict]:
    d = cache_dir(out, backend, dim=dim, vision_tokens=vision_tokens)
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(d, "items.jsonl"), encoding="utf-8") as f:
            items = [Item(**json.loads(line)) for line in f]
        img = np.load(os.path.join(d, "images.npy"))
        tv = np.load(os.path.join(d, "texts.npy"))
    except OSError as err:
        proj_arg = f"--project {project} " if project else ""
        raise DeskError(
            "STATE_NO_INDEX",
            f"no {backend} index under {d}; run: dataset-desk index {proj_arg}--backend {backend}",
            cause=err,
        ) from err
    t_shas = meta.pop("text_shas", [])
    c_shas = meta.pop("cluster_text_shas", [])
    c_path = os.path.join(d, "cluster_texts.npy")
    if c_shas and os.path.isfile(c_path):
        ctv = np.load(c_path)
        meta["cluster_txt"] = dict(zip(c_shas, ctv, strict=True))
    return items, img, dict(zip(t_shas, tv, strict=True)), meta


def index(project: str, out: str, emb, limit: int | None = None, grant: str | None = None,
          strict: bool = False, images_only: bool = False, allow_multimodal_pool: bool = False,
          log=print, threads: int = 1) -> dict:
    """Embed every unique image, and every approved record's description as a search query. Reuses vectors already
    cached for the same backend, revision and settings, and shares scan.jsonl across backends."""
    can_text = "text" in getattr(emb, "modalities", ("text", "image"))
    eff_images_only = bool(images_only or emb.name == "none" or not can_text)
    check_multimodal_pool(project, emb.name, images_only=eff_images_only, allow_multimodal_pool=allow_multimodal_pool)
    items, stats = scan(project, strict=strict, out=out, threads=threads)
    if limit:
        items = items[:limit]
    old_img, old_txt, old_ctxt = {}, {}, {}
    vtokens = int(getattr(emb, "vision_tokens", 280))
    emb_dim = int(getattr(emb, "dim", 0))
    max_tokens = int(getattr(emb, "max_tokens", 0))
    dtype_name = getattr(emb, "dtype_name", "float32")
    index_role = "query"
    expected_meta = {
        "revision": emb.revision,
        "dim": emb_dim,
        "vision_tokens": vtokens,
        "max_tokens": max_tokens,
        "max_chunks": TEXT_MAX_CHUNKS,
        "role": index_role,
        "dtype": dtype_name,
    }
    try:
        o_items, o_img, o_txt, o_meta = load_cache(out, emb.name, dim=emb_dim, vision_tokens=vtokens)
        if all(o_meta.get(k) == v for k, v in expected_meta.items()):
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
    if not eff_images_only:
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

    dim = int(getattr(emb, "dim", 0))
    img = np.vstack([old_img[it.sha256] for it in items]) if (items and dim > 0) else np.zeros((len(items), dim), np.float32)
    keep = (
        {it.sha256 for it in items if it.status == "approved" and (it.record or {}).get("explanation")}
        if not eff_images_only
        else set()
    )
    txt = {s: v for s, v in old_txt.items() if s in keep}
    cluster_txt = {s: v for s, v in old_ctxt.items() if s in keep}
    final_dim = int(img.shape[1]) if items else dim
    meta = {
        "backend": emb.name,
        "revision": emb.revision,
        "dim": final_dim,
        "vision_tokens": vtokens,
        "max_tokens": max_tokens,
        "max_chunks": TEXT_MAX_CHUNKS,
        "role": index_role,
        "dtype": getattr(emb, "dtype_name", dtype_name),
        "images_only": eff_images_only,
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
    save_cache(out, emb.name, items, img, txt, meta, cluster_txt=cluster_txt, dim=final_dim, vision_tokens=vtokens)
    log(f"indexed {len(items)} unique images ({len(todo)} new, {t_img:.1f} s) and {len(txt)} descriptions "
        f"with {emb.name}; cache: {cache_dir(out, emb.name, dim=final_dim, vision_tokens=vtokens)}")
    return meta


# ---- the analyses: cache only, except where a query has to be embedded --------------------------------------------

def pairs_at_least(vecs: np.ndarray, thr: float, chunk: int | None = None):
    if vecs.ndim != 2 or vecs.shape[1] == 0:
        return
    rows = _block_rows(len(vecs), vecs.itemsize) if chunk is None else chunk
    for i in range(0, len(vecs), rows):
        sims = vecs[i:i + rows] @ vecs.T
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


def dupes(
    items: list[Item],
    img: np.ndarray,
    near: int = NEAR_BITS,
    semantic: float | None = None,
    backend: str | None = None,
) -> dict:
    exact = [it for it in items if len(it.paths) > 1]
    has_vecs = img.ndim == 2 and img.shape[1] > 0
    views = [_asset_view(it) for it in items]
    skipped_view_pairs: set[tuple[int, int]] = set()

    def _same_asset_diff_view(a: int, b: int) -> bool:
        aa, va = views[a]
        ab, vb = views[b]
        if aa and ab and aa == ab and va and vb and va != vb:
            skipped_view_pairs.add((min(a, b), max(a, b)))
            return True
        return False

    if backend == "dinov2" and semantic is None:
        rule = "dinov2-two-signal"
        near_pairs: list[tuple[int, int, int]] = []
        sem_pairs: list[tuple[int, int, float]] = []
        if has_vecs:
            for a, b, s in pairs_at_least(img, DINOV2_COS_LOW):
                fh = flip_hamming(items[a], items[b]) if (items[a].dhash and items[b].dhash) else 256
                if (s >= DINOV2_COS_LOW and fh <= TWO_SIGNAL_BITS) or s >= DINOV2_COS_HIGH:
                    if not _same_asset_diff_view(a, b):
                        sem_pairs.append((a, b, s))
        thresholds = {
            "near_bits": TWO_SIGNAL_BITS,
            "semantic_cosine": DINOV2_COS_LOW,
            "dinov2_cos_low": DINOV2_COS_LOW,
            "dinov2_cos_high": DINOV2_COS_HIGH,
            "two_signal_bits": TWO_SIGNAL_BITS,
        }
    else:
        rule = "near+cosine"
        eff_semantic = SEMANTIC if semantic is None else float(semantic)
        near_pairs = [(a, b, h) for a, b, h in near_pairs_within(items, near) if not _same_asset_diff_view(a, b)]
        near_set = {(a, b) for a, b, _ in near_pairs}
        sem_pairs = []
        if has_vecs:
            for a, b, s in pairs_at_least(img, eff_semantic):
                if (a, b) in near_set:
                    continue
                if not _same_asset_diff_view(a, b):
                    sem_pairs.append((a, b, s))
        thresholds = {"near_bits": near, "semantic_cosine": eff_semantic}

    return {
        "exact": exact,
        "near": near_pairs,
        "semantic": sem_pairs,
        "semantic_groups": clusters(len(items), [(a, b) for a, b, _ in sem_pairs]),
        "rule": rule,
        "view_pairs_skipped": len(skipped_view_pairs),
        "thresholds": thresholds,
        "top1_percentiles": top1_percentiles(img) if has_vecs else {},
    }


def top1_percentiles(img: np.ndarray, chunk: int | None = None) -> dict:
    """Each image's cosine to its nearest other image, as percentiles: the scale to set the semantic cut on."""
    if len(img) < 2 or img.ndim != 2 or img.shape[1] == 0:
        return {}
    rows = _block_rows(len(img), img.itemsize) if chunk is None else chunk
    best = []
    for i in range(0, len(img), rows):
        sims = img[i:i + rows] @ img.T
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
    XA = img[A]
    kk = min(k, len(A) - 1)
    rows = _block_rows(len(A), XA.itemsize)
    fit = np.empty(len(A), dtype=XA.dtype)
    for i in range(0, len(A), rows):
        S = XA[i:i + rows] @ XA.T
        r = np.arange(S.shape[0])
        S[r, i + r] = -np.inf
        top = np.sort(np.partition(S, S.shape[1] - kk, axis=1)[:, -kk:], axis=1)
        fit[i:i + S.shape[0]] = top.mean(axis=1)
    least = [(A[i], float(fit[i])) for i in np.argsort(fit)[:n]]
    others = [i for i in range(len(items)) if items[i].status != "approved"]
    best = np.zeros(len(others), dtype=XA.dtype)
    for i in range(0, len(others), rows):
        best[i:i + rows] = (img[others[i:i + rows]] @ XA.T).max(axis=1)
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
    kk = min(k, len(A) - 1)
    # Euclidean distance as |x|^2 + |y|^2 - 2 x.y, in float64 and block by block: a matrix product instead of an
    # n x n x 30 difference array, so memory stays bounded and many images take seconds, not minutes.
    X64 = X.astype(np.float64)
    sq = (X64 * X64).sum(axis=1)
    rows = _block_rows(len(A), 8)
    fit = np.empty(len(A), dtype=np.float64)
    for i in range(0, len(A), rows):
        D2 = sq[i:i + rows, None] + sq[None, :] - 2.0 * (X64[i:i + rows] @ X64.T)
        D = np.sqrt(np.maximum(D2, 0.0))
        r = np.arange(D.shape[0])
        D[r, i + r] = np.inf
        fit[i:i + D.shape[0]] = np.sort(np.partition(D, kk - 1, axis=1)[:, :kk], axis=1).mean(axis=1)
    return {A[i]: float(v) for i, v in enumerate(fit)}


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


_RANK_TIE_EPS = 1e-6


def mid_rank(sims: np.ndarray, own: int) -> float:
    """1 + the images scored above the own image + half the ties, so a tie never follows file order."""
    s = float(sims[own])
    above = int((sims > s + _RANK_TIE_EPS).sum())
    equal = max(0, int((np.abs(sims - s) <= _RANK_TIE_EPS).sum()) - 1)
    return 1.0 + float(above) + 0.5 * float(equal)


def _rank_descriptions(
    img: np.ndarray,
    T: np.ndarray,
    own_idx: list[int] | np.ndarray,
    *,
    second_idx: list[int] | np.ndarray | None = None,
    sibling_groups: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Rank stacked description vectors T against img in _block_rows blocks.
    Returns (own_ranks, own_cosines, second_ranks, group_ranks)."""
    m = len(T)
    own_ranks = np.empty(m, dtype=np.float64)
    own_cosines = np.empty(m, dtype=np.float64)
    second_ranks = np.empty(m, dtype=np.float64) if second_idx is not None else None
    group_ranks = np.empty(m, dtype=np.float64) if sibling_groups is not None else None
    if m == 0:
        return own_ranks, own_cosines, second_ranks, group_ranks

    rows = _block_rows(len(img), img.itemsize)
    for i in range(0, m, rows):
        S = T[i:i + rows] @ img.T
        for r in range(S.shape[0]):
            k = i + r
            sims = S[r]
            own = int(own_idx[k])
            s = float(sims[own])
            own_cosines[k] = s
            rk = mid_rank(sims, own)
            own_ranks[k] = rk
            if second_ranks is not None and second_idx is not None:
                second_ranks[k] = mid_rank(sims, int(second_idx[k]))
            if group_ranks is not None and sibling_groups is not None:
                sibs = sibling_groups[k]
                if len(sibs) > 1:
                    sib_scores = sims[sibs]
                    above = int((sims > s + _RANK_TIE_EPS).sum()) - int((sib_scores > s + _RANK_TIE_EPS).sum())
                    equal = max(
                        0,
                        (int((np.abs(sims - s) <= _RANK_TIE_EPS).sum()) - 1)
                        - (int((np.abs(sib_scores - s) <= _RANK_TIE_EPS).sum()) - 1),
                    )
                    group_ranks[k] = 1.0 + float(above) + 0.5 * float(equal)
                else:
                    group_ranks[k] = rk
    return own_ranks, own_cosines, second_ranks, group_ranks


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

    own_idx: list[int] = []
    vecs: list[np.ndarray] = []
    sib_groups: list[list[int]] = []
    for sha, v in txt.items():
        if sha not in pos:
            continue
        i = pos[sha]
        if items[i].status != "approved" or not (items[i].record or {}).get("explanation"):
            continue
        own_idx.append(i)
        vecs.append(v)
        sib_groups.append(desc_groups.get(norm_desc.get(i, ""), [i]))

    if vecs:
        T = np.vstack(vecs)
        own_ranks, own_cosines, _, group_ranks = _rank_descriptions(
            img, T, own_idx, sibling_groups=sib_groups
        )
        assert group_ranks is not None
        rows = [
            {
                "item": own_idx[k],
                "rank": float(own_ranks[k]),
                "group_rank": float(group_ranks[k]),
                "siblings": len(sib_groups[k]),
                "cosine": float(own_cosines[k]),
            }
            for k in range(len(own_idx))
        ]
    else:
        rows = []
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

    T = np.vstack([txt[sh] for sh in shas])
    inv_perm = np.argsort(perm)
    own_idx = [pos[sh] for sh in shas]
    second_idx = [pos[shas[int(inv_perm[k])]] for k in range(n)]
    t, _, second_ranks, _ = _rank_descriptions(img, T, own_idx, second_idx=second_idx)
    assert second_ranks is not None
    w = np.empty(n, dtype=np.float64)
    w[inv_perm] = second_ranks
    t_s = np.sort(t)
    left = np.searchsorted(t_s, w, side="left")    # true ranks below each swapped rank
    right = np.searchsorted(t_s, w, side="right")
    auc = float((left.sum() + 0.5 * (right - left).sum()) / (len(w) * len(t)))
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
    has_vecs = img.ndim == 2 and img.shape[1] > 0
    XA = img[A] if (A and has_vecs) else None
    look = look_fit(items)
    look_rank = {i: r for r, (i, _) in enumerate(sorted(look.items(), key=lambda kv: -kv[1]), start=1)}
    rows = []
    for s, q in zip(subjects, qv, strict=True):
        if XA is not None:
            sims = XA @ q
            top = [(A[i], float(sims[i])) for i in np.argsort(-sims)[:k]]
        else:
            top = []
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
    bname = meta.get("backend", "eg2")
    spec = registry.REGISTRY.get(bname)
    can_text = spec is None or "text" in spec.modalities
    eff_images_only = bool(images_only or meta.get("images_only") or bname == "none" or not can_text)
    check_multimodal_pool(project, bname, images_only=eff_images_only,
                          allow_multimodal_pool=allow_multimodal_pool)
    has_vecs = img.ndim == 2 and img.shape[1] > 0
    d = dupes(items, img, backend=meta.get("backend"))
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


# ---- text and code input: --source and --rnd-data ---------------------------------------------------------------

@dataclass
class TextItem:
    sha256: str
    paths: list[str]
    name: str = ""
    suffix: str = ""
    run_id: str | None = None
    outcome: str | None = None
    occurrences: list[dict] = field(default_factory=list)
    chunks: int = 1


def _has_nul_first_4kb(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\x00" in f.read(NUL_PROBE_BYTES)


_TOKEN_SPAN_RE = re.compile(r"\S{1,6}\s*|\s+")


class _SpanTokenizer:
    """Deterministic span tokenizer used when an offline/stand-in embedder has no HF tokenizer attached."""

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN_SPAN_RE.finditer(text)]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(self.spans(text))))

    def __call__(self, text: str, **_kw) -> dict:
        return {"input_ids": self.encode(text)}


_FALLBACK_TOKENIZER = _SpanTokenizer()


def _resolve_tokenizer(emb=None, tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    if emb is None:
        return _FALLBACK_TOKENIZER
    tok = getattr(emb, "tokenizer", None)
    if tok is not None:
        return tok
    tok = backends._extract_model_tokenizer(getattr(emb, "model", None), getattr(emb, "processor", None))
    if tok is not None:
        return tok
    emb_name = getattr(emb, "name", None)
    if emb_name in ("fake", "none") or type(emb) in (backends.FakeEmbedder, backends.NoneEmbedder):
        return _FALLBACK_TOKENIZER
    raise DeskError(
        "INPUT_BAD_ARGUMENT",
        f"REFUSED: backend {emb_name or type(emb).__name__!r} has no tokenizer for token-counted chunking",
    )


def _encode_token_ids(tok, text: str) -> list:
    if hasattr(tok, "encode"):
        try:
            return list(tok.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tok.encode(text))
    if callable(tok):
        try:
            res = tok(text, add_special_tokens=False, truncation=False, padding=False)
        except TypeError:
            res = tok(text)
        return list(res["input_ids"] if isinstance(res, dict) else res)
    return _FALLBACK_TOKENIZER.encode(text)


def chunk_text(
    text: str,
    emb=None,
    *,
    max_tokens: int | None = None,
    prefix: str = "",
    prefix_tokens: int | None = None,
    tokenizer=None,
    max_chunks: int = TEXT_MAX_CHUNKS,
) -> list[str]:
    """Split `text` into at most `max_chunks` chunks of at most `max_tokens - prefix_tokens - 4` tokens."""
    if not text:
        return [""]
    if isinstance(emb, int) and max_tokens is None:
        eff_max_tokens = emb
        emb = None
    elif max_tokens is not None:
        eff_max_tokens = int(max_tokens)
    elif emb is not None and getattr(emb, "max_tokens", 0) > 0:
        eff_max_tokens = int(emb.max_tokens)
    else:
        eff_max_tokens = registry.REGISTRY["bge-m3"].max_tokens

    tok = _resolve_tokenizer(emb, tokenizer)
    if prefix_tokens is not None:
        eff_prefix_tokens = max(0, int(prefix_tokens))
    elif prefix:
        eff_prefix_tokens = len(_encode_token_ids(tok, prefix))
    else:
        eff_prefix_tokens = 0

    chunk_budget = max(1, eff_max_tokens - eff_prefix_tokens - 4)

    if tok is _FALLBACK_TOKENIZER:
        spans = tok.spans(text)
        if len(spans) <= chunk_budget:
            return [text]
        chunks: list[str] = []
        i = 0
        while i < len(spans) and len(chunks) < max_chunks:
            end_idx = min(i + chunk_budget, len(spans))
            chunks.append(text[spans[i][0]:spans[end_idx - 1][1]])
            i = end_idx
        return chunks

    if hasattr(tok, "encode") and hasattr(tok, "decode"):
        ids = _encode_token_ids(tok, text)
        if len(ids) <= chunk_budget:
            return [text]
        chunks = []
        i = 0
        while i < len(ids) and len(chunks) < max_chunks:
            sub_ids = ids[i:i + chunk_budget]
            try:
                ch = tok.decode(sub_ids, skip_special_tokens=True)
            except TypeError:
                ch = tok.decode(sub_ids)
            chunks.append(ch)
            i += chunk_budget
        return chunks

    spans = _FALLBACK_TOKENIZER.spans(text)
    if len(_encode_token_ids(tok, text)) <= chunk_budget or not spans:
        return [text]
    chunks = []
    i = 0
    while i < len(spans) and len(chunks) < max_chunks:
        lo = i + 1
        hi = min(i + chunk_budget, len(spans))
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[spans[i][0]:spans[mid - 1][1]]
            if len(_encode_token_ids(tok, cand)) <= chunk_budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunks.append(text[spans[i][0]:spans[best - 1][1]])
        i = best
    return chunks


def embed_file_chunks(emb, chunks: list[str], role: str = "document", title: str | None = None) -> np.ndarray:
    """Embed a file's chunks and return their unit-normalized mean vector."""
    if not chunks:
        chunks = [""]
    titles = [title] * len(chunks) if (title is not None and role == "document") else None
    vecs = emb.texts(chunks, role, titles=titles) if titles is not None else emb.texts(chunks, role)
    if vecs.ndim != 2 or vecs.shape[1] == 0:
        return np.zeros((0,), np.float32)
    return backends.unit_rows(vecs.mean(axis=0))[0]


def _check_not_private_path(path: str) -> None:
    for candidate in (os.path.abspath(path), os.path.realpath(os.path.abspath(path))):
        parts = [p.lower() for p in pathlib.Path(candidate).parts]
        if "rnd-data-private" in parts or "blobs" in parts:
            raise DeskError("PERM_PRIVATE_SOURCE", f"REFUSED: {path!r} names a private or blob store")


def _check_out_not_inside(source_dir: str, out_dir: str) -> None:
    try:
        inside = os.path.commonpath([os.path.abspath(source_dir), os.path.abspath(out_dir)]) == os.path.abspath(source_dir)
    except ValueError:
        inside = False
    if inside:
        raise DeskError("PERM_OUT_INSIDE_PROJECT", "REFUSED: --out is inside the source folder; the desk never writes into a source")


def scan_text_source(source: str) -> tuple[list[TextItem], dict]:
    """Scan every file under `source` whose suffix is in TEXT_EXT, skipping NUL-containing files, distinct by sha256."""
    _check_not_private_path(source)
    src_abs = os.path.realpath(os.path.abspath(source))
    if not os.path.isdir(src_abs):
        raise DeskError("INPUT_BAD_ARGUMENT", f"source folder {source!r} does not exist")
    by_sha: dict[str, TextItem] = {}
    seen_files = 0
    nul_skipped = 0
    for root, dirs, files in os.walk(src_abs):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXT:
                continue
            full = os.path.join(root, fname)
            real = os.path.realpath(full)
            try:
                in_root = os.path.commonpath([src_abs, real]) == src_abs
            except ValueError:
                in_root = False
            if not in_root:
                raise DeskError("PERM_PRIVATE_SOURCE", f"REFUSED: path {fname!r} resolves outside source folder")
            if _has_nul_first_4kb(real):
                nul_skipped += 1
                continue
            seen_files += 1
            rel = os.path.relpath(full, src_abs).replace("\\", "/")
            sha = sha256_file(real)
            it = by_sha.get(sha)
            if it is None:
                it = TextItem(sha256=sha, paths=[], name=fname, suffix=ext)
                by_sha[sha] = it
            it.paths.append(rel)
            it.occurrences.append({"path": rel, "run_id": None, "name": fname, "suffix": ext, "outcome": None})
    return list(by_sha.values()), {"files": seen_files, "unique": len(by_sha), "nul_skipped": nul_skipped}


def _read_run_outcome(run_dir: str) -> str | None:
    outcome_path = os.path.join(run_dir, "files", "outcome.json")
    if not os.path.isfile(outcome_path):
        return None
    try:
        with open(outcome_path, encoding="utf-8") as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    run_sub = obj.get("run")
    if isinstance(run_sub, dict) and run_sub.get("outcome") is not None:
        return str(run_sub["outcome"])
    if obj.get("outcome") is not None:
        return str(obj["outcome"])
    return None


def scan_rnd_data(checkout: str) -> tuple[list[TextItem], dict]:
    """Read rnd-data's public files from runs/*/manifest.json copies whose store is 'rnd-data'."""
    _check_not_private_path(checkout)
    checkout_abs = os.path.realpath(os.path.abspath(checkout))
    runs_dir = os.path.join(checkout_abs, "runs")
    if not os.path.isdir(runs_dir):
        raise DeskError("PERM_PRIVATE_SOURCE", f"REFUSED: {checkout!r} has no runs/ directory")

    by_sha: dict[str, TextItem] = {}
    seen_files = 0
    nul_skipped = 0
    runs_seen = 0

    for run_folder in sorted(os.listdir(runs_dir)):
        rdir = os.path.join(runs_dir, run_folder)
        if not os.path.isdir(rdir):
            continue
        mpath = os.path.join(rdir, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            raise DeskError("INPUT_BAD_ARGUMENT", f"invalid manifest {mpath}", cause=err) from err

        if not isinstance(manifest, dict):
            continue
        runs_seen += 1
        run_id = str(manifest.get("id") or run_folder)
        if "rnd-data-private" in str(manifest.get("source") or "").lower():
            raise DeskError("PERM_PRIVATE_SOURCE", f"REFUSED: manifest {run_id!r} names rnd-data-private")

        outcome = _read_run_outcome(rdir)
        for copy in manifest.get("copies") or []:
            if not isinstance(copy, dict):
                continue
            store = str(copy.get("store") or "")
            raw_loc = str(copy.get("location") or "")
            loc = raw_loc.replace("\\", "/")

            if "rnd-data-private" in store.lower() or "private" in store.lower() or "rnd-data-private" in loc.lower():
                raise DeskError(
                    "PERM_PRIVATE_SOURCE",
                    f"REFUSED: manifest {run_id!r} names private store or path ({store!r}, {raw_loc!r})",
                )
            if os.path.isabs(raw_loc) or os.path.isabs(loc) or ".." in loc.split("/"):
                raise DeskError(
                    "PERM_PRIVATE_SOURCE",
                    f"REFUSED: manifest {run_id!r} location {raw_loc!r} is outside the checkout",
                )
            if store != "rnd-data":
                continue
            if not loc.startswith("runs/"):
                raise DeskError(
                    "PERM_PRIVATE_SOURCE",
                    f"REFUSED: manifest {run_id!r} rnd-data copy {raw_loc!r} is outside runs/",
                )
            candidate = os.path.join(checkout_abs, loc)
            real = os.path.realpath(os.path.abspath(candidate))
            try:
                in_runs = os.path.commonpath([runs_dir, real]) == runs_dir
            except ValueError:
                in_runs = False
            if not in_runs:
                raise DeskError(
                    "PERM_PRIVATE_SOURCE",
                    f"REFUSED: manifest {run_id!r} copy {raw_loc!r} escapes checkout runs/",
                )
            if not os.path.isfile(real):
                raise DeskError(
                    "PERM_PRIVATE_SOURCE",
                    f"REFUSED: manifest {run_id!r} copy {raw_loc!r} is missing from checkout",
                )
            fname = os.path.basename(loc)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXT:
                continue
            if _has_nul_first_4kb(real):
                nul_skipped += 1
                continue
            seen_files += 1
            sha = sha256_file(real)
            it = by_sha.get(sha)
            if it is None:
                it = TextItem(sha256=sha, paths=[], name=fname, suffix=ext, run_id=run_id, outcome=outcome)
                by_sha[sha] = it
            it.paths.append(loc)
            it.occurrences.append({
                "path": loc,
                "run_id": run_id,
                "name": fname,
                "suffix": ext,
                "outcome": outcome,
            })

    return list(by_sha.values()), {
        "files": seen_files,
        "unique": len(by_sha),
        "runs": runs_seen,
        "nul_skipped": nul_skipped,
    }


def text_cache_dir(
    out: str,
    backend: str,
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> str:
    def_dim = _default_dim_for_backend(backend)
    folder = f"{backend}-d{dim}" if (dim is not None and dim > 0 and dim != def_dim) else backend
    return os.path.join(out, folder, "text")


def save_text_cache(
    out: str,
    backend: str,
    items: list[TextItem],
    vecs: np.ndarray,
    meta: dict,
    *,
    symmetric_vecs: np.ndarray | None = None,
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> None:
    d = text_cache_dir(out, backend, dim=dim, vision_tokens=vision_tokens)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    np.save(os.path.join(d, "texts.npy"), vecs)
    sym_path = os.path.join(d, "texts_symmetric.npy")
    if symmetric_vecs is not None:
        np.save(sym_path, symmetric_vecs)
    elif os.path.isfile(sym_path):
        os.remove(sym_path)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def load_text_cache(
    out: str,
    backend: str,
    *,
    role: str = "document",
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> tuple[list[TextItem], np.ndarray, dict]:
    d = text_cache_dir(out, backend, dim=dim, vision_tokens=vision_tokens)
    try:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(d, "items.jsonl"), encoding="utf-8") as f:
            items = [TextItem(**json.loads(line)) for line in f if line.strip()]
        sym_path = os.path.join(d, "texts_symmetric.npy")
        needs_symmetric = role == "symmetric" and (
            backend in ("eg2", "fake") or meta.get("role") == "document+symmetric"
        )
        if needs_symmetric:
            if not os.path.isfile(sym_path):
                raise DeskError(
                    "STATE_NO_INDEX",
                    f"no symmetric vectors in {backend} text index under {d}; rerun: dataset-desk text-index --backend {backend}",
                )
            vec_file = sym_path
        elif role == "symmetric" and os.path.isfile(sym_path):
            vec_file = sym_path
        else:
            vec_file = os.path.join(d, "texts.npy")
        vecs = np.load(vec_file)
    except OSError as err:
        raise DeskError(
            "STATE_NO_INDEX",
            f"no {backend} text index under {d}; run: dataset-desk text-index --backend {backend}",
            cause=err,
        ) from err
    return items, vecs, meta


def text_index(
    out: str,
    emb,
    *,
    source: str | None = None,
    rnd_data: str | None = None,
    limit: int | None = None,
    grant: str | None = None,
    log=print,
) -> dict:
    """Embed text/code files from --source or --rnd-data into <out>/<backend>/text/, reusing cached vectors by sha256
    and settings."""
    if bool(source) == bool(rnd_data):
        raise DeskError("INPUT_BAD_ARGUMENT", "text-index needs either --source <folder> or --rnd-data <checkout>")
    root_dir = os.path.realpath(os.path.abspath(source or rnd_data))
    _check_out_not_inside(root_dir, out)
    items, stats = scan_text_source(root_dir) if source else scan_rnd_data(root_dir)
    if limit:
        items = items[:limit]

    is_prefix_sensitive = bool(getattr(emb, "has_symmetric_vectors", False) or emb.name in ("eg2", "fake"))
    emb_dim = int(getattr(emb, "dim", 0))
    vtokens = int(getattr(emb, "vision_tokens", 0))
    max_tokens = int(getattr(emb, "max_tokens", 0))
    dtype_name = getattr(emb, "dtype_name", "float32")
    role_meta = "document+symmetric" if is_prefix_sensitive else "document"
    expected_meta = {
        "revision": emb.revision,
        "dim": emb_dim,
        "vision_tokens": vtokens,
        "max_tokens": max_tokens,
        "max_chunks": TEXT_MAX_CHUNKS,
        "role": role_meta,
        "dtype": dtype_name,
    }

    old_vecs: dict[str, np.ndarray] = {}
    old_sym_vecs: dict[str, np.ndarray] = {}
    old_chunks: dict[str, int] = {}
    try:
        o_items, o_vecs, o_meta = load_text_cache(out, emb.name, role="document", dim=emb_dim)
        if all(o_meta.get(k) == v for k, v in expected_meta.items()):
            if is_prefix_sensitive:
                sym_file = os.path.join(text_cache_dir(out, emb.name, dim=emb_dim), "texts_symmetric.npy")
                if os.path.isfile(sym_file):
                    o_sym = np.load(sym_file)
                    old_vecs = dict(zip((it.sha256 for it in o_items), o_vecs, strict=True))
                    old_sym_vecs = dict(zip((it.sha256 for it in o_items), o_sym, strict=True))
                    old_chunks = {it.sha256: it.chunks for it in o_items}
            else:
                old_vecs = dict(zip((it.sha256 for it in o_items), o_vecs, strict=True))
                old_chunks = {it.sha256: it.chunks for it in o_items}
    except SystemExit:
        pass

    t0 = time.time()
    todo = [it for it in items if it.sha256 not in old_vecs]
    for it in items:
        if it.sha256 in old_chunks:
            it.chunks = old_chunks[it.sha256]

    read_chars = (max_tokens or registry.REGISTRY["bge-m3"].max_tokens) * TEXT_MAX_CHUNKS * 6 + 1024
    if todo and emb.name != "none":
        for it in todo:
            full = os.path.join(root_dir, it.paths[0])
            with open(full, encoding="utf-8", errors="replace") as f:
                raw = f.read(read_chars)
            doc_prefix = backends.format_eg2_prefix("document", it.name) if is_prefix_sensitive else ""
            ch_doc = chunk_text(raw, emb, prefix=doc_prefix)
            it.chunks = len(ch_doc)
            old_vecs[it.sha256] = embed_file_chunks(emb, ch_doc, role="document", title=it.name)
            if is_prefix_sensitive:
                sym_prefix = backends.format_eg2_prefix("symmetric")
                ch_sym = chunk_text(raw, emb, prefix=sym_prefix)
                old_sym_vecs[it.sha256] = embed_file_chunks(emb, ch_sym, role="symmetric", title=None)

    t_sec = time.time() - t0
    dim = int(getattr(emb, "dim", 0))
    if items and emb.name != "none" and old_vecs:
        vecs = np.vstack([old_vecs[it.sha256] for it in items])
        dim = int(vecs.shape[1])
        sym_vecs = np.vstack([old_sym_vecs[it.sha256] for it in items]) if is_prefix_sensitive else None
    else:
        vecs = np.zeros((len(items), dim), np.float32)
        sym_vecs = np.zeros((len(items), dim), np.float32) if (is_prefix_sensitive and emb.name != "none") else None

    meta = {
        "backend": emb.name,
        "revision": emb.revision,
        "dim": dim,
        "vision_tokens": vtokens,
        "max_tokens": max_tokens,
        "max_chunks": TEXT_MAX_CHUNKS,
        "role": role_meta,
        "dtype": getattr(emb, "dtype_name", dtype_name),
        "mode": "source" if source else "rnd-data",
        "grant": grant,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "embedded_now": len(todo),
        "text_seconds": round(t_sec, 2),
        **stats,
    }
    save_text_cache(out, emb.name, items, vecs, meta, symmetric_vecs=sym_vecs, dim=dim)
    log(
        f"indexed {len(items)} unique text files ({len(todo)} new, {t_sec:.1f} s) "
        f"with {emb.name}; cache: {text_cache_dir(out, emb.name, dim=dim)}"
    )
    return meta


def text_dupes(
    items: list[TextItem],
    vecs: np.ndarray,
    semantic: float | None = None,
    task: str = "text",
    backend: str | None = None,
) -> dict:
    """Exact and semantic duplicates at or above `semantic` (defaults to CODE_SEMANTIC for code on eg2,
    TEXT_SEMANTIC for text on bge-m3); paths, scores, labels and top1_percentiles only."""
    if task == "code":
        keep = [
            i for i, it in enumerate(items)
            if (it.suffix or os.path.splitext(it.name or (it.paths[0] if it.paths else ""))[1]).lower() in CODE_EXT
        ]
        items = [items[i] for i in keep]
        if vecs.ndim == 2 and len(vecs) > 0:
            vecs = vecs[keep]

    if semantic is not None:
        eff_semantic = float(semantic)
    else:
        calibrated = "eg2" if task == "code" else "bge-m3"
        if backend is not None and backend != calibrated:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"REFUSED: backend {backend!r} has no calibrated threshold for task {task!r}; pass --semantic",
            )
        eff_semantic = CODE_SEMANTIC if task == "code" else TEXT_SEMANTIC

    exact = [it.paths for it in items if len(it.paths) > 1]
    has_vecs = vecs.ndim == 2 and vecs.shape[1] > 0
    sem_raw = list(pairs_at_least(vecs, eff_semantic)) if has_vecs else []
    pairs = [
        {
            "a": items[a].paths[0],
            "b": items[b].paths[0],
            "cosine": round(s, 4),
            "run_a": items[a].run_id,
            "run_b": items[b].run_id,
            "outcome_a": items[a].outcome,
            "outcome_b": items[b].outcome,
        }
        for a, b, s in sem_raw
    ]
    semantic_tuples = [(items[a].paths[0], items[b].paths[0], round(s, 4)) for a, b, s in sem_raw]
    sem_groups = [
        [items[i].paths[0] for i in g]
        for g in clusters(len(items), [(a, b) for a, b, _ in sem_raw])
    ]
    return {
        "exact": exact,
        "pairs": pairs,
        "semantic": semantic_tuples,
        "semantic_groups": sem_groups,
        "threshold": eff_semantic,
        "top1_percentiles": top1_percentiles(vecs) if has_vecs else {},
    }


def text_search(items: list[TextItem], vecs: np.ndarray, q: np.ndarray, k: int = 10) -> list[dict]:
    """Nearest text/code files to query vector `q`; returns paths, scores, and outcome labels only."""
    hits = search(vecs, q, k=k)
    return [
        {
            "item": i,
            "path": items[i].paths[0],
            "name": items[i].name,
            "suffix": items[i].suffix,
            "run_id": items[i].run_id,
            "outcome": items[i].outcome,
            "cosine": round(score, 4),
        }
        for i, score in hits
    ]


def _lineage_top_k(occs: list[dict], vecs: np.ndarray | None, k: int) -> tuple[list[tuple], bool]:
    """The k best pairs of one file name's occurrences, in bounded memory and bounded Python work.

    Returns (heap, any_pair). Each heap entry is (rcos, -path_rank_a, -path_rank_b, -a, -b) with a < b occurrence
    indices, so heap[0] is the worst pair kept and the order matches 0.2.0's stable sort on (-cosine, a, b). Rows are
    pruned against heap[0] before Python touches a cell, and a surviving row offers at most k cells."""
    m = len(occs)
    item_indices = np.array([o["item_idx"] for o in occs], dtype=np.int64)
    _, path_rank = np.unique([o["path"] for o in occs], return_inverse=True)
    path_rank = path_rank.reshape(-1).astype(np.int64)
    run_codes: dict = {}
    run_int = np.array(
        [-1 if o["run_id"] is None else run_codes.setdefault(o["run_id"], len(run_codes)) for o in occs],
        dtype=np.int64,
    )
    V_occ = vecs[item_indices] if vecs is not None else None
    col_idx = np.arange(m)
    heap: list[tuple] = []
    any_pair = False
    rows = _block_rows(m, 8)
    for i in range(0, m, rows):
        ie = min(m, i + rows)
        if V_occ is not None:
            sims = np.round((V_occ[i:ie] @ V_occ.T).astype(np.float64), 4)
        else:
            sims = np.zeros((ie - i, m), dtype=np.float64)
        sims[item_indices[i:ie, None] == item_indices[None, :]] = 1.0
        blocked = col_idx[None, :] <= col_idx[i:ie, None]
        blocked |= (run_int[i:ie, None] >= 0) & (run_int[i:ie, None] == run_int[None, :])
        blocked |= path_rank[i:ie, None] == path_rank[None, :]
        sims[blocked] = -np.inf
        del blocked
        row_max = sims.max(axis=1)
        for r in np.lexsort((np.arange(ie - i), path_rank[i:ie], -row_max)):
            rmax = row_max[r]
            if rmax == -np.inf:
                continue
            any_pair = True
            a = i + int(r)
            pa = int(path_rank[a])
            row = sims[r]
            if len(heap) < k:
                cols = np.flatnonzero(row > -np.inf)
            else:
                if k < 1:
                    continue
                floor = heap[0][0]
                max_pa_rank = -heap[0][1]
                if rmax < floor or (rmax == floor and pa > max_pa_rank):
                    continue
                cols = np.flatnonzero(row > floor) if pa > max_pa_rank else np.flatnonzero(row >= floor)
            cols = cols[np.lexsort((cols, path_rank[cols], -row[cols]))[:k]]
            for b in cols:
                b = int(b)
                cand = (float(row[b]), -pa, -int(path_rank[b]), -a, -b)
                if len(heap) < k:
                    heapq.heappush(heap, cand)
                elif cand > heap[0]:
                    heapq.heapreplace(heap, cand)
                else:
                    break  # the row's cells come best first, so none after this one can enter
    return heap, any_pair


def lineage(items: list[TextItem], vecs: np.ndarray, k: int = 5) -> dict:
    """For code files, find the nearest same-name versions across runs (paths, scores, and labels only)."""
    has_vecs = vecs.ndim == 2 and vecs.shape[1] > 0
    by_name: dict[str, list[dict]] = {}
    for idx, it in enumerate(items):
        occs = it.occurrences or [
            {"path": p, "run_id": it.run_id, "name": os.path.basename(p), "suffix": it.suffix, "outcome": it.outcome}
            for p in it.paths
        ]
        for occ in occs:
            ext = (occ.get("suffix") or os.path.splitext(occ.get("name") or "")[1]).lower()
            if ext not in CODE_EXT:
                continue
            fname = occ.get("name") or os.path.basename(occ["path"])
            by_name.setdefault(fname, []).append({**occ, "item_idx": idx})

    files_out: dict[str, list[dict]] = {}
    all_pairs: list[dict] = []
    for fname in sorted(by_name):
        occs = by_name[fname]
        m = len(occs)
        if m < 2:
            continue
        heap, any_pair = _lineage_top_k(occs, vecs if has_vecs else None, k)
        if any_pair:
            top_k = [
                {
                    "name": fname,
                    "a": occs[-na]["path"],
                    "b": occs[-nb]["path"],
                    "run_a": occs[-na]["run_id"],
                    "run_b": occs[-nb]["run_id"],
                    "outcome_a": occs[-na]["outcome"],
                    "outcome_b": occs[-nb]["outcome"],
                    "cosine": rcos,
                }
                for rcos, _, _, na, nb in sorted(heap, reverse=True)
            ]
            files_out[fname] = top_k
            all_pairs.extend(top_k)

    all_pairs.sort(key=lambda r: (-r["cosine"], r["name"], r["a"], r["b"]))
    return {"files": files_out, "pairs": all_pairs}


def command_task(command: str, task: str | None = None) -> str:
    """Return the task name for a CLI command unless overridden by `--task`."""
    if task:
        return task
    if command == "dupes":
        return "dupes"
    if command == "lineage":
        return "code"
    if command in ("text-index", "text-dupes", "text-search"):
        return "text"
    return "images"


def command_backend(command: str, backend: str | None = None, task: str | None = None) -> str:
    """Resolve the embedder backend for `command`: `--backend` overrides the task's default."""
    return registry.resolve_backend(backend, command_task(command, task))


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
    image_cmds = ("index", "dupes", "search", "coverage", "report")
    text_cmds = ("text-index", "text-dupes", "text-search", "lineage")
    all_cmds = [*image_cmds, *text_cmds]
    all_backends = [*registry.REGISTRY, "none", "fake"]

    ap = argparse.ArgumentParser(
        prog="dataset-desk", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("commands:\n"
                "  index        scan the project and embed its images (and descriptions) into the cache under --out\n"
                "  dupes        exact, near and semantic duplicates, as JSON\n"
                "  search       the nearest images to --text, --image, or both\n"
                "  coverage     the best images for each subject in --subjects\n"
                "  report       the Markdown report: duplicates, outliers, descriptions, coverage\n"
                "  text-index   scan --source or --rnd-data and embed text/code files into <out>/<backend>/text/\n"
                "  text-dupes   text pairs at or above --semantic, as JSON\n"
                "  text-search  the nearest text/code files to --text\n"
                "  lineage      for code files, the nearest same-name versions across runs\n\n"
                "exit codes: 0 ok, 1 user error or refusal, 2 unexpected error"))
    ap.add_argument("command", choices=all_cmds, help="what to do")
    ap.add_argument("--version", action="version", version=f"dataset-desk {__version__}")
    ap.add_argument("--project", help="a style-dataset-lab project folder (read only)")
    ap.add_argument("--source", help="a folder of text and code files (read only)")
    ap.add_argument("--rnd-data", dest="rnd_data", help="an rnd-data checkout folder (reads public rnd-data copies only)")
    ap.add_argument("--task", choices=list(registry.TASK_DEFAULTS),
                    help="task whose default backend to use when --backend is omitted")
    ap.add_argument("--backend", default=None, choices=all_backends,
                    help="embedder backend (defaults to the command/task default in registry.TASK_DEFAULTS)")
    ap.add_argument("--out", help="where the cache and report go (default: desk-out/<name> here)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (on a shared machine, half the cores at most)")
    ap.add_argument("--device", default="cpu", help="cpu (default) or a torch device such as cuda; needs --grant")
    ap.add_argument("--grant", help="a GPU grant id from whoever schedules the card; required for any device but cpu")
    ap.add_argument("--vram-budget", type=int, default=backends.DEFAULT_VRAM_BUDGET_MIB,
                    help=f"VRAM ceiling in MiB (default: {backends.DEFAULT_VRAM_BUDGET_MIB})")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="seconds to poll (every 10 s) for free VRAM before refusing with PERM_GPU_BUSY")
    ap.add_argument("--dim", type=int, default=None,
                    help="Matryoshka embedding dimension for eg2/fake (768, 512, 256; 128 refused)")
    ap.add_argument("--vision-tokens", dest="vision_tokens", type=int, default=None,
                    help="vision soft tokens per image for eg2/fake (70, 140, 280, 560, 1120; default: 280)")
    ap.add_argument("--semantic", type=float, default=None,
                    help="cosine threshold override for semantic duplicates (default: DINOv2 two-signal or 0.95 for "
                         "dupes, 0.98 for text-dupes on bge-m3, 0.95 for --task code on eg2)")
    ap.add_argument("--limit", type=int, help="index only the first N unique items (a smoke test)")
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
    ap.add_argument("--text", help="search/text-search: a query string (combine with --image on eg2/fake)")
    ap.add_argument("--image", help="search: an image file")
    ap.add_argument("--subjects", help="coverage/report: a file with one wanted subject per line")
    ap.add_argument("-k", type=int, default=10, help="search: how many results")
    ap.add_argument("--debug", action="store_true", help="show stack traces for unexpected errors")
    a = ap.parse_args(argv)

    backend = command_backend(a.command, a.backend, a.task)
    backends.check_device(a.device, a.grant)
    backends.validate_embedder_options(backend, dim=a.dim, vision_tokens=a.vision_tokens)
    backends.cap_threads(a.threads)

    make_kw: dict = {"device": a.device, "threads": a.threads, "grant": a.grant}
    if a.vram_budget != backends.DEFAULT_VRAM_BUDGET_MIB:
        make_kw["vram_budget"] = a.vram_budget
    if a.wait:
        make_kw["wait"] = a.wait
    if a.dim is not None:
        make_kw["dim"] = a.dim
    if a.vision_tokens is not None:
        make_kw["vision_tokens"] = a.vision_tokens

    if a.command in text_cmds:
        make_kw["text_only"] = True
        if a.source and a.rnd_data:
            raise DeskError("INPUT_BAD_ARGUMENT", "pass either --source or --rnd-data, not both")
        src_root = os.path.abspath(a.source or a.rnd_data) if (a.source or a.rnd_data) else None
        if src_root:
            _check_not_private_path(src_root)
        if not a.out and not src_root:
            raise DeskError("INPUT_BAD_ARGUMENT", f"{a.command} needs --out (or --source / --rnd-data)")
        out = os.path.abspath(a.out or os.path.join("desk-out", os.path.basename(src_root)))
        if src_root:
            _check_out_not_inside(src_root, out)
        if backend == "none" and a.command == "text-search":
            raise DeskError("INPUT_BAD_ARGUMENT", "text-search requires an embedding backend, not 'none'")
        needs_text_model = a.command in ("text-index", "text-search")
        emb = None
        if needs_text_model:
            if backend not in ("fake", "none"):
                lower_priority()
            emb = make(backend, **make_kw)
        if a.command == "text-index":
            text_index(out, emb, source=a.source, rnd_data=a.rnd_data, limit=a.limit, grant=a.grant)
            return 0
        read_role = "document" if a.command == "text-search" else "symmetric"
        t_items, t_vecs, t_meta = load_text_cache(out, backend, role=read_role, dim=a.dim)
        if a.command == "text-dupes":
            t_task = command_task(a.command, a.task)
            print(json.dumps(text_dupes(t_items, t_vecs, semantic=a.semantic, task=t_task,
                                        backend=t_meta.get("backend") or backend), indent=1))
        elif a.command == "text-search":
            if not a.text:
                raise DeskError("INPUT_BAD_ARGUMENT", "text-search needs --text")
            q_role = "code_query" if command_task(a.command, a.task) == "code" else "query"
            q = emb.texts([a.text], q_role)[0]
            for hit in text_search(t_items, t_vecs, q, k=a.k):
                label = hit["outcome"] or hit["run_id"] or "unrecorded"
                print(f"{hit['cosine']:.4f}  {label:12s}  {hit['path']}")
        elif a.command == "lineage":
            print(json.dumps(lineage(t_items, t_vecs, k=a.k), indent=1))
        return 0

    if not a.project:
        ap.error("the following arguments are required: --project")
    project = os.path.abspath(a.project)
    out = os.path.abspath(a.out or os.path.join("desk-out", os.path.basename(project)))
    try:
        out_inside = os.path.commonpath([project, out]) == project
    except ValueError:  # different drives on Windows: --out cannot be inside the project
        out_inside = False
    if out_inside:
        raise DeskError("PERM_OUT_INSIDE_PROJECT", "REFUSED: --out is inside the project; the desk never writes into a project")
    if a.command in ("index", "report"):
        check_multimodal_pool(project, backend, images_only=a.images_only,
                              allow_multimodal_pool=a.allow_multimodal_pool)
    if backend == "none" and a.command in ("search", "coverage"):
        raise DeskError("INPUT_BAD_ARGUMENT",
                        f"command {a.command!r} requires an embedding backend (eg2, siglip2, or fake), not 'none'")
    needs_model = a.command in ("index", "search", "coverage") or (a.command == "report" and a.subjects)
    emb = None
    if needs_model:
        if backend not in ("fake", "none"):
            lower_priority()
        emb = make(backend, **make_kw)
    if a.command == "index":
        index(project, out, emb, limit=a.limit, grant=a.grant, strict=a.strict,
              images_only=a.images_only, allow_multimodal_pool=a.allow_multimodal_pool,
              threads=a.threads)
        return 0
    items, img, txt, meta = load_cache(out, backend, dim=a.dim, vision_tokens=a.vision_tokens, project=a.project)
    subjects = None
    if a.subjects:
        with open(a.subjects, encoding="utf-8") as f:
            subjects = [s.strip() for s in f if s.strip() and not s.startswith("#")]
    if a.command == "dupes":
        d = dupes(items, img, semantic=a.semantic, backend=meta.get("backend") or backend)
        print(json.dumps({"exact": [it.paths for it in d["exact"]],
                          "near": [(items[x].paths[0], items[y].paths[0], h) for x, y, h in d["near"]],
                          "semantic": [(items[x].paths[0], items[y].paths[0], round(s, 4)) for x, y, s in d["semantic"]],
                          "rule": d["rule"],
                          "view_pairs_skipped": d["view_pairs_skipped"],
                          "thresholds": d["thresholds"], "top1_percentiles": d["top1_percentiles"]}, indent=1))
    elif a.command == "search":
        if not a.text and not a.image:
            raise DeskError("INPUT_BAD_ARGUMENT", "search needs --text or --image (or both on eg2/fake for multimodal search)")
        if a.text and a.image:
            if not hasattr(emb, "multimodal"):
                raise DeskError("INPUT_BAD_ARGUMENT",
                                f"backend {backend!r} does not support joint --text + --image search; use eg2")
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
        cov = coverage(items, img, subjects, emb.texts(subjects, "query")) if (subjects and backend != "none") else None
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
