"""Embedding backends for the dataset desk.

Every backend maps images and texts into one space and returns unit-norm float32 rows. Thread pools are capped before
numpy, torch, the tokenizers or the BLAS libraries load. CPU is the default. A GPU needs a grant id (from whoever schedules the card), which
the desk records in its cache. The 'none' backend needs no torch and powers the lite tier's model-free report sections.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time

DEFAULT_THREADS = 4
_TORCH_INTEROP_SET = False


def _argv_threads(argv: list[str] | None = None, default: int = DEFAULT_THREADS) -> int:
    args = sys.argv[1:] if argv is None else argv
    for i, a in enumerate(args):
        if a == "--threads" and i + 1 < len(args):
            try:
                return max(1, int(args[i + 1]))
            except ValueError:
                break
        if a.startswith("--threads="):
            try:
                return max(1, int(a.split("=", 1)[1]))
            except ValueError:
                break
    return default


def cap_threads(threads: int = DEFAULT_THREADS) -> None:
    """Cap every thread pool before numpy, torch, the tokenizers or the BLAS libraries load."""
    t = str(max(1, int(threads)))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = t
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"


# Cap env vars immediately at import time, before numpy loads its BLAS pool.
cap_threads(_argv_threads())

import numpy as np  # noqa: E402

from dataset_desk.errors import DeskError  # noqa: E402
from dataset_desk.registry import REGISTRY, EmbedderSpec  # noqa: E402

EG2_REPO, EG2_REVISION = REGISTRY["eg2"].repo, REGISTRY["eg2"].revision
SIGLIP2_REPO, SIGLIP2_REVISION = REGISTRY["siglip2"].repo, REGISTRY["siglip2"].revision
BGE_M3_REPO, BGE_M3_REVISION = REGISTRY["bge-m3"].repo, REGISTRY["bge-m3"].revision
QWEN3_EMB_REPO, QWEN3_EMB_REVISION = REGISTRY["qwen3-emb-0.6b"].repo, REGISTRY["qwen3-emb-0.6b"].revision
GTE_MODERNBERT_REPO, GTE_MODERNBERT_REVISION = (
    REGISTRY["gte-modernbert"].repo,
    REGISTRY["gte-modernbert"].revision,
)
DINOV2_REPO, DINOV2_REVISION = REGISTRY["dinov2"].repo, REGISTRY["dinov2"].revision

VRAM_MARGIN_MIB = 1024
DEFAULT_VRAM_BUDGET_MIB = 26624
VRAM_POLL_SECONDS = 10.0
TEXT_MAX_SEQ_LENGTH = REGISTRY["bge-m3"].max_tokens

# From Qwen/Qwen3-Embedding-0.6B config_sentence_transformers.json at revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3.
QWEN3_QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

# EmbeddingGemma 2's official task prefixes from its model card. Images take no prefix.
EG2_PREFIX = {
    "query": "task: search result | query: ",
    "code_query": "task: code retrieval | query: ",
    "document": "title: none | text: ",
    "symmetric": "task: sentence similarity | query: ",
    "clustering": "task: clustering | query: ",
    "classification": "task: classification | query: ",
}

# Matryoshka truncation floor for multimodal embeddings (model card MMEB v2 drops 13.4 pts at 128d).
EG2_VALID_DIMS = (768, 512, 256)
# Supported soft vision token counts in transformers.models.gemma4.image_processing_gemma4._SUPPORTED_SOFT_TOKENS.
EG2_VISION_TOKENS = (70, 140, 280, 560, 1120)


def format_eg2_prefix(role: str, title: str | None = None) -> str:
    """Return the EmbeddingGemma 2 task prefix for `role`. Only `document` takes a title (base file name or 'none')."""
    if role not in EG2_PREFIX:
        raise ValueError(f"unknown EG2 text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
    if role == "document":
        clean = os.path.basename(str(title).replace("\\", "/")).strip() if title else ""
        return f"title: {clean or 'none'} | text: "
    return EG2_PREFIX[role]


def _resolve_titles(n: int, titles: list[str | None] | tuple[str | None, ...] | str | None) -> list[str | None]:
    if titles is None:
        return [None] * n
    if isinstance(titles, str):
        return [titles] * n
    return [titles[i] if i < len(titles) else None for i in range(n)]


def enter_torch_region(threads: int = DEFAULT_THREADS) -> None:
    """Set OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1 inside torch regions (the embed-serve disposition) and guard
    torch.set_num_interop_threads(1) so it runs at most once per process."""
    global _TORCH_INTEROP_SET
    t = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(t)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"

    import torch

    torch.set_num_threads(t)
    if not _TORCH_INTEROP_SET:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _TORCH_INTEROP_SET = True


def unit_rows(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.size and not np.isfinite(a).all():
        raise ValueError("embedding backend returned non-finite values (NaN or Inf)")
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def check_device(device: str, grant: str | None) -> None:
    if device != "cpu" and not grant:
        raise DeskError("PERM_GPU_NEEDS_GRANT", f"REFUSED: device {device!r} needs a grant id (--grant); the GPU is shared")


def _scale_mib(free_val: int, total_val: int) -> tuple[int, int]:
    if max(free_val, total_val) >= (1 << 20):
        return free_val // (1 << 20), total_val // (1 << 20)
    return free_val, total_val


def _to_mib(free_val: int, total_val: int) -> int:
    return _scale_mib(free_val, total_val)[0]


def _vram_info_mib(device: str = "cuda") -> tuple[int, int]:
    """Read (free_mib, total_mib) on CUDA via torch.cuda.mem_get_info(), or cuda.core if torch isn't loaded yet."""
    dev_idx = 0
    if ":" in device:
        with contextlib.suppress(ValueError):
            dev_idx = int(device.split(":", 1)[1])

    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        dev_cls = None
        try:
            from cuda.core import Device as dev_cls  # type: ignore[import-not-found]
        except ImportError:
            with contextlib.suppress(ImportError):
                from cuda.core.experimental import Device as dev_cls  # type: ignore[import-not-found]
        if dev_cls is not None:
            dev = dev_cls(dev_idx)
            mem = getattr(dev, "memory_info", None)
            if callable(mem):
                mem = mem()
            if isinstance(mem, tuple) and len(mem) >= 2:
                return _scale_mib(int(mem[0]), int(mem[1]))
            if mem is not None and hasattr(mem, "free"):
                return _scale_mib(int(mem.free), int(getattr(mem, "total", mem.free)))
        import torch as torch_mod  # type: ignore[no-redef]

    mem_fn = getattr(getattr(torch_mod, "cuda", None), "mem_get_info", None)
    if mem_fn is None:
        raise RuntimeError("torch.cuda.mem_get_info is not available")
    try:
        try:
            free_val, total_val = mem_fn(device) if ":" in device else mem_fn()
        except TypeError:
            free_val, total_val = mem_fn()
    except (AssertionError, RuntimeError) as exc:
        raise DeskError(
            "ENV_NO_CUDA_TORCH",
            f"REFUSED: --device {device!r} requested, but the installed torch has no CUDA support ({exc})",
            cause=exc,
        ) from exc
    return _scale_mib(int(free_val), int(total_val))


def _free_vram_mib(device: str = "cuda") -> int:
    """Read free CUDA memory in MiB via _vram_info_mib()."""
    return _vram_info_mib(device)[0]


def check_vram(
    device: str,
    budget_mib: int,
    *,
    vram_budget: int = DEFAULT_VRAM_BUDGET_MIB,
    wait: float = 0.0,
    name: str = "model",
) -> int:
    """Wait or refuse before loading a model on CUDA; never evict, signal, or stop another process."""
    if device == "cpu" or not device.startswith("cuda"):
        return 0
    need_mib = int(budget_mib) + VRAM_MARGIN_MIB
    ceiling_mib = int(vram_budget)
    wait_limit = max(0.0, float(wait or 0.0))
    waited = 0.0

    while True:
        free_mib, total_mib = _vram_info_mib(device)
        used_mib = max(0, total_mib - free_mib)
        if need_mib > ceiling_mib:
            raise DeskError(
                "PERM_GPU_BUSY",
                f"REFUSED: {name} needs {need_mib} MiB ({budget_mib} MiB + {VRAM_MARGIN_MIB} MiB margin), "
                f"exceeding --vram-budget {ceiling_mib} MiB (used {used_mib} MiB, free {free_mib} MiB, "
                f"need {need_mib} MiB, budget {ceiling_mib} MiB on {device})",
                retryable=False,
            )
        if free_mib >= need_mib and used_mib + need_mib <= ceiling_mib:
            return free_mib
        if waited >= wait_limit:
            if free_mib < need_mib:
                raise DeskError(
                    "PERM_GPU_BUSY",
                    f"REFUSED: GPU {device} busy (used {used_mib} MiB, free {free_mib} MiB, "
                    f"need {need_mib} MiB ({budget_mib} MiB + {VRAM_MARGIN_MIB} MiB margin), "
                    f"budget {ceiling_mib} MiB for {name})",
                    retryable=True,
                )
            raise DeskError(
                "PERM_GPU_BUSY",
                f"REFUSED: GPU {device} busy: used {used_mib} MiB + need {need_mib} MiB = {used_mib + need_mib} MiB "
                f"exceeds --vram-budget {ceiling_mib} MiB (used {used_mib} MiB, free {free_mib} MiB, "
                f"need {need_mib} MiB ({budget_mib} MiB + {VRAM_MARGIN_MIB} MiB margin), "
                f"budget {ceiling_mib} MiB for {name})",
                retryable=True,
            )
        step = min(VRAM_POLL_SECONDS, wait_limit - waited)
        time.sleep(step)
        waited += step


class NoneEmbedder:
    """No-embedding backend for dataset-desk-lite: needs only numpy and Pillow (no torch)."""

    name, revision, dim, vision_tokens, max_tokens, dtype_name = "none", "none-1", 0, 0, 0, "none"
    has_symmetric_vectors = False

    def images(self, paths: list[str]) -> np.ndarray:
        return np.zeros((len(paths), 0), np.float32)

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        return np.zeros((len(texts), 0), np.float32)


class FakeEmbedder:
    """Offline and deterministic, for tests and dry runs.

    An image's vector is its mean colour. A text's vector counts the words red, green and blue. The fourth axis keeps
    vectors nonzero."""

    name, revision, dim, vision_tokens, max_tokens, dtype_name = "fake", "fake-1", 4, 280, 8192, "float32"
    has_symmetric_vectors = True

    def __init__(
        self,
        *,
        dim: int | None = None,
        vision_tokens: int | None = None,
        text_only: bool = False,
        **_kw,
    ) -> None:
        if dim is not None and dim not in (*EG2_VALID_DIMS, 4):
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"REFUSED: fake dim={dim} not in {EG2_VALID_DIMS} (128d degrades multimodal quality)",
            )
        if vision_tokens is not None and vision_tokens not in EG2_VISION_TOKENS:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"REFUSED: fake vision_tokens={vision_tokens} must be one of {EG2_VISION_TOKENS}",
            )
        self.dim = 4 if dim is None else int(dim)
        self.vision_tokens = 280 if vision_tokens is None else int(vision_tokens)
        self.max_tokens = 8192
        self.seen_calls: list[dict] = []

    def _fit_rows(self, rows: list[list[float]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, self.dim), np.float32)
        arr = np.asarray(rows, dtype=np.float32)
        if self.dim != arr.shape[1]:
            reps = (self.dim + arr.shape[1] - 1) // arr.shape[1]
            arr = np.tile(arr, (1, reps))[:, :self.dim]
        return unit_rows(arr)

    def images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        rows = []
        for p in paths:
            with Image.open(p) as im:
                px = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            r, g, b = px.reshape(-1, 3).mean(axis=0)
            rows.append([float(r), float(g), float(b), 0.05])
        return self._fit_rows(rows)

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        if role not in EG2_PREFIX:
            raise ValueError(f"unknown text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        resolved_titles = _resolve_titles(len(texts), titles)
        formatted = [format_eg2_prefix(role, tl) + t for t, tl in zip(texts, resolved_titles, strict=True)]
        self.seen_calls.append({"role": role, "titles": resolved_titles, "formatted": formatted})
        bias = 0.052 if role == "document" else (0.054 if role == "symmetric" else 0.05)
        rows = []
        for t in texts:
            w = t.lower()
            rows.append([float(w.count("red")), float(w.count("green")), float(w.count("blue")), bias])
        return self._fit_rows(rows)

    def multimodal(self, items: list[dict], role: str | None = "query") -> np.ndarray:
        """Interleaved text + image items: [{'text': '...', 'images': [path, ...]}]."""
        if role is not None and role not in EG2_PREFIX:
            raise ValueError(f"unknown text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        rows = []
        neutral_tv = np.tile(np.array([0.0, 0.0, 0.0, 0.05], dtype=np.float32), (self.dim + 3) // 4)[:self.dim]
        for item in items:
            txt = item.get("text") or ""
            im_paths = item.get("images") or ([item["image"]] if item.get("image") else [])
            tv = self.texts([txt], role=role or "query")[0] if txt else neutral_tv
            iv = self.images(im_paths).mean(axis=0) if im_paths else np.zeros(self.dim, dtype=np.float32)
            rows.append(0.5 * tv + 0.5 * iv)
        return unit_rows(rows) if rows else np.zeros((0, self.dim), np.float32)


class EG2Embedder:
    """EmbeddingGemma 2 with text and vision (audio left out, about 440M), through sentence-transformers.

    Per the EmbeddingGemma 2 model card and transformers 5.19 / sentence-transformers 6.1 source:
    - Inference must run in float32 or bfloat16, never float16 (activations overflow float16 -> NaN).
    - Multimodal Matryoshka truncation is supported down to 256d (128d drops 13.4 MMEB points and is refused).
    - Vision soft tokens are one of {70, 140, 280, 560, 1120} per image (default 280), wired via
      processor_kwargs={'image_seq_length': vision_tokens} on SentenceTransformer and
      model[0].processing_kwargs={'image': {'max_soft_tokens': vision_tokens}} plus processor.image_processor.max_soft_tokens.
    - Interleaved text + image inputs share one 8,192-token window via <|image|> placeholders."""

    name, revision = "eg2", EG2_REVISION
    vram_mib = REGISTRY["eg2"].vram_mib
    max_tokens = REGISTRY["eg2"].max_tokens
    has_symmetric_vectors = True

    def __init__(self, device: str = "cpu", threads: int = DEFAULT_THREADS, dim: int | None = 768, batch: int = 4,
                 vision_tokens: int | None = 280, dtype: str = "auto",
                 vram_budget: int = DEFAULT_VRAM_BUDGET_MIB, wait: float = 0.0,
                 text_only: bool = False):
        eff_dim = 768 if dim is None else int(dim)
        eff_vt = 280 if vision_tokens is None else int(vision_tokens)
        if eff_dim not in EG2_VALID_DIMS:
            raise ValueError(f"REFUSED: EG2 dim={eff_dim} not in {EG2_VALID_DIMS} (128d degrades multimodal quality)")
        if eff_vt not in EG2_VISION_TOKENS:
            raise ValueError(f"REFUSED: EG2 vision_tokens={eff_vt} must be one of {EG2_VISION_TOKENS}")
        if dtype in ("float16", "fp16", "half"):
            raise ValueError("REFUSED: float16 overflows EmbeddingGemma 2 activations; use float32 or bfloat16")

        self.device = device
        self.threads = threads
        self.dim = eff_dim
        self.batch = batch
        self.text_only = bool(text_only)
        self.vision_tokens = 0 if self.text_only else eff_vt
        self.max_tokens = REGISTRY["eg2"].max_tokens
        self.vram_budget, self.wait = vram_budget, wait
        self._vram_checked = False
        self._requested_dtype = dtype
        self.dtype_name = "bfloat16" if dtype == "bfloat16" else "float32"
        self.processor_kwargs = {} if self.text_only else {"image_seq_length": eff_vt}
        self.processing_kwargs = {} if self.text_only else {"image": {"max_soft_tokens": eff_vt}}
        self.model = None
        self._tokenizer = None
        self.params: int | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _extract_model_tokenizer(self.model) or _load_repo_tokenizer(EG2_REPO, EG2_REVISION)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.device != "cpu" and not self._vram_checked:
            check_vram(self.device, self.vram_mib, vram_budget=self.vram_budget, wait=self.wait, name=self.name)
            self._vram_checked = True
        enter_torch_region(self.threads)
        import torch
        from sentence_transformers import SentenceTransformer

        dtype = self._requested_dtype
        if dtype == "bfloat16" or (dtype == "auto" and self.device.startswith("cuda") and torch.cuda.is_bf16_supported()):
            torch_dtype = torch.bfloat16
            self.dtype_name = "bfloat16"
        else:
            torch_dtype = torch.float32
            self.dtype_name = "float32"

        with transformers_warnings_off():
            if self.text_only:
                self.model = SentenceTransformer(
                    EG2_REPO,
                    revision=EG2_REVISION,
                    device=self.device,
                    config_kwargs={"vision_config": None, "audio_config": None},
                    model_kwargs={"dtype": torch_dtype},
                )
            else:
                self.model = SentenceTransformer(
                    EG2_REPO,
                    revision=EG2_REVISION,
                    device=self.device,
                    config_kwargs={"audio_config": None},
                    model_kwargs={"dtype": torch_dtype},
                    processor_kwargs=dict(self.processor_kwargs),
                )
                if len(self.model) > 0:
                    mod0 = self.model[0]
                    mod0.processing_kwargs = dict(self.processing_kwargs)
                    proc = getattr(mod0, "processor", None)
                    if proc is not None:
                        proc.image_seq_length = self.vision_tokens
                        if getattr(proc, "image_processor", None) is not None:
                            proc.image_processor.max_soft_tokens = self.vision_tokens
        self.params = sum(p.numel() for p in self.model.parameters())

    def _encode(self, inputs: list) -> np.ndarray:
        self._ensure_model()
        out = self.model.encode(inputs, batch_size=self.batch, normalize_embeddings=True, truncate_dim=self.dim,
                                convert_to_numpy=True, show_progress_bar=False)
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] > self.dim:
            arr = arr[:, :self.dim]
        return unit_rows(arr)

    def images(self, paths: list[str]) -> np.ndarray:
        if self.text_only:
            raise DeskError("INPUT_BAD_ARGUMENT", "REFUSED: EG2Embedder loaded with text_only=True cannot embed images")
        if not paths:
            return np.zeros((0, self.dim), np.float32)
        from PIL import Image

        rows = []
        for i in range(0, len(paths), self.batch):
            ims = []
            for p in paths[i:i + self.batch]:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            rows.append(self._encode(ims))  # a PIL image is the image modality; no prefix
        return np.vstack(rows) if rows else np.zeros((0, self.dim), np.float32)

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        if role not in EG2_PREFIX:
            raise ValueError(f"unknown EG2 text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        resolved_titles = _resolve_titles(len(texts), titles)
        formatted = [format_eg2_prefix(role, tl) + t for t, tl in zip(texts, resolved_titles, strict=True)]
        return self._encode(formatted)

    def multimodal(self, items: list[dict], role: str | None = "query") -> np.ndarray:
        """Encode interleaved text + image dicts into one shared 8K context window.
        Each item is {'text': '... <|image|> ...', 'images': [path, ...]}."""
        if self.text_only:
            raise DeskError("INPUT_BAD_ARGUMENT", "REFUSED: EG2Embedder loaded with text_only=True cannot embed images")
        if role is not None and role not in EG2_PREFIX:
            raise ValueError(f"unknown EG2 text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        if not items:
            return np.zeros((0, self.dim), np.float32)
        from PIL import Image

        payloads = []
        prefix = format_eg2_prefix(role) if role else ""
        for item in items:
            raw_text = item.get("text") or ""
            im_paths = item.get("images") or ([item["image"]] if item.get("image") else [])
            ims = []
            for p in im_paths:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            count = raw_text.count("<|image|>")
            if count < len(ims):
                pad = " ".join(["<|image|>"] * (len(ims) - count))
                raw_text = f"{pad} {raw_text}".strip()
            payloads.append({"text": prefix + raw_text, "image": ims if len(ims) != 1 else ims[0]})
        return self._encode(payloads)


@contextlib.contextmanager
def transformers_warnings_off():
    """Hold transformers' log at error level and disable its progress bar while a model loads, then restore both.

    SigLIP2's published config sets bos/eos token ids outside its text vocabulary, and transformers logs a warning on
    every load. The desk never uses those ids, so the warning only adds noise. Errors still show."""
    try:
        from transformers.utils import logging as tf_logging
    except (ImportError, ValueError):
        yield
        return
    previous = tf_logging.get_verbosity()
    is_pb_enabled = getattr(tf_logging, "is_progress_bar_enabled", None)
    disable_pb = getattr(tf_logging, "disable_progress_bar", None)
    enable_pb = getattr(tf_logging, "enable_progress_bar", None)
    prev_pb = bool(is_pb_enabled()) if callable(is_pb_enabled) else None
    tf_logging.set_verbosity_error()
    if callable(disable_pb):
        disable_pb()
    try:
        yield
    finally:
        if prev_pb is True and callable(enable_pb):
            enable_pb()
        elif prev_pb is False and callable(disable_pb):
            disable_pb()
        tf_logging.set_verbosity(previous)


def _load_repo_tokenizer(repo: str, revision: str):
    from transformers import AutoTokenizer

    with transformers_warnings_off():
        return AutoTokenizer.from_pretrained(repo, revision=revision)


def _extract_model_tokenizer(model=None, processor=None):
    if model is not None:
        tok = getattr(model, "tokenizer", None)
        if tok is not None:
            return tok
        if hasattr(model, "__len__") and len(model) > 0:
            tok = getattr(model[0], "tokenizer", None)
            if tok is not None:
                return tok
    if processor is not None:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            return tok
    return None


class SigLIP2Embedder:
    """SigLIP2 so400m/384, the model ai-eyes runs, with its trained padding of 64 text tokens."""

    name, revision, dim, vision_tokens, dtype_name = "siglip2", SIGLIP2_REVISION, 1152, 729, "float32"
    vram_mib = REGISTRY["siglip2"].vram_mib
    max_tokens = REGISTRY["siglip2"].max_tokens
    has_symmetric_vectors = False

    def __init__(self, device: str = "cpu", threads: int = DEFAULT_THREADS, batch: int = 4,
                 vram_budget: int = DEFAULT_VRAM_BUDGET_MIB, wait: float = 0.0):
        self.device, self.threads, self.batch = device, threads, batch
        self.vram_budget, self.wait = vram_budget, wait
        self._vram_checked = False
        self.torch = None
        self.processor = None
        self.model = None
        self._tokenizer = None
        self.params: int | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = (
                _extract_model_tokenizer(self.model, self.processor)
                or _load_repo_tokenizer(SIGLIP2_REPO, SIGLIP2_REVISION)
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.device != "cpu" and not self._vram_checked:
            check_vram(self.device, self.vram_mib, vram_budget=self.vram_budget, wait=self.wait, name=self.name)
            self._vram_checked = True
        enter_torch_region(self.threads)
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        with transformers_warnings_off():
            self.processor = AutoProcessor.from_pretrained(SIGLIP2_REPO, revision=SIGLIP2_REVISION)
            self.model = AutoModel.from_pretrained(SIGLIP2_REPO, revision=SIGLIP2_REVISION,
                                                   dtype=torch.float32).to(self.device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    def _pooled(self, out) -> np.ndarray:
        if not self.torch.is_tensor(out):  # transformers 5.x returns an output object
            out = out.pooler_output
        return unit_rows(out.detach().cpu().numpy())

    def images(self, paths: list[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, 1152), np.float32)
        self._ensure_model()
        from PIL import Image

        rows = []
        for i in range(0, len(paths), self.batch):
            ims = []
            for p in paths[i:i + self.batch]:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                rows.append(self._pooled(self.model.get_image_features(**inputs)))
        return np.vstack(rows) if rows else np.zeros((0, 1152), np.float32)

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1152), np.float32)
        self._ensure_model()
        rows = []
        for i in range(0, len(texts), 16):
            inputs = self.processor(text=texts[i:i + 16], padding="max_length", max_length=64, truncation=True,
                                    return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                rows.append(self._pooled(self.model.get_text_features(**inputs)))
        return np.vstack(rows) if rows else np.zeros((0, 1152), np.float32)


class SentenceTransformerTextEmbedder:
    """Text-only embedder loaded through SentenceTransformer(repo, revision=...) with bf16 on CUDA, fp32 on CPU,
    max_seq_length following spec.max_tokens, and unit-normalized output."""

    vision_tokens = 0
    max_seq_length = TEXT_MAX_SEQ_LENGTH
    has_symmetric_vectors = False

    def __init__(
        self,
        spec: EmbedderSpec,
        device: str = "cpu",
        threads: int = DEFAULT_THREADS,
        batch: int = 8,
        vram_budget: int = DEFAULT_VRAM_BUDGET_MIB,
        wait: float = 0.0,
    ):
        self.spec = spec
        self.name = spec.name
        self.repo = spec.repo
        self.revision = spec.revision
        self.licence = spec.licence
        self.modalities = spec.modalities
        self.vram_mib = spec.vram_mib
        self.dim = spec.dim
        self.max_tokens = spec.max_tokens
        self.max_seq_length = spec.max_tokens or TEXT_MAX_SEQ_LENGTH
        self.device, self.threads, self.batch = device, threads, batch
        self.vram_budget, self.wait = vram_budget, wait
        self._vram_checked = False
        self.dtype_name = "bfloat16" if device.startswith("cuda") else "float32"
        self.model = None
        self._tokenizer = None
        self.params: int | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = _extract_model_tokenizer(self.model) or _load_repo_tokenizer(self.repo, self.revision)
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.device != "cpu" and not self._vram_checked:
            check_vram(self.device, self.vram_mib, vram_budget=self.vram_budget, wait=self.wait, name=self.name)
            self._vram_checked = True
        enter_torch_region(self.threads)
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device.startswith("cuda"):
            torch_dtype = torch.bfloat16
            self.dtype_name = "bfloat16"
        else:
            torch_dtype = torch.float32
            self.dtype_name = "float32"

        with transformers_warnings_off():
            self.model = SentenceTransformer(
                self.repo,
                revision=self.revision,
                device=self.device,
                model_kwargs={"dtype": torch_dtype},
            )
        self.model.max_seq_length = self.max_seq_length
        self.params = sum(p.numel() for p in self.model.parameters())

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        self._ensure_model()
        encode_kw: dict = {
            "batch_size": self.batch,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self.name == "qwen3-emb-0.6b" and role in ("query", "code_query"):
            encode_kw["prompt_name"] = "query"
        out = self.model.encode(texts, **encode_kw)
        res = unit_rows(out)
        if res.ndim == 2 and res.shape[1] > 0:
            self.dim = int(res.shape[1])
        return res

    def images(self, paths: list[str]) -> np.ndarray:
        raise DeskError("INPUT_BAD_ARGUMENT", f"backend {self.name!r} is text-only and cannot embed images")


class DINOv2Embedder:
    """DINOv2 vision embedder loaded through AutoImageProcessor and AutoModel at its pinned revision, pooled and
    unit-normalized."""

    vision_tokens = 256
    max_tokens = REGISTRY["dinov2"].max_tokens
    dtype_name = "float32"
    has_symmetric_vectors = False

    def __init__(
        self,
        spec: EmbedderSpec = REGISTRY["dinov2"],
        device: str = "cpu",
        threads: int = DEFAULT_THREADS,
        batch: int = 8,
        vram_budget: int = DEFAULT_VRAM_BUDGET_MIB,
        wait: float = 0.0,
    ):
        self.spec = spec
        self.name = spec.name
        self.repo = spec.repo
        self.revision = spec.revision
        self.licence = spec.licence
        self.modalities = spec.modalities
        self.vram_mib = spec.vram_mib
        self.dim = spec.dim
        self.max_tokens = spec.max_tokens
        self.device, self.threads, self.batch = device, threads, batch
        self.vram_budget, self.wait = vram_budget, wait
        self._vram_checked = False
        self.torch = None
        self.processor = None
        self.model = None
        self._tokenizer = None
        self.params: int | None = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value) -> None:
        self._tokenizer = value

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if self.device != "cpu" and not self._vram_checked:
            check_vram(self.device, self.vram_mib, vram_budget=self.vram_budget, wait=self.wait, name=self.name)
            self._vram_checked = True
        enter_torch_region(self.threads)
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        with transformers_warnings_off():
            self.processor = AutoImageProcessor.from_pretrained(self.repo, revision=self.revision)
            self.model = AutoModel.from_pretrained(self.repo, revision=self.revision,
                                                   dtype=torch.float32).to(self.device).eval()
        self.params = sum(p.numel() for p in self.model.parameters())

    def _pooled(self, out) -> np.ndarray:
        if not self.torch.is_tensor(out):
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return unit_rows(out.detach().cpu().numpy())

    def images(self, paths: list[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), np.float32)
        self._ensure_model()
        from PIL import Image

        rows = []
        for i in range(0, len(paths), self.batch):
            ims = []
            for p in paths[i:i + self.batch]:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            inputs = self.processor(images=ims, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                rows.append(self._pooled(self.model(**inputs)))
        res = np.vstack(rows) if rows else np.zeros((0, self.dim), np.float32)
        if res.ndim == 2 and res.shape[1] > 0:
            self.dim = int(res.shape[1])
        return res

    def texts(self, texts: list[str], role: str = "query", titles=None) -> np.ndarray:
        raise DeskError("INPUT_BAD_ARGUMENT", f"backend {self.name!r} is image-only and cannot embed texts")


PLAIN_SIGHT_REPO = "florence-community/Florence-2-large"
PLAIN_SIGHT_REVISION = "4271c66b88cdbc05735372ec13b2360108de5317"
FLORENCE2_REPO, FLORENCE2_REVISION = PLAIN_SIGHT_REPO, PLAIN_SIGHT_REVISION


class FakeDrafter:
    """Offline and deterministic drafter for tests.

    Accepts ONLY image paths (never prompts, canon.subject, or current descriptions) and records the exact paths
    received in `seen_paths`."""

    name = "fake"
    drafter_id = "plain-sight/florence-2-fake"
    revision = "fake-florence2-1"

    def __init__(self) -> None:
        self.seen_paths: list[str] = []

    def draft(self, image_paths: list[str]) -> list[str]:
        if not isinstance(image_paths, list) or any(not isinstance(p, str) for p in image_paths):
            raise TypeError("drafter.draft() accepts a list[str] of image file paths only")
        from PIL import Image

        out: list[str] = []
        for p in image_paths:
            self.seen_paths.append(p)
            tone = "neutral"
            if p and os.path.isfile(p):
                try:
                    with Image.open(p) as im:
                        px = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32)
                    r, g, b = px.reshape(-1, 3).mean(axis=0)
                    tone = "crimson" if r > max(g, b) else ("emerald" if g > max(r, b) else "cobalt")
                except OSError:
                    pass
            stem = os.path.splitext(os.path.basename(p or "plate"))[0]
            out.append(
                f"a weathered {tone} figure beside carved stone pillars in a dim courtyard ({stem})"
            )
        return out


class PlainSightDrafter:
    """plain-sight (florence-community/Florence-2-large @ 4271c66b…) blind drafter.

    - Uses `plain_sight.engine.Florence2Engine` lazily when `plain-sight` is installed.
    - Otherwise mirrors `plain-sight`'s exact pin (`florence-community/Florence-2-large` at revision
      `4271c66b88cdbc05735372ec13b2360108de5317`) using native `transformers` classes only.
    - Enforces `local_files_only=True` and `HF_HUB_OFFLINE=1` around model loading so a missing snapshot raises a named
      error instead of downloading.
    - Records the `drafter_id` and `revision` that actually loaded."""

    name = "plain-sight"
    TASK_TOKEN = "<DETAILED_CAPTION>"

    def __init__(
        self,
        device: str = "cpu",
        threads: int = DEFAULT_THREADS,
        model_id: str = PLAIN_SIGHT_REPO,
        revision: str = PLAIN_SIGHT_REVISION,
        detail: str = "medium",
    ):
        self.device, self.threads = device, threads
        self.model_id = model_id
        self.requested_revision = revision
        self.detail = detail
        self.drafter_id = model_id
        self.revision = revision
        self.engine = None
        self.torch = None
        self.processor = None
        self.model = None

    def _ensure_model(self) -> None:
        if self.engine is not None or self.model is not None:
            return
        enter_torch_region(self.threads)

        prev_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            try:
                from plain_sight.engine import Florence2Engine
            except ImportError:
                Florence2Engine = None

            if Florence2Engine is not None:
                try:
                    with transformers_warnings_off():
                        eng = Florence2Engine(
                            model_id=self.model_id,
                            revision=self.requested_revision,
                            device=self.device,
                            dtype="float32" if self.device == "cpu" else None,
                            eager_load=True,
                        )
                        if not getattr(eng, "loaded", False) and hasattr(eng, "_ensure_loaded"):
                            eng._ensure_loaded()
                except Exception as exc:
                    raise RuntimeError(
                        f"plain-sight offline load failed for repo {self.model_id!r} at revision "
                        f"{self.requested_revision!r} (HF_HUB_OFFLINE=1; no downloads permitted): {exc}"
                    ) from exc

                prov = eng.output_provenance() if hasattr(eng, "output_provenance") else {}
                loaded_id = prov.get("model_id") or getattr(eng, "model_id", None) or self.model_id
                loaded_rev = (
                    prov.get("revision_resolved")
                    or (eng.resolved_revision() if hasattr(eng, "resolved_revision") else None)
                    or getattr(eng, "revision", None)
                    or self.requested_revision
                )
                self.drafter_id = str(loaded_id)
                self.revision = str(loaded_rev)
                self.engine = eng
                return

            # Mirror plain-sight's exact pin and native transformers classes offline when plain-sight isn't installed
            import torch
            from transformers import AutoProcessor

            try:
                with transformers_warnings_off():
                    processor = AutoProcessor.from_pretrained(
                        self.model_id,
                        revision=self.requested_revision,
                        local_files_only=True,
                    )
                    try:
                        from transformers import AutoModelForImageTextToText

                        model = AutoModelForImageTextToText.from_pretrained(
                            self.model_id,
                            revision=self.requested_revision,
                            local_files_only=True,
                        )
                    except (ValueError, KeyError, ImportError):
                        from transformers import Florence2ForConditionalGeneration

                        model = Florence2ForConditionalGeneration.from_pretrained(
                            self.model_id,
                            revision=self.requested_revision,
                            local_files_only=True,
                        )
                    model = model.eval().to(self.device)
            except Exception as exc:
                # Only reached when plain-sight isn't installed: the engine path above returns or raises.
                hint = ("install plain-sight (https://github.com/mcp-tool-shop-org/plain-sight) and make sure the "
                        "pinned snapshot is in the Hugging Face cache")
                raise RuntimeError(
                    f"plain-sight offline snapshot missing for repo {self.model_id!r} at revision "
                    f"{self.requested_revision!r} (local_files_only=True, HF_HUB_OFFLINE=1; {hint}): {exc}"
                ) from exc

            cfg = getattr(model, "config", None)
            loaded_id = getattr(cfg, "_name_or_path", None) or self.model_id
            loaded_rev = getattr(cfg, "_commit_hash", None) or self.requested_revision
            self.drafter_id = str(loaded_id)
            self.revision = str(loaded_rev)
            self.torch = torch
            self.processor = processor
            self.model = model
        finally:
            if prev_hf_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
            if prev_tf_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = prev_tf_offline

    def draft(self, image_paths: list[str]) -> list[str]:
        if not isinstance(image_paths, list) or any(not isinstance(p, str) for p in image_paths):
            raise TypeError("drafter.draft() accepts a list[str] of image file paths only")
        if not image_paths:
            return []
        self._ensure_model()

        if self.engine is not None:
            return [str(self.engine.describe(p, detail=self.detail)).strip() for p in image_paths]

        from PIL import Image

        task_prompt = self.TASK_TOKEN
        out: list[str] = []
        for p in image_paths:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                w, h = rgb.size
            inputs = self.processor(text=task_prompt, images=rgb, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )
            text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(text, task=task_prompt, image_size=(w, h))
            out.append(str(parsed.get(task_prompt, text)).strip())
        return out


def make_drafter(name: str, device: str = "cpu", threads: int = DEFAULT_THREADS, grant: str | None = None):
    if name in ("none", "", None):
        return None
    check_device(device, grant)
    cap_threads(threads)
    if name == "fake":
        return FakeDrafter()
    if name in ("plain-sight", "florence2"):
        return PlainSightDrafter(device=device, threads=threads)
    raise DeskError("INPUT_UNKNOWN_DRAFTER", f"unknown drafter {name!r}: plain-sight, fake, or none")


_STANDINS = {
    "none": lambda **_kw: NoneEmbedder(),
    "fake": lambda **kw: FakeEmbedder(**kw),
}

_LOADERS = {
    "eg2": lambda spec, **kw: EG2Embedder(**kw),
    "siglip2": lambda spec, **kw: SigLIP2Embedder(**kw),
    "sentence_transformer": lambda spec, **kw: SentenceTransformerTextEmbedder(spec, **kw),
    "dinov2": lambda spec, **kw: DINOv2Embedder(spec, **kw),
}


def validate_embedder_options(
    backend: str,
    *,
    dim: int | None = None,
    vision_tokens: int | None = None,
) -> None:
    """Validate --dim and --vision-tokens against backend capabilities."""
    if backend not in ("eg2", "fake") and (dim is not None or vision_tokens is not None):
        raise DeskError(
            "INPUT_BAD_ARGUMENT",
            f"REFUSED: backend {backend!r} does not support --dim or --vision-tokens (only 'eg2' and 'fake' do)",
        )
    if dim is not None:
        allowed_dims = EG2_VALID_DIMS if backend == "eg2" else (*EG2_VALID_DIMS, 4)
        if dim not in allowed_dims:
            raise DeskError(
                "INPUT_BAD_ARGUMENT",
                f"REFUSED: {backend} dim={dim} not in {EG2_VALID_DIMS} (128d degrades multimodal quality)",
            )
    if vision_tokens is not None and vision_tokens not in EG2_VISION_TOKENS:
        raise DeskError(
            "INPUT_BAD_ARGUMENT",
            f"REFUSED: {backend} vision_tokens={vision_tokens} must be one of {EG2_VISION_TOKENS}",
        )


def _can_probe_vram() -> bool:
    if sys.modules.get("torch") is not None or sys.modules.get("cuda.core") is not None:
        return True
    import importlib.util

    return importlib.util.find_spec("torch") is not None or importlib.util.find_spec("cuda") is not None


def make(
    backend: str,
    device: str = "cpu",
    threads: int = DEFAULT_THREADS,
    grant: str | None = None,
    *,
    vram_budget: int = DEFAULT_VRAM_BUDGET_MIB,
    wait: float = 0.0,
    dim: int | None = None,
    vision_tokens: int | None = None,
    text_only: bool = False,
    **kwargs,
):
    check_device(device, grant)
    cap_threads(threads)
    if backend not in _STANDINS and backend not in REGISTRY:
        valid = ", ".join([*REGISTRY, *_STANDINS])
        raise DeskError("INPUT_UNKNOWN_BACKEND", f"unknown backend {backend!r}: {valid}")
    validate_embedder_options(backend, dim=dim, vision_tokens=vision_tokens)
    extra_kw = dict(kwargs)
    if backend in ("eg2", "fake"):
        if dim is not None:
            extra_kw["dim"] = dim
        if vision_tokens is not None:
            extra_kw["vision_tokens"] = vision_tokens
        if text_only:
            extra_kw["text_only"] = True
    standin = _STANDINS.get(backend)
    if standin is not None:
        return standin(device=device, threads=threads, **extra_kw)
    spec = REGISTRY[backend]
    emb = _LOADERS[spec.loader](
        spec,
        device=device,
        threads=threads,
        vram_budget=vram_budget,
        wait=wait,
        **extra_kw,
    )
    if device != "cpu" and _can_probe_vram():
        check_vram(device, spec.vram_mib, vram_budget=vram_budget, wait=wait, name=spec.name)
        emb._vram_checked = True
    return emb

