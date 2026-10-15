"""Embedding backends for the dataset desk.

Every backend maps images and texts into one space and returns unit-norm float32 rows. Thread pools are capped before
numpy, torch, the tokenizers or the BLAS libraries load. CPU is the default. A GPU needs a grant id (from whoever schedules the card), which
the desk records in its cache. The 'none' backend needs no torch and powers the lite tier's model-free report sections.
"""

from __future__ import annotations

import os
import sys

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

EG2_REPO, EG2_REVISION = "google/embeddinggemma-2", "914f7f89142e33e77833254d9c9b90c3cef7303b"
SIGLIP2_REPO, SIGLIP2_REVISION = "google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"

# EmbeddingGemma 2's official task prefixes from its model card. Images take no prefix.
EG2_PREFIX = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "symmetric": "task: sentence similarity | query: ",
    "clustering": "task: clustering | query: ",
    "classification": "task: classification | query: ",
}

# Matryoshka truncation floor for multimodal embeddings (model card MMEB v2 drops 13.4 pts at 128d).
EG2_VALID_DIMS = (768, 512, 256)
# Supported soft vision token counts in transformers.models.gemma4.image_processing_gemma4._SUPPORTED_SOFT_TOKENS.
EG2_VISION_TOKENS = (70, 140, 280, 560, 1120)


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


class NoneEmbedder:
    """No-embedding backend for dataset-desk-lite: needs only numpy and Pillow (no torch)."""

    name, revision, dim, vision_tokens, dtype_name = "none", "none-1", 0, 0, "none"

    def images(self, paths: list[str]) -> np.ndarray:
        return np.zeros((len(paths), 0), np.float32)

    def texts(self, texts: list[str], role: str = "query") -> np.ndarray:
        return np.zeros((len(texts), 0), np.float32)


class FakeEmbedder:
    """Offline and deterministic, for tests and dry runs.

    An image's vector is its mean colour. A text's vector counts the words red, green and blue. The fourth axis keeps
    vectors nonzero."""

    name, revision, dim, vision_tokens, dtype_name = "fake", "fake-1", 4, 280, "float32"

    def images(self, paths: list[str]) -> np.ndarray:
        from PIL import Image

        rows = []
        for p in paths:
            with Image.open(p) as im:
                px = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            r, g, b = px.reshape(-1, 3).mean(axis=0)
            rows.append([r, g, b, 0.05])
        return unit_rows(rows) if rows else np.zeros((0, self.dim), np.float32)

    def texts(self, texts: list[str], role: str = "query") -> np.ndarray:
        if role not in EG2_PREFIX:
            raise ValueError(f"unknown text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        rows = []
        for t in texts:
            w = t.lower()
            rows.append([w.count("red"), w.count("green"), w.count("blue"), 0.05])
        return unit_rows(rows) if rows else np.zeros((0, self.dim), np.float32)

    def multimodal(self, items: list[dict], role: str | None = "query") -> np.ndarray:
        """Interleaved text + image items: [{'text': '...', 'images': [path, ...]}]."""
        if role is not None and role not in EG2_PREFIX:
            raise ValueError(f"unknown text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        rows = []
        for item in items:
            txt = item.get("text") or ""
            im_paths = item.get("images") or ([item["image"]] if item.get("image") else [])
            tv = self.texts([txt], role=role or "query")[0] if txt else np.array([0, 0, 0, 0.05], dtype=np.float32)
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

    def __init__(self, device: str = "cpu", threads: int = DEFAULT_THREADS, dim: int = 768, batch: int = 4,
                 vision_tokens: int = 280, dtype: str = "auto"):
        if dim not in EG2_VALID_DIMS:
            raise ValueError(f"REFUSED: EG2 dim={dim} not in {EG2_VALID_DIMS} (128d degrades multimodal quality)")
        if vision_tokens not in EG2_VISION_TOKENS:
            raise ValueError(f"REFUSED: EG2 vision_tokens={vision_tokens} must be one of {EG2_VISION_TOKENS}")
        if dtype in ("float16", "fp16", "half"):
            raise ValueError("REFUSED: float16 overflows EmbeddingGemma 2 activations; use float32 or bfloat16")

        self.device = device
        self.threads = threads
        self.dim, self.batch, self.vision_tokens = dim, batch, vision_tokens
        self._requested_dtype = dtype
        self.dtype_name = "bfloat16" if dtype == "bfloat16" else "float32"
        self.processor_kwargs = {"image_seq_length": vision_tokens}
        self.processing_kwargs = {"image": {"max_soft_tokens": vision_tokens}}
        self.model = None
        self.params: int | None = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
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

        self.model = SentenceTransformer(
            EG2_REPO,
            revision=EG2_REVISION,
            device=self.device,
            config_kwargs={"audio_config": None},
            model_kwargs={"torch_dtype": torch_dtype},
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
        return unit_rows(out)

    def images(self, paths: list[str]) -> np.ndarray:
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

    def texts(self, texts: list[str], role: str = "query") -> np.ndarray:
        if role not in EG2_PREFIX:
            raise ValueError(f"unknown EG2 text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        return self._encode([EG2_PREFIX[role] + t for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def multimodal(self, items: list[dict], role: str | None = "query") -> np.ndarray:
        """Encode interleaved text + image dicts into one shared 8K context window.
        Each item is {'text': '... <|image|> ...', 'images': [path, ...]}."""
        if role is not None and role not in EG2_PREFIX:
            raise ValueError(f"unknown EG2 text role {role!r}; expected one of {sorted(EG2_PREFIX)}")
        if not items:
            return np.zeros((0, self.dim), np.float32)
        from PIL import Image

        payloads = []
        prefix = EG2_PREFIX[role] if role else ""
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


class SigLIP2Embedder:
    """SigLIP2 so400m/384, the model ai-eyes runs, with its trained padding of 64 text tokens."""

    name, revision, dim, vision_tokens, dtype_name = "siglip2", SIGLIP2_REVISION, 1152, 729, "float32"

    def __init__(self, device: str = "cpu", threads: int = DEFAULT_THREADS, batch: int = 4):
        self.device, self.threads, self.batch = device, threads, batch
        self.torch = None
        self.processor = None
        self.model = None
        self.params: int | None = None

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        enter_torch_region(self.threads)
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(SIGLIP2_REPO, revision=SIGLIP2_REVISION)
        self.model = AutoModel.from_pretrained(SIGLIP2_REPO, revision=SIGLIP2_REVISION,
                                               torch_dtype=torch.float32).to(self.device).eval()
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

    def texts(self, texts: list[str], role: str = "query") -> np.ndarray:
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


def make(backend: str, device: str = "cpu", threads: int = DEFAULT_THREADS, grant: str | None = None, **kwargs):
    check_device(device, grant)
    cap_threads(threads)
    if backend == "none":
        return NoneEmbedder()
    if backend == "fake":
        return FakeEmbedder()
    if backend == "eg2":
        return EG2Embedder(device=device, threads=threads, **kwargs)
    if backend == "siglip2":
        return SigLIP2Embedder(device=device, threads=threads, **kwargs)
    raise DeskError("INPUT_UNKNOWN_BACKEND", f"unknown backend {backend!r}: eg2, siglip2, none or fake")

