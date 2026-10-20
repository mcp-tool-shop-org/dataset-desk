"""Embedder registry and provisional task defaults (calibration rung 1).

Every model entry pins its Hugging Face Hub repository, full 40-hex commit revision, declared licence, supported
modalities, loader kind, and measured VRAM budget in MiB. Task defaults are looked up here and never hard-wired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmbedderSpec:
    """One registered embedding model and its resource/provenance contract."""

    name: str
    repo: str
    revision: str
    licence: str
    modalities: tuple[str, ...]
    loader: str
    vram_mib: int
    dim: int = 768
    max_tokens: int = 0

    @property
    def loader_kind(self) -> str:
        return self.loader

    @property
    def vram_budget(self) -> int:
        return self.vram_mib

    @property
    def vram_budget_mib(self) -> int:
        return self.vram_mib

    def to_dict(self) -> dict:
        return asdict(self)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


REGISTRY: dict[str, EmbedderSpec] = {
    "eg2": EmbedderSpec(
        name="eg2",
        repo="google/embeddinggemma-2",
        revision="914f7f89142e33e77833254d9c9b90c3cef7303b",
        licence="Apache-2.0; deployments follow the Gemma Prohibited Use Policy",
        modalities=("text", "image"),
        loader="eg2",
        vram_mib=13119,  # conservative whole-card figure 13,119 MiB: registered 12,550 (multimodal) + context 569 (F1); the dogfood's text-only real run measured 9,527; RTX 5090, R&D 2026-10-20
        dim=768,
        max_tokens=8000,
    ),
    "siglip2": EmbedderSpec(
        name="siglip2",
        repo="google/siglip2-so400m-patch14-384",
        revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        licence="Apache-2.0",
        modalities=("image", "text"),
        loader="siglip2",
        vram_mib=6207,  # conservative whole-card figure 6,207 MiB: the 0.2.0 dogfood's whole-card peak on real images; registered 5,340 + context 518 = 5,858; RTX 5090, R&D 2026-10-20
        dim=1152,
        max_tokens=64,
    ),
    "bge-m3": EmbedderSpec(
        name="bge-m3",
        repo="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        licence="MIT",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=8442,  # conservative whole-card figure 8,442 MiB: the 0.2.0 dogfood's whole-card peak on rnd-data text; registered 7,488 + context 591 = 8,079; RTX 5090, R&D 2026-10-20
        dim=1024,
        max_tokens=8192,
    ),
    "qwen3-emb-0.6b": EmbedderSpec(
        name="qwen3-emb-0.6b",
        repo="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        licence="Apache-2.0",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=15065,  # conservative whole-card figure 15,065 MiB: registered 14,470 (rung 2b) + context 595 (F1); F1 measured 13,661; RTX 5090, R&D 2026-10-20
        dim=1024,
        max_tokens=8192,
    ),
    "gte-modernbert": EmbedderSpec(
        name="gte-modernbert",
        repo="Alibaba-NLP/gte-modernbert-base",
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        licence="Apache-2.0",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=7112,  # conservative whole-card figure 7,112 MiB: registered 6,544 (rung 2b) + context 568 (F1); F1 measured 6,022; RTX 5090, R&D 2026-10-20
        dim=768,
        max_tokens=8192,
    ),
    "dinov2": EmbedderSpec(
        name="dinov2",
        repo="facebook/dinov2-base",
        revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        licence="Apache-2.0",
        modalities=("image",),
        loader="dinov2",
        vram_mib=1272,  # conservative whole-card figure 1,272 MiB: registered 636 + context 636 (F1); the dogfood measured 1,111 and F1 1,158; RTX 5090, R&D 2026-10-20
        dim=768,
        max_tokens=0,
    ),
}

# Provisional task defaults from calibration rungs 1 and 2 on the 5090.
TASK_DEFAULTS: dict[str, str] = {
    "images": "siglip2",
    "image-variety": "dinov2",
    "dupes": "dinov2",  # provisional: DINOv2 two-signal duplicate rule (rung 2)
    "text": "bge-m3",
    "code": "eg2",
}


def default_for_task(task: str) -> str:
    """Return the default backend name for a task."""
    if task not in TASK_DEFAULTS:
        raise KeyError(f"unknown task {task!r}; expected one of {sorted(TASK_DEFAULTS)}")
    return TASK_DEFAULTS[task]


def resolve_backend(backend: str | None, task: str = "images") -> str:
    """Return `backend` when explicitly passed, else the task's default from TASK_DEFAULTS."""
    if backend is not None and backend != "":
        return backend
    return default_for_task(task)
