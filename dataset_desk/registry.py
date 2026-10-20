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
        licence="Apache-2.0",
        modalities=("text", "image"),
        loader="eg2",
        vram_mib=12550,
        dim=768,
    ),
    "siglip2": EmbedderSpec(
        name="siglip2",
        repo="google/siglip2-so400m-patch14-384",
        revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        licence="Apache-2.0",
        modalities=("image", "text"),
        loader="siglip2",
        vram_mib=5340,
        dim=1152,
    ),
    "bge-m3": EmbedderSpec(
        name="bge-m3",
        repo="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        licence="MIT",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=1674,
        dim=1024,
    ),
    "qwen3-emb-0.6b": EmbedderSpec(
        name="qwen3-emb-0.6b",
        repo="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        licence="Apache-2.0",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=3814,
        dim=1024,
    ),
    "gte-modernbert": EmbedderSpec(
        name="gte-modernbert",
        repo="Alibaba-NLP/gte-modernbert-base",
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        licence="Apache-2.0",
        modalities=("text",),
        loader="sentence_transformer",
        vram_mib=954,
        dim=768,
    ),
    "dinov2": EmbedderSpec(
        name="dinov2",
        repo="facebook/dinov2-base",
        revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        licence="Apache-2.0",
        modalities=("image",),
        loader="dinov2",
        vram_mib=636,
        dim=768,
    ),
}

# Provisional task defaults from calibration rung 1 on the 5090.
TASK_DEFAULTS: dict[str, str] = {
    "images": "siglip2",
    "image-variety": "dinov2",
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
