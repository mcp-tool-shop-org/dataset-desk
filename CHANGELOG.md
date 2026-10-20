# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `release.yml`: publishing a GitHub release builds the wheel and sdist, checks them with `twine check`, and publishes
  to PyPI by trusted publishing from the `pypi` environment, with SLSA provenance. It never fires on a push.

### Notes

- No NumPy upper bound. The desk doesn't use numba. An environment that adds `numba-cuda` beside it needs
  `numpy>=2.4,<2.5` until numba-cuda supports NumPy 2.5 (studio smoke rung on the RTX 5090, 2026-10-20).

## [0.1.2] - 2026-10-15

### Added

- `report --count-tokens` runs the gate's 64-token test with the cached SigLIP2 tokenizer, offline. Before it, the
  command line never passed a token counter, so a record that passed every other test stayed `unchecked`.
- Error code `DEP_TOKENIZER_MISSING` (exit 1) when `--count-tokens` can't find the tokenizer.

### Fixed

- `report` no longer stops when `--out` and the project are on different Windows drives: links fall back to
  `file://` URIs where a relative path can't cross drives.
- The tokenizer load no longer logs the bos/eos config warning, like the model load in 0.1.1.

### Changed

- 152 offline tests; line coverage 96%, CI floor 90.

## [0.1.1] - 2026-10-15

### Fixed

- Model loads pass `dtype` instead of the deprecated `torch_dtype` (SigLIP2, and EmbeddingGemma 2's `model_kwargs`).
- SigLIP2 loads no longer log transformers' bos/eos token-id warning. Its published config sets those ids outside
  the text vocabulary, and the desk never uses them. Errors still show.

### Changed

- The `models` extra now needs `transformers>=4.56` (the first with `dtype`) and `sentence-transformers>=5`.
- 145 offline tests; line coverage 96%, CI floor 90.

## [0.1.0] - 2026-10-15

The first public release: a prototype, tested offline and not yet calibrated on real data.

### Added

- `dataset-desk` command with `index`, `dupes`, `search`, `coverage` and `report`.
- Backends: EmbeddingGemma 2 (`eg2`), SigLIP2 so400m (`siglip2`), and `none`, the torch-free lite tier.
- Duplicates in three tiers: byte-identical (sha256), near (256-bit dHash) and semantic (embedding cosine).
- A 30-number look channel (palette, tone, native-resolution texture), and outliers ranked by content and by look.
- The model-free description gate: prompt echo, a wrapped subject line, length, pool duplicates and the SigLIP2
  64-token cap, with a frozen stopword list whose sha256 is exposed.
- `description-review.jsonl` and a blind re-description worksheet: each flagged image beside its nearest siblings,
  with no prompt or old text in view.
- An optional blind drafter through plain-sight (Florence-2), offline at a pinned revision, recorded as
  `captioner_assisted`.
- Guards: `--out` may not sit inside the project; a GPU needs `--grant`; locked benchmark projects refuse description
  embeddings and model-based acceptance without `--allow-multimodal-pool`.
- Structured errors (`error <CODE>: <message>` plus a hint), exit codes 0/1/2, and `--debug` for stack traces.
- 142 offline tests at 96% line coverage, with CI holding a 90% floor.
