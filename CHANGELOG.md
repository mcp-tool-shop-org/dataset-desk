# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
