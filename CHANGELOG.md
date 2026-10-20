# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.1] - 2026-10-20

### Changed
- `registry.REGISTRY`: the `vram_mib` figures are now **conservative whole-card figures**. Each is the larger of the
  highest real-run peak measured and the earlier reserved figure plus that backend's measured CUDA context, on an
  RTX 5090, so `need_mib = vram_mib + 1,024 MiB` covers CUDA context overhead on a shared GPU.
  - A real run's peak can exceed a short synthetic one, because the allocator's reserved pool grows over many
    batches; the 1,024 MiB margin is the buffer for that.
  - Other cards and corpora may differ.

  | backend | vram_mib | source |
  |---|---|---|
  | dinov2 | 1,272 | registered 636 + context 636 |
  | siglip2 | 6,207 | 0.2.0 dogfood's whole-card peak on real images |
  | gte-modernbert | 7,112 | registered 6,544 + context 568 |
  | bge-m3 | 8,442 | 0.2.0 dogfood's whole-card peak on rnd-data text |
  | eg2 | 13,119 | registered 12,550 (multimodal) + context 569 |
  | qwen3-emb-0.6b | 15,065 | registered 14,470 + context 595 |

- `desk.scan`: about 2× faster per image on one thread, with identical results (`scan.jsonl` rows and `SCAN_V = 2`
  are unchanged). The first uncached scan runs in parallel under `--threads` (a `ProcessPoolExecutor` when
  `threads > 1` and there are at least 8 uncached images), and since the CLI's default is `--threads 4`, a default
  run is now parallel.
- `description_gate.audit_records`: replaced the full-postings Jaccard candidate loop with rare-first prefix filtering
  (`L - ceil(tau * L) + 1`), producing identical `near_pairs` and `near_groups`.
- `desk.mid_rank` and `desk._rank_descriptions`: with `_RANK_TIE_EPS = 1e-6`, tied image vectors now stay tied
  whatever the BLAS block size.
- `desk.lineage`: keeps only the top k pairs per file name, in bounded memory.
  - Its cosines come from a blocked float32 matrix product.
  - So a pair whose cosine sits on a 4-decimal rounding boundary can differ from 0.2.0 by 0.0001, and the order of
    such near-ties can differ with it.
- `desk.pairs_at_least`, `desk.top1_percentiles`, `desk.coverage`: bounded block sizing via `_block_rows` and
  single-slice `XA = img[A]`.
- `backends.transformers_warnings_off`: also disables and restores transformers' weight-loading progress bars across all model loaders.

### Changed field: `max_content_jaccard`
- `description_gate.audit_records`: a record's `max_content_jaccard` is now the exact value at ≥ 0.85 and `null`
  below. The flags are unchanged.

### Fixed
- `backends._vram_info_mib`: catches CPU-only torch builds when `--device cuda` is requested and raises `ENV_NO_CUDA_TORCH` with a hint to install CUDA torch from `https://download.pytorch.org` or pass `--device cpu`.
- `desk.check_multimodal_pool`, `desk.index`, `desk.report`: image-only backends (`dinov2`) no longer require `--images-only` on locked multimodal benchmark pools when called from Python or CLI.

## [0.2.0] - 2026-10-20

### Added

- **Text and code input:** `text-index`, `text-dupes`, `text-search` and `lineage` read a `--source` folder or an
  rnd-data checkout.
  - rnd-data's private companion and blob stores are refused.
  - Only relative paths are printed.
- **An embedder registry** with six models, each pinned to a full commit, with its licence and VRAM figure:
  - EmbeddingGemma 2 and SigLIP2;
  - bge-m3, Qwen3-Embedding 0.6B and gte-modernbert;
  - DINOv2.

  Each task has a default (`--task`): bge-m3 for text, EmbeddingGemma 2 for code, DINOv2 for duplicates, SigLIP2 for
  image search.
- **A VRAM governor.**
  - It refuses a load when used plus needed memory would pass `--vram-budget` (default 26,624 MiB).
  - `--wait` polls until the load fits.
  - Every refusal names used, free, needed and the budget.
- **EmbeddingGemma 2 is used the way its model card says:**
  - a task prefix for each role, with file names as document titles;
  - text-only loading for text commands;
  - `--dim` (768, 512 or 256), and `--vision-tokens`.
- **Chunks counted by each model's own tokenizer:** up to the model's limit (8,000 or 8,192 tokens), at most 8 per
  file.
- **Duplicate output names its rule:** `dupes` reports `rule` and `view_pairs_skipped`, and `text-dupes` reports
  `top1_percentiles`.
- **A mirror-aware dHash:** `dhash_flip`, computed from the flipped image.

### Changed

- **Duplicate rules, from the studio's calibration on the RTX 5090.** Every threshold is marked provisional.
  - On a DINOv2 index, an image pair is a duplicate when `(cos ≥ 0.94 and flip-aware dHash within 44 bits) or
    cos ≥ 0.985`. That's recall 0.91 with no cross-project false positives on the studio's pool.
  - Other backends use the mirror-aware dHash within 26 bits, plus cosine ≥ 0.95.
- **Two views of one object are never duplicates.** The view comes from the record's `provenance.asset_id` and
  `render_id`, or from a file name like `<asset>__y<yaw>_e<elev>`.
- **`text-dupes` defaults:** 0.98 on bge-m3, and 0.95 for `--task code` on EmbeddingGemma 2. Any other backend needs
  `--semantic`.
- **The registered VRAM figures for the text models are long-input peaks** (bge-m3 7,488, Qwen3 14,470,
  gte-modernbert 6,544 MiB).
- **Caches record and check every setting that changes a vector:** revision, dim, vision tokens, max tokens, chunks,
  role and dtype. A non-default `--dim` or `--vision-tokens` gets its own folder.
- **`scan.jsonl` rows are versioned** (`scan_v: 2`). Older rows are recomputed.

### Fixed

- **Large projects no longer run out of memory or take hours.** Measured on synthetic sets, before → after:
  - the near-duplicate search (dHash, flip-aware) was a Python loop over every pair: 27 s at 4,000 images, about 5
    hours at 100,000. It is now vectorised in blocks: 0.7 s at 4,000, 4.4 minutes at 100,000. Same pairs, same order.
  - `look_fit` built an n × n × 30 array: 3.6 GB at 4,000 images, so it crashed long before 100,000. It now works in
    blocks with a matrix-product distance: 3.1 minutes at 100,000, under 400 MB.
  - `outliers` built the full n × n similarity matrix (about 76 GB at 100,000). It now works in blocks: 21 s at
    50,000.
  - The description gate kept every pair of identical captions in memory, and compared every description with every
    other one. It now indexes content words, so only descriptions that share a word are compared: about 6× faster,
    with identical output (checked on 300 random record sets).
  - `report`'s swap check no longer builds an m × m comparison, and it and the caption check rank descriptions in
    blocks, so large described sets no longer run out of memory or take hours.
- Every pairwise block is capped at about 64 MiB (`BLOCK_BYTES`).

### Known limits

- The description gate's exact `max_content_jaccard` for every record is still quadratic when a few content words
  appear in most descriptions (74 s at 32,000 records on a heavy-vocabulary synthetic set). A faster exact threshold
  check that changes what that field reports is a design question, open.

## [0.1.3] - 2026-10-20

### Fixed

- `dataset-desk[models]` now installs `sentence-transformers[image]`. Without it, a fresh install couldn't load
  EmbeddingGemma 2's processor (found by the studio's calibration run on 2026-10-20).

### Changed

- Install from PyPI: `pip install dataset-desk`, or `pip install "dataset-desk[models]"`.

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
