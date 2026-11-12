# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- **Colab lane, before the first real L4 window (private training images go to the VM):**
  - Cleanup kills only its own descendants: tracked PIDs are signalled only if alive, no older than the launcher, and
    their parent chain reaches the launcher (a recycled PID is never touched).
  - No rig paths in kept files: every text file under the out dir is scrubbed at exit (home, WSL/Windows user dirs,
    drive roots, the studio config path become `<rig>`, `<out>`, `<project>`, `<repo>`, `<studio-config>`).
  - `rejected/` (any case) never uploads in any mode; every uploaded asset must be an image by magic bytes (AVIF and
    HEIC now detected); a non-image asset refuses `PROJECT_ASSET_NOT_IMAGE`.
  - A bundle that trips `PRIVATE_IMAGE_IN_BUNDLE` on the rig is deleted by the staging clean; its sha, size and the
    refusal stay in `private_image_bundle_refusal.json`.
  - L4 is Standard only (Publisher fleet ruling 2026-11-09, polyglot-mcp 66eddd9): default shape `standard`;
    `--shape high-mem` refuses `SHAPE_HIGH_MEM_UNAVAILABLE` before any colab call; V5E1/V6E1 stay single high-mem.
    The vendored `colab_window.sh` now differs from the sdlab df9a1b3b export by exactly that rule.
  - Weight download verifies all 16 files of the pinned revision: 4 LFS shards by sha256 and the 12 plain git files by
    git blob id (`WEIGHT_BLOB_MISMATCH`, `WEIGHT_BLOB_ID_MISSING`, `WEIGHT_FILE_MISSING`); the record carries
    `sha_verified_count` and `blob_verified_count`.
  - Coverage unchanged (the lane scripts sit outside the measured `dataset_desk` package; floor 90 stays).

### Added
- **Colab caption-drafting lane (`scripts/colab/`):**
  - Vendors `style-dataset-lab` window library (`colab_window.sh`, `launch_guards.py`, `remote_runner.py`) at commit
    `df9a1b3be2f8729a57776253e9b0cb32b7f5299c` with SHA-256 input verification and `launch_guards.sha256_self_normalized`
    self-pinning.
  - Adds `scripts/colab/launch_desk_draft.sh`, `scripts/colab/desk_draft_helper.py`, `scripts/colab/desk_bundle_now.py`,
    and hash-locked `scripts/colab/locks/desk_draft.lock` with `--dry`, `--stub` (`CPU-high-mem` real install +
    weight prefetch & sha check + offline `AutoConfig` load + `compose --draft fake` on 2 synthetic images), and GPU
    drafting (`L4`/`A100`/`H100`, VRAM >= 22,000 MiB, disk & host-RAM guards, rig-built wheel + uv venv install,
    local-disk `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` fetch with mode-600 token
    shredding, HTTP 429 backoff inside download bound, `HF_HUB_OFFLINE=1`, per-record heartbeat, chunked private-image
    upload & verified output fetch, VM workdir deletion before stop, and non-null `refusal_code`/`why` markers with
    HTTP 412 never-retry vs HTTP 503 retry-later classification).
  - Adds `EMBEDDING_BACKEND_FORBIDDEN` (`DATASET_DESK_FORBID_EMBEDDERS=1`), `DATASET_DESK_EMBEDDER_LOG_FILE`, and
    `DATASET_DESK_DRAFT_HEARTBEAT_FILE` support in `dataset_desk/backends.py` and `dataset_desk/errors.py`.
  - Privacy and bounds hardening: compose-allowlist project upload tar (`prune_project_walk_dirs`, followed-link cycle
    refusal `PROJECT_LINK_CYCLE`, and `uploaded_file_count` / `uploaded_total_bytes` window telemetry without file
    names), path-free `pool_detection` and manifests/window records, rig chunk-staging cleanup across normal exits,
    refusals, guard trips, and signals, `+70 s` VM-cleanup step reserve in `COLAB_CLEANUP_STOP_ATTEMPTED_BY_S` and
    `COLAB_CLEANUP_BOUND_S`, separate `--model-load-stall-s` (`420` s default, `MODEL_LOAD_STALLED`) and
    `--desk-cmd-timeout-s` (`300` s default, `DESK_COMMAND_TIMEOUT`), and magic-byte image inspection on output bundles.

## [0.3.0] - 2026-11-12

### Added
- **Training-caption commands and options:**
  - `scan`: scans a `--project` into `<out>/scan.jsonl` and `<out>/scan-report.json` without loading an embedder.
  - `sidecar-import` and `--sidecars` (on `scan`, `index`, `report`, and `compose`): pairs image files with same-stem
    `.txt` caption sidecars when `records/` is absent, extracts embedded ComfyUI PNG `tEXt` generation prompts and wave
    `generation.json` prompts for prompt-echo checks, reports images without a `.txt` sidecar instead of inventing text,
    and writes `<out>/sidecar-records.jsonl` and `<out>/sidecar-import.json` strictly under `--out`.
  - `compose` (and `report --compose`, `--manifest-name`, `--all`): assembles training captions into
    `<out>/captions-manifest.json` (`style-dataset-lab` lane-3 `run_dataset` format) from per-project
    `caption-rules.json` (`--caption-config`: `prefix`, `classes`, `class_field`, `class_pattern`, `backdrop`,
    `banned_words`, `drafter`, `drafter_prompt`) and imported seat edits, with per-part `writers` provenance
    (`config:<file>:<line>`, `drafter:<id>@<rev>:<prompt sha>`, `seat:<name>`, `record:<file>`, `sidecar:<file>`) and
    per-item `gate_status` (`"passed"`, `"failed"`, or `"unchecked"`), `flags`, and `reasons` on every manifest item
    (preserving keep-all while marking flagged items for downstream trainers). By
    default, `compose` and `report --compose` draft **only** records flagged by the model-free gate into the review
    queue (`description-review.jsonl` and `description-worksheet.md`) — never straight into the training caption — and
    leave unflagged records untouched; `--all` also drafts unflagged records as side-by-side offers (`draft_offer` and
    `draft_offer_writer`), and human and sidecar text is never replaced unless a seat edit is imported.
  - `import-edits` (`--worksheet`, `--seat`): re-imports edited rows from `description-worksheet.md` into
    `<out>/seat-edits.jsonl` with `seat:<name>`, recording the draft text and drafter writer the seat saw.
  - **Drafter registry (`registry.DRAFTER_REGISTRY`):** adds `qwen3-vl` (`Qwen/Qwen3-VL-8B-Instruct` pinned at
    `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, Apache-2.0, `is_arm_model=False`, VRAM-governed, `HF_HUB_OFFLINE`
    aware, raw output logged to `<out>/drafter-raw.jsonl`) alongside default `plain-sight` (`--draft`,
    `--drafter-prompt`).
  - **Pool lock tag, fail-closed renamed-copy sources, user config, and `pool_detection`:**
    - `"multimodal_pool"` project tag in `caption-rules.json`, `project.json`, `desk-config.json`, or
      `--caption-config`.
    - Locked-pool hash/directory configuration via `--pool-manifest` / `DATASET_DESK_POOL_MANIFEST`,
      `--locked-pool-dir` / `DATASET_DESK_POOL_DIRS`, and user config (`DATASET_DESK_USER_CONFIG`,
      `~/.config/dataset-desk/config.json`, or `%APPDATA%\dataset-desk\config.json`).
    - Per-run `pool_detection` (`status: "on" | "off"`, configured pools, `stem_check` status, per-pool and per-source
      image-hash counts, and source paths — never hash values) recorded across image, caption, and text commands. When a
      hashes-only manifest provides no image stems or paths, `pool_detection` records `"stem_check": "no stems
      configured"` (or names the stem-less pools) and text commands emit a single `stderr` warning noting that `.txt`
      sidecar stem matching had no stems to compare.

### Breaking changes
- **Stderr warning when renamed-copy detection is `off`:** every `--project` command (`scan`, `index`, `dupes`,
  `search`, `coverage`, `report`, `sidecar-import`, `compose`, `import-edits`) and every text command (`text-index`,
  `text-dupes`, `text-search`, `lineage`) prints a single warning line to `stderr` when no pool manifest or locked-pool
  folder is configured.
- **Project and source config files are read and validated:** `caption-rules.json`, `project.json`, `desk-config.json`
  (and `--caption-config` when passed) are read on `--project` and `--source` / `--rnd-data` commands; invalid JSON, a
  non-object file, a non-string or unknown `multimodal_pool` (`INPUT_BAD_ARGUMENT`), a `caption-rules.json` schema
  error, or a `multimodal_pool` tag conflicting with a locked folder's `realpath` name (`PERM_LOCKED_POOL`) exits `1`.
- **User config and pool sources fail closed on malformed or empty inputs:** the user config
  (`DATASET_DESK_USER_CONFIG` when non-empty, else `~/.config/dataset-desk/config.json` or
  `%APPDATA%\dataset-desk\config.json`) and any configured pool manifest, environment variable, or pool folder are read
  on every command; malformed JSON, a non-object user config or manifest, a non-object `"pools"` value
  (`INPUT_BAD_ARGUMENT`), or **any** explicitly configured but empty pool source — an empty manifest or empty user
  config, an empty `pool_manifest` or `locked_pool_dirs` / `pool_dirs` in user config or a manifest, an empty or
  separator-only `DATASET_DESK_POOL_DIRS` or `--locked-pool-dir`, an empty `DATASET_DESK_POOL_MANIFEST` or
  `--pool-manifest`, or any configured pool entry or folder yielding zero image hashes (`INPUT_EMPTY_POOL_SOURCE`) —
  fails every command with exit `1`. Set `DATASET_DESK_USER_CONFIG=""` to ignore user config files.
- **Fail-closed pool lock across all project and text commands (links included):** an untagged `--project` or
  `--source` / `--rnd-data` folder sharing any image `sha256` with a configured pool (following symlinks and Windows
  directory junctions via `enumerate_project_images`) is refused (`PERM_LOCKED_POOL`, exit `1`) for all commands —
  `scan`, `dupes`, `--backend none`, `sidecar-import`, `compose`, and `text-*` included — unless
  `--allow-multimodal-pool` is passed:
  - One shared directory-pruning rule (`desk.prune_project_walk_dirs`) governs both project reads and `sha256` pool
    hashing (pruning `.git` and `__pycache__` at every depth and only top-level `<project>/records`, so images under
    `outputs/records/` or `outputs/<wave>/records/` are hashed and refused identically to scan reads).
  - Every `asset_path` named in `<project>/records/*.json` is hashed regardless of file extension, refusing renamed
    byte-copies such as `.dat`.
  - Explicitly configured empty pool sources outside a manifest (empty `locked_pool_dirs` / `pool_manifest` in user
    config, or separator-only `DATASET_DESK_POOL_DIRS`) fail closed (`INPUT_EMPTY_POOL_SOURCE`) instead of falling back
    to `"off"`.
  - Text commands (`text-index`, `text-dupes`, `text-search`, `lineage`) also refuse untagged sources or cached text
    indexes containing `.txt` files whose stem matches an image of a locked pool, and refuse embedding backends on named
    or tagged locked pools without `--allow-multimodal-pool`. Folder-name pool checks compare `realpath` rather than
    `abspath`.
- **New output files under `--out` and new `pool_detection` / provenance fields:** commands write `pool-detection.json`,
  `scan-report.json`, `<cmd>-<backend>.json` (`dupes`, `search`, `coverage`, `text-dupes`, `text-search`, `lineage`),
  `drafter-raw.jsonl`, `seat-edits.jsonl`, `captions-manifest.json`, `sidecar-import.json`, and
  `sidecar-records.jsonl` under `--out`; `<out>/<backend>/meta.json`, `<out>/<backend>/text/meta.json`, and
  `description-review.jsonl` gain `pool_detection` (and review rows gain provenance fields including `writers`,
  `parts`, `draft`, `drafter_writer`, `draft_offer`, `draft_offer_writer`, `seat_edit`, `seat_id`, and
  `final_caption`). New 0.3.0 JSON-stdout commands
  (`scan`, `sidecar-import`, `import-edits`) include `pool_detection` in stdout JSON; existing 0.2.x JSON-stdout
  commands (`dupes`, `text-dupes`, `lineage`) keep their stdout JSON shapes unchanged.
- **`--draft` takes an optional value; `compose` drafts only flagged records into the review queue:** `--draft` now
  accepts an optional drafter name (`none`, `fake`, `plain-sight`, `florence2`, `qwen3-vl`), and a bare `--draft` means
  `plain-sight`. `compose` (and `report --compose`) drafts only records flagged by the model-free gate into the review
  queue (using `rules.drafter` if set in `caption-rules.json`, else `plain-sight`; pass `--draft none` to skip
  drafting), leaving unflagged records untouched and never writing unedited drafts into training captions; `--all`
  offers drafts beside unflagged originals (`draft_offer` / `draft_offer_writer`), and human and sidecar text is never
  replaced without an imported seat edit.
- **Python API, provenance strata, and error codes:** `desk.check_multimodal_pool` returns the `pool_detection` `dict`
  (was `None`); `desk.check_drafter_pool` defaults `getattr(drafter, "is_arm_model", True)` so unknown drafter objects
  are treated as arm models; sidecar descriptions carry `author_kind: "sidecar"` (never `"human"`) and `"sidecar"` is
  counted in `strata_counts`; flagged drafted review rows align `caption_stratum` with `author_kind`
  (`"captioner_assisted"`); `audit_records` initializes `writers` as a `list[str]` and `import-edits` appends
  `seat:<name>` onto `writers` rather than overwriting; added `MODEL_REVISION_MISMATCH`, `DEP_DRAFTER_OFFLINE`, and
  `INPUT_EMPTY_POOL_SOURCE`; updated the `INPUT_UNKNOWN_DRAFTER` hint.

### Known limits
- Renamed-copy detection matches images by file `sha256`: neither a copy made only of new images (none shared with a
  locked pool) nor a resized, re-encoded, or metadata-stripped copy matches by `sha256`. Tag such projects with
  `"multimodal_pool"` in `caption-rules.json` or `project.json`.

## [0.2.2] - 2026-10-20

0.2.1 was tagged on GitHub but never published to PyPI. 0.2.2 ships 0.2.1's changes, listed below, with one
security update to the release pipeline.

### Security
- The release workflow now pins `pypa/gh-action-pypi-publish` v1.14.3 (commit 7245906f), which includes the fix for
  [GHSA-5xjx-3w67-w9x9](https://github.com/pypa/gh-action-pypi-publish/security/advisories/GHSA-5xjx-3w67-w9x9)
  (pypa#427: certifi and sigstore upgraded) and hash-checks the action's
  runtime dependencies. It follows the
  [eMudhra/emSign CA certificate incident](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576)
  (2026-10-20). 0.2.1's release run, which used v1.14.2, was stopped before upload.

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

## [0.2.1] - 2026-10-20

Not published to PyPI: superseded by 0.2.2 (publish-action security update). Its changes ship in 0.2.2.

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
