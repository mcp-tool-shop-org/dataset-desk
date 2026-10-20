<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/dataset-desk/readme.png" alt="dataset-desk" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/dataset-desk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk"><img src="https://codecov.io/gh/mcp-tool-shop-org/dataset-desk/graph/badge.svg" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/dataset-desk/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

<p align="center">
  A desk for making and reviewing visual datasets: find the copies, the near-copies and the odd ones out,<br>
  search a set by words or by picture, and check that every image's description really describes it.
</p>

---

## Why

A style LoRA learns from its training images and its captions. Two quiet defects spoil both:
- **Repeats.** The same picture saved twice, or two renders a few pixels apart, teach one image twice.
- **Descriptions that aren't descriptions.** A caption copied from the generation prompt, or one ruling pasted onto 54 images, tells the model nothing about the picture. Worse, if that text is later used to test search, it measures recall of the prompt, not understanding of the image.

dataset-desk finds both, on a project folder, without changing it. It writes a report and a review queue for a person to act on. **Nothing is removed automatically.**

## What it does

| Command | What you get |
|---|---|
| `index` | Scans the project (records, wave metadata, PNG generation data) and embeds every unique image into a cache under `--out` |
| `dupes` | Byte-identical copies (sha256), near-duplicates (a mirror-aware 256-bit dHash) and semantic duplicates (embedding cosine), as JSON. On a DINOv2 index it uses a calibrated two-signal rule. Two views of one object are never paired. |
| `search` | The nearest images to a description (`--text`), a picture (`--image`), or both together |
| `coverage` | The best images for each subject in a wanted-subjects list, so gaps show |
| `report` | A Markdown report: duplicates, the least typical approved images by content and by look, the description audit, and whether each description finds its own image |
| `text-index` | Embeds the text and code files of a `--source` folder (or an rnd-data checkout), counting chunks with the model's own tokenizer |
| `text-dupes` | Exact and near-duplicate files. `--task code` compares scripts only. |
| `text-search` | The nearest files to a query (`--text`). `--task code` uses the model's code-retrieval prompt. |
| `lineage` | Versions of the same script across runs, ranked by similarity |

**The thresholds** come from the studio's calibration on its own data. Every one is marked provisional in the code, with the measurement it came from:
- **Images:** on a DINOv2 index, a pair is a duplicate when `(cosine ≥ 0.94 and the mirror-aware dHash is within 44 bits) or cosine ≥ 0.985`. Other indexes use the dHash within 26 bits, plus cosine ≥ 0.95.
- **Text:** 0.98 with bge-m3, and 0.95 for code with EmbeddingGemma 2. Any other model needs `--semantic`.

Two renders of the same object from different camera angles are never duplicates. The view comes from each record's `provenance.asset_id` and `render_id`, or from file names like `<asset>__y<yaw>_e<elev>`.

**The description gate** runs without any model. A description is flagged when it:
- echoes its prompt: it equals the prompt or sits inside it, shares a run of 6+ words with it, or 90%+ of its words (80%+ of its content words) appear in it;
- wraps the record's subject line: a 4+ word `canon.subject` appears as a contiguous run of its words;
- is too short: under 8 words or under 5 content words;
- duplicates another description in the set: identical once punctuation is removed, or a content-word Jaccard of 0.85 or more;
- is too long for SigLIP2: more than 64 tokens (the one check that needs a tokenizer; unchecked records are never marked as passing).

Flagged records go to `description-review.jsonl` and to a **blind worksheet**: each flagged image beside 3–5 visually nearest siblings, with no prompt, no subject line and no old text in view, so the person rewriting it describes the picture. An optional drafter ([plain-sight](https://github.com/mcp-tool-shop-org/plain-sight), Florence-2) can write a first draft from the pixels alone; drafts are recorded as `captioner_assisted`, and a person edits them.

**The look channel** is 30 numbers per image (palette, tone, and texture at native resolution), so "this one's rendering is off" shows separately from "this one's subject is odd".

## Install

```bash
pip install dataset-desk
```

That gives the **lite tier**: numpy and Pillow only, no torch. The duplicate checks, the look channel, the description gate and the report all work in it (`--backend none`).

For the embedding index (search, coverage, semantic duplicates, the description fit check), add the model libraries:

```bash
pip install "dataset-desk[models]"
```

On Windows with an NVIDIA GPU, install a CUDA build of torch from `https://download.pytorch.org` first; PyPI's Windows torch is CPU-only.

The models are fetched from Hugging Face at pinned revisions the first time they're used, then load from your cache. Each task has a default, and `--backend` picks another:

| Backend | Model | Default for |
|---|---|---|
| `dinov2` | [DINOv2](https://huggingface.co/facebook/dinov2-base) | `dupes` |
| `siglip2` | [SigLIP2 so400m](https://huggingface.co/google/siglip2-so400m-patch14-384) | `index`, `search`, `coverage`, `report` |
| `bge-m3` | [bge-m3](https://huggingface.co/BAAI/bge-m3) | text |
| `eg2` | [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) | code (`--task code`, `lineage`) |
| `qwen3-emb-0.6b`, `gte-modernbert` | [Qwen3-Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | — |

EmbeddingGemma 2 is used the way its model card says: a task prefix per command, text-only loading for text, and `--dim 768|512|256`. Set `HF_HUB_OFFLINE=1` to forbid downloads. The plain-sight drafter is always offline: it never downloads.

**Runs on:** Python 3.11 and 3.12, on Windows and Linux. CPU by default, with every thread pool capped (4 threads unless `--threads` says otherwise), and a model run drops the process to below-normal priority, so it can share a machine.

## Usage

```bash
# The lite pass: no model, no GPU
dataset-desk index  --project path/to/project --backend none --out desk-out/project
dataset-desk report --project path/to/project --backend none --out desk-out/project

# With an embedding index
dataset-desk index    --project path/to/project --backend siglip2 --out desk-out/project
dataset-desk search   --project path/to/project --backend siglip2 --out desk-out/project --text "a squat bog creature"
dataset-desk coverage --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt
dataset-desk report   --project path/to/project --backend siglip2 --out desk-out/project --subjects wanted.txt

# Image duplicates with the calibrated DINOv2 rule
dataset-desk index --project path/to/project --backend dinov2 --out desk-out/project
dataset-desk dupes --project path/to/project --out desk-out/project

# Text and code files
dataset-desk text-index  --source path/to/folder --out desk-out/texts               # bge-m3, for text
dataset-desk text-dupes  --source path/to/folder --out desk-out/texts
dataset-desk text-index  --source path/to/folder --out desk-out/texts --task code   # EmbeddingGemma 2, for code
dataset-desk text-search --source path/to/folder --out desk-out/texts --task code --text "where the cache is written"
```

`dataset-desk --help` lists every flag. Exit codes: `0` ok, `1` a user error or a refusal, `2` an unexpected error. Errors print as `error <CODE>: <message>` with a `hint:` line; `--debug` adds the stack trace.

**The project layout it reads** is [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab)'s:

```text
project/
  records/*.json                  one per image: judgment (status, explanation), canon (subject), provenance (prompt)
  outputs/<wave>/*.png            the images; a wave's generation.json and _curation.json are read when present
```

A project with no `records/` works too, as long as the images sit in folders under `outputs/` (files at the top of `outputs/` are skipped as contact sheets): every image is "unrecorded", and the duplicate and look checks still run.

## Threat model and trust

- **What it reads:** the project folder you name with `--project`: images, `records/*.json`, wave metadata, and the generation data embedded in PNG files. The text commands read the `--source` folder or `--rnd-data` checkout you name. They refuse rnd-data's private companion and blob stores, and they print relative paths only.
- **What it writes:** only under `--out`: its scan cache, embedding caches, the report, `description-review.jsonl` and the worksheet. It refuses an `--out` inside the project (`PERM_OUT_INSIDE_PROJECT`), so a project is never modified.
- **What it never touches:** anything outside those two folders. It deletes nothing, anywhere.
- **Network:** the lite tier never uses it. A model backend fetches its pinned weights from Hugging Face on first use, unless `HF_HUB_OFFLINE=1` is set; the drafter loads with `HF_HUB_OFFLINE=1` and `local_files_only=True` every time. No model runs remote code.
- **Permissions:** an ordinary user account. A GPU needs `--device` plus a `--grant` id, so a shared card isn't taken by accident. Before loading a model, a VRAM governor checks that memory already in use plus the model's measured need fits under `--vram-budget` (default 26,624 MiB). It never evicts anything; `--wait` polls until the load fits.
- **Secrets and telemetry:** it reads no credentials, and it collects and sends no telemetry.
- **Guarded steps:** on projects locked for a running benchmark, description embeddings and any model-based acceptance of descriptions are refused unless `--allow-multimodal-pool` is passed, so the benchmark's queries aren't shaped by the models it measures.

Report a vulnerability as described in [SECURITY.md](SECURITY.md).

## Status

**v0.2.0, a public prototype.** The code is tested offline: 188 tests and 95% line coverage, with small stand-ins for the model libraries.
- **Calibration:** the studio's runs on its own data, on an RTX 5090, set the duplicate thresholds and the VRAM figures. They're provisional, and they rest on one pool per kind. The code thresholds rest on 17 planted copies. The method and the data are in [newsletter issue 16](https://mcp-tool-shop-org.github.io/rnd/newsletter/issue-16/).
- **Large projects:** duplicates, outliers and the look channel work in bounded blocks. On synthetic sets, the near-duplicate search and the look channel take minutes at 100,000 images, where they took hours or ran out of memory. The description gate is about 6× faster. Its Jaccard check still slows sharply on very large sets that share many words, and that design question is open.
- **Expect the command line to change before 1.0.**

The design record, its research grounding and its review history live in the studio's research library: [rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).

## Development

```bash
git clone https://github.com/mcp-tool-shop-org/dataset-desk && cd dataset-desk
python -m pip install -e ".[dev]"
bash verify.sh                       # lint, the tests, and a lite-mode smoke run
python -m coverage run -m unittest discover -s tests && python -m coverage report
```

CI holds line coverage at 90% or more.

## License

[MIT](LICENSE). The models it can load are not part of this package, and each carries its own licence; read the model card before you use one.

---

<p align="center">Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a></p>
