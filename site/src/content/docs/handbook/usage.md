---
title: Usage
description: Each command, and how to read the report.
sidebar:
  order: 2
---

Every command takes `--project` (read) and `--out` (written). Use the same `--project`, `--backend` and `--out` for
`index` and the commands after it: they read the cache that `index` wrote.

## index

```bash
dataset-desk index --project my-project --backend siglip2 --out desk-out/my-project
```

Scans the project and embeds every unique image, and the approved descriptions, into `desk-out/my-project/<backend>/`.
A re-run embeds only what is new.

- `--limit N` indexes only the first N unique images: a smoke test before a long run.
- `--images-only` embeds images and skips descriptions.
- `--strict` stops on the first unreadable image instead of logging it.

## dupes

```bash
dataset-desk index --project my-project --backend dinov2 --out desk-out/my-project
dataset-desk dupes --project my-project --out desk-out/my-project > dupes.json
```

`dupes` defaults to the DINOv2 index. Prints JSON with:
- `exact`: groups of byte-identical files;
- `near` and `semantic`: pairs, with their dHash distance or cosine;
- `rule`: which duplicate rule ran;
- `thresholds`;
- `view_pairs_skipped`: pairs left out because they are two views of one object;
- `top1_percentiles`: each image's cosine to its nearest neighbour, the scale to set a semantic cut against.

**The rules.** The dHash is mirror-aware: a pair is compared both as is and with one image flipped left to right.
- On a DINOv2 index (`"rule": "dinov2-two-signal"`), a pair is a duplicate when the cosine is at least 0.94 and the
  dHash is within 44 bits, or the cosine is at least 0.985. `near` is empty, because the rule replaces it.
- On any other index, or with `--semantic`, `near` holds pairs within 26 bits and `semantic` holds the cosine pairs at
  0.95 (or your `--semantic`) that aren't already near (`"rule": "near+cosine"`).
- **Views are never duplicates.** Two items of the same asset with different views are never paired. The asset and view
  come from a record's `provenance.asset_id` and `render_id`, or from a file name like `<asset>__y<yaw>_e<elev>`.

The thresholds come from the studio's calibration and are provisional. Each is marked in the code with the measurement
it came from.

## search

```bash
dataset-desk search --project my-project --backend siglip2 --out desk-out/my-project --text "a squat bog creature"
dataset-desk search --project my-project --backend siglip2 --out desk-out/my-project --image ref.png -k 5
dataset-desk search --project my-project --backend eg2     --out desk-out/my-project --image ref.png --text "at night"
```

Prints `score  status  path`, best first. `--text` and `--image` together need `eg2`, which embeds a picture and words
as one query.

## coverage

```bash
dataset-desk coverage --project my-project --backend siglip2 --out desk-out/my-project --subjects wanted.txt
```

`wanted.txt` holds one subject per line (`#` starts a comment). For each subject it prints the three best images and
their scores. A subject whose best score is low is a gap in the set.

## report

```bash
dataset-desk report --project my-project --backend siglip2 --out desk-out/my-project [--subjects wanted.txt]
```

Writes `report-<backend>.md`, `description-review.jsonl` and `description-worksheet.md`.

### Reading the report

| Section | What to do with it |
|---|---|
| Duplicates | Look at each pair. Keep one of a byte-identical group; decide near and semantic pairs by eye. |
| Diversity (Vendi) | The effective number of distinct images. A set of 200 with a Vendi of 40 repeats itself. |
| Least typical of the approved set | Content-atypical but look-typical: probably an odd subject in the right style. Look-atypical: check its rendering. |
| Unapproved images nearest the approved set | Candidates that already look like canon. |
| Description audit | The gate's counts; the details are in `description-review.jsonl`. |
| Do the descriptions find their own images? | Each approved description searched against the set. A description that ranks its own image low may not describe it. |

## Re-describing flagged images

1. Open `description-worksheet.md`. Each row shows a flagged image, 3–5 visually nearest siblings, and an instruction:
   describe this picture in 15–35 words so a person could pick it out from the siblings.
2. Optionally, ask for a first draft from the pixels: `report --draft plain-sight` runs
   [plain-sight](https://github.com/mcp-tool-shop-org/plain-sight) (Florence-2) offline. Drafts are recorded as
   `author_kind: captioner_assisted`.
3. Write or edit the description in the worksheet. Rewrite a draft; don't click it through.
4. Bring the approved text back into the project with your own tooling (style-dataset-lab's importer is planned).

The worksheet never shows the prompt, the subject line or the old text, so the new description comes from the picture.

## Text and code files

The text commands work on a folder of text and code files (`--source`), or on an
[rnd-data](https://github.com/mcp-tool-shop-org/rnd-data) checkout (`--rnd-data`), instead of a `--project`.

```bash
dataset-desk text-index  --source my-notes --out desk-out/notes                   # bge-m3, for text
dataset-desk text-dupes  --source my-notes --out desk-out/notes
dataset-desk text-index  --source my-scripts --out desk-out/scripts --task code   # EmbeddingGemma 2, for code
dataset-desk text-dupes  --source my-scripts --out desk-out/scripts --task code
dataset-desk text-search --source my-scripts --out desk-out/scripts --task code --text "where the cache is written"
dataset-desk text-index  --rnd-data rnd-data --out desk-out/rnd-data --task code   # lineage reads the code index
dataset-desk lineage     --rnd-data rnd-data --out desk-out/rnd-data
```

- **`text-index`** embeds each file in chunks counted with the model's own tokenizer, into `<out>/<backend>/text/`. A
  re-run reuses vectors by sha256.
- **`text-dupes`** prints exact and near-duplicate files as JSON. Text defaults to 0.98 on bge-m3; `--task code` keeps
  scripts only and defaults to 0.95 on EmbeddingGemma 2. Any other model has no calibrated threshold, so it needs
  `--semantic`.
- **`text-search`** prints the nearest files to `--text`. `--task code` uses EmbeddingGemma 2's code-retrieval prompt.
- **`lineage`** lists versions of the same script across runs, ranked by similarity.
- **With `--rnd-data`,** only rnd-data's public copies are read. The private companion and the blob stores are refused
  (`PERM_PRIVATE_SOURCE`), and output carries relative paths only, never file contents.

## Choosing a model

Each command has a default model (see [Architecture](../architecture/#backends)); `--backend` picks another, and
`--task` picks a task's default. EmbeddingGemma 2 runs the way its model card says: a task prefix for each command,
text-only loading for text, `--dim 768|512|256` and `--vision-tokens 70|140|280|560|1120`. Each setting gets its own
cache.

## Using a GPU

```bash
dataset-desk index --project my-project --backend dinov2 --out desk-out/my-project --device cuda --grant my-run-1
```

- A GPU needs `--device` and a `--grant` id, so a shared card isn't taken by accident.
- **The VRAM governor:** before a model loads, it checks that memory already in use plus the model's measured peak fits
  under `--vram-budget` (default 26,624 MiB).
  - If not, it refuses with `PERM_GPU_BUSY`, or polls every 10 s for up to `--wait` seconds.
  - It never evicts another process.
  - Set `--vram-budget` to your card: on a 12 GB card, try 11000.

## Large projects

Duplicates, outliers and the look channel compare every pair, in blocks of about 64 MiB, so memory stays bounded. On
synthetic sets on a desktop CPU, the near-duplicate search took about 4.5 minutes at 100,000 images, and the look channel
about 3 minutes. The description gate compares descriptions that share a word; on very large sets where a few words
appear in most descriptions, it slows sharply.
