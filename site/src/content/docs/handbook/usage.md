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
dataset-desk dupes --project my-project --backend siglip2 --out desk-out/my-project > dupes.json
```

Prints JSON with `exact` (groups of byte-identical files), `near` (pairs and their dHash distance), `semantic` (pairs and
their cosine), the `thresholds` used, and `top1_percentiles`: each image's cosine to its nearest neighbour, which is the
scale to set a semantic cut against.

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
