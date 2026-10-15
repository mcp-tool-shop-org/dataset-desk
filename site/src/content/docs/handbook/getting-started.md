---
title: Getting started
description: Install dataset-desk, lay out a project, and run the lite pass.
sidebar:
  order: 1
---

## Install

dataset-desk needs Python 3.11 or 3.12. It runs on Windows and Linux.

The **lite tier** needs only numpy and Pillow:

```bash
pip install "dataset-desk @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
dataset-desk --version
```

For the embedding index, add the `models` extra (torch, transformers and sentence-transformers):

```bash
pip install "dataset-desk[models] @ git+https://github.com/mcp-tool-shop-org/dataset-desk@v0.1.2"
```

## Lay out a project

dataset-desk reads [style-dataset-lab](https://github.com/mcp-tool-shop-org/style-dataset-lab)'s project layout:

```text
my-project/
  records/
    img-0001.json            one record per image (optional)
  outputs/
    wave-01/
      img-0001.png
      generation.json        optional: what generated this wave
      _curation.json         optional: review verdicts for this wave
    wave-02/
      ...
```

- Images must sit in a folder **under** `outputs/`. Files at the top of `outputs/` are skipped: they are usually
  contact sheets.
- Records are optional. A record names its image with `asset_path` and can carry:

```json
{
  "id": "img-0001",
  "asset_path": "outputs/wave-01/img-0001.png",
  "judgment": { "status": "approved", "explanation": "A squat bog creature with moss on its back, half in water." },
  "canon": { "subject": "bog creature" },
  "provenance": { "prompt": "the prompt that generated it" }
}
```

`judgment.status` is `approved`, `borderline` or `rejected`. Only approved records are audited as descriptions, and a
rejection note is never used as a description.

## Run the lite pass

```bash
dataset-desk index  --project my-project --backend none --out desk-out/my-project
dataset-desk report --project my-project --backend none --out desk-out/my-project
```

`index` scans the project: it hashes every image, computes its dHash and its 30-number look, and joins the records and
wave metadata. `report` writes `desk-out/my-project/report-none.md` and prints its path.

Open the report. Every line in it is a prompt to look, not a verdict.

## Add an embedding index

```bash
dataset-desk index  --project my-project --backend siglip2 --out desk-out/my-project
dataset-desk report --project my-project --backend siglip2 --out desk-out/my-project
```

The first run with a model backend fetches its weights from Hugging Face at a pinned revision. To forbid downloads, set
`HF_HUB_OFFLINE=1`; then a missing model is an error, not a download.

The scan cache (`scan.jsonl`) is shared between backends, so the second backend only embeds.

## Run it politely

- It uses 4 CPU threads unless you pass `--threads`. On a shared machine, keep to half the cores at most.
- A model run drops the process to below-normal priority.
- A GPU needs `--device cuda` **and** `--grant <id>`, so a shared card is never taken by accident.
