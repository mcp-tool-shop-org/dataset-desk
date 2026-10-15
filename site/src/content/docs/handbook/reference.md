---
title: Reference
description: Commands, flags, outputs, error codes and exit codes.
sidebar:
  order: 4
---

## Commands

```text
dataset-desk {index,dupes,search,coverage,report} --project PROJECT [options]
dataset-desk --version
python -m dataset_desk ...        same as dataset-desk
```

| Command | Does |
|---|---|
| `index` | scan the project and embed its images (and descriptions) into the cache under `--out` |
| `dupes` | exact, near and semantic duplicates, as JSON |
| `search` | the nearest images to `--text`, `--image`, or both |
| `coverage` | the best images for each subject in `--subjects` |
| `report` | the Markdown report, the review queue and the worksheet |

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--project PATH` | required | the project folder; read only |
| `--out PATH` | `desk-out/<project>` | where caches and reports go; must not be inside the project |
| `--backend` | `eg2` | `eg2`, `siglip2`, `none` (lite) or `fake` (tests) |
| `--threads N` | 4 | CPU threads; half the cores at most on a shared machine |
| `--device` | `cpu` | `cpu`, or a torch device such as `cuda` |
| `--grant ID` | | required for any device but `cpu` |
| `--limit N` | | `index`: only the first N unique images |
| `--strict` | off | stop on the first unreadable image |
| `--count-tokens` | off | `report`: run the gate's 64-token test with the cached SigLIP2 tokenizer, offline |
| `--images-only` | off | embed images only; skip descriptions |
| `--allow-multimodal-pool` | off | allow description embeddings and model-based checks on locked benchmark projects |
| `--draft` | `none` | `report`: draft flagged descriptions with `plain-sight` (alias `florence2`) or `fake` |
| `--text TEXT` | | `search`: a description |
| `--image PATH` | | `search`: a picture |
| `--subjects FILE` | | `coverage`, `report`: one wanted subject per line |
| `-k N` | 10 | `search`: how many results |
| `--debug` | off | show the stack trace for an unexpected error |
| `--version` | | print the version |

## Outputs under `--out`

| File | Written by | Holds |
|---|---|---|
| `scan.jsonl` | `index`, `report` | per unique image: sha256, paths, dHash, the 30-number look, record fields; shared by every backend |
| `<backend>/meta.json` | `index` | the backend, its revision, dimensions, timings, and the grant id if any |
| `<backend>/items.jsonl` | `index` | the indexed items, in order |
| `<backend>/images.npy`, `texts.npy` | `index` | unit-norm float32 embeddings |
| `report-<backend>.md` | `report` | the report |
| `description-review.jsonl` | `report` | the gate's verdict for every approved record |
| `description-worksheet.md` | `report` | the blind re-description worksheet |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | done |
| 1 | a user error or a refusal: bad arguments, a missing cache, a guarded step |
| 2 | an unexpected error |
| 130 | interrupted |

## Error codes

Errors print to stderr as:

```text
error STATE_NO_INDEX: no siglip2 index under desk-out/my-project/siglip2; run: dataset-desk index --backend siglip2
hint: run `dataset-desk index` with the same --project, --backend and --out first
```

| Code | When | Exit |
|---|---|---|
| `INPUT_BAD_ARGUMENT` | a command is missing what it needs (`search` without `--text` or `--image`, `coverage` without `--subjects`) | 1 |
| `INPUT_UNKNOWN_BACKEND` | an unknown `--backend` | 1 |
| `INPUT_UNKNOWN_DRAFTER` | an unknown `--draft` | 1 |
| `PERM_OUT_INSIDE_PROJECT` | `--out` is inside `--project` | 1 |
| `PERM_GPU_NEEDS_GRANT` | a device other than `cpu` without `--grant` | 1 |
| `PERM_LOCKED_POOL` | description embeddings or a model-based check on a locked benchmark project | 1 |
| `DEP_TOKENIZER_MISSING` | `--count-tokens` without the SigLIP2 tokenizer in the local cache | 1 |
| `STATE_NO_INDEX` | a command needs a cache that `index` hasn't written | 1 |
| `RUNTIME_UNEXPECTED` | anything else; rerun with `--debug` | 2 |

Codes are stable from 0.1.0 on.

## Python API

The modules can be imported directly. The description gate needs only the standard library:

```python
from dataset_desk import description_gate

audit = description_gate.audit_project("my-project")
print(audit["passed_gate"], audit["checked"])
print(description_gate.STOPWORDS_SHA256)
```
