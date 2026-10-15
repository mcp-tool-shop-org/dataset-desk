---
title: Architecture
description: The modules, the caches, the backends, and the guards.
sidebar:
  order: 5
---

## Modules

| Module | Needs | Does |
|---|---|---|
| `dataset_desk/desk.py` | numpy, Pillow | the scan, the caches, the analyses, the report and the CLI |
| `dataset_desk/backends.py` | numpy; torch and transformers only when a model backend runs | thread caps, the embedding backends, the drafters |
| `dataset_desk/description_gate.py` | the standard library; a tokenizer only for the token count | the description gate, the review queue, the worksheet |
| `dataset_desk/errors.py` | the standard library | the structured error and the error codes |

## The flow

1. **Scan.** Walk `outputs/<folder>/`, hash every image (sha256), and merge byte-identical files into one item. For each
   unique image, compute a 256-bit dHash and a 30-number look vector, and join its record, its wave's
   `generation.json` and `_curation.json`, and the generation data in its PNG text chunks. The results go to
   `scan.jsonl`, keyed by sha256, so later runs and other backends reuse them.
2. **Embed** (model backends only). Images, and the approved descriptions, become unit-norm vectors in one space. Only
   items not already in the cache are embedded.
3. **Analyse.** Duplicates in three tiers, the Vendi score, outliers by content and by look, nearest and furthest
   unapproved images, coverage, and the description fit check.
4. **Gate.** The description gate runs on every approved record, with no model.
5. **Report.** Markdown, the review queue and the worksheet.

## Backends

| Backend | Model | Pinned revision | Notes |
|---|---|---|---|
| `eg2` | `google/embeddinggemma-2` | `914f7f8` | text, images, and both together; float32 or bfloat16, never float16 |
| `siglip2` | `google/siglip2-so400m-patch14-384` | `e8e4872` | text padded to 64 tokens, the way it was trained |
| `none` | | | no embeddings; the lite tier |
| `fake` | | | deterministic vectors for tests |

The drafter `plain-sight` runs `florence-community/Florence-2-large` at revision `4271c66`, through plain-sight's own
engine when it is installed, otherwise through the native transformers classes. It always loads offline, and never with
remote code.

## The look channel

Thirty bounded numbers per image, computed at native resolution: palette and tone statistics, plus surface and
brushwork texture measures. It is cheap, needs no model, and separates "the style is off" from "the subject is odd".

## Guards

- **Thread caps.** Importing `dataset_desk.backends` sets the BLAS and OpenMP thread variables before numpy loads, and a
  torch region holds BLAS to one thread with torch's own pool at `--threads`.
- **Read-only projects.** The CLI refuses an `--out` inside the project.
- **The GPU.** Any device but `cpu` needs `--grant`, and the grant id is written into the cache's `meta.json`.
- **Locked benchmark projects.** A project being used as a retrieval benchmark must not have its queries accepted or
  shaped by the models the benchmark measures. On the locked projects, description embeddings and model-based acceptance
  (`eg2_fit_accept`, `ai_eyes_verify`) are refused until `--allow-multimodal-pool` is passed after the counted run.
