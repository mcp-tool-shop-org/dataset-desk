---
title: dataset-desk Handbook
description: What dataset-desk is for, what it checks, and how this handbook is laid out.
sidebar:
  order: 0
---

dataset-desk is a review desk for visual datasets: the folders of images, and their descriptions, that train a style
LoRA or test an image search. It reads a project, writes a report and a review queue, and leaves every decision to a
person. **It never removes or edits anything in the project.**

## The two defects it looks for

**Repeated pictures.** A training set that holds the same image twice, or two renders a few pixels apart, teaches that
image twice and something else less. dataset-desk finds three kinds of repeat:

| Kind | How it's found | What it usually is |
|---|---|---|
| Byte-identical | sha256 of the file | the same render saved into two folders |
| Near | 256-bit dHash, within 10 bits | a re-save, a small crop, a seed neighbour |
| Semantic | embedding cosine | the same subject and pose, rendered differently |

**Descriptions that aren't descriptions.** When an image's description is its generation prompt, or one sentence pasted
onto dozens of images, it says nothing about that picture. A model trained on it learns the prompt; a search benchmark
scored on it measures recall of the prompt. The [description gate](./description-gate/) catches these without any
model.

## What you get back

- `report-<backend>.md`: duplicates, the least typical approved images (by content and by look), the description audit,
  and how well each description finds its own image.
- `description-review.jsonl`: one line per approved record, with the gate's verdict and reasons.
- `description-worksheet.md`: the flagged images, each beside its nearest siblings, ready to be re-described.

## How this handbook is laid out

- [Getting started](./getting-started/): install it and run the lite pass.
- [Usage](./usage/): each command, and how to read the report.
- [The description gate](./description-gate/): every test, with its exact arithmetic.
- [Reference](./reference/): flags, outputs, error codes and exit codes.
- [Architecture](./architecture/): the modules, the caches and the backends.
- [Security](./security/): what it reads, writes and never touches.

## Status

v0.1.3 is a public prototype. It is tested offline (152 tests, 96% line coverage). A first calibration on the studio's own data
ranked the models by task, but its thresholds aren't pinned yet: they are still starting points. The design record and its review history are in the studio's research library:
[rnd / experiments / dataset-desk](https://github.com/mcp-tool-shop-org/rnd/tree/main/experiments/dataset-desk).
