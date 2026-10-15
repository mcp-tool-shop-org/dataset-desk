---
title: Security
description: What dataset-desk reads, writes and never touches, and how to report a problem.
sidebar:
  order: 6
---

## Threat model

| | |
|---|---|
| **Reads** | the folder named by `--project`: images, `records/*.json`, wave metadata, PNG text chunks |
| **Writes** | only under `--out`; an `--out` inside the project is refused |
| **Deletes** | nothing, anywhere |
| **Network** | none in the lite tier; a model backend fetches pinned weights from Hugging Face on first use unless `HF_HUB_OFFLINE=1`; the drafter is always offline |
| **Code execution** | no model loads with remote code; records and PNG metadata are parsed as data |
| **Credentials** | none read, stored or sent |
| **Telemetry** | none |
| **Permissions** | an ordinary user account; a GPU needs `--grant` |

## Untrusted input

A project may come from someone else. dataset-desk treats it as data:

- JSON records that fail to parse are listed under the report's problems and skipped.
- An unreadable image is logged and skipped, or stops the run with `--strict`.
- PNG text chunks are read as strings and parsed as JSON only; nothing in them is executed.

## Reporting a vulnerability

Email **64996768+mcp-tool-shop@users.noreply.github.com**, or use GitHub's private vulnerability reporting on the
repository. Acknowledgement within 48 hours, an assessment within 7 days, a fix within 30 days. See
[SECURITY.md](https://github.com/mcp-tool-shop-org/dataset-desk/blob/main/SECURITY.md).
