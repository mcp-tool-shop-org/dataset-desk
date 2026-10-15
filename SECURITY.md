# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

dataset-desk is a prototype (0.x). Fixes land on the newest 0.1 release.

## Reporting a Vulnerability

Email: **64996768+mcp-tool-shop@users.noreply.github.com**

Or use GitHub's private vulnerability reporting on this repository (Security → Report a vulnerability).

Include:
- a description of the vulnerability;
- steps to reproduce;
- the version affected;
- the potential impact.

### Response timeline

| Action | Target |
|--------|--------|
| Acknowledge report | 48 hours |
| Assess severity | 7 days |
| Release fix | 30 days |

## Scope

dataset-desk runs **locally only**.
- **Data touched:** the project folder named by `--project` (read only) and the output folder named by `--out` (written). It refuses an `--out` inside the project, and it deletes nothing.
- **Network:** the lite tier (`--backend none`) never uses it. A model backend fetches its pinned weights from Hugging Face on first use unless `HF_HUB_OFFLINE=1` is set. The plain-sight drafter is always offline.
- **Code execution:** no model is loaded with remote code. PNG metadata and record files are parsed as data, never executed.
- **No secrets handling:** it does not read, store, or transmit credentials.
- **No telemetry** is collected or sent.

In scope: path handling that could write outside `--out`, parsing of untrusted images, PNG text chunks or JSON records, and anything that would make a model run remote code or download when told not to.
