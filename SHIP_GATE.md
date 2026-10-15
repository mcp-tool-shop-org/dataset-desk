# Ship Gate

> No repo is "done" until every applicable line is checked.
> Copy this into your repo root. Check items off per-release.

**Tags:** `[all]` `[cli]`. Released on GitHub only (not on PyPI); the `[pypi]` lines that still apply (a Python
version floor, a clean wheel and sdist) are checked because CI builds and checks both.

---

## A. Security Baseline

- [x] `[all]` SECURITY.md exists (report email, supported versions, response timeline) — executed by `npx @mcptoolshop/shipcheck security-docs` (2026-10-15)
- [x] `[all]` README includes threat model paragraph (data touched, data NOT touched, permissions required) — executed by `npx @mcptoolshop/shipcheck security-docs` (2026-10-15)
- [x] `[all]` No secrets, tokens, or credentials in source or diagnostics output — `shipcheck secrets` has no npm package to scan here; the org identity scan reports RESULT CLEAN on the tracked tree (2026-10-15)
- [x] `[all]` No telemetry by default — stated in README and SECURITY.md (2026-10-15)

### Default safety posture

- [x] `[cli|mcp|desktop]` Dangerous actions (kill, delete, restart) require explicit `--allow-*` flag — the desk deletes nothing; the guarded steps are a GPU (`--grant`) and locked benchmark projects (`--allow-multimodal-pool`) (2026-10-15)
- [x] `[cli|mcp|desktop]` File operations constrained to known directories — reads `--project`, writes only `--out`, refuses an `--out` inside the project (2026-10-15)
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server

## B. Error Handling

- [x] `[all]` Errors follow the Structured Error Shape: `code`, `message`, `hint`, `cause?`, `retryable?` — `DeskError` in `dataset_desk/errors.py` (2026-10-15)
- [x] `[cli]` Exit codes: 0 ok · 1 user error · 2 runtime error · 3 partial success — 0, 1 and 2 are used; no command reports partial success (2026-10-15)
- [x] `[cli]` No raw stack traces without `--debug` — `desk.run` turns any failure into `RUNTIME_UNEXPECTED` (2026-10-15)
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[desktop]` SKIP: not a desktop app
- [ ] `[vscode]` SKIP: not a VS Code extension

## C. Operator Docs

- [x] `[all]` README is current: what it does, install, usage, supported platforms + runtime versions (2026-10-15)
- [x] `[all]` CHANGELOG.md (Keep a Changelog format) (2026-10-15)
- [x] `[all]` LICENSE file present and repo states support status — MIT; README "Status" and SECURITY.md state the 0.1.x support (2026-10-15)
- [x] `[cli]` `--help` output accurate for all commands and flags (2026-10-15)
- [ ] `[cli|mcp|desktop]` SKIP: no log stream — the CLI prints only results (stdout) and structured errors (stderr), with `--debug` for stack traces; it handles no secrets to redact
- [ ] `[mcp]` SKIP: not an MCP server
- [ ] `[complex]` SKIP: no daemons, state files or operational modes; the handbook on the landing page covers operation

## D. Shipping Hygiene

- [x] `[all]` `verify` script exists (test + build + smoke in one command) — `verify.sh`: lint, tests, lite-mode smoke; CI adds build + twine check + install smoke (2026-10-15)
- [x] `[all]` Version in manifest matches git tag — executed by `npx @mcptoolshop/shipcheck manifest`; 0.1.0 in `pyproject.toml` and `dataset_desk.__version__`, tag `v0.1.0` (2026-10-15)
- [x] `[all]` Dependency scanning runs in CI (ecosystem-appropriate) — executed by `npx @mcptoolshop/shipcheck ci` (pip-audit) (2026-10-15)
- [x] `[all]` No known high/critical vulnerabilities in any dependency tree, and Dependabot alerts are enabled — `pip-audit --strict .` reports none; Dependabot alerts enabled on the repo (2026-10-15)
- [ ] `[all]` SKIP: no automated update bot, by the org rule (Dependabot PRs only on request); pip-audit in CI and Dependabot alerts cover the security outcome
- [ ] `[npm]` SKIP: not an npm package
- [ ] `[npm]` SKIP: not an npm package
- [x] `[npm]` `engines.node` set · `[pypi]` `python_requires` set — `requires-python = ">=3.11"`, executed by `shipcheck manifest` (2026-10-15)
- [x] `[npm]` Lockfile committed · `[pypi]` Clean wheel + sdist build — `uv.lock` committed; CI builds the wheel and sdist and runs `twine check --strict` (2026-10-15)
- [ ] `[vsix]` SKIP: not a VS Code extension
- [ ] `[desktop]` SKIP: not a desktop app

## E. Identity (soft gate — does not block ship)

- [x] `[all]` Logo in README header — brand/logos/dataset-desk/readme.png (2026-10-15)
- [x] `[all]` Translations (polyglot-mcp, 8 languages) — English plus 7, TranslateGemma 27B, local (2026-10-15)
- [x] `[org]` Landing page (@mcptoolshop/site-theme) — live with the Starlight handbook and Pagefind search (2026-10-15)
- [x] `[all]` GitHub repo metadata: description, homepage, topics (2026-10-15)

---

## Gate Rules

**Hard gate (A–D):** Must pass before any version is tagged or published.
If a section doesn't apply, mark `SKIP:` with justification — don't leave it unchecked.

**Soft gate (E):** Should be done. Product ships without it, but isn't "whole."

**Executed vs attested.** `shipcheck audit` only *counts these checkboxes* — it does not read your repo, so a box can be green while the fact is false. The lines that say **"executed by `npx @mcptoolshop/shipcheck <gate>`"** are backed by a command that reads the real artifact and exits 1 on the real defect.

**Checking off:**
```
- [x] `[all]` SECURITY.md exists (2026-02-27)
```

**Skipping:**
```
- [ ] `[pypi]` SKIP: not a Python project
```
