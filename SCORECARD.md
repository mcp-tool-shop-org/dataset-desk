# Scorecard

**Repo:** dataset-desk
**Date:** 2026-10-15
**Type tags:** `[all]` `[cli]`

## Pre-Remediation Assessment

Scored on the research prototype as it left the rnd library (a folder of scripts, run with `python desk.py`).

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 6/10 | Read-only on the project, `--out` guard, GPU grant guard; no SECURITY.md or threat model |
| B. Error Handling | 3/10 | Refusals were bare `SystemExit` strings; unexpected errors printed stack traces |
| C. Operator Docs | 5/10 | A design-heavy README; no CHANGELOG; `--help` missing several flags |
| D. Shipping Hygiene | 3/10 | Tests ran offline, but no package, no CI, no coverage measure, no dependency scan |
| E. Identity (soft) | 0/10 | No logo, translations, landing page or metadata |
| **Overall** | **17/50** | |

## Key Gaps

1. No package or console script: it ran only from its own folder with `sys.path` edits.
2. No structured errors or exit codes.
3. 78% coverage without the model libraries, and two tests that needed torch.
4. No CI, Codecov, dependency scan or Atlas map.
5. No public-facing docs.

## Remediation Priority

| Priority | Item | Estimated effort |
|----------|------|-----------------|
| 1 | Package layout, `dataset-desk` script, structured errors | 1 h |
| 2 | Offline fakes for the model libraries; coverage to 90%+ | 2 h |
| 3 | CI (tests, coverage floor, build, pip-audit, Atlas), README, SECURITY, CHANGELOG | 1 h |

## Post-Remediation

| Category | Before | After |
|----------|--------|-------|
| A. Security | 6/10 | 10/10 |
| B. Error Handling | 3/10 | 10/10 |
| C. Operator Docs | 5/10 | 10/10 |
| D. Shipping Hygiene | 3/10 | 10/10 |
| E. Identity (soft) | 0/10 | 10/10 |
| **Overall** | 17/50 | 50/50 |
