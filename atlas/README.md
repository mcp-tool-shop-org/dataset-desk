# dataset-desk: how it works

Mapped at 2026-10-20 from commit 30802b9 by Atlas 1.24.0.

## What this is

dataset-desk: a review desk for visual datasets — duplicates, outliers, search, and a model-free description gate. (written by a person)

5 parts, mostly Python (13 files), CSS (2), TypeScript (2), Astro (1), JavaScript (1) and shell (1). Work enters through 4 doors; the busiest is CI, which reaches 3 parts. It publishes to PyPI. It deploys a site to GitHub Pages. People run dataset-desk.

## What changed since 2026-10-20 (25d9cdd)

Nothing structural changed since 2026-10-20; 1 file added and 20 changed content.

## What comes in

1. **CI.** On a pull request to main; on a push to main touching 7 paths; or by hand. Runs verify.sh and tests/; checks dataset_desk/.
2. **Deploy site to GitHub Pages.** On a push to main touching 2 paths; or by hand. Runs site/astro.config.mjs and site/src/.
3. **Release.** When a release is published; or by hand. Checks dataset_desk/.
4. **dataset-desk** (a command people run). Runs dataset_desk/desk.py.

## What happens through CI

1. The workflow runs verify.sh in the repository root and tests/ in tests; it checks dataset_desk/ in dataset_desk.
2. It uploads coverage to Codecov.

## Who reads the results

CI writes nothing this map can see.

## The other doors

**Deploy site to GitHub Pages** runs site/astro.config.mjs and site/src/, and deploys the site.

**Release** checks dataset_desk/ and publishes to PyPI and uploads dist/*.tar.gz and dist/*.whl to the release on a release event.

**dataset-desk** (a command people run) runs dataset_desk/desk.py.

## What breaks what

- **dataset_desk** is imported only from tests, by 1 part (tests), and sits on the path of 3 doors.

## What tends to change together

No two source files changed together often enough to name.

Window: 180 days; a pair counts from 3 shared commits, since the window holds fewer than 30 qualifying commits.

## What no test touches

Every code part is imported by at least one test.

## Written but never read

No place this map can see is written, so none goes unread.

## Helpers that look duplicated

No two parts export a helper that looks alike.

## Generated, never hand-edited

Nothing in this repository writes to a tracked place this map can see.

## Hand-authored

People write .github/, the repository root and site/. Nothing in this repository writes to them.

## Where to start

dataset_desk/desk.py → dataset_desk/errors.py

Read those in order to follow one run of dataset-desk end to end. This path follows dataset-desk (a command people run) from its entry, since CI runs only tests and checks.

## What this map cannot see

- 1 read uses a path built at run time and is not named here.
- 5 writes and 19 reads go to a path their caller passes, not to this repository.
- 7 writes and 4 reads go to the directory the command is run in (desk-out/), not to this repository.
- Statistics confidence is low: fewer than 30 qualifying commits in the window, and fewer than 25 source files reach 10 revisions.

Regenerate with `npx --yes @dogfood-lab/atlas map`.
