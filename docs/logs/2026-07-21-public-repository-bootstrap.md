# Repository quality pipeline and public bootstrap

Status: complete
Date/time: 2026-07-21T20:33:23-04:00
Code revision: parent 8875d313cff4f6d660621375c5ba22c206392cc8; dirty
corrective state validated before the authorized commit
Run/config identifiers: local quality validation; GitHub Actions runs
29878570532 and 29879593300
Primary artifacts: `.pre-commit-config.yaml`, `pyproject.toml`,
`.github/workflows/quality.yml`, `LICENSE`, `AGENTS.md`, and `README.md`
Remote: https://github.com/nquaz/evolving-world-models

## Overview

This record consolidates the previously separate code-quality and public-
repository bootstrap logs. The milestone established a pinned local quality
pipeline, repaired its initial lint and typing baseline, published a reviewed
Apache-2.0-licensed snapshot, and added independent continuous integration for
Python 3.9 and 3.12.

The public baseline contains the tabular and factored transition-model core,
optional visualization adapters, a worked notebook, tests, documentation, and
reproducible contributor tooling. A same-day addendum records a README math-
rendering regression that passed source-level checks and CI but failed in
GitHub's client-side renderer.

## Research question, hypothesis, or acceptance criteria

This was an engineering milestone, not a scientific experiment. The question
was whether the project could become a clean public baseline while:

- retaining the declared Python 3.9 minimum;
- enforcing deterministic formatting, high-signal linting, strict production
  typing, all behavioral tests, and compilation checks;
- keeping fast pre-commit checks separate from complete pre-push checks;
- publishing no credential, cache, environment, or unintended large artifact;
- recording an explicit Apache License 2.0 in source and wheel metadata;
- executing the notebook from a fresh kernel and validating package builds;
- running the same non-mutating gates independently on GitHub Actions; and
- preserving an auditable publication and failure history.

Success required clean development installation on Python 3.9, 33 passing tests
on Python 3.9 and 3.12, passing hook stages, a clean publication audit, correct
wheel licensing, a public remote with truthful initial history, and a passing
two-version CI matrix.

## Setup

- Platform: Linux 5.14.0-503.40.1.el9_5, x86-64
- Local environment: Conda `world_models`, CPython 3.12.13
- Minimum-version probe: CPython 3.9.21 in a clean temporary environment
- Git: 2.43.5
- GitHub CLI: 2.96.0
- Ruff: 0.15.22
- mypy: 1.19.1
- pre-commit: 4.3.0
- Hook revisions: `pre-commit-hooks` v6.0.0, `ruff-pre-commit` v0.15.22,
  and `mirrors-mypy` v1.19.1
- CI: Python 3.9 and 3.12 on `ubuntu-24.04`
- Default branch and visibility: `main`, public
- License: Apache-2.0
- Random seeds, data, and accelerators: not applicable
- Initial quality-pipeline work: approximately 25 minutes

GitHub CLI used HTTPS authentication. With no system credential store, its
token was kept outside the repository in a mode-0600 file. It was never printed,
placed in a command argument, logged, or tracked.

## Replication instructions

From a fresh checkout, create the documented environment and run the complete
local verification sequence:

```bash
git clone https://github.com/nquaz/evolving-world-models.git
cd evolving-world-models
conda create --name world_models python=3.12
conda activate world_models
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -B -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Repeat the minimum-version gates with an available Python 3.9 interpreter:

```bash
minimum_env="$(mktemp -d)"
python3.9 -m venv "$minimum_env"
"$minimum_env/bin/python" -m pip install -e '.[dev]'
"$minimum_env/bin/python" -m ruff check scripts tests
"$minimum_env/bin/python" -m mypy scripts
"$minimum_env/bin/python" -B -m unittest discover -s tests -v
```

Execute the notebook from a fresh kernel and build a wheel into unique temporary
locations:

```bash
notebook_run_dir="$(mktemp -d)"
wheel_dir="$(mktemp -d)"
python -m jupyter nbconvert --to notebook --execute \
  --output-dir="$notebook_run_dir" \
  --ExecutePreprocessor.timeout=120 \
  notebooks/Factored_MDP_Demo.ipynb
python -m pip wheel --no-deps --wheel-dir "$wheel_dir" .
python -m pip check
```

Expected results are clean Ruff and mypy output, `Ran 33 tests ... OK`, passing
hooks, a notebook with nine sequentially executed code cells and no errors, and
a wheel containing Apache-2.0 license metadata and the license file.

The one-time remote bootstrap used `gh repo create` for the confirmed public
repository, added `origin`, pushed `main`, and inspected the resulting workflow
with `gh run list`. Do not repeat repository creation for an existing remote.

## Methods

### Quality pipeline and baseline repairs

Ruff supplies both linting and formatting, avoiding overlapping Black, isort,
and Flake8 configurations. Rules `E4`, `E7`, `E9`, `F`, `I`, and `B` target
syntax, import ordering, and common correctness defects without forcing a broad
style migration. Ruff targets Python 3.9 with an 88-character line length.

Mypy runs in strict mode over the three production modules. The first pass found
13 issues; repairs added precise generic and collection types, separated reused
runtime type variables, and made already-validated finite domains explicit to
the type checker. No unsafe casts or transition-semantic changes were added.

The first Ruff pass found eight issues: four unsorted import blocks, one unused
import, and three assigned lambdas in tests. Imports were organized, the unused
import was removed, the lambdas became equivalent nested helpers, and Python
files were formatted.

Pre-commit checks whitespace, line endings, TOML/YAML/JSON, conflicts, symlinks,
large additions, debug statements, private keys, Ruff, formatting, and mypy.
Pre-push adds all unit tests and Python compilation in the caller's development
environment.

### Repository and hook validation

The checkout initially had an empty, unusable `.git` directory. Before the user
confirmed this was a new repository, real hooks were validated in an isolated
temporary Git repository instead of fabricating live history. After that
confirmation, Git was initialized on `main`, both hook stages were installed,
and the candidate snapshot was staged and reviewed explicitly.

### Publication, notebook, and packaging audit

Git-aware and no-ignore scans covered filenames, sizes, symlinks, executable
bits, private-key headers, common token patterns, caches, environments, and
generated outputs. No likely credential, special file, or unignored file over
1 MiB was found. This targeted audit was not an entropy-based secret scan.

Ignore rules were expanded narrowly for local environments, coverage and test
outputs, editor files, and operating-system artifacts. Missing drawing
documentation was reconstructed from implementation and tests. The notebook
stopped printing an absolute checkout path, gained a standalone figure caption,
and was re-executed into nine sequential code cells with zero errors and one
intentional inline figure.

`LICENSE` matched GitHub's canonical Apache-2.0 template byte-for-byte.
`pyproject.toml` records the SPDX expression and includes the license through
PEP 639 metadata. The isolated wheel used Core Metadata 2.4 and contained both
`License-Expression: Apache-2.0` and the license file.

### Commit and CI construction

The published history at the time of this consolidation is:

- `73092a3` — Initialize evolving world-model foundation
- `6cf6a27` — Add GitHub Actions quality checks
- `a8156da` — Document public repository bootstrap
- `8875d31` — Fix README math rendering

The Quality workflow runs on pushes to `main`, pull requests, and manual
dispatch. It grants read-only contents permission, disables persisted checkout
credentials, cancels superseded runs, and limits matrix jobs to 20 minutes.
Third-party actions are pinned to immutable revisions:

- `actions/checkout` v6.0.2:
  `de0fac2e4500dabe0009e67214ff5f5447ce83dd`
- `actions/setup-python` v6.2.0:
  `a309ff8b426b58ec0e2a45f0f869d46889d02405`

Each matrix job validates configuration and notebook JSON, checks Ruff and
mypy, runs tests and compilation, executes the notebook from a fresh kernel,
checks the installed environment, and builds a wheel. Checksum-verified
`actionlint` 1.7.12 validated the workflow before publication.

### Public-disclosure authorization

An initial public-repository request was rejected by an external-data safeguard.
Work stopped while the user was shown the source, notebook output,
documentation, log, commit-metadata, and retained historical-path disclosure
scope. The public repository was created only after explicit confirmation.

## Validation and quality checks

Validation covered clean Python 3.9 installation, direct Ruff/format/mypy/test
and compilation commands, both hook stages, notebook JSON and fresh execution,
wheel contents and metadata, actionlint, staged-content and credential scans,
remote identity, local/remote revision equality, and both GitHub Actions matrix
jobs. Formatting and typing changes were compared with temporary pre-change
copies before publication.

## Results

| Check | Scope | Result |
| --- | --- | --- |
| Clean `.[dev]` installation | Python 3.9.21 | Passed |
| Ruff lint and format | Python 3.9 and 3.12 | Passed |
| Strict mypy | 3 production modules | Passed |
| Unit tests | 33 tests, Python 3.9 | Passed in 0.678 s |
| Unit tests | 33 tests, Python 3.12 | Passed in 0.522 s |
| Hook stages | Temporary and live repositories | Passed |
| Compilation and notebook JSON | Python 3.9/3.12 as applicable | Passed |
| Fresh notebook execution | 9 code cells | Passed; 0 errors, 1 figure |
| Isolated wheel | Python 3.12 | Passed; Apache metadata present |
| Publication audit | Initial snapshot | No likely secret or unintended large file found |
| GitHub Actions 29878570532 | Python 3.9 / 3.12 | Passed in 51 s / 35 s |
| GitHub Actions 29879593300 | Python 3.9 / 3.12 | Passed in 47 s / 39 s |

Run 29878570532 validated revision `6cf6a27`. Run 29879593300 validated
`8875d31`; it passed every source, test, notebook, and packaging gate but did
not detect the client-side README macro failure described below. These are
deterministic software checks, so sampling uncertainty is not applicable.

## Figures and tables

No new research figure was produced. The notebook retained one explanatory
network figure after fresh execution. The table above reports deterministic
checks and observed runtimes; uncertainty intervals and metric direction are
not applicable.

## Conclusions

The project became a public, licensed baseline that can be cloned, installed,
tested, executed, and packaged in clean Python 3.9 and 3.12 environments. Local
hooks provide rapid feedback, and CI independently enforces the source,
behavioral, notebook, and packaging contracts.

The README regression demonstrates a narrower conclusion: valid Markdown
delimiters and passing source-level CI do not prove that every LaTeX command is
accepted by GitHub's downstream client renderer. Live rendering needs either a
compatible-command policy or dedicated browser-level validation.

## Limitations and threats to validity

- The credential audit was targeted rather than historical or entropy-based.
- Compatible dependency ranges are not a complete lock, so future resolutions
  may differ.
- CI covers Python 3.9 and 3.12, not every supported interpreter.
- The named hosted-runner image evolves over time.
- Strict mypy covers production modules, not tests and notebooks.
- CI validates Markdown source but does not execute GitHub's math renderer.
- Branch protection, required checks, dependency automation, `SECURITY.md`,
  release automation, and artifact attestations remain unconfigured.
- Two successful CI runs establish bootstrap viability, not long-term
  reliability or scientific validity.

## Deviations, failures, and negative results

- Mypy 1.20.2 and pre-commit 4.6.0 did not install on Python 3.9; the final
  compatible pins are mypy 1.19.1 and pre-commit 4.3.0.
- Initial hook validation encountered a read-only cache, a wrong temporary-
  repository working directory, and sandboxed dependency downloads. A writable
  `PRE_COMMIT_HOME`, the correct working directory, and approved network access
  resolved those failures.
- A Python 3.9 run emitted a non-fatal duplicate NetworkX backend warning; all
  visualization tests passed.
- The initial empty `.git` directory prevented live revision checks and hooks.
  No history was fabricated before the user confirmed a new repository.
- Notebook execution initially encountered a read-only Jupyter configuration
  path; a temporary configuration directory resolved it. An `Agg` backend run
  suppressed the intended inline figure, so the cell was made explicitly inline
  and re-executed.
- Public repository creation paused at the disclosure safeguard until the user
  explicitly confirmed publication.
- Commit `8875d31` replaced unsupported `\(...\)` and `\[...\]` delimiters but
  retained `\operatorname`, which GitHub's live renderer rejects. Local source
  checks, the raw Markdown API, pre-push hooks, and CI did not expose that
  downstream restriction.
- The assistant inferred commit-and-push authorization for `8875d31` from an
  earlier general workflow discussion. That inference was too broad. The local
  `AGENTS.md` correction now requires fresh, action-specific, one-shot commands.

## Addendum: README renderer correction

Date/time: 2026-07-21T20:22:02-04:00

GitHub displayed "the following macros is not allowed: operatorname" for each
README expression containing `\operatorname`.
[GitHub markup issue 1688](https://github.com/github/markup/issues/1688)
records the same downstream renderer limitation. The local correction replaces
`\operatorname{pa}` with base MathJax `\mathrm{pa}` in `README.md` and
`AGENTS.md`; the older July 20 log also receives GitHub-compatible delimiters
and notation.

This correction also consolidates the two July 21 logs and strengthens the Git
authorization policy. The changes were initially left uncommitted and unpushed
because the editing request did not authorize either action. At
2026-07-21T20:33:23-04:00, the user issued a fresh, explicit command to commit
and push this corrective change; this log is included in that authorized commit.

## Next directions

1. Confirm the live README renders without the macro error after the explicitly
   authorized push.
2. Add a small documentation check that rejects known unsupported GitHub math
   macros, while recognizing that it cannot replace the downstream renderer.
3. Decide on branch protection and required status checks.
4. Add reviewed dependency automation, `SECURITY.md`, and contribution guidance.
5. Evaluate lock or constraints files for long-term experiment replication.
6. Continue the scientific roadmap with rewards, Bayesian transition beliefs,
   objectives, downstream-task distributions, preregistered experiments, saved
   result tables, and publication-quality figures.

## Change and artifact summary

The milestone added or materially changed the quality configuration, packaging
metadata, ignore rules, production typing, test formatting, licensing, CI,
README, notebook, drawing documentation, and repository policy. Temporary
virtual environments, hook caches, notebook executions, wheels, and validation
repositories remained outside the project.

This consolidation keeps
`docs/logs/2026-07-21-public-repository-bootstrap.md` as the canonical July 21
record and removes the redundant
`docs/logs/2026-07-21-code-quality-pipeline.md`. The corrective commit also
touches `AGENTS.md`, `README.md`, and `docs/logs/2026-07-20.md`.

## Verification

The historical milestone passed these direct checks:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -B -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Both named CI runs passed their complete Python 3.9/3.12 matrices. The addendum
corrections also passed:

```bash
git diff --check
rg -n -F '\operatorname{' README.md AGENTS.md docs/logs/2026-07-20.md
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

The scoped `rg` command returned no matches, so none of the affected rendered
math invokes the unsupported macro. Both hook stages passed every applicable
check. The verification commands themselves made no commit or remote mutation;
those subsequent actions received the separate one-shot authorization recorded
above.
