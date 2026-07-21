# Git workflow and code-quality pipeline

Status: complete
Date/time: 2026-07-21T18:38:16-04:00
Code revision: unavailable; the checkout has no usable Git metadata
Run/config identifiers: Ruff 0.15.22, mypy 1.19.1, pre-commit 4.3.0
Primary artifacts: `.pre-commit-config.yaml`, `pyproject.toml`, `AGENTS.md`

## Overview

This milestone adds a practical, pinned quality pipeline and documents safe Git
and commit practices for a shared research worktree. The pipeline uses fast
file-hygiene checks, Ruff linting and formatting, strict mypy checking of the
production package, all 33 unit tests, and Python compilation checks.

Git 2.43.5 is installed, but the live checkout's `.git` directory is empty.
Consequently, status, diff, staging, commits, hook installation, and revision
verification are unavailable in the live checkout. No attempt was made to
initialize or reconstruct repository history. The actual hooks were instead
tested in an isolated temporary Git repository containing a copy of the project.

## Research question, hypothesis, or acceptance criteria

This is an engineering milestone rather than a scientific experiment. The
question was whether the repository could adopt a useful quality gate that:

- remains installable under the declared Python 3.9 minimum;
- provides deterministic formatting and conservative, high-signal linting;
- makes strict static typing a passing gate for production modules;
- runs the existing behavioral suite without changing transition semantics;
- separates fast commit checks from the complete pre-push test gate;
- is reproducible from declared and pinned development dependencies; and
- remains testable without mutating the live checkout's missing Git metadata.

Success required clean installation under Python 3.9, passing Ruff and strict
mypy checks, 33 passing tests under Python 3.9 and 3.12, valid hook
configuration, and successful execution of both hook stages in a temporary Git
repository.

## Setup

- Repository root:
  `/data/user_data/nquazi/repos/evolving-world-models`
- Operating system: Linux 5.14.0-503.40.1.el9_5, x86-64
- Git: 2.43.5, installed but unusable in the live checkout
- Minimum-version environment: CPython 3.9.21 in a clean temporary virtualenv
- Current project environment: CPython 3.12.13 in `world_models`
- Ruff: 0.15.22
- mypy: 1.19.1, selected as the newest tested line installable on Python 3.9
- pre-commit: 4.3.0, selected as the newest tested line installable on Python
  3.9
- Hook revisions: `pre-commit-hooks` v6.0.0, `ruff-pre-commit` v0.15.22,
  `mirrors-mypy` v1.19.1
- Random seeds: not applicable; no stochastic experiment was run
- Data: not applicable
- Material hardware: not applicable; all checks are CPU-light
- Actual elapsed work: approximately 25 minutes, including two failed
  minimum-version dependency probes and clean environment installation

The first hook installation requires network access to retrieve pinned hook
repositories and their isolated environments. Subsequent runs reuse the local
pre-commit cache.

## Replication instructions

From the repository root, install all development capabilities:

```bash
python -m pip install -e '.[dev]'
```

When the checkout contains valid Git metadata, install both configured hook
stages and exercise them across the repository:

```bash
pre-commit install --install-hooks
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Run the non-mutating gates directly when debugging a failure or when Git is not
available:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -B -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
```

To reproduce the minimum-version installation on a machine where `python3`
resolves to Python 3.9:

```bash
quality_env="$(mktemp -d /tmp/ewm-py39.XXXXXX)"
python3 -m venv "$quality_env"
"$quality_env/bin/python" -m pip install -e '.[dev]'
"$quality_env/bin/python" -m ruff check scripts tests
"$quality_env/bin/python" -m ruff format --check scripts tests
"$quality_env/bin/python" -m mypy scripts
"$quality_env/bin/python" -B -m unittest discover -s tests -v
```

Expected output is a clean Ruff result, `Success: no issues found in 3 source
files` from mypy, and `Ran 33 tests ... OK` from unittest. Hook execution should
show every applicable hook as `Passed`; the broken-symlink hook may report that
there are no matching files.

## Methods

### Tool selection and configuration

Ruff supplies both formatting and linting to avoid overlapping Black, isort,
and Flake8 configurations. The initial rule set is intentionally conservative:
`E4`, `E7`, `E9`, `F`, `I`, and `B`. These cover syntax and common correctness
errors, import ordering, and high-signal bug patterns without forcing a broad
style migration. Ruff targets Python 3.9, uses an 88-character line length, and
excludes notebooks because notebook execution and output state need separate
validation.

Mypy runs in strict mode over `scripts/`, the production package. Tests are not
part of the initial strict target because their dynamic fixtures and mocks have
different typing value. The mypy target version is Python 3.9.

The pre-commit stage includes:

- trailing whitespace, final newline, mixed line ending, and merge marker
  checks;
- TOML, YAML, and JSON/notebook structural checks;
- case conflict and symlink checks;
- a 1 MiB added-file limit, debug-statement detection, and private-key
  detection;
- Ruff safe fixes followed by Ruff formatting; and
- strict mypy checking of production modules.

The pre-push stage adds the complete unittest suite and compilation checks.
Those hooks use the caller's active development environment because the
visualization tests require optional development dependencies. CI is expected
to repeat the same checks independently using non-mutating Ruff commands.

### Baseline repairs

The first strict mypy pass reported 13 issues. Repairs added precise generic
types, separated two locally reused runtime type variables, typed graph layout
and edge collections, and cached already-validated finite tabular domains in
non-optional internal tuples. This last change makes the constructor's finite
domain invariant visible to the type checker without an unsafe cast.

The first Ruff pass reported eight issues: four unsorted import blocks, one
unused import, and three lambda assignments in tests. Imports were organized,
the unused import removed, the lambdas converted to small nested functions, and
the Python files formatted. No transition probabilities, public signatures, or
test expectations were changed.

### Isolated hook validation

Because the live `.git` metadata is unusable, the project files were copied to
a unique directory under `/tmp`, Git was initialized only in that temporary
copy, and all files were staged there. Both hook stages were then run under the
pinned configuration. This validated real pre-commit staging and hook behavior
without creating history or staging data in the live checkout.

## Validation and quality checks

- Parsed and validated `.pre-commit-config.yaml` with pre-commit 4.3.0.
- Parsed `pyproject.toml` during the clean editable build under Python 3.9 and
  validated its TOML structure in the current environment.
- Installed `.[dev]` in a clean Python 3.9 virtualenv from a temporary project
  copy.
- Imported `scripts` from outside that copied checkout after installation.
- Ran Ruff lint and format checks against `scripts` and `tests`.
- Ran strict mypy against all three production modules.
- Ran all 33 tests under Python 3.9 and Python 3.12.
- Ran both hook stages against all staged files in an isolated temporary Git
  repository, using both pre-commit 4.3.0/Python 3.9 and the current Python 3.12
  environment.
- Ran Python compilation and notebook JSON structure validation.
- Compared formatted Python files against temporary pre-format copies to review
  every mechanical and typing-related change.

## Results

| Gate | Environment | Result |
| --- | --- | --- |
| Clean `.[dev]` installation | Python 3.9.21 | Passed |
| Ruff lint | Python 3.9 and 3.12 | Passed |
| Ruff format check | Python 3.9 and 3.12 | Passed |
| Strict mypy, 3 modules | Python 3.9 and isolated hook | Passed |
| Unit tests, 33 tests | Python 3.9.21 | Passed in 0.678 s |
| Unit tests, 33 tests | Python 3.12.13 | Passed in 0.522 s |
| Pre-commit stage | Temporary Git repository | Passed |
| Pre-push stage | Temporary Git repository | Passed |
| Python compilation | Python 3.9 and 3.12 | Passed |
| Notebook JSON structure | Python 3.12 | Passed |

No sampling or statistical uncertainty applies to these deterministic software
checks. One Python 3.9 test run emitted a non-fatal NetworkX warning that the
`nx-loopback` backend was defined more than once in the isolated environment;
all visualization contracts still passed.

## Figures and tables

No figures were produced. The results table above reports deterministic
pass/fail checks and observed test runtimes; sampling uncertainty and metric
direction are not applicable.

## Conclusions

The repository now has an executable, pinned quality pipeline rather than a
documentation-only recommendation. It preserves Python 3.9 support, gives rapid
format/lint/type feedback before commit, runs the complete behavioral suite
before push, and provides direct non-mutating commands suitable for CI.

The core transition behavior remains intact: all 33 contract tests pass under
both the minimum and current Python environments. Strict typing is now a real
passing gate rather than an aspirational instruction.

## Limitations and threats to validity

- The live checkout is not a valid Git repository, so hooks could not be
  installed or executed in place and no commit could be created.
- No CI-provider workflow was added because the repository's hosting provider
  and remote configuration are unavailable. Local hooks can be bypassed, so CI
  should eventually become the authoritative enforcement layer.
- The project does not yet contain a dependency lockfile. Direct quality tools
  and hook revisions are pinned, while transitive dependencies remain resolved
  at installation time.
- Strict mypy currently covers production modules, not tests or notebooks.
- Notebook JSON structure was validated, but the notebook was not re-executed
  because it was not changed by this milestone.
- The private-key hook is a useful guard but is not a comprehensive secret
  scanner.

## Deviations, failures, and negative results

The initial plan used current mypy 1.20.2 and pre-commit 4.6.0. Clean Python 3.9
resolution showed that neither version installs on the declared minimum Python
version. The final pins were therefore changed to mypy 1.19.1 and pre-commit
4.3.0, and the full clean install and hook suite were rerun successfully.

The first pre-commit validation attempt tried to use a read-only home cache.
Subsequent validation set `PRE_COMMIT_HOME` to a temporary writable cache. The
first isolated hook invocation also started from the live working directory
rather than the temporary Git root; it failed before running hooks and was
rerun from the correct temporary directory. Finally, the first Python 3.9 hook
environment build lacked sandbox network access; it succeeded when rerun with
approved network access. No failed attempt modified production results.

## Next directions

1. Confirm whether the missing Git metadata is intentional and restore the
   canonical repository history through the appropriate external workflow.
2. Once Git is valid, run `pre-commit install --install-hooks` and both all-file
   hook commands in the live checkout.
3. Add provider-appropriate CI with Python 3.9 and a current Python version,
   using non-mutating Ruff checks, strict mypy, unit tests, compilation,
   notebook structure validation, and a packaging smoke test.
4. Decide whether to add a lockfile or constraints file for fully repeatable
   development environments.
5. Expand strict typing or Ruff rules only through focused changes with a clean
   baseline and documented rationale.

## Change and artifact summary

- `.pre-commit-config.yaml`: new pinned pre-commit and pre-push pipeline.
- `pyproject.toml`: pinned development tools and Ruff/mypy configuration.
- `.gitignore`: tool caches, coverage output, build artifacts, and local virtual
  environments.
- `AGENTS.md`: Git authorization, safety, atomic commit, staging, message,
  branch/remote, hook, and missing-metadata practices.
- `README.md`: setup, hook execution, and direct quality-check commands.
- `scripts/mdp.py`: strict typing annotations and explicit cached finite domains,
  plus Ruff formatting.
- `scripts/visualization.py`: strict collection annotations and Ruff formatting.
- `tests/test_mdp.py`: Ruff formatting only.
- `tests/test_visualization.py`: Ruff formatting and equivalent nested helper
  functions replacing assigned lambdas.
- `docs/logs/2026-07-21-code-quality-pipeline.md`: this reproducibility record.

Temporary virtualenvs, hook caches, and validation repositories were created
only under `/tmp`; they are not project artifacts and may be discarded.

## Verification

The following final checks passed:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -B -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
```

Both all-file hook stages also passed in the temporary Git repository:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Git status, diff, staged-diff, commit, and revision checks were unavailable in
the live checkout. No commit or remote operation was attempted.
