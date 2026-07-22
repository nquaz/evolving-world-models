# Public repository bootstrap and CI validation

Date: 2026-07-21
Timezone: America/New_York
Status: Complete
Validated revision: 6cf6a275d281a66e055ed06de179ff301f8bf725
Root revision: 73092a3d00e3309ac52b070c8333997eeb340b68
Remote: https://github.com/nquaz/evolving-world-models
CI run: https://github.com/nquaz/evolving-world-models/actions/runs/29878570532
Primary artifacts: LICENSE, pyproject.toml, .pre-commit-config.yaml,
.github/workflows/quality.yml

## Overview

This milestone converted the validated local project into a new public GitHub
repository with explicit licensing, privacy-aware commit metadata, local Git
hooks, and independent continuous integration. The initial source snapshot and
the CI workflow were kept as separate atomic commits. The public remote was
created only after a staged-content audit, successful local gates, and explicit
informed approval of the disclosure scope.

The published baseline implements tabular and factored transition models,
optional NetworkX, Matplotlib, and Graphviz adapters, a worked notebook, tests,
engineering and visualization documentation, and the pinned local quality
pipeline.

## Objective

The operational question was whether the project could be published as a clean,
reproducible baseline while preserving the following properties:

- no accidental credential, cache, environment, or large-artifact publication;
- an explicit permissive license represented correctly in package metadata;
- honest initial history rather than reconstructed intermediate commits;
- local pre-commit and pre-push enforcement;
- independent CI on the minimum documented Python version and a current
  development version;
- fresh-kernel notebook execution and package-build validation; and
- an auditable record of publication decisions, results, and remaining risks.

## Setup

- Local platform: Linux 5.14.0-503.40.1.el9_5.x86_64
- Local environment: Conda environment world_models
- Local Python: 3.12.13
- GitHub CLI: 2.96.0
- pre-commit: 4.3.0
- Ruff: 0.15.22
- mypy: 1.19.1
- Default branch: main
- Repository visibility: public
- License: Apache License 2.0
- Git author: GitHub account name with the account's ID-based noreply address
- CI matrix: Python 3.9 and Python 3.12 on ubuntu-24.04

GitHub CLI used HTTPS authentication. Because no system credential store was
available, its token was stored outside the repository in a file with mode
0600. The token was never printed, copied into a command, or added to Git.

## Replication instructions

### Contributor setup

From a fresh machine with Git, Conda, and Python available:

~~~bash
git clone https://github.com/nquaz/evolving-world-models.git
cd evolving-world-models
conda create --name world_models python=3.12
conda activate world_models
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
~~~

Run the complete local verification sequence:

~~~bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -B -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
~~~

Execute the notebook from a fresh kernel:

~~~bash
python -m jupyter nbconvert \
  --to notebook \
  --execute \
  --output Factored_MDP_Demo.executed.ipynb \
  --output-dir /tmp \
  --ExecutePreprocessor.timeout=120 \
  notebooks/Factored_MDP_Demo.ipynb
python -m json.tool /tmp/Factored_MDP_Demo.executed.ipynb > /dev/null
~~~

Build the distribution in isolation:

~~~bash
python -m pip wheel --no-deps --wheel-dir /tmp/ewm-wheel .
python -m pip check
~~~

### Maintainer bootstrap verification

The one-time remote workflow used:

~~~bash
gh auth login --hostname github.com --git-protocol https --web
gh repo create nquaz/evolving-world-models \
  --public \
  --source=. \
  --remote=origin \
  --description 'Transition-model foundations for studying how world-model objectives shape downstream control.'
git remote -v
git push --set-upstream origin main
gh run list --repo nquaz/evolving-world-models --workflow quality.yml
~~~

Do not run the repository-creation command for an existing remote. Never place
an authentication token in a shell argument, log, notebook, or tracked file.

## Methods

### Snapshot audit

The candidate initial snapshot was enumerated with Git-aware and no-ignore file
searches. File sizes, symlinks, executable bits, generated caches, credential-
like filenames, private-key headers, and common service-token patterns were
checked. The scan found no likely credentials, symlinks, special files, or
unignored files over 1 MiB. This was a targeted regex and filesystem audit, not
a substitute for a dedicated secret-scanning product.

Generated mypy, Ruff, bytecode, and editable-install artifacts were confirmed
ignored. The ignore policy was expanded for local environment files, additional
virtual-environment layouts, coverage/test output, editors, and operating-system
artifacts without broadly ignoring research data or result directories.

### Publication cleanup

The missing docs/drawing.md contract was reconstructed from the implementation
and tests. The demonstration notebook was changed to avoid printing its absolute
checkout path, given a standalone figure caption, and re-executed top-to-bottom.
The resulting nine code cells had sequential execution counts, zero error
outputs, one retained inline figure, and no machine-specific path string.

Three intentionally retained historical path references were identified in
AGENTS.md and docs/logs/2026-07-21-code-quality-pipeline.md. The user was
explicitly informed that these references, along with source, notebook outputs,
documentation, logs, and commit metadata, would become public.

### Licensing and packaging

The root LICENSE is byte-for-byte equal to GitHub's canonical Apache-2.0
template. pyproject.toml declares the SPDX expression Apache-2.0 and explicitly
lists LICENSE under PEP 639 metadata, with setuptools 77.0.3 or newer as the
minimum supporting backend.

An isolated wheel build produced Core Metadata 2.4 with
License-Expression: Apache-2.0 and License-File: LICENSE. Inspection confirmed
that the wheel contains its license under the distribution metadata licenses
directory.

### Commit construction

The initial baseline was committed as:

- 73092a3: Initialize evolving world-model foundation

The CI addition was kept separate:

- 6cf6a27: Add GitHub Actions quality checks

Both commits ran the configured hooks without bypasses. The final pre-publication
worktree was clean, and the local and remote main references both resolved to
6cf6a275d281a66e055ed06de179ff301f8bf725 before this log was added.

### CI design

The Quality workflow runs on pushes to main, pull requests, and manual
dispatches. It has read-only contents permission, disables persisted checkout
credentials, cancels superseded runs in the same concurrency group, and limits
each matrix job to 20 minutes.

Third-party actions are pinned to immutable commits:

- actions/checkout v6.0.2:
  de0fac2e4500dabe0009e67214ff5f5447ce83dd
- actions/setup-python v6.2.0:
  a309ff8b426b58ec0e2a45f0f869d46889d02405

Each Python job validates configuration and notebook JSON, runs Ruff and mypy,
runs all unit tests and compilation checks, executes the notebook from a fresh
kernel, checks the installed environment, and builds a wheel. The workflow was
validated locally with checksum-verified actionlint 1.7.12 before publication.

### Public-disclosure authorization

The first repository-creation attempt was rejected by an external-data
safeguard because a public repository exposes workspace content to an untrusted
external destination. No workaround or indirect publication attempt was made.
The user was informed of the exact disclosure categories and the three retained
local-path references, then explicitly confirmed public publication. Only after
that confirmation was the repository created and main pushed.

## Results

| Check | Scope | Result |
| --- | --- | --- |
| Credential/artifact audit | Initial snapshot | No likely secret or unintended large file found |
| Ruff lint | scripts and tests | Passed |
| Ruff format check | 5 Python files | Passed |
| Strict mypy | 3 production modules | Passed |
| Unit tests | 33 tests | Passed |
| Python compilation | scripts and tests | Passed |
| Notebook structural validation | Committed notebook | Passed |
| Notebook clean execution | 9 code cells | Passed; 0 errors, 1 figure |
| Pre-commit stage | All files | Passed |
| Pre-push stage | All files | Passed |
| Isolated wheel build | Python 3.12 | Passed; Apache metadata and file present |
| Remote identity | nquaz/evolving-world-models | Public, default branch main |
| Remote SHA check | origin/main | Matched 6cf6a275 before this log |
| GitHub Actions Python 3.9 | Run 29878570532 | Passed in 51 seconds |
| GitHub Actions Python 3.12 | Run 29878570532 | Passed in 35 seconds |

The initial GitHub Actions run completed successfully at
https://github.com/nquaz/evolving-world-models/actions/runs/29878570532.
Every named step passed for both matrix jobs, including dependency installation,
fresh-kernel notebook execution, and wheel construction.

## Conclusions

The project now has a public, licensed, and independently validated baseline.
Local hooks provide fast feedback before commits and pushes, while GitHub
Actions repeats non-mutating checks in clean Python 3.9 and 3.12 environments.
The successful first remote run demonstrates that the repository can be cloned,
installed, tested, executed, and packaged outside the originating workspace.

Separating the foundation and CI commits preserved a truthful initial history.
Explicit disclosure review, privacy-preserving author metadata, immutable action
pins, read-only workflow permissions, and package-license inspection materially
reduced avoidable publication and supply-chain risks.

## Limitations

- The credential scan used targeted patterns and pre-commit's private-key check;
  it was not a full historical or entropy-based secret scan.
- Optional scientific and notebook dependencies use compatible version ranges
  rather than a complete environment lock, so future resolutions may differ.
- CI covers Python 3.9 and 3.12, not every interpreter satisfying Python 3.9 or
  newer.
- ubuntu-24.04 is named explicitly, but the hosted runner image evolves.
- The three disclosed historical checkout-path references remain public by
  informed choice.
- Branch protection, required status checks, Dependabot, a SECURITY policy,
  release automation, and artifact attestations are not yet configured.
- A single successful remote run establishes bootstrap viability, not long-term
  reliability.

## Deviations and issues

The live checkout originally contained an empty, read-only .git directory.
After the user confirmed that this was a new repository, the empty directory was
made writable and Git was initialized on main. No history was reconstructed.

The first notebook execution attempt tried to create Jupyter configuration in a
read-only home location. Temporary Jupyter directories resolved that
environmental issue. A subsequent headless run using MPLBACKEND=Agg suppressed
the intended inline figure; the visualization cell was made explicitly inline
and re-executed, restoring the figure without a redundant Axes representation.

The first public-repository creation request was rejected by the publication
safeguard. Work stopped at that boundary until explicit informed confirmation
was received.

## Artifacts

- LICENSE: canonical Apache License 2.0 terms.
- pyproject.toml: project, dependency, tool, and PEP 639 license metadata.
- .pre-commit-config.yaml: pinned pre-commit and pre-push gates.
- .github/workflows/quality.yml: read-only Python matrix CI.
- README.md: installation, API, verification, CI, and license overview.
- AGENTS.md: repository engineering and research-production policy.
- docs/drawing.md: exact visualization semantics and API contract.
- notebooks/Factored_MDP_Demo.ipynb: executed worked example and figure.
- docs/logs/2026-07-21-code-quality-pipeline.md: local quality-pipeline study.

## Next directions

1. Decide and document branch-protection and required-status-check policy after
   confirming the preferred solo- or multi-contributor workflow.
2. Add Dependabot or an equivalent reviewed dependency-update process for Python
   packages, pre-commit hooks, and GitHub Actions.
3. Add SECURITY.md, CONTRIBUTING.md, issue templates, and pull-request guidance
   before soliciting external contributions.
4. Define a release policy, versioning rules, changelog format, package build
   provenance, and artifact-attestation workflow before the first release.
5. Evaluate a lock or constraint strategy for exact long-term experiment
   replication while retaining minimum-version compatibility testing.
6. Add current-Python coverage beyond 3.12 when the dependency stack is
   validated, and periodically test the lowest supported dependency versions.
7. Continue the scientific roadmap with rewards, Bayesian transition beliefs,
   learning objectives, downstream-task distributions, preregistered
   experiments, machine-readable result tables, and publication-quality figures.
