# Engineering and Research Instructions

## Scope and interpretation

This file applies to the entire repository. Treat this project as both a
scientific codebase and a codebase that may grow into a production system.
Optimize for correctness, clarity, reproducibility, maintainability, and
auditable evidence.

The words **MUST**, **SHOULD**, and **MAY** are normative:

- **MUST** is a repository requirement unless it conflicts with a
  higher-priority instruction.
- **SHOULD** is the default; document the reason when choosing differently.
- **MAY** is optional.

Apply rigor proportionally. A change is substantive when it changes scientific
semantics, public behavior, architecture, dependencies, data, experiments,
figures, or a conclusion others may rely on. Typo-only and similarly mechanical
edits do not require a new research log or unnecessary infrastructure.

Unless explicitly stated otherwise, these requirements apply prospectively to
new or materially changed artifacts and to artifacts used as evidence for the
current task. Record legacy gaps, but do not broaden unrelated work solely to
backfill them.

## Project mission and invariants

This repository studies how world-model learning objectives affect performance
on downstream tasks, especially the contrast between predictable but
uncontrollable regularities and action-sensitive, controllable regularities.
The current foundation models $p(x' \mid x, \mathrm{pa}(x))$.

Preserve these established boundaries unless the task explicitly changes them:

- `scripts/` is an importable Python package despite its historical name.
- `scripts/mdp.py` owns transition semantics. Rewards, beliefs, learning
  algorithms, objectives, task distributions, orchestration, and evaluation
  belong in separate cohesive modules.
- The transition-model core remains standard-library-only.
- Visualization and notebook dependencies remain optional. The core must import
  without them; visualization adapters import optional libraries only when the
  corresponding feature is called.
- Factored transitions are synchronous. A cross-factor parent is read from the
  current joint state, never from another factor's sampled next state.
- Factor output scopes are disjoint. Only parents not predicted by any
  constituent factor are external parents of the composite model.
- Notebooks demonstrate and analyze library behavior; they are not the source
  of truth for reusable logic.
- Preserve the weather/umbrella example and its `open`/`close` actions unless the
  user explicitly requests a terminology change.

Discover the repository root from the checkout rather than embedding an
absolute path in code or artifacts. In the current managed workspace, the
expected checkout is `/data/user_data/nquazi/repos/evolving-world-models`. Do
not recreate or symlink the retired `/data/user_data/nquazi/evolving_world_models`
path. Do not assume that Git metadata is usable: verify it before relying on
status, diffs, or a revision, and state explicitly when revision information is
unavailable.

## Working process

Before substantive work:

1. Confirm the working directory and inspect the relevant implementation,
   tests, README, documentation, and recent logs.
2. Restate the objective, affected contracts, acceptance criteria, and any
   scientific hypothesis. Resolve materially ambiguous requirements before
   committing to an incompatible design.
3. Prefer the smallest coherent design that solves the current problem. Avoid
   speculative abstractions and unrelated cleanup.
4. Identify compatibility, reproducibility, numerical, security, and data-loss
   risks before changing files or running expensive experiments.

While working:

- Preserve unrelated user work and investigate unexpected file changes rather
  than overwriting them.
- Keep implementation, tests, public exports, examples, and documentation in
  sync.
- Separate observations from interpretations and speculation.
- Do not claim that a command, test, notebook, experiment, or figure was run
  unless it was actually run in the stated environment.
- Record failures and negative results; they are evidence, not clutter.
- Make external writes, network calls, expensive jobs, and destructive actions
  explicit. They must never be hidden in imports or constructors.

For a broad architectural change, write a short design note before
implementation covering the problem, constraints, proposed interfaces,
alternatives, migration, and validation plan. Keep the note with the relevant
log or documentation.

## Version control and commit discipline

### Authorization and repository safety

- Read-only Git inspection, such as `status`, `diff`, `log`, and `show`, MAY be
  used whenever Git metadata is valid.
- Creating or amending a commit requires a fresh, explicit user command for the
  current change, such as "commit these changes." A request to edit, fix,
  implement, verify, stage, prepare, or continue work does not authorize a
  commit. Do not infer authorization from an earlier turn, a prior commit, a
  standing workflow preference, or a general statement about committing in the
  future.
- Pushing requires its own fresh, explicit user command for the current change,
  such as "push this commit." Authorization to commit does not authorize a push,
  and authorization to push does not authorize a commit. A single command that
  explicitly says "commit and push" authorizes both actions in that order.
  Each authorization is one-shot; after the named action succeeds or the change
  scope materially changes, require another explicit command.
- Creating or amending tags, branches, or releases requires a fresh, explicit
  command naming that action. Authorization for one Git mutation never implies
  authorization for another.
- Verify the repository root with `git rev-parse --show-toplevel` before relying
  on Git. Never initialize, repair, replace, or symlink `.git` merely to make
  Git commands work.
- Treat a dirty worktree as shared. Pre-existing or unexpected changes belong to
  the user or another collaborator until proven otherwise.
- Never discard work with `git reset --hard`, `git clean`, `git checkout --`, a
  destructive `git restore`, or a broad stash. Do not amend another
  contributor's commit.
- Branch switches, rebases, stashes, and commits affect everyone sharing the
  worktree. Confirm that they will not disrupt another writer before proceeding.

### Staging and commit construction

- Keep commits small, cohesive, and independently understandable. Include the
  implementation, tests, documentation, and research log that make one logical
  change complete; separate unrelated refactors and generated artifacts.
- Stage only reviewed paths or hunks belonging to the task. Prefer explicit
  paths or `git add -p`; avoid `git add .`, `git add -A`, and `git commit -a` in
  a dirty shared worktree.
- Before committing, inspect at minimum:

  ```bash
  git status --short --branch
  git diff --check
  git diff --cached --name-status
  git diff --cached
  ```

- Verify that no unrelated file, credential, cache, local environment, raw
  dataset, checkpoint, or unexpectedly large binary is staged.
- Follow an established commit-message convention when one exists. Otherwise,
  use an imperative summary of roughly 50--72 characters. Add a body when it
  helps explain motivation, scientific or compatibility implications,
  alternatives, migration, and verification. Identify breaking changes
  explicitly.
- Do not bypass hooks with `--no-verify`. If an exceptional skip is explicitly
  approved, record the skipped check and reason in the handoff and relevant log.
- After committing, inspect the resulting commit and worktree. Report the commit
  identifier and any remaining staged or unstaged changes.

### Branches and remote history

- Keep branches focused and short-lived. Do not commit directly to a protected
  branch unless the repository workflow explicitly permits it.
- Fetching, pulling, merging, rebasing, force-pushing, and opening or merging
  pull requests require a fresh, explicit command naming that action because
  they affect shared or remote state. Apply the same one-shot, current-change
  scope used for commits and pushes.
- Never rebase published shared history without coordination. If rewriting a
  remote branch is explicitly approved and unavoidable, use
  `--force-with-lease`, never plain `--force`, and verify the exact remote and
  branch immediately beforehand.
- Resolve conflicts by understanding both sides. Never select ours or theirs
  wholesale merely to make a conflict disappear.

### Pre-commit and pre-push pipeline

Install the development tools and both hook stages from the repository root:

```bash
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
```

The first hook installation may require network access. The pinned pipeline is:

1. **Pre-commit:** whitespace, line-ending, TOML/YAML/JSON, merge-conflict,
   symlink, large-file, debug-statement, and private-key checks.
2. **Pre-commit:** Ruff applies safe lint fixes, organizes imports, and formats
   Python files; strict mypy checks production modules.
3. **Pre-push:** the complete unit suite and Python compilation checks run in
   the caller's development environment.

Run both stages across the repository before the first commit after setup and
before requesting review:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Formatting and hygiene hooks may modify files. Inspect every modification,
re-stage the intended changes, and rerun until clean. Do not run repository-wide
mutating hooks while another collaborator is actively editing the same files.
Never treat local hooks as the only enforcement layer: CI SHOULD independently
run the non-mutating Ruff, mypy, test, compilation, notebook-structure, and
packaging checks documented below.

Hook and tool versions are deliberately pinned. Update them in a dedicated,
reviewable change; read release notes, run `pre-commit autoupdate`, run every
quality gate, and record any resulting code-format or rule changes.

### When Git metadata is unavailable

If `git rev-parse --show-toplevel` fails, do not initialize or fabricate
history. Review changed files directly, run the full applicable verification
suite, maintain an explicit changed-file and artifact summary, and record `Code
revision: unavailable` in research logs. State in the handoff that status,
diff, staging, commit, and revision verification could not be performed.

## Software engineering standards

### Design and architecture

- Prefer cohesive modules, small composable functions, explicit interfaces,
  and immutable value objects where they clarify ownership.
- Keep domain logic independent of transport, storage, plotting, notebook, and
  command-line layers.
- Make dependencies point inward: adapters may depend on the domain model; the
  domain model must not depend on adapters.
- Avoid hidden mutable global state. Inject randomness, clocks, filesystem
  locations, configuration, and external clients when they affect behavior.
- Imports MUST NOT open windows, write artifacts, contact networks, or mutate
  process-global configuration. A clearly named API may perform an explicit,
  documented side effect when that is its purpose and the caller controls the
  target.
- Prefer clear standard-library implementations to new dependencies. When a
  dependency is justified, place it in the narrowest appropriate optional extra
  and update installation documentation and tests.
- Measure before optimizing. Preserve a readable reference path when adding a
  complex optimization, and benchmark both correctness and performance.

### Python and public APIs

- Support the minimum Python version declared in `pyproject.toml`, currently
  Python 3.9. Do not use newer syntax or library APIs without updating the
  version declaration, documentation, and compatibility checks together.
- Add type annotations to public APIs and non-obvious internal interfaces. Use
  types to communicate contracts; do not use casts to hide a design error.
- Treat documented APIs, names exported by `scripts/__init__.py`, serialized
  artifact schemas, and CLI arguments as compatibility contracts.
- Breaking changes require an explicit decision, updated callers and tests, and
  a migration note. Do not introduce accidental API changes during refactoring.
- Validate inputs at public boundaries. Reject missing or extra keys, invalid
  domains, shapes, ranges, non-finite values, overlapping scopes, malformed
  probabilities, unsafe paths, and invalid options with actionable errors.
- Do not introduce undocumented coercions of scientifically meaningful inputs,
  infer units, discard malformed records, or repair invalid models. Any
  coercion, normalization, or fallback retained for compatibility must be
  explicit, documented, and tested; tightening it is a compatibility decision.
- Return immutable views or defensive copies when exposing internal state.
- Maintain deterministic observable ordering. Never rely on set iteration or
  filesystem enumeration order for variables, factors, rows, records, or
  descriptions.
- Catch only exceptions that can be handled meaningfully. Preserve causal
  context with exception chaining and never use a bare `except` to hide failure.
- Use explicit UTF-8 encoding for text files and context managers for resources.
  Use atomic replacement when a partial write could corrupt an important
  artifact.
- `to_dict()` and `str(model)` are deterministic inspection representations,
  not a persistent interchange format. Introduce a separately versioned schema
  before treating them as durable storage.

### Numerical correctness and determinism

- All stochastic APIs MUST accept an injected random generator or explicit
  seed. Avoid hidden module-global randomness.
- Given the same code, inputs, configuration, dependency environment, and seed,
  results SHOULD reproduce within documented tolerances. Disclose platform or
  accelerator sources of nondeterminism.
- Preserve stable support and factor ordering because equivalent probability
  tables with different traversal order can produce different sampled
  sequences.
- Discrete probability masses must be finite, fall in `[0, 1]`, and normalize
  within a documented tolerance. Continuous densities must be finite and
  nonnegative where evaluated and satisfy their distribution-specific
  normalization contract; density values may exceed one. Use stable summation
  and justified tolerances rather than an arbitrary epsilon.
- Use log-space calculations when products can underflow. Specify and test
  behavior for zero-probability events.
- Check normalization and invariants across every finite conditioning context
  when feasible, not only hand-picked examples.
- Define the shape, dtype, coordinate convention, domain, range, and abbreviated
  units of numerical values wherever ambiguity is possible.

### Production and full-stack additions

If services, APIs, storage, or user interfaces are introduced:

- Keep request handlers and UI components thin; domain behavior remains in
  tested modules behind explicit contracts.
- Version external schemas. Validate requests and responses at boundaries, and
  document compatibility and migration behavior.
- Make retries bounded and safe, set timeouts, design mutating operations to be
  idempotent where practical, and expose useful failure states.
- Use structured, severity-appropriate logs with run/request identifiers. Never
  log secrets or unnecessarily sensitive data.
- Add health, latency, error, and resource observability appropriate to the
  deployment, while avoiding high-cardinality or private fields.
- Use semantic HTML, keyboard-accessible controls, visible focus states,
  sufficient contrast, responsive layouts, and explicit loading, empty, error,
  and success states. Provide a text/table alternative for important visual
  results.
- Apply least privilege, authenticate and authorize server-side, escape output,
  validate uploads and paths, and keep credentials in an approved secret store
  or environment—not source code, notebooks, logs, or client bundles.
- Validate configuration at startup, enforce request/upload/resource limits,
  make storage migrations forward- and rollback-aware, and document deployment
  and rollback procedures for externally visible changes.
- Test domain logic, API contracts, integration boundaries, and critical user
  journeys at the lowest reliable level. Do not replace unit tests with brittle
  end-to-end tests.

## Documentation standards

Documentation is part of the implementation. Update it in the same change as
the behavior it describes. Prefer one authoritative source for each fact and
cross-link rather than copying details that will drift.

### Python module and script documentation

Every new or materially changed nontrivial `scripts/*.py` module MUST start with
a substantive module docstring that covers, as applicable:

- purpose, repository role, responsibilities, and explicit non-goals;
- scientific or mathematical contract and relevant notation;
- public entry points and a minimal usage example;
- inputs and outputs, including schemas, shapes, domains, ranges, and units;
- important assumptions, invariants, approximations, and failure modes;
- randomness, ordering, numerical tolerances, and determinism guarantees;
- side effects and artifacts written;
- required and optional dependencies; and
- important complexity or scaling behavior.

Every new or materially changed public class, function, and method MUST have a
useful docstring. Prefer Google-style `Args`, `Returns`/`Yields`, `Raises`, and
`Notes` sections for nontrivial APIs. A concise docstring is sufficient for a
genuinely self-evident accessor. Document private helpers when their algorithm,
assumptions, or invariants are not obvious. Comments should explain *why* a
choice is necessary, not narrate the syntax.

When adding or materially changing a nontrivial executable, experiment driver,
data pipeline, or plotting program, create or update
`docs/scripts/<module-name>.md`. It MUST document:

1. Purpose and placement in the workflow.
2. Public API or CLI contract.
3. Input/output schemas, units, and example files.
4. Configuration precedence and defaults.
5. Algorithm or data flow, with equations when useful.
6. Validation, errors, overwrite/resume semantics, and recovery.
7. Determinism, seeds, expected runtime, and resource needs.
8. Complete invocations and expected artifacts.
9. Test coverage and known limitations.

Executable modules SHOULD expose `main(argv=None) -> int` and invoke it with
`raise SystemExit(main())` under the main guard. They should provide validated
`argparse` arguments and useful `--help`, return a nonzero status on failure,
and perform no work at import time. Prefer machine-readable outputs for
downstream automation.

### User-facing and architectural documentation

- Update the README when installation, layout, public APIs, examples, or
  canonical workflows change.
- Update domain-specific documentation when semantics, schemas, graph meanings,
  artifact formats, or extension points change.
- Include small runnable examples. Verify examples rather than treating them as
  illustrative pseudocode unless clearly labeled.
- Cite the source of externally derived algorithms, equations, datasets, or
  evaluation protocols.
- Record important design decisions and rejected alternatives when the reason
  would otherwise be lost.

## Testing and verification

- Use the established `unittest` style unless the repository deliberately
  adopts and configures another framework.
- Every behavior change MUST test normal behavior and applicable boundaries and
  failures. Defect fixes MUST add a regression test when feasible.
- As applicable, preserve and extend coverage of exact scopes and domains,
  table completeness, probability normalization, impossible events, log-space
  behavior, deterministic
  descriptions, seeded sampling, synchronous factor projection, graph
  semantics, optional-dependency errors, and headless rendering.
- Prefer exact contract tests to flaky sampling-frequency assertions. For
  approximate results, use a tolerance justified by the numerical contract.
- Unit tests and the default verification suite MUST be isolated,
  order-independent, offline, deterministic, and free of machine-specific paths
  and wall-clock sleeps. Use temporary directories for filesystem tests. Tests
  requiring provisioned services or external systems must be explicitly marked,
  separately invoked, bounded by timeouts, and documented.
- Keep plotting tests headless, never call `show()`, close created figures, and
  test semantic structure rather than brittle pixel snapshots.
- Test at the narrowest useful layer, then run the complete applicable suite.
- Do not weaken or delete a failing test merely to make a change pass. Update a
  test only when the intended contract changed, and document that decision.

Baseline verification from the repository root, using the documented
environment with development extras, is:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
```

The JSON command checks notebook structure only; it does not execute cells. Any
new or materially changed notebook, or notebook used as result evidence, MUST
also be executed from a fresh kernel into a temporary output, for example:

```bash
notebook_run_dir="$(mktemp -d)"
jupyter nbconvert --execute --to notebook \
  --output-dir="$notebook_run_dir" path/to/notebook.ipynb
```

Run additional tests for new modules, CLIs, artifact schemas, experiments, or
front-end/service layers. If linting, formatting, typing, coverage, packaging,
or notebook-execution tools become required, declare and configure them in
`pyproject.toml` and add their exact commands here. Never claim unconfigured
quality gates as if they ran. Report every skipped or failed check and why.
When package layout, exports, or `pyproject.toml` changes, build and install the
distribution in a clean temporary environment and import it from outside the
checkout. Compatibility-sensitive changes must be tested on Python 3.9 and at
least one current supported Python version, preferably through CI once it is
configured. When the core or dependency declarations change, also verify core
imports and focused core tests in an environment without optional extras.

## Notebook practices

- An affected notebook or one used as evidence MUST execute top-to-bottom from
  a fresh kernel in the documented environment. If execution is blocked, do not
  claim it was verified; document the blocker and treat notebook-dependent work
  as incomplete unless the task explicitly accepts that limitation.
- Keep execution counts sequential, or clear them all. Remove stale and error
  outputs. Retain only small, intentional outputs that help the reader.
- Keep setup, imports, versions, seeds, and configuration near the top. Do not
  depend on hidden state, undocumented environment variables, or absolute local
  paths.
- Use package implementations rather than copying production algorithms into
  cells. Move reusable transformations, metrics, and plotting helpers into
  tested modules.
- Assert important invariants in code cells so failures are visible.
- Use a portable kernelspec that resolves in the documented setup. Avoid
  machine-specific metadata; record the exact execution environment in the
  research log rather than relying on notebook metadata to encode it.
- Do not commit credentials, private data, huge embedded binaries, debug noise,
  or machine-specific metadata.
- Treat notebook findings as demonstrations unless a reproducible run and
  research log support the claim.

## Research-production standards

Treat every reported result as an auditable scientific claim. Canonical
experiments SHOULD run from scripts or modules; notebooks are appropriate for
exposition and exploration.

### Reproducibility and provenance

Every substantive experiment, benchmark, or scientific claim MUST record:

- exact copy-pasteable commands run from the repository root;
- code revision and working-tree status, or an explicit statement that version
  control metadata was unavailable;
- Python and relevant package versions, operating system, and material hardware;
- complete configuration, including consequential defaults;
- the exact seed list and which random generators were seeded;
- data/environment origin, version, selection criteria, split, preprocessing,
  license, and checksum when practical;
- output paths and a manifest of raw measurements, checkpoints, tables, and
  figures;
- expected runtime and resource requirements, plus actual start/end time,
  elapsed runtime, and material compute use for completed runs;
- hyperparameter search spaces, tuning budgets, checkpoint-selection rules, and
  selection metrics when tuning is performed; and
- known nondeterminism and whether exact reproduction or statistical replication
  is expected.

Use repository-relative paths in documentation. Never depend on secrets, hidden
notebook state, or undocumented local state. Runs MUST use unique output
locations and MUST NOT silently overwrite earlier raw results. Preserve raw
per-run measurements before aggregation, and generate tables and figures from
saved measurements rather than manual transcription.
If a material artifact is too large to commit, record a durable storage URI,
retrieval instructions, checksum, access requirements, and retention policy;
an ephemeral local path is not sufficient provenance.

### Experimental design and statistical reporting

Before a confirmatory run, record and timestamp the research question,
hypothesis, primary metric and direction, evaluation distribution, baseline,
exclusion rules, and success criterion. If these are set after outcomes are
examined, label the analysis exploratory. Do not tune on the final test set.

For quantitative results:

- Identify the independent experimental unit. Episodes from one trained seed
  are not independent training replicates.
- Report the exact numbers of seeds, tasks, environments, episodes, and samples
  as applicable.
- Preserve individual-run values in a machine-readable artifact, or document
  why preservation is impractical.
- For sampled or stochastic results, report an appropriate central estimate and
  a named measure of variability or uncertainty, such as standard deviation or
  a 95% confidence interval. For exact deterministic results, state that
  sampling uncertainty is not applicable.
- Aggregate at the correct level; account for observations nested within seeds
  or tasks.
- Prefer paired comparisons using matched seeds and task instances when valid.
- When inferential tests are used, report effect sizes and intervals rather than
  only p-values, and state the test and its assumptions. When multiple
  comparisons are tested, state the correction or justify why none was used.
- Define every error bar; never report an unexplained `+/-`.
- Predefine exclusion rules and account for every failed or excluded run.
- Compare methods under equivalent data, tuning, evaluation, and compute budgets,
  or disclose the difference.
- Distinguish correlation from causation and exploratory patterns from
  confirmatory evidence.

Choose the number of independent replications based on expected variability and
the decision being made. Never cherry-pick seeds, checkpoints, tasks, metrics,
or plot ranges.

### Required mini-paper logs

Create or update one log under `docs/logs/` for every substantive implementation
milestone, experiment, benchmark, or interpretation used to support a decision.
Use `YYYY-MM-DD-short-descriptive-slug.md`; use distinct slugs for unrelated work
on the same date. A coherent multi-step effort may share one evolving log.

Logs are durable records. Include null, negative, incomplete, and failed work.
After a result has been shared or relied upon, correct it with a clearly dated
addendum instead of silently rewriting the historical conclusion.

Every log MUST contain the following sections. Retain a non-applicable section
and explain briefly why it does not apply.

```markdown
# Descriptive title

Status: planned | running | complete | superseded
Date/time: ISO 8601 with UTC offset
Code revision: revision plus dirty/clean status, or unavailable
Run/config identifiers:
Primary artifacts:

## Overview

## Research question, hypothesis, or acceptance criteria

## Setup

## Replication instructions

## Methods

## Validation and quality checks

## Results

## Figures and tables

## Conclusions

## Limitations and threats to validity

## Deviations, failures, and negative results

## Next directions

## Change and artifact summary

## Verification
```

The body MUST provide enough detail to reproduce the work without access to the
original author's shell history or memory:

- **Overview:** what changed or was tested, why it matters, and the principal
  outcome.
- **Question/hypothesis:** the pre-run hypothesis or engineering acceptance
  criteria, primary metric, metric direction, and success criteria.
- **Setup:** environment and dependency versions, hardware where relevant,
  inputs and provenance, data/task splits, configuration, seeds, and material
  environment variables.
- **Replication:** environment setup, data preparation, exact commands,
  aggregation and figure commands, expected and actual runtime, outputs, and
  success checks.
- **Methods:** models, algorithms, objectives, baselines, equations/references,
  hyperparameters and any search/tuning budget, checkpoint-selection rules,
  protocols, aggregation, and statistics.
- **Validation:** tests, invariants, sanity checks, leakage checks, and diagnostic
  comparisons.
- **Results:** quantitative results with applicable sample sizes and uncertainty
  (or a statement that the result is exact), artifact links, test outcomes, and
  failed or null results.
- **Figures/tables:** artifact links and standalone captions defining conditions,
  applicable sample sizes, aggregation, uncertainty, units, and metric
  direction or interpretation.
- **Conclusions:** observations supported by evidence, separated from bounded
  interpretation and speculation.
- **Limitations:** confounders, nondeterminism, power, measurement constraints,
  distribution shift, and generalization boundaries.
- **Deviations/failures:** deviations from plan, exclusions with reasons, crashes,
  partial runs, and recovery steps.
- **Next directions:** concrete prioritized follow-ups and the evidence each
  would seek.
- **Change/artifacts:** changed files, public behavior, compatibility impact,
  configs, raw data, checkpoints, tables, and figures.
- **Verification:** exact test, lint, type-check, notebook, and replication
  commands actually run, with outcomes.

Add sections such as Background, Design alternatives, Ablations, Ethics, Safety,
Data governance, Cost, or References when pertinent. For implementation-only
milestones, test results, contract checks, and demonstrated behavior may be the
results; never imply that an unrun scientific experiment succeeded.

## Publication-quality figures

Finished research figures MUST be generated by repository code. Empirical
figures must use saved machine-readable measurements; analytical figures and
structural diagrams may use a fully recorded deterministic model or
configuration. Conceptual diagrams may instead use a versioned specification.
Do not manually alter plotted values or use image editing as part of the
analysis pipeline. Preserve plotted measurements or the generating
specification, record the exact generation command, and record the code revision
when available—or explicitly state that revision metadata was unavailable.

Every coordinate-based quantitative plot MUST have:

- a concise, descriptive title;
- meaningful x- and y-axis labels;
- abbreviated units in parentheses when applicable, such as `Time (s)`,
  `Memory (GiB)`, `Normalized return (unitless)`, or `Training steps (M)`;
- a legend or direct labels when multiple series appear;
- legible ticks and text at final publication size; and
- a standalone caption stating what is shown, experimental conditions,
  applicable sample size, aggregation and uncertainty definition, units, and
  the metric direction or interpretation. Define any normalization reference.
  For an exact result, state that sampling uncertainty is not applicable.

Coordinate-free diagrams, images, and network schematics are exempt from x/y
labels only when axes have no scientific meaning. They still require a concise
title or caption and an explanation of node, edge, color, shape, line, or panel
encodings. Heatmaps MUST label both axes and the color bar, including units when
applicable.

Figure design MUST follow these rules:

- Use a restrained, colorblind-friendly palette, such as Okabe-Ito or a
  perceptually uniform sequential palette. Avoid rainbow/jet maps and
  red-green-only comparisons.
- Do not encode an important distinction with color alone; add line styles,
  markers, hatching, shapes, or direct labels.
- Keep colors, ordering, fonts, names, scales, and metric direction consistent
  across related figures.
- For sampled empirical results, define uncertainty explicitly as SD, SE, CI,
  quantiles, or another named quantity, and include the sample/seed count. For
  exact results, say that sampling uncertainty is not applicable.
- Use honest scales. Bar charts ordinarily start at zero; mark and justify axis
  truncation or breaks. Label logarithmic axes and transformations.
- Avoid dual y-axes, decorative 3-D effects, chartjunk, saturated backgrounds,
  excessive precision, and legends that obscure data.
- Include chance, baseline, or oracle references when they clarify the claim.
- Use consistent panel labels such as `(a)` and `(b)` and comparable limits where
  visual comparison depends on them.
- Shared x/y labels and a single overall title are acceptable for multi-panel
  figures when the mapping is unambiguous.
- Prevent clipped labels and overlapping annotations. Inspect every final figure
  at its intended display size; successful rendering alone is insufficient.
- Save line art and text as PDF or SVG when practical. Save raster-dependent
  figures at publication resolution, normally at least 300 dpi, and provide a
  convenient PNG preview when useful.
- Use deterministic descriptive filenames. Never silently overwrite an earlier
  result from a different run or configuration.

Plotting library functions SHOULD accept caller-owned Matplotlib axes and return
figure/axes or artist objects. They MUST NOT call `show()` or save implicitly.
Batch scripts must close figures. Low-level artist constructors need not impose
presentation-level titles or labels; the code assembling a finished figure is
responsible for them. Stable finished-figure builders SHOULD have headless tests
for titles, axis labels, units, legends, expected series, and deterministic
semantic structure where practical. Lower-level primitives should test their
own semantic contract.

## Publication-quality tables

Quantitative result tables MUST include a descriptive title or standalone
caption, applicable units, metric direction or interpretation, applicable sample
sizes, and uncertainty definitions. For exact results, state that sampling
uncertainty is not applicable. Use precision justified by the measurement and
uncertainty. Explain missing values and exclusions. Keep terminology and
ordering consistent with figures, do not use color as the only highlight, and
preserve a machine-readable version of the table.

## Security, privacy, and artifact hygiene

- Never commit credentials, tokens, private keys, `.env` files, personally
  identifying data, or proprietary datasets.
- Read secrets from an approved secret store or environment and never dump the
  entire environment or configuration when it may contain secrets.
- Validate paths and file formats. Avoid `eval`, unsafe deserialization, and
  untrusted pickle-like artifacts.
- Record data origin, license, permissions, version, checksum, preprocessing,
  exclusions, and split construction.
- Never overwrite raw data or canonical results in place. Use explicit,
  run-specific derived-output locations.
- Before adding local virtual/Conda environment directories, downloads,
  datasets, checkpoints, or generated outputs, add narrow ignore rules and
  document how each artifact is recreated.
- Do not commit caches, temporary files, large outputs, or embedded binaries
  unless they are intentional, justified, and documented.

## Definition of done

Before declaring work complete, verify all applicable items:

- The requested behavior is implemented without unrelated changes.
- Architecture, public contracts, and compatibility remain coherent.
- Targeted tests and the full applicable verification suite pass.
- The configured pre-commit and pre-push gates pass, or every unavailable check
  is reported with its reason.
- Numerical behavior, determinism, error cases, and resource cleanup are tested
  as applicable to the change.
- Public APIs, script pages, README, examples, and dependency declarations match
  the implementation.
- A substantive effort has a complete mini-paper log with exact replication
  instructions, artifacts, results, limitations, and next directions.
- Affected or relied-upon notebooks run from fresh state and contain no hidden
  or stale execution state.
- New, changed, or reported figures and tables pass the publication-quality
  checklist and link to source data or their generating specification.
- Secrets, private data, generated clutter, and unintended large files are
  absent.
- The final handoff states what changed, what was verified, what was not run,
  remaining limitations, and the next useful direction.

If Git is usable, inspect the final diff and working-tree status. If it is not,
review the changed files directly and say that Git verification was unavailable.
