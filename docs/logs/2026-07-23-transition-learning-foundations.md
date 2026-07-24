# Transition-learning foundations: example, beliefs, planning, and allocation

Status: complete
Date/time: 2026-07-23T09:49:03-04:00 to 2026-07-24T03:49:02-04:00
Code revision: `ce801b3b6ca1c724e4ca20b08ce19d90c2a65068`;
dirty working tree containing all milestones recorded here
Run/config identifiers: `lock-door-example-v1`, `final-notebook2-S6yA5S`,
`dirichlet-beliefs-v1`, `belief-contract-py39`, `belief-contract-py312`,
`repository-quality-py312`, `clean-wheel-v2`,
`predictive-log-underflow-regression`,
`two-clock-20260723T213913748853Z-75d60366814d`,
`allocation-sensitivity-v1`,
`two-clock-20260723T222025252737Z-69ce6650bd38`
Primary artifacts: `notebooks/Factored_MDP_Demo.ipynb`,
`scripts/beliefs.py`,
`docs/reference/beliefs.md`, `scripts/planning.py`,
`scripts/clock_experiment.py`, `scripts/experiment_plotting.py`,
`notebooks/Two_Clock_Allocation_Experiment.ipynb`,
`docs/reference/two-clock-allocation.md`,
`results/two-clock/two-clock-20260723T222025252737Z-69ce6650bd38`
(local, Git-ignored, and not durable)

## Overview

### Lock/door example rewrite

The first milestone replaced the active weather/umbrella example with a
lock/door world while preserving synchronous factored-MDP semantics. The lock
is predictable but not controlled by the modeled action. The door is
action-sensitive and conditions on the current lock setting. The July 20 and
July 21 logs remain unchanged because they accurately document the previous
repository state.

The active README, notebook source and saved outputs, graph, repository
invariant, and transition/visualization fixtures now use the lock/door example.
The notebook executes from a fresh kernel, and the saved output is semantically
identical to an independent final execution.

### Conjugate Dirichlet transition beliefs

The second milestone implemented conjugate Bayesian beliefs over complete
finite transition tables while preserving the boundary between fixed
transition semantics and mutable epistemic state. `scripts/mdp.py` remained
unchanged; `scripts/beliefs.py` owns explicit posterior updates,
posterior-predictive queries, fixed posterior-mean snapshots, and samples of
plausible MDPs.

The implementation includes tabular and factored beliefs. A factored update
projects cross-factor parents from the current joint state and validates every
local target before incrementing any count. The critical lock/door regression
therefore updates the door row conditioned on current `lock=unlocked`, even
when the observed next lock is `locked`.

All requested local acceptance checks passed on Python 3.9 and 3.12, including
the post-documentation repository hooks. CI was not run because no commit or
push was authorized.

### Two-clock finite-budget allocation experiment

The third milestone added a standard-library planning layer and an experiment
orchestrator for two independent cyclic clocks. Clock `x` is controllable by a
`hand` action; clock `y` is predictable but uncontrollable. Both factors share
the same intended-transition reliability, but the predictable belief
deliberately retains a separate row for every `(y, hand)` context. The
experiment allocates a finite number of updates per local context between the
two factors, samples one plausible posterior model, plans by exact
finite-horizon dynamic programming, and evaluates each policy in the true
environment.

The implementation produces versioned raw trials, summaries, a complete seed
ledger, manifests, and publication-oriented heatmaps. A fresh-kernel bounded
notebook demonstration exercised the pipeline end to end. It is engineering
and exploratory validation only; no confirmatory budget-allocation or task
performance conclusion is claimed.

### Allocation sensitivity and planning runtime

The fourth milestone replaced the notebook's fixed-total allocation table with
a two-panel sensitivity figure over total per-context budget
\(N=N_x+N_y\). For each task and positive even \(N\), it compares equal
allocation with the best observed below-diagonal allocation
\((N_x>N_y)\) and best observed above-diagonal allocation
\((N_x<N_y)\). Deterministic ties prefer the allocation nearest the diagonal
and then canonical \((N_x,N_y)\) order.

The selector retains the original cell counts, point estimate, Monte Carlo
standard error, and Wilson 95% interval. The plotting adapter shows three
ordered, labeled lines with pointwise asymmetric Wilson error bars and
\((N_x,N_y)\) annotations. The best-side curves are explicitly labeled
post-hoc oracle envelopes: maximizing and evaluating on the same outcomes
creates winner's-curse bias, and the displayed cellwise intervals are not
selection-adjusted.

An exact fresh-kernel exploratory run used \(M=6,H=30,T=100\), a
\(5\times5\) allocation grid, and fixed totals \(N\in\{2,4,6\}\). It
produced 5,000 binary records, 50 summaries, 18 selected strategy points, and
both heatmap and sensitivity PDF/PNG pairs. The complete verified run remains
available locally under the ignored `results/two-clock/` tree. Its manifest
makes the current local copy auditable, but no external durable URI was
provisioned, so it is not durable publication evidence.

The first one-trial runtime check exposed repeated transition-row validation:
30.741 seconds before optimization. A local per-invocation row cache in
`scripts/planning.py` reduced the same unprofiled check to 7.953 seconds
without changing public APIs, arithmetic order, tie-randomization order, or
model state. The exact notebook runner then took 846.696 seconds. A separate
instrumented profile attributed essentially all cumulative time to the 30
planning and 30 exact-evaluation calls rather than belief construction.

## Research question, hypothesis, or acceptance criteria

### Lock/door example rewrite

This was an implementation milestone, not a sampled experiment. Acceptance
required:

- `lock in {locked, unlocked}`, `door in {closed, open}`, and
  `action in {open, close}`;
- $p(\ell',d'\mid\ell,d,a)=p(\ell'\mid\ell)p(d'\mid d,\ell,a)$;
- lock-setting persistence of 0.95 with no action parent;
- `action` as the composite model's only external parent;
- the door factor to use $\ell_t$, never $\ell_{t+1}$;
- exact representation of all eight recommended door rows;
- `locked` + `open` in the unconstrained Cartesian state space;
- normalization of all 10 local rows and all 8 joint contexts;
- exact probability $0.05\times0.9=0.045$ for current
  `unlocked`/`closed`, action `open`, and next `locked`/`open`; and
- a fresh-kernel notebook run with a legible saved graph.

These are exact table and structure checks. Metric direction, experimental
unit, sample size, and statistical uncertainty are not applicable.

### Conjugate Dirichlet transition beliefs

This was an engineering and mathematical-correctness milestone, not a
confirmatory performance experiment. The question was whether independent
Dirichlet beliefs could represent every finite transition row and compose
factorwise without changing existing MDP semantics or adding a runtime
dependency.

Acceptance required:

- public `AbstractTransitionBelief`, `TabularDirichletBelief`, and
  `FactoredDirichletBelief` classes;
- positive finite scalar priors and complete explicit concentration rows;
- exact updates and posterior-predictive probabilities
  $(\alpha+n)/\sum_z(\alpha_z+n_z)$;
- concentration-space log probabilities that remain finite when a positive
  normalized float mass underflows to zero;
- context isolation, defensive inspection, and deterministic descriptions;
- fresh posterior-mean MDP snapshots;
- normalized, reproducible seeded samples of plausible MDPs;
- exact factored products and atomic joint updates;
- current-time, rather than next-time, cross-factor parent projection;
- Python 3.9 compatibility and no optional dependency at core import time; and
- synchronized exports, README, reference documentation, tests, and research
  records.

The regression-critical observation was:

```text
current:    lock=unlocked, door=closed
parents:    action=open
next_state: lock=locked, door=open
```

Success required one lock update in the current-unlocked row, one door update
in the current-unlocked parent row, and no change to the door row conditioned
on current `lock=locked`. Exact contract-test success was the primary metric;
metric direction and sampling uncertainty were not applicable.

### Two-clock finite-budget allocation experiment

The engineering question was whether the repository could express and
reproduce the complete observation-to-evaluation pipeline while preserving
the intended scientific controls:

- two $M$-state cyclic factors with
  $P(z'=g_z)=q$ and uniform residual mass $(1-q)/(M-1)$;
- `hand in {left, right, stay}`, with `x` action-sensitive and true `y`
  action-invariant;
- independent action-conditioned beliefs for all $3M$ contexts of each
  factor, initialized with symmetric concentration $\alpha=1$ per next state;
- budgets $N_x$ and $N_y$ interpreted as updates to every local
  `(state, action)` context;
- nested matched observation streams and factor-specific posterior-model
  streams;
- exact finite-horizon feedback planning under one posterior-sampled model;
- one recorded binary true-environment terminal outcome per task, cell, and
  trial, with navigation rollouts reused across $N_y$ and synchronization
  evaluated separately for each $(N_x,N_y)$ pair;
- exact navigation invariance across $N_y$ as a negative-control contract;
- complete versioned trial, summary, seed, and manifest artifacts; and
- exactly two heatmaps, using caller-selectable `viridis` and `plasma`
  defaults, fixed color limits $[0,1]$, labeled axes and colorbars, and
  uncertainty retained in the summary table rather than encoded as extra
  heatmaps.

The primary scientific estimand for a future substantive run is the cell mean
$k/T$ of independent top-level binary trial outcomes, with Wilson 95%
binomial intervals. Higher success is better. Allocation comparisons are
valid only on configured diagonals $N=N_x+N_y$; the full rectangle is a
sample-size response surface.

For this implementation milestone, success meant exact contract tests,
fresh-kernel notebook execution, artifact round trips and manifest
verification, and semantic figure inspection. The bounded $T=8$
demonstration was not a powered experiment and had no confirmatory scientific
success criterion.

### Allocation-sensitivity extension

The implementation acceptance criteria were:

- accept distinct positive even totals represented by both task summaries;
- select exactly \((N/2,N/2)\) for equal allocation;
- define below the heatmap diagonal as \(N_x>N_y\) and above as
  \(N_x<N_y\);
- maximize observed \(k/T\) independently within each side, task, and total;
- break exact observed-mean ties by smallest \(|N_x-N_y|\), then ascending
  \((N_x,N_y)\);
- preserve the selected cell's counts and uncertainty fields without pooling
  or recomputation;
- return deterministic task, total, and strategy order;
- render two task panels with three labeled lines, explicit units,
  \([0,1]\) limits, exact total-budget ticks, accessible redundant encodings,
  pointwise Wilson 95% error bars, and selected-allocation annotations;
- reject incomplete, duplicate, malformed, geometrically invalid, or
  statistically inconsistent records before plotting;
- identify the two best-side lines and intervals as post-hoc and
  non-selection-adjusted in the notebook, documentation, and figure caption;
- execute the \(M=6,H=30,T=100\) notebook from a fresh kernel, round-trip
  5,000 raw records and 50 summaries, verify the final artifact manifest, and
  visually inspect both PNG figures; and
- cache each transition row at most once per used state/action pair in one
  planning or evaluation call while preserving exact outputs and validation
  order.

The primary metric remained binary terminal success \(k/T\), higher is better.
The exploratory expectation was that navigation would respond to \(N_x\) and
remain exactly invariant to \(N_y\) within matched trials, while
synchronization could respond to both factors. No confirmatory superiority
criterion was declared, and post-hoc side maxima cannot test that expectation.

## Setup

### Shared repository and environment

- Repository: `/data/user_data/nquazi/repos/evolving-world-models`
- Base revision:
  `ce801b3b6ca1c724e4ca20b08ce19d90c2a65068`
- Worktree: dirty throughout these prospective milestones
- Platform: Linux `5.14.0-503.40.1.el9_5.x86_64`, x86-64
- CPU: AMD EPYC 7763 64-Core Processor; 128 logical CPUs visible
- Primary environment: Conda `world_models`
- Primary Python: CPython 3.12.13, Anaconda build, GCC 14.3.0
- Compatibility Python: CPython 3.9.21, GCC 11.5.0
- Jupyter 1.1.1; nbconvert 7.17.1
- Matplotlib 3.11.1; NetworkX 3.6.1
- Ruff 0.15.22; mypy 1.19.1; pre-commit 4.3.0
- Accelerators, external services, datasets, data splits, preprocessing,
  hyperparameter searches, and checkpoints: not used

Fresh notebook runs used isolated writable Matplotlib, IPython, and Jupyter
directories under `/tmp`. Jupyter required local loopback sockets. No input
data license or checksum applies because all transition observations were
generated from the fully specified in-repository finite model.

### Lock/door milestone

No seed was used. The final notebook code-cell interval was about 1.24 seconds;
the first cold-cache execution took longer because it initialized isolated
caches.

### Belief milestone

Belief sampling tests used `random.Random` seeds `2026`, `2027`, and `90210`.
The documented examples used seed `20260723`. No seed was selected based on
its outcome. Tests were deterministic and offline. The full effort took about
32 minutes of wall time, including review, documentation, notebook execution,
and recovery from environment and packaging-validation failures. The focused
tests themselves took less than 0.1 seconds per interpreter.

### Two-clock bounded demonstration

The visible notebook configuration was:

```text
M=3, q=0.80, predictable_direction=right, H=2, T=8
N_x=(0, 1, 2), N_y=(0, 1, 2), fixed_total_budgets=(2,)
alpha=1.0, master_seed=20260723
navigation_colormap=viridis, synchronization_colormap=plasma
```

The independent experimental unit was one top-level trial group. Within a
group, navigation performed one rollout for each $N_x$ and recorded the matched
outcome across all $N_y$ cells; synchronization performed one rollout for each
$(N_x,N_y)$ pair. The runner recorded 144 binary task/cell evaluation records
and 18 cell summaries. The final experiment runner reported 0.556 seconds of
elapsed wall time; this is not a general benchmark.

The successful fresh execution used a temporary notebook copy under
`/tmp/two-clock-final-notebook.uxSDlU`. Its run directory was:

```text
/tmp/two-clock-final-notebook.uxSDlU/results/two-clock/
two-clock-20260723T213913748853Z-75d60366814d
```

The run was intentionally ephemeral engineering evidence, not a durable
scientific result. Its exact derived seed namespaces and values were recorded
in the generated `seeds.csv`; the master seed and deterministic SHA-256
derivation method are part of the repository notebook/configuration contract.

### Expanded allocation-sensitivity run

The visible scientific configuration was:

```text
M=6, q=0.80, predictable_direction=right, H=30, T=100
N_x=(0, 1, 2, 3, 4), N_y=(0, 1, 2, 3, 4)
fixed_total_budgets=(2, 4, 6)
alpha=1.0, master_seed=20260723
navigation_colormap=viridis, synchronization_colormap=plasma
```

Each of the 25 allocation cells contained 100 independent top-level trial
groups, producing 5,000 binary task/cell evaluation records and 50 summaries.
Within each group, the five navigation rollouts—one per $N_x$—were reused
across $N_y$, while synchronization used 25 rollouts—one per $(N_x,N_y)$.
Random streams and task instances were matched across cells as documented in
Methods. There was no hyperparameter search, checkpoint selection, excluded
run, dataset, or train/test split.

The fresh notebook cells executed from
`2026-07-23T22:06:17.790965Z` through
`2026-07-23T22:20:26.663518Z`, 848.873 seconds wall time. The experiment
runner itself reported 846.696 seconds. The run identifier and retained
repository-local, Git-ignored directory are:

```text
results/two-clock/two-clock-20260723T222025252737Z-69ce6650bd38
```

The environment, CPU, Python, Matplotlib, and Jupyter versions were the shared
versions recorded above. The run used the Conda `world_models` interpreter and
isolated Jupyter, IPython, Matplotlib, and runtime state under
`/tmp/two-clock-sensitivity.idyo2J`. Jupyter local sockets required the
previously documented explicit approval.

For performance diagnosis, an unprofiled \(T=1\) run of the same
\(M,H,N_x,N_y\) configuration took 30.741 seconds before row caching and
7.953 seconds afterward. A `cProfile` run took 29.307 seconds because of
instrumentation overhead and recorded 20.306 cumulative seconds in 30
planning calls and 8.825 cumulative seconds in 30 exact policy evaluations;
belief/model learning was below the top 35 cumulative functions. Profiled
absolute time is not a production runtime estimate.

## Replication instructions

Activate the documented development environment from the repository root:

```bash
conda activate world_models
python -m pip install -e '.[dev]'
```

Run the baseline quality checks:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
python -m json.tool \
  notebooks/Two_Clock_Allocation_Experiment.ipynb > /dev/null
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Execute the lock/door notebook from a fresh kernel:

```bash
notebook_run_dir="$(mktemp -d /tmp/ewm-final-notebook.XXXXXX)"
mkdir -p \
  "$notebook_run_dir/matplotlib" \
  "$notebook_run_dir/ipython" \
  "$notebook_run_dir/jupyter-config" \
  "$notebook_run_dir/jupyter-data" \
  "$notebook_run_dir/jupyter-runtime"
MPLCONFIGDIR="$notebook_run_dir/matplotlib" \
IPYTHONDIR="$notebook_run_dir/ipython" \
JUPYTER_CONFIG_DIR="$notebook_run_dir/jupyter-config" \
JUPYTER_DATA_DIR="$notebook_run_dir/jupyter-data" \
JUPYTER_RUNTIME_DIR="$notebook_run_dir/jupyter-runtime" \
python -m jupyter nbconvert \
  --execute \
  --to notebook \
  --output Factored_MDP_Demo.executed.ipynb \
  --output-dir="$notebook_run_dir" \
  --ExecutePreprocessor.timeout=120 \
  notebooks/Factored_MDP_Demo.ipynb
```

Expected lock/door outputs are:

```text
Joint variables: ('lock', 'door')
External parents: ('action',)
0.05 x 0.90 = 0.045
All 10 local rows and 8 joint rows sum to one.
open  -> P(next lock=unlocked)=0.95, P(next door=open)=0.90
close -> P(next lock=unlocked)=0.95, P(next door=open)=0.00
```

Execute the clock notebook the same way, with isolated writable runtime
directories, a 1,200-second per-cell timeout, a noninteractive plotting backend, and a
unique temporary working directory:

```bash
clock_run_dir="$(mktemp -d)"
mkdir -p \
  "$clock_run_dir/matplotlib" \
  "$clock_run_dir/ipython" \
  "$clock_run_dir/jupyter-config" \
  "$clock_run_dir/jupyter-data" \
  "$clock_run_dir/jupyter-runtime"
cp notebooks/Two_Clock_Allocation_Experiment.ipynb "$clock_run_dir/"
(
  cd "$clock_run_dir"
  MPLBACKEND=Agg \
  MPLCONFIGDIR="$clock_run_dir/matplotlib" \
  IPYTHONDIR="$clock_run_dir/ipython" \
  JUPYTER_CONFIG_DIR="$clock_run_dir/jupyter-config" \
  JUPYTER_DATA_DIR="$clock_run_dir/jupyter-data" \
  JUPYTER_RUNTIME_DIR="$clock_run_dir/jupyter-runtime" \
  python -m jupyter nbconvert \
    --execute \
    --to notebook \
    --output Two_Clock_Allocation_Experiment.executed.ipynb \
    --ExecutePreprocessor.timeout=1200 \
    Two_Clock_Allocation_Experiment.ipynb
)
```

The notebook creates a unique run under
`$clock_run_dir/results/two-clock/`, reloads its machine-readable tables,
generates PDF and 300-dpi PNG figures from the saved summary, refreshes the
manifest, and verifies it. Because this command runs from the temporary working
directory, that run is ephemeral and outside the checkout; it neither recreates
nor modifies the retained repository-local run recorded in Setup. Before
relying on an empirical figure as durable evidence, copy the complete verified
run to external durable storage and record its URI, manifest or checksums,
access requirements, and retention policy. The retained run recorded in Setup
remains local and is not recreated by this command.

For a clean distribution check, ensure ignored `build/` and
`evolving_world_models.egg-info/` artifacts are absent from the source used
for the build, then run:

```bash
wheel_dir="$(mktemp -d /tmp/ewm-clean-wheel.XXXXXX)"
clean_env="$(mktemp -d /tmp/ewm-clean-env.XXXXXX)"
python -m pip wheel --no-cache-dir --no-deps \
  --wheel-dir "$wheel_dir" .
python -m venv "$clean_env"
"$clean_env/bin/python" -m pip install --no-deps \
  "$wheel_dir"/evolving_world_models-*.whl
(
  cd /tmp
  "$clean_env/bin/python" -c \
    'import importlib.util, sys; from scripts import AbstractTransitionBelief, FactoredDirichletBelief, TabularDirichletBelief, Variable; belief = TabularDirichletBelief((Variable("state", (0, 1)),), prior=1.0); assert belief.transition_probability({"state": 1}, {"state": 0}) == 0.5; assert importlib.util.find_spec("scripts.visualization") is not None; assert not {"matplotlib", "networkx", "graphviz"}.intersection(sys.modules)'
)
```

Temporary wheels, virtual environments, executed notebooks, and the bounded
clock run are validation artifacts rather than durable scientific outputs.

## Methods

### Lock/door transition model

The lock factor has two `TabularMDP` rows. Each assigns 0.95 to retaining the
current lock setting and 0.05 to switching. The door factor has one row for
each current-door, current-lock, and action context:

| Current door | Current lock | Action | $P(d'=\mathrm{open})$ |
| --- | --- | --- | ---: |
| closed | locked | open | 0.0 |
| closed | unlocked | open | 0.9 |
| open | either | open | 1.0 |
| closed | either | close | 0.0 |
| open | either | close | 0.1 |

`FactoredMDP` projects the current joint lock into the door factor and
multiplies the local distributions. The table intentionally does not impose a
constraint between latch setting and door position: `locked` describes the
latch mechanism and remains meaningful while the door is open.

The notebook builds the public library objects, asserts the exact product and
normalization contracts, compares both actions, and renders the deterministic
two-slice factor graph. Reusable behavior remains in `scripts/`; the notebook
contains exposition and contract assertions only.

### Dirichlet transition beliefs

For every complete current/parent context $c$ and joint next-state outcome
$y$, the tabular belief stores a strictly positive prior concentration
$\alpha_{c,y}$ separately from an integer observation count $n_{c,y}$:

$$
p(x'=y\mid c,D)
=
\frac{\alpha_{c,y}+n_{c,y}}
{\sum_z(\alpha_{c,z}+n_{c,z})}.
$$

Contexts and outcomes are enumerated in declared variable and domain order. A
scalar prior applies symmetrically to all rows. An explicit prior must cover
every context and Cartesian next outcome exactly once.

`posterior_mean_mdp()` materializes a fresh complete `TabularMDP`. Sampling
draws $g_y\sim\mathrm{Gamma}(\alpha_{c,y}+n_{c,y},1)$ in canonical order and
normalizes with `math.fsum`. An injected `random.Random`-compatible generator
controls reproducibility; module-global randomness is not used.

Predictive distributions retain
$\log(\alpha_{c,y}+n_{c,y})-\log\sum_z(\alpha_{c,z}+n_{c,z})$ separately from
their normalized float rows. This preserves a finite direct-belief log query
when an extreme positive ordinary probability underflows to zero. Fixed
`TabularMDP` snapshots retain only normalized floats and cannot preserve that
additional log-space information.

A factored belief retains tabular factor beliefs with disjoint outputs. Joint
prediction multiplies local posterior-predictive rows. Joint update first
validates the complete observation and every local target, then increments
each factor once. Internal parents come from the joint current state; only
unpredicted parents are external.

No dataset, empirical baseline, statistical aggregation, exclusion rule,
inference test, or multiple-comparison procedure applied to this milestone.

### Two-clock learning and evaluation

Let

$$
x_t,y_t\in\mathbb Z_M,\qquad
a_t\in\{\mathrm{left},\mathrm{right},\mathrm{stay}\}.
$$

The controllable intended successor is $x-1$, $x+1$, or $x$ modulo $M$ for
`left`, `right`, or `stay`. The predictable intended successor is the
configured fixed direction, here $y+1$ modulo $M$. For either factor $z$:

$$
P(z'=g_z)=q,\qquad
P(z'=u)=\frac{1-q}{M-1}\quad (u\ne g_z).
$$

The true predictable rows are identical across actions, but its belief keeps a
separate row for every `(y, hand)` context. This makes both factors comparable
at $3M$ belief contexts without pooling action invariance, translation
symmetry, or the common $q$.

Every `(x, hand)` row receives exactly $N_x$ observations and every
`(y, hand)` row exactly $N_y$. Thus one cell and trial represent
$3M(N_x+N_y)$ local updates. The leaf beliefs are updated separately; using a
joint factored update would incorrectly spend budget on both factors at once.

Stable named seeds are derived with SHA-256 from canonical JSON namespaces.
Each factor-local context generates its largest requested observation stream
once, and smaller budgets use prefixes. An `x` model seed depends on its trial
and $N_x$ but not $N_y$; the symmetric rule holds for `y`. Initial states,
navigation targets, and true-rollout streams are matched across cells.

Each trial group draws $x_0$, $y_0$, and the navigation target independently
and uniformly. For exactly $H$ transitions, dynamic programming computes:

$$
V_H(s)=R_H(s),\qquad
Q_t(s,a)=\sum_{s'}\widehat P(s'\mid s,a)V_{t+1}(s'),\qquad
V_t(s)=\max_a Q_t(s,a).
$$

Exact maximizing ties use an injected named random stream. Navigation uses the
mathematically equivalent `x`-marginal policy and terminal reward
$\mathbf 1[x=g]$; synchronization uses the joint `(x,y)` model and
$\mathbf 1[x=y]$. Each trial evaluates one navigation rollout per $N_x$ and
reuses its matched outcome across $N_y$; it evaluates synchronization once per
$(N_x,N_y)$ pair. There is no online belief update. Summaries report $k/T$,
Monte Carlo standard error, and Wilson 95% bounds.

The artifact layer uses explicit versioned schemas for configuration, trials,
summaries, seeds, and manifests. It creates a unique run directory, refuses to
overwrite an existing run, uses atomic replacement for individual UTF-8
JSON/CSV files, rejects unsafe identifiers and symlinks, and verifies complete
manifest coverage, sizes, schema versions, and SHA-256 digests. A failed
multi-file run may leave a partial unique directory; recovery uses a new run
identifier rather than silently resuming or overwriting it.

Plotting reloads saved summaries and creates exactly two discrete,
unsmoothed heatmaps. Both use fixed limits $[0,1]$, explicit budget ticks,
labeled axes in updates per context, and separate colorbars labeled
`Estimated success probability (unitless)`. Colormaps are presentation
variables and default to `viridis` for navigation and `plasma` for
synchronization.

For every requested even total \(N\), the allocation selector partitions the
fixed-total cells into equal, below-diagonal, and above-diagonal regions:

$$
\mathcal E_N=\{(N/2,N/2)\},\qquad
\mathcal B_N=\{(N_x,N_y):N_x+N_y=N,\ N_x>N_y\},
$$

$$
\mathcal A_N=\{(N_x,N_y):N_x+N_y=N,\ N_x<N_y\}.
$$

It retains the unique equal cell and selects the maximum observed \(k/T\)
within \(\mathcal B_N\) and \(\mathcal A_N\). Ties minimize
\(|N_x-N_y|\) and then use ascending coordinates. The sensitivity plot uses
the original selected-cell Wilson 95% bounds as asymmetric error bars. No
selection adjustment, resampling, smoothing, pooling, or interpolation is
performed.

`plan_finite_horizon` and `evaluate_finite_horizon_policy` now keep a local
dictionary from canonical `(state, action)` to one validated enumerated
transition row. Planning lazily populates every row on first deterministic
visit; policy evaluation populates only rows selected by the policy. The cache
is discarded at return, so it cannot survive a model mutation or contaminate
another sampled model. Backward-induction arithmetic and exact-tie random
draws remain in their original order.

## Validation and quality checks

### Lock/door milestone

The focused transition and visualization suite passed all 51 tests. The final
full repository suite at that stage, including the concurrent belief
milestone, passed all 81 tests. Ruff, formatting, strict mypy, compilation,
notebook JSON structure, pre-commit, and the corrected pre-push hook run
passed.

The final fresh execution had code-cell counts 1 through 9 and zero error
outputs. Its output reported the expected joint and external scopes, exact
0.045 product, all normalization counts, and action comparison.

After removing only per-cell execution timing metadata, the stored notebook
and independent final output had the same SHA-256:
`909dc495ad5a771426bd248fe5b8f1e3ed6e490afcfcaa545c6ce50242466b9f`.
The embedded final PNG was 1590 by 1090 pixels, 84,769 bytes, with SHA-256
`13f67f3d76dcbd4af9fa99c059d220af2227e7a63d4e687bc9a992b9e281123c`.

The image was visually inspected at its final size. Labels were legible,
nodes and factor markers did not overlap, and no content was clipped. The
independent final execution produced the identical PNG hash.

### Belief milestone

The 30 focused belief tests covered:

- abstractness, metadata, and package-root exports;
- symmetric and explicit asymmetric priors;
- complete Cartesian support for multi-variable outcomes;
- exact conjugate updates and context isolation;
- malformed scopes, assignments, priors, concentrations, and RNGs;
- atomic failures and defensive copies;
- mean-snapshot independence;
- sampled-row normalization, seeded reproducibility, and fresh samples;
- preserved concentration-space log mass under normalized-float underflow;
- deterministic descriptions independent of input row order;
- factored metadata, external-parent ordering, products, and snapshots;
- current-time cross-factor projection and the lock/door regression; and
- import without visualization or numerical dependencies.

The same 30 tests passed on Python 3.9.21 and 3.12.13. The full Python 3.12
suite at that stage passed 81 tests. Ruff, Ruff formatting, strict mypy,
compilation, notebook JSON structure, fresh notebook execution, both hook
stages, runnable README/reference examples, and clean wheel
installation/import also passed.

The final clean wheel contained `scripts/__init__.py`, `scripts/mdp.py`,
`scripts/beliefs.py`, and `scripts/visualization.py`. It was 35,734 bytes with
SHA-256
`6c6a4e5f8784ddf661964473bdf5af158117a8563b242341a01e4d33f14ccbc8`.

### Two-clock milestone

The focused suites passed 13 planning tests, 16 clock-experiment tests, and
four headless plotting tests. They cover exact backward induction, horizon
zero, feedback tables, exact tie behavior, validation and true-model policy
evaluation; true clock kernels, budget accounting, nested prefixes,
cross-axis factor reuse, deterministic reproduction, negative controls,
aggregation, Wilson intervals, schemas, round trips, manifests, corruption
and symlink rejection, exact trial-field-to-seed-namespace validation; and
semantic heatmap structure without pixel snapshots.

An earlier complete Python 3.12 suite passed 114 tests in 0.935 seconds. The
final source-state rerun passed the same 114 tests in 0.687 seconds. These
later counts supersede neither the 51-test focused lock/door result nor the
30-test focused and 81-test repository results recorded at their earlier
milestones.

The new notebook executed all nine code cells sequentially with no error
outputs. Exhaustive kernel and normalization assertions, exact per-context
budget accounting, artifact round trips, navigation invariance across $N_y$,
and final manifest verification passed. The PNG was visually inspected and
had legible titles, axes, discrete cells, and separate fixed-range colorbars;
no clipping was observed.

Because NumPy 2.4's installed stubs use Python 3.12 type-alias syntax and
Matplotlib source contains pattern matching that cannot be parsed under
mypy's configured Python 3.9 target, `pyproject.toml` now applies a narrow
`follow_imports = "skip"` override to `numpy`, `numpy.*`, `matplotlib`, and
`matplotlib.*`. Strict mypy continues to check all seven repository production
modules and passed after the override.

### Allocation-sensitivity extension

The focused source-state suites passed 14 planning tests, 22
clock-experiment tests, and nine headless plotting tests. New coverage includes
one transition-row query per used state/action within an invocation,
equal/below/above selection, deterministic tie and output order, positive-even
total validation, incomplete-region rejection, immutable retained statistics,
package-root exports, exact Wilson error-bar geometry, annotations, colors,
caller axes, and presentation-boundary validation.

The complete Python 3.12 suite passed 126 tests in 0.750 seconds. The
dependency-free Python 3.9 core suite passed 89 tests in 0.397 seconds. Ruff
lint, Ruff formatting, strict mypy over all seven production modules,
compilation, both notebook JSON checks, `git diff --check`, pre-commit
configuration validation, and both complete hook stages passed.

The expanded notebook executed nine code cells with counts 1 through 9,
11 intentional outputs, and no errors. It verified 36 true local rows,
round-tripped 5,000 trials and 50 summaries, established exact navigation
invariance across \(N_y\), selected 18 complete strategy points, and verified
all nine run files. The normalized saved notebook has no per-cell execution
timing metadata or absolute temporary paths and embeds the exact generated
heatmap and sensitivity PNGs.

Both PNGs were inspected at original size. The heatmap was 3032 by 1294
pixels; the sensitivity figure was 3034 by 1294 pixels. Titles, axes,
abbreviated units, ticks, colorbars or legends, color/marker/line-style
redundancy, Wilson intervals, and allocation annotations were legible and
unclipped.

The final package check reported no broken requirements, built a 69,159-byte
wheel with SHA-256
`f004013f764d9f6f58066d98448dcc69c3e78e400b684c98ff12e39066e6a139`,
installed it without dependencies in a clean virtual environment, changed
outside the checkout, imported the allocation selector from `site-packages`,
and confirmed that Matplotlib was absent.

GitHub Actions was not run because no commit or push was authorized.

## Results

### Lock/door milestone

All mathematical, state-space, synchronous-projection, graph-structure, and
presentation acceptance criteria passed. The active model has joint variables
`('lock', 'door')` and only `('action',)` as its external parent.

From current `lock=unlocked, door=closed`:

- both actions leave the next-lock-unlocked marginal at 0.95;
- `open` gives next-door-open probability 0.90;
- `close` gives next-door-open probability 0.00; and
- next `lock=locked, door=open` under `open` has exact probability 0.045.

The active notebook, README, and visualization reference contain no stale
weather/umbrella terminology. Small weather-named primitive test fixtures
remain in `tests/test_mdp.py` because they test generic variable and
assignment behavior, not the canonical example.

### Belief milestone

With a symmetric binary prior and one matching observation, the updated local
outcome has predictive probability $2/3$ while an untouched row remains
$1/2$. In the critical factored observation, both matching local outcomes have
probability $2/3$, giving the exact joint product $4/9$. Inspection confirmed
that the door row conditioned on current `lock=unlocked` received one count
and the current-locked row received zero.

Seeded samples reproduced exactly for equal seeds, differed for the tested
distinct seed, and every sampled row normalized to one. Mean and sampled MDP
snapshots remained unchanged after later belief updates. These are exact
contract results; statistical uncertainty is not applicable.

For posterior concentrations `5e-324` and `1e308`, the small ordinary
probability underflowed to `0.0`, while the direct belief log probability was
the finite value `-1453.636280563547`. The regression verifies that a fixed
posterior-mean `TabularMDP` retains the documented float underflow and returns
`-inf`, so numerically extreme comparisons must query the belief in log space.

The fresh lock/door notebook retained sequential execution counts 1 through 9,
zero error outputs, and the expected values. Its final code cells executed
from 2026-07-23T14:24:45.274878Z to
2026-07-23T14:24:46.522533Z. Ignoring execution timing metadata, the
temporary output and stored notebook had identical SHA-256
`909dc495ad5a771426bd248fe5b8f1e3ed6e490afcfcaa545c6ce50242466b9f`.

No learning-capacity or downstream-task result was produced by the belief
milestone.

### Two-clock milestone

The bounded demonstration generated
$2\times8\times3\times3=144$ binary records and 18 summary rows. It
round-tripped the recorded trials and summaries, retained a deterministic seed
ledger, plotted the reloaded summaries, and verified the refreshed manifest.

The ephemeral artifacts and byte sizes were:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 518 | `75d60366814d0732c5b8f82334fdd0ad895f877734d7dc9d042423bbef2defc2` |
| `trials.csv` | 36,644 | `3ea9d467d024670f6248a6b713fd8151c57dd528c15171f8a901912f3ee53f58` |
| `summary.csv` | 2,026 | `8a3c89ae96c37b1793b83ff246d58c320294258cf0cabf0e2b79f5264c0c936a` |
| `seeds.csv` | 27,168 | `6ba7f2fb6a1cb2f07428b2955c8c385d0c11741a3adf1ed8c2d3fa55facae055` |
| `manifest.json` | 1,252 | `8a1d55a185731bfd0d6820949b5a910050feb89b4716492d306edc00206d756b` |
| `figures/two-clock-success.pdf` | 21,704 | `98970857dcab838366af57836312a21724dc495f4a27f821ec776e85c850d604` |
| `figures/two-clock-success.png` | 217,732 | `24ded8e7885e2ab2e07aa16a152b8f62d0e2f878975a912107f5a9851dee3cc3` |

These measurements demonstrate that the requested workflow and artifact
contract execute coherently. With only $T=8$ matched trial groups per cell,
the heatmap values are too noisy for claims about allocation, monotonicity,
causality, or task performance. No seed, cell, or outcome was selected after
inspection, and no confirmatory experiment was run.

### Expanded allocation-sensitivity run

The exact exploratory run produced 5,000 raw binary evaluations, 50 cell
summaries, and 18 selected allocation-strategy records. The selected estimates
are shown below as `success [Wilson 95%]; (N_x,N_y)`.

| Task | \(N\) | Equal | Best below \(N_x>N_y\) | Best above \(N_x<N_y\) |
| --- | ---: | --- | --- | --- |
| Navigation | 2 | 0.32 [0.237, 0.417]; (1,1) | 0.42 [0.328, 0.518]; (2,0) | 0.15 [0.093, 0.233]; (0,2) |
| Navigation | 4 | 0.42 [0.328, 0.518]; (2,2) | 0.54 [0.443, 0.634]; (4,0) | 0.32 [0.237, 0.417]; (1,3) |
| Navigation | 6 | 0.51 [0.413, 0.606]; (3,3) | 0.54 [0.443, 0.634]; (4,2) | 0.42 [0.328, 0.518]; (2,4) |
| Synchronization | 2 | 0.19 [0.125, 0.278]; (1,1) | 0.16 [0.101, 0.244]; (2,0) | 0.13 [0.078, 0.210]; (0,2) |
| Synchronization | 4 | 0.21 [0.142, 0.300]; (2,2) | 0.21 [0.142, 0.300]; (3,1) | 0.23 [0.158, 0.322]; (1,3) |
| Synchronization | 6 | 0.24 [0.167, 0.332]; (3,3) | 0.18 [0.117, 0.267]; (4,2) | 0.28 [0.201, 0.375]; (2,4) |

Navigation's selected coordinates and estimates reflect its exact
\(N_y\)-invariance: for a fixed \(N_x\), all \(N_y\) cells reuse the same
policy and paired outcome. The table does not establish that below-diagonal
allocation is generally optimal. Similarly, the synchronization above-side
maxima at \(N=4\) and \(N=6\) are observations from this run, not evidence
that an independently evaluated policy will generalize.

The local-only retained artifacts are:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 564 | `69ce6650bd38877d118b85f73b8c7e3cdfc668d974a9b5f1ad43515fade5b915` |
| `trials.csv` | 1,296,080 | `6285fa5b4a7c0cd1cdddbdde2fe05e0d61270740d79629ecba84259bb5538ddd` |
| `summary.csv` | 5,560 | `f31cca32656c600442245da1816c1c03d628b762c5726ce873adb3056ab6100f` |
| `seeds.csv` | 666,575 | `a9b2ef920d43616786b5037ae500bd2d3fbb778509426beaf6fc8b2186121b39` |
| `manifest.json` | 1,596 | `5cdef8ca59a7aecb6937708cc6528e87ab20e0a85b4c8d83e705f6d1619affc0` |
| `figures/two-clock-success.pdf` | 22,394 | `057cbfda6fb9ab1deb995c3a4e2ded20c81704ae1a807322a176d36f7b73010e` |
| `figures/two-clock-success.png` | 225,740 | `db5e36eb9e59be7315d068c5c1e04a8d41c51a77da21b9e7f283a8974aa97821` |
| `figures/allocation-sensitivity.pdf` | 24,234 | `6f432c02a5b6059bfe9c091364eae3bc48d167be794f1d4ab79f477f67679477` |
| `figures/allocation-sensitivity.png` | 316,059 | `4a04e65951d3c37a76d0bdb08a3225cf26706416c46c1a9d6662c754864dfeea` |

`verify_run_manifest` passed again after the full directory was copied from
the isolated execution root into the repository-local ignored results tree.
Sampling uncertainty applies to every cell. The side maxima incur additional
unquantified selection uncertainty, so their ordinary Wilson intervals must
not be interpreted as coverage for an unknown optimal allocation.

## Figures and tables

### Lock/door structural figure

**Figure 1. Lock/door transition structure.** Blue circles encode current
variables, the orange diamond encodes the external action parent, gray squares
encode transition factors, and green circles encode next variables. Solid
arrows encode state/factor or prediction links; the dashed arrow encodes
external-parent conditioning. Labels show time slices and finite domains. Axes
are omitted because the diagram's coordinates have no scientific meaning. The
figure is an exact deterministic structural diagram, so units, sample size,
aggregation, and uncertainty are not applicable.

The door-probability table in Methods is the complete compact representation
of the eight rows; “either” denotes two exact lock-specific rows with the same
value.

### Belief milestone

No belief-specific figure or quantitative result table was created. The
lock/door structural figure belongs to the example milestone. Axes, units,
sample sizes, aggregation, and uncertainty are not applicable.

### Two-clock heatmaps and summary table

**Figure 2. Two-clock terminal success, bounded demonstration.** Each cell is
the fraction of $T=8$ independent matched trial groups whose recorded
evaluation ended in the task's binary terminal goal. Navigation outcomes are
matched across $N_y$; synchronization has one rollout per $(N_x,N_y)$ pair.
The panels show target navigation (`viridis`) and clock synchronization
(`plasma`) for
$M=3$, $q=0.80$, predictable direction right, $H=2$, $\alpha=1$, master seed
`20260723`, and $N_x,N_y\in\{0,1,2\}$ updates per factor-local
`(state, action)` context. Both unitless color scales span zero to one and
higher is better. Observations use nested prefixes; `summary.csv` retains
Monte Carlo standard errors and Wilson 95% binomial intervals. This small-$T$
figure is engineering validation, not a confirmatory result.

The PNG was visually inspected. The PDF was generated and included in the
verified manifest but was not separately visually inspected. The summary CSV,
rather than extra heatmaps, is the authoritative uncertainty table.

### Expanded heatmaps and allocation sensitivity

**Figure 3. Two-clock terminal success, expanded exploratory run.** Each cell
is the fraction of \(T=100\) matched top-level trial groups whose recorded
evaluation ended in the binary goal. Navigation outcomes are matched across
\(N_y\); synchronization has one rollout per \((N_x,N_y)\) pair. The panels
show target navigation (`viridis`) and clock synchronization (`plasma`) for \(M=6\),
\(q=0.80\), predictable direction right, \(H=30\), \(\alpha=1\), master
seed `20260723`, and \(N_x,N_y\in\{0,1,2,3,4\}\) updates per local
`(state, action)` context. Both unitless color scales span zero to one and
higher is better. The PNG is 3032 by 1294 pixels; axes, units, panel titles,
ticks, and colorbars were visually inspected and were legible and unclipped.
The saved summary table is the authoritative source for uncertainty.

**Figure 4. Allocation sensitivity to total observation budget.** For each
task and \(N\in\{2,4,6\}\) updates per context, the figure compares equal
allocation with the highest observed success rate below the diagonal
\((N_x>N_y)\) and above the diagonal \((N_x<N_y)\). Each point aggregates
\(T=100\) matched trial groups under the same conditions as Figure 3; higher
unitless success probability is better. Point labels report the selected
\((N_x,N_y)\), and error bars are pointwise Wilson 95% binomial intervals.
The best-side lines are post-hoc oracle envelopes whose intervals do not
account for selection. The PNG is 3034 by 1294 pixels; titles, axes, units,
redundant colors, markers, line styles, and legend labels, intervals, and
annotations were visually inspected and were legible and unclipped.

Both PDFs were generated by the notebook and covered by the verified manifest
but were not separately rendered for visual inspection. The exact PNG bytes
were embedded into the saved notebook outputs after the headless `Agg`
execution emitted only textual figure representations; no pixel or plotted
value was modified.

## Conclusions

### Observations supported by evidence

- The lock/door example satisfies the exact synchronous factor contract and
  makes the current-parent rule directly testable when current and next lock
  settings differ.
- The finite Dirichlet implementation satisfies the specified tabular and
  factored contracts on both supported Python versions tested locally,
  preserves the fixed-MDP boundary, and correctly projects synchronous
  internal parents.
- The two-clock planning, budget-allocation, artifact, and plotting layers
  pass their focused contracts and run end to end in a bounded fresh-kernel
  demonstration.
- The expanded notebook run validates the selector, four-figure artifact
  workflow, manifest, and exact matched navigation negative control at
  \(M=6,H=30,T=100\).
- Reusing sampled factor models across tasks and the opposite budget axis was
  already present. Profiling identifies dynamic programming and exact policy
  evaluation, rather than belief generation, as the dominant runtime cost.
- Caching validated transition rows once per used state/action pair reduces
  the measured one-trial runtime by approximately 74% for the expanded
  configuration while retaining exact contract tests.

### Bounded interpretation

The lock/door world and conjugate belief layer provide a coherent foundation
for finite-budget learning research. The two-clock implementation now makes
the update-allocation question executable and auditable. None of these
engineering results establishes a performance advantage, an optimal
allocation, or any causal relationship between learned factor and task
success. In particular, the observed best-side curves are optimistic
post-selection summaries, not estimates of prespecified allocation policies.

## Limitations and threats to validity

### Lock/door and belief foundations

- Lock/door probabilities are illustrative, not calibrated from physical-door
  data.
- The lock/door state space is an unconstrained Cartesian product and permits
  `lock=locked, door=open`.
- Beliefs support only finite, fully enumerated domains and complete transition
  support.
- A multi-variable tabular factor uses one joint Dirichlet row; sparse and
  constrained supports are unavailable.
- Forgetting, weighted observations, hierarchical priors, continuous
  variables, and durable belief serialization are not implemented.
- Factored composition accepts built-in tabular belief factors only.
- Seeded gamma sampling also depends on Python implementation and traversal
  order.
- Ordinary probabilities, sampled rows, and fixed MDP snapshots use finite
  floats. Extreme positive normalized masses can underflow to zero; only the
  direct belief log query retains concentration-space information.
- Dependency ranges do not constitute a complete environment lock.

### Two-clock experiment

- Independent rows do not pool the shared $q$, clock translation symmetry, or
  the predictable factor's true action invariance.
- The dense symmetric prior assigns positive mass to all next states.
- Balanced generative row sampling is not an online trajectory or physical
  interaction budget.
- Navigation performs one rollout per sampled controllable model and reuses the
  matched outcome across $N_y$; synchronization performs one rollout per
  sampled $(N_x,N_y)$ model pair. Both retain substantial binomial Monte Carlo
  uncertainty at small $T$.
- Exact dense synchronization planning scales as $O(H|A|M^4)$.
- Posterior-sampled planning does not guarantee pointwise monotonic
  performance.
- The bounded demonstration's $T=8$ was chosen for pipeline runtime, not power.
  Its results do not generalize to scientifically consequential settings.
- Raster output can vary with Matplotlib, FreeType, fonts, and platform; figure
  tests inspect semantic artists instead of hashes.
- Manifest generation times and PDF metadata make those files' hashes
  run-specific even when the saved measurements and semantic figure are
  unchanged; the final per-run hashes are recorded above.
- A full rectangular grid changes total evidence between cells. Allocation
  interpretation must be restricted to fixed-total diagonals.
- The best below- and above-diagonal series maximize noisy cell estimates on
  the same outcomes used for display. Their pointwise Wilson intervals ignore
  selection and multiple candidate comparisons.
- \(T=100\) narrows ordinary cellwise binomial uncertainty but does not remove
  winner's-curse bias, provide independent strategy-selection data, or turn
  the exploratory run into a preregistered comparison.
- Only three even totals and the allocations available inside the
  \(0,\ldots,4\) grid were compared. A reported side maximum is conditional on
  that finite candidate set.
- Trial groups are independent across `trial_id`, but cells and tasks are
  intentionally matched within a group. Pointwise intervals do not quantify
  paired differences between curves.
- The exact expanded runner took 846.696 seconds on the recorded CPU. The
  row cache removes repeated model queries but dense Python expectation
  arithmetic still dominates.
- CI remains unrun until a commit and push are explicitly authorized; local
  validation cannot substitute for CI.
- The retained expanded run exists only in ignored local storage and has no
  durable external URI or retention guarantee. Its hashes support auditing on
  the current machine, but the run must be archived externally before serving
  as durable publication evidence.

Most importantly, the belief milestone's explicit factored update increments
every factor. The two-clock orchestrator creates the allocation problem by
selectively updating leaf beliefs, but it does not yet model an online
interaction budget, learned representation bottleneck, or Bayes-adaptive
policy.

## Deviations, failures, and negative results

### Lock/door and belief milestones

- A bare `python` command failed in the non-activated shell. All substantive
  commands used the known Conda interpreter, and replication requires
  activating `world_models`.
- One convenience version probe tried `jupyter.__version__`, which does not
  exist. `importlib.metadata` subsequently reported Jupyter 1.1.1 and the
  other versions in Setup.
- The first isolated lock/door notebook displayed `0.09999999999999998` for
  one door row. The display expression was changed to query the intended open
  probability directly; no transition value changed.
- The first final lock/door notebook attempt failed because the managed
  sandbox blocks local sockets. The identical command succeeded with
  permission to open the fresh kernel's loopback sockets; no notebook code
  failed.
- The first cold-cache lock/door execution took longer because isolated
  Matplotlib and Jupyter caches were initialized.
- The first isolated wheel attempt could not download build dependencies under
  restricted network access. The authorized retry succeeded.
- Archive inspection found that the first wheel had reused an ignored stale
  `build/` tree and omitted `scripts/visualization.py`. A diagnostic rebuild
  refreshed the build tree. A final isolated build was then performed with
  ignored build metadata temporarily moved aside and restored afterward; its
  archive contained all four source modules and passed a clean install/import.
- The first pre-push-hook invocation used a non-activated `PATH`, so its
  `python` executable was absent. Rerunning with the Conda environment first
  on `PATH` passed unit tests and compilation.
- The first in-memory Markdown-fence validator was incorrectly shell-quoted,
  and its safe retry then exposed pseudocode in one design-note `python`
  fence. The snippet was made valid Python and all Python/Bash fences
  subsequently passed syntax checks.
- Independent numerical review exposed that the generic categorical log query
  treated an underflowed positive predictive mass as impossible. A private
  predictive distribution now retains concentration-space log masses, and a
  cross-version regression covers the extreme finite ratio. The first Ruff
  format check after this patch correctly reported two files requiring
  formatting; Ruff formatted them and the subsequent check passed.

### Two-clock milestone

- The first fresh clock-notebook attempt failed before kernel startup because
  `/home/nquazi/.jupyter` was read-only in the managed environment.
- After redirecting Jupyter, IPython, and Matplotlib state into `/tmp`, the
  next attempt reached kernel startup but failed because the sandbox blocked
  local sockets. The identical bounded execution succeeded after explicit
  loopback-socket approval; no notebook cell failed.
- Initial strict mypy analysis entered installed NumPy 2.4 stubs and
  Matplotlib sources that use syntax incompatible with the configured Python
  3.9 target. The narrow third-party import-following override described in
  Validation resolved that dependency-analysis incompatibility while
  retaining strict checks on repository modules.
- One early Ruff format check reported `tests/test_clock_experiment.py`;
  Ruff formatted it and the subsequent check passed.
- An earlier successful notebook validation reported a 0.748-second runner
  time. The notebook's exhaustive predictable-kernel assertion was then
  generalized to follow the configurable `PREDICTABLE_DIRECTION` rather than
  hardcoding the rightward successor. The notebook was re-executed again after
  exact seed-namespace validation and mathematical axis labels were finalized;
  the final source-state run reported 0.556 seconds.
- A bounded in-memory smoke run produced 16 trial rows, eight summaries, and
  70 seed records; CSV round trips and the manifest passed. It was diagnostic
  only and not used as scientific evidence.
- The fresh notebook run wrote its validation artifacts under `/tmp`.
  Consequently they are ephemeral and must not be cited as durable evidence
  for a later scientific claim.

### Allocation-sensitivity extension

- Clearing stale notebook outputs first failed because Jupyter attempted to
  write `/home/nquazi/.jupyter` in the managed read-only filesystem. A second
  command accidentally resolved `python` to `/usr/bin/python`, which lacked
  Jupyter. The successful command used the explicit Conda interpreter and
  redirected Jupyter/IPython state into `/tmp`.
- The first expanded fresh-kernel launch failed before cell execution because
  the sandbox denied the local sockets required by Jupyter. The identical
  command succeeded with explicit loopback-socket approval.
- One exact \(T=1\) benchmark took 30.741 seconds, projecting an impractical
  roughly 51-minute notebook. A measured, semantics-preserving transition-row
  cache reduced the same check to 7.953 seconds. The exact \(T=100\) runner
  subsequently took 846.696 seconds, so the notebook and CI per-cell timeout
  were raised from 300 to 1,200 seconds and the CI job timeout from 20 to 45
  minutes.
- `cProfile` expanded the one-trial diagnostic to 29.307 seconds and 121.8
  million calls. Its cumulative attribution is useful, but the absolute
  profiled duration is not comparable to unprofiled runtime.
- The forced `Agg` backend generated valid PNG/PDF files but stored only
  textual figure representations in the executed notebook. The exact
  code-generated PNG bytes were added to the corresponding `display_data`
  outputs, per-cell execution timing metadata was removed, and notebook source
  structure was verified byte-for-byte equivalent before the normalized
  executed notebook replaced the cleared source copy.
- The accepted run was initially isolated under `/tmp`. Because retaining an
  empirical figure without its raw records would violate provenance, the
  complete verified run directory was copied to a new, non-colliding sibling
  under `results/two-clock/`. That tree is now explicitly local, ignored
  working storage and is not intended for version control. The pre-existing
  user run
  `two-clock-20260723T214509858708Z-75d60366814d` was not modified.
- A final narrow hygiene command incorrectly supplied two hook identifiers to
  one `pre-commit run` invocation. The CLI rejected the arguments before
  running or modifying files; the two hooks were rerun separately and passed.
- A documentation-consolidation review found that rollout wording conflated
  recorded task/cell outcomes with distinct true-environment rollouts. The
  README, notebook prose and saved caption, module docstring, planning
  reference, conceptual allocation reference, and this log now state the exact
  reuse contract: one navigation rollout per $N_x$ and one synchronization
  rollout per $(N_x,N_y)$ within each top-level trial group.
- The same review found that all three sensitivity curves used circles and
  solid lines even though the log claimed redundant visual encodings. Equal,
  below-diagonal, and above-diagonal strategies now use solid circles, dashed
  squares, and dotted triangles respectively. Only the sensitivity PDF/PNG and
  manifest were regenerated from the unchanged saved summary; `summary.csv`
  retained SHA-256
  `f31cca32656c600442245da1816c1c03d628b762c5726ce873adb3056ab6100f`.
  The previous presentation files and manifest were copied to
  `/tmp/ewm-sensitivity-finalization.rqjSMd` before replacement. The refreshed
  manifest verified, the embedded notebook PNG exactly matches the refreshed
  file, and the revised PNG was visually inspected at 3034 by 1294 pixels.
- Initial final quality commands invoked from the non-activated default shell
  resolved to `/usr/bin/python`, where Ruff, mypy, and pre-commit are absent.
  The unit tests and dependency-free checks still passed there. Every missing
  development-tool gate was rerun with the documented Conda `world_models`
  executables and passed.
- An isolated notebook verification was started after the prose correction,
  then stopped before execution when the redundant-encoding issue was found.
  A new final fresh-kernel run started only after the plotting correction. Its
  first sandbox launch failed before any cell because Jupyter requires local
  sockets; the identical isolated retry completed in 877.61 seconds with nine
  sequential code-cell counts, no errors, and a reported experiment time of
  872.483 seconds. It used run ID
  `two-clock-20260723T231854450134Z-69ce6650bd38` under
  `/tmp/evolving-world-models-clock-notebook-final-snbmy0`; it did not modify
  repository files.

No run was excluded, no failing test was weakened, no transition probability
was changed to conceal a failure, and no seed or result was selected after
inspection. The side maxima are intentionally selected after inspection and
are labeled as such rather than treated as confirmatory outcomes. GitHub
Actions was not run because no remote mutation was authorized.

## Next directions

1. Define a preregistered confirmatory two-clock configuration with a
   justified number of trial groups, durable external output retention,
   predefined allocation for every total, held-out evaluation groups, and
   matched-seed uncertainty analysis. This would test whether a strategy
   selected independently generalizes without winner's-curse bias.
2. Profile and benchmark a compiled state-index representation, reusable
   true-environment evaluation kernels, and deterministic parallel execution
   across independent trial groups. Require exact result equivalence before
   adopting any optimization.
3. Separate intrinsic and extrinsic reward/objective interfaces.
4. Add learning and Bayesian planning policies, including posterior
   integration rather than one-model Thompson-style sampling.
5. Distinguish finite physical interaction budgets from balanced generative
   per-context update budgets.
6. Define downstream task distributions and strict train/evaluation
   separation.
7. Preserve matched-seed raw measurements locally during analysis, archive
   accepted evidence externally, and generate publication figures only from
   schema-versioned saved tables.

## Change and artifact summary

### Lock/door example

- `AGENTS.md`: canonical-example invariant changed to lock/door.
- `README.md`: active model, semantics, runnable example, and limitations.
- `docs/reference/visualization.md`: lock/door graph schema and expectations.
- `notebooks/Factored_MDP_Demo.ipynb`: source, assertions, outputs, and graph.
- `tests/test_mdp.py`: canonical transition fixture and exact row/product
  tests.
- `tests/test_visualization.py`: graph, layout, label, and rendering fixtures.

No transition API changed. Historical July 20 and July 21 logs remain
unchanged. No raw data, checkpoint, or standalone generated figure was added.

### Dirichlet beliefs

- `scripts/beliefs.py`: new standard-library-only belief implementation.
- `scripts/__init__.py`: additive exports for the three belief classes.
- `tests/test_beliefs.py`: 30 deterministic contract and regression tests.
- `docs/reference/beliefs.md`: conceptual rationale, rejected alternatives,
  public API, schemas, equations, errors, determinism, complexity, examples,
  and limitations.
- `README.md`: concise public belief workflow and roadmap boundary.
- `AGENTS.md`: invariant assigning belief state and current-time projection to
  `scripts/beliefs.py`.

There is no breaking MDP API change and no new core dependency. No raw data,
checkpoint, result table, or belief-specific figure was produced. Temporary
notebook, wheel, and virtual-environment artifacts remain under `/tmp` and are
not intended for version control.

### Two-clock experiment

- `docs/reference/two-clock-allocation.md`: implemented
  scientific contract, interfaces, alternatives, artifact contract, and
  validation plan.
- `scripts/planning.py`: standard-library exact terminal-reward
  finite-horizon dynamic programming with per-invocation validated-row
  caching.
- `scripts/clock_experiment.py`: true clock construction, balanced selective
  belief updates, deterministic experiment runner, statistics, schemas,
  artifact I/O, exact seed-namespace checks, manifest validation, and
  deterministic fixed-total strategy selection.
- `scripts/experiment_plotting.py`: optional Matplotlib heatmap and
  allocation-sensitivity adapters with mathematical budget labels, pointwise
  Wilson error bars, accessible strategy colors, redundant markers and line
  styles, and allocation annotations.
- `scripts/__init__.py`: additive planning, clock-orchestration, and allocation
  selector exports; plotting intentionally remains an explicit submodule
  import.
- `tests/test_planning.py`, `tests/test_clock_experiment.py`, and
  `tests/test_experiment_plotting.py`: deterministic domain, orchestration,
  artifact, and headless presentation contracts.
- `docs/reference/planning.md`: durable planning semantics and complexity.
- Module and public API docstrings: operational workflow, schemas, algorithms,
  errors, determinism, artifacts, and limitations.
- `notebooks/Two_Clock_Allocation_Experiment.ipynb`: exact expanded
  exploratory execution with sequential outputs, heatmaps, allocation
  sensitivity figure, standalone captions, and evidence-bounded
  interpretation.
- `README.md`: public workflow and notebook index.
- `pyproject.toml`: narrow third-party mypy parsing override.
- `.github/workflows/quality.yml`: structural and fresh-kernel verification
  for both notebooks, with a 1,200-second clock-cell timeout and 45-minute job
  allowance measured for the expanded configuration.
- `results/two-clock/two-clock-20260723T222025252737Z-69ce6650bd38`:
  local Git-ignored configuration, raw trials, summaries, seed ledger,
  manifest, and four figure artifacts for the saved empirical notebook output;
  no external durable copy was provisioned.

These changes are additive. `scripts/mdp.py`, `scripts/beliefs.py`, and the
lock/door contracts remain intact. The expanded result is exploratory and
retained locally for figure provenance; it is neither a confirmatory
scientific result nor durable publication evidence.

### Documentation consolidation

- `AGENTS.md`: reserves `docs/reference/` for durable major concepts, places
  operational contracts in docstrings, user workflows in the README, and
  run-specific evidence in the daily mini-paper log, and prohibits new
  `docs/design/` and `docs/scripts/` hierarchies. It also keeps optional
  presentation dependencies out of core/package-root imports while permitting
  explicit presentation submodules to import their declared dependencies
  normally.
- `docs/reference/beliefs.md`: absorbed the durable rationale from the
  temporary dated belief-design note.
- `docs/reference/two-clock-allocation.md`: replaced the dated design path as
  the canonical conceptual reference.
- The duplicated per-script pages and temporary design hierarchy were retired;
  live links now target the canonical references.
- `.gitignore` and `AGENTS.md`: experiment run directories are local-only,
  `results/` is ignored, and durable claims require a logged external archive.
  Existing local runs were preserved; no result artifact was deleted.

## Verification

### Earlier lock/door and belief milestone commands

The final commands actually run from the repository root for the first two
milestones included:

```bash
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m unittest tests.test_mdp tests.test_visualization -v
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest tests.test_beliefs -v
python3.9 -B -m unittest tests.test_beliefs -v
/data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  validate-config .pre-commit-config.yaml
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff format --check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python -m mypy scripts
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest discover -s tests -v
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m compileall -q scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m json.tool notebooks/Factored_MDP_Demo.ipynb
/data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit run --all-files
env PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files --hook-stage pre-push
```

Focused transition/visualization: 51 tests passed. Focused beliefs on Python
3.12: 30 tests in 0.075 seconds, passed. Focused beliefs on Python 3.9:
30 tests in 0.073 seconds, passed. The repository suite at that historical
point passed 81 tests in 0.611 seconds. The non-mutating gates, post-log
pre-commit stage, corrected post-log pre-push stage, lock/door notebook
execution, documentation examples, and final clean-wheel import and stable-log
query passed. `git diff --check` passed, and historical July 20 and July 21
logs were unchanged.

### Two-clock milestone commands and outcomes

The focused and complete commands actually run included:

```bash
MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest tests.test_planning -v
MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest tests.test_clock_experiment -v
MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest tests.test_experiment_plotting -v
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff format --check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python -m mypy scripts
MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest discover -s tests -v
/usr/bin/python3.9 -B -m unittest \
  tests.test_mdp tests.test_beliefs tests.test_planning \
  tests.test_clock_experiment -v
/data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit run --all-files
PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files --hook-stage pre-push
pkgcheck_handoff_dir="$(mktemp -d \
  /tmp/evolving-world-models-handoff-package-check.XXXXXX)"
mkdir -p "$pkgcheck_handoff_dir/dist" "$pkgcheck_handoff_dir/outside"
/data/user_data/nquazi/.conda/envs/world_models/bin/python -m pip check
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$pkgcheck_handoff_dir/dist" .
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m venv "$pkgcheck_handoff_dir/venv"
"$pkgcheck_handoff_dir/venv/bin/python" -m pip install --no-deps \
  "$pkgcheck_handoff_dir"/dist/*.whl
cd "$pkgcheck_handoff_dir/outside"
"$pkgcheck_handoff_dir/venv/bin/python" -c \
  "import importlib.util, scripts; assert importlib.util.find_spec('matplotlib') is None; assert scripts.ClockExperimentConfig; assert scripts.TabularDirichletBelief; assert scripts.plan_finite_horizon; print(scripts.__file__)"
```

The focused suites passed 13 planning, 16 clock-experiment, and four plotting
tests. Strict mypy reported `Success: no issues found in 7 source files.` The
final complete Python 3.12 suite passed 114 tests in 0.687 seconds. The
dependency-free Python 3.9 core suite passed 82 tests in 0.383 seconds. Both
repository-wide hook stages passed, as did Ruff, Ruff formatting, compilation,
and both notebook JSON checks.

The two-clock notebook was also checked as JSON, then executed from a fresh
kernel with isolated temporary state and approved loopback sockets. The
successful final output was
`/tmp/two-clock-final-notebook.uxSDlU/Two_Clock_Allocation_Experiment.executed.ipynb`;
it contained sequential code-cell counts 1 through 9, ten intentional outputs,
and no errors. The runner reported 0.556 seconds. The records and summaries
round-tripped, navigation invariance held, the manifest verified, and the PNG
was visually inspected.

The final packaging check ran `pip check`, built a 63,231-byte wheel with
SHA-256
`45eefc0170c1ac407d55956a08f5fabb1a77eed66cd7fda53af0a7b66f311d05`,
installed it without dependencies into a fresh virtual environment, changed
outside the checkout, and imported `scripts` from `site-packages`. Matplotlib
was absent there, while the belief, planning, and clock APIs imported
successfully.

GitHub Actions remained unrun because no commit or push was authorized. No
remote status is claimed.

### Allocation-sensitivity extension commands and outcomes

The exact expanded notebook execution used:

```bash
cd /tmp/two-clock-sensitivity.idyo2J
env \
  MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/two-clock-sensitivity.idyo2J/matplotlib \
  JUPYTER_CONFIG_DIR=/tmp/two-clock-sensitivity.idyo2J/jupyter \
  JUPYTER_RUNTIME_DIR=/tmp/two-clock-sensitivity.idyo2J/runtime \
  IPYTHONDIR=/tmp/two-clock-sensitivity.idyo2J/ipython \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m jupyter nbconvert \
  --to notebook \
  --execute \
  --output Two_Clock_Allocation_Experiment.executed.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  Two_Clock_Allocation_Experiment.ipynb
```

The post-cache one-trial benchmark used:

```bash
/data/user_data/nquazi/.conda/envs/world_models/bin/python -c \
  'from time import perf_counter; from scripts.clock_experiment import ClockExperimentConfig, run_clock_experiment; c=ClockExperimentConfig(num_states=6,intended_probability=0.8,predictable_direction="right",horizon=30,trials=1,x_updates=(0,1,2,3,4),y_updates=(0,1,2,3,4),fixed_total_budgets=(2,4,6),prior_concentration=1.0,master_seed=20260723); t=perf_counter(); r=run_clock_experiment(c); print("records",len(r.trials),"summaries",len(r.summaries),"elapsed_seconds",perf_counter()-t)'
```

The attribution profile used the same `c` configuration:

```bash
/data/user_data/nquazi/.conda/envs/world_models/bin/python -c \
  'import cProfile,pstats; from scripts.clock_experiment import ClockExperimentConfig,run_clock_experiment; c=ClockExperimentConfig(num_states=6,intended_probability=0.8,predictable_direction="right",horizon=30,trials=1,x_updates=(0,1,2,3,4),y_updates=(0,1,2,3,4),fixed_total_budgets=(2,4,6),prior_concentration=1.0,master_seed=20260723); p=cProfile.Profile(); p.enable(); run_clock_experiment(c); p.disable(); pstats.Stats(p).strip_dirs().sort_stats("cumulative").print_stats(35)'
```

The final source-state quality commands from the repository root were:

```bash
/data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  validate-config .pre-commit-config.yaml
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff format --check scripts tests
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m mypy scripts
env MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest discover -s tests -v
/usr/bin/python3.9 -B -m unittest \
  tests.test_mdp tests.test_beliefs tests.test_planning \
  tests.test_clock_experiment -v
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
python -m json.tool \
  notebooks/Two_Clock_Allocation_Experiment.ipynb > /dev/null
git diff --check
env \
  MPLBACKEND=Agg \
  PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files
env \
  MPLBACKEND=Agg \
  PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files --hook-stage pre-push
```

The package check used:

```bash
package_check_dir="$(mktemp -d /tmp/ewm-allocation-package.XXXXXX)"
mkdir -p "$package_check_dir/dist" "$package_check_dir/outside"
/data/user_data/nquazi/.conda/envs/world_models/bin/python -m pip check
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir "$package_check_dir/dist" .
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m venv "$package_check_dir/venv"
wheel_path="$(find "$package_check_dir/dist" -maxdepth 1 \
  -type f -name '*.whl' -print -quit)"
"$package_check_dir/venv/bin/python" -m pip install --no-deps "$wheel_path"
cd "$package_check_dir/outside"
"$package_check_dir/venv/bin/python" -c \
  'import importlib.util, scripts; assert scripts.AllocationStrategyPoint; assert scripts.select_allocation_strategies; assert scripts.ALLOCATION_STRATEGY_ORDER == ("equal", "best_below_diagonal", "best_above_diagonal"); assert importlib.util.find_spec("matplotlib") is None; print(scripts.__file__)'
```

All commands in the final quality and package groups passed. The exact
notebook run completed without cell errors and reported 846.696 seconds for
the experiment runner. The final 126-test Python 3.12 suite and 89-test Python
3.9 suite passed, as did both hook stages and the clean external import.

After documentation consolidation and redundant plot styles were finalized,
the clock notebook was executed once more from a fresh kernel using an isolated
temporary working directory:

```bash
cd /tmp/evolving-world-models-clock-notebook-final-snbmy0
env \
  PYTHONPATH=/data/user_data/nquazi/repos/evolving-world-models \
  MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/evolving-world-models-clock-notebook-final-snbmy0/mpl \
  JUPYTER_CONFIG_DIR=/tmp/evolving-world-models-clock-notebook-final-snbmy0/jupyter-config \
  JUPYTER_DATA_DIR=/tmp/evolving-world-models-clock-notebook-final-snbmy0/jupyter-data \
  JUPYTER_RUNTIME_DIR=/tmp/evolving-world-models-clock-notebook-final-snbmy0/jupyter-runtime \
  IPYTHONDIR=/tmp/evolving-world-models-clock-notebook-final-snbmy0/ipython \
  XDG_CACHE_HOME=/tmp/evolving-world-models-clock-notebook-final-snbmy0/xdg-cache \
  /usr/bin/time -p \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m jupyter nbconvert --to notebook --execute \
  --output executed.ipynb --ExecutePreprocessor.timeout=1200 \
  input.ipynb
```

The escalated socket-enabled retry completed in 877.61 seconds; the notebook
reported 872.483 seconds for the experiment. It had sequential execution counts
1 through 9, no error outputs, 5,000 binary records, a verified manifest, the
correct navigation-reuse wording in source and rendered caption, and the exact
solid-circle, dashed-square, and dotted-triangle sensitivity styles on both
panels. Its configuration, trials, summaries, seeds, and both PNGs matched the
retained artifacts exactly; the PDFs differed only in expected generation
metadata.

### Experiment-result storage policy verification

The repository policy now excludes every generated experiment-run directory
from version control, regardless of artifact size. The root-anchored
`/results/` ignore rule was verified against the accepted run's `trials.csv`;
`git status --short --branch` no longer listed `results/`. The existing local
run was preserved without deletion or modification.

The focused documentation and ignore-file hooks passed:

```bash
env MPLBACKEND=Agg \
  PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit run \
  --files .gitignore AGENTS.md README.md \
  docs/reference/two-clock-allocation.md \
  docs/logs/2026-07-23-transition-learning-foundations.md
```

The complete Python 3.12 unit suite also passed again:

```bash
env MPLBACKEND=Agg /usr/bin/time -p \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest discover -s tests -v
```

It ran 126 tests in 0.746 seconds; including interpreter and test-discovery
overhead, `/usr/bin/time` reported 1.24 seconds wall time. The two-clock
notebook was not re-executed for this documentation-and-ignore-only policy
change because neither its source nor the experiment implementation changed.

### Pre-commit integrity corrections

A read-only pre-commit review found two integrity gaps that the earlier test
suite did not cover:

1. `TrialRecord` accepted a binary `success` value that contradicted its
   recorded terminal state. Because aggregation trusted that redundant field,
   a malformed record could silently change the primary metric.
2. `write_run_manifest()` could refresh a manifest after a canonical artifact
   was deleted, and `verify_run_manifest()` could accept that incomplete run
   when the missing file was also removed from the manifest.

The record constructor now requires navigation success to equal
`final_x == target_x` and synchronization success to equal
`final_x == final_y`. Manifest writing and verification now independently
require `config.json`, `trials.csv`, `summary.csv`, and `seeds.csv`. Regression
tests cover both task definitions and removal of each required artifact. The
accepted local run passed the stricter verifier without modification.

The same review corrected three documentation inconsistencies: README CI
wording now covers both notebooks, the artifact contract lists `seeds.csv`,
and the replication instructions distinguish an ephemeral temporary-CWD run
from the retained repository-local run.

Focused verification used:

```bash
env MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest tests.test_clock_experiment -v
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m ruff check scripts/clock_experiment.py tests/test_clock_experiment.py
/data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -m mypy scripts/clock_experiment.py
/data/user_data/nquazi/.conda/envs/world_models/bin/python -B -c \
  'from scripts.clock_experiment import verify_run_manifest; verify_run_manifest("results/two-clock/two-clock-20260723T222025252737Z-69ce6650bd38")'
```

All 23 focused experiment tests, Ruff, mypy, and accepted-run manifest
verification passed. The notebook was not re-executed because these
corrections reject inconsistent records and incomplete artifact sets without
changing any valid generated record, transition, policy, random stream,
aggregation, plot, or notebook source.

The first staged `pre-commit run --all-files` failed only because the isolated
mypy hook does not install the optional Matplotlib dependency. Repository
mypy, which runs in the declared development environment, passed. The
Matplotlib/NumPy override in `pyproject.toml` now explicitly permits those
optional imports to be missing in an isolated hook while strict-checking the
repository modules. The focused mypy hook and both complete hook stages then
passed.

Final post-correction verification additionally included:

```bash
env MPLBACKEND=Agg \
  /data/user_data/nquazi/.conda/envs/world_models/bin/python \
  -B -m unittest discover -s tests -v
/usr/bin/python3.9 -B -m unittest \
  tests.test_mdp tests.test_beliefs tests.test_planning \
  tests.test_clock_experiment -v
env MPLBACKEND=Agg \
  PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files
env MPLBACKEND=Agg \
  PATH=/data/user_data/nquazi/.conda/envs/world_models/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /data/user_data/nquazi/.conda/envs/world_models/bin/pre-commit \
  run --all-files --hook-stage pre-push
```

The complete Python 3.12 suite passed 127 tests, and the dependency-free
Python 3.9 core suite passed 90 tests. A clean wheel was rebuilt, installed
without optional dependencies in
`/tmp/ewm-precommit-package.lJc2p3/venv`, and imported from outside the
checkout; `scripts` imported without Matplotlib, and the wheel SHA-256 was
`67b59a563eba4b4b8db3ce544564d3016e84cf6ba957ac25c3692143f27191ec`.
