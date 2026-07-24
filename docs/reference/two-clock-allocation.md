# Two-clock finite-budget allocation experiment

This page is the durable conceptual reference for the two-clock allocation
experiment. Run-specific configuration, evidence, and replication details
belong in the corresponding research log.

## Problem

The repository needs its first downstream-control experiment over learned
transition beliefs. The experiment compares learning an action-sensitive
controllable clock with learning an action-invariant predictable clock under a
finite allocation of factor-local belief updates.

Both clocks have the same finite state space and the same intended-transition
reliability. They differ only in whether the intended next state is selected by
the current action. Each trial learns independent tabular Dirichlet rows,
samples one plausible transition model, plans under that sampled model, and
evaluates the resulting policy in the true environment.

The primary empirical outputs are a pair of heatmaps indexed by controllable
and predictable updates per local conditioning context and a two-panel
fixed-total allocation-sensitivity diagnostic. One heatmap reports terminal
target-navigation success and the other reports terminal clock-synchronization
success. The sensitivity panels compare equal and best observed
off-diagonal allocations as total budget changes.

## Scientific contract

Let

$$
x_t,y_t\in\mathbb Z_M,
\qquad
a_t\in\{\mathrm{left},\mathrm{right},\mathrm{stay}\},
$$

where $x$ is the controllable clock, $y$ is the predictable clock, and $a$ is
the external `hand` action. The implementation uses variable names `x`, `y`,
and `hand`; $P$ and $p$ remain reserved for probability notation.

Clock arithmetic wraps modulo $M$, with $M\geq 3$. Define

$$
g_x(x,a)=
\begin{cases}
x-1 \pmod M & a=\mathrm{left},\\
x+1 \pmod M & a=\mathrm{right},\\
x            & a=\mathrm{stay},
\end{cases}
$$

and

$$
g_y(y;d_y)=y+d_y\pmod M,
\qquad d_y\in\{-1,+1\}.
$$

The experimenter selects the fixed predictable direction $d_y$. For either
factor $z\in\{x,y\}$, the intended successor has probability $q$ and every
other next state has equal residual mass:

$$
P(z'=g_z)=q,
\qquad
P(z'=u)=\frac{1-q}{M-1}
\quad\text{for }u\neq g_z.
$$

The configuration requires $1/M<q\leq1$, so the intended successor is uniquely
most likely. The predictable factor declares `hand` as a parent to give its
belief one independent row for every $(y,a)$ context, but its true rows are
identical across actions:

$$
P(y'\mid y,a)=P(y'\mid y).
$$

This deliberately permits finite data to induce spurious action dependence in
the learned predictable factor. The model does not pool the known common value
of $q$ across states, actions, or factors.

## Learning-budget semantics

The budgets $N_x$ and $N_y$ count updates to every factor-local context, not
joint-state contexts and not trajectory steps:

- each $(x,a)$ row receives exactly $N_x$ independently sampled outcomes;
- each $(y,a)$ row receives exactly $N_y$ independently sampled outcomes; and
- predictable observations are independently sampled for each action row even
  though the true rows are equal.

There are $3M$ contexts per factor. A heatmap cell therefore consumes

$$
B_x=3MN_x,\qquad
B_y=3MN_y,\qquad
B_{\mathrm{total}}=3M(N_x+N_y)
$$

local observations per trial.

Selective learning is implemented by updating the two leaf
`TabularDirichletBelief` instances separately. Calling
`FactoredDirichletBelief.update()` would incorrectly spend one update on both
factors at once and is therefore not part of this experiment's learning path.

A rectangular $N_x\times N_y$ grid is a sample-size response surface because
its total evidence changes between cells. Allocation claims are restricted to
configured diagonals satisfying $N=N_x+N_y$.

## Tasks and estimands

Each trial group draws $x_0$, $y_0$, and the navigation target $g$
independently and uniformly from $\mathbb Z_M$. The task instance is known to
the planner. The policy observes both clocks after every transition and may
choose a time-dependent feedback action.

The episode contains exactly $H$ actions and transitions. Stage rewards are
zero and the undiscounted terminal rewards are

$$
R_H^{\mathrm{target}}(x,y;g)=\mathbf 1[x=g]
$$

and

$$
R_H^{\mathrm{sync}}(x,y)=\mathbf 1[x=y].
$$

For a posterior-sampled model $\widehat P$, exact finite-horizon dynamic
programming computes

$$
V_H(s)=R_H(s),
\qquad
Q_t(s,a)=\sum_{s'}\widehat P(s'\mid s,a)V_{t+1}(s'),
\qquad
V_t(s)=\max_a Q_t(s,a).
$$

Exact maximizing ties are resolved uniformly with an injected, named random
stream; there is no approximate equality tolerance. The sampled model and the
resulting deterministic policy are held fixed during evaluation, and no online
belief update occurs.

The primary trial metric is one binary terminal outcome from a rollout in the
true model. A cell reports the sample mean $k/T$ and a Wilson 95% binomial
interval over the $T$ independent top-level trial groups. Exact true-model
policy success probability is retained only as a lower-variance diagnostic and
does not replace the requested binary heatmap metric.

The target task is mathematically independent of $y$. Its planner may use the
equivalent $x$-marginal model, and matched random streams make invariance to
$N_y$ an exact negative-control contract. Synchronization plans over the full
$(x,y)$ state.

### Allocation-sensitivity diagnostic

The notebook additionally compares allocation strategies as the fixed total
per-context budget

$$
N=N_x+N_y
$$

changes. Only configured positive even totals with complete diagonal support
are eligible. For each task and total, the derived strategy records are:

- **equal allocation:** the unique cell $N_x=N_y=N/2$;
- **best below diagonal:** the largest observed $k/T$ among cells with
  $N_x>N_y$; and
- **best above diagonal:** the largest observed $k/T$ among cells with
  $N_x<N_y$.

“Below” and “above” refer to the heatmap geometry, where $N_x$ is horizontal
and $N_y$ is vertical. Tied best cells are resolved deterministically toward
the smallest $|N_x-N_y|$, followed by canonical $(N_x,N_y)$ order. Each
selected record retains its original Wilson 95% interval and allocation
coordinates.

The two best-side curves are descriptive, post-hoc oracle envelopes, not
prespecified policies. Their pointwise Wilson intervals describe the selected
cells and are not adjusted for choosing the maximum among multiple cells.
They must not be interpreted as valid confidence intervals for an unknown
best strategy or as confirmatory method comparisons.

## Reproducibility and pairing

One integer master seed is expanded into stable named sub-seeds using SHA-256
over a canonical JSON namespace. Python's process-dependent `hash()` is never
used.

For each trial and factor-local context, the largest requested observation
sequence is generated once. Smaller budgets use prefixes of that sequence.
Controllable observation and posterior-model streams omit $N_y$ from their
seed namespace; predictable streams omit $N_x$. Initial states, targets, and
true-environment rollout streams are matched across cells. Task-specific tie
and rollout streams remain separate.

For a fixed trial and $N_x$, the sampled controllable factor is reused across
all $N_y$ values. The symmetric rule applies to the predictable factor. The
two sampled factors are composed into one `FactoredMDP` for synchronization;
navigation uses the same sampled controllable factor through its equivalent
$x$-marginal planner.

This common-random-number design supports paired fixed-budget comparisons and
turns unexpected target-task variation across $N_y$ into a regression signal.

## Interfaces

### Planning

`scripts/planning.py` remains standard-library-only and introduces:

```python
plan_finite_horizon(
    model,
    terminal_reward,
    horizon,
    *,
    rng=None,
) -> FiniteHorizonPolicy

evaluate_finite_horizon_policy(
    model,
    policy,
    terminal_reward,
    initial_state,
) -> float
```

`FiniteHorizonPolicy` stores complete time-indexed actions and values over the
finite state space. Every external parent of the supplied model is treated as
an action variable. Public queries validate the exact state scope, domain, and
step. Planning requires finite variable and action domains and finitely
enumerable transition distributions.

### Experiment orchestration

`scripts/clock_experiment.py` introduces immutable configuration, world,
belief, trial-record, summary-record, and experiment-result value objects plus
these principal operations:

```python
build_clock_world(config) -> ClockWorld
build_clock_beliefs(config, world=None) -> ClockBeliefs
run_clock_experiment(config) -> ClockExperimentResult
select_allocation_strategies(
    summaries,
    total_budgets,
) -> tuple[AllocationStrategyPoint, ...]
write_clock_experiment(result, output_root, run_id=None) -> RunArtifacts
read_summary_csv(path) -> tuple[CellSummary, ...]
write_run_manifest(run_directory) -> Path
```

The configuration explicitly records $M$, $q$, predictable direction, $H$,
$T$, the ordered $N_x$ and $N_y$ grids, fixed-total diagonals, symmetric
per-outcome concentration $\alpha$ (default `1.0`), and the master seed.
Serialized configuration and result schemas carry explicit version numbers.

The runner performs no I/O. Artifact writing is a separate explicit call,
creates a unique directory, refuses to overwrite an existing run, writes UTF-8
JSON and CSV atomically, and records file sizes and SHA-256 checksums.

### Plotting

`scripts/experiment_plotting.py` is an optional adapter with ordinary
module-level Matplotlib imports. It is not imported from `scripts/__init__.py`,
so importing the transition, belief, planning, or orchestration layers does not
require Matplotlib.

The primary plotting call accepts saved summary records and caller-provided
axes. Its task-specific colormaps are presentation parameters, not scientific
experiment parameters:

```python
plot_success_heatmaps(
    summaries,
    *,
    navigation_colormap="viridis",
    synchronization_colormap="plasma",
    axes=None,
)

plot_allocation_sensitivity(
    allocation_strategy_points,
    *,
    axes=None,
)
```

Each panel has its own colorbar because the colormaps differ. Both use the
identical fixed range $[0,1]$, discrete unsmoothed cells, explicit integer
ticks, concise titles, and axes labeled in updates per factor-local context.
The allocation-sensitivity call creates task-specific line panels over total
budget, displays Wilson 95% error bars for the selected cells, and annotates
the selected $(N_x,N_y)$ allocations. Plotting functions never call `show()`
or write files.

## Artifact contract

Every explicit run directory contains:

- `config.json`: versioned complete scientific configuration and seed method;
- `trials.csv`: one row per task, cell, and trial with task instance, budgets,
  derived seed identifiers, terminal state, binary success, planned value, and
  exact true-policy diagnostic;
- `seeds.csv`: the complete canonical namespace-to-derived-seed ledger for
  every random stream used by the run;
- `summary.csv`: successes, $T$, mean, Monte Carlo standard error, and Wilson
  95% bounds per task and cell;
- `figures/two-clock-success.pdf` and a 300-dpi PNG generated after reloading
  `summary.csv`;
- `figures/allocation-sensitivity.pdf` and a 300-dpi PNG generated from
  deterministic strategy selections over the same reloaded summaries; and
- `manifest.json`: relative paths, byte sizes, SHA-256 checksums, schema
  versions, and generation metadata.

The notebook writes run artifacts to the ignored local `results/` tree or a
temporary run root. Run directories are never committed. Small intentional
notebook outputs may remain for exposition, but they are not canonical data.
Before a result supports a durable scientific claim, archive the complete
verified run externally and record its URI, manifest or checksums, access
requirements, and retention policy in the research log.

## Notebook

`notebooks/Two_Clock_Allocation_Experiment.ipynb` is a thin, top-to-bottom
executable analysis. It contains one visible configuration cell, equations,
budget accounting, kernel and planner assertions, an exploratory run,
artifact reload checks, the two heatmaps, the allocation-sensitivity
diagnostic, and evidence-bounded limitations. Reusable algorithms, selection
logic, and schema behavior remain in tested modules.

The notebook labels its output as exploratory, not a
confirmatory result. A confirmatory run must predeclare its configuration,
sample size, success criterion, and durable external artifact location in the
daily research log before execution.

## Alternatives considered

### Action-free predictable belief

Removing `hand` from the predictable belief would pool away the intentional
comparison in per-context observation counts. It is rejected for this
experiment even though it is the more parsimonious model of the true kernel.

### Joint observation updates

One physical joint transition reveals both clocks. Updating both factor
beliefs from every transition would eliminate the allocation mechanism.
Selective local generative observations instead represent a finite
learning/update-capacity budget. This interpretation must remain explicit.

### Different residual supports

Using only local failure moves for $x$ while spreading $y$ failures over every
state would confound controllability with row entropy and support size. Both
factors therefore use the same full residual support.

### Posterior-mean planning

The requested trial samples one plausible transition model. Planning under the
posterior mean or integrating over the posterior would estimate a different
agent and is not substituted.

### Notebook-owned algorithms

Embedding belief updates, dynamic programming, aggregation, or plotting logic
only in notebook cells would make behavior difficult to test and reuse. The
notebook remains an orchestration and exposition layer.

### Making Matplotlib a core import

Matplotlib is conventional in ML environments, but repository policy keeps
visualization optional. A dedicated plotting module with ordinary top-level
imports provides a clean dependency boundary without unnecessary function-
level lazy imports.

## Compatibility and migration

The change is additive. `scripts/mdp.py`, `scripts/beliefs.py`, and the
lock/door example retain their contracts. New public planning and experiment
objects may be exported from `scripts/__init__.py` only when doing so does not
import Matplotlib. The plotting adapter remains an explicit submodule import.

Artifact dictionaries and CSV files are portable only under their declared
schema versions. This schema contract does not authorize committing generated
run artifacts; accepted durable evidence must be archived externally under the
storage policy above. Existing `to_dict()` inspection representations are not
used as experiment persistence formats.

## Validation plan

Implementation is accepted when:

1. every true clock row has the requested intended and residual probabilities,
   normalizes across all contexts, wraps correctly, and the true $y$ rows are
   action invariant;
2. both beliefs contain exactly $3M$ rows, each row receives the requested
   number of updates, and neither factor is updated implicitly;
3. nested observation prefixes, cross-axis factor reuse, and complete run
   output reproduce exactly under the same master seed and remain invariant to
   grid enumeration order;
4. finite-horizon planning matches hand-computed examples, validates all
   boundaries, handles $H=0$, and exact policy evaluation agrees with the
   planner on its own model;
5. target-task policies and paired binary outcomes are invariant to $N_y$,
   while synchronization uses both learned factors;
6. trial and summary schemas are complete, binary outcomes aggregate exactly,
   Wilson intervals are correct, artifacts round-trip, manifests verify, and
   existing paths are never overwritten;
7. headless figure tests verify titles, labels, ticks, colormaps, color ranges,
   separate colorbars, strategy lines, allocation annotations, and Wilson
   intervals without brittle pixel comparisons;
8. the notebook executes from a fresh kernel with sequential counts, no errors,
   compact intentional outputs, and figures generated from reloaded saved
   measurements; and
9. Ruff, formatting, strict mypy, the complete unit suite, compilation,
   notebook JSON checks, packaging, both hook stages, and supported-Python
   checks pass or any environmental blocker is recorded exactly.

## Implementation status

The interfaces and validation plan described here are implemented. Exact
configurations, measured runtimes, artifacts, results, failures, and quality
checks are recorded in the
[transition-learning foundations log](../logs/2026-07-23-transition-learning-foundations.md)
rather than duplicated in this conceptual reference.

## Known limitations

- Independent rows do not pool the known shared $q$, translations around the
  ring, or the predictable factor's action invariance.
- A dense symmetric prior assigns positive mass to nonlocal jumps even if a
  future variant uses sparse true dynamics.
- Balanced generative row sampling is not an online trajectory or physical
  interaction budget.
- Navigation performs one rollout for each sampled controllable model and
  records that matched outcome across all $N_y$ cells. Synchronization performs
  one rollout for each sampled $(N_x,N_y)$ model pair. Both tasks therefore
  retain substantial binomial Monte Carlo uncertainty at small $T$.
- Exact dense joint dynamic programming scales as
  $O(H\,|A|\,M^4)$ for the two-clock state space. Measured runtime justified
  caching validated transition rows, but expectation arithmetic remains dense
  and dominates the expanded experiment.
- Pointwise monotonic improvement is not guaranteed for posterior-sampled
  planning, especially at small sample sizes.
- Selecting the maximum observed cell separately for each task, side, and
  total creates winner's-curse bias. The sensitivity plot is descriptive
  unless a future protocol prespecifies strategies or performs
  selection-adjusted inference on held-out trials.
