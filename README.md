# Evolving World Models

[![Quality](https://github.com/nquaz/evolving-world-models/actions/workflows/quality.yml/badge.svg)](https://github.com/nquaz/evolving-world-models/actions/workflows/quality.yml)

This repository studies how the objectives used to learn a world model shape
performance on later tasks. The motivating distinction is between learning
predictable but uncontrollable regularities and learning action-sensitive
regularities that support control.

The transition-model foundation, conjugate Bayesian beliefs over complete
finite transition tables, exact finite-horizon terminal-reward planning, and a
reproducible two-clock learning-budget experiment are implemented. General
reward interfaces, online learning algorithms, physical interaction budgets,
and broader downstream-task distributions remain later milestones.

## Transition model

`AbstractMDP` represents a named conditional transition model

$$
p(x' \mid x, \mathrm{pa}(x)).
$$

Here, `variables` is the scope of the values in $x$ and `parent_variables`
is the union of the values in $\mathrm{pa}(x)$. A conventional MDP is
the special case where `variables` describes the state and
`parent_variables` describes the action.

`FactoredMDP` combines constituent models with disjoint predicted scopes:

$$
p(x' \mid x, \mathrm{pa}(x))
= \prod_i p_i(x_i' \mid x_i, \mathrm{pa}(x_i)).
$$

A factor may use another factor's variable as a parent. The composition reads
that parent's **current-time** value from the joint $x$; only parents not
predicted by any factor remain external inputs to the composite model. Factor
updates are therefore synchronous.

The worked example uses a persistent lock and an action-sensitive door:

$$
p(\ell', d' \mid \ell, d, a)
= p(\ell' \mid \ell)\,p(d' \mid d, \ell, a).
$$

The lock is the latch mechanism's setting and evolves independently of the
agent's action, retaining its current setting with probability $0.95$. The door
factor reads the current lock setting $\ell_t$, never $\ell_{t+1}$. An opening
attempt from a closed door succeeds with probability $0.9$ only when unlocked;
an open door remains open under `open`, while `close` leaves it open with
probability $0.1$. The finite state space is Cartesian, so a locked latch with
an open door is a valid modeled state; enforcing forbidden joint states would
require a separate constrained-state feature.

## Repository layout

- `scripts/mdp.py` contains variables, immutable assignments, transition
  distributions, the abstract MDP interfaces, the complete finite
  `TabularMDP`, and the product `FactoredMDP`.
- `scripts/beliefs.py` contains Dirichlet beliefs, explicit posterior updates,
  posterior-predictive queries, fixed posterior-mean snapshots, and seeded
  sampling of plausible finite transition models.
- `scripts/planning.py` contains exact finite-horizon backward induction and
  exact evaluation of fixed feedback policies.
- `scripts/clock_experiment.py` contains the two-clock true model, balanced
  factor-local belief updates, matched random streams, trial aggregation, and
  versioned artifact schemas.
- `scripts/experiment_plotting.py` contains the optional two-panel success
  heatmap adapter with configurable `viridis` and `plasma` defaults and the
  fixed-total allocation-sensitivity line-plot adapter.
- `scripts/visualization.py` contains the optional NetworkX, Matplotlib, and
  Graphviz adapters used by the inherited visualization methods.
- `notebooks/Factored_MDP_Demo.ipynb` builds a lock/door world, verifies the
  factor product exactly, checks every transition context for normalization,
  and contrasts action-sensitive with action-invariant dynamics.
- `notebooks/Two_Clock_Allocation_Experiment.ipynb` runs the reproducible clock
  experiment and generates heatmap and allocation-sensitivity figures from
  saved summaries.
- [`docs/reference/visualization.md`](docs/reference/visualization.md) documents
  graph semantics, rendering backends, and the visualization API.
- [`docs/reference/beliefs.md`](docs/reference/beliefs.md) documents belief
  construction, updates, inspection, snapshots, sampling, and factored
  current-time projection.
- [`docs/reference/planning.md`](docs/reference/planning.md) documents the
  terminal timing, policy API, exact tie handling, and complexity contract.
- [`docs/reference/two-clock-allocation.md`](docs/reference/two-clock-allocation.md)
  documents the finite-budget allocation contract, estimands, pairing,
  artifacts, and validation criteria.
- The `tests/` modules separately cover transitions, beliefs, planning,
  orchestration, graph visualization, and experiment plotting.
- `.github/workflows/quality.yml` independently checks supported Python
  versions, executes both notebooks, and builds the distribution.

The transition-model core uses only the Python standard library and supports
Python 3.9 or newer. Visualization dependencies are optional.

## Installation

Install the package in editable mode with the capabilities you need:

```bash
python -m pip install -e '.[viz]'       # NetworkX + Matplotlib
python -m pip install -e '.[graphviz]'  # NetworkX + Python Graphviz wrapper
python -m pip install -e '.[dev]'       # quality tools, tests, notebooks, backends
```

Graphviz file rendering also requires the Graphviz system executables, such as
`dot`, to be installed and available on `PATH`.

## Core API

```python
from itertools import product
from math import isclose

from scripts.mdp import FactoredMDP, TabularMDP, Variable

lock_state = Variable("lock", ("locked", "unlocked"))
door_state = Variable("door", ("closed", "open"))
action = Variable("action", ("open", "close"))

lock_model = TabularMDP(
    variables=(lock_state,),
    transitions=(
        (
            {"lock": "locked"},
            {},
            (({"lock": "locked"}, 0.95), ({"lock": "unlocked"}, 0.05)),
        ),
        (
            {"lock": "unlocked"},
            {},
            (({"lock": "locked"}, 0.05), ({"lock": "unlocked"}, 0.95)),
        ),
    ),
)

door_open_probability = {
    ("closed", "locked", "open"): 0.0,
    ("closed", "unlocked", "open"): 0.9,
    ("open", "locked", "open"): 1.0,
    ("open", "unlocked", "open"): 1.0,
    ("closed", "locked", "close"): 0.0,
    ("closed", "unlocked", "close"): 0.0,
    ("open", "locked", "close"): 0.1,
    ("open", "unlocked", "close"): 0.1,
}
door_rows = []
for current_door, current_lock, requested_action in product(
    door_state.domain,
    lock_state.domain,
    action.domain,
):
    open_probability = door_open_probability[
        current_door,
        current_lock,
        requested_action,
    ]
    door_rows.append(
        (
            {"door": current_door},
            {"lock": current_lock, "action": requested_action},
            (
                ({"door": "closed"}, 1.0 - open_probability),
                ({"door": "open"}, open_probability),
            ),
        )
    )

door_model = TabularMDP(
    variables=(door_state,),
    parent_variables=(lock_state, action),
    transitions=door_rows,
)
world = FactoredMDP((lock_model, door_model))

probability = world.transition_probability(
    next_state={"lock": "locked", "door": "open"},
    current={"lock": "unlocked", "door": "closed"},
    parents={"action": "open"},
)
assert isclose(probability, 0.05 * 0.9)
assert world.parent_variables == (action,)
```

The public operations are:

- `transition_distribution(current, parents=None)`
- `transition_probability(next_state, current, parents=None)`
- `transition_log_probability(next_state, current, parents=None)`
- `sample_transition(current, parents=None, rng=None)`
- `to_dict()` and `str(model)` for JSON-safe model inspection
- `to_networkx(view="factor")`
- `draw(ax=None, view="factor", layout="layered", ...)`
- `to_graphviz(view="factor", rankdir="LR")`

All calls validate exact variable names and domain membership. Tabular rows are
also checked for complete conditioning-context coverage, valid next-state
assignments, and normalized finite probabilities.

## Bayesian transition beliefs

`TabularDirichletBelief` places an independent Dirichlet posterior over every
complete finite transition row. For context $c=(x,\mathrm{pa}(x))$ and next
outcome $y$, its posterior-predictive probability is

$$
p(x'=y\mid c,D)
=
\frac{\alpha_{c,y}+n_{c,y}}
{\sum_z(\alpha_{c,z}+n_{c,z})}.
$$

A scalar prior supplies the same positive concentration to every outcome.
Complete explicit concentration rows are also supported. Priors and integer
observation counts remain separate and can be inspected through defensive
copies.

```python
from random import Random

from scripts import (
    FactoredDirichletBelief,
    TabularDirichletBelief,
    Variable,
)

lock_state = Variable("lock", ("locked", "unlocked"))
door_state = Variable("door", ("closed", "open"))
action = Variable("action", ("open", "close"))

lock_belief = TabularDirichletBelief((lock_state,), prior=1.0)
door_belief = TabularDirichletBelief(
    (door_state,),
    parent_variables=(lock_state, action),
    prior=1.0,
)
world_belief = FactoredDirichletBelief((lock_belief, door_belief))

observation = {
    "current": {"lock": "unlocked", "door": "closed"},
    "next_state": {"lock": "locked", "door": "open"},
    "parents": {"action": "open"},
}
world_belief.update(**observation)

assert door_belief.transition_probability(
    next_state={"door": "open"},
    current={"door": "closed"},
    parents={"lock": "unlocked", "action": "open"},
) == 2.0 / 3.0

mean_world = world_belief.posterior_mean_mdp()
sampled_world = world_belief.sample_mdp(Random(20260723))
```

The joint update projects the door factor's lock parent from the current state:
the observed next lock is deliberately irrelevant to that conditioning row.
Every explicit factored update currently increments every factor once. It does
not silently spend or allocate a learning budget; the two-clock orchestration
layer instead updates retained leaf beliefs separately and explicitly.

See [`docs/reference/beliefs.md`](docs/reference/beliefs.md) for the complete
prior schema, validation contract, deterministic inspection format, ownership
rules, reproducibility requirements, boundary decisions, and rejected
alternatives.

## Finite-horizon planning

`plan_finite_horizon()` performs exact backward induction for a finite model
and caller-supplied terminal reward. For an episode containing exactly \(H\)
actions and transitions,

$$
V_H(s)=R_H(s),
\qquad
Q_t(s,a)=\sum_{s'}P(s'\mid s,a)V_{t+1}(s'),
\qquad
V_t(s)=\max_a Q_t(s,a).
$$

The returned policy is time dependent and fully observed. Exact maximizing
ties use an injected random generator, while unique maximizers consume no
tie-breaking randomness. `evaluate_finite_horizon_policy()` computes the exact
terminal-reward expectation of a fixed policy under the supplied model.

See [`docs/reference/planning.md`](docs/reference/planning.md) for the complete
finite-domain, validation, determinism, and complexity contracts.

## Two-clock learning-budget experiment

The experiment compares two independent cyclic clocks \(x,y\in\mathbb Z_M\)
under the external hand action
`("left", "right", "stay")`. The intended successor of controllable \(x\)
depends on the action. Predictable \(y\) has one configured intended direction
and is truly action invariant. In either factor, the intended next state has
probability \(q\), while every other state has
\((1-q)/(M-1)\).

Both beliefs deliberately contain an independent row for every local
`(state, hand)` context. \(N_x\) and \(N_y\) therefore mean updates delivered
to *each* corresponding local row. Since each factor has \(3M\) rows, a cell
uses

$$
3M(N_x+N_y)
$$

local observations per trial. A rectangular heatmap changes total evidence;
fixed-budget allocation comparisons must follow diagonals satisfying
\(N=N_x+N_y\).

The allocation-sensitivity diagnostic compares three cells at each requested
positive even total \(N\):

- equal allocation at \((N_x,N_y)=(N/2,N/2)\);
- the highest observed success mean below the heatmap diagonal, where
  \(N_x>N_y\); and
- the highest observed success mean above the heatmap diagonal, where
  \(N_x<N_y\).

Tied observed means prefer the cell nearest the diagonal and then canonical
ascending \((N_x,N_y)\). `select_allocation_strategies(summaries,
total_budgets)` returns immutable `AllocationStrategyPoint` records containing
the chosen allocation, binary counts, point estimate, Monte Carlo standard
error, and Wilson 95% bounds. `plot_allocation_sensitivity(points, ...)`
renders two task panels with three ordered strategy lines, pointwise Wilson
error bars, and optional \((N_x,N_y)\) annotations. Its axes are
`Total updates, $N$ (updates/context)` and
`Estimated success probability (unitless)`; the default strategy colors are
accessible blue, vermillion, and green. Solid circles, dashed squares, and
dotted triangles redundantly distinguish equal, below-diagonal, and
above-diagonal strategies.

The “best below” and “best above” curves are post-hoc oracle envelopes:
candidate cells are selected using the same observed \(k/T\) values shown in
the figure. They are therefore subject to winner's curse. Their Wilson
intervals describe the selected cells pointwise and are not adjusted for
selection, so these lines must not be read as confirmatory evidence that a
precommitted allocation policy generalizes.

```python
from scripts import ClockExperimentConfig, run_clock_experiment

config = ClockExperimentConfig(
    num_states=3,
    intended_probability=0.8,
    predictable_direction="right",
    horizon=2,
    trials=2,
    x_updates=(0, 1, 2),
    y_updates=(0, 1, 2),
    fixed_total_budgets=(2,),
    prior_concentration=1.0,
    master_seed=20260723,
)
result = run_clock_experiment(config)

assert len(result.trials) == 2 * 2 * 3 * 3  # tasks × trials × N_x × N_y
```

Each top-level trial samples balanced observations, constructs independent
Dirichlet beliefs, and samples one plausible factor model per local budget.
Navigation plans with the sampled controllable factor; synchronization plans
with the composed sampled factors. Navigation performs one true-environment
rollout per sampled controllable model and records that matched outcome across
all \(N_y\) cells; synchronization performs one rollout per sampled
\((N_x,N_y)\) model pair. Named SHA-256-derived streams provide nested
observation prefixes and matched comparisons. The target-navigation task is an
exact negative control for \(N_y\); synchronization depends on both learned
factors.

The accepted model, estimands, alternatives, and validation criteria are in
the [two-clock allocation reference](docs/reference/two-clock-allocation.md).
Operational contracts live in the module and public API docstrings; exact run
configuration, commands, results, and failures live in the research log.

The notebook provides the canonical exploratory workflow and generates figures
from reloaded measurements. Exact configurations, result counts, runtimes,
artifacts, and validation outcomes are recorded in the
[current research log](docs/logs/2026-07-23-transition-learning-foundations.md).

### Experiment result storage

Experiment runs are local artifacts and are not committed to Git. The
repository-root `results/` directory is ignored and is the canonical local
destination for generated configurations, raw and aggregate measurements,
seed ledgers, manifests, checkpoints, and standalone empirical figures. A
fresh clone therefore contains the experiment code and notebook, not prior run
directories; rerun the notebook or experiment driver to recreate them.

Before relying on a run as durable scientific evidence, archive its complete
verified directory in durable external storage and record the URI, manifest or
checksums, access requirements, and retention policy in the research log.
Small intentional notebook outputs may remain for exposition, but they are not
canonical data.

## Model descriptions

All MDP subclasses inherit `AbstractMDP.__str__`. It returns deterministic,
indented JSON containing the model type, variables, parent variables, and
subclass-specific fields. Tabular models include canonical transition rows;
factored models recursively include their constituent models.

```python
print(world)
description = world.to_dict()
```

The representation is designed for inspection and logging, not as a stable
serialization/deserialization format. See
[`docs/reference/visualization.md`](docs/reference/visualization.md) for the
visualization layer built on the same structural metadata.

## Visualization

Every MDP inherits three visualization methods. `to_networkx()` exposes the
directed graph and its metadata, `draw()` renders it onto a supplied or new
Matplotlib axes without showing or saving it, and `to_graphviz()` returns an
unrendered Graphviz `Digraph`.

```python
graph = world.to_networkx()
ax = world.draw(show_domains=True)
dot = world.to_graphviz(rankdir="LR")
```

The default factor view is a two-slice graph with explicit transition-factor
nodes. The optional `view="dependencies"` removes factor nodes and shows all
possible input-to-output dependencies implied by each factor scope. See
[`docs/reference/visualization.md`](docs/reference/visualization.md) for the
exact node/edge schema and customization options.

## Development workflow

The development extra pins Ruff, mypy, and pre-commit. Install the Git hooks
after creating or activating the development environment:

```bash
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
```

The pre-commit stage applies repository hygiene checks, safe Ruff lint fixes,
Ruff formatting, and strict type checking. The pre-push stage runs the complete
unit suite and compilation checks. Run both stages across the repository before
requesting review:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Hooks can modify files. Review those edits and rerun the commands until every
hook passes. See `AGENTS.md` for commit construction, shared-worktree safety,
and the complete research-production requirements.

GitHub Actions repeats the non-mutating quality gates on Python 3.9 and 3.12,
executes both notebooks from fresh kernels, and builds a wheel. The workflow
has read-only repository permissions and pins third-party actions to immutable
commit identifiers.

## Verification

Run the non-mutating quality checks from the repository root:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
python -m json.tool notebooks/Two_Clock_Allocation_Experiment.ipynb > /dev/null
```

Open either notebook in Jupyter to run the corresponding worked example or
clock allocation experiment.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for the full
terms.
