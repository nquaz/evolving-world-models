# Finite-horizon planning

This page documents the finite-horizon dynamic-programming layer in
`scripts/planning.py`. The minimal fragment below is illustrative because it
intentionally assumes a caller-supplied model and target. Run-specific
integration evidence and quality results belong in the daily research log.

See the
[two-clock allocation reference](two-clock-allocation.md)
for the first experiment that consumes this planner.

## Purpose and boundary

The planning layer computes an exact, time-dependent policy for a finite
transition model and a terminal reward. It is separate from:

- `scripts/mdp.py`, which owns fixed transition semantics;
- `scripts/beliefs.py`, which owns mutable posterior state;
- experiment orchestration, which decides what to observe and which model to
  plan under; and
- evaluation rollouts, which generate sampled binary outcomes in a true
  environment.

The module is standard-library-only. Planning is a read-only operation: it does
not update a belief, sample a transition model, or mutate the supplied MDP.

## Mathematical contract

For state \(s\), action assignment \(a\), terminal reward \(R_H\), and horizon
\(H\), backward induction computes

$$
V_H(s)=R_H(s),
$$

$$
Q_t(s,a)
=
\sum_{s'}P(s'\mid s,a)V_{t+1}(s'),
\qquad
V_t(s)=\max_a Q_t(s,a),
$$

for \(t=H-1,\ldots,0\). There are exactly \(H\) actions and \(H\)
transitions. There are no stage rewards or discount factor in this interface.
The terminal reward is evaluated after the final transition.

All variables in `model.variables` form the state. Every external parent in
`model.parent_variables` is treated as an action variable; the action space is
their finite Cartesian product. Internal cross-factor parents remain part of
the current state and retain the synchronous semantics defined by
`FactoredMDP`.

## Public API

The exported `TerminalReward` type alias is a callable from an `Assignment` to
an `int` or `float`.

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

### `plan_finite_horizon`

`model`
: A finite `AbstractMDP`. Its state and external-parent domains must be
  nonempty and finitely enumerable, and every transition distribution must
  expose finite outcomes through `items()`.

`terminal_reward`
: A side-effect-free callable from a complete state assignment to a finite
  real terminal reward. The task or target may be captured explicitly by the
  callable.

`horizon`
: A nonnegative integer number of actions and transitions. At `horizon=0`,
  the policy contains no action steps and the value is the terminal reward of
  the initial state.

`rng`
: An optional object exposing `random()`. It is consumed only when there are
  multiple exactly maximizing actions; a unique maximum consumes no random
  draw. Supplying a seeded generator makes tie resolution reproducible. The
  planner uses no approximate-equality tolerance when identifying ties.
  Omission creates a private `random.Random` and never consumes module-global
  random state.

The result is an immutable `FiniteHorizonPolicy` containing complete
time-indexed action and value tables over the finite state space. It exposes:

- `variables`: the ordered state-variable tuple;
- `action_variables`: the ordered external-parent tuple;
- `horizon`: the number of action steps;
- `action(step, state) -> Assignment`: the complete external-parent assignment
  selected at that step and state; and
- `value(step, state) -> float`: the planned expected terminal reward.

The assignment returned by `action()` can be passed directly as the `parents`
argument of the MDP transition API. `action()` accepts
`0 <= step < horizon`; `value()` also accepts the terminal step
`step == horizon`. Public queries validate the exact state scope, domain
membership, and step range. The policy is fixed after planning and is suitable
for feedback control: the action at step \(t\) may depend on the state observed
at that step.

### `evaluate_finite_horizon_policy`

This function propagates the supplied fixed policy through `model` exactly
from one complete `initial_state` and returns its expected terminal reward.
It does not sample a rollout. The evaluation model may have different
transition probabilities from the planning model, but its ordered state and
external-parent variable specifications must match the policy exactly.

This exact expectation is useful for contract tests and lower-variance
diagnostics. It does not replace the two-clock experiment's primary metric,
which is a binary true-environment outcome per recorded task/cell/trial. Within
each top-level trial group, navigation performs one rollout per $N_x$ and reuses
that matched outcome across $N_y$; synchronization performs one rollout per
$(N_x,N_y)$ pair.

## Minimal usage

The following fragment assumes `model` is an already constructed finite MDP
whose state includes `x`, and that `target` is in the declared domain:

```python
from random import Random

from scripts.planning import (
    evaluate_finite_horizon_policy,
    plan_finite_horizon,
)


def terminal_reward(state):
    return 1.0 if state["x"] == target else 0.0


policy = plan_finite_horizon(
    model,
    terminal_reward,
    horizon=5,
    rng=Random(20260723),
)
success_probability = evaluate_finite_horizon_policy(
    model,
    policy,
    terminal_reward,
    initial_state={"x": 0},
)
```

The exact import and call sequence above has not yet been run as a standalone
documentation example.

## Validation and failures

Public boundaries reject:

- a negative, Boolean, or non-integral horizon;
- missing or extra state or action keys;
- values outside a declared domain;
- non-finite terminal rewards;
- models with non-finite domains or non-enumerable transition outcomes;
- a policy whose variables, action variables, domains, or horizon do not match
  the evaluation model; and
- invalid time steps in policy queries.

The planner independently validates every enumerated row as nonempty and free
of duplicate outcomes, with finite real probabilities in \([0,1]\) that sum to
one within `rel_tol=1e-9` and `abs_tol=1e-12`. Within one planning or
evaluation invocation, the validated row for a used state/action pair is
cached and reused across time steps. The cache is local to the call and cannot
become stale across model mutations or separate plans. The implementation
does not normalize, repair, or silently drop malformed probabilities.

## Determinism and ordering

State, action, and next-state traversal follows the deterministic variable and
domain order declared by the model. Exact ties are selected uniformly using
the injected random source. Therefore reproducibility requires the same model,
domain ordering, terminal reward, horizon, and RNG state.

The two-clock experiment derives a named tie-breaking stream from its master
seed. Task-specific streams prevent one task's ties from consuming random
draws intended for another task.

## Complexity and limitations

A dense reference implementation performs \(O(|A||S|^2)\) transition-row
construction and validation, followed by

$$
O\!\left(H\,|A|\,|S|^2\right)
$$

expectation arithmetic, and stores the cached rows plus complete time-indexed
action and value tables. Policy evaluation lazily caches only the distinct
state/action rows selected by the policy. In the two-clock world,
\(|S|=M^2\), so dense planning still scales as
\(O(H\,|A|\,M^4)\), while row construction is no longer repeated at every
horizon step.

The interface does not currently cover stage rewards, discounting,
partial observability, continuous spaces, approximate planning, posterior
integration, or online replanning.

## Verification

The focused tests cover the planning contract, including:

- horizons zero and one;
- exact backward-induction values and actions;
- seeded, reproducible uniform selection among exact ties;
- no random draw for a unique maximizing action;
- one empty action assignment for an autonomous model;
- complete feedback tables over every state and time step;
- at most one transition-model query per used state/action row within each
  planning or evaluation invocation;
- invalid states, actions, steps, rewards, and horizons;
- exact fixed-policy evaluation;
- evaluation under true dynamics that differ from the planning dynamics;
- agreement between a policy's planned value and its evaluation in the same
  model; and
- model/policy scope mismatches and malformed enumerable distributions.

Run the focused suite from the repository root:

```bash
python -m unittest tests.test_planning -v
```

Run the complete repository quality pipeline documented in `AGENTS.md` before
relying on a release. Exact command outcomes, supported-Python checks, notebook
executions, and packaging evidence are recorded in the corresponding research
log rather than duplicated here.
