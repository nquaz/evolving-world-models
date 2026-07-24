# Conjugate beliefs for finite transition models

The belief layer represents uncertainty about finite transition tables without
making the transition models themselves mutable. For each conditioning context
$c=(x,\mathrm{pa}(x))$ and joint next-state outcome $y$:

$$
\theta_c \sim \mathrm{Dir}(\alpha_c),
\qquad
p(x'=y\mid c,D)
=
\frac{\alpha_{c,y}+n_{c,y}}
{\sum_z(\alpha_{c,z}+n_{c,z})}.
$$

Here, $\alpha_{c,y}>0$ is a prior concentration and $n_{c,y}$ is the integer
number of matching observed transitions. The implementation is
standard-library-only and supports finite, fully enumerated variable domains.

This page is the canonical conceptual and API reference for the accepted
interface in [`scripts/beliefs.py`](../../scripts/beliefs.py).

## Design rationale and alternatives

The belief layer is deliberately separate from the fixed transition kernels in
`scripts/mdp.py`. A mutable posterior is not itself one fixed environment
hypothesis, so callers obtain explicit posterior-mean or sampled MDP snapshots
when a planner needs transition semantics.

The implementation retains prior concentrations and integer observation counts
separately rather than storing posterior concentrations alone. That preserves
the provenance of each conjugate update and permits exact defensive inspection.
Complete finite row support matches `TabularMDP` validation and avoids an
ambiguous sparse-prior contract.

NumPy and SciPy were not added to the transition core: standard-library gamma
draws are sufficient for the finite Dirichlet sampler and preserve the core's
dependency boundary. Selective factor updates are also intentionally absent.
Choosing which factor receives a finite learning budget is an experiment-design
decision owned by an explicit orchestration layer, not hidden belief behavior.
Sparse support, custom belief factors, forgetting, and hierarchical priors
remain deferred until an experiment requires their contracts.

## Installation

The belief layer has no optional runtime dependencies:

```bash
python -m pip install -e .
```

Install the development extra to run tests and quality checks:

```bash
python -m pip install -e '.[dev]'
```

## Public API

The three public classes are available from the package root:

```python
from scripts import (
    AbstractTransitionBelief,
    FactoredDirichletBelief,
    TabularDirichletBelief,
)
```

`AbstractTransitionBelief` is a mutable belief abstraction, not an
`AbstractMDP` subclass. It deliberately uses the transition model's query names
for its posterior-predictive distribution:

```python
belief.transition_distribution(current, parents=None)
belief.transition_probability(next_state, current, parents=None)
belief.transition_log_probability(next_state, current, parents=None)
belief.update(current, next_state, parents=None)
belief.posterior_mean_mdp()
belief.sample_mdp(rng=None)
belief.to_dict()
str(belief)
```

`update()` follows temporal order: current state, next state, then optional
parents. Keyword arguments are recommended when passing a complete
observation:

```python
belief.update(
    current={"state": "before"},
    next_state={"state": "after"},
    parents={"action": "advance"},
)
```

All assignments must contain exactly the required variable names and values
from their declared domains. Predictive queries do not update the belief.

## Tabular beliefs

Construct a tabular belief from its predicted variables, optional parent
variables, and prior:

```python
belief = TabularDirichletBelief(
    variables=(state_variable,),
    parent_variables=(action_variable,),
    prior=1.0,
)
```

Every predicted variable and parent must have a finite, nonempty domain. A
multi-variable predicted scope has one Dirichlet distribution over the full
Cartesian joint next-state support; its variables are not silently treated as
independent.

### Scalar symmetric prior

A real scalar prior assigns the same concentration to every next-state outcome
in every conditioning context. For a binary next-state support, `prior=1.0`
therefore gives prior-predictive probability $1/2$ to each outcome.

The scalar must be numeric, finite, and strictly positive. Boolean values and
strings are rejected rather than coerced into concentrations.

### Explicit concentration rows

An iterable prior uses the same outer row shape as `TabularMDP`:

```python
(
    current_assignment,
    parent_assignment,
    (
        (next_assignment, positive_concentration),
        ...
    ),
)
```

For example, concentrations in a $19:1$ ratio express a prior mean of $0.95$
for retaining the lock's current setting:

```python
from scripts import TabularDirichletBelief, Variable

lock_variable = Variable("lock", ("locked", "unlocked"))

persistent_lock_prior = (
    (
        {"lock": "locked"},
        {},
        (
            ({"lock": "locked"}, 19.0),
            ({"lock": "unlocked"}, 1.0),
        ),
    ),
    (
        {"lock": "unlocked"},
        {},
        (
            ({"lock": "locked"}, 1.0),
            ({"lock": "unlocked"}, 19.0),
        ),
    ),
)

lock_belief = TabularDirichletBelief(
    variables=(lock_variable,),
    prior=persistent_lock_prior,
)
```

An explicit prior must contain every current/parent context exactly once and
every Cartesian joint next-state outcome exactly once within each row. Missing,
extra, or duplicate contexts and outcomes are errors. Each concentration and
each complete row sum must be finite and strictly positive.

Concentrations are evidence weights, not probabilities; they need not sum to
one. Multiplying every concentration in a row by the same constant preserves
the prior mean while changing its strength.

## Updates and posterior prediction

A valid tabular update increments one count:

```text
counts[(current, parents)][next_state] += 1
```

Prior concentrations remain unchanged. The posterior-predictive probability is
computed from prior plus observed counts:

$$
p(x'=y\mid x,\mathrm{pa}(x),D)
=
\frac{\alpha_{c,y}+n_{c,y}}
{\sum_z(\alpha_{c,z}+n_{c,z})}.
$$

Predictions cover the complete Cartesian next-state support in declared
variable and domain order. Invalid observations are rejected before mutation,
so a failed update leaves all counts unchanged.

`transition_distribution()` returns a `TransitionDistribution` over the
posterior-predictive row. `transition_probability()` evaluates one outcome, and
`transition_log_probability()` provides the corresponding log probability.

### Extreme concentration ratios

Ordinary probabilities are Python floats. If one positive posterior
concentration is extremely small relative to a finite row total, normalizing
it can underflow to `0.0`. The belief still knows that the outcome has positive
Dirichlet mass: `transition_log_probability()` computes

$$
\log(\alpha_{c,y}+n_{c,y})
-
\log\left(\sum_z(\alpha_{c,z}+n_{c,z})\right)
$$

before float normalization and therefore remains finite when both terms are
finite. Use the log query for numerically extreme comparisons. A returned
ordinary probability of zero can thus be a numeric underflow rather than a
structural zero; complete Dirichlet rows have no structural zeros because all
concentrations are strictly positive.

## Inspecting concentrations and counts

`TabularDirichletBelief` exposes:

```python
belief.prior_concentrations
belief.counts
belief.posterior_concentrations
```

Each property returns a fresh nested dictionary:

```text
{
    (current_assignment, parent_assignment): {
        next_assignment: value,
        ...
    },
    ...
}
```

The assignment keys are immutable `Assignment` objects. Prior and posterior
concentrations are floats; counts are integers. Mutating a returned outer or
inner dictionary cannot change the belief.

`to_dict()` and `str(belief)` provide deterministic JSON-like inspection
descriptions. They enumerate contexts and outcomes in canonical declared order
and include prior concentration, count, and posterior concentration. These
representations are intended for inspection and logging, not as versioned
storage or deserialization formats.

## Posterior-mean snapshots

`posterior_mean_mdp()` materializes the current posterior predictive as a fixed
transition model:

- a `TabularDirichletBelief` returns a new `TabularMDP`; and
- a `FactoredDirichletBelief` returns a new `FactoredMDP` containing new local
  `TabularMDP` snapshots.

Every call returns independent model objects and transition tables. Later
belief updates cannot change an earlier snapshot.

`TabularMDP` rows store normalized floats and do not retain the originating
concentrations. Consequently, an extreme positive mass that underflows in
`transition_probability()` is also zero in a posterior-mean snapshot, whose
generic log query then returns `-inf`. Query the belief directly when preserved
concentration-space log mass matters.

## Sampling plausible transition models

`sample_mdp(rng)` independently samples each complete transition row from its
Dirichlet posterior. For each outcome $y$ in a row:

$$
g_y \sim \mathrm{Gamma}(\alpha_{c,y}+n_{c,y},1),
\qquad
\theta_{c,y}=\frac{g_y}{\sum_z g_z}.
$$

Pass a caller-owned, seeded `random.Random` for reproducibility:

```python
from random import Random

sampled_mdp = belief.sample_mdp(Random(20260723))
```

The RNG must expose `gammavariate(shape, scale)`. Passing `None` creates one
private `random.Random()` instance; the module never uses or seeds
module-global randomness.

Rows are traversed in canonical context/outcome order. Reproduction requires
the same initial RNG state, code, Python implementation, and traversal order.
Each sampled row is normalized with `math.fsum`. Nonnumeric, negative, or
non-finite gamma draws and all-zero or non-finite row sums raise errors rather
than producing an invalid MDP. An individual zero draw is permitted because a
small valid concentration may underflow in finite precision.

Sampling an MDP draws one fixed, plausible transition model. It is distinct
from sampling a next state from the posterior-predictive distribution.

## Factored beliefs

A factored belief combines tabular leaf beliefs with disjoint predicted scopes:

```python
world_belief = FactoredDirichletBelief(
    factors=(lock_belief, door_belief),
)
```

This first interface accepts `TabularDirichletBelief` factors only. Repeated
variable specifications must agree, factor output scopes must be disjoint, and
external parents use deterministic first-seen order matching `FactoredMDP`.

The `factors` property is an immutable tuple, but its entries are the original
mutable belief objects. Updating a retained factor directly changes the
corresponding component of the factored belief. Use
`posterior_mean_mdp()` or `sample_mdp()` when an independent fixed model is
required.

### Posterior prediction

For a fixed current state and external-parent assignment, the joint
posterior-predictive distribution is the product of local predictive
distributions:

$$
p(x'\mid x,\mathrm{pa}(x),D)
=
\prod_i p_i(x_i'\mid x_i,\mathrm{pa}(x_i),D).
$$

As with `FactoredMDP`, only parents not predicted by any factor are supplied as
external parents.

### Atomic synchronous updates

`FactoredDirichletBelief.update()` validates the complete joint current state,
joint next state, and external-parent assignment before changing any count.
Each factor receives:

- its current variables projected from the joint current state;
- its next variables projected from the observed joint next state; and
- each internal cross-factor parent from the joint **current** state.

An internal parent is never read from the observed next state. After all local
targets validate, the method increments every factor exactly once. A validation
failure leaves all factors unchanged.

## Runnable lock/door example

This example uses symmetric priors to make one conjugate update easy to inspect.
The concentrations represent uncertainty about the transition tables, not the
canonical lock/door environment's ground-truth probabilities.

```python
from math import isclose
from random import Random

from scripts import (
    FactoredDirichletBelief,
    TabularDirichletBelief,
    Variable,
)

lock_variable = Variable("lock", ("locked", "unlocked"))
door_variable = Variable("door", ("closed", "open"))
action_variable = Variable("action", ("open", "close"))

lock_belief = TabularDirichletBelief(
    variables=(lock_variable,),
    prior=1.0,
)
door_belief = TabularDirichletBelief(
    variables=(door_variable,),
    parent_variables=(lock_variable, action_variable),
    prior=1.0,
)
world_belief = FactoredDirichletBelief((lock_belief, door_belief))

observation = {
    "current": {"lock": "unlocked", "door": "closed"},
    "next_state": {"lock": "locked", "door": "open"},
    "parents": {"action": "open"},
}
world_belief.update(**observation)

# The door row uses the current lock setting, not the observed next setting.
door_open_given_current_unlocked = door_belief.transition_probability(
    next_state={"door": "open"},
    current={"door": "closed"},
    parents={"lock": "unlocked", "action": "open"},
)
door_open_given_current_locked = door_belief.transition_probability(
    next_state={"door": "open"},
    current={"door": "closed"},
    parents={"lock": "locked", "action": "open"},
)
assert isclose(door_open_given_current_unlocked, 2.0 / 3.0)
assert isclose(door_open_given_current_locked, 1.0 / 2.0)

# Both updated binary factors assign probability 2/3 to the observation.
joint_probability = world_belief.transition_probability(
    next_state=observation["next_state"],
    current=observation["current"],
    parents=observation["parents"],
)
assert isclose(joint_probability, 4.0 / 9.0)

mean_world = world_belief.posterior_mean_mdp()
sampled_world = world_belief.sample_mdp(Random(20260723))

# A later update changes the belief, but not the independent mean snapshot.
world_belief.update(**observation)
assert isclose(
    mean_world.transition_probability(
        next_state=observation["next_state"],
        current=observation["current"],
        parents=observation["parents"],
    ),
    4.0 / 9.0,
)
assert sampled_world.variables == world_belief.variables
```

The regression-critical detail is the door update: although the observed next
lock is `locked`, the door count changes only in the context whose parent lock
is current `unlocked`.

The Cartesian joint state space intentionally permits
`{"lock": "locked", "door": "open"}`. Here, `lock` is the latch mechanism's
setting, which can be engaged while the door is open. Forbidding that joint
state would require a separate constrained-state abstraction.

## Validation and errors

Construction rejects:

- variables or parents without finite, nonempty domains;
- duplicate or conflicting variable specifications;
- overlapping factor output scopes;
- empty factorizations or non-tabular factors;
- Boolean, nonnumeric, non-finite, zero, or negative concentrations;
- incomplete, extra, or duplicate prior contexts or outcomes; and
- explicit concentration rows whose finite sum cannot be represented.

Queries and updates reject missing or extra assignment keys and values outside
declared domains. A factored observation is validated atomically before any
constituent count changes.

Sampling rejects an RNG without a callable `gammavariate()` method and invalid
gamma results. No validation path silently coerces malformed scientific inputs
or repairs an invalid sampled row.

## Determinism and side effects

Context, outcome, factor, variable, and parent ordering follows declarations and
finite domain order. Equivalent explicit prior rows yield the same inspection
description regardless of their input row order.

Importing the belief module performs no updates, file writes, plotting,
network access, or process-global random seeding. Constructing and querying a
belief has no external side effects. `update()` mutates only the explicitly
targeted belief counts.

## Complexity

Let $C$ be the number of current/parent contexts and $K$ the number of joint
next-state outcomes:

- initialization and storage use $O(CK)$ time and space;
- a posterior-predictive row query uses $O(K)$ time;
- one tabular update uses $O(1)$ table access after assignment validation; and
- full inspection, mean-snapshot construction, and MDP sampling use $O(CK)$
  time.

Factored costs sum the corresponding local costs. Enumerating all items of a
joint product distribution additionally scales with the Cartesian product of
the factor supports.

## Limitations and ownership boundaries

The initial implementation intentionally excludes:

- continuous or unbounded variables;
- sparse transition support and constrained joint-state spaces;
- forgetting, weighted observations, and hierarchical priors;
- durable belief serialization;
- custom non-tabular belief factors;
- rewards, objectives, policies, planning, and interaction orchestration; and
- selective updates under a finite learning budget.

The last limitation is scientifically important. One explicit joint factored
update increments every factor, so this belief layer alone does not model an
allocation-of-learning-capacity problem. A later learning/orchestration
interface must make interaction and update budgets explicit and decide which
observations or factors receive updates.

## Verification

Run the focused belief tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_beliefs.py' -v
```

Run the complete repository quality pipeline before relying on a release:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```
