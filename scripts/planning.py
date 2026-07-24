"""Exact finite-horizon planning for finite transition models.

This module is the control layer above :mod:`scripts.mdp`.  Transition models
continue to own only conditional dynamics; callers supply a terminal reward
when they request a plan.  Every external parent of the supplied model is
treated as an action variable, so a transition is queried as
``model.transition_distribution(state, action)``.

For a horizon of ``H`` actions, :func:`plan_finite_horizon` computes

``V_H(s) = R_H(s)``

and, working backward,

``Q_t(s, a) = sum_{s'} P(s' | s, a) V_{t+1}(s')`` and
``V_t(s) = max_a Q_t(s, a)``.

The returned :class:`FiniteHorizonPolicy` contains an action for every finite
state at steps ``0`` through ``H - 1`` and a value for every state at steps
``0`` through ``H``.  For example::

    plan = plan_finite_horizon(
        model,
        terminal_reward=lambda state: float(state["position"] == 2),
        horizon=4,
        rng=random.Random(7),
    )
    action = plan.action(0, {"position": 0})

State and action domains must be finite and nonempty, transition
distributions must support exact enumeration through ``items()``, and terminal
rewards must be finite real numbers.  Variable, domain, state, action, and
outcome order follow their declaration order.  Exact maximizing ties are
selected uniformly with the injected generator's ``random()`` method; no
approximate equality tolerance is used.  Omitting ``rng`` creates a private
``random.Random`` instance and never consumes module-global random state.

Planning and evaluation mutate no model or policy and perform no I/O.  The
module uses only the Python standard library and the dependency-free
transition interfaces.  Each validated transition row is cached once per
invocation.  With ``S`` states, ``A`` joint actions, ``K`` enumerated next
outcomes, and horizon ``H``, planning performs ``O(SAK)`` row construction
plus ``O(HSAK)`` arithmetic and stores ``O(HS + SAK)`` policy, value, and row
state.  Policy evaluation performs ``O(HSK)`` arithmetic after constructing
only the distinct selected rows.  This is a readable exact reference
implementation; sparse, approximate, discounted, partially observable, and
infinite-horizon planning are explicit non-goals.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from itertools import product
from numbers import Real
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .mdp import AbstractMDP, Assignment, Variable

# ``numbers.Real`` is the runtime validation contract, but typeshed does not
# model built-in ``float`` as a nominal subtype of ``numbers.Real``. The public
# callable alias therefore names the ordinary scalar types accepted by the
# current finite-task APIs while runtime validation still handles other Real
# implementations correctly when called dynamically.
TerminalReward = Callable[[Assignment], Union[int, float]]
_TimeState = Tuple[int, Assignment]
_TransitionKey = Tuple[Assignment, Assignment]
_TransitionRow = Tuple[Tuple[Assignment, float], ...]


def _validate_horizon(horizon: object) -> int:
    """Return a valid count of actions and transitions."""

    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be a nonnegative integer.")
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    return horizon


def _finite_assignments(
    variables: Sequence[Variable],
    *,
    label: str,
) -> Tuple[Assignment, ...]:
    """Enumerate assignments in declared variable and domain order."""

    domains = []
    for variable in variables:
        if variable.domain is None:
            raise ValueError(
                "{} requires a finite domain for variable {!r}.".format(
                    label, variable.name
                )
            )
        domains.append(variable.domain)

    return tuple(
        Assignment(
            {
                variable.name: value
                for variable, value in zip(variables, assignment_values)
            }
        )
        for assignment_values in product(*domains)
    )


def _canonical_assignment(
    values: Mapping[str, Hashable],
    variables: Sequence[Variable],
    *,
    label: str,
) -> Assignment:
    """Validate and order one state or action assignment."""

    if not isinstance(values, MappingABC):
        raise TypeError(
            "{} must be a mapping from variable names to values.".format(label)
        )

    expected_names = tuple(variable.name for variable in variables)
    expected = set(expected_names)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing {}".format(missing))
        if extra:
            details.append("unexpected {}".format(extra))
        raise ValueError("{} has {}.".format(label, " and ".join(details)))

    canonical: Dict[str, Hashable] = {}
    for variable in variables:
        value = values[variable.name]
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(
                "{} value for {!r} must be hashable.".format(label, variable.name)
            ) from error
        if not variable.accepts(value):
            raise ValueError(
                "{} value {!r} is outside the domain of {!r}: {!r}.".format(
                    label, value, variable.name, variable.domain
                )
            )
        canonical[variable.name] = value
    return Assignment(canonical)


def _terminal_reward_value(
    terminal_reward: TerminalReward,
    state: Assignment,
) -> float:
    """Evaluate and validate one terminal reward."""

    value: object = terminal_reward(state)
    if not isinstance(value, Real):
        raise TypeError(
            "terminal_reward must return a finite real number; received {!r} "
            "for state {!r}.".format(value, state)
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "terminal_reward must return a finite real number for state {!r}.".format(
                state
            )
        ) from error
    if not math.isfinite(numeric):
        raise ValueError(
            "terminal_reward must return a finite real number; received {!r} "
            "for state {!r}.".format(value, state)
        )
    return numeric


def _enumerated_items(
    model: AbstractMDP,
    state: Assignment,
    action: Assignment,
) -> _TransitionRow:
    """Return one validated finite transition row."""

    distribution = model.transition_distribution(state, action)
    try:
        raw_items = distribution.items()
    except TypeError as error:
        raise TypeError(
            "Finite-horizon planning requires transition distributions with "
            "finitely enumerable items()."
        ) from error

    if not raw_items:
        raise ValueError("An enumerable transition distribution cannot be empty.")

    items: List[Tuple[Assignment, float]] = []
    seen = set()
    for raw_outcome, raw_probability in raw_items:
        outcome = _canonical_assignment(
            raw_outcome,
            model.variables,
            label="Enumerated next state",
        )
        if outcome in seen:
            raise ValueError(
                "An enumerable transition distribution contains duplicate "
                "outcome {!r}.".format(outcome)
            )
        seen.add(outcome)

        probability_candidate: object = raw_probability
        if isinstance(probability_candidate, bool) or not isinstance(
            probability_candidate, Real
        ):
            raise TypeError(
                "Enumerated transition probabilities must be finite real numbers."
            )
        try:
            probability = float(probability_candidate)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "Enumerated transition probabilities must be finite and lie in [0, 1]."
            ) from error
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(
                "Enumerated transition probabilities must be finite and lie in [0, 1]."
            )
        items.append((outcome, probability))

    total = math.fsum(probability for _, probability in items)
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "Enumerated transition probabilities must sum to 1; received "
            "{:.17g}.".format(total)
        )
    return tuple(items)


def _cached_transition_row(
    model: AbstractMDP,
    state: Assignment,
    action: Assignment,
    rows: Dict[_TransitionKey, _TransitionRow],
) -> _TransitionRow:
    """Return a row that is validated at most once in this invocation."""

    key = (state, action)
    if key not in rows:
        rows[key] = _enumerated_items(model, state, action)
    return rows[key]


def _expected_value(
    row: _TransitionRow,
    next_values: Mapping[Assignment, float],
) -> float:
    """Compute one finite transition-row expectation."""

    terms = []
    for next_state, probability in row:
        term = probability * next_values[next_state]
        if not math.isfinite(term):
            raise ValueError(
                "A transition expectation overflowed to a non-finite value."
            )
        terms.append(term)
    try:
        result = math.fsum(terms)
    except OverflowError as error:
        raise ValueError(
            "A transition expectation overflowed to a non-finite value."
        ) from error
    if not math.isfinite(result):
        raise ValueError("A transition expectation produced a non-finite value.")
    return result


def _random_tie_index(rng: Any, count: int) -> int:
    """Select uniformly from ``count`` exact ties using ``rng.random()``."""

    draw = rng.random()
    if isinstance(draw, bool) or not isinstance(draw, Real):
        raise ValueError("rng.random() must return a real number in [0, 1).")
    try:
        numeric_draw = float(draw)
    except (OverflowError, ValueError) as error:
        raise ValueError("rng.random() must return a real number in [0, 1).") from error
    if not math.isfinite(numeric_draw) or not 0.0 <= numeric_draw < 1.0:
        raise ValueError("rng.random() must return a real number in [0, 1).")
    return min(int(numeric_draw * count), count - 1)


@dataclass(frozen=True)
class FiniteHorizonPolicy:
    """An immutable complete finite-horizon feedback policy and value table.

    Policies are created by :func:`plan_finite_horizon`.  State queries require
    exactly the declared model state scope.  :meth:`action` accepts steps
    ``0 <= step < horizon`` and returns the complete external-parent
    assignment.  :meth:`value` also accepts the terminal step ``horizon``.
    """

    _variables: Tuple[Variable, ...] = field(repr=False)
    _action_variables: Tuple[Variable, ...] = field(repr=False)
    _horizon: int
    _actions: Mapping[_TimeState, Assignment] = field(repr=False)
    _values: Mapping[_TimeState, float] = field(repr=False)

    def __post_init__(self) -> None:
        """Defensively own the tables supplied by the planner."""

        object.__setattr__(self, "_variables", tuple(self._variables))
        object.__setattr__(self, "_action_variables", tuple(self._action_variables))
        object.__setattr__(self, "_horizon", _validate_horizon(self._horizon))
        object.__setattr__(
            self,
            "_actions",
            MappingProxyType(dict(self._actions)),
        )
        object.__setattr__(
            self,
            "_values",
            MappingProxyType(dict(self._values)),
        )

    @property
    def variables(self) -> Tuple[Variable, ...]:
        """State variables in their deterministic planning order."""

        return self._variables

    @property
    def action_variables(self) -> Tuple[Variable, ...]:
        """External-parent variables treated as the joint action."""

        return self._action_variables

    @property
    def horizon(self) -> int:
        """Number of actions and transitions in the policy."""

        return self._horizon

    def action(
        self,
        step: int,
        state: Mapping[str, Hashable],
    ) -> Assignment:
        """Return the selected joint action for ``state`` at ``step``.

        Args:
            step: Zero-based decision step, strictly less than ``horizon``.
            state: Complete assignment to :attr:`variables`.

        Returns:
            The complete assignment to :attr:`action_variables`.

        Raises:
            TypeError: If ``step`` is not an integer or ``state`` is not a
                mapping.
            ValueError: If the step or state is outside the policy contract.
        """

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer.")
        if not 0 <= step < self.horizon:
            raise ValueError(
                "action step must satisfy 0 <= step < {}; received {}.".format(
                    self.horizon, step
                )
            )
        canonical_state = _canonical_assignment(
            state,
            self.variables,
            label="Policy state",
        )
        return self._actions[(step, canonical_state)]

    def value(
        self,
        step: int,
        state: Mapping[str, Hashable],
    ) -> float:
        """Return the optimal expected terminal reward at ``step`` and ``state``.

        Args:
            step: Zero-based step in the inclusive interval ``[0, horizon]``.
            state: Complete assignment to :attr:`variables`.

        Returns:
            The stored finite optimal value.

        Raises:
            TypeError: If ``step`` is not an integer or ``state`` is not a
                mapping.
            ValueError: If the step or state is outside the policy contract.
        """

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer.")
        if not 0 <= step <= self.horizon:
            raise ValueError(
                "value step must satisfy 0 <= step <= {}; received {}.".format(
                    self.horizon, step
                )
            )
        canonical_state = _canonical_assignment(
            state,
            self.variables,
            label="Policy state",
        )
        return self._values[(step, canonical_state)]


def plan_finite_horizon(
    model: AbstractMDP,
    terminal_reward: TerminalReward,
    horizon: int,
    *,
    rng: Optional[Any] = None,
) -> FiniteHorizonPolicy:
    """Compute an exact time-dependent policy by backward induction.

    Args:
        model: Finite transition model.  Its external parents are treated as
            action variables.
        terminal_reward: Callable returning a finite real value for every
            complete state after exactly ``horizon`` transitions.
        horizon: Nonnegative number of actions and transitions.
        rng: Optional object exposing ``random()`` for uniform exact-tie
            selection.  Omission creates a private generator.

    Returns:
        An immutable complete policy and optimal value table.

    Raises:
        TypeError: If an interface or returned value has the wrong type, or if
            a transition distribution is not finitely enumerable.
        ValueError: If a domain, horizon, reward, random draw, transition row,
            or computed expectation violates the finite planning contract.

    Notes:
        Tie randomization occurs while state-time pairs are visited in reverse
        time and declared state order.  Reproduction therefore requires the
        same model ordering, horizon, reward, and random stream.
    """

    if not isinstance(model, AbstractMDP):
        raise TypeError("model must be an AbstractMDP.")
    if not callable(terminal_reward):
        raise TypeError("terminal_reward must be callable.")
    validated_horizon = _validate_horizon(horizon)

    generator = random.Random() if rng is None else rng
    if not callable(getattr(generator, "random", None)):
        raise TypeError("rng must expose a callable random() method.")

    states = _finite_assignments(model.variables, label="Finite-horizon planning")
    actions = _finite_assignments(
        model.parent_variables,
        label="Finite-horizon planning",
    )

    policy_actions: Dict[_TimeState, Assignment] = {}
    policy_values: Dict[_TimeState, float] = {}
    next_values = {
        state: _terminal_reward_value(terminal_reward, state) for state in states
    }
    for state, value in next_values.items():
        policy_values[(validated_horizon, state)] = value

    transition_rows: Dict[_TransitionKey, _TransitionRow] = {}
    for step in reversed(range(validated_horizon)):
        current_values: Dict[Assignment, float] = {}
        for state in states:
            action_values = tuple(
                (
                    action,
                    _expected_value(
                        _cached_transition_row(
                            model,
                            state,
                            action,
                            transition_rows,
                        ),
                        next_values,
                    ),
                )
                for action in actions
            )
            best_value = max(value for _, value in action_values)
            maximizing_actions = tuple(
                action for action, value in action_values if value == best_value
            )
            if len(maximizing_actions) == 1:
                selected = maximizing_actions[0]
            else:
                selected = maximizing_actions[
                    _random_tie_index(generator, len(maximizing_actions))
                ]
            current_values[state] = best_value
            policy_actions[(step, state)] = selected
            policy_values[(step, state)] = best_value
        next_values = current_values

    return FiniteHorizonPolicy(
        tuple(model.variables),
        tuple(model.parent_variables),
        validated_horizon,
        policy_actions,
        policy_values,
    )


def evaluate_finite_horizon_policy(
    model: AbstractMDP,
    policy: FiniteHorizonPolicy,
    terminal_reward: TerminalReward,
    initial_state: Mapping[str, Hashable],
) -> float:
    """Evaluate a fixed finite-horizon policy exactly under ``model``.

    Args:
        model: Finite transition model used for evaluation.
        policy: Complete feedback policy, often planned under another model.
        terminal_reward: Callable returning the finite reward after the
            policy's final transition.
        initial_state: Complete state assignment at step zero.

    Returns:
        Expected terminal reward under ``model`` from ``initial_state``.

    Raises:
        TypeError: If an interface or returned value has the wrong type, or if
            a transition distribution is not finitely enumerable.
        ValueError: If model and policy scopes differ or any finite planning
            invariant is violated.

    Notes:
        The policy and model may have different transition probabilities, but
        their ordered state and external-parent variable specifications must
        match exactly.
    """

    if not isinstance(model, AbstractMDP):
        raise TypeError("model must be an AbstractMDP.")
    if not isinstance(policy, FiniteHorizonPolicy):
        raise TypeError("policy must be a FiniteHorizonPolicy.")
    if not callable(terminal_reward):
        raise TypeError("terminal_reward must be callable.")
    if tuple(model.variables) != policy.variables:
        raise ValueError("model state variables must match policy variables exactly.")
    if tuple(model.parent_variables) != policy.action_variables:
        raise ValueError(
            "model parent variables must match policy action variables exactly."
        )

    states = _finite_assignments(
        model.variables,
        label="Finite-horizon policy evaluation",
    )
    _finite_assignments(
        model.parent_variables,
        label="Finite-horizon policy evaluation",
    )
    canonical_initial = _canonical_assignment(
        initial_state,
        model.variables,
        label="Initial state",
    )

    next_values = {
        state: _terminal_reward_value(terminal_reward, state) for state in states
    }
    transition_rows: Dict[_TransitionKey, _TransitionRow] = {}
    for step in reversed(range(policy.horizon)):
        next_values = {
            state: _expected_value(
                _cached_transition_row(
                    model,
                    state,
                    policy.action(step, state),
                    transition_rows,
                ),
                next_values,
            )
            for state in states
        }
    return next_values[canonical_initial]


__all__ = [
    "FiniteHorizonPolicy",
    "TerminalReward",
    "evaluate_finite_horizon_policy",
    "plan_finite_horizon",
]
