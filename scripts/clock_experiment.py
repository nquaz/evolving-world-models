"""Reproducible finite-budget learning experiments for two cyclic clocks.

This module is the standard-library orchestration layer for the repository's
first downstream-control experiment. It builds two independent ``M``-state
clock factors:

* ``x`` is controllable, with the `hand` action selecting the intended
  left, right, or stationary successor; and
* ``y`` is predictable but uncontrollable, with one experimenter-selected
  intended direction and true transition rows that are identical across hand
  actions.

For either factor, the intended next state has probability ``q`` and every
other state has probability ``(1 - q) / (M - 1)``. Although the true ``y``
kernel is action invariant, its belief deliberately retains a distinct
Dirichlet row for every ``(y, hand)`` context. Both beliefs use a symmetric
per-outcome concentration, ``alpha=1`` by default.

The budgets ``N_x`` and ``N_y`` count observations delivered to *each*
factor-local ``(state, action)`` context. Because there are three actions and
``M`` states, one cell consumes ``3*M*(N_x + N_y)`` local observations per
trial. Balanced observations are generated directly from each true local row;
they are not a trajectory-interaction budget. The leaf beliefs are updated
separately so finite learning capacity can be allocated between factors.

Each trial uses stable SHA-256-derived random streams. Observation samples are
nested prefixes across increasing budgets. A sampled ``x`` factor depends only
on ``(trial, N_x)`` and a sampled ``y`` factor only on ``(trial, N_y)``.
Initial states, task targets, and true-environment random streams are matched
across cells. This common-random-number design makes target-navigation
invariance to ``N_y`` an exact negative-control contract.

The public entry points are :class:`ClockExperimentConfig`,
:func:`build_clock_world`, :func:`build_clock_beliefs`,
:func:`run_clock_experiment`, :func:`select_allocation_strategies`,
:func:`write_clock_experiment`,
:func:`read_trial_csv`, :func:`read_summary_csv`,
:func:`write_run_manifest`, and :func:`verify_run_manifest`. Reusable dynamic
programming lives in :mod:`scripts.planning`; plotting lives in the optional
:mod:`scripts.experiment_plotting` adapter.

Running an experiment mutates only newly constructed in-memory belief counts.
It performs no filesystem, plotting, network, or process-global side effects.
Artifact writing is a separate explicit call. Run directories are unique and
never overwritten, individual files are atomically replaced during initial
construction, and a versioned manifest records byte sizes and SHA-256 hashes.
Every valid run manifest requires the canonical configuration, raw trials,
summaries, and seed ledger; refreshing a manifest cannot bless an incomplete
run.

All stochastic operations accept the configuration's explicit master seed and
use private ``random.Random`` instances. Reproducibility additionally depends
on the Python implementation and the stable variable, domain, context, and
outcome order. Exact dense joint planning scales as ``O(H * |A| * M**4)`` per
sampled synchronization model; this readable reference implementation does not
preemptively add an optimized numerical dependency.

The module supports Python 3.9 and uses only the Python standard library plus
the repository's dependency-free transition, belief, and planning modules.
It does not implement online learning, Bayes-adaptive planning, hierarchical
priors, pooled translation invariance, partial observability, or scientific
claims from the bounded demonstration notebook.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from numbers import Real
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .beliefs import FactoredDirichletBelief, TabularDirichletBelief
from .mdp import Assignment, FactoredMDP, TabularMDP, Variable
from .planning import (
    FiniteHorizonPolicy,
    TerminalReward,
    evaluate_finite_horizon_policy,
    plan_finite_horizon,
)

CONFIG_SCHEMA_VERSION = "two-clock-config-v1"
TRIAL_SCHEMA_VERSION = "two-clock-trials-v1"
SUMMARY_SCHEMA_VERSION = "two-clock-summary-v1"
SEED_SCHEMA_VERSION = "two-clock-seeds-v1"
MANIFEST_SCHEMA_VERSION = "two-clock-manifest-v1"

NAVIGATION_TASK = "navigation"
SYNCHRONIZATION_TASK = "synchronization"
TASK_ORDER = (NAVIGATION_TASK, SYNCHRONIZATION_TASK)
EQUAL_ALLOCATION_STRATEGY = "equal"
BEST_BELOW_DIAGONAL_STRATEGY = "best_below_diagonal"
BEST_ABOVE_DIAGONAL_STRATEGY = "best_above_diagonal"
ALLOCATION_STRATEGY_ORDER = (
    EQUAL_ALLOCATION_STRATEGY,
    BEST_BELOW_DIAGONAL_STRATEGY,
    BEST_ABOVE_DIAGONAL_STRATEGY,
)
HAND_ACTIONS = ("left", "right", "stay")
_WILSON_95_Z = 1.959963984540054
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED_RUN_ARTIFACTS = (
    "config.json",
    "trials.csv",
    "summary.csv",
    "seeds.csv",
)


def _strict_integer(
    value: object,
    *,
    label: str,
    minimum: Optional[int] = None,
) -> int:
    """Return a non-boolean integer satisfying an optional lower bound."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer, not {!r}.".format(label, value))
    if minimum is not None and value < minimum:
        raise ValueError("{} must be at least {}.".format(label, minimum))
    return value


def _finite_real(
    value: object,
    *,
    label: str,
    lower_exclusive: Optional[float] = None,
    upper_inclusive: Optional[float] = None,
) -> float:
    """Validate one finite non-boolean real configuration value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a real number, not {!r}.".format(label, value))
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("{} must be finite.".format(label)) from error
    if not math.isfinite(numeric):
        raise ValueError("{} must be finite.".format(label))
    if lower_exclusive is not None and numeric <= lower_exclusive:
        raise ValueError("{} must be greater than {}.".format(label, lower_exclusive))
    if upper_inclusive is not None and numeric > upper_inclusive:
        raise ValueError("{} must be at most {}.".format(label, upper_inclusive))
    return numeric


def _normalized_integer_grid(
    values: Iterable[int],
    *,
    label: str,
    allow_empty: bool = False,
) -> Tuple[int, ...]:
    """Validate, deduplicate-check, and sort a finite integer grid.

    Sorting is deliberate and documented: grid enumeration order is not a
    scientific parameter, and canonical ascending axes make serialized records
    and seeded comparisons independent of caller iteration order.
    """

    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("{} must be an iterable of integers.".format(label))
    try:
        supplied = tuple(values)
    except TypeError as error:
        raise TypeError("{} must be an iterable of integers.".format(label)) from error
    if not supplied and not allow_empty:
        raise ValueError("{} must contain at least one value.".format(label))
    normalized = tuple(
        _strict_integer(value, label="{} entry".format(label), minimum=0)
        for value in supplied
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("{} must not contain duplicate values.".format(label))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ClockExperimentConfig:
    """Complete scientific configuration for one two-clock grid.

    Args:
        num_states: Number ``M`` of states on each cyclic clock; at least three.
        intended_probability: Probability ``q`` of each row's intended next
            state. It must lie in ``(1/M, 1]``.
        predictable_direction: Fixed intended direction for ``y``, either
            ``"left"`` or ``"right"``.
        horizon: Number ``H`` of actions and transitions before terminal
            success is evaluated. Zero is valid.
        trials: Number ``T`` of independent top-level trial groups per cell.
        x_updates: Requested controllable updates per local ``(x, hand)`` row.
        y_updates: Requested predictable updates per local ``(y, hand)`` row.
        fixed_total_budgets: Optional values of ``N_x + N_y`` whose diagonals
            are designated for allocation comparisons.
        prior_concentration: Symmetric Dirichlet concentration applied to every
            possible next state in every row.
        master_seed: Integer expanded into stable, named sub-seeds.

    Notes:
        Budget grids and fixed-total budgets are canonicalized to ascending
        tuples after duplicate validation. This makes grid enumeration order
        observationally irrelevant.
    """

    num_states: int
    intended_probability: float
    predictable_direction: str
    horizon: int
    trials: int
    x_updates: Tuple[int, ...]
    y_updates: Tuple[int, ...]
    fixed_total_budgets: Tuple[int, ...] = ()
    prior_concentration: float = 1.0
    master_seed: int = 20260723

    def __post_init__(self) -> None:
        num_states = _strict_integer(self.num_states, label="num_states", minimum=3)
        intended_probability = _finite_real(
            self.intended_probability,
            label="intended_probability",
            lower_exclusive=1.0 / num_states,
            upper_inclusive=1.0,
        )
        if self.predictable_direction not in ("left", "right"):
            raise ValueError(
                "predictable_direction must be 'left' or 'right', not {!r}.".format(
                    self.predictable_direction
                )
            )
        horizon = _strict_integer(self.horizon, label="horizon", minimum=0)
        trials = _strict_integer(self.trials, label="trials", minimum=1)
        x_updates = _normalized_integer_grid(self.x_updates, label="x_updates")
        y_updates = _normalized_integer_grid(self.y_updates, label="y_updates")
        fixed_total_budgets = _normalized_integer_grid(
            self.fixed_total_budgets,
            label="fixed_total_budgets",
            allow_empty=True,
        )
        for total in fixed_total_budgets:
            if not any(n_x + n_y == total for n_x in x_updates for n_y in y_updates):
                raise ValueError(
                    "fixed_total_budgets value {} has no represented "
                    "(N_x, N_y) cell.".format(total)
                )
        prior_concentration = _finite_real(
            self.prior_concentration,
            label="prior_concentration",
            lower_exclusive=0.0,
        )
        master_seed = _strict_integer(self.master_seed, label="master_seed")

        object.__setattr__(self, "num_states", num_states)
        object.__setattr__(self, "intended_probability", intended_probability)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "trials", trials)
        object.__setattr__(self, "x_updates", x_updates)
        object.__setattr__(self, "y_updates", y_updates)
        object.__setattr__(self, "fixed_total_budgets", fixed_total_budgets)
        object.__setattr__(self, "prior_concentration", prior_concentration)
        object.__setattr__(self, "master_seed", master_seed)

    @property
    def contexts_per_factor(self) -> int:
        """Return the number ``3M`` of local rows in either factor belief."""

        return len(HAND_ACTIONS) * self.num_states

    def total_local_updates(self, n_x: int, n_y: int) -> int:
        """Return ``3M(N_x+N_y)`` after validating a configured cell."""

        if n_x not in self.x_updates or n_y not in self.y_updates:
            raise ValueError(
                "(n_x, n_y)=({}, {}) is not a configured grid cell.".format(n_x, n_y)
            )
        return self.contexts_per_factor * (n_x + n_y)

    def to_dict(self) -> Dict[str, object]:
        """Return the versioned JSON-safe scientific configuration."""

        return {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "num_states": self.num_states,
            "intended_probability": self.intended_probability,
            "predictable_direction": self.predictable_direction,
            "horizon": self.horizon,
            "trials": self.trials,
            "x_updates": list(self.x_updates),
            "y_updates": list(self.y_updates),
            "fixed_total_budgets": list(self.fixed_total_budgets),
            "prior_concentration_per_outcome": self.prior_concentration,
            "master_seed": self.master_seed,
            "hand_actions": list(HAND_ACTIONS),
            "contexts_per_factor": self.contexts_per_factor,
            "seed_derivation": "sha256(canonical-json(master_seed, namespace))[:8]",
        }


@dataclass(frozen=True)
class ClockWorld:
    """True variables, factors, and factored transition model."""

    x: Variable
    y: Variable
    hand: Variable
    x_factor: TabularMDP
    y_factor: TabularMDP
    model: FactoredMDP


@dataclass(frozen=True)
class ClockBeliefs:
    """Retained leaf beliefs and their factored posterior view."""

    x: TabularDirichletBelief
    y: TabularDirichletBelief
    model: FactoredDirichletBelief


def _seed_namespace_text(*namespace: object) -> str:
    """Return the canonical JSON representation of one seed namespace."""

    return json.dumps(
        list(namespace),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class SeedRecord:
    """One exact deterministic mapping from a stream namespace to an RNG seed."""

    namespace: str
    seed: int

    def __post_init__(self) -> None:
        """Validate canonical JSON namespace text and its unsigned seed."""

        if not isinstance(self.namespace, str):
            raise TypeError("Seed namespace must be a string.")
        try:
            decoded = json.loads(self.namespace)
        except json.JSONDecodeError as error:
            raise ValueError("Seed namespace must be valid JSON.") from error
        if not isinstance(decoded, list):
            raise ValueError("Seed namespace JSON must contain a list.")
        canonical = _seed_namespace_text(*decoded)
        if canonical != self.namespace:
            raise ValueError("Seed namespace must use canonical JSON formatting.")
        _strict_integer(self.seed, label="seed", minimum=0)

    def to_row(self) -> Dict[str, object]:
        """Return the versioned CSV representation."""

        return {
            "schema_version": SEED_SCHEMA_VERSION,
            "namespace": self.namespace,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TrialRecord:
    """One binary task evaluation for one grid cell and trial group.

    ``success`` is redundant by design and validated against the recorded
    terminal state: navigation succeeds exactly when ``final_x == target_x``;
    synchronization succeeds exactly when ``final_x == final_y``. This keeps a
    malformed raw record from silently corrupting downstream aggregation.
    """

    trial_id: int
    task: str
    n_x: int
    n_y: int
    n_total_per_context: int
    total_local_updates: int
    initial_x: int
    initial_y: int
    target_x: Optional[int]
    final_x: int
    final_y: int
    success: int
    sampled_model_value: float
    true_policy_success_probability: float
    initial_x_seed: int
    initial_y_seed: int
    target_seed: int
    x_model_seed: int
    y_model_seed: int
    tie_seed: int
    rollout_x_seed: int
    rollout_y_seed: int

    def __post_init__(self) -> None:
        _strict_integer(self.trial_id, label="trial_id", minimum=0)
        if self.task not in TASK_ORDER:
            raise ValueError("Unknown task {!r}.".format(self.task))
        for integer_label, integer_value in (
            ("n_x", self.n_x),
            ("n_y", self.n_y),
            ("n_total_per_context", self.n_total_per_context),
            ("total_local_updates", self.total_local_updates),
            ("initial_x", self.initial_x),
            ("initial_y", self.initial_y),
            ("final_x", self.final_x),
            ("final_y", self.final_y),
        ):
            _strict_integer(integer_value, label=integer_label, minimum=0)
        if self.n_total_per_context != self.n_x + self.n_y:
            raise ValueError("n_total_per_context must equal n_x + n_y.")
        if self.target_x is not None:
            _strict_integer(self.target_x, label="target_x", minimum=0)
        if self.task == NAVIGATION_TASK and self.target_x is None:
            raise ValueError("Navigation records require target_x.")
        if self.task == SYNCHRONIZATION_TASK and self.target_x is not None:
            raise ValueError("Synchronization records must not define target_x.")
        _strict_integer(self.success, label="success", minimum=0)
        if self.success not in (0, 1):
            raise ValueError("success must be integer 0 or 1.")
        expected_success = (
            int(self.final_x == self.target_x)
            if self.task == NAVIGATION_TASK
            else int(self.final_x == self.final_y)
        )
        if self.success != expected_success:
            raise ValueError(
                "success is inconsistent with the recorded terminal task outcome."
            )
        for probability_label, probability_value in (
            ("sampled_model_value", self.sampled_model_value),
            (
                "true_policy_success_probability",
                self.true_policy_success_probability,
            ),
        ):
            numeric = _finite_real(probability_value, label=probability_label)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError("{} must lie in [0, 1].".format(probability_label))
            object.__setattr__(self, probability_label, numeric)
        for seed_label, seed_value in (
            ("initial_x_seed", self.initial_x_seed),
            ("initial_y_seed", self.initial_y_seed),
            ("target_seed", self.target_seed),
            ("x_model_seed", self.x_model_seed),
            ("y_model_seed", self.y_model_seed),
            ("tie_seed", self.tie_seed),
            ("rollout_x_seed", self.rollout_x_seed),
            ("rollout_y_seed", self.rollout_y_seed),
        ):
            _strict_integer(seed_value, label=seed_label, minimum=0)

    def to_row(self) -> Dict[str, object]:
        """Return the versioned deterministic raw-trial CSV row."""

        return {
            "schema_version": TRIAL_SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "task": self.task,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "n_total_per_context": self.n_total_per_context,
            "total_local_updates": self.total_local_updates,
            "initial_x": self.initial_x,
            "initial_y": self.initial_y,
            "target_x": "" if self.target_x is None else self.target_x,
            "final_x": self.final_x,
            "final_y": self.final_y,
            "success": self.success,
            "sampled_model_value": self.sampled_model_value,
            "true_policy_success_probability": (self.true_policy_success_probability),
            "initial_x_seed": self.initial_x_seed,
            "initial_y_seed": self.initial_y_seed,
            "target_seed": self.target_seed,
            "x_model_seed": self.x_model_seed,
            "y_model_seed": self.y_model_seed,
            "tie_seed": self.tie_seed,
            "rollout_x_seed": self.rollout_x_seed,
            "rollout_y_seed": self.rollout_y_seed,
        }


@dataclass(frozen=True)
class CellSummary:
    """Aggregate binary performance and uncertainty for one task/cell."""

    task: str
    n_x: int
    n_y: int
    successes: int
    trials: int
    success_probability: float
    monte_carlo_standard_error: float
    wilson_95_lower: float
    wilson_95_upper: float

    def __post_init__(self) -> None:
        if self.task not in TASK_ORDER:
            raise ValueError("Unknown task {!r}.".format(self.task))
        for integer_label, integer_value in (
            ("n_x", self.n_x),
            ("n_y", self.n_y),
            ("successes", self.successes),
        ):
            _strict_integer(integer_value, label=integer_label, minimum=0)
        trials = _strict_integer(self.trials, label="trials", minimum=1)
        if self.successes > trials:
            raise ValueError("successes must not exceed trials.")
        for probability_label, probability_value in (
            ("success_probability", self.success_probability),
            (
                "monte_carlo_standard_error",
                self.monte_carlo_standard_error,
            ),
            ("wilson_95_lower", self.wilson_95_lower),
            ("wilson_95_upper", self.wilson_95_upper),
        ):
            numeric = _finite_real(probability_value, label=probability_label)
            if not 0.0 <= numeric <= 1.0:
                raise ValueError("{} must lie in [0, 1].".format(probability_label))
            object.__setattr__(self, probability_label, numeric)
        if self.wilson_95_lower > self.wilson_95_upper:
            raise ValueError("Wilson lower bound must not exceed its upper bound.")
        expected_probability = self.successes / trials
        expected_standard_error = math.sqrt(
            expected_probability * (1.0 - expected_probability) / trials
        )
        expected_lower, expected_upper = wilson_interval(self.successes, trials)
        for label, actual, expected in (
            (
                "success_probability",
                self.success_probability,
                expected_probability,
            ),
            (
                "monte_carlo_standard_error",
                self.monte_carlo_standard_error,
                expected_standard_error,
            ),
            ("wilson_95_lower", self.wilson_95_lower, expected_lower),
            ("wilson_95_upper", self.wilson_95_upper, expected_upper),
        ):
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    "{} is inconsistent with successes/trials.".format(label)
                )

    def to_row(self) -> Dict[str, object]:
        """Return the versioned deterministic summary CSV row."""

        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "task": self.task,
            "n_x": self.n_x,
            "n_y": self.n_y,
            "successes": self.successes,
            "trials": self.trials,
            "success_probability": self.success_probability,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "wilson_95_lower": self.wilson_95_lower,
            "wilson_95_upper": self.wilson_95_upper,
        }


@dataclass(frozen=True)
class AllocationStrategyPoint:
    """One selected fixed-total allocation and its original cell statistics.

    ``total_budget`` is the per-context budget ``N = N_x + N_y``. Strategy
    geometry follows the heatmap convention in which ``N_x`` is horizontal and
    ``N_y`` is vertical: below-diagonal points satisfy ``N_x > N_y`` and
    above-diagonal points satisfy ``N_x < N_y``.

    The statistical fields are copied without recomputation or pooling from
    the selected :class:`CellSummary`. Wilson bounds therefore quantify the
    selected cell only; they do not adjust for post-hoc maximization over
    candidate allocations.
    """

    task: str
    total_budget: int
    strategy: str
    n_x: int
    n_y: int
    successes: int
    trials: int
    success_probability: float
    monte_carlo_standard_error: float
    wilson_95_lower: float
    wilson_95_upper: float

    def __post_init__(self) -> None:
        """Validate strategy geometry and the retained binomial statistics."""

        total_budget = _strict_integer(
            self.total_budget,
            label="total_budget",
            minimum=1,
        )
        if total_budget % 2:
            raise ValueError("total_budget must be even.")
        n_x = _strict_integer(self.n_x, label="n_x", minimum=0)
        n_y = _strict_integer(self.n_y, label="n_y", minimum=0)
        if n_x + n_y != total_budget:
            raise ValueError("n_x + n_y must equal total_budget.")
        if self.strategy not in ALLOCATION_STRATEGY_ORDER:
            raise ValueError("Unknown allocation strategy {!r}.".format(self.strategy))
        if self.strategy == EQUAL_ALLOCATION_STRATEGY:
            if n_x != n_y:
                raise ValueError("Equal allocation requires n_x == n_y.")
        elif self.strategy == BEST_BELOW_DIAGONAL_STRATEGY:
            if n_x <= n_y:
                raise ValueError("Below-diagonal allocation requires n_x > n_y.")
        elif n_x >= n_y:
            raise ValueError("Above-diagonal allocation requires n_x < n_y.")

        validated = CellSummary(
            task=self.task,
            n_x=n_x,
            n_y=n_y,
            successes=self.successes,
            trials=self.trials,
            success_probability=self.success_probability,
            monte_carlo_standard_error=self.monte_carlo_standard_error,
            wilson_95_lower=self.wilson_95_lower,
            wilson_95_upper=self.wilson_95_upper,
        )
        object.__setattr__(
            self,
            "success_probability",
            validated.success_probability,
        )
        object.__setattr__(
            self,
            "monte_carlo_standard_error",
            validated.monte_carlo_standard_error,
        )
        object.__setattr__(self, "wilson_95_lower", validated.wilson_95_lower)
        object.__setattr__(self, "wilson_95_upper", validated.wilson_95_upper)


@dataclass(frozen=True)
class ClockExperimentResult:
    """Complete in-memory result with raw trials, summaries, and seed ledger."""

    config: ClockExperimentConfig
    trials: Tuple[TrialRecord, ...]
    summaries: Tuple[CellSummary, ...]
    seeds: Tuple[SeedRecord, ...]

    def __post_init__(self) -> None:
        """Cross-validate raw data, summaries, configuration, and seed ledger."""

        if not isinstance(self.config, ClockExperimentConfig):
            raise TypeError("config must be a ClockExperimentConfig.")
        if any(not isinstance(record, TrialRecord) for record in self.trials):
            raise TypeError("trials must contain only TrialRecord objects.")
        if any(not isinstance(summary, CellSummary) for summary in self.summaries):
            raise TypeError("summaries must contain only CellSummary objects.")
        if any(not isinstance(seed, SeedRecord) for seed in self.seeds):
            raise TypeError("seeds must contain only SeedRecord objects.")

        expected_summaries = summarize_trials(self.trials, config=self.config)
        if self.summaries != expected_summaries:
            raise ValueError(
                "summaries must equal the canonical aggregation of trials."
            )
        for record in self.trials:
            for label, value in (
                ("initial_x", record.initial_x),
                ("initial_y", record.initial_y),
                ("final_x", record.final_x),
                ("final_y", record.final_y),
            ):
                if value >= self.config.num_states:
                    raise ValueError(
                        "{}={} is outside the configured clock domain.".format(
                            label, value
                        )
                    )
            if (
                record.target_x is not None
                and record.target_x >= self.config.num_states
            ):
                raise ValueError("target_x is outside the configured clock domain.")
            if record.total_local_updates != self.config.total_local_updates(
                record.n_x, record.n_y
            ):
                raise ValueError(
                    "Trial total_local_updates is inconsistent with config."
                )

        namespaces = [record.namespace for record in self.seeds]
        if namespaces != sorted(namespaces) or len(namespaces) != len(set(namespaces)):
            raise ValueError(
                "Seed records must have unique namespaces in canonical order."
            )
        seed_by_namespace = {
            seed_record.namespace: seed_record.seed for seed_record in self.seeds
        }
        for seed_record in self.seeds:
            namespace = json.loads(seed_record.namespace)
            if seed_record.seed != derive_seed(self.config.master_seed, *namespace):
                raise ValueError(
                    "Seed ledger value does not match its namespace and master seed."
                )
        for record in self.trials:
            tie_namespace: Tuple[object, ...]
            if record.task == NAVIGATION_TASK:
                tie_namespace = (
                    "tie-break",
                    record.task,
                    record.trial_id,
                    record.n_x,
                )
            else:
                tie_namespace = (
                    "tie-break",
                    record.task,
                    record.trial_id,
                    record.n_x,
                    record.n_y,
                )
            expected_seed_references = (
                (
                    "initial_x_seed",
                    record.initial_x_seed,
                    ("task-instance", "initial-x", record.trial_id),
                ),
                (
                    "initial_y_seed",
                    record.initial_y_seed,
                    ("task-instance", "initial-y", record.trial_id),
                ),
                (
                    "target_seed",
                    record.target_seed,
                    ("task-instance", "target-x", record.trial_id),
                ),
                (
                    "x_model_seed",
                    record.x_model_seed,
                    ("posterior-model", "x", record.trial_id, record.n_x),
                ),
                (
                    "y_model_seed",
                    record.y_model_seed,
                    ("posterior-model", "y", record.trial_id, record.n_y),
                ),
                ("tie_seed", record.tie_seed, tie_namespace),
                (
                    "rollout_x_seed",
                    record.rollout_x_seed,
                    ("rollout", record.task, "x", record.trial_id),
                ),
                (
                    "rollout_y_seed",
                    record.rollout_y_seed,
                    ("rollout", record.task, "y", record.trial_id),
                ),
            )
            for label, actual_seed, namespace in expected_seed_references:
                expected_seed = seed_by_namespace.get(_seed_namespace_text(*namespace))
                if expected_seed is None:
                    raise ValueError(
                        "{} references a namespace absent from the seed ledger.".format(
                            label
                        )
                    )
                if actual_seed != expected_seed:
                    raise ValueError(
                        "{} does not match its exact seed namespace.".format(label)
                    )


@dataclass(frozen=True)
class RunArtifacts:
    """Paths created for one unique persisted experiment run."""

    run_directory: Path
    config_path: Path
    trials_path: Path
    summary_path: Path
    seeds_path: Path
    figures_directory: Path
    manifest_path: Path


class _SeedFactory:
    """Derive and retain exact named seeds from one master seed."""

    def __init__(self, master_seed: int) -> None:
        self._master_seed = master_seed
        self._seeds: Dict[str, int] = {}

    def derive(self, *namespace: object) -> int:
        """Return the stable 64-bit seed for one JSON-safe namespace."""

        namespace_json = _seed_namespace_text(*namespace)
        payload = json.dumps(
            {
                "master_seed": self._master_seed,
                "namespace": json.loads(namespace_json),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        previous = self._seeds.setdefault(namespace_json, seed)
        if previous != seed:
            raise RuntimeError("Seed namespace collision produced inconsistent values.")
        return seed

    def records(self) -> Tuple[SeedRecord, ...]:
        """Return every used seed in canonical namespace order."""

        return tuple(
            SeedRecord(namespace=namespace, seed=self._seeds[namespace])
            for namespace in sorted(self._seeds)
        )


def derive_seed(master_seed: int, *namespace: object) -> int:
    """Derive one stable seed without retaining a ledger.

    Args:
        master_seed: Non-boolean integer root seed.
        namespace: JSON-safe namespace components.

    Returns:
        An unsigned 64-bit integer derived from SHA-256.
    """

    validated = _strict_integer(master_seed, label="master_seed")
    return _SeedFactory(validated).derive(*namespace)


def _intended_successor(
    state: int,
    action_or_direction: str,
    num_states: int,
) -> int:
    """Return one cyclic intended successor."""

    displacement = {
        "left": -1,
        "right": 1,
        "stay": 0,
    }[action_or_direction]
    return (state + displacement) % num_states


def _clock_rows(
    variable: Variable,
    hand: Variable,
    *,
    num_states: int,
    intended_probability: float,
    fixed_direction: Optional[str],
) -> Tuple[
    Tuple[
        Mapping[str, int],
        Mapping[str, str],
        Tuple[Tuple[Mapping[str, int], float], ...],
    ],
    ...,
]:
    """Construct every dense true transition row for one clock factor."""

    residual = (1.0 - intended_probability) / (num_states - 1)
    rows: List[
        Tuple[
            Mapping[str, int],
            Mapping[str, str],
            Tuple[Tuple[Mapping[str, int], float], ...],
        ]
    ] = []
    assert variable.domain is not None
    assert hand.domain is not None
    for current_state, action in product(variable.domain, hand.domain):
        if not isinstance(current_state, int) or not isinstance(action, str):
            raise TypeError("Clock variable domains have unexpected value types.")
        direction = action if fixed_direction is None else fixed_direction
        intended = _intended_successor(current_state, direction, num_states)
        probability_entries: List[Tuple[Mapping[str, int], float]] = []
        for next_state in variable.domain:
            if not isinstance(next_state, int):
                raise TypeError("Clock variable domains must contain integers.")
            probability_entries.append(
                (
                    {variable.name: next_state},
                    intended_probability if next_state == intended else residual,
                )
            )
        probabilities = tuple(probability_entries)
        rows.append(
            (
                {variable.name: current_state},
                {hand.name: action},
                probabilities,
            )
        )
    return tuple(rows)


def build_clock_world(config: ClockExperimentConfig) -> ClockWorld:
    """Build the true two-clock environment from a validated configuration.

    Args:
        config: Complete two-clock scientific configuration.

    Returns:
        Variables, local dense tables, and their independent factored product.
    """

    if not isinstance(config, ClockExperimentConfig):
        raise TypeError("config must be a ClockExperimentConfig.")
    domain = tuple(range(config.num_states))
    x = Variable("x", domain)
    y = Variable("y", domain)
    hand = Variable("hand", HAND_ACTIONS)
    x_factor = TabularMDP(
        (x,),
        parent_variables=(hand,),
        transitions=_clock_rows(
            x,
            hand,
            num_states=config.num_states,
            intended_probability=config.intended_probability,
            fixed_direction=None,
        ),
    )
    y_factor = TabularMDP(
        (y,),
        parent_variables=(hand,),
        transitions=_clock_rows(
            y,
            hand,
            num_states=config.num_states,
            intended_probability=config.intended_probability,
            fixed_direction=config.predictable_direction,
        ),
    )
    model = FactoredMDP((x_factor, y_factor))
    return ClockWorld(
        x=x,
        y=y,
        hand=hand,
        x_factor=x_factor,
        y_factor=y_factor,
        model=model,
    )


def build_clock_beliefs(
    config: ClockExperimentConfig,
    world: Optional[ClockWorld] = None,
) -> ClockBeliefs:
    """Build action-conditioned independent Dirichlet beliefs for both clocks."""

    if not isinstance(config, ClockExperimentConfig):
        raise TypeError("config must be a ClockExperimentConfig.")
    active_world = build_clock_world(config) if world is None else world
    if not isinstance(active_world, ClockWorld):
        raise TypeError("world must be a ClockWorld.")
    expected_domain = tuple(range(config.num_states))
    if (
        active_world.x != Variable("x", expected_domain)
        or active_world.y != Variable("y", expected_domain)
        or active_world.hand != Variable("hand", HAND_ACTIONS)
    ):
        raise ValueError("world variable specifications do not match config.")
    x_belief = TabularDirichletBelief(
        (active_world.x,),
        parent_variables=(active_world.hand,),
        prior=config.prior_concentration,
    )
    y_belief = TabularDirichletBelief(
        (active_world.y,),
        parent_variables=(active_world.hand,),
        prior=config.prior_concentration,
    )
    return ClockBeliefs(
        x=x_belief,
        y=y_belief,
        model=FactoredDirichletBelief((x_belief, y_belief)),
    )


def _learn_factor_models(
    *,
    config: ClockExperimentConfig,
    true_factor: TabularMDP,
    variable: Variable,
    hand: Variable,
    budgets: Sequence[int],
    factor_name: str,
    trial_id: int,
    seeds: _SeedFactory,
) -> Tuple[Dict[int, TabularMDP], Dict[int, int]]:
    """Learn nested posterior samples for every requested factor-local budget."""

    belief = TabularDirichletBelief(
        (variable,),
        parent_variables=(hand,),
        prior=config.prior_concentration,
    )
    assert variable.domain is not None
    assert hand.domain is not None
    context_rngs: Dict[Tuple[int, str], random.Random] = {}
    for current_state, action in product(variable.domain, hand.domain):
        if not isinstance(current_state, int) or not isinstance(action, str):
            raise TypeError("Clock variable domains have unexpected value types.")
        seed = seeds.derive(
            "observation",
            factor_name,
            trial_id,
            current_state,
            action,
        )
        context_rngs[(current_state, action)] = random.Random(seed)

    requested = set(budgets)
    max_budget = max(budgets)
    models: Dict[int, TabularMDP] = {}
    model_seeds: Dict[int, int] = {}
    for observations_per_context in range(max_budget + 1):
        if observations_per_context in requested:
            model_seed = seeds.derive(
                "posterior-model",
                factor_name,
                trial_id,
                observations_per_context,
            )
            models[observations_per_context] = belief.sample_mdp(
                random.Random(model_seed)
            )
            model_seeds[observations_per_context] = model_seed
        if observations_per_context == max_budget:
            break
        for current_state, action in product(variable.domain, hand.domain):
            if not isinstance(current_state, int) or not isinstance(action, str):
                raise TypeError("Clock variable domains have unexpected value types.")
            next_state = true_factor.sample_transition(
                current={variable.name: current_state},
                parents={hand.name: action},
                rng=context_rngs[(current_state, action)],
            )
            belief.update(
                current={variable.name: current_state},
                next_state=next_state,
                parents={hand.name: action},
            )

    return models, model_seeds


def _uniform_clock_draw(
    config: ClockExperimentConfig,
    seeds: _SeedFactory,
    *namespace: object,
) -> Tuple[int, int]:
    """Return one uniform clock value and its exact derived seed."""

    seed = seeds.derive(*namespace)
    return random.Random(seed).randrange(config.num_states), seed


def _navigation_reward(target: int) -> TerminalReward:
    """Return the terminal binary reward over the controllable factor."""

    def reward(state: Assignment) -> float:
        return 1.0 if state["x"] == target else 0.0

    return reward


def _synchronization_reward(state: Assignment) -> float:
    """Return one when the two terminal clock states match."""

    return 1.0 if state["x"] == state["y"] else 0.0


def _sampled_clock_value(value: object, *, variable_name: str) -> int:
    """Return a sampled clock state without silently coercing its type."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "Sampled value for {!r} must be an integer.".format(variable_name)
        )
    return value


def _rollout_navigation(
    *,
    world: ClockWorld,
    policy: FiniteHorizonPolicy,
    initial_x: int,
    initial_y: int,
    horizon: int,
    x_rng: random.Random,
    y_rng: random.Random,
) -> Tuple[int, int]:
    """Roll out an x-marginal navigation policy in both true factors."""

    x_value = initial_x
    y_value = initial_y
    for step in range(horizon):
        action = policy.action(step, {"x": x_value})
        parents = {"hand": action["hand"]}
        x_value = _sampled_clock_value(
            world.x_factor.sample_transition({"x": x_value}, parents, x_rng)["x"],
            variable_name="x",
        )
        y_value = _sampled_clock_value(
            world.y_factor.sample_transition({"y": y_value}, parents, y_rng)["y"],
            variable_name="y",
        )
    return x_value, y_value


def _rollout_synchronization(
    *,
    world: ClockWorld,
    policy: FiniteHorizonPolicy,
    initial_x: int,
    initial_y: int,
    horizon: int,
    x_rng: random.Random,
    y_rng: random.Random,
) -> Tuple[int, int]:
    """Roll out a joint synchronization policy through the true local factors."""

    x_value = initial_x
    y_value = initial_y
    for step in range(horizon):
        action = policy.action(step, {"x": x_value, "y": y_value})
        parents = {"hand": action["hand"]}
        x_value = _sampled_clock_value(
            world.x_factor.sample_transition({"x": x_value}, parents, x_rng)["x"],
            variable_name="x",
        )
        y_value = _sampled_clock_value(
            world.y_factor.sample_transition({"y": y_value}, parents, y_rng)["y"],
            variable_name="y",
        )
    return x_value, y_value


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = _WILSON_95_Z,
) -> Tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion.

    Args:
        successes: Integer number of successful independent trials.
        trials: Positive integer number of independent trials.
        z: Positive finite standard-normal critical value. The default is the
            conventional 95% value.

    Returns:
        Lower and upper probability bounds in ``[0, 1]``.
    """

    success_count = _strict_integer(successes, label="successes", minimum=0)
    trial_count = _strict_integer(trials, label="trials", minimum=1)
    if success_count > trial_count:
        raise ValueError("successes must not exceed trials.")
    critical = _finite_real(z, label="z", lower_exclusive=0.0)
    proportion = success_count / trial_count
    squared = critical * critical
    denominator = 1.0 + squared / trial_count
    center = (proportion + squared / (2.0 * trial_count)) / denominator
    half_width = (
        critical
        * math.sqrt(
            proportion * (1.0 - proportion) / trial_count
            + squared / (4.0 * trial_count * trial_count)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def summarize_trials(
    records: Iterable[TrialRecord],
    *,
    config: Optional[ClockExperimentConfig] = None,
) -> Tuple[CellSummary, ...]:
    """Aggregate binary raw records into deterministic cell summaries.

    Args:
        records: Raw task evaluations.
        config: When supplied, enforce complete coverage of every configured
            task, cell, and trial identifier and preserve canonical grid order.

    Returns:
        One summary per task and grid cell.

    Raises:
        TypeError: If an item is not a :class:`TrialRecord`.
        ValueError: If records are duplicated, incomplete, or internally
            inconsistent.
    """

    record_tuple = tuple(records)
    if any(not isinstance(record, TrialRecord) for record in record_tuple):
        raise TypeError("records must contain only TrialRecord objects.")
    grouped: Dict[Tuple[str, int, int], List[TrialRecord]] = {}
    seen = set()
    for record in record_tuple:
        identity = (record.task, record.n_x, record.n_y, record.trial_id)
        if identity in seen:
            raise ValueError("Duplicate raw trial identity {!r}.".format(identity))
        seen.add(identity)
        grouped.setdefault((record.task, record.n_x, record.n_y), []).append(record)

    if config is None:
        keys = tuple(
            sorted(
                grouped,
                key=lambda key: (
                    TASK_ORDER.index(key[0]),
                    key[2],
                    key[1],
                ),
            )
        )
    else:
        if not isinstance(config, ClockExperimentConfig):
            raise TypeError("config must be a ClockExperimentConfig.")
        keys = tuple(
            (task, n_x, n_y)
            for task in TASK_ORDER
            for n_y in config.y_updates
            for n_x in config.x_updates
        )
        unexpected = set(grouped) - set(keys)
        if unexpected:
            raise ValueError(
                "Raw records contain unexpected task/cell keys, for example {!r}.".format(
                    sorted(unexpected)[0]
                )
            )

    summaries = []
    for key in keys:
        cell_records = grouped.get(key, [])
        if not cell_records:
            raise ValueError("Raw records are missing task/cell {!r}.".format(key))
        ordered = sorted(cell_records, key=lambda record: record.trial_id)
        if config is not None:
            expected_ids = list(range(config.trials))
            actual_ids = [record.trial_id for record in ordered]
            if actual_ids != expected_ids:
                raise ValueError(
                    "Task/cell {!r} must contain trial ids 0..{} exactly once.".format(
                        key, config.trials - 1
                    )
                )
        successes = sum(record.success for record in ordered)
        trials = len(ordered)
        probability = successes / trials
        lower, upper = wilson_interval(successes, trials)
        summaries.append(
            CellSummary(
                task=key[0],
                n_x=key[1],
                n_y=key[2],
                successes=successes,
                trials=trials,
                success_probability=probability,
                monte_carlo_standard_error=math.sqrt(
                    probability * (1.0 - probability) / trials
                ),
                wilson_95_lower=lower,
                wilson_95_upper=upper,
            )
        )
    return tuple(summaries)


def select_allocation_strategies(
    summaries: Iterable[CellSummary],
    total_budgets: Iterable[int],
) -> Tuple[AllocationStrategyPoint, ...]:
    """Select equal and best off-diagonal allocations at fixed total budgets.

    For every task and requested positive even total ``N``, selection requires
    the exact equal cell ``(N/2, N/2)``, at least one below-diagonal candidate
    with ``N_x > N_y``, and at least one above-diagonal candidate with
    ``N_x < N_y``. The best candidate on each side maximizes the observed
    success probability. Ties prefer the smallest ``abs(N_x - N_y)`` and then
    canonical ``(N_x, N_y)`` order.

    Args:
        summaries: Valid cell summaries. Cells outside the requested fixed-total
            diagonals are permitted and ignored after global identity
            validation.
        total_budgets: Distinct positive even per-context totals. Caller order
            is not semantically meaningful and is canonicalized.

    Returns:
        Immutable points ordered by repository task order, ascending total
        budget, and :data:`ALLOCATION_STRATEGY_ORDER`.

    Raises:
        TypeError: If either input is not a supported iterable or a summary
            entry is not a :class:`CellSummary`.
        ValueError: If totals are empty, duplicated, nonpositive, or odd; if
            summary identities are duplicated; or if a task/total lacks any
            required strategy cell.

    Notes:
        Best-side selection is a descriptive post-hoc oracle operation. The
        copied Wilson intervals describe individual selected cells and are not
        adjusted for selecting a maximum.
    """

    try:
        summary_tuple = tuple(summaries)
    except TypeError as error:
        raise TypeError(
            "summaries must be an iterable of CellSummary objects."
        ) from error
    if any(not isinstance(summary, CellSummary) for summary in summary_tuple):
        raise TypeError("summaries must contain only CellSummary objects.")

    totals = _normalized_integer_grid(total_budgets, label="total_budgets")
    for total in totals:
        if total == 0:
            raise ValueError("total_budgets entries must be positive.")
        if total % 2:
            raise ValueError("total_budgets entries must be even.")

    by_identity: Dict[Tuple[str, int, int], CellSummary] = {}
    for summary in summary_tuple:
        identity = (summary.task, summary.n_x, summary.n_y)
        if identity in by_identity:
            raise ValueError("Duplicate cell summary identity {!r}.".format(identity))
        by_identity[identity] = summary

    selected: List[AllocationStrategyPoint] = []
    for task in TASK_ORDER:
        for total in totals:
            diagonal = tuple(
                summary
                for summary in summary_tuple
                if summary.task == task and summary.n_x + summary.n_y == total
            )
            equal = tuple(summary for summary in diagonal if summary.n_x == summary.n_y)
            if len(equal) != 1:
                raise ValueError(
                    "Task {!r} total budget {} requires exactly one equal "
                    "allocation cell.".format(task, total)
                )
            below = tuple(summary for summary in diagonal if summary.n_x > summary.n_y)
            if not below:
                raise ValueError(
                    "Task {!r} total budget {} requires at least one "
                    "below-diagonal cell with n_x > n_y.".format(task, total)
                )
            above = tuple(summary for summary in diagonal if summary.n_x < summary.n_y)
            if not above:
                raise ValueError(
                    "Task {!r} total budget {} requires at least one "
                    "above-diagonal cell with n_x < n_y.".format(task, total)
                )

            below_best = min(
                below,
                key=lambda summary: (
                    -summary.success_probability,
                    abs(summary.n_x - summary.n_y),
                    summary.n_x,
                    summary.n_y,
                ),
            )
            above_best = min(
                above,
                key=lambda summary: (
                    -summary.success_probability,
                    abs(summary.n_x - summary.n_y),
                    summary.n_x,
                    summary.n_y,
                ),
            )
            for strategy, summary in (
                (EQUAL_ALLOCATION_STRATEGY, equal[0]),
                (BEST_BELOW_DIAGONAL_STRATEGY, below_best),
                (BEST_ABOVE_DIAGONAL_STRATEGY, above_best),
            ):
                selected.append(
                    AllocationStrategyPoint(
                        task=summary.task,
                        total_budget=total,
                        strategy=strategy,
                        n_x=summary.n_x,
                        n_y=summary.n_y,
                        successes=summary.successes,
                        trials=summary.trials,
                        success_probability=summary.success_probability,
                        monte_carlo_standard_error=(summary.monte_carlo_standard_error),
                        wilson_95_lower=summary.wilson_95_lower,
                        wilson_95_upper=summary.wilson_95_upper,
                    )
                )
    return tuple(selected)


def run_clock_experiment(
    config: ClockExperimentConfig,
) -> ClockExperimentResult:
    """Run the complete matched, nested two-clock experiment in memory.

    Args:
        config: Validated scientific configuration.

    Returns:
        Raw binary trials, cell summaries, and every derived random seed.

    Notes:
        The runner performs no filesystem or plotting operations. A fresh
        sampled factor model is created for every trial and local budget. Each
        factor is reused across the opposite heatmap axis; the sampled ``x``
        factor is used by both tasks, while sampled ``y`` is needed only for
        synchronization. Each trial evaluates one navigation rollout per
        controllable budget and records that matched outcome across the
        predictable-budget axis. Synchronization evaluates one rollout per
        controllable/predictable budget pair. Evaluation performs no online
        belief update.
    """

    if not isinstance(config, ClockExperimentConfig):
        raise TypeError("config must be a ClockExperimentConfig.")
    world = build_clock_world(config)
    seeds = _SeedFactory(config.master_seed)
    records: List[TrialRecord] = []

    for trial_id in range(config.trials):
        x_models, x_model_seeds = _learn_factor_models(
            config=config,
            true_factor=world.x_factor,
            variable=world.x,
            hand=world.hand,
            budgets=config.x_updates,
            factor_name="x",
            trial_id=trial_id,
            seeds=seeds,
        )
        y_models, y_model_seeds = _learn_factor_models(
            config=config,
            true_factor=world.y_factor,
            variable=world.y,
            hand=world.hand,
            budgets=config.y_updates,
            factor_name="y",
            trial_id=trial_id,
            seeds=seeds,
        )

        initial_x, initial_x_seed = _uniform_clock_draw(
            config, seeds, "task-instance", "initial-x", trial_id
        )
        initial_y, initial_y_seed = _uniform_clock_draw(
            config, seeds, "task-instance", "initial-y", trial_id
        )
        target_x, target_seed = _uniform_clock_draw(
            config, seeds, "task-instance", "target-x", trial_id
        )

        navigation_by_x: Dict[int, Tuple[int, int, float, float, int, int, int]] = {}
        for n_x in config.x_updates:
            navigation_reward = _navigation_reward(target_x)
            tie_seed = seeds.derive("tie-break", NAVIGATION_TASK, trial_id, n_x)
            policy = plan_finite_horizon(
                x_models[n_x],
                navigation_reward,
                config.horizon,
                rng=random.Random(tie_seed),
            )
            planned_value = policy.value(
                0 if config.horizon > 0 else config.horizon,
                {"x": initial_x},
            )
            true_probability = evaluate_finite_horizon_policy(
                world.x_factor,
                policy,
                navigation_reward,
                {"x": initial_x},
            )
            rollout_x_seed = seeds.derive("rollout", NAVIGATION_TASK, "x", trial_id)
            rollout_y_seed = seeds.derive("rollout", NAVIGATION_TASK, "y", trial_id)
            final_x, final_y = _rollout_navigation(
                world=world,
                policy=policy,
                initial_x=initial_x,
                initial_y=initial_y,
                horizon=config.horizon,
                x_rng=random.Random(rollout_x_seed),
                y_rng=random.Random(rollout_y_seed),
            )
            navigation_by_x[n_x] = (
                final_x,
                final_y,
                planned_value,
                true_probability,
                tie_seed,
                rollout_x_seed,
                rollout_y_seed,
            )

        for n_y in config.y_updates:
            for n_x in config.x_updates:
                (
                    final_x,
                    final_y,
                    planned_value,
                    true_probability,
                    navigation_tie_seed,
                    navigation_rollout_x_seed,
                    navigation_rollout_y_seed,
                ) = navigation_by_x[n_x]
                records.append(
                    TrialRecord(
                        trial_id=trial_id,
                        task=NAVIGATION_TASK,
                        n_x=n_x,
                        n_y=n_y,
                        n_total_per_context=n_x + n_y,
                        total_local_updates=config.total_local_updates(n_x, n_y),
                        initial_x=initial_x,
                        initial_y=initial_y,
                        target_x=target_x,
                        final_x=final_x,
                        final_y=final_y,
                        success=int(final_x == target_x),
                        sampled_model_value=planned_value,
                        true_policy_success_probability=true_probability,
                        initial_x_seed=initial_x_seed,
                        initial_y_seed=initial_y_seed,
                        target_seed=target_seed,
                        x_model_seed=x_model_seeds[n_x],
                        y_model_seed=y_model_seeds[n_y],
                        tie_seed=navigation_tie_seed,
                        rollout_x_seed=navigation_rollout_x_seed,
                        rollout_y_seed=navigation_rollout_y_seed,
                    )
                )

                sampled_world = FactoredMDP((x_models[n_x], y_models[n_y]))
                synchronization_tie_seed = seeds.derive(
                    "tie-break",
                    SYNCHRONIZATION_TASK,
                    trial_id,
                    n_x,
                    n_y,
                )
                synchronization_policy = plan_finite_horizon(
                    sampled_world,
                    _synchronization_reward,
                    config.horizon,
                    rng=random.Random(synchronization_tie_seed),
                )
                synchronization_planned_value = synchronization_policy.value(
                    0 if config.horizon > 0 else config.horizon,
                    {"x": initial_x, "y": initial_y},
                )
                synchronization_true_probability = evaluate_finite_horizon_policy(
                    world.model,
                    synchronization_policy,
                    _synchronization_reward,
                    {"x": initial_x, "y": initial_y},
                )
                synchronization_rollout_x_seed = seeds.derive(
                    "rollout", SYNCHRONIZATION_TASK, "x", trial_id
                )
                synchronization_rollout_y_seed = seeds.derive(
                    "rollout", SYNCHRONIZATION_TASK, "y", trial_id
                )
                sync_final_x, sync_final_y = _rollout_synchronization(
                    world=world,
                    policy=synchronization_policy,
                    initial_x=initial_x,
                    initial_y=initial_y,
                    horizon=config.horizon,
                    x_rng=random.Random(synchronization_rollout_x_seed),
                    y_rng=random.Random(synchronization_rollout_y_seed),
                )
                records.append(
                    TrialRecord(
                        trial_id=trial_id,
                        task=SYNCHRONIZATION_TASK,
                        n_x=n_x,
                        n_y=n_y,
                        n_total_per_context=n_x + n_y,
                        total_local_updates=config.total_local_updates(n_x, n_y),
                        initial_x=initial_x,
                        initial_y=initial_y,
                        target_x=None,
                        final_x=sync_final_x,
                        final_y=sync_final_y,
                        success=int(sync_final_x == sync_final_y),
                        sampled_model_value=synchronization_planned_value,
                        true_policy_success_probability=(
                            synchronization_true_probability
                        ),
                        initial_x_seed=initial_x_seed,
                        initial_y_seed=initial_y_seed,
                        target_seed=target_seed,
                        x_model_seed=x_model_seeds[n_x],
                        y_model_seed=y_model_seeds[n_y],
                        tie_seed=synchronization_tie_seed,
                        rollout_x_seed=synchronization_rollout_x_seed,
                        rollout_y_seed=synchronization_rollout_y_seed,
                    )
                )

    trial_tuple = tuple(records)
    return ClockExperimentResult(
        config=config,
        trials=trial_tuple,
        summaries=summarize_trials(trial_tuple, config=config),
        seeds=seeds.records(),
    )


_TRIAL_FIELDS = (
    "schema_version",
    "trial_id",
    "task",
    "n_x",
    "n_y",
    "n_total_per_context",
    "total_local_updates",
    "initial_x",
    "initial_y",
    "target_x",
    "final_x",
    "final_y",
    "success",
    "sampled_model_value",
    "true_policy_success_probability",
    "initial_x_seed",
    "initial_y_seed",
    "target_seed",
    "x_model_seed",
    "y_model_seed",
    "tie_seed",
    "rollout_x_seed",
    "rollout_y_seed",
)
_SUMMARY_FIELDS = (
    "schema_version",
    "task",
    "n_x",
    "n_y",
    "successes",
    "trials",
    "success_probability",
    "monte_carlo_standard_error",
    "wilson_95_lower",
    "wilson_95_upper",
)
_SEED_FIELDS = ("schema_version", "namespace", "seed")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text within an existing parent directory."""

    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _json_text(value: object) -> str:
    """Return canonical, human-readable JSON with a final newline."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _csv_text(
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> str:
    """Serialize deterministic CSV using explicit LF line endings."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


def _validate_run_id(run_id: str) -> str:
    """Reject ambiguous or path-like run identifiers."""

    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string.")
    if not _RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ValueError(
            "run_id must contain only letters, digits, '.', '_', and '-', "
            "must start with a letter or digit, and must not contain '..'."
        )
    return run_id


def _default_run_id(config: ClockExperimentConfig) -> str:
    """Create a unique timestamped identifier with a configuration fingerprint."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    fingerprint = hashlib.sha256(
        _json_text(config.to_dict()).encode("utf-8")
    ).hexdigest()[:12]
    return "two-clock-{}-{}".format(timestamp, fingerprint)


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_clock_experiment(
    result: ClockExperimentResult,
    output_root: Union[str, Path],
    run_id: Optional[str] = None,
) -> RunArtifacts:
    """Persist one result into a new, non-overwriting run directory.

    Args:
        result: Complete in-memory experiment result.
        output_root: Caller-controlled parent directory.
        run_id: Optional validated directory name. Omitting it creates a unique
            UTC timestamp plus configuration fingerprint.

    Returns:
        Paths to the written config, raw data, summary, seed ledger, figures
        directory, and manifest.

    Raises:
        FileExistsError: If the chosen run directory already exists.
        ValueError: If ``run_id`` is unsafe.

    Notes:
        The initial manifest covers config, CSV, and seed files. Call
        :func:`write_run_manifest` again after saving figures so the manifest
        includes them.
    """

    if not isinstance(result, ClockExperimentResult):
        raise TypeError("result must be a ClockExperimentResult.")
    root = Path(output_root)
    chosen_id = _default_run_id(result.config) if run_id is None else run_id
    validated_id = _validate_run_id(chosen_id)
    root.mkdir(parents=True, exist_ok=True)
    run_directory = root / validated_id
    run_directory.mkdir(exist_ok=False)
    figures_directory = run_directory / "figures"
    figures_directory.mkdir()

    config_path = run_directory / "config.json"
    trials_path = run_directory / "trials.csv"
    summary_path = run_directory / "summary.csv"
    seeds_path = run_directory / "seeds.csv"
    manifest_path = run_directory / "manifest.json"
    _atomic_write_text(config_path, _json_text(result.config.to_dict()))
    _atomic_write_text(
        trials_path,
        _csv_text(
            (record.to_row() for record in result.trials),
            _TRIAL_FIELDS,
        ),
    )
    _atomic_write_text(
        summary_path,
        _csv_text(
            (summary.to_row() for summary in result.summaries),
            _SUMMARY_FIELDS,
        ),
    )
    _atomic_write_text(
        seeds_path,
        _csv_text(
            (record.to_row() for record in result.seeds),
            _SEED_FIELDS,
        ),
    )
    write_run_manifest(run_directory)
    return RunArtifacts(
        run_directory=run_directory,
        config_path=config_path,
        trials_path=trials_path,
        summary_path=summary_path,
        seeds_path=seeds_path,
        figures_directory=figures_directory,
        manifest_path=manifest_path,
    )


def _safe_manifest_relative_path(run_directory: Path, relative: str) -> Path:
    """Resolve one manifest entry without allowing directory traversal."""

    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            "Manifest contains unsafe relative path {!r}.".format(relative)
        )
    resolved_root = run_directory.resolve()
    resolved_candidate = (run_directory / candidate).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            "Manifest path {!r} escapes the run directory.".format(relative)
        ) from error
    return resolved_candidate


def write_run_manifest(run_directory: Union[str, Path]) -> Path:
    """Create or refresh the versioned checksum manifest for a complete run.

    Args:
        run_directory: Existing nonsymlink run directory containing the
            canonical configuration, trials, summary, and seed-ledger files.

    Returns:
        Path to the atomically written ``manifest.json``.

    Raises:
        ValueError: If the directory is invalid, contains a symbolic link, or
            lacks any canonical run artifact.
    """

    directory = Path(run_directory)
    if directory.is_symlink():
        raise ValueError("run_directory must not be a symbolic link.")
    if not directory.is_dir():
        raise ValueError("run_directory must be an existing directory.")
    manifest_path = directory / "manifest.json"
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Run artifacts must not contain symbolic links.")
        if path == manifest_path or path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    listed_paths = {entry["path"] for entry in files}
    missing_required = [
        relative for relative in _REQUIRED_RUN_ARTIFACTS if relative not in listed_paths
    ]
    if missing_required:
        raise ValueError(
            "Run directory is missing required artifacts: {!r}.".format(
                missing_required
            )
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_versions": {
            "config": CONFIG_SCHEMA_VERSION,
            "trials": TRIAL_SCHEMA_VERSION,
            "summary": SUMMARY_SCHEMA_VERSION,
            "seeds": SEED_SCHEMA_VERSION,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_excludes_itself": True,
        "files": files,
    }
    _atomic_write_text(manifest_path, _json_text(manifest))
    return manifest_path


def verify_run_manifest(run_directory: Union[str, Path]) -> None:
    """Validate completeness, checksums, and path safety for one run.

    Args:
        run_directory: Existing run directory whose ``manifest.json`` should
            cover every file except the manifest itself.

    Raises:
        ValueError: If the manifest or directory is unsafe, malformed,
            incomplete, changed, or inconsistent with declared schemas.
    """

    directory = Path(run_directory)
    if directory.is_symlink():
        raise ValueError("run_directory must not be a symbolic link.")
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("Run artifacts must not contain symbolic links.")
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError("Run manifest does not exist.") from error
    except json.JSONDecodeError as error:
        raise ValueError("Run manifest is not valid JSON.") from error
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must contain a JSON object.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported run manifest schema version.")
    expected_schema_versions = {
        "config": CONFIG_SCHEMA_VERSION,
        "trials": TRIAL_SCHEMA_VERSION,
        "summary": SUMMARY_SCHEMA_VERSION,
        "seeds": SEED_SCHEMA_VERSION,
    }
    if manifest.get("artifact_schema_versions") != expected_schema_versions:
        raise ValueError("Run manifest artifact schema versions are invalid.")
    file_entries = manifest.get("files")
    if not isinstance(file_entries, list):
        raise ValueError("Run manifest files must be a list.")
    listed = set()
    for entry in file_entries:
        if not isinstance(entry, dict):
            raise ValueError("Each run manifest file entry must be an object.")
        relative = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
        ):
            raise ValueError("Run manifest contains a malformed file entry.")
        if relative in listed:
            raise ValueError("Run manifest contains duplicate file paths.")
        listed.add(relative)
        path = _safe_manifest_relative_path(directory, relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError("Manifest artifact {!r} is missing.".format(relative))
        if path.stat().st_size != expected_bytes:
            raise ValueError("Manifest byte size mismatch for {!r}.".format(relative))
        if _sha256_file(path) != expected_hash:
            raise ValueError("Manifest SHA-256 mismatch for {!r}.".format(relative))

    missing_required = [
        relative for relative in _REQUIRED_RUN_ARTIFACTS if relative not in listed
    ]
    if missing_required:
        raise ValueError(
            "Run manifest is missing required artifacts: {!r}.".format(missing_required)
        )

    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != listed:
        missing_entries = sorted(actual - listed)
        missing_files = sorted(listed - actual)
        raise ValueError(
            "Manifest coverage mismatch; unlisted={!r}, missing={!r}.".format(
                missing_entries, missing_files
            )
        )


def _read_csv_rows(
    path: Union[str, Path], fields: Sequence[str]
) -> List[Dict[str, str]]:
    """Read UTF-8 CSV and enforce its exact ordered header."""

    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(
                "CSV header does not match schema; expected {!r}, received {!r}.".format(
                    tuple(fields), tuple(reader.fieldnames or ())
                )
            )
        rows: List[Dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    "CSV row {} contains unexpected extra columns.".format(row_number)
                )
            cleaned: Dict[str, str] = {}
            for field in fields:
                value = row.get(field)
                if value is None:
                    raise ValueError(
                        "CSV row {} is missing field {!r}.".format(row_number, field)
                    )
                cleaned[field] = value
            rows.append(cleaned)
        return rows


def _parse_int(value: str, *, label: str) -> int:
    """Parse one exact base-10 integer without float coercion."""

    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ValueError("{} must be an integer.".format(label)) from error
    return parsed


def _parse_float(value: str, *, label: str) -> float:
    """Parse and validate one finite CSV float."""

    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError("{} must be a float.".format(label)) from error
    if not math.isfinite(parsed):
        raise ValueError("{} must be finite.".format(label))
    return parsed


def read_trial_csv(path: Union[str, Path]) -> Tuple[TrialRecord, ...]:
    """Load and validate the versioned raw-trial CSV artifact."""

    records = []
    for row_number, row in enumerate(_read_csv_rows(path, _TRIAL_FIELDS), start=2):
        if row["schema_version"] != TRIAL_SCHEMA_VERSION:
            raise ValueError(
                "Row {} has unsupported trial schema version.".format(row_number)
            )
        target_text = row["target_x"]
        records.append(
            TrialRecord(
                trial_id=_parse_int(row["trial_id"], label="trial_id"),
                task=row["task"],
                n_x=_parse_int(row["n_x"], label="n_x"),
                n_y=_parse_int(row["n_y"], label="n_y"),
                n_total_per_context=_parse_int(
                    row["n_total_per_context"],
                    label="n_total_per_context",
                ),
                total_local_updates=_parse_int(
                    row["total_local_updates"],
                    label="total_local_updates",
                ),
                initial_x=_parse_int(row["initial_x"], label="initial_x"),
                initial_y=_parse_int(row["initial_y"], label="initial_y"),
                target_x=(
                    None
                    if target_text == ""
                    else _parse_int(target_text, label="target_x")
                ),
                final_x=_parse_int(row["final_x"], label="final_x"),
                final_y=_parse_int(row["final_y"], label="final_y"),
                success=_parse_int(row["success"], label="success"),
                sampled_model_value=_parse_float(
                    row["sampled_model_value"],
                    label="sampled_model_value",
                ),
                true_policy_success_probability=_parse_float(
                    row["true_policy_success_probability"],
                    label="true_policy_success_probability",
                ),
                initial_x_seed=_parse_int(
                    row["initial_x_seed"], label="initial_x_seed"
                ),
                initial_y_seed=_parse_int(
                    row["initial_y_seed"], label="initial_y_seed"
                ),
                target_seed=_parse_int(row["target_seed"], label="target_seed"),
                x_model_seed=_parse_int(row["x_model_seed"], label="x_model_seed"),
                y_model_seed=_parse_int(row["y_model_seed"], label="y_model_seed"),
                tie_seed=_parse_int(row["tie_seed"], label="tie_seed"),
                rollout_x_seed=_parse_int(
                    row["rollout_x_seed"], label="rollout_x_seed"
                ),
                rollout_y_seed=_parse_int(
                    row["rollout_y_seed"], label="rollout_y_seed"
                ),
            )
        )
    return tuple(records)


def read_summary_csv(path: Union[str, Path]) -> Tuple[CellSummary, ...]:
    """Load and validate the versioned aggregate summary CSV artifact."""

    summaries = []
    for row_number, row in enumerate(_read_csv_rows(path, _SUMMARY_FIELDS), start=2):
        if row["schema_version"] != SUMMARY_SCHEMA_VERSION:
            raise ValueError(
                "Row {} has unsupported summary schema version.".format(row_number)
            )
        summaries.append(
            CellSummary(
                task=row["task"],
                n_x=_parse_int(row["n_x"], label="n_x"),
                n_y=_parse_int(row["n_y"], label="n_y"),
                successes=_parse_int(row["successes"], label="successes"),
                trials=_parse_int(row["trials"], label="trials"),
                success_probability=_parse_float(
                    row["success_probability"],
                    label="success_probability",
                ),
                monte_carlo_standard_error=_parse_float(
                    row["monte_carlo_standard_error"],
                    label="monte_carlo_standard_error",
                ),
                wilson_95_lower=_parse_float(
                    row["wilson_95_lower"],
                    label="wilson_95_lower",
                ),
                wilson_95_upper=_parse_float(
                    row["wilson_95_upper"],
                    label="wilson_95_upper",
                ),
            )
        )
    return tuple(summaries)


__all__ = [
    "ALLOCATION_STRATEGY_ORDER",
    "BEST_ABOVE_DIAGONAL_STRATEGY",
    "BEST_BELOW_DIAGONAL_STRATEGY",
    "CONFIG_SCHEMA_VERSION",
    "EQUAL_ALLOCATION_STRATEGY",
    "AllocationStrategyPoint",
    "ClockBeliefs",
    "ClockExperimentConfig",
    "ClockExperimentResult",
    "ClockWorld",
    "CellSummary",
    "HAND_ACTIONS",
    "MANIFEST_SCHEMA_VERSION",
    "NAVIGATION_TASK",
    "RunArtifacts",
    "SEED_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "SYNCHRONIZATION_TASK",
    "SeedRecord",
    "TRIAL_SCHEMA_VERSION",
    "TrialRecord",
    "build_clock_beliefs",
    "build_clock_world",
    "derive_seed",
    "read_summary_csv",
    "read_trial_csv",
    "run_clock_experiment",
    "select_allocation_strategies",
    "summarize_trials",
    "verify_run_manifest",
    "wilson_interval",
    "write_clock_experiment",
    "write_run_manifest",
]
