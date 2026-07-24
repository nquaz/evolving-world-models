"""Contract tests for conjugate beliefs over finite transition tables."""

from __future__ import annotations

import inspect
import json
import math
import random
import subprocess
import sys
import unittest
from abc import ABC
from pathlib import Path
from typing import Optional

import scripts
from scripts.beliefs import (
    AbstractTransitionBelief,
    FactoredDirichletBelief,
    TabularDirichletBelief,
)
from scripts.mdp import (
    AbstractMDP,
    Assignment,
    FactoredMDP,
    ProductDistribution,
    TabularMDP,
    Variable,
)


def assignment(**values: object) -> Assignment:
    """Construct concise, hashable keys for concentration and count inspection."""

    return Assignment(values)


def context(
    current: dict[str, object],
    parents: Optional[dict[str, object]] = None,
) -> tuple[Assignment, Assignment]:
    """Return a canonical-looking public context key."""

    return Assignment(current), Assignment({} if parents is None else parents)


def binary_belief(*, prior: object = 1.0) -> TabularDirichletBelief:
    """Create a two-state, two-action belief used by focused tabular tests."""

    state = Variable("state", (0, 1))
    action = Variable("action", ("stay", "flip"))
    return TabularDirichletBelief(
        (state,),
        parent_variables=(action,),
        prior=prior,  # type: ignore[arg-type]
    )


def explicit_binary_prior_rows(
    *,
    reverse_rows: bool = False,
    reverse_outcomes: bool = False,
):
    """Build a complete, asymmetric prior with optional noncanonical ordering."""

    rows = []
    row_index = 0
    for current_state in (0, 1):
        for current_action in ("stay", "flip"):
            outcomes = [
                ({"state": 0}, float(row_index + 1)),
                ({"state": 1}, float(row_index + 2)),
            ]
            if reverse_outcomes:
                outcomes.reverse()
            rows.append(
                (
                    {"state": current_state},
                    {"action": current_action},
                    tuple(outcomes),
                )
            )
            row_index += 1
    if reverse_rows:
        rows.reverse()
    return tuple(rows)


def lock_door_belief(
    *,
    prior: float = 1.0,
) -> tuple[
    FactoredDirichletBelief,
    TabularDirichletBelief,
    TabularDirichletBelief,
]:
    """Create the canonical synchronous lock/door belief."""

    lock_state = Variable("lock", ("locked", "unlocked"))
    door_state = Variable("door", ("closed", "open"))
    action = Variable("action", ("open", "close"))
    lock_factor = TabularDirichletBelief((lock_state,), prior=prior)
    door_factor = TabularDirichletBelief(
        (door_state,),
        parent_variables=(lock_state, action),
        prior=prior,
    )
    return (
        FactoredDirichletBelief((lock_factor, door_factor)),
        lock_factor,
        door_factor,
    )


def assert_tabular_rows_normalized(
    test_case: unittest.TestCase,
    model: TabularMDP,
) -> None:
    """Assert every serialized finite row is a valid probability mass function."""

    for row in model.to_dict()["transitions"]:
        probabilities = [outcome["probability"] for outcome in row["outcomes"]]
        test_case.assertTrue(
            all(
                isinstance(probability, (int, float))
                and math.isfinite(probability)
                and 0.0 <= probability <= 1.0
                for probability in probabilities
            )
        )
        test_case.assertTrue(math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-12))


class ConstantGammaRng:
    """Minimal deterministic RNG test double for invalid gamma draws."""

    def __init__(self, value: object) -> None:
        self.value = value

    def gammavariate(self, shape: float, scale: float) -> object:
        del shape, scale
        return self.value


class AlternatingGammaRng:
    """Return one zero and one positive draw for every binary row."""

    def __init__(self) -> None:
        self.calls = 0

    def gammavariate(self, shape: float, scale: float) -> float:
        del shape, scale
        value = 0.0 if self.calls % 2 == 0 else 1.0
        self.calls += 1
        return value


class NonCallableGammaRng:
    """Expose the expected attribute with an invalid non-callable value."""

    gammavariate = 1.0


class AbstractTransitionBeliefTests(unittest.TestCase):
    def test_base_class_is_abstract_and_not_an_mdp(self) -> None:
        self.assertTrue(issubclass(AbstractTransitionBelief, ABC))
        self.assertTrue(inspect.isabstract(AbstractTransitionBelief))
        with self.assertRaises(TypeError):
            AbstractTransitionBelief()  # type: ignore[abstract]

        belief = binary_belief()
        self.assertNotIsInstance(belief, AbstractMDP)


class TabularDirichletBeliefTests(unittest.TestCase):
    def setUp(self) -> None:
        self.belief = binary_belief()
        (self.state,) = self.belief.variables
        (self.action,) = self.belief.parent_variables

    def test_metadata_and_symmetric_prior_cover_every_context(self) -> None:
        self.assertEqual(self.belief.variables, (self.state,))
        self.assertEqual(self.belief.parent_variables, (self.action,))
        self.assertEqual(self.belief.parents, (self.action,))

        priors = self.belief.prior_concentrations
        counts = self.belief.counts
        posteriors = self.belief.posterior_concentrations
        self.assertEqual(len(priors), 4)
        self.assertEqual(set(priors), set(counts))
        self.assertEqual(set(priors), set(posteriors))
        for row_context, prior_row in priors.items():
            self.assertEqual(
                prior_row,
                {
                    assignment(state=0): 1.0,
                    assignment(state=1): 1.0,
                },
            )
            self.assertEqual(
                counts[row_context],
                {
                    assignment(state=0): 0,
                    assignment(state=1): 0,
                },
            )
            self.assertEqual(posteriors[row_context], prior_row)

        for current_state in self.state.domain:
            for current_action in self.action.domain:
                distribution = self.belief.transition_distribution(
                    {"state": current_state},
                    {"action": current_action},
                )
                self.assertEqual(
                    tuple(outcome for outcome, _ in distribution.items()),
                    (assignment(state=0), assignment(state=1)),
                )
                self.assertAlmostEqual(distribution.probability({"state": 0}), 0.5)
                self.assertAlmostEqual(distribution.probability({"state": 1}), 0.5)

    def test_asymmetric_prior_controls_exact_predictive_probabilities(self) -> None:
        belief = binary_belief(prior=explicit_binary_prior_rows())

        self.assertAlmostEqual(
            belief.transition_probability(
                {"state": 0},
                {"state": 0},
                {"action": "stay"},
            ),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            belief.transition_probability(
                {"state": 1},
                {"state": 1},
                {"action": "flip"},
            ),
            5.0 / 9.0,
        )

    def test_repeated_updates_are_exact_and_isolated_by_context(self) -> None:
        target_context = context({"state": 0}, {"action": "stay"})
        target_outcome = assignment(state=1)

        for _ in range(3):
            self.belief.update(
                current={"state": 0},
                next_state={"state": 1},
                parents={"action": "stay"},
            )

        self.assertEqual(self.belief.counts[target_context][target_outcome], 3)
        self.assertEqual(
            self.belief.posterior_concentrations[target_context][target_outcome],
            4.0,
        )
        self.assertAlmostEqual(
            self.belief.transition_probability(
                {"state": 1},
                {"state": 0},
                {"action": "stay"},
            ),
            4.0 / 5.0,
        )
        self.assertAlmostEqual(
            self.belief.transition_log_probability(
                {"state": 1},
                {"state": 0},
                {"action": "stay"},
            ),
            math.log(4.0 / 5.0),
        )

        for row_context, count_row in self.belief.counts.items():
            expected_total = 3 if row_context == target_context else 0
            self.assertEqual(sum(count_row.values()), expected_total)
        self.assertAlmostEqual(
            self.belief.transition_probability(
                {"state": 1},
                {"state": 0},
                {"action": "flip"},
            ),
            0.5,
        )
        self.assertAlmostEqual(
            self.belief.transition_probability(
                {"state": 1},
                {"state": 1},
                {"action": "stay"},
            ),
            0.5,
        )

    def test_multi_variable_scope_uses_full_cartesian_outcome_support(self) -> None:
        x = Variable("x", (0, 1))
        y = Variable("y", ("a", "b"))
        belief = TabularDirichletBelief((x, y), prior=1.0)
        current = {"x": 0, "y": "a"}
        target = {"x": 1, "y": "b"}

        distribution = belief.transition_distribution(current)
        expected_outcomes = (
            assignment(x=0, y="a"),
            assignment(x=0, y="b"),
            assignment(x=1, y="a"),
            assignment(x=1, y="b"),
        )
        self.assertEqual(
            tuple(outcome for outcome, _ in distribution.items()),
            expected_outcomes,
        )
        for outcome in expected_outcomes:
            self.assertAlmostEqual(distribution.probability(outcome), 0.25)

        belief.update(current=current, next_state=target)
        updated = belief.transition_distribution(current)
        self.assertAlmostEqual(updated.probability(target), 2.0 / 5.0)
        for outcome in expected_outcomes[:-1]:
            self.assertAlmostEqual(updated.probability(outcome), 1.0 / 5.0)

    def test_invalid_observations_are_atomic(self) -> None:
        invalid_observations = (
            ({"state": 2}, {"state": 0}, {"action": "stay"}),
            ({"state": 0}, {"state": 2}, {"action": "stay"}),
            ({}, {"state": 0}, {"action": "stay"}),
            ({"state": 0, "extra": 1}, {"state": 0}, {"action": "stay"}),
            ({"state": 0}, {}, {"action": "stay"}),
            ({"state": 0}, {"state": 0, "extra": 1}, {"action": "stay"}),
            ({"state": 0}, {"state": 0}, {}),
            ({"state": 0}, {"state": 0}, {"action": "stay", "extra": 1}),
            ({"state": 0}, {"state": 0}, {"action": "unknown"}),
        )
        for current, next_state, parents in invalid_observations:
            with self.subTest(
                current=current,
                next_state=next_state,
                parents=parents,
            ):
                before = self.belief.counts
                with self.assertRaises((TypeError, ValueError)):
                    self.belief.update(current, next_state, parents)
                self.assertEqual(self.belief.counts, before)

        before = self.belief.counts
        with self.assertRaises((TypeError, ValueError)):
            self.belief.update(  # type: ignore[arg-type]
                current=["not", "a", "mapping"],
                next_state={"state": 0},
                parents={"action": "stay"},
            )
        self.assertEqual(self.belief.counts, before)

    def test_query_assignments_are_validated_exactly(self) -> None:
        invalid_queries = (
            lambda: self.belief.transition_distribution(
                {},
                {"action": "stay"},
            ),
            lambda: self.belief.transition_distribution(
                {"state": 0, "extra": 1},
                {"action": "stay"},
            ),
            lambda: self.belief.transition_distribution(
                {"state": 0},
                {},
            ),
            lambda: self.belief.transition_probability(
                {"state": 2},
                {"state": 0},
                {"action": "stay"},
            ),
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    query()

    def test_concentration_and_count_inspection_is_defensive(self) -> None:
        target_context = context({"state": 0}, {"action": "stay"})
        target_outcome = assignment(state=0)

        priors = self.belief.prior_concentrations
        counts = self.belief.counts
        posteriors = self.belief.posterior_concentrations
        priors[target_context][target_outcome] = 99.0
        counts[target_context][target_outcome] = 99
        posteriors[target_context][target_outcome] = 99.0
        priors.clear()
        counts[context({"state": 1}, {"action": "flip"})].clear()

        self.assertEqual(
            self.belief.prior_concentrations[target_context][target_outcome],
            1.0,
        )
        self.assertEqual(self.belief.counts[target_context][target_outcome], 0)
        self.assertEqual(
            self.belief.posterior_concentrations[target_context][target_outcome],
            1.0,
        )
        self.assertEqual(len(self.belief.prior_concentrations), 4)
        self.assertEqual(
            len(self.belief.counts[context({"state": 1}, {"action": "flip"})]),
            2,
        )

    def test_description_is_deterministic_and_canonical(self) -> None:
        canonical = binary_belief(prior=explicit_binary_prior_rows())
        reordered = binary_belief(
            prior=explicit_binary_prior_rows(
                reverse_rows=True,
                reverse_outcomes=True,
            )
        )
        canonical.update(
            current={"state": 1},
            next_state={"state": 0},
            parents={"action": "flip"},
        )
        reordered.update(
            current={"state": 1},
            next_state={"state": 0},
            parents={"action": "flip"},
        )

        self.assertEqual(canonical.to_dict(), reordered.to_dict())
        self.assertEqual(str(canonical), str(reordered))
        self.assertEqual(json.loads(str(canonical)), canonical.to_dict())
        description = canonical.to_dict()
        self.assertEqual(description["type"], "TabularDirichletBelief")
        self.assertEqual(
            description["variables"], [{"name": "state", "domain": [0, 1]}]
        )
        self.assertEqual(
            description["parent_variables"],
            [{"name": "action", "domain": ["stay", "flip"]}],
        )
        self.assertEqual(
            [row["current"] for row in description["contexts"]],
            [{"state": 0}, {"state": 0}, {"state": 1}, {"state": 1}],
        )
        self.assertEqual(
            [row["parents"] for row in description["contexts"]],
            [
                {"action": "stay"},
                {"action": "flip"},
                {"action": "stay"},
                {"action": "flip"},
            ],
        )
        self.assertEqual(
            [
                outcome["next_state"]
                for outcome in description["contexts"][0]["outcomes"]
            ],
            [{"state": 0}, {"state": 1}],
        )

    def test_posterior_mean_snapshots_are_fresh_and_independent(self) -> None:
        first_snapshot = self.belief.posterior_mean_mdp()
        second_snapshot = self.belief.posterior_mean_mdp()
        self.assertIsInstance(first_snapshot, TabularMDP)
        self.assertIsNot(first_snapshot, second_snapshot)
        self.assertEqual(first_snapshot.to_dict(), second_snapshot.to_dict())

        before = first_snapshot.transition_probability(
            {"state": 1},
            {"state": 0},
            {"action": "stay"},
        )
        self.belief.update(
            current={"state": 0},
            next_state={"state": 1},
            parents={"action": "stay"},
        )

        self.assertAlmostEqual(before, 0.5)
        self.assertAlmostEqual(
            first_snapshot.transition_probability(
                {"state": 1},
                {"state": 0},
                {"action": "stay"},
            ),
            0.5,
        )
        self.assertAlmostEqual(
            self.belief.posterior_mean_mdp().transition_probability(
                {"state": 1},
                {"state": 0},
                {"action": "stay"},
            ),
            2.0 / 3.0,
        )

    def test_log_probability_preserves_mass_after_float_underflow(self) -> None:
        state = Variable("state", (0, 1))
        tiny = 5e-324
        large = 1e308
        prior = tuple(
            (
                {"state": current},
                {},
                (
                    ({"state": 0}, tiny),
                    ({"state": 1}, large),
                ),
            )
            for current in state.domain
        )
        belief = TabularDirichletBelief((state,), prior=prior)
        current = {"state": 0}
        tiny_outcome = {"state": 0}

        self.assertEqual(
            belief.transition_probability(tiny_outcome, current),
            0.0,
        )
        expected_log_probability = math.log(tiny) - math.log(math.fsum((tiny, large)))
        self.assertTrue(math.isfinite(expected_log_probability))
        self.assertAlmostEqual(
            belief.transition_log_probability(tiny_outcome, current),
            expected_log_probability,
        )

        # A fixed TabularMDP stores normalized floats and therefore retains the
        # documented probability underflow rather than the concentration logs.
        snapshot = belief.posterior_mean_mdp()
        self.assertEqual(
            snapshot.transition_probability(tiny_outcome, current),
            0.0,
        )
        self.assertEqual(
            snapshot.transition_log_probability(tiny_outcome, current),
            -math.inf,
        )


class TabularPriorValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = Variable("state", (0, 1))
        self.action = Variable("action", ("stay", "flip"))

    def test_variable_and_parent_schemas_are_validated(self) -> None:
        invalid_constructors = (
            lambda: TabularDirichletBelief(()),
            lambda: TabularDirichletBelief(("state",)),  # type: ignore[arg-type]
            lambda: TabularDirichletBelief((self.state, self.state)),
            lambda: TabularDirichletBelief(
                (self.state,),
                parent_variables=(Variable("state", (0, 1)),),
            ),
            lambda: TabularDirichletBelief((Variable("continuous"),)),
            lambda: TabularDirichletBelief(
                (self.state,),
                parent_variables=(Variable("context"),),
            ),
            lambda: TabularDirichletBelief(
                (self.state,),
                parent_variables=("action",),  # type: ignore[arg-type]
            ),
        )
        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises((TypeError, ValueError)):
                    constructor()

    def test_scalar_prior_must_be_finite_positive_real_and_not_boolean(self) -> None:
        for prior in (True, False, "1", 0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(prior=prior):
                with self.assertRaises((TypeError, ValueError)):
                    TabularDirichletBelief((self.state,), prior=prior)  # type: ignore[arg-type]

        belief = TabularDirichletBelief((self.state,), prior=2)
        for row in belief.prior_concentrations.values():
            self.assertEqual(set(row.values()), {2.0})

    def test_explicit_prior_requires_complete_unique_contexts_and_outcomes(
        self,
    ) -> None:
        valid_rows = explicit_binary_prior_rows()
        first_current, first_parents, first_outcomes = valid_rows[0]
        malformed_priors = (
            (),
            valid_rows[:-1],
            valid_rows + (valid_rows[0],),
            ((first_current, first_parents),) + valid_rows[1:],
            (
                (
                    first_current,
                    first_parents,
                    first_outcomes[:-1],
                ),
            )
            + valid_rows[1:],
            (
                (
                    first_current,
                    first_parents,
                    (first_outcomes[0], first_outcomes[0]),
                ),
            )
            + valid_rows[1:],
            (
                (
                    {"state": 2},
                    first_parents,
                    first_outcomes,
                ),
            )
            + valid_rows[1:],
            (
                (
                    first_current,
                    {},
                    first_outcomes,
                ),
            )
            + valid_rows[1:],
            (
                (
                    first_current,
                    first_parents,
                    (({"state": 2}, 1.0), first_outcomes[1]),
                ),
            )
            + valid_rows[1:],
        )
        for prior in malformed_priors:
            with self.subTest(prior=prior):
                with self.assertRaises((TypeError, ValueError)):
                    binary_belief(prior=prior)

    def test_explicit_concentrations_are_validated(self) -> None:
        valid_rows = explicit_binary_prior_rows()
        first_current, first_parents, first_outcomes = valid_rows[0]
        for invalid in ("1", True, 0.0, -1.0, math.inf, -math.inf, math.nan):
            invalid_first_row = (
                first_current,
                first_parents,
                ((first_outcomes[0][0], invalid), first_outcomes[1]),
            )
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    binary_belief(prior=(invalid_first_row,) + valid_rows[1:])

        overflow_first_row = (
            first_current,
            first_parents,
            ((first_outcomes[0][0], 1e308), (first_outcomes[1][0], 1e308)),
        )
        with self.assertRaises((TypeError, ValueError, OverflowError)):
            binary_belief(prior=(overflow_first_row,) + valid_rows[1:])


class TabularSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.belief = binary_belief()
        self.belief.update(
            current={"state": 0},
            next_state={"state": 1},
            parents={"action": "flip"},
        )

    def test_seeded_samples_are_reproducible_normalized_and_fresh(self) -> None:
        first = self.belief.sample_mdp(random.Random(2026))
        second = self.belief.sample_mdp(random.Random(2026))
        third = self.belief.sample_mdp(random.Random(2027))

        self.assertIsInstance(first, TabularMDP)
        self.assertIsNot(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotEqual(first.to_dict(), third.to_dict())
        assert_tabular_rows_normalized(self, first)
        assert_tabular_rows_normalized(self, third)

        frozen_description = first.to_dict()
        self.belief.update(
            current={"state": 0},
            next_state={"state": 1},
            parents={"action": "flip"},
        )
        self.assertEqual(first.to_dict(), frozen_description)

    def test_none_rng_uses_a_private_generator_and_returns_valid_mdp(self) -> None:
        sampled = self.belief.sample_mdp()
        self.assertIsInstance(sampled, TabularMDP)
        assert_tabular_rows_normalized(self, sampled)

    def test_invalid_rng_interfaces_and_draws_are_rejected(self) -> None:
        invalid_rngs = (
            object(),
            NonCallableGammaRng(),
            ConstantGammaRng("not numeric"),
            ConstantGammaRng(-1.0),
            ConstantGammaRng(math.inf),
            ConstantGammaRng(math.nan),
            ConstantGammaRng(0.0),
        )
        for rng in invalid_rngs:
            with self.subTest(rng=rng):
                with self.assertRaises((TypeError, ValueError)):
                    self.belief.sample_mdp(rng)

    def test_individual_zero_gamma_draws_are_allowed_when_row_sum_is_positive(
        self,
    ) -> None:
        sampled = self.belief.sample_mdp(AlternatingGammaRng())
        assert_tabular_rows_normalized(self, sampled)
        for row in sampled.to_dict()["transitions"]:
            self.assertEqual(
                sorted(outcome["probability"] for outcome in row["outcomes"]),
                [0.0, 1.0],
            )


class FactoredDirichletBeliefTests(unittest.TestCase):
    def setUp(self) -> None:
        self.belief, self.lock_factor, self.door_factor = lock_door_belief()

    def test_metadata_factor_ownership_and_deterministic_description(self) -> None:
        self.assertEqual(
            self.belief.variables,
            (
                Variable("lock", ("locked", "unlocked")),
                Variable("door", ("closed", "open")),
            ),
        )
        self.assertEqual(
            self.belief.parent_variables,
            (Variable("action", ("open", "close")),),
        )
        self.assertEqual(self.belief.parents, self.belief.parent_variables)
        self.assertIsInstance(self.belief.factors, tuple)
        self.assertEqual(
            self.belief.factors,
            (self.lock_factor, self.door_factor),
        )
        with self.assertRaises(TypeError):
            self.belief.factors[0] = self.door_factor  # type: ignore[index]

        description = self.belief.to_dict()
        self.assertEqual(description["type"], "FactoredDirichletBelief")
        self.assertEqual(
            [factor["type"] for factor in description["factors"]],
            ["TabularDirichletBelief", "TabularDirichletBelief"],
        )
        self.assertEqual(json.loads(str(self.belief)), description)
        self.assertEqual(str(self.belief), str(self.belief))

    def test_posterior_prediction_is_exact_factor_product(self) -> None:
        self.lock_factor.update(
            current={"lock": "unlocked"},
            next_state={"lock": "locked"},
        )
        self.door_factor.update(
            current={"door": "closed"},
            next_state={"door": "open"},
            parents={"lock": "unlocked", "action": "open"},
        )
        current = {"lock": "unlocked", "door": "closed"}
        parents = {"action": "open"}
        next_state = {"lock": "locked", "door": "open"}

        distribution = self.belief.transition_distribution(current, parents)
        self.assertIsInstance(distribution, ProductDistribution)
        self.assertAlmostEqual(
            self.belief.transition_probability(next_state, current, parents),
            (2.0 / 3.0) * (2.0 / 3.0),
        )
        self.assertAlmostEqual(
            self.belief.transition_log_probability(next_state, current, parents),
            math.log(2.0 / 3.0) + math.log(2.0 / 3.0),
        )
        self.assertTrue(
            math.isclose(
                math.fsum(probability for _, probability in distribution.items()),
                1.0,
                abs_tol=1e-12,
            )
        )

    def test_joint_update_updates_every_factor_once_using_current_lock(self) -> None:
        self.belief.update(
            current={"lock": "unlocked", "door": "closed"},
            next_state={"lock": "locked", "door": "open"},
            parents={"action": "open"},
        )

        lock_context = context({"lock": "unlocked"})
        door_current_unlocked = context(
            {"door": "closed"},
            {"lock": "unlocked", "action": "open"},
        )
        door_current_locked = context(
            {"door": "closed"},
            {"lock": "locked", "action": "open"},
        )
        self.assertEqual(
            self.lock_factor.counts[lock_context][assignment(lock="locked")],
            1,
        )
        self.assertEqual(
            self.door_factor.counts[door_current_unlocked][assignment(door="open")],
            1,
        )
        self.assertEqual(
            self.door_factor.counts[door_current_locked][assignment(door="open")],
            0,
        )
        self.assertEqual(
            sum(
                count
                for row in self.lock_factor.counts.values()
                for count in row.values()
            ),
            1,
        )
        self.assertEqual(
            sum(
                count
                for row in self.door_factor.counts.values()
                for count in row.values()
            ),
            1,
        )

    def test_invalid_joint_updates_leave_all_factor_counts_unchanged(self) -> None:
        invalid_observations = (
            (
                {"lock": "unlocked", "door": "closed"},
                {"lock": "locked", "door": "ajar"},
                {"action": "open"},
            ),
            (
                {"lock": "unlocked"},
                {"lock": "locked", "door": "open"},
                {"action": "open"},
            ),
            (
                {"lock": "unlocked", "door": "closed"},
                {"lock": "locked", "door": "open"},
                {},
            ),
            (
                {"lock": "unlocked", "door": "closed"},
                {"lock": "locked", "door": "open"},
                {"action": "open", "lock": "unlocked"},
            ),
        )
        for current, next_state, parents in invalid_observations:
            with self.subTest(
                current=current,
                next_state=next_state,
                parents=parents,
            ):
                lock_before = self.lock_factor.counts
                door_before = self.door_factor.counts
                with self.assertRaises((TypeError, ValueError)):
                    self.belief.update(current, next_state, parents)
                self.assertEqual(self.lock_factor.counts, lock_before)
                self.assertEqual(self.door_factor.counts, door_before)

    def test_factored_queries_validate_exact_joint_assignments(self) -> None:
        invalid_queries = (
            lambda: self.belief.transition_distribution(
                {"lock": "unlocked"},
                {"action": "open"},
            ),
            lambda: self.belief.transition_distribution(
                {"lock": "unlocked", "door": "closed", "extra": 1},
                {"action": "open"},
            ),
            lambda: self.belief.transition_distribution(
                {"lock": "unlocked", "door": "closed"},
                {},
            ),
            lambda: self.belief.transition_probability(
                {"lock": "locked", "door": "ajar"},
                {"lock": "unlocked", "door": "closed"},
                {"action": "open"},
            ),
        )
        for query in invalid_queries:
            with self.subTest(query=query):
                with self.assertRaises((TypeError, ValueError)):
                    query()

    def test_factor_composition_validation(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            FactoredDirichletBelief(())
        with self.assertRaises((TypeError, ValueError)):
            FactoredDirichletBelief((object(),))  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            FactoredDirichletBelief((self.belief,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FactoredDirichletBelief((self.lock_factor, self.lock_factor))

        incompatible_lock = Variable("lock", ("secured", "free"))
        door = Variable("door", ("closed", "open"))
        action = Variable("action", ("open", "close"))
        incompatible_door_factor = TabularDirichletBelief(
            (door,),
            parent_variables=(incompatible_lock, action),
        )
        with self.assertRaises(ValueError):
            FactoredDirichletBelief((self.lock_factor, incompatible_door_factor))

    def test_external_parent_metadata_is_deduplicated_in_first_seen_order(
        self,
    ) -> None:
        left_state = Variable("left", (0, 1))
        right_state = Variable("right", (0, 1))
        left_context = Variable("left_context", ("low", "high"))
        shared_context = Variable("shared_context", ("off", "on"))
        right_context = Variable("right_context", ("cold", "hot"))
        left_factor = TabularDirichletBelief(
            (left_state,),
            parent_variables=(left_context, shared_context),
        )
        right_factor = TabularDirichletBelief(
            (right_state,),
            parent_variables=(shared_context, right_context),
        )

        factored = FactoredDirichletBelief((left_factor, right_factor))

        self.assertEqual(
            factored.parent_variables,
            (left_context, shared_context, right_context),
        )

    def test_factored_mean_snapshots_are_fresh_and_independent(self) -> None:
        first = self.belief.posterior_mean_mdp()
        second = self.belief.posterior_mean_mdp()
        self.assertIsInstance(first, FactoredMDP)
        self.assertIsNot(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(all(isinstance(factor, TabularMDP) for factor in first.factors))
        self.assertTrue(
            all(left is not right for left, right in zip(first.factors, second.factors))
        )

        current = {"lock": "unlocked", "door": "closed"}
        parents = {"action": "open"}
        next_state = {"lock": "locked", "door": "open"}
        frozen_probability = first.transition_probability(
            next_state,
            current,
            parents,
        )
        self.belief.update(current, next_state, parents)

        self.assertAlmostEqual(frozen_probability, 0.25)
        self.assertAlmostEqual(
            first.transition_probability(next_state, current, parents),
            0.25,
        )
        self.assertAlmostEqual(
            self.belief.posterior_mean_mdp().transition_probability(
                next_state,
                current,
                parents,
            ),
            4.0 / 9.0,
        )

    def test_factored_samples_are_reproducible_normalized_and_independent(
        self,
    ) -> None:
        first = self.belief.sample_mdp(random.Random(90210))
        second = self.belief.sample_mdp(random.Random(90210))
        self.assertIsInstance(first, FactoredMDP)
        self.assertIsNot(first, second)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertTrue(all(isinstance(factor, TabularMDP) for factor in first.factors))
        for factor in first.factors:
            assert_tabular_rows_normalized(self, factor)

        frozen_description = first.to_dict()
        self.belief.update(
            current={"lock": "unlocked", "door": "closed"},
            next_state={"lock": "locked", "door": "open"},
            parents={"action": "open"},
        )
        self.assertEqual(first.to_dict(), frozen_description)


class ImportBoundaryTests(unittest.TestCase):
    def test_belief_classes_are_exported_from_package_root(self) -> None:
        expected_exports = {
            "AbstractTransitionBelief": AbstractTransitionBelief,
            "FactoredDirichletBelief": FactoredDirichletBelief,
            "TabularDirichletBelief": TabularDirichletBelief,
        }

        for name, expected in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(getattr(scripts, name), expected)
                self.assertIn(name, scripts.__all__)

    def test_belief_import_does_not_require_optional_dependencies(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        code = """
import builtins

forbidden = {"graphviz", "matplotlib", "networkx", "numpy", "scipy"}
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in forbidden:
        raise AssertionError("optional dependency imported: {}".format(name))
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import scripts.beliefs
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repository_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg="stdout:\n{}\nstderr:\n{}".format(
                completed.stdout,
                completed.stderr,
            ),
        )


if __name__ == "__main__":
    unittest.main()
