"""Contract tests for exact finite-horizon planning and policy evaluation."""

from __future__ import annotations

import math
import random
import unittest
from typing import Any

from scripts.mdp import (
    AbstractMDP,
    Assignment,
    TabularMDP,
    TransitionDistribution,
    Variable,
)
from scripts.planning import (
    FiniteHorizonPolicy,
    evaluate_finite_horizon_policy,
    plan_finite_horizon,
)


def position_one_reward(state: Assignment) -> float:
    """Reward terminal position one."""

    return float(state["position"] == 1)


def state_one_reward(state: Assignment) -> float:
    """Reward terminal generic state one."""

    return float(state["state"] == 1)


def zero_reward(state: Assignment) -> float:
    """Return the constant zero terminal reward."""

    del state
    return 0.0


def controlled_model(
    *,
    flip_probability: float = 1.0,
    state_name: str = "position",
) -> TabularMDP:
    """Build a binary stay/flip model in canonical state/action order."""

    state = Variable(state_name, (0, 1))
    action = Variable("action", ("stay", "flip"))
    rows = []
    for current in state.domain:
        for selected_action in action.domain:
            probability = 0.0 if selected_action == "stay" else float(flip_probability)
            rows.append(
                (
                    {state.name: current},
                    {"action": selected_action},
                    (
                        ({state.name: current}, 1.0 - probability),
                        ({state.name: 1 - current}, probability),
                    ),
                )
            )
    return TabularMDP(
        (state,),
        parent_variables=(action,),
        transitions=rows,
    )


def tied_model() -> TabularMDP:
    """Build a model whose two action values are exactly equal everywhere."""

    state = Variable("state", (0, 1))
    action = Variable("action", ("first", "second"))
    rows = [
        (
            {"state": current},
            {"action": selected_action},
            (({"state": current}, 1.0),),
        )
        for current in state.domain
        for selected_action in action.domain
    ]
    return TabularMDP((state,), parent_variables=(action,), transitions=rows)


class ConstantRandom:
    """Minimal injected random generator with a controlled draw."""

    def __init__(self, draw: object) -> None:
        self.draw = draw

    def random(self) -> object:
        return self.draw


class FailIfCalledRandom:
    """Verify that unique maximizers do not consume tie randomness."""

    def random(self) -> float:
        raise AssertionError("random() must only be called for an exact tie.")


class NonEnumerableDistribution(TransitionDistribution):
    """Finite-looking distribution that intentionally omits enumeration."""

    def _probability(self, outcome: Assignment) -> float:
        del outcome
        return 1.0

    def _sample(self, rng: Any) -> Assignment:
        del rng
        return Assignment({"state": 0})


class NonEnumerableMDP(AbstractMDP):
    """MDP fixture used to verify the exact-enumeration requirement."""

    def __init__(self) -> None:
        super().__init__(
            (Variable("state", (0, 1)),),
            (Variable("action", ("stay",)),),
        )

    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> TransitionDistribution:
        del current, parents
        return NonEnumerableDistribution(self.variables)


class DomainlessMDP(AbstractMDP):
    """Abstract-compatible model with a state domain planning cannot enumerate."""

    def __init__(self) -> None:
        super().__init__((Variable("state"),))

    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> TransitionDistribution:
        del current, parents
        raise AssertionError("Planning must reject the missing domain first.")


class CountingMDP(AbstractMDP):
    """Delegate transitions while recording conditioning-row query counts."""

    def __init__(self, delegate: TabularMDP) -> None:
        super().__init__(delegate.variables, delegate.parent_variables)
        self.delegate = delegate
        self.query_counts: dict[tuple[Assignment, Assignment], int] = {}

    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> TransitionDistribution:
        key = (current, parents)
        self.query_counts[key] = self.query_counts.get(key, 0) + 1
        return self.delegate.transition_distribution(current, parents)


class FiniteHorizonPlanningTests(unittest.TestCase):
    def test_backward_induction_produces_feedback_actions_and_values(self) -> None:
        model = controlled_model()

        policy = plan_finite_horizon(
            model,
            position_one_reward,
            horizon=1,
            rng=random.Random(11),
        )

        self.assertIsInstance(policy, FiniteHorizonPolicy)
        self.assertEqual(policy.variables, model.variables)
        self.assertEqual(policy.action_variables, model.parent_variables)
        self.assertEqual(policy.horizon, 1)
        self.assertEqual(policy.action(0, {"position": 0}), {"action": "flip"})
        self.assertEqual(policy.action(0, {"position": 1}), {"action": "stay"})
        self.assertEqual(policy.value(0, {"position": 0}), 1.0)
        self.assertEqual(policy.value(0, {"position": 1}), 1.0)
        self.assertEqual(policy.value(1, {"position": 0}), 0.0)
        self.assertEqual(policy.value(1, {"position": 1}), 1.0)

    def test_stochastic_value_is_an_exact_transition_expectation(self) -> None:
        model = controlled_model(flip_probability=0.75)

        policy = plan_finite_horizon(
            model,
            position_one_reward,
            horizon=1,
            rng=random.Random(3),
        )

        self.assertEqual(policy.action(0, {"position": 0}), {"action": "flip"})
        self.assertAlmostEqual(policy.value(0, {"position": 0}), 0.75)
        self.assertEqual(policy.action(0, {"position": 1}), {"action": "stay"})
        self.assertAlmostEqual(policy.value(0, {"position": 1}), 1.0)

    def test_zero_horizon_has_terminal_values_and_no_actions(self) -> None:
        model = controlled_model()
        policy = plan_finite_horizon(
            model,
            lambda state: float(state["position"]),
            horizon=0,
            rng=random.Random(5),
        )

        self.assertEqual(policy.value(0, {"position": 0}), 0.0)
        self.assertEqual(policy.value(0, {"position": 1}), 1.0)
        with self.assertRaisesRegex(ValueError, "action step"):
            policy.action(0, {"position": 0})

    def test_exact_ties_use_the_injected_random_stream(self) -> None:
        model = tied_model()

        first = plan_finite_horizon(
            model,
            zero_reward,
            horizon=1,
            rng=ConstantRandom(0.0),
        )
        second = plan_finite_horizon(
            model,
            zero_reward,
            horizon=1,
            rng=ConstantRandom(0.999999),
        )

        self.assertEqual(first.action(0, {"state": 0}), {"action": "first"})
        self.assertEqual(second.action(0, {"state": 0}), {"action": "second"})

    def test_unique_maximizers_do_not_consume_tie_randomness(self) -> None:
        policy = plan_finite_horizon(
            controlled_model(),
            position_one_reward,
            horizon=1,
            rng=FailIfCalledRandom(),
        )

        self.assertEqual(policy.action(0, {"position": 0}), {"action": "flip"})
        self.assertEqual(policy.action(0, {"position": 1}), {"action": "stay"})

    def test_seeded_tie_breaking_is_reproducible(self) -> None:
        model = tied_model()

        first = plan_finite_horizon(
            model,
            lambda state: float(state["state"]),
            horizon=4,
            rng=random.Random(2026),
        )
        second = plan_finite_horizon(
            model,
            lambda state: float(state["state"]),
            horizon=4,
            rng=random.Random(2026),
        )

        for step in range(4):
            for state in (0, 1):
                self.assertEqual(
                    first.action(step, {"state": state}),
                    second.action(step, {"state": state}),
                )

    def test_autonomous_model_uses_one_empty_action_assignment(self) -> None:
        state = Variable("state", (0, 1))
        model = TabularMDP(
            (state,),
            transitions=(
                ({"state": 0}, {}, (({"state": 1}, 1.0),)),
                ({"state": 1}, {}, (({"state": 1}, 1.0),)),
            ),
        )
        policy = plan_finite_horizon(
            model,
            state_one_reward,
            horizon=1,
            rng=random.Random(1),
        )

        self.assertEqual(policy.action_variables, ())
        self.assertEqual(policy.action(0, {"state": 0}), Assignment())
        self.assertEqual(policy.value(0, {"state": 0}), 1.0)
        self.assertEqual(
            evaluate_finite_horizon_policy(
                model,
                policy,
                state_one_reward,
                {"state": 0},
            ),
            1.0,
        )

    def test_exact_evaluation_matches_planned_value_on_same_model(self) -> None:
        model = controlled_model(flip_probability=0.7)
        policy = plan_finite_horizon(
            model,
            position_one_reward,
            horizon=3,
            rng=random.Random(19),
        )

        for state in (0, 1):
            exact = evaluate_finite_horizon_policy(
                model,
                policy,
                position_one_reward,
                {"position": state},
            )
            self.assertAlmostEqual(exact, policy.value(0, {"position": state}))

    def test_transition_rows_are_queried_once_per_planning_or_evaluation(self) -> None:
        reference_model = controlled_model(flip_probability=0.7)
        counting_model = CountingMDP(reference_model)
        policy = plan_finite_horizon(
            counting_model,
            position_one_reward,
            horizon=5,
            rng=random.Random(19),
        )

        self.assertEqual(len(counting_model.query_counts), 4)
        self.assertEqual(set(counting_model.query_counts.values()), {1})
        self.assertAlmostEqual(
            policy.value(0, {"position": 0}),
            1.0 - 0.3**5,
        )

        counting_model.query_counts.clear()
        exact = evaluate_finite_horizon_policy(
            counting_model,
            policy,
            position_one_reward,
            {"position": 0},
        )

        self.assertAlmostEqual(exact, policy.value(0, {"position": 0}))
        self.assertTrue(counting_model.query_counts)
        self.assertEqual(set(counting_model.query_counts.values()), {1})

    def test_exact_evaluation_can_use_different_true_dynamics(self) -> None:
        planning_model = controlled_model(flip_probability=1.0)
        true_model = controlled_model(flip_probability=0.0)
        policy = plan_finite_horizon(
            planning_model,
            position_one_reward,
            horizon=1,
            rng=random.Random(2),
        )

        self.assertEqual(policy.value(0, {"position": 0}), 1.0)
        self.assertEqual(
            evaluate_finite_horizon_policy(
                true_model,
                policy,
                position_one_reward,
                {"position": 0},
            ),
            0.0,
        )

    def test_public_queries_validate_steps_and_exact_state_scope(self) -> None:
        policy = plan_finite_horizon(
            controlled_model(),
            lambda state: float(state["position"]),
            horizon=1,
            rng=random.Random(8),
        )

        with self.assertRaises(TypeError):
            policy.action(True, {"position": 0})
        with self.assertRaises(ValueError):
            policy.value(2, {"position": 0})
        with self.assertRaisesRegex(ValueError, "missing"):
            policy.action(0, {})
        with self.assertRaisesRegex(ValueError, "unexpected"):
            policy.value(0, {"position": 0, "extra": 1})
        with self.assertRaisesRegex(ValueError, "outside the domain"):
            policy.action(0, {"position": 2})

    def test_planning_validates_horizon_reward_domains_and_rng(self) -> None:
        model = controlled_model()

        with self.assertRaises(TypeError):
            plan_finite_horizon(model, lambda state: 0.0, True)
        with self.assertRaises(TypeError):
            plan_finite_horizon(model, lambda state: 0.0, 1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            plan_finite_horizon(model, lambda state: 0.0, -1)
        with self.assertRaises(TypeError):
            plan_finite_horizon(model, "reward", 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            plan_finite_horizon(model, lambda state: "bad", 0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            plan_finite_horizon(model, lambda state: math.inf, 0)
        with self.assertRaises(TypeError):
            plan_finite_horizon(model, lambda state: 0.0, 1, rng=object())
        with self.assertRaises(ValueError):
            plan_finite_horizon(
                tied_model(),
                lambda state: 0.0,
                1,
                rng=ConstantRandom(1.0),
            )
        with self.assertRaisesRegex(ValueError, "finite domain"):
            plan_finite_horizon(DomainlessMDP(), lambda state: 0.0, 1)

    def test_planning_rejects_nonenumerable_transition_distributions(self) -> None:
        with self.assertRaisesRegex(TypeError, "finitely enumerable"):
            plan_finite_horizon(
                NonEnumerableMDP(),
                lambda state: 0.0,
                horizon=1,
                rng=random.Random(1),
            )

    def test_evaluation_rejects_incompatible_model_or_initial_state(self) -> None:
        policy = plan_finite_horizon(
            controlled_model(),
            lambda state: float(state["position"]),
            horizon=1,
            rng=random.Random(1),
        )

        with self.assertRaisesRegex(ValueError, "state variables"):
            evaluate_finite_horizon_policy(
                controlled_model(state_name="other"),
                policy,
                lambda state: 0.0,
                {"other": 0},
            )
        with self.assertRaisesRegex(ValueError, "Initial state"):
            evaluate_finite_horizon_policy(
                controlled_model(),
                policy,
                lambda state: 0.0,
                {},
            )


if __name__ == "__main__":
    unittest.main()
