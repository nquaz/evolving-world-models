"""Contract tests for the reproducible two-clock allocation experiment."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path
from unittest import mock

import scripts
from scripts.beliefs import TabularDirichletBelief
from scripts.clock_experiment import (
    ALLOCATION_STRATEGY_ORDER,
    BEST_ABOVE_DIAGONAL_STRATEGY,
    BEST_BELOW_DIAGONAL_STRATEGY,
    CONFIG_SCHEMA_VERSION,
    EQUAL_ALLOCATION_STRATEGY,
    HAND_ACTIONS,
    MANIFEST_SCHEMA_VERSION,
    NAVIGATION_TASK,
    SEED_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    SYNCHRONIZATION_TASK,
    TRIAL_SCHEMA_VERSION,
    AllocationStrategyPoint,
    CellSummary,
    ClockExperimentConfig,
    ClockExperimentResult,
    build_clock_beliefs,
    build_clock_world,
    derive_seed,
    read_summary_csv,
    read_trial_csv,
    run_clock_experiment,
    select_allocation_strategies,
    summarize_trials,
    verify_run_manifest,
    wilson_interval,
    write_clock_experiment,
    write_run_manifest,
)


def tiny_config(**overrides: object) -> ClockExperimentConfig:
    """Build a fast, nontrivial configuration for runner and artifact tests."""

    values = {
        "num_states": 3,
        "intended_probability": 0.8,
        "predictable_direction": "right",
        "horizon": 1,
        "trials": 2,
        "x_updates": (0, 1),
        "y_updates": (0, 1),
        "fixed_total_budgets": (1,),
        "prior_concentration": 1.0,
        "master_seed": 8675309,
    }
    values.update(overrides)
    return ClockExperimentConfig(**values)  # type: ignore[arg-type]


def trial_identity(record):
    """Return the durable identity of one raw task evaluation."""

    return record.task, record.trial_id, record.n_x, record.n_y


def cell_summary(
    task: str,
    n_x: int,
    n_y: int,
    successes: int,
    trials: int = 10,
) -> CellSummary:
    """Build one internally consistent summary for selector tests."""

    probability = successes / trials
    lower, upper = wilson_interval(successes, trials)
    return CellSummary(
        task=task,
        n_x=n_x,
        n_y=n_y,
        successes=successes,
        trials=trials,
        success_probability=probability,
        monte_carlo_standard_error=math.sqrt(
            probability * (1.0 - probability) / trials
        ),
        wilson_95_lower=lower,
        wilson_95_upper=upper,
    )


def allocation_summaries():
    """Return complete diagonals with deliberate maxima and ties."""

    success_counts = {
        NAVIGATION_TASK: {
            (0, 2): 4,
            (1, 1): 5,
            (2, 0): 7,
            (0, 4): 6,
            (1, 3): 6,
            (2, 2): 5,
            (3, 1): 8,
            (4, 0): 9,
        },
        SYNCHRONIZATION_TASK: {
            (0, 2): 8,
            (1, 1): 6,
            (2, 0): 4,
            (0, 4): 9,
            (1, 3): 5,
            (2, 2): 6,
            (3, 1): 7,
            (4, 0): 7,
        },
    }
    return tuple(
        cell_summary(task, n_x, n_y, successes)
        for task in (NAVIGATION_TASK, SYNCHRONIZATION_TASK)
        for (n_x, n_y), successes in success_counts[task].items()
    )


class ConfigurationAndSeedTests(unittest.TestCase):
    def test_grids_are_canonical_and_budget_accounting_is_explicit(self) -> None:
        config = tiny_config(
            x_updates=(3, 0, 1),
            y_updates=(2, 0),
            fixed_total_budgets=(3, 1),
        )

        self.assertEqual(config.x_updates, (0, 1, 3))
        self.assertEqual(config.y_updates, (0, 2))
        self.assertEqual(config.fixed_total_budgets, (1, 3))
        self.assertEqual(config.contexts_per_factor, 9)
        self.assertEqual(config.total_local_updates(3, 2), 45)
        self.assertEqual(config.to_dict()["schema_version"], CONFIG_SCHEMA_VERSION)
        self.assertEqual(config.to_dict()["contexts_per_factor"], 9)
        with self.assertRaisesRegex(ValueError, "not a configured grid cell"):
            config.total_local_updates(2, 2)

    def test_invalid_scientific_configuration_is_rejected(self) -> None:
        invalid_cases = (
            ({"num_states": 2}, ValueError, "at least 3"),
            (
                {"intended_probability": 1.0 / 3.0},
                ValueError,
                "greater than",
            ),
            ({"intended_probability": 1.1}, ValueError, "at most"),
            ({"predictable_direction": "stay"}, ValueError, "left.*right"),
            ({"horizon": -1}, ValueError, "at least 0"),
            ({"trials": 0}, ValueError, "at least 1"),
            ({"x_updates": (0, 0)}, ValueError, "duplicate"),
            ({"y_updates": (-1,)}, ValueError, "at least 0"),
            (
                {"fixed_total_budgets": (99,)},
                ValueError,
                "no represented",
            ),
            ({"prior_concentration": 0.0}, ValueError, "greater than"),
            ({"master_seed": True}, TypeError, "integer"),
        )
        for override, error_type, message in invalid_cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(error_type, message):
                    tiny_config(**override)

    def test_seed_derivation_is_stable_namespaced_and_non_global(self) -> None:
        first = derive_seed(1234, "observation", "x", 0, 1, "left")
        second = derive_seed(1234, "observation", "x", 0, 1, "left")

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**64)
        self.assertNotEqual(
            first,
            derive_seed(1234, "observation", "x", 0, 1, "right"),
        )
        self.assertNotEqual(
            first,
            derive_seed(1235, "observation", "x", 0, 1, "left"),
        )
        with self.assertRaisesRegex(TypeError, "master_seed"):
            derive_seed(True, "namespace")


class ClockConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config(
            num_states=4,
            intended_probability=0.7,
            predictable_direction="left",
            x_updates=(0,),
            y_updates=(0,),
            fixed_total_budgets=(0,),
            prior_concentration=1.25,
        )
        self.world = build_clock_world(self.config)

    def test_true_rows_wrap_normalize_and_share_the_probability_contract(
        self,
    ) -> None:
        self.assertEqual(self.world.x.name, "x")
        self.assertEqual(self.world.y.name, "y")
        self.assertEqual(self.world.hand.domain, HAND_ACTIONS)
        self.assertEqual(
            tuple(variable.name for variable in self.world.model.variables),
            ("x", "y"),
        )
        self.assertEqual(
            tuple(variable.name for variable in self.world.model.parent_variables),
            ("hand",),
        )

        residual = 0.1
        action_displacements = {"left": -1, "right": 1, "stay": 0}
        for state in range(self.config.num_states):
            for action, displacement in action_displacements.items():
                x_distribution = self.world.x_factor.transition_distribution(
                    {"x": state},
                    {"hand": action},
                )
                x_intended = (state + displacement) % self.config.num_states
                x_probabilities = [
                    x_distribution.probability({"x": next_state})
                    for next_state in range(self.config.num_states)
                ]
                self.assertAlmostEqual(math.fsum(x_probabilities), 1.0)
                for next_state, probability in enumerate(x_probabilities):
                    self.assertAlmostEqual(
                        probability,
                        0.7 if next_state == x_intended else residual,
                    )

                y_distribution = self.world.y_factor.transition_distribution(
                    {"y": state},
                    {"hand": action},
                )
                y_intended = (state - 1) % self.config.num_states
                y_probabilities = [
                    y_distribution.probability({"y": next_state})
                    for next_state in range(self.config.num_states)
                ]
                self.assertAlmostEqual(math.fsum(y_probabilities), 1.0)
                for next_state, probability in enumerate(y_probabilities):
                    self.assertAlmostEqual(
                        probability,
                        0.7 if next_state == y_intended else residual,
                    )

            reference = self.world.y_factor.transition_distribution(
                {"y": state},
                {"hand": "left"},
            )
            for action in HAND_ACTIONS[1:]:
                candidate = self.world.y_factor.transition_distribution(
                    {"y": state},
                    {"hand": action},
                )
                self.assertEqual(reference.items(), candidate.items())

    def test_both_beliefs_keep_every_action_conditioned_context(self) -> None:
        beliefs = build_clock_beliefs(self.config, self.world)

        self.assertEqual(beliefs.model.factors, (beliefs.x, beliefs.y))
        self.assertEqual(
            tuple(variable.name for variable in beliefs.model.parent_variables),
            ("hand",),
        )
        for factor, variable_name in ((beliefs.x, "x"), (beliefs.y, "y")):
            priors = factor.prior_concentrations
            counts = factor.counts
            self.assertEqual(len(priors), self.config.contexts_per_factor)
            self.assertEqual(set(priors), set(counts))
            observed_actions = set()
            for (current, parents), prior_row in priors.items():
                self.assertEqual(tuple(current), (variable_name,))
                observed_actions.add(parents["hand"])
                self.assertEqual(len(prior_row), self.config.num_states)
                self.assertEqual(set(prior_row.values()), {1.25})
                self.assertEqual(sum(counts[(current, parents)].values()), 0)
            self.assertEqual(observed_actions, set(HAND_ACTIONS))


class ClockExperimentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tiny_config()
        cls.result = run_clock_experiment(cls.config)

    def test_runner_spends_the_exact_budget_on_every_local_context(self) -> None:
        config = tiny_config(
            horizon=0,
            trials=1,
            x_updates=(0, 2),
            y_updates=(1, 3),
            fixed_total_budgets=(),
        )
        original_sample = TabularDirichletBelief.sample_mdp
        snapshots = []

        def recording_sample(belief, rng=None):
            row_totals = tuple(sum(row.values()) for row in belief.counts.values())
            snapshots.append((belief.variables[0].name, row_totals))
            return original_sample(belief, rng)

        with mock.patch.object(
            TabularDirichletBelief,
            "sample_mdp",
            new=recording_sample,
        ):
            result = run_clock_experiment(config)

        self.assertEqual(
            [(name, set(totals), len(totals)) for name, totals in snapshots],
            [
                ("x", {0}, 9),
                ("x", {2}, 9),
                ("y", {1}, 9),
                ("y", {3}, 9),
            ],
        )
        self.assertEqual(len(snapshots), 4)
        for record in result.trials:
            self.assertEqual(
                record.total_local_updates,
                9 * (record.n_x + record.n_y),
            )

    def test_complete_run_is_deterministic_nested_and_grid_order_invariant(
        self,
    ) -> None:
        reversed_config = tiny_config(
            x_updates=(1, 0),
            y_updates=(1, 0),
        )
        self.assertEqual(reversed_config, self.config)
        self.assertEqual(run_clock_experiment(reversed_config), self.result)
        self.assertEqual(run_clock_experiment(self.config), self.result)

        expanded = tiny_config(
            trials=2,
            x_updates=(0, 1, 2),
            y_updates=(0, 1, 2),
            fixed_total_budgets=(1, 2),
        )
        expanded_result = run_clock_experiment(expanded)
        expanded_trials = {
            trial_identity(record): record for record in expanded_result.trials
        }
        for record in self.result.trials:
            self.assertEqual(
                expanded_trials[trial_identity(record)],
                record,
            )

        expanded_seeds = {
            record.namespace: record.seed for record in expanded_result.seeds
        }
        for record in self.result.seeds:
            self.assertEqual(expanded_seeds[record.namespace], record.seed)

    def test_raw_records_have_exact_counts_pairing_and_factor_isolation(self) -> None:
        expected_records = (
            2
            * self.config.trials
            * len(self.config.x_updates)
            * len(self.config.y_updates)
        )
        self.assertEqual(len(self.result.trials), expected_records)
        self.assertEqual(
            len(self.result.summaries),
            2 * len(self.config.x_updates) * len(self.config.y_updates),
        )

        identities = {trial_identity(record) for record in self.result.trials}
        self.assertEqual(len(identities), expected_records)
        for record in self.result.trials:
            self.assertIn(record.success, (0, 1))
            self.assertEqual(record.n_total_per_context, record.n_x + record.n_y)
            self.assertEqual(
                record.total_local_updates,
                self.config.contexts_per_factor * (record.n_x + record.n_y),
            )
            self.assertEqual(
                record.initial_x_seed,
                derive_seed(
                    self.config.master_seed,
                    "task-instance",
                    "initial-x",
                    record.trial_id,
                ),
            )
            self.assertEqual(
                record.x_model_seed,
                derive_seed(
                    self.config.master_seed,
                    "posterior-model",
                    "x",
                    record.trial_id,
                    record.n_x,
                ),
            )
            self.assertEqual(
                record.y_model_seed,
                derive_seed(
                    self.config.master_seed,
                    "posterior-model",
                    "y",
                    record.trial_id,
                    record.n_y,
                ),
            )

        for trial_id in range(self.config.trials):
            trial_records = [
                record for record in self.result.trials if record.trial_id == trial_id
            ]
            self.assertEqual(
                len({record.initial_x for record in trial_records}),
                1,
            )
            self.assertEqual(
                len({record.initial_y for record in trial_records}),
                1,
            )
            self.assertEqual(
                len({record.target_seed for record in trial_records}),
                1,
            )
            for n_x in self.config.x_updates:
                self.assertEqual(
                    len(
                        {
                            record.x_model_seed
                            for record in trial_records
                            if record.n_x == n_x
                        }
                    ),
                    1,
                )
            for n_y in self.config.y_updates:
                self.assertEqual(
                    len(
                        {
                            record.y_model_seed
                            for record in trial_records
                            if record.n_y == n_y
                        }
                    ),
                    1,
                )

        seed_namespaces = [record.namespace for record in self.result.seeds]
        self.assertEqual(len(seed_namespaces), len(set(seed_namespaces)))
        parsed_namespaces = {tuple(json.loads(value)) for value in seed_namespaces}
        observation_namespaces = {
            namespace
            for namespace in parsed_namespaces
            if namespace[0] == "observation"
        }
        self.assertEqual(
            len(observation_namespaces),
            2 * self.config.trials * self.config.contexts_per_factor,
        )

    def test_navigation_is_exactly_invariant_to_predictable_budget(self) -> None:
        for trial_id in range(self.config.trials):
            for n_x in self.config.x_updates:
                records = sorted(
                    (
                        record
                        for record in self.result.trials
                        if record.task == NAVIGATION_TASK
                        and record.trial_id == trial_id
                        and record.n_x == n_x
                    ),
                    key=lambda record: record.n_y,
                )
                projections = {
                    (
                        record.initial_x,
                        record.initial_y,
                        record.target_x,
                        record.final_x,
                        record.final_y,
                        record.success,
                        record.sampled_model_value,
                        record.true_policy_success_probability,
                        record.x_model_seed,
                        record.tie_seed,
                        record.rollout_x_seed,
                        record.rollout_y_seed,
                    )
                    for record in records
                }
                self.assertEqual(len(projections), 1)
                self.assertEqual(
                    {record.n_y for record in records},
                    set(self.config.y_updates),
                )

        navigation_summaries = [
            summary
            for summary in self.result.summaries
            if summary.task == NAVIGATION_TASK
        ]
        for n_x in self.config.x_updates:
            probabilities = {
                summary.success_probability
                for summary in navigation_summaries
                if summary.n_x == n_x
            }
            self.assertEqual(len(probabilities), 1)

    def test_summaries_exactly_aggregate_binary_trials_and_wilson_bounds(
        self,
    ) -> None:
        self.assertEqual(
            summarize_trials(self.result.trials, config=self.config),
            self.result.summaries,
        )
        expected_keys = [
            (task, n_x, n_y)
            for task in (NAVIGATION_TASK, SYNCHRONIZATION_TASK)
            for n_y in self.config.y_updates
            for n_x in self.config.x_updates
        ]
        self.assertEqual(
            [
                (summary.task, summary.n_x, summary.n_y)
                for summary in self.result.summaries
            ],
            expected_keys,
        )
        for summary in self.result.summaries:
            records = [
                record
                for record in self.result.trials
                if (
                    record.task,
                    record.n_x,
                    record.n_y,
                )
                == (summary.task, summary.n_x, summary.n_y)
            ]
            successes = sum(record.success for record in records)
            probability = successes / self.config.trials
            self.assertEqual(summary.successes, successes)
            self.assertEqual(summary.trials, self.config.trials)
            self.assertEqual(summary.success_probability, probability)
            self.assertAlmostEqual(
                summary.monte_carlo_standard_error,
                math.sqrt(probability * (1.0 - probability) / self.config.trials),
            )
            self.assertEqual(
                (summary.wilson_95_lower, summary.wilson_95_upper),
                wilson_interval(successes, self.config.trials),
            )

        with self.assertRaisesRegex(ValueError, "Duplicate raw trial identity"):
            summarize_trials(
                self.result.trials + (self.result.trials[0],),
                config=self.config,
            )
        with self.assertRaisesRegex(ValueError, "trial ids|missing task/cell"):
            summarize_trials(self.result.trials[1:], config=self.config)

    def test_wilson_interval_matches_known_binomial_cases(self) -> None:
        lower_zero, upper_zero = wilson_interval(0, 10)
        lower_half, upper_half = wilson_interval(5, 10)
        lower_full, upper_full = wilson_interval(10, 10)

        self.assertEqual(lower_zero, 0.0)
        self.assertAlmostEqual(upper_zero, 0.2775327998628892)
        self.assertAlmostEqual(lower_half, 0.236593090512564)
        self.assertAlmostEqual(upper_half, 0.7634069094874361)
        self.assertAlmostEqual(lower_full, 1.0 - upper_zero)
        self.assertAlmostEqual(upper_full, 1.0)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            wilson_interval(2, 1)
        with self.assertRaisesRegex(TypeError, "successes"):
            wilson_interval(True, 1)

    def test_records_summaries_and_complete_results_cross_validate(self) -> None:
        with self.assertRaisesRegex(TypeError, "success"):
            replace(self.result.trials[0], success=1.0)
        normalized_record = replace(
            self.result.trials[0],
            sampled_model_value=Fraction(1, 2),
            true_policy_success_probability=Fraction(1, 2),
        )
        self.assertIs(type(normalized_record.sampled_model_value), float)
        self.assertIs(type(normalized_record.true_policy_success_probability), float)
        for task in (NAVIGATION_TASK, SYNCHRONIZATION_TASK):
            record = next(
                candidate for candidate in self.result.trials if candidate.task == task
            )
            with self.subTest(task=task):
                with self.assertRaisesRegex(ValueError, "terminal task outcome"):
                    replace(record, success=1 - record.success)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            replace(
                self.result.summaries[0],
                success_probability=0.123,
            )
        normalized_summary = replace(
            self.result.summaries[0],
            success_probability=Fraction(
                self.result.summaries[0].successes,
                self.result.summaries[0].trials,
            ),
        )
        self.assertIs(type(normalized_summary.success_probability), float)
        with self.assertRaisesRegex(ValueError, "canonical aggregation"):
            ClockExperimentResult(
                config=self.config,
                trials=self.result.trials,
                summaries=tuple(reversed(self.result.summaries)),
                seeds=self.result.seeds,
            )
        records_with_swapped_seed = list(self.result.trials)
        records_with_swapped_seed[0] = replace(
            records_with_swapped_seed[0],
            x_model_seed=records_with_swapped_seed[0].y_model_seed,
        )
        with self.assertRaisesRegex(ValueError, "exact seed namespace"):
            ClockExperimentResult(
                config=self.config,
                trials=tuple(records_with_swapped_seed),
                summaries=self.result.summaries,
                seeds=self.result.seeds,
            )

        summary = self.result.summaries[0]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            CellSummary(
                task=summary.task,
                n_x=summary.n_x,
                n_y=summary.n_y,
                successes=summary.successes,
                trials=summary.trials,
                success_probability=summary.success_probability,
                monte_carlo_standard_error=0.123,
                wilson_95_lower=summary.wilson_95_lower,
                wilson_95_upper=summary.wilson_95_upper,
            )


class AllocationStrategySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = allocation_summaries()

    def test_selects_equal_and_best_side_cells_in_canonical_order(self) -> None:
        points = select_allocation_strategies(self.summaries, (4, 2))

        self.assertEqual(
            [(point.task, point.total_budget, point.strategy) for point in points],
            [
                (task, total, strategy)
                for task in (NAVIGATION_TASK, SYNCHRONIZATION_TASK)
                for total in (2, 4)
                for strategy in ALLOCATION_STRATEGY_ORDER
            ],
        )
        selected_coordinates = {
            (point.task, point.total_budget, point.strategy): (
                point.n_x,
                point.n_y,
            )
            for point in points
        }
        self.assertEqual(
            selected_coordinates,
            {
                (NAVIGATION_TASK, 2, EQUAL_ALLOCATION_STRATEGY): (1, 1),
                (NAVIGATION_TASK, 2, BEST_BELOW_DIAGONAL_STRATEGY): (2, 0),
                (NAVIGATION_TASK, 2, BEST_ABOVE_DIAGONAL_STRATEGY): (0, 2),
                (NAVIGATION_TASK, 4, EQUAL_ALLOCATION_STRATEGY): (2, 2),
                (NAVIGATION_TASK, 4, BEST_BELOW_DIAGONAL_STRATEGY): (4, 0),
                (NAVIGATION_TASK, 4, BEST_ABOVE_DIAGONAL_STRATEGY): (1, 3),
                (SYNCHRONIZATION_TASK, 2, EQUAL_ALLOCATION_STRATEGY): (1, 1),
                (
                    SYNCHRONIZATION_TASK,
                    2,
                    BEST_BELOW_DIAGONAL_STRATEGY,
                ): (2, 0),
                (
                    SYNCHRONIZATION_TASK,
                    2,
                    BEST_ABOVE_DIAGONAL_STRATEGY,
                ): (0, 2),
                (
                    SYNCHRONIZATION_TASK,
                    4,
                    EQUAL_ALLOCATION_STRATEGY,
                ): (2, 2),
                (
                    SYNCHRONIZATION_TASK,
                    4,
                    BEST_BELOW_DIAGONAL_STRATEGY,
                ): (3, 1),
                (
                    SYNCHRONIZATION_TASK,
                    4,
                    BEST_ABOVE_DIAGONAL_STRATEGY,
                ): (0, 4),
            },
        )

        summaries_by_identity = {
            (summary.task, summary.n_x, summary.n_y): summary
            for summary in self.summaries
        }
        for point in points:
            source = summaries_by_identity[(point.task, point.n_x, point.n_y)]
            self.assertEqual(point.successes, source.successes)
            self.assertEqual(point.trials, source.trials)
            self.assertEqual(
                point.success_probability,
                source.success_probability,
            )
            self.assertEqual(
                point.monte_carlo_standard_error,
                source.monte_carlo_standard_error,
            )
            self.assertEqual(point.wilson_95_lower, source.wilson_95_lower)
            self.assertEqual(point.wilson_95_upper, source.wilson_95_upper)

    def test_selector_contract_is_exported_from_package_root(self) -> None:
        expected_exports = {
            "ALLOCATION_STRATEGY_ORDER": ALLOCATION_STRATEGY_ORDER,
            "AllocationStrategyPoint": AllocationStrategyPoint,
            "BEST_ABOVE_DIAGONAL_STRATEGY": BEST_ABOVE_DIAGONAL_STRATEGY,
            "BEST_BELOW_DIAGONAL_STRATEGY": BEST_BELOW_DIAGONAL_STRATEGY,
            "EQUAL_ALLOCATION_STRATEGY": EQUAL_ALLOCATION_STRATEGY,
            "select_allocation_strategies": select_allocation_strategies,
        }

        for name, expected in expected_exports.items():
            with self.subTest(name=name):
                self.assertIs(getattr(scripts, name), expected)
                self.assertIn(name, scripts.__all__)

    def test_ties_prefer_the_nearest_diagonal_and_input_order_is_irrelevant(
        self,
    ) -> None:
        expected = select_allocation_strategies(self.summaries, (2, 4))
        reversed_input = select_allocation_strategies(
            reversed(self.summaries),
            (value for value in (4, 2)),
        )

        self.assertEqual(reversed_input, expected)
        navigation_above = next(
            point
            for point in expected
            if point.task == NAVIGATION_TASK
            and point.total_budget == 4
            and point.strategy == BEST_ABOVE_DIAGONAL_STRATEGY
        )
        synchronization_below = next(
            point
            for point in expected
            if point.task == SYNCHRONIZATION_TASK
            and point.total_budget == 4
            and point.strategy == BEST_BELOW_DIAGONAL_STRATEGY
        )
        self.assertEqual((navigation_above.n_x, navigation_above.n_y), (1, 3))
        self.assertEqual(
            (synchronization_below.n_x, synchronization_below.n_y),
            (3, 1),
        )

    def test_rejects_incomplete_diagonals_for_either_task(self) -> None:
        cases = (
            (
                tuple(
                    summary
                    for summary in self.summaries
                    if not (
                        summary.task == NAVIGATION_TASK
                        and (summary.n_x, summary.n_y) == (1, 1)
                    )
                ),
                "exactly one equal",
            ),
            (
                tuple(
                    summary
                    for summary in self.summaries
                    if not (
                        summary.task == NAVIGATION_TASK
                        and summary.n_x > summary.n_y
                        and summary.n_x + summary.n_y == 2
                    )
                ),
                "below-diagonal",
            ),
            (
                tuple(
                    summary
                    for summary in self.summaries
                    if not (
                        summary.task == NAVIGATION_TASK
                        and summary.n_x < summary.n_y
                        and summary.n_x + summary.n_y == 2
                    )
                ),
                "above-diagonal",
            ),
            (
                tuple(
                    summary
                    for summary in self.summaries
                    if summary.task != SYNCHRONIZATION_TASK
                ),
                "synchronization.*equal",
            ),
        )
        for summaries, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    select_allocation_strategies(summaries, (2,))

    def test_rejects_invalid_totals_summary_types_and_duplicate_cells(self) -> None:
        invalid_totals = (
            ((0,), ValueError, "positive"),
            ((3,), ValueError, "even"),
            ((2, 2), ValueError, "duplicate"),
            ((True,), TypeError, "integer"),
            ((), ValueError, "at least one"),
            (2, TypeError, "iterable"),
        )
        for totals, error_type, message in invalid_totals:
            with self.subTest(totals=totals):
                with self.assertRaisesRegex(error_type, message):
                    select_allocation_strategies(self.summaries, totals)  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "iterable"):
            select_allocation_strategies(42, (2,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "only CellSummary"):
            select_allocation_strategies(
                self.summaries + (object(),),  # type: ignore[arg-type]
                (2,),
            )
        with self.assertRaisesRegex(ValueError, "Duplicate cell summary"):
            select_allocation_strategies(
                self.summaries + (self.summaries[0],),
                (2,),
            )

    def test_strategy_points_are_immutable_and_validate_geometry(self) -> None:
        point = select_allocation_strategies(self.summaries, (2,))[0]
        with self.assertRaises(FrozenInstanceError):
            point.n_x = 99  # type: ignore[misc]

        fields = {
            "task": point.task,
            "total_budget": point.total_budget,
            "strategy": point.strategy,
            "n_x": point.n_x,
            "n_y": point.n_y,
            "successes": point.successes,
            "trials": point.trials,
            "success_probability": point.success_probability,
            "monte_carlo_standard_error": point.monte_carlo_standard_error,
            "wilson_95_lower": point.wilson_95_lower,
            "wilson_95_upper": point.wilson_95_upper,
        }
        invalid = (
            ({"total_budget": 3}, "even"),
            ({"total_budget": 4}, "must equal"),
            ({"strategy": "unknown"}, "Unknown allocation strategy"),
            (
                {
                    "strategy": BEST_BELOW_DIAGONAL_STRATEGY,
                    "n_x": 1,
                    "n_y": 1,
                },
                "n_x > n_y",
            ),
            (
                {
                    "strategy": BEST_ABOVE_DIAGONAL_STRATEGY,
                    "n_x": 1,
                    "n_y": 1,
                },
                "n_x < n_y",
            ),
        )
        for changes, message in invalid:
            values = dict(fields)
            values.update(changes)
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    AllocationStrategyPoint(**values)  # type: ignore[arg-type]


class ClockExperimentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tiny_config(
            horizon=0,
            trials=1,
            x_updates=(0,),
            y_updates=(0,),
            fixed_total_budgets=(0,),
        )
        cls.result = run_clock_experiment(cls.config)

    def test_artifacts_round_trip_refuse_overwrite_and_detect_tampering(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_clock_experiment(
                self.result,
                root,
                run_id="unit-test-run",
            )

            self.assertEqual(
                json.loads(artifacts.config_path.read_text(encoding="utf-8")),
                self.config.to_dict(),
            )
            self.assertEqual(read_trial_csv(artifacts.trials_path), self.result.trials)
            self.assertEqual(
                read_summary_csv(artifacts.summary_path),
                self.result.summaries,
            )
            self.assertTrue(artifacts.figures_directory.is_dir())
            verify_run_manifest(artifacts.run_directory)

            with artifacts.seeds_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as stream:
                seed_rows = tuple(csv.DictReader(stream))
            self.assertEqual(len(seed_rows), len(self.result.seeds))
            self.assertTrue(
                all(row["schema_version"] == SEED_SCHEMA_VERSION for row in seed_rows)
            )

            before = {
                path.relative_to(artifacts.run_directory): path.read_bytes()
                for path in artifacts.run_directory.rglob("*")
                if path.is_file()
            }
            with self.assertRaises(FileExistsError):
                write_clock_experiment(
                    self.result,
                    root,
                    run_id="unit-test-run",
                )
            after = {
                path.relative_to(artifacts.run_directory): path.read_bytes()
                for path in artifacts.run_directory.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

            figure_path = artifacts.figures_directory / "two-clock-success-test.txt"
            figure_path.write_text("semantic figure placeholder\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                verify_run_manifest(artifacts.run_directory)
            write_run_manifest(artifacts.run_directory)
            verify_run_manifest(artifacts.run_directory)

            manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["schema_version"],
                MANIFEST_SCHEMA_VERSION,
            )
            self.assertEqual(
                manifest["artifact_schema_versions"],
                {
                    "config": CONFIG_SCHEMA_VERSION,
                    "trials": TRIAL_SCHEMA_VERSION,
                    "summary": SUMMARY_SCHEMA_VERSION,
                    "seeds": SEED_SCHEMA_VERSION,
                },
            )
            self.assertTrue(manifest["manifest_excludes_itself"])
            listed_paths = {entry["path"] for entry in manifest["files"]}
            self.assertNotIn("manifest.json", listed_paths)
            self.assertIn(
                "figures/two-clock-success-test.txt",
                listed_paths,
            )

            artifacts.summary_path.write_text(
                artifacts.summary_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "byte size mismatch|SHA-256 mismatch",
            ):
                verify_run_manifest(artifacts.run_directory)

    def test_manifests_cannot_bless_missing_required_artifacts(self) -> None:
        for required_relative in (
            "config.json",
            "trials.csv",
            "summary.csv",
            "seeds.csv",
        ):
            with self.subTest(required_relative=required_relative):
                with tempfile.TemporaryDirectory() as temporary:
                    artifacts = write_clock_experiment(
                        self.result,
                        temporary,
                        run_id="missing-required",
                    )
                    required_path = artifacts.run_directory / required_relative
                    required_path.unlink()

                    with self.assertRaisesRegex(
                        ValueError,
                        "missing required artifacts.*{}".format(
                            re.escape(required_relative)
                        ),
                    ):
                        write_run_manifest(artifacts.run_directory)

                    manifest = json.loads(
                        artifacts.manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["files"] = [
                        entry
                        for entry in manifest["files"]
                        if entry["path"] != required_relative
                    ]
                    artifacts.manifest_path.write_text(
                        json.dumps(manifest),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "missing required artifacts.*{}".format(
                            re.escape(required_relative)
                        ),
                    ):
                        verify_run_manifest(artifacts.run_directory)

    def test_readers_reject_extra_columns_and_inconsistent_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = write_clock_experiment(
                self.result,
                temporary,
                run_id="strict-csv",
            )
            trial_lines = artifacts.trials_path.read_text(encoding="utf-8").splitlines()
            extra_trials = Path(temporary) / "extra-trials.csv"
            extra_trials.write_text(
                "\n".join(
                    [trial_lines[0]] + [line + ",value" for line in trial_lines[1:]]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "extra columns"):
                read_trial_csv(extra_trials)

            with artifacts.summary_path.open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                summary_rows = list(reader)
            assert fieldnames is not None
            summary_rows[0]["success_probability"] = "0.123"
            corrupt_summary = Path(temporary) / "corrupt-summary.csv"
            with corrupt_summary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                read_summary_csv(corrupt_summary)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_manifests_reject_symlinked_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = write_clock_experiment(
                self.result,
                root,
                run_id="symlink-check",
            )
            external_directory = root / "external"
            external_directory.mkdir()
            link = artifacts.run_directory / "linked-directory"
            link.symlink_to(external_directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                write_run_manifest(artifacts.run_directory)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                verify_run_manifest(artifacts.run_directory)

            run_link = root / "linked-run"
            run_link.symlink_to(artifacts.run_directory, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                write_run_manifest(run_link)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                verify_run_manifest(run_link)

    def test_artifact_schemas_and_unsafe_run_identifiers_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "run_id"):
                write_clock_experiment(
                    self.result,
                    root,
                    run_id="../escape",
                )
            self.assertEqual(tuple(root.iterdir()), ())

            artifacts = write_clock_experiment(
                self.result,
                root,
                run_id="schema-check",
            )
            trial_header = artifacts.trials_path.read_text(
                encoding="utf-8"
            ).splitlines()[0]
            summary_header = artifacts.summary_path.read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertTrue(trial_header.startswith("schema_version,"))
            self.assertTrue(summary_header.startswith("schema_version,"))
            self.assertIn(TRIAL_SCHEMA_VERSION, artifacts.trials_path.read_text())
            self.assertIn(
                SUMMARY_SCHEMA_VERSION,
                artifacts.summary_path.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
