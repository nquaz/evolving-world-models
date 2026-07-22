"""Contract tests for the discrete, factored MDP primitives."""

from __future__ import annotations

import inspect
import json
import math
import random
import unittest
from abc import ABC

from scripts.mdp import (
    AbstractMDP,
    Assignment,
    CategoricalDistribution,
    FactoredMDP,
    ProductDistribution,
    TabularMDP,
    Variable,
)


def assignment(**values: object) -> Assignment:
    """Keep the fixtures concise while exercising the mapping constructor."""

    return Assignment(values)


class PrimitiveTests(unittest.TestCase):
    def test_abstract_mdp_is_actually_abstract(self) -> None:
        self.assertTrue(issubclass(AbstractMDP, ABC))
        self.assertTrue(inspect.isabstract(AbstractMDP))
        with self.assertRaises(TypeError):
            AbstractMDP((Variable("state", (0, 1)),))  # type: ignore[abstract]

    def test_variable_preserves_exact_domain(self) -> None:
        variable = Variable("weather", ("sun", "rain"))

        self.assertEqual(variable.name, "weather")
        self.assertEqual(variable.domain, ("sun", "rain"))

        with self.assertRaises(ValueError):
            Variable("empty", ())
        with self.assertRaises(ValueError):
            Variable("duplicate", (0, 0))

    def test_assignment_is_an_immutable_mapping(self) -> None:
        values = {"weather": "sun"}
        state = Assignment(values)
        values["weather"] = "rain"

        self.assertEqual(state["weather"], "sun")
        self.assertEqual(dict(state), {"weather": "sun"})
        with self.assertRaises(TypeError):
            state["weather"] = "rain"  # type: ignore[index]

    def test_categorical_distribution_normalizes_and_samples_seededly(self) -> None:
        weather = Variable("weather", ("sun", "rain"))
        sun = assignment(weather="sun")
        rain = assignment(weather="rain")
        distribution = CategoricalDistribution((weather,), {sun: 0.75, rain: 0.25})

        self.assertAlmostEqual(distribution.probability(sun), 0.75)
        self.assertAlmostEqual(distribution.probability(rain), 0.25)
        self.assertAlmostEqual(
            sum(probability for _, probability in distribution.items()), 1.0
        )

        first_rng = random.Random(2026)
        second_rng = random.Random(2026)
        first = [distribution.sample(first_rng) for _ in range(25)]
        second = [distribution.sample(second_rng) for _ in range(25)]
        self.assertEqual(first, second)
        self.assertTrue(set(first) <= {sun, rain})

        with self.assertRaises(ValueError):
            CategoricalDistribution((weather,), {sun: 0.4, rain: 0.4})
        with self.assertRaises(ValueError):
            CategoricalDistribution((weather,), {sun: 1.1, rain: -0.1})

        almost_normalized = CategoricalDistribution(
            (weather,), {sun: 0.75, rain: 0.2499999999}
        )
        self.assertAlmostEqual(
            sum(probability for _, probability in almost_normalized.items()), 1.0
        )
        with self.assertRaises(ValueError):
            CategoricalDistribution((weather,), {sun: 1.0000000005})

    def test_product_log_probability_does_not_underflow(self) -> None:
        factors = []
        outcome = {}
        for index in range(200):
            variable = Variable("x{}".format(index), (0, 1))
            factors.append(
                CategoricalDistribution(
                    (variable,),
                    (({variable.name: 0}, 0.99), ({variable.name: 1}, 0.01)),
                )
            )
            outcome[variable.name] = 1

        distribution = ProductDistribution(factors)
        self.assertEqual(distribution.probability(outcome), 0.0)
        self.assertAlmostEqual(
            distribution.log_probability(outcome), 200 * math.log(0.01)
        )


class TabularMDPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.position = Variable("position", (0, 1))
        self.action = Variable("action", ("stay", "flip"))

        rows = []
        for current in self.position.domain:
            for action in self.action.domain:
                probability_of_flip = 0.1 if action == "stay" else 0.8
                rows.append(
                    (
                        {"position": current},
                        {"action": action},
                        [
                            ({"position": current}, 1.0 - probability_of_flip),
                            ({"position": 1 - current}, probability_of_flip),
                        ],
                    )
                )

        self.mdp = TabularMDP(
            (self.position,), parent_variables=(self.action,), transitions=rows
        )

    def test_local_table_probability_and_distribution(self) -> None:
        current = assignment(position=0)
        parents = assignment(action="flip")
        next_position = assignment(position=1)

        self.assertAlmostEqual(
            self.mdp.transition_probability(next_position, current, parents), 0.8
        )
        distribution = self.mdp.transition_distribution(current, parents)
        self.assertAlmostEqual(distribution.probability(next_position), 0.8)
        self.assertAlmostEqual(
            sum(probability for _, probability in distribution.items()), 1.0
        )

    def test_string_is_inherited_deterministic_json(self) -> None:
        self.assertNotIn("__str__", TabularMDP.__dict__)

        description = json.loads(str(self.mdp))
        self.assertEqual(description["type"], "TabularMDP")
        self.assertEqual(
            description["variables"],
            [{"name": "position", "domain": [0, 1]}],
        )
        self.assertEqual(
            description["parent_variables"],
            [{"name": "action", "domain": ["stay", "flip"]}],
        )
        self.assertEqual(len(description["transitions"]), 4)
        first_row = description["transitions"][0]
        self.assertEqual(first_row["current"], {"position": 0})
        self.assertEqual(first_row["parents"], {"action": "stay"})
        self.assertEqual(
            first_row["outcomes"],
            [
                {"next_state": {"position": 0}, "probability": 0.9},
                {"next_state": {"position": 1}, "probability": 0.1},
            ],
        )
        self.assertEqual(str(self.mdp), str(self.mdp))

        # Implementation details and future caches are not representation fields.
        self.mdp._temporary_cache = {"should_not": "leak"}
        self.assertNotIn("temporary_cache", str(self.mdp))

    def test_string_canonicalizes_transition_input_order(self) -> None:
        reversed_rows = []
        for current in reversed(self.position.domain):
            for action in reversed(self.action.domain):
                probability_of_flip = 0.1 if action == "stay" else 0.8
                reversed_rows.append(
                    (
                        {"position": current},
                        {"action": action},
                        [
                            ({"position": 1 - current}, probability_of_flip),
                            ({"position": current}, 1.0 - probability_of_flip),
                        ],
                    )
                )
        equivalent = TabularMDP(
            (self.position,),
            parent_variables=(self.action,),
            transitions=reversed_rows,
        )

        self.assertEqual(str(self.mdp), str(equivalent))

    def test_string_tags_non_json_domain_values(self) -> None:
        structured = Variable(
            "structured",
            (("tuple", 1), frozenset(("b", "a")), b"\xff"),
        )
        rows = [
            (
                {"structured": value},
                {},
                (({"structured": value}, 1.0),),
            )
            for value in structured.domain
        ]
        description = json.loads(str(TabularMDP((structured,), transitions=rows)))

        self.assertEqual(
            description["variables"][0]["domain"],
            [
                {"type": "tuple", "items": ["tuple", 1]},
                {"type": "frozenset", "items": ["a", "b"]},
                {"type": "bytes", "hex": "ff"},
            ],
        )

    def test_local_sampling_is_seeded(self) -> None:
        current = assignment(position=0)
        parents = assignment(action="flip")
        first_rng = random.Random(11)
        second_rng = random.Random(11)

        first = [self.mdp.sample_next(current, parents, first_rng) for _ in range(30)]
        second = [self.mdp.sample_next(current, parents, second_rng) for _ in range(30)]

        self.assertEqual(first, second)
        self.assertTrue(set(first) <= {assignment(position=0), assignment(position=1)})

    def test_transition_queries_require_exact_keys_and_valid_domains(self) -> None:
        valid_current = assignment(position=0)
        valid_parents = assignment(action="stay")
        valid_next = assignment(position=0)

        invalid_calls = (
            lambda: self.mdp.transition_distribution(Assignment({}), valid_parents),
            lambda: self.mdp.transition_distribution(
                assignment(position=0, extra=True), valid_parents
            ),
            lambda: self.mdp.transition_distribution(
                assignment(position=9), valid_parents
            ),
            lambda: self.mdp.transition_distribution(valid_current, Assignment({})),
            lambda: self.mdp.transition_distribution(
                valid_current, assignment(action="unknown")
            ),
            lambda: self.mdp.transition_probability(
                Assignment({}), valid_current, valid_parents
            ),
            lambda: self.mdp.transition_probability(
                assignment(position=0, extra=True), valid_current, valid_parents
            ),
            lambda: self.mdp.transition_probability(
                assignment(position=9), valid_current, valid_parents
            ),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

        # The valid controls make failures above easier to diagnose.
        self.assertAlmostEqual(
            self.mdp.transition_probability(valid_next, valid_current, valid_parents),
            0.9,
        )

    def test_table_rejects_invalid_next_assignment_and_missing_context(self) -> None:
        state = Variable("state", (0, 1))
        action = Variable("action", ("go",))
        one_context = [({"state": 0}, {"action": "go"}, [({"state": 0}, 1.0)])]
        with self.assertRaises(ValueError):
            TabularMDP((state,), parent_variables=(action,), transitions=one_context)

        rows_with_bad_next = [
            ({"state": current}, {"action": "go"}, [({"state": 2}, 1.0)])
            for current in state.domain
        ]
        with self.assertRaises(ValueError):
            TabularMDP(
                (state,), parent_variables=(action,), transitions=rows_with_bad_next
            )


class FactoredMDPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.weather = Variable("weather", ("sun", "rain"))
        self.umbrella = Variable("umbrella", ("closed", "open"))
        self.season = Variable("season", ("dry", "wet"))
        self.action = Variable("action", ("open", "close"))

        weather_rows = []
        for weather in self.weather.domain:
            for season in self.season.domain:
                rain_probability = 0.2 if season == "dry" else 0.7
                weather_rows.append(
                    (
                        {"weather": weather},
                        {"season": season},
                        [
                            ({"weather": "sun"}, 1.0 - rain_probability),
                            ({"weather": "rain"}, rain_probability),
                        ],
                    )
                )

        umbrella_rows = []
        for umbrella in self.umbrella.domain:
            for weather in self.weather.domain:
                for season in self.season.domain:
                    for action in self.action.domain:
                        if action == "open":
                            open_probability = 0.9 if weather == "sun" else 0.6
                        elif umbrella == "open":
                            open_probability = 0.05 if weather == "sun" else 0.15
                        else:
                            open_probability = 0.0
                        umbrella_rows.append(
                            (
                                {"umbrella": umbrella},
                                {
                                    "weather": weather,
                                    "season": season,
                                    "action": action,
                                },
                                [
                                    (
                                        {"umbrella": "closed"},
                                        1.0 - open_probability,
                                    ),
                                    ({"umbrella": "open"}, open_probability),
                                ],
                            )
                        )

        self.weather_mdp = TabularMDP(
            (self.weather,),
            parent_variables=(self.season,),
            transitions=weather_rows,
        )
        # ``weather`` is another factor's current variable; ``season`` is a
        # parent shared by both factors; ``action`` remains an external parent.
        self.umbrella_mdp = TabularMDP(
            (self.umbrella,),
            parent_variables=(self.weather, self.season, self.action),
            transitions=umbrella_rows,
        )
        self.mdp = FactoredMDP((self.weather_mdp, self.umbrella_mdp))

    def test_factor_metadata_deduplicates_shared_and_internal_parents(self) -> None:
        self.assertEqual(self.mdp.variables, (self.weather, self.umbrella))
        self.assertEqual(self.mdp.parent_variables, (self.season, self.action))
        self.assertEqual(self.mdp.parents, (self.season, self.action))

    def test_string_recursively_describes_factors(self) -> None:
        self.assertNotIn("__str__", FactoredMDP.__dict__)

        description = json.loads(str(self.mdp))
        self.assertEqual(description["type"], "FactoredMDP")
        self.assertEqual(
            [variable["name"] for variable in description["variables"]],
            ["weather", "umbrella"],
        )
        self.assertEqual(
            [variable["name"] for variable in description["parent_variables"]],
            ["season", "action"],
        )
        self.assertEqual(
            [factor["type"] for factor in description["factors"]],
            ["TabularMDP", "TabularMDP"],
        )
        self.assertTrue(
            all("transitions" in factor for factor in description["factors"])
        )

    def test_joint_probability_is_product_with_cross_factor_projection(self) -> None:
        current = assignment(weather="sun", umbrella="closed")
        parents = assignment(season="dry", action="open")
        next_values = assignment(weather="rain", umbrella="open")

        # Weather contributes 0.2. Umbrella sees *current* weather=sun through
        # its parent projection and contributes 0.9.
        self.assertAlmostEqual(
            self.mdp.transition_probability(next_values, current, parents), 0.18
        )

        rainy_current = assignment(weather="rain", umbrella="closed")
        # Changing current weather changes umbrella's local term from .9 to .6.
        self.assertAlmostEqual(
            self.mdp.transition_probability(next_values, rainy_current, parents),
            0.12,
        )

    def test_open_and_close_change_umbrella_without_controlling_weather(self) -> None:
        current = assignment(weather="rain", umbrella="closed")
        open_distribution = self.mdp.transition_distribution(
            current,
            assignment(season="dry", action="open"),
        )
        close_distribution = self.mdp.transition_distribution(
            current,
            assignment(season="dry", action="close"),
        )

        def marginal(
            distribution: CategoricalDistribution,
            variable: str,
            value: object,
        ) -> float:
            return sum(
                probability
                for outcome, probability in distribution.items()
                if outcome[variable] == value
            )

        self.assertAlmostEqual(marginal(open_distribution, "umbrella", "open"), 0.6)
        self.assertAlmostEqual(marginal(close_distribution, "umbrella", "open"), 0.0)
        for weather in self.weather.domain:
            self.assertAlmostEqual(
                marginal(open_distribution, "weather", weather),
                marginal(close_distribution, "weather", weather),
            )

    def test_joint_distribution_has_exact_support_and_is_normalized(self) -> None:
        distribution = self.mdp.transition_distribution(
            assignment(weather="sun", umbrella="closed"),
            assignment(season="dry", action="open"),
        )

        expected_support = {
            assignment(weather=weather, umbrella=umbrella)
            for weather in self.weather.domain
            for umbrella in self.umbrella.domain
        }
        self.assertEqual(
            {outcome for outcome, _ in distribution.items()}, expected_support
        )
        self.assertAlmostEqual(
            sum(probability for _, probability in distribution.items()), 1.0
        )

    def test_joint_sampling_is_seeded_and_returns_full_assignments(self) -> None:
        current = assignment(weather="sun", umbrella="closed")
        parents = assignment(season="dry", action="open")
        first_rng = random.Random(31)
        second_rng = random.Random(31)

        first = [self.mdp.sample_next(current, parents, first_rng) for _ in range(30)]
        second = [self.mdp.sample_next(current, parents, second_rng) for _ in range(30)]

        self.assertEqual(first, second)
        for outcome in first:
            self.assertEqual(set(outcome), {"weather", "umbrella"})

    def test_factored_queries_enforce_exact_external_parent_keys(self) -> None:
        current = assignment(weather="sun", umbrella="closed")

        with self.assertRaises(ValueError):
            self.mdp.transition_distribution(current, assignment(season="dry"))
        with self.assertRaises(ValueError):
            self.mdp.transition_distribution(
                current,
                assignment(
                    season="dry",
                    action="open",
                    weather="sun",
                ),
            )

    def test_overlapping_factor_outputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FactoredMDP((self.weather_mdp, self.weather_mdp))

    def test_conflicting_shared_parent_domains_are_rejected(self) -> None:
        state = Variable("other", (0, 1))
        incompatible_season = Variable("season", ("spring", "fall"))
        rows = [
            (
                {"other": current},
                {"season": season},
                [({"other": 0}, 0.5), ({"other": 1}, 0.5)],
            )
            for current in state.domain
            for season in incompatible_season.domain
        ]
        conflicting = TabularMDP(
            (state,), parent_variables=(incompatible_season,), transitions=rows
        )

        with self.assertRaises(ValueError):
            FactoredMDP((self.weather_mdp, conflicting))


if __name__ == "__main__":
    unittest.main()
