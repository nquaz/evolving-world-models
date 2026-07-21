"""Contract tests for transition-structure visualization adapters."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from itertools import product
from pathlib import Path
from typing import Sequence
from unittest import mock

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "evolving-world-models-matplotlib"),
)

import matplotlib

# Drawing tests must remain headless in CI and must not open an interactive UI.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scripts.mdp import FactoredMDP, TabularMDP, Variable


def deterministic_kernel(
    variable: Variable,
    parents: Sequence[Variable] = (),
) -> TabularMDP:
    """Build a small, valid table whose values do not matter to drawing."""

    parent_tuple = tuple(parents)
    rows = []
    assert variable.domain is not None
    for values in product(
        variable.domain,
        *(parent.domain for parent in parent_tuple),
    ):
        current_value, *parent_values = values
        rows.append(
            (
                {variable.name: current_value},
                {
                    parent.name: value
                    for parent, value in zip(parent_tuple, parent_values)
                },
                (({variable.name: current_value}, 1.0),),
            )
        )
    return TabularMDP(
        (variable,),
        parent_variables=parent_tuple,
        transitions=rows,
    )


class VisualizationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.weather = Variable("weather", ("sun", "rain"))
        self.pantry = Variable("pantry", ("empty", "full"))
        self.season = Variable("season", ("dry", "wet"))
        self.action = Variable("action", ("wait", "shop"))

        self.weather_mdp = deterministic_kernel(
            self.weather,
            (self.season,),
        )
        self.pantry_mdp = deterministic_kernel(
            self.pantry,
            (self.weather, self.season, self.action),
        )
        self.mdp = FactoredMDP((self.weather_mdp, self.pantry_mdp))


class NetworkXStructureTests(VisualizationFixture):
    def expected_variable_nodes(self):
        return {
            ("current", "weather"): {
                "kind": "variable",
                "role": "current",
                "layer": 0,
                "order": 0,
                "variable": "weather",
                "domain": ("sun", "rain"),
                "time": "t",
                "label": "weather (t)",
            },
            ("next", "weather"): {
                "kind": "variable",
                "role": "next",
                "layer": 2,
                "order": 0,
                "variable": "weather",
                "domain": ("sun", "rain"),
                "time": "t+1",
                "label": "weather (t+1)",
            },
            ("current", "pantry"): {
                "kind": "variable",
                "role": "current",
                "layer": 0,
                "order": 1,
                "variable": "pantry",
                "domain": ("empty", "full"),
                "time": "t",
                "label": "pantry (t)",
            },
            ("next", "pantry"): {
                "kind": "variable",
                "role": "next",
                "layer": 2,
                "order": 1,
                "variable": "pantry",
                "domain": ("empty", "full"),
                "time": "t+1",
                "label": "pantry (t+1)",
            },
            ("parent", "season"): {
                "kind": "variable",
                "role": "parent",
                "layer": 0,
                "order": 2,
                "variable": "season",
                "domain": ("dry", "wet"),
                "time": "t",
                "label": "season (t)",
            },
            ("parent", "action"): {
                "kind": "variable",
                "role": "parent",
                "layer": 0,
                "order": 3,
                "variable": "action",
                "domain": ("wait", "shop"),
                "time": "t",
                "label": "action (t)",
            },
        }

    def test_factor_view_has_exact_nodes_attributes_and_edges(self) -> None:
        graph = self.mdp.to_networkx(view="factor")

        expected_nodes = self.expected_variable_nodes()
        expected_nodes.update(
            {
                ("factor", (0,)): {
                    "kind": "factor",
                    "role": "factor",
                    "layer": 1,
                    "order": 0,
                    "factor_index": (0,),
                    "factor_path": (0,),
                    "factor_type": "TabularMDP",
                    "variables": ("weather",),
                    "parents": ("season",),
                    "label": "P(weather′ | weather, season)",
                },
                ("factor", (1,)): {
                    "kind": "factor",
                    "role": "factor",
                    "layer": 1,
                    "order": 1,
                    "factor_index": (1,),
                    "factor_path": (1,),
                    "factor_type": "TabularMDP",
                    "variables": ("pantry",),
                    "parents": ("weather", "season", "action"),
                    "label": "P(pantry′ | pantry, weather, season, action)",
                },
            }
        )
        self.assertEqual(dict(graph.nodes(data=True)), expected_nodes)
        self.assertEqual(
            dict(graph.graph),
            {
                "model_type": "FactoredMDP",
                "view": "factor",
                "semantics": "two_slice_transition_model",
            },
        )

        def condition(index, kind):
            return {
                "kind": kind,
                "factor_index": index,
                "factor_path": index,
            }

        def predicts(index):
            return {
                "kind": "prediction",
                "factor_index": index,
                "factor_path": index,
            }

        expected_edges = {
            (("current", "weather"), ("factor", (0,))): condition(
                (0,), "current_state"
            ),
            (("parent", "season"), ("factor", (0,))): condition(
                (0,), "external_parent"
            ),
            (("factor", (0,)), ("next", "weather")): predicts((0,)),
            (("current", "pantry"), ("factor", (1,))): condition((1,), "current_state"),
            (("current", "weather"), ("factor", (1,))): condition(
                (1,), "internal_parent"
            ),
            (("parent", "season"), ("factor", (1,))): condition(
                (1,), "external_parent"
            ),
            (("parent", "action"), ("factor", (1,))): condition(
                (1,), "external_parent"
            ),
            (("factor", (1,)), ("next", "pantry")): predicts((1,)),
        }
        actual_edges = {
            (source, target): attributes
            for source, target, attributes in graph.edges(data=True)
        }
        self.assertEqual(actual_edges, expected_edges)

        # Cross-factor weather is read synchronously from x_t. It must not be
        # represented as an external parent or as next-time weather.
        self.assertNotIn(("parent", "weather"), graph)
        self.assertIn(
            (("current", "weather"), ("factor", (1,))),
            graph.edges,
        )
        self.assertNotIn(
            (("next", "weather"), ("factor", (1,))),
            graph.edges,
        )

    def test_dependencies_view_encodes_possible_scope_dependencies(self) -> None:
        graph = self.mdp.to_networkx(view="dependencies")

        self.assertEqual(dict(graph.nodes(data=True)), self.expected_variable_nodes())
        self.assertEqual(graph.graph["view"], "dependencies")

        def dependency(index, input_kind):
            return {
                "kind": "possible_dependency",
                "input_kind": input_kind,
                "factor_index": index,
                "factor_path": index,
            }

        expected_edges = {
            (("current", "weather"), ("next", "weather")): dependency(
                (0,), "current_state"
            ),
            (("parent", "season"), ("next", "weather")): dependency(
                (0,), "external_parent"
            ),
            (("current", "pantry"), ("next", "pantry")): dependency(
                (1,), "current_state"
            ),
            (("current", "weather"), ("next", "pantry")): dependency(
                (1,), "internal_parent"
            ),
            (("parent", "season"), ("next", "pantry")): dependency(
                (1,), "external_parent"
            ),
            (("parent", "action"), ("next", "pantry")): dependency(
                (1,), "external_parent"
            ),
        }
        self.assertEqual(
            {
                (source, target): attributes
                for source, target, attributes in graph.edges(data=True)
            },
            expected_edges,
        )
        self.assertFalse(
            any(
                attributes["kind"] == "factor"
                for _, attributes in graph.nodes(data=True)
            )
        )

    def test_repeated_conversion_has_deterministic_structure_and_order(self) -> None:
        first = self.mdp.to_networkx()
        second = self.mdp.to_networkx()

        self.assertIsNot(first, second)
        self.assertEqual(first.graph, second.graph)
        self.assertEqual(list(first.nodes(data=True)), list(second.nodes(data=True)))
        self.assertEqual(list(first.edges(data=True)), list(second.edges(data=True)))

    def test_nested_factored_models_are_flattened_to_stable_leaf_paths(self) -> None:
        nested = FactoredMDP((self.mdp,))
        graph = nested.to_networkx()

        factor_nodes = {
            node: attributes
            for node, attributes in graph.nodes(data=True)
            if attributes["kind"] == "factor"
        }
        self.assertEqual(set(factor_nodes), {("factor", (0, 0)), ("factor", (0, 1))})
        self.assertEqual(
            factor_nodes[("factor", (0, 0))]["variables"],
            ("weather",),
        )
        self.assertEqual(
            factor_nodes[("factor", (0, 1))]["variables"],
            ("pantry",),
        )
        self.assertEqual(
            factor_nodes[("factor", (0, 0))]["factor_type"],
            "TabularMDP",
        )
        self.assertNotIn(("factor", (0,)), graph)
        self.assertIn(
            (("current", "weather"), ("factor", (0, 1))),
            graph.edges,
        )


class DrawingTests(VisualizationFixture):
    def test_draw_returns_supplied_axes_and_never_calls_show(self) -> None:
        figure, axes = plt.subplots()
        try:
            with mock.patch("matplotlib.pyplot.show") as show:
                returned = self.mdp.draw(ax=axes)

            self.assertIs(returned, axes)
            show.assert_not_called()
            self.assertFalse(axes.axison)
        finally:
            plt.close(figure)

    def test_draw_supports_domain_annotations_and_label_overrides(self) -> None:
        figure, axes = plt.subplots()
        try:
            self.mdp.draw(
                ax=axes,
                show_domains=True,
                labels={
                    "weather": "Weather",
                    ("next", "weather"): "Tomorrow's weather",
                },
            )
            rendered_labels = {text.get_text() for text in axes.texts}

            self.assertIn("Weather\n{'sun', 'rain'}", rendered_labels)
            self.assertIn("Tomorrow's weather\n{'sun', 'rain'}", rendered_labels)
            self.assertIn("pantry (t)\n{'empty', 'full'}", rendered_labels)
            self.assertIn("P(weather′\n| ·)", rendered_labels)
        finally:
            plt.close(figure)

    def test_complete_custom_layout_is_accepted(self) -> None:
        graph = self.mdp.to_networkx()
        positions = {
            node: (float(index), float(-index)) for index, node in enumerate(graph)
        }
        figure, axes = plt.subplots()
        try:
            self.assertIs(self.mdp.draw(ax=axes, layout=positions), axes)
        finally:
            plt.close(figure)


class ValidationTests(VisualizationFixture):
    def test_invalid_view_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "view must be one of"):
            self.mdp.to_networkx(view="direct")

    def test_invalid_layout_is_rejected(self) -> None:
        figure, axes = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "layout must be"):
                self.mdp.draw(ax=axes, layout="spring")
        finally:
            plt.close(figure)

    def test_incomplete_custom_layout_is_rejected(self) -> None:
        graph = self.mdp.to_networkx()
        positions = {node: (0.0, 0.0) for node in graph}
        positions.pop(next(iter(positions)))
        figure, axes = plt.subplots()
        try:
            with self.assertRaisesRegex(ValueError, "missing positions"):
                self.mdp.draw(ax=axes, layout=positions)
        finally:
            plt.close(figure)

    def test_invalid_rank_direction_is_rejected_before_graphviz_import(self) -> None:
        with self.assertRaisesRegex(ValueError, "rankdir must be one of"):
            self.mdp.to_graphviz(rankdir="north")

    def test_unknown_style_and_invalid_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown draw style"):
            self.mdp.draw(sparkles=True)

        figure, axes = plt.subplots()
        try:
            with self.assertRaisesRegex(TypeError, "labels must be a mapping"):
                self.mdp.draw(ax=axes, labels=["not", "a", "mapping"])
            with self.assertRaisesRegex(ValueError, "unknown node or variable"):
                self.mdp.draw(ax=axes, labels={"typo": "Typo"})
        finally:
            plt.close(figure)


class GraphvizTests(VisualizationFixture):
    def test_graphviz_adapter_or_actionable_missing_dependency_error(self) -> None:
        if importlib.util.find_spec("graphviz") is None:
            with self.assertRaisesRegex(
                ImportError,
                r"graphviz.*evolving-world-models\[graphviz\]",
            ):
                self.mdp.to_graphviz()
            return

        dot = self.mdp.to_graphviz(rankdir="lr")
        source = dot.source
        self.assertIn("rankdir=LR", source)
        self.assertIn("P(weather′ | weather, season)", source)
        self.assertIn("P(pantry′ | pantry, weather, season, action)", source)
        self.assertIn('fillcolor="#D9D9D9"', source)
        self.assertIn("rank=same", source)


if __name__ == "__main__":
    unittest.main()
