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

import scripts.visualization as visualization
from scripts.mdp import FactoredMDP, TabularMDP, Variable
from scripts.visualization import _factor_node_size_for_labels, _wrapped_factor_label


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


class WrappedFactorLabelTests(unittest.TestCase):
    def test_wraps_kernel_labels_without_splitting_probability_prefix(self) -> None:
        self.assertEqual(
            _wrapped_factor_label("P(lock′ | lock)"),
            "P(lock′|·)",
        )
        self.assertEqual(
            _wrapped_factor_label("P(door′ | door, lock, action)"),
            "P(door′|·)",
        )

    def test_non_kernel_labels_are_left_unchanged(self) -> None:
        self.assertEqual(_wrapped_factor_label("custom"), "custom")

    def test_blank_and_whitespace_label_extents_are_zero(self) -> None:
        for label in ("", " ", "   ", "\t"):
            with self.subTest(label=label):
                self.assertEqual(
                    visualization._label_extent_points(label, 10.0),
                    (0.0, 0.0),
                )

    def test_multiline_extent_accounts_for_every_line(self) -> None:
        first = visualization._label_extent_points("Door", 10.0)
        second = visualization._label_extent_points("transition", 10.0)
        combined = visualization._label_extent_points(
            "Door\ntransition",
            10.0,
        )

        self.assertGreaterEqual(combined[0], max(first[0], second[0]))
        self.assertGreaterEqual(combined[1], first[1] + second[1])

    @mock.patch.object(visualization, "_label_extent_points")
    def test_factor_node_size_uses_widest_label_dimension(self, measure) -> None:
        measure.side_effect = [
            (20.0, 120.0),
            (100.0, 10.0),
        ]

        size = _factor_node_size_for_labels(
            {
                ("factor", (0,)): "tall",
                ("factor", (1,)): "wide",
            },
            font_size=10.0,
            base_node_size=2300.0,
        )

        expected = (120.0 * visualization._FACTOR_LABEL_PADDING) ** 2
        self.assertAlmostEqual(size, expected)
        self.assertEqual(
            measure.call_args_list,
            [mock.call("tall", 10.0), mock.call("wide", 10.0)],
        )

    @mock.patch.object(
        visualization,
        "_label_extent_points",
        return_value=(0.0, 0.0),
    )
    def test_blank_factor_labels_preserve_base_node_size(self, measure) -> None:
        self.assertEqual(
            _factor_node_size_for_labels(
                {("factor", (0,)): ""},
                font_size=9.0,
                base_node_size=2300.0,
            ),
            2300.0,
        )
        measure.assert_called_once_with("", 9.0)

    def test_marker_edge_margins_follow_circle_square_and_diamond_boundaries(
        self,
    ) -> None:
        size = 3600.0
        half_side = 30.0

        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "o", (1.0, 0.0)),
            half_side,
        )
        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "s", (1.0, 0.0)),
            half_side,
        )
        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "s", (1.0, 1.0)),
            half_side * 2.0**0.5,
        )
        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "D", (1.0, 0.0)),
            half_side * 2.0**0.5,
        )
        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "D", (1.0, 1.0)),
            half_side,
        )
        self.assertAlmostEqual(
            visualization._marker_edge_margin(size, "X", (1.0, 1.0)),
            half_side * 2.0**0.5,
        )

    def test_per_edge_style_selection_cycles_and_preserves_rgb_colors(self) -> None:
        self.assertEqual(
            visualization._style_value_for_edge((1.0, 2.0), 3, "width"),
            2.0,
        )
        color = (0.1, 0.2, 0.3)
        self.assertEqual(
            visualization._style_value_for_edge(color, 2, "edge_color"),
            color,
        )
        with self.assertRaisesRegex(ValueError, "width sequence must not be empty"):
            visualization._style_value_for_edge([], 0, "width")


class VisualizationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_state = Variable("lock", ("locked", "unlocked"))
        self.door_state = Variable("door", ("closed", "open"))
        self.action = Variable("action", ("open", "close"))

        self.lock_mdp = deterministic_kernel(
            self.lock_state,
        )
        self.door_mdp = deterministic_kernel(
            self.door_state,
            (self.lock_state, self.action),
        )
        self.mdp = FactoredMDP((self.lock_mdp, self.door_mdp))


class NetworkXStructureTests(VisualizationFixture):
    def expected_variable_nodes(self):
        return {
            ("current", "lock"): {
                "kind": "variable",
                "role": "current",
                "layer": 0,
                "order": 0,
                "variable": "lock",
                "domain": ("locked", "unlocked"),
                "time": "t",
                "label": "lock (t)",
            },
            ("next", "lock"): {
                "kind": "variable",
                "role": "next",
                "layer": 2,
                "order": 0,
                "variable": "lock",
                "domain": ("locked", "unlocked"),
                "time": "t+1",
                "label": "lock (t+1)",
            },
            ("current", "door"): {
                "kind": "variable",
                "role": "current",
                "layer": 0,
                "order": 1,
                "variable": "door",
                "domain": ("closed", "open"),
                "time": "t",
                "label": "door (t)",
            },
            ("next", "door"): {
                "kind": "variable",
                "role": "next",
                "layer": 2,
                "order": 1,
                "variable": "door",
                "domain": ("closed", "open"),
                "time": "t+1",
                "label": "door (t+1)",
            },
            ("parent", "action"): {
                "kind": "variable",
                "role": "parent",
                "layer": 0,
                "order": 2,
                "variable": "action",
                "domain": ("open", "close"),
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
                    "variables": ("lock",),
                    "parents": (),
                    "label": "P(lock′ | lock)",
                },
                ("factor", (1,)): {
                    "kind": "factor",
                    "role": "factor",
                    "layer": 1,
                    "order": 1,
                    "factor_index": (1,),
                    "factor_path": (1,),
                    "factor_type": "TabularMDP",
                    "variables": ("door",),
                    "parents": ("lock", "action"),
                    "label": "P(door′ | door, lock, action)",
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
            (("current", "lock"), ("factor", (0,))): condition((0,), "current_state"),
            (("factor", (0,)), ("next", "lock")): predicts((0,)),
            (("current", "door"), ("factor", (1,))): condition((1,), "current_state"),
            (("current", "lock"), ("factor", (1,))): condition((1,), "internal_parent"),
            (("parent", "action"), ("factor", (1,))): condition(
                (1,), "external_parent"
            ),
            (("factor", (1,)), ("next", "door")): predicts((1,)),
        }
        actual_edges = {
            (source, target): attributes
            for source, target, attributes in graph.edges(data=True)
        }
        self.assertEqual(actual_edges, expected_edges)

        # The cross-factor lock setting is read synchronously from x_t. It must
        # not be represented as an external parent or as the next-time lock.
        self.assertNotIn(("parent", "lock"), graph)
        self.assertIn(
            (("current", "lock"), ("factor", (1,))),
            graph.edges,
        )
        self.assertNotIn(
            (("next", "lock"), ("factor", (1,))),
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
            (("current", "lock"), ("next", "lock")): dependency((0,), "current_state"),
            (("current", "door"), ("next", "door")): dependency((1,), "current_state"),
            (("current", "lock"), ("next", "door")): dependency(
                (1,), "internal_parent"
            ),
            (("parent", "action"), ("next", "door")): dependency(
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
            ("lock",),
        )
        self.assertEqual(
            factor_nodes[("factor", (0, 1))]["variables"],
            ("door",),
        )
        self.assertEqual(
            factor_nodes[("factor", (0, 0))]["factor_type"],
            "TabularMDP",
        )
        self.assertNotIn(("factor", (0,)), graph)
        self.assertIn(
            (("current", "lock"), ("factor", (0, 1))),
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
                    "lock": "Lock setting",
                    ("next", "lock"): "Next lock setting",
                },
            )
            rendered_labels = {text.get_text() for text in axes.texts}

            self.assertIn(
                "Lock setting\n{'locked', 'unlocked'}",
                rendered_labels,
            )
            self.assertIn(
                "Next lock setting\n{'locked', 'unlocked'}",
                rendered_labels,
            )
            self.assertIn("door (t)\n{'closed', 'open'}", rendered_labels)
            self.assertIn("P(lock′|·)", rendered_labels)
        finally:
            plt.close(figure)

    def test_layered_layout_spans_the_densest_layer(self) -> None:
        graph = self.mdp.to_networkx()
        positions = visualization._layered_layout(graph)

        self.assertEqual(positions[("current", "lock")], (0.0, 1.0))
        self.assertEqual(positions[("current", "door")], (0.0, 0.0))
        self.assertEqual(positions[("parent", "action")], (0.0, -1.0))
        self.assertEqual(positions[("factor", (0,))], (1.0, 1.0))
        self.assertEqual(positions[("factor", (1,))], (1.0, -1.0))
        self.assertEqual(positions[("next", "lock")], (2.0, 1.0))
        self.assertEqual(positions[("next", "door")], (2.0, -1.0))

    def test_default_factor_markers_do_not_overlap_at_render_size(self) -> None:
        figure, axes = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        try:
            self.mdp.draw(ax=axes)
            figure.canvas.draw()

            factor_collection = max(
                axes.collections,
                key=lambda collection: float(collection.get_sizes()[0]),
            )
            factor_size = float(factor_collection.get_sizes()[0])
            centers = axes.transData.transform(factor_collection.get_offsets())
            center_separation = abs(float(centers[0, 1] - centers[1, 1]))
            marker_side = factor_size**0.5 * figure.dpi / 72.0

            self.assertGreaterEqual(center_separation + 1.0, marker_side)
        finally:
            plt.close(figure)

    def test_edge_drawing_preserves_finalized_axes_limits(self) -> None:
        import networkx as nx

        edge_limits = []
        draw_edges = nx.draw_networkx_edges
        figure, axes = plt.subplots()

        def capture_limits(*args, **kwargs):
            before = (axes.get_xlim(), axes.get_ylim())
            result = draw_edges(*args, **kwargs)
            after = (axes.get_xlim(), axes.get_ylim())
            edge_limits.append((before, after))
            return result

        try:
            with mock.patch(
                "networkx.draw_networkx_edges",
                side_effect=capture_limits,
            ):
                self.mdp.draw(ax=axes)

            self.assertTrue(edge_limits)
            finalized_limits = edge_limits[0][0]
            for before, after in edge_limits:
                self.assertEqual(before, finalized_limits)
                self.assertEqual(after, finalized_limits)
            self.assertEqual((axes.get_xlim(), axes.get_ylim()), finalized_limits)
        finally:
            plt.close(figure)

    def test_custom_factor_labels_support_blank_whitespace_and_multiline(
        self,
    ) -> None:
        factor_node = ("factor", (1,))

        for label in ("", "   ", "Door\ntransition"):
            with self.subTest(label=label):
                figure, axes = plt.subplots()
                try:
                    returned = self.mdp.draw(
                        ax=axes,
                        labels={factor_node: label},
                    )

                    self.assertIs(returned, axes)
                    self.assertIn(label, [text.get_text() for text in axes.texts])
                finally:
                    plt.close(figure)

    def test_omitted_node_size_grows_only_factors_and_clips_by_role(self) -> None:
        factor_size = 6400.0
        margin_calls = []
        edge_colors = ("#112233", "#445566")
        edge_widths = (1.0, 2.0)
        arrow_sizes = (12, 18)

        def marker_margin(node_size, node_shape, direction):
            margin_calls.append((node_size, node_shape, direction))
            return float(len(margin_calls))

        figure, axes = plt.subplots()
        try:
            with (
                mock.patch.object(
                    visualization,
                    "_factor_node_size_for_labels",
                    return_value=factor_size,
                ) as compute_size,
                mock.patch.object(
                    visualization,
                    "_marker_edge_margin",
                    side_effect=marker_margin,
                ),
                mock.patch("networkx.draw_networkx_nodes") as draw_nodes,
                mock.patch("networkx.draw_networkx_edges") as draw_edges,
                mock.patch("networkx.draw_networkx_labels"),
            ):
                self.mdp.draw(
                    ax=axes,
                    edge_color=edge_colors,
                    width=edge_widths,
                    arrowsize=arrow_sizes,
                )

            sizes_by_role = {
                call.kwargs["nodelist"][0][0]: call.kwargs["node_size"]
                for call in draw_nodes.call_args_list
            }
            self.assertEqual(
                sizes_by_role,
                {
                    "current": 2300.0,
                    "parent": 2300.0,
                    "factor": factor_size,
                    "next": 2300.0,
                },
            )

            factor_labels, font_size, base_size = compute_size.call_args.args
            self.assertEqual(factor_labels[("factor", (1,))], "P(door′|·)")
            self.assertEqual(font_size, 9.0)
            self.assertEqual(base_size, 2300.0)

            graph = self.mdp.to_networkx()
            observed_edges = set()
            shape_by_role = {
                "current": "o",
                "parent": "D",
                "factor": "s",
                "next": "o",
            }
            style_indices = {"solid": 0, "dashed": 0}
            positions = visualization._layered_layout(graph)
            for index, call in enumerate(draw_edges.call_args_list):
                self.assertEqual(len(call.kwargs["edgelist"]), 1)
                source, target = call.kwargs["edgelist"][0]
                observed_edges.add((source, target))
                self.assertEqual(call.kwargs["nodelist"], [source, target])
                self.assertEqual(call.kwargs["node_size"], 0.0)
                source_size, source_shape, source_direction = margin_calls[2 * index]
                target_size, target_shape, target_direction = margin_calls[
                    2 * index + 1
                ]
                source_role = graph.nodes[source]["role"]
                target_role = graph.nodes[target]["role"]
                source_x, source_y = axes.transData.transform(positions[source])
                target_x, target_y = axes.transData.transform(positions[target])
                self.assertAlmostEqual(source_direction[0], target_x - source_x)
                self.assertAlmostEqual(source_direction[1], target_y - source_y)
                self.assertEqual(
                    source_size,
                    factor_size if source_role == "factor" else 2300.0,
                )
                self.assertEqual(
                    target_size,
                    factor_size if target_role == "factor" else 2300.0,
                )
                self.assertEqual(source_shape, shape_by_role[source_role])
                self.assertEqual(target_shape, shape_by_role[target_role])
                self.assertEqual(
                    target_direction,
                    (-source_direction[0], -source_direction[1]),
                )
                self.assertEqual(
                    call.kwargs["min_source_margin"],
                    float(2 * index + 1),
                )
                self.assertEqual(
                    call.kwargs["min_target_margin"],
                    float(2 * index + 2),
                )
                line_style = call.kwargs["style"]
                style_index = style_indices[line_style]
                self.assertEqual(
                    call.kwargs["edge_color"],
                    edge_colors[style_index % len(edge_colors)],
                )
                self.assertEqual(
                    call.kwargs["width"],
                    edge_widths[style_index % len(edge_widths)],
                )
                self.assertEqual(
                    call.kwargs["arrowsize"],
                    arrow_sizes[style_index % len(arrow_sizes)],
                )
                style_indices[line_style] += 1
            self.assertEqual(observed_edges, set(graph.edges))
        finally:
            plt.close(figure)

    def test_explicit_node_size_is_uniform_and_disables_factor_growth(self) -> None:
        requested_size = 3100.0
        figure, axes = plt.subplots()
        try:
            with (
                mock.patch.object(
                    visualization,
                    "_factor_node_size_for_labels",
                ) as compute_size,
                mock.patch("networkx.draw_networkx_nodes") as draw_nodes,
                mock.patch("networkx.draw_networkx_edges"),
                mock.patch("networkx.draw_networkx_labels"),
            ):
                self.mdp.draw(ax=axes, node_size=requested_size)

            compute_size.assert_not_called()
            self.assertEqual(
                {call.kwargs["node_size"] for call in draw_nodes.call_args_list},
                {requested_size},
            )
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

    def test_size_options_must_be_finite_nonnegative_scalars(self) -> None:
        for option, value, error_type in (
            ("node_size", [2300.0], TypeError),
            ("node_size", -1.0, ValueError),
            ("node_size", float("inf"), ValueError),
            ("node_size", float("nan"), ValueError),
            ("font_size", [9.0], TypeError),
            ("font_size", -1.0, ValueError),
        ):
            with self.subTest(option=option, value=value):
                with self.assertRaisesRegex(error_type, "finite nonnegative scalar"):
                    self.mdp.draw(**{option: value})


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
        self.assertIn("P(lock′ | lock)", source)
        self.assertIn("P(door′ | door, lock, action)", source)
        self.assertIn('fillcolor="#D9D9D9"', source)
        self.assertIn("rank=same", source)


if __name__ == "__main__":
    unittest.main()
