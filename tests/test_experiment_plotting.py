"""Headless contract tests for two-clock experiment research figures."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "evolving-world-models-matplotlib"),
)

import matplotlib

# Plotting tests must never open an interactive window.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure

from scripts.clock_experiment import (
    BEST_ABOVE_DIAGONAL_STRATEGY,
    BEST_BELOW_DIAGONAL_STRATEGY,
    EQUAL_ALLOCATION_STRATEGY,
    NAVIGATION_TASK,
    SYNCHRONIZATION_TASK,
    AllocationStrategyPoint,
    wilson_interval,
)
from scripts.experiment_plotting import (
    plot_allocation_sensitivity,
    plot_success_heatmaps,
)


def mapping_summaries():
    """Return a deliberately unordered complete two-by-two task grid."""

    return (
        {"task": "synchronization", "n_x": 2, "n_y": 3, "mean_success": 0.8},
        {"task": "navigation", "n_x": 0, "n_y": 3, "mean_success": 0.3},
        {"task": "synchronization", "n_x": 0, "n_y": 0, "mean_success": 0.5},
        {"task": "navigation", "n_x": 2, "n_y": 0, "mean_success": 0.2},
        {"task": "navigation", "n_x": 0, "n_y": 0, "mean_success": 0.1},
        {"task": "synchronization", "n_x": 0, "n_y": 3, "mean_success": 0.7},
        {"task": "navigation", "n_x": 2, "n_y": 3, "mean_success": 0.4},
        {"task": "synchronization", "n_x": 2, "n_y": 0, "mean_success": 0.6},
    )


@dataclass(frozen=True)
class SummaryRecord:
    """Protocol-like summary used to exercise attribute access."""

    task: str
    n_x: int
    n_y: int
    success_probability: float


def attribute_summaries():
    """Return the smallest complete grid through object attributes."""

    return (
        SummaryRecord("target_navigation", 1, 4, 0.25),
        SummaryRecord("clock_synchronization", 1, 4, 0.75),
    )


def allocation_point(
    task: str,
    total_budget: int,
    strategy: str,
    n_x: int,
    n_y: int,
    successes: int,
    *,
    trials: int = 10,
) -> AllocationStrategyPoint:
    """Construct one internally consistent selected-allocation point."""

    probability = successes / trials
    lower, upper = wilson_interval(successes, trials)
    return AllocationStrategyPoint(
        task=task,
        total_budget=total_budget,
        strategy=strategy,
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


def allocation_points():
    """Return complete, deliberately unordered strategy curves at two budgets."""

    allocations = {
        2: {
            EQUAL_ALLOCATION_STRATEGY: (1, 1),
            BEST_BELOW_DIAGONAL_STRATEGY: (2, 0),
            BEST_ABOVE_DIAGONAL_STRATEGY: (0, 2),
        },
        4: {
            EQUAL_ALLOCATION_STRATEGY: (2, 2),
            BEST_BELOW_DIAGONAL_STRATEGY: (3, 1),
            BEST_ABOVE_DIAGONAL_STRATEGY: (1, 3),
        },
    }
    successes = {
        NAVIGATION_TASK: {
            EQUAL_ALLOCATION_STRATEGY: (5, 7),
            BEST_BELOW_DIAGONAL_STRATEGY: (8, 9),
            BEST_ABOVE_DIAGONAL_STRATEGY: (4, 6),
        },
        SYNCHRONIZATION_TASK: {
            EQUAL_ALLOCATION_STRATEGY: (4, 6),
            BEST_BELOW_DIAGONAL_STRATEGY: (5, 7),
            BEST_ABOVE_DIAGONAL_STRATEGY: (8, 9),
        },
    }
    points = []
    for task in (SYNCHRONIZATION_TASK, NAVIGATION_TASK):
        for budget_index, total_budget in enumerate((4, 2)):
            canonical_index = 1 - budget_index
            for strategy in (
                BEST_ABOVE_DIAGONAL_STRATEGY,
                EQUAL_ALLOCATION_STRATEGY,
                BEST_BELOW_DIAGONAL_STRATEGY,
            ):
                n_x, n_y = allocations[total_budget][strategy]
                points.append(
                    allocation_point(
                        task,
                        total_budget,
                        strategy,
                        n_x,
                        n_y,
                        successes[task][strategy][canonical_index],
                    )
                )
    return tuple(points)


class SuccessHeatmapTests(unittest.TestCase):
    def tearDown(self) -> None:
        """Close every figure even when an assertion fails."""

        plt.close("all")

    def test_default_heatmaps_have_exact_semantics_and_orientation(self) -> None:
        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(Figure, "savefig") as savefig,
        ):
            figure, axes = plot_success_heatmaps(mapping_summaries())

        show.assert_not_called()
        savefig.assert_not_called()
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(axes[0].get_title(), "Target navigation")
        self.assertEqual(axes[1].get_title(), "Clock synchronization")

        for axis in axes:
            self.assertEqual(
                axis.get_xlabel(),
                r"Controllable updates, $N_x$ (updates/context)",
            )
            self.assertEqual(
                axis.get_ylabel(),
                r"Predictable updates, $N_y$ (updates/context)",
            )
            self.assertEqual(
                [tick.get_text() for tick in axis.get_xticklabels()],
                ["0", "2"],
            )
            self.assertEqual(
                [tick.get_text() for tick in axis.get_yticklabels()],
                ["0", "3"],
            )
            self.assertEqual(axis.images[0].get_clim(), (0.0, 1.0))
            self.assertEqual(axis.images[0].get_interpolation(), "none")

        self.assertEqual(axes[0].images[0].get_cmap().name, "viridis")
        self.assertEqual(axes[1].images[0].get_cmap().name, "plasma")
        self.assertEqual(
            axes[0].images[0].get_array().tolist(),
            [[0.1, 0.2], [0.3, 0.4]],
        )
        self.assertEqual(
            axes[1].images[0].get_array().tolist(),
            [[0.5, 0.6], [0.7, 0.8]],
        )
        self.assertEqual(
            [axis.get_ylabel() for axis in figure.axes[2:]],
            [
                "Estimated success probability (unitless)",
                "Estimated success probability (unitless)",
            ],
        )

    def test_caller_axes_and_colormaps_are_preserved_and_overridable(self) -> None:
        figure, supplied = plt.subplots(1, 2)

        returned_figure, returned_axes = plot_success_heatmaps(
            attribute_summaries(),
            navigation_colormap="cividis",
            synchronization_colormap="magma",
            axes=supplied,
        )

        self.assertIs(returned_figure, figure)
        self.assertIs(returned_axes[0], supplied[0])
        self.assertIs(returned_axes[1], supplied[1])
        self.assertEqual(returned_axes[0].images[0].get_cmap().name, "cividis")
        self.assertEqual(returned_axes[1].images[0].get_cmap().name, "magma")
        self.assertEqual(returned_axes[0].images[0].get_array().tolist(), [[0.25]])
        self.assertEqual(returned_axes[1].images[0].get_array().tolist(), [[0.75]])

    def test_invalid_or_incomplete_summaries_are_rejected(self) -> None:
        valid = list(mapping_summaries())
        invalid_cases = (
            ((), ValueError, "at least one"),
            (
                valid + [valid[0]],
                ValueError,
                "Duplicate synchronization summary",
            ),
            (valid[:-1], ValueError, "complete shared grid"),
            (
                (
                    {
                        "task": "navigation",
                        "n_x": 0,
                        "n_y": 0,
                        "mean_success": 1.1,
                    },
                ),
                ValueError,
                r"\[0, 1\]",
            ),
            (
                (
                    {
                        "task": "forecasting",
                        "n_x": 0,
                        "n_y": 0,
                        "mean_success": 0.5,
                    },
                ),
                ValueError,
                "navigation or synchronization",
            ),
            (
                (
                    {
                        "task": "navigation",
                        "n_x": True,
                        "n_y": 0,
                        "mean_success": 0.5,
                    },
                ),
                TypeError,
                "nonnegative integer",
            ),
        )
        for summaries, error_type, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    plot_success_heatmaps(summaries)

    def test_invalid_colormaps_and_axes_are_rejected_before_plotting(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Matplotlib colormap"):
            plot_success_heatmaps(
                mapping_summaries(),
                navigation_colormap="not-a-colormap",
            )

        first_figure, first_axis = plt.subplots()
        second_figure, second_axis = plt.subplots()
        self.addCleanup(plt.close, first_figure)
        self.addCleanup(plt.close, second_figure)
        with self.assertRaisesRegex(ValueError, "same figure"):
            plot_success_heatmaps(
                mapping_summaries(),
                axes=(first_axis, second_axis),
            )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            plot_success_heatmaps(mapping_summaries(), axes=(first_axis,))


class AllocationSensitivityTests(unittest.TestCase):
    def tearDown(self) -> None:
        """Close every figure even when an assertion fails."""

        plt.close("all")

    def test_default_plot_has_ordered_curves_intervals_and_annotations(self) -> None:
        with (
            mock.patch.object(plt, "show") as show,
            mock.patch.object(Figure, "savefig") as savefig,
        ):
            figure, axes = plot_allocation_sensitivity(allocation_points())

        show.assert_not_called()
        savefig.assert_not_called()
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(axes[0].get_title(), "Target navigation")
        self.assertEqual(axes[1].get_title(), "Clock synchronization")

        expected_labels = [
            "Equal allocation",
            r"Best below diagonal ($N_x > N_y$)",
            r"Best above diagonal ($N_x < N_y$)",
        ]
        expected_navigation = ([0.5, 0.7], [0.8, 0.9], [0.4, 0.6])
        expected_synchronization = ([0.4, 0.6], [0.5, 0.7], [0.8, 0.9])
        expected_markers = ("o", "s", "^")
        expected_line_styles = ("-", "--", ":")
        for axis, expected_curves in zip(
            axes,
            (expected_navigation, expected_synchronization),
        ):
            self.assertEqual(
                axis.get_xlabel(),
                r"Total updates, $N$ (updates/context)",
            )
            self.assertEqual(
                axis.get_ylabel(),
                "Estimated success probability (unitless)",
            )
            self.assertEqual(axis.get_ylim(), (0.0, 1.0))
            self.assertEqual(
                [tick.get_text() for tick in axis.get_xticklabels()],
                ["2", "4"],
            )
            handles, labels = axis.get_legend_handles_labels()
            self.assertEqual(labels, expected_labels)
            self.assertTrue(
                all(isinstance(handle, ErrorbarContainer) for handle in handles)
            )
            self.assertEqual(len(axis.containers), 3)
            self.assertEqual(len(axis.collections), 3)
            for container, expected_y, expected_marker, expected_line_style in zip(
                axis.containers,
                expected_curves,
                expected_markers,
                expected_line_styles,
            ):
                self.assertIsInstance(container, ErrorbarContainer)
                central_line = container.lines[0]
                self.assertEqual(central_line.get_xdata().tolist(), [2, 4])
                self.assertEqual(central_line.get_ydata().tolist(), expected_y)
                self.assertEqual(central_line.get_marker(), expected_marker)
                self.assertEqual(
                    central_line.get_linestyle(),
                    expected_line_style,
                )
                self.assertTrue(container.has_yerr)
            self.assertEqual(
                [annotation.get_text() for annotation in axis.texts],
                [
                    "(1, 1)",
                    "(2, 2)",
                    "(2, 0)",
                    "(3, 1)",
                    "(0, 2)",
                    "(1, 3)",
                ],
            )
            self.assertTrue(any(line.get_visible() for line in axis.get_xgridlines()))
            self.assertTrue(any(line.get_visible() for line in axis.get_ygridlines()))

        self.assertEqual(
            [container.lines[0].get_color() for container in axes[0].containers],
            ["#0072B2", "#D55E00", "#009E73"],
        )

        # Check the actual asymmetric Wilson extent of the first navigation
        # error bar, rather than merely the presence of an errorbar artist.
        first_point = allocation_point(
            NAVIGATION_TASK,
            2,
            EQUAL_ALLOCATION_STRATEGY,
            1,
            1,
            5,
        )
        segments = axes[0].containers[0].lines[2][0].get_segments()
        self.assertAlmostEqual(segments[0][0][1], first_point.wilson_95_lower)
        self.assertAlmostEqual(segments[0][1][1], first_point.wilson_95_upper)

    def test_caller_axes_colors_and_annotation_control_are_respected(self) -> None:
        figure, supplied = plt.subplots(1, 2)

        returned_figure, returned_axes = plot_allocation_sensitivity(
            allocation_points(),
            axes=supplied,
            equal_color="black",
            below_color="#123456",
            above_color="tab:purple",
            annotate_allocations=False,
        )

        self.assertIs(returned_figure, figure)
        self.assertIs(returned_axes[0], supplied[0])
        self.assertIs(returned_axes[1], supplied[1])
        for axis in returned_axes:
            self.assertEqual(
                [container.lines[0].get_color() for container in axis.containers],
                ["black", "#123456", "tab:purple"],
            )
            self.assertEqual(len(axis.texts), 0)

    def test_incomplete_duplicate_and_wrong_record_support_are_rejected(self) -> None:
        valid = list(allocation_points())
        invalid_cases = (
            ((), ValueError, "at least one"),
            (valid[:-1], ValueError, "same task, total-budget, and strategy"),
            (
                valid + [valid[0]],
                ValueError,
                "Duplicate synchronization best_above_diagonal",
            ),
            (
                ({"task": NAVIGATION_TASK},),
                TypeError,
                "AllocationStrategyPoint objects",
            ),
        )
        for points, error_type, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    plot_allocation_sensitivity(points)

    def test_mutated_numeric_interval_and_geometry_fields_are_rejected(self) -> None:
        base = allocation_point(
            NAVIGATION_TASK,
            2,
            EQUAL_ALLOCATION_STRATEGY,
            1,
            1,
            5,
        )
        invalid_cases = (
            ("success_probability", float("nan"), ValueError, "finite"),
            ("trials", 0, ValueError, "at least 1"),
            ("wilson_95_lower", 0.45, ValueError, "inconsistent"),
            ("n_x", 2, ValueError, "n_x \\+ n_y"),
        )
        for field, value, error_type, message in invalid_cases:
            with self.subTest(field=field):
                point = allocation_point(
                    base.task,
                    base.total_budget,
                    base.strategy,
                    base.n_x,
                    base.n_y,
                    base.successes,
                )
                object.__setattr__(point, field, value)
                with self.assertRaisesRegex(error_type, message):
                    plot_allocation_sensitivity((point,))

    def test_invalid_colors_annotation_flag_and_axes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid Matplotlib color"):
            plot_allocation_sensitivity(
                allocation_points(),
                equal_color="not-a-color",
            )
        with self.assertRaisesRegex(TypeError, "color string"):
            plot_allocation_sensitivity(
                allocation_points(),
                below_color=object(),
            )
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            plot_allocation_sensitivity(
                allocation_points(),
                annotate_allocations=1,
            )

        first_figure, first_axis = plt.subplots()
        second_figure, second_axis = plt.subplots()
        self.addCleanup(plt.close, first_figure)
        self.addCleanup(plt.close, second_figure)
        with self.assertRaisesRegex(ValueError, "same figure"):
            plot_allocation_sensitivity(
                allocation_points(),
                axes=(first_axis, second_axis),
            )


if __name__ == "__main__":
    unittest.main()
