"""Publication-oriented figures for the two-clock allocation experiment.

This optional presentation adapter turns aggregated two-clock experiment
summaries into the experiment's primary pair of success-probability heatmaps.
It deliberately owns no transition, belief, planning, aggregation, artifact,
or statistical semantics.  Callers should load the versioned ``summary.csv``
through the experiment orchestration layer and pass the resulting records to
:func:`plot_success_heatmaps`.

The companion :func:`plot_allocation_sensitivity` visualizes how three
fixed-total allocation strategies change with total per-context update budget
``N = N_x + N_y``.  It consumes validated
:class:`~scripts.clock_experiment.AllocationStrategyPoint` records selected by
the orchestration layer: equal allocation, the best allocation below the
heatmap diagonal (``N_x > N_y``), and the best allocation above it
(``N_x < N_y``).  Navigation and synchronization appear in separate panels.
Wilson 95% intervals are rendered as asymmetric error bars, and optional
``(N_x, N_y)`` annotations make the selected allocation at each point
auditable. Strategy identity is encoded redundantly: equal allocation uses a
solid line with circles, best below-diagonal uses a dashed line with squares,
and best above-diagonal uses a dotted line with triangles.

Each input record must expose ``task``, ``n_x``, ``n_y``, and
``success_probability`` either as mapping keys or as object attributes.
``task`` accepts ``"navigation"`` and ``"synchronization"`` plus the explicit
aliases documented by :func:`plot_success_heatmaps`.  The two tasks must cover
the same complete rectangular grid exactly once.  Update counts are
nonnegative integers, and success probabilities are finite values in
``[0, 1]``.

The function creates exactly two data panels: target navigation uses
``"viridis"`` by default and clock synchronization uses ``"plasma"``.  Both
colormaps are caller-controlled presentation parameters.  Each panel has its
own colorbar, while both share a fixed probability scale of ``[0, 1]``.
Discrete cells are rendered without interpolation, with controllable updates
on the x-axis and predictable updates on the y-axis.

Matplotlib is imported normally at module import time.  This module is not
imported from :mod:`scripts`, so the standard-library-only transition,
belief, planning, and experiment layers remain usable without visualization
dependencies.  Plotting creates in-memory Matplotlib artists only: it never
calls ``show()``, saves a file, or mutates process-global style settings.
Callers own the returned figure and are responsible for displaying, saving,
and closing it.

For ``R`` records and an ``N_x`` by ``N_y`` grid, heatmap validation takes
``O(R + N_x N_y)`` time and the plotted matrices use ``O(N_x N_y)`` space.
Sensitivity validation, ordering, and rendering take ``O(R log R)`` time and
``O(R)`` auxiliary space.  Rendering time and memory are otherwise determined
by Matplotlib.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure

from .clock_experiment import (
    ALLOCATION_STRATEGY_ORDER,
    BEST_ABOVE_DIAGONAL_STRATEGY,
    BEST_BELOW_DIAGONAL_STRATEGY,
    EQUAL_ALLOCATION_STRATEGY,
    AllocationStrategyPoint,
)

_NAVIGATION = "navigation"
_SYNCHRONIZATION = "synchronization"
_TASK_ALIASES = {
    "navigation": _NAVIGATION,
    "target": _NAVIGATION,
    "target_navigation": _NAVIGATION,
    "synchronization": _SYNCHRONIZATION,
    "sync": _SYNCHRONIZATION,
    "clock_synchronization": _SYNCHRONIZATION,
}
_N_X_FIELDS = (
    "n_x",
    "controllable_updates",
    "controllable_updates_per_context",
)
_N_Y_FIELDS = (
    "n_y",
    "predictable_updates",
    "predictable_updates_per_context",
)
_SUCCESS_FIELDS = (
    "success_probability",
    "mean_success",
    "success_mean",
    "success_rate",
    "mean",
)
_MISSING = object()


def _field(record: object, names: Sequence[str]) -> object:
    """Read the first named field from a mapping or protocol-like object."""

    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
    else:
        for name in names:
            value = getattr(record, name, _MISSING)
            if value is not _MISSING:
                return value
    raise ValueError(
        "Each summary record must expose one of the fields {}.".format(tuple(names))
    )


def _task_name(value: object) -> str:
    """Canonicalize one documented task label without guessing other names."""

    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING:
        value = enum_value
    if not isinstance(value, str):
        raise TypeError("Summary task must be a string or string-valued enum.")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _TASK_ALIASES[normalized]
    except KeyError as error:
        raise ValueError(
            "Summary task must identify navigation or synchronization; "
            "received {!r}.".format(value)
        ) from error


def _update_count(value: object, *, field: str) -> int:
    """Validate a nonnegative per-context observation count."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("{} must be a nonnegative integer.".format(field))
    count = int(value)
    if count < 0:
        raise ValueError("{} must be nonnegative.".format(field))
    return count


def _probability(value: object) -> float:
    """Validate one finite unitless success probability."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("success_probability must be a real number.")
    try:
        probability = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "success_probability must be finite and lie in [0, 1]."
        ) from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("success_probability must be finite and lie in [0, 1].")
    return probability


def _colormap(name: object, *, parameter: str) -> str:
    """Validate a named Matplotlib colormap before creating any artists."""

    if not isinstance(name, str):
        raise TypeError("{} must be a Matplotlib colormap name.".format(parameter))
    if not name:
        raise ValueError("{} must be a Matplotlib colormap name.".format(parameter))
    try:
        colormaps[name]
    except KeyError as error:
        raise ValueError(
            "{} names an unknown Matplotlib colormap: {!r}.".format(parameter, name)
        ) from error
    return name


def _line_color(value: object, *, parameter: str) -> str:
    """Validate a caller-controlled Matplotlib line color."""

    if not isinstance(value, str):
        raise TypeError("{} must be a Matplotlib color string.".format(parameter))
    if not value or not is_color_like(value):
        raise ValueError(
            "{} must be a valid Matplotlib color string.".format(parameter)
        )
    return value


def _allocation_sensitivity_tables(
    points: Sequence[AllocationStrategyPoint],
) -> Tuple[
    Tuple[int, ...],
    Dict[str, Dict[str, Dict[int, AllocationStrategyPoint]]],
]:
    """Validate complete strategy support and return deterministic task tables."""

    try:
        point_records = tuple(points)
    except TypeError as error:
        raise TypeError(
            "points must be a finite sequence of AllocationStrategyPoint objects."
        ) from error
    if not point_records:
        raise ValueError("points must contain at least one allocation strategy point.")

    tables: Dict[str, Dict[str, Dict[int, AllocationStrategyPoint]]] = {
        task: {strategy: {} for strategy in ALLOCATION_STRATEGY_ORDER}
        for task in (_NAVIGATION, _SYNCHRONIZATION)
    }
    for record_number, point in enumerate(point_records, start=1):
        if not isinstance(point, AllocationStrategyPoint):
            raise TypeError(
                "points must contain AllocationStrategyPoint objects; "
                "record {} has type {}.".format(
                    record_number,
                    type(point).__name__,
                )
            )

        # Reconstruct the immutable value to revalidate every numerical and
        # statistical invariant at the presentation boundary.  This protects
        # caller-owned axes if an object was populated through an untrusted
        # deserialization path or deliberately mutated around frozen-dataclass
        # safeguards.
        point = AllocationStrategyPoint(
            task=point.task,
            total_budget=point.total_budget,
            strategy=point.strategy,
            n_x=point.n_x,
            n_y=point.n_y,
            successes=point.successes,
            trials=point.trials,
            success_probability=point.success_probability,
            monte_carlo_standard_error=point.monte_carlo_standard_error,
            wilson_95_lower=point.wilson_95_lower,
            wilson_95_upper=point.wilson_95_upper,
        )
        task = _task_name(point.task)
        total_budget = _update_count(point.total_budget, field="total_budget")
        n_x = _update_count(point.n_x, field="n_x")
        n_y = _update_count(point.n_y, field="n_y")
        if n_x + n_y != total_budget:
            raise ValueError(
                "Allocation strategy point must satisfy n_x + n_y = total_budget."
            )
        if point.strategy not in ALLOCATION_STRATEGY_ORDER:
            raise ValueError(
                "Allocation strategy must be one of {}; received {!r}.".format(
                    ALLOCATION_STRATEGY_ORDER,
                    point.strategy,
                )
            )
        if point.strategy == EQUAL_ALLOCATION_STRATEGY and n_x != n_y:
            raise ValueError("Equal allocation requires n_x == n_y.")
        if point.strategy == BEST_BELOW_DIAGONAL_STRATEGY and n_x <= n_y:
            raise ValueError("Best below-diagonal allocation requires n_x > n_y.")
        if point.strategy == BEST_ABOVE_DIAGONAL_STRATEGY and n_x >= n_y:
            raise ValueError("Best above-diagonal allocation requires n_x < n_y.")

        probability = _probability(point.success_probability)
        lower = _probability_interval_endpoint(
            point.wilson_95_lower,
            field="wilson_95_lower",
        )
        upper = _probability_interval_endpoint(
            point.wilson_95_upper,
            field="wilson_95_upper",
        )
        if lower > probability or probability > upper:
            raise ValueError(
                "Wilson interval must satisfy lower <= success_probability <= upper."
            )

        strategy_table = tables[task][point.strategy]
        if total_budget in strategy_table:
            raise ValueError(
                "Duplicate {} {} point for total_budget={} at record {}.".format(
                    task,
                    point.strategy,
                    total_budget,
                    record_number,
                )
            )
        strategy_table[total_budget] = point

    totals = tuple(
        sorted(
            {
                total
                for task_tables in tables.values()
                for strategy_table in task_tables.values()
                for total in strategy_table
            }
        )
    )
    expected_totals = set(totals)
    for task in (_NAVIGATION, _SYNCHRONIZATION):
        for strategy in ALLOCATION_STRATEGY_ORDER:
            observed_totals = set(tables[task][strategy])
            if observed_totals != expected_totals:
                missing = sorted(expected_totals.difference(observed_totals))
                extra = sorted(observed_totals.difference(expected_totals))
                raise ValueError(
                    "Allocation sensitivity points must cover the same task, "
                    "total-budget, and strategy support exactly once; {} {} "
                    "has missing totals {} and extra totals {}.".format(
                        task,
                        strategy,
                        missing,
                        extra,
                    )
                )
    return totals, tables


def _probability_interval_endpoint(value: object, *, field: str) -> float:
    """Validate one finite probability interval endpoint."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a real number.".format(field))
    try:
        endpoint = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "{} must be finite and lie in [0, 1].".format(field)
        ) from error
    if not math.isfinite(endpoint) or not 0.0 <= endpoint <= 1.0:
        raise ValueError("{} must be finite and lie in [0, 1].".format(field))
    return endpoint


def _summary_tables(
    summaries: Sequence[object],
) -> Tuple[
    Tuple[int, ...],
    Tuple[int, ...],
    Dict[str, Dict[Tuple[int, int], float]],
]:
    """Validate records and return task tables in deterministic grid order."""

    if isinstance(summaries, (str, bytes)) or not summaries:
        raise ValueError("summaries must contain at least one summary record.")

    tables: Dict[str, Dict[Tuple[int, int], float]] = {
        _NAVIGATION: {},
        _SYNCHRONIZATION: {},
    }
    for record_number, record in enumerate(summaries, start=1):
        task = _task_name(_field(record, ("task",)))
        n_x = _update_count(
            _field(record, _N_X_FIELDS),
            field="n_x",
        )
        n_y = _update_count(
            _field(record, _N_Y_FIELDS),
            field="n_y",
        )
        probability = _probability(_field(record, _SUCCESS_FIELDS))
        coordinate = (n_x, n_y)
        if coordinate in tables[task]:
            raise ValueError(
                "Duplicate {} summary for n_x={}, n_y={} at record {}.".format(
                    task, n_x, n_y, record_number
                )
            )
        tables[task][coordinate] = probability

    coordinates = set(tables[_NAVIGATION]).union(tables[_SYNCHRONIZATION])
    n_x_values = tuple(sorted({coordinate[0] for coordinate in coordinates}))
    n_y_values = tuple(sorted({coordinate[1] for coordinate in coordinates}))
    expected = {(n_x, n_y) for n_y in n_y_values for n_x in n_x_values}
    for task in (_NAVIGATION, _SYNCHRONIZATION):
        missing = expected.difference(tables[task])
        if missing:
            example_n_x, example_n_y = min(missing)
            raise ValueError(
                "{} summaries must cover the complete shared grid exactly once; "
                "missing {} cell(s), including n_x={}, n_y={}.".format(
                    task,
                    len(missing),
                    example_n_x,
                    example_n_y,
                )
            )
    return n_x_values, n_y_values, tables


def _plot_axes(
    axes: Optional[Sequence[Axes]],
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Create two axes or validate two caller-owned axes on one figure."""

    if axes is None:
        figure, created = plt.subplots(
            1,
            2,
            figsize=(10.0, 4.2),
            constrained_layout=True,
        )
        return figure, (created[0], created[1])

    try:
        axis_tuple = tuple(axes)
    except TypeError as error:
        raise TypeError(
            "axes must be a sequence containing exactly two Axes."
        ) from error
    if len(axis_tuple) != 2 or any(not isinstance(axis, Axes) for axis in axis_tuple):
        raise ValueError("axes must contain exactly two Matplotlib Axes.")
    first, second = axis_tuple
    if first.figure is not second.figure:
        raise ValueError("Both supplied axes must belong to the same figure.")
    return first.figure, (first, second)


def plot_success_heatmaps(
    summaries: Sequence[object],
    *,
    navigation_colormap: str = "viridis",
    synchronization_colormap: str = "plasma",
    axes: Optional[Sequence[Axes]] = None,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Plot target-navigation and clock-synchronization success heatmaps.

    Args:
        summaries: Complete rectangular summaries for both tasks.  Every
            mapping or protocol-like object must expose ``task``, ``n_x``,
            ``n_y``, and ``success_probability``.  For compatibility with the
            durable summary schema, ``controllable_updates``,
            ``predictable_updates``, and ``mean_success`` are also accepted,
            as are their explicit ``*_per_context`` aliases.  Task labels may
            be ``navigation``/``target_navigation`` or
            ``synchronization``/``clock_synchronization``.
        navigation_colormap: Matplotlib colormap name for target navigation.
        synchronization_colormap: Matplotlib colormap name for synchronization.
        axes: Optional two caller-owned Matplotlib axes on the same figure.
            When omitted, a new one-row, two-column figure is created.

    Returns:
        The owning figure and the two data axes in navigation,
        synchronization order.  Each call also adds one colorbar axis per
        panel to the owning figure.

    Raises:
        TypeError: If a field, colormap, or axes container has the wrong type.
        ValueError: If records are empty, duplicated, incomplete, invalid, or
            do not share one rectangular grid; if a colormap is unknown; or
            if supplied axes are invalid or belong to different figures.

    Notes:
        Controllable updates ``n_x`` are columns and predictable updates
        ``n_y`` are rows, both ordered from smallest to largest.  The function
        creates artists only and never displays or saves the figure.
    """

    n_x_values, n_y_values, tables = _summary_tables(summaries)
    navigation_colormap = _colormap(
        navigation_colormap,
        parameter="navigation_colormap",
    )
    synchronization_colormap = _colormap(
        synchronization_colormap,
        parameter="synchronization_colormap",
    )
    figure, data_axes = _plot_axes(axes)

    task_specs = (
        (
            _NAVIGATION,
            "Target navigation",
            navigation_colormap,
            data_axes[0],
        ),
        (
            _SYNCHRONIZATION,
            "Clock synchronization",
            synchronization_colormap,
            data_axes[1],
        ),
    )
    for task, title, colormap, axis in task_specs:
        values = [
            [tables[task][(n_x, n_y)] for n_x in n_x_values] for n_y in n_y_values
        ]
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="none",
            cmap=colormap,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xlabel(r"Controllable updates, $N_x$ (updates/context)")
        axis.set_ylabel(r"Predictable updates, $N_y$ (updates/context)")
        axis.set_xticks(range(len(n_x_values)), labels=n_x_values)
        axis.set_yticks(range(len(n_y_values)), labels=n_y_values)
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("Estimated success probability (unitless)")

    return figure, data_axes


def plot_allocation_sensitivity(
    points: Sequence[AllocationStrategyPoint],
    *,
    axes: Optional[Sequence[Axes]] = None,
    equal_color: str = "#0072B2",
    below_color: str = "#D55E00",
    above_color: str = "#009E73",
    annotate_allocations: bool = True,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Plot task performance versus total budget for three allocations.

    Args:
        points: Complete :class:`AllocationStrategyPoint` records for both
            tasks, every total budget, and all three strategies: equal,
            best below the heatmap diagonal (``N_x > N_y``), and best above it
            (``N_x < N_y``).  Every task-strategy pair must contain exactly the
            same total-budget support.
        axes: Optional two caller-owned Matplotlib axes on the same figure.
            When omitted, a new one-row, two-column figure is created.
        equal_color: Matplotlib color string for equal allocation.
        below_color: Matplotlib color string for best below-diagonal allocation.
        above_color: Matplotlib color string for best above-diagonal allocation.
        annotate_allocations: Whether to annotate every point with its selected
            ``(N_x, N_y)`` allocation.

    Returns:
        The owning figure and the navigation and synchronization axes, in that
        order.

    Raises:
        TypeError: If records, numeric fields, colors, axes, or
            ``annotate_allocations`` have the wrong type.
        ValueError: If records are empty, invalid, duplicated, incomplete, or
            have inconsistent task/total/strategy support; if interval or
            allocation relations are invalid; if a color is unknown; or if
            supplied axes are invalid or belong to different figures.

    Notes:
        Error bars are the records' asymmetric Wilson 95% intervals around the
        estimated binary success probability.  Color, marker, line style, and
        legend text redundantly identify each strategy.  Strategy lines and
        legend entries always appear in equal, below-diagonal, above-diagonal
        order.  This function creates artists only and never displays or saves
        the figure.
    """

    totals, tables = _allocation_sensitivity_tables(points)
    equal_color = _line_color(equal_color, parameter="equal_color")
    below_color = _line_color(below_color, parameter="below_color")
    above_color = _line_color(above_color, parameter="above_color")
    if not isinstance(annotate_allocations, bool):
        raise TypeError("annotate_allocations must be a boolean.")
    figure, data_axes = _plot_axes(axes)

    strategy_specs = (
        (EQUAL_ALLOCATION_STRATEGY, "Equal allocation", equal_color, "o", "-"),
        (
            BEST_BELOW_DIAGONAL_STRATEGY,
            r"Best below diagonal ($N_x > N_y$)",
            below_color,
            "s",
            "--",
        ),
        (
            BEST_ABOVE_DIAGONAL_STRATEGY,
            r"Best above diagonal ($N_x < N_y$)",
            above_color,
            "^",
            ":",
        ),
    )
    task_specs = (
        (_NAVIGATION, "Target navigation", data_axes[0]),
        (_SYNCHRONIZATION, "Clock synchronization", data_axes[1]),
    )
    annotation_offsets = {
        EQUAL_ALLOCATION_STRATEGY: (-18, 8),
        BEST_BELOW_DIAGONAL_STRATEGY: (0, -8),
        BEST_ABOVE_DIAGONAL_STRATEGY: (18, 8),
    }

    for task, title, axis in task_specs:
        for strategy, label, color, marker, line_style in strategy_specs:
            selected = [tables[task][strategy][total] for total in totals]
            probabilities = [point.success_probability for point in selected]
            lower_errors = [
                point.success_probability - point.wilson_95_lower for point in selected
            ]
            upper_errors = [
                point.wilson_95_upper - point.success_probability for point in selected
            ]
            axis.errorbar(
                totals,
                probabilities,
                yerr=(lower_errors, upper_errors),
                label=label,
                color=color,
                marker=marker,
                linestyle=line_style,
                markersize=4.5,
                linewidth=1.6,
                capsize=3.0,
            )
            if annotate_allocations:
                for total, point in zip(totals, selected):
                    horizontal_offset, vertical_offset = annotation_offsets[strategy]
                    vertical_alignment = "bottom" if vertical_offset > 0 else "top"
                    if point.success_probability >= 0.9 and vertical_offset > 0:
                        vertical_offset = -8
                        vertical_alignment = "top"
                    elif point.success_probability <= 0.1 and vertical_offset < 0:
                        vertical_offset = 8
                        vertical_alignment = "bottom"
                    axis.annotate(
                        "({}, {})".format(point.n_x, point.n_y),
                        xy=(total, point.success_probability),
                        xytext=(horizontal_offset, vertical_offset),
                        textcoords="offset points",
                        ha="center",
                        va=vertical_alignment,
                        fontsize="x-small",
                        color=color,
                    )

        axis.set_title(title)
        axis.set_xlabel(r"Total updates, $N$ (updates/context)")
        axis.set_ylabel("Estimated success probability (unitless)")
        axis.set_xticks(totals)
        axis.set_ylim(0.0, 1.0)
        axis.set_axisbelow(True)
        axis.grid(True, axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.legend()

    return figure, data_axes


__all__ = ["plot_allocation_sensitivity", "plot_success_heatmaps"]
