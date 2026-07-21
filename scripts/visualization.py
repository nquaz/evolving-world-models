"""Optional visualization adapters for MDP transition structure.

The transition-model core remains dependency-free.  Functions in this module
import NetworkX, Matplotlib, and Graphviz only when their corresponding public
methods are called.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from .mdp import AbstractFactoredMDP, AbstractMDP, Variable

_VALID_VIEWS = ("factor", "dependencies")
_VALID_RANK_DIRECTIONS = ("TB", "BT", "LR", "RL")

_ROLE_STYLES = {
    "current": {"color": "#9ECAE1", "shape": "o"},
    "parent": {"color": "#FDD0A2", "shape": "D"},
    "factor": {"color": "#D9D9D9", "shape": "s"},
    "next": {"color": "#A1D99B", "shape": "o"},
}

_DRAW_DEFAULTS = {
    "node_size": 2300,
    "font_size": 9,
    "font_color": "#111111",
    "edge_color": "#555555",
    "width": 1.5,
    "arrowsize": 18,
    "alpha": 1.0,
    "linewidths": 1.0,
    "current_color": _ROLE_STYLES["current"]["color"],
    "parent_color": _ROLE_STYLES["parent"]["color"],
    "factor_color": _ROLE_STYLES["factor"]["color"],
    "next_color": _ROLE_STYLES["next"]["color"],
    "current_shape": _ROLE_STYLES["current"]["shape"],
    "parent_shape": _ROLE_STYLES["parent"]["shape"],
    "factor_shape": _ROLE_STYLES["factor"]["shape"],
    "next_shape": _ROLE_STYLES["next"]["shape"],
}


def _require_module(module_name: str, extra: str, purpose: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            "{} requires the optional {!r} dependency. Install it with "
            "`pip install 'evolving-world-models[{}]'`.".format(
                purpose, module_name, extra
            )
        ) from error


def _validate_view(view: str) -> str:
    if view not in _VALID_VIEWS:
        raise ValueError(
            "view must be one of {}; received {!r}.".format(_VALID_VIEWS, view)
        )
    return view


def _iter_leaf_factors(
    model: AbstractMDP,
    path: Tuple[int, ...] = (),
    active_ids: Optional[Set[int]] = None,
) -> Iterator[Tuple[Tuple[int, ...], AbstractMDP]]:
    """Yield leaf kernels with stable paths through nested factorizations."""

    if active_ids is None:
        active_ids = set()
    model_id = id(model)
    if model_id in active_ids:
        raise ValueError("Cannot draw a cyclic factored MDP composition.")

    if isinstance(model, AbstractFactoredMDP):
        active_ids.add(model_id)
        try:
            for index, factor in enumerate(model.factors):
                yield from _iter_leaf_factors(factor, path + (index,), active_ids)
        finally:
            active_ids.remove(model_id)
        return

    yield (path or (0,)), model


def _variable_attributes(
    variable: Variable,
    role: str,
    layer: int,
    order: int,
) -> Dict[str, Any]:
    time = "t+1" if role == "next" else "t"
    return {
        "kind": "variable",
        "role": role,
        "layer": layer,
        "order": order,
        "variable": variable.name,
        "domain": variable.domain,
        "time": time,
        "label": "{} ({})".format(variable.name, time),
    }


def _factor_label(factor: AbstractMDP) -> str:
    outputs = ", ".join("{}′".format(variable.name) for variable in factor.variables)
    conditions = [variable.name for variable in factor.variables]
    conditions.extend(variable.name for variable in factor.parent_variables)
    return "P({} | {})".format(outputs, ", ".join(conditions))


def to_networkx(model: AbstractMDP, view: str = "factor") -> Any:
    """Build a directed two-slice graph for an MDP.

    ``view='factor'`` preserves transition-factor nodes.  The
    ``'dependencies'`` view connects every input in a factor's declared scope
    directly to every output and therefore represents *possible*, not
    necessarily minimal, dependencies.
    """

    view = _validate_view(view)
    nx = _require_module("networkx", "viz", "to_networkx()")
    graph = nx.DiGraph(
        model_type=type(model).__name__,
        view=view,
        semantics="two_slice_transition_model",
    )

    tracked_by_name = {variable.name: variable for variable in model.variables}
    for order, variable in enumerate(model.variables):
        graph.add_node(
            ("current", variable.name),
            **_variable_attributes(variable, "current", 0, order),
        )
        graph.add_node(
            ("next", variable.name),
            **_variable_attributes(variable, "next", 2, order),
        )

    parent_offset = len(model.variables)
    for order, variable in enumerate(model.parent_variables):
        graph.add_node(
            ("parent", variable.name),
            **_variable_attributes(variable, "parent", 0, parent_offset + order),
        )

    leaf_factors = tuple(_iter_leaf_factors(model))
    leaf_output_names = [
        variable.name for _, factor in leaf_factors for variable in factor.variables
    ]
    expected_output_names = [variable.name for variable in model.variables]
    if len(leaf_output_names) != len(set(leaf_output_names)) or set(
        leaf_output_names
    ) != set(expected_output_names):
        raise ValueError(
            "Leaf factor outputs must partition the composite MDP variables; "
            "received {!r} for {!r}.".format(leaf_output_names, expected_output_names)
        )

    for factor_order, (factor_path, factor) in enumerate(leaf_factors):
        output_names = tuple(variable.name for variable in factor.variables)
        parent_names = tuple(variable.name for variable in factor.parent_variables)
        factor_node = ("factor", factor_path)
        if view == "factor":
            graph.add_node(
                factor_node,
                kind="factor",
                role="factor",
                layer=1,
                order=factor_order,
                factor_index=factor_path,
                factor_path=factor_path,
                factor_type=type(factor).__name__,
                variables=output_names,
                parents=parent_names,
                label=_factor_label(factor),
            )

        inputs = factor.variables + factor.parent_variables
        own_variable_names = set(output_names)
        for variable in inputs:
            input_role = "current" if variable.name in tracked_by_name else "parent"
            input_node = (input_role, variable.name)
            if input_node not in graph:
                raise ValueError(
                    "Factor {!r} references parent {!r}, which is not present in "
                    "the composite MDP scope.".format(factor_path, variable.name)
                )
            if variable.name in own_variable_names:
                input_kind = "current_state"
            elif variable.name in tracked_by_name:
                input_kind = "internal_parent"
            else:
                input_kind = "external_parent"

            if view == "factor":
                graph.add_edge(
                    input_node,
                    factor_node,
                    kind=input_kind,
                    factor_index=factor_path,
                    factor_path=factor_path,
                )
            else:
                for output in factor.variables:
                    output_node = ("next", output.name)
                    graph.add_edge(
                        input_node,
                        output_node,
                        kind="possible_dependency",
                        input_kind=input_kind,
                        factor_index=factor_path,
                        factor_path=factor_path,
                    )

        if view == "factor":
            for output in factor.variables:
                graph.add_edge(
                    factor_node,
                    ("next", output.name),
                    kind="prediction",
                    factor_index=factor_path,
                    factor_path=factor_path,
                )

    return graph


def _layered_layout(graph: Any) -> Dict[Any, Tuple[float, float]]:
    positions: Dict[Any, Tuple[float, float]] = {}
    layers: Dict[int, List[Any]] = {}
    for node, attributes in graph.nodes(data=True):
        layers.setdefault(int(attributes["layer"]), []).append(node)

    for layer, nodes in layers.items():
        nodes.sort(
            key=lambda node: (
                graph.nodes[node].get("order", 0),
                repr(node),
            )
        )
        midpoint = (len(nodes) - 1) / 2.0
        for index, node in enumerate(nodes):
            positions[node] = (float(layer), midpoint - index)
    return positions


def _resolve_layout(graph: Any, layout: Any) -> Dict[Any, Tuple[float, float]]:
    if layout == "layered":
        return _layered_layout(graph)
    if isinstance(layout, Mapping):
        missing = [node for node in graph if node not in layout]
        if missing:
            raise ValueError(
                "Custom layout is missing positions for nodes: {!r}.".format(missing)
            )
        return {node: tuple(layout[node]) for node in graph}
    raise ValueError("layout must be 'layered' or a complete node-position mapping.")


def _domain_text(domain: Optional[Sequence[Any]]) -> str:
    if domain is None:
        return ""
    return "{" + ", ".join(repr(value) for value in domain) + "}"


def _wrapped_factor_label(label: str) -> str:
    """Compact a conditional-kernel label to fit inside a square node.

    Incoming arrows already communicate the conditioning scope; the complete
    formula remains available in the NetworkX ``label`` attribute.
    """

    if " | " not in label:
        return label
    outputs, _ = label.split(" | ", 1)
    return "{}\n| ·)".format(outputs)


def _node_labels(
    graph: Any,
    labels: Optional[Mapping[Any, str]],
    show_domains: bool,
) -> Dict[Any, str]:
    if labels is not None and not isinstance(labels, Mapping):
        raise TypeError("labels must be a mapping keyed by node ID or variable name.")
    if labels is not None:
        valid_keys = set(graph.nodes)
        valid_keys.update(
            attributes["variable"]
            for _, attributes in graph.nodes(data=True)
            if "variable" in attributes
        )
        unknown_keys = [key for key in labels if key not in valid_keys]
        if unknown_keys:
            raise ValueError(
                "labels contains unknown node or variable keys: {!r}.".format(
                    unknown_keys
                )
            )

    resolved = {}
    for node, attributes in graph.nodes(data=True):
        label = attributes["label"]
        overridden = False
        if labels is not None:
            if node in labels:
                label = str(labels[node])
                overridden = True
            elif attributes.get("variable") in labels:
                label = str(labels[attributes["variable"]])
                overridden = True
        if attributes["kind"] == "factor" and not overridden:
            label = _wrapped_factor_label(label)
        if show_domains and attributes["kind"] == "variable":
            domain = _domain_text(attributes.get("domain"))
            if domain:
                label = "{}\n{}".format(label, domain)
        resolved[node] = label
    return resolved


def draw_mdp(
    model: AbstractMDP,
    *,
    ax: Optional[Any] = None,
    view: str = "factor",
    layout: Any = "layered",
    show_domains: bool = False,
    labels: Optional[Mapping[Any, str]] = None,
    **style: Any,
) -> Any:
    """Draw an MDP without showing or saving the Matplotlib figure."""

    unknown_style = sorted(set(style) - set(_DRAW_DEFAULTS))
    if unknown_style:
        raise ValueError(
            "Unknown draw style option(s): {}. Supported options are {}.".format(
                unknown_style, sorted(_DRAW_DEFAULTS)
            )
        )
    options = dict(_DRAW_DEFAULTS)
    options.update(style)

    graph = to_networkx(model, view=view)
    nx = _require_module("networkx", "viz", "draw()")
    pyplot = _require_module("matplotlib.pyplot", "viz", "draw()")
    if ax is None:
        _, ax = pyplot.subplots()

    positions = _resolve_layout(graph, layout)
    for role in ("current", "parent", "factor", "next"):
        nodes = [
            node
            for node, attributes in graph.nodes(data=True)
            if attributes["role"] == role
        ]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=ax,
            nodelist=nodes,
            node_color=options["{}_color".format(role)],
            node_shape=options["{}_shape".format(role)],
            node_size=options["node_size"],
            alpha=options["alpha"],
            linewidths=options["linewidths"],
            edgecolors="#444444",
        )

    external_edges: List[Tuple[Any, Any]] = []
    regular_edges: List[Tuple[Any, Any]] = []
    for source, target, attributes in graph.edges(data=True):
        input_kind = attributes.get("input_kind", attributes.get("kind"))
        target_list = (
            external_edges if input_kind == "external_parent" else regular_edges
        )
        target_list.append((source, target))
    for edges, line_style in (
        (regular_edges, "solid"),
        (external_edges, "dashed"),
    ):
        if not edges:
            continue
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=ax,
            edgelist=edges,
            edge_color=options["edge_color"],
            width=options["width"],
            style=line_style,
            arrows=True,
            arrowsize=options["arrowsize"],
            arrowstyle="-|>",
            node_size=options["node_size"],
        )
    nx.draw_networkx_labels(
        graph,
        positions,
        ax=ax,
        labels=_node_labels(graph, labels, show_domains),
        font_size=options["font_size"],
        font_color=options["font_color"],
    )
    ax.margins(0.15)
    ax.set_axis_off()
    return ax


def _graphviz_node_id(node: Tuple[str, Any]) -> str:
    role, identifier = node
    if isinstance(identifier, tuple):
        identifier = list(identifier)
    return json.dumps([role, identifier], ensure_ascii=False, separators=(",", ":"))


def to_graphviz(
    model: AbstractMDP,
    *,
    view: str = "factor",
    rankdir: str = "LR",
) -> Any:
    """Return a Graphviz ``Digraph`` without rendering it."""

    view = _validate_view(view)
    rankdir = rankdir.upper()
    if rankdir not in _VALID_RANK_DIRECTIONS:
        raise ValueError(
            "rankdir must be one of {}; received {!r}.".format(
                _VALID_RANK_DIRECTIONS, rankdir
            )
        )

    graphviz = _require_module("graphviz", "graphviz", "to_graphviz()")
    structure = to_networkx(model, view=view)
    dot = graphviz.Digraph(
        name=type(model).__name__,
        graph_attr={"rankdir": rankdir},
        node_attr={"fontname": "Helvetica"},
        edge_attr={"color": "#555555"},
    )

    for node, attributes in structure.nodes(data=True):
        role = attributes["role"]
        dot.node(
            _graphviz_node_id(node),
            label=attributes["label"],
            shape=(
                "box"
                if role == "factor"
                else "diamond"
                if role == "parent"
                else "ellipse"
            ),
            style="filled",
            fillcolor=_ROLE_STYLES[role]["color"],
        )
    for source, target, attributes in structure.edges(data=True):
        input_kind = attributes.get("input_kind", attributes.get("kind"))
        edge_attributes = {"style": "dashed"} if input_kind == "external_parent" else {}
        dot.edge(
            _graphviz_node_id(source),
            _graphviz_node_id(target),
            **edge_attributes,
        )

    for layer in (0, 1, 2):
        layer_nodes = [
            node
            for node, attributes in structure.nodes(data=True)
            if attributes["layer"] == layer
        ]
        if not layer_nodes:
            continue
        with dot.subgraph(name="layer_{}".format(layer)) as same_rank:
            same_rank.attr(rank="same")
            for node in layer_nodes:
                same_rank.node(_graphviz_node_id(node))

    return dot


__all__ = ["draw_mdp", "to_graphviz", "to_networkx"]
