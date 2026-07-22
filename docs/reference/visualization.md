# Transition-structure visualization

The visualization layer exposes the declared structure of an MDP transition
model without adding plotting dependencies to the transition-model core.
Optional libraries are imported only when a visualization method is called.

The diagrams are structural: they show variable scopes and possible
dependencies, not transition probabilities, effect sizes, or learned causal
relationships.

## Installation

Install NetworkX and Matplotlib for graph inspection and drawing:

```bash
python -m pip install -e '.[viz]'
```

Install NetworkX and the Python Graphviz wrapper for Graphviz export:

```bash
python -m pip install -e '.[graphviz]'
```

`to_graphviz()` only constructs a `graphviz.Digraph`. Rendering that object to a
file also requires Graphviz system executables such as `dot` on `PATH`.

## Public API

Every `AbstractMDP` inherits:

```python
model.to_networkx(view="factor")
model.draw(
    ax=None,
    view="factor",
    layout="layered",
    show_domains=False,
    labels=None,
    **style,
)
model.to_graphviz(view="factor", rankdir="LR")
```

The corresponding adapter functions are exported from
`scripts.visualization`:

```python
to_networkx(model, view="factor")
draw_mdp(
    model,
    ax=None,
    view="factor",
    layout="layered",
    show_domains=False,
    labels=None,
    **style,
)
to_graphviz(model, view="factor", rankdir="LR")
```

Prefer the inherited methods in application and notebook code.

## Runnable example

This example creates two deterministic factors. `weather` is predicted by the
first factor. The umbrella factor reads the current weather synchronously, while
`season` and `action` remain external parents of the composite model.

```python
from itertools import product
from typing import Sequence

from scripts.mdp import FactoredMDP, TabularMDP, Variable


def identity_factor(
    variable: Variable,
    parents: Sequence[Variable] = (),
) -> TabularMDP:
    """Create a complete deterministic table for visualization examples."""

    if variable.domain is None:
        raise ValueError("The predicted variable must have a finite domain.")

    parent_domains = []
    for parent in parents:
        if parent.domain is None:
            raise ValueError("Every parent must have a finite domain.")
        parent_domains.append(parent.domain)

    rows = []
    for values in product(variable.domain, *parent_domains):
        current_value, *parent_values = values
        rows.append(
            (
                {variable.name: current_value},
                {
                    parent.name: value
                    for parent, value in zip(parents, parent_values)
                },
                (({variable.name: current_value}, 1.0),),
            )
        )

    return TabularMDP(
        variables=(variable,),
        parent_variables=tuple(parents),
        transitions=rows,
    )


weather = Variable("weather", ("sun", "rain"))
umbrella = Variable("umbrella", ("closed", "open"))
season = Variable("season", ("dry", "wet"))
action = Variable("action", ("open", "close"))

weather_factor = identity_factor(weather, (season,))
umbrella_factor = identity_factor(umbrella, (weather, season, action))
world = FactoredMDP((weather_factor, umbrella_factor))
```

Inspect the graph as structured data:

```python
graph = world.to_networkx(view="factor")

assert graph.graph == {
    "model_type": "FactoredMDP",
    "view": "factor",
    "semantics": "two_slice_transition_model",
}
assert graph.nodes[("factor", (1,))]["parents"] == (
    "weather",
    "season",
    "action",
)
assert graph.edges[
    ("current", "weather"),
    ("factor", (1,)),
]["kind"] == "internal_parent"

for node, attributes in graph.nodes(data=True):
    print(node, attributes["role"], attributes["label"])

for source, target, attributes in graph.edges(data=True):
    print(source, "->", target, attributes["kind"])
```

## Graph views and semantics

Both views are directed two-slice transition graphs.

### `view="factor"`

The default view preserves explicit transition-factor nodes:

```text
current variables ─┐
external parents ──┼─> transition factor ─> next variables
internal parents ──┘
```

Each leaf factor represents its declared conditional kernel. For example,

```text
P(umbrella′ | umbrella, weather, season, action)
```

The graph does not claim that every declared input has a nonzero effect in
every transition table row.

A factor parent predicted elsewhere in the composite is represented by that
variable's current-time node. It is not represented as an external parent or
read from the variable's next-time node. This preserves the synchronous
transition semantics of `FactoredMDP`.

### `view="dependencies"`

The dependency view removes factor nodes and connects every declared factor
input directly to each output of that factor:

```python
dependencies = world.to_networkx(view="dependencies")

assert not any(
    attributes["kind"] == "factor"
    for _, attributes in dependencies.nodes(data=True)
)
assert dependencies.edges[
    ("current", "weather"),
    ("next", "umbrella"),
]["input_kind"] == "internal_parent"
```

These edges mean “possible dependency implied by the declared factor scope.”
They are not a minimal conditional-independence graph, proof of causal effect,
or assertion that changing the input changes the output distribution.

`view` is case-sensitive and must be exactly `"factor"` or `"dependencies"`.

## NetworkX schema

`to_networkx()` returns a new `networkx.DiGraph` on every call. Given the same
model, node and edge structure and iteration order are deterministic. Mutating
the returned graph does not mutate the MDP.

### Graph attributes

| Attribute | Value |
| --- | --- |
| `model_type` | Concrete model class name |
| `view` | `"factor"` or `"dependencies"` |
| `semantics` | `"two_slice_transition_model"` |

### Variable nodes

Variable node identifiers are:

| Role | Node identifier | Layer |
| --- | --- | ---: |
| Current predicted variable | `("current", variable_name)` | 0 |
| External parent | `("parent", variable_name)` | 0 |
| Next predicted variable | `("next", variable_name)` | 2 |

Every variable node has:

| Attribute | Meaning |
| --- | --- |
| `kind` | `"variable"` |
| `role` | `"current"`, `"parent"`, or `"next"` |
| `layer` | `0` for inputs and `2` for outputs |
| `order` | Stable order derived from the model declaration |
| `variable` | Variable name |
| `domain` | The declared domain tuple, or `None` |
| `time` | `"t"` for inputs or `"t+1"` for outputs |
| `label` | For example, `"weather (t+1)"` |

Current and next nodes use the order of `model.variables`. External parents
follow them in the order of `model.parent_variables`.

### Factor nodes

Factor nodes occur only in the factor view. Their identifier is:

```python
("factor", factor_path)
```

`factor_path` is a tuple of child indexes through nested factorizations.
Top-level factors therefore have paths such as `(0,)` and `(1,)`; nested leaves
may have paths such as `(0, 1)`. Nested factored models are flattened to their
leaf kernels in stable declaration order.

Every factor node has:

| Attribute | Meaning |
| --- | --- |
| `kind` | `"factor"` |
| `role` | `"factor"` |
| `layer` | `1` |
| `order` | Stable leaf-factor order |
| `factor_index` | The factor path |
| `factor_path` | The factor path |
| `factor_type` | Concrete leaf-factor class name |
| `variables` | Tuple of predicted variable names |
| `parents` | Tuple of declared parent names |
| `label` | Full conditional-kernel label |

The full label lists predicted variables before the conditioning bar, then the
factor's current variables followed by its declared parents.

### Factor-view edges

Every factor-view edge records `factor_index` and `factor_path`.

| Source → target | `kind` | Meaning |
| --- | --- | --- |
| Current variable → factor | `current_state` | The factor predicts that variable |
| Current variable → factor | `internal_parent` | Another factor predicts this parent |
| External parent → factor | `external_parent` | The parent is not predicted by the composite |
| Factor → next variable | `prediction` | The factor predicts the output |

### Dependency-view edges

Each direct input-to-output edge has:

| Attribute | Value |
| --- | --- |
| `kind` | `"possible_dependency"` |
| `input_kind` | `"current_state"`, `"internal_parent"`, or `"external_parent"` |
| `factor_index` | Path of the originating leaf factor |
| `factor_path` | Path of the originating leaf factor |

The documented graph metadata is an inspection contract. Changes to identifiers,
attributes, meanings, or deterministic ordering should update the implementation,
tests, examples, and this document together.

## Matplotlib drawing

`draw()` returns a Matplotlib `Axes`. It uses the deterministic layered layout
unless the caller supplies a complete node-position mapping.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
world.draw(
    ax=ax,
    show_domains=True,
    node_size=3_500,
    font_size=8,
)
ax.set_title("Weather–umbrella transition structure")
fig.tight_layout()
fig.savefig(
    "weather-umbrella-transition-structure.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)
```

Saving the PNG in this example is an explicit caller-controlled side effect.
`draw()` itself never calls `show()` and never writes a file. If `ax` is
omitted, it creates a figure and axes internally and returns the axes. In both
cases it disables the coordinate axes because the positions have no
quantitative meaning.

A suitable standalone caption is:

> Declared two-slice factor structure for the deterministic weather–umbrella
> example. Circles denote current and next variables, diamonds denote external
> parents, squares denote transition factors, and dashed arrows denote external
> conditioning inputs. The diagram is structural and deterministic; sample
> size, physical units, and sampling uncertainty are not applicable.

### Layout

`layout="layered"` places:

- current variables and external parents in layer 0;
- factor nodes, when present, in layer 1; and
- next variables in layer 2.

Nodes within each layer use stable declared order. Non-singleton layers span the
same vertical range as the densest layer so automatically enlarged factor
markers do not crowd at the center; a singleton remains centered. Arbitrary
NetworkX layout names such as `"spring"` are not accepted.

A custom layout must be a mapping containing every graph node:

```python
graph = world.to_networkx()
positions = {
    node: (float(attributes["layer"]) * 3.0, -float(index) * 1.5)
    for index, (node, attributes) in enumerate(graph.nodes(data=True))
}

fig, ax = plt.subplots(figsize=(10, 7))
world.draw(ax=ax, layout=positions)
ax.set_title("Custom transition-graph layout")
fig.tight_layout()
plt.close(fig)
```

Extra mapping entries are ignored. Missing graph nodes raise `ValueError`.
Position values are passed to NetworkX and should be valid plotting
coordinates.

### Labels and domains

Set `show_domains=True` to append each variable's finite domain using value
representations:

```text
weather (t)
{'sun', 'rain'}
```

Customize labels with a mapping keyed by a variable name or an exact node
identifier:

```python
world.draw(
    labels={
        "weather": "Weather",
        ("next", "weather"): "Tomorrow's weather",
        ("factor", (1,)): "Umbrella transition",
    }
)
```

A node-specific label takes precedence over a variable-name label.
Variable-name labels apply to each node for that variable unless overridden.
Domain text is appended after resolving the custom label.

NetworkX stores full factor labels. Matplotlib drawing compacts an unmodified
factor label into a single-line form such as `P(weather′|·)` because arrows
already show the conditioning scope. When `node_size` is omitted, all factor
squares use a common area large enough to contain the widest resolved factor
label. Explicit factor-node label overrides remain verbatim, including blank,
whitespace-only, and multiline labels, and participate in automatic sizing.

Supplying `node_size` disables automatic factor growth and sets the exact,
uniform marker area for every node role. This lets callers control marker sizes
directly when composing a figure with a custom layout.

Directed edges terminate at boundaries computed from each endpoint's resolved
marker size and shape, preserving visible arrowheads when different roles use
circles, diamonds, and squares. Boundary calculations are exact for those
default shapes. Other built-in markers use a conservative circumscribed-square
clearance; custom `Path` markers use that same approximate fallback.
Establish the final figure size and axes layout before drawing, or redraw after
an operation that materially changes the axes geometry.

`labels` must be a mapping. Unknown node or variable keys raise `ValueError`.

### Default role encodings

| Role | Color | Matplotlib shape | Graphviz shape |
| --- | --- | --- | --- |
| Current variable | `#9ECAE1` | `o` (circle) | `ellipse` |
| External parent | `#FDD0A2` | `D` (diamond) | `diamond` |
| Transition factor | `#D9D9D9` | `s` (square) | `box` |
| Next variable | `#A1D99B` | `o` (circle) | `ellipse` |

External-parent edges are dashed. Other edges are solid. Variable labels also
encode time, so current and next variables are not distinguished by color
alone.

### Style options

Only the following keyword options are accepted:

| Option | Default |
| --- | --- |
| `node_size` | `2300` pt²; optional automatic factor growth |
| `font_size` | `9` |
| `font_color` | `"#111111"` |
| `edge_color` | `"#555555"` |
| `width` | `1.5` |
| `arrowsize` | `18` |
| `alpha` | `1.0` |
| `linewidths` | `1.0` |
| `current_color` | `"#9ECAE1"` |
| `parent_color` | `"#FDD0A2"` |
| `factor_color` | `"#D9D9D9"` |
| `next_color` | `"#A1D99B"` |
| `current_shape` | `"o"` |
| `parent_shape` | `"D"` |
| `factor_shape` | `"s"` |
| `next_shape` | `"o"` |

Style values are forwarded to the corresponding NetworkX drawing operations
after the factor-sizing and marker-aware edge-clipping behavior described
above. `node_size` and `font_size` must be finite nonnegative scalars.
Iterable `edge_color`, `width`, and `arrowsize` values are applied per edge and
cycle independently within the solid and dashed edge groups.
Unsupported option names raise `ValueError` rather than being silently ignored.

For publication use, choose the final figure size before tuning text and node
sizes, add a concise title and standalone caption, and preserve the generating
code. These structural diagrams are coordinate-free, so scientific x- and
y-axis labels are not applicable.

## Graphviz export

`to_graphviz()` returns an unrendered `graphviz.Digraph`:

```python
dot = world.to_graphviz(view="factor", rankdir="LR")
print(dot.source)
```

Rendering is a separate, explicit operation:

```python
output_path = dot.render(
    "weather-umbrella-transition-structure",
    format="svg",
    cleanup=True,
)
print(output_path)
```

The second example writes an SVG and requires a working Graphviz executable.

`rankdir` is case-insensitive and must be one of:

- `"TB"`: top to bottom;
- `"BT"`: bottom to top;
- `"LR"`: left to right; or
- `"RL"`: right to left.

Graphviz export uses:

- full, unwrapped factor labels;
- stable JSON-derived node identifiers;
- the role colors and shapes listed above;
- dashed external-parent edges;
- `#555555` edges and Helvetica node text; and
- same-rank subgraphs for each populated semantic layer.

Graphviz currently does not append variable domains or accept Matplotlib style
overrides.

## Validation and failure modes

The adapters fail explicitly when:

- `view` is not `"factor"` or `"dependencies"`;
- `layout` is neither `"layered"` nor a complete position mapping;
- a custom layout omits graph nodes;
- `labels` is not a mapping or contains unknown keys;
- an unsupported drawing style option is supplied;
- `rankdir` is invalid;
- a factored composition contains a cycle;
- flattened leaf-factor outputs do not partition the composite output scope;
- a factor refers to an input absent from the composite scope; or
- a required optional dependency is unavailable.

Missing optional-library errors name the dependency and the appropriate
installation extra. Invalid `rankdir` values are rejected before Graphviz is
imported.

## Side effects and determinism

Importing `scripts.mdp` or `scripts.visualization` does not import optional
plotting libraries, create figures, open windows, write files, or render
Graphviz output.

Calling:

- `to_networkx()` constructs and returns a fresh in-memory graph;
- `draw()` may create a Matplotlib figure when no axes are supplied, but does
  not show or save it; and
- `to_graphviz()` constructs an in-memory `Digraph`, but does not render it.

For a fixed model, view, labels, layout, and style configuration, graph
structure and default layout are deterministic. Rendering can still vary
slightly across Matplotlib, NetworkX, Graphviz, font, and platform versions.

## Verification

Visualization contracts are covered by `tests/test_visualization.py`, including
schema equality, stable ordering, nested factor paths, synchronous internal
parents, both views, label and domain behavior, custom layouts, headless
drawing, Graphviz output, and failure cases.

Run the focused suite from the repository root:

```bash
python -m unittest discover -s tests -p 'test_visualization.py' -v
```
