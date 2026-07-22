# Evolving World Models

[![Quality](https://github.com/nquaz/evolving-world-models/actions/workflows/quality.yml/badge.svg)](https://github.com/nquaz/evolving-world-models/actions/workflows/quality.yml)

This repository studies how the objectives used to learn a world model shape
performance on later tasks. The motivating distinction is between learning
predictable but uncontrollable regularities and learning action-sensitive
regularities that support control.

The first milestone implements the transition-model foundation. Rewards,
Bayesian beliefs over transition parameters, learning algorithms, and the
downstream-task distribution are intentionally left for later milestones.

## Transition model

`AbstractMDP` represents a named conditional transition model

$$
p(x' \mid x, \operatorname{pa}(x)).
$$

Here, `variables` is the scope of the values in $x$ and `parent_variables`
is the union of the values in $\operatorname{pa}(x)$. A conventional MDP is
the special case where `variables` describes the state and
`parent_variables` describes the action.

`FactoredMDP` combines constituent models with disjoint predicted scopes:

$$
p(x' \mid x, \operatorname{pa}(x))
= \prod_i p_i(x_i' \mid x_i, \operatorname{pa}(x_i)).
$$

A factor may use another factor's variable as a parent. The composition reads
that parent's **current-time** value from the joint $x$; only parents not
predicted by any factor remain external inputs to the composite model. Factor
updates are therefore synchronous.

## Repository layout

- `scripts/mdp.py` contains variables, immutable assignments, transition
  distributions, the abstract MDP interfaces, the complete finite
  `TabularMDP`, and the product `FactoredMDP`.
- `scripts/visualization.py` contains the optional NetworkX, Matplotlib, and
  Graphviz adapters used by the inherited visualization methods.
- `notebooks/Factored_MDP_Demo.ipynb` builds a weather/pantry world, verifies
  the factor product exactly, checks every transition context for
  normalization, and contrasts action-sensitive with action-invariant
  dynamics.
- `docs/drawing.md` documents graph semantics, rendering backends, and the
  visualization API.
- `tests/test_mdp.py` tests the dependency-free transition core, while
  `tests/test_visualization.py` tests graph structure and rendering adapters.
- `.github/workflows/quality.yml` independently checks supported Python
  versions, executes the demonstration notebook, and builds the distribution.

The transition-model core uses only the Python standard library and supports
Python 3.9 or newer. Visualization dependencies are optional.

## Installation

Install the package in editable mode with the capabilities you need:

```bash
python -m pip install -e '.[viz]'       # NetworkX + Matplotlib
python -m pip install -e '.[graphviz]'  # NetworkX + Python Graphviz wrapper
python -m pip install -e '.[dev]'       # quality tools, tests, notebooks, backends
```

Graphviz file rendering also requires the Graphviz system executables, such as
`dot`, to be installed and available on `PATH`.

## Core API

```python
from scripts.mdp import FactoredMDP, TabularMDP, Variable

weather = Variable("weather", ("sunny", "rainy"))

weather_model = TabularMDP(
    variables=(weather,),
    transitions=(
        (
            {"weather": "sunny"},
            {},
            (({"weather": "sunny"}, 0.8), ({"weather": "rainy"}, 0.2)),
        ),
        (
            {"weather": "rainy"},
            {},
            (({"weather": "sunny"}, 0.3), ({"weather": "rainy"}, 0.7)),
        ),
    ),
)

probability = weather_model.transition_probability(
    next_state={"weather": "sunny"},
    current={"weather": "rainy"},
)
assert probability == 0.3
```

The public operations are:

- `transition_distribution(current, parents=None)`
- `transition_probability(next_state, current, parents=None)`
- `transition_log_probability(next_state, current, parents=None)`
- `sample_transition(current, parents=None, rng=None)`
- `to_dict()` and `str(model)` for JSON-safe model inspection
- `to_networkx(view="factor")`
- `draw(ax=None, view="factor", layout="layered", ...)`
- `to_graphviz(view="factor", rankdir="LR")`

All calls validate exact variable names and domain membership. Tabular rows are
also checked for complete conditioning-context coverage, valid next-state
assignments, and normalized finite probabilities.

## Model descriptions

All MDP subclasses inherit `AbstractMDP.__str__`. It returns deterministic,
indented JSON containing the model type, variables, parent variables, and
subclass-specific fields. Tabular models include canonical transition rows;
factored models recursively include their constituent models.

```python
print(weather_model)
description = weather_model.to_dict()
```

The representation is designed for inspection and logging, not as a stable
serialization/deserialization format. See `docs/drawing.md` for the
visualization layer built on the same structural metadata.

## Visualization

Every MDP inherits three visualization methods. `to_networkx()` exposes the
directed graph and its metadata, `draw()` renders it onto a supplied or new
Matplotlib axes without showing or saving it, and `to_graphviz()` returns an
unrendered Graphviz `Digraph`.

```python
graph = weather_model.to_networkx()
ax = weather_model.draw(show_domains=True)
dot = weather_model.to_graphviz(rankdir="LR")
```

The default factor view is a two-slice graph with explicit transition-factor
nodes. The optional `view="dependencies"` removes factor nodes and shows all
possible input-to-output dependencies implied by each factor scope. See
`docs/drawing.md` for the exact node/edge schema and customization options.

## Development workflow

The development extra pins Ruff, mypy, and pre-commit. Install the Git hooks
after creating or activating the development environment:

```bash
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
```

The pre-commit stage applies repository hygiene checks, safe Ruff lint fixes,
Ruff formatting, and strict type checking. The pre-push stage runs the complete
unit suite and compilation checks. Run both stages across the repository before
requesting review:

```bash
pre-commit run --all-files
pre-commit run --all-files --hook-stage pre-push
```

Hooks can modify files. Review those edits and rerun the commands until every
hook passes. See `AGENTS.md` for commit construction, shared-worktree safety,
and the complete research-production requirements.

GitHub Actions repeats the non-mutating quality gates on Python 3.9 and 3.12,
executes the notebook from a fresh kernel, and builds a wheel. The workflow has
read-only repository permissions and pins third-party actions to immutable
commit identifiers.

## Verification

Run the non-mutating quality checks from the repository root:

```bash
pre-commit validate-config .pre-commit-config.yaml
python -m ruff check scripts tests
python -m ruff format --check scripts tests
python -m mypy scripts
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
python -m json.tool notebooks/Factored_MDP_Demo.ipynb > /dev/null
```

Open `notebooks/Factored_MDP_Demo.ipynb` in Jupyter to run the worked example.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` for the full
terms.
