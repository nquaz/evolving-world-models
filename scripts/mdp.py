"""Core transition-model abstractions for discrete, factored MDPs.

The module deliberately keeps rewards out of the transition model.  A model
describes a conditional distribution

    p(x' | x, pa(x)),

where ``variables`` is the named scope of ``x`` and ``parent_variables`` is
the named scope of ``pa(x)``.  This lets the same learned dynamics later be
paired with many intrinsic or extrinsic reward functions.
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import product
from types import MappingProxyType
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Variable:
    """A named random variable and, optionally, its finite domain.

    A finite domain is required by :class:`TabularMDP`.  Leaving ``domain``
    as ``None`` keeps the abstract interfaces usable by future parametric or
    continuous transition distributions.
    """

    name: str
    domain: Optional[Tuple[Hashable, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("A variable name must be a non-empty string.")

        if self.domain is None:
            return

        domain = tuple(self.domain)
        if not domain:
            raise ValueError(
                "Variable {!r} must have a non-empty domain.".format(self.name)
            )
        for value in domain:
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    "Domain values must be hashable; {!r} in {!r} is not.".format(
                        value, self.name
                    )
                ) from error
        if len(set(domain)) != len(domain):
            raise ValueError(
                "Variable {!r} has duplicate domain values.".format(self.name)
            )
        object.__setattr__(self, "domain", domain)

    def accepts(self, value: Hashable) -> bool:
        """Return whether ``value`` belongs to this variable's domain."""

        return self.domain is None or value in self.domain


def _json_safe_value(value: Hashable) -> Any:
    """Convert a hashable domain value into a deterministic JSON value.

    JSON-native values remain unchanged. Common non-JSON hashable values use
    tagged encodings so that, for example, a tuple-valued state is not confused
    with a list used by the representation schema. Unknown custom values fall
    back to a tagged ``repr``; deterministic output then requires that object's
    ``repr`` to itself be deterministic.
    """

    if isinstance(value, Enum):
        enum_type = type(value)
        return {
            "type": "{}.{}".format(enum_type.__module__, enum_type.__qualname__),
            "name": value.name,
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "infinity"
        else:
            label = "-infinity"
        return {"type": "float", "value": label}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_json_safe_value(item) for item in value],
        }
    if isinstance(value, frozenset):
        items = [_json_safe_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        return {"type": "frozenset", "items": items}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}

    custom_type = type(value)
    return {
        "type": "{}.{}".format(custom_type.__module__, custom_type.__qualname__),
        "repr": repr(value),
    }


def _variable_description(variable: Variable) -> Dict[str, Any]:
    domain = None
    if variable.domain is not None:
        domain = [_json_safe_value(value) for value in variable.domain]
    return {"name": variable.name, "domain": domain}


def _assignment_description(assignment: Mapping[str, Hashable]) -> Dict[str, Any]:
    return {name: _json_safe_value(value) for name, value in assignment.items()}


class Assignment(Mapping[str, Hashable]):
    """An immutable, hashable mapping from variable names to values."""

    __slots__ = ("_items", "_values")

    def __init__(
        self,
        values: Optional[Mapping[str, Hashable]] = None,
        **named_values: Hashable,
    ) -> None:
        combined: Dict[str, Hashable] = {}
        if values is not None:
            if not isinstance(values, Mapping):
                raise TypeError("Assignment values must be supplied as a mapping.")
            combined.update(values)
        overlap = set(combined).intersection(named_values)
        if overlap:
            raise ValueError(
                "Assignment received duplicate keys: {}.".format(sorted(overlap))
            )
        combined.update(named_values)

        for name, value in combined.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Assignment keys must be non-empty strings.")
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    "Assignment value for {!r} must be hashable.".format(name)
                ) from error

        self._items = tuple(combined.items())
        self._values = MappingProxyType(dict(self._items))

    def __getitem__(self, key: str) -> Hashable:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(frozenset(self._items))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return "Assignment({!r})".format(dict(self._items))

    def to_dict(self) -> Dict[str, Hashable]:
        """Return a mutable copy, useful for display and serialization."""

        return dict(self._items)

    def project(self, names: Iterable[str]) -> "Assignment":
        """Project this assignment onto ``names``, preserving their order."""

        return Assignment({name: self[name] for name in names})

    @classmethod
    def merge(cls, *assignments: Mapping[str, Hashable]) -> "Assignment":
        """Merge compatible assignments, rejecting conflicting values."""

        merged: Dict[str, Hashable] = {}
        for assignment in assignments:
            for name, value in assignment.items():
                if name in merged and merged[name] != value:
                    raise ValueError(
                        "Cannot merge conflicting values for {!r}: {!r} and {!r}.".format(
                            name, merged[name], value
                        )
                    )
                merged[name] = value
        return cls(merged)


def _validate_variable_specs(
    variables: Sequence[Variable], label: str, allow_empty: bool
) -> Tuple[Variable, ...]:
    specs = tuple(variables)
    if not allow_empty and not specs:
        raise ValueError("{} must contain at least one variable.".format(label))
    if any(not isinstance(variable, Variable) for variable in specs):
        raise TypeError("{} must contain only Variable objects.".format(label))
    names = [variable.name for variable in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("{} contains duplicate names: {}.".format(label, duplicates))
    return specs


def _coerce_assignment(
    values: Optional[Mapping[str, Hashable]],
    variables: Sequence[Variable],
    label: str,
) -> Assignment:
    if values is None:
        values = {}
    if not isinstance(values, Mapping):
        raise TypeError(
            "{} must be a mapping from variable names to values.".format(label)
        )

    expected_names = [variable.name for variable in variables]
    expected = set(expected_names)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append("missing {}".format(missing))
        if extra:
            details.append("unexpected {}".format(extra))
        raise ValueError("{} has {}.".format(label, " and ".join(details)))

    canonical: Dict[str, Hashable] = {}
    for variable in variables:
        value = values[variable.name]
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(
                "{} value for {!r} must be hashable.".format(label, variable.name)
            ) from error
        if not variable.accepts(value):
            raise ValueError(
                "{} value {!r} is outside the domain of {!r}: {!r}.".format(
                    label, value, variable.name, variable.domain
                )
            )
        canonical[variable.name] = value
    return Assignment(canonical)


class TransitionDistribution(ABC):
    """A conditional distribution over a named next-state scope."""

    def __init__(self, variables: Sequence[Variable]) -> None:
        self._variables = _validate_variable_specs(
            variables, "Distribution variables", allow_empty=False
        )

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self._variables

    def probability(self, outcome: Mapping[str, Hashable]) -> float:
        """Evaluate the probability mass (or future density) of an outcome."""

        canonical = _coerce_assignment(outcome, self.variables, "Outcome")
        probability = float(self._probability(canonical))
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(
                "A transition distribution returned invalid probability {!r}.".format(
                    probability
                )
            )
        return probability

    @abstractmethod
    def _probability(self, outcome: Assignment) -> float:
        """Evaluate a validated, canonical outcome."""

    def log_probability(self, outcome: Mapping[str, Hashable]) -> float:
        """Evaluate log probability, returning ``-inf`` for impossible outcomes."""

        probability = self.probability(outcome)
        return -math.inf if probability == 0.0 else math.log(probability)

    def sample(self, rng: Optional[Any] = None) -> Assignment:
        """Draw one outcome using an object exposing ``random()``."""

        generator = random.Random() if rng is None else rng
        if not callable(getattr(generator, "random", None)):
            raise TypeError("rng must expose a callable random() method.")
        sampled = self._sample(generator)
        return _coerce_assignment(sampled, self.variables, "Sampled outcome")

    @abstractmethod
    def _sample(self, rng: Any) -> Mapping[str, Hashable]:
        """Sample an outcome; public validation is handled by :meth:`sample`."""

    def items(self) -> Tuple[Tuple[Assignment, float], ...]:
        """Enumerate weighted outcomes when supported.

        General parametric or continuous distributions need not be enumerable.
        """

        raise TypeError("This transition distribution is not finitely enumerable.")


class CategoricalDistribution(TransitionDistribution):
    """A validated finite probability mass function over assignments."""

    def __init__(
        self,
        variables: Sequence[Variable],
        probabilities: Iterable[Tuple[Mapping[str, Hashable], float]],
    ) -> None:
        super().__init__(variables)
        entries = (
            probabilities.items()
            if isinstance(probabilities, Mapping)
            else probabilities
        )
        canonical_probabilities: Dict[Assignment, float] = {}
        for outcome, probability in entries:
            canonical = _coerce_assignment(
                outcome, self.variables, "Categorical outcome"
            )
            numeric_probability = float(probability)
            if (
                not math.isfinite(numeric_probability)
                or numeric_probability < 0.0
                or numeric_probability > 1.0
            ):
                raise ValueError(
                    "Probability for {!r} must be finite and lie in [0, 1].".format(
                        canonical
                    )
                )
            if canonical in canonical_probabilities:
                raise ValueError(
                    "Categorical distribution contains duplicate outcome {!r}.".format(
                        canonical
                    )
                )
            canonical_probabilities[canonical] = numeric_probability

        if not canonical_probabilities:
            raise ValueError("A categorical distribution needs at least one outcome.")
        total = math.fsum(canonical_probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                "Categorical probabilities must sum to 1; received {:.17g}.".format(
                    total
                )
            )
        # Normalize values admitted by the floating-point tolerance so
        # probability queries and inverse-CDF sampling describe exactly the
        # same PMF.
        self._probabilities = {
            outcome: probability / total
            for outcome, probability in canonical_probabilities.items()
        }

    @property
    def probabilities(self) -> Mapping[Assignment, float]:
        """Return a defensive copy of the outcome-to-probability mapping."""

        return dict(self._probabilities)

    def _probability(self, outcome: Assignment) -> float:
        return self._probabilities.get(outcome, 0.0)

    def _sample(self, rng: Any) -> Assignment:
        draw = float(rng.random())
        if not 0.0 <= draw < 1.0:
            raise ValueError("rng.random() must return a value in [0, 1).")
        cumulative = 0.0
        last_outcome = None
        for outcome, probability in self._probabilities.items():
            cumulative += probability
            last_outcome = outcome
            if draw < cumulative:
                return outcome
        # Handles a tiny floating-point gap in an otherwise normalized PMF.
        assert last_outcome is not None
        return last_outcome

    def items(self) -> Tuple[Tuple[Assignment, float], ...]:
        return tuple(self._probabilities.items())


class ProductDistribution(TransitionDistribution):
    """A lazy product of conditionally independent factor distributions."""

    def __init__(self, factors: Sequence[TransitionDistribution]) -> None:
        self._factors = tuple(factors)
        if not self._factors:
            raise ValueError("A product distribution needs at least one factor.")
        if any(
            not isinstance(factor, TransitionDistribution) for factor in self._factors
        ):
            raise TypeError("Product factors must be TransitionDistribution objects.")

        variables = []
        seen = set()
        for factor in self._factors:
            for variable in factor.variables:
                if variable.name in seen:
                    raise ValueError(
                        "Product factors overlap on variable {!r}.".format(
                            variable.name
                        )
                    )
                seen.add(variable.name)
                variables.append(variable)
        super().__init__(variables)

    @property
    def factors(self) -> Tuple[TransitionDistribution, ...]:
        return self._factors

    def _probability(self, outcome: Assignment) -> float:
        result = 1.0
        for factor in self.factors:
            names = (variable.name for variable in factor.variables)
            result *= factor.probability(outcome.project(names))
        return result

    def log_probability(self, outcome: Mapping[str, Hashable]) -> float:
        """Evaluate the product in log space without probability underflow."""

        canonical = _coerce_assignment(outcome, self.variables, "Outcome")
        result = 0.0
        for factor in self.factors:
            names = (variable.name for variable in factor.variables)
            local_log_probability = factor.log_probability(canonical.project(names))
            if local_log_probability == -math.inf:
                return -math.inf
            result += local_log_probability
        return result

    def _sample(self, rng: Any) -> Assignment:
        return Assignment.merge(*(factor.sample(rng) for factor in self.factors))

    def items(self) -> Tuple[Tuple[Assignment, float], ...]:
        factor_items = [factor.items() for factor in self.factors]
        weighted_outcomes = []
        for combination in product(*factor_items):
            outcomes, probabilities = zip(*combination)
            weighted_outcomes.append(
                (Assignment.merge(*outcomes), math.prod(probabilities))
            )
        return tuple(weighted_outcomes)


class AbstractMDP(ABC):
    """Abstract conditional transition model ``p(x' | x, pa(x))``."""

    def __init__(
        self,
        variables: Sequence[Variable],
        parent_variables: Sequence[Variable] = (),
    ) -> None:
        self._variables = _validate_variable_specs(
            variables, "MDP variables", allow_empty=False
        )
        self._parent_variables = _validate_variable_specs(
            parent_variables, "MDP parent variables", allow_empty=True
        )
        overlap = sorted(
            {variable.name for variable in self.variables}.intersection(
                variable.name for variable in self.parent_variables
            )
        )
        if overlap:
            raise ValueError(
                "An MDP's variables and parent variables must be disjoint; "
                "overlap: {}.".format(overlap)
            )

    @property
    def variables(self) -> Tuple[Variable, ...]:
        """The variables in ``x`` whose next values this model predicts."""

        return self._variables

    @property
    def parent_variables(self) -> Tuple[Variable, ...]:
        """The variables in the union of ``pa(x)``."""

        return self._parent_variables

    @property
    def parents(self) -> Tuple[Variable, ...]:
        """Concise alias for :attr:`parent_variables`."""

        return self.parent_variables

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Return subclass-specific fields for the human-readable description."""

        return {}

    def _to_description(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Build a JSON-safe description while guarding recursive models."""

        model_id = id(self)
        if model_id in active_ids:
            return {"type": type(self).__name__, "reference": "cycle"}

        active_ids.add(model_id)
        try:
            description = {
                "type": type(self).__name__,
                "variables": [
                    _variable_description(variable) for variable in self.variables
                ],
                "parent_variables": [
                    _variable_description(variable)
                    for variable in self.parent_variables
                ],
            }
            description.update(self._description_fields(active_ids))
            return description
        finally:
            active_ids.remove(model_id)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-safe model description used by :meth:`__str__`.

        The result is intended for inspection and logging rather than as a
        stable deserialization format.
        """

        return self._to_description(set())

    def __str__(self) -> str:
        """Return a deterministic, indented JSON representation of the MDP."""

        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    def to_networkx(self, view: str = "factor") -> Any:
        """Return a NetworkX two-slice graph of the transition structure.

        Install the optional visualization dependencies with
        ``pip install 'evolving-world-models[viz]'``.
        """

        from .visualization import to_networkx

        return to_networkx(self, view=view)

    def draw(
        self,
        *,
        ax: Optional[Any] = None,
        view: str = "factor",
        layout: Any = "layered",
        show_domains: bool = False,
        labels: Optional[Mapping[Any, str]] = None,
        **style: Any,
    ) -> Any:
        """Draw the transition structure and return a Matplotlib ``Axes``.

        The method never calls ``show()`` or writes a file.  Pass an existing
        axes to compose the graph with another figure.
        """

        from .visualization import draw_mdp

        return draw_mdp(
            self,
            ax=ax,
            view=view,
            layout=layout,
            show_domains=show_domains,
            labels=labels,
            **style,
        )

    def to_graphviz(
        self,
        *,
        view: str = "factor",
        rankdir: str = "LR",
    ) -> Any:
        """Return an optional Graphviz ``Digraph`` of the transition structure."""

        from .visualization import to_graphviz

        return to_graphviz(self, view=view, rankdir=rankdir)

    def transition_distribution(
        self,
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> TransitionDistribution:
        """Return ``p(x' | x=current, pa(x)=parents)``."""

        canonical_current = _coerce_assignment(current, self.variables, "Current state")
        canonical_parents = _coerce_assignment(
            parents, self.parent_variables, "Parent assignment"
        )
        distribution = self._transition_distribution(
            canonical_current, canonical_parents
        )
        if not isinstance(distribution, TransitionDistribution):
            raise TypeError("An MDP kernel must return a TransitionDistribution.")
        if distribution.variables != self.variables:
            raise ValueError(
                "Transition distribution scope {!r} does not match MDP scope {!r}.".format(
                    distribution.variables, self.variables
                )
            )
        return distribution

    @abstractmethod
    def _transition_distribution(
        self, current: Assignment, parents: Assignment
    ) -> TransitionDistribution:
        """Return a distribution for validated, canonical conditioning values."""

    def transition_probability(
        self,
        next_state: Mapping[str, Hashable],
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> float:
        """Evaluate one next-state probability."""

        return self.transition_distribution(current, parents).probability(next_state)

    def transition_log_probability(
        self,
        next_state: Mapping[str, Hashable],
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> float:
        """Evaluate one next-state log probability."""

        return self.transition_distribution(current, parents).log_probability(
            next_state
        )

    def sample_transition(
        self,
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
        rng: Optional[Any] = None,
    ) -> Assignment:
        """Sample ``x'`` from the transition model."""

        return self.transition_distribution(current, parents).sample(rng)

    def sample_next(
        self,
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
        rng: Optional[Any] = None,
    ) -> Assignment:
        """Alias for :meth:`sample_transition`."""

        return self.sample_transition(current, parents, rng)


class TabularMDP(AbstractMDP):
    """A fully specified finite transition table.

    Each transition row is a triple ``(current, parents, probabilities)``.
    ``probabilities`` is an iterable of ``(next_assignment, probability)``
    pairs.  Every finite current/parent context must have exactly one row.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        parent_variables: Sequence[Variable] = (),
        transitions: Iterable[
            Tuple[
                Mapping[str, Hashable],
                Mapping[str, Hashable],
                Iterable[Tuple[Mapping[str, Hashable], float]],
            ]
        ] = (),
    ) -> None:
        super().__init__(variables, parent_variables)
        variable_domains: List[Tuple[Hashable, ...]] = []
        parent_domains: List[Tuple[Hashable, ...]] = []
        for variable in self.variables:
            if variable.domain is None:
                raise ValueError(
                    "TabularMDP requires a finite domain for every variable."
                )
            variable_domains.append(variable.domain)
        for variable in self.parent_variables:
            if variable.domain is None:
                raise ValueError(
                    "TabularMDP requires a finite domain for every variable."
                )
            parent_domains.append(variable.domain)
        self._variable_domains = tuple(variable_domains)
        self._parent_domains = tuple(parent_domains)

        table: Dict[Tuple[Assignment, Assignment], CategoricalDistribution] = {}
        for row_number, row in enumerate(transitions, start=1):
            try:
                current, parents, probabilities = row
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Transition row {} must be a (current, parents, probabilities) "
                    "triple.".format(row_number)
                ) from error
            canonical_current = _coerce_assignment(
                current,
                self.variables,
                "Transition row {} current state".format(row_number),
            )
            canonical_parents = _coerce_assignment(
                parents,
                self.parent_variables,
                "Transition row {} parent assignment".format(row_number),
            )
            key = (canonical_current, canonical_parents)
            if key in table:
                raise ValueError(
                    "Duplicate transition row for current={!r}, parents={!r}.".format(
                        canonical_current, canonical_parents
                    )
                )
            table[key] = CategoricalDistribution(self.variables, probabilities)

        expected_contexts = set(self._all_contexts())
        actual_contexts = set(table)
        missing = expected_contexts - actual_contexts
        extra = actual_contexts - expected_contexts
        if missing or extra:
            message = (
                "Transition table must specify every conditioning context exactly once."
            )
            if missing:
                example_current, example_parents = next(iter(missing))
                message += " Missing {} context(s), for example current={!r}, parents={!r}.".format(
                    len(missing), example_current, example_parents
                )
            if extra:
                message += " Found {} unexpected context(s).".format(len(extra))
            raise ValueError(message)
        self._table = table

    def _all_contexts(self) -> Iterator[Tuple[Assignment, Assignment]]:
        for current_values in product(*self._variable_domains):
            current = Assignment(
                {
                    variable.name: value
                    for variable, value in zip(self.variables, current_values)
                }
            )
            for parent_values in product(*self._parent_domains):
                parents = Assignment(
                    {
                        variable.name: value
                        for variable, value in zip(self.parent_variables, parent_values)
                    }
                )
                yield current, parents

    def _transition_distribution(
        self, current: Assignment, parents: Assignment
    ) -> CategoricalDistribution:
        return self._table[(current, parents)]

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        transitions = []
        for current, parents in self._all_contexts():
            distribution = self._table[(current, parents)]
            outcomes = []
            for values in product(*self._variable_domains):
                next_state = Assignment(
                    {
                        variable.name: value
                        for variable, value in zip(self.variables, values)
                    }
                )
                outcomes.append(
                    {
                        "next_state": _assignment_description(next_state),
                        "probability": distribution.probability(next_state),
                    }
                )
            transitions.append(
                {
                    "current": _assignment_description(current),
                    "parents": _assignment_description(parents),
                    "outcomes": outcomes,
                }
            )
        return {"transitions": transitions}


class AbstractFactoredMDP(AbstractMDP, ABC):
    """Abstract composition whose joint transition is a product of factors."""

    @property
    @abstractmethod
    def factors(self) -> Tuple[AbstractMDP, ...]:
        """Constituent transition models with disjoint predicted scopes."""

    def _transition_distribution(
        self, current: Assignment, parents: Assignment
    ) -> ProductDistribution:
        local_distributions = []
        current_names = set(current)
        for factor in self.factors:
            local_current = current.project(
                variable.name for variable in factor.variables
            )
            local_parent_values = {}
            for variable in factor.parent_variables:
                source = current if variable.name in current_names else parents
                local_parent_values[variable.name] = source[variable.name]
            local_distributions.append(
                factor.transition_distribution(local_current, local_parent_values)
            )
        return ProductDistribution(local_distributions)

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        return {
            "factors": [factor._to_description(active_ids) for factor in self.factors]
        }


class FactoredMDP(AbstractFactoredMDP):
    """Concrete product composition of constituent MDP transition models.

    If a factor parent is predicted by another factor, its current-time value
    is read from the composite ``current`` assignment.  Remaining parents are
    exposed as the composite model's external ``parent_variables``.  All
    factors therefore update synchronously.
    """

    def __init__(self, factors: Sequence[AbstractMDP]) -> None:
        factor_tuple = tuple(factors)
        if not factor_tuple:
            raise ValueError("A FactoredMDP needs at least one factor.")
        if any(not isinstance(factor, AbstractMDP) for factor in factor_tuple):
            raise TypeError("FactoredMDP factors must be AbstractMDP instances.")

        predicted_variables = []
        predicted_by_name: Dict[str, Variable] = {}
        for factor in factor_tuple:
            for variable in factor.variables:
                if variable.name in predicted_by_name:
                    raise ValueError(
                        "Factor output scopes overlap on variable {!r}.".format(
                            variable.name
                        )
                    )
                predicted_by_name[variable.name] = variable
                predicted_variables.append(variable)

        specifications: Dict[str, Variable] = dict(predicted_by_name)
        for factor in factor_tuple:
            for variable in factor.parent_variables:
                if (
                    variable.name in specifications
                    and specifications[variable.name] != variable
                ):
                    raise ValueError(
                        "Variable {!r} has inconsistent specifications across factors.".format(
                            variable.name
                        )
                    )
                specifications[variable.name] = variable

        external_parents = []
        external_names = set()
        for factor in factor_tuple:
            for variable in factor.parent_variables:
                if variable.name in predicted_by_name:
                    continue
                if variable.name not in external_names:
                    external_names.add(variable.name)
                    external_parents.append(variable)

        self._factors = factor_tuple
        super().__init__(predicted_variables, external_parents)

    @property
    def factors(self) -> Tuple[AbstractMDP, ...]:
        return self._factors


__all__ = [
    "AbstractFactoredMDP",
    "AbstractMDP",
    "Assignment",
    "CategoricalDistribution",
    "FactoredMDP",
    "ProductDistribution",
    "TabularMDP",
    "TransitionDistribution",
    "Variable",
]
