"""Conjugate Bayesian beliefs for complete finite transition tables.

This module is the epistemic-state layer above :mod:`scripts.mdp`.  The MDP
module owns fixed transition semantics; this module owns mutable observations
and uncertainty about those fixed kernels.  For every conditioning context
``c = (x, pa(x))`` and joint next-state outcome ``y``, a tabular belief uses

``theta_c ~ Dirichlet(alpha_c)``

and returns the posterior-predictive probability

``p(x' = y | c, D) = (alpha[c, y] + n[c, y]) / sum_z(alpha[c, z] + n[c, z])``.

The public entry points are :class:`AbstractTransitionBelief`,
:class:`TabularDirichletBelief`, and :class:`FactoredDirichletBelief`.  A
minimal symmetric-prior workflow is::

    state = Variable("state", (0, 1))
    action = Variable("action", ("stay", "flip"))
    belief = TabularDirichletBelief(
        variables=(state,),
        parent_variables=(action,),
        prior=1.0,
    )
    belief.update(
        current={"state": 0},
        next_state={"state": 1},
        parents={"action": "flip"},
    )
    probability = belief.transition_probability(
        next_state={"state": 1},
        current={"state": 0},
        parents={"action": "flip"},
    )

All variables must have finite, nonempty domains.  A multi-variable tabular
scope places one Dirichlet distribution over the full Cartesian joint
next-state support; it does not introduce unrequested within-factor
independence.  Explicit prior tables must completely cover every conditioning
context and every joint outcome.  Concentrations are finite, strictly positive
real values, while observations are retained separately as exact integer
counts.

Factored beliefs retain synchronous transition semantics.  Cross-factor
parents are projected from the current joint state, never from the observed
next state.  The factors' Dirichlet tables are conditionally independent, so a
joint posterior-predictive distribution is the product of the local
posterior-predictive distributions.

Updates are the module's only side effect and mutate only in-memory counts.
Imports, construction, prediction, inspection, and snapshot creation perform
no I/O, network access, plotting, or process-global configuration changes.
Sampling accepts an injected ``random.Random``-compatible generator; omitting
it creates a private generator rather than using module-global randomness.
Seeded reproducibility additionally depends on the Python implementation and
the declared factor, variable, domain, context, and outcome order.

Ordinary probabilities and fixed MDP snapshots use finite Python floats, so an
extremely small positive mass can underflow to ``0.0`` after normalization.
Posterior-predictive log probabilities are instead evaluated as
``log(alpha + n) - log(sum(alpha + n))`` and therefore preserve such positive
masses whenever the concentrations and their sum are finite.  A numeric zero
caused by float underflow is not a structural zero in the Dirichlet model.

The implementation uses only the Python standard library and the dependency-
free transition primitives in :mod:`scripts.mdp`.  For ``C`` contexts and
``K`` joint outcomes, tabular storage and full inspection use ``O(CK)`` space
and time, a predictive row uses ``O(K)`` time, and a validated update uses
``O(1)`` table access.  Sparse support, continuous variables, constrained joint
states, forgetting, hierarchical priors, rewards, planning, and allocation of
finite learning budgets are explicit non-goals for this milestone.
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable as IterableABC
from itertools import product
from numbers import Real
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .mdp import (
    AbstractMDP,
    Assignment,
    CategoricalDistribution,
    FactoredMDP,
    ProductDistribution,
    TabularMDP,
    TransitionDistribution,
    Variable,
    _assignment_description,
    _coerce_assignment,
    _validate_variable_specs,
    _variable_description,
)

_Context = Tuple[Assignment, Assignment]
_ConcentrationEntries = Iterable[Tuple[Mapping[str, Hashable], float]]
_ConcentrationRow = Tuple[
    Mapping[str, Hashable],
    Mapping[str, Hashable],
    _ConcentrationEntries,
]
_PriorSpecification = Union[float, Iterable[_ConcentrationRow]]
_PreparedUpdate = Tuple[_Context, Assignment]
_TransitionRow = Tuple[
    Mapping[str, Hashable],
    Mapping[str, Hashable],
    Iterable[Tuple[Mapping[str, Hashable], float]],
]


def _finite_domains(
    variables: Sequence[Variable],
    *,
    owner: str,
) -> Tuple[Tuple[Hashable, ...], ...]:
    """Return finite domains or reject a variable unsupported by this module.

    Args:
        variables: Validated variable specifications in observable order.
        owner: Human-readable owner used in an actionable error message.

    Returns:
        The variables' finite domains in the same order.

    Raises:
        ValueError: If any variable has no finite domain.
    """

    domains = []
    for variable in variables:
        if variable.domain is None:
            raise ValueError(
                "{} requires a finite domain for variable {!r}.".format(
                    owner, variable.name
                )
            )
        domains.append(variable.domain)
    return tuple(domains)


def _finite_positive_real(value: object, *, label: str) -> float:
    """Validate one scientifically meaningful Dirichlet concentration.

    Boolean and string values are intentionally not coerced even though Python
    can convert them to floats.  Silent coercion would make malformed prior
    specifications difficult to audit.

    Args:
        value: Candidate concentration.
        label: Location included in an error message.

    Returns:
        The concentration as a finite positive ``float``.

    Raises:
        TypeError: If ``value`` is not a non-boolean real number.
        ValueError: If conversion overflows or the result is non-finite or not
            strictly positive.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("{} must be a real number, not {!r}.".format(label, value))
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "{} must be finite and strictly positive.".format(label)
        ) from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("{} must be finite and strictly positive.".format(label))
    return numeric


def _finite_nonnegative_draw(value: object, *, label: str) -> float:
    """Validate a gamma draw supplied by an injected random generator.

    Individual zeros are admitted because valid, very small gamma variates can
    underflow to zero in finite precision.  The caller separately rejects an
    all-zero row.

    Args:
        value: Candidate result from ``rng.gammavariate``.
        label: Location included in an error message.

    Returns:
        The draw as a finite nonnegative ``float``.

    Raises:
        ValueError: If ``value`` is boolean, nonnumeric, negative, non-finite,
            or cannot be represented as a finite float.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            "{} must be a finite nonnegative real number; received {!r}.".format(
                label, value
            )
        )
    try:
        numeric = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "{} must be a finite nonnegative real number.".format(label)
        ) from error
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            "{} must be a finite nonnegative real number; received {!r}.".format(
                label, value
            )
        )
    return numeric


def _finite_fsum(values: Iterable[float], *, label: str) -> float:
    """Compute a finite floating-point sum with an actionable overflow error.

    Args:
        values: Finite terms to sum.
        label: Quantity named in an error message.

    Returns:
        The finite sum.

    Raises:
        ValueError: If summation overflows or produces a non-finite value.
    """

    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ValueError("{} must have a finite sum.".format(label)) from error
    if not math.isfinite(total):
        raise ValueError("{} must have a finite sum.".format(label))
    return total


class _DirichletPredictiveDistribution(CategoricalDistribution):
    """Categorical predictive row with concentration-space log probabilities.

    The inherited probability, sampling, and finite-enumeration interfaces use
    normalized floats, as required by the transition core.  This subclass
    retains log masses computed before normalization so a mathematically
    positive outcome does not become an impossible event merely because its
    ordinary probability underflows to ``0.0``.

    Args:
        variables: Finite predicted next-state scope.
        concentrations: Complete canonical outcomes paired with finite,
            strictly positive posterior concentrations.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        concentrations: Iterable[Tuple[Assignment, float]],
    ) -> None:
        entries = tuple(concentrations)
        total = _finite_fsum(
            (concentration for _, concentration in entries),
            label="Posterior concentration row",
        )
        if total <= 0.0:
            raise ValueError("Posterior concentration row must have positive mass.")

        super().__init__(
            variables,
            ((outcome, concentration / total) for outcome, concentration in entries),
        )
        log_total = math.log(total)
        self._log_probabilities = {
            outcome: math.log(concentration) - log_total
            for outcome, concentration in entries
        }

    def log_probability(self, outcome: Mapping[str, Hashable]) -> float:
        """Evaluate predictive log mass without normalized-float underflow."""

        canonical = _coerce_assignment(outcome, self.variables, "Outcome")
        return self._log_probabilities.get(canonical, -math.inf)


class AbstractTransitionBelief(ABC):
    """Mutable epistemic state over a finite conditional transition kernel.

    The query names intentionally match :class:`scripts.mdp.AbstractMDP`, but
    each returned distribution integrates over the current posterior rather
    than identifying one fixed environment hypothesis.  Use
    :meth:`posterior_mean_mdp` for a fixed mean kernel or :meth:`sample_mdp` for
    one coherent plausible kernel.

    Args:
        variables: Nonempty predicted next-state scope.  Every variable must
            have a finite domain.
        parent_variables: External conditioning scope, also with finite
            domains.  Its names must be disjoint from ``variables``.

    Raises:
        TypeError: If a scope contains a non-:class:`Variable` object.
        ValueError: If a scope is empty when prohibited, contains duplicate
            names, overlaps the other scope, or contains a non-finite variable.

    Notes:
        Subclasses receive canonical immutable :class:`Assignment` objects in
        their protected query and update methods.  Public boundary validation
        is completed before any subclass mutation begins.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        parent_variables: Sequence[Variable] = (),
    ) -> None:
        self._variables = _validate_variable_specs(
            variables, "Belief variables", allow_empty=False
        )
        self._parent_variables = _validate_variable_specs(
            parent_variables, "Belief parent variables", allow_empty=True
        )
        overlap = sorted(
            {variable.name for variable in self.variables}.intersection(
                variable.name for variable in self.parent_variables
            )
        )
        if overlap:
            raise ValueError(
                "A transition belief's variables and parent variables must be "
                "disjoint; overlap: {}.".format(overlap)
            )
        self._variable_domains = _finite_domains(
            self.variables, owner=type(self).__name__
        )
        self._parent_domains = _finite_domains(
            self.parent_variables, owner=type(self).__name__
        )

    @property
    def variables(self) -> Tuple[Variable, ...]:
        """The finite variables whose next values the belief predicts."""

        return self._variables

    @property
    def parent_variables(self) -> Tuple[Variable, ...]:
        """The finite external variables conditioning the transition rows."""

        return self._parent_variables

    @property
    def parents(self) -> Tuple[Variable, ...]:
        """Concise alias for :attr:`parent_variables`."""

        return self.parent_variables

    def transition_distribution(
        self,
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> TransitionDistribution:
        """Return the posterior-predictive distribution for one context.

        Args:
            current: Exact assignment to :attr:`variables` at the current time.
            parents: Exact assignment to :attr:`parent_variables`, or ``None``
                only when the parent scope is empty.

        Returns:
            A finite transition distribution over :attr:`variables`.

        Raises:
            TypeError: If an assignment is not a mapping or a subclass returns
                something other than a :class:`TransitionDistribution`.
            ValueError: If keys, values, or the returned distribution scope do
                not satisfy the belief contract.
        """

        canonical_current = _coerce_assignment(current, self.variables, "Current state")
        canonical_parents = _coerce_assignment(
            parents, self.parent_variables, "Parent assignment"
        )
        distribution = self._transition_distribution(
            canonical_current, canonical_parents
        )
        if not isinstance(distribution, TransitionDistribution):
            raise TypeError("A transition belief must return a TransitionDistribution.")
        if distribution.variables != self.variables:
            raise ValueError(
                "Posterior-predictive scope {!r} does not match belief scope "
                "{!r}.".format(distribution.variables, self.variables)
            )
        return distribution

    @abstractmethod
    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> TransitionDistribution:
        """Return a predictive distribution for canonical conditioning values."""

    def transition_probability(
        self,
        next_state: Mapping[str, Hashable],
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> float:
        """Evaluate one posterior-predictive transition probability.

        Args:
            next_state: Exact next assignment to :attr:`variables`.
            current: Exact current assignment to :attr:`variables`.
            parents: Exact assignment to :attr:`parent_variables`, if any.

        Returns:
            A finite probability in ``[0, 1]``.
        """

        return self.transition_distribution(current, parents).probability(next_state)

    def transition_log_probability(
        self,
        next_state: Mapping[str, Hashable],
        current: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> float:
        """Evaluate one posterior-predictive log probability.

        Args:
            next_state: Exact next assignment to :attr:`variables`.
            current: Exact current assignment to :attr:`variables`.
            parents: Exact assignment to :attr:`parent_variables`, if any.

        Returns:
            The natural logarithm of the predictive mass, or ``-inf`` for an
            impossible event.
        """

        return self.transition_distribution(current, parents).log_probability(
            next_state
        )

    def update(
        self,
        current: Mapping[str, Hashable],
        next_state: Mapping[str, Hashable],
        parents: Optional[Mapping[str, Hashable]] = None,
    ) -> None:
        """Condition the belief on one fully observed transition.

        Args:
            current: Exact assignment to :attr:`variables` at time ``t``.
            next_state: Exact assignment to :attr:`variables` at time ``t+1``.
            parents: Exact assignment to :attr:`parent_variables` at time
                ``t``, or ``None`` only when the parent scope is empty.

        Raises:
            TypeError: If an assignment is not a mapping.
            ValueError: If an assignment has missing or extra keys or an
                out-of-domain value.

        Notes:
            Validation of all public assignments finishes before
            :meth:`_update` can mutate counts.
        """

        canonical_current = _coerce_assignment(current, self.variables, "Current state")
        canonical_next = _coerce_assignment(next_state, self.variables, "Next state")
        canonical_parents = _coerce_assignment(
            parents, self.parent_variables, "Parent assignment"
        )
        self._update(canonical_current, canonical_next, canonical_parents)

    @abstractmethod
    def _update(
        self,
        current: Assignment,
        next_state: Assignment,
        parents: Assignment,
    ) -> None:
        """Apply one fully validated observation."""

    @abstractmethod
    def posterior_mean_mdp(self) -> AbstractMDP:
        """Return a fixed MDP snapshot of the current posterior mean."""

    @abstractmethod
    def sample_mdp(self, rng: Optional[Any] = None) -> AbstractMDP:
        """Sample one fixed plausible MDP using an injectable gamma RNG."""

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Return subclass fields for the deterministic inspection description."""

        return {}

    def _to_description(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Build a JSON-safe description while guarding recursive references."""

        belief_id = id(self)
        if belief_id in active_ids:
            return {"type": type(self).__name__, "reference": "cycle"}

        active_ids.add(belief_id)
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
            active_ids.remove(belief_id)

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-safe inspection representation.

        Returns:
            A new nested dictionary describing scopes, priors, counts, and
            subclass structure.

        Notes:
            The representation is intended for inspection and logging, not as
            a stable persistence or interchange schema.
        """

        return self._to_description(set())

    def __str__(self) -> str:
        """Return the deterministic inspection representation as indented JSON."""

        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


class TabularDirichletBelief(AbstractTransitionBelief):
    """Independent Dirichlet posteriors for complete finite transition rows.

    Args:
        variables: Nonempty finite scope predicted by each row.
        parent_variables: Finite external conditioning scope.
        prior: Either one finite positive real concentration applied
            symmetrically to every outcome in every context, or an iterable of
            complete ``(current, parents, concentrations)`` rows.

    Raises:
        TypeError: If the prior schema, a concentration, or a variable
            specification has the wrong type.
        ValueError: If scopes are invalid, domains are not finite, prior rows
            are malformed, coverage is incomplete or duplicated, or a
            concentration or row sum is invalid.

    Notes:
        The table stores immutable prior floats and mutable integer counts
        separately.  Input iteration order never affects predictive support,
        sampling order, or descriptions.
    """

    def __init__(
        self,
        variables: Sequence[Variable],
        parent_variables: Sequence[Variable] = (),
        prior: _PriorSpecification = 1.0,
    ) -> None:
        super().__init__(variables, parent_variables)
        self._contexts = tuple(self._all_contexts())
        self._outcomes = tuple(self._all_outcomes())
        self._prior_concentrations = self._build_prior(prior)
        self._counts: Dict[_Context, Dict[Assignment, int]] = {
            context: {outcome: 0 for outcome in self._outcomes}
            for context in self._contexts
        }

    def _all_contexts(self) -> Iterator[_Context]:
        """Yield every conditioning context in canonical Cartesian order."""

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

    def _all_outcomes(self) -> Iterator[Assignment]:
        """Yield every joint next-state outcome in canonical Cartesian order."""

        for values in product(*self._variable_domains):
            yield Assignment(
                {
                    variable.name: value
                    for variable, value in zip(self.variables, values)
                }
            )

    def _build_prior(
        self,
        prior: object,
    ) -> Dict[_Context, Dict[Assignment, float]]:
        """Validate and canonicalize a scalar or complete explicit prior.

        Args:
            prior: Symmetric scalar concentration or complete concentration
                rows.

        Returns:
            A complete table ordered by canonical contexts and outcomes.

        Raises:
            TypeError: If ``prior`` or one concentration has the wrong type.
            ValueError: If a value is invalid or explicit coverage is not
                exactly complete.
        """

        if isinstance(prior, bool):
            raise TypeError("Prior must be a real scalar or concentration rows.")
        if isinstance(prior, Real):
            concentration = _finite_positive_real(
                prior, label="Symmetric prior concentration"
            )
            _finite_fsum(
                (concentration for _ in self._outcomes),
                label="Symmetric prior row",
            )
            return {
                context: {outcome: concentration for outcome in self._outcomes}
                for context in self._contexts
            }
        if isinstance(prior, (str, bytes, bytearray)):
            raise TypeError("Prior must be a real scalar or concentration rows.")
        if not isinstance(prior, IterableABC):
            raise TypeError(
                "Prior must be a real scalar or iterable concentration rows."
            )

        try:
            rows = iter(prior)
        except TypeError as error:
            raise TypeError(
                "Prior must be a real scalar or iterable concentration rows."
            ) from error

        supplied: Dict[_Context, Dict[Assignment, float]] = {}
        for row_number, row in enumerate(rows, start=1):
            try:
                current, parents, entries = row
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Prior row {} must be a (current, parents, concentrations) "
                    "triple.".format(row_number)
                ) from error

            canonical_current = _coerce_assignment(
                current,
                self.variables,
                "Prior row {} current state".format(row_number),
            )
            canonical_parents = _coerce_assignment(
                parents,
                self.parent_variables,
                "Prior row {} parent assignment".format(row_number),
            )
            context = (canonical_current, canonical_parents)
            if context in supplied:
                raise ValueError(
                    "Duplicate prior row for current={!r}, parents={!r}.".format(
                        canonical_current, canonical_parents
                    )
                )

            canonical_entries: Dict[Assignment, float] = {}
            try:
                entry_iterator = iter(entries)
            except TypeError as error:
                raise ValueError(
                    "Prior row {} concentrations must be an iterable of "
                    "(next_state, concentration) pairs.".format(row_number)
                ) from error
            for entry_number, entry in enumerate(entry_iterator, start=1):
                try:
                    next_state, concentration_value = entry
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Prior row {} concentration {} must be a "
                        "(next_state, concentration) pair.".format(
                            row_number, entry_number
                        )
                    ) from error
                canonical_next = _coerce_assignment(
                    next_state,
                    self.variables,
                    "Prior row {} outcome {}".format(row_number, entry_number),
                )
                if canonical_next in canonical_entries:
                    raise ValueError(
                        "Prior row {} contains duplicate outcome {!r}.".format(
                            row_number, canonical_next
                        )
                    )
                canonical_entries[canonical_next] = _finite_positive_real(
                    concentration_value,
                    label="Prior row {} outcome {} concentration".format(
                        row_number, entry_number
                    ),
                )

            missing_outcomes = [
                outcome
                for outcome in self._outcomes
                if outcome not in canonical_entries
            ]
            if missing_outcomes:
                raise ValueError(
                    "Prior row {} must specify every joint next-state outcome "
                    "exactly once; missing {} outcome(s), for example {!r}.".format(
                        row_number, len(missing_outcomes), missing_outcomes[0]
                    )
                )
            _finite_fsum(
                (canonical_entries[outcome] for outcome in self._outcomes),
                label="Prior row {} concentrations".format(row_number),
            )
            supplied[context] = canonical_entries

        missing_contexts = [
            context for context in self._contexts if context not in supplied
        ]
        if missing_contexts:
            example_current, example_parents = missing_contexts[0]
            raise ValueError(
                "Prior table must specify every conditioning context exactly "
                "once; missing {} context(s), for example current={!r}, "
                "parents={!r}.".format(
                    len(missing_contexts), example_current, example_parents
                )
            )

        return {
            context: {outcome: supplied[context][outcome] for outcome in self._outcomes}
            for context in self._contexts
        }

    @property
    def prior_concentrations(
        self,
    ) -> Dict[_Context, Dict[Assignment, float]]:
        """Return a defensive copy of the complete prior-concentration table."""

        return {
            context: dict(self._prior_concentrations[context])
            for context in self._contexts
        }

    @property
    def counts(self) -> Dict[_Context, Dict[Assignment, int]]:
        """Return a defensive copy of the complete integer observation table."""

        return {context: dict(self._counts[context]) for context in self._contexts}

    @property
    def posterior_concentrations(
        self,
    ) -> Dict[_Context, Dict[Assignment, float]]:
        """Return prior-plus-count concentrations as a defensive complete copy."""

        return {
            context: {
                outcome: self._posterior_concentration(context, outcome)
                for outcome in self._outcomes
            }
            for context in self._contexts
        }

    def _posterior_concentration(
        self,
        context: _Context,
        outcome: Assignment,
    ) -> float:
        """Return one finite posterior concentration or report overflow."""

        value = (
            self._prior_concentrations[context][outcome]
            + self._counts[context][outcome]
        )
        if not math.isfinite(value):
            raise ValueError(
                "Posterior concentration must remain finite for context={!r}, "
                "outcome={!r}.".format(context, outcome)
            )
        return value

    def _posterior_row(
        self,
        context: _Context,
    ) -> Tuple[Tuple[Assignment, float], ...]:
        """Return one normalized posterior-predictive row in stable order."""

        concentrations = tuple(
            self._posterior_concentration(context, outcome)
            for outcome in self._outcomes
        )
        total = _finite_fsum(concentrations, label="Posterior concentration row")
        if total <= 0.0:
            raise ValueError("Posterior concentration row must have positive mass.")
        return tuple(
            (outcome, concentration / total)
            for outcome, concentration in zip(self._outcomes, concentrations)
        )

    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> _DirichletPredictiveDistribution:
        """Return a predictive row with concentration-space log masses."""

        context = (current, parents)
        return _DirichletPredictiveDistribution(
            self.variables,
            (
                (outcome, self._posterior_concentration(context, outcome))
                for outcome in self._outcomes
            ),
        )

    def _prepare_update(
        self,
        current: Assignment,
        next_state: Assignment,
        parents: Assignment,
    ) -> _PreparedUpdate:
        """Resolve a validated observation to its exact mutable count cell.

        Args:
            current: Canonical current local assignment.
            next_state: Canonical next local assignment.
            parents: Canonical local parent assignment.

        Returns:
            The context and outcome keys used by :meth:`_apply_prepared_update`.

        Raises:
            ValueError: If the supplied canonical-looking assignments are not a
                row and outcome owned by this belief.
        """

        context = (current, parents)
        if context not in self._counts:
            raise ValueError(
                "Observation does not identify a conditioning context in this "
                "belief: current={!r}, parents={!r}.".format(current, parents)
            )
        if next_state not in self._counts[context]:
            raise ValueError(
                "Observation next state {!r} is outside this belief's support.".format(
                    next_state
                )
            )
        return context, next_state

    def _apply_prepared_update(self, target: _PreparedUpdate) -> None:
        """Increment one previously validated count cell exactly once."""

        context, next_state = target
        self._counts[context][next_state] += 1

    def _update(
        self,
        current: Assignment,
        next_state: Assignment,
        parents: Assignment,
    ) -> None:
        """Apply one canonical tabular observation atomically."""

        target = self._prepare_update(current, next_state, parents)
        self._apply_prepared_update(target)

    def posterior_mean_mdp(self) -> TabularMDP:
        """Return an independent fixed snapshot of the posterior mean.

        Returns:
            A newly constructed complete :class:`TabularMDP`.  Future updates
            to this belief cannot alter its probabilities.
        """

        rows: List[_TransitionRow] = []
        for current, parents in self._contexts:
            rows.append(
                (
                    current,
                    parents,
                    self._posterior_row((current, parents)),
                )
            )
        return TabularMDP(
            self.variables,
            parent_variables=self.parent_variables,
            transitions=rows,
        )

    def sample_mdp(self, rng: Optional[Any] = None) -> TabularMDP:
        """Sample an independent fixed MDP from the current posterior.

        Args:
            rng: A ``random.Random``-compatible object exposing callable
                ``gammavariate(shape, scale)``.  ``None`` creates a private
                generator.

        Returns:
            A new complete :class:`TabularMDP` whose rows are independent
            Dirichlet draws.

        Raises:
            TypeError: If ``rng`` does not expose ``gammavariate``.
            ValueError: If a custom RNG returns an invalid draw or a sampled
                row has zero or non-finite total mass.
        """

        generator = random.Random() if rng is None else rng
        gamma = getattr(generator, "gammavariate", None)
        if not callable(gamma):
            raise TypeError(
                "rng must expose a callable gammavariate(shape, scale) method."
            )

        rows: List[_TransitionRow] = []
        for current, parents in self._contexts:
            context = (current, parents)
            draws = []
            for outcome in self._outcomes:
                shape = self._posterior_concentration(context, outcome)
                draw = _finite_nonnegative_draw(
                    gamma(shape, 1.0),
                    label="Gamma draw for context={!r}, outcome={!r}".format(
                        context, outcome
                    ),
                )
                draws.append(draw)
            total = _finite_fsum(draws, label="Sampled Dirichlet row")
            if total <= 0.0:
                raise ValueError(
                    "Sampled Dirichlet row has zero total mass for "
                    "current={!r}, parents={!r}.".format(current, parents)
                )
            probabilities = tuple(
                (outcome, draw / total) for outcome, draw in zip(self._outcomes, draws)
            )
            rows.append((current, parents, probabilities))

        return TabularMDP(
            self.variables,
            parent_variables=self.parent_variables,
            transitions=rows,
        )

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Describe priors and observations in canonical table order."""

        contexts = []
        for current, parents in self._contexts:
            context = (current, parents)
            outcomes = []
            for next_state in self._outcomes:
                prior = self._prior_concentrations[context][next_state]
                count = self._counts[context][next_state]
                outcomes.append(
                    {
                        "next_state": _assignment_description(next_state),
                        "prior_concentration": prior,
                        "count": count,
                        "posterior_concentration": (
                            self._posterior_concentration(context, next_state)
                        ),
                    }
                )
            contexts.append(
                {
                    "current": _assignment_description(current),
                    "parents": _assignment_description(parents),
                    "outcomes": outcomes,
                }
            )
        return {"contexts": contexts}


class FactoredDirichletBelief(AbstractTransitionBelief):
    """Product of retained mutable tabular Dirichlet factor beliefs.

    Args:
        factors: Nonempty tabular beliefs with disjoint predicted scopes and
            consistent specifications for every repeated variable name.

    Raises:
        TypeError: If any factor is not a
            :class:`TabularDirichletBelief`.
        ValueError: If factors are empty, output scopes overlap, or repeated
            variable specifications conflict.

    Notes:
        The factor tuple itself is immutable, but it retains the caller's
        mutable factor-belief objects.  Updating either a retained factor or
        this composite therefore changes the shared epistemic state.

        A factor parent predicted by another factor is resolved from the
        composite current state.  Only non-predicted parents are exposed
        through :attr:`parent_variables`.
    """

    def __init__(
        self,
        factors: Sequence[TabularDirichletBelief],
    ) -> None:
        factor_tuple = tuple(factors)
        if not factor_tuple:
            raise ValueError("A FactoredDirichletBelief needs at least one factor.")
        if any(
            not isinstance(factor, TabularDirichletBelief) for factor in factor_tuple
        ):
            raise TypeError(
                "FactoredDirichletBelief factors must be "
                "TabularDirichletBelief instances."
            )

        predicted_variables = []
        predicted_by_name: Dict[str, Variable] = {}
        for factor in factor_tuple:
            for variable in factor.variables:
                if variable.name in predicted_by_name:
                    raise ValueError(
                        "Factor belief output scopes overlap on variable {!r}.".format(
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
                        "Variable {!r} has inconsistent specifications across "
                        "factor beliefs.".format(variable.name)
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
    def factors(self) -> Tuple[TabularDirichletBelief, ...]:
        """The retained tabular factor beliefs in deterministic product order."""

        return self._factors

    def _local_context(
        self,
        factor: TabularDirichletBelief,
        current: Assignment,
        parents: Assignment,
    ) -> Tuple[Assignment, Assignment]:
        """Project one factor's current state and synchronous parent context.

        Args:
            factor: Factor whose context is required.
            current: Canonical composite current state.
            parents: Canonical composite external-parent assignment.

        Returns:
            Canonical local current and parent assignments.
        """

        local_current = current.project(variable.name for variable in factor.variables)
        current_names = set(current)
        local_parent_values: Dict[str, Hashable] = {}
        for variable in factor.parent_variables:
            source = current if variable.name in current_names else parents
            local_parent_values[variable.name] = source[variable.name]
        local_parents = _coerce_assignment(
            local_parent_values,
            factor.parent_variables,
            "Local factor parent assignment",
        )
        return local_current, local_parents

    def _transition_distribution(
        self,
        current: Assignment,
        parents: Assignment,
    ) -> ProductDistribution:
        """Return the product of local posterior-predictive rows."""

        local_distributions = []
        for factor in self.factors:
            local_current, local_parents = self._local_context(factor, current, parents)
            local_distributions.append(
                factor.transition_distribution(local_current, local_parents)
            )
        return ProductDistribution(local_distributions)

    def _update(
        self,
        current: Assignment,
        next_state: Assignment,
        parents: Assignment,
    ) -> None:
        """Prepare every local target before atomically incrementing factors."""

        prepared: List[Tuple[TabularDirichletBelief, _PreparedUpdate]] = []
        for factor in self.factors:
            local_current, local_parents = self._local_context(factor, current, parents)
            local_next = next_state.project(
                variable.name for variable in factor.variables
            )
            canonical_next = _coerce_assignment(
                local_next,
                factor.variables,
                "Local factor next state",
            )
            target = factor._prepare_update(
                local_current, canonical_next, local_parents
            )
            prepared.append((factor, target))

        for factor, target in prepared:
            factor._apply_prepared_update(target)

    def posterior_mean_mdp(self) -> FactoredMDP:
        """Return a fresh factored snapshot of every posterior mean.

        Returns:
            A :class:`FactoredMDP` containing newly constructed local
            :class:`TabularMDP` snapshots.
        """

        return FactoredMDP(
            tuple(factor.posterior_mean_mdp() for factor in self.factors)
        )

    def sample_mdp(self, rng: Optional[Any] = None) -> FactoredMDP:
        """Sample a fresh factored MDP in stable factor order.

        Args:
            rng: One ``random.Random``-compatible gamma generator shared across
                every factor.  ``None`` creates one private generator for the
                entire composite sample.

        Returns:
            A :class:`FactoredMDP` containing independent local posterior
            samples.
        """

        generator = random.Random() if rng is None else rng
        return FactoredMDP(
            tuple(factor.sample_mdp(generator) for factor in self.factors)
        )

    def _description_fields(self, active_ids: Set[int]) -> Dict[str, Any]:
        """Recursively describe retained factors in deterministic order."""

        return {
            "factors": [factor._to_description(active_ids) for factor in self.factors]
        }


__all__ = [
    "AbstractTransitionBelief",
    "FactoredDirichletBelief",
    "TabularDirichletBelief",
]
