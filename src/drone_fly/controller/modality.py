"""Modality → neuron-population selection over a connectome / slice (UC-13, AC2/AC6).

UC-02's :mod:`drone_fly.controller.populations` selects populations by the coarse
``superclass`` bucket (``visual_projection`` sensory, ``descending_neuron`` motor) and,
critically, **degrades to a placeholder** when the metadata is missing rather than raising.
That degrade is the right call for the flight I/O contract, but it is the *wrong* call for
biologically-bound observations: if you ask to bind a proprioceptive block and the
proprioceptive neurons are simply not in the active slice, silently binding to a placeholder
(or to nothing) hides a modelling error.

This module is the **distinct, fail-loud** path. Given a :class:`ConnectomeData` (full or
pruned/sliced — indices are returned into *that object's* row order, so a caller can select
on whatever graph the actor is actually built from) and a modality name, it returns the
neuron indices of that modality's biological population and RAISES when the population is
absent (:class:`ModalityAbsentError`) or the required metadata column is missing
(:class:`ModalityMetadataError`). It never returns a placeholder and never returns an empty
selection (AC6).

Modality taxonomy (grounded in the MaleCNS labelling research — see the use case)
--------------------------------------------------------------------------------
*Cleanly labelled* modalities map to an exact ``superclass`` or ``class`` label (or a label
prefix) and are marked ``approximate=False``:

* ``vision`` — ``superclass == "visual_projection"`` (the visual afferents into the brain,
  the same sensory population UC-02/UC-04 already use).
* ``proprioceptive`` — ``class == "mechanosensory_proprioceptive"``.
* ``mechanosensory`` — every ``class`` starting ``"mechanosensory"`` (tactile, proprioceptive,
  bristle, ...); the superset of the proprioceptive class.
* ``gustatory`` / ``olfactory`` / ``thermosensory`` / ``hygrosensory`` — the matching exact
  ``class`` label.

*Approximate* modalities have no clean connectome label; they are resolved by a **curated,
best-effort substring match over a fine-grained name label** (``subclass`` or ``cell_type``,
per rule) and are marked ``approximate=True`` so a caller can surface the caveat. They are
exposed (per the use case) but must never be mistaken for authoritative populations:

* ``motion`` — motion-detection-associated ``subclass`` names (e.g. the ``T4``/``T5`` elementary
  motion detectors).
* ``hunger`` — internal-state / feeding-associated ``cell_type`` names (IPCs, Hugin, NPF, and
  insulin/DILP peptidergic cells; UC-17). Bound over ``cell_type`` — where these populations are
  actually labelled — rather than ``subclass`` (where they do not appear at all).

Both approximate modalities still fail loud on zero match, exactly like the clean ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_fly.connectome.loader import ConnectomeData


class ModalityError(ValueError):
    """Base class for modality-selection failures (a :class:`ValueError` subclass)."""


class UnknownModalityError(ModalityError):
    """Raised when a requested modality name is not in :data:`MODALITY_RULES`."""


class ModalityMetadataError(ModalityError):
    """Raised when the connectome lacks the metadata column a modality needs.

    Distinct from :class:`ModalityAbsentError`: here the label column itself is ``None``
    (an older/reduced fixture), so we cannot even ask whether the population is present.
    """


class ModalityAbsentError(ModalityError):
    """Raised when the metadata is present but zero neurons match the modality (AC6).

    This is the load-bearing fail-loud case: e.g. proprioceptive neurons that were dropped
    by a vision→motor prune, so binding a proprioceptive block to the pruned slice must fail
    rather than bind to nothing.
    """


#: The three ``match`` strategies a rule may use over its label column.
_EXACT = "exact"  # label equals any of ``values``
_PREFIX = "prefix"  # label starts with any of ``values``
_SUBSTRING = "substring"  # any of ``values`` is a case-insensitive substring of the label


@dataclass(frozen=True)
class _ModalityRule:
    """How one modality resolves to neuron indices.

    Attributes
    ----------
    attr:
        Which :class:`ConnectomeData` label attribute to read (``"superclass"``,
        ``"neuron_class"`` or ``"subclass"``).
    match:
        One of :data:`_EXACT`, :data:`_PREFIX`, :data:`_SUBSTRING`.
    values:
        The label(s) / prefix(es) / substring(s) to match.
    approximate:
        ``True`` for curated-name-list modalities (``motion`` / ``hunger``) whose binding
        is heuristic, not authoritative.
    description:
        Human-readable rule text, echoed on the returned :class:`ModalitySelection` and in
        errors so a caller can see exactly what was matched.
    """

    attr: str
    match: str
    values: tuple[str, ...]
    approximate: bool
    description: str


@dataclass(frozen=True)
class ModalitySelection:
    """The result of :func:`select_modality`.

    Attributes
    ----------
    modality:
        The requested modality name.
    indices:
        1-D ``int64`` array of neuron indices, sorted ascending, **into the row order of the
        connectome passed to :func:`select_modality`** (works identically on a full or a
        sliced/pruned graph). Never empty.
    approximate:
        Whether the binding is heuristic (curated name list) rather than a clean label.
    rule:
        The human-readable rule that produced the selection.
    """

    modality: str
    indices: np.ndarray
    approximate: bool
    rule: str


#: The modality registry. Keys are the modality names callers pass to :func:`select_modality`;
#: values describe how each resolves. Label strings are VERIFIED against the canonical MaleCNS
#: meta (``class``/``superclass``/``subclass`` columns).
MODALITY_RULES: dict[str, _ModalityRule] = {
    "vision": _ModalityRule(
        attr="superclass",
        match=_EXACT,
        values=("visual_projection",),
        approximate=False,
        description="superclass == 'visual_projection'",
    ),
    "proprioceptive": _ModalityRule(
        attr="neuron_class",
        match=_EXACT,
        values=("mechanosensory_proprioceptive",),
        approximate=False,
        description="class == 'mechanosensory_proprioceptive'",
    ),
    "mechanosensory": _ModalityRule(
        attr="neuron_class",
        match=_PREFIX,
        values=("mechanosensory",),
        approximate=False,
        description="class starts with 'mechanosensory'",
    ),
    "gustatory": _ModalityRule(
        attr="neuron_class",
        match=_EXACT,
        values=("gustatory",),
        approximate=False,
        description="class == 'gustatory'",
    ),
    "olfactory": _ModalityRule(
        attr="neuron_class",
        match=_EXACT,
        values=("olfactory",),
        approximate=False,
        description="class == 'olfactory'",
    ),
    "thermosensory": _ModalityRule(
        attr="neuron_class",
        match=_EXACT,
        values=("thermosensory",),
        approximate=False,
        description="class == 'thermosensory'",
    ),
    "hygrosensory": _ModalityRule(
        attr="neuron_class",
        match=_EXACT,
        values=("hygrosensory",),
        approximate=False,
        description="class == 'hygrosensory'",
    ),
    # --- Approximate (curated-name-list) modalities — flagged, never authoritative --------
    "motion": _ModalityRule(
        attr="subclass",
        match=_SUBSTRING,
        values=("t4", "t5"),
        approximate=True,
        description="curated name list (approximate): subclass contains one of ['t4', 't5']",
    ),
    "hunger": _ModalityRule(
        # UC-17: repointed from ``subclass`` to ``cell_type``. The internal-state / feeding
        # ("hunger") neurons (IPCs, Hugin, NPF, and insulin/DILP peptidergic cells) are labelled
        # in the authoritative ``cell_type`` column, NOT in ``subclass`` — the old ``subclass``
        # tokens matched 0 neurons in the full MaleCNS meta, making the binding unbuildable. The
        # ``cell_type`` substring match resolves the real approximate population (~22 in the
        # canonical matrix: {IPC, Hugin-RG, NPFL1-I}); ``insulin``/``dilp`` are future-proofing
        # synonyms (0 matches today, harmless). Still approximate / non-authoritative.
        attr="cell_type",
        match=_SUBSTRING,
        values=("ipc", "hugin", "npf", "insulin", "dilp"),
        approximate=True,
        description=(
            "curated name list (approximate): cell_type contains one of "
            "['ipc', 'hugin', 'npf', 'insulin', 'dilp']"
        ),
    ),
}

#: Human-friendly, sorted attribute-name map so error messages can point at the CSV column.
_ATTR_TO_COLUMN = {
    "superclass": "superclass",
    "neuron_class": "class",
    "subclass": "subclass",
    "cell_type": "cell_type",
}


def available_modalities() -> tuple[str, ...]:
    """Return the registered modality names, sorted (for menus / error messages)."""
    return tuple(sorted(MODALITY_RULES))


def _match_mask(labels: np.ndarray, rule: _ModalityRule) -> np.ndarray:
    """Boolean mask over ``labels`` for ``rule`` (NaN/None labels never match)."""
    text = np.array([("" if v is None else str(v)) for v in labels.tolist()], dtype=object)
    if rule.match == _EXACT:
        wanted = set(rule.values)
        return np.array([t in wanted for t in text], dtype=bool)
    if rule.match == _PREFIX:
        return np.array([any(t.startswith(v) for v in rule.values) for t in text], dtype=bool)
    if rule.match == _SUBSTRING:
        lowered = [t.lower() for t in text]
        return np.array([any(v in t for v in rule.values) for t in lowered], dtype=bool)
    raise AssertionError(f"unhandled match strategy {rule.match!r}")  # pragma: no cover


def select_modality(data: ConnectomeData, modality: str) -> ModalitySelection:
    """Resolve ``modality`` to its neuron indices in ``data`` (fail-loud) (AC2/AC6).

    Parameters
    ----------
    data:
        The connectome to select within. Indices are returned into *this* object's row
        order, so passing a pruned/sliced graph selects within that slice.
    modality:
        A registered modality name (see :func:`available_modalities`).

    Returns
    -------
    ModalitySelection
        The matched neuron indices (sorted, non-empty) plus the ``approximate`` flag.

    Raises
    ------
    UnknownModalityError
        If ``modality`` is not registered.
    ModalityMetadataError
        If the label column the modality needs is absent from ``data``.
    ModalityAbsentError
        If the column is present but no neuron matches (e.g. the population was pruned out).
    """
    rule = MODALITY_RULES.get(modality)
    if rule is None:
        raise UnknownModalityError(
            f"Unknown modality {modality!r}. Available modalities: "
            f"{', '.join(available_modalities())}."
        )

    labels = getattr(data, rule.attr)
    if labels is None:
        column = _ATTR_TO_COLUMN.get(rule.attr, rule.attr)
        raise ModalityMetadataError(
            f"Cannot select modality {modality!r}: the connectome lacks the '{column}' "
            f"metadata column ({rule.attr} is None). Provision a connectome whose "
            f"*_meta.csv carries a '{column}' column."
        )

    labels = np.asarray(labels)
    mask = _match_mask(labels, rule)
    indices = np.nonzero(mask)[0].astype(np.int64)
    if indices.size == 0:
        raise ModalityAbsentError(
            f"Modality {modality!r} has no neurons in this connectome "
            f"({data.neuron_count} neurons; rule: {rule.description}). The population is "
            f"absent from the active slice — bind it on a graph that contains it, or select "
            f"a modality that is present. (No silent bind-to-nothing; AC6.)"
        )
    return ModalitySelection(
        modality=modality,
        indices=np.sort(indices),
        approximate=rule.approximate,
        rule=rule.description,
    )
