"""Pre-train capacity guardrail — the impure orchestration around the pure engine (UC-23).

Where :mod:`drone_fly.train.health` is pure logic, this module does the side-effecting work
that logic can't: it counts the seeded actor's trainable parameters, logs the pre-train
verdict on **every** training start (AC4), and enforces the policy when the actor is
under-capacity (AC5):

* ``strict`` (opt-in ``--strict-capacity`` / ``strict_capacity: true``) → abort in either
  mode with a non-zero exit and guidance.
* interactive TTY → prompt the user to confirm; a decline aborts cleanly.
* non-interactive (no TTY / autonomous / CI) → warn-and-continue (cannot block on input).

A sufficiently-capable start (at/above the floor) never prompts and changes no behavior.

The abort path raises :class:`CapacityAbort`; the CLI maps it to a one-line message + exit
code 3 (config errors already own exit code 2).

Torch/SB3 are imported lazily inside the functions so ``from
drone_fly.train.capacity_guard import CapacityAbort`` stays cheap for the CLI's top-level
exception handling (importing the class must not drag in torch for e.g. ``drone-fly clean``).
"""

from __future__ import annotations

import logging
import sys

from drone_fly.train.health import (
    DEFAULT_THRESHOLDS,
    CapacityFacts,
    HealthThresholds,
    HealthVerdict,
    assess_training_health,
)

logger = logging.getLogger(__name__)


class CapacityAbort(RuntimeError):
    """Raised to abort a training start deemed under-capacity (strict or declined prompt).

    A clean, user-facing signal — the CLI turns it into a one-line message + exit code 3, no
    stack trace (mirroring the ``ConfigError`` → exit 2 convention).
    """


def count_actor_trainable_params(model) -> int:
    """Return the resolved actor's trainable-parameter count (the capacity signal).

    Reaches the :class:`~drone_fly.controller.actor.ConnectomeActorNetwork` via
    :func:`~drone_fly.controller.sb3.actor_from_model` and sums ``numel()`` over every
    ``requires_grad`` parameter (dominated by the sparse per-edge connectome weights).
    """
    from drone_fly.controller.sb3 import actor_from_model

    actor = actor_from_model(model)
    return int(sum(p.numel() for p in actor.parameters() if p.requires_grad))


def _capacity_facts(model) -> CapacityFacts:
    """Assemble :class:`CapacityFacts` from a built SB3 model (params + obs/action dims)."""
    from drone_fly.controller.sb3 import actor_from_model

    actor = actor_from_model(model)
    params = int(sum(p.numel() for p in actor.parameters() if p.requires_grad))

    def _dim(space) -> int:
        shape = getattr(space, "shape", None)
        return int(shape[0]) if shape else 0

    return CapacityFacts(
        trainable_params=params,
        obs_dim=_dim(getattr(model, "observation_space", None)),
        action_dim=_dim(getattr(model, "action_space", None)),
        neuron_count=getattr(actor, "n_neurons", None),
    )


def _is_interactive() -> bool:
    """True only when both stdin and stdout are TTYs (so a prompt won't deadlock/loop)."""
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (ValueError, OSError):  # pragma: no cover - closed streams are environment-specific
        return False


def enforce_capacity(
    model,
    *,
    strict: bool,
    thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
    is_interactive=None,
) -> HealthVerdict:
    """Run the pre-train capacity guardrail; return the verdict (AC4/AC5).

    Always logs the verdict line (the check runs on every start). When under-capacity:

    * ``strict`` → raise :class:`CapacityAbort` (both TTY and non-TTY modes).
    * interactive TTY → prompt to confirm; decline → raise :class:`CapacityAbort`.
    * non-interactive → warn-and-continue.

    At/above the floor: no prompt, no behavior change. ``is_interactive`` overrides TTY
    detection (for tests); ``None`` uses :func:`_is_interactive`.
    """
    facts = _capacity_facts(model)
    verdict = assess_training_health(capacity=facts, thresholds=thresholds)

    # AC4: surface the pre-train capacity verdict on every start, at minimum as a log line.
    log = logger.warning if verdict.status != "normal" else logger.info
    log("Pre-train capacity check: %s", verdict.message)

    if verdict.status == "normal":
        return verdict

    guidance = (
        "Under-capacity connectome slice: the seeded actor has too few trainable parameters "
        "to be expected to learn. Increase the slice size (a larger --prune-k / prune_k, or an "
        "unpruned connectome), or lower the capacity_floor if this is intentional."
    )

    if strict:
        raise CapacityAbort(guidance)

    interactive = _is_interactive() if is_interactive is None else bool(is_interactive)
    if interactive:
        try:
            answer = input("Actor may be under-capacity. Proceed with training anyway? [y/N] ")
        except EOFError:  # no input available despite the TTY probe — treat as decline
            raise CapacityAbort(guidance) from None
        if answer.strip().lower() not in ("y", "yes"):
            raise CapacityAbort("Training aborted by user (under-capacity actor not confirmed).")
        logger.warning("Proceeding with an under-capacity actor (user confirmed at the prompt).")
    else:
        logger.warning(
            "Non-interactive start with an under-capacity actor: warn-and-continue. Set "
            "strict_capacity (--strict-capacity / strict_capacity: true) to abort instead."
        )
    return verdict


__all__ = ["CapacityAbort", "count_actor_trainable_params", "enforce_capacity"]
