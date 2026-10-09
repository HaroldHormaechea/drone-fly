"""Hard ``log_std`` annealing callback (research toggle — default off).

Motivation (acro deterministic-mean collapse). The connectome actor is a *linear* readout
``action = bias + W·features`` with an **unbounded diagonal-Gaussian** head whose ``log_std`` is a
free optimization parameter. On the acro stabilization task the *stochastic* policy flies the gate
course (~1/25) but the *deterministic* mean does not: the sampling noise supplies the rapidly-varying
corrective control the plant needs, so the policy is never pressured to migrate that corrective
authority into the mean weights ``W`` — reducing ``std`` *hurts* return as long as the noise is
load-bearing, so the gradient keeps ``std`` high on its own. Lowering ``ent_coef`` (tried to 5.8M)
does not fix this: it only removes the pressure keeping ``std`` up, it does not force it down.

This callback **forces** ``log_std`` down on a schedule and freezes it (``requires_grad=False``) so
the optimizer can no longer raise it. With the noise budget shrinking, the only way to retain reward
is to strengthen the deterministic feedback law ``W`` — exactly the collapse we want to break.

Opt-in via env var ``DRONE_FLY_LOGSTD_ANNEAL="<target_log_std>:<fraction>"``:
- ``target_log_std`` — the floor to drive every action dim's ``log_std`` to (e.g. ``-2.3`` ⇒ std≈0.1).
- ``fraction``      — fraction of this run's step budget over which to anneal linearly from the
  value the policy starts this run with to ``target_log_std``; held at the floor afterwards.

Unset ⇒ the callback is never constructed and training is byte-identical to before.
"""

from __future__ import annotations

import logging
import os

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

ENV_VAR = "DRONE_FLY_LOGSTD_ANNEAL"


def parse_spec(spec: str | None) -> tuple[float, float] | None:
    """Parse ``"<target>:<fraction>"`` → ``(target_log_std, fraction)`` or ``None`` when unset.

    Raises ``ValueError`` on a malformed spec so a typo fails loud rather than silently disabling
    the experiment.
    """
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"{ENV_VAR} must be '<target_log_std>:<fraction>', got {spec!r}."
        )
    target = float(parts[0])
    fraction = float(parts[1])
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"{ENV_VAR} fraction must be in (0, 1], got {fraction!r}.")
    return target, fraction


class LogStdAnnealCallback(BaseCallback):
    """Linearly drive ``policy.log_std`` to ``target`` over ``fraction`` of ``total_steps``.

    ``total_steps`` is this ``learn()`` call's step budget (passed explicitly so a warm resume,
    where ``num_timesteps`` starts non-zero, anneals over the run window rather than from epoch 0).
    """

    def __init__(self, target_log_std: float, anneal_fraction: float, total_steps: int):
        super().__init__()
        self.target = float(target_log_std)
        self.anneal_fraction = float(anneal_fraction)
        self.total_steps = max(int(total_steps), 1)
        self._start_ts = 0
        self._start_val = 0.0

    def _on_training_start(self) -> None:
        import torch

        log_std = self.model.policy.log_std
        self._start_val = float(log_std.data.mean().item())
        self._start_ts = int(self.num_timesteps)
        # Freeze: the optimizer must not be able to raise std back up while we drive it down.
        log_std.requires_grad_(False)
        with torch.no_grad():
            log_std.data.fill_(self._start_val)
        logger.info(
            "%s active: annealing log_std %.3f -> %.3f over %.0f%% of %d steps (std %.3f -> %.3f), "
            "parameter frozen.",
            ENV_VAR,
            self._start_val,
            self.target,
            self.anneal_fraction * 100.0,
            self.total_steps,
            pow(2.718281828, self._start_val),
            pow(2.718281828, self.target),
        )

    def _current_value(self) -> float:
        done = (int(self.num_timesteps) - self._start_ts) / self.total_steps
        frac = min(1.0, max(0.0, done / self.anneal_fraction))
        return self._start_val + (self.target - self._start_val) * frac

    def _on_rollout_start(self) -> None:
        import torch

        val = self._current_value()
        with torch.no_grad():
            self.model.policy.log_std.data.fill_(val)

    def _on_step(self) -> bool:  # noqa: D401 - BaseCallback contract
        return True
