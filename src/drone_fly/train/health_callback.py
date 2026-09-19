"""SB3 runtime wiring for the training-health engine (UC-23, AC6/AC7).

An SB3 :class:`~stable_baselines3.common.callbacks.BaseCallback` that, once per rollout,
snapshots the current training signals and calls the pure
:func:`~drone_fly.train.health.assess_training_health` engine. It:

* **self-accounts** episode reward + success from step infos, because ``build_vec_env`` wraps
  no ``Monitor`` (so SB3 logs no ``rollout/ep_rew_mean`` / ``rollout/success_rate``). Reward
  is accumulated per env and banked on ``dones[i]``; success is read from
  ``info["is_success"]`` on that same step (set by the env at episode end). This keeps the env
  stack untouched (AC7) — same approach as :class:`RecordingCallback`.
* owns its **own** ``n_updates`` counter (incremented in ``_on_rollout_end``), never reading a
  non-existent stable key from SB3's ``name_to_value``.
* reads ``train/*`` metrics via ``name_to_value.get(key)`` with a ``None`` default so the
  first rollout — which fires **before** that iteration's ``PPO.train()`` and therefore has no
  ``train/*`` keys yet — never ``KeyError``s; the warm-up guard short-circuits anyway.
* stores ``latest_verdict`` and invokes an optional injected ``on_verdict`` callback (the seam
  UC-22's status bar plugs into).
* logs only on **status change or every ``log_every`` updates** (AC6 no-spam cadence).

Fully defensive: any accounting/assessment error disables the callback and logs, but never
crashes training (the :class:`RecordingCallback` contract).
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable

from stable_baselines3.common.callbacks import BaseCallback

from drone_fly.train.health import (
    DEFAULT_THRESHOLDS,
    HealthThresholds,
    HealthVerdict,
    TrainingSignals,
    assess_training_health,
)

logger = logging.getLogger(__name__)

#: Default rolling-window length (in rollouts) kept per metric.
DEFAULT_WINDOW = 10
#: Default log cadence: emit a health line at least every N updates even without a status change.
DEFAULT_LOG_EVERY = 10

#: SB3 ``name_to_value`` keys read each rollout (all optional; absent → skipped).
_EV_KEY = "train/explained_variance"
_STD_KEY = "train/std"
_KL_KEY = "train/approx_kl"
_VLOSS_KEY = "train/value_loss"


class HealthCallback(BaseCallback):
    """Assess training health once per rollout and expose the verdict (AC6/AC7)."""

    def __init__(
        self,
        *,
        thresholds: HealthThresholds = DEFAULT_THRESHOLDS,
        window: int = DEFAULT_WINDOW,
        log_every: int = DEFAULT_LOG_EVERY,
        on_verdict: Callable[[HealthVerdict], None] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.thresholds = thresholds
        self.window = max(int(window), 1)
        self.log_every = max(int(log_every), 1)
        self.on_verdict = on_verdict

        self.n_updates = 0
        self.latest_verdict: HealthVerdict | None = None

        # Rolling per-metric windows (oldest first, current last).
        self._ep_rew: deque[float] = deque(maxlen=self.window)
        self._success: deque[float] = deque(maxlen=self.window)
        self._ev: deque[float] = deque(maxlen=self.window)
        self._entropy: deque[float] = deque(maxlen=self.window)
        self._kl: deque[float] = deque(maxlen=self.window)
        self._vloss: deque[float] = deque(maxlen=self.window)

        # Per-env reward accumulators + this-rollout episode banks (set at training start).
        self._rew_accum: list[float] = []
        self._rollout_ep_rewards: list[float] = []
        self._rollout_ep_successes: list[float] = []

        self._last_logged_status: str | None = None
        self._enabled = True

    # -- lifecycle -------------------------------------------------------------------------

    def _on_training_start(self) -> None:
        try:
            n_envs = int(getattr(self.training_env, "num_envs", 1) or 1)
        except Exception:  # noqa: BLE001 - never crash training over env introspection
            n_envs = 1
        self._rew_accum = [0.0] * n_envs
        self._rollout_ep_rewards = []
        self._rollout_ep_successes = []

    def _on_step(self) -> bool:
        if not self._enabled:
            return True
        try:
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            infos = self.locals.get("infos") or []
            if rewards is None or dones is None:
                return True
            for i in range(len(rewards)):
                if i >= len(self._rew_accum):  # env count changed unexpectedly — grow safely
                    self._rew_accum.extend([0.0] * (i - len(self._rew_accum) + 1))
                self._rew_accum[i] += float(rewards[i])
                if bool(dones[i]):
                    self._rollout_ep_rewards.append(self._rew_accum[i])
                    info = infos[i] if i < len(infos) else {}
                    self._rollout_ep_successes.append(1.0 if info.get("is_success", False) else 0.0)
                    self._rew_accum[i] = 0.0
        except Exception as exc:  # never crash training over accounting
            logger.warning("HealthCallback step accounting disabled after error: %s", exc)
            self._enabled = False
        return True

    def _on_rollout_end(self) -> None:
        if not self._enabled:
            return
        try:
            self.n_updates += 1

            # Bank this rollout's self-accounted episode reward/success means (if any finished).
            if self._rollout_ep_rewards:
                self._ep_rew.append(sum(self._rollout_ep_rewards) / len(self._rollout_ep_rewards))
            if self._rollout_ep_successes:
                self._success.append(
                    sum(self._rollout_ep_successes) / len(self._rollout_ep_successes)
                )
            self._rollout_ep_rewards = []
            self._rollout_ep_successes = []

            # Read train/* defensively: absent on the first rollout (fires before PPO.train()).
            name_to_value = {}
            logger_obj = getattr(self.model, "logger", None)
            if logger_obj is not None:
                name_to_value = getattr(logger_obj, "name_to_value", {}) or {}
            self._append_metric(self._ev, name_to_value.get(_EV_KEY))
            self._append_metric(self._entropy, name_to_value.get(_STD_KEY))
            self._append_metric(self._kl, name_to_value.get(_KL_KEY))
            self._append_metric(self._vloss, name_to_value.get(_VLOSS_KEY))

            signals = TrainingSignals(
                n_updates=self.n_updates,
                ep_rew_mean=tuple(self._ep_rew),
                success_rate=tuple(self._success),
                explained_variance=tuple(self._ev),
                entropy_std=tuple(self._entropy),
                approx_kl=tuple(self._kl),
                value_loss=tuple(self._vloss),
            )
            verdict = assess_training_health(signals=signals, thresholds=self.thresholds)
            self.latest_verdict = verdict

            if self.on_verdict is not None:
                try:
                    self.on_verdict(verdict)
                except Exception as exc:  # an injected consumer must not crash training
                    logger.warning("HealthCallback on_verdict consumer raised: %s", exc)

            self._maybe_log(verdict)
        except Exception as exc:  # never crash training over health assessment
            logger.warning("HealthCallback disabled after error: %s", exc)
            self._enabled = False

    # -- helpers ---------------------------------------------------------------------------

    @staticmethod
    def _append_metric(window: deque[float], value) -> None:
        """Append ``value`` to ``window`` when it is a usable float; ignore ``None``/garbage."""
        if value is None:
            return
        try:
            window.append(float(value))
        except (TypeError, ValueError):
            pass

    def _maybe_log(self, verdict: HealthVerdict) -> None:
        """Emit a log line on a status change or every ``log_every`` updates (AC6)."""
        status_changed = verdict.status != self._last_logged_status
        cadence_hit = (self.n_updates % self.log_every) == 0
        if not (status_changed or cadence_hit):
            return
        level = logging.WARNING if verdict.status != "normal" else logging.INFO
        logger.log(level, "Training health [update %d]: %s", self.n_updates, verdict.message)
        self._last_logged_status = verdict.status


__all__ = ["HealthCallback", "DEFAULT_WINDOW", "DEFAULT_LOG_EVERY"]
