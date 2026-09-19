"""SB3 → dashboard bridge callback for the training TUI (UC-22).

:class:`TuiCallback` snapshots one rollout's metrics at ``_on_rollout_end`` and feeds them to
the :class:`~drone_fly.train.tui.dashboard.TrainingDashboard`, then redraws. It is fully
defensive: any error disables the callback and is logged, never crashing training.

Two timing subtleties (verified against SB3 2.9.0 ``on_policy_algorithm.py``) drive where each
value is read:

* **ROLLOUT stats come from the model's episode buffers, NOT the logger.** ``collect_rollouts``
  updates ``ep_info_buffer`` / ``ep_success_buffer`` and fires ``on_rollout_end`` *before*
  ``dump_logs`` records ``rollout/*`` and clears ``name_to_value`` — so at callback time the
  buffers are fresh but ``rollout/*`` is absent. ``safe_mean`` of an empty buffer yields ``nan``
  (first rollout / no completed episodes), which the data layer treats as "not observed".
* **TRAIN stats come from ``logger.name_to_value`` and are one iteration lagged.** They are
  recorded by the *previous* iteration's ``PPO.train()`` and are therefore ``None`` on the very
  first rollout (and ``train/std`` is only logged for a continuous action space). The data
  layer's None/nan tolerance renders those as placeholders — never ``0``, never a crash.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

logger = logging.getLogger(__name__)

_ENTROPY_LOSS_KEY = "train/entropy_loss"
_STD_KEY = "train/std"
_VLOSS_KEY = "train/value_loss"
_KL_KEY = "train/approx_kl"
_EV_KEY = "train/explained_variance"

#: Minimum wall-clock gap between intra-rollout heartbeats (~4×/sec, matching the dashboard
#: ``Live`` refresh cadence ``_REFRESH_PER_SECOND``). The time gate keeps the per-step fast path
#: a subtract+compare, so fast slices are unaffected (AC-2).
_HEARTBEAT_MIN_INTERVAL = 0.25


class TuiCallback(BaseCallback):
    """Push a rollout snapshot into the dashboard model each PPO update, then redraw.

    Between rollout boundaries a throttled ``_on_step`` heartbeat (UC-30) keeps the dashboard
    visibly alive during a slow collection: it ticks the live elapsed clock and a within-rollout
    step-progress indicator (~4×/sec) and refreshes the log pane, without touching the full
    rollout-end snapshot.
    """

    def __init__(
        self, dashboard, *, now: Callable[[], float] = time.monotonic, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self.dashboard = dashboard
        #: Injectable monotonic clock (defaults to ``time.monotonic``) — the seam AC-2 tests use
        #: to drive the heartbeat throttle deterministically without real sleeps.
        self._now = now
        self._n_updates = 0
        self._start_time: float | None = None
        self._enabled = True
        #: Timestamp of the last heartbeat redraw; ``None`` forces the next step to redraw at once.
        self._last_heartbeat: float | None = None
        #: ``num_timesteps`` captured at the current rollout's start, for step-progress math.
        self._rollout_start_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_time = self._now()

    def _on_rollout_start(self) -> None:
        """Reset the per-rollout step baseline + heartbeat gate (first step redraws instantly)."""
        self._rollout_start_timesteps = self.num_timesteps
        self._last_heartbeat = None

    def _on_step(self) -> bool:
        if not self._enabled:
            return True
        now = self._now()
        if (
            self._last_heartbeat is not None
            and (now - self._last_heartbeat) < _HEARTBEAT_MIN_INTERVAL
        ):
            return True
        self._last_heartbeat = now
        try:
            current = max(0, self.num_timesteps - self._rollout_start_timesteps)
            n_steps = int(getattr(self.model, "n_steps", 0) or 0)
            n_envs = int(
                getattr(self.training_env, "num_envs", 0) or getattr(self.model, "n_envs", 1) or 1
            )
            target = n_steps * n_envs
            elapsed = self._now() - self._start_time if self._start_time is not None else 0.0
            self.dashboard.model.tick(
                elapsed_seconds=elapsed,
                rollout_steps=current,
                rollout_target=target,
            )
            self.dashboard.redraw()
        except Exception as exc:  # noqa: BLE001 - never crash training over the TUI
            logger.warning("TuiCallback heartbeat disabled after error: %s", exc)
            self._enabled = False
        return True

    def _on_rollout_end(self) -> None:
        if not self._enabled:
            return
        try:
            self._n_updates += 1

            # ROLLOUT — fresh, from the model buffers (VecMonitor-populated), NOT the logger.
            ep_buf = list(self.model.ep_info_buffer or [])
            ep_rew = safe_mean([ep["r"] for ep in ep_buf]) if ep_buf else float("nan")
            ep_len = safe_mean([ep["l"] for ep in ep_buf]) if ep_buf else float("nan")
            succ_buf = list(self.model.ep_success_buffer or [])
            success = safe_mean(succ_buf) if succ_buf else float("nan")

            # TRAIN — one-iteration-lagged, None-tolerant, from name_to_value.
            name_to_value = {}
            logger_obj = getattr(self.model, "logger", None)
            if logger_obj is not None:
                name_to_value = getattr(logger_obj, "name_to_value", {}) or {}

            elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0

            self.dashboard.model.update(
                n_updates=self._n_updates,
                elapsed_seconds=elapsed,
                ep_rew_mean=ep_rew,
                ep_len_mean=ep_len,
                success_rate=success,
                entropy_loss=name_to_value.get(_ENTROPY_LOSS_KEY),
                std=name_to_value.get(_STD_KEY),
                value_loss=name_to_value.get(_VLOSS_KEY),
                approx_kl=name_to_value.get(_KL_KEY),
                explained_variance=name_to_value.get(_EV_KEY),
            )
            self.dashboard.redraw()
        except Exception as exc:  # noqa: BLE001 - never crash training over the TUI
            logger.warning("TuiCallback disabled after error: %s", exc)
            self._enabled = False


__all__ = ["TuiCallback"]
