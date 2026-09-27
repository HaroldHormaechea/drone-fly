"""Structured JSONL training-status emitter for the desktop app (UC-61 AC6).

The training TUI (UC-22) renders progress to the terminal only; there is no machine-readable
progress stream the desktop app can tail. :class:`StatusEmitterCallback` closes that gap: once
per PPO rollout it appends **one newline-terminated JSON object** to a ``status.jsonl`` file
describing the run's live state (timesteps + target, rollout metrics, train metrics, the
UC-22/UC-23 health verdict, and the UC-49 drone-dynamics summary). The desktop backend tails
this file to drive a progress bar and curves (a CSV-tail fallback exists, but this is primary).

Design invariants:

* **Same data sources, no new assessment.** Rollout stats come from the model's episode
  buffers (``ep_info_buffer`` / ``ep_success_buffer``), train stats from
  ``logger.name_to_value`` (one-iteration-lagged, None-tolerant) — exactly as
  :class:`~drone_fly.train.tui.callback.TuiCallback` reads them. The health verdict is read from
  an injected :class:`~drone_fly.train.health_callback.HealthCallback` (its ``latest_verdict``),
  so this callback never re-runs the health assessment; it only serialises it. The drone-dynamics
  summary is built via the shared ``drone_fly.adapter.dynamics_summary.drone_dynamics_summary``.
* **Fully defensive (mirrors TuiCallback).** Any write/format/introspection error disables the
  callback and is logged — it never propagates into ``model.learn``. A single bad rollout can
  never crash a training run.
* **Independent of the TUI.** Wired in ``loop.py`` whenever ``status_path`` is set, regardless of
  whether the dashboard is active, so a plain ``drone-fly train --no-tui`` (how the app launches
  runs) still emits status. When ``status_path is None`` the callback is never constructed, so
  ``smoke_train`` and existing tests stay byte-identical.
* **Every value JSON-safe.** Non-finite floats (``nan``/``inf`` — e.g. an empty-buffer
  ``safe_mean`` on the first rollout, or an unobserved train metric) are serialised as JSON
  ``null`` so a strict parser (``JSON.parse`` in the browser) never chokes. Lines are written
  with ``allow_nan=False`` after sanitisation.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean

logger = logging.getLogger(__name__)

_LOSS_KEY = "train/loss"
_VLOSS_KEY = "train/value_loss"
_KL_KEY = "train/approx_kl"
_ENTROPY_LOSS_KEY = "train/entropy_loss"
_EV_KEY = "train/explained_variance"
_STD_KEY = "train/std"
_FPS_KEY = "time/fps"


def _finite(value: Any) -> Any:
    """Coerce ``value`` to a JSON-safe scalar: a finite python number, ``None``, or unchanged.

    Two jobs:

    * **Non-finite → null.** JSON has no representation for ``nan``/``inf``; a strict reader (the
      browser's ``JSON.parse``) rejects the JavaScript ``NaN``/``Infinity`` tokens Python would
      otherwise emit. ``safe_mean`` of an empty buffer yields ``nan`` (first rollout), and an
      unobserved train metric is ``None`` — both collapse to JSON ``null``, the "not observed"
      sentinel.
    * **numpy scalar → python.** SB3 metrics (and ``safe_mean`` results) are frequently numpy
      ``float32``/``int64``, which the stdlib ``json`` cannot serialise. Any real number is
      converted to a python ``int``/``float`` here so ``json.dumps`` succeeds.

    Non-numeric values (strings such as the backend name) are returned unchanged.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):  # python int (and not bool, handled above) — keep exact
        return value
    try:
        as_float = float(value)  # python float, numpy float/int scalar, or numeric-only path
    except (TypeError, ValueError):
        return value  # non-numeric (e.g. a string) — leave as-is
    if not math.isfinite(as_float):
        return None
    return as_float


class StatusEmitterCallback(BaseCallback):
    """Append one JSON status line per rollout to ``status_path`` (UC-61 AC6).

    Parameters
    ----------
    status_path:
        Destination ``status.jsonl`` file. Its parent directory is created on training start.
    health_cb:
        The run's :class:`~drone_fly.train.health_callback.HealthCallback`, read for the current
        ``latest_verdict``. Must be registered *before* this callback in the callback list so its
        verdict is fresh when this callback serialises the rollout. ``None`` disables the health
        field (it is emitted as ``null``).
    tw_preserving:
        The run's ``EnvConfig.pybullet_tw_preserving`` flag, forwarded to
        :func:`~drone_fly.adapter.dynamics_summary.drone_dynamics_summary` so the reported applied
        mass / T/W match the env's UC-48 resolution.
    """

    def __init__(
        self,
        status_path: str,
        *,
        health_cb: BaseCallback | None = None,
        tw_preserving: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._status_path = str(status_path)
        self._health_cb = health_cb
        self._tw_preserving = bool(tw_preserving)
        self._n_updates = 0
        self._start_time: float | None = None
        self._enabled = True

    def _on_training_start(self) -> None:
        try:
            import os

            os.makedirs(os.path.dirname(self._status_path) or ".", exist_ok=True)
            self._start_time = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - never crash training over status I/O
            logger.warning("StatusEmitterCallback disabled after setup error: %s", exc)
            self._enabled = False

    def _on_step(self) -> bool:  # noqa: D401 - the emitter is a per-rollout writer
        return True

    def _on_rollout_end(self) -> None:
        if not self._enabled:
            return
        try:
            self._n_updates += 1
            line = json.dumps(self._build_record(), allow_nan=False)
            with open(self._status_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - never crash training over status I/O
            logger.warning("StatusEmitterCallback disabled after error: %s", exc)
            self._enabled = False

    # -- record assembly -------------------------------------------------------------------

    def _build_record(self) -> dict[str, Any]:
        """Assemble one status record from the SAME sources the TUI/health callbacks read."""
        # ROLLOUT — fresh, from the model buffers (VecMonitor-populated), NOT the logger.
        ep_buf = list(self.model.ep_info_buffer or [])
        ep_rew = safe_mean([ep["r"] for ep in ep_buf]) if ep_buf else float("nan")
        ep_len = safe_mean([ep["l"] for ep in ep_buf]) if ep_buf else float("nan")
        succ_buf = list(self.model.ep_success_buffer or [])
        success = safe_mean(succ_buf) if succ_buf else float("nan")

        # TRAIN — one-iteration-lagged, None-tolerant, from name_to_value.
        name_to_value: dict[str, Any] = {}
        logger_obj = getattr(self.model, "logger", None)
        if logger_obj is not None:
            name_to_value = getattr(logger_obj, "name_to_value", {}) or {}

        elapsed = time.monotonic() - self._start_time if self._start_time is not None else 0.0
        target = getattr(self.model, "_total_timesteps", None)

        record: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "target_timesteps": int(target) if isinstance(target, (int, float)) else None,
            "n_updates": self._n_updates,
            "elapsed_seconds": _finite(elapsed),
            "fps": _finite(name_to_value.get(_FPS_KEY)),
            "rollout": {
                "ep_rew_mean": _finite(ep_rew),
                "ep_len_mean": _finite(ep_len),
                "success_rate": _finite(success),
            },
            "train": {
                "loss": _finite(name_to_value.get(_LOSS_KEY)),
                "value_loss": _finite(name_to_value.get(_VLOSS_KEY)),
                "approx_kl": _finite(name_to_value.get(_KL_KEY)),
                "entropy_loss": _finite(name_to_value.get(_ENTROPY_LOSS_KEY)),
                "explained_variance": _finite(name_to_value.get(_EV_KEY)),
                "std": _finite(name_to_value.get(_STD_KEY)),
            },
            "health": self._health(),
            "dynamics": self._drone_dynamics(),
        }
        return record

    def _health(self) -> dict[str, Any] | None:
        """Serialise the injected HealthCallback's latest verdict (status + message)."""
        cb = self._health_cb
        if cb is None:
            return None
        verdict = getattr(cb, "latest_verdict", None)
        if verdict is None:
            return None
        try:
            return {"status": verdict.status, "message": verdict.message}
        except Exception as exc:  # noqa: BLE001 - best-effort; health is optional
            logger.debug("StatusEmitterCallback health read skipped this rollout: %s", exc)
            return None

    def _drone_dynamics(self) -> dict[str, Any] | None:
        """Build the shared drone-dynamics summary from env-0 as a JSON-safe dict (UC-49).

        Best-effort and self-contained (mirrors ``TuiCallback._push_drone_dynamics``): any
        failure returns ``None`` so the rest of the record still writes.
        """
        try:
            from drone_fly.adapter.dynamics_summary import drone_dynamics_summary
            from drone_fly.env.config import DynamicsParams

            venv = self.training_env
            if venv is None:
                return None
            backend = venv.get_attr("backend")[0]
            dyn = venv.get_attr("active_dynamics")[0] or DynamicsParams()
            try:
                spawn_z = venv.get_attr("spawn_z")[0]
            except Exception:  # noqa: BLE001 - env may predate the accessor; leave unknown
                spawn_z = None
            summary = drone_dynamics_summary(
                backend=backend,
                sampled_mass=dyn.mass,
                max_body_rate=dyn.max_body_rate,
                max_thrust=dyn.max_thrust,
                tw_preserving=self._tw_preserving,
                spawn_z=spawn_z,
                pybullet_mass_ratio=dyn.pybullet_mass_ratio,
                target_tw=dyn.thrust_to_weight,
                arm_length=dyn.arm_length,
            )
            return {k: _finite(v) for k, v in asdict(summary).items()}
        except Exception as exc:  # noqa: BLE001 - best-effort; the segment is optional
            logger.debug("StatusEmitterCallback drone-dynamics summary skipped: %s", exc)
            return None


__all__ = ["StatusEmitterCallback"]
