"""Lossless-pause checkpoint-on-signal handler for the desktop app (UC-61 AC7).

The desktop app pauses a training run by sending it a termination signal; to make that pause
**lossless** the run must flush a resumable checkpoint before it exits.
:class:`CheckpointOnSignalCallback` does exactly that, structured to avoid the UC-53 mid-write
corruption class:

* **The signal handler is I/O-free.** SIGINT / SIGTERM only set ``self._stop_requested`` — they
  never touch the disk. (Saving *inside* an async signal handler is what could interrupt a
  cloudpickle write mid-stream and produce the UC-53 ``BadZipFile``.)
* **The flush happens at a safe loop boundary.** The next :meth:`_on_step` (called by SB3 between
  environment steps, never mid-serialisation) sees the flag, flushes a checkpoint, and returns
  ``False`` so ``model.learn`` stops cleanly; ``train()`` then runs its normal final-save + return.
* **The flush is atomic.** The model is written through the existing
  :meth:`~drone_fly.train.progress_ppo.ProgressReportingPPO.save` (temp file + ``os.replace``,
  with ``_progress_sink`` excluded), and the VecNormalize stats are written with the same
  temp-then-``os.replace`` discipline here. Both use the exact
  ``ppo_racer_<steps>_steps.zip`` / ``ppo_racer_vecnormalize_<steps>_steps.pkl`` names that
  :func:`~drone_fly.train.loop.find_latest_checkpoint` / ``_infer_vecnormalize_path`` expect, so a
  subsequent ``resume: auto`` picks the flushed checkpoint up automatically.

A **second** signal restores the previous handler and re-raises, so a second Ctrl-C still
force-kills (the pause is a courtesy on the first signal, not a trap). Handlers are registered on
:meth:`_on_training_start` and always restored on :meth:`_on_training_end`. When the callback is
constructed but signal registration is impossible (not the main thread), the feature disables
itself gracefully — training is never affected. Gated off by default in ``loop.py``
(``checkpoint_on_signal=False``) so a run that does not opt in is byte-identical.
"""

from __future__ import annotations

import logging
import os
import signal
import tempfile

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

#: Signals that trigger a lossless-pause flush. SIGTERM is how a subprocess manager stops a run;
#: SIGINT is an interactive Ctrl-C. Both are handled identically.
_TRAPPED_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class CheckpointOnSignalCallback(BaseCallback):
    """Trap SIGINT/SIGTERM and flush an atomic, resumable checkpoint before stopping (UC-61 AC7).

    Parameters
    ----------
    models_dir:
        The run's checkpoint directory (``training/<name>/checkpoints``), i.e. ``cfg.models_dir``.
    name_prefix:
        Checkpoint filename prefix (``CHECKPOINT_PREFIX`` = ``"ppo_racer"``), so the flushed files
        match the ``CheckpointCallback`` naming ``resume: auto`` scans for.
    """

    def __init__(
        self,
        models_dir: str,
        *,
        name_prefix: str,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._models_dir = str(models_dir)
        self._name_prefix = str(name_prefix)
        self._stop_requested = False
        self._flushed = False
        self._signum: int | None = None
        self._registered = False
        self._previous_handlers: dict[int, object] = {}

    # -- lifecycle -------------------------------------------------------------------------

    def _on_training_start(self) -> None:
        """Install the I/O-free signal handlers (main thread only; degrade gracefully otherwise)."""
        for sig in _TRAPPED_SIGNALS:
            try:
                self._previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError, RuntimeError) as exc:
                # signal.signal only works in the main thread; a worker/thread context makes the
                # lossless-pause feature unavailable — never a training error.
                logger.warning(
                    "CheckpointOnSignalCallback: could not trap signal %s (%s); "
                    "lossless-pause disabled for this run.",
                    sig,
                    exc,
                )
                self._restore_handlers()
                return
        self._registered = True

    def _on_step(self) -> bool:
        """At a safe loop boundary, flush once if a stop was requested and ask learn to stop."""
        if self._stop_requested and not self._flushed:
            self._flushed = True
            self._flush_checkpoint()
            return False  # stop model.learn cleanly; train() runs its normal final-save next
        return True

    def _on_training_end(self) -> None:
        self._restore_handlers()

    # -- signal handling (I/O-free) --------------------------------------------------------

    def _handle_signal(self, signum, frame) -> None:  # noqa: ANN001 - signal handler signature
        """Set the stop flag on the first signal; restore + re-raise on the second (force-kill)."""
        if self._stop_requested:
            # Second signal: honour a hard abort. Restore the previous handlers and re-deliver so
            # the default action (KeyboardInterrupt / process termination) takes effect.
            self._restore_handlers()
            try:
                signal.raise_signal(signum)
            except Exception:  # noqa: BLE001 - re-delivery is best-effort
                pass
            return
        self._stop_requested = True
        self._signum = signum
        logger.warning(
            "Signal %s received: will flush a checkpoint at the next safe step and stop "
            "(send again to force-quit).",
            signum,
        )

    def _restore_handlers(self) -> None:
        """Restore any previously-installed signal handlers (idempotent)."""
        for sig, handler in list(self._previous_handlers.items()):
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, RuntimeError):
                pass
        self._previous_handlers.clear()
        self._registered = False

    # -- flush -----------------------------------------------------------------------------

    def _flush_checkpoint(self) -> None:
        """Write an atomic, resumable model + VecNormalize checkpoint at the current step count.

        Mirrors the ``CheckpointCallback`` naming so ``resume: auto`` finds it. Any failure is
        logged, never raised — a failed pause-flush must not turn into a training crash.
        """
        try:
            steps = int(self.num_timesteps)
            os.makedirs(self._models_dir, exist_ok=True)
            model_path = os.path.join(self._models_dir, f"{self._name_prefix}_{steps}_steps.zip")
            # ProgressReportingPPO.save is already atomic (temp + os.replace) and excludes the
            # unpicklable _progress_sink (UC-53), so this reuses the hardened path.
            self.model.save(model_path)

            vecnorm = self.model.get_vec_normalize_env()
            if vecnorm is not None:
                vec_path = os.path.join(
                    self._models_dir,
                    f"{self._name_prefix}_vecnormalize_{steps}_steps.pkl",
                )
                self._atomic_vecnormalize_save(vecnorm, vec_path)
            logger.warning("Flushed lossless-pause checkpoint at %d steps: %s", steps, model_path)
        except Exception as exc:  # noqa: BLE001 - a failed flush must not crash training
            logger.error("CheckpointOnSignalCallback flush failed: %s", exc)

    @staticmethod
    def _atomic_vecnormalize_save(vecnorm, path: str) -> None:
        """Save VecNormalize stats atomically (temp beside target + ``os.replace``), UC-53 style.

        ``VecNormalize.save`` pickles straight to the path (non-atomic); an interrupted write
        would leave a truncated ``.pkl`` that ``resume: auto`` cannot load. Writing to a temp in
        the same directory and ``os.replace``-ing keeps any prior stats byte-intact on failure.
        """
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
        )
        os.close(fd)
        try:
            vecnorm.save(temp_name)
            os.replace(temp_name, path)
        except BaseException:
            try:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            except OSError:
                pass
            raise


__all__ = ["CheckpointOnSignalCallback"]
