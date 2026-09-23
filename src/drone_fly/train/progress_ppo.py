"""Optimize-phase progress instrumentation for the training TUI (UC-52).

PPO's optimize phase — :meth:`PPO.train`'s ``n_epochs × minibatch`` gradient loop — has **no
SB3 callback hook**, and the TUI redraw runs on the same thread that is blocked inside that
loop, so during optimize the screen is frozen at ``collecting 100%``. On the large connectome
actor the optimize phase is the majority of each iteration's wall-clock, so most of every
iteration is currently invisible.

:class:`ProgressReportingPPO` closes that gap with the **minimum viable** instrumentation: it
temporarily wraps ``rollout_buffer.get(batch_size)`` with a counting-proxy generator for the
duration of ``super().train()`` and reports ``(epoch, minibatch)`` progress to an optional sink.
The gradient/loss/optimizer loop is **never copied or modified** — only the buffer's minibatch
iterator is observed.

Design invariants (UC-52 ACs):

* **Inert when unattached (AC-4).** With no sink attached (``--no-tui`` / headless / smoke /
  resume-before-attach), :meth:`train` returns ``super().train()`` immediately — byte-identical
  to plain PPO, zero overhead, ``rollout_buffer.get`` never touched.
* **Never perturbs training (AC-5/AC-6).** Every sink call is guarded; the ``get`` wrap is always
  restored in a ``finally``; any setup failure degrades to plain ``super().train()``; and
  disable-on-anomaly stops reporting (but not training) if the observed epoch/minibatch counts
  exceed the expected ``N``/``M`` (a ``target_kl`` early-stop only *reduces* counts, so it is
  tolerated — only *exceeding* trips the disable).
* **Single owner of begin/end (AC-10b).** :meth:`train` is the sole caller of
  ``begin_optimize`` / ``end_optimize`` / ``set_optimize_duration`` — the counting-proxy only
  ever calls ``tick_optimize`` — so exactly one begin and one end fire per optimize phase.

Sink protocol (duck-typed; the :class:`~drone_fly.train.tui.dashboard.TrainingDashboard` is the
production sink): ``begin_optimize(n_epochs, total_minibatches)`` /
``tick_optimize(epoch, minibatch)`` / ``end_optimize()`` / ``set_optimize_duration(seconds)`` /
``set_collect_duration(seconds)``.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


def counting_get_proxy(real_get, n_epochs, total_minibatches, on_tick):
    """Wrap a ``RolloutBuffer.get`` bound method with a minibatch-counting generator.

    Pure and import-light (no PPO/torch reference), so it is unit-testable with fake iterables.
    The returned callable has ``get``'s signature: each call is one optimize **epoch** (the epoch
    counter, 0-based, increments per call), and each yielded minibatch (1-based within the epoch)
    invokes ``on_tick(epoch, minibatch)`` then forwards the sample **untouched**.

    Disable-on-anomaly (AC-6/AC-7): if more calls arrive than ``n_epochs``, or a single epoch
    yields more than ``total_minibatches`` minibatches, reporting is silently disabled for the
    rest of this optimize phase — the samples are still yielded unchanged, so training is never
    perturbed. A ``target_kl`` early-stop only *reduces* the observed counts, so it never trips
    this (the bar simply never reaches 100% that iteration).
    """
    state = {"epoch": 0, "disabled": False}

    def proxy(*args, **kwargs):
        epoch = state["epoch"]
        state["epoch"] = epoch + 1
        # More get() calls than expected epochs → anomaly; stop reporting (still yield samples).
        if epoch >= n_epochs:
            state["disabled"] = True
        minibatch = 0
        for sample in real_get(*args, **kwargs):
            minibatch += 1
            if not state["disabled"]:
                if total_minibatches > 0 and minibatch > total_minibatches:
                    state["disabled"] = True
                else:
                    try:
                        on_tick(epoch, minibatch)
                    except Exception:  # noqa: BLE001 - a sink glitch must never perturb training
                        state["disabled"] = True
            yield sample

    return proxy


class ProgressReportingPPO(PPO):
    """A :class:`~stable_baselines3.PPO` that reports optimize-phase progress to an optional sink.

    Deliberately has **no** ``__init__`` override: the fresh ``cls(...)`` and resume
    ``load(...) → cls(..., _init_setup_model=False)`` construction paths stay byte-identical to
    plain PPO, and the sink is read lazily via ``getattr``. When no sink is attached the class is
    behaviourally indistinguishable from :class:`PPO` (AC-4/AC-8).
    """

    def attach_progress_sink(self, sink) -> None:
        """Attach (or, with ``None``, detach) the optimize-phase progress sink."""
        self._progress_sink = sink

    def _excluded_save_params(self) -> list[str]:
        """Exclude the transient progress sink from SB3's cloudpickle save (UC-53).

        The attached sink is the live :class:`~drone_fly.train.tui.dashboard.TrainingDashboard`,
        which holds a ``threading.RLock`` and is therefore unpicklable. SB3 serializes
        ``self.__dict__`` minus this list, so without ``_progress_sink`` here ``save()`` crashes
        mid-write with ``TypeError: cannot pickle '_thread.RLock'`` and truncates the checkpoint.

        Calls ``super()`` and appends (never hardcodes SB3's list, which varies by version); the
        exclusion is a harmless no-op when no sink is attached. On ``load()`` the attribute is
        simply absent (``getattr(self, "_progress_sink", None) is None``) and ``loop.py``
        re-attaches a sink when the TUI is active.
        """
        return super()._excluded_save_params() + ["_progress_sink"]

    def save(self, path, exclude=None, include=None) -> None:
        """Atomically write a checkpoint so a failed save never corrupts the target (UC-53).

        Serializes to a uniquely-named temp file **in the same directory** as the resolved final
        path, then ``os.replace``s it into place — an atomic same-volume rename on both POSIX and
        Windows/NTFS. A partial/failed serialization (e.g. the historical RLock pickle crash) thus
        leaves any pre-existing checkpoint byte-intact and never publishes a truncated ``.zip``.

        Composes with :meth:`_excluded_save_params`: the temp is written via ``super().save``,
        which drops ``_progress_sink``. When ``path`` is a buffer/file-like object rather than a
        filesystem path, atomic rename is meaningless, so we delegate straight to ``super``.
        """
        # Buffer/file-like target: atomic rename is meaningless — delegate to SB3 unchanged.
        if not isinstance(path, (str, os.PathLike)):
            return super().save(path, exclude=exclude, include=include)

        # Resolve the final path exactly as SB3's open_path does: append ``.zip`` only when the
        # caller supplied no extension (any existing suffix is kept verbatim).
        final = Path(path)
        if final.suffix == "":
            final = Path(f"{final}.zip")

        # Match SB3's auto-create-parent behavior (challenger Minor #1) before creating the temp.
        final.parent.mkdir(parents=True, exist_ok=True)

        # Temp beside the target ⇒ same volume ⇒ os.replace is truly atomic. A NON-EMPTY suffix
        # keeps SB3's open_path from re-appending ``.zip`` to the temp (double-extension trap).
        fd, temp_name = tempfile.mkstemp(
            prefix=f"{final.name}.", suffix=".tmp", dir=str(final.parent)
        )
        # Close the temp fd immediately (challenger Minor #2): a lingering handle can block
        # super().save()'s own open("wb") and the os.replace on Windows.
        os.close(fd)
        temp = Path(temp_name)

        try:
            super().save(str(temp), exclude=exclude, include=include)
            os.replace(temp, final)
        except BaseException:
            # Best-effort cleanup; ``final`` is untouched (only the terminal os.replace writes it).
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass
            raise

    @staticmethod
    def _safe_sink_call(sink, method_name, *args) -> bool:
        """Call ``sink.method_name(*args)`` defensively; return whether it succeeded.

        A missing method or any raised exception is swallowed (and reported at debug) so a sink
        problem can never perturb training — the caller degrades gracefully off the return value.
        """
        try:
            fn = getattr(sink, method_name, None)
            if fn is None:
                return False
            fn(*args)
            return True
        except Exception as exc:  # noqa: BLE001 - never perturb training over the TUI
            logger.debug("progress sink %s failed: %s", method_name, exc)
            return False

    def train(self) -> None:
        """Run PPO's optimize loop, reporting ``(epoch, minibatch)`` progress when a sink is set.

        With no sink (or an un-instrumentable buffer) this returns ``super().train()`` immediately
        — byte-identical to plain PPO. Otherwise it wraps ``rollout_buffer.get`` with the
        counting-proxy for the duration of the real optimize loop, times it, and reports through
        the guarded sink; the wrap is **always** restored in ``finally`` and exactly one
        ``end_optimize`` fires.
        """
        sink = getattr(self, "_progress_sink", None)
        if sink is None:
            return super().train()

        buffer = getattr(self, "rollout_buffer", None)
        original_get = getattr(buffer, "get", None) if buffer is not None else None
        if original_get is None:
            # Nothing safe to wrap → behave exactly like plain PPO.
            return super().train()

        # Set up instrumentation defensively; ANY failure here degrades to plain super().train().
        began = False
        try:
            n_epochs = int(self.n_epochs)
            batch_size = int(self.batch_size)
            total_minibatches = (
                -(-int(self.n_steps) * int(self.n_envs) // batch_size) if batch_size > 0 else 0
            )
            began = self._safe_sink_call(sink, "begin_optimize", n_epochs, total_minibatches)
            buffer.get = counting_get_proxy(
                original_get,
                n_epochs,
                total_minibatches,
                lambda epoch, minibatch: self._safe_sink_call(
                    sink, "tick_optimize", epoch, minibatch
                ),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to plain training, never crash
            logger.debug("optimize-progress instrumentation disabled: %s", exc)
            buffer.get = original_get
            if began:
                self._safe_sink_call(sink, "end_optimize")
            return super().train()

        # Instrumentation installed. Run the real optimize loop; always restore + close out. An
        # exception from super().train() propagates (real training error) after the finally runs.
        start = time.monotonic()
        try:
            return super().train()
        finally:
            elapsed = time.monotonic() - start
            buffer.get = original_get
            self._safe_sink_call(sink, "set_optimize_duration", elapsed)
            if began:
                self._safe_sink_call(sink, "end_optimize")


__all__ = ["ProgressReportingPPO", "counting_get_proxy"]
