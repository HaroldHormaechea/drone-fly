"""UC-61 AC7 — the lossless-pause checkpoint-on-signal handler (``CheckpointOnSignalCallback``).

Hermetic unit tests (no PPO, no real training, no real signals to the test process). They pin the
UC-53-safe pause contract:

* **The signal handler is I/O-free.** Delivering SIGTERM only sets a flag — it never touches disk
  (saving inside an async handler is the UC-53 ``BadZipFile`` corruption class).
* **The flush happens at the next safe ``_on_step`` boundary**, writes a resumable checkpoint under
  the exact ``ppo_racer_<steps>_steps.zip`` / ``ppo_racer_vecnormalize_<steps>_steps.pkl`` names
  ``resume: auto`` scans for, then returns ``False`` to stop ``model.learn`` cleanly.
* **The VecNormalize flush is atomic** — a crash mid-save leaves any prior ``.pkl`` byte-intact.
* **A second signal restores the previous handler and re-raises** (force-quit).
* **Handlers are always restored on training end**, and registration degrades gracefully off the
  main thread. Gated off by default so a non-opted-in run is byte-identical.

Signals are exercised by calling the handler directly (``cb._handle_signal(signum, None)``) — no
real signal is sent to the pytest process, keeping the suite anti-flake and CI-safe.
"""

from __future__ import annotations

import os
import signal
import threading

import pytest

from drone_fly.train.checkpoint_signal import CheckpointOnSignalCallback

PREFIX = "ppo_racer"


# --- fakes --------------------------------------------------------------------------------


class _FakeVecNormalize:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.saved_to: list[str] = []

    def save(self, path):
        if self.fail:
            raise RuntimeError("boom during vecnormalize save")
        self.saved_to.append(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("vecnormalize-stats")


class _FakeModel:
    """A model whose ``save`` writes a marker file (already-atomic in the real ProgressReportingPPO)."""

    def __init__(self, *, num_timesteps=1234, vecnorm=None):
        self.num_timesteps = num_timesteps
        self._vecnorm = vecnorm
        self.saved_to: list[str] = []

    def save(self, path):
        self.saved_to.append(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("model-zip")

    def get_vec_normalize_env(self):
        return self._vecnorm

    def get_env(self):
        return None


def _make_cb(models_dir, *, num_timesteps=1234, vecnorm=None):
    cb = CheckpointOnSignalCallback(str(models_dir), name_prefix=PREFIX)
    cb.model = _FakeModel(num_timesteps=num_timesteps, vecnorm=vecnorm)
    cb.num_timesteps = num_timesteps
    return cb


# --- I/O-free handler + flush-at-step -----------------------------------------------------


def test_handler_is_io_free_no_flush_until_on_step(tmp_path):
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    cb._on_training_start()
    try:
        # First signal: sets the flag only, writes NOTHING.
        cb._handle_signal(signal.SIGTERM, None)
        assert cb._stop_requested is True
        assert cb.model.saved_to == []  # I/O-free handler — no disk touch
        assert os.listdir(tmp_path) == []
    finally:
        cb._on_training_end()


def test_flush_at_on_step_writes_resumable_checkpoint_and_stops(tmp_path):
    vec = _FakeVecNormalize()
    cb = _make_cb(tmp_path, num_timesteps=4096, vecnorm=vec)
    cb._on_training_start()
    try:
        cb._handle_signal(signal.SIGTERM, None)
        result = cb._on_step()  # the safe loop boundary
    finally:
        cb._on_training_end()

    assert result is False  # asks model.learn to stop cleanly
    ckpt = tmp_path / f"{PREFIX}_4096_steps.zip"
    vecp = tmp_path / f"{PREFIX}_vecnormalize_4096_steps.pkl"
    assert ckpt.is_file()  # exact naming resume:auto scans for
    assert vecp.is_file()
    assert cb.model.saved_to == [str(ckpt)]
    # The vecnormalize save is atomic (writes a temp beside the target, then os.replace), so the
    # recorded path is the temp; the important guarantees are (a) it was saved exactly once and
    # (b) the final canonical .pkl exists (asserted above).
    assert len(vec.saved_to) == 1
    assert vec.saved_to[0].startswith(str(vecp))


def test_flush_happens_only_once(tmp_path):
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    cb._on_training_start()
    try:
        cb._handle_signal(signal.SIGTERM, None)
        assert cb._on_step() is False  # first: flushes + stops
        # A subsequent _on_step must not re-flush (already flushed) and returns True.
        assert cb._on_step() is True
    finally:
        cb._on_training_end()
    assert len(cb.model.saved_to) == 1


def test_on_step_is_noop_when_no_signal(tmp_path):
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    cb._on_training_start()
    try:
        assert cb._on_step() is True  # no stop requested → keep training
        assert cb.model.saved_to == []
    finally:
        cb._on_training_end()


def test_flush_without_vecnormalize_writes_only_model(tmp_path):
    cb = _make_cb(tmp_path, num_timesteps=64, vecnorm=None)
    cb._on_training_start()
    try:
        cb._handle_signal(signal.SIGINT, None)
        assert cb._on_step() is False
    finally:
        cb._on_training_end()
    assert (tmp_path / f"{PREFIX}_64_steps.zip").is_file()
    assert not (tmp_path / f"{PREFIX}_vecnormalize_64_steps.pkl").exists()


# --- atomicity ----------------------------------------------------------------------------


def test_vecnormalize_save_is_atomic_prior_file_intact_on_failure(tmp_path):
    """A failing vecnormalize save leaves any pre-existing .pkl byte-intact (temp + os.replace)."""
    vecp = tmp_path / f"{PREFIX}_vecnormalize_10_steps.pkl"
    vecp.write_text("previous-good-stats", encoding="utf-8")
    failing = _FakeVecNormalize(fail=True)

    # _atomic_vecnormalize_save must not clobber the target and must not leave temp litter.
    with pytest.raises(RuntimeError):
        CheckpointOnSignalCallback._atomic_vecnormalize_save(failing, str(vecp))

    assert vecp.read_text(encoding="utf-8") == "previous-good-stats"  # untouched
    leftover = [p for p in os.listdir(tmp_path) if p != vecp.name]
    assert leftover == []  # temp file cleaned up


def test_flush_failure_is_swallowed_not_raised(tmp_path):
    """A failed pause-flush must never turn into a training crash (logged, not raised)."""
    cb = _make_cb(tmp_path, num_timesteps=5, vecnorm=_FakeVecNormalize(fail=True))
    cb._on_training_start()
    try:
        cb._handle_signal(signal.SIGTERM, None)
        # The vecnormalize save raises internally; _flush_checkpoint swallows it and _on_step
        # still returns False (stop requested) without propagating.
        assert cb._on_step() is False
    finally:
        cb._on_training_end()
    # The model zip was written before the vecnormalize failure; the run is still asked to stop.
    assert (tmp_path / f"{PREFIX}_5_steps.zip").is_file()


# --- handler lifecycle --------------------------------------------------------------------


def test_handlers_registered_then_restored_on_training_end(tmp_path):
    original = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    cb._on_training_start()
    try:
        # While active, our handler is installed (differs from the original).
        assert signal.getsignal(signal.SIGTERM) == cb._handle_signal
        assert cb._registered is True
    finally:
        cb._on_training_end()
    # Restored exactly to the pre-run handlers — no leak into the caller's process.
    for sig, handler in original.items():
        assert signal.getsignal(sig) == handler
    assert cb._registered is False


def test_second_signal_restores_and_reraises(tmp_path, monkeypatch):
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    pre_run_term = signal.getsignal(signal.SIGTERM)  # the handler before the run installs ours
    cb._on_training_start()
    reraised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", lambda s: reraised.append(s))
    try:
        cb._handle_signal(signal.SIGTERM, None)  # first: arm
        cb._handle_signal(signal.SIGTERM, None)  # second: force-quit path
    finally:
        # handlers already restored by the second signal; guard against double-restore.
        cb._on_training_end()
    assert reraised == [signal.SIGTERM]  # re-delivered for the default action
    # After the second signal the pre-run handler is back in place (no leak).
    assert signal.getsignal(signal.SIGTERM) == pre_run_term


def test_registration_off_main_thread_degrades_gracefully(tmp_path):
    """signal.signal only works on the main thread; a worker context disables the feature, not train."""
    cb = _make_cb(tmp_path, vecnorm=_FakeVecNormalize())
    errors: list[BaseException] = []

    def worker():
        try:
            cb._on_training_start()  # must NOT raise off the main thread
        except BaseException as exc:  # noqa: BLE001 - the whole point is nothing escapes
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    assert cb._registered is False  # feature unavailable, but training is untouched
    # An _on_step in this state is a harmless no-op (no stop requested).
    assert cb._on_step() is True
