"""UC-53 — checkpoint-save RLock crash fix + atomic-save hardening.

Regression + hardening coverage for the two composing overrides added to
:class:`~drone_fly.train.progress_ppo.ProgressReportingPPO` in
``use-cases/plans/53-fix-checkpoint-save-rlock-crash.md``:

* ``_excluded_save_params()`` → ``super()`` list **plus** ``"_progress_sink"``, so the live
  :class:`~drone_fly.train.tui.dashboard.TrainingDashboard` (which holds a ``threading.RLock``
  and is therefore unpicklable) is dropped from SB3's cloudpickle save instead of crashing it
  with ``TypeError: cannot pickle '_thread.RLock'``.
* ``save()`` → serialize to a uniquely-named temp **in the target's directory**, then
  ``os.replace`` it into place — an atomic same-volume rename — so a failed/partial write can
  never truncate the target or destroy a pre-existing checkpoint. Buffer/file-like targets
  delegate straight to ``super`` (atomic rename is meaningless for a buffer).

The bug shipped in UC-52 (PR #58) precisely because its hermetic tests only ever saved with a
*picklable* recording sink — so every AC1 assertion here attaches a sink that is **genuinely
unpicklable** (a real ``threading.RLock``) and precondition-asserts that fact so the regression
can never rot into a silent no-op.

All hermetic: a fully-constructed tiny :class:`ProgressReportingPPO` on ``CartPole-v1`` with a
small ``MlpPolicy`` on CPU — real ``save``/``load`` round trips, no connectome/pybullet/GPU.
"""

from __future__ import annotations

import io
import pickle
import threading
import warnings
from pathlib import Path

import pytest

from drone_fly.train.progress_ppo import ProgressReportingPPO

# SB3's PPO ctor emits benign UserWarnings on a tiny CartPole config (e.g. n_steps not a
# multiple of batch_size); silence so the hermetic suite stays quiet.
warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Test doubles + helpers
# --------------------------------------------------------------------------- #
class _UnpicklableSink:
    """A duck-typed progress sink holding a real ``threading.RLock`` — genuinely unpicklable.

    Stands in for the production :class:`TrainingDashboard`: exactly the kind of object whose
    presence in ``model.__dict__`` crashed ``save()`` before UC-53 excluded ``_progress_sink``.
    Implements just enough of the optimize-sink protocol to be a plausible sink.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.events: list = []

    def begin_optimize(self, n_epochs, total_minibatches) -> None:
        with self._lock:
            self.events.append(("begin", n_epochs, total_minibatches))

    def tick_optimize(self, epoch, minibatch) -> None:
        with self._lock:
            self.events.append(("tick", epoch, minibatch))

    def end_optimize(self) -> None:
        with self._lock:
            self.events.append(("end",))


def _temp_leftovers(directory: Path) -> list[str]:
    """Return any stray ``*.tmp`` atomic-save temp files lingering in ``directory``."""
    return sorted(p.name for p in Path(directory).glob("*.tmp"))


def _build_tiny_ppo() -> ProgressReportingPPO:
    """A fully-constructed tiny ProgressReportingPPO (NOT the ``__new__`` shortcut UC-52 uses).

    Small ``MlpPolicy`` on ``CartPole-v1``, minimal ``n_steps``/``batch_size``, CPU, seeded — so
    real ``save``/``load`` round trips exercise the actual SB3 serialization path cheaply.
    """
    return ProgressReportingPPO(
        "MlpPolicy",
        "CartPole-v1",
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": [8]},
    )


@pytest.fixture(scope="module")
def tiny_model() -> ProgressReportingPPO:
    """One constructed model reused across tests. ``save()`` does not mutate model state; each
    test sets the sink it needs, and ``load()`` always yields a brand-new instance — so module
    scope is safe and keeps the suite fast."""
    return _build_tiny_ppo()


# --------------------------------------------------------------------------- #
# AC1 — save succeeds with an unpicklable (RLock-bearing) sink attached
# --------------------------------------------------------------------------- #
def test_ac1_save_with_unpicklable_rlock_sink_does_not_raise(tiny_model, tmp_path) -> None:
    """The exact prior crash mode is gone: a sink holding a real ``RLock`` no longer breaks save.

    Precondition-asserts the sink is genuinely unpicklable so this regression can't rot into a
    no-op (that rot is exactly what let UC-52 ship the bug)."""
    sink = _UnpicklableSink()
    with pytest.raises(TypeError):
        pickle.dumps(sink)  # precondition: reproduces the historical cloudpickle failure surface

    tiny_model.attach_progress_sink(sink)
    try:
        # Before UC-53 this raised: TypeError: cannot pickle '_thread.RLock' object.
        tiny_model.save(tmp_path / "ckpt")
    finally:
        tiny_model.attach_progress_sink(None)

    assert (tmp_path / "ckpt.zip").is_file()
    assert _temp_leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# AC2 — _excluded_save_params == base list + "_progress_sink" (base preserved)
# --------------------------------------------------------------------------- #
def test_ac2_excluded_params_is_base_plus_sink(tiny_model) -> None:
    """The override calls ``super()`` and appends — never hardcodes or reorders SB3's list."""
    from stable_baselines3.common.base_class import BaseAlgorithm

    base = BaseAlgorithm._excluded_save_params(tiny_model)  # the un-overridden SB3 2.9.0 list
    override = tiny_model._excluded_save_params()

    assert override == base + ["_progress_sink"]
    assert override[: len(base)] == base  # base entries preserved, in order
    assert "_progress_sink" not in base  # the override is what adds it
    assert override[-1] == "_progress_sink"


# --------------------------------------------------------------------------- #
# AC3 — round trip: loaded model has no sink; re-attaching one still functions
# --------------------------------------------------------------------------- #
def test_ac3_round_trip_load_drops_sink_then_reattach_functions(tiny_model, tmp_path) -> None:
    sink = _UnpicklableSink()
    tiny_model.attach_progress_sink(sink)
    try:
        tiny_model.save(tmp_path / "ckpt")
    finally:
        tiny_model.attach_progress_sink(None)

    loaded = ProgressReportingPPO.load(tmp_path / "ckpt", device="cpu")
    assert isinstance(loaded, ProgressReportingPPO)
    # The sink was excluded from the pickle → absent on the loaded model.
    assert getattr(loaded, "_progress_sink", None) is None

    # Re-attaching a sink on the loaded model still functions: another unpicklable-sink save
    # round-trips cleanly (the override applies to the loaded instance too).
    sink2 = _UnpicklableSink()
    loaded.attach_progress_sink(sink2)
    assert loaded._progress_sink is sink2
    loaded.save(tmp_path / "ckpt2")
    assert (tmp_path / "ckpt2.zip").is_file()
    reloaded = ProgressReportingPPO.load(tmp_path / "ckpt2", device="cpu")
    assert getattr(reloaded, "_progress_sink", None) is None


# --------------------------------------------------------------------------- #
# AC4 — no-sink save/load path is unaffected (exclusion is a no-op when absent)
# --------------------------------------------------------------------------- #
def test_ac4_no_sink_save_load_unaffected(tiny_model, tmp_path) -> None:
    """With no sink attached, ``save``/``load`` behave exactly like plain PPO."""
    assert getattr(tiny_model, "_progress_sink", None) is None  # nothing attached

    tiny_model.save(tmp_path / "plain")
    assert (tmp_path / "plain.zip").is_file()
    assert _temp_leftovers(tmp_path) == []

    loaded = ProgressReportingPPO.load(tmp_path / "plain", device="cpu")
    assert isinstance(loaded, ProgressReportingPPO)
    assert getattr(loaded, "_progress_sink", None) is None
    # A loaded model can be re-saved (still a valid, functional model object).
    loaded.save(tmp_path / "plain_resaved")
    assert (tmp_path / "plain_resaved.zip").is_file()


# --------------------------------------------------------------------------- #
# AC7 — atomic temp+rename lands the final .zip and leaves no stray temp
# --------------------------------------------------------------------------- #
def test_ac7_atomic_save_lands_final_zip_no_stray_temp(tiny_model, tmp_path) -> None:
    tiny_model.save(tmp_path / "ckpt")

    # The SB3-resolved path (extensionless target normalized to .zip) is the ONLY artifact.
    assert (tmp_path / "ckpt.zip").is_file()
    assert _temp_leftovers(tmp_path) == []  # no *.tmp left behind by the temp+rename
    # exactly one artifact in the dir (no double-extension e.g. ckpt.tmp.zip)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.zip"]

    # The final file is a valid, loadable archive.
    loaded = ProgressReportingPPO.load(tmp_path / "ckpt.zip", device="cpu")
    assert isinstance(loaded, ProgressReportingPPO)


# --------------------------------------------------------------------------- #
# AC8 — extension normalization + buffer passthrough
# --------------------------------------------------------------------------- #
def test_ac8_extension_normalization_same_target(tiny_model, tmp_path) -> None:
    """``save("ckpt")`` and ``save("ckpt.zip")`` resolve to the SAME file (SB3's .zip norm)."""
    tiny_model.save(tmp_path / "ckpt")  # -> ckpt.zip
    tiny_model.save(tmp_path / "ckpt.zip")  # -> ckpt.zip (same)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.zip"]
    assert _temp_leftovers(tmp_path) == []


def test_ac8_buffer_target_delegates_to_super_no_rename(tiny_model, tmp_path, monkeypatch) -> None:
    """A file-like/buffer target delegates straight to ``super`` — round-trips, never renames."""
    # Assert the atomic path is NOT taken for a buffer: os.replace must never be called.
    import drone_fly.train.progress_ppo as ppo_mod

    def _no_replace(*args, **kwargs):  # pragma: no cover - only fires on a regression
        raise AssertionError("os.replace must not run on the buffer passthrough path")

    monkeypatch.setattr(ppo_mod.os, "replace", _no_replace)

    buf = io.BytesIO()
    tiny_model.save(buf)  # must not raise, must not attempt a rename
    assert buf.getbuffer().nbytes > 0

    buf.seek(0)
    loaded = ProgressReportingPPO.load(buf, device="cpu")
    assert isinstance(loaded, ProgressReportingPPO)
    # Buffer path touched no filesystem temp in the working dir.
    assert _temp_leftovers(tmp_path) == []


# --------------------------------------------------------------------------- #
# AC9 (key) — a mid-save failure leaves the prior checkpoint byte-intact
# --------------------------------------------------------------------------- #
def test_ac9_failed_save_preserves_prior_checkpoint(tiny_model, tmp_path, monkeypatch) -> None:
    """Simulate a crash AFTER the temp is written but BEFORE the atomic publish.

    A valid checkpoint already exists at the target; the save then fails at ``os.replace``.
    The pre-existing checkpoint must be byte-unchanged and still loadable, and no temp may leak.
    """
    target = tmp_path / "ckpt"
    # 1) Lay down a valid pre-existing checkpoint and snapshot its bytes.
    tiny_model.save(target)
    final = tmp_path / "ckpt.zip"
    assert final.is_file()
    original_bytes = final.read_bytes()

    # 2) Break the terminal publish: os.replace raises *after* the temp has been serialized.
    import drone_fly.train.progress_ppo as ppo_mod

    def _boom_replace(src, dst):
        raise OSError("simulated mid-save failure before atomic publish")

    monkeypatch.setattr(ppo_mod.os, "replace", _boom_replace)

    # 3) The save must surface the failure.
    with pytest.raises(OSError, match="simulated mid-save failure"):
        tiny_model.save(target)

    # 4) The pre-existing checkpoint is untouched, byte-for-byte, and still loads.
    assert final.read_bytes() == original_bytes
    reloaded = ProgressReportingPPO.load(final, device="cpu")
    assert isinstance(reloaded, ProgressReportingPPO)

    # 5) The failed attempt left no stray temp file behind (except/cleanup unlinked it).
    assert _temp_leftovers(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt.zip"]
