"""UC-45 — recording fidelity + provenance regression (all hermetic, no pybullet).

Two deliverables, one regression module:

* **(A) Action-channel fidelity** — the training-time :class:`RecordingCallback` records the
  action *actually applied* to ``env.step`` (SB3's ``clipped_actions``), NOT the raw unclipped
  PPO Gaussian sample. The PRIMARY guard drives out-of-box raw actions through the callback and
  asserts the recorder stored the clipped values; a companion assertion proves the clip truly
  discriminates (raw ≠ clipped), so the test would fail on the pre-fix code that logged
  ``actions[0]``. Plus: a fallback path (``clipped_actions`` absent → record ``actions`` + warn),
  per-axis position storage (AC4), and a full-stack lateral-translation check that the training
  env's ``info["position"]`` equals a bare :class:`SimpleDroneAdapter` replay component-wise (AC5).

* **(B) Recording provenance** — :func:`resolve_git_sha` is total (sha / ``-dirty`` / ``"unknown"``
  on any failure), ``meta`` always carries ``git_sha`` and passes a real ``checkpoint`` through,
  ``meta.dynamics`` is presence-guarded (present when ``set_dynamics`` got a ``DynamicsParams``,
  omitted when ``None``), ``RaceEnv.active_dynamics`` reflects the per-episode sampled physics, and
  a resume run stamps the REAL resume-checkpoint path (from-scratch keeps ``"(training)"``).

Hermetic throughout: ``adapter="simple"`` (no pybullet), the committed fixture connectome, and the
git resolver monkeypatched wherever a run would otherwise touch the runner's real git state.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pytest

from drone_fly.adapter import make_adapter
from drone_fly.controller.encoding import ACTION_DIM
from drone_fly.env.config import (
    DynamicsParams,
    EnvConfig,
    EpisodeConfig,
    RandomizationConfig,
)
from drone_fly.env.racing_env import RaceEnv
from drone_fly.record import provenance
from drone_fly.record.provenance import UNKNOWN_GIT_SHA, resolve_git_sha
from drone_fly.record.recorder import ActivationRecorder
from drone_fly.train.record_callback import RecordingCallback

# The training action box: throttle in [0, 1], roll/pitch/yaw in [-1, 1] (RaceEnv.action_space).
_ACTION_LOW = np.array([0.0, -1.0, -1.0, -1.0])
_ACTION_HIGH = np.array([1.0, 1.0, 1.0, 1.0])


# =====================================================================================
# (A) Action-channel fidelity
# =====================================================================================
def _prime_callback(connectome, tmp_path: Path, **recorder_kwargs) -> RecordingCallback:
    """A capturing RecordingCallback whose recorder has an open episode + a primed sink.

    Mirrors the mid-episode state ``_on_step`` runs in: enabled, capturing, one activation
    already sunk (so ``capture_frame`` has a pending frame), no actor wired (we drive
    ``_on_step`` directly rather than through PPO's forward pass).
    """
    recorder = ActivationRecorder(
        connectome, tmp_path / "acts", backend="simple", dt=0.05, **recorder_kwargs
    )
    recorder.start_episode(episode_index=0, seed=0)
    recorder.sink(np.zeros(connectome.neuron_count, dtype=np.float32))
    callback = RecordingCallback(recorder, record_every=1, seed=0)
    callback._enabled = True
    callback._capturing = True
    callback._actor = None
    return callback


def test_recording_callback_records_applied_not_raw_action(connectome, tmp_path: Path) -> None:
    """PRIMARY guard (AC3): the callback records the CLIPPED (applied) action, not the raw sample.

    Drives an out-of-box raw action (throttle 2.6, roll 3.1, pitch -3.1) plus its SB3 clip through
    ``_on_step`` and asserts the recorder stored the clipped ``[1, 1, -1, 0.5]``. The companion
    ``clipped != raw`` assertion proves the clip genuinely discriminates, so the pre-fix code
    (which recorded ``actions[0]``) would fail this test.
    """
    callback = _prime_callback(connectome, tmp_path)

    raw = np.array([[2.6, 3.1, -3.1, 0.5]])  # (1, ACTION_DIM), deliberately outside the box
    clipped = np.clip(raw, _ACTION_LOW, _ACTION_HIGH)
    # Sanity: the clip actually changes the values — otherwise the test could not discriminate.
    assert not np.array_equal(raw[0], clipped[0])
    assert list(clipped[0]) == [1.0, 1.0, -1.0, 0.5]

    callback.locals = {
        "infos": [{"position": np.array([0.1, 0.2, 0.3])}],
        "dones": np.array([False]),
        "rewards": np.array([0.0]),
        "actions": raw,
        "clipped_actions": clipped,
    }
    assert callback._on_step() is True

    stored = callback.recorder._actions
    assert len(stored) == 1
    # Recorded the APPLIED (clipped) action ...
    assert stored[0] == [1.0, 1.0, -1.0, 0.5]
    # ... NOT the raw unclipped sample.
    assert stored[0] != [float(v) for v in raw[0]]


def test_recording_callback_falls_back_to_actions_when_clipped_absent(
    connectome, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Fallback (AC3): with ``clipped_actions`` absent, record ``actions[0]`` and warn once."""
    callback = _prime_callback(connectome, tmp_path)

    raw = np.array([[0.7, -0.4, 0.2, -0.1]])  # in-box; the fallback records it verbatim
    callback.locals = {
        "infos": [{"position": np.array([0.0, 0.0, 1.0])}],
        "dones": np.array([False]),
        "rewards": np.array([0.0]),
        "actions": raw,
        # NOTE: no "clipped_actions" key — forces the fallback branch.
    }
    with caplog.at_level(logging.WARNING, logger="drone_fly.train.record_callback"):
        assert callback._on_step() is True

    stored = callback.recorder._actions
    assert stored[0] == [0.7, -0.4, 0.2, -0.1]
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "clipped_actions absent" in msgs


def test_recorder_stores_position_per_axis_componentwise(connectome, tmp_path: Path) -> None:
    """AC4: the recorder writes the adapter's exact per-step position on all three axes.

    A recorder/env that froze or dropped an axis would fail this — each frame advances on x, y and
    z independently, and the serialised ``frames.drone_position`` must match component-wise.
    """
    rec = ActivationRecorder(connectome, tmp_path / "acts", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    positions = [
        np.array([0.10, 0.50, 1.00]),
        np.array([0.20, 0.55, 1.10]),
        np.array([0.30, 0.60, 1.20]),
        np.array([0.40, 0.65, 1.30]),
    ]
    for pos in positions:
        rec.sink(np.zeros(n, dtype=np.float32))
        rec.capture_frame(np.zeros(ACTION_DIM), pos)
    path = rec.finish_episode(completed=True, completion_time=0.2, total_reward=1.0, steps=4)

    stored = json.loads(path.read_text())["frames"]["drone_position"]
    assert stored == [[float(c) for c in p] for p in positions]
    # Every axis genuinely varies across frames (no frozen/dropped axis).
    arr = np.asarray(stored)
    for axis in range(3):
        assert len(np.unique(arr[:, axis])) == len(positions)


def test_training_env_position_matches_bare_adapter_replay_laterally(tmp_path: Path) -> None:
    """AC5: a laterally-commanded training env translates on x AND y, byte-faithful to the adapter.

    Drives ``RaceEnv(adapter="simple", floor_start=False)`` with a fixed tilted action and compares
    each step's ``info["position"]`` against a bare :class:`SimpleDroneAdapter` replay of the same
    actions. The env does not constrain lateral motion: x and y both strictly change, and the
    recorded-position source equals the true adapter state component-wise.
    """
    cfg = EnvConfig(floor_start=False)
    env = RaceEnv(cfg, adapter="simple")
    course = env.config.course

    bare = make_adapter(
        "simple",
        course.start,
        floor_z=course.floor_z,
        ceiling_z=course.ceiling_z,
        dt=cfg.episode.dt,
    )

    _obs, _info = env.reset(seed=0)
    bare_state = bare.reset(seed=0)
    start_pos = bare_state.position.copy()

    action = np.array([0.7, 0.3, 0.3, 0.0])  # throttle + nonzero roll & pitch -> lateral tilt
    env_positions = []
    bare_positions = []
    for _ in range(8):
        _obs, _reward, terminated, truncated, info = env.step(action)
        env_positions.append(info["position"].copy())
        bare_positions.append(bare.step(action).position.copy())
        if terminated or truncated:
            break

    assert env_positions, "env produced no steps"
    for env_pos, bare_pos in zip(env_positions, bare_positions, strict=True):
        assert np.allclose(env_pos, bare_pos), (env_pos, bare_pos)

    last = env_positions[-1]
    assert abs(last[0] - start_pos[0]) > 1e-3, "x did not translate"
    assert abs(last[1] - start_pos[1]) > 1e-3, "y did not translate"


# =====================================================================================
# (B) Recording provenance — resolve_git_sha
# =====================================================================================
class _FakeCompleted:
    """Minimal stand-in for ``subprocess.CompletedProcess`` (returncode + stdout/stderr)."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_resolve_git_sha_returns_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """rc 0 → the trimmed ``git describe`` stdout is returned verbatim."""
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: _FakeCompleted(0, stdout="a1b2c3d\n")
    )
    assert resolve_git_sha() == "a1b2c3d"


def test_resolve_git_sha_marks_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``-dirty`` describe string (uncommitted edits) is passed through unmodified."""
    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: _FakeCompleted(0, stdout="a1b2c3d-dirty\n")
    )
    assert resolve_git_sha() == "a1b2c3d-dirty"


def test_resolve_git_sha_unknown_when_git_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """git not installed (``FileNotFoundError``) → ``"unknown"``, never raises."""

    def _boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", _boom)
    assert resolve_git_sha() == UNKNOWN_GIT_SHA


def test_resolve_git_sha_unknown_when_not_a_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero return code (not a git work tree) → ``"unknown"``."""
    monkeypatch.setattr(
        provenance.subprocess,
        "run",
        lambda *a, **k: _FakeCompleted(128, stderr="not a git repository"),
    )
    assert resolve_git_sha() == UNKNOWN_GIT_SHA


def test_resolve_git_sha_unknown_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung git (``TimeoutExpired``) is caught → ``"unknown"`` (recording never stalls)."""

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="git", timeout=2.0)

    monkeypatch.setattr(provenance.subprocess, "run", _timeout)
    assert resolve_git_sha() == UNKNOWN_GIT_SHA


# =====================================================================================
# (B) Recording provenance — enriched meta
# =====================================================================================
def _record_one(connectome, out_dir: Path, **recorder_kwargs) -> dict:
    """Record a tiny 2-frame episode and return the parsed document."""
    rec = ActivationRecorder(connectome, out_dir, backend="simple", dt=0.05, **recorder_kwargs)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    for f in range(2):
        rec.sink(np.zeros(n, dtype=np.float32))
        rec.capture_frame(np.zeros(ACTION_DIM), np.array([float(f), 0.0, 1.0]))
    path = rec.finish_episode(completed=True, completion_time=0.1, total_reward=1.0, steps=2)
    return json.loads(path.read_text())


def test_meta_includes_git_sha_and_checkpoint(connectome, tmp_path: Path) -> None:
    """AC8/AC9: ``meta`` always carries ``git_sha`` and passes a real ``checkpoint`` through."""
    meta = _record_one(connectome, tmp_path / "acts", git_sha="abc1234", checkpoint="/m/model.zip")[
        "meta"
    ]
    assert meta["git_sha"] == "abc1234"
    assert meta["checkpoint"] == "/m/model.zip"


def test_meta_includes_dynamics_when_set(connectome, tmp_path: Path) -> None:
    """AC9: ``set_dynamics(DynamicsParams(...))`` stamps all five resolved physics fields."""
    rec = ActivationRecorder(connectome, tmp_path / "acts", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(episode_index=0, seed=0)
    rec.set_dynamics(
        DynamicsParams(mass=1.3, drag=0.2, max_body_rate=3.5, max_thrust=21.0, latency_steps=2)
    )
    rec.sink(np.zeros(n, dtype=np.float32))
    rec.capture_frame(np.zeros(ACTION_DIM), np.array([0.0, 0.0, 1.0]))
    path = rec.finish_episode(completed=True, completion_time=0.05, total_reward=1.0, steps=1)

    dynamics = json.loads(path.read_text())["meta"]["dynamics"]
    assert dynamics == {
        "mass": 1.3,
        "drag": 0.2,
        "max_thrust": 21.0,
        "max_body_rate": 3.5,
        "latency_steps": 2,
    }


def test_meta_omits_dynamics_when_none(connectome, tmp_path: Path) -> None:
    """AC9 back-compat: no ``set_dynamics`` (randomization off) → no ``meta.dynamics`` key."""
    meta = _record_one(connectome, tmp_path / "acts")["meta"]
    assert "dynamics" not in meta


# =====================================================================================
# (B) Recording provenance — RaceEnv.active_dynamics
# =====================================================================================
def test_active_dynamics_exposed_when_randomization_on() -> None:
    """AC9: dynamics randomization on → ``active_dynamics`` is the sampled ``DynamicsParams``."""
    cfg = EnvConfig(randomization=RandomizationConfig(enable_dynamics=True))
    env = RaceEnv(cfg, adapter="simple")
    env.reset(seed=0)
    assert isinstance(env.active_dynamics, DynamicsParams)


def test_active_dynamics_none_when_off() -> None:
    """AC9: with randomization off (default), ``active_dynamics`` is ``None`` after reset."""
    env = RaceEnv(EnvConfig(), adapter="simple")
    env.reset(seed=0)
    assert env.active_dynamics is None


# =====================================================================================
# (B) Recording provenance — resume stamps a real checkpoint (AC8b)
# =====================================================================================
def test_resume_run_stamps_real_checkpoint(
    connectome, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC8b: a resumed run stamps the REAL resume-checkpoint path; scratch keeps '(training)'.

    Hermetic end-to-end: train from scratch (record on) → the recording stamps ``"(training)"``;
    then ``train(resume=<final.zip>, record=True)`` → the recording stamps that exact path. The git
    resolver is monkeypatched so ``meta.git_sha`` never depends on the runner's real git state.
    """
    import drone_fly.train.loop as loop
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import CHECKPOINT_PREFIX

    # Pin git_sha so the recording is fully hermetic (the resolver is imported inside train()).
    monkeypatch.setattr(provenance, "resolve_git_sha", lambda *a, **k: "testsha")

    def _episode_meta(record_dir: Path) -> dict:
        files = sorted(record_dir.glob("episode_*.json"))
        assert files, f"no recordings written to {record_dir}"
        return json.loads(files[0].read_text())["meta"]

    cfg = TrainConfig(
        models_dir=str(tmp_path / "models"),
        logs_dir=str(tmp_path / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        seed=0,
    )
    # Short mid-air episodes so recordings appear within the tiny training budget.
    ecfg = EnvConfig(episode=EpisodeConfig(max_steps=5), floor_start=False)

    scratch_dir = tmp_path / "scratch_acts"
    loop.train(
        cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        tui=False,
        total_timesteps=160,
        record=True,
        record_dir=str(scratch_dir),
        env_config=ecfg,
    )
    # From-scratch: no checkpoint the run started from → the '(training)' placeholder stands.
    assert _episode_meta(scratch_dir)["checkpoint"] == "(training)"

    ckpt = str(tmp_path / "models" / f"{CHECKPOINT_PREFIX}_final.zip")
    assert Path(ckpt).is_file(), "from-scratch run did not save a final checkpoint"

    resume_dir = tmp_path / "resume_acts"
    loop.train(
        cfg,
        connectome=connectome,
        adapter="simple",
        device="cpu",
        tui=False,
        total_timesteps=160,
        record=True,
        record_dir=str(resume_dir),
        env_config=ecfg,
        resume=ckpt,
    )
    # Resume: the recording pins the REAL weights the run started from, not the placeholder.
    assert _episode_meta(resume_dir)["checkpoint"] == ckpt
