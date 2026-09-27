"""UC-61 AC6 — the structured JSONL training-status emitter (``StatusEmitterCallback``).

Hermetic unit tests with a fake SB3 model (no PPO, no env, no GPU). They pin the AC6 contract:

* **One newline-terminated JSON object per rollout**, appended to ``status.jsonl``.
* **Same data sources as the TUI/health callbacks** — rollout stats from the model's episode
  buffers, train stats from ``logger.name_to_value`` (one-iteration-lagged, ``None``-tolerant), the
  health verdict from an injected ``HealthCallback``, and the UC-49 drone-dynamics summary.
* **Every value JSON-safe** — the numpy-``float32`` / non-finite (``nan``/``inf``) regression: a
  status line whose metrics are numpy scalars or ``nan`` must still parse under a strict reader
  (``json.loads`` with the default ``allow_nan`` semantics the browser's ``JSON.parse`` mirrors).
* **Fully defensive** — any write/format error disables the callback and is never propagated into
  ``model.learn``.

Also asserts the CLI wiring (AC6 × AC7 × AC10): ``_run_train`` passes a derived ``status_path`` +
``checkpoint_on_signal=True`` so every ``drone-fly train`` run emits the stream the app tails.
"""

from __future__ import annotations

import json
import os
import types

import pytest

from drone_fly.train.status_emitter import StatusEmitterCallback, _finite


# --- fakes --------------------------------------------------------------------------------


class _FakeVecEnv:
    """Minimal VecEnv stand-in exposing the attributes the dynamics summary reads (env-0)."""

    def __init__(self, *, backend="simple", dynamics=None, spawn_z=0.1, has_spawn=True):
        self._backend = backend
        self._dynamics = dynamics
        self._spawn_z = spawn_z
        self._has_spawn = has_spawn

    def get_attr(self, name):
        if name == "backend":
            return [self._backend]
        if name == "active_dynamics":
            return [self._dynamics]
        if name == "spawn_z":
            if not self._has_spawn:
                raise AttributeError("spawn_z")
            return [self._spawn_z]
        raise AttributeError(name)


def _fake_model(*, ep_info=None, ep_success=None, name_to_value=None, total=100_000, venv=None):
    logger = types.SimpleNamespace(name_to_value=name_to_value or {})
    return types.SimpleNamespace(
        ep_info_buffer=ep_info,
        ep_success_buffer=ep_success,
        logger=logger,
        _total_timesteps=total,
        get_env=(lambda: venv),
    )


def _fake_health(status="normal", message="all clear"):
    verdict = types.SimpleNamespace(status=status, message=message)
    return types.SimpleNamespace(latest_verdict=verdict)


def _fire(cb: StatusEmitterCallback, model, *, timesteps=2048) -> None:
    cb.model = model
    cb.num_timesteps = timesteps
    if not cb._enabled or cb._start_time is None:
        cb._on_training_start()
    cb._on_rollout_end()


def _read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.read().splitlines() if line.strip()]


# --- _finite helper -----------------------------------------------------------------------


def test_finite_coerces_nonfinite_and_numpy_to_json_safe():
    np = pytest.importorskip("numpy")
    assert _finite(float("nan")) is None
    assert _finite(float("inf")) is None
    assert _finite(float("-inf")) is None
    assert _finite(np.float32("nan")) is None
    assert _finite(np.float32(1.5)) == pytest.approx(1.5)
    assert isinstance(_finite(np.int64(7)), int) or _finite(np.int64(7)) == 7
    assert _finite(None) is None
    assert _finite("simple") == "simple"  # non-numeric passes through
    assert _finite(True) is True  # bool preserved (not coerced to 1.0)


# --- one line per rollout + schema --------------------------------------------------------


def test_one_jsonl_line_per_rollout_with_full_schema(tmp_path):
    path = str(tmp_path / "status.jsonl")
    venv = _FakeVecEnv(backend="simple")
    model = _fake_model(
        ep_info=[{"r": -100.0, "l": 200}, {"r": -300.0, "l": 220}],
        ep_success=[0.0, 1.0],
        name_to_value={
            "train/loss": 12.0,
            "train/value_loss": 51.2,
            "train/approx_kl": 0.007,
            "train/entropy_loss": -2.5,
            "train/explained_variance": 0.63,
            "train/std": 0.5,
            "time/fps": 480,
        },
        total=100_000,
        venv=venv,
    )
    cb = StatusEmitterCallback(path, health_cb=_fake_health("warn", "kl high"), tw_preserving=True)
    _fire(cb, model, timesteps=2048)
    _fire(cb, model, timesteps=4096)

    lines = _read_lines(path)
    assert len(lines) == 2  # exactly one line per rollout
    rec = lines[0]
    assert rec["timesteps"] == 2048
    assert rec["target_timesteps"] == 100_000
    assert rec["n_updates"] == 1
    assert rec["rollout"]["ep_rew_mean"] == pytest.approx(-200.0)
    assert rec["rollout"]["ep_len_mean"] == pytest.approx(210.0)
    assert rec["rollout"]["success_rate"] == pytest.approx(0.5)
    assert rec["train"]["approx_kl"] == pytest.approx(0.007)
    assert rec["train"]["explained_variance"] == pytest.approx(0.63)
    assert rec["fps"] == pytest.approx(480)
    assert rec["health"] == {"status": "warn", "message": "kl high"}
    assert rec["dynamics"] is not None
    assert "applied_mass" in rec["dynamics"]
    assert lines[1]["timesteps"] == 4096
    assert lines[1]["n_updates"] == 2  # counter advances per rollout


def test_numpy_and_nonfinite_metrics_still_parse(tmp_path):
    """Regression: numpy-float32 / NaN-ish metrics must serialise to strict, parseable JSON."""
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "status.jsonl")
    model = _fake_model(
        ep_info=[],  # empty buffer → safe_mean == nan → must become JSON null
        ep_success=[],
        name_to_value={
            "train/loss": np.float32(3.14),  # numpy scalar → python float
            "train/approx_kl": np.float32("nan"),  # non-finite → null
            "time/fps": np.int64(512),
        },
        venv=None,  # dynamics best-effort → null
    )
    cb = StatusEmitterCallback(path, health_cb=None, tw_preserving=True)
    _fire(cb, model)

    # The bug class: json.dumps of a numpy float32 / NaN would raise or emit non-strict tokens.
    lines = _read_lines(path)  # this re-parses with strict json.loads — the real assertion
    (rec,) = lines
    assert rec["rollout"]["ep_rew_mean"] is None  # nan → null
    assert rec["train"]["loss"] == pytest.approx(3.14, abs=1e-4)
    assert rec["train"]["approx_kl"] is None  # nan → null
    assert rec["fps"] == 512
    assert rec["health"] is None  # no health callback injected
    assert rec["dynamics"] is None  # no env → best-effort null


def test_missing_train_keys_become_null_not_zero(tmp_path):
    path = str(tmp_path / "status.jsonl")
    model = _fake_model(ep_info=[], ep_success=[], name_to_value={}, venv=None)
    cb = StatusEmitterCallback(path, health_cb=None)
    _fire(cb, model)
    (rec,) = _read_lines(path)
    for key in ("loss", "value_loss", "approx_kl", "entropy_loss", "explained_variance", "std"):
        assert rec["train"][key] is None  # absent metric → null, never a fabricated 0


def test_dynamics_summary_present_when_env_available(tmp_path):
    from drone_fly.env.config import DynamicsParams

    path = str(tmp_path / "status.jsonl")
    venv = _FakeVecEnv(backend="pybullet", dynamics=DynamicsParams(), spawn_z=0.5)
    model = _fake_model(ep_info=[{"r": 1.0, "l": 10}], ep_success=[1.0], venv=venv)
    cb = StatusEmitterCallback(path, health_cb=None, tw_preserving=True)
    _fire(cb, model)
    (rec,) = _read_lines(path)
    assert isinstance(rec["dynamics"], dict)
    assert "applied_mass" in rec["dynamics"] and "thrust_to_weight" in rec["dynamics"]


def test_dynamics_survives_env_without_spawn_z_accessor(tmp_path):
    from drone_fly.env.config import DynamicsParams

    path = str(tmp_path / "status.jsonl")
    venv = _FakeVecEnv(backend="pybullet", dynamics=DynamicsParams(), has_spawn=False)
    model = _fake_model(ep_info=[{"r": 1.0, "l": 10}], ep_success=[1.0], venv=venv)
    cb = StatusEmitterCallback(path, health_cb=None)
    _fire(cb, model)
    (rec,) = _read_lines(path)
    # spawn_z accessor missing is tolerated; the rest of the summary still writes.
    assert isinstance(rec["dynamics"], dict)


# --- defensive contract -------------------------------------------------------------------


def test_write_error_disables_callback_and_never_raises(tmp_path):
    # Point status_path at a *directory* so open(..., "a") raises IsADirectoryError on write.
    bad = tmp_path / "adir"
    bad.mkdir()
    model = _fake_model(ep_info=[{"r": 1.0, "l": 5}], ep_success=[1.0], venv=None)
    cb = StatusEmitterCallback(str(bad), health_cb=None)
    cb.model = model
    cb.num_timesteps = 10
    cb._on_training_start()
    cb._on_rollout_end()  # must NOT raise
    assert cb._enabled is False
    # A subsequent rollout is a silent no-op (still disabled), never raising.
    cb._on_rollout_end()
    assert cb._enabled is False


def test_build_error_disables_but_does_not_propagate(tmp_path):
    path = str(tmp_path / "status.jsonl")
    cb = StatusEmitterCallback(path, health_cb=None)
    # A model missing the expected attributes makes _build_record raise internally; the callback
    # must swallow it, disable, and never propagate into model.learn.
    cb.model = types.SimpleNamespace()  # no ep_info_buffer etc.
    cb.num_timesteps = 0
    cb._on_training_start()
    cb._on_rollout_end()  # must NOT raise
    assert cb._enabled is False


def test_on_step_is_a_noop_true(tmp_path):
    cb = StatusEmitterCallback(str(tmp_path / "s.jsonl"), health_cb=None)
    cb.model = _fake_model(ep_info=[], ep_success=[])
    assert cb._on_step() is True  # the emitter writes on rollout end, never per step


# --- CLI wiring (AC6 × AC7 × AC10): every `drone-fly train` run emits + traps signal ------


def test_run_train_passes_status_path_and_checkpoint_on_signal(tmp_path, monkeypatch):
    import yaml

    import drone_fly.train.loop as loop
    from drone_fly import cli

    captured: dict = {}

    def fake_train(train_cfg, **kw):
        captured.update(kw)
        captured["models_dir"] = train_cfg.models_dir
        return None

    monkeypatch.setattr(loop, "train", fake_train)
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "demo.yaml"
    cfg_path.write_text(yaml.safe_dump({"name": "demo"}), encoding="utf-8")

    rc = cli._run_train(str(cfg_path), no_tui=True)

    assert rc == 0
    # AC6: a derived, run-scoped JSONL status path (training/<name>/status.jsonl).
    assert captured["status_path"] == os.path.join("training", "demo", "status.jsonl")
    # AC7: lossless pause is armed for every CLI run.
    assert captured["checkpoint_on_signal"] is True
    # --no-tui really disables the dashboard (the app launches with --no-tui).
    assert captured["tui"] is False
