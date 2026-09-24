"""UC-51 (AC4/AC5) — airborne-spawn curriculum shape and monotonicity on the NEW defaults.

UC-51 restaggered the training curricula so floor-takeoff is the LAST, isolated difficulty step.
UC-58 then **retired the collision-penalty curriculum entirely** (the whole training-time
collision-penalty machinery — ``collision_penalty_at`` and ``drone_fly.train.collision_curriculum``
— was removed once the reward redesign fixed the early-termination trap at the source). UC-55 had
already retired the attitude-authority curriculum. So the ONLY curriculum that remains is the
**airborne-spawn reverse curriculum** (UC-44), and this module now pins that surviving schedule's
SHAPE — endpoints, monotonicity, and the late-tail floor-takeoff — as a pure-function property of
the default :class:`TrainConfig`.

Per AC5 it asserts the shape (monotonicity + endpoints + "floor only at the very end"), NOT the
exact fraction values (those are advisory and may be tuned). Fully hermetic: the schedule is a pure
function of ``num_timesteps`` (no pybullet/GPU/env).
"""

from __future__ import annotations

import pytest

from drone_fly.train.airborne_curriculum import spawn_z_at
from drone_fly.train.config import TrainConfig

# Airborne endpoints: the training loop derives high_z = floor_z + altitude_target (UC-58) at wire
# time; the concrete values are irrelevant to the SHAPE assertions, so use the same floor/high band
# the airborne unit tests use.
_FLOOR_Z = 0.0
_HIGH_Z = 1.0


def _spawn(t: int, cfg: TrainConfig) -> float:
    return spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)


def _samples(cfg: TrainConfig, n: int = 2001) -> list[int]:
    """``n`` evenly spaced timesteps across ``[0, total_timesteps]`` (inclusive)."""
    total = cfg.total_timesteps
    return [round(i * total / (n - 1)) for i in range(n)]


def test_default_airborne_schedule_starts_airborne_and_is_monotone_non_increasing() -> None:
    """AC5: the airborne spawn stays at its airborne (high) start value throughout the warmup hold
    and is monotone NON-INCREASING across the run — no re-ascent, a clean reverse curriculum."""
    cfg = TrainConfig()
    assert _spawn(0, cfg) == pytest.approx(_HIGH_Z)  # airborne at the start
    vals = [_spawn(t, cfg) for t in _samples(cfg)]
    for a, b in zip(vals, vals[1:], strict=False):
        assert b <= a + 1e-12, "spawn schedule must be monotone non-increasing"


def test_default_airborne_schedule_reaches_floor_only_at_the_very_end() -> None:
    """AC4/AC5: floor-takeoff is the LAST, isolated difficulty step — the spawn is still fully
    airborne past the midpoint (the descent is a late tail) and reaches the floor only at the
    terminal timestep."""
    cfg = TrainConfig()
    assert _spawn(cfg.total_timesteps // 2, cfg) == pytest.approx(_HIGH_Z)  # still airborne at mid
    assert _spawn(cfg.total_timesteps, cfg) == _FLOOR_Z  # exact floor at the terminal state


def test_default_airborne_schedule_is_composition_valid() -> None:
    """AC4 guardrail: the default airborne schedule is composition-valid (warmup ≤ anneal) and does
    not raise anywhere across the run — the defence-in-depth backstop stays silent on defaults."""
    cfg = TrainConfig()
    for t in _samples(cfg, n=101):
        _spawn(t, cfg)  # never raises on the defaults
    assert cfg.airborne_curriculum_warmup_fraction <= cfg.airborne_curriculum_anneal_fraction


def test_collision_curriculum_module_is_retired() -> None:
    """AC6 (UC-58): the training-time collision-penalty curriculum module is GONE — importing it
    fails, so no dead schedule code lingers alongside the surviving airborne curriculum."""
    with pytest.raises(ModuleNotFoundError):
        import drone_fly.train.collision_curriculum  # noqa: F401
