"""UC-51 (AC4/AC5) — cross-curriculum ordering, isolation, shape, and monotonicity on the NEW
restaggered defaults.

The training curricula — collision penalty (UC-41) and airborne spawn (UC-44) — used to anneal
on the same ``anneal_fraction ≈ 0.5`` schedule (alongside the now-retired UC-46 attitude-authority
curriculum), stacking difficulty spikes at the 50% mark (the diagnosed synchronized-cliff wall).
UC-51 restaggers the DEFAULTS so the difficulty steps are isolated and ordered, with floor-takeoff
last:

    collision_penalty_full  <  spawn_reaches_floor

(UC-55 retired the attitude-authority curriculum — the inner-loop rate controller makes it
obsolete — so the ordering it used to lead is gone; the collision→floor ordering it fronted is
unchanged.)

This module pins that ordering/isolation and each schedule's SHAPE (monotonicity + endpoints) as a
pure-function property of the default :class:`TrainConfig`. Per AC5 it asserts the **ordering,
isolation, and monotonicity — NOT the exact fraction values** (those are advisory and may be tuned).
Fully hermetic: the schedules are pure functions of ``num_timesteps`` (no pybullet/GPU/env).
"""

from __future__ import annotations

import pytest

from drone_fly.train.airborne_curriculum import spawn_z_at
from drone_fly.train.collision_curriculum import collision_penalty_at
from drone_fly.train.config import TrainConfig

# Airborne endpoints: the training loop derives high_z = floor_z + climb_target_height at wire time;
# the concrete values are irrelevant to the SHAPE assertions, so use the same floor/high band the
# airborne unit tests use.
_FLOOR_Z = 0.0
_HIGH_Z = 1.0


def _spawn(t: int, cfg: TrainConfig) -> float:
    return spawn_z_at(t, cfg, floor_z=_FLOOR_Z, high_z=_HIGH_Z)


def _samples(cfg: TrainConfig, n: int = 2001) -> list[int]:
    """``n`` evenly spaced timesteps across ``[0, total_timesteps]`` (inclusive)."""
    total = cfg.total_timesteps
    return [round(i * total / (n - 1)) for i in range(n)]


def _first_step_where(cfg: TrainConfig, predicate) -> int:
    """The earliest sampled timestep at which ``predicate(t)`` becomes true (the difficulty step
    "reaches full"). Asserts it happens at all."""
    for t in _samples(cfg):
        if predicate(t):
            return t
    raise AssertionError("predicate never became true across the run")


def test_default_schedule_orders_the_difficulty_steps_isolated() -> None:
    """AC4: on the NEW defaults the two remaining curricula reach their hard endpoint in the
    mandated order — collision penalty FIRST, airborne floor-takeoff LAST — asserting the
    ORDERING (strict ``<``), not the exact fractions."""
    cfg = TrainConfig()
    end = cfg.collision_penalty_end

    t_collision_full = _first_step_where(cfg, lambda t: collision_penalty_at(t, cfg) >= end - 1e-6)
    t_spawn_floor = _first_step_where(cfg, lambda t: _spawn(t, cfg) <= _FLOOR_Z + 1e-9)

    assert t_collision_full < t_spawn_floor, (
        f"expected collision_full ({t_collision_full}) < spawn_floor ({t_spawn_floor})"
    )


def test_default_schedule_steps_are_isolated_not_simultaneous() -> None:
    """AC4/AC5 (the crux — the fix for the synchronized cliff): when the earlier difficulty step
    reaches full, the LATER curriculum is still easy — the spikes do NOT land together.
    Specifically: when the collision penalty reaches full the spawn is STILL fully airborne (its
    descent has not begun)."""
    cfg = TrainConfig()
    end = cfg.collision_penalty_end

    t_collision_full = _first_step_where(cfg, lambda t: collision_penalty_at(t, cfg) >= end - 1e-6)

    # At collision-full, the spawn is STILL fully airborne — floor-takeoff is isolated to the tail.
    assert _spawn(t_collision_full, cfg) == pytest.approx(_HIGH_Z)


def test_default_collision_schedule_shape_reaches_full_by_mid_training() -> None:
    """AC5: the collision penalty starts at its low held value, is monotone NON-DECREASING, reaches
    its end value by roughly mid-training (well before the end), and holds. Asserts the shape +
    "by ≈ mid, not at the very end", not the exact fraction."""
    cfg = TrainConfig()
    end = cfg.collision_penalty_end
    assert collision_penalty_at(0, cfg) == pytest.approx(cfg.collision_penalty_start)
    vals = [collision_penalty_at(t, cfg) for t in _samples(cfg)]
    for a, b in zip(vals, vals[1:], strict=False):
        assert b >= a - 1e-9, "collision penalty must be monotone non-decreasing"
    assert vals[-1] == pytest.approx(end)
    # Reaches full strictly before the final stretch — it is NOT the last difficulty step.
    t_collision_full = _first_step_where(cfg, lambda t: collision_penalty_at(t, cfg) >= end - 1e-6)
    assert t_collision_full < cfg.total_timesteps * 0.75


def test_default_airborne_schedule_shape_holds_then_descends_to_floor_last() -> None:
    """AC5: the airborne spawn stays at its airborne (high) start value throughout the warmup hold,
    is monotone NON-INCREASING, and reaches the floor only at the very end of training (the LAST,
    isolated difficulty step). Asserts the shape, not the exact warmup fraction."""
    cfg = TrainConfig()
    assert _spawn(0, cfg) == pytest.approx(_HIGH_Z)  # airborne at the start
    vals = [_spawn(t, cfg) for t in _samples(cfg)]
    for a, b in zip(vals, vals[1:], strict=False):
        assert b <= a + 1e-12, "spawn schedule must be monotone non-increasing"
    # It is still airborne past the midpoint (the descent is a late tail), and floored at the end.
    assert _spawn(cfg.total_timesteps // 2, cfg) == pytest.approx(_HIGH_Z)
    assert _spawn(cfg.total_timesteps, cfg) == _FLOOR_Z  # exact floor at the terminal state


def test_default_schedules_are_composition_valid() -> None:
    """AC4 guardrail: the default schedule is composition-valid on both curricula that carry a
    two-part window — the airborne warmup does not run past its anneal, and the collision hold+ramp
    fits inside the run — so neither pure schedule raises on the defaults."""
    cfg = TrainConfig()
    # Neither raises across the whole run (defence-in-depth backstops stay silent on defaults).
    for t in _samples(cfg, n=101):
        collision_penalty_at(t, cfg)
        _spawn(t, cfg)
    assert cfg.airborne_curriculum_warmup_fraction <= cfg.airborne_curriculum_anneal_fraction
    assert cfg.collision_curriculum_hold_fraction + cfg.collision_curriculum_warmup_fraction <= 1.0
