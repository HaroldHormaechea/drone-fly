"""UC-87 AC5: the altitude envelope — a configurable ceiling and a floor-margin reward term
(``drone_fly.sim.course.CourseEnv``).

Not one of the eight plan-listed files, added so every acceptance criterion maps to a test. The
ceiling altitude is a config field (``ceiling_z``, default the historical ``CEIL`` = 3.5; UC-87
sets 5.0, AC5 range 4..6 m) driving the ceiling crash bound; the floor-margin shaping term is wired
when ``course.floor`` is on. CPU-hermetic, no connectome.
"""

from __future__ import annotations

import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CEIL, FLOOR_KILL_Z, CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)


def _env(*, ceiling_z=3.5, floor=False, n=4, seed=0, air_z=3.0):
    cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=True,
        ceiling_z=ceiling_z,
        floor=floor,
        floor_grace_s=0.0,  # so the ceiling / floor bounds are live from the first step
        pool_size=32,
        episode_s=28.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(air_z, air_z)),
    )
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(50.0), 0.0,
        action_spec=V2,
    )  # fmt: skip


def _hover(n):
    return torch.zeros(n, 4, dtype=torch.float64)


def test_default_ceiling_matches_the_historical_constant():
    assert _env().ceil_z == CEIL == 3.5


def test_ceiling_z_is_configurable_in_the_ac5_range():
    for z in (4.0, 5.0, 6.0):
        assert _env(ceiling_z=z).ceil_z == z


def test_drone_above_the_ceiling_crashes_on_the_ceiling():
    env = _env(ceiling_z=2.0, n=4)
    env.pos = torch.tensor([[0.0, 0.0, 5.0]]).expand(4, 3).clone()  # above the 2 m ceiling
    _, _, done, _ = env.step(_hover(4))
    assert done.all() and env.cause_ceil.all()


def test_a_higher_ceiling_keeps_the_same_drone_in_the_fly_zone():
    env = _env(ceiling_z=6.0, n=4)
    env.pos = torch.tensor([[0.0, 0.0, 5.0]]).expand(4, 3).clone()  # below the 6 m ceiling
    _, _, done, _ = env.step(_hover(4))
    assert not env.cause_ceil.any()


def test_floor_kill_bound_still_fires_below_the_floor():
    env = _env(n=4)
    env.pos = torch.tensor([[0.0, 0.0, FLOOR_KILL_Z - 0.02]]).expand(4, 3).clone()
    _, _, done, _ = env.step(_hover(4))
    assert done.all() and env.cause_floor.all()


def test_floor_margin_shaping_term_changes_the_reward_when_enabled():
    """With ``floor`` on, a drone within the floor-margin band gets the extra shaping term, so its
    reward differs from the otherwise-identical no-floor env (same seed ⇒ identical dynamics)."""
    base = _env(floor=False, n=4, air_z=0.3)  # spawn inside the floor-margin cap
    shaped = _env(floor=True, n=4, air_z=0.3)
    a = torch.zeros(4, 4, dtype=torch.float64)
    a[:, 0] = 0.2  # a gentle climb so the floor margin changes between steps
    _, r_base, _, _ = base.step(a.clone())
    _, r_shaped, _, _ = shaped.step(a.clone())
    assert not torch.allclose(r_base, r_shaped)
