"""UC-92: vertically-spread gates (altitude-robust courses).

The flat 0–2 training slab is kept as the per-gate BASE height (``cen`` z, byte-identical to today);
a separate per-course TALL target z (``gate_z_tall`` in the course dict, ``czt`` in the pool) is
drawn from the SAME k/phase/znoise around a taller per-course centre, and ``CourseEnv._gate`` lerps
the gate z from base toward tall by the runtime ``vspread_scale`` (ramped 0->1 by the curriculum's
``vspread_frac``). All new knobs default OFF ⇒ no extra RNG, no ``czt`` key, byte-identical world.

CPU-hermetic, mirroring ``tests/test_uc90_gate_altitude.py``. The crux checks:

* **worktree guard** — the edited ``course.py`` is the one under test (the editable install points
  at the MAIN checkout, which lacks these fields).
* **default-off byte-identity** — spread off emits no ``gate_z_tall`` / ``czt`` key, the base slab is
  untouched, and an off env is bit-for-bit identical across a same-seed rollout.
* **feasibility over the FULL shipped 1024-course uc92 pool** — every edge's 3D spacing ∈ [1, 5] at
  five ramp scales {0, .25, .5, .75, 1}, every edge's horizontal chord ≥ 1 (the MIN_SPACING proof),
  and every gate z ∈ [gate_z_min, gate_z_max]. Proves no degenerate rejection-fallback ships.
* **pool shape** — ``czt`` present with spread on / absent off; ``cen`` z stays the base slab.
* **_gate lerp** — z(0)=base, z(1)=czt, z(.5) strictly between; default-off env unaffected.
* **vspread_frac validation** — good pairs coerce to tuples, bad pairs raise, None ⇒ 1.0 ramp.
* **extra_dim / memory_dim invariance** — the new knobs change neither observation width.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig, CourseEnv, StartConfig, build_pool
from drone_fly.sim.course_gen import GATE_Z_MAX, MAX_SPACING, MIN_SPACING, sample_course
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from drone_fly.training.config import CurriculumConfig, TrainConfigError
from drone_fly.training.ppo import _ramp_between
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)

# The PRIMARY uc92 config's course knobs (uc92-altitude-robust-warmstart-m13.yaml). The feasibility
# test exercises the EXACT band that ships.
UC92_CENTER_RANGE = (1.0, 3.8)
UC92_AMP = 1.1
UC92_ZMAX = 5.0
UC92_ZMIN = 0.0
UC92_KNOBS = dict(
    gate_z_center_range=UC92_CENTER_RANGE,
    gate_z_amp=UC92_AMP,
    gate_z_max=UC92_ZMAX,
    gate_z_min=UC92_ZMIN,
)


# ---------------------------------------------------------------------------
# Worktree guard
# ---------------------------------------------------------------------------
def test_worktree_code_under_test() -> None:
    """We must be exercising the worktree's edited ``course.py`` (the editable install points at the
    MAIN checkout, which has none of the UC-92 vertical-spread fields)."""
    assert hasattr(CourseConfig, "gate_z_center_range"), "testing stale course.py (not the worktree)"
    cc = CourseConfig(reward_reference_dt_s=REF_DT)
    # Defaults keep today's behaviour exactly: spread OFF, base band [0, 2], amp 0.98.
    assert cc.gate_z_center_range is None
    assert cc.gate_z_amp == 0.98
    assert cc.gate_z_max == 2.0 and cc.gate_z_min == 0.0


def _cfg(**kw: object) -> CourseConfig:
    """A small no-obstacle CPU course config."""
    common = dict(
        reward_reference_dt_s=REF_DT,
        no_obs=True,
        pool_size=48,
        episode_s=28.0,
        ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
    )
    common.update(kw)
    return CourseConfig(**common)


def _env(cfg: CourseConfig, *, n: int = 8, seed: int = 0, hz: float = 50.0) -> CourseEnv:
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(hz), 0.0,
        action_spec=V2, log_capacity=None,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Case 1 — default-off byte-identity
# ---------------------------------------------------------------------------
def test_sample_course_off_emits_no_tall_key() -> None:
    """With the spread knobs absent, ``sample_course`` draws no extra RNG and emits no
    ``gate_z_tall`` key; repeated calls are identical and base gate z stays in the [0, 2] slab."""
    c = sample_course(123)
    assert "gate_z_tall" not in c
    c2 = sample_course(123)
    assert c == c2  # fully reproducible, no divergent RNG
    zs = np.array([g["center"][2] for g in c["gates"]])
    assert zs.min() >= 0.0 - 1e-6 and zs.max() <= 2.0 + 1e-6


def test_default_pool_has_no_czt() -> None:
    """A default (spread-off) pool carries no ``czt`` array and an off env reports ``_vspread_on``
    False, so ``_gate`` is byte-identical to today."""
    pool = build_pool(size=16, base=0, no_obs=True)
    assert "czt" not in pool
    env = _env(_cfg())
    assert env._vspread_on is False


def test_off_env_is_byte_identical() -> None:
    """A config built WITHOUT the new fields vs one with ``gate_z_center_range=None`` produce a
    bit-for-bit identical same-seed rollout (reward, return, extra)."""
    unset = _env(_cfg())  # fields left at defaults
    explicit = _env(_cfg(gate_z_center_range=None))  # spread explicitly off
    assert unset._vspread_on is False and explicit._vspread_on is False
    a = torch.zeros(8, 4)
    a[:, 0] = 0.6  # a little collective so they actually fly toward the gates
    for _ in range(10):
        _, r0, d0, _ = unset.step(a.clone())
        _, r1, d1, _ = explicit.step(a.clone())
        assert torch.equal(d0, d1)
        assert torch.equal(r0, r1)
        assert torch.equal(unset.ep_return, explicit.ep_return)
        assert torch.equal(unset.extra(), explicit.extra())


# ---------------------------------------------------------------------------
# Case 2 — feasibility over the FULL shipped 1024-course uc92 pool
# ---------------------------------------------------------------------------
def test_full_uc92_pool_is_flyable_across_the_whole_ramp() -> None:
    """Build the FULL 1024-course uc92 pool (the exact knobs the PRIMARY config ships) and assert,
    for every course and every gate edge:

    * horizontal chord ``chord_xy`` ≥ MIN_SPACING — the rigorous MIN proof (3D ≥ chord_xy at any
      scale), independent of the vertical spread;
    * 3D spacing ∈ [MIN_SPACING, MAX_SPACING] at EVERY ramp scale {0, .25, .5, .75, 1} — the convex
      endpoint bound made concrete at interior samples too;
    * every gate z ∈ [gate_z_min, gate_z_max].

    Passing over all 1024 courses proves no degenerate rejection-fallback course ships."""
    cfg = CourseConfig(reward_reference_dt_s=REF_DT, **UC92_KNOBS)
    assert cfg.pool_size == 1024  # the shipped default — the FULL pool
    pool = build_pool(cfg.pool_size, cfg.pool_base, cfg.no_obs, **cfg.pool_kwargs)
    assert "czt" in pool, "spread-on pool must carry the per-gate tall z"
    cen, czt, ng = pool["cen"], pool["czt"], pool["ng"]
    P = cen.shape[0]
    assert P == 1024

    tol = 1e-4
    worst_min = np.inf
    worst_max = -np.inf
    for p in range(P):
        n = int(ng[p])
        assert n >= 2
        base_z = cen[p, :n, 2].astype(np.float64)
        tall_z = czt[p, :n].astype(np.float64)
        xy = cen[p, :n, :2].astype(np.float64)
        # every gate z (base + tall, and therefore every lerp between them) within the clip band
        assert base_z.min() >= UC92_ZMIN - tol and base_z.max() <= GATE_Z_MAX + tol
        assert tall_z.min() >= UC92_ZMIN - tol and tall_z.max() <= UC92_ZMAX + tol
        for i in range(n):
            j = (i + 1) % n
            chord_xy = float(np.linalg.norm(xy[j] - xy[i]))
            # MIN proof: 3D spacing ≥ chord_xy at every scale, so chord_xy ≥ 1 guarantees the lower
            # bound for the whole ramp.
            assert chord_xy >= MIN_SPACING - tol, f"course {p} edge {i}: chord_xy {chord_xy} < 1"
            for s in SCALES:
                zi = base_z[i] + s * (tall_z[i] - base_z[i])
                zj = base_z[j] + s * (tall_z[j] - base_z[j])
                d3 = float(np.hypot(chord_xy, zj - zi))
                worst_min = min(worst_min, d3)
                worst_max = max(worst_max, d3)
                assert MIN_SPACING - tol <= d3 <= MAX_SPACING + tol, (
                    f"course {p} edge {i} scale {s}: 3D spacing {d3} out of [1, 5]"
                )
    # sanity: the band is genuinely exercised (not a vacuous all-equal pool)
    assert worst_min < worst_max
    assert worst_max > MIN_SPACING  # real spread, not all-degenerate-minimal


def test_tall_band_actually_rises_above_the_base_slab() -> None:
    """The TALL band must reach heights the base 0–2 slab cannot — otherwise the retrain would see
    no new altitudes. Over the shipped pool, some gate's tall z exceeds the base ceiling (2.0)."""
    cfg = CourseConfig(reward_reference_dt_s=REF_DT, **UC92_KNOBS)
    pool = build_pool(cfg.pool_size, cfg.pool_base, cfg.no_obs, **cfg.pool_kwargs)
    cen, czt, ng = pool["cen"], pool["czt"], pool["ng"]
    assert cen[..., 2].max() <= 2.0 + 1e-4, "base cen z must stay the [0, 2] slab"
    # at least one course lifts a gate above the base slab (the whole point of the UC)
    rose = False
    for p in range(cen.shape[0]):
        n = int(ng[p])
        if czt[p, :n].max() > 2.0 + 0.1:
            rose = True
            break
    assert rose, "tall band never exceeds the base slab — spread is inert"


# ---------------------------------------------------------------------------
# Case 3 — _gate lerp: endpoints + monotonicity
# ---------------------------------------------------------------------------
def test_gate_lerp_endpoints_and_monotonicity() -> None:
    """``_gate`` lerps the gate z from the base slab (scale 0) to the tall band (scale 1), with the
    half-way scale strictly between wherever the two differ. The xy channels are unchanged."""
    env = _env(_cfg(**UC92_KNOBS))
    assert env._vspread_on is True
    idx = torch.zeros(env.n, dtype=torch.long)
    base = env.P["cen"][env.cid, idx]
    czt = env.P["czt"][env.cid, idx]

    env.vspread_scale = 0.0
    z0 = env._gate(idx)[0][..., 2]
    env.vspread_scale = 1.0
    z1 = env._gate(idx)[0][..., 2]
    env.vspread_scale = 0.5
    cen_half, _, _ = env._gate(idx)
    zhalf = cen_half[..., 2]

    torch.testing.assert_close(z0, base[..., 2], rtol=0, atol=1e-6)  # scale 0 ⇒ base
    torch.testing.assert_close(z1, czt, rtol=0, atol=1e-6)  # scale 1 ⇒ tall
    torch.testing.assert_close(zhalf, base[..., 2] + 0.5 * (czt - base[..., 2]), rtol=0, atol=1e-6)
    # xy unaffected by the vertical lerp
    torch.testing.assert_close(cen_half[..., :2], base[..., :2], rtol=0, atol=1e-6)

    differ = (czt - base[..., 2]).abs() > 1e-4
    assert differ.any(), "degenerate pool: tall == base everywhere (test would be vacuous)"
    # strictly between the endpoints wherever the gate actually moves
    lo = torch.minimum(z0, z1)
    hi = torch.maximum(z0, z1)
    assert torch.all(zhalf[differ] > lo[differ] + 1e-6)
    assert torch.all(zhalf[differ] < hi[differ] - 1e-6)


def test_off_gate_is_scale_invariant() -> None:
    """A spread-off env ignores ``vspread_scale`` entirely — ``_gate`` returns the base z at any
    scale (the default path is untouched)."""
    env = _env(_cfg())  # spread off
    assert env._vspread_on is False
    idx = torch.zeros(env.n, dtype=torch.long)
    env.vspread_scale = 0.0
    z0 = env._gate(idx)[0][..., 2].clone()
    env.vspread_scale = 1.0
    z1 = env._gate(idx)[0][..., 2].clone()
    torch.testing.assert_close(z0, z1, rtol=0, atol=0)
    torch.testing.assert_close(z0, env.P["cen"][env.cid, idx][..., 2], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Case 4 — CourseConfig validation + from_dict coercion
# ---------------------------------------------------------------------------
def test_course_config_from_dict_coerces_and_validates() -> None:
    cc = CourseConfig.from_dict(
        {
            "reward_reference_dt_s": REF_DT,
            "gate_z_center_range": [1.0, 3.8],  # a YAML list ...
            "gate_z_amp": 1.1,
            "gate_z_max": 5.0,
            "gate_z_min": 0.0,
        }
    )
    assert cc.gate_z_center_range == (1.0, 3.8)  # ... coerced to a hashable tuple
    assert isinstance(cc.gate_z_center_range, tuple)
    # the tuple must be hashable for the pool cache key (mirrors spawn_distance_m)
    hash(tuple(sorted(cc.pool_kwargs.items())))
    # defaults survive a dict that omits the fields
    base = CourseConfig.from_dict({"reward_reference_dt_s": REF_DT})
    assert base.gate_z_center_range is None and base.gate_z_amp == 0.98


def test_course_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_z_amp=-0.1)
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_z_min=3.0, gate_z_max=2.0)
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_z_center_range=(3.0, 1.0))  # lo > hi
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_z_center_range=(-1.0, 2.0))  # lo < 0


# ---------------------------------------------------------------------------
# Case 5 — CurriculumConfig.vspread_frac validation + ramp semantics
# ---------------------------------------------------------------------------
def test_vspread_frac_validates_and_coerces() -> None:
    cur = CurriculumConfig(vspread_frac=[0.1, 0.7])
    assert cur.vspread_frac == (0.1, 0.7)
    assert isinstance(cur.vspread_frac, tuple)
    assert CurriculumConfig(vspread_frac=None).vspread_frac is None


@pytest.mark.parametrize(
    "bad",
    [
        [0.7, 0.1],   # start > end
        [-0.1, 0.5],  # start < 0
        [0.5, 1.5],   # end > 1
        [0.5],        # wrong length
        [0.1, 0.5, 0.9],
    ],
)
def test_vspread_frac_rejects_bad_pairs(bad: list) -> None:
    with pytest.raises(TrainConfigError):
        CurriculumConfig(vspread_frac=bad)


def test_vspread_ramp_semantics() -> None:
    """``_ramp_between`` drives ``vspread_scale``: None ⇒ 1.0 (full tall), otherwise 0 before start,
    linear between, 1 after end."""
    assert _ramp_between(0.0, None) == 1.0
    assert _ramp_between(0.9, None) == 1.0
    frac = (0.1, 0.7)
    assert _ramp_between(0.05, frac) == 0.0  # before start
    assert _ramp_between(0.7, frac) == pytest.approx(1.0)  # at end
    assert _ramp_between(0.9, frac) == 1.0  # after end
    mid = _ramp_between(0.4, frac)  # (0.4-0.1)/(0.7-0.1) = 0.5
    assert mid == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Case 6 — observation width is unchanged by the new knobs
# ---------------------------------------------------------------------------
def test_extra_dim_and_memory_dim_unchanged() -> None:
    """Turning the vertical spread on must not alter the observation layout (the new knobs touch
    course geometry only, never the obs vector)."""
    off = CourseConfig(reward_reference_dt_s=REF_DT)
    on = CourseConfig(reward_reference_dt_s=REF_DT, **UC92_KNOBS)
    assert on.extra_dim == off.extra_dim
    assert on.memory_dim == off.memory_dim
