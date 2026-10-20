"""UC-90: the gate-approach altitude-tracking reward (``CourseEnv.step_command``).

A new dense term pulls the drone's altitude toward the **current target gate's** z as it closes
in horizontally, so it levels off at gate height instead of climbing over the gate (live, the
UC-89 deploy policy flew a median +11 m above gate 0 -> 0/33). The term is::

    reward -= gate_alt_weight * prox(h) * |drone_z - gate_z| * rscale

with ``prox(h) = clamp(1 - h / gate_alt_radius_m, 0, 1)`` the horizontal-proximity gate,
``h`` the horizontal distance to the current target gate, and ``rscale = dt / reward_reference_dt_s``
the same rate-scaling the other dense terms use. It is stateless and guarded by
``if gate_alt_weight > 0`` ⇒ default-off is byte-identical.

CPU-hermetic, mirroring ``tests/test_reward_penalties.py``: two otherwise-identical envs (same seed
⇒ identical dynamics; the reward never feeds back into the plant) are stepped with the same action,
so the per-step reward DELTA isolates the altitude term exactly. Expected values are computed from
the *post-step* env state (``pos`` / ``_target_pos()`` / ``last_dt``), which is identical in both
envs, so the formula assertion is exact regardless of the small intra-step plant drift.

Acceptance criteria covered (AC1/AC2/AC5/AC6; AC3/AC4/AC7 are orchestrator/deferred — see the use
case). The AC->test map is printed in the QA summary.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02


def test_worktree_code_under_test() -> None:
    """Guard: we must be exercising the worktree's edited ``course.py`` (the editable install
    points at the MAIN checkout, which has no altitude term)."""
    assert hasattr(CourseConfig, "gate_alt_weight"), "testing stale course.py (not the worktree)"
    cc = CourseConfig(reward_reference_dt_s=REF_DT)
    assert cc.gate_alt_weight == 0.0 and cc.gate_alt_radius_m == 3.0


def _env(
    *,
    w: float = 0.0,
    radius: float = 3.0,
    gf: float = 0.0,
    sd: tuple[float, float] | None = None,
    n: int = 8,
    seed: int = 0,
    hz: float = 50.0,
    set_default_field: bool = True,
) -> CourseEnv:
    """A small no-obstacle env. ``set_default_field=False`` constructs the config WITHOUT passing
    ``gate_alt_weight`` (field left at its dataclass default) — used by the byte-identity test."""
    common = dict(
        reward_reference_dt_s=REF_DT,
        no_obs=True,
        pool_size=32,
        episode_s=28.0,
        ceiling_z=50.0,
        gate_alt_radius_m=radius,
        spawn_distance_m=sd,
        start=StartConfig(mode="mixed", ground_fraction=gf, air_z_m=(3.0, 3.0)),
    )
    cfg = CourseConfig(**common) if not set_default_field else CourseConfig(gate_alt_weight=w, **common)
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(hz), 0.0,
        action_spec=V2, log_capacity=None,
    )  # fmt: skip


def _place(env: CourseEnv, *, dx: float = 0.0, dy: float = 0.0, dz: torch.Tensor | float = 0.0) -> None:
    """Put the drone at the current target gate's xy + (dx, dy) and z + ``dz`` (``dz`` may be a
    per-env tensor), so horizontal distance and vertical error to the gate are controlled."""
    gp = env._target_pos().clone()
    p = gp.clone()
    p[:, 0] += dx
    p[:, 1] += dy
    p[:, 2] = p[:, 2] + (dz if isinstance(dz, torch.Tensor) else float(dz))
    env.pos = p


def _expected_pen(env: CourseEnv, w: float, radius: float) -> torch.Tensor:
    """The altitude penalty the formula predicts from ``env``'s POST-step state (float32)."""
    pos = env.pos
    tp = env._target_pos()
    rscale = (env.last_dt / REF_DT).to(pos.dtype)
    h = (tp[:, :2] - pos[:, :2]).norm(dim=-1)
    prox = (1.0 - h / radius).clamp(0.0, 1.0)
    zerr = (pos[:, 2] - tp[:, 2]).abs()
    return w * prox * zerr * rscale


# ---------------------------------------------------------------------------
# Case 1 — default-off byte-identity (AC1/AC5/AC6)
# ---------------------------------------------------------------------------
def test_default_off_is_byte_identical() -> None:
    """``gate_alt_weight=0`` (explicit) vs the field left unset: reward, ``ep_return`` and
    ``extra()`` are bit-for-bit identical across a same-seed multi-step rollout."""
    unset = _env(set_default_field=False)  # CourseConfig built without gate_alt_weight
    zero = _env(w=0.0)  # gate_alt_weight explicitly 0.0
    assert unset.cfg.gate_alt_weight == 0.0  # sanity: the default really is 0
    a = torch.zeros(8, 4)
    a[:, 0] = 0.6  # a little collective so the drones actually fly toward / through gates
    for _ in range(12):
        _, r0, d0, _ = unset.step(a.clone())
        _, r1, d1, _ = zero.step(a.clone())
        assert torch.equal(d0, d1)
        assert torch.equal(r0, r1)  # exact, not approximate
        assert torch.equal(unset.ep_return, zero.ep_return)
        assert torch.equal(unset.extra(), zero.extra())


# ---------------------------------------------------------------------------
# Case 2 — pure penalty, maximal at alignment, grows with vertical error (AC1/AC2/AC6)
# ---------------------------------------------------------------------------
def test_penalty_equals_formula_and_is_maximal_at_gate_altitude() -> None:
    """Two same-seed envs differing only in ``gate_alt_weight``; the per-step reward delta equals
    ``w·prox·zerr·rscale`` exactly. At the gate's z (zerr=0) the contribution is 0 (reward maximal);
    it grows monotonically with the vertical error."""
    w, radius = 0.7, 3.0
    base, pen = _env(w=0.0, radius=radius), _env(w=w, radius=radius)
    # horizontal offset 1.5 (< radius ⇒ prox 0.5, and 3D dist ≥ 1.5 > cap/2 ⇒ no gate pass), with a
    # ladder of vertical errors including exact alignment (zerr=0).
    dz = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    _place(base, dx=1.5, dz=dz)
    _place(pen, dx=1.5, dz=dz)
    a = torch.zeros(8, 4)
    _, rb, _, _ = base.step(a.clone())
    _, rp, _, _ = pen.step(a.clone())
    delta = rb - rp  # base has no term; pen subtracts the penalty ⇒ delta = +penalty
    exp = _expected_pen(pen, w, radius)
    torch.testing.assert_close(delta, exp, rtol=1e-5, atol=1e-6)
    # maximal at alignment: zerr≈0 ⇒ ≈0 contribution
    zerr = (pen.pos[:, 2] - pen._target_pos()[:, 2]).abs()
    assert zerr[0] < 1e-3 and delta[0].abs() < 1e-6
    # monotone non-decreasing in zerr (prox held constant across the batch)
    order = torch.argsort(zerr)
    sorted_delta = delta[order]
    assert torch.all(sorted_delta[1:] >= sorted_delta[:-1] - 1e-6)
    assert delta[-1] > delta[1]  # genuinely varies with zerr


# ---------------------------------------------------------------------------
# Case 3 — proximity gating (AC1/AC6)
# ---------------------------------------------------------------------------
def test_proximity_gating_zero_when_far_active_when_near() -> None:
    """The term is ≈0 when the horizontal distance ≥ ``gate_alt_radius_m`` (even with large zerr),
    and active (= formula) when inside the radius."""
    w, radius = 0.7, 3.0
    # Far: horizontal offset 4.0 > radius ⇒ prox 0, large zerr=3.0 ⇒ still no contribution.
    base_f, pen_f = _env(w=0.0, radius=radius), _env(w=w, radius=radius)
    _place(base_f, dx=4.0, dz=3.0)
    _place(pen_f, dx=4.0, dz=3.0)
    a = torch.zeros(8, 4)
    _, rbf, _, _ = base_f.step(a.clone())
    _, rpf, _, _ = pen_f.step(a.clone())
    h_far = (pen_f._target_pos()[:, :2] - pen_f.pos[:, :2]).norm(dim=-1)
    assert torch.all(h_far >= radius)
    torch.testing.assert_close(rbf - rpf, torch.zeros_like(rbf), rtol=0, atol=1e-6)

    # Near: horizontal offset 1.0 < radius ⇒ prox > 0, contribution matches the formula.
    base_n, pen_n = _env(w=0.0, radius=radius), _env(w=w, radius=radius)
    _place(base_n, dx=1.0, dz=2.0)
    _place(pen_n, dx=1.0, dz=2.0)
    _, rbn, _, _ = base_n.step(a.clone())
    _, rpn, _, _ = pen_n.step(a.clone())
    h_near = (pen_n._target_pos()[:, :2] - pen_n.pos[:, :2]).norm(dim=-1)
    assert torch.all(h_near < radius)
    delta_n = rbn - rpn
    assert torch.all(delta_n > 0)
    torch.testing.assert_close(delta_n, _expected_pen(pen_n, w, radius), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Case 4 — keys on the CURRENT target gate, advancing with the gate index (AC1/AC2)
# ---------------------------------------------------------------------------
def test_keys_on_current_target_gate_after_pass() -> None:
    """After a gate pass (``tgt`` increments) the term tracks the NEW gate's z, not gate 0's —
    proving it generalizes to every gate. We verify the post-pass penalty equals the formula for
    the new target and differs from what gate 0 would have given."""
    w, radius = 0.7, 3.0
    base, pen = _env(w=0.0, radius=radius), _env(w=w, radius=radius)
    gate0 = base._target_pos().clone()
    # Step 1: drive both drones through gate 0 (place AT the centre ⇒ 3D dist 0 < cap/2 ⇒ pass).
    base.pos = gate0.clone()
    pen.pos = gate0.clone()
    a = torch.zeros(8, 4)
    base.step(a.clone())
    pen.step(a.clone())
    assert torch.all(base.tgt == 1) and torch.all(pen.tgt == 1), "gate 0 was not passed"
    gate1 = base._target_pos().clone()
    # Step 2: sit near gate 1 (prox>0, vertical offset ⇒ no re-pass), keyed on the NEW target.
    _place(base, dx=1.0, dz=2.0)
    _place(pen, dx=1.0, dz=2.0)
    _, rb, _, _ = base.step(a.clone())
    _, rp, _, _ = pen.step(a.clone())
    assert torch.all(pen.tgt == 1), "unexpectedly advanced past gate 1"
    delta = rb - rp
    exp_new = _expected_pen(pen, w, radius)  # uses gate 1 (current target)
    torch.testing.assert_close(delta, exp_new, rtol=1e-5, atol=1e-6)
    # Had it (wrongly) kept keying on gate 0, the prediction would differ on the envs where the two
    # gates sit at different altitudes — confirm the test actually discriminates.
    pos = pen.pos
    rscale = (pen.last_dt / REF_DT).to(pos.dtype)
    h0 = (gate0[:, :2] - pos[:, :2]).norm(dim=-1)
    prox0 = (1.0 - h0 / radius).clamp(0.0, 1.0)
    zerr0 = (pos[:, 2] - gate0[:, 2]).abs()
    exp_old = w * prox0 * zerr0 * rscale
    differ = (gate0[:, 2] - gate1[:, 2]).abs() > 0.1
    assert differ.any(), "degenerate pool: gate0.z == gate1.z everywhere"
    assert torch.any((exp_new - exp_old).abs()[differ] > 1e-4)  # the two keyings really diverge


# ---------------------------------------------------------------------------
# Case 5 — grounded-launch non-interference (regression guard, AC3-adjacent)
# ---------------------------------------------------------------------------
def test_grounded_launch_is_not_disturbed() -> None:
    """At a grounded spawn with ``spawn_distance_m`` set so gate 0 is ≥ its lower bound away, and
    ``gate_alt_radius_m`` ≤ that lower bound, the horizontal distance to gate 0 exceeds the radius
    ⇒ prox 0 ⇒ the term contributes ≈0, so it cannot pull the drone off the pad (despite a nonzero
    gate-vs-pad altitude gap)."""
    w, radius = 0.7, 3.0  # radius 3.0 ≤ spawn-distance lower bound 5.0
    base = _env(w=0.0, radius=radius, gf=1.0, sd=(5.0, 6.0))
    pen = _env(w=w, radius=radius, gf=1.0, sd=(5.0, 6.0))
    a = torch.zeros(8, 4)
    _, rb, _, _ = base.step(a.clone())
    _, rp, _, _ = pen.step(a.clone())
    # grounded launch: far from gate 0 horizontally, so the altitude term is gated off ...
    h = (pen._target_pos()[:, :2] - pen.pos[:, :2]).norm(dim=-1)
    assert torch.all(h >= radius)
    # ... even though there is a real vertical gap to the gate (what the term would otherwise bite).
    zerr = (pen.pos[:, 2] - pen._target_pos()[:, 2]).abs()
    assert torch.any(zerr > 0.3)
    torch.testing.assert_close(rb - rp, torch.zeros_like(rb), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Case 6 — config passthrough + validation (AC5)
# ---------------------------------------------------------------------------
def test_from_dict_passthrough() -> None:
    cc = CourseConfig.from_dict(
        {"reward_reference_dt_s": REF_DT, "gate_alt_weight": 0.8, "gate_alt_radius_m": 2.5}
    )
    assert cc.gate_alt_weight == 0.8
    assert cc.gate_alt_radius_m == 2.5
    # defaults survive a dict that omits the fields
    base = CourseConfig.from_dict({"reward_reference_dt_s": REF_DT})
    assert base.gate_alt_weight == 0.0 and base.gate_alt_radius_m == 3.0


def test_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_alt_weight=-0.1)
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_alt_radius_m=0.0)
    with pytest.raises(ValueError):
        CourseConfig(reward_reference_dt_s=REF_DT, gate_alt_radius_m=-1.0)
    # same guards via from_dict
    with pytest.raises(ValueError):
        CourseConfig.from_dict({"reward_reference_dt_s": REF_DT, "gate_alt_weight": -1.0})
    with pytest.raises(ValueError):
        CourseConfig.from_dict({"reward_reference_dt_s": REF_DT, "gate_alt_radius_m": -2.0})
    # a valid boundary: weight 0 is allowed (default-off), radius must be strictly positive
    CourseConfig(reward_reference_dt_s=REF_DT, gate_alt_weight=0.0, gate_alt_radius_m=0.001)
