"""UC70 AC2 / AC5: the student's training env (``sim.student_env.StudentCourseEnv``).

* The single ``MultiCourseEnv`` hook: ``step(action, terminate=None)`` is identical to the
  teacher's env, a hook that never fires changes nothing, ``gate_lost`` is APPENDED to
  ``OUTCOMES`` (earlier codes unchanged) and a firing hook ends episodes as ``gate_lost`` with the
  crash penalty.
* No leakage across a respawn: sentinel perception / history state of an env that ends is gone
  (slots cleared, pose ring refilled with the spawn pose, history restarted); other envs keep
  theirs. ``terminate_on_lost`` False runs perception without ending episodes.
* Coverage strata and ``uncovered: exclude``.
Hand-built course (``tests/oracle_support.py``), synthetic noise model, CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.multi_course import (
    EPISODE_COLUMNS,
    OUTCOMES,
    MultiCourseConfig,
    MultiCourseEnv,
)
from drone_fly.sim.perception import PerceptionConfig
from drone_fly.sim.pool import PoolGroup, pack_courses
from drone_fly.sim.student_env import (
    COVERED,
    INVISIBLE,
    PROXY,
    StudentCourseEnv,
    course_coverage,
)
from drone_fly.sim.student_obs import DIM, block_slice
from tests.oracle_support import DRONE, loop_passages, make_course, opening, volume
from tests.student_support import load_model

SPEC = ActionSpec.v2(1.0, 4.0, 3.0)


def make_env(n=8, seed=3, courses=None):
    from drone_fly.drones.store import load

    courses = courses or [make_course(loop_passages(), laps=1, circuit=True, box=False)]
    return MultiCourseEnv(load(DRONE), n, pack_courses(courses),
                          [PoolGroup("g", list(range(len(courses))), 1.0)], device="cpu",
                          seed=seed, config=MultiCourseConfig(reward_reference_dt_s=0.02),
                          action_spec=SPEC, log_capacity=1024)  # fmt: skip


def hover(n, dtype):
    return torch.zeros(n, 4, dtype=dtype)


def senv_of(env, model, **pc):
    cfg = PerceptionConfig.from_model(model, pc.pop("mode", "noise"), **pc)
    return StudentCourseEnv(env, model, cfg, k=1.0, course_scale=[1.0] * env.n_courses, seed=1)


# ----------------------------------------------------------------------------- the hook


def test_outcomes_are_appended_not_reordered():
    assert OUTCOMES[:6] == ("none", "floor", "collision", "out-of-bounds", "timeout", "completed")
    assert OUTCOMES[-1] == "gate_lost" and len(OUTCOMES) == 7


def test_terminate_none_and_a_silent_hook_leave_the_env_identical():
    a, b, c = make_env(), make_env(), make_env()
    for e in (a, b, c):
        e.reset_all()
    g = torch.Generator().manual_seed(0)
    for _ in range(60):
        act = 0.3 * torch.randn(a.n, 4, generator=g, dtype=a.dtype)
        ra, da = a.step(act)
        rb, db = b.step(act, terminate=None)
        rc, dc = c.step(act, terminate=lambda crossed: torch.zeros_like(crossed, dtype=torch.bool))
        for (r, d), e in (((rb, db), b), ((rc, dc), c)):
            assert torch.equal(r, ra) and torch.equal(d, da)
            assert torch.equal(e.pos, a.pos) and torch.equal(e.quat, a.quat)
            assert torch.equal(e.k, a.k) and torch.equal(e.last_outcome, a.last_outcome)
    np.testing.assert_array_equal(a.episode_rows(), c.episode_rows())


def test_a_firing_hook_ends_episodes_as_gate_lost_with_the_crash_penalty():
    a, b = make_env(n=4), make_env(n=4)
    a.reset_all()
    b.reset_all()
    act = hover(4, a.dtype)
    seen = []
    ra, _ = a.step(act)
    rb, db = b.step(act, terminate=lambda crossed: (seen.append(crossed.clone()), torch.ones_like(
        crossed, dtype=torch.bool))[1])  # fmt: skip
    assert len(seen) == 1 and seen[0].dtype == torch.bool
    assert bool(db.all())
    lost = OUTCOMES.index("gate_lost")
    assert (b.last_outcome == lost).all()
    np.testing.assert_allclose((ra - rb).numpy(), b.cfg.rewards.crash, rtol=1e-5)
    rows = b.episode_rows()
    col = EPISODE_COLUMNS.index("outcome")
    assert len(rows) == 4 and np.all(rows[:, col] == lost)


# ----------------------------------------------------------------------------- the student env


def test_student_env_returns_both_observations(tmp_path):
    model = load_model(tmp_path, a=2.0)
    s = senv_of(make_env(n=4), model)
    s.reset_all()
    assert s.student_obs().shape == (4, DIM) and s.teacher_obs().shape[0] == 4
    assert torch.isfinite(s.student_obs()).all()
    r, d = s.step(hover(4, s.dtype))
    assert s.student_obs().shape == (4, DIM) and r.shape == (4,) and d.shape == (4,)


def test_unacquired_perception_ends_episodes_as_gate_lost(tmp_path):
    # never detected: p_det ~ 0 and miss bursts far longer than the run
    model = load_model(tmp_path, a=-30.0, burst_s=100.0, tracker={"acquire_timeout_s": 0.2})
    s = senv_of(make_env(n=6), model)
    s.reset_all()
    lost = OUTCOMES.index("gate_lost")
    ended = torch.zeros(6, dtype=torch.bool)
    outcome = torch.zeros(6, dtype=torch.long)
    for _ in range(40):
        _, d = s.step(hover(6, s.dtype))
        outcome = torch.where(d & ~ended, s.env.last_outcome, outcome)
        ended |= d
    assert bool(ended.all())
    assert (outcome == lost).float().mean() > 0.5, outcome
    # perception keeps running without ending episodes when terminate_on_lost is False
    s2 = senv_of(make_env(n=6), model)
    s2.reset_all()
    s2.terminate_on_lost = False
    for _ in range(40):
        _, d = s2.step(hover(6, s2.dtype))
        assert not bool((s2.env.last_outcome[d] == lost).any())
    assert bool(s2.last_lost.any())


def test_no_leakage_across_a_respawn(tmp_path):
    model = load_model(tmp_path, a=-30.0, burst_s=100.0, tracker={"lost_timeout_s": 0.2})
    n = 4
    s = senv_of(make_env(n=n), model)
    s.reset_all()
    s.step(hover(n, s.dtype))
    p, h = s.perc, s.hist
    big = 999.0
    # sentinels everywhere; env 0 holds a target last seen long ago -> lost -> respawn
    p.t_acq[:] = True
    p.n_known[:] = True
    p.t_world[:] = big
    p.n_world[:] = big
    p.t_seen[:] = s.env.t
    p.t_seen[0] = s.env.t[0] - 10.0
    h.ema_rel[:] = big
    h.ema_act[:] = big
    h.prev_target[:] = big
    _, d = s.step(hover(n, s.dtype))
    assert bool(d[0]) and not bool(d[1:].any())
    assert int(s.env.last_outcome[0]) == OUTCOMES.index("gate_lost")
    out = s.perception_outputs()
    assert not bool(out["acquired"][0]) and bool(out["acquired"][1:].all())
    assert float(p.t_world[0].abs().max()) == 0.0 and not bool(p.n_known[0])
    torch.testing.assert_close(out["target"][0], s.env.pos[0].to(out["target"].dtype))
    # the ring holds only the spawn pose of env 0
    spawn = s.env.pos[0].to(p.ring_pos.dtype)
    assert torch.allclose(p.ring_pos[0], spawn.expand_as(p.ring_pos[0]), atol=1e-6)
    obs = s.student_obs()
    hist = obs[:, block_slice("history")]
    assert float(hist[0].abs().max()) < 100.0  # restarted: no sentinel
    assert float(hist[1:, :7].abs().max()) > 100.0  # the others kept theirs
    assert float(obs[0, block_slice("harness30")][:3].abs().max()) == 0.0  # target = own pos


def test_exact_mode_env_detects_the_first_gate(tmp_path):
    model = load_model(tmp_path, a=-30.0)
    s = senv_of(make_env(n=4), model, mode="exact")
    s.env.route_starts = False
    s.reset_all()
    for _ in range(10):
        s.step(hover(4, s.dtype))
    out = s.perception_outputs()
    tgt, _ = s.env.target_index(0)
    centre = s.env.passage(tgt)["center"]
    acq = out["acquired"]
    assert bool(acq.any())
    torch.testing.assert_close(out["target"][acq], centre[acq].to(out["target"].dtype),
                               atol=1e-4, rtol=0)  # fmt: skip


# ----------------------------------------------------------------------------- coverage


def test_coverage_strata():
    arch = opening((5.0, 0.0, 1.5), 0.0)
    arch.cls = "inflatable_arch"
    other = opening((10.0, 0.0, 1.5), 0.0)  # cls synthetic-square
    covered = make_course([arch], box=False)
    proxy = make_course([arch, other], box=False)
    invisible = make_course([arch, volume((10.0, 0.0, 1.5), 0.0)], box=False)
    cls = ["inflatable_arch"]
    assert course_coverage(covered, cls) == COVERED
    assert course_coverage(proxy, cls) == PROXY
    assert course_coverage(invisible, cls) == INVISIBLE


def test_uncovered_exclude_drops_proxy_and_invisible_courses(tmp_path):
    from drone_fly.harness.errors import HarnessConfigError
    from drone_fly.tracks.loader import LoadRecord, Pool
    from drone_fly.training.distill import coverage_of, filter_uncovered

    model = load_model(tmp_path)
    arch = opening((5.0, 0.0, 1.5), 0.0)
    arch.cls = "inflatable_arch"
    cs = [make_course([arch], box=False, course_id="a"),
          make_course([opening((5.0, 0.0, 1.5), 0.0)], box=False, course_id="b"),
          make_course([arch], box=False, course_id="c")]  # fmt: skip
    pool = Pool(cs, [PoolGroup("x", [0, 1], 1.0), PoolGroup("y", [2], 1.0)], "train", {},
                LoadRecord())  # fmt: skip
    assert coverage_of(pool, model).tolist() == [COVERED, PROXY, COVERED]
    kept = filter_uncovered(pool, model)
    assert [c.meta["id"] for c in kept.courses] == ["a", "c"]
    assert [g.courses for g in kept.groups] == [[0], [1]]
    bad = Pool(cs, [PoolGroup("x", [0], 1.0), PoolGroup("only-proxy", [1], 1.0)], "train", {},
               LoadRecord())  # fmt: skip
    with pytest.raises(HarnessConfigError, match="only-proxy"):
        filter_uncovered(bad, model)
