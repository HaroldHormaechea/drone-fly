"""UC-82 AC5/AC7: the proc-course collision-metric math (``drone_fly.training.proc_eval``).

CPU-only, hermetic, and sim-free: the outcome tally (:func:`_run_outcomes`) is exercised with a
scripted fake env + actor so the arithmetic is pinned without flying a real course, and
:func:`compare_collision` (the AC7 relative-reduction verdict) is pure-dictionary.

The GPU pieces of AC5/AC7 — the actual held-out numbers and the >=20% outcome — need a trained
checkpoint and a GPU run and are NOT unit-testable here (see the coverage table in the QA report).
"""

from __future__ import annotations

import torch

from drone_fly.sim.course import OUTCOMES
from drone_fly.training.proc_eval import (
    DEFAULT_REL_THRESHOLD,
    _run_outcomes,
    compare_collision,
)


class _FakeActor:
    """Deterministic MEAN action; ignores observations."""

    def __init__(self, n, action_dim=4):
        self._mean = torch.zeros(n, action_dim)

    def features(self, obs):
        return obs

    def act(self, feat, ext):
        return self._mean, None


class _FakeEnv:
    """A scripted course: at the first step each env lands on one outcome (or never dones ->
    timeout). Exposes exactly the surface :func:`_run_outcomes` reads."""

    def __init__(self, outcome_by_env):
        # outcome_by_env: list of outcome names ("obstacle"/"arena"/"ceiling"/"floor"/
        # "completed") or None for a timeout env.
        self.n = len(outcome_by_env)
        self._script = outcome_by_env
        self.pos = torch.zeros(self.n, 3)
        self._obs = torch.zeros(self.n, 2)
        self._ext = torch.zeros(self.n, 2)
        self._stepped = False
        self._set_causes(done=torch.zeros(self.n, dtype=torch.bool))

    def _set_causes(self, done):
        def mask(name):
            return torch.tensor([done[i] and self._script[i] == name for i in range(self.n)])

        self.cause_obstacle = mask("obstacle")
        self.cause_arena = mask("arena")
        self.cause_ceil = mask("ceiling")
        self.cause_floor = mask("floor")
        self.last_completed = mask("completed")

    def obs(self):
        return self._obs

    def extra(self):
        return self._ext

    def step(self, action):
        # done only on the first step, and only for envs with a scripted outcome
        if not self._stepped:
            self._stepped = True
            done = torch.tensor([o is not None for o in self._script])
        else:
            done = torch.zeros(self.n, dtype=torch.bool)
        self._set_causes(done)
        return self._obs, torch.zeros(self.n), done, torch.zeros(self.n)


def test_run_outcomes_tally_sums_and_keys():
    script = ["obstacle", "obstacle", "arena", "ceiling", "floor", "completed", None, None]
    env = _FakeEnv(script)
    out = _run_outcomes(env, _FakeActor(env.n), max_steps=3)

    assert out["episodes"] == 8
    assert set(out["outcome_counts"]) == set(OUTCOMES)  # keys == course.OUTCOMES
    assert sum(out["outcome_counts"].values()) == 8  # tally sums to the episode count
    assert out["outcome_counts"]["obstacle"] == 2
    assert out["outcome_counts"]["arena"] == 1
    assert out["outcome_counts"]["ceiling"] == 1
    assert out["outcome_counts"]["floor"] == 1
    assert out["outcome_counts"]["completed"] == 1
    assert out["outcome_counts"]["timeout"] == 2  # the two never-done envs
    assert out["outcome_counts"]["none"] == 0

    assert abs(sum(out["outcome_shares"].values()) - 1.0) < 1e-9  # shares sum to 1
    assert out["collision_share"] == out["outcome_shares"]["obstacle"] == 0.25
    assert out["completion"] == out["outcome_shares"]["completed"] == 0.125


def test_run_outcomes_all_timeout_when_nothing_dones():
    env = _FakeEnv([None, None, None, None])
    out = _run_outcomes(env, _FakeActor(env.n), max_steps=2)
    assert out["outcome_counts"]["timeout"] == 4
    assert out["collision_share"] == 0.0
    assert sum(out["outcome_counts"].values()) == 4


# ----------------------------------------------------------------------------- compare_collision


def test_compare_collision_passes_on_material_reduction():
    rays = {"collision_share": 0.20, "completion": 0.4}
    blank = {"collision_share": 0.50, "completion": 0.1}
    v = compare_collision(rays, blank)
    assert v["rel_threshold"] == DEFAULT_REL_THRESHOLD == 0.20
    assert abs(v["relative_reduction"] - 0.6) < 1e-9  # (0.5 - 0.2) / 0.5
    assert abs(v["absolute_reduction"] - 0.3) < 1e-9
    assert v["passed"] is True
    assert v["verdict"].startswith("PASS")
    assert v["completion_rays"] == 0.4 and v["completion_blank"] == 0.1


def test_compare_collision_fails_below_threshold():
    v = compare_collision({"collision_share": 0.45, "completion": 0.0},
                          {"collision_share": 0.50, "completion": 0.0})  # fmt: skip
    assert abs(v["relative_reduction"] - 0.1) < 1e-9
    assert v["passed"] is False
    assert v["verdict"].startswith("NEGATIVE")


def test_compare_collision_guards_a_zero_blank_control():
    # if the blanked control never collides the ratio is undefined -> None, not a pass.
    v = compare_collision({"collision_share": 0.0, "completion": 0.0},
                          {"collision_share": 0.0, "completion": 0.0})  # fmt: skip
    assert v["relative_reduction"] is None
    assert v["passed"] is False


def test_compare_collision_honours_a_custom_threshold():
    rays = {"collision_share": 0.30, "completion": 0.0}
    blank = {"collision_share": 0.50, "completion": 0.0}  # rel reduction = 0.4
    assert compare_collision(rays, blank, rel_threshold=0.2)["passed"] is True
    assert compare_collision(rays, blank, rel_threshold=0.5)["passed"] is False
