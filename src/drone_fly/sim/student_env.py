"""The student's training env (UC70 D5): :class:`StudentCourseEnv`.

A :class:`~drone_fly.sim.multi_course.MultiCourseEnv` (the teacher's env, unchanged) plus the
simulated perception (:class:`~drone_fly.sim.perception.GatePerception`) and the observation
history. Each step, in order:

1. decode, 2. physics, 3. truth route progress (all inside ``MultiCourseEnv.step``), then via its
   ``terminate`` hook: 4. push the pose into the perception ring, 5. the perception update,
   6. ``lost`` -> ``gate_lost``; back in the env: 7. terminals + the episode log, 8. respawn;
   then here: 9. reset the perception (slots cleared, ring refilled with the spawn pose, dropout /
   AR / frame phase / latency redrawn) and the history of respawned envs, 10. the observations.

Both observations are returned: :meth:`student_obs` (``student-obs`` v1, what the harness can
build) and :meth:`teacher_obs` (the exact ``teacher-obs`` v1 the frozen teacher labels from).

Coverage strata per course (for the report): ``covered`` (every passage is an opening of a class
the noise model measured), ``proxy`` (some opening of another / unknown class: the measured
class's model stands in, optimistic) and ``invisible`` (a volume checkpoint — nothing to see; the
proxy model stands in). ``uncovered: exclude`` drops proxy / invisible courses from the pool
(:func:`course_coverage` + the pool filter in ``training.distill``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from drone_fly.sim.multi_course import MultiCourseEnv
from drone_fly.sim.noise_model import NoiseModel
from drone_fly.sim.perception import GatePerception, PerceptionConfig
from drone_fly.sim.student_obs import HistoryState, build_student_obs

COVERED, PROXY, INVISIBLE = 0, 1, 2
STRATA = ("covered", "proxy", "invisible")
UNCOVERED_POLICIES = ("proxy", "exclude")


def course_coverage(course: Any, classes: list[str] | tuple[str, ...]) -> int:
    """The coverage stratum of one :class:`~drone_fly.sim.pool.SimCourse`."""
    if any(p.kind == "volume" for p in course.passages):
        return INVISIBLE
    if any(p.cls not in classes for p in course.passages):
        return PROXY
    return COVERED


class StudentCourseEnv:
    """See the module docstring. ``k`` = the student's world scale (course units per student
    unit for world-scale-1 courses); ``course_scale`` = metres per course unit, per course."""

    def __init__(
        self,
        env: MultiCourseEnv,
        model: NoiseModel,
        perception: PerceptionConfig,
        *,
        k: float,
        course_scale: Any,
        coverage: Any = None,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.n = env.n
        self.device = env.device
        self.k = float(k)
        self.pcfg = perception
        self.gen = torch.Generator(device=env.device).manual_seed(int(seed) + 7919)
        self.perc = GatePerception(model, env.n, env.device, config=perception,
                                   generator=self.gen, dtype=env.dtype)  # fmt: skip
        self.hist = HistoryState.zeros(env.n, env.device, env.dtype)
        self.course_scale = torch.as_tensor(np.asarray(course_scale, np.float64),
                                            device=env.device).to(env.dtype)  # fmt: skip
        cov = np.zeros(env.n_courses, np.int64) if coverage is None else np.asarray(coverage)
        self.coverage = torch.as_tensor(cov, device=env.device)
        self._obs: torch.Tensor | None = None
        self.last_lost = torch.zeros(env.n, dtype=torch.bool, device=env.device)
        #: False for teacher-bound evaluations: perception runs but never ends an episode.
        self.terminate_on_lost = True
        self._reset_perception(torch.ones(env.n, dtype=torch.bool, device=env.device))
        self._obs = self._build_obs(torch.zeros(env.n, dtype=torch.float64, device=env.device))

    # ---- views ----
    @property
    def dtype(self) -> torch.dtype:
        return self.env.dtype

    @property
    def log(self) -> Any:
        return self.env.log

    def scale(self) -> torch.Tensor:
        return self.course_scale[self.env.cid]

    def stratum(self) -> torch.Tensor:
        return self.coverage[self.env.cid]

    # ---- perception plumbing ----
    def _passages(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        e = self.env
        tgt, tvalid = e.target_index(0)
        nxt, nvalid = e.target_index(1)
        return e.passage(tgt), e.passage(nxt), nvalid, tvalid

    def _reset_perception(self, mask: torch.Tensor) -> None:
        e = self.env
        self.perc.reset(mask, e.t, e.pos, e.quat)
        self.hist.reset(mask)

    def _terminate(self, crossed: torch.Tensor) -> torch.Tensor:
        e = self.env
        e_t = e.t
        self.perc.push_pose(e_t, e.pos, e.quat)
        tg, nx, nvalid, tvalid = self._passages()
        self.perc.step(e_t, e.pos, e.quat, tg, nx, nvalid, crossed, self.scale(),
                       target_valid=tvalid)  # fmt: skip
        self.last_lost = self.perc.lost.clone()
        if not self.terminate_on_lost:
            return torch.zeros_like(self.last_lost)
        return self.last_lost

    def _build_obs(self, dt: torch.Tensor) -> torch.Tensor:
        e = self.env
        out = self.perc.outputs(e.t)
        arm = (e.arm_until - e.t).clamp(min=0.0)
        return build_student_obs(
            pos=e.pos, vel=e.vel, quat=e.quat, omega=e.omega, perc=out, k=self.k,
            nominal_aperture=self.pcfg.nominal_aperture, lost_timeout_s=self.pcfg.lost_timeout_s,
            prev_action=e.prev_action, arm_remaining_s=arm, history=self.hist, dt=dt,
            battery=e.battery(),
        )  # fmt: skip

    # ---- the env API ----
    def reset_all(self, course_ids: Any = None) -> None:
        self.env.reset_all(course_ids)
        self._reset_perception(torch.ones(self.n, dtype=torch.bool, device=self.device))
        self._obs = self._build_obs(torch.zeros(self.n, dtype=torch.float64, device=self.device))

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reward, done = self.env.step(action, terminate=self._terminate)
        if bool(done.any()):
            self._reset_perception(done)
        self._obs = self._build_obs(self.env.last_dt)
        return reward, done

    def student_obs(self) -> torch.Tensor:
        assert self._obs is not None
        return self._obs

    def teacher_obs(self) -> torch.Tensor:
        return self.env.teacher_obs()

    def critic_extra(self) -> torch.Tensor:
        return self.env.critic_extra()

    def episode_rows(self) -> np.ndarray:
        return self.env.episode_rows()

    def perception_outputs(self) -> dict[str, torch.Tensor]:
        return self.perc.outputs(self.env.t)
