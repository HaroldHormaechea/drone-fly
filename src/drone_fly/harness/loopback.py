"""Sim-loopback harness: the contract's reference ``plant_step`` driven through the harness parts.

It proves the harness reproduces the reference inference path (AC4): ground-truth estimators
(gate senses from the known course, rays from the contract's ``raycast()``), a state source that
reads the plant, and a sink that applies ``plant_step`` to the policy's **raw** action. It runs on
a :class:`~drone_fly.harness.runner.SimClock` with no frame source.

A contract-v1 checkpoint is stepped by ``plant_step`` on its raw action (bit-exact). A v2
checkpoint (UC67: hover-relative ``collective_g``) is stepped by a single-env
:class:`drone_fly.sim.Plant` on the decoded command — the config's ``drone`` profile, else the
reference ``drones/meteor75`` — with the ``euler-legacy`` integrator at the contract ``DT``.

Tick order equals the training env's ``step()``: ``plant_step`` at the end of tick ``k``; tick
``k+1`` reads that state, applies the pass rule / advance (wrap + lap), then assembles — so the
target the policy sees is never one step behind.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drone_fly.harness.config import load_yaml_file, require_choice, require_int
from drone_fly.harness.course import Course, course_from_mapping
from drone_fly.harness.course_senses import (
    GateProgress,
    curr_gate_block,
    next_lookahead_block,
    obstacle_vision_block,
)
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import (
    ActionSink,
    BuildContext,
    FreeSpaceEstimator,
    GateEstimator,
    Harness,
    StateSource,
)
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import (
    ContractCommand,
    FlightState,
    Frame,
    FreeSpaceEstimate,
    GateEstimate,
)

#: course_gen keyword set of the PROC_TRACTABLE tier the obstacle model was trained on.
TRACTABLE_KW = {"ng_hi": 11, "laps_lo": 1, "laps_hi": 2, "ap_lo": 0.75}
START_MIN_Z = 0.12  # the env clamps the spawn onto the ground


class LoopbackPlant:
    """Holds the contract state dict; :class:`PlantActionSink` advances it with ``plant_step``."""

    def __init__(
        self, contract: Any, start: np.ndarray, clock: Any = None, drone: Any = None
    ) -> None:
        self.contract = contract
        self.clock = clock
        self.drone = drone  # DroneProfile for v2 commands (None = drones/meteor75)
        self._sim: Any = None
        self.start = np.asarray(start, np.float32).copy()
        self.start[2] = max(float(self.start[2]), START_MIN_Z)
        self.state = contract.init_state(pos=tuple(float(x) for x in self.start))
        self.steps = 0
        self.in_failsafe = False

    def step(self, raw_action: np.ndarray) -> None:
        self.state = self.contract.plant_step(self.state, raw_action)
        self.steps += 1

    def step_command(self, command: ContractCommand) -> None:
        """v1: ``plant_step`` on the raw action; v2: the sim plant on the decoded command."""
        if command.contract_version < 2 or command.collective_g is None:
            self.step(command.raw)
            return
        import torch

        from drone_fly.sim import Plant, PlantConfig

        if self._sim is None:
            profile = self.drone
            if profile is None:
                from drone_fly import drones
                from drone_fly.sim.action import REFERENCE_DRONE

                profile = drones.load(REFERENCE_DRONE)
            cfg = PlantConfig(integrator="euler-legacy", dtype="float64")
            self._sim = Plant(profile, 1, "cpu", 0, cfg, 0.0)
            self._sim.reset(torch.ones(1, dtype=torch.bool), motor_g=1.0)
        p = self._sim
        for key in ("pos", "vel", "quat", "omega"):
            setattr(p, key, torch.as_tensor(self.state[key], dtype=torch.float64).reshape(1, -1))
        p.step(
            torch.tensor([float(command.collective_g)], dtype=torch.float64),
            torch.as_tensor(command.rates, dtype=torch.float64).reshape(1, 3),
            float(self.contract.DT),
        )
        self.state = {
            key: getattr(p, key)[0].numpy().astype(np.float32)
            for key in ("pos", "vel", "quat", "omega")
        }
        self.steps += 1


class PlantStateSource(StateSource):
    def __init__(self, plant: LoopbackPlant) -> None:
        self.plant = plant

    def latest(self) -> FlightState:
        now = self.plant.clock.now() if self.plant.clock is not None else 0.0
        return FlightState.from_state_dict(
            self.plant.state, t=self.plant.steps * float(self.plant.contract.DT), received_at=now
        )

    def age(self, now: float) -> float:
        return 0.0


class PlantActionSink(ActionSink):
    """Steps the loopback plant: ``plant_step`` on a v1 raw action (bit-exact with the
    reference), the sim plant on a v2 command."""

    def __init__(self, plant: LoopbackPlant) -> None:
        self.plant = plant

    def send(self, command: ContractCommand) -> dict[str, Any]:
        if self.plant.in_failsafe:
            return {"stepped": False}
        self.plant.step_command(command)
        return {"stepped": True, "plant_step": self.plant.steps}

    def failsafe(self) -> None:
        self.plant.in_failsafe = True


class CourseGateEstimator(GateEstimator):
    """Ground-truth gate senses from known course geometry (+ the env's pass rule)."""

    provides = frozenset({"target", "curr_gate", "next_lookahead"})
    provenance = "metadata"

    def __init__(self, course: Course) -> None:
        self.course = course
        self.progress = GateProgress(course)

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> CourseGateEstimator:
        """Built by name outside the loopback factory (needs ``ctx.resources["course"]``)."""
        return cls(ctx.resources["course"])

    def reset(self) -> None:
        self.progress.reset()

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        passed = self.progress.advance_if_passed(state.pos)
        gate = self.progress.current
        status = "finished" if self.progress.finished else "tracking"
        return GateEstimate(
            target=gate.center.copy(),
            curr_gate=curr_gate_block(gate),
            next_lookahead=next_lookahead_block(
                state.pos, self.progress.next, self.course.obstacles
            ),
            gate_index=self.progress.tgt,
            lap=self.progress.lap,
            status=status,
            stamp=now,
            passed=passed,
        )


class RaycastFreeSpaceEstimator(FreeSpaceEstimator):
    """Ground-truth ``raycast`` via the contract's ``raycast()`` + the ``obstacle_vision`` block."""

    provides = frozenset({"obstacle_vision", "raycast"})
    provenance = "metadata"

    def __init__(self, course: Course, contract: Any) -> None:
        self.course = course
        self.contract = contract
        self._obstacles = course.raycast_obstacles()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> RaycastFreeSpaceEstimator:
        """Built by name (e.g. a composite child): ``ctx.resources["course"]`` + the contract."""
        return cls(ctx.resources["course"], ctx.contract)

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        rays = np.asarray(
            self.contract.raycast(state.as_state_dict(), self._obstacles, target), np.float32
        )
        return FreeSpaceEstimate(
            obstacle_vision=obstacle_vision_block(state.pos, self.course.obstacles),
            raycast=rays,
            in_fov_mask=np.ones(rays.shape[0], dtype=bool),
            stamp=now,
        )


def load_course(cfg: dict[str, Any], contract: Any, world_scale: WorldScale) -> Course:
    """Course from config: ``source: course_gen`` (seeded, via ``import_reference``) / ``yaml`` /
    ``inline``. YAML and inline geometry are in target-world metres and get ``world_scale``;
    course_gen courses are already training scale."""
    source = require_choice(cfg.get("source"), "course.source", ("course_gen", "yaml", "inline"))
    if source == "course_gen":
        seed = require_int(cfg.get("seed", 0), "course.seed", min_value=0)
        kw = dict(TRACTABLE_KW) if cfg.get("tractable", True) else {}
        raw = contract.import_reference("course_gen").sample_course(seed, **kw)
        return course_from_mapping(raw)
    if source == "yaml":
        if not cfg.get("path"):
            raise HarnessConfigError("course.source=yaml needs course.path")
        raw = load_yaml_file(cfg["path"])
    else:
        raw = cfg.get("inline") or {}
    return course_from_mapping(raw).scaled(world_scale)


def build_loopback_harness(config: Any, ctx: BuildContext) -> Harness:
    """Factory registered as ``sim-loopback``."""
    from drone_fly.harness.registry import build_part
    from drone_fly.harness.runner import SimClock

    contract = ctx.contract
    if contract is None:
        raise HarnessConfigError("the sim-loopback harness needs the inference contract")
    if abs(1.0 / config.runner.rate_hz - float(contract.DT)) > 1e-9:
        raise HarnessConfigError(
            f"sim-loopback must run at the plant rate 1/DT = {1.0 / float(contract.DT):g} Hz "
            f"(runner.rate_hz = {config.runner.rate_hz:g})"
        )
    course = load_course(config.section("course"), contract, WorldScale(config.world_scale))
    clock = SimClock()
    from drone_fly.harness.config import load_drone_profile

    plant = LoopbackPlant(contract, course.start, clock, load_drone_profile(config.data))
    parts = config.parts
    builtin_parts = {
        ("state_source", "plant"): lambda: PlantStateSource(plant),
        ("gate_estimator", "course"): lambda: CourseGateEstimator(course),
        ("free_space_estimator", "raycast"): lambda: RaycastFreeSpaceEstimator(course, contract),
        ("action_sink", "plant"): lambda: PlantActionSink(plant),
    }
    lctx = BuildContext(
        config=config,
        contract=contract,
        extra_dim=ctx.extra_dim,
        dry_run=ctx.dry_run,
        resources={"course": course, "plant": plant, "clock": clock},
    )

    def make(kind: str, name: str | None) -> Any:
        if name is None:
            return None
        if (kind, name) in builtin_parts:
            return builtin_parts[(kind, name)]()
        # Any other registered part (e.g. a programmatically registered fake estimator) gets
        # the course + plant through ``ctx.resources``.
        return build_part(kind, name, (config.get("part_options") or {}).get(kind) or {}, lctx)

    sink_name = "recording" if ctx.dry_run else parts.action_sink
    return Harness(
        name="sim-loopback",
        state_source=make("state_source", parts.state_source),
        gate_estimator=make("gate_estimator", parts.gate_estimator),
        action_sink=make("action_sink", sink_name),
        frame_source=make("frame_source", parts.frame_source),
        free_space_estimator=make("free_space_estimator", parts.free_space_estimator),
        abort_signal=make("abort_signal", parts.abort_signal),
        clock=clock,
        info={"course_gates": course.n_gates, "course_laps": course.laps},
    )
