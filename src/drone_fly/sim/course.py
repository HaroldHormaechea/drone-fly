"""The procedural random-course env: a port of ``gpu_prototype/gates_proc_env.BatchedProcCourse``.

Every env flies a different random course drawn from a pre-generated pool (sorted easy -> hard
for the curriculum), re-drawn at each reset. Observations (``obs()`` 12 dims, ``extra()`` 30 or
70 dims), the gate-pass rule, the obstacle contact nudge, the occlusion-aware lookahead, the
raycast and every reward term are the prototype's. What changed:

* **Dynamics** are a profile-driven :class:`~drone_fly.sim.plant.Plant` (dynamics
  randomization, motor lag, drag, delay line, ground contact) stepped with the per-env
  :class:`~drone_fly.sim.timing.ControlTiming` (control period, action delay + jitter, skipped
  ticks). The raw policy action is decoded by the checkpoint's
  :class:`~drone_fly.sim.action.ActionSpec` (v1 legacy fraction or v2 hover-relative g).
* **Flags** are :class:`CourseConfig` fields instead of ``PROC_*`` environment variables.
* **Rate invariance.** The dense per-step terms (upright, low-spin, obstacle contact, gap
  steering) are multiplied by ``dt / reward_reference_dt_s``; potential-difference terms
  (progress, clearance and floor shaping) and events (gate pass, completion, crash) are not.
  ``reward_reference_dt_s`` has no default (the configs set 0.02 s). Episode length and the
  floor-kill grace are times (the prototype's 1400 and 20 steps of 20 ms).
* **Starts** (:class:`StartConfig`): ``legacy`` reproduces the prototype (spawn at the course
  start, z >= 0.12, no contact model, floor kill after the grace); ``mixed`` draws a ground start
  (at rest on the floor, level, motors idle for ``arm_delay_s`` whatever the policy outputs —
  the policy still observes) with probability ``ground_fraction``, else an air start (level,
  motor state 1 g, z from ``air_z_m``). A ground start's floor contact is non-terminal until its
  first lift-off; after it the floor crash applies.
* **No host synchronisation in step()**: respawn and contact are masked tensor ops; the
  completion metric is an on-device running average.
* **Per-episode log** (:class:`EpisodeLog`): every dynamics and timing sample and the start mode
  at reset, the return / length / outcome at the end, buffered on the device and drained once
  per PPO iteration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from drone_fly.sim.action import (
    ActionSpec,
    SlewSpec,
    decode_torch,
    raw_of_command,
    slew_clamp,
)
from drone_fly.sim.course_gen import sample_course
from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.quat import gravity_body
from drone_fly.sim.randomization import LOG_COLUMNS
from drone_fly.sim.timing import TIMING_LOG_COLUMNS, ControlTiming, TimingConfig

NG_MAX = 20  # pad gate lists to this
MO = 10  # max obstacles per course
NVIS = 2  # nearest obstacles the readout sees
MARGIN = 0.12  # drone half-extent for obstacle contact
BOUNCE = 0.6  # outward impulse on obstacle contact (recoverable nudge)
CONTACT_PEN = 0.5
ARENA = 16.0  # xy crash bound (courses are centred on origin, radius <~9)
CEIL = 3.5
FLOOR_KILL_Z = 0.08
SPAWN_MIN_Z = 0.12  # the prototype clamps a legacy spawn onto the ground at this height
SPAWN_NOISE_M = 0.05
PASS_SLACK = 0.6  # precision curriculum: extra pass radius at difficulty 0, -> 0 at difficulty 1
PARK = 3.0  # position curriculum: metres an obstacle is parked off-path at obs_pos_curr = 0
SHAPE_BETA = 4.0  # potential-based clearance shaping weight
GAP_BETA = 0.4  # gap-steering reward weight
FLOOR_BETA = 3.0  # floor-margin shaping weight
FLOOR_CAP = 0.5  # only shape altitude within this many metres of the floor
CLEAR_CAP = 1.5  # only shape clearance within this many metres of an obstacle surface
N_RAYS = 40  # forward-cone raycasts
CONE_HALF_DEG = 45.0  # 90-degree total cone
RAY_RANGE = 4.0  # max sensing distance (m)
BASE_EXTRA_DIM = 12 + 3 + 7 + NVIS * 4  # obs + curr gate + next lookahead + obstacle vision
#: ``TRACTABLE`` tier (the 07 model's): 1 lap, 5-10 gates, apertures 0.75-1.5 m.
TRACTABLE_KW = {"ng_hi": 11, "laps_lo": 1, "laps_hi": 2, "ap_lo": 0.75}
START_MODES = ("legacy", "mixed")
#: Start-mode codes in the episode log.
START_LEGACY, START_GROUND, START_AIR = 0, 1, 2
#: Crash-cause codes in the episode log.
OUTCOMES = ("none", "floor", "ceiling", "arena", "obstacle", "timeout", "completed")


@dataclass(frozen=True)
class StartConfig:
    mode: str = "mixed"
    #: ``mixed``: probability of a ground start (else an air start).
    ground_fraction: float = 0.5
    #: Air-start height range (m).
    air_z_m: tuple[float, float] = (0.8, 1.5)
    #: Motors idle this long after a ground start (Liftoff: arm press, then the 0.5 s settle).
    arm_delay_s: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in START_MODES:
            raise ValueError(f"start.mode must be one of {START_MODES}, got {self.mode!r}")
        if not 0.0 <= self.ground_fraction <= 1.0:
            raise ValueError("start.ground_fraction must be in [0, 1]")
        lo, hi = self.air_z_m
        if not 0 < lo <= hi:
            raise ValueError("start.air_z_m needs 0 < lo <= hi")
        if self.arm_delay_s < 0:
            raise ValueError("start.arm_delay_s must be >= 0")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> StartConfig:
        d = dict(d or {})
        if "air_z_m" in d:
            d["air_z_m"] = tuple(float(x) for x in d["air_z_m"])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown start keys {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class CourseConfig:
    """The prototype's ``PROC_*`` flags as fields (see the module docstring)."""

    #: Dense rewards are scaled by ``dt / reward_reference_dt_s`` (required: no sim default).
    reward_reference_dt_s: float
    tier: str = "tractable"  # tractable | full
    tunnels: bool = False
    multilap: bool = False
    variety: bool = False
    no_obs: bool = False
    obs_terminal: bool = False
    shaping: bool = False
    gap: bool = False
    floor: bool = False
    raycast: bool = False
    #: UC-82 obstacle-ray perception error. ``ray_noise`` is a path to a ``ray-noise.yaml``
    #: (:class:`~drone_fly.sim.ray_noise.RayNoiseModel`) injected into the 40 raycast rays of the
    #: observation; ``ray_blank`` replaces those rays with constant clear (1.0) — the AC8 control.
    #: Both require ``raycast`` and are mutually exclusive. With both unset the ray path is
    #: byte-identical to today (no extra RNG is drawn).
    ray_noise: str | None = None
    ray_blank: bool = False
    pool_size: int = 1024
    pool_base: int = 0
    episode_s: float = 28.0  # the prototype's 1400 steps of 20 ms
    floor_grace_s: float = 0.4  # the prototype's 20 steps of 20 ms
    start: StartConfig = field(default_factory=StartConfig)

    # ---- UC-87 deploy-robust fields (all default-OFF ⇒ byte-identical to the model-11 env) ----
    #: Append the policy-memory block to ``extra()`` (D1): ``prev_action(4) || obs_history(H*12)``,
    #: strictly AFTER the raycast block (the D2 end-append invariant). Columns 0..(pre-memory) stay
    #: byte-identical to model 11, so ``warm_start``'s zero-init reproduces it.
    memory: bool = False
    #: H — the number of past policy-step 12-dim core-obs vectors stacked in the memory block.
    history_len: int = 2
    #: λ for the action-rate penalty ``λ·‖tanh(a_t)−tanh(a_{t-1})‖²`` (rate-scaled; AC3).
    action_rate_lambda: float = 0.0
    #: Weight for the body-rate saturation penalty ``Σ relu(|tanh(a_i)|−thr)`` over the rate dims.
    saturation_weight: float = 0.0
    #: ``|tanh(a)|`` above this counts as saturated (AC6 uses 3.8/4.0 = 0.95 on the metric side).
    saturation_threshold: float = 0.9
    #: Ceiling altitude (m) for the crash bound (AC5, configurable 4–6 m). Default matches the
    #: historical fixed ``CEIL`` (3.5) so existing configs are unchanged; UC-87 sets 5.0 in its cfg.
    ceiling_z: float = 3.5
    #: Randomise the spawn attitude yaw uniformly over 0–360° (AC4; live spawns off-axis).
    spawn_yaw: bool = False
    #: Randomise the spawn's horizontal distance to the first gate, ``(lo, hi)`` m (AC4).
    spawn_distance_m: tuple[float, float] | None = None
    #: White gyro/omega observation noise σ (rad/s), injected into the OBSERVED omega only (AC4).
    gyro_noise_sigma: float = 0.0
    #: Coloured (AR(1)) vibration component: stationary σ (rad/s) and the AR coefficient.
    gyro_vibration_sigma: float = 0.0
    gyro_vibration_rho: float = 0.0
    #: Make the floor plane (z = ground_z) visible to the forward raycast (AC4).
    raycast_floor: bool = False
    #: Zero the ``obstacle_vision`` block in ``extra()`` (live sees zeros; AC4).
    obstacle_vision_zero: bool = False
    #: Gate-estimate perception noise on the tracked (current-target) gate (AC4). All 0 ⇒ off.
    gate_pos_noise_m: float = 0.0
    gate_outlier_p: float = 0.0
    gate_outlier_m: float = 0.4
    gate_staleness_s: float = 0.0
    gate_dropout_p: float = 0.0

    # ---- UC-88 command slew-rate limiter (action filter; default-OFF ⇒ UC-87 byte-identical) ----
    #: Controller-realistic slew cap: the time (s) the applied command would take to cross its FULL
    #: per-axis range at the cap (rates span ``2·max_rate_rad_s``, collective spans the plant's
    #: ``max_g``), so the per-axis per-step cap is ``full_range · dt / slew_time_s``. ``None``
    #: disables the filter — the command passes through unchanged and ``prev_applied_raw`` stays
    #: equal to the raw intent, so existing configs/checkpoints are unaffected (AC1/AC5).
    slew_time_s: float | None = None

    # ---- UC-89 grounded-launch frame alignment (default-OFF ⇒ byte-identical) ----
    #: Fraction of **ground** starts for which the whole course is rigidly rotated about the spawn
    #: so the start->gate-0 direction becomes world +x — i.e. gate-0 is presented dead-ahead exactly
    #: as the deploy spawn-yaw aligner presents the live scene (`harness/yaw_align.py`). The drone
    #: keeps its identity spawn attitude (no spawn-yaw DR); only the observed/flown scene rotates,
    #: so the heading reference stays observable. ``0.0`` disables it — no RNG is drawn and every
    #: per-env world read is byte-identical to today. Pair with ``spawn_distance_m`` (≈3.2 at
    #: world_scale 12 ≈ live 38 m) to place a grounded gate-0 in view at the live geometry.
    ground_gate_ahead_frac: float = 0.0
    #: Pre-acquisition neutral window (s). For this long at the start of a **ground** episode the
    #: policy observes the SAME full-neutral gate the live harness emits on no-detection —
    #: ``rel = 0`` (hover) and ``curr_gate = [0, 0, aperture]`` (frame-neutral; cos/sin zeroed) —
    #: then the real gate is acquired. Models the live grounded pre-lock window and makes the sim's
    #: no-detection semantics identical to ``gate_cv.py`` in BOTH channels (AC2). Only the policy's
    #: *observation* is held neutral; the true geometry/reward are untouched. ``0.0`` disables it ⇒
    #: byte-identical.
    pre_acquire_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.reward_reference_dt_s or self.reward_reference_dt_s <= 0:
            raise ValueError("course.reward_reference_dt_s must be positive")
        if self.tier not in ("tractable", "full"):
            raise ValueError("course.tier must be tractable or full")
        if (self.ray_noise or self.ray_blank) and not self.raycast:
            raise ValueError("course.ray_noise / ray_blank require course.raycast")
        if self.ray_noise and self.ray_blank:
            raise ValueError("course.ray_noise and course.ray_blank are mutually exclusive")
        if self.history_len < 0:
            raise ValueError("course.history_len must be >= 0")
        if self.memory and self.history_len < 1:
            raise ValueError("course.memory needs history_len >= 1")
        if not 0.0 <= self.ceiling_z:
            raise ValueError("course.ceiling_z must be >= 0")
        if self.spawn_distance_m is not None:
            lo, hi = self.spawn_distance_m
            if not 0 < lo <= hi:
                raise ValueError("course.spawn_distance_m needs 0 < lo <= hi")
        if not 0.0 <= self.gate_outlier_p <= 1.0 or not 0.0 <= self.gate_dropout_p <= 1.0:
            raise ValueError("course.gate_outlier_p / gate_dropout_p must be in [0, 1]")
        if not -1.0 < self.gyro_vibration_rho < 1.0:
            raise ValueError("course.gyro_vibration_rho must be in (-1, 1)")
        if self.slew_time_s is not None and (
            not math.isfinite(self.slew_time_s) or self.slew_time_s <= 0
        ):
            raise ValueError("course.slew_time_s must be a positive, finite number when set")
        if not 0.0 <= self.ground_gate_ahead_frac <= 1.0:
            raise ValueError("course.ground_gate_ahead_frac must be in [0, 1]")
        if self.pre_acquire_s < 0:
            raise ValueError("course.pre_acquire_s must be >= 0")

    @property
    def pool_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = dict(TRACTABLE_KW) if self.tier == "tractable" else {}
        if self.tunnels:
            kw["with_tunnels"] = True
        if self.multilap:
            kw.update(laps_lo=2, laps_hi=5)
        if self.variety:
            kw["with_variety"] = True
        return kw

    @property
    def memory_dim(self) -> int:
        """Width of the appended memory block (0 when ``memory`` is off)."""
        return (4 + self.history_len * 12) if self.memory else 0

    @property
    def extra_dim(self) -> int:
        return BASE_EXTRA_DIM + (N_RAYS if self.raycast else 0) + self.memory_dim

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CourseConfig:
        d = dict(d)
        start = StartConfig.from_dict(d.pop("start", None))
        if d.get("spawn_distance_m") is not None:
            d["spawn_distance_m"] = tuple(float(x) for x in d["spawn_distance_m"])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown course keys {sorted(unknown)}")
        if "reward_reference_dt_s" not in d:
            raise ValueError("course.reward_reference_dt_s is required")
        return cls(start=start, **d)

    @classmethod
    def from_flags(cls, flags: dict[str, Any], **kw: Any) -> CourseConfig:
        """From a ``gates_proc_env`` module's flag globals (``NO_OBS``, ``RAYCAST``, ...)."""
        pool_kw = flags.get("POOL_KW") or {}
        return cls(
            tier="tractable" if pool_kw.get("ng_hi") == TRACTABLE_KW["ng_hi"] else "full",
            tunnels=bool(pool_kw.get("with_tunnels")),
            multilap=pool_kw.get("laps_lo") == 2 and pool_kw.get("laps_hi") == 5,
            variety=bool(pool_kw.get("with_variety")),
            no_obs=bool(flags.get("NO_OBS")),
            obs_terminal=bool(flags.get("OBS_TERMINAL")),
            shaping=bool(flags.get("SHAPING")),
            gap=bool(flags.get("GAP")),
            floor=bool(flags.get("FLOOR")),
            raycast=bool(flags.get("RAYCAST")),
            **kw,
        )


# ----------------------------------------------------------------------------- course pool

_POOLS: dict[tuple, dict[str, np.ndarray]] = {}


def build_pool(
    size: int = 1024, base: int = 0, no_obs: bool = False, **ckw: Any
) -> dict[str, np.ndarray]:
    """``size`` courses packed into padded arrays, sorted easy -> hard (prototype ``build_pool``).

    Cached per argument set (NumPy only; tensors are made per env instance).
    """
    key = (size, base, no_obs, tuple(sorted(ckw.items())))
    if key in _POOLS:
        return _POOLS[key]
    P = size
    cen = np.zeros((P, NG_MAX, 3), np.float32)
    yaw = np.zeros((P, NG_MAX), np.float32)
    ap = np.ones((P, NG_MAX), np.float32)
    gmask = np.zeros((P, NG_MAX), bool)
    start = np.zeros((P, 3), np.float32)
    ng = np.zeros(P, np.int64)
    laps = np.zeros(P, np.int64)
    oc = np.zeros((P, MO, 3), np.float32)
    odim = np.ones((P, MO, 3), np.float32)
    ovec = np.zeros((P, MO, 3), np.float32)
    otype = np.zeros((P, MO), np.int64)
    omask = np.zeros((P, MO), bool)
    for p in range(P):
        c = sample_course(base + p, **ckw)
        n = c["n_gates"]
        ng[p] = n
        laps[p] = c["laps"]
        for i, g in enumerate(c["gates"]):
            cen[p, i] = g["center"]
            yaw[p, i] = g["yaw"]
            ap[p, i] = g["aperture"]
            gmask[p, i] = True
        start[p] = c["start"]
        for j, o in enumerate(c["obstacles"][:MO]):
            oc[p, j] = o["center"]
            omask[p, j] = True
            ovec[p, j] = o.get("ovec", [0.0, 0.0, 0.0])
            if o["kind"] == "cylinder":
                otype[p, j] = 0
                odim[p, j] = [o["radius"], o["radius"], o["half_h"]]
            else:
                otype[p, j] = 1
                odim[p, j] = o["half"]
    # difficulty (low = easy): more gates, obstacles, laps and tighter apertures = harder
    mean_ap = np.nanmean(np.where(gmask, ap, np.nan), axis=1)
    nobs = omask.sum(axis=1)
    score = ng + 2.0 * nobs + 4.0 / mean_ap + 3.0 * (laps - 2)
    order = np.argsort(score)
    pool = {
        "cen": cen[order],
        "yaw": yaw[order],
        "ap": ap[order],
        "gmask": gmask[order],
        "start": start[order],
        "ng": ng[order],
        "laps": laps[order],
        "oc": oc[order],
        "odim": odim[order],
        "ovec": ovec[order],
        "otype": otype[order],
        "omask": omask[order],
    }
    if no_obs:
        pool["omask"] = np.zeros_like(pool["omask"])  # obstacle-free tier
    _POOLS[key] = pool
    return pool


def ray_cone(n_rays: int = N_RAYS, half_deg: float = CONE_HALF_DEG) -> np.ndarray:
    """Canonical ray directions in a cone around +z (Fibonacci spiral), ``(n_rays, 3)``."""
    ga = math.pi * (3 - math.sqrt(5))
    cz_min = math.cos(math.radians(half_deg))
    out = []
    for i in range(n_rays):
        cz = 1.0 - (i / max(1, n_rays - 1)) * (1 - cz_min)
        st = math.sqrt(max(0.0, 1 - cz * cz))
        phi = i * ga
        out.append((st * math.cos(phi), st * math.sin(phi), cz))
    return np.asarray(out, np.float32)


# ----------------------------------------------------------------------------- episode log

EPISODE_HEADER = (
    "env",
    "episode",
    "start_mode",
    "course_id",
    "n_gates",
    "laps",
    *LOG_COLUMNS,
    *TIMING_LOG_COLUMNS,
)
EPISODE_TAIL = (
    "return",
    "length_s",
    "steps",
    "completed",
    "outcome",
    # UC-87 AC3: per-episode SUM of the (unweighted) smoothness terms; divide by ``steps`` for the
    # per-tick mean. Always present (0 when the penalties are off) so the schema is stable.
    "action_rate",
    "saturation",
)
EPISODE_COLUMNS = EPISODE_HEADER + EPISODE_TAIL


class EpisodeLog:
    """Device-side buffer of finished-episode rows (:data:`EPISODE_COLUMNS`), no host sync.

    Rows are scattered to ``count + cumsum(done) - 1``; rows past ``capacity`` are dropped and
    counted. :meth:`drain` (one sync) returns the rows and empties the buffer.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.buf = torch.zeros(self.capacity + 1, len(EPISODE_COLUMNS), device=device)
        self.count = torch.zeros((), dtype=torch.long, device=device)
        self.dropped = 0

    def add(self, done: torch.Tensor, rows: torch.Tensor) -> None:
        pos = self.count + torch.cumsum(done.long(), 0) - 1
        pos = torch.where(done & (pos < self.capacity), pos, self.capacity)
        self.buf.index_copy_(0, pos, rows.to(self.buf.dtype))
        self.count = self.count + done.long().sum()

    def drain(self) -> np.ndarray:
        n = int(self.count)
        kept = min(n, self.capacity)
        self.dropped += n - kept
        rows = self.buf[:kept].cpu().numpy().copy()
        self.count.zero_()
        return rows


# ----------------------------------------------------------------------------- the env


class CourseEnv:
    """Batched procedural gate course on a profile-driven plant; see the module docstring."""

    def __init__(
        self,
        profile: Any,
        n_envs: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        config: CourseConfig | None = None,
        plant_config: PlantConfig | None = None,
        timing: TimingConfig | None = None,
        randomization_scale: float = 0.0,
        *,
        action_spec: ActionSpec,
        log_capacity: int | None = None,
    ) -> None:
        if config is None:
            raise ValueError("CourseEnv needs a CourseConfig (reward_reference_dt_s is required)")
        self.cfg = config
        self.n = n = int(n_envs)
        self.device = dev = torch.device(device)
        self.action_spec = action_spec
        self.timing_config = timing or TimingConfig()
        pc = plant_config or PlantConfig()
        # delay-line slots: the longest delay over the shortest step, plus margin
        p_delay = profile.latency.input_s + max(r.delay_s for r in profile.rates.values())
        rnd = profile.randomization
        p_delay += max(
            [rnd[k].max for k in rnd if k.endswith("delay_s") or k.startswith("latency")] or [0.0]
        )
        min_dt = 1.0 / (
            max(self.timing_config.control_hz_set or self.timing_config.control_hz)
            if self.timing_config.mode == "random"
            else self.timing_config.fixed_hz
        )
        need = math.ceil((self.timing_config.max_delay_s + p_delay) / min_dt) + 3
        if need > pc.delay_slots:
            pc = replace(pc, delay_slots=need)
        self.plant = Plant(profile, n, dev, seed, pc, randomization_scale)
        self.dtype = self.plant.dtype
        self.gen = self.plant.generator
        self.timing = ControlTiming(self.timing_config, n, dev, self.gen)
        pool = build_pool(config.pool_size, config.pool_base, config.no_obs, **config.pool_kwargs)
        self.P = {
            k: torch.as_tensor(v, device=dev).to(self.dtype)
            if v.dtype == np.float32
            else torch.as_tensor(v, device=dev)
            for k, v in pool.items()
        }
        self.cone = torch.as_tensor(ray_cone(), device=dev, dtype=self.dtype)
        self.ngates = NG_MAX
        self.extra_dim = config.extra_dim
        self.difficulty = 1.0  # curriculum: fraction of the sorted pool that can be drawn
        self.obs_pos_curr = 1.0  # 0 = obstacles parked PARK m off-path, 1 = on-path
        self.obs_count_curr = 1.0  # 0 = at most 1 obstacle active, 1 = all active
        # UC-87 redesign: three runtime ramp scales (default 1.0 = full ⇒ byte-identical off).
        # The train loop sets them per iteration from the CurriculumConfig schedule; they have NO
        # effect on existing configs/eval (which leave them at 1.0).
        self.deploy_dr_scale = 1.0  # control timing/delay/skip + plant dynamics-randomisation scale
        self.input_dr_scale = 1.0  # gyro/gate/ray input-noise magnitude (0 ⇒ zero noise effect)
        self.penalty_scale = 1.0  # multiplies cfg.action_rate_lambda ONLY (λ mild→target anneal)
        #: Base (config) plant DR scale; effective = ``_base_dr_scale * deploy_dr_scale``.
        self._base_dr_scale = float(randomization_scale)
        self._applied_dr_scale = float(randomization_scale)  # last scale pushed to the plant
        self._ar_n = torch.arange(n, device=dev)
        z = torch.zeros(n, device=dev, dtype=torch.long)
        self.tgt = z.clone()
        self.lap = z.clone()
        self.cid = z.clone()
        self.steps = z.clone()
        self.episode = z.clone()
        self.start_mode = z.clone()
        self.arm_until = torch.zeros(n, dtype=torch.float64, device=dev)
        zf = torch.zeros(n, dtype=self.dtype, device=dev)
        self.prev_clear = torch.full_like(zf, CLEAR_CAP)
        self.prev_floor = torch.full_like(zf, FLOOR_CAP)
        self.ep_return = zf.clone()
        # UC-87 ceiling (configurable; was the fixed CEIL constant)
        self.ceil_z = float(config.ceiling_z)
        # UC-87 policy-memory state (D1/D3): the previous raw action and the recent core-obs ring.
        # ``prev_action`` is always maintained (cheap, internal) so the action-rate penalty works
        # independently of the memory toggle; ``obs_hist`` is only allocated when memory is on.
        self.prev_action = torch.zeros(n, 4, dtype=self.dtype, device=dev)
        self.obs_hist = (
            torch.zeros(n, config.history_len, 12, dtype=self.dtype, device=dev)
            if config.memory
            else None
        )
        # UC-88 command slew-rate limiter (action filter). THREE distinct state pieces:
        #  * ``prev_action`` (above)       — the raw INTENT, unchanged; drives the action-rate +
        #    saturation penalty ONLY (reward path provably untouched on and off).
        #  * ``prev_applied_command``      — the limiter's command-space carry ``[cg, rates]``
        #    (enabled only); the clamp ramps from it toward each step's new command.
        #  * ``prev_applied_raw``          — the effective-raw of the APPLIED command; drives ONLY
        #    the policy-memory block (``_memory_block``). ≡ raw intent when the filter is OFF (so
        #    the memory block is byte-identical to UC-87), = a' when ON ("policy observes what was
        #    actually applied", AC4). Always maintained; zero-seeded like ``prev_action``.
        self._slew_spec = SlewSpec(config.slew_time_s)
        self._slew_on = self._slew_spec.enabled
        self.prev_applied_command = torch.zeros(n, 4, dtype=self.dtype, device=dev)
        self.prev_applied_raw = torch.zeros(n, 4, dtype=self.dtype, device=dev)
        # UC-89 grounded-launch frame alignment (default-off ⇒ no RNG, byte-identical). Per-env
        # rigid rotation of the course about the spawn so gate-0 lands on +x. ``_gstart_active``
        # gates which envs rotate; the rest keep identity (c=1, s=0, θ=0) so ``torch.where`` returns
        # the original world reads untouched. ``_gstart_cxy`` is the per-env rotation centre.
        self._gstart_on = config.ground_gate_ahead_frac > 0
        self._gstart_active = torch.zeros(n, dtype=torch.bool, device=dev)
        self._gstart_theta = zf.clone()
        self._gstart_c = torch.ones(n, dtype=self.dtype, device=dev)
        self._gstart_s = zf.clone()
        self._gstart_cxy = torch.zeros(n, 2, dtype=self.dtype, device=dev)
        # UC-89 pre-acquisition neutral window (default-off ⇒ byte-identical). Holds the policy's
        # observed gate at full neutral for the first ``pre_acquire_s`` of a ground episode.
        self._preacq_on = config.pre_acquire_s > 0
        self._preacq_s = float(config.pre_acquire_s)
        # UC-87 AC3 per-episode smoothness accumulators (logged in EPISODE_TAIL).
        self.ep_arate = zf.clone()
        self.ep_sat = zf.clone()
        # UC-87 AC4 input-domain-randomisation state (all default-off ⇒ no RNG, byte-identical).
        self._gyro_on = config.gyro_noise_sigma > 0 or config.gyro_vibration_sigma > 0
        self._gate_on = (
            config.gate_pos_noise_m > 0
            or config.gate_outlier_p > 0
            or config.gate_dropout_p > 0
            or config.gate_staleness_s > 0
        )
        self._gyro_ar = torch.zeros(n, 3, dtype=self.dtype, device=dev)
        self._gyro_obs = torch.zeros(n, 3, dtype=self.dtype, device=dev)
        self._gate_rel = torch.zeros(n, 3, dtype=self.dtype, device=dev)
        self._gate_age = torch.zeros(n, dtype=torch.long, device=dev)
        self._gate_hold = torch.zeros(n, dtype=torch.long, device=dev)
        #: A dedicated generator for the UC-87 input DR so it never perturbs the plant/timing RNG
        #: stream (the default-off path stays byte-identical).
        self._perc_gen: Any = None
        if self._gyro_on or self._gate_on:
            self._perc_gen = torch.Generator(device=dev)
            self._perc_gen.manual_seed(int(seed) ^ 0x44524E4F)  # "DRNO"
        self.comp_rate = torch.zeros((), dtype=self.dtype, device=dev)
        self.last_completed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_crashed = torch.zeros_like(self.last_completed)
        self.last_any_hit = torch.zeros_like(self.last_completed)
        self.last_dt = torch.zeros(n, dtype=torch.float64, device=dev)
        self.last_skipped = torch.zeros_like(self.last_completed)
        self.header = torch.zeros(n, len(EPISODE_HEADER), device=dev)
        self.log = EpisodeLog(log_capacity, dev) if log_capacity else None
        self._episode_counter = 0
        # UC-82 obstacle-ray perception error (opt-in; the default path is untouched).
        self._ray_noise: Any = None
        self._ray_camera: Any = None
        self._ray_gen: Any = None
        if config.ray_noise:
            from drone_fly.sim.ray_noise import RayNoiseModel

            self._ray_noise = RayNoiseModel.load(self._resolve_ray_noise(config.ray_noise))
            self._ray_camera = self._ray_noise.camera_twin()
            # a dedicated generator so the injected noise is reproducible and never perturbs the
            # plant / timing RNG stream (keeps the default-off path byte-identical)
            self._ray_gen = torch.Generator(device=dev)
            self._ray_gen.manual_seed(int(seed) ^ 0x52415953)  # "RAYS"
        self.reset_all()

    @staticmethod
    def _resolve_ray_noise(ref: str) -> str:
        """A ray-noise path as given, or resolved relative to the repo root (config convention)."""
        from pathlib import Path

        p = Path(ref)
        if p.is_absolute() or p.exists():
            return str(p)
        from drone_fly.harness.config import find_repo_root

        root = find_repo_root()
        return str((root / ref) if root is not None else p)

    # ---- state views (the plant owns the state) ----
    @property
    def pos(self) -> torch.Tensor:
        return self.plant.pos

    @pos.setter
    def pos(self, v: torch.Tensor) -> None:
        self.plant.pos = v

    @property
    def vel(self) -> torch.Tensor:
        return self.plant.vel

    @vel.setter
    def vel(self, v: torch.Tensor) -> None:
        self.plant.vel = v

    @property
    def quat(self) -> torch.Tensor:
        return self.plant.quat

    @quat.setter
    def quat(self, v: torch.Tensor) -> None:
        self.plant.quat = v

    @property
    def omega(self) -> torch.Tensor:
        return self.plant.omega

    @omega.setter
    def omega(self, v: torch.Tensor) -> None:
        self.plant.omega = v

    @property
    def t(self) -> torch.Tensor:
        """Episode time (s)."""
        return self.plant.t

    # ---- per-env course accessors ----
    def _apply_gstart_rot(self, p3: torch.Tensor) -> torch.Tensor:
        """UC-89: rigidly rotate the xy of world positions ``p3`` (``(n, 3)`` or ``(n, K, 3)``)
        about each env's ground-start spawn centre by its stored angle. A strict no-op (byte-exact)
        for envs without the rotation active and whenever the feature is off (``_gstart_on``),
        because inactive envs are returned verbatim via ``torch.where`` (no float round-trip)."""
        if not self._gstart_on:
            return p3
        shape = (self.n,) + (1,) * (p3.dim() - 2)  # broadcast over any mid dims (e.g. MO)
        c = self._gstart_c.view(shape)
        s = self._gstart_s.view(shape)
        cx = self._gstart_cxy[:, 0].view(shape)
        cy = self._gstart_cxy[:, 1].view(shape)
        active = self._gstart_active.view(shape)
        x, y = p3[..., 0], p3[..., 1]
        dx, dy = x - cx, y - cy
        rx = cx + c * dx - s * dy
        ry = cy + s * dx + c * dy
        out = p3.clone()
        out[..., 0] = torch.where(active, rx, x)
        out[..., 1] = torch.where(active, ry, y)
        return out

    def _gate(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cen = self.P["cen"][self.cid, idx]
        yaw = self.P["yaw"][self.cid, idx]
        ap = self.P["ap"][self.cid, idx]
        if self._gstart_on:
            cen = self._apply_gstart_rot(cen)
            yaw = torch.where(self._gstart_active, yaw + self._gstart_theta, yaw)
        return cen, yaw, ap

    def _ng(self) -> torch.Tensor:
        return self.P["ng"][self.cid]

    def _odim(self) -> torch.Tensor:
        # OBSTACLE-SIZE curriculum: obstacles grow from 0 (difficulty 0) to full (difficulty 1)
        return self.P["odim"][self.cid] * max(float(self.difficulty), 0.0)

    def _omask(self) -> torch.Tensor:
        # OBSTACLE-COUNT curriculum: at most K active obstacles per course, K growing 1 -> all
        om = self.P["omask"][self.cid]
        if float(self.obs_count_curr) >= 1.0:
            return om
        nobs = om.sum(dim=1)
        k = (1 + torch.round(float(self.obs_count_curr) * (nobs - 1).clamp(min=0))).long()
        rank = om.cumsum(dim=1)
        return om & (rank <= k.unsqueeze(1))

    def _ocen(self) -> torch.Tensor:
        # OBSTACLE-POSITION curriculum: parked PARK m off to the side, sliding onto the path
        off = PARK * (1.0 - max(0.0, min(1.0, float(self.obs_pos_curr))))
        oc = self.P["oc"][self.cid] + self.P["ovec"][self.cid] * off
        # UC-89: rotate obstacle world centres with the rest of the scene (no-op when off). NOTE:
        # box obstacles are axis-aligned (AABB) with no stored orientation, so their faces do NOT
        # rotate — only their centres do. Frame-correct for gates and cylinders; see guard note.
        return self._apply_gstart_rot(oc)

    def _target_pos(self) -> torch.Tensor:
        return self._gate(self.tgt)[0]

    # ---- reset ----
    def reset_all(self, course_ids: torch.Tensor | None = None) -> None:
        """Respawn every env (on ``course_ids`` when given: paired evaluations)."""
        self._spawn(torch.ones(self.n, dtype=torch.bool, device=self.device), course_ids)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _perc_randn(self, *shape: int) -> torch.Tensor:
        """Standard-normal draw from the dedicated UC-87 input-DR generator."""
        return torch.randn(*shape, generator=self._perc_gen, device=self.device, dtype=self.dtype)

    def _perc_rand(self, *shape: int) -> torch.Tensor:
        """Uniform [0, 1) draw from the dedicated UC-87 input-DR generator."""
        return torch.rand(*shape, generator=self._perc_gen, device=self.device, dtype=self.dtype)

    def _apply_dr_scale(self) -> None:
        """Push the current effective plant DR scale (``_base_dr_scale * deploy_dr_scale``) to the
        plant when it changes (UC-87 deployment ramp). Recomputing is cheap but guarded so the
        default (deploy_dr_scale == 1.0) is a no-op after construction."""
        eff = self._base_dr_scale * float(self.deploy_dr_scale)
        if eff != self._applied_dr_scale:
            self.plant.set_dr_scale(eff)
            self._applied_dr_scale = eff

    def _spawn(self, mask: torch.Tensor, course_ids: torch.Tensor | None = None) -> None:
        """Respawn the envs in ``mask`` on fresh courses (masked, no host sync)."""
        n, dev = self.n, self.device
        m1 = mask[:, None]
        lim = max(16, int(self.difficulty * self.P["cen"].shape[0]))
        cid = torch.randint(0, lim, (n,), generator=self.gen, device=dev)
        if course_ids is not None:
            cid = torch.as_tensor(course_ids, dtype=torch.long, device=dev)
        self.cid = torch.where(mask, cid, self.cid)
        self.tgt = torch.where(mask, torch.zeros_like(self.tgt), self.tgt)
        self.lap = torch.where(mask, torch.zeros_like(self.lap), self.lap)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        start = self.P["start"][self.cid]
        pos = start + SPAWN_NOISE_M * self._randn(n, 3)
        sc = self.cfg.start
        rest_z = self.plant.rest_z
        if sc.mode == "legacy":
            pos[:, 2] = start[:, 2].clamp(min=SPAWN_MIN_Z)
            mode = torch.full((n,), START_LEGACY, dtype=torch.long, device=dev)
        else:
            ground = self._rand(n) < sc.ground_fraction
            air_z = sc.air_z_m[0] + (sc.air_z_m[1] - sc.air_z_m[0]) * self._rand(n)
            pos[:, 2] = torch.where(ground, torch.full_like(air_z, rest_z), air_z)
            mode = torch.where(ground, START_GROUND, START_AIR)
        ground = mode == START_GROUND
        motor = torch.where(ground, 0.0, 1.0).to(self.dtype)
        command = torch.zeros(n, 4, dtype=self.dtype, device=dev)
        command[:, 0] = motor
        # UC-87 AC4: randomise the horizontal spawn distance to the first gate. Place the drone
        # ``dist`` metres back along the (horizontal) start->gate-0 direction; the z handling above
        # is kept. Ground starts stay grounded regardless (xy only).
        if self.cfg.spawn_distance_m is not None:
            lo, hi = self.cfg.spawn_distance_m
            gate0 = self.P["cen"][self.cid, 0]
            dxy = (gate0 - start)[:, :2]
            dn = dxy.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            u = dxy / dn
            dist = (lo + (hi - lo) * self._rand(n)).unsqueeze(-1)
            new_xy = gate0[:, :2] - u * dist
            pos = pos.clone()
            pos[:, :2] = new_xy
        # UC-89: pick the ground starts whose course is rigidly rotated about the spawn so gate-0 is
        # dead-ahead (+x). θ = −atan2(gate0 − spawn) rotates the start->gate-0 direction onto +x,
        # rotation centre is the finalised spawn xy, so gate-0 lands on +x from where the drone is.
        # Only drawn when the feature is on ⇒ the default path draws no extra RNG.
        if self._gstart_on:
            frac = float(self.cfg.ground_gate_ahead_frac)
            sel = (self._rand(n) < frac) & ground
            gate0 = self.P["cen"][self.cid, 0]
            d = (gate0 - pos)[:, :2]
            theta = -torch.atan2(d[:, 1], d[:, 0])
            c, s = torch.cos(theta), torch.sin(theta)
            z_t, one = torch.zeros_like(theta), torch.ones_like(theta)
            # masked envs take the rotation when selected, else identity; others keep their state.
            self._gstart_active = torch.where(mask, sel, self._gstart_active)
            self._gstart_theta = torch.where(mask, torch.where(sel, theta, z_t), self._gstart_theta)
            self._gstart_c = torch.where(mask, torch.where(sel, c, one), self._gstart_c)
            self._gstart_s = torch.where(mask, torch.where(sel, s, z_t), self._gstart_s)
            self._gstart_cxy = torch.where(mask[:, None], pos[:, :2], self._gstart_cxy)
        # UC-87 AC4: randomise the spawn yaw uniformly over 0..2π (rotation about world z).
        quat_arg = None
        if self.cfg.spawn_yaw:
            yaw = 2.0 * math.pi * self._rand(n)
            half = 0.5 * yaw
            quat_arg = torch.zeros(n, 4, dtype=self.dtype, device=dev)
            quat_arg[:, 0] = torch.cos(half)
            quat_arg[:, 3] = torch.sin(half)
        # UC-87 redesign: apply the deployment-DR ramp. The plant's dynamics-randomisation scale
        # (carrying the tau-override span) and the control-timing scale both ride deploy_dr_scale —
        # a newly-reset env draws its dynamics/timing at the current ramp level. Scale 0 ⇒ nominal
        # plant + clean fixed-rate timing; 1 ⇒ the full configured ranges.
        self._apply_dr_scale()
        self.timing.scale = float(self.deploy_dr_scale)
        self.plant.reset(
            mask, pos=pos, quat=quat_arg, motor_g=motor, grounded=ground, command=command
        )
        self.timing.resample(mask)
        self.start_mode = torch.where(mask, mode, self.start_mode)
        arm = torch.where(ground, sc.arm_delay_s, 0.0).to(torch.float64)
        self.arm_until = torch.where(mask, arm, self.arm_until)
        if self.cfg.shaping:
            self.prev_clear = torch.where(mask, self._clearance(), self.prev_clear)
        if self.cfg.floor:
            fl = (self.pos[:, 2] - FLOOR_KILL_Z).clamp(0.0, FLOOR_CAP)
            self.prev_floor = torch.where(mask, fl, self.prev_floor)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return), self.ep_return)
        # UC-87: zero the policy-memory, penalty accumulators and input-DR state on (re)spawn (D3:
        # zero-seed + per-episode reset). The step-time memory update then skips these done envs.
        zcol = torch.zeros_like(self.prev_action)
        self.prev_action = torch.where(m1, zcol, self.prev_action)
        # UC-88: zero the slew carry + applied-raw on (re)spawn (same D3 boundary as prev_action),
        # so a freshly-spawned env ramps its command from 0 and its memory starts clean.
        self.prev_applied_command = torch.where(m1, zcol, self.prev_applied_command)
        self.prev_applied_raw = torch.where(m1, zcol, self.prev_applied_raw)
        if self.obs_hist is not None:
            self.obs_hist = torch.where(
                mask[:, None, None], torch.zeros_like(self.obs_hist), self.obs_hist
            )  # noqa: E501
        self.ep_arate = torch.where(mask, torch.zeros_like(self.ep_arate), self.ep_arate)
        self.ep_sat = torch.where(mask, torch.zeros_like(self.ep_sat), self.ep_sat)
        if self._gyro_on:
            z3 = torch.zeros_like(self._gyro_ar)
            self._gyro_ar = torch.where(mask[:, None], z3, self._gyro_ar)
            self._gyro_obs = torch.where(mask[:, None], z3, self._gyro_obs)
        if self._gate_on:
            true_rel = self._target_pos() - self.pos
            self._gate_rel = torch.where(mask[:, None], true_rel, self._gate_rel)
            self._gate_age = torch.where(mask, torch.zeros_like(self._gate_age), self._gate_age)
            # per-episode staleness hold (steps an estimate is held), 0..stale/ref_dt, ramped by the
            # input-DR scale (scale 0 ⇒ hold 0 ⇒ always refresh ⇒ zero staleness effect).
            stale_s = self.cfg.gate_staleness_s * float(self.input_dr_scale)
            if stale_s > 0:
                hi = stale_s / self.cfg.reward_reference_dt_s
                hold = (self._perc_rand(n) * hi).round().long()
            else:
                hold = torch.zeros(n, dtype=torch.long, device=dev)
            self._gate_hold = torch.where(mask, hold, self._gate_hold)
        ids = self._episode_counter * n + self._ar_n
        self._episode_counter += 1
        self.episode = torch.where(mask, ids, self.episode)
        header = torch.cat(
            [
                torch.stack(
                    [
                        self._ar_n.float(),
                        self.episode.float(),
                        self.start_mode.float(),
                        self.cid.float(),
                        self._ng().float(),
                        self.P["laps"][self.cid].float(),
                    ],
                    1,
                ),
                self.plant.params.columns().float(),
                self.timing.columns().float(),
            ],
            1,
        )
        self.header = torch.where(m1, header, self.header)

    def _clearance(self) -> torch.Tensor:
        """Clamped distance to the nearest in-path obstacle surface (``CLEAR_CAP`` when far)."""
        oc, od, om = self._ocen(), self._odim(), self._omask()
        cen_d = (oc - self.pos.unsqueeze(1)).norm(dim=-1)
        orad = od[:, :, :2].max(dim=-1).values
        clear = torch.where(om, cen_d - orad, torch.full_like(cen_d, 1e4))
        return clear.min(dim=1).values.clamp(0.0, CLEAR_CAP)

    # ---- observations ----
    def _pre_acq_mask(self) -> torch.Tensor:
        """UC-89: ground-start envs still inside the pre-acquisition neutral window (``(n,)`` bool).
        Empty (all-False) whenever the window is off, so callers stay byte-identical."""
        return (self.start_mode == START_GROUND) & (self.plant.t < self._preacq_s)

    def obs(self) -> torch.Tensor:
        # UC-87 AC4: the perceived (noisy / held) gate-relative vector when gate noise is on, else
        # the true rel; the OBSERVED omega carries the gyro noise (training-only, NOT the plant's
        # true omega). Both default-off ⇒ byte-identical to the model-11 obs.
        rel = self._gate_rel if self._gate_on else (self._target_pos() - self.pos)
        omega = self.omega + self._gyro_obs if self._gyro_on else self.omega
        if self._preacq_on:
            # UC-89 AC2: hover-neutral the observed gate rel during the pre-acquisition window (the
            # live harness emits rel=0 on no-detection). Observation only — reward uses true gate.
            rel = torch.where(self._pre_acq_mask().unsqueeze(-1), torch.zeros_like(rel), rel)
        return torch.cat([rel, gravity_body(self.quat), self.vel, omega], dim=-1)

    def _obstacle_vision(self) -> torch.Tensor:
        """Nearest-NVIS obstacles: rel (dx, dy, dz) + size proxy, masked to real obstacles."""
        if self.cfg.obstacle_vision_zero:  # AC4: live sees a zeroed obstacle_vision block
            return torch.zeros(self.n, NVIS * 4, dtype=self.dtype, device=self.device)
        oc, od, om = self._ocen(), self._odim(), self._omask()
        rel = oc - self.pos.unsqueeze(1)
        dist = torch.where(om, rel.norm(dim=-1), torch.full_like(rel[..., 0], 1e4))
        near = torch.topk(-dist, min(NVIS, MO), dim=1).indices
        ar = self._ar_n.unsqueeze(1)
        rsel = rel[ar, near]
        size = od[ar, near].max(dim=-1).values.unsqueeze(-1)
        valid = om[ar, near].to(self.dtype).unsqueeze(-1)
        return (torch.cat([rsel, size], dim=-1) * valid).reshape(self.n, -1)

    def _next_lookahead(self) -> torch.Tensor:
        """Occlusion-aware next-gate features: bearing(3), orient(2), dist(1), vis(1)."""
        nxt = (self.tgt + 1) % self._ng()
        nc, ny, _ = self._gate(nxt)
        d = nc - self.pos
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        bearing = d / dist
        orient = torch.stack([torch.cos(ny), torch.sin(ny)], -1)
        oc, od, om = self._ocen(), self._odim(), self._omask()
        seg = nc - self.pos
        seglen2 = (seg * seg).sum(-1, keepdim=True) + 1e-6
        to_o = oc - self.pos.unsqueeze(1)
        tt = ((to_o * seg.unsqueeze(1)).sum(-1) / seglen2).clamp(0, 1)
        closest = self.pos.unsqueeze(1) + tt.unsqueeze(-1) * seg.unsqueeze(1)
        dseg = (oc - closest).norm(dim=-1)
        size = od.max(dim=-1).values + 0.15
        blocked = (dseg < size) & om & (tt > 0.02) & (tt < 0.98)
        occ = blocked.any(dim=1, keepdim=True)
        vis = (~occ).to(self.dtype)
        bearing = torch.where(occ, bearing + 0.15 * self._randn(self.n, 3), bearing)
        bearing = bearing / bearing.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        orient = torch.where(occ, torch.zeros_like(orient), orient)
        distn = torch.where(occ, torch.zeros_like(dist), (dist / 10.0).clamp(max=1.0))
        return torch.cat([bearing, orient, distn, vis], dim=-1)

    def _ray_basis(self) -> torch.Tensor:
        """World ray directions ``(n, R, 3)``: the cone around the heading (velocity, or the
        target direction below 0.3 m/s)."""
        R = N_RAYS
        v = self.vel
        sp = v.norm(dim=-1, keepdim=True)
        tdir = self._target_pos() - self.pos
        fwd = torch.where(
            sp > 0.3, v / (sp + 1e-6), tdir / (tdir.norm(dim=-1, keepdim=True) + 1e-6)
        )
        wup = torch.zeros_like(fwd)
        wup[:, 2] = 1.0
        right = torch.cross(fwd, wup, dim=-1)
        rn = right.norm(dim=-1, keepdim=True)
        ex = torch.zeros_like(fwd)
        ex[:, 0] = 1.0
        right = torch.where(rn < 1e-3, ex, right / (rn + 1e-6))
        up = torch.cross(right, fwd, dim=-1)
        c = self.cone
        cx, cy, cz = c[:, 0].view(1, R, 1), c[:, 1].view(1, R, 1), c[:, 2].view(1, R, 1)
        return cx * right.unsqueeze(1) + cy * up.unsqueeze(1) + cz * fwd.unsqueeze(1)

    def _cast(self, rays: torch.Tensor) -> torch.Tensor:
        """Normalised free-space distance in [0, 1] per ray (1 = clear to RAY_RANGE)."""
        n, R = self.n, N_RAYS
        oc, od, om = self._ocen(), self._odim(), self._omask()
        ot = self.P["otype"][self.cid]
        o_e = self.pos.view(n, 1, 1, 3)
        r_e = rays.view(n, R, 1, 3)
        oc_e = oc.view(n, 1, MO, 3)
        od_e = od.view(n, 1, MO, 3)
        om_e = om.view(n, 1, MO)
        ot_e = ot.view(n, 1, MO)
        INF = torch.full((n, R, MO), 1e4, device=self.device, dtype=self.dtype)
        # cylinder (vertical): quadratic in xy + z-slab
        dxy = r_e[..., :2]
        fxy = (o_e - oc_e)[..., :2]
        a = (dxy * dxy).sum(-1)
        b = 2 * (dxy * fxy).sum(-1)
        cc = (fxy * fxy).sum(-1) - od_e[..., 0] ** 2
        disc = b * b - 4 * a * cc
        tcyl = (-b - torch.sqrt(disc.clamp(min=0))) / (2 * a + 1e-9)
        zhit = o_e[..., 2] + tcyl * r_e[..., 2]
        cok = (
            (disc > 0)
            & (tcyl > 1e-3)
            & (zhit > oc_e[..., 2] - od_e[..., 2])
            & (zhit < oc_e[..., 2] + od_e[..., 2])
        )
        tcyl = torch.where(cok, tcyl, INF)
        # box (AABB): slab method
        dsafe = torch.where(r_e.abs() < 1e-6, torch.full_like(r_e, 1e-6), r_e)
        t1 = (oc_e - od_e - o_e) / dsafe
        t2 = (oc_e + od_e - o_e) / dsafe
        tmn = torch.minimum(t1, t2).max(-1).values
        tmx = torch.maximum(t1, t2).min(-1).values
        bhit = torch.where(tmn > 1e-3, tmn, tmx)
        bok = (tmx >= tmn.clamp(min=0)) & (tmx > 1e-3)
        tbox = torch.where(bok, bhit, INF)
        t = torch.where(ot_e == 0, tcyl, tbox)
        t = torch.where(om_e & (t > 0) & (t < RAY_RANGE), t, INF)
        t_hit = t.min(dim=2).values  # (n, R) nearest obstacle hit
        if self.cfg.raycast_floor:  # AC4: the floor plane (z = ground_z) is visible to the rays
            rz = rays[..., 2]
            ground = float(self.plant.config.ground_z)
            tf = (ground - self.pos[:, 2:3]) / rz
            tf = torch.where((rz < -1e-6) & (tf > 1e-3), tf, torch.full_like(tf, 1e4))
            t_hit = torch.minimum(t_hit, tf)
        return t_hit.clamp(max=RAY_RANGE) / RAY_RANGE

    def _raycast(self) -> torch.Tensor:
        return self._cast(self._ray_basis())

    def _gap_reward(self) -> torch.Tensor:
        """Reward flying along the clearest gate-ward cone direction (gap steering)."""
        rays = self._ray_basis()
        free = self._cast(rays)
        tdir = self._target_pos() - self.pos
        gdir = tdir / (tdir.norm(dim=-1, keepdim=True) + 1e-6)
        align = (rays * gdir.unsqueeze(1)).sum(-1).clamp(min=0.0)
        score = free * (0.5 + 0.5 * align)
        best = score.argmax(dim=1)
        best_dir = rays[self._ar_n, best]
        v = self.vel
        sp = v.norm(dim=-1, keepdim=True)
        follow = ((v / (sp + 1e-6)) * best_dir).sum(-1).clamp(min=0.0)
        return follow * sp.squeeze(-1).clamp(max=3.0) / 3.0

    def _ray_obs(self) -> torch.Tensor:
        """The 40 obstacle rays as the observation sees them: exact, blanked, or noise-injected.

        The default (``ray_blank`` and ``ray_noise`` both off) returns the exact cast unchanged and
        draws no extra randomness, so the observation path is byte-identical to today.
        """
        if self.cfg.ray_blank:  # AC8 control: constant clear (NOT zeroed)
            return torch.ones(self.n, N_RAYS, dtype=self.dtype, device=self.device)
        if self._ray_noise is not None:  # AC2/AC5: inject the measured ray error
            basis = self._ray_basis()
            free = self._cast(basis)
            in_view = self._ray_camera.in_view(basis, self.quat)
            noisy = self._ray_noise.apply(free, in_view, self._ray_gen)
            # UC-87 redesign: ramp the ray-noise magnitude with input_dr_scale (0 ⇒ exact rays).
            s = float(self.input_dr_scale)
            if s >= 1.0:
                return noisy
            return free + s * (noisy - free)
        return self._raycast()

    def _memory_block(self) -> torch.Tensor:
        """The D1 memory block: ``prev_applied_raw(4) || obs_history(H*12)`` (newest-first history).

        The first slot is the APPLIED command's effective-raw (UC-88 AC4): ≡ the raw intent when the
        slew filter is off (byte-identical to UC-87), = the post-limiter ``a'`` when it is on."""
        return torch.cat([self.prev_applied_raw, self.obs_hist.reshape(self.n, -1)], dim=-1)

    def extra(self) -> torch.Tensor:
        _, cy, cap = self._gate(self.tgt)
        curr = torch.stack([torch.cos(cy), torch.sin(cy), cap], -1)
        if self._preacq_on:
            # UC-89 AC2: frame-neutral curr_gate [0, 0, aperture] during the pre-acquisition
            # window — identical to the harness no-detection placeholder (cos/sin zeroed, ap kept).
            neutral = torch.stack([torch.zeros_like(cy), torch.zeros_like(cy), cap], -1)
            curr = torch.where(self._pre_acq_mask().unsqueeze(-1), neutral, curr)
        feats = [self.obs(), curr, self._next_lookahead(), self._obstacle_vision()]
        if self.cfg.raycast:
            feats.append(self._ray_obs())
        # UC-87 D2: the memory block is appended STRICTLY AFTER raycast, so columns 0..(pre-memory)
        # are byte-identical to model 11.
        if self.cfg.memory:
            feats.append(self._memory_block())
        return torch.cat(feats, dim=-1)

    def metric(self) -> str:
        base = f"compl {float(self.comp_rate) * 100:4.1f}% lap {float(self.lap.float().mean()):.1f}"
        if self._slew_on:  # UC-88: surface the active slew cap (kept out of the EPISODE_TAIL log)
            base += f" slew {float(self._slew_spec.traversal_time_s):.2f}s"
        return base

    # ---- step ----
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Raw policy action ``(n, 4)`` -> ``(obs, reward, done, tilt)`` (auto-respawn)."""
        action = action.to(self.dtype)
        cg, rates = decode_torch(action, self.action_spec, max_g=self.plant.max_g)
        return self.step_command(cg, rates, raw_action=action)

    def step_command(
        self,
        collective_g: torch.Tensor,
        rates: torch.Tensor,
        *,
        raw_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Like :meth:`step`, from a decoded ``collective_g`` (n,) + body ``rates`` (n, 3).

        ``raw_action`` (the policy's pre-decode output, (n, 4)) drives the UC-87 action-rate /
        saturation penalties and the policy-memory update; it is ``None`` for non-policy callers
        (replay), which leaves both off.
        """
        cfg = self.cfg
        # UC-87 (D1/D3): the core obs the policy observed THIS step (pre-plant-step), pushed into
        # the history ring after the transition so the next step's memory block holds it.
        obs_now = self.obs() if (cfg.memory and raw_action is not None) else None
        arming = self.plant.t < self.arm_until  # ground start: motors idle until armed
        cg = torch.where(arming, torch.zeros_like(collective_g), collective_g)
        rates = torch.where(arming[:, None], torch.zeros_like(rates), rates)
        dt, delay, skipped = self.timing.tick()
        self.last_dt, self.last_skipped = dt, skipped
        rscale = (dt / cfg.reward_reference_dt_s).to(self.dtype)

        # UC-88: clamp the APPLIED command's per-axis slew to the controller-realistic cap, in
        # command space, AFTER arming and BEFORE the plant step (so the plant sees exactly what a
        # real transmitter could produce). ``applied_raw`` is the effective-raw of the applied
        # command, fed to the policy-memory block (AC4). Default-off ⇒ pass-through, applied_raw ≡
        # raw intent, byte-identical to UC-87.
        applied_raw = raw_action
        if self._slew_on:
            new_cmd = torch.cat([cg.unsqueeze(-1), rates], dim=-1)
            applied = slew_clamp(
                self.prev_applied_command,
                new_cmd,
                dt,
                self._slew_spec,
                self.action_spec,
                max_g=float(self.plant.max_g),
            )
            self.prev_applied_command = applied
            cg = applied[:, 0]
            rates = applied[:, 1:4]
            if raw_action is not None:
                applied_raw = raw_of_command(applied, self.action_spec)

        prev_dist = (self._target_pos() - self.pos).norm(dim=-1)
        self.plant.step(cg, rates, dt, delay)
        self.steps = self.steps + 1

        g_body = gravity_body(self.quat)
        cos_tilt = -g_body[:, 2]
        tilt = torch.arccos(torch.clamp(cos_tilt, -1, 1))

        cc, _, cap = self._gate(self.tgt)
        passed = (self.pos - cc).norm(dim=-1) < (cap / 2 + PASS_SLACK * (1.0 - self.difficulty))
        self.tgt = torch.where(passed, self.tgt + 1, self.tgt)
        wrap = self.tgt >= self._ng()
        self.lap = torch.where(wrap, self.lap + 1, self.lap)
        self.tgt = torch.where(wrap, torch.zeros_like(self.tgt), self.tgt)

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        r_prog = 6.0 * (prev_dist - curr_dist)
        r_up = torch.clamp(cos_tilt, 0, 1) * 0.3
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2
        reward = r_prog + (r_up + r_spin) * rscale + 20.0 * passed.to(self.dtype)

        completed = self.lap >= self.P["laps"][self.cid]
        reward = reward + 300.0 * completed.to(self.dtype)

        # obstacle contact -> non-terminal nudge away from the NEAREST touched obstacle
        oc, od, om = self._ocen(), self._odim(), self._omask()
        otype = self.P["otype"][self.cid]
        rel = self.pos.unsqueeze(1) - oc
        dxy = rel[:, :, :2].norm(dim=-1)
        cyl_hit = (dxy < od[:, :, 0] + MARGIN) & (rel[:, :, 2].abs() < od[:, :, 2] + MARGIN)
        box_hit = (rel.abs() < od + MARGIN).all(dim=-1)
        hit = torch.where(otype == 0, cyl_hit, box_hit) & om
        any_hit = hit.any(dim=1)
        dcen = torch.where(
            hit, (oc - self.pos.unsqueeze(1)).norm(dim=-1), torch.full_like(dxy, 1e4)
        )
        ni = dcen.min(dim=1).indices
        away = self.pos[:, :2] - oc[self._ar_n, ni, :2]
        away = away / (away.norm(dim=-1, keepdim=True) + 1e-6)
        vel = self.plant.vel.clone()
        vel[:, :2] = vel[:, :2] + any_hit.to(self.dtype).unsqueeze(-1) * away * BOUNCE
        self.plant.vel = vel
        reward = reward - CONTACT_PEN * any_hit.to(self.dtype) * rscale

        if cfg.shaping:
            clear_now = self._clearance()
            reward = reward + SHAPE_BETA * (clear_now - self.prev_clear)
            self.prev_clear = clear_now
        if cfg.gap:
            reward = reward + GAP_BETA * self._gap_reward() * rscale
        if cfg.floor:
            floor_now = (self.pos[:, 2] - FLOOR_KILL_Z).clamp(0.0, FLOOR_CAP)
            reward = reward + FLOOR_BETA * (floor_now - self.prev_floor)
            self.prev_floor = floor_now

        # UC-87 AC3: action-rate + saturation penalties (rate-scaled, like the dense terms). The
        # previous raw action is ``self.prev_action`` (zero at episode start); it is updated to
        # ``raw_action`` by the memory step after the transition.
        if raw_action is not None and (cfg.action_rate_lambda > 0 or cfg.saturation_weight > 0):
            ta = torch.tanh(raw_action)
            ta_prev = torch.tanh(self.prev_action)
            arate = ((ta - ta_prev) ** 2).sum(-1)
            sat = torch.relu(ta[:, 1:4].abs() - cfg.saturation_threshold).sum(-1)
            # UC-87 redesign: penalty_scale anneals ONLY the action-rate weight (λ mild→target) so
            # completion can establish first; the saturation guard-rail is intentionally un-annealed
            # (small/constant — annealing it adds no AC value).
            eff_lambda = cfg.action_rate_lambda * float(self.penalty_scale)
            reward = reward - (eff_lambda * arate + cfg.saturation_weight * sat) * rscale
            self.ep_arate = self.ep_arate + arate
            self.ep_sat = self.ep_sat + sat

        t = self.plant.t
        xy_off = self.pos[:, :2].norm(dim=-1)
        if self._gstart_on:
            # UC-89: the arena is an origin-centred disk, so it is the one world read NOT relative
            # to the drone. Map the drone's world xy back into the un-rotated course frame (inverse
            # of the per-env course rotation about the spawn) before the origin-distance test, so
            # the arena bounds the rotated scene exactly. Byte-identical for inactive envs.
            cx, cy = self._gstart_cxy[:, 0], self._gstart_cxy[:, 1]
            dx, dy = self.pos[:, 0] - cx, self.pos[:, 1] - cy
            c, s = self._gstart_c, self._gstart_s
            qx = cx + c * dx + s * dy  # R(−θ) about the spawn centre
            qy = cy - s * dx + c * dy
            xy_off = torch.where(
                self._gstart_active, torch.sqrt(qx * qx + qy * qy), xy_off
            )
        grace = t > cfg.floor_grace_s + 1e-6
        ground_start = self.start_mode == START_GROUND
        floor_live = torch.where(ground_start, self.plant.lifted, grace)
        hit_floor = (self.pos[:, 2] < FLOOR_KILL_Z) & floor_live
        hit_ceil = (self.pos[:, 2] > self.ceil_z) & grace
        hit_arena = (xy_off > ARENA) & grace
        crashed = hit_floor | hit_ceil | hit_arena
        if cfg.obs_terminal:
            crashed = crashed | any_hit
        reward = reward - 5.0 * crashed.to(self.dtype)
        timeout = t >= cfg.episode_s - 1e-6
        done = completed | crashed | timeout
        self.last_completed = completed
        self.last_crashed = crashed
        self.last_crash_tilt = tilt
        self.cause_floor = hit_floor
        self.cause_ceil = hit_ceil
        self.cause_arena = hit_arena
        self.cause_obstacle = any_hit if cfg.obs_terminal else torch.zeros_like(crashed)
        self.last_any_hit = any_hit
        self.ep_return = self.ep_return + reward

        # on-device completion EMA over finished episodes (the prototype's metric, no sync)
        nd = done.sum()
        frac = (completed & done).sum().to(self.dtype) / nd.clamp(min=1).to(self.dtype)
        self.comp_rate = torch.where(nd > 0, 0.98 * self.comp_rate + 0.02 * frac, self.comp_rate)

        if self.log is not None:
            outcome = torch.full_like(self.steps, OUTCOMES.index("timeout"))
            for name, flag in (
                ("obstacle", self.cause_obstacle),
                ("arena", hit_arena),
                ("ceiling", hit_ceil),
                ("floor", hit_floor),
                ("completed", completed),
            ):
                outcome = torch.where(flag, OUTCOMES.index(name), outcome)
            tail = torch.stack(
                [
                    self.ep_return.float(),
                    t.float(),
                    self.steps.float(),
                    completed.float(),
                    outcome.float(),
                    self.ep_arate.float(),
                    self.ep_sat.float(),
                ],
                1,
            )
            self.log.add(done, torch.cat([self.header, tail], 1))

        self._spawn(done)
        # UC-87: after the respawn (which zeroes memory/DR for done envs), advance the memory ring
        # and the input-DR for the SURVIVORS, so the next obs()/extra() reflect this transition.
        if raw_action is not None:
            self._update_memory(obs_now, raw_action, applied_raw, done)
        self._update_input_dr(done)
        return self.obs(), reward, done, tilt

    # ---- UC-87 per-step state updates (memory + input domain randomisation) ----
    def _update_memory(
        self,
        obs_now: torch.Tensor | None,
        raw_action: torch.Tensor,
        applied_raw: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Survivors record this step's state and push ``obs_now`` into the history ring
        (newest-first); done envs were already zeroed by :meth:`_spawn`.

        ``prev_action`` carries the raw INTENT (drives the penalty); ``prev_applied_raw`` carries
        the effective-raw of the APPLIED command (drives the memory block, UC-88 AC4). They are
        identical when the slew filter is off, so the memory stays byte-identical to UC-87."""
        surv = (~done)[:, None]
        self.prev_action = torch.where(surv, raw_action, self.prev_action)
        self.prev_applied_raw = torch.where(surv, applied_raw, self.prev_applied_raw)
        if self.obs_hist is not None and obs_now is not None:
            rolled = self.obs_hist.roll(1, dims=1)
            rolled[:, 0] = obs_now
            self.obs_hist = torch.where((~done)[:, None, None], rolled, self.obs_hist)

    def _update_input_dr(self, done: torch.Tensor) -> None:
        """Advance the gyro-noise and gate-perception state for the SURVIVORS (AC4). Done envs keep
        the fresh state :meth:`_spawn` set. No-op (and no RNG drawn) when both are off."""
        surv = ~done
        # UC-87 redesign: the input-DR magnitudes ride ``input_dr_scale``. At scale 0 every noise
        # term is 0 and every hold/dropout probability is 0, so the observed omega equals the true
        # omega and the gate estimate equals the true rel — ZERO EFFECT (not RNG-identity). The ramp
        # opens them linearly to the full cfg magnitudes at scale 1.
        s = float(self.input_dr_scale)
        if self._gyro_on:
            rho = float(self.cfg.gyro_vibration_rho)
            vib = float(self.cfg.gyro_vibration_sigma) * s
            white = float(self.cfg.gyro_noise_sigma) * s
            eps = self._perc_randn(self.n, 3)
            ar = rho * self._gyro_ar + ((1.0 - rho * rho) ** 0.5) * vib * eps
            noise = ar + white * self._perc_randn(self.n, 3)
            if s <= 0.0:  # scale 0 ⇒ no coloured-noise memory can accumulate
                ar = torch.zeros_like(ar)
                noise = torch.zeros_like(noise)
            self._gyro_ar = torch.where(surv[:, None], ar, self._gyro_ar)
            self._gyro_obs = torch.where(surv[:, None], noise, self._gyro_obs)
        if self._gate_on:
            cfg = self.cfg
            true_rel = self._target_pos() - self.pos
            # staleness: hold the estimate while age < per-episode hold; Markov dropout also holds.
            # Both the dropout probability and the noise magnitude scale with input_dr_scale.
            age = self._gate_age + 1
            fresh_due = age > self._gate_hold
            dropped = self._perc_rand(self.n) < (cfg.gate_dropout_p * s)
            refresh = fresh_due & (~dropped)
            # a freshly sampled estimate: true rel + gaussian noise, or an outlier offset
            noise = (cfg.gate_pos_noise_m * s) * self._perc_randn(self.n, 3)
            if cfg.gate_outlier_p > 0:
                outl = self._perc_rand(self.n) < (cfg.gate_outlier_p * s)
                d = self._perc_randn(self.n, 3)
                d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                noise = torch.where(outl[:, None], d * cfg.gate_outlier_m, noise)
            new_rel = true_rel + noise
            r1 = refresh[:, None]
            self._gate_rel = torch.where(surv[:, None] & r1, new_rel, self._gate_rel)
            self._gate_age = torch.where(
                surv & refresh, torch.zeros_like(age), torch.where(surv, age, self._gate_age)
            )  # noqa: E501
