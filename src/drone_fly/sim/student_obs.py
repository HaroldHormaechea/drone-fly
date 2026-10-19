"""The deployable student's observation, schema ``student-obs`` v2: 61 float32 (UC70 D4).

Everything the live harness can assemble from the learned gate detector, the telemetry and its
own action history, in **student units** (Liftoff metres / ``k``, the harness ``WorldScale(k)``;
rates, quaternions and times unscaled). Blocks, in order (:data:`BLOCKS`):

1. ``harness30`` [0:30): ``ContractObservationAssembler.assemble(state_k, GateEstimate_k,
   FreeSpaceEstimate(obstacle_vision = zeros), 30)``: the contract ``make_obs`` (rel to the
   PERCEIVED target, gravity in body, velocity, body rates), ``curr_gate`` (perceived yaw, the
   detector class's nominal aperture), ``next_lookahead`` (visible / unseen forms of the
   perceived next gate) and ``obstacle_vision`` = zeros (``free_space: none`` +
   ``assume_no_obstacles``).
2. ``attitude6`` [30:36): the first two columns of R(body -> world).
3. ``vel_body`` [36:39): body-frame velocity / k.
4. ``prev_action`` [39:43): the RAW policy output of the previous tick (before decoding /
   clipping / the arm override).
5. ``arm_remaining_s`` [43].
6. ``perception`` [44:47): ``min(target_age, T) / T``, ``target_valid``, ``min(next_age, T) / T``
   with ``T = lost_timeout_s``; age = now - the capture stamp of the frame behind the last match.
7. ``history`` [47:59): EMA (tau :data:`EMA_TAU_S`, ``alpha = 1 - exp(-dt / tau)``) of the
   body-frame target vector / k (3) and of ``prev_action`` (4); the change of the WORLD-frame
   target estimate since the previous tick, rotated into the current body frame (3; zero unless
   the target matched this tick, so zero while holding even under pure rotation, and zero on a
   hand-over); ticks since the last match, ``min(n, TICKS_NORM) / TICKS_NORM`` (1); the hand-over
   flag (1). Reset at episode / session start (the EMAs start at the first tick's values).
8. ``battery`` [59:61) (v2): ``charge`` (Liftoff's ``batt_charge``, 0..1) and the pack voltage per
   cell (``batt_volts / cells``, V) — exactly ``teacher-obs`` v2's battery block
   (``tracks.teacher_obs_ref.battery_from_telemetry``); in the sim, the plant's power model
   (a full pack without one).

The NumPy reference (``drone_fly.student.obs_ref.build_student_obs``) calls the REAL harness
assembler; the harness's ``StudentObservationAssembler`` calls that same function. This module
holds the schema and the batched torch twin (tolerance-tested: atol 1e-5, not bit-identical).
No import of ``drone_fly.harness`` or ``drone_fly.student``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drone_fly.sim.perception import NEXT_VISIBLE
from drone_fly.sim.quat import gravity_body

NAME = "student-obs"
VERSION = 3
BLOCKS: tuple[tuple[str, int], ...] = (
    ("harness30", 30),
    ("attitude6", 6),
    ("vel_body", 3),
    ("prev_action", 4),
    ("arm_remaining_s", 1),
    ("perception", 3),
    ("history", 12),
    ("battery", 2),
    # v3 (UC70 run-2): the next-gate pointer — a body-frame unit bearing to the current target
    # gate (3; Liftoff's on-screen next-gate HUD arrow) and an availability bit (1). A noisy HUD
    # proxy; appended at the tail so the legacy [0:61] columns and offsets are unchanged and the
    # warm start zeros these new columns.
    ("pointer", 4),
)
DIM = sum(w for _, w in BLOCKS)
HARNESS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("obs", 12),
    ("curr_gate", 3),
    ("next_lookahead", 7),
    ("obstacle_vision", 8),
)
#: The harness30 prefix's ``make_obs`` source (as ``teacher_obs``; the harness checks it).
OBS_SOURCE = "contract make_obs (gpu_prototype/inference.py, INFERENCE_CONTRACT 1)"
ASSEMBLER = "harness.assembler.ContractObservationAssembler"
ASSEMBLER_VERSION = 1
EMA_TAU_S = 0.25
TICKS_NORM = 50
LOOKAHEAD_DIST_NORM = 10.0


def block_slice(name: str) -> slice:
    start = 0
    for n, w in BLOCKS:
        if n == name:
            return slice(start, start + w)
        start += w
    raise KeyError(name)


def input_scale() -> list[float]:
    """Fixed per-value multipliers for networks that read the raw observation (recorded in the
    schema). ``harness30`` uses the teacher's; the K1 readout keeps its legacy 30 columns RAW (the
    warm start) and applies this to the other 29 only."""
    from drone_fly.sim.teacher_obs import input_scale as teacher_scale

    out = list(teacher_scale()[:30])
    out += [1.0] * 6  # attitude6
    out += [1 / 10.0] * 3  # vel_body
    out += [1.0] * 4  # prev_action
    out += [2.0]  # arm_remaining_s
    out += [1.0] * 3  # perception
    out += [1 / 20.0] * 3 + [1.0] * 4 + [1.0] * 3 + [1.0, 1.0]  # history
    out += list(teacher_scale(2)[-2:])  # battery: as teacher-obs v2 (charge, volts / 4.2)
    out += [1.0] * 4  # pointer: a unit bearing (3, already O(1)) + the availability bit (1)
    assert len(out) == DIM
    return out


def schema(
    *,
    world_scale: float,
    lost_timeout_s: float,
    acquire_timeout_s: float,
    camera: dict[str, Any],
    assoc_radius: float = 1.0,
    battery_cells: int = 3,
    lost_policy: str = "terminate",
    lost_hard_stop_s: float = 6.0,
    drop_after_s: float = 0.5,
    pointer_bias_sigma_rad: float = 0.0,
    pointer_jitter_sigma_rad: float = 0.0,
    pointer_dropout_p: float = 0.0,
    pointer_bearing_tol_deg: float = 20.0,
    privileged: tuple[str, ...] = (),
    privileged_version: int | None = None,
) -> dict[str, Any]:
    """The JSON-able ``obs_schema`` written into the student checkpoint (pins ``k``, the tracker
    timeouts and the camera the noise model and perception assumed).

    UC79: ``privileged`` (default empty) is the opt-in list of diagnostic teacher blocks appended
    to the obs. When empty the schema is **byte-identical** to before (nothing added; AC3). When
    set, ``dim`` grows by the privileged widths, the privileged block layout is appended to
    ``blocks`` and a top-level ``privileged`` key records the names/widths (always exact).
    ``privileged_version`` is the teacher-obs version whose encoding the blocks mirror (only
    ``teacher_full``'s width depends on it); defaults to the current teacher-obs version."""
    from drone_fly.sim import teacher_obs as _t

    priv = _t._privileged_canonical(privileged)
    pv = int(privileged_version) if privileged_version is not None else _t.VERSION
    start = 0
    blocks = []
    for n, w in BLOCKS:
        blocks.append({"name": n, "start": start, "width": w})
        start += w
    dim = DIM
    if priv:
        for entry in _t.privileged_layout(priv, pv):
            blocks.append({**entry, "start": entry["start"] + DIM})
        dim = DIM + _t.privileged_total_width(priv, pv)
    out = {
        "name": NAME,
        "version": VERSION,
        "dim": dim,
        "dtype": "float32",
        "units": "student units = Liftoff metres / world_scale (harness WorldScale); rates, "
        "quaternions, times unscaled; body FLU",
        "world_scale": float(world_scale),
        "blocks": blocks,
        "harness30": {
            "layout": [list(x) for x in HARNESS_LAYOUT],
            "extra_dim": 30,
            "assembler": ASSEMBLER,
            "assembler_version": ASSEMBLER_VERSION,
            "obs_source": OBS_SOURCE,
            "obstacle_vision": "zeros",
            "lookahead_dist_norm": LOOKAHEAD_DIST_NORM,
        },
        "tracker": {
            "lost_timeout_s": float(lost_timeout_s),
            "acquire_timeout_s": float(acquire_timeout_s),
            "assoc_radius": float(assoc_radius),
            "lost_policy": str(lost_policy),
            "lost_hard_stop_s": float(lost_hard_stop_s),
            "drop_after_s": float(drop_after_s),
        },
        "camera": {
            "resolution": [int(v) for v in camera["resolution"]],
            "hfov_deg": float(camera["hfov_deg"]),
            "uptilt_deg": float(camera.get("uptilt_deg", 0.0)),
        },
        "history": {"ema_tau_s": EMA_TAU_S, "ticks_norm": TICKS_NORM},
        "pointer": {
            "dim": 4,
            "layout": "body-frame unit bearing (3) + availability bit (1)",
            "bias_sigma_rad": float(pointer_bias_sigma_rad),
            "jitter_sigma_rad": float(pointer_jitter_sigma_rad),
            "dropout_p": float(pointer_dropout_p),
            "bearing_tol_deg": float(pointer_bearing_tol_deg),
            "source": "sim: noisy truth bearing; live: UC74 classic-OpenCV HUD reader when a "
            "camera.pointer reader is configured, else unavailable (bit = 0)",
        },
        "input_scale": input_scale(),
        "battery": {**teacher_battery_schema(), "cells": int(battery_cells)},
        "requires": {
            "noise_model": True,
            "gate_estimator": "learned",
            "free_space_estimator": "none",
            "assume_no_obstacles": True,
        },  # fmt: skip
        "prev_action": "raw policy output of the previous tick, before decoding / clipping",
    }
    if priv:
        out["input_scale"] = input_scale() + _t.privileged_input_scale(priv, pv)
        out["privileged"] = {
            "blocks": list(priv),
            "widths": {b: _t.privileged_width(b, pv) for b in priv},
            "teacher_obs_version": pv,
            "exactness": "always exact (computed by the teacher's own code path on env truth, "
            "regardless of the student's perception mode)",
            "role": "diagnostic; a student with any privileged block is NOT deployable "
            "(train export / harness run refuse it)",
        }
    return out


def teacher_battery_schema() -> dict[str, Any]:
    """``teacher-obs`` v2's battery block definitions (the student copies them)."""
    from drone_fly.sim.teacher_obs import schema as teacher_schema

    return dict(teacher_schema(2)["battery"])


def _unit(d: torch.Tensor) -> torch.Tensor:
    """``course_senses._unit``: normalise with a 1e-3 floor, twice."""
    b = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    return b / b.norm(dim=-1, keepdim=True).clamp(min=1e-3)


@dataclass
class HistoryState:
    """Batched twin of ``student.obs_ref.StudentHistory``."""

    ema_rel: torch.Tensor
    ema_act: torch.Tensor
    prev_target: torch.Tensor
    prev_valid: torch.Tensor
    ticks: torch.Tensor
    init: torch.Tensor

    @classmethod
    def zeros(cls, n: int, device: Any, dtype: torch.dtype = torch.float32) -> HistoryState:
        z = lambda *s: torch.zeros(*s, dtype=dtype, device=device)  # noqa: E731
        b = torch.zeros(n, dtype=torch.bool, device=device)
        return cls(
            z(n, 3),
            z(n, 4),
            z(n, 3),
            b.clone(),
            torch.zeros(n, dtype=torch.long, device=device),
            b.clone(),
        )

    def reset(self, mask: torch.Tensor) -> None:
        m1 = mask.unsqueeze(-1)
        self.ema_rel = torch.where(m1, torch.zeros_like(self.ema_rel), self.ema_rel)
        self.ema_act = torch.where(m1, torch.zeros_like(self.ema_act), self.ema_act)
        self.prev_target = torch.where(m1, torch.zeros_like(self.prev_target), self.prev_target)
        self.prev_valid = self.prev_valid & ~mask
        self.init = self.init & ~mask
        self.ticks = torch.where(mask, torch.zeros_like(self.ticks), self.ticks)


def history_step(
    h: HistoryState,
    R: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    prev_action: torch.Tensor,
    matched: torch.Tensor,
    hand_over: torch.Tensor,
    valid: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Advance the history and return its 12 values (student units in, student units out)."""
    rel_b = (R * (target - pos).unsqueeze(-1)).sum(-2)
    a = (1.0 - torch.exp(-dt.to(rel_b.dtype) / EMA_TAU_S)).unsqueeze(-1)
    a = torch.where(dt.unsqueeze(-1) > 0, a, torch.ones_like(a))
    first = ~h.init
    h.ema_rel = torch.where(first.unsqueeze(-1), rel_b, h.ema_rel + a * (rel_b - h.ema_rel))
    pa = prev_action.to(rel_b.dtype)
    h.ema_act = torch.where(first.unsqueeze(-1), pa, h.ema_act + a * (pa - h.ema_act))
    h.init = torch.ones_like(h.init)
    use = matched & ~hand_over & h.prev_valid
    dw = torch.where(use.unsqueeze(-1), target - h.prev_target, torch.zeros_like(target))
    delta = (R * dw.unsqueeze(-1)).sum(-2)
    h.prev_target = target.clone()
    h.prev_valid = valid.clone()
    h.ticks = torch.where(matched, torch.zeros_like(h.ticks), h.ticks + 1)
    tn = (h.ticks.clamp(max=TICKS_NORM).to(rel_b.dtype) / TICKS_NORM).unsqueeze(-1)
    return torch.cat([h.ema_rel, h.ema_act, delta, tn, hand_over.to(rel_b.dtype).unsqueeze(-1)],
                     -1)  # fmt: skip


def build_student_obs(
    *,
    pos: torch.Tensor,
    vel: torch.Tensor,
    quat: torch.Tensor,
    omega: torch.Tensor,
    perc: dict[str, torch.Tensor],
    k: float,
    nominal_aperture: float,
    lost_timeout_s: float,
    prev_action: torch.Tensor,
    arm_remaining_s: torch.Tensor,
    history: HistoryState,
    dt: torch.Tensor,
    battery: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(n, 59)`` student observations from course-frame state + :meth:`GatePerception.outputs`
    (``k`` = course units per student unit, i.e. the student's world scale for a
    world_scale-1 course)."""
    from drone_fly.sim.multi_course import quat_matrix

    dt_ = pos.dtype
    pk, vk = pos / k, vel / k
    tk = perc["target"].to(dt_) / k
    obs12 = torch.cat([tk - pk, gravity_body(quat), vk, omega], -1)
    acq = perc["acquired"]
    y = perc["target_yaw"].to(dt_)
    ap = torch.full_like(y, nominal_aperture / k)
    curr = torch.where(
        acq.unsqueeze(-1),
        torch.stack([torch.cos(y), torch.sin(y), ap], -1),
        # UC-89: frame-neutral no-detection curr_gate [0, 0, ap] (torch twin of perception_ref /
        # gate_cv). ~acq branch only — the acquired path and aperture are byte-identical.
        torch.stack([torch.zeros_like(y), torch.zeros_like(y), ap], -1),
    )
    nc = perc["next_center"].to(dt_) / k
    d = nc - pk
    dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    bearing = _unit(d)
    ny = perc["next_yaw"].to(dt_).unsqueeze(-1)
    vis = torch.cat(
        [
            bearing,
            torch.cos(ny),
            torch.sin(ny),
            (dist / LOOKAHEAD_DIST_NORM).clamp(max=1.0),
            torch.ones_like(dist),
        ],
        -1,
    )
    unseen = torch.cat([bearing, torch.zeros_like(bearing), torch.zeros_like(dist)], -1)
    look = torch.where((perc["next_mode"] == NEXT_VISIBLE).unsqueeze(-1), vis, unseen)
    h30 = torch.cat([obs12, curr, look, torch.zeros(pos.shape[0], 8, dtype=dt_,
                                                    device=pos.device)], -1)  # fmt: skip
    R = quat_matrix(quat)
    vb = (R * vk.unsqueeze(-1)).sum(-2)
    T = float(lost_timeout_s)
    perc3 = torch.stack([
        perc["target_age_s"].to(dt_).clamp(max=T) / T,
        perc["target_valid"].to(dt_),
        perc["next_age_s"].to(dt_).clamp(max=T) / T,
    ], -1)  # fmt: skip
    hist = history_step(history, R, pk, tk, prev_action, perc["matched"], perc["hand_over"],
                        perc["target_valid"], dt)  # fmt: skip
    if battery is None:  # a full pack (no power model), as teacher-obs v2
        from drone_fly.sim.teacher_obs import FULL_CELL_V

        battery = pos.new_tensor([1.0, FULL_CELL_V]).expand(pos.shape[0], 2)
    pointer = perc["pointer"].to(dt_)  # body bearing (3) + availability bit (1), already unitless
    parts = [h30, R[:, :, 0], R[:, :, 1], vb, prev_action.to(dt_),
             arm_remaining_s.to(dt_).unsqueeze(-1), perc3, hist, battery.to(dt_), pointer]  # fmt: skip
    return torch.cat(parts, -1)


def ema_alpha(dt: float) -> float:
    return 1.0 - math.exp(-dt / EMA_TAU_S) if dt > 0 else 1.0
