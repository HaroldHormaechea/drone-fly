"""UC70 AC3 / AC9: student-observation parity (``student-obs`` v2, 61 float32).

* The live harness (:class:`~drone_fly.harness.student.StudentObservationAssembler`) and the
  training-side reference (:func:`drone_fly.student.obs_ref.build_student_obs`) are BIT-identical
  (``np.array_equal``) for identical inputs over several ticks, and the ``harness30`` prefix is the
  REAL ``ContractObservationAssembler``'s 30-dim output.
* The batched torch twin (:func:`drone_fly.sim.student_obs.build_student_obs`, what the simulator
  trains on) matches the reference within the stated tolerance (atol 1e-5) for random perception
  states, including unacquired targets, visible / unseen next gates, hand-overs and ages.
* ``k`` scaling equals the harness ``WorldScale(k)``.
* v2's battery block [59:61) is ``battery_from_telemetry(batt_charge, batt_volts, cells)`` of the
  tick's ``FlightState.battery`` (Liftoff's raw ``(charge, volts)``); a tick without the pack
  reading is a coverage error.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.student import StudentObservationAssembler, StudentStepState
from drone_fly.harness.types import FlightState, FreeSpaceEstimate
from drone_fly.sim.perception import (
    NEXT_UNSEEN_KNOWN,
    NEXT_UNSEEN_NONE,
    NEXT_UNSEEN_TARGET,
    NEXT_VISIBLE,
    PerceptionConfig,
)
from drone_fly.sim.student_obs import (
    BLOCKS,
    DIM,
    NAME,
    VERSION,
    HistoryState,
    block_slice,
    input_scale,
    schema,
    teacher_battery_schema,
)
from drone_fly.sim.student_obs import build_student_obs as build_torch
from drone_fly.student.obs_ref import StudentHistory, build_student_obs, scale_state
from drone_fly.student.perception_ref import estimate_from_raw
from drone_fly.tracks.teacher_obs_ref import battery_from_telemetry
from tests.student_support import CAMERA

K = 6.0
LOST = 1.3
NOMINAL = 3.69
CELLS = 3


def with_pack(state, charge, volts):
    """``state`` carrying Liftoff's raw battery block ``(batt_charge, batt_volts)``."""
    return dataclasses.replace(state, battery=np.array([charge, volts], np.float32))


def ref_pack(state):
    """The reference's battery argument for a state built by :func:`with_pack`."""
    return battery_from_telemetry(float(state.battery[0]), float(state.battery[1]), CELLS)


def rand_quat(g):
    q = g.normal(size=4)
    q[0] = abs(q[0]) + 1.0  # mostly upright
    return q / np.linalg.norm(q)


def raw_perception(g, pos, *, mode=None, acquired=None, hand_over=None, matched=None):
    acq = bool(g.random() < 0.8) if acquired is None else acquired
    mode = int(g.integers(0, 4)) if mode is None else mode
    if not acq and mode in (NEXT_VISIBLE, NEXT_UNSEEN_KNOWN):
        mode = NEXT_UNSEEN_NONE
    target = pos + g.uniform(-20, 20, 3) if acq else pos.copy()
    if mode == NEXT_UNSEEN_TARGET:
        nc = target
    elif mode == NEXT_UNSEEN_NONE:
        nc = pos + np.array([1.0, 0.0, 0.0])
    else:
        nc = pos + g.uniform(-40, 40, 3)
    return {
        "status": int(g.integers(0, 3)),
        "target": target,
        "target_yaw": float(g.uniform(-math.pi, math.pi)) if acq else 0.0,
        "acquired": acq,
        "next_mode": mode,
        "next_center": nc,
        "next_yaw": float(g.uniform(-math.pi, math.pi)),
        "target_age_s": float(g.uniform(0, 2.5)) if acq else LOST,
        "next_age_s": float(g.uniform(0, 2.5)),
        "target_valid": acq,
        "hand_over": bool(g.random() < 0.15) if hand_over is None else hand_over,
        "passed": False,
        "matched": (acq and bool(g.random() < 0.6)) if matched is None else matched,
    }


def cfg() -> PerceptionConfig:
    return PerceptionConfig(mode="exact", lost_timeout_s=LOST, nominal_aperture=NOMINAL)


# ----------------------------------------------------------------------------- harness == reference


class Clock:
    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


def test_harness_assembler_is_bit_identical_to_the_reference(harness_contract):
    g = np.random.default_rng(0)
    times = np.cumsum(g.uniform(0.02, 0.035, 12))
    step = StudentStepState(arm_total_s=0.5, clock=Clock(times))
    asm = StudentObservationAssembler(harness_contract, step, lost_timeout_s=LOST)
    assert asm.required_blocks(DIM) == ("target", "curr_gate", "next_lookahead")
    hist = StudentHistory()
    prev_t = None
    for i, now in enumerate(times):
        pos = g.uniform(-10, 10, 3)
        raw = raw_perception(g, pos)
        est = estimate_from_raw(raw, pos, cfg(), K, float(now))
        state = scale_state({"pos": pos, "vel": g.uniform(-8, 8, 3), "quat": rand_quat(g),
                             "omega": g.uniform(-3, 3, 3)}, K)  # fmt: skip
        state = with_pack(state, g.uniform(0.0, 1.0), g.uniform(9.9, 12.6))  # a draining 3S
        prev = g.uniform(-1.5, 1.5, 4).astype(np.float32)
        step.prev_action = prev.copy()
        arm = step.arm_remaining_s(float(est.stamp))
        obs, extra = asm.assemble(state, est, None, DIM)
        dt = 0.0 if prev_t is None else float(now - prev_t)
        prev_t = now
        want = build_student_obs(harness_contract, state, est, prev_action=prev,
                                 arm_remaining_s=arm, lost_timeout_s=LOST, history=hist,
                                 dt=dt, battery=ref_pack(state))  # fmt: skip
        assert extra.dtype == np.float32 and extra.shape == (DIM,)
        assert np.array_equal(extra, want), (i, np.flatnonzero(extra != want))
        assert np.array_equal(obs, extra[:12])
        # the prefix IS the contract assembler's 30-dim observation (obstacle_vision zeros)
        _, h30 = ContractObservationAssembler(harness_contract).assemble(
            state, est, FreeSpaceEstimate(obstacle_vision=np.zeros(8, np.float32)), 30
        )
        assert np.array_equal(extra[block_slice("harness30")], h30)
        assert np.all(extra[22:30] == 0.0)
        np.testing.assert_allclose(extra[block_slice("battery")], ref_pack(state), rtol=1e-6)
    with pytest.raises(ValueError, match=f"{DIM} values, not 30"):
        step.clock = Clock([99.0])
        asm.assemble(state, est, None, 30)


def test_harness_session_reset_restarts_the_history(harness_contract):
    step = StudentStepState(clock=Clock([0.0, 0.03, 10.0]))
    asm = StudentObservationAssembler(harness_contract, step, lost_timeout_s=LOST)
    g = np.random.default_rng(1)
    pos = np.zeros(3)
    raw = raw_perception(g, pos, acquired=True, matched=True, hand_over=False)
    state = scale_state({"pos": pos, "vel": np.zeros(3), "quat": [1, 0, 0, 0],
                         "omega": np.zeros(3)}, K)  # fmt: skip
    state = with_pack(state, 1.0, 12.6)
    est = estimate_from_raw(raw, pos, cfg(), K, 0.0)
    first = asm.assemble(state, est, None, DIM)[1]
    asm.assemble(state, est, None, DIM)
    step.reset()
    assert step.last_now is None and not step.history.init
    again = asm.assemble(state, est, None, DIM)[1]
    assert np.array_equal(first, again)


def test_harness_battery_block_from_liftoff_telemetry(harness_contract):
    step = StudentStepState(clock=Clock([0.0]))
    asm = StudentObservationAssembler(harness_contract, step, lost_timeout_s=LOST,
                                      battery_cells=CELLS)  # fmt: skip
    g = np.random.default_rng(5)
    pos = np.zeros(3)
    raw = raw_perception(g, pos, acquired=True, matched=True, hand_over=False)
    state = scale_state({"pos": pos, "vel": np.zeros(3), "quat": [1, 0, 0, 0],
                         "omega": np.zeros(3)}, K)  # fmt: skip
    est = estimate_from_raw(raw, pos, cfg(), K, 0.0)
    extra = asm.assemble(with_pack(state, 0.8, 11.7), est, None, DIM)[1]
    assert block_slice("battery") == slice(59, 61)
    np.testing.assert_allclose(extra[59:61], [0.8, 3.9], rtol=1e-6)  # (charge, 11.7 V / 3)
    # the schema's input scale makes a full pack (1.0, 4.2 V per cell) read as (1, 1)
    np.testing.assert_allclose(np.array(input_scale()[59:61]) * [1.0, 4.2], [1.0, 1.0])


def test_harness_tick_without_the_pack_reading_is_a_coverage_error(harness_contract):
    step = StudentStepState(clock=Clock([0.0, 0.03]))
    asm = StudentObservationAssembler(harness_contract, step, lost_timeout_s=LOST)
    g = np.random.default_rng(6)
    pos = np.zeros(3)
    raw = raw_perception(g, pos, acquired=True, matched=True, hand_over=False)
    state = scale_state({"pos": pos, "vel": np.zeros(3), "quat": [1, 0, 0, 0],
                         "omega": np.zeros(3)}, K)  # fmt: skip
    assert state.battery is None
    est = estimate_from_raw(raw, pos, cfg(), K, 0.0)
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.assemble(state, est, None, DIM)
    assert ei.value.missing == ("battery",)


# ----------------------------------------------------------------------------- torch == reference


def test_torch_twin_matches_the_reference(harness_contract):
    g = np.random.default_rng(2)
    n, ticks = 64, 8
    th = HistoryState.zeros(n, "cpu")
    rh = [StudentHistory() for _ in range(n)]
    modes_seen = set()
    for tick in range(ticks):
        pos = g.uniform(-15, 15, (n, 3))
        vel = g.uniform(-10, 10, (n, 3))
        quat = np.stack([rand_quat(g) for _ in range(n)])
        omega = g.uniform(-4, 4, (n, 3))
        prev = g.uniform(-1.5, 1.5, (n, 4))
        arm = g.uniform(0, 0.5, n)
        dt = np.full(n, 0.0 if tick == 0 else 0.025)
        raws = [raw_perception(g, pos[j]) for j in range(n)]
        packs = [battery_from_telemetry(g.uniform(0, 1), g.uniform(9.9, 12.6), CELLS)
                 for _ in range(n)]  # fmt: skip
        F = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
        perc = {
            "target": F([r["target"] for r in raws]),
            "target_yaw": F([r["target_yaw"] for r in raws]),
            "acquired": torch.tensor([r["acquired"] for r in raws]),
            "next_mode": torch.tensor([r["next_mode"] for r in raws]),
            "next_center": F([r["next_center"] for r in raws]),
            "next_yaw": F([r["next_yaw"] for r in raws]),
            "target_age_s": F([r["target_age_s"] for r in raws]),
            "next_age_s": F([r["next_age_s"] for r in raws]),
            "target_valid": torch.tensor([r["target_valid"] for r in raws]),
            "hand_over": torch.tensor([r["hand_over"] for r in raws]),
            "matched": torch.tensor([r["matched"] for r in raws]),
        }
        got = build_torch(pos=F(pos), vel=F(vel), quat=F(quat), omega=F(omega), perc=perc, k=K,
                          nominal_aperture=NOMINAL, lost_timeout_s=LOST, prev_action=F(prev),
                          arm_remaining_s=F(arm), history=th,
                          dt=torch.tensor(dt, dtype=torch.float64), battery=F(packs))  # fmt: skip
        assert got.shape == (n, DIM)
        for j in range(n):
            est = estimate_from_raw(raws[j], pos[j], cfg(), K, 0.0)
            st = scale_state({"pos": pos[j], "vel": vel[j], "quat": quat[j], "omega": omega[j]}, K)
            want = build_student_obs(harness_contract, st, est,
                                     prev_action=prev[j].astype(np.float32),
                                     arm_remaining_s=float(arm[j]), lost_timeout_s=LOST,
                                     history=rh[j], dt=float(dt[j]),
                                     battery=packs[j])  # fmt: skip
            diff = np.abs(got[j].numpy() - want)
            bad = [name for name, _ in BLOCKS if diff[block_slice(name)].max() > 1e-5]
            assert not bad, (tick, j, bad, diff.max(), raws[j]["next_mode"])
            modes_seen.add(raws[j]["next_mode"])
    assert modes_seen == {0, 1, 2, 3}


# ----------------------------------------------------------------------------- k scaling + schema


def test_k_scaling_equals_the_harness_world_scale():
    g = np.random.default_rng(3)
    s = {"pos": g.uniform(-30, 30, 3), "vel": g.uniform(-9, 9, 3), "quat": rand_quat(g),
         "omega": g.uniform(-3, 3, 3)}  # fmt: skip
    for k in (1.0, 4.0, 6.0, 10.0):
        st = scale_state(s, k)
        ws = WorldScale(k)
        assert np.array_equal(st.pos, ws.vector(s["pos"]))
        assert np.array_equal(st.vel, ws.vector(s["vel"]))
        np.testing.assert_array_equal(st.quat, np.asarray(s["quat"], np.float32))
        np.testing.assert_array_equal(st.omega, np.asarray(s["omega"], np.float32))
    fs = FlightState(
        pos=np.ones(3, np.float32) * 12,
        vel=np.ones(3, np.float32) * 6,
        quat=np.array([1, 0, 0, 0], np.float32),
        omega=np.zeros(3, np.float32),
        t=3.0,
    )
    st = scale_state(fs, 6.0)
    assert st.t == 3.0 and np.allclose(st.pos, 2.0) and np.allclose(st.vel, 1.0)


def test_torch_obs_at_k_equals_prescaled_inputs_at_k1():
    g = np.random.default_rng(4)
    n = 16
    F = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    pos, vel = F(g.uniform(-30, 30, (n, 3))), F(g.uniform(-9, 9, (n, 3)))
    quat = F(np.stack([rand_quat(g) for _ in range(n)]))
    omega = F(g.uniform(-3, 3, (n, 3)))
    perc = {"target": pos + F(g.uniform(-20, 20, (n, 3))), "target_yaw": F(g.uniform(-3, 3, n)),
            "acquired": torch.ones(n, dtype=torch.bool),
            "next_mode": torch.zeros(n, dtype=torch.long),
            "next_center": pos + F(g.uniform(-40, 40, (n, 3))), "next_yaw": F(g.uniform(-3, 3, n)),
            "target_age_s": F(g.uniform(0, 1, n)), "next_age_s": F(g.uniform(0, 1, n)),
            "target_valid": torch.ones(n, dtype=torch.bool),
            "hand_over": torch.zeros(n, dtype=torch.bool),
            "matched": torch.ones(n, dtype=torch.bool)}  # fmt: skip
    common = dict(
        quat=quat,
        omega=omega,
        lost_timeout_s=LOST,
        prev_action=F(np.zeros((n, 4))),
        arm_remaining_s=F(np.zeros(n)),
        dt=torch.zeros(n, dtype=torch.float64),
    )
    a = build_torch(pos=pos, vel=vel, perc=perc, k=K, nominal_aperture=NOMINAL,
                    history=HistoryState.zeros(n, "cpu"), **common)  # fmt: skip
    p1 = dict(perc, target=perc["target"] / K, next_center=perc["next_center"] / K)
    b = build_torch(pos=pos / K, vel=vel / K, perc=p1, k=1.0, nominal_aperture=NOMINAL / K,
                    history=HistoryState.zeros(n, "cpu"), **common)  # fmt: skip
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_schema_pins_the_layout_and_parameters():
    s = schema(world_scale=K, lost_timeout_s=LOST, acquire_timeout_s=3.0, camera=CAMERA)
    assert (NAME, VERSION, DIM) == ("student-obs", 2, 61)
    assert (s["name"], s["version"], s["dim"], s["dtype"]) == (NAME, VERSION, DIM, "float32")
    assert [(b["name"], b["width"]) for b in s["blocks"]] == list(BLOCKS)
    assert sum(w for _, w in BLOCKS) == DIM
    assert s["blocks"][-1] == {"name": "battery", "start": 59, "width": 2}
    assert s["blocks"][-1]["start"] + s["blocks"][-1]["width"] == DIM
    # the battery block is teacher-obs v2's, plus the pack's cell count
    assert s["battery"] == {**teacher_battery_schema(), "cells": 3}
    assert s["battery"]["features"] == ["charge", "volts_per_cell"]
    assert schema(world_scale=K, lost_timeout_s=LOST, acquire_timeout_s=3.0, camera=CAMERA,
                  battery_cells=4)["battery"]["cells"] == 4  # fmt: skip
    assert s["world_scale"] == K and s["tracker"]["lost_timeout_s"] == LOST
    assert s["camera"] == {"resolution": [640, 480], "hfov_deg": 90.0, "uptilt_deg": 0.0}
    assert s["harness30"]["obstacle_vision"] == "zeros" and s["harness30"]["extra_dim"] == 30
    assert s["requires"] == {
        "noise_model": True,
        "gate_estimator": "learned",
        "free_space_estimator": "none",
        "assume_no_obstacles": True,
    }
    assert len(s["input_scale"]) == DIM == len(input_scale())
    assert block_slice("perception") == slice(44, 47) and block_slice("history") == slice(47, 59)
    assert input_scale()[59:61] == pytest.approx([1.0, 1 / 4.2])
