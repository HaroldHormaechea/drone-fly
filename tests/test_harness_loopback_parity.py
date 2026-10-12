"""UC-62 AC4: sim-loopback parity with the contract's reference inference loop.

The reference loop below uses ONLY contract functions (``init_state``, ``make_obs``, ``raycast``,
``plant_step``, ``load_policy(...).act``) plus the course-sense port, does its own concatenation
and follows the env order (plant_step -> pass check -> advance with wrap/lap -> obs). The harness
run (``run_harness('sim-loopback', ...)``) is recorded to ``ticks.jsonl`` and compared per step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.course import Course, course_from_mapping
from drone_fly.harness.course_senses import (
    GateProgress,
    curr_gate_block,
    next_lookahead_block,
    obstacle_vision_block,
)
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import ray_directions
from drone_fly.harness.loopback import START_MIN_Z, TRACTABLE_KW
from drone_fly.harness.session import run_harness

#: Gates 0 and 1 straddle the spawn (passed on ticks 0 and 1), gate 2 is far down range, so the
#: fixture model (which need not fly well) produces >= 2 passes; obstacles sit in the ray cone.
PASS_COURSE = {
    "gates": [
        {"center": [0.0, 0.0, 0.55], "yaw": 0.0, "aperture": 1.0},
        {"center": [0.05, 0.0, 0.6], "yaw": 0.3, "aperture": 1.0},
        {"center": [6.0, 0.0, 1.5], "yaw": 0.0, "aperture": 1.0},
    ],
    "obstacles": [
        {"type": "cylinder", "center": [1.5, 0.0, 0.5], "radius": 0.3, "half_height": 1.5},
        {"type": "box", "center": [3.0, 1.0, 1.0], "half_extents": [0.5, 0.5, 0.5]},
    ],
    "start": [0.0, 0.0, 0.5],
    "laps": 1,
}


def _reference_loop(contract, checkpoint: Path, course: Course, n_ticks: int):
    c = contract
    policy = c.load_policy(str(checkpoint), extra_dim=70, device="cpu")
    start = np.asarray(course.start, np.float32).copy()
    start[2] = max(float(start[2]), START_MIN_Z)
    s = c.init_state(pos=tuple(float(x) for x in start))
    prog = GateProgress(course)
    obstacles = course.raycast_obstacles()
    steps = []
    for _ in range(n_ticks):
        prog.advance_if_passed(s["pos"])
        if prog.finished:
            break
        gate, nxt = prog.current, prog.next
        target = gate.center
        obs = c.make_obs(s, target)
        extra = np.concatenate(
            [
                obs,
                curr_gate_block(gate),
                next_lookahead_block(s["pos"], nxt, course.obstacles),
                obstacle_vision_block(s["pos"], course.obstacles),
                c.raycast(s, obstacles, target),
            ]
        ).astype(np.float32)
        a = policy.act(obs, extra)
        steps.append({"obs": obs, "extra": extra, "action": a, "tgt": prog.tgt, "lap": prog.lap})
        s = c.plant_step(s, a)
    return steps


def _harness_ticks(run_dir: Path) -> list[dict]:
    lines = (run_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _compare(steps, ticks):
    assert len(ticks) == len(steps)
    max_diff = 0.0
    for k, (ref, tick) in enumerate(zip(steps, ticks, strict=True)):
        assert tick["tick"] == k
        obs = np.asarray(tick["obs"], np.float32)
        extra = np.asarray(tick["extra"], np.float32)
        raw = np.asarray(tick["command"]["raw"], np.float32)
        np.testing.assert_allclose(obs, ref["obs"], rtol=0, atol=1e-6, err_msg=f"obs tick {k}")
        np.testing.assert_allclose(
            extra, ref["extra"], rtol=0, atol=1e-6, err_msg=f"extra tick {k}"
        )
        np.testing.assert_allclose(
            raw, ref["action"], rtol=0, atol=1e-6, err_msg=f"action tick {k}"
        )
        assert tick["gate"]["gate_index"] == ref["tgt"], k
        assert tick["gate"]["lap"] == ref["lap"], k
        max_diff = max(
            max_diff,
            float(np.max(np.abs(extra - ref["extra"]))),
            float(np.max(np.abs(raw - ref["action"]))),
        )
    return max_diff


def _run(contract, checkpoint, k1, tmp_path, config, duration):
    session = run_harness(
        "sim-loopback",
        checkpoint,
        70,
        config=config,
        k1_path=str(k1),
        device="cpu",
        duration_s=duration,
        output_root=str(tmp_path),
    )
    assert session.run_dir is not None
    return session


def test_parity_on_course_gen_seed0(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    session = _run(
        harness_contract,
        harness_checkpoint_70,
        harness_k1_dir,
        tmp_path,
        {"course": {"source": "course_gen", "seed": 0}},
        4.0,
    )
    assert session.result.reason == "completed" and session.result.ticks == 200
    course = course_from_mapping(
        harness_contract.import_reference("course_gen").sample_course(0, **TRACTABLE_KW)
    )
    assert len(course.obstacles) > 0
    steps = _reference_loop(harness_contract, harness_checkpoint_70, course, 200)
    ticks = _harness_ticks(session.run_dir)
    assert _compare(steps, ticks) <= 1e-6
    # The raycast block is live (the obstacle model's critical input), not a constant.
    rays = np.array([t["extra"][30:70] for t in ticks])
    assert rays.shape == (200, 40) and np.all((rays >= 0) & (rays <= 1))


def test_parity_with_gate_passes_and_target_sequence(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    session = _run(
        harness_contract,
        harness_checkpoint_70,
        harness_k1_dir,
        tmp_path,
        {"course": {"source": "inline", "inline": PASS_COURSE}},
        2.0,
    )
    assert session.result.ticks == 100
    course = course_from_mapping(PASS_COURSE)
    steps = _reference_loop(harness_contract, harness_checkpoint_70, course, 100)
    ticks = _harness_ticks(session.run_dir)
    _compare(steps, ticks)
    passes = [e for t in ticks for e in t["events"] if e.startswith("gate_passed")]
    assert len(passes) >= 2
    seq = [t["gate"]["gate_index"] for t in ticks]
    assert seq == [s["tgt"] for s in steps]
    assert seq[0] == 1 and seq[1] == 2  # advanced on the ticks the drone was inside gates 0, 1
    # Raycast sees the in-cone pillar at some point.
    assert min(min(t["extra"][30:70]) for t in ticks) < 1.0
    # obstacle_vision sees both obstacles.
    assert all(t["extra"][25] > 0 and t["extra"][29] > 0 for t in ticks)


def test_course_finished_is_a_normal_stop(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    two_at_start = {
        "gates": PASS_COURSE["gates"][:2],
        "start": PASS_COURSE["start"],
        "laps": 1,
    }
    session = _run(
        harness_contract,
        harness_checkpoint_70,
        harness_k1_dir,
        tmp_path,
        {"course": {"source": "inline", "inline": two_at_start}},
        2.0,
    )
    r = session.result
    assert r.reason == "course_finished" and not r.is_failsafe_stop
    assert r.ticks == 2
    meta = json.loads((session.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "course_finished"


def test_dry_run_records_but_never_steps_the_plant(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    session = run_harness(
        "sim-loopback",
        harness_checkpoint_70,
        70,
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.1,
        dry_run=True,
        output_root=str(tmp_path),
    )
    ticks = _harness_ticks(session.run_dir)
    assert len(ticks) == 5
    assert all(t["sink_output"] == {"dry_run": True} for t in ticks)
    # State never changes because nothing actuates.
    assert all(t["state"] == ticks[0]["state"] for t in ticks)


def test_loopback_refuses_rate_other_than_plant_rate(
    harness_contract, harness_checkpoint_70, harness_k1_dir
):
    with pytest.raises(HarnessConfigError, match="plant rate"):
        run_harness(
            "sim-loopback",
            harness_checkpoint_70,
            70,
            config={"runner": {"rate_hz": 100}},
            k1_path=str(harness_k1_dir),
            device="cpu",
            record=False,
        )


# ----------------------------------------------------------------------------- ray basis


def _assert_basis_matches_raycast(contract, pos, vel, target):
    """Each harness ray direction is confirmed by a tiny box probe hit by the contract raycast()."""
    pos = np.asarray(pos, np.float32)
    s = contract.init_state(pos=tuple(pos), vel=tuple(vel))
    dirs = ray_directions(pos, np.asarray(vel, np.float32), target, contract)
    assert dirs.shape == (contract.N_RAYS, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-5)
    d, h = 2.0, 0.01
    for i, ray in enumerate(dirs):
        probe = [
            {"center": pos + d * ray, "half_extents": np.full(3, h, np.float32), "type": "box"}
        ]
        hit = contract.raycast(s, probe, target)
        assert hit[i] < 1.0, f"ray {i} missed its probe"
        assert hit[i] * contract.RAY_RANGE == pytest.approx(d, abs=2 * h)


@pytest.mark.parametrize(
    "vel,target",
    [
        ((1.0, 0.5, 0.2), None),  # velocity heading (> 0.3 m/s)
        ((0.1, 0.0, 0.0), (0.0, 3.0, 1.0)),  # low speed -> target direction
        ((0.0, 0.0, 0.0), None),  # no target -> +x
        ((0.0, 0.0, 2.0), None),  # vertical heading -> the right-vector fallback
    ],
)
def test_ray_directions_equal_contract_raycast_basis(harness_contract, vel, target):
    tgt = None if target is None else np.asarray(target, np.float32)
    _assert_basis_matches_raycast(harness_contract, (0.0, 0.0, 1.0), vel, tgt)


def test_parity_on_tunnel_course(harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path):
    """A course_gen PROC_TUNNELS course (fed inline) keeps per-step parity with the reference."""
    raw = harness_contract.import_reference("course_gen").sample_course(
        1, with_tunnels=True, **TRACTABLE_KW
    )
    raw = json.loads(json.dumps(raw, default=float))  # plain YAML-like values (no numpy scalars)
    session = _run(
        harness_contract,
        harness_checkpoint_70,
        harness_k1_dir,
        tmp_path,
        {"course": {"source": "inline", "inline": raw}},
        2.0,
    )
    course = course_from_mapping(raw)
    assert len(course.obstacles) >= 5
    steps = _reference_loop(harness_contract, harness_checkpoint_70, course, 100)
    assert _compare(steps, _harness_ticks(session.run_dir)) <= 1e-6


def test_inline_course_gen_dict_with_numpy_values_records(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    """A course_gen dict passed straight through (numpy scalars inside) records a valid run."""
    import yaml

    raw = harness_contract.import_reference("course_gen").sample_course(
        2, with_tunnels=True, **TRACTABLE_KW
    )
    raw["start"] = np.asarray(raw["start"], np.float64)  # force numpy values into the config
    raw["gates"][0]["aperture"] = np.float32(raw["gates"][0]["aperture"])
    session = _run(
        harness_contract,
        harness_checkpoint_70,
        harness_k1_dir,
        tmp_path,
        {"course": {"source": "inline", "inline": raw}},
        0.2,
    )
    assert session.result.reason == "completed" and session.result.ticks == 10
    cfg = yaml.safe_load((session.run_dir / "config.yaml").read_text(encoding="utf-8"))
    inline = cfg["course"]["inline"]
    assert isinstance(inline["start"], list) and all(isinstance(v, float) for v in inline["start"])
    assert isinstance(inline["gates"][0]["aperture"], float)
    meta = json.loads((session.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "completed"
    assert len(_harness_ticks(session.run_dir)) == 10


# ----------------------------------------------------------------------------- UC67 contract v2


def test_v1_command_steps_plant_step_bit_exactly(harness_contract):
    from drone_fly.harness.loopback import LoopbackPlant
    from drone_fly.harness.policy import decode_action

    c = harness_contract
    plant = LoopbackPlant(c, np.array([0.0, 0.0, 1.0], np.float32))
    state = dict(plant.state)
    rng = np.random.default_rng(1)
    for _ in range(20):
        raw = rng.uniform(-1.5, 1.5, 4).astype(np.float32)
        plant.step_command(decode_action(raw, c))
        state = c.plant_step(state, raw)
        for key in ("pos", "vel", "quat", "omega"):
            np.testing.assert_array_equal(plant.state[key], state[key])


def test_v2_command_steps_the_sim_plant_like_the_equivalent_v1_action(harness_contract):
    """On meteor75 a v2 action ``1 + a0`` g equals the v1 action ``2 (1 + a0) / 2.5 - 1``."""
    from drone_fly.harness.loopback import LoopbackPlant
    from drone_fly.harness.policy import decode_action
    from drone_fly.sim.action import ActionSpec

    c = harness_contract
    spec = ActionSpec.v2(1.0, 4.0, max_g=2.5)
    plant = LoopbackPlant(c, np.array([0.0, 0.0, 1.0], np.float32))
    state = dict(plant.state)
    rng = np.random.default_rng(2)
    for _ in range(50):
        raw2 = rng.uniform(-0.9, 0.9, 4).astype(np.float32)
        cmd = decode_action(raw2, c, spec)
        assert cmd.contract_version == 2
        raw1 = raw2.copy()
        raw1[0] = 2.0 * cmd.collective_g / 2.5 - 1.0
        plant.step_command(cmd)
        state = c.plant_step(state, raw1)
        for key in ("pos", "vel", "quat", "omega"):
            np.testing.assert_allclose(plant.state[key], state[key], atol=1e-4, err_msg=key)
    assert plant.steps == 50


def test_v2_checkpoint_runs_in_loopback_and_records_its_contract(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    import shutil

    from drone_fly.sim.action import ActionSpec, write_sidecar

    ckpt = tmp_path / "ckpt" / "model_readout.pt"
    ckpt.parent.mkdir()
    shutil.copyfile(harness_checkpoint_70, ckpt)
    write_sidecar(ckpt, ActionSpec.v2(1.0, 4.0, max_g=2.5), extra_dim=70,
                  actor="k1-reservoir-aug", drone="meteor75", run_id="t")  # fmt: skip
    session = _run(
        harness_contract,
        ckpt,
        harness_k1_dir,
        tmp_path / "runs",
        {"course": {"source": "course_gen", "seed": 0}},
        0.4,
    )
    ticks = _harness_ticks(session.run_dir)
    assert len(ticks) == 20
    assert all(t["command"]["contract_version"] == 2 for t in ticks)
    assert all(t["command"]["collective_g"] is not None for t in ticks)
    meta = json.loads((session.run_dir / "meta.json").read_text(encoding="utf-8"))

    def find(d, key):
        if isinstance(d, dict):
            if key in d:
                return d[key]
            for v in d.values():
                r = find(v, key)
                if r is not None:
                    return r
        return None

    assert find(meta, "contract_version") == 2
    assert find(meta, "action_contract")["collective_scale_g"] == 1.0
    # the plant was stepped (moved off the spawn)
    assert ticks[-1]["obs"] != ticks[0]["obs"]
