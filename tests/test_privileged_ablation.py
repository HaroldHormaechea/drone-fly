"""UC-79: the opt-in, sim-only, DIAGNOSTIC privileged teacher blocks appended to the student obs.

The ablation adds one teacher block at a time to ``student-obs`` so the comparison table can show
which missing inputs close the UC-77 student's 0 % gap. These tests pin the verifiable acceptance
criteria (AC1-AC5, AC11) without a GPU training run:

* **obs dim per block** (AC1/AC11): ``student_obs.schema()`` dim and the built obs width both grow
  by exactly the privileged block widths (``65 + Σ widths``), for each block and a combination.
* **default unchanged** (AC3): with the default (empty) selection the schema is byte-identical
  (no ``privileged`` key, dim 65, scale length 65) and the config snapshot drops the field.
* **teacher-equality per block** (AC2): every privileged block equals the teacher's own value for
  the same state — the five slice blocks against their ``build_teacher_obs`` slices and, crucially,
  ``floor`` against the REAL floor row of ``build_teacher_obs`` on an obstacle-free course (so the
  equality is non-circular: ``floor`` is produced by the shared ``surface_feature`` encoder, not a
  slice). Driven through the config via ``build_student_env`` — the single choke point train and
  eval both use — so the obs matches train/eval behaviour.
* **export / harness refusal** (AC4): a privileged sidecar/meta makes ``train export`` and
  ``harness run`` refuse the checkpoint with "diagnostic privileged student, not deployable", even
  with ``--baseline-export`` / ``--allow-baseline``.
* **shipped configs** (AC6/AC7): the 7 diagnostic configs load, carry the MLP actor, the equal
  step budget, no ``max_hours`` cap, and the right privileged list / obs dim; the baseline drops
  the field from its snapshot (AC3).
* **smoke train** (AC11): one DAgger iteration of an ``[altitude]`` MLP student on the CPU records
  the block list and the larger obs dim in a non-deployable, diagnostic sidecar.

CPU / hermetic: hand-built obstacle-free courses, a synthetic noise model, the fixture connectome.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch

from drone_fly import drones
from drone_fly.sim import student_obs as so
from drone_fly.sim import teacher_obs as to
from drone_fly.training import config as tconfig
from drone_fly.training import distill
from tests.oracle_support import loop_passages, make_course
from tests.student_support import (
    LIFTOFF_CAMERA,
    load_model,
    tiny_pool,
    tiny_student_config,
    train_inputs,
)

#: (block -> appended width) at teacher-obs v2; ``teacher_full`` == the full teacher dim.
V2_WIDTHS = {
    "altitude": 1,
    "floor": 8,
    "gate_next_next": 13,
    "gates_full": 39,
    "surfaces": 64,
    "teacher_full": to.dim(2),
}
SLICE_BLOCKS = ("altitude", "gate_next_next", "gates_full", "surfaces", "teacher_full")

SCHEMA_KW = dict(world_scale=6.0, lost_timeout_s=1.3, acquire_timeout_s=3.0, camera=LIFTOFF_CAMERA)


# ----------------------------------------------------------------------------- env helpers


def _open_pool(n_courses: int = 1):
    """A pool of hand-built obstacle-free loop courses (``box=False`` → no scene prims, so the
    floor is the only surface — the non-circular ``floor`` equality rests on this)."""
    from drone_fly.sim.pool import PoolGroup
    from drone_fly.tracks.loader import LoadRecord, Pool

    courses = [
        make_course(loop_passages(), laps=1, circuit=True, box=False, course_id=f"open{i}")
        for i in range(n_courses)
    ]
    for c in courses:
        c.meta.update(map="SyntheticMap", source="synthetic")
    return Pool(courses, [PoolGroup("g", list(range(n_courses)), 1.0)], "eval", {}, LoadRecord())


def _priv_senv(tmp_path, blocks, *, n: int = 4):
    """A :class:`StudentCourseEnv` built through ``build_student_env`` with ``student_privileged``
    sourced from the config (the single choke point), MLP actor, on the CPU."""
    model = load_model(tmp_path)
    cfg = tiny_student_config(
        tmp_path / "warm.pt", actor="mlp-student", distill={"student_privileged": list(blocks)}
    )
    senv = distill.build_student_env(
        cfg, drones.load(cfg.drone), "cpu", _open_pool(), model, n_envs=n, seed=1, mode="exact"
    )
    return senv


def _set_pose(env, pos, *, vel=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0),
              omega=(0.0, 0.0, 0.0), k: int = 0):  # fmt: skip
    """Drive every env to a fixed state (course frame) and route index ``k``."""
    n = env.n
    env.reset_all(torch.zeros(n, dtype=torch.long))
    mask = torch.ones(n, dtype=torch.bool)
    q = torch.as_tensor(quat, dtype=env.dtype)
    q = q / q.norm()
    env.plant.reset(
        mask,
        pos=torch.as_tensor(pos, dtype=env.dtype).expand(n, 3).clone(),
        vel=torch.as_tensor(vel, dtype=env.dtype).expand(n, 3).clone(),
        quat=q.expand(n, 4).clone(),
        omega=torch.as_tensor(omega, dtype=env.dtype).expand(n, 3).clone(),
        motor_g=1.0,
        grounded=False,
        resample=False,
    )
    env.k = torch.full((n,), int(k), dtype=torch.long)


def _priv_tail(senv) -> torch.Tensor:
    """Rebuild the obs at the current env state and return only the appended privileged columns."""
    obs = senv._build_obs(torch.zeros(senv.n, dtype=torch.float64))
    return obs[:, so.DIM :]


# ----------------------------------------------------------------------------- AC1/AC11 obs dim


@pytest.mark.parametrize("block", list(V2_WIDTHS))
def test_schema_dim_grows_by_the_block_width(block):
    w = V2_WIDTHS[block]
    sch = so.schema(**SCHEMA_KW, privileged=(block,))
    assert sch["dim"] == so.DIM + w
    assert len(sch["input_scale"]) == so.DIM + w
    assert sch["privileged"]["blocks"] == [block]
    assert sch["privileged"]["widths"] == {block: w}
    assert sch["privileged"]["teacher_obs_version"] == to.VERSION
    # the privileged block layout is appended after the base blocks, offsets starting at DIM
    tail = [b for b in sch["blocks"] if b["name"].startswith("privileged.")]
    assert [b["name"] for b in tail] == [f"privileged.{block}"]
    assert tail[0]["start"] == so.DIM and tail[0]["width"] == w


def test_schema_dim_for_a_block_combination():
    sch = so.schema(**SCHEMA_KW, privileged=("floor", "altitude"))  # given out of canonical order
    assert sch["dim"] == so.DIM + 1 + 8
    # canonical declaration order: altitude before floor
    assert sch["privileged"]["blocks"] == ["altitude", "floor"]
    names = [b["name"] for b in sch["blocks"] if b["name"].startswith("privileged.")]
    assert names == ["privileged.altitude", "privileged.floor"]


@pytest.mark.parametrize("block", list(V2_WIDTHS))
def test_built_obs_width_grows_by_the_block_width(tmp_path, block):
    senv = _priv_senv(tmp_path, (block,))
    expected = so.DIM + to.privileged_width(block, senv.env.obs_version)
    assert senv.student_obs().shape[1] == expected
    assert senv.privileged == (block,)


# --------------------------------------------------------------------------- AC3 default unchanged


def test_default_schema_is_byte_identical():
    plain = so.schema(**SCHEMA_KW)
    empty = so.schema(**SCHEMA_KW, privileged=())
    assert "privileged" not in plain
    assert plain == empty
    assert plain["dim"] == so.DIM == 65
    assert len(plain["input_scale"]) == so.DIM


def test_default_built_obs_is_the_deployable_width(tmp_path):
    senv = _priv_senv(tmp_path, ())
    assert senv.privileged == ()
    assert senv.student_obs().shape[1] == so.DIM


# ----------------------------------------------------------------------------- AC2 teacher-equality


@pytest.mark.parametrize("block", SLICE_BLOCKS)
def test_slice_block_equals_the_teacher_value(tmp_path, block):
    senv = _priv_senv(tmp_path, (block,))
    env = senv.env
    _set_pose(env, (5.0, 5.0, 3.0), vel=(1.0, 2.0, -0.5), quat=(0.94, 0.17, 0.26, 0.0),
              omega=(0.3, -0.2, 0.1), k=0)  # fmt: skip
    tail = _priv_tail(senv)
    t = to.build_teacher_obs(env, env.obs_version)
    gates = to.block_slice("gates")
    want = {
        "altitude": t[:, to.block_slice("altitude")],
        "gates_full": t[:, gates],
        "surfaces": t[:, to.block_slice("obstacles")],
        "teacher_full": t,
    }
    if block == "gate_next_next":
        off = gates.start + (to.GATES_AHEAD - 1) * to.GATE_FEATURES
        want["gate_next_next"] = t[:, off : off + to.GATE_FEATURES]
    torch.testing.assert_close(tail, want[block], rtol=0, atol=0)


def test_floor_block_is_the_real_teacher_floor_row(tmp_path):
    """Non-circular: on an obstacle-free course the floor is the nearest in-range surface, so it
    sorts to row 0 of the teacher's ``obstacles`` block; the ``floor`` privileged block (built by
    the shared ``surface_feature`` encoder, not a slice) must equal that real row."""
    senv = _priv_senv(tmp_path, ("floor",))
    env = senv.env
    _set_pose(env, (5.0, 5.0, 3.0), k=0)  # 3 m over floor_z = 0, centred in the loop
    tail = _priv_tail(senv)
    t = to.build_teacher_obs(env, env.obs_version)
    obstacles = t[:, to.block_slice("obstacles")].reshape(
        env.n, to.N_OBSTACLES, to.OBSTACLE_FEATURES
    )
    row0 = obstacles[:, 0, :]
    assert torch.allclose(row0[:, 7], torch.ones(env.n))  # row 0 is a valid surface (the floor)
    assert torch.allclose(obstacles[:, 1:, :], torch.zeros_like(obstacles[:, 1:, :]))  # only one
    assert torch.allclose(row0[:, 3], torch.full((env.n,), 3.0), atol=1e-4)  # range to the floor
    torch.testing.assert_close(tail, row0, rtol=0, atol=0)


# ----------------------------------------------------------------------------- AC4 refusals


def _mlp_run_with_privileged(tmp_path, blocks=("altitude",)):
    """An ``mlp-student`` run folder whose sidecar is tagged with a privileged block list."""
    from drone_fly.sim.action import read_sidecar
    from tests.student_support import write_student_run

    run = write_student_run(tmp_path, actor="mlp-student", name="priv-run")
    side_path = run / "checkpoints" / "student.contract.json"
    data = json.loads(side_path.read_text("utf-8"))
    data["privileged"] = list(blocks)
    data["obs_dim"] = so.DIM + to.privileged_total_width(blocks)
    side_path.write_text(json.dumps(data), "utf-8")
    assert read_sidecar(run / "checkpoints" / "student.pt")["privileged"] == list(blocks)
    return run


def test_export_refuses_a_privileged_student(tmp_path):
    from drone_fly.harness.errors import ExportError
    from drone_fly.training.export import export_student

    run = _mlp_run_with_privileged(tmp_path)
    out = tmp_path / "priv.pt"
    with pytest.raises(ExportError, match="diagnostic privileged student, not deployable"):
        export_student(run, out, bench=False)
    # fires even with --baseline-export (unconditional, before the mlp-student gate)
    with pytest.raises(ExportError, match="diagnostic privileged student, not deployable"):
        export_student(run, out, baseline_export=True, bench=False)
    assert not out.exists()


def _baseline_export(tmp_path):
    """A valid (non-privileged) baseline MLP export .pt + its loaded (state, meta)."""
    from drone_fly.student.checkpoint import load_export
    from drone_fly.training.export import export_student
    from tests.student_support import write_student_run

    run = write_student_run(tmp_path, actor="mlp-student", name="base-run")
    out = tmp_path / "base.pt"
    export_student(run, out, baseline_export=True, bench=False)
    state, meta = load_export(out)
    return out, state, meta


@pytest.mark.parametrize("via", ["meta", "obs_schema"])
def test_harness_refuses_a_privileged_student(tmp_path, via):
    from drone_fly.harness.errors import StudentSchemaError
    from drone_fly.harness.student import plan_student
    from drone_fly.student.checkpoint import save_export

    _, state, meta = _baseline_export(tmp_path)
    meta = copy.deepcopy(meta)
    if via == "meta":
        meta["privileged"] = ["altitude"]
    else:
        meta["obs_schema"]["privileged"] = {"blocks": ["floor"]}
    p = save_export(tmp_path / f"priv-{via}.pt", state, meta)
    with pytest.raises(StudentSchemaError, match="diagnostic privileged student, not deployable"):
        plan_student(p)
    # fires even with --allow-baseline (unconditional, before validate_meta)
    with pytest.raises(StudentSchemaError, match="diagnostic privileged student, not deployable"):
        plan_student(p, allow_baseline=True)


# ----------------------------------------------------------------------------- AC6/AC7 configs


_VARIANTS = ["baseline", "altitude", "floor", "gate_next_next", "gates_full", "surfaces",
             "teacher_full"]  # fmt: skip


@pytest.mark.parametrize("variant", _VARIANTS)
def test_shipped_config_loads_and_plumbs_the_block(variant):
    raw = tconfig.load_raw(f"student-liftoff-open-priv-{variant}")
    cfg = tconfig.student_from_mapping(raw, validate_tracks=False)
    assert cfg.student.actor == "mlp-student"
    assert cfg.base.budget == 1.8e7  # the equal step budget (AC6)
    assert cfg.finetune.max_hours is None  # the wall-clock cap is removed (AC6)
    if variant == "baseline":
        assert cfg.distill.student_privileged == ()
    else:
        assert cfg.distill.student_privileged == (variant,)


def test_all_variants_share_one_budget():
    budgets = set()
    for v in _VARIANTS:
        cfg = tconfig.student_from_mapping(
            tconfig.load_raw(f"student-liftoff-open-priv-{v}"), validate_tracks=False
        )
        budgets.add(cfg.base.budget)
    assert budgets == {1.8e7}  # like-for-like (AC6/AC7)


def test_baseline_snapshot_drops_the_privileged_field():
    base = tconfig.student_from_mapping(
        tconfig.load_raw("student-liftoff-open-priv-baseline"), validate_tracks=False
    )
    assert "student_privileged" not in base.snapshot()["distill"]  # byte-identical default (AC3)
    var = tconfig.student_from_mapping(
        tconfig.load_raw("student-liftoff-open-priv-altitude"), validate_tracks=False
    )
    assert var.snapshot()["distill"]["student_privileged"] == ["altitude"]


def test_privileged_rejected_for_a_non_mlp_actor():
    raw = tconfig.load_raw("student-liftoff-open-priv-altitude")
    raw["student"] = dict(raw["student"], actor="k1-student")
    with pytest.raises(tconfig.TrainConfigError, match="mlp-student"):
        tconfig.student_from_mapping(raw, validate_tracks=False)


# ----------------------------------------------------------------------------- AC11 smoke train


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc79-train"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def test_smoke_train_one_iteration_privileged_cpu(inputs, tmp_path, tiny_val):
    from drone_fly.sim.action import read_sidecar

    cfg = tiny_student_config(
        inputs.k07, actor="mlp-student", distill={"student_privileged": ["altitude"]}
    )
    res = distill.train_student(
        cfg, drones.load(cfg.drone), teacher=inputs.teacher, noise=inputs.noise,
        run_root=tmp_path / "out", max_iterations=1, k1_path=str(inputs.k1),
        skip_feasibility=True, process_lister=lambda: [],
    )  # fmt: skip
    assert res.iterations == 1
    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    assert side["actor"] == "mlp-student"
    assert side["privileged"] == ["altitude"]
    assert side["obs_dim"] == so.DIM + 1 == 66
    assert side["deployable"] is False and side["role"] == "diagnostic"
    sch = side["obs_schema"]
    assert sch["dim"] == 66 and sch["privileged"]["blocks"] == ["altitude"]
