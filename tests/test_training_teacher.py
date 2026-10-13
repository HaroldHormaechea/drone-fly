"""UC68 AC5 / AC6 (D1 / D8 / D9): ``drone-fly train teacher`` on CPU over the synthetic library.

* A 2-iteration run writes ``<root>/<drone>/<run>/``: the config snapshot (kind teacher), the
  drone profile + hash, ``checkpoints/teacher.pt`` + ``teacher_it<N>.pt`` with sidecars
  (``obs_schema`` teacher-obs v1, ``world_scale``, ``course_conversion``), ``resume.pt``, the logs
  (train.log, metrics.jsonl with per-group completion, episodes.csv, progress.json) and the
  ``tracks/`` snapshot (catalogues, courses.json, files.sha256).
* The checkpoint loads as an ``MLPTeacher``; the actor starts near hover.
* ``--resume`` continues the iteration count, appends to the logs and keeps ``config.yaml``; a
  ``STOP`` file saves and exits.
* The CLI runs and ``--tracks / --procedural`` rebuild the mix.
* Config: the packaged ``teacher-liftoff.yaml`` reward weights are the documented a-priori values
  (ADR 0007 D8 + the pre-pilot rebalance upright / spin 0.005 / 0.005); validation refuses
  warm starts, unknown keys and legacy use of a teacher config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from drone_fly import drones
from drone_fly.cli.main import main
from drone_fly.controller.mlp_teacher import MLPTeacher
from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM, EPISODE_COLUMNS, RewardConfig
from drone_fly.sim.teacher_obs import DIM, input_scale
from drone_fly.training.config import (
    TrainConfigError,
    from_mapping,
    load_raw,
    load_teacher_config,
    teacher_from_mapping,
)
from tests.tracks_support import build_library, use_library, write_teacher_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def tiny_run(lib, tmp_path_factory):
    from drone_fly.training.ppo import train_teacher

    base = tmp_path_factory.mktemp("teacher")
    with pytest.MonkeyPatch.context() as mp:
        use_library(mp, lib)
        cfg = load_teacher_config(write_teacher_config(lib, base / "teacher.yaml"))
        res = train_teacher(cfg, drones.load(cfg.drone), run_root=base / "runs", run_id="r1",
                            max_iterations=2)  # fmt: skip
    return res, cfg, base


# ----------------------------------------------------------------------------- config


def test_packaged_teacher_config_reward_weights_are_the_documented_a_priori_values():
    raw = load_raw("teacher-liftoff")
    r = raw["env"]["rewards"]
    assert r == {"progress_per_m": 1.0, "gate": 10.0, "finish": 50.0, "crash": 10.0,
                 "upright": 0.005, "spin": 0.005, "action_rate": 0.01}  # fmt: skip
    assert RewardConfig() == RewardConfig.from_dict(r)  # the code defaults agree
    adr = (REPO / "docs" / "adr" / "0007-track-library-and-teacher.md").read_text("utf-8")
    assert "0.005 / 0.005" in adr and "gate +10, finish +50, crash −10" in adr
    assert "1.0 per metre" in adr and "−0.01·‖Δa‖²" in adr
    assert raw["env"]["reward_reference_dt_s"] == 0.02


def test_packaged_teacher_config_matches_the_plan():
    cfg = load_teacher_config("teacher-liftoff", validate_tracks=False)
    assert cfg.drone == "liftoff-gremlin-micro" and cfg.actor.kind == "mlp-teacher"
    assert cfg.obs == "teacher-obs-v1" and cfg.action.version == 2
    assert cfg.n_envs == 8192 and cfg.budget == 2e9 and cfg.randomization_scale == 1.0
    assert cfg.tracks.world_scale == 1.0 and cfg.tracks.scene_policy == "allow-missing"
    assert [g["select"] for g in cfg.tracks.mix] == ["liftoff:train", "procedural:proc-liftoff"]
    assert [g["weight"] for g in cfg.tracks.mix] == [0.6, 0.4]
    assert cfg.tracks.mix[0]["scene_weight"] == 2.0
    s = cfg.env.start
    assert (s.ground_fraction, s.route_fraction, s.arm_delay_s) == (0.5, 0.3, 0.5)
    assert cfg.env.drone_radius_m == 0.12 and cfg.env.floor_kill_m == 0.08
    p = cfg.ppo
    assert (p.roll, p.gamma, p.lam, p.epochs, p.minibatches, p.lr, p.logstd_init) == (
        64, 0.995, 0.95, 4, 16, 3e-4, -0.5)  # fmt: skip
    assert p.logstd_final is None  # learned
    assert (cfg.slack.init_m, cfg.slack.frac) == (0.5, 0.5)
    assert cfg.checkpoint_every_s == 600


def test_teacher_config_validation():
    base = load_raw("teacher-liftoff")
    with pytest.raises(TrainConfigError, match="teacher config"):
        from_mapping(base)  # the legacy trainer refuses it
    with pytest.raises(TrainConfigError, match="from scratch"):
        teacher_from_mapping({**base, "warm_start": {"checkpoint": "x.pt"}}, validate_tracks=False)
    with pytest.raises(TrainConfigError, match="unknown tracks keys"):
        teacher_from_mapping({**base, "tracks": {**base["tracks"], "warp": 1}},
                             validate_tracks=False)  # fmt: skip
    with pytest.raises(TrainConfigError, match="mix entries"):
        teacher_from_mapping({**base, "tracks": {**base["tracks"], "mix": [{"sel": "x"}]}},
                             validate_tracks=False)  # fmt: skip
    with pytest.raises(TrainConfigError, match="obs must be"):
        teacher_from_mapping({**base, "obs": "teacher-obs-v2"}, validate_tracks=False)
    with pytest.raises(TrainConfigError, match="unknown env keys"):
        teacher_from_mapping({**base, "env": {**base["env"], "gravity": 1}},
                             validate_tracks=False)  # fmt: skip


def test_teacher_mix_from_the_cli_flags():
    from drone_fly.training.cli import teacher_mix

    data = load_raw("teacher-liftoff")
    mix = teacher_mix(data, "liftoff:map=LiftoffArena", 0.25)["tracks"]["mix"]
    assert mix == [{"select": "liftoff:map=LiftoffArena", "weight": 0.75, "scene_weight": 2.0},
                   {"select": "procedural:proc-liftoff", "weight": 0.25}]  # fmt: skip
    only = teacher_mix(data, None, 0.0)["tracks"]["mix"]
    assert only == [{"select": "liftoff:train", "weight": 1.0, "scene_weight": 2.0}]


# ----------------------------------------------------------------------------- the actor


def test_mlp_teacher_shapes_hover_start_and_round_trip():
    torch.manual_seed(0)
    m = MLPTeacher(DIM, CRITIC_EXTRA_DIM, input_scale())
    obs = torch.randn(16, DIM)
    mean, std = m.act(m.features(obs), obs)
    assert m.features(obs).shape == (16, 0) and mean.shape == (16, 4) and std.shape == (16, 4)
    assert float(mean.abs().max()) < 0.2  # starts near a0 = 0 (1 g, hover)
    torch.testing.assert_close(std[0], torch.full((4,), float(np.exp(-0.5))))
    v = m.value(torch.randn(16, DIM + CRITIC_EXTRA_DIM))
    assert v.shape == (16,)
    torch.testing.assert_close(m.input_scale, torch.tensor(input_scale()))
    back = MLPTeacher.from_state_dict(m.readout_state_dict())
    torch.testing.assert_close(back.mean_action(obs), m.mean_action(obs))
    with pytest.raises(ValueError):
        MLPTeacher(DIM, 2, [1.0] * 3)


# ----------------------------------------------------------------------------- the run


def test_run_folder_contents(tiny_run):
    res, cfg, _ = tiny_run
    run = res.run.path
    assert run.parent.name == "liftoff-gremlin-micro" and run.name == "r1"
    assert res.iterations == 2 and res.steps == 2 * 4 * 4
    snap = yaml.safe_load((run / "config.yaml").read_text("utf-8"))
    assert snap["kind"] == "teacher" and snap["contract_version"] == 2 and snap["run_id"] == "r1"
    assert snap["tracks"]["mix"][0]["select"] == "liftoff:train"
    assert (run / "drone.yaml").is_file() and (run / "drone.sha256").is_file()
    ck = run / "checkpoints"
    for name in ("teacher.pt", "teacher.contract.json", "teacher_it1.pt", "teacher_it1.contract.json",
                 "resume.pt"):  # fmt: skip
        assert (ck / name).is_file(), name
    logs = run / "logs"
    for name in ("train.log", "metrics.jsonl", "episodes.csv", "progress.json"):
        assert (logs / name).is_file(), name
    tr = run / "tracks"
    for name in ("catalogue-liftoff.yaml", "catalogue-procedural.yaml", "courses.json",
                 "files.sha256"):  # fmt: skip
        assert (tr / name).is_file(), name
    rows = [json.loads(x) for x in (logs / "metrics.jsonl").read_text("utf-8").splitlines()]
    assert [r["it"] for r in rows] == [1, 2]
    assert {"steps", "fps", "slack_m", "reward_per_step", "log_std"} <= set(rows[-1])
    prog = json.loads((logs / "progress.json").read_text("utf-8"))
    assert prog["it"] == 2 and prog["steps"] == res.steps and "eta_s" in prog
    header = (logs / "episodes.csv").read_text("utf-8").splitlines()[0].split(",")
    assert header == list(EPISODE_COLUMNS)


def test_sidecar_keys(tiny_run):
    from drone_fly.tracks.liftoff import CONVERSION_VERSION, ConversionConfig

    res, cfg, _ = tiny_run
    side = json.loads((res.run.path / "checkpoints" / "teacher.contract.json").read_text("utf-8"))
    assert side["contract_version"] == 2 and side["actor"] == "mlp-teacher"
    assert side["drone"] == "liftoff-gremlin-micro" and side["extra_dim"] == DIM
    assert side["obs_schema"]["name"] == "teacher-obs" and side["obs_schema"]["dim"] == 148
    assert side["world_scale"] == 1.0
    cc = side["course_conversion"]
    assert cc["version"] == CONVERSION_VERSION
    assert cc["parameters"]["time_max_s"] == 0.1 and cc["parameters"]["harness_cap"] == 64
    assert ConversionConfig.from_dict(cc["parameters"]).fingerprint() == cc["fingerprint"]
    assert set(cc["scenes"]) == {"FakeArena", "FakeDesert"} and cc["props"]["build_id"] == "1000"
    assert set(cc["catalogues"]) == {"liftoff", "procedural"}
    assert set(cc["rules"]) >= {"openings", "solids", "support", "harness_list", "floor"}


def test_checkpoint_loads_as_an_mlp_teacher(tiny_run):
    res, _, _ = tiny_run
    state = torch.load(res.checkpoint, map_location="cpu", weights_only=True)
    m = MLPTeacher.from_state_dict(state)
    assert m.mean_action(torch.zeros(3, DIM)).shape == (3, 4)


def test_tracks_snapshot(tiny_run):
    res, _, _ = tiny_run
    tr = res.run.path / "tracks"
    courses = json.loads((tr / "courses.json").read_text("utf-8"))
    assert {c["group"] for c in courses} == {"liftoff:train", "procedural:proc-test"}
    assert {c["id"] for c in courses if c["source"] == "liftoff-xml"} == {
        "fake-arena-01", "fake-arena-02", "fake-desert-01", "fake-desert-02-v2"}  # fmt: skip
    lines = (tr / "files.sha256").read_text("utf-8").splitlines()
    kinds = [dict(kv.split("=", 1) for kv in ln.split("  ")[2].split()) ["kind"] for ln in lines]
    assert kinds.count("scene") == 2 and kinds.count("props") == 1 and kinds.count("track") == 4
    assert any("resolved=" in ln for ln in lines if "kind=scene" in ln)


def test_metrics_report_per_group_completion(tiny_run, lib, tmp_path, monkeypatch):
    from drone_fly.training.ppo import group_metrics

    res, _, _ = tiny_run
    rows = np.loadtxt(res.run.path / "logs" / "episodes.csv", delimiter=",", skiprows=1, ndmin=2)
    from drone_fly.sim.pool import PoolGroup

    groups = [PoolGroup("liftoff:train", [0, 1, 2, 3], 0.6), PoolGroup("proc", [4], 0.4)]
    fake = np.zeros((4, len(EPISODE_COLUMNS)))
    col = EPISODE_COLUMNS.index
    fake[:, col("group")] = [0, 0, 1, 1]
    fake[:, col("completed")] = [1, 0, 0, 0]
    fake[:, col("start_mode")] = [1, 2, 3, 1]
    fake[:, col("crossings")] = [5, 1, 2, 0]
    fake[:, col("total_crossings")] = [5, 5, 4, 4]
    fake[:, col("outcome")] = [5, 1, 2, 4]
    m = group_metrics(fake, groups)
    assert m["groups"]["liftoff:train"] == {"episodes": 2, "completion": 0.5, "route_fraction": 0.6}
    assert m["groups"]["proc"]["completion"] == 0.0
    assert m["outcomes"] == {"floor": 0.25, "collision": 0.25, "timeout": 0.25, "completed": 0.25}
    assert m["start_modes"] == {"ground": 0.5, "air": 0.0, "route": 0.0}
    assert group_metrics(np.zeros((0, len(EPISODE_COLUMNS))), groups) == {}
    assert rows.shape[1] == len(EPISODE_COLUMNS)


def test_resume_continues_and_keeps_the_config(tiny_run, lib, monkeypatch):
    from drone_fly.training.ppo import train_teacher

    res, cfg, base = tiny_run
    use_library(monkeypatch, lib)
    run = res.run.path
    before = (run / "config.yaml").read_bytes()
    res2 = train_teacher(cfg, drones.load(cfg.drone), resume=run, max_iterations=3)
    assert res2.run.path == run and res2.iterations == 3 and res2.steps == 3 * 16
    assert (run / "config.yaml").read_bytes() == before
    its = [json.loads(x)["it"] for x in (run / "logs" / "metrics.jsonl").read_text().splitlines()]
    assert its == [1, 2, 3]
    st = torch.load(run / "checkpoints" / "resume.pt", map_location="cpu", weights_only=False)
    assert st["it"] == 3 and st["steps"] == 48
    assert {"actor", "optimizer", "torch_rng", "env_rng", "slack"} <= set(st)
    assert "resumed r1 at it 2" in (run / "logs" / "train.log").read_text("utf-8")


def test_stop_file_saves_and_exits(lib, tmp_path, monkeypatch):
    from drone_fly.training.ppo import train_teacher

    use_library(monkeypatch, lib)
    cfg = load_teacher_config(write_teacher_config(lib, tmp_path / "t.yaml"))
    run = tmp_path / "runs" / "liftoff-gremlin-micro" / "stopme"
    run.mkdir(parents=True)
    (run / "STOP").write_text("", "utf-8")
    res = train_teacher(cfg, drones.load(cfg.drone), run_root=tmp_path / "runs", run_id="stopme",
                        max_iterations=50)  # fmt: skip
    assert res.iterations == 1
    assert (run / "checkpoints" / "teacher.pt").is_file()
    assert (run / "checkpoints" / "resume.pt").is_file()
    assert "STOP file found" in (run / "logs" / "train.log").read_text("utf-8")
    # a resume clears the STOP file and continues
    res2 = train_teacher(cfg, drones.load(cfg.drone), resume=run, max_iterations=2)
    assert not (run / "STOP").exists() and res2.iterations == 2


def test_cli_train_teacher(lib, tmp_path, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    p = write_teacher_config(lib, tmp_path / "t.yaml")
    rc = main(["train", "teacher", "--config", str(p), "--max-iterations", "1", "--output-root",
               str(tmp_path / "runs"), "--device", "cpu", "--n-envs", "2", "--procedural", "0.5"])  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    assert "run: " in out and "steps 8 in 1 iterations" in out
    run = Path(out.split("run: ")[1].splitlines()[0])
    snap = yaml.safe_load((run / "config.yaml").read_text("utf-8"))
    assert snap["n_envs"] == 2
    assert [g["weight"] for g in snap["tracks"]["mix"]] == [0.5, 0.5]
