"""UC67 AC1 / AC6: ``drone_fly.training`` and the ``drone-fly train`` CLI.

* ``drone-fly train --help`` lists ``run`` / ``eval`` / ``bench``; ``sim replay-sysid`` exists.
* A tiny CPU run on the 322-neuron fixture connectome writes the full run folder
  (``<root>/<drone-id>/<run-id>/``: config snapshot with ``contract_version``, profile byte copy +
  hash, checkpoints with the contract sidecar, train.log / metrics.jsonl / episodes.csv), and the
  checkpoint loads through the harness (contract v2 from the sidecar).
* The promoted ``K1ReservoirAug`` loads a prototype checkpoint and acts identically; the v1 -> v2
  warm start keeps the commanded collective.
* Config loading (``extends``, validation) and the packaged configs; ``eval`` and ``bench`` run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
import yaml

from drone_fly import drones
from drone_fly.cli.main import build_parser, main
from drone_fly.controller.reservoir import K1ReservoirAug, load_readout, warm_start
from drone_fly.sim.action import ActionSpec, decode_numpy
from drone_fly.sim.course import EPISODE_COLUMNS
from drone_fly.training.config import (
    TrainConfigError,
    load_train_config,
    packaged_names,
)
from drone_fly.training.run_dir import CHECKPOINT_NAME

TINY = {
    "n_envs": 8,
    "budget": 64,
    "device": "cpu",
    "course": {"pool_size": 32, "episode_s": 0.1},
    "ppo": {"roll": 8, "epochs": 1, "minibatches": 2},
    "warm_start": {"checkpoint": None},
    "log_every_iters": 1,
}


@pytest.fixture(scope="module")
def fixture_k1(harness_contract, harness_k1_dir):
    """The fixture connectome the session's contract was loaded with (prototype == promoted)."""
    from drone_fly.connectome import load_connectome

    return load_connectome(harness_k1_dir)


# ----------------------------------------------------------------------------- CLI surface


def test_train_help_lists_the_subcommands(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["train", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for sub in ("run", "eval", "bench"):
        assert sub in out


def test_top_level_parser_has_train_and_sim_groups():
    parser = build_parser()
    args = parser.parse_args(["train", "eval", "ck.pt", "--config", "meteor75-legacy"])
    assert args.train_command == "eval" and args.episodes == 192 and args.seed == 1234
    args = parser.parse_args(["sim", "replay-sysid", "s", "--drone", "x"])
    assert args.sim_command == "replay-sysid" and args.gate is None
    args = parser.parse_args(["train", "bench"])
    assert args.n_envs == 8192 and args.config == "proc-course-dr"


# ----------------------------------------------------------------------------- configs


def test_packaged_configs():
    assert {"meteor75-legacy", "proc-course-dr"} <= set(packaged_names())
    legacy = load_train_config("meteor75-legacy")
    assert legacy.drone == "meteor75" and legacy.randomization_scale == 0.0
    assert legacy.timing.mode == "fixed" and legacy.timing.fixed_hz == 50.0
    assert legacy.plant.integrator == "euler-legacy" and legacy.plant.substeps == 1
    assert legacy.course.start.mode == "legacy" and legacy.course.no_obs
    assert legacy.course.tier == "tractable" and legacy.action == ActionSpec()
    assert legacy.course.reward_reference_dt_s == 0.02
    dr = load_train_config("proc-course-dr")
    assert dr.randomization_scale == 1.0 and dr.timing.mode == "random"
    assert dr.course.start.mode == "mixed" and dr.course.start.ground_fraction == 0.5
    assert dr.course.start.arm_delay_s == 0.5
    assert dr.action == ActionSpec.v2(1.0, 4.0)
    assert dr.warm_start.legacy_v1 and dr.course.reward_reference_dt_s == 0.02


def test_extends_merges_and_overrides(tmp_path):
    child = tmp_path / "uc68-micro.yaml"
    child.write_text(
        "extends: proc-course-dr\ndrone: liftoff-gremlin-micro\nbudget: 1.0e6\n"
        "course:\n  start:\n    ground_fraction: 0.8\n",
        encoding="utf-8",
    )
    cfg = load_train_config(child, {"n_envs": 16})
    assert cfg.name == "uc68-micro" and cfg.drone == "liftoff-gremlin-micro"
    assert cfg.budget == 1.0e6 and cfg.n_envs == 16
    assert cfg.course.start.ground_fraction == 0.8
    assert cfg.course.start.arm_delay_s == 0.5  # inherited
    assert cfg.action.version == 2  # inherited
    snap = cfg.snapshot()
    assert snap["contract_version"] == 2 and snap["action"]["collective_scale_g"] == 1.0


@pytest.mark.parametrize(
    "body,match",
    [
        ("extends: proc-course-dr\nbogus: 1\n", "unknown training config keys"),
        ("extends: proc-course-dr\nactor: {extra_dim: 70}\n", "extra_dim"),
        ("extends: proc-course-dr\naction: {version: 1}\n", "legacy_v1"),
        ("extends: proc-course-dr\nppo: {lrate: 1}\n", "unknown ppo keys"),
        ("extends: proc-course-dr\ntiming: {hz: 3}\n", "unknown timing keys"),
        ("drone: meteor75\naction: {version: 1}\n", "needs 'course'"),
    ],
)
def test_invalid_configs_are_refused(tmp_path, body, match):
    p = tmp_path / "bad.yaml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(TrainConfigError, match=match):
        load_train_config(p)


def test_unknown_config_name_lists_the_packaged_ones():
    with pytest.raises(TrainConfigError, match="proc-course-dr"):
        load_train_config("no-such-config")


# ----------------------------------------------------------------------------- run folder


@pytest.fixture(scope="module")
def tiny_run(fixture_k1, tmp_path_factory):
    from drone_fly.training.ppo import train

    cfg = load_train_config("proc-course-dr", TINY)
    root = tmp_path_factory.mktemp("training")
    res = train(cfg, drones.load("meteor75"), run_root=root, connectome=fixture_k1,
                device="cpu", max_iterations=2)  # fmt: skip
    return cfg, root, res


def test_tiny_run_writes_the_run_folder(tiny_run):
    cfg, root, res = tiny_run
    run = res.run.path
    assert run.parent == root / "meteor75"
    assert run.name.endswith("-proc-course-dr") and run.name == res.run.run_id
    assert res.iterations == 1 and res.steps == 64  # the budget ends it after one rollout
    snap = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert snap["contract_version"] == 2 and snap["run_id"] == res.run.run_id
    assert snap["drone"] == "meteor75" and snap["n_envs"] == 8
    src = drones.store.profile_path("meteor75")
    assert (run / "drone.yaml").read_bytes() == src.read_bytes()
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    assert (run / "drone.sha256").read_text(encoding="utf-8").split()[0] == digest
    ckpt = run / "checkpoints" / CHECKPOINT_NAME
    assert res.checkpoint == ckpt and ckpt.is_file()
    side = json.loads((run / "checkpoints" / "model_readout.contract.json").read_text("utf-8"))
    assert side["contract_version"] == 2 and side["extra_dim"] == 30
    assert side["action"] == {"collective_scale_g": 1.0, "max_rate_rad_s": 4.0, "max_g": 2.5}
    assert side["drone"] == "meteor75" and side["run_id"] == res.run.run_id
    assert side["actor"] == "k1-reservoir-aug"
    logs = run / "logs"
    assert "run " in (logs / "train.log").read_text(encoding="utf-8")
    metrics = [json.loads(x) for x in (logs / "metrics.jsonl").read_text("utf-8").splitlines()]
    assert metrics and metrics[-1]["steps"] == 64
    with open(logs / "episodes.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == EPISODE_COLUMNS and len(rows) > 1  # 0.1 s episodes finished


def test_trained_checkpoint_is_stripped_and_loads_in_the_harness(tiny_run, harness_contract):
    from drone_fly.harness.policy import ContractPolicyRunner

    _, _, res = tiny_run
    sd = torch.load(res.checkpoint, map_location="cpu")
    assert sd and not any(k.startswith("body.layer.") for k in sd)
    runner = ContractPolicyRunner(harness_contract, res.checkpoint, 30)
    runner.load()
    assert runner.contract_version == 2
    a = runner.act(np.zeros(12, np.float32), np.zeros(30, np.float32))
    assert a.shape == (4,) and np.all(np.isfinite(a))


def test_v2_actor_starts_at_hover(fixture_k1):
    actor = K1ReservoirAug(30, connectome=fixture_k1, collective_bias=0.0)
    assert actor.pi[-1].bias[0].item() == 0.0
    legacy = K1ReservoirAug(30, connectome=fixture_k1, collective_bias=-0.2)
    assert legacy.pi[-1].bias[0].item() == pytest.approx(-0.2)


# ----------------------------------------------------------------------------- reservoir actor


def _obs_extra(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 12, generator=g), torch.randn(n, 30, generator=g)


def test_promoted_actor_loads_a_prototype_checkpoint_and_acts_identically(
    harness_contract, harness_checkpoint_30, fixture_k1
):
    ours = load_readout(harness_checkpoint_30, 30, connectome=fixture_k1)
    ref = sys.modules["inference"].load_policy(str(harness_checkpoint_30), extra_dim=30)
    obs, extra = _obs_extra()
    with torch.no_grad():
        mean, _ = ours.act(ours.features(obs), extra)
    for i in range(obs.shape[0]):
        expect = ref.act(obs[i].numpy(), extra[i].numpy())
        np.testing.assert_allclose(mean[i].numpy(), expect, atol=1e-5, rtol=0)


def test_v1_warm_start_keeps_the_commanded_collective(harness_checkpoint_30, fixture_k1):
    src = load_readout(harness_checkpoint_30, 30, connectome=fixture_k1)
    dst = K1ReservoirAug(30, connectome=fixture_k1)
    warm_start(dst, harness_checkpoint_30, 30, legacy_v1_to_scale_g=1.0)
    obs, extra = _obs_extra(64, seed=1)
    with torch.no_grad():
        m1, _ = src.act(src.features(obs), extra)
        m2, _ = dst.act(dst.features(obs), extra)
    torch.testing.assert_close(m1[:, 1:], m2[:, 1:])  # rate outputs unchanged
    spec = ActionSpec.v2(1.0, 4.0, max_g=2.5)
    compared = 0
    for a1, a2 in zip(m1.numpy(), m2.numpy(), strict=True):
        if -1.0 < a1[0] < 1.0:  # exact inside the v1 clip range
            g1 = decode_numpy(a1, ActionSpec()).collective_g
            g2 = decode_numpy(a2, spec).collective_g
            assert g2 == pytest.approx(g1, abs=1e-5)
            compared += 1
    assert compared > 10


def test_warm_start_into_more_extra_senses_zeroes_the_new_columns(
    harness_checkpoint_30, fixture_k1
):
    dst = K1ReservoirAug(70, connectome=fixture_k1)
    warm_start(dst, harness_checkpoint_30, 30)
    w = dst.pi[0].weight
    r = dst.feat_dim
    assert torch.all(w[:, r + 30 :] == 0)
    src = torch.load(harness_checkpoint_30, map_location="cpu")
    torch.testing.assert_close(w[:, : r + 30], src["pi.0.weight"])
    assert torch.all(dst.vf[0].weight[:, 30:] == 0)


# ----------------------------------------------------------------------------- eval / bench


def _short_legacy(tmp_path):
    p = tmp_path / "legacy-short.yaml"
    p.write_text(
        "extends: meteor75-legacy\ncourse:\n  pool_size: 32\n  episode_s: 0.5\n", encoding="utf-8"
    )
    return p


def test_eval_runs_paired_episodes(tmp_path, harness_checkpoint_30, fixture_k1):
    from drone_fly.training.cli import evaluate

    res = evaluate(str(harness_checkpoint_30), str(_short_legacy(tmp_path)), episodes=8,
                   seed=3, device="cpu", connectome=fixture_k1)  # fmt: skip
    assert res["contract_version"] == 1 and res["drone"] == "meteor75"
    new = res["new"]
    assert new["episodes"] == 8
    assert new["unfinished"] == 0 and new["completed"] + new["crashed"] <= 8  # rest timed out
    assert 0 < new["mean_steps"] <= 0.5 / 0.02 + 1
    assert 0.0 <= new["completion"] <= 1.0 and "prototype" not in res


def test_bench_reports_env_steps_per_second(tmp_path):
    from drone_fly.training.cli import bench

    p = tmp_path / "dr-short.yaml"
    p.write_text("extends: proc-course-dr\ncourse:\n  pool_size: 32\n", encoding="utf-8")
    res = bench(config=str(p), n_envs=16, steps=3, warmup=1, device="cpu", substeps=(1, 2),
                prototype=False)  # fmt: skip
    assert res["n_envs"] == 16 and "prototype" not in res
    for k in ("substeps_1", "substeps_2"):
        row = res["new"][k]
        assert row["env_steps_per_s"] > 0
        for part in ("plant_s", "timing_s", "course_step_s", "extra_s", "full_step_s"):
            assert row[part] > 0


def test_package_layout_and_the_prototype_pointer():
    """AC1: the simulator / trainer homes, the default run root, and gpu_prototype's pointer."""
    from pathlib import Path

    from drone_fly.training.run_dir import default_root

    src = Path(__file__).resolve().parent.parent / "src" / "drone_fly"
    for mod in ("plant", "randomization", "timing", "course", "action", "replay"):
        assert (src / "sim" / f"{mod}.py").is_file(), mod
    for mod in ("ppo", "cli", "config", "run_dir"):
        assert (src / "training" / f"{mod}.py").is_file(), mod
    assert default_root().parts[-2:] == ("artifacts", "training")
    readme = (src.parent.parent / "gpu_prototype" / "README.md").read_text(encoding="utf-8")
    head = "\n".join(readme.splitlines()[:30])
    assert "src/drone_fly/sim" in head and "src/drone_fly/training" in head
