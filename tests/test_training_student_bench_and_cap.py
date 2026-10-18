"""UC70 D9: the student's world scale, the wall-clock cap and ``train student --bench``.

CPU, hermetic (the fixtures of ``test_training_student``): the shipped config's ``k`` agrees with
the ADR 0009 parameter row; ``distill.max_hours`` / ``finetune.max_hours`` load (default None,
override) and stop the DAgger and PPO phases cleanly (saved checkpoint, logged cap, normal return);
``--max-hours`` reaches both sections; ``bench_student`` returns its per-part breakdown on a few
CPU steps; ``--bench`` / ``--bench-steps`` / ``--n-envs`` are wired to it.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from drone_fly import drones
from drone_fly.training import distill
from drone_fly.training.config import load_raw
from tests.student_support import tiny_pool, tiny_student_config, train_inputs

#: The shipped caps (student-liftoff: distill.max_hours, finetune.max_hours; None = no cap).
SHIPPED_CAPS = tuple(
    (load_raw("student-liftoff").get(sec) or {}).get("max_hours") for sec in ("distill", "finetune")
)
ADR = Path(__file__).resolve().parents[1] / "docs" / "adr"
ADR_0009 = ADR / "0009-student-distillation-and-detector-noise.md"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc70-cap"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def train(inputs, tmp_path, cfg=None, **kw):
    cfg = cfg or tiny_student_config(inputs.k07)
    args = dict(teacher=inputs.teacher, noise=inputs.noise, run_root=tmp_path / "out",
                max_iterations=2, k1_path=str(inputs.k1), skip_feasibility=True,
                process_lister=lambda: [])  # fmt: skip
    args.update(kw)
    return distill.train_student(cfg, drones.load(cfg.drone), **args)


def _readout(path: Path, key: str = "k1.pi.0.weight") -> torch.Tensor:
    return torch.load(path, weights_only=True)[key]


# ----------------------------------------------------------------------------- k


def test_shipped_world_scale_matches_the_adr_parameter_row():
    k = float(load_raw("student-liftoff")["student"]["world_scale"])
    row = next(x for x in ADR_0009.read_text("utf-8").splitlines()
               if x.startswith("| Student world scale"))  # fmt: skip
    adr_k = float(re.search(r"\|\s*\*\*([\d.]+)\*\*\s*\|", row).group(1))
    assert k == adr_k == 10.0


# ----------------------------------------------------------------------------- config


def test_max_hours_default_and_override(inputs):
    cfg = tiny_student_config(inputs.k07)
    from drone_fly.training.config import DistillSection, FinetuneSection

    assert DistillSection().max_hours is None and FinetuneSection().max_hours is None
    assert (cfg.distill.max_hours, cfg.finetune.max_hours) == SHIPPED_CAPS
    cfg = tiny_student_config(inputs.k07, distill={"max_hours": 2.5}, finetune={"max_hours": 3})
    assert cfg.distill.max_hours == 2.5 and cfg.finetune.max_hours == 3
    raw = load_raw("student-liftoff", {"distill": {"max_hours": 0.5}})
    assert raw["distill"]["max_hours"] == 0.5 and raw["distill"]["rollout"] == 64  # merged


# ----------------------------------------------------------------------------- DAgger cap


def test_dagger_stops_at_the_wall_clock_cap(inputs, tmp_path, tiny_val, caplog):
    cfg = tiny_student_config(inputs.k07, distill={"max_hours": 0.0})  # capped after 1 iteration
    with caplog.at_level("INFO", logger=distill.log.name):
        res = train(inputs, tmp_path, cfg, max_iterations=50)
    assert res.iterations == 1 and res.steps == 32
    assert "wall-clock cap of 0.00 h reached" in caplog.text
    assert "(done)" in caplog.text  # a normal finish, not an interruption
    run = res.run.path
    assert (run / "checkpoints" / "student.pt").is_file()
    rs = torch.load(run / "checkpoints" / "resume.pt", weights_only=False)
    assert rs["it"] == 1 and rs["steps"] == 32
    prog = json.loads((run / "logs" / "progress.json").read_text("utf-8"))
    assert prog["it"] == 1
    sel = json.loads((run / "eval" / "student_selection.json").read_text("utf-8"))
    assert (run / "checkpoints" / sel["checkpoint"]).is_file()  # the final val selection ran


def test_dagger_without_a_cap_runs_to_max_iterations(inputs, tmp_path, tiny_val):
    cfg = tiny_student_config(inputs.k07, distill={"max_hours": 1000.0})
    assert train(inputs, tmp_path, cfg, max_iterations=3).iterations == 3


# ----------------------------------------------------------------------------- PPO cap


class FakeClock:
    """``distill.time`` with a ``time()`` that advances 1 s per call (the rest is the real one)."""

    def __init__(self) -> None:
        self.t = time.time()

    def time(self) -> float:
        self.t += 1.0
        return self.t

    def __getattr__(self, name: str):
        return getattr(time, name)


def _spy_ppo(monkeypatch):
    """Runs the PPO phase under :class:`FakeClock` and records its result."""
    real = distill.ppo_finetune
    out: dict = {}

    def spy(*a, **k):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(distill, "time", FakeClock())
            out.update(real(*a, **k))
        return out

    monkeypatch.setattr(distill, "ppo_finetune", spy)
    return out


def test_ppo_phase_stops_at_the_wall_clock_cap(inputs, tmp_path, tiny_val, monkeypatch):
    # 1e6 fine-tune steps is ~31k iterations of 32 env-steps; the 2.5 s cap (fake clock, 1 s per
    # call) lets one or two through
    cfg = tiny_student_config(inputs.k07, finetune={"steps": 1.0e6, "critic_warmup_iters": 0,
                                                    "max_hours": 2.5 / 3600.0})  # fmt: skip
    ppo = _spy_ppo(monkeypatch)
    res = train(inputs, tmp_path, cfg, max_iterations=1)
    assert 1 <= ppo["iterations"] <= 2 and ppo["steps"] < 1.0e6 and ppo["capped"] is True
    run = res.run.path
    hist = [json.loads(x) for x in (run / "eval" / "val_history.jsonl").read_text().splitlines()]
    assert [h["it"] for h in hist if h["it"] >= 100000] == [100000 + ppo["iterations"]]
    resume = torch.load(run / "checkpoints" / "resume.pt", weights_only=False)["actor"]
    after = _readout(run / "checkpoints" / "student.pt")
    assert not torch.equal(after, resume["k1.pi.0.weight"])  # PPO ran and its result was saved


def test_ppo_phase_without_a_cap_stops_on_steps(inputs, tmp_path, tiny_val, monkeypatch):
    cfg = tiny_student_config(inputs.k07, finetune={"steps": 64.0, "critic_warmup_iters": 0,
                                                    "max_hours": 1000.0})  # fmt: skip
    ppo = _spy_ppo(monkeypatch)
    train(inputs, tmp_path, cfg, max_iterations=1)
    assert ppo["iterations"] == 2 and ppo["steps"] == 64 and ppo["capped"] is False


def test_ppo_phase_logs_the_cap(inputs, tmp_path, tiny_val, monkeypatch, caplog):
    cfg = tiny_student_config(inputs.k07, finetune={"steps": 1.0e6, "max_hours": 2.5 / 3600.0})
    _spy_ppo(monkeypatch)
    with caplog.at_level("INFO", logger=distill.log.name):
        train(inputs, tmp_path, cfg, max_iterations=1)
    assert caplog.text.count("wall-clock cap") == 1  # the PPO phase's (DAgger is uncapped)


def test_resume_at_the_dagger_budget_goes_straight_to_ppo(inputs, tmp_path, tiny_val, monkeypatch,
                                                          caplog):  # fmt: skip
    cfg = tiny_student_config(inputs.k07, budget=32.0)  # one DAgger iteration of 4 x 8
    res = train(inputs, tmp_path, cfg, max_iterations=5)
    assert res.iterations == 1 and res.steps == 32  # finished on the budget, no PPO (steps 0)
    run = res.run.path
    n_rows = len((run / "logs" / "metrics.jsonl").read_text().splitlines())
    ppo = _spy_ppo(monkeypatch)
    cfg2 = tiny_student_config(inputs.k07, budget=32.0,
                               finetune={"steps": 32.0, "critic_warmup_iters": 0})  # fmt: skip
    with caplog.at_level("INFO", logger=distill.log.name):
        res2 = train(inputs, tmp_path, cfg2, resume=run, max_iterations=5)
    assert res2.run.path == run
    assert res2.iterations == 1 and res2.steps == 32  # no extra DAgger iteration
    assert "DAgger budget already reached" in caplog.text
    assert len((run / "logs" / "metrics.jsonl").read_text().splitlines()) == n_rows
    rs = torch.load(run / "checkpoints" / "resume.pt", weights_only=False)
    assert rs["it"] == 1 and rs["steps"] == 32
    assert ppo["iterations"] == 1 and ppo["steps"] == 32  # PPO ran its full 32 steps
    assert not torch.equal(_readout(run / "checkpoints" / "student.pt"),
                           rs["actor"]["k1.pi.0.weight"])  # fmt: skip


# ----------------------------------------------------------------------------- bench


def test_bench_student_on_cpu(inputs):
    cfg = tiny_student_config(inputs.k07)
    res = distill.bench_student(cfg, drones.load(cfg.drone), teacher=inputs.teacher,
                                noise=inputs.noise, n_envs=4, steps=2, warmup=1, device="cpu",
                                k1_path=str(inputs.k1))  # fmt: skip
    assert {"n_envs", "device", "actor", "gpu", "courses", "pool_build_s", "full_step_s",
            "env_step_s", "student_s", "teacher_s", "env_steps_per_s", "peak_gpu_mb",
            "steps_per_hour"} <= set(res)  # fmt: skip
    assert res["n_envs"] == 4 and res["device"] == "cpu" and res["actor"] == "k1-student"
    assert res["gpu"] is None and not res["peak_gpu_mb"]  # None (or 0) on the CPU
    assert res["courses"] > 0
    for k in ("full_step_s", "env_step_s", "student_s", "teacher_s"):
        assert res[k] > 0.0, k
    assert res["env_steps_per_s"] == pytest.approx(4 / res["full_step_s"])
    assert res["steps_per_hour"] == pytest.approx(3600.0 * res["env_steps_per_s"])
    json.dumps(res)  # the CLI prints it as JSON


def test_bench_student_mlp_baseline_on_cpu(inputs):
    cfg = tiny_student_config(inputs.k07, actor="mlp-student")
    res = distill.bench_student(cfg, drones.load(cfg.drone), teacher=inputs.teacher,
                                noise=inputs.noise, n_envs=2, steps=1, warmup=0,
                                device="cpu")  # fmt: skip
    assert res["actor"] == "mlp-student" and res["n_envs"] == 2 and res["env_steps_per_s"] > 0


# ----------------------------------------------------------------------------- CLI


def _cli_config(inputs, tmp_path: Path) -> Path:
    raw = {"extends": "student-liftoff", "n_envs": 8, "device": "cpu",
           "tracks": {"mix": [{"select": "procedural:proc-liftoff", "weight": 1.0}]},
           "student": {"warm_start": str(inputs.k07)}}  # fmt: skip
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw), "utf-8")
    return p


def _capture(monkeypatch, name: str, ret):
    seen: dict = {}

    def fake(cfg, profile, **kw):
        seen.update(cfg=cfg, profile=profile, **kw)
        return ret

    monkeypatch.setattr(distill, name, fake)
    return seen


def _student_args(inputs, cfg_path: Path, *extra: str) -> list[str]:
    return ["train", "student", "--teacher", str(inputs.teacher), "--noise", str(inputs.noise),
            "--config", str(cfg_path), "--k1-path", str(inputs.k1), *extra]  # fmt: skip


def _fake_result(tmp_path: Path):
    return SimpleNamespace(run=SimpleNamespace(path=tmp_path / "run"),
                           checkpoint=tmp_path / "run" / "student.pt", steps=0, iterations=0,
                           seconds=0.0)  # fmt: skip


def test_cli_max_hours_reaches_both_phases(inputs, tmp_path, monkeypatch, capsys):
    from drone_fly.cli import main as cli

    seen = _capture(monkeypatch, "train_student", _fake_result(tmp_path))
    rc = cli.main(_student_args(inputs, _cli_config(inputs, tmp_path), "--max-hours", "2.5"))
    assert rc == 0, capsys.readouterr().out
    assert seen["cfg"].distill.max_hours == 2.5 and seen["cfg"].finetune.max_hours == 2.5
    rc = cli.main(_student_args(inputs, _cli_config(inputs, tmp_path)))
    assert rc == 0
    assert (seen["cfg"].distill.max_hours, seen["cfg"].finetune.max_hours) == SHIPPED_CAPS


def test_cli_max_hours_applies_on_resume(inputs, tmp_path, monkeypatch, tiny_val):
    from drone_fly.cli import main as cli

    res = train(inputs, tmp_path, max_iterations=1)
    seen = _capture(monkeypatch, "train_student", _fake_result(tmp_path))
    rc = cli.main(["train", "student", "--teacher", str(inputs.teacher), "--resume",
                   str(res.run.path), "--max-hours", "2.5"])  # fmt: skip
    assert rc == 0
    assert seen["cfg"].distill.max_hours == 2.5 and seen["cfg"].finetune.max_hours == 2.5


def test_cli_resume_applies_ppo_steps_and_max_hours(inputs, tmp_path, monkeypatch, tiny_val):
    from drone_fly.cli import main as cli

    res = train(inputs, tmp_path, max_iterations=1)
    snap = yaml.safe_load((res.run.path / "config.yaml").read_text("utf-8"))
    seen = _capture(monkeypatch, "train_student", _fake_result(tmp_path))
    rc = cli.main(["train", "student", "--teacher", str(inputs.teacher), "--resume",
                   str(res.run.path), "--ppo-steps", "4096", "--max-hours", "1.5"])  # fmt: skip
    assert rc == 0 and seen["resume"] == str(res.run.path)
    cfg = seen["cfg"]
    assert cfg.finetune.steps == 4096 and cfg.finetune.max_hours == 1.5
    assert cfg.distill.max_hours == 1.5
    # the rest of both sections still comes from the run's snapshot (merged, not replaced)
    assert cfg.distill.rollout == snap["distill"]["rollout"] == 4
    assert cfg.finetune.roll == snap["finetune"]["roll"] == 4
    assert cfg.finetune.lr == snap["finetune"]["lr"]
    # without the flags the snapshot's values stand
    cli.main(["train", "student", "--teacher", str(inputs.teacher), "--resume", str(res.run.path)])
    assert seen["cfg"].finetune.steps == snap["finetune"]["steps"]
    caps = (snap["distill"].get("max_hours"), snap["finetune"].get("max_hours"))
    assert caps == SHIPPED_CAPS  # the run was built from the shipped config
    assert (seen["cfg"].distill.max_hours, seen["cfg"].finetune.max_hours) == caps


def test_cli_bench_parses_and_skips_training(inputs, tmp_path, monkeypatch, capsys):
    from drone_fly.cli import main as cli

    bench = {"actor": "k1-student", "n_envs": 3, "gpu": None, "device": "cpu",
             "env_steps_per_s": 1500.0, "steps_per_hour": 5.4e6, "full_step_s": 0.002,
             "env_step_s": 0.001, "student_s": 0.0005, "teacher_s": 0.0002,
             "peak_gpu_mb": None}  # fmt: skip
    seen = _capture(monkeypatch, "bench_student", bench)
    monkeypatch.setattr(distill, "train_student", lambda *a, **k: pytest.fail("trained"))
    cfg_path = _cli_config(inputs, tmp_path)
    rc = cli.main(_student_args(inputs, cfg_path, "--bench", "--bench-steps", "7", "--n-envs", "3",
                                "--device", "cpu"))  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0, out
    assert seen["n_envs"] == 3 and seen["steps"] == 7 and seen["device"] == "cpu"
    assert seen["teacher"] == str(inputs.teacher) and seen["noise"] == str(inputs.noise)
    assert seen["k1_path"] == str(inputs.k1) and seen["cfg"].base.n_envs == 3
    assert "k1-student @ 3 envs on cpu: 1,500 env-steps/s" in out and '"n_envs": 3' in out
    # defaults: --bench-steps 50 and the config's n_envs
    rc = cli.main(_student_args(inputs, cfg_path, "--bench"))
    assert rc == 0 and seen["steps"] == 50 and seen["n_envs"] == 8


def test_cli_bench_end_to_end_on_cpu(inputs, tmp_path, capsys):
    from drone_fly.cli import main as cli

    args = ["--bench", "--bench-steps", "1", "--n-envs", "2", "--device", "cpu"]
    rc = cli.main(_student_args(inputs, _cli_config(inputs, tmp_path), *args))
    out = capsys.readouterr().out
    assert rc == 0, out
    js = json.loads(out[out.index("{") :])
    assert js["n_envs"] == 2 and js["device"] == "cpu" and js["peak_gpu_mb"] is None
