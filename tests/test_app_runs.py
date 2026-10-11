"""UC-63 AC3 / AC4: the reservoir RunRegistry — argv/env assembly, state machine, guards (hermetic).

Every OS seam is a fake (see :mod:`tests.app_support`): no micromamba, no subprocess, no GPU, no
git. Spawn argv, CWD and the child's environment are asserted on the fake spawner's records.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app import configs_io, logs, runs, status
from app.configs_io import RecipeError
from app.reservoir_launch import parse_args
from app.runs import RunError

from tests.app_support import (
    MICROMAMBA,
    Counter,
    append_log,
    make_registry,
    save_run,
    touch,
)

PREFIX = [MICROMAMBA, "run", "-n", "dronefly", "python"]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def _abs(root: Path, rel: str) -> str:
    return configs_io.resolve_save_path(str(root), rel)


def _saved(root: Path, name: str, rel: str = "") -> str:
    return f"[aug] saved {_abs(root, rel or f'training/{name}/model.pt')}\n"


# --------------------------------------------------------------------------- argv + cwd


@pytest.mark.parametrize(
    "recipe", ["exp11_count-curriculum", "exp12_gap-steering", "exp13_count-gap", "train_shaping"]
)
def test_named_recipe_argv_and_cwd(root: Path, recipe: str) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", recipe=recipe)
    reg.launch("run-a")
    call = fakes["spawn"].last
    launcher = os.path.join(str(root), "gpu_prototype", configs_io.RECIPES[recipe]["launcher"])
    assert call["argv"] == [*PREFIX, launcher, _abs(root, "training/run-a/model.pt"), "14000000"]
    assert call["cwd"] == os.path.join(str(root), "gpu_prototype")
    assert call["log"] == logs.log_path_for(str(root), "run-a")


def test_named_recipe_forwards_the_edited_budget(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", recipe="exp13_count-gap", knobs={"budget": 2.5e6})
    reg.launch("run-a")
    assert fakes["spawn"].last["argv"][-1] == "2500000"


def test_generic_recipe_argv_forwards_only_set_knobs(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(
        root,
        "gen",
        knobs={
            "N": 256,
            "logstd_init": -0.7,
            "count_curr_frac": 0.25,
            "warm_from": "training/src/model.pt",
            "warm_src_extra_dim": 30,
        },
    )
    reg.launch("gen")
    argv = fakes["spawn"].last["argv"]
    launcher = os.path.join(str(root), "app", "reservoir_launch.py")
    assert argv[: len(PREFIX) + 1] == [*PREFIX, launcher]
    rest = argv[len(PREFIX) + 1 :]
    assert rest == [
        "--save-path",
        _abs(root, "training/gen/model.pt"),
        "--budget",
        "20000000",
        "--N",
        "256",
        "--logstd-init",
        "-0.7",
        "--count-curr-frac",
        "0.25",
        "--warm-from",
        _abs(root, "training/src/model.pt"),  # made absolute: the child's CWD is gpu_prototype/
        "--warm-src-extra-dim",
        "30",
    ]
    assert fakes["spawn"].last["cwd"] == os.path.join(str(root), "gpu_prototype")


def test_generic_argv_always_has_budget_and_round_trips_through_reservoir_launch(
    root: Path,
) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "gen", knobs={"epochs": 8, "warm_from": "training/src/model.pt"})
    reg.launch("gen")
    argv = fakes["spawn"].last["argv"]
    args = parse_args(argv[len(PREFIX) + 1 :])
    assert args.save_path == _abs(root, "training/gen/model.pt")
    assert args.budget == 20e6
    assert args.epochs == 8
    assert args.warm_from == _abs(root, "training/src/model.pt")
    # Unset knobs are absent from argv, so train_aug keeps its own defaults.
    assert args.N is None and args.roll is None and args.warm_src_extra_dim is None


def test_format_budget_and_knob_flag() -> None:
    assert runs.format_budget(14e6) == "14000000"
    assert runs.format_budget(2.5) == "2.5"
    assert runs.knob_flag("N") == "--N"
    assert runs.knob_flag("warm_src_extra_dim") == "--warm-src-extra-dim"


# --------------------------------------------------------------------------- child env


def test_child_env_sets_selected_flags_and_scrubs_inherited(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", flags=["PROC_TRACTABLE", "PROC_OBS_TERMINAL"])
    reg.launch("run-a")
    env = fakes["spawn"].last["env"]
    proc = {k: v for k, v in env.items() if k.startswith("PROC_")}
    assert proc == {"PROC_TRACTABLE": "1", "PROC_OBS_TERMINAL": "1"}
    assert "PROC_TUNNELS" not in env  # inherited, not selected → scrubbed
    assert "PROC_GAP" not in env  # inherited as "0" (which gates_proc_env reads as ON) → scrubbed
    assert "AUDIT_SEED" not in env
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/home/owner"
    assert "0" not in proc.values()
    # The app's own environment is never mutated.
    assert fakes["environ"]["PROC_TUNNELS"] == "1"


def test_child_env_with_no_flags_has_no_proc_vars(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", flags=[])
    reg.launch("run-a")
    assert not [k for k in fakes["spawn"].last["env"] if k.startswith("PROC_")]


def test_build_child_env_extra_overrides() -> None:
    env = runs.build_child_env({"AUDIT_SEED": "1", "X": "y"}, ["PROC_GAP"], {"AUDIT_SEED": "7"})
    assert env == {"X": "y", "PROC_GAP": "1", "PYTHONUNBUFFERED": "1", "AUDIT_SEED": "7"}


# --------------------------------------------------------------------------- interpreter


def test_settings_are_reread_on_every_launch(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "a")
    save_run(root, "b")
    reg.launch("a")
    assert fakes["spawn"].last["argv"][:5] == PREFIX
    fakes["settings"].update({"micromamba_executable": "mm2", "conda_env": "other"})
    fakes["which"].result = lambda exe: "/x/mm2" if exe == "mm2" else None
    reg.launch("b")
    assert fakes["spawn"].last["argv"][:5] == ["/x/mm2", "run", "-n", "other", "python"]


def test_missing_micromamba_raises_run_error_and_spawns_nothing(root: Path) -> None:
    reg, fakes = make_registry(root)
    fakes["which"].result = None
    save_run(root, "run-a")
    with pytest.raises(RunError, match="micromamba"):
        reg.launch("run-a")
    assert fakes["spawn"].calls == []
    assert fakes["probe"].calls == []
    assert reg.describe("run-a")["state"] == "ready"


def test_micromamba_path_from_settings_is_used_when_it_exists(root: Path) -> None:
    exe = touch(root / "tools" / "micromamba.exe")
    reg, fakes = make_registry(root, settings={"micromamba_executable": exe})
    fakes["which"].result = None
    save_run(root, "run-a")
    reg.launch("run-a")
    assert fakes["spawn"].last["argv"][0] == os.path.abspath(exe)


def test_failed_env_probe_raises_actionable_run_error(root: Path) -> None:
    reg, fakes = make_registry(root, settings={"conda_env": "nope"})
    fakes["probe"].result = False
    save_run(root, "run-a")
    with pytest.raises(RunError, match="nope"):
        reg.launch("run-a")
    assert fakes["spawn"].calls == []
    assert fakes["probe"].calls == [(MICROMAMBA, "nope")]


def test_env_probe_is_memoised_per_exe_and_env(root: Path) -> None:
    reg, fakes = make_registry(root)
    for name in ("a", "b"):
        save_run(root, name)
    reg.launch("a")
    reg.launch("b")
    reg.check_interpreter()
    assert fakes["probe"].calls == [(MICROMAMBA, "dronefly")]
    fakes["settings"]["conda_env"] = "env2"
    reg.check_interpreter()
    assert fakes["probe"].calls[-1] == (MICROMAMBA, "env2")
    assert len(fakes["probe"].calls) == 2


# --------------------------------------------------------------------------- state machine


def test_launch_writes_marker_then_running(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", recipe="exp11_count-curriculum")
    desc = reg.launch("run-a")
    assert desc["state"] == "running" and desc["running"] is True
    assert desc["pid"] == fakes["spawn"].popens[-1].pid
    log = Path(logs.log_path_for(str(root), "run-a")).read_text(encoding="utf-8")
    marker = log.splitlines()[0]
    assert marker.startswith(status.LAUNCH_MARKER)
    assert "recipe=exp11_count-curriculum" in marker
    assert f"save_path={_abs(root, 'training/run-a/model.pt')}" in marker


def test_exit_zero_with_saved_line_is_finished(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    fakes["spawn"].next_output = "[aug] it   20 steps 0.2M\n" + _saved(root, "run-a")
    reg.launch("run-a")
    touch(root / "training" / "run-a" / "model.pt")
    fakes["spawn"].popens[-1].exit(0)
    desc = reg.describe("run-a")
    assert desc["state"] == "finished"
    assert desc["partial"] is False and desc["partial_note"] is None
    assert desc["pid"] is None


def test_exit_zero_without_saved_line_is_failed(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    reg.launch("run-a")
    fakes["spawn"].popens[-1].exit(0)
    assert reg.describe("run-a")["state"] == "failed"


def test_nonzero_exit_is_failed_even_with_saved_line(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    fakes["spawn"].next_output = _saved(root, "run-a")
    reg.launch("run-a")
    fakes["spawn"].popens[-1].exit(1)
    assert reg.describe("run-a")["state"] == "failed"


def test_saved_line_for_another_path_does_not_count(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    fakes["spawn"].next_output = "[aug] saved /somewhere/else.pt\n"
    reg.launch("run-a")
    fakes["spawn"].popens[-1].exit(0)
    assert reg.describe("run-a")["state"] == "failed"


def test_saved_line_from_a_previous_attempt_does_not_count(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    append_log(logs.log_path_for(str(root), "run-a"), _saved(root, "run-a"))
    reg.launch("run-a", overwrite=True)  # writes a new launch marker after the old saved line
    fakes["spawn"].popens[-1].exit(0)
    assert reg.describe("run-a")["state"] == "failed"


def test_stop_signals_then_stopping_then_stopped_with_partial_checkpoint(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    reg.launch("run-a")
    popen = fakes["spawn"].popens[-1]
    desc = reg.stop("run-a")
    assert fakes["signaler"].calls == [(popen, "stop")]
    assert desc["state"] == "stopping" and desc["running"] is True
    touch(root / "training" / "run-a" / "model.pt")  # a periodic checkpoint was written
    popen.exit(-15)
    desc = reg.describe("run-a")
    assert desc["state"] == "stopped"
    assert desc["partial"] is True and desc["partial_note"] == runs.PARTIAL_NOTE


def test_stop_when_not_running_is_run_error(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    with pytest.raises(RunError, match="not running"):
        reg.stop("run-a")
    reg.launch("run-a")
    fakes["spawn"].popens[-1].exit(0)
    with pytest.raises(RunError):
        reg.stop("run-a")
    assert fakes["signaler"].calls == []


def test_double_launch_is_refused(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    reg.launch("run-a")
    with pytest.raises(RunError, match="running"):
        reg.launch("run-a")
    assert len(fakes["spawn"].calls) == 1


def test_relaunch_after_exit_returns_to_running(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    reg.launch("run-a")
    fakes["spawn"].popens[-1].exit(1)
    assert reg.describe("run-a")["state"] == "failed"
    assert reg.launch("run-a")["state"] == "running"


# --------------------------------------------------------------------------- restart reconciliation


def test_restart_reconciles_finished_from_the_saved_line(root: Path) -> None:
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    append_log(
        logs.log_path_for(str(root), "run-a"),
        f"{status.LAUNCH_MARKER} t\n[aug] it 20 steps 0.2M\n" + _saved(root, "run-a"),
    )
    reg, _ = make_registry(root)
    assert reg.describe("run-a")["state"] == "finished"


def test_restart_reconciles_partial_from_checkpoint_without_saved_line(root: Path) -> None:
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    append_log(logs.log_path_for(str(root), "run-a"), f"{status.LAUNCH_MARKER} t\n")
    reg, _ = make_registry(root)
    desc = reg.describe("run-a")
    assert desc["state"] == "partial"
    assert desc["partial"] is True and desc["partial_note"] == runs.PARTIAL_NOTE
    assert desc["audit_eligible"] is True


def test_restart_reconciles_ready_without_checkpoint(root: Path) -> None:
    save_run(root, "run-a")
    reg, _ = make_registry(root)
    desc = reg.describe("run-a")
    assert desc["state"] == "ready"
    assert desc["has_checkpoint"] is False
    assert desc["audit_eligible"] is False
    assert desc["kind"] == "app" and desc["readonly"] is False


def test_restart_reconciles_with_a_custom_save_path(root: Path) -> None:
    save_run(root, "run-a", save_path="ckpts/a.pt")
    touch(root / "ckpts" / "a.pt")
    append_log(logs.log_path_for(str(root), "run-a"), _saved(root, "run-a", "ckpts/a.pt"))
    reg, _ = make_registry(root)
    desc = reg.describe("run-a")
    assert desc["state"] == "finished"
    assert desc["checkpoint"] == "ckpts/a.pt"


# --------------------------------------------------------------------------- overwrite guard


def test_existing_checkpoint_needs_overwrite(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    with pytest.raises(RunError, match="overwrite"):
        reg.launch("run-a")
    assert fakes["spawn"].calls == []
    assert reg.launch("run-a", overwrite=True)["state"] == "running"


def test_launch_without_saved_config_is_run_error(root: Path) -> None:
    reg, fakes = make_registry(root)
    with pytest.raises(RunError, match="save it first"):
        reg.launch("ghost")
    assert fakes["spawn"].calls == []


def test_launch_rejects_invalid_name(root: Path) -> None:
    reg, _ = make_registry(root)
    with pytest.raises(RecipeError):
        reg.launch("../escape")


# --------------------------------------------------------------------------- committed / external


def test_numbered_dir_is_committed_without_asking_git(root: Path) -> None:
    (root / "training" / "07-proc-course-flight").mkdir(parents=True)
    reg, fakes = make_registry(root)
    desc = reg.describe("07-proc-course-flight")
    assert desc["kind"] == "committed" and desc["state"] == "committed"
    assert desc["readonly"] is True
    assert fakes["tracked_probe"].calls == []


def test_git_tracked_dir_is_committed_and_probe_is_cached(root: Path) -> None:
    (root / "training" / "oval").mkdir(parents=True)
    reg, fakes = make_registry(root)
    fakes["tracked_probe"].result = True
    assert reg.describe("oval")["kind"] == "committed"
    reg.describe("oval")
    reg.list_runs()
    assert fakes["tracked_probe"].calls == [("training/oval",)]


def test_untracked_dir_without_run_json_is_external(root: Path) -> None:
    (root / "training" / "cli-run").mkdir(parents=True)
    reg, fakes = make_registry(root)
    desc = reg.describe("cli-run")
    assert desc["kind"] == "external" and desc["state"] == "external"
    assert fakes["tracked_probe"].calls == [("training/cli-run",)]


def test_list_runs_enumerates_app_committed_external_and_skips_bad_names(root: Path) -> None:
    save_run(root, "mine")
    (root / "training" / "01-hover").mkdir(parents=True)
    (root / "training" / "cli-run").mkdir(parents=True)
    (root / "training" / ".scratch").mkdir(parents=True)  # not a valid run name
    touch(root / "training" / "README.md")  # a file, not a run
    reg, _ = make_registry(root)
    listed = {r["name"]: r["kind"] for r in reg.list_runs()}
    assert listed == {"mine": "app", "01-hover": "committed", "cli-run": "external"}


def test_unknown_run_is_file_not_found(root: Path) -> None:
    reg, _ = make_registry(root)
    with pytest.raises(FileNotFoundError):
        reg.describe("ghost")


@pytest.mark.parametrize("dirname", ["01-hover", "cli-run"])
def test_launch_record_delete_refused_on_read_only_dirs(root: Path, dirname: str) -> None:
    touch(root / "training" / dirname / "model_readout.pt")
    reg, fakes = make_registry(root)
    with pytest.raises(RunError, match="read-only"):
        reg.launch(dirname)
    with pytest.raises(RunError, match="read-only"):
        reg.record(dirname)
    with pytest.raises(RunError, match="read-only"):
        reg.delete(dirname)
    assert fakes["spawn"].calls == []
    assert (root / "training" / dirname / "model_readout.pt").is_file()


# --------------------------------------------------------------------------- audit eligibility


def test_committed_without_readout_is_not_audit_eligible(root: Path) -> None:
    (root / "training" / "08-x").mkdir(parents=True)
    reg, _ = make_registry(root)
    assert reg.audit_eligible("08-x") == (False, "no model_readout.pt")
    desc = reg.describe("08-x")
    assert desc["audit_eligible"] is False and desc["audit_reason"] == "no model_readout.pt"


def test_committed_with_readout_is_audit_eligible(root: Path) -> None:
    touch(root / "training" / "11-x" / "model_readout.pt")
    reg, _ = make_registry(root)
    assert reg.audit_eligible("11-x") == (True, None)
    assert reg.checkpoint_path("11-x") == str(root / "training" / "11-x" / "model_readout.pt")


def test_live_run_is_not_audit_eligible(root: Path) -> None:
    reg, _ = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    reg.launch("run-a", overwrite=True)
    eligible, reason = reg.audit_eligible("run-a")
    assert eligible is False and "running" in reason


def test_unknown_run_is_not_audit_eligible(root: Path) -> None:
    reg, _ = make_registry(root)
    assert reg.audit_eligible("ghost") == (False, "unknown run")


# --------------------------------------------------------------------------- record


def test_record_argv_env_and_log(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a", flags=["PROC_RAYCAST", "PROC_GAP"])
    touch(root / "training" / "run-a" / "model.pt")
    out = reg.record("run-a")
    call = fakes["spawn"].last
    assert call["argv"] == [
        *PREFIX,
        os.path.join(str(root), "gpu_prototype", "record_aug.py"),
        "proc",
        _abs(root, "training/run-a/model.pt"),
        "run-a",
        "10",
    ]
    assert call["cwd"] == os.path.join(str(root), "gpu_prototype")
    assert call["log"] == os.path.join(str(root), "training", "run-a", "logs", "record.log")
    proc = {k: v for k, v in call["env"].items() if k.startswith("PROC_")}
    assert proc == {"PROC_RAYCAST": "1", "PROC_GAP": "1"}
    assert out == {"name": "run-a", "episodes": 10, "pid": fakes["spawn"].popens[-1].pid}
    assert reg.describe("run-a")["recording"] is True


def test_record_custom_episode_count(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    reg.record("run-a", episodes="3")
    assert fakes["spawn"].last["argv"][-1] == "3"


@pytest.mark.parametrize("episodes", [0, -1, "x", None])
def test_record_rejects_bad_episode_count(root: Path, episodes) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    with pytest.raises(RecipeError):
        reg.record("run-a", episodes=episodes)
    assert fakes["spawn"].calls == []


def test_record_refused_while_run_is_live(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    reg.launch("run-a", overwrite=True)
    with pytest.raises(RunError, match="running"):
        reg.record("run-a")
    reg.stop("run-a")
    with pytest.raises(RunError, match="stopping"):
        reg.record("run-a")
    assert len(fakes["spawn"].calls) == 1


def test_record_refused_without_checkpoint(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    with pytest.raises(RunError, match="no checkpoint"):
        reg.record("run-a")


def test_live_recording_blocks_launch_delete_and_second_record(root: Path) -> None:
    reg, fakes = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    reg.record("run-a")
    with pytest.raises(RunError, match="recording"):
        reg.launch("run-a", overwrite=True)
    with pytest.raises(RunError, match="recording"):
        reg.delete("run-a")
    with pytest.raises(RunError, match="already recording"):
        reg.record("run-a")
    fakes["spawn"].popens[-1].exit(0)
    assert reg.describe("run-a")["recording"] is False
    reg.record("run-a")  # allowed again once the recorder exited


# --------------------------------------------------------------------------- delete


def test_delete_removes_only_the_training_dir(root: Path) -> None:
    reg, _ = make_registry(root)
    save_run(root, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    append_log(logs.log_path_for(str(root), "run-a"), "x\n")
    sibling = touch(root / "training" / "run-b" / "keep.txt")
    out = reg.delete("run-a")
    assert out == {"name": "run-a", "deleted": True, "kept_save_path": None}
    assert not (root / "training" / "run-a").exists()
    assert os.path.isfile(sibling)


def test_delete_reports_but_keeps_an_outside_save_path(root: Path) -> None:
    reg, _ = make_registry(root)
    save_run(root, "run-a", save_path="ckpts/a.pt")
    ckpt = touch(root / "ckpts" / "a.pt")
    out = reg.delete("run-a")
    assert out["kept_save_path"] == "ckpts/a.pt"
    assert os.path.isfile(ckpt)
    assert not (root / "training" / "run-a").exists()


def test_delete_refuses_a_live_run(root: Path) -> None:
    reg, _ = make_registry(root)
    save_run(root, "run-a")
    reg.launch("run-a")
    with pytest.raises(RunError, match="running"):
        reg.delete("run-a")
    assert (root / "training" / "run-a" / "run.json").is_file()


def test_delete_unknown_and_invalid_names(root: Path) -> None:
    reg, _ = make_registry(root)
    with pytest.raises(FileNotFoundError):
        reg.delete("ghost")
    with pytest.raises(RecipeError):
        reg.delete("..")


def test_delete_refuses_a_symlink_escaping_the_training_root(root: Path, tmp_path_factory) -> None:
    outside = tmp_path_factory.mktemp("outside")
    touch(outside / "run.json", b"{}")
    (root / "training").mkdir(exist_ok=True)
    try:
        os.symlink(outside, root / "training" / "evil", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this host")
    reg, _ = make_registry(root)
    with pytest.raises(RunError, match="symlink escape"):
        reg.delete("evil")
    assert (outside / "run.json").is_file()


# --------------------------------------------------------------------------- GPU probe


def test_gpu_probe_is_cached_for_the_ttl_with_the_injected_clock(root: Path) -> None:
    reg, fakes = make_registry(root)
    clock, probe = fakes["clock"], fakes["gpu_probe"]
    first = reg.gpu_status()
    assert first == {"util_pct": 50.0, "mem_used_mb": 1000.0, "mem_total_mb": 8000.0}
    clock.now = runs.GPU_PROBE_TTL_S - 0.01
    assert reg.gpu_status() == first
    assert len(probe.calls) == 1
    clock.now = runs.GPU_PROBE_TTL_S
    reg.gpu_status()
    assert len(probe.calls) == 2


def test_gpu_probe_failure_is_none_and_cached(root: Path) -> None:
    probe = Counter(RuntimeError("no nvidia-smi"))
    reg, fakes = make_registry(root, gpu_probe=probe)
    assert reg.gpu_status() is None
    probe.result = None
    fakes["clock"].now = 1.0
    assert reg.gpu_status() is None
    assert len(probe.calls) == 1  # the failure is cached too


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("37, 1024, 8192\n", {"util_pct": 37.0, "mem_used_mb": 1024.0, "mem_total_mb": 8192.0}),
        ("5, 10, 20\n90, 1, 2\n", {"util_pct": 5.0, "mem_used_mb": 10.0, "mem_total_mb": 20.0}),
        ("", None),
        ("[N/A], 1, 2\n", None),
        ("garbage\n", None),
    ],
)
def test_parse_nvidia_smi(text: str, expected) -> None:
    assert runs.parse_nvidia_smi(text) == expected
