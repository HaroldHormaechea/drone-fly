"""UC69 CLI: ``harness run --teacher ...`` argument rules and exit codes, ``harness oracle-report``.

``run_harness`` / ``write_report`` are replaced: these tests pin the CLI contract (exit 6 for a
misaligned start, the ``--teacher`` / ``--checkpoint`` exclusion), not the session itself.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from drone_fly.cli import main as cli
from drone_fly.harness import session


@pytest.fixture
def fake_run(monkeypatch):
    calls: list[dict] = []
    state = {"reason": "completed", "failsafe": False}

    def run(*args, **kw):
        calls.append({"args": args, "kw": kw})
        r = SimpleNamespace(
            reason=state["reason"], ticks=3, error=None, missed_deadlines=0,
            is_failsafe_stop=state["failsafe"], failsafe_ok=True,
        )  # fmt: skip
        return SimpleNamespace(result=r, run_dir=None, session_dir=Path("S"),
                               overlay_process=None)  # fmt: skip

    monkeypatch.setattr(session, "run_harness", run)
    return SimpleNamespace(calls=calls, state=state)


TEACHER = ["harness", "run", "--harness", "liftoff", "--teacher", "runs/t1", "--track", "t-01"]


@pytest.mark.parametrize(
    "extra",
    [["--checkpoint", "m.pt"], ["--extra-dim", "30"], ["--checkpoint", "m.pt", "--extra-dim", "30"]],
)
def test_teacher_is_exclusive_with_checkpoint_and_extra_dim(fake_run, extra, capsys):
    assert cli.main(TEACHER + extra) == 2
    assert "exclusive" in capsys.readouterr().err
    assert fake_run.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        ["harness", "run", "--harness", "liftoff"],
        ["harness", "run", "--harness", "liftoff", "--checkpoint", "m.pt"],
        ["harness", "run", "--harness", "liftoff", "--extra-dim", "30"],
    ],
)
def test_checkpoint_needs_extra_dim_or_teacher(fake_run, argv):
    assert cli.main(argv) == 2 and fake_run.calls == []


def test_oracle_needs_teacher(fake_run, capsys):
    argv = ["harness", "run", "--harness", "liftoff", "--checkpoint", "m.pt", "--extra-dim", "30",
            "--perception", "oracle"]  # fmt: skip
    assert cli.main(argv) == 2
    assert "--perception oracle needs --teacher" in capsys.readouterr().err


def test_extra_dim_still_validated_by_argparse():
    with pytest.raises(SystemExit) as ei:
        cli.main(["harness", "run", "--harness", "liftoff", "--checkpoint", "m.pt",
                  "--extra-dim", "148"])  # fmt: skip
    assert ei.value.code == 2


def test_teacher_flags_reach_run_harness(fake_run, capsys):
    argv = TEACHER + ["--perception", "oracle", "--trackdata-root", "td", "--scenes-root", "sc",
                      "--allow-concurrent-training", "--session-stamp", "S1", "--dry-run"]
    assert cli.main(argv) == 0
    kw = fake_run.calls[0]["kw"]
    args = fake_run.calls[0]["args"]
    assert args == ("liftoff", None, None)
    assert kw["teacher"] == "runs/t1" and kw["track"] == "t-01" and kw["perception"] == "oracle"
    assert (kw["trackdata_root"], kw["scenes_root"]) == ("td", "sc")
    assert kw["allow_concurrent_training"] is True and kw["session_stamp"] == "S1"
    assert kw["session_dir"] is None and kw["dry_run"] is True
    assert "session: S" in capsys.readouterr().out


def test_legacy_run_gets_no_teacher_kwargs(fake_run):
    assert cli.main(["harness", "run", "--harness", "liftoff", "--checkpoint", "m.pt",
                     "--extra-dim", "30"]) == 0  # fmt: skip
    assert "teacher" not in fake_run.calls[0]["kw"]


@pytest.mark.parametrize(
    "reason,failsafe,code",
    [
        ("start_misaligned", True, 6),
        ("episode_end:crash_settled", True, 5),
        ("stale_telemetry", True, 5),
        ("course_finished", False, 0),
        ("completed", False, 0),
    ],
)
def test_exit_codes(fake_run, reason, failsafe, code):
    fake_run.state.update(reason=reason, failsafe=failsafe)
    assert cli.main(TEACHER) == code


def test_oracle_report_subcommand(monkeypatch, capsys, tmp_path):
    from drone_fly.harness.offline import oracle_report

    calls = []

    def fake_write(session, *, sim_episodes, sim):
        calls.append((session, sim_episodes, sim))
        return {"verdict": {"text": "Control transfers: ..."},
                "written": {"md": str(tmp_path / "report.md")}}  # fmt: skip

    monkeypatch.setattr(oracle_report, "write_report", fake_write)
    assert cli.main(["harness", "oracle-report", "--session", "sess"]) == 0
    assert cli.main(["harness", "oracle-report", "--session", "sess", "--sim-episodes", "8",
                     "--no-sim"]) == 0  # fmt: skip
    assert calls == [("sess", 32, True), ("sess", 8, False)]
    out = capsys.readouterr().out
    assert "Control transfers" in out and "report.md" in out


def test_oracle_report_needs_a_session():
    with pytest.raises(SystemExit) as ei:
        cli.main(["harness", "oracle-report"])
    assert ei.value.code == 2
