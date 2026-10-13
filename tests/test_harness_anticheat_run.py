"""UC69 P2: every live Liftoff ``harness run`` refuses while League / Riot Vanguard is running.

The check runs before arming on the detector path and the teacher path alike; dry runs and the
recording-only sink (nothing actuated) skip it. No game, no pad: a fake actuating sink is
registered, telemetry is replayed from the synthetic UC-62 CSV.
"""

from __future__ import annotations

import pytest

from drone_fly.harness import registry
from drone_fly.harness.liftoff import anticheat
from drone_fly.harness.liftoff.anticheat import AntiCheatRunningError
from drone_fly.harness.session import run_harness
from tests.harness_support import DATA_DIR, FakeSink

FIXTURE_CSV = DATA_DIR / "liftoff_telemetry.csv"


class _Sink(FakeSink):
    instances: list[_Sink] = []

    @classmethod
    def from_config(cls, cfg, ctx):
        s = cls()
        cls.instances.append(s)
        return s


@pytest.fixture
def fake_sink():
    _Sink.instances.clear()
    registry.register("action_sink", "uc69-fake-pad", _Sink)
    yield _Sink.instances
    registry.unregister("action_sink", "uc69-fake-pad")


def _cfg(sink: str) -> dict:
    return {
        "parts": {
            "state_source": "liftoff-csv",
            "frame_source": None,
            "gate_estimator": "opencv",
            "action_sink": sink,
            "abort_signal": None,
            "focus_probe": None,
        },
        "telemetry": {"replay_csv": str(FIXTURE_CSV)},
        "free_space": {"assume_no_obstacles": True},
    }


def _run(ck, k1, tmp_path, sink, **kw):
    return run_harness(
        "liftoff", ck, 30, config=_cfg(sink), k1_path=str(k1), device="cpu", duration_s=0.2,
        output_root=str(tmp_path), **kw,
    )  # fmt: skip


def test_live_run_refuses_with_league_running_before_arming(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, fake_sink, monkeypatch
):
    """The DEFAULT check (tasklist) on the detector path: League present -> refused, unarmed."""
    monkeypatch.setattr(
        anticheat, "running_process_names", lambda: ["explorer.exe", "LeagueClient.exe", "vgc.exe"]
    )
    with pytest.raises(AntiCheatRunningError) as ei:
        _run(harness_checkpoint_30, harness_k1_dir, tmp_path, "uc69-fake-pad")
    assert ei.value.processes == ["LeagueClient.exe", "vgc.exe"]
    sink = fake_sink[-1]
    assert sink.armed is False and sink.sent == [] and sink.closed >= 1


def test_live_run_proceeds_without_anticheat(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, fake_sink
):
    calls = []
    s = _run(harness_checkpoint_30, harness_k1_dir, tmp_path, "uc69-fake-pad",
             anticheat=lambda: calls.append(1))  # fmt: skip
    assert calls == [1] and s.result.reason == "completed"
    assert fake_sink[-1].armed is True and fake_sink[-1].sent


def test_injected_check_refusal_maps_to_a_config_error(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, fake_sink
):
    def league():
        raise AntiCheatRunningError(["League of Legends.exe"])

    with pytest.raises(AntiCheatRunningError):
        _run(harness_checkpoint_30, harness_k1_dir, tmp_path, "uc69-fake-pad", anticheat=league)
    assert fake_sink[-1].armed is False


def test_dry_run_skips_the_check(harness_contract, harness_checkpoint_30, harness_k1_dir,
                                 tmp_path):  # fmt: skip
    def never():
        raise AssertionError("the anti-cheat check must not run on a dry run")

    s = _run(harness_checkpoint_30, harness_k1_dir, tmp_path, "uc69-fake-pad", dry_run=True,
             anticheat=never)  # fmt: skip
    assert s.result.reason == "completed"


def test_recording_only_sink_skips_the_check(harness_contract, harness_checkpoint_30,
                                             harness_k1_dir, tmp_path):  # fmt: skip
    def never():
        raise AssertionError("nothing is actuated: no check")

    s = _run(harness_checkpoint_30, harness_k1_dir, tmp_path, "recording", anticheat=never)
    assert s.result.reason == "completed"


def test_cli_exits_2_on_anticheat(monkeypatch, capsys):
    from drone_fly.cli import main as cli
    from drone_fly.harness import session

    def refuse(*a, **k):
        raise AntiCheatRunningError(["RiotClientServices.exe"])

    monkeypatch.setattr(session, "run_harness", refuse)
    rc = cli.main(["harness", "run", "--harness", "liftoff", "--checkpoint", "x.pt",
                   "--extra-dim", "30"])  # fmt: skip
    assert rc == 2
    assert "RiotClientServices.exe" in capsys.readouterr().err
