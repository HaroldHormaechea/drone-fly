"""UC-66: refuse to drive Liftoff while League of Legends / Riot Vanguard is running.

Pure pattern matching plus the refusals: no test here lists real processes, clicks, or sends pad
input (the enumerator is injected; ``tasklist`` is replaced).
"""

from __future__ import annotations

import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff import anticheat
from drone_fly.harness.liftoff.anticheat import (
    AntiCheatRunningError,
    anticheat_processes,
    require_no_anticheat,
)


@pytest.mark.parametrize(
    "name",
    [
        "LeagueClient.exe",
        "LeagueClientUx.exe",
        "leagueclientuxrender.exe",
        "League of Legends.exe",
        "RiotClientServices.exe",
        "RiotClientUx",
        "vgc.exe",
        "VGC.EXE",
        "vgc",
        "vgtray.exe",
        "  vgtray.exe  ",
    ],
)
def test_anticheat_names_match_case_insensitively_with_or_without_exe(name):
    assert anticheat_processes([name]) == [name.strip()]


@pytest.mark.parametrize(
    "name",
    [
        "Liftoff.exe",
        "explorer.exe",
        "svchost.exe",
        "vgcx.exe",  # vgc is an exact name, not a prefix
        "myvgc.exe",
        "steam.exe",
        "python.exe",
        "Leaguetool.exe",
        "",
    ],
)
def test_no_false_positives(name):
    assert anticheat_processes([name]) == []


def test_matches_keep_order_and_drop_repeats():
    names = ["Liftoff.exe", "vgc.exe", "LeagueClient.exe", "vgc.exe", "steam.exe"]
    assert anticheat_processes(names) == ["vgc.exe", "LeagueClient.exe"]


def test_require_no_anticheat_names_the_processes_and_is_a_config_error():
    with pytest.raises(AntiCheatRunningError) as exc:
        require_no_anticheat(lambda: ["Liftoff.exe", "vgc.exe", "RiotClientServices.exe"])
    assert isinstance(exc.value, HarnessConfigError)  # CLI exit 2
    assert exc.value.processes == ["vgc.exe", "RiotClientServices.exe"]
    assert "vgc.exe" in str(exc.value) and "Vanguard" in str(exc.value)


def test_require_no_anticheat_passes_on_a_clean_list():
    require_no_anticheat(lambda: ["Liftoff.exe", "steam.exe"])


def test_default_enumerator_parses_tasklist_csv(monkeypatch):
    out = (
        '"System Idle Process","0","Services","0","8 K"\n'
        '"Liftoff.exe","4242","Console","1","1,234,567 K"\n'
        '"vgc.exe","77","Services","0","9,000 K"\n'
    )

    class Done:
        stdout = out

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return Done()

    monkeypatch.setattr(anticheat.subprocess, "run", fake_run)
    assert anticheat.running_process_names() == ["System Idle Process", "Liftoff.exe", "vgc.exe"]
    assert calls == [["tasklist", "/FO", "CSV", "/NH"]]
    with pytest.raises(AntiCheatRunningError, match="vgc.exe"):
        require_no_anticheat()


def test_missing_tasklist_is_a_config_error(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("tasklist")

    monkeypatch.setattr(anticheat.subprocess, "run", boom)
    with pytest.raises(HarnessConfigError, match="tasklist"):
        anticheat.running_process_names()


def test_select_drone_refuses_before_any_game_access(monkeypatch):
    from drone_fly.harness.liftoff import ui_driver

    def touched(*a, **kw):
        pytest.fail("must not touch the game while anti-cheat runs")

    for fn in ("_focused", "click", "_click_xy", "_grab_screen"):
        monkeypatch.setattr(ui_driver, fn, touched)
    monkeypatch.setattr(ui_driver, "DRONE_MENU_PATH", ("single_player",))
    with pytest.raises(AntiCheatRunningError):
        ui_driver.select_drone(
            "Test small drone", anticheat=lambda: require_no_anticheat(lambda: ["vgc.exe"])
        )
    # the default check uses the process list
    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["LeagueClient.exe"])
    with pytest.raises(AntiCheatRunningError):
        ui_driver.select_drone("Test small drone")


def test_live_sysid_refuses_before_any_input(monkeypatch, tmp_path):
    """``sysid liftoff`` on the live plant checks the process list before building any part."""
    from drone_fly.sysid import session

    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["vgtray.exe"])
    for fn in ("select_drone", "reset"):
        monkeypatch.setattr(
            f"drone_fly.harness.liftoff.ui_driver.{fn}",
            lambda *a, **kw: pytest.fail("must not touch the game"),
        )
    with pytest.raises(AntiCheatRunningError):
        session.run_sysid_liftoff(
            "liftoff-test-micro", session="s1", plant="liftoff", select=True,
            liftoff_name="Test small drone", out_root=tmp_path,
        )  # fmt: skip
    assert not (tmp_path / "liftoff-test-micro" / "s1").exists()


def test_cli_exit_code_2_when_anticheat_runs(monkeypatch, tmp_path, capsys):
    from drone_fly.cli.main import main

    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["vgc.exe"])
    monkeypatch.setattr("drone_fly.harness.config.find_repo_root", lambda *a, **kw: tmp_path)
    assert main(["sysid", "liftoff", "--drone", "liftoff-test-micro", "--session", "s"]) == 2
    assert "vgc.exe" in capsys.readouterr().err
    assert main(["harness", "ui", "select-drone", "--name", "Test small drone"]) == 2
