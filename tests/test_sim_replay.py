"""UC67 AC8: open-loop replay of a sysid session through the nominal plant.

A hermetic synthetic session (``sysid.synthetic.generate_session``: the real sequencer, recorder
and pad sink against a known micro-like plant) is fitted into a profile, then replayed. The
replay must reproduce the session closely (the plant and the synthetic world share the sysid
signal chain), a wrong profile must show up as a large error, windows shorter than 2 s are left
out of the 2 s statistic with the counts reported [C7], and the ``sysid.md`` section is written
and replaced idempotently and survives a report regeneration. The real UC66 Liftoff result (the
gated micro number) is local evidence, not a CI test: see the drone's ``sysid.md``.
"""

from __future__ import annotations

import dataclasses

import pytest

from drone_fly import drones
from drone_fly.cli.main import main
from drone_fly.sim import replay as replay_mod
from drone_fly.sim.replay import (
    GATE_ATT_DEG,
    GATE_VEL_M_S,
    REPORT_BEGIN,
    REPORT_END,
    ReplayResult,
    WindowResult,
    extract_report_section,
    replay_session,
    report_markdown,
    write_report_section,
)
from drone_fly.sysid.fit import fit_session, load_session
from drone_fly.sysid.model import MICRO_LIKE
from drone_fly.sysid.report import write_report
from drone_fly.sysid.schedule import ScheduleSettings
from drone_fly.sysid.synthetic import generate_session

ID = "liftoff-test-micro"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """``(session dir, fit, drones root, profile, replay result)`` of a short synthetic session."""
    base = tmp_path_factory.mktemp("replay")
    sdir = base / "20261012T100000Z"
    generate_session(
        MICRO_LIKE, sdir, seed=3,
        schedule_settings=ScheduleSettings(amplitudes=(0.3, 0.7, 1.0), passes=2),
        expected_hover=(0.40, 0.56),
    )  # fmt: skip
    fit = fit_session(sdir, ID)
    root = base / "drones"
    write_report(fit, root / ID, drone_id=ID, micro=True)
    prof = drones.load(ID, root)
    return sdir, fit, root, prof, replay_session(sdir, prof)


def test_replay_of_a_synthetic_session_has_small_errors(session):
    _, fit, _, prof, r = session
    assert prof.fit_quality == "good" and r.gated  # a good fit is gated by default
    assert len(r.windows) >= 10
    assert r.median("att_deg", 1.0) < 0.5
    assert r.median("vel_m_s", 1.0) < 0.1
    assert r.median_mean("att_mean_deg") < 0.5 and r.median_mean("vel_mean_m_s") < 0.1
    assert r.passed and r.verdict.startswith("AC8 MET")
    assert r.drone_id == ID and r.session == "20261012T100000Z"


def test_only_completed_held_out_windows_are_replayed(session):
    sdir, _, _, _, r = session
    data = load_session(sdir)
    held = {e.id for e in data.schedule if e.held_out and e.id in data.windows}
    assert {w.experiment for w in r.windows} == held
    assert not any(e.held_out is False and e.id in {w.experiment for w in r.windows}
                   for e in data.schedule)  # fmt: skip


def test_windows_shorter_than_2s_are_left_out_of_the_2s_statistic(session):
    r = session[4]
    long = [w for w in r.windows if w.span_s >= 2.0]
    short = [w for w in r.windows if w.span_s < 2.0]
    assert long and short  # the fixture session exercises both
    for w in short:
        assert w.att_deg[2.0] is None and w.vel_m_s[2.0] is None
        assert w.att_deg[1.0] is not None
    for w in long:
        assert w.att_deg[2.0] is not None
    assert r.counts(2.0) == len(long) and r.counts(1.0) == len(r.windows)
    md = report_markdown(r)
    assert f"| windows | {len(r.windows)} | {len(long)} | {len(r.windows)} |" in md


def test_a_wrong_profile_shows_up_as_a_large_error(session):
    sdir, _, _, prof, good = session
    wrong_rates = {
        a: dataclasses.replace(r, rc_rate=r.rc_rate * 1.6) for a, r in prof.rates.items()
    }
    wrong = dataclasses.replace(prof, rates=wrong_rates)
    bad = replay_session(sdir, wrong)
    assert bad.median("att_deg", 1.0) > 20 * good.median("att_deg", 1.0)
    weak = dataclasses.replace(
        prof,
        thrust_curve=dataclasses.replace(
            prof.thrust_curve, accel_g=tuple(0.8 * g for g in prof.thrust_curve.accel_g)
        ),
    )
    assert replay_session(sdir, weak).median("vel_m_s", 1.0) > 0.5


def _window(att1, vel1, span=2.5):
    att = {1.0: att1, 2.0: att1 if span >= 2 else None}
    vel = {1.0: vel1, 2.0: vel1 if span >= 2 else None}
    return WindowResult("e", "coast", span, att, vel, att1, vel1)


def test_gate_and_verdict():
    ok = ReplayResult("s", "d", "good", [_window(5.0, 0.5), _window(12.0, 0.2, 1.5)], True)
    assert ok.median("att_deg", 1.0) == 8.5 and ok.median("att_deg", 2.0) == 5.0
    assert ok.counts(2.0) == 1 and ok.passed and "AC8 MET" in ok.verdict
    over_att = ReplayResult("s", "d", "good", [_window(GATE_ATT_DEG + 0.1, 0.1)], True)
    over_vel = ReplayResult("s", "d", "good", [_window(1.0, GATE_VEL_M_S + 0.01)], True)
    assert not over_att.passed and "NOT MET" in over_att.verdict
    assert not over_vel.passed
    poor = ReplayResult("s", "d", "poor", [_window(30.0, 3.0)], False)
    assert poor.verdict.startswith("Not gated (fit_quality poor)")


# ----------------------------------------------------------------------------- report


def test_report_section_is_written_and_replaced_idempotently(tmp_path, session):
    r = session[4]
    md = tmp_path / "sysid.md"
    md.write_text("# Sysid\n\nfitted values\n", encoding="utf-8")
    write_report_section(md, r)
    text1 = md.read_text(encoding="utf-8")
    assert text1.startswith("# Sysid\n\nfitted values\n")
    assert text1.count(REPORT_BEGIN) == 1 and text1.count(REPORT_END) == 1
    assert "## Plant replay (UC67)" in text1 and r.verdict in text1
    write_report_section(md, r)
    assert md.read_text(encoding="utf-8") == text1  # idempotent
    other = ReplayResult("other-session", ID, "good", [_window(3.0, 0.3)], True)
    write_report_section(md, other)
    text2 = md.read_text(encoding="utf-8")
    assert text2.count(REPORT_BEGIN) == 1 and "other-session" in text2
    assert r.session not in text2 and text2.startswith("# Sysid\n\nfitted values\n")
    assert extract_report_section(text2) == report_markdown(other).rstrip("\n")
    assert extract_report_section("# no block") is None
    fresh = tmp_path / "new" / "sysid.md"
    fresh.parent.mkdir()
    write_report_section(fresh, r)
    assert fresh.read_text(encoding="utf-8").startswith(REPORT_BEGIN)


def test_report_section_survives_a_sysid_report_regeneration(tmp_path, session):
    _, fit, _, _, r = session
    out = tmp_path / ID
    _, md = write_report(fit, out, drone_id=ID, micro=True)
    write_report_section(md, r)
    block = extract_report_section(md.read_text(encoding="utf-8"))
    _, md2 = write_report(fit, out, drone_id=ID, micro=True)  # refit -> rewrite the report
    text = md2.read_text(encoding="utf-8")
    assert extract_report_section(text) == block and text.count(REPORT_BEGIN) == 1
    assert "Plant replay (UC67)" in text and text.index(REPORT_BEGIN) > 0


# ----------------------------------------------------------------------------- CLI


def test_cli_replay_sysid_writes_the_report(session, monkeypatch, capsys):
    sdir, _, root, _, _ = session
    monkeypatch.setenv(drones.store.ENV_ROOT, str(root))
    rc = main(["sim", "replay-sysid", str(sdir), "--drone", ID, "--write-report"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "AC8 MET" in out and "1 s: median attitude" in out and "2 s:" in out
    text = (root / ID / "sysid.md").read_text(encoding="utf-8")
    assert text.count(REPORT_BEGIN) == 1


def test_cli_exit_code_follows_the_gate(session, monkeypatch, capsys):
    sdir, _, root, _, _ = session
    monkeypatch.setenv(drones.store.ENV_ROOT, str(root))
    failing = ReplayResult("s", ID, "good", [_window(25.0, 2.0)], True)
    calls = []

    def fake(session_dir, profile, *, gated=None, **kw):
        calls.append(gated)
        return dataclasses.replace(failing, gated=True if gated is None else gated)

    monkeypatch.setattr(replay_mod, "replay_session", fake)
    assert main(["sim", "replay-sysid", str(sdir), "--drone", ID]) == 1
    assert main(["sim", "replay-sysid", str(sdir), "--drone", ID, "--no-gate"]) == 0
    assert main(["sim", "replay-sysid", str(sdir), "--drone", ID, "--gate"]) == 1
    assert calls == [None, False, True]
    assert "NOT MET" in capsys.readouterr().out


@pytest.mark.parametrize(
    "drone_id,verdict",
    [("liftoff-gremlin-micro", "**AC8 MET:"), ("liftoff-5inch-vortex-pro-250", "**Not gated")],
)
def test_committed_profiles_carry_the_generated_replay_section(drone_id, verdict):
    """The real UC66 Liftoff replay (gitignored sessions, run locally) is recorded in sysid.md:
    the micro is gated and met, the poor-fit 5-inch is reported as-is."""
    text = drones.store.profile_path(drone_id).with_name("sysid.md").read_text(encoding="utf-8")
    block = extract_report_section(text)
    assert block is not None and text.count(REPORT_BEGIN) == 1
    assert "## Plant replay (UC67)" in block and verdict in block
