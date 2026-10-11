"""UC-63 AC6 / AC7: audit parsing, the honesty verdict, and the audit runner (hermetic).

Fixtures under ``tests/fixtures/app_audit/`` are built from the exact print statements of
``gpu_prototype/verify_proc.py`` / ``diagnose_proc.py``. The audit runner goes through a
:class:`~app.runs.RunRegistry` with a fake spawner (no micromamba, no GPU); the fake "child" writes
a fixture into the audit log, exactly as the real script's stdout would land there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from app import audits
from app.audits import honesty_verdict, parse_diagnose, parse_verify
from app.configs_io import RecipeError
from app.runs import RunError

from tests.app_support import MICROMAMBA, fixture_text, make_registry, save_run, touch

TERMINAL = ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING", "PROC_OBS_TERMINAL"]
NON_TERMINAL = ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING"]


def _verdict(fixture: str, flags: list[str]) -> dict:
    return honesty_verdict(parse_verify(fixture_text(fixture)), flags)


def _parsed(**overrides) -> dict:
    """A clean, honest parse_verify() result; override fields to hit one rule at a time."""
    base = {
        "completion": {"n": 300, "eps": 512, "pct": 59.0},
        "laps_required": 1.0,
        "gate_passes_mean": 5.0,
        "barging_pct": 0.0,
        "avg_inside_steps": 0.0,
        "min_clearance_m": 0.6,
        "mean_speed": 2.3,
        "mean_tilt_deg": 21.0,
        "mean_steps_alive": 600.0,
        "max_steps": 1500,
        "completed_only": {"barged_pct": 0.0, "avg_inside_steps": 0.0, "mean_speed": 2.4},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- parse_verify


def test_parse_verify_honest_fixture_reads_every_field() -> None:
    assert parse_verify(fixture_text("verify_honest.txt")) == {
        "completion": {"n": 302, "eps": 512, "pct": 59.0},
        "laps_required": 1.0,
        "gate_passes_mean": 5.3,
        "barging_pct": 38.0,
        "avg_inside_steps": 0.41,
        "min_clearance_m": 0.62,
        "mean_speed": 2.31,
        "mean_tilt_deg": 21.0,
        "mean_steps_alive": 612.0,
        "max_steps": 1500,
        "completed_only": {"barged_pct": 0.0, "avg_inside_steps": 0.0, "mean_speed": 2.45},
    }


def test_parse_verify_negative_clearance_and_completed_barging() -> None:
    parsed = parse_verify(fixture_text("verify_barging.txt"))
    assert parsed["min_clearance_m"] == -0.21
    assert parsed["completed_only"] == {
        "barged_pct": 64.0,
        "avg_inside_steps": 15.12,
        "mean_speed": 2.91,
    }


def test_parse_verify_zero_completions_has_no_completed_line() -> None:
    parsed = parse_verify(fixture_text("verify_hovering.txt"))
    assert parsed["completion"] == {"n": 0, "eps": 512, "pct": 0.0}
    assert parsed["completed_only"] is None
    assert parsed["mean_speed"] == 0.04 and parsed["mean_tilt_deg"] == 1.0


def test_parse_verify_truncated_output_leaves_missing_fields_none() -> None:
    parsed = parse_verify(fixture_text("verify_truncated.txt"))
    assert parsed["completion"]["n"] == 302
    assert parsed["gate_passes_mean"] == 5.3
    for key in ("barging_pct", "min_clearance_m", "mean_speed", "mean_tilt_deg", "completed_only"):
        assert parsed[key] is None


@pytest.mark.parametrize("text", ["", "garbage\n", None, 42])
def test_parse_verify_never_raises(text) -> None:
    parsed = parse_verify(text)
    assert all(v is None for v in parsed.values())


def test_parse_verify_last_run_wins_when_log_has_two() -> None:
    text = fixture_text("verify_barging.txt") + fixture_text("verify_honest.txt")
    assert parse_verify(text)["completed_only"]["barged_pct"] == 0.0


def test_parse_verify_handles_crlf_output() -> None:
    crlf = fixture_text("verify_honest.txt").replace("\n", "\r\n")
    assert parse_verify(crlf) == parse_verify(fixture_text("verify_honest.txt"))


# --------------------------------------------------------------------------- parse_diagnose


def test_parse_diagnose_sample() -> None:
    parsed = parse_diagnose(fixture_text("diagnose_sample.txt"))
    assert parsed["completion"] == {"n": 302, "eps": 512, "pct": 59.0}
    assert parsed["by_obstacle_count"] == [
        {"label": "0 obstacles", "n": 140, "pct": 86.0, "completed": 120},
        {"label": "1 obstacle", "n": 180, "pct": 61.0, "completed": 110},
        {"label": "2+ obstacles", "n": 192, "pct": 38.0, "completed": 72},
    ]
    assert parsed["by_tightest_gate"][0] == {
        "label": "tight <0.8",
        "n": 0,
        "pct": None,
        "completed": None,
    }
    assert [r["label"] for r in parsed["by_gate_count"]] == ["<=4 gates", "5-7 gates", "8+ gates"]
    failures = parsed["failures"]
    assert failures["count"] == 210
    assert failures["causes"] == [
        {"label": "obstacle collision", "n": 150, "pct": 71.0},
        {"label": "crash: floor", "n": 30, "pct": 14.0},
        {"label": "crash: ceiling", "n": 5, "pct": 2.0},
        {"label": "crash: out-of-arena", "n": 10, "pct": 5.0},
        {"label": "timeout (ran out)", "n": 15, "pct": 7.0},
    ]
    assert failures["mean_death_tilt_deg"] == 47.0
    assert failures["progress_before_death_pct"] == 41.0
    assert failures["obstacle_deaths_on_multi_pct"] == 97.0


def test_parse_diagnose_handles_crlf_output() -> None:
    lf = fixture_text("diagnose_sample.txt")
    assert parse_diagnose(lf.replace("\n", "\r\n")) == parse_diagnose(lf)


@pytest.mark.parametrize("text", ["", "nonsense", None])
def test_parse_diagnose_never_raises(text) -> None:
    parsed = parse_diagnose(text)
    assert parsed["completion"] is None and parsed["failures"] is None
    assert parsed["by_obstacle_count"] == []


def test_parse_diagnose_with_zero_failures() -> None:
    text = "OVERALL completion: 512/512 (100%)\n\nFAILURE ANATOMY  (0 failures):\n"
    parsed = parse_diagnose(text)
    assert parsed["completion"]["pct"] == 100.0
    assert parsed["failures"]["count"] == 0 and parsed["failures"]["causes"] == []


# --------------------------------------------------------------------------- verdict: fixtures


@pytest.mark.parametrize(
    ("fixture", "flags", "verdict"),
    [
        ("verify_honest.txt", TERMINAL, "honest"),
        ("verify_barging.txt", NON_TERMINAL, "dishonest"),
        ("verify_barging.txt", TERMINAL, "dishonest"),  # completed-barging is never excused
        ("verify_barging_overall.txt", NON_TERMINAL, "suspect"),
        ("verify_barging_overall.txt", TERMINAL, "honest"),  # = collision-death rate, shown only
        ("verify_suspect_1pct.txt", TERMINAL, "suspect"),
        ("verify_lowtilt.txt", TERMINAL, "suspect"),
        ("verify_hovering.txt", TERMINAL, "dishonest"),  # hovering with 0 completions
        ("verify_none_completed.txt", TERMINAL, "nothing-completed"),
        ("verify_none_completed.txt", NON_TERMINAL, "suspect"),  # 12% overall barging
        ("verify_completed_line_missing.txt", TERMINAL, "unverified"),
        ("verify_truncated.txt", TERMINAL, "unverified"),
        ("verify_traceback.txt", TERMINAL, "unverified"),
    ],
)
def test_verdict_per_fixture(fixture: str, flags: list[str], verdict: str) -> None:
    result = _verdict(fixture, flags)
    assert result["verdict"] == verdict, result["reasons"]
    if verdict != "honest":
        assert result["reasons"]  # a non-honest verdict always says why


def test_hovering_reasons_mention_speed_and_tilt() -> None:
    reasons = " ".join(_verdict("verify_hovering.txt", TERMINAL)["reasons"])
    assert "hovering" in reasons and "low tilt" in reasons


# --------------------------------------------------------------------------- verdict: rule table


@pytest.mark.parametrize("missing", ["completion", "barging_pct", "mean_speed"])
def test_rule_a_missing_line_is_unverified_and_stops(missing: str) -> None:
    # Even an otherwise-damning hover signature is not evaluated past rule (a).
    parsed = _parsed(**{missing: None})
    if missing != "mean_speed":
        parsed["mean_speed"] = 0.01
    result = honesty_verdict(parsed, TERMINAL)
    assert result["verdict"] == "unverified"
    assert len(result["reasons"]) == 1 and "missing" in result["reasons"][0]


def test_rule_b_speed_threshold_is_strict() -> None:
    assert honesty_verdict(_parsed(mean_speed=0.5), TERMINAL)["verdict"] == "honest"
    assert honesty_verdict(_parsed(mean_speed=0.49), TERMINAL)["verdict"] == "dishonest"


def test_rule_b_tilt_threshold_is_strict() -> None:
    assert honesty_verdict(_parsed(mean_tilt_deg=5.0), TERMINAL)["verdict"] == "honest"
    assert honesty_verdict(_parsed(mean_tilt_deg=4.9), TERMINAL)["verdict"] == "suspect"


def test_rule_b_applies_regardless_of_completion() -> None:
    parsed = _parsed(completion={"n": 0, "eps": 512, "pct": 0.0}, completed_only=None)
    assert honesty_verdict({**parsed, "mean_speed": 0.1}, TERMINAL)["verdict"] == "dishonest"
    assert honesty_verdict({**parsed, "mean_tilt_deg": 2.0}, TERMINAL)["verdict"] == "suspect"


def test_rule_c_overall_barging_only_counts_without_obs_terminal() -> None:
    at = _parsed(barging_pct=2.0)
    above = _parsed(barging_pct=2.1)
    assert honesty_verdict(at, NON_TERMINAL)["verdict"] == "honest"
    assert honesty_verdict(above, NON_TERMINAL)["verdict"] == "suspect"
    assert honesty_verdict(above, TERMINAL)["verdict"] == "honest"
    assert honesty_verdict(_parsed(barging_pct=90.0), TERMINAL)["verdict"] == "honest"
    assert honesty_verdict(above, None)["verdict"] == "suspect"  # no flags → non-terminal


def test_rule_d_completed_barging_thresholds() -> None:
    def cb(pct: float) -> str:
        only = {"barged_pct": pct, "avg_inside_steps": 0.1, "mean_speed": 2.0}
        return honesty_verdict(_parsed(completed_only=only), TERMINAL)["verdict"]

    assert cb(0.0) == "honest"
    assert cb(0.5) == "suspect"
    assert cb(2.0) == "suspect"
    assert cb(2.1) == "dishonest"


def test_rule_d_completed_line_missing_is_unverified() -> None:
    assert honesty_verdict(_parsed(completed_only=None), TERMINAL)["verdict"] == "unverified"


def test_rule_e_nothing_completed_only_without_other_reasons() -> None:
    nothing = _parsed(completion={"n": 0, "eps": 512, "pct": 0.0}, completed_only=None)
    result = honesty_verdict(nothing, TERMINAL)
    assert result == {
        "verdict": "nothing-completed",
        "reasons": ["no episode completed the course"],
    }
    # With a (c) reason present, (e) does not fire.
    flagged = honesty_verdict({**nothing, "barging_pct": 10.0}, NON_TERMINAL)
    assert flagged["verdict"] == "suspect"
    assert all("no episode completed" not in r for r in flagged["reasons"])


def test_rule_f_honest_has_no_reasons() -> None:
    assert honesty_verdict(_parsed(), TERMINAL) == {"verdict": "honest", "reasons": []}


def test_reasons_accumulate_and_most_severe_wins() -> None:
    parsed = _parsed(
        mean_speed=0.3,  # (b) dishonest
        mean_tilt_deg=3.0,  # (b) suspect
        barging_pct=10.0,  # (c) suspect (non-terminal)
        completed_only={"barged_pct": 1.0, "avg_inside_steps": 0.0, "mean_speed": 0.3},  # (d)
    )
    result = honesty_verdict(parsed, NON_TERMINAL)
    assert result["verdict"] == "dishonest"
    assert len(result["reasons"]) == 4


def test_unverified_outranks_suspect() -> None:
    parsed = _parsed(mean_tilt_deg=1.0, completed_only=None)
    assert honesty_verdict(parsed, TERMINAL)["verdict"] == "unverified"


def test_severity_order_and_thresholds_are_named() -> None:
    assert audits.SEVERITY == ("dishonest", "unverified", "suspect", "nothing-completed", "honest")
    assert audits.THRESHOLDS == {
        "HOVER_SPEED_MS": 0.5,
        "HOVER_TILT_DEG": 5.0,
        "OVERALL_BARGE_NONTERMINAL_PCT": 2.0,
        "COMPLETED_BARGE_DISHONEST_PCT": 2.0,
    }


# --------------------------------------------------------------------------- runner


@pytest.fixture
def setup(tmp_path: Path):
    reg, fakes = make_registry(tmp_path)
    save_run(tmp_path, "run-a", flags=TERMINAL)
    touch(tmp_path / "training" / "run-a" / "model.pt")
    return tmp_path, reg, fakes


def _finish(fakes, code: int = 0) -> None:
    fakes["spawn"].popens[-1].exit(code)


def test_verify_audit_argv_env_seed_and_storage(setup) -> None:
    root, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("verify_honest.txt")
    result = audits.start_audit(reg, "run-a", "verify", ts="20261011-120000")
    call = fakes["spawn"].last
    ckpt = os.path.join(str(root), "training", "run-a", "model.pt")
    assert call["argv"] == [
        MICROMAMBA,
        "run",
        "-n",
        "dronefly",
        "python",
        os.path.join(str(root), "gpu_prototype", "verify_proc.py"),
        os.path.normpath(ckpt),
        "512",
    ]
    assert call["cwd"] == os.path.join(str(root), "gpu_prototype")
    env = call["env"]
    assert env["AUDIT_SEED"] == "1234"  # the inherited "99" is replaced
    assert {k for k in env if k.startswith("PROC_")} == set(TERMINAL)  # from run.json
    assert "PROC_TUNNELS" not in env
    audit_dir = os.path.join(str(root), ".dev-app", "audits", "run-a")
    assert call["log"] == os.path.join(audit_dir, "20261011-120000-verify.log")
    assert result["id"] == "20261011-120000-verify"
    assert result["state"] == "running" and result["verdict"] == "unverified"
    meta = json.loads(Path(audit_dir, "20261011-120000-verify.json").read_text(encoding="utf-8"))
    assert meta["flags"] == TERMINAL and meta["seed"] == 1234 and meta["eps"] == 512
    assert meta["ckpt"] == "training/run-a/model.pt" and meta["exit_code"] is None
    # Nothing is written into training/.
    assert sorted(os.listdir(root / "training" / "run-a")) == ["model.pt", "run.json"]


def test_finished_verify_bundles_completion_with_verdict_and_cheat_checks(setup) -> None:
    root, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("verify_barging.txt")
    audit_id = audits.start_audit(reg, "run-a", "verify")["id"]
    _finish(fakes)
    result = audits.audit_result(reg, "run-a", audit_id)
    assert result["state"] == "done" and result["exit_code"] == 0
    assert result["completion"] == {"n": 440, "eps": 512, "pct": 86.0}
    assert result["verdict"] == "dishonest"  # 86% completion is NOT a success
    assert result["reasons"]
    assert result["verify"]["barging_pct"] == 71.0
    assert result["verify"]["mean_speed"] == 2.88
    assert result["verify"]["completed_only"]["barged_pct"] == 64.0
    assert result["obs_terminal"] is True
    assert result["thresholds"] == audits.THRESHOLDS
    assert result["log_tail"][-1].startswith("  among COMPLETED runs")
    # The exit code is persisted to the sidecar.
    meta = json.loads(
        Path(audits.audits_dir(str(root), "run-a"), audit_id + ".json").read_text("utf-8")
    )
    assert meta["exit_code"] == 0


def test_nonzero_exit_with_traceback_is_failed_and_unverified(setup) -> None:
    _, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("verify_traceback.txt")
    audit_id = audits.start_audit(reg, "run-a", "verify")["id"]
    _finish(fakes, 1)
    result = audits.audit_result(reg, "run-a", audit_id)
    assert result["state"] == "failed"
    assert result["verdict"] == "unverified"
    assert result["reasons"] == ["audit process exited 1"]
    assert any("OutOfMemoryError" in line for line in result["log_tail"])
    # The partial parse is still available for the failed banner.
    assert result["completion"]["n"] == 302


def test_seed_eps_and_flags_overrides(setup) -> None:
    _, reg, fakes = setup
    audits.start_audit(reg, "run-a", "diagnose", flags=["PROC_GAP"], seed=7, eps="128")
    call = fakes["spawn"].last
    assert call["argv"][-3].endswith("diagnose_proc.py")
    assert call["argv"][-1] == "128"
    assert call["env"]["AUDIT_SEED"] == "7"
    assert {k for k in call["env"] if k.startswith("PROC_")} == {"PROC_GAP"}


def test_settings_supply_seed_and_eps_defaults(tmp_path: Path) -> None:
    reg, fakes = make_registry(tmp_path, settings={"audit_seed": 42, "audit_eps": 64})
    save_run(tmp_path, "run-a")
    touch(tmp_path / "training" / "run-a" / "model.pt")
    audits.start_audit(reg, "run-a", "verify")
    assert fakes["spawn"].last["argv"][-1] == "64"
    assert fakes["spawn"].last["env"]["AUDIT_SEED"] == "42"


def test_seed_zero_is_allowed(setup) -> None:
    _, reg, fakes = setup
    audits.start_audit(reg, "run-a", "verify", seed=0)
    assert fakes["spawn"].last["env"]["AUDIT_SEED"] == "0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "bogus"},
        {"eps": 0},
        {"eps": "many"},
        {"seed": -1},
        {"seed": True},
        {"flags": ["PROC_NOPE"]},
    ],
)
def test_bad_audit_requests_are_recipe_errors(setup, kwargs) -> None:
    _, reg, fakes = setup
    kind = kwargs.pop("kind", "verify")
    with pytest.raises(RecipeError):
        audits.start_audit(reg, "run-a", kind, **kwargs)
    assert fakes["spawn"].calls == []


def test_audit_refused_while_run_is_live(setup) -> None:
    _, reg, fakes = setup
    reg.launch("run-a", overwrite=True)
    with pytest.raises(RunError, match="running"):
        audits.start_audit(reg, "run-a", "verify")
    assert len(fakes["spawn"].calls) == 1  # only the launch


def test_audit_refused_without_checkpoint(tmp_path: Path) -> None:
    reg, fakes = make_registry(tmp_path)
    save_run(tmp_path, "run-a")
    with pytest.raises(RunError, match="no checkpoint"):
        audits.start_audit(reg, "run-a", "verify")
    assert fakes["spawn"].calls == []


def test_audit_unknown_run_is_not_found(setup) -> None:
    _, reg, _ = setup
    with pytest.raises(FileNotFoundError):
        audits.start_audit(reg, "ghost", "verify")


def test_committed_model_audit_requires_flags_and_writes_nothing_into_training(
    tmp_path: Path,
) -> None:
    reg, fakes = make_registry(tmp_path)
    readout = touch(tmp_path / "training" / "11-count-curriculum" / "model_readout.pt")
    before = sorted(os.listdir(tmp_path / "training" / "11-count-curriculum"))
    with pytest.raises(RecipeError, match="required"):
        audits.start_audit(reg, "11-count-curriculum", "verify")
    audits.start_audit(reg, "11-count-curriculum", "verify", flags=TERMINAL)
    call = fakes["spawn"].last
    assert call["argv"][-2] == readout
    assert call["log"].startswith(os.path.join(str(tmp_path), ".dev-app", "audits"))
    assert sorted(os.listdir(tmp_path / "training" / "11-count-curriculum")) == before


def test_committed_model_without_readout_is_refused(tmp_path: Path) -> None:
    reg, _ = make_registry(tmp_path)
    (tmp_path / "training" / "09-x").mkdir(parents=True)
    with pytest.raises(RunError, match="no model_readout.pt"):
        audits.start_audit(reg, "09-x", "verify", flags=TERMINAL)


def test_audit_ids_do_not_collide_within_the_same_second(setup) -> None:
    _, reg, _ = setup
    first = audits.start_audit(reg, "run-a", "verify", ts="20261011-1200")["id"]
    second = audits.start_audit(reg, "run-a", "verify", ts="20261011-1200")["id"]
    assert first != second


def test_results_survive_a_backend_restart(setup) -> None:
    root, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("verify_honest.txt")
    done_id = audits.start_audit(reg, "run-a", "verify", ts="20261011-1000")["id"]
    _finish(fakes)
    audits.audit_result(reg, "run-a", done_id)  # persists the exit code
    pending_id = audits.start_audit(reg, "run-a", "verify", ts="20261011-1100")["id"]

    fresh, _ = make_registry(root)
    done = audits.audit_result(fresh, "run-a", done_id)
    assert done["state"] == "done" and done["verdict"] == "honest"
    pending = audits.audit_result(fresh, "run-a", pending_id)
    assert pending["state"] == "interrupted" and pending["verdict"] == "unverified"


def test_unknown_audit_id_is_not_found(setup) -> None:
    _, reg, _ = setup
    with pytest.raises(FileNotFoundError):
        audits.audit_result(reg, "run-a", "nope")


def test_diagnose_is_unverified_until_a_verify_audit_finishes(setup) -> None:
    _, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("diagnose_sample.txt")
    diag_id = audits.start_audit(reg, "run-a", "diagnose", ts="20261011-0900")["id"]
    _finish(fakes)
    result = audits.audit_result(reg, "run-a", diag_id)
    assert result["state"] == "done"
    assert result["verdict"] == "unverified" and result["verdict_source"] is None
    assert result["completion"]["pct"] == 59.0  # completion still bundled with the verdict
    assert result["diagnose"]["failures"]["count"] == 210
    assert result["verify"] is None

    fakes["spawn"].next_output = fixture_text("verify_lowtilt.txt")
    verify_id = audits.start_audit(reg, "run-a", "verify", ts="20261011-1000")["id"]
    _finish(fakes)
    result = audits.audit_result(reg, "run-a", diag_id)
    assert result["verdict"] == "suspect" and result["verdict_source"] == verify_id


def test_diagnose_ignores_failed_verify_audits(setup) -> None:
    _, reg, fakes = setup
    fakes["spawn"].next_output = fixture_text("verify_honest.txt")
    audits.start_audit(reg, "run-a", "verify", ts="20261011-0800")
    _finish(fakes, 2)
    fakes["spawn"].next_output = fixture_text("diagnose_sample.txt")
    diag_id = audits.start_audit(reg, "run-a", "diagnose", ts="20261011-0900")["id"]
    _finish(fakes)
    assert audits.audit_result(reg, "run-a", diag_id)["verdict_source"] is None


def test_list_audits_newest_first_with_verdicts(setup) -> None:
    _, reg, fakes = setup
    for ts, fixture in (("20261011-0800", "verify_hovering.txt"), ("20261011-0900", None)):
        if fixture:
            fakes["spawn"].next_output = fixture_text(fixture)
        audits.start_audit(reg, "run-a", "verify" if fixture else "diagnose", ts=ts)
        _finish(fakes)
    listed = audits.list_audits(reg, "run-a")
    assert [a["id"] for a in listed] == ["20261011-0900-diagnose", "20261011-0800-verify"]
    assert [a["verdict"] for a in listed] == ["dishonest", "dishonest"]
    assert listed[0]["verdict_source"] == "20261011-0800-verify"
    for entry in listed:
        assert {"completion", "verdict", "reasons", "verify", "diagnose"} <= set(entry)


def test_list_audits_empty_and_invalid_name(setup) -> None:
    _, reg, _ = setup
    assert audits.list_audits(reg, "run-a") == []
    with pytest.raises(RecipeError):
        audits.list_audits(reg, "../x")
