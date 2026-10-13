"""UC68 AC7 (D9): ``drone-fly train eval <run> --tracks <selector>``, the per-track report.

* On a tiny CPU teacher run: ``eval/<split>.json`` + ``<split>.md``; per mode (randomized =
  the counting mode, nominal) the macro completion over the selection's LIFTOFF tracks only,
  the scene strata, and per track: completion, gates passed (count / fraction), per-passage and
  per-lap success, lap / race times, outcome shares, stats, flags and scene status.
* Deterministic: the same seed gives the same report.
* ``track_summary`` maps outcomes / crossings / times exactly (synthetic results).
* Refused: an unknown observation schema, another conversion version, a changed conversion
  fingerprint, a non-mlp-teacher actor, a checkpoint without a sidecar.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly import drones
from drone_fly.cli.main import main
from drone_fly.sim.multi_course import OUTCOMES
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.training import evaluate
from drone_fly.training.config import load_teacher_config
from tests.tracks_support import build_library, use_library, write_teacher_config

TRACK_KEYS = {"id", "map", "split", "scene", "flags", "episodes", "completion",
              "gates_passed_mean", "gates_fraction_mean", "total_crossings",
              "per_passage_success", "per_lap_success", "lap_time_s", "full_time_s", "outcomes",
              "unfinished", "stats"}  # fmt: skip


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def run(lib, tmp_path_factory):
    from drone_fly.training.ppo import train_teacher

    base = tmp_path_factory.mktemp("evalrun")
    with pytest.MonkeyPatch.context() as mp:
        use_library(mp, lib)
        cfg = load_teacher_config(write_teacher_config(lib, base / "t.yaml"))
        res = train_teacher(cfg, drones.load(cfg.drone), run_root=base / "runs", run_id="e",
                            max_iterations=1)  # fmt: skip
    return res.run.path


@pytest.fixture
def library(lib, monkeypatch):
    use_library(monkeypatch, lib)
    return lib


def _eval(run, sel=("liftoff:train",), **kw):
    kw.setdefault("episodes", 3)
    kw.setdefault("device", "cpu")
    return evaluate.evaluate_teacher(run, list(sel), **kw)


def test_report_shape_and_files(run, library):
    res = _eval(run)
    assert set(res["modes"]) == {"randomized", "nominal"} and res["counting_mode"] == "randomized"
    assert res["obs_schema"] == "teacher-obs v1" and res["episodes"] == 3
    for mode, m in res["modes"].items():
        assert m["tracks"] == 4  # Liftoff train tracks; procedural courses are excluded
        assert set(m["strata"]) == {"scene", "no-scene (optimistic: no walls or buildings)",
                                    "flagged"}  # fmt: skip
        assert m["strata"]["scene"]["tracks"] == 4  # arena ok + desert stale-build
        assert m["strata"]["flagged"]["tracks"] == 2  # desert: stale-build, no-terrain
        ids = [t["id"] for t in m["per_track"]]
        assert sorted(ids) == ["fake-arena-01", "fake-arena-02", "fake-desert-01",
                               "fake-desert-02-v2"]  # fmt: skip
        for t in m["per_track"]:
            assert set(t) == TRACK_KEYS
            assert t["episodes"] == 3 and 0.0 <= t["completion"] <= 1.0
            assert sum(t["outcomes"].values()) == pytest.approx(1.0)
            assert set(t["outcomes"]) == set(OUTCOMES) - {"none"}
            assert t["stats"]["passages"] is not None
        assert m["macro_completion"] == pytest.approx(np.mean([t["completion"]
                                                               for t in m["per_track"]]))  # fmt: skip
    d = run / "eval"
    data = json.loads((d / "train.json").read_text("utf-8"))
    assert data["modes"]["randomized"]["tracks"] == 4
    md = (d / "train.md").read_text("utf-8")
    assert "# Teacher evaluation: train" in md and "## randomized" in md and "## nominal" in md
    assert "fake-desert-02-v2" in md and "counting mode: **randomized**" in md


def test_held_out_report_and_unconvertible_entries(run, library):
    res = _eval(run, ("liftoff:held-out",), modes=("nominal",))
    assert [t["id"] for t in res["modes"]["nominal"]["per_track"]] == ["fake-hall-02"]
    t = res["modes"]["nominal"]["per_track"][0]
    assert t["scene"] == "no-scene" and "no-scene" in t["flags"]
    assert (run / "eval" / "held-out.json").is_file()


def test_evaluation_is_deterministic(run, library):
    a = _eval(run, modes=("randomized",), write=False)
    b = _eval(run, modes=("randomized",), write=False)
    strip = lambda r: [{k: v for k, v in t.items()} for t in r["modes"]["randomized"]["per_track"]]  # noqa: E731
    assert strip(a) == strip(b)


def test_cli_teacher_eval(run, library, capsys):
    rc = main(["train", "eval", str(run), "--tracks", "liftoff:val", "--episodes", "2", "--mode",
               "nominal", "--device", "cpu"])  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    assert "nominal: macro completion" in out and "report:" in out
    assert json.loads((run / "eval" / "val.json").read_text("utf-8"))["episodes"] == 2


# ----------------------------------------------------------------------------- outcome mapping


def test_track_summary_maps_outcomes_crossings_and_times():
    course = SimpleNamespace(n_passages=2, circuit=True, total_crossings=5, id="c",
                             meta={"map": "M", "split": "train", "scene": "ok", "flags": []})  # fmt: skip
    nan = np.nan
    res = {
        "k": np.array([5, 3, 0, 1]),
        "outcome": np.array([OUTCOMES.index(o) for o in
                             ("completed", "collision", "floor", "timeout")]),  # fmt: skip
        "t_end": np.zeros(4),
        "cross_t": np.array([[1.0, 2.0, 3.0, 4.5, 5.0], [1.0, 2.0, 3.0, nan, nan],
                             [nan] * 5, [0.5, nan, nan, nan, nan]]),  # fmt: skip
        "unfinished": np.array([False, False, False, True]),
    }
    s = evaluate.track_summary(course, res, np.arange(4), {"stats": {"passages": 2, "laps": 2}})
    assert s["completion"] == 0.25 and s["gates_passed_mean"] == pytest.approx(9 / 4)
    assert s["gates_fraction_mean"] == pytest.approx(np.mean([1, 0.6, 0, 0.2]))
    assert s["per_passage_success"] == pytest.approx(9 / (9 + 3))
    # laps done: (k - 1) // 2 -> 2, 1, 0, 0
    assert s["per_lap_success"] == pytest.approx(3 / (3 + 3))
    assert s["lap_time_s"] == pytest.approx(np.mean([2.0, 2.0, 2.0]))  # ct[2]-ct[0], ct[4]-ct[2]
    assert s["full_time_s"] == pytest.approx(4.0)
    assert s["outcomes"] == {"floor": 0.25, "collision": 0.25, "out-of-bounds": 0.0,
                             "timeout": 0.25, "completed": 0.25}  # fmt: skip
    assert s["unfinished"] == 1 and s["stats"]["passages"] == 2


def test_strata():
    tracks = [{"scene": "ok", "flags": [], "completion": 1.0},
              {"scene": "stale-build", "flags": ["stale-build"], "completion": 0.5},
              {"scene": "no-scene", "flags": ["no-scene"], "completion": 0.0}]  # fmt: skip
    s = evaluate.strata(tracks)
    assert s["scene"] == {"tracks": 2, "completion": 0.75}
    assert s["no-scene (optimistic: no walls or buildings)"] == {"tracks": 1, "completion": 0.0}
    assert s["flagged"] == {"tracks": 1, "completion": 0.5}
    assert evaluate.strata([])["scene"] == {"tracks": 0, "completion": None}


# ----------------------------------------------------------------------------- refusals


@pytest.fixture
def ckpt_copy(run, tmp_path):
    d = tmp_path / "r" / "checkpoints"
    d.mkdir(parents=True)
    shutil.copy(run / "checkpoints" / "teacher.pt", d / "teacher.pt")
    shutil.copy(run / "checkpoints" / "teacher.contract.json", d / "teacher.contract.json")
    return d / "teacher.pt"


def _edit(ck, fn):
    p = ck.with_name("teacher.contract.json")
    side = json.loads(p.read_text("utf-8"))
    fn(side)
    p.write_text(json.dumps(side), "utf-8")


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda s: s["obs_schema"].update(version=2), "unknown observation schema"),
        (lambda s: s["obs_schema"].update(name="other"), "unknown observation schema"),
        (lambda s: s["obs_schema"].update(dim=150), "unknown observation schema"),
        (lambda s: s.pop("obs_schema"), "unknown observation schema"),
        (lambda s: s["course_conversion"].update(version=1), "course conversion v1"),
        (lambda s: s.update(actor="k1-reservoir-aug"), "only mlp-teacher"),
    ],
)
def test_unknown_schema_or_conversion_is_refused(ckpt_copy, edit, match):
    _edit(ckpt_copy, edit)
    with pytest.raises(TrackDataError, match=match):
        evaluate.load_teacher(ckpt_copy, "cpu")


def test_changed_conversion_fingerprint_is_refused(ckpt_copy):
    _edit(ckpt_copy, lambda s: s["course_conversion"]["parameters"].update(corridor_m=30.0))
    _, side = evaluate.load_teacher(ckpt_copy, "cpu")
    with pytest.raises(TrackDataError, match="differs from this build"):
        evaluate.conversion_from_sidecar(side)


def test_checkpoint_without_sidecar_is_refused(ckpt_copy):
    ckpt_copy.with_name("teacher.contract.json").unlink()
    with pytest.raises(TrackDataError, match="no sidecar"):
        evaluate.load_teacher(ckpt_copy, "cpu")


def test_resolve_run():
    from pathlib import Path

    ck = Path("x") / "runs" / "r" / "checkpoints" / "teacher_it3.pt"
    assert evaluate.resolve_run(ck) == (Path("x") / "runs" / "r", ck)
