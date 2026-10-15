"""UC77 AC1 / AC2 / AC5 / AC6 (b) (c): the ``tracks.obstacles`` config flag and the teacher side.

* Config: ``tracks.obstacles`` is validated (``TrainConfigError``); ``student-liftoff-open`` is a
  packaged, strictly-loading config that differs from ``student-liftoff-run2b`` in exactly its
  name + ``tracks.obstacles`` + ``distill.track_evals`` + ``distill.final_report``; run2 / run2b
  are untouched.
* Snapshots (N4 / N6): default configs never write ``tracks.obstacles`` / ``distill.track_evals`` /
  ``distill.final_report``; the open config writes and round-trips them; an old snapshot without
  the keys loads as the defaults.
* Teacher: ``build_teacher_pool`` forwards the mode (augmented copies included); ``train eval``
  (``evaluate_teacher``) records the mode, uses it for its pool, and suffixes the report files only
  when the mode is overridden; the CLI forwards ``--obstacles``.

CPU, hermetic (the UC68 fixture library; a 1-iteration teacher run).
"""

from __future__ import annotations

import json

import pytest
import yaml

from drone_fly import drones
from drone_fly.training import evaluate
from drone_fly.training.cli import student_raw_from_snapshot, teacher_raw_from_snapshot
from drone_fly.training.config import (
    TrainConfigError,
    load_raw,
    load_teacher_config,
    packaged_names,
    student_from_mapping,
    teacher_from_mapping,
)
from tests.tracks_support import build_library, use_library, write_teacher_config

UC77_KEYS = {"tracks.obstacles", "distill.track_evals", "distill.final_report"}


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _student(name: str):
    return student_from_mapping(load_raw(name), validate_tracks=False)


# ----------------------------------------------------------------------------- config (c)


def test_open_config_is_packaged_and_loads_strictly():
    assert "student-liftoff-open" in packaged_names()
    cfg = _student("student-liftoff-open")
    assert cfg.name == "student-liftoff-open"
    assert cfg.base.tracks.obstacles == "none"
    assert cfg.distill.track_evals == ("liftoff:liftoff-arena-04",)
    assert isinstance(cfg.distill.track_evals, tuple)
    assert cfg.distill.final_report is True


def test_open_config_overrides_exactly_the_uc77_keys_over_run2b():
    run2b, open_ = load_raw("student-liftoff-run2b"), load_raw("student-liftoff-open")
    fb, fo = _flatten(run2b), _flatten(open_)
    fb.pop("extends", None)
    fo.pop("extends", None)
    assert set(fo) - set(fb) == UC77_KEYS and not set(fb) - set(fo)
    assert {k for k in fb if fb[k] != fo[k]} == {"name"}
    assert fo["tracks.obstacles"] == "none"
    assert fo["distill.track_evals"] == ["liftoff:liftoff-arena-04"]
    assert fo["distill.final_report"] is True


@pytest.mark.parametrize(
    "name",
    [
        "student-liftoff-run2",
        "student-liftoff-run2b",
        "student-liftoff",
        "teacher-liftoff",
        "teacher-liftoff-aug",
    ],
)
def test_existing_configs_keep_the_defaults(name):
    flat = _flatten(load_raw(name))
    assert not UC77_KEYS & set(flat), name
    if name.startswith("student"):
        cfg = _student(name)
        assert cfg.base.tracks.obstacles == "all"
        assert cfg.distill.track_evals == () and cfg.distill.final_report is False
    else:
        assert load_teacher_config(name, validate_tracks=False).tracks.obstacles == "all"


@pytest.mark.parametrize("bad", ["some", "None", "", "ALL", None, 0])
def test_a_bad_obstacles_mode_raises_a_config_error(bad):
    raw = load_raw("student-liftoff-open")
    raw["tracks"] = dict(raw["tracks"], obstacles=bad)
    with pytest.raises(TrainConfigError, match="obstacles"):
        student_from_mapping(raw, validate_tracks=False)
    traw = load_raw("teacher-liftoff")
    traw["tracks"] = dict(traw["tracks"], obstacles=bad)
    with pytest.raises(TrainConfigError, match="obstacles"):
        teacher_from_mapping(traw, validate_tracks=False)


def test_unknown_distill_keys_are_still_rejected():
    raw = load_raw("student-liftoff-open")
    raw["distill"] = dict(raw["distill"], track_eval=["liftoff:val"])
    with pytest.raises(TrainConfigError, match="distill"):
        student_from_mapping(raw, validate_tracks=False)


# ----------------------------------------------------------------------------- snapshots (b)


@pytest.mark.parametrize(
    "name", ["student-liftoff-run2b", "student-liftoff-run2", "student-liftoff"]
)
def test_default_student_snapshot_omits_the_uc77_keys(name):
    snap = _student(name).snapshot()
    assert "obstacles" not in snap["tracks"]
    assert "track_evals" not in snap["distill"] and "final_report" not in snap["distill"]


@pytest.mark.parametrize("name", ["teacher-liftoff", "teacher-liftoff-aug"])
def test_default_teacher_snapshot_omits_tracks_obstacles(name):
    snap = load_teacher_config(name, validate_tracks=False).snapshot()
    assert "obstacles" not in snap["tracks"]
    again = teacher_from_mapping(teacher_raw_from_snapshot(yaml.safe_load(yaml.safe_dump(snap))),
                                 validate_tracks=False)  # fmt: skip
    assert again.tracks.obstacles == "all"


def test_open_snapshot_keeps_and_round_trips_the_uc77_keys():
    cfg = _student("student-liftoff-open")
    snap = yaml.safe_load(yaml.safe_dump(cfg.snapshot(), sort_keys=False))
    assert snap["tracks"]["obstacles"] == "none"
    assert snap["distill"]["track_evals"] == ["liftoff:liftoff-arena-04"]
    assert snap["distill"]["final_report"] is True
    again = student_from_mapping(student_raw_from_snapshot(snap), validate_tracks=False)
    assert again.base.tracks.obstacles == "none"
    assert again.distill.track_evals == ("liftoff:liftoff-arena-04",)
    assert again.distill.final_report is True
    assert again.snapshot() == cfg.snapshot()


def test_partial_uc77_overrides_snapshot_only_the_non_defaults():
    raw = load_raw("student-liftoff-run2b")
    raw["distill"] = dict(raw["distill"], final_report=True)
    snap = student_from_mapping(raw, validate_tracks=False).snapshot()
    assert snap["distill"]["final_report"] is True
    assert "track_evals" not in snap["distill"] and "obstacles" not in snap["tracks"]


def test_old_snapshots_without_the_keys_load_as_the_defaults():
    snap = _student("student-liftoff-run2b").snapshot()
    snap["tracks"].pop("obstacles", None)
    cfg = student_from_mapping(student_raw_from_snapshot(snap), validate_tracks=False)
    assert cfg.base.tracks.obstacles == "all"
    assert cfg.distill.track_evals == () and cfg.distill.final_report is False


def test_teacher_snapshot_with_obstacle_free_tracks_round_trips():
    raw = load_raw("teacher-liftoff")
    raw["tracks"] = dict(raw["tracks"], obstacles="none")
    snap = teacher_from_mapping(raw, validate_tracks=False).snapshot()
    assert snap["tracks"]["obstacles"] == "none"
    again = teacher_from_mapping(teacher_raw_from_snapshot(snap), validate_tracks=False)
    assert again.tracks.obstacles == "none"


def test_obstacles_mode_is_not_a_conversion_parameter():
    """The student enforces the teacher sidecar's conversion fingerprint: the mode stays out."""
    from drone_fly.training.ppo import teacher_conversion

    a = load_teacher_config("teacher-liftoff", validate_tracks=False)
    raw = load_raw("teacher-liftoff")
    raw["tracks"] = dict(raw["tracks"], obstacles="none")
    b = teacher_from_mapping(raw, validate_tracks=False)
    assert "obstacles" not in a.tracks.conversion_dict()
    assert teacher_conversion(a).fingerprint() == teacher_conversion(b).fingerprint()


# ----------------------------------------------------------------------------- teacher side


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture
def library(lib, monkeypatch):
    use_library(monkeypatch, lib)
    return lib


def _open_tracks(lib, **extra):
    return {
        "mix": [{"select": "liftoff:train", "weight": 0.6, "scene_weight": 2.0},
                {"select": "procedural:proc-test", "weight": 0.4}],
        "root": str(lib.root),
        "scenes_root": str(lib.scenes_root),
        "conversion": {"time_base_s": 0.05, "time_max_s": 0.1},
        "obstacles": "none",
        **extra,
    }  # fmt: skip


def test_build_teacher_pool_is_obstacle_free_with_augmentation(library, tmp_path):
    from drone_fly.training.ppo import build_teacher_pool

    path = write_teacher_config(library, tmp_path / "t.yaml", tracks=_open_tracks(
        library, augment={"mirror": True, "scales": [1.0, 1.2]}))  # fmt: skip
    cfg = load_teacher_config(path)
    pool = build_teacher_pool(cfg)
    assert any("augment" in c.meta for c in pool.courses)
    for c in pool.courses:
        assert len(c.harness) == 0 and "obstacle-free" in c.meta["flags"], c.id
        if c.meta["source"] == "procedural":
            assert len(c.prims) == 0
    default = build_teacher_pool(load_teacher_config(write_teacher_config(library, tmp_path /
                                                                          "d.yaml")))  # fmt: skip
    assert any(len(c.harness) for c in default.courses)
    assert not any("obstacle-free" in c.meta["flags"] for c in default.courses)


@pytest.fixture(scope="module")
def open_run(lib, tmp_path_factory):
    from drone_fly.training.ppo import train_teacher

    base = tmp_path_factory.mktemp("open-teacher")
    with pytest.MonkeyPatch.context() as mp:
        use_library(mp, lib)
        cfg = load_teacher_config(write_teacher_config(lib, base / "t.yaml",
                                                       tracks=_open_tracks(lib)))  # fmt: skip
        res = train_teacher(cfg, drones.load(cfg.drone), run_root=base / "runs", run_id="open",
                            max_iterations=1)  # fmt: skip
    return res.run.path


def _spy_build_pool(monkeypatch):
    from drone_fly.tracks import loader

    seen = []
    real = loader.build_pool

    def spy(*a, **kw):
        pool = real(*a, **kw)
        seen.append((kw.get("obstacles", "all"), pool))
        return pool

    monkeypatch.setattr(loader, "build_pool", spy)
    return seen


def test_teacher_run_snapshot_records_the_mode(open_run):
    snap = yaml.safe_load((open_run / "config.yaml").read_text("utf-8"))
    assert snap["tracks"]["obstacles"] == "none"


def test_teacher_eval_uses_and_records_the_run_mode(open_run, library, monkeypatch):
    seen = _spy_build_pool(monkeypatch)
    res = evaluate.evaluate_teacher(open_run, ["liftoff:train"], episodes=1, device="cpu",
                                    modes=("randomized",))  # fmt: skip
    assert res["obstacles"] == "none"
    assert [m for m, _ in seen] == ["none"]
    assert all(len(c.harness) == 0 for c in seen[0][1].courses)
    d = open_run / "eval"
    data = json.loads((d / "train.json").read_text("utf-8"))  # the run's own mode: no suffix
    assert data["obstacles"] == "none"
    assert "course obstacles `none`" in (d / "train.md").read_text("utf-8")


def test_teacher_eval_override_is_recorded_and_suffixed(open_run, library, monkeypatch):
    seen = _spy_build_pool(monkeypatch)
    res = evaluate.evaluate_teacher(open_run, ["liftoff:train"], episodes=1, device="cpu",
                                    modes=("randomized",), obstacles="all")  # fmt: skip
    assert res["obstacles"] == "all" and [m for m, _ in seen] == ["all"]
    assert any(len(c.harness) for c in seen[0][1].courses)
    d = open_run / "eval"
    data = json.loads((d / "train-obstacles-all.json").read_text("utf-8"))
    assert data["obstacles"] == "all"
    assert "course obstacles `all`" in (d / "train-obstacles-all.md").read_text("utf-8")


def test_teacher_eval_write_false_writes_nothing(open_run, library):
    before = sorted(p.name for p in (open_run / "eval").glob("*"))
    res = evaluate.evaluate_teacher(open_run, ["liftoff:train"], episodes=1, device="cpu",
                                    modes=("randomized",), obstacles="all",
                                    write=False)  # fmt: skip
    assert res["obstacles"] == "all"
    assert sorted(p.name for p in (open_run / "eval").glob("*")) == before


@pytest.mark.parametrize(
    ("argv", "want"),
    [([], None), (["--obstacles", "none"], "none"), (["--obstacles", "all"], "all")],
)
def test_cli_teacher_eval_forwards_obstacles(tmp_path, monkeypatch, argv, want):
    from drone_fly.cli import main as cli

    seen = {}

    def fake(run, selectors, **kw):
        seen.update(kw, selectors=selectors)
        return {"modes": {}, "written": None}

    monkeypatch.setattr(evaluate, "evaluate_teacher", fake)
    assert cli.main(["train", "eval", str(tmp_path), "--tracks", "liftoff:liftoff-arena-04",
                     *argv]) == 0  # fmt: skip
    assert seen["obstacles"] == want and seen["selectors"] == ["liftoff:liftoff-arena-04"]


def test_cli_rejects_an_unknown_obstacles_mode(tmp_path):
    from drone_fly.cli import main as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["train", "eval", str(tmp_path), "--obstacles", "some"])
    assert exc.value.code == 2
