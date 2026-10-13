"""UC68 AC9: the trainer never loads ``split: val`` or ``held-out`` entries.

* A spy on ``tracks.loader.load_course`` sees only TRAIN ids during a real (tiny) teacher run
  and during the pool build.
* Every selector that reaches a val / held-out entry (split, id, ``map=``, ``all``) raises
  ``TrackLeakageError`` for ``purpose="train"`` (never a silent filter): at config validation,
  at the pool build, from the CLI (no run folder is created) and on ``--resume`` of a run whose
  config was edited to point at val.
* The run snapshot (``tracks/files.sha256``, ``courses.json``) lists train files only.
* Val selection (``--select-val``) evaluates on ``liftoff:val`` only, never held-out.
"""

from __future__ import annotations

import json

import pytest
import yaml

from drone_fly import drones
from drone_fly.cli.main import main
from drone_fly.tracks import catalogue as cat
from drone_fly.tracks import loader
from drone_fly.training.config import load_raw, load_teacher_config, teacher_from_mapping
from tests.tracks_support import TRAIN_IDS, build_library, use_library, write_teacher_config

LEAKY = ["liftoff:val", "liftoff:held-out", "liftoff:all", "liftoff:map=FakeHall",
         "liftoff:fake-field-01", "liftoff:fake-hall-02"]  # fmt: skip


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture
def spy(monkeypatch):
    seen = []
    real = loader.load_course

    def wrapped(entry, **kw):
        seen.append((entry["id"], entry["split"]))
        return real(entry, **kw)

    monkeypatch.setattr(loader, "load_course", wrapped)
    return seen


def test_training_run_loads_train_entries_only(lib, tmp_path, monkeypatch, spy):
    from drone_fly.training.ppo import train_teacher

    use_library(monkeypatch, lib)
    cfg = load_teacher_config(write_teacher_config(lib, tmp_path / "t.yaml"))
    res = train_teacher(cfg, drones.load(cfg.drone), run_root=tmp_path / "runs", run_id="leak",
                        max_iterations=1)  # fmt: skip
    assert spy and {s for _, s in spy} == {"train"}
    assert {i for i, _ in spy} == TRAIN_IDS | {"proc-test"}
    tr = res.run.path / "tracks"
    files = (tr / "files.sha256").read_text("utf-8")
    assert "FakeField" not in files and "FakeHall" not in files
    assert "FakeArenaTrack01_0001.xml" in files and "FakeDesertRace02_0002.xml" in files
    assert {c["split"] for c in json.loads((tr / "courses.json").read_text("utf-8"))} == {"train"}


@pytest.mark.parametrize("selector", LEAKY)
def test_select_refuses_non_train_entries_for_training(lib, selector):
    with pytest.raises(cat.TrackLeakageError):
        cat.select(lib.liftoff, selector, "train")
    cat.select(lib.liftoff, selector, "eval")  # evaluation may read them


@pytest.mark.parametrize("selector", LEAKY)
def test_pool_build_refuses_before_loading(lib, selector, spy):
    with pytest.raises(cat.TrackLeakageError):
        loader.build_pool([{"select": "liftoff:train"}, {"select": selector}], "train",
                          catalogues={"liftoff": lib.liftoff}, root=lib.root,
                          scenes_root=lib.scenes_root, use_cache=False)  # fmt: skip
    assert all(s == "train" for _, s in spy)


@pytest.mark.parametrize("selector", LEAKY)
def test_config_validation_refuses_non_train_selectors(lib, selector, monkeypatch, tmp_path):
    use_library(monkeypatch, lib)
    raw = load_raw(write_teacher_config(lib, tmp_path / "t.yaml"))
    raw["tracks"]["mix"] = [{"select": selector, "weight": 1.0}]
    with pytest.raises(cat.TrackLeakageError):
        teacher_from_mapping(raw)


@pytest.mark.parametrize("selector", ["liftoff:held-out", "liftoff:val"])
def test_cli_refuses_and_creates_no_run(lib, selector, monkeypatch, tmp_path, capsys, spy):
    use_library(monkeypatch, lib)
    p = write_teacher_config(lib, tmp_path / "t.yaml")
    rc = main(["train", "teacher", "--config", str(p), "--tracks", selector, "--max-iterations",
               "1", "--output-root", str(tmp_path / "runs")])  # fmt: skip
    assert rc != 0
    assert "training may only load split: train" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists() and not spy


def test_resume_of_an_edited_config_is_refused(lib, tmp_path, monkeypatch, spy):
    from drone_fly.training.ppo import train_teacher

    use_library(monkeypatch, lib)
    cfg = load_teacher_config(write_teacher_config(lib, tmp_path / "t.yaml"))
    res = train_teacher(cfg, drones.load(cfg.drone), run_root=tmp_path / "runs", run_id="r",
                        max_iterations=1)  # fmt: skip
    p = res.run.path / "config.yaml"
    snap = yaml.safe_load(p.read_text("utf-8"))
    snap["tracks"]["mix"] = [{"select": "liftoff:val", "weight": 1.0}]
    p.write_text(yaml.safe_dump(snap), "utf-8")
    spy.clear()
    rc = main(["train", "teacher", "--resume", str(res.run.path), "--max-iterations", "2"])
    assert rc != 0 and not spy


def test_val_selection_never_touches_held_out(tmp_path, monkeypatch):
    from drone_fly.training import evaluate

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for name in ("teacher_it10.pt", "teacher_it2.pt", "teacher.pt"):
        (run / "checkpoints" / name).write_bytes(b"")
    calls = []

    def fake(path, selectors, **kw):
        calls.append((list(selectors), kw["checkpoint"].name, kw["seed"], kw["modes"]))
        score = {"teacher_it2.pt": 0.4, "teacher_it10.pt": 0.7, "teacher.pt": 0.6}
        return {"modes": {"randomized": {"macro_completion": score[kw["checkpoint"].name]}}}

    monkeypatch.setattr(evaluate, "evaluate_teacher", fake)
    sel = evaluate.select_val(run, episodes=3)
    assert {tuple(c[0]) for c in calls} == {("liftoff:val",)}
    assert [c[1] for c in calls] == ["teacher_it2.pt", "teacher_it10.pt", "teacher.pt"]
    assert {c[2] for c in calls} == {evaluate.VAL_SEED} and {c[3] for c in calls} == {
        ("randomized",)}  # fmt: skip
    assert sel["checkpoint"] == "teacher_it10.pt" and sel["val_macro_completion"] == 0.7
    saved = json.loads((run / "eval" / evaluate.SELECTION_FILE).read_text("utf-8"))
    assert saved["checkpoint"] == "teacher_it10.pt"
    assert evaluate.resolve_run(run) == (run, run / "checkpoints" / "teacher_it10.pt")
