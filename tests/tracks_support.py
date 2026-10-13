"""Shared helpers for the UC68 track-library / multi-course / teacher tests.

* :data:`FIXTURE_DIR` is the synthetic trackdata tree under ``tests/data/tracks/`` (see its
  ``generate_fixtures.py``): four fake maps, eight track/race pairs, ``index.json``,
  ``props.json`` (build 1000) and three scene caches. No game data.
* :func:`build_library` copies it to a temp folder and runs the real ``tracks scan`` over it
  with the map splits pinned by a pre-existing catalogue (:data:`SPLITS`: FakeArena + FakeDesert
  train, FakeField val, FakeHall held-out), plus a tiny procedural catalogue. The fixture scene
  caches hold a handful of colliders, so the analyzer's 500-collider identification floor
  (``liftoff_extract.scene.MIN_COLLIDERS``) is lowered to 1 while scanning; every other
  validation rule is the real one.
* :func:`use_library` points ``tracks.catalogue.catalogue_dir`` (``load_named``) and the loader's
  conversion cache at the temp library, so nothing reads the committed ``tracks/*.yaml`` or
  writes under ``artifacts/``.
* :func:`make_env` builds a CPU :class:`~drone_fly.sim.multi_course.MultiCourseEnv` over given
  courses (no randomization, fixed 50 Hz timing unless asked otherwise).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "data" / "tracks"
SPLITS = {"FakeArena": "train", "FakeDesert": "train", "FakeField": "val", "FakeHall": "held-out"}
TRAIN_IDS = {"fake-arena-01", "fake-arena-02", "fake-desert-01", "fake-desert-02-v2"}
PROC_SPEC = {
    "seed_base": 5000,
    "pool_size": 4,
    "ng": [3, 5],
    "laps": [1, 3],
    "tier": "tractable",
    "tunnels": False,
    "variety": True,
    "no_obs": False,
    "scale": [2.0, 3.0],
    "aperture_m": [1.5, 3.0],
}


@dataclass
class Library:
    base: Path
    root: Path  # trackdata (XMLs, index.json, props.json)
    scenes_root: Path
    tracks_dir: Path  # catalogues (liftoff.yaml, procedural.yaml)
    cache_dir: Path
    liftoff: Any  # Catalogue
    procedural: Any  # Catalogue


def copy_fixtures(dst: Path) -> Path:
    shutil.copytree(FIXTURE_DIR, dst, ignore=shutil.ignore_patterns("*.py", "__pycache__"))
    return dst


def write_split_header(path: Path, splits: dict[str, str], seed: int = 68) -> Path:
    """A header-only catalogue that pins each map's split (what a re-scan preserves)."""
    from drone_fly.tracks import catalogue as cat

    header = {"schema_version": 1, "source": "liftoff-xml", "split": {"seed": seed},
              "maps": {m: {"split": s} for m, s in splits.items()}}  # fmt: skip
    return cat.dump_catalogue(cat.Catalogue(path, header, []), path)


def scan(root: Path, out: Path, *, splits: dict[str, str] | None = SPLITS, **kw: Any) -> Any:
    """``scan_liftoff`` over a fixture tree (the identification floor lowered, see above)."""
    from drone_fly.tracks.liftoff_extract import scene as ex
    from drone_fly.tracks.scan import scan_liftoff

    if splits is not None and not out.is_file():
        write_split_header(out, splits)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ex, "MIN_COLLIDERS", 1)
        return scan_liftoff(root=root, scenes_root=root / "scenes", out_path=out,
                            log_fn=lambda *a: None, **kw)  # fmt: skip


def build_library(base: Path, proc_spec: dict[str, Any] | None = None) -> Library:
    from drone_fly.tracks import catalogue as cat

    root = copy_fixtures(base / "trackdata")
    tracks = base / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    lift = scan(root, tracks / "liftoff.yaml")
    # the temp catalogue names its absolute temp root so every loader finds the copied files
    lift.header["root"] = str(root)
    cat.dump_catalogue(lift, tracks / "liftoff.yaml")
    proc = cat.Catalogue(
        tracks / "procedural.yaml",
        {"schema_version": 1, "source": "procedural", "world_scale": 1.0},
        [{"id": "proc-test", "source": "procedural", "map": "procedural",
          "gate_types": ["procedural-square"], "laps": None, "split": "train", "status": "ok",
          "spec": dict(proc_spec or PROC_SPEC)}],
    )  # fmt: skip
    cat.dump_catalogue(proc, tracks / "procedural.yaml")
    return Library(
        base,
        root,
        root / "scenes",
        tracks,
        base / "cache",
        cat.load_catalogue(tracks / "liftoff.yaml"),
        cat.load_catalogue(tracks / "procedural.yaml"),
    )


def use_library(mp: pytest.MonkeyPatch, lib: Library) -> None:
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks import loader

    mp.setattr(cat, "catalogue_dir", lambda: lib.tracks_dir)
    mp.setattr(loader, "cache_root", lambda: lib.cache_dir)
    mp.delenv(cat.TRACKDATA_ENV, raising=False)
    mp.delenv(cat.SCENES_ENV, raising=False)


def load(lib: Library, entry_id: str, **kw: Any) -> Any:
    """The one :class:`~drone_fly.sim.pool.SimCourse` of a Liftoff entry (no disk cache)."""
    from drone_fly.tracks.loader import load_course

    e = lib.liftoff.by_id(entry_id)
    return load_course(e, catalogue=lib.liftoff, root=lib.root, scenes_root=lib.scenes_root,
                       use_cache=False, **kw)[0]  # fmt: skip


def write_teacher_config(lib: Library, path: Path, **over: Any) -> Path:
    """A tiny CPU teacher config (``extends: teacher-liftoff``) over the temp library.

    4 envs, 4-step rollouts, a checkpoint every iteration, and course time limits capped at 0.1 s
    (``tracks.conversion``) so episodes end (timeout) within a rollout.
    """
    import yaml

    data = {
        "extends": "teacher-liftoff",
        "name": "teacher-test",
        "n_envs": 4,
        "budget": 1.0e9,
        "device": "cpu",
        "seed": 3,
        "ppo": {"roll": 4, "epochs": 1, "minibatches": 2},
        "log_every_iters": 1,
        "checkpoint_every_iters": 1,
        "checkpoint_every_s": 1.0e9,
        "episode_log_capacity": 512,
        "tracks": {
            "mix": [{"select": "liftoff:train", "weight": 0.6, "scene_weight": 2.0},
                    {"select": "procedural:proc-test", "weight": 0.4}],
            "root": str(lib.root),
            "scenes_root": str(lib.scenes_root),
            "conversion": {"time_base_s": 0.05, "time_max_s": 0.1},
        },
    }  # fmt: skip
    data.update(over)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def action_spec() -> Any:
    from drone_fly.sim.action import ActionSpec

    return ActionSpec.v2(1.0, 4.0)


def make_env(
    courses: list[Any],
    *,
    n_envs: int = 4,
    groups: list[Any] | None = None,
    scene_usable: Any = None,
    start: dict[str, Any] | None = None,
    seed: int = 0,
    randomization_scale: float = 0.0,
    timing: Any = None,
    log_capacity: int | None = 64,
    env: dict[str, Any] | None = None,
) -> Any:
    from drone_fly import drones
    from drone_fly.sim.multi_course import MultiCourseConfig, MultiCourseEnv
    from drone_fly.sim.pool import PoolGroup, pack_courses
    from drone_fly.sim.timing import TimingConfig

    groups = groups or [PoolGroup("all", list(range(len(courses))), 1.0)]
    cfg = MultiCourseConfig.from_dict({"reward_reference_dt_s": 0.02, **(env or {}),
                                       "start": start or {"ground_fraction": 1.0}})  # fmt: skip
    return MultiCourseEnv(
        drones.load("liftoff-gremlin-micro"),
        n_envs,
        pack_courses(courses),
        groups,
        scene_usable=scene_usable,
        device="cpu",
        seed=seed,
        config=cfg,
        timing=timing or TimingConfig.fixed(50.0),
        randomization_scale=randomization_scale,
        action_spec=action_spec(),
        log_capacity=log_capacity,
    )
