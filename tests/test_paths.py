"""The data root (``drone_fly.paths``): environment precedence, the repo default, the no-repo
fallback, the fixed layout helpers and the model kinds. Pure pathlib, nothing is created."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_fly import paths
from drone_fly.harness import config

LAYOUT = {
    "captures_root": "captures",
    "trackdata_root": "trackdata",
    "scenes_root": "scenes",
    "track_library_root": "track-library",
    "labels_root": "labels",
    "reference_root": "reference",
    "sysid_root": "sysid",
    "pruned_root": "pruned",
}


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv(paths.DATA_ROOT_ENV, raising=False)


def test_constants():
    assert paths.DATA_ROOT_ENV == "DRONE_FLY_DATA_ROOT"
    assert paths.DEFAULT_DATA_ROOT == Path("artifacts", "harness", "liftoff", "training-data")
    assert paths.MODEL_KINDS == ("detector-v1", "detector-v2", "teacher", "student")


def test_the_environment_wins(monkeypatch, tmp_path):
    def never():
        raise AssertionError("the repo is not consulted when the variable is set")

    monkeypatch.setattr(config, "find_repo_root", never)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "data"))
    assert paths.data_root() == tmp_path / "data"
    monkeypatch.setenv(paths.DATA_ROOT_ENV, f"  {tmp_path / 'padded'}  \n")
    assert paths.data_root() == tmp_path / "padded"  # surrounding whitespace stripped
    monkeypatch.setenv(paths.DATA_ROOT_ENV, "relative/data")
    assert paths.data_root() == Path("relative/data")  # kept as given, not resolved


def test_the_environment_expands_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv(paths.DATA_ROOT_ENV, "~/drone-data")
    assert paths.data_root() == tmp_path / "drone-data"
    assert "~" not in str(paths.data_root())


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_an_empty_or_blank_variable_counts_as_unset(monkeypatch, tmp_path, value):
    monkeypatch.setattr(config, "find_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setenv(paths.DATA_ROOT_ENV, value)
    assert paths.data_root() == tmp_path / "repo" / paths.DEFAULT_DATA_ROOT


def test_the_repo_default(no_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "find_repo_root", lambda: tmp_path)
    assert paths.data_root() == (tmp_path / "artifacts" / "harness" / "liftoff" / "training-data")


def test_the_real_repo_default(no_env):
    root = config.find_repo_root()
    assert root is not None  # the tests run from a checkout
    assert paths.data_root() == root / paths.DEFAULT_DATA_ROOT


def test_without_a_repo_the_default_is_under_the_current_directory(no_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "find_repo_root", lambda: None)
    monkeypatch.chdir(tmp_path)
    assert paths.data_root() == tmp_path / paths.DEFAULT_DATA_ROOT
    assert paths.data_root().is_absolute()


@pytest.mark.parametrize(("helper", "name"), sorted(LAYOUT.items()))
def test_every_layout_helper_is_a_child_of_the_data_root(monkeypatch, tmp_path, helper, name):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    got = getattr(paths, helper)()
    assert isinstance(got, Path) and got == tmp_path / name and got.parent == tmp_path
    assert not got.exists()  # pure pathlib: nothing is created


def test_the_layout_follows_the_data_root(no_env, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "find_repo_root", lambda: tmp_path)
    for helper, name in LAYOUT.items():
        assert getattr(paths, helper)() == tmp_path / paths.DEFAULT_DATA_ROOT / name


def test_models_root_and_its_kinds(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path))
    assert paths.models_root() == tmp_path / "models"
    assert paths.models_root(None) == tmp_path / "models"
    for kind in paths.MODEL_KINDS:
        assert paths.models_root(kind) == tmp_path / "models" / kind
    for bad in ("detector-v3", "Teacher", "", "models", "../student"):
        with pytest.raises(ValueError, match=f"unknown model kind {bad!r}"):
            paths.models_root(bad)
    assert not (tmp_path / "models").exists()


def test_nothing_is_cached_between_calls(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "a"))
    assert paths.captures_root() == tmp_path / "a" / "captures"
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "b"))
    assert paths.captures_root() == tmp_path / "b" / "captures"


def test_results_survive_an_as_posix_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "data root"))  # a space too
    results = [paths.data_root(), paths.models_root(), *(paths.models_root(k) for k in
               paths.MODEL_KINDS), *(getattr(paths, h)() for h in LAYOUT)]  # fmt: skip
    for p in results:
        assert isinstance(p, Path)
        assert Path(p.as_posix()) == p
        assert "\\" not in p.as_posix()
