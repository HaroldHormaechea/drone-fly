"""The data root wired into the UC71 tools (2026-10-14): track data and scene roots, the captures
root and the K1 connectome path resolve through :mod:`drone_fly.paths`.

Every test points ``DRONE_FLY_DATA_ROOT`` at a temp folder, so nothing reads or writes a real
data root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from drone_fly import paths
from drone_fly.harness import contract
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline import capture_index
from drone_fly.tracks import catalogue as cat


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(root))
    monkeypatch.delenv(cat.TRACKDATA_ENV, raising=False)
    monkeypatch.delenv(cat.SCENES_ENV, raising=False)
    monkeypatch.delenv(contract.K1_PATH_ENV, raising=False)
    return root


# ----------------------------------------------------------------------------- track data


def test_trackdata_root_flag_env_header_then_data_root(data_root, tmp_path, monkeypatch):
    hdr = {"root": str(tmp_path / "hdr")}
    assert cat.resolve_root(None, hdr) == tmp_path / "hdr"  # a real header root
    monkeypatch.setenv(cat.TRACKDATA_ENV, str(tmp_path / "env"))
    assert cat.resolve_root(None, hdr) == tmp_path / "env"
    assert cat.resolve_root(str(tmp_path / "flag"), hdr) == tmp_path / "flag"
    monkeypatch.delenv(cat.TRACKDATA_ENV)
    # the legacy header value (what the committed catalogue keeps) and no root: the data root
    assert cat.DEFAULT_ROOT == "artifacts/harness/trackdata"
    legacy = [{"root": cat.DEFAULT_ROOT}, {"root": ""}, {"root": None}, {}]
    if os.name == "nt":  # a Windows-written header: backslashes are separators there only
        legacy.append({"root": "artifacts\\harness\\trackdata"})
    for header in legacy:
        assert cat.resolve_root(None, header) == data_root / "trackdata" == paths.trackdata_root()
    # a non-legacy relative header root stays repo-relative
    assert cat.resolve_root(None, {"root": "elsewhere/td"}) == cat.repo_root() / "elsewhere/td"


def test_scenes_root_flag_env_header_then_data_root_or_beside(data_root, tmp_path, monkeypatch):
    td = paths.trackdata_root()
    assert cat.resolve_scenes_root(None, {}, td) == data_root / "scenes" == paths.scenes_root()
    other = tmp_path / "td"
    assert cat.resolve_scenes_root(None, {}, other) == other / "scenes"  # beside a custom root
    hdr = {"scenes_root": str(tmp_path / "hdr-scenes")}
    assert cat.resolve_scenes_root(None, hdr, td) == tmp_path / "hdr-scenes"
    monkeypatch.setenv(cat.SCENES_ENV, str(tmp_path / "env-scenes"))
    assert cat.resolve_scenes_root(None, hdr, td) == tmp_path / "env-scenes"
    assert cat.resolve_scenes_root(str(tmp_path / "flag"), hdr, td) == tmp_path / "flag"


def test_a_legacy_catalogue_resolves_both_roots_into_the_data_root(data_root):
    hdr = {"root": cat.DEFAULT_ROOT}
    root = cat.resolve_root(None, hdr)
    assert cat.resolve_scenes_root(None, hdr, root) == data_root / "scenes"


# ----------------------------------------------------------------------------- captures


def test_captures_root_is_the_data_roots(data_root, tmp_path, monkeypatch):
    assert capture_index.captures_root() == paths.captures_root() == data_root / "captures"
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(tmp_path / "other"))
    assert capture_index.captures_root() == tmp_path / "other" / "captures"
    d = data_root / "captures" / "FakeArena" / "fake-arena-2026-10-14-manual-01"
    d.mkdir(parents=True)
    (d / "frames.csv").write_text("t_wall_ns,file\n", "utf-8")
    (d / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n", "utf-8")
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(data_root))
    assert capture_index.iter_capture_dirs() == [d]  # discovery defaults to it


# ----------------------------------------------------------------------------- K1 connectome


def test_k1_path_prefers_the_data_root_over_the_legacy_repo_path(data_root, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    legacy = repo / "artifacts" / "pruned" / "k1"
    monkeypatch.setattr(contract, "find_repo_root", lambda: repo)
    assert contract.resolve_k1_path(None) is None  # neither exists
    legacy.mkdir(parents=True)
    assert contract.resolve_k1_path(None) == legacy.resolve()  # the legacy repo path
    k1 = data_root / "pruned" / "k1"
    k1.mkdir(parents=True)
    assert contract.resolve_k1_path(None) == k1.resolve()  # the data root wins
    explicit = tmp_path / "mine"
    explicit.mkdir()
    assert contract.resolve_k1_path(explicit) == explicit.resolve()  # an explicit path wins
    monkeypatch.setenv(contract.K1_PATH_ENV, str(explicit))
    assert contract.resolve_k1_path(None) == explicit.resolve()
    with pytest.raises(HarnessConfigError, match="does not exist"):
        contract.resolve_k1_path(tmp_path / "missing")


def test_no_helper_creates_folders(data_root):
    cat.resolve_root(None, {})
    cat.resolve_scenes_root(None, {}, paths.trackdata_root())
    capture_index.captures_root()
    assert not data_root.exists()
    assert isinstance(paths.trackdata_root(), Path)
