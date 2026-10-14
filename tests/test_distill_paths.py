"""UC70 distillation input resolution: precedence and existence guards for the default
teacher / noise-model / track-library helpers (``drone_fly.training.distill``) and the K1
connectome resolver (``drone_fly.harness.contract.resolve_k1_path``).

Pure pathlib, mirroring ``tests/test_paths.py``: a tmp data root via ``$DRONE_FLY_DATA_ROOT``,
every distillation-input env var cleared, and the legacy repo fallbacks neutralised, so each
helper is exercised rung by rung. Dirs/files are created only where a rung's existence guard
requires one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from drone_fly import paths
from drone_fly.harness import contract
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.tracks.catalogue import TRACKDATA_ENV
from drone_fly.training import distill


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A clean tmp data root with every distillation-input env var unset.

    ``$DRONE_FLY_DATA_ROOT`` points the whole layout at ``tmp_path/data`` (so nothing consults
    the real checkout), and the catalogue / teacher / noise-model / k1 env vars are removed so
    each helper falls through to its data-root default. ``contract`` binds ``find_repo_root`` by
    name at import, so the legacy ``find_repo_root()/artifacts/pruned/k1`` fallback is patched
    there to ``None`` — otherwise a real checkout's k1 would leak into the "nothing exists" case.
    """
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(root))
    for var in (distill.TEACHER_ENV, distill.NOISE_MODEL_ENV, TRACKDATA_ENV, contract.K1_PATH_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(contract, "find_repo_root", lambda *a, **kw: None)
    return root


# ----------------------------------------------------------------------------- default_teacher


def test_default_teacher_precedence(data_root, monkeypatch):
    # explicit wins and is not existence-checked
    assert distill.default_teacher("/explicit/teacher") == "/explicit/teacher"
    # the env var beats the data-root default
    monkeypatch.setenv(distill.TEACHER_ENV, "/env/teacher")
    assert distill.default_teacher() == "/env/teacher"
    monkeypatch.delenv(distill.TEACHER_ENV, raising=False)
    # the default resolves only when the UC68 run dir actually exists under the data root
    assert distill.default_teacher() is None
    run = paths.models_root("teacher") / distill.DEFAULT_TEACHER_RUN
    run.mkdir(parents=True)
    assert distill.default_teacher() == str(run)


# -------------------------------------------------------------------------- default_noise_model


def test_default_noise_model_precedence(data_root, monkeypatch):
    # explicit -> configured -> env -> the data root's detector-v1 noise model
    assert distill.default_noise_model("/explicit/noise.yaml") == "/explicit/noise.yaml"
    assert distill.default_noise_model(None, "/configured/noise.yaml") == "/configured/noise.yaml"
    monkeypatch.setenv(distill.NOISE_MODEL_ENV, "/env/noise.yaml")
    assert distill.default_noise_model() == "/env/noise.yaml"
    monkeypatch.delenv(distill.NOISE_MODEL_ENV, raising=False)
    # the default resolves only when the file (not just the dir) exists
    assert distill.default_noise_model() is None
    kind, det, name = distill.DEFAULT_NOISE_MODEL
    target = paths.models_root(kind) / det / name
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    assert distill.default_noise_model() == str(target)


# ------------------------------------------------------------------------- student_tracks_root


def test_student_tracks_root_precedence(data_root, monkeypatch):
    # explicit wins
    assert distill.student_tracks_root("/explicit/tracks") == "/explicit/tracks"
    # when the catalogue's own env var is set, defer to the catalogue default (None)
    monkeypatch.setenv(TRACKDATA_ENV, "/some/trackdata")
    assert distill.student_tracks_root() is None
    monkeypatch.delenv(TRACKDATA_ENV, raising=False)
    # the default resolves only when the track-library copy exists
    assert distill.student_tracks_root() is None
    lib = paths.track_library_root() / distill.TRACK_LIBRARY_TRACKDATA
    lib.mkdir(parents=True)
    assert distill.student_tracks_root() == str(lib)


# ------------------------------------------------------------------------- resolve_teacher_path


def test_resolve_teacher_path_existing_returned_as_is(data_root, tmp_path):
    recorded = tmp_path / "recorded" / "teacher.pt"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("x", encoding="utf-8")
    assert distill.resolve_teacher_path(recorded) == str(recorded)


def test_resolve_teacher_path_remaps_checkpoint_under_the_data_root(data_root, tmp_path):
    run = "20990101T000000Z-run"
    recorded = tmp_path / "old" / "teacher" / run / "checkpoints" / "ckpt_000100.pt"
    assert not recorded.exists()
    # the run folder is the part before ``checkpoints`` -> remapped under models/teacher/<run>/...
    remapped = paths.models_root("teacher") / run / "checkpoints" / "ckpt_000100.pt"
    remapped.parent.mkdir(parents=True)
    remapped.write_text("x", encoding="utf-8")
    assert distill.resolve_teacher_path(recorded) == str(remapped)


def test_resolve_teacher_path_checkpoint_remap_absent_returns_original(data_root, tmp_path):
    run = "20990101T000000Z-run"
    recorded = tmp_path / "old" / "teacher" / run / "checkpoints" / "ckpt_000100.pt"
    assert not recorded.exists()
    # ``checkpoints`` segment present but the remap target was NOT created -> the ``else str(p)``
    # branch returns the original recorded path unchanged.
    assert distill.resolve_teacher_path(recorded) == str(Path(recorded))


def test_resolve_teacher_path_remaps_bare_name_under_the_data_root(data_root, tmp_path):
    recorded = tmp_path / "gone" / "teacher.pt"  # no ``checkpoints`` segment -> bare-name fallback
    assert not recorded.exists()
    remapped = paths.models_root("teacher") / "teacher.pt"
    remapped.parent.mkdir(parents=True)
    remapped.write_text("x", encoding="utf-8")
    assert distill.resolve_teacher_path(recorded) == str(remapped)


def test_resolve_teacher_path_missing_with_nothing_returns_original(data_root, tmp_path):
    recorded = tmp_path / "gone" / "teacher.pt"
    assert not recorded.exists()
    assert distill.resolve_teacher_path(recorded) == str(Path(recorded))


# ------------------------------------------------------------- harness.contract.resolve_k1_path
# The K1 connectome is a distillation input (the reservoir tap store), but its resolver lives in
# the harness contract, not in ``distill`` — surfaced here with the other UC70 inputs.


def test_resolve_k1_path_defaults_to_the_data_root_pruned_k1(data_root):
    # nothing present (no env, no pruned/k1, legacy fallback neutralised) -> None
    assert contract.resolve_k1_path() is None
    k1 = paths.pruned_root() / "k1"
    k1.mkdir(parents=True)
    assert contract.resolve_k1_path() == k1.resolve()
    assert contract.resolve_k1_path() == (data_root / "pruned" / "k1").resolve()


def test_resolve_k1_path_env_pointing_nowhere_raises(data_root, tmp_path, monkeypatch):
    missing = tmp_path / "no" / "such" / "k1"
    monkeypatch.setenv(contract.K1_PATH_ENV, str(missing))
    with pytest.raises(HarnessConfigError):
        contract.resolve_k1_path()
