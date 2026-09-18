"""Tests for the full-connectome provisioning module (UC-14).

Guards :func:`drone_fly.connectome.fetch.ensure_full_connectome` — the *only* place the full
whole-brain MaleCNS matrix is fetched — against the AC2/AC7 contract:

* **reuse** — both destination files present ⇒ no network, no cache copy (AC2);
* **download** — absent ⇒ provisioned, with the meta saved *renamed* to ``<npz-stem>_meta.csv``
  so the loader's sidecar inference finds it (AC2);
* **atomicity (AC7)** — in-destination ``.part`` temps; integrity-checked before any move;
  ``os.replace`` meta-first / npz-last so a crash before the npz commit leaves **no** ``.npz``
  (a clean, retriable state) and never a ``.part`` leftover;
* **integrity** — a truncated (size-floor) or non-zip ``.npz`` raises
  :class:`ConnectomeDownloadError` and leaves nothing partial behind.

**Hermeticity.** The real full connectome is cached on this machine at
``~/.cache/drone-fly/connectome_src/`` (``fetch.CACHE_SRC_DIR``). An unmocked call would copy that
82 MB cache and silently pass — hiding regressions and defeating offline CI. Every test here either
runs under the ``no_network`` fixture, redirects ``CACHE_SRC_DIR`` to an empty/synthetic temp dir,
or mocks ``urllib.request.urlretrieve`` — so **no** test ever touches the machine cache or the
network. Tests that exercise the download path assert it went through the mock.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from drone_fly.connectome import fetch
from drone_fly.connectome.fetch import (
    DEST_META_NAME,
    DEST_NPZ_NAME,
    META_MIN_BYTES,
    NPZ_MIN_BYTES,
    SOURCE_META_NAME,
    SOURCE_NPZ_NAME,
    ConnectomeDownloadError,
    ensure_full_connectome,
)

# --------------------------------------------------------------------------- #
# Synthetic artifact builders — cheap stand-ins that pass the integrity gate
# (valid zip + size floors) without needing the real 82 MB matrix.
# --------------------------------------------------------------------------- #


def _write_valid_npz(path: Path, min_bytes: int = NPZ_MIN_BYTES + 50_000) -> None:
    """Write a *valid* (uncompressed) zip archive comfortably above the npz size floor."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("data.npy", b"\x00" * min_bytes)
    assert zipfile.is_zipfile(path) and path.stat().st_size >= NPZ_MIN_BYTES


def _write_valid_meta(path: Path, min_bytes: int = META_MIN_BYTES + 50_000) -> None:
    """Write a CSV comfortably above the meta size floor."""
    path.write_bytes(b"bodyid,idx\n" + b"0,0\n" * (min_bytes // 4))
    assert path.stat().st_size >= META_MIN_BYTES


def _make_fake_urlretrieve(npz_writer=_write_valid_npz, meta_writer=_write_valid_meta):
    """Return ``(fake_urlretrieve, calls)`` — a stand-in that writes synthetic files locally.

    ``calls`` records every ``(url, filename)`` so a test can assert the download path (not the
    machine cache) was taken and that the temp lands *in the destination* dir.
    """
    calls: list[tuple[str, str]] = []

    def fake(url, filename):  # noqa: ANN001, ANN202
        calls.append((str(url), str(filename)))
        target = Path(filename)
        if str(url).endswith(".npz"):
            npz_writer(target)
        else:
            meta_writer(target)

    return fake, calls


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    """Redirect ``CACHE_SRC_DIR`` at an empty dir so cache-fallback misses -> download path."""
    cache = tmp_path / "empty_cache"
    cache.mkdir()
    monkeypatch.setattr(fetch, "CACHE_SRC_DIR", cache)
    return cache


@pytest.fixture
def synthetic_cache(tmp_path, monkeypatch):
    """Redirect ``CACHE_SRC_DIR`` at a dir holding *synthetic* source files (not the real cache).

    Lets the cache-fallback path run fully offline without copying the machine's 82 MB artifact.
    """
    cache = tmp_path / "synthetic_cache"
    cache.mkdir()
    _write_valid_npz(cache / SOURCE_NPZ_NAME)
    _write_valid_meta(cache / SOURCE_META_NAME)
    monkeypatch.setattr(fetch, "CACHE_SRC_DIR", cache)
    return cache


def _boom_urlretrieve(*_a, **_k):  # noqa: ANN002, ANN003
    raise AssertionError("network download attempted but must not happen in this test")


# --------------------------------------------------------------------------- #
# Reuse — both files present => zero network, zero cache copy (AC2)
# --------------------------------------------------------------------------- #


def test_reuse_when_present_does_no_network_or_copy(tmp_path, monkeypatch, no_network) -> None:
    dest = tmp_path / "connectome"
    dest.mkdir()
    (dest / DEST_NPZ_NAME).write_bytes(b"existing-npz")
    (dest / DEST_META_NAME).write_bytes(b"existing-meta")
    # Any provisioning attempt (download OR cache copy) must be a hard failure.
    monkeypatch.setattr("urllib.request.urlretrieve", _boom_urlretrieve)
    monkeypatch.setattr(fetch.shutil, "copyfile", _boom_urlretrieve)

    result = ensure_full_connectome(dest)

    assert result == dest
    # Files are untouched (reuse, not re-provision).
    assert (dest / DEST_NPZ_NAME).read_bytes() == b"existing-npz"
    assert (dest / DEST_META_NAME).read_bytes() == b"existing-meta"


def test_partial_presence_is_not_reuse(tmp_path, synthetic_cache) -> None:
    """Only the npz present (no meta) must NOT count as reuse — it re-provisions both."""
    dest = tmp_path / "connectome"
    dest.mkdir()
    (dest / DEST_NPZ_NAME).write_bytes(b"stale-lonely-npz")

    ensure_full_connectome(dest)

    assert (dest / DEST_META_NAME).is_file()  # the missing sidecar was provisioned
    # The lone stale npz was replaced by a freshly-provisioned valid archive.
    assert zipfile.is_zipfile(dest / DEST_NPZ_NAME)


# --------------------------------------------------------------------------- #
# Download success + meta rename + in-dest temp location (AC2)
# --------------------------------------------------------------------------- #


def test_download_success_writes_both_files_via_mock(tmp_path, empty_cache, monkeypatch) -> None:
    fake, calls = _make_fake_urlretrieve()
    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    dest = tmp_path / "connectome"

    result = ensure_full_connectome(dest)

    assert result == dest
    assert (dest / DEST_NPZ_NAME).is_file()
    assert (dest / DEST_META_NAME).is_file()
    # It went through the mock (download path), not the machine cache: both URLs fetched.
    assert len(calls) == 2
    urls = {url for url, _ in calls}
    assert any(u.endswith(SOURCE_NPZ_NAME) for u in urls)
    assert any(u.endswith(SOURCE_META_NAME) for u in urls)


def test_meta_saved_with_renamed_stem(tmp_path, empty_cache, monkeypatch) -> None:
    """The source meta (``mcns_all_neuron_meta.csv``) lands renamed to ``<npz-stem>_meta.csv``."""
    fake, _ = _make_fake_urlretrieve()
    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    dest = tmp_path / "connectome"

    ensure_full_connectome(dest)

    assert DEST_META_NAME == "mcns_inprop_all_neuron_meta.csv"
    assert (dest / DEST_META_NAME).is_file()
    # The source name must NOT survive — the loader infers the sidecar from the npz stem.
    assert not (dest / SOURCE_META_NAME).exists()


def test_temps_are_in_destination_dir(tmp_path, empty_cache, monkeypatch) -> None:
    """``.part`` temps are created *inside* dest (same-FS atomic rename), never in cache/tmp."""
    fake, calls = _make_fake_urlretrieve()
    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    dest = tmp_path / "connectome"

    ensure_full_connectome(dest)

    for _url, filename in calls:
        temp = Path(filename)
        assert temp.parent == dest
        assert temp.name.startswith(".") and temp.name.endswith(".part")


def test_download_uses_cache_when_present_offline(
    tmp_path, synthetic_cache, monkeypatch, no_network
) -> None:
    """The cache-fallback source is used (offline) when present — no network call."""
    monkeypatch.setattr("urllib.request.urlretrieve", _boom_urlretrieve)
    dest = tmp_path / "connectome"

    result = ensure_full_connectome(dest)

    assert result == dest
    assert (dest / DEST_NPZ_NAME).is_file()
    assert (dest / DEST_META_NAME).is_file()  # renamed sidecar from the synthetic cache


def test_force_reprovisions_even_when_present(tmp_path, synthetic_cache) -> None:
    dest = tmp_path / "connectome"
    dest.mkdir()
    (dest / DEST_NPZ_NAME).write_bytes(b"old")
    (dest / DEST_META_NAME).write_bytes(b"old")

    ensure_full_connectome(dest, force=True)

    # force=True ignores the reuse short-circuit and re-provisions a valid archive.
    assert zipfile.is_zipfile(dest / DEST_NPZ_NAME)


# --------------------------------------------------------------------------- #
# Resolution precedence (explicit -> env -> default) — AC1/AC2 default location
# --------------------------------------------------------------------------- #


def test_resolution_uses_env_var_when_no_explicit_dir(
    tmp_path, synthetic_cache, monkeypatch, no_network
) -> None:
    env_dir = tmp_path / "env_connectome"
    monkeypatch.setenv("DRONE_FLY_CONNECTOME_DIR", str(env_dir))

    result = ensure_full_connectome()  # no explicit dir -> resolves via the env var

    assert result == env_dir
    assert (env_dir / DEST_NPZ_NAME).is_file()
    assert (env_dir / DEST_META_NAME).is_file()


# --------------------------------------------------------------------------- #
# Atomicity (AC7) — npz-last commit; integrity gate; no partial left behind
# --------------------------------------------------------------------------- #


def test_npz_last_crash_leaves_no_npz_and_no_part(
    tmp_path, synthetic_cache, monkeypatch, no_network
) -> None:
    """A crash *between* the two moves (meta done, npz not) leaves NO .npz and NO .part.

    The npz is the loader's commit point, so this is a clean, retriable state — never a
    lone-npz / missing-meta corruption.
    """
    import os as _os

    real_replace = _os.replace

    def flaky_replace(src, dst):  # noqa: ANN001, ANN202
        if str(dst).endswith(".npz"):
            raise OSError("simulated crash before the npz commit point")
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", flaky_replace)
    dest = tmp_path / "connectome"

    with pytest.raises(ConnectomeDownloadError):
        ensure_full_connectome(dest)

    # npz never committed -> clean retriable state (loader keys off the .npz).
    assert not (dest / DEST_NPZ_NAME).exists()
    # No .part temp left behind for either file.
    assert list(dest.glob(".*.part")) == []


def test_integrity_size_floor_failure_leaves_nothing(tmp_path, empty_cache, monkeypatch) -> None:
    """A truncated npz (below the size floor) -> ConnectomeDownloadError, nothing partial left."""

    def tiny_npz(path: Path) -> None:
        path.write_bytes(b"x" * 100)  # far below NPZ_MIN_BYTES

    fake, _ = _make_fake_urlretrieve(npz_writer=tiny_npz)
    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    dest = tmp_path / "connectome"

    with pytest.raises(ConnectomeDownloadError, match="bytes"):
        ensure_full_connectome(dest)

    assert not (dest / DEST_NPZ_NAME).exists()
    assert not (dest / DEST_META_NAME).exists()
    assert list(dest.glob(".*.part")) == []


def test_integrity_not_a_zip_failure_leaves_nothing(tmp_path, empty_cache, monkeypatch) -> None:
    """A big-but-not-a-zip npz -> ConnectomeDownloadError (is_zipfile guard), nothing left."""

    def not_a_zip(path: Path) -> None:
        path.write_bytes(b"x" * (NPZ_MIN_BYTES + 1_000))  # passes size floor, fails is_zipfile

    fake, _ = _make_fake_urlretrieve(npz_writer=not_a_zip)
    monkeypatch.setattr("urllib.request.urlretrieve", fake)
    dest = tmp_path / "connectome"

    with pytest.raises(ConnectomeDownloadError, match="zip"):
        ensure_full_connectome(dest)

    assert not (dest / DEST_NPZ_NAME).exists()
    assert not (dest / DEST_META_NAME).exists()
    assert list(dest.glob(".*.part")) == []


def test_network_error_is_wrapped_and_cleaned(tmp_path, empty_cache, monkeypatch) -> None:
    """A raw OSError from the transport surfaces as an actionable ConnectomeDownloadError."""

    def boom(url, filename):  # noqa: ANN001, ANN202
        raise OSError("name resolution failed")

    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    dest = tmp_path / "connectome"

    with pytest.raises(ConnectomeDownloadError):
        ensure_full_connectome(dest)

    assert list(dest.glob(".*.part")) == []
