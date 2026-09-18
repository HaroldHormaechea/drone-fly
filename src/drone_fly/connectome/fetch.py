"""Provision the full MaleCNS connectome from the public ``connectome_data_prep`` source.

This is the runtime counterpart to the offline :mod:`drone_fly.connectome.loader`: the loader
never touches the network, and this module is the *only* place the full whole-brain matrix is
fetched. It mirrors the ``urllib`` style of ``scripts/fetch_soma_positions.py`` but is
production code (imported by the CLI's ``prune`` and ``fetch-connectome`` commands), so it is
written to be safe under interruption:

* **Atomic:** both files are written to in-destination ``.part`` temp paths and integrity-checked
  *before* any file is moved into place, so a partial or corrupt download never poisons the cache.
* **npz-last commit point:** the meta CSV is ``os.replace``-d first and the ``.npz`` last. The
  loader keys off the ``.npz`` (it globs ``*.npz`` and infers the ``<stem>_meta.csv`` sidecar), so
  a crash before the final rename leaves *no* ``.npz`` — a clean, retriable state rather than a
  lone-``.npz`` whose metadata is missing.
* **Reuse:** if both destination files already exist the network is never touched (``force=True``
  overrides). Destination resolution matches the loader exactly (explicit arg →
  ``DRONE_FLY_CONNECTOME_DIR`` → ``data/connectome``).

The canonical source is the public CC-BY URLs. As a per-machine convenience, if the source files
are already present in ``~/.cache/drone-fly/connectome_src/`` (e.g. left by
``scripts/fetch_soma_positions.py``) they are copied from there instead of re-downloaded; CI never
has that cache, so it always exercises the real (mocked, in tests) network path.

The source meta is saved **renamed** to ``<npz-stem>_meta.csv`` (i.e.
``mcns_inprop_all_neuron_meta.csv``) so the loader's sidecar inference finds it — the source repo
names the two files with mismatched stems.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from drone_fly.connectome.loader import _resolve_artifact_dir

logger = logging.getLogger("drone_fly.connectome.fetch")

#: Base URL of the public CC-BY ``connectome_data_prep`` MaleCNS data directory.
RAW_BASE = "https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS"

#: Source filenames in the upstream repo. Note the mismatched stems: the meta is named after
#: ``mcns_all_neuron`` but the matrix after ``mcns_inprop_all_neuron`` — hence the rename below.
SOURCE_NPZ_NAME = "mcns_inprop_all_neuron.npz"
SOURCE_META_NAME = "mcns_all_neuron_meta.csv"

#: Destination filenames. The ``.npz`` keeps its source name; the meta is saved as
#: ``<npz-stem>_meta.csv`` so :func:`drone_fly.connectome.loader.load_connectome` (which infers the
#: sidecar from the npz stem) finds it.
DEST_NPZ_NAME = SOURCE_NPZ_NAME
DEST_META_NAME = f"{Path(SOURCE_NPZ_NAME).stem}_meta.csv"

NPZ_URL = f"{RAW_BASE}/{SOURCE_NPZ_NAME}"
META_URL = f"{RAW_BASE}/{SOURCE_META_NAME}"

#: Per-machine convenience source (shared with ``scripts/fetch_soma_positions.py``). Checked
#: before the network; absent in CI so CI always takes the download path.
CACHE_SRC_DIR = Path.home() / ".cache" / "drone-fly" / "connectome_src"

#: Coarse size floors (bytes) used as a cheap truncation/error-page guard before a file is
#: committed. The real artifacts are ~82 MB (npz) and ~27 MB (meta); these floors only need to
#: reject HTML error pages and grossly truncated transfers. Deeper corruption is caught by
#: ``zipfile.is_zipfile`` on the npz and by the loader's row-count/shape checks on load.
NPZ_MIN_BYTES = 1_000_000
META_MIN_BYTES = 100_000


class ConnectomeDownloadError(RuntimeError):
    """A user-facing failure while provisioning the full connectome.

    Raised with an actionable, single-line message (network down, integrity check failed, etc.).
    The CLI maps it to a one-line error + exit code 2, matching the ``ConfigError`` convention —
    never a stack trace. On any failure the partial ``.part`` temp files are removed first, so a
    retry is never poisoned.
    """


def _provision_source(url: str, cache_src: Path, temp_path: str | os.PathLike[str]) -> None:
    """Populate ``temp_path`` from the per-machine cache if present, else download from ``url``."""
    temp_path = os.fspath(temp_path)
    if cache_src.is_file():
        logger.info("using cached source %s", cache_src)
        shutil.copyfile(cache_src, temp_path)
        return
    logger.info("downloading %s", url)
    urllib.request.urlretrieve(url, temp_path)  # noqa: S310 - fixed trusted https URL


def _check_min_size(path: str | os.PathLike[str], minimum: int, what: str) -> None:
    """Raise :class:`ConnectomeDownloadError` if ``path`` is smaller than ``minimum`` bytes."""
    size = os.path.getsize(path)
    if size < minimum:
        raise ConnectomeDownloadError(
            f"downloaded {what} is only {size} bytes (< {minimum} expected); the source may be "
            f"unreachable or returned an error page. No connectome was written; re-run to retry."
        )


def ensure_full_connectome(
    dest_dir: str | os.PathLike[str] | None = None, *, force: bool = False
) -> Path:
    """Ensure the full MaleCNS connectome is present at ``dest_dir``, downloading if absent.

    Destination resolution mirrors :func:`drone_fly.connectome.loader.load_connectome` exactly:
    explicit ``dest_dir`` → ``DRONE_FLY_CONNECTOME_DIR`` → ``data/connectome``.

    If both ``<dest>/mcns_inprop_all_neuron.npz`` and its ``<stem>_meta.csv`` sidecar already
    exist (and ``force`` is false), the directory is returned unchanged with **no** network I/O.
    Otherwise both files are fetched (from the per-machine cache if available, else the public
    CC-BY URLs), written to in-destination ``.part`` temps, integrity-checked, and atomically
    moved into place — meta first, ``.npz`` last.

    Parameters
    ----------
    dest_dir:
        Optional destination directory. ``None`` uses the loader's resolution order.
    force:
        Re-download even if the destination files already exist.

    Returns
    -------
    Path
        The resolved destination directory (ready to hand to ``load_connectome``).

    Raises
    ------
    ConnectomeDownloadError
        On any provisioning failure (network error, integrity check, filesystem error). No
        partial ``.part`` file is left behind.
    """
    dest = _resolve_artifact_dir(dest_dir)
    dest_npz = dest / DEST_NPZ_NAME
    dest_meta = dest / DEST_META_NAME

    if not force and dest_npz.is_file() and dest_meta.is_file():
        logger.info("full connectome already present at %s (reusing, no download)", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    # In-destination temps so every os.replace is a same-filesystem atomic rename.
    npz_temp = dest / f".{DEST_NPZ_NAME}.part"
    meta_temp = dest / f".{DEST_META_NAME}.part"

    def _cleanup() -> None:
        for temp in (npz_temp, meta_temp):
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    try:
        # 1. Provision BOTH temps (cache copy or download) before touching the destination files.
        _provision_source(NPZ_URL, CACHE_SRC_DIR / SOURCE_NPZ_NAME, npz_temp)
        _provision_source(META_URL, CACHE_SRC_DIR / SOURCE_META_NAME, meta_temp)

        # 2. Integrity-check BOTH temps before ANY move.
        _check_min_size(npz_temp, NPZ_MIN_BYTES, "connectome matrix (.npz)")
        if not zipfile.is_zipfile(npz_temp):
            raise ConnectomeDownloadError(
                "downloaded connectome matrix is not a valid .npz (zip) archive; the transfer "
                "was likely truncated or corrupted. No connectome was written; re-run to retry."
            )
        _check_min_size(meta_temp, META_MIN_BYTES, "connectome metadata (.csv)")

        # 3. Atomic commit: meta FIRST, .npz LAST. The .npz is the commit point the loader keys
        #    off, so a crash before its rename leaves no .npz = a clean, retriable state.
        os.replace(meta_temp, dest_meta)
        os.replace(npz_temp, dest_npz)
    except ConnectomeDownloadError:
        _cleanup()
        raise
    except OSError as e:
        _cleanup()
        raise ConnectomeDownloadError(
            f"failed to download the full MaleCNS connectome into {dest}: {e}. Check your network "
            f"connection (source: {RAW_BASE}) and re-run. No partial connectome was left behind."
        ) from e

    logger.info("full connectome ready at %s", dest)
    return dest
