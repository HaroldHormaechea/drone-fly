"""AC5/AC8/AC9/AC11/AC12 — the viewer's JSON contract, static assets, and docs.

The static viewer (``viz/``) is validated by the JSON contract it consumes — **no browser
or network in CI** (AC11). Verified:

* **AC5/AC8/AC9** — a recorded file carries every field the three synced panels read: the
  spatial brain map (``positions.{coords2d,coords3d,has_position,source,projection}``), the
  role-ordered heatmap (``meta.roles``), and the flight panel (``frames.actions`` +
  ``frames.drone_position`` + ``meta.action_layout``), all on one frame timeline.
* **AC5/AC11** — the viewer assets exist, are dependency-free (no build step / npm), load
  the file via a local file-picker (works from ``file://``), and never fetch from the
  network.
* **AC12** — the README documents the record flags, the artifacts path, viewer usage, and
  the coordinate provisioning (tokenless default + token + fallback); ``.env.example``
  documents the token.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.record.recorder import ActivationRecorder

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VIZ = _REPO_ROOT / "viz"


@pytest.fixture
def recorded_doc(connectome: ConnectomeData, tmp_path: Path) -> dict:
    """A small recorded episode document (drives the viewer contract assertions)."""
    rec = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(0, seed=0)
    for f in range(3):
        rec.sink(np.linspace(-1.0, 1.0, n, dtype=np.float32) * (f + 1) / 3)
        rec.capture_frame(np.array([0.4, -0.1, 0.2, -0.3]), np.array([float(f), 0.1, 1.0]))
    path = rec.finish_episode(completed=False, completion_time=None, total_reward=-0.5, steps=3)
    return json.loads(path.read_text())


# --- AC5/AC8/AC9 the recorded file carries the full viewer contract --------------------
def test_recorded_file_satisfies_viewer_contract(recorded_doc: dict) -> None:
    doc = recorded_doc
    n = doc["meta"]["n_neurons"]
    n_frames = doc["meta"]["n_frames"]

    # (a) spatial brain-map panel — projected soma positions + provenance (AC5/AC6).
    positions = doc["meta"]["positions"]
    assert set(positions) >= {"coords2d", "coords3d", "has_position", "source", "projection"}
    assert len(positions["coords2d"]) == n
    assert len(positions["coords3d"]) == n
    assert len(positions["has_position"]) == n
    assert positions["projection"] in {"xz", "xy", "yz"}
    assert np.isfinite(np.asarray(positions["coords2d"], dtype=float)).all()

    # (b) heatmap panel — one role per neuron for row ordering/colour (AC9).
    roles = doc["meta"]["roles"]
    assert len(roles) == n
    assert set(roles) <= {"sensory", "interneuron", "motor"}

    # (c) flight panel — 4 action traces (labelled) + drone path, on the same timeline (AC8).
    assert doc["meta"]["action_layout"] == ["throttle", "roll", "pitch", "yaw"]
    assert len(doc["frames"]["actions"]) == n_frames
    assert len(doc["frames"]["drone_position"]) == n_frames
    assert all(len(a) == 4 for a in doc["frames"]["actions"])
    assert all(len(p) == 3 for p in doc["frames"]["drone_position"])

    # One shared timeline: activations / actions / positions all have n_frames rows.
    assert len(doc["frames"]["activations"]) == n_frames


# --- AC5/AC11 static assets, dependency-free, no network -------------------------------
def test_viewer_assets_exist() -> None:
    for name in ("viewer.html", "viewer.js", "viewer.css"):
        assert (_VIZ / name).is_file(), f"missing viz/{name}"


def test_viewer_html_uses_local_file_picker_and_no_npm() -> None:
    html = (_VIZ / "viewer.html").read_text()
    assert 'id="file-input"' in html  # local file-picker (works from file://)
    assert 'type="file"' in html
    assert "viewer.js" in html and "viewer.css" in html
    # Dependency-free: no bundler/CDN <script src="http..."> imports.
    assert "node_modules" not in html
    assert "https://" not in html and "http://" not in html


def test_viewer_js_references_contract_fields_and_avoids_network() -> None:
    js = (_VIZ / "viewer.js").read_text()
    # Reads exactly the schema fields the recorder writes (AC5/AC8/AC9).
    for field in (
        "schema_version",
        "positions",
        "coords2d",
        "coords3d",
        "has_position",
        "roles",
        "action_layout",
        "activations",
        "actions",
        "drone_position",
    ):
        assert field in js, f"viewer.js does not reference contract field {field!r}"
    # No network: the file is read locally (FileReader / DecompressionStream), never fetched.
    assert "fetch(" not in js
    assert "XMLHttpRequest" not in js
    assert "https://" not in js and "http://" not in js
    assert "DecompressionStream" in js  # gzip handled in-browser


# --- AC12 docs -------------------------------------------------------------------------
def test_readme_documents_recording_and_viewer() -> None:
    readme = (_REPO_ROOT / "README.md").read_text()
    for token in (
        "--record",
        "--record-every",
        "artifacts/activations",
        "viz/viewer.html",
        "NEUPRINT_TOKEN",
        "fetch_soma_positions.py",
    ):
        assert token in readme, f"README missing {token!r}"
    lower = readme.lower()
    assert "fallback" in lower  # computed-layout fallback documented
    assert "soma" in lower  # coordinate provisioning documented


def test_env_example_documents_token() -> None:
    env = (_REPO_ROOT / ".env.example").read_text()
    assert "NEUPRINT_TOKEN" in env
    assert "soma" in env.lower()
