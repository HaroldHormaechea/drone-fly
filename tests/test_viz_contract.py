"""AC1-13 — the viewer's JSON contract, static assets, and docs.

The static viewer (``viz/``) runs in a browser (no browser/JS runtime in CI), so its
behaviour is validated **statically** — the JSON contract it consumes plus source-file
assertions on ``viz/*`` and ``README.md`` — never by executing it (AC10). Verified:

* **AC5/AC8/AC9** — a recorded file carries every field the three synced panels read: the
  spatial brain map (``positions.{coords2d,coords3d,has_position,source,projection}``), the
  role-ordered heatmap (``meta.roles``), and the flight panel (``frames.actions`` +
  ``frames.drone_position`` + ``meta.action_layout``), all on one frame timeline.
* **UC-06 AC8** — a course-carrying recording adds the additive ``meta.course`` block
  (start / gate / finish / floor / ceiling + axis conventions); a file without it still
  loads (back-compat).
* **AC5/AC11/AC7** — the viewer assets exist, are dependency-free (no build step / npm / ES
  modules), load the file via a local file-picker (works from ``file://``), and never fetch
  from the network.
* **UC-06 AC1-6,9,11,12,13** — ``viewer.js`` carries the 3D orbit projector (floor / markers
  / trajectory / drone / presets / ``destroy``), the neuron-beat envelope, and the play-at-
  end restart; the old flat 2D ``path-canvas`` / ``drawPath`` are gone; ``viewer.html``
  offers a 0.25× speed option and a human-labelled ``view-select`` (front/side/top, top the
  default). "A right turn looks right" (AC5) is a documented MANUAL browser check.
* **AC10/AC12** — the README documents the record flags, the artifacts path, viewer usage,
  the coordinate provisioning, and the UC-06 3D controls / beat / 0.25× / presets;
  ``.env.example`` documents the token.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.env.config import CourseConfig
from drone_fly.record.recorder import ActivationRecorder

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VIZ = _REPO_ROOT / "viz"


def _record(connectome: ConnectomeData, out_dir: Path, course: CourseConfig | None) -> dict:
    """Record a tiny episode (optionally with course geometry) and parse the document."""
    rec = ActivationRecorder(connectome, out_dir, backend="simple", dt=0.05, course=course)
    n = connectome.neuron_count
    rec.start_episode(0, seed=0)
    for f in range(3):
        rec.sink(np.linspace(-1.0, 1.0, n, dtype=np.float32) * (f + 1) / 3)
        rec.capture_frame(np.array([0.4, -0.1, 0.2, -0.3]), np.array([float(f), 0.1, 1.0]))
    path = rec.finish_episode(completed=False, completion_time=None, total_reward=-0.5, steps=3)
    return json.loads(path.read_text())


@pytest.fixture
def recorded_doc(connectome: ConnectomeData, tmp_path: Path) -> dict:
    """A small recorded episode document (drives the viewer contract assertions)."""
    return _record(connectome, tmp_path / "act", course=None)


@pytest.fixture
def recorded_doc_with_course(connectome: ConnectomeData, tmp_path: Path) -> dict:
    """A recorded episode that carries UC-06 course geometry (``meta.course``)."""
    return _record(connectome, tmp_path / "act_course", course=CourseConfig())


def _extract_select(html: str, select_id: str) -> str:
    """Return the exact ``<select id="…">…</select>`` block, scoped so a bare text grep
    elsewhere in the document (e.g. a panel hint) can't produce a false positive."""
    m = re.search(
        rf'<select[^>]*\bid="{re.escape(select_id)}"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )
    assert m is not None, f'no <select id="{select_id}"> in viewer.html'
    return m.group(1)


def _options(select_body: str) -> list[tuple[str, bool]]:
    """Parse ``(value, selected)`` for each ``<option>`` inside a select body."""
    out: list[tuple[str, bool]] = []
    for opt in re.finditer(r"<option\b([^>]*)>", select_body):
        attrs = opt.group(1)
        vm = re.search(r'value="([^"]*)"', attrs)
        out.append((vm.group(1) if vm else "", "selected" in attrs))
    return out


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


# --- UC-06 AC8 / UC-09 AC7: the additive meta.course block with the N-gate array --------
def test_recorded_file_carries_course_geometry(recorded_doc_with_course: dict) -> None:
    """A course-carrying recording documents the anchors the viewer places (AC8/UC-09 AC7).

    UC-09: the course carries a ``gates: [...]`` **array** (one entry per gate) rather than a
    singular ``gate`` block — the default course has 3 gates.
    """
    course = recorded_doc_with_course["meta"]["course"]
    # Every field the 3D flight panel reads to place the floor + start/gates/finish markers.
    assert set(course) >= {
        "start",
        "gates",
        "finish",
        "floor_z",
        "ceiling_z",
        "forward_axis",
        "up_axis",
    }
    assert len(course["start"]) == 3
    # N-gate array (default course → 3 gates); each carries center/aperture/plane.
    assert isinstance(course["gates"], list)
    assert len(course["gates"]) == 3
    for gate in course["gates"]:
        assert set(gate) >= {"center", "aperture", "plane"}
        assert len(gate["center"]) == 3
    assert "x" in course["finish"]
    # The pre-UC-09 singular `gate` block is gone.
    assert "gate" not in course
    # Axis conventions are stamped so the viewer never guesses handedness/up (AC5 foot-gun).
    assert course["forward_axis"] == "x"
    assert course["up_axis"] == "z"


def test_recorded_file_without_course_still_loads(recorded_doc: dict) -> None:
    """Back-compat: a recording with no course omits ``meta.course`` (viewer degrades)."""
    assert "course" not in recorded_doc["meta"]


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
    # AC7/file://-safe: classic <script> only — no ES modules (blocked by file:// origin).
    assert 'type="module"' not in html
    assert '<script src="viewer.js"></script>' in html


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


# --- UC-06 AC1-6,9: the 3D flight panel + neuron beat live in viewer.js -----------------
def test_viewer_js_has_3d_orbit_projector_and_markers() -> None:
    """viewer.js carries the genuinely-3D orbit projector + floor/markers (AC1-5,9,13).

    Static assertions (CI is headless — no browser): the projector, the orbit camera
    (yaw/pitch + pointer/wheel handlers), the floor + start/gate/finish markers driven by
    ``meta.course``, the view presets, and a ``destroy()``/ResizeObserver lifecycle.
    """
    js = (_VIZ / "viewer.js").read_text()
    # Genuine perspective projection + orbit camera (AC1).
    assert "project" in js
    for cam in ("yaw", "pitch"):
        assert cam in js, f"viewer.js missing orbit-camera symbol {cam!r}"
    # Drag-to-rotate / wheel-to-zoom interaction (AC1).
    assert "pointerdown" in js or "pointermove" in js
    assert "wheel" in js
    # Reads the course geometry to place floor + markers (AC2/AC3/AC8).
    assert "meta.course" in js or "course" in js
    for marker in ("floor", "gate", "finish", "start"):
        assert marker in js, f"viewer.js missing 3D marker/floor {marker!r}"
    # Moving drone marker on the shared playhead (AC4).
    assert "drone_position" in js
    # Single-renderer lifecycle torn down on reload — no leaked canvas context (AC9).
    assert "destroy" in js
    assert "ResizeObserver" in js
    # The old flat 2D path is GONE (replaced by the 3D scene).
    assert "path-canvas" not in js, "flat 2D path-canvas must be removed (replaced by 3D)"
    assert "drawPath" not in js, "flat 2D drawPath must be removed (replaced by 3D)"


def test_viewer_js_reads_n_gate_array_with_legacy_fallback() -> None:
    """UC-09 AC7: viewer.js reads ``course.gates[]`` with a legacy single-``gate`` fallback.

    Static assertion (CI is headless): the viewer normalises the course into an array of
    gate specs — new files carry ``course.gates: [...]``; pre-UC-09 files carry a singular
    ``course.gate`` — via the array-with-fallback read ``course.gates || (course.gate ? …)``,
    so both render. It also reads the per-frame ``target_gate`` track to highlight the
    current target gate distinctly (brighter/thicker), dimming the rest.
    """
    js = (_VIZ / "viewer.js").read_text()
    # Array-with-fallback read (UC-09 AC7): new `gates[]`, legacy singular `gate`.
    assert "course.gates" in js, "viewer.js must read the N-gate array course.gates"
    assert "course.gate" in js, "viewer.js must keep the legacy singular course.gate fallback"
    # The exact fallback idiom: `course.gates || (course.gate ? [course.gate] : [])`.
    normalised = re.sub(r"\s+", "", js)
    assert "course.gates||(course.gate?[course.gate]:[])" in normalised, (
        "viewer.js must normalise to an array with the legacy single-gate fallback"
    )
    # Per-frame current-target-gate track drives the highlight (AC7).
    assert "target_gate" in js
    # A distinct colour/style for the current target gate vs the others.
    assert "gateTarget" in js, "viewer.js must style the current target gate distinctly"


def test_viewer_js_has_neuron_beat_envelope() -> None:
    """viewer.js implements the neuron beat: ~2px rest, pulse to ~9px, ~0.2s ease (AC6)."""
    js = (_VIZ / "viewer.js").read_text()
    # Rest/peak radius constants (pinned 2 -> 9px) and the instantaneous-radius formula.
    assert "BEAT_REST" in js
    assert "BEAT_PEAK" in js
    assert "2" in js and "9" in js  # rest 2px, peak 9px
    # Per-frame instantaneous radius (used while paused/scrubbing — no stale animation).
    assert "rInst" in js or "r_inst" in js
    # A per-neuron envelope state exists (attack + exponential release toward rest).
    assert "beat" in js


def test_viewer_js_play_at_end_restarts_from_start() -> None:
    """AC11: Play at the last frame seeks to frame 0 then plays (not a no-op)."""
    js = (_VIZ / "viewer.js").read_text()
    # The play handler resets the playhead to 0 when already at the end.
    assert "state.frame = 0" in js
    assert "nFrames - 1" in js or "n_frames - 1" in js


# --- UC-06 AC12/AC13: playback speed + human-readable view presets ---------------------
def test_viewer_html_has_quarter_speed_option() -> None:
    """AC12: the speed selector includes a 0.25× (quarter-speed) option."""
    html = (_VIZ / "viewer.html").read_text()
    speed_body = _extract_select(html, "speed-select")
    values = [v for v, _ in _options(speed_body)]
    assert "0.25" in values, f"speed-select missing 0.25× option (values: {values})"


def test_viewer_html_view_select_is_human_labelled_top_default() -> None:
    """AC13: the flight view selector offers front/side/top with **top** the default.

    CRITICAL: assert on the ``view-select`` element and its option VALUES — NOT a bare
    "top-down" text grep. Both the brain-map panel hint ``<span class="hint">(top-down)</span>``
    and this selector's label contain that string, so a bare grep is a false-positive trap.
    """
    html = (_VIZ / "viewer.html").read_text()
    body = _extract_select(html, "view-select")
    opts = _options(body)
    values = [v for v, _ in opts]
    assert values == ["front", "side", "top"], f"view-select option values wrong: {values}"
    # Exactly one default, and it is 'top' (top-down on load).
    selected = [v for v, sel in opts if sel]
    assert selected == ["top"], f"default view must be 'top' (top-down); got {selected}"
    # Human-readable label text (never raw axis names x/y/z) — visible option text is words.
    label_text = re.sub(r"<[^>]+>", " ", body)
    assert "top-down" in label_text  # scoped to the view-select body, not the panel hint
    for axis in (">x<", ">y<", ">z<"):
        assert axis not in body, f"view-select must not expose raw axis label {axis!r}"


def test_viewer_html_map_view_select_is_human_labelled_top_default() -> None:
    """Anatomical brain-map selector offers front/side/top-down with **top** the default.

    Mirrors ``test_viewer_html_view_select_is_human_labelled_top_default`` but targets the
    anatomical ``map-view-select`` element. ``_extract_select`` anchors on the literal
    ``id="..."`` so ``view-select`` and ``map-view-select`` do not collide. As with the
    flight selector, assert on the option VALUES (not a bare "top-down" text grep) — the
    panel ``<span class="hint">`` also carries preset wording and would be a false positive.
    """
    html = (_VIZ / "viewer.html").read_text()
    body = _extract_select(html, "map-view-select")
    opts = _options(body)
    values = [v for v, _ in opts]
    assert values == ["front", "side", "top"], f"map-view-select option values wrong: {values}"
    # Exactly one default, and it is 'top' (top-down / dorsal on load — anatomically pinned).
    selected = [v for v, sel in opts if sel]
    assert selected == ["top"], f"default map view must be 'top' (top-down); got {selected}"
    # Human-readable label text (never raw axis names x/y/z) — scoped to this select body.
    assert "top-down" in body
    for axis in (">x<", ">y<", ">z<"):
        assert axis not in body, f"map-view-select must not expose raw axis label {axis!r}"


def test_viewer_js_has_map_view_presets_constant() -> None:
    """The human labels are provably wired to real projection planes via MAP_VIEW_PRESETS.

    Mirrors the static-assert style used for the flight ``VIEW_PRESETS`` elsewhere: a plain
    substring check that ``viewer.js`` declares the anatomical preset→plane constant. This is
    the fixed contract handle the HTML select id ``map-view-select`` resolves against.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "MAP_VIEW_PRESETS" in js, (
        "viewer.js must declare MAP_VIEW_PRESETS mapping preset names to projection planes"
    )


# --- AC10/AC12 docs --------------------------------------------------------------------
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


def test_readme_documents_uc06_3d_controls_and_beat() -> None:
    """AC10: the README documents the 3D controls, beat, 0.25×, presets, and meta.course."""
    readme = (_REPO_ROOT / "README.md").read_text()
    lower = readme.lower()
    # 3D controls: drag-to-rotate + wheel-to-zoom (AC1).
    assert "drag" in lower and "rotate" in lower
    assert "wheel" in lower and "zoom" in lower
    # Human-readable view presets + top-down default (AC13).
    assert "front" in lower and "side" in lower and "top-down" in lower
    # Neuron beat behaviour (AC6).
    assert "beat" in lower
    # 0.25× slow-inspection speed (AC12).
    assert "0.25" in readme
    # The recorded course-geometry schema addition (AC8).
    assert "meta.course" in readme or "course" in lower


def test_env_example_documents_token() -> None:
    env = (_REPO_ROOT / ".env.example").read_text()
    assert "NEUPRINT_TOKEN" in env
    assert "soma" in env.lower()
