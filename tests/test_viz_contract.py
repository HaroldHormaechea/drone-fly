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
  / trajectory / drone / presets / ``destroy``) and the play-at-end restart; the old flat 2D
  ``path-canvas`` / ``drawPath`` are gone; ``viewer.html`` offers a 0.25× speed option and a
  human-labelled ``view-select`` (front/side/top, top the default). "A right turn looks right"
  (AC5) is a documented MANUAL browser check.
* **UC-12 AC1,2,4,5,7,8** — the anatomical panel is now an MRI-style heatmap over a static,
  registered brain outline: the per-neuron dot cloud AND the UC-06 neuron beat are removed;
  the committed ``viz/brain_outline.js`` asset defines ``BRAIN_OUTLINE`` (offline, no URL) and
  is wired into both ``viewer.html`` (before ``viewer.js``) and ``viewer.js`` (with a
  missing-asset guard); the render path composites additive kernel-density splats
  (``globalCompositeOperation="lighter"``) through a "hot" colormap; a ``map-norm-select``
  toggles per-frame (default) vs global normalization; ``node --check`` parses both JS assets.
  The visual ACs (AC3 animate, AC4 registration, AC5 contrast change, AC6 per-plane
  projection) are documented MANUAL browser checks — not automated here.
* **AC10/AC12** — the README documents the record flags, the artifacts path, viewer usage,
  the coordinate provisioning, and the UC-06 3D flight controls / 0.25× / presets and the
  UC-12 MRI heatmap + outline asset; ``.env.example`` documents the token.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.env.config import CourseConfig, ObstacleSpec, PadSpec
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

    # (a2) UC-28 full-coverage fields: placement / region / display3d are additive, full-length,
    # and display3d is finite for EVERY neuron so the viewer renders all N on any plane (AC-2).
    assert set(positions) >= {"placement", "region", "display3d"}
    assert len(positions["placement"]) == n
    assert len(positions["region"]) == n
    assert len(positions["display3d"]) == n
    assert set(positions["placement"]) <= {"anatomical", "schematic", "computed"}
    assert np.isfinite(np.asarray(positions["display3d"], dtype=float)).all()
    assert all(len(p) == 3 for p in positions["display3d"])

    # (a3) UC-28 modality tag (AC-7): one tag per neuron, from the fixed recorded set (or "").
    modality = doc["meta"]["modality"]
    assert len(modality) == n
    assert set(modality) <= {"vision", "proprioceptive", "hunger", ""}

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
        # UC-28 additive contract fields.
        "display3d",
        "placement",
        "modality",
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


# --- UC-15 AC7: the viewer reads + draws obstacle pillars, degrading gracefully ---------
@pytest.fixture
def recorded_doc_with_obstacles(connectome: ConnectomeData, tmp_path: Path) -> dict:
    """A recorded episode whose course carries obstacle pillars (``meta.course.obstacles``)."""
    course = CourseConfig(
        obstacles=(
            ObstacleSpec(center=(3.25, 1.5), radius=0.3, height=2.5),
            ObstacleSpec(center=(4.75, -1.6), radius=0.4, height=2.0),
        )
    )
    return _record(connectome, tmp_path / "act_obs", course=course)


def test_recorded_file_carries_obstacle_geometry(recorded_doc_with_obstacles: dict) -> None:
    """AC7: an obstacle-carrying recording documents each pillar the viewer draws."""
    course = recorded_doc_with_obstacles["meta"]["course"]
    assert isinstance(course["obstacles"], list)
    assert len(course["obstacles"]) == 2
    for pillar in course["obstacles"]:
        assert set(pillar) >= {"center", "radius", "height"}
        assert len(pillar["center"]) == 2  # (x, y) axis; floor-anchored height gives the z-extent


def test_recorded_file_without_obstacles_omits_field(recorded_doc_with_course: dict) -> None:
    """AC7 back-compat: a course without pillars omits ``obstacles`` (viewer draws none)."""
    assert "obstacles" not in recorded_doc_with_course["meta"]["course"]


def test_viewer_js_draws_obstacle_pillars_with_graceful_degradation() -> None:
    """AC7: viewer.js has ``drawObstacles`` reading ``course.obstacles`` with an absent-field guard.

    Static assertion (CI is headless): the viewer defines an obstacle draw routine, reads the
    additive ``course.obstacles`` array, and guards it (``|| []``) so a recording without the
    field draws nothing rather than throwing.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "drawObstacles" in js, "viewer.js must define an obstacle draw routine"
    assert "obstacles" in js, "viewer.js must read the additive course.obstacles array"
    # Graceful degradation: the obstacle list is guarded so a field-absent file draws nothing.
    normalised = re.sub(r"\s+", "", js)
    assert "course.obstacles)||[]" in normalised or "obstacles||[]" in normalised, (
        "viewer.js must guard obstacles with `|| []` for graceful degradation"
    )
    # drawObstacles is actually invoked from the render path.
    assert "drawObstacles()" in js


# --- UC-21: pads carry a kind + the viewer draws pad discs and a visible obstacle footprint ---
@pytest.fixture
def recorded_doc_with_pads(connectome: ConnectomeData, tmp_path: Path) -> dict:
    """A recorded episode whose course carries recharge / repair / plain pads (meta.course.pads)."""
    course = CourseConfig(
        pads=(
            PadSpec(center=(1.0, 0.5), radius=0.4, rechargeable=True),
            PadSpec(center=(2.0, -0.5), radius=0.3, repairable=True),
            PadSpec(center=(3.0, 0.0), radius=0.2),
        )
    )
    return _record(connectome, tmp_path / "act_pads", course=course)


def test_recorded_pads_carry_kind(recorded_doc_with_pads: dict) -> None:
    """AC1: a pad-carrying recording documents each pad's floor geometry + display ``kind``."""
    pads = recorded_doc_with_pads["meta"]["course"]["pads"]
    assert isinstance(pads, list)
    assert len(pads) == 3
    for pad in pads:
        assert set(pad) >= {"center", "radius", "kind"}
        assert len(pad["center"]) == 2  # (x, y); floor-anchored, z is course floor_z
    assert [pad["kind"] for pad in pads] == ["recharge", "repair", "plain"]


def test_recorded_file_without_pads_omits_field(recorded_doc_with_course: dict) -> None:
    """AC5 back-compat: a course without pads omits ``pads`` (viewer draws none, no error)."""
    assert "pads" not in recorded_doc_with_course["meta"]["course"]


def test_viewer_js_draws_pads_by_kind_with_graceful_degradation() -> None:
    """AC2/AC3: viewer.js defines ``drawPads`` reading ``course.pads`` (guarded) + colours by kind.

    Static assertion (CI is headless): the viewer defines a pad draw routine, reads the additive
    ``course.pads`` array with a ``|| []`` guard so a field-absent file draws nothing, resolves a
    pad's ``kind`` to a colour, and is actually invoked from the render path.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "drawPads" in js, "viewer.js must define a pad draw routine"
    # Graceful degradation: buildFlightScene reads course.pads with a `|| []` guard.
    normalised = re.sub(r"\s+", "", js)
    assert "course.pads)||[]" in normalised, (
        "viewer.js must guard course.pads with `|| []` for graceful degradation"
    )
    # The pad's serialised `kind` is read to pick a colour.
    assert "kind" in js, "viewer.js must read each pad's `kind` to colour it"
    # drawPads is actually invoked from the render path.
    assert "drawPads()" in js


def test_viewer_js_pad_kind_colors_are_defined_and_distinct() -> None:
    """AC3: three distinct, documented pad-kind colours exist in the palette (+ obstacle colour)."""
    js = (_VIZ / "viewer.js").read_text()
    for token in ("padRecharge", "padRepair", "padPlain"):
        assert token in js, f"viewer.js palette must define {token!r}"
    # The chosen hexes, synced with the legend + README (distinct from start/gate/finish).
    for hex_color in ("#00e676", "#ff6d00", "#90a4ae"):
        assert hex_color in js, f"viewer.js must define pad colour {hex_color}"
    # Three distinct pad colours (no accidental collision).
    assert len({"#00e676", "#ff6d00", "#90a4ae"}) == 3


def test_viewer_js_shares_floordisc_between_pads_and_obstacles() -> None:
    """AC4: a shared ``floorDisc`` helper is invoked from BOTH draw routines (visibility fix).

    The user's "obstacles not shown" complaint is a visibility issue — the fix adds a
    floor-anchored base ring + filled disc (the shared ``floorDisc`` helper) to both pads and
    obstacles so both read against the floor grid. Assert the helper exists and both call it.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "function floorDisc" in js, "viewer.js must define the shared floorDisc helper"
    # floorDisc is invoked (>= 2 call sites: drawPads + drawObstacles), beyond its definition.
    assert js.count("floorDisc(") >= 2, "floorDisc must be invoked from both draw routines"
    # Both draw bodies reference the shared helper.
    draw_pads = js[js.index("function drawPads") :]
    assert "floorDisc(" in draw_pads[: draw_pads.index("\n  }")], "drawPads must call floorDisc"
    draw_obstacles = js[js.index("function drawObstacles") :]
    obstacle_body = draw_obstacles[: draw_obstacles.index("\n  }")]
    assert "floorDisc(" in obstacle_body, "drawObstacles must call floorDisc (visibility fix)"


def test_viewer_html_legend_has_pad_kinds_and_obstacle() -> None:
    """AC3/AC4: the flight-3d legend gains a recharge / repair / plain pad + obstacle entry.

    Scoped to the ``#flight-3d-legend`` block so a colour appearing elsewhere can't false-positive.
    Colours + labels are synced to ``COURSE_COLORS`` in viewer.js.
    """
    html = (_VIZ / "viewer.html").read_text()
    m = re.search(
        r'<div[^>]*\bid="flight-3d-legend"[^>]*>(.*?)</div>',
        html,
        re.DOTALL,
    )
    assert m is not None, 'no <div id="flight-3d-legend"> in viewer.html'
    legend = m.group(1)
    # Three pad-kind swatches + labels, plus the obstacle swatch + label.
    for hex_color, label in (
        ("#00e676", "recharge"),
        ("#ff6d00", "repair"),
        ("#90a4ae", "plain"),
        ("#ba68c8", "obstacle"),
    ):
        assert hex_color in legend, f"flight-3d-legend missing swatch {hex_color}"
        assert label in legend, f"flight-3d-legend missing {label!r} label"


def test_viewer_js_anatomical_dots_and_beat_removed() -> None:
    """UC-12 AC1: the per-neuron dot cloud AND the UC-06 neuron-beat are gone.

    Supersedes the UC-06 ``test_viewer_js_has_neuron_beat_envelope`` (authorized test
    evolution — the plan removes dots+beat from the anatomical panel entirely; the heatmap
    "lights up" is the new beat, and the flight panel never carried one). Static assertion
    (CI is headless): none of the beat envelope's symbols survive in ``viewer.js``.
    """
    js = (_VIZ / "viewer.js").read_text()
    for beat_sym in (
        "BEAT_REST",
        "BEAT_PEAK",
        "BEAT_SPAN",
        "BEAT_RELEASE_TAU",
        "seedBeat",
        "updateBeat",
    ):
        assert beat_sym not in js, f"UC-12 removes the neuron beat: {beat_sym!r} must be gone"
    # The per-neuron beat envelope state and its instantaneous-radius formula are gone too.
    assert "state.beat" not in js, "state.beat must be removed (no per-neuron beat state)"
    assert "rInst" not in js and "r_inst" not in js, (
        "instantaneous beat-radius formula must be gone"
    )


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
        # UC-11 migrated recording from the removed `--record*` flags to config keys; the
        # README documents recording via the YAML `record` / `record_every` keys writing to
        # the per-run `training/<name>/recordings/` layout (replacing `artifacts/activations`).
        "record:",
        "record_every",
        "recordings",
        "viz/viewer.html",
        "NEUPRINT_TOKEN",
        "fetch_soma_positions.py",
    ):
        assert token in readme, f"README missing {token!r}"
    lower = readme.lower()
    assert "fallback" in lower  # computed-layout fallback documented
    assert "soma" in lower  # coordinate provisioning documented


def test_readme_documents_uc06_3d_controls() -> None:
    """AC10: the README documents the 3D flight controls, 0.25×, presets, and meta.course.

    UC-12 reconciliation: the anatomical panel no longer has a per-neuron "beat" (dots+beat
    were removed for the MRI heatmap), so the beat assertion is dropped here — the rewritten
    README no longer mentions it. The 3D *flight-panel* controls this test pins are untouched
    by UC-12 and still documented.
    """
    readme = (_REPO_ROOT / "README.md").read_text()
    lower = readme.lower()
    # 3D controls: drag-to-rotate + wheel-to-zoom (AC1).
    assert "drag" in lower and "rotate" in lower
    assert "wheel" in lower and "zoom" in lower
    # Human-readable view presets + top-down default (AC13).
    assert "front" in lower and "side" in lower and "top-down" in lower
    # 0.25× slow-inspection speed (AC12).
    assert "0.25" in readme
    # The recorded course-geometry schema addition (AC8).
    assert "meta.course" in readme or "course" in lower


def test_env_example_documents_token() -> None:
    env = (_REPO_ROOT / ".env.example").read_text()
    assert "NEUPRINT_TOKEN" in env
    assert "soma" in env.lower()


# --- UC-12: MRI-style full-brain activation heatmap over a static outline ---------------
# CI is headless (no browser/JS runtime beyond `node --check` syntax), so UC-12 is validated
# statically: the committed outline asset, its wiring into both HTML and JS, the additive
# splat + colormap render path, and the per-frame/global normalization control. The visual
# ACs (AC3 animate / AC4 registration / AC5 contrast change / AC6 per-plane projection) are
# documented as MANUAL browser checks — see this test's coverage summary — and are NOT
# claimed as automated coverage here.


def test_brain_outline_asset_exists_and_is_offline() -> None:
    """AC4/AC7: a committed static brain-outline asset ships, defines ``BRAIN_OUTLINE``, and
    carries no network URL (dependency-free, ``file://``-loadable — no fetch/build step)."""
    asset = _VIZ / "brain_outline.js"
    assert asset.is_file(), "missing viz/brain_outline.js (committed static outline asset)"
    src = asset.read_text()
    # Declares the global the viewer consumes, via a classic (non-module) const.
    assert "BRAIN_OUTLINE" in src, "brain_outline.js must define BRAIN_OUTLINE"
    assert "type=" not in src  # it's pure JS, not markup — no module wiring lives here
    # No network literal anywhere (AC7): neither bare scheme nor attribution URL.
    assert "https://" not in src and "http://" not in src, (
        "brain_outline.js must contain no http(s):// literal (offline, file://-safe)"
    )
    # Registered to the soma coordinate frame by construction: voxel-space + per-plane polys.
    assert "voxel_space" in src
    assert "planes" in src and "bbox3d" in src


def test_brain_outline_referenced_by_html_and_js() -> None:
    """AC4/AC7: the outline asset is wired into BOTH the page and the viewer.

    ``viewer.html`` loads ``brain_outline.js`` via a classic ``<script>`` placed on the line
    *immediately before* ``viewer.js`` (so the global exists first); ``viewer.js`` consumes
    ``BRAIN_OUTLINE``.
    """
    html = (_VIZ / "viewer.html").read_text()
    assert '<script src="brain_outline.js"></script>' in html, (
        "viewer.html must load brain_outline.js via a classic <script>"
    )
    # brain_outline.js must precede viewer.js (the const must be defined before the viewer runs).
    assert html.index("brain_outline.js") < html.index('src="viewer.js"'), (
        "brain_outline.js must be loaded before viewer.js"
    )
    js = (_VIZ / "viewer.js").read_text()
    assert "BRAIN_OUTLINE" in js, "viewer.js must consume the BRAIN_OUTLINE global"
    # AC7/AC8: missing-asset guard — degrade, never a ReferenceError.
    assert "typeof BRAIN_OUTLINE" in js, (
        "viewer.js must guard `typeof BRAIN_OUTLINE` (degrade if absent)"
    )


def test_viewer_js_renders_additive_splat_heatmap() -> None:
    """AC1/AC2: the anatomical panel is an additive kernel-density splat heatmap through a
    colormap — not a per-neuron dot cloud."""
    js = (_VIZ / "viewer.js").read_text()
    # Additive compositing of the splats ("lighter" = source + destination).
    assert "globalCompositeOperation" in js
    assert '"lighter"' in js or "'lighter'" in js, "splats must be composited additively (lighter)"
    # A colormap maps normalized intensity → RGB (MRI/fMRI "hot" ramp).
    assert re.search(r"[Cc]olormap", js), "viewer.js must map intensity through a colormap"


def test_viewer_html_map_norm_select_per_frame_and_global() -> None:
    """AC5: the anatomical panel offers an intensity-normalization toggle with ``per-frame``
    and ``global`` options, **per-frame** the default.

    Mirrors the ``map-view-select`` assertion style: anchor on the ``map-norm-select`` element
    and assert on its option VALUES, not a bare text grep (the panel copy also says "per-frame").
    """
    html = (_VIZ / "viewer.html").read_text()
    body = _extract_select(html, "map-norm-select")
    opts = _options(body)
    values = [v for v, _ in opts]
    assert values == ["per-frame", "global"], f"map-norm-select option values wrong: {values}"
    # Exactly one default, and it is 'per-frame' (punchy "lights up" contrast on load).
    selected = [v for v, sel in opts if sel]
    assert selected == ["per-frame"], f"default normalization must be 'per-frame'; got {selected}"


def test_viewer_js_has_normalization_toggle_handler() -> None:
    """AC5: viewer.js wires the ``map-norm-select`` control to a per-frame/global mode."""
    js = (_VIZ / "viewer.js").read_text()
    assert "map-norm-select" in js, "viewer.js must read/handle the map-norm-select control"
    # State carries the current normalization mode; both modes are represented.
    assert "mapNorm" in js
    assert '"global"' in js and '"frame"' in js


@pytest.mark.parametrize("asset", ["viewer.js", "brain_outline.js"])
def test_viewer_assets_pass_node_check(asset: str) -> None:
    """AC7: the JS assets parse cleanly under ``node --check`` (syntax gate).

    ``node`` is the only JS runtime CI is guaranteed to have; if it is absent the check is
    skipped rather than silently passing.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available — cannot run `node --check`")
    result = subprocess.run(
        [node, "--check", str(_VIZ / asset)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"`node --check {asset}` failed:\n{result.stderr}"


# --- UC-28: full-coverage body-schematic render + modality overlay (static viewer checks) ------
def test_viewer_js_renders_display3d_full_coverage() -> None:
    """AC-1/AC-2: viewer.js prefers ``display3d`` (full-coverage finite render coords) for points.

    Static assertion (CI is headless): the point projector consumes ``positions.display3d`` so
    every neuron — real soma or schematic body — is placed on any plane, and the fixed-mode
    transform is widened to the UNION of the registered outline bbox and the display3d extent so
    body clusters placed outside the brain bbox are not clipped off-canvas.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "display3d" in js, "viewer.js must consume positions.display3d for full coverage"
    # Fixed-mode transform widened to outline.bbox3d ∪ display3d extent (else clusters clip).
    assert "bbox3d" in js
    assert "Math.min(" in js and "Math.max(" in js  # the union widening arithmetic


def test_viewer_js_distinguishes_schematic_from_anatomical() -> None:
    """AC-6: schematic (body) neurons are rendered distinguishably from real-anatomy neurons.

    Static assertion: the viewer reads ``positions.placement`` and stamps ``"schematic"`` splats
    at a reduced weight (a named constant) so a schematic dot never reads as bright as a real
    soma — visual distinction on top of the spatial separation the placement gives.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "placement" in js
    assert '"schematic"' in js or "'schematic'" in js
    assert "SCHEMATIC_SPLAT_WEIGHT" in js, "viewer.js must down-weight schematic splats (AC-6)"


def test_viewer_js_has_modality_overlay_toggle() -> None:
    """AC-7: viewer.js draws a modality-tag overlay ON TOP of the activation colormap, as a toggle.

    Static assertion: a ``drawModalityOverlay`` routine reads the ``modality-tag-select`` control
    and the per-neuron ``meta.modality`` tags, colouring exactly the recorded set
    (vision / proprioceptive / hunger). It overlays rings — it does not replace the hot colormap.
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "drawModalityOverlay" in js
    assert "modality-tag-select" in js
    assert "MODALITY_COLORS" in js
    for name in ("vision", "proprioceptive", "hunger"):
        assert name in js, f"viewer.js modality overlay must know {name!r}"
    # `damage` is documented as unavailable (no MaleCNS nociceptive label) — never faked.
    assert "damage" in js.lower()


def test_viewer_html_has_modality_tag_select_off_default() -> None:
    """AC-7: viewer.html offers the modality toggle with 'off' the default (never overrides colour).

    Anchored on the ``modality-tag-select`` element (mirrors the map-view/map-norm assertion
    style) so a bare text grep elsewhere can't produce a false positive.
    """
    html = (_VIZ / "viewer.html").read_text()
    body = _extract_select(html, "modality-tag-select")
    values = [v for v, _ in _options(body)]
    assert values == ["off", "vision", "proprioceptive", "hunger"], (
        f"modality-tag-select option values wrong: {values}"
    )
    selected = [v for v, sel in _options(body) if sel]
    assert selected == ["off"], f"modality overlay must default to 'off'; got {selected}"


def test_viewer_html_documents_damage_unavailable() -> None:
    """AC-7 honesty: the viewer legend documents damage/nociception as unavailable (no fake tag)."""
    html = (_VIZ / "viewer.html").read_text().lower()
    assert "damage" in html and "unavailable" in html
