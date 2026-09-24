"""AC1-13 — the viewer's JSON contract, static assets, and docs.

The static viewer (``viz/``) runs in a browser (no browser/JS runtime in CI), so its
behaviour is validated **statically** — the JSON contract it consumes plus source-file
assertions on ``viz/*`` and ``README.md`` — never by executing it (AC10). Verified:

* **AC5/AC8/AC9** — a recorded file carries every field the two rendered panels read: the
  spatial brain map (``positions.{coords2d,coords3d,has_position,source,projection}``) and
  the flight panel (``frames.actions`` + ``frames.drone_position`` + ``meta.action_layout``),
  all on one frame timeline. ``meta.roles`` is retained in the recording contract for
  back-compat but is no longer consumed by the viewer (the neurons×time heatmap it fed was
  removed in UC-34).
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


def _extract_js_function(js: str, name: str) -> str:
    """Return the full source of ``function <name>(...) { ... }`` via balanced-brace matching.

    Used both to isolate a pure function for hermetic execution under ``node`` (UC-59's
    ``bucketSomaless``) and to scope a static assertion to one function's body. The extracted
    functions contain no ``{``/``}`` inside string literals, so a simple depth counter is exact.
    """
    marker = f"function {name}"
    start = js.find(marker)
    assert start != -1, f"viewer.js has no `function {name}`"
    brace = js.find("{", start)
    assert brace != -1, f"`function {name}` has no body"
    depth = 0
    for i in range(brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start : i + 1]
    raise AssertionError(f"unbalanced braces extracting `function {name}`")


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

    # (b) meta.roles — one role per neuron, retained in the recording contract for
    # back-compat (AC9). No longer consumed by the viewer: UC-34 removed the neurons×time
    # heatmap panel that used it. The recorder still emits the field, so it stays validated.
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
    # Reads the schema fields it renders (AC5/AC8/AC9). ``meta.roles`` is intentionally
    # absent: UC-34 removed the neurons×time heatmap that consumed it, so viewer.js no longer
    # references the field (the recorder still emits it — see the JSON-contract check above).
    for field in (
        "schema_version",
        "positions",
        "coords2d",
        "coords3d",
        "has_position",
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


def test_viewer_has_no_neurons_by_time_heatmap() -> None:
    """UC-34 AC1/AC4: the neurons×time heatmap panel is fully gone — no lingering symbol in
    ``viewer.js`` and no panel/canvas in ``viewer.html``. The anatomical ``drawBrainMap`` MRI
    heatmap and its ``hotColormap`` are a *different* feature and must survive, so this guard
    targets the specific removed symbols (a bare ``heatmap`` grep would false-positive on the
    surviving anatomical-map code)."""
    js = (_VIZ / "viewer.js").read_text()
    html = (_VIZ / "viewer.html").read_text()
    for symbol in (
        "drawHeatmap",
        "buildHeatmap",
        "computeRowOrder",
        "state.heatmap",
        "state.rowOrder",
        "rowOrder",
        "ROLE_COLORS",
        "ROLE_RANK",
        "FALLBACK_COLOR",
        "roles",  # only the heatmap consumed meta.roles; viewer.js no longer references it
    ):
        assert symbol not in js, f"viewer.js still references removed heatmap symbol {symbol!r}"
    assert "heatmap-canvas" not in html
    # The removed panel's header/label is gone (its wording described neurons × time). Other
    # "neurons" mentions in the anatomical-map panel copy are legitimate and left intact.
    assert "activation heatmap" not in html.lower()
    assert "neurons × time" not in html and "neurons x time" not in html.lower()
    # The anatomical brain map (a different feature) must remain wired up.
    assert "drawBrainMap" in js


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


def test_viewer_js_weights_splats_by_activation_magnitude_from_rest() -> None:
    """The heatmap animates only if splat weight tracks activation MAGNITUDE RELATIVE TO REST,
    not the raw uint8 code.

    The recorder quantizes real activation ∈ [-1, 1] to a code ∈ [0, 255] with rest (0) → code
    ~127. Weighting a splat by ``act[i] / 255`` therefore pins every resting neuron at ~0.5
    brightness, so the whole map glows at a constant half-lit floor and the frame-to-frame
    signal (a fraction of a percent of that floor) is invisible. The weight must instead be the
    magnitude of the dequantized activation — ``|code * scale + offset|`` — sourced from the
    file's own ``activation_scale`` / ``activation_offset`` meta (with a canonical fallback), so
    a resting neuron reads dark and only deviations light up and fade over time.
    """
    js = (_VIZ / "viewer.js").read_text()
    # Weight is magnitude of the dequantized activation, not the raw code / 255.
    assert "act[i] / 255" not in js and "act[i]/255" not in js, (
        "viewer.js must NOT weight splats by the raw uint8 code (rest would glow at ~0.5)"
    )
    assert re.search(r"Math\.abs\(\s*act\[i\]\s*\*\s*ascale\s*\+\s*aoffset\s*\)", js), (
        "viewer.js must weight splats by |act[i] * scale + offset| (activation magnitude from rest)"
    )
    # Scale/offset are sourced from the recording meta (not hardcoded), with a legacy fallback.
    assert "activation_scale" in js and "activation_offset" in js, (
        "viewer.js must read activation_scale/activation_offset from the recording meta"
    )


def test_viewer_js_applies_per_neuron_temporal_autogain() -> None:
    """Magnitude-from-rest alone leaves the map nearly constant frame-to-frame (the activations
    drift slowly), so the splat weight is additionally auto-gained per neuron: each neuron's
    magnitude is stretched to its OWN episode min→max range (with a floored divisor so
    quantization noise on near-rest neurons is not blown up), making slow swings fill the
    dark→bright range and visibly animate.
    """
    js = (_VIZ / "viewer.js").read_text()
    # A per-recording gain pre-pass exists and is memoized (not recomputed per view).
    assert "ensureActivationGain" in js, "viewer.js must compute a per-neuron temporal auto-gain"
    assert "actGain" in js, "the auto-gain must be memoized on state (per recording, not per view)"
    # The divisor is floored by a named constant so near-rest noise is not amplified.
    assert "MAP_GAIN_MIN_RANGE" in js, "auto-gain divisor must be floored (MAP_GAIN_MIN_RANGE)"
    # The stamp weight applies the gain: clamp01((mag - lo) * inv).
    assert re.search(r"\(\s*mag\s*-\s*gainLo\[i\]\s*\)\s*\*\s*gainInv\[i\]", js), (
        "stampFrame must apply the per-neuron auto-gain: (mag - gainLo[i]) * gainInv[i]"
    )


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
    """AC-1/AC-2 (UC-60): viewer.js still consumes ``display3d`` for full-coverage projected
    points, but the fixed-mode brain transform is now sized to the registered outline bounding
    box **ALONE** — UC-59's union with the out-of-outline point bounds is dropped (AC2), because
    out-of-boundary neurons are relocated to the boxes rather than stretched into the canvas.

    Supersedes the UC-28/UC-59 assertion that the transform was *widened* to the union of the
    outline bbox and the display3d extent. Static assertion (CI is headless).
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "display3d" in js, "viewer.js must consume positions.display3d for full coverage"
    cache_src = _extract_js_function(js, "ensureMapCache")
    # Fixed transform sized to the outline bbox ALONE: uMin/uMax/vMin/vMax read straight off
    # outline.bbox3d min/max for the plane axes — no union widening with point bounds (AC2).
    assert "bbox3d" in cache_src, "ensureMapCache fixed branch must size to outline.bbox3d"
    norm = re.sub(r"\s+", "", cache_src)
    assert "uMin=mn[a0],uMax=mx[a0],vMin=mn[a1],vMax=mx[a1]" in norm, (
        "UC-60: the fixed transform must be sized to the outline bbox alone (no union widening)"
    )


def test_viewer_js_excludes_out_of_boundary_from_brain_stamp() -> None:
    """UC-60 AC3/AC4: the brain-map stamp is gated by the BOUNDARY partition — only in-boundary
    neurons with a projected point splat into the brain map; out-of-boundary (or null-point)
    neurons are relocated to the tagged boxes and contribute no splat.

    Supersedes UC-59's ``test_viewer_js_excludes_schematic_from_brain_stamp`` (authorized test
    evolution): exclusion is no longer keyed on a ``"schematic"`` placement tag (``isSchematic``
    is gone) but on ``partitionByBoundary``'s per-neuron ``inside[]`` flag. The UC-28 schematic
    splat-weight constant stays removed. Static assertion (CI is headless).
    """
    js = (_VIZ / "viewer.js").read_text()
    # The UC-28 schematic splat weight stays gone (UC-59 removed it; the boxes carry that signal).
    assert "SCHEMATIC_SPLAT_WEIGHT" not in js, (
        "the schematic splat-weight constant must stay removed"
    )
    # The old placement-tag masking is gone — exclusion is boundary-based now.
    assert "isSchematic" not in js, "UC-60 replaces isSchematic masking with the boundary partition"
    assert "anatPts" not in js, "the old anatomical-points buffer must be gone"
    cache_src = _extract_js_function(js, "ensureMapCache")
    # ensureMapCache reads the boundary partition (single source of truth) ...
    assert "ensurePartition()" in cache_src, "ensureMapCache must consume the boundary partition"
    norm = re.sub(r"\s+", "", cache_src)
    # ... and stamps a neuron only when it is inside the boundary AND has a projected point.
    assert "if(!inside[i]||!p)continue" in norm, (
        "ensureMapCache must gate the stamp on inside[i] && points[i] != null (boundary partition)"
    )


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


# NOTE (UC-60 AC10): ``test_viewer_html_documents_damage_unavailable`` was REMOVED here — the
# descriptive brain-map legend (which carried the "damage/nociception: unavailable" text) is deleted
# from viewer.html to declutter the shortened panel. The honesty guarantee is preserved by
# ``test_viewer_js_has_modality_overlay_toggle`` (below), which greps the retained viewer.js
# comment.


# --- UC-59: three-zone layout + animated tagged soma-less boxes -------------------------
# CI is headless (no browser), so UC-60's structure/wiring is validated statically and the
# boundary-partition logic is proven by EXECUTING the pure ``partitionByBoundary`` (plus its two
# module-siblings ``pointInPolygon`` / ``pointInBBox``) under `node`. UC-60 RETIRED
# ``bucketSomaless``: box membership is now position-based (outside the brain outline → boxed),
# not placement-tag based. The visual verification (AC12 — a real render of the shortened brain +
# relocated boxes) CANNOT run in this sandbox (no chromium/chrome/playwright) and is a documented
# MANUAL step (see the coverage summary + PR).

_SOMALESS_TITLES = ("vision (external)", "proprioceptive", "hunger", "other (untagged)")

# A synthetic axis-aligned square outline in voxel space, reused by the partition cases below.
# "inside" points fall within it (stay in the brain map); "outside" points are relocated to boxes.
_SQUARE = [[0, 0], [10, 0], [10, 10], [0, 10]]
_SQUARE_BOUNDARY = {"kind": "polygon", "polygon": _SQUARE}
_IN = [5, 5]  # clearly inside _SQUARE
_OUT = [50, 50]  # clearly outside _SQUARE


def _run_node_json(src: str, tmp_path: Path, what: str):
    """Write ``src`` to a temp .mjs, execute it under node, and return the parsed JSON stdout."""
    node = shutil.which("node")
    if node is None:
        pytest.skip(f"node not available — cannot execute {what}")
    script = tmp_path / f"{what}_driver.mjs"
    script.write_text(src)
    result = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"node execution of {what} failed:\n{result.stderr}"
    return json.loads(result.stdout)


def _run_partition(cases: list[dict], tmp_path: Path) -> list[dict]:
    """Execute the *real* ``partitionByBoundary`` from viewer.js under node on synthetic inputs.

    Concatenates the three pure, self-contained module siblings straight from source
    (``pointInPolygon`` + ``pointInBBox`` + ``partitionByBoundary`` — none closes over module
    scope, so they run under node in isolation) and drives ``partitionByBoundary(points, boundary,
    modality)`` per case, returning the parsed ``{inside, buckets}`` objects. This is the hermetic
    proof of AC3/AC5/AC13 grouping semantics — no browser, no fixture file.
    """
    js = (_VIZ / "viewer.js").read_text()
    src = (
        _extract_js_function(js, "pointInPolygon")
        + "\n"
        + _extract_js_function(js, "pointInBBox")
        + "\n"
        + _extract_js_function(js, "partitionByBoundary")
        + "\nconst CASES = "
        + json.dumps(cases)
        + ";\nconst out = CASES.map((c) => partitionByBoundary(c.points, c.boundary, c.modality));"
        + "\nprocess.stdout.write(JSON.stringify(out));\n"
    )
    return _run_node_json(src, tmp_path, "partitionByBoundary")


def test_point_in_polygon_against_real_outline_and_square(tmp_path: Path) -> None:
    """AC13: the pure ``pointInPolygon`` ray-cast test is correct against the REAL registered
    ``BRAIN_OUTLINE`` polygons (all three views) and a synthetic square — executed under node.

    Concatenates the real ``brain_outline.js`` asset with the extracted ``pointInPolygon`` so the
    test runs against the shipped concave polygons, not a stand-in. Checks: each plane's polygon
    centroid is inside; a far point is outside; the square's inside/outside/near-edge behaviour is
    correct; degenerate (<3 vertices) and null inputs return ``false`` (no throw).
    """
    js = (_VIZ / "viewer.js").read_text()
    outline_src = (_VIZ / "brain_outline.js").read_text()
    driver = (
        outline_src
        + "\n"
        + _extract_js_function(js, "pointInPolygon")
        + "\nfunction centroid(poly){let x=0,y=0;for(const p of poly){x+=p[0];y+=p[1];}"
        + "return [x/poly.length, y/poly.length];}"
        + "\nconst square = "
        + json.dumps(_SQUARE)
        + ";"
        + "\nconst res = {};"
        + "\nfor (const k of ['top','front','side']) {"
        + "\n  const poly = BRAIN_OUTLINE.planes[k].polygon;"
        + "\n  res[k] = { centroidInside: pointInPolygon(centroid(poly), poly),"
        + "\n             farOutside: pointInPolygon([1e9, 1e9], poly) };"
        + "\n}"
        + "\nres.square = {"
        + "\n  inside: pointInPolygon([5,5], square),"
        + "\n  outside: pointInPolygon([50,50], square),"
        + "\n  nearInside: pointInPolygon([0.01,5], square),"
        + "\n  nearOutside: pointInPolygon([-0.01,5], square),"
        + "\n  onEdgeIsBool: typeof pointInPolygon([5,0], square) === 'boolean',"
        + "\n  degenerate: pointInPolygon([1,1], [[0,0],[2,2]]),"
        + "\n  nullPt: pointInPolygon(null, square),"
        + "\n};"
        + "\nprocess.stdout.write(JSON.stringify(res));\n"
    )
    res = _run_node_json(driver, tmp_path, "pointInPolygon")
    for plane in ("top", "front", "side"):
        assert res[plane]["centroidInside"] is True, f"{plane} outline centroid must be inside"
        assert res[plane]["farOutside"] is False, f"a far point must be outside the {plane} outline"
    sq = res["square"]
    assert sq["inside"] is True and sq["outside"] is False
    assert sq["nearInside"] is True, "a point just inside an edge must read inside"
    assert sq["nearOutside"] is False, "a point just outside an edge must read outside"
    assert sq["onEdgeIsBool"] is True, "an on-edge point must return a deterministic boolean"
    assert sq["degenerate"] is False, "a <3-vertex polygon must be treated as no boundary (false)"
    assert sq["nullPt"] is False, "a null point must be outside (false), never a throw"


def test_partition_relocates_out_of_boundary_neurons_under_node(tmp_path: Path) -> None:
    """AC3/AC5/AC13: every OUT-OF-BOUNDARY (or null-point) neuron lands in EXACTLY one titled box;
    in-boundary neurons stay in the brain map and are never boxed.

    Executes the real ``partitionByBoundary`` under node: ``inside[i]`` is true only for a present
    point inside the boundary; outside/null points are relocated, each into exactly one bucket, with
    the documented titles/order; no in-boundary neuron leaks into a box.
    """
    # i0 inside/vision→stays · i1 outside/vision→box · i2 outside/proprioceptive→box
    # · i3 null/""→box(other) · i4 outside/"banana"(legacy)→box(other) · i5 inside/None→stays
    points = [_IN, _OUT, _OUT, None, [99, 99], _IN]
    modality = ["vision", "vision", "proprioceptive", "", "banana", None]
    [res] = _run_partition(
        [{"points": points, "boundary": _SQUARE_BOUNDARY, "modality": modality}], tmp_path
    )
    buckets = res["buckets"]

    # In/out flags: only the two inside points stay in the brain.
    assert res["inside"] == [True, False, False, False, False, True]

    # Titles + order: hunger omitted (zero members); others in the fixed order.
    assert [b["title"] for b in buckets] == [
        "vision (external)",
        "proprioceptive",
        "other (untagged)",
    ]
    by_title = {b["title"]: b["indices"] for b in buckets}
    assert by_title["vision (external)"] == [1]
    assert by_title["proprioceptive"] == [2]
    # A null-point neuron AND an unknown/legacy tag both fall into the catch-all — none dropped.
    assert by_title["other (untagged)"] == [3, 4]

    # Partition property: every relocated (inside==False) index appears in exactly one bucket; no
    # in-boundary neuron leaks in; no duplicates.
    relocated_idx = {i for i, ins in enumerate(res["inside"]) if not ins}
    all_indices = [i for b in buckets for i in b["indices"]]
    assert sorted(all_indices) == sorted(relocated_idx)
    assert len(all_indices) == len(set(all_indices)), "a neuron appeared in more than one bucket"


def test_partition_full_tagset_fixed_order_under_node(tmp_path: Path) -> None:
    """AC5: with all tags present (all neurons out-of-boundary) the four documented boxes appear in
    the fixed order.

    Also proves ``null``/``undefined``/unknown modality values all collapse into the single
    catch-all box.
    """
    points = [_OUT] * 6  # all outside the square → all relocated
    # vision, proprioceptive, hunger, then "" / null / unknown → all "other (untagged)".
    modality = ["vision", "proprioceptive", "hunger", "", None, "legacy-tag"]
    [res] = _run_partition(
        [{"points": points, "boundary": _SQUARE_BOUNDARY, "modality": modality}], tmp_path
    )
    assert res["inside"] == [False] * 6
    assert [b["title"] for b in res["buckets"]] == list(_SOMALESS_TITLES)
    other = next(b for b in res["buckets"] if b["title"] == "other (untagged)")
    assert other["indices"] == [3, 4, 5]


def test_partition_bbox_boundary_under_node(tmp_path: Path) -> None:
    """AC9 degrade #1: a view whose outline has no polygon uses a ``{kind:"bbox"}`` boundary; the
    partition then routes the in/out test through ``pointInBBox`` (inclusive on the edges)."""
    bbox = {"kind": "bbox", "bbox": {"minX": 0, "minY": 0, "maxX": 10, "maxY": 10}}
    # inside · on-corner (inclusive → inside) · outside-x · outside-y
    points = [[5, 5], [10, 10], [11, 5], [5, 11]]
    modality = ["vision", "vision", "hunger", None]
    [res] = _run_partition([{"points": points, "boundary": bbox, "modality": modality}], tmp_path)
    assert res["inside"] == [True, True, False, False]
    # Only the two out-of-bbox neurons are boxed: hunger (i2) and other/None (i3).
    assert [b["title"] for b in res["buckets"]] == ["hunger", "other (untagged)"]
    by_title = {b["title"]: b["indices"] for b in res["buckets"]}
    assert by_title["hunger"] == [2]
    assert by_title["other (untagged)"] == [3]


def test_partition_omits_zero_member_and_empty_cases_under_node(tmp_path: Path) -> None:
    """AC5/AC9 edge cases: zero-member buckets omitted; all-inside and ``boundary=null`` → all
    inside with NO buckets; no-modality → everything relocated into the catch-all."""
    cases = [
        # (a) only vision out-of-boundary → single box, others omitted.
        {"points": [_OUT, _OUT], "boundary": _SQUARE_BOUNDARY, "modality": ["vision", "vision"]},
        # (b) all neurons inside the boundary → no boxes at all.
        {"points": [_IN, _IN], "boundary": _SQUARE_BOUNDARY, "modality": ["vision", "hunger"]},
        # (c) boundary disabled (null) → every neuron stays inside, buckets = [] (degrade #2).
        {"points": [_OUT], "boundary": None, "modality": ["vision"]},
        # (d) out-of-boundary present but NO modality array → all fall into the catch-all.
        {"points": [_OUT, _OUT], "boundary": _SQUARE_BOUNDARY, "modality": None},
    ]
    only_vision, all_inside, no_boundary, no_modality = _run_partition(cases, tmp_path)

    assert [b["title"] for b in only_vision["buckets"]] == ["vision (external)"]
    assert only_vision["buckets"][0]["indices"] == [0, 1]

    assert all_inside["inside"] == [True, True]
    assert all_inside["buckets"] == []

    # boundary=null → every neuron kept in the brain map, zero buckets.
    assert no_boundary["inside"] == [True]
    assert no_boundary["buckets"] == []

    assert [b["title"] for b in no_modality["buckets"]] == ["other (untagged)"]
    assert no_modality["buckets"][0]["indices"] == [0, 1]


def test_viewer_html_has_three_zone_layout() -> None:
    """AC1/AC2/AC3: brain top-left ‖ actions top-right, full-width 3D flight below.

    Structural (CI is headless): the two upper panels live inside ``.top-zones`` (brain first =
    left, actions second = right) and the 3D flight scene lives in the ``.bottom-zone`` after
    them. Index ordering proves the zone membership without executing the layout.
    """
    html = (_VIZ / "viewer.html").read_text()
    i_top = html.find('class="top-zones"')
    i_bottom = html.find('class="bottom-zone"')
    assert i_top != -1, "viewer.html must define a .top-zones container"
    assert i_bottom != -1, "viewer.html must define a .bottom-zone container"
    assert i_top < i_bottom, ".top-zones must come before .bottom-zone"

    i_brain = html.find('id="brain-canvas"')
    i_actions = html.find('id="actions-canvas"')
    i_flight = html.find('id="flight-canvas"')
    for name, idx in (
        ("brain-canvas", i_brain),
        ("actions-canvas", i_actions),
        ("flight-canvas", i_flight),
    ):
        assert idx != -1, f"viewer.html missing #{name}"

    # Brain (top-left) and actions (top-right) are BOTH inside the top zone, brain first.
    assert i_top < i_brain < i_actions < i_bottom, (
        "top zone must hold brain (left) then actions (right), both above the bottom zone"
    )
    # The 3D flight scene is the full-width bottom zone — a SEPARATE zone from the actions (AC3).
    assert i_flight > i_bottom, "flight-canvas (3D scene) must live in the bottom zone, not the top"


def test_viewer_css_bottom_zone_full_width_brain_dynamic_and_panels_equal_height() -> None:
    """AC1/AC4/AC8 (UC-60): the bottom 3D zone spans full width; the brain canvas is NO LONGER a
    forced ``1/1`` square (its aspect is JS-driven to hug the outline — AC1); and the two top
    panels are equal height via ``.top-zones { align-items: stretch }`` (AC8).

    Supersedes UC-59's "width-driven square" check (authorized test evolution). Structural CSS
    assertion.
    """
    css = (_VIZ / "viewer.css").read_text()
    norm = re.sub(r"\s+", " ", css)
    # Two-column top grid that reflows to a single column at narrow widths (AC2).
    assert ".top-zones" in css and "grid-template-columns" in css
    assert re.search(
        r"@media[^{]*max-width[^{]*\{[^}]*\.top-zones[^}]*grid-template-columns:\s*1fr", norm
    ), ".top-zones must collapse to a single column at narrow widths (AC2 reflow)"
    # AC8: equal-height top panels — the top-zones row stretches both panels to the taller.
    assert re.search(r"\.top-zones\s*\{[^}]*align-items:\s*stretch", norm), (
        ".top-zones must use align-items: stretch for equal-height top panels (AC8)"
    )
    # AC1: #brain-canvas still declares an aspect-ratio (JS overrides it per view for the
    # outline-hugging shape) but is NO LONGER a forced 1/1 square.
    assert re.search(r"#brain-canvas[^}]*aspect-ratio", norm), (
        "#brain-canvas must still declare an aspect-ratio (JS-driven outline-hugging default)"
    )
    assert not re.search(r"#brain-canvas[^}]*aspect-ratio:\s*1\s*/\s*1", norm), (
        "UC-60: #brain-canvas must no longer be a forced 1/1 square (AC1)"
    )
    # Full-width 3D flight canvas.
    assert re.search(r"#flight-canvas[^}]*width:\s*100%", norm), "#flight-canvas must be full width"
    # Soma-box strip is present.
    assert ".somaless-boxes" in css and ".soma-box" in css


def test_viewer_html_has_somaless_boxes_host_and_js_documents_titles() -> None:
    """AC5/AC10 (UC-60): the tagged-boxes host exists (empty — JS fills it per view) and the four
    box titles are authored in ``partitionByBoundary`` (the single source of truth).

    UC-60 removed the descriptive brain-map legend that previously *also* carried the titles in
    ``viewer.html`` (AC10 declutter), so the titles now live ONLY in JS and are injected into the
    boxes at runtime — they are no longer static HTML text. This supersedes the UC-59 assertion
    that the legend copy mirrored the titles (authorized test evolution).
    """
    html = (_VIZ / "viewer.html").read_text()
    js = (_VIZ / "viewer.js").read_text()
    assert 'id="somaless-boxes"' in html, "viewer.html must host the soma-less tagged boxes"
    src = _extract_js_function(js, "partitionByBoundary")
    for title in _SOMALESS_TITLES:
        assert title in src, f"partitionByBoundary must title a box {title!r}"


def test_viewer_js_wires_drawboxes_into_renderall() -> None:
    """AC6/AC8: the tagged boxes animate on the shared timeline — drawBoxes runs in renderAll.

    Static assertion: ``renderAll`` (which fires on play/scrub/speed) invokes ``drawBoxes()``,
    which reuses ``ensureActivationGain`` + ``hotColormap`` (the exact signal the old splat
    carried), and the boxes are (re)built per recording via ``buildSomalessBoxes``.
    """
    js = (_VIZ / "viewer.js").read_text()
    render_all = _extract_js_function(js, "renderAll")
    assert "drawBoxes()" in render_all, "renderAll must call drawBoxes() (AC6/AC8 animation)"
    # drawBoxes reuses the existing magnitude-from-rest gain + hot colormap (preserved signal).
    draw_boxes = _extract_js_function(js, "drawBoxes")
    assert "ensureActivationGain()" in draw_boxes
    assert "hotColormap(" in draw_boxes
    # The boxes are rebuilt for each loaded recording.
    assert "function buildSomalessBoxes" in js
    assert "buildSomalessBoxes()" in js, "buildSomalessBoxes must be invoked on load"


def test_viewer_js_boxes_are_responsive_without_resize_loop() -> None:
    """AC1 (responsive sizing) without the fixed-canvas pitfall: a DPR-aware backing store + a
    guarded ResizeObserver that cannot feed back into layout (canvases have a backing-store-
    independent CSS box, so re-deriving the backing store never changes the CSS box).
    """
    js = (_VIZ / "viewer.js").read_text()
    assert "function resizeBackingStore" in js, "viewer.js must define the shared DPR-aware sizer"
    assert "resizeBackingStore(" in js
    assert "ResizeObserver" in js  # (also asserted by the 3D-panel test; kept local for clarity)


def test_screenshot_helper_exists_and_parses_but_is_not_imported(tmp_path: Path) -> None:
    """AC10: the dev-only headless-render helper ships under scripts/, parses under node, and is
    NEVER imported by the test suite (keeping the gate hermetic).

    The real render (AC10) cannot be produced in THIS sandbox (no chromium/chrome/playwright), so
    the helper degrades honestly and the visual inspection is a documented MANUAL step referenced
    in the PR — not asserted here. We only prove the helper exists and is syntactically valid.
    """
    helper = _REPO_ROOT / "scripts" / "screenshot_viewer.mjs"
    assert helper.is_file(), "missing scripts/screenshot_viewer.mjs (dev-only render helper)"
    # It must NOT leak into the hermetic gate: no OTHER pytest module imports/runs it. (This file
    # is excluded — it names the helper only in prose/assertions above, never executing it.)
    this_file = Path(__file__).resolve()
    tests_dir = _REPO_ROOT / "tests"
    for test_file in tests_dir.rglob("*.py"):
        if test_file.resolve() == this_file:
            continue
        text = test_file.read_text()
        assert "screenshot_viewer" not in text, (
            f"{test_file.name} references screenshot_viewer — the helper must stay out of the "
            "hermetic pytest gate (it needs a real browser; run it manually for AC10)"
        )
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available — cannot `node --check` the helper")
    result = subprocess.run([node, "--check", str(helper)], capture_output=True, text=True)
    assert result.returncode == 0, f"`node --check screenshot_viewer.mjs` failed:\n{result.stderr}"
