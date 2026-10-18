"""Liftoff's stored recordings (UC71 AC2): parsing, catalogue linking, completeness, inventory.

Liftoff keeps every saved recording as one XML file under
``%USERPROFILE%\\AppData\\LocalLow\\LuGus Studios\\Liftoff\\Recordings\\<Mode>\\`` (root
``LocalGhostStatesRecording``; format documented in ``docs/harness/liftoff-captures.md`` § AC1a).
The pose trajectory is ``statesByte``: base64 of little-endian ``float32`` records of 12 values
(``stateLayout`` ``POSITION_STICKS``): Unity-world position x y z, Unity quaternion x y z w, sticks
(throttle, yaw, pitch, roll) and the game time, at 10 Hz. The recordings are only ever read; nothing
is copied into the repo.

**Catalogue match**, in order: the track + race uuids against ``entries[].uuids``; else the map
(``environment``) + the entry's display name as the recording name's prefix; else ``unmatched``
(listed, never capturable). A FreeFlight / StuntMode recording (no race) matches by its track uuid
only when exactly one entry uses that track.

**Completeness** (``complete`` | ``incomplete`` | ``unknown``), first source available:

1. ``laps`` - the recording's ``lapTimes`` (completed laps) against the entry's required laps; a
   race time of ``00_00_000`` in the name means not finished (the format has no checkpoint events);
2. ``pose`` - the pose trajectory replayed through the env pass rule
   (:func:`~drone_fly.sim.pool.gate_crossed` + ``RouteProgress``) on the catalogue course;
3. ``capture-meta`` - the completeness of a capture already made from this recording (sha-matched
   ``meta.json``).

``unknown`` only when none applies (no race / no catalogue match / no track files and no capture).

**Camera.** Each recording names its drone and camera (``droneConfiguration/Blueprint/camera``:
``id`` and ``angle`` = the uptilt); the field of view is not stored. Replay renders the recording's
own camera (AC1b, P1). A recording whose uptilt differs from the preset camera's by more than
:data:`UPTILT_TOL_DEG` (e.g. the Borrum 180 at 30 deg vs the preset's 20) is ``train-only`` on a
train map (captured with its own uptilt in ``meta.json``) and ``no`` on a val / held-out map:
model selection and evaluation stay on the deployed drone's camera (ADR 0010).

**Source.** ``ai-attempt`` when the name matches the liftoff-episodes pattern
``<MapName>-model-<model>-attempt-<n>`` (:data:`AI_ATTEMPT_RE`), otherwise ``user``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

STATE_FLOATS = 12
STATE_LAYOUT = "POSITION_STICKS"
AI_ATTEMPT_RE = re.compile(r"-model-.+-attempt-\d+$")
RACE_TIME_RE = re.compile(r" - (\d{2})_(\d{2})_(\d{3})(?: - |$)")
UPTILT_TOL_DEG = 0.5
#: a pose step longer than this (m) is treated as a respawn when replaying the pass rule
RESPAWN_JUMP_M = 20.0
GAP_S = 1.0


class RecordingError(HarnessConfigError):
    """A recording file that cannot be parsed."""


def default_recordings_dir() -> Path:
    base = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(base) / "AppData" / "LocalLow" / "LuGus Studios" / "Liftoff" / "Recordings"


@dataclass
class Recording:
    path: Path
    name: str
    mode_folder: str
    gamemode: str | None
    environment: str | None
    track_uuid: str | None
    race_uuid: str | None
    created: str | None
    total_time: float
    lap_start_indices: list[int]
    lap_times: list[float]
    drone: str | None
    camera_id: str | None
    uptilt_deg: float | None
    game_version: str | None
    state_layout: str | None
    _states: np.ndarray | None = field(default=None, repr=False)
    _raw: str = field(default="", repr=False)

    @property
    def states(self) -> np.ndarray:
        """``(N, 12)`` float32 records (see the module docstring)."""
        if self._states is None:
            self._states = decode_states(self._raw, self.state_layout)
        return self._states

    @property
    def times(self) -> np.ndarray:
        return self.states[:, 11].astype(np.float64)

    @property
    def positions(self) -> np.ndarray:
        """Unity-world positions ``(N, 3)``."""
        return self.states[:, 0:3].astype(np.float64)

    @property
    def quats_xyzw(self) -> np.ndarray:
        """Unity quaternions ``(N, 4)`` (x, y, z, w)."""
        return self.states[:, 3:7].astype(np.float64)

    @property
    def duration_s(self) -> float:
        t = self.times
        return float(t[-1] - t[0]) if len(t) > 1 else 0.0

    @property
    def rate_hz(self) -> float | None:
        t = self.times
        if len(t) < 3:
            return None
        dt = float(np.median(np.diff(t)))
        return 1.0 / dt if dt > 0 else None

    @property
    def source(self) -> str:
        return "ai-attempt" if AI_ATTEMPT_RE.search(self.name) else "user"

    @property
    def race_time_s(self) -> float | None:
        """The race time in the name (``MM_SS_mmm``); 0 = not finished; None = not in the name."""
        m = RACE_TIME_RE.search(self.name)
        if m is None:
            return None
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000.0

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def decode_states(raw_b64: str, layout: str | None = STATE_LAYOUT) -> np.ndarray:
    if layout != STATE_LAYOUT:
        raise RecordingError(f"unsupported stateLayout {layout!r} (known: {STATE_LAYOUT})")
    try:
        raw = base64.b64decode(raw_b64.strip(), validate=False)
    except (ValueError, TypeError) as exc:
        raise RecordingError(f"statesByte is not base64: {exc}") from exc
    if len(raw) % (4 * STATE_FLOATS):
        raise RecordingError(f"statesByte has {len(raw)} bytes, not a multiple of 48")
    return np.frombuffer(raw, "<f4").reshape(-1, STATE_FLOATS)


def _text(root: ET.Element, path: str) -> str | None:
    v = root.findtext(path)
    return None if v is None else v.strip() or None


def parse_recording(path: str | Path) -> Recording:
    """Parse one recording XML (header eagerly, states lazily)."""
    p = Path(path)
    try:
        root = ET.parse(p).getroot()
    except (ET.ParseError, OSError) as exc:
        raise RecordingError(f"cannot read recording {p}: {exc}") from exc
    if root.tag != "LocalGhostStatesRecording":
        raise RecordingError(f"{p} is not a Liftoff recording (root {root.tag!r})")
    laps = root.find("lapStartIndices")
    times = root.find("lapTimes")
    angle = _text(root, "droneConfiguration/Blueprint/camera/angle")
    try:
        total = float(_text(root, "totalTime") or 0.0)
        lap_idx = [int((e.text or "").strip()) for e in (laps if laps is not None else [])]
        lap_t = [float((e.text or "").strip()) for e in (times if times is not None else [])]
        uptilt = None if angle is None else float(angle)
    except ValueError as exc:
        raise RecordingError(f"{p}: malformed number: {exc}") from exc
    return Recording(
        path=p,
        name=_text(root, "name") or p.stem,
        mode_folder=p.parent.name,
        gamemode=_text(root, "gamemode"),
        environment=_text(root, "environment"),
        track_uuid=_text(root, "trackID/str"),
        race_uuid=_text(root, "raceID/str"),
        created=_text(root, "creationTime"),
        total_time=total,
        lap_start_indices=lap_idx,
        lap_times=lap_t,
        drone=_text(root, "droneConfiguration/name"),
        camera_id=_text(root, "droneConfiguration/Blueprint/camera/id"),
        uptilt_deg=uptilt,
        game_version=_text(root, "gameVersion"),
        state_layout=_text(root, "stateLayout"),
        _raw=_text(root, "statesByte") or "",
    )


def iter_recordings(folder: str | Path | None = None) -> list[Path]:
    base = Path(folder) if folder is not None else default_recordings_dir()
    if not base.is_dir():
        raise HarnessConfigError(f"no Liftoff recordings folder at {base}")
    return sorted(base.rglob("*.xml"))


# ----------------------------------------------------------------------------- catalogue


def display_name(entry: dict[str, Any]) -> str | None:
    return entry.get("name")


def match_entry(rec: Recording, catalogue: Any) -> tuple[dict[str, Any] | None, str]:
    """``(entry or None, how)``; ``how``: ``uuid`` | ``track-uuid`` | ``name`` | ``unmatched``."""
    entries = [e for e in catalogue.entries if e.get("source", "liftoff-xml") == "liftoff-xml"]
    if rec.track_uuid and rec.race_uuid:
        for e in entries:
            u = e.get("uuids") or {}
            if u.get("track") == rec.track_uuid and u.get("race") == rec.race_uuid:
                return e, "uuid"
    if rec.track_uuid and not rec.race_uuid:
        hits = [e for e in entries if (e.get("uuids") or {}).get("track") == rec.track_uuid]
        if len(hits) == 1:
            return hits[0], "track-uuid"
    if rec.environment:
        for e in entries:
            n = display_name(e)
            if e.get("map") == rec.environment and n and rec.name.startswith(n):
                return e, "name"
    return None, "unmatched"


# ----------------------------------------------------------------------------- completeness


def completeness_from_laps(rec: Recording, entry: dict[str, Any] | None) -> dict[str, Any] | None:
    """Source 1: completed laps vs the entry's laps (None when the recording has no race)."""
    if not rec.race_uuid or entry is None or not entry.get("laps"):
        return None
    required = int(entry["laps"])
    done = len(rec.lap_times)
    finished = rec.race_time_s is None or rec.race_time_s > 0
    status = "complete" if done >= required and finished else "incomplete"
    return {"status": status, "source": "laps", "passed": done, "required": required}


def pose_segments(rec: Recording, jump_m: float = RESPAWN_JUMP_M) -> list[tuple[int, int]]:
    """``[start, end)`` ranges split at time gaps (> 1 s, or backwards) and respawn jumps."""
    t, pos = rec.times, rec.positions
    if len(t) < 2:
        return []
    jump = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    cut = np.flatnonzero((np.diff(t) > GAP_S) | (np.diff(t) < 0) | (jump > jump_m)) + 1
    bounds = [0, *cut.tolist(), len(t)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b - a >= 2]


def completeness_from_pose(rec: Recording, course: Any) -> dict[str, Any]:
    """Source 2: the pose replayed through the env pass rule on ``course`` (world scale 1)."""
    from drone_fly.harness.geometry import UNITY_TO_CONTRACT
    from drone_fly.sim.pool import RouteProgress, gate_crossed, total_crossings

    m = np.asarray(UNITY_TO_CONTRACT, np.float64)
    pos = np.array([course.frame.to_course_point(m @ p) for p in rec.positions], np.float64)
    best = 0
    for a, b in pose_segments(rec):
        prog = RouteProgress(course.n_passages, course.laps, course.circuit)
        for i in range(a + 1, b):
            if not prog.finished and gate_crossed(pos[i - 1], pos[i], course.passages[prog.target]):
                prog.advance()
        best = max(best, prog.k)
    required = int(total_crossings(course.n_passages, course.laps, course.circuit))
    return {
        "status": "complete" if best >= required else "incomplete",
        "source": "pose",
        "passed": best,
        "required": required,
    }


def capture_meta_index(captures_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """``{recording sha256: capture meta}`` of captures made from recordings."""
    from drone_fly.harness.offline.capture_index import discover

    out = {}
    for c in discover(captures_root):
        rec = (c.meta or {}).get("recording") or {}
        if rec.get("sha256"):
            out[rec["sha256"]] = c.meta
    return out


# ----------------------------------------------------------------------------- inventory


@dataclass
class InventoryRow:
    name: str
    mode: str
    map: str | None
    track_uuid: str | None
    race_uuid: str | None
    duration_s: float
    states: int
    rate_hz: float | None
    source: str
    catalogue_id: str | None
    match: str
    split: str | None
    completeness: dict[str, Any]
    drone: str | None
    camera_id: str | None
    uptilt_deg: float | None
    camera: str  # preset | different
    capturable: str  # yes | pending-camera | no
    reason: str | None
    sha256: str
    path: str

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def inventory(
    folder: str | Path | None = None,
    *,
    catalogue: Any = None,
    trackdata_root: str | Path | None = None,
    preset_uptilt_deg: float | None = None,
    pose_completeness: bool = True,
    captures_root: str | Path | None = None,
) -> list[InventoryRow]:
    """``captures inventory``: one row per recording (see the module docstring)."""
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.catalogue import TrackDataError

    if catalogue is None:
        catalogue = cat.load_named("liftoff")
    if preset_uptilt_deg is None:
        from drone_fly.harness.config import load_config

        preset_uptilt_deg = float(
            load_config("liftoff", None, preset="liftoff").section("camera")["uptilt_deg"]
        )
    root = None
    try:
        root = cat.resolve_root(trackdata_root, catalogue.header)
    except TrackDataError:
        root = None
    metas = capture_meta_index(captures_root)
    courses: dict[str, Any] = {}
    rows = []
    for p in iter_recordings(folder):
        try:
            rec = parse_recording(p)
            n = len(rec.states)
        except RecordingError as exc:
            rows.append(_bad_row(p, str(exc)))
            continue
        entry, how = match_entry(rec, catalogue)
        comp = completeness_from_laps(rec, entry)
        if comp is None and entry is not None and pose_completeness and rec.race_uuid:
            course = courses.get(entry["id"])
            if course is None and root is not None:
                try:
                    from drone_fly.tracks.loader import load_course

                    course = load_course(entry, catalogue=catalogue, root=root,
                                         scenes_root=None)[0]  # fmt: skip
                    courses[entry["id"]] = course
                except (TrackDataError, HarnessConfigError, OSError):
                    course = None
            if course is not None:
                comp = completeness_from_pose(rec, course)
        sha = rec.sha256()
        if comp is None and sha in metas:
            comp = {**metas[sha]["completeness"], "source": "capture-meta"}
        if comp is None:
            comp = {"status": "unknown", "source": "none", "passed": None, "required": None}
        same_cam = rec.uptilt_deg is not None and abs(rec.uptilt_deg - preset_uptilt_deg) <= (
            UPTILT_TOL_DEG
        )
        if entry is None:
            cap, reason = "no", "no catalogue match"
        elif n < 2:
            cap, reason = "no", "no pose states"
        elif entry.get("status") not in (None, "ok"):
            cap, reason = "no", f"catalogue entry status {entry.get('status')}"
        elif not same_cam and entry.get("split") != "train":
            cap = "no"
            reason = (
                f"different drone/camera ({rec.drone}, uptilt {rec.uptilt_deg}) on a "
                f"{entry.get('split')} map: selection and evaluation stay on the preset camera"
            )
        elif not same_cam:
            cap = "train-only"
            reason = (
                f"different drone/camera ({rec.drone}, {rec.camera_id}, uptilt {rec.uptilt_deg} vs "
                f"preset {preset_uptilt_deg:g}): train maps only, captured with its own uptilt"
            )
        else:
            cap, reason = "yes", None
        rows.append(
            InventoryRow(
                name=rec.name,
                mode=rec.mode_folder,
                map=rec.environment,
                track_uuid=rec.track_uuid,
                race_uuid=rec.race_uuid,
                duration_s=round(rec.duration_s, 2),
                states=n,
                rate_hz=None if rec.rate_hz is None else round(rec.rate_hz, 2),
                source=rec.source,
                catalogue_id=None if entry is None else entry["id"],
                match=how,
                split=None if entry is None else entry.get("split"),
                completeness=comp,
                drone=rec.drone,
                camera_id=rec.camera_id,
                uptilt_deg=rec.uptilt_deg,
                camera="preset" if same_cam else "different",
                capturable=cap,
                reason=reason,
                sha256=sha,
                path=str(p),
            )
        )
    return rows


def _bad_row(p: Path, reason: str) -> InventoryRow:
    return InventoryRow(p.stem, p.parent.name, None, None, None, 0.0, 0, None, "user", None,
                        "unmatched", None, {"status": "unknown", "source": "none", "passed": None,
                                           "required": None}, None, None, None, "different", "no",
                        f"unreadable: {reason}", "", str(p))  # fmt: skip


def inventory_table(rows: list[InventoryRow]) -> str:
    lines = [f"{'recording':58s} {'mode':10s} {'catalogue id':24s} {'split':8s} {'dur s':>6s} "
             f"{'complete':22s} {'camera':9s} capturable"]  # fmt: skip
    for r in rows:
        c = r.completeness
        comp = f"{c['status']}({c['source']} {c.get('passed')}/{c.get('required')})"
        lines.append(
            f"{r.name[:58]:58s} {r.mode[:10]:10s} {(r.catalogue_id or '-')[:24]:24s} "
            f"{(r.split or '-'):8s} {r.duration_s:6.0f} {comp[:22]:22s} {r.camera:9s} "
            f"{r.capturable}{'' if not r.reason else ' - ' + r.reason}"
        )
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.capturable] = counts.get(r.capturable, 0) + 1
    lines.append(f"{len(rows)} recordings: {counts}")
    return "\n".join(lines)
