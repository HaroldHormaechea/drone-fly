"""Synthetic Liftoff captures for the UC-64 offline tests (capture loader, labelling, training).

Writes the manual-capture layout the owner's ``capture.py`` produces — ``telemetry_raw.csv``
(``t_wall_ns,len,hex``: raw datagrams packed with the configured :class:`TelemetryFormat`),
``frames.csv`` (``t_wall_ns,file``) and ``frames/`` — with small ``.npy`` frames so nothing needs
OpenCV. Poses are given in Unity (x right, y up, z forward; quaternion ``x, y, z, w``).

UC71: :func:`write_recording` writes a synthetic Liftoff recording XML
(``LocalGhostStatesRecording``, schema in ``docs/harness/liftoff-captures.md`` AC1a), invented
values only.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drone_fly.harness.liftoff.telemetry import TelemetryFormat
from drone_fly.harness.types import TelemetrySample

#: The micro/Arena preset's telemetry layout: 97-byte datagrams (uint8 RPM count prefix).
FMT = TelemetryFormat(rpm_count_prefix=True)
T0_NS = 1_791_717_800_000_000_000

ARCH_TRACK_XML = """<?xml version="1.0" encoding="utf-8"?>
<Track>
  <blueprints>
{items}
  </blueprints>
</Track>
"""


def arch_track(path: Path, arches: Sequence[tuple[str, tuple[float, float, float], float]]) -> Path:
    """Write a track with ``InflatableArchBrandless01`` items ``(instance, position, yaw_deg)``."""
    items = "\n".join(
        f"    <TrackBlueprint><itemID>InflatableArchBrandless01</itemID>"
        f"<instanceID>{iid}</instanceID>"
        f"<position><x>{p[0]}</x><y>{p[1]}</y><z>{p[2]}</z></position>"
        f"<rotation><x>0</x><y>{yaw}</y><z>0</z></rotation></TrackBlueprint>"
        for iid, p, yaw in arches
    )
    path.write_text(ARCH_TRACK_XML.format(items=items), encoding="utf-8")
    return path


def yaw_quat_xyzw(yaw_deg: float) -> np.ndarray:
    """Unity rotation about +y (a heading change), as ``x, y, z, w``."""
    h = math.radians(yaw_deg) / 2.0
    return np.array([0.0, math.sin(h), 0.0, math.cos(h)], np.float32)


@dataclass
class Pose:
    t_sim: float
    position: tuple[float, float, float]
    quat_xyzw: np.ndarray


def straight_flight(
    duration_s: float = 4.0,
    rate_hz: float = 100.0,
    start=(0.0, 1.74, 0.0),
    velocity=(0.0, 0.0, 5.0),
    yaw_deg: float = 0.0,
) -> list[Pose]:
    n = int(round(duration_s * rate_hz)) + 1
    q = yaw_quat_xyzw(yaw_deg)
    return [
        Pose(
            i / rate_hz,
            tuple(float(s + v * i / rate_hz) for s, v in zip(start, velocity, strict=True)),
            q,
        )
        for i in range(n)
    ]


def sample_of(pose: Pose) -> TelemetrySample:
    return TelemetrySample(
        t=pose.t_sim,
        position=np.asarray(pose.position, np.float32),
        quat_xyzw=np.asarray(pose.quat_xyzw, np.float32),
        velocity=np.zeros(3, np.float32),
        gyro=np.zeros(3, np.float32),
        inputs=np.zeros(4, np.float32),
        battery=np.array([1.0, 12.0], np.float32),
        rpm=np.full(4, 20000.0, np.float32),
    )


def write_manual_capture(
    root: Path,
    poses: Sequence[Pose],
    *,
    wall_s: Sequence[float] | None = None,
    frame_times_s: Sequence[float] = (),
    image: Callable[[int], np.ndarray] | None = None,
    size: tuple[int, int] = (64, 27),
    bad_rows: int = 0,
    skip_images: Sequence[int] = (),
) -> Path:
    """Write a manual capture; ``wall_s`` defaults to ``t_sim`` (receive clock = sim clock)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)
    walls = [p.t_sim for p in poses] if wall_s is None else list(wall_s)
    lines = ["t_wall_ns,len,hex"]
    for pose, w in zip(poses, walls, strict=True):
        data = FMT.pack(sample_of(pose))
        lines.append(f"{T0_NS + int(round(w * 1e9))},{len(data)},{data.hex()}")
    for k in range(bad_rows):
        lines.append(f"{T0_NS + k},3,zz00")
    (root / "telemetry_raw.csv").write_text("\n".join(lines) + "\n", encoding="ascii")
    rows = ["t_wall_ns,file"]
    w, h = size
    for k, t in enumerate(frame_times_s):
        t_ns = T0_NS + int(round(t * 1e9))
        name = f"frames/{t_ns}.npy"
        if k not in skip_images:
            img = image(k) if image is not None else np.full((h, w, 3), k % 255, np.uint8)
            np.save(root / name, img)
        rows.append(f"{t_ns},{name}")
    (root / "frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


#: Preset overlay for small synthetic frames (256 x 108 = the Arena 2560 x 1080 / 10, no masks).
SMALL_CAMERA = {"camera": {"resolution": [256, 108], "masks": []}}


def labelled_capture(tmp: Path, duration_s: float = 12.0, fps: float = 10.0) -> tuple[Path, Path]:
    """A straight flight at 1.74 m along +z_u through arches every 15 m, auto-labelled.

    Returns ``(capture_dir, labels_dir)`` (labels made with ``SMALL_CAMERA`` on the micro preset).
    """
    from drone_fly.harness.offline.autolabel import run_autolabel

    poses = straight_flight(duration_s=duration_s)
    frames = [k / fps for k in range(1, int(duration_s * fps))]
    cap = write_manual_capture(
        tmp / "cap",
        poses,
        frame_times_s=frames,
        size=(256, 108),
        image=lambda k: np.random.default_rng(k).integers(0, 255, (108, 256, 3), np.uint8),
    )
    track = arch_track(
        tmp / "track.xml", [(str(i), (0.0, 0.0, 15.0 * i), 0.0) for i in range(1, 6)]
    )
    labels = tmp / "labels"
    run_autolabel(cap, track, preset="liftoff", config=SMALL_CAMERA, out=labels)
    return cap, labels


# ----------------------------------------------------------------------------- UC71 recordings

#: The two drone setups seen in real recordings (``droneConfiguration``): name, camera id, uptilt.
RECORDING_DRONES = {
    "small": ("Test small drone", "CameraFX805MicroCam01", 20.0315),
    "borrum": ("Borrum 180 X (Tutorial)", "CameraVortex01", 30.0),
}
RECORDING_TRACK_UUID = "0f6c1d2e-1111-4a5b-9c0d-a1b2c3d4e5f6"
RECORDING_RACE_UUID = "7a8b9c0d-2222-4e5f-8a9b-f6e5d4c3b2a1"
RECORDING_RATE_HZ = 10.0
RECORDING_STATE_BYTES = 48  # 12 little-endian float32


def recording_states(poses: Sequence[Pose], t0: float = 1234.5) -> np.ndarray:
    """``(N, 12)`` float32 POSITION_STICKS records: position, quat xyzw, sticks, game time."""
    out = np.zeros((len(poses), 12), np.float32)
    for i, p in enumerate(poses):
        out[i, 0:3] = p.position
        out[i, 3:7] = p.quat_xyzw
        # sticks: throttle 0..1, then yaw, pitch, roll in -1..1 (invented, smooth)
        out[i, 7:11] = (0.55, 0.1 * math.sin(p.t_sim), -0.2, 0.05 * math.cos(p.t_sim))
        out[i, 11] = t0 + p.t_sim
    return out


def _race_time(total: float) -> str:
    mm, ss = divmod(total, 60.0)
    return f"{int(mm):02d}_{int(ss):02d}_{int(round((ss % 1) * 1000)):03d}"


def write_recording(
    path: Path,
    poses: Sequence[Pose] | None = None,
    *,
    completed: bool = True,
    drone: str = "small",
    duplicate_stamp: int | None = None,
    laps: int = 2,
    lap_time_s: float = 48.021,
    environment: str = "FakeArena",
    gamemode: str = "Race",
    track_uuid: str = RECORDING_TRACK_UUID,
    race_uuid: str | None = RECORDING_RACE_UUID,
    name: str | None = None,
    t0: float = 1234.5,
    game_version: str = "1.7.6",
) -> Path:
    """Write a synthetic ``LocalGhostStatesRecording`` (see ``docs/harness/liftoff-captures.md``).

    ``poses`` default to a 12 s straight flight at 10 Hz (:data:`RECORDING_RATE_HZ`).
    ``completed=False`` gives an unfinished race: ``totalTime`` 0, ``00_00_000`` in the name and one
    lap fewer in ``lapTimes`` than laps started. ``drone`` is a :data:`RECORDING_DRONES` key
    (``small`` at uptilt 20.0315, ``borrum`` at 30). ``duplicate_stamp=k`` gives state ``k + 1`` the
    game time of state ``k`` (a time stall: step 0, under 0.2 m of motion, as in 5 real files).
    ``race_uuid=None`` omits ``raceID`` (FreeFlight / StuntMode).
    """
    if poses is None:
        poses = straight_flight(duration_s=12.0, rate_hz=RECORDING_RATE_HZ, velocity=(0, 0, 1.0))
    states = recording_states(poses, t0)
    if duplicate_stamp is not None:
        k = int(duplicate_stamp)
        if not 0 <= k < len(states) - 1:
            raise ValueError(f"duplicate_stamp {k} out of range for {len(states)} states")
        states[k + 1, 11] = states[k, 11]
    dname, cam_id, angle = RECORDING_DRONES[drone]
    n = len(states)
    starts = [round(i * n / (laps + 1)) for i in range(laps)]
    done = laps if completed else laps - 1
    lap_times = [round(lap_time_s + 0.137 * i, 3) for i in range(done)]
    total = round(sum(lap_times), 3) if completed else 0.0
    race_time = _race_time(total) if completed else "00_00_000"
    name = name or f"01 - {environment} - {race_time} - 2026-10-13"
    payload = base64.b64encode(states.astype("<f4").tobytes()).decode("ascii")
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<!-- Synthetic UC71 Liftoff recording (liftoff_capture_support.py); invented values. -->",
        '<LocalGhostStatesRecording xmlns:xsd="http://www.w3.org/2001/XMLSchema" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
        f"  <gameVersion>{game_version}</gameVersion>",
        f"  <name>{name}</name>",
        f"  <environment>{environment}</environment>",
        f"  <gamemode>{gamemode}</gamemode>",
        "  <gameModifiers>DisableGodMode EnableCameraNoise EnableBattery</gameModifiers>",
        "  <isCrashed>false</isCrashed>",
        f"  <trackID><str>{track_uuid}</str></trackID>",
        *([f"  <raceID><str>{race_uuid}</str></raceID>"] if race_uuid else []),
        "  <creationTime>20261013.101500</creationTime>",
        f"  <totalTime>{total}</totalTime>",
        "  <lapStartIndices>" + "".join(f"<int>{i}</int>" for i in starts) + "</lapStartIndices>",
        "  <lapTimes>" + "".join(f"<float>{t}</float>" for t in lap_times) + "</lapTimes>",
        "  <droneConfiguration>",
        f"    <name>{dname}</name>",
        "    <FlightController><rates><roll>0.7</roll></rates></FlightController>",
        f"    <Blueprint><camera><id>{cam_id}</id><angle>{angle}</angle></camera></Blueprint>",
        "  </droneConfiguration>",
        "  <stunts />",
        "  <events />",
        "  <stateLayout>POSITION_STICKS</stateLayout>",
        f"  <statesByte>{payload}</statesByte>",
        "</LocalGhostStatesRecording>",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_recording_states(path: Path) -> np.ndarray:
    """Decode a recording's ``statesByte`` back to ``(N, 12)`` float32 (test-side helper)."""
    import xml.etree.ElementTree as ET

    raw = base64.b64decode(ET.parse(path).getroot().findtext("statesByte") or "")
    return np.frombuffer(raw, "<f4").reshape(-1, 12)
