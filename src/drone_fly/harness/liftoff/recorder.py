"""In-repo capture recorder (UC71 AC3/AC4): raw UDP telemetry + wall-stamped JPEG frames.

Writes the **manual capture layout** that :func:`~drone_fly.harness.offline.capture.load_capture`
reads (the owner's external ``capture.py`` wrote the same layout):

* ``telemetry_raw.csv`` - ``t_wall_ns,len,hex``: one raw Liftoff UDP datagram per row;
* ``frames.csv`` - ``t_wall_ns,file``; ``frames/<t_wall_ns>.jpg``;
* ``status.txt`` - the last counters (datagrams, frames), for a human checking a running capture.

Both streams are stamped with the same wall clock (``time.time_ns``) when they arrive. With
``frames=False`` it records telemetry only (the AC1b replay datagram count); with
``telemetry=False`` frames only (strategy B: the pose then comes from the recording XML, see
:func:`write_pose_capture`).

The UDP socket and the frame grabber are injectable, so the recorder is testable without the game.
"""

from __future__ import annotations

import csv
import logging
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError, lazy_import

log = logging.getLogger(__name__)

DEFAULT_PORT = 9001
DEFAULT_FPS = 10.0
JPEG_QUALITY = 90


@dataclass
class RecorderStats:
    datagrams: int = 0
    frames: int = 0
    started_ns: int = 0
    stopped_ns: int = 0

    @property
    def duration_s(self) -> float:
        return max(0, self.stopped_ns - self.started_ns) / 1e9

    def as_dict(self) -> dict[str, Any]:
        d = self.duration_s
        return {
            "datagrams": self.datagrams,
            "frames": self.frames,
            "duration_s": round(d, 3),
            "datagram_rate_hz": round(self.datagrams / d, 2) if d > 0 else None,
            "frame_rate_hz": round(self.frames / d, 2) if d > 0 else None,
        }


class CaptureRecorder:
    """Record UDP telemetry and/or screen frames into ``out_dir`` (see the module docstring).

    ``grab`` returns a BGR ``uint8`` image or ``None``; default: an mss/dxcam screen grabber of
    ``monitor``. ``sock`` is a bound UDP socket (default: ``127.0.0.1:port``, as the harness).
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        port: int = DEFAULT_PORT,
        fps: float = DEFAULT_FPS,
        telemetry: bool = True,
        frames: bool = True,
        monitor: int = 1,
        grab: Callable[[], np.ndarray | None] | None = None,
        sock: Any = None,
        clock_ns: Callable[[], int] = time.time_ns,
        jpeg_quality: int = JPEG_QUALITY,
    ) -> None:
        if not (telemetry or frames):
            raise HarnessConfigError("the recorder needs telemetry and/or frames")
        if fps <= 0:
            raise HarnessConfigError("--fps must be > 0")
        self.out = Path(out_dir)
        self.port, self.fps, self.monitor = int(port), float(fps), int(monitor)
        self.telemetry, self.frames = telemetry, frames
        self._grab, self._sock = grab, sock
        self.clock_ns = clock_ns
        self.jpeg_quality = int(jpeg_quality)
        self.stats = RecorderStats()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: list[BaseException] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if (self.out / "frames.csv").exists() or (self.out / "telemetry_raw.csv").exists():
            raise HarnessConfigError(f"{self.out} already holds a capture; pick a new folder")
        self.out.mkdir(parents=True, exist_ok=True)
        self.stats = RecorderStats(started_ns=self.clock_ns())
        self._stop.clear()
        if self.telemetry:
            if self._sock is None:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.bind(("127.0.0.1", self.port))  # as the harness; 0.0.0.0 trips the firewall
                self._sock = s
            self._sock.settimeout(0.2)
            self._spawn(self._telemetry_loop, "capture-udp")
        if self.frames:
            (self.out / "frames").mkdir(exist_ok=True)
            if self._grab is None:
                self._grab = _screen_grabber(self.monitor)
            self._spawn(self._frame_loop, "capture-frames")

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        t = threading.Thread(target=self._guard(target), name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _guard(self, fn: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced by stop()
                self._errors.append(exc)
                self._stop.set()

        return run

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self.stats.stopped_ns = self.clock_ns()
        self._write_status()
        if self._errors:
            raise HarnessConfigError(f"the recorder failed: {self._errors[0]}") from self._errors[0]
        return self.stats.as_dict()

    def run(
        self, max_s: float | None = None, should_stop: Callable[[], bool] | None = None
    ) -> dict:
        """Start, record until ``max_s`` elapses, ``should_stop()`` or Ctrl+C, then stop."""
        self.start()
        t0 = time.monotonic()
        try:
            while not self._stop.is_set():
                if max_s is not None and time.monotonic() - t0 >= max_s:
                    break
                if should_stop is not None and should_stop():
                    break
                time.sleep(0.2)
                self._write_status()
        except KeyboardInterrupt:
            log.info("recorder: stopped by the user")
        return self.stop()

    def _write_status(self) -> None:
        with self._lock:
            s = self.stats
            (self.out / "status.txt").write_text(
                f"datagrams={s.datagrams} frames={s.frames}\n", encoding="utf-8"
            )

    # ------------------------------------------------------------------ loops

    def _telemetry_loop(self) -> None:
        with (self.out / "telemetry_raw.csv").open("w", newline="", encoding="ascii") as fh:
            fh.write("t_wall_ns,len,hex\n")
            while not self._stop.is_set():
                try:
                    data, _ = self._sock.recvfrom(65535)
                except TimeoutError:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                fh.write(f"{self.clock_ns()},{len(data)},{data.hex()}\n")
                with self._lock:
                    self.stats.datagrams += 1

    def _frame_loop(self) -> None:
        cv2 = lazy_import("cv2", package="opencv-python", purpose="the capture recorder")
        period_ns = int(1e9 / self.fps)
        with (self.out / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t_wall_ns", "file"])
            nxt = self.clock_ns()
            while not self._stop.is_set():
                img = self._grab()
                t = self.clock_ns()
                if img is not None:
                    name = f"frames/{t}.jpg"
                    ok = cv2.imwrite(str(self.out / name), img,
                                     [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])  # fmt: skip
                    if not ok:
                        raise HarnessConfigError(f"could not write {self.out / name}")
                    w.writerow([t, name])
                    fh.flush()
                    with self._lock:
                        self.stats.frames += 1
                nxt += period_ns
                delay = (nxt - self.clock_ns()) / 1e9
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    nxt = self.clock_ns()


def _screen_grabber(monitor: int) -> Callable[[], np.ndarray | None]:
    from drone_fly.harness.errors import HarnessDependencyError
    from drone_fly.harness.liftoff.frames import make_dxcam_grabber, make_mss_grabber

    try:
        return make_dxcam_grabber(monitor, None)
    except HarnessDependencyError:
        return make_mss_grabber(monitor, None)


# ----------------------------------------------------------------------------- strategy B


POSE_RESAMPLE_HZ = 100.0


def _nlerp(q: np.ndarray, idx: np.ndarray, u: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):  # hemisphere-continuous
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out = q[idx] * (1 - u)[:, None] + q[np.minimum(idx + 1, len(q) - 1)] * u[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def pose_rows(
    rec_times: np.ndarray,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    sticks: np.ndarray | None,
    *,
    t0_frame_s: float,
    offset_s: float = 0.0,
    rate: float = 1.0,
    hz: float = POSE_RESAMPLE_HZ,
) -> list[dict[str, float]]:
    """A recording's pose on the frame clock, resampled to ``hz`` (``telemetry.csv`` rows).

    Recording time ``t`` maps to ``t0_frame_s + (t - t[0]) / rate + offset_s``. Positions: linear,
    attitudes nlerp; velocity is the finite difference of the recording's positions, gyro 0
    (the recording has neither). ``hz`` (100) keeps every frame within the pairing tolerance
    after any lag ``captures verify`` finds.
    """
    t = np.asarray(rec_times, np.float64)
    keep = np.concatenate([[True], np.diff(t) > 0])  # drop repeated stamps
    t, p, q = t[keep], np.asarray(positions, np.float64)[keep], np.asarray(quats_xyzw)[keep]
    s = None if sticks is None else np.asarray(sticks, np.float64)[keep]
    if len(t) < 2:
        raise HarnessConfigError("the recording has fewer than 2 pose states")
    vel = np.gradient(p, t, axis=0)
    grid = np.arange(t[0], t[-1], rate / hz)  # hz on the frame clock (frame dt = rec dt / rate)
    idx = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    u = (grid - t[idx]) / (t[idx + 1] - t[idx])
    pos = p[idx] + u[:, None] * (p[idx + 1] - p[idx])
    vel_i = vel[idx] + u[:, None] * (vel[idx + 1] - vel[idx])
    qi = _nlerp(q, idx, u)
    frame_t = t0_frame_s + (grid - t[0]) / rate + offset_s
    rows = []
    for k in range(len(grid)):
        r = {
            "t": float(grid[k]),
            "pos_x": pos[k, 0], "pos_y": pos[k, 1], "pos_z": pos[k, 2],
            "quat_x": qi[k, 0], "quat_y": qi[k, 1], "quat_z": qi[k, 2], "quat_w": qi[k, 3],
            "vel_x": vel_i[k, 0], "vel_y": vel_i[k, 1], "vel_z": vel_i[k, 2],
            "gyro_pitch": 0.0, "gyro_roll": 0.0, "gyro_yaw": 0.0,
            "received_at": float(frame_t[k]),
        }  # fmt: skip
        if s is not None:
            st = s[idx[k]]
            r.update(in_throttle=st[0], in_yaw=st[1], in_pitch=st[2], in_roll=st[3])
        rows.append(r)
    return rows


def write_pose_capture(
    out_dir: str | Path,
    recording: Any,
    *,
    t0_frame_s: float,
    offset_s: float = 0.0,
    rate: float = 1.0,
) -> Path:
    """Strategy B: turn a frames-only capture + its recording XML into the **run layout**.

    ``out_dir`` holds ``frames.csv`` (``t_wall_ns,file``, from :class:`CaptureRecorder` with
    ``telemetry=False``) and ``frames/``. Writes ``telemetry.csv`` (:data:`CSV_COLUMNS` +
    ``received_at`` on the frame clock, seconds) from the recording's pose, and rewrites
    ``frames.csv`` as :data:`~drone_fly.harness.recording.FRAME_COLUMNS` (``stamp`` in seconds), so
    ``load_capture`` reads it unchanged. The original is kept as ``frames_wall.csv``.
    """
    from drone_fly.harness.liftoff.telemetry import CSV_COLUMNS
    from drone_fly.harness.recording import FRAME_COLUMNS

    d = Path(out_dir)
    src = d / "frames.csv"
    if not src.is_file():
        raise HarnessConfigError(f"{d} has no frames.csv")
    with src.open(newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    if not frames or "t_wall_ns" not in frames[0]:
        raise HarnessConfigError(f"{src} is not a recorder frames.csv (t_wall_ns,file)")
    st = recording.states
    rows = pose_rows(recording.times, st[:, 0:3], st[:, 3:7], st[:, 7:11],
                     t0_frame_s=t0_frame_s, offset_s=offset_s, rate=rate)  # fmt: skip
    cols = [*CSV_COLUMNS[:18], "received_at"]
    with (d / "telemetry.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    src.replace(d / "frames_wall.csv")
    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FRAME_COLUMNS)
        for i, f in enumerate(frames):
            w.writerow([i, i, int(f["t_wall_ns"]) / 1e9, "", "", f["file"]])
    return d / "telemetry.csv"


# ----------------------------------------------------------------------------- capture commands

#: The AC3 capture strategy: "A" (UDP telemetry streams during replay playback), "B" (frames +
#: the recording XML pose, aligned by `captures verify`) or "C" (neither: manual fallback only).
#: P0a found a 10 Hz pose in every recording; P1 (AC1b) measured no UDP during replay: B.
REPLAY_STRATEGY = "B"
#: Extra masks for the replay HUD on top of the preset's (MEASURED 2026-10-13: none needed). The
#: replay shows the live OSD (ALT/SPD, stick widgets) plus a "<pilot> - <time>" label at
#: x 2280-2540, y 524-554, inside the preset's ghost_list mask; the playback bar hides in play.
REPLAY_HUD_MASKS: list = []

MANUAL_FALLBACK = """\
Replay capture is not possible ({why}). Capture the map by hand instead
(docs/harness/liftoff-captures.md, "AC4. Manual fallback"):
  1. load the catalogue track {track} in Liftoff with the preset drone;
  2. drone-fly harness captures record --track {track} --max-s 600;
  3. fly 3-4 laps, varying height, speed and line;
  4. stop the recorder (Ctrl+C);
  5. drone-fly harness captures verify <the new session folder> --search-lag"""


def record_manual(
    catalogue_id: str,
    *,
    max_s: float | None = None,
    fps: float = DEFAULT_FPS,
    port: int = DEFAULT_PORT,
    captures_root: str | Path | None = None,
    trackdata_root: str | Path | None = None,
    preset: str | None = "liftoff",
) -> tuple[Path, dict[str, Any]]:
    """``captures record`` (AC4): record a hand-flown capture of a catalogue track + its meta."""
    from datetime import date

    from drone_fly.harness.offline import capture_index as ci
    from drone_fly.tracks import catalogue as cat

    entry = cat.load_named("liftoff").by_id(catalogue_id)
    root = Path(captures_root) if captures_root else ci.captures_root()
    out = ci.next_session_dir(root, entry["map"], date.today().isoformat(), "manual")
    log.info("recording %s -> %s (Ctrl+C to stop)", catalogue_id, out)
    stats = CaptureRecorder(out, port=port, fps=fps).run(max_s)
    if not stats["datagrams"]:
        raise HarnessConfigError(f"no telemetry arrived on UDP {port} (is Liftoff flying?)")
    meta = ci.register(out, catalogue_id, preset=preset, trackdata_root=trackdata_root)
    return out, {"stats": stats, "meta": meta}


def find_recording(name: str, folder: str | Path | None = None) -> Path:
    """A recording by file stem (or a path to its XML)."""
    from drone_fly.harness.offline.recordings import iter_recordings

    p = Path(name)
    if p.suffix.lower() == ".xml" and p.is_file():
        return p
    hits = [q for q in iter_recordings(folder) if q.stem == name]
    if len(hits) != 1:
        raise HarnessConfigError(
            f"recording {name!r}: {len(hits)} matches in the Recordings folder (need exactly 1)"
        )
    return hits[0]


def record_replay(
    recording: str,
    *,
    max_s: float | None = None,
    fps: float = DEFAULT_FPS,
    port: int = DEFAULT_PORT,
    strategy: str = "auto",
    recordings_dir: str | Path | None = None,
    captures_root: str | Path | None = None,
    trackdata_root: str | Path | None = None,
    preset: str | None = "liftoff",
    allow_pending_camera: bool = False,
    browser_pos: tuple[int | None, int | None, int] | None = None,
    dropdown_counts: tuple[int | None, int | None] = (None, None),
) -> tuple[Path, dict[str, Any]]:
    """``captures replay`` (AC3): play a stored recording in Liftoff and capture it.

    ``browser_pos`` = ``(environment index, race index, list row)`` in the replay browser
    (:func:`~drone_fly.harness.liftoff.ui_replay.select_recording`; None keeps a filter). A wrong
    pick fails ``captures verify`` (the replayed pose does not match the frames).

    Strategy A records frames + UDP; B records frames and rebuilds the pose from the recording XML
    (:func:`write_pose_capture`); C refuses with the manual fallback. The capture's ``meta.json``
    links the catalogue entry and the recording (name + sha256); ``captures verify --search-lag``
    then sets ``verified``.
    """
    from datetime import date

    from drone_fly.harness.config import load_config
    from drone_fly.harness.liftoff import ui_replay
    from drone_fly.harness.offline import align
    from drone_fly.harness.offline import capture_index as ci
    from drone_fly.harness.offline import recordings as rc
    from drone_fly.tracks import catalogue as cat

    strat = REPLAY_STRATEGY if strategy == "auto" else strategy
    if strat not in ("A", "B", "C"):
        raise HarnessConfigError(f"--strategy must be auto, A, B or C, got {strategy!r}")
    path = find_recording(recording, recordings_dir)
    rec = rc.parse_recording(path)
    catalogue = cat.load_named("liftoff")
    entry, how = rc.match_entry(rec, catalogue)
    track = entry["id"] if entry else "<catalogue-id>"
    if strat == "C":
        raise HarnessConfigError(MANUAL_FALLBACK.format(why="strategy C", track=track))
    if entry is None:
        raise HarnessConfigError(f"recording {rec.name!r} matches no catalogue entry")
    cfg = load_config("liftoff", None, preset=preset)
    cam = cfg.section("camera")
    if rec.uptilt_deg is None or abs(rec.uptilt_deg - float(cam["uptilt_deg"])) > rc.UPTILT_TOL_DEG:
        if entry.get("split") != "train":
            raise HarnessConfigError(
                f"recording {rec.name!r} uses another drone/camera ({rec.drone}, uptilt "
                f"{rec.uptilt_deg}) on a {entry.get('split')} map: val / held-out captures stay "
                "on the preset camera (ADR 0010)"
            )
        if not allow_pending_camera:
            raise HarnessConfigError(
                f"recording {rec.name!r} uses another drone/camera ({rec.drone}, uptilt "
                f"{rec.uptilt_deg}): train maps only; pass --allow-pending-camera to capture it"
            )
    if browser_pos is None:
        raise HarnessConfigError(
            "pass --browser-pos ENV,RACE,ROW: the replay browser's dropdown / list indices of the "
            "recording (the driver cannot read the entries' text)"
        )
    root = Path(captures_root) if captures_root else ci.captures_root()
    out = ci.next_session_dir(root, entry["map"], date.today().isoformat(), "replay")
    duration = rec.duration_s + 2.0
    limit = duration if max_s is None else min(max_s, duration)
    out.mkdir(parents=True, exist_ok=True)
    env_i, race_i, row = browser_pos
    ui_replay.ensure_running()
    ui_replay.open_replays()
    ui_replay.select_recording(
        env_i, race_i, row, str(out / "browser.png"),
        env_count=dropdown_counts[0], race_count=dropdown_counts[1],
    )  # fmt: skip
    recorder = CaptureRecorder(out, port=port, fps=fps, telemetry=strat == "A")
    recorder.start()  # frames before Play have no pose and are dropped when loading
    try:
        t_play = ui_replay.play()
        lost = ui_replay.watch_focus(t_play + limit)
    finally:
        stats = recorder.stop()
        ui_replay.stop_to_menu()
    if lost:
        raise HarnessConfigError(
            f"another window took the focus during the replay at "
            f"{[round(t - t_play, 1) for t in lost]} s: the frames in {out} are not all "
            "Liftoff's; discard the capture and replay it"
        )
    if strat == "B":
        write_pose_capture(out, rec, t0_frame_s=t_play)
    comp = rc.completeness_from_laps(rec, entry) or {
        "status": "unknown", "source": "none", "passed": None, "required": None}  # fmt: skip
    meta = ci.build_meta(
        out,
        entry,
        camera={**cam, "uptilt_deg": rec.uptilt_deg or cam["uptilt_deg"]},
        drone=str(cfg.data.get("drone") or "unknown"),
        source="replay",
        recording={"name": rec.name, "sha256": rec.sha256(), "origin": rec.source},
        completeness=comp,
        pose_source="udp" if strat == "A" else "recording-xml",
        time={"offset_s": 0.0, "rate": 1.0},
        masks=list(REPLAY_HUD_MASKS),
        camera_source="recording" if rec.uptilt_deg is not None else "preset",
    )
    ci.write_meta(out, meta)
    res = align.verify_capture(out, search_lag=True, trackdata_root=trackdata_root, preset=preset)
    return out, {"stats": stats, "match": how, "alignment": res.alignment_block()}
