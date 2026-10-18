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
The ``captures record`` / ``captures replay`` commands built on it live in
:mod:`drone_fly.harness.offline.capture_session` (they read recording XMLs and the catalogue).
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
