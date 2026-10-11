"""Screen capture frame source (dxcam on Windows, else mss), latest frame wins.

Both backends are imported lazily inside the capture thread; a missing backend raises
:class:`HarnessDependencyError` from :meth:`ScreenFrameSource.start`. A ``grabber`` callable
(``() -> BGR ndarray | None``) can be injected for tests.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from drone_fly.harness.config import require_choice, require_int, require_number
from drone_fly.harness.errors import HarnessConfigError, HarnessDependencyError, lazy_import
from drone_fly.harness.interfaces import FrameSource
from drone_fly.harness.types import Frame

log = logging.getLogger(__name__)

Grabber = Callable[[], "np.ndarray | None"]


def _region(cfg_region: Any) -> tuple[int, int, int, int] | None:
    if cfg_region is None:
        return None
    if not isinstance(cfg_region, (list, tuple)) or len(cfg_region) != 4:
        raise HarnessConfigError("camera.region must be [left, top, width, height] or null")
    left, top, w, h = (int(v) for v in cfg_region)
    if w <= 0 or h <= 0:
        raise HarnessConfigError("camera.region width/height must be > 0")
    return left, top, w, h


def make_dxcam_grabber(monitor: int, region: tuple[int, int, int, int] | None) -> Grabber:
    dxcam = lazy_import("dxcam", purpose="screen capture (dxcam backend)")
    cam = dxcam.create(output_idx=max(monitor - 1, 0), output_color="BGR")
    box = (
        None
        if region is None
        else (region[0], region[1], region[0] + region[2], region[1] + region[3])
    )

    def grab() -> np.ndarray | None:
        img = cam.grab(region=box) if box is not None else cam.grab()
        return None if img is None else np.ascontiguousarray(img)

    return grab


def make_mss_grabber(monitor: int, region: tuple[int, int, int, int] | None) -> Grabber:
    mss = lazy_import("mss", purpose="screen capture (mss backend)")
    sct = mss.mss()  # must be created on the capturing thread
    if region is not None:
        mon = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
    else:
        mon = sct.monitors[monitor]

    def grab() -> np.ndarray | None:
        shot = np.asarray(sct.grab(mon))
        return np.ascontiguousarray(shot[:, :, :3])  # BGRA -> BGR

    return grab


class ScreenFrameSource(FrameSource):
    """Captures the screen (or a region) at ``fps`` on a background thread."""

    def __init__(
        self,
        backend: str = "auto",
        monitor: int = 1,
        region: tuple[int, int, int, int] | None = None,
        fps: float = 60.0,
        *,
        grabber_factory: Callable[[], Grabber] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.monitor = monitor
        self.region = region
        self.fps = fps
        self.clock = clock
        self._grabber_factory = grabber_factory
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> ScreenFrameSource:
        return cls(
            backend=require_choice(
                cfg.get("backend", "auto"), "camera.backend", ("auto", "dxcam", "mss")
            ),
            monitor=require_int(cfg.get("monitor", 1), "camera.monitor", min_value=0),
            region=_region(cfg.get("region")),
            fps=float(
                require_number(cfg.get("fps", 60), "camera.fps", min_value=1.0, max_value=480.0)
            ),
        )

    def _make_grabber(self) -> Grabber:
        if self._grabber_factory is not None:
            return self._grabber_factory()
        if self.backend in ("auto", "dxcam"):
            try:
                return make_dxcam_grabber(self.monitor, self.region)
            except HarnessDependencyError:
                if self.backend == "dxcam":
                    raise
        return make_mss_grabber(self.monitor, self.region)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._loop, name="screen-capture", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._error is not None:
            err, self._error = self._error, None
            self.stop()
            raise err

    def _loop(self) -> None:
        try:
            grab = self._make_grabber()
        except BaseException as exc:  # noqa: BLE001 - surfaced by start()
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        period = 1.0 / self.fps
        nxt = self.clock()
        while not self._stop.is_set():
            img = grab()
            stamp = self.clock()
            if img is not None:
                with self._lock:
                    self._count += 1
                    self._latest = Frame(image=img, stamp=stamp, index=self._count)
            nxt += period
            delay = nxt - self.clock()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = self.clock()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
