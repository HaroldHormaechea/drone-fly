"""Vision eyes as a live free-space estimator (``parts.free_space_estimator: eyes``).

Wraps :class:`drone_fly.eyes.infer.EyesEstimator` (the UC-84/85 vision-only ray net) as the
harness's ``raycast`` / ``obstacle_vision`` source for the 70-dim obstacle models.

Provider contract (what any pluggable free-space perception must honour; this part is the
reference implementation):

* **Inputs** -- the screen :class:`~drone_fly.harness.types.Frame`, the telemetry
  :class:`~drone_fly.harness.types.FlightState` (training scale) and the gate target produced by an
  ON-SCREEN gate detector. Never track geometry, labels or oracle data: ``from_config`` refuses
  ``parts.gate_estimator`` ``course`` / ``oracle``.
* **Outputs** -- ``raycast`` float32 ``(40,)`` in ``[0, 1]`` (1 = clear to ``RAY_RANGE``) in the
  contract ``CONE`` order; ``in_fov_mask`` bool ``(40,)`` (rays the camera sees);
  ``obstacle_vision`` float32 ``(8,)`` (``zeros``, or ``ray_clusters`` derived from the rays only).
* **stamp** -- capture time (``perf_counter``) of the newest frame that produced a VALID estimate
  (history complete); ``-inf`` when there is none, so the runner's ``max_estimate_age_s`` check
  fails safe (``perception_stale``). An invalid output (startup, a history gap) never replaces the
  held estimate.
* **Latency rule** -- the worker's GPU duty (forward time x rate) stays <= :data:`MAX_GPU_DUTY`
  (a warning otherwise) and the estimate age <= ``runner.max_estimate_age_s``.
* **info** -- ``frame_index`` and ``inference_s`` (read by the recorder), plus eyes diagnostics.

Threading: with a frame source (live) the net runs on a daemon worker thread ("eyes-worker") at
``max_rate_hz`` (``auto``: 30 Hz for EyesNet v1/v2, 10 Hz for the EyesDepthNet v3), latest frame
wins, and ``update()`` only reads the published result. On CUDA the worker (and the start-up
warm-up) run on a dedicated CUDA stream, so the eyes never serialise behind -- or stall -- the
policy and gate detector on the default stream; only host NumPy results cross threads. One thread
owns the estimator: a ``reset()`` while the worker runs is applied on the worker thread. Without a
frame source (offline eval, tests) ``update()`` processes the frame it is given synchronously on
the caller's current stream.

torch is imported lazily (a missing torch raises :class:`HarnessDependencyError`); a v3 weights
file additionally needs ``transformers`` / ``pillow`` for its frozen depth backbone.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import find_repo_root, require_choice, require_number
from drone_fly.harness.course_senses import obstacle_vision_block
from drone_fly.harness.errors import (
    HarnessConfigError,
    HarnessDependencyError,
    HarnessError,
    lazy_import,
)
from drone_fly.harness.interfaces import FreeSpaceEstimator
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate

log = logging.getLogger(__name__)

OBSTACLE_MODES = ("zeros", "ray_clusters")
DEVICES = ("auto", "cpu", "cuda")
#: ``max_rate_hz: auto`` per net class (from the weights meta).
AUTO_RATE_HZ = {"EyesNet": 30.0, "EyesDepthNet": 10.0}
#: Above this fraction of wall time spent in the eyes forward pass, start() logs a warning.
MAX_GPU_DUTY = 0.5
#: Warm-up forward passes in start() (the median forward time is recorded).
WARMUP_PASSES = 3
#: Gate estimators that read track geometry: the eyes must stay screen-only.
NON_SCREEN_GATE_ESTIMATORS = ("course", "oracle")
#: A ray reads a hit below this normalised free space.
HIT_EPS = 1e-3
_POLL_S = 0.002


# ----------------------------------------------------------------------------- pure helpers


def net_class(meta: dict[str, Any]) -> str:
    """``EyesDepthNet`` when the meta has a depth backbone (v3), else ``EyesNet`` (v1/v2)."""
    return "EyesDepthNet" if (meta.get("arch") or {}).get("depth_backbone") else "EyesNet"


def min_rate_hz(meta: dict[str, Any]) -> float:
    """Slowest rate whose frames still fill the history: ``1 / (2 * hist_tol_s)``."""
    inp = meta.get("input") or {}
    dt_s = float(inp.get("dt_s", 0.2))
    tol_s = float(inp.get("hist_tol_s", 0.5 * dt_s))
    return 1.0 / (2.0 * tol_s)


def resolve_rate(value: Any, meta: dict[str, Any]) -> float:
    """``eyes.max_rate_hz`` (``auto`` or a number > 0) -> Hz, refusing rates below the floor."""
    if value in (None, "auto"):
        return AUTO_RATE_HZ[net_class(meta)]
    rate = float(require_number(value, "eyes.max_rate_hz", min_value=0.0, min_exclusive=True))
    floor = min_rate_hz(meta)
    if rate < floor:
        raise HarnessConfigError(
            f"config 'eyes.max_rate_hz' ({rate:g} Hz) is below {floor:g} Hz = 1 / (2 * "
            "hist_tol_s) of the weights: frames would be too sparse to fill the history"
        )
    return rate


def gpu_duty(forward_s: float | None, rate_hz: float) -> float | None:
    """Fraction of wall time the forward pass occupies at ``rate_hz``."""
    return None if forward_s is None else float(forward_s) * float(rate_hz)


def check_duty(duty: float | None, limit: float = MAX_GPU_DUTY) -> bool:
    """True when ``duty`` is within ``limit`` (unknown counts as ok); logs a warning otherwise."""
    if duty is None or duty <= limit:
        return True
    log.warning(
        "eyes GPU duty %.0f%% exceeds %.0f%%: lower eyes.max_rate_hz or use lighter weights",
        100.0 * duty,
        100.0 * limit,
    )
    return False


def obstacle_vision_from_rays(
    pos: np.ndarray,
    rays_world: np.ndarray,
    free: np.ndarray,
    ray_range: float,
    link_m: float = 0.5,
    min_size: float = 0.15,
) -> np.ndarray:
    """``obstacle_vision(8)`` from the eyes rays alone (training metres).

    Hit rays (``free < 1 - 1e-3``) give surface points ``pos + dir * free * ray_range``. Twice
    (``NVIS`` = 2): the nearest unassigned point seeds a cluster that grows by single linkage
    (``link_m``); size = ``max(half-extent, min_size)``, centre = surface mean pushed ``size``
    along the view direction (visible surface -> body centre, as the depth estimator does).
    """
    pos = np.asarray(pos, np.float64).reshape(3)
    dirs = np.asarray(rays_world, np.float64).reshape(-1, 3)
    f = np.asarray(free, np.float64).reshape(-1)
    hit = f < 1.0 - HIT_EPS
    pts = pos + dirs[hit] * (f[hit] * float(ray_range))[:, None]
    remaining = np.ones(len(pts), bool)
    centers, sizes = [], []
    for _ in range(2):
        idx = np.flatnonzero(remaining)
        if not len(idx):
            break
        seed = idx[np.argmin(np.linalg.norm(pts[idx] - pos, axis=1))]
        member = np.zeros(len(pts), bool)
        member[seed] = True
        frontier = [seed]
        while frontier:
            p = pts[frontier.pop()]
            near = remaining & ~member & (np.linalg.norm(pts - p, axis=1) <= link_m)
            member |= near
            frontier.extend(np.flatnonzero(near).tolist())
        remaining &= ~member
        c = pts[member]
        size = float(max(0.5 * float((c.max(axis=0) - c.min(axis=0)).max()), min_size))
        surface = c.mean(axis=0)
        view = surface - pos
        view /= np.linalg.norm(view) + 1e-9
        centers.append(surface + view * size)
        sizes.append(size)
    return obstacle_vision_block(
        pos, centers=np.array(centers).reshape(-1, 3), sizes=np.array(sizes)
    )


def resolve_weights_path(value: Any) -> Path:
    """``eyes.weights``: absolute, or relative to the repo root (else the cwd); must exist."""
    if not value:
        raise HarnessConfigError(
            "config 'eyes.weights' is required for the eyes estimator (an eyes.pt file)"
        )
    path = Path(str(value))
    if not path.is_absolute():
        root = find_repo_root()
        path = (root / path) if root is not None else (Path.cwd() / path)
    if not path.is_file():
        raise HarnessConfigError(f"config 'eyes.weights': file not found: {path}")
    return path


def check_meta(meta: dict[str, Any], contract: Any, world_scale: float, cam: dict) -> None:
    """The weights' geometry must match the contract and the config (HarnessConfigError)."""

    def mismatch(what: str, weights: Any, config: Any) -> HarnessConfigError:
        return HarnessConfigError(
            f"eyes weights {what} ({weights}) does not match the {what} of this run ({config})"
        )

    n = int(meta.get("n_rays", -1))
    if contract is not None:
        if n != int(contract.N_RAYS):
            raise mismatch("n_rays", n, int(contract.N_RAYS))
        cone_w = np.asarray(meta.get("cone"), np.float64).reshape(-1, 3)
        cone_c = np.asarray(contract.CONE, np.float64).reshape(-1, 3)
        if cone_w.shape != cone_c.shape or not np.allclose(cone_w, cone_c, atol=1e-5):
            raise mismatch("ray cone", f"shape {cone_w.shape}", f"contract CONE {cone_c.shape}")
        rr = float(meta.get("ray_range", math.nan))
        if not math.isclose(rr, float(contract.RAY_RANGE), rel_tol=1e-6):
            raise mismatch("ray_range", rr, float(contract.RAY_RANGE))
    rrs = float(meta.get("ray_range_scale", 1.0))
    if not math.isclose(rrs, 1.0, rel_tol=1e-6):
        raise mismatch("ray_range_scale", rrs, 1.0)
    ws = float(meta.get("world_scale", math.nan))
    if not math.isclose(ws, float(world_scale), rel_tol=1e-6):
        raise mismatch("world_scale", ws, float(world_scale))
    mcam = meta.get("camera") or {}
    for key, default in (("hfov_deg", None), ("uptilt_deg", 0.0)):
        want = cam.get(key, default)
        got = mcam.get(key, default)
        if want is None or got is None or not math.isclose(float(got), float(want), abs_tol=1e-3):
            raise mismatch(f"camera.{key}", got, want)


def cuda_stream_for(device: str) -> Any:
    """A dedicated CUDA stream on ``device`` ordered after the weight upload, else ``None``."""
    if not str(device).startswith("cuda"):
        return None
    torch = lazy_import("torch", purpose="the eyes free-space estimator")
    stream = torch.cuda.Stream(device)
    # The weights were copied on the default stream: the worker stream must not run ahead.
    stream.wait_stream(torch.cuda.current_stream(device))
    return stream


# ----------------------------------------------------------------------------- estimator


class EyesFreeSpaceEstimator(FreeSpaceEstimator):
    provides = frozenset({"obstacle_vision", "raycast"})

    def __init__(
        self,
        estimator: Any,
        *,
        frame_source: Any = None,
        state_source: Any = None,
        max_rate_hz: float | str = "auto",
        warmup_timeout_s: float = 30.0,
        obstacle_mode: str = "zeros",
        cluster_link_m: float = 0.5,
        min_obstacle_size_m: float = 0.15,
        weights: str | None = None,
        stream: Any = "auto",
        clock: Any = time.perf_counter,
    ) -> None:
        require_choice(obstacle_mode, "eyes.obstacle_vision", OBSTACLE_MODES)
        self.estimator = estimator
        self.meta = getattr(estimator, "meta", {}) or {}
        self.n_rays = int(getattr(estimator, "n_rays", 40))
        self.ray_range = float(self.meta.get("ray_range", 4.0))
        self.frame_source = frame_source
        self.state_source = state_source
        self.rate_hz = resolve_rate(max_rate_hz, self.meta)
        self.warmup_timeout_s = float(warmup_timeout_s)
        self.obstacle_mode = obstacle_mode
        self.cluster_link_m = float(cluster_link_m)
        self.min_obstacle_size_m = float(min_obstacle_size_m)
        self.weights = weights
        device = str(getattr(estimator, "device", "cpu"))
        self._stream = cuda_stream_for(device) if stream == "auto" else stream
        self.clock = clock
        self.forward_s: float | None = None  # median warm-up forward time
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._target: np.ndarray | None = None
        self._published: tuple[Any, np.ndarray, int, float] | None = None
        self._generation = 0
        self._reset_pending = False
        self._last_stamp: float | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> EyesFreeSpaceEstimator:
        conf = ctx.config
        gate = conf.parts.gate_estimator
        if gate in NON_SCREEN_GATE_ESTIMATORS:
            raise HarnessConfigError(
                f"the eyes estimator is screen-only: parts.gate_estimator {gate!r} reads track "
                "geometry; use an on-screen detector (learned | opencv)"
            )
        ecfg = conf.section("eyes")
        cam = conf.section("camera")
        device = require_choice(ecfg.get("device", "auto"), "eyes.device", DEVICES)
        rate = ecfg.get("max_rate_hz", "auto")
        if rate != "auto":
            require_number(rate, "eyes.max_rate_hz", min_value=0.0, min_exclusive=True)

        def num(key: str, default: float, **kw: Any) -> float:
            return float(require_number(ecfg.get(key, default), f"eyes.{key}", **kw))

        warmup = num("warmup_timeout_s", 30.0, min_value=0.0)
        mode = require_choice(
            ecfg.get("obstacle_vision", "zeros"), "eyes.obstacle_vision", OBSTACLE_MODES
        )
        link = num("cluster_link_m", 0.5, min_value=0.0, min_exclusive=True)
        min_size = num("min_obstacle_size_m", 0.15, min_value=0.0, min_exclusive=True)
        path = resolve_weights_path(ecfg.get("weights"))
        try:
            lazy_import("torch", purpose="the eyes free-space estimator")
            from drone_fly.eyes.infer import EyesEstimator, load_eyes_weights
            from drone_fly.eyes.model import build_from_meta

            meta, state = load_eyes_weights(path)
            check_meta(meta, ctx.contract, conf.world_scale, cam)
            net = build_from_meta(meta)
            net.load_state_dict(state)
            estimator = EyesEstimator(net, meta, device)
        except HarnessError:
            raise
        except ModuleNotFoundError as exc:
            raise HarnessDependencyError(
                (exc.name or "unknown").split(".")[0], "the eyes free-space estimator"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - any load failure is a config problem
            raise HarnessConfigError(f"config 'eyes.weights': cannot load {path}: {exc}") from exc
        return cls(
            estimator,
            frame_source=ctx.resources.get("frame_source"),
            state_source=ctx.resources.get("state_source"),
            max_rate_hz=rate,
            warmup_timeout_s=warmup,
            obstacle_mode=mode,
            cluster_link_m=link,
            min_obstacle_size_m=min_size,
            weights=str(path),
        )

    # ------------------------------------------------------------------ lifecycle

    def _stream_ctx(self) -> Any:
        if self._stream is None:
            return contextlib.nullcontext()
        torch = lazy_import("torch", purpose="the eyes free-space estimator")
        return torch.cuda.stream(self._stream)

    def _sync_stream(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()

    def warm_up(self) -> float | None:
        """Forward passes on blank frames (history filled); returns the median forward time."""
        est = self.estimator
        meta_cam = self.meta.get("camera") or {}
        w, h = (int(v) for v in meta_cam.get("resolution", (est.width, est.height)))
        image = np.zeros((h, w, 3), np.uint8)
        state = FlightState(
            pos=np.zeros(3, np.float32),
            vel=np.zeros(3, np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
            omega=np.zeros(3, np.float32),
        )
        dt = float(getattr(est, "dt_s", 0.2))
        k = int(getattr(est, "k_frames", 1))
        forwards = []
        with self._stream_ctx():
            for i in range(k - 1 + WARMUP_PASSES):
                out = est.update(Frame(image, i * dt, -1 - i), state, None)
                if out.history_ok:
                    forwards.append(float(out.timings_s.get("forward_s", 0.0)))
        self._sync_stream()
        est.reset()
        return float(np.median(forwards)) if forwards else None

    def start(self) -> None:
        """Warm up, then (with a frame source) start the worker and wait for a first estimate."""
        if self._thread is not None:
            return
        self.forward_s = self.warm_up()
        duty = gpu_duty(self.forward_s, self.rate_hz)
        log.info(
            "eyes: rate %.1f Hz, stream %s, forward %s ms, duty %s",
            self.rate_hz,
            "on" if self._stream is not None else "off",
            "?" if self.forward_s is None else f"{1e3 * self.forward_s:.1f}",
            "?" if duty is None else f"{100 * duty:.0f}%",
        )
        check_duty(duty)
        if self.frame_source is None:
            return
        self._stop.clear()
        self._worker_error = None
        self._thread = threading.Thread(target=self._worker, name="eyes-worker", daemon=True)
        self._thread.start()
        deadline = self.clock() + self.warmup_timeout_s
        while self._latest() is None:
            if self._worker_error is not None:
                raise self._worker_error
            if self.clock() >= deadline:
                log.warning(
                    "eyes: no valid estimate within %.1f s; the runner fails safe until one "
                    "arrives",
                    self.warmup_timeout_s,
                )
                break
            time.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset(self) -> None:
        """Drop the held estimate and the history (a telemetry discontinuity / respawn)."""
        with self._lock:
            self._published = None
            self._generation += 1
            self._last_stamp = None
            if self._thread is not None:
                self._reset_pending = True  # applied by the thread that owns the estimator
                return
        self.estimator.reset()

    def _worker(self) -> None:
        last_index = None
        next_t = -math.inf
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            try:
                with self._lock:
                    pending, self._reset_pending = self._reset_pending, False
                if pending:
                    self.estimator.reset()
                frame = self.frame_source.latest()
                state = self.state_source.latest() if self.state_source is not None else None
                if frame is None or state is None or frame.index == last_index:
                    time.sleep(_POLL_S)
                    continue
                now = self.clock()
                if now < next_t:
                    time.sleep(min(_POLL_S, next_t - now))
                    continue
                last_index = frame.index
                next_t = now + period
                with self._lock:
                    target, gen = self._target, self._generation
                with self._stream_ctx():
                    self._process(frame, state, target, gen)
            except BaseException as exc:  # noqa: BLE001 - surfaced by start() / update()
                self._worker_error = exc
                log.exception("eyes worker failed")
                return

    def _process(self, frame: Frame, state: FlightState, target: Any, gen: int) -> None:
        """One estimator update; publishes the result when its history is complete."""
        t0 = time.perf_counter()
        out = self.estimator.update(frame, state, target)
        self._sync_stream()
        inference_s = time.perf_counter() - t0
        if not out.history_ok:
            return  # never replaces the held estimate
        if self.obstacle_mode == "ray_clusters" and out.rays_world is not None:
            ov = obstacle_vision_from_rays(
                state.pos,
                out.rays_world,
                out.free,
                self.ray_range,
                self.cluster_link_m,
                self.min_obstacle_size_m,
            )
        else:
            ov = np.zeros(8, np.float32)
        with self._lock:
            if gen == self._generation:  # a reset() during the update discards it
                self._published = (out, ov, int(frame.index), inference_s)

    def _latest(self) -> tuple[Any, np.ndarray, int, float] | None:
        with self._lock:
            return self._published

    # ------------------------------------------------------------------ estimation

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        with self._lock:
            self._target = None if target is None else np.array(target, np.float32)
            gen = self._generation
        if self.frame_source is None and frame is not None:
            if self._last_stamp is None or float(frame.stamp) != self._last_stamp:
                self._last_stamp = float(frame.stamp)
                self._process(frame, state, self._target, gen)  # synchronous mode
        if self._worker_error is not None:
            raise RuntimeError("eyes worker failed") from self._worker_error
        latest = self._latest()
        n = self.n_rays
        if latest is None:
            # No valid estimate yet: stamped -inf, which the runner's age check rejects.
            return FreeSpaceEstimate(
                obstacle_vision=np.zeros(8, np.float32),
                raycast=np.ones(n, np.float32),
                in_fov_mask=np.zeros(n, bool),
                stamp=-np.inf,
                info={"eyes": "none"},
            )
        out, ov, frame_index, inference_s = latest
        conf = np.asarray(out.conf, np.float32)
        in_view = np.asarray(out.in_view, bool)
        fwd = out.timings_s.get("forward_s")
        return FreeSpaceEstimate(
            obstacle_vision=ov,
            raycast=np.clip(np.asarray(out.free, np.float32), 0.0, 1.0).astype(np.float32),
            in_fov_mask=in_view,
            stamp=float(out.stamp),
            info={
                "frame_index": frame_index,
                # eyes worker wall time per frame, includes the stream sync
                "inference_s": inference_s,
                "preprocess_s": out.timings_s.get("preprocess_s"),
                "forward_s": fwd,
                "history_ok": bool(out.history_ok),
                "conf_mean": float(conf.mean()) if conf.size else 0.0,
                "conf": [round(float(c), 3) for c in conf],
                "coverage": float(in_view.mean()) if in_view.size else 0.0,
                "weights": self.weights,
                "rate_hz": self.rate_hz,
                "stream": self._stream is not None,
                "gpu_duty_est": gpu_duty(
                    self.forward_s if self.forward_s is not None else fwd, self.rate_hz
                ),
            },
        )
