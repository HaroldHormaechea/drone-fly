"""Learned gate estimator (``learned``): keypoint network per gate type + PnP + the shared tracker.

Per frame (:meth:`LearnedGateEstimator.detect`, image already masked by the base class):

1. resize with ``torch.nn.functional.interpolate`` to ``gates.learned.input_width`` (no OpenCV on
   this path), ImageNet-normalise, run :class:`~drone_fly.harness.liftoff.gate_net.GateKeypointNet`
   (``fp16`` on CUDA), decode heatmap peaks;
2. keep each detection's keypoints that are predicted visible, inside the image and outside every
   mask; at least ``min_keypoints`` are needed;
3. planar PnP (IPPE, OpenCV) of those keypoints against the class's object points from the
   ``gate_types`` table -> camera-frame pose -> world :class:`GateDetection` with the class's
   aperture. Tracking, pass, lost and acquire semantics are the base class's
   (:class:`~drone_fly.harness.liftoff.gate_cv.TrackingGateEstimator`).

Network outputs are cached per ``frame.index`` (the loop may tick faster than frames arrive), and
``info["inference_s"]`` records the CUDA-synchronised network time of the latest frame.

With ``gates.learned.async_inference: true`` the network runs on a background thread instead
(like the depth estimator): each tick hands the newest masked frame to the worker and uses the
latest *finished* detections, so a 20-40 ms forward pass no longer blocks the 50 Hz loop. PnP,
tracking and the world transform still run per tick with the current state; ``info`` reports
``detection_lag_frames`` (newest frame index minus the frame the detections came from).

The weights file (``torch.save`` dict: ``state_dict`` + ``meta``) is written by
``drone-fly harness train-gates``. Its metadata (classes, keypoints, gate-type hash, backbone +
config, input width, held-out split, captures, git head) is validated against the configured
``gate_types`` when the estimator is built, and the network is loaded and warmed up there too —
so a missing or mismatched file refuses the run before the pad is armed.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import (
    find_repo_root,
    require_bool,
    require_choice,
    require_int,
    require_number,
)
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.liftoff.gate_cv import (
    GateDetection,
    TrackingGateEstimator,
    camera_from_config,
    solve_planar_pnp,
)
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame

log = logging.getLogger(__name__)

TRAIN_HINT = (
    "train one with `drone-fly harness train-gates` (docs/adr/0003-real-liftoff-calibration.md)"
)


RELEASE_URL = "https://github.com/HaroldHormaechea/drone-fly/releases/download/liftoff-gates-v1"

# Published weights (GitHub release ``liftoff-gates-v1``): file name -> sha256. A configured weights
# path whose file name is listed here is downloaded from the release on first use when missing.
RELEASED_WEIGHTS = {
    "liftoff-gates-deploy.pt": "753452e742973bfe666f30c3c91bfce74bf7108f04d54930c0bcc7e623d94e00",
    "liftoff-gates.pt": "161f7a83aa2a2866b5efd3c34b096ccda6ba87e201ecffece2833a5a4be5c111",
}


def _urlretrieve(url: str, dest: Path) -> None:
    import urllib.request

    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https release URL


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_released_weights(dest: Path, download: Any = None) -> Path:
    """Fetch the released weights named ``dest.name`` into ``dest`` and verify their sha256.

    Downloads to a temporary sibling and renames it only after the checksum matches, so an
    interrupted or corrupted download never leaves a file at ``dest``.
    """
    expected = RELEASED_WEIGHTS.get(dest.name)
    if expected is None:
        raise HarnessConfigError(f"no published gate weights named {dest.name!r}; {TRAIN_HINT}")
    url = f"{RELEASE_URL}/{dest.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    log.info("downloading gate weights %s -> %s", url, dest)
    try:
        (download or _urlretrieve)(url, tmp)
        actual = _sha256(tmp)
        if actual != expected:
            raise HarnessConfigError(
                f"downloaded gate weights {dest.name} failed the checksum "
                f"(sha256 {actual}, expected {expected}); not using them"
            )
        tmp.replace(dest)
    except HarnessConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 - network/disk errors become a clear config error
        raise HarnessConfigError(
            f"gate weights not found: {dest}, and downloading {url} failed: {exc}; {TRAIN_HINT}"
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def resolve_weights_path(path: str | Path | None, download: Any = None) -> Path:
    """``gates.learned.weights``: absolute, or relative to the repo root (else the cwd).

    A missing file whose name is one of :data:`RELEASED_WEIGHTS` is downloaded from the GitHub
    release (sha256-verified) into that path; any other missing file is a config error.
    """
    if not path:
        raise HarnessConfigError(f"gates.learned.weights is not set; {TRAIN_HINT}")
    p = Path(path).expanduser()
    if not p.is_absolute():
        root = find_repo_root()
        p = (root if root is not None else Path.cwd()) / p
    if p.is_file():
        return p
    if p.name in RELEASED_WEIGHTS:
        return download_released_weights(p, download=download)
    raise HarnessConfigError(f"gate weights not found: {p}; {TRAIN_HINT}")


def load_weights(path: Path, table: GateTypeTable, device: str = "cpu") -> tuple[Any, dict]:
    """Load + validate a weights file against ``table``; returns ``(net, meta)`` (eval mode)."""
    torch = lazy_import("torch", purpose="the learned gate estimator")
    from drone_fly.harness.liftoff.gate_net import build_from_meta

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        meta, state = blob["meta"], blob["state_dict"]
    except Exception as exc:  # noqa: BLE001 - any unreadable file is a config problem
        raise HarnessConfigError(f"cannot read gate weights {path}: {exc}; {TRAIN_HINT}") from exc
    sig = table.signature()
    if meta.get("classes") != sig["classes"]:
        raise HarnessConfigError(
            f"gate weights {path} were trained for classes {meta.get('classes')}, but gate_types "
            f"configures {sig['classes']}; retrain ({TRAIN_HINT})"
        )
    if meta.get("gate_types_hash") != table.hash():
        raise HarnessConfigError(
            f"gate weights {path} do not match the configured gate_types (keypoints / flip pairs "
            f"changed: hash {meta.get('gate_types_hash')} != {table.hash()}); "
            f"retrain ({TRAIN_HINT})"
        )
    net = build_from_meta(meta)
    net.load_state_dict(state)
    net.to(device).eval()
    return net, meta


class LearnedGateEstimator(TrackingGateEstimator):
    def __init__(
        self,
        net: Any,
        meta: dict[str, Any],
        table: GateTypeTable,
        camera: Any,
        scale: WorldScale | None = None,
        *,
        device: str = "cpu",
        fp16: bool = False,
        input_width: int = 768,
        score_threshold: float = 0.3,
        min_keypoints: int = 4,
        max_detections: int = 8,
        masks: FrameMasks | None = None,
        async_inference: bool = False,
        **tracker: Any,
    ) -> None:
        scale = scale or WorldScale(1.0)
        super().__init__(
            camera,
            scale,
            default_aperture=scale.length(table.types[0].min_aperture),
            masks=masks,
            **tracker,
        )
        self.torch = lazy_import("torch", purpose="the learned gate estimator")
        self.net = net
        self.meta = meta
        self.table = table
        self.device = device
        self.fp16 = bool(fp16 and str(device).startswith("cuda"))
        if self.fp16:
            self.net.half()
        self.input_width = int(input_width)
        self.score_threshold = float(score_threshold)
        self.min_keypoints = int(min_keypoints)
        self.max_detections = int(max_detections)
        self._cache_index: int | None = None
        self._cache: list[dict[str, Any]] = []
        self._frame_index: int | None = None
        self.weights_path: Path | None = None
        self.inference_s: float | None = None
        self.async_inference = bool(async_inference)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._pending: tuple[int | None, np.ndarray] | None = None
        self._done: tuple[int | None, list[dict[str, Any]]] | None = None
        self._worker: threading.Thread | None = None
        self._stopping = False
        self._worker_error: BaseException | None = None
        self._latest_index: int | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> LearnedGateEstimator:
        conf = ctx.config
        torch = lazy_import("torch", purpose="the learned gate estimator")
        lc = cfg.get("learned") or {}
        table = GateTypeTable.from_config(conf.get("gate_types"))
        dev = require_choice(
            lc.get("device", "auto"), "gates.learned.device", ("auto", "cpu", "cuda")
        )
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        weights_path = resolve_weights_path(lc.get("weights"))
        net, meta = load_weights(weights_path, table, dev)
        cam = conf.section("camera")
        gpl = cfg.get("gates_per_lap")
        num = lambda k, d, **kw: float(require_number(cfg.get(k, d), f"gates.{k}", **kw))  # noqa: E731
        est = cls(
            net,
            meta,
            table,
            camera_from_config(cam),
            WorldScale(conf.world_scale),
            device=dev,
            fp16=require_bool(lc.get("fp16", True), "gates.learned.fp16"),
            input_width=require_int(
                lc.get("input_width", meta.get("input_width", 768)),
                "gates.learned.input_width",
                min_value=64,
            ),
            score_threshold=float(
                require_number(
                    lc.get("score_threshold", 0.3),
                    "gates.learned.score_threshold",
                    min_value=0.0,
                    max_value=1.0,
                )
            ),
            min_keypoints=require_int(
                lc.get("min_keypoints", 4), "gates.learned.min_keypoints", min_value=4
            ),
            max_detections=require_int(
                lc.get("max_detections", 8), "gates.learned.max_detections", min_value=1
            ),
            masks=FrameMasks.from_config(cam),
            async_inference=require_bool(
                lc.get("async_inference", False), "gates.learned.async_inference"
            ),
            assoc_radius=num("assoc_radius", 1.0, min_value=0.0, min_exclusive=True),
            lost_timeout_s=num("lost_timeout_s", 1.0, min_value=0.0, min_exclusive=True),
            acquire_timeout_s=num("acquire_timeout_s", 3.0, min_value=0.0),
            pass_area_frac=num("pass_area_frac", 0.2, min_value=0.0, max_value=1.0),
            gates_per_lap=None if gpl is None else int(gpl),
        )
        est.weights_path = weights_path
        est.warmup()
        return est

    # ------------------------------------------------------------------ network

    def warmup(self) -> None:
        """One forward pass at the configured camera resolution (CUDA kernels, cuDNN autotune)."""
        blank = np.zeros((self.camera.height, self.camera.width, 3), np.uint8)
        self.infer(blank)
        self.inference_s = None

    def infer(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Decoded detections in **frame** pixels for one (masked) BGR image."""
        from drone_fly.harness.liftoff.gate_net import decode, prepare_input

        torch = self.torch
        cuda = str(self.device).startswith("cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            x, scale = prepare_input(image, self.input_width, self.device)
            if self.fp16:
                x = x.half()
            out = self.net(x)
            dets = decode(
                out, score_threshold=self.score_threshold, max_detections=self.max_detections
            )[0]
        if cuda:
            torch.cuda.synchronize()
        self.inference_s = time.perf_counter() - t0
        for d in dets:
            d["center"] = d["center"] / scale
            d["keypoints"] = d["keypoints"] / scale
        return dets

    def raw_detections(self, image: np.ndarray, frame_index: int | None = None) -> list[dict]:
        """:meth:`infer`, cached per ``frame_index`` (``None`` = always run).

        In async mode: queue ``image`` for the worker (newest frame wins) and return the latest
        finished detections, possibly from an older frame (empty until the first one finishes).
        """
        if self.async_inference:
            return self._raw_detections_async(image, frame_index)
        if frame_index is not None and frame_index == self._cache_index:
            return self._cache
        dets = self.infer(image)
        self._cache_index, self._cache = frame_index, dets
        return dets

    # ------------------------------------------------------------------ async worker

    def start(self) -> None:
        if self.async_inference and self._worker is None:
            self._stopping = False
            self._worker = threading.Thread(
                target=self._worker_loop, name="gate-inference", daemon=True
            )
            self._worker.start()

    def stop(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._stopping = True
        self._wake.set()
        worker.join(timeout=2.0)
        self._worker = None

    def _worker_loop(self) -> None:
        while not self._stopping:
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            index, image = job
            try:
                dets = self.infer(image)
            except BaseException as exc:  # noqa: BLE001 - surfaced on the next tick
                with self._lock:
                    self._worker_error = exc
                return
            with self._lock:
                self._done = (index, dets)

    def _raw_detections_async(self, image: np.ndarray, frame_index: int | None) -> list[dict]:
        if self._worker is None:
            self.start()
        with self._lock:
            if self._worker_error is not None:
                raise RuntimeError("gate inference worker failed") from self._worker_error
            if frame_index is None or frame_index != self._latest_index:
                self._pending = (frame_index, image)
                self._latest_index = frame_index
                self._wake.set()
            done = self._done
        if done is None:
            return []
        self._cache_index, self._cache = done
        return done[1]

    def detection_lag_frames(self) -> int | None:
        """Newest submitted frame index minus the index the current detections came from."""
        if not self.async_inference or self._latest_index is None or self._cache_index is None:
            return None
        return int(self._latest_index) - int(self._cache_index)

    def detect_frame(self, frame: Frame, state: FlightState) -> list[GateDetection]:
        self._frame_index = frame.index
        try:
            return super().detect_frame(frame, state)
        finally:
            self._frame_index = None

    def extra_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {} if self.inference_s is None else {"inference_s": self.inference_s}
        lag = self.detection_lag_frames()
        if lag is not None:
            info["detection_lag_frames"] = lag
        return info

    # ------------------------------------------------------------------ detection

    def detect(self, image: np.ndarray, state: FlightState) -> list[GateDetection]:
        cv2 = lazy_import(
            "cv2", package="opencv-python", purpose="PnP in the learned gate estimator"
        )
        camera = self._camera_for(image)
        h, w = image.shape[:2]
        raw = self.raw_detections(image, self._frame_index)
        out = []
        for d in raw:
            gtype = self.table.types[d["cls"]]
            k = gtype.n_keypoints
            kp = d["keypoints"][:k]
            keep = (
                (d["visibility"][:k] >= 0.5)
                & (kp[:, 0] >= 0)
                & (kp[:, 0] <= w - 1)
                & (kp[:, 1] >= 0)
                & (kp[:, 1] <= h - 1)
            )
            if self.masks:
                keep &= ~self.masks.contains(kp, w, h)
            if int(keep.sum()) < self.min_keypoints:
                continue
            try:
                t_cam, n_cam = solve_planar_pnp(gtype.object_points()[keep], kp[keep], camera)
            except (ValueError, cv2.error):
                continue
            pts = kp[keep].astype(np.float32)
            area = float(cv2.contourArea(cv2.convexHull(pts))) / float(w * h)
            det = self.detection_from_pose(
                t_cam,
                n_cam,
                kp[keep],
                area,
                state,
                camera,
                gtype.min_aperture,
                cls=gtype.cls,
                pixel_center=d["center"],
            )
            if det is not None:
                out.append(det)
        return out
