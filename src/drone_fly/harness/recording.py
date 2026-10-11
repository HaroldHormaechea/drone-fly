"""Per-run recording under ``artifacts/harness/<harness>-<UTC stamp>/`` (never under ``training/``).

Layout::

    config.yaml      merged run configuration
    meta.json        checkpoint path + sha256, extra_dim, device, git HEAD, contract dir,
                     stop reason, tick / missed-deadline / dropped-record counts
    telemetry.csv    raw telemetry samples (source frame, unconverted)
    ticks.jsonl      one line per tick: obs, extra, raw action, decoded command, sink output,
                     estimator outputs / status / ages, per-stage latencies, events
    frames/ + frames.csv   every Nth frame (N = recording.frame_every_n; 0 = metadata only),
                     written as PNG with OpenCV when available, else ``.npy``

Writes happen on a background thread fed by a bounded queue; when the queue is full the
record is dropped and counted (the control loop never blocks on disk).
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.types import Frame, TelemetrySample, TickRecord

log = logging.getLogger(__name__)

TELEMETRY_COLUMNS = (
    ["received_at", "t", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    + ["vel_x", "vel_y", "vel_z", "gyro_pitch", "gyro_roll", "gyro_yaw"]
    + ["in_throttle", "in_yaw", "in_pitch", "in_roll", "batt_charge", "batt_volts"]
    + ["rpm_0", "rpm_1", "rpm_2", "rpm_3"]
)
FRAME_COLUMNS = ["tick", "frame_index", "stamp", "width", "height", "file"]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_output_root(output_root: str | Path | None, repo_root: Path | None = None) -> Path:
    """Resolve the recording root and refuse anything inside the repo's ``training/``.

    Default: ``<repo-root>/artifacts/harness``. Without a repo root, an explicit root is
    required. The check uses the *real* path, so ``artifacts/../training/x`` is refused too.
    """
    repo = repo_root if repo_root is not None else find_repo_root()
    if output_root is None:
        if repo is None:
            raise HarnessConfigError(
                "cannot find the repo root for the default artifacts/harness folder; "
                "pass --output-root"
            )
        root = repo / "artifacts" / "harness"
    else:
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
    real = Path(root).resolve()
    guards = []
    if repo is not None:
        guards.append(repo.resolve() / "training")
    for parent in (real, *real.parents):  # any training/ that sits beside a gpu_prototype/
        if parent.name == "training" and (parent.parent / "gpu_prototype").is_dir():
            guards.append(parent)
    for g in guards:
        if _inside(real, g.resolve()):
            raise HarnessConfigError(
                f"refusing to record into {real}: it is inside {g} (training artifacts are "
                "read-only for the harness). Choose another --output-root."
            )
    return real


def to_plain(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays (and tuples) to plain YAML/JSON-safe Python."""
    if isinstance(value, np.ndarray):
        return [to_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def git_head(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _row(sample: TelemetrySample) -> list[Any]:
    def vals(arr: np.ndarray | None, n: int) -> list[Any]:
        return [""] * n if arr is None else [float(x) for x in np.asarray(arr).reshape(-1)[:n]]

    return (
        [sample.received_at, sample.t]
        + vals(sample.position, 3)
        + vals(sample.quat_xyzw, 4)
        + vals(sample.velocity, 3)
        + vals(sample.gyro, 3)
        + vals(sample.inputs, 4)
        + vals(sample.battery, 2)
        + vals(sample.rpm, 4)
    )


class RunRecorder:
    """Background writer for one run folder."""

    def __init__(
        self,
        output_root: str | Path | None,
        harness: str,
        config: dict[str, Any],
        meta: dict[str, Any],
        *,
        queue_size: int = 2048,
        repo_root: Path | None = None,
    ) -> None:
        root = resolve_output_root(output_root, repo_root)
        try:  # serialise first, so a bad config fails before the run folder exists
            config_text = yaml.safe_dump(to_plain(config), sort_keys=False)
        except yaml.YAMLError as exc:
            raise HarnessConfigError(f"run config is not YAML-serialisable: {exc}") from exc
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_dir = root / f"{harness}-{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "frames").mkdir()
        self.meta = dict(meta)
        self.meta.setdefault("harness", harness)
        self.meta.setdefault("started_utc", stamp)
        self.meta.setdefault("git_head", git_head(repo_root or find_repo_root()))
        self.dropped = 0
        self._q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=queue_size)
        (self.run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
        self._write_meta()
        self._ticks = open(self.run_dir / "ticks.jsonl", "w", encoding="utf-8")  # noqa: SIM115
        self._tele_fh = open(self.run_dir / "telemetry.csv", "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._tele = csv.writer(self._tele_fh)
        self._tele.writerow(TELEMETRY_COLUMNS)
        self._frames_fh = open(self.run_dir / "frames.csv", "w", newline="", encoding="utf-8")  # noqa: SIM115
        self._frames = csv.writer(self._frames_fh)
        self._frames.writerow(FRAME_COLUMNS)
        self._closed = False
        self._thread = threading.Thread(target=self._worker, name="harness-recorder", daemon=True)
        self._thread.start()

    # -- producer side (runner thread) --

    def _put(self, kind: str, payload: Any) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait((kind, payload))
        except queue.Full:
            self.dropped += 1

    def record_tick(self, record: TickRecord) -> None:
        self._put("tick", record)

    def record_telemetry(self, samples: list[TelemetrySample]) -> None:
        for s in samples:
            self._put("telemetry", s)

    def record_frame(self, tick: int, frame: Frame, *, save_image: bool) -> None:
        self._put("frame", (tick, frame, save_image))

    def update_meta(self, **fields: Any) -> None:
        self.meta.update(fields)

    # -- consumer side (writer thread) --

    def _worker(self) -> None:
        while True:
            kind, payload = self._q.get()
            try:
                if kind == "stop":
                    return
                if kind == "tick":
                    self._ticks.write(json.dumps(payload.as_dict()) + "\n")
                elif kind == "telemetry":
                    self._tele.writerow(_row(payload))
                elif kind == "frame":
                    self._write_frame(*payload)
            except Exception:  # noqa: BLE001 - recording must never kill the run
                log.exception("recording write failed (%s)", kind)
            finally:
                self._q.task_done()

    def _write_frame(self, tick: int, frame: Frame, save_image: bool) -> None:
        h, w = frame.image.shape[:2]
        name = ""
        if save_image:
            try:
                import cv2  # optional

                name = f"frames/{tick:07d}.png"
                cv2.imwrite(str(self.run_dir / name), frame.image)
            except ImportError:
                name = f"frames/{tick:07d}.npy"
                np.save(self.run_dir / name, frame.image)
        self._frames.writerow([tick, frame.index, frame.stamp, w, h, name])

    def _write_meta(self) -> None:
        self.meta["dropped_records"] = self.dropped
        (self.run_dir / "meta.json").write_text(
            json.dumps(to_plain(self.meta), indent=2, default=str), encoding="utf-8"
        )

    def close(self, **final_meta: Any) -> Path:
        if self._closed:
            return self.run_dir
        self._closed = True
        self._q.put(("stop", None))  # blocking put: the worker drains everything before it
        self._thread.join(timeout=30)
        for fh in (self._ticks, self._tele_fh, self._frames_fh):
            fh.close()
        self.meta.update(final_meta)
        self._write_meta()
        return self.run_dir
