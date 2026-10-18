"""Per-run recording under ``artifacts/harness/<harness>-<UTC stamp>/`` (never under ``training/``).

Layout::

    config.yaml      merged run configuration
    meta.json        checkpoint path + sha256, extra_dim, device, GPU, git HEAD, contract dir,
                     stop reason, tick / missed-deadline / dropped-record counts, fail-safe
                     result and the sink's neutral output (``failsafe_output``)
    telemetry.csv    raw telemetry samples (source frame, unconverted)
    ticks.jsonl      one line per tick: obs, extra, raw action, decoded command, sink output,
                     estimator outputs / status / ages, per-stage latencies, events
    frames/ + frames.csv   every Nth frame (N = recording.frame_every_n; 0 = metadata only),
                     written as PNG with OpenCV when available, else ``.npy``
    latency_summary.json   written on close: p50/p95/p99/max per stage (state, gate detector,
                     depth worker + update, policy, actuation, live-overlay publish, total),
                     effective rate, missed
                     deadlines, estimate ages (:func:`summarize_ticks`; ``drone-fly harness
                     latency-report`` prints it as a table)

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
LATENCY_FILE = "latency_summary.json"

#: latency_summary.json rows: name -> where the per-tick value comes from (seconds).
LATENCY_STAGES = (
    ("state", "latency_s", "state"),
    ("gate_detector", "perception_s", "gate"),
    ("gate_inference", "gate_info", "inference_s"),
    ("depth_worker", "free_space_info", "inference_s"),
    ("depth_update", "perception_s", "free_space"),
    ("perception", "latency_s", "perception"),
    ("policy", "latency_s", "inference"),
    ("actuation", "latency_s", "actuation"),
    ("overlay", "latency_s", "overlay"),
    ("total", "latency_s", "total"),
)
#: Stages reported only when at least one tick carries them (a run without the live overlay
#: keeps the pre-UC-65 summary shape).
OPTIONAL_STAGES = frozenset({"overlay"})
AGE_KEYS = ("telemetry", "frame", "gate", "free_space")
#: UC-86 per-provider rows: ``perception_s`` / ``ages_s`` keys with this prefix (the runner's
#: PerceptionRuntime writes ``provider:<name>`` and ``provider:<name>:frame``).
PROVIDER_PREFIX = "provider:"


def _stats_ms(values: list[float]) -> dict[str, float] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], np.float64)
    if arr.size == 0:
        return None
    ms = arr * 1000.0
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "max": float(ms.max()),
    }


def summarize_ticks(ticks: list[dict[str, Any]], rate_hz: float | None = None) -> dict[str, Any]:
    """Per-stage latency / age percentiles (ms) over ``ticks.jsonl`` records.

    The depth worker's inference time is counted once per depth frame (it is reported on every
    tick that reuses the same result). Per-provider rows (``provider:<name>``) are added only
    when the ticks carry them.
    """
    series: dict[str, list[float]] = {name: [] for name, _, _ in LATENCY_STAGES}
    ages: dict[str, list[float]] = {k: [] for k in AGE_KEYS}
    seen_depth: set[Any] = set()
    provider_lat: dict[str, list[float]] = {}
    provider_age: dict[str, list[float]] = {}
    times: list[float] = []
    missed = 0
    for t in ticks:
        missed += bool(t.get("missed_deadline"))
        clock = t.get("sim_time") if t.get("sim_time") is not None else t.get("wall_time")
        if clock is not None:
            times.append(float(clock))
        gate_info = (t.get("gate") or {}).get("info") or {}
        fs_info = (t.get("free_space") or {}).get("info") or {}
        src = {
            "latency_s": t.get("latency_s") or {},
            "perception_s": t.get("perception_s") or {},
            "gate_info": gate_info,
            "free_space_info": fs_info,
        }
        for name, where, key in LATENCY_STAGES:
            value = src[where].get(key)
            if value is None:
                continue
            if name == "depth_worker":
                frame_key = fs_info.get("frame_index")
                if frame_key in seen_depth:
                    continue
                seen_depth.add(frame_key)
            series[name].append(float(value))
        for k in AGE_KEYS:
            v = (t.get("ages_s") or {}).get(k)
            if v is not None:
                ages[k].append(float(v))
        for out, values in ((provider_lat, src["perception_s"]), (provider_age, t.get("ages_s"))):
            for k, v in (values or {}).items():
                if k.startswith(PROVIDER_PREFIX) and v is not None:
                    out.setdefault(k, []).append(float(v))
    n = len(ticks)
    span = (max(times) - min(times)) if len(times) > 1 else 0.0
    return {
        "ticks": n,
        "rate_hz_target": rate_hz,
        "effective_rate_hz": (len(times) - 1) / span if span > 0 else None,
        "missed_deadlines": missed,
        "missed_deadline_rate": missed / n if n else 0.0,
        "latency_ms": {
            **{n: _stats_ms(v) for n, v in series.items() if v or n not in OPTIONAL_STAGES},
            **{k: _stats_ms(v) for k, v in provider_lat.items()},
        },
        "ages_ms": {
            **{k: _stats_ms(v) for k, v in ages.items()},
            **{k: _stats_ms(v) for k, v in provider_age.items()},
        },
    }


def latency_markdown(summary: dict[str, Any]) -> str:
    """``summary`` (see :func:`summarize_ticks`) as a Markdown table."""
    lines = [
        "| stage | n | p50 ms | p95 ms | p99 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, st: dict[str, float] | None) -> str:
        if st is None:
            return f"| {name} | 0 | - | - | - | - |"
        return (
            f"| {name} | {st['n']} | {st['p50']:.1f} | {st['p95']:.1f} | {st['p99']:.1f} "
            f"| {st['max']:.1f} |"
        )

    for name, st in summary.get("latency_ms", {}).items():
        lines.append(row(name, st))
    for name, st in summary.get("ages_ms", {}).items():
        lines.append(row(f"age: {name}", st))
    rate = summary.get("effective_rate_hz")
    target = summary.get("rate_hz_target")
    rate_text = "-" if rate is None else f"{rate:.1f}"
    target_text = f" (target {target:g} Hz)" if target else ""
    lines.append("")
    lines.append(
        f"ticks {summary.get('ticks', 0)}, effective rate {rate_text} Hz{target_text}, "
        f"missed deadlines {summary.get('missed_deadlines', 0)} "
        f"({100 * summary.get('missed_deadline_rate', 0.0):.1f} %)"
    )
    if summary.get("gpu"):
        lines.append(f"GPU: {summary['gpu']}")
    return "\n".join(lines)


def load_latency_summary(run_dir: str | Path, *, recompute: bool = False) -> dict[str, Any]:
    """``latency_summary.json`` of a run folder, recomputed from ``ticks.jsonl`` when missing."""
    run = Path(run_dir)
    path = run / LATENCY_FILE
    if path.is_file() and not recompute:
        return json.loads(path.read_text(encoding="utf-8"))
    ticks_path = run / "ticks.jsonl"
    if not ticks_path.is_file():
        raise HarnessConfigError(f"{run} has no ticks.jsonl (is it a harness run folder?)")
    ticks = [
        json.loads(line)
        for line in ticks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    cfg_path = run / "config.yaml"
    rate = None
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        rate = (cfg.get("runner") or {}).get("rate_hz")
    summary = summarize_ticks(ticks, rate)
    summary["gpu"] = meta.get("gpu")
    return summary


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


def _latency_view(d: dict[str, Any]) -> dict[str, Any]:
    """The subset of a tick dict :func:`summarize_ticks` reads (kept in memory until close)."""
    gate_info = (d.get("gate") or {}).get("info") or {}
    fs_info = (d.get("free_space") or {}).get("info") or {}
    return {
        "sim_time": d.get("sim_time"),
        "wall_time": d.get("wall_time"),
        "missed_deadline": d.get("missed_deadline"),
        "latency_s": d.get("latency_s"),
        "perception_s": d.get("perception_s"),
        "ages_s": d.get("ages_s"),
        "gate": {"info": {"inference_s": gate_info.get("inference_s")}},
        "free_space": {
            "info": {k: fs_info.get(k) for k in ("inference_s", "frame_index")},
        },
    }


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
        rate_hz: float | None = None,
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
        self.rate_hz = rate_hz
        self._tick_views: list[dict[str, Any]] = []
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
                    d = payload.as_dict()
                    self._ticks.write(json.dumps(d) + "\n")
                    self._tick_views.append(_latency_view(d))
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
        try:
            summary = summarize_ticks(self._tick_views, self.rate_hz)
            summary["gpu"] = self.meta.get("gpu")
            (self.run_dir / LATENCY_FILE).write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - a summary failure must not lose the run folder
            log.exception("writing %s failed", LATENCY_FILE)
        return self.run_dir
