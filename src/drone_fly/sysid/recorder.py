"""The sysid session folder: ``artifacts/sysid/<drone-id>/<session>/`` (gitignored).

* ``telemetry.csv`` — every raw telemetry sample (``recording.TELEMETRY_COLUMNS``: receive time on
  the session clock, game time, Unity-frame pose / velocity / gyro, echoed inputs, battery, RPM)
  plus the ``segment`` it belongs to (a resume starts a new segment)
* ``commands.csv`` — every stick command: session-clock and wall time, segment, experiment id,
  phase, the four channel sticks
* ``meta.json``     — written atomically after every experiment (see :class:`SessionRecorder.meta`)
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from drone_fly.harness.recording import TELEMETRY_COLUMNS, to_plain
from drone_fly.harness.recording import _row as telemetry_row

TELEMETRY_FILE = "telemetry.csv"
COMMANDS_FILE = "commands.csv"
META_FILE = "meta.json"
TELEMETRY_HEADER = [*TELEMETRY_COLUMNS, "segment"]
COMMAND_HEADER = [
    "perf_t",
    "wall_t",
    "segment",
    "experiment",
    "phase",
    "throttle",
    "roll",
    "pitch",
    "yaw",
]


def default_session_name() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class SessionRecorder:
    """Appends to the session CSVs (resume-safe) and rewrites ``meta.json`` atomically."""

    def __init__(self, session_dir: Path, *, clock: Any = None, segment: int = 0) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.segment = segment
        self.meta: dict[str, Any] = {}
        if (self.dir / META_FILE).is_file():
            self.meta = json.loads((self.dir / META_FILE).read_text(encoding="utf-8"))
        self._tel_fh, self._tel = self._open(TELEMETRY_FILE, TELEMETRY_HEADER)
        self._cmd_fh, self._cmd = self._open(COMMANDS_FILE, COMMAND_HEADER)

    def _open(self, name: str, header: list[str]) -> tuple[Any, Any]:
        path = self.dir / name
        new = not path.is_file() or path.stat().st_size == 0
        fh = open(path, "a", newline="", encoding="utf-8")  # noqa: SIM115 - closed in close()
        writer = csv.writer(fh)
        if new:
            writer.writerow(header)
        return fh, writer

    def record_telemetry(self, samples: list[Any]) -> None:
        for s in samples:
            self._tel.writerow([*telemetry_row(s), self.segment])

    def record_command(self, perf_t: float, experiment: str, phase: str, sticks: dict) -> None:
        self._cmd.writerow(
            [
                f"{perf_t:.6f}",
                f"{time.time():.6f}",
                self.segment,
                experiment,
                phase,
                *(f"{float(sticks.get(c, 0.0)):.5f}" for c in ("throttle", "roll", "pitch", "yaw")),
            ]
        )

    def write_meta(self, meta: dict[str, Any] | None = None) -> Path:
        if meta is not None:
            self.meta = meta
        self.flush()
        path = self.dir / META_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(to_plain(self.meta), indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def flush(self) -> None:
        for fh in (self._tel_fh, self._cmd_fh):
            if not fh.closed:
                fh.flush()

    def close(self) -> None:
        for fh in (self._tel_fh, self._cmd_fh):
            if not fh.closed:
                fh.close()


class RecordingSource:
    """Wraps a state source so every raw sample drained (by anyone) lands in ``telemetry.csv``."""

    def __init__(self, source: Any, recorder: SessionRecorder) -> None:
        self.source = source
        self.recorder = recorder
        self.last_sample: Any = None
        self.pending: list[Any] = []  # drained but not yet seen by the session's crash check

    def drain_raw(self) -> list[Any]:
        samples = self.source.drain_raw()
        if samples:
            self.recorder.record_telemetry(samples)
            self.last_sample = samples[-1]
            self.pending.extend(samples)
        return samples

    def take_pending(self) -> list[Any]:
        out, self.pending = self.pending, []
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)
