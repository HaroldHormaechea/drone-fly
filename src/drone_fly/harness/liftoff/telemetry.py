"""Liftoff telemetry: UDP datagram parser, recorded-CSV reader, Unity -> contract conversion.

Liftoff streams one UDP datagram per physics tick (~100 Hz) when telemetry is enabled in
``TelemetryConfiguration.json`` (``EndPoint`` + ``StreamFormat``). Each datagram is the
configured blocks packed back-to-back as little-endian ``float32``:

==============  =====  ============================================
block           width  contents
==============  =====  ============================================
``timestamp``   1      seconds
``position``    3      Unity world x, y, z (metres)
``attitude``    4      quaternion x, y, z, w (Unity)
``velocity``    3      Unity world x, y, z (m/s)
``gyro``        3      pitch, roll, yaw rates (``telemetry.gyro_units``)
``inputs``      4      throttle, yaw, pitch, roll sticks
``battery``     2      charge fraction, volts
``motor_rpm``   4      per-motor RPM (optionally preceded by a uint8 motor count)
==============  =====  ============================================

The block order and widths are configurable (``telemetry.stream_format`` /
``telemetry.block_widths``) because they depend on the owner's Liftoff configuration.

Conversion to the contract (:class:`LiftoffConverter`): positions / velocities via
:func:`~drone_fly.harness.geometry.unity_to_contract_position` then :class:`WorldScale`;
the attitude via :func:`~drone_fly.harness.geometry.unity_to_contract_quat`; body rates via a
configurable axis / sign / unit table (``telemetry.gyro_map``) whose correctness
:mod:`~drone_fly.harness.liftoff.gyro_check` measures from the quaternion stream. The defaults
(``wx = -roll``, ``wy = +pitch``, ``wz = -yaw``, deg/s) and the MotorRPM count prefix were
measured on a live Liftoff capture in UC-64 (``docs/adr/0003-real-liftoff-calibration.md``).
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import logging
import socket
import struct
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import (
    require_bool,
    require_choice,
    require_int,
    require_number,
)
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import unity_to_contract_position, unity_to_contract_quat
from drone_fly.harness.interfaces import StateSource
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, TelemetrySample

log = logging.getLogger(__name__)

DEFAULT_STREAM_FORMAT = (
    "timestamp",
    "position",
    "attitude",
    "velocity",
    "gyro",
    "inputs",
    "battery",
    "motor_rpm",
)
DEFAULT_BLOCK_WIDTHS = {
    "timestamp": 1,
    "position": 3,
    "attitude": 4,
    "velocity": 3,
    "gyro": 3,
    "inputs": 4,
    "battery": 2,
    "motor_rpm": 4,
}
REQUIRED_BLOCKS = ("timestamp", "position", "attitude", "velocity", "gyro")
GYRO_FIELDS = ("pitch", "roll", "yaw")

#: Column layout of the recorded Liftoff telemetry CSVs.
CSV_COLUMNS = (
    ["t", "pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    + ["vel_x", "vel_y", "vel_z", "gyro_pitch", "gyro_roll", "gyro_yaw"]
    + ["in_throttle", "in_yaw", "in_pitch", "in_roll", "batt_charge", "batt_volts"]
    + ["rpm_0", "rpm_1", "rpm_2", "rpm_3"]
)


# ----------------------------------------------------------------------------- format


@dataclass(frozen=True)
class TelemetryFormat:
    """Ordered datagram blocks + widths, and whether ``motor_rpm`` has a uint8 count prefix."""

    blocks: tuple[str, ...] = DEFAULT_STREAM_FORMAT
    widths: tuple[tuple[str, int], ...] = tuple(DEFAULT_BLOCK_WIDTHS.items())
    rpm_count_prefix: bool = False

    def __post_init__(self) -> None:
        w = dict(self.widths)
        unknown = [b for b in self.blocks if b not in DEFAULT_BLOCK_WIDTHS]
        if unknown:
            raise HarnessConfigError(
                f"unknown telemetry blocks {unknown}; known: {list(DEFAULT_BLOCK_WIDTHS)}"
            )
        missing = [b for b in REQUIRED_BLOCKS if b not in self.blocks]
        if missing:
            raise HarnessConfigError(f"telemetry stream_format must include {missing}")
        fixed = {"timestamp": 1, "position": 3, "attitude": 4, "velocity": 3, "gyro": 3}
        for b, n in fixed.items():
            if w.get(b, n) != n:
                raise HarnessConfigError(f"telemetry block {b!r} must have width {n}")

    @property
    def width_map(self) -> dict[str, int]:
        return {**DEFAULT_BLOCK_WIDTHS, **dict(self.widths)}

    @property
    def size(self) -> int:
        n = 4 * sum(self.width_map[b] for b in self.blocks)
        return n + (1 if self.rpm_count_prefix and "motor_rpm" in self.blocks else 0)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TelemetryFormat:
        blocks = cfg.get("stream_format") or list(DEFAULT_STREAM_FORMAT)
        widths = {**DEFAULT_BLOCK_WIDTHS, **(cfg.get("block_widths") or {})}
        for k, v in widths.items():
            require_int(v, f"telemetry.block_widths.{k}", min_value=1)
        return cls(
            blocks=tuple(blocks),
            widths=tuple(widths.items()),
            rpm_count_prefix=require_bool(
                cfg.get("rpm_count_prefix", False), "telemetry.rpm_count_prefix"
            ),
        )

    def parse(self, data: bytes, received_at: float = 0.0) -> TelemetrySample:
        """Unpack one datagram into a :class:`TelemetrySample` (source frame, unconverted)."""
        if len(data) < self.size and not self.rpm_count_prefix:
            raise ValueError(f"telemetry datagram too short: {len(data)} < {self.size} bytes")
        widths = self.width_map
        values: dict[str, np.ndarray] = {}
        off = 0
        for b in self.blocks:
            n = widths[b]
            if b == "motor_rpm" and self.rpm_count_prefix:
                count = int(data[off])
                off += 1
                n = count or n
            values[b] = np.array(struct.unpack_from(f"<{n}f", data, off), dtype=np.float32)
            off += 4 * n
        return TelemetrySample(
            t=float(values["timestamp"][0]),
            position=values["position"],
            quat_xyzw=values["attitude"],
            velocity=values["velocity"],
            gyro=values["gyro"],
            inputs=values.get("inputs"),
            battery=values.get("battery"),
            rpm=values.get("motor_rpm"),
            received_at=received_at,
        )

    def pack(self, sample: TelemetrySample) -> bytes:
        """Inverse of :meth:`parse` (used to build synthetic datagrams)."""
        widths = self.width_map
        src = {
            "timestamp": [sample.t],
            "position": sample.position,
            "attitude": sample.quat_xyzw,
            "velocity": sample.velocity,
            "gyro": sample.gyro,
            "inputs": sample.inputs,
            "battery": sample.battery,
            "motor_rpm": sample.rpm,
        }
        out = bytearray()
        for b in self.blocks:
            n = widths[b]
            vals = src[b]
            arr = np.zeros(n, np.float32) if vals is None else np.asarray(vals, np.float32)
            if b == "motor_rpm" and self.rpm_count_prefix:
                out += bytes([len(arr)])
                n = len(arr)
            out += struct.pack(f"<{n}f", *arr.reshape(-1)[:n])
        return bytes(out)


# ----------------------------------------------------------------------------- conversion


@dataclass(frozen=True)
class GyroMap:
    """Contract body rate ``[wx, wy, wz]`` <- ``sign * gyro[source] * unit`` per axis.

    Defaults are Liftoff's measured convention: ``wx = -roll``, ``wy = +pitch``, ``wz = -yaw``.
    """

    sources: tuple[str, str, str] = ("roll", "pitch", "yaw")
    signs: tuple[float, float, float] = (-1.0, 1.0, -1.0)
    units: str = "deg_s"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> GyroMap:
        units = require_choice(
            cfg.get("gyro_units", "deg_s"), "telemetry.gyro_units", ("deg_s", "rad_s")
        )
        table = cfg.get("gyro_map") or {}
        sources, signs = [], []
        defaults = {"wx": ("roll", -1.0), "wy": ("pitch", 1.0), "wz": ("yaw", -1.0)}
        for axis, (dsrc, dsign) in defaults.items():
            entry = table.get(axis, {"source": dsrc, "sign": dsign})
            if not isinstance(entry, dict):
                raise HarnessConfigError(f"telemetry.gyro_map.{axis} must be {{source, sign}}")
            src = require_choice(
                entry.get("source"), f"telemetry.gyro_map.{axis}.source", GYRO_FIELDS
            )
            sign = require_number(entry.get("sign"), f"telemetry.gyro_map.{axis}.sign")
            if sign not in (-1.0, 1.0):
                raise HarnessConfigError(f"telemetry.gyro_map.{axis}.sign must be 1 or -1")
            sources.append(src)
            signs.append(float(sign))
        return cls(tuple(sources), tuple(signs), units)  # type: ignore[arg-type]

    def apply(self, gyro: np.ndarray) -> np.ndarray:
        g = np.asarray(gyro, np.float64).reshape(3)
        unit = np.pi / 180.0 if self.units == "deg_s" else 1.0
        return np.array(
            [
                s * g[GYRO_FIELDS.index(src)] * unit
                for src, s in zip(self.sources, self.signs, strict=True)
            ],
            dtype=np.float32,
        )

    def invert(self, omega: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`apply`: contract body rates (rad/s) -> raw ``[pitch, roll, yaw]``."""
        w = np.asarray(omega, np.float64).reshape(3)
        unit = np.pi / 180.0 if self.units == "deg_s" else 1.0
        raw = np.zeros(3, np.float64)
        for value, src, s in zip(w, self.sources, self.signs, strict=True):
            raw[GYRO_FIELDS.index(src)] = s * value / unit
        return raw.astype(np.float32)


@dataclass(frozen=True)
class LiftoffConverter:
    """Unity telemetry sample -> contract :class:`FlightState` (training scale)."""

    gyro_map: GyroMap = GyroMap()
    scale: WorldScale = WorldScale(1.0)

    def convert(self, sample: TelemetrySample, *, discontinuity: bool = False) -> FlightState:
        return FlightState(
            pos=self.scale.vector(unity_to_contract_position(sample.position)),
            vel=self.scale.vector(unity_to_contract_position(sample.velocity)),
            quat=unity_to_contract_quat(sample.quat_xyzw),
            omega=self.gyro_map.apply(sample.gyro),
            t=sample.t,
            received_at=sample.received_at,
            discontinuity=discontinuity,
        )


class DiscontinuityDetector:
    """Flags respawns / resets: ``t`` going backwards or a position jump (target metres)."""

    def __init__(self, max_position_jump_m: float = 5.0) -> None:
        self.max_jump = max_position_jump_m
        self._last: TelemetrySample | None = None

    def reset(self) -> None:
        self._last = None

    def check(self, sample: TelemetrySample) -> bool:
        last, self._last = self._last, sample
        if last is None:
            return False
        if sample.t < last.t:
            return True
        jump = float(np.linalg.norm(np.asarray(sample.position) - np.asarray(last.position)))
        return jump > self.max_jump


def converter_from_config(cfg: dict[str, Any], world_scale: float) -> LiftoffConverter:
    return LiftoffConverter(GyroMap.from_config(cfg), WorldScale(world_scale))


# ----------------------------------------------------------------------------- CSV


def read_telemetry_csv(path: str | Path) -> list[TelemetrySample]:
    """Read a recorded Liftoff telemetry CSV (:data:`CSV_COLUMNS` header; extra columns ignored)."""
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in CSV_COLUMNS[:14] if c not in header]
        if missing:
            raise HarnessConfigError(f"{p} is missing telemetry columns {missing}")

        def opt(row: dict[str, str], cols: list[str]) -> np.ndarray | None:
            if not all(c in row and row[c] not in ("", None) for c in cols):
                return None
            return np.array([float(row[c]) for c in cols], np.float32)

        out = []
        for row in reader:

            def f(*cols: str, row: dict[str, str] = row) -> np.ndarray:
                return np.array([float(row[c]) for c in cols], np.float32)

            out.append(
                TelemetrySample(
                    t=float(row["t"]),
                    position=f("pos_x", "pos_y", "pos_z"),
                    quat_xyzw=f("quat_x", "quat_y", "quat_z", "quat_w"),
                    velocity=f("vel_x", "vel_y", "vel_z"),
                    gyro=f("gyro_pitch", "gyro_roll", "gyro_yaw"),
                    inputs=opt(row, ["in_throttle", "in_yaw", "in_pitch", "in_roll"]),
                    battery=opt(row, ["batt_charge", "batt_volts"]),
                    rpm=opt(row, ["rpm_0", "rpm_1", "rpm_2", "rpm_3"]),
                    received_at=float(row.get("received_at") or row["t"]),
                )
            )
    return out


class LiftoffCsvTelemetryReader:
    """Iterate a recorded CSV as raw samples or converted states (offline / replay use)."""

    def __init__(self, path: str | Path, converter: LiftoffConverter | None = None) -> None:
        self.path = Path(path)
        self.samples = read_telemetry_csv(self.path)
        self.converter = converter or LiftoffConverter()

    def __iter__(self) -> Iterator[TelemetrySample]:
        return iter(self.samples)

    def states(self, max_position_jump_m: float = 5.0) -> list[FlightState]:
        det = DiscontinuityDetector(max_position_jump_m)
        return [self.converter.convert(s, discontinuity=det.check(s)) for s in self.samples]

    def gyro_report(self) -> Any:
        """Gyro-convention check over this recording (see :mod:`gyro_check`)."""
        from drone_fly.harness.liftoff.gyro_check import check_gyro_convention

        return check_gyro_convention(self.samples, self.converter.gyro_map)


# ----------------------------------------------------------------------------- state sources


class _LatestStateMixin:
    """Thread-safe latest state + ring buffer of raw samples + discontinuity counting."""

    def _init_latest(self, converter: LiftoffConverter, max_jump: float, ring: int) -> None:
        self.converter = converter
        self._detector = DiscontinuityDetector(max_jump)
        self._lock = threading.Lock()
        self._latest: FlightState | None = None
        self._raw: collections.deque[TelemetrySample] = collections.deque(maxlen=ring)
        self._last_sample: TelemetrySample | None = None
        self.discontinuity_count = 0
        self.samples_received = 0

    def _ingest(self, sample: TelemetrySample) -> None:
        disc = self._detector.check(sample)
        state = self.converter.convert(sample, discontinuity=disc)
        with self._lock:
            self._latest = state
            self._last_sample = sample
            self._raw.append(sample)
            self.samples_received += 1
            if disc:
                self.discontinuity_count += 1

    def latest(self) -> FlightState | None:
        with self._lock:
            return self._latest

    def age(self, now: float) -> float:
        with self._lock:
            return float("inf") if self._latest is None else now - self._latest.received_at

    def latest_sample(self) -> TelemetrySample | None:
        """The newest raw sample (not drained by ``drain_raw``; e.g. the pack reading)."""
        with self._lock:
            return self._last_sample

    def drain_raw(self) -> list[TelemetrySample]:
        with self._lock:
            out = list(self._raw)
            self._raw.clear()
        return out


class LiftoffUdpStateSource(_LatestStateMixin, StateSource):
    """Listens for Liftoff telemetry datagrams on a background thread (latest wins).

    ``received_at`` is ``time.perf_counter()`` — the same clock as the runner's
    :class:`~drone_fly.harness.runner.WallClock`.
    """

    def __init__(
        self,
        fmt: TelemetryFormat,
        converter: LiftoffConverter,
        host: str = "127.0.0.1",
        port: int = 9001,
        *,
        max_position_jump_m: float = 5.0,
        ring_size: int = 1024,
        clock: Any = time.perf_counter,
    ) -> None:
        self.fmt = fmt
        self.host = host
        self.port = port
        self.clock = clock
        self._init_latest(converter, max_position_jump_m, ring_size)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.parse_errors = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> LiftoffUdpStateSource:
        return cls(
            TelemetryFormat.from_config(cfg),
            converter_from_config(cfg, ctx.config.world_scale),
            host=str(cfg.get("host", "127.0.0.1")),
            port=require_int(cfg.get("port", 9001), "telemetry.port", min_value=0),
            max_position_jump_m=float(
                require_number(
                    cfg.get("max_position_jump_m", 5.0),
                    "telemetry.max_position_jump_m",
                    min_value=0.0,
                    min_exclusive=True,
                )
            ),
            ring_size=require_int(cfg.get("ring_size", 1024), "telemetry.ring_size", min_value=1),
        )

    @property
    def bound_port(self) -> int | None:
        return None if self._sock is None else self._sock.getsockname()[1]

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.port))
        sock.settimeout(0.1)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="liftoff-telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(4096)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            try:
                self._ingest(self.fmt.parse(data, received_at=self.clock()))
            except (ValueError, struct.error):
                self.parse_errors += 1
                if self.parse_errors in (1, 100, 10000):
                    log.warning(
                        "bad telemetry datagram (%d so far, %d bytes)", self.parse_errors, len(data)
                    )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class CsvReplayStateSource(_LatestStateMixin, StateSource):
    """Replays a recorded telemetry CSV in real time (dry runs without the game)."""

    def __init__(
        self,
        path: str | Path,
        converter: LiftoffConverter,
        *,
        loop: bool = False,
        max_position_jump_m: float = 5.0,
        clock: Any = time.perf_counter,
    ) -> None:
        self.samples = read_telemetry_csv(path)
        if not self.samples:
            raise HarnessConfigError(f"telemetry CSV {path} has no rows")
        self.loop = loop
        self.clock = clock
        self._init_latest(converter, max_position_jump_m, 1024)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> CsvReplayStateSource:
        path = cfg.get("replay_csv")
        if not path:
            raise HarnessConfigError("state source liftoff-csv needs telemetry.replay_csv")
        return cls(path, converter_from_config(cfg, ctx.config.world_scale))

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="liftoff-csv-replay", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0_wall, t0 = self.clock(), self.samples[0].t
            for s in self.samples:
                while not self._stop.is_set() and self.clock() - t0_wall < s.t - t0:
                    time.sleep(0.001)
                if self._stop.is_set():
                    return
                self._ingest(dataclasses.replace(s, received_at=self.clock()))
            if not self.loop:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
