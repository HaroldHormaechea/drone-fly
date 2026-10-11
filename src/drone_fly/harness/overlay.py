"""Runner-side live-overlay publisher (UC-65): non-blocking localhost UDP, drop on any error.

:class:`OverlayPublisher` sends one ``tick`` datagram per runner tick and, every
``overlay.brain_every_n`` ticks, a ``brain`` datagram carrying the full propagated K1 state
quantised to ``uint8`` with the recorder's scheme — or, opt-in (``overlay.brain_scale:
frame_peak``), with a per-frame scale from the frame's own peak (:func:`quantize_frame_peak`)
(wire format: :mod:`drone_fly.overlay.protocol`).
The socket is non-blocking with a ~1 MB send buffer; **any** send or serialisation error
(including ``BlockingIOError`` when the buffer is full) drops that datagram, increments
:attr:`OverlayPublisher.drops` and is logged at most once per second — it never raises into the
control loop. Nobody listening is fine: UDP to a closed localhost port just disappears.

The runner measures what publishing costs into ``latency_s["overlay"]`` (serialise + the
GPU->CPU copy of the brain state on brain ticks + send); ``inference`` and ``total`` keep their
meaning. A tick datagram carries the *previous* tick's overlay cost (this tick's is only known
after it was sent).

Toolkit-free and torch-free at import: the brain tensor is converted by duck typing, and the
recorder's quantiser is imported when the publisher is constructed (the contract has loaded torch
by then). :func:`tick_payload`, :func:`build_meta` and :func:`stop_payload` are pure, so the
overlay's replay mode builds exactly the same payloads from a recorded run folder.
"""

from __future__ import annotations

import logging
import math
import secrets
import socket
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from drone_fly.overlay.protocol import (
    DEFAULT_OFFSET,
    DEFAULT_SCALE,
    MAX_DATAGRAM,
    encode_brain,
    encode_json,
)

log = logging.getLogger(__name__)

#: Requested send buffer (bytes); a full buffer drops datagrams instead of blocking.
SEND_BUFFER_BYTES = 1 << 20
META_PERIOD_S = 1.0
STATUS_PERIOD_S = 0.25  # status datagrams are rate-limited to 4 Hz
LOG_PERIOD_S = 1.0
STOP_REPEATS = 2
BRAIN_SCALES = ("recorder", "frame_peak")
#: ``frame_peak`` floor for max |a| (an all-zero frame still gets a finite, non-zero scale).
MIN_FRAME_PEAK = 1e-6

RUN_RUNNING = "running"
RUN_WAITING_FOR_STATE = "waiting_for_state"
RUN_WAITING_FOR_FOCUS = "waiting_for_focus"
RUN_STOPPED = "stopped"


# ----------------------------------------------------------------------------- pure payloads


def recordable_panel_masks(data: Mapping[str, Any], ocfg: Any) -> list[dict[str, Any]]:
    """``camera.masks`` entries covering the overlay's panels (``overlay.recordable``).

    A recordable overlay is visible to the harness's own screen capture, so its panels must be
    masked out of gate detection and depth like the game HUD. The default layout's panel rects are
    fractions of the overlay monitor; they are mapped onto ``camera.resolution`` assuming the
    frame is that whole monitor (``camera.region: null``, as in the packaged presets). The
    ``mask_confined`` layout only draws inside existing masks, so it needs none.
    """
    from drone_fly.overlay.layout import DEFAULT_RECTS

    if ocfg.layout != "default":
        return []
    camera = data.get("camera") or {}
    res = camera.get("resolution")
    if not res:
        log.warning("overlay.recordable: camera.resolution is not set; cannot mask the panels")
        return []
    if camera.get("region"):
        log.warning(
            "overlay.recordable with camera.region set: panel masks assume the frame is the whole "
            "monitor and may be off"
        )
    w, h = float(res[0]), float(res[1])
    masks = []
    for name, (x0, y0, x1, y1) in DEFAULT_RECTS.items():
        if ocfg.panel_enabled(name):
            rect = [round(x0 * w), round(y0 * h), round((x1 - x0) * w), round((y1 - y0) * h)]
            masks.append({"name": f"overlay_{name}", "rect": rect})
    return masks


def sample_extras(sample: Any) -> dict[str, Any] | None:
    """Telemetry extras not in ``TickRecord.state``: battery, stick inputs, motor RPM."""
    if sample is None:
        return None

    def vec(v: Any) -> list[float] | None:
        return None if v is None else [float(x) for x in np.asarray(v).reshape(-1)]

    return {
        "battery": vec(getattr(sample, "battery", None)),
        "inputs": vec(getattr(sample, "inputs", None)),
        "rpm": vec(getattr(sample, "rpm", None)),
    }


def tick_payload(
    record: Mapping[str, Any],
    *,
    run_info: Mapping[str, Any] | None = None,
    run_state: str = RUN_RUNNING,
    missed_deadlines: int = 0,
    drops: int = 0,
    focus: bool | None = None,
    focus_window: str | None = None,
    sample: Mapping[str, Any] | None = None,
    overlay_prev_s: float | None = None,
) -> dict[str, Any]:
    """The ``tick`` datagram body: a ``TickRecord.as_dict()`` (or a ``ticks.jsonl`` line) + extras.

    The record's own ``state`` key is the flight state, so the run state travels as
    ``run_state``.
    """
    out = dict(record)
    latency = dict(out.get("latency_s") or {})
    if "overlay" not in latency and overlay_prev_s is not None:
        latency["overlay"] = float(overlay_prev_s)
    out["latency_s"] = latency
    out.update(
        run_state=run_state,
        run_info=dict(run_info or {}),
        missed_deadlines=int(missed_deadlines),
        drops=int(drops),
        focus=focus,
        focus_window=focus_window,
        sample=None if sample is None else dict(sample),
    )
    return out


def contract_constants(contract: Any) -> dict[str, Any]:
    """The contract facts the panels draw against (``{}`` when no contract is loaded)."""
    if contract is None:
        return {}
    out: dict[str, Any] = {}
    for key, attr in (
        ("max_body_rate", "MAX_BODY_RATE"),
        ("max_thrust", "MAX_THRUST"),
        ("hover_thr", "HOVER_THR"),
        ("ray_range", "RAY_RANGE"),
    ):
        try:
            out[key] = float(getattr(contract, attr))
        except (AttributeError, TypeError, ValueError):
            continue
    try:
        out["cone_xy"] = np.asarray(contract.CONE, np.float64)[:, :2].tolist()
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    return out


def build_meta(
    cfg: Mapping[str, Any],
    *,
    extra_dim: int | None,
    constants: Mapping[str, Any] | None = None,
    brain_info: Mapping[str, Any] | None = None,
    k1_path: str | None = None,
) -> dict[str, Any]:
    """The ``meta`` datagram body from a merged config mapping (``HarnessConfig.data`` or a run
    folder's ``config.yaml``). ``extra_layout`` is ``[[block, start, width], ...]`` truncated to
    ``extra_dim``; ``age_limits`` are the fail-safe limits the health panel colours against."""
    from drone_fly.harness.assembler import EXTRA_LAYOUT

    layout: list[list[Any]] = []
    start = 0
    for name, width in EXTRA_LAYOUT:
        if extra_dim is None or start >= extra_dim:
            break
        layout.append([name, start, min(width, extra_dim - start)])
        start += width
    runner = cfg.get("runner") or {}
    gates = cfg.get("gates") or {}
    camera = cfg.get("camera") or {}
    scaling = cfg.get("scaling") or {}
    age_limits = {
        "telemetry": runner.get("stale_after_s"),
        "free_space": runner.get("max_estimate_age_s"),
        "gate": gates.get("lost_timeout_s"),
    }
    return {
        "harness": cfg.get("harness"),
        "extra_dim": extra_dim,
        "extra_layout": layout,
        "rate_hz": runner.get("rate_hz"),
        "age_limits": {k: v for k, v in age_limits.items() if v is not None},
        "world_scale": scaling.get("world_scale"),
        "camera": {
            "monitor": camera.get("monitor"),
            "region": camera.get("region"),
            "resolution": camera.get("resolution"),
        },
        "brain": dict(brain_info or {}),
        "k1_path": k1_path,
        **dict(constants or {}),
    }


def stop_payload(result: Any) -> dict[str, Any]:
    """The ``stop`` datagram body from a ``RunResult`` (or a run folder's ``meta.json``)."""
    if isinstance(result, Mapping):
        d = result
        reason = d.get("reason", d.get("stop_reason"))
        error = d.get("error", d.get("stop_error"))
    else:
        d = {k: getattr(result, k, None) for k in ("ticks", "missed_deadlines", "failsafe")}
        d.update(
            failsafe_ok=getattr(result, "failsafe_ok", None),
            failsafe_output=getattr(result, "failsafe_output", None),
        )
        reason, error = getattr(result, "reason", None), getattr(result, "error", None)
    return {
        "reason": reason,
        "error": error,
        "ticks": d.get("ticks"),
        "missed_deadlines": d.get("missed_deadlines"),
        "failsafe": d.get("failsafe"),
        "failsafe_ok": d.get("failsafe_ok"),
        "failsafe_output": d.get("failsafe_output"),
    }


def _brain_to_numpy(brain: Any) -> np.ndarray:
    if hasattr(brain, "detach"):  # a torch tensor: this is the GPU -> CPU copy (a CUDA sync)
        return brain.detach().float().cpu().numpy().reshape(-1)
    return np.asarray(brain, dtype=np.float32).reshape(-1)


def quantize_frame_peak(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """``(codes, scale, offset)`` with a symmetric scale from the frame's own peak ``max |a|``.

    Real K1 activations sit far inside the recorder's fixed ``[-1, 1]`` range (a steady
    loopback run had 99.97 % of neuron-ticks on codes 127/128), so ``overlay.brain_scale:
    frame_peak`` spends all 256 codes on ``[-peak, peak]`` instead; the datagram carries the
    scale / offset, so the receiver dequantises to real units either way. Non-finite values
    count as 0, and the peak is floored at :data:`MIN_FRAME_PEAK` (no division by zero, no NaN).
    """
    a = np.nan_to_num(np.asarray(values, np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    peak = max(float(np.max(np.abs(a))) if a.size else 0.0, MIN_FRAME_PEAK)
    scale = 2.0 * peak / 255.0
    codes = np.clip(np.round((a + peak) / scale), 0, 255).astype(np.uint8)
    return codes, scale, -peak


# ----------------------------------------------------------------------------- publisher


class OverlayPublisher:
    """Publishes runner ticks to the overlay; never blocks, never raises into the loop."""

    def __init__(
        self,
        port: int,
        brain_every_n: int = 3,
        run_info: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        *,
        sock: Any = None,
        quantize: Callable[[np.ndarray], np.ndarray] | None = None,
        host: str = "127.0.0.1",
        clock: Callable[[], float] = time.monotonic,
        brain_scale: str = "recorder",
    ) -> None:
        if isinstance(brain_every_n, bool) or int(brain_every_n) < 1:
            raise ValueError(f"brain_every_n must be >= 1, got {brain_every_n!r}")
        if brain_scale not in BRAIN_SCALES:
            raise ValueError(f"brain_scale must be one of {BRAIN_SCALES}, got {brain_scale!r}")
        self.brain_scale = brain_scale
        self.address = (host, int(port))
        self.brain_every_n = int(brain_every_n)
        self.run_info = dict(run_info or {})
        self.meta = dict(meta or {})
        self.run_id = secrets.token_hex(8)
        self.clock = clock
        if quantize is None:
            from drone_fly.record.recorder import (
                ACTIVATION_OFFSET,
                ACTIVATION_SCALE,
                quantize_activation,
            )

            self._quantize = quantize_activation
            self.scale, self.offset = float(ACTIVATION_SCALE), float(ACTIVATION_OFFSET)
        else:
            self._quantize = quantize
            self.scale, self.offset = DEFAULT_SCALE, DEFAULT_OFFSET
        self.sock = sock if sock is not None else self._make_socket()
        self.drops = 0
        self.sent = 0
        self._last_meta = -math.inf
        self._last_status = -math.inf
        self._last_log = -math.inf
        self._closed = False

    @staticmethod
    def _make_socket() -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setblocking(False)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUFFER_BYTES)
        except OSError as exc:  # best effort
            log.warning("overlay publisher: could not set SO_SNDBUF=%d: %s", SEND_BUFFER_BYTES, exc)
        return s

    # -- internals --

    def _drop(self, what: str, exc: BaseException | None = None) -> None:
        self.drops += 1
        now = self.clock()
        if now - self._last_log >= LOG_PERIOD_S:
            self._last_log = now
            log.warning(
                "overlay publisher dropped a %s datagram (%s); %d dropped so far",
                what,
                "oversized" if exc is None else repr(exc),
                self.drops,
            )

    def _send(self, data: bytes, what: str) -> bool:
        if self._closed:
            return False
        if len(data) > MAX_DATAGRAM:
            self._drop(what)
            return False
        try:
            self.sock.sendto(data, self.address)
        except OSError as exc:  # includes BlockingIOError (send buffer full)
            self._drop(what, exc)
            return False
        self.sent += 1
        return True

    def _send_json(self, kind: str, payload: Callable[[], dict[str, Any]]) -> bool:
        try:
            data = encode_json(kind, self.run_id, payload())
        except Exception as exc:  # noqa: BLE001 - serialisation must never stop the run
            self._drop(kind, exc)
            return False
        return self._send(data, kind)

    def _maybe_meta(self) -> None:
        now = self.clock()
        if now - self._last_meta >= META_PERIOD_S:
            self._last_meta = now
            self.publish_meta()

    # -- public API --

    def wants_brain(self, tick: int) -> bool:
        """Whether tick ``tick`` carries a brain frame (the runner's single source of truth)."""
        return int(tick) % self.brain_every_n == 0

    def publish_meta(self) -> bool:
        return self._send_json("meta", lambda: {**self.meta, "run_info": self.run_info})

    def publish_tick(
        self,
        record: Any,
        *,
        brain: Any = None,
        sample: Any = None,
        missed_deadlines: int = 0,
        focus: bool | None = None,
        focus_window: str | None = None,
        overlay_prev_s: float | None = None,
    ) -> None:
        """Send the tick datagram (meta first when due), plus the brain datagram iff ``brain``."""
        try:
            self._maybe_meta()
            self._send_json(
                "tick",
                lambda: tick_payload(
                    record.as_dict(),
                    run_info=self.run_info,
                    missed_deadlines=missed_deadlines,
                    drops=self.drops,
                    focus=focus,
                    focus_window=focus_window,
                    sample=sample_extras(sample),
                    overlay_prev_s=overlay_prev_s,
                ),
            )
            if brain is not None:
                try:
                    values = _brain_to_numpy(brain)
                    if self.brain_scale == "frame_peak":
                        codes, scale, offset = quantize_frame_peak(values)
                    else:
                        codes, scale, offset = self._quantize(values), self.scale, self.offset
                    data = encode_brain(self.run_id, record.tick, codes, scale, offset)
                except Exception as exc:  # noqa: BLE001
                    self._drop("brain", exc)
                else:
                    self._send(data, "brain")
        except Exception as exc:  # noqa: BLE001 - belt and braces: never raise into the loop
            self._drop("tick", exc)

    def publish_status(self, run_state: str) -> None:
        """While the runner waits: ``run_state`` at most 4 times per second (plus meta)."""
        try:
            now = self.clock()
            if now - self._last_status < STATUS_PERIOD_S:
                return
            self._last_status = now
            self._maybe_meta()
            self._send_json(
                "status",
                lambda: {"run_state": run_state, "run_info": self.run_info, "drops": self.drops},
            )
        except Exception as exc:  # noqa: BLE001
            self._drop("status", exc)

    def publish_stop(self, result: Any) -> None:
        """The stop reason and fail-safe outcome, sent twice (UDP may lose one)."""
        try:
            for _ in range(STOP_REPEATS):
                self._send_json(
                    "stop",
                    lambda: {
                        **stop_payload(result),
                        "run_state": RUN_STOPPED,
                        "run_info": self.run_info,
                        "drops": self.drops,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            self._drop("stop", exc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.sock.close()
        except Exception:  # noqa: BLE001
            log.debug("closing the overlay socket failed", exc_info=True)
