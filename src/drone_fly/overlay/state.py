"""Latest-wins overlay state (toolkit-free, thread-safe, clock-injected).

The receiver (or replay) thread calls :meth:`OverlayState.ingest` for every datagram; the window's
redraw timer calls :meth:`OverlayState.snapshot`. Only the **latest** tick, brain frame, meta and
status are kept — older packets are discarded, never queued — so a slow redraw can never fall
behind the stream.

Modes (:attr:`StateSnapshot.mode`):

``waiting``  nothing received yet: "waiting for harness";
``stale``    no datagram for more than ``stale_s``: "no data for N s";
``live``     datagrams are arriving (the message names the run state, e.g. waiting for focus);
``stopped``  a ``stop`` datagram arrived: the stop reason stays visible until a datagram with a
             new ``run_id`` (the next run) arrives.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from drone_fly.overlay.protocol import BrainFrame, Packet, decode_datagram

MODE_WAITING = "waiting"
MODE_STALE = "stale"
MODE_LIVE = "live"
MODE_STOPPED = "stopped"

#: Tick-rate EWMA weight of the newest inter-arrival interval.
RATE_ALPHA = 0.1
#: Inter-arrival gaps longer than this do not feed the tick-rate estimate.
RATE_MAX_GAP_S = 2.0


@dataclass(frozen=True)
class StateSnapshot:
    """An immutable view of the overlay state for one redraw."""

    mode: str
    message: str
    run_id: str | None
    tick: dict[str, Any] | None
    brain: BrainFrame | None
    meta: dict[str, Any] | None
    status: dict[str, Any] | None
    last_stop: dict[str, Any] | None
    tick_rate_hz: float | None
    since_last_s: float | None
    #: Increments with every accepted brain frame (renderers recompute only when it changes).
    brain_seq: int
    packets: int
    garbled: int


class OverlayState:
    """Holds the latest tick / brain / meta / status of the current run."""

    def __init__(self, stale_s: float = 1.0, clock: Callable[[], float] = time.monotonic) -> None:
        if not stale_s > 0:
            raise ValueError(f"stale_s must be > 0, got {stale_s!r}")
        self.stale_s = float(stale_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._tick: dict[str, Any] | None = None
        self._brain: BrainFrame | None = None
        self._meta: dict[str, Any] | None = None
        self._status: dict[str, Any] | None = None
        self._status_at = -math.inf
        self._tick_at = -math.inf
        self._last_stop: dict[str, Any] | None = None
        self._stopped = False
        self._last_packet: float | None = None
        self._rate: float | None = None
        self._brain_seq = 0
        self._packets = 0
        self._garbled = 0

    # -- producer side --

    def ingest_bytes(self, data: bytes) -> Packet | None:
        """Decode and ingest one datagram (garbled input is counted and ignored)."""
        packet = decode_datagram(data)
        self.ingest(packet)
        return packet

    def _new_run(self, run_id: str) -> None:
        self._run_id = run_id
        self._tick = None
        self._brain = None
        self._meta = None
        self._status = None
        self._status_at = self._tick_at = -math.inf
        self._last_stop = None
        self._stopped = False
        self._rate = None

    def ingest(self, packet: Packet | None) -> None:
        with self._lock:
            if packet is None:
                self._garbled += 1
                return
            now = self.clock()
            self._packets += 1
            self._last_packet = now
            if packet.run_id != self._run_id:
                self._new_run(packet.run_id)
            kind = packet.kind
            if kind == "tick":
                tick = packet.data.get("tick")
                old = None if self._tick is None else self._tick.get("tick")
                if isinstance(tick, int) and isinstance(old, int) and tick < old:
                    return  # reordered: an older tick never replaces a newer one
                if self._tick_at > -math.inf:
                    dt = now - self._tick_at
                    if 0 < dt <= RATE_MAX_GAP_S:
                        inst = 1.0 / dt
                        self._rate = (
                            inst
                            if self._rate is None
                            else (1 - RATE_ALPHA) * self._rate + RATE_ALPHA * inst
                        )
                self._tick = packet.data
                self._tick_at = now
            elif kind == "brain" and packet.brain is not None:
                if self._brain is not None and packet.brain.tick < self._brain.tick:
                    return
                self._brain = packet.brain
                self._brain_seq += 1
            elif kind == "meta":
                self._meta = packet.data
            elif kind == "status":
                self._status = packet.data
                self._status_at = now
            elif kind == "stop":
                self._last_stop = packet.data
                self._stopped = True

    # -- consumer side --

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            now = self.clock()
            since = None if self._last_packet is None else now - self._last_packet
            if self._stopped and self._last_stop is not None:
                mode = MODE_STOPPED
                reason = self._last_stop.get("reason") or "unknown"
                message = f"stopped: {reason} (waiting for the next run)"
            elif since is None:
                mode, message = MODE_WAITING, "waiting for harness"
            elif since > self.stale_s:
                mode, message = MODE_STALE, f"no data for {since:.0f} s"
            else:
                mode = MODE_LIVE
                if self._status is not None and self._status_at > self._tick_at:
                    message = str(self._status.get("run_state", "")).replace("_", " ")
                else:
                    message = "live"
            return StateSnapshot(
                mode=mode,
                message=message,
                run_id=self._run_id,
                tick=self._tick,
                brain=self._brain,
                meta=self._meta,
                status=self._status,
                last_stop=self._last_stop,
                tick_rate_hz=self._rate,
                since_last_s=since,
                brain_seq=self._brain_seq,
                packets=self._packets,
                garbled=self._garbled,
            )
