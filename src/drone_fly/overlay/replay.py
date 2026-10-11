"""Replay a UC-62/UC-64 run folder through the overlay (``harness overlay --replay <run dir>``).

Reads the folder's ``meta.json``, ``config.yaml`` and ``ticks.jsonl`` and yields ``(delay,
Packet)`` pairs — a synthesized ``meta`` first, then one ``tick`` per recorded line (built with
the publisher's own :func:`~drone_fly.harness.overlay.tick_payload`, paced by the recorded
``wall_time`` deltas), then a ``stop`` from ``meta.json`` — exactly the packets a live run would
have sent, minus brain frames: run folders carry none, so the brain panel shows "no brain data
(replay)". :meth:`ReplaySource.packets` never sleeps (tests iterate it directly);
:meth:`ReplaySource.play` sleeps the delays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.overlay import (
    RUN_RUNNING,
    RUN_STOPPED,
    build_meta,
    stop_payload,
    tick_payload,
)
from drone_fly.overlay.protocol import Packet, decode_datagram, encode_json
from drone_fly.overlay.state import OverlayState

log = logging.getLogger(__name__)

#: Gaps between recorded ticks are capped so a paused / stalled recording does not freeze replay.
MAX_DELAY_S = 1.0
STOP_DELAY_S = 0.1


def _packet(kind: str, run_id: str, payload: dict[str, Any]) -> Packet:
    """Round-trip through the wire encoding, so replay packets match live ones exactly."""
    packet = decode_datagram(encode_json(kind, run_id, payload))
    if packet is None:  # pragma: no cover - encode/decode are inverse
        raise HarnessConfigError(f"could not encode a replay {kind} packet")
    return packet


class ReplaySource:
    """A recorded run folder as an overlay packet stream."""

    def __init__(self, run_dir: str | Path, *, speed: float = 1.0) -> None:
        self.run_dir = Path(run_dir)
        if not speed > 0:
            raise HarnessConfigError(f"replay speed must be > 0, got {speed!r}")
        self.speed = float(speed)
        ticks_path = self.run_dir / "ticks.jsonl"
        if not ticks_path.is_file():
            raise HarnessConfigError(f"{self.run_dir} has no ticks.jsonl (is it a run folder?)")
        self.ticks: list[dict[str, Any]] = []
        for i, line in enumerate(ticks_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                self.ticks.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("replay: skipping unreadable ticks.jsonl line %d", i + 1)
        meta_path = self.run_dir / "meta.json"
        self.meta: dict[str, Any] = (
            json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        )
        cfg_path = self.run_dir / "config.yaml"
        self.config: dict[str, Any] = (
            (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
            if cfg_path.is_file()
            else {}
        )
        digest = hashlib.sha1(str(self.run_dir.resolve()).encode("utf-8")).hexdigest()
        self.run_id = digest[:16]
        sha = self.meta.get("checkpoint_sha256") or ""
        checkpoint = self.meta.get("checkpoint")
        self.run_info = {
            "harness": self.meta.get("harness", self.config.get("harness")),
            "checkpoint": None if checkpoint is None else Path(str(checkpoint)).name,
            "sha8": sha[:8],
            "extra_dim": self.meta.get("extra_dim"),
            "replay": self.run_dir.name,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def meta_payload(self) -> dict[str, Any]:
        meta = build_meta(
            self.config,
            extra_dim=self.meta.get("extra_dim"),
            k1_path=self.meta.get("k1_path"),
        )
        meta.update(replay=True, run_info=self.run_info)
        return meta

    def packets(self) -> Iterator[tuple[float, Packet]]:
        """``(delay_s, packet)`` pairs at recorded speed / :attr:`speed` (no sleeping)."""
        yield 0.0, _packet("meta", self.run_id, self.meta_payload())
        missed = 0
        prev: float | None = None
        for t in self.ticks:
            wall = t.get("wall_time")
            delay = 0.0
            if isinstance(wall, (int, float)) and prev is not None:
                delay = min(max(float(wall) - prev, 0.0), MAX_DELAY_S) / self.speed
            if isinstance(wall, (int, float)):
                prev = float(wall)
            missed += bool(t.get("missed_deadline"))
            payload = tick_payload(
                t, run_info=self.run_info, run_state=RUN_RUNNING, missed_deadlines=missed
            )
            yield delay, _packet("tick", self.run_id, payload)
        stop = {**stop_payload(self.meta), "run_state": RUN_STOPPED, "run_info": self.run_info}
        if stop["ticks"] is None:
            stop["ticks"] = len(self.ticks)
        yield STOP_DELAY_S / self.speed, _packet("stop", self.run_id, stop)

    def play(self, state: OverlayState, *, sleep: Callable[[float], Any] | None = None) -> None:
        """Feed every packet into ``state`` at recorded speed (stoppable via :meth:`stop`)."""
        for delay, packet in self.packets():
            if self._stop.is_set():
                return
            if delay > 0:
                if sleep is None:
                    if self._stop.wait(delay):
                        return
                else:
                    sleep(delay)
            state.ingest(packet)

    def start(self, state: OverlayState) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self.play, args=(state,), name="overlay-replay", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


def replay_sleep_free(run_dir: str | Path, state: OverlayState) -> None:
    """Feed a whole run folder into ``state`` without sleeping (tests / tooling)."""
    ReplaySource(run_dir).play(state, sleep=lambda _s: None)
