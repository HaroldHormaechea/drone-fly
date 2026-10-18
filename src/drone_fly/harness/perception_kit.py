"""Perception-provider conformance kit (UC-86 AC7): ``check_provider(factory, role=...)``.

Drives any provider on synthetic frames and telemetry (CPU, no Liftoff, no network) and checks
the contract of :mod:`drone_fly.harness.perception`:

1. the declaration (:func:`~drone_fly.harness.perception.check_spec`), then ``start()``;
2. free-space role -- before any frame the stamp is ``-inf`` (or ``now`` for a constant provider)
   and never newer than the newest input; a finite-stamp estimate appears within ``n_frames``,
   stamped with (sync) / no later than (async) the newest frame; every estimate passes
   :func:`~drone_fly.harness.perception.validate_estimate`; a gap (same frame, no frame) holds
   the stamp; after ``reset()`` the stamp does not move forward until a new frame recovers it;
3. gate role -- the stamp is finite and ``<= now``, and equals ``frame.stamp`` when a frame was
   given to a frame-stamped provider;
4. async mode -- ``update()`` hammered with interleaved ``reset()`` while the worker runs;
5. ``stop()`` joins: no thread started by the provider is alive ``timeout_s`` after it.

Usage (pytest)::

    from drone_fly.harness.perception_kit import check_provider

    def test_my_provider_conforms():
        check_provider(lambda frames, states: MyProvider(), role="free_space_estimator")

``factory(frame_source, state_source)`` builds the provider; ``frame_source`` is ``None`` in sync
mode (the provider processes the frame handed to ``update()``). A failed check raises
:class:`~drone_fly.harness.errors.PerceptionContractError` naming the provider and the check;
success returns a report dict.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from drone_fly.harness.errors import PerceptionContractError
from drone_fly.harness.perception import check_spec, validate_estimate
from drone_fly.harness.types import FlightState, Frame

ROLES = ("gate_estimator", "free_space_estimator")


def synthetic_frames(
    n: int, shape: tuple[int, int, int] = (1080, 2560, 3), *, dt_s: float = 0.05,
    t0: float | None = None, seed: int = 0,
) -> list[Frame]:  # fmt: skip
    """``n`` textured BGR frames (a shifting random texture), ``dt_s`` apart on ``perf_counter``."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=shape, dtype=np.uint8)
    t0 = time.perf_counter() if t0 is None else t0
    return [
        Frame(image=np.roll(base, 3 * i, axis=1), stamp=t0 + i * dt_s, index=i) for i in range(n)
    ]


def synthetic_state(t: float, speed: float = 2.0) -> FlightState:
    """Level forward flight along +x at 1.5 m (training scale)."""
    return FlightState(
        pos=np.array([speed * t, 0.0, 1.5], np.float32),
        vel=np.array([speed, 0.0, 0.0], np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        omega=np.zeros(3, np.float32),
        t=float(t),
        received_at=float(t),
    )


class KitFrameSource:
    """Thread-safe latest-frame source the kit publishes to (async providers pull from it)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None

    def publish(self, frame: Frame | None) -> None:
        with self._lock:
            self._frame = frame

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame


class KitStateSource:
    """Thread-safe latest-state source."""

    discontinuity_count = 0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: FlightState | None = None

    def publish(self, state: FlightState) -> None:
        with self._lock:
            self._state = state

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self) -> FlightState | None:
        with self._lock:
            return self._state

    def age(self, now: float) -> float:
        s = self.latest()
        return math.inf if s is None else max(0.0, now - s.received_at)

    def drain_raw(self) -> list[Any]:
        return []


class _Kit:
    def __init__(self, name: str, provider: Any, role: str) -> None:
        self.name = name
        self.provider = provider
        self.role = role
        self.checks: list[str] = []

    def fail(self, check: str, problem: str) -> None:
        raise PerceptionContractError(self.name, None, f"conformance check {check!r}: {problem}")

    def ok(self, check: str) -> None:
        if check not in self.checks:
            self.checks.append(check)

    def validate(self, estimate: Any, check: str) -> None:
        try:
            validate_estimate(self.name, self.provider, estimate, self.role)
        except PerceptionContractError as exc:
            self.fail(check, str(exc))


def check_provider(
    factory: Callable[[Any, Any], Any],
    *,
    role: str,
    name: str = "provider",
    frames: Sequence[Frame] | None = None,
    n_frames: int = 12,
    frame_shape: tuple[int, int, int] = (1080, 2560, 3),
    dt_s: float = 0.05,
    async_mode: bool = False,
    timeout_s: float = 5.0,
    state_fn: Callable[[float], FlightState] | None = None,
    target_offset: Sequence[float] = (5.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Run the conformance checks on the provider ``factory`` builds; see the module docstring.

    ``frames`` replaces the synthetic frames (at least ``n_frames`` + a few are used);
    ``state_fn(t)`` the synthetic forward flight; ``target_offset`` the gate target relative to the
    drone (free-space role).
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    need = 2 * n_frames + 8
    if frames is None:
        frames = synthetic_frames(need, frame_shape, dt_s=dt_s)
    frames = list(frames)
    if len(frames) < need:
        raise ValueError(f"check_provider needs at least {need} frames, got {len(frames)}")
    state_fn = state_fn or synthetic_state
    t_base = frames[0].stamp
    frame_source = KitFrameSource() if async_mode else None
    state_source = KitStateSource()
    state_source.publish(state_fn(0.0))
    threads_before = set(threading.enumerate())
    provider = factory(frame_source, state_source)
    kit = _Kit(name, provider, role)
    try:
        check_spec(provider, role, name)
    except PerceptionContractError as exc:
        kit.fail("declaration", str(exc))
    kit.ok("declaration")
    report: dict[str, Any] = {"provider": name, "role": role, "async": async_mode}
    started = False
    try:
        provider.start()
        started = True
        kit.ok("start")
        if role == "free_space_estimator":
            report.update(
                _free_space(kit, frames, state_fn, state_source, frame_source, t_base,
                            n_frames, np.asarray(target_offset, np.float32), timeout_s)
            )  # fmt: skip
        else:
            report.update(_gate(kit, frames, state_fn, state_source, t_base, n_frames))
    finally:
        if started:
            try:
                provider.stop()
            except Exception as exc:  # noqa: BLE001 - a failing stop is a conformance failure
                kit.fail("stop", f"stop() raised {exc!r}")
    deadline = time.monotonic() + timeout_s
    while True:
        alive = [t for t in threading.enumerate() if t not in threads_before and t.is_alive()]
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    if alive:
        kit.fail("stop joins", f"threads still alive after stop(): {[t.name for t in alive]}")
    kit.ok("stop joins")
    report["checks"] = list(kit.checks)
    return report


def _state(state_fn: Callable[[float], FlightState], frame: Frame, t_base: float) -> FlightState:
    return state_fn(frame.stamp - t_base)


def _free_space(
    kit: _Kit, frames: list[Frame], state_fn: Callable[[float], FlightState],
    states: KitStateSource, frame_source: KitFrameSource | None, t_base: float, n_frames: int,
    offset: np.ndarray, timeout_s: float,
) -> dict[str, Any]:  # fmt: skip
    p = kit.provider
    async_mode = frame_source is not None

    def update(frame: Frame | None, now: float) -> Any:
        ref = frame if frame is not None else frames[0]
        state = _state(state_fn, ref, t_base)
        states.publish(state)
        if frame_source is not None and frame is not None:
            frame_source.publish(frame)
        est = p.update(frame, state, state.pos + offset, now)
        kit.validate(est, "estimate contract")
        return est

    # -- before any frame --
    now0 = frames[0].stamp - 0.001
    est = update(None, now0)
    constant = float(est.stamp) == now0
    if not constant and float(est.stamp) != -math.inf:
        kit.fail("stamp before first frame", f"expected -inf (or now), got {est.stamp}")
    kit.ok("stamp before first frame")

    def feed(start: int, count: int, check: str) -> tuple[Any, int]:
        """Feed frames until a finite-stamp estimate; return it and the next frame index."""
        last = None
        i = start
        for i in range(start, start + count):
            f = frames[i]
            now = f.stamp + 0.001
            last = update(f, now)
            if async_mode:
                end = time.monotonic() + timeout_s / max(count, 1)
                while not math.isfinite(float(last.stamp)) and time.monotonic() < end:
                    time.sleep(0.005)
                    last = update(f, now)
            s = float(last.stamp)
            if constant:
                if s != now:
                    kit.fail(check, f"a constant provider must stamp now ({now}), got {s}")
            elif math.isfinite(s):
                if s > f.stamp + 1e-9:
                    kit.fail(check, f"stamp {s} is newer than the newest frame {f.stamp}")
                if not async_mode and s != f.stamp:
                    kit.fail(check, f"sync stamp {s} != the newest frame stamp {f.stamp}")
                return last, i + 1
            elif s != -math.inf:
                kit.fail(check, f"stamp {s} is neither finite nor -inf")
        if constant:
            return last, i + 1
        kit.fail(check, f"no finite-stamp estimate within {count} frames")
        raise AssertionError  # unreachable

    est, nxt = feed(0, n_frames, "first estimate")
    kit.ok("first estimate")
    held = float(est.stamp)
    newest = frames[nxt - 1]
    # -- gap: the same frame again, then no frame --
    if not constant:
        for frame in (newest, None):
            g = update(frame, newest.stamp + 0.002)
            s = float(g.stamp)
            if async_mode:
                if not (held <= s <= newest.stamp + 1e-9):
                    kit.fail("gap holds stamp", f"stamp {s} outside [{held}, {newest.stamp}]")
                held = s
            elif s != held:
                kit.fail("gap holds stamp", f"stamp moved from {held} to {s} without a new frame")
        kit.ok("gap holds stamp")
    # -- reset --
    p.reset()
    r = update(None, newest.stamp + 0.003)
    if not constant and float(r.stamp) > held + 1e-9:
        kit.fail("reset", f"stamp moved forward after reset() ({held} -> {r.stamp})")
    est, nxt = feed(nxt, n_frames, "recovery after reset")
    kit.ok("reset")
    kit.ok("estimate contract")
    out: dict[str, Any] = {"constant": constant, "frames_used": nxt}
    if async_mode:
        # -- hammer: updates with interleaved resets while the worker runs --
        for j, i in enumerate(range(nxt, min(len(frames), nxt + 8))):
            f = frames[i]
            for _ in range(5):
                update(f, f.stamp + 0.001)
                if j % 2 == 0:
                    p.reset()
                time.sleep(0.001)
        kit.ok("async hammer")
    return out


def _gate(
    kit: _Kit, frames: list[Frame], state_fn: Callable[[float], FlightState],
    states: KitStateSource, t_base: float, n_frames: int,
) -> dict[str, Any]:  # fmt: skip
    p = kit.provider
    frame_stamped = False

    def update(frame: Frame | None, ref: Frame, now: float) -> Any:
        nonlocal frame_stamped
        state = _state(state_fn, ref, t_base)
        states.publish(state)
        est = p.update(frame, state, now)
        kit.validate(est, "estimate contract")
        s = float(est.stamp)
        if not math.isfinite(s) or s > now + 1e-9:
            kit.fail("gate stamp", f"stamp {s} must be finite and <= now ({now})")
        if frame is not None and s != now:
            if s != frame.stamp:
                kit.fail("gate stamp", f"stamp {s} is neither now ({now}) nor frame.stamp")
            frame_stamped = True
        return est

    for i in range(n_frames):
        f = frames[i]
        update(f, f, f.stamp + 0.001)
    kit.ok("gate stamp")
    update(None, frames[n_frames - 1], frames[n_frames - 1].stamp + 0.002)
    p.reset()
    for i in range(n_frames, n_frames + 3):
        f = frames[i]
        update(f, f, f.stamp + 0.001)
    kit.ok("reset")
    kit.ok("estimate contract")
    return {"frame_stamped": frame_stamped}
