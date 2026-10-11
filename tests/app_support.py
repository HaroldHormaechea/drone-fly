"""Hermetic fakes for the UC-63 desktop-app tests (no micromamba, no GPU, no git, no torch).

:func:`make_registry` builds a :class:`app.runs.RunRegistry` over a temp project root with every OS
seam injected: the spawner records ``(argv, cwd, env, log_path)`` and returns a :class:`FakePopen`
whose exit code the test controls (and can write scripted stdout into the run's log, as the real
child would); ``which`` / the env probe / the git-tracked probe / the GPU probe / the clock are all
fakes with call counters.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app import configs_io
from app.runs import RunRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "app_audit"
MICROMAMBA = os.path.abspath(os.path.join(os.sep, "opt", "bin", "micromamba"))

_pids = itertools.count(4000)


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def append_log(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


class FakePopen:
    """Stands in for ``subprocess.Popen``: ``poll()`` returns :attr:`code` (``None`` = alive)."""

    def __init__(self) -> None:
        self.pid = next(_pids)
        self.code: int | None = None

    def poll(self) -> int | None:
        return self.code

    def exit(self, code: int) -> None:
        self.code = code


class FakeSpawn:
    """Records every spawn; optionally writes scripted output into the child's log."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.popens: list[FakePopen] = []
        #: Text the next spawned "child" writes to its log (consumed per spawn).
        self.next_output: str | None = None

    def __call__(
        self,
        argv: list[str],
        cwd: str,
        *,
        env: dict[str, str] | None = None,
        log_path: str | None = None,
    ) -> FakePopen:
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(env or {}), "log": log_path})
        if self.next_output is not None and log_path is not None:
            append_log(log_path, self.next_output)
            self.next_output = None
        popen = FakePopen()
        self.popens.append(popen)
        return popen

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


class Counter:
    """A callable returning ``result`` (or ``result(*args)`` when callable) and logging calls."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, *args: Any) -> Any:
        self.calls.append(args)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result(*args) if callable(self.result) else self.result


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


INHERITED_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/home/owner",
    "PROC_TUNNELS": "1",
    "PROC_GAP": "0",
    "PROC_NO_OBS": "",
    "AUDIT_SEED": "99",
}


def make_registry(root: str | Path, **overrides: Any) -> tuple[RunRegistry, dict[str, Any]]:
    """Return ``(registry, fakes)``; ``fakes`` holds every injected seam for assertions."""
    root = str(root)
    os.makedirs(os.path.join(root, "training"), exist_ok=True)
    os.makedirs(os.path.join(root, "gpu_prototype"), exist_ok=True)
    settings: dict[str, Any] = overrides.pop("settings", {})
    fakes: dict[str, Any] = {
        "spawn": FakeSpawn(),
        "signaler": Counter(None),
        "probe": Counter(True),
        "which": Counter(lambda exe: MICROMAMBA if exe == "micromamba" else None),
        "tracked_probe": Counter(False),
        "gpu_probe": Counter({"util_pct": 50.0, "mem_used_mb": 1000.0, "mem_total_mb": 8000.0}),
        "clock": FakeClock(),
        "settings": settings,
        "environ": dict(INHERITED_ENV),
    }
    fakes.update(overrides)
    settings_reader: Callable[[], dict[str, Any]] = lambda: fakes["settings"]  # noqa: E731
    registry = RunRegistry(
        root,
        spawn=fakes["spawn"],
        signaler=fakes["signaler"],
        probe=fakes["probe"],
        settings_reader=settings_reader,
        tracked_probe=fakes["tracked_probe"],
        gpu_probe=fakes["gpu_probe"],
        clock=fakes["clock"],
        which=fakes["which"],
        environ=fakes["environ"],
    )
    return registry, fakes


def save_run(root: str | Path, name: str, **payload: Any) -> dict[str, Any]:
    payload.setdefault("recipe", "train_aug")
    return configs_io.save_run_config(str(root), name, payload)


def touch(path: str | Path, data: bytes = b"ckpt") -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)
