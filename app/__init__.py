"""drone-fly desktop app (UC-61).

A lightweight, local, single-user desktop application that replaces the training TUI + the
standalone HTML recording viewer. It is a thin FastAPI/Uvicorn front-end over the existing
``drone-fly`` CLIs (it shells out to them) plus the existing dependency-free brain/inputs/course
recording viewer (``viz/viewer.{html,js,css}``, reused UNCHANGED).

The package is intentionally split so the backend is hermetically testable without a display:

* :mod:`app.server` — ``create_app()`` builds the FastAPI app (routers + static mounts). Driveable
  via starlette's ``TestClient`` with no window (this is what CI exercises).
* :mod:`app.__main__` — ``python -m app``: starts Uvicorn on a loopback ephemeral port in a thread
  and opens a ``pywebview`` native window at that URL. This is the only display-dependent path and
  is owner-eyeball only (never run in CI).
* :mod:`app.runs`, :mod:`app.configs_io`, :mod:`app.status`, :mod:`app.recordings` — the backend
  logic (run registry + subprocess lifecycle, YAML (de)serialisation with omit-vs-default, status
  tailing, server-side-gunzipped recording reads). All hermetic.

Everything reduces to invoking an existing ``drone-fly`` subcommand or reading a file the CLIs
already produce; the app itself lives outside ``src/drone_fly`` (AC10).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
