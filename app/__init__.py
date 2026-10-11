"""drone-fly desktop app (UC-61, repointed at the reservoir pipeline in UC-63).

A lightweight, local, single-user desktop application over the frozen-K1 reservoir pipeline in
``gpu_prototype/``: launch a reservoir run, monitor its ``[aug]`` progress live, run the
honesty-gated audits (``verify_proc.py`` / ``diagnose_proc.py``), record episodes
(``record_aug.py``), and replay recordings in the reused dependency-free brain/inputs/course viewer
(``viz/viewer.{html,js,css}``, UNCHANGED). It is a thin FastAPI/Uvicorn front-end that shells out
to the existing scripts (``micromamba run -n <env> python <script> ...``).

The package is intentionally split so the backend is hermetically testable without a display:

* :mod:`app.server` — ``create_app()`` builds the FastAPI app (routes + static mounts). Driveable
  via starlette's ``TestClient`` with no window (this is what CI exercises).
* :mod:`app.__main__` — ``python -m app``: starts Uvicorn on a loopback ephemeral port in a thread
  and opens a ``pywebview`` native window at that URL. This is the only display-dependent path and
  is owner-eyeball only (never run in CI).
* :mod:`app.configs_io` (recipe/knob schema + ``run.json``), :mod:`app.runs` (run registry +
  subprocess lifecycle), :mod:`app.status` (``[aug]`` line parser), :mod:`app.audits` (audit
  runner, output parsers, honesty verdict), :mod:`app.logs`, :mod:`app.recordings` — the backend
  logic. All hermetic: none of them imports torch, pybullet, or SB3.
* :mod:`app.reservoir_launch` — the generic ``train_aug`` launcher (pure argument forwarding).

Everything reduces to invoking an existing ``gpu_prototype/`` script or reading a file those scripts
already produce; the app itself lives outside ``src/drone_fly`` (AC10).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
