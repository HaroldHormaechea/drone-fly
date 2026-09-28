"""FastAPI backend for the drone-fly desktop app (UC-61).

:func:`create_app` builds the FastAPI application: JSON API routers for settings, slice/train
config editing, run lifecycle, live status, and recording playback, plus static mounts for the app
shell (``app/static``) and the reused viewer (``viz/``). It is driveable head-less via starlette's
``TestClient`` (what CI exercises) — no window, no GPU, no display. The native window is opened
separately by :mod:`app.__main__`.

Every route reduces to a filesystem read or a ``drone-fly`` subcommand invocation via the injected
:class:`~app.runs.RunRegistry` (AC10). All mutating config writes go through
:mod:`app.configs_io`, which validates against the real config dataclasses (a bad config becomes
HTTP 400, never a stack trace).
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import configs_io, logs, recordings, status
from app.runs import RunError, RunRegistry
from drone_fly.config import ConfigError

# Repo layout: this file is <root>/app/server.py, so the project root is two levels up.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.dirname(_APP_DIR)
_STATIC_DIR = os.path.join(_APP_DIR, "static")

_DEFAULT_SETTINGS: dict[str, Any] = {
    "project_root": None,  # filled with the resolved root at load time
    "default_connectome": None,
    "host": "127.0.0.1",
    "port": 0,  # 0 → ephemeral port chosen at bind time
    # UC-61 item 4: the interpreter used to launch training. Empty/None → auto-detect a
    # pybullet-capable venv; set it to a venv dir / its python / a drone-fly script to override.
    "train_executable": None,
}


def _settings_path(project_root: str) -> str:
    return os.path.join(project_root, ".dev-app", "settings.json")


def _load_settings(project_root: str) -> dict[str, Any]:
    path = _settings_path(project_root)
    data: dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    merged = {**_DEFAULT_SETTINGS, **data}
    if not merged.get("project_root"):
        merged["project_root"] = project_root
    return merged


def _save_settings(project_root: str, settings: dict[str, Any]) -> dict[str, Any]:
    merged = {**_load_settings(project_root), **settings}
    path = _settings_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return merged


def create_app(
    project_root: str | None = None,
    *,
    registry: RunRegistry | None = None,
) -> FastAPI:
    """Build the FastAPI app rooted at ``project_root`` (defaults to the repo root).

    ``registry`` may be injected (tests pass one with a fake process spawner); otherwise a real
    :class:`RunRegistry` is created for ``project_root``.
    """
    root = os.path.abspath(project_root or _DEFAULT_ROOT)

    def _read_train_executable() -> str | None:
        # Re-read fresh each launch so a Settings change takes effect without a restart (item 4).
        val = _load_settings(root).get("train_executable")
        return val or None

    reg = registry or RunRegistry(root, settings_reader=_read_train_executable)
    viz_dir = os.path.join(root, "viz")

    app = FastAPI(title="drone-fly desktop", version="0.1.0")

    # -- shell + static assets ------------------------------------------------------------
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    if os.path.isdir(_STATIC_DIR):
        app.mount("/app", StaticFiles(directory=_STATIC_DIR), name="app-static")
    if os.path.isdir(viz_dir):
        # The reused viewer (viz/viewer.{html,js,css}) served UNCHANGED for iframe embedding (AC8).
        app.mount("/viewer", StaticFiles(directory=viz_dir), name="viewer")

    # -- settings (AC2 Settings pane) -----------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _load_settings(root)

    @app.post("/api/settings")
    def post_settings(payload: dict[str, Any]) -> dict[str, Any]:
        return _save_settings(root, payload)

    # -- form schemas (AC3/AC4) -----------------------------------------------------------
    @app.get("/api/train-configs/schema")
    def train_schema() -> dict[str, Any]:
        return {"fields": configs_io.describe_train_fields()}

    @app.get("/api/slice-schema")
    def slice_schema() -> dict[str, Any]:
        return {"fields": configs_io.describe_prune_fields()}

    # -- connectome suggestions (item 4: the connectome combobox datalist) -----------------
    @app.get("/api/connectomes")
    def list_connectomes() -> dict[str, Any]:
        # Suggestion-only enumeration of pruned-slice paths for the connectome field's datalist.
        return {"connectomes": configs_io.list_connectomes(root)}

    # -- live-status field descriptors (status-view polish items 5/6) ----------------------
    @app.get("/api/status-fields")
    def status_fields() -> dict[str, Any]:
        # Backend-owned descriptor list (label / group / path|computed / format / help) that the
        # grouped status renderer consumes so JS and Python never drift on shape or formatting.
        return {"fields": configs_io.describe_status_fields()}

    # -- train configs (AC4/AC9) ----------------------------------------------------------
    @app.get("/api/train-configs")
    def list_train_configs() -> dict[str, Any]:
        return {"names": configs_io.list_train_config_names(root)}

    @app.get("/api/train-configs/{name}")
    def get_train_config(name: str) -> dict[str, Any]:
        return {"name": name, "config": configs_io.load_train_config(root, name)}

    @app.post("/api/train-configs/{name}")
    def save_train_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        mapping = payload.get("config", payload)
        try:
            path = configs_io.save_train_config(root, name, mapping)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": name, "path": os.path.relpath(path, root)}

    # -- slices (AC3) ---------------------------------------------------------------------
    @app.post("/api/slices")
    def create_slice(payload: dict[str, Any]) -> dict[str, Any]:
        mapping = payload.get("config", payload)
        slug = payload.get("slug") or _slug_from_out(mapping.get("out"))
        try:
            path = configs_io.prune_config_to_yaml(root, mapping, slug=slug)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        pid = reg.run_prune(path)
        return {"config": os.path.relpath(path, root), "pid": pid, "status": "launched"}

    # -- saved slice configs (item 3: Slices menu enumeration + per-name load/regenerate) --
    @app.get("/api/slice-configs")
    def list_slice_configs() -> dict[str, Any]:
        return {"names": configs_io.list_prune_config_names(root)}

    @app.get("/api/slice-configs/{name}")
    def get_slice_config(name: str) -> dict[str, Any]:
        return {"name": name, "config": configs_io.load_prune_config(root, name)}

    @app.post("/api/slice-configs/{name}")
    def save_slice_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        # Save the prune config under configs/prune/<name>.yaml, then regenerate the slice.
        mapping = payload.get("config", payload)
        try:
            path = configs_io.prune_config_to_yaml(root, mapping, slug=name)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        pid = reg.run_prune(path)
        return {
            "name": name,
            "config": os.path.relpath(path, root),
            "pid": pid,
            "status": "launched",
        }

    # -- runs (AC5/AC7) -------------------------------------------------------------------
    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": reg.list_runs()}

    @app.get("/api/runs/{name}")
    def get_run(name: str) -> dict[str, Any]:
        return reg.describe(name)

    @app.post("/api/runs/{name}/launch")
    def launch_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.launch(name))

    @app.post("/api/runs/{name}/resume")
    def resume_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.resume(name))

    @app.post("/api/runs/{name}/pause")
    def pause_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.pause(name))

    @app.post("/api/runs/{name}/stop")
    def stop_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.stop(name))

    @app.delete("/api/runs/{name}")
    def delete_run(name: str, delete_config: bool = False) -> dict[str, Any]:
        # Item 3: delete a run's generated outputs (and, when delete_config=true, its saved YAML).
        # A live run → RunError → 409 (mirrors the lifecycle guard); nothing to delete →
        # FileNotFoundError → 404 (mirrors the recordings route).
        try:
            return reg.delete(name, delete_config=delete_config)
        except RunError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- live status (AC6) ----------------------------------------------------------------
    @app.get("/api/runs/{name}/status")
    def run_status(name: str, since: int = 0) -> dict[str, Any]:
        return status.progress_snapshot(root, name, since=since)

    # -- live process logs (item 5) -------------------------------------------------------
    @app.get("/api/runs/{name}/logs")
    def run_logs(name: str, since: int = 0) -> dict[str, Any]:
        # Byte-offset tail of training/<name>/logs/app.log; `since` mirrors the status route's
        # cursor. Returns {"lines": [filtered], "next": <offset>}; the raw log is kept on disk.
        return logs.tail(logs.log_path_for(root, name), since=since)

    @app.get("/api/runs/{name}/status/stream")
    def run_status_stream(name: str) -> StreamingResponse:
        # Server-Sent Events: emit each new status line as it appears. Best-effort; the poll
        # endpoint above is the tested/authoritative source (AC6). The generator is bounded by the
        # client disconnecting (StreamingResponse stops iterating on disconnect).
        def _events():
            cursor = 0
            import time

            # A finite guard so a stale/abandoned stream cannot spin forever in a test/headless run.
            for _ in range(10_000):
                snap = status.progress_snapshot(root, name, since=cursor)
                for line in snap["lines"]:
                    yield f"data: {json.dumps(line)}\n\n"
                cursor = snap["next"]
                st = reg.describe(name)["state"]
                if st not in ("running", "pausing", "stopping") and not snap["lines"]:
                    yield f"event: done\ndata: {json.dumps({'state': st})}\n\n"
                    return
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    # -- recordings (AC8) -----------------------------------------------------------------
    @app.get("/api/runs/{name}/recordings")
    def list_run_recordings(name: str) -> dict[str, Any]:
        return {"name": name, "recordings": recordings.list_recordings(root, name)}

    @app.get("/api/runs/{name}/recordings/{episode}")
    def get_run_recording(name: str, episode: int) -> JSONResponse:
        try:
            doc = recordings.load_recording(root, name, episode)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return JSONResponse(doc)

    app.state.project_root = root
    app.state.registry = reg
    return app


def _guard(fn):
    """Run a registry lifecycle call, mapping :class:`RunError` to HTTP 409."""
    try:
        return fn()
    except RunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _slug_from_out(out: Any) -> str:
    """Derive a config slug from a prune ``out`` path (its leaf), defaulting to ``"slice"``."""
    if not out or not isinstance(out, str):
        return "slice"
    leaf = os.path.basename(out.rstrip("/\\")) or "slice"
    safe = "".join(c if (c.isalnum() or c in "._-") else "-" for c in leaf)
    return safe or "slice"


__all__ = ["create_app"]
