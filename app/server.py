"""FastAPI backend for the drone-fly desktop app (UC-61, repointed at the reservoir in UC-63).

:func:`create_app` builds the FastAPI application: JSON API routes for settings, preflight, the
reservoir recipe/knob schema, per-run config, run lifecycle, live progress, logs, audits,
recording, and recording playback, plus static mounts for the app shell (``app/static``) and the
reused viewer (``viz/``). It is driveable head-less via starlette's ``TestClient`` (what CI
exercises) — no window, no GPU, no display. The native window is opened by :mod:`app.__main__`.

Every route reduces to a filesystem read or a ``gpu_prototype/`` script invocation via the injected
:class:`~app.runs.RunRegistry` (AC10). Errors map to HTTP: :class:`~app.runs.RunError` → 409,
:class:`~app.configs_io.RecipeError` → 400, :class:`FileNotFoundError` → 404.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import audits, configs_io, logs, recordings, status
from app.configs_io import RecipeError
from app.runs import RunError, RunRegistry

# Repo layout: this file is <root>/app/server.py, so the project root is two levels up.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.dirname(_APP_DIR)
_STATIC_DIR = os.path.join(_APP_DIR, "static")

_DEFAULT_SETTINGS: dict[str, Any] = {
    "project_root": None,  # filled with the resolved root at load time
    "host": "127.0.0.1",
    "port": 0,  # 0 → ephemeral port chosen at bind time
    "micromamba_executable": "micromamba",
    "conda_env": "dronefly",
    "audit_seed": 1234,
    "audit_eps": 512,
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
    merged = {**_DEFAULT_SETTINGS, **{k: v for k, v in data.items() if k in _DEFAULT_SETTINGS}}
    if not merged.get("project_root"):
        merged["project_root"] = project_root
    return merged


def _save_settings(project_root: str, settings: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in settings.items() if k in _DEFAULT_SETTINGS}
    merged = {**_load_settings(project_root), **clean}
    path = _settings_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return merged


def _guard(fn: Callable[[], Any]) -> Any:
    """Run ``fn``, mapping RunError → 409, RecipeError → 400, FileNotFoundError → 404."""
    try:
        return fn()
    except RunError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RecipeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def create_app(
    project_root: str | None = None,
    *,
    registry: RunRegistry | None = None,
) -> FastAPI:
    """Build the FastAPI app rooted at ``project_root`` (defaults to the repo root).

    ``registry`` may be injected (tests pass one with fake spawner / probes); otherwise a real
    :class:`RunRegistry` is created whose Settings reader re-reads ``.dev-app/settings.json`` on
    every launch, audit and record.
    """
    root = os.path.abspath(project_root or _DEFAULT_ROOT)
    reg = registry or RunRegistry(root, settings_reader=lambda: _load_settings(root))
    viz_dir = os.path.join(root, "viz")

    app = FastAPI(title="drone-fly desktop", version="0.2.0")

    # -- shell + static assets ------------------------------------------------------------
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    if os.path.isdir(_STATIC_DIR):
        app.mount("/app", StaticFiles(directory=_STATIC_DIR), name="app-static")
    if os.path.isdir(viz_dir):
        # The reused viewer (viz/viewer.{html,js,css}) served UNCHANGED for iframe embedding (AC8).
        app.mount("/viewer", StaticFiles(directory=viz_dir), name="viewer")

    # -- settings + preflight -------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return _load_settings(root)

    @app.post("/api/settings")
    def post_settings(payload: dict[str, Any]) -> dict[str, Any]:
        return _save_settings(root, payload)

    @app.get("/api/preflight")
    def preflight() -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        interp = reg.check_interpreter()
        checks.append(
            {
                "id": "micromamba",
                "ok": interp["micromamba"] is not None,
                "message": interp["micromamba"] or interp["message"],
            }
        )
        if interp["micromamba"] is not None:
            checks.append({"id": "env", "ok": interp["ok"], "message": interp["message"]})
        on_workspace = os.path.normpath(root).replace("\\", "/") == configs_io.WORKSPACE_ROOT
        checks.append(
            {
                "id": "project_root",
                "ok": on_workspace,
                "message": "project root is " + configs_io.WORKSPACE_ROOT
                if on_workspace
                else f"project root is {root}, but gpu_prototype hardcodes "
                f"{configs_io.WORKSPACE_ROOT} (connectome, named-recipe warm starts, record_aug "
                "output) — training, audits and recordings only work there",
            }
        )
        for name, spec in configs_io.RECIPES.items():
            base = spec["fixed"].get("warm_from")
            if not base:
                continue
            exists = os.path.isfile(base)
            checks.append(
                {
                    "id": f"base:{name}",
                    "ok": exists,
                    "message": f"{name} warm-start base {base}"
                    + (" found" if exists else " is missing"),
                }
            )
        return {
            "checks": checks,
            "warnings": [c["message"] for c in checks if not c["ok"]],
        }

    # -- schema ---------------------------------------------------------------------------
    @app.get("/api/schema")
    def schema() -> dict[str, Any]:
        return configs_io.describe_schema()

    # -- runs -----------------------------------------------------------------------------
    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": reg.list_runs()}

    @app.get("/api/runs/{name}")
    def get_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.describe(name))

    @app.get("/api/runs/{name}/config")
    def get_run_config(name: str) -> dict[str, Any]:
        return _guard(lambda: {"name": name, "config": configs_io.load_run_config(root, name)})

    @app.post("/api/runs/{name}/config")
    def save_run_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        def _save() -> dict[str, Any]:
            configs_io.validate_run_name(name)
            if reg.kind(name) == "app" and reg.describe(name)["running"]:
                raise RunError(f"cannot change {name!r} while it is running — stop it first")
            config = configs_io.save_run_config(root, name, payload.get("config", payload))
            return {"name": name, "config": config}

        return _guard(_save)

    @app.post("/api/runs/{name}/launch")
    def launch_run(name: str, overwrite: bool = False) -> dict[str, Any]:
        return _guard(lambda: reg.launch(name, overwrite=overwrite))

    @app.post("/api/runs/{name}/stop")
    def stop_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.stop(name))

    @app.delete("/api/runs/{name}")
    def delete_run(name: str) -> dict[str, Any]:
        return _guard(lambda: reg.delete(name))

    # -- live progress + logs (AC5) -------------------------------------------------------
    @app.get("/api/runs/{name}/status")
    def run_status(name: str, since: int = 0) -> dict[str, Any]:
        def _status() -> dict[str, Any]:
            run = reg.describe(name)
            snap = status.progress_snapshot(root, name, since=since)
            return {**snap, "run": run, "gpu": reg.gpu_status()}

        return _guard(_status)

    @app.get("/api/runs/{name}/logs")
    def run_logs(name: str, since: int = 0) -> dict[str, Any]:
        _guard(lambda: configs_io.validate_run_name(name))
        return logs.tail(logs.log_path_for(root, name), since=since)

    @app.get("/api/runs/{name}/status/stream")
    def run_status_stream(name: str) -> StreamingResponse:
        # Server-Sent Events: emit each new progress point as it appears. Best-effort; the poll
        # endpoint above is the tested/authoritative source. Bounded so an abandoned stream ends.
        _guard(lambda: reg.describe(name))

        def _events():
            cursor = 0
            for _ in range(10_000):
                snap = status.progress_snapshot(root, name, since=cursor)
                for point in snap["points"]:
                    yield f"data: {json.dumps(point)}\n\n"
                cursor = snap["next"]
                try:
                    state = reg.describe(name)["state"]
                except (RecipeError, FileNotFoundError):
                    return
                if state not in ("running", "stopping") and not snap["points"]:
                    yield f"event: done\ndata: {json.dumps({'state': state})}\n\n"
                    return
                time.sleep(1.0)

        return StreamingResponse(_events(), media_type="text/event-stream")

    # -- audits (AC6/AC7) -----------------------------------------------------------------
    @app.get("/api/runs/{name}/audits")
    def get_audits(name: str) -> dict[str, Any]:
        def _list() -> dict[str, Any]:
            run = reg.describe(name)
            return {
                "name": name,
                "run": run,
                "audits": audits.list_audits(reg, name),
                "thresholds": dict(audits.THRESHOLDS),
            }

        return _guard(_list)

    @app.post("/api/runs/{name}/audits")
    def post_audit(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return _guard(
            lambda: audits.start_audit(
                reg,
                name,
                payload.get("kind", "verify"),
                flags=payload.get("flags"),
                seed=payload.get("seed"),
                eps=payload.get("eps"),
            )
        )

    # -- recording (record_aug.py) --------------------------------------------------------
    @app.post("/api/runs/{name}/record")
    def record_run(name: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        episodes = (payload or {}).get("episodes", 10)
        return _guard(lambda: reg.record(name, episodes=episodes))

    # -- recordings playback (AC8) --------------------------------------------------------
    @app.get("/api/runs/{name}/recordings")
    def list_run_recordings(name: str) -> dict[str, Any]:
        _guard(lambda: configs_io.validate_run_name(name))
        return {"name": name, "recordings": recordings.list_recordings(root, name)}

    @app.get("/api/runs/{name}/recordings/{episode}")
    def get_run_recording(name: str, episode: int) -> JSONResponse:
        _guard(lambda: configs_io.validate_run_name(name))
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


__all__ = ["create_app"]
