"""UC-61 — the FastAPI backend driven headlessly via starlette ``TestClient`` (``app.server``).

No window, no GPU, no display, no real subprocess (the injected ``RunRegistry`` uses fake spawn +
signaler). Pins the HTTP contract for the ACs the server owns:

* **AC1** — ``create_app`` builds a FastAPI app importable WITHOUT ``pywebview`` (the native window
  is a separate, owner-eyeball path); ``/`` serves the shell; ``viz/`` is mounted for the reused
  viewer; recording gzip is decompressed server-side (the route returns parsed JSON).
* **AC2** — the Settings pane round-trips; the static shell exposes the left-nav + hash routes.
* **AC3** — ``POST /api/slices`` serialises a ``PruneRunConfig`` and invokes ``prune --config``.
* **AC4/AC9** — ``POST /api/train-configs/{name}`` saves in place; a bad config → HTTP 400.
* **AC5/AC7** — launch / pause / resume / stop map to the registry; double-launch → HTTP 409.
* **AC8** — recording list + server-side-gunzipped fetch; a missing episode → HTTP 404.
"""

from __future__ import annotations

import gzip
import json
import os
import sys

import yaml
from app import configs_io
from app.runs import RunRegistry
from app.server import create_app
from fastapi.testclient import TestClient


class _FakePopen:
    _pid = 5000

    def __init__(self, argv, cwd):
        self.argv = argv
        self.cwd = cwd
        _FakePopen._pid += 1
        self.pid = _FakePopen._pid
        self._rc = None

    def poll(self):
        return self._rc

    def finish(self, rc=0):
        self._rc = rc

    def terminate(self):
        self._rc = -15


def _client(root):
    spawned: list[_FakePopen] = []

    def fake_spawn(argv, cwd):
        p = _FakePopen(argv, cwd)
        spawned.append(p)
        return p

    def fake_signaler(popen, sig):
        popen.finish(0)  # process reacts to the signal by exiting (checkpoint flushed)

    reg = RunRegistry(str(root), spawn=fake_spawn, signaler=fake_signaler, cli="drone-fly")
    app = create_app(str(root), registry=reg)
    return TestClient(app), reg, spawned


# --- AC1: framework / shell / no pywebview ------------------------------------------------


def test_server_imports_without_pywebview():
    # Importing the backend must not pull in pywebview (it needs a display; owner-eyeball only).
    assert "webview" not in sys.modules


def test_index_serves_the_shell(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_static_app_assets_served(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.get("/app/app.js")
    assert r.status_code == 200


def test_viewer_mounted_when_viz_present(tmp_path):
    viz = tmp_path / "viz"
    viz.mkdir()
    (viz / "viewer.html").write_text("<html>viewer</html>", encoding="utf-8")
    client, _, _ = _client(tmp_path)
    r = client.get("/viewer/viewer.html")
    assert r.status_code == 200
    assert "viewer" in r.text


# --- AC2: settings pane + shell structure -------------------------------------------------


def test_settings_round_trip(tmp_path):
    client, _, _ = _client(tmp_path)
    got = client.get("/api/settings").json()
    assert got["host"] == "127.0.0.1"  # default surfaced
    saved = client.post("/api/settings", json={"default_connectome": "malecns"}).json()
    assert saved["default_connectome"] == "malecns"
    # Persisted: a fresh GET reflects it.
    assert client.get("/api/settings").json()["default_connectome"] == "malecns"


def test_shell_exposes_left_nav_and_hash_routes():
    """AC2 structural check of the front-end (headless): nav groups + hash routes present."""
    static_dir = os.path.join(__import__("app").__path__[0], "static")
    index = open(os.path.join(static_dir, "index.html"), encoding="utf-8").read()
    for label in ("Generate slices", "Train", "Settings"):
        assert label in index
    for route in ("#/slices/new", "#/train/new", "#/settings"):
        assert route in index
    app_js = open(os.path.join(static_dir, "app.js"), encoding="utf-8").read()
    # Hash-based routing (back/forward work) — the actual render is owner-eyeball.
    assert "hashchange" in app_js
    assert "#/train/" in app_js


def test_forms_have_explicit_save_no_autosave():
    """AC9 structural check: a Save button exists; edits are tracked 'dirty', not auto-saved."""
    app_dir = __import__("app").__path__[0]
    forms = open(os.path.join(app_dir, "static", "forms.js"), encoding="utf-8").read()
    app_js = open(os.path.join(app_dir, "static", "app.js"), encoding="utf-8").read()
    assert "markDirty" in forms  # edits set a dirty flag rather than persisting on blur/change
    assert "Save" in app_js or "save" in app_js  # an explicit save action exists


# --- AC3: slice creation ------------------------------------------------------------------


def test_create_slice_serialises_and_invokes_prune(tmp_path):
    client, _, spawned = _client(tmp_path)
    r = client.post("/api/slices", json={"config": {"out": "artifacts/pruned", "prune_k": 8}})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "launched"
    # A prune config YAML was written and prune --config invoked (AC3/AC10).
    written = tmp_path / body["config"]
    assert written.is_file()
    assert spawned[-1].argv[:3] == ["drone-fly", "prune", "--config"]


def test_create_slice_bad_config_returns_400(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/slices", json={"config": {"prune_k": 8}})  # missing required 'out'
    assert r.status_code == 400


# --- AC4/AC9: train config schema + save --------------------------------------------------


def test_train_schema_endpoint_lists_every_key(tmp_path):
    client, _, _ = _client(tmp_path)
    fields = client.get("/api/train-configs/schema").json()["fields"]
    names = [f["name"] for f in fields]
    assert "name" in names and "prune" in names and "prune_k" in names
    assert len(names) == 43  # every TrainRunConfig key


def test_slice_schema_endpoint_lists_four_fields(tmp_path):
    client, _, _ = _client(tmp_path)
    fields = client.get("/api/slice-schema").json()["fields"]
    assert [f["name"] for f in fields] == ["connectome", "out", "prune_k", "prune_rule"]


def test_save_train_config_in_place_then_load(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/train-configs/demo", json={"config": {"timesteps": 5000}})
    assert r.status_code == 200
    on_disk = yaml.safe_load(
        open(configs_io.train_config_path(str(tmp_path), "demo"), encoding="utf-8")
    )
    assert on_disk == {"name": "demo", "timesteps": 5000}  # omit-vs-default preserved
    loaded = client.get("/api/train-configs/demo").json()["config"]
    assert loaded == {"name": "demo", "timesteps": 5000}


def test_save_train_config_bad_returns_400(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/train-configs/demo", json={"config": {"timesteps": "nope"}})
    assert r.status_code == 400


def test_list_train_configs_endpoint(tmp_path):
    client, _, _ = _client(tmp_path)
    client.post("/api/train-configs/alpha", json={"config": {}})
    client.post("/api/train-configs/zeta", json={"config": {}})
    assert client.get("/api/train-configs").json()["names"] == ["alpha", "zeta"]


# --- AC5/AC7: run lifecycle over HTTP -----------------------------------------------------


def test_launch_pause_resume_stop_over_http(tmp_path):
    client, _, spawned = _client(tmp_path)
    client.post("/api/train-configs/demo", json={"config": {}})

    # launch (AC5)
    r = client.post("/api/runs/demo/launch")
    assert r.status_code == 200
    assert r.json()["state"] == "running"
    assert spawned[-1].argv == [
        "drone-fly",
        "train",
        "--config",
        configs_io.train_config_path(str(tmp_path), "demo"),
        "--no-tui",
    ]

    # double-launch → 409 (AC5)
    assert client.post("/api/runs/demo/launch").status_code == 409

    # pause (AC7): first SIGTERM → the fake process exits → paused
    assert client.post("/api/runs/demo/pause").json()["state"] == "paused"

    # a checkpoint now exists (write it) → resume relaunches (AC7)
    ck = tmp_path / "training" / "demo" / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "ppo_racer_100_steps.zip").write_bytes(b"zip")
    r = client.post("/api/runs/demo/resume")
    assert r.status_code == 200
    assert r.json()["state"] == "running"

    # stop (AC5)
    assert client.post("/api/runs/demo/stop").json()["state"] == "stopped"


def test_get_runs_and_status_endpoints(tmp_path):
    client, _, _ = _client(tmp_path)
    client.post("/api/train-configs/demo", json={"config": {}})
    assert any(r["name"] == "demo" for r in client.get("/api/runs").json()["runs"])
    # status with no stream yet → source "none"
    snap = client.get("/api/runs/demo/status").json()
    assert snap["source"] == "none"
    # write a status line and re-poll
    sp = tmp_path / "training" / "demo" / "status.jsonl"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"timesteps": 42}) + "\n", encoding="utf-8")
    snap2 = client.get("/api/runs/demo/status").json()
    assert snap2["source"] == "jsonl"
    assert snap2["latest"]["timesteps"] == 42


# --- AC8: recordings over HTTP ------------------------------------------------------------


def test_recording_list_and_gzip_fetch_over_http(tmp_path):
    client, _, _ = _client(tmp_path)
    d = tmp_path / "training" / "demo" / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    doc = {"schema": "drone-fly-recording", "frames": [1, 2, 3]}
    with gzip.open(d / "episode_1.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)

    listed = client.get("/api/runs/demo/recordings").json()["recordings"]
    assert listed[0]["episode"] == 1 and listed[0]["gzip"] is True

    # Server-side gunzip → the route returns the PARSED JSON object (AC8).
    r = client.get("/api/runs/demo/recordings/1")
    assert r.status_code == 200
    assert r.json() == doc


def test_missing_recording_returns_404(tmp_path):
    client, _, _ = _client(tmp_path)
    (tmp_path / "training" / "demo" / "recordings").mkdir(parents=True, exist_ok=True)
    assert client.get("/api/runs/demo/recordings/99").status_code == 404


# --- AC11: docs describe how to launch + use the app --------------------------------------


def test_docs_describe_launch_and_usage():
    """AC11: README/docs describe how to launch and use the app (+ the lossless-pause behaviour)."""
    root = os.path.dirname(__import__("app").__path__[0])
    readme = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    assert "Desktop app (UC-61)" in readme
    assert "python -m app" in readme  # how to launch the native window
    assert "--extra app" in readme  # how to install the runtime deps
    # The behaviour-change every CLI run now has (first SIGINT/SIGTERM → lossless pause).
    assert "lossless" in readme.lower()
    # The maintainer module doc exists too.
    assert os.path.isfile(os.path.join(__import__("app").__path__[0], "README.md"))
