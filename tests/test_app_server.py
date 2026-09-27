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
from app import runs as runs_mod
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

    def fake_spawn(argv, cwd, *, log_path=None):
        p = _FakePopen(argv, cwd)
        p.log_path = log_path
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
    """AC2 structural check of the front-end (headless): nav labels + hash routes present.

    Item 1/3 moved the nav tree out of ``index.html`` — it is now rendered by ``app.js`` so the
    Slices/Train lists and their expanded state stay live. So the nav labels + the new per-run leaf
    routes are asserted against ``app.js``; ``index.html`` now only has to reference the shared
    assets (the fly favicon/brand mark + ``modal.js``, loaded before ``app.js``).
    """
    static_dir = os.path.join(__import__("app").__path__[0], "static")
    index = open(os.path.join(static_dir, "index.html"), encoding="utf-8").read()
    # index.html references the shared assets the nav/app depend on (item 5 icon + item 6 modal).
    assert "fly.svg" in index
    assert "modal.js" in index

    app_js = open(os.path.join(static_dir, "app.js"), encoding="utf-8").read()
    # The three nav groups (item 3 renamed "Generate slices" → "Slices") live in app.js now.
    for label in ("Slices", "Train", "Settings"):
        assert label in app_js
    # The base routes (unchanged) + the new per-run leaf routes (item 1) + slice-detail (item 3).
    for route in ("#/slices/new", "#/train/new", "#/settings", "#/slices/"):
        assert route in app_js
    for leaf in ('"/status"', '"/recordings"', '"/config"'):
        assert leaf in app_js, leaf
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


# --- item 3: saved-slice enumeration + per-name load/regenerate over HTTP -------------------


def test_list_slice_configs_endpoint(tmp_path):
    client, _, _ = _client(tmp_path)
    client.post("/api/slice-configs/alpha", json={"config": {"out": "artifacts/a"}})
    client.post("/api/slice-configs/zeta", json={"config": {"out": "artifacts/z"}})
    assert client.get("/api/slice-configs").json()["names"] == ["alpha", "zeta"]


def test_get_and_save_slice_config_round_trip_and_regenerate(tmp_path):
    client, _, spawned = _client(tmp_path)
    # POST /api/slice-configs/{name} saves the prune config AND regenerates the slice (item 3).
    r = client.post(
        "/api/slice-configs/myslice", json={"config": {"out": "artifacts/x", "prune_k": 9}}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "launched"
    assert spawned[-1].argv[:3] == ["drone-fly", "prune", "--config"]  # regenerate ran prune
    assert (tmp_path / "configs" / "prune" / "myslice.yaml").is_file()
    # GET reflects the saved config verbatim (only present keys).
    got = client.get("/api/slice-configs/myslice").json()
    assert got["name"] == "myslice"
    assert got["config"] == {"out": "artifacts/x", "prune_k": 9}


def test_save_slice_config_bad_returns_400(tmp_path):
    client, _, _ = _client(tmp_path)
    r = client.post("/api/slice-configs/bad", json={"config": {"prune_k": 5}})  # missing 'out'
    assert r.status_code == 400


# --- item 4: interpreter Setting + no-interpreter → HTTP 409 --------------------------------


def test_settings_includes_train_executable(tmp_path):
    client, _, _ = _client(tmp_path)
    got = client.get("/api/settings").json()
    assert "train_executable" in got  # item 4: interpreter override surfaced in Settings
    assert got["train_executable"] is None  # default unset → auto-detect
    saved = client.post("/api/settings", json={"train_executable": "/opt/venv"}).json()
    assert saved["train_executable"] == "/opt/venv"
    assert client.get("/api/settings").json()["train_executable"] == "/opt/venv"


def test_launch_with_no_interpreter_surfaces_as_409(tmp_path, monkeypatch):
    """No cli= seam + no pybullet-capable env → RunError → HTTP 409 (item 4, hermetic)."""
    sysbin = tmp_path / "_sysbin"
    sysbin.mkdir(parents=True, exist_ok=True)
    (sysbin / ("python.exe" if os.name == "nt" else "python3")).write_text("", encoding="utf-8")
    monkeypatch.setattr(
        runs_mod.sys,
        "executable",
        str(sysbin / ("python.exe" if os.name == "nt" else "python3")),
        raising=False,
    )

    spawned: list[_FakePopen] = []

    def fake_spawn(argv, cwd, *, log_path=None):
        p = _FakePopen(argv, cwd)
        p.log_path = log_path
        spawned.append(p)
        return p

    # No .venv* on disk, sysbin has no drone-fly script, probe always False → nothing resolves.
    reg = RunRegistry(
        str(tmp_path),
        spawn=fake_spawn,
        signaler=lambda p, s: p.finish(0),
        probe=lambda python_exe: False,
    )
    client = TestClient(create_app(str(tmp_path), registry=reg))
    client.post("/api/train-configs/demo", json={"config": {}})
    r = client.post("/api/runs/demo/launch")
    assert r.status_code == 409
    assert spawned == []  # resolution failed before any spawn


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


# --- item 5: live process-log tail endpoint (GET /api/runs/{name}/logs?since=) --------------


def _write_log(tmp_path, name, text):
    p = tmp_path / "training" / name / "logs" / "app.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_run_logs_endpoint_tails_from_since_cursor(tmp_path):
    client, _, _ = _client(tmp_path)
    log = _write_log(tmp_path, "demo", "first line\nsecond line\n")
    size = log.stat().st_size

    body = client.get("/api/runs/demo/logs").json()
    assert body["lines"] == ["first line", "second line"]
    assert body["next"] == size  # byte offset to poll from next time

    # Nothing new since the cursor → empty, cursor unchanged.
    assert client.get(f"/api/runs/demo/logs?since={size}").json()["lines"] == []

    # Append and re-poll from the cursor → only the new line comes back.
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("third line\n")
    r = client.get(f"/api/runs/demo/logs?since={size}").json()
    assert r["lines"] == ["third line"]
    assert r["next"] == log.stat().st_size


def test_run_logs_endpoint_applies_deny_filter(tmp_path):
    client, _, _ = _client(tmp_path)
    _write_log(tmp_path, "demo", "pybullet build time: May  1 2024\nSaving checkpoint to disk\n")
    body = client.get("/api/runs/demo/logs").json()
    # Boot noise dropped; the real checkpoint line kept (endpoint routes through app.logs.tail).
    assert body["lines"] == ["Saving checkpoint to disk"]


def test_run_logs_endpoint_absent_file_is_empty(tmp_path):
    client, _, _ = _client(tmp_path)
    assert client.get("/api/runs/demo/logs").json() == {"lines": [], "next": 0}


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


# --- item 2: embedded-viewer static-grep guards (no JS harness → owner-eyeball + grep) -------
#
# The embedded viewer (brain TL / actions BL / map full-height R, fully-bare chrome incl.
# #meta-bar) and the standalone viewer rendering identically are OWNER-EYEBALL. These cheap
# static greps catch the regressions that would silently break the standalone/embed split:
# the embed behaviour must stay gated behind the `?embed=1` param / `.embed` class.


def _viz_file(name: str) -> str:
    root = os.path.dirname(__import__("app").__path__[0])
    return open(os.path.join(root, "viz", name), encoding="utf-8").read()


def test_viewer_js_gates_embed_class_behind_the_param():
    js = _viz_file("viewer.js")
    # The ONE embed-aware edit: read ?embed, and ONLY then tag <body> with the embed class.
    assert 'get("embed")' in js
    assert 'classList.add("embed")' in js


def test_viewer_css_embed_rules_are_embed_scoped():
    css = _viz_file("viewer.css")
    # Standalone viewer matches no `.embed` selector → renders identically. The chrome-hide
    # (incl. the #meta-bar via `.embed header`) and 2-col layout are all `.embed`-scoped.
    assert ".embed header" in css
    assert ".embed #app" in css
    # Every rule that mentions the embed layout keeps the `.embed` prefix (no bare selector leaked
    # into the standalone cascade). Scan the UC-61 embed block for un-scoped structural selectors.
    embed_block = css[css.index("EMBEDDED MODE") :]
    for line in embed_block.splitlines():
        stripped = line.strip()
        # Selector lines end in "{"; skip comments, at-rules, and property/closing lines.
        if stripped.endswith("{") and not stripped.startswith(("/*", "*", "@")):
            assert ".embed" in stripped, f"un-scoped embed selector: {stripped!r}"


def test_viewer_embed_iframe_requests_embed_param():
    app_dir = __import__("app").__path__[0]
    js = open(os.path.join(app_dir, "static", "viewer-embed.js"), encoding="utf-8").read()
    assert "?embed=1" in js  # the app's iframe opts into embed mode; standalone URL is untouched
