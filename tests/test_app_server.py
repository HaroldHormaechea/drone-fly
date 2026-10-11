"""UC-63: the FastAPI backend driven head-less through starlette's TestClient (hermetic).

The app is built over a temp project root with an injected :class:`~app.runs.RunRegistry` whose
seams are all fakes (no micromamba, no GPU, no git, no window). Covers every route and the error
mapping RunError → 409, RecipeError → 400, FileNotFoundError → 404, and that audit payloads always
carry completion together with the honesty verdict and cheat checks.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest
from app import logs, status
from app import server as server_mod
from app.server import create_app
from fastapi.testclient import TestClient

from tests.app_support import append_log, fixture_text, make_registry, touch

TERMINAL = ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING", "PROC_OBS_TERMINAL"]


@pytest.fixture
def env(tmp_path: Path):
    reg, fakes = make_registry(tmp_path)
    viz = tmp_path / "viz"
    viz.mkdir()
    (viz / "viewer.html").write_text("<html>viewer</html>", encoding="utf-8")
    client = TestClient(create_app(str(tmp_path), registry=reg))
    return tmp_path, client, reg, fakes


def _config(client: TestClient, name: str, **payload):
    payload.setdefault("recipe", "train_aug")
    return client.post(f"/api/runs/{name}/config", json={"config": payload})


# --------------------------------------------------------------------------- shell + static


def test_index_and_static_mounts(env) -> None:
    _, client, _, _ = env
    index = client.get("/")
    assert index.status_code == 200 and "<html" in index.text.lower()
    assert client.get("/app/app.js").status_code == 200
    viewer = client.get("/viewer/viewer.html")
    assert viewer.status_code == 200 and "viewer" in viewer.text


def test_default_registry_is_built_when_none_is_injected(tmp_path: Path) -> None:
    app = create_app(str(tmp_path))
    assert app.state.project_root == os.path.abspath(str(tmp_path))
    assert app.state.registry.project_root == os.path.abspath(str(tmp_path))


# --------------------------------------------------------------------------- settings


def test_settings_defaults_save_and_unknown_keys_dropped(env) -> None:
    root, client, _, _ = env
    got = client.get("/api/settings").json()
    assert got["micromamba_executable"] == "micromamba"
    assert got["conda_env"] == "dronefly"
    assert got["audit_seed"] == 1234 and got["audit_eps"] == 512
    assert got["project_root"] == os.path.abspath(str(root))

    saved = client.post("/api/settings", json={"conda_env": "df2", "evil": 1}).json()
    assert saved["conda_env"] == "df2" and "evil" not in saved
    on_disk = json.loads((root / ".dev-app" / "settings.json").read_text(encoding="utf-8"))
    assert on_disk["conda_env"] == "df2" and "evil" not in on_disk
    assert client.get("/api/settings").json()["conda_env"] == "df2"


def test_corrupt_settings_file_falls_back_to_defaults(env) -> None:
    root, client, _, _ = env
    (root / ".dev-app").mkdir()
    (root / ".dev-app" / "settings.json").write_text("{not json", encoding="utf-8")
    assert client.get("/api/settings").json()["conda_env"] == "dronefly"


# --------------------------------------------------------------------------- preflight + schema


def test_preflight_ok_interpreter_warns_about_root_and_bases(env) -> None:
    _, client, _, _ = env
    body = client.get("/api/preflight").json()
    checks = {c["id"]: c for c in body["checks"]}
    assert checks["micromamba"]["ok"] is True
    assert checks["env"]["ok"] is True
    assert checks["project_root"]["ok"] is False  # a temp root is not /workspace/drone-fly
    assert "/workspace/drone-fly" in checks["project_root"]["message"]
    assert {k for k in checks if k.startswith("base:")} == {
        "base:exp11_count-curriculum",
        "base:exp12_gap-steering",
        "base:exp13_count-gap",
        "base:train_shaping",
    }
    assert checks["project_root"]["message"] in body["warnings"]


def test_preflight_reports_missing_micromamba(env) -> None:
    _, client, _, fakes = env
    fakes["which"].result = None
    body = client.get("/api/preflight").json()
    checks = {c["id"]: c for c in body["checks"]}
    assert checks["micromamba"]["ok"] is False
    assert "micromamba" in checks["micromamba"]["message"]
    assert "env" not in checks
    assert checks["micromamba"]["message"] in body["warnings"]


def test_preflight_reports_failed_env_probe(env) -> None:
    _, client, _, fakes = env
    fakes["probe"].result = False
    checks = {c["id"]: c for c in client.get("/api/preflight").json()["checks"]}
    assert checks["env"]["ok"] is False and "dronefly" in checks["env"]["message"]


def test_schema_route(env) -> None:
    _, client, _, _ = env
    schema = client.get("/api/schema").json()
    assert [r["name"] for r in schema["recipes"]][-1] == "train_aug"
    assert len(schema["flags"]) == 5 and len(schema["knobs"]) == 11


# --------------------------------------------------------------------------- runs + config


def test_config_save_get_and_list(env) -> None:
    _, client, _, _ = env
    res = _config(client, "run-a", flags=["PROC_GAP"], knobs={"N": 64})
    assert res.status_code == 200
    assert res.json()["config"]["knobs"] == {"N": 64}
    got = client.get("/api/runs/run-a/config").json()
    assert got == {"name": "run-a", "config": res.json()["config"]}
    runs = client.get("/api/runs").json()["runs"]
    assert [(r["name"], r["state"], r["kind"]) for r in runs] == [("run-a", "ready", "app")]
    assert client.get("/api/runs/run-a").json()["flags"] == ["PROC_GAP"]


def test_config_accepts_unwrapped_payload(env) -> None:
    _, client, _, _ = env
    res = client.post("/api/runs/run-a/config", json={"recipe": "train_aug", "knobs": {"N": 8}})
    assert res.status_code == 200 and res.json()["config"]["knobs"] == {"N": 8}


@pytest.mark.parametrize(
    "payload",
    [
        {"recipe": "nope"},
        {"recipe": "train_aug", "knobs": {"N": "x"}},
        {"recipe": "exp11_count-curriculum", "knobs": {"N": 4}},
    ],
)
def test_invalid_config_is_400(env, payload) -> None:
    _, client, _, _ = env
    res = client.post("/api/runs/run-a/config", json={"config": payload})
    assert res.status_code == 400
    assert res.json()["detail"]


def test_invalid_run_name_is_400(env) -> None:
    _, client, _, _ = env
    assert client.get("/api/runs/a..b").status_code == 400
    assert _config(client, "a..b").status_code == 400
    assert client.post("/api/runs/a..b/launch").status_code == 400


def test_missing_config_and_unknown_run_are_404(env) -> None:
    _, client, _, _ = env
    assert client.get("/api/runs/ghost/config").status_code == 404
    assert client.get("/api/runs/ghost").status_code == 404
    assert client.delete("/api/runs/ghost").status_code == 404
    assert client.get("/api/runs/ghost/status").status_code == 404


def test_config_change_refused_while_running(env) -> None:
    _, client, _, _ = env
    _config(client, "run-a")
    client.post("/api/runs/run-a/launch")
    assert _config(client, "run-a", knobs={"N": 4}).status_code == 409


def test_config_on_committed_dir_is_400(env) -> None:
    root, client, _, _ = env
    (root / "training" / "07-proc").mkdir(parents=True)
    assert _config(client, "07-proc").status_code == 400
    assert not (root / "training" / "07-proc" / "run.json").exists()


# --------------------------------------------------------------------------- lifecycle


def test_launch_stop_and_conflicts(env) -> None:
    _, client, _, fakes = env
    _config(client, "run-a", recipe="exp12_gap-steering", flags=TERMINAL)
    res = client.post("/api/runs/run-a/launch")
    assert res.status_code == 200 and res.json()["state"] == "running"
    assert fakes["spawn"].last["argv"][-3].endswith("exp12_gap-steering.py")
    assert client.post("/api/runs/run-a/launch").status_code == 409

    stopped = client.post("/api/runs/run-a/stop")
    assert stopped.status_code == 200 and stopped.json()["state"] == "stopping"
    fakes["spawn"].popens[-1].exit(-15)
    assert client.get("/api/runs/run-a").json()["state"] == "stopped"
    assert client.post("/api/runs/run-a/stop").status_code == 409


def test_launch_overwrite_query(env) -> None:
    root, client, _, fakes = env
    _config(client, "run-a")
    touch(root / "training" / "run-a" / "model.pt")
    res = client.post("/api/runs/run-a/launch")
    assert res.status_code == 409 and "overwrite" in res.json()["detail"]
    assert fakes["spawn"].calls == []
    assert client.post("/api/runs/run-a/launch?overwrite=true").status_code == 200


def test_launch_without_micromamba_is_409(env) -> None:
    _, client, _, fakes = env
    fakes["which"].result = None
    _config(client, "run-a")
    res = client.post("/api/runs/run-a/launch")
    assert res.status_code == 409 and "micromamba" in res.json()["detail"]


def test_delete_routes(env) -> None:
    root, client, _, _ = env
    _config(client, "run-a")
    res = client.delete("/api/runs/run-a")
    assert res.status_code == 200 and res.json()["deleted"] is True
    assert not (root / "training" / "run-a").exists()

    touch(root / "training" / "01-hover" / "model_readout.pt")
    assert client.delete("/api/runs/01-hover").status_code == 409
    assert (root / "training" / "01-hover" / "model_readout.pt").is_file()

    _config(client, "live")
    client.post("/api/runs/live/launch")
    assert client.delete("/api/runs/live").status_code == 409


# --------------------------------------------------------------------------- status + logs


def test_status_poll_returns_points_run_and_gpu(env) -> None:
    root, client, _, _ = env
    _config(client, "run-a")
    path = logs.log_path_for(str(root), "run-a")
    append_log(path, f"{status.LAUNCH_MARKER} t\n" + fixture_text("aug_log_sample.log"))
    body = client.get("/api/runs/run-a/status").json()
    assert [p["it"] for p in body["points"]] == [20, 40, 60, 80]
    assert body["latest"]["fps"] == 41500.0
    assert body["reset"] is True
    assert body["run"]["name"] == "run-a"
    assert body["gpu"] == {"util_pct": 50.0, "mem_used_mb": 1000.0, "mem_total_mb": 8000.0}
    again = client.get(f"/api/runs/run-a/status?since={body['next']}").json()
    assert again["points"] == [] and again["latest"]["it"] == 80


def test_status_with_gpu_unavailable(env) -> None:
    _, client, _, fakes = env
    fakes["gpu_probe"].result = None
    _config(client, "run-a")
    assert client.get("/api/runs/run-a/status").json()["gpu"] is None


def test_logs_tail_route(env) -> None:
    root, client, _, _ = env
    append_log(logs.log_path_for(str(root), "run-a"), "line one\nline two\n")
    body = client.get("/api/runs/run-a/logs").json()
    assert body["lines"] == ["line one", "line two"]
    assert client.get(f"/api/runs/run-a/logs?since={body['next']}").json()["lines"] == []
    assert client.get("/api/runs/a..b/logs").status_code == 400
    assert client.get("/api/runs/ghost/logs").json() == {"lines": [], "next": 0}


def test_status_stream_emits_points_then_done(env, monkeypatch) -> None:
    root, client, _, _ = env
    monkeypatch.setattr(server_mod.time, "sleep", lambda *_: None)
    _config(client, "run-a")
    append_log(logs.log_path_for(str(root), "run-a"), fixture_text("aug_log_sample.log"))
    res = client.get("/api/runs/run-a/status/stream")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/event-stream")
    data = [json.loads(ln[6:]) for ln in res.text.splitlines() if ln.startswith('data: {"it"')]
    assert [p["it"] for p in data] == [20, 40, 60, 80]
    assert "event: done" in res.text


def test_status_stream_unknown_run_is_404(env) -> None:
    _, client, _, _ = env
    assert client.get("/api/runs/ghost/status/stream").status_code == 404


# --------------------------------------------------------------------------- audits


def _audited_run(root: Path, client: TestClient) -> None:
    _config(client, "run-a", flags=TERMINAL)
    touch(root / "training" / "run-a" / "model.pt")


def test_post_audit_and_list_bundle_completion_with_verdict(env) -> None:
    root, client, _, fakes = env
    _audited_run(root, client)
    fakes["spawn"].next_output = fixture_text("verify_barging.txt")
    res = client.post("/api/runs/run-a/audits", json={"kind": "verify", "seed": 5, "eps": 64})
    assert res.status_code == 200
    started = res.json()
    assert started["state"] == "running" and started["verdict"] == "unverified"
    assert fakes["spawn"].last["env"]["AUDIT_SEED"] == "5"
    assert fakes["spawn"].last["argv"][-1] == "64"
    fakes["spawn"].popens[-1].exit(0)

    body = client.get("/api/runs/run-a/audits").json()
    assert body["thresholds"]["HOVER_SPEED_MS"] == 0.5
    assert body["run"]["name"] == "run-a"
    (audit,) = body["audits"]
    assert audit["completion"]["pct"] == 86.0
    assert audit["verdict"] == "dishonest" and audit["reasons"]
    assert audit["verify"]["barging_pct"] == 71.0
    assert audit["verify"]["min_clearance_m"] == -0.21
    assert audit["verify"]["mean_tilt_deg"] == 24.0


def test_post_audit_defaults_to_verify(env) -> None:
    root, client, _, fakes = env
    _audited_run(root, client)
    assert client.post("/api/runs/run-a/audits", json={}).json()["kind"] == "verify"
    assert fakes["spawn"].last["argv"][-3].endswith("verify_proc.py")


def test_audit_error_mapping(env) -> None:
    root, client, _, _ = env
    touch(root / "training" / "11-model" / "model_readout.pt")
    # Committed model without flags → 400; unknown kind → 400.
    assert client.post("/api/runs/11-model/audits", json={"kind": "verify"}).status_code == 400
    _audited_run(root, client)
    assert client.post("/api/runs/run-a/audits", json={"kind": "lint"}).status_code == 400
    # Unknown run → 404; live run → 409.
    assert client.post("/api/runs/ghost/audits", json={}).status_code == 404
    assert client.get("/api/runs/ghost/audits").status_code == 404
    client.post("/api/runs/run-a/launch?overwrite=true")
    assert client.post("/api/runs/run-a/audits", json={}).status_code == 409


def test_committed_model_audit_with_flags(env) -> None:
    root, client, _, fakes = env
    touch(root / "training" / "11-model" / "model_readout.pt")
    res = client.post("/api/runs/11-model/audits", json={"kind": "verify", "flags": TERMINAL})
    assert res.status_code == 200
    assert fakes["spawn"].last["argv"][-2].endswith("model_readout.pt")


# --------------------------------------------------------------------------- record + recordings


def test_record_route(env) -> None:
    root, client, _, fakes = env
    _audited_run(root, client)
    res = client.post("/api/runs/run-a/record", json={"episodes": 3})
    assert res.status_code == 200 and res.json()["episodes"] == 3
    assert fakes["spawn"].last["argv"][-1] == "3"
    assert client.post("/api/runs/run-a/record").status_code == 409  # already recording


def test_record_defaults_to_ten_episodes(env) -> None:
    root, client, _, fakes = env
    _audited_run(root, client)
    assert client.post("/api/runs/run-a/record").json()["episodes"] == 10
    assert fakes["spawn"].last["argv"][-1] == "10"


def test_record_error_mapping(env) -> None:
    root, client, _, _ = env
    _audited_run(root, client)
    assert client.post("/api/runs/run-a/record", json={"episodes": 0}).status_code == 400
    client.post("/api/runs/run-a/launch?overwrite=true")
    assert client.post("/api/runs/run-a/record").status_code == 409
    touch(root / "training" / "05-x" / "model_readout.pt")
    assert client.post("/api/runs/05-x/record").status_code == 409


def test_recordings_list_and_server_side_gunzip(env) -> None:
    root, client, _, _ = env
    rec = root / "training" / "run-a" / "recordings"
    rec.mkdir(parents=True)
    doc = {"schema": "drone-fly-record", "frames": [1, 2, 3]}
    with gzip.open(rec / "episode_0.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(doc, fh)
    (rec / "episode_1.json").write_text(json.dumps(doc), encoding="utf-8")
    (rec / "episode_2.json").write_text("{broken", encoding="utf-8")
    listed = client.get("/api/runs/run-a/recordings").json()["recordings"]
    assert [(r["episode"], r["gzip"]) for r in listed] == [(0, True), (1, False), (2, False)]
    assert client.get("/api/runs/run-a/recordings/0").json() == doc
    assert client.get("/api/runs/run-a/recordings/1").json() == doc
    assert client.get("/api/runs/run-a/recordings/2").status_code == 422
    assert client.get("/api/runs/run-a/recordings/9").status_code == 404
    assert client.get("/api/runs/a..b/recordings").status_code == 400
    assert client.get("/api/runs/ghost/recordings").json()["recordings"] == []


def test_recordings_of_a_committed_model_are_listed(env) -> None:
    root, client, _, _ = env
    rec = root / "training" / "11-model" / "recordings"
    rec.mkdir(parents=True)
    with gzip.open(rec / "episode_3.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"ok": True}, fh)
    assert client.get("/api/runs/11-model/recordings/3").json() == {"ok": True}
