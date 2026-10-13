"""UC70 host memory of a student run: the replay buffer, its pinning, the course-geometry strip
and the memory preflight (``training.memcheck``).

CPU, hermetic, tiny (8 envs, a 200-sample buffer): ``strip_pool_geometry`` runs after ``tracks/``
is written and changes neither ``courses.json`` nor anything training, val or eval read (a run with
the strip trains bit-identically to one without; a fresh eval pool is never stripped); the buffer
defaults to 1e6 samples; the sidecar records ``replay_buffer``; the preflight warns on overcommit
from mocked OS numbers, stays quiet otherwise and never raises.
"""

from __future__ import annotations

import json
import logging

import pytest
import torch

from drone_fly import drones
from drone_fly.training import distill, memcheck
from drone_fly.training.config import DistillSection, load_raw
from tests.student_support import tiny_pool, tiny_student_config, train_inputs

GEOMETRY = ("prims", "cells", "harness", "heightfield")
GB = 1 << 30


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc70-mem"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def train(inputs, tmp_path, cfg=None, **kw):
    cfg = cfg or tiny_student_config(inputs.k07)
    args = dict(teacher=inputs.teacher, noise=inputs.noise, run_root=tmp_path / "out",
                max_iterations=1, k1_path=str(inputs.k1), skip_feasibility=True,
                process_lister=lambda: [])  # fmt: skip
    args.update(kw)
    return distill.train_student(cfg, drones.load(cfg.drone), **args)


# ----------------------------------------------------------------------------- the strip


def test_strip_drops_only_the_geometry():
    pool = tiny_pool(2, "train")
    table = json.dumps(pool.table())
    keep = [(c.id, json.dumps(c.meta, sort_keys=True, default=str)) for c in pool.courses]
    assert all(c.prims is not None and c.cells is not None for c in pool.courses)
    freed = distill.strip_pool_geometry(pool)
    assert freed > 0
    for c in pool.courses:
        for name in GEOMETRY:
            assert getattr(c, name) is None, name
    assert json.dumps(pool.table()) == table  # courses.json does not read the geometry
    assert [(c.id, json.dumps(c.meta, sort_keys=True, default=str)) for c in pool.courses] == keep
    assert distill.strip_pool_geometry(pool) == 0  # idempotent


def test_a_new_pool_is_never_stripped_by_an_earlier_strip():
    """The val / eval pools are built after the train strip; courses are not shared objects."""
    train_pool = tiny_pool(2, "train")
    distill.strip_pool_geometry(train_pool)
    ev = tiny_pool(2, "eval")
    assert all(c.prims is not None and c.cells is not None for c in ev.courses)
    assert all(a is not b for a, b in zip(train_pool.courses, ev.courses, strict=True))


def test_tracks_are_written_before_the_strip(inputs, tmp_path, tiny_val, monkeypatch):
    real = distill.strip_pool_geometry
    seen = {}

    def spy(pool):
        runs = list((tmp_path / "out").rglob("tracks/courses.json"))
        seen["written"] = len(runs) == 1
        seen["table"] = pool.table()
        return real(pool)

    monkeypatch.setattr(distill, "strip_pool_geometry", spy)
    res = train(inputs, tmp_path)
    assert seen["written"]
    on_disk = json.loads((res.run.path / "tracks" / "courses.json").read_text("utf-8"))
    assert on_disk == json.loads(json.dumps(seen["table"]))


def test_the_strip_does_not_change_training_val_or_ppo(inputs, tmp_path, monkeypatch):
    """A run with the strip and one without it end bit-identical (DAgger, val selection, PPO)."""
    val_pools = []

    def fresh_val(*a, **k):
        p = tiny_pool(2)
        val_pools.append(p)
        return p

    monkeypatch.setattr(distill, "eval_pool", fresh_val)
    cfg = tiny_student_config(inputs.k07, finetune={"steps": 32.0, "critic_warmup_iters": 0},
                              distill={"val_every_iters": 1})  # fmt: skip
    stripped = train(inputs, tmp_path / "strip", cfg, max_iterations=2)
    assert all(c.prims is not None for p in val_pools for c in p.courses)  # val never stripped
    monkeypatch.setattr(distill, "strip_pool_geometry", lambda pool: 0)
    kept = train(inputs, tmp_path / "keep", cfg, max_iterations=2)
    for name in ("student.pt", "resume.pt"):
        a = torch.load(stripped.run.path / "checkpoints" / name, weights_only=False)
        b = torch.load(kept.run.path / "checkpoints" / name, weights_only=False)
        sa = a if name == "student.pt" else a["actor"]
        sb = b if name == "student.pt" else b["actor"]
        assert sa.keys() == sb.keys()
        for k in sa:
            assert torch.equal(sa[k], sb[k]), (name, k)
    va = (stripped.run.path / "eval" / "val_history.jsonl").read_text().splitlines()
    vb = (kept.run.path / "eval" / "val_history.jsonl").read_text().splitlines()
    strip_t = lambda rows: [{k: v for k, v in json.loads(r).items() if "time" not in k}  # noqa: E731
                            for r in rows]  # fmt: skip
    assert strip_t(va) == strip_t(vb) and len(va) >= 3  # 2 DAgger + the PPO selection


def test_the_strip_logs_what_it_released(inputs, tmp_path, tiny_val, caplog):
    with caplog.at_level("INFO", logger=distill.log.name):
        train(inputs, tmp_path)
    assert "of host course geometry after packing" in caplog.text


# ----------------------------------------------------------------------------- buffer + sidecar


def test_buffer_default_is_one_million():
    assert DistillSection().buffer == 1_000_000
    raw = load_raw("student-liftoff")["distill"]
    assert raw["buffer"] == 1_000_000 and raw["pin_memory"] is None  # null = auto
    assert DistillSection().pin_memory is None


def test_buffer_bytes():
    assert distill.ReplayBuffer.nbytes(10, 61, 0) == 10 * (4 * 61 + 16)
    assert distill.ReplayBuffer.nbytes(1, 61, 100) == 4 * 61 + 16 + 2 * 100
    buf = distill.ReplayBuffer(7, obs_dim=61, feat_dim=100)
    real = buf.obs.nbytes + buf.label.nbytes + buf.taps.nbytes
    assert real == distill.ReplayBuffer.nbytes(7, 61, 100)


@pytest.mark.parametrize("pin_memory", [None, True, False])
def test_sidecar_records_the_replay_buffer(inputs, tmp_path, tiny_val, pin_memory):
    from drone_fly.sim.action import read_sidecar
    from drone_fly.sim.student_obs import DIM

    cfg = tiny_student_config(inputs.k07, distill={"pin_memory": pin_memory})
    res = train(inputs, tmp_path, cfg)
    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    rb = side["replay_buffer"]
    feat = side["actor_arch"]["feat_dim"]
    assert rb["samples"] == cfg.distill.buffer == 200
    assert rb["bytes"] == distill.ReplayBuffer.nbytes(200, DIM, feat)
    assert rb["pinned"] is False  # CPU: nothing is pinned whatever pin_memory says
    mem = rb["memory_preflight"]
    assert mem["about_to_allocate"] == rb["bytes"] and isinstance(mem["overcommit"], bool)
    assert {"physical_total", "physical_available", "commit_total", "commit_limit",
            "process_private", "process_working_set"} <= set(mem)  # fmt: skip


def test_the_buffer_built_matches_the_recorded_size(inputs, tmp_path, tiny_val, monkeypatch):
    built = []
    real = distill.ReplayBuffer

    class Spy(real):
        def __init__(self, capacity, obs_dim, feat_dim, pin=False):
            built.append((capacity, obs_dim, feat_dim, pin))
            super().__init__(capacity, obs_dim, feat_dim, pin=pin)

    monkeypatch.setattr(distill, "ReplayBuffer", Spy)
    cfg = tiny_student_config(inputs.k07, distill={"pin_memory": True})
    train(inputs, tmp_path, cfg)
    (cap, obs_dim, feat_dim, pin) = built[0]
    assert cap == 200 and pin is False  # explicit true is still off on a CPU device


# ----------------------------------------------------------------------------- memcheck


def _status(total, commit, **kw):
    st = dict.fromkeys(("physical_total", "physical_available", "commit_total", "commit_limit",
                        "process_private", "process_working_set"))  # fmt: skip
    st.update(physical_total=total, commit_total=commit, **kw)
    return st


@pytest.mark.parametrize(
    ("total", "commit", "alloc", "warn"),
    [
        (16 * GB, 12 * GB, 5 * GB, True),  # 17 > 16: would page
        (16 * GB, 12 * GB, 3 * GB, False),
        (16 * GB, 12 * GB, 4 * GB, False),  # exactly physical: not over
        (16 * GB, 16 * GB, 1, True),
        (None, 12 * GB, 5 * GB, False),  # unknown numbers: quiet
        (16 * GB, None, 5 * GB, False),
        (0, 12 * GB, 5 * GB, False),
    ],
)
def test_preflight_warns_only_on_overcommit(monkeypatch, caplog, total, commit, alloc, warn):
    monkeypatch.setattr(memcheck, "memory_status", lambda: _status(total, commit))
    with caplog.at_level(logging.INFO, logger=memcheck.log.name):
        st = memcheck.preflight(alloc, "the test buffer")
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert st["overcommit"] is warn and st["about_to_allocate"] == alloc
    assert bool(warnings) is warn
    if warn:
        assert "Not stopping anything" in warnings[0].getMessage()
        assert "the test buffer" in warnings[0].getMessage()
    assert "memory preflight (the test buffer)" in caplog.text  # always logged


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_memory_status_never_raises(monkeypatch, platform):
    def boom():
        raise OSError("no counters")

    monkeypatch.setattr(memcheck.sys, "platform", platform)
    monkeypatch.setattr(memcheck, "_windows", boom)
    monkeypatch.setattr(memcheck, "_linux", boom)
    st = memcheck.memory_status()
    assert set(st) >= {"physical_total", "commit_total"} and all(v is None for v in st.values())
    out = memcheck.preflight(5 * GB)
    assert out["overcommit"] is False


def test_memory_status_on_an_unknown_platform(monkeypatch):
    monkeypatch.setattr(memcheck.sys, "platform", "sunos5")
    assert all(v is None for v in memcheck.memory_status().values())


def _real_status():
    import sys

    if not (sys.platform == "win32" or sys.platform.startswith("linux")):
        pytest.skip("no memory counters on this platform")
    return memcheck.memory_status()


def test_memory_status_reads_this_machine():
    """No mocks: the real system counters on this OS (Windows / Linux) are positive integers."""
    st = _real_status()
    for k in ("physical_total", "physical_available", "commit_total", "commit_limit"):
        assert isinstance(st[k], int) and st[k] > 0, k
    assert st["physical_available"] <= st["physical_total"]
    assert st["commit_total"] <= st["commit_limit"]


@pytest.mark.xfail(
    __import__("sys").platform == "win32",
    strict=True,
    reason="memcheck._windows calls GetProcessMemoryInfo with GetCurrentProcess's default c_int "
    "restype: the -1 pseudo-handle is truncated, the call fails (ERROR_INVALID_HANDLE) and the "
    "process counters stay 0",
)
def test_memory_status_reads_this_process():
    st = _real_status()
    for k in ("process_private", "process_working_set"):
        assert isinstance(st[k], int) and st[k] > 0, k
