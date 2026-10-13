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


# ----------------------------------------------------------------------------- replay_pin


class FakeDevice:
    def __init__(self, type_):
        self.type = type_


ONE_GIB = 1 << 30


@pytest.mark.parametrize(
    ("buf_bytes", "pin_memory", "want"),
    [
        (ONE_GIB - 1, None, True),  # auto: pin only below 1 GiB
        (ONE_GIB, None, False),
        (ONE_GIB + 1, None, False),
        (ONE_GIB - 1, True, True),  # an explicit value wins over the size
        (ONE_GIB, True, True),
        (ONE_GIB + 1, True, True),
        (ONE_GIB - 1, False, False),
        (ONE_GIB, False, False),
        (ONE_GIB + 1, False, False),
    ],
)
def test_replay_pin_on_cuda(buf_bytes, pin_memory, want):
    assert distill.AUTO_PIN_MAX_BYTES == ONE_GIB
    for dev in (FakeDevice("cuda"), torch.device("cuda"), torch.device("cuda", 0), "cuda"):
        assert distill.replay_pin(buf_bytes, pin_memory, dev) is want, dev


@pytest.mark.parametrize("pin_memory", [None, True, False])
@pytest.mark.parametrize("device", [FakeDevice("cpu"), torch.device("cpu"), "cpu", "mps"])
@pytest.mark.parametrize("buf_bytes", [0, ONE_GIB - 1, ONE_GIB + 1])
def test_replay_pin_is_always_false_off_cuda(buf_bytes, device, pin_memory):
    assert distill.replay_pin(buf_bytes, pin_memory, device) is False


@pytest.mark.parametrize("dev", ["cuda:0", "cuda:1"])
def test_replay_pin_with_an_indexed_device_string(dev):
    assert distill.replay_pin(1, True, dev) is True
    assert distill.replay_pin(ONE_GIB - 1, None, dev) is True
    assert distill.replay_pin(ONE_GIB, None, dev) is False
    assert distill.replay_pin(1, False, dev) is False
    assert distill.replay_pin(1, True, "cpu:0") is False


def test_the_run_uses_replay_pin(inputs, tmp_path, tiny_val, monkeypatch):
    from drone_fly.sim.action import read_sidecar

    seen = []
    real = distill.replay_pin

    def spy(buf_bytes, pin_memory, device):
        out = real(buf_bytes, pin_memory, device)
        seen.append((buf_bytes, pin_memory, str(device), out))
        return out

    monkeypatch.setattr(distill, "replay_pin", spy)
    cfg = tiny_student_config(inputs.k07, distill={"pin_memory": True})
    res = train(inputs, tmp_path, cfg)
    rb = read_sidecar(res.run.path / "checkpoints" / "student.pt")["replay_buffer"]
    assert seen == [(rb["bytes"], True, "cpu", False)] and rb["pinned"] is False


# ----------------------------------------------------------------------------- memcheck


def _status(total=16 * GB, avail=8 * GB, private=1 * GB, commit=40 * GB):
    """Mocked OS numbers; commit is deliberately far above physical (Windows' usual state)."""
    st = dict.fromkeys(("physical_total", "physical_available", "commit_total", "commit_limit",
                        "process_private", "process_working_set"))  # fmt: skip
    st.update(physical_total=total, physical_available=avail, process_private=private,
              commit_total=commit)  # fmt: skip
    return st


def _preflight(monkeypatch, caplog, alloc, **numbers):
    monkeypatch.setattr(memcheck, "memory_status", lambda: _status(**numbers))
    with caplog.at_level(logging.INFO, logger=memcheck.log.name):
        st = memcheck.preflight(alloc, "the test buffer")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    return st, warnings


@pytest.mark.parametrize(
    ("alloc", "numbers", "over_process", "over_available"),
    [
        # physical 16, available 8, private 1 (GB) unless overridden; commit 40 never matters
        (2 * GB, {}, False, False),  # quiet, although commit 40 + 2 > 16
        (9 * GB, {"avail": 10 * GB}, False, False),  # 1 + 9 = 10 <= 16, 9 <= 10
        (9 * GB, {}, False, True),  # 9 > available 8
        (8 * GB, {}, False, False),  # == available: not over
        (8 * GB + 1, {}, False, True),
        (6 * GB, {"private": 11 * GB}, True, False),  # 11 + 6 = 17 > 16, 6 <= 8
        (5 * GB, {"private": 11 * GB}, False, False),  # 11 + 5 == 16: not over
        (5 * GB + 1, {"private": 11 * GB}, True, False),
        (10 * GB, {"private": 7 * GB}, True, True),  # both
        (9 * GB, {"private": None}, False, True),  # unknown private: only the available test
        (20 * GB, {"private": None, "avail": 30 * GB}, False, False),
        (17 * GB, {"avail": None, "private": 0}, True, False),  # unknown available
        (2 * GB, {"avail": None}, False, False),
        (5 * GB, {"total": None, "private": 15 * GB}, False, False),  # unknown physical
        (5 * GB, {"total": 0, "private": 15 * GB}, False, False),
        (5 * GB, {"total": None, "avail": None, "private": None, "commit": None}, False, False),
    ],
)
def test_preflight_overcommit_rule(monkeypatch, caplog, alloc, numbers, over_process,
                                   over_available):  # fmt: skip
    st, warnings = _preflight(monkeypatch, caplog, alloc, **numbers)
    warn = over_process or over_available
    assert st["over_process"] is over_process and st["over_available"] is over_available
    assert st["overcommit"] is warn and st["about_to_allocate"] == alloc
    assert len(warnings) == int(warn)
    assert "memory preflight (the test buffer)" in caplog.text  # the info line, always
    if warn:
        msg = warnings[0]
        assert "the test buffer" in msg and "Not stopping anything" in msg
        # both comparisons are named, with their numbers
        assert "this process private" in msg and "vs physical" in msg and "vs available" in msg


def test_preflight_warning_numbers(monkeypatch, caplog):
    st, warnings = _preflight(monkeypatch, caplog, 6 * GB, private=11 * GB, avail=5 * GB)
    assert st["over_process"] and st["over_available"]
    assert warnings == [
        "memory preflight: 6.0 GB for the test buffer; this process private 11.0 GB + 6.0 GB = "
        "17.0 GB vs physical 16.0 GB, and 6.0 GB vs available 5.0 GB: the machine will page "
        "(slower training). Consider a smaller distill.buffer, distill.pin_memory: false, or "
        "closing other programs. Not stopping anything."
    ]
    assert "system commit 40.0 GB" in caplog.text  # commit is logged, not judged


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
    assert out["overcommit"] is out["over_process"] is out["over_available"] is False


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


def test_memory_status_reads_this_process():
    st = _real_status()
    for k in ("process_private", "process_working_set"):
        assert isinstance(st[k], int) and st[k] > 0, k
    # it follows a real allocation (64 MB, touched)
    blob = bytearray(64 << 20)
    after = memcheck.memory_status()["process_private"]
    assert after - st["process_private"] > 32 << 20
    del blob
