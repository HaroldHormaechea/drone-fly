"""UC-86 AC6 golden observations: generator + scenario code shared with the migration test.

The golden ``tests/fixtures/uc86/observations_golden.npz`` was produced from the pre-UC-86 base
commit (156432e) with **only base-commit APIs** (no ``drone_fly.harness.perception``)::

    git archive 156432e src | tar -x -C $TEMP/uc86-base
    PYTHONPATH=$TEMP/uc86-base/src CUDA_VISIBLE_DEVICES= python tests/uc86_golden.py \
        --out tests/fixtures/uc86/observations_golden.npz

``tests/test_perception_migration.py`` re-runs :func:`compute` on the current tree with
``runtime=True`` (a ``PerceptionRuntime`` validating every estimate and writing its latency rows)
and asserts every assembled ``obs`` / ``extra`` vector is ``np.array_equal`` to the golden.

Scenarios (each a sequence of assembler outputs):

* ``loopback70`` / ``loopback30`` -- ``run_harness("sim-loopback")`` with the default parts
  (course gates + raycast free space), seeded fixture checkpoints, 20 ticks;
* ``liftoff_none30`` -- the ``none`` free-space part (``assume_no_obstacles: true``) built by name
  from the packaged liftoff config, 30-dim assembly;
* ``depth70`` -- the ``depth`` part built from the liftoff config with an injected deterministic
  depth model (synchronous), 70-dim assembly;
* ``eyes70`` -- the ``eyes`` part built from the liftoff config with seeded tiny weights
  (synchronous, ``ray_clusters`` obstacle vision), 70-dim assembly.

The obs are captured at ``ContractObservationAssembler.assemble`` (before anything the runner may
do with them afterwards), so the comparison isolates the perception path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GPU_PROTOTYPE_DIR = REPO / "gpu_prototype"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURE_DIR / "uc86" / "observations_golden.npz"
SCENARIOS = ("loopback70", "loopback30", "liftoff_none30", "depth70", "eyes70")
EYES_CAM = {"resolution": [128, 54], "hfov_deg": 120.0, "uptilt_deg": 20.0}
DEPTH_RES = (320, 240)
STEPS = 12


def load_fixture_contract(k1_dir: Path) -> Any:
    """``load_contract`` exactly as the ``harness_contract`` conftest fixture does."""
    from drone_fly.harness.contract import load_contract

    prev = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        contract = load_contract(GPU_PROTOTYPE_DIR, k1_dir, "cpu")
        contract.import_reference("course_gen")
        contract.import_reference("gates_proc_env")
    finally:
        sys.dont_write_bytecode = prev
    return contract


def save_checkpoint(extra_dim: int, path: Path) -> Path:
    """The seeded stripped readout checkpoint of the conftest fixtures."""
    import torch

    torch.manual_seed(0)
    actor = sys.modules["reservoir_aug"].K1ReservoirAug(extra_dim)
    stripped = {k: v for k, v in actor.state_dict().items() if not k.startswith("body.layer.")}
    torch.save(stripped, path)
    return path


# ----------------------------------------------------------------------------- helpers


class _Capture:
    """Records every ``ContractObservationAssembler.assemble`` output while active."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.extra: list[np.ndarray] = []

    def __enter__(self) -> _Capture:
        from drone_fly.harness.assembler import ContractObservationAssembler

        self._cls = ContractObservationAssembler
        self._orig = ContractObservationAssembler.assemble
        cap = self

        def assemble(self_, *args: Any, **kw: Any) -> Any:
            obs, extra = cap._orig(self_, *args, **kw)
            cap.obs.append(np.array(obs, copy=True))
            cap.extra.append(np.array(extra, copy=True))
            return obs, extra

        ContractObservationAssembler.assemble = assemble  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        self._cls.assemble = self._orig  # type: ignore[method-assign]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.stack(self.obs), np.stack(self.extra)


def _state(k: int) -> Any:
    from drone_fly.harness.types import FlightState

    yaw = 0.05 * k
    return FlightState(
        pos=np.array([0.4 * k, 0.1 * np.sin(k), 1.0 + 0.02 * k], np.float32),
        vel=np.array([4.0, 0.5 * np.cos(k), 0.1], np.float32),
        quat=np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], np.float32),
        omega=np.array([0.01 * k, -0.02, 0.03], np.float32),
        t=0.1 * k,
        received_at=0.1 * k,
    )


def _gate(k: int) -> Any:
    from drone_fly.harness.types import GateEstimate

    return GateEstimate(
        target=np.array([6.0 + 0.4 * k, 0.5, 1.2], np.float32),
        curr_gate=np.array([1.0, 0.0, 1.5], np.float32),
        next_lookahead=np.array([0.9, 0.1, 0.0, 1.0, 0.0, 0.8, 1.0], np.float32),
        stamp=0.1 * k,
    )


def _image(k: int, w: int, h: int) -> np.ndarray:
    return np.random.default_rng(100 + k).integers(0, 256, (h, w, 3), dtype=np.uint8)


class _DepthModel:
    """Deterministic depth from the image (no weights): 0.5 m .. 4.5 m."""

    def load(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        return (0.5 + image[..., 0].astype(np.float32) / 64.0).astype(np.float32)


def _drive(provider: Any, contract: Any, extra_dim: int, w: int, h: int, runtime: bool) -> Any:
    """Feed ``STEPS`` frames through ``provider`` and assemble each tick."""
    from drone_fly.harness.assembler import ContractObservationAssembler
    from drone_fly.harness.types import Frame

    asm = ContractObservationAssembler(contract)
    rt = None
    if runtime:
        from drone_fly.harness.perception import PerceptionRuntime

        gate_part = SimpleNamespace(provides=frozenset({"target", "curr_gate", "next_lookahead"}),
                                    provenance="screen", provider_name="golden-gate")  # fmt: skip
        rt = PerceptionRuntime(
            SimpleNamespace(gate_estimator=gate_part, free_space_estimator=provider)
        )
    provider.start()
    try:
        with _Capture() as cap:
            for k in range(STEPS):
                st, gate = _state(k), _gate(k)
                now = 0.1 * k
                fs = provider.update(Frame(_image(k, w, h), now, k), st, gate.target, now)
                if rt is not None:
                    rec = SimpleNamespace(perception_s={}, ages_s={})
                    rt.after_update(rec, gate, fs, 0.001, 0.002, now)
                asm.assemble(st, gate, fs, extra_dim)
    finally:
        provider.stop()
    return cap.arrays()


# ----------------------------------------------------------------------------- scenarios


def _loopback(extra_dim: int, ckpt: Path, k1_dir: Path, out: Path) -> Any:
    from drone_fly.harness.session import run_harness

    with _Capture() as cap:
        session = run_harness(
            "sim-loopback",
            str(ckpt),
            extra_dim,
            k1_path=str(k1_dir),
            gpu_prototype_dir=str(GPU_PROTOTYPE_DIR),
            device="cpu",
            duration_s=0.4,
            output_root=str(out),
            record=False,
        )
    assert session.result.reason == "completed", session.result
    return cap.arrays()


def _liftoff_none(contract: Any, runtime: bool) -> Any:
    from drone_fly.harness.config import load_config
    from drone_fly.harness.registry import build_part

    cfg = load_config("liftoff", {"free_space": {"assume_no_obstacles": True}}, preset="liftoff")
    ctx = SimpleNamespace(config=cfg, contract=contract, resources={"frame_source": None})
    fs = build_part("free_space_estimator", "none", cfg.section("free_space"), ctx)
    return _drive(fs, contract, 30, 64, 32, runtime)


def _depth(contract: Any, runtime: bool) -> Any:
    from drone_fly.harness.config import load_config
    from drone_fly.harness.registry import build_part

    cfg = load_config(
        "liftoff",
        {"parts": {"free_space_estimator": "depth"}, "camera": {"resolution": list(DEPTH_RES)}},
        preset="liftoff",
    )
    ctx = SimpleNamespace(
        config=cfg,
        contract=contract,
        resources={"depth_model": _DepthModel(), "frame_source": None, "state_source": None},
    )
    fs = build_part("free_space_estimator", "depth", cfg.section("free_space"), ctx)
    return _drive(fs, contract, 70, *DEPTH_RES, runtime)


def _eyes(contract: Any, tmp: Path, runtime: bool) -> Any:
    import torch

    from drone_fly.eyes.model import build_from_meta, default_meta
    from drone_fly.harness.config import load_config
    from drone_fly.harness.registry import build_part

    meta = default_meta(
        cone=np.asarray(contract.CONE),
        camera=dict(EYES_CAM),
        masks=[],
        world_scale=4.0,
        ray_range_scale=1.0,
        width=64,
        height=32,
        ray_range=float(contract.RAY_RANGE),
    )
    torch.manual_seed(7)
    net = build_from_meta(meta)
    weights = tmp / "eyes.pt"
    torch.save({"meta": meta, "state_dict": net.state_dict()}, weights)
    cfg = load_config(
        "liftoff",
        {
            "parts": {"free_space_estimator": "eyes"},
            "eyes": {"weights": str(weights), "device": "cpu", "obstacle_vision": "ray_clusters"},
        },
        preset="liftoff",
    )
    ctx = SimpleNamespace(
        config=cfg, contract=contract, resources={"frame_source": None, "state_source": None}
    )
    fs = build_part("free_space_estimator", "eyes", cfg.section("free_space"), ctx)
    w, h = EYES_CAM["resolution"]
    return _drive(fs, contract, 70, w, h, runtime)


def compute(
    contract: Any, ckpt70: Path, ckpt30: Path, k1_dir: Path, tmp: Path, *, runtime: bool = False
) -> dict[str, np.ndarray]:
    """Every scenario's stacked ``obs`` / ``extra`` (keys ``<scenario>_obs`` / ``_extra``).

    ``runtime=True`` (current tree only) attaches a ``PerceptionRuntime`` to the direct-drive
    scenarios; ``run_harness`` always attaches one on the current tree.
    """
    torch_threads(1)
    results = {
        "loopback70": _loopback(70, ckpt70, k1_dir, tmp / "lb70"),
        "loopback30": _loopback(30, ckpt30, k1_dir, tmp / "lb30"),
        "liftoff_none30": _liftoff_none(contract, runtime),
        "depth70": _depth(contract, runtime),
        "eyes70": _eyes(contract, tmp, runtime),
    }
    out: dict[str, np.ndarray] = {}
    for name, (obs, extra) in results.items():
        out[f"{name}_obs"] = obs
        out[f"{name}_extra"] = extra
    return out


def torch_threads(n: int) -> None:
    import torch

    torch.set_num_threads(n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(GOLDEN))
    args = ap.parse_args(argv)
    import drone_fly

    print(f"drone_fly from {drone_fly.__file__}")
    with tempfile.TemporaryDirectory(prefix="uc86-golden-") as td:
        tmp = Path(td)
        k1 = tmp / "k1"
        k1.mkdir()
        for src in FIXTURE_DIR.iterdir():
            if src.is_file():
                shutil.copy2(src, k1 / src.name)
        load_fixture_contract(k1)
        ck70 = save_checkpoint(70, tmp / "ck70.pt")
        ck30 = save_checkpoint(30, tmp / "ck30.pt")
        contract = load_fixture_contract(k1)
        arrays = compute(contract, ck70, ck30, k1, tmp, runtime=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    for k, v in arrays.items():
        print(k, v.shape, v.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
