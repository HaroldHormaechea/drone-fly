"""UC-78 AC5: the deployable eyes path is vision-only.

* **Static**: the deployable modules (camera_rays, history, ego, model, infer) import only numpy,
  torch, ``drone_fly.harness.geometry`` / ``drone_fly.harness.types`` and each other -- at module
  level AND inside functions (an AST walk of every import statement).
* **Dynamic**: a fresh interpreter builds an ``EyesEstimator`` from saved weights and runs
  ``update`` with history; afterwards no ``drone_fly.sim``, tracks, harness offline / contract /
  oracle / course / loopback / liftoff module, no gpu_prototype contract module and no offline
  eyes module is loaded (OpenCV / capture backends are blocked so a lazy import would fail).
* **Signature**: ``update(frame, state, target)`` -- no geometry, label or oracle argument.
"""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from drone_fly.eyes.infer import EyesEstimator

REPO = Path(__file__).resolve().parent.parent
EYES = REPO / "src" / "drone_fly" / "eyes"
DEPLOYABLE = ("camera_rays", "history", "ego", "model", "infer")
ALLOWED_ROOTS = {
    "numpy",
    "torch",
    "__future__",
    "collections",
    "dataclasses",
    "types",
    "typing",
    "time",
    "pathlib",
    "math",
}
ALLOWED_DRONE_FLY = {
    "drone_fly.harness.geometry",
    "drone_fly.harness.types",
    *(f"drone_fly.eyes.{m}" for m in DEPLOYABLE),
}
FORBIDDEN_PREFIXES = (
    "drone_fly.sim",
    "drone_fly.tracks",
    "drone_fly.harness.offline",
    "drone_fly.harness.contract",
    "drone_fly.harness.oracle",
    "drone_fly.harness.course",
    "drone_fly.harness.loopback",
    "drone_fly.harness.liftoff",
    "drone_fly.eyes.labels",
    "drone_fly.eyes.dataset",
    "drone_fly.eyes.train",
    "drone_fly.eyes.evaluate",
    "drone_fly.eyes.fov",
    "drone_fly.eyes.capture_meta",
    "drone_fly.eyes.bench",
    "drone_fly.eyes.cli",
    "inference",
    "gates_proc_env",
    "course_gen",
)


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
    return out


def test_deployable_modules_import_only_the_allowed_set():
    for name in DEPLOYABLE:
        for mod in _imports(EYES / f"{name}.py"):
            if mod.startswith("drone_fly"):
                assert mod in ALLOWED_DRONE_FLY, f"{name}.py imports {mod}"
            else:
                assert mod.split(".")[0] in ALLOWED_ROOTS, f"{name}.py imports {mod}"


def test_package_init_imports_nothing():
    assert _imports(EYES / "__init__.py") == set()


def test_update_signature_takes_only_frame_state_target():
    sig = inspect.signature(EyesEstimator.update)
    assert list(sig.parameters) == ["self", "frame", "state", "target"]
    init = inspect.signature(EyesEstimator.__init__)
    assert list(init.parameters) == ["self", "net", "meta", "device"]


_PROBE = textwrap.dedent(
    """
    import sys
    for _n in ("cv2", "mss", "dxcam", "transformers", "vgamepad", "PIL", "PySide6"):
        sys.modules[_n] = None
    import json, tempfile, os
    import numpy as np, torch
    from drone_fly.eyes.infer import EyesEstimator
    from drone_fly.eyes.model import build_from_meta, default_meta
    from drone_fly.harness.types import Frame, FlightState

    i = np.arange(40) + 0.5
    z = 1 - (1 - np.cos(np.radians(45))) * i / 40
    r = np.sqrt(1 - z * z); phi = np.pi * (3 - np.sqrt(5)) * i
    cone = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    meta = default_meta(cone=cone, camera={"resolution": [256, 108], "hfov_deg": 120.0,
                        "uptilt_deg": 20.0}, masks=[[0, 0, 0.1, 0.1]], world_scale=4.0,
                        ray_range_scale=1.0, ray_range=4.0, width=64, height=32)
    path = os.path.join(tempfile.mkdtemp(), "eyes.pt")
    torch.save({"meta": meta, "state_dict": build_from_meta(meta).state_dict()}, path)
    est = EyesEstimator.from_weights(path, "cpu")
    rng = np.random.default_rng(0)
    ok = []
    for k in range(8):
        st = FlightState(np.array([0.3 * k, 0, 1], np.float32), np.array([3, 0, 0], np.float32),
                         np.array([0.98, 0, 0.17, 0], np.float32), np.zeros(3, np.float32))
        img = rng.integers(0, 256, (108, 256, 3), dtype=np.uint8)
        ok.append(est.update(Frame(img, 0.09 * k, k), st, np.array([5.0, 0, 1])).history_ok)
    print(json.dumps({"ok": ok, "modules": sorted(sys.modules)}))
    """
)


def test_fresh_interpreter_update_loads_no_geometry_or_oracle_module():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO / "src"),
            "PYTHONIOENCODING": "utf-8",
        },
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["ok"][-1] is True  # the network actually ran
    loaded = [
        m
        for m in res["modules"]
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PREFIXES)
    ]
    assert loaded == []
    eyes = {m for m in res["modules"] if m.startswith("drone_fly.eyes")}
    assert eyes <= {"drone_fly.eyes", *(f"drone_fly.eyes.{m}" for m in DEPLOYABLE)}
