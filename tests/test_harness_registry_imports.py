"""UC-62 AC1 / AC13: harness interfaces, name registry, lazy backends, hermetic CI imports.

The hermetic checks run in a fresh interpreter with every heavy backend (torch, OpenCV, screen
capture, transformers, vgamepad, PIL) blocked at import, because the test process itself has
torch loaded by other tests.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness import registry
from drone_fly.harness.errors import (
    INSTALL_HINT,
    HarnessConfigError,
    HarnessDependencyError,
    PerceptionCoverageError,
    lazy_import,
)
from drone_fly.harness.interfaces import (
    AbortSignal,
    ActionSink,
    FrameSource,
    FreeSpaceEstimator,
    GateEstimator,
    Harness,
    ObservationAssembler,
    PolicyRunner,
    StateSource,
)
from tests.harness_support import HEAVY_MODULES

REPO = Path(__file__).resolve().parent.parent
HARNESS_SRC = REPO / "src" / "drone_fly" / "harness"

GPU_PROTOTYPE_MODULES = (
    "hover_gpu",
    "gates_gpu",
    "gates_k1_reservoir",
    "reservoir_aug",
    "gates_proc_env",
    "course_gen",
    "inference",
)

#: Prelude for the hermetic subprocess: every heavy backend behaves as "not installed"
#: (a ``None`` entry in sys.modules makes ``import`` raise ImportError and ``find_spec`` -> None).
_BLOCKER = textwrap.dedent(
    f"""
    import sys
    BLOCKED = {HEAVY_MODULES!r}
    for _name in BLOCKED:
        sys.modules[_name] = None
    def loaded_heavy():
        return sorted(
            m for m, v in sys.modules.items() if v is not None and m.split(".")[0] in BLOCKED
        )
    """
)


def _run_hermetic(body: str, *args: str) -> subprocess.CompletedProcess[str]:
    code = _BLOCKER + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=180,
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(REPO / "src"),
            "PYTHONIOENCODING": "utf-8",
        },
    )


# ----------------------------------------------------------------------------- interfaces


@pytest.mark.parametrize(
    "abc",
    [
        StateSource,
        FrameSource,
        GateEstimator,
        FreeSpaceEstimator,
        ObservationAssembler,
        ActionSink,
        PolicyRunner,
        AbortSignal,
    ],
)
def test_interfaces_are_abstract(abc):
    with pytest.raises(TypeError):
        abc()


def test_harness_provided_blocks_union_of_estimators():
    class G(GateEstimator):
        provides = frozenset({"target", "curr_gate"})

        def update(self, frame, state, now):  # pragma: no cover - not called
            raise NotImplementedError

    class F(FreeSpaceEstimator):
        provides = frozenset({"obstacle_vision"})

        def update(self, frame, state, target, now):  # pragma: no cover - not called
            raise NotImplementedError

    class S(ActionSink):
        def send(self, command):  # pragma: no cover
            return {}

        def failsafe(self):  # pragma: no cover
            pass

    class St(StateSource):
        def latest(self):  # pragma: no cover
            return None

        def age(self, now):  # pragma: no cover
            return 0.0

    h = Harness("x", St(), G(), S())
    assert h.provided_blocks() == frozenset({"target", "curr_gate"})
    h.free_space_estimator = F()
    assert h.provided_blocks() == frozenset({"target", "curr_gate", "obstacle_vision"})


# ----------------------------------------------------------------------------- errors


def test_dependency_error_names_package_and_install_hint():
    err = HarnessDependencyError("opencv-python", "the OpenCV gate estimator")
    assert isinstance(err, ImportError)
    assert "opencv-python" in str(err) and INSTALL_HINT in str(err)
    assert INSTALL_HINT == "uv sync --extra harness"


def test_lazy_import_missing_module_raises_dependency_error():
    with pytest.raises(HarnessDependencyError, match="no-such-dist") as ei:
        lazy_import("no_such_module_xyz_62", package="no-such-dist", purpose="testing")
    assert ei.value.package == "no-such-dist"
    assert lazy_import("json") is json


def test_coverage_error_names_every_missing_block():
    err = PerceptionCoverageError(["raycast", "obstacle_vision"], 70)
    assert err.missing == ("raycast", "obstacle_vision")
    assert "raycast" in str(err) and "obstacle_vision" in str(err) and "70" in str(err)


def test_config_error_is_value_error():
    assert issubclass(HarnessConfigError, ValueError)


# ----------------------------------------------------------------------------- registry


def test_builtins_registered_under_every_expected_name():
    expected = {
        "harness": {"sim-loopback", "liftoff"},
        "state_source": {"plant", "liftoff-udp", "liftoff-csv"},
        "frame_source": {"screen"},
        "gate_estimator": {"course", "opencv"},
        "free_space_estimator": {"raycast", "none", "depth", "eyes", "composite"},
        "action_sink": {"plant", "vgamepad", "recording"},
        "abort_signal": {"hotkey"},
        "depth_model": {"depth-anything-v2"},
    }
    for kind, names in expected.items():
        assert names <= set(registry.names(kind)), kind
    assert registry.entry("harness", "liftoff").default_config == "liftoff_5inch.yaml"
    assert registry.entry("harness", "sim-loopback").default_config == "loopback.yaml"


def test_builtin_targets_are_lazy_strings():
    for kind in registry.KINDS:
        for name in registry.names(kind):
            target = registry.entry(kind, name).target
            assert isinstance(target, str) and ":" in target, (kind, name)


def test_register_resolve_unregister_roundtrip():
    class Thing:
        pass

    registry.register("gate_estimator", "uc62-test-thing", Thing)
    try:
        assert registry.resolve("gate_estimator", "uc62-test-thing") is Thing
        with pytest.raises(HarnessConfigError, match="already registered"):
            registry.register("gate_estimator", "uc62-test-thing", Thing)
        registry.register("gate_estimator", "uc62-test-thing", "json:dumps", replace=True)
        assert registry.resolve("gate_estimator", "uc62-test-thing") is json.dumps
    finally:
        registry.unregister("gate_estimator", "uc62-test-thing")
    assert "uc62-test-thing" not in registry.names("gate_estimator")


def test_registry_rejects_unknown_kind_name_and_bad_target():
    with pytest.raises(HarnessConfigError, match="unknown part kind"):
        registry.register("not_a_kind", "x", object)
    with pytest.raises(HarnessConfigError, match="module:attr"):
        registry.register("gate_estimator", "x", "no_colon_here")
    with pytest.raises(HarnessConfigError, match="registered: .*course"):
        registry.entry("gate_estimator", "does-not-exist")


def test_resolve_missing_backend_raises_dependency_error_with_pip_name():
    registry.register(
        "gate_estimator",
        "uc62-missing",
        "no_such_pkg_uc62.sub:Thing",
        packages={"no_such_pkg_uc62": "no-such-pip-dist"},
        requires=("no_such_pkg_uc62",),
    )
    try:
        with pytest.raises(HarnessDependencyError) as ei:
            registry.resolve("gate_estimator", "uc62-missing")
        assert "no-such-pip-dist" in str(ei.value) and INSTALL_HINT in str(ei.value)
        probe = registry.probe("gate_estimator", "uc62-missing")
        assert probe == {"available": False, "missing": ["no-such-pip-dist"]}
    finally:
        registry.unregister("gate_estimator", "uc62-missing")


def test_probe_alternatives_any_one_suffices():
    registry.register("frame_source", "uc62-alt", "json:dumps", requires=("no_such_x_62|json",))
    try:
        assert registry.probe("frame_source", "uc62-alt")["available"] is True
    finally:
        registry.unregister("frame_source", "uc62-alt")


def test_build_part_prefers_from_config_then_callable():
    seen = {}

    class WithFactory:
        @classmethod
        def from_config(cls, cfg, ctx):
            seen["factory"] = (cfg, ctx)
            return "built-by-factory"

    def plain(cfg, ctx):
        seen["plain"] = (cfg, ctx)
        return "built-by-call"

    registry.register("action_sink", "uc62-f", WithFactory)
    registry.register("action_sink", "uc62-p", plain)
    try:
        assert registry.build_part("action_sink", "uc62-f", {"a": 1}, "CTX") == "built-by-factory"
        assert registry.build_part("action_sink", "uc62-p", {"b": 2}, "CTX") == "built-by-call"
        assert seen == {"factory": ({"a": 1}, "CTX"), "plain": ({"b": 2}, "CTX")}
    finally:
        registry.unregister("action_sink", "uc62-f")
        registry.unregister("action_sink", "uc62-p")


# ----------------------------------------------------------------------------- AC13


def test_programmatically_registered_estimator_runs_through_unchanged_runner(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    """A new estimator = implement + register; the loopback factory and runner are not edited."""
    from drone_fly.harness.loopback import CourseGateEstimator
    from drone_fly.harness.session import run_harness

    calls = {"n": 0, "resources": None}

    class CountingGateEstimator(CourseGateEstimator):
        @classmethod
        def from_config(cls, cfg, ctx):
            calls["resources"] = sorted(ctx.resources)
            calls["cfg"] = cfg
            return cls(ctx.resources["course"])

        def update(self, frame, state, now):
            calls["n"] += 1
            return super().update(frame, state, now)

    registry.register("gate_estimator", "uc62-counting", CountingGateEstimator)
    try:
        session = run_harness(
            "sim-loopback",
            harness_checkpoint_70,
            70,
            config={
                "parts": {"gate_estimator": "uc62-counting"},
                "part_options": {"gate_estimator": {"note": "free-form"}},
            },
            k1_path=str(harness_k1_dir),
            device="cpu",
            duration_s=0.2,
            output_root=str(tmp_path),
            record=False,
        )
    finally:
        registry.unregister("gate_estimator", "uc62-counting")
    assert session.result.reason == "completed"
    assert session.result.ticks == 10 and calls["n"] == 10
    assert calls["resources"] == ["clock", "course", "plant"]
    assert calls["cfg"] == {"note": "free-form"}


# ----------------------------------------------------------------------------- hermetic CI path


def test_hermetic_import_and_list_without_heavy_backends():
    """AC1: the package, every runtime module, the CLI and `harness list` work with no backends."""
    proc = _run_hermetic(
        """
        import importlib, io, contextlib, json
        GPU_MODULES = GPU_MODULES_PLACEHOLDER
        mods = [
            "drone_fly.harness", "drone_fly.harness.session", "drone_fly.harness.loopback",
            "drone_fly.harness.runner", "drone_fly.harness.recording", "drone_fly.harness.contract",
            "drone_fly.harness.policy", "drone_fly.harness.assembler", "drone_fly.harness.geometry",
            "drone_fly.harness.liftoff", "drone_fly.harness.liftoff.telemetry",
            "drone_fly.harness.liftoff.gyro_check", "drone_fly.harness.liftoff.frames",
            "drone_fly.harness.liftoff.gate_cv", "drone_fly.harness.liftoff.gamepad",
            "drone_fly.harness.liftoff.abort", "drone_fly.harness.liftoff.depth",
            "drone_fly.harness.estimators", "drone_fly.cli.main",
            "drone_fly.harness.perception", "drone_fly.harness.composite",
            "drone_fly.harness.perception_kit", "drone_fly.harness.examples.brightness",
        ]
        for m in mods:
            importlib.import_module(m)
        from drone_fly.cli.main import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["harness", "list"])
        print(json.dumps({
            "code": code,
            "out": buf.getvalue(),
            "heavy": loaded_heavy(),
            "gpu": sorted(m for m in sys.modules if m in GPU_MODULES),
            "offline": "drone_fly.harness.offline" in sys.modules,
        }))
        """.replace("GPU_MODULES_PLACEHOLDER", repr(GPU_PROTOTYPE_MODULES))
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["code"] == 0
    assert res["heavy"] == [] and res["gpu"] == []
    assert res["offline"] is False, "the runtime must never import the offline (XML) package"
    out = res["out"]
    for name in ("sim-loopback", "liftoff", "opencv", "depth", "vgamepad", "screen", "hotkey"):
        assert re.search(rf"\b{re.escape(name)}\b", out), name
    # Backends blocked -> their parts are listed as missing with the install hint.
    assert re.search(
        r"gate_estimator\s+opencv\s+missing.*opencv-python.*uv sync --extra harness", out
    )
    assert re.search(r"action_sink\s+vgamepad\s+missing", out)
    assert re.search(r"harness\s+sim-loopback\s+available", out)


def test_hermetic_uc86_perception_modules_import_without_heavy_backends():
    """UC-86 AC3: the perception contract, composite, kit, eval, CLI and example plugin import
    (and an import-reference plugin builds) with every heavy backend blocked."""
    proc = _run_hermetic(
        """
        import importlib, json
        for m in ("drone_fly.harness.perception", "drone_fly.harness.composite",
                  "drone_fly.harness.perception_kit", "drone_fly.harness.offline.perception_cli",
                  "drone_fly.harness.examples.brightness",
                  "drone_fly.harness.offline.perception_eval"):
            importlib.import_module(m)
        from types import SimpleNamespace
        from drone_fly.harness.registry import build_part
        part = build_part("free_space_estimator", "drone_fly.harness.estimators:"
                          "NoObstacleFreeSpaceEstimator", {}, SimpleNamespace(config=None))
        print(json.dumps({"heavy": loaded_heavy(), "name": part.provider_name}))
        """
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["heavy"] == []
    assert res["name"] == "drone_fly.harness.estimators:NoObstacleFreeSpaceEstimator"


def test_hermetic_heavy_use_raises_actionable_dependency_errors():
    """Using a backend without it installed raises HarnessDependencyError naming the package."""
    proc = _run_hermetic(
        """
        import json
        import numpy as np
        from drone_fly.harness.errors import HarnessDependencyError
        from drone_fly.harness.contract import load_contract
        from drone_fly.harness.geometry import CameraModel
        from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator
        from drone_fly.harness.liftoff.gamepad import default_pad_factory
        from drone_fly.harness.liftoff.depth import DepthAnythingV2Model
        out = {}
        def attempt(key, fn):
            try:
                fn()
                out[key] = "no error"
            except HarnessDependencyError as exc:
                out[key] = [exc.package, str(exc)]
        attempt("contract", lambda: load_contract(sys.argv[1], None, "cpu"))
        hsv = [([0, 0, 0], [10, 255, 255])]
        est = OpenCvGateEstimator(CameraModel(64, 48, 90.0), 1.5, 1.5, hsv)
        attempt("cv2", lambda: est.detect_quads(np.zeros((48, 64, 3), np.uint8)))
        attempt("vgamepad", default_pad_factory)
        attempt("depth", lambda: DepthAnythingV2Model().load())
        print(json.dumps(out))
        """,
        str(REPO / "gpu_prototype"),
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["contract"][0] == "torch"
    assert out["cv2"][0] == "opencv-python"
    assert out["vgamepad"][0] == "vgamepad"
    assert out["depth"][0] in ("transformers", "torch")
    for key in ("contract", "cv2", "vgamepad", "depth"):
        assert "uv sync --extra harness" in out[key][1], key


def test_hermetic_cli_run_exits_with_dependency_code():
    proc = _run_hermetic(
        """
        from drone_fly.cli.main import main
        sys.exit(main(["harness", "run", "--harness", "sim-loopback", "--checkpoint", "x.pt",
                       "--extra-dim", "70", "--device", "cpu", "--no-record"]))
        """
    )
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert "uv sync --extra harness" in proc.stderr


def _imports_of(path: Path) -> list[str]:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_runtime_modules_never_import_offline_or_heavy_at_module_level():
    """Static check: only harness/offline reads track XML; heavy backends are never top-level."""
    import ast

    runtime = [
        p for p in HARNESS_SRC.rglob("*.py") if "offline" not in p.relative_to(HARNESS_SRC).parts
    ]
    assert runtime
    for path in runtime:
        assert not any("offline" in n or "track_xml" in n for n in _imports_of(path)), path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert m.split(".")[0] not in HEAVY_MODULES, (path, m)
    assert "xml.etree" not in " ".join(n for p in runtime for n in _imports_of(p)), (
        "track/race XML must only be parsed by the offline package"
    )


def test_harness_package_never_reaches_gpu_prototype_except_through_contract():
    for path in HARNESS_SRC.rglob("*.py"):
        if path.name == "contract.py":
            continue
        imports = _imports_of(path)
        for m in GPU_PROTOTYPE_MODULES:
            assert m not in imports, (path, m)


def test_f32_vectors_are_float32_and_length_checked():
    from drone_fly.harness.types import f32

    v = f32([1, 2, 3], 3)
    assert v.dtype == np.float32
    with pytest.raises(ValueError):
        f32([1, 2], 3)
