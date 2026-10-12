"""UC67 AC2 / [C4]: no Meteor75 (airframe) constants in ``src/drone_fly/sim``.

The prototype's airframe is ``M = 0.032``, ``TW = 2.5``, ``RATE_KP = 18`` (``KP * DT = 0.36``),
``HOVER_THR = 0.4``, ``MAX_BODY_RATE = 4.0``, ``MAX_THRUST`` and ``DT = 0.02``. In the promoted
simulator those numbers live only in ``drones/meteor75/drone.yaml``, the training configs and
the checkpoint sidecars. The scan is targeted rather than allowlisted: the values that are never
legitimate there (0.032, 2.5, 18, 0.36) are refused anywhere; the ambiguous ones (0.4, 4.0,
0.02 are also course geometry, reward weights, a sensing range and an EMA weight) are refused in
an airframe role: bound to an airframe-like name, as a parameter default of one, or scaling a
``tanh`` (the full-stick body rate). The prototype's constant names may not be defined either,
and the action / reward scales have no default [C4]. The checker is itself tested on a planted
violation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SIM = Path(__file__).resolve().parent.parent / "src" / "drone_fly" / "sim"

NEVER = {0.032, 2.5, 18.0, 0.36}
AMBIGUOUS = {0.4, 4.0, 0.02}
AIRFRAME_NAME = re.compile(
    r"hover|thr|thrust|collective|rate|body|kp|mass|weight|tw|(^|_)dt($|_)|period|hz",
    re.IGNORECASE,
)
PROTOTYPE_NAMES = {
    "M", "TW", "G", "WEIGHT", "RATE_KP", "HOVER_THR", "MAX_BODY_RATE", "MAX_THRUST", "DT",
}  # fmt: skip
NO_DEFAULT = {"collective_scale_g", "max_rate_rad_s", "reward_reference_dt_s"}


def _num(node: ast.AST) -> float | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _num(node.operand)
        return None if v is None else -v
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in _target_names(e)]
    return []


def _is_tanh(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Attribute) and f.attr == "tanh") or (
        isinstance(f, ast.Name) and f.id == "tanh"
    )


def violations(source: str, name: str = "<src>") -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []

    def bad(node: ast.AST, why: str) -> None:
        out.append(f"{name}:{getattr(node, 'lineno', '?')}: {why}")

    for node in ast.walk(tree):
        v = _num(node)
        if v is not None and abs(v) in NEVER:
            bad(node, f"airframe literal {v}")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n for t in targets for n in _target_names(t)]
            for n in names:
                if n in PROTOTYPE_NAMES:
                    bad(node, f"prototype constant name {n}")
            val = node.value
            if val is not None and _num(val) in AMBIGUOUS:
                for n in names:
                    if AIRFRAME_NAME.search(n):
                        bad(node, f"{n} = {_num(val)}")
            if val is not None:
                for n in names:
                    if n in NO_DEFAULT and not (
                        isinstance(val, ast.Constant) and val.value is None
                    ):
                        bad(node, f"{n} has a default")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            pos = a.posonlyargs + a.args
            pairs = list(zip(pos[len(pos) - len(a.defaults) :], a.defaults, strict=True))
            pairs += [(k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d]
            for arg, d in pairs:
                if _num(d) in AMBIGUOUS and AIRFRAME_NAME.search(arg.arg):
                    bad(d, f"default {arg.arg}={_num(d)}")
                if arg.arg in NO_DEFAULT and not (isinstance(d, ast.Constant) and d.value is None):
                    bad(d, f"{arg.arg} has a default")
        if isinstance(node, ast.keyword) and node.arg and _num(node.value) in AMBIGUOUS:
            if AIRFRAME_NAME.search(node.arg):
                bad(node.value, f"keyword {node.arg}={_num(node.value)}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for x, y in ((node.left, node.right), (node.right, node.left)):
                if _is_tanh(x) and _num(y) is not None:
                    bad(node, f"tanh(...) * {_num(y)} (a hard-coded full-stick rate)")
    return out


def test_the_checker_catches_planted_violations():
    planted = """
MAX_BODY_RATE = 4.0
hover_thr = 0.4
def f(x, dt=0.02, *, collective_scale_g=1.0):
    return torch.tanh(x) * 4.0 + 0.032
class C:
    reward_reference_dt_s: float = 0.02
    max_rate_rad_s: float | None = None
plant.step(a, b, dt=0.02)
"""
    found = "\n".join(violations(planted))
    for needle in (
        "prototype constant name MAX_BODY_RATE",
        "hover_thr = 0.4",
        "default dt=0.02",
        "collective_scale_g has a default",
        "tanh(...) * 4.0",
        "airframe literal 0.032",
        "reward_reference_dt_s has a default",
        "keyword dt=0.02",
    ):
        assert needle in found, needle
    assert "max_rate_rad_s" not in found  # None = no default
    # legitimate uses of the ambiguous values are not flagged
    clean = "RAY_RANGE = 4.0\nGAP_BETA = 0.4\nok = (tt > 0.02) & (x < 0.98)\nn = a[1:4]\n"
    assert violations(clean) == []


@pytest.mark.parametrize("path", sorted(SIM.glob("*.py")), ids=lambda p: p.name)
def test_no_airframe_constants_in_sim(path):
    assert violations(path.read_text(encoding="utf-8"), path.name) == []


def test_sim_does_not_import_the_prototype_or_the_harness():
    """[C10] sim computes the Betaflight curve itself; it never imports drone_fly.harness, and
    nothing in it reaches gpu_prototype."""
    for path in sorted(SIM.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not m.startswith("drone_fly.harness"), (path.name, m)
                assert m.split(".")[0] not in {
                    "hover_gpu", "gates_dr", "gates_proc_env", "reservoir_aug", "inference",
                    "gates_gpu", "course_gen",
                } or m.startswith("drone_fly"), (path.name, m)  # fmt: skip


def test_importing_sim_loads_no_harness_module():
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import drone_fly.sim, drone_fly.sim.course, drone_fly.sim.replay\n"
        "print(sorted(m for m in sys.modules if m.startswith('drone_fly.harness')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"
