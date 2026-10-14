"""UC70: ``drone_fly.sim`` stays free of the harness and of the student contract package.

The simulator side (noise model, perception, student observation, student env) may be imported by
``drone_fly.student`` and the harness, never the other way round. Checked statically (no import
statement anywhere in ``sim/`` names them, lazy ones included) and dynamically (a fresh
interpreter importing every UC70 sim module loads neither).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parent.parent / "src" / "drone_fly" / "sim"
FORBIDDEN = ("drone_fly.harness", "drone_fly.student")
PROBE = """
import sys
import drone_fly.sim.noise_model, drone_fly.sim.perception, drone_fly.sim.student_obs
import drone_fly.sim.student_env, drone_fly.sim.multi_course
from drone_fly.sim.student_obs import input_scale
input_scale()
bad = [m for m in sys.modules if m.startswith(("drone_fly.harness", "drone_fly.student"))]
print(bad)
sys.exit(1 if bad else 0)
"""


def _imports(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out


def test_no_sim_module_imports_harness_or_student():
    bad = {p.name: [m for m in _imports(p) if m.startswith(FORBIDDEN)] for p in SIM.rglob("*.py")}
    assert {k: v for k, v in bad.items() if v} == {}


def test_importing_the_uc70_sim_modules_loads_neither():
    env = {**os.environ, "PYTHONPATH": str(SIM.parent.parent)}
    r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env,
                       timeout=300)  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
