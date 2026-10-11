"""UC-65 AC10 / AC11 / AC15: pure-model separation and hermetic CI for the overlay.

The checks run in a fresh interpreter with every heavy backend blocked at import — torch,
OpenCV, capture, transformers, vgamepad, PIL **and PySide6** (``HEAVY_MODULES``) — because the
test venv has PySide6 installed: only an import blocker proves CI with ``--extra dev`` works.
Statically, ``drone_fly/overlay/window.py`` is the only module that imports PySide6, and the
toolkit lives only in the ``overlay`` extra.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

from drone_fly.harness.errors import (
    INSTALL_HINT,
    OVERLAY_INSTALL_HINT,
    HarnessDependencyError,
    lazy_import,
)
from tests.harness_support import HEAVY_MODULES
from tests.test_harness_registry_imports import _run_hermetic

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "drone_fly"
OVERLAY_SRC = SRC / "overlay"
PURE_MODULES = (
    "drone_fly.overlay",
    "drone_fly.overlay.protocol",
    "drone_fly.overlay.state",
    "drone_fly.overlay.receiver",
    "drone_fly.overlay.replay",
    "drone_fly.overlay.brain",
    "drone_fly.overlay.panels",
    "drone_fly.overlay.layout",
    "drone_fly.overlay.win32",
    "drone_fly.overlay.app",
    "drone_fly.harness.overlay",
)


def _last_json(proc) -> dict:
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_pyside6_is_a_heavy_module():
    assert "PySide6" in HEAVY_MODULES


def test_hermetic_import_and_pure_pipeline_without_toolkit_or_torch():
    """Every overlay module except window imports, and the whole data path runs (publisher ->
    protocol -> state -> panels -> layout -> brain renderer), with torch and PySide6 blocked."""
    proc = _run_hermetic(
        """
        import importlib, json
        import numpy as np
        for m in MODULES:
            importlib.import_module(m)
        from drone_fly.harness.overlay import OverlayPublisher
        from drone_fly.overlay.state import OverlayState
        from drone_fly.overlay.panels import build_panels, ScreenMap
        from drone_fly.overlay.layout import resolve_layout, Rect
        from drone_fly.overlay.brain import BrainAsset, BrainRenderer
        from drone_fly.overlay.win32 import configure_overlay_window
        from drone_fly.overlay.replay import replay_sleep_free

        class Sock:
            def __init__(self):
                self.sent = []
            def sendto(self, data, addr):
                self.sent.append(data)
            def close(self):
                pass

        class Rec:
            tick = 0
            def as_dict(self):
                return {"tick": 0, "command": {"raw": [0, 0, 0, 0], "collective": 0.5,
                        "rates": [0, 0, 0]}, "obs": [0.0] * 12}

        sock = Sock()
        q = lambda v: np.clip(np.round((v + 1) * 127.5), 0, 255).astype(np.uint8)
        pub = OverlayPublisher(9100, 3, sock=sock, quantize=q, meta={"extra_dim": 12})
        pub.publish_tick(Rec(), brain=np.zeros(10, np.float32))
        state = OverlayState()
        for d in sock.sent:
            state.ingest_bytes(d)
        snap = state.snapshot()
        models = build_panels(snap, ScreenMap((0, 0, 1920, 1080)))
        lay = resolve_layout("default", 1920, 1080, {"health": True}, affinity_ok=False,
                             masks=[Rect(0, 0, 400, 300)])
        n = 10
        asset = BrainAsset(n=n, points=np.random.default_rng(0).uniform(0, 1, (n, 2)),
                           anatomical=False, modality=[""] * n)
        img = BrainRenderer(asset, 40, 40).render(snap.brain, 0.0).image

        class Shim:
            def get_ex_style(self, h): return 0
            def set_ex_style(self, h, s): pass
            def set_window_pos(self, *a): pass
            def show_no_activate(self, h): pass
            def set_display_affinity(self, h, a): return True
        setup = configure_overlay_window(1, Shim(), (0, 0, 10, 10))
        replay_state = OverlayState()
        replay_sleep_free(sys.argv[1], replay_state)
        window_error = None
        try:
            importlib.import_module("drone_fly.overlay.window")
        except ImportError as exc:
            window_error = type(exc).__name__
        print(json.dumps({
            "heavy": loaded_heavy(),
            "mode": snap.mode,
            "brain_n": snap.brain.n,
            "action": models.action.available,
            "layout": lay.mode,
            "img": list(img.shape),
            "style": setup.ex_style,
            "replay_stop": replay_state.snapshot().last_stop["reason"],
            "window_error": window_error,
        }))
        """.replace("MODULES", repr(PURE_MODULES)),
        str(REPO / "tests" / "data" / "overlay" / "run"),
    )
    res = _last_json(proc)
    assert res["heavy"] == []
    assert res["mode"] == "live" and res["brain_n"] == 10 and res["action"] is True
    assert res["layout"] == "mask_confined" and res["img"] == [40, 40, 4]
    assert res["style"] == 0x080800A8 and res["replay_stop"] == "focus_lost"
    assert res["window_error"] in ("ImportError", "ModuleNotFoundError")  # window needs Qt


def test_hermetic_cli_overlay_without_pyside6_exits_3_with_install_hint():
    proc = _run_hermetic(
        """
        from drone_fly.cli.main import main
        sys.exit(main(["harness", "overlay", "--replay", "nowhere"]))
        """
    )
    assert proc.returncode == 3, (proc.stdout, proc.stderr)
    assert "--extra overlay" in proc.stderr and "PySide6" in proc.stderr


def test_hermetic_harness_run_path_still_never_loads_pyside6():
    proc = _run_hermetic(
        """
        import json
        import drone_fly.harness.session, drone_fly.harness.runner, drone_fly.cli.main
        print(json.dumps({"heavy": loaded_heavy(),
                          "overlay_window": "drone_fly.overlay.window" in sys.modules}))
        """
    )
    res = _last_json(proc)
    assert res == {"heavy": [], "overlay_window": False}


def _imports(path: Path) -> list[tuple[str, bool]]:
    """``(module, top_level)`` for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {id(n) for n in tree.body}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, id(node) in top))
    return out


def test_only_window_py_imports_pyside6():
    importers = sorted(
        str(p.relative_to(SRC))
        for p in SRC.rglob("*.py")
        if any(m.split(".")[0] == "PySide6" for m, _ in _imports(p))
    )
    assert importers == [str(Path("overlay") / "window.py")]


def test_overlay_modules_never_import_heavy_backends_at_module_level():
    for path in [*OVERLAY_SRC.glob("*.py"), SRC / "harness" / "overlay.py"]:
        if path.name == "window.py":
            continue
        for mod, top in _imports(path):
            if top:
                assert mod.split(".")[0] not in HEAVY_MODULES, (path.name, mod)
                assert not mod.startswith("drone_fly.record"), (path.name, mod)  # pulls torch


def test_overlay_extra_and_dev_extra():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    overlay = extras["overlay"]
    assert any(re.match(r"PySide6(-Essentials)?\b", e) for e in overlay)
    assert all("win32" in e for e in overlay if e.startswith("PySide6"))
    for name, deps in extras.items():
        if name != "overlay":
            assert not any("pyside" in d.lower() for d in deps), name
    assert not any("pyside" in d.lower() for d in data["project"]["dependencies"])


def test_lockfile_knows_the_overlay_extra():
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert re.search(r'name = "pyside6-essentials"', lock, re.I)


def test_lazy_import_forwards_the_overlay_hint():
    try:
        lazy_import("no_such_toolkit_65", package="PySide6-Essentials", hint=OVERLAY_INSTALL_HINT)
    except HarnessDependencyError as exc:
        msg = str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected HarnessDependencyError")
    assert OVERLAY_INSTALL_HINT == "uv sync --extra overlay"
    assert "PySide6-Essentials" in msg and OVERLAY_INSTALL_HINT in msg
    assert INSTALL_HINT not in msg
    try:
        lazy_import("no_such_backend_65")
    except HarnessDependencyError as exc:
        assert INSTALL_HINT in str(exc)  # the harness extra stays the default hint
