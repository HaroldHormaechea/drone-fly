"""Registration of the built-in harnesses and parts (lazy ``module:attr`` targets only).

Importing this module imports none of the backends; :func:`registry.resolve` does that on
demand. ``requires`` drives the availability column of ``drone-fly harness list``.
"""

from __future__ import annotations

from drone_fly.harness.registry import register

_PKG = "drone_fly.harness"
_CV2 = {"cv2": "opencv-python"}

_BUILTINS: list[tuple[str, str, str, dict]] = [
    # --- harnesses (factories) ---
    (
        "harness",
        "sim-loopback",
        f"{_PKG}.loopback:build_loopback_harness",
        {
            "description": "contract reference plant driven through the harness interfaces",
            "default_config": "loopback.yaml",
        },
    ),
    # --- sim-loopback parts ---
    (
        "state_source",
        "plant",
        f"{_PKG}.loopback:PlantStateSource",
        {"description": "state of the contract reference plant"},
    ),
    (
        "gate_estimator",
        "course",
        f"{_PKG}.loopback:CourseGateEstimator",
        {"description": "ground-truth gate senses from known course geometry"},
    ),
    (
        "free_space_estimator",
        "raycast",
        f"{_PKG}.loopback:RaycastFreeSpaceEstimator",
        {"description": "ground-truth rays via the contract raycast() on known obstacles"},
    ),
    (
        "action_sink",
        "plant",
        f"{_PKG}.loopback:PlantActionSink",
        {"description": "steps the contract reference plant_step()"},
    ),
    # --- Liftoff ---
    (
        "harness",
        "liftoff",
        f"{_PKG}.liftoff:build_liftoff_harness",
        {
            "description": "Liftoff FPV: UDP telemetry + screen vision + ViGEm pad",
            "default_config": "liftoff_5inch.yaml",
        },
    ),
    (
        "state_source",
        "liftoff-udp",
        f"{_PKG}.liftoff.telemetry:LiftoffUdpStateSource",
        {"description": "Liftoff UDP telemetry listener"},
    ),
    (
        "state_source",
        "liftoff-csv",
        f"{_PKG}.liftoff.telemetry:CsvReplayStateSource",
        {"description": "replays a recorded Liftoff telemetry CSV in real time"},
    ),
    (
        "frame_source",
        "screen",
        f"{_PKG}.liftoff.frames:ScreenFrameSource",
        {"description": "screen capture", "requires": ("dxcam|mss",)},
    ),
    (
        "gate_estimator",
        "learned",
        f"{_PKG}.liftoff.gate_learned:LearnedGateEstimator",
        {
            "description": "keypoint net (per gate type) + PnP",
            "requires": ("torch", "cv2"),
            "packages": _CV2,
        },
    ),
    (
        "gate_estimator",
        "opencv",
        f"{_PKG}.liftoff.gate_cv:OpenCvGateEstimator",
        {"description": "HSV quads + PnP gate pose", "requires": ("cv2",), "packages": _CV2},
    ),
    (
        "free_space_estimator",
        "none",
        f"{_PKG}.estimators:NoObstacleFreeSpaceEstimator",
        {"description": "obstacle-free world: zero obstacle_vision, never raycast"},
    ),
    (
        "free_space_estimator",
        "depth",
        f"{_PKG}.liftoff.depth:MonocularDepthFreeSpaceEstimator",
        {
            "description": "monocular depth net -> 40-ray scan + obstacle_vision",
            "requires": ("transformers", "torch", "PIL"),
            "packages": {"PIL": "pillow"},
        },
    ),
    (
        "depth_model",
        "depth-anything-v2",
        f"{_PKG}.liftoff.depth:DepthAnythingV2Model",
        {
            "description": "Depth-Anything-V2 metric (transformers)",
            "requires": ("transformers", "torch", "PIL"),
            "packages": {"PIL": "pillow"},
        },
    ),
    (
        "action_sink",
        "vgamepad",
        f"{_PKG}.liftoff.gamepad:VgamepadActionSink",
        {"description": "ViGEm virtual Xbox 360 pad", "requires": ("vgamepad",)},
    ),
    (
        "abort_signal",
        "hotkey",
        f"{_PKG}.liftoff.abort:HotkeyAbortSignal",
        {"description": "Win32 abort hotkey (GetAsyncKeyState)"},
    ),
    (
        "focus_probe",
        "win32-foreground",
        f"{_PKG}.liftoff.focus:Win32ForegroundFocusProbe",
        {"description": "Win32 foreground window == the game process"},
    ),
    (
        "action_sink",
        "recording",
        f"{_PKG}.runner:RecordingOnlySink",
        {"description": "dry run: records commands, actuates nothing"},
    ),
]


def register_builtins() -> None:
    for kind, name, target, kw in _BUILTINS:
        register(kind, name, target, replace=True, **kw)


register_builtins()
