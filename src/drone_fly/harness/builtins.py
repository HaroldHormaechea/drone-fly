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
