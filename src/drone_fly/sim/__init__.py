"""The profile-driven batched simulator (UC67), promoted from ``gpu_prototype/``.

* :mod:`~drone_fly.sim.plant`          :class:`Plant`: thrust curve, Betaflight rates, delay line,
  motor lag, drag, ground contact; parameterised by a ``drones/<id>`` profile
* :mod:`~drone_fly.sim.randomization`  per-episode dynamics samples from the profile's ranges
* :mod:`~drone_fly.sim.timing`         control-rate / latency / skipped-tick randomization
* :mod:`~drone_fly.sim.action`         the v2 action contract (hover-relative collective) + v1
* :mod:`~drone_fly.sim.course`         :class:`~drone_fly.sim.course.CourseEnv`, the procedural
  gate course (port of ``gates_proc_env.BatchedProcCourse``)
* :mod:`~drone_fly.sim.course_gen`     the procedural course generator (verbatim port)
* :mod:`~drone_fly.sim.replay`         open-loop replay of a sysid session through the plant

Torch only; no module-level device or tensors, no environment variables, and no import of
``drone_fly.harness``. Not to be confused with ``drone_fly.env`` (the pybullet-era config
surface still used by ``record/``).
"""

from drone_fly.sim.action import CONTRACT_VERSION, ActionSpec
from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.randomization import DynamicsRandomizer, DynamicsSamples
from drone_fly.sim.timing import ControlTiming, TimingConfig

__all__ = [
    "CONTRACT_VERSION",
    "ActionSpec",
    "ControlTiming",
    "DynamicsRandomizer",
    "DynamicsSamples",
    "Plant",
    "PlantConfig",
    "TimingConfig",
]
