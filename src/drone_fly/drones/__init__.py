"""Drone profiles: one committed ``drones/<id>/drone.yaml`` per airframe (UC-66).

``load(id)`` validates a profile and returns a :class:`DroneProfile`; every error is a
:class:`DroneProfileError` naming the offending field. A leaf package: YAML + NumPy only.
"""

from drone_fly.drones.schema import DroneProfile, DroneProfileError
from drone_fly.drones.store import dump, list_ids, load, profile_path

__all__ = ["DroneProfile", "DroneProfileError", "dump", "list_ids", "load", "profile_path"]
