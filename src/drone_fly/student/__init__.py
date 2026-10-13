"""The training <-> harness contract of the UC70 deployable student.

* :mod:`~drone_fly.student.perception_ref` — the NumPy single-drone reference of the simulated
  perception (``sim.perception``), driven by explicit draws; returns a harness ``GateEstimate``.
* :mod:`~drone_fly.student.obs_ref` — ``build_student_obs``: the ``student-obs`` v1 vector from a
  ``GateEstimate`` through the REAL ``ContractObservationAssembler`` (the harness's
  ``StudentObservationAssembler`` calls the same function: bit-identical by construction).
* :mod:`~drone_fly.student.checkpoint` — the export format, its constants and the version guard.

``drone_fly.sim`` never imports this package (or ``drone_fly.harness``); this package may import
both.
"""
