"""System identification of Liftoff airframes (UC-66): fly scripted manoeuvres, fit, report.

* :mod:`~drone_fly.sysid.schedule`   the seeded experiment schedule (last pass held out)
* :mod:`~drone_fly.sysid.session`    ``drone-fly sysid liftoff``: the stabilized sequencer
* :mod:`~drone_fly.sysid.fit`        ``drone-fly sysid fit``: the parameter fit + validation
* :mod:`~drone_fly.sysid.report`     ``drones/<id>/drone.yaml`` + ``sysid.md`` + PNGs
* :mod:`~drone_fly.sysid.synthetic`  a known-parameter plant for dry runs and tests

Raw sessions live under ``artifacts/sysid/<drone-id>/<session>/`` (gitignored).
"""
