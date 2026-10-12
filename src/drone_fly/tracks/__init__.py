"""The track library (UC68): catalogues of courses and their conversion to the simulator.

Committed catalogues live in the top-level ``tracks/`` folder (``liftoff.yaml``,
``procedural.yaml``; schema in ``tracks/README.md``). Game data (XMLs, ``props.json``, scene
caches) stays gitignored under ``artifacts/harness/trackdata/``.

* :mod:`~drone_fly.tracks.catalogue`       schema, selectors, the by-map split, ``select`` + the
  leakage guard (``TrackLeakageError``)
* :mod:`~drone_fly.tracks.scan`            ``tracks scan``: pairing, conversion, stats, catalogue
* :mod:`~drone_fly.tracks.scenes`          scene caches: roots, build ids, validation, status
* :mod:`~drone_fly.tracks.liftoff`         Liftoff XML + props + scene -> ``SimCourse``
* :mod:`~drone_fly.tracks.procedural`      procedural specs -> ``SimCourse``
* :mod:`~drone_fly.tracks.loader`          ``load_course`` (both sources) and ``build_pool``
* :mod:`~drone_fly.tracks.teacher_obs_ref` the NumPy single-drone teacher observation
* :mod:`~drone_fly.tracks.liftoff_extract` optional UnityPy scene / props extraction (port)
* :mod:`~drone_fly.tracks.extract`         ``tracks extract-scenes`` / ``extract-props``
* :mod:`~drone_fly.tracks.cli`             ``drone-fly tracks ...``

Importing the package is light (no torch, no UnityPy).
"""
