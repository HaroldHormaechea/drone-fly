"""Vision "eyes" (UC-78): a learned camera estimator of the 70-input connectome's 40 obstacle rays.

The estimator reads the last ``K`` Liftoff frames plus telemetry ego-motion and outputs, per ray of
the training raycast cone (``inference.raycast``: 40 Fibonacci rays, +-45 deg around the velocity
heading, ``RAY_RANGE`` 4 m), a normalised free distance and a confidence.

Deployable (vision-only; numpy, torch, :mod:`drone_fly.harness.geometry`,
:mod:`drone_fly.harness.types` and each other -- never ``drone_fly.sim``, tracks, the oracle, the
contract or any offline module):

* :mod:`~drone_fly.eyes.camera_rays` -- ray basis, camera projection, in-view flags, frame prep;
* :mod:`~drone_fly.eyes.history` -- time-stride history frame selection;
* :mod:`~drone_fly.eyes.ego` -- the 10-value ego-motion vector;
* :mod:`~drone_fly.eyes.model` -- :class:`~drone_fly.eyes.model.EyesNet`;
* :mod:`~drone_fly.eyes.infer` -- :class:`~drone_fly.eyes.infer.EyesEstimator`.

Offline (labels, data, training, reports): :mod:`~drone_fly.eyes.capture_meta`,
:mod:`~drone_fly.eyes.fov`, :mod:`~drone_fly.eyes.labels`, :mod:`~drone_fly.eyes.dataset`,
:mod:`~drone_fly.eyes.train`, :mod:`~drone_fly.eyes.evaluate`, :mod:`~drone_fly.eyes.bench` and
the ``drone-fly eyes`` CLI (:mod:`~drone_fly.eyes.cli`). See
``docs/experiments/uc78-vision-eyes.md``.

Importing this package imports nothing.
"""
