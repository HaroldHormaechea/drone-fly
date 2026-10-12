"""Training on the batched simulator (UC67), promoted from ``gpu_prototype/``.

* :mod:`~drone_fly.training.config`   :class:`TrainConfig` + the YAML loader (packaged configs
  ``meteor75-legacy`` and ``proc-course-dr`` under ``configs/``)
* :mod:`~drone_fly.training.ppo`      PPO for the frozen-K1 reservoir actor on ``CourseEnv``, and
  the UC68 teacher loop (MLP actor-critic on ``MultiCourseEnv``, resume, STOP, per-group logs)
* :mod:`~drone_fly.training.run_dir`  ``artifacts/training/<drone-id>/<run-id>/``
* :mod:`~drone_fly.training.evaluate` the UC68 per-track teacher report (``train eval --tracks``)
* :mod:`~drone_fly.training.cli`      ``drone-fly train run | teacher | eval | bench``

Importing this package is light (no torch): ``drone-fly --help`` builds the ``train`` parser
from :mod:`~drone_fly.training.cli` without loading the simulator.
"""
