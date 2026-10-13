"""Optional Liftoff scene / props extraction (UC68 D4), ported from liftoff-flight-analyzer.

Ported (not depended on) from https://github.com/HaroldHormaechea/liftoff-flight-analyzer at
commit d7250d5 (MIT License, Copyright (c) 2026 Harold Hormaechea); each file names its source.

* :mod:`.unityfs`     UnityFS node-table sniffing (pure Python LZ4, C strings)
* :mod:`.transforms`  quaternion maths, TRS composition, point-in-collider, witness test
* :mod:`.scene`       scene-bundle discovery, ``read_scene`` and identification
* :mod:`.props`       the ``props.json`` builder and the gate ``opening()`` measurement
* :mod:`.game`        the Liftoff install and its Steam build id

``UnityPy`` (the ``liftoff-extract`` extra) is imported lazily, only inside the functions that
read bundles; importing this package never imports it. Reading the cached JSON needs nothing.
"""
