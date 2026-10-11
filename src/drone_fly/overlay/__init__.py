"""Live brain + debug overlay for the live harness (UC-65): ``drone-fly harness overlay``.

A separate process that floats a click-through, non-activating, capture-excluded window over the
game and draws what the runner publishes on a localhost UDP stream (see
:mod:`drone_fly.harness.overlay` for the publisher and :mod:`drone_fly.overlay.protocol` for the
wire format).

Every module here is toolkit-free and torch-free at import — packet decoding, latest-wins state,
replay, the brain heatmap, panel models, layouts and the Win32 shim are plain Python/NumPy and
unit-tested without a display. Only :mod:`drone_fly.overlay.window` imports the GUI toolkit
(PySide6, the ``overlay`` extra), and :mod:`drone_fly.overlay.app` imports it lazily.
"""
