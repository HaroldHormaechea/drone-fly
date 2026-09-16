"""Smoke tests: the package and its subpackages import cleanly.

These assert only that the scaffold wires together — they do NOT assert any RL
learning behavior (which is stochastic and slow). Real behavior is validated
empirically via the evaluate stage.
"""


def test_package_imports():
    import drone_fly

    assert drone_fly.__version__


def test_subpackages_import():
    import importlib

    for name in ("connectome", "controller", "env", "train", "evaluate", "cli"):
        importlib.import_module(f"drone_fly.{name}")
