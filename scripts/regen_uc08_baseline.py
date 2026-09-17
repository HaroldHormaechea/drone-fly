"""Regenerate the seed-42 baseline rollout fixture under the new UC-09 default course.

Why this exists (UC-09 AC6)
---------------------------
UC-09 changes two things that make the *old* ``tests/data/uc08_baseline_rollout.npz``
un-reproducible bit-for-bit:

1. gate passage switched from forward **plane-crossing** to **3D proximity** detection, and
2. the **default course is now 3 gates** (was a single scalar gate).

So the seed-42 golden trace is **regenerated** rather than preserved: the regression guard
in ``tests/test_env_contract.py`` becomes "the current default env @ seed 42 reproduces the
committed baseline" (same seed → identical rollout), not "matches a frozen pre-UC-09 trace".

What it does
------------
Loads the stored **action script** from the existing fixture (unchanged), rolls the current
default :class:`~drone_fly.env.config.EnvConfig` (3-gate course, randomization OFF) through
it at ``seed=42`` with the hermetic ``simple`` adapter, and re-saves ``env_trace`` (the
stacked observation trace, exactly as ``tests/test_env_contract.py::_rollout_obs`` builds
it). All other arrays already in the fixture (``actions``, ``ad_pos``) are preserved verbatim.

Determinism
-----------
The ``simple`` adapter is noise-free and the default course fixes the geometry, so a given
seed yields a bit-identical trace — re-running this script is idempotent.

Usage
-----
    uv run python scripts/regen_uc08_baseline.py            # overwrite the committed fixture
    uv run python scripts/regen_uc08_baseline.py --out /tmp/x.npz   # dry-run elsewhere

**Commit note (AC6):** when you commit the regenerated fixture, call the regeneration out
explicitly in the message so it is not mistaken for an accidental regression.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drone_fly.env import EnvConfig, make_env

#: Committed fixture path (relative to the repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = _REPO_ROOT / "tests" / "data" / "uc08_baseline_rollout.npz"

#: The seed the baseline guard reproduces.
BASELINE_SEED = 42


def rollout_obs(config: EnvConfig, seed: int, actions: np.ndarray) -> np.ndarray:
    """Return the stacked observation trace for ``actions`` — mirrors the test's ``_rollout_obs``.

    Starts from ``reset(seed=seed)``'s observation, then appends the observation after each
    action, stopping early on termination/truncation. Uses the hermetic ``simple`` adapter.
    """
    env = make_env(config, adapter="simple")
    obs, _ = env.reset(seed=seed)
    trace = [obs.copy()]
    for a in actions:
        obs, _r, terminated, truncated, _info = env.step(np.asarray(a, dtype=np.float32))
        trace.append(obs.copy())
        if terminated or truncated:
            break
    env.close()
    return np.array(trace, dtype=np.float32)


def regenerate(fixture: Path, out: Path | None = None) -> Path:
    """Regenerate ``env_trace`` in ``fixture`` and write to ``out`` (defaults to ``fixture``)."""
    if not fixture.exists():
        raise FileNotFoundError(
            f"Baseline fixture not found at {fixture}; cannot read the stored action script."
        )
    existing = dict(np.load(fixture))
    if "actions" not in existing:
        raise KeyError(f"{fixture} has no 'actions' array to replay.")

    actions = existing["actions"]
    env_trace = rollout_obs(EnvConfig(), seed=BASELINE_SEED, actions=actions)

    # Preserve every other array (e.g. actions, ad_pos) verbatim; only env_trace is refreshed.
    payload = dict(existing)
    payload["env_trace"] = env_trace

    target = out or fixture
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **payload)
    print(
        f"Regenerated baseline: {target} "
        f"(env_trace {env_trace.shape}, from {len(actions)} stored actions, seed {BASELINE_SEED})"
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to the existing fixture to read the action script from.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where to write the regenerated fixture (defaults to --fixture, overwriting it).",
    )
    args = parser.parse_args()
    regenerate(args.fixture, args.out)


if __name__ == "__main__":
    main()
