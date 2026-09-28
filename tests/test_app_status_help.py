"""UC-61 status-view polish (items 5/6) — the backend-owned live-status field descriptors.

Hermetic (no torch/SB3/pybullet at import; the completeness test drives the real
``StatusEmitterCallback`` with fake SB3 objects — same technique as ``test_status_emitter.py``).
Pins the descriptor contract the grouped renderer relies on:

* **Well-formedness** — every descriptor is EITHER path-backed OR computed (never both, never
  neither); ``format.type`` is drawn from the closed vocabulary the JS renderer knows; ``key`` /
  ``label`` / merged ``help`` are all present and non-empty; ``STATUS_HELP`` carries no empty prose.
* **Completeness (the drift guard)** — every scalar leaf a *real* emitted ``status.jsonl`` record
  carries (top-level scalars + ``rollout.*`` + ``train.*`` + ``dynamics.*``) has a matching
  path-backed descriptor. A new emitter leaf without a descriptor fails here rather than silently
  shipping an unrendered metric.
"""

from __future__ import annotations

import json
import types

from app import configs_io, field_help

# The closed set the JS renderer understands. A backend that emits a type outside this set would
# render as a blank/garbage box, so the descriptor layer must never introduce one.
_ALLOWED_FORMAT_TYPES = {"float", "int", "seconds", "duration", "badge", "text"}


# --- well-formedness ----------------------------------------------------------------------


def test_describe_status_fields_are_well_formed():
    fields = configs_io.describe_status_fields()
    assert isinstance(fields, list) and fields

    seen_keys: set[str] = set()
    for f in fields:
        key = f.get("key")
        assert isinstance(key, str) and key.strip(), f
        assert key not in seen_keys, f"duplicate descriptor key {key!r}"
        seen_keys.add(key)

        assert isinstance(f.get("label"), str) and f["label"].strip(), f
        assert f.get("group") in field_help.STATUS_GROUP_ORDER, f

        # help is merged in from STATUS_HELP; every box must have non-empty prose for its popup.
        assert isinstance(f.get("help"), str) and f["help"].strip(), f

        # EXACTLY ONE of path-backed / computed — so the computed ETA never false-fails a
        # path-based check and a path-backed field is never also flagged computed.
        has_path = "path" in f
        is_computed = f.get("computed") is True
        assert has_path != is_computed, f"descriptor must be path XOR computed: {f}"
        if has_path:
            assert isinstance(f["path"], list) and f["path"], f
            assert all(isinstance(p, str) and p for p in f["path"]), f

        fmt = f.get("format")
        assert isinstance(fmt, dict), f
        assert fmt.get("type") in _ALLOWED_FORMAT_TYPES, f


def test_every_group_used_is_declared_in_order():
    fields = configs_io.describe_status_fields()
    used = {f["group"] for f in fields}
    assert used <= set(field_help.STATUS_GROUP_ORDER)


def test_status_help_has_no_empty_prose():
    assert field_help.STATUS_HELP  # non-empty map
    for key, entry in field_help.STATUS_HELP.items():
        assert isinstance(entry, dict), key
        assert isinstance(entry.get("help"), str) and entry["help"].strip(), key


def test_computed_descriptors_have_no_path_and_path_descriptors_are_not_computed():
    # Restates the XOR at the collection level so a regression in either direction is unambiguous.
    fields = configs_io.describe_status_fields()
    computed = [f for f in fields if f.get("computed") is True]
    path_backed = [f for f in fields if "path" in f]
    assert computed, "expected at least the ETA computed descriptor"
    assert all("path" not in f for f in computed)
    assert all(f.get("computed") is not True for f in path_backed)
    # eta is the concrete computed descriptor the ETA box relies on.
    assert any(f["key"] == "eta" for f in computed)


# --- completeness derived from a REAL emitted record (drift guard) -------------------------


class _FakeVecEnv:
    """Minimal VecEnv stand-in exposing the attrs the dynamics summary reads (env-0)."""

    def __init__(self, *, backend="pybullet", dynamics=None, spawn_z=0.5):
        self._backend = backend
        self._dynamics = dynamics
        self._spawn_z = spawn_z

    def get_attr(self, name):
        if name == "backend":
            return [self._backend]
        if name == "active_dynamics":
            return [self._dynamics]
        if name == "spawn_z":
            return [self._spawn_z]
        raise AttributeError(name)


def _emit_one_real_record(tmp_path) -> dict:
    """Drive the real StatusEmitterCallback once and return the parsed status.jsonl record.

    Uses fake SB3 objects (no PPO/env/GPU) but the genuine ``_build_record`` path, so the leaf
    keys are exactly what a live run emits — including a fully-populated ``health`` and
    ``dynamics`` block so every leaf in those groups is exercised.
    """
    from drone_fly.env.config import DynamicsParams
    from drone_fly.train.status_emitter import StatusEmitterCallback

    path = str(tmp_path / "status.jsonl")
    venv = _FakeVecEnv(backend="pybullet", dynamics=DynamicsParams(), spawn_z=0.5)
    logger = types.SimpleNamespace(
        name_to_value={
            "train/loss": 12.0,
            "train/value_loss": 51.2,
            "train/approx_kl": 0.007,
            "train/entropy_loss": -2.5,
            "train/explained_variance": 0.63,
            "train/std": 0.5,
            "time/fps": 480,
        }
    )
    model = types.SimpleNamespace(
        ep_info_buffer=[{"r": -100.0, "l": 200}, {"r": -300.0, "l": 220}],
        ep_success_buffer=[0.0, 1.0],
        logger=logger,
        _total_timesteps=100_000,
        get_env=(lambda: venv),
    )
    health = types.SimpleNamespace(
        latest_verdict=types.SimpleNamespace(status="normal", message="all clear")
    )
    cb = StatusEmitterCallback(path, health_cb=health, tw_preserving=True)
    cb.model = model
    cb.num_timesteps = 2048
    cb._on_training_start()
    cb._on_rollout_end()

    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(ln) for ln in fh.read().splitlines() if ln.strip()]
    assert lines, "emitter wrote no status line"
    return lines[-1]


def _leaf_paths(record: dict) -> set[tuple[str, ...]]:
    """Every scalar-leaf path in the emitted record, as tuples (dive one level into sub-dicts)."""
    paths: set[tuple[str, ...]] = set()
    for key, val in record.items():
        if isinstance(val, dict):
            for sub in val:
                paths.add((key, sub))
        else:
            paths.add((key,))
    return paths


def test_every_emitted_leaf_has_a_descriptor(tmp_path):
    record = _emit_one_real_record(tmp_path)
    assert record["dynamics"] is not None and record["health"] is not None  # groups populated

    descriptor_paths = {
        tuple(f["path"]) for f in configs_io.describe_status_fields() if "path" in f
    }

    # Leaves the descriptor layer deliberately does NOT surface as its own box:
    #  - ("line",): the emitter's status-cursor bookkeeping, not a metric (excluded if present).
    #  - ("health", "message"): shown inside the Health info popup, not as a standalone box.
    # ("health", "status") IS path-backed (the badge) and must be covered.
    excluded = {("line",), ("health", "message")}

    emitted = _leaf_paths(record) - excluded
    missing = {p for p in emitted if p not in descriptor_paths}
    assert not missing, (
        "emitted status leaves without a matching descriptor path (emitter drift — add a "
        f"STATUS_FIELDS descriptor): {sorted(missing)}"
    )


def test_every_dynamics_leaf_is_covered(tmp_path):
    # Focused check on the Dynamics block (the UC-47/48/49 provenance surface): every key the
    # dynamics summary emits must have a descriptor, so a new dynamics field can't ship unrendered.
    record = _emit_one_real_record(tmp_path)
    descriptor_dyn_keys = {
        f["path"][1]
        for f in configs_io.describe_status_fields()
        if f.get("path", [None])[0] == "dynamics"
    }
    for leaf in record["dynamics"]:
        assert leaf in descriptor_dyn_keys, f"dynamics leaf {leaf!r} has no descriptor"
