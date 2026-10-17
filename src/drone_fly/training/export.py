"""``drone-fly train export <run> [--checkpoint val|final|it<N>] --out <path> [--baseline-export]``.

Writes the deployable student (format ``drone-fly-student`` v1, :mod:`drone_fly.student.checkpoint`)
that ``harness run --checkpoint`` loads. The **no-swap rule**: the deployable export is always the
K1 student; an ``mlp-student`` run is refused (:class:`~drone_fly.harness.errors.ExportError`)
unless ``--baseline-export``, which writes ``deployable: false``, ``role: baseline`` (and the
harness then refuses it without ``--allow-baseline``).

**Latency bench** (``--bench``, CUDA only): batch-1 mean action over 200 passes after a warm-up;
``latency: {device, p50_ms, p99_ms, budget_ms: 10, pass}``. A failure only warns; without CUDA
(or with ``--no-bench``) ``latency`` is null with the reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import ExportError

log = logging.getLogger(__name__)

BENCH_PASSES = 200


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_student_checkpoint(run: str | Path, which: str = "val") -> Path:
    from drone_fly.training.distill import SELECTION_FILE, STUDENT_NAME

    r = Path(run)
    if r.is_file():
        return r
    ck = r / "checkpoints"
    if which == "final":
        p = ck / STUDENT_NAME
    elif which == "val":
        sel = r / "eval" / SELECTION_FILE
        if not sel.is_file():
            raise ExportError(f"{r}: no val selection ({sel.name}); use --checkpoint final")
        p = ck / json.loads(sel.read_text(encoding="utf-8"))["checkpoint"]
    elif which.startswith("it"):
        p = ck / f"student_{which}.pt"
    else:
        raise ExportError(f"--checkpoint must be val, final or it<N>, got {which!r}")
    if not p.is_file():
        raise ExportError(f"student checkpoint {p} not found")
    return p


def k1_provenance(k1_path: Any = None) -> dict[str, Any]:
    from drone_fly.controller.reservoir import resolve_k1

    try:
        root = resolve_k1(k1_path)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal for a baseline
        return {"path": None, "error": str(exc)}
    return {
        "path": str(root),
        "prune_provenance_sha256": _sha(root / "PRUNE_PROVENANCE.md"),
        "connectome_sha256": _sha(root / "connectome_pruned.npz"),
    }


def bench_latency(actor: Any, obs_dim: int, device: str = "cuda") -> dict[str, Any]:
    import torch

    from drone_fly.student.checkpoint import LATENCY_BUDGET_MS

    actor = actor.to(device).eval()
    x = torch.zeros(1, obs_dim, device=device)
    with torch.no_grad():
        for _ in range(10):
            actor.mean_action(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(BENCH_PASSES):
            t0 = time.perf_counter()
            actor.mean_action(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    p99 = float(np.percentile(a, 99))
    out = {"device": torch.cuda.get_device_name(0), "passes": BENCH_PASSES,
           "p50_ms": float(np.percentile(a, 50)), "p99_ms": p99,
           "budget_ms": LATENCY_BUDGET_MS, "pass": p99 <= LATENCY_BUDGET_MS}  # fmt: skip
    if not out["pass"]:
        log.warning("student latency p99 %.1f ms exceeds the %.0f ms budget (not blocking)",
                    p99, LATENCY_BUDGET_MS)  # fmt: skip
    return out


def export_student(
    run: str | Path,
    out: str | Path,
    *,
    checkpoint: str = "val",
    baseline_export: bool = False,
    bench: bool | None = None,
    k1_path: Any = None,
) -> dict[str, Any]:
    """Export a student run (see the module docstring); returns the written meta."""
    import torch

    from drone_fly.controller.student import student_from_state
    from drone_fly.sim.action import read_sidecar
    from drone_fly.student.checkpoint import save_export, validate_meta

    ck = resolve_student_checkpoint(run, checkpoint)
    side = read_sidecar(ck)
    if side is None:
        raise ExportError(f"{ck}: no .contract.json sidecar (not a student checkpoint)")
    actor_kind = side.get("actor")
    if actor_kind not in ("k1-student", "mlp-student"):
        raise ExportError(f"{ck}: actor {actor_kind!r} is not a student")
    privileged = side.get("privileged")
    if privileged:  # UC79: unconditional (fires even with --baseline-export) — AC4
        raise ExportError(
            f"{ck}: diagnostic privileged student, not deployable "
            f"(privileged blocks {list(privileged)})"
        )
    if actor_kind == "mlp-student" and not baseline_export:
        raise ExportError(
            f"{ck} is the MLP comparison baseline: the deployable export is always the K1 "
            "student (pass --baseline-export to write it labelled deployable: false)"
        )
    state = torch.load(ck, map_location="cpu", weights_only=True)
    deployable = actor_kind == "k1-student"
    meta = dict(side)
    meta.update(
        deployable=deployable,
        role="deployable" if deployable else "baseline",
        source_checkpoint=str(ck),
        source_sha256=_sha(ck),
        selection=checkpoint,
        k1=k1_provenance(k1_path) if deployable else None,
        requires={
            "gate_estimator": "learned",
            "detector_sha256": (side.get("noise_model") or {}).get("detector_sha256"),
            "free_space_estimator": "none",
            "assume_no_obstacles": True,
            "camera": side.get("camera"),
        },
        exported_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    meta.pop("contention", None)
    validate_meta(meta, ck)
    if bench is None:
        bench = torch.cuda.is_available()
    if bench:
        if not torch.cuda.is_available():
            raise ExportError("--bench needs CUDA")
        actor = student_from_state(actor_kind, side.get("actor_arch") or {}, state,
                                   k1_path=k1_path)  # fmt: skip
        meta["latency"] = bench_latency(actor, int(meta["obs_schema"]["dim"]))
    else:
        meta["latency"] = None
        meta["latency_note"] = "not benchmarked (no CUDA or --no-bench); rerun with --bench"
    save_export(out, state, meta)
    return meta
