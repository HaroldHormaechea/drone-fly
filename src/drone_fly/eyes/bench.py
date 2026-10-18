"""AC7: estimator latency (``drone-fly eyes bench``).

Times :meth:`~drone_fly.eyes.infer.EyesEstimator.update` on synthetic full-resolution frames
(the preset's camera, 2560x1080 for ``liftoff``) moving at a racing speed, after ``warmup``
iterations: preprocessing, forward and total, as p50 / p95 / p99 in ms, against the live tick
budget (``1000 / runner.rate_hz`` = 25 ms at 40 Hz). With ``--with-gate-detector`` the learned gate
detector (``gates.learned.weights`` / ``release``) runs on the same frame each iteration
(interleaved, as in the live loop) and is timed too; missing weights skip it with a reason (no
download).

Peak CUDA memory (``max_memory_allocated`` / ``max_memory_reserved``) is reported for the eyes
alone and for eyes + detector; on CPU these fields are ``null`` with a reason. The connectome
policy's footprint is NOT measured here (follow-up: the three share the RTX 3050's 8 GB).

Writes ``bench.json`` + ``bench.md`` (default: next to the weights).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.types import FlightState, Frame

CONNECTOME_NOTE = "the connectome policy's GPU footprint is not measured here (follow-up)"


def _pct_ms(a: list[float]) -> dict[str, float | None]:
    if not a:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    x = np.asarray(a) * 1000.0
    return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "mean": float(x.mean())}  # fmt: skip


def _detector(view: Any, gate_weights: str | Path | None, device: str):
    """``(callable(image), info)`` or ``(None, {"skipped": reason})``."""
    from drone_fly.harness.config import find_repo_root
    from drone_fly.harness.liftoff.gate_learned import configured_weights

    lc = (view.config.get("gates") or {}).get("learned") or {}
    raw = gate_weights or configured_weights(lc)
    if not raw:
        return None, {"skipped": "no gates.learned.weights / release configured"}
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (find_repo_root() or Path.cwd()) / p
    if not p.is_file():
        return None, {"skipped": f"gate weights not found: {p}"}
    import torch

    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.liftoff.gate_learned import load_weights
    from drone_fly.harness.liftoff.gate_net import prepare_input

    table = GateTypeTable.from_config(view.config.get("gate_types"))
    net, meta = load_weights(p, table, device)
    width = int(lc.get("input_width", meta.get("input_width", 768)))
    fp16 = bool(lc.get("fp16", True)) and device.startswith("cuda")
    if fp16:
        net.half()

    def run(image: np.ndarray) -> None:
        with torch.no_grad():
            x, _ = prepare_input(image, width, device)
            net(x.half() if fp16 else x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    return run, {"weights": str(p), "input_width": width, "fp16": fp16}


def run_bench(
    weights: str | Path,
    *,
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    device: str = "auto",
    iters: int = 500,
    warmup: int = 20,
    with_gate_detector: bool = False,
    gate_weights: str | Path | None = None,
    budget_ms: float | None = None,
    out: str | Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    from drone_fly.eyes.fov import preset_view
    from drone_fly.eyes.infer import EyesEstimator, resolve_device

    view = preset_view(preset, config, harness)
    dev = resolve_device(device)
    cuda = dev.startswith("cuda")
    rate = float((view.config.get("runner") or {}).get("rate_hz", 40.0))
    budget = float(budget_ms if budget_ms is not None else 1000.0 / rate)
    rng = np.random.default_rng(seed)
    w, h = view.camera.width, view.camera.height
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(4)]
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    est = EyesEstimator.from_weights(weights, dev)
    dt = float(est.dt_s) / 2.0  # several frames per stride: history is always available
    det, det_info = (None, {"skipped": "not requested"})
    timings: dict[str, list[float]] = {"preprocess": [], "forward": [], "total": [], "detector": [],
                                       "tick": []}  # fmt: skip
    mem: dict[str, Any] = {}
    no_mem = {"max_allocated_mb": None, "max_reserved_mb": None, "reason": "CPU run (no CUDA)"}

    tick = [0]

    def loop(n: int, record: bool, with_det: bool) -> None:
        for _ in range(n):
            t = tick[0]
            tick[0] += 1
            st = FlightState(pos=np.array([0.1 * t, 0, 1], np.float32),
                             vel=np.array([5.0, 0, 0], np.float32),
                             quat=np.array([0.98, 0, 0.17, 0], np.float32),
                             omega=np.zeros(3, np.float32))  # fmt: skip
            img = images[t % len(images)]
            t0 = time.perf_counter()
            o = est.update(Frame(img, t * dt, t), st, None)
            t1 = time.perf_counter()
            if with_det and det is not None:
                det(img)
            t2 = time.perf_counter()
            if record:
                timings["preprocess"].append(o.timings_s.get("preprocess_s", 0.0))
                timings["forward"].append(o.timings_s.get("forward_s", 0.0))
                timings["total"].append(t1 - t0)
                if with_det and det is not None:
                    timings["detector"].append(t2 - t1)
                timings["tick"].append(t2 - t0)

    loop(warmup, False, False)
    loop(iters, True, False)
    eyes_ms = {k: _pct_ms(timings[k]) for k in ("preprocess", "forward", "total")}
    mem["eyes"] = ({"max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                    "max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
                   if cuda else dict(no_mem))  # fmt: skip
    combined = None
    if with_gate_detector:
        det, det_info = _detector(view, gate_weights, dev)
        if det is not None:
            timings = {k: [] for k in timings}
            loop(warmup, False, True)
            loop(iters, True, True)
            combined = {k: _pct_ms(timings[k]) for k in ("total", "detector", "tick")}
            mem["eyes_plus_detector"] = (
                {"max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                 "max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}
                if cuda else dict(no_mem)
            )  # fmt: skip
    if "eyes_plus_detector" not in mem:
        mem["eyes_plus_detector"] = {**no_mem, "reason": det_info.get("skipped", no_mem["reason"])}
    p95 = eyes_ms["total"]["p95"]
    rep = {
        "format": 1,
        "weights": str(Path(weights).resolve()),
        "device": dev,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "frame": [w, h],
        "input": est.meta["input"],
        "iters": int(iters),
        "warmup": int(warmup),
        "budget_ms": budget,
        # UC-85: the frozen depth backbone (if any) is timed inside the eyes forward via
        # EyesEstimator.update, so eyes_ms already includes it; recorded here for the AC5 doc.
        "depth_backbone": (est.meta.get("arch") or {}).get("depth_backbone", False),
        "backbone_id": (est.meta.get("arch") or {}).get("backbone_id"),
        "eyes_ms": eyes_ms,
        "eyes_p95_within_budget": None if p95 is None else bool(p95 <= budget),
        "detector": det_info,
        "eyes_plus_detector_ms": combined,
        "combined_p95_within_budget": None
        if combined is None
        else bool(combined["tick"]["p95"] <= budget),
        "memory": mem,
        "note": CONNECTOME_NOTE,
    }
    out_dir = Path(out) if out is not None else Path(weights).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / "bench.md").write_text(bench_markdown(rep), encoding="utf-8")
    rep["out_dir"] = str(out_dir)
    return rep


def _f(v: Any) -> str:
    return "-" if v is None else f"{v:.2f}"


def bench_markdown(rep: dict[str, Any]) -> str:
    lines = [
        "# Eyes latency (UC-78 AC7)",
        "",
        f"Device `{rep['device']}` ({rep['gpu'] or 'CPU'}), frames {rep['frame'][0]}x"
        f"{rep['frame'][1]}, {rep['iters']} iterations after {rep['warmup']} warmup; budget "
        f"{rep['budget_ms']:.1f} ms.",
        "",
        f"Depth backbone (UC-85): `{rep.get('backbone_id') or rep.get('depth_backbone')}`."
        if rep.get("depth_backbone")
        else "Depth backbone (UC-85): none (eyes v1/v2 CNN trunk).",
        "",
        "| stage | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|",
    ]
    for k, v in rep["eyes_ms"].items():
        lines.append(f"| eyes {k} | {_f(v['p50'])} | {_f(v['p95'])} | {_f(v['p99'])} |")
    if rep["eyes_plus_detector_ms"]:
        for k, v in rep["eyes_plus_detector_ms"].items():
            lines.append(f"| +detector {k} | {_f(v['p50'])} | {_f(v['p95'])} | {_f(v['p99'])} |")
    else:
        lines.append(f"| gate detector | skipped: {rep['detector'].get('skipped')} | | |")
    lines += ["", f"Eyes p95 within budget: {rep['eyes_p95_within_budget']}; with the detector: "
              f"{rep['combined_p95_within_budget']}.", "", "| memory | max allocated MB | "
              "max reserved MB | note |", "|---|---:|---:|---|"]  # fmt: skip
    for k, v in rep["memory"].items():
        lines.append(f"| {k} | {_f(v['max_allocated_mb'])} | {_f(v['max_reserved_mb'])} | "
                     f"{v.get('reason') or ''} |")  # fmt: skip
    lines += ["", f"Note: {rep['note']}.", ""]
    return "\n".join(lines)
