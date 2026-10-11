"""Train the learned gate estimator's keypoint network on auto-labelled captures.

``drone-fly harness train-gates --labels <dir> [<dir> ...] [--out ...] [--preset ...]
[--epochs N] [--backbone resnet18-hf|tiny-cnn] [--holdout-frac 0.2 | --holdout t0:t1]
[--val-frac 0.1] [--guard-s 2] [--select best-val|last] [--patience 10]``

* Inputs: one or more label folders written by ``drone-fly harness label`` (``labels.jsonl`` +
  ``labels_meta.json``, which names the capture). Their gate-type hash must match the configured
  ``gate_types``.
* Images are masked (``camera.masks``) and resized exactly as at runtime
  (:func:`~drone_fly.harness.liftoff.gate_net.prepare_input` building blocks), then cached in
  memory. Augmentation: brightness / contrast / saturation jitter, blur, and horizontal flips
  that swap the gate type's ``flip_pairs``.
* **Three disjoint time slices** per capture (in-flight time; :func:`split_capture`):

  - *held out* (evaluation only): the last ``holdout_frac`` (20 %), or ``--holdout t0:t1``
    (seconds from the capture's first labelled frame). Never loaded for training or selection;
    ``drone-fly harness eval --capture`` scores only these frames and checks the disjointness.
  - *validation* (model selection): the ``val_frac`` (10 %) of in-flight time immediately before
    the held-out slice.
  - *train*: the rest.

  A ``guard_s`` (2 s) gap separates neighbouring slices against temporal leakage. All ranges are
  stored in the weights metadata (``splits``). With one capture, held-out laps revisit the same
  viewpoints: the evaluation is optimistic.
* AdamW + cosine schedule, mixed precision on CUDA. By default (``--select best-val``) the epoch
  with the lowest validation loss is kept, with early stopping after ``patience`` (10) epochs
  without improvement; ``--select last`` keeps the final epoch. ``training_log.json`` (per-epoch
  train and validation losses, chosen epoch) is written beside the weights.
  (Plan amendment during UC-64: the first run overfit the single capture, so model selection got
  its own validation slice rather than reusing the held-out evaluation frames.)
* ``resnet18-hf`` downloads ``microsoft/resnet-18`` once; if that fails the error names the model
  id and the ``tiny-cnn`` fallback.

Default output: ``artifacts/harness/liftoff/models/liftoff-gates.pt``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import find_repo_root, load_config
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.recording import git_head

log = logging.getLogger(__name__)

MAX_TRAIN_RANGE_M = 80.0
IN_FLIGHT_GAP_S = 1.0


@dataclass
class LabelSet:
    labels_dir: Path
    meta: dict[str, Any]
    rows: list[dict[str, Any]]

    @property
    def capture(self) -> Path:
        return Path(self.meta["capture"])

    @property
    def name(self) -> str:
        return str(self.meta.get("capture_name") or self.capture.name)


def load_labels(labels_dir: str | Path) -> LabelSet:
    d = Path(labels_dir)
    meta_path, rows_path = d / "labels_meta.json", d / "labels.jsonl"
    if not meta_path.is_file() or not rows_path.is_file():
        raise HarnessConfigError(
            f"{d} has no labels.jsonl / labels_meta.json (run `harness label`)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return LabelSet(d, meta, rows)


def in_flight_clock(stamps: np.ndarray, gap_s: float = IN_FLIGHT_GAP_S) -> np.ndarray:
    """Cumulative in-flight seconds at each (sorted) stamp; gaps longer than ``gap_s`` count 0."""
    dt = np.diff(np.asarray(stamps, np.float64), prepend=stamps[0] if len(stamps) else 0.0)
    dt[dt > gap_s] = 0.0
    return np.cumsum(dt)


def _ranges(stamps: np.ndarray) -> list[list[float]]:
    """Contiguous (in sorted order) stamp runs -> ``[[start, end], ...]``."""
    return [[float(stamps[0]), float(stamps[-1])]] if len(stamps) else []


def split_capture(
    rows: list[dict[str, Any]],
    holdout_frac: float = 0.2,
    val_frac: float = 0.1,
    guard_s: float = 2.0,
    window: tuple[float, float] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Assign each row to ``train`` / ``val`` / ``holdout`` / ``guard`` (see the module docstring).

    Returns the per-row slice names (input order) and the split record: stamp ranges per slice
    (absolute, inclusive; ``train`` may be two runs around a ``--holdout`` window) and the
    parameters.
    """
    if not rows:
        raise HarnessConfigError("a label set has no frames")
    if not 0.0 < holdout_frac < 1.0 or not 0.0 <= val_frac < 1.0 or guard_s < 0:
        raise HarnessConfigError("holdout_frac in (0, 1), val_frac in [0, 1), guard_s >= 0")
    stamps = np.array([r["stamp"] for r in rows], np.float64)
    order = np.argsort(stamps, kind="stable")
    st = stamps[order]
    clock = in_flight_clock(st)
    total = float(clock[-1])
    if window is not None:
        hold = (st >= st[0] + window[0]) & (st <= st[0] + window[1])
        if not hold.any():
            raise HarnessConfigError(f"--holdout {window} selects no frame")
        h0, h1 = float(clock[hold].min()), float(clock[hold].max())
        hold_rec = {"t_start": float(st[0] + window[0]), "t_end": float(st[0] + window[1])}
    else:
        h0, h1 = (1.0 - holdout_frac) * total, total
        hold = clock >= h0
        hold_rec = {"t_start": float(st[hold].min()), "t_end": float(st[-1])}
    v1 = h0 - guard_s
    v0 = h0 - val_frac * total
    val = (clock >= v0) & (clock <= v1) & ~hold if val_frac > 0 else np.zeros(len(st), bool)
    train = ((clock < v0 - guard_s) | (clock > h1 + guard_s)) & ~hold & ~val
    names_sorted = np.where(
        hold, "holdout", np.where(val, "val", np.where(train, "train", "guard"))
    )
    names = np.empty(len(rows), dtype=object)
    names[order] = names_sorted
    before = st[train & (clock < h0)]
    after = st[train & (clock > h1)]
    record = {
        "mode": "window" if window is not None else "last_fraction",
        "holdout_frac": None if window is not None else holdout_frac,
        "holdout_window_s": None if window is None else list(window),
        "val_frac": val_frac,
        "guard_s": guard_s,
        "in_flight_s": total,
        "holdout": hold_rec,
        "val": _ranges(st[val]),
        "train": _ranges(before) + _ranges(after),
        "counts": {k: int(np.sum(names_sorted == k)) for k in ("train", "val", "holdout", "guard")},
    }
    check_disjoint(record)
    return [str(n) for n in names], record


def check_disjoint(split: dict[str, Any]) -> None:
    """Raise unless the held-out range overlaps no train / validation range."""
    h0, h1 = split["holdout"]["t_start"], split["holdout"]["t_end"]
    for name in ("train", "val"):
        for a, b in split.get(name) or []:
            if a <= h1 and b >= h0:
                raise HarnessConfigError(
                    f"held-out range [{h0}, {h1}] overlaps the {name} range [{a}, {b}]"
                )


def is_held_out(stamp: float, window: dict[str, Any]) -> bool:
    return window["t_start"] <= stamp <= window["t_end"]


def parse_window(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    try:
        a, b = (float(v) for v in text.split(":"))
    except ValueError as exc:
        raise HarnessConfigError(f"--holdout must be t0:t1 seconds, got {text!r}") from exc
    if b <= a:
        raise HarnessConfigError(f"--holdout t1 must be > t0, got {text!r}")
    return a, b


def _train_gates(row: dict[str, Any], table: GateTypeTable, scale: float) -> list[dict[str, Any]]:
    """Label gates -> network targets in input pixels (centre in image, in range, unmasked)."""
    out = []
    for g in row["gates"]:
        if not g["center_in_image"] or g.get("center_masked") or g["range_m"] > MAX_TRAIN_RANGE_M:
            continue
        kp = np.asarray(g["keypoints"], np.float64)
        front = np.asarray(g["in_front"], bool)
        if not front.all():
            kp = kp.copy()
            kp[~front] = np.asarray(g["center_px"], np.float64)  # behind the camera: no extent
        out.append(
            {
                "cls": table.class_index(g["cls"]),
                "center": np.asarray(g["center_px"], np.float64) * scale,
                "keypoints": kp * scale,
                "visible": np.asarray(g["visible"], bool),
            }
        )
    return out


def _flip_gates(gates: list[dict[str, Any]], table: GateTypeTable, width: int) -> list[dict]:
    out = []
    for g in gates:
        perm = table.types[g["cls"]].flip_permutation()
        kp = g["keypoints"][perm].copy()
        kp[:, 0] = (width - 1) - kp[:, 0]
        c = g["center"].copy()
        c[0] = (width - 1) - c[0]
        out.append({"cls": g["cls"], "center": c, "keypoints": kp, "visible": g["visible"][perm]})
    return out


def _augment_colour(torch: Any, x: Any, rng: np.random.Generator) -> Any:
    """Per-image brightness / contrast / saturation jitter and occasional blur ([0, 255] RGB)."""
    import torch.nn.functional as F

    b = x.shape[0]
    dev = x.device
    bright = torch.tensor(rng.uniform(0.7, 1.3, b), device=dev, dtype=x.dtype).view(b, 1, 1, 1)
    contrast = torch.tensor(rng.uniform(0.7, 1.3, b), device=dev, dtype=x.dtype).view(b, 1, 1, 1)
    sat = torch.tensor(rng.uniform(0.7, 1.3, b), device=dev, dtype=x.dtype).view(b, 1, 1, 1)
    x = x * bright
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * sat + gray
    blur = rng.random(b) < 0.3
    if blur.any():
        idx = torch.tensor(np.flatnonzero(blur), device=dev)
        x[idx] = F.avg_pool2d(x[idx], 3, stride=1, padding=1, count_include_pad=False)
    return x.clamp(0, 255)


def train_gates(
    labels: list[str | Path],
    *,
    out: str | Path | None = None,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = None,
    harness: str = "liftoff",
    epochs: int = 60,
    backbone: str = "resnet18-hf",
    batch_size: int = 8,
    lr: float = 1e-3,
    holdout_frac: float = 0.2,
    holdout: str | None = None,
    val_frac: float = 0.1,
    guard_s: float = 2.0,
    select: str = "best-val",
    patience: int | None = 10,
    input_width: int | None = None,
    device: str = "auto",
    seed: int = 0,
) -> dict[str, Any]:
    """Train and write the weights file; returns its metadata (see the module docstring)."""
    torch = lazy_import("torch", purpose="gate-net training")
    from drone_fly.harness.liftoff.gate_net import (
        BACKBONES,
        STRIDE,
        GateKeypointNet,
        build_targets,
        gate_net_loss,
        input_geometry,
        normalize_pad,
        resize_images,
    )
    from drone_fly.harness.offline.evaluate import load_image

    if select not in ("best-val", "last"):
        raise HarnessConfigError(f"--select must be best-val or last, got {select!r}")
    if select == "best-val" and val_frac <= 0:
        raise HarnessConfigError("--select best-val needs a validation slice (--val-frac > 0)")
    if backbone not in BACKBONES:
        raise HarnessConfigError(f"--backbone must be one of {BACKBONES}, got {backbone!r}")
    cfg = load_config(harness, config, preset=preset)
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    masks = FrameMasks.from_config(cfg.section("camera"))
    lw = input_width or int((cfg.section("gates").get("learned") or {}).get("input_width", 768))
    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    window = parse_window(holdout)

    sets = [load_labels(d) for d in labels]
    for s in sets:
        if s.meta.get("gate_types_hash") != table.hash():
            raise HarnessConfigError(
                f"labels in {s.labels_dir} were made with different gate_types (hash "
                f"{s.meta.get('gate_types_hash')} != {table.hash()}); re-run `harness label`"
            )

    # -- load + preprocess every frame once (masked, resized as at runtime) --
    images, targets_raw, is_val = [], [], []
    geometry = None
    split_meta, captures = {}, []
    t_load = time.perf_counter()
    for s in sets:
        slices, record = split_capture(s.rows, holdout_frac, val_frac, guard_s, window)
        split_meta[s.name] = record
        for row, sl in zip(s.rows, slices, strict=True):
            if sl not in ("train", "val"):
                continue  # held-out and guard frames are never loaded
            img = load_image(s.capture / row["file"])
            h, w = img.shape[:2]
            g = input_geometry(w, h, lw)
            if geometry is None:
                geometry = g
            elif g != geometry:
                raise HarnessConfigError("all captures must share one frame size")
            if masks:
                img = masks.apply(img)
            x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1).flip(0)
            x = resize_images(x.unsqueeze(0), g[0], g[1])[0].round().clamp(0, 255).to(torch.uint8)
            images.append(x.cpu())
            targets_raw.append(_train_gates(row, table, g[3]))
            is_val.append(sl == "val")
        captures.append(
            {
                "name": s.name,
                "capture": str(s.capture),
                "labels_dir": str(s.labels_dir),
                **{f"n_{k}": v for k, v in record["counts"].items()},
            }
        )
    assert geometry is not None
    in_w, in_h, padded, _ = geometry
    out_h, out_w = padded // STRIDE, in_w // STRIDE
    val_arr = np.array(is_val)
    train_idx, val_idx = np.flatnonzero(~val_arr), np.flatnonzero(val_arr)
    if len(train_idx) == 0:
        raise HarnessConfigError("no training frames left after the time split")
    log.info(
        "loaded %d frames (%d train / %d validation; held-out frames not loaded) in %.0fs at %dx%d",
        len(images),
        len(train_idx),
        len(val_idx),
        time.perf_counter() - t_load,
        in_w,
        in_h,
    )
    data = torch.stack(images)  # (N, 3, in_h, in_w) uint8, CPU

    net = GateKeypointNet(
        len(table), table.max_keypoints, backbone, pretrained=backbone != "tiny-cnn"
    )
    net.to(dev)
    backbone_params = list(net.backbone.parameters())
    bb_ids = {id(p) for p in backbone_params}
    head_params = [p for p in net.parameters() if id(p) not in bb_ids]
    bb_lr = lr * (0.2 if backbone != "tiny-cnn" else 1.0)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": bb_lr}, {"params": head_params, "lr": lr}],
        weight_decay=1e-4,
    )
    steps_per_epoch = math.ceil(len(train_idx) / batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    )
    cuda = str(dev).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    def batch(idx: np.ndarray, augment: bool) -> tuple[Any, dict[str, Any]]:
        x = data[torch.as_tensor(idx)].to(dev).float()
        gates = [targets_raw[i] for i in idx]
        if augment:
            flip = rng.random(len(idx)) < 0.5
            if flip.any():
                fi = torch.as_tensor(np.flatnonzero(flip), device=dev)
                x[fi] = x[fi].flip(-1)
                gates = [
                    _flip_gates(g, table, in_w) if f else g
                    for g, f in zip(gates, flip, strict=True)
                ]
            x = _augment_colour(torch, x, rng)
        tg = [build_targets(g, out_h, out_w, len(table), table.max_keypoints) for g in gates]
        tt = {k: torch.as_tensor(np.stack([t[k] for t in tg])).to(dev) for k in tg[0]}
        return normalize_pad(x, padded), tt

    def evaluate() -> float | None:
        if len(val_idx) == 0:
            return None
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for k in range(0, len(val_idx), batch_size):
                idx = val_idx[k : k + batch_size]
                x, tt = batch(idx, augment=False)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    loss, _ = gate_net_loss(net(x), tt)
                total += float(loss) * len(idx)
                n += len(idx)
        net.train()
        return total / max(n, 1)

    history = []
    best: dict[str, Any] = {"epoch": None, "val_loss": math.inf, "state": None}
    stopped_early = False
    net.train()
    t_train = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(train_idx)
        sums: dict[str, float] = {}
        n_batches = 0
        for k in range(0, len(perm), batch_size):
            idx = perm[k : k + batch_size]
            x, tt = batch(idx, augment=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out_t = net(x)
            loss, parts = gate_net_loss(out_t, tt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            parts["total"] = float(loss.detach())
            for key, v in parts.items():
                sums[key] = sums.get(key, 0.0) + v
            n_batches += 1
        entry = {
            "epoch": epoch,
            "train": {k: v / max(n_batches, 1) for k, v in sums.items()},
            "val_loss": evaluate(),
            "lr": opt.param_groups[1]["lr"],
            "elapsed_s": time.perf_counter() - t_train,
        }
        history.append(entry)
        val_loss = entry["val_loss"]
        if select == "best-val" and val_loss is not None and val_loss < best["val_loss"]:
            best = {
                "epoch": epoch,
                "val_loss": val_loss,
                "state": {k: v.detach().clone().cpu() for k, v in net.state_dict().items()},
            }
        log.info(
            "epoch %d/%d train %.4f val %s (%.0fs)",
            epoch,
            epochs,
            entry["train"]["total"],
            "-" if entry["val_loss"] is None else f"{entry['val_loss']:.4f}",
            entry["elapsed_s"],
        )
        if (
            select == "best-val"
            and patience is not None
            and best["epoch"] is not None
            and epoch - best["epoch"] >= patience
        ):
            stopped_early = True
            log.info("early stop: no validation improvement for %d epochs", patience)
            break

    chosen_epoch = history[-1]["epoch"] if history else 0
    if select == "best-val" and best["state"] is not None:
        net.load_state_dict(best["state"])
        chosen_epoch = best["epoch"]
    chosen = next((h for h in history if h["epoch"] == chosen_epoch), None)
    root = find_repo_root()
    out_path = (
        Path(out)
        if out is not None
        else (root if root is not None else Path.cwd())
        / "artifacts"
        / "harness"
        / "liftoff"
        / "models"
        / "liftoff-gates.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "classes": table.classes,
        "num_keypoints": table.max_keypoints,
        "keypoints_per_class": [t.n_keypoints for t in table.types],
        "gate_types_hash": table.hash(),
        "gate_types": cfg.get("gate_types"),
        "backbone": backbone,
        "backbone_config": net.backbone_config,
        "fpn_channels": 96,
        "input_width": in_w,
        "stride": STRIDE,
        # held-out (evaluation) stamp range per capture, as read by `eval --capture`
        "holdout": {name: rec["holdout"] for name, rec in split_meta.items()},
        "splits": split_meta,
        "captures": captures,
        "epochs": epochs,
        "epochs_run": len(history),
        "select": select,
        "patience": patience,
        "stopped_early": stopped_early,
        "chosen_epoch": chosen_epoch,
        "chosen_train_loss": None if chosen is None else chosen["train"]["total"],
        "chosen_val_loss": None if chosen is None else chosen["val_loss"],
        "batch_size": batch_size,
        "lr": lr,
        "preset": preset,
        "final_train_loss": history[-1]["train"]["total"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "git_head": git_head(root),
    }
    state = {
        k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu()
        for k, v in net.state_dict().items()
    }
    torch.save({"state_dict": state, "meta": meta}, out_path)
    log_path = out_path.parent / "training_log.json"
    log_path.write_text(
        json.dumps({"meta": meta, "chosen_epoch": chosen_epoch, "history": history}, indent=2),
        encoding="utf-8",
    )
    meta["weights"] = str(out_path)
    meta["training_log"] = str(log_path)
    return meta
