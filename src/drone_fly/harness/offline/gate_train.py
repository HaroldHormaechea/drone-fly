"""Train the learned gate estimator's keypoint network on auto-labelled captures.

``drone-fly harness train-gates --labels <dir> [<dir> ...] [--out ...] [--preset ...]
[--epochs N] [--backbone resnet18-hf|tiny-cnn] [--holdout-frac 0.2 | --holdout t0:t1]``

* Inputs: one or more label folders written by ``drone-fly harness label`` (``labels.jsonl`` +
  ``labels_meta.json``, which names the capture). Their gate-type hash must match the configured
  ``gate_types``.
* Images are masked (``camera.masks``) and resized exactly as at runtime
  (:func:`~drone_fly.harness.liftoff.gate_net.prepare_input` building blocks), then cached in
  memory. Augmentation: brightness / contrast / saturation jitter, blur, and horizontal flips
  that swap the gate type's ``flip_pairs``.
* **Time split**: by default the last ``holdout_frac`` (20 %) of each capture's in-flight time is
  held out (never trained on); ``--holdout t0:t1`` holds out that window instead (seconds from
  the capture's first labelled frame). The split is stored in the weights metadata so
  ``drone-fly harness eval --capture`` scores only held-out frames. With one capture, held-out
  laps revisit the same viewpoints: the split is optimistic.
* AdamW + cosine schedule, mixed precision on CUDA; ``training_log.json`` beside the weights.
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


def holdout_window(
    rows: list[dict[str, Any]], frac: float = 0.2, window: tuple[float, float] | None = None
) -> dict[str, Any]:
    """The held-out stamp range of one capture (absolute stamps, inclusive)."""
    stamps = np.sort(np.array([r["stamp"] for r in rows], np.float64))
    if len(stamps) == 0:
        raise HarnessConfigError("a label set has no frames")
    if window is not None:
        t0, t1 = window
        return {
            "mode": "window",
            "t_start": float(stamps[0] + t0),
            "t_end": float(stamps[0] + t1),
            "window_s": [t0, t1],
        }
    if not 0.0 < frac < 1.0:
        raise HarnessConfigError(f"--holdout-frac must be in (0, 1), got {frac}")
    clock = in_flight_clock(stamps)
    split = int(np.searchsorted(clock, (1.0 - frac) * clock[-1]))
    split = min(max(split, 1), len(stamps) - 1)
    return {
        "mode": "last_fraction",
        "frac": frac,
        "t_start": float(stamps[split]),
        "t_end": float(stamps[-1]),
        "in_flight_s": float(clock[-1]),
    }


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
    images, targets_raw, held, names = [], [], [], []
    geometry = None
    split_meta, captures = {}, []
    t_load = time.perf_counter()
    for s in sets:
        win = holdout_window(s.rows, holdout_frac, window)
        split_meta[s.name] = win
        n_train = n_hold = 0
        for row in s.rows:
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
            ho = is_held_out(row["stamp"], win)
            held.append(ho)
            names.append(s.name)
            n_hold += ho
            n_train += not ho
        captures.append(
            {
                "name": s.name,
                "capture": str(s.capture),
                "labels_dir": str(s.labels_dir),
                "n_train": n_train,
                "n_holdout": n_hold,
            }
        )
    assert geometry is not None
    in_w, in_h, padded, _ = geometry
    out_h, out_w = padded // STRIDE, in_w // STRIDE
    held_arr = np.array(held)
    train_idx, val_idx = np.flatnonzero(~held_arr), np.flatnonzero(held_arr)
    if len(train_idx) == 0:
        raise HarnessConfigError("no training frames left after the hold-out split")
    log.info(
        "loaded %d frames (%d train / %d held out) in %.0fs at %dx%d",
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
        log.info(
            "epoch %d/%d train %.4f val %s (%.0fs)",
            epoch,
            epochs,
            entry["train"]["total"],
            "-" if entry["val_loss"] is None else f"{entry['val_loss']:.4f}",
            entry["elapsed_s"],
        )

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
        "holdout": split_meta,
        "holdout_frac": None if window is not None else holdout_frac,
        "captures": captures,
        "epochs": epochs,
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
    log_path.write_text(json.dumps({"meta": meta, "history": history}, indent=2), encoding="utf-8")
    meta["weights"] = str(out_path)
    meta["training_log"] = str(log_path)
    return meta
