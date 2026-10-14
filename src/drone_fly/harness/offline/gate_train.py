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
* **Deploy models**: ``--holdout-frac 0 --val-frac 0 --select last`` trains on every labelled frame
  (metadata ``purpose: deploy``, no held-out range). Such weights can be flown but not scored:
  ``eval --capture`` refuses them.
* AdamW + cosine schedule, mixed precision on CUDA. By default (``--select best-val``) the epoch
  with the lowest validation loss is kept, with early stopping after ``patience`` (10) epochs
  without improvement; ``--select last`` keeps the final epoch. ``<weights>.training_log.json``
  (per-epoch
  train and validation losses, chosen epoch) is written beside the weights.
  (Plan amendment during UC-64: the first run overfit the single capture, so model selection got
  its own validation slice rather than reusing the held-out evaluation frames.)
* ``resnet18-hf`` downloads ``microsoft/resnet-18`` once; if that fails the error names the model
  id and the ``tiny-cnn`` fallback.

Default output: the data root's ``models/detector-v1/liftoff-gates.pt``.

**UC71: multi-map training** (:func:`train_multi`, ``drone-fly harness gates train``) replaces the
time split with the track catalogue's map split: train-map captures train, val-map captures select
(``--select val-detection``: decode + PnP + the eval matching rule every ``--val-every`` epochs),
held-out-map captures are refused (:mod:`~drone_fly.harness.offline.gate_dataset`). The frames
come from a memmap cache, sampled map x class balanced; v1's held-out window stays held out.
Outputs go to the data root's ``models/detector-v2/<detector-id>/``. ``train-gates`` (above) stays
for single-capture UC64 training.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import find_repo_root, load_config
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.offline.roll import (
    FALLBACK_ROLL_WEIGHTS,
    KEEP_SIGN_P,
    LIVE_ROLL_EDGES,
    LIVE_ROLL_WEIGHTS,
    ROLL_MAX_DEG,
    item_camera_rolls,
    sample_target_deltas,
)
from drone_fly.harness.recording import git_head

log = logging.getLogger(__name__)

LIVE_VAL_SEED = 71  # the fixed re-roll of the val frames for val_det_live_roll
RESUME_NAME = "resume.pt"  # per-epoch training state in <out_root>/<detector-id>/ (--resume)
# One-off (R2, liftoff-gates-v2-r2-20261014-1021): runs started before 2026-10-14 recorded the
# protect weights by this path; their resume.pt is accepted when today's weights have its content.
LEGACY_PROTECT_PATHS = (r"C:\dev\drone-fly\artifacts\harness\liftoff\models\liftoff-gates.pt",)


def _legacy_resume_match(want: str, ds: Any, catalogue: Any, protect_sha: str | None) -> str | None:
    """Why a resume.pt written under the pre-2026-10-14 manifest hash still matches ``ds``, or
    ``None``. The old hash held the protect weights' path and the catalogue's checkout bytes
    (CRLF on Windows): each legacy path counts only when its weights' sha256 (the file if it is
    still there, else the published release of that name) equals today's ``protect_sha``."""
    from pathlib import PureWindowsPath

    from drone_fly.harness.liftoff.gate_learned import RELEASED_WEIGHTS
    from drone_fly.harness.offline import gate_dataset as gd
    from drone_fly.tracks.catalogue import sha256_file

    cands: list[str | None] = []
    if protect_sha is None:
        cands.append(None)
    for legacy in LEGACY_PROTECT_PATHS:
        p = Path(legacy)
        known = (
            sha256_file(p) if p.is_file() else RELEASED_WEIGHTS.get(PureWindowsPath(legacy).name)
        )
        if protect_sha is not None and known == protect_sha:
            cands.append(legacy)
    for legacy in cands:
        for cat in dict.fromkeys(gd.catalogue_sha_variants(catalogue)):
            if ds.legacy_manifest_sha(protect=legacy, catalogue_sha256=cat) == want:
                return (
                    f"accepted a pre-2026-10-14 resume.pt: protect weights recorded as {legacy!r} "
                    f"(same sha256 as today's), catalogue sha {cat[:12]}"
                )
    return None


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
    if not 0.0 <= holdout_frac < 1.0 or not 0.0 <= val_frac < 1.0 or guard_s < 0:
        raise HarnessConfigError("holdout_frac in [0, 1), val_frac in [0, 1), guard_s >= 0")
    stamps = np.array([r["stamp"] for r in rows], np.float64)
    if window is None and holdout_frac == 0.0:
        if val_frac > 0:
            raise HarnessConfigError("a validation slice needs a held-out slice after it")
        st = np.sort(stamps)
        return ["train"] * len(rows), {
            "mode": "none",
            "holdout": None,
            "val": [],
            "train": _ranges(st),
            "in_flight_s": float(in_flight_clock(st)[-1]),
            "counts": {"train": len(rows), "val": 0, "holdout": 0, "guard": 0},
        }
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
    """Raise unless there is a held-out range and it overlaps no train / validation range."""
    if not split.get("holdout"):
        raise HarnessConfigError("these weights have no held-out split (a deploy model)")
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


FILLS = ("mean", "reflect")


@dataclass(frozen=True)
class RollAug:
    """Roll augmentation (UC71 R2): rotate a frame about the camera's principal point.

    A pure rotation about the optical axis keeps every keypoint's ray angle to the axis, so the
    PnP range of a gate is unchanged; zoom / scale / crop would change the apparent range and are
    deliberately not offered. **Target-matched** (ADR 0010: the training frames are far more level
    than live flight): with probability ``p`` a frame's camera roll is brought to a target drawn
    from the live distribution - a random sign times a magnitude from the ``edges`` / ``weights``
    buckets, clamped to +-135 deg - by rotating it ``target - frame_roll``
    (:func:`~drone_fly.harness.offline.roll.sample_target_deltas`); a frame with an unknown roll
    gets a raw angle from ``fallback_weights``. Rotation is not a reflection: keypoint identity
    is kept (left stays left, top stays top even when it ends up below on screen); ``flip_pairs``
    belong to the horizontal flip only. ``shift`` (fraction of the width, <= 0.05) adds a small
    translation the keypoints follow. Keypoints rotated out of the frame become not-visible; a gate
    whose centre leaves it is dropped. ``fill``: ``mean`` paints the uncovered corners in the
    ImageNet mean colour (0 after ``normalize_pad``, like its bottom padding: no structure);
    ``reflect`` mirrors the image into them (it can mirror unlabelled gates in).
    """

    p: float = 0.0
    edges: tuple[float, ...] = LIVE_ROLL_EDGES
    weights: tuple[float, ...] = LIVE_ROLL_WEIGHTS
    fallback_weights: tuple[float, ...] = FALLBACK_ROLL_WEIGHTS
    shift: float = 0.0
    fill: str = "mean"
    keep_sign_p: float = KEEP_SIGN_P

    @property
    def enabled(self) -> bool:
        return self.p > 0 or self.shift > 0

    def validate(self) -> RollAug:
        e = np.asarray(self.edges, float)
        ok = (
            len(e) >= 2
            and e[0] >= 0
            and bool((np.diff(e) > 0).all())
            and e[-1] <= ROLL_MAX_DEG
            and 0 <= self.p <= 1
            and 0 <= self.shift <= 0.05
            and self.fill in FILLS
            and 0 <= self.keep_sign_p <= 1
        )
        for w in (self.weights, self.fallback_weights):
            wa = np.asarray(w, float)
            ok = ok and len(wa) == len(e) - 1 and bool((wa >= 0).all()) and wa.sum() > 0
        if not ok:
            raise HarnessConfigError(
                f"roll augmentation: increasing bucket edges in [0, {ROLL_MAX_DEG:g}] deg with one "
                "non-negative weight per bucket (target and fallback), p in [0, 1], shift in "
                f"[0, 0.05], fill in {FILLS}"
            )
        return self

    def sample(
        self, rng: np.random.Generator, n: int, width: int, frame_roll: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per image ``(rotation_deg, shift_px (n, 2))``; 0 / (0, 0) when not augmented."""
        ang = np.zeros(n)
        on = rng.random(n) < self.p
        if on.any():
            r = np.full(n, np.nan) if frame_roll is None else np.asarray(frame_roll, float)
            d = sample_target_deltas(rng, r[on], self.edges, self.weights, self.fallback_weights,
                                     keep_sign_p=self.keep_sign_p)  # fmt: skip
            ang[on] = d
        sh = (
            rng.uniform(-1.0, 1.0, (n, 2)) * self.shift * width
            if self.shift > 0
            else np.zeros((n, 2))
        )
        return ang, sh


def rotate_batch(torch: Any, x: Any, mats: list[np.ndarray], fill: str = "mean") -> Any:
    """``grid_sample`` a ``(B, 3, H, W)`` [0, 255] batch through per-image forward pixel maps."""
    import torch.nn.functional as F

    from drone_fly.harness.liftoff.gate_net import IMAGENET_MEAN

    _, _, h, w = x.shape
    theta = torch.as_tensor(np.stack([_sample_theta(m, w, h) for m in mats]),
                            dtype=x.dtype, device=x.device)  # fmt: skip
    grid = F.affine_grid(theta, [len(mats), 3, h, w], align_corners=False)
    if fill == "reflect":
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                             align_corners=False)  # fmt: skip
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1) * 255.0
    return F.grid_sample(x - mean, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False) + mean  # fmt: skip


def roll_affine(angle_deg: float, center: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Forward 2x3 pixel map of a roll: ``p' = R(angle) (p - c) + c + shift`` (image y down, so a
    positive angle turns the content clockwise on screen). Pure."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(center, np.float64)
    return np.concatenate([rot, (c - rot @ c + np.asarray(shift, np.float64))[:, None]], axis=1)


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    return p @ m[:, :2].T + m[:, 2]


def roll_gates(gates: list[dict[str, Any]], m: np.ndarray, width: int, height: int) -> list[dict]:
    """Network targets through a forward pixel map ``m``: keypoints moved, the ones leaving the
    ``width x height`` frame made not-visible, gates whose centre leaves it dropped. Pure."""
    out = []
    for g in gates:
        c = apply_affine(m, g["center"])[0]
        if not (0 <= c[0] <= width - 1 and 0 <= c[1] <= height - 1):
            continue
        kp = apply_affine(m, g["keypoints"])
        inside = (
            (kp[:, 0] >= 0) & (kp[:, 0] <= width - 1) & (kp[:, 1] >= 0) & (kp[:, 1] <= height - 1)
        )
        out.append({"cls": g["cls"], "center": c, "keypoints": kp,
                    "visible": np.asarray(g["visible"], bool) & inside})  # fmt: skip
    return out


def _sample_theta(m: np.ndarray, width: int, height: int) -> np.ndarray:
    """``affine_grid`` theta (normalised output -> input, align_corners=False) of a forward map."""
    inv = np.linalg.inv(np.vstack([m, [0.0, 0.0, 1.0]]))[:2]  # output px -> input px
    s = np.diag([width / 2.0, height / 2.0])
    t = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    s_inv = np.linalg.inv(s)
    a = s_inv @ inv[:, :2] @ s
    b = s_inv @ (inv[:, :2] @ t + inv[:, 2] - t)
    return np.concatenate([a, b[:, None]], axis=1)


def _augment_roll(
    torch: Any, x: Any, gates: list[list[dict]], rng: np.random.Generator, aug: RollAug,
    center: np.ndarray, height: int, frame_roll: np.ndarray | None = None,
    applied: list[float] | None = None,
) -> tuple[Any, list[list[dict]]]:  # fmt: skip
    """:class:`RollAug` on a ``(B, 3, H, W)`` [0, 255] batch (GPU ``grid_sample``) + its targets;
    the applied rotations (deg, non-zero only) are appended to ``applied``."""
    b, _, h, w = x.shape
    ang, sh = aug.sample(rng, b, w, frame_roll)
    if applied is not None:
        applied.extend(float(a) for a in ang if a != 0)
    moved = np.flatnonzero((ang != 0) | (np.abs(sh).sum(axis=1) > 0))
    if not len(moved):
        return x, gates
    ms = {int(i): roll_affine(ang[i], center, sh[i]) for i in moved}
    sel = torch.as_tensor(moved, device=x.device)
    x = x.clone()
    x[sel] = rotate_batch(torch, x[sel], [ms[int(i)] for i in moved], aug.fill)
    gates = [roll_gates(g, ms[i], w, height) if i in ms else g for i, g in enumerate(gates)]
    return x, gates


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
    if select == "best-val" and len(val_idx) == 0:
        raise HarnessConfigError(
            "--select best-val needs validation frames, but the validation slice is empty "
            "(increase --val-frac or use --select last)"
        )
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
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:  # a lowered scale means AMP skipped opt.step()
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
    from drone_fly import paths

    out_path = (
        Path(out) if out is not None else paths.models_root("detector-v1") / "liftoff-gates.pt"
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
        "purpose": (
            "deploy" if all(rec["holdout"] is None for rec in split_meta.values()) else "evaluate"
        ),
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
        "git_head": git_head(find_repo_root()),
    }
    state = {
        k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu()
        for k, v in net.state_dict().items()
    }
    torch.save({"state_dict": state, "meta": meta}, out_path)
    log_path = out_path.with_name(f"{out_path.stem}.training_log.json")
    log_path.write_text(
        json.dumps({"meta": meta, "chosen_epoch": chosen_epoch, "history": history}, indent=2),
        encoding="utf-8",
    )
    meta["weights"] = str(out_path)
    meta["training_log"] = str(log_path)
    return meta


# ----------------------------------------------------------------------------- UC71: multi-map

SELECT_MODES = ("val-detection", "val-loss", "last")
DEFAULT_WEIGHTS_NAME = "liftoff-gates-v2"


def _protected_split(rows_stamps: np.ndarray, window: dict[str, Any], n_used: int) -> dict:
    """A ``splits`` record for a capture trained around a protected (held-out) window."""
    st = np.sort(np.asarray(rows_stamps, np.float64))
    before, after = st[st < window["t_start"]], st[st > window["t_end"]]
    record = {
        "mode": "protected",
        "holdout": {"t_start": float(window["t_start"]), "t_end": float(window["t_end"])},
        "val": [],
        "train": _ranges(before) + _ranges(after),
        "counts": {
            "train": int(n_used),
            "val": 0,
            "holdout": int(window.get("frames") or 0),
            "guard": 0,
        },
    }
    check_disjoint(record)
    return record


class SelectionScorer:
    """Val-map detection rate of the in-training network (decode + PnP + the eval matching rule).

    Uses the cached network inputs (no image reload) and a ``LearnedGateEstimator`` around the
    live network for PnP; the detection rate over in-range gates (4-25 m), ties broken by the lower
    median relative position error.
    """

    def __init__(
        self,
        ds: Any,
        sel_index: np.ndarray,
        table: GateTypeTable,
        cfg: Any,
        telemetry_cfg: dict[str, Any],
        geometry: tuple,
        device: str,
    ) -> None:
        from drone_fly.harness.liftoff.gate_cv import camera_from_config
        from drone_fly.harness.liftoff.masks import FrameMasks
        from drone_fly.harness.offline.capture import load_capture
        from drone_fly.harness.offline.capture_index import camera_config

        self.ds, self.idx, self.table = ds, sel_index, table
        self.geometry, self.device = geometry, device
        self.states: dict[int, Any] = {}
        self.cams: dict[str, Any] = {}
        self.masks: dict[str, Any] = {}
        lc = cfg.section("gates").get("learned") or {}
        self.min_keypoints = int(lc.get("min_keypoints", 4))
        self.score_threshold = float(lc.get("score_threshold", 0.3))
        by_cap: dict[str, list[int]] = {}
        for k in sel_index:
            by_cap.setdefault(ds.items[k].capture, []).append(int(k))
        caps = {c.name: c for c in ds.used}
        for name, ks in by_cap.items():
            c = caps[name]
            cam_cfg = camera_config(cfg.section("camera"), c.meta)
            self.cams[name] = camera_from_config(cam_cfg)
            self.masks[name] = FrameMasks.from_config(cam_cfg)
            lag = 0.0
            if c.labels_dir is not None:
                lm = json.loads((c.labels_dir / "labels_meta.json").read_text(encoding="utf-8"))
                lag = float(lm.get("lag_ms", 0.0))
            cap = load_capture(c.path, telemetry_cfg, 1.0, lag)
            by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
            for k in ks:
                f = by_file.get(ds.items[k].row["file"])
                if f is not None:
                    self.states[k] = f.state

    def score(
        self, net: Any, data: Any, batch_size: int = 8,
        deltas: dict[int, float] | None = None, principal: np.ndarray | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        """Val detection rate; with ``deltas`` (item -> roll, deg) on the rolled frames.

        A rolled frame is scored with the attitude whose camera sees it
        (:func:`~drone_fly.harness.offline.roll.rolled_quat`) and with each gate's
        ``center_in_image`` re-judged at its rolled centre, so the same eval rule applies.
        """
        import dataclasses

        torch = lazy_import("torch", purpose="gate-net selection")
        from drone_fly.harness.liftoff.gate_learned import LearnedGateEstimator
        from drone_fly.harness.liftoff.gate_net import decode, normalize_pad
        from drone_fly.harness.offline.evaluate import score_capture_frame
        from drone_fly.harness.offline.roll import rolled_quat
        from drone_fly.harness.scale import WorldScale

        _, _, padded, scale = self.geometry
        ks = [int(k) for k in self.idx if int(k) in self.states]
        det, rel = [], []
        per_cls: dict[str, list[bool]] = {}
        was_training = net.training
        net.eval()
        est = None
        with torch.no_grad():
            for b in range(0, len(ks), batch_size):
                kk = ks[b : b + batch_size]
                x = torch.as_tensor(np.stack([data[k] for k in kk])).to(self.device).float()
                if deltas is not None:
                    x = rotate_batch(torch, x, [roll_affine(deltas[k], principal, (0, 0))
                                                for k in kk])  # fmt: skip
                outs = decode(net(normalize_pad(x, padded)), score_threshold=self.score_threshold)
                for k, dets in zip(kk, outs, strict=True):
                    item = self.ds.items[k]
                    cam = self.cams[item.capture]
                    state, gates = self.states[k], item.row["gates"]
                    if deltas is not None:
                        state = dataclasses.replace(
                            state, quat=rolled_quat(cam, state.quat, deltas[k])
                        )
                        m = roll_affine(deltas[k], np.array([cam.cx, cam.cy]), (0, 0))
                        gates = []
                        for g in item.row["gates"]:
                            c = apply_affine(m, g["center_px"])[0]
                            inside = 0 <= c[0] <= cam.width - 1 and 0 <= c[1] <= cam.height - 1
                            fm = self.masks[item.capture]
                            masked = (
                                bool(fm.contains(c[None], cam.width, cam.height)[0])
                                if fm
                                else False
                            )
                            gates.append(
                                {
                                    **g,
                                    "center_in_image": bool(g["center_in_image"]) and inside,
                                    "center_masked": masked,  # the fixed OSD masks don't roll
                                }
                            )
                    if est is None:
                        est = LearnedGateEstimator(
                            net, {}, self.table, cam, WorldScale(1.0),
                            min_keypoints=self.min_keypoints,
                        )  # fmt: skip
                    est.camera, est.masks = cam, self.masks[item.capture]
                    for d in dets:
                        d["center"] = d["center"] / scale
                        d["keypoints"] = d["keypoints"] / scale
                    est._cache_index, est._cache, est._frame_index = k, dets, k
                    blank = np.empty((cam.height, cam.width, 3), np.uint8)
                    found = est.detect(blank, state)
                    res = score_capture_frame(found, gates, WorldScale(1.0))
                    for e in res["scored"]:
                        det.append(e["detected"])
                        per_cls.setdefault(str(e.get("cls")), []).append(e["detected"])
                        if e["detected"]:
                            rel.append(e["position_error_rel"])
        if was_training:
            net.train()
        return {
            "n": len(det),
            "detection_rate": float(np.mean(det)) if det else 0.0,
            "position_error_rel_median": float(np.median(rel)) if rel else None,
            "per_class": {
                c: {"n": len(v), "detection_rate": float(np.mean(v))}
                for c, v in sorted(per_cls.items())
            },
        }

    def score_rotated(
        self, net: Any, data: Any, angles: tuple[float, ...], principal: np.ndarray,
        batch_size: int = 8,
    ) -> dict[str, dict[str, Any]]:  # fmt: skip
        """Image-space detection rate on the val frames rolled by each fixed angle (report only).

        A gate counts when its in-range (4-25 m) centre stays in the rolled frame; it is found
        when a decoded detection of its class lies within a quarter of its keypoint extent (at
        least 4 input px) of the rolled centre with >= ``min_keypoints`` visible keypoints in
        frame. No PnP: a roll about the principal point leaves the range unchanged, and the world
        match would need the rolled camera attitude. 0 deg gives the same metric unrolled.
        """
        torch = lazy_import("torch", purpose="gate-net selection")
        from drone_fly.harness.liftoff.gate_net import decode, normalize_pad
        from drone_fly.harness.offline.gate_dataset import in_range

        in_w, in_h, padded, scale = self.geometry
        ks = [int(k) for k in self.idx]
        out: dict[str, dict[str, Any]] = {}
        was_training = net.training
        net.eval()
        with torch.no_grad():
            for ang in angles:
                m = roll_affine(float(ang), principal, (0.0, 0.0))
                n = found = 0
                for b in range(0, len(ks), batch_size):
                    kk = ks[b : b + batch_size]
                    x = torch.as_tensor(np.stack([data[k] for k in kk])).to(self.device).float()
                    if ang:
                        x = rotate_batch(torch, x, [m] * len(kk))
                    outs = decode(net(normalize_pad(x, padded)),
                                  score_threshold=self.score_threshold)  # fmt: skip
                    for k, dets in zip(kk, outs, strict=True):
                        for g in self.ds.items[k].row["gates"]:
                            if not in_range(g):
                                continue
                            c = apply_affine(m, np.asarray(g["center_px"], float) * scale)[0]
                            if not (0 <= c[0] <= in_w - 1 and 0 <= c[1] <= in_h - 1):
                                continue
                            kp = np.asarray(g["keypoints"], float) * scale
                            ext = float(np.linalg.norm(kp.max(axis=0) - kp.min(axis=0)))
                            rad = max(4.0, 0.25 * ext)
                            n += 1
                            for d in dets:
                                if self.table.types[d["cls"]].cls != g["cls"]:
                                    continue
                                if np.linalg.norm(d["center"] - c) > rad:
                                    continue
                                p = d["keypoints"]
                                ok = (
                                    (d["visibility"] >= 0.5)
                                    & (p[:, 0] >= 0)
                                    & (p[:, 0] <= in_w - 1)
                                )
                                ok &= (p[:, 1] >= 0) & (p[:, 1] <= in_h - 1)
                                if int(ok.sum()) >= self.min_keypoints:
                                    found += 1
                                    break
                out[f"{float(ang):g}"] = {"n": n, "rate": found / n if n else None}
        if was_training:
            net.train()
        return out


def better_selection(a: dict[str, Any], b: dict[str, Any] | None) -> bool:
    """Selection order: higher detection rate, then lower median relative error."""
    if b is None:
        return True
    if a["detection_rate"] != b["detection_rate"]:
        return a["detection_rate"] > b["detection_rate"]
    ea, eb = a["position_error_rel_median"], b["position_error_rel_median"]
    return ea is not None and (eb is None or ea < eb)


def train_multi(
    captures: list[str | Path],
    *,
    trackdata_root: str | Path | None = None,
    split_source: str | Path | None = None,
    detector_id: str | None = None,
    weights_name: str = DEFAULT_WEIGHTS_NAME,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = "liftoff",
    epochs: int = 40,
    backbone: str = "resnet18-hf",
    batch_size: int = 8,
    lr: float = 1e-3,
    input_width: int | None = None,
    frame_stride: int = 1,
    max_frames_per_capture: int | None = None,
    select: str = "val-detection",
    val_every: int = 2,
    patience: int | None = None,
    protect_holdout_of: str | Path | None = "release:v1",
    deploy: bool = False,
    device: str = "auto",
    seed: int = 0,
    out_root: str | Path | None = None,
    roll: RollAug | None = None,
    rotated_val: tuple[float, ...] = (),
    live_roll_val: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """``drone-fly harness gates train`` (UC71 AC6): multi-capture, catalogue-split training.

    Train-map captures train; val-map captures select (``val-detection`` every ``val_every``
    epochs, or ``val-loss``); held-out-map captures raise. ``deploy`` retrains on train + val maps
    for ``epochs`` (``select last``) into ``<weights_name>-deploy.pt``. Writes
    ``<out_root>/<detector-id>/`` (weights, training log, ``data_manifest.json``).
    ``protect_holdout_of`` (default the v1 release weights) names weights whose held-out windows
    are never loaded.

    After every epoch the full training state (network, optimiser, scheduler, AMP scaler, RNGs,
    history, best-so-far) goes to ``<detector-id>/resume.pt``; ``resume`` continues from it when
    the data manifest (captures and every training parameter) is unchanged. A finished run
    removes the file.
    """
    torch = lazy_import("torch", purpose="gate-net training")
    from drone_fly.harness.liftoff.gate_cv import camera_from_config
    from drone_fly.harness.liftoff.gate_learned import configured_weights
    from drone_fly.harness.liftoff.gate_net import (
        BACKBONES,
        STRIDE,
        GateKeypointNet,
        build_targets,
        gate_net_loss,
        normalize_pad,
    )
    from drone_fly.harness.offline import gate_dataset as gd
    from drone_fly.harness.offline.capture_index import resolve_captures
    from drone_fly.tracks import catalogue as cat

    if resume and not detector_id:
        raise HarnessConfigError("--resume needs the --detector-id of the interrupted run")
    if select not in SELECT_MODES:
        raise HarnessConfigError(f"--select must be one of {SELECT_MODES}, got {select!r}")
    if deploy:
        select = "last"
    if backbone not in BACKBONES:
        raise HarnessConfigError(f"--backbone must be one of {BACKBONES}, got {backbone!r}")
    if val_every < 1:
        raise HarnessConfigError("--val-every must be >= 1")
    cfg = load_config("liftoff", config, preset=preset)
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    catalogue = cat.load_catalogue(split_source) if split_source else cat.load_named("liftoff")
    lw = input_width or int((cfg.section("gates").get("learned") or {}).get("input_width", 768))
    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if protect_holdout_of == "release:v1":
        protect_holdout_of = configured_weights({"release": "v1"})
    protect = gd.read_weights_meta(protect_holdout_of) if protect_holdout_of else None
    protect_sha = None
    if protect_holdout_of:
        from drone_fly.harness.liftoff.gate_learned import resolve_weights_path
        from drone_fly.tracks.catalogue import sha256_file

        # by content, not path: the manifest (and --resume) survive moving the data
        protect_sha = sha256_file(resolve_weights_path(protect_holdout_of))
    params = {
        "epochs": epochs,
        "backbone": backbone,
        "batch_size": batch_size,
        "lr": lr,
        "input_width": lw,
        "select": select,
        "val_every": val_every,
        "seed": seed,
        "deploy": deploy,
        "protect_holdout_of": None if protect_sha is None else f"sha256:{protect_sha}",
        "roll_aug": None if roll is None or not roll.enabled else asdict(roll.validate()),
        "rotated_val": list(rotated_val),
        "live_roll_val": live_roll_val,
    }
    ds = gd.assemble(
        resolve_captures([str(c) for c in captures]),
        catalogue=catalogue,
        table=table,
        not_gates=getattr(cfg, "not_gates", ()),
        trackdata_root=trackdata_root,
        protect=protect,
        frame_stride=frame_stride,
        max_frames_per_capture=max_frames_per_capture,
        telemetry_cfg=cfg.section("telemetry"),
        include_selection_in_train=deploy,
        params=params,
        preset_uptilt_deg=float(cfg.section("camera")["uptilt_deg"]),
    )
    for c in ds.refused:
        log.warning("capture %s refused: %s", c.name, c.reason)
    train_idx = np.array([i for i, it in enumerate(ds.items) if it.role == gd.ROLE_TRAIN], int)
    sel_idx = np.array([i for i, it in enumerate(ds.items) if it.role == gd.ROLE_SELECTION], int)
    if len(train_idx) == 0:
        raise HarnessConfigError("no training frames (no usable train-map capture)")
    if select != "last" and len(sel_idx) == 0:
        raise HarnessConfigError(
            f"--select {select} needs val-map captures; none is usable (or use --select last)"
        )
    weights = np.array([ds.items[i].weight for i in train_idx], np.float64)
    weights /= weights.sum()

    data, idx_meta = gd.build_frame_cache(ds, lw, cfg.section("camera"))
    geometry = tuple(idx_meta["geometry"])
    in_w, _, padded, scale = geometry
    out_h, out_w = padded // STRIDE, in_w // STRIDE
    targets_raw = [_train_gates(it.row, table, scale) for it in ds.items]
    in_h = int(geometry[1])
    cam = camera_from_config(cfg.section("camera"))
    principal = np.array([cam.cx * scale, cam.cy * scale])  # the optical axis in input pixels
    scorer = None
    if select == "val-detection":
        scorer = SelectionScorer(ds, sel_idx, table, cfg, cfg.section("telemetry"), geometry, dev)

    net = GateKeypointNet(
        len(table), table.max_keypoints, backbone, pretrained=backbone != "tiny-cnn"
    )
    net.to(dev)
    bb_params = list(net.backbone.parameters())
    bb_ids = {id(p) for p in bb_params}
    head = [p for p in net.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW(
        [
            {"params": bb_params, "lr": lr * (0.2 if backbone != "tiny-cnn" else 1.0)},
            {"params": head, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    total_steps = max(1, epochs * math.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    )
    cuda = str(dev).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    item_roll = None
    live_deltas = None
    if (roll is not None and roll.enabled) or live_roll_val:
        item_roll = item_camera_rolls(ds, cfg.section("telemetry"), cfg.section("camera"))
        for role, ix in (("train", train_idx), ("val", sel_idx)):
            r = np.abs(item_roll[ix])
            r = r[np.isfinite(r)]
            if len(r):
                log.info("%s camera |roll| p50 %.1f p90 %.1f deg (%d of %d frames known)",
                         role, np.median(r), np.percentile(r, 90), len(r), len(ix))  # fmt: skip
    if live_roll_val and scorer is not None:
        # one fixed re-roll per val frame (same every epoch): targets drawn from the live mix
        d = sample_target_deltas(np.random.default_rng(LIVE_VAL_SEED), item_roll[sel_idx])
        live_deltas = {int(k): float(v) for k, v in zip(sel_idx, d, strict=True)}

    def batch(idx: np.ndarray, augment: bool) -> tuple[Any, dict[str, Any]]:
        x = torch.as_tensor(np.stack([data[i] for i in idx])).to(dev).float()
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
            if roll is not None and roll.enabled:
                fr = None if item_roll is None else np.where(flip, -1.0, 1.0) * item_roll[idx]
                x, gates = _augment_roll(torch, x, gates, rng, roll, principal, in_h, fr,
                                         roll_applied)  # fmt: skip
            x = _augment_colour(torch, x, rng)
        tg = [build_targets(g, out_h, out_w, len(table), table.max_keypoints) for g in gates]
        tt = {k: torch.as_tensor(np.stack([t[k] for t in tg])).to(dev) for k in tg[0]}
        return normalize_pad(x, padded), tt

    def val_loss() -> float | None:
        if len(sel_idx) == 0:
            return None
        net.eval()
        total = 0.0
        with torch.no_grad():
            for k in range(0, len(sel_idx), batch_size):
                idx = sel_idx[k : k + batch_size]
                x, tt = batch(idx, augment=False)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    loss, _ = gate_net_loss(net(x), tt)
                total += float(loss) * len(idx)
        net.train()
        return total / len(sel_idx)

    # -- outputs: <out_root>/<detector-id>/ --
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    did = detector_id or f"{weights_name}-{stamp}"
    out_dir = (Path(out_root) if out_root else gd.detector_models_root()) / did
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / RESUME_NAME

    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"epoch": None, "metric": None, "state": None}
    stopped_early = False
    start_epoch, elapsed0 = 1, 0.0
    if resume:
        if not resume_path.is_file():
            raise HarnessConfigError(f"--resume: no training state at {resume_path}")
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        if ck["manifest_sha256"] != ds.manifest_sha():
            note = _legacy_resume_match(ck["manifest_sha256"], ds, catalogue, protect_sha)
            if note is None:
                raise HarnessConfigError(
                    f"--resume: {resume_path} was written for other data or training parameters"
                )
            log.warning("--resume: %s", note)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        if cuda and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        history, best, stopped_early = ck["history"], ck["best"], ck["stopped_early"]
        start_epoch = epochs + 1 if stopped_early else ck["epoch"] + 1
        elapsed0 = float(ck["elapsed_s"])
        log.info("resumed after epoch %d from %s", ck["epoch"], resume_path)

    def save_resume(epoch: int) -> None:
        tmp = resume_path.with_suffix(".tmp")
        torch.save(
            {
                "manifest_sha256": ds.manifest_sha(),
                "epoch": epoch,
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if cuda else None,
                "history": history,
                "best": best,
                "stopped_early": stopped_early,
                "elapsed_s": time.perf_counter() - t0,
            },
            tmp,
        )
        os.replace(tmp, resume_path)

    net.train()
    t0 = time.perf_counter() - elapsed0
    roll_applied: list[float] = []
    for epoch in range(start_epoch, epochs + 1):
        roll_applied.clear()
        perm = rng.choice(train_idx, size=len(train_idx), replace=True, p=weights)
        sums: dict[str, float] = {}
        nb = 0
        for k in range(0, len(perm), batch_size):
            x, tt = batch(perm[k : k + batch_size], augment=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                out_t = net(x)
            loss, parts = gate_net_loss(out_t, tt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() >= scale:  # a lowered scale means AMP skipped opt.step()
                sched.step()
            parts["total"] = float(loss.detach())
            for key, v in parts.items():
                sums[key] = sums.get(key, 0.0) + v
            nb += 1
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train": {k: v / max(nb, 1) for k, v in sums.items()},
            "val_loss": val_loss(),
            "lr": opt.param_groups[1]["lr"],
            "elapsed_s": time.perf_counter() - t0,
        }
        metric = None
        if scorer is not None and (epoch % val_every == 0 or epoch == epochs):
            metric = scorer.score(net, data, batch_size)
            entry["val_detection"] = metric
            if live_deltas is not None:
                entry["val_det_live_roll"] = scorer.score(net, data, batch_size, live_deltas,
                                                          principal)  # fmt: skip
            if rotated_val:
                entry["rotated_val"] = scorer.score_rotated(
                    net, data, tuple(rotated_val), principal, batch_size
                )
        elif select == "val-loss" and entry["val_loss"] is not None:
            metric = {"detection_rate": -entry["val_loss"], "position_error_rel_median": None}
        history.append(entry)
        if metric is not None and better_selection(metric, best["metric"]):
            state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
            best = {"epoch": epoch, "metric": metric, "state": state}
        log.info(
            "epoch %d/%d train %.4f val_loss %s val_det %s (%.0fs)",
            epoch,
            epochs,
            entry["train"]["total"],
            entry["val_loss"],
            (entry.get("val_detection") or {}).get("detection_rate"),
            entry["elapsed_s"],
        )
        if roll_applied:
            d = np.abs(np.asarray(roll_applied))
            ra = {
                "n": int(len(d)),
                "p50": float(np.median(d)),
                "p90": float(np.percentile(d, 90)),
                "max": float(d.max()),
            }
            entry["roll_applied_deg"] = ra
            log.info(
                "  roll |delta| p50 %.1f p90 %.1f max %.1f deg over %d images",
                ra["p50"], ra["p90"], ra["max"], ra["n"],
            )  # fmt: skip
        if "val_detection" in entry:
            pc = entry["val_detection"].get("per_class") or {}
            log.info("  val_det per class: %s", {c: round(v["detection_rate"], 3)
                                                 for c, v in pc.items()})  # fmt: skip
        if "val_det_live_roll" in entry:
            log.info("  val_det_live_roll %.3f", entry["val_det_live_roll"]["detection_rate"])
        if "rotated_val" in entry:
            log.info("  rotated val: %s", {a: v["rate"] for a, v in entry["rotated_val"].items()})
        stopped_early = bool(
            patience
            and select != "last"
            and best["epoch"] is not None
            and epoch - best["epoch"] >= patience
        )
        save_resume(epoch)
        if stopped_early:
            break

    chosen_epoch = history[-1]["epoch"]
    if select != "last" and best["state"] is not None:
        net.load_state_dict(best["state"])
        chosen_epoch = best["epoch"]

    stem = f"{weights_name}-deploy" if deploy else weights_name
    out_path = out_dir / f"{stem}.pt"
    manifest_path = out_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(ds.manifest(), indent=2), encoding="utf-8")
    trained, selection, splits, holdout = [], [], {}, {}
    for c in ds.used:
        used = [it for it in ds.items if it.capture == c.name]
        rec = {
            "name": c.name,
            "capture": str(c.path),
            "catalogue_id": c.meta["catalogue_id"],
            "map": c.meta["map"],
            "split_role": c.role,
            "capture_sha256": c.sha256,
            "n_used": len(used),
        }
        if c.role != gd.ROLE_TRAIN:
            selection.append(rec)
            continue
        trained.append(rec)
        if c.protected and not deploy:
            stamps = np.array([it.row["stamp"] for it in used])
            splits[c.name] = _protected_split(stamps, c.protected, len(used))
            holdout[c.name] = splits[c.name]["holdout"]
    chosen_det = best["metric"] if select == "val-detection" else None
    meta = {
        "format": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detector_id": did,
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
        # protected (held-out) stamp windows per capture, as read by `eval` / detector_noise
        "holdout": holdout,
        "splits": splits,
        "purpose": "deploy" if deploy else "evaluate",
        # train-map captures only: val-map captures (selection) count as never trained
        "captures": trained,
        "selection_captures": selection,
        # in-range gate-frames per class in the training data ("seen" types for the D-U1 rule)
        "trained_types": dict(
            Counter(
                g["cls"]
                for it in ds.items
                if it.role == gd.ROLE_TRAIN
                for g in it.row["gates"]
                if gd.in_range(g)
            )
        ),
        "split_source": str(getattr(catalogue, "path", split_source)),
        "catalogue_sha256": ds.catalogue_sha256,
        "data_manifest_sha256": ds.manifest_sha(),
        "epochs": epochs,
        "epochs_run": len(history),
        "select": select,
        "val_every": val_every,
        "stopped_early": stopped_early,
        "chosen_epoch": chosen_epoch,
        "chosen_val_detection": chosen_det,
        "batch_size": batch_size,
        "lr": lr,
        "preset": preset,
        "git_head": ds.git_head,
    }
    state = {
        k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu()
        for k, v in net.state_dict().items()
    }
    torch.save({"state_dict": state, "meta": meta}, out_path)
    log_path = out_path.with_name(f"{out_path.stem}.training_log.json")
    log_path.write_text(
        json.dumps({"meta": meta, "chosen_epoch": chosen_epoch, "history": history}, indent=2),
        encoding="utf-8",
    )
    resume_path.unlink(missing_ok=True)
    meta["weights"] = str(out_path)
    meta["training_log"] = str(log_path)
    meta["data_manifest"] = str(manifest_path)
    return meta
