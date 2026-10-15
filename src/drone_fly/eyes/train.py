"""AC4: train :class:`~drone_fly.eyes.model.EyesNet` (``drone-fly eyes train``; offline).

* AdamW (lr 1e-3, weight decay 1e-4), cosine decay to 1e-5 over the epochs, gradient clip 1.0,
  AMP (fp16 autocast + GradScaler) on CUDA only.
* Augmentation per sample, identical across its K frames: brightness / contrast / per-channel
  gain and a random box blur (motion-blur stand-in). **No flips**: a mirrored frame would need
  mirrored rays and ego-motion.
* Loss: smooth-L1 (beta 0.05) on ``free`` with weight ``w_hit`` (4) on rays whose truth is a hit,
  plus BCE of the confidence logit against ``|pred - truth| * RAY_RANGE * s < tol_m`` (0.5
  training m).
* Model selection: best validation loss (``--select best-val``, early stop after ``--patience``
  epochs without improvement) or the last epoch.
* Resume: ``resume.pt`` is written atomically at the end of every epoch (model, optimiser,
  scheduler, scaler, log, RNG states, the dataset sha). Every epoch reseeds from
  ``seed * 1000 + epoch``, so an interrupted run resumed from ``resume.pt`` is bit-identical (CPU,
  ``num_workers 0``). ``--resume`` refuses a different dataset. A ``STOP`` file in the run folder
  or Ctrl-C ends the run at the last completed epoch.

Outputs (``<models root>/eyes/<UTC stamp>-<dataset id>/``): ``eyes.pt`` (``{meta, state_dict}``,
the weights meta of :func:`~drone_fly.eyes.model.default_meta`), ``eyes.meta.json``,
``training_log.json`` and ``resume.pt``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from drone_fly.eyes.dataset import EyesSamples, load_manifest
from drone_fly.eyes.infer import resolve_device
from drone_fly.eyes.model import build_from_meta, count_parameters, default_meta
from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

STOP_FILE = "STOP"
RESUME_FILE = "resume.pt"
WEIGHTS_FILE = "eyes.pt"
SMOOTH_L1_BETA = 0.05
GRAD_CLIP = 1.0
LR_MIN = 1e-5
MAX_PARAMS = 1_500_000


def runs_root() -> Path:
    from drone_fly import paths

    return paths.models_root() / "eyes"


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _rng_states() -> dict[str, Any]:
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch": torch.get_rng_state()}  # fmt: skip
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_states(st: dict[str, Any]) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def augment(x: torch.Tensor, rng: np.random.Generator, k_frames: int) -> torch.Tensor:
    """Photometric + blur augmentation of ``(B, 3K, H, W)`` 0-255 floats (same per sample)."""
    b = x.shape[0]
    dev = x.device
    valid = (x.view(b, k_frames, 3, *x.shape[2:]).sum(dim=2, keepdim=True) > 0).float()
    valid = valid.expand(-1, -1, 3, -1, -1).reshape_as(x)
    bright = torch.as_tensor(rng.uniform(0.75, 1.25, (b, 1, 1, 1)), dtype=x.dtype, device=dev)
    contrast = torch.as_tensor(rng.uniform(0.75, 1.25, (b, 1, 1, 1)), dtype=x.dtype, device=dev)
    gain = torch.as_tensor(rng.uniform(0.9, 1.1, (b, 1, 3, 1, 1)), dtype=x.dtype, device=dev)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    y = ((x - mean) * contrast + mean) * bright
    y = (y.view(b, k_frames, 3, *x.shape[2:]) * gain).reshape_as(x)
    blur = rng.random(b) < 0.3
    if blur.any():
        ks = int(rng.choice([3, 5]))
        sel = torch.as_tensor(np.flatnonzero(blur), device=dev)
        y[sel] = F.avg_pool2d(y[sel], ks, stride=1, padding=ks // 2, count_include_pad=False)
    return (y * valid).clamp(0.0, 255.0)


def _batch(data: EyesSamples, idx: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    d = data.data

    def t(a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype).to(device)

    return {
        "frames": t(data.frames_for(idx), torch.uint8),
        "uv": t(d["uv"][idx]),
        "rays_cam": t(d["rays_cam"][idx]),
        "in_view": t(d["in_view"][idx].astype(np.float32)),
        "ego": t(d["ego"][idx]),
        "free": t(d["free"][idx]),
        "hit": t(d["hit"][idx].astype(np.float32)),
    }


def losses(
    free_pred: torch.Tensor,
    logit: torch.Tensor,
    free: torch.Tensor,
    hit: torch.Tensor,
    *,
    range_train: float,
    w_hit: float,
    tol_m: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``(total, parts)``: weighted smooth-L1 on ``free`` + confidence BCE."""
    w = 1.0 + (w_hit - 1.0) * hit
    reg = (F.smooth_l1_loss(free_pred.float(), free, beta=SMOOTH_L1_BETA, reduction="none")
           * w).sum() / w.sum()  # fmt: skip
    err_m = (free_pred.detach().float() - free).abs() * range_train
    target = (err_m < tol_m).float()
    bce = F.binary_cross_entropy_with_logits(logit.float(), target)
    return reg + bce, {
        "reg": float(reg.detach()),
        "bce": float(bce.detach()),
        "mae_m": float(err_m.mean()),
    }


def _evaluate(net: Any, data: EyesSamples, device: str, batch: int, kw: dict[str, Any]):
    net.eval()
    tot, n, mae = 0.0, 0, 0.0
    with torch.no_grad():
        for a in range(0, len(data), batch):
            idx = np.arange(a, min(a + batch, len(data)))
            b = _batch(data, idx, device)
            fp, lg = net(b["frames"], b["uv"], b["rays_cam"], b["in_view"], b["ego"])
            loss, parts = losses(fp, lg, b["free"], b["hit"], **kw)
            tot += float(loss) * len(idx)
            mae += parts["mae_m"] * len(idx)
            n += len(idx)
    net.train()
    return (tot / n, mae / n) if n else (None, None)


def train_eyes(
    dataset: str | Path,
    *,
    out: str | Path | None = None,
    epochs: int = 40,
    batch: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    w_hit: float = 4.0,
    tol_m: float = 0.5,
    device: str = "auto",
    select: str = "best-val",
    patience: int = 8,
    seed: int = 78,
    num_workers: int = 0,
    resume: bool = False,
    max_samples: int | None = None,
    cache_root: str | Path | None = None,
    gpu_prototype_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train (see the module docstring); returns the training log."""
    from drone_fly.eyes.fov import array_sha, offline_contract

    if select not in ("best-val", "last"):
        raise HarnessConfigError(f"--select must be best-val or last, got {select!r}")
    if num_workers != 0:
        raise HarnessConfigError("--num-workers: only 0 is supported (in-process batches)")
    dataset = Path(dataset)
    man = load_manifest(dataset)
    ls = man.get("label_settings") or {}
    contract = offline_contract(gpu_prototype_dir)
    if ls.get("cone_sha256") and array_sha(contract.CONE) != ls["cone_sha256"]:
        raise HarnessConfigError("the contract's CONE differs from the labels' (cone sha)")
    range_train = float(ls["ray_range"]) * float(ls["ray_range_scale"])
    dev = resolve_device(device)
    cuda = dev.startswith("cuda")
    if out is None:
        if resume:
            raise HarnessConfigError("--resume needs --out <the run folder>")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = runs_root() / f"{stamp}-{man['id']}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inp = man["input"]
    meta = default_meta(
        cone=contract.CONE,
        camera=man["preset"]["camera"],
        masks=man["masks"],
        world_scale=float(ls["world_scale"]),
        ray_range_scale=float(ls["ray_range_scale"]),
        ray_range=float(ls["ray_range"]),
        k_frames=int(inp["k_frames"]),
        dt_s=float(inp["dt_s"]),
        hist_tol_s=float(inp["hist_tol_s"]),
        width=int(inp["width"]),
        height=int(inp["height"]),
        tol_m=float(tol_m),
        label_settings_sha256=man["label_settings_sha256"],
        dataset_id=man["id"],
        dataset_sha256=man["dataset_sha256"],
        split_maps=man["split_maps"],
    )
    train = EyesSamples(dataset, "train", max_samples=max_samples, cache_root=cache_root)
    if not len(train):
        raise HarnessConfigError(f"{dataset} has no train samples")
    val = EyesSamples(dataset, "val", max_samples=max_samples, cache_root=cache_root)
    has_val = len(val) > 0
    if select == "best-val" and not has_val:
        log.warning("no validation samples: selecting the last epoch")
        select = "last"

    torch.manual_seed(seed)
    net = build_from_meta(meta).to(dev)
    n_params = count_parameters(net)
    if n_params > MAX_PARAMS:
        raise HarnessConfigError(f"EyesNet has {n_params} parameters (> {MAX_PARAMS})")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=LR_MIN)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    args = {"epochs": epochs, "batch": batch, "lr": lr, "weight_decay": weight_decay,
            "w_hit": w_hit, "tol_m": tol_m, "select": select, "patience": patience, "seed": seed,
            "max_samples": max_samples, "device": dev}  # fmt: skip
    hist: list[dict[str, Any]] = []
    start, best, best_epoch, best_state = 0, float("inf"), -1, None
    rp = out / RESUME_FILE
    if resume:
        if not rp.is_file():
            raise HarnessConfigError(f"--resume: no {RESUME_FILE} in {out}")
        ck = torch.load(rp, map_location="cpu", weights_only=False)
        if ck["dataset_sha256"] != man["dataset_sha256"]:
            raise HarnessConfigError(
                f"--resume: {out} was trained on dataset {ck['dataset_id']} "
                f"(sha {ck['dataset_sha256'][:12]}), not {man['id']}"
            )
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, best_epoch = int(ck["epoch"]) + 1, float(ck["best"]), int(ck["best_epoch"])
        best_state, hist = ck["best_state"], list(ck["log"])
        _set_rng_states(ck["rng"])
    kw = {"range_train": range_train, "w_hit": w_hit, "tol_m": tol_m}
    k = int(inp["k_frames"])
    status = "completed"
    stop = out / STOP_FILE
    net.train()
    try:
        for ep in range(start, epochs):
            es = seed * 1000 + ep
            rng = np.random.default_rng(es)
            torch.manual_seed(es)
            perm = rng.permutation(len(train))
            t0 = time.perf_counter()
            tl, tn = 0.0, 0
            for a in range(0, len(perm), batch):
                if stop.exists():
                    raise _Stop
                idx = perm[a : a + batch]
                b = _batch(train, idx, dev)
                x = augment(b["frames"].float(), rng, k)
                with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                    fp, lg = net(x, b["uv"], b["rays_cam"], b["in_view"], b["ego"])
                loss, _ = losses(fp, lg, b["free"], b["hit"], **kw)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
                tl += float(loss.detach()) * len(idx)
                tn += len(idx)
            sched.step()
            vl, vmae = _evaluate(net, val, dev, batch, kw) if has_val else (None, None)
            score = vl if select == "best-val" else -ep
            if score is not None and score < best:
                best, best_epoch = float(score), ep
                best_state = {n_: v.detach().cpu().clone() for n_, v in net.state_dict().items()}
            hist.append({"epoch": ep, "train_loss": tl / max(tn, 1), "val_loss": vl,
                         "val_mae_m": vmae, "lr": opt.param_groups[0]["lr"],
                         "seconds": time.perf_counter() - t0})  # fmt: skip
            log.info("epoch %d: train %.4f val %s", ep, tl / max(tn, 1), vl)
            _atomic_save({"model": net.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": ep,
                          "best": best, "best_epoch": best_epoch, "best_state": best_state,
                          "log": hist, "rng": _rng_states(), "args": args,
                          "dataset_id": man["id"],
                          "dataset_sha256": man["dataset_sha256"]}, rp)  # fmt: skip
            if select == "best-val" and ep - best_epoch >= patience:
                status = "early-stopped"
                break
    except (_Stop, KeyboardInterrupt):
        status = "stopped"
        log.warning("training stopped; resume with --resume --out %s", out)
    if not hist:
        raise HarnessConfigError("no epoch completed: nothing to save")
    state = best_state if best_state is not None else net.state_dict()
    final = {**meta, "params": n_params, "trained_epochs": len(hist), "best_epoch": best_epoch,
             "select": select, "status": status}  # fmt: skip
    _atomic_save({"meta": final, "state_dict": state}, out / WEIGHTS_FILE)
    (out / "eyes.meta.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    result = {"run_dir": str(out), "status": status, "args": args, "params": n_params,
              "best_epoch": best_epoch, "dataset_id": man["id"], "train_samples": len(train),
              "val_samples": len(val), "log": hist}  # fmt: skip
    (out / "training_log.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class _Stop(Exception):
    """The STOP file appeared."""
