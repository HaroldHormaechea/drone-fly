"""AC6: accuracy report on held-out (or val) maps (``drone-fly eyes eval``; offline).

Refuses the ``train`` split and any map the weights trained on. Every figure is computed by the
pure-NumPy :func:`ray_metrics` / :func:`calibration` / :func:`evaluate_arrays` (unit-testable):

* per ray and aggregated: median absolute error (training m and Liftoff m), **miss rate** (a truth
  hit within range whose predicted distance is beyond truth + ``tol_m``) and **false-alarm rate**
  (predicted distance below truth - ``tol_m``; over truth-clear rays and over all rays);
* each split into in-view / out-of-view rays and into distance bins on the truth (fractions of the
  range ``[0, .25) [.25, .5) [.5, .75) [.75, 1)`` + ``clear``; edges also in Liftoff m);
* confidence calibration (10 equal-width bins + ECE, the count-weighted mean ``|conf - acc|``,
  accuracy = ``|err| < tol_m``), overall and per distance bin;
* per map, per obstacle size (the hit obstacle's thickness ``2 x`` min half-extent, Liftoff m:
  thin < 0.2, medium < 1.0, large; ``n/a`` = no hit) and warning time (stats + the miss rate on
  frames with warning < 0.5 s);
* baselines: always-clear and the train-split per-ray mean.

``tol_m`` is in training metres (default: the weights meta's). Writes ``eval_<split>.json`` +
``.md`` next to the weights (default).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

DIST_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLEAR_EPS = 1e-6
N_CAL_BINS = 10
THIN_M, LARGE_M = 0.2, 1.0
WARNING_SHORT_S = 0.5


def ray_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    range_train: float,
    world_scale: float,
    tol_m: float,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Error / miss / false-alarm figures over the selected rays (arrays of free in [0, 1])."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(true, np.float64)
    if mask is not None:
        p, t = p[mask], t[mask]
    p, t = p.reshape(-1), t.reshape(-1)
    n = int(len(t))
    if not n:
        return {"n": 0}
    pd, td = p * range_train, t * range_train
    err = np.abs(pd - td)
    hit = t < 1.0 - CLEAR_EPS
    clear = ~hit
    fa = pd < td - tol_m
    miss = hit & (pd > td + tol_m)
    return {
        "n": n,
        "n_hit": int(hit.sum()),
        "median_abs_err_train_m": float(np.median(err)),
        "median_abs_err_liftoff_m": float(np.median(err) * world_scale),
        "mean_abs_err_train_m": float(err.mean()),
        "miss_rate": float(miss.sum() / hit.sum()) if hit.any() else None,
        "false_alarm_rate_clear": float((fa & clear).sum() / clear.sum()) if clear.any() else None,
        "false_alarm_rate_all": float(fa.mean()),
    }


def calibration(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_CAL_BINS) -> dict[str, Any]:
    """Equal-width confidence bins + ECE (count-weighted mean ``|mean conf - accuracy|``)."""
    c = np.asarray(conf, np.float64).reshape(-1)
    a = np.asarray(correct, bool).reshape(-1)
    if not len(c):
        return {"n": 0, "ece": None, "bins": []}
    idx = np.minimum((c * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        sel = idx == b
        k = int(sel.sum())
        row = {"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": k, "conf": None, "acc": None}
        if k:
            row["conf"], row["acc"] = float(c[sel].mean()), float(a[sel].mean())
            ece += k / len(c) * abs(row["conf"] - row["acc"])
        bins.append(row)
    return {"n": int(len(c)), "ece": float(ece), "bins": bins}


def distance_bin(true: np.ndarray) -> np.ndarray:
    """Bin index per ray: 0..3 = the range quarters, 4 = clear."""
    t = np.asarray(true, np.float64)
    b = np.clip(np.searchsorted(DIST_BINS, t, side="right") - 1, 0, 3)
    return np.where(t >= 1.0 - CLEAR_EPS, 4, b)


def size_bin(min_half_m: np.ndarray, thin_m: float = THIN_M, large_m: float = LARGE_M):
    """``thin`` / ``medium`` / ``large`` by thickness ``2 x min half-extent``; ``n/a`` = no hit."""
    h = np.asarray(min_half_m, np.float64)
    thick = 2.0 * h
    out = np.full(h.shape, "n/a", dtype=object)
    ok = np.isfinite(h)
    out[ok & (thick < thin_m)] = "thin"
    out[ok & (thick >= thin_m) & (thick < large_m)] = "medium"
    out[ok & (thick >= large_m)] = "large"
    return out


def _split_view(pred, true, iv, rt, W, tol, mask=None) -> dict[str, Any]:
    m = np.ones(true.shape, bool) if mask is None else mask
    return {
        "all": ray_metrics(pred, true, rt, W, tol, m),
        "in_view": ray_metrics(pred, true, rt, W, tol, m & iv),
        "out_of_view": ray_metrics(pred, true, rt, W, tol, m & ~iv),
    }


def evaluate_arrays(
    pred: np.ndarray,
    conf: np.ndarray | None,
    true: np.ndarray,
    in_view: np.ndarray,
    *,
    range_train: float,
    world_scale: float,
    tol_m: float,
    hit_min_half_m: np.ndarray | None = None,
    warning_s: np.ndarray | None = None,
    maps: np.ndarray | None = None,
    train_mean: np.ndarray | None = None,
    thin_m: float = THIN_M,
    large_m: float = LARGE_M,
) -> dict[str, Any]:
    """The whole report body for ``(N, R)`` prediction / truth arrays."""
    pred = np.asarray(pred, np.float64)
    true = np.asarray(true, np.float64)
    iv = np.asarray(in_view, bool)
    rt, W, tol = float(range_train), float(world_scale), float(tol_m)
    n, r = true.shape
    rep: dict[str, Any] = {
        "frames": int(n),
        "rays": int(r),
        "aggregate": _split_view(pred, true, iv, rt, W, tol),
    }
    rep["per_ray"] = [
        {
            "ray": j,
            "in_view_share": float(iv[:, j].mean()) if n else None,
            **ray_metrics(pred[:, j], true[:, j], rt, W, tol),
        }
        for j in range(r)
    ]
    db = distance_bin(true)
    names = ["0-25%", "25-50%", "50-75%", "75-100%", "clear"]
    rep["distance_bins"] = []
    correct = np.abs(pred - true) * rt < tol
    for b, name in enumerate(names):
        m = db == b
        row = {
            "bin": name,
            "train_m": [DIST_BINS[b] * rt, DIST_BINS[b + 1] * rt] if b < 4 else None,
            "liftoff_m": [DIST_BINS[b] * rt * W, DIST_BINS[b + 1] * rt * W] if b < 4 else None,
            **_split_view(pred, true, iv, rt, W, tol, m),
        }
        if conf is not None:
            row["calibration"] = calibration(np.asarray(conf)[m], correct[m])
        rep["distance_bins"].append(row)
    if conf is not None:
        rep["calibration"] = calibration(conf, correct)
    if maps is not None:
        maps = np.asarray(maps)
        rep["per_map"] = {
            str(mp): _split_view(
                pred, true, iv, rt, W, tol, np.broadcast_to((maps == mp)[:, None], true.shape)
            )
            for mp in sorted(set(maps.tolist()))
        }
    if hit_min_half_m is not None:
        sb = size_bin(hit_min_half_m, thin_m, large_m)
        rep["size_bins"] = {
            name: {"thickness_m": edges, **ray_metrics(pred, true, rt, W, tol, sb == name)}
            for name, edges in (
                ("thin", [0, thin_m]),
                ("medium", [thin_m, large_m]),
                ("large", [large_m, None]),
                ("n/a", None),
            )
        }
    if warning_s is not None:
        w = np.asarray(warning_s, np.float64)
        fin = w[np.isfinite(w)]
        short = np.isfinite(w) & (w < WARNING_SHORT_S)
        rep["warning"] = {
            "finite_frames": int(len(fin)),
            "p05": float(np.percentile(fin, 5)) if len(fin) else None,
            "p50": float(np.percentile(fin, 50)) if len(fin) else None,
            "p95": float(np.percentile(fin, 95)) if len(fin) else None,
            "share_lt_0_3": float(np.mean(fin < 0.3)) if len(fin) else None,
            "share_lt_0_5": float(np.mean(fin < 0.5)) if len(fin) else None,
            "short_frames": int(short.sum()),
            "short": ray_metrics(
                pred, true, rt, W, tol, np.broadcast_to(short[:, None], true.shape)
            ),
        }
    base = {"always_clear": _split_view(np.ones_like(true), true, iv, rt, W, tol)}
    if train_mean is not None:
        tm = np.broadcast_to(np.asarray(train_mean, np.float64)[None], true.shape)
        base["train_mean"] = _split_view(tm, true, iv, rt, W, tol)
    rep["baselines"] = base
    return rep


# ----------------------------------------------------------------------------- run


def predict(net: Any, data: Any, device: str, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """``(free, conf)`` ``(N, R)`` of a model over :class:`~drone_fly.eyes.dataset.EyesSamples`."""
    import torch

    from drone_fly.eyes.train import _batch

    net.eval()
    fr, cf = [], []
    with torch.no_grad():
        for a in range(0, len(data), batch):
            idx = np.arange(a, min(a + batch, len(data)))
            b = _batch(data, idx, device)
            f, lg = net(b["frames"], b["uv"], b["rays_cam"], b["in_view"], b["ego"], b["hist_dt"])
            fr.append(f.float().cpu().numpy())
            cf.append(torch.sigmoid(lg).float().cpu().numpy())
    r = int(data.data["free"].shape[1]) if len(data) else 0
    return (
        np.concatenate(fr) if fr else np.zeros((0, r)),
        np.concatenate(cf) if cf else np.zeros((0, r)),
    )


def run_eval(
    weights: str | Path,
    dataset: str | Path,
    *,
    split: str = "held-out",
    tol_m: float | None = None,
    thin_m: float = THIN_M,
    large_m: float = LARGE_M,
    device: str = "auto",
    out: str | Path | None = None,
    batch: int = 64,
    max_samples: int | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    from drone_fly.eyes.dataset import SAMPLES_FILE, SPLITS, EyesSamples, load_manifest
    from drone_fly.eyes.infer import load_eyes_weights, resolve_device
    from drone_fly.eyes.model import build_from_meta

    if split == "train":
        raise HarnessConfigError("eyes eval refuses the train split (evaluate val or held-out)")
    if split not in SPLITS:
        raise HarnessConfigError(f"--split must be val or held-out, got {split!r}")
    meta, state = load_eyes_weights(weights)
    man = load_manifest(dataset)
    trained = set((meta.get("split_maps") or {}).get("train") or [])
    overlap = trained & set(man["split_maps"].get(split) or [])
    if overlap:
        raise HarnessConfigError(
            f"eyes eval refuses maps the weights trained on: {sorted(overlap)}"
        )
    if meta.get("label_settings_sha256") != man.get("label_settings_sha256"):
        raise HarnessConfigError("the dataset's label settings differ from the weights' (relabel)")
    dev = resolve_device(device)
    net = build_from_meta(meta)
    net.load_state_dict(state)
    net.to(dev)
    data = EyesSamples(dataset, split, max_samples=max_samples, cache_root=cache_root)
    rt = float(meta["ray_range"]) * float(meta["ray_range_scale"])
    W = float(meta["world_scale"])
    tol = float(tol_m if tol_m is not None else meta.get("tol_m", 0.5))
    with np.load(Path(dataset) / SAMPLES_FILE, allow_pickle=False) as z:
        tr = z["split"] == SPLITS.index("train")
        train_mean = z["free"][tr].mean(axis=0) if tr.any() else None
    if len(data):
        pred, conf = predict(net, data, dev, batch)
        d = data.data
        body = evaluate_arrays(
            pred,
            conf,
            d["free"],
            d["in_view"],
            range_train=rt,
            world_scale=W,
            tol_m=tol,
            hit_min_half_m=d["hit_min_half_m"],
            warning_s=d["warning_s"],
            maps=data.maps,
            train_mean=train_mean,
            thin_m=thin_m,
            large_m=large_m,
        )
    else:
        body = {"frames": 0, "note": f"no frames in the {split} split"}
    rep = {
        "format": 1,
        "weights": str(Path(weights).resolve()),
        "dataset": man["id"],
        "split": split,
        "maps": man["split_maps"].get(split) or [],
        "tol_m": tol,
        "range_train_m": rt,
        "range_liftoff_m": rt * W,
        "world_scale": W,
        "targets_note": "targets are recorded, not gating: this is a measured baseline (AC6)",
        **body,
    }
    out_dir = Path(out) if out is not None else Path(weights).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{split}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / f"eval_{split}.md").write_text(eval_markdown(rep), encoding="utf-8")
    rep["out_dir"] = str(out_dir)
    return rep


def _f(v: Any, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _row(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m.get('n', 0)} | {m.get('n_hit', '-')} | "
        f"{_f(m.get('median_abs_err_liftoff_m'), 2)} | {_f(m.get('miss_rate'))} | "
        f"{_f(m.get('false_alarm_rate_clear'))} | {_f(m.get('false_alarm_rate_all'))} |"
    )


def eval_markdown(rep: dict[str, Any]) -> str:
    head = "| group | rays | hits | median err (Liftoff m) | miss | FA (clear) | FA (all) |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [
        f"# Eyes accuracy: {rep['split']} ({', '.join(rep['maps']) or 'no maps'})",
        "",
        f"Dataset `{rep['dataset']}`, tol {rep['tol_m']} training m, range "
        f"{rep['range_liftoff_m']:g} Liftoff m. {rep['targets_note']}.",
        "",
    ]
    if not rep.get("frames"):
        return "\n".join([*lines, rep.get("note", "no frames"), ""])
    ag = rep["aggregate"]
    lines += ["## Aggregate", "", head, sep]
    for k in ("all", "in_view", "out_of_view"):
        lines.append(_row(f"model {k}", ag[k]))
    for b, v in rep["baselines"].items():
        for k in ("all", "in_view", "out_of_view"):
            lines.append(_row(f"{b} {k}", v[k]))
    lines += ["", "## Distance bins (truth)", "", head, sep]
    for row in rep["distance_bins"]:
        for k in ("in_view", "out_of_view"):
            lines.append(_row(f"{row['bin']} {k}", row[k]))
    if "calibration" in rep:
        lines += [
            "",
            f"Calibration ECE {_f(rep['calibration']['ece'])}; per distance bin: "
            + ", ".join(
                f"{r['bin']} {_f((r.get('calibration') or {}).get('ece'))}"
                for r in rep["distance_bins"]
            ),
        ]
    if rep.get("per_map"):
        lines += ["", "## Per map", "", head, sep]
        for mp, v in rep["per_map"].items():
            lines.append(_row(mp, v["all"]))
    if rep.get("size_bins"):
        lines += ["", "## Obstacle size (thickness, Liftoff m)", "", head, sep]
        for name, v in rep["size_bins"].items():
            lines.append(_row(name, v))
    if rep.get("warning"):
        w = rep["warning"]
        lines += [
            "",
            f"Warning time: p05 {_f(w['p05'], 2)} s, p50 {_f(w['p50'], 2)} s, "
            f"< 0.3 s {_f(w['share_lt_0_3'])}, < 0.5 s {_f(w['share_lt_0_5'])}; miss rate "
            f"on frames with warning < 0.5 s: {_f(w['short'].get('miss_rate'))}.",
        ]
    return "\n".join(lines) + "\n"
