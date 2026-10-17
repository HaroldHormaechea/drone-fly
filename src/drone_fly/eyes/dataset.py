"""AC3: the eyes dataset, split by map (``drone-fly eyes dataset``; offline).

Over a captures root (UC71 layout, :func:`~drone_fly.eyes.capture_meta.discover_captures`) and
their eyes labels (:mod:`drone_fly.eyes.labels`; ``--label-missing`` labels the missing / stale):

* **Split** from the track catalogue's ``maps.<map>.split`` -- never from a capture's
  ``meta.json`` -- for the map of the capture's catalogue entry. A capture whose meta map differs
  from the catalogue entry's map is refused. :func:`check_leakage` raises
  :class:`~drone_fly.tracks.catalogue.TrackLeakageError` if a map lands in two splits or a non-train
  map in ``train``.
* **Exclusions** (reasons in the manifest): no meta, no labels, unverified alignment (every
  split), a camera uptilt more than ``--uptilt-tol-deg`` from the preset's (kept on train maps
  only, flagged), a label settings hash that differs from the first included capture's (refused:
  one dataset = one label hash).
* **Samples** are frames with ``K - 1`` history frames (:func:`~drone_fly.eyes.history.
  select_history`, time stride ``--stride-s``); others are dropped as ``no-history``.
* **Frame cache** (:func:`ensure_cache`): one uint8 ``(rows, 3, H, W)`` ``.npy`` memmap per
  capture under ``<cache root>/<key>/`` (key = sha of capture sha + input size + mask hash),
  filled by :func:`~drone_fly.eyes.model.prepare_frame` and resumable (``<name>.progress.json``).
  A projected-size guard refuses when free disk < projected + margin.

Writes ``samples.npz`` + ``manifest.json`` / ``manifest.md`` to
``<data root>/datasets/eyes/<id>/`` (``id = eyes-<UTC stamp>-<label hash[:8]>``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.eyes.capture_meta import capture_sha, discover_captures, read_capture_meta
from drone_fly.eyes.history import default_tol, history_dt, select_history
from drone_fly.eyes.labels import (
    AUX_KINDS,
    LABELS_FILE,
    default_out,
    labels_root_default,
    read_labels_meta,
    run_label,
)
from drone_fly.eyes.model import DT_S, INPUT_HEIGHT, INPUT_WIDTH, K_FRAMES
from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "held-out")
SAMPLES_FILE = "samples.npz"
MANIFEST = "manifest.json"
MANIFEST_MD = "manifest.md"
UPTILT_TOL_DEG = 2.0
DISK_MARGIN_GB = 1.0
#: Label arrays copied per sample into ``samples.npz``.
SAMPLE_FIELDS = (
    "free", "hit", "dist_m", "hit_min_half_m", "aux_free", "aux_kind", "rays_cam", "uv",
    "in_view", "ego",
)  # fmt: skip
FRAME_FIELDS = ("warning_s", "speed_mps")


def datasets_root() -> Path:
    from drone_fly import paths

    return paths.data_root() / "datasets" / "eyes"


def cache_root_default() -> Path:
    from drone_fly import paths

    return paths.data_root() / "cache" / "eyes"


def check_leakage(assign: dict[str, str], catalogue_splits: dict[str, str]) -> None:
    """``assign``: map -> dataset split. Raise on a non-train map in train or a mismatch."""
    from drone_fly.tracks.catalogue import TrackLeakageError

    for m, split in assign.items():
        cs = catalogue_splits.get(m)
        if split == "train" and cs != "train":
            raise TrackLeakageError(f"map {m} (catalogue split {cs}) would be in the train split")
        if cs is not None and split != cs:
            raise TrackLeakageError(f"map {m} is {cs} in the catalogue but assigned to {split}")


def masks_hash(masks: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps(masks, sort_keys=True).encode()).hexdigest()


def cache_key(cap_sha: str, width: int, height: int, mask_sha: str) -> str:
    return hashlib.sha256(f"{cap_sha}|{width}x{height}|{mask_sha}".encode()).hexdigest()[:16]


def _pct(a: np.ndarray) -> dict[str, float | None]:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"p05": None, "p50": None, "p95": None}
    return {f"p{q:02d}": float(np.percentile(a, q)) for q in (5, 50, 95)}


def warning_stats(w: np.ndarray) -> dict[str, Any]:
    w = np.asarray(w, np.float64)
    fin = w[np.isfinite(w)]
    return {
        **_pct(fin),
        "finite": int(len(fin)),
        "share_lt_0_3": float(np.mean(fin < 0.3)) if len(fin) else None,
        "share_lt_0_5": float(np.mean(fin < 0.5)) if len(fin) else None,
    }


# ----------------------------------------------------------------------------- frame cache


def _free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return int(shutil.disk_usage(p).free)


def cache_path(entry: dict[str, Any], cache_root: Path) -> Path:
    return Path(cache_root) / entry["cache_key"] / f"{entry['name']}.npy"


def projected_bytes(entries: list[dict[str, Any]], cache_root: Path, width: int, height: int):
    """Bytes the not-yet-complete caches still need."""
    total = 0
    for e in entries:
        p = cache_path(e, cache_root)
        if p.is_file() and _progress(p) >= e["rows"]:
            continue
        total += e["rows"] * 3 * width * height
    return total


def _progress(npy: Path) -> int:
    f = npy.with_suffix(".progress.json")
    if not f.is_file() or not npy.is_file():
        return 0
    try:
        return int(json.loads(f.read_text(encoding="utf-8")).get("rows_done", 0))
    except (OSError, ValueError):
        return 0


def ensure_cache(
    manifest: dict[str, Any],
    *,
    cache_root: str | Path | None = None,
    disk_margin_gb: float = DISK_MARGIN_GB,
    device: str = "cpu",
) -> dict[str, Path]:
    """Build (or resume) every included capture's frame cache; returns ``name -> .npy``."""
    from drone_fly.eyes.model import prepare_frame
    from drone_fly.harness.offline.evaluate import load_image

    inp = manifest["input"]
    w, h = int(inp["width"]), int(inp["height"])
    masks = manifest["masks"]
    root = Path(cache_root or manifest.get("cache_root") or cache_root_default())
    entries = [e for e in manifest["captures"] if e.get("included")]
    need = projected_bytes(entries, root, w, h)
    free = _free_bytes(root)
    if need and free < need + disk_margin_gb * 1e9:
        raise HarnessConfigError(
            f"frame cache needs {need / 1e9:.2f} GB + {disk_margin_gb:g} GB margin but only "
            f"{free / 1e9:.2f} GB are free under {root}"
        )
    out = {}
    for e in entries:
        p = cache_path(e, root)
        out[e["name"]] = p
        done = _progress(p)
        if done >= e["rows"]:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.is_file() and done > 0:
            arr = np.load(p, mmap_mode="r+")
            if arr.shape != (e["rows"], 3, h, w):
                done, arr = 0, None
        else:
            arr = None
        if arr is None:
            arr = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8,
                                            shape=(e["rows"], 3, h, w))  # fmt: skip
        with np.load(Path(e["labels_dir"]) / LABELS_FILE, allow_pickle=False) as z:
            files = z["frame_file"]
        log.info("caching %s: rows %d..%d", e["name"], done, e["rows"])
        prog = p.with_suffix(".progress.json")
        for r in range(done, e["rows"]):
            img = load_image(Path(e["capture"]) / str(files[r]))
            arr[r] = prepare_frame(img, w, h, masks, device).cpu().numpy()
            if (r + 1) % 64 == 0 or r + 1 == e["rows"]:
                arr.flush()
                prog.write_text(json.dumps({"rows_done": r + 1}), encoding="utf-8")
        del arr
    return out


# ----------------------------------------------------------------------------- build


def _catalogue() -> Any:
    from drone_fly.tracks import catalogue as cat

    return cat.load_named("liftoff")


def hit_bias_select(
    hit_rows: np.ndarray, hit_frac: float, seed: int
) -> np.ndarray:
    """Indices (into ``hit_rows``) of a hit-biased subset reaching ~``hit_frac`` hit-bearing frames.

    ``hit_rows`` is ``(M,)`` bool: whether each candidate frame carries any hit ray. All hit-bearing
    frames are kept; clear frames are subsampled (seeded, without replacement) so the kept set's
    hit-bearing fraction is about ``hit_frac``. No-op (returns all) when already at/above the target
    or there are no hits to bias toward. Never duplicates frames (filter, not oversample).
    """
    has_hit = np.asarray(hit_rows, bool)
    m = len(has_hit)
    n_hit = int(has_hit.sum())
    if m == 0 or n_hit == 0 or n_hit >= hit_frac * m:
        return np.arange(m)
    n_clear_keep = int(round(n_hit * (1.0 - hit_frac) / hit_frac))
    clear_idx = np.flatnonzero(~has_hit)
    n_clear_keep = min(n_clear_keep, len(clear_idx))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(clear_idx, size=n_clear_keep, replace=False) if n_clear_keep else []
    mask = has_hit.copy()
    mask[np.asarray(chosen, np.int64)] = True
    return np.flatnonzero(mask)


def build_dataset(
    captures_root: str | Path | None = None,
    *,
    labels_root: str | Path | None = None,
    out: str | Path | None = None,
    label_missing: bool = False,
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
    k_frames: int = K_FRAMES,
    stride_s: float = DT_S,
    hist_tol_s: float | None = None,
    uptilt_tol_deg: float = UPTILT_TOL_DEG,
    build_cache: bool = False,
    cache_root: str | Path | None = None,
    disk_margin_gb: float = DISK_MARGIN_GB,
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    catalogue: Any = None,
    label_kwargs: dict[str, Any] | None = None,
    hit_frac: float | None = None,
    mining_seed: int = 0,
) -> dict[str, Any]:
    """Build the dataset (see the module docstring); returns the manifest.

    UC-83 AC2: ``hit_frac`` (default ``None`` => off, byte-identical to v1) applies an opt-in
    hit-biased frame filter to the **train split only** (val / held-out are untouched so evaluation
    stays comparable): clear frames are subsampled so train hit-bearing frames reach ~``hit_frac``.
    The resulting hit-ray fraction is recorded in the manifest's ``mining`` block and ``splits``.
    """
    from drone_fly import paths
    from drone_fly.eyes.fov import preset_view
    from drone_fly.tracks.catalogue import sha256_file

    if hit_frac is not None and not (0.0 < float(hit_frac) < 1.0):
        raise HarnessConfigError(f"--hit-frac must be in (0, 1), got {hit_frac!r}")
    hist_tol_s = default_tol(stride_s) if hist_tol_s is None else float(hist_tol_s)
    croot = Path(captures_root or paths.captures_root())
    lroot = Path(labels_root or labels_root_default())
    view = preset_view(preset, config, harness)
    cat = catalogue or _catalogue()
    map_split = {m: (v or {}).get("split") for m, v in cat.maps().items()}
    entries: list[dict[str, Any]] = []
    label_hash: str | None = None
    for cdir in discover_captures(croot):
        e: dict[str, Any] = {"name": cdir.name, "capture": str(cdir), "included": False,
                             "flags": []}  # fmt: skip
        entries.append(e)
        cm = read_capture_meta(cdir)
        if cm is None:
            e["excluded"] = "no-meta"
            continue
        ldir = default_out(cdir.name, lroot)
        if label_missing:
            res = run_label(cdir, preset=preset, config=config, harness=harness,
                            labels_root=lroot, **(label_kwargs or {}))  # fmt: skip
            if res.get("skipped"):
                e["excluded"] = f"label-skipped: {res['skipped']}"
                continue
        lm = read_labels_meta(ldir)
        if lm is None:
            e["excluded"] = "no-labels"
            continue
        if lm.get("capture_sha256") != capture_sha(cdir):
            e["excluded"] = "stale-labels (capture changed)"
            continue
        entry = cat.by_id(lm["catalogue_id"])
        if cm.map is not None and cm.map != entry["map"]:
            raise HarnessConfigError(
                f"{cdir}: meta map {cm.map} != catalogue entry {entry['id']} map {entry['map']}"
            )
        split = map_split.get(entry["map"])
        if split not in SPLITS:
            e["excluded"] = f"map {entry['map']} has no catalogue split"
            continue
        e.update({"map": entry["map"], "split": split, "catalogue_id": entry["id"],
                  "labels_dir": str(ldir), "capture_sha256": lm["capture_sha256"],
                  "label_settings_sha256": lm["settings_sha256"]})  # fmt: skip
        if not cm.alignment_verified:
            e["excluded"] = "alignment-unverified"
            continue
        up = float((lm.get("camera") or {}).get("uptilt_deg", view.camera.uptilt_deg))
        if abs(up - view.camera.uptilt_deg) > uptilt_tol_deg:
            if split != "train":
                e["excluded"] = f"uptilt {up:g} deg vs preset {view.camera.uptilt_deg:g}"
                continue
            e["flags"].append("uptilt-mismatch-train-only")
        if float(lm["settings"]["ego_dt_s"]) != float(stride_s):
            raise HarnessConfigError(
                f"{ldir}: labels' ego stride {lm['settings']['ego_dt_s']} s != --stride-s "
                f"{stride_s} (relabel or use the same stride)"
            )
        if label_hash is None:
            label_hash = lm["settings_sha256"]
        elif lm["settings_sha256"] != label_hash:
            raise HarnessConfigError(
                f"{ldir}: label settings hash {lm['settings_sha256'][:12]} differs from "
                f"{label_hash[:12]} (one dataset = one label setting; relabel)"
            )
        e["included"] = True
    inc = [e for e in entries if e["included"]]
    if not inc:
        reasons = [(e["name"], e.get("excluded")) for e in entries]
        raise HarnessConfigError(f"no usable labelled capture under {croot}: {reasons}")
    check_leakage({e["map"]: e["split"] for e in inc}, map_split)

    # ---- samples
    cols: dict[str, list[np.ndarray]] = {k: [] for k in (*SAMPLE_FIELDS, *FRAME_FIELDS)}
    idx_cap, idx_row, idx_hist, idx_split, idx_hdt = [], [], [], [], []
    drops: Counter[str] = Counter()
    for ci, e in enumerate(entries):
        if not e["included"]:
            continue
        with np.load(Path(e["labels_dir"]) / LABELS_FILE, allow_pickle=False) as z:
            lab = {k: z[k] for k in z.files}
        n = len(lab["stamp"])
        e["rows"] = int(n)
        e["cache_key"] = cache_key(e["capture_sha256"], width, height, masks_hash(view.masks))
        keep: list[int] = []
        hists: list[np.ndarray] = []
        hdts: list[np.ndarray] = []
        for i in range(n):
            hist = select_history(lab["stamp"], i, k_frames, stride_s, hist_tol_s, lab["segment"])
            if hist is None:
                drops["no-history"] += 1
                continue
            keep.append(i)
            hists.append(hist)
            hdts.append(history_dt(lab["stamp"], i, hist))
        if hit_frac is not None and e["split"] == "train" and keep:
            keep_rows = np.asarray(keep, np.int64)
            has_hit = np.asarray(lab["hit"][keep_rows], bool).any(axis=1)
            sel = hit_bias_select(has_hit, float(hit_frac), mining_seed + ci)
            drops["hit-bias-filtered"] += len(keep) - len(sel)
            keep = [keep[j] for j in sel]
            hists = [hists[j] for j in sel]
            hdts = [hdts[j] for j in sel]
        idx_hist.extend(hists)
        idx_hdt.extend(hdts)
        keep_a = np.asarray(keep, np.int64)
        e["samples"] = int(len(keep_a))
        idx_cap += [ci] * len(keep_a)
        idx_row.append(keep_a)
        idx_split += [SPLITS.index(e["split"])] * len(keep_a)
        for k in (*SAMPLE_FIELDS, *FRAME_FIELDS):
            cols[k].append(lab[k][keep_a])
    samples = {
        "capture": np.asarray(idx_cap, np.int32),
        "row": np.concatenate(idx_row) if idx_row else np.zeros(0, np.int64),
        "hist": np.asarray(idx_hist, np.int64).reshape(-1, max(k_frames - 1, 0)),
        "hist_dt": np.asarray(idx_hdt, np.float32).reshape(-1, max(k_frames - 1, 0)),
        "split": np.asarray(idx_split, np.int8),
        **{k: np.concatenate(v) for k, v in cols.items()},
    }

    # ---- write
    created = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    ds_id = f"eyes-{created}-{(label_hash or 'none')[:8]}"
    out_dir = Path(out) if out is not None else datasets_root() / ds_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (SAMPLES_FILE + ".tmp.npz")
    np.savez(tmp, **samples)
    tmp.replace(out_dir / SAMPLES_FILE)
    dataset_sha = sha256_file(out_dir / SAMPLES_FILE)
    first = read_labels_meta(inc[0]["labels_dir"]) or {}
    splits = split_stats(samples, entries)
    manifest = {
        "format": 1,
        "id": ds_id,
        "created_utc": created,
        "dataset_sha256": dataset_sha,
        "label_settings_sha256": label_hash,
        "label_settings": first.get("settings"),
        "catalogue_sha256": sha256_file(cat.path) if cat.path is not None else None,
        "captures_root": str(croot),
        "labels_root": str(lroot),
        "cache_root": str(Path(cache_root or cache_root_default())),
        "input": {
            "width": int(width),
            "height": int(height),
            "k_frames": int(k_frames),
            "dt_s": float(stride_s),
            "hist_tol_s": float(hist_tol_s),
        },  # fmt: skip
        "preset": view.as_dict(),
        "masks": view.masks,
        "masks_sha256": masks_hash(view.masks),
        "uptilt_tol_deg": float(uptilt_tol_deg),
        "split_maps": {s: sorted({e["map"] for e in inc if e["split"] == s}) for s in SPLITS},
        "captures": entries,
        "drop_reasons": dict(drops),
        "splits": splits,
        "mining": {
            "hit_frac_target": float(hit_frac) if hit_frac is not None else None,
            "mining_seed": int(mining_seed),
            "applied": hit_frac is not None,
            # resulting train-split fractions after any hit-biased filtering (AC2 requirement):
            "train_rays_hit_share": splits.get("train", {}).get("rays_hit_share"),
            "train_frames_with_hit_share": splits.get("train", {}).get("frames_with_hit_share"),
        },
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / MANIFEST_MD).write_text(manifest_markdown(manifest), encoding="utf-8")
    if build_cache:
        ensure_cache(manifest, cache_root=cache_root, disk_margin_gb=disk_margin_gb)
    manifest["out_dir"] = str(out_dir)
    return manifest


def split_stats(samples: dict[str, np.ndarray], entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Per split and map: counts, hit shares, label vs aux kinds and warning-time stats."""
    out: dict[str, Any] = {}
    for si, s in enumerate(SPLITS):
        sel = samples["split"] == si
        maps: dict[str, Any] = {}
        for m in sorted({entries[c]["map"] for c in samples["capture"][sel]}):
            ms = sel & np.isin(samples["capture"],
                               [i for i, e in enumerate(entries) if e.get("map") == m])  # fmt: skip
            maps[m] = _group_stats(samples, ms)
        out[s] = {**_group_stats(samples, sel), "maps": maps}
    return out


def _group_stats(samples: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, Any]:
    n = int(sel.sum())
    if not n:
        return {"samples": 0}
    hit = samples["hit"][sel]
    kinds = samples["aux_kind"][sel]
    return {
        "samples": n,
        "rays_hit_share": float(hit.mean()),
        "frames_with_hit_share": float(hit.any(axis=1).mean()),
        "in_view_share": float(samples["in_view"][sel].mean()),
        "label_kinds": {"obstacle": int(hit.sum()), "none": int((~hit).sum())},
        "aux_kinds": {k: int((kinds == i).sum()) for i, k in enumerate(AUX_KINDS)},
        "warning_s": warning_stats(samples["warning_s"][sel]),
    }


def manifest_markdown(m: dict[str, Any]) -> str:
    lines = [
        f"# Eyes dataset `{m['id']}`",
        "",
        f"Label settings sha `{(m['label_settings_sha256'] or '')[:12]}`, samples sha "
        f"`{m['dataset_sha256'][:12]}`, input {m['input']['width']}x{m['input']['height']} "
        f"K={m['input']['k_frames']} stride {m['input']['dt_s']} s.",
        "",
        "| split | maps | samples | ray hit share | warning p50 (s) | warning < 0.3 s | < 0.5 s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for s in SPLITS:
        st = m["splits"][s]
        w = st.get("warning_s") or {}
        lines.append(
            f"| {s} | {', '.join(m['split_maps'][s]) or '-'} | {st['samples']} | "
            f"{st.get('rays_hit_share', 0):.3f} | {w.get('p50')} | {w.get('share_lt_0_3')} | "
            f"{w.get('share_lt_0_5')} |"
        )
    lines += ["", "## Captures", "", "| capture | map | split | samples | status |",
              "|---|---|---|---:|---|"]  # fmt: skip
    for e in m["captures"]:
        status = "included" if e["included"] else f"excluded: {e.get('excluded')}"
        if e.get("flags"):
            status += f" ({', '.join(e['flags'])})"
        lines.append(f"| {e['name']} | {e.get('map', '-')} | {e.get('split', '-')} | "
                     f"{e.get('samples', '-')} | {status} |")  # fmt: skip
    lines += ["", f"Dropped samples: {m['drop_reasons']}", ""]
    return "\n".join(lines)


def load_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    p = Path(dataset_dir) / MANIFEST
    if not p.is_file():
        raise HarnessConfigError(f"{dataset_dir} is not an eyes dataset (no {MANIFEST})")
    return json.loads(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- access


class EyesSamples:
    """One split of a dataset: label arrays + memmapped frame caches (CPU, numpy)."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        max_samples: int | None = None,
        cache_root: str | Path | None = None,
        build_cache: bool = True,
    ) -> None:
        self.dir = Path(dataset_dir)
        self.manifest = load_manifest(self.dir)
        z = np.load(self.dir / SAMPLES_FILE, allow_pickle=False)
        sel = np.flatnonzero(z["split"] == SPLITS.index(split))
        if max_samples is not None:
            sel = sel[: int(max_samples)]
        self.split = split
        self.data = {k: z[k][sel] for k in z.files}
        caps = self.manifest["captures"]
        if build_cache:
            paths = ensure_cache(self.manifest, cache_root=cache_root)
        else:
            root = Path(cache_root or self.manifest["cache_root"])
            paths = {e["name"]: cache_path(e, root) for e in caps if e.get("included")}
        self.frames = {}
        for ci in np.unique(self.data["capture"]):
            self.frames[int(ci)] = np.load(paths[caps[int(ci)]["name"]], mmap_mode="r")
        self.maps = np.array([caps[int(c)].get("map", "") for c in self.data["capture"]])

    def __len__(self) -> int:
        return int(len(self.data["row"]))

    def frames_for(self, idx: np.ndarray) -> np.ndarray:
        """``(B, 3K, H, W)`` uint8: current frame then the history frames."""
        out = []
        for i in idx:
            f = self.frames[int(self.data["capture"][i])]
            rows = [int(self.data["row"][i]), *[int(r) for r in self.data["hist"][i]]]
            out.append(np.concatenate([f[r] for r in rows], axis=0))
        return np.stack(out)
