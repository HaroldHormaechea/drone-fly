"""Multi-map gate dataset (UC71 AC6): assembly, split enforcement, manifest, frame cache, G1 report.

:func:`assemble` turns catalogue-linked captures (``meta.json``, see
:mod:`~drone_fly.harness.offline.capture_index`) and their label folders into one dataset:

* **Split routing** from the catalogue at run time (never from ``meta.split``): train-map
  captures -> ``train``; val-map captures -> ``selection`` (model selection only); a held-out-map
  capture raises :class:`TrackLeakageError` - it is never silently filtered.
* **Refusals** (with a reason, the capture is left out): no / invalid ``meta.json`` (no catalogue
  id), alignment not verified, a track with unknown items (gate coverage), no labels, labels made
  with another gate table, and a **non-preset camera on a val map** (``preset_uptilt_deg``; ADR
  0010: another drone's recordings, e.g. the Borrum 180 X at uptilt 30, add view diversity on train
  maps only - selection and held-out evaluation stay on the deployed drone's camera).
* **Protected windows.** Frames inside a held-out window recorded by ``--protect-holdout-of``
  weights (default v1) are never loaded; the window is recorded so the new weights keep it held
  out (``detector_noise.heldout_window``, the Infinity Loop regression in ``eval --compare``).
* Per frame: the static trim (``capture.trim_static``), then ``frame_stride`` and
  ``max_frames_per_capture`` (evenly spaced).
* **Balanced sampling.** Each training frame is keyed by ``(map, class)`` of its nearest in-range
  gate (``(map, none)`` without one); its weight is ``1 / count(key)``, so every map x class pair
  contributes equally per epoch.

The **manifest** (``data_manifest.json``) lists every capture (name, catalogue id, map, split role,
capture sha256, counts, protected windows) and the catalogue sha256, gate-types hash, git head and
hyper-parameters; its first 8 hex characters name the frame cache
``artifacts/harness/liftoff/datasets/<manifest-sha8>/`` (masked + resized frames as at runtime, a
``uint8`` memmap, so 20k frames do not need ~15 GB of RAM).

:func:`dataset_report` counts in-range gate-frames (4-25 m, centre in the image, unmasked, not
flagged occluded) by split x map x type with the 4-8 m bin separate, and :func:`g1_verdict` applies
the data-sufficiency gate G1 (ADR 0010).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.errors import HarnessConfigError

ROLE_TRAIN, ROLE_SELECTION = "train", "selection"
SPLIT_ROLE = {"train": ROLE_TRAIN, "val": ROLE_SELECTION}
IN_RANGE_M = (4.0, 25.0)
NEAR_BIN_M = (4.0, 8.0)
RANGE_BINS = ((4.0, 8.0), (8.0, 15.0), (15.0, 25.0))
#: v1's protected held-out window: in-range arch gate-frames recorded in UC70 (ADR 0009).
PROTECTED_IN_RANGE = {"2026-10-11-infinity-manual": 148}
#: G1 thresholds (plan, AC6 / ADR 0010).
G1 = {
    "train_frames": 3000,
    "train_maps": 5,
    "train_near_frames": 300,
    "val_maps": 2,
    "val_frames_per_map": 300,
    "heldout_maps": 2,
    "heldout_type_frames": 30,
}


class TrackLeakageError(HarnessConfigError):
    """A held-out-map capture was offered to training (never silently filtered)."""


def models_root() -> Path:
    root = find_repo_root()
    return (root if root is not None else Path.cwd()) / "artifacts" / "harness" / "liftoff"


#: uptilt difference (deg) still counted as the preset camera (the 20.03 drone file vs 20 preset)
CAMERA_TOL_DEG = 0.5


def same_camera(meta: dict[str, Any], preset_uptilt_deg: float) -> bool:
    """A capture's camera is the preset's (uptilt within :data:`CAMERA_TOL_DEG`). Pure."""
    return abs(float(meta["camera"]["uptilt_deg"]) - float(preset_uptilt_deg)) <= CAMERA_TOL_DEG


def labels_dir_for(capture_dir: Path, meta: dict[str, Any] | None) -> Path | None:
    """``labels/<Map>/<session>`` (new layout) or ``labels/<session>`` (legacy), if present."""
    base = models_root() / "labels"
    cands = []
    if meta:
        cands.append(base / meta["map"] / capture_dir.name)
    cands.append(base / capture_dir.name)
    return next((c for c in cands if (c / "labels_meta.json").is_file()), None)


def in_range(g: dict[str, Any], lo: float = IN_RANGE_M[0], hi: float = IN_RANGE_M[1]) -> bool:
    return (
        lo <= float(g["range_m"]) <= hi
        and bool(g["center_in_image"])
        and not g.get("center_masked")
        and not g.get("occluded")
    )


@dataclass
class DatasetItem:
    capture: str
    map: str
    role: str
    path: Path  # image file
    row: dict[str, Any]  # the label row
    weight: float = 1.0


@dataclass
class DatasetCapture:
    name: str
    path: Path
    meta: dict[str, Any] | None
    labels_dir: Path | None
    role: str | None  # train | selection | None (refused)
    reason: str | None = None
    sha256: str | None = None
    counts: dict[str, int] = field(default_factory=dict)
    protected: dict[str, Any] | None = None


@dataclass
class Dataset:
    captures: list[DatasetCapture]
    items: list[DatasetItem]
    params: dict[str, Any]
    catalogue_sha256: str
    gate_types_hash: str
    git_head: str | None

    @property
    def used(self) -> list[DatasetCapture]:
        return [c for c in self.captures if c.role is not None]

    @property
    def refused(self) -> list[DatasetCapture]:
        return [c for c in self.captures if c.role is None]

    def role_items(self, role: str) -> list[DatasetItem]:
        return [it for it in self.items if it.role == role]

    def manifest(self) -> dict[str, Any]:
        return {
            "format": 1,
            "catalogue_sha256": self.catalogue_sha256,
            "gate_types_hash": self.gate_types_hash,
            "git_head": self.git_head,
            "params": self.params,
            "captures": [
                {
                    "name": c.name,
                    "catalogue_id": c.meta["catalogue_id"],
                    "map": c.meta["map"],
                    "split_role": c.role,
                    "capture_sha256": c.sha256,
                    "counts": dict(c.counts),
                    "protected_window": c.protected,
                }
                for c in self.used
            ],
            "refused": [{"name": c.name, "reason": c.reason} for c in self.refused],
        }

    def manifest_sha(self) -> str:
        m = self.manifest()
        m.pop("git_head")
        return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def _catalogue_sha(catalogue: Any) -> str:
    from drone_fly.tracks.catalogue import sha256_file

    path = getattr(catalogue, "path", None)
    return sha256_file(path) if path is not None and Path(path).is_file() else "unknown"


def protected_windows(weights_meta: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """``{capture name: {t_start, t_end, frames}}`` held out by ``weights_meta``."""
    if not weights_meta:
        return {}
    n_hold = {c.get("name"): c.get("n_holdout") for c in weights_meta.get("captures") or []}
    out = {}
    for name, w in (weights_meta.get("holdout") or {}).items():
        if w:
            out[name] = {"t_start": float(w["t_start"]), "t_end": float(w["t_end"]),
                         "frames": n_hold.get(name)}  # fmt: skip
    return out


def read_weights_meta(path: str | Path | None) -> dict[str, Any] | None:
    """The ``meta`` of a weights file (``None`` for ``None``); missing file -> config error."""
    if path is None:
        return None
    from drone_fly.harness.errors import lazy_import
    from drone_fly.harness.liftoff.gate_learned import resolve_weights_path

    torch = lazy_import("torch", purpose="reading gate weights metadata")
    p = resolve_weights_path(path)
    try:
        return dict(torch.load(p, map_location="cpu", weights_only=False)["meta"])
    except Exception as exc:  # noqa: BLE001
        raise HarnessConfigError(f"cannot read gate weights {p}: {exc}") from exc


def _evenly(n: int, k: int | None) -> np.ndarray:
    if k is None or n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def _dominant_class(row: dict[str, Any]) -> str:
    gs = [g for g in row["gates"] if in_range(g)]
    if not gs:
        return "none"
    return str(min(gs, key=lambda g: float(g["range_m"]))["cls"])


def balanced_weights(items: list[DatasetItem]) -> None:
    """Set ``weight = 1 / count((map, dominant class))`` on every item, normalised to sum 1."""
    keys = [(it.map, _dominant_class(it.row)) for it in items]
    counts = Counter(keys)
    w = np.array([1.0 / counts[k] for k in keys], np.float64)
    w /= max(w.sum(), 1e-12)
    for it, v in zip(items, w, strict=True):
        it.weight = float(v)


def assemble(
    capture_dirs: list[Path],
    *,
    catalogue: Any,
    table: Any,
    not_gates: Any = (),
    trackdata_root: str | Path | None = None,
    protect: dict[str, Any] | None = None,
    frame_stride: int = 1,
    max_frames_per_capture: int | None = None,
    static_trim: bool = True,
    telemetry_cfg: dict[str, Any] | None = None,
    include_selection_in_train: bool = False,
    params: dict[str, Any] | None = None,
    preset_uptilt_deg: float | None = None,
) -> Dataset:
    """Assemble the dataset (see the module docstring). ``protect`` is a weights ``meta``."""
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.capture_index import (
        CaptureMetaError,
        capture_sha256,
        link_entry,
        read_meta,
    )
    from drone_fly.harness.offline.gate_train import is_held_out, load_labels
    from drone_fly.harness.offline.track_xml import parse_race_route, parse_track_items
    from drone_fly.harness.recording import git_head
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.gate_coverage import unknown_items

    if frame_stride < 1:
        raise HarnessConfigError("--frame-stride must be >= 1")
    root = cat.resolve_root(trackdata_root, catalogue.header)
    windows = protected_windows(protect)
    caps: list[DatasetCapture] = []
    items: list[DatasetItem] = []
    for d in capture_dirs:
        d = Path(d)
        try:
            meta = read_meta(d)
        except CaptureMetaError as exc:
            caps.append(DatasetCapture(d.name, d, None, None, None, f"no catalogue id: {exc}"))
            continue
        entry = link_entry(meta, catalogue)
        split = str(entry["split"])
        if split not in SPLIT_ROLE:
            raise TrackLeakageError(
                f"capture {d.name} is on {meta['map']} ({entry['id']}), a {split} map: held-out "
                "maps are never trained or selected on; remove it from --captures"
            )
        dc = DatasetCapture(d.name, d, meta, labels_dir_for(d, meta), None)
        caps.append(dc)
        if (
            split == "val"
            and preset_uptilt_deg is not None
            and not same_camera(meta, preset_uptilt_deg)
        ):
            dc.reason = (
                f"camera uptilt {meta['camera']['uptilt_deg']:g} is not the preset's "
                f"{preset_uptilt_deg:g}: other cameras are for train maps only"
            )
            continue
        if not meta["alignment"].get("verified"):
            dc.reason = "alignment not verified (run `harness captures verify`)"
            continue
        track = root / entry["track"]
        try:
            items_xml = parse_track_items(track)
            route = parse_race_route(root / entry["race"])
        except (HarnessConfigError, OSError) as exc:
            dc.reason = f"track files unavailable: {exc}"
            continue
        by_inst = {it.instance_id: it.item_id for it in items_xml if it.instance_id}
        route_ids = [by_inst[c] for c in (route.checkpoint_ids if route else ()) if c in by_inst]
        unknown = unknown_items(items_xml, table, not_gates, route_ids)
        if unknown:
            dc.reason = f"uncovered items {unknown} (gate-types file)"
            continue
        if dc.labels_dir is None:
            dc.reason = "no labels (run `harness label`)"
            continue
        labels = load_labels(dc.labels_dir)
        if labels.meta.get("gate_types_hash") != table.hash():
            dc.reason = "labels made with another gate table (re-run `harness label`)"
            continue
        dc.role = SPLIT_ROLE[split]
        if include_selection_in_train:
            dc.role = ROLE_TRAIN
        dc.sha256 = capture_sha256(d)
        counts: Counter[str] = Counter(labels=len(labels.rows))
        rows = labels.rows
        win = windows.get(d.name)
        if win is not None:
            prot = [r for r in rows if is_held_out(r["stamp"], win)]
            rows = [r for r in rows if not is_held_out(r["stamp"], win)]
            counts["protected"] = len(prot)
            dc.protected = {
                **win,
                "frames_present": len(prot),
                "in_range_gate_frames": sum(in_range(g) for r in prot for g in r["gates"]),
                "expected_in_range": PROTECTED_IN_RANGE.get(d.name),
            }
        if static_trim and telemetry_cfg is not None:
            cap = load_capture(d, telemetry_cfg, 1.0, float(labels.meta.get("lag_ms", 0.0)),
                               static_trim=True)  # fmt: skip
            keep = {str(f.path.relative_to(cap.root)).replace("\\", "/") for f in cap.frames}
            before = len(rows)
            rows = [r for r in rows if r["file"] in keep]
            counts["static"] = before - len(rows)
        rows = sorted(rows, key=lambda r: r["stamp"])
        strided = rows[::frame_stride]
        counts["stride"] = len(rows) - len(strided)
        pick = _evenly(len(strided), max_frames_per_capture)
        counts["cap"] = len(strided) - len(pick)
        counts["used"] = len(pick)
        dc.counts = dict(counts)
        cap_root = Path(labels.meta.get("capture") or d)
        if not cap_root.is_dir():
            cap_root = d
        for k in pick:
            r = strided[k]
            items.append(DatasetItem(d.name, meta["map"], dc.role, cap_root / r["file"], r))
    train = [it for it in items if it.role == ROLE_TRAIN]
    if train:
        balanced_weights(train)
    return Dataset(
        caps,
        items,
        dict(
            params or {},
            frame_stride=frame_stride,
            max_frames_per_capture=max_frames_per_capture,
            static_trim=static_trim,
        ),  # fmt: skip
        _catalogue_sha(catalogue),
        table.hash(),
        git_head(find_repo_root()),
    )


# ----------------------------------------------------------------------------- report / G1


def _bin(r: float) -> str | None:
    for lo, hi in RANGE_BINS:
        if lo <= r <= hi:
            return f"{lo:g}-{hi:g}"
    return None


def count_gate_frames(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """``{class: {"4-25": n, "4-8": n, "8-15": n, "15-25": n}}`` of in-range gate-frames."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for g in r["gates"]:
            if in_range(g):
                out[g["cls"]]["4-25"] += 1
                b = _bin(float(g["range_m"]))
                if b:
                    out[g["cls"]][b] += 1
    return {k: dict(v) for k, v in out.items()}


def dataset_report(
    ds: Dataset, heldout_rows: dict[str, list[dict[str, Any]]] | None = None
) -> dict[str, Any]:
    """In-range gate-frame counts by split x map x type (+ held-out maps, if their label rows are
    given as ``{map: rows}``), refusals, protected windows and the G1 verdict."""
    by: dict[str, dict[str, list]] = {ROLE_TRAIN: defaultdict(list),
                                      ROLE_SELECTION: defaultdict(list)}  # fmt: skip
    for it in ds.items:
        by[it.role][it.map].append(it.row)
    splits: dict[str, Any] = {
        role: {m: count_gate_frames(rows) for m, rows in sorted(maps.items())}
        for role, maps in by.items()
    }
    held = sorted((heldout_rows or {}).items())
    splits["held-out"] = {m: count_gate_frames(rows) for m, rows in held}
    rep = {
        "splits": splits,
        "frames": {role: len(ds.role_items(role)) for role in (ROLE_TRAIN, ROLE_SELECTION)},
        "captures": [
            {"name": c.name, "role": c.role, "counts": c.counts, "protected": c.protected}
            for c in ds.used
        ],
        "refused": [{"name": c.name, "reason": c.reason} for c in ds.refused],
        "protected": {c.name: c.protected for c in ds.used if c.protected},
    }
    rep["g1"] = g1_verdict(rep)
    return rep


def _visible(counts: dict[str, dict[str, int]]) -> int:
    return sum(v.get("4-25", 0) for v in counts.values())


def g1_verdict(rep: dict[str, Any], thresholds: dict[str, int] | None = None) -> dict[str, Any]:
    """The data-sufficiency gate G1: ``{"go": bool, "checks": {name: {ok, value, need}}}``."""
    t = {**G1, **(thresholds or {})}
    tr = rep["splits"][ROLE_TRAIN]
    va = rep["splits"][ROLE_SELECTION]
    ho = rep["splits"]["held-out"]
    train_frames = sum(_visible(c) for c in tr.values())
    train_maps = sum(1 for c in tr.values() if _visible(c) > 0)
    near = sum(v.get("4-8", 0) for c in tr.values() for v in c.values())
    val_maps = sum(1 for c in va.values() if _visible(c) >= t["val_frames_per_map"])
    trained_types = {k for c in tr.values() for k, v in c.items() if v.get("4-25", 0) > 0}
    ho_maps = sum(1 for c in ho.values() if _visible(c) > 0)
    seen: Counter[str] = Counter()
    for c in ho.values():
        for k, v in c.items():
            if k in trained_types:
                seen[k] += v.get("4-25", 0)
    seen_ok = [k for k, n in seen.items() if n >= t["heldout_type_frames"]]
    prot = rep.get("protected") or {}
    prot_checks = {
        name: bool(
            p.get("frames") is not None
            and p["frames_present"] == p["frames"]
            and (
                p.get("expected_in_range") is None
                or p["in_range_gate_frames"] == p["expected_in_range"]
            )
        )  # fmt: skip
        for name, p in prot.items()
    }
    checks = {
        "train_frames": {"ok": train_frames >= t["train_frames"], "value": train_frames,
                         "need": t["train_frames"]},
        "train_maps": {"ok": train_maps >= t["train_maps"], "value": train_maps,
                       "need": t["train_maps"]},
        "train_near_frames": {"ok": near >= t["train_near_frames"], "value": near,
                              "need": t["train_near_frames"]},
        "val_maps": {"ok": val_maps >= t["val_maps"], "value": val_maps, "need": t["val_maps"]},
        "heldout_maps": {"ok": ho_maps >= t["heldout_maps"], "value": ho_maps,
                         "need": t["heldout_maps"]},
        "heldout_seen_type": {"ok": bool(seen_ok), "value": dict(seen),
                              "need": f">= 1 seen type with >= {t['heldout_type_frames']}"},
        "protected_window": {"ok": bool(prot_checks) and all(prot_checks.values()),
                             "value": prot_checks, "need": "the v1 window fully present"},
    }  # fmt: skip
    return {"go": all(c["ok"] for c in checks.values()), "checks": checks}


def report_summary(rep: dict[str, Any]) -> str:
    lines = [f"frames: {rep['frames']}"]
    for role, maps in rep["splits"].items():
        for m, types in maps.items():
            for cls, n in sorted(types.items()):
                lines.append(f"  {role:9s} {m:22s} {cls:28s} 4-25 {n.get('4-25', 0):6d}  "
                             f"4-8 {n.get('4-8', 0):5d}")  # fmt: skip
    for r in rep["refused"]:
        lines.append(f"  refused {r['name']}: {r['reason']}")
    g = rep["g1"]
    lines.append("G1: " + ("GO" if g["go"] else "NO-GO"))
    for k, c in g["checks"].items():
        lines.append(f"  {'ok  ' if c['ok'] else 'FAIL'} {k}: {c['value']} (need {c['need']})")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- frame cache


def cache_dir(ds: Dataset, input_width: int) -> Path:
    return models_root() / "datasets" / f"{ds.manifest_sha()[:8]}-w{input_width}"


def build_frame_cache(
    ds: Dataset, input_width: int, camera_cfg: dict[str, Any], out: Path | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Masked + resized frames (as at runtime) of every item in a ``uint8`` memmap ``(N, 3, H, W)``.

    Reused when ``index.json`` matches (same manifest, width and item files). All frames must share
    one input geometry. Masks are per capture (``capture_index.camera_config``).
    """
    from drone_fly.harness.errors import lazy_import
    from drone_fly.harness.liftoff.gate_net import bgr_to_tensor, input_geometry, resize_images
    from drone_fly.harness.liftoff.masks import FrameMasks
    from drone_fly.harness.offline.capture_index import camera_config
    from drone_fly.harness.offline.evaluate import load_image

    torch = lazy_import("torch", purpose="the gate dataset frame cache")
    d = out or cache_dir(ds, input_width)
    files = [str(it.path) for it in ds.items]
    index_path, data_path = d / "index.json", d / "frames.u8"
    if index_path.is_file() and data_path.is_file():
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        if idx.get("files") == files and idx.get("input_width") == input_width:
            shape = tuple(idx["shape"])
            return np.memmap(data_path, np.uint8, "r", shape=shape), idx
    d.mkdir(parents=True, exist_ok=True)
    masks_by_cap = {c.name: FrameMasks.from_config(camera_config(camera_cfg, c.meta))
                    for c in ds.used}  # fmt: skip
    geometry = None
    mm = None
    for i, it in enumerate(ds.items):
        img = load_image(it.path)
        h, w = img.shape[:2]
        g = input_geometry(w, h, input_width)
        if geometry is None:
            geometry = g
            mm = np.memmap(data_path, np.uint8, "w+", shape=(len(ds.items), 3, g[1], g[0]))
        elif g != geometry:
            raise HarnessConfigError("all captures must share one frame size")
        m = masks_by_cap.get(it.capture)
        if m:
            img = m.apply(img)
        x = resize_images(bgr_to_tensor(img, "cpu"), g[0], g[1])[0]
        mm[i] = x.round().clamp(0, 255).to(torch.uint8).numpy()
    if mm is None:
        raise HarnessConfigError("the dataset has no frames")
    mm.flush()
    idx = {"files": files, "input_width": input_width, "shape": list(mm.shape),
           "geometry": list(geometry), "manifest_sha256": ds.manifest_sha()}  # fmt: skip
    index_path.write_text(json.dumps(idx), encoding="utf-8")
    return np.memmap(data_path, np.uint8, "r", shape=mm.shape), idx
