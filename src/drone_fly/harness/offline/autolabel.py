"""Auto-label gates in a recorded flight by projecting track geometry through the telemetry pose.

``drone-fly harness label --capture <dir> --track <xml> [--race <xml>] [--props <json>]
[--preset ...] [--out ...] [--preview N] [--lag-ms ...]``

For every frame paired with telemetry (:func:`~drone_fly.harness.offline.capture.load_capture`;
frames in respawn / discontinuity segments are dropped), every placed item that matches the
``gate_types`` table is posed with its XML position, full Unity Euler rotation and scale. Its
keypoints go Unity-local -> Unity world -> contract (Liftoff metres) and are projected with the
harness :class:`~drone_fly.harness.geometry.CameraModel` (``world_to_camera`` + ``project``:
pinhole, camera pitched up by ``uptilt`` — the convention verified on the reference capture).

Per gate the label holds the class, keypoint pixels, per-keypoint visibility (in front, in the
image and outside every ``camera.masks`` region), the aperture-centre pixel, range and bearing,
the world centre and normal (for scoring), and whether the left/right pairs were swapped because
the gate is seen from behind (see :mod:`~drone_fly.harness.gate_types`). Occlusion is not
modelled.

Outputs (default ``artifacts/harness/liftoff/labels/<capture-name>/``): ``labels.jsonl`` (one
frame per line), ``labels_meta.json`` (config snapshot, counts, drop reasons, ``--props``
warnings); ``--preview N`` writes overlays to ``artifacts/harness/liftoff/reports/label-preview/``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import HarnessConfig, find_repo_root, load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateType, GateTypeTable
from drone_fly.harness.geometry import UNITY_TO_CONTRACT, CameraModel, quat_to_matrix
from drone_fly.harness.liftoff.gate_cv import camera_from_config
from drone_fly.harness.liftoff.masks import FrameMasks
from drone_fly.harness.offline.capture import Capture, load_capture
from drone_fly.harness.offline.track_xml import PlacedItem, parse_race_route, parse_track_items
from drone_fly.harness.recording import git_head

_M = UNITY_TO_CONTRACT.astype(np.float64)
ARTIFACTS = Path("artifacts") / "harness" / "liftoff"
MAX_RANGE_M = 100.0


def artifacts_root() -> Path:
    """``<repo>/artifacts/harness/liftoff`` (the cwd when no repo root is found)."""
    root = find_repo_root()
    return (root if root is not None else Path.cwd()) / ARTIFACTS


# ----------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class PlacedGate:
    """A gate item posed in the contract world frame (Liftoff metres, unscaled)."""

    item: PlacedItem
    gtype: GateType
    keypoints_w: np.ndarray  # (K, 3)
    center_w: np.ndarray  # aperture centre
    forward_w: np.ndarray  # unit gate-plane normal (item +z)

    def local_to_world(self, local: np.ndarray) -> np.ndarray:
        """Unity-local ``(N, 3)`` points of this item -> contract world (Liftoff metres)."""
        p = np.asarray(local, np.float64) * self.item.scale
        return (p @ self.item.rotation.T + self.item.position) @ _M.T


def place_gate(item: PlacedItem, gtype: GateType) -> PlacedGate:
    def world(local: np.ndarray) -> np.ndarray:
        p = np.asarray(local, np.float64) * item.scale
        return (p @ item.rotation.T + item.position) @ _M.T

    fwd = _M @ (item.rotation @ np.array([0.0, 0.0, 1.0]))
    return PlacedGate(
        item=item,
        gtype=gtype,
        keypoints_w=world(gtype.local_points()),
        center_w=world(np.asarray([gtype.center_offset], np.float64))[0],
        forward_w=fwd / (np.linalg.norm(fwd) + 1e-12),
    )


def placed_gates(items: list[PlacedItem], table: GateTypeTable) -> list[PlacedGate]:
    out = []
    for it in items:
        gtype = table.classify(it.item_id)
        if gtype is not None:
            out.append(place_gate(it, gtype))
    return out


def project_points(
    camera: CameraModel, pos: np.ndarray, quat: np.ndarray, points_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> ``(pixels, in_front, camera-frame points)`` for a drone pose."""
    p_cam = (
        np.asarray(points_w, np.float64) - np.asarray(pos, np.float64)
    ) @ camera.world_to_camera(quat).T
    px, front = camera.project(p_cam)
    return px, front, p_cam


def label_gate(
    gate: PlacedGate,
    camera: CameraModel,
    masks: FrameMasks,
    pos: np.ndarray,
    quat: np.ndarray,
    max_range_m: float = MAX_RANGE_M,
) -> dict[str, Any] | None:
    """One gate's label for a drone pose (Liftoff metres), or ``None`` when not in view."""
    pos = np.asarray(pos, np.float64)
    cpx, cfront, c_cam = project_points(camera, pos, quat, gate.center_w[None])
    rng = float(np.linalg.norm(gate.center_w - pos))
    if not cfront[0] or rng > max_range_m:
        return None
    px, front, _ = project_points(camera, pos, quat, gate.keypoints_w)
    in_img = camera.in_image(px, front)
    center_in = bool(camera.in_image(cpx, cfront)[0])
    if not in_img.any() and not center_in:
        return None
    w, h = camera.width, camera.height
    masked = masks.contains(px, w, h) if masks else np.zeros(len(px), bool)
    visible = in_img & ~masked
    flipped = float(np.dot(gate.center_w - pos, gate.forward_w)) < 0  # seen from behind
    if flipped and gate.gtype.flip_pairs:
        perm = gate.gtype.flip_permutation()
        px, front, in_img, visible = px[perm], front[perm], in_img[perm], visible[perm]
    c = c_cam[0]
    fwd = gate.forward_w
    return {
        "instance": gate.item.instance_id,
        "item_id": gate.item.item_id,
        "cls": gate.gtype.cls,
        "keypoints": np.round(px, 2).tolist(),
        "visible": visible.tolist(),
        "in_image": in_img.tolist(),
        "in_front": front.tolist(),
        "center_px": np.round(cpx[0], 2).tolist(),
        "center_in_image": center_in,
        "center_masked": bool(masks.contains(cpx, w, h)[0]) if masks else False,
        "range_m": rng,
        "bearing_deg": [
            math.degrees(math.atan2(c[0], c[2])),
            math.degrees(math.atan2(-c[1], math.hypot(c[0], c[2]))),
        ],
        "center_world": np.round(gate.center_w, 4).tolist(),
        "normal_yaw_deg": math.degrees(math.atan2(fwd[1], fwd[0])),
        "t_cam": np.round(c, 4).tolist(),
        "flipped": bool(flipped),
    }


# ----------------------------------------------------------------------------- props check


def _rot(q_xyzw: Any) -> np.ndarray:
    x, y, z, w = (float(v) for v in (q_xyzw or (0.0, 0.0, 0.0, 1.0)))
    return quat_to_matrix([w, x, y, z])


def collider_bounds(colliders: list[dict[str, Any]]) -> dict[str, Any]:
    """Item-local AABB of solid colliders + the trigger box (``trig``) centre / half-size."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    trigger = None
    for c in colliders:
        t = c.get("t")
        p = np.asarray(c.get("p", (0.0, 0.0, 0.0)), np.float64)
        if t == "box":
            half = np.abs(_rot(c.get("q"))) @ np.asarray(c["s"], np.float64)
            if c.get("trig"):
                trigger = {"center": p.tolist(), "half": half.tolist()}
                continue
            a, b = p - half, p + half
        elif t == "cap":
            axis = np.zeros(3)
            axis[int(c.get("a", 1))] = 1.0
            d = _rot(c.get("q")) @ axis * max(float(c["h"]) / 2.0 - float(c["r"]), 0.0)
            r = float(c["r"])
            a, b = np.minimum(p - d, p + d) - r, np.maximum(p - d, p + d) + r
        elif t == "sph":
            r = float(c["r"])
            a, b = p - r, p + r
        elif t == "mesh" and c.get("v"):
            v = np.asarray(c["v"], np.float64)
            a, b = v.min(axis=0), v.max(axis=0)
        else:
            continue
        lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    solid = None if not np.isfinite(lo).all() else {"min": lo.tolist(), "max": hi.tolist()}
    return {"solid": solid, "trigger": trigger}


def load_props(path: str | Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessConfigError(f"cannot read props file {path}: {exc}") from exc
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, dict):
        raise HarnessConfigError(f"{path} is not a liftoff-flight-analyzer props.json")
    return items


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-6)


def check_gate_types_against_props(
    table: GateTypeTable, item_ids: set[str], props: dict[str, Any], tol: float = 0.05
) -> list[str]:
    """Warnings where ``gate_types`` disagree with collider-derived bounds by more than ``tol``."""
    warnings = []
    for item_id in sorted(item_ids):
        gtype = table.classify(item_id)
        entry = props.get(item_id)
        if gtype is None:
            continue
        if not entry or not entry.get("colliders"):
            warnings.append(f"{item_id}: no collider shapes in props; {gtype.cls} not checked")
            continue
        b = collider_bounds(entry["colliders"])
        if b["solid"] is not None:
            lo, hi = np.asarray(b["solid"]["min"]), np.asarray(b["solid"]["max"])
            width, height = hi[0] - lo[0], hi[1] - max(lo[1], 0.0)
            for name, got, want in (
                ("outer width", width, gtype.outer[0]),
                ("outer height", height, gtype.outer[1]),
            ):
                if _rel(got, want) > tol:
                    warnings.append(
                        f"{item_id} ({gtype.cls}): {name} {want:.3f} m in gate_types vs "
                        f"{got:.3f} m from colliders"
                    )
        trig = b["trigger"]
        if trig is not None:
            half, center = np.asarray(trig["half"]), np.asarray(trig["center"])
            for name, got, want in (
                ("aperture width", 2 * half[0], gtype.aperture[0]),
                ("aperture height", 2 * half[1], gtype.aperture[1]),
                ("aperture centre y", center[1], gtype.center_offset[1]),
            ):
                if _rel(got, want) > tol:
                    warnings.append(
                        f"{item_id} ({gtype.cls}): {name} {want:.3f} m in gate_types vs "
                        f"{got:.3f} m from the trigger box"
                    )
    return warnings


#: aperture bottom this far above the prop base -> a closed frame (a bottom bar), else an open arch
CLOSED_BOTTOM_M = 0.3


def _class_name(item_id: str) -> str:
    import re

    base = re.sub(r"\d+$", "", item_id)
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", base).lower()


def draft_gate_types(props: dict[str, Any], item_ids: list[str]) -> list[dict[str, Any]]:
    """Candidate ``gate_types`` entries from props.json colliders (UC71 ``gates draft-types``).

    Outer size = the solid colliders' bounds; aperture + centre = the trigger box; ``bottom`` =
    ``closed`` when the aperture starts more than :data:`CLOSED_BOTTOM_M` above the prop base.
    Default keypoints: closed frames 4 outer + 4 aperture corners; open arches feet, mid-legs, top
    corners and top centre; left/right flip pairs always. Candidates only: every class is checked
    with label previews before it is trained on.
    """
    out = []
    for item_id in item_ids:
        entry = props.get(item_id)
        draft: dict[str, Any] = {"class": _class_name(item_id), "match": f"^{item_id}"}
        if not entry or not entry.get("colliders"):
            draft["error"] = "no collider shapes in props.json"
            out.append(draft)
            continue
        b = collider_bounds(entry["colliders"])
        if b["solid"] is None:
            draft["error"] = "no solid collider"
            out.append(draft)
            continue
        lo, hi = np.asarray(b["solid"]["min"]), np.asarray(b["solid"]["max"])
        cx = float((lo[0] + hi[0]) / 2.0)
        base = float(max(lo[1], 0.0))
        w, h = float(hi[0] - lo[0]), float(hi[1] - base)
        trig = b["trigger"]
        if trig is not None:
            half, centre = np.asarray(trig["half"]), np.asarray(trig["center"])
            ap = [round(2 * float(half[0]), 3), round(2 * float(half[1]), 3)]
            centre_xyz = [round(float(v), 3) for v in centre]
            ap_bottom = float(centre[1] - half[1])
        else:
            draft["warning"] = "no trigger: aperture guessed as 80 % of the outer frame"
            ap = [round(0.8 * w, 3), round(0.8 * h, 3)]
            centre_xyz = [round(cx, 3), round(base + h / 2.0, 3), 0.0]
            ap_bottom = base + 0.1 * h
        closed = ap_bottom - base > CLOSED_BOTTOM_M
        x0, x1, y1 = cx - w / 2.0, cx + w / 2.0, base + h
        if closed:
            ax0 = centre_xyz[0] - ap[0] / 2.0
            ax1 = centre_xyz[0] + ap[0] / 2.0
            ay0, ay1 = centre_xyz[1] - ap[1] / 2.0, centre_xyz[1] + ap[1] / 2.0
            kps = [[x0, y1], [x1, y1], [x0, base], [x1, base],
                   [ax0, ay1], [ax1, ay1], [ax0, ay0], [ax1, ay0]]  # fmt: skip
            pairs = [[0, 1], [2, 3], [4, 5], [6, 7]]
        else:
            kps = [[x0, base], [x1, base], [x0, base + h / 2.0], [x1, base + h / 2.0],
                   [x0, y1], [x1, y1], [cx, y1]]  # fmt: skip
            pairs = [[0, 1], [2, 3], [4, 5]]
        draft.update(
            aperture=ap,
            outer=[round(w, 3), round(h, 3)],
            center_offset=centre_xyz,
            bottom="closed" if closed else "open",
            keypoints=[[round(float(x), 3), round(float(y), 3)] for x, y in kps],
            flip_pairs=pairs,
        )
        out.append(draft)
    return out


def check_item_table_against_props(
    item_table: list[dict[str, Any]], item_ids: set[str], props: dict[str, Any], tol: float = 0.05
) -> list[str]:
    """Warnings where ``offline.item_table`` obstacle sizes disagree with collider bounds."""
    import re

    warnings = []
    for item_id in sorted(item_ids):
        entry = next((e for e in item_table if re.search(e.get("match", "$^"), item_id)), None)
        prop = props.get(item_id)
        if entry is None or entry.get("kind") not in ("box", "cylinder") or not prop:
            continue
        b = collider_bounds(prop.get("colliders") or [])["solid"]
        if b is None:
            continue
        ext = np.asarray(b["max"]) - np.asarray(b["min"])
        if entry["kind"] == "box":
            want = np.asarray(entry["size"], np.float64)
            for axis, got, w in zip("xyz", ext, want, strict=True):
                if _rel(got, w) > tol:
                    warnings.append(
                        f"{item_id}: item_table size {axis} {w:.3f} m vs {got:.3f} m from colliders"
                    )
        else:
            if _rel(ext[1], float(entry["height"])) > tol:
                warnings.append(
                    f"{item_id}: item_table height {float(entry['height']):.3f} m vs "
                    f"{ext[1]:.3f} m from colliders"
                )
    return warnings


# ----------------------------------------------------------------------------- labelling


def label_capture(
    capture: Capture,
    gates: list[PlacedGate],
    camera: CameraModel,
    masks: FrameMasks,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Label every paired frame of ``capture`` (states must be at world scale 1)."""
    rows, counts = [], Counter()
    for f in capture.frames:
        labels = []
        for g in gates:
            lab = label_gate(g, camera, masks, f.state.pos, f.state.quat)
            if lab is not None:
                labels.append(lab)
                counts[f"gates:{lab['cls']}"] += 1
                if sum(lab["visible"]) >= 4:
                    counts[f"gates_4kp:{lab['cls']}"] += 1
                if lab["center_in_image"] and 4.0 <= lab["range_m"] <= 25.0:
                    counts[f"gates_4_25m_centre_in_image:{lab['cls']}"] += 1
        counts["frames_with_gate"] += bool(labels)
        rows.append(
            {
                "frame": f.index,
                "file": str(f.path.relative_to(capture.root)).replace("\\", "/"),
                "stamp": f.stamp,
                "segment": f.segment,
                "sample_index": f.sample_index,
                "gates": labels,
            }
        )
    return rows, counts


def draw_preview(image: np.ndarray, row: dict[str, Any], masks: FrameMasks) -> np.ndarray:
    from drone_fly.harness.errors import lazy_import

    cv2 = lazy_import("cv2", package="opencv-python", purpose="label previews")
    img = image.copy()
    h, w = img.shape[:2]
    for x0, y0, x1, y1 in masks.rects(w, h):
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 3)
    for g in row["gates"]:
        kp = np.asarray(g["keypoints"])
        for k, (pt, vis, front) in enumerate(zip(kp, g["visible"], g["in_front"], strict=True)):
            if not front or not np.isfinite(pt).all() or abs(pt).max() > 1e5:
                continue
            col = (0, 255, 0) if vis else (0, 0, 255)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, col, -1)
            cv2.putText(
                img, str(k), (int(pt[0]) + 8, int(pt[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2
            )
        c = g["center_px"]
        if g["center_in_image"]:
            cv2.drawMarker(img, (int(c[0]), int(c[1])), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.putText(
                img,
                f"{g['cls']} #{g['instance']} {g['range_m']:.1f}m{' flip' if g['flipped'] else ''}",
                (int(c[0]) + 12, int(c[1]) + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
            )
    return cv2.resize(img, (w // 2, h // 2))


def refuse_unknown_items(
    items: list[PlacedItem], table: GateTypeTable, not_gates: Any, route: Any, track: Any
) -> None:
    """UC71 AC5: refuse a track with itemIDs the gate-type table does not cover (never guessed)."""
    from drone_fly.tracks.gate_coverage import unknown_items

    by_instance = {it.instance_id: it.item_id for it in items if it.instance_id}
    checkpoints = route.checkpoint_ids if route else ()
    route_ids = [by_instance[c] for c in checkpoints if c in by_instance]
    unknown = unknown_items(items, table, not_gates, route_ids)
    if unknown:
        raise HarnessConfigError(
            f"{track} has itemIDs that are not a gate_types class, a checkpoint volume or a "
            f"not_gates entry: {unknown}; add them to the gate-types file (see "
            "`drone-fly tracks check --gate-coverage`)"
        )


def run_autolabel(
    capture: str | Path,
    track: str | Path,
    race: str | Path | None = None,
    props: str | Path | None = None,
    *,
    config: str | Path | dict[str, Any] | None = None,
    preset: str | None = None,
    harness: str = "liftoff",
    out: str | Path | None = None,
    preview: int = 0,
    preview_dir: str | Path | None = None,
    lag_ms: float = 0.0,
) -> dict[str, Any]:
    """Label a capture; returns the ``labels_meta.json`` content (see the module docstring)."""
    cfg: HarnessConfig = load_config(harness, config, preset=preset)
    cam_cfg = cfg.section("camera")
    if cam_cfg.get("resolution") is None:
        raise HarnessConfigError(
            "auto-labelling needs camera.resolution (the capture's pixel grid)"
        )
    camera = camera_from_config(cam_cfg)
    masks = FrameMasks.from_config(cam_cfg)
    table = GateTypeTable.from_config(cfg.get("gate_types"))
    cap = load_capture(capture, cfg.section("telemetry"), 1.0, lag_ms)
    items = parse_track_items(track)
    route = parse_race_route(race) if race is not None else None
    refuse_unknown_items(items, table, cfg.not_gates, route, track)
    gates = placed_gates(items, table)
    if not gates:
        raise HarnessConfigError(
            f"no item in {track} matches gate_types {[t.match for t in table.types]}"
        )
    props_warnings: list[str] = []
    if props is not None:
        prop_items = load_props(props)
        ids = {it.item_id for it in items}
        props_warnings = check_gate_types_against_props(table, ids, prop_items)
    rows, counts = label_capture(cap, gates, camera, masks)

    out_dir = Path(out) if out is not None else artifacts_root() / "labels" / cap.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "format": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture": str(Path(capture).resolve()),
        "capture_name": cap.name,
        "track": str(Path(track).resolve()),
        "race": None if race is None else str(Path(race).resolve()),
        "props": None if props is None else str(Path(props).resolve()),
        "route_checkpoints": None if route is None else list(route.checkpoint_ids),
        "harness": harness,
        "preset": preset,
        "lag_ms": lag_ms,
        "camera": {
            "resolution": [camera.width, camera.height],
            "hfov_deg": camera.hfov_deg,
            "uptilt_deg": camera.uptilt_deg,
        },
        "masks": cam_cfg.get("masks") or [],
        "gate_types": cfg.get("gate_types"),
        "gate_types_hash": table.hash(),
        "gate_instances": [
            {"instance": g.item.instance_id, "item_id": g.item.item_id, "cls": g.gtype.cls}
            for g in gates
        ],
        "config": cfg.data,
        "capture_summary": cap.summary(),
        "counts": {
            "frames_total": cap.n_frames_total,
            "frames_labelled": len(rows),
            **dict(sorted(counts.items())),
        },
        "drop_reasons": cap.dropped,
        "props_warnings": props_warnings,
        "git_head": git_head(find_repo_root()),
    }
    (out_dir / "labels_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    previews = []
    if preview > 0:
        from drone_fly.harness.offline.evaluate import load_image

        pdir = Path(preview_dir) if preview_dir else artifacts_root() / "reports" / "label-preview"
        pdir = pdir / cap.name
        pdir.mkdir(parents=True, exist_ok=True)
        good = [r for r in rows if any(sum(g["visible"]) >= 4 for g in r["gates"])]
        pick = [good[i] for i in np.linspace(0, len(good) - 1, min(preview, len(good))).astype(int)]
        for r in pick:
            from drone_fly.harness.errors import lazy_import

            cv2 = lazy_import("cv2", package="opencv-python", purpose="label previews")
            img = draw_preview(load_image(cap.root / r["file"]), r, masks)
            name = pdir / f"{Path(r['file']).stem}.jpg"
            cv2.imwrite(str(name), img)
            previews.append(str(name))
    meta["out_dir"] = str(out_dir)
    meta["previews"] = previews
    return meta
