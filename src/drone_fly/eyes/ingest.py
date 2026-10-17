"""UC-84 AC3: validate dropped-in Liftoff captures and measure the in-view hit-ray fraction.

``drone-fly eyes ingest`` is the in-sandbox backstop for the (user-side, Liftoff) capture drop. It
does **not** execute or re-verify flights -- it checks that the captures the user assembled conform
to what the eyes pipeline needs, and reports how obstacle-dense they are:

* **Discovery / conformance** -- reuses :func:`~drone_fly.eyes.capture_meta.discover_captures` and
  the same meta / catalogue / alignment checks the dataset build applies. Each capture that cannot
  be used is reported with an actionable reason (``no-meta``, ``bad-meta``, ``no-catalogue``,
  ``map-mismatch``, ``alignment-unverified``, ``wrong-split``, ``no-trackdata``, ``no-scene``,
  ``no-telemetry``, ``no-labels``, ``stale-labels``).
* **Held-out-split invariant** -- the split is resolved from the **track catalogue** (never the
  capture ``meta.json``), exactly as :mod:`~drone_fly.eyes.dataset`. UC-84 only grows the *train*
  split, so any capture that resolves to ``val`` / ``held-out`` is a ``wrong-split`` integrity
  violation: it is reported and then :func:`run_ingest` raises, so a leaked capture can never
  silently shift the cross-run comparison baseline.
* **In-view hit-ray fraction** -- for each usable capture, labels (via
  :func:`~drone_fly.eyes.labels.run_label`; existing labels are reused unless ``label_missing``)
  carry per-ray ``hit`` + ``in_view`` arrays. The fraction ``hit[in_view].mean()`` is computed with
  the one shared helper :func:`~drone_fly.eyes.dataset.in_view_hit_counts` (the same path feeds the
  dataset manifest's ``in_view_hit_ray_share``). Counts are summed per map and overall; the overall
  fraction is compared against the ~5 % baseline and **flagged when below the 15 % target** so the
  user knows to fly more / use the degraded-flying fallback (AC1 Section f). The flag is advisory:
  the AC4 driver (``scripts/uc84_obstacle_capture.sh``) reads it to gate its retrain.

Scope: conformance + pairing accounting + the fraction. NOT independent alignment re-verification
(that ran user-side when the capture was assembled and set ``alignment.verified``).

Writes ``ingest.json`` + ``ingest.md`` to ``<out>`` (default ``<data root>/reports/eyes/ingest``).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.eyes.capture_meta import (
    capture_sha,
    discover_captures,
    read_capture_meta,
)
from drone_fly.eyes.dataset import in_view_hit_counts
from drone_fly.eyes.labels import (
    LABELS_FILE,
    default_out,
    labels_root_default,
    read_labels_meta,
    run_label,
)
from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

BASELINE_SHARE = 0.05
TARGET_SHARE = 0.15
REPORT_JSON = "ingest.json"
REPORT_MD = "ingest.md"

#: Reasons a capture is not counted toward the in-view hit-ray fraction (see the module docstring).
REASONS = (
    "no-meta", "bad-meta", "no-catalogue", "map-mismatch", "alignment-unverified", "wrong-split",
    "no-split", "no-trackdata", "no-scene", "no-telemetry", "no-labels", "stale-labels",
)  # fmt: skip


def _skip_reason(skipped: str) -> str:
    """Map a :func:`run_label` ``skipped`` string to an ingest reason."""
    if skipped.startswith("no-geometry"):
        return "no-trackdata"
    if skipped.startswith("unconvertible"):
        return "no-scene"
    if skipped.startswith("no-telemetry"):
        return "no-telemetry"
    if skipped.startswith("map-mismatch"):
        return "map-mismatch"
    if skipped == "no-meta":
        return "no-meta"
    return skipped


def reports_root() -> Path:
    from drone_fly import paths

    return paths.data_root() / "reports" / "eyes" / "ingest"


def _share(hits: int, rays: int) -> float | None:
    return (hits / rays) if rays else None


def run_ingest(
    captures_root: str | Path | None = None,
    *,
    labels_root: str | Path | None = None,
    out: str | Path | None = None,
    label_missing: bool = False,
    baseline: float = BASELINE_SHARE,
    target: float = TARGET_SHARE,
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    catalogue: Any = None,
    label_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the captures under ``captures_root`` and report the in-view hit-ray fraction.

    Returns the report dict (also written to ``<out>``). Raises :class:`HarnessConfigError` when a
    capture resolves to a non-train catalogue split (``wrong-split``) -- the held-out-split
    invariant -- after writing the report, so the offending captures are visible.
    """
    from drone_fly import paths

    croot = Path(captures_root or paths.captures_root())
    lroot = Path(labels_root or labels_root_default())
    cat = catalogue or _catalogue()
    map_split = {m: (v or {}).get("split") for m, v in cat.maps().items()}
    lkw = dict(label_kwargs or {})

    captures: list[dict[str, Any]] = []
    for cdir in sorted(discover_captures(croot)):
        captures.append(_ingest_one(
            cdir, cat=cat, map_split=map_split, lroot=lroot, label_missing=label_missing,
            preset=preset, config=config, harness=harness, label_kwargs=lkw,
        ))  # fmt: skip

    # ---- aggregate per map + overall (sum counts, never average fractions) ----
    per_map: dict[str, dict[str, Any]] = {}
    tot_hits = tot_rays = tot_frames = included = 0
    for c in captures:
        if not c["included"]:
            continue
        included += 1
        tot_hits += c["in_view_hits"]
        tot_rays += c["in_view_rays"]
        tot_frames += c["frames"]
        pm = per_map.setdefault(c["map"], {"in_view_hits": 0, "in_view_rays": 0, "frames": 0,
                                           "captures": 0})  # fmt: skip
        pm["in_view_hits"] += c["in_view_hits"]
        pm["in_view_rays"] += c["in_view_rays"]
        pm["frames"] += c["frames"]
        pm["captures"] += 1
    for pm in per_map.values():
        pm["in_view_hit_ray_share"] = _share(pm["in_view_hits"], pm["in_view_rays"])

    overall_share = _share(tot_hits, tot_rays)
    below_target = overall_share is None or overall_share < target
    wrong_split = [c["name"] for c in captures if c.get("reason") == "wrong-split"]

    report = {
        "format": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "captures_root": str(croot),
        "labels_root": str(lroot),
        "catalogue": str(cat.path) if cat.path is not None else None,
        "baseline_in_view_hit_ray_share": float(baseline),
        "target_in_view_hit_ray_share": float(target),
        "overall": {
            "in_view_hits": tot_hits,
            "in_view_rays": tot_rays,
            "in_view_hit_ray_share": overall_share,
            "frames": tot_frames,
            "captures_included": included,
            "captures_total": len(captures),
        },
        "below_target": below_target,
        "wrong_split": wrong_split,
        "per_map": {m: per_map[m] for m in sorted(per_map)},
        "captures": captures,
    }

    out_dir = Path(out) if out is not None else reports_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(_markdown(report), encoding="utf-8")
    report["out_dir"] = str(out_dir)

    if wrong_split:
        raise HarnessConfigError(
            "held-out-split invariant violated: these captures resolve to a non-train catalogue "
            f"split and must not be added for UC-84: {wrong_split}. UC-84 only grows the train "
            f"split (frozen: val + held-out maps). See {out_dir / REPORT_JSON}."
        )
    return report


def _catalogue() -> Any:
    from drone_fly.tracks import catalogue as cat

    return cat.load_named("liftoff")


def _ingest_one(
    cdir: Path,
    *,
    cat: Any,
    map_split: dict[str, Any],
    lroot: Path,
    label_missing: bool,
    preset: str | None,
    config: Any,
    harness: str,
    label_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from drone_fly.tracks.catalogue import TrackDataError

    rec: dict[str, Any] = {
        "name": cdir.name, "capture": str(cdir), "included": False, "reason": None,
        "map": None, "split": None, "catalogue_id": None,
        "in_view_hits": 0, "in_view_rays": 0, "in_view_hit_ray_share": None, "frames": 0,
        "pairing": None,
    }  # fmt: skip

    def fail(reason: str) -> dict[str, Any]:
        rec["reason"] = reason
        return rec

    try:
        cm = read_capture_meta(cdir)
    except HarnessConfigError as exc:
        return fail("no-catalogue" if "catalogue_id" in str(exc) else "bad-meta")
    if cm is None:
        return fail("no-meta")
    rec["catalogue_id"] = cm.catalogue_id

    try:
        entry = cat.by_id(cm.catalogue_id)
    except TrackDataError:
        return fail("no-catalogue")
    rec["map"] = entry["map"]
    if cm.map is not None and cm.map != entry["map"]:
        return fail("map-mismatch")

    split = map_split.get(entry["map"])
    rec["split"] = split
    if split is None:
        return fail("no-split")
    if split != "train":
        # Held-out-split invariant: reported here, hard-raised by run_ingest after the report.
        return fail("wrong-split")

    if not cm.alignment_verified:
        return fail("alignment-unverified")

    ldir = default_out(cdir.name, lroot)
    if label_missing:
        res = run_label(cdir, preset=preset, config=config, harness=harness,
                        labels_root=lroot, **label_kwargs)  # fmt: skip
        if res.get("skipped"):
            return fail(_skip_reason(res["skipped"]))

    lm = read_labels_meta(ldir)
    if lm is None:
        return fail("no-labels")
    if lm.get("capture_sha256") != capture_sha(cdir):
        return fail("stale-labels")
    rec["pairing"] = lm.get("drop_reasons")

    with np.load(ldir / LABELS_FILE, allow_pickle=False) as z:
        hits, rays = in_view_hit_counts(z["hit"], z["in_view"])
        rec["frames"] = int(len(z["hit"]))
    rec["in_view_hits"] = hits
    rec["in_view_rays"] = rays
    rec["in_view_hit_ray_share"] = _share(hits, rays)
    rec["included"] = True
    return rec


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def _markdown(r: dict[str, Any]) -> str:
    ov = r["overall"]
    base = r["baseline_in_view_hit_ray_share"]
    tgt = r["target_in_view_hit_ray_share"]
    verdict = "BELOW TARGET" if r["below_target"] else "meets target"
    lines = [
        "# Eyes capture ingest (UC-84 AC3)",
        "",
        f"Captures root `{r['captures_root']}` -- {ov['captures_included']}/"
        f"{ov['captures_total']} captures usable, {ov['frames']} labelled frames.",
        "",
        f"**Overall in-view hit-ray fraction: {_fmt(ov['in_view_hit_ray_share'])}** "
        f"(baseline ~{base:.2f}, target >= {tgt:.2f}) -- **{verdict}**.",
        "",
    ]
    if r["below_target"]:
        lines += [
            "> Below the target: fly more obstacle-dense courses, or use the degraded/perturbed-"
            "flying fallback (capture protocol doc, Section f), then re-run `eyes ingest`.",
            "",
        ]
    if r["wrong_split"]:
        lines += [
            f"> **Held-out-split violation** -- non-train captures present: {r['wrong_split']}. "
            "Remove them; UC-84 only grows the train split.",
            "",
        ]
    lines += [
        "| map | captures | frames | in-view hit-ray share |",
        "|---|---:|---:|---:|",
    ]
    for m, pm in r["per_map"].items():
        lines.append(
            f"| {m} | {pm['captures']} | {pm['frames']} | "
            f"{_fmt(pm['in_view_hit_ray_share'])} |"
        )
    lines += [
        "",
        "## Captures",
        "",
        "| capture | map | split | frames | in-view hit-ray share | status |",
        "|---|---|---|---:|---:|---|",
    ]
    for c in r["captures"]:
        status = "included" if c["included"] else f"excluded: {c['reason']}"
        lines.append(
            f"| {c['name']} | {c['map'] or '-'} | {c['split'] or '-'} | {c['frames']} | "
            f"{_fmt(c['in_view_hit_ray_share'])} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)
