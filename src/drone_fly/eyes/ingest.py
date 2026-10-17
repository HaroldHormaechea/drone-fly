"""UC-84 AC3: validate dropped-in Liftoff captures and measure the in-view hit-ray fraction.

``drone-fly eyes ingest`` is the in-sandbox backstop for the (user-side, Liftoff) capture drop. It
does **not** execute or re-verify flights -- it checks that the captures the user assembled conform
to what the eyes pipeline needs, and reports how obstacle-dense they are:

* **Discovery / conformance** -- reuses :func:`~drone_fly.eyes.capture_meta.discover_captures` and
  the same meta / catalogue / alignment checks the dataset build applies. Each capture that cannot
  be used is reported with an actionable reason (``no-meta``, ``bad-meta``, ``no-catalogue``,
  ``map-mismatch``, ``alignment-unverified``, ``frozen-eval``, ``frozen-eval-modified``,
  ``wrong-split``, ``no-trackdata``, ``no-scene``, ``no-telemetry``, ``no-labels``,
  ``stale-labels``).
* **Held-out-split invariant** -- the split is resolved from the **track catalogue** (never the
  capture ``meta.json``), exactly as :mod:`~drone_fly.eyes.dataset`. UC-84 only grows the *train*
  split; the existing val / held-out captures are the **frozen eval baseline** that ``eyes
  dataset`` / ``eyes eval`` read from the same root, recorded in the committed manifest
  ``tracks/liftoff-frozen-eval-captures.json`` keyed by (capture folder name, :func:`capture_sha`).
  A non-train capture whose name is listed **and** whose current sha matches is ``frozen-eval``
  (reported, not counted, not a failure). A listed name whose content changed is
  ``frozen-eval-modified``; an unlisted one is ``wrong-split``. Both are integrity violations:
  they are reported and then :func:`check_held_out_invariant` raises, so a new or edited eval
  capture can never silently shift the cross-run comparison baseline. Re-baselining the frozen set
  is a deliberate act: regenerate the manifest with :func:`build_frozen_manifest` and commit it.
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
    "no-meta", "bad-meta", "no-catalogue", "map-mismatch", "alignment-unverified", "frozen-eval",
    "frozen-eval-modified", "wrong-split", "no-split", "no-trackdata", "no-scene", "no-telemetry",
    "no-labels", "stale-labels",
)  # fmt: skip
#: Reasons that violate the held-out-split invariant (reported, then raised).
VIOLATIONS = ("wrong-split", "frozen-eval-modified")

FROZEN_MANIFEST_FILE = "liftoff-frozen-eval-captures.json"
FROZEN_MANIFEST_FORMAT = 1
FROZEN_SPLITS = ("val", "held-out")


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


def frozen_manifest_default() -> Path:
    """The committed frozen-eval capture manifest (``tracks/liftoff-frozen-eval-captures.json``)."""
    from drone_fly.tracks.catalogue import catalogue_dir

    return catalogue_dir() / FROZEN_MANIFEST_FILE


def load_frozen_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """``{capture name: entry}`` from the frozen-eval manifest.

    A missing file is an empty mapping (safe default: every non-train capture then fails); a
    malformed one raises :class:`HarnessConfigError`.
    """
    p = Path(path) if path is not None else frozen_manifest_default()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HarnessConfigError(f"cannot read frozen-eval manifest {p}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FROZEN_MANIFEST_FORMAT:
        raise HarnessConfigError(f"{p}: unsupported frozen-eval manifest (format must be 1)")
    entries = data.get("captures")
    if not isinstance(entries, list):
        raise HarnessConfigError(f"{p}: 'captures' must be a list")
    out: dict[str, dict[str, Any]] = {}
    for i, e in enumerate(entries):
        if not (isinstance(e, dict) and isinstance(e.get("name"), str)
                and isinstance(e.get("capture_sha256"), str)):  # fmt: skip
            raise HarnessConfigError(f"{p}: captures[{i}] needs string 'name' + 'capture_sha256'")
        if e["name"] in out:
            raise HarnessConfigError(f"{p}: duplicate capture name {e['name']!r}")
        out[e["name"]] = e
    return out


def build_frozen_manifest(
    captures_root: str | Path | None = None,
    *,
    out: str | Path | None = None,
    catalogue: Any = None,
) -> dict[str, Any]:
    """(Re)write the frozen-eval manifest from every val / held-out capture under ``captures_root``.

    This is the deliberate **re-baseline** procedure (no CLI flag on purpose): it records the
    current :func:`capture_sha` of each capture whose catalogue split is val / held-out, so
    afterwards ``eyes ingest`` treats exactly those captures as the frozen eval baseline. Commit
    the result. Returns the manifest dict (also written to ``out``, default
    :func:`frozen_manifest_default`).
    """
    from drone_fly import paths
    from drone_fly.tracks.catalogue import TrackDataError, sha256_file

    croot = Path(captures_root or paths.captures_root())
    cat = catalogue or _catalogue()
    map_split = {m: (v or {}).get("split") for m, v in cat.maps().items()}
    entries: list[dict[str, Any]] = []
    for cdir in sorted(discover_captures(croot)):
        try:
            cm = read_capture_meta(cdir)
            entry = cat.by_id(cm.catalogue_id) if cm is not None else None
        except (HarnessConfigError, TrackDataError):
            continue
        if entry is None or (cm.map is not None and cm.map != entry["map"]):
            continue
        split = map_split.get(entry["map"])
        if split not in FROZEN_SPLITS:
            continue
        entries.append({
            "name": cdir.name, "map": entry["map"], "split": split,
            "catalogue_id": cm.catalogue_id, "capture_sha256": capture_sha(cdir),
        })  # fmt: skip
    entries.sort(key=lambda e: e["name"])
    manifest = {
        "format": FROZEN_MANIFEST_FORMAT,
        "why": (
            "Frozen val/held-out eval captures (UC-84). `eyes ingest` accepts a val/held-out "
            "capture only if its folder name is listed here AND its capture_sha256 still matches; "
            "anything else violates the held-out-split invariant. Only name + capture_sha256 are "
            "checked; map/split/catalogue_id are informational. Re-baseline deliberately with "
            "drone_fly.eyes.ingest.build_frozen_manifest() and commit the result."
        ),
        "catalogue": cat.path.name if cat.path is not None else None,
        "catalogue_sha256_at_freeze": sha256_file(cat.path) if cat.path is not None else None,
        "captures": entries,
    }
    p = Path(out) if out is not None else frozen_manifest_default()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Always LF, so the committed file's sha (logged in the UC-84 provenance) is stable.
    p.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


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
    frozen_manifest: str | Path | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate the captures under ``captures_root`` and report the in-view hit-ray fraction.

    Returns the report dict (also written to ``<out>``). When ``strict`` (the default) it then calls
    :func:`check_held_out_invariant`, which raises :class:`HarnessConfigError` if a non-train
    capture is not a matching entry of the frozen-eval manifest (``wrong-split`` /
    ``frozen-eval-modified``) -- after writing the report, so the offending captures are visible.
    ``strict=False`` returns the report and leaves that check to the caller.
    """
    from drone_fly import paths

    croot = Path(captures_root or paths.captures_root())
    lroot = Path(labels_root or labels_root_default())
    cat = catalogue or _catalogue()
    map_split = {m: (v or {}).get("split") for m, v in cat.maps().items()}
    lkw = dict(label_kwargs or {})
    fpath = Path(frozen_manifest) if frozen_manifest is not None else frozen_manifest_default()
    frozen = load_frozen_manifest(fpath)

    captures: list[dict[str, Any]] = []
    for cdir in sorted(discover_captures(croot)):
        captures.append(_ingest_one(
            cdir, cat=cat, map_split=map_split, lroot=lroot, label_missing=label_missing,
            preset=preset, config=config, harness=harness, label_kwargs=lkw, frozen=frozen,
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
    # ``wrong_split`` keeps its name but lists every invariant violation (wrong-split + modified).
    wrong_split = [c["name"] for c in captures if c.get("reason") in VIOLATIONS]
    frozen_eval = [c["name"] for c in captures if c.get("reason") == "frozen-eval"]
    seen = {c["name"] for c in captures}
    frozen_missing = sorted(n for n in frozen if n not in seen)

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
            "captures_frozen_eval": len(frozen_eval),
        },
        "below_target": below_target,
        "wrong_split": wrong_split,
        "frozen_eval": frozen_eval,
        "frozen_eval_missing": frozen_missing,
        "frozen_manifest": str(fpath),
        "per_map": {m: per_map[m] for m in sorted(per_map)},
        "captures": captures,
    }

    out_dir = Path(out) if out is not None else reports_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / REPORT_MD).write_text(_markdown(report), encoding="utf-8")
    report["out_dir"] = str(out_dir)

    if strict:
        check_held_out_invariant(report)
    return report


def check_held_out_invariant(report: dict[str, Any]) -> None:
    """Raise :class:`HarnessConfigError` if the ingest ``report`` lists invariant violations."""
    bad = report.get("wrong_split") or []
    if not bad:
        return
    where = Path(report["out_dir"]) / REPORT_JSON if report.get("out_dir") else REPORT_JSON
    raise HarnessConfigError(
        "held-out-split invariant violated: these new or modified captures on frozen val/held-out "
        f"maps are not in {report.get('frozen_manifest')}: {bad}. UC-84 only grows the train "
        "split; remove them. To deliberately re-baseline the frozen eval set, regenerate the "
        "manifest (drone_fly.eyes.ingest.build_frozen_manifest, see "
        f"docs/experiments/uc84-capture-protocol.md) and commit it. See {where}."
    )


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
    frozen: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from drone_fly.tracks.catalogue import TrackDataError

    rec: dict[str, Any] = {
        "name": cdir.name, "capture": str(cdir), "included": False, "reason": None,
        "map": None, "split": None, "catalogue_id": None,
        "in_view_hits": 0, "in_view_rays": 0, "in_view_hit_ray_share": None, "frames": 0,
        "pairing": None, "capture_sha256": None,
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
        # Held-out-split invariant: only the frozen eval captures (name + sha in the manifest) are
        # allowed; violations are reported here and raised by check_held_out_invariant.
        sha = capture_sha(cdir)
        rec["capture_sha256"] = sha
        fe = (frozen or {}).get(cdir.name)
        if fe is None:
            return fail("wrong-split")
        if fe["capture_sha256"] != sha:
            return fail("frozen-eval-modified")
        return fail("frozen-eval")

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
    nfrozen = len(r.get("frozen_eval") or [])
    if nfrozen:
        lines += [
            f"{nfrozen} frozen val/held-out eval captures recognised (manifest "
            f"`{r.get('frozen_manifest')}`) -- not counted toward the fraction.",
            "",
        ]
    if r["wrong_split"]:
        lines += [
            "> **Held-out-split violation** -- new or modified captures on frozen val/held-out "
            f"maps: {r['wrong_split']}. Remove them; UC-84 only grows the train split.",
            "",
        ]
    if r.get("frozen_eval_missing"):
        lines += [
            "> Warning: frozen-eval manifest entries not found under the captures root: "
            f"{r['frozen_eval_missing']}.",
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
        if c["included"]:
            status = "included"
        elif c["reason"] == "frozen-eval":
            status = "frozen-eval (not counted)"
        else:
            status = f"excluded: {c['reason']}"
        lines.append(
            f"| {c['name']} | {c['map'] or '-'} | {c['split'] or '-'} | {c['frames']} | "
            f"{_fmt(c['in_view_hit_ray_share'])} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)
