# UC-84 — Liftoff obstacle-capture protocol (AC1 + AC2)

How to capture obstacle-dense Liftoff flights that raise the eyes estimator's **in-view hit-ray
fraction** from the ~5 % UC-83 baseline to the **≥ 15 %** UC-84 target, and how to get those
captures into the shape `drone-fly eyes dataset` already expects. This is a *runbook*, not code: the
flights run on your Windows/Liftoff machine; the in-sandbox side only validates the drop and runs
the retrain (see `docs/experiments/uc84-liftoff-obstacle-capture.md` for the AC5 results).

## Scope and the Liftoff/sandbox boundary

Capture **execution and assembly are user-side** (Liftoff is Windows-only; the CUDA sandbox has no
Liftoff). There is no in-repo writer of the `captures/<Map>/<cap>/` layout on this branch — only
readers (`drone_fly.eyes.capture_meta`, `drone_fly.harness.offline.capture`). The 8 captures already
on disk were assembled off-branch by the UC-71 tooling from saved Liftoff race recordings. So the
dev-team deliverable for AC1/AC2 is this protocol plus the in-sandbox backstop (`eyes ingest`, AC3)
and the retrain driver (`scripts/uc84_obstacle_capture.sh`, AC4). Nothing here flies a drone or
re-verifies an alignment.

## In-view hit-ray fraction (the target metric)

The metric UC-84 optimises is the **in-view hit-ray fraction**: of the rays the camera actually sees
(`in_view`), what share strike an obstacle (`hit`) — i.e. `hit[in_view].mean()` over the per-ray
label arrays in `eyes_labels.npz`. Obstacles are taken from the forward cone
(`raycast_obstacles()`), so a frame with an obstacle *ahead* counts even when the pilot later avoids
it. This is **not** `rays_hit_share` (over all 40 rays) nor `in_view_share` (geometry only); it is
the conditional. It is computed by one shared helper — `drone_fly.eyes.dataset.in_view_hit_counts`
— used by both `eyes ingest` and the dataset manifest's `in_view_hit_ray_share`, so the protocol
target and the dataset report can never drift apart.

UC-83 measured ~5 % here on the 5 train maps; the hit head collapsed (in-view recall 0.027) because
there was almost no close-obstacle signal. UC-84 attacks that data ceiling directly.

## Map list (AC1a)

**Keep** the 5 already-captured train maps: BandoCity, Hannover, LiftoffArena, MarinaBay,
PineValley.

**Add** obstacle-dense catalogue train maps that already carry track/race XML in `tracks/liftoff.yaml`:
BardwellsYard, Hall26, ParisDroneFestival, Permafrost, ShortCircuit, StrawBale, TheGreen.

Every new map must be catalogued **`split: train`** (it already is for the seven above). Do **not**
capture on any `val` or `held-out` map — those are frozen so the UC-83-vs-UC-84 comparison stays
valid (see *Held-out-split invariant*). `eyes ingest` hard-fails on any non-train capture, so a
mistake here is caught before it can shift the baseline.

## Course selection (AC1b)

Pick, per map, the race/free-fly course with the **most placed non-gate obstacles in the forward
flight line** — pillars, arches, trees, scaffolding, walls, tunnels — so the forward cone frequently
contains geometry. Prefer tighter, lower, cluttered courses over open high-speed ones. A course
qualifies when a normal lap keeps obstacles ahead for a meaningful fraction of the flight (this is
exactly what the metric rewards); `eyes ingest`'s per-map fraction is the empirical check.

## Driving method (AC1c)

Fly with the **70-input obstacle-aware model `training/11-count-curriculum/model_readout.pt`**
(reproducible, scriptable, and the most obstacle-competent checkpoint we have — the same one used in
UC-82). Drive it through the `liftoff-episodes` skill, which flies N unattended attempts and saves an
in-game recording every K attempts.

**Backup pilot:** the teacher policy, if the 70-input model is unavailable or flies a given course
too conservatively.

## Frame target per map (AC1d)

Size the capture so the *overall* in-view hit-ray fraction reaches ≥ 15 %. As a planning figure, aim
for roughly **1,500–3,000 labelled frames per map** across the ~12 train maps (the existing train
captures are ~1,400–1,500 frames each) — but the frame count is only a means. The **enforced** gate
is AC3: `eyes ingest` computes the achieved fraction and flags `below_target`; keep flying (or use
the fallback below) until it clears 15 %. Capturing more frames on the most obstacle-dense maps
moves the overall number fastest.

## Why ≥ 15 % (AC1e)

15 % is ~3× the ~5 % baseline that UC-83's train-time rebalancing (hit-weighted sampler + focal loss
+ hit-biased dataset filter) could **not** overcome — rebalancing re-weights existing signal, it
cannot add close-obstacle frames that were never flown. Tripling the genuine in-view hit density is
the smallest step with a credible chance of giving the hit head enough signal to stop collapsing.
It may still be insufficient (monocular ranging may need depth); that outcome routes to UC-85 and is
recorded as a negative in the AC5 doc.

## Fallback when the clean pilot flies too clean (AC1f)

A well-trained pilot that dodges everything produces few in-view hits. If `eyes ingest` reports
`below_target` after a normal pass, escalate in this order and re-ingest after each:

1. **Perturb the pilot** — add action noise / jitter so it skims obstacles instead of cleanly
   avoiding them.
2. **Slow down** — lower the target speed so more frames have obstacles still ahead in the cone.
3. **Tighten the course** — pick narrower lines / lower altitude bands with denser geometry.
4. **Sample colliding runs** — keep attempts that do clip obstacles (the frames leading up to a
   collision are exactly the close-obstacle signal we lack).
5. **Manual near-obstacle passes** — hand-fly deliberate close approaches as a last resort.

The fallback is gate-triggered: you only reach for it when AC3 says the clean captures are short.

## Held-out-split invariant

UC-84 **only grows the train split.** The val maps (AutumnFields, DubaiLegends, Rustline, Surtur)
and the held-out maps (HangarC03, MinusTwo, ThePit, TheRussianWoodpecker) are **frozen** — no new
captures — so `eyes eval` on the intact held-out split measures the same thing before and after
UC-84. The split is resolved from `tracks/liftoff.yaml` (`maps.<map>.split`), never from a capture's
`meta.json`. `eyes ingest` asserts every ingested capture resolves to `train` and hard-fails
otherwise; `scripts/uc84_obstacle_capture.sh` additionally logs the frozen set + the catalogue
sha256 so a shifted baseline is detectable after the fact.

## Capture tooling: Fly → Assemble → Drop (AC2)

No new capture-side format is invented. Reuse the UC-62/UC-71 lineage; captures land exactly where
`eyes dataset` already looks (`<data root>/captures/<Map>/<capture>/`).

### 1 — Fly (Liftoff machine)

Use the `liftoff-episodes` skill to fly the 70-input model unattended and save in-game recordings:

- Launch Liftoff, load the last session, fly N attempts on the chosen obstacle-dense course, reset
  between attempts, and save an in-game recording every K attempts; then close Liftoff.
- Run on the Liftoff machine through `.venv-cuda\Scripts\drone-fly.exe` (never a bare `drone-fly` /
  `uv run`), per the skill. Everything is telemetry/pixel-verified; no screenshot reading in the
  normal loop.

### 2 — Assemble (off-branch UC-71 capture tool)

Convert each saved recording into a capture folder with the off-branch UC-71 tool. For each capture
it:

- extracts `frames/` + `frames.csv` and the telemetry stream,
- runs the **alignment + camera-check** that sets `alignment.verified` and
  `alignment.camera_check.ok` (this is the verification step — the dev-team does **not** redo it),
- writes `meta.json` with the catalogue id, map, camera and provenance (see the schema contract).

This tool is **not on this branch** (confirmed: only readers of the layout live here). It is part of
the UC-71 capture lineage on the capture machine.

### 3 — Schema contract (`meta.json`)

Each `<capture>/` folder must contain `frames/` (the RGB frames), `frames.csv`, a telemetry file
(`telemetry_raw.csv` or `telemetry.csv`), and a `meta.json` with **`schema_version: 1`**. Mirror the
**exact field names used by real on-disk captures** — the following is a real train-map `meta.json`
(abridged), copy its shape:

```json
{
  "schema_version": 1,
  "name": "<capture folder name>",
  "catalogue_id": "<id from tracks/liftoff.yaml, e.g. bando-city-01>",
  "map": "<catalogue map, e.g. BandoCity>",
  "split": "train",
  "source": "replay",
  "pose_source": "recording-xml",
  "drone": "drones/liftoff-gremlin-micro",
  "camera": {
    "resolution": [2560, 1080],
    "hfov_deg": 120.0,
    "uptilt_deg": 20.03,
    "source": "recording"
  },
  "alignment": {
    "verified": true,
    "lag_ms": 0.0,
    "offset_s": 0.0,
    "camera_check": { "ok": true, "best_hfov_deg": 120.0, "best_uptilt_deg": 19.03,
                      "meta_hfov_deg": 120.0, "meta_uptilt_deg": 20.03 }
  },
  "race":  { "file": "<Race>.xml",  "sha256": "<sha>" },
  "track": { "file": "<Track>.xml", "sha256": "<sha>" },
  "recording": { "name": "<in-game recording name>", "origin": "user", "sha256": "<sha>" },
  "masks": [],
  "time": { "offset_s": 0.0, "rate": 1.0 }
}
```

**Fields the eyes pipeline reads / requires** (the assembler must get these right):

| Field | Why it matters |
|---|---|
| `schema_version: 1` | `read_capture_meta` rejects any other value. |
| `catalogue_id` | Must resolve in `tracks/liftoff.yaml`; drives split + track geometry. Missing → `no-catalogue`. |
| `map` | Must equal the catalogue entry's map, else `map-mismatch`. |
| `camera.resolution` / `camera.hfov_deg` / `camera.uptilt_deg` | Build the ray camera; an uptilt far from the preset is flagged (train) or excluded (val/held-out). |
| `camera.source: "recording"` | Provenance: camera parameters came from the recording. |
| `alignment.verified: true` | Required; `false`/absent → `alignment-unverified` (excluded). |
| `alignment.lag_ms` | Frame↔telemetry lag used when pairing. |
| `alignment.camera_check.ok` | The camera-check verdict from assembly. |
| `pose_source: "recording-xml"` | Pose provenance (from the recording XML). |

> **Note on `source`.** Real on-disk captures carry a top-level `"source": "replay"` **and**
> `"pose_source": "recording-xml"` **and** `"camera.source": "recording"`. The authoritative
> provenance markers the eyes code relies on are `pose_source` and `camera.source`; `source` is
> recorded verbatim (`read_capture_meta` reads it) but is not used for gating. Reproduce all three as
> shown rather than inventing a different marker.

### 4 — Drop (into the captures root)

Copy each assembled `<capture>/` into `<data root>/captures/<Map>/` on the sandbox machine (the same
root `eyes dataset` / `eyes ingest` read by default — resolve it with `drone-fly` paths). Then
validate (next section). `eyes dataset --label-missing` will pick the new drops up with no code
change.

## Validate the drop — `eyes ingest` (AC3 backstop)

```bash
# Validate every dropped capture, (re)label missing/stale ones, and report the fraction.
uv run drone-fly eyes ingest --label-missing
```

It writes `ingest.json` + `ingest.md` (default `<data root>/reports/eyes/ingest/`) with the overall
and per-map in-view hit-ray fraction, `below_target` (< 15 %), and a per-capture reason for anything
excluded (`no-meta`, `bad-meta`, `no-catalogue`, `map-mismatch`, `alignment-unverified`,
`wrong-split`, `no-split`, `no-trackdata`, `no-scene`, `no-telemetry`, `no-labels`, `stale-labels`).
It **exits non-zero** if any capture resolves to a non-train split. If `below_target`, fly more or
use the fallback above, then re-ingest.

## Run the retrain/compare (AC4)

Once the fraction clears 15 %:

```bash
# Gate on the fraction, then run the UC-83 v2 recipe over the enlarged train data and compare
# on the intact held-out split. Resumable; writes markers + provenance under artifacts/uc84/.
scripts/uc84_obstacle_capture.sh artifacts/uc84
#   --force / UC84_FORCE=1   proceed even if the fraction is still below target
#   UC84_TARGET=0.15         gate threshold
#   UC83_RUN_AC5=1           also run the heavy AC5 / UC-82 rays-vs-blank re-run (off by default)
```

The AC4 verdict (≥ 30 % relative nearest-quarter in-view miss reduction, false-alarm-on-clear
≤ 0.10) and the result table are written up in `docs/experiments/uc84-liftoff-obstacle-capture.md`.
