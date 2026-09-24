"use strict";
/* drone-fly activation playback viewer — vanilla JS, no build step.
 * Loads a recorded episode file (plain JSON or gzip) picked from disk (FileReader /
 * DecompressionStream, so it works from file:// with no server) and renders two panels
 * synced on one timeline: an anatomical brain map rendered as an MRI/fMRI-style activation
 * heatmap over a static registered brain outline (UC-12), and a
 * flight panel (4 action traces + an orbitable 3D flight scene). See use-cases/05, 06 & 12
 * and the recorder schema.
 *
 * The anatomical panel consumes the committed static asset `brain_outline.js` (global
 * `BRAIN_OUTLINE`, loaded via a classic <script> immediately before this file). It is
 * optional: if absent the panel degrades to auto-fit splats with no outline (never a
 * ReferenceError). Regenerate the asset with scripts/build_brain_outline.py. */

const PROJECTIONS = { xz: [0, 2], xy: [0, 1], yz: [1, 2] };
// Anatomical brain-map view presets → projection plane. Only top/xz is anatomically pinned
// (dorsal/top-down, per record/coordinates.py); front↔side is a reversible labeling convention.
const MAP_VIEW_PRESETS = { front: "xy", side: "yz", top: "xz" };
const ACTION_COLORS = ["#ffd54f", "#4fc3f7", "#81c784", "#ff8a65"];
const SUPPORTED_SCHEMA = 1;

// ---- anatomical brain-map heatmap (UC-12) ---------------------------------------------
// The panel is an MRI/fMRI-style activation heatmap: per active neuron an additive
// kernel-density Gaussian splat is stamped (weighted by that neuron's activation MAGNITUDE
// RELATIVE TO REST — |code*scale + offset|, sourced from the recording's own quantization
// meta) into a float accumulation buffer, the buffer is normalized and mapped through a
// "hot" colormap. Weighting by magnitude-from-rest (not the raw uint8 code) is what makes
// the map animate: a resting neuron (real activation ≈ 0, code ≈ 127) reads DARK, and only
// neurons that deviate from rest light up and fade as the state evolves. Using the raw code
// instead put every neuron at ~0.5 brightness permanently (rest → code 127 → 0.5), burying
// the frame-to-frame signal (~0.03% of that constant floor) under a static half-lit blob.
// (black→red→orange→yellow→white), then drawn under a static registered brain outline.
// Per-neuron screen positions are precomputed once per view (not per frame — see MAP_SS).
const MAP_SS = 2; // accumulation-buffer downscale (softness + ~4× fewer stamp writes)
const MAP_KERNEL_R = 7; // Gaussian splat radius, in downscaled buffer pixels
const MAP_KERNEL_SIGMA = MAP_KERNEL_R / 2.4;
const MAP_MIN_WEIGHT = 2 / 255; // skip near-rest neurons (perf; deviation-from-rest below one code step)
// Per-neuron temporal auto-gain: each neuron's magnitude is stretched to its OWN episode
// min→max so slow, small swings still fill the dark→bright range and the map visibly animates
// (magnitude-from-rest alone leaves the pattern nearly constant frame-to-frame — the drift is
// real but tiny). The divisor is floored at MAP_GAIN_MIN_RANGE so a near-rest neuron whose only
// "range" is quantization noise is NOT blown up to full brightness — it stays proportionally dim.
const MAP_GAIN_MIN_RANGE = 0.05;

// ---- UC-28 / UC-59: soma-less afferents + modality overlay ----------------------------
// Soma-less afferents are placed in a schematic fly body around the brain (placement ===
// "schematic"). UC-28 splatted them (fainter) into the brain-canvas; UC-59 RELOCATES them out
// of the brain map entirely — the brain map now shows only real-anatomy neurons (maximized),
// and the soma-less afferents render in the tagged boxes strip below it (see bucketSomaless /
// buildSomalessBoxes / drawBoxes). Their animated activation (the signal the old faint splat
// carried) is preserved in those per-box mini heatmaps.
// Modality-tag overlay colours (AC-7). Exactly the recorder's RECORDED_MODALITIES set — vision
// / proprioceptive / hunger. `damage` is intentionally absent (MaleCNS has no nociceptive
// label; documented as unavailable, never faked). The overlay is drawn ON TOP of the hot
// activation colormap and does not replace it.
const MODALITY_COLORS = {
  vision: "#4fc3f7",
  proprioceptive: "#81c784",
  hunger: "#ff8a65",
};

// ---- 3D flight panel constants --------------------------------------------------------
// World frame (confirmed from env/config.py + adapter): z UP, +x FORWARD, +y = LEFT,
// −y = RIGHT, right-handed. A right bank (+roll → −y) must read as a right turn on screen,
// so the scene vertical is world +z and y is NOT mirrored. AC5 foot-gun — do not "fix" this.
const FLIGHT_FOV = Math.PI / 3; // 60° vertical field of view
const FLIGHT_MAX_PITCH = Math.PI / 2 - 0.01; // clamp shy of straight-down (gimbal guard)
// View presets (AC13): human labels front/side/top-down; top-down is the default. Derived
// from the confirmed up-axis so the user never sees x/y/z:
//   top   = straight down −z  (+x up-screen, −y → screen-right)
//   front = down the +x course/forward axis (altitude vertical)
//   side  = along +y (altitude profile; +x to the right)
const VIEW_PRESETS = {
  front: { yaw: Math.PI, pitch: 0 },
  side: { yaw: -Math.PI / 2, pitch: 0 },
  top: { yaw: Math.PI, pitch: FLIGHT_MAX_PITCH },
};
// gate = the amber ring for gates NOT currently targeted (dimmed); gateTarget = the brighter
// cyan ring for the current target gate (UC-09 AC7). Legacy files (no per-frame target_gate)
// draw every gate with the uniform `gate` colour — no highlight.
const COURSE_COLORS = {
  start: "#66bb6a",
  gate: "#ffca28",
  gateTarget: "#4fc3f7",
  finish: "#ff5252",
  drone: "#ffffff",
  obstacle: "#ba68c8", // UC-15: floor-anchored pillar wireframes (purple, distinct from gates)
  // UC-21: landing-pad floor discs, coloured by kind. Chosen distinct from the markers above:
  // padRecharge green is a brighter shade than the start-green dot; padRepair is a deep orange
  // distinct from gate-yellow and finish-red; padPlain is a neutral grey.
  padRecharge: "#00e676",
  padRepair: "#ff6d00",
  padPlain: "#90a4ae",
};

// tiny vec3 helpers (plain arrays, no deps)
const v3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ],
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  norm: (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
};

//: Live 3D flight controller over a single reused canvas. Created on file load and torn
//: down (destroy()) on the next load so no canvas context leaks (AC9 lifecycle).
let flight = null;

const el = (id) => document.getElementById(id);
const state = {
  data: null,
  frame: 0,
  playing: false,
  speed: 1,
  mapNorm: "frame", // brain-map intensity normalization: "frame" (per-frame) | "global" (AC5)
  mapCache: null, // per-view brain-map cache (screen positions, transform, global peak) — see ensureMapCache
  somaBoxes: null, // UC-59: per-recording soma-less tagged boxes [{canvas, indices}] — see buildSomalessBoxes
  lastTs: 0,
  acc: 0,
};

// ---- file loading ---------------------------------------------------------------------
el("file-input").addEventListener("change", async (ev) => {
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  setStatus(`loading ${file.name}…`);
  try {
    const text = await readMaybeGzip(file);
    const doc = JSON.parse(text);
    loadDocument(doc, file.name);
  } catch (err) {
    setStatus(`failed to load: ${err.message}`, true);
    console.error(err);
  }
});

async function readMaybeGzip(file) {
  const isGzip = file.name.endsWith(".gz");
  if (!isGzip) return await file.text();
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser lacks DecompressionStream; gunzip the file first");
  }
  const stream = file.stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).text();
}

function setStatus(msg, isError) {
  const s = el("load-status");
  s.textContent = msg;
  s.classList.toggle("error", !!isError);
}

// ---- document load --------------------------------------------------------------------
function loadDocument(doc, name) {
  if (!doc || typeof doc !== "object" || !doc.meta || !doc.frames) {
    throw new Error("not a recognised recording (missing meta/frames)");
  }
  if (doc.schema_version !== SUPPORTED_SCHEMA) {
    throw new Error(`unsupported schema_version ${doc.schema_version} (need ${SUPPORTED_SCHEMA})`);
  }
  const nFrames = doc.frames.activations.length;
  const nNeurons = doc.meta.n_neurons;
  if (nFrames && doc.frames.activations[0].length !== nNeurons) {
    throw new Error("frame width does not match n_neurons");
  }

  state.data = doc;
  state.frame = 0;
  state.playing = false;
  state.mapCache = null; // rebuilt lazily by ensureMapCache() on the next brain-map draw
  state.actGain = null; // per-neuron temporal auto-gain, recomputed lazily for this recording
  const normSel = el("map-norm-select");
  state.mapNorm = normSel && normSel.value === "global" ? "global" : "frame";

  renderMetaBar(doc, name);
  renderPositionSource(doc.meta.positions);
  renderFlightLegend(doc.meta.action_layout);
  buildSomalessBoxes(); // UC-59: (re)build the soma-less tagged boxes for this recording
  renderOutcome(doc.outcome);

  // (Re)build the 3D flight scene: tear down any previous controller (AC9), create a fresh
  // one over the single reused canvas, load this episode's geometry, and apply the current
  // view preset (top-down by default on first load).
  destroyFlight3D();
  flight = createFlight3D(el("flight-canvas"));
  flight.setScene(doc);
  flight.applyPreset(el("view-select").value);

  const scrubber = el("scrubber");
  scrubber.max = Math.max(0, nFrames - 1);
  scrubber.value = 0;

  el("app").classList.remove("hidden");
  el("play-btn").textContent = "▶ Play";
  setStatus(`${name} — ${nFrames} frames, ${nNeurons} neurons`);
  renderAll();
}

function renderMetaBar(doc, name) {
  const m = doc.meta;
  const bar = el("meta-bar");
  bar.classList.remove("hidden");
  bar.innerHTML =
    `<span><b>episode</b> ${m.episode_index}</span>` +
    `<span><b>neurons</b> ${m.n_neurons}</span>` +
    `<span><b>frames</b> ${m.n_frames}</span>` +
    `<span><b>seed</b> ${m.seed ?? "—"}</span>` +
    `<span><b>backend</b> ${m.backend ?? "—"}</span>` +
    `<span><b>dt</b> ${m.dt ?? "—"} s</span>`;
}

function renderPositionSource(positions) {
  const badge = el("pos-source");
  const anatomical = /^anatomical/i.test(positions.source || "");
  badge.textContent = anatomical ? "anatomical soma positions" : "computed layout (NOT anatomical)";
  badge.className = "pos-source " + (anatomical ? "anatomical" : "computed");
  badge.title = positions.source || "";
}

function renderFlightLegend(layout) {
  const names = layout || ["throttle", "roll", "pitch", "yaw"];
  el("flight-legend").innerHTML = names
    .map((n, i) => `<span><span class="dot" style="background:${ACTION_COLORS[i % 4]}"></span>${n}</span>`)
    .join("");
}

function renderOutcome(outcome) {
  const o = outcome || {};
  const time = o.completion_time == null ? "—" : `${o.completion_time.toFixed(2)} s`;
  el("outcome").innerHTML =
    `<span><b>completed</b> ${o.completed ? "yes" : "no"}</span>` +
    `<span><b>completion time</b> ${time}</span>` +
    `<span><b>total reward</b> ${o.total_reward == null ? "—" : o.total_reward.toFixed(2)}</span>` +
    `<span><b>steps</b> ${o.steps ?? "—"}</span>`;
}

// ---- rendering ------------------------------------------------------------------------
function renderAll() {
  if (!state.data) return;
  el("scrubber").value = state.frame;
  const n = state.data.frames.activations.length;
  el("frame-label").textContent = `frame ${state.frame} / ${Math.max(0, n - 1)}`;
  drawBrainMap();
  drawBoxes(); // UC-59: animate the soma-less tagged boxes on the same timeline (AC6/AC8)
  drawActions();
  if (flight) flight.render();
}

function projectedPoints() {
  const pos = state.data.meta.positions;
  const plane = MAP_VIEW_PRESETS[el("map-view-select").value] || "xz";
  const [a0, a1] = PROJECTIONS[plane] || PROJECTIONS.xz;
  const isDefault = plane === (pos.projection || "xz");
  // UC-28: prefer display3d — the full-coverage finite render coords (real soma or schematic
  // body cluster) — so every neuron renders on EVERY plane (AC-2). Legacy recordings without
  // display3d fall back to the UC-27 behaviour (real coords3d, else the default-plane coords2d).
  const disp = pos.display3d;
  const pts = new Array(pos.has_position.length);
  for (let i = 0; i < pts.length; i++) {
    const d3 = disp && disp[i];
    if (d3) { pts[i] = [d3[a0], d3[a1]]; continue; }
    const c3 = pos.coords3d[i];
    if (c3 != null) pts[i] = [c3[a0], c3[a1]];
    else if (isDefault && pos.coords2d[i]) pts[i] = pos.coords2d[i].slice();
    else pts[i] = null; // no 3D anatomy for this neuron on a non-default plane
  }
  return pts;
}

function bounds(pts) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (!p) continue;
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  if (!isFinite(minX)) return { minX: 0, minY: 0, maxX: 1, maxY: 1 };
  return { minX, minY, maxX, maxY };
}

// MRI/fMRI "hot" colormap: t∈[0,1] → [r,g,b] (black→red→orange→yellow→white). Orange is the
// natural blend where the red plateau overlaps the rising green channel.
function hotColormap(t) {
  const r = clamp01(t / 0.375);
  const g = clamp01((t - 0.375) / 0.375);
  const b = clamp01((t - 0.75) / 0.25);
  return [Math.round(255 * r), Math.round(255 * g), Math.round(255 * b)];
}
function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// Precomputed Gaussian splat kernel (built once — constants are fixed). Each active neuron
// stamps this kernel, weighted by its activation, additively into the accumulation buffer.
const MAP_KERNEL = (() => {
  const R = MAP_KERNEL_R, size = 2 * R + 1, s2 = 2 * MAP_KERNEL_SIGMA * MAP_KERNEL_SIGMA;
  const data = new Float32Array(size * size);
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      data[(dy + R) * size + (dx + R)] = Math.exp(-(dx * dx + dy * dy) / s2);
    }
  }
  return { R, size, data };
})();

// Reused offscreen for the downscaled colorized heatmap (upscaled with smoothing on blit).
let _mapOffscreen = null;
function mapOffscreen(w, h) {
  if (!_mapOffscreen) _mapOffscreen = document.createElement("canvas");
  if (_mapOffscreen.width !== w || _mapOffscreen.height !== h) {
    _mapOffscreen.width = w;
    _mapOffscreen.height = h;
  }
  return _mapOffscreen;
}

// Fixed voxel→canvas transform: UNIFORM scale (preserve aspect) + centering + the same
// `H - y` Y-flip the splats use, so the outline polygon and the activation splats co-register
// exactly (they share this transform). Replaces the old per-recording non-uniform stretch-fit.
function makeTransform(W, H, pad, uMin, uMax, vMin, vMax) {
  const extU = uMax - uMin || 1, extV = vMax - vMin || 1;
  const s = Math.min((W - 2 * pad) / extU, (H - 2 * pad) / extV);
  const offX = (W - extU * s) / 2, offY = (H - extV * s) / 2;
  return { s, pt: (u, v) => [offX + (u - uMin) * s, H - offY - (v - vMin) * s] };
}

// Per-neuron temporal auto-gain (memoized per recording, not per view — it depends only on the
// activations, not the projection plane). For each neuron: lo = min magnitude-from-rest over all
// frames, inv = 1 / max(range, MAP_GAIN_MIN_RANGE). The splat weight is then clamp01((mag-lo)*inv),
// so a neuron that swings across the episode fills the full dark→bright range while a near-static
// neuron stays dim/dark. Sourced from the file's own quantization meta (canonical fallback).
function ensureActivationGain() {
  if (state.actGain) return state.actGain;
  const meta = state.data.meta || {};
  const ascale = typeof meta.activation_scale === "number" ? meta.activation_scale : 2 / 255;
  const aoffset = typeof meta.activation_offset === "number" ? meta.activation_offset : -1;
  const frames = state.data.frames.activations;
  const nF = frames.length;
  const N = meta.n_neurons || (nF ? frames[0].length : 0);
  const lo = new Float32Array(N).fill(Infinity);
  const hi = new Float32Array(N).fill(-Infinity);
  for (let f = 0; f < nF; f++) {
    const a = frames[f];
    for (let i = 0; i < N; i++) {
      const mag = Math.abs(a[i] * ascale + aoffset);
      if (mag < lo[i]) lo[i] = mag;
      if (mag > hi[i]) hi[i] = mag;
    }
  }
  const inv = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    if (!isFinite(lo[i])) { lo[i] = 0; inv[i] = 0; continue; }
    inv[i] = 1 / Math.max(hi[i] - lo[i], MAP_GAIN_MIN_RANGE);
  }
  state.actGain = { lo, inv, ascale, aoffset };
  return state.actGain;
}

// Build (once per view) the brain-map cache: per-neuron accumulation-buffer positions, the
// active transform, the registered outline polygon (fixed mode only), and the exact global
// normalization peak. Rebuilt when the view plane or canvas size changes.
function ensureMapCache(W, H, pad) {
  const viewKey = el("map-view-select").value;
  const c = state.mapCache;
  if (c && c.viewKey === viewKey && c.W === W && c.H === H) return c;

  const pos = state.data.meta.positions;
  const plane = MAP_VIEW_PRESETS[viewKey] || "xz";
  const [a0, a1] = PROJECTIONS[plane] || PROJECTIONS.xz;
  const pts = projectedPoints();
  const anatomical = /^anatomical/i.test(pos.source || "");
  // UC-59: soma-less afferents (placement === "schematic") are relocated to the tagged boxes and
  // no longer stamp into the brain map; size the transform to the REAL-ANATOMY points only so the
  // brain fills the canvas (AC4). `anatPts` masks schematic entries to null.
  const placement = pos.placement || null;
  const isSchematic = (i) => placement != null && placement[i] === "schematic";
  const anatPts = pts.map((p, i) => (isSchematic(i) ? null : p));
  const hasAnat = anatPts.some(Boolean);
  // ⟨C1⟩ Missing-asset guard: `typeof` on an undeclared identifier never throws. Without the
  // asset (or for non-anatomical/legacy recordings) we degrade to auto-fit splats, no outline.
  const outline = typeof BRAIN_OUTLINE !== "undefined" ? BRAIN_OUTLINE : null;
  const fixed = anatomical && hasAnat && outline != null;

  let tf, poly = null;
  if (fixed) {
    const mn = outline.bbox3d.min, mx = outline.bbox3d.max;
    // UC-59: size the fixed transform to the UNION of the registered outline bbox and the
    // NON-schematic points' bounds — so any real-anatomy neuron sitting just outside the outline
    // still stays on-canvas, while the schematic body clusters (now boxed) no longer widen it.
    const b = bounds(anatPts);
    const uMin = Math.min(mn[a0], b.minX), uMax = Math.max(mx[a0], b.maxX);
    const vMin = Math.min(mn[a1], b.minY), vMax = Math.max(mx[a1], b.maxY);
    tf = makeTransform(W, H, pad, uMin, uMax, vMin, vMax);
    const pl = outline.planes && outline.planes[viewKey];
    if (pl && pl.polygon) poly = pl.polygon.map(([u, v]) => tf.pt(u, v));
  } else {
    // AC8 graceful degradation: fit the splats to whatever real-anatomy points exist (schematic
    // excluded); fall back to all points only when there are no non-schematic ones at all.
    const b = bounds(hasAnat ? anatPts : pts);
    tf = makeTransform(W, H, pad, b.minX, b.maxX, b.minY, b.maxY);
  }

  // Quantization mapping (real activation = code * scale + offset) and per-neuron temporal
  // auto-gain, both sourced from the file's own meta (canonical fallback for legacy files).
  const { lo: gainLo, inv: gainInv, ascale, aoffset } = ensureActivationGain();

  const bw = Math.max(1, Math.ceil(W / MAP_SS)), bh = Math.max(1, Math.ceil(H / MAP_SS));
  const sx = new Int32Array(pts.length), sy = new Int32Array(pts.length);
  const valid = new Uint8Array(pts.length);
  // UC-59: only real-anatomy neurons stamp into the brain map. Schematic entries are masked to
  // null in anatPts, so they get valid=0 here — they contribute no splat, and drawModalityOverlay
  // auto-drops their rings via the same !valid[i] guard (their modality now shows in the boxes).
  for (let i = 0; i < pts.length; i++) {
    const p = anatPts[i];
    if (!p) continue; // null (schematic, or no coords on this plane) → does not contribute
    const xy = tf.pt(p[0], p[1]);
    sx[i] = Math.round(xy[0] / MAP_SS);
    sy[i] = Math.round(xy[1] / MAP_SS);
    valid[i] = 1;
  }

  const cache = {
    viewKey, W, H, plane, tf, poly, fixed, bw, bh, sx, sy, valid, ascale, aoffset,
    gainLo, gainInv,
    buf: new Float32Array(bw * bh),
    img: el("brain-canvas").getContext("2d").createImageData(bw, bh),
    globalPeak: 0,
  };
  cache.globalPeak = computeGlobalPeak(cache); // ⟨C3⟩ exact global max, one-time per view
  state.mapCache = cache;
  return cache;
}

// Zero the accumulation buffer and stamp every active neuron's weighted Gaussian into it for
// one frame's activations. Returns the frame's peak intensity (for per-frame normalization).
function stampFrame(cache, act) {
  const { buf, bw, bh, sx, sy, valid, ascale, aoffset, gainLo, gainInv } = cache;
  buf.fill(0);
  const R = MAP_KERNEL.R, size = MAP_KERNEL.size, kd = MAP_KERNEL.data;
  let peak = 0;
  for (let i = 0; i < valid.length; i++) {
    if (!valid[i]) continue;
    // Weight by activation MAGNITUDE RELATIVE TO REST (|real activation| = |code*scale + offset|),
    // NOT the raw code: a resting neuron (real ≈ 0) is dark, so only deviations light up and fade
    // as the state evolves. Then apply per-neuron temporal auto-gain — stretch to this neuron's own
    // episode min→max (floored divisor) so slow, small swings still fill the dark→bright range and
    // the map visibly animates. UC-59: only real-anatomy neurons reach here (valid[i]===0 for schematic).
    const mag = Math.abs(act[i] * ascale + aoffset);
    const g = clamp01((mag - gainLo[i]) * gainInv[i]);
    const w = g;
    if (w < MAP_MIN_WEIGHT) continue; // perf: skip near-silent (near-rest) neurons
    const cx = sx[i], cy = sy[i];
    for (let dy = -R; dy <= R; dy++) {
      const py = cy + dy;
      if (py < 0 || py >= bh) continue;
      const krow = (dy + R) * size + R;
      const brow = py * bw;
      for (let dx = -R; dx <= R; dx++) {
        const px = cx + dx;
        if (px < 0 || px >= bw) continue;
        const val = buf[brow + px] + w * kd[krow + dx];
        buf[brow + px] = val;
        if (val > peak) peak = val;
      }
    }
  }
  return peak;
}

// ⟨C3⟩ Exact global normalization peak: replay every frame's accumulation once and track the
// true maximum. One-time per view/canvas-size; O(nFrames × neurons × kernel). Warns (does not
// approximate or clip) for very large recordings.
function computeGlobalPeak(cache) {
  const frames = state.data.frames.activations;
  const nF = frames.length;
  if (nF * cache.valid.length > 4000000) {
    console.warn(
      `brain-map global-norm pre-pass: ${nF} frames × ${cache.valid.length} neurons — may be slow`,
    );
  }
  let peak = 0;
  for (let f = 0; f < nF; f++) {
    const p = stampFrame(cache, frames[f]);
    if (p > peak) peak = p;
  }
  return peak;
}

// Anatomical panel (UC-12): an MRI/fMRI-style activation heatmap over a static registered
// brain outline. No per-neuron dots. Region intensity = summed activation of the neurons whose
// soma coordinates fall there, drawn as additive kernel-density splats (AC2), normalized either
// per-frame or against the fixed global peak (AC5), mapped through a "hot" colormap, and drawn
// under the outline polygon (AC4). Degrades to auto-fit splats with no outline for non-anatomical
// or legacy recordings (AC8).
function drawBrainMap() {
  const canvas = el("brain-canvas");
  resizeBackingStore(canvas); // UC-59: DPR-aware backing store from the stable CSS box (AC4)
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 18;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, W, H);
  if (!state.data) return;

  const cache = ensureMapCache(W, H, pad);
  const act = state.data.frames.activations[state.frame];
  const frameMax = stampFrame(cache, act);
  // AC5 normalization: per-frame uses this frame's own peak (punchy "lights up" contrast);
  // global uses the fixed cross-frame peak (frame-to-frame comparable). Clamped to [0,1].
  const norm = state.mapNorm === "global" ? cache.globalPeak : frameMax;
  const inv = norm > 1e-9 ? 1 / norm : 0;
  const buf = cache.buf, img = cache.img, data = img.data;
  for (let j = 0; j < buf.length; j++) {
    const p = j * 4;
    const t = clamp01(buf[j] * inv);
    if (t <= 0) {
      data[p] = data[p + 1] = data[p + 2] = data[p + 3] = 0;
      continue;
    }
    const rgb = hotColormap(t);
    data[p] = rgb[0];
    data[p + 1] = rgb[1];
    data[p + 2] = rgb[2];
    data[p + 3] = Math.round(255 * clamp01(t * 1.25)); // soft alpha ramp near the low end
  }

  // Blit the downscaled colorized buffer up to the canvas with smoothing (MRI-style softness),
  // composited additively ("lighter") over the near-black panel — the heatmap is a sum of
  // additive kernel-density splats.
  const off = mapOffscreen(cache.bw, cache.bh);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(off, 0, 0, cache.bw, cache.bh, 0, 0, W, H);
  ctx.globalCompositeOperation = "source-over";

  // Static registered brain outline on top (AC4): faint fill + stroke. Fixed mode only.
  if (cache.poly && cache.poly.length > 1) {
    ctx.beginPath();
    ctx.moveTo(cache.poly[0][0], cache.poly[0][1]);
    for (let i = 1; i < cache.poly.length; i++) ctx.lineTo(cache.poly[i][0], cache.poly[i][1]);
    ctx.closePath();
    ctx.fillStyle = "rgba(120,140,190,0.05)";
    ctx.fill();
    ctx.strokeStyle = "rgba(150,170,220,0.55)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // UC-28 AC-7: modality-tag overlay, drawn ON TOP of the hot colormap (does not replace it).
  drawModalityOverlay(cache);
}

// UC-28 AC-7: when a modality is selected, ring every neuron tagged with it (from meta.modality)
// at its screen position, in a distinct colour. This is an additive overlay — it never touches
// the underlying activation heatmap colours. A no-op when the toggle is "off", the recording has
// no meta.modality (legacy files), or the modality is unknown. Uses source-over so the rings sit
// crisply above the additively-composited heatmap.
function drawModalityOverlay(cache) {
  const sel = el("modality-tag-select");
  const value = sel ? sel.value : "off";
  if (!value || value === "off") return;
  const mod = state.data.meta.modality;
  if (!mod) return;
  const color = MODALITY_COLORS[value];
  if (!color) return;
  const ctx = el("brain-canvas").getContext("2d");
  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  const n = Math.min(mod.length, cache.valid.length);
  for (let i = 0; i < n; i++) {
    if (mod[i] !== value || !cache.valid[i]) continue;
    const x = cache.sx[i] * MAP_SS, y = cache.sy[i] * MAP_SS;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawActions() {
  const canvas = el("actions-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 8;
  ctx.clearRect(0, 0, W, H);
  const actions = state.data.frames.actions;
  const nFrames = actions.length;
  // shared vertical scale over [-1, 1]
  const toY = (v) => H - pad - ((v + 1) / 2) * (H - 2 * pad);
  // zero / mid gridline
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath(); ctx.moveTo(pad, toY(0)); ctx.lineTo(W - pad, toY(0)); ctx.stroke();
  const toX = (f) => pad + (nFrames > 1 ? (f / (nFrames - 1)) * (W - 2 * pad) : 0);
  for (let ch = 0; ch < 4; ch++) {
    ctx.strokeStyle = ACTION_COLORS[ch];
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let f = 0; f < nFrames; f++) {
      const x = toX(f), y = toY(actions[f][ch]);
      if (f === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  // playhead
  const px = toX(state.frame);
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
}

// ---- UC-59: soma-less tagged boxes ----------------------------------------------------
// Shared DPR-aware backing-store sizer (mirrors createFlight3D.resize): set canvas.width/height
// from the element's CURRENT CSS box × min(DPR, 2). Both #brain-canvas and each .soma-box-canvas
// have a stable CSS box that is NOT derived from the backing store (brain = width-driven square via
// aspect-ratio:1/1; boxes = fixed strip height), so re-deriving the backing store here never feeds
// back into layout — no ResizeObserver oscillation. Returns true when the backing store changed.
function resizeBackingStore(canvas) {
  const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  const cssW = canvas.clientWidth || canvas.width || 1;
  const cssH = canvas.clientHeight || canvas.height || 1;
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width === w && canvas.height === h) return false;
  canvas.width = w;
  canvas.height = h;
  return true;
}

// Partition the soma-less (schematic) neurons into modality-tagged buckets. PURE, DOM-free, and
// SELF-CONTAINED (its known-tag set is inlined — NO closure over module-scope constants) so it can
// be extracted from source and executed under node in isolation. It is the SINGLE source of truth
// for box grouping: both buildSomalessBoxes (DOM) and drawBoxes consume its output; neither
// re-derives the partition. Contract (UC-59 AC5/AC7):
//   - considers ONLY placement[i] === "schematic"; every such neuron lands in EXACTLY one bucket;
//   - modality "vision" → "vision (external)", "proprioceptive" → "proprioceptive",
//     "hunger" → "hunger"; ANY other value ("", null/undefined, or an unknown/legacy tag) →
//     the catch-all "other (untagged)" (none dropped);
//   - fixed bucket order (vision, proprioceptive, hunger, other); zero-member buckets omitted;
//   - zero schematic neurons (or no placement array) → [] (empty result).
function bucketSomaless(placement, modality) {
  const KNOWN = ["vision", "proprioceptive", "hunger"];
  const TITLES = { vision: "vision (external)", proprioceptive: "proprioceptive", hunger: "hunger" };
  const OTHER_KEY = "other";
  const OTHER_TITLE = "other (untagged)";
  const order = ["vision", "proprioceptive", "hunger", OTHER_KEY];
  const members = { vision: [], proprioceptive: [], hunger: [], other: [] };
  const n = placement ? placement.length : 0;
  for (let i = 0; i < n; i++) {
    if (placement[i] !== "schematic") continue;
    const tag = modality ? modality[i] : null;
    if (KNOWN.indexOf(tag) !== -1) members[tag].push(i);
    else members[OTHER_KEY].push(i);
  }
  const out = [];
  for (const key of order) {
    if (!members[key].length) continue;
    out.push({ key, title: key === OTHER_KEY ? OTHER_TITLE : TITLES[key], indices: members[key] });
  }
  return out;
}

// (Re)build the soma-less tagged-boxes DOM for the current recording — one outlined box per
// non-empty modality bucket (bucketSomaless is the single source of truth), each a top-center title
// + a mini heatmap canvas that drawBoxes animates. Zero schematic (or a legacy file with no
// placement) → a single muted note, no layout break. Records [{canvas, indices}] in state.somaBoxes.
function buildSomalessBoxes() {
  const host = el("somaless-boxes");
  state.somaBoxes = null;
  if (!host) return;
  host.textContent = "";
  if (!state.data) return;
  const pos = state.data.meta.positions || {};
  const buckets = bucketSomaless(pos.placement || null, state.data.meta.modality || null);
  if (!buckets.length) {
    const note = document.createElement("div");
    note.className = "soma-empty";
    note.textContent = "no soma-less afferents in this recording";
    host.appendChild(note);
    return;
  }
  const boxes = [];
  for (const b of buckets) {
    const box = document.createElement("div");
    box.className = "soma-box";
    const title = document.createElement("div");
    title.className = "soma-box-title";
    title.textContent = b.title;
    const canvas = document.createElement("canvas");
    canvas.className = "soma-box-canvas";
    box.appendChild(title);
    box.appendChild(canvas);
    host.appendChild(box);
    boxes.push({ canvas, indices: b.indices });
  }
  state.somaBoxes = boxes;
}

// Animate each tagged box's member neurons' live activation — a mini per-box heatmap, redrawn from
// renderAll on play/scrub/speed (AC6/AC8). Reuses ensureActivationGain() (the exact
// magnitude-from-rest→[0,1] signal the old schematic-body splat carried) and hotColormap(), packing
// members into a near-square grid of cells. No-op for recordings with no soma-less afferents.
function drawBoxes() {
  const boxes = state.somaBoxes;
  if (!boxes || !boxes.length || !state.data) return;
  const act = state.data.frames.activations[state.frame];
  if (!act) return;
  const { lo, inv, ascale, aoffset } = ensureActivationGain();
  for (const box of boxes) {
    resizeBackingStore(box.canvas);
    const ctx = box.canvas.getContext("2d");
    const W = box.canvas.width, H = box.canvas.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0a0c17"; // near-black backdrop, matching the other canvases
    ctx.fillRect(0, 0, W, H);
    const idx = box.indices, M = idx.length;
    if (!M) continue;
    const cols = Math.max(1, Math.round(Math.sqrt((M * W) / Math.max(1, H))));
    const cw = W / cols, ch = H / Math.ceil(M / cols);
    for (let k = 0; k < M; k++) {
      const i = idx[k];
      const mag = Math.abs(act[i] * ascale + aoffset);
      const t = clamp01((mag - lo[i]) * inv[i]);
      const rgb = hotColormap(t);
      const x = (k % cols) * cw, y = Math.floor(k / cols) * ch;
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(cw), Math.ceil(ch));
    }
  }
}

// ---- 3D flight scene ------------------------------------------------------------------
// A dependency-free canvas-2D perspective projector (ported from the owner's
// liftoff-flight-analyzer approach, extended with a floor grid + start/gate/finish markers).
// Genuinely 3D & orbitable (AC1), no three.js, no npm, no ES modules — stays file://-safe.

// Build the scene geometry once per loaded episode: the flight path plus the display extent
// (bbox over the path ∪ course anchors, padded, z clamped to the arena) that sizes the floor
// and the finish plane so the floor always contains the trajectory. Works with or without
// meta.course (graceful degradation — AC8): without it, no markers and the floor sits at the
// path's own minimum z.
function buildFlightScene(doc) {
  const path = (doc.frames.drone_position || []).map((p) => [p[0], p[1], p[2]]);
  const course = doc.meta.course || null;
  // N gates as an array, with a legacy single-`gate` fallback (UC-09 AC7): new files carry
  // `course.gates: [...]`; pre-UC-09 files carry a singular `course.gate`.
  const gates = courseGates(course);

  // Obstacle pillars (UC-15), or [] when the recording has no `obstacles` field (graceful
  // degradation — older files + no-obstacle runs draw no pillars).
  const obstacles = (course && course.obstacles) || [];

  // Landing pads (UC-21), or [] when the recording has no `pads` field (graceful degradation —
  // older files + no-pad runs draw no pads).
  const pads = (course && course.pads) || [];

  // Anchor points that must always be inside the display extent — every gate centre plus
  // start and finish (and every pillar's footprint extents), so the floor/finish plane always
  // contain the whole course.
  const anchors = path.slice();
  if (course) {
    if (course.start) anchors.push(course.start);
    for (const g of gates) {
      if (g && g.center) anchors.push(g.center);
    }
    // Pillar bounding extents so the floor/bbox always contains every obstacle.
    const fz = course.floor_z != null ? course.floor_z : 0;
    for (const o of obstacles) {
      if (!o || !o.center) continue;
      const r = o.radius || 0;
      const top = fz + (o.height || 0);
      anchors.push([o.center[0] - r, o.center[1] - r, fz]);
      anchors.push([o.center[0] + r, o.center[1] + r, top]);
    }
    // Pad footprints (UC-21) so the floor/bbox always contains every pad disc (floor-anchored).
    for (const p of pads) {
      if (!p || !p.center) continue;
      const r = p.radius || 0;
      anchors.push([p.center[0] - r, p.center[1] - r, fz]);
      anchors.push([p.center[0] + r, p.center[1] + r, fz]);
    }
    if (course.finish && course.finish.x != null) {
      // finish anchor: use the LAST gate's lateral centre at the finish x so x-extent reaches
      // it (the finish inherits the last gate's y/z — see geometry.current_target).
      const lastGate = gates.length ? gates[gates.length - 1] : null;
      const cy = lastGate && lastGate.center ? lastGate.center[1] : 0;
      const cz = lastGate && lastGate.center ? lastGate.center[2] : 0;
      anchors.push([course.finish.x, cy, cz]);
    }
  }

  const bb = bbox3(anchors.length ? anchors : [[0, 0, 0], [1, 1, 1]]);
  // Pad ~10% laterally so nothing hugs the edge.
  const padX = (bb.max[0] - bb.min[0] || 1) * 0.1;
  const padY = (bb.max[1] - bb.min[1] || 1) * 0.1;
  bb.min[0] -= padX; bb.max[0] += padX;
  bb.min[1] -= padY; bb.max[1] += padY;

  // Vertical extent: clamp to the arena [floor_z, ceiling_z] when the course provides it.
  const floorZ = course && course.floor_z != null ? course.floor_z : bb.min[2];
  const ceilZ = course && course.ceiling_z != null ? course.ceiling_z : bb.max[2];
  bb.min[2] = Math.min(bb.min[2], floorZ);
  bb.max[2] = Math.max(bb.max[2], ceilZ);

  const center = [
    (bb.min[0] + bb.max[0]) / 2,
    (bb.min[1] + bb.max[1]) / 2,
    (bb.min[2] + bb.max[2]) / 2,
  ];
  const radius =
    0.5 * v3.len([bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]]) || 1;

  return { path, course, gates, obstacles, pads, bb, center, radius, floorZ, ceilZ };
}

// Normalise a meta.course into an array of gate specs (UC-09 AC7). New recordings carry
// `course.gates: [...]`; pre-UC-09 recordings carry a singular `course.gate`. Returns [] when
// there is no course (graceful degradation).
function courseGates(course) {
  if (!course) return [];
  return course.gates || (course.gate ? [course.gate] : []);
}

// The current target-gate index at the playhead frame (UC-09 AC7), or null when the recording
// has no per-frame `target_gate` track (legacy/single-gate files → uniform, no highlight).
function targetGateAtPlayhead() {
  const tg = state.data && state.data.frames ? state.data.frames.target_gate : null;
  if (!tg || !tg.length) return null;
  const i = Math.min(state.frame, tg.length - 1);
  return tg[i];
}

function bbox3(pts) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of pts) {
    if (!p) continue;
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  for (let k = 0; k < 3; k++) {
    if (!isFinite(min[k])) { min[k] = 0; max[k] = 1; }
    if (max[k] - min[k] < 1e-6) { min[k] -= 0.5; max[k] += 0.5; }
  }
  return { min, max };
}

function createFlight3D(canvas) {
  const ctx = canvas.getContext("2d");
  // Orbit camera in spherical coords about the scene centre; scene up = world +z.
  const cam = { yaw: VIEW_PRESETS.top.yaw, pitch: VIEW_PRESETS.top.pitch, dist: 10 };
  let scene = null;
  let dpr = 1, cssW = canvas.clientWidth || 440, cssH = canvas.clientHeight || 320;
  const drag = { active: false, x: 0, y: 0 };

  function resize() {
    dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    cssW = canvas.clientWidth || cssW;
    cssH = canvas.clientHeight || cssH;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    render();
  }

  // Build the view basis and project a world point to CSS-pixel screen space + depth.
  function project(p) {
    const eye = v3.add(scene.center, [
      cam.dist * Math.cos(cam.pitch) * Math.cos(cam.yaw),
      cam.dist * Math.cos(cam.pitch) * Math.sin(cam.yaw),
      cam.dist * Math.sin(cam.pitch),
    ]);
    const f = v3.norm(v3.sub(scene.center, eye)); // forward (into screen)
    let r = v3.cross(f, [0, 0, 1]); // right = forward × world-up
    if (v3.len(r) < 1e-6) r = v3.cross(f, [1, 0, 0]); // gimbal fallback (straight up/down)
    r = v3.norm(r);
    const u = v3.cross(r, f); // true up = right × forward
    const rel = v3.sub(p, eye);
    const camZ = v3.dot(rel, f); // depth
    if (camZ <= 0.01) return null; // behind the camera
    const focal = 0.5 * cssH / Math.tan(FLIGHT_FOV / 2);
    const s = focal / camZ;
    return { x: cssW / 2 + v3.dot(rel, r) * s, y: cssH / 2 - v3.dot(rel, u) * s, z: camZ };
  }

  function line(a, b, style, width) {
    const pa = project(a), pb = project(b);
    if (!pa || !pb) return;
    ctx.strokeStyle = style;
    ctx.lineWidth = width || 1;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  function dot(p, color, radiusPx) {
    const s = project(p);
    if (!s) return;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(s.x, s.y, radiusPx, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawFloorGrid() {
    const bb = scene.bb, z = scene.floorZ;
    const N = 10;
    ctx.save();
    for (let i = 0; i <= N; i++) {
      const tx = bb.min[0] + (bb.max[0] - bb.min[0]) * (i / N);
      const ty = bb.min[1] + (bb.max[1] - bb.min[1]) * (i / N);
      const grid = "rgba(120,140,190,0.22)";
      line([tx, bb.min[1], z], [tx, bb.max[1], z], grid, 1);
      line([bb.min[0], ty, z], [bb.max[0], ty, z], grid, 1);
    }
    ctx.restore();
  }

  // Draw a single gate as a ring of radius `rad` in the y–z plane at `center`. When a
  // target track exists (`highlightActive`), the current target gate is brighter/thicker in
  // COURSE_COLORS.gateTarget and non-targets are dimmed; without a track every gate uses the
  // uniform amber `gate` colour (UC-09 AC7).
  function drawGateRing(center, rad, isTarget, highlightActive) {
    const [gx, gy, gz] = center;
    const SEG = 48;
    let style = COURSE_COLORS.gate;
    let width = 2;
    if (highlightActive) {
      style = isTarget ? COURSE_COLORS.gateTarget : "rgba(255,202,40,0.35)";
      width = isTarget ? 3 : 1.5;
    }
    ctx.strokeStyle = style;
    ctx.lineWidth = width;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i <= SEG; i++) {
      const a = (i / SEG) * Math.PI * 2;
      const s = project([gx, gy + rad * Math.cos(a), gz + rad * Math.sin(a)]);
      if (!s) { started = false; continue; }
      if (!started) { ctx.moveTo(s.x, s.y); started = true; } else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
  }

  function drawMarkers() {
    const c = scene.course;
    if (!c) return; // graceful degradation: no course geometry → no markers
    // start (green)
    if (c.start) dot(c.start, COURSE_COLORS.start, 6);
    // gates: one yz-plane ring per gate at gate.center, radius = aperture (UC-09 AC7). The
    // current target gate (from frames.target_gate at the playhead) is drawn brighter/thicker
    // in a distinct colour; the rest are dimmed. Legacy files with no target_gate track →
    // every gate uniform (targetIdx === null → no highlight).
    const gates = scene.gates || [];
    const targetIdx = targetGateAtPlayhead();
    for (let gi = 0; gi < gates.length; gi++) {
      const g = gates[gi];
      if (!g || !g.center) continue;
      const isTarget = targetIdx !== null && gi === targetIdx;
      drawGateRing(g.center, g.aperture || 0.5, isTarget, targetIdx !== null);
    }
    // finish: low-alpha wireframe rectangle in the x = finish_x plane
    if (c.finish && c.finish.x != null) {
      const fx = c.finish.x, bb = scene.bb;
      const corners = [
        [fx, bb.min[1], scene.floorZ],
        [fx, bb.max[1], scene.floorZ],
        [fx, bb.max[1], scene.ceilZ],
        [fx, bb.min[1], scene.ceilZ],
      ];
      // translucent fill + wireframe edges
      const proj = corners.map(project);
      if (proj.every(Boolean)) {
        ctx.fillStyle = "rgba(255,82,82,0.10)";
        ctx.beginPath();
        ctx.moveTo(proj[0].x, proj[0].y);
        for (let i = 1; i < proj.length; i++) ctx.lineTo(proj[i].x, proj[i].y);
        ctx.closePath();
        ctx.fill();
      }
      for (let i = 0; i < corners.length; i++) {
        line(corners[i], corners[(i + 1) % corners.length], "rgba(255,82,82,0.7)", 1.5);
      }
    }
  }

  // Stroke a projected 3D polyline in one colour (breaks the path where a vertex is behind
  // the camera). Used for the obstacle wireframe ellipses.
  function strokeProjected(points, style, width) {
    ctx.strokeStyle = style;
    ctx.lineWidth = width || 1;
    ctx.beginPath();
    let started = false;
    for (const p of points) {
      const s = project(p);
      if (!s) { started = false; continue; }
      if (!started) { ctx.moveTo(s.x, s.y); started = true; } else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
  }

  // UC-15: draw each obstacle as a floor-anchored wireframe cylinder — a bottom ellipse at
  // floor_z, a top ellipse at floor_z + height, and a few vertical edges between them. Guarded
  // by `scene.obstacles || []` so a recording without the field draws nothing (graceful
  // degradation, matching the finish/gate markers).
  function drawObstacles() {
    const obstacles = scene.obstacles || [];
    if (!obstacles.length) return;
    const fz = scene.floorZ;
    const SEG = 32;
    const EDGES = 8;
    for (const o of obstacles) {
      if (!o || !o.center) continue;
      const cx = o.center[0], cy = o.center[1];
      const r = o.radius || 0;
      const top = fz + (o.height || 0);
      // UC-21 visibility fix: draw a floor-anchored base ring + low-alpha filled disc so the
      // pillar's footprint reads against the floor grid (the bare wireframe was near-invisible),
      // and bump the wireframe stroke to 2px. Shares the DRY floorDisc helper with drawPads().
      floorDisc(cx, cy, r, fz, COURSE_COLORS.obstacle, fillAlpha(COURSE_COLORS.obstacle, 0.15));
      const bottom = [], upper = [];
      for (let i = 0; i <= SEG; i++) {
        const a = (i / SEG) * Math.PI * 2;
        const px = cx + r * Math.cos(a), py = cy + r * Math.sin(a);
        bottom.push([px, py, fz]);
        upper.push([px, py, top]);
      }
      strokeProjected(bottom, COURSE_COLORS.obstacle, 2);
      strokeProjected(upper, COURSE_COLORS.obstacle, 2);
      for (let i = 0; i < EDGES; i++) {
        const a = (i / EDGES) * Math.PI * 2;
        const px = cx + r * Math.cos(a), py = cy + r * Math.sin(a);
        line([px, py, fz], [px, py, top], COURSE_COLORS.obstacle, 2);
      }
    }
  }

  // UC-21: convert a #rrggbb palette colour to an rgba() string at a given alpha — used for the
  // low-alpha filled floor discs (pads + obstacle footprints).
  function fillAlpha(hex, a) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  }

  // UC-21: shared floor-anchored disc marker — a ring outline (2px) plus a low-alpha filled disc
  // at world z. DRY helper used by both drawPads() (pad footprints) and drawObstacles() (the
  // obstacle visibility base ring). Skips the fill when a vertex is behind the camera.
  function floorDisc(cx, cy, r, z, strokeColor, fillColor) {
    const SEG = 48;
    const ring = [];
    for (let i = 0; i <= SEG; i++) {
      const a = (i / SEG) * Math.PI * 2;
      ring.push([cx + r * Math.cos(a), cy + r * Math.sin(a), z]);
    }
    if (fillColor) {
      const proj = ring.map(project);
      if (proj.every(Boolean)) {
        ctx.fillStyle = fillColor;
        ctx.beginPath();
        ctx.moveTo(proj[0].x, proj[0].y);
        for (let i = 1; i < proj.length; i++) ctx.lineTo(proj[i].x, proj[i].y);
        ctx.closePath();
        ctx.fill();
      }
    }
    strokeProjected(ring, strokeColor, 2);
  }

  // UC-21: map a pad's serialised `kind` to its palette colour. Unknown/absent kinds (legacy
  // recordings, or a pad missing the field) degrade to plain grey — no error.
  function padColor(kind) {
    if (kind === "recharge") return COURSE_COLORS.padRecharge;
    if (kind === "repair") return COURSE_COLORS.padRepair;
    return COURSE_COLORS.padPlain;
  }

  // UC-21: draw each landing pad as a floor-anchored disc coloured by kind (recharge green /
  // repair deep-orange / plain grey). Guarded by `scene.pads || []` so a recording without the
  // field draws nothing (graceful degradation, matching obstacles/markers). Every field read is
  // optional-guarded so a legacy pad missing `kind`/`center`/`radius` degrades cleanly.
  function drawPads() {
    const pads = scene.pads || [];
    if (!pads.length) return;
    const fz = scene.floorZ;
    for (const p of pads) {
      if (!p || !p.center) continue;
      const cx = p.center[0], cy = p.center[1];
      const r = p.radius || 0;
      const color = padColor(p.kind);
      floorDisc(cx, cy, r, fz, color, fillAlpha(color, 0.18));
    }
  }

  function drawTrajectory() {
    const path = scene.path;
    if (!path.length) return;
    const cut = Math.min(state.frame, path.length - 1);
    // remaining (dim)
    ctx.strokeStyle = "rgba(108,168,255,0.30)";
    ctx.lineWidth = 1.4;
    strokePolyline(path, cut, path.length - 1);
    // flown (bright)
    ctx.strokeStyle = "rgba(108,168,255,0.95)";
    ctx.lineWidth = 2.2;
    strokePolyline(path, 0, cut);
    // moving drone marker at the shared playhead
    dot(path[cut], COURSE_COLORS.drone, 4.5);
  }

  function strokePolyline(path, i0, i1) {
    ctx.beginPath();
    let started = false;
    for (let i = i0; i <= i1; i++) {
      const s = project(path[i]);
      if (!s) { started = false; continue; }
      if (!started) { ctx.moveTo(s.x, s.y); started = true; } else ctx.lineTo(s.x, s.y);
    }
    ctx.stroke();
  }

  function render() {
    if (!scene) {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // work in CSS pixels; crisp on HiDPI
    ctx.clearRect(0, 0, cssW, cssH);
    drawFloorGrid();
    drawMarkers();
    drawObstacles();
    drawPads();
    drawTrajectory();
  }

  // -- interaction: drag to orbit, wheel to zoom (camera stays orbitable after a preset) --
  function onPointerDown(ev) {
    drag.active = true;
    drag.x = ev.clientX;
    drag.y = ev.clientY;
    if (canvas.setPointerCapture) canvas.setPointerCapture(ev.pointerId);
  }
  function onPointerMove(ev) {
    if (!drag.active) return;
    const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
    drag.x = ev.clientX;
    drag.y = ev.clientY;
    cam.yaw += dx * 0.01;
    cam.pitch = clamp(cam.pitch + dy * 0.01, -FLIGHT_MAX_PITCH, FLIGHT_MAX_PITCH);
    render();
  }
  function onPointerUp(ev) {
    drag.active = false;
    if (canvas.releasePointerCapture && ev.pointerId != null) {
      try { canvas.releasePointerCapture(ev.pointerId); } catch (_e) { /* ignore */ }
    }
  }
  function onWheel(ev) {
    ev.preventDefault();
    const factor = Math.exp(ev.deltaY * 0.001);
    cam.dist = clamp(cam.dist * factor, 0.05, 1e6);
    render();
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver(() => resize()) : null;
  if (ro) ro.observe(canvas);

  resize();

  return {
    setScene(doc) {
      scene = buildFlightScene(doc);
      // Fit distance so the whole scene is comfortably in frame.
      cam.dist = (scene.radius / Math.tan(FLIGHT_FOV / 2)) * 1.6;
      render();
    },
    applyPreset(name) {
      const p = VIEW_PRESETS[name] || VIEW_PRESETS.top;
      cam.yaw = p.yaw;
      cam.pitch = p.pitch;
      render();
    },
    render,
    destroy() {
      if (ro) ro.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      canvas.removeEventListener("wheel", onWheel);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      scene = null;
    },
  };
}

function destroyFlight3D() {
  if (flight) {
    flight.destroy();
    flight = null;
  }
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// ---- transport --------------------------------------------------------------------
el("play-btn").addEventListener("click", () => {
  if (!state.data) return;
  state.playing = !state.playing;
  el("play-btn").textContent = state.playing ? "⏸ Pause" : "▶ Play";
  if (state.playing) {
    // AC11: pressing Play at the end restarts from frame 0 (seek-to-start-then-play),
    // instead of the old no-op that left the playhead pinned at the last frame.
    const nFrames = state.data.frames.activations.length;
    if (state.frame >= nFrames - 1) {
      state.frame = 0;
      renderAll();
    }
    state.lastTs = 0;
    state.acc = 0;
    requestAnimationFrame(tick);
  }
});

el("scrubber").addEventListener("input", (ev) => {
  state.playing = false;
  el("play-btn").textContent = "▶ Play";
  state.frame = parseInt(ev.target.value, 10) || 0;
  renderAll();
});

el("speed-select").addEventListener("change", (ev) => {
  state.speed = parseFloat(ev.target.value) || 1;
});

// Changing the view plane invalidates the per-view cache (screen positions, outline polygon,
// global peak all depend on the plane), so drop it and let drawBrainMap rebuild.
el("map-view-select").addEventListener("change", () => {
  state.mapCache = null;
  drawBrainMap();
});

// Intensity normalization toggle (AC5): per-frame vs fixed global scale. The global peak is
// already cached per view, so switching just re-normalizes the current frame — no rebuild.
el("map-norm-select").addEventListener("change", (ev) => {
  state.mapNorm = ev.target.value === "global" ? "global" : "frame";
  drawBrainMap();
});

// UC-28 modality-tag overlay toggle (AC-7). Positions are unchanged, so no cache rebuild is
// needed — just redraw the brain map (which re-stamps the heatmap and re-draws the overlay).
const modalitySelect = el("modality-tag-select");
if (modalitySelect) {
  modalitySelect.addEventListener("change", () => {
    if (state.data) drawBrainMap();
  });
}

// View presets (AC13): change the 3D camera angle; the camera stays freely orbitable after.
el("view-select").addEventListener("change", (ev) => {
  if (flight) flight.applyPreset(ev.target.value);
});

// UC-59: keep the DPR-aware anatomical canvases crisp when the layout reflows (window resize, the
// top-zones collapsing to one column, the boxes strip wrapping). Both #brain-canvas and the
// .soma-box-canvas elements have a stable CSS box that is INDEPENDENT of their backing store (brain:
// width-driven square via aspect-ratio; boxes: fixed strip height), so re-deriving the backing store
// in the redraw never changes the CSS box → no feedback loop (mirrors the #flight-canvas precedent).
// Redraw only the anatomical panel; inert (guarded on state.data) before a file is loaded.
if (typeof ResizeObserver !== "undefined") {
  const anatomicalRO = new ResizeObserver(() => {
    if (!state.data) return;
    drawBrainMap();
    drawBoxes();
  });
  const brainCanvas = el("brain-canvas");
  const boxesHost = el("somaless-boxes");
  if (brainCanvas) anatomicalRO.observe(brainCanvas);
  if (boxesHost) anatomicalRO.observe(boxesHost);
}

function tick(ts) {
  if (!state.playing || !state.data) return;
  const nFrames = state.data.frames.activations.length;
  const dt = state.data.meta.dt || 0.05; // simulated seconds per frame
  if (state.lastTs === 0) state.lastTs = ts;
  const elapsed = (ts - state.lastTs) / 1000; // real seconds
  state.lastTs = ts;
  state.acc += (elapsed * state.speed) / dt; // frames to advance
  let frameChanged = false;
  if (state.acc >= 1) {
    state.frame += Math.floor(state.acc);
    state.acc -= Math.floor(state.acc);
    if (state.frame >= nFrames - 1) {
      state.frame = nFrames - 1;
      state.playing = false;
      el("play-btn").textContent = "▶ Play";
    }
    frameChanged = true;
  }
  // ⟨C2⟩ With the beat gone, the brain-map splats only change when the frame advances, so
  // redraw the panels only on a frame change — no re-splatting thousands of gaussians 60×/s
  // while idle. (`elapsed` still drives the frame accumulator above.)
  if (frameChanged) renderAll();
  if (state.playing) requestAnimationFrame(tick);
}
