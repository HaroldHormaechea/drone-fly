"use strict";
/* drone-fly activation playback viewer — vanilla JS, no build step.
 * Loads a recorded episode file (plain JSON or gzip) picked from disk (FileReader /
 * DecompressionStream, so it works from file:// with no server) and renders three panels
 * synced on one timeline: an anatomical top-down brain map (with a per-neuron activation
 * "beat"), a neurons×time heatmap, and a flight panel (4 action traces + an orbitable 3D
 * flight scene). See use-cases/05 & 06 and the recorder schema. */

const ROLE_COLORS = {
  sensory: [79, 195, 247],
  interneuron: [176, 182, 208],
  motor: [255, 112, 67],
};
const FALLBACK_COLOR = [107, 112, 137];
const ROLE_RANK = { sensory: 0, interneuron: 1, motor: 2 };
const PROJECTIONS = { xz: [0, 2], xy: [0, 1], yz: [1, 2] };
// Anatomical brain-map view presets → projection plane. Only top/xz is anatomically pinned
// (dorsal/top-down, per record/coordinates.py); front↔side is a reversible labeling convention.
const MAP_VIEW_PRESETS = { front: "xy", side: "yz", top: "xz" };
const ACTION_COLORS = ["#ffd54f", "#4fc3f7", "#81c784", "#ff8a65"];
const SUPPORTED_SCHEMA = 1;

// ---- neuron "beat" (AC6) --------------------------------------------------------------
// Brain-map neurons render small at rest and pulse when active: radius = 2px at rest, ~9px
// at full activation (r_inst = 2 + 7·act/255). During continuous playback a per-neuron
// envelope adds a fast attack + exponential release toward rest (real-time, ~0.2s), floored
// at the instantaneous radius. When paused/scrubbing we draw the instantaneous radius for
// the current frame (no stale decay) so a frame's size always reflects that frame.
const BEAT_REST = 2; // px radius at rest
const BEAT_PEAK = 9; // px radius at full activation
const BEAT_SPAN = BEAT_PEAK - BEAT_REST; // 7
const BEAT_RELEASE_TAU = 0.06; // s; exponential release ~= back to rest over ~0.2s

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
  rowOrder: null, // neuron indices ordered sensory->inter->motor (heatmap rows)
  heatmap: null, // offscreen canvas (n_frames x n_neurons)
  beat: null, // per-neuron pulse envelope radii (AC6); Float32Array or null
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
  state.rowOrder = computeRowOrder(doc.meta.roles);
  state.heatmap = buildHeatmap(doc, state.rowOrder);
  state.beat = new Float32Array(nNeurons).fill(BEAT_REST);

  renderMetaBar(doc, name);
  renderPositionSource(doc.meta.positions);
  renderFlightLegend(doc.meta.action_layout);
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

// ---- heatmap precompute ---------------------------------------------------------------
function computeRowOrder(roles) {
  const idx = roles.map((_, i) => i);
  idx.sort((a, b) => {
    const ra = ROLE_RANK[roles[a]] ?? 1;
    const rb = ROLE_RANK[roles[b]] ?? 1;
    return ra - rb || a - b;
  });
  return idx;
}

function buildHeatmap(doc, rowOrder) {
  const frames = doc.frames.activations;
  const nFrames = frames.length;
  const nNeurons = doc.meta.n_neurons;
  const off = document.createElement("canvas");
  off.width = Math.max(1, nFrames);
  off.height = Math.max(1, nNeurons);
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(off.width, off.height);
  const roles = doc.meta.roles;
  for (let row = 0; row < nNeurons; row++) {
    const n = rowOrder[row];
    const color = ROLE_COLORS[roles[n]] || FALLBACK_COLOR;
    for (let f = 0; f < nFrames; f++) {
      const b = frames[f][n] / 255; // uint8 -> [0,1]
      const p = (row * off.width + f) * 4;
      img.data[p] = color[0] * b;
      img.data[p + 1] = color[1] * b;
      img.data[p + 2] = color[2] * b;
      img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

// ---- rendering ------------------------------------------------------------------------
function renderAll() {
  if (!state.data) return;
  el("scrubber").value = state.frame;
  const n = state.data.frames.activations.length;
  el("frame-label").textContent = `frame ${state.frame} / ${Math.max(0, n - 1)}`;
  drawBrainMap();
  drawHeatmap();
  drawActions();
  if (flight) flight.render();
}

function projectedPoints() {
  const pos = state.data.meta.positions;
  const plane = MAP_VIEW_PRESETS[el("map-view-select").value] || "xz";
  const [a0, a1] = PROJECTIONS[plane] || PROJECTIONS.xz;
  const isDefault = plane === (pos.projection || "xz");
  const pts = new Array(pos.has_position.length);
  for (let i = 0; i < pts.length; i++) {
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

function drawBrainMap() {
  const canvas = el("brain-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 18;
  ctx.clearRect(0, 0, W, H);
  const doc = state.data;
  const pts = projectedPoints();
  const b = bounds(pts);
  const sx = (W - 2 * pad) / (b.maxX - b.minX || 1);
  const sy = (H - 2 * pad) / (b.maxY - b.minY || 1);
  const act = doc.frames.activations[state.frame];
  const roles = doc.meta.roles;
  const hasPos = doc.meta.positions.has_position;
  // Beat (AC6): during continuous playback use the per-neuron envelope (fast attack +
  // real-time release); when paused/scrubbing use the instantaneous radius for this frame.
  const usingEnvelope = state.playing && state.beat;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!p) continue;
    const x = pad + (p[0] - b.minX) * sx;
    const y = H - pad - (p[1] - b.minY) * sy; // flip Y so +axis points up
    const brightness = act[i] / 255;
    const rInst = BEAT_REST + BEAT_SPAN * brightness;
    const radius = usingEnvelope ? state.beat[i] : rInst;
    const color = hasPos[i] ? (ROLE_COLORS[roles[i]] || FALLBACK_COLOR) : FALLBACK_COLOR;
    const alpha = 0.18 + 0.82 * brightness;
    ctx.beginPath();
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha.toFixed(3)})`;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

// Evolve the per-neuron beat envelope by `dtReal` seconds of real (wall-clock) time. Fast
// attack + exponential release toward rest, floored at each neuron's instantaneous radius so
// the current frame's activation is never under-drawn (AC6). Called once per rAF while playing.
function updateBeat(dtReal) {
  if (!state.beat || !state.data) return;
  const act = state.data.frames.activations[state.frame];
  const decay = Math.exp(-Math.max(0, dtReal) / BEAT_RELEASE_TAU);
  for (let i = 0; i < state.beat.length; i++) {
    const rInst = BEAT_REST + BEAT_SPAN * (act[i] / 255);
    const released = BEAT_REST + (state.beat[i] - BEAT_REST) * decay;
    state.beat[i] = Math.max(released, rInst);
  }
}

// Seed the envelope to the current frame's instantaneous radii (called when playback starts)
// so the first animated frames don't inherit a stale pulse from a previous run.
function seedBeat() {
  if (!state.beat || !state.data) return;
  const act = state.data.frames.activations[state.frame];
  for (let i = 0; i < state.beat.length; i++) {
    state.beat[i] = BEAT_REST + BEAT_SPAN * (act[i] / 255);
  }
}

function drawHeatmap() {
  const canvas = el("heatmap-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(state.heatmap, 0, 0, W, H);
  // playhead
  const nFrames = state.data.frames.activations.length;
  const x = nFrames > 1 ? (state.frame / (nFrames - 1)) * (W - 1) : 0;
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x + 0.5, 0);
  ctx.lineTo(x + 0.5, H);
  ctx.stroke();
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

  // Anchor points that must always be inside the display extent — every gate centre plus
  // start and finish, so the floor/finish plane always contain the whole course.
  const anchors = path.slice();
  if (course) {
    if (course.start) anchors.push(course.start);
    for (const g of gates) {
      if (g && g.center) anchors.push(g.center);
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

  return { path, course, gates, bb, center, radius, floorZ, ceilZ };
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
    seedBeat();
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

el("map-view-select").addEventListener("change", () => drawBrainMap());

// View presets (AC13): change the 3D camera angle; the camera stays freely orbitable after.
el("view-select").addEventListener("change", (ev) => {
  if (flight) flight.applyPreset(ev.target.value);
});

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
  // Evolve the beat envelope in REAL time every rAF (decoupled from the frame accumulator),
  // then redraw: a full sync when the frame advanced, else just the animating brain map.
  updateBeat(elapsed);
  if (frameChanged) renderAll();
  else drawBrainMap();
  if (state.playing) requestAnimationFrame(tick);
}
