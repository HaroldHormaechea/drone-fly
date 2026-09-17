"use strict";
/* drone-fly activation playback viewer — vanilla JS, no build step.
 * Loads a recorded episode file (plain JSON or gzip) picked from disk (FileReader /
 * DecompressionStream, so it works from file:// with no server) and renders three panels
 * synced on one timeline: an anatomical top-down brain map, a neurons×time heatmap, and a
 * flight panel (4 action traces + drone path). See use-cases/05 and the recorder schema. */

const ROLE_COLORS = {
  sensory: [79, 195, 247],
  interneuron: [176, 182, 208],
  motor: [255, 112, 67],
};
const FALLBACK_COLOR = [107, 112, 137];
const ROLE_RANK = { sensory: 0, interneuron: 1, motor: 2 };
const PROJECTIONS = { xz: [0, 2], xy: [0, 1], yz: [1, 2] };
const ACTION_COLORS = ["#ffd54f", "#4fc3f7", "#81c784", "#ff8a65"];
const SUPPORTED_SCHEMA = 1;

const el = (id) => document.getElementById(id);
const state = {
  data: null,
  frame: 0,
  playing: false,
  speed: 1,
  rowOrder: null, // neuron indices ordered sensory->inter->motor (heatmap rows)
  heatmap: null, // offscreen canvas (n_frames x n_neurons)
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

  renderMetaBar(doc, name);
  renderPositionSource(doc.meta.positions);
  renderFlightLegend(doc.meta.action_layout);
  renderOutcome(doc.outcome);

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
  drawPath();
}

function projectedPoints() {
  const pos = state.data.meta.positions;
  const [a0, a1] = PROJECTIONS[el("axis-select").value] || PROJECTIONS.xz;
  const isDefault = el("axis-select").value === (pos.projection || "xz");
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
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!p) continue;
    const x = pad + (p[0] - b.minX) * sx;
    const y = H - pad - (p[1] - b.minY) * sy; // flip Y so +axis points up
    const brightness = act[i] / 255;
    const color = hasPos[i] ? (ROLE_COLORS[roles[i]] || FALLBACK_COLOR) : FALLBACK_COLOR;
    const alpha = 0.18 + 0.82 * brightness;
    ctx.beginPath();
    ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},${alpha.toFixed(3)})`;
    ctx.arc(x, y, 2.6 + 2.4 * brightness, 0, Math.PI * 2);
    ctx.fill();
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

function drawPath() {
  const canvas = el("path-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, pad = 14;
  ctx.clearRect(0, 0, W, H);
  const pos = state.data.frames.drone_position;
  if (!pos.length) return;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pos) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  }
  const sx = (W - 2 * pad) / (maxX - minX || 1);
  const sy = (H - 2 * pad) / (maxY - minY || 1);
  const px = (p) => pad + (p[0] - minX) * sx;
  const py = (p) => H - pad - (p[1] - minY) * sy;
  // faint full path
  ctx.strokeStyle = "rgba(108,168,255,0.25)"; ctx.lineWidth = 1;
  ctx.beginPath();
  pos.forEach((p, i) => (i ? ctx.lineTo(px(p), py(p)) : ctx.moveTo(px(p), py(p))));
  ctx.stroke();
  // travelled portion
  ctx.strokeStyle = "rgba(108,168,255,0.95)"; ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i <= state.frame && i < pos.length; i++) {
    const p = pos[i];
    if (i === 0) ctx.moveTo(px(p), py(p)); else ctx.lineTo(px(p), py(p));
  }
  ctx.stroke();
  // current marker
  const cur = pos[Math.min(state.frame, pos.length - 1)];
  ctx.fillStyle = "#ffffff";
  ctx.beginPath(); ctx.arc(px(cur), py(cur), 4, 0, Math.PI * 2); ctx.fill();
}

// ---- transport --------------------------------------------------------------------
el("play-btn").addEventListener("click", () => {
  if (!state.data) return;
  state.playing = !state.playing;
  el("play-btn").textContent = state.playing ? "⏸ Pause" : "▶ Play";
  if (state.playing) {
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

el("axis-select").addEventListener("change", () => drawBrainMap());

function tick(ts) {
  if (!state.playing || !state.data) return;
  const nFrames = state.data.frames.activations.length;
  const dt = state.data.meta.dt || 0.05; // simulated seconds per frame
  if (state.lastTs === 0) state.lastTs = ts;
  const elapsed = (ts - state.lastTs) / 1000; // real seconds
  state.lastTs = ts;
  state.acc += (elapsed * state.speed) / dt; // frames to advance
  if (state.acc >= 1) {
    state.frame += Math.floor(state.acc);
    state.acc -= Math.floor(state.acc);
    if (state.frame >= nFrames - 1) {
      state.frame = nFrames - 1;
      state.playing = false;
      el("play-btn").textContent = "▶ Play";
    }
    renderAll();
  }
  if (state.playing) requestAnimationFrame(tick);
}
