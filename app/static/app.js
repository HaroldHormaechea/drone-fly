/* app.js — vanilla hash-routed shell for the drone-fly desktop app (UC-61, reservoir in UC-63).
 *
 * Left-nav is a keyboard-operable disclosure tree (Train / Settings / About) with inline-SVG icons
 * driving a right-side main view. Routes:
 *   #/train                          run list (app runs + committed / external dirs, read-only)
 *   #/train/new                      new reservoir run (recipe, flags, knobs, save_path)
 *   #/train/<name>                   → redirects to .../status
 *   #/train/<name>/status            live [aug] progress, curves, GPU, log tail, Launch / Stop
 *   #/train/<name>/audits            honesty-gated verify / diagnose audits
 *   #/train/<name>/recordings        recordings picker + embedded viewer + "Record N episodes"
 *   #/train/<name>/config            per-run config editor (app runs only)
 *   #/settings                       local tool settings (micromamba, env, audit seed/eps)
 *   #/about                          static about page
 * Back/forward work because navigation is pure hashchange. Config edits live in local state and
 * persist ONLY via an explicit Save button (UC-61 AC9); an unsaved-changes guard warns on navigation.
 */
(function () {
  "use strict";

  const view = () => document.getElementById("view");
  let pollTimer = null;
  let dirty = false;
  // Persisted disclosure state so a 5s nav rebuild never collapses what the user opened.
  const navExpanded = new Set(["group:train"]);
  // "Continue from this checkpoint" pre-fill, consumed once by the next #/train/new render.
  let pendingPrefill = null;

  // ---- inline SVG icons (no icon font, no CDN) -----------------------------------------
  const ICONS = {
    train:
      '<path d="M12 2c3 2 4.5 5 4.5 8 0 2.5-1.5 4.7-4.5 7-3-2.3-4.5-4.5-4.5-7 0-3 1.5-6 4.5-8Z"/>' +
      '<circle cx="12" cy="9.5" r="1.6"/><path d="M8.5 15 L6 20 M15.5 15 L18 20"/>',
    settings:
      '<circle cx="12" cy="12" r="3.2"/><path d="M12 2v3 M12 19v3 M2 12h3 M19 12h3 ' +
      'M4.9 4.9l2.1 2.1 M17 17l2.1 2.1 M19.1 4.9l-2.1 2.1 M7 17l-2.1 2.1"/>',
    chevron: '<path d="M9 6l6 6-6 6"/>',
    plus: '<path d="M12 5v14 M5 12h14"/>',
    status: '<path d="M3 12h4l2 6 4-16 2 10h6"/>',
    audit: '<path d="M12 3l8 3v6c0 4.5-3.4 8-8 9-4.6-1-8-4.5-8-9V6z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    recordings: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z"/>',
    config: '<path d="M4 6h16 M4 12h16 M4 18h16"/><circle cx="9" cy="6" r="1.6"/>' +
      '<circle cx="15" cy="12" r="1.6"/><circle cx="8" cy="18" r="1.6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><circle cx="12" cy="7.6" r="0.4"/>',
  };
  function icon(name, cls) {
    return (
      '<svg class="ic ' + (cls || "") + '" viewBox="0 0 24 24" width="16" height="16" ' +
      'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" ' +
      'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">' + ICONS[name] + "</svg>"
    );
  }

  const SETTINGS_HELP = {
    project_root: { help: "Root of the drone-fly project the app operates on (gpu_prototype/, training/). gpu_prototype hardcodes /workspace/drone-fly, so training, audits and recordings only work there.", example: "/workspace/drone-fly" },
    micromamba_executable: { help: "micromamba executable (a name on PATH or a full path). Every launch, audit and record runs `micromamba run -n <env> python ...`.", example: "micromamba" },
    conda_env: { help: "micromamba env with torch + the gpu_prototype deps. Probed once with `import torch`.", example: "dronefly" },
    audit_seed: { help: "AUDIT_SEED for verify_proc / diagnose_proc: the same seed draws identical courses, so models are comparable.", example: "1234" },
    audit_eps: { help: "Default episode count for audits.", example: "512" },
    host: { help: "Loopback host the local server binds to.", example: "127.0.0.1" },
    port: { help: "Server port. 0 chooses an ephemeral free port at startup.", example: "0" },
  };

  // ---- helpers -------------------------------------------------------------------------
  class ApiError extends Error {
    constructor(message, status) { super(message); this.status = status; }
  }
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail || detail; } catch (e) {}
      throw new ApiError(detail, res.status);
    }
    return res.status === 204 ? null : res.json();
  }
  const getJSON = (p) => api(p);
  const postJSON = (p, body) =>
    api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

  function toast(msg, isError) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = "toast show" + (isError ? " error" : "");
    setTimeout(() => { t.className = "toast" + (isError ? " error" : ""); }, 3600);
  }
  function fmt(v, digits) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") {
      if (!Number.isFinite(v)) return "—";
      if (digits !== undefined) return v.toFixed(digits);
      return Number.isInteger(v) ? v.toLocaleString("en-US") : String(v);
    }
    return String(v);
  }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function h(tag, attrs, html) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k === "onclick") n.addEventListener("click", attrs[k]);
      else n.setAttribute(k, attrs[k]);
    }
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  function clearPoll() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }
  function setDirty(v) { dirty = v; }
  function guardNav() {
    if (dirty) return confirm("You have unsaved changes. Leave without saving?");
    return true;
  }
  const enc = encodeURIComponent;

  let schemaCache = null;
  async function getSchema() { if (!schemaCache) schemaCache = await getJSON("/api/schema"); return schemaCache; }

  // ---- navigation tree -----------------------------------------------------------------
  function toggleNode(key, btn, childrenEl) {
    const open = !navExpanded.has(key);
    if (open) navExpanded.add(key); else navExpanded.delete(key);
    btn.setAttribute("aria-expanded", String(open));
    childrenEl.hidden = !open;
  }

  function disclosure(key, iconName, label, childrenEl, opts) {
    opts = opts || {};
    const wrap = h("div", { class: "nav-node" });
    const btn = h("button", { type: "button", class: "nav-disclosure" + (opts.run ? " run" : "") });
    btn.setAttribute("aria-expanded", String(navExpanded.has(key)));
    childrenEl.hidden = !navExpanded.has(key);
    btn.innerHTML =
      icon(iconName) + '<span class="nav-label">' + esc(label) + "</span>" + (opts.pillHTML || "") +
      icon("chevron", "chev");
    btn.addEventListener("click", () => toggleNode(key, btn, childrenEl));
    wrap.appendChild(btn);
    wrap.appendChild(childrenEl);
    return wrap;
  }

  function leaf(href, iconName, label) {
    const a = h("a", { class: "nav-leaf", "data-route": href, href: href },
      icon(iconName) + '<span class="nav-label">' + esc(label) + "</span>");
    if (location.hash === href) { a.classList.add("active"); a.setAttribute("aria-current", "page"); }
    return a;
  }

  function topLink(href, iconName, label) {
    const a = h("a", { class: "nav-leaf nav-top", "data-route": href, href: href },
      icon(iconName) + '<span class="nav-label">' + esc(label) + "</span>");
    if (location.hash === href) { a.classList.add("active"); a.setAttribute("aria-current", "page"); }
    return a;
  }

  function statePill(r) {
    const cls = r.state === "running" ? "pill running" : (r.readonly ? "pill readonly" : "pill");
    return '<span class="' + cls + '">' + esc(r.state) + "</span>";
  }

  async function refreshNav() {
    let runs = [];
    try { runs = (await getJSON("/api/runs")).runs; } catch (e) { return; }

    const hash = location.hash || "#/train";
    if (hash.startsWith("#/train")) navExpanded.add("group:train");
    const rm = hash.match(/^#\/train\/([^/]+)\//);
    if (rm) navExpanded.add("run:" + decodeURIComponent(rm[1]));

    const tree = document.getElementById("nav-tree");
    tree.innerHTML = "";

    const trainKids = h("div", { class: "nav-children" });
    trainKids.appendChild(leaf("#/train/new", "plus", "New run"));
    if (!runs.length) trainKids.appendChild(h("div", { class: "nav-empty" }, "no runs yet"));
    runs.forEach((r) => {
      const base = "#/train/" + enc(r.name);
      const runKids = h("div", { class: "nav-children run-children" });
      runKids.appendChild(leaf(base + "/status", "status", "Status"));
      runKids.appendChild(leaf(base + "/audits", "audit", "Audits"));
      runKids.appendChild(leaf(base + "/recordings", "recordings", "Recordings"));
      if (!r.readonly) runKids.appendChild(leaf(base + "/config", "config", "Config"));
      trainKids.appendChild(disclosure("run:" + r.name, "train", r.name, runKids, { run: true, pillHTML: statePill(r) }));
    });
    tree.appendChild(disclosure("group:train", "train", "Train", trainKids));

    const setGroup = h("div", { class: "nav-node" });
    setGroup.appendChild(topLink("#/settings", "settings", "Settings"));
    setGroup.appendChild(topLink("#/about", "info", "About"));
    tree.appendChild(setGroup);
  }

  // ---- shared run actions --------------------------------------------------------------
  function stateTag(r) {
    return '<span class="state-tag ' + esc(r.state) + '">' + esc(r.state) + "</span>";
  }

  // Launch, asking before train_aug overwrites an existing checkpoint (it restarts from scratch).
  async function launchRun(name) {
    try {
      await postJSON("/api/runs/" + enc(name) + "/launch");
    } catch (e) {
      if (!(e instanceof ApiError) || e.status !== 409 || e.message.indexOf("already exists") === -1) throw e;
      const m = openModal({
        title: "Overwrite checkpoint?",
        bodyHTML: '<p class="mh-help">' + esc(e.message) + '</p><p class="mh-help">train_aug has no ' +
          "resume: relaunching starts from scratch and overwrites the existing checkpoint.</p>",
        actions: [{ label: "Cancel", value: "cancel" }, { label: "Overwrite and launch", value: "go", class: "danger" }],
        invoker: document.activeElement,
      });
      if ((await m.choice) !== "go") return false;
      await postJSON("/api/runs/" + enc(name) + "/launch?overwrite=true");
    }
    toast("Launched " + name);
    return true;
  }

  // "Continue from this checkpoint": pre-fill #/train/new with the generic recipe warm-started from
  // this run's checkpoint and its flags. warm_src_extra_dim stays blank (resolved at runtime).
  function continueFrom(r) {
    pendingPrefill = {
      name: r.name.replace(/^\d{2}-/, "") + "-cont",
      config: { recipe: "train_aug", flags: (r.flags || []).slice(), knobs: { warm_from: r.checkpoint } },
    };
    location.hash = "#/train/new";
  }
  function canContinue(r) {
    if (!r.has_checkpoint || r.running) return false;
    if (r.readonly) return true;
    return r.state === "finished" || r.partial;
  }

  // ---- views ---------------------------------------------------------------------------
  async function renderTrainList() {
    const runs = (await getJSON("/api/runs")).runs;
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("Runs", "New run", "#/train/new"));
    root.appendChild(await preflightBanner());
    if (!runs.length) { root.appendChild(emptyState("No runs yet.", "Create one with “New run”.")); return; }
    const card = h("div", { class: "card" });
    const table = h("table", { class: "list" });
    table.innerHTML = "<thead><tr><th>Name</th><th>State</th><th>Recipe</th><th>Checkpoint</th><th></th></tr></thead>";
    const tbody = h("tbody");
    runs.forEach((r) => {
      const tr = h("tr");
      const kindBadge = r.readonly ? ' <span class="kind-badge">' + esc(r.kind) + " · read-only</span>" : "";
      tr.innerHTML =
        '<td><a class="link" href="#/train/' + enc(r.name) + '/status">' + esc(r.name) + "</a>" + kindBadge + "</td>" +
        "<td>" + stateTag(r) + "</td>" +
        "<td>" + esc(r.recipe || "—") + "</td>" +
        "<td>" + (r.has_checkpoint ? esc(r.checkpoint) + (r.partial ? ' <span class="subtle">(partial)</span>' : "") : "—") + "</td>";
      const actTd = h("td");
      const actions = h("div", { class: "row-actions" });
      actions.appendChild(h("a", { class: "link", href: "#/train/" + enc(r.name) + "/status" }, "open →"));
      if (!r.readonly) {
        const delBtn = h("button", { class: "danger btn-sm", "aria-label": "Delete run " + r.name }, "Delete");
        delBtn.disabled = r.running || r.recording;
        delBtn.addEventListener("click", () => onDeleteRun(r));
        actions.appendChild(delBtn);
      }
      actTd.appendChild(actions);
      tr.appendChild(actTd);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    card.appendChild(table);
    root.appendChild(card);
  }

  async function preflightBanner() {
    const box = h("div");
    try {
      const pf = await getJSON("/api/preflight");
      if (pf.warnings && pf.warnings.length) {
        box.className = "banner warn";
        box.innerHTML = "<strong>Preflight</strong><ul>" +
          pf.warnings.map((w) => "<li>" + esc(w) + "</li>").join("") + "</ul>";
      }
    } catch (e) {}
    return box;
  }

  async function onDeleteRun(r) {
    const m = openModal({
      title: "Delete " + r.name,
      bodyHTML:
        '<p class="mh-help">Permanently delete <code>training/' + esc(r.name) + "/</code> — its run.json, " +
        "logs, recordings and (when it lives there) the checkpoint. This can’t be undone.</p>" +
        '<p class="mh-help">A save_path outside that dir is kept and reported back.</p>',
      actions: [{ label: "Cancel", value: "cancel" }, { label: "Delete", value: "delete", class: "danger" }],
      invoker: document.activeElement,
    });
    if ((await m.choice) !== "delete") return;
    try {
      const resp = await api("/api/runs/" + enc(r.name), { method: "DELETE" });
      toast("Deleted " + r.name + (resp && resp.kept_save_path ? " (kept " + resp.kept_save_path + ")" : ""));
      if (location.hash.indexOf("#/train/" + enc(r.name) + "/") === 0) location.hash = "#/train";
      else { await refreshNav(); await renderTrainList(); }
    } catch (e) { toast(e.message, true); }
  }

  // New run (also the "Continue from this checkpoint" target) and the per-run config editor.
  async function renderRunForm(name) {
    const schema = await getSchema();
    const isNew = name === null;
    const root = view();
    root.innerHTML = "";
    const saveBtn = h("button", { class: "primary" }, "Save");
    const launchBtn = h("button", null, "Save &amp; launch");
    if (isNew) root.appendChild(headBar("New run", null, null, [saveBtn, launchBtn]));
    else root.appendChild(runHead(name, "Config", [saveBtn, launchBtn]));

    let values = {};
    let nameInput = null;
    const prefill = isNew ? pendingPrefill : null;
    pendingPrefill = null;
    if (isNew) {
      const nameCard = h("div", { class: "card" });
      nameCard.appendChild(h("h3", null, "Run name"));
      nameInput = h("input", { type: "text", placeholder: "e.g. exp21-wider-gaps" });
      nameInput.addEventListener("input", () => setDirty(true));
      nameCard.appendChild(nameInput);
      nameCard.appendChild(h("p", { class: "subtle" }, "Outputs go to <code>training/&lt;name&gt;/</code> (run.json, logs/app.log, recordings/)."));
      root.appendChild(nameCard);
      if (prefill) {
        nameInput.value = prefill.name;
        values = prefill.config;
        root.appendChild(h("p", { class: "banner info" }, "Continuing from <code>" + esc(values.knobs.warm_from) +
          "</code> with the generic train_aug recipe. Check the flags match the source model (they change EXTRA_DIM)."));
      }
    } else {
      values = (await getJSON("/api/runs/" + enc(name) + "/config")).config || {};
    }
    const formCard = h("div", { class: "card" });
    formCard.appendChild(h("h3", null, "Reservoir run"));
    const formHost = h("div");
    formCard.appendChild(formHost);
    root.appendChild(formCard);
    const form = Forms.buildRunForm(formHost, schema, values, () => setDirty(true));
    if (prefill) setDirty(true);

    async function save() {
      const n = isNew ? nameInput.value.trim() : name;
      if (!n) { toast("A run name is required.", true); return null; }
      const res = await postJSON("/api/runs/" + enc(n) + "/config", { config: form.collect() });
      setDirty(false);
      toast("Saved training/" + n + "/run.json → " + res.config.save_path);
      await refreshNav();
      return n;
    }
    saveBtn.addEventListener("click", async () => {
      try { const n = await save(); if (n && isNew) location.hash = "#/train/" + enc(n) + "/status"; }
      catch (e) { toast(e.message, true); }
    });
    launchBtn.addEventListener("click", async () => {
      try {
        const n = await save(); if (!n) return;
        if (await launchRun(n)) location.hash = "#/train/" + enc(n) + "/status";
      } catch (e) { toast(e.message, true); }
    });
  }

  function runHead(name, sub, actions) {
    const bar = h("div", { class: "view-head" });
    const wrap = h("div");
    wrap.appendChild(h("h2", null, esc(name)));
    wrap.appendChild(h("p", { class: "view-sub subtle" }, sub));
    bar.appendChild(wrap);
    const slot = h("div", { class: "view-head-actions" });
    (actions || []).forEach((node) => slot.appendChild(node));
    bar.appendChild(slot);
    return bar;
  }

  // Minimal inline-SVG line chart (no chart library). `ys` may contain nulls (skipped).
  function sparkline(title, xs, ys, unit) {
    const box = h("div", { class: "chart" });
    box.appendChild(h("div", { class: "chart-title" }, esc(title)));
    const pts = [];
    xs.forEach((x, i) => { if (ys[i] !== null && ys[i] !== undefined && Number.isFinite(ys[i])) pts.push([x, ys[i]]); });
    if (pts.length < 2) { box.appendChild(h("div", { class: "subtle chart-empty" }, "not enough points yet")); return box; }
    const W = 320, H = 110, P = 6;
    const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
    let y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]));
    if (y1 === y0) { y1 += 1; y0 -= 1; }
    const sx = (x) => P + (W - 2 * P) * (x1 === x0 ? 0 : (x - x0) / (x1 - x0));
    const sy = (y) => H - P - (H - 2 * P) * (y - y0) / (y1 - y0);
    const d = pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
    box.insertAdjacentHTML("beforeend",
      '<svg viewBox="0 0 ' + W + " " + H + '" class="chart-svg" preserveAspectRatio="none" role="img" aria-label="' + esc(title) + '">' +
      '<path d="' + d + '" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>' +
      '<div class="chart-range subtle">' + esc(fmt(y0, 3) + " … " + fmt(y1, 3) + (unit || "")) + "</div>");
    return box;
  }

  async function renderRunStatus(name) {
    const root = view();
    root.innerHTML = "";
    root.classList.add("view-fill");
    root.appendChild(runHead(name, "Status"));

    const monitor = h("div", { class: "card" });
    monitor.appendChild(h("h3", null, "Live progress"));
    const controls = h("div", { class: "btn-row" });
    monitor.appendChild(controls);
    const note = h("p", { class: "subtle" });
    monitor.appendChild(note);
    const progWrap = h("div", { class: "progress" }); progWrap.appendChild(h("span"));
    const progLabel = h("p", { class: "subtle" }, "—");
    monitor.appendChild(progLabel); monitor.appendChild(progWrap);
    const metrics = h("div", { class: "metrics" }); metrics.style.marginTop = "12px";
    monitor.appendChild(metrics);
    const charts = h("div", { class: "charts" });
    monitor.appendChild(charts);
    monitor.appendChild(h("p", { class: "subtle log-head" }, "Process log (training/" + esc(name) + "/logs/app.log)"));
    const logPanel = h("pre", { class: "log-panel" });
    monitor.appendChild(logPanel);
    root.appendChild(monitor);

    let logCursor = 0, statusCursor = 0;
    let points = [];
    let budget = null;
    try {
      const cfg = (await getJSON("/api/runs/" + enc(name) + "/config")).config;
      const recipe = (await getSchema()).recipes.find((r) => r.name === cfg.recipe);
      budget = (cfg.knobs && cfg.knobs.budget) || (recipe ? recipe.default_budget : null);
    } catch (e) { /* committed / external: no run.json */ }

    function renderControls(run) {
      controls.innerHTML = stateTag(run);
      if (run.readonly) {
        controls.insertAdjacentHTML("beforeend", '<span class="subtle">' + esc(run.kind) + " dir — read-only (no launch, record or delete)</span>");
      } else {
        const launch = h("button", { class: "primary" }, "Launch");
        launch.disabled = run.running || run.recording;
        launch.addEventListener("click", async () => { try { await launchRun(name); await tick(); await refreshNav(); } catch (e) { toast(e.message, true); } });
        controls.appendChild(launch);
        const stop = h("button", { class: "danger" }, "Stop");
        stop.disabled = !run.running || run.state === "stopping";
        stop.addEventListener("click", async () => {
          try { await postJSON("/api/runs/" + enc(name) + "/stop"); toast("Stopping " + name); await tick(); await refreshNav(); }
          catch (e) { toast(e.message, true); }
        });
        controls.appendChild(stop);
      }
      if (canContinue(run)) {
        const cont = h("button", null, "Continue from this checkpoint");
        cont.addEventListener("click", () => continueFrom(run));
        controls.appendChild(cont);
      }
      const bits = [];
      if (run.partial_note) bits.push(run.partial_note);
      if (run.recording) bits.push("recording episodes…");
      if (!run.readonly) bits.push("No pause/resume: Stop kills the process group; relaunching restarts from scratch.");
      note.textContent = bits.join(" · ");
    }
    function metricCard(k, v) {
      const m = h("div", { class: "metric" });
      m.innerHTML = '<div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + "</div>";
      return m;
    }
    function renderMetrics(latest, gpu) {
      metrics.innerHTML = "";
      const L = latest || {};
      [
        ["Iteration", fmt(L.it)],
        ["Steps", L.msteps != null ? fmt(L.msteps, 1) + "M" : "—"],
        ["fps", fmt(L.fps, 0)],
        ["ep_rew/step", fmt(L.ep_rew_per_step, 3)],
        ["Tilt", L.tilt_deg != null ? fmt(L.tilt_deg, 1) + "°" : "—"],
        ["Completion EMA", L.compl_pct != null ? fmt(L.compl_pct, 1) + "%" : "—"],
        ["Lap", fmt(L.lap, 1)],
        ["GPU", gpu ? fmt(gpu.util_pct, 0) + "% · " + fmt(gpu.mem_used_mb, 0) + "/" + fmt(gpu.mem_total_mb, 0) + " MB" : "—"],
      ].forEach(([k, v]) => metrics.appendChild(metricCard(k, v)));
    }
    function renderCharts() {
      charts.innerHTML = "";
      const xs = points.map((p) => p.msteps);
      charts.appendChild(sparkline("ep_rew/step vs Msteps", xs, points.map((p) => p.ep_rew_per_step)));
      charts.appendChild(sparkline("completion EMA vs Msteps", xs, points.map((p) => p.compl_pct), "%"));
      charts.appendChild(sparkline("tilt vs Msteps", xs, points.map((p) => p.tilt_deg), "°"));
    }
    async function tick() {
      let snap;
      try { snap = await getJSON("/api/runs/" + enc(name) + "/status?since=" + statusCursor); } catch (e) { return; }
      const run = snap.run;
      renderControls(run);
      if (snap.reset) points = [];
      points = points.concat(snap.points || []);
      statusCursor = snap.next;
      const latest = snap.latest;
      if (snap.saved) {
        progWrap.classList.remove("indeterminate");
        progWrap.firstChild.style.width = "100%";
        progLabel.textContent = "saved " + snap.saved;
      } else if (latest && budget) {
        progWrap.classList.remove("indeterminate");
        progWrap.firstChild.style.width = Math.min(100, 100 * latest.msteps * 1e6 / budget) + "%";
        progLabel.textContent = fmt(latest.msteps, 1) + "M / " + fmt(budget / 1e6, 1) + "M steps · it " + fmt(latest.it);
      } else if (run.running) {
        progWrap.classList.add("indeterminate");
        progLabel.textContent = "starting… (the first [aug] line appears after 20 iterations)";
      } else {
        progWrap.classList.remove("indeterminate");
        progWrap.firstChild.style.width = "0%";
        progLabel.textContent = run.readonly ? "no app log for this dir" : "no progress yet";
      }
      renderMetrics(latest, snap.gpu);
      if ((snap.points && snap.points.length) || snap.reset || !charts.childNodes.length) renderCharts();
      await tickLogs();
    }
    async function tickLogs() {
      try {
        const lr = await getJSON("/api/runs/" + enc(name) + "/logs?since=" + logCursor);
        if (lr.lines && lr.lines.length) {
          const atBottom = logPanel.scrollTop + logPanel.clientHeight >= logPanel.scrollHeight - 4;
          lr.lines.forEach((line) => logPanel.appendChild(document.createTextNode(line + "\n")));
          if (atBottom) logPanel.scrollTop = logPanel.scrollHeight;
        }
        logCursor = lr.next;
      } catch (e) {}
    }
    await tick();
    clearPoll();
    pollTimer = setInterval(tick, 2500);
  }

  // ---- audits (AC6/AC7) ----------------------------------------------------------------
  const VERDICT_LABEL = {
    honest: "Honest",
    suspect: "Suspect",
    dishonest: "Dishonest",
    unverified: "Unverified",
    "nothing-completed": "Nothing completed",
  };

  // Completion is ONLY ever rendered inside this panel, next to the verdict and cheat checks.
  function verdictPanel(a) {
    const failed = a.state === "failed" || a.state === "interrupted";
    const bad = failed || a.verdict === "dishonest" || a.verdict === "unverified";
    const cls = bad ? "bad" : (a.verdict === "suspect" ? "warn" : (a.verdict === "honest" ? "good" : "neutral"));
    const panel = h("div", { class: "verdict-panel " + cls });
    const comp = a.completion ? fmt(a.completion.pct, 0) + "% (" + a.completion.n + "/" + a.completion.eps + ")" : "—";
    let head;
    if (bad) {
      head = '<div class="verdict-banner">Not a success: ' + esc((a.reasons || []).join("; ") || VERDICT_LABEL[a.verdict]) + "</div>";
    } else {
      head = '<div class="verdict-head"><span class="verdict-tag">' + esc(VERDICT_LABEL[a.verdict] || a.verdict) + "</span>" +
        ((a.reasons || []).length ? ' <span class="subtle">' + esc(a.reasons.join("; ")) + "</span>" : "") + "</div>";
    }
    panel.innerHTML = head +
      '<div class="verdict-row"><span class="k">Honest?</span> <strong>' + esc(VERDICT_LABEL[a.verdict] || a.verdict) +
      '</strong><span class="k">Completion</span> <span class="' + (bad ? "struck" : "") + '">' + esc(comp) + "</span></div>" +
      (a.kind === "diagnose" ? '<p class="subtle">Verdict from ' + (a.verdict_source ? "verify audit " + esc(a.verdict_source) : "no verify audit yet") + ".</p>" : "");
    return panel;
  }

  function cheatChecks(a) {
    const v = a.verify;
    if (!v) return h("div");
    const co = v.completed_only;
    const grid = h("div", { class: "metrics" });
    const overallHelp = a.obs_terminal
      ? "PROC_OBS_TERMINAL was on: contact ends the episode, so overall barging equals the collision-death rate — shown, not a cheat signal. The honest check is completed-only barging."
      : "PROC_OBS_TERMINAL was off: episodes may barge through obstacles and continue. Above 2% marks the run suspect.";
    const cells = [
      ["Barging (all episodes)", v.barging_pct != null ? fmt(v.barging_pct, 0) + "%" : "—", overallHelp],
      ["Barging (completed runs)", co ? fmt(co.barged_pct, 0) + "%" : (a.completion && a.completion.n === 0 ? "n/a (none completed)" : "—"), "Among completed episodes: any inside-obstacle step. Above 0% is suspect, above 2% dishonest."],
      ["Avg inside-steps/ep", fmt(v.avg_inside_steps, 2), null],
      ["Min surface clearance", v.min_clearance_m != null ? fmt(v.min_clearance_m, 2) + " m" : "—", null],
      ["Mean speed", v.mean_speed != null ? fmt(v.mean_speed, 2) + " m/s" : "—", "Below 0.5 m/s is a hovering signature (dishonest)."],
      ["Mean tilt", v.mean_tilt_deg != null ? fmt(v.mean_tilt_deg, 0) + "°" : "—", "Below 5° looks like hovering (suspect)."],
      ["Mean steps alive", v.mean_steps_alive != null ? fmt(v.mean_steps_alive, 0) + "/" + fmt(v.max_steps) : "—", null],
      ["Gate passes/ep", fmt(v.gate_passes_mean, 1), null],
    ];
    cells.forEach(([k, val, help]) => {
      const m = h("div", { class: "metric" });
      const kk = h("div", { class: "k" }); kk.appendChild(document.createTextNode(k));
      if (help) kk.appendChild(infoButton(k, help, null, null));
      m.appendChild(kk);
      m.appendChild(h("div", { class: "v" }, esc(val)));
      grid.appendChild(m);
    });
    return grid;
  }

  function diagnoseTables(a) {
    const d = a.diagnose;
    const box = h("div", { class: "diag" });
    if (!d) return box;
    const table = (title, rows) => {
      if (!rows || !rows.length) return "";
      return "<h4>" + esc(title) + '</h4><table class="list compact"><thead><tr><th></th><th>n</th><th>completion</th></tr></thead><tbody>' +
        rows.map((r) => "<tr><td>" + esc(r.label) + "</td><td>" + fmt(r.n) + "</td><td>" +
          (r.pct != null ? fmt(r.pct, 0) + "% (" + r.completed + "/" + r.n + ")" : "—") + "</td></tr>").join("") + "</tbody></table>";
    };
    let html = table("By obstacle count", d.by_obstacle_count) + table("By tightest gate", d.by_tightest_gate) + table("By gate count", d.by_gate_count);
    const f = d.failures;
    if (f) {
      html += "<h4>Failure anatomy (" + fmt(f.count) + " failures)</h4><ul class=\"plain\">" +
        f.causes.map((c) => "<li>" + esc(c.label) + ": " + fmt(c.n) + (c.pct != null ? " (" + fmt(c.pct, 0) + "%)" : "") + "</li>").join("") +
        (f.mean_death_tilt_deg != null ? "<li>mean death tilt: " + fmt(f.mean_death_tilt_deg, 0) + "°</li>" : "") +
        (f.progress_before_death_pct != null ? "<li>progress before death: " + fmt(f.progress_before_death_pct, 0) + "% of required gates</li>" : "") +
        (f.obstacle_deaths_on_multi_pct != null ? "<li>obstacle deaths on 2+ obstacle courses: " + fmt(f.obstacle_deaths_on_multi_pct, 0) + "%</li>" : "") +
        "</ul>";
    }
    box.innerHTML = html;
    return box;
  }

  function auditCard(a) {
    const card = h("div", { class: "card audit-card" });
    card.appendChild(h("h3", null, esc(a.kind) + ' <span class="subtle">· ' + esc(a.id) + " · " + esc(a.state) +
      " · seed " + esc(a.seed) + " · eps " + esc(a.eps) + " · flags " + esc((a.flags || []).join(", ") || "none") + "</span>"));
    card.appendChild(verdictPanel(a));
    if (a.state === "failed" || a.state === "interrupted") {
      card.appendChild(h("p", { class: "subtle" }, "Partial output (not a result):"));
    }
    card.appendChild(cheatChecks(a));
    card.appendChild(diagnoseTables(a));
    if (a.state !== "done") {
      card.appendChild(h("p", { class: "subtle log-head" }, "Log tail"));
      card.appendChild(h("pre", { class: "log-panel" }, esc((a.log_tail || []).join("\n"))));
    }
    return card;
  }

  async function renderRunAudits(name) {
    const schema = await getSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(runHead(name, "Audits"));
    let data = await getJSON("/api/runs/" + enc(name) + "/audits");
    const settings = await getJSON("/api/settings");
    const run = data.run;

    const formCard = h("div", { class: "card" });
    formCard.appendChild(h("h3", null, "Run an audit"));
    formCard.appendChild(h("p", { class: "subtle" },
      "verify_proc checks completion <strong>and</strong> the cheat checks (barging, hovering, clearance); diagnose_proc breaks failures down. " +
      "Use the flags the model was trained with — gates_proc_env reads them at import time. Checkpoint: <code>" + esc(run.checkpoint || "—") + "</code>"));
    const grid = h("div", { class: "form-grid" });
    formCard.appendChild(grid);
    const row = (label, ctrl) => {
      const r = h("div", { class: "field" });
      r.appendChild(h("label", { class: "key" }, esc(label)));
      const c = h("div", { class: "control" }); c.appendChild(ctrl); r.appendChild(c);
      grid.appendChild(r);
    };
    const kind = h("select");
    kind.innerHTML = '<option value="verify">verify_proc (honesty gate)</option><option value="diagnose">diagnose_proc (failure breakdown)</option>';
    row("kind", kind);
    const flagWrap = h("div", { class: "flag-boxes" });
    const flagBoxes = {};
    schema.flags.forEach((f) => {
      const cb = h("input", { type: "checkbox" });
      cb.checked = (run.flags || []).indexOf(f.name) !== -1;
      flagBoxes[f.name] = cb;
      const l = h("label", { class: "enable" }); l.appendChild(cb); l.appendChild(document.createTextNode(f.name));
      flagWrap.appendChild(l);
    });
    row("flags" + (run.readonly ? " (required: no run.json)" : ""), flagWrap);
    const seed = h("input", { type: "number" }); seed.value = settings.audit_seed != null ? settings.audit_seed : 1234;
    row("AUDIT_SEED", seed);
    const eps = h("input", { type: "number" }); eps.value = settings.audit_eps != null ? settings.audit_eps : 512;
    row("episodes", eps);
    const runBtn = h("button", { class: "primary" }, "Run audit");
    const btnRow = h("div", { class: "btn-row" }); btnRow.appendChild(runBtn);
    const reason = h("span", { class: "subtle" });
    btnRow.appendChild(reason);
    formCard.appendChild(btnRow);
    root.appendChild(formCard);

    const list = h("div");
    root.appendChild(list);

    function paint() {
      const r = data.run;
      runBtn.disabled = !r.audit_eligible;
      [kind, seed, eps].concat(Object.values(flagBoxes)).forEach((n) => { n.disabled = !r.audit_eligible; });
      reason.textContent = r.audit_eligible ? "" : "Disabled: " + (r.audit_reason || "not eligible");
      list.innerHTML = "";
      if (!data.audits.length) list.appendChild(emptyState("No audits yet.", "Run verify_proc first — completion only counts alongside its cheat checks."));
      data.audits.forEach((a) => list.appendChild(auditCard(a)));
    }
    async function refresh() {
      try { data = await getJSON("/api/runs/" + enc(name) + "/audits"); paint(); } catch (e) {}
    }
    runBtn.addEventListener("click", async () => {
      try {
        await postJSON("/api/runs/" + enc(name) + "/audits", {
          kind: kind.value,
          flags: Object.keys(flagBoxes).filter((f) => flagBoxes[f].checked),
          seed: parseInt(seed.value, 10),
          eps: parseInt(eps.value, 10),
        });
        toast(kind.value + " audit started");
        await refresh();
      } catch (e) { toast(e.message, true); }
    });
    paint();
    clearPoll();
    pollTimer = setInterval(() => { if (data.audits.some((a) => a.state === "running") || data.run.running) refresh(); }, 3000);
  }

  async function renderRunRecordings(name) {
    const root = view();
    root.innerHTML = "";
    const run = await getJSON("/api/runs/" + enc(name));
    const actions = [];
    if (!run.readonly) {
      const n = h("input", { type: "number", min: "1", class: "num-sm", "aria-label": "Episodes to record" });
      n.value = 10;
      const rec = h("button", null, "Record episodes");
      rec.disabled = !run.audit_eligible || run.recording;
      rec.title = run.recording ? "already recording" : (run.audit_eligible ? "record_aug.py proc <save_path> " + name + " <n>" : (run.audit_reason || ""));
      rec.addEventListener("click", async () => {
        try { await postJSON("/api/runs/" + enc(name) + "/record", { episodes: parseInt(n.value, 10) || 10 }); toast("Recording " + n.value + " episodes of " + name); rec.disabled = true; }
        catch (e) { toast(e.message, true); }
      });
      actions.push(n, rec);
    }
    root.appendChild(runHead(name, "Recordings", actions));
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Episode playback"));
    const pickerRow = h("div", { class: "picker-row" });
    card.appendChild(pickerRow);
    const viewerHost = h("div", { class: "viewer-host" });
    card.appendChild(viewerHost);
    root.appendChild(card);

    let recs = [];
    try { recs = (await getJSON("/api/runs/" + enc(name) + "/recordings")).recordings; }
    catch (e) { pickerRow.appendChild(h("p", { class: "subtle" }, "—")); return; }

    if (!recs.length) {
      pickerRow.appendChild(h("p", { class: "subtle" }, run.readonly
        ? "No recordings in this dir."
        : "No recordings yet. Use “Record episodes” once the run has a checkpoint (record_aug.py writes them to training/" + esc(name) + "/recordings/)."));
      return;
    }
    const label = h("label", { class: "picker-label", for: "rec-picker" }, "Episode");
    const select = h("select", { class: "picker", id: "rec-picker" });
    recs.forEach((r) => select.appendChild(h("option", { value: String(r.episode) }, "episode " + r.episode + (r.gzip ? " (gz)" : ""))));
    pickerRow.appendChild(label);
    pickerRow.appendChild(select);
    select.addEventListener("change", () => ViewerEmbed.embedRecording(viewerHost, name, select.value));
    ViewerEmbed.embedRecording(viewerHost, name, recs[0].episode);
  }

  async function renderSettings() {
    const s = await getJSON("/api/settings");
    const root = view();
    root.innerHTML = "";
    const saveBtn = h("button", { class: "primary" }, "Save");
    root.appendChild(headBar("Settings", null, null, [saveBtn]));
    const card = h("div", { class: "card" });
    const rows = [
      ["project_root", "Project root", s.project_root || "", "text"],
      ["micromamba_executable", "micromamba executable", s.micromamba_executable || "micromamba", "text"],
      ["conda_env", "micromamba env", s.conda_env || "dronefly", "text"],
      ["audit_seed", "Audit seed (AUDIT_SEED)", s.audit_seed != null ? s.audit_seed : 1234, "number"],
      ["audit_eps", "Audit episodes", s.audit_eps != null ? s.audit_eps : 512, "number"],
      ["host", "Server host", s.host || "127.0.0.1", "text"],
      ["port", "Server port (0 = ephemeral)", s.port != null ? s.port : 0, "number"],
    ];
    const inputs = {};
    rows.forEach(([k, label, val, type]) => {
      const row = h("div", { class: "field" });
      const keyLabel = h("label", { class: "key" }, esc(label));
      const help = SETTINGS_HELP[k];
      if (help) keyLabel.appendChild(infoButton(label, help.help, help.example, null));
      row.appendChild(keyLabel);
      const inp = h("input", { type: type });
      inp.value = val;
      inp.addEventListener("input", () => setDirty(true));
      const ctrl = h("div", { class: "control" }); ctrl.appendChild(inp);
      row.appendChild(ctrl);
      card.appendChild(row);
      inputs[k] = inp;
    });
    root.appendChild(card);
    root.appendChild(await preflightBanner());
    saveBtn.addEventListener("click", async () => {
      try {
        await postJSON("/api/settings", {
          project_root: inputs.project_root.value || null,
          micromamba_executable: inputs.micromamba_executable.value || "micromamba",
          conda_env: inputs.conda_env.value || "dronefly",
          audit_seed: parseInt(inputs.audit_seed.value, 10) || 0,
          audit_eps: parseInt(inputs.audit_eps.value, 10) || 512,
          host: inputs.host.value || "127.0.0.1",
          port: parseInt(inputs.port.value, 10) || 0,
        });
        setDirty(false); toast("Settings saved.");
      } catch (e) { toast(e.message, true); }
    });
  }

  async function renderAbout() {
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("About", null, null));
    const card = h("div", { class: "card about-card" });
    card.innerHTML =
      '<h3>drone<span class="dot">·</span>fly desktop</h3>' +
      "<p>A lightweight, local, single-user desktop app over the frozen-K1 reservoir pipeline " +
      "(gpu_prototype/): launch reservoir runs, monitor live [aug] progress and logs, run " +
      "honesty-gated audits (a completion % that barges or hovers is flagged, not celebrated), " +
      "record episodes, and replay them in the embedded brain + actions + course viewer.</p>" +
      '<dl class="about-meta">' +
      "<dt>Version</dt><dd>0.2.0</dd>" +
      "<dt>Built by</dt><dd>Harold Hormaechea</dd>" +
      "</dl>";
    root.appendChild(card);
  }

  // ---- shared UI bits ------------------------------------------------------------------
  function infoButton(title, helpText, example, constraintsHTML) {
    const btn = h("button", { type: "button", class: "info-btn", "aria-label": "About " + title }, icon("info"));
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      let body = "";
      if (helpText) body += '<p class="mh-help">' + esc(helpText) + "</p>";
      if (example) body += '<p class="mh-example"><span class="mh-k">Example</span> <code>' + esc(example) + "</code></p>";
      if (constraintsHTML) body += constraintsHTML;
      if (!body) body = '<p class="subtle">No additional details.</p>';
      openModal({ title: title, bodyHTML: body, invoker: btn });
    });
    return btn;
  }
  // Exposed so forms.js can build identical info buttons for form fields.
  window.AppUI = { infoButton, icon, esc };

  function emptyState(title, sub) {
    const box = h("div", { class: "empty-state" });
    box.appendChild(h("p", { class: "empty-title" }, title));
    if (sub) box.appendChild(h("p", { class: "subtle" }, sub));
    return box;
  }

  function headBar(title, actionLabel, actionHash, actions) {
    const bar = h("div", { class: "view-head" });
    bar.appendChild(h("h2", null, esc(title)));
    const slot = h("div", { class: "view-head-actions" });
    if (actionLabel && actionHash) slot.appendChild(h("a", { class: "btn primary", href: actionHash }, actionLabel));
    (actions || []).forEach((node) => slot.appendChild(node));
    bar.appendChild(slot);
    return bar;
  }

  // ---- router --------------------------------------------------------------------------
  function setActiveNav(hash) {
    const tree = document.getElementById("nav-tree");
    if (!tree) return;
    tree.querySelectorAll("[data-route]").forEach((a) => {
      const on = a.getAttribute("data-route") === hash;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  function skeletonKind(parts) {
    if (parts[0] === "train" && parts[1] && parts[1] !== "new") {
      if (parts[2] === "recordings") return "recordings";
      if (parts[2] === "config" || parts[2] === "audits") return "form";
      return "status";
    }
    if (parts[0] === "train" && parts[1] === "new") return "form";
    if (parts[0] === "settings") return "settings";
    if (parts[0] === "about") return "about";
    return "list";
  }
  function paintSkeleton(kind) {
    const block = (height) => '<div class="skel-block" style="height:' + height + '"></div>';
    const line = (width) => '<div class="skel-line" style="width:' + width + '"></div>';
    let body;
    if (kind === "list") body = block("40px") + block("40px") + block("40px");
    else if (kind === "status") body = block("18px") + block("150px") + block("240px");
    else if (kind === "recordings") body = block("42px") + block("360px");
    else if (kind === "settings") body = block("40px") + block("40px") + block("40px") + block("40px");
    else if (kind === "about") body = block("60px") + line("70%") + line("55%");
    else body = block("120px") + block("240px");
    view().innerHTML =
      '<div class="skeleton"><div class="skel-head">' + line("160px") + "</div>" + body + "</div>";
  }

  let currentHash = null;
  async function route() {
    const hash = location.hash || "#/train";
    if (currentHash !== null && hash !== currentHash && !guardNav()) {
      location.hash = currentHash; // revert
      return;
    }
    currentHash = hash;
    setDirty(false);
    clearPoll();
    view().classList.remove("view-fill");
    const parts = hash.replace(/^#\//, "").split("/");
    setActiveNav(hash);
    const skelT = setTimeout(() => paintSkeleton(skeletonKind(parts)), 120);
    try {
      if (parts[0] === "train" && parts[1] === "new") await renderRunForm(null);
      else if (parts[0] === "train" && parts[1]) {
        const name = decodeURIComponent(parts[1]);
        const leafName = parts[2];
        if (leafName === "recordings") await renderRunRecordings(name);
        else if (leafName === "config") await renderRunForm(name);
        else if (leafName === "audits") await renderRunAudits(name);
        else if (leafName === "status") await renderRunStatus(name);
        else { location.hash = "#/train/" + enc(name) + "/status"; return; }
      }
      else if (parts[0] === "settings") await renderSettings();
      else if (parts[0] === "about") await renderAbout();
      else await renderTrainList();
    } catch (e) {
      view().innerHTML = '<p class="subtle">Error: ' + esc(e && e.message ? e.message : e) + "</p>";
    } finally {
      clearTimeout(skelT);
    }
    await refreshNav();
  }

  window.addEventListener("hashchange", route);
  window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
  window.addEventListener("DOMContentLoaded", async () => {
    if (!location.hash) location.hash = "#/train";
    await refreshNav();
    await route();
    setInterval(refreshNav, 5000);
  });
})();
