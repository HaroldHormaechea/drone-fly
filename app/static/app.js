/* app.js — vanilla hash-routed shell for the drone-fly desktop app (UC-61 + follow-up items 1-7).
 *
 * Left-nav is a keyboard-operable disclosure tree (Slices / Train / Settings) with inline-SVG icons
 * driving a right-side main view. Routes:
 *   #/slices/new                     define/generate a slice
 *   #/slices/<name>                  edit + regenerate a saved slice
 *   #/train                          run list
 *   #/train/new                      new training config
 *   #/train/<name>                   → redirects to .../status
 *   #/train/<name>/status            live status + lifecycle controls (owns the poll timer)
 *   #/train/<name>/recordings        recordings picker + embedded viewer
 *   #/train/<name>/config            per-run config editor
 *   #/settings                       local tool settings (incl. the training interpreter override)
 * Back/forward work because navigation is pure hashchange. Config edits live in local state and
 * persist ONLY via an explicit Save button (AC9); an unsaved-changes guard warns on navigation.
 */
(function () {
  "use strict";

  const view = () => document.getElementById("view");
  let pollTimer = null;
  let dirty = false;
  // item 1: persisted disclosure state so a 5s nav rebuild never collapses what the user opened.
  // Keys: "group:slices", "group:train", "run:<name>".
  const navExpanded = new Set(["group:train"]);

  // ---- inline SVG icons (item 7: no icon font, no CDN) ---------------------------------
  const ICONS = {
    slices:
      '<path d="M12 2 L21 7 L12 12 L3 7 Z"/><path d="M3 12 L12 17 L21 12"/><path d="M3 17 L12 22 L21 17"/>',
    train:
      '<path d="M12 2c3 2 4.5 5 4.5 8 0 2.5-1.5 4.7-4.5 7-3-2.3-4.5-4.5-4.5-7 0-3 1.5-6 4.5-8Z"/>' +
      '<circle cx="12" cy="9.5" r="1.6"/><path d="M8.5 15 L6 20 M15.5 15 L18 20"/>',
    settings:
      '<circle cx="12" cy="12" r="3.2"/><path d="M12 2v3 M12 19v3 M2 12h3 M19 12h3 ' +
      'M4.9 4.9l2.1 2.1 M17 17l2.1 2.1 M19.1 4.9l-2.1 2.1 M7 17l-2.1 2.1"/>',
    chevron: '<path d="M9 6l6 6-6 6"/>',
    plus: '<path d="M12 5v14 M5 12h14"/>',
    status: '<path d="M3 12h4l2 6 4-16 2 10h6"/>',
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

  // Settings-field help (item 4/6): app-layer copy mirroring app/field_help.py SETTINGS_HELP.
  const SETTINGS_HELP = {
    project_root: { help: "Root of the drone-fly project the app operates on (configs/, training/, artifacts/).", example: "/home/you/drone-fly" },
    default_connectome: { help: "Default connectome path pre-filled into new configs. Optional.", example: "data/connectome" },
    host: { help: "Loopback host the local server binds to.", example: "127.0.0.1" },
    port: { help: "Server port. 0 chooses an ephemeral free port at startup.", example: "0" },
    train_executable: {
      help: "Interpreter/executable used to launch training. Point it at a venv that has pybullet " +
        "(a venv dir, its python, or a drone-fly console script) when the app's own venv lacks it. " +
        "Leave blank to auto-detect a pybullet-capable venv.",
      example: "/home/you/drone-fly/.venv-cuda",
    },
  };

  // ---- helpers -------------------------------------------------------------------------
  async function api(path, opts) {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { const j = await res.json(); detail = j.detail || detail; } catch (e) {}
      throw new Error(detail);
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
    setTimeout(() => { t.className = "toast" + (isError ? " error" : ""); }, 3200);
  }
  function fmt(v, digits) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return digits !== undefined ? v.toFixed(digits) : String(v);
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
  function clearPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
  function setDirty(v) { dirty = v; }
  function guardNav() {
    if (dirty) return confirm("You have unsaved changes. Leave without saving?");
    return true;
  }

  // ---- schema cache --------------------------------------------------------------------
  let trainSchema = null, sliceSchema = null;
  async function getTrainSchema() { if (!trainSchema) trainSchema = (await getJSON("/api/train-configs/schema")).fields; return trainSchema; }
  async function getSliceSchema() { if (!sliceSchema) sliceSchema = (await getJSON("/api/slice-schema")).fields; return sliceSchema; }

  // ---- navigation tree (items 1/3/7) ---------------------------------------------------
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
    const pill = opts.pillHTML || "";
    btn.innerHTML =
      icon(iconName) + '<span class="nav-label">' + esc(label) + "</span>" + pill +
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

  async function refreshNav() {
    let runs = [], slices = [];
    try {
      runs = (await getJSON("/api/runs")).runs;
      slices = (await getJSON("/api/slice-configs")).names;
    } catch (e) { return; }

    // Auto-expand the group/run for the active route so the current view is always revealed.
    const hash = location.hash || "#/train";
    if (hash.startsWith("#/slices")) navExpanded.add("group:slices");
    if (hash.startsWith("#/train")) navExpanded.add("group:train");
    const rm = hash.match(/^#\/train\/([^/]+)\//);
    if (rm) navExpanded.add("run:" + decodeURIComponent(rm[1]));

    const tree = document.getElementById("nav-tree");
    tree.innerHTML = "";

    // -- Slices group --
    const sliceKids = h("div", { class: "nav-children" });
    sliceKids.appendChild(leaf("#/slices/new", "plus", "New slice"));
    if (!slices.length) sliceKids.appendChild(h("div", { class: "nav-empty" }, "no slices yet"));
    slices.forEach((n) => sliceKids.appendChild(leaf("#/slices/" + encodeURIComponent(n), "slices", n)));
    tree.appendChild(disclosure("group:slices", "slices", "Slices", sliceKids));

    // -- Train group --
    const trainKids = h("div", { class: "nav-children" });
    trainKids.appendChild(leaf("#/train/new", "plus", "New training"));
    if (!runs.length) trainKids.appendChild(h("div", { class: "nav-empty" }, "no trainings yet"));
    runs.forEach((r) => {
      const runKids = h("div", { class: "nav-children run-children" });
      runKids.appendChild(leaf("#/train/" + encodeURIComponent(r.name) + "/status", "status", "Status"));
      runKids.appendChild(leaf("#/train/" + encodeURIComponent(r.name) + "/recordings", "recordings", "Recordings"));
      runKids.appendChild(leaf("#/train/" + encodeURIComponent(r.name) + "/config", "config", "Config"));
      const pillCls = r.state === "running" ? "pill running" : (r.state === "paused" ? "pill paused" : "pill");
      const pillHTML = '<span class="' + pillCls + '">' + esc(r.state) + "</span>";
      trainKids.appendChild(disclosure("run:" + r.name, "train", r.name, runKids, { run: true, pillHTML }));
    });
    tree.appendChild(disclosure("group:train", "train", "Train", trainKids));

    // -- Settings (single leaf, no disclosure) --
    const setGroup = h("div", { class: "nav-node" });
    setGroup.appendChild(leaf("#/settings", "settings", "Settings"));
    tree.appendChild(setGroup);
  }

  // ---- views ---------------------------------------------------------------------------
  async function renderTrainList() {
    const runs = (await getJSON("/api/runs")).runs;
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("Trainings", "New training", "#/train/new"));
    if (!runs.length) { root.appendChild(emptyState("No trainings yet.", "Create one with “New training”.")); return; }
    const card = h("div", { class: "card" });
    const table = h("table", { class: "list" });
    table.innerHTML = "<thead><tr><th>Name</th><th>State</th><th>Checkpoints</th><th>Final</th><th></th></tr></thead>";
    const tbody = h("tbody");
    runs.forEach((r) => {
      const tr = h("tr");
      tr.innerHTML =
        '<td><a class="link" href="#/train/' + encodeURIComponent(r.name) + '/status">' + esc(r.name) + "</a></td>" +
        '<td><span class="state-tag ' + r.state + '">' + esc(r.state) + "</span></td>" +
        "<td>" + (r.has_checkpoints ? "yes" : "—") + "</td>" +
        "<td>" + (r.has_final ? "yes" : "—") + "</td>" +
        '<td><a class="link" href="#/train/' + encodeURIComponent(r.name) + '/status">open →</a></td>';
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    card.appendChild(table);
    root.appendChild(card);
  }

  async function renderTrainNew() {
    const fields = await getTrainSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("New training", null, null));
    const nameCard = h("div", { class: "card" });
    nameCard.appendChild(h("h3", null, "Run name"));
    const nameInput = h("input", { type: "text", placeholder: "e.g. baseline" });
    nameInput.addEventListener("input", () => setDirty(true));
    nameCard.appendChild(nameInput);
    root.appendChild(nameCard);

    const formCard = h("div", { class: "card" });
    formCard.appendChild(h("h3", null, "Training configuration"));
    const formHost = h("div");
    formCard.appendChild(formHost);
    root.appendChild(formCard);
    const form = Forms.buildForm(formHost, fields.filter((f) => f.name !== "name"), {});
    formHost.addEventListener("input", () => setDirty(true));

    const actions = h("div", { class: "btn-row" });
    const saveBtn = h("button", { class: "primary" }, "Save");
    const launchBtn = h("button", null, "Save &amp; launch");
    actions.appendChild(saveBtn); actions.appendChild(launchBtn);
    root.appendChild(actions);

    async function save() {
      const name = nameInput.value.trim();
      if (!name) { toast("A run name is required.", true); return null; }
      const config = form.collect();
      await postJSON("/api/train-configs/" + encodeURIComponent(name), { config });
      setDirty(false);
      toast("Saved configs/train/" + name + ".yaml");
      await refreshNav();
      return name;
    }
    saveBtn.addEventListener("click", async () => { try { const n = await save(); if (n) location.hash = "#/train/" + encodeURIComponent(n) + "/status"; } catch (e) { toast(e.message, true); } });
    launchBtn.addEventListener("click", async () => {
      try { const n = await save(); if (!n) return; await postJSON("/api/runs/" + encodeURIComponent(n) + "/launch"); toast("Launched " + n); location.hash = "#/train/" + encodeURIComponent(n) + "/status"; }
      catch (e) { toast(e.message, true); }
    });
  }

  // -- item 1: split run detail into Status / Recordings / Config --
  function runHead(name, sub) {
    const bar = h("div", { class: "view-head" });
    const wrap = h("div");
    wrap.appendChild(h("h2", null, esc(name)));
    wrap.appendChild(h("p", { class: "view-sub subtle" }, sub));
    bar.appendChild(wrap);
    return bar;
  }

  async function renderRunStatus(name) {
    const root = view();
    root.innerHTML = "";
    root.appendChild(runHead(name, "Status"));

    const monitor = h("div", { class: "card" });
    monitor.appendChild(h("h3", null, "Live status"));
    const controls = h("div", { class: "btn-row" });
    monitor.appendChild(controls);
    const progWrap = h("div", { class: "progress" }); progWrap.appendChild(h("span"));
    const progLabel = h("p", { class: "subtle" }, "—");
    monitor.appendChild(progLabel); monitor.appendChild(progWrap);
    const metrics = h("div", { class: "metrics" }); metrics.style.marginTop = "12px";
    monitor.appendChild(metrics);
    root.appendChild(monitor);

    function renderControls(run) {
      controls.innerHTML = "";
      const tag = h("span", { class: "state-tag " + run.state }, run.state);
      controls.appendChild(tag);
      const mk = (label, cls, fn, enabled) => { const b = h("button", { class: cls || "" }, label); b.disabled = !enabled; b.addEventListener("click", fn); controls.appendChild(b); };
      const live = run.running;
      mk("Launch", "primary", () => act(name, "launch"), !live);
      mk("Pause", "", () => act(name, "pause"), live);
      mk("Resume", "", () => act(name, "resume"), !live && run.resumable);
      mk("Stop", "danger", () => act(name, "stop"), live);
    }
    async function act(n, verb) {
      try { await postJSON("/api/runs/" + encodeURIComponent(n) + "/" + verb); toast(verb + " → " + n); await tick(); await refreshNav(); }
      catch (e) { toast(e.message, true); }
    }
    function metricCard(k, v) { const m = h("div", { class: "metric" }); m.innerHTML = '<div class="k">' + esc(k) + '</div><div class="v">' + esc(v) + "</div>"; return m; }
    function renderMetrics(l) {
      metrics.innerHTML = "";
      const ro = l.rollout || {}, tr = l.train || {}, he = l.health || {}, dy = l.dynamics || {};
      metrics.appendChild(metricCard("ep_rew_mean", fmt(ro.ep_rew_mean, 3)));
      metrics.appendChild(metricCard("ep_len_mean", fmt(ro.ep_len_mean, 1)));
      metrics.appendChild(metricCard("success", fmt(ro.success_rate, 3)));
      metrics.appendChild(metricCard("value_loss", fmt(tr.value_loss, 4)));
      metrics.appendChild(metricCard("approx_kl", fmt(tr.approx_kl, 4)));
      metrics.appendChild(metricCard("expl_var", fmt(tr.explained_variance, 3)));
      if (he.status) { const m = h("div", { class: "metric" }); m.innerHTML = '<div class="k">health</div><div class="v"><span class="badge ' + he.status + '">' + esc(he.status) + "</span></div>"; metrics.appendChild(m); }
      if (dy.thrust_to_weight !== undefined && dy.thrust_to_weight !== null) metrics.appendChild(metricCard("T/W", fmt(dy.thrust_to_weight, 2)));
    }
    function renderMetricsCsv(row) {
      metrics.innerHTML = "";
      Object.keys(row).slice(0, 8).forEach((k) => metrics.appendChild(metricCard(k, fmt(row[k], 3))));
    }
    async function tick() {
      let run;
      try { run = await getJSON("/api/runs/" + encodeURIComponent(name)); } catch (e) { return; }
      renderControls(run);
      try {
        const snap = await getJSON("/api/runs/" + encodeURIComponent(name) + "/status");
        const latest = snap.latest;
        if (latest && snap.source === "jsonl") {
          const pct = latest.target_timesteps ? Math.min(100, 100 * latest.timesteps / latest.target_timesteps) : 0;
          progWrap.firstChild.style.width = pct + "%";
          progLabel.textContent = "step " + fmt(latest.timesteps) + " / " + fmt(latest.target_timesteps) +
            " · update " + fmt(latest.n_updates) + " · " + fmt(latest.fps, 0) + " fps · source: jsonl";
          renderMetrics(latest);
        } else if (latest && snap.source === "csv") {
          progLabel.textContent = "source: progress.csv (fallback)";
          renderMetricsCsv(latest);
        } else {
          progLabel.textContent = "no progress yet";
        }
      } catch (e) {}
    }
    await tick();
    clearPoll();
    pollTimer = setInterval(tick, 2500);
  }

  async function renderRunRecordings(name) {
    const root = view();
    root.innerHTML = "";
    root.appendChild(runHead(name, "Recordings"));
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Episode playback"));
    const pickerRow = h("div", { class: "picker-row" });
    card.appendChild(pickerRow);
    const viewerHost = h("div", { class: "viewer-host" });
    card.appendChild(viewerHost);
    root.appendChild(card);

    let recs = [];
    try { recs = (await getJSON("/api/runs/" + encodeURIComponent(name) + "/recordings")).recordings; }
    catch (e) { pickerRow.appendChild(h("p", { class: "subtle" }, "—")); return; }

    if (!recs.length) {
      pickerRow.appendChild(h("p", { class: "subtle" }, "No recordings yet. They appear here once a run records episodes (enable “record” in the run config)."));
      return;
    }
    const label = h("label", { class: "picker-label", for: "rec-picker" }, "Episode");
    const select = h("select", { class: "picker", id: "rec-picker" });
    recs.forEach((r) => {
      const o = h("option", { value: String(r.episode) }, "episode " + r.episode + (r.gzip ? " (gz)" : ""));
      select.appendChild(o);
    });
    pickerRow.appendChild(label);
    pickerRow.appendChild(select);
    select.addEventListener("change", () => ViewerEmbed.embedRecording(viewerHost, name, select.value));
    // Auto-load the first episode.
    ViewerEmbed.embedRecording(viewerHost, name, recs[0].episode);
  }

  async function renderRunConfig(name) {
    const fields = await getTrainSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(runHead(name, "Config"));
    const cfgCard = h("div", { class: "card" });
    cfgCard.appendChild(h("h3", null, "Configuration"));
    const cfgHost = h("div");
    cfgCard.appendChild(cfgHost);
    const saved = (await getJSON("/api/train-configs/" + encodeURIComponent(name))).config || {};
    const form = Forms.buildForm(cfgHost, fields.filter((f) => f.name !== "name"), saved);
    cfgHost.addEventListener("input", () => setDirty(true));
    const cfgActions = h("div", { class: "btn-row" });
    const saveBtn = h("button", { class: "primary" }, "Save");
    cfgActions.appendChild(saveBtn);
    cfgCard.appendChild(cfgActions);
    root.appendChild(cfgCard);
    saveBtn.addEventListener("click", async () => {
      try { await postJSON("/api/train-configs/" + encodeURIComponent(name), { config: form.collect() }); setDirty(false); toast("Saved."); }
      catch (e) { toast(e.message, true); }
    });
  }

  async function renderSliceNew() {
    const fields = await getSliceSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("New slice", null, null));
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Slice (prune) configuration"));
    const host = h("div"); card.appendChild(host);
    root.appendChild(card);
    const form = Forms.buildForm(host, fields, {});
    host.addEventListener("input", () => setDirty(true));
    const actions = h("div", { class: "btn-row" });
    const genBtn = h("button", { class: "primary" }, "Generate slice");
    actions.appendChild(genBtn);
    root.appendChild(actions);
    genBtn.addEventListener("click", async () => {
      try {
        const config = form.collect();
        const res = await postJSON("/api/slices", { config });
        setDirty(false);
        toast("Slice build launched → " + res.config);
        await refreshNav();
      } catch (e) { toast(e.message, true); }
    });
  }

  async function renderSliceDetail(name) {
    const fields = await getSliceSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("Slice · " + name, null, null));
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Slice (prune) configuration"));
    const host = h("div"); card.appendChild(host);
    root.appendChild(card);
    const saved = (await getJSON("/api/slice-configs/" + encodeURIComponent(name))).config || {};
    const form = Forms.buildForm(host, fields, saved);
    host.addEventListener("input", () => setDirty(true));
    const actions = h("div", { class: "btn-row" });
    const genBtn = h("button", { class: "primary" }, "Regenerate slice");
    actions.appendChild(genBtn);
    root.appendChild(actions);
    genBtn.addEventListener("click", async () => {
      try {
        const config = form.collect();
        const res = await postJSON("/api/slice-configs/" + encodeURIComponent(name), { config });
        setDirty(false);
        toast("Slice regenerate launched → " + res.config);
        await refreshNav();
      } catch (e) { toast(e.message, true); }
    });
  }

  async function renderSettings() {
    const s = await getJSON("/api/settings");
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("Settings", null, null));
    const card = h("div", { class: "card" });
    const rows = [
      ["project_root", "Project root", s.project_root || "", "text"],
      ["default_connectome", "Default connectome path", s.default_connectome || "", "text"],
      ["train_executable", "Training interpreter", s.train_executable || "", "text"],
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
    const actions = h("div", { class: "btn-row" });
    const saveBtn = h("button", { class: "primary" }, "Save");
    actions.appendChild(saveBtn);
    root.appendChild(actions);
    saveBtn.addEventListener("click", async () => {
      try {
        await postJSON("/api/settings", {
          project_root: inputs.project_root.value || null,
          default_connectome: inputs.default_connectome.value || null,
          train_executable: inputs.train_executable.value || null,
          host: inputs.host.value || "127.0.0.1",
          port: parseInt(inputs.port.value, 10) || 0,
        });
        setDirty(false); toast("Settings saved.");
      } catch (e) { toast(e.message, true); }
    });
  }

  // ---- shared UI bits ------------------------------------------------------------------
  // item 6: an inline-SVG info button that opens the accessible modal with help/example/constraints.
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
  // Exposed so forms.js can build identical info buttons for config fields.
  window.AppUI = { infoButton, icon, esc };

  function emptyState(title, sub) {
    const box = h("div", { class: "empty-state" });
    box.appendChild(h("p", { class: "empty-title" }, title));
    if (sub) box.appendChild(h("p", { class: "subtle" }, sub));
    return box;
  }

  function headBar(title, actionLabel, actionHash) {
    const bar = h("div", { class: "view-head" });
    bar.appendChild(h("h2", null, esc(title)));
    if (actionLabel && actionHash) {
      const a = h("a", { class: "btn primary", href: actionHash }, actionLabel);
      bar.appendChild(a);
    }
    return bar;
  }

  // ---- router --------------------------------------------------------------------------
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
    try {
      const parts = hash.replace(/^#\//, "").split("/");
      if (parts[0] === "train" && parts[1] === "new") await renderTrainNew();
      else if (parts[0] === "train" && parts[1]) {
        const name = decodeURIComponent(parts[1]);
        const leafName = parts[2];
        if (leafName === "recordings") await renderRunRecordings(name);
        else if (leafName === "config") await renderRunConfig(name);
        else if (leafName === "status") await renderRunStatus(name);
        else { location.hash = "#/train/" + encodeURIComponent(name) + "/status"; return; }
      }
      else if (parts[0] === "train") await renderTrainList();
      else if (parts[0] === "slices" && parts[1] && parts[1] !== "new") await renderSliceDetail(decodeURIComponent(parts[1]));
      else if (parts[0] === "slices") await renderSliceNew();
      else if (parts[0] === "settings") await renderSettings();
      else await renderTrainList();
    } catch (e) {
      view().innerHTML = '<p class="subtle">Error: ' + esc(e && e.message ? e.message : e) + "</p>";
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
