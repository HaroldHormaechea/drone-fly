/* app.js — vanilla hash-routed shell for the drone-fly desktop app (UC-61 AC2).
 *
 * Left-nav (Generate slices / Train / Settings) drives a right-side main view. Routes:
 *   #/train                 run list
 *   #/train/new             new training config
 *   #/train/<name>          monitor + edit + recordings for one run
 *   #/slices/new            define/generate a slice
 *   #/settings              local tool settings
 * Back/forward work because navigation is pure hashchange. Config edits live in local state and
 * persist ONLY via an explicit Save button (AC9); an unsaved-changes guard warns on navigation.
 */
(function () {
  "use strict";

  const view = () => document.getElementById("view");
  let pollTimer = null;
  let dirty = false;

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

  // ---- navigation ----------------------------------------------------------------------
  async function refreshNav() {
    const box = document.getElementById("nav-runs");
    let runs = [];
    try { runs = (await getJSON("/api/runs")).runs; } catch (e) { return; }
    box.innerHTML = "";
    runs.forEach((r) => {
      const a = h("a", { class: "nav-item", "data-route": "#/train/" + r.name, href: "#/train/" + encodeURIComponent(r.name) });
      const pillCls = r.state === "running" ? "pill running" : (r.state === "paused" ? "pill paused" : "pill");
      a.innerHTML = '<span>' + r.name + '</span><span class="' + pillCls + '">' + r.state + '</span>';
      box.appendChild(a);
    });
    highlightNav();
  }
  function highlightNav() {
    document.querySelectorAll(".nav-item").forEach((n) => {
      n.classList.toggle("active", n.getAttribute("data-route") === location.hash ||
        (location.hash.startsWith(n.getAttribute("data-route")) && n.getAttribute("data-route").length > 3));
    });
  }

  // ---- views ---------------------------------------------------------------------------
  async function renderTrainList() {
    const runs = (await getJSON("/api/runs")).runs;
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("Trainings", "New training", "#/train/new"));
    if (!runs.length) { root.appendChild(h("p", { class: "subtle" }, "No trainings yet. Create one with “New training”.")); return; }
    const card = h("div", { class: "card" });
    const table = h("table", { class: "list" });
    table.innerHTML = "<thead><tr><th>Name</th><th>State</th><th>Checkpoints</th><th>Final</th><th></th></tr></thead>";
    const tbody = h("tbody");
    runs.forEach((r) => {
      const tr = h("tr");
      tr.innerHTML =
        '<td><a class="link" href="#/train/' + encodeURIComponent(r.name) + '">' + r.name + "</a></td>" +
        '<td><span class="state-tag ' + r.state + '">' + r.state + "</span></td>" +
        "<td>" + (r.has_checkpoints ? "yes" : "—") + "</td>" +
        "<td>" + (r.has_final ? "yes" : "—") + "</td>" +
        '<td><a class="link" href="#/train/' + encodeURIComponent(r.name) + '">open →</a></td>';
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
    saveBtn.addEventListener("click", async () => { try { const n = await save(); if (n) location.hash = "#/train/" + encodeURIComponent(n); } catch (e) { toast(e.message, true); } });
    launchBtn.addEventListener("click", async () => {
      try { const n = await save(); if (!n) return; await postJSON("/api/runs/" + encodeURIComponent(n) + "/launch"); toast("Launched " + n); location.hash = "#/train/" + encodeURIComponent(n); }
      catch (e) { toast(e.message, true); }
    });
  }

  async function renderTrainDetail(name) {
    const fields = await getTrainSchema();
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar(name, null, null));

    // Controls + live status card.
    const monitor = h("div", { class: "card" });
    monitor.appendChild(h("h3", null, "Status"));
    const controls = h("div", { class: "btn-row" });
    monitor.appendChild(controls);
    const progWrap = h("div", { class: "progress" }); progWrap.appendChild(h("span"));
    const progLabel = h("p", { class: "subtle" }, "—");
    monitor.appendChild(progLabel); monitor.appendChild(progWrap);
    const metrics = h("div", { class: "metrics" }); metrics.style.marginTop = "12px";
    monitor.appendChild(metrics);
    root.appendChild(monitor);

    // Recordings.
    const recCard = h("div", { class: "card" });
    recCard.appendChild(h("h3", null, "Recordings"));
    const recList = h("div", { class: "subtle" }, "—");
    recCard.appendChild(recList);
    const viewerHost = h("div"); viewerHost.style.marginTop = "12px";
    recCard.appendChild(viewerHost);
    root.appendChild(recCard);

    // Config editor.
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

    async function loadRecordings() {
      try {
        const recs = (await getJSON("/api/runs/" + encodeURIComponent(name) + "/recordings")).recordings;
        recList.innerHTML = "";
        if (!recs.length) { recList.textContent = "No recordings yet."; return; }
        recs.forEach((r) => {
          const a = h("a", { class: "link" }, "episode " + r.episode + (r.gzip ? " (gz)" : ""));
          a.style.marginRight = "14px";
          a.addEventListener("click", () => ViewerEmbed.embedRecording(viewerHost, name, r.episode));
          recList.appendChild(a);
        });
      } catch (e) { recList.textContent = "—"; }
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
    function metricCard(k, v) { const m = h("div", { class: "metric" }); m.innerHTML = '<div class="k">' + k + '</div><div class="v">' + v + "</div>"; return m; }
    function renderMetrics(l) {
      metrics.innerHTML = "";
      const ro = l.rollout || {}, tr = l.train || {}, he = l.health || {}, dy = l.dynamics || {};
      metrics.appendChild(metricCard("ep_rew_mean", fmt(ro.ep_rew_mean, 3)));
      metrics.appendChild(metricCard("ep_len_mean", fmt(ro.ep_len_mean, 1)));
      metrics.appendChild(metricCard("success", fmt(ro.success_rate, 3)));
      metrics.appendChild(metricCard("value_loss", fmt(tr.value_loss, 4)));
      metrics.appendChild(metricCard("approx_kl", fmt(tr.approx_kl, 4)));
      metrics.appendChild(metricCard("expl_var", fmt(tr.explained_variance, 3)));
      if (he.status) { const m = h("div", { class: "metric" }); m.innerHTML = '<div class="k">health</div><div class="v"><span class="badge ' + he.status + '">' + he.status + "</span></div>"; metrics.appendChild(m); }
      if (dy.thrust_to_weight !== undefined && dy.thrust_to_weight !== null) metrics.appendChild(metricCard("T/W", fmt(dy.thrust_to_weight, 2)));
    }
    function renderMetricsCsv(row) {
      metrics.innerHTML = "";
      Object.keys(row).slice(0, 8).forEach((k) => metrics.appendChild(metricCard(k, fmt(row[k], 3))));
    }

    await tick();
    await loadRecordings();
    clearPoll();
    pollTimer = setInterval(async () => { await tick(); await loadRecordings(); }, 2500);
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
      ["project_root", "Project root", s.project_root || ""],
      ["default_connectome", "Default connectome path", s.default_connectome || ""],
      ["host", "Server host", s.host || "127.0.0.1"],
      ["port", "Server port (0 = ephemeral)", s.port != null ? s.port : 0],
    ];
    const inputs = {};
    rows.forEach(([k, label, val]) => {
      const row = h("div", { class: "field" });
      row.appendChild(h("label", { class: "key" }, label));
      const inp = h("input", { type: k === "port" ? "number" : "text" });
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
          host: inputs.host.value || "127.0.0.1",
          port: parseInt(inputs.port.value, 10) || 0,
        });
        setDirty(false); toast("Settings saved.");
      } catch (e) { toast(e.message, true); }
    });
  }

  function headBar(title, actionLabel, actionHash) {
    const bar = h("div", { class: "view-head" });
    bar.appendChild(h("h2", null, title));
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
    highlightNav();
    try {
      const parts = hash.replace(/^#\//, "").split("/");
      if (parts[0] === "train" && parts[1] === "new") await renderTrainNew();
      else if (parts[0] === "train" && parts[1]) await renderTrainDetail(decodeURIComponent(parts.slice(1).join("/")));
      else if (parts[0] === "train") await renderTrainList();
      else if (parts[0] === "slices") await renderSliceNew();
      else if (parts[0] === "settings") await renderSettings();
      else await renderTrainList();
    } catch (e) {
      view().innerHTML = '<p class="subtle">Error: ' + (e && e.message ? e.message : e) + "</p>";
    }
    highlightNav();
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
