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
 *   #/about                          static about page (app name/purpose, version, author)
 * Back/forward work because navigation is pure hashchange. Config edits live in local state and
 * persist ONLY via an explicit Save button (AC9); an unsaved-changes guard warns on navigation.
 */
(function () {
  "use strict";

  const view = () => document.getElementById("view");
  let pollTimer = null;
  // item 1: a 1 Hz ticker that advances the Elapsed box between the 2.5 s status polls. Lives at
  // module scope so clearPoll() tears it down alongside pollTimer (no leak across route() changes).
  let elapsedTimer = null;
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
    if (typeof v === "number") {
      if (digits !== undefined) return v.toFixed(digits);
      // item 3: unformatted integers get thousands separators (e.g. 2,000,000); non-integers keep
      // their full precision. Callers passing an explicit `digits` are unaffected.
      return Number.isInteger(v) ? v.toLocaleString("en-US") : String(v);
    }
    return String(v);
  }
  // item 3: format a value as a grouped integer (thousands separators, no decimals).
  function fmtInt(v) {
    if (v === null || v === undefined) return "—";
    return typeof v === "number" ? v.toLocaleString("en-US") : String(v);
  }
  // status-view polish item 2: format a duration (seconds) as "1h 23m" / "2m 05s" / "45s".
  // Non-finite / negative → "—" so ETA and Elapsed degrade cleanly.
  function fmtDuration(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) return h + "h " + m + "m";
    if (m > 0) return m + "m " + String(s).padStart(2, "0") + "s";
    return s + "s";
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
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }  // item 1: no timer leak.
  }
  function setDirty(v) { dirty = v; }
  function guardNav() {
    if (dirty) return confirm("You have unsaved changes. Leave without saving?");
    return true;
  }

  // ---- schema cache --------------------------------------------------------------------
  let trainSchema = null, sliceSchema = null, statusFields = null;
  async function getTrainSchema() { if (!trainSchema) trainSchema = (await getJSON("/api/train-configs/schema")).fields; return trainSchema; }
  async function getSliceSchema() { if (!sliceSchema) sliceSchema = (await getJSON("/api/slice-schema")).fields; return sliceSchema; }
  // status-view polish items 5/6: the backend-owned live-status descriptor list (label / group /
  // path|computed / format / help), fetched once and cached like the config schemas.
  async function getStatusFields() { if (!statusFields) statusFields = (await getJSON("/api/status-fields")).fields; return statusFields; }

  // item 4: prefetch dynamic combobox suggestions for any field carrying an `options_endpoint`, so
  // the (synchronous) form builder can attach them as a datalist. Unique endpoints are fetched once;
  // the response's first array value is used. A failed fetch degrades silently to no suggestions
  // (the field stays a plain text input). Returns a fieldName -> [strings] map.
  async function fetchFieldOptions(fields) {
    const byField = {};
    const endpoints = {};  // endpoint URL -> [field names using it]
    fields.forEach((f) => {
      if (f.options_endpoint) (endpoints[f.options_endpoint] = endpoints[f.options_endpoint] || []).push(f.name);
    });
    await Promise.all(Object.keys(endpoints).map(async (ep) => {
      try {
        const data = await getJSON(ep);
        let arr = null;
        if (Array.isArray(data)) arr = data;
        else if (data && typeof data === "object") {
          for (const k in data) { if (Array.isArray(data[k])) { arr = data[k]; break; } }
        }
        if (arr) endpoints[ep].forEach((fn) => { byField[fn] = arr; });
      } catch (e) { /* degrade to a plain input */ }
    }));
    return byField;
  }

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

  // item 1: a top-level nav link (no chevron, no child indent) aligned with the group heads,
  // used for standalone destinations that have no children (Settings, About).
  function topLink(href, iconName, label) {
    const a = h("a", { class: "nav-leaf nav-top", "data-route": href, href: href },
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

    // -- Settings + About (top-level leaves, no disclosure, aligned with group heads) --
    const setGroup = h("div", { class: "nav-node" });
    setGroup.appendChild(topLink("#/settings", "settings", "Settings"));
    setGroup.appendChild(topLink("#/about", "info", "About"));
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
      const enc = encodeURIComponent(r.name);
      const tr = h("tr");
      tr.innerHTML =
        '<td><a class="link" href="#/train/' + enc + '/status">' + esc(r.name) + "</a></td>" +
        '<td><span class="state-tag ' + r.state + '">' + esc(r.state) + "</span></td>" +
        "<td>" + (r.has_checkpoints ? "yes" : "—") + "</td>" +
        "<td>" + (r.has_final ? "yes" : "—") + "</td>";
      // item 3: per-row actions — the existing "open →" link plus a destructive Delete button.
      const actTd = h("td");
      const actions = h("div", { class: "row-actions" });
      actions.appendChild(h("a", { class: "link", href: "#/train/" + enc + "/status" }, "open →"));
      const delBtn = h("button", { class: "danger btn-sm", "aria-label": "Delete run " + r.name }, "Delete");
      delBtn.addEventListener("click", () => onDeleteRun(r));
      actions.appendChild(delBtn);
      actTd.appendChild(actions);
      tr.appendChild(actTd);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    card.appendChild(table);
    root.appendChild(card);
  }

  // item 3: confirm + perform a destructive run delete. Default deletes only the generated outputs
  // (training/<name>/); the modal's opt-in checkbox also removes the committed configs/train YAML.
  // Challenger note 1: with the default (config kept) the run stays listed as a config-only row, so
  // the copy states this explicitly.
  async function confirmDeleteRun(name) {
    const bodyHTML =
      '<p class="mh-help">Permanently delete the generated outputs for <strong>' + esc(name) +
      "</strong> — checkpoints, recordings and logs under <code>training/" + esc(name) +
      "/</code>. This can’t be undone.</p>" +
      '<p class="mh-help">The saved run definition (<code>configs/train/' + esc(name) +
      ".yaml</code>) is <strong>kept</strong> by default, so the run stays in the list as a " +
      "config-only entry you can relaunch. Tick the box to also remove the saved config.</p>" +
      '<label class="enable"><input type="checkbox" id="del-config-cb"> ' +
      "Also delete the saved config (removes the run definition entirely)</label>";
    const m = openModal({
      title: "Delete " + name,
      bodyHTML,
      actions: [
        { label: "Cancel", value: "cancel" },
        { label: "Delete", value: "delete", class: "danger" },
      ],
      invoker: document.activeElement,
    });
    // Capture the checkbox node BEFORE awaiting — it stays readable even after the modal detaches.
    const cb = m.body.querySelector("#del-config-cb");
    const choice = await m.choice;
    return { confirmed: choice === "delete", deleteConfig: !!(cb && cb.checked) };
  }

  async function onDeleteRun(r) {
    let decision;
    try { decision = await confirmDeleteRun(r.name); } catch (e) { return; }
    if (!decision.confirmed) return;
    const enc = encodeURIComponent(r.name);
    try {
      const resp = await api(
        "/api/runs/" + enc + (decision.deleteConfig ? "?delete_config=true" : ""),
        { method: "DELETE" }
      );
      if (resp && resp.deleted_config) toast("Deleted " + r.name + " (outputs + saved config).");
      else if (resp && resp.deleted_outputs)
        toast("Deleted " + r.name + " outputs. Run definition kept — tick the box to also remove it.");
      else toast("Nothing to delete for " + r.name + ".");
      const viewingThis = location.hash.indexOf("#/train/" + enc + "/") === 0;
      if (viewingThis) {
        // Bounce back to the list; the hashchange → route() renders it (and refreshes the nav).
        location.hash = "#/train";
      } else {
        // Already on the list → refresh nav + re-render in place.
        await refreshNav();
        await renderTrainList();
      }
    } catch (e) { toast(e.message, true); }
  }

  async function renderTrainNew() {
    const fields = await getTrainSchema();
    const root = view();
    root.innerHTML = "";
    // item 2: Save + "Save & launch" live in the top-right header slot (two distinct buttons —
    // launch stays separate from a plain save).
    const saveBtn = h("button", { class: "primary" }, "Save");
    const launchBtn = h("button", null, "Save &amp; launch");
    root.appendChild(headBar("New training", null, null, [saveBtn, launchBtn]));
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
    const optionsByField = await fetchFieldOptions(fields);  // item 4: connectome/resume suggestions.
    const form = Forms.buildForm(formHost, fields.filter((f) => f.name !== "name"), {}, optionsByField);
    formHost.addEventListener("input", () => setDirty(true));

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
  // `actions` (item 2): optional button nodes rendered in the right-aligned header slot (Save).
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

  // status-view polish item 5/6: read a leaf out of the status record by descriptor path.
  // Returns undefined when the path is unresolved (any ancestor null / absent / non-object), which
  // the formatter renders as "—" — so a box is shape-stable across polls even before data arrives.
  function readPath(obj, path) {
    let cur = obj;
    for (let i = 0; i < path.length; i++) {
      if (cur === null || cur === undefined || typeof cur !== "object") return undefined;
      cur = cur[path[i]];
    }
    return cur;
  }
  // Format a status leaf per its descriptor's closed `format.type` vocabulary. null / undefined /
  // NaN always collapse to "—" (the "not observed" sentinel), regardless of type.
  function formatStatusValue(value, format) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "number" && !Number.isFinite(value)) return "—";
    const type = format && format.type;
    if (type === "int") return fmtInt(value);
    if (type === "float") return fmt(value, format.digits);
    if (type === "seconds" || type === "duration") return fmtDuration(value);
    // "text" (and any unknown type) → plain string.
    return String(value);
  }
  // item 2: derive ETA from the live record (no backend field). rate = fps when positive, else the
  // average steps/sec so far; clamps to "—" (unknown target / no rate) and "done" (target reached).
  function computeEta(latest) {
    const target = latest.target_timesteps;
    if (target === null || target === undefined) return "—";
    const ts = typeof latest.timesteps === "number" ? latest.timesteps : 0;
    const fps = latest.fps;
    const elapsed = latest.elapsed_seconds;
    let rate = 0;
    if (typeof fps === "number" && Number.isFinite(fps) && fps > 0) rate = fps;
    else if (typeof elapsed === "number" && elapsed > 0) rate = ts / elapsed;
    if (rate <= 0) return "—";
    const remaining = target - ts;
    if (remaining <= 0) return "done";
    return fmtDuration(remaining / rate);
  }

  async function renderRunStatus(name) {
    const fields = await getStatusFields();
    const root = view();
    root.innerHTML = "";
    // item 4: fill the window height in the status view so the log panel grows to the bottom.
    // `route()` clears `.view-fill` on every dispatch, so only this view uses fill mode.
    root.classList.add("view-fill");
    root.appendChild(runHead(name, "Status"));

    const monitor = h("div", { class: "card" });
    monitor.appendChild(h("h3", null, "Live status"));
    const controls = h("div", { class: "btn-row" });
    monitor.appendChild(controls);
    const progWrap = h("div", { class: "progress" }); progWrap.appendChild(h("span"));
    const progLabel = h("p", { class: "subtle" }, "—");
    monitor.appendChild(progLabel); monitor.appendChild(progWrap);
    // items 5/6: grouped metric boxes are stacked (heading + grid per group) inside this container.
    const metrics = h("div", { class: "metrics-groups" }); metrics.style.marginTop = "12px";
    monitor.appendChild(metrics);
    // item 5 (prior): live process-log panel below the metric boxes. Filled by a byte-offset tail
    // folded into the same 2.5s poll; `logCursor` adopts the server's returned `next` verbatim.
    monitor.appendChild(h("p", { class: "subtle log-head" }, "Process logs"));
    const logPanel = h("pre", { class: "log-panel" });
    monitor.appendChild(logPanel);
    let logCursor = 0;
    // item 1: drift-free elapsed clock. `elapsedBase` is the last server-reported elapsed_seconds,
    // stamped against `elapsedBaseAt` (a performance.now() reading) so the 1 Hz ticker can add the
    // wall-time since the last poll. `runActive` gates the ticker (freeze the clock once stopped).
    let elapsedBase = null, elapsedBaseAt = null, runActive = false;
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
    // items 5/6: build one grouped, backend-described metric box (label + info button + value).
    function statusBox(d, latest) {
      const box = h("div", { class: "metric" });
      const k = h("div", { class: "k" });
      k.appendChild(document.createTextNode(d.label));
      if (d.help) {
        // item 3: health.message rides along in the Health box's info popup when present.
        let extra = null;
        if (d.format && d.format.type === "badge") {
          const msg = latest.health && latest.health.message;
          if (msg) extra = '<p class="mh-help">' + esc(msg) + "</p>";
        }
        k.appendChild(infoButton(d.label, d.help, d.example, extra));
      }
      box.appendChild(k);
      const v = h("div", { class: "v" });
      // item 1: tag the Elapsed value node so the 1 Hz ticker can advance it between polls.
      if (d.key === "elapsed_seconds") v.classList.add("js-elapsed");
      if (d.computed && d.key === "eta") {
        v.textContent = computeEta(latest);
      } else if (d.format && d.format.type === "badge") {
        const raw = readPath(latest, d.path);
        if (raw === null || raw === undefined) v.textContent = "—";
        else v.innerHTML = '<span class="badge ' + esc(raw) + '">' + esc(raw) + "</span>";
      } else {
        v.textContent = formatStatusValue(readPath(latest, d.path), d.format);
      }
      box.appendChild(v);
      return box;
    }
    // Whole-group suppression (item 5): hide a heading+grid only when every descriptor in it is a
    // path-backed nested leaf (path length >= 2) whose top-level block is null/absent — i.e. the
    // optional Health / Dynamics blocks on a just-launched run. Progress/Rollout/Train always show
    // (they carry length-1 paths or the computed ETA, or their block is always present).
    function groupSuppressed(descriptors, latest) {
      return descriptors.every((d) =>
        !d.computed && Array.isArray(d.path) && d.path.length >= 2 &&
        (latest[d.path[0]] === null || latest[d.path[0]] === undefined));
    }
    function renderMetricsGrouped(latest) {
      metrics.innerHTML = "";
      // Group by descriptor order → groups appear in the backend's STATUS_GROUP_ORDER (the
      // descriptors are authored contiguously per group).
      const order = [];
      const byGroup = new Map();
      fields.forEach((d) => {
        if (!byGroup.has(d.group)) { byGroup.set(d.group, []); order.push(d.group); }
        byGroup.get(d.group).push(d);
      });
      order.forEach((group) => {
        const descriptors = byGroup.get(group);
        if (groupSuppressed(descriptors, latest)) return;
        metrics.appendChild(h("p", { class: "metric-group-title" }, esc(group)));
        const grid = h("div", { class: "metrics" });
        descriptors.forEach((d) => grid.appendChild(statusBox(d, latest)));
        metrics.appendChild(grid);
      });
    }
    function renderMetricsCsv(row) {
      metrics.innerHTML = "";
      // CSV fallback stays flat/ungrouped/help-less (different key namespace); wrapped in a single
      // `.metrics` grid so it looks identical to the pre-grouping layout.
      const grid = h("div", { class: "metrics" });
      // item 3: integer-valued cells (e.g. 100000) render as grouped ints; genuine decimals keep 3dp.
      Object.keys(row).slice(0, 8).forEach((k) =>
        grid.appendChild(metricCard(k, Number.isInteger(row[k]) ? fmtInt(row[k]) : fmt(row[k], 3))));
      metrics.appendChild(grid);
    }
    // item 2: switch the progress bar between determinate (known target → real width) and an
    // indeterminate marquee (running but no measurable progress yet, e.g. just launched or no
    // target_timesteps). Keeps the existing determinate label/width path byte-for-byte.
    function setProgress(mode, label) {
      if (mode === "indeterminate") progWrap.classList.add("indeterminate");
      else progWrap.classList.remove("indeterminate");
      if (label !== undefined) progLabel.textContent = label;
    }
    async function tick() {
      let run;
      try { run = await getJSON("/api/runs/" + encodeURIComponent(name)); } catch (e) { return; }
      renderControls(run);
      runActive = !!run.running;  // item 1: gate the 1 Hz elapsed ticker.
      const live = !!run.running;
      try {
        const snap = await getJSON("/api/runs/" + encodeURIComponent(name) + "/status");
        const latest = snap.latest;
        // item 1: reconcile the elapsed clock's base against every fresh server reading.
        if (latest && typeof latest.elapsed_seconds === "number" && Number.isFinite(latest.elapsed_seconds)) {
          elapsedBase = latest.elapsed_seconds;
          elapsedBaseAt = performance.now();
        }
        if (latest && snap.source === "jsonl") {
          const determinate = !!(latest.target_timesteps && latest.timesteps != null);
          if (determinate) {
            const pct = Math.min(100, 100 * latest.timesteps / latest.target_timesteps);
            progWrap.firstChild.style.width = pct + "%";
            // item 4 (prior): no "source:" indicator — just the progress line.
            setProgress("determinate", "step " + fmt(latest.timesteps) + " / " + fmt(latest.target_timesteps) +
              " · update " + fmt(latest.n_updates) + " · " + fmt(latest.fps, 0) + " fps");
          } else if (live) {
            // item 2: running but no target/steps to measure against → marquee instead of a dead bar.
            setProgress("indeterminate", "starting…");
          } else {
            setProgress("determinate", "no progress yet");
          }
          renderMetricsGrouped(latest);
        } else if (latest && snap.source === "csv") {
          // item 4 (prior): neutral label (the CSV fallback has no step counter); no "source:" wording.
          // item 2: the CSV fallback has no step target, so a live run shows the marquee.
          setProgress(live ? "indeterminate" : "determinate", "live metrics");
          renderMetricsCsv(latest);
        } else {
          // item 2: no data yet — marquee while launching, static once idle/stopped.
          setProgress(live ? "indeterminate" : "determinate", live ? "starting…" : "no progress yet");
        }
      } catch (e) {}
      await tickLogs();
    }
    async function tickLogs() {
      try {
        const lr = await getJSON("/api/runs/" + encodeURIComponent(name) + "/logs?since=" + logCursor);
        if (lr.lines && lr.lines.length) {
          // Auto-scroll only when the user is already pinned to the bottom (don't yank them up).
          const atBottom = logPanel.scrollTop + logPanel.clientHeight >= logPanel.scrollHeight - 4;
          lr.lines.forEach((line) => logPanel.appendChild(document.createTextNode(line + "\n")));
          if (atBottom) logPanel.scrollTop = logPanel.scrollHeight;
        }
        logCursor = lr.next;  // adopt the server's cursor verbatim (don't assume prev + length).
      } catch (e) {}
    }
    await tick();
    clearPoll();
    pollTimer = setInterval(tick, 2500);
    // item 1: advance the Elapsed box once a second (ETA stays on the 2.5 s poll cadence). Guards:
    // only while the run is active, only when a base reading exists, and only if the box is present
    // (suppressed on a just-launched run or in the CSV fallback → the ticker no-ops).
    elapsedTimer = setInterval(() => {
      if (!runActive || elapsedBase === null || elapsedBaseAt === null) return;
      const node = document.querySelector(".js-elapsed");
      if (!node) return;
      node.textContent = fmtDuration(elapsedBase + (performance.now() - elapsedBaseAt) / 1000);
    }, 1000);
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
    const saveBtn = h("button", { class: "primary" }, "Save");
    root.appendChild(runHead(name, "Config", [saveBtn]));  // item 2: Save top-right.
    const cfgCard = h("div", { class: "card" });
    cfgCard.appendChild(h("h3", null, "Configuration"));
    const cfgHost = h("div");
    cfgCard.appendChild(cfgHost);
    const saved = (await getJSON("/api/train-configs/" + encodeURIComponent(name))).config || {};
    const optionsByField = await fetchFieldOptions(fields);  // item 4: connectome/resume suggestions.
    const form = Forms.buildForm(cfgHost, fields.filter((f) => f.name !== "name"), saved, optionsByField);
    cfgHost.addEventListener("input", () => setDirty(true));
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
    const genBtn = h("button", { class: "primary" }, "Generate slice");
    root.appendChild(headBar("New slice", null, null, [genBtn]));  // item 2: primary action top-right.
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Slice (prune) configuration"));
    const host = h("div"); card.appendChild(host);
    root.appendChild(card);
    const optionsByField = await fetchFieldOptions(fields);  // item 4: connectome suggestions.
    const form = Forms.buildForm(host, fields, {}, optionsByField);
    host.addEventListener("input", () => setDirty(true));
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
    const genBtn = h("button", { class: "primary" }, "Regenerate slice");
    root.appendChild(headBar("Slice · " + name, null, null, [genBtn]));  // item 2: primary action top-right.
    const card = h("div", { class: "card" });
    card.appendChild(h("h3", null, "Slice (prune) configuration"));
    const host = h("div"); card.appendChild(host);
    root.appendChild(card);
    const saved = (await getJSON("/api/slice-configs/" + encodeURIComponent(name))).config || {};
    const optionsByField = await fetchFieldOptions(fields);  // item 4: connectome suggestions.
    const form = Forms.buildForm(host, fields, saved, optionsByField);
    host.addEventListener("input", () => setDirty(true));
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
    const saveBtn = h("button", { class: "primary" }, "Save");
    root.appendChild(headBar("Settings", null, null, [saveBtn]));  // item 2: Save top-right.
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

  // items 6+7: a static, offline About page (app name/purpose + version + author).
  async function renderAbout() {
    const root = view();
    root.innerHTML = "";
    root.appendChild(headBar("About", null, null));
    const card = h("div", { class: "card about-card" });
    card.innerHTML =
      '<h3>drone<span class="dot">·</span>fly desktop</h3>' +
      '<p>A lightweight, local, single-user desktop app for the drone-fly project: define ' +
      "connectome slices, configure and launch trainings, monitor live progress and logs, and " +
      "replay recorded episodes — a single window replacing the training TUI and the standalone " +
      "HTML recording viewer.</p>" +
      '<dl class="about-meta">' +
      "<dt>Version</dt><dd>0.1.0</dd>" +
      "<dt>Built by</dt><dd>Harold Hormaechea</dd>" +
      "</dl>";
    root.appendChild(card);
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

  // item 2: a view header with a right-aligned actions slot. `actionLabel`/`actionHash` render a
  // link button (e.g. "New training"); `actions` is an optional array of button nodes (e.g. the
  // primary Save) — the top-right Save convention across every screen.
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
  // item 5: instantly mark the nav link for `hash` active (before refreshNav rebuilds the tree), so
  // a click on a slow view gives immediate feedback. refreshNav() later reconciles authoritatively.
  function setActiveNav(hash) {
    const tree = document.getElementById("nav-tree");
    if (!tree) return;
    tree.querySelectorAll("[data-route]").forEach((a) => {
      const on = a.getAttribute("data-route") === hash;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  // item 5: map a parsed hash to a rough skeleton layout kind.
  function skeletonKind(parts) {
    if (parts[0] === "train" && parts[1] && parts[1] !== "new") {
      if (parts[2] === "recordings") return "recordings";
      if (parts[2] === "config") return "form";
      return "status";
    }
    if (parts[0] === "train" && parts[1] === "new") return "form";
    if (parts[0] === "train") return "list";
    if (parts[0] === "slices") return "form";
    if (parts[0] === "settings") return "settings";
    if (parts[0] === "about") return "about";
    return "list";
  }
  // item 5: paint greyed shimmer placeholders roughly matching the target view's layout. Whatever
  // the renderer paints next (root.innerHTML = "") replaces this; a fast view never shows it (the
  // 120 ms threshold timer is cleared first).
  function paintSkeleton(kind) {
    const block = (height) => '<div class="skel-block" style="height:' + height + '"></div>';
    const line = (width) => '<div class="skel-line" style="width:' + width + '"></div>';
    let body;
    if (kind === "list") body = block("40px") + block("40px") + block("40px");
    else if (kind === "status") body = block("18px") + block("150px") + block("240px");
    else if (kind === "recordings") body = block("42px") + block("360px");
    else if (kind === "settings") body = block("40px") + block("40px") + block("40px") + block("40px");
    else if (kind === "about") body = block("60px") + line("70%") + line("55%");
    else body = block("120px") + block("240px");  // "form"
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
    // item 4: fill mode is status-view only; clear it before dispatch so every other view lays out
    // normally (renderRunStatus re-adds it).
    view().classList.remove("view-fill");
    const parts = hash.replace(/^#\//, "").split("/");
    // item 5: instant nav highlight + a threshold skeleton (only shows if the view takes > 120 ms).
    setActiveNav(hash);
    const skelT = setTimeout(() => paintSkeleton(skeletonKind(parts)), 120);
    try {
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
      else if (parts[0] === "about") await renderAbout();
      else await renderTrainList();
    } catch (e) {
      view().innerHTML = '<p class="subtle">Error: ' + esc(e && e.message ? e.message : e) + "</p>";
    } finally {
      clearTimeout(skelT);  // item 5: fast view → no skeleton flash; redirect return → also cleared.
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
