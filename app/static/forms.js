/* forms.js — the reservoir-run form (UC-63 AC2), built from GET /api/schema.
 *
 * A run is a recipe + env flags + train_aug knobs + a .pt save path. The schema says, per recipe,
 * which knobs are editable; a named launcher (exp11/12/13, train_shaping) hardcodes everything but
 * the budget, so its other knobs render read-only as "set by <launcher>" with the fixed value. The
 * generic train_aug recipe exposes every knob; a blank knob is not forwarded (train_aug's own
 * default applies), and the placeholder shows that default.
 *
 * buildRunForm(container, schema, values, onChange) renders the controls and returns { collect },
 * where collect() -> { recipe, flags, knobs, save_path } with only the knobs the user set. Values
 * live in the form until the caller's explicit Save (no auto-save, UC-61 AC9).
 */
(function (global) {
  "use strict";

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") node.className = attrs[k];
      else if (k === "text") node.textContent = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    (children || []).forEach((c) => node.appendChild(c));
    return node;
  }

  function fmtValue(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number" && Number.isInteger(v)) return v.toLocaleString("en-US");
    return String(v);
  }

  function info(title, help, extraHTML) {
    if (window.AppUI && window.AppUI.infoButton) return window.AppUI.infoButton(title, help, null, extraHTML || null);
    return document.createTextNode("");
  }

  function fieldRow(label, meta, control, help, required) {
    const row = el("div", { class: "field" + (required ? " is-required" : "") });
    const key = el("label", { class: "key", text: label });
    if (meta) key.appendChild(el("span", { class: "meta", text: meta }));
    if (help) key.appendChild(info(label, help));
    row.appendChild(key);
    const ctrl = el("div", { class: "control" });
    ctrl.appendChild(control);
    row.appendChild(ctrl);
    return row;
  }

  function buildRunForm(container, schema, values, onChange) {
    container.innerHTML = "";
    values = values || {};
    const changed = () => { if (onChange) onChange(); };
    const recipes = {};
    schema.recipes.forEach((r) => { recipes[r.name] = r; });
    const knobDefs = {};
    schema.knobs.forEach((k) => { knobDefs[k.name] = k; });

    const wrap = el("div", { class: "form-sections" });
    container.appendChild(wrap);

    // -- recipe + save path --
    const runGrid = el("div", { class: "form-grid" });
    wrap.appendChild(el("div", { class: "form-section" }, [el("h4", { class: "form-section-title", text: "Recipe" }), runGrid]));
    const recipeSel = el("select");
    schema.recipes.forEach((r) => recipeSel.appendChild(el("option", { value: r.name, text: r.name + (r.generic ? " (generic)" : "") })));
    recipeSel.value = values.recipe && recipes[values.recipe] ? values.recipe : schema.default_recipe;
    runGrid.appendChild(fieldRow("recipe", null, recipeSel,
      "Named launchers hardcode every knob except the budget; the generic train_aug recipe forwards any knob you set.", true));
    const launcherNote = el("p", { class: "subtle form-note" });
    runGrid.appendChild(launcherNote);
    const savePath = el("input", { type: "text", placeholder: "training/<name>/model.pt" });
    savePath.value = values.save_path || "";
    savePath.addEventListener("input", changed);
    runGrid.appendChild(fieldRow("save_path", ".pt · blank = training/<name>/model.pt", savePath,
      "Where train_aug writes the checkpoint (every 200 iterations and at the end). Must be inside the project root and not inside a committed dir."));

    // -- flags --
    const flagGrid = el("div", { class: "form-grid" });
    wrap.appendChild(el("div", { class: "form-section" }, [el("h4", { class: "form-section-title", text: "Env flags" }), flagGrid]));
    const flagBoxes = {};
    schema.flags.forEach((f) => {
      const cb = el("input", { type: "checkbox" });
      cb.addEventListener("change", changed);
      flagBoxes[f.name] = cb;
      flagGrid.appendChild(fieldRow(f.name, "set to \"1\" when ticked; absent otherwise", cb, f.help));
    });

    // -- knobs (re-rendered when the recipe changes) --
    const knobGrid = el("div", { class: "form-grid" });
    wrap.appendChild(el("div", { class: "form-section" }, [el("h4", { class: "form-section-title", text: "train_aug knobs" }), knobGrid]));
    let knobInputs = {};

    function renderKnobs(recipe, knobValues) {
      knobGrid.innerHTML = "";
      knobInputs = {};
      const spec = recipes[recipe];
      launcherNote.textContent = "Runs " + (spec.generic ? spec.launcher : "gpu_prototype/" + spec.launcher) +
        (spec.generic ? " — only the knobs you set are forwarded." : " <save_path> <budget> — other knobs are fixed by the launcher.");
      spec.knobs.forEach((k) => {
        const def = knobDefs[k.name];
        if (!k.editable) {
          const v = el("span", { class: "fixed-value", text: fmtValue(k.fixed_value) });
          knobGrid.appendChild(fieldRow(k.name, "set by " + spec.launcher + " (read-only)", v, def.help));
          return;
        }
        const input = el("input", { type: def.type === "str" ? "text" : "number" });
        if (def.type !== "int" && def.type !== "str") input.step = "any";
        const dflt = k.default !== undefined ? k.default : def.default;
        if (dflt !== null && dflt !== undefined) input.placeholder = String(dflt);
        const v = knobValues ? knobValues[k.name] : undefined;
        if (v !== undefined && v !== null) input.value = v;
        input.addEventListener("input", () => { changed(); syncWarmSrc(); });
        knobInputs[k.name] = { input, def };
        const meta = def.type + (def.nullable ? " · optional" : "") +
          (dflt !== null && dflt !== undefined ? " · blank = " + fmtValue(dflt) : " · blank = not set");
        knobGrid.appendChild(fieldRow(k.name, meta, input, def.help));
      });
      syncWarmSrc();
    }
    // warm_src_extra_dim only makes sense when warm_from is set.
    function syncWarmSrc() {
      const warm = knobInputs.warm_from, src = knobInputs.warm_src_extra_dim;
      if (!warm || !src) return;
      const on = warm.input.value.trim() !== "";
      src.input.disabled = !on;
      if (!on) src.input.value = "";
    }

    function applyRecipeDefaults(recipe) {
      const spec = recipes[recipe];
      Object.keys(flagBoxes).forEach((f) => { flagBoxes[f].checked = spec.default_flags.indexOf(f) !== -1; });
    }

    if (values.flags) Object.keys(flagBoxes).forEach((f) => { flagBoxes[f].checked = values.flags.indexOf(f) !== -1; });
    else applyRecipeDefaults(recipeSel.value);
    renderKnobs(recipeSel.value, values.knobs || {});

    recipeSel.addEventListener("change", () => {
      applyRecipeDefaults(recipeSel.value);
      renderKnobs(recipeSel.value, {});
      changed();
    });

    function collect() {
      const knobs = {};
      Object.keys(knobInputs).forEach((name) => {
        const { input, def } = knobInputs[name];
        const raw = input.value.trim();
        if (raw === "" || input.disabled) return;
        if (def.type === "int") knobs[name] = parseInt(raw, 10);
        else if (def.type === "str") knobs[name] = raw;
        else knobs[name] = parseFloat(raw);
      });
      return {
        recipe: recipeSel.value,
        flags: Object.keys(flagBoxes).filter((f) => flagBoxes[f].checked),
        knobs,
        save_path: savePath.value.trim() || null,
      };
    }

    return { collect };
  }

  global.Forms = { buildRunForm };
})(window);
