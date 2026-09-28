/* forms.js — build config forms from backend field descriptors, honouring omit-vs-default.
 *
 * A field descriptor (from /api/train-configs/schema or /api/slice-schema) is:
 *   { name, base_type, nullable, required, always_resolves, default, choices, control }
 *
 * Emit rules (AC4/AC9 — "a key is emitted only when the user sets it"):
 *   - required            → always emitted.
 *   - always_resolves     → class (a): has a concrete default; emitted only when the value
 *                           DIFFERS from that default (set-to-default == omit → byte-identical run).
 *   - nullable (class b)  → emitted only when the user ticks its "set" box (number/text/select) or
 *                           chooses a concrete tri-state value (unset/true/false).
 *
 * buildForm(container, fields, values) renders the controls and returns { collect } where
 * collect() -> a plain mapping of exactly the keys that should be persisted.
 */
(function (global) {
  "use strict";

  // item 4: monotonic id source for the per-open-field <datalist> elements (unique within a build).
  let datalistSeq = 0;

  function coerce(baseType, raw) {
    if (baseType === "int") {
      const n = parseInt(raw, 10);
      return Number.isFinite(n) ? n : null;
    }
    if (baseType === "float") {
      const n = parseFloat(raw);
      return Number.isFinite(n) ? n : null;
    }
    if (baseType === "bool") return raw === true || raw === "true";
    return raw; // str
  }

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

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // item 6: the machine-checkable constraints block shown in a field's info modal.
  function constraintsHTML(f) {
    const rows = [["type", f.base_type], ["required", f.required ? "yes" : "no"]];
    if (f.nullable && !f.required) rows.push(["optional", "may be left unset"]);
    if (f.default !== null && f.default !== undefined) rows.push(["default", JSON.stringify(f.default)]);
    if (f.choices && f.choices.length) rows.push(["choices", f.choices.join(", ")]);
    // item 4: open-set fields advertise their static suggestions + that free text is allowed.
    if (f.options && f.options.length) rows.push(["suggestions", f.options.join(", ")]);
    if (f.open) rows.push(["free text", "yes — any value allowed"]);
    let html = '<dl class="mh-constraints">';
    rows.forEach(([k, v]) => { html += "<dt>" + esc(k) + "</dt><dd>" + esc(v) + "</dd>"; });
    return html + "</dl>";
  }

  function buildForm(container, fields, values, optionsByField) {
    container.innerHTML = "";
    values = values || {};
    // item 4: dynamic suggestion lists prefetched by the caller (fieldName -> [strings]); static
    // suggestions ride on the descriptor's own `options`. Absent → open fields just get no datalist.
    optionsByField = optionsByField || {};
    const rows = [];

    // item 2: fields carrying a `section` are wrapped into titled cards laid out in a responsive
    // grid (`.form-sections`); fields with no section render flat in a `.form-grid` (the prune
    // form, unchanged). Sections are created lazily in first-seen order — since the backend emits
    // fields in descriptor order and each section is a contiguous run, this preserves field order.
    let sectionsWrap = null, flatGrid = null;
    const sectionGrids = {};
    function flat() {
      if (!flatGrid) flatGrid = el("div", { class: "form-grid" });
      return flatGrid;
    }
    function sectionGrid(title) {
      if (!sectionsWrap) sectionsWrap = el("div", { class: "form-sections" });
      if (!sectionGrids[title]) {
        const g = el("div", { class: "form-grid" });
        const card = el("div", { class: "form-section" }, [
          el("h4", { class: "form-section-title", text: title }),
          g,
        ]);
        sectionsWrap.appendChild(card);
        sectionGrids[title] = g;
      }
      return sectionGrids[title];
    }

    fields.forEach((f) => {
      const present = Object.prototype.hasOwnProperty.call(values, f.name);
      const row = el("div", { class: "field" + (f.required ? " is-required" : "") });

      const keyLabel = el("label", { class: "key", text: f.name });
      const meta = el("span", { class: "meta" });
      meta.textContent =
        f.base_type +
        (f.nullable ? " · optional" : "") +
        (f.default !== null && f.default !== undefined ? " · default " + JSON.stringify(f.default) : "");
      keyLabel.appendChild(meta);
      // item 6: per-field "ⓘ" button → accessible modal (purpose + example + constraints).
      if (window.AppUI && window.AppUI.infoButton) {
        keyLabel.appendChild(window.AppUI.infoButton(f.name, f.help, f.example, constraintsHTML(f)));
      }

      const control = el("div", { class: "control" });
      let input, enableBox = null, extraNode = null;

      // "set" toggle for nullable non-bool fields (class b) so absence is meaningful.
      const needsEnable = f.nullable && f.control !== "tristate" && !f.required;

      if (f.control === "checkbox") {
        input = el("input", { type: "checkbox" });
        input.checked = present ? !!values[f.name] : !!f.default;
      } else if (f.control === "tristate") {
        input = el("select");
        [["", "(unset)"], ["true", "true"], ["false", "false"]].forEach(([v, t]) => {
          const o = el("option", { value: v, text: t });
          input.appendChild(o);
        });
        input.value = present ? String(!!values[f.name]) : "";
      } else if (f.control === "select") {
        input = el("select");
        if (f.nullable && !f.required) input.appendChild(el("option", { value: "", text: "(unset)" }));
        (f.choices || []).forEach((c) => input.appendChild(el("option", { value: String(c), text: String(c) })));
        if (present) input.value = String(values[f.name]);
        else if (f.default !== null && f.default !== undefined) input.value = String(f.default);
        else input.value = "";
      } else {
        input = el("input", { type: f.control === "number" ? "number" : "text" });
        if (f.base_type === "float") input.step = "any";
        if (present) input.value = values[f.name];
        else if (f.default !== null && f.default !== undefined) input.value = f.default;
        // item 4: open-set field (e.g. resume / connectome) → a <datalist> of suggestions backing a
        // plain text input. Free text is preserved, so collect() stays byte-identical; the
        // suggestions merely autocomplete. Static `f.options` + any dynamic prefetched values.
        if (f.open) {
          const suggestions = [];
          (f.options || []).forEach((o) => { if (suggestions.indexOf(String(o)) === -1) suggestions.push(String(o)); });
          (optionsByField[f.name] || []).forEach((o) => { if (suggestions.indexOf(String(o)) === -1) suggestions.push(String(o)); });
          if (suggestions.length) {
            const listId = "dl-" + f.name + "-" + (++datalistSeq);
            const dl = el("datalist", { id: listId });
            suggestions.forEach((o) => dl.appendChild(el("option", { value: o })));
            input.setAttribute("list", listId);
            extraNode = dl;
          }
        }
      }

      if (needsEnable) {
        enableBox = el("input", { type: "checkbox" });
        enableBox.checked = present;
        input.disabled = !present;
        const wrap = el("label", { class: "enable" }, [enableBox, document.createTextNode("set")]);
        control.appendChild(wrap);
        enableBox.addEventListener("change", () => { input.disabled = !enableBox.checked; markDirty(); });
      }
      control.appendChild(input);
      if (extraNode) control.appendChild(extraNode);  // item 4: the sibling <datalist>.
      row.appendChild(keyLabel);
      row.appendChild(control);
      (f.section ? sectionGrid(f.section) : flat()).appendChild(row);

      function markDirty() { row.classList.add("dirty"); }
      input.addEventListener("input", markDirty);
      input.addEventListener("change", markDirty);

      rows.push({ f, input, enableBox });
    });

    if (sectionsWrap) container.appendChild(sectionsWrap);
    if (flatGrid) container.appendChild(flatGrid);

    function collect() {
      const out = {};
      rows.forEach(({ f, input, enableBox }) => {
        if (f.control === "checkbox") {
          const v = input.checked;
          if (f.required || v !== f.default) out[f.name] = v;
          return;
        }
        if (f.control === "tristate") {
          if (input.value === "") return; // unset → omit
          out[f.name] = input.value === "true";
          return;
        }
        if (enableBox && !enableBox.checked) return; // nullable, not enabled → omit
        let raw = input.value;
        if (raw === "" && !f.required) return; // empty non-required → omit
        const v = coerce(f.base_type, raw);
        if (v === null) return;
        // class (a): omit when equal to the default (set-to-default == omit).
        if (f.always_resolves && !f.required && v === f.default) return;
        out[f.name] = v;
      });
      return out;
    }

    return { collect };
  }

  global.Forms = { buildForm };
})(window);
