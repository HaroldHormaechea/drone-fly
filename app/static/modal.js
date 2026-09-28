/* modal.js — a tiny accessible modal dialog for field-help popovers (UC-61 item 6).
 *
 * Vanilla, dependency-free, offline. openModal({title, bodyHTML, invoker}) shows a single-instance
 * dialog with:
 *   - role="dialog" aria-modal="true" aria-labelledby (WAI-ARIA dialog semantics),
 *   - a focus trap (Tab / Shift+Tab cycle within the dialog),
 *   - ESC and backdrop-click to close, an inline-SVG close button,
 *   - focus moved into the dialog on open and RESTORED to the invoking control on close.
 *
 * Loaded before app.js so `window.openModal` is ready when forms/settings wire their info buttons.
 */
(function (global) {
  "use strict";

  var current = null; // the single live modal instance ({overlay, invoker, onKeydown}) or null.

  var CLOSE_SVG =
    '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">' +
    '<path d="M6 6 L18 18 M18 6 L6 18" fill="none" stroke="currentColor" stroke-width="2" ' +
    'stroke-linecap="round"/></svg>';

  function focusable(root) {
    return Array.prototype.slice
      .call(
        root.querySelectorAll(
          'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])'
        )
      )
      .filter(function (n) {
        return n.offsetParent !== null || n === document.activeElement;
      });
  }

  function closeModal() {
    if (!current) return;
    var invoker = current.invoker;
    var resolve = current.resolve;
    var resolved = current.resolved;
    document.removeEventListener("keydown", current.onKeydown, true);
    if (current.overlay && current.overlay.parentNode) {
      current.overlay.parentNode.removeChild(current.overlay);
    }
    current = null;
    // Restore focus to the control that opened the dialog (accessibility requirement).
    if (invoker && typeof invoker.focus === "function") invoker.focus();
    // A confirm dialog dismissed without picking an action (ESC / backdrop / close) resolves null.
    if (resolve && !resolved) resolve(null);
  }

  function openModal(opts) {
    opts = opts || {};
    // Single-instance: replace any open dialog (do not stack).
    if (current) closeModal();

    var titleId = "modal-title";
    var overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    var dialog = document.createElement("div");
    dialog.className = "modal-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-labelledby", titleId);

    var head = document.createElement("div");
    head.className = "modal-head";
    var h = document.createElement("h3");
    h.className = "modal-title";
    h.id = titleId;
    h.textContent = opts.title || "";
    var closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "modal-close";
    closeBtn.setAttribute("aria-label", "Close");
    closeBtn.innerHTML = CLOSE_SVG;
    closeBtn.addEventListener("click", closeModal);
    head.appendChild(h);
    head.appendChild(closeBtn);

    var body = document.createElement("div");
    body.className = "modal-body";
    body.innerHTML = opts.bodyHTML || "";

    dialog.appendChild(head);
    dialog.appendChild(body);

    // Optional footer action buttons (UC-61 item 3: a confirm dialog). Each action is
    // {label, value, class?}. Clicking one resolves `choice` with its `value`; dismissing the dialog
    // any other way resolves null. Focus trap / ESC / backdrop all keep working (buttons are just
    // more focusable nodes inside the same dialog). `firstFocus` is the button focused on open.
    var resolveChoice = null;
    var choice = null;
    var firstFocus = closeBtn;
    if (opts.actions && opts.actions.length) {
      choice = new Promise(function (res) { resolveChoice = res; });
      var foot = document.createElement("div");
      foot.className = "modal-foot";
      opts.actions.forEach(function (a, i) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = a.class || "";
        b.textContent = a.label;
        b.addEventListener("click", function () {
          if (current) current.resolved = true;  // mark handled so closeModal won't resolve null.
          var r = resolveChoice;
          closeModal();
          if (r) r(a.value);
        });
        foot.appendChild(b);
        if (i === 0) firstFocus = b;  // focus the first action (place the safe/cancel action first).
      });
      dialog.appendChild(foot);
    }

    overlay.appendChild(dialog);

    overlay.addEventListener("mousedown", function (ev) {
      if (ev.target === overlay) closeModal(); // backdrop click (not a drag from inside)
    });

    function onKeydown(ev) {
      if (ev.key === "Escape") {
        ev.preventDefault();
        closeModal();
        return;
      }
      if (ev.key === "Tab") {
        var items = focusable(dialog);
        if (!items.length) {
          ev.preventDefault();
          return;
        }
        var first = items[0];
        var last = items[items.length - 1];
        if (ev.shiftKey && document.activeElement === first) {
          ev.preventDefault();
          last.focus();
        } else if (!ev.shiftKey && document.activeElement === last) {
          ev.preventDefault();
          first.focus();
        }
      }
    }

    document.body.appendChild(overlay);
    current = {
      overlay: overlay,
      invoker: opts.invoker || document.activeElement,
      onKeydown: onKeydown,
      resolve: resolveChoice,  // null for a plain info dialog (no actions).
      resolved: false,
    };
    document.addEventListener("keydown", onKeydown, true);
    // Move focus into the dialog (the first action, or the always-present close button).
    firstFocus.focus();
    // `body` + `choice` let a confirm caller read live form state (e.g. a checkbox) and await the
    // picked action; `close` is kept for the info-dialog callers that ignore the rest.
    return { close: closeModal, choice: choice, body: body };
  }

  global.openModal = openModal;
  global.closeModal = closeModal;
})(window);
