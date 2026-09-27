/* viewer-embed.js — embed the reused viz/viewer.html and drive it via window.loadDocument (AC8).
 *
 * The recording viewer (viz/viewer.{html,js,css}, UC-59/60) is reused UNCHANGED. It is served at
 * /viewer/ and embedded in an iframe. Because viewer.js is a classic script, its top-level
 * `function loadDocument(doc, name)` is exposed as a property on the iframe's window — so once the
 * iframe has loaded we fetch the recording (gunzipped SERVER-SIDE by the backend, returned as
 * parsed JSON) and call `iframe.contentWindow.loadDocument(parsed, name)`. No DecompressionStream,
 * no file:// picker, no edit to viewer.js.
 */
(function (global) {
  "use strict";

  function embedRecording(container, name, episode) {
    container.innerHTML = "";
    const iframe = document.createElement("iframe");
    iframe.className = "viewer-frame";
    iframe.src = "/viewer/viewer.html";
    container.appendChild(iframe);

    const label = name + " · episode " + episode;

    iframe.addEventListener("load", async () => {
      try {
        const res = await fetch(
          "/api/runs/" + encodeURIComponent(name) + "/recordings/" + encodeURIComponent(episode)
        );
        if (!res.ok) {
          throw new Error("recording fetch failed (" + res.status + ")");
        }
        const doc = await res.json();
        const w = iframe.contentWindow;
        if (!w || typeof w.loadDocument !== "function") {
          throw new Error("viewer did not expose loadDocument()");
        }
        w.loadDocument(doc, label);
      } catch (err) {
        container.innerHTML =
          '<p class="subtle">Could not load recording: ' + (err && err.message ? err.message : err) + "</p>";
      }
    });
    return iframe;
  }

  global.ViewerEmbed = { embedRecording };
})(window);
