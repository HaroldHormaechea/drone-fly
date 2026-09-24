#!/usr/bin/env node
// scripts/screenshot_viewer.mjs — DEV-ONLY headless-browser screenshot of the drone-fly viewer.
//
// UC-59 visual-verification helper (AC10). This file is NOT part of the shipped viewer, is NOT a
// runtime dependency of anything under viz/, and is NEVER imported by any pytest — the test gate
// stays fully hermetic. It renders viz/viewer.html with a small INLINE SYNTHETIC recording (no
// committed fixture) by calling `window.loadDocument(doc, name)`, then saves a PNG so a human can
// eyeball the three-zone layout (brain ‖ actions above the full-width 3D flight) and the animated
// soma-less tagged boxes.
//
// Browser resolution — degrade-honest chain:
//   1. a system Chromium/Chrome (via CHROME_PATH / PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH env or a
//      common install path), driven through the `playwright` package when it is importable;
//   2. Playwright's own bundled Chromium (after a one-time `npx playwright install chromium`);
//   3. otherwise print a clear "no headless browser available" message and exit non-zero (exit 3).
//
// In a headless CI sandbox with neither Chromium nor Playwright installed this HONESTLY hits (3) —
// that is expected. Run it locally where Chromium is installed to actually produce the PNG:
//   npm i -D playwright && npx playwright install chromium
//   node scripts/screenshot_viewer.mjs [out.png]

import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join, resolve } from "node:path";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const viewerHtml = join(repoRoot, "viz", "viewer.html");
const outPath = process.argv[2] ? resolve(process.argv[2]) : join(repoRoot, "viewer-screenshot.png");

const NO_BROWSER_EXIT = 3;

function findSystemChromium() {
  const candidates = [
    process.env.CHROME_PATH,
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/snap/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p)) || null;
}

async function loadPlaywright() {
  try {
    return await import("playwright");
  } catch {
    return null;
  }
}

// A minimal but schema-valid synthetic recording (schema_version 1) that exercises every zone: a
// few real-anatomy neurons + soma-less afferents spanning ALL modality tags including an untagged
// ("") and an unknown/legacy one (so the "other (untagged)" catch-all box appears), a short
// activation/flight timeline that visibly animates, and a tiny course. Everything is inline here —
// no committed fixture, no dependency on the recorder.
function syntheticDoc() {
  const nNeurons = 8;
  const nFrames = 24;
  const scale = 2 / 255;
  const offset = -1;
  const placement = [
    "anatomical", "anatomical", "schematic", "schematic",
    "schematic", "schematic", "schematic", "anatomical",
  ];
  const modality = ["vision", "", "vision", "proprioceptive", "hunger", "", "legacy", ""];
  const coords3d = placement.map((pl, i) =>
    pl === "anatomical" ? [Math.cos(i) * 30, Math.sin(i) * 30, (i - 4) * 8] : null,
  );
  const encode = (v) => Math.max(0, Math.min(255, Math.round((v - offset) / scale)));
  const activations = [];
  const actions = [];
  const dronePosition = [];
  for (let f = 0; f < nFrames; f++) {
    const row = [];
    for (let i = 0; i < nNeurons; i++) {
      row.push(encode(Math.sin((f / nFrames) * Math.PI * 2 + i) * 0.8)); // animate each neuron
    }
    activations.push(row);
    const t = f / (nFrames - 1);
    actions.push([Math.sin(t * 6), Math.cos(t * 5), Math.sin(t * 4), Math.cos(t * 3)]);
    dronePosition.push([t * 4, Math.sin(t * 3) * 0.5, 1 + t * 2]);
  }
  return {
    schema_version: 1,
    meta: {
      episode_index: 0,
      n_neurons: nNeurons,
      n_frames: nFrames,
      seed: 1,
      backend: "synthetic",
      dt: 0.05,
      activation_scale: scale,
      activation_offset: offset,
      action_layout: ["throttle", "roll", "pitch", "yaw"],
      modality,
      positions: {
        source: "anatomical (synthetic demo)",
        projection: "xz",
        has_position: placement.map(() => true),
        placement,
        region: placement.map((pl) => (pl === "schematic" ? "body" : "brain")),
        coords3d,
        coords2d: coords3d.map((c) => (c ? [c[0], c[2]] : null)),
        display3d: placement.map((pl, i) =>
          pl === "schematic" ? [Math.cos(i) * 80, Math.sin(i) * 80, (i - 4) * 10] : coords3d[i],
        ),
      },
      course: {
        floor_z: 0,
        ceiling_z: 4,
        start: [0, 0, 1],
        gates: [{ center: [2, 0, 2], aperture: 0.6 }],
        finish: { x: 4 },
      },
    },
    frames: {
      activations,
      actions,
      drone_position: dronePosition,
      target_gate: activations.map(() => 0),
    },
    outcome: { completed: true, completion_time: 1.2, total_reward: 3.4, steps: nFrames },
  };
}

async function main() {
  if (!existsSync(viewerHtml)) {
    console.error(`viewer HTML not found at ${viewerHtml}`);
    process.exit(1);
  }

  const playwright = await loadPlaywright();
  const sysChromium = findSystemChromium();

  if (!playwright) {
    console.error(
      "no headless browser available: the `playwright` package is not installed.\n" +
        "This dev-only helper needs a headless Chromium to render the viewer. Run it locally where\n" +
        "Chromium is installed:\n" +
        "  npm i -D playwright && npx playwright install chromium\n" +
        "  node scripts/screenshot_viewer.mjs [out.png]\n" +
        (sysChromium
          ? `(a system Chromium was found at ${sysChromium}, but the Playwright package is needed to drive it)\n`
          : ""),
    );
    process.exit(NO_BROWSER_EXIT);
  }

  const launchOpts = { headless: true };
  if (sysChromium) launchOpts.executablePath = sysChromium;

  let browser;
  try {
    browser = await playwright.chromium.launch(launchOpts);
  } catch (err) {
    console.error(
      "no headless browser available: Playwright is installed but could not launch Chromium " +
        `(${err.message}).\n` +
        "Install a browser with `npx playwright install chromium` (or set CHROME_PATH to a system\n" +
        "Chromium/Chrome) and retry.",
    );
    process.exit(NO_BROWSER_EXIT);
  }

  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 1400 },
      deviceScaleFactor: 2,
    });
    await page.goto(pathToFileURL(viewerHtml).href);
    await page.waitForFunction(() => typeof window.loadDocument === "function");
    await page.evaluate((doc) => window.loadDocument(doc, "synthetic-sample"), syntheticDoc());
    await page.waitForTimeout(400); // let the canvases paint (brain map, boxes, actions, 3D scene)
    await page.screenshot({ path: outPath, fullPage: true });
    console.log(`wrote ${outPath}`);
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
