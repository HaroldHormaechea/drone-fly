---
plan_for: use-cases/65-live-brain-debug-overlay.md
work_branch: feat/uc-65-live-brain-debug-overlay
run: drone-fly-uc-65
approved: 2026-10-11
---

# UC-65 implementation plan — live brain + debug overlay

Approved by challenger-2 (round 2 of 6, v2). This is the consolidated v1+v2 text. TARGET_DIR = the worktree, stacked on UC-64; the PR base is the UC-64 branch. Paths are relative to it. paths.production = src/drone_fly/**, paths.test = tests/**. No profiles are active.

## Analysis

- Runner: src/drone_fly/harness/runner.py.
  - `_tick` measures state, perception, inference (t2..t3: assemble + policy.act + decode_action) and actuation, and sets `total = t4 - t0`.
  - `_loop` checks the missed deadline after `_tick` returns, then records the tick.
  - `run()` latches the fail-safe in finally.
  - Telemetry samples are drained only when a recorder is present.
- Policy: harness/policy.py `ContractPolicyRunner.act` calls `self._policy.act`. gpu_prototype/inference.py:72-81 `Policy.act` = `actor.tap(actor.propagated_full(o))`, then `actor.act(feats, e)`, under no_grad, and returns `mean[0].cpu().numpy()`.
  - `K1ReservoirAug.propagated_full` (reservoir_aug.py:53) starts from zeros every call, so it is feed-forward, and each tanh hop keeps values in [-1, 1].
  - K1 N = 25,627 neurons, so a uint8 frame is about 25.6 KB.
- `quantize_activation` (record/recorder.py:73): importing that module pulls in torch. The runner must stay torch-free at import, because hermetic and static checks exist in tests/test_harness_registry_imports.py.
- Positions: record/coordinates.py `resolve_positions` (display3d, source). The outline is viz/brain_outline.js (`const BRAIN_OUTLINE = {json};`).
- Viewer method (viz/viewer.js):
  - MAP_SS=2, kernel R=7, sigma=R/2.4
  - weight = clamp01((|a|-lo)/max(hi-lo, 0.05)), skip below 2/255
  - hot colormap, default view "top" (xz)
  - UC-60: out-of-outline neurons go to modality boxes (vision / proprioceptive / hunger / other); non-anatomical data degrades to auto-fit with no outline and no boxes
- Config: harness/config.py rejects keys absent from the packaged default, so the overlay section must be added to configs/liftoff_5inch.yaml and configs/loopback.yaml.
- Gate keypoints: `TrackingGateEstimator.update` (liftoff/gate_cv.py:293-369) has the detection corners and pixel_center, but does not expose them in info.
- errors.py: `lazy_import` doesn't forward `hint`.
- Masks: liftoff/masks.py FrameMasks.rects. Focus: `_wait_for_focus` plus per-tick has_focus; `Win32ForegroundFocusProbe.last_foreground`.
- camera.monitor is the mss index: 1-based, with monitors[0] = the virtual desktop (frames.py:61). dxcam uses output_idx = monitor-1 (frames.py:41).

## Proposed Solution

### A. Brain state (AC1)
- harness/interfaces.py `PolicyRunner` gets non-abstract defaults:
  - `act_with_brain(obs, extra)` returns `(self.act(obs, extra), None)`.
  - `brain_info()` returns `{}`.
- harness/policy.py `ContractPolicyRunner.act_with_brain`:
  - Runs exactly the Policy.act augmented sequence inside torch.no_grad: obs tensor reshaped (1, -1) on policy.device, then actor.propagated_full, then actor.tap, then the extra tensor, then actor.act.
  - Returns `(mean[0].cpu().numpy() float32, full[0])`, with the full state left as a device tensor. torch is lazily imported inside the method.
  - `brain_info()` returns {n_neurons: actor.N, tap_index: actor.tap_index.cpu().tolist()}.
  - inference.py is untouched.
- runner.py calls act_with_brain only when a publisher is present and `publisher.wants_brain(k)`. Otherwise it keeps today's `self.policy.act` line unchanged.

### B. Publisher (AC2, AC3) — new harness/overlay.py (toolkit-free, torch-free at import)
- `OverlayPublisher(port, brain_every_n, run_info, meta, sock=None, quantize=None)`:
  - Non-blocking UDP socket aimed at 127.0.0.1:port, with SO_SNDBUF ≈ 1 MB (best effort, logged on failure). The socket is injectable.
  - `quantize_activation` is resolved at construction, by a function-level import.
  - A random hex run_id is created per run.
- `wants_brain(tick) = tick % brain_every_n == 0` (stateless, single source; default n=3).
- `publish_tick(record, *, brain, sample, missed_deadlines, focus, focus_window)`:
  - The JSON payload is `record.as_dict()` plus: v, kind="tick", run_id, state "running", run_info (harness, checkpoint name, sha256[:8], extra_dim), cumulative missed deadlines, drops, focus (bool or null), the focus window process name, and sample battery/inputs/rpm.
  - It sends the brain datagram iff `brain is not None`: tensor → .detach().float().cpu().numpy(), otherwise np.asarray; then quantize; then header + codes.
  - Any OSError (including BlockingIOError) or serialisation error drops the datagram, increments drops, logs (rate-limited) and never raises. Oversized payloads (over MAX_DATAGRAM) are dropped and counted.
- `publish_meta()`: on the first tick and then once per second. Carries:
  - the extra block layout (assembler.EXTRA_LAYOUT truncated to extra_dim)
  - MAX_BODY_RATE, MAX_THRUST, HOVER_THR
  - CONE[:, :2]
  - age limits (telemetry stale_after_s, free_space max_estimate_age_s, gate gates.lost_timeout_s)
  - world_scale
  - camera.region/resolution
  - brain_info and the K1 path
- `publish_status(state)`: "waiting_for_state" / "waiting_for_focus", rate-limited to 4 Hz, called from the runner's two wait loops.
- `publish_stop(result)`: reason, error, ticks, missed, failsafe, failsafe_ok, failsafe_output. Sent twice.
- `close()`.
- Pure helpers reused by replay: `tick_payload(...)` and `build_meta(...)`.
- runner.py changes:
  - New `publisher=None` constructor arg.
  - Drain telemetry when the recorder or the publisher is set, keeping `_last_sample`.
  - Keep the runner's own focus result; the publisher never calls the probe.
  - After t4: call publish_tick inside try/except Exception (never `_finish`), then `latency_s["overlay"] = t5 - t4`. inference and total keep their definitions; the overlay cost still counts toward the missed-deadline check.
  - `run()` finally: after `_failsafe`, call publish_stop (guarded). The wait loops call publish_status (guarded).
- recording.py LATENCY_STAGES gains ("overlay", "latency_s", "overlay").

### C. Protocol — new overlay/protocol.py
- SCHEMA_VERSION=1, DEFAULT_PORT=9100, MAX_DATAGRAM=60000.
- Text datagrams are JSON with v/kind/run_id.
- The brain datagram is: magic b"DFOB", version, run_id, tick, n, scale, offset (little-endian struct), then n uint8 codes.
- `decode_datagram` returns a Packet, or None for garbled input, and never raises.

### D. Config
- Add an `overlay:` section to configs/liftoff_5inch.yaml and configs/loopback.yaml, each key commented:
  - enabled: false
  - port: 9100
  - brain_every_n: 3
  - layout: default
  - fps: 20
  - stale_s: 1.0
  - gain_window_s: 5.0
  - opacity: 0.85
  - tap_outline: false
  - panels: {brain, action, inputs, raycast, perception, health, telemetry: true}
- config.py `OverlayConfig.from_config(cfg)` returns defaults when the section is absent and validates:
  - port: int 1024..65535 and != telemetry.port
  - brain_every_n ≥ 1
  - fps in 1..20
  - stale_s > 0, gain_window_s > 0
  - opacity in 0..1
  - layout in {default, mask_confined}
  - panels: bools

### E. Session + CLI
- session.py `run_harness(..., overlay=None, overlay_launch=False)`:
  - `overlay` maps to the override `overlay.enabled`. When enabled, it builds the publisher after prepare(arm=False) and passes it to the runner.
  - With `overlay_launch` (CLI flag only, no config key), an injectable launcher is called BEFORE runner.run, so the focus wait follows the launch.
  - The default launcher is one function: subprocess.Popen of `sys.executable -m drone_fly.cli.main harness overlay --harness H --port N [--preset/--config]`, with creationflags=CREATE_NO_WINDOW on win32 and 0 elsewhere. It is never killed.
  - A helper `start_overlay_and_run(runner, launcher)` serves AC12. The publisher is closed in the finally.
- cli/main.py:
  - `harness run` gains `--overlay` and `--overlay-launch`.
  - New `harness overlay` subcommand: `--port N`, `--monitor M`, `--layout NAME`, `--replay RUN_DIR`, `--fps`, `--k1-path`, plus `_common` (--preset/--config/--harness, default liftoff). Defaults come from the merged config.
  - A missing PySide6 raises HarnessDependencyError with the hint "uv sync --extra overlay", which gives exit 3.
- errors.py: `lazy_import(..., hint=INSTALL_HINT)`, plus `OVERLAY_INSTALL_HINT`.

### F. Keypoints (AC8d)
- gate_cv.py `TrackingGateEstimator.update` adds to info, when a frame was processed:
  - `detections_px` = [{cls, points (frame px), center, matched}]
  - `frame_size` [w, h]
  - `frame_index`
- Both estimators inherit it, and tracking is unchanged.

### G. Overlay process — new package src/drone_fly/overlay/
Only window.py imports PySide6; torch/record imports happen only inside the brain-asset loader.

- `state.py` `OverlayState` (clock-injected, locked):
  - Holds only the latest tick/brain/meta/status.
  - Modes: waiting ("waiting for harness"), stale (no packet for more than stale_s: "no data for N s"), live, stopped.
  - last_stop persists until a new run_id arrives.
  - Tick rate is an EWMA.
- `receiver.py` `UdpReceiver(port, state, sock=None)`:
  - A thread on 127.0.0.1:port with SO_RCVBUF 4 MB that drains every pending datagram into state.
  - A bind failure raises HarnessConfigError "port N in use".
- `replay.py` `ReplaySource(run_dir)`:
  - Reads meta.json, config.yaml and ticks.jsonl.
  - Yields (delay, Packet): a synthesized meta, then ticks via tick_payload, then a stop from meta.json.
  - Paced by wall_time deltas, and iterable without sleeping.
  - The brain panel shows "no brain data (replay)".
- `brain.py`:
  - `load_brain_asset(k1_path, view="top")` covers:
    - the connectome loader plus `resolve_positions(persist=False)`
    - modality tags via RECORDED_MODALITIES + select_modality, fail-soft
    - viz/brain_outline.js parsed via find_repo_root
    - CPU/numpy only: no CUDA use or device query (stated in the docstring and the ADR)
  - `BrainRenderer` precomputes:
    - the partition (vectorised point-in-polygon mirroring viewer.js partitionByBoundary)
    - the outline-bbox uniform transform with the y-flip
    - buffer coordinates at size/2
    - the separable Gaussian
    - the box layouts
  - Per frame:
    - dequantise; mag = |a| (rest = 0)
    - running per-neuron lo/hi with a = 1 - exp(-dt/gain_window_s)
    - weight = clamp01((mag - lo)/max(hi - lo, 0.05)), dropping weights below 2/255
    - bincount plus a separable scipy convolve1d (equal to the additive stamps)
    - decaying running-peak global gain, then the viewer's hot ramp, then RGBA upscaled 2x, plus box mini-heatmaps, the outline polyline and optional tap points
  - The live-gain deviation is documented in-source. When n != asset size it reports "neuron count mismatch".
- `panels.py`: pure dataclass models:
  - Action: raw bars, collective with the HOVER_THR marker, rates vs MAX_BODY_RATE, sink sticks.
  - Inputs: obs as rel/g_body/vel/omega; extra by contract block, as signed bars.
  - Raycast: 40 rays at disk coordinates, grey when out of FOV. Geometry comes from meta CONE, otherwise from `contract_cone_xy()`, a tiny numpy mirror of inference.py _build_cone (documented as a mirror).
  - Perception: status, idx/lap, class, range (training and × world_scale), rel, frame age; keypoints mapped to screen.
  - Health: tick rate; stacked state/gate/free_space/inference/actuation/overlay; missed deadlines; age ok/amber (≥ 0.8 × limit)/red (> limit); focus + window; stop reason + failsafe; drops.
  - Telemetry: speed/alt from state × world_scale; roll/pitch/yaw from the quat; battery/inputs from the sample, or n/a.
  - `frame_to_screen(points, frame_size, region, monitor_rect, dpr)`: region monitor-relative, null = full monitor.
- `layout.py`:
  - Layouts are "default" (left column x 0.005-0.175 with action/brain/inputs; right column x 0.825-0.995 with perception text/health/telemetry/raycast; centre clear) and "mask_confined" (forced fallback, selectable). `overlay.panels` toggles apply.
  - `decide_capture_mode(affinity_ok, masks_present)` returns:
    - "excluded" (normal layout)
    - "mask_confined" (affinity failed, masks configured)
    - "hidden" (affinity failed, no masks: all panels off plus a loud warning)
  - `mask_confined_layout`: greedy, priority health > brain > action > telemetry > inputs > perception-text into the largest masks (FrameMasks.rects mapped via frame_to_screen). The ray fan and in-game keypoints are disabled; panels that don't fit are hidden and logged.
- `win32.py`:
  - Constants: WS_EX_LAYERED 0x80000, TRANSPARENT 0x20, TOPMOST 0x8, NOACTIVATE 0x08000000, TOOLWINDOW 0x80, APPWINDOW 0x40000 (cleared), GWL_EXSTYLE -20, WDA_EXCLUDEFROMCAPTURE 0x11, SW_SHOWNOACTIVATE 4, HWND_TOPMOST -1, PMv2 -4.
  - A `Win32Shim` protocol: set_dpi_awareness, monitor_rects (EnumDisplayMonitors = the order mss uses), get/set_ex_style, set_window_pos, show_no_activate, set_display_affinity → bool, foreground_hwnd.
  - `CtypesWin32` is lazy and win32-only, with a clear error elsewhere.
  - `prepare_process(shim)` sets DPI awareness before the GUI app.
  - `monitor_rect(rects, m)`:
    - m ≥ 1 → rects[m-1]
    - m == 0 → the union bounding rect
    - out of range, bool or non-int → HarnessConfigError
    - `--monitor` defaults to camera.monitor with the same mss convention
  - `configure_overlay_window(hwnd, shim, rect)` returns WindowSetup(ex_style, affinity_ok). Steps: OR the five styles and clear APPWINDOW; SetWindowPos TOPMOST over the rect with NOACTIVATE; SW_SHOWNOACTIVATE; SetWindowDisplayAffinity.
  - No SetLayeredWindowAttributes.
- `window.py` (the only PySide6 importer):
  - QT_ENABLE_HIGHDPI_SCALING=0 before QApplication (dpr still passed as a safety net).
  - Flags Frameless | StaysOnTop | Tool | WindowTransparentForInput | WindowDoesNotAcceptFocus; WA_TranslucentBackground / ShowWithoutActivating / TransparentForMouseEvents.
  - setGeometry(rect) before show(), then configure_overlay_window, then re-assert once after the first paint.
  - A QTimer at fps ≤ 20 rebuilds the models; QPainter draws them, with the brain as a QImage (reference kept).
  - A loud log warning if the foreground hwnd == own hwnd.
- `app.py` `run_overlay(...)`:
  - Merged config plus OverlayConfig plus CLI overrides.
  - K1 via contract.resolve_k1_path. A failed brain load gives a "no brain asset" panel.
  - Builds the state plus the receiver/replay.
  - `lazy_import("PySide6", package="PySide6-Essentials", hint=OVERLAY_INSTALL_HINT)` → window → Qt loop.
- pyproject.toml: extra `overlay = ["PySide6-Essentials; sys_platform == 'win32'"]`, not in dev. Regenerate uv.lock; fall back to PySide6 if needed.

### H. Docs (AC14)
- docs/adr/0004-live-overlay.md covers:
  - the process/UDP design, schema and decimation (≈ 50/n Hz)
  - styles/affinity, and why no SetLayeredWindowAttributes
  - PySide6 vs WebView2
  - the live-gain deviation
  - the borderless-fullscreen requirement
  - the monitor convention (mss index; dxcam monitor-1; DXGI order assumed = EnumDisplayMonitors) and DPI
  - CPU-only overlay
  - the capture-exclusion result and the AC13 table, with items (i)-(v) pending (user live run)
  - the AC13(iv) baseline = a same-config run without --overlay, compared via missed_deadline_rate (+1 pp) and the overlay p99 ≤ 1 ms via latency-report
- README § Live harness gains an "Overlay" subsection: commands, --extra overlay, start order (overlay → run --overlay → click Liftoff), the sim-loopback dev path, replay, fallback and the AC13(iv) comparison.

## Files Affected

### Production (developer)
- src/drone_fly/harness/{interfaces,policy,runner,session,config,errors,recording}.py, harness/liftoff/gate_cv.py
- harness/overlay.py (new)
- harness/configs/{liftoff_5inch,loopback}.yaml
- src/drone_fly/cli/main.py
- src/drone_fly/overlay/{__init__,protocol,state,receiver,replay,brain,panels,layout,win32,window,app}.py (new)
- pyproject.toml, uv.lock
- docs/adr/0004-live-overlay.md, README.md

### Test (qa)
- tests/test_overlay_policy_brain.py — AC1: bit-identical parity vs contract.inference.Policy with a stub actor and the real harness_checkpoint_70; brain == propagated_full; with no publisher, act is called and act_with_brain never is.
- tests/test_overlay_publisher.py — AC2/AC3:
  - every AC2 field present
  - brain on ticks 0, 3, 6…
  - brain round-trip within one code
  - drop on BlockingIOError/OSError
  - a raising publisher → the run completes
  - stop packet
  - the SNDBUF request
  - SimClock latency_s["overlay"], inference unchanged, and the overlay row in summarize_ticks
  - sample forwarded
- tests/test_overlay_config.py: section in both defaults; port == telemetry rejected; fps > 20 rejected; unknown key rejected; the --overlay override works.
- tests/test_overlay_state.py — AC4: latest-wins, waiting/stale/live, last stop kept until a new run_id, garbled datagram ignored, receiver with a fake socket.
- tests/test_overlay_win32.py — AC5/AC6:
  - FakeShim: exact style bits, APPWINDOW cleared, SHOWNOACTIVATE, TOPMOST+NOACTIVATE, affinity 0x11, DPI first
  - monitor_rect for 0/1/2/3/-1 with a non-zero-origin second monitor; the selected rect reaches configure_overlay_window and frame_to_screen
  - decide_capture_mode for 3 cases
  - mask_confined_layout panels inside masks with drawings disabled
- tests/test_overlay_brain.py — AC7:
  - a hot spot at a known neuron; resting = dark
  - an out-of-outline neuron goes to its box
  - gain decay; count mismatch
  - the fixture loader (tests/fixtures/mcns_fixture.npz)
  - torch.cuda.is_initialized() False after the load, guarded with importorskip
- tests/test_overlay_panels.py — AC8: the models, frame_to_screen with region/dpr, amber/red, toggles/centre clear, `contract_cone_xy()` allclose to harness_contract.CONE[:, :2].
- tests/test_overlay_replay.py + tests/data/overlay/run/{ticks.jsonl, meta.json, config.yaml} (tiny, synthetic) — AC9.
- tests/test_overlay_hermetic.py — AC10/AC11:
  - hermetic import of the overlay modules except window, plus harness.overlay, with torch + PySide6 blocked
  - only window.py imports PySide6
  - `harness overlay` without PySide6 → exit 3 and "--extra overlay"
  - PySide6 only in the overlay extra
  - add "PySide6" to tests/harness_support.py HEAVY_MODULES
- tests/test_overlay_focus.py — AC12: ScriptedFocus + FakeLauncher through start_overlay_and_run: launch → focus wait → ticks; focus calls/result identical to a run without overlay; the publisher never calls the probe.
- tests/test_harness_gate_cv.py (extend): detections_px/frame_size/frame_index.

## Risks & Considerations
- **Focus theft.** Mitigations: NOACTIVATE + SHOWNOACTIVATE + Qt no-focus flags + TRANSPARENT; re-assert after the first paint; foreground diagnostic; CREATE_NO_WINDOW launch before the focus wait. README start order; AC12 + AC13(ii).
- **Capture pollution.** Affinity with dxcam vs mss and with layered windows is verified only live (AC13(iii)). Automatic mask_confined/hidden fallback; `--layout mask_confined` forces it.
- **Exclusive fullscreen.** A documented requirement.
- **Multi-monitor / DPI.** The monitor convention above; DXGI order is an ADR assumption verified live; PMv2 + Qt scaling off; region monitor-relative.
- **Feed-forward brain.** Confirmed.
- **Live gain.** It differs from the visualizer; documented.
- **GPU→CPU copy.** Only on brain ticks, inside the measured overlay latency. Contingencies: a larger brain_every_n, or a worker thread with pinned memory (not built).
- **Machine load.** fps ≤ 20, half-res splat, CPU-only overlay.
- **Port collision.** Validated against telemetry.port; a clear bind error.
- **Concurrent work.** Handled by stacking; nothing under training/, gpu_prototype/, docs/INFERENCE_CONTRACT.md, training configs or artifacts changes (AC15).
- **Wire cost.** Tick JSON is about 5-8 KB; the second as_dict() on the runner thread is estimated under 1 ms and measured by AC3. detections_px adds about 1 KB/tick to recordings.
- **uv.lock** needs network to regenerate.
- **K1 mismatch** between the overlay and the harness is detected and shown.
- **Out of scope.** Brain frames in run folders; hotkeys/dragging; the "minimal" layout and the overlay.launch config key were dropped.

## Challenger verdict
Approved (challenger-2, round 2): the v2 monitor_rect mapping resolves the only Major issue; all 7 minors addressed; all 15 ACs map to code + tests.

## Orchestrator notes
- User away; proceeding without a plan-approval gate as instructed ("Tackle this autonomously after the current UC").
- AC13 live check is left for the user; AC3's live limits are measured then.
