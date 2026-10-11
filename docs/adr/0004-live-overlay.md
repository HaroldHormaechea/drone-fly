# ADR 0004: Live brain + debug overlay over the game

- **Status:** accepted (UC-65); live check (AC13) **pending user live run**
- **Date:** 2026-10-11
- **Builds on:** [ADR 0002](0002-live-harness-layering.md) (harness layering),
  [ADR 0003](0003-real-liftoff-calibration.md) (focus fail-safe, learned gates, HUD masks)
- **Code:** `src/drone_fly/harness/overlay.py` (publisher), `harness/policy.py`
  (`act_with_brain`), `harness/runner.py`, `harness/session.py`, `src/drone_fly/overlay/`
  (`protocol`, `state`, `receiver`, `replay`, `brain`, `panels`, `layout`, `win32`, `window`, `app`)

## Context

During a live run we want to see two things over Liftoff: the K1 brain activity, as the visualizer
shows it for recordings, and what the model is doing (its inputs, its actions and the sticks sent,
what perception detected, timings, fail-safe state).

Three harness facts constrain the design:

- **Focus.** The UC-64 runner fails safe the moment Liftoff stops being the foreground window, so
  the overlay must never take focus or input.
- **Capture.** The harness screen-captures the game for the gate detector and the depth model, so
  overlay pixels must never reach that capture.
- **Timing.** The 50 Hz loop must not block or jitter because of the overlay.

## Decision

### Process and transport

- **The overlay is a separate process** (`drone-fly harness overlay`). A GUI event loop and CPU
  rendering never run on, or contend with, the runner thread.
- **The transport is non-blocking localhost UDP**, latest-wins, dropping on error. The publisher's
  socket is non-blocking with a ~1 MB send buffer. Any send or serialisation error, including
  `BlockingIOError` when the buffer is full, drops that datagram, increments a counter and is
  logged at most once a second. It never raises into the loop. UDP to a port nobody listens on
  simply disappears. Shared memory was judged more complex for no prototype benefit.

### Wire schema (`overlay/protocol.py`, `SCHEMA_VERSION = 1`, default port 9100)

Text datagrams are JSON objects carrying `v`, `kind` and `run_id`:

| kind | when | body |
|---|---|---|
| `tick` | every tick | `TickRecord.as_dict()` (obs, extra, command, sink output, gate incl. `detections_px` / `frame_size` / `frame_index`, free space, `latency_s`, `perception_s`, `ages_s`, events) plus `run_state`, `run_info` (harness, checkpoint, sha256[:8], extra_dim), cumulative `missed_deadlines`, `drops`, `focus`, `focus_window`, `sample` (battery / stick inputs / RPM from the tick's `TelemetrySample`) |
| `meta` | first tick, then 1 Hz | extra-block layout, `MAX_BODY_RATE`, `MAX_THRUST`, `HOVER_THR`, `RAY_RANGE`, `CONE[:, :2]`, age limits (telemetry `stale_after_s`, free-space `max_estimate_age_s`, gate `lost_timeout_s`), `world_scale`, camera monitor / region / resolution, brain info (`n_neurons`, `tap_index`), K1 path |
| `status` | while waiting (≤ 4 Hz) | `run_state` = `waiting_for_state` / `waiting_for_focus` |
| `stop` | after the fail-safe (sent twice) | reason, error, ticks, missed deadlines, fail-safe flags and output |

The run state travels as `run_state` because the record's `state` key is the flight state. A tick
datagram carries the previous tick's overlay cost; the current tick's cost is only known after the
send. Ticks are about 5-8 KB.

The brain datagram is binary: magic `DFOB`, then a little-endian header (`version`, 16-character
`run_id`, `tick`, `n`, `scale`, `offset`, all floats as float32), then `n` uint8 codes. For K1,
n = 25,627, so a frame is about 25.6 KB and fits one datagram (`MAX_DATAGRAM` 60,000). Oversized
datagrams are dropped and counted.

The receiver keeps only the latest tick, brain frame, meta and status. Older packets are discarded,
never queued. Garbled datagrams decode to nothing and are counted.

### Decimation

A brain frame goes out every `overlay.brain_every_n` ticks; the default of 3 gives about 16 Hz. The
runner calls `policy.act_with_brain` only on those ticks. Every other tick, and every tick without
a publisher, calls the unchanged `policy.act`.

`act_with_brain` runs exactly the ops `inference.Policy.act` runs, in the same order and under
`no_grad`: `actor.propagated_full` → `actor.tap` → `actor.act`. It keeps the `propagated_full`
output instead of discarding it, so nothing is computed twice. The parity test pins the action
bit-identical. `gpu_prototype/inference.py` is untouched.

K1 is **feed-forward per tick**: `propagated_full` starts from zeros every call. The panel
therefore shows exactly what the readout's tap sees, not a hidden temporal state.

### Loop budget

The publisher's cost is measured into `latency_s["overlay"]`: serialise, plus the GPU→CPU copy of
the brain state on brain ticks, plus send. `inference` and `total` keep their definitions, but the
overlay cost still counts toward the missed-deadline check. `latency_summary.json` and
`harness latency-report` gain an `overlay` row only when ticks carry it, so runs without the
overlay keep the old summary shape.

### Brain quantisation: recorder scheme by default, opt-in `frame_peak`

AC2 mandates the recorder's `quantize_activation`: fixed `[-1, 1]`, `code = round((a+1)/(2/255))`.
Measured against real K1:

- **Steady loopback run** (sim-loopback, 70-dim model, obs replayed through K1 on CPU):
  - |a| p50 8e-6, p99 6.4e-4, max 0.027.
  - Rest (a = 0) is not representable: it falls on 127.5, so on code 127 or 128, and both decode
    to |a| = 0.0039 (78x the mean |a| of 0.00005).
  - **99.97 %** of neuron-ticks sat on codes 127 or 128 (histogram {124:1, 125:6, 126:38,
    127:248654, 128:263740, 129:98, 130:3}).
  - Only 0.035 % of neurons moved by 2 codes or more.
- **Existing flight recordings** (`training/10-proc-obstacle-avoid/recordings`, same scheme):
  - Better, because flight drives K1 harder: codes span 44-251.
  - Still, 97 % of entries are 127/128, and a neuron's code range over an episode has p50 = 1 and
    p99 = 7-10.

So in flight the panel looks like the visualizer does on those recordings: a few hundred to about
1,000 neurons light up. Sub-code variation (most neurons) is invisible, and a steady or hover scene
is mostly dark.

The decision:

- Keep `recorder` as the default (AC2-compliant).
- Add the opt-in `overlay.brain_scale: frame_peak`, a symmetric per-frame scale from the frame's own
  peak: `offset = -max|a|`, `scale = 2·max|a|/255`. The peak is floored at 1e-6 and NaN/inf count
  as 0, so there is no NaN or division by zero. No wire change is needed: every brain datagram
  carries its own scale/offset, and the receiver always dequantises from the header.
- On the loopback data, `frame_peak` cuts the mean magnitude error **~100x** (0.00388 → 0.00004).
- **Try `brain_scale: frame_peak` during the AC13 live check.**

### Brain panel "as in the visualizer" (`overlay/brain.py`)

What is reproduced from `viz/viewer.js` (UC-12/28/60):

- the recordings' neuron positions (`resolve_positions(persist=False)`, `display3d` projected on the
  `top` = xz plane);
- the registered outline (`viz/brain_outline.js`) fixing the uniform, aspect-preserving transform,
  with the `H - y` flip;
- UC-60 out-of-outline neurons in tagged boxes (vision / proprioceptive / hunger / other), with the
  same order and titles;
- splats weighted by magnitude from rest (`|a|`, so a resting neuron is dark), radius 7 buffer
  pixels at half resolution, sigma = R/2.4;
- the hot ramp with its soft alpha, and the skip of weights below 2/255.

The splat is a `bincount` followed by two separable `convolve1d` passes. This is exactly the
additive stamps, because `exp(-(dx²+dy²)/2σ²) = g(dx)·g(dy)`. It renders a 435x378 panel in about
3 ms.

**Live-gain deviation.** The viewer's per-neuron gain and global peak are computed over a whole
recorded episode, which is unavailable live. The overlay instead uses:

- a **running per-neuron `lo`/`hi`** that relaxes toward the current magnitude with
  `a = 1 - exp(-dt/gain_window_s)` (default 5 s);
- a **decaying running-peak** global gain.

The first frame is dark, and the picture is similar to, not identical with, an offline recording of
the same flight.

The viewer's 0.05 per-neuron range floor is applied **in quantisation steps of the frame**
(`0.05 · scale / (2/255)`). This is identical to the viewer for recorder frames. Without it,
`frame_peak` frames, whose real magnitudes are around 1e-3, would all sit under the floor and render
black.

The brain asset loads on the **CPU only**, with no CUDA use or device query. The overlay never
touches the GPU, which belongs to Liftoff, the depth model, the gate net and the policy. Rendering
is capped at `overlay.fps` ≤ 20 Hz with a half-resolution splat.

### Window behaviour (`overlay/win32.py`, `overlay/window.py`)

The Win32 calls go through an injectable shim, so the unit tests run on any OS. Steps:

1. The process is made per-monitor-DPI-aware (v2) **before** Qt starts.
2. `QT_ENABLE_HIGHDPI_SCALING=0`, so Qt coordinates are physical pixels and match game pixels. The
   device pixel ratio is still passed into the frame→screen mapping as a safety net.
3. The Qt window is Frameless | StaysOnTop | Tool | WindowTransparentForInput |
   WindowDoesNotAcceptFocus, with `WA_TranslucentBackground` / `WA_ShowWithoutActivating` /
   `WA_TransparentForMouseEvents`. Its geometry is set before `show()`.
4. On the native handle the overlay then:
   - ORs in `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE |
     WS_EX_TOOLWINDOW` and clears `WS_EX_APPWINDOW` (no taskbar or Alt-Tab entry, clicks pass
     through);
   - calls `SetWindowPos(HWND_TOPMOST, monitor rect, SWP_NOACTIVATE | SWP_FRAMECHANGED)`;
   - calls `ShowWindow(SW_SHOWNOACTIVATE)`;
   - calls `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`.
5. All of step 4 is re-asserted once after the first paint.
6. If the foreground window is ever the overlay, a loud warning is logged.

**No `SetLayeredWindowAttributes`.** Qt already renders a per-pixel-alpha layered window
(`UpdateLayeredWindow`). A colour key or constant alpha would switch the window to the other
layered mode and break per-pixel transparency.

**Toolkit: PySide6, not WebView2.** Qt gives reliable per-pixel transparency and native-handle
access, and QPainter plus NumPy images keep rendering on the CPU. A transparent, click-through,
non-activating WebView2 (pywebview, the desktop app's stack) is unreliable, and embedding a browser
engine costs GPU/CPU. PySide6-Essentials is in a new `overlay` extra, Windows-only, never in `dev`.
It is imported only by `overlay/window.py` and lazily via `lazy_import`, so CI imports and tests
every other overlay module. A missing extra gives exit code 3 with `uv sync --extra overlay`.

### Monitors, DPI and coordinates

- `--monitor` defaults to `camera.monitor` and uses the harness's **mss convention**: 1-based, and
  `0` = the union of all monitors. dxcam uses `output_idx = monitor - 1`.
- The overlay maps the index onto `EnumDisplayMonitors` order, which is the order mss enumerates.
  **Assumption:** DXGI output order (dxcam) matches it. On this machine there is one monitor
  (2560x1080 at 0,0), so this is verified only trivially and must be re-checked on multi-monitor
  setups.
- `camera.region` is taken as **monitor-relative**; null means the whole monitor. Note that mss
  treats a region as virtual-desktop absolute and dxcam as output-relative. They coincide on the
  primary monitor at the origin.
- A detected keypoint at frame pixel `p` maps to
  `region_origin + p · region_size / frame_size`, in window-local pixels.

### Capture exclusion and the fallback

- `WDA_EXCLUDEFROMCAPTURE` needs Windows 10 2004+. Whether the capture path in use honours it
  (dxcam Desktop Duplication vs mss GDI BitBlt) is verified only live (AC13 iii).
- If the call fails, the overlay logs a loud error and **switches automatically to the
  `mask_confined` layout**. Each panel greedily takes the largest remaining `camera.masks` region
  (the HUD regions the gate detector, auto-labeller and depth sampling already ignore), in priority
  order health > brain > action > telemetry > inputs > perception text. The ray fan and the
  in-game keypoints are disabled, and panels that do not fit are hidden and logged.
- With no masks configured, every panel is hidden rather than risk polluting perception.
- `--layout mask_confined` forces the fallback.

### Focus safety

- The overlay never takes focus, via the styles and flags above.
- The publisher never calls the focus probe. The runner hands it the runner's own per-tick result
  (and the last foreground process name), so the focus result and the probe calls are identical with
  and without the overlay (AC12).
- `--overlay-launch` starts the overlay with `CREATE_NO_WINDOW` **before** the runner's focus wait,
  so the wait still follows the launch. The process is never killed by the harness; it keeps showing
  the stop reason.
- Recommended start order: overlay → `harness run --overlay` → click into Liftoff.

### Borderless fullscreen requirement

DirectX exclusive-fullscreen games are not composited by DWM, so nothing can draw over them.
Liftoff must run borderless fullscreen or windowed. UC-64's capture setup already assumes a
composited desktop.

## Consequences

- One extra JSON serialisation of the tick record runs on the runner thread (estimated well under
  1 ms), plus a GPU→CPU copy of 25.6k floats (a CUDA sync) on brain ticks. Both are inside the
  measured `overlay` latency.
- Contingencies if they hurt: a larger `brain_every_n`, or an asynchronous copy into pinned memory
  on a worker thread (not built).
- `detections_px` adds about 1 KB per tick to `ticks.jsonl` recordings.
- The overlay detects a K1 mismatch with the harness (different neuron count) and shows it on the
  brain panel instead of rendering garbage.
- Out of scope: brain frames in run folders, hotkeys or dragging, the "minimal" layout and an
  `overlay.launch` config key.

## Build-time verification (developer, this machine; no game running)

- **Loopback publish** (real K1, 70-dim checkpoint, CPU, 100 ticks, `--overlay`): a UDP listener
  received 100 tick, 34 brain (n = 25,627), 20 meta and 2 stop datagrams, with 0 drops. The
  `overlay` latency row appears in `latency-report`.
- **Replay smoke-launch** (`harness overlay --harness sim-loopback --replay <run>`, 14 s):
  - The window covered (0, 0, 2560, 1080).
  - Native ex-style `0x080800a8`: LAYERED, TRANSPARENT, TOPMOST, NOACTIVATE, TOOLWINDOW set;
    APPWINDOW clear.
  - `GetWindowDisplayAffinity` = `0x11`.
  - The foreground window was unchanged throughout; the overlay was never foreground.
- **Offscreen render** of replay ticks plus synthetic brain frames: the default layout keeps the
  centre clear, the brain heatmap sits inside the outline with its tagged boxes, and the bar and
  ray-fan panels draw.

## AC13 live check (pending user live run)

Setup: Liftoff in borderless fullscreen on the capture monitor. Start `drone-fly harness overlay
--preset liftoff_micro_arena`, then run the UC-64 stage (b) configuration with `--overlay` (70-dim
model, depth free space). Optionally repeat with `overlay.brain_scale: frame_peak`.

Screenshots must be taken with a phone or an external capture: the overlay excludes itself from
screen capture, so a normal screenshot will not show it.

| item | check | result |
|---|---|---|
| (i) | overlay visible for the whole run; brain panel animates; action / inputs / health change every tick | pending user live run |
| (ii) | no `focus_lost` stop caused by the overlay; clicks and mouse over panels reach Liftoff / the desktop | pending user live run |
| (iii) | a frame saved by the harness's own frame source (`frames/`) contains no overlay pixels (or, if affinity failed, every overlay pixel lies inside the HUD masks) — per capture backend (dxcam / mss) | pending user live run |
| (iv) | overlay p99 ≤ 1 ms (`harness latency-report <run>`, `overlay` row), and `missed_deadline_rate` no more than +1 percentage point above a same-config baseline run **without** `--overlay` (compare both runs' `latency_summary.json`) | pending user live run |
| (v) | the overlay shows the final stop reason after F12 | pending user live run |

Capture-exclusion result per backend: pending user live run.
