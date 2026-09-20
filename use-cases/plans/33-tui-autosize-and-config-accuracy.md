---
plan_for: use-cases/33-tui-autosize-and-config-accuracy.md
work_branch: feat/uc-33-tui-autosize-and-config-accuracy
team: drone-fly-uc-33
approved: 2026-09-20
---

# UC-33 — TUI status-box autosize + Windows full-screen/resize + accurate OOM & test-command guidance

Analyst↔challenger peer loop complete in **1 round — challenger APPROVED** (no revision needed). All code claims verified against the worktree.

**Orchestrator authorization (PROJECT_BRIEF.md):** `PROJECT_BRIEF.md` is outside the developer's
authorized paths. **The developer is authorized for this run to make the Item-3 two-line
test-command correction** in `PROJECT_BRIEF.md` (frontmatter `build.commands.test` line ~14 +
prose line ~287): `uv run pytest` → `uv run --extra dev pytest`. No other brief edits.

## Decisions recorded (challenger)
- **Item 2 → OPTION (b): correct the message only** (no batch_size/n_steps exposure). Zero training-dynamics risk; prune_k (the ~122k-neuron slice) is the real VRAM lever and already validated.
- **Item 1 → design (ii): width-aware autosize.** `HealthVerdict.message` is always a single logical line (`_summarize` never emits `\n`), so the real clip is line-WRAPPING, not embedded newlines.
- **Item 3 → NO archive sweep.** Canonical docs only: `PROJECT_BRIEF.md` + `README.md` + `allowed-commands.yaml` comment. `ci.yml` left as-is (AC-12).

## Scope
Four self-contained papercuts: UI-layout + messaging + doc accuracy only. No change to training dynamics, obs schema, connectome, reward, checkpoints, or env/adapter (AC-14). Existing configs byte-for-byte unchanged.

## Item 1 — TUI status-box autosize (render.py)
`src/drone_fly/train/tui/render.py` — today `build_layout` hardcodes `Layout(name="status", size=3)`.
- Add documented constants: `STATUS_MAX_CONTENT_LINES` (small cap, e.g. 6) and `STATUS_BORDER_OVERHEAD` (=2; current size=3 = 1 content + 2 overhead, so a 1-line verdict stays visually identical → AC-3).
- Dedup the verdict-message extraction (logic in `build_status_bar`) into ONE private helper used by both `build_status_bar` and the sizing fn so they never drift.
- Add pure `status_region_size(message, width=None) -> int`: split message on `\n`; if `width` is a positive int, count WRAPPED rows per logical line against `inner_width = width − 4` (border 2 + Rich default padding (0,1)=2); else 1 row per logical line. Sum (min 1), return `min(content_lines, MAX) + STATUS_BORDER_OVERHEAD`.
- **Challenger rec (fold in):** measure wrapped rows via Rich itself for the given width (word-aware, deterministic given a fixed width int) rather than `ceil(len/inner_width)` char-division, which under-counts word-wrapped text and could re-clip the last line. If char-division kept, round conservatively.
- `build_layout` gains `status_width: int|None=None`, uses `size=status_region_size(msg, status_width)`. Default None → logical-line count → size 3 → every existing render test stays byte-compatible.
- Graceful truncation (AC-2): capped fixed region clips extra rows to first N; no message mutation.
- Dashboard wiring (`dashboard.py`): both `build_layout(...)` call sites (`_start` initial Live; `_redraw_locked`) pass `status_width=self._console.width if self._console else None`.
- AC-5: "platform-independent" = the sizing fn is deterministic given (message, width), NOT identical runtime widths across OSes. Test with fixed width ints.

## Item 2 — Accurate OOM guidance, OPTION (b) (device.py)
`src/drone_fly/train/device.py` — rewrite `CUDA_OOM_HINT` (logged on auto-CUDA and explicit `device='cuda'` in `resolve_device`) to name ONLY existing levers: `n_envs`, a smaller pruned slice (`prune`/`prune_k`), and `device: cpu`. Must mention NEITHER `batch_size` NOR `n_steps`. AC-6/AC-9 satisfied; no config/loop/PPO changes.

## Item 3 — Test-command accuracy
Working command = `uv run --extra dev pytest` (pytest is in the pyproject `dev` extra).
- `PROJECT_BRIEF.md` line ~14 (frontmatter `build.commands.test`) and ~287 (prose) → `uv run --extra dev pytest`. **ORCHESTRATOR-AUTHORIZED developer edit (see top).**
- `README.md` line ~388 → update the "CI gates" line to the canonical working command; keep it truthful (present as the canonical command a dev runs locally, don't fabricate what CI literally runs). [developer-authorized]
- `.claude/allowed-commands.yaml` line ~5 comment → update for accuracy. [developer-maintained]
- `.github/workflows/ci.yml:34` → LEAVE AS-IS (bare `uv run pytest` works there because line ~25 runs `uv sync --extra dev` first; AC-12).
- NO sweep of `use-cases/*.md` or `use-cases/plans/*.md` (historical records).

## Item 4 — Windows full-screen + live resize (dashboard.py)
`src/drone_fly/train/tui/dashboard.py` — root cause: `_build_windows_console` calls bare `os.get_terminal_size()` (reads fd 1, already redirected to native.log) → OSError → Rich defaults ~80×25; and passing fixed width/height LOCKS the console so resizes aren't followed. Real terminal fd saved as `self._win_saved_fd1`.
- Extract a single win-guarded helper `_resolve_win_terminal_size() -> tuple[int,int]|None` that calls `os.get_terminal_size(self._win_saved_fd1)` → (columns, lines); returns None on OSError or if saved fd is None (→ let Rich self-detect, never crash). SINGLE size source for both construction and redraw.
- `_build_windows_console`: build Console (force_terminal=True + display file as today) WITHOUT a locking fixed width/height; if helper returns a size, set it via the console `size` property once.
- `_redraw_locked`: add a win-only block (guarded by `self._is_win` and `self._console is not None`) that re-queries the helper and updates `self._console.size` before `self._live.update(...)` — FOLLOWS resizes (AC-16), not a construction lock. Helper None → do NOT touch console.size.
- mac/Linux path untouched (all new code win-guarded) → AC-17.
- AC-18 (hermetic): test calls the helper directly with a sentinel `_win_saved_fd1` (no sys.platform dependency), asserts `os.get_terminal_size` called with the SAVED fd (not bare fd 1) and no locking fixed size passed. On-screen draw + live resize = documented real-Windows smoke-test item (UC-32 residual class).

## Files Affected
**Production code (DEVELOPER):**
- `src/drone_fly/train/tui/render.py` — Item 1 (constants + `status_region_size` + `build_layout` status_width).
- `src/drone_fly/train/tui/dashboard.py` — Item 1 (status_width at 2 sites) + Item 4 (win size helper, unlock console, per-redraw size update).
- `src/drone_fly/train/device.py` — Item 2b (CUDA_OOM_HINT rewrite).
- `README.md` — Item 3 (line ~388). [developer-authorized]
- `.claude/allowed-commands.yaml` — Item 3 (comment line ~5). [developer-maintained]

**ORCHESTRATOR-AUTHORIZED (outside dev paths):**
- `PROJECT_BRIEF.md` — Item 3 (frontmatter line ~14 + prose line ~287).

**LEAVE UNCHANGED (AC-12):** `.github/workflows/ci.yml:34`.

**Test code (QA):**
- `tests/test_tui_render.py` — Item 1: `status_region_size` unit tests — K-line (via `\n`) → min(K+overhead, MAX); 1-line → baseline 3; oversized → clamp to MAX+overhead; width-aware wrap case with a known width int; determinism (AC-5).
- `tests/test_tui_dashboard.py` (± `test_tui_refresh_clock.py`) — Item 4: helper reads saved fd (mock `os.get_terminal_size`, assert called with sentinel saved fd, not bare fd 1), no locking size, graceful None degrade.
- `tests/test_device.py` — Item 2b: `CUDA_OOM_HINT` mentions the real levers AND mentions neither `batch_size` nor `n_steps` (both directions).
- Item 3 doc-consistency test (AC-10/11): parse `PROJECT_BRIEF.md` frontmatter `build.commands.test` + README, assert canonical command contains `--extra dev`; SCOPE to brief frontmatter + README ONLY (do NOT scan use-cases archives or it fails on history).

## Risks & Considerations
- Item 1: width None → logical-line count → size 3 → zero visual regression. Long single line clamps to MAX and clips extras (AC-2). Use Rich-based wrap measurement to avoid under-counting word-wrapped text.
- Item 3: PROJECT_BRIEF.md edit gated on orchestrator authorization (granted). CI + archives intentionally untouched.
- Item 4: on-screen full-screen/resize NOT hermetically verifiable — mocked size-source unit test + documented smoke-test item; graceful degrade if saved fd fails (never regress `--no-tui`/fail-fast paths).
- Cross-cutting AC-14: no training-dynamics/obs/connectome/reward/checkpoint/env-adapter change.

## Challenger final verdict
**Approve** (round 1, no revision). Every code claim verified against the worktree. Item 2 = option (b); item 1 = width-aware autosize; item 3 = canonical docs only. Non-blocking recs folded in (Rich-native wrap measurement; truthful README wording; Item-4 single size source + graceful None degrade + hermetic AC-18 mock).
