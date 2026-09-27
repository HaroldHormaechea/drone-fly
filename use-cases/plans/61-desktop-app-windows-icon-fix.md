---
plan_for: use-cases/61-desktop-app-training-viewer.md (bugfix: Windows pywebview window-icon crash)
work_branch: feat/uc-61-desktop-app-training-viewer
team: drone-fly-uc-61
approved: 2026-09-27
---

APPROVED PROPOSAL — Windows pywebview window-icon crash fix (UC-61 item 5, branch `feat/uc-61-desktop-app-training-viewer`). Analyst↔challenger approved; challenger independently reproduced all findings.

### Analysis
- Bug at `app/__main__.py:88`: `icon_path` is hardcoded to `static/fly.png` and passed to `webview.start(icon=...)` on **all** platforms. On Windows, pywebview's WinForms backend feeds it to `System.Drawing.Icon.Initialize`, which requires a real `.ico` → `System.ArgumentException` raised on pywebview's GUI thread. The existing `except TypeError:` guard only catches old pywebview lacking the `icon=` kwarg; it cannot catch a .NET exception on the GUI thread.
- **Load-bearing finding (independently reproduced):** the shipped `app/static/fly.ico` has a valid header (6 entries: 16/32/48/64/128/256) but **every entry is PNG-compressed** (Pillow 12.3.0 default), including the 16×16. No BMP entry at any size → "keep the ICO and let WinForms pick a small BMP frame" is impossible; `System.Drawing.Icon` can reject PNG-encoded frames at any size. **Regenerating the ICO (option b) is REQUIRED.**
- Fix verified in the project venv: `sizes=[(16,16),(32,32),(48,48),(64,64)]` + `bitmap_format="bmp"` on Pillow 12.3.0 produces 4 BMP (DIB) entries, 0 PNG.

### Proposed Solution
1. **`app/__main__.py`** — extract the icon choice into a pure, importable helper `_window_icon_path(static_dir, platform=sys.platform)` returning the chosen absolute path or `None`. Logic: Windows (`win32` / `os.name=="nt"`) → `static/fly.ico`; Linux/macOS → `static/fly.png`; `None` if the chosen file is missing. In `main()` (~L85-92) call it: path → `webview.start(icon=path)`; `None` → `webview.start()`. Retain the `except TypeError:` fallback to `webview.start()` for old pywebview. Best-effort, never fatal; Linux/GTK path unchanged.
2. **`scripts/make_app_icons.py`** — change the `fly.ico` save to `sizes=[(16,16),(32,32),(48,48),(64,64)]` and add `bitmap_format="bmp"`. `fly.png` (256×256) generation untouched.
3. **`app/static/fly.ico`** — developer re-runs `python scripts/make_app_icons.py` (project venv/uv) and re-commits the regenerated all-BMP binary. Do not hand-edit.

### Files Affected
- **Production [developer]:** `app/__main__.py` (helper + platform-conditional start), `scripts/make_app_icons.py` (BMP-safe sizes + `bitmap_format`), `app/static/fly.ico` (regenerated binary).
- **Test [qa]:** new hermetic `tests/test_app_window_icon.py` with **two REQUIRED assertions**:
  1. Selection logic: `_window_icon_path` returns `fly.ico` for `win32`, `fly.png` for `linux`/`darwin`, and `None` when the file is absent (temp dir). No window / no pywebview import.
  2. ICO encoding guard: parse `app/static/fly.ico` and assert **no** entry's image data begins with the PNG magic (`\x89PNG`) — makes the committed binary verifiable and blocks a future Pillow-default regression from silently reintroducing PNG-in-ICO.

### Risks & Considerations
- **Owner-eyeball on Windows** — no Windows/.NET runner in CI; the hermetic test covers only the deterministic parts (selection + encoding).
- **Known cosmetic tradeoff (accepted):** dropping the 128/256 entries means Windows upscales the 64px entry for large/Alt-Tab views (slightly soft) — the correct conventional choice; 256px BMP entries are large and historically problematic for `System.Drawing.Icon`.
- Binary asset committed: the encoding test makes its provenance verifiable.
- macOS Cocoa ignores `icon=`; passing `fly.png` is harmless. No Linux/macOS regression.
- Scope confined to the three files + one test.

### Orchestrator (team-lead) note
Independently verified: diagnosis correct (`__main__.py:88` passes fly.png on all platforms), the all-PNG-ICO finding is the crux (prevention via BMP regeneration, not catching, since the throw is on pywebview's GUI thread), and the two required tests are the right hermetic surface. Windows window itself is owner-eyeball. Proceed to Phase 2.
