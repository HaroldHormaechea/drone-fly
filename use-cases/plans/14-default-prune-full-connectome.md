---
plan_for: use-cases/14-default-prune-full-connectome.md
work_branch: feat/uc-14-default-prune-full-connectome
team: drone-fly-uc-14
approved: 2026-09-18
---

# UC-14 — Default `prune` to the full auto-downloaded MaleCNS connectome

Challenger-APPROVED (v2, after one revision round fixing AC5 hermeticity + AC7 atomicity).

## Analysis (verified against the worktree)
- `PruneRunConfig.connectome` is currently `required=True`; CLI `_run_prune_export` calls `load_connectome(cfg.connectome)` unconditionally; `fetch-connectome` is a print-only stub; `main()` already maps `ConfigError → exit 2, no stack trace`.
- Loader precedence: explicit arg → `DRONE_FLY_CONNECTOME_DIR` → `data/connectome` (`DEFAULT_CONNECTOME_DIR`). `_find_matrix_and_meta` requires **exactly one** `.npz` in a dir and infers the sidecar as `<npz-stem>_meta.csv` (globs `*.npz`, so `.part` temps are ignored).
- **Naming trap (confirmed, README lines 73–77):** source npz `mcns_inprop_all_neuron.npz` + source meta `mcns_all_neuron_meta.csv`; the meta MUST be saved renamed to `mcns_inprop_all_neuron_meta.csv` so the loader's sidecar inference finds it.
- **Offline feasibility confirmed:** full CC-BY matrix cached on this machine at `~/.cache/drone-fly/connectome_src/` (82MB npz + 26.7MB meta); `data/connectome/` doesn't exist yet.
- **Prune-on-full viable:** full meta carries `superclass` with `visual_projection` (9200 = SENSORY_SUPERCLASS) + `descending_neuron` (1310 = MOTOR_SUPERCLASS), so `prune_to_subcircuit` finds both populations.
- **CI hermeticity:** no test references `configs/prune/*.yaml`; prune coverage is `tests/test_prune.py` (fixture) + `tests/test_cli.py::test_prune_dispatch_writes_reusable_slice` (explicit fixture). CI runs bare `uv run pytest` with no network kill-switch, so offline-ness must be *enforced* by a test (see AC5 fix).
- AC4/AC8: nothing defaults to `tests/fixtures` today (train/eval/smoke-train `connectome` default None → loader `data/connectome`); the fixture is only reached when explicitly named — so train/evaluate defaults need NO change.

## Proposed Solution

**1. New download module — `src/drone_fly/connectome/fetch.py` (production).** Mirrors `scripts/fetch_soma_positions.py`'s urllib style. Provides constants (base URL `https://raw.githubusercontent.com/YijieYin/connectome_data_prep/main/data/maleCNS`, source npz `mcns_inprop_all_neuron.npz`, source meta `mcns_all_neuron_meta.csv`, dest meta derived as `<npz-stem>_meta.csv`), a `ConnectomeDownloadError`, and `ensure_full_connectome(dest_dir=None, *, force=False)`:
- Resolve dest dir with the SAME precedence as the loader (explicit → `DRONE_FLY_CONNECTOME_DIR` → `data/connectome`).
- **Reuse-decision key:** both `<dest>/mcns_inprop_all_neuron.npz` AND `<dest>/mcns_inprop_all_neuron_meta.csv` present ⇒ reuse, return dir, no network. Else download.
- **Atomic download (AC7):** download BOTH files to **in-dest** temp paths (`<dest>/.mcns_inprop_all_neuron.npz.part`, `<dest>/.mcns_inprop_all_neuron_meta.csv.part`) — never in cache/`/tmp` — so every `os.replace` is a same-FS atomic rename. Integrity-check BOTH temps before ANY move: npz validated via `zipfile.is_zipfile` + size-floor; meta via size-floor (loader's `len(ids) != n` guard catches truncation). Then **`os.replace` the meta FIRST, the `.npz` LAST** (npz is the commit point → any crash before it leaves no `.npz` = clean retriable state, never a lone-npz/`superclass=None` corruption). On any failure: remove all `.part` temps, raise `ConnectomeDownloadError` with an actionable message.
- **Cache-fallback source:** before network, check `~/.cache/drone-fly/connectome_src/`; if the source files exist there, copy them into the in-dest `.part` temps (then same integrity-check + atomic-move path). Canonical source stays the public URLs; CI never has the cache and never triggers download.

**2. Make `fetch-connectome` real (AC3).** In `cli/__init__.py`, replace the stub branch with `ensure_full_connectome()`, keeping it independently invokable with a small flag surface (`--connectome-dir` optional dest, `--force` optional re-download). Print a clear one-line result (downloaded vs. already-present). Update BOTH the module docstring AND the subparser help string (currently `"[stub] Provision connectome data"`).

**3. Prune defaults to full connectome + auto-download (AC1/AC2).**
- `config.py`: `PruneRunConfig.connectome` becomes **optional** (default `None`, type `str | None`).
- `cli/__init__.py` `_run_prune_export`: if `cfg.connectome is None` → `ensure_full_connectome()` (download-if-absent) then load from returned dir, AND emit the "full-matrix prune runtime/memory expectation" INFO log **scoped to this None branch only**; if `cfg.connectome` is set → `load_connectome(cfg.connectome)` with NO auto-download.
- `main()`: catch `ConnectomeDownloadError` → one-line error + `return 2` (matches existing no-stack-trace convention).

**4. Fixture only when explicitly targeted (AC4).** Falls out of #3; no train/evaluate change (AC8).

**5. Example prune configs → full connectome (AC6).** Edit `configs/prune/{k0,k1,k2}.yaml` to **omit** the `connectome:` key (rely on new default), with a comment noting first-run downloads ~109MB + prunes the full matrix (minutes). Keep `out: artifacts/pruned/k*` + `prune_k` values.

**6. README/docs (AC8).** Update "How to create a slice" + "Connectome data": prune defaults to full connectome (auto-downloaded first run, reused after), `fetch-connectome` really downloads, fixture only when explicitly named, clean failure mode, full-matrix runtime/memory note. Leave train/evaluate docs unchanged.

## Files Affected

**Production code (developer):**
- `src/drone_fly/connectome/fetch.py` — NEW (download, atomic in-dest temps, npz-last, integrity check, cache fallback, `ConnectomeDownloadError`).
- `src/drone_fly/connectome/__init__.py` — export `ensure_full_connectome` + `ConnectomeDownloadError` (+ any constants).
- `src/drone_fly/config.py` — `PruneRunConfig.connectome` optional.
- `src/drone_fly/cli/__init__.py` — real `fetch-connectome` (branch + subparser help + docstring); prune None→auto-download resolution with scoped runtime log; `ConnectomeDownloadError` handling in `main()`.
- `configs/prune/k0.yaml`, `k1.yaml`, `k2.yaml` — default to full connectome.
- `README.md` — docs.

**Test code (qa):**
- `tests/test_fetch.py` — NEW: reuse-when-present (under `no_network`, asserts zero network); download-success (monkeypatch `urlretrieve` and/or tmp cache); meta-rename correctness; **npz-last** ordering / partial-crash leaves no `.npz` = clean retriable; integrity/size + zipfile check failure → `ConnectomeDownloadError`, no `.part` left; in-dest temp location.
- `tests/test_cli.py` — update `test_fetch_connectome_stub` (no longer a stub); add prune-omitted-connectome calls `ensure_full_connectome` (mocked at call site, no cache copy) + prune-explicit-fixture does NOT call it; **add `test_prune_cli_fixture_is_offline`** (real `prune --config` with explicit fixture under `no_network`, asserts slice written + round-trips with sockets disabled — enforces AC5); keep `test_prune_dispatch_writes_reusable_slice`.
- `tests/test_config.py` — update `test_prune_missing_required_keys_error` (now only `out` required; match `out`) + `test_prune_defaults_match_existing_flag_defaults` (assert `connectome is None` when omitted).

## Risks & Considerations
- Full-matrix prune (~161k neurons / ~25M edges) runs in minutes + real memory (accepted per UC pitfall); scoped INFO log prevents "looks like a hang".
- No published checksum for the CC-BY artifact; integrity = `zipfile.is_zipfile` + size-floor on npz, size-floor on meta, plus loader failing loudly on corruption (challenger accepted; no pinned checksum).
- Cache-fallback is a per-machine convenience; canonical source is the public URLs; CI stays hermetic (no cache, fixture-explicit tests, offline-enforced).

## Challenger verdict
**APPROVED** (v2). v1's two Major gaps fixed and verified against the code: (1) AC5 hermeticity now *enforced* by `test_prune_cli_fixture_is_offline` under the `no_network` fixture (not merely conventional); (2) AC7 atomicity — meta-replaced-first / npz-last commit point, both temps integrity-checked before any move, in-dest `.part` temps for same-FS atomic renames, no lone-npz corruption, no cross-FS OSError. Folded recs: zipfile validity + size floor, subparser help string, runtime-cost log scoped to the auto-download branch, mock isolation of the 82MB machine cache. Prune-scoped; train/evaluate untouched (AC8).
