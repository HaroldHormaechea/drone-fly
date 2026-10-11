"""Honesty-gated audits for reservoir checkpoints (UC-63 AC6/AC7).

Runs ``gpu_prototype/verify_proc.py <ckpt> <eps>`` (completion + the cheat checks) and
``gpu_prototype/diagnose_proc.py <ckpt> <eps>`` (failure breakdown) through the
:class:`~app.runs.RunRegistry` spawner, with the audit's ``PROC_*`` flags and ``AUDIT_SEED`` on the
child's environment. Output goes to ``<root>/.dev-app/audits/<name>/<id>.log`` with a sidecar
``<id>.json`` recording ``{kind, flags, seed, eps, exit_code, ...}`` — never into ``training/``, so
committed model dirs are untouched, and results survive a backend restart.

**Honesty first.** A high completion % that barges through obstacles or hovers is not a success
(``training/EXPERIMENTS.md``: "a higher % that cheats does NOT count"). :func:`honesty_verdict`
applies an ordered rule table to a parsed ``verify_proc`` output, fails closed (a missing line →
``unverified``), and every audit result bundles completion **with** the verdict and cheat checks.

Parsers are tolerant (a missing line → ``None``) and never raise. Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from app import configs_io
from app.configs_io import RecipeError, validate_run_name
from app.runs import RunError, RunRegistry

#: ``mean speed`` below this (m/s) is a hovering signature → ``dishonest``.
HOVER_SPEED_MS = 0.5
#: ``mean tilt`` below this (deg) looks like hovering → ``suspect``.
HOVER_TILT_DEG = 5.0
#: Without ``PROC_OBS_TERMINAL``, overall barging above this % → ``suspect``.
OVERALL_BARGE_NONTERMINAL_PCT = 2.0
#: Completed runs that barged above this % → ``dishonest`` (above 0 and at most this → ``suspect``).
COMPLETED_BARGE_DISHONEST_PCT = 2.0

THRESHOLDS = {
    "HOVER_SPEED_MS": HOVER_SPEED_MS,
    "HOVER_TILT_DEG": HOVER_TILT_DEG,
    "OVERALL_BARGE_NONTERMINAL_PCT": OVERALL_BARGE_NONTERMINAL_PCT,
    "COMPLETED_BARGE_DISHONEST_PCT": COMPLETED_BARGE_DISHONEST_PCT,
}

#: Most severe first.
SEVERITY = ("dishonest", "unverified", "suspect", "nothing-completed", "honest")

KINDS = {"verify": "verify_proc.py", "diagnose": "diagnose_proc.py"}

_NUM = r"-?\d+(?:\.\d+)?"

# -- verify_proc ------------------------------------------------------------------------

_V_COMPLETION = re.compile(
    rf"completion:\s+(\d+)/(\d+)\s+\(({_NUM})%\)(?:\s+laps required:\s+({_NUM}) avg)?"
)
_V_GATES = re.compile(rf"gate passes / episode: mean ({_NUM})")
_V_BARGING = re.compile(
    rf"obstacle BARGING: episodes with any inside-obstacle step: ({_NUM})%\s+\|\s+avg "
    rf"inside-steps/ep ({_NUM})\s+\|\s+min surface clearance ({_NUM}) m"
)
_V_HOVER = re.compile(rf"HOVERING: mean speed ({_NUM}) m/s\s+mean tilt ({_NUM}) deg")
_V_ALIVE = re.compile(rf"mean steps alive ({_NUM})/(\d+)")
_V_COMPLETED = re.compile(
    rf"among COMPLETED runs: barged \(any inside step\) ({_NUM})%\s+\|\s+avg inside-steps "
    rf"({_NUM})\s+\|\s+mean speed ({_NUM}) m/s"
)


def _search(regex: re.Pattern[str], text: str) -> re.Match[str] | None:
    matches = list(regex.finditer(text))
    return matches[-1] if matches else None  # the last match wins (a re-run appended to the log)


def parse_verify(text: str) -> dict[str, Any]:
    """Parse ``verify_proc.py`` stdout. Each field is ``None`` when its line is missing."""
    text = text if isinstance(text, str) else ""
    out: dict[str, Any] = {
        "completion": None,
        "laps_required": None,
        "gate_passes_mean": None,
        "barging_pct": None,
        "avg_inside_steps": None,
        "min_clearance_m": None,
        "mean_speed": None,
        "mean_tilt_deg": None,
        "mean_steps_alive": None,
        "max_steps": None,
        "completed_only": None,
    }
    m = _search(_V_COMPLETION, text)
    if m:
        out["completion"] = {"n": int(m.group(1)), "eps": int(m.group(2)), "pct": float(m.group(3))}
        if m.group(4) is not None:
            out["laps_required"] = float(m.group(4))
    m = _search(_V_GATES, text)
    if m:
        out["gate_passes_mean"] = float(m.group(1))
    m = _search(_V_BARGING, text)
    if m:
        out["barging_pct"] = float(m.group(1))
        out["avg_inside_steps"] = float(m.group(2))
        out["min_clearance_m"] = float(m.group(3))
    m = _search(_V_HOVER, text)
    if m:
        out["mean_speed"] = float(m.group(1))
        out["mean_tilt_deg"] = float(m.group(2))
    m = _search(_V_ALIVE, text)
    if m:
        out["mean_steps_alive"] = float(m.group(1))
        out["max_steps"] = int(m.group(2))
    m = _search(_V_COMPLETED, text)
    if m:
        out["completed_only"] = {
            "barged_pct": float(m.group(1)),
            "avg_inside_steps": float(m.group(2)),
            "mean_speed": float(m.group(3)),
        }
    return out


# -- diagnose_proc -----------------------------------------------------------------------

_D_OVERALL = re.compile(rf"OVERALL completion: (\d+)/(\d+) \(({_NUM})%\)")
_D_ROW = re.compile(rf"^\s+(.+?)\s+n=(\d+)(?:\s+completion\s+({_NUM})%\s+\((\d+)/(\d+)\))?\s*$")
_D_FAILURES = re.compile(r"FAILURE ANATOMY\s+\((\d+) failures\):")
_D_CAUSE = re.compile(r"^\s+(obstacle collision|crash: [a-z-]+|timeout \(ran out\))\s*:\s*(\d+)")
_D_CAUSE_PCT = re.compile(rf"\(({_NUM})%\)")
_D_TILT = re.compile(rf"mean death tilt\s*:\s*({_NUM}) deg")
_D_PROGRESS = re.compile(rf"progress before death: ({_NUM})% of required gates cleared")
_D_MULTI = re.compile(rf"of obstacle-collision deaths: ({_NUM})% on 2\+ obstacle courses")
_D_SECTIONS = {
    "by OBSTACLE COUNT:": "by_obstacle_count",
    "by TIGHTEST GATE aperture:": "by_tightest_gate",
    "by GATE COUNT:": "by_gate_count",
}


def parse_diagnose(text: str) -> dict[str, Any]:
    """Parse ``diagnose_proc.py`` stdout into the completion, breakdown tables, failure anatomy."""
    text = text if isinstance(text, str) else ""
    out: dict[str, Any] = {
        "completion": None,
        "by_obstacle_count": [],
        "by_tightest_gate": [],
        "by_gate_count": [],
        "failures": None,
    }
    m = _search(_D_OVERALL, text)
    if m:
        out["completion"] = {"n": int(m.group(1)), "eps": int(m.group(2)), "pct": float(m.group(3))}
    section: str | None = None
    failures: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped in _D_SECTIONS:
            section = _D_SECTIONS[stripped]
            out[section] = []  # a re-run in the same log restarts the table
            continue
        fm = _D_FAILURES.search(stripped)
        if fm:
            section = None
            failures = {
                "count": int(fm.group(1)),
                "causes": [],
                "mean_death_tilt_deg": None,
                "progress_before_death_pct": None,
                "obstacle_deaths_on_multi_pct": None,
            }
            out["failures"] = failures
            continue
        if not stripped:
            section = None
            continue
        if section is not None:
            rm = _D_ROW.match(line)
            if rm:
                out[section].append(
                    {
                        "label": rm.group(1),
                        "n": int(rm.group(2)),
                        "pct": float(rm.group(3)) if rm.group(3) is not None else None,
                        "completed": int(rm.group(4)) if rm.group(4) is not None else None,
                    }
                )
            continue
        if failures is not None:
            cm = _D_CAUSE.match(line)
            if cm:
                pm = _D_CAUSE_PCT.search(line)
                failures["causes"].append(
                    {
                        "label": cm.group(1),
                        "n": int(cm.group(2)),
                        "pct": float(pm.group(1)) if pm else None,
                    }
                )
                continue
            for regex, key in (
                (_D_TILT, "mean_death_tilt_deg"),
                (_D_PROGRESS, "progress_before_death_pct"),
                (_D_MULTI, "obstacle_deaths_on_multi_pct"),
            ):
                vm = regex.search(line)
                if vm:
                    failures[key] = float(vm.group(1))
    return out


# -- verdict -----------------------------------------------------------------------------


def _most_severe(verdicts: list[str]) -> str:
    for level in SEVERITY:
        if level in verdicts:
            return level
    return "honest"


def honesty_verdict(parsed: dict[str, Any], audit_flags: list[str] | None) -> dict[str, Any]:
    """Apply the ordered honesty rules to a :func:`parse_verify` result.

    (a) completion, BARGING or HOVERING line missing → ``unverified`` (stop).
    (b) mean speed < :data:`HOVER_SPEED_MS` → ``dishonest`` (hovering); mean tilt <
        :data:`HOVER_TILT_DEG` → ``suspect`` — regardless of completion.
    (c) without ``PROC_OBS_TERMINAL``: overall barging > :data:`OVERALL_BARGE_NONTERMINAL_PCT` →
        ``suspect`` (with it on, overall barging is the collision-death rate: shown only).
    (d) completed > 0: the among-COMPLETED line must be present (else ``unverified``); completed
        barging > :data:`COMPLETED_BARGE_DISHONEST_PCT` → ``dishonest``, > 0 → ``suspect``.
    (e) completed == 0 and nothing from (b)/(c) → ``nothing-completed``.
    (f) otherwise → ``honest``.

    Reasons accumulate; the verdict is the most severe
    (``dishonest > unverified > suspect > nothing-completed > honest``).
    """
    parsed = parsed or {}
    flags = list(audit_flags or [])
    missing = [
        label
        for label, ok in (
            ("completion", parsed.get("completion") is not None),
            ("obstacle BARGING", parsed.get("barging_pct") is not None),
            ("HOVERING", parsed.get("mean_speed") is not None),
        )
        if not ok
    ]
    if missing:
        return {
            "verdict": "unverified",
            "reasons": [f"audit output is missing the {', '.join(missing)} line(s)"],
        }

    verdicts: list[str] = []
    reasons: list[str] = []

    speed = parsed["mean_speed"]
    tilt = parsed.get("mean_tilt_deg")
    if speed < HOVER_SPEED_MS:
        verdicts.append("dishonest")
        reasons.append(f"hovering: mean speed {speed:.2f} m/s < {HOVER_SPEED_MS} m/s")
    if tilt is not None and tilt < HOVER_TILT_DEG:
        verdicts.append("suspect")
        reasons.append(f"low tilt: mean tilt {tilt:.0f} deg < {HOVER_TILT_DEG:.0f} deg")

    barging = parsed["barging_pct"]
    if "PROC_OBS_TERMINAL" not in flags and barging > OVERALL_BARGE_NONTERMINAL_PCT:
        verdicts.append("suspect")
        reasons.append(
            f"overall barging {barging:.0f}% > {OVERALL_BARGE_NONTERMINAL_PCT:.0f}% "
            "(PROC_OBS_TERMINAL off, so contact did not end the episode)"
        )

    completed = parsed["completion"]["n"]
    if completed > 0:
        only = parsed.get("completed_only")
        if only is None:
            verdicts.append("unverified")
            reasons.append("completed runs exist but the among-COMPLETED barging line is missing")
        else:
            cb = only["barged_pct"]
            if cb > COMPLETED_BARGE_DISHONEST_PCT:
                verdicts.append("dishonest")
                reasons.append(
                    f"completed runs barged: {cb:.0f}% > {COMPLETED_BARGE_DISHONEST_PCT:.0f}%"
                )
            elif cb > 0:
                verdicts.append("suspect")
                reasons.append(f"completed runs barged: {cb:.0f}% (> 0%)")
    elif not verdicts:
        verdicts.append("nothing-completed")
        reasons.append("no episode completed the course")

    return {"verdict": _most_severe(verdicts), "reasons": reasons}


# -- storage + running -------------------------------------------------------------------


def audits_dir(project_root: str, name: str) -> str:
    """Return ``<root>/.dev-app/audits/<name>``."""
    return os.path.join(project_root, ".dev-app", "audits", name)


def _sidecar(project_root: str, name: str, audit_id: str) -> str:
    return os.path.join(audits_dir(project_root, name), audit_id + ".json")


def _log(project_root: str, name: str, audit_id: str) -> str:
    return os.path.join(audits_dir(project_root, name), audit_id + ".log")


def _write_json(path: str, data: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise RecipeError(f"'{label}' must be an integer") from exc
    if out < 0 or (out == 0 and not allow_zero):
        raise RecipeError(f"'{label}' must be {'non-negative' if allow_zero else 'positive'}")
    return out


def _aux_key(name: str, audit_id: str) -> str:
    return f"audit:{name}:{audit_id}"


def start_audit(
    registry: RunRegistry,
    name: str,
    kind: str,
    *,
    flags: list[str] | None = None,
    seed: Any = None,
    eps: Any = None,
    ts: str | None = None,
) -> dict[str, Any]:
    """Spawn a ``verify`` / ``diagnose`` audit of ``name``'s checkpoint; return its result record.

    ``flags`` default to the run's ``run.json`` flags and are required for committed / external
    models (which have no ``run.json``). ``seed`` / ``eps`` default to the Settings ``audit_seed``
    (1234) / ``audit_eps`` (512). Refused (:class:`RunError`, 409) while the run is live or when
    there is no checkpoint.
    """
    name = validate_run_name(name)
    if kind not in KINDS:
        raise RecipeError(f"unknown audit kind {kind!r} (allowed: {', '.join(KINDS)})")
    run_kind = registry.kind(name)
    if run_kind is None:
        raise FileNotFoundError(f"unknown run {name!r}")
    eligible, reason = registry.audit_eligible(name)
    if not eligible:
        raise RunError(f"cannot audit {name!r}: {reason}")
    if flags is None:
        if run_kind != "app":
            raise RecipeError(
                f"'flags' are required to audit a {run_kind} model (use the flags it was trained "
                "with — gates_proc_env reads them at import time)"
            )
        cfg = configs_io.load_run_config(registry.project_root, name)
        flags = list(cfg.get("flags") or [])
    flags = configs_io.validate_flags(flags)
    settings = registry.settings()
    seed = _positive_int(
        seed if seed not in (None, "") else settings.get("audit_seed", 1234),
        "seed",
        allow_zero=True,
    )
    eps = _positive_int(eps if eps not in (None, "") else settings.get("audit_eps", 512), "eps")

    prefix = registry.interpreter_prefix()
    ckpt = registry.checkpoint_path(name)
    assert ckpt is not None  # audit_eligible guarantees a checkpoint
    argv = [*prefix, os.path.join(registry.gpu_prototype_dir, KINDS[kind]), ckpt, str(eps)]
    env = registry.child_env(flags, {"AUDIT_SEED": str(seed)})

    directory = audits_dir(registry.project_root, name)
    os.makedirs(directory, exist_ok=True)
    stamp = ts or time.strftime("%Y%m%d-%H%M%S")
    audit_id = f"{stamp}-{kind}"
    n = 2
    while os.path.exists(_sidecar(registry.project_root, name, audit_id)):
        audit_id = f"{stamp}-{n}-{kind}"
        n += 1
    meta = {
        "id": audit_id,
        "kind": kind,
        "flags": flags,
        "seed": seed,
        "eps": eps,
        "ckpt": os.path.relpath(ckpt, registry.project_root).replace("\\", "/"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_code": None,
    }
    _write_json(_sidecar(registry.project_root, name, audit_id), meta)
    registry.spawn_aux(
        _aux_key(name, audit_id),
        argv,
        env=env,
        log_path=_log(registry.project_root, name, audit_id),
    )
    return audit_result(registry, name, audit_id)


def _read_text(path: str, limit: int = 256 * 1024) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _refresh_exit(registry: RunRegistry, name: str, meta: dict[str, Any]) -> str:
    """Update ``meta['exit_code']`` from a finished process; return the audit's state."""
    if meta.get("exit_code") is not None:
        return "done" if meta["exit_code"] == 0 else "failed"
    proc = registry.aux_proc(_aux_key(name, meta["id"]))
    if proc is None:
        return "interrupted"  # no exit code and no live process (backend restarted mid-audit)
    code = proc.poll()
    if code is None:
        return "running"
    meta["exit_code"] = code
    _write_json(_sidecar(registry.project_root, name, meta["id"]), meta)
    return "done" if code == 0 else "failed"


def _load_meta(project_root: str, name: str, audit_id: str) -> dict[str, Any]:
    path = _sidecar(project_root, name, audit_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no audit {audit_id!r} for run {name!r}")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if not isinstance(meta, dict):
        raise FileNotFoundError(f"audit {audit_id!r} sidecar is unreadable")
    meta.setdefault("id", audit_id)
    return meta


def _audit_ids(project_root: str, name: str) -> list[str]:
    directory = audits_dir(project_root, name)
    if not os.path.isdir(directory):
        return []
    return sorted(
        (fn[: -len(".json")] for fn in os.listdir(directory) if fn.endswith(".json")),
        reverse=True,
    )


def audit_result(
    registry: RunRegistry,
    name: str,
    audit_id: str,
    *,
    _verify_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one audit's result: state, parsed output, and completion bundled with the verdict."""
    meta = _load_meta(registry.project_root, name, audit_id)
    state = _refresh_exit(registry, name, meta)
    text = _read_text(_log(registry.project_root, name, audit_id))
    kind = meta.get("kind")
    flags = meta.get("flags") or []
    verify = parse_verify(text) if kind == "verify" else None
    diagnose = parse_diagnose(text) if kind == "diagnose" else None
    parsed = verify if verify is not None else (diagnose or {})
    verdict_source = audit_id

    if state == "running":
        verdict = {"verdict": "unverified", "reasons": ["audit still running"]}
    elif state == "interrupted":
        verdict = {
            "verdict": "unverified",
            "reasons": ["audit process did not report an exit code (backend restarted?)"],
        }
    elif state == "failed":
        verdict = {
            "verdict": "unverified",
            "reasons": [f"audit process exited {meta.get('exit_code')}"],
        }
    elif kind == "verify":
        verdict = honesty_verdict(verify or {}, flags)
    else:
        fallback = _verify_fallback
        if fallback is None:
            fallback = latest_verify_verdict(registry, name)
        if fallback is None:
            verdict = {
                "verdict": "unverified",
                "reasons": [
                    "diagnose_proc does not run the cheat checks and no finished verify audit "
                    "exists for this run — run a verify audit"
                ],
            }
            verdict_source = None
        else:
            verdict = {"verdict": fallback["verdict"], "reasons": list(fallback["reasons"])}
            verdict_source = fallback["id"]

    lines = text.splitlines()
    return {
        "id": meta["id"],
        "name": name,
        "kind": kind,
        "state": state,
        "exit_code": meta.get("exit_code"),
        "flags": flags,
        "seed": meta.get("seed"),
        "eps": meta.get("eps"),
        "ckpt": meta.get("ckpt"),
        "started": meta.get("started"),
        "completion": parsed.get("completion"),
        "verdict": verdict["verdict"],
        "reasons": verdict["reasons"],
        "verdict_source": verdict_source,
        "obs_terminal": "PROC_OBS_TERMINAL" in flags,
        "verify": verify,
        "diagnose": diagnose,
        "log_tail": lines[-40:],
        "thresholds": dict(THRESHOLDS),
    }


def latest_verify_verdict(registry: RunRegistry, name: str) -> dict[str, Any] | None:
    """Return ``{id, verdict, reasons}`` of the newest finished verify audit of ``name``."""
    for audit_id in _audit_ids(registry.project_root, name):
        try:
            meta = _load_meta(registry.project_root, name, audit_id)
        except (OSError, ValueError):
            continue
        if meta.get("kind") != "verify":
            continue
        if _refresh_exit(registry, name, meta) != "done":
            continue
        text = _read_text(_log(registry.project_root, name, audit_id))
        verdict = honesty_verdict(parse_verify(text), meta.get("flags") or [])
        return {"id": audit_id, **verdict}
    return None


def list_audits(registry: RunRegistry, name: str) -> list[dict[str, Any]]:
    """Return every audit of ``name``, newest first."""
    name = validate_run_name(name)
    fallback = latest_verify_verdict(registry, name)
    out = []
    for audit_id in _audit_ids(registry.project_root, name):
        try:
            out.append(audit_result(registry, name, audit_id, _verify_fallback=fallback))
        except (OSError, ValueError):
            continue
    return out


__all__ = [
    "COMPLETED_BARGE_DISHONEST_PCT",
    "HOVER_SPEED_MS",
    "HOVER_TILT_DEG",
    "KINDS",
    "OVERALL_BARGE_NONTERMINAL_PCT",
    "SEVERITY",
    "THRESHOLDS",
    "audit_result",
    "audits_dir",
    "honesty_verdict",
    "latest_verify_verdict",
    "list_audits",
    "parse_diagnose",
    "parse_verify",
    "start_audit",
]
