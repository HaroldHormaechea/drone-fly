"""The UC69 diagnosis report of an oracle teacher session: ``drone-fly harness oracle-report``.

A session folder (``artifacts/eval/<drone>/<run>/oracle/<track>/sessions/<UTC>-<sha8>/``) holds
the frozen checkpoint (``checkpoint/``) and one recorded run per attempt (``attempts/``). The
report scores every attempt with :mod:`drone_fly.harness.scoring` against the course the teacher
was trained on, runs the teacher in the simulator on the same track (``evaluate_teacher``, the
frozen checkpoint, CPU, ground starts at the spawn, randomized + nominal) and writes
``report.md`` / ``report.json`` into the session and a copy at the track level.

Headline: ``completion_sim_rule`` (crossings before the first contact under the sim's own rules)
with its Wilson 95 % interval over N_effective (attempts that ended by the episode monitor, the
duration cap or the course finish; fail-safe and refused attempts are listed by reason and
excluded); ``completion_raw`` beside it. Verdict: **control transfers** when the headline is
within :data:`TRANSFER_PP` of the sim's randomized completion, else **does not transfer** with
the dominant failure mode; **inconclusive** when the sim itself completes under
:data:`INCONCLUSIVE_SIM`; a **geometry disagreement** flag when raw and sim-rule completion differ
by more than :data:`TRANSFER_PP`. Every output is labelled ORACLE (diagnostic only).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

TRANSFER_PP = 0.20
INCONCLUSIVE_SIM = 0.50
FAILSAFE_RERUN_SHARE = 0.20
ARM_DELAY_FLAG_S = 0.25
#: Thresholds compare with this tolerance: shares of N = 10 are multiples of 0.1, and
#: ``abs(0.7 - 0.9)`` is ``0.20000000000000007`` (a 20 pp gap still transfers).
EPS = 1e-9
EXCLUDED_CLASSES = ("failsafe", "refused")
Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Wilson score interval of ``k`` successes in ``n`` trials (None when ``n`` is 0)."""
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def _ci(ci: tuple[float, float] | list[float] | None) -> str:
    return "-" if ci is None else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] %"


def _stat(values: list[float]) -> dict[str, float] | None:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": float(np.mean(v)), "p50": float(np.median(v)),
            "min": float(np.min(v)), "max": float(np.max(v))}  # fmt: skip


# ----------------------------------------------------------------------------- inputs


def attempt_dirs(session: Path) -> list[Path]:
    root = session / "attempts"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "meta.json").is_file())


def frozen_checkpoint(session: Path) -> Path:
    cks = sorted((session / "checkpoint").glob("*.pt"))
    if len(cks) != 1:
        from drone_fly.harness.errors import HarnessConfigError

        raise HarnessConfigError(f"{session}/checkpoint must hold exactly one frozen .pt")
    return cks[0]


def outer_sizes(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """``gate_types`` class -> outer (w, h) from a run's merged harness config."""
    out = {}
    for gt in config.get("gate_types") or []:
        outer = gt.get("outer")
        if gt.get("class") and outer:
            out[str(gt["class"])] = (float(outer[0]), float(outer[1]))
    return out


# ----------------------------------------------------------------------------- aggregation


def live_summary(scores: list[dict[str, Any]], course: Any) -> dict[str, Any]:
    """Session-level live figures over the N_effective attempts (fail-safes listed apart)."""
    valid = [s for s in scores if s.get("stop_class") not in EXCLUDED_CLASSES]
    excluded = [s for s in scores if s.get("stop_class") in EXCLUDED_CLASSES]
    n = len(valid)
    k_sim = sum(bool(s["completion_sim_rule"]) for s in valid)
    k_raw = sum(bool(s["completion_raw"]) for s in valid)
    passed = sum(s["k_sim_rule"] for s in valid)
    failed = sum(not s["completion_sim_rule"] for s in valid)
    laps = sum(_laps_sim(s, course) for s in valid)
    modes = Counter(s["failure"]["primary"] for s in valid if s.get("failure"))
    nfail = sum(modes.values())
    lap_times = [t for s in valid for t in s["lap_times_s"]]
    full = [s["full_time_s"] for s in valid if s["full_time_s"] is not None]
    arm = [s["arm_confirm_delay_s"] for s in scores if s.get("arm_confirm_delay_s") is not None]
    return {
        "attempts": len(scores),
        "n_effective": n,
        "completion_sim_rule": None if not n else k_sim / n,
        "completion_sim_rule_ci95": wilson(k_sim, n),
        "completion_raw": None if not n else k_raw / n,
        "completion_raw_ci95": wilson(k_raw, n),
        "gates_passed_mean_raw": None if not n else float(np.mean([s["k_raw"] for s in valid])),
        "gates_passed_mean_sim_rule": None
        if not n
        else float(np.mean([s["k_sim_rule"] for s in valid])),
        "total_crossings": course.total_crossings,
        "per_passage_success": passed / max(passed + failed, 1) if n else None,
        "per_lap_success": laps / max(laps + failed, 1) if n else None,
        "lap_time_s": _stat(lap_times),
        "full_time_s": _stat(full),
        "time_to_first_gate_s": _stat([s["time_to_first_gate_s"] for s in valid]),
        "first_window_max_tilt_deg": _stat([s["first_window"]["max_tilt_deg"] for s in valid]),
        "first_window_max_rate_rad_s": _stat([s["first_window"]["max_rate_rad_s"] for s in valid]),
        "stop_class_share": _shares([s.get("stop_class") for s in scores]),
        "stop_reason_share": _shares([s.get("stop_reason") for s in scores]),
        "failure_mode_share": {m: c / nfail for m, c in modes.most_common()} if nfail else {},
        "dominant_failure_mode": modes.most_common(1)[0][0] if modes else None,
        "excluded": [
            {
                "run": s["run"],
                "stop_class": s.get("stop_class"),
                "reason": s.get("stop_reason"),
                "error": s.get("stop_error"),
            }
            for s in excluded
        ],  # fmt: skip
        "excluded_share": len(excluded) / len(scores) if scores else 0.0,
        "arm_confirm_delay_s": None
        if not arm
        else {
            "p50": float(np.median(arm)),
            "max": float(np.max(arm)),
            "flagged": bool(np.max(arm) > ARM_DELAY_FLAG_S + EPS),
        },  # fmt: skip
        "contacts": [
            {"run": s["run"], **s["contact"]} for s in valid if s.get("contact") is not None
        ],
        "live_mismatches": sum(len(s["live_mismatches"]) for s in scores),
        "training_active": any(s.get("training_active") for s in scores),
    }


def _laps_sim(score: dict[str, Any], course: Any) -> int:
    from drone_fly.harness.scoring import laps_done

    return laps_done(int(score["k_sim_rule"]), course)


def _shares(values: list[Any]) -> dict[str, float]:
    c = Counter(str(v) for v in values)
    n = sum(c.values())
    return {k: v / n for k, v in c.most_common()} if n else {}


def tick_summary(dirs: list[Path], metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Tick rate and latency figures across the attempts (``latency_summary.json`` + meta)."""
    hz, missed, ticks = [], 0, 0
    stages: dict[str, dict[str, list[float]]] = {}
    for d, meta in zip(dirs, metas, strict=True):
        lat_path = d / "latency_summary.json"
        if lat_path.is_file():
            lat = json.loads(lat_path.read_text(encoding="utf-8"))
            if lat.get("effective_rate_hz") is not None:
                hz.append(float(lat["effective_rate_hz"]))
            missed += int(lat.get("missed_deadlines") or 0)
            ticks += int(lat.get("ticks") or 0)
            lm = lat.get("latency_ms") or {}
            for name, key in (("total", "total"), ("oracle", "gate_detector"),
                              ("assemble_policy_decode", "policy")):  # fmt: skip
                st = lm.get(key)
                if st:
                    for q in ("p95", "p99"):
                        stages.setdefault(name, {}).setdefault(q, []).append(float(st[q]))
        for name, st in (meta.get("teacher_timing_ms") or {}).items():
            if st:
                for q in ("p95", "p99"):
                    stages.setdefault(name, {}).setdefault(q, []).append(float(st[q]))
    return {
        "effective_hz_p50": float(np.median(hz)) if hz else None,
        "effective_hz_min": float(np.min(hz)) if hz else None,
        "missed_deadlines": missed,
        "ticks": ticks,
        "missed_deadline_rate": missed / ticks if ticks else None,
        "stages_ms": {
            name: {
                q: {"p50_over_attempts": float(np.median(v)), "max": float(np.max(v))}
                for q, v in qs.items()
            }
            for name, qs in stages.items()
        },  # fmt: skip
    }


# ----------------------------------------------------------------------------- sim side


def sim_summary(
    run_dir: Path,
    track: str,
    checkpoint: Path,
    *,
    episodes: int,
    roots: dict[str, str],
    evaluate: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The teacher's sim completion on ``track`` (frozen checkpoint, CPU, both modes)."""
    from drone_fly.harness.teacher import patched_env

    if evaluate is None:
        from drone_fly.training.evaluate import evaluate_teacher as evaluate
    with patched_env(roots):
        res = evaluate(
            run_dir, [f"liftoff:{track}"], checkpoint=checkpoint, device="cpu", write=False,
            episodes=episodes, modes=("randomized", "nominal"),
        )  # fmt: skip
    out: dict[str, Any] = {"episodes": episodes, "modes": {}}
    for mode, m in (res.get("modes") or {}).items():
        per = [t for t in m.get("per_track") or [] if t.get("id") == track]
        t = per[0] if per else {}
        out["modes"][mode] = {
            k: t.get(k)
            for k in ("completion", "gates_passed_mean", "per_passage_success", "per_lap_success",
                      "lap_time_s", "full_time_s", "outcomes", "episodes")
        }  # fmt: skip
    return out


# ----------------------------------------------------------------------------- verdict


def verdict(live: dict[str, Any], sim: dict[str, Any] | None) -> dict[str, Any]:
    """The verdict, flags and the one-paragraph conclusion."""
    n = live["n_effective"]
    head = live["completion_sim_rule"]
    raw = live["completion_raw"]
    rand = ((sim or {}).get("modes") or {}).get("randomized") or {}
    sim_c = rand.get("completion")
    mode = live["dominant_failure_mode"]
    flags: list[str] = []
    if n and head is not None and raw is not None and abs(raw - head) > TRANSFER_PP + EPS:
        kinds = Counter(c["kind"] for c in live["contacts"])
        flags.append(
            f"Geometry disagreement: raw completion {_pct(raw)} vs sim-rule {_pct(head)} "
            f"(contacts: {dict(kinds)})"
        )
    if live["excluded_share"] > FAILSAFE_RERUN_SHARE + EPS:
        flags.append(
            f"Fail-safe / refused attempts are {_pct(live['excluded_share'])} of the session "
            f"(> {_pct(FAILSAFE_RERUN_SHARE)}): re-run the session for AC9"
        )
    if live["training_active"]:
        from drone_fly.harness.teacher import CONCURRENT_LABEL

        flags.append(CONCURRENT_LABEL)
    if not n:
        kind, text = "no-data", "No attempt ended by the episode monitor: nothing to compare."
    elif sim_c is None:
        kind = "no-sim"
        text = f"Live sim-rule completion {_pct(head)} (N={n}); the sim comparison is unavailable."
    elif sim_c < INCONCLUSIVE_SIM - EPS:
        kind = "inconclusive"
        text = (
            f"Inconclusive: the teacher completes only {_pct(sim_c)} of randomized sim episodes on "
            f"this track (< {_pct(INCONCLUSIVE_SIM)}), so the completion gap says little. "
            f"Per-passage success: live {_pct(live['per_passage_success'])} vs sim "
            f"{_pct(rand.get('per_passage_success'))}."
        )
    elif abs(head - sim_c) <= TRANSFER_PP + EPS:
        kind = "transfers"
        text = (
            f"Control transfers: live sim-rule completion {_pct(head)} "
            f"{_ci(live['completion_sim_rule_ci95'])} (N={n}) is within "
            f"{round(100 * TRANSFER_PP)} pp of the sim's {_pct(sim_c)}."
        )
    else:
        kind = "does-not-transfer"
        text = (
            f"Does not transfer: live sim-rule completion {_pct(head)} "
            f"{_ci(live['completion_sim_rule_ci95'])} (N={n}) vs the sim's {_pct(sim_c)}; "
            f"dominant failure mode: {mode or 'none recorded'}."
        )
    if kind == "transfers":
        meaning = "the remaining gap to a live student is perception (UC70/UC71)."
    elif kind == "does-not-transfer":
        meaning = "the gap is sim-to-Liftoff dynamics or timing, not perception" + (
            f" (mostly {mode})." if mode else "."
        )
    else:
        meaning = "the evidence does not yet separate control from the teacher's own sim skill."
    conclusion = f"{text} With oracle gates (exact poses from the track XML), {meaning}" + (
        f" Flags: {'; '.join(flags)}." if flags else ""
    )
    return {"kind": kind, "text": text, "flags": flags, "conclusion": conclusion,
            "sim_randomized_completion": sim_c, "transfer_pp": TRANSFER_PP}  # fmt: skip


# ----------------------------------------------------------------------------- report


def build_report(
    session: str | Path,
    *,
    sim_episodes: int = 32,
    sim: bool = True,
    evaluate: Callable[..., dict[str, Any]] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Score a session (see the module docstring); returns the JSON-able report."""
    import yaml

    from drone_fly.harness.errors import HarnessConfigError
    from drone_fly.harness.oracle import ORACLE_LABEL, PERCEPTION, load_oracle_course
    from drone_fly.harness.scoring import (
        FAILURE_MODES,
        MODE_PRIORITY,
        ContactRules,
        load_attempt,
        score_attempt,
        thresholds,
    )
    from drone_fly.harness.teacher import env_roots
    from drone_fly.sim.action import read_sidecar

    sdir = Path(session)
    dirs = attempt_dirs(sdir)
    if not dirs:
        raise HarnessConfigError(f"{sdir}: no recorded attempts under attempts/")
    ck = frozen_checkpoint(sdir)
    side = read_sidecar(ck) or {}
    records = [load_attempt(d) for d in dirs]
    metas = [r.meta for r in records]
    m0 = next((m for m in metas if m.get("teacher")), metas[0])
    teacher = m0.get("teacher") or {}
    track = m0.get("track_id")
    if not track:
        raise HarnessConfigError(f"{dirs[0]}: meta has no track_id (not an oracle teacher run)")
    cinfo = teacher.get("course") or {}
    roots = {"root": cinfo.get("trackdata_root"), "scenes_root": cinfo.get("scenes_root")}
    course, info = load_oracle_course(
        track, side, root=roots["root"], scenes_root=roots["scenes_root"], use_cache=use_cache
    )
    cfg_path = sdir / "checkpoint" / "config.yaml"
    run_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    rules = ContactRules.from_env((run_cfg or {}).get("env"))
    scores = [
        score_attempt(r, course, rules, outer_by_class=outer_sizes(r.config)) for r in records
    ]
    live = live_summary(scores, course)
    sim_res: dict[str, Any] | None = None
    if sim:
        try:
            sim_res = sim_summary(
                Path(teacher.get("run_dir") or ck.parent), track, ck, episodes=sim_episodes,
                roots=env_roots(roots["root"], roots["scenes_root"]), evaluate=evaluate,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 - the live side still reports
            log.exception("sim evaluation failed")
            sim_res = {"error": repr(exc)}
    labels = [
        ORACLE_LABEL,
        f"{str(info.get('split')).upper()} track, {info.get('scene')} scene",
    ]
    if live["training_active"]:
        from drone_fly.harness.teacher import CONCURRENT_LABEL

        labels.append(CONCURRENT_LABEL)
    return {
        "perception": PERCEPTION,
        "labels": labels,
        "session": str(sdir),
        "track": {k: v for k, v in info.items() if k != "files"},
        "teacher": {
            k: teacher.get(k)
            for k in (
                "run_id",
                "run_dir",
                "sha256",
                "selection",
                "selection_label",
                "it",
                "steps",
                "obs_schema",
                "conversion_fingerprint",
                "preset_world_scale_ignored",
                "world_scale_effective",
            )
        },  # fmt: skip
        "checkpoint": str(ck),
        "contact_rules": rules.as_dict(course),
        "episode_rules": ((metas[0].get("oracle") or {}).get("episode_rules")),
        "classifier_thresholds": thresholds(),
        "failure_modes": {m: MODE_PRIORITY[m] for m in FAILURE_MODES},
        "live": live,
        "ticks": tick_summary(dirs, metas),
        "sim": sim_res,
        "verdict": verdict(live, sim_res),
        "attempts": scores,
    }


def markdown(rep: dict[str, Any]) -> str:
    live, tk, v = rep["live"], rep["ticks"], rep["verdict"]
    t, tr = rep["track"], rep["teacher"]
    sim_modes = ((rep.get("sim") or {}).get("modes")) or {}
    lines = [
        f"# Oracle teacher session: {t.get('track_id')} ({t.get('name')})",
        "",
        *[f"> **{lab}**" for lab in rep["labels"]],
        "",
        f"- perception: **{rep['perception']}**; teacher `{tr.get('run_id')}` "
        f"checkpoint sha `{str(tr.get('sha256'))[:8]}` ({tr.get('selection_label')})",
        f"- conversion `{tr.get('conversion_fingerprint')}`; world scale "
        f"{tr.get('world_scale_effective')} (preset's {tr.get('preset_world_scale_ignored')} "
        "ignored); "
        f"{t.get('total_crossings')} crossings ({t.get('passages')} passages x {t.get('laps')} "
        f"laps{', circuit' if t.get('circuit') else ''}); time limit {t.get('time_limit_s')} s",
        "",
        "## Verdict",
        "",
        f"**{v['text']}**",
        "",
        *[f"- {f}" for f in v["flags"]],
        "",
        v["conclusion"],
        "",
        "## Live (Liftoff, oracle gates) vs sim",
        "",
        "| figure | live | sim randomized | sim nominal |",
        "|---|---|---|---|",
    ]
    r = sim_modes.get("randomized") or {}
    nm = sim_modes.get("nominal") or {}
    n = live["n_effective"]

    def sim_cell(m: dict[str, Any], k: str, pct: bool = True) -> str:
        x = m.get(k)
        return _pct(x) if pct else ("-" if x is None else f"{x:.1f}")

    def stat(s: dict[str, Any] | None) -> str:
        return "-" if not s else f"{s['p50']:.2f} (min {s['min']:.2f}, max {s['max']:.2f})"

    lines += [
        f"| completion (sim rules; headline) | **{_pct(live['completion_sim_rule'])}** "
        f"{_ci(live['completion_sim_rule_ci95'])}, N_eff {n} | {sim_cell(r, 'completion')} | "
        f"{sim_cell(nm, 'completion')} |",
        f"| completion (raw) | {_pct(live['completion_raw'])} {_ci(live['completion_raw_ci95'])}"
        f", N_eff {n} | | |",
        f"| gates passed (mean, sim rules / raw) | "
        f"{_fmt(live['gates_passed_mean_sim_rule'])} / {_fmt(live['gates_passed_mean_raw'])} of "
        f"{live['total_crossings']} | {sim_cell(r, 'gates_passed_mean', False)} | "
        f"{sim_cell(nm, 'gates_passed_mean', False)} |",
        f"| per-passage success | {_pct(live['per_passage_success'])} | "
        f"{sim_cell(r, 'per_passage_success')} | {sim_cell(nm, 'per_passage_success')} |",
        f"| per-lap success | {_pct(live['per_lap_success'])} | "
        f"{sim_cell(r, 'per_lap_success')} | {sim_cell(nm, 'per_lap_success')} |",
        f"| lap time s (p50) | {stat(live['lap_time_s'])} | {sim_cell(r, 'lap_time_s', False)} | "
        f"{sim_cell(nm, 'lap_time_s', False)} |",
        f"| race time s (p50) | {stat(live['full_time_s'])} | "
        f"{sim_cell(r, 'full_time_s', False)} | {sim_cell(nm, 'full_time_s', False)} |",
        f"| time to first gate s | {stat(live['time_to_first_gate_s'])} | | |",
        f"| first 2 s max tilt deg | {stat(live['first_window_max_tilt_deg'])} | | |",
        f"| first 2 s max rate rad/s | {stat(live['first_window_max_rate_rad_s'])} | | |",
        "",
    ]
    if (rep.get("sim") or {}).get("error"):
        lines += [f"Sim evaluation failed: `{rep['sim']['error']}`", ""]
    lines += ["## Stops and failure modes", ""]
    lines.append(
        "- stop classes: "
        + ", ".join(f"{k} {_pct(x)}" for k, x in live["stop_class_share"].items())
    )
    lines.append(
        "- stop reasons: "
        + ", ".join(f"{k} {_pct(x)}" for k, x in live["stop_reason_share"].items())
    )
    lines.append(
        f"- failure modes (of the N_eff {n} attempts that did not finish): "
        + (", ".join(f"{k} {_pct(x)}" for k, x in live["failure_mode_share"].items()) or "none")
    )
    if live["excluded"]:
        lines.append(f"- excluded (fail-safe / refused), {len(live['excluded'])}:")
        by = Counter(e["reason"] for e in live["excluded"])
        lines += [f"  - {k}: {c}" for k, c in by.most_common()]
    arm = live["arm_confirm_delay_s"]
    if arm:
        lines.append(
            f"- arm confirmation delay p50 {arm['p50']:.2f} s, max {arm['max']:.2f} s"
            + (f" (FLAG: above {ARM_DELAY_FLAG_S} s)" if arm["flagged"] else "")
        )
    lines.append(f"- live vs offline crossing mismatches: {live['live_mismatches']}")
    lines += [
        "",
        "## Tick rate",
        "",
        f"- effective rate p50 {_fmt(tk['effective_hz_p50'])} Hz, min "
        f"{_fmt(tk['effective_hz_min'])} Hz; missed deadlines {tk['missed_deadlines']} of "
        f"{tk['ticks']} ticks ({_pct(tk['missed_deadline_rate'])})",
    ]
    for name, qs in tk["stages_ms"].items():
        lines.append(
            f"- {name}: "
            + ", ".join(
                f"{q} {x['p50_over_attempts']:.1f} ms (worst {x['max']:.1f})" for q, x in qs.items()
            )
        )
    lines += [
        "",
        "## Attempts",
        "",
        "| run | stop | class | sim-rule k | raw k | contact | failure mode |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in rep["attempts"]:
        c = s.get("contact")
        f = s.get("failure")
        lines.append(
            f"| {s['run']} | {s['stop_reason']} | {s.get('stop_class')} | {s['k_sim_rule']} | "
            f"{s['k_raw']} | {'-' if not c else c['kind']} | {'-' if not f else f['primary']} |"
        )
    lines += [
        "",
        "## Rules (fixed a priori)",
        "",
        f"- contact rules (teacher run env): `{json.dumps(rep['contact_rules'])}`",
        f"- episode monitor: `{json.dumps(rep['episode_rules'])}`",
        f"- failure-mode priority: `{json.dumps(rep['failure_modes'])}`",
        f"- classifier thresholds: `{json.dumps(rep['classifier_thresholds'])}`",
        f"- verdict: transfer within {round(100 * TRANSFER_PP)} pp; inconclusive below "
        f"{_pct(INCONCLUSIVE_SIM)} sim completion; re-run above {_pct(FAILSAFE_RERUN_SHARE)} "
        "fail-safe attempts",
        f"- N = {live['attempts']} attempts on one TRAIN track: the interval is wide by design.",
        "",
    ]
    return "\n".join(lines)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def write_report(
    session: str | Path,
    *,
    sim_episodes: int = 32,
    sim: bool = True,
    evaluate: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the report and write ``report.md`` / ``report.json`` (session + track level)."""
    from drone_fly.harness.recording import to_plain

    sdir = Path(session)
    rep = build_report(sdir, sim_episodes=sim_episodes, sim=sim, evaluate=evaluate)
    md = markdown(rep)
    out = {"md": str(sdir / "report.md"), "json": str(sdir / "report.json")}
    (sdir / "report.md").write_text(md, encoding="utf-8")
    (sdir / "report.json").write_text(
        json.dumps(to_plain(rep), indent=2, default=str) + "\n", encoding="utf-8"
    )
    track_dir = sdir.parent.parent
    if sdir.parent.name == "sessions":
        for name in ("report.md", "report.json"):
            shutil.copyfile(sdir / name, track_dir / name)
        out["track_md"] = str(track_dir / "report.md")
    rep["written"] = out
    return rep
