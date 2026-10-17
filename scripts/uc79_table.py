#!/usr/bin/env python3
"""UC-79 comparison-table reader (AC10): turn the 7 ablation runs' eval JSON into Markdown rows.

Reads, per variant, the UC-77 final_report written by `drone-fly train student`:
  <STATE_ROOT>/<variant>/<profile>/<run_id>/eval/student-held-out.json      (generalisation metric)
  <STATE_ROOT>/<variant>/<profile>/<run_id>/eval/student-liftoff-arena-04.json  (Infinity Loop check)

and prints one Markdown table whose rows are the teacher, the baseline and the 6 variants, and
whose columns are the AC8 metrics: held-out macro_completion and Infinity Loop completion, each with
exact and noise perception, plus the held-out outcome breakdown (floor / collision / out-of-bounds /
gate_lost / timeout). The UC-77 reference row is kept in the doc itself (static), not emitted here.

Usage:
  python scripts/uc79_table.py [STATE_ROOT]          # default artifacts/uc79-ablation

No new CLI subcommand, no training imports — pure JSON reading, runnable anywhere the runs are.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

VARIANTS = ("baseline", "altitude", "floor", "gate_next_next", "gates_full", "surfaces",
            "teacher_full")  # fmt: skip
HELD_OUT = "student-held-out.json"
INFINITY = "student-liftoff-arena-04.json"
# AC8 outcome breakdown (names as drone_fly.sim.multi_course.OUTCOMES), held-out / noise.
OUTCOME_KEYS = ("floor", "collision", "out-of-bounds", "gate_lost", "timeout")


def pct(x: object) -> str:
    return "—" if x is None else f"{100 * float(x):.1f}%"


def newest_run(vroot: Path) -> Path | None:
    """The newest run folder (a dir containing config.yaml) under a variant's state root."""
    if not vroot.is_dir():
        return None
    runs = sorted({p.parent for p in vroot.rglob("config.yaml")})
    return runs[-1] if runs else None


def load(run: Path | None, name: str) -> dict | None:
    if run is None:
        return None
    p = run / "eval" / name
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def macro(ev: dict | None, key: str) -> object:
    return None if ev is None else (ev.get("macro") or {}).get(key)


def outcomes_cell(ev: dict | None) -> str:
    """Held-out, noise-perception outcome shares as floor/collision/oob/gate_lost/timeout."""
    if ev is None:
        return "—"
    oc = (ev.get("outcomes") or {}).get("student_noise") or {}
    return " / ".join(pct(oc.get(k, 0.0)) for k in OUTCOME_KEYS)


def row(label: str, ho: dict | None, il: dict | None, *, teacher: bool = False) -> str:
    if teacher:
        cells = [pct(macro(ho, "teacher_exact")), "—",  # held-out exact(teacher) / noise n/a
                 pct(macro(il, "teacher_exact")), "—", "—"]  # fmt: skip
    else:
        cells = [
            pct(macro(ho, "student_noise")),
            pct(macro(ho, "student_exact")),
            pct(macro(il, "student_noise")),
            pct(macro(il, "student_exact")),
            outcomes_cell(ho),
        ]
    return "| " + " | ".join([label, *cells]) + " |"


def main(argv: list[str]) -> int:
    state_root = Path(argv[1]) if len(argv) > 1 else Path("artifacts/uc79-ablation")
    runs = {v: newest_run(state_root / v) for v in VARIANTS}
    evs = {v: (load(runs[v], HELD_OUT), load(runs[v], INFINITY)) for v in VARIANTS}

    header = ("| Run | Held-out (noise) | Held-out (exact) | Infinity (noise) | Infinity (exact) "
              "| Held-out outcomes noise (floor/coll/oob/lost/timeout) |")
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]

    # Teacher row (teacher_exact is identical across runs; read it from the baseline run).
    b_ho, b_il = evs["baseline"]
    lines.append(row("teacher (exact)", b_ho, b_il, teacher=True))
    for v in VARIANTS:
        ho, il = evs[v]
        lines.append(row(v, ho, il))

    print("\n".join(lines))

    missing = [v for v in VARIANTS if runs[v] is None or evs[v][0] is None]
    if missing:
        print(f"\n> note: no held-out eval yet for: {', '.join(missing)} "
              "(run/finish them via scripts/uc79_ablation.sh)", file=sys.stderr)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
