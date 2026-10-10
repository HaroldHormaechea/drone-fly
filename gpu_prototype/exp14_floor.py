"""exp14 -- FLOOR-CRASH SHAPING (Phase A).

exp11 (count curriculum, 67%) has ~8% of its failures as floor crashes (descending into the ground),
the rest multi-obstacle clips. exp14 RESUMES exp11 and adds a potential-based altitude-margin term near
the floor (PROC_FLOOR=1): reward regaining height when low, penalise sinking, capped at FLOOR_CAP so it's
inert at normal altitude (no fly-high bias). Obstacles stay full on-path (count curriculum already done in
exp11 -- no re-ramp). Clearance shaping + terminal stay on. Target: recover some of the 8% without losing
the multi-obstacle skill.

Run: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_FLOOR=1 python exp14_floor.py <save> [budget]
Audit: AUDIT_SEED=1234 ... verify_proc.py <save> 1024  +  diagnose_proc.py <save> 1024
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 14e6
BASE = "/workspace/drone-fly/gpu_prototype/exp11_count.pt"   # resume the 67% winner (same arch, full warm)

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.7, logstd_final=-1.1, anneal_start=0.4,   # mild re-inject, gentle settle
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
