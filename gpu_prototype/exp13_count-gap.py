"""exp13 -- COUNT CURRICULUM + GAP-STEERING (combination).

exp11 (count curriculum) jumped honest deterministic to ~59% (from 36%); exp12 (gap-steering) alone was
modest (~44% EMA). The two are orthogonal and compose cleanly: count curriculum is a training SCHEDULE
(grow active obstacles 1->N), gap-steering is a REWARD term that rewards flying along the clearest
gate-ward cone direction. The gap reward reads obstacles through the same _omask() the curriculum caps, so
during the ramp it only considers active obstacles -- consistent by construction.

Clean ablation: resume the SAME baseline (proc_shapeA_term.pt) as exp11/exp12 with BOTH on, so the result
is directly comparable -- does gap-steering add anything on top of the count curriculum?

Run: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_GAP=1 python exp13_count-gap.py <save> [budget]
Audit: verify_proc.py <save> 512  +  diagnose_proc.py <save> 512
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 14e6
BASE = "/workspace/drone-fly/gpu_prototype/proc_shapeA_term.pt"

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.5, logstd_final=-1.0, anneal_start=0.5,
          count_curr_frac=0.5,                 # count curriculum (exp11)
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,   # + PROC_GAP=1 at runtime (exp12)
          save_path=SAVE)
