"""exp11 -- OBSTACLE-COUNT CURRICULUM.

Phase-0 diagnosis (corrected): 84% of the 36% baseline's failures are obstacle collisions, and 100% of
those occur on courses with 2+ obstacles. The model handles a single in-path obstacle (what the position
curriculum taught) but clips one while dodging another when 2+ are present.

exp11 RESUMES the converged baseline (proc_shapeA_term.pt -- warm_start copies input_projection + body +
readout, same architecture) and fine-tunes it with the new COUNT curriculum: active obstacles grow 1 -> all
over the first half of training (count_curr_frac=0.5). Obstacles stay full on-path (NO position curriculum
re-run -- the baseline already trained through it; re-sliding them off-path would regress it). Clearance
shaping + terminal contact stay on (the honest-avoidance recipe). Mildly-stochastic anneal re-injects a
little exploration so it can discover multi-obstacle arcs.

Run: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python exp11_count-curriculum.py <save> [budget]
Audit: verify_proc.py <save> 512  (honesty gate)  +  diagnose_proc.py <save> 512  (did multi-obstacle improve?)
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 14e6
BASE = "/workspace/drone-fly/gpu_prototype/proc_shapeA_term.pt"   # the 36% baseline (full ckpt, input_projection intact)

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.5, logstd_final=-1.0, anneal_start=0.5,
          count_curr_frac=0.5,                 # grow active obstacles 1 -> all over first half
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,   # same-arch resume (full copy)
          save_path=SAVE)
