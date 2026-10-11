"""exp17 -- MULTI-LAP tier (Phase B).

exp11 already flies 2-4 lap courses at 31% honest (vs 67% on 1-2 laps) -- compounding crash-risk over
3 laps is the limiter, not a missing skill. exp17 RESUMES exp11 and fine-tunes on the multi-lap pool
(PROC_MULTILAP=1 -> 2-4 laps) to build endurance.

LESSON FROM exp14 (floor-shaping regressed when restarted at logstd -0.7): a fine-tune of a converged
model must use LOW exploration. Multi-lap is an endurance task, not a new skill, so start near exp11's
converged logstd (-1.0) and settle gently -- do NOT re-inject noise that would knock it off its optimum.

Run: PROC_TRACTABLE=1 PROC_MULTILAP=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python exp17_multilap.py <save> [budget]
Audit: AUDIT_SEED=1234 ... (same flags) verify_proc.py <save> 1024
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 16e6
BASE = "/workspace/drone-fly/gpu_prototype/exp11_count.pt"

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-1.0, logstd_final=-1.3, anneal_start=0.2,   # LOW exploration: gentle endurance fine-tune
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
