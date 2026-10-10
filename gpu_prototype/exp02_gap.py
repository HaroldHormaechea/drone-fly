"""exp02 -- GAP-STEERING REWARD.

Independent A/B against the baseline (no count curriculum here, so its effect is attributable): resume the
36% baseline (proc_shapeA_term.pt, same-arch warm) and fine-tune with the gap-steering reward
(PROC_GAP=1): a small term rewarding flight along the clearest cone direction that also points toward the
gate, so the drone threads the free gap between multiple obstacles instead of clipping one while dodging
another (84% of baseline failures are obstacle collisions, all on 2+ obstacle courses). Clearance shaping +
terminal contact stay on; mildly-stochastic anneal re-injects exploration.

If both exp01 (count curriculum) and exp02 (gap) beat the baseline honestly, exp03 combines them.

Run: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_GAP=1 python exp02_gap.py <save> [budget]
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
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
