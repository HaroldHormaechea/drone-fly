"""exp16 -- TUNNELS + varied scenery (Phase C).

Trains on the expanded course distribution: every course now also has a fly-THROUGH tunnel/doorway
(PROC_TUNNELS=1 -> 3 boxes: two side walls + a ceiling, clear central lane, terminal contact). Resumes
exp11 (the 67% multi-obstacle avoider) and fine-tunes it to thread tunnels on top of the existing
obstacles. Floor shaping ON (threading a tunnel at gate height makes ground-contact a real risk).
No count curriculum: tunnels are fixed structure, so obstacles are full from the start; warm-start from
exp11 avoids the cold-timidity trap. Mildly-stochastic anneal re-injects exploration to learn the new arc.

Run: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_FLOOR=1 PROC_TUNNELS=1 \
     python exp16_tunnels.py <save> [budget]
Audit (same flags incl PROC_TUNNELS): AUDIT_SEED=1234 ... verify_proc.py <save> 1024 + diagnose_proc.py
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 16e6
BASE = "/workspace/drone-fly/gpu_prototype/exp11_count.pt"   # resume the 67% multi-obstacle avoider

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.5, logstd_final=-1.1, anneal_start=0.5,
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
