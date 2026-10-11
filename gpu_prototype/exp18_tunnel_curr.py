"""exp18 -- TUNNEL GAP CURRICULUM (Phase C improvement).

exp16 navigates tunnels at 28% honest but 63% end in a fatal wall-clip -- it hasn't mastered the tight
gap. Apply the insight that cracked multi-obstacle avoidance (the count curriculum): start EASY and tighten.
Tunnel walls now carry an OUTWARD ovec, so the existing position curriculum (pos_curr_frac) starts the walls
far apart (wide, forgiving gap) + ceiling high, and slides them to the designed tight gap over the first
fraction of training. Resume exp16 (already knows the rough tunnel arc), gentle exploration.

Run: PROC_TRACTABLE=1 PROC_TUNNELS=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_FLOOR=1 \
     python exp18_tunnel_curr.py <save> [budget]
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 18e6
BASE = "/workspace/drone-fly/gpu_prototype/exp16_tunnels.pt"   # resume the 28% tunnel model

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.7, logstd_final=-1.2, anneal_start=0.5,
          pos_curr_frac=0.5,                 # widen->narrow the tunnel gap over the first half
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
