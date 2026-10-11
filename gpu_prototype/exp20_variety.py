"""exp20 -- VARIED OBSTACLE SHAPES (Phase C): fly-under overhangs + tall walls.

Adds two new shape challenges beyond pillars/blocks/tunnels (PROC_VARIETY=1): a fly-UNDER overhang (duck
low under a horizontal bar -- the vertical inverse of the fly-over slab) and a tall WALL (a wider lateral
detour than a thin pillar). A new capability, so -- per the campaign lesson (new capabilities gain, re-tunes
regress) -- resume exp11 (the 67% avoider) and learn the new shapes, with a position curriculum
(pos_curr_frac) easing them in (overhang lowers, wall slides onto the lane) and moderate exploration.

Run: PROC_TRACTABLE=1 PROC_VARIETY=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 PROC_FLOOR=1 \
     python exp20_variety.py <save> [budget]
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 16e6
BASE = "/workspace/drone-fly/gpu_prototype/exp11_count.pt"

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.6, logstd_final=-1.1, anneal_start=0.5,
          pos_curr_frac=0.5,              # ease the new shapes in (overhang lowers, wall slides onto lane)
          warm_from=BASE, warm_src_extra_dim=EXTRA_DIM,
          save_path=SAVE)
