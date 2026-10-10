"""Clearance-shaping experiment launcher: augmented frozen-K1 reservoir (MLP, warm-started from the 96%
proc flier so it starts AGGRESSIVE, not cold/timid) + potential-based clearance shaping (PROC_SHAPING=1)
+ position curriculum + mildly-stochastic annealing (logstd_final=-1.0).

Terminal vs non-terminal contact is set by PROC_OBS_TERMINAL (the two experiments):
  A: PROC_SHAPING=1 PROC_OBS_TERMINAL=1 ...  -> shaping gives a steer-around gradient under the crash penalty
  B: PROC_SHAPING=1 (no terminal) ...        -> no crash penalty (no timidity driver); shaping discourages barging

Usage: PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 [PROC_OBS_TERMINAL=1] python train_shaping.py <save_path> [budget]
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import train_aug

SAVE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 14e6
PROC = "/workspace/drone-fly/training/07-proc-course-flight/model_readout.pt"

train_aug(BatchedProcCourse, EXTRA_DIM, N=384, budget=BUDGET, roll=32, epochs=4, mb=8,
          logstd_init=-0.5, logstd_final=-1.0, anneal_start=0.5, pos_curr_frac=0.4,
          warm_from=PROC, warm_src_extra_dim=30, save_path=SAVE)
