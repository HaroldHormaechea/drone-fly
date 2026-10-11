"""exp19 -- TWO-STAGE GRU, STAGE 1: aggressive obstacle-free flight with the recurrent head (Phase D).

The first GRU attempt collapsed to timid near-hover because a fresh recurrent head trained cold under a
terminal crash penalty finds caution the safe local optimum. Fix = two stages. STAGE 1 (this file): train
the GRU OBSTACLE-FREE (PROC_NO_OBS=1) so the recurrent head itself learns aggressive gate flight (not just
the frozen body) -- no crash-penalty timidity trap because there are no obstacles. Warm the frozen brain +
tap from the 96% obstacle-free flier (07-proc-course-flight). Output gru_stage1.pt feeds STAGE 2, which adds
obstacles + shaping + terminal and resumes from this aggressive recurrent policy.

Run: PROC_TRACTABLE=1 PROC_NO_OBS=1 PROC_RAYCAST=1 python exp19_gru_s1.py [N] [budget]
"""
import sys
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_gru import train_gru

N = int(sys.argv[1]) if len(sys.argv) > 1 else 384
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 14e6
PROC07 = "/workspace/drone-fly/training/07-proc-course-flight/model_readout.pt"
SAVE = "/workspace/drone-fly/gpu_prototype/gru_stage1.pt"

train_gru(BatchedProcCourse, EXTRA_DIM, N=N, budget=BUDGET, roll=32, chunk=16, epochs=3, mb_env=4,
          logstd_init=-0.5, logstd_final=-1.0, anneal_start=0.5,
          pos_curr_frac=1.0,             # no obstacles in stage 1, so this is a harmless no-op
          warm_from=PROC07, save_path=SAVE)
