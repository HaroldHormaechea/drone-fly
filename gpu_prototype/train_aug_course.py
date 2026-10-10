"""Train the augmented frozen-K1 reservoir ([brain-tap || obs || senses]) on any course.
Usage: train_aug_course.py <lap|track|obstacles> [N] [budget]"""
import os
import sys
from reservoir_aug import train_aug

course = sys.argv[1] if len(sys.argv) > 1 else "lap"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 15e6

warm_from = None
curriculum_frac = None
OVAL = "/workspace/drone-fly/training/oval-lap-flight/model_readout.pt"
if course == "lap":
    from gates_lap import BatchedLapCourse as C; E = 12
elif course == "track":
    from gates_track import BatchedTrackCourse as C; E = 12
elif course == "obstacles":
    from gates_obstacles import BatchedObstacleCourse as C, EXTRA_DIM as E
    # warm-start from the proven oval flight + CURRICULUM: pillars start off-path (loop centre) and
    # slide onto the path over the first half of training, so the policy learns avoidance incrementally
    # instead of hover-trapping on a sudden wall.
    warm_from = OVAL
    curriculum_frac = 0.5
elif course == "pads":
    from gates_pads import BatchedPadCourse as C, EXTRA_DIM as E
    warm_from = OVAL
elif course == "proc":
    from gates_proc_env import BatchedProcCourse as C, EXTRA_DIM as E
    curriculum_frac = 0.6
    if os.environ.get("PROC_NO_OBS", ""):
        # obstacle-free tier: warm-start from the figure-8/chicane flier (obs-only extra, 12 dims)
        warm_from = "/workspace/drone-fly/training/figure8-chicane-flight/model_readout.pt"; warm_src = 12
    else:
        # with-obstacles tier: warm-start from the PROC no-obstacle flier (same 30-dim extra -> full
        # readout copy) so it already flies random courses and only has to LEARN obstacle avoidance.
        warm_from = "/workspace/drone-fly/training/proc-course-flight/model_readout.pt"; warm_src = E
else:
    raise SystemExit(f"unknown course {course!r}")

# Warm-started runs re-inject only MILD exploration (std~0.22) so the competent flying head is
# perturbed gently to learn the new sense (avoidance/recharge) instead of being crashed back to hover.
logstd_init = -1.5 if warm_from is not None else -0.5
warm_src = locals().get("warm_src", 12)
train_aug(C, E, N=N, budget=budget, logstd_final=-2.0, anneal_start=0.4, logstd_init=logstd_init,
          warm_from=warm_from, warm_src_extra_dim=warm_src, curriculum_frac=curriculum_frac,
          save_path=f"/workspace/drone-fly/gpu_prototype/{course}_aug.pt")
