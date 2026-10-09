"""Train the augmented frozen-K1 reservoir ([brain-tap || obs || senses]) on any course.
Usage: train_aug_course.py <lap|track|obstacles> [N] [budget]"""
import sys
from reservoir_aug import train_aug

course = sys.argv[1] if len(sys.argv) > 1 else "lap"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 15e6

if course == "lap":
    from gates_lap import BatchedLapCourse as C; E = 12
elif course == "track":
    from gates_track import BatchedTrackCourse as C; E = 12
elif course == "obstacles":
    from gates_obstacles import BatchedObstacleCourse as C, EXTRA_DIM as E
else:
    raise SystemExit(f"unknown course {course!r}")

train_aug(C, E, N=N, budget=budget, logstd_final=-2.0, anneal_start=0.4,
          save_path=f"/workspace/drone-fly/gpu_prototype/{course}_aug.pt")
