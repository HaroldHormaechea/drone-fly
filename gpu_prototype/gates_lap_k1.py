"""Frozen-K1 reservoir on the non-linear LAP course (turns + closed loop)."""
import sys
from gates_k1_reservoir import train
from gates_lap import BatchedLapCourse

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 12e6
    train(N=N, budget=budget, env_cls=BatchedLapCourse,
          save_path="/workspace/drone-fly/gpu_prototype/gates_lap_k1.pt")
