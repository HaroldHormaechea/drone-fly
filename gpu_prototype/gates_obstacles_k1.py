"""Augmented frozen-K1 reservoir on the oval+obstacles course (brain=flight, readout=flight+obstacle-vision)."""
import sys
from reservoir_aug import train_aug
from gates_obstacles import BatchedObstacleCourse, EXTRA_DIM

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 16e6
    train_aug(BatchedObstacleCourse, EXTRA_DIM, N=N, budget=budget, logstd_final=-2.0, anneal_start=0.4,
              save_path="/workspace/drone-fly/gpu_prototype/gates_obstacles_k1.pt")
