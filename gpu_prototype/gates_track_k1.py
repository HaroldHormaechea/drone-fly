"""Frozen-K1 reservoir on the randomized figure-8/chicane track (turns both ways, mixed per episode)."""
import sys
from gates_k1_reservoir import train
from gates_track import BatchedTrackCourse

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 20e6
    train(N=N, budget=budget, env_cls=BatchedTrackCourse,
          save_path="/workspace/drone-fly/gpu_prototype/gates_track_k1.pt")
