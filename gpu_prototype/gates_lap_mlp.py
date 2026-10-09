"""Validate the non-linear LAP course is learnable with a fast MLP (before the K1 reservoir run)."""
import sys
from hover_gpu import train
from gates_gpu import AC
from gates_lap import BatchedLapCourse

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 8e6
    train(AC(), tag=f"lap_mlp_N{N}", N=N, budget=budget, env_fn=BatchedLapCourse, ent_coef=0.01,
          roll=32, mb=8, epochs=4, save_path="/workspace/drone-fly/gpu_prototype/gates_lap_mlp.pt")
