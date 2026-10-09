"""Gentler/more-level gate flight: penalize pitch beyond 25deg (trade speed for uprightness)."""
import gates_gpu

gates_gpu.TILT_PEN_W = 2.0
gates_gpu.TILT_PEN_DEG = 25.0

from gates_gpu import AC, BatchedGateCourse  # noqa: E402
from hover_gpu import train  # noqa: E402

if __name__ == "__main__":
    train(AC(), tag="gates_gentle", budget=120e6, env_fn=BatchedGateCourse, ent_coef=0.01,
          save_path="/workspace/drone-fly/gpu_prototype/gates_gentle.pt")
