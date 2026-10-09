"""Gate course with the CONNECTOME actor (the project's point) on the GPU sim.

Reuses the ConnectomeAC (ConnectomeActorNetwork features + pi head) and the BatchedGateCourse task.
If this flies the gates deterministically + controlled, the connectome learns real racing control at
scale — the end-to-end validation the whole campaign was after.
"""
from hover_conn import ConnectomeAC
from hover_gpu import train
from gates_gpu import BatchedGateCourse

if __name__ == "__main__":
    train(ConnectomeAC(), tag="gates_conn", N=2048, budget=160e6, env_fn=BatchedGateCourse,
          ent_coef=0.01, save_path="/workspace/drone-fly/gpu_prototype/gates_conn.pt")
