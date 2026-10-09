"""Verify the sparse connectome-actor propagation: equivalence + memory/perf.

Compares the two edge-sparse propagation paths inside ``ConnectomeActorNetwork``:
  * ``scatter`` — the autograd-stable ``index_add`` reference (materialises a dense
    ``(B, E)`` message tensor; the old default).
  * ``sparse``  — the new default (``torch.sparse.mm`` over a precomputed-coalesced COO;
    peak memory scales with the ``(B, N)`` state, not ``batch × edges``).

Both must be numerically identical (< 1e-4 max abs diff) on IDENTICAL obs + IDENTICAL
weights, for the 247-neuron fixture AND the 25,627-neuron K1 slice.
"""

from __future__ import annotations

import gc
import time

import torch

from drone_fly.connectome import load_connectome
from drone_fly.connectome.prune import prune_to_subcircuit
from drone_fly.controller.actor import ConnectomeActorNetwork


def _paired_actors(data, device, seed=0):
    """Two actors (scatter + sparse) with BIT-IDENTICAL weights on ``device``."""
    torch.manual_seed(seed)
    a_scatter = ConnectomeActorNetwork(data, propagation_mode="scatter").to(device)
    a_sparse = ConnectomeActorNetwork(data, propagation_mode="sparse").to(device)
    a_sparse.load_state_dict(a_scatter.state_dict())  # make weights identical
    return a_scatter.eval(), a_sparse.eval()


def equivalence(name, data, device, batch):
    a_scatter, a_sparse = _paired_actors(data, device)
    torch.manual_seed(123)
    obs = torch.randn(batch, 12, device=device)
    with torch.no_grad():
        o_scatter = a_scatter(obs)
        o_sparse = a_sparse(obs)
    diff = (o_scatter - o_sparse).abs().max().item()
    print(f"[equivalence] {name:10s} N={data.neuron_count:6d} E={data.edge_count:8d} "
          f"B={batch:5d} device={str(device):4s} -> max|diff| = {diff:.3e}  "
          f"{'PASS' if diff < 1e-4 else 'FAIL'}")
    del a_scatter, a_sparse, obs, o_scatter, o_sparse
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return diff


def grad_equivalence(name, data, device, batch):
    """Gradients to every learnable param must match between the two paths."""
    a_scatter, a_sparse = _paired_actors(data, device)
    torch.manual_seed(7)
    obs = torch.randn(batch, 12, device=device)
    for a in (a_scatter, a_sparse):
        a.zero_grad()
        a(obs).sum().backward()
    diffs = {}
    for (n1, p1), (n2, p2) in zip(a_scatter.named_parameters(), a_sparse.named_parameters()):
        assert n1 == n2
        if p1.grad is not None and p2.grad is not None:
            diffs[n1] = (p1.grad - p2.grad).abs().max().item()
    worst = max(diffs.values())
    print(f"[grad-equiv ] {name:10s} B={batch:5d} -> worst param-grad max|diff| = {worst:.3e}  "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}  ({', '.join(f'{k}:{v:.1e}' for k,v in diffs.items())})")
    del a_scatter, a_sparse, obs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return worst


def perf_table(name, data, device, mode, batches):
    print(f"\n--- {name} / mode={mode} / {device} ---")
    for B in batches:
        try:
            a = ConnectomeActorNetwork(data, propagation_mode=mode).to(device).eval()
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            obs = torch.randn(B, 12, device=device)
            with torch.no_grad():
                for _ in range(3):
                    a(obs)
                if device == "cuda":
                    torch.cuda.synchronize()
                t = time.time()
                for _ in range(10):
                    a(obs)
                if device == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t) / 10
            peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
            print(f"  fwd      B={B:5d}: peak={peak:7.0f} MB  fps={B/dt:8.0f}")
            del a, obs
        except RuntimeError as e:
            print(f"  fwd      B={B:5d}: OOM/ERROR: {str(e)[:70]}")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def train_fit(name, data, device, mode, B):
    try:
        a = ConnectomeActorNetwork(data, propagation_mode=mode).to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        obs = torch.randn(B, 12, device=device)
        a(obs).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
        print(f"  fwd+bwd  B={B:5d}: peak={peak:7.0f} MB  -> TRAINING FITS")
        del a, obs
    except RuntimeError as e:
        print(f"  fwd+bwd  B={B:5d}: OOM/ERROR: {str(e)[:70]}")
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    fixture = prune_to_subcircuit(load_connectome("tests/fixtures"), k=2)
    k1 = load_connectome("artifacts/pruned/k1")
    has_cuda = torch.cuda.is_available()

    print("=" * 78)
    print("NUMERICAL EQUIVALENCE (scatter reference vs sparse default)")
    print("=" * 78)
    # Fixture: both paths fit anywhere -> check CPU and GPU, several batches.
    equivalence("fixture", fixture, "cpu", 1)
    equivalence("fixture", fixture, "cpu", 64)
    if has_cuda:
        equivalence("fixture", fixture, "cuda", 256)
    grad_equivalence("fixture", fixture, "cpu", 32)

    # K1: scatter OOMs on GPU at batch, so run the equivalence on CPU (small batch).
    equivalence("k1", k1, "cpu", 1)
    equivalence("k1", k1, "cpu", 8)
    grad_equivalence("k1", k1, "cpu", 4)

    if has_cuda:
        print("\n" + "=" * 78)
        print("K1 MEMORY + THROUGHPUT  (GPU)")
        print("=" * 78)
        perf_table("k1", k1, "cuda", "scatter", [256, 1024, 4096])
        perf_table("k1", k1, "cuda", "sparse", [256, 1024, 4096])
        print("\n--- K1 training (fwd+bwd) fit, sparse ---")
        for B in [256, 1024, 4096]:
            train_fit("k1", k1, "cuda", "sparse", B)
        print("\n--- K1 training (fwd+bwd) fit, scatter (old default) ---")
        train_fit("k1", k1, "cuda", "scatter", 256)

        print("\n--- fixture throughput (GPU), before(scatter) vs after(sparse) ---")
        perf_table("fixture", fixture, "cuda", "scatter", [256, 4096])
        perf_table("fixture", fixture, "cuda", "sparse", [256, 4096])
