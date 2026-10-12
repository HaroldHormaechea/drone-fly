"""Batched quaternion helpers (``[w, x, y, z]``, body -> world), ported from ``hover_gpu.py``.

``quat_mul`` / ``quat_rotate`` / ``quat_rotate_inv`` are the prototype's exact formulas (the
Meteor75 equivalence test depends on it). ``quat_exp_step`` is the exponential-map attitude
step of the ``exact`` integrator: ``q <- q * exp(omega * h / 2)``.
"""

from __future__ import annotations

import torch


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (body) into the world frame."""
    qw = q[..., :1]
    qv = q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2 * (qw * uv + uuv)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return qc


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` (world) into the body frame."""
    return quat_rotate(quat_conj(q), v)


def quat_euler_step(q: torch.Tensor, omega: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``hover_gpu``'s first-order step ``q + 0.5 q * [0, omega] h``, then normalised."""
    wq = torch.cat([torch.zeros_like(omega[..., :1]), omega], dim=-1)
    q = q + 0.5 * quat_mul(q, wq) * h
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)


def quat_exp_step(q: torch.Tensor, omega: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Exact rotation by a constant body rate ``omega`` over ``h``: ``q * exp(omega h / 2)``."""
    n = omega.norm(dim=-1, keepdim=True)
    half = 0.5 * n * h
    # sin(half) / n -> h / 2 as n -> 0 (no division by zero)
    safe = torch.where(n > 1e-12, n, torch.ones_like(n))
    s = torch.where(n > 1e-12, torch.sin(half) / safe, 0.5 * h * torch.ones_like(n))
    dq = torch.cat([torch.cos(half), omega * s], dim=-1)
    q = quat_mul(q, dq)
    return q / q.norm(dim=-1, keepdim=True)


def gravity_body(q: torch.Tensor) -> torch.Tensor:
    """The world ``[0, 0, -1]`` (down) in the body frame — the obs ``g_body`` block."""
    down = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    down[..., 2] = -1.0
    return quat_rotate_inv(q, down)


def geodesic_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (degrees) of the rotation taking ``a`` to ``b``."""
    dot = (a * b).sum(-1).abs().clamp(max=1.0)
    return torch.rad2deg(2.0 * torch.arccos(dot))
