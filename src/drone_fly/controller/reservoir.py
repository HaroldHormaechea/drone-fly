"""The augmented frozen-K1 reservoir actor (promoted from ``gpu_prototype/reservoir_aug.py``).

The frozen K1 body (``ConnectomeActorNetwork``: input projection -> real-weight sparse
propagation) processes the 12-dim flight obs; a small trainable readout sees
``[LayerNorm(tap(R neurons)) || extra(E)]`` and the value head sees ``extra``. Parameter and
buffer names are the prototype's (``body.*``, ``tap_index``, ``tap_norm``, ``pi``, ``vf``,
``log_std``), so every shipped ``model_readout.pt`` (stripped of the rebuildable
``body.layer.*``) loads here with the contract's ``strict=False`` / ``body.layer.*``-only rule.

The K1 connectome path is explicit or resolved by
:func:`drone_fly.harness.contract.resolve_k1_path` (``$DRONE_FLY_K1_PATH`` ->
``<repo>/artifacts/pruned/k1``); it is never hardcoded.

The collective output's readout bias starts at hover for the action contract in use
(:meth:`drone_fly.sim.action.ActionSpec.collective_bias`: v1 ``2 * hover - 1`` of the reference
plant, v2 ``0``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from drone_fly.controller.actor import ConnectomeActorNetwork

log = logging.getLogger(__name__)

R = 1024  # reservoir readout width (neurons tapped from the propagated state)
HID = 128
INIT_LOGSTD = -0.5
ACTOR_KIND = "k1-reservoir-aug"


def resolve_k1(k1_path: str | os.PathLike[str] | None) -> Path:
    from drone_fly.harness.contract import resolve_k1_path
    from drone_fly.harness.errors import HarnessConfigError

    p = resolve_k1_path(k1_path)
    if p is None:
        raise HarnessConfigError(
            "the K1 connectome was not found (artifacts/pruned/k1 is absent from this checkout); "
            "pass --k1-path or set DRONE_FLY_K1_PATH"
        )
    return p


class K1ReservoirAug(nn.Module):
    """Frozen K1 body + readout over ``[normalised brain tap || extra senses]``."""

    def __init__(
        self,
        extra_dim: int,
        k1_path: str | os.PathLike[str] | None = None,
        *,
        connectome: Any = None,
        collective_bias: float = 0.0,
        obs_dim: int = 12,
        act_dim: int = 4,
    ) -> None:
        super().__init__()
        from drone_fly.connectome import load_connectome

        self.extra_dim = int(extra_dim)
        data = connectome if connectome is not None else load_connectome(resolve_k1(k1_path))
        body = ConnectomeActorNetwork(data)
        for p in body.parameters():
            p.requires_grad_(False)
        body.eval()
        self.body = body
        self.N = body.n_neurons
        spread = torch.linspace(0, self.N - 1, R).long()
        tap = torch.unique(torch.cat([body.motor_index.cpu(), spread]))[:R]
        self.register_buffer("tap_index", tap)
        self.feat_dim = int(tap.shape[0])
        self.tap_norm = nn.LayerNorm(self.feat_dim)
        self.pi = nn.Sequential(
            nn.Linear(self.feat_dim + self.extra_dim, HID), nn.Tanh(), nn.Linear(HID, act_dim)
        )
        self.vf = nn.Sequential(
            nn.Linear(self.extra_dim, HID),
            nn.Tanh(),
            nn.Linear(HID, HID),
            nn.Tanh(),
            nn.Linear(HID, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) + INIT_LOGSTD)
        with torch.no_grad():
            self.pi[-1].bias[0] = float(collective_bias)

    @torch.no_grad()
    def propagated_full(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.to(torch.float32)
        projected = self.body.input_projection(x)
        state = torch.zeros(x.shape[0], self.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, self.body.sensory_index, projected)
        return self.body.layer(state)

    def tap(self, propagated: torch.Tensor) -> torch.Tensor:
        return propagated.index_select(1, self.tap_index)

    @torch.no_grad()
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.tap(self.propagated_full(obs))

    def act(
        self, feats_tap: torch.Tensor, extra: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.pi(torch.cat([self.tap_norm(feats_tap), extra.to(feats_tap.dtype)], dim=-1))
        return mean, self.log_std.exp()

    def value(self, extra: torch.Tensor) -> torch.Tensor:
        return self.vf(extra.to(torch.float32)).squeeze(-1)

    def forward(self, obs: torch.Tensor, extra: torch.Tensor) -> tuple[torch.Tensor, ...]:
        feats = self.features(obs)
        mean, std = self.act(feats, extra)
        return mean, std, self.value(extra)

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        """The state dict without the rebuildable ``body.layer.*`` (the shipped format)."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("body.layer.")}


def load_state_strict_false(actor: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """The contract's load rule: only ``body.layer.*`` may be missing, nothing unexpected."""
    missing, unexpected = actor.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("body.layer.")]
    if bad or unexpected:
        raise ValueError(f"bad checkpoint: missing={bad} unexpected={list(unexpected)}")


def load_readout(
    path: str | os.PathLike[str],
    extra_dim: int,
    k1_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    *,
    connectome: Any = None,
) -> K1ReservoirAug:
    """Build a :class:`K1ReservoirAug` and load a stripped ``model_readout.pt`` into it."""
    actor = K1ReservoirAug(extra_dim, k1_path, connectome=connectome).to(device)
    load_state_strict_false(actor, torch.load(path, map_location=device))
    actor.eval()
    return actor


def v1_to_v2_collective(
    weight: torch.Tensor, bias: torch.Tensor, collective_scale_g: float, reference: Any = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-express a v1 collective output row as v2 (affine on the reference plant).

    v1: ``g = ref_max_g * (a0 + 1) / 2`` (meteor75: ``1.25 * a0 + 1.25``) on a linear reference
    thrust curve; v2: ``g = 1 + collective_scale_g * a0'``. So
    ``a0' = (ref_max_g / 2 * a0 + ref_max_g / 2 - 1) / collective_scale_g``, exact inside the
    v1 clip range.
    """
    from drone_fly.sim.action import legacy_reference

    ref = reference or legacy_reference()
    if len(ref.knots_g) != 2 or ref.knots_g[0] != 0.0:
        raise ValueError("v1 -> v2 warm start needs a linear reference thrust curve")
    half = ref.max_g / 2.0
    s = float(collective_scale_g)
    return weight * (half / s), (bias * half + half - 1.0) / s


def warm_start(
    ac: K1ReservoirAug,
    path: str | os.PathLike[str],
    src_extra_dim: int,
    *,
    legacy_v1_to_scale_g: float | None = None,
) -> None:
    """Initialise ``ac`` from a readout trained with FEWER (or the same) extra senses.

    The readout input is ``[tap(R) || extra]``: the source's first ``R + src_extra_dim`` input
    columns are copied and the NEW sense columns start at zero influence; the value head
    (input = extra) likewise. With ``legacy_v1_to_scale_g`` the source is a v1 model and its
    collective output row is converted to v2 with that ``collective_scale_g``.
    """
    src = torch.load(path, map_location="cpu")
    tgt = ac.state_dict()
    r = ac.feat_dim
    for k, v in src.items():
        if k not in tgt:
            continue
        tv = tgt[k]
        if v.shape == tv.shape:
            tv.copy_(v)
        elif k == "pi.0.weight":  # (HID, R + extra): copy [tap || src extra], zero the new
            tv.zero_()
            tv[:, : r + src_extra_dim].copy_(v)
        elif k == "vf.0.weight":  # (HID, extra)
            tv.zero_()
            tv[:, :src_extra_dim].copy_(v)
    if legacy_v1_to_scale_g is not None:
        w, b = v1_to_v2_collective(tgt["pi.2.weight"][0], tgt["pi.2.bias"][0], legacy_v1_to_scale_g)
        tgt["pi.2.weight"][0] = w
        tgt["pi.2.bias"][0] = b
    ac.load_state_dict(tgt)
    log.info("warm-started from %s (src_extra_dim=%d)", path, src_extra_dim)
