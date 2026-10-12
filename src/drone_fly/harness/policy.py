"""K1 policy runner + the contract action decoding.

Loading goes through the contract's own ``load_policy(path, extra_dim, device)`` (``strict=False``
with the ``body.layer.*``-only missing-keys assertion) — nothing is reimplemented here.

The action contract version comes from the checkpoint's sidecar
(``<stem>.contract.json``, :mod:`drone_fly.sim.action`); a checkpoint without one is v1.
:func:`decode_action` delegates to the sim's NumPy decoders: v1 is the contract §2 mapping
(linear-clamp collective, **not** a sigmoid, and ``tanh * MAX_BODY_RATE`` rates), bit-identical
to the pre-UC67 decoding; v2 is the hover-relative collective. The parity test pins v1 against
``plant_step``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import PolicyRunner
from drone_fly.harness.types import ContractCommand

log = logging.getLogger(__name__)

#: extra widths of the shipped augmented models (contract §4): 12, 30 (course), 70 (+raycast).
SUPPORTED_EXTRA_DIMS = (12, 30, 70)


def validate_extra_dim(extra_dim: Any) -> int:
    if extra_dim is None:
        raise HarnessConfigError(
            "the pure reservoir (extra_dim=None) is not supported by the harness: its K1 path is "
            "hardcoded in gates_k1_reservoir. Use an augmented model (extra_dim 12, 30 or 70)."
        )
    if isinstance(extra_dim, bool) or extra_dim not in SUPPORTED_EXTRA_DIMS:
        raise HarnessConfigError(
            f"extra_dim must be one of {SUPPORTED_EXTRA_DIMS} (it must match the checkpoint), "
            f"got {extra_dim!r}"
        )
    return int(extra_dim)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_action(raw: np.ndarray, contract: Any, spec: Any = None) -> ContractCommand:
    """Decode a raw action with the checkpoint's :class:`~drone_fly.sim.action.ActionSpec`.

    v1 (``spec`` None or version 1): ``collective = clip((a0+1)/2, 0, 1)``, ``thrust =
    collective * MAX_THRUST``, ``rates = tanh(a1:4) * MAX_BODY_RATE`` (bit-identical to the
    pre-UC67 decoding) and ``collective_g`` = the reference thrust curve at ``collective``.
    v2: ``collective_g = clamp(1 + collective_scale_g * a0, 0, max_g)``, ``rates = tanh(a1:4) *
    max_rate_rad_s``; ``collective`` / ``thrust`` are derived on the reference plant.
    """
    from drone_fly.sim.action import LEGACY_VERSION, ActionSpec, decode_numpy

    spec = spec or ActionSpec()
    d = decode_numpy(raw, spec)
    a = np.asarray(raw, dtype=np.float32).reshape(-1)
    return ContractCommand(
        raw=a.copy(),
        collective=d.fraction,
        thrust=float(np.float32(d.fraction) * contract.MAX_THRUST),  # float32, as pre-UC67
        rates=d.rates,
        collective_g=d.collective_g,
        contract_version=LEGACY_VERSION if spec.version == LEGACY_VERSION else spec.version,
    )


class ContractPolicyRunner(PolicyRunner):
    """Loads a stripped ``model_readout.pt`` via the contract and acts deterministically."""

    def __init__(self, contract: Any, checkpoint: str | Path, extra_dim: int) -> None:
        self.contract = contract
        self.checkpoint = Path(checkpoint)
        self._extra_dim = validate_extra_dim(extra_dim)
        self._policy: Any = None
        self.sha256: str | None = None
        #: The checkpoint's action contract (sidecar; v1 when absent), set by :meth:`load`.
        self.action_spec: Any = None

    @property
    def contract_version(self) -> int | None:
        return None if self.action_spec is None else int(self.action_spec.version)

    @property
    def extra_dim(self) -> int:
        return self._extra_dim

    def load(self) -> None:
        if not self.checkpoint.is_file():
            raise HarnessConfigError(f"checkpoint not found: {self.checkpoint}")
        if self.contract.k1_path is None:
            raise HarnessConfigError(
                "the K1 connectome was not found (artifacts/pruned/k1 is absent from this "
                "checkout). Pass --k1-path, set contract.k1_path, or set DRONE_FLY_K1_PATH."
            )
        from drone_fly.sim.action import read_sidecar, spec_for_checkpoint

        try:
            self.action_spec = spec_for_checkpoint(self.checkpoint)
            side = read_sidecar(self.checkpoint) or {}
        except ValueError as exc:
            raise HarnessConfigError(
                f"invalid contract sidecar of {self.checkpoint}: {exc}"
            ) from exc
        if side.get("extra_dim") is not None and int(side["extra_dim"]) != self._extra_dim:
            raise HarnessConfigError(
                f"--extra-dim {self._extra_dim} does not match the checkpoint's sidecar "
                f"extra_dim {side['extra_dim']}"
            )
        log.info(
            "checkpoint %s: action contract v%d%s",
            self.checkpoint.name,
            self.action_spec.version,
            "" if side else " (no .contract.json sidecar: legacy)",
        )
        self.sha256 = sha256_file(self.checkpoint)
        self._policy = self.contract.load_policy(
            str(self.checkpoint), extra_dim=self._extra_dim, device=self.contract.device
        )

    def warm_up(self, passes: int = 3) -> None:
        """Run a few dummy forward passes (both act paths) so CUDA/cuBLAS initialisation happens
        before the first live tick instead of inside it (the first call can take ~0.3 s, enough to
        trip the stale-telemetry fail-safe)."""
        obs = np.zeros(12, np.float32)
        extra = np.zeros(self._extra_dim, np.float32)
        for _ in range(passes):
            self.act(obs, extra)
            self.act_with_brain(obs, extra)

    def act(self, obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("policy not loaded; call load() first")
        return np.asarray(self._policy.act(obs, extra), dtype=np.float32)

    def act_with_brain(self, obs: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, Any]:
        """The raw action plus the full propagated K1 state (UC-65 live overlay).

        Runs exactly the ops ``inference.Policy.act`` runs for an augmented model, in the same
        order and under ``no_grad`` — ``actor.propagated_full`` -> ``actor.tap`` -> ``actor.act``
        — and keeps the ``propagated_full`` output instead of discarding it, so nothing is
        computed twice and the action is bit-identical to :meth:`act`. The brain state is
        returned as the device tensor ``(N,)``; the caller copies it to the host only when it
        publishes it. ``inference.py`` is not modified.
        """
        if self._policy is None:
            raise RuntimeError("policy not loaded; call load() first")
        import torch  # already loaded by the contract

        p = self._policy
        actor = p.actor
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs, np.float32), device=p.device).reshape(1, -1)
            full = actor.propagated_full(o)
            feats = actor.tap(full)
            e = torch.as_tensor(np.asarray(extra, np.float32), device=p.device).reshape(1, -1)
            mean, _ = actor.act(feats, e)
            raw = mean[0].cpu().numpy()
        return np.asarray(raw, dtype=np.float32), full[0]

    def brain_info(self) -> dict[str, Any]:
        """``{n_neurons, tap_index}`` of the loaded K1 actor (``{}`` before :meth:`load`)."""
        if self._policy is None:
            return {}
        actor = self._policy.actor
        return {
            "n_neurons": int(actor.N),
            "tap_index": [int(i) for i in actor.tap_index.cpu().tolist()],
        }
