"""Torch device auto-detection with an Apple-Silicon-aware policy (AC8).

The selection rule is deliberately **not** a naive ``cuda → mps → cpu`` ordering. On the
owner's Apple-Silicon M4 the connectome substrate uses sparse / scatter propagation whose
MPS support is spotty, and PyBullet physics is CPU-bound anyway, so defaulting the tiny
policy to CPU is the sane choice. MPS is therefore **opt-in only**.

Resolution order
----------------
1. **Explicit override wins.** ``resolve_device("cpu"|"cuda"|"mps")`` returns that device
   (validated). ``override="mps"`` additionally sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so
   ops without an MPS kernel fall back to CPU instead of raising.
2. Else if **CUDA** is available → ``"cuda"``.
3. Else if **MPS** is available but there is no opt-in → ``"cpu"`` (deliberate: sparse-op
   MPS gaps + CPU-bound PyBullet make policy-on-CPU the sane M4 default).
4. Else → ``"cpu"``.

The logic reads ``torch.cuda.is_available`` / ``torch.backends.mps.is_available`` so it is
unit-testable on any host by monkeypatching those, without a real GPU.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

VALID_DEVICES = ("cpu", "cuda", "mps")

#: The documented Apple-Silicon note, surfaced in logs and the README (AC8).
APPLE_SILICON_NOTE = (
    "Apple-Silicon note: MPS is never auto-selected. PyTorch's MPS backend has incomplete "
    "sparse-tensor support and the connectome substrate propagates sparsely; PyBullet physics "
    "is CPU-bound and the policy net is tiny, so CPU is the sane default on an M4. To opt into "
    "MPS, pass device='mps' (which also sets PYTORCH_ENABLE_MPS_FALLBACK=1 so unsupported ops "
    "fall back to CPU rather than erroring)."
)

#: CUDA VRAM guidance (UC-31), surfaced in logs, the README, and the Windows setup script.
#: 8 GB (an RTX 3050) is tight for the large connectome slice (~122k neurons), so on a CUDA
#: out-of-memory error the user trades throughput for a smaller VRAM footprint via the config
#: knobs below. Logged whenever CUDA is selected (auto or explicit override).
#: UC-33 (Item 2b): the hint names ONLY levers that actually exist. ``batch_size`` / ``n_steps``
#: are ``TrainConfig`` field defaults, not validated YAML train-config keys, so recommending them
#: pointed users at knobs a train config would reject as unknown. The real VRAM levers are the
#: worker count (``n_envs``), a smaller pruned connectome slice (``prune`` / ``prune_k``), and
#: falling back to CPU (``device: cpu``).
CUDA_OOM_HINT = (
    "CUDA note: 8 GB VRAM is tight for the large connectome slice (~122k neurons). On a CUDA "
    "out-of-memory error, lower n_envs, train a smaller pruned slice (a smaller prune / prune_k "
    "in your train config), and/or fall back to device: cpu; step these down until the run fits, "
    "then tune back up. These levers trade throughput for a smaller VRAM footprint and do not "
    "otherwise change training dynamics."
)


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive; torch always exposes this
        return False


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_device(override: str | None = None) -> str:
    """Return the torch device string per the AC8 policy.

    Parameters
    ----------
    override:
        Explicit device (``"cpu"``, ``"cuda"``, or ``"mps"``), or ``None`` for
        auto-detection. An unknown override raises ``ValueError``.
    """
    if override is not None:
        override = override.lower()
        if override not in VALID_DEVICES:
            raise ValueError(f"device override must be one of {VALID_DEVICES}, got {override!r}.")
        if override == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.info(
                "device='mps' (explicit opt-in); set PYTORCH_ENABLE_MPS_FALLBACK=1. %s",
                APPLE_SILICON_NOTE,
            )
        elif override == "cuda":
            logger.info("device='cuda' (explicit override). %s", CUDA_OOM_HINT)
        else:
            logger.info("device=%r (explicit override).", override)
        return override

    if _cuda_available():
        logger.info("device='cuda' (auto: CUDA available). %s", CUDA_OOM_HINT)
        return "cuda"

    if _mps_available():
        # Deliberate: MPS is available but not opted into -> CPU.
        logger.info("device='cpu' (auto: MPS available but not selected). %s", APPLE_SILICON_NOTE)
        return "cpu"

    logger.info("device='cpu' (auto: no accelerator).")
    return "cpu"
