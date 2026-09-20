"""AC8 — device auto-detection is Apple-Silicon-aware and unit-testable without a GPU.

The whole point of this AC is that the selection logic is checkable on any host by
monkeypatching ``torch.cuda.is_available`` / ``torch.backends.mps.is_available`` — no real
accelerator required. The rule under test (deliberately **not** a naive cuda→mps→cpu):

* explicit override always wins (``cpu`` / ``cuda`` / ``mps``);
* ``mps`` override also sets ``PYTORCH_ENABLE_MPS_FALLBACK=1``;
* auto: CUDA→``cuda``; MPS-available-but-no-opt-in→``cpu`` (the anti-naive regression);
* nothing available→``cpu``.
"""

from __future__ import annotations

import logging

import pytest
import torch

from drone_fly.train import device as device_mod
from drone_fly.train.device import resolve_device


@pytest.fixture
def patch_backends(monkeypatch):
    """Return a setter that fakes cuda/mps availability for the resolver."""

    def _set(*, cuda: bool, mps: bool):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

        class _MpsBackend:
            @staticmethod
            def is_available():
                return mps

        monkeypatch.setattr(torch.backends, "mps", _MpsBackend, raising=False)

    return _set


# --- explicit overrides -------------------------------------------------------------
def test_override_cuda_wins_even_without_hardware(patch_backends) -> None:  # UC-31
    patch_backends(cuda=False, mps=False)
    assert resolve_device("cuda") == "cuda"


def test_override_cpu(patch_backends) -> None:
    patch_backends(cuda=True, mps=True)  # even with everything available
    assert resolve_device("cpu") == "cpu"


def test_override_mps_selects_mps_and_sets_fallback(monkeypatch, patch_backends) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    patch_backends(cuda=False, mps=False)
    assert resolve_device("mps") == "mps"
    import os

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_unknown_override_raises() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_override_is_case_insensitive(patch_backends) -> None:
    patch_backends(cuda=False, mps=False)
    assert resolve_device("CUDA") == "cuda"


# --- auto policy --------------------------------------------------------------------
def test_auto_prefers_cuda_when_available(patch_backends) -> None:  # UC-31
    patch_backends(cuda=True, mps=True)
    assert resolve_device(None) == "cuda"


def test_auto_mps_available_but_no_opt_in_returns_cpu(patch_backends) -> None:
    # THE anti-naive-ordering regression: MPS present, no CUDA, no override → CPU, not mps.
    patch_backends(cuda=False, mps=True)
    assert resolve_device(None) == "cpu"


def test_auto_nothing_available_returns_cpu(patch_backends) -> None:
    patch_backends(cuda=False, mps=False)
    assert resolve_device(None) == "cpu"


def test_apple_silicon_note_is_documented() -> None:
    # AC8 requires a documented Apple-Silicon note; it is surfaced from the module.
    note = device_mod.APPLE_SILICON_NOTE
    assert "MPS" in note
    assert "CPU" in note or "cpu" in note


# --- UC-31: CUDA VRAM/OOM guidance surfaced whenever CUDA is selected (AC-4) ---------
def test_cuda_oom_hint_logged_on_auto_cuda_path(caplog, patch_backends) -> None:  # UC-31
    # Auto-selection lands on CUDA -> the OOM/VRAM hint must be logged.
    patch_backends(cuda=True, mps=False)
    with caplog.at_level(logging.INFO, logger="drone_fly.train.device"):
        assert resolve_device(None) == "cuda"
    assert device_mod.CUDA_OOM_HINT in caplog.text, (
        "CUDA_OOM_HINT must be logged when CUDA is auto-selected"
    )


def test_cuda_oom_hint_logged_on_override_cuda_path(caplog, patch_backends) -> None:  # UC-31
    # Explicit device="cuda" override (even without hardware) -> same hint must be logged.
    patch_backends(cuda=False, mps=False)
    with caplog.at_level(logging.INFO, logger="drone_fly.train.device"):
        assert resolve_device("cuda") == "cuda"
    assert device_mod.CUDA_OOM_HINT in caplog.text, (
        "CUDA_OOM_HINT must be logged when device='cuda' is passed as an override"
    )


def test_cuda_oom_hint_mentions_vram_knobs() -> None:  # UC-31, amended by UC-33
    # AC-4 (UC-31): the hint tells the user which knobs to lower on a CUDA OOM.
    # UC-33 AC-6/AC-8: those knobs must be levers that ACTUALLY exist. ``batch_size`` used to be
    # named here, but it is not a validated train-config key (see the both-directions test below);
    # the real VRAM lever is a smaller pruned slice, so assert that one is present instead.
    hint = device_mod.CUDA_OOM_HINT
    assert "n_envs" in hint
    assert "prune" in hint  # prune / prune_k — a smaller connectome slice, a real VRAM lever
    assert "OOM" in hint or "out-of-memory" in hint.lower()


# --- UC-33 Item 2b (AC-6/AC-8): the OOM hint names only levers that actually exist ---------
def test_cuda_oom_hint_names_only_real_levers() -> None:
    """AC-6/AC-8 (both directions): the corrected message references only actionable levers that
    exist today (``n_envs``, a smaller pruned slice ``prune``/``prune_k``, ``device: cpu``) and
    mentions NEITHER ``batch_size`` NOR ``n_steps`` — neither is a validated YAML train-config key,
    so recommending them pointed users at knobs a train config rejects as unknown."""
    hint = device_mod.CUDA_OOM_HINT
    # (positive) every real lever is named
    assert "n_envs" in hint
    assert "prune" in hint  # covers both "prune" and "prune_k"
    assert "cpu" in hint  # device: cpu fallback
    # (negative) the non-existent knobs are gone, in BOTH directions
    assert "batch_size" not in hint
    assert "n_steps" not in hint
