"""The deployable student (UC70 D3): the user's K1 connectome lineage, plus a comparison baseline.

:class:`K1Student` (``actor: k1-student``) — the frozen K1 reservoir actor
(:class:`~drone_fly.controller.reservoir.K1ReservoirAug`), warm-started from
``training/07-proc-course-flight`` through the UC67 v1 -> v2 conversion:

* **body**: unchanged and frozen (incl. its ``input_projection``). It sees the student
  observation's first 12 values: the contract ``obs`` in student units (Liftoff metres / k for rel
  and velocity; gravity and rates unscaled) — exactly what the harness's ``WorldScale(k)`` gives.
* **readout** ``pi`` over ``[LayerNorm(tap) || extra || enc]``: ``extra`` = the 59 student values
  (the legacy 30 ``harness30`` columns RAW, so they line up with 07's; the other 29 scaled by
  :func:`drone_fly.sim.student_obs.input_scale`), ``enc`` = :attr:`K1Student.encoder` (L1: an MLP
  over the 29 non-legacy senses, 29 -> 64 -> 32, ELU). The warm start copies 07's tap + 30-column
  weights; every new column (the 29 senses and ``enc``) starts at ZERO, so at init the student
  equals 07 on the legacy 30. **Deviation from the plan**: only the readout columns of ``enc`` are
  zeroed, not the encoder's output layer too — with both at zero neither ever gets a gradient.
* **residual** (L2, optional): a 256-128 ELU MLP over the readout input whose LAST layer starts at
  zero, added to ``pi``.
* trainable: ``tap_norm``, ``pi``, ``encoder``, ``residual``, ``log_std``; frozen: the body (incl.
  ``input_projection``) and the reservoir's unused value head.
* **body kernel** (``body_kernel``, recorded in :meth:`K1Student.arch`): the frozen body's
  propagation runs on a CSR copy of its connectome matrix, built once (the weights never change):
  ``csr16`` (fp16 values and states, the default; ~6.5x the COO path on the RTX 3050, max state
  error 2.5e-4), ``csr32`` (exact, ~2.8x) or ``coo`` (the reservoir's own ``propagated_full``).
  Training and the live harness use the same kernel. ``csr16`` falls back to ``csr32`` on the CPU
  (no fp16 sparse kernels there).

:class:`MLPStudent` (``actor: mlp-student``) — a comparison baseline only: 256-256-128 ELU over the
59 scaled values. It is never the deployable export (``train export`` refuses it without
``--baseline-export``).

Both implement ``features(obs)`` (the K1 tap, cached by DAgger; ``(n, 0)`` for the MLP),
``act(features, obs) -> (mean, std)``, ``mean_action(obs)`` and ``readout_state_dict()``.
:class:`StudentPPOAdapter` adds the asymmetric PPO critic (512-512 over teacher obs || critic
extras; never exported).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from drone_fly.controller.mlp_teacher import _mlp
from drone_fly.controller.reservoir import K1ReservoirAug, warm_start
from drone_fly.sim.student_obs import DIM, input_scale

LEGACY = 30
SENSES = DIM - LEGACY  # 29
ENC_HIDDEN, ENC_OUT = 64, 32
RESIDUAL_HIDDEN = (256, 128)
MLP_HIDDEN = (256, 256, 128)
ACT_DIM = 4
STUDENT_KINDS = ("k1-student", "mlp-student")
BODY_KERNELS = ("csr16", "csr32", "coo")


class K1Student(nn.Module):
    """See the module docstring."""

    kind = "k1-student"

    def __init__(
        self,
        k1_path: str | os.PathLike[str] | None = None,
        *,
        connectome: Any = None,
        encoder: bool = True,
        residual: bool = False,
        logstd_init: float = -1.5,
        body_kernel: str = "csr16",
    ) -> None:
        super().__init__()
        if body_kernel not in BODY_KERNELS:
            raise ValueError(f"body_kernel must be one of {BODY_KERNELS}, got {body_kernel!r}")
        self.body_kernel = body_kernel
        self._csr: Any = None
        self._csr_key: Any = None
        self.use_encoder = bool(encoder)
        self.use_residual = bool(residual)
        self.k1 = K1ReservoirAug(DIM + ENC_OUT, k1_path, connectome=connectome)
        for p in self.k1.vf.parameters():
            p.requires_grad_(False)
        self.register_buffer("input_scale", torch.tensor(input_scale(), dtype=torch.float32))
        self.feat_dim = self.k1.feat_dim
        self.encoder = nn.Sequential(nn.Linear(SENSES, ENC_HIDDEN), nn.ELU(),
                                     nn.Linear(ENC_HIDDEN, ENC_OUT), nn.ELU())  # fmt: skip
        self.residual = _mlp(self.feat_dim + DIM + ENC_OUT, RESIDUAL_HIDDEN, ACT_DIM)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        if not self.use_residual:
            for p in self.residual.parameters():
                p.requires_grad_(False)
        if not self.use_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.log_std = self.k1.log_std
        with torch.no_grad():
            self.log_std.fill_(float(logstd_init))

    def warm_start(self, path: str | os.PathLike[str], collective_scale_g: float,
                   *, logstd_init: float | None = None) -> None:  # fmt: skip
        """07 (v1, extra 30) -> this student: legacy columns copied, new columns zero."""
        keep = None if logstd_init is None else float(logstd_init)
        warm_start(self.k1, path, src_extra_dim=LEGACY, legacy_v1_to_scale_g=collective_scale_g)
        with torch.no_grad():
            r = self.feat_dim
            self.k1.pi[0].weight[:, r + LEGACY :].zero_()  # the 29 senses + enc
            if keep is not None:
                self.log_std.fill_(keep)

    # ---- the frozen body ----
    def __getstate__(self) -> dict[str, Any]:
        """The CSR cache is derived (rebuilt on first use): never pickled or deep-copied."""
        state = self.__dict__.copy()
        state["_csr"] = None
        state["_csr_key"] = None
        return state

    def __deepcopy__(self, memo: dict[int, Any]) -> K1Student:
        import copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__getstate__().items():
            new.__dict__[k] = copy.deepcopy(v, memo)
        return new

    def kernel_in_use(self, device: Any) -> str:
        if self.body_kernel == "csr16" and torch.device(device).type != "cuda":
            return "csr32"
        return self.body_kernel

    def _csr_matrix(self, device: Any, dtype: torch.dtype) -> torch.Tensor:
        key = (str(device), dtype)
        if self._csr is None or self._csr_key != key:
            layer = self.k1.body.layer
            with torch.no_grad():
                w = layer.effective_weight().detach().index_select(0, layer._edge_order)
                n = layer.n_neurons
                coo = torch.sparse_coo_tensor(layer._coalesced_edge_index, w, size=(n, n))
                csr = coo.coalesce().to_sparse_csr()
                self._csr = torch.sparse_csr_tensor(
                    csr.crow_indices(), csr.col_indices(), csr.values().to(dtype), size=(n, n)
                ).to(device)
            self._csr_key = key
        return self._csr

    @torch.no_grad()
    def propagate(self, obs12: torch.Tensor) -> torch.Tensor:
        """The frozen body's full propagated state ``(B, N)`` (float32) for the 12-value obs."""
        kernel = self.kernel_in_use(obs12.device)
        if kernel == "coo":
            return self.k1.propagated_full(obs12)
        body = self.k1.body
        x = obs12.to(torch.float32)
        projected = body.input_projection(x)
        state = torch.zeros(x.shape[0], self.k1.N, dtype=torch.float32, device=x.device)
        state = state.index_add(1, body.sensory_index, projected)
        dtype = torch.float16 if kernel == "csr16" else torch.float32
        m = self._csr_matrix(x.device, dtype)
        y = state.t().contiguous().to(dtype)
        for _ in range(body.layer.n_steps):
            y = torch.tanh(m @ y)
        return y.t().float()

    # ---- the actor protocol ----
    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.k1.tap(self.propagate(obs[:, :12]))

    def readout_input(self, feats: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(feats.dtype if feats.dtype.is_floating_point else torch.float32)
        senses = obs[:, LEGACY:] * self.input_scale[LEGACY:]
        if self.use_encoder:
            enc = self.encoder(senses)
        else:
            enc = senses.new_zeros(senses.shape[0], ENC_OUT)
        return torch.cat([self.k1.tap_norm(feats.float()), obs[:, :LEGACY], senses, enc], -1)

    def act(self, feats: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.readout_input(feats, obs)
        mean = self.k1.pi(x)
        if self.use_residual:
            mean = mean + self.residual(x)
        return mean, self.log_std.exp().expand_as(mean)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(self.features(obs), obs)[0]

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        """Everything but the rebuildable ``k1.body.layer.*`` (the shipped format)."""
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if not k.startswith("k1.body.layer.")}  # fmt: skip

    def trainable_names(self) -> list[str]:
        return [k for k, p in self.named_parameters() if p.requires_grad]

    def frozen_names(self) -> list[str]:
        return [k for k, p in self.named_parameters() if not p.requires_grad]

    def arch(self) -> dict[str, Any]:
        return {"kind": self.kind, "encoder": self.use_encoder, "residual": self.use_residual,
                "body_kernel": self.body_kernel,
                "feat_dim": self.feat_dim, "enc": [SENSES, ENC_HIDDEN, ENC_OUT],
                "residual_hidden": list(RESIDUAL_HIDDEN)}  # fmt: skip


class MLPStudent(nn.Module):
    """The comparison baseline (see the module docstring)."""

    kind = "mlp-student"

    def __init__(self, *, logstd_init: float = -1.5) -> None:
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(input_scale(), dtype=torch.float32))
        self.pi = _mlp(DIM, MLP_HIDDEN, ACT_DIM)
        nn.init.zeros_(self.pi[-1].bias)
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), float(logstd_init)))
        self.feat_dim = 0

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.new_zeros(obs.shape[0], 0)

    def act(self, feats: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.pi(obs.float() * self.input_scale)
        return mean, self.log_std.exp().expand_as(mean)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act(self.features(obs), obs)[0]

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def trainable_names(self) -> list[str]:
        return [k for k, p in self.named_parameters() if p.requires_grad]

    def frozen_names(self) -> list[str]:
        return []

    def arch(self) -> dict[str, Any]:
        return {"kind": self.kind, "hidden": list(MLP_HIDDEN)}


def build_student(kind: str, *, k1_path: Any = None, connectome: Any = None,
                  encoder: bool = True, residual: bool = False,
                  logstd_init: float = -1.5, body_kernel: str = "csr16") -> nn.Module:  # fmt: skip
    if kind == "k1-student":
        return K1Student(k1_path, connectome=connectome, encoder=encoder, residual=residual,
                         logstd_init=logstd_init, body_kernel=body_kernel)  # fmt: skip
    if kind == "mlp-student":
        return MLPStudent(logstd_init=logstd_init)
    raise ValueError(f"unknown student actor {kind!r} ({STUDENT_KINDS})")


def student_from_state(kind: str, arch: dict[str, Any], state: dict[str, torch.Tensor],
                       *, k1_path: Any = None, connectome: Any = None) -> nn.Module:  # fmt: skip
    """Rebuild a student from an exported / saved readout state (``body.layer.*`` rebuilt)."""
    m = build_student(kind, k1_path=k1_path, connectome=connectome,
                      encoder=bool(arch.get("encoder", True)),
                      residual=bool(arch.get("residual", False)),
                      body_kernel=str(arch.get("body_kernel", "coo")))  # fmt: skip
    missing, unexpected = m.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("k1.body.layer.")]
    if bad or unexpected:
        raise ValueError(f"bad student checkpoint: missing={bad} unexpected={list(unexpected)}")
    return m


class StudentPPOAdapter(nn.Module):
    """A student + the asymmetric critic (teacher obs || critic extras; never exported)."""

    def __init__(self, student: nn.Module, critic_in: int, critic_scale: Any = None) -> None:
        super().__init__()
        self.student = student
        self.vf = _mlp(critic_in, (512, 512), 1)
        cs = torch.ones(critic_in) if critic_scale is None else torch.as_tensor(critic_scale)
        self.register_buffer("critic_scale", cs.float())
        self.log_std = student.log_std
        self.feat_dim = int(getattr(student, "feat_dim", 0))

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.student.features(obs)

    def act(self, feats: torch.Tensor, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student.act(feats, obs)

    def value(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.vf(critic_input.float() * self.critic_scale).squeeze(-1)

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        return self.student.readout_state_dict()
