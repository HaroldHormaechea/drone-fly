"""Battery + thrust-at-speed model of the batched plant (UC68 run 2; ADR 0007 D11).

Opt-in twice: the drone profile must carry a ``power`` block (:class:`~drone_fly.drones.schema.
PowerSpec`) AND the plant config must set ``power: true``. Without either, the plant is exactly
the UC67 plant (meteor75 / UC67 parity is untouched).

Fitted on Liftoff telemetry (UC66 sysid + UC69 live; ``drones/<id>/sysid.md``). Per env it carries
the pack charge ``q`` (1 = full), a persistent sag ``s`` (V), the loaded voltage ``V`` and a
latched ``cutoff``. Per substep, with ``T`` the mean body-z thrust (g) the plant is about to apply
and ``v_axial`` the body-z velocity (the flow into the props):

* **available thrust** (what full stick delivers now)::

      inflow  = max(min_factor, 1 - inflow_coeff * inflow_scale * max(v_axial, 0) / v_ref)
      T_avail = max_g * thrust_scale * (V / V_full) ** n_v * inflow

  The applied thrust and the motor state are capped at ``T_avail``; below full stick the command
  is honoured (Liftoff's RPM follows the throttle, independent of the pack voltage);
* **motor effort** ``E = T * inflow ** -effort_inflow_exponent`` (g): the electrical load, in the
  units of the zero-inflow thrust of the motors' RPM (Liftoff's RPM is linear in it; at speed the
  same thrust needs more RPM, but less than ``1 / inflow`` more: exponent fitted on RPM);
* **drain** ``dq/dt = -drain_scale * (drain_idle + drain_load * E ** drain_exponent)``;
* **persistent sag** ``ds/dt = sag_scale * sag_rate * E ** sag_exponent`` (never recovers within
  an episode: Liftoff's pack voltage stays down after a hard push);
* **voltage** ``V = OCV(q) - s - ir_scale * ir_v * E ** ir_exponent``;
* **cutoff**: once ``V < cutoff_v_per_cell * cells`` the motors go to idle for good
  (``T_avail = 0``), Liftoff's low-voltage cutoff (the env ends the episode: ``power_loss``).

The voltage uses the previous substep's effort (no algebraic loop). The per-env randomization
(``drain_scale`` = capacity, ``sag_scale`` and ``ir_scale`` = internal resistance,
``inflow_scale``, ``initial_charge``) has its own generator, so
the UC67 dynamics samples are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

#: Randomizable power parameters (multipliers with nominal 1.0; ``initial_charge`` is a charge).
POWER_RANDOMIZABLE = ("drain_scale", "sag_scale", "ir_scale", "inflow_scale", "initial_charge")


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.clamp(xp[0], xp[-1])
    i = torch.bucketize(x, xp[1:-1].contiguous())
    x0, x1 = xp[i], xp[i + 1]
    return fp[i] + (x - x0) / (x1 - x0) * (fp[i + 1] - fp[i])


class PowerState:
    """Batched battery / available-thrust state (see the module docstring)."""

    def __init__(
        self,
        spec: Any,
        n: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
        randomization_scale: float,
    ) -> None:
        self.spec = spec
        self.n = n
        self.device, self.dtype = device, dtype
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed) + 7919)
        self.rscale = float(randomization_scale)

        def t(v: Any) -> torch.Tensor:
            return torch.as_tensor(v, dtype=dtype, device=device)

        self.ocv_q = t(spec.ocv_charge)
        self.ocv_v = t(spec.ocv_volts)
        self.v_full = float(spec.ocv_volts[-1])
        self.v_cut = float(spec.cutoff_v_per_cell) * int(spec.cells)
        z = torch.zeros(n, dtype=dtype, device=device)
        self.charge = z + 1.0
        self.sag = z.clone()
        self.volts = z + self.v_full
        self.effort = z.clone()
        self.inflow = z + 1.0
        self.cutoff = torch.zeros(n, dtype=torch.bool, device=device)
        self.drain_scale = z + 1.0
        self.sag_scale = z + 1.0
        self.ir_scale = z + 1.0
        self.inflow_scale = z + 1.0
        self.initial_charge = z + 1.0
        self.resample(torch.ones(n, dtype=torch.bool, device=device))
        self.reset(torch.ones(n, dtype=torch.bool, device=device), resample=False)

    # ---- randomization ----
    def _draw(self, key: str, nominal: float) -> torch.Tensor:
        r = self.spec.randomization.get(key)
        if r is None or self.rscale <= 0:
            return torch.full((self.n,), nominal, dtype=self.dtype, device=self.device)
        lo = nominal + (r.min - nominal) * self.rscale
        hi = nominal + (r.max - nominal) * self.rscale
        u = torch.rand(self.n, generator=self.gen, device=self.device, dtype=self.dtype)
        return lo + (hi - lo) * u

    def resample(self, mask: torch.Tensor) -> None:
        for key in POWER_RANDOMIZABLE:
            setattr(self, key, torch.where(mask, self._draw(key, 1.0), getattr(self, key)))

    def reset(self, mask: torch.Tensor, resample: bool = True) -> None:
        if resample:
            self.resample(mask)
        z = torch.zeros_like(self.charge)
        self.charge = torch.where(mask, self.initial_charge, self.charge)
        self.sag = torch.where(mask, z, self.sag)
        self.effort = torch.where(mask, z, self.effort)
        self.inflow = torch.where(mask, z + 1.0, self.inflow)
        self.volts = torch.where(mask, self.ocv(self.charge), self.volts)
        self.cutoff = torch.where(mask, torch.zeros_like(self.cutoff), self.cutoff)

    # ---- model ----
    def ocv(self, q: torch.Tensor) -> torch.Tensor:
        return _interp(q, self.ocv_q, self.ocv_v)

    def available_g(self, max_g: torch.Tensor, v_axial: torch.Tensor) -> torch.Tensor:
        s = self.spec
        inflow = 1.0 - s.inflow_coeff * self.inflow_scale * v_axial.clamp(min=0.0) / s.inflow_v_ref
        self.inflow = inflow.clamp(min=s.inflow_min_factor)
        avail = max_g * self.inflow
        if s.thrust_voltage_exponent:
            avail = avail * (self.volts / self.v_full).clamp(min=0.0) ** s.thrust_voltage_exponent
        return torch.where(self.cutoff, torch.zeros_like(avail), avail)

    def update(self, thrust_g: torch.Tensor, h: torch.Tensor) -> None:
        """Advance ``h`` seconds at mean thrust ``thrust_g`` (g): drain, sag, latch the cutoff."""
        s = self.spec
        E = thrust_g.clamp(min=0.0) * self.inflow ** (-s.effort_inflow_exponent)
        self.effort = E
        rate = s.drain_idle_per_s + s.drain_load_per_s * E**s.drain_exponent
        self.charge = (self.charge - self.drain_scale * rate * h).clamp(min=0.0)
        self.sag = self.sag + self.sag_scale * s.sag_rate_v_per_s * E**s.sag_exponent * h
        self.volts = self.ocv(self.charge) - self.sag - self.ir_scale * s.ir_v * E**s.ir_exponent
        self.cutoff = self.cutoff | (self.volts < self.v_cut)

    def columns(self) -> torch.Tensor:
        """``(n, 2)``: charge fraction, volts per cell (observation / critic features)."""
        return torch.stack([self.charge, self.volts / float(self.spec.cells)], 1)
