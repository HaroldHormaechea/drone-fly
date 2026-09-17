"""AC2 (the #1 back-compat test) — the recording sink is non-invasive.

Exposing the policy's per-step neuron state must NOT alter the forward numerics or the
gradients. This is the hard back-compat contract UC-05 adds to UC-01..04's actor:

* the actor's forward output is **bit-identical** with the sink off vs on (same obs +
  seed; ``torch.equal``);
* the captured array is a **real clone** (an independent copy that shares no storage with
  the live tensor) — an in-place op on the recording can never corrupt the forward output;
* gradients are **unchanged** by capture (the sink detaches);
* the sink defaults to ``None`` (recording is off by default).

UC-06 adds a second back-compat contract on the **recorded file**: the new ``meta.course``
block is additive, so a recording made without a course (a UC-05-era file) stays schema-
compatible — it carries every field the viewer reads and simply lacks the optional course
block (the viewer degrades gracefully, omitting the markers it lacks; AC8).

Hermetic: pure-torch / direct-recorder on the committed fixture, no env / checkpoint / network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.record.recorder import ActivationRecorder


def test_sink_defaults_to_none(connectome: ConnectomeData) -> None:
    """Recording is off by default — a fresh actor has no sink (AC1/AC2)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    assert net.sink is None


def test_forward_bit_identical_sink_off_vs_on(connectome: ConnectomeData) -> None:
    """Same actor + obs: enabling the sink does not change the output by one bit (AC2)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    net.eval()
    obs = torch.randn(OBS_DIM)

    with torch.no_grad():
        out_off = net(obs).clone()

    captured: list[np.ndarray] = []
    net.sink = captured.append
    with torch.no_grad():
        out_on = net(obs).clone()

    assert torch.equal(out_off, out_on)
    assert len(captured) == 1  # exactly one capture per forward
    # And the captured activation is the full post-propagation neuron state.
    assert captured[0].shape[-1] == connectome.neuron_count


def test_captured_array_is_an_independent_clone(connectome: ConnectomeData) -> None:
    """The recorded array is a decoupled copy — mutating it can't corrupt the policy.

    ``actor.forward`` hands the sink ``propagated.detach().cpu().clone().numpy()``. The
    mandatory ``clone()`` is what matters: the captured array shares no storage with the
    live post-propagation tensor used for the motor readout, so an in-place write on the
    recording can never perturb the forward output or a later capture.
    """
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    net.eval()

    captured: list[np.ndarray] = []
    net.sink = captured.append
    obs = torch.randn(OBS_DIM)

    with torch.no_grad():
        out_before = net(obs).clone()
    arr0 = captured[0]
    assert arr0.shape[-1] == connectome.neuron_count

    # A second forward on the SAME obs is deterministic -> same captured values...
    with torch.no_grad():
        net(obs)
    arr1 = captured[1]
    assert np.array_equal(arr0, arr1)

    # ...yet each capture is an independent clone: mutating the first leaves the second
    # (and the earlier forward output) untouched — no shared storage.
    arr0[...] = 12345.0
    assert not np.array_equal(arr0, arr1)

    # And an in-place op on the recording can't corrupt a subsequent forward's output.
    with torch.no_grad():
        out_after = net(obs)
    assert torch.equal(out_before, out_after)


def test_gradients_unaffected_by_capture(connectome: ConnectomeData) -> None:
    """The detached sink leaves the backward pass bit-identical (AC2)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    obs = torch.randn(2, OBS_DIM)

    net.zero_grad()
    net(obs).sum().backward()
    grads_off = {
        name: p.grad.detach().clone() for name, p in net.named_parameters() if p.grad is not None
    }

    captured: list[np.ndarray] = []
    net.sink = captured.append
    net.zero_grad()
    net(obs).sum().backward()
    grads_on = {
        name: p.grad.detach().clone() for name, p in net.named_parameters() if p.grad is not None
    }

    assert grads_off.keys() == grads_on.keys()
    assert grads_off  # sanity: gradients actually flowed
    for name in grads_off:
        assert torch.equal(grads_off[name], grads_on[name]), name
    assert len(captured) == 1  # capture happened, yet grads are identical


# --- UC-06 AC8: a course-less recording stays schema-compatible ------------------------
# The viewer reads these from every recording regardless of course geometry; a UC-05-era
# file (no meta.course) must still carry them so the viewer loads it and just omits markers.
_VIEWER_REQUIRED_META = {
    "neuron_ids",
    "roles",
    "positions",
    "n_frames",
    "n_neurons",
    "action_layout",
    "activation_scale",
    "activation_offset",
}


def test_recording_without_course_is_backcompatible(
    connectome: ConnectomeData, tmp_path: Path
) -> None:
    """A recording made with no course omits ``meta.course`` yet stays a valid viewer file.

    This is the file-side of AC8's back-compat guarantee: the new block is purely additive,
    so an older-style recording (course=None, the default) carries the full pre-UC-06 schema
    the three synced panels consume — the viewer degrades gracefully, omitting the markers it
    cannot place rather than failing to load.
    """
    rec = ActivationRecorder(connectome, tmp_path / "act", backend="simple", dt=0.05)
    n = connectome.neuron_count
    rec.start_episode(0, seed=0)
    for f in range(3):
        rec.sink(np.linspace(-1.0, 1.0, n, dtype=np.float32) * (f + 1) / 3)
        rec.capture_frame(np.zeros(ACTION_DIM), np.array([float(f), 0.0, 1.0]))
    path = rec.finish_episode(completed=False, completion_time=None, total_reward=-0.5, steps=3)

    doc = json.loads(path.read_text())
    # The optional UC-06 block is absent (default course=None) ...
    assert "course" not in doc["meta"]
    # ... yet every field the viewer relies on for the three panels is present.
    assert _VIEWER_REQUIRED_META <= set(doc["meta"])
    assert set(doc["frames"]) == {"activations", "actions", "drone_position"}
    assert len(doc["frames"]["drone_position"]) == doc["meta"]["n_frames"]
