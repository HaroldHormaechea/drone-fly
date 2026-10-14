"""UC70 AC3 (challenger Minor 2): the student observation's ``history`` block.

12 values: EMA (tau 0.25 s) of the body-frame target vector and of ``prev_action``, the change of
the WORLD-frame target estimate rotated into the current body frame (zero unless the target
matched this tick: zero while holding — even under pure rotation — and zero on a hand-over), the
normalised ticks since the last match, and the hand-over flag. The torch twin
(``sim.student_obs.history_step``) and the reference (``student.obs_ref.StudentHistory``) agree
over many ticks; reset restarts both.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.sim.student_obs import EMA_TAU_S, TICKS_NORM, HistoryState, ema_alpha, history_step
from drone_fly.student.obs_ref import StudentHistory


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rand_rot(g) -> np.ndarray:
    q, _ = np.linalg.qr(g.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def ref_step(h, R, pos, target, prev, matched, hand_over, valid, dt):
    return h.step(R, np.asarray(pos, np.float32), np.asarray(target, np.float32),
                  np.asarray(prev, np.float32), matched, hand_over, valid, dt)  # fmt: skip


def torch_step(h, R, pos, target, prev, matched, hand_over, valid, dt):
    F = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)  # noqa: E731
    B = lambda x: torch.tensor(np.asarray(x), dtype=torch.bool)  # noqa: E731
    return history_step(h, F(R), F(pos), F(target), F(prev), B(matched), B(hand_over), B(valid),
                        torch.tensor(np.asarray(dt), dtype=torch.float64)).numpy()  # fmt: skip


DELTA = slice(7, 10)


def test_torch_and_reference_agree_over_many_ticks():
    g = np.random.default_rng(0)
    n, ticks = 32, 40
    th = HistoryState.zeros(n, "cpu")
    rh = [StudentHistory() for _ in range(n)]
    target = g.uniform(-5, 5, (n, 3))
    for tick in range(ticks):
        R = np.stack([rand_rot(g) for _ in range(n)])
        pos = g.uniform(-5, 5, (n, 3))
        matched = g.random(n) < 0.5
        hand = g.random(n) < 0.1
        valid = g.random(n) < 0.9
        target = np.where(matched[:, None], target + g.normal(0, 0.3, (n, 3)), target)
        prev = g.uniform(-1, 1, (n, 4))
        dt = np.full(n, 0.0 if tick == 0 else g.uniform(0.02, 0.04))
        if tick == 20:  # a partial reset (respawn) in the middle
            mask = g.random(n) < 0.5
            th.reset(torch.tensor(mask))
            for j in np.flatnonzero(mask):
                rh[j].reset()
        got = torch_step(th, R, pos, target, prev, matched, hand, valid, dt)
        for j in range(n):
            want = ref_step(rh[j], R[j], pos[j], target[j], prev[j], bool(matched[j]),
                            bool(hand[j]), bool(valid[j]), float(dt[j]))  # fmt: skip
            np.testing.assert_allclose(got[j], want, atol=1e-5, err_msg=str((tick, j)))


def test_delta_is_zero_while_holding_even_under_pure_rotation():
    h = StudentHistory()
    pos = np.zeros(3)
    target = np.array([10.0, 0.0, 0.0])
    ref_step(h, rot_z(0.0), pos, target, np.zeros(4), True, False, True, 0.0)
    rels = []
    for i in range(1, 10):  # the drone spins in place, the held target is unchanged
        out = ref_step(h, rot_z(0.3 * i), pos, target, np.zeros(4), False, False, True, 0.025)
        assert np.all(out[DELTA] == 0.0)
        rels.append(out[:3].copy())
    assert not np.allclose(rels[0], rels[-1])  # the body-frame EMA does move
    # matched again with the SAME world estimate under rotation: still zero (world-frame change)
    out = ref_step(h, rot_z(2.0), pos, target, np.zeros(4), True, False, True, 0.025)
    np.testing.assert_allclose(out[DELTA], 0.0, atol=1e-6)
    # a real move of the estimate shows up rotated into the current body frame
    R = rot_z(math.pi / 2)
    out = ref_step(h, R, pos, target + [0.0, 1.0, 0.0], np.zeros(4), True, False, True, 0.025)
    np.testing.assert_allclose(out[DELTA], R.T @ np.array([0.0, 1.0, 0.0]), atol=1e-6)
    # the torch twin does the same
    th = HistoryState.zeros(1, "cpu")
    torch_step(th, rot_z(0.0)[None], pos[None], target[None], np.zeros((1, 4)), [True], [False],
               [True], [0.0])  # fmt: skip
    for i in range(1, 5):
        out = torch_step(th, rot_z(0.4 * i)[None], pos[None], target[None], np.zeros((1, 4)),
                         [False], [False], [True], [0.025])  # fmt: skip
        assert np.all(out[0, DELTA] == 0.0)


def test_hand_over_zeroes_the_delta_and_raises_the_flag():
    h = StudentHistory()
    pos = np.zeros(3)
    ref_step(h, np.eye(3), pos, [10.0, 0, 0], np.zeros(4), True, False, True, 0.0)
    out = ref_step(h, np.eye(3), pos, [25.0, 4.0, 1.0], np.zeros(4), True, True, True, 0.02)
    assert np.all(out[DELTA] == 0.0) and out[11] == 1.0
    out = ref_step(h, np.eye(3), pos, [25.5, 4.0, 1.0], np.zeros(4), True, False, True, 0.02)
    np.testing.assert_allclose(out[DELTA], [0.5, 0.0, 0.0], atol=1e-6)
    assert out[11] == 0.0


def test_no_delta_after_an_invalid_target():
    h = StudentHistory()
    ref_step(h, np.eye(3), np.zeros(3), np.zeros(3), np.zeros(4), False, False, False, 0.0)
    out = ref_step(h, np.eye(3), np.zeros(3), [10.0, 0, 0], np.zeros(4), True, False, True, 0.02)
    assert np.all(out[DELTA] == 0.0)  # acquiring: the previous "target" was the drone itself


def test_ticks_since_detection_and_ema_alpha():
    h = StudentHistory()
    pos = np.zeros(3)
    out = ref_step(h, np.eye(3), pos, [4.0, 0, 0], [1.0, 0, 0, 0], True, False, True, 0.0)
    np.testing.assert_allclose(out[:3], [4.0, 0, 0])  # the EMA starts at the first value
    np.testing.assert_allclose(out[3:7], [1.0, 0, 0, 0])
    assert out[10] == 0.0
    dt = 0.03
    out = ref_step(h, np.eye(3), pos, [8.0, 0, 0], [0.0, 0, 0, 0], False, False, True, dt)
    a = 1.0 - math.exp(-dt / EMA_TAU_S)
    assert ema_alpha(dt) == pytest.approx(a) and ema_alpha(0.0) == 1.0
    assert out[0] == pytest.approx(4.0 + a * 4.0, rel=1e-6)
    assert out[3] == pytest.approx(1.0 - a, rel=1e-6)
    assert out[10] == pytest.approx(1 / TICKS_NORM)
    for _ in range(80):
        out = ref_step(h, np.eye(3), pos, [8.0, 0, 0], np.zeros(4), False, False, True, dt)
    assert out[10] == 1.0  # clamped at TICKS_NORM


def test_reset_restarts_both_histories():
    h = StudentHistory()
    th = HistoryState.zeros(2, "cpu")
    for _ in range(3):
        ref_step(h, np.eye(3), np.zeros(3), [5.0, 1, 0], np.ones(4), False, False, True, 0.03)
        torch_step(th, np.stack([np.eye(3)] * 2), np.zeros((2, 3)), [[5.0, 1, 0]] * 2,
                   np.ones((2, 4)), [False] * 2, [False] * 2, [True] * 2, [0.03] * 2)  # fmt: skip
    h.reset()
    th.reset(torch.tensor([True, False]))
    assert not h.init and h.ticks == 0 and not h.prev_valid
    assert th.init.tolist() == [False, True] and th.ticks.tolist() == [0, 3]
    fresh = ref_step(h, np.eye(3), np.zeros(3), [2.0, 0, 0], np.zeros(4), True, False, True, 0.03)
    np.testing.assert_allclose(fresh[:3], [2.0, 0, 0])  # no memory of the old EMA
    assert np.all(fresh[DELTA] == 0.0)
    out = torch_step(th, np.stack([np.eye(3)] * 2), np.zeros((2, 3)), [[2.0, 0, 0]] * 2,
                     np.zeros((2, 4)), [True] * 2, [False] * 2, [True] * 2, [0.03] * 2)  # fmt: skip
    np.testing.assert_allclose(out[0], fresh, atol=1e-6)
    assert not np.allclose(out[1, :3], [2.0, 0, 0])  # the unmasked env kept its EMA
