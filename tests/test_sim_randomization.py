"""UC67 AC4: per-episode dynamics randomization from the profile's ``randomization`` block.

Every listed key is drawn uniformly in its range scaled about the nominal (0 = nominal, 1 = the
profile range), unlisted keys stay nominal, a scalar drag nominal shares one draw over the three
body axes, masked resets redraw only the masked envs, and the samples are logged per episode
(``LOG_COLUMNS`` -> ``episodes.csv``).
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.drones.schema import RANDOMIZABLE
from drone_fly.sim.course import EPISODE_COLUMNS
from drone_fly.sim.randomization import _LAYOUT, LOG_COLUMNS, DynamicsRandomizer
from tests.sim_support import meteor75, micro, rng_range, variant

N = 4000


def _table(profile, scale, n=N, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = DynamicsRandomizer(profile, scale, gen, dtype=torch.float64)
    return r, r.sample(n).table


def _cols(key):
    return [i for i, (k, _) in enumerate(_LAYOUT) if k == key]


def _shift(r, key):
    # the c2g column holds the common-mode OFFSET (sample - nominal)
    return r.nominal[key][0] if key == "latency.control_to_gyro_s" else 0.0


def test_log_columns_cover_every_randomizable_key():
    assert len(LOG_COLUMNS) == len(_LAYOUT) == 18
    assert {k for k, _ in _LAYOUT} == set(RANDOMIZABLE)
    assert len(set(LOG_COLUMNS)) == len(LOG_COLUMNS)
    # the episode log carries every sample
    assert set(LOG_COLUMNS) <= set(EPISODE_COLUMNS)


@pytest.mark.parametrize("profile_fn", [meteor75, micro])
def test_scale_one_samples_every_listed_key_within_the_profile_range(profile_fn):
    prof = profile_fn()
    r, t = _table(prof, 1.0)
    for key, rng in prof.randomization.items():
        sh = _shift(r, key)
        for c in _cols(key):
            col = t[:, c]
            assert col.min() >= rng.min - sh - 1e-12 and col.max() <= rng.max - sh + 1e-12, key
            span = rng.max - rng.min
            # uniform over the whole range: both ends are approached, the mean is central
            assert col.min() - (rng.min - sh) < 0.01 * span, key
            assert (rng.max - sh) - col.max() < 0.01 * span, key
            assert col.mean().item() == pytest.approx(
                (rng.min + rng.max) / 2 - sh, abs=0.03 * span
            ), key


@pytest.mark.parametrize("profile_fn", [meteor75, micro])
def test_scale_zero_is_the_nominal_plant(profile_fn):
    prof = profile_fn()
    r, t = _table(prof, 0.0, n=64)
    for i, (key, comp) in enumerate(_LAYOUT):
        nom = r.nominal[key][comp] - _shift(r, key)
        torch.testing.assert_close(t[:, i], torch.full((64,), nom, dtype=torch.float64))


def test_intermediate_scale_shrinks_the_range_about_the_nominal():
    prof = micro()
    r, t = _table(prof, 0.5)
    rng = prof.randomization["thrust_scale"]
    col = t[:, _cols("thrust_scale")[0]]
    lo, hi = 1.0 + 0.5 * (rng.min - 1.0), 1.0 + 0.5 * (rng.max - 1.0)
    assert col.min() >= lo - 1e-12 and col.max() <= hi + 1e-12
    assert col.min() < lo + 0.01 * (hi - lo) and col.max() > hi - 0.01 * (hi - lo)
    assert r.bounds("thrust_scale") == ((pytest.approx(lo),), (pytest.approx(hi),))


def test_unlisted_keys_stay_nominal():
    prof = meteor75()  # lists thrust_scale, tau_s, motor lag, c2g, quadratic drag only
    r, t = _table(prof, 1.0)
    unlisted = [k for k in RANDOMIZABLE if k not in prof.randomization]
    assert "rates.roll.rate_scale" in unlisted and "drag.linear_per_s" in unlisted
    for key in unlisted:
        for c in _cols(key):
            assert torch.all(t[:, c] == t[0, c]), key
            comp = _LAYOUT[c][1]
            assert t[0, c].item() == pytest.approx(r.nominal[key][comp]), key
    listed = [k for k in RANDOMIZABLE if k in prof.randomization]
    for key in listed:
        for c in _cols(key):
            assert t[:, c].std() > 0, key


def test_isotropic_drag_shares_one_draw_and_per_axis_drag_draws_each_axis():
    iso = variant(micro(), drag=(0.06, 0.025))
    _, t = _table(iso, 1.0, n=256)
    lin = t[:, _cols("drag.linear_per_s")]
    assert torch.all(lin[:, 0] == lin[:, 1]) and torch.all(lin[:, 1] == lin[:, 2])
    per_axis = variant(
        micro(),
        drag=((0.06, 0.07, 0.08), 0.025),
        randomization={"drag.linear_per_s": rng_range(0.03, 0.12)},
    )
    _, t = _table(per_axis, 1.0, n=256)
    lin = t[:, _cols("drag.linear_per_s")]
    assert not torch.all(lin[:, 0] == lin[:, 1])
    # scale 0 keeps each axis at its own nominal
    _, t0 = _table(per_axis, 0.0, n=4)
    assert t0[0, _cols("drag.linear_per_s")].tolist() == pytest.approx([0.06, 0.07, 0.08])


def test_masked_resample_redraws_only_the_masked_envs():
    gen = torch.Generator().manual_seed(9)
    r = DynamicsRandomizer(micro(), 1.0, gen)
    s = r.sample(8)
    before = s.table.clone()
    mask = torch.tensor([True, False] * 4)
    r.resample(s, mask)
    assert torch.equal(s.table[~mask], before[~mask])
    assert not torch.any(torch.all(s.table[mask] == before[mask], dim=1))


def test_named_views_index_the_table():
    _, t = _table(micro(), 1.0, n=8)
    s = DynamicsRandomizer(micro(), 1.0, torch.Generator().manual_seed(0)).sample(8)
    names = list(LOG_COLUMNS)
    assert torch.equal(s.thrust_scale, s.table[:, names.index("thrust_scale")])
    assert torch.equal(s.tau_s[:, 2], s.table[:, names.index("tau_yaw_s")])
    assert torch.equal(s.delay_s[:, 0], s.table[:, names.index("delay_roll_s")])
    assert torch.equal(s.drag_quadratic[:, 1], s.table[:, names.index("drag_quadratic_y")])
    assert torch.equal(s.motor_lag_s, s.table[:, names.index("motor_lag_s")])
    assert torch.equal(s.c2g_offset_s, s.table[:, names.index("c2g_offset_s")])
    assert s.columns() is s.table


def test_negative_scale_is_refused():
    with pytest.raises(ValueError, match="scale"):
        DynamicsRandomizer(micro(), -0.1)
