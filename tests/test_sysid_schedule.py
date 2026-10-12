"""UC-66 AC5 / AC9: the seeded sysid experiment schedule.

Deterministic for a seed, every experiment type in every pass, the last pass held out, the
full-stick amplitude present, at least two passes.
"""

from __future__ import annotations

import pytest

from drone_fly.sysid.schedule import (
    AXES,
    DEFAULT_AMPLITUDES,
    KINDS,
    Experiment,
    ScheduleSettings,
    build_schedule,
)


def _ids(schedule):
    return [e.id for e in schedule]


def test_same_seed_same_schedule():
    a, b = build_schedule(7), build_schedule(7)
    assert a == b
    assert [e.as_dict() for e in a] == [e.as_dict() for e in b]


def test_different_seeds_reorder_the_experiments():
    a, b = build_schedule(1), build_schedule(2)
    assert _ids(a) != _ids(b)
    # same experiments overall (doublet signs may differ by seed); same count
    assert len(a) == len(b)
    assert sorted(e.kind for e in a) == sorted(e.kind for e in b)


def test_default_is_three_passes_last_held_out():
    s = build_schedule(0)
    passes = sorted({e.pass_index for e in s})
    assert passes == [0, 1, 2]
    for e in s:
        assert e.held_out == (e.pass_index == 2)
    # passes run in order (the session's pass boundaries rely on it)
    idx = [e.pass_index for e in s]
    assert idx == sorted(idx)


def test_ids_are_unique():
    s = build_schedule(3)
    assert len(set(_ids(s))) == len(s)


@pytest.mark.parametrize("passes", [2, 3, 4])
def test_every_type_in_every_pass(passes):
    s = build_schedule(5, ScheduleSettings(passes=passes))
    for p in range(passes):
        exps = [e for e in s if e.pass_index == p]
        assert {e.kind for e in exps} == set(KINDS)
        doublet_axes = {e.axis for e in exps if e.kind == "rate_doublet"}
        assert doublet_axes == set(AXES)
        for axis in AXES:
            amps = sorted(e.amplitude for e in exps if e.kind == "rate_doublet" and e.axis == axis)
            assert amps == sorted(DEFAULT_AMPLITUDES)
        steps = [e for e in exps if e.kind == "throttle_step"]
        assert {(e.amplitude, e.sign) for e in steps} == {
            (a, s) for a in DEFAULT_AMPLITUDES for s in (1, -1)
        }
        assert len([e for e in exps if e.kind == "coast"]) >= 2
        assert {e.sign for e in exps if e.kind == "coast"} == {1, -1}  # alternating heading


def test_full_stick_amplitude_is_present():
    s = build_schedule(0)
    assert max(DEFAULT_AMPLITUDES) == 1.0
    for axis in AXES:
        assert any(e.kind == "rate_doublet" and e.axis == axis and e.amplitude == 1.0 for e in s)
    assert any(e.kind == "throttle_burst" and e.amplitude == 1.0 for e in s)


def test_custom_amplitudes_and_duration_scale():
    base = build_schedule(0, ScheduleSettings(amplitudes=(0.3, 0.7, 1.0)))
    slow = build_schedule(0, ScheduleSettings(amplitudes=(0.3, 0.7, 1.0), duration_scale=2.0))
    assert _ids(base) == _ids(slow)
    for a, b in zip(base, slow, strict=True):
        assert b.hold_s == pytest.approx(2 * a.hold_s) and b.tail_s == pytest.approx(2 * a.tail_s)
    assert {e.amplitude for e in base if e.kind == "rate_doublet"} == {0.3, 0.7, 1.0}


def test_optional_chirp():
    s = build_schedule(0, ScheduleSettings(chirp=False))
    assert not any(e.kind == "throttle_chirp" for e in s)


@pytest.mark.parametrize(
    "kw",
    [
        {"passes": 1},
        {"passes": 0},
        {"amplitudes": ()},
        {"amplitudes": (0.0, 0.5)},
        {"amplitudes": (0.5, 1.2)},
        {"duration_scale": 0.0},
    ],
)
def test_invalid_settings_are_refused(kw):
    with pytest.raises(ValueError):
        ScheduleSettings(**kw)


def test_experiment_dict_round_trip():
    for e in build_schedule(11)[:20]:
        assert Experiment.from_dict(e.as_dict()) == e
