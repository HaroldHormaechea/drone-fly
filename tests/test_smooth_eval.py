"""UC-87 AC7 / AC8: the smoothness evaluator's PASS/FAIL bar logic and sweep enumeration
(``drone_fly.training.smooth_eval``), tested on SYNTHETIC inputs — no rollout, no training, no GPU.

``evaluate_bars`` is pure arithmetic: it turns the four preset-run result dicts into the five AC7
bars — the acceptance contract the post-merge GPU run (D6) is judged against. ``sweep_grid`` /
``sweep_cell_config`` enumerate the AC8 robustness grid deterministically and build per-cell eval
configs. The ``perfect`` / ``live-like`` preset transforms (D5) are also checked here (config-only).
"""

from __future__ import annotations

import pytest

from drone_fly.training.smooth_eval import (
    AC7_COMPLETION_GAP_PP,
    AC7_MEAN_ABS_DELTA,
    AC7_PERFECT_REGRESS_PP,
    AC7_SATURATION,
    AC7_YAW_FLIPS,
    evaluate_bars,
    preset_config,
    sweep_cell_config,
    sweep_grid,
)

# a baseline where every AC7 bar passes comfortably (model-11 perfect completion = 0.80)
M11P = {"completion": 0.80}
FT_PERFECT = {"completion": 0.80}


def _livelike(**over):
    base = {
        "mean_abs_delta": 0.40,
        "saturation_share": 0.05,
        "yaw_flips_per_s": 2.0,
        "completion": 0.75,
    }
    base.update(over)
    return base


def test_all_bars_pass_for_a_good_policy():
    out = evaluate_bars(_livelike(), FT_PERFECT, M11P)
    assert out["pass"] is True
    assert all(b["pass"] for b in out["bars"].values())
    # the bars record their thresholds verbatim
    assert out["bars"]["mean_abs_delta"]["bar"] == AC7_MEAN_ABS_DELTA == 0.6
    assert out["bars"]["saturation"]["bar"] == AC7_SATURATION == 0.10
    assert out["bars"]["yaw_flips"]["bar"] == AC7_YAW_FLIPS == 3.0
    assert out["bars"]["completion_gap"]["bar_pp"] == AC7_COMPLETION_GAP_PP == 0.10
    assert out["bars"]["perfect_no_regress"]["bar_pp"] == AC7_PERFECT_REGRESS_PP == 0.05


@pytest.mark.parametrize(
    "key, at_bar, just_over",
    [
        ("mean_abs_delta", 0.6, 0.6001),  # ≤ 0.6 rad/s per tick
        ("saturation_share", 0.10, 0.1001),  # ≤ 10 %
        ("yaw_flips_per_s", 3.0, 3.0001),  # ≤ 3 /s
    ],
)
def test_live_like_bars_flip_at_their_threshold(key, at_bar, just_over):
    bar_name = {"mean_abs_delta": "mean_abs_delta", "saturation_share": "saturation",
                "yaw_flips_per_s": "yaw_flips"}[key]  # fmt: skip
    assert evaluate_bars(_livelike(**{key: at_bar}), FT_PERFECT, M11P)["bars"][bar_name]["pass"]
    over = evaluate_bars(_livelike(**{key: just_over}), FT_PERFECT, M11P)
    assert over["bars"][bar_name]["pass"] is False
    assert over["pass"] is False  # any single failure fails the whole verdict


def test_completion_gap_bar_flips_at_ten_points_below_model11_perfect():
    # within 10 pp of 0.80 (≥ ~0.70) passes; clearly below fails. Values are kept off the exact
    # float boundary (0.80 − 0.10 is 0.7000000000000001) so the test checks the flip, not rounding.
    ok = evaluate_bars(_livelike(completion=0.72), FT_PERFECT, M11P)["bars"]["completion_gap"]
    bad = evaluate_bars(_livelike(completion=0.68), FT_PERFECT, M11P)["bars"]["completion_gap"]
    assert ok["pass"] is True and bad["pass"] is False
    assert ok["reference"] == 0.80


def test_perfect_no_regress_bar_flips_at_five_points_below_model11_perfect():
    # fine-tuned perfect within 5 pp of 0.80 (≥ ~0.75) passes; clearly below fails
    ok = evaluate_bars(_livelike(), {"completion": 0.77}, M11P)["bars"]["perfect_no_regress"]
    bad = evaluate_bars(_livelike(), {"completion": 0.73}, M11P)["bars"]["perfect_no_regress"]
    assert ok["pass"] is True and bad["pass"] is False


def test_completion_gap_allows_the_finetuned_to_beat_model11():
    # a higher live-like completion than model-11-perfect obviously passes the gap bar
    out = evaluate_bars(_livelike(completion=0.95), FT_PERFECT, M11P)
    assert out["bars"]["completion_gap"]["pass"] is True


# ----------------------------------------------------------------------------- AC8 sweep

def test_sweep_grid_is_the_full_cartesian_product():
    grid = sweep_grid()
    assert len(grid) == 4 * 4 * 4 * 4  # control period × delay × τ × spawn yaw
    keys = {"control_period_s", "delay_s", "tau_s", "spawn_yaw_deg"}
    assert all(set(cell) == keys for cell in grid)
    # deterministic + unique cells
    tuples = {tuple(c[k] for k in sorted(keys)) for c in grid}
    assert len(tuples) == len(grid)
    assert sweep_grid() == grid  # pure enumeration, stable across calls


def test_sweep_grid_honours_custom_axes():
    grid = sweep_grid(
        control_periods_s=(0.02, 0.05),
        delays_s=(0.0, 0.1),
        tau_s=(0.005,),
        spawn_yaws_deg=(0.0, 90.0, 180.0),
    )
    assert len(grid) == 2 * 2 * 1 * 3
    assert grid[0]["control_period_s"] == 0.02 and grid[0]["tau_s"] == 0.005


def test_sweep_cell_config_fixes_timing_tau_and_spawn_yaw():
    cfg = _uc87()
    cell = {"control_period_s": 0.05, "delay_s": 0.075, "tau_s": 0.010, "spawn_yaw_deg": 90.0}
    sc = sweep_cell_config(cfg, cell, extra_dim=98)
    assert sc.timing.control_hz == (20.0, 20.0)  # 1 / 0.05
    assert sc.timing.action_delay_s == (0.075, 0.075) and sc.timing.skip_prob == (0.0, 0.0)
    assert all(v == (0.010, 0.010) for v in sc.plant.range_overrides.values())
    assert sc.course.spawn_yaw is True and sc.course.memory is True
    assert sc.actor.extra_dim == 98 and sc.randomization_scale == 1.0


def test_sweep_cell_config_disables_spawn_yaw_at_zero_degrees():
    cfg = _uc87()
    cell = {"control_period_s": 0.02, "delay_s": 0.0, "tau_s": 0.005, "spawn_yaw_deg": 0.0}
    sc = sweep_cell_config(cfg, cell, extra_dim=98)
    assert sc.course.spawn_yaw is False


# ----------------------------------------------------------------------------- D5 presets

def _uc87():
    from drone_fly.training.config import load_train_config

    return load_train_config("uc87-deploy-robust")


def test_perfect_preset_is_training_defaults_with_no_noise():
    sc = preset_config(_uc87(), "perfect", extra_dim=98)
    assert sc.timing.mode == "fixed" and sc.timing.fixed_hz == 50.0
    assert sc.plant.range_overrides == {}  # nominal τ
    assert sc.randomization_scale == 0.0
    # every input randomiser is off
    assert sc.course.gyro_noise_sigma == 0.0 and sc.course.spawn_yaw is False
    assert sc.course.obstacle_vision_zero is False and sc.course.gate_dropout_p == 0.0
    assert sc.course.memory is True  # the 98-dim model still flies WITH the memory block


def test_live_like_preset_uses_fast_tau_50ms_control_and_the_config_delay():
    cfg = _uc87()
    sc = preset_config(cfg, "live-like", extra_dim=98)
    assert sc.timing.control_hz == (20.0, 20.0)  # 50 ms control period
    assert sc.timing.action_delay_s == (cfg.timing.delay_mode_s, cfg.timing.delay_mode_s)
    # the fast end (low) of each τ override — the fast-Betaflight loop
    assert all(lo == hi for lo, hi in sc.plant.range_overrides.values())
    for k, (lo, _hi) in sc.plant.range_overrides.items():
        assert lo == cfg.plant.range_overrides[k][0]
    assert sc.randomization_scale == 1.0 and sc.course.memory is True


def test_perfect_preset_for_a_70dim_model_flies_without_memory():
    sc = preset_config(_uc87(), "perfect", extra_dim=70)
    assert sc.course.memory is False and sc.actor.extra_dim == 70


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="unknown preset"):
        preset_config(_uc87(), "sideways", extra_dim=98)
