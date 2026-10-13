"""UC70 AC8: the live harness ticks inside the student's training control range.

The ``liftoff`` preset runs at 40 Hz, inside the student's ``control_hz`` (the noise model's
measured live timing, else the defaults), assumes no obstacles (the student's obstacle_vision is
zeros) and uses the fitted ``lost_timeout_s``; the training timing is taken from the noise model.
The session's checkpoint overrides are covered end to end in ``test_student_checkpoint.py``.
"""

from __future__ import annotations

import pytest
import yaml

from drone_fly.harness.config import load_config
from drone_fly.sim.noise_model import DEFAULT_TIMING, NoiseModel
from drone_fly.training.config import load_student_config
from drone_fly.training.distill import perception_config, student_timing
from tests.student_support import load_model, noise_model_dict

MEASURED = {"frame_period_s": 0.0199, "frame_jitter_s": 0.004, "frame_skip_p": 0.05,
            "detection_lag_s": [0.03, 0.08], "control_hz": [26.7, 51.2],
            "obs_action_delay_s": [0.0, 0.025], "missed_deadline_rate": 0.5}  # fmt: skip


@pytest.fixture(scope="module")
def preset():
    return load_config("liftoff", None, None, preset="liftoff")


@pytest.fixture(scope="module")
def cfg():
    return load_student_config("student-liftoff", validate_tracks=False)


def test_preset_rate_is_inside_the_default_student_range(preset):
    lo, hi = DEFAULT_TIMING["control_hz"]
    rate = float(preset.section("runner")["rate_hz"])
    assert rate == 40.0 and lo <= rate <= hi


def test_preset_rate_is_inside_the_measured_student_range(preset, cfg, tmp_path):
    t = student_timing(cfg, load_model(tmp_path, timing=MEASURED))
    assert t.mode == "random" and t.control_hz == (26.7, 51.2)
    assert t.control_hz[0] <= float(preset.section("runner")["rate_hz"]) <= t.control_hz[1]
    assert t.action_delay_s == (0.0, 0.025)
    assert t.skip_prob == (0.0, 0.5)  # from the measured missed-deadline rate


def test_without_measured_timing_the_config_timing_is_used(cfg, tmp_path):
    assert student_timing(cfg, load_model(tmp_path, timing=None)) == cfg.base.timing


def test_preset_free_space_and_lost_timeout(preset):
    assert preset.section("free_space")["assume_no_obstacles"] is True
    assert float(preset.section("gates")["lost_timeout_s"]) == 1.3
    assert preset.parts.gate_estimator == "learned"


def test_perception_timeouts_come_from_the_noise_model(cfg, tmp_path):
    d = noise_model_dict(timing=MEASURED, tracker={"lost_timeout_s": 1.0, "assoc_radius": 1.0})
    d["recommended"] = {"lost_timeout_s": 1.3}
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(d), "utf-8")
    pc = perception_config(cfg, NoiseModel.load(p))
    assert pc.lost_timeout_s == 1.3  # the recommendation, not the tracker's 1.0
    assert pc.frame_period_s == pytest.approx(0.0199) and pc.detection_lag_s == (0.03, 0.08)
    assert pc.assoc_limit == pytest.approx(cfg.student.world_scale * 1.0)  # k = 6 > aperture 2
