"""UC70 AC1 / AC9: the parametric detector-noise fallback is recovered from synthetic errors.

``fit_parametric`` (maximum likelihood): P(det) = sigmoid(a + b * range + c * edge), radial bias
``c_r * range``, diagonal sigmas proportional to range and a constant yaw bias / sigma. The
simulator's :class:`~drone_fly.sim.noise_model.ClassNoise` evaluates the fitted form the same way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_fly.harness.offline.detector_noise import (
    fit_logistic,
    fit_parametric,
    parametric_bias,
    parametric_sigma,
)
from tests.test_detector_noise_fit import CLS, TRUE, iid_records, sigmoid


@pytest.fixture(scope="module")
def par():
    return fit_parametric(iid_records(40000, seed=21).gates)


def test_detection_logistic_is_recovered(par):
    p = par["p_det"]
    assert p["form"] == "sigmoid(a + b * range_m + c * edge)"
    assert (p["a"], p["b"], p["c"]) == pytest.approx((TRUE["a"], TRUE["b"], TRUE["c"]), abs=0.08)
    # and the curve itself, over the fitted range
    r = np.linspace(2, 38, 10)
    np.testing.assert_allclose(
        sigmoid(p["a"] + p["b"] * r + p["c"] * 0.5),
        sigmoid(TRUE["a"] + TRUE["b"] * r + TRUE["c"] * 0.5),
        atol=0.015,
    )


def test_error_model_is_recovered(par):
    assert par["radial_bias_per_m"] == pytest.approx(TRUE["c_r"], abs=0.002)
    np.testing.assert_allclose(par["sigma_per_m"], TRUE["s"], rtol=0.03)
    assert par["yaw_bias_deg"] == pytest.approx(TRUE["yaw_b"], abs=0.2)
    assert par["yaw_sigma_deg"] == pytest.approx(TRUE["yaw_s"], rel=0.03)
    assert par["n"] == 40000 and 0 < par["n_detected"] < 40000
    assert "errors_unmeasured" not in par


def test_bias_and_sigma_helpers_evaluate_the_form(par):
    r = np.array([5.0, 20.0])
    b = parametric_bias(par, r)
    s = parametric_sigma(par, r)
    assert b[:, 0] == pytest.approx(par["radial_bias_per_m"] * r)
    assert np.all(b[:, 1:3] == 0) and np.all(b[:, 3] == par["yaw_bias_deg"])
    np.testing.assert_allclose(s[:, :3], r[:, None] * np.array(par["sigma_per_m"]))
    assert np.all(s[:, 3] == par["yaw_sigma_deg"])


def test_simulator_class_noise_uses_the_fitted_parametric_form(tmp_path, par):
    from tests.student_support import load_model

    m = load_model(tmp_path, a=par["p_det"]["a"], b=par["p_det"]["b"], c=par["p_det"]["c"],
                   radial_bias_per_m=par["radial_bias_per_m"],
                   sigma_per_m=tuple(par["sigma_per_m"]), yaw_bias_deg=par["yaw_bias_deg"],
                   yaw_sigma_deg=par["yaw_sigma_deg"])  # fmt: skip
    cn = m.class_noise(CLS)
    r, e = np.array([3.0, 30.0]), np.array([1.0, 0.2])
    np.testing.assert_allclose(cn.p_det(r, e), sigmoid(par["p_det"]["a"] + par["p_det"]["b"] * r
                                                       + par["p_det"]["c"] * e))  # fmt: skip
    np.testing.assert_allclose(cn.sigma(r)[:, :3], r[:, None] * np.array(par["sigma_per_m"]))
    assert cn.sigma(r)[0, 3] == pytest.approx(math.radians(par["yaw_sigma_deg"]))
    assert cn.bias(r)[1, 0] == pytest.approx(par["radial_bias_per_m"] * 30.0)


def test_logistic_stays_finite_on_separable_data():
    x = np.linspace(0, 10, 50)
    X = np.stack([np.ones_like(x), x], 1)
    w = fit_logistic(X, (x > 5).astype(float))
    assert np.all(np.isfinite(w)) and w[1] > 0


def test_no_detections_keeps_wide_flagged_errors():
    recs = iid_records(30, seed=22).gates
    for x in recs:
        x["detected"], x["err"] = False, None
    p = fit_parametric(recs)
    assert p["errors_unmeasured"] is True and p["n_detected"] == 0
    assert p["yaw_sigma_deg"] >= 10.0 and min(p["sigma_per_m"]) > 0
