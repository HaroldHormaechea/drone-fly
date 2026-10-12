"""UC-66 AC10: the two committed Liftoff profiles (identified live) load, validate and are complete.

Reads only committed files (``drones/<id>/``); no Liftoff. The ground-ramp hover throttles ``h0``
are read back from each report's "Airframe check" section, so the band test uses what the live
sessions actually measured.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from drone_fly import drones
from drone_fly.harness.liftoff.ui_driver import DRONE_SLOTS
from drone_fly.sysid.report import UC64_HOVER_STICK
from drone_fly.sysid.session import expected_band

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "drones"
MICRO = "liftoff-gremlin-micro"
FIVE = "liftoff-5inch-vortex-pro-250"
LIFTOFF_NAMES = {MICRO: "Test small drone", FIVE: "Vortex Pro 250"}
PLOTS = {f"rates_{a}.png" for a in ("roll", "pitch", "yaw")} | {
    "thrust_curve.png",
    "drag.png",
    "delay.png",
}
_H0 = re.compile(r"^- ok: h0 ([0-9.]+), band", re.M)


def ramp_h0(drone_id: str) -> list[float]:
    text = (ROOT / drone_id / "sysid.md").read_text(encoding="utf-8")
    return [float(x) for x in _H0.findall(text)]


@pytest.mark.parametrize("drone_id", [MICRO, FIVE])
def test_profile_loads_and_validates(drone_id):
    prof = drones.load(drone_id, ROOT)
    assert prof.id == drone_id and prof.source == "liftoff-sysid"
    assert prof.liftoff["drone_name"] == LIFTOFF_NAMES[drone_id]
    assert prof.liftoff["drone_name"] in DRONE_SLOTS
    assert prof.liftoff.get("map")  # AC10: the map is recorded
    assert drone_id in drones.list_ids(ROOT)
    # the loader enforced provenance; spot-check it names the live session
    session = prof.provenance("hover_throttle")["session"]
    for leaf in ("thrust_curve", "rates.roll.tau_s", "drag.linear_per_s", "motor_lag_s",
                 "latency.control_to_gyro_s"):  # fmt: skip
        assert prof.provenance(leaf)["session"] == session, leaf
    # a usable airframe for UC67: hover on the curve, max thrust above hover
    assert prof.thrust_g(prof.hover_throttle) == pytest.approx(1.0, abs=0.05)
    assert prof.max_g > 1.5
    for key, rng in prof.randomization.items():
        assert rng.min <= rng.max, key


def test_fit_quality_matches_the_reports():
    micro, five = drones.load(MICRO, ROOT), drones.load(FIVE, ROOT)
    assert micro.fit_quality == "good" and five.fit_quality == "poor"
    m = (ROOT / MICRO / "sysid.md").read_text(encoding="utf-8")
    f = (ROOT / FIVE / "sysid.md").read_text(encoding="utf-8")
    assert "**fit_quality: good.** AC8 met" in m
    # AC8: not met -> the report says so explicitly and the profile is still written as poor
    assert "**fit_quality: poor.** AC8 NOT MET" in f


@pytest.mark.parametrize("drone_id", [MICRO, FIVE])
def test_report_and_plots_are_committed(drone_id):
    folder = ROOT / drone_id
    text = (folder / "sysid.md").read_text(encoding="utf-8")
    for name in PLOTS:
        assert (folder / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
        assert f"![{name}]({name})" in text, name
    for section in ("## Fitted values", "## Held-out validation", "## Default randomization"):
        assert section in text
    assert len(ramp_h0(drone_id)) >= 2  # one airframe check per fly-up, all ok


def test_no_raw_telemetry_is_committed():
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "drones"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out, "drones/ is tracked"
    assert all(Path(p).suffix in {".yaml", ".md", ".png"} for p in out), out


def test_micro_report_compares_with_uc64():
    text = (ROOT / MICRO / "sysid.md").read_text(encoding="utf-8")
    assert "UC64 comparison" in text and f"{UC64_HOVER_STICK:+.2f}" in text
    prof = drones.load(MICRO, ROOT)
    stick = 2 * prof.hover_throttle - 1
    assert stick == pytest.approx(UC64_HOVER_STICK, abs=0.05)  # within 0.05 stick of UC64


def test_measured_h0_separates_the_airframes():
    """Each drone's measured ramp h0 sits inside its own guard band and outside the other's, for
    both band sources: the committed profile (hover +- 0.08) and ui_driver.DRONE_SLOTS."""
    h0 = {d: ramp_h0(d) for d in (MICRO, FIVE)}
    assert all(0.45 < x < 0.48 for x in h0[MICRO]), h0  # ~0.463
    assert all(0.28 < x < 0.31 for x in h0[FIVE]), h0  # ~0.298
    bands = {}
    for d in (MICRO, FIVE):
        prof_band, src = expected_band(d, LIFTOFF_NAMES[d], None, root=ROOT)
        assert "drone.yaml" in src
        hov = drones.load(d, ROOT).hover_throttle
        assert prof_band == pytest.approx((hov - 0.08, hov + 0.08))
        slot_band = DRONE_SLOTS[LIFTOFF_NAMES[d]]["hover_band"]
        bands[d] = (prof_band, slot_band)
    for d, other in ((MICRO, FIVE), (FIVE, MICRO)):
        for lo, hi in bands[d]:
            for x in h0[d]:
                assert lo <= x <= hi, (d, x, (lo, hi))
            for x in h0[other]:
                assert not lo <= x <= hi, (other, x, "inside", d, (lo, hi))
    # the DRONE_SLOTS bands do not overlap. (The profile bands, fitted hover +- 0.08, DO overlap
    # in 0.391..0.411 because the fitted hovers are only 0.14 apart; the measured h0 values still
    # fall outside the other drone's band, which is what the guard needs.)
    (a_lo, a_hi), (b_lo, b_hi) = bands[MICRO][1], bands[FIVE][1]
    assert b_hi <= a_lo or a_hi <= b_lo
