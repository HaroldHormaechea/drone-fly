"""UC71 P3: scene occlusion of labelled gates (``harness/offline/occlusion.py``).

Synthetic only: hand-made ``ColliderArrays`` boxes / spheres in the Unity frame, and a synthetic
auto-labelled straight flight (``labelled_capture``) with a one-box scene cache.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.offline import occlusion as oc
from drone_fly.harness.offline.capture import load_capture
from drone_fly.harness.offline.gate_dataset import in_range
from drone_fly.harness.offline.gate_diagnose import POOLED, range_bin
from drone_fly.harness.offline.gate_train import load_labels
from drone_fly.tracks.liftoff_extract.scene import ColliderArrays
from tests.liftoff_capture_support import labelled_capture

TELEMETRY = load_config("liftoff", preset="liftoff").section("telemetry")
IDENTITY = [0.0, 0.0, 0.0, 1.0]


def box(p, half):
    return {"t": "box", "p": list(p), "q": IDENTITY, "s": list(half)}


def cols(*colliders):
    return ColliderArrays.from_colliders(list(colliders))


# ----------------------------------------------------------------------------- sight_points


def test_sight_points_are_trimmed_and_evenly_spaced():
    drone, gate = np.array([1.0, 2.0, 0.0]), np.array([1.0, 2.0, 10.0])
    pts = oc.sight_points(drone, gate)
    t = pts[:, 2]
    assert t[0] == pytest.approx(oc.NEAR_DRONE_M)  # the drone end is skipped
    assert t[-1] < 10.0 - oc.NEAR_GATE_M + 1e-9  # the gate's own frame is skipped
    assert t[-1] > 10.0 - oc.NEAR_GATE_M - oc.STEP_M
    assert np.allclose(np.diff(t), oc.STEP_M)
    assert np.allclose(pts[:, :2], [1.0, 2.0])  # on the segment
    assert len(pts) == len(np.arange(0.3, 9.0, 0.25))
    # any direction: the same spacing along the segment
    diag = oc.sight_points(np.zeros(3), np.array([3.0, 4.0, 0.0]) * 2)  # 10 m
    assert np.allclose(np.linalg.norm(np.diff(diag, axis=0), axis=1), oc.STEP_M)


@pytest.mark.parametrize("length", [0.0, 0.5, 1.2, oc.NEAR_DRONE_M + oc.NEAR_GATE_M - 0.01])
def test_a_short_sight_line_has_no_points(length):
    pts = oc.sight_points(np.zeros(3), np.array([0.0, 0.0, length]))
    assert pts.shape == (0, 3)


# ----------------------------------------------------------------------------- occluded

DRONE, GATE = np.array([0.0, 1.74, 0.0]), np.array([0.0, 1.74, 20.0])


def test_a_box_between_drone_and_gate_occludes():
    wall = cols(box((0.0, 2.0, 10.0), (3.0, 2.0, 0.5)))
    assert oc.occluded(wall, DRONE, GATE)
    thin = cols(box((0.0, 1.74, 7.0), (0.5, 0.5, 0.13)))  # thinner than a step: still sampled
    assert oc.occluded(thin, DRONE, GATE)
    ball = cols({"t": "sph", "p": [0.0, 1.74, 12.0], "r": 0.5})
    assert oc.occluded(ball, DRONE, GATE)


@pytest.mark.parametrize(
    "where",
    [
        (5.0, 2.0, 10.0),  # beside the sight line
        (0.0, 8.0, 10.0),  # above it
        (0.0, 2.0, 25.0),  # behind the gate
        (0.0, 2.0, -5.0),  # behind the drone
        (0.0, 1.74, 19.5),  # within NEAR_GATE_M: the gate's own frame / supports
        (0.0, 1.74, 0.1),  # within NEAR_DRONE_M of the drone
    ],
)
def test_clear_sight_lines(where):
    assert not oc.occluded(cols(box(where, (0.4, 0.4, 0.1))), DRONE, GATE)


def test_no_colliders_or_no_points_is_clear():
    assert not oc.occluded(cols(), DRONE, GATE)
    near_gate = DRONE + np.array([0.0, 0.0, 1.0])  # too short to sample
    assert not oc.occluded(cols(box((0, 1.74, 0.5), (5, 5, 5))), DRONE, near_gate)


def test_load_scene_reads_a_scene_cache(tmp_path):
    p = tmp_path / "scene.json"
    p.write_text(json.dumps({"colliders": [box((0, 2, 10), (3, 2, 0.5))]}), "utf-8")
    assert oc.occluded(oc.load_scene(p), DRONE, GATE)
    empty = tmp_path / "empty.json"
    empty.write_text("{}", "utf-8")
    assert len(oc.load_scene(empty)) == 0


# ----------------------------------------------------------------------------- capture


#: A wall across the straight +z flight between z 20 and 22 (Unity), wider and taller than it.
WALL_Z = (20.0, 22.0)


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    base = tmp_path_factory.mktemp("occ")
    cap, labels = labelled_capture(base)  # 5 m/s along +z at y 1.74; arches every 15 m
    scene = base / "scene.json"
    wall = box((0.0, 5.0, sum(WALL_Z) / 2), (12.0, 5.0, (WALL_Z[1] - WALL_Z[0]) / 2))
    scene.write_text(json.dumps({"colliders": [wall]}), "utf-8")
    return cap, labels, scene


def expected_counts(cap_dir, labels_dir):
    """Independent rule for a straight +z sight line through the wall slab."""
    m_inv = np.asarray(UNITY_TO_CONTRACT, np.float64).T
    labels = load_labels(labels_dir)
    cap = load_capture(cap_dir, TELEMETRY, 1.0)
    by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
    out = defaultdict(lambda: defaultdict(lambda: {"n": 0, "occluded": 0}))
    for r in labels.rows:
        f = by_file.get(r["file"])
        if f is None:
            continue
        dz = (m_inv @ np.asarray(f.state.pos, np.float64))[2]
        for g in r["gates"]:
            if not in_range(g):
                continue
            gz = (m_inv @ np.asarray(g["center_world"], np.float64))[2]
            # the sampled stretch [drone + 0.3, gate - 1.0] overlaps the 2 m slab
            hit = dz + oc.NEAR_DRONE_M <= WALL_Z[1] and gz - oc.NEAR_GATE_M >= WALL_Z[0]
            for b in (range_bin(float(g["range_m"])), POOLED):
                if b is not None:
                    out[g["cls"]][b]["n"] += 1
                    out[g["cls"]][b]["occluded"] += int(hit)
    return {k: dict(v) for k, v in out.items()}


def test_capture_occlusion_counts_per_class_and_bin(labelled):
    cap, labels, scene = labelled
    got = oc.capture_occlusion(cap, labels, scene, TELEMETRY)
    assert got == expected_counts(cap, labels)
    arch = got["inflatable_arch"]
    assert set(arch) <= {"4-8", "8-15", "15-25", POOLED}
    assert arch[POOLED]["n"] == sum(arch[b]["n"] for b in arch if b != POOLED)
    assert 0 < arch[POOLED]["occluded"] < arch[POOLED]["n"]  # some behind the wall, some not
    # the n column is the in-range gate count of the labels (4-25 m, centre in the image)
    rows = load_labels(labels).rows
    assert arch[POOLED]["n"] == sum(in_range(g) for r in rows for g in r["gates"])


def test_capture_occlusion_without_scene_colliders_is_all_clear(labelled, tmp_path):
    cap, labels, _ = labelled
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"colliders": []}), "utf-8")
    got = oc.capture_occlusion(cap, labels, empty, TELEMETRY)
    assert got and all(c["occluded"] == 0 for bins in got.values() for c in bins.values())
    assert got == {k: {b: {**c, "occluded": 0} for b, c in v.items()}
                   for k, v in expected_counts(cap, labels).items()}  # fmt: skip
