"""UC71 AC6: the multi-map gate dataset, its G1 report and multi-map training.

A synthetic world: one auto-labelled straight flight through five arches, copied as captures of
four catalogue maps (FakeA / FakeB train, FakeC val, FakeD held-out), each with its own
``meta.json`` and label folder; a synthetic catalogue (``cat.Catalogue``) and synthetic weights
metadata stand in for the v1 release. ``gate_dataset.models_root`` is redirected to a temp folder
so nothing is written under ``artifacts/``. Tiny-cnn training on CPU only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from drone_fly.cli.main import build_parser, main  # noqa: E402
from drone_fly.harness.config import load_config  # noqa: E402
from drone_fly.harness.errors import HarnessConfigError, NoiseLeakageError  # noqa: E402
from drone_fly.harness.gate_types import GateTypeTable  # noqa: E402
from drone_fly.harness.offline import gate_dataset as gd  # noqa: E402
from drone_fly.harness.offline.autolabel import draft_gate_types, run_autolabel  # noqa: E402
from drone_fly.harness.offline.capture_index import build_meta, write_meta  # noqa: E402
from drone_fly.harness.offline.detector_noise import heldout_window  # noqa: E402
from drone_fly.harness.offline.gate_dataset import (  # noqa: E402
    DatasetItem,
    TrackLeakageError,
    assemble,
    balanced_weights,
    build_frame_cache,
    count_gate_frames,
    dataset_report,
    g1_verdict,
    report_summary,
)
from drone_fly.harness.offline.gate_train import load_labels, train_multi  # noqa: E402
from drone_fly.tracks import catalogue as cat  # noqa: E402
from tests.liftoff_capture_support import (  # noqa: E402
    SMALL_CAMERA,
    Pose,
    arch_track,
    straight_flight,
    write_manual_capture,
)

CFG = load_config("liftoff", SMALL_CAMERA, preset="liftoff")
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
TELEMETRY = CFG.section("telemetry")
MAPS = {"FakeA": "train", "FakeB": "train", "FakeC": "val", "FakeD": "held-out", "FakeU": "train"}
ARCHES = [(str(i), (0.0, 0.0, 15.0 * i), 0.0) for i in range(1, 6)]


def session(m: str, nn: int = 1) -> str:
    return f"{cat.kebab(m)}-2026-10-13-manual-{nn:02d}"


def entry_id(m: str) -> str:
    return f"{cat.kebab(m)}-01"


def flat_race(path: Path, ids) -> Path:
    cps = "".join(f"<Checkpoint><instanceID>{i}</instanceID></Checkpoint>" for i in ids)
    path.write_text(f"<Race><laps>1</laps><checkpoints>{cps}</checkpoints></Race>", "utf-8")
    return path


def random_image(k):
    return np.random.default_rng(k).integers(0, 255, (108, 256, 3), np.uint8)


def labelled_source(base: Path, poses, frames) -> tuple[Path, Path]:
    cap = write_manual_capture(base / "cap", poses, frame_times_s=frames, size=(256, 108),
                               image=random_image)  # fmt: skip
    labels = base / "labels"
    run_autolabel(cap, base.parent / "trackdata" / "track_FakeA.xml", preset="liftoff",
                  config=SMALL_CAMERA, out=labels)  # fmt: skip
    return cap, labels


class World:
    """Trackdata, catalogue, labelled captures; ``add`` makes one more capture of a map."""

    def __init__(self, base: Path):
        self.base, self.root = base, base / "trackdata"
        self.models = base / "models-root"
        self.captures = base / "captures"
        self.root.mkdir(parents=True)
        entries = []
        for m, split in MAPS.items():
            track = arch_track(self.root / f"track_{m}.xml", ARCHES)
            if m == "FakeU":  # an uncovered gate-like item on this map
                text = track.read_text("utf-8").replace(
                    "</blueprints>",
                    "<TrackBlueprint><itemID>FakeTrussGate01</itemID><instanceID>9</instanceID>"
                    "<position><x>9</x><y>0</y><z>9</z></position>"
                    "<rotation><x>0</x><y>0</y><z>0</z></rotation></TrackBlueprint></blueprints>",
                )
                track.write_text(text, "utf-8")
            race = flat_race(self.root / f"race_{m}.xml", [a[0] for a in ARCHES])
            entries.append({"id": entry_id(m), "map": m, "split": split, "track": track.name,
                            "race": race.name,
                            "sha256": {"track": cat.sha256_file(track),
                                       "race": cat.sha256_file(race)}})  # fmt: skip
        path = base / "liftoff.yaml"
        path.write_text(yaml.safe_dump({"entries": entries}), "utf-8")
        self.catalogue = cat.Catalogue(path, {"source": "liftoff-xml", "root": str(self.root)},
                                       entries)  # fmt: skip
        # a 12 s straight flight (moving), and one with a 3 s hover in the middle (static)
        self.src = labelled_source(base / "src", straight_flight(duration_s=12.0),
                                   [k / 10 for k in range(1, 120)])  # fmt: skip
        hover = straight_flight(duration_s=4.0)
        end = hover[-1]
        hover += [Pose(end.t_sim + k / 100, end.position, end.quat_xyzw) for k in range(1, 301)]
        hover += [Pose(7.0 + p.t_sim + 0.01, (p.position[0], p.position[1], p.position[2] + 20.0),
                       p.quat_xyzw) for p in straight_flight(duration_s=4.0)]  # fmt: skip
        self.src_static = labelled_source(base / "src-static", hover,
                                          [k / 10 for k in range(1, 110)])  # fmt: skip

    def add(self, m: str, nn: int = 1, *, verified=True, labels=True, static=False,
            meta=True, uptilt=None) -> Path:  # fmt: skip
        src_cap, src_labels = self.src_static if static else self.src
        d = shutil.copytree(src_cap, self.captures / m / session(m, nn))
        if meta:
            e = self.catalogue.by_id(entry_id(m))
            cam = {**CFG.section("camera")}
            if uptilt is not None:  # another drone's camera (e.g. the Borrum 180 at 30 deg)
                cam["uptilt_deg"] = uptilt
            mt = build_meta(d, e, camera=cam, drone="liftoff-gremlin-micro")
            mt["alignment"] = {"verified": verified, "margin": 1.4 if verified else 1.0}
            write_meta(d, mt)
        if labels:
            lab = shutil.copytree(src_labels, self.models / "labels" / m / d.name)
            lm = json.loads((lab / "labels_meta.json").read_text("utf-8"))
            lm["capture"] = str(d)
            (lab / "labels_meta.json").write_text(json.dumps(lm), "utf-8")
        return d

    def labels(self, d: Path):
        return load_labels(self.models / "labels" / d.parent.name / d.name)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    w = World(tmp_path_factory.mktemp("gd"))
    w.a, w.b = w.add("FakeA"), w.add("FakeB")
    w.c, w.d = w.add("FakeC"), w.add("FakeD")
    return w


@pytest.fixture(autouse=True)
def _models_root(world, monkeypatch):
    monkeypatch.setattr(gd, "models_root", lambda: world.models)


def build(world, dirs, **kw):
    kw.setdefault("telemetry_cfg", TELEMETRY)
    return assemble(dirs, catalogue=world.catalogue, table=TABLE, **kw)


def window_meta(world, d, lo, hi, frames=None):
    """Synthetic v1-style weights meta protecting label rows ``lo:hi`` of capture ``d``."""
    st = sorted(r["stamp"] for r in world.labels(d).rows)
    win = {"t_start": st[lo], "t_end": st[hi - 1]}
    n = hi - lo if frames is None else frames
    return {"holdout": {d.name: win}, "captures": [{"name": d.name, "n_holdout": n}]}, win


# ----------------------------------------------------------------------------- routing / refusals


def test_captures_route_by_the_catalogue_split(world):
    ds = build(world, [world.a, world.b, world.c])
    roles = {c.name: c.role for c in ds.captures}
    assert roles == {world.a.name: "train", world.b.name: "train", world.c.name: "selection"}
    assert {it.role for it in ds.items if it.capture == world.c.name} == {"selection"}
    assert ds.refused == [] and len(ds.role_items("train")) == 2 * len(world.labels(world.a).rows)


def test_the_split_comes_from_the_catalogue_not_meta(world, tmp_path):
    meta = json.loads((world.c / "meta.json").read_text("utf-8"))
    assert meta["split"] == "val"
    d = shutil.copytree(world.c, tmp_path / world.c.name)
    meta["split"] = "train"  # informational only
    write_meta(d, meta)
    lab = world.models / "labels" / "FakeC" / world.c.name
    ds = build(world, [d])
    assert lab.is_dir() and ds.captures[0].role == "selection"


def test_a_held_out_capture_raises_track_leakage(world):
    with pytest.raises(TrackLeakageError, match="held-out maps are never trained"):
        build(world, [world.a, world.d])
    assert issubclass(TrackLeakageError, HarnessConfigError)


def test_other_cameras_are_for_train_maps_only(world):
    assert gd.CAMERA_TOL_DEG == 0.5
    assert gd.same_camera({"camera": {"uptilt_deg": 20.5}}, 20.0)
    assert not gd.same_camera({"camera": {"uptilt_deg": 20.6}}, 20.0)
    assert not gd.same_camera({"camera": {"uptilt_deg": 30}}, 20.0)
    train30 = world.add("FakeB", 31, uptilt=30.0)
    val30 = world.add("FakeC", 32, uptilt=30.0)
    val_near = world.add("FakeC", 33, uptilt=20.4)
    ds = build(world, [world.a, train30, val30, val_near], preset_uptilt_deg=20.0)
    roles = {c.name: c.role for c in ds.captures}
    assert roles[train30.name] == "train"  # view diversity on a train map
    assert roles[val_near.name] == "selection"  # within the tolerance
    assert roles[val30.name] is None
    reason = next(c.reason for c in ds.captures if c.name == val30.name)
    assert reason == (
        "camera uptilt 30 is not the preset's 20: other cameras are for train maps only"
    )
    assert {it.capture for it in ds.items} == {world.a.name, train30.name, val_near.name}
    assert {r["name"] for r in ds.manifest()["refused"]} == {val30.name}
    # without a preset uptilt (the old call) nothing is refused for its camera
    assert build(world, [val30]).captures[0].role == "selection"


def test_include_selection_in_train_routes_val_maps_to_train(world):
    ds = build(world, [world.a, world.c], include_selection_in_train=True)
    assert {c.role for c in ds.captures} == {"train"}


def test_refusals_carry_a_reason(world):
    nometa = world.add("FakeA", 11, meta=False)
    unverified = world.add("FakeA", 12, verified=False)
    uncovered = world.add("FakeU", 13)
    nolabels = world.add("FakeB", 14, labels=False)
    mismatch = world.add("FakeB", 15)
    lm_path = world.models / "labels" / "FakeB" / mismatch.name / "labels_meta.json"
    lm = json.loads(lm_path.read_text("utf-8"))
    lm["gate_types_hash"] = "0" * 16
    lm_path.write_text(json.dumps(lm), "utf-8")
    ds = build(world, [world.a, nometa, unverified, uncovered, nolabels, mismatch])
    reasons = {c.name: c.reason for c in ds.refused}
    assert set(reasons) == {nometa.name, unverified.name, uncovered.name, nolabels.name,
                            mismatch.name}  # fmt: skip
    assert reasons[nometa.name].startswith("no catalogue id")
    assert "alignment not verified" in reasons[unverified.name]
    assert "uncovered items ['FakeTrussGate01']" in reasons[uncovered.name]
    assert reasons[nolabels.name].startswith("no labels")
    assert "another gate table" in reasons[mismatch.name]
    assert {it.capture for it in ds.items} == {world.a.name}  # none of their frames load
    assert {r["name"] for r in ds.manifest()["refused"]} == set(reasons)


def test_not_gates_admits_the_uncovered_track(world):
    d = world.add("FakeU", 16)
    ds = build(world, [d], not_gates=["^FakeTruss"])
    assert ds.captures[0].role == "train"


# ----------------------------------------------------------------------------- protected window


def test_protected_window_frames_are_never_loaded_and_recorded(world):
    protect, win = window_meta(world, world.a, 40, 60)
    ds = build(world, [world.a, world.b], protect=protect)
    a = next(c for c in ds.captures if c.name == world.a.name)
    stamps = [it.row["stamp"] for it in ds.items if it.capture == world.a.name]
    assert not any(win["t_start"] <= s <= win["t_end"] for s in stamps)
    assert a.counts["protected"] == 20 and len(stamps) == a.counts["labels"] - 20
    assert a.protected["frames"] == 20 and a.protected["frames_present"] == 20
    assert a.protected["t_start"] == win["t_start"] and a.protected["expected_in_range"] is None
    rows = world.labels(world.a).rows
    in_win = [r for r in rows if win["t_start"] <= r["stamp"] <= win["t_end"]]
    assert a.protected["in_range_gate_frames"] == sum(gd.in_range(g) for r in in_win
                                                      for g in r["gates"])  # fmt: skip
    b = next(c for c in ds.captures if c.name == world.b.name)
    assert b.protected is None and "protected" not in b.counts
    assert ds.manifest()["captures"][0]["protected_window"] == a.protected


def _g1_protected(rep):
    return g1_verdict(rep, {k: 0 for k in gd.G1})["checks"]["protected_window"]


def test_g1_requires_the_full_protected_window(world, monkeypatch):
    protect, _ = window_meta(world, world.a, 40, 60)
    full = dataset_report(build(world, [world.a], protect=protect))
    assert _g1_protected(full)["ok"] is True
    # PARTIAL: v1 recorded 25 frames for that window, only 20 are present (challenger Minor)
    protect, _ = window_meta(world, world.a, 40, 60, frames=25)
    partial = dataset_report(build(world, [world.a], protect=protect))
    assert _g1_protected(partial) == {"ok": False, "value": {world.a.name: False},
                                      "need": "the v1 window fully present"}  # fmt: skip
    # the in-range count must match the recorded one too
    protect, _ = window_meta(world, world.a, 40, 60)
    monkeypatch.setitem(gd.PROTECTED_IN_RANGE, world.a.name, 10_000)
    wrong = dataset_report(build(world, [world.a], protect=protect))
    assert _g1_protected(wrong)["ok"] is False
    # no window at all (the protected capture missing from --captures): not GO either
    assert _g1_protected(dataset_report(build(world, [world.b], protect=protect)))["ok"] is False


# ----------------------------------------------------------------------------- counts / weights


def test_static_stride_and_cap_counts(world):
    d = world.add("FakeB", 21, static=True)
    n = len(world.labels(d).rows)
    off = build(world, [d], static_trim=False).captures[0].counts
    assert off == {"labels": n, "stride": 0, "cap": 0, "used": n}
    on = build(world, [d]).captures[0].counts
    assert on["static"] > 20  # the 3 s hover keeps one frame
    k = n - on["static"]
    assert on["used"] == k
    c = build(world, [d], frame_stride=3, max_frames_per_capture=10).captures[0].counts
    strided = len(range(0, k, 3))
    assert c["stride"] == k - strided and c["cap"] == strided - 10 and c["used"] == 10
    assert c["static"] + c["stride"] + c["cap"] + c["used"] == n
    with pytest.raises(HarnessConfigError, match="frame-stride"):
        build(world, [d], frame_stride=0)


def test_capped_frames_are_evenly_spaced(world):
    ds = build(world, [world.a], max_frames_per_capture=5)
    st = np.array([it.row["stamp"] for it in ds.items])
    all_st = sorted(r["stamp"] for r in world.labels(world.a).rows)
    assert len(st) == 5 and st[0] == all_st[0] and st[-1] == all_st[-1]
    gaps = np.diff(st)
    assert gaps.max() / gaps.min() < 1.2


def _row(*gates):
    return {"gates": [{"cls": c, "range_m": r, "center_in_image": True} for c, r in gates]}


def test_balanced_weights_give_each_map_class_equal_mass():
    items = (
        [DatasetItem("a", "A", "train", Path("x"), _row(("arch", 10.0))) for _ in range(6)]
        + [DatasetItem("a", "A", "train", Path("x"), _row(("arch", 30.0))) for _ in range(2)]
        + [DatasetItem("b", "B", "train", Path("x"), _row(("arch", 5.0), ("square", 4.5)))]
        + [DatasetItem("b", "B", "train", Path("x"), _row(("square", 12.0))) for _ in range(3)]
    )  # A/arch 6, A/none 2 (out of range), B/square 1 + 3 (the NEAREST in-range gate)
    balanced_weights(items)
    w = np.array([it.weight for it in items])
    assert w.sum() == pytest.approx(1.0)
    mass = {}
    for it in items:
        key = (it.map, gd._dominant_class(it.row))
        mass[key] = mass.get(key, 0.0) + it.weight
    assert set(mass) == {("A", "arch"), ("A", "none"), ("B", "square")}
    assert all(v == pytest.approx(1 / 3) for v in mass.values())


def test_assembled_train_weights_balance_the_maps(world):
    ds = build(world, [world.a, world.b, world.c], max_frames_per_capture=30)
    train = ds.role_items("train")
    assert sum(it.weight for it in train) == pytest.approx(1.0)
    assert all(it.weight == 1.0 for it in ds.role_items("selection"))  # not sampled
    per_map = {m: sum(it.weight for it in train if it.map == m) for m in ("FakeA", "FakeB")}
    assert per_map["FakeA"] == pytest.approx(per_map["FakeB"])


# ----------------------------------------------------------------------------- manifest / cache


def test_manifest_contents_and_sha(world, tmp_path):
    ds = build(world, [world.a, world.c], max_frames_per_capture=12)
    m = ds.manifest()
    assert m["format"] == 1 and m["gate_types_hash"] == TABLE.hash()
    assert m["catalogue_sha256"] == cat.sha256_file(world.catalogue.path)
    assert m["params"]["max_frames_per_capture"] == 12 and m["params"]["static_trim"] is True
    rec = {c["name"]: c for c in m["captures"]}
    assert rec[world.a.name]["split_role"] == "train" and rec[world.c.name]["split_role"] == (
        "selection"
    )
    assert rec[world.a.name]["catalogue_id"] == "fake-a-01" and rec[world.a.name]["map"] == "FakeA"
    assert rec[world.a.name]["capture_sha256"] and rec[world.a.name]["counts"]["used"] == 12
    sha = ds.manifest_sha()
    assert len(sha) == 64 and build(world, [world.a, world.c], max_frames_per_capture=12)\
        .manifest_sha() == sha  # fmt: skip
    # a changed capture (here its meta.json) changes the manifest
    d = shutil.copytree(world.a, tmp_path / world.a.name)
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    meta["alignment"]["margin"] = 1.5
    write_meta(d, meta)
    assert build(world, [d, world.c], max_frames_per_capture=12).manifest_sha() != sha
    # and so do the parameters
    assert build(world, [world.a, world.c], max_frames_per_capture=11).manifest_sha() != sha


def test_frame_cache_round_trip_and_reuse(world, tmp_path, monkeypatch):
    from drone_fly.harness.liftoff.gate_net import bgr_to_tensor, input_geometry, resize_images
    from drone_fly.harness.offline import evaluate

    ds = build(world, [world.a, world.c], max_frames_per_capture=4)
    out = tmp_path / "cache"
    mm, idx = build_frame_cache(ds, 128, CFG.section("camera"), out=out)
    g = input_geometry(256, 108, 128)
    assert mm.shape == (8, 3, g[1], g[0]) and mm.dtype == np.uint8
    assert idx["files"] == [str(it.path) for it in ds.items]
    assert idx["manifest_sha256"] == ds.manifest_sha() and tuple(idx["geometry"]) == g
    img = evaluate.load_image(ds.items[3].path)
    want = resize_images(bgr_to_tensor(img, "cpu"), g[0], g[1])[0].round().clamp(0, 255)
    assert np.array_equal(np.asarray(mm[3]), want.to(torch.uint8).numpy())

    def no_reload(path):
        raise AssertionError("cache must be reused")

    monkeypatch.setattr(evaluate, "load_image", no_reload)
    mm2, idx2 = build_frame_cache(ds, 128, CFG.section("camera"), out=out)
    assert idx2 == idx and np.array_equal(np.asarray(mm2), np.asarray(mm))
    monkeypatch.undo()
    monkeypatch.setattr(gd, "models_root", lambda: world.models)
    del mm, mm2  # Windows: an open memmap blocks rewriting its file
    _, idx3 = build_frame_cache(ds, 160, CFG.section("camera"), out=out)  # another width: rebuilt
    assert idx3["input_width"] == 160


def test_default_cache_dir_is_keyed_by_the_manifest(world):
    ds = build(world, [world.a], max_frames_per_capture=2)
    assert gd.cache_dir(ds, 128) == world.models / "datasets" / f"{ds.manifest_sha()[:8]}-w128"


# ----------------------------------------------------------------------------- report / G1


def test_dataset_report_counts_by_split_map_and_range(world):
    ds = build(world, [world.a, world.c])
    d_rows = world.labels(world.d).rows
    rep = dataset_report(ds, {"FakeD": d_rows})
    a_rows = world.labels(world.a).rows
    assert rep["splits"]["train"]["FakeA"] == count_gate_frames(a_rows)
    assert rep["splits"]["selection"]["FakeC"] == count_gate_frames(a_rows)
    assert rep["splits"]["held-out"]["FakeD"] == count_gate_frames(d_rows)
    arch = rep["splits"]["train"]["FakeA"]["inflatable_arch"]
    assert arch["4-25"] > 0
    assert arch["4-25"] == arch.get("4-8", 0) + arch.get("8-15", 0) + arch.get("15-25", 0)
    want = sum(1 for r in a_rows for g in r["gates"] if gd.in_range(g)
               and 4.0 <= g["range_m"] <= 8.0)  # fmt: skip
    assert arch.get("4-8", 0) == want
    assert rep["frames"] == {"train": len(a_rows), "selection": len(a_rows)}
    assert rep["g1"]["go"] is False  # far below the real thresholds
    text = report_summary(rep)
    assert "G1: NO-GO" in text and "FAIL train_frames" in text


def test_in_range_rule():
    g = {"range_m": 10.0, "center_in_image": True}
    assert gd.in_range(g)
    assert not gd.in_range({**g, "range_m": 3.9}) and not gd.in_range({**g, "range_m": 25.1})
    assert not gd.in_range({**g, "center_in_image": False})
    assert not gd.in_range({**g, "center_masked": True})
    assert not gd.in_range({**g, "occluded": True})


def _counts(n, near=0, cls="arch"):
    return {cls: {"4-25": n, "4-8": near}}


def synthetic_report(**over):
    rep = {
        "splits": {
            "train": {f"T{i}": _counts(700, near=80) for i in range(5)},
            "selection": {"V0": _counts(300), "V1": _counts(400)},
            "held-out": {"H0": _counts(30), "H1": _counts(5, cls="new_type")},
        },
        "protected": {"cap": {"frames": 10, "frames_present": 10, "in_range_gate_frames": 7,
                              "expected_in_range": 7}},
    }  # fmt: skip
    rep.update(over)
    return rep


def test_g1_go():
    v = g1_verdict(synthetic_report())
    assert v["go"] is True and all(c["ok"] for c in v["checks"].values())
    assert v["checks"]["train_frames"]["value"] == 3500
    assert v["checks"]["heldout_seen_type"]["value"] == {"arch": 30}  # new_type is not seen


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        (lambda r: r["splits"]["train"].pop("T4"), {"train_maps", "train_frames"}),
        (lambda r: r["splits"]["train"].update(T0=_counts(10, near=80)), {"train_frames"}),
        (lambda r: r["splits"]["train"].update({f"T{i}": _counts(700, near=50) for i in range(5)}),
         {"train_near_frames"}),
        (lambda r: r["splits"]["selection"].update(V1=_counts(299)), {"val_maps"}),
        (lambda r: r["splits"]["held-out"].pop("H1"), {"heldout_maps"}),
        (lambda r: r["splits"]["held-out"].update(H0=_counts(29)), {"heldout_seen_type"}),
        (lambda r: r["protected"]["cap"].update(frames_present=9), {"protected_window"}),
        (lambda r: r["protected"]["cap"].update(in_range_gate_frames=6), {"protected_window"}),
        (lambda r: r.update(protected={}), {"protected_window"}),
    ],
)  # fmt: skip
def test_g1_no_go_reasons(change, failing):
    rep = synthetic_report()
    change(rep)
    v = g1_verdict(rep)
    assert v["go"] is False
    assert {k for k, c in v["checks"].items() if not c["ok"]} == failing


def test_g1_thresholds_can_be_overridden():
    rep = synthetic_report()
    rep["splits"]["train"].pop("T4")
    assert g1_verdict(rep, {"train_maps": 4, "train_frames": 2800})["go"] is True


# ----------------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def protect_file(world):
    st = sorted(r["stamp"] for r in world.labels(world.a).rows)
    win = {"t_start": st[40], "t_end": st[59]}
    p = world.base / "v1-like.pt"
    torch.save({"state_dict": {}, "meta": {"holdout": {world.a.name: win},
                                           "captures": [{"name": world.a.name,
                                                         "n_holdout": 20}]}}, p)  # fmt: skip
    return p, win


def _train(world, protect, tmp, monkeypatch, **kw):
    monkeypatch.setattr(cat, "load_catalogue", lambda path: world.catalogue)
    args = dict(trackdata_root=world.root, split_source=world.catalogue.path, config=SMALL_CAMERA,
                epochs=2, backbone="tiny-cnn", batch_size=4, input_width=128,
                max_frames_per_capture=8, val_every=1, protect_holdout_of=protect, device="cpu",
                out_root=tmp, detector_id="t-uc71")  # fmt: skip
    return train_multi([world.a, world.b, world.c], **{**args, **kw})


def test_train_multi_records_train_selection_and_protected_window(
    world, protect_file, tmp_path, monkeypatch
):
    path, win = protect_file
    meta = _train(world, path, tmp_path, monkeypatch, select="val-detection")
    out = tmp_path / "t-uc71"
    assert Path(meta["weights"]) == out / "liftoff-gates-v2.pt" and Path(meta["weights"]).is_file()
    assert (out / "data_manifest.json").is_file() and Path(meta["training_log"]).is_file()
    assert meta["purpose"] == "evaluate" and meta["select"] == "val-detection"
    assert [c["name"] for c in meta["captures"]] == [world.a.name, world.b.name]  # train maps
    assert [c["name"] for c in meta["selection_captures"]] == [world.c.name]  # val maps
    assert {c["split_role"] for c in meta["captures"]} == {"train"}
    assert meta["holdout"] == {world.a.name: win} and set(meta["splits"]) == {world.a.name}
    assert meta["splits"][world.a.name]["mode"] == "protected"
    assert meta["splits"][world.a.name]["counts"]["holdout"] == 20
    det = meta["chosen_val_detection"]
    assert det is not None and 0.0 <= det["detection_rate"] <= 1.0 and det["n"] > 0
    assert meta["chosen_epoch"] in (1, 2) and meta["data_manifest_sha256"]
    log = json.loads(Path(meta["training_log"]).read_text("utf-8"))
    assert all("val_detection" in e for e in log["history"])  # val_every 1
    # detector_noise: val captures count as never trained; the protected window stays held out
    assert heldout_window(meta, world.c.name) is None
    assert heldout_window(meta, world.a.name) == win
    with pytest.raises(NoiseLeakageError):
        heldout_window(meta, world.b.name)  # trained on every frame
    stamps = [r["stamp"] for r in world.labels(world.a).rows]
    assert any(win["t_start"] <= s <= win["t_end"] for s in stamps)
    manifest = json.loads((out / "data_manifest.json").read_text("utf-8"))
    assert manifest == {**manifest, "gate_types_hash": TABLE.hash()}


def test_train_multi_deploy_naming_and_purpose(world, protect_file, tmp_path, monkeypatch):
    path, _ = protect_file
    meta = _train(world, path, tmp_path, monkeypatch, deploy=True, epochs=1)
    assert Path(meta["weights"]).name == "liftoff-gates-v2-deploy.pt"
    assert meta["purpose"] == "deploy" and meta["select"] == "last"
    assert meta["selection_captures"] == []
    assert {c["name"] for c in meta["captures"]} == {world.a.name, world.b.name, world.c.name}
    assert meta["holdout"] == {} and meta["chosen_val_detection"] is None
    with pytest.raises(NoiseLeakageError):
        heldout_window(meta, world.a.name)


def test_train_multi_argument_errors(world, tmp_path, monkeypatch):
    with pytest.raises(HarnessConfigError, match="--select"):
        _train(world, None, tmp_path, monkeypatch, select="best")
    with pytest.raises(HarnessConfigError, match="--val-every"):
        _train(world, None, tmp_path, monkeypatch, val_every=0)
    monkeypatch.setattr(cat, "load_catalogue", lambda path: world.catalogue)
    with pytest.raises(HarnessConfigError, match="needs val-map captures"):
        train_multi([world.a], trackdata_root=world.root, split_source=world.catalogue.path,
                    config=SMALL_CAMERA, backbone="tiny-cnn", protect_holdout_of=None,
                    device="cpu", out_root=tmp_path)  # fmt: skip


# ----------------------------------------------------------------------------- draft-types


def _box(p, s, trig=False):
    return {"t": "box", "p": list(p), "s": list(s), **({"trig": True} if trig else {})}


PROPS = {
    # a square frame 2 x 2.5 m with a bottom bar: aperture 1.6 x 1.6 starting 0.45 m up
    "FakeSquareGate01": {"colliders": [
        _box((-0.9, 1.25, 0), (0.1, 1.25, 0.1)), _box((0.9, 1.25, 0), (0.1, 1.25, 0.1)),
        _box((0, 2.4, 0), (1.0, 0.1, 0.1)), _box((0, 0.2, 0), (1.0, 0.2, 0.1)),
        _box((0, 1.25, 0), (0.8, 0.8, 0.05), trig=True)]},
    # an open arch: legs on the ground, the aperture reaches the ground
    "FakeArchGate01": {"colliders": [
        _box((-2.0, 1.5, 0), (0.25, 1.5, 0.25)), _box((2.0, 1.5, 0), (0.25, 1.5, 0.25)),
        _box((0, 3.0, 0), (2.25, 0.25, 0.25)), _box((0, 1.4, 0), (1.75, 1.4, 0.1), trig=True)]},
    "FakeNoTrigger01": {"colliders": [_box((0, 1.0, 0), (1.0, 1.0, 0.1))]},
    "FakeEmpty01": {"colliders": []},
}  # fmt: skip


def test_draft_gate_types_open_vs_closed():
    drafts = {d["class"]: d for d in draft_gate_types(PROPS, list(PROPS) + ["FakeMissing01"])}
    sq = drafts["fake_square_gate"]
    assert sq["match"] == "^FakeSquareGate01" and sq["bottom"] == "closed"
    assert sq["aperture"] == [1.6, 1.6] and sq["outer"] == [2.0, 2.5]
    assert sq["center_offset"] == [0.0, 1.25, 0.0]
    assert len(sq["keypoints"]) == 8 and sq["flip_pairs"] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    arch = drafts["fake_arch_gate"]
    assert arch["bottom"] == "open" and arch["aperture"] == [3.5, 2.8]
    assert arch["outer"] == [4.5, 3.25] and len(arch["keypoints"]) == 7
    assert arch["flip_pairs"] == [[0, 1], [2, 3], [4, 5]]
    assert "aperture guessed" in drafts["fake_no_trigger"]["warning"]
    assert drafts["fake_empty"]["error"] == "no collider shapes in props.json"
    assert drafts["fake_missing"]["error"] == "no collider shapes in props.json"
    # a draft is a valid gate-types file entry (bottom given)
    ok = [d for d in drafts.values() if "error" not in d]
    assert GateTypeTable.from_config(ok, require_bottom=True).classes == [d["class"] for d in ok]


# ----------------------------------------------------------------------------- CLI


def test_cli_parsers():
    p = build_parser()
    a = p.parse_args(["harness", "gates", "train", "--captures", "x", "y", "--select", "val-loss",
                      "--deploy", "--backbone", "tiny-cnn",
                      "--max-frames-per-capture", "50"])  # fmt: skip
    assert a.func.__name__ == "_cmd_gates_train" and a.captures == ["x", "y"]
    assert a.select == "val-loss" and a.deploy and a.max_frames_per_capture == 50
    assert a.protect_holdout_of == "release:v1" and a.val_every == 2 and a.epochs == 40
    assert a.frame_stride == 1 and a.device == "auto"
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "gates", "train", "--select", "best"])
    r = p.parse_args(["harness", "gates", "dataset-report", "--all-captures", "--json", "o.json",
                      "--protect-holdout-of", "none"])  # fmt: skip
    assert r.func.__name__ == "_cmd_gates_dataset_report" and r.all_captures and r.json == "o.json"
    d = p.parse_args(["harness", "gates", "draft-types", "--props", "p.json", "--items", "A", "B"])
    assert d.func.__name__ == "_cmd_gates_draft_types" and d.items == ["A", "B"]
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "gates", "draft-types", "--items", "A"])  # --props required


def test_cli_draft_types_prints_candidates(tmp_path, capsys):
    p = tmp_path / "props.json"
    p.write_text(json.dumps({"items": PROPS}), "utf-8")
    assert main(["harness", "gates", "draft-types", "--props", str(p), "--items",
                 "FakeSquareGate01"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert out.startswith("# CANDIDATES ONLY")
    assert yaml.safe_load(out)["gate_types"][0]["bottom"] == "closed"


def test_cli_dataset_report_exit_codes(world, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cat, "load_catalogue", lambda path: world.catalogue)
    cfg = tmp_path / "small.yaml"
    cfg.write_text(json.dumps(SMALL_CAMERA), "utf-8")
    args = ["harness", "gates", "dataset-report", "--captures", str(world.a), str(world.c),
            str(world.d), "--split-source", str(world.catalogue.path), "--trackdata-root",
            str(world.root), "--protect-holdout-of", "none", "--config", str(cfg),
            "--json", str(tmp_path / "rep.json")]  # fmt: skip
    assert main(args) == 4  # EXIT_COVERAGE: NO-GO
    out = capsys.readouterr().out
    assert "G1: NO-GO" in out
    rep = json.loads((tmp_path / "rep.json").read_text("utf-8"))
    assert set(rep["splits"]["held-out"]) == {"FakeD"}  # held-out counted, never assembled
    assert world.d.name not in {c["name"] for c in rep["captures"]}
    monkeypatch.setattr(gd, "G1", {k: 0 for k in gd.G1})
    monkeypatch.setitem(gd.G1, "heldout_type_frames", 1)
    assert main(args) == 4  # still NO-GO: no protected window present
    out = capsys.readouterr().out
    assert "FAIL protected_window" in out and "FAIL train_frames" not in out
