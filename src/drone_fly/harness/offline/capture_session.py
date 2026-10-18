"""Capture commands (UC71 AC3/AC4): ``captures record`` / ``captures replay`` over the recorder.

:func:`record_manual` records a hand-flown capture of a catalogue track; :func:`record_replay`
plays a stored Liftoff recording and captures it (strategy A: frames + UDP, B: frames + the
recording XML pose, C: refuse with :data:`MANUAL_FALLBACK`). Both register the capture's
``meta.json`` (:mod:`~drone_fly.harness.offline.capture_index`).

They live in the offline package because they read recording / track XMLs and the catalogue; the
runtime recorder itself (:class:`~drone_fly.harness.liftoff.recorder.CaptureRecorder`,
:func:`~drone_fly.harness.liftoff.recorder.write_pose_capture`) stays in ``harness/liftoff``,
which never imports ``offline``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff.recorder import (
    DEFAULT_FPS,
    DEFAULT_PORT,
    CaptureRecorder,
    write_pose_capture,
)

log = logging.getLogger(__name__)


#: The AC3 capture strategy: "A" (UDP telemetry streams during replay playback), "B" (frames +
#: the recording XML pose, aligned by `captures verify`) or "C" (neither: manual fallback only).
#: P0a found a 10 Hz pose in every recording; P1 (AC1b) measured no UDP during replay: B.
REPLAY_STRATEGY = "B"
#: Extra masks for the replay HUD on top of the preset's (MEASURED 2026-10-13: none needed). The
#: replay shows the live OSD (ALT/SPD, stick widgets) plus a "<pilot> - <time>" label at
#: x 2280-2540, y 524-554, inside the preset's ghost_list mask; the playback bar hides in play.
REPLAY_HUD_MASKS: list = []

MANUAL_FALLBACK = """\
Replay capture is not possible ({why}). Capture the map by hand instead
(docs/harness/liftoff-captures.md, "AC4. Manual fallback"):
  1. load the catalogue track {track} in Liftoff with the preset drone;
  2. drone-fly harness captures record --track {track} --max-s 600;
  3. fly 3-4 laps, varying height, speed and line;
  4. stop the recorder (Ctrl+C);
  5. drone-fly harness captures verify <the new session folder> --search-lag"""


def record_manual(
    catalogue_id: str,
    *,
    max_s: float | None = None,
    fps: float = DEFAULT_FPS,
    port: int = DEFAULT_PORT,
    captures_root: str | Path | None = None,
    trackdata_root: str | Path | None = None,
    preset: str | None = "liftoff",
) -> tuple[Path, dict[str, Any]]:
    """``captures record`` (AC4): record a hand-flown capture of a catalogue track + its meta."""
    from datetime import date

    from drone_fly.harness.offline import capture_index as ci
    from drone_fly.tracks import catalogue as cat

    entry = cat.load_named("liftoff").by_id(catalogue_id)
    root = Path(captures_root) if captures_root else ci.captures_root()
    out = ci.next_session_dir(root, entry["map"], date.today().isoformat(), "manual")
    log.info("recording %s -> %s (Ctrl+C to stop)", catalogue_id, out)
    stats = CaptureRecorder(out, port=port, fps=fps).run(max_s)
    if not stats["datagrams"]:
        raise HarnessConfigError(f"no telemetry arrived on UDP {port} (is Liftoff flying?)")
    meta = ci.register(out, catalogue_id, preset=preset, trackdata_root=trackdata_root)
    return out, {"stats": stats, "meta": meta}


def find_recording(name: str, folder: str | Path | None = None) -> Path:
    """A recording by file stem (or a path to its XML)."""
    from drone_fly.harness.offline.recordings import iter_recordings

    p = Path(name)
    if p.suffix.lower() == ".xml" and p.is_file():
        return p
    hits = [q for q in iter_recordings(folder) if q.stem == name]
    if len(hits) != 1:
        raise HarnessConfigError(
            f"recording {name!r}: {len(hits)} matches in the Recordings folder (need exactly 1)"
        )
    return hits[0]


def record_replay(
    recording: str,
    *,
    max_s: float | None = None,
    fps: float = DEFAULT_FPS,
    port: int = DEFAULT_PORT,
    strategy: str = "auto",
    recordings_dir: str | Path | None = None,
    captures_root: str | Path | None = None,
    trackdata_root: str | Path | None = None,
    preset: str | None = "liftoff",
    allow_pending_camera: bool = False,
    browser_pos: tuple[int | None, int | None, int] | None = None,
    dropdown_counts: tuple[int | None, int | None] = (None, None),
) -> tuple[Path, dict[str, Any]]:
    """``captures replay`` (AC3): play a stored recording in Liftoff and capture it.

    ``browser_pos`` = ``(environment index, race index, list row)`` in the replay browser
    (:func:`~drone_fly.harness.liftoff.ui_replay.select_recording`; None keeps a filter). A wrong
    pick fails ``captures verify`` (the replayed pose does not match the frames).

    Strategy A records frames + UDP; B records frames and rebuilds the pose from the recording XML
    (:func:`write_pose_capture`); C refuses with the manual fallback. The capture's ``meta.json``
    links the catalogue entry and the recording (name + sha256); ``captures verify --search-lag``
    then sets ``verified``.
    """
    from datetime import date

    from drone_fly.harness.config import load_config
    from drone_fly.harness.liftoff import ui_replay
    from drone_fly.harness.offline import align
    from drone_fly.harness.offline import capture_index as ci
    from drone_fly.harness.offline import recordings as rc
    from drone_fly.tracks import catalogue as cat

    strat = REPLAY_STRATEGY if strategy == "auto" else strategy
    if strat not in ("A", "B", "C"):
        raise HarnessConfigError(f"--strategy must be auto, A, B or C, got {strategy!r}")
    path = find_recording(recording, recordings_dir)
    rec = rc.parse_recording(path)
    catalogue = cat.load_named("liftoff")
    entry, how = rc.match_entry(rec, catalogue)
    track = entry["id"] if entry else "<catalogue-id>"
    if strat == "C":
        raise HarnessConfigError(MANUAL_FALLBACK.format(why="strategy C", track=track))
    if entry is None:
        raise HarnessConfigError(f"recording {rec.name!r} matches no catalogue entry")
    cfg = load_config("liftoff", None, preset=preset)
    cam = cfg.section("camera")
    if rec.uptilt_deg is None or abs(rec.uptilt_deg - float(cam["uptilt_deg"])) > rc.UPTILT_TOL_DEG:
        if entry.get("split") != "train":
            raise HarnessConfigError(
                f"recording {rec.name!r} uses another drone/camera ({rec.drone}, uptilt "
                f"{rec.uptilt_deg}) on a {entry.get('split')} map: val / held-out captures stay "
                "on the preset camera (ADR 0010)"
            )
        if not allow_pending_camera:
            raise HarnessConfigError(
                f"recording {rec.name!r} uses another drone/camera ({rec.drone}, uptilt "
                f"{rec.uptilt_deg}): train maps only; pass --allow-pending-camera to capture it"
            )
    if browser_pos is None:
        raise HarnessConfigError(
            "pass --browser-pos ENV,RACE,ROW: the replay browser's dropdown / list indices of the "
            "recording (the driver cannot read the entries' text)"
        )
    root = Path(captures_root) if captures_root else ci.captures_root()
    out = ci.next_session_dir(root, entry["map"], date.today().isoformat(), "replay")
    duration = rec.duration_s + 2.0
    limit = duration if max_s is None else min(max_s, duration)
    out.mkdir(parents=True, exist_ok=True)
    env_i, race_i, row = browser_pos
    ui_replay.ensure_running()
    ui_replay.open_replays()
    ui_replay.select_recording(
        env_i, race_i, row, str(out / "browser.png"),
        env_count=dropdown_counts[0], race_count=dropdown_counts[1],
    )  # fmt: skip
    recorder = CaptureRecorder(out, port=port, fps=fps, telemetry=strat == "A")
    recorder.start()  # frames before Play have no pose and are dropped when loading
    try:
        t_play = ui_replay.play()
        lost = ui_replay.watch_focus(t_play + limit)
    finally:
        stats = recorder.stop()
        ui_replay.stop_to_menu()
    if lost:
        raise HarnessConfigError(
            f"another window took the focus during the replay at "
            f"{[round(t - t_play, 1) for t in lost]} s: the frames in {out} are not all "
            "Liftoff's; discard the capture and replay it"
        )
    if strat == "B":
        write_pose_capture(out, rec, t0_frame_s=t_play)
    comp = rc.completeness_from_laps(rec, entry) or {
        "status": "unknown", "source": "none", "passed": None, "required": None}  # fmt: skip
    meta = ci.build_meta(
        out,
        entry,
        camera={**cam, "uptilt_deg": rec.uptilt_deg or cam["uptilt_deg"]},
        drone=str(cfg.data.get("drone") or "unknown"),
        source="replay",
        recording={"name": rec.name, "sha256": rec.sha256(), "origin": rec.source},
        completeness=comp,
        pose_source="udp" if strat == "A" else "recording-xml",
        time={"offset_s": 0.0, "rate": 1.0},
        masks=list(REPLAY_HUD_MASKS),
        camera_source="recording" if rec.uptilt_deg is not None else "preset",
    )
    ci.write_meta(out, meta)
    res = align.verify_capture(out, search_lag=True, trackdata_root=trackdata_root, preset=preset)
    return out, {"stats": stats, "match": how, "alignment": res.alignment_block()}
