"""UC-61 AC8 — recording enumeration + server-side gunzip (``app.recordings``).

Hermetic filesystem tests. Pins the AC8 contract that makes the embedded viewer independent of the
system webview's ``DecompressionStream``:

* Enumeration lists ``episode_<n>.json[.gz]`` sorted by episode index, flagging gzip + size.
* Loading a plain ``.json`` and a gzip ``.json.gz`` BOTH return the same parsed JSON object
  (server-side gunzip), which is what the front-end hands to ``window.loadDocument``.
* A missing episode raises ``FileNotFoundError`` (→ HTTP 404); invalid JSON raises ``ValueError``.
* A crafted ``name``/episode cannot escape the run's recordings directory (path-traversal guard).
"""

from __future__ import annotations

import gzip
import json

import pytest
from app import recordings

_DOC = {"schema": "drone-fly-recording", "frames": [{"t": 0}, {"t": 1}], "name": "ep"}


def _rec_dir(tmp_path, name="demo"):
    d = tmp_path / "training" / name / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_list_recordings_sorted_with_gzip_flag(tmp_path):
    d = _rec_dir(tmp_path)
    (d / "episode_2.json").write_text(json.dumps(_DOC), encoding="utf-8")
    with gzip.open(d / "episode_10.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(_DOC, fh)
    (d / "episode_1.json").write_text(json.dumps(_DOC), encoding="utf-8")
    (d / "not-a-recording.txt").write_text("ignore me", encoding="utf-8")

    listed = recordings.list_recordings(str(tmp_path), "demo")
    assert [e["episode"] for e in listed] == [1, 2, 10]  # numeric sort, non-recordings excluded
    by_ep = {e["episode"]: e for e in listed}
    assert by_ep[10]["gzip"] is True
    assert by_ep[2]["gzip"] is False
    assert by_ep[1]["bytes"] > 0


def test_list_recordings_empty_when_no_dir(tmp_path):
    assert recordings.list_recordings(str(tmp_path), "ghost") == []


def test_load_plain_json_recording(tmp_path):
    d = _rec_dir(tmp_path)
    (d / "episode_3.json").write_text(json.dumps(_DOC), encoding="utf-8")
    assert recordings.load_recording(str(tmp_path), "demo", 3) == _DOC


def test_load_gzip_recording_gunzipped_server_side(tmp_path):
    d = _rec_dir(tmp_path)
    with gzip.open(d / "episode_4.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(_DOC, fh)
    # Decompressed + parsed server-side → the same object as the plain-json path (AC8).
    assert recordings.load_recording(str(tmp_path), "demo", 4) == _DOC


def test_plain_json_preferred_over_gz_for_same_episode(tmp_path):
    d = _rec_dir(tmp_path)
    (d / "episode_5.json").write_text(json.dumps({"which": "plain"}), encoding="utf-8")
    with gzip.open(d / "episode_5.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({"which": "gz"}, fh)
    assert recordings.load_recording(str(tmp_path), "demo", 5) == {"which": "plain"}


def test_missing_episode_raises_file_not_found(tmp_path):
    _rec_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        recordings.load_recording(str(tmp_path), "demo", 99)


def test_invalid_json_raises_value_error(tmp_path):
    d = _rec_dir(tmp_path)
    (d / "episode_6.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        recordings.load_recording(str(tmp_path), "demo", 6)


def test_invalid_gzip_json_raises_value_error(tmp_path):
    d = _rec_dir(tmp_path)
    with gzip.open(d / "episode_7.json.gz", "wt", encoding="utf-8") as fh:
        fh.write("still-not-json")
    with pytest.raises(ValueError):
        recordings.load_recording(str(tmp_path), "demo", 7)
