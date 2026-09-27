"""UC-61 follow-up item 5 — bounded, filtered tail of a run's raw log (``app.logs``).

Hermetic: every test drives :func:`app.logs.tail` (and the private deny-filter it uses) against a
temp fixture file — NO real subprocess, signals, network, GPU, or display. Pins:

* **Deny-filter** — known pybullet / OpenGL / thread-init boot noise (anchored, case-insensitive)
  is DROPPED, while everything else is KEPT. Critically the negative cases: a real line that merely
  *contains* ``semaphore`` / ``OpenGL`` (not the anchored boot form) survives, as do SB3 rollout
  tables, checkpoint-save messages, health verdicts, warnings, and tracebacks.
* **Bounded byte-offset cursor** — ``since`` offset tailing; the 64 KB per-poll cap with a single
  truncation marker + torn-fragment drop; a torn in-flight tail held then completed; ``since`` past
  the file size reset; an absent file yielding an empty result.
"""

from __future__ import annotations

import os

from app import logs

# --- deny-filter: noise dropped ------------------------------------------------------------

#: Lines the owner-confirmed deny-list must DROP (each is the anchored boot form).
NOISE_LINES = [
    "pybullet build time: May 20 2024 12:00:00",
    "argv[0]=--unused",
    "b3Printf: Selected demo: Physics Server",
    "b3Warning[examples/SharedMemory/PhysicsServerExample.cpp,4]",
    "b3xyz could not create semaphore handle",  # matches the anchored `b3.*semaphore`
    "ven = Intel Open Source Technology Center",
    "GL_VENDOR=Intel Open Source Technology Center",
    "GL_RENDERER=Mesa DRI Intel",
    "GL_VERSION=3.0 Mesa",
    "Vendor = Intel Open Source Technology Center",
    "Renderer = Mesa DRI Intel(R)",
    "Version = 3.0 Mesa 20.0",
    "startThreads creating 1 threads.",
    "stopThreads: Not enough space",
    "numActiveThreads = 0",
    "Thread with id 140218 crashed",
    "EGL device choosing failed",
    "GLX: failed to create context",
    "MesaGL warning: something",
    "Workaround for some crash in the Intel OpenGL driver",
    "ExampleBrowserThreadFunc started",
    "",  # blank line
    "   ",  # whitespace-only line
]

#: Lines that must be KEPT — including the negative cases the deny-list must NOT over-match.
SIGNAL_LINES = [
    "training released the semaphore after checkpoint",  # 'semaphore' NOT anchored to b3 → keep
    "WARNING: OpenGL context could not be created, falling back",  # 'OpenGL' mid-line → keep
    "| rollout/ep_rew_mean | -0.87 |",  # SB3 rollout table row
    "| time/fps            | 512  |",
    "Saving checkpoint to training/demo/checkpoints/ppo_racer_4096_steps.zip",
    "HEALTH: T/W ratio 1.99 — OK (flyable)",
    "UserWarning: entropy coefficient is quite high",
    "Traceback (most recent call last):",
    '  File "train.py", line 10, in <module>',
    "drone-fly: step 4096 / 2000000 reward 0.42",
]


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def test_deny_filter_drops_known_boot_noise(tmp_path):
    path = _write(tmp_path / "app.log", NOISE_LINES)
    assert logs.tail(str(path))["lines"] == []  # every noise line filtered out


def test_deny_filter_keeps_real_lines_including_semaphore_and_opengl(tmp_path):
    path = _write(tmp_path / "app.log", SIGNAL_LINES)
    assert logs.tail(str(path))["lines"] == SIGNAL_LINES  # nothing over-matched


def test_deny_filter_keeps_signal_when_interleaved_with_noise(tmp_path):
    interleaved = []
    for sig, noise in zip(SIGNAL_LINES, NOISE_LINES, strict=False):
        interleaved.extend([noise, sig])
    path = _write(tmp_path / "app.log", interleaved)
    kept = logs.tail(str(path))["lines"]
    assert kept == SIGNAL_LINES[: len(kept)]  # only the signal survives, in order
    assert kept == SIGNAL_LINES[: min(len(SIGNAL_LINES), len(NOISE_LINES))]


# --- cursor: since offset, cap+marker, torn tail, reset, absent file -------------------------


def test_since_offset_returns_only_new_complete_lines(tmp_path):
    path = _write(tmp_path / "app.log", ["epoch 1 reward 0.1", "epoch 2 reward 0.2"])
    first = logs.tail(str(path), since=0)
    assert first["lines"] == ["epoch 1 reward 0.1", "epoch 2 reward 0.2"]
    assert first["next"] == path.stat().st_size

    # Re-poll from the cursor → nothing new.
    assert logs.tail(str(path), since=first["next"])["lines"] == []

    # Append and re-poll → only the appended line.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("epoch 3 reward 0.3\n")
    nxt = logs.tail(str(path), since=first["next"])
    assert nxt["lines"] == ["epoch 3 reward 0.3"]
    assert nxt["next"] == path.stat().st_size


def test_bounded_cap_emits_truncation_marker_and_drops_torn_head(tmp_path):
    # A file well over the 64 KB per-poll cap read from offset 0.
    lines = [f"epoch {i:05d} reward {i % 7}" for i in range(4000)]
    path = _write(tmp_path / "app.log", lines)
    size = path.stat().st_size
    assert size > logs._MAX_READ  # precondition: the cap actually bites

    result = logs.tail(str(path), since=0)
    assert result["lines"][0] == logs._TRUNCATION_MARKER  # the skip is made visible
    assert result["next"] == size  # cursor advanced to end (file ends on a newline)
    # Only the tail within the cap is returned (the earliest lines were skipped), and the torn
    # first fragment was dropped rather than emitted as a bogus partial line.
    assert lines[0] not in result["lines"]
    assert lines[-1] in result["lines"]
    assert logs._TRUNCATION_MARKER not in lines  # marker is synthetic, not from the file
    # A subsequent poll from the returned cursor is caught up (empty).
    assert logs.tail(str(path), since=result["next"])["lines"] == []


def test_torn_tail_is_held_then_completed(tmp_path):
    path = tmp_path / "app.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A complete line followed by an in-flight partial (no trailing newline yet).
    path.write_text("complete line one\npartial second", encoding="utf-8")

    first = logs.tail(str(path), since=0)
    assert first["lines"] == ["complete line one"]  # partial tail withheld
    assert first["next"] == len(b"complete line one\n")

    # The rest of the torn line arrives.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(" now finished\n")
    second = logs.tail(str(path), since=first["next"])
    assert second["lines"] == ["partial second now finished"]  # re-read whole, once complete


def test_no_complete_line_yet_holds_cursor(tmp_path):
    path = tmp_path / "app.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("in-flight-with-no-newline", encoding="utf-8")
    result = logs.tail(str(path), since=0)
    assert result == {"lines": [], "next": 0}  # nothing complete → cursor unmoved


def test_since_beyond_filesize_resets_to_start(tmp_path):
    path = _write(tmp_path / "app.log", ["line a", "line b"])
    # A cursor past EOF (file rotated/truncated) restarts from the top rather than erroring.
    result = logs.tail(str(path), since=10_000)
    assert result["lines"] == ["line a", "line b"]
    assert result["next"] == path.stat().st_size


def test_negative_since_resets_to_start(tmp_path):
    path = _write(tmp_path / "app.log", ["line a"])
    assert logs.tail(str(path), since=-5)["lines"] == ["line a"]


def test_absent_file_returns_empty_and_preserves_cursor(tmp_path):
    missing = tmp_path / "training" / "demo" / "logs" / "app.log"
    assert logs.tail(str(missing), since=7) == {"lines": [], "next": 7}


# --- path helper ---------------------------------------------------------------------------


def test_log_path_for_builds_training_logs_app_log():
    p = logs.log_path_for("/proj", "demo")
    assert p == os.path.join("/proj", "training", "demo", "logs", "app.log")
