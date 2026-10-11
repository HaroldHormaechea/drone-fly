"""Trim the committed real-telemetry fixture ``liftoff_real_datagrams.csv`` (UC-64 AC2).

Source: the hand-flown reference capture ``artifacts/harness/captures/2026-10-11-infinity-manual``
(Liftoff Arena, race "04 - Infinity Loop", micro "Test small drone"; gitignored, not committed).
Its ``telemetry_raw.csv`` holds one raw UDP datagram per row (``t_wall_ns,len,hex``, 97 bytes:
the 23 configured floats + a uint8 MotorRPM count prefix). The fixture keeps data rows
``START_ROW`` .. ``START_ROW + N_ROWS`` (0-based, header excluded) verbatim: ~4 s of fast
flying in the second half of the capture with rotation on all three axes (gyro std > 80 deg/s
per channel). It passes ``gyro_check`` under the measured map (wx = -roll, wy = +pitch,
wz = -yaw) with |corr| > 0.99 and |scale| within 1 +- 0.01, and fails under the UC-62 signs.

Only telemetry bytes are copied — no game frames, track XMLs or props.

    uv run python tests/data/harness/extract_real_datagrams.py <capture-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

START_ROW = 24450
N_ROWS = 600


def main(capture: str) -> None:
    src = Path(capture) / "telemetry_raw.csv"
    lines = src.read_text(encoding="ascii").splitlines()
    header, rows = lines[0], lines[1:]
    out = Path(__file__).with_name("liftoff_real_datagrams.csv")
    out.write_text(
        "\n".join([header, *rows[START_ROW : START_ROW + N_ROWS]]) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main(sys.argv[1])
