"""History frame selection by time stride (deployable; shared by the dataset and the estimator).

Captures run at ~10-12 fps and the live loop at up to 40 Hz, so history is chosen by **time**, not
by frame count: for frame ``i`` at ``t_i`` and ``j = 1 .. K-1``, the frame nearest to
``t_i - j * dt`` within ``+-tol`` (40 ms), strictly earlier than ``t_i``, in the same segment
(a telemetry cut or estimator reset starts a new one). A frame is never used twice: if two strides
pick the same frame, or any stride has no candidate, there is no history (``None``) -- missing
history is reported, never filled by duplicating frames.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Default stride tolerance (seconds).
HIST_TOL_S = 0.04


def select_history(
    times: Sequence[float] | np.ndarray,
    i: int,
    k_frames: int,
    dt_s: float,
    tol_s: float = HIST_TOL_S,
    segments: Sequence[int] | np.ndarray | None = None,
) -> list[int] | None:
    """Indices ``[i_1, ..., i_{K-1}]`` (most recent first) of frame ``i``'s history, or ``None``.

    ``times`` must be ascending; ``segments`` (same length, optional) restricts candidates to
    frame ``i``'s segment. ``K = 1`` returns ``[]``.
    """
    t = np.asarray(times, np.float64)
    seg = None if segments is None else np.asarray(segments)
    ti = float(t[i])
    out: list[int] = []
    for j in range(1, int(k_frames)):
        want = ti - j * float(dt_s)
        lo = int(np.searchsorted(t, want - tol_s, side="left"))
        hi = int(np.searchsorted(t, want + tol_s, side="right"))
        hi = min(hi, i)  # strictly earlier than frame i
        cand = np.arange(lo, hi)
        if seg is not None and len(cand):
            cand = cand[seg[cand] == seg[i]]
        if not len(cand):
            return None
        best = int(cand[np.argmin(np.abs(t[cand] - want))])
        if best in out or best == i:
            return None
        out.append(best)
    return out
