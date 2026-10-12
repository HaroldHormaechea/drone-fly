"""Procedural random course generator: a verbatim NumPy port of ``gpu_prototype/course_gen.py``.

Every random draw and every arithmetic expression is kept in the prototype's order, so
``sample_course(seed, **kw)`` returns the same course as the prototype for the same arguments
(tests compare them through ``harness.contract.import_reference``). Only the layout changed (ruff
formatting) and the prototype's ``__main__`` spec-validation script was dropped.

A course is a circular closed loop (laps), 5-20 gates each rotated to its travel direction, varied
gate heights, a chicane, pillar / box obstacles (one blocking, one wide fly-over block), optional
tunnels and varied shapes, consecutive spacing 1-5 m, and a start on the ground before gate 0.
Plain Python / NumPy, backend-agnostic: :mod:`drone_fly.sim.course` packs courses into tensors.
"""

from __future__ import annotations

import math

import numpy as np

DRONE_WIDTH = 0.15  # ~Meteor75 tip-to-tip envelope (m); apertures are sized in these
APERTURE_MIN = 3 * DRONE_WIDTH  # tightest gate = 3x drone width (0.45 m)
APERTURE_MAX = 10 * DRONE_WIDTH  # widest gate = 10x drone width (1.50 m)
APERTURE = 1.0  # nominal reference (plotting only); real apertures are per-gate random
GATE_Z_MIN = 0.0  # center at 0 => bottom half under the floor ("ground level" gate)
GATE_Z_MAX = 2.0  # up to ~2 m above the floor
MIN_SPACING, MAX_SPACING = 1.0, 5.0
START_Z = 0.12  # the start is on the ground


def _rng(seed):
    return np.random.default_rng(seed)


def _gen_centers(r, ng, target_chord):
    """One candidate loop of gate centers (perturbed circle + smooth varied heights + chicane)."""
    R = target_chord / (2 * math.sin(math.pi / ng))
    ang = np.array([2 * math.pi * i / ng for i in range(ng)]) + r.normal(0, 0.05, ng)
    # ABSOLUTE radial perturbation (metres, independent of R): spacing doesn't blow up on big loops
    rr = R + r.normal(0, 0.22, ng)
    # CHICANE: 3 consecutive gates alternate in/out by a bounded absolute amount
    c0 = int(r.integers(0, ng))
    for j, sgn in zip(range(3), (+1, -1, +1), strict=True):
        rr[(c0 + j) % ng] += 0.55 * sgn
    cx, cy = rr * np.cos(ang), rr * np.sin(ang)
    # SMOOTH heights: full 0..2 range but bounded adjacent change (1-2 sinusoids + small noise)
    k = int(r.integers(1, 3))
    phase = r.uniform(0, 2 * math.pi)
    cz = 1.0 + 0.98 * np.sin(k * ang + phase) + r.normal(0, 0.08, ng)
    cz = np.clip(cz, GATE_Z_MIN, GATE_Z_MAX)
    return np.stack([cx, cy, cz], axis=1), c0


def sample_course(
    seed,
    ng_lo=5,
    ng_hi=21,
    laps_lo=2,
    laps_hi=4,
    ap_lo=APERTURE_MIN,
    ap_hi=APERTURE_MAX,
    with_tunnels=False,
    with_variety=False,
):
    r = _rng(seed)
    ng = int(r.integers(ng_lo, ng_hi))  # gate count range (default 5..20)
    laps = int(r.integers(laps_lo, laps_hi))  # lap range (default 2..3)
    # rejection-sample a loop whose every consecutive spacing (incl. the closing link) is in [1,5]
    centers, c0 = None, 0
    for _try in range(40):
        tc = r.uniform(1.8, 3.0)
        cand, c0 = _gen_centers(r, ng, tc)
        sp = [float(np.linalg.norm(cand[(i + 1) % ng] - cand[i])) for i in range(ng)]
        if all(MIN_SPACING <= d <= MAX_SPACING for d in sp):
            centers = cand
            break
    if centers is None:
        centers = cand  # fallback: best-effort (rare)

    # gate orientation: yaw = travel direction (prev -> next tangent), projected to xy
    yaw = np.zeros(ng)
    for i in range(ng):
        nxt, prv = centers[(i + 1) % ng], centers[(i - 1) % ng]
        t = nxt - prv
        yaw[i] = math.atan2(t[1], t[0])

    # start ON THE GROUND, just "before" gate 0 along the reversed gate0->gate1 heading
    d01 = centers[1] - centers[0]
    d01 = d01 / (np.linalg.norm(d01) + 1e-9)
    start = centers[0] - d01 * 2.0
    start[2] = START_Z  # on the ground

    apertures = r.uniform(ap_lo, ap_hi, ng)  # per-gate aperture (default 3x..10x drone width)
    gates = [
        {"center": centers[i].tolist(), "yaw": float(yaw[i]), "aperture": float(apertures[i])}
        for i in range(ng)
    ]

    # ---- obstacles ----
    # Each obstacle carries a unit "ovec" perpendicular (in xy) to the gate segment it sits on, so
    # the env can PARK it off to the side early in training and slide it onto the path.
    def _perp(seg):
        v = np.array([-seg[1], seg[0], 0.0], np.float64)
        n = np.linalg.norm(v)
        return (v / n).tolist() if n > 1e-6 else [1.0, 0.0, 0.0]

    obstacles = []
    # (a) a BLOCKING obstacle on the straightest consecutive gate pair (lowest turn angle)
    turn = np.full(ng, 9.9)
    for i in range(ng):
        a = centers[i] - centers[(i - 1) % ng]
        b = centers[(i + 1) % ng] - centers[i]
        cross2d = float(a[0] * b[1] - a[1] * b[0])
        turn[i] = abs(math.atan2(cross2d, float(np.dot(a[:2], b[:2]))))
    straight_i = int(np.argmin(turn))
    g0, g1 = centers[straight_i], centers[(straight_i + 1) % ng]
    mid = 0.5 * (g0 + g1)
    perp_a = _perp(g1 - g0)
    if r.random() < 0.5:
        # pillar that blocks the gap: a tall thin cylinder on the mid-line
        obstacles.append(
            {
                "kind": "cylinder",
                "center": [mid[0], mid[1], 1.0],
                "radius": 0.35,
                "half_h": 1.5,
                "blocks": True,
                "ovec": perp_a,
            }
        )
    else:
        # full block spanning the gate line: a box narrow across the path, tall
        obstacles.append(
            {
                "kind": "box",
                "center": [mid[0], mid[1], 0.9],
                "half": [0.45, 0.45, 0.9],
                "blocks": True,
                "ovec": perp_a,
            }
        )
    # (b) a WIDE-FLAT block to fly OVER: between another gate pair, wider than tall, low
    j = (straight_i + ng // 2) % ng
    gj, gk = centers[j], centers[(j + 1) % ng]
    midj = 0.5 * (gj + gk)
    obstacles.append(
        {
            "kind": "box",
            "center": [midj[0], midj[1], 0.4],
            "half": [0.9, 0.9, 0.3],
            "blocks": False,
            "fly_over": True,
            "ovec": _perp(gk - gj),
        }
    )
    # (c) optional extra pillar(s)
    for _ in range(int(r.integers(0, 2))):
        k = int(r.integers(0, ng))
        mk = 0.5 * (centers[k] + centers[(k + 1) % ng]) + r.normal(0, 0.3, 3)
        obstacles.append(
            {
                "kind": "cylinder",
                "center": [float(mk[0]), float(mk[1]), 1.0],
                "radius": float(r.uniform(0.2, 0.35)),
                "half_h": 1.5,
                "blocks": False,
                "ovec": _perp(centers[(k + 1) % ng] - centers[k]),
            }
        )

    # (d) TUNNEL / doorway: 3 axis-aligned boxes around a gate-segment midpoint -- a clear central
    # lane, a wall on each side and a ceiling. Placed a third of the way round from the blocking
    # obstacle so they don't overlap.
    if with_tunnels:
        tn = (straight_i + ng // 3) % ng
        a0, a1 = centers[tn], centers[(tn + 1) % ng]
        midt = 0.5 * (a0 + a1)
        px, py, _ = _perp(a1 - a0)  # unit perpendicular in xy
        zc = float(midt[2])  # gate-height of this segment
        GW = 0.65  # half-width of the clear central lane
        WALL = 0.55  # wall box half-extent (xy)
        off = GW + WALL  # wall-centre offset along the perpendicular
        for sgn in (+1.0, -1.0):  # left + right walls
            # ovec points OUTWARD so the position curriculum starts the walls far apart (wide,
            # easy gap) and slides them in to the designed gap.
            obstacles.append(
                {
                    "kind": "box",
                    "center": [
                        float(midt[0] + sgn * px * off),
                        float(midt[1] + sgn * py * off),
                        1.0,
                    ],
                    "half": [WALL, WALL, 1.6],
                    "blocks": True,
                    "tunnel": True,
                    "ovec": [float(sgn * px), float(sgn * py), 0.0],
                }
            )
        obstacles.append(
            {  # ceiling over the lane (starts high, lowers in)
                "kind": "box",
                "center": [float(midt[0]), float(midt[1]), zc + 1.1],
                "half": [off + WALL, off + WALL, 0.3],
                "blocks": True,
                "tunnel": True,
                "ovec": [0.0, 0.0, 1.0],
            }
        )

    # (e) VARIED SHAPES: a fly-UNDER overhang (a horizontal bar above gate height) and a tall WALL
    # (a bigger lateral detour than a thin pillar).
    if with_variety:
        jo = (straight_i + 2 * ng // 3) % ng
        go0, go1 = centers[jo], centers[(jo + 1) % ng]
        mo = 0.5 * (go0 + go1)
        zco = float(mo[2])
        obstacles.append(
            {
                "kind": "box",
                "center": [float(mo[0]), float(mo[1]), zco + 1.0],
                "half": [1.1, 1.1, 0.25],
                "blocks": True,
                "fly_under": True,
                "ovec": [0.0, 0.0, 1.0],
            }
        )  # overhang: lower it onto the lane over the curriculum
        jw = (straight_i + ng // 4) % ng
        gw0, gw1 = centers[jw], centers[(jw + 1) % ng]
        mw = 0.5 * (gw0 + gw1)
        pw = _perp(gw1 - gw0)
        obstacles.append(
            {
                "kind": "box",
                "center": [float(mw[0]), float(mw[1]), 1.0],
                "half": [0.9, 0.3, 1.5],
                "blocks": True,
                "wall": True,
                "ovec": pw,
            }
        )

    # Keep the START spawn clear of obstacles: on small loops the wide fly-over box can land on
    # the start point. Push any offending obstacle just outside the spawn clearance.
    sx, sy, sz = float(start[0]), float(start[1]), float(start[2])
    for o in obstacles:
        if o.get("tunnel"):
            continue  # tunnel walls are fixed structure; never nudge them
        cx, cy, cz = o["center"]
        rxy = o["radius"] if o["kind"] == "cylinder" else max(o["half"][0], o["half"][1])
        hz = o["half_h"] if o["kind"] == "cylinder" else o["half"][2]
        dxy = math.hypot(sx - cx, sy - cy)
        need = rxy + 0.4
        if abs(sz - cz) < hz + 0.2 and dxy < need:
            if dxy > 1e-3:
                ux, uy = (cx - sx) / dxy, (cy - sy) / dxy
            else:
                ux, uy = o["ovec"][0], o["ovec"][1]
            o["center"][0] = sx + ux * need
            o["center"][1] = sy + uy * need

    # spacing, incl. the lap-closing gateN->gate0 link
    spac = [float(np.linalg.norm(centers[(i + 1) % ng] - centers[i])) for i in range(ng)]
    return {
        "seed": int(seed),
        "n_gates": ng,
        "laps": laps,
        "gates": gates,
        "start": start.tolist(),
        "obstacles": obstacles,
        "spacing": spac,
    }
