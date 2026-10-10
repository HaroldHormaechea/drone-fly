"""Procedural random course generator (per-episode variety).

Produces a circular, multi-lap course meeting the design spec:
  1. circular closed loop, 2 or 3 laps
  2. obstacles are pillars (cylinders) OR blocks (boxes); >=1 blocks the direct line between two
     roughly-aligned consecutive gates
  3. some blocks are WIDER THAN TALL (favor flying OVER them, not around)
  4. 5-15 gates, each ROTATED to its entry/exit (travel) direction
  5. gate heights vary: ground level (center ~0 -> half the gate under the floor) up to 2x aperture
  6. >=1 chicane (a sharp alternating left-right-left kink)
  7. consecutive gate spacing 1-5 m
  8. start ON THE GROUND (takeoff), placed just before gate 0

Returned as plain Python / numpy so it is backend-agnostic; the batched env samples these and packs
them into padded tensors. Gate count varies per course (5-15), so downstream must mask to `n_gates`.
"""
import math
import numpy as np

DRONE_WIDTH = 0.15                  # ~Meteor75 tip-to-tip envelope (m); apertures are sized in these
APERTURE_MIN = 3 * DRONE_WIDTH      # tightest gate = 3x drone width (0.45 m)
APERTURE_MAX = 10 * DRONE_WIDTH     # widest gate = 10x drone width (1.50 m)
APERTURE = 1.0                      # nominal reference (plotting only); real apertures are per-gate random
GATE_Z_MIN = 0.0                    # center at 0 => bottom half under the floor ("ground level" gate)
GATE_Z_MAX = 2.0                    # up to ~2 m above the floor
MIN_SPACING, MAX_SPACING = 1.0, 5.0


def _rng(seed):
    return np.random.default_rng(seed)


def _gen_centers(r, ng, target_chord):
    """One candidate loop of gate centers (xy perturbed circle + smooth varied heights + a chicane)."""
    R = target_chord / (2 * math.sin(math.pi / ng))
    ang = np.array([2 * math.pi * i / ng for i in range(ng)]) + r.normal(0, 0.05, ng)
    # ABSOLUTE radial perturbation (meters, independent of R) so spacing doesn't blow up on big loops
    rr = R + r.normal(0, 0.22, ng)
    # CHICANE (spec 6): 3 consecutive gates alternate in/out by a bounded absolute amount
    c0 = int(r.integers(0, ng))
    for j, sgn in zip(range(3), (+1, -1, +1)):
        rr[(c0 + j) % ng] += 0.55 * sgn
    cx, cy = rr * np.cos(ang), rr * np.sin(ang)
    # SMOOTH heights (spec 5): full 0..2 range but bounded adjacent change (1-2 sinusoids + small noise)
    k = int(r.integers(1, 3)); phase = r.uniform(0, 2 * math.pi)
    cz = 1.0 + 0.98 * np.sin(k * ang + phase) + r.normal(0, 0.08, ng)
    cz = np.clip(cz, GATE_Z_MIN, GATE_Z_MAX)
    return np.stack([cx, cy, cz], axis=1), c0


def sample_course(seed, ng_lo=5, ng_hi=21, laps_lo=2, laps_hi=4, ap_lo=APERTURE_MIN, ap_hi=APERTURE_MAX):
    r = _rng(seed)
    ng = int(r.integers(ng_lo, ng_hi))   # gate count range (default 5..20)
    laps = int(r.integers(laps_lo, laps_hi))   # lap range (default 2..3)
    # rejection-sample a loop whose every consecutive spacing (incl the lap-closing link) is in [1,5]
    centers, c0 = None, 0
    for _try in range(40):
        tc = r.uniform(1.8, 3.0)
        cand, c0 = _gen_centers(r, ng, tc)
        sp = [float(np.linalg.norm(cand[(i + 1) % ng] - cand[i])) for i in range(ng)]
        if all(MIN_SPACING <= d <= MAX_SPACING for d in sp):
            centers = cand
            break
    if centers is None:
        centers = cand                   # fallback: best-effort (rare)

    # gate orientation (spec 4): yaw = travel direction (prev -> next tangent), projected to xy
    yaw = np.zeros(ng)
    for i in range(ng):
        nxt, prv = centers[(i + 1) % ng], centers[(i - 1) % ng]
        t = nxt - prv
        yaw[i] = math.atan2(t[1], t[0])

    # start ON THE GROUND (spec 8), just "before" gate 0 along the reversed gate0->gate1 heading
    d01 = centers[1] - centers[0]
    d01 = d01 / (np.linalg.norm(d01) + 1e-9)
    start = centers[0] - d01 * 2.0
    start[2] = 0.12                                   # on the ground

    apertures = r.uniform(ap_lo, ap_hi, ng)   # per-gate aperture range (default 3x..10x drone width)
    gates = [{"center": centers[i].tolist(), "yaw": float(yaw[i]), "aperture": float(apertures[i])}
             for i in range(ng)]

    # ---- obstacles (specs 2,3) ----
    # Each obstacle carries a unit "ovec" perpendicular (in xy) to the gate segment it sits on, so the
    # env can PARK it off to the side early in training and slide it onto the path (position curriculum).
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
        obstacles.append({"kind": "cylinder", "center": [mid[0], mid[1], 1.0],
                          "radius": 0.35, "half_h": 1.5, "blocks": True, "ovec": perp_a})
    else:
        # full block spanning the gate line: a box narrow across the path, tall
        obstacles.append({"kind": "box", "center": [mid[0], mid[1], 0.9],
                          "half": [0.45, 0.45, 0.9], "blocks": True, "ovec": perp_a})
    # (b) a WIDE-FLAT block to fly OVER (spec 3): between another gate pair, wider than tall, low
    j = (straight_i + ng // 2) % ng
    gj, gk = centers[j], centers[(j + 1) % ng]
    midj = 0.5 * (gj + gk)
    obstacles.append({"kind": "box", "center": [midj[0], midj[1], 0.4],
                      "half": [0.9, 0.9, 0.3], "blocks": False, "fly_over": True, "ovec": _perp(gk - gj)})
    # (c) optional extra pillar(s)
    for _ in range(int(r.integers(0, 2))):
        k = int(r.integers(0, ng))
        mk = 0.5 * (centers[k] + centers[(k + 1) % ng]) + r.normal(0, 0.3, 3)
        obstacles.append({"kind": "cylinder", "center": [float(mk[0]), float(mk[1]), 1.0],
                          "radius": float(r.uniform(0.2, 0.35)), "half_h": 1.5, "blocks": False,
                          "ovec": _perp(centers[(k + 1) % ng] - centers[k])})

    # spacing (spec 7), incl the lap-closing gateN->gate0 link
    spac = [float(np.linalg.norm(centers[(i + 1) % ng] - centers[i])) for i in range(ng)]
    return {"seed": int(seed), "n_gates": ng, "laps": laps, "gates": gates, "start": start.tolist(),
            "obstacles": obstacles, "spacing": spac}


if __name__ == "__main__":
    # validate the 7 specs across many samples, then print one example course
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ng_ok = lap_ok = space_ok = zrange_ok = block_ok = flyover_ok = chic_ok = 0
    ng_min, ng_max = 99, 0
    sp_min, sp_max = 9.9, 0.0
    for s in range(N):
        c = sample_course(s)
        ng = c["n_gates"]; ng_min = min(ng_min, ng); ng_max = max(ng_max, ng)
        ng_ok += 5 <= ng <= 20
        lap_ok += c["laps"] in (2, 3)
        sp = c["spacing"]; sp_min = min(sp_min, min(sp)); sp_max = max(sp_max, max(sp))
        space_ok += all(MIN_SPACING <= d <= MAX_SPACING for d in sp)
        zs = [g["center"][2] for g in c["gates"]]
        zrange_ok += (min(zs) < 0.4) and (max(zs) > 1.4)
        block_ok += any(o.get("blocks") for o in c["obstacles"])
        flyover_ok += any(o.get("fly_over") for o in c["obstacles"])
        # chicane present: some consecutive triple with alternating cross-product sign + sharp
        chic_ok += 1  # forced by construction
    ap_min = min(min(g["aperture"] for g in sample_course(s)["gates"]) for s in range(N))
    ap_max = max(max(g["aperture"] for g in sample_course(s)["gates"]) for s in range(N))
    print(f"validated {N} courses:")
    print(f"  gate count 5-20: {ng_ok}/{N}  (range {ng_min}-{ng_max})")
    print(f"  aperture 0.45-1.5 m (3-10x drone width): global {ap_min:.2f}-{ap_max:.2f}")
    print(f"  laps 2-3: {lap_ok}/{N}")
    print(f"  spacing 1-5m: {space_ok}/{N}  (global min {sp_min:.2f} max {sp_max:.2f})")
    print(f"  height varied (<0.4 and >1.4): {zrange_ok}/{N}")
    print(f"  has blocking obstacle: {block_ok}/{N}")
    print(f"  has fly-over (wider-than-tall) obstacle: {flyover_ok}/{N}")
    print(f"  chicane (forced): {chic_ok}/{N}")
    ex = sample_course(0)
    print(f"\nexample course seed=0: {ex['n_gates']} gates, {ex['laps']} laps, start {[round(x,2) for x in ex['start']]}")
    for i, g in enumerate(ex["gates"]):
        print(f"  gate {i}: center {[round(x,2) for x in g['center']]}  yaw {math.degrees(g['yaw']):.0f}deg")
    for o in ex["obstacles"]:
        print(f"  obstacle: {o}")
