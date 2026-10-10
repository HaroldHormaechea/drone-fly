"""Render example procedural courses as 3D perspective views (gates as oriented rings at their real
heights, pillars as cylinders, blocks as 3D boxes, closed-loop route, ground start)."""
import math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from course_gen import sample_course, APERTURE

seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2]


def cylinder(ax, x, y, r, z0, z1, color, alpha=0.5):
    th = np.linspace(0, 2 * math.pi, 24)
    TH, ZZ = np.meshgrid(th, np.array([z0, z1]))
    ax.plot_surface(x + r * np.cos(TH), y + r * np.sin(TH), ZZ, color=color, alpha=alpha,
                    linewidth=0, shade=True)


def box(ax, cx, cy, cz, hx, hy, hz, color, alpha=0.45):
    ax.bar3d(cx - hx, cy - hy, cz - hz, 2 * hx, 2 * hy, 2 * hz, color=color, alpha=alpha, shade=True)


for seed in seeds:
    c = sample_course(seed)
    g = np.array([gg["center"] for gg in c["gates"]])
    yaw = np.array([gg["yaw"] for gg in c["gates"]])
    ng = c["n_gates"]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ground plane
    xr = (g[:, 0].min() - 1.5, g[:, 0].max() + 1.5); yr = (g[:, 1].min() - 1.5, g[:, 1].max() + 1.5)
    XX, YY = np.meshgrid(np.linspace(*xr, 2), np.linspace(*yr, 2))
    ax.plot_surface(XX, YY, np.zeros_like(XX), color="0.9", alpha=0.3, linewidth=0)

    # closed-loop route (flown `laps` times)
    loop = np.vstack([g, g[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], "-", color="royalblue", lw=2.2, alpha=0.85)
    # HEIGHT CUE: a dot at each gate centre, a DOTTED line straight down to the floor, and a larger dot
    # on the ground directly below -> height is readable at a glance.
    for i in range(ng):
        ax.plot([g[i, 0], g[i, 0]], [g[i, 1], g[i, 1]], [0, g[i, 2]], color="0.45", lw=0.8, ls=":")
        ax.scatter([g[i, 0]], [g[i, 1]], [g[i, 2]], color="k", s=14, zorder=6)                 # centre point
        ax.scatter([g[i, 0]], [g[i, 1]], [0], color="0.25", s=45, alpha=0.8, zorder=6)         # ground dot

    # gates: vertical ring (aperture) oriented so you fly through along yaw; colored by height
    t = np.linspace(0, 2 * math.pi, 30)
    for i in range(ng):
        th = yaw[i]; rr = APERTURE / 2
        side = np.array([-math.sin(th), math.cos(th), 0.0]); up = np.array([0, 0, 1.0])
        ring = g[i] + rr * (np.outer(np.cos(t), side) + np.outer(np.sin(t), up))
        col = plt.cm.viridis(g[i, 2] / 2)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=col, lw=2.5)
        # travel-direction arrow (gate orientation)
        ax.quiver(g[i, 0], g[i, 1], g[i, 2], 0.6 * math.cos(th), 0.6 * math.sin(th), 0,
                  color="crimson", lw=1.5)
        ax.text(g[i, 0], g[i, 1], g[i, 2] + 0.15, str(i), fontsize=8)

    # obstacles
    for o in c["obstacles"]:
        cc = o["center"]
        if o["kind"] == "cylinder":
            cylinder(ax, cc[0], cc[1], o["radius"], cc[2] - o["half_h"], cc[2] + o["half_h"], "saddlebrown")
        else:
            h = o["half"]
            col = "teal" if o.get("fly_over") else "firebrick"
            box(ax, cc[0], cc[1], cc[2], h[0], h[1], h[2], col)
            ax.text(cc[0], cc[1], cc[2] + h[2] + 0.1,
                    "fly-over" if o.get("fly_over") else ("blocks" if o.get("blocks") else ""),
                    fontsize=7, ha="center")

    ax.scatter([c["start"][0]], [c["start"][1]], [c["start"][2]], c="k", marker="*", s=220)
    ax.text(c["start"][0], c["start"][1], c["start"][2] + 0.2, "START (ground)", fontsize=9, weight="bold")

    ax.set_title(f"seed {seed}: {ng} gates, {c['laps']} laps, spacing "
                 f"{min(c['spacing']):.1f}-{max(c['spacing']):.1f} m\n"
                 "rings=gates (color=height, red arrow=fly-through dir), brown=pillar, "
                 "red box=blocking, teal box=fly-over", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_zlim(0, 2.5)
    try:
        ax.set_box_aspect((xr[1] - xr[0], yr[1] - yr[0], 2.5))
    except Exception:
        pass
    ax.view_init(elev=24, azim=-58)                 # offset perspective (not top-down)
    out = f"/workspace/drone-fly/gpu_prototype/course_seed{seed}.png"
    fig.tight_layout(); fig.savefig(out, dpi=95); plt.close(fig)
    print(f"wrote {out}", flush=True)
