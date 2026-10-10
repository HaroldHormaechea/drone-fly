"""Generate a Claude-docs diagram-widget module (one-line JSX returning an isometric SVG) for a course.

Everything is INLINED as literals (no JS loops, no computed indices) so it satisfies the widget
validator: every typed <text> gets its own literal data-claude-text-id, colors are cds tokens, JSX
attribute names are camelCase. Prints the single-line `code` string to paste into a batch `create`.
"""
import sys, math, html
from course_gen import sample_course

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
c = sample_course(seed)
G = c["gates"]

# isometric projection: higher z -> higher on screen; (x+y) gives depth
K, K2, KZ = 15.0, 7.5, 22.0
def iso(x, y, z):
    return ((x - y) * K, -(z * KZ) + (x + y) * K2)   # (sx, sy) pre-offset; +sy = lower on screen

pts = []  # (gate screen, ground screen)
for g in G:
    x, y, z = g["center"]
    pts.append((iso(x, y, z), iso(x, y, 0.0)))
sx0 = iso(*c["start"][:2], c["start"][2]); sg0 = iso(*c["start"][:2], 0.0)
obs_pts = [(iso(o["center"][0], o["center"][1], o["center"][2]), o) for o in c["obstacles"]]

# fit to viewBox 0..760 wide with margins
allx = [p[0][0] for p in pts] + [p[1][0] for p in pts] + [sx0[0], sg0[0]] + [op[0][0] for op in obs_pts]
ally = [p[0][1] for p in pts] + [p[1][1] for p in pts] + [sx0[1], sg0[1]] + [op[0][1] for op in obs_pts]
minx, maxx, miny, maxy = min(allx), max(allx), min(ally), max(ally)
W = 760; margin = 60
scale = (W - 2 * margin) / (maxx - minx)
Hv = int((maxy - miny) * scale + 2 * margin + 40)
def TX(sx): return margin + (sx - minx) * scale
def TY(sy): return margin + 40 + (sy - miny) * scale

AX = "var(--cds-chart-axis)"; GRID = "var(--cds-chart-grid)"; INK = "var(--cds-text-primary)"
QUIET = "var(--cds-text-secondary)"; ACC = "var(--cds-chart-categorical-1)"
GOOD = "var(--cds-chart-status-good)"; CRIT = "var(--cds-chart-status-critical)"
WARN = "var(--cds-chart-categorical-3)"

el = []
el.append(f"<text data-claude-text-id='title' x='24' y='28' fontSize='15' fontWeight='600' fill={{'{INK}'}}>"
          f"seed {seed}: {c['n_gates']} gates, {c['laps']} laps (isometric; dotted line = height to floor)</text>")
# route polyline (closed loop)
route = " ".join(f"{TX(p[0][0]):.0f},{TY(p[0][1]):.0f}" for p in pts)
route += f" {TX(pts[0][0][0]):.0f},{TY(pts[0][0][1]):.0f}"
el.append(f"<polyline points='{route}' fill='none' stroke={{'{ACC}'}} strokeWidth='1.5' opacity='0.8'/>")
# per gate: dotted drop-line, ground dot, gate dot, z label
for i, (gp, grp) in enumerate(pts):
    gx, gy = TX(gp[0]), TY(gp[1]); fx, fy = TX(grp[0]), TY(grp[1])
    el.append(f"<line x1='{gx:.0f}' y1='{gy:.0f}' x2='{fx:.0f}' y2='{fy:.0f}' stroke={{'{GRID}'}} strokeWidth='1' strokeDasharray='3 3'/>")
    el.append(f"<circle cx='{fx:.0f}' cy='{fy:.0f}' r='4' fill={{'{AX}'}} opacity='0.5'/>")
    el.append(f"<circle cx='{gx:.0f}' cy='{gy:.0f}' r='5' fill={{'{ACC}'}}/>")
    el.append(f"<text data-claude-text-id='gate-{i}' x='{gx+7:.0f}' y='{gy-6:.0f}' fontSize='10.5' fill={{'{QUIET}'}}>"
              f"{i} (z{G[i]['center'][2]:.1f})</text>")
# obstacles
for j, (op, o) in enumerate(obs_pts):
    ox, oy = TX(op[0]), TY(op[1])
    if o["kind"] == "cylinder":
        el.append(f"<circle cx='{ox:.0f}' cy='{oy:.0f}' r='7' fill={{'{WARN}'}} fillOpacity='0.5' stroke={{'{WARN}'}}/>")
        lbl = "pillar"
    else:
        col = GOOD if o.get("fly_over") else CRIT
        el.append(f"<rect x='{ox-8:.0f}' y='{oy-8:.0f}' width='16' height='16' rx='2' fill={{'{col}'}} fillOpacity='0.4' stroke={{'{col}'}}/>")
        lbl = "fly-over" if o.get("fly_over") else "blocks"
    el.append(f"<text data-claude-text-id='obs-{j}' x='{ox:.0f}' y='{oy+20:.0f}' fontSize='10' textAnchor='middle' fill={{'{QUIET}'}}>{lbl}</text>")
# start marker
el.append(f"<circle cx='{TX(sg0[0]):.0f}' cy='{TY(sg0[1]):.0f}' r='6' fill={{'{INK}'}}/>")
el.append(f"<text data-claude-text-id='start' x='{TX(sg0[0])+9:.0f}' y='{TY(sg0[1])+4:.0f}' fontSize='11' fontWeight='600' fill={{'{INK}'}}>START (ground)</text>")

svg = (f"<svg viewBox='0 0 760 {Hv}' role='img' aria-label='isometric view of procedural course {seed}' fontSize='12'>"
       + "".join(el) + "</svg>")
module = "export default () => " + svg + ";"
print(module)
