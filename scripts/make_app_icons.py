"""Render the desktop-app window icons (fly.png / fly.ico) from the fly silhouette (UC-61 item 5).

Dev-time helper only — NOT a runtime dependency. The app itself never imports Pillow; the native
window (``python -m app``) loads the pre-rendered ``app/static/fly.png`` / ``fly.ico`` best-effort.
Regenerate the binaries after editing ``app/static/fly.svg`` so the raster icons stay in sync:

    python scripts/make_app_icons.py

Kept deliberately simple (Pillow primitives, no SVG rasteriser) — it mirrors the SVG's fly shape
at a window-icon size, which is all the OS chrome shows.
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_STATIC = os.path.join(os.path.dirname(_HERE), "app", "static")
_ACCENT = (77, 163, 255, 255)  # matches fly.svg / app.css --accent
_TRANSPARENT = (0, 0, 0, 0)


def _draw_fly(size: int) -> Image.Image:
    """Draw the fly silhouette (wings + body segments + antennae) on a transparent square."""
    # Supersample for smooth edges, then downscale.
    scale = 8
    s = size * scale
    img = Image.new("RGBA", (s, s), _TRANSPARENT)
    d = ImageDraw.Draw(img)

    def rel(box):  # 0..32 design coords → pixels
        return [c / 32.0 * s for c in box]

    lw = max(1, int(0.05 * s))
    # Wings (outlined ellipses, rotated) — approximate with plain ellipses (icon-size fidelity).
    d.ellipse(rel([1, 7.5, 14, 14.5]), outline=_ACCENT, width=lw)
    d.ellipse(rel([18, 7.5, 31, 14.5]), outline=_ACCENT, width=lw)
    # Antennae.
    d.line(rel([14, 7, 12, 3.5]), fill=_ACCENT, width=lw)
    d.line(rel([18, 7, 20, 3.5]), fill=_ACCENT, width=lw)
    # Body: head, thorax, abdomen (filled).
    d.ellipse(rel([13.4, 5.9, 18.6, 11.1]), fill=_ACCENT)
    d.ellipse(rel([12.9, 11.6, 19.1, 18.4]), fill=_ACCENT)
    d.ellipse(rel([13.4, 17.3, 18.6, 27.7]), fill=_ACCENT)

    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    os.makedirs(_STATIC, exist_ok=True)
    png = _draw_fly(256)
    png.save(os.path.join(_STATIC, "fly.png"))
    # ICO with the usual window/taskbar sizes.
    png.save(
        os.path.join(_STATIC, "fly.ico"),
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {os.path.join(_STATIC, 'fly.png')} and fly.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
