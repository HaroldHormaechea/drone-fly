"""``python -m app`` — launch the drone-fly desktop app in a native window (UC-61 AC1).

Starts the FastAPI/Uvicorn server on a loopback ephemeral port in a background thread, then opens a
``pywebview`` native window pointed at it. This is the ONLY display-dependent path in the app and is
owner-eyeball only — it is never run in CI (``pywebview`` is deliberately outside the ``dev`` extra;
install the ``app`` extra to run this). The hermetic backend logic lives in :mod:`app.server` and
friends and is exercised head-less via ``TestClient``.

The window close tears the process down; ``--no-window`` runs the server alone (browse to the
printed URL), which is handy for debugging without a webview.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time


def _pick_free_port(host: str) -> int:
    """Bind an ephemeral port on ``host`` and return it (closed at once; small TOCTOU window)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _serve(app, host: str, port: int):
    """Run uvicorn for ``app`` on ``host:port`` (blocking); returns the server for shutdown."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drone-fly-app", description="drone-fly desktop app")
    parser.add_argument("--project-root", default=None, help="Project root (default: repo root).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=0, help="Bind port (default: ephemeral).")
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run the server only (no native window); browse to the printed URL. Useful for "
        "debugging on a machine without a webview backend.",
    )
    args = parser.parse_args(argv)

    from app.server import create_app

    host = args.host
    port = args.port or _pick_free_port(host)
    app = create_app(args.project_root)
    url = f"http://{host}:{port}/"

    server_thread = threading.Thread(target=_serve, args=(app, host, port), daemon=True)
    server_thread.start()

    # Give uvicorn a moment to bind before the window points at it.
    time.sleep(0.5)
    print(f"drone-fly desktop app serving at {url}", file=sys.stderr)

    if args.no_window:
        try:
            server_thread.join()
        except KeyboardInterrupt:
            return 0
        return 0

    try:
        import webview  # pywebview; only needed for the windowed path (owner machine).
    except ImportError:
        print(
            "pywebview is not installed. Install the app extra (uv sync --extra app / "
            "pip install -e '.[app]'), or run with --no-window and open the URL in a browser.",
            file=sys.stderr,
        )
        return 2

    webview.create_window("drone-fly", url, width=1280, height=860)
    # UC-61 item 5: set the native window/taskbar icon best-effort. `icon=` is honoured on the
    # GTK/Qt backends and ignored elsewhere; a pywebview too old to accept the kwarg falls back to
    # the default icon. Never fatal — this path is owner-eyeball only and never runs in CI.
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "fly.png")
    try:
        if os.path.isfile(icon_path):
            webview.start(icon=icon_path)
        else:
            webview.start()
    except TypeError:
        webview.start()  # older pywebview without the `icon` kwarg
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
