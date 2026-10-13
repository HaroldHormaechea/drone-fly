"""The Liftoff install (found through Steam) and its build id.

Ported from liftoff-flight-analyzer ``src/fpv_review/engines/lugus/steam.py`` and
``src/fpv_review/sources/liftoff/tracks.py`` (``find_game``, ``bundle_dir``, ``build_id``) at
commit d7250d5. MIT License, Copyright (c) 2026 Harold Hormaechea.

Read-only: nothing here launches the game or writes into its folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

STEAM_APPID = "410340"
LIFTOFF_DIR_ENV = "LIFTOFF_DIR"
_PROBE = Path("Liftoff_Data") / "StreamingAssets" / "aa" / "StandaloneWindows64"


def _vdf_paths(vdf: Path) -> list[Path]:
    try:
        text = vdf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [Path(p.replace("\\\\", "\\")) for p in re.findall(r'"path"\s+"([^"]+)"', text)]


def steam_libraries() -> list[Path]:
    """Every Steam library folder named by the registry and ``libraryfolders.vdf``."""
    roots: list[Path] = []
    try:
        import winreg

        for hive, key, name in (
            (winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath"),
        ):
            try:
                with winreg.OpenKey(hive, key) as handle:
                    roots.append(Path(winreg.QueryValueEx(handle, name)[0]))
            except OSError:
                pass
    except ImportError:
        pass
    for guess in (
        r"C:\Program Files (x86)\Steam",
        r"C:\Program Files\Steam",
        os.path.expanduser("~/.steam/steam"),
        os.path.expanduser("~/.local/share/Steam"),
    ):
        roots.append(Path(guess))
    libs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for lib in [root, *_vdf_paths(root / "steamapps" / "libraryfolders.vdf")]:
            key = os.path.normcase(str(lib))
            if key in seen:
                continue
            seen.add(key)
            if (lib / "steamapps").is_dir():
                libs.append(lib)
    return libs


def find_game(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The Liftoff install root (``--game-dir`` -> ``$LIFTOFF_DIR`` -> Steam libraries)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(LIFTOFF_DIR_ENV):
        candidates.append(Path(os.environ[LIFTOFF_DIR_ENV]))
    for lib in steam_libraries():
        candidates.append(lib / "steamapps" / "common" / "Liftoff")
    for cand in candidates:
        if (cand / _PROBE).exists():
            return cand
    return None


def bundle_dir(game: Path) -> Path:
    return Path(game) / _PROBE


def build_id(game: Path | None) -> str | None:
    """Steam's build id of the install, from ``appmanifest_410340.acf`` beside it."""
    if game is None:
        return None
    acf = Path(game).parent.parent / f"appmanifest_{STEAM_APPID}.acf"
    try:
        found = re.search(r'"buildid"\s+"(\d+)"', acf.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return found.group(1) if found else None


def unity_version(default: str = "2022.3.62f3") -> str:
    """The engine version from Liftoff's ``Player.log`` (UnityPy needs it), else ``default``."""
    log = (
        Path(os.environ.get("LOCALAPPDATA", "")).parent
        / "LocalLow"
        / "LuGus Studios"
        / "Liftoff"
        / "Player.log"
    )
    try:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "Initialize engine version" in line:
                return line.split("Initialize engine version:")[1].strip().split()[0]
    except (OSError, IndexError):
        pass
    return default
