"""Host-memory preflight for long training runs (UC70): log, and warn — never act.

:func:`memory_status` reads the physical memory, the system commit and this process's private
bytes (Windows: ``GlobalMemoryStatusEx`` / ``GetProcessMemoryInfo`` through ``ctypes``; Linux:
``/proc/meminfo`` + ``/proc/self/status``; ``None`` fields elsewhere). :func:`preflight` logs them
with an estimate of what the run is about to allocate and WARNS when that estimate plus the current
system commit exceeds the physical memory (the machine would page). No dependency, no killing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

log = logging.getLogger(__name__)
GB = float(1 << 30)


def _windows() -> dict[str, Any]:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return {
        "physical_total": int(ms.ullTotalPhys),
        "physical_available": int(ms.ullAvailPhys),
        "commit_total": int(ms.ullTotalPageFile - ms.ullAvailPageFile),
        "commit_limit": int(ms.ullTotalPageFile),
        "process_private": int(pmc.PrivateUsage),
        "process_working_set": int(pmc.WorkingSetSize),
    }


def _linux() -> dict[str, Any]:
    info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) * 1024
    priv = rss = None
    with open("/proc/self/status", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            if line.startswith("RssAnon:"):
                priv = int(line.split()[1]) * 1024
    return {
        "physical_total": info.get("MemTotal"),
        "physical_available": info.get("MemAvailable"),
        "commit_total": info.get("Committed_AS"),
        "commit_limit": info.get("CommitLimit"),
        "process_private": priv,
        "process_working_set": rss,
    }


def memory_status() -> dict[str, Any]:
    try:
        if sys.platform == "win32":
            return _windows()
        if sys.platform.startswith("linux"):
            return _linux()
    except Exception:  # noqa: BLE001 - informational only
        log.debug("memory status unavailable", exc_info=True)
    keys = ("physical_total", "physical_available", "commit_total", "commit_limit",
            "process_private", "process_working_set")  # fmt: skip
    return dict.fromkeys(keys)


def preflight(about_to_allocate: int, what: str = "the run") -> dict[str, Any]:
    """Log the memory status and warn when ``about_to_allocate`` bytes would overcommit."""
    st = memory_status()
    st["about_to_allocate"] = int(about_to_allocate)
    tot, commit = st["physical_total"], st["commit_total"]

    def g(v: Any) -> str:
        return "?" if v is None else f"{v / GB:.1f} GB"

    log.info(
        "memory preflight (%s): physical %s, available %s, system commit %s, this process "
        "private %s (working set %s); about to allocate %s",
        what, g(tot), g(st["physical_available"]), g(commit), g(st["process_private"]),
        g(st["process_working_set"]), g(about_to_allocate),
    )  # fmt: skip
    st["overcommit"] = bool(tot and commit is not None and commit + about_to_allocate > tot)
    if st["overcommit"]:
        log.warning(
            "memory preflight: system commit %s + %s for %s exceeds physical memory %s: the "
            "machine will page (slower training). Consider a smaller distill.buffer, "
            "distill.pin_memory: false, or closing other programs. Not stopping anything.",
            g(commit), g(about_to_allocate), what, g(tot),
        )  # fmt: skip
    return st
