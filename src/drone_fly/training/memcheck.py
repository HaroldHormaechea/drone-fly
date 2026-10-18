"""Host-memory preflight for long training runs (UC70): log, and warn — never act.

:func:`memory_status` reads the physical memory, the system commit and this process's private
bytes (Windows: ``GlobalMemoryStatusEx`` / ``GetProcessMemoryInfo`` through ``ctypes``; Linux:
``/proc/meminfo`` + ``/proc/self/status``; ``None`` fields elsewhere). :func:`preflight` logs them
with an estimate of what the run is about to allocate and WARNS when this process's private bytes
plus that estimate exceed the physical memory, or the estimate exceeds the available physical
memory (the machine would page). The system commit is logged but not used: on Windows it routinely
exceeds the physical memory (pagefile-backed reservations), so a rule on it would always fire. No
dependency, no killing.
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

    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    # The default c_int restype truncates the -1 pseudo-handle -> ERROR_INVALID_HANDLE (reads 0).
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError()
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
    tot, avail, priv = st["physical_total"], st["physical_available"], st["process_private"]

    def g(v: Any) -> str:
        return "?" if v is None else f"{v / GB:.1f} GB"

    log.info(
        "memory preflight (%s): physical %s, available %s, system commit %s, this process "
        "private %s (working set %s); about to allocate %s",
        what, g(tot), g(avail), g(st["commit_total"]), g(priv),
        g(st["process_working_set"]), g(about_to_allocate),
    )  # fmt: skip
    over_process = bool(tot and priv is not None and priv + about_to_allocate > tot)
    over_available = bool(avail is not None and about_to_allocate > avail)
    st["over_process"], st["over_available"] = over_process, over_available
    st["overcommit"] = over_process or over_available
    if st["overcommit"]:
        log.warning(
            "memory preflight: %s for %s; this process private %s + %s = %s vs physical %s, "
            "and %s vs available %s: the machine will page (slower training). Consider a smaller "
            "distill.buffer, distill.pin_memory: false, or closing other programs. Not stopping "
            "anything.",
            g(about_to_allocate), what, g(priv), g(about_to_allocate),
            g(None if priv is None else priv + about_to_allocate), g(tot),
            g(about_to_allocate), g(avail),
        )  # fmt: skip
    return st
