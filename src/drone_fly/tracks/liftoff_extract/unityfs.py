"""UnityFS node-table reading: just enough to tell scene bundles apart without UnityPy.

Ported from liftoff-flight-analyzer ``src/fpv_review/engines/lugus/unityfs.py`` (``lz4_block``,
``_cstr``) and ``src/fpv_review/sources/liftoff/scene.py`` (``_node_names``) at commit d7250d5.
MIT License, Copyright (c) 2026 Harold Hormaechea.

A streamed scene bundle carries a ``.sharedAssets`` node; an asset bundle does not. Reading the
node table needs the blocks-info block only, so 274 bundles are sifted in about a second.
"""

from __future__ import annotations

import struct
from pathlib import Path


def lz4_block(src: bytes, out_size: int) -> bytes:
    """Decompress one LZ4 block (LZ4 and LZ4HC decompress identically)."""
    dst = bytearray(out_size)
    src_at = dst_at = 0
    end = len(src)
    while src_at < end:
        token = src[src_at]
        src_at += 1
        literals = token >> 4
        if literals == 15:
            while True:
                more = src[src_at]
                src_at += 1
                literals += more
                if more != 255:
                    break
        dst[dst_at : dst_at + literals] = src[src_at : src_at + literals]
        src_at += literals
        dst_at += literals
        if src_at >= end:
            break  # the last sequence is literals only
        offset = src[src_at] | (src[src_at + 1] << 8)
        src_at += 2
        match = token & 15
        if match == 15:
            while True:
                more = src[src_at]
                src_at += 1
                match += more
                if more != 255:
                    break
        match += 4
        from_at = dst_at - offset
        if offset >= match:
            dst[dst_at : dst_at + match] = dst[from_at : from_at + match]
            dst_at += match
        else:
            for _ in range(match):  # overlapping run; byte by byte
                dst[dst_at] = dst[from_at]
                dst_at += 1
                from_at += 1
    return bytes(dst)


def cstr(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\x00", pos)
    return buf[pos:end].decode("utf-8", "replace"), end + 1


def node_names(path: str | Path, cap: int = 2_000_000) -> list[str]:
    """Node names of a UnityFS bundle from its blocks-info block (``[]`` if not UnityFS)."""
    with open(path, "rb") as fh:
        raw = fh.read(cap)
    sig, at = cstr(raw, 0)
    if sig != "UnityFS":
        return []
    (version,) = struct.unpack_from(">I", raw, at)
    at += 4
    _, at = cstr(raw, at)
    _, at = cstr(raw, at)
    _size, packed, unpacked, flags = struct.unpack_from(">qIII", raw, at)
    at += 20
    if flags & 0x80:
        return []  # info at the end; not seen in practice
    if version >= 7:
        at = (at + 15) // 16 * 16
    info = raw[at : at + packed]
    if flags & 0x3F in (2, 3):
        info = lz4_block(info, unpacked)
    walk = 16
    (count,) = struct.unpack_from(">I", info, walk)
    walk += 4 + 10 * count
    (count,) = struct.unpack_from(">I", info, walk)
    walk += 4
    names = []
    for _ in range(count):
        walk += 20
        name, walk = cstr(info, walk)
        names.append(name)
    return names


def is_scene_bundle(path: str | Path) -> bool:
    try:
        return any(n.endswith(".sharedAssets") for n in node_names(path))
    except Exception:  # an unreadable bundle is simply not a candidate
        return False
