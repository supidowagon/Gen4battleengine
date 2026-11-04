"""Test-only VCDIFF encoder (RFC 3284 plus xdelta3's secondary LZMA and Adler-32).

It emits every op with the default code table's single-instruction code whose size
field is 0 (the size follows as a varint), so every address mode is reachable:
``RUN`` is code 0, ``ADD`` code 1, ``COPY`` in mode ``m`` code ``19 + 16 * m``.
With ``secondary=True`` each section is one finished xz stream (Python's ``lzma``
has no sync flush), so secondary tests use one window.
"""

from __future__ import annotations

import lzma
import zlib
from dataclasses import dataclass

NEAR_SIZE, SAME_SIZE = 4, 3


@dataclass(frozen=True)
class Window:
    """One target window.

    ``source_segment`` is ``(position, length)`` in the source file, or in the
    target written so far when ``target_from_target``; ``None`` means no segment.
    ``ops`` are ``("add", bytes)``, ``("run", byte, n)`` or ``("copy", addr, n, mode)``
    with ``addr`` the decoded address (in the segment followed by this window's target).
    """

    source_segment: tuple[int, int] | None
    target_from_target: bool
    ops: list[tuple]


def _varint(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


class _Cache:
    """RFC 3284 §5.1 address cache, mirrored by the encoder."""

    def __init__(self) -> None:
        self.near = [0] * NEAR_SIZE
        self.next_slot = 0
        self.same = [0] * (SAME_SIZE * 256)

    def encode(self, addr: int, here: int, mode: int) -> bytes:
        if mode == 0:
            enc = _varint(addr)
        elif mode == 1:
            if addr > here:
                raise ValueError(f"HERE mode cannot reach {addr} from {here}")
            enc = _varint(here - addr)
        elif 2 <= mode < 2 + NEAR_SIZE:
            base = self.near[mode - 2]
            if addr < base:
                raise ValueError(f"near mode {mode} ({base}) cannot reach {addr}")
            enc = _varint(addr - base)
        elif mode < 2 + NEAR_SIZE + SAME_SIZE:
            idx = addr % (SAME_SIZE * 256)
            if self.same[idx] != addr or idx // 256 != mode - 2 - NEAR_SIZE:
                raise ValueError(f"same mode {mode} cannot reach {addr}")
            enc = bytes([idx % 256])
        else:
            raise ValueError(f"no address mode {mode}")
        self.near[self.next_slot] = addr
        self.next_slot = (self.next_slot + 1) % NEAR_SIZE
        self.same[addr % (SAME_SIZE * 256)] = addr
        return enc


def _window(w: Window, segment: bytes, secondary: bool, adler: bool | int) -> tuple[bytes, bytes]:
    """Encode one window; returns ``(window bytes, decoded target)``."""
    data, inst, addrs = bytearray(), bytearray(), bytearray()
    cache = _Cache()
    target = bytearray()
    for op in w.ops:
        if op[0] == "add":
            inst += bytes([1]) + _varint(len(op[1]))
            data += op[1]
            target += op[1]
        elif op[0] == "run":
            _, byte, n = op
            inst += bytes([0]) + _varint(n)
            data.append(byte)
            target += bytes([byte]) * n
        elif op[0] == "copy":
            _, addr, n, mode = op
            addrs += cache.encode(addr, len(segment) + len(target), mode)
            inst += bytes([19 + 16 * mode]) + _varint(n)
            for i in range(n):  # byte by byte: the reference semantics for overlaps
                pos = addr + i
                ref = segment + target
                target.append(ref[pos] if pos < len(ref) else 0)
        else:
            raise ValueError(f"unknown op {op[0]!r}")

    sections = [bytes(data), bytes(inst), bytes(addrs)]
    delta_indicator = 0
    if secondary:
        delta_indicator = 0x07
        sections = [_varint(len(s)) + lzma.compress(s, format=lzma.FORMAT_XZ) for s in sections]
    checksum = zlib.adler32(bytes(target)) if adler is True else adler
    body = _varint(len(target)) + bytes([delta_indicator])
    body += b"".join(_varint(len(s)) for s in sections)
    if adler is not False:
        body += checksum.to_bytes(4, "big")
    body += b"".join(sections)

    indicator = 0x04 if adler is not False else 0
    head = b""
    if w.source_segment is not None:
        indicator |= 0x02 if w.target_from_target else 0x01
        pos, length = w.source_segment
        head = _varint(length) + _varint(pos)
    return bytes([indicator]) + head + _varint(len(body)) + body, bytes(target)


def encode(
    windows: list[Window],
    *,
    app_header: bytes | None = None,
    secondary: bool = False,
    adler: bool | int = True,
    source: bytes = b"",
) -> bytes:
    """Encode ``windows`` as a VCDIFF patch against ``source``.

    ``adler`` is ``True`` for the real per-window Adler-32, ``False`` for none, or an
    ``int`` written as every window's checksum. ``source`` is only read to simulate
    the target (for the checksum and ``VCD_TARGET`` segments).
    """
    hdr = 0x01 if secondary else 0
    hdr |= 0x04 if app_header is not None else 0
    out = bytearray(b"\xd6\xc3\xc4\x00" + bytes([hdr]))
    if secondary:
        out.append(2)  # xdelta3's LZMA
    if app_header is not None:
        out += _varint(len(app_header)) + app_header
    written = bytearray()
    for w in windows:
        segment = b""
        if w.source_segment is not None:
            pos, length = w.source_segment
            segment = bytes((written if w.target_from_target else source)[pos : pos + length])
            if len(segment) != length:
                raise ValueError(f"segment {w.source_segment} outside the given bytes")
        chunk, target = _window(w, segment, secondary, adler)
        out += chunk
        written += target
    return bytes(out)
