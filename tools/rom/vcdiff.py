"""Pure-Python VCDIFF decoder: RFC 3284 plus the xdelta3 extensions the Kaizo patch uses.

Supported: the default code table (§5.6) with its near (4) and same (3 x 256) address
caches (§5.3), ``VCD_SOURCE`` and ``VCD_TARGET`` segments, COPYs that overlap the
write position, and xdelta3's application header (skipped), secondary LZMA (one xz
stream per section kind, continued across windows) and per-window Adler-32 (checked).
Anything else, or any byte that does not add up, raises ``VcdiffError`` naming the
window and the patch offset.
"""

from __future__ import annotations

import lzma
import zlib

MAGIC = b"\xd6\xc3\xc4\x00"

# Hdr_Indicator bits (RFC 3284 §4.1).
VCD_DECOMPRESS, VCD_CODETABLE, VCD_APPHEADER = 0x01, 0x02, 0x04
# Win_Indicator bits (§4.2; VCD_ADLER32 is xdelta3's).
VCD_SOURCE, VCD_TARGET, VCD_ADLER32 = 0x01, 0x02, 0x04
# Delta_Indicator bits: which sections are secondary-compressed.
SECTION_BITS = (0x01, 0x02, 0x04)
SECTION_NAMES = ("data", "instructions", "addresses")
# xdelta3's secondary compressor id for LZMA.
LZMA_ID = 2

NOOP, ADD, RUN, COPY = 0, 1, 2, 3
NEAR_SIZE, SAME_SIZE = 4, 3


class VcdiffError(Exception):
    """The patch is malformed, unsupported, or does not apply to the source."""


class _Reader:
    """Cursor over ``buf``; errors name ``where`` and the absolute offset."""

    def __init__(self, buf: bytes, where: str, base: int = 0) -> None:
        self.buf, self.where, self.base, self.pos = buf, where, base, 0

    def fail(self, msg: str, pos: int | None = None) -> VcdiffError:
        at = self.base + (self.pos if pos is None else pos)
        return VcdiffError(f"{self.where} at offset {at}: {msg}")

    def at_end(self) -> bool:
        return self.pos >= len(self.buf)

    def byte(self) -> int:
        if self.pos >= len(self.buf):
            raise self.fail("truncated (expected a byte)")
        self.pos += 1
        return self.buf[self.pos - 1]

    def varint(self) -> int:
        """RFC 3284 §2 integer: base 128, most significant digit first."""
        start, value = self.pos, 0
        for _ in range(10):
            b = self.byte()
            value = (value << 7) | (b & 0x7F)
            if not b & 0x80:
                return value
        raise self.fail("integer longer than 10 bytes", start)

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise self.fail(f"truncated (need {n} bytes, {len(self.buf) - self.pos} left)")
        self.pos += n
        return self.buf[self.pos - n : self.pos]


def _default_code_table() -> list[tuple[int, int, int, int, int, int]]:
    """RFC 3284 §5.6: ``(inst1, size1, mode1, inst2, size2, mode2)`` per code."""
    table = [(RUN, 0, 0, NOOP, 0, 0)]
    table += [(ADD, size, 0, NOOP, 0, 0) for size in range(18)]
    for mode in range(9):
        table += [(COPY, size, mode, NOOP, 0, 0) for size in (0, *range(4, 19))]
    for mode in range(6):
        table += [(ADD, a, 0, COPY, c, mode) for a in range(1, 5) for c in range(4, 7)]
    for mode in range(6, 9):
        table += [(ADD, a, 0, COPY, 4, mode) for a in range(1, 5)]
    table += [(COPY, 4, mode, ADD, 1, 0) for mode in range(9)]
    assert len(table) == 256
    return table


CODE_TABLE = _default_code_table()


class _AddressCache:
    """RFC 3284 §5.1-5.3 near and same caches; reset for every window."""

    def __init__(self) -> None:
        self.near = [0] * NEAR_SIZE
        self.next_slot = 0
        self.same = [0] * (SAME_SIZE * 256)

    def decode(self, mode: int, here: int, addrs: _Reader) -> int:
        if mode == 0:  # VCD_SELF
            addr = addrs.varint()
        elif mode == 1:  # VCD_HERE
            addr = here - addrs.varint()
        elif mode < 2 + NEAR_SIZE:
            addr = self.near[mode - 2] + addrs.varint()
        else:
            addr = self.same[(mode - 2 - NEAR_SIZE) * 256 + addrs.byte()]
        self.near[self.next_slot] = addr
        self.next_slot = (self.next_slot + 1) % NEAR_SIZE
        self.same[addr % (SAME_SIZE * 256)] = addr
        return addr


class _Secondary:
    """xdelta3 secondary LZMA: one xz stream per section kind for the whole file.

    A compressed section is a varint of its decoded length followed by the next
    chunk of that kind's stream (Ruling 2). A stream that has ended is replaced by a
    new decompressor when another compressed chunk arrives.
    """

    def __init__(self) -> None:
        self.streams: list[lzma.LZMADecompressor | None] = [None, None, None]

    def decode(self, kind: int, section: _Reader) -> bytes:
        size = section.varint()
        stream = self.streams[kind]
        if stream is None or stream.eof:
            stream = self.streams[kind] = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
        chunk = section.buf[section.pos :]
        try:
            out = stream.decompress(chunk)
        except lzma.LZMAError as e:
            raise section.fail(f"LZMA {SECTION_NAMES[kind]} section: {e}") from e
        if stream.eof and stream.unused_data:
            raise section.fail(f"{len(stream.unused_data)} bytes after the xz stream end")
        if len(out) != size:
            raise section.fail(
                f"LZMA {SECTION_NAMES[kind]} section decoded to {len(out)} bytes, "
                f"header says {size}"
            )
        return out


def _parse_header(r: _Reader) -> _Secondary | None:
    """RFC 3284 §4.1 header; returns the secondary decoder when one is declared."""
    if r.take(4) != MAGIC:
        raise r.fail("bad magic (not a VCDIFF version 0 file)", 0)
    indicator = r.byte()
    if indicator & VCD_CODETABLE:
        raise r.fail("custom code table (VCD_CODETABLE) is not supported", 4)
    if indicator & ~(VCD_DECOMPRESS | VCD_APPHEADER):
        raise r.fail(f"unsupported header indicator bits 0x{indicator:02x}", 4)
    secondary = None
    if indicator & VCD_DECOMPRESS:
        compressor = r.byte()
        if compressor != LZMA_ID:
            raise r.fail(f"unsupported secondary compressor id {compressor}", r.pos - 1)
        secondary = _Secondary()
    if indicator & VCD_APPHEADER:
        r.take(r.varint())  # xdelta3 file names; not needed
    return secondary


def _copy(target: bytearray, segment: bytes, addr: int, n: int) -> None:
    """Append ``n`` bytes from ``addr`` in segment + target (RFC 3284 §3).

    A copy that overlaps the write position repeats the chunk already written:
    each pass appends everything between the read and write positions, doubling.
    """
    seglen = len(segment)
    if addr < seglen:
        k = min(n, seglen - addr)
        target += segment[addr : addr + k]
        addr, n = seglen, n - k
    t = addr - seglen
    while n:
        k = min(n, len(target) - t)
        target += target[t : t + k]
        t, n = t + k, n - k


def _run_instructions(
    segment: bytes, target_len: int, data: _Reader, inst: _Reader, addrs: _Reader
) -> bytearray:
    """Execute one window's instructions into its target (RFC 3284 §5)."""
    target = bytearray()
    cache = _AddressCache()
    while not inst.at_end():
        code_at = inst.pos
        inst1, size1, mode1, inst2, size2, mode2 = CODE_TABLE[inst.byte()]
        if inst1 != NOOP and size1 == 0:
            size1 = inst.varint()
        if inst2 != NOOP and size2 == 0:
            size2 = inst.varint()
        for op, size, mode in ((inst1, size1, mode1), (inst2, size2, mode2)):
            if op == NOOP:
                continue
            if len(target) + size > target_len:
                raise inst.fail(f"instruction overruns the {target_len}-byte target", code_at)
            if op == ADD:
                target += data.take(size)
            elif op == RUN:
                target += bytes([data.byte()]) * size
            else:
                here = len(segment) + len(target)
                addr_at = addrs.pos
                addr = cache.decode(mode, here, addrs)
                if not 0 <= addr < here:
                    raise addrs.fail(
                        f"COPY address {addr} (mode {mode}) outside the "
                        f"{len(segment)}-byte source segment plus {len(target)} bytes of target",
                        addr_at,
                    )
                _copy(target, segment, addr, size)
    for section in (data, addrs):
        if not section.at_end():
            raise section.fail(f"{len(section.buf) - section.pos} unused bytes")
    if len(target) != target_len:
        raise inst.fail(f"target is {len(target)} bytes, header says {target_len}")
    return target


def _segment(r: _Reader, indicator: int, source: bytes, output: bytearray) -> bytes:
    """The window's source segment: from the source file or the target so far."""
    if not indicator & (VCD_SOURCE | VCD_TARGET):
        return b""
    at = r.pos
    length, position = r.varint(), r.varint()
    origin, name = (source, "source") if indicator & VCD_SOURCE else (output, "target")
    if position + length > len(origin):
        raise r.fail(
            f"source segment {position}+{length} outside the {len(origin)}-byte {name}", at
        )
    return bytes(origin[position : position + length])


def _decode_window(
    r: _Reader, source: bytes, output: bytearray, secondary: _Secondary | None
) -> bytearray:
    """One target window (RFC 3284 §4.2-4.3), checked against its Adler-32."""
    start = r.pos
    indicator = r.byte()
    if indicator & ~(VCD_SOURCE | VCD_TARGET | VCD_ADLER32) or (
        indicator & VCD_SOURCE and indicator & VCD_TARGET
    ):
        raise r.fail(f"unsupported window indicator 0x{indicator:02x}", start)
    segment = _segment(r, indicator, source, output)
    delta_len = r.varint()
    delta_start = r.pos
    target_len = r.varint()
    delta_indicator = r.byte()
    if delta_indicator & ~0x07 or (delta_indicator and secondary is None):
        raise r.fail(f"unsupported delta indicator 0x{delta_indicator:02x}", r.pos - 1)
    lengths = [r.varint() for _ in SECTION_NAMES]
    checksum = int.from_bytes(r.take(4), "big") if indicator & VCD_ADLER32 else None
    if r.pos - delta_start + sum(lengths) != delta_len:
        raise r.fail(f"delta encoding length {delta_len} does not match its sections", start)

    sections = []
    for kind, length in enumerate(lengths):
        at = r.base + r.pos
        raw = _Reader(r.take(length), r.where, at)
        if delta_indicator & SECTION_BITS[kind]:
            assert secondary is not None
            where = f"{r.where} {SECTION_NAMES[kind]} (LZMA-decoded from offset {at})"
            raw = _Reader(secondary.decode(kind, raw), where)
        sections.append(raw)
    target = _run_instructions(segment, target_len, *sections)
    if checksum is not None and zlib.adler32(target) != checksum:
        raise r.fail(
            f"Adler-32 mismatch: patch says {checksum:08x}, target is {zlib.adler32(target):08x}",
            start,
        )
    return target


def apply_vcdiff(source: bytes, patch: bytes) -> bytes:
    """Apply the VCDIFF ``patch`` to ``source`` and return the target file."""
    r = _Reader(patch, "header")
    secondary = _parse_header(r)
    output = bytearray()
    index = 0
    while not r.at_end():
        window = _Reader(patch, f"window {index}")
        window.pos = r.pos
        output += _decode_window(window, source, output, secondary)
        r.pos = window.pos
        index += 1
    return bytes(output)
