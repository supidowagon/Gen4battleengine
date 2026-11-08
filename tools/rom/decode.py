"""Local decoding of what the Kaizo ROM changes: code as Thumb/ARM disassembly, NARC members
as hex words. Everything goes to stdout; no ROM byte is ever written to a file.

``python -m tools.rom.decode code <module> <symbol>`` shows each changed range of a symbol
(``<symbol>`` is ``0xSTART-0xEND`` for a range outside every symbol), vanilla and patched, with
four instructions of context either side. ``member <narc> <index>`` shows a NARC member's
32-bit words on both sides (battle scripts have no reader here). ``at <hex> <n>`` disassembles
``n`` instructions of the patched ROM at an address (injected code included).

The instruction set follows the linker map's ``$a``/``$t`` labels (Thumb when there is none).
Disassembly is linear from the start of the containing symbol, so literal pools in the middle
of a function can show as bogus instructions; the range markers (``>``) still point at the
changed bytes.
"""

from __future__ import annotations

import argparse
import bisect
import functools
import struct
import sys
from dataclasses import dataclass

from tools.rom.diff import KAIZO_INJECTED_CODE, CodeChange, RomDiff, diff_roms

CONTEXT = 4


@dataclass(frozen=True)
class RangeDecode:
    start: int  # vanilla RAM addresses, as in CodeChange
    end: int
    vanilla: list[str]
    patched: list[str]


@dataclass(frozen=True)
class _Context:
    vanilla: object
    patched: object
    symbols: object
    diff: RomDiff


@functools.cache
def _context() -> _Context:
    from tools.rom.inputs import VANILLA_SHA1, load_inputs, read_checked
    from tools.rom.nds import RomFiles
    from tools.rom.patch import kaizo_rom
    from tools.rom.symbols import DEFAULT_MAP, SymbolMap

    inputs = load_inputs()
    vanilla = RomFiles.open(read_checked(inputs.vanilla_rom, VANILLA_SHA1, "sha1"))
    patched = RomFiles.open(kaizo_rom(inputs)[0].read_bytes())
    symbols = SymbolMap.from_map(DEFAULT_MAP)
    diff = diff_roms(vanilla, patched, symbols, KAIZO_INJECTED_CODE)
    return _Context(vanilla, patched, symbols, diff)


def _regions(files, module: str) -> list[tuple[int, bytes]]:
    if module == "arm9":
        return list(files.arm9_sections())
    i = int(module.removeprefix("overlay"))
    return [(files.overlay_info(i)[0], files.overlay(i))]


def _region_at(files, module: str, addr: int) -> tuple[int, bytes]:
    for base, data in _regions(files, module):
        if base <= addr < base + len(data):
            return base, data
    raise ValueError(f"{module}: {addr:#010x} is outside the module image")


def _is_arm(symbols, module: str, addr: int) -> bool:
    """True when the last ``$a``/``$t``/``$d`` label at or before ``addr`` is ``$a``."""
    labels = [s for s in symbols.symbols(module) if s.size == 0 and s.name in ("$a", "$t", "$d")]
    i = bisect.bisect_right([s.address for s in labels], addr) - 1
    return i >= 0 and labels[i].name == "$a"


def _disassemble(
    data: bytes, base: int, start: int, stop: int, arm: bool
) -> list[tuple[int, int, str]]:
    """(address, size, text) of the instructions from ``start`` up to ``stop``."""
    from capstone import CS_ARCH_ARM, CS_MODE_ARM, CS_MODE_THUMB, Cs

    md = Cs(CS_ARCH_ARM, CS_MODE_ARM if arm else CS_MODE_THUMB)
    step = 4 if arm else 2
    out: list[tuple[int, int, str]] = []
    pos = start
    while pos < stop:
        got = next(md.disasm(data[pos - base : pos - base + 4], pos, 1), None)
        if got is None:
            size = min(step, stop - pos)
            word = data[pos - base : pos - base + size]
            out.append((pos, size, f".word {word.hex()}"))
            pos += size
        else:
            out.append((pos, got.size, f"{got.mnemonic} {got.op_str}".strip()))
            pos += got.size
    return out


def _window(data, base, sym_start, lo, hi, arm, context) -> list[str]:
    """Instructions covering ``[lo, hi)`` plus ``context`` either side; ``>`` marks the
    ones overlapping the range. Disassembles linearly from ``sym_start``."""
    stop = min(hi + 4 * context + 4, base + len(data))
    insns = _disassemble(data, base, sym_start, stop, arm)
    hit = [i for i, (a, n, _) in enumerate(insns) if a < max(hi, lo + 1) and a + n > lo]
    if not hit:  # an insertion at the very end of what was disassembled
        hit = [len(insns) - 1]
    first, last = max(0, hit[0] - context), hit[-1] + context + 1
    marked = set(hit) if hi > lo else {hit[0]}
    return [
        f"{'>' if i in marked else ' '} {a:#010x}  {text}"
        for i, (a, _, text) in enumerate(insns)
        if first <= i < last
    ]


def _changes(ctx: _Context, module: str, symbol: str) -> list[CodeChange]:
    def label(c: CodeChange) -> str:
        return c.symbol or f"{c.start:#010x}-{c.end:#010x}"

    found = [
        c for c in ctx.diff.code if c.module == module and label(c) in (symbol, symbol.lower())
    ]
    if not found:
        raise KeyError(f"no code change in {module} for {symbol!r}")
    return found


def _symbol_start(symbols, module: str, c: CodeChange) -> int:
    best = None
    for s in symbols.symbols(module):
        inside = s.size and s.address <= c.start < s.address + s.size
        if inside and (best is None or s.size < best.size):
            best = s
    return best.address if best else c.start & ~3


def decode_code(module: str, symbol: str) -> list[RangeDecode]:
    """One ``RangeDecode`` per changed range of ``symbol`` in ``module``."""
    ctx = _context()
    out = []
    for c in _changes(ctx, module, symbol):
        back = c.start - _symbol_start(ctx.symbols, module, c)
        arm = _is_arm(ctx.symbols, module, c.start)
        sides = []
        for files, start, end in (
            (ctx.vanilla, c.start, c.end),
            (ctx.patched, c.patched_start, c.patched_end),
        ):
            base, data = _region_at(files, module, start)
            sides.append(_window(data, base, start - back, start, end, arm, CONTEXT))
        out.append(RangeDecode(c.start, c.end, sides[0], sides[1]))
    return out


def _words(data: bytes) -> list[str]:
    whole = len(data) // 4 * 4
    lines = [
        f"{4 * n:#06x}  {w:#010x}" for n, (w,) in enumerate(struct.iter_unpack("<I", data[:whole]))
    ]
    if len(data) > whole:
        lines.append(f"{whole:#06x}  tail {data[whole:].hex()}")
    return lines


def decode_member(narc: str, index: int) -> tuple[list[str], list[str]]:
    """The member's 32-bit words, vanilla then patched (empty when a side lacks it)."""
    ctx = _context()
    sides = []
    for files in (ctx.vanilla, ctx.patched):
        members = files.narc(narc)
        sides.append(_words(members[index]) if 0 <= index < len(members) else [])
    return sides[0], sides[1]


def disassemble_at(
    addr: int, count: int, module: str | None = None, arm: bool | None = None
) -> list[str]:
    """``count`` instructions of the patched ROM at RAM ``addr``: injected code first, then
    ARM9, then the overlay holding it (``module`` picks one when several overlays do)."""
    ctx = _context()
    found: list[tuple[str, int, bytes]] = []
    if not module:
        for path, member, load in KAIZO_INJECTED_CODE:
            data = ctx.patched.narc(path)[member]
            if load <= addr < load + len(data):
                found.append((f"{path}[{member}]", load, data))
    names = [module] if module else ["arm9"] + [f"overlay{i}" for i in ctx.patched.overlay_ids()]
    for name in names:
        for base, data in _regions(ctx.patched, name):
            if base <= addr < base + len(data):
                found.append((name, base, data))
    if not found:
        raise ValueError(f"{addr:#010x} is in no module of the patched ROM")
    if len(found) > 1 and found[0][0] != "arm9" and "[" not in found[0][0]:
        raise ValueError(f"{addr:#010x} is in {[f[0] for f in found]}: pass --module")
    name, base, data = found[0]
    if arm is None:
        arm = "[" not in name and _is_arm(ctx.symbols, name, addr)
    stop = min(addr + 4 * count, base + len(data))
    insns = _disassemble(data, base, addr, stop, arm)[:count]
    return [f"  {a:#010x}  {text}" for a, _, text in insns]


def _print_pair(title_a: str, a: list[str], title_b: str, b: list[str]) -> None:
    print(title_a)
    print("\n".join(a))
    print(title_b)
    print("\n".join(b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.rom.decode", description="Decode Kaizo ROM changes to stdout."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    code = sub.add_parser("code", help="changed ranges of a symbol, vanilla and patched")
    code.add_argument("module")
    code.add_argument("symbol")
    member = sub.add_parser("member", help="a NARC member's words, vanilla and patched")
    member.add_argument("narc")
    member.add_argument("index", type=int)
    at = sub.add_parser("at", help="instructions of the patched ROM at an address")
    at.add_argument("address", type=lambda s: int(s, 16))
    at.add_argument("count", type=int)
    at.add_argument("--module")
    at.add_argument("--arm", action="store_true", default=None)
    args = ap.parse_args(argv)

    try:
        if args.command == "code":
            for r in decode_code(args.module, args.symbol):
                _print_pair(
                    f"== {args.module} {args.symbol} {r.start:#010x}-{r.end:#010x} vanilla",
                    r.vanilla,
                    "-- patched",
                    r.patched,
                )
        elif args.command == "member":
            van, pat = decode_member(args.narc, args.index)
            _print_pair(
                f"== {args.narc}[{args.index}] vanilla ({len(van)} words)",
                van,
                f"-- patched ({len(pat)} words)",
                pat,
            )
        else:
            print("\n".join(disassemble_at(args.address, args.count, args.module, args.arm)))
    except (KeyError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
