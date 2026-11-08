"""Static check of the battle-RAM offsets that ``tools/trace_battle.lua`` reads.

``trace_battle.lua`` reads ``BattleSystem``, ``BattleContext`` and ``BattleMon`` fields at
offsets laid out by hand from the decomp's struct definitions (docs/tracing.md, "RAM
layout"). This module checks them against the machine code instead: for each field of the
script's ``SYS``, ``CTX`` and ``MON`` tables it disassembles decomp functions that touch the
field (``ACCESSES``, with the source line of the access), follows the struct pointer
through each function, and collects the offsets of the loads and stores of the field's
width made through it. ``check_offsets(game)`` does this on that game's ROM. A field is
``ok`` when the script's offset is among the offsets found in every listed function.

``MON`` fields are found through ``BattleContext.battleMons[battler]``: an access at
context offset ``x`` inside ``battleMons`` counts as ``BattleMon`` offset
``(x - battleMons) % 0xC0``. ``MON.size`` is checked against the multipliers the code
applies to a battler index.

Locating a function: vanilla uses the decomp build's linker map. Kaizo moved overlay16 (the
battle engine) by +4, +8, +0x1C or +0x20 bytes after its patched accuracy and switch-in
routines (``_KAIZO_OVERLAY16_SHIFT``, from the K3a overlay16 notes); a Kaizo function is
read at its vanilla address plus that shift, and its instruction sequence (mnemonics, with
literal pools skipped) must equal the vanilla one, or ``check_offsets`` raises. Every
function listed here is in overlay16 and outside the patched routines.

The analysis is a linear walk over the function's Thumb code. Per register it tracks a
constant, a pointer (``base + offset``, where a base is an argument register at entry, or
the word loaded through a pointer) or nothing. A pointer plus an unknown value (an array
index such as ``battler * 0xC0``) keeps its offset. Literal-pool loads read the ROM;
``sp`` slots carry spilled registers. Branches are not followed: after a call or an
unconditional branch the scratch registers r0-r3 and r12 are forgotten. Nothing is written
to disk.

CLI: ``uv run --group rom python -m tools.rom.offsets [--game kaizo]``.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from nuzline.data.games import VANILLA, get_game

LUA = Path(__file__).resolve().parents[2] / "tools" / "trace_battle.lua"

# Kaizo overlay16: (vanilla addresses below this bound, shift); patched = vanilla + shift.
# The None band is inside the patched BattleSystem_TriggerEffectOnSwitch (mixed shifts).
_KAIZO_OVERLAY16_SHIFT = (
    (0x0224F068, 0x0),
    (0x0224F386, 0x4),
    (0x02256440, 0x8),
    (0x02256778, None),
    (0x02270B20, 0x1C),
    (0x100000000, 0x20),
)


@dataclass(frozen=True)
class Access:
    """A decomp function whose code touches a field: the struct pointer is in argument
    register ``arg`` at entry, and the access is ``width`` bytes wide (0: the field is
    ``MON.size``, found as a multiplier of the battler index)."""

    function: str
    source: str  # external/pokeplatinum file:line of the access
    arg: int
    width: int


@dataclass(frozen=True)
class OffsetCheck:
    """One field of trace_battle.lua (``"CTX.totalTurns"``): its offset there, the offsets
    found in the ROM's code for an access of the field's width through the struct
    pointer, and whether every listed function has the script's offset among them."""

    field: str
    offset: int
    found: tuple[int, ...]
    ok: bool


BATTLE_MONS = (0x2D40, 4, 0xC0)  # BattleContext.battleMons: offset, count, BattleMon size

_BL = "src/battle/battle_lib.c"
_BCP = "src/battle/battle_controller_player.c"
_BS = "src/battle/battle_script.c"
_SYS = "src/battle/battle_system.c"

# (table, field) -> the functions that pin it. BattleSystem functions take the system in
# r0; BattleContext ones take (battleSys, battleCtx) or (battleCtx, ...).
ACCESSES: dict[tuple[str, str], tuple[Access, ...]] = {
    ("SYS", "battleCtx"): (Access("BattleSystem_GetBattleContext", f"{_SYS}:111", 0, 4),),
    ("SYS", "parties"): (Access("BattleSystem_GetParty", f"{_SYS}:127", 0, 4),),
    ("SYS", "seedRandNext"): (Access("BattleSystem_RandNext", f"{_SYS}:1310", 0, 4),),
    ("CTX", "command"): (
        Access("BattleControllerPlayer_UseMove", f"{_BCP}:3337", 1, 4),
        Access("BattleControllerPlayer_TurnEnd", f"{_BCP}:1857", 1, 4),
    ),
    ("CTX", "commandNext"): (Access("BattleControllerPlayer_UseMove", f"{_BCP}:3338", 1, 4),),
    ("CTX", "attacker"): (Access("BtlCmd_TrickRoom", f"{_BS}:7863", 1, 4),),
    ("CTX", "defender"): (Access("BtlCmd_SetRandomTarget", f"{_BS}:9224", 1, 4),),
    ("CTX", "totalTurns"): (
        Access("BattleControllerPlayer_TurnEnd", f"{_BCP}:1852", 1, 4),
        Access("Battler_CheckTruant", f"{_BL}:3309", 0, 4),
    ),
    ("CTX", "fieldConditionsMask"): (
        Access("Move_FailsInHighGravity", f"{_BL}:3398", 1, 4),
        Access("Battler_Ability", f"{_BL}:3094", 0, 4),
    ),
    ("CTX", "damage"): (Access("BtlCmd_CalcMaxDamage", f"{_BS}:1407", 1, 4),),
    ("CTX", "criticalMul"): (Access("BattleSystem_SetupLoop", f"{_BL}:3360", 1, 4),),
    ("CTX", "hpCalcTemp"): (Access("BtlCmd_TrySubstitute", f"{_BS}:5220", 1, 4),),
    ("CTX", "moveStatusFlags"): (
        Access("BattleSystem_SetupLoop", f"{_BL}:3359", 1, 4),
        Access("BtlCmd_FlickerMon", f"{_BS}:1667", 1, 4),
    ),
    ("CTX", "selectedPartySlot"): (Access("BattleMessage_PokeTag", f"{_BS}:11861", 1, 1),),
    ("CTX", "battleMons"): (Access("BattleSystem_CalcCriticalMulti", f"{_BL}:7119", 1, 2),),
    ("CTX", "moveCur"): (
        Access("Battler_LockMoveChoice", f"{_BL}:1831", 1, 4),
        Access("BtlCmd_CheckCurMoveIsType", f"{_BS}:9394", 1, 4),
    ),
    ("MON", "size"): (
        Access("Battler_CountMoves", f"{_BL}:3465", 1, 0),
        Access("Battler_StatusCondition", f"{_BL}:1845", 0, 0),
    ),
    ("MON", "species"): (Access("BattleSystem_CalcCriticalMulti", f"{_BL}:7119", 1, 2),),
    ("MON", "moves"): (Access("Battler_CountMoves", f"{_BL}:3465", 1, 2),),
    ("MON", "statBoosts"): (Access("BtlCmd_ResetAllStatChanges", f"{_BS}:3541", 1, 1),),
    ("MON", "ppCur"): (Access("BattleSystem_DecPPForPressure", f"{_BL}:3393", 0, 1),),
    ("MON", "level"): (Access("BattleSystem_CanWhirlwind", f"{_BL}:3076", 1, 1),),
    ("MON", "curHP"): (Access("Battler_CryModulation", f"{_BL}:6214", 0, 4),),
    ("MON", "maxHP"): (Access("Battler_CryModulation", f"{_BL}:6214", 0, 4),),
    ("MON", "status"): (
        Access("Battler_CryModulation", f"{_BL}:6216", 0, 4),
        Access("Battler_StatusCondition", f"{_BL}:1845", 0, 4),
    ),
    ("MON", "heldItem"): (Access("BattleSystem_NotHoldingMail", f"{_BL}:3069", 0, 2),),
}


def lua_tables(path: Path = LUA) -> dict[tuple[str, str], int]:
    """``(table, field) -> offset`` of the ``SYS``, ``CTX`` and ``MON`` tables of
    trace_battle.lua."""
    text = path.read_text(encoding="utf-8")
    out = {}
    for table in ("SYS", "CTX", "MON"):
        body = re.search(rf"^local {table} = \{{(.*?)^\}}", text, re.DOTALL | re.MULTILINE)
        if body is None:
            raise ValueError(f"{path}: no `local {table} = {{` table")
        for name, value in re.findall(
            r"^\s*(\w+) = (0x[0-9A-Fa-f]+|\d+)", body.group(1), re.MULTILINE
        ):
            out[(table, name)] = int(value, 0)
    return out


# --------------------------------------------------------------------------- analysis

_WIDTH = {"ldr": 4, "str": 4, "ldrh": 2, "strh": 2, "ldrsh": 2, "ldrb": 1, "strb": 1, "ldrsb": 1}
_SCRATCH = ("r0", "r1", "r2", "r3", "r12")
_CONDITIONAL = re.compile(r"b(eq|ne|cs|hs|cc|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le)")


@dataclass(frozen=True)
class _Ptr:
    base: str  # "arg0".."arg3", or "[<base>+0x..]": the word loaded through a pointer
    off: int


@dataclass(frozen=True)
class MemAccess:
    """A load or store through a tracked pointer."""

    addr: int  # instruction address
    base: str
    off: int
    width: int
    store: bool


@dataclass(frozen=True)
class Analysis:
    accesses: tuple[MemAccess, ...]
    multipliers: tuple[int, ...]  # constants a register was multiplied by
    mnemonics: tuple[str, ...]  # the instruction sequence, literal pools skipped


def _add(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return (a + b) & 0xFFFFFFFF
    if isinstance(a, int):
        a, b = b, a
    if isinstance(a, _Ptr) and isinstance(b, int):
        return _Ptr(a.base, a.off + (b if b < 0x80000000 else b - 0x100000000))
    if isinstance(a, _Ptr) and b is None:
        return a  # pointer + unknown index keeps its offset
    if isinstance(b, _Ptr) and a is None:
        return b
    return None


def analyse(data: bytes, base: int, start: int, size: int) -> Analysis:
    """The Thumb function at ``[start, start + size)`` of the image ``data`` loaded at
    ``base`` (module docstring)."""
    from capstone import CS_ARCH_ARM, CS_MODE_THUMB, Cs
    from capstone.arm import ARM_OP_IMM, ARM_OP_REG

    md = Cs(CS_ARCH_ARM, CS_MODE_THUMB)
    md.detail = True
    regs: dict[str, object] = {f"r{i}": _Ptr(f"arg{i}", 0) for i in range(4)}
    stack: dict[int, object] = {}
    sp = 0  # relative to entry
    pools: set[int] = set()
    accesses: list[MemAccess] = []
    muls: list[int] = []
    mnemonics: list[str] = []
    pending: dict[int, tuple[dict, dict, int]] = {}  # branch target -> state there
    dead = False  # the previous instruction never falls through
    pos, end = start, start + size

    def forget(names):
        for r in names:
            regs.pop(r, None)

    def merge(a: dict, b: dict) -> dict:
        return {k: v for k, v in a.items() if k in b and b[k] == v}

    def jump(target: int) -> None:
        here = (dict(regs), dict(stack), sp)
        if target in pending:
            r, s, p = pending[target]
            here = (merge(r, here[0]), merge(s, here[1]), p)
        pending[target] = here

    while pos < end:
        if pos in pools:
            pos += 4
            continue
        if pos in pending:
            r, s, p = pending.pop(pos)
            if dead:
                regs, stack, sp = dict(r), dict(s), p
            else:
                regs, stack = merge(regs, r), merge(stack, s)
        elif dead:
            forget(_SCRATCH)  # reached only by a backward branch or a jump table
        dead = False
        insn = next(md.disasm(data[pos - base : pos - base + 4], pos, 1), None)
        if insn is None:
            mnemonics.append(".hword")
            pos += 2
            continue
        pos += insn.size
        mn = insn.mnemonic
        mnemonics.append(mn)
        ops = insn.operands
        names = [md.reg_name(o.reg) if o.type == ARM_OP_REG else None for o in ops]
        if mn == "push":
            sp -= 4 * len(ops)
        elif mn == "pop":
            sp += 4 * len(ops)
            dead = "pc" in names
        elif mn in ("bl", "blx"):
            forget(_SCRATCH)
        elif mn == "bx":
            dead = True
        elif mn == "b" or _CONDITIONAL.fullmatch(mn):
            if ops and ops[0].type == ARM_OP_IMM:
                jump(ops[0].imm)
            dead = mn == "b"
        elif mn.startswith(("ldm", "stm")):
            forget(names if mn.startswith("ldm") else names[:1])
        elif mn in _WIDTH:
            rd, mem = names[0], ops[1]
            store = mn.startswith("str")
            rn = md.reg_name(mem.mem.base)
            if rn == "pc":
                lit = ((insn.address + 4) & ~3) + mem.mem.disp
                pools.add(lit)
                if base <= lit <= base + len(data) - 4:
                    regs[rd] = struct.unpack_from("<I", data, lit - base)[0]
                else:
                    forget([rd])
            elif rn == "sp":
                slot = sp + mem.mem.disp
                if store:
                    stack[slot] = regs.get(rd)
                else:
                    regs[rd] = stack.get(slot)
            else:
                index = md.reg_name(mem.mem.index) if mem.mem.index else None
                p = _add(regs.get(rn), regs.get(index) if index else mem.mem.disp)
                if isinstance(p, _Ptr):
                    accesses.append(MemAccess(insn.address, p.base, p.off, _WIDTH[mn], store))
                if not store:
                    if isinstance(p, _Ptr) and _WIDTH[mn] == 4:
                        regs[rd] = _Ptr(f"[{p.base}+{p.off:#x}]", 0)
                    else:
                        forget([rd])
        elif ops and ops[0].type == ARM_OP_REG and names[0] not in ("pc",):
            rd = names[0]
            srcs: list[object] = []
            for o, name in zip(ops[1:], names[1:], strict=True):
                if o.type == ARM_OP_IMM:
                    srcs.append(o.imm)
                elif o.type == ARM_OP_REG:
                    srcs.append("sp" if name in ("sp", "pc") else regs.get(name))
                else:
                    srcs.append(None)
            two = (regs.get(rd), srcs[0]) if len(srcs) == 1 else tuple(srcs[:2])
            if rd == "sp":
                if mn in ("add", "sub") and srcs and isinstance(srcs[-1], int):
                    sp += srcs[-1] if mn == "add" else -srcs[-1]
            elif mn in ("cmp", "cmn", "tst"):
                pass
            elif "sp" in srcs:
                forget([rd])  # a stack address: not tracked
            elif mn in ("mov", "movs"):
                regs[rd] = srcs[0] if srcs else None
            elif mn in ("add", "adds"):
                regs[rd] = _add(*two)
            elif mn in ("sub", "subs") and isinstance(two[1], int):
                regs[rd] = _add(two[0], -two[1])
            elif mn in ("lsl", "lsls") and all(isinstance(v, int) for v in two):
                regs[rd] = (two[0] << two[1]) & 0xFFFFFFFF
            elif mn in ("mul", "muls"):
                muls.extend(v for v in two if isinstance(v, int))
                forget([rd])
            else:
                forget([rd])
    return Analysis(tuple(accesses), tuple(muls), tuple(mnemonics))


# --------------------------------------------------------------------------- the check


def _shift(game: str, addr: int) -> int:
    if game == VANILLA:
        return 0
    for bound, shift in _KAIZO_OVERLAY16_SHIFT:
        if addr < bound:
            if shift is None:
                raise ValueError(f"{addr:#010x} is inside Kaizo's patched TriggerEffectOnSwitch")
            return shift
    raise AssertionError("unreachable")


class _Rom:
    """Functions of one game's ROM, analysed once each."""

    def __init__(self, game: str):
        from tools.rom.decode import _context

        self.game = get_game(game).id
        self.ctx = _context()
        self.cache: dict[str, Analysis] = {}

    def _analyse(self, files, symbol, address) -> Analysis:
        from tools.rom.decode import _is_arm, _region_at

        if symbol.module != "overlay16" or _is_arm(self.ctx.symbols, symbol.module, address):
            raise ValueError(f"{symbol.name}: only overlay16 Thumb functions are analysed")
        base, data = _region_at(files, symbol.module, address)
        return analyse(data, base, address, symbol.size)

    def function(self, name: str) -> Analysis:
        if name not in self.cache:
            symbol = self.ctx.symbols.find(name)
            vanilla = self._analyse(self.ctx.vanilla, symbol, symbol.address)
            if self.game == VANILLA:
                self.cache[name] = vanilla
            else:
                moved = symbol.address + _shift(self.game, symbol.address)
                patched = self._analyse(self.ctx.patched, symbol, moved)
                if patched.mnemonics != vanilla.mnemonics:
                    raise ValueError(
                        f"{name}: the code at {moved:#010x} in {self.game} does not match "
                        f"vanilla {symbol.address:#010x}"
                    )
                self.cache[name] = patched
        return self.cache[name]


def _found(rom: _Rom, table: str, access: Access) -> tuple[int, ...]:
    a = rom.function(access.function)
    if access.width == 0:
        return tuple(sorted(set(a.multipliers)))
    offs = {m.off for m in a.accesses if m.base == f"arg{access.arg}" and m.width == access.width}
    if table == "MON":
        first, count, size = BATTLE_MONS
        offs = {(x - first) % size for x in offs if first <= x < first + count * size}
    return tuple(sorted(offs))


def check_offsets(game: str) -> list[OffsetCheck]:
    """One ``OffsetCheck`` per ``SYS``/``CTX``/``MON`` field of trace_battle.lua, from
    ``game``'s ROM (id or alias). A field with no entry in ``ACCESSES`` fails."""
    rom = _Rom(game)
    out = []
    for (table, name), offset in lua_tables().items():
        accesses = ACCESSES.get((table, name), ())
        per_function = [_found(rom, table, a) for a in accesses]
        found = tuple(sorted({x for f in per_function for x in f}))
        ok = bool(per_function) and all(offset in f for f in per_function)
        out.append(OffsetCheck(f"{table}.{name}", offset, found, ok))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.rom.offsets", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--game", default="vanilla", help="vanilla or kaizo")
    args = ap.parse_args(argv)
    checks = check_offsets(args.game)
    for c in checks:
        table, name = c.field.split(".")
        where = ", ".join(f"{a.function} ({a.source})" for a in ACCESSES.get((table, name), ()))
        found = " ".join(f"{x:#x}" for x in c.found)
        print(
            f"{'ok ' if c.ok else 'BAD'} {c.field:24} {c.offset:#07x}  found [{found}]  in {where}"
        )
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
