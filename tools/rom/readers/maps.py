"""Trainer battle start conditions from map headers, zone events and field scripts.

A trainer's maps (Ruling 12) are the map headers whose ``zone_event.narc`` events have an
object with the trainer's battle script, or whose ``scr_seq.narc`` script file reaches a
battle command naming the trainer when walked by control flow, plus, for a Vs. Seeker
rematch, its base trainer's maps. The weather of those maps (``sMapHeaders``, with the
yearly weathers looked up in ``sYearlyWeather``) gives the condition, through the same
``WEATHER_CONDITION``, ``_start_condition`` and overrides file as ``extract_decomp``.

Two controller rulings (Task 12) refine Ruling 12 for the script walk (details in
``script_battles``):

* A. A battle operand >= 0x4000 is a variable and resolves only to the constants assigned
  to it (``SetVar``/``SetVarFromValue``) along the walked control-flow paths that reach
  that battle command. The file-wide rule ("every ``SetVarFromValue`` of the variable in
  the file") picked up item and accessory ids written to ``VAR_0x8004`` and misplaced 12
  vanilla trainers (e.g. 390 Arthur, 447 Willie).
* B. A constant assigned to ``VAR_PARTNER_TRAINER_ID`` on a reached path places that
  trainer on the map: it becomes the player's walking partner and joins the map's battles
  (``src/scrcmd_trainer.c:133-134``, ``src/overlay006/wild_encounters.c:344,531,661``).
  This places 608 Cheryl, 609 Riley, 610 Marley, 611 Buck and 612 Mira as the decomp does.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tools.extract_decomp import (
    START_CONDITIONS_FILE,
    WEATHER_CONDITION,
    _constants,
    _start_condition,
    _start_condition_overrides,
)
from tools.rom.readers import ReaderError, symbol_bytes

if TYPE_CHECKING:
    from tools.rom.nds import RomFiles
    from tools.rom.symbols import SymbolMap

SCRIPTS_NARC = "fielddata/script/scr_seq.narc"  # platinum.us/filesys.csv:189
EVENTS_NARC = "fielddata/eventdata/zone_event.narc"  # platinum.us/filesys.csv:259

# MapHeader (include/map_header.h:10-31): 24 bytes; scriptsArchiveID u16 at 4,
# eventsArchiveID u16 at 16, the mapLabelTextID/mapLabelWindowID bitfield u16 at 18,
# weather u8 at 20.
MAP_HEADER_SYMBOL = "sMapHeaders"
MAP_HEADER_SIZE = 24
HEADER_SCRIPTS, HEADER_EVENTS, HEADER_WEATHER = 4, 16, 20
# u8 [366][OVERWORLD_WEATHER_YEARLY_COUNT] (src/field_overworld_weather.c:17)
YEARLY_SYMBOL = "sYearlyWeather"
DAYS_OF_YEAR = 366
REMATCH_SYMBOL = "gVsSeekerRematchData"  # src/overlay005/vs_seeker.c:79-81,125
REMATCH_SLOTS = 6  # VS_SEEKER_MAX_REMATCHES, vs_seeker.c:43
REMATCH_NONE = (0, 0xFFFF)  # VS_SEEKER_REMATCH_DATA_END / _NONE, vs_seeker.c:44-45

# zone_event.narc member (tools/jsoncnv/event.py:25-131): bg events, object events, warp
# events and coord events, each a u32 count followed by fixed-size records.
EVENT_RECORD_SIZES = (20, 32, 12, 16)
OBJECT_EVENTS = 1
OBJECT_SCRIPT = 10  # script u16 at +10 (event.py:57)
# Battle scripts of object events (include/script_manager.h:93-95,
# tools/jsoncnv/convert.py:79-83): 3000 + id - 1 single, 5000 + id - 1 double.
SINGLE_BATTLE_SCRIPTS = 3000
DOUBLE_BATTLE_SCRIPTS = 5000
VISIBLE_ITEM_SCRIPTS = 7000  # the next range (script_manager.h:95)

# The script file header (asm/macros/scrcmd.inc:5105-5116): ScriptEntry is
# ``.long label-.-4`` and ScriptEntryEnd ``.short 0xFD13``.
SCRIPT_ENTRY_END = 0xFD13
# Operands at or above VARS_START are variables (src/script_manager.c:428-452,
# generated/vars_flags.txt:4107).
VARS_START = 0x4000
PARTNER_VAR = "VAR_PARTNER_TRAINER_ID"
SCRIPT_STACK_SIZE = 20  # ScriptContext.stack (include/field_script_context.h:20)

SCRCMD_H = Path("include") / "data" / "scripts" / "scrcmd.h"
SCRCMD_INC = Path("asm") / "macros" / "scrcmd.inc"
_WIDTHS = {"byte": 1, "short": 2, "long": 4}

# Commands whose last operand is a u32 offset from the end of the command that the
# script may continue at (src/scrcmd.c:1004-1083, src/scrcmd_trainer.c:310-326).
_BRANCHES = {
    "SCRCMD_GOTO",
    "SCRCMD_CALL",
    "SCRCMD_GOTOIF",
    "SCRCMD_CALLIF",
    "SCRCMD_GOTOIFTARGETTRAINERDEFEATED",
    "SCRCMD_UNUSED_017",
    "SCRCMD_UNUSED_018",
    "SCRCMD_UNUSED_019",
}
# Commands after which execution does not fall through: ScrCmd_End stops the context,
# ScrCmd_Return pops it, ScrCmd_GoTo jumps (src/scrcmd.c:780-784,1004-1008,1058-1062).
_PATH_ENDS = {"SCRCMD_END", "SCRCMD_RETURN", "SCRCMD_GOTO"}
# Battle commands whose every operand is a trainer (asm/macros/scrcmd.inc:1427,1780,4020).
_BATTLES = {"SCRCMD_STARTTRAINERBATTLE", "SCRCMD_STARTTAGBATTLE", "SCRCMD_STARTFIRSTBATTLE"}
_SET_VAR = "SCRCMD_SETVARFROMVALUE"  # destVarID u16, value u16 (scrcmd.inc:297-301)


class ScriptDecodeError(ReaderError):
    """A script holds bytes the command table cannot decode."""

    def __init__(self, script: str, offset: int, opcode: int | None, why: str):
        self.script, self.offset, self.opcode = script, offset, opcode
        op = "" if opcode is None else f" opcode {opcode:#x}"
        super().__init__(f"{script} offset {offset:#x}{op}: {why}")


# A variable-length command's decoder: (script, operand offset) -> the offset after it.
Decoder = Callable[[bytes, int], int]


class CommandTable(dict):
    """opcode -> (macro name, operand widths), plus each opcode's ``SCRCMD_*`` constant
    (``constants[opcode]``) and the decoders of the variable-length commands
    (``variable[opcode]``, whose widths are only the leading selector), and the id of
    ``VAR_PARTNER_TRAINER_ID`` (``partner_var``)."""

    def __init__(
        self,
        items,
        constants: list[str],
        variable: dict[int, Decoder],
        partner_var: int,
    ):
        super().__init__(items)
        self.constants = constants
        self.opcode = {c: i for i, c in enumerate(constants)}
        self.variable = variable
        self.partner_var = partner_var


def _defines(decomp: Path, rel: str, prefix: str) -> dict[str, int]:
    path = decomp / rel
    text = path.read_text(encoding="utf-8")
    out = {
        k: int(v, 0)
        for k, v in re.findall(rf"#define ({prefix}\w+)\s+(\d+)\s*$", text, re.MULTILINE)
    }
    if not out:
        raise ReaderError(f"{path}: no {prefix}* constants")
    return out


def _selector_decoder(
    name: str, width: int, operands: dict[int, int], cite: str, strict: bool = True
) -> Decoder:
    """A command whose first operand (``width`` bytes) selects how many u16 operands
    follow (``operands``), as its handler reads them (``cite``). A selector the handler
    does not list raises (``strict``), since nothing in a reached script should use one."""

    def decode(script: bytes, pos: int) -> int:
        if pos + width > len(script):
            raise ReaderError(f"{name}: selector past the end of the script")
        sel = script[pos] if width == 1 else struct.unpack_from("<H", script, pos)[0]
        if sel not in operands:
            raise ReaderError(f"{name}: selector {sel} not handled by {cite}")
        return pos + width + 2 * operands[sel]

    decode.__doc__ = f"{name}: {cite}"
    return decode


def _variable_decoders(decomp: Path) -> dict[str, tuple[int, Decoder]]:
    """Decoders for the six commands whose length depends on an operand, reading as their
    handlers do. ``ScriptContext_GetVar`` and ``ScriptContext_GetVarPointer`` each read a
    u16 (src/field_script_context.c)."""
    fm = _defines(decomp, "include/constants/scrcmd.h", "FIELD_MOVE_FUNC_")
    # ScrCmd_DoStrengthFunc / DoFlashFunc / DoDefogFunc (src/scrcmd_system_flags.c:161-238):
    # a u8 func; CHECK_ACTIVE reads a destination var; any other value asserts.
    field_move = {
        fm["FIELD_MOVE_FUNC_CLEAR_ACTIVE"]: 0,
        fm["FIELD_MOVE_FUNC_SET_ACTIVE"]: 0,
        fm["FIELD_MOVE_FUNC_CHECK_ACTIVE"]: 1,
    }
    gc = _defines(decomp, "include/constants/records_mixed_rng.h", "GC_ACTION_")
    # ScrCmd_DoGroupConnectionAction (src/scrcmd_group_connection.c:21-90): a u16 action.
    group = {
        gc["GC_ACTION_CHECK_IS_ENTRY_VALID"]: 2,
        gc["GC_ACTION_CHECK_IS_ENTRY_OVERRIDE"]: 2,
        gc["GC_ACTION_BUFFER_GROUP_NAME"]: 2,
        gc["GC_ACTION_BUFFER_LEADER_NAME"]: 2,
        gc["GC_ACTION_OPEN_GROUP_NAMING_SCREEN"]: 1,
        gc["GC_ACTION_JOIN_GROUP"]: 1,
        gc["GC_ACTION_MAKE_NEW_GROUP"]: 0,
    }
    tv = _defines(decomp, "include/constants/tv_broadcast.h", "TV_BROADCAST_CALL_")
    # ScrCmd_CallTVBroadcast (src/scrcmd_tv_broadcast.c:60-108): a u16 call id.
    broadcast = {
        tv["TV_BROADCAST_CALL_CHECK_STATUS"]: 1,
        tv["TV_BROADCAST_CALL_LOAD_FRAMING_MESSAGE"]: 3,
        tv["TV_BROADCAST_CALL_LOAD_SEGMENT"]: 3,
        tv["TV_BROADCAST_CALL_FINISH_PROGRAM"]: 0,
        tv["TV_BROADCAST_CALL_LOAD_COMMERCIAL"]: 2,
        tv["TV_BROADCAST_CALL_UNUSED"]: 3,
        tv["TV_BROADCAST_CALL_GET_NEXT_SEGMENT_ID"]: 1,
    }
    stage = {
        s: i
        for i, s in enumerate(_constants(decomp, "mystery_gift_delivery_stages", "MYSTERY_GIFT_"))
    }
    # ScrCmd_MysteryGiftGive (src/scrcmd_mystery_gift.c:89-150): a u16 stage.
    gift = {
        stage["MYSTERY_GIFT_LOAD"]: 0,
        stage["MYSTERY_GIFT_UNLOAD_NOT_RECEIVED"]: 0,
        stage["MYSTERY_GIFT_UNLOAD_RECEIVED"]: 0,
        stage["MYSTERY_GIFT_CHECK_AVAILABLE_PGT"]: 1,
        stage["MYSTERY_GIFT_GET_PGT_TYPE"]: 1,
        stage["MYSTERY_GIFT_CHECK_CAN_RECEIVE"]: 1,
        stage["MYSTERY_GIFT_GIVE"]: 0,
        stage["MYSTERY_GIFT_RECEIVED"]: 2,
        stage["MYSTERY_GIFT_CANT_RECEIVE"]: 2,
    }
    flags = "src/scrcmd_system_flags.c:161-238"
    return {
        "DoStrengthFunc": (1, _selector_decoder("DoStrengthFunc", 1, field_move, flags)),
        "DoFlashFunc": (1, _selector_decoder("DoFlashFunc", 1, field_move, flags)),
        "DoDefogFunc": (1, _selector_decoder("DoDefogFunc", 1, field_move, flags)),
        "DoGroupConnectionAction": (
            2,
            _selector_decoder(
                "DoGroupConnectionAction", 2, group, "src/scrcmd_group_connection.c:21-90"
            ),
        ),
        "CallTVBroadcast": (
            2,
            _selector_decoder("CallTVBroadcast", 2, broadcast, "src/scrcmd_tv_broadcast.c:60-108"),
        ),
        "MysteryGiftGive": (
            2,
            _selector_decoder("MysteryGiftGive", 2, gift, "src/scrcmd_mystery_gift.c:89-150"),
        ),
    }


def command_table(decomp: Path) -> CommandTable:
    """opcode -> (macro name, operand widths).

    The opcode is the index of the command's ``ScriptCommand(constant, handler)`` line in
    ``include/data/scripts/scrcmd.h`` (``data/scripts/cmd_table.h`` numbers them with
    ``inc_enum``); every line counts, including the one whose constant is spelled
    ``ScrCmd_GETRANDOMBATTLEGROUNDTRAINERS``. The widths come from the
    ``asm/macros/scrcmd.inc`` macro whose body is ``.short <constant>`` followed only by
    ``.byte``/``.short``/``.long`` lines. The six macros with conditional bodies get the
    decoders of ``_variable_decoders``; any other command macro shape raises, as do two
    macros that disagree on one opcode's widths, or an opcode left without a macro.
    """
    decomp = Path(decomp)
    hpath = decomp / SCRCMD_H
    constants = []
    for lineno, line in enumerate(hpath.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("ScriptCommand("):
            m = re.fullmatch(r"ScriptCommand\((\w+),\s*(\w+)\)\s*", line)
            if not m:
                raise ReaderError(f"{hpath}:{lineno}: unparsed command line {line!r}")
            constants.append(m.group(1))
    opcode = {c: i for i, c in enumerate(constants)}
    if len(opcode) != len(constants):
        raise ReaderError(f"{hpath}: a command constant is listed twice")

    variable = _variable_decoders(decomp)
    ipath = decomp / SCRCMD_INC
    src = ipath.read_text(encoding="utf-8")
    table: dict[int, tuple[str, tuple[int, ...]]] = {}
    decoders: dict[int, Decoder] = {}
    seen_variable = set()
    for m in re.finditer(
        r"^[ \t]*\.macro[ \t]+(\w+)[^\n]*\n(.*?)^[ \t]*\.endm\b", src, re.MULTILINE | re.DOTALL
    ):
        name, body = m.group(1), m.group(2)
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", body, flags=re.DOTALL)
        lines = [s.strip() for s in body.splitlines() if s.strip()]
        head = re.fullmatch(r"\.short\s+(\w+)", lines[0]) if lines else None
        if not head or head.group(1) not in opcode:
            continue  # a wrapper around another macro, or not a command
        op = opcode[head.group(1)]
        lineno = src.count("\n", 0, m.start()) + 1
        if name in variable:
            width, decode = variable[name]
            widths: tuple[int, ...] = (width,)
            decoders[op] = decode
            seen_variable.add(name)
        else:
            parsed = [re.fullmatch(r"\.(byte|short|long)\s+([^,]+)", s) for s in lines[1:]]
            if not all(parsed):
                raise ReaderError(f"{ipath}:{lineno}: macro {name} has a variable body")
            widths = tuple(_WIDTHS[p.group(1)] for p in parsed)
        if op in table and table[op][1] != widths:
            raise ReaderError(
                f"{ipath}:{lineno}: {name} gives {constants[op]} operands {widths}, "
                f"{table[op][0]} gives {table[op][1]}"
            )
        table.setdefault(op, (name, widths))
    missing = [constants[i] for i in range(len(constants)) if i not in table]
    if missing or seen_variable != set(variable):
        raise ReaderError(
            f"{ipath}: no macro for {missing}; variable macros not found "
            f"{sorted(set(variable) - seen_variable)}"
        )
    return CommandTable(table, constants, decoders, var_ids(decomp)[PARTNER_VAR])


def var_ids(decomp: Path) -> dict[str, int]:
    """Var/flag constant -> id from ``generated/vars_flags.txt`` (a metang enum,
    ``generated/meson.build:112``): a bare name is one more than the previous entry, and
    ``NAME = X`` sets it to the integer or earlier name ``X``, as a C enum counts."""
    path = Path(decomp) / "generated" / "vars_flags.txt"
    out: dict[str, int] = {}
    value = -1
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z_]\w*)(?:\s*=\s*(\w+))?", line)
        if not m:
            raise ReaderError(f"{path}:{lineno}: unparsed line {raw!r}")
        name, expr = m.groups()
        if expr is None:
            value += 1
        elif expr.isdecimal():
            value = int(expr)
        elif expr in out:
            value = out[expr]
        else:
            raise ReaderError(f"{path}:{lineno}: {expr!r} is not defined earlier")
        out[name] = value
    return out


def script_entries(script: bytes, name: str = "script") -> list[int]:
    """Absolute offsets of a script file's entries (``scrcmd.inc:5105-5116``): u32
    ``label-.-4`` offsets until the u16 ``0xFD13``.

    Four vanilla files have no ``ScriptEntryEnd`` (``scripts_battles.s``,
    ``scripts_hearthome_city_dp_gym_leader_room.s``, ``scripts_hidden_items.s``,
    ``scripts_sunyshore_market.s``); the game indexes entries without a bound
    (``src/script_manager.c:412-416``), so their header ends where the first code an entry
    points at begins. An entry pointing into the header raises.
    """
    out: list[int] = []
    pos = 0
    while not out or pos < min(out):
        if pos + 2 > len(script):
            raise ScriptDecodeError(name, pos, None, "header has no 0xFD13 terminator")
        if struct.unpack_from("<H", script, pos)[0] == SCRIPT_ENTRY_END:
            pos += 2
            break
        if pos + 4 > len(script):
            raise ScriptDecodeError(name, pos, None, "truncated header entry")
        out.append(pos + 4 + struct.unpack_from("<i", script, pos)[0])
        pos += 4
    if out and min(out) < pos:
        raise ScriptDecodeError(name, min(out), None, f"entry points into the header (0..{pos})")
    return out


def _decode(script: bytes, table: CommandTable, name: str, pos: int):
    """(opcode, fixed operand values, offset after the command) of the command at ``pos``."""
    if not 0 <= pos <= len(script) - 2:
        raise ScriptDecodeError(name, pos, None, "command outside the script")
    op = struct.unpack_from("<H", script, pos)[0]
    if op not in table:
        raise ScriptDecodeError(name, pos, op, "opcode not in the command table")
    at = pos + 2
    values: list[int] = []
    if op in table.variable:
        try:
            end = table.variable[op](script, at)
        except ReaderError as e:
            raise ScriptDecodeError(name, pos, op, str(e)) from None
    else:
        end = at
        for w in table[op][1]:
            values.append(int.from_bytes(script[end : end + w], "little", signed=w == 4))
            end += w
    if end > len(script):
        raise ScriptDecodeError(name, pos, op, f"{table[op][0]} runs past the end")
    return op, values, end


# A path's constant variable values: var -> the values its last SetVarFromValue gave.
_Env = dict[int, frozenset[int]]


def script_battles(script: bytes, table: CommandTable, name: str = "script") -> set[int]:
    """Trainer ids placed on the map by the script, walking by control flow from every
    header entry.

    * Every operand of a reached ``StartTrainerBattle``, ``StartTagBattle`` or
      ``StartFirstBattle`` counts (Ruling 12). 0 is no trainer.
    * An operand at or above ``VARS_START`` is a variable. It resolves only to the
      constants assigned to it (``SetVarFromValue``, which the ``SetVar`` macro assembles
      to for a constant, ``scrcmd.inc:309-315``) on the paths that reach that battle
      command: each path carries var -> values, a ``SetVarFromValue`` replaces the
      variable's values, and paths that meet at a command (with the same call stack) are
      merged. ``Call``/``CallIf`` push the return offset and ``Return`` pops it; an empty
      stack ends the path, as ``ScriptContext_Pop`` returns NULL
      (``src/field_script_context.c:112-135``). Controller ruling A (Task 12): the
      file-wide rule ("every ``SetVarFromValue`` of the variable in the file") picked up
      item and accessory ids written to ``VAR_0x8004`` for unrelated reasons and
      misplaced 12 vanilla trainers (e.g. 390 Arthur from ``ITEM_TM63`` in Veilstone
      City, 447 Willie from ``ITEM_SUPER_ROD`` in the Fight Area). Variables written by
      other commands are not constants and add nothing.
    * A constant assigned to ``VAR_PARTNER_TRAINER_ID`` on a reached path places that
      trainer too (controller ruling B): it makes the trainer the player's walking
      partner, who joins the battles on that map. The battle setup reads it through
      ``SystemVars_GetPartnerTrainerID`` (``src/system_vars.c:60-63``) for script
      trainer battles (``src/scrcmd_trainer.c:133-134``) and wild battles
      (``src/overlay006/wild_encounters.c:344,531,661``).
    """
    oc = table.opcode
    branch = {oc[c] for c in _BRANCHES}
    calls = {oc["SCRCMD_CALL"], oc["SCRCMD_CALLIF"]}
    ends = {oc[c] for c in _PATH_ENDS}
    battles = {oc[c] for c in _BATTLES}
    set_var, ret, call = oc[_SET_VAR], oc["SCRCMD_RETURN"], oc["SCRCMD_CALL"]
    for op in branch:
        if table[op][1][-1:] != (4,):
            raise ReaderError(f"{table.constants[op]}: last operand is not a u32 offset")

    decoded: dict[int, tuple[int, list[int], int]] = {}
    seen: dict[tuple[int, tuple[int, ...]], _Env] = {}
    out: set[int] = set()
    work: list[tuple[int, tuple[int, ...], _Env]] = [
        (e, (), {}) for e in script_entries(script, name)
    ]
    while work:
        pos, stack, env = work.pop()
        old = seen.get((pos, stack))
        if old is not None:
            merged = {
                v: old.get(v, frozenset()) | env.get(v, frozenset()) for v in old.keys() | env
            }
            if merged == old:
                continue
            env = merged
        seen[(pos, stack)] = env
        if pos not in decoded:
            decoded[pos] = _decode(script, table, name, pos)
        op, values, end = decoded[pos]
        if op in battles:
            for v in values:
                out |= env.get(v, frozenset()) if v >= VARS_START else {v}
        elif op == set_var:
            if values[0] == table.partner_var:
                out.add(values[1])
            env = {**env, values[0]: frozenset({values[1]})}
        if op == ret:
            if stack:
                work.append((stack[-1], stack[:-1], env))
            continue
        if op in branch:
            target = end + values[-1]
            if op in calls:
                if len(stack) + 1 >= SCRIPT_STACK_SIZE:
                    raise ScriptDecodeError(name, pos, op, "call stack overflow")
                work.append((target, (*stack, end), env))
            else:
                work.append((target, stack, env))
            if op == call:
                continue  # the path comes back to `end` through Return
        if op in ends:
            continue
        work.append((end, stack, env))
    out.discard(0)
    return out


def _events_object_scripts(member: bytes, index: int) -> list[int]:
    """The object events' script ids of one ``zone_event.narc`` member."""
    pos, scripts = 0, []
    for kind, size in enumerate(EVENT_RECORD_SIZES):
        if pos + 4 > len(member):
            raise ReaderError(f"{EVENTS_NARC} member {index} offset {pos}: truncated count")
        n = struct.unpack_from("<I", member, pos)[0]
        pos += 4
        if pos + n * size > len(member):
            raise ReaderError(
                f"{EVENTS_NARC} member {index} offset {pos}: {n} records of {size} bytes "
                f"overrun the member ({len(member)} bytes)"
            )
        if kind == OBJECT_EVENTS:
            scripts = [
                struct.unpack_from("<H", member, pos + i * size + OBJECT_SCRIPT)[0]
                for i in range(n)
            ]
        pos += n * size
    if pos != len(member):
        raise ReaderError(
            f"{EVENTS_NARC} member {index} offset {pos}: {len(member) - pos} trailing bytes"
        )
    return scripts


def _weather_names(decomp: Path) -> tuple[dict[int, str], int, int]:
    """Overworld weather id -> constant (``include/constants/overworld_weather.h``), as
    ``_weather_conditions`` reads it, with the yearly weathers' first id and count."""
    path = decomp / "include" / "constants" / "overworld_weather.h"
    hdr = path.read_text(encoding="utf-8")
    names = {
        int(v): k
        for k, v in re.findall(r"^#define (OVERWORLD_WEATHER_\w+)\s+(\d+)\s*$", hdr, re.MULTILINE)
    }
    start = int(re.search(r"#define OVERWORLD_WEATHER_YEARLY_START\s+(\d+)", hdr).group(1))
    yearly = re.findall(
        r"#define (OVERWORLD_WEATHER_\w+)\s+\(OVERWORLD_WEATHER_YEARLY_START \+ (\d+)\)", hdr
    )
    count = sum(1 for n, _ in yearly if n != "OVERWORLD_WEATHER_YEARLY_END")
    names.pop(start, None)  # OVERWORLD_WEATHER_YEARLY_START is the first yearly id
    return names, start, count


def _map_conditions(rom: RomFiles, decomp: Path, symbols: SymbolMap, weathers: list[int]):
    """Map header weather id -> the set of start conditions it can give."""
    names, start, count = _weather_names(decomp)
    yearly = symbol_bytes(rom, symbols, YEARLY_SYMBOL)
    if len(yearly) != DAYS_OF_YEAR * count:
        raise ReaderError(f"{YEARLY_SYMBOL}: {len(yearly)} bytes, expected {DAYS_OF_YEAR}x{count}")
    out = {}
    for w in set(weathers):
        if start <= w < start + count:
            days = [yearly[d * count + w - start] for d in range(DAYS_OF_YEAR)]
        else:
            days = [w]
        conds = set()
        for d in days:
            if d not in names:
                raise ReaderError(f"weather {d} (map weather {w}) is not an overworld weather")
            conds.add(WEATHER_CONDITION.get(names[d], "none"))
        out[w] = conds
    return out


def trainer_maps(
    rom: RomFiles, decomp: Path, symbols: SymbolMap, n_trainers: int
) -> tuple[dict[int, set[int]], list[int]]:
    """Trainer id -> the map header ids where a battle with it starts, and each map
    header's weather id."""
    raw = symbol_bytes(rom, symbols, MAP_HEADER_SYMBOL)
    if len(raw) % MAP_HEADER_SIZE:
        raise ReaderError(f"{MAP_HEADER_SYMBOL}: {len(raw)} bytes, not whole {MAP_HEADER_SIZE}s")
    scripts = rom.narc(SCRIPTS_NARC)
    events = rom.narc(EVENTS_NARC)
    table = command_table(decomp)
    battles: dict[int, set[int]] = {}
    where: dict[int, set[int]] = {}
    weathers = []

    def place(trainer: int, header: int, source: str) -> None:
        if not 0 < trainer < n_trainers:
            raise ReaderError(f"map header {header}: {source} names unknown trainer {trainer}")
        where.setdefault(trainer, set()).add(header)

    for header in range(len(raw) // MAP_HEADER_SIZE):
        base = header * MAP_HEADER_SIZE
        script_id, events_id = (
            struct.unpack_from("<H", raw, base + HEADER_SCRIPTS)[0],
            struct.unpack_from("<H", raw, base + HEADER_EVENTS)[0],
        )
        weathers.append(raw[base + HEADER_WEATHER])
        if events_id >= len(events) or script_id >= len(scripts):
            raise ReaderError(f"map header {header}: events {events_id} / scripts {script_id}")
        for s in _events_object_scripts(events[events_id], events_id):
            if SINGLE_BATTLE_SCRIPTS <= s < DOUBLE_BATTLE_SCRIPTS:
                place(s - SINGLE_BATTLE_SCRIPTS + 1, header, f"{EVENTS_NARC} {events_id}")
            elif DOUBLE_BATTLE_SCRIPTS <= s < VISIBLE_ITEM_SCRIPTS:
                place(s - DOUBLE_BATTLE_SCRIPTS + 1, header, f"{EVENTS_NARC} {events_id}")
        if script_id not in battles:
            battles[script_id] = script_battles(
                scripts[script_id], table, f"{SCRIPTS_NARC} member {script_id}"
            )
        for t in battles[script_id]:
            place(t, header, f"{SCRIPTS_NARC} {script_id}")

    rematch = symbol_bytes(rom, symbols, REMATCH_SYMBOL)
    row = 2 * REMATCH_SLOTS
    if len(rematch) % row:
        raise ReaderError(f"{REMATCH_SYMBOL}: {len(rematch)} bytes, not whole {row}-byte rows")
    for off in range(0, len(rematch), row):
        base, *rematches = struct.unpack_from(f"<{REMATCH_SLOTS}H", rematch, off)
        for t in rematches:
            if t in REMATCH_NONE:
                continue
            if not 0 < t < n_trainers:
                raise ReaderError(f"{REMATCH_SYMBOL} offset {off}: unknown trainer {t}")
            where.setdefault(t, set()).update(where.get(base, set()))
    return where, weathers


def trainer_start_conditions(
    rom: RomFiles,
    decomp: Path,
    symbols: SymbolMap,
    *,
    overrides: Path | None = START_CONDITIONS_FILE,
) -> list[str]:
    """One start condition per trainer id, derived as ``extract_decomp.extract_trainers``
    does (§6.3) from the maps of ``trainer_maps``."""
    decomp = Path(decomp)
    trainers = _constants(decomp, "trainers", "TRAINER_")
    where, weathers = trainer_maps(rom, decomp, symbols, len(trainers))
    conditions = _map_conditions(rom, decomp, symbols, weathers)
    fixed = _start_condition_overrides(overrides, trainers) if overrides else {}
    out = []
    for tid, const in enumerate(trainers):
        derived = set()
        for header in where.get(tid, ()):
            derived |= conditions[weathers[header]]
        out.append(_start_condition(const, derived, fixed))
    return out
