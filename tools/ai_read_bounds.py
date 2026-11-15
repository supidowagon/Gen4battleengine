"""Generate ``src/nuzline/ai/read_bounds.py``: the most random reads each trainer-AI flag script
makes on one move slot's path, per kind of move (E1 Ruling R6).

A path through a flag script is a walk of ``src/battle/trainer_ai/script.s`` from the flag's
``FlagTable`` entry (script.s:18-50) to its ``PopOrEnd``. The script has no calls and no
loops, so the most reads over every walk is a depth-first maximum. A read is a draw the exact
per-slot layout keys by path order (E1 Ruling R4): ``IfRandomLessThan`` (trainer_ai.c:528-538)
and ``LoadBattlerAbility`` on the defender, which flips a coin when the AI does not know the
player's ability (trainer_ai.c:1170-1209; counted always, an upper bound).
``CheckBattlerAbility`` draws nothing (:1212-1265). ``IfSpeedCompareEqualTo``'s tie is a read of the exact layout only for COMPARE_SPEED_FASTER (E1 Ruling R8; plan K3c Ruling 3 for SLOWER).

Branches whose outcome is fixed by the move are decided, so the bound follows the move:

* ``IfCurrentMoveEffectEqualTo`` / ``NotEqualTo`` (trainer_ai.c:1441-1465) by its effect;
  ``LoadCurrentMoveEffect`` (:1164) loads it, and the ``IfLoaded*`` compares and table tests
  that follow (``IfLoadedEqualTo``/``NotEqualTo``, ``IfLoadedInTable``/``NotInTable``,
  :776-891) are decided until another command loads ``calcTemp``;
* ``FlagMoveDamageScore`` (:1012-1063) makes a damage comparison exactly when the move is
  damaging (an ``sAltPowerMoveEffects`` effect whatever its power, else power above 1 and not
  a ``sNoDamageCalcMoveEffects`` effect; ``commands.is_damaging``), so a following ``IfLoaded(Not)EqualTo AI_NO_COMPARISON_MADE``
  is decided; ``IfCurrentMoveKills`` (:1525-1560) never jumps for a non-damaging move;
* ``IfMoveEffectivenessEquals TYPE_MULTI_QUADRUPLE_DAMAGE`` (:1311) never jumps when no two
  distinct types take the move's type super effectively (the second defender type is skipped
  when it equals the first, battle_lib.c:2635);
* ``IfMoveEqualTo`` (:834) never jumps when the named move's battle effect in the Kaizo move
  table (``Tables.move_ai_effect``, the effect the scripts read) is not the current one;
* ``IfTargetIsPartner`` never jumps in singles.

Every other branch is taken both ways, so the table is an upper bound: a path never makes more
reads than its row says (a layout bound that is never exceeded is never a refusal).

Run from the repository root with the decomp and the built Kaizo tables present:
    uv run python -m tools.ai_read_bounds
``tests/ai/test_read_bounds.py`` regenerates the table and compares.
"""

from __future__ import annotations

import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECOMP = ROOT / "external/pokeplatinum"
OUT = ROOT / "src/nuzline/ai/read_bounds.py"

# FlagTable bits whose scripts are bounded (the Kaizo policy's transcribed flags,
# layout.KAIZO_FLAG_BITS), with their entry labels (script.s:19-28).
FLAG_LABELS = {
    0: "Basic_Main",
    1: "EvalAttack_Main",
    2: "Expert_Main",
    3: "SetupFirstTurn_Main",
    4: "Risky_Main",
    5: "PrioritizeExtremes_Main",
    8: "CheckHP_Main",
    9: "Weather_Main",
}
_ENDS = {"PopOrEnd", "Escape"}
# Commands that leave calcTemp as it is: jumps on the state, scoring and flow.
_KEEPS_CALC = {"AddToMoveScore", "GoTo", "PopOrEnd", "Escape"}


class Script:
    def __init__(self, decomp: Path, move_effect):
        """``move_effect``: the battle effect of each move id (``Tables.move_ai_effect``)."""
        self.lines = (decomp / "src/battle/trainer_ai/script.s").read_text("utf-8").splitlines()
        self.effects = (decomp / "generated/move_battle_effects.txt").read_text("utf-8").split()
        self.moves = (decomp / "generated/moves.txt").read_text("utf-8").split()
        self.move_effect = [int(e) for e in move_effect]
        self.labels: dict[str, int] = {}
        for i, line in enumerate(self.lines):
            s = line.strip()
            if s.endswith(":") and " " not in s and not s.startswith("."):
                self.labels[s[:-1]] = i
        flag_table = self.labels["FlagTable"]
        for bit, label in FLAG_LABELS.items():
            entry = self.lines[flag_table + 1 + bit].replace(",", " ").split()
            assert entry[:2] == ["LabelDistance", label], (bit, entry)

    def command(self, i: int):
        s = self.lines[i].split("//")[0].strip()
        if not s or s.endswith(":") or s.startswith((".", "#")):
            return None
        parts = s.replace(",", " ").split()
        return parts[0], parts[1:]

    def table(self, label: str) -> frozenset[int]:
        out, i = set(), self.labels[label] + 1
        while True:
            cmd, args = self.command(i) or (None, None)
            if cmd == "TableEntry":
                if args[0] == "TABLE_END":
                    return frozenset(out)
                out.add(self.effects.index(args[0]))
            elif cmd is not None:
                raise ValueError(f"{label}: unexpected {cmd} at script.s:{i + 1}")
            i += 1

    def effect_id(self, name: str) -> int | None:
        return self.effects.index(name) if name in self.effects else None


def flag_reads(script: Script, bit: int, effect: int, damaging: bool, can_quad: bool) -> int:
    """The most reads flag ``bit``'s script makes on one path for a move of ``effect``."""
    lines = script.lines

    @cache
    def most(i: int, calc: tuple) -> int:
        while True:
            if i >= len(lines):
                raise RuntimeError(f"flag {bit}: fell off the end of script.s")
            c = script.command(i)
            if c is None:
                i += 1
                continue
            cmd, args = c
            if cmd == "TableEntry":
                raise RuntimeError(f"flag {bit}: ran into a table at script.s:{i + 1}")
            if cmd in _ENDS:
                return 0
            reads = 0
            if cmd == "IfRandomLessThan":
                reads = 1
            elif cmd == "LoadBattlerAbility":
                reads = int(args[0] == "AI_BATTLER_DEFENDER")
            elif cmd == "IfSpeedCompareEqualTo":
                reads = int(args[0] != "COMPARE_SPEED_SLOWER")  # the tie (E1 Ruling R8)
            target = args[-1] if args and args[-1] in script.labels else None
            if cmd == "GoTo":
                return reads + most(script.labels[target], calc)
            taken = _decided(script, cmd, args, calc, effect, damaging, can_quad)
            nxt = _next_calc(cmd, calc, effect, damaging)
            if target is None or not cmd.startswith("If"):
                if reads:
                    return reads + most(i + 1, nxt)
                calc = nxt
                i += 1
                continue
            if taken is True:
                return reads + most(script.labels[target], nxt)
            if taken is False:
                if reads:
                    return reads + most(i + 1, nxt)
                calc = nxt
                i += 1
                continue
            return reads + max(most(script.labels[target], nxt), most(i + 1, nxt))

    return most(script.labels[FLAG_LABELS[bit]] + 1, ("unknown",))


def _next_calc(cmd: str, calc: tuple, effect: int, damaging: bool) -> tuple:
    if cmd == "LoadCurrentMoveEffect":
        return ("effect", effect)
    if cmd == "FlagMoveDamageScore":
        return ("compared", damaging)
    if cmd.startswith("If") or cmd in _KEEPS_CALC:
        return calc
    return ("unknown",)


def _decided(script, cmd, args, calc, effect, damaging, can_quad) -> bool | None:
    """True / False when the jump of ``cmd`` is fixed by the move, else None."""
    if cmd == "IfTargetIsPartner":
        return False
    if cmd in ("IfCurrentMoveEffectEqualTo", "IfCurrentMoveEffectNotEqualTo"):
        same = script.effect_id(args[0]) == effect
        return same if cmd.endswith("EqualTo") and "Not" not in cmd else not same
    if cmd == "IfMoveEqualTo" and script.move_effect[script.moves.index(args[0])] != effect:
        return False
    if cmd == "IfCurrentMoveKills" and not damaging:
        return False
    if (
        cmd == "IfMoveEffectivenessEquals"
        and args[0] == "TYPE_MULTI_QUADRUPLE_DAMAGE"
        and not can_quad
    ):
        return False
    if cmd in ("IfLoadedEqualTo", "IfLoadedNotEqualTo"):
        equal = None
        if calc[0] == "compared" and args[0] == "AI_NO_COMPARISON_MADE":
            equal = not calc[1]
        elif calc[0] == "effect" and script.effect_id(args[0]) is not None:
            equal = script.effect_id(args[0]) == calc[1]
        if equal is None:
            return None
        return equal if cmd == "IfLoadedEqualTo" else not equal
    if cmd in ("IfLoadedInTable", "IfLoadedNotInTable") and calc[0] == "effect":
        inside = calc[1] in script.table(args[0])
        return inside if cmd == "IfLoadedInTable" else not inside
    return None


def read_bounds(script: Script, effects) -> dict[tuple[int, bool, bool], tuple[int, ...]]:
    return {
        (e, damaging, can_quad): tuple(
            flag_reads(script, bit, e, damaging, can_quad) for bit in FLAG_LABELS
        )
        for e in sorted(effects)
        for damaging in (False, True)
        for can_quad in (False, True)
    }


def render(bounds) -> str:
    rows = "\n".join(f"    {key!r}: {value!r}," for key, value in bounds.items())
    return f'''"""Generated by tools/ai_read_bounds.py (E1 Ruling R6); do not edit by hand.

The most random reads (``IfRandomLessThan`` and defender-ability guesses) each transcribed
trainer-AI flag script makes on one move slot's path, per kind of move:
``READ_BOUNDS[(battle effect, damaging, can be quadruple effective)]`` holds one count per
bit of ``READ_BOUND_BITS``, in that order. ``layout.fight_slot_reads`` sums the counts of a
fight's flag set and takes the most over its opponents' moves.
"""

READ_BOUND_BITS = {tuple(FLAG_LABELS)!r}

READ_BOUNDS: dict[tuple[int, bool, bool], tuple[int, ...]] = {{
{rows}
}}
'''


def main() -> int:
    sys.setrecursionlimit(100000)
    import numpy as np

    from nuzline.ai.outcome import KAIZO_TRANSCRIBED
    from nuzline.data.tables import load_tables

    tables = load_tables(ROOT / "data/gen4_platinum_kaizo")
    script = Script(DECOMP, np.asarray(tables.move_ai_effect))
    OUT.write_text(render(read_bounds(script, KAIZO_TRANSCRIBED)), "utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
