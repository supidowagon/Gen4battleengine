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
* ``IfTargetIsPartner`` never jumps in singles;
* ``IfHPPercentLessThan`` / ``GreaterThan`` / ``EqualTo`` / ``NotEqualTo`` (trainer_ai.c:588-646)
  compare a battler's ``curHP * 100 / maxHP``, which no AI command changes, so the path keeps
  each battler argument's percent range (0-100) and a compare the range already fixes is
  decided (Expert_StatusEvasionUp's :2204 needs 90 % or more, its :2215 50 % or less: never
  both on one path).

Every other branch is taken both ways, so the table is an upper bound: a path never makes more
reads than its row says (a layout bound that is never exceeded is never a refusal).

The damage commands make one read each where ``TrainerAI_CalcDamage`` draws (E1 ruling G-39:
Psywave's and Magnitude's id cases, trainer_ai.c:3014-3048; ``ai.outcome`` docstring), so the
bound also follows the moveset (``drawn``): ``IfBattlerDealsMoreDamage`` (:2187-2240) always
(the player's last move may be one; the layout does not know the player's moves);
``IfPartyMemberDealsMoreDamage`` (:2106-2174) when the fight's moves include a drawing one;
``IfCurrentMoveKills`` (:1525-1567) when the scored move is one (and damaging);
``FlagMoveDamageScore`` (:1012-1063) when the fight has one, the move is damaging and a later
compare reads the result as other than ``AI_NO_COMPARISON_MADE`` (counted at that compare;
script.s:62, :112, :6511 and :6572 compare only with it and are no read). ``READ_BOUNDS`` holds
the fights without a drawing move, ``DRAWN_READ_BOUNDS`` (keyed by ``current``, the scored move
draws, too) the rows that differ in a fight with one.

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
# layout.KAIZO_FLAG_BITS), with their entry labels (script.s:19-29).
FLAG_LABELS = {
    0: "Basic_Main",
    1: "EvalAttack_Main",
    2: "Expert_Main",
    3: "SetupFirstTurn_Main",
    4: "Risky_Main",
    5: "PrioritizeExtremes_Main",
    6: "BatonPass_Main",
    8: "CheckHP_Main",
    9: "Weather_Main",
    10: "Harrassment_Main",
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


def flag_reads(
    script: Script, bit: int, effect: int, damaging: bool, can_quad: bool, drawn: int = 0
) -> int:
    """The most reads flag ``bit``'s script makes on one path for a move of ``effect``;
    ``drawn``: 0 no move of the fight draws in ``TrainerAI_CalcDamage``, 1 some does, 2 the
    scored move does (module docstring)."""
    lines = script.lines

    @cache
    def most(i: int, calc: tuple, hp: tuple = ()) -> int:
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
            elif cmd == "IfBattlerDealsMoreDamage":
                reads = 1
            elif cmd == "IfPartyMemberDealsMoreDamage":
                reads = int(drawn >= 1)
            elif cmd in ("IfCurrentMoveKills", "IfCurrentMoveDoesNotKill"):
                reads = int(drawn == 2 and damaging)
            target = args[-1] if args and args[-1] in script.labels else None
            if cmd == "GoTo":
                return reads + most(script.labels[target], calc, hp)
            taken = _decided(script, cmd, args, calc, effect, damaging, can_quad)
            hp_jump = hp_stay = hp
            if cmd in _HP_COMPARES and target is not None:
                taken, hp_jump, hp_stay = _hp_branch(cmd, args, hp)
            nxt = _next_calc(cmd, calc, effect, damaging, drawn)
            if calc[0] == "compared" and calc[2] and cmd.startswith("IfLoaded") and taken is None:
                reads += 1  # FlagMoveDamageScore's pending read: its result is read here
                nxt = (calc[0], calc[1], False)
            if target is None or not cmd.startswith("If"):
                if reads:
                    return reads + most(i + 1, nxt, hp)
                calc = nxt
                i += 1
                continue
            if taken is True:
                return reads + most(script.labels[target], nxt, hp_jump)
            if taken is False:
                if reads:
                    return reads + most(i + 1, nxt, hp_stay)
                calc, hp = nxt, hp_stay
                i += 1
                continue
            return reads + max(most(script.labels[target], nxt, hp_jump), most(i + 1, nxt, hp_stay))

    return most(script.labels[FLAG_LABELS[bit]] + 1, ("unknown",))


_HP_COMPARES = {
    "IfHPPercentLessThan",
    "IfHPPercentGreaterThan",
    "IfHPPercentEqualTo",
    "IfHPPercentNotEqualTo",
}


def _hp_branch(cmd: str, args, hp: tuple) -> tuple[bool | None, tuple, tuple]:
    """An HP-percent compare on the path's ranges ``hp`` (sorted ``(battler, lo, hi)``; a
    battler not in it has 0-100): whether it jumps (None when both ways are open) and the
    ranges after the jump and after falling through (trainer_ai.c:588-646)."""
    battler, value = args[0], args[1]
    if not value.isdigit():
        return None, hp, hp
    v = int(value)
    ranges = {b: (lo, hi) for b, lo, hi in hp}
    lo, hi = ranges.get(battler, (0, 100))

    def narrowed(*keep):
        # keep: the percents the branch allows, as (lo, hi) pieces; the range is their hull.
        pieces = [(max(lo, a), min(hi, b)) for a, b in keep]
        pieces = [(a, b) for a, b in pieces if a <= b]
        if not pieces:
            return None
        out = dict(ranges)
        out[battler] = (min(a for a, _ in pieces), max(b for _, b in pieces))
        return tuple(sorted((b, a, c) for b, (a, c) in out.items()))

    if cmd == "IfHPPercentLessThan":
        jump, stay = narrowed((0, v - 1)), narrowed((v, 100))
    elif cmd == "IfHPPercentGreaterThan":
        jump, stay = narrowed((v + 1, 100)), narrowed((0, v))
    elif cmd == "IfHPPercentEqualTo":
        jump, stay = narrowed((v, v)), narrowed((0, v - 1), (v + 1, 100))
    else:
        jump, stay = narrowed((0, v - 1), (v + 1, 100)), narrowed((v, v))
    if jump is None:
        return False, hp, stay
    if stay is None:
        return True, jump, hp
    return None, jump, stay


def _next_calc(cmd: str, calc: tuple, effect: int, damaging: bool, drawn: int = 0) -> tuple:
    if cmd == "LoadCurrentMoveEffect":
        return ("effect", effect)
    if cmd == "FlagMoveDamageScore":
        return ("compared", damaging, drawn >= 1 and damaging)
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


def drawn_read_bounds(
    script: Script, effects, base
) -> dict[tuple[int, bool, bool, bool], tuple[int, ...]]:
    """The rows of a fight with a drawing move (``flag_reads``' ``drawn`` 1, or 2 for
    ``current``) that differ from ``base`` (``read_bounds``)."""
    out = {}
    for e in sorted(effects):
        for damaging in (False, True):
            for can_quad in (False, True):
                for current in (False, True):
                    row = tuple(
                        flag_reads(script, bit, e, damaging, can_quad, 2 if current else 1)
                        for bit in FLAG_LABELS
                    )
                    if row != base[(e, damaging, can_quad)]:
                        out[(e, damaging, can_quad, current)] = row
    return out


def render(bounds, drawn=None) -> str:
    rows = "\n".join(f"    {key!r}: {value!r}," for key, value in bounds.items())
    drawn_rows = "\n".join(f"    {key!r}: {value!r}," for key, value in (drawn or {}).items())
    return f'''"""Generated by tools/ai_read_bounds.py (E1 Ruling R6); do not edit by hand.

The most random reads (``IfRandomLessThan`` and defender-ability guesses) each transcribed
trainer-AI flag script makes on one move slot's path, per kind of move:
``READ_BOUNDS[(battle effect, damaging, can be quadruple effective)]`` holds one count per
bit of ``READ_BOUND_BITS``, in that order. ``layout.fight_slot_reads`` sums the counts of a
fight's flag set and takes the most over its opponents' moves.

A damage command whose estimate draws (Psywave's and Magnitude's id cases, E1 ruling G-39) is
one read too: ``READ_BOUNDS`` is for a fight none of whose moves draws (its
``IfBattlerDealsMoreDamage`` still counts one: the player's move may draw), and in a fight with
such a move ``DRAWN_READ_BOUNDS[(battle effect, damaging, can be quadruple effective, the move
itself draws)]`` replaces the rows that differ.
"""

READ_BOUND_BITS = {tuple(FLAG_LABELS)!r}

READ_BOUNDS: dict[tuple[int, bool, bool], tuple[int, ...]] = {{
{rows}
}}

DRAWN_READ_BOUNDS: dict[tuple[int, bool, bool, bool], tuple[int, ...]] = {{
{drawn_rows}
}}
'''


def main() -> int:
    sys.setrecursionlimit(100000)
    import numpy as np

    from nuzline.ai.outcome import KAIZO_TRANSCRIBED
    from nuzline.data.tables import load_tables

    tables = load_tables(ROOT / "data/gen4_platinum_kaizo")
    script = Script(DECOMP, np.asarray(tables.move_ai_effect))
    base = read_bounds(script, KAIZO_TRANSCRIBED)
    drawn = drawn_read_bounds(script, KAIZO_TRANSCRIBED, base)
    OUT.write_text(render(base, drawn), "utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
