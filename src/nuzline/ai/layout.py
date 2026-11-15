"""The per-game trainer-AI policy: what differs between the vanilla and the Kaizo AI.

The Kaizo AI script is the vanilla ``gTrainerAITable`` with one changed word, and the AI's
C code is unchanged (``docs/ai/kaizo_roark.md`` 2). So both games share the transcription
in ``nuzline.ai`` and differ only where an ``AiLayout`` says so. A game's ``Tables`` carry
the policy index ``ai_policy`` (``game_constants.AI_POLICIES`` order) and, set per fight, the
trainer's AI flags ``ai_flags``; ``layout_of`` reads them on concrete tables, before any
trace, and the AI's traced functions take the layout as a static argument.

**Flag sets (E1 Ruling R2).** ``TrainerAI_Init`` copies the trainer's ``aiMask``
(trainer_ai.c:249), and ``TrainerAI_MainSingles`` runs the scripts of its set bits in bit
order, each over all four move slots, and no other (:297-310; ``FlagTable``,
script.s:18-30). The Kaizo policy has one ``AiLayout`` per flag set (``kaizo_layout``):
``flags`` is static, so each fight's programs trace only its own flags' scripts, and the
search's program caches key on ``AiLayout.key``. A set with a bit the policy does not
transcribe (``KAIZO_FLAG_BITS``) has no layout (coverage refuses its trainer by flag). The
vanilla policy keeps its one set, BASIC, EVAL_ATTACK and EXPERT (vanilla Roark's 7,
docs/ai/roark.md 1), whatever ``ai_flags`` says (coverage refuses a vanilla trainer without
all three, or with any other).

The fields:

* ``name``: the policy, ``game_constants.POLICY``'s value for the game.
* ``flags``: the trainer AI flags the policy runs (``policy._flags``), in bit order
  (``generated/ai_flags.txt``: bit 0 BASIC, 1 EVAL_ATTACK, 2 EXPERT, 3 SETUP_FIRST_TURN,
  4 RISKY, 5 PRIORITIZE_EXTREMES, 8 CHECK_HP, 9 WEATHER).
* ``per_slot_sites``: the random reads are keyed by move slot (K3c Task 3) and by their
  order on the slot's path (E1 Ruling R4; ``outcome`` docstring).
* ``max_script_rolls``: the flat ``AiOutcome.script_rolls`` width (vanilla).
* ``slot_reads``: the draws of one move slot (per-slot layouts): the most random reads
  (``IfRandomLessThan`` and guessing ``LoadBattlerAbility DEFENDER``) one move's scoring path
  makes for the fight's moves under its flags (``fight_slot_reads``, E1 Ruling R6). A path that
  makes more is refused
  (``AiContext.missing_read``). 0 for the flat layout.
* ``switch_sites``: the voluntary-switch draw sites ``AiOutcome.switch_stop`` encodes:
  ``switching.SWITCH_SITES`` (9) for vanilla Roark's three Pokemon,
  ``switching.KAIZO_SWITCH_SITES`` (44, every candidate draw of a party of six) for Kaizo.
* ``water2_ability``: the ability ``Basic_CheckForImmunity``'s last compare tests
  (script.s:78): Levitate (26) in vanilla, a second Levitate test that never matches (the
  decomp's bug, since :77 already takes Levitate), and Dry Skin (87) in Kaizo (overlay14
  word 0x022249CC, ``data/kaizo/rom_diff.toml``).
* ``transcribed``: the opponent moves whose script paths are transcribed. Vanilla keys
  them by decomp move id (``commands.MOVE_EFFECTS``); Kaizo by battle effect, the one the
  scripts read (``tables.move_ai_effect``; ``IfCurrentMoveEffectEqualTo``, trainer_ai.c:1441):
  ``outcome.KAIZO_TRANSCRIBED``.
* ``expert_transcribed``: of the effects of ``outcome.KAIZO_EXPERT_LABELS`` (a transcribed
  effect whose ``Expert_Main`` entry, script.s:1628-1808, goes to a label added after the
  vanilla transcription), those whose Expert label is transcribed. A per-slot layout refuses
  an opponent move with any other of those effects (``policy._refused``), so a missing
  Expert path is refused, never scored. Kaizo's holds all nine since K3c Task 6.
"""

from __future__ import annotations

import hashlib
import sys
from functools import cache
from typing import NamedTuple

import jax
import numpy as np

from nuzline.ai.commands import MOVE_EFFECTS
from nuzline.ai.outcome import (
    KAIZO_EXPERT_TRANSCRIBED,
    KAIZO_SLOT_READS,
    KAIZO_TRANSCRIBED,
    MAX_SCRIPT_ROLLS,
)
from nuzline.ai.read_bounds import READ_BOUND_BITS, READ_BOUNDS
from nuzline.ai.switching import KAIZO_SWITCH_SITES, SWITCH_SITES
from nuzline.data.game_constants import AI_POLICIES
from nuzline.data.tables import (
    AI_FLAGS_UNSET,
    AI_MOVES_ROWS,
    BATTLE_TYPE_TRAINER,
    BATTLE_TYPES,
    Tables,
)
from nuzline.engine.enumerate import WIDTH_CAP

__all__ = [
    "KAIZO",
    "KAIZO_FLAG_BITS",
    "VANILLA",
    "AiLayout",
    "fight_effects",
    "fight_layout",
    "fight_slot_reads",
    "fight_tables",
    "kaizo_layout",
    "layout_of",
    "move_read_rows",
]

# AI flag bits (generated/ai_flags.txt, line n + 2 is bit n; FlagTable, script.s:18-30).
BASIC, EVAL_ATTACK, EXPERT, SETUP_FIRST_TURN, RISKY = 0, 1, 2, 3, 4
PRIORITIZE_EXTREMES, BATON_PASS, CHECK_HP, WEATHER = 5, 6, 8, 9
# The flag bits whose scripts the Kaizo policy transcribes (policy._flags):
# engine/coverage.py IMPLEMENTED_AI_FLAGS["kaizo"] lists the same bits.
KAIZO_FLAG_BITS = frozenset(
    {
        BASIC,
        EVAL_ATTACK,
        EXPERT,
        SETUP_FIRST_TURN,
        RISKY,
        PRIORITIZE_EXTREMES,
        BATON_PASS,
        CHECK_HP,
        WEATHER,
    }
)
assert set(READ_BOUND_BITS) == KAIZO_FLAG_BITS


class AiLayout(NamedTuple):
    name: str
    per_slot_sites: bool
    max_script_rolls: int
    slot_reads: int
    switch_sites: int
    flags: int
    water2_ability: int
    transcribed: frozenset[int]
    expert_transcribed: frozenset[int] = frozenset()

    def runs(self, bit: int) -> bool:
        """The policy runs flag ``bit``'s script."""
        return bool(self.flags >> bit & 1)

    @property
    def key(self) -> str:
        """The layout's name in a program cache key: the policy, the flag set and, for a
        per-slot layout, the read bound (its draws' shape) and, when it is a fight's own
        (E1 Ruling R7), a digest of its transcribed effects."""
        if self.per_slot_sites:
            key = f"{self.name}-{self.flags:#x}-r{self.slot_reads}"
            if self.transcribed != KAIZO_TRANSCRIBED:
                text = ",".join(str(e) for e in sorted(self.transcribed))
                key += "-e" + hashlib.sha1(text.encode()).hexdigest()[:10]
            return key
        return f"{self.name}-{self.flags:#x}"


VANILLA = AiLayout(
    name="vanilla",
    per_slot_sites=False,
    max_script_rolls=MAX_SCRIPT_ROLLS,
    slot_reads=0,  # the vanilla layout is flat
    switch_sites=SWITCH_SITES,
    flags=0b111,  # BASIC, EVAL_ATTACK, EXPERT: vanilla Roark's ai_flags 7 (roark.md 1)
    water2_ability=26,  # ABILITY_LEVITATE
    transcribed=frozenset(MOVE_EFFECTS),
)

# The Kaizo layout of Kaizo Roark's ai_flags, 23 = 0b10111: BASIC, EVAL_ATTACK, EXPERT and
# RISKY (data/intermediate_kaizo/trainers.json, trainer 246); kaizo_layout gives the other
# flag sets (E1 Ruling R2). The random reads are keyed by move slot (K3c Task 3,
# docs/ai/kaizo_roark.md 8) and by read order on the path (E1 Ruling R4). Task 5 added
# Roark's effects to KAIZO_TRANSCRIBED with their BASIC and EVAL_ATTACK paths, Task 6 their
# EXPERT paths (expert_transcribed) and RISKY, and Task 7 the six-Pokemon switch sites.
KAIZO = AiLayout(
    name="kaizo",
    per_slot_sites=True,
    max_script_rolls=MAX_SCRIPT_ROLLS,  # unused: the per-slot layout is not flat
    slot_reads=KAIZO_SLOT_READS,
    # 4 keep draws + 2 passes x 5 bench Pokemon x 4 moves (trainer_ai.c:3576-3592,
    # :3756-3804; docs/ai/kaizo_roark.md 11): a party of six never has more.
    switch_sites=KAIZO_SWITCH_SITES,
    flags=0b10111,
    water2_ability=87,  # ABILITY_DRY_SKIN
    transcribed=KAIZO_TRANSCRIBED,
    expert_transcribed=KAIZO_EXPERT_TRANSCRIBED,
)

LAYOUTS = (VANILLA, KAIZO)  # by Tables.ai_policy (KAIZO: Roark's flag set)
assert tuple(layout.name for layout in LAYOUTS) == AI_POLICIES


def kaizo_layout(flags: int, slot_reads: int, effects=None) -> AiLayout:
    """The Kaizo layout that runs the trainer AI flags ``flags`` (E1 Ruling R2) with the read
    bound ``slot_reads`` (E1 Ruling R6; ``fight_slot_reads``) and, when ``effects`` (battle
    effects) is given, only those of the transcribed effects (E1 Ruling R7: a fight's
    programs trace the labels of its own moves only). ``ValueError`` for a set with a bit
    outside ``KAIZO_FLAG_BITS``, or a bound whose scoring width (``switch_sites + 1 + 4 *
    2**slot_reads``, ``policy.ai_width``) passes ``WIDTH_CAP``."""
    effects = None if effects is None else frozenset(int(e) for e in effects)
    return _with(KAIZO, int(flags), int(slot_reads), effects)


@cache
def _with(layout: AiLayout, flags: int, slot_reads: int, effects=None) -> AiLayout:
    """``layout`` (a per-slot policy's) running ``flags`` with ``slot_reads`` reads a slot and
    the transcribed effects among ``effects`` (all of them when None)."""
    bits = {b for b in range(max(flags.bit_length(), 1)) if flags >> b & 1}
    if flags < 0 or bits - KAIZO_FLAG_BITS:
        raise ValueError(
            f"AI flags {flags:#x}: the Kaizo policy transcribes only the flag bits "
            f"{sorted(KAIZO_FLAG_BITS)} (engine/coverage.py refuses the trainer)"
        )
    if layout.switch_sites + 1 + 4 * 2**slot_reads > WIDTH_CAP:
        raise ValueError(
            f"AI read bound {slot_reads}: {4 * 2**slot_reads} scorings per decision pass "
            f"WIDTH_CAP {WIDTH_CAP} (E1 Ruling R6)"
        )
    transcribed = layout.transcribed if effects is None else layout.transcribed & effects
    expert = layout.expert_transcribed if effects is None else layout.expert_transcribed & effects
    out = layout._replace(
        flags=flags, slot_reads=slot_reads, transcribed=transcribed, expert_transcribed=expert
    )
    return layout if out == layout else out


def move_read_rows(tables: Tables, moves) -> np.ndarray:
    """``int8[n, 8]``: per move of ``moves`` (move ids, 0 or less for none), the most
    random reads each transcribed flag script makes on its path (``read_bounds.READ_BOUNDS``,
    by its battle effect, whether it is damaging, and whether its type can be quadruple
    effective; E1 Ruling R6). A move whose effect is not transcribed is refused
    (``policy._refused``), and its row is 0."""
    effect = np.asarray(tables.move_ai_effect)
    power = np.asarray(tables.move_power)
    move_type = np.asarray(tables.move_type)
    chart = np.asarray(tables.type_chart)
    from nuzline.ai.commands import ALT_POWER_EFFECTS, NO_DAMAGE_CALC_EFFECTS

    moves = [int(m) for m in np.asarray(moves).ravel() if int(m) > 0]
    rows = np.zeros((max(len(moves), 1), len(READ_BOUND_BITS)), np.int8)
    for i, move in enumerate(moves):
        e = int(effect[move])
        # FlagMoveDamageScore's test (trainer_ai.c:1035-1036), as commands.is_damaging:
        # an alt-power effect whatever its power, else power above 1 and a calc effect.
        damaging = e in ALT_POWER_EFFECTS or (
            bool(power[move] > 1) and e not in NO_DAMAGE_CALC_EFFECTS
        )
        # QUADRUPLE needs two distinct
        # defender types each taking the move super effectively (battle_lib.c:2611-2646), while
        # an AI Adaptability Pokemon stays refused (its STAB of 2, :2592-2594, makes one
        # super-effective type QUADRUPLE: then this must allow one).
        can_quad = int((chart[int(move_type[move])] == 20).sum()) >= 2
        reads = READ_BOUNDS.get((e, damaging, can_quad))
        if reads is not None:
            rows[i] = reads
    return rows


def fight_slot_reads(rows, flags: int) -> int:
    """The read bound of a fight (E1 Ruling R6): the most, over its opponents' moves
    (``move_read_rows``), of the reads the flag scripts of ``flags`` make on one path; at
    least 1."""
    rows = np.asarray(rows, np.int32)
    columns = [i for i, bit in enumerate(READ_BOUND_BITS) if flags >> bit & 1]
    return max(1, int(rows[:, columns].sum(axis=1).max()))


def fight_effects(tables: Tables, moves) -> frozenset[int]:
    """The battle effects (``Tables.move_ai_effect``) of ``moves`` (move ids, 0 or less for
    none): a fight's own (E1 Ruling R7). It assumes the AI's moves are its party's: when
    Transform or Mimic (effects 57, 82) are modelled, a move they copy is scored by its effect's
    label, so the layout must add the effects those can copy (until then coverage refuses
    them)."""
    effect = np.asarray(tables.move_ai_effect)
    return frozenset(int(effect[int(m)]) for m in np.asarray(moves).ravel() if int(m) > 0)


def fight_layout(tables: Tables, flags: int, moves) -> AiLayout:
    """The Kaizo layout of a fight with the flag set ``flags`` against opponents whose moves
    are ``moves``: their read bound (E1 Ruling R6) and their transcribed effects (R7)."""
    rows = move_read_rows(tables, moves)
    return kaizo_layout(flags, fight_slot_reads(rows, flags), fight_effects(tables, moves))


def layout_of(tables: Tables) -> AiLayout:
    """The ``AiLayout`` of ``tables``' game (``tables.ai_policy``) and, for Kaizo, of the
    fight: its trainer AI flags (``tables.ai_flags``) and its opponents' moves
    (``tables.ai_moves``, ``fight_layout``). ``ValueError`` while they are unset: the fight's
    entry point sets them (``fight_tables``). Concrete tables only: under a trace (jit, vmap)
    the policy is not known, and this raises ``TypeError`` naming the caller, which must
    resolve the layout before tracing and pass it as a static argument."""
    if any(
        isinstance(x, jax.core.Tracer) for x in (tables.ai_policy, tables.ai_flags, tables.ai_moves)
    ):
        caller = sys._getframe(1).f_code.co_qualname
        raise TypeError(
            f"layout_of(tables) called from {caller} on traced tables: resolve the AI "
            "layout on the concrete tables before tracing and pass it as a static argument"
        )
    layout = LAYOUTS[int(tables.ai_policy)]
    if not layout.per_slot_sites:
        return layout
    flags = int(tables.ai_flags)
    moves = np.asarray(tables.ai_moves)
    if flags == int(AI_FLAGS_UNSET) or (moves < 0).any():
        raise ValueError(
            "the fight's trainer AI flags or opponent moves are not set on the tables: "
            "ai.layout.fight_tables (E1 Rulings R2, R6)"
        )
    return fight_layout(tables, flags, moves)


def fight_tables(tables: Tables, trainer, party, flags: int | None = None) -> Tables:
    """``tables`` for a fight against ``trainer`` (a ``data.trainer.Trainer``) with its
    ``party`` (``PokemonSpec``s): its AI flags (``Tables.ai_flags``; ``flags`` overrides
    them) and its moves (``Tables.ai_moves``), so ``layout_of`` gives the fight's layout
    (E1 Rulings R2, R6, R7). The fight's entry points call it once (``cli``,
    ``engine.coverage``)."""
    moves = [m for spec in party for m in spec.moves]
    if len(moves) > AI_MOVES_ROWS:
        raise ValueError(f"{len(moves)} opponent moves: more than {AI_MOVES_ROWS}")
    padded = np.zeros(AI_MOVES_ROWS, np.int16)
    padded[: len(moves)] = moves
    return tables._replace(
        ai_flags=np.int32(trainer.ai_flags if flags is None else flags),
        ai_moves=padded,
        battle_type=BATTLE_TYPES.get(trainer.format, BATTLE_TYPE_TRAINER),
    )
