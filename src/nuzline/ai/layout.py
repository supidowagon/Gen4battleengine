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
  (``generated/ai_flags.txt``: bit 0 BASIC, 1 EVAL_ATTACK, 2 EXPERT, 4 RISKY).
* ``per_slot_sites``: the random reads are keyed by move slot (K3c Task 3) and by their
  order on the slot's path (E1 Ruling R4; ``outcome`` docstring).
* ``max_script_rolls``: the flat ``AiOutcome.script_rolls`` width (vanilla).
* ``slot_reads``: the draws of one move slot (per-slot layouts): the most random reads
  (``IfRandomLessThan`` and guessing ``LoadBattlerAbility DEFENDER``) one move's scoring path
  makes (``outcome.KAIZO_SLOT_READS``). A path that makes more is refused
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
from nuzline.ai.switching import KAIZO_SWITCH_SITES, SWITCH_SITES
from nuzline.data.game_constants import AI_POLICIES
from nuzline.data.tables import AI_FLAGS_UNSET, Tables

__all__ = [
    "KAIZO",
    "KAIZO_FLAG_BITS",
    "VANILLA",
    "AiLayout",
    "fight_tables",
    "kaizo_layout",
    "layout_of",
]

# AI flag bits (generated/ai_flags.txt, line n + 2 is bit n; FlagTable, script.s:18-30).
BASIC, EVAL_ATTACK, EXPERT, RISKY = 0, 1, 2, 4
# The flag bits whose scripts the Kaizo policy transcribes (policy._flags):
# engine/coverage.py IMPLEMENTED_AI_FLAGS["kaizo"] lists the same bits.
KAIZO_FLAG_BITS = frozenset({BASIC, EVAL_ATTACK, EXPERT, RISKY})


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
        """The layout's name in a program cache key: the policy and the flag set."""
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


@cache
def kaizo_layout(flags: int) -> AiLayout:
    """The Kaizo layout that runs the trainer AI flags ``flags`` (E1 Ruling R2): ``KAIZO`` with
    its flag set replaced. ``ValueError`` for a set with a bit outside ``KAIZO_FLAG_BITS``."""
    flags = int(flags)
    bits = {b for b in range(max(flags.bit_length(), 1)) if flags >> b & 1}
    if flags < 0 or bits - KAIZO_FLAG_BITS:
        raise ValueError(
            f"AI flags {flags:#x}: the Kaizo policy transcribes only the flag bits "
            f"{sorted(KAIZO_FLAG_BITS)} (engine/coverage.py refuses the trainer)"
        )
    return KAIZO if flags == KAIZO.flags else KAIZO._replace(flags=flags)


def layout_of(tables: Tables) -> AiLayout:
    """The ``AiLayout`` of ``tables``' game (``tables.ai_policy``) and, for Kaizo, of the
    fight's trainer AI flags (``tables.ai_flags``, ``kaizo_layout``; ``ValueError`` while they
    are ``AI_FLAGS_UNSET``: the fight's entry point sets them,
    ``tables._replace(ai_flags=trainer.ai_flags)``). Concrete tables only: under a trace
    (jit, vmap) the policy is not known, and this raises ``TypeError`` naming the caller,
    which must resolve the layout before tracing and pass it down."""
    if isinstance(tables.ai_policy, jax.core.Tracer) or isinstance(
        tables.ai_flags, jax.core.Tracer
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
    if flags == int(AI_FLAGS_UNSET):
        raise ValueError(
            "the fight's trainer AI flags are not set on the tables: "
            "tables._replace(ai_flags=trainer.ai_flags) (E1 Ruling R2)"
        )
    return kaizo_layout(flags)


def fight_tables(tables: Tables, trainer) -> Tables:
    """``tables`` for a fight against ``trainer`` (a ``data.trainer.Trainer``): its AI flags
    set (``Tables.ai_flags``), so ``layout_of`` gives its flag set's layout (E1 Ruling R2).
    The fight's entry points call it once (``cli``)."""
    return tables._replace(ai_flags=np.int32(trainer.ai_flags))
