"""The per-game trainer-AI policy: what differs between the vanilla and the Kaizo AI.

The Kaizo AI script is the vanilla ``gTrainerAITable`` with one changed word, and the AI's
C code is unchanged (``docs/ai/kaizo_roark.md`` 2). So both games share the transcription
in ``nuzline.ai`` and differ only where an ``AiLayout`` says so. A game's ``Tables`` carry
the policy index ``ai_policy`` (``game_constants.AI_POLICIES`` order); ``layout_of`` reads
it on concrete tables, before any trace, and the AI's traced functions take the layout as a
static argument.

The fields:

* ``name``: the policy, ``game_constants.POLICY``'s value for the game.
* ``per_slot_sites``: the script-roll sites are keyed by move slot (K3c Task 3 reads it).
* ``max_script_rolls``: the flat ``AiOutcome.script_rolls`` width (vanilla).
* ``slot_sites``: the script-roll sites of one move slot (per-slot layouts).
* ``switch_sites``: the voluntary-switch draw sites (``switching.SWITCH_SITES``).
* ``risky``: the policy runs ``AI_FLAG_RISKY`` (script.s:6521-6531) after EXPERT.
* ``water2_ability``: the ability ``Basic_CheckForImmunity``'s last compare tests
  (script.s:78): Levitate (26) in vanilla, a second Levitate test that never matches (the
  decomp's bug, since :77 already takes Levitate), and Dry Skin (87) in Kaizo (overlay14
  word 0x022249CC, ``data/kaizo/rom_diff.toml``).
* ``transcribed``: the opponent moves whose script paths are transcribed. Vanilla keys
  them by decomp move id (``commands.MOVE_EFFECTS``); Kaizo by battle effect, the one the
  scripts read (``tables.move_ai_effect``; ``IfCurrentMoveEffectEqualTo``, trainer_ai.c:1441).
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import jax

from nuzline.ai.commands import MOVE_EFFECTS
from nuzline.ai.outcome import MAX_EXPERT_SITES, MAX_SCRIPT_ROLLS
from nuzline.ai.switching import SWITCH_SITES
from nuzline.data.game_constants import AI_POLICIES
from nuzline.data.tables import Tables

__all__ = ["KAIZO", "VANILLA", "AiLayout", "layout_of"]


class AiLayout(NamedTuple):
    name: str
    per_slot_sites: bool
    max_script_rolls: int
    slot_sites: int
    switch_sites: int
    risky: bool
    water2_ability: int
    transcribed: frozenset[int]


VANILLA = AiLayout(
    name="vanilla",
    per_slot_sites=False,
    max_script_rolls=MAX_SCRIPT_ROLLS,
    slot_sites=0,  # the vanilla layout is flat
    switch_sites=SWITCH_SITES,
    risky=False,  # Roark's ai_flags are 7 (docs/ai/roark.md 1)
    water2_ability=26,  # ABILITY_LEVITATE
    transcribed=frozenset(MOVE_EFFECTS),
)

# Kaizo Roark's ai_flags are 23, 0b10111: BASIC, EVAL_ATTACK, EXPERT and RISKY
# (data/intermediate_kaizo/trainers.json, trainer 246). Tasks 3, 6 and 7 of K3c fill in
# the per-slot sites, Risky_Main and the six-Pokemon switch sites; until then the counts
# are the vanilla transcription's (1 EVAL_ATTACK site and MAX_EXPERT_SITES EXPERT sites
# per move) and the transcribed effects are those of the vanilla-transcribed moves, none
# of which is in Risky_RiskyEffects (script.s:6536-6561), so RISKY scores nothing yet.
KAIZO = AiLayout(
    name="kaizo",
    per_slot_sites=True,
    max_script_rolls=MAX_SCRIPT_ROLLS,
    slot_sites=1 + MAX_EXPERT_SITES,
    switch_sites=SWITCH_SITES,
    risky=True,
    water2_ability=87,  # ABILITY_DRY_SKIN
    transcribed=frozenset(int(e) for e in MOVE_EFFECTS.values()),
)

LAYOUTS = (VANILLA, KAIZO)  # by Tables.ai_policy
assert tuple(layout.name for layout in LAYOUTS) == AI_POLICIES


def layout_of(tables: Tables) -> AiLayout:
    """The ``AiLayout`` of ``tables``' game (``tables.ai_policy``). Concrete tables only:
    under a trace (jit, vmap) the policy is not known, and this raises ``TypeError``
    naming the caller, which must resolve the layout before tracing and pass it down."""
    if isinstance(tables.ai_policy, jax.core.Tracer):
        caller = sys._getframe(1).f_code.co_qualname
        raise TypeError(
            f"layout_of(tables) called from {caller} on traced tables: resolve the AI "
            "layout on the concrete tables before tracing and pass it as a static argument"
        )
    return LAYOUTS[int(tables.ai_policy)]
