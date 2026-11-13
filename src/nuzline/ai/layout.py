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
* ``slot_sites``: the script-roll columns of one move slot (per-slot layouts), the most
  sites any transcribed effect reaches.
* ``switch_sites``: the voluntary-switch draw sites ``AiOutcome.switch_stop`` encodes:
  ``switching.SWITCH_SITES`` (9) for vanilla Roark's three Pokemon,
  ``switching.KAIZO_SWITCH_SITES`` (44, every candidate draw of a party of six) for Kaizo.
* ``risky``: the policy runs ``AI_FLAG_RISKY`` (``Risky_Main``, script.s:6521-6534) after
  EXPERT (``policy._flags``). A layout that transcribes an effect of ``Risky_RiskyEffects``
  (``commands.RISKY_EFFECTS``) must run it and give the effect the ``:6530`` site (checked
  at import), so such a move is never scored without the Risky roll.
* ``water2_ability``: the ability ``Basic_CheckForImmunity``'s last compare tests
  (script.s:78): Levitate (26) in vanilla, a second Levitate test that never matches (the
  decomp's bug, since :77 already takes Levitate), and Dry Skin (87) in Kaizo (overlay14
  word 0x022249CC, ``data/kaizo/rom_diff.toml``).
* ``transcribed``: the opponent moves whose script paths are transcribed. Vanilla keys
  them by decomp move id (``commands.MOVE_EFFECTS``); Kaizo by battle effect, the one the
  scripts read (``tables.move_ai_effect``; ``IfCurrentMoveEffectEqualTo``, trainer_ai.c:1441).
* ``effect_sites``: a per-slot layout's script-roll site table, ``((effect, ((line, N),
  ...)), ...)``: the ``script.s`` line and N of each ``IfRandomLessThan`` a move with that
  battle effect can reach, in script order (``outcome.slot_site_table``). Kaizo's is
  ``outcome.EFFECT_SITE_LINES``, and its ``transcribed`` set is that table's effects.
  Empty for the flat layout, whose sites are ``outcome._SITES`` by move id.
* ``expert_transcribed``: of the effects of ``outcome.KAIZO_EXPERT_LABELS`` (a transcribed
  effect whose ``Expert_Main`` entry, script.s:1628-1808, goes to a label added after the
  vanilla transcription), those whose Expert label is transcribed. A per-slot layout refuses
  an opponent move with any other of those effects (``policy._refused``), so a missing
  Expert path is refused, never scored. Kaizo's holds all nine since K3c Task 6.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import jax

from nuzline.ai.commands import MOVE_EFFECTS, RISKY_EFFECTS
from nuzline.ai.outcome import (
    EFFECT_SITE_LINES,
    KAIZO_EXPERT_TRANSCRIBED,
    KAIZO_SLOT_SITES,
    MAX_SCRIPT_ROLLS,
    SITE_N,
)
from nuzline.ai.switching import KAIZO_SWITCH_SITES, SWITCH_SITES
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
    effect_sites: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = ()
    expert_transcribed: frozenset[int] = frozenset()


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
# (data/intermediate_kaizo/trainers.json, trainer 246); the policy runs exactly these four
# (plan K3c Ruling 10). The script-roll sites are keyed by move slot (K3c Task 3,
# docs/ai/kaizo_roark.md 8) and by battle effect. Task 5 added Roark's effects to
# EFFECT_SITE_LINES with their BASIC and EVAL_ATTACK paths, Task 6 their EXPERT paths
# (expert_transcribed) and RISKY, and Task 7 the six-Pokemon switch sites.
KAIZO = AiLayout(
    name="kaizo",
    per_slot_sites=True,
    max_script_rolls=MAX_SCRIPT_ROLLS,  # unused: the per-slot layout is not flat
    slot_sites=KAIZO_SLOT_SITES,
    # 4 keep draws + 2 passes x 5 bench Pokemon x 4 moves (trainer_ai.c:3576-3592,
    # :3756-3804; docs/ai/kaizo_roark.md 11): a party of six never has more.
    switch_sites=KAIZO_SWITCH_SITES,
    risky=True,
    water2_ability=87,  # ABILITY_DRY_SKIN
    transcribed=frozenset(EFFECT_SITE_LINES),
    effect_sites=tuple(
        (effect, tuple((line, SITE_N[line]) for line in lines))
        for effect, lines in sorted(EFFECT_SITE_LINES.items())
    ),
    expert_transcribed=KAIZO_EXPERT_TRANSCRIBED,
)
assert max(len(sites) for _, sites in KAIZO.effect_sites) <= KAIZO.slot_sites

LAYOUTS = (VANILLA, KAIZO)  # by Tables.ai_policy
assert tuple(layout.name for layout in LAYOUTS) == AI_POLICIES


def _check_risky() -> None:
    """Risky_Main (script.s:6526-6531) rolls :6530 for exactly the effects of
    Risky_RiskyEffects. A layout keyed by effect gives each such effect that site and runs
    RISKY; the vanilla layout transcribes none of them (by move id, commands.MOVE_EFFECTS)."""
    for layout in LAYOUTS:
        for effect, sites in layout.effect_sites:
            risky = effect in RISKY_EFFECTS
            assert ((6530, 128) in sites) == risky, (layout.name, effect)
            assert layout.risky or not risky, (layout.name, effect)
    assert not set(MOVE_EFFECTS.values()) & set(RISKY_EFFECTS)


_check_risky()


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
