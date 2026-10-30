"""Roark's trainer AI: move scores, the chosen action and its exact distribution.

``docs/ai/roark.md`` is the specification. One decision runs as the decomp does
(roark.md 2): ``TrainerAI_ShouldSwitch`` (``maybe_switch``), then
``TrainerAI_ShouldUseItem`` (``maybe_use_item``), then ``TrainerAI_Main``: base scores,
the BASIC, EVAL_ATTACK and EXPERT flag scripts over move slots 0-3 (``ai_scores``), and
the highest score with ties broken by ``AiOutcome.tiebreak``. Every random draw is a
field of the ``AiOutcome`` input; nothing here draws randomness.

The scripts are transcribed label by label from ``src/battle/trainer_ai/script.s``
(paths relative to ``external/pokeplatinum``) as straight-line JAX over the commands of
``nuzline.ai.commands``: a jump splits ``AiContext.live`` into the fall-through path
and the taken path, and each label block runs under the union of the paths that reach
it. Only the branches that can be taken for the transcribed moves
(``commands.MOVE_EFFECTS``) are written out; the effect jump tables list just those
moves' effects (roark.md 5.4).

Refusals (``ai_refusals``): an opponent move outside ``MOVE_EFFECTS``, a moveset with
more script-roll sites than ``AiOutcome`` encodes, the abilities, items and moves that
change ``effectiveness_class``, a bag item other than a Potion, and every refusal of
the voluntary-switch rule (``switching.switching_refusals``). On concrete inputs the
public functions raise ``UnsupportedAi``; traced (jit, vmap) they cannot raise, so
``ai_scores`` gives -1 for every slot and ``ai_policy`` gives
``switching.REFUSED_ACTION``. Callers that trace (Stage A) run ``ai_refusals`` on the
fight first.

Knowledge (roark.md 9): the scripts read the player's moves only through the AI's seen
list (``BattleState.ai_seen_moves``, ``AiContext.seen_moves``: a move the player has not
used is not known) and the player's ability only through ``resolve_player_ability``
(``AiContext.player_ability``). ``step`` keeps both up to date.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai import commands as c
from nuzline.ai import switching
from nuzline.ai.commands import AiContext, BattleEffect, Battler, LoadType, TypeMulti
from nuzline.ai.outcome import (
    MAX_SCRIPT_ROLLS,
    SIM_ROLL_MAX,
    SWITCH_NONE,
    SWITCH_SITES,
    TIEBREAK_MOD,
    TIEBREAK_P,
    AiOutcome,
    enumerate_ai_outcomes,
    script_roll_count,
    script_roll_probs,
    switch_stop_probs,
)
from nuzline.ai.switching import (
    NO_SWITCH,
    REFUSED_ACTION,
    switch_action,
    switch_sites,
    switching_refusals,
)
from nuzline.data.tables import Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION, NUM_OPP_ACTIONS, usable_moves
from nuzline.engine.ids import POTION_HP_RESTORED, Stat, TrainerItemId
from nuzline.engine.state import BAG_SLOTS, MOVES, BattleState

__all__ = [
    "NO_ITEM",
    "UnsupportedAi",
    "ai_action_distribution",
    "ai_action_distribution_enumerated",
    "ai_policy",
    "ai_refusals",
    "ai_scores",
    "maybe_switch",
    "maybe_use_item",
]

NO_ITEM = -1  # maybe_use_item: the AI uses no item
_I32 = jnp.int32
_A, _D = Battler.ATTACKER, Battler.DEFENDER
_ABILITY = c.ABILITY_IDS
_MOVE = c.MOVE_IDS
_TYPE = c.TYPE_IDS


class UnsupportedAi(ValueError):
    """The trainer AI meets a mechanic this module does not transcribe."""


# --- Straight-line control flow ---------------------------------------------------


def _split(ctx: AiContext) -> tuple[AiContext, jnp.ndarray]:
    """After a jump command: the context on the fall-through path, and the mask of the
    path that jumps."""
    taken = ctx.live & ctx.jump
    return ctx._replace(live=ctx.live & ~ctx.jump), taken


def _at(ctx: AiContext, mask) -> AiContext:
    """Enter a label block reached by the paths in ``mask``."""
    return ctx._replace(live=jnp.asarray(mask))


def _score(ctx: AiContext, taken, value: int) -> AiContext:
    """A jump to ``ScoreMinusN`` / ``ScorePlusN`` (script.s:1567-1621): ``AddToMoveScore
    value`` and ``PopOrEnd`` on the taken path; the fall-through path is unchanged."""
    live = ctx.live
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, taken), value))
    return _at(ctx, live)


def _terminate(ctx: AiContext) -> AiContext:
    """``IfTargetIsPartner Terminate``: drop the (never taken) jump."""
    ctx, _ = _split(c.if_target_is_partner(ctx))
    return ctx


# --- The flag scripts --------------------------------------------------------------


def _basic(ctx: AiContext) -> AiContext:
    """``Basic_Main`` (script.s:52; roark.md 5.1)."""
    ctx = _terminate(ctx)
    ctx, ohko = _split(c.if_move_equal_to(ctx, _MOVE["Fissure"]))
    ctx, t = _split(c.if_move_equal_to(ctx, _MOVE["Horn Drill"]))
    ohko = ohko | t
    ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    ctx, soundproof = _split(c.if_loaded_equal_to(ctx, c.AI_NO_COMPARISON_MADE))

    # Basic_CheckForImmunity (:65)
    ctx = _at(ctx, ctx.live | ohko)
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _A)
    ctx, no_immunity = _split(c.if_loaded_equal_to(ctx, _ABILITY["Mold Breaker"]))
    ctx = c.load_battler_ability(ctx, _D)
    ctx, electric = _split(c.if_loaded_equal_to(ctx, _ABILITY["Volt Absorb"]))
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Motor Drive"]))
    electric = electric | t
    ctx, water = _split(c.if_loaded_equal_to(ctx, _ABILITY["Water Absorb"]))
    ctx, fire = _split(c.if_loaded_equal_to(ctx, _ABILITY["Flash Fire"]))
    ctx, wonder_guard = _split(c.if_loaded_equal_to(ctx, _ABILITY["Wonder Guard"]))
    ctx, ground = _split(c.if_loaded_equal_to(ctx, _ABILITY["Levitate"]))
    # (quirk) :78 should test Dry Skin; a second Levitate test can never match.
    ctx, water2 = _split(c.if_loaded_equal_to(ctx, _ABILITY["Levitate"]))
    ctx, t = _split(c.go_to(ctx))
    no_immunity = no_immunity | t

    # Basic_CheckElectricAbsorption (:81), _CheckWaterAbsorption (:86),
    # _CheckFireAbsorption (:91), _CheckGroundAbsorption (:101), _CheckWaterAbsorption2 (:106)
    for reached, type_name in (
        (electric, "Electric"),
        (water, "Water"),
        (fire, "Fire"),
        (ground, "Ground"),
        (water2, "Water"),
    ):
        ctx = c.load_type_from(_at(ctx, reached), LoadType.MOVE_TYPE)
        ctx, t = _split(c.if_temp_equal_to(ctx, _TYPE[type_name]))
        ctx = _score(ctx, t, -12)
        ctx, t = _split(c.go_to(ctx))
        no_immunity = no_immunity | t

    # Basic_CheckWonderGuard (:96)
    ctx = _at(ctx, wonder_guard)
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.DOUBLE))
    no_immunity = no_immunity | t
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.QUADRUPLE))
    no_immunity = no_immunity | t
    ctx, t = _split(c.go_to(ctx))
    ctx = _score(ctx, t, -12)

    # Basic_NoImmunityAbility (:111)
    ctx = _at(ctx, no_immunity)
    ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AI_NO_COMPARISON_MADE))
    soundproof = soundproof | t | ctx.live

    # Basic_CheckSoundproof (:115)
    ctx = c.load_battler_ability(_at(ctx, soundproof), _D)
    ctx, effect = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Soundproof"]))
    ctx = c.load_battler_ability(ctx, _A)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Mold Breaker"]))
    effect = effect | t
    for move in c.SOUND_MOVES:
        ctx, t = _split(c.if_move_equal_to(ctx, move))
        ctx = _score(ctx, t, -10)
    effect = effect | ctx.live

    # Basic_ScoreMoveEffect (:133): the jump-table entries of the transcribed effects.
    ctx = _at(ctx, effect)
    ctx, defense = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.DEF_DOWN))
    ctx, t = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.DEF_DOWN_2))
    defense = defense | t
    ctx, stealth_rock = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.STEALTH_ROCK))
    ctx = c.pop_or_end(ctx)  # :286

    # Basic_CheckLowStatStage_Defense (:434)
    ctx = _at(ctx, defense)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.DEF, 0))
    ctx = _score(ctx, t, -10)
    ctx, clear_body = _split(c.go_to(ctx))

    # Basic_CheckClearBodyEffect (:469)
    ctx = c.load_battler_ability(_at(ctx, clear_body), _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Clear Body"]))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["White Smoke"]))
    ctx = _score(ctx, t, -10)
    ctx = c.pop_or_end(ctx)

    # Basic_CheckStealthRock (:1540)
    ctx = _at(ctx, stealth_rock)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.STEALTH_ROCK))
    ctx = _score(ctx, t, -10)
    ctx = c.count_alive_party_battlers(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, 0))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _eval_attack(ctx: AiContext) -> AiContext:
    """``EvalAttack_Main`` (script.s:6353; roark.md 5.2). The Explosion, Focus Punch,
    Sucker Punch, Future Sight and priority-1 effect tests (:6364-6366, :6389-6399) match
    no transcribed move and are not written out."""
    ctx = _terminate(ctx)
    ctx, kill = _split(c.if_current_move_kills(ctx, c.USE_MAX_DAMAGE))
    ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AI_NOT_HIGHEST_DAMAGE))
    ctx = _score(ctx, t, -1)
    ctx, quad = _split(c.go_to(ctx))

    # EvalAttack_CheckQuadEffective (:6377)
    ctx = _at(ctx, quad)
    ctx, plus2 = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.QUADRUPLE))
    ctx = c.pop_or_end(ctx)

    # EvalAttack_TryScorePlus2 (:6382)
    ctx = _at(ctx, plus2)
    ctx, _ = _split(c.if_random_less_than(ctx, 80, ctx.sites[ctx.slot, 0]))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 2))

    # EvalAttack_ApplyKillBonuses (:6387) -> EvalAttack_ScorePlus4 (:6411)
    ctx = _at(ctx, kill)
    ctx, plus4 = _split(c.go_to(ctx))
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, plus4), 4))
    return ctx


def _expert(ctx: AiContext) -> AiContext:
    """``Expert_Main`` (script.s:1623; roark.md 5.3)."""
    ctx = _terminate(ctx)
    ctx, defense = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.DEF_DOWN))
    ctx, t = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.DEF_DOWN_2))
    defense = defense | t
    ctx, pursuit = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.HIT_BEFORE_SWITCH))
    ctx, stealth_rock = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.STEALTH_ROCK))
    ctx = c.pop_or_end(ctx)  # :1811
    site = ctx.sites[ctx.slot]

    # Expert_StatusDefenseDown (:2318)
    ctx = _at(ctx, defense)
    ctx, roll = _split(c.if_hp_percent_less_than(ctx, _A, 70))
    ctx, target_hp = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.DEF, 3))
    roll = roll | ctx.live
    # Expert_StatusDefenseDown_TryScoreMinus2 (:2327)
    ctx = _at(ctx, roll)
    ctx, t = _split(c.if_random_less_than(ctx, 50, site[1]))
    ctx = c.add_to_move_score(ctx, -2)
    target_hp = target_hp | t | ctx.live
    # Expert_StatusDefenseDown_CheckTargetHP (:2331)
    ctx = _at(ctx, target_hp)
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 70))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, -2))

    # Expert_Pursuit (:3706)
    ctx = c.load_is_first_turn_in_battle(_at(ctx, pursuit), _A)
    ctx, plus1 = _split(c.if_loaded_not_equal_to(ctx, 0))
    for target, type_name in (
        (LoadType.DEFENDER_TYPE_1, "Ghost"),
        (LoadType.DEFENDER_TYPE_1, "Psychic"),
        (LoadType.DEFENDER_TYPE_2, "Ghost"),
        (LoadType.DEFENDER_TYPE_2, "Psychic"),
    ):
        ctx = c.load_type_from(ctx, target)
        ctx, t = _split(c.if_loaded_equal_to(ctx, _TYPE[type_name]))
        plus1 = plus1 | t
    ctx, u_turn = _split(c.go_to(ctx))
    # Expert_Pursuit_TryScorePlus1 (:3725)
    ctx = _at(ctx, plus1)
    ctx, t = _split(c.if_random_less_than(ctx, 128, site[1]))
    ctx = c.add_to_move_score(ctx, 1)
    u_turn = u_turn | t | ctx.live
    # Expert_Pursuit_CheckUturn (:3729)
    ctx = _at(ctx, u_turn)
    ctx, _ = _split(c.if_move_not_known(ctx, _D, _MOVE["U-turn"]))
    ctx, _ = _split(c.if_random_less_than(ctx, 128, site[2]))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 1))

    # Expert_StealthRock (:6268). Expert_StealthRock_TryScorePlus1 (:6280, a 64 roll) is
    # reached only when the attacker knows Roar or Whirlwind, which are not transcribed
    # moves (ai_refusals), so the jumps below are never taken.
    ctx = _at(ctx, stealth_rock)
    ctx, _ = _split(c.if_random_less_than(ctx, 128, site[1]))
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.if_move_known(ctx, _A, _MOVE["Roar"]))
    ctx, _ = _split(c.if_move_known(ctx, _A, _MOVE["Whirlwind"]))
    ctx, _ = _split(c.go_to(ctx))
    return ctx


_FLAGS = (_basic, _eval_attack, _expert)  # Roark's ai_flags = 7, in bit order (roark.md 1)


# --- Refusals ----------------------------------------------------------------------

_ATTACKER_ABILITIES = ("Normalize", "Adaptability", "Scrappy", "Tinted Lens")
_DEFENDER_ABILITIES = ("Levitate", "Wonder Guard", "Filter", "Solid Rock")
_STATE_MOVE_IDS = {
    n: switching.MOVE_IDS[n]
    for n in ("Foresight", "Odor Sleuth", "Miracle Eye", "Roost", "Magnet Rise", "Ingrain")
}


class _Refused(NamedTuple):
    move: jnp.ndarray  # bool[SLOTS, MOVES]: opponent move without transcribed script paths
    sites: jnp.ndarray  # bool[SLOTS]: opponent moveset with too many script-roll sites
    ability: jnp.ndarray  # bool[2, SLOTS]
    item: jnp.ndarray  # bool[2, SLOTS]
    state_move: jnp.ndarray  # bool[2, SLOTS, MOVES]
    bag: jnp.ndarray  # bool[BAG_SLOTS]: bag item other than a Potion
    scoring: jnp.ndarray  # bool: ai_scores is refused
    any: jnp.ndarray  # bool: ai_policy is refused (switching aside)


def _refused(tables: Tables, state: BattleState) -> _Refused:
    present = state.species > 0
    moves = state.moves.astype(_I32)
    transcribed = jnp.isin(moves[1], jnp.array(list(c.MOVE_EFFECTS), _I32)) | (moves[1] == 0)
    move = present[1][:, None] & ~transcribed
    sites = present[1] & (jax.vmap(script_roll_count)(moves[1]) > MAX_SCRIPT_ROLLS)
    species = state.species.astype(_I32)
    ability = tables.species_abilities_decomp[species, state.ability_slot.astype(_I32)]
    side_abilities = (_DEFENDER_ABILITIES, _ATTACKER_ABILITIES)
    ability_bad = jnp.stack(
        [
            jnp.isin(ability[side], jnp.array([_ABILITY[n] for n in names], _I32))
            for side, names in enumerate(side_abilities)
        ]
    )
    ability_bad = present & ability_bad
    items = state.item.astype(_I32)
    item_bad = present & (
        items == jnp.array([[switching.ITEM_IRON_BALL], [c.ITEM_EXPERT_BELT]], _I32)
    )
    state_ids = jnp.array(list(_STATE_MOVE_IDS.values()), _I32)
    state_move = present[..., None] & jnp.isin(moves, state_ids)
    bag_items = state.trainer_items[1].astype(_I32)
    bag = (bag_items != 0) & (tables.trainer_item_effect[bag_items] != TrainerItemId.POTION)
    scoring = (
        jnp.any(move)
        | jnp.any(sites)
        | jnp.any(ability_bad)
        | jnp.any(item_bad)
        | jnp.any(state_move)
    )
    return _Refused(
        move, sites, ability_bad, item_bad, state_move, bag, scoring, scoring | jnp.any(bag)
    )


@lru_cache(maxsize=1)
def _names():
    from nuzline.data.tables import load_names

    try:
        return load_names()
    except OSError:
        return None


def _move_name(move: int) -> str:
    names = _names()
    return names.move_names[move] if names is not None else f"move id {move}"


def _item_name(item: int) -> str:
    names = _names()
    return names.item_names[item] if names is not None else f"item id {item}"


def _scoring_refusals(tables: Tables, state: BattleState) -> list[str]:
    r = _refused(tables, state)
    moves = np.asarray(state.moves)
    out: list[str] = []
    for s, k in zip(*np.nonzero(np.asarray(r.move)), strict=True):
        out.append(
            f"ai scoring: opponent slot {s}: the script paths of move "
            f"{_move_name(int(moves[1, s, k]))} are not transcribed (docs/ai/roark.md 5)"
        )
    for s in np.flatnonzero(np.asarray(r.sites)):
        out.append(
            f"ai scoring: opponent slot {s}: its moveset has more IfRandomLessThan sites "
            f"than the {MAX_SCRIPT_ROLLS} AiOutcome.script_rolls encodes"
        )
    species = np.asarray(state.species).astype(np.int64)
    ability = np.asarray(tables.species_abilities_decomp)[
        species, np.asarray(state.ability_slot).astype(np.int64)
    ]
    ability_names = {v: k for k, v in _ABILITY.items()}
    for side, s in zip(*np.nonzero(np.asarray(r.ability)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: ability {ability_names[int(ability[side, s])]} "
            "changes IfMoveEffectivenessEquals and is not implemented"
        )
    items = np.asarray(state.item)
    for side, s in zip(*np.nonzero(np.asarray(r.item)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: item {_item_name(int(items[side, s]))} "
            "changes IfMoveEffectivenessEquals and is not implemented"
        )
    move_names = {v: k for k, v in _STATE_MOVE_IDS.items()}
    for side, s, k in zip(*np.nonzero(np.asarray(r.state_move)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: move {move_names[int(moves[side, s, k])]} "
            "leaves a type-chart state the engine does not carry"
        )
    return out


def _item_refusals(tables: Tables, state: BattleState) -> list[str]:
    bag = np.asarray(state.trainer_items[1])
    return [
        f"item use: bag slot {k}: {_item_name(int(bag[k]))} is not a transcribed "
        "TrainerAI_ShouldUseItem item (only Potion)"
        for k in np.flatnonzero(np.asarray(_refused(tables, state).bag))
    ]


def ai_refusals(tables: Tables, state: BattleState) -> list[str]:
    """Every reason the AI of this (concrete) state is refused, one human-readable line
    each; empty when ``ai_policy`` is exact for it."""
    return (
        _scoring_refusals(tables, state)
        + _item_refusals(tables, state)
        + switching_refusals(tables, state)
    )


def _check(tables: Tables, state: BattleState, reasons) -> None:
    """Raise ``UnsupportedAi`` with ``reasons(tables, state)`` when there are any; under
    a trace (jit, vmap, even of a closed-over state) the check cannot run."""
    if isinstance(_refused(tables, state).any, jax.core.Tracer):
        return
    found = reasons(tables, state)
    if found:
        raise UnsupportedAi("; ".join(found))


# --- Scores, switch, item, pick -------------------------------------------------------


def resolve_player_ability(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome
) -> jnp.ndarray:
    """``int32``: the player's ability as ``LoadBattlerAbility DEFENDER`` sees it
    (trainer_ai.c:1170; roark.md 9.2), a decomp ability id: the true ability when
    ``ai_ability_known`` is set for the player's active Pokemon (or it is Shadow Tag,
    Magnet Pull or Arena Trap), else the species' ability in slot
    ``ai_outcome.ability_guess``, or its single ability when it has one
    (``commands.defender_ability``, which ``ai_scores`` uses for every such read)."""
    return c.defender_ability(tables, state, ai_outcome.ability_guess)


def _scores(tables: Tables, state: BattleState, ai_outcome: AiOutcome) -> jnp.ndarray:
    ctx = c.new_context(tables, state, ai_outcome)
    for flag in _FLAGS:  # TrainerAI_MainSingles (trainer_ai.c:297-310)
        for slot in range(MOVES):  # TrainerAI_EvalMoves (:483-525)
            ctx = flag(c.at_slot(ctx, slot))
            # A MOVE_NONE slot runs no command and scores 0 (:499-501).
            ctx = ctx._replace(
                scores=ctx.scores.at[slot].set(jnp.where(ctx.move == 0, 0, ctx.scores[slot]))
            )
    return jnp.where(_refused(tables, state).scoring, -1, ctx.scores).astype(_I32)


_scores_jit = jax.jit(_scores)


def ai_scores(tables: Tables, state: BattleState, ai_outcome: AiOutcome) -> jnp.ndarray:
    """``int32[4]``: the opponent's move scores after Roark's three flag scripts, from
    base 100 (``TrainerAI_Init``, trainer_ai.c:223; invalid slots 0). Raises
    ``UnsupportedAi`` on a refused concrete state; -1 per slot when traced."""
    _check(tables, state, _scoring_refusals)
    return _scores_jit(tables, state, ai_outcome)


def maybe_switch(tables: Tables, state: BattleState, ai_outcome: AiOutcome) -> jnp.ndarray:
    """``int32``: ``TrainerAI_ShouldSwitch`` (trainer_ai.c:3894; roark.md 11.1) under
    ``ai_outcome.switch_stop``: the switch action ``4 + j``, or ``NO_SWITCH``
    (``switching.switch_action``)."""
    return switch_action(tables, state, ai_outcome.switch_stop)


def _use_item(tables: Tables, state: BattleState) -> jnp.ndarray:
    # TrainerAI_ShouldUseItem (trainer_ai.c:4056; roark.md 12). Embargo is not in M1.
    alive = jnp.sum(((state.species[1] > 0) & (state.hp[1] > 0)).astype(_I32))
    slot = jnp.arange(BAG_SLOTS)
    count = state.trainer_item_count[1].astype(_I32)
    considered = (slot == 0) | (alive <= count - slot + 1)  # :4092
    items = state.trainer_items[1].astype(_I32)
    a = state.active[1]
    hp, max_hp = state.hp[1, a].astype(_I32), state.max_hp[1, a].astype(_I32)
    # The HP-restore rule (:4103-4116) for a Potion (hpRestored 20).
    heals = (hp > 0) & ((hp < max_hp // 4) | (max_hp - hp > POTION_HP_RESTORED))
    potion = tables.trainer_item_effect[items] == TrainerItemId.POTION
    use = considered & (items != 0) & potion & heals
    return jnp.where(jnp.any(use), FIRST_ITEM_ACTION + jnp.argmax(use), NO_ITEM).astype(_I32)


def maybe_use_item(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: ``TrainerAI_ShouldUseItem`` (trainer_ai.c:4056; roark.md 12): the item
    action ``10 + k`` of the first bag slot it uses, or ``NO_ITEM``. It draws nothing.
    Both engines empty the later slots the decomp's loop also zeroes."""
    _check(tables, state, _item_refusals)
    return _use_item(tables, state)


def _pick(tables: Tables, state: BattleState, scores, tiebreak) -> jnp.ndarray:
    """``TrainerAI_MainSingles``'s pick (trainer_ai.c:315-338; roark.md 10): slot 0 and
    every slot with a move are candidates, the top score wins, ties go to
    ``list[RandNext % n]`` with ``tiebreak % n``. A usable move always outscores an
    invalid one (at most 3 against at least 86 for Roark), so the pick is legal; with no
    usable move the battle uses Struggle (battle_controller_player.c:433), action 0."""
    has = (state.moves[1, state.active[1]] != 0).at[0].set(True)
    top = jnp.max(jnp.where(has, scores, -1))
    tied = has & (scores == top)
    n = jnp.sum(tied.astype(_I32))
    index = jnp.asarray(tiebreak, _I32) % n
    order = jnp.cumsum(tied.astype(_I32)) - 1
    pick = jnp.argmax(tied & (order == index)).astype(_I32)
    return jnp.where(jnp.any(usable_moves(tables, state, 1)), pick, 0)


def _policy(tables: Tables, state: BattleState, ai_outcome: AiOutcome) -> jnp.ndarray:
    switch = maybe_switch(tables, state, ai_outcome)
    item = _use_item(tables, state)
    move = _pick(tables, state, _scores(tables, state, ai_outcome), ai_outcome.tiebreak)
    action = jnp.where(switch != NO_SWITCH, switch, jnp.where(item != NO_ITEM, item, move))
    return jnp.where(_refused(tables, state).any, REFUSED_ACTION, action).astype(_I32)


_policy_jit = jax.jit(_policy)


def ai_policy(tables: Tables, state: BattleState, ai_outcome: AiOutcome) -> jnp.ndarray:
    """``int32``: the opponent's action in plan 1a's encoding (``TrainerAI_PickCommand``
    then ``TrainerAI_Main``; roark.md 2): a voluntary switch ``4 + j``, else a bag item
    ``10 + k``, else the top-scoring move slot ``0..3``. Raises ``UnsupportedAi`` on a
    refused concrete state; ``switching.REFUSED_ACTION`` when traced."""
    _check(tables, state, ai_refusals)
    return _policy_jit(tables, state, ai_outcome)


@jax.jit
def ai_action_distribution_enumerated(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``float32[NUM_OPP_ACTIONS]``: the opponent's action distribution by brute force,
    ``ai_policy`` under ``vmap`` over every ``AiOutcome`` of ``enumerate_ai_outcomes``
    (3840) weighted by its probability; NaN when any outcome's action is refused. It is
    the test oracle of the factored ``ai_action_distribution`` (``tests/ai``), kept
    because it shares no reasoning with the factoring. Always traced (``jax.jit``), so
    it never raises: a refused state gives NaN. The 3840 terms are summed pairwise
    (``_pairwise_sum``): a sequential float32 sum drifts by about 2e-6."""
    outcomes, probs = enumerate_ai_outcomes(tables, state)
    actions = jax.vmap(lambda o: _policy(tables, state, o))(outcomes)
    terms = jnp.where(actions[:, None] == jnp.arange(NUM_OPP_ACTIONS), probs[:, None], 0)
    dist = _pairwise_sum(terms.astype(jnp.float32))
    return jnp.where(jnp.any(actions < 0), jnp.nan, dist)  # refused (traced only)


def _pairwise_sum(x: jnp.ndarray) -> jnp.ndarray:
    """The sum over axis 0 as a balanced binary tree (zero-padded to a power of two):
    float32 rounding error O(log n) instead of O(n)."""
    pad = (1 << (len(x) - 1).bit_length()) - len(x)
    x = jnp.concatenate([x, jnp.zeros((pad, *x.shape[1:]), x.dtype)])
    while len(x) > 1:
        x = x[0::2] + x[1::2]
    return x[0]


_GUESSES = 2  # AiOutcome.ability_guess: 0 or 1, 1/2 each


def _move_distribution(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``float32[MOVES]``: the distribution of ``TrainerAI_Main``'s move pick, over the
    only draws ``_scores`` and ``_pick`` read: ``ability_guess`` x ``script_rolls``
    (32 scorings) and, for each scoring, ``tiebreak`` through ``_pick`` itself."""
    rolls, p_rolls = script_roll_probs(state)  # [16, 4], [16]
    n = _GUESSES * len(rolls)
    outcomes = AiOutcome(
        sim_roll=jnp.full((n, MOVES), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.repeat(jnp.arange(_GUESSES, dtype=jnp.int8), len(rolls)),
        script_rolls=jnp.tile(jnp.asarray(rolls), (_GUESSES, 1)),
        tiebreak=jnp.zeros(n, jnp.int8),  # _scores does not read it
        switch_stop=jnp.full(n, SWITCH_NONE, jnp.int8),  # nor this
    )
    scores = jax.vmap(lambda o: _scores(tables, state, o))(outcomes)  # [32, MOVES]
    tiebreaks = jnp.arange(TIEBREAK_MOD, dtype=jnp.int8)
    picks = jax.vmap(  # [32, TIEBREAK_MOD]
        lambda sc: jax.vmap(lambda t: _pick(tables, state, sc, t))(tiebreaks)
    )(scores)
    p_tiebreak = jnp.asarray(TIEBREAK_P, jnp.float32)
    shares = jnp.einsum("t,stm->sm", p_tiebreak, jax.nn.one_hot(picks, MOVES))  # [32, MOVES]
    weight = jnp.tile(p_rolls, _GUESSES) / _GUESSES  # [32]
    return weight @ shares


@jax.jit
def _distribution(tables: Tables, state: BattleState) -> jnp.ndarray:
    """The exact distribution, factored as ``_policy`` decides (roark.md 2): the switch
    when ``switch_stop`` means one, else the item when ``_use_item`` gives one, else the
    move pick. The switch reads only ``switch_stop``, the item no draw, the move pick
    only ``ability_guess``, ``script_rolls`` and ``tiebreak``, so the 3840 outcomes of
    ``ai_action_distribution_enumerated`` reduce to 10 switch stops and 32 scorings.
    NaN exactly where the enumeration gives NaN: a refused state, or a refused switch
    layout (``switch_action`` gives ``REFUSED_ACTION`` for every stop)."""
    stops = jnp.arange(SWITCH_SITES + 1, dtype=jnp.int8)
    p_stop = switch_stop_probs(switch_sites(tables, state))  # [SWITCH_SITES + 1]
    switch = jax.vmap(lambda stop: switch_action(tables, state, stop))(stops)
    switches = switch >= 0
    dist = jnp.zeros(NUM_OPP_ACTIONS, jnp.float32)
    dist = dist.at[jnp.where(switches, switch, 0)].add(jnp.where(switches, p_stop, 0))
    p_stay = jnp.sum(jnp.where(switch == NO_SWITCH, p_stop, 0))
    item = _use_item(tables, state)
    moves = (
        jnp.zeros(NUM_OPP_ACTIONS, jnp.float32).at[:MOVES].set(_move_distribution(tables, state))
    )
    stay = jnp.where(item != NO_ITEM, jax.nn.one_hot(item, NUM_OPP_ACTIONS), moves)
    refused = _refused(tables, state).any | jnp.any(switch == REFUSED_ACTION)
    return jnp.where(refused, jnp.nan, dist + p_stay * stay)


def ai_action_distribution(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``float32[NUM_OPP_ACTIONS]``: the exact probability of each opponent action over
    every ``AiOutcome``, computed factored (``_distribution``: 32 scorings instead of
    the 3840 outcomes of ``ai_action_distribution_enumerated``, which it equals).
    Raises ``UnsupportedAi`` on a refused concrete state; NaN when traced."""
    _check(tables, state, ai_refusals)
    return _distribution(tables, state)
