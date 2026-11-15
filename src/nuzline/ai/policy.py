"""Roark's trainer AI: move scores, the chosen action and its exact distribution.

``docs/ai/roark.md`` is the specification. One decision runs as the decomp does
(roark.md 2): ``TrainerAI_ShouldSwitch`` (``maybe_switch``), then
``TrainerAI_ShouldUseItem`` (``maybe_use_item``), then ``TrainerAI_Main``: base scores,
the BASIC, EVAL_ATTACK and EXPERT flag scripts over move slots 0-3 (``ai_scores``), and
the highest score with ties broken by ``AiOutcome.tiebreak``. The Kaizo policy runs the
flags of the trainer's set (``layout.flags``; E1 Ruling R2, docs/ai/kaizo_roark.md 1). Every random draw is a
field of the ``AiOutcome`` input; nothing here draws randomness.

The scripts are transcribed label by label from ``src/battle/trainer_ai/script.s``
(paths relative to ``external/pokeplatinum``) as straight-line JAX over the commands of
``nuzline.ai.commands``: a jump splits ``AiContext.live`` into the fall-through path
and the taken path, and each label block runs under the union of the paths that reach
it. Only the branches that can be taken for the transcribed moves
(``commands.MOVE_EFFECTS`` in vanilla, the effects of ``layout.KAIZO.transcribed`` in
Kaizo) are written out; the effect jump tables list just those moves' effects, chosen per
layout at trace time (``_dispatch``; roark.md 5.4, docs/ai/kaizo_roark.md 5).

The game's policy (``nuzline.ai.layout``): every public function takes an optional
``layout``; without one it reads ``layout_of(tables)`` when ``tables`` are concrete, and
takes ``layout.VANILLA`` when they are traced, refusing tables of another policy
(``_Refused.policy``). The traced internals take the layout as a static argument. A
per-slot layout (Kaizo) keys every random read of the scoring by move slot and by its order
on the slot's path (E1 Ruling R4; ``outcome`` docstring), and its exact distribution is
factored per slot (``_slot_move_distribution``; docs/ai/kaizo_roark.md 8);
``ai_action_distribution_loose`` is its test oracle, with every decomp draw its own.

Refusals (``ai_refusals``): an opponent move whose script paths are not transcribed
(``layout.transcribed``), Curse by a Ghost type (its Ghost labels are not transcribed),
or, in a per-slot layout, a move whose effect's Expert_Main label is not transcribed
(``outcome.KAIZO_EXPERT_LABELS`` less ``layout.expert_transcribed``; none for Kaizo
Roark since K3c Task 6), an opponent move the AI's damage estimate or move type takes
by a move-id case it does not model, a vanilla moveset with more script-roll sites than
``AiOutcome`` encodes, a per-slot path with more random reads than ``layout.slot_reads``
(``ai_path_refusals``), the abilities, items and moves that change
``effectiveness_class``, a bag item other than a Potion, and every refusal of the
voluntary-switch rule (``switching.switching_refusals``). On concrete inputs the public
functions raise ``UnsupportedAi``; traced (jit, vmap) they cannot raise, so
``ai_scores`` gives -1 for every slot and ``ai_policy`` gives
``switching.REFUSED_ACTION``. Callers that trace (Stage A) run ``ai_refusals`` on the
fight first.

Knowledge (roark.md 9): the scripts read the player's moves only through the AI's seen
list (``BattleState.ai_seen_moves``, ``AiContext.seen_moves``: a move the player has not
used is not known) and the player's ability only through ``resolve_player_ability``
(``AiContext.player_ability``). ``step`` keeps both up to date.
"""

from __future__ import annotations

import itertools
from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.ai import commands as c
from nuzline.ai import switching
from nuzline.ai.commands import AiContext, BattleEffect, Battler, LoadType, TypeMulti
from nuzline.ai.layout import (
    BASIC,
    CHECK_HP,
    EVAL_ATTACK,
    EXPERT,
    PRIORITIZE_EXTREMES,
    RISKY,
    SETUP_FIRST_TURN,
    VANILLA,
    WEATHER,
    AiLayout,
    layout_of,
)
from nuzline.ai.outcome import (
    ABILITY_GUESSES,
    ENUMERATION_WIDTH,
    KAIZO_EXPERT_LABELS,
    MAX_SCRIPT_ROLLS,
    READ_DRAWS,
    SIM_ROLL_MAX,
    SWITCH_NONE,
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
from nuzline.data.game_constants import AI_POLICIES
from nuzline.data.tables import Tables
from nuzline.engine.actions import FIRST_ITEM_ACTION, NUM_OPP_ACTIONS, locked_slot, usable_moves
from nuzline.engine.damage import decomp_abilities
from nuzline.engine.enumerate import WIDTH_CAP
from nuzline.engine.ids import POTION_HP_RESTORED, Stat, TrainerItemId
from nuzline.engine.order import COMPARE_SPEED_SLOWER
from nuzline.engine.state import BAG_SLOTS, MOVES, BattleState

__all__ = [
    "NO_ITEM",
    "UnsupportedAi",
    "ai_action_distribution",
    "ai_action_distribution_enumerated",
    "ai_action_distribution_loose",
    "ai_flat_width",
    "ai_path_refusals",
    "ai_policy",
    "ai_refusals",
    "ai_scores",
    "ai_width",
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


def _basic(ctx: AiContext, layout: AiLayout) -> AiContext:
    """``Basic_Main`` (script.s:52; roark.md 5.1, docs/ai/kaizo_roark.md 5.1).
    ``layout.water2_ability`` is the ability of the compare at script.s:78."""
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
    # :78 is Levitate again in vanilla (quirk: the decomp's comment says it should be Dry
    # Skin; a second Levitate test never matches) and Dry Skin in Kaizo (layout.KAIZO).
    ctx, water2 = _split(c.if_loaded_equal_to(ctx, layout.water2_ability))
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

    # Basic_ScoreMoveEffect (:133): the jump table (_BASIC_LABELS), then the labels.
    return _dispatch(_at(ctx, effect), _BASIC_LABELS, layout)


def _basic_check_cannot_explode(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotExplode`` (script.s:297-319)."""
    # Basic_CheckCannotExplode (:297)
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _A)
    ctx, last_mon = _split(c.if_loaded_equal_to(ctx, _ABILITY["Mold Breaker"]))
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Damp"]))
    ctx = _score(ctx, t, -10)
    # Basic_CheckLastMon (:307). The attacker's other living Pokemon jump to
    # Basic_Explode_Terminate (:318), a PopOrEnd: no change.
    ctx = c.count_alive_party_battlers(_at(ctx, last_mon | ctx.live), _A)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, 0))
    ctx = c.count_alive_party_battlers(ctx, _D)
    ctx, t = _split(c.if_loaded_not_equal_to(ctx, 0))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.go_to(ctx))
    ctx = _score(ctx, t, -1)
    return ctx


def _basic_check_can_recover_hp(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCanRecoverHP`` (script.s:511-517)."""
    # Basic_CheckCanRecoverHP (:511): HP% not 100 jumps to its PopOrEnd (:516).
    ctx = _at(ctx, mask)
    ctx, _ = _split(c.if_hp_percent_not_equal_to(ctx, _A, 100))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, -8))
    return ctx


def _basic_check_cannot_poison(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotPoison`` (script.s:519-547)."""
    # Basic_CheckCannotPoison (:519)
    ctx = _at(ctx, mask)
    for target in (LoadType.DEFENDER_TYPE_1, LoadType.DEFENDER_TYPE_2):
        ctx = c.load_type_from(ctx, target)
        for type_name in ("Steel", "Poison"):
            ctx, t = _split(c.if_loaded_equal_to(ctx, _TYPE[type_name]))
            ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    for ability in ("Immunity", "Magic Guard", "Poison Heal"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[ability]))
        ctx = _score(ctx, t, -10)
    ctx, hydration = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Leaf Guard"]))
    ctx = c.load_current_weather(ctx)
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AiWeather.SUNNY))
    ctx = _score(ctx, t, -10)
    # Basic_CheckCannotPoison_Hydration (:538)
    ctx = c.load_battler_ability(_at(ctx, hydration | ctx.live), _D)
    ctx, status = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Hydration"]))
    ctx = c.load_current_weather(ctx)
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AiWeather.RAINING))
    ctx = _score(ctx, t, -10)
    # Basic_CheckCannotPoison_StatusOrSafeguard (:544)
    ctx = _at(ctx, status | ctx.live)
    ctx, t = _split(c.if_status(ctx, _D, c.MonCondition.ANY))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.SAFEGUARD))
    ctx = _score(ctx, t, -10)
    ctx = c.pop_or_end(ctx)
    return ctx


def _basic_check_light_screen(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckAlreadyUnderLightScreen`` (script.s:549-552)."""
    # Basic_CheckAlreadyUnderLightScreen (:549)
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_side_condition(ctx, _A, c.SideCondition.LIGHT_SCREEN))
    ctx = _score(ctx, t, -8)
    ctx = c.pop_or_end(ctx)
    return ctx


def _basic_check_speed(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckHighStatStage_Speed`` (script.s:363-372)."""
    # Basic_CheckHighStatStage_Speed (:363)
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_field_conditions_mask(ctx, c.FieldCondition.TRICK_ROOM))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _A)
    ctx, no_simple = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Simple"]))
    ctx, t = _split(c.if_stat_stage_greater_than(ctx, _A, Stat.SPE, 8))
    ctx = _score(ctx, t, -10)
    ctx = c.pop_or_end(ctx)
    # Basic_CheckHighStatStage_Speed_NoSimple (:370)
    ctx = _at(ctx, no_simple)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _A, Stat.SPE, 12))
    ctx = _score(ctx, t, -10)
    ctx = c.pop_or_end(ctx)
    return ctx


def _high_stat_stage(label: str, stat: Stat, line: int):
    """A ``Basic_CheckHighStatStage_<stat>`` label (script.s:343-392, at ``line``) of Attack,
    Defense, Sp. Atk or Sp. Def: -10 when the attacker's stage is already at the top (above +2
    with Simple, which doubles its changes; +6 otherwise)."""

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = c.load_battler_ability(_at(ctx, mask), _A)
        ctx, no_simple = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Simple"]))
        ctx, t = _split(c.if_stat_stage_greater_than(ctx, _A, stat, 8))
        ctx = c.pop_or_end(_score(ctx, t, -10))
        # <label>_NoSimple
        ctx, t = _split(c.if_stat_stage_equal_to(_at(ctx, no_simple), _A, stat, 12))
        return c.pop_or_end(_score(ctx, t, -10))

    check.__name__ = check.__qualname__ = f"_basic_check_high_{stat.name.lower()}"
    check.__doc__ = f"``{label}`` (script.s:{line})."
    return check


_basic_check_high_atk = _high_stat_stage("Basic_CheckHighStatStage_Attack", Stat.ATK, 343)
_basic_check_high_def = _high_stat_stage("Basic_CheckHighStatStage_Defense", Stat.DEF, 353)
_basic_check_high_spa = _high_stat_stage("Basic_CheckHighStatStage_SpAttack", Stat.SPA, 374)
_basic_check_high_spd = _high_stat_stage("Basic_CheckHighStatStage_SpDefense", Stat.SPD, 384)


def _basic_check_cannot_sleep(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotSleep`` (script.s:288-295)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_status(ctx, _D, c.MonCondition.ANY))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.SAFEGUARD))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    for name in ("Insomnia", "Vital Spirit"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[name]))
        ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _basic_check_cannot_confuse(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotConfuse`` (script.s:597-605): already confused -5 (:599)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_volatile_status(ctx, _D, c.VolatileCondition.CONFUSION))
    ctx = _score(ctx, t, -5)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Own Tempo"]))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.SAFEGUARD))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _basic_check_cannot_paralyze(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotParalyze`` (script.s:612-631)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    for name in ("Limber", "Magic Guard"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[name]))
        ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _A)
    ctx, status = _split(c.if_loaded_equal_to(ctx, _ABILITY["Mold Breaker"]))
    ctx, wave = _split(c.if_move_equal_to(ctx, _MOVE["Thunder Wave"]))
    ctx, t = _split(c.go_to(ctx))
    status = status | t
    # Basic_CheckCannotParalyze_ThunderWave (:623), falling into _ImmuneToStatus.
    ctx = c.load_battler_ability(_at(ctx, wave), _D)
    for name in ("Motor Drive", "Volt Absorb"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[name]))
        ctx = _score(ctx, t, -10)
    # Basic_CheckCannotParalyze_ImmuneToStatus (:628)
    ctx = _at(ctx, status | ctx.live)
    ctx, t = _split(c.if_status(ctx, _D, c.MonCondition.ANY))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.SAFEGUARD))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _basic_check_cannot_leech_seed(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotLeechSeed`` (script.s:639-648)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_move_effect(ctx, _D, c.MoveEffectMask.LEECH_SEED))
    ctx = _score(ctx, t, -10)
    for target in (LoadType.DEFENDER_TYPE_1, LoadType.DEFENDER_TYPE_2):
        ctx = c.load_type_from(ctx, target)
        ctx, t = _split(c.if_loaded_equal_to(ctx, _TYPE["Grass"]))
        ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Magic Guard"]))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _basic_check_cannot_attract(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotAttract`` (script.s:731-752): -10 unless the two are of opposite
    genders (a genderless one on either side is -10)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_volatile_status(ctx, _D, c.VolatileCondition.ATTRACT))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Oblivious"]))
    ctx = _score(ctx, t, -10)
    ctx = c.load_gender(ctx, _A)
    ctx, male = _split(c.if_loaded_equal_to(ctx, c.Gender.MALE))
    ctx, female = _split(c.if_loaded_equal_to(ctx, c.Gender.FEMALE))
    ctx, t = _split(c.go_to(ctx))
    minus10 = t
    # Basic_CheckCannotAttract_BothMale (:741) / _BothFemale (:746)
    for reached, other in ((male, c.Gender.FEMALE), (female, c.Gender.MALE)):
        ctx = c.load_gender(_at(ctx, reached), _D)
        ctx, _ = _split(c.if_loaded_equal_to(ctx, other))  # -> _Terminate
        ctx, t = _split(c.go_to(ctx))
        minus10 = minus10 | t
    ctx = _score(_at(ctx, jnp.zeros_like(minus10)), minus10, -10)
    return c.pop_or_end(ctx)


def _basic_check_cannot_burn(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCannotBurn`` (script.s:879-890)."""
    ctx = _at(ctx, mask)
    ctx = c.load_battler_ability(ctx, _D)
    for name in ("Water Veil", "Magic Guard"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[name]))
        ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_status(ctx, _D, c.MonCondition.ANY))
    ctx = _score(ctx, t, -10)
    for target in (LoadType.DEFENDER_TYPE_1, LoadType.DEFENDER_TYPE_2):
        ctx = c.load_type_from(ctx, target)
        ctx, t = _split(c.if_loaded_equal_to(ctx, _TYPE["Fire"]))
        ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.SAFEGUARD))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _double_stat_stage(label: str, first: Stat, second: Stat, line: int, trick_room=False):
    """``Basic_CheckBulkUp`` (script.s:965-979) / ``_CalmMind`` (:986-1000) /
    ``_DragonDance`` (:1002-1019, which first gives -10 under Trick Room): with Simple, -10
    when either stat is above decomp 8; without, -10 when the first is at +6 and -8 when the
    second is."""

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = _at(ctx, mask)
        if trick_room:
            ctx, t = _split(c.if_field_conditions_mask(ctx, c.FieldCondition.TRICK_ROOM))
            ctx = _score(ctx, t, -10)
        ctx = c.load_battler_ability(ctx, _A)
        ctx, no_simple = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Simple"]))
        for stat in (first, second):
            ctx, t = _split(c.if_stat_stage_greater_than(ctx, _A, stat, 8))
            ctx = _score(ctx, t, -10)
        ctx = c.pop_or_end(ctx)
        # <label>_NoSimple
        ctx, t = _split(c.if_stat_stage_equal_to(_at(ctx, no_simple), _A, first, 12))
        ctx = _score(ctx, t, -10)
        ctx, t = _split(c.if_stat_stage_equal_to(ctx, _A, second, 12))
        return c.pop_or_end(_score(ctx, t, -8))

    check.__name__ = check.__qualname__ = f"_basic_check_{label.split('_')[-1].lower()}"
    check.__doc__ = f"``{label}`` (script.s:{line})."
    return check


_basic_check_bulk_up = _double_stat_stage("Basic_CheckBulkUp", Stat.ATK, Stat.DEF, 965)
_basic_check_calm_mind = _double_stat_stage("Basic_CheckCalmMind", Stat.SPA, Stat.SPD, 986)
_basic_check_dragon_dance = _double_stat_stage(
    "Basic_CheckDragonDance", Stat.ATK, Stat.SPE, 1002, trick_room=True
)


def _hazard_layers(label: str, condition, full: int, line: int):
    """``Basic_CheckSpikes`` (script.s:707-713) / ``Basic_CheckToxicSpikes`` (:1445-1453): -10
    when the player's side already has ``full`` layers or the player has no other Pokemon
    alive."""

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = c.load_spikes_layers(_at(ctx, mask), _D, condition)
        ctx, t = _split(c.if_loaded_equal_to(ctx, full))
        ctx = _score(ctx, t, -10)
        ctx = c.count_alive_party_battlers(ctx, _D)
        ctx, t = _split(c.if_loaded_equal_to(ctx, 0))
        return c.pop_or_end(_score(ctx, t, -10))

    check.__name__ = check.__qualname__ = (
        f"_basic_check_{label.removeprefix('Basic_Check').lower()}"
    )
    check.__doc__ = f"``{label}`` (script.s:{line})."
    return check


_basic_check_spikes = _hazard_layers("Basic_CheckSpikes", c.SideCondition.SPIKES, 3, 707)
_basic_check_toxic_spikes = _hazard_layers(
    "Basic_CheckToxicSpikes", c.SideCondition.TOXIC_SPIKES, 2, 1445
)


def _basic_check_curse(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckCurse`` (script.s:679-698)."""
    # Basic_CheckCurse (:679). A Ghost-type attacker jumps to Basic_CheckCurse_GhostType
    # (:700), which is not transcribed: _refused refuses Curse on a Ghost type, so the
    # jumps below are never taken.
    ctx = _at(ctx, mask)
    for target in (LoadType.ATTACKER_TYPE_1, LoadType.ATTACKER_TYPE_2):
        ctx = c.load_type_from(ctx, target)
        ctx, _ = _split(c.if_loaded_equal_to(ctx, _TYPE["Ghost"]))
    ctx = c.load_battler_ability(ctx, _A)
    ctx, no_simple = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Simple"]))
    for stat in (Stat.ATK, Stat.DEF):
        ctx, t = _split(c.if_stat_stage_greater_than(ctx, _A, stat, 8))
        ctx = _score(ctx, t, -10)
    ctx = c.pop_or_end(ctx)
    # Basic_CheckCurse_NoSimple (:694)
    ctx = _at(ctx, no_simple)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _A, Stat.ATK, 12))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _A, Stat.DEF, 12))
    ctx = _score(ctx, t, -8)
    ctx = c.pop_or_end(ctx)
    return ctx


def _clear_body(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckClearBodyEffect`` (script.s:469-473): -10 against Clear Body or White
    Smoke (the player's ability, guessed or known)."""
    ctx = c.load_battler_ability(_at(ctx, mask), _D)
    for name in ("Clear Body", "White Smoke"):
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY[name]))
        ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _basic_check_low_attack(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckLowStatStage_Attack`` (script.s:428-432, then :469)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.ATK, 0))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Hyper Cutter"]))
    ctx = _score(ctx, t, -10)
    ctx, clear_body = _split(c.go_to(ctx))
    return _clear_body(ctx, clear_body)


def _basic_check_low_speed(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckLowStatStage_Speed`` (script.s:438-443, then :469): Trick Room, the
    player at -6, or Speed Boost as ``CheckBattlerAbility`` sees it, -10."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_field_conditions_mask(ctx, c.FieldCondition.TRICK_ROOM))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.SPE, 0))
    ctx = _score(ctx, t, -10)
    ctx = c.check_battler_ability(ctx, _D, _ABILITY["Speed Boost"])
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AI_HAVE))
    ctx = _score(ctx, t, -10)
    ctx, clear_body = _split(c.go_to(ctx))
    return _clear_body(ctx, clear_body)


def _basic_check_low_spd(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckLowStatStage_SpDefense`` (script.s:449-451, then :469)."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.SPD, 0))
    ctx = _score(ctx, t, -10)
    ctx, clear_body = _split(c.go_to(ctx))
    return _clear_body(ctx, clear_body)


def _basic_check_low_evasion(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckLowStatStage_Evasion`` (script.s:462-467, falling into :469): the player
    at -6, or No Guard on either side, -10."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.EVA, 0))
    ctx = _score(ctx, t, -10)
    for battler in (_A, _D):
        ctx = c.load_battler_ability(ctx, battler)
        ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["No Guard"]))
        ctx = _score(ctx, t, -10)
    return _clear_body(ctx, ctx.live)


def _basic_check_reflect(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckAlreadyUnderReflect`` (script.s:607-610): -8 under the AI's own Reflect."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_side_condition(ctx, _A, c.SideCondition.REFLECT))
    return c.pop_or_end(_score(ctx, t, -8))


def _basic_check_tailwind(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckTailwind`` (script.s:1128-1133): Trick Room, or the AI's side already under
    Tailwind, -10."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_field_conditions_mask(ctx, c.FieldCondition.TRICK_ROOM))
    ctx = _score(ctx, t, -10)
    ctx, t = _split(c.if_side_condition(ctx, _A, c.SideCondition.TAILWIND))
    return c.pop_or_end(_score(ctx, t, -10))


def _basic_check_non_standard_damage(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckNonStandardDamageOrChargeTurn`` (script.s:572-585): immune -10; against
    Wonder Guard (a guessing read), unless the attacker has Mold Breaker or the move is
    super effective, -10."""
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    ctx = _score(ctx, t, -10)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Wonder Guard"]))  # -> _Terminate
    ctx = c.load_battler_ability(ctx, _A)
    ctx, _ = _split(c.if_loaded_equal_to(ctx, _ABILITY["Mold Breaker"]))
    ctx, _ = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.DOUBLE))
    ctx, _ = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.QUADRUPLE))
    ctx, t = _split(c.go_to(ctx))  # -> ScoreMinus10
    return _score(ctx, t, -10)


def _basic_check_attacker_asleep(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckAttackerAsleep`` (script.s:660-663): the attacker not asleep, -8."""
    ctx, t = _split(c.if_not_status(_at(ctx, mask), _A, c.MonCondition.SLEEP))
    return c.pop_or_end(_score(ctx, t, -8))


def _basic_check_first_turn(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckFirstTurnInBattle`` (script.s:829-833): not the attacker's first turn in
    battle, -10."""
    ctx = c.load_is_first_turn_in_battle(_at(ctx, mask), _A)
    ctx, t = _split(c.if_loaded_equal_to(ctx, 0))
    return c.pop_or_end(_score(ctx, t, -10))


def _basic_check_rain_dance(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckRainDance`` (script.s:786-801): unless the attacker has Swift Swim or
    Hydration, against a statused Hydration player (a guessing read) -8; then in rain -8."""
    ctx = c.load_battler_ability(_at(ctx, mask), _A)
    ctx, rain = _split(c.if_loaded_equal_to(ctx, _ABILITY["Swift Swim"]))
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Hydration"]))
    rain = rain | t
    ctx = c.load_battler_ability(ctx, _D)
    ctx, t = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Hydration"]))
    rain = rain | t
    ctx, t = _split(c.if_status(ctx, _D, c.MonCondition.ANY))
    ctx = _score(ctx, t, -8)
    # Basic_CheckCurrentWeatherIsRain (:797)
    ctx = c.load_current_weather(_at(ctx, rain | ctx.live))
    ctx, t = _split(c.if_loaded_equal_to(ctx, int(c.AiWeather.RAINING)))
    return c.pop_or_end(_score(ctx, t, -8))


def _basic_check_hail(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckHail`` (script.s:853-871): in hail -8; else against an Ice Body player (a
    guessing read) -8, and +8 back when the attacker has Ice Body too."""
    ctx = c.load_current_weather(_at(ctx, mask))
    ctx, t = _split(c.if_loaded_equal_to(ctx, int(c.AiWeather.HAILING)))
    ctx = _score(ctx, t, -8)
    ctx = c.load_battler_ability(ctx, _D)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Ice Body"]))  # -> _Terminate
    ctx = c.add_to_move_score(ctx, -8)
    ctx = c.load_battler_ability(ctx, _A)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Ice Body"]))  # -> _Terminate
    return c.pop_or_end(c.add_to_move_score(ctx, 8))


def _basic_check_defense(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckLowStatStage_Defense`` (script.s:434-473)."""
    # Basic_CheckLowStatStage_Defense (:434)
    ctx = _at(ctx, mask)
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
    return ctx


def _basic_check_stealth_rock(ctx: AiContext, mask) -> AiContext:
    """``Basic_CheckStealthRock`` (script.s:1540-1547)."""
    # Basic_CheckStealthRock (:1540)
    ctx = _at(ctx, mask)
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.STEALTH_ROCK))
    ctx = _score(ctx, t, -10)
    ctx = c.count_alive_party_battlers(ctx, _D)
    ctx, t = _split(c.if_loaded_equal_to(ctx, 0))
    ctx = _score(ctx, t, -10)
    return c.pop_or_end(ctx)


def _eval_attack(ctx: AiContext) -> AiContext:
    """``EvalAttack_Main`` (script.s:6353; roark.md 5.2, docs/ai/kaizo_roark.md 5.2). The
    Focus Punch and Sucker Punch tests (:6366-6367) and the Focus Punch, Sucker Punch and
    Future Sight kill tests (:6394-6396, with their 170 roll at :6405) match no transcribed
    effect and are not written out. HALVE_DEFENSE (:6365, :6389) and PRIORITY_1 (:6400) are:
    the kill bonus keys on the effect, not the move's priority (Ruling 8), and a
    HALVE_DEFENSE move never kills, since IfCurrentMoveKills computes no damage for a
    ``NO_DAMAGE_CALC_EFFECTS`` move (trainer_ai.c:1539-1554)."""
    ctx = _terminate(ctx)
    ctx, kill = _split(c.if_current_move_kills(ctx, c.USE_MAX_DAMAGE))
    ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    ctx, t = _split(c.if_loaded_equal_to(ctx, c.AI_NOT_HIGHEST_DAMAGE))
    ctx = _score(ctx, t, -1)
    ctx, deprioritize = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.HALVE_DEFENSE))
    ctx, quad = _split(c.go_to(ctx))

    # EvalAttack_MaybeDeprioritize (:6372): -2 unless the 51 roll jumps.
    ctx = _at(ctx, deprioritize)
    ctx, t = _split(c.if_random_less_than(ctx, 51, 6374))
    ctx = c.add_to_move_score(ctx, -2)
    quad = quad | t | ctx.live

    # EvalAttack_CheckQuadEffective (:6377)
    ctx = _at(ctx, quad)
    ctx, plus2 = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.QUADRUPLE))
    ctx = c.pop_or_end(ctx)

    # EvalAttack_TryScorePlus2 (:6382)
    ctx = _at(ctx, plus2)
    ctx, _ = _split(c.if_random_less_than(ctx, 80, 6383))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 2))

    # EvalAttack_ApplyKillBonuses (:6387): HALVE_DEFENSE ends at EvalAttack_Terminate
    # (:6389); PRIORITY_1 takes EvalAttack_ScorePlus2 (:6400 -> :6408), which falls into
    # EvalAttack_ScorePlus4 (:6411); anything else jumps to it (:6401).
    ctx = _at(ctx, kill)
    ctx, _ = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.HALVE_DEFENSE))
    ctx, priority = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.PRIORITY_1))
    ctx, plus4 = _split(c.go_to(ctx))
    ctx = c.add_to_move_score(_at(ctx, priority), 2)
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, priority | plus4), 4))
    return ctx


def _not_resisted(ctx: AiContext) -> tuple[AiContext, jnp.ndarray]:
    """``IfMoveEffectivenessEquals`` IMMUNE, HALF_DAMAGE, QUARTER_DAMAGE in a row, each to
    the same label: the fall-through context and the mask of the paths that jump."""
    ctx, taken = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    for cls in (TypeMulti.HALF, TypeMulti.QUARTER):
        ctx, t = _split(c.if_move_effectiveness_equals(ctx, cls))
        taken = taken | t
    return ctx, taken


def _expert(ctx: AiContext, layout: AiLayout) -> AiContext:
    """``Expert_Main`` (script.s:1623; roark.md 5.3, docs/ai/kaizo_roark.md 5.3): the jump
    table (``_EXPERT_LABELS``) restricted to ``layout``'s effects, then the labels reached;
    every other effect ends at :1811."""
    return _dispatch(_terminate(ctx), _EXPERT_LABELS, layout)


def _expert_drain_move(ctx: AiContext, mask) -> AiContext:
    """``Expert_DrainMove`` (script.s:1827-1839)."""
    # Expert_DrainMove (:1827): immune or resisted, -3 unless the 50 roll jumps.
    ctx, minus3 = _not_resisted(_at(ctx, mask))
    ctx, _ = _split(c.go_to(ctx))  # :1832
    ctx, _ = _split(c.if_random_less_than(_at(ctx, minus3), 50, 1835))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, -3))
    return ctx


def _expert_explosion(ctx: AiContext, mask) -> AiContext:
    """``Expert_Explosion`` (script.s:1841-1884)."""
    # Expert_Explosion (:1841). The player's evasion at decomp 7 or more is -1 (:1856-1857);
    # at 10 or more, -1 again unless the 128 roll jumps (:1858-1860).
    ctx = _at(ctx, mask)
    ctx, high_hp = _split(c.if_stat_stage_less_than(ctx, _D, Stat.EVA, 7))
    ctx = c.add_to_move_score(ctx, -1)
    ctx, t = _split(c.if_stat_stage_less_than(ctx, _D, Stat.EVA, 10))
    high_hp = high_hp | t
    ctx, t = _split(c.if_random_less_than(ctx, 128, 1859))
    ctx = c.add_to_move_score(ctx, -1)
    high_hp = high_hp | t | ctx.live
    # Expert_Explosion_CheckUserHighHP (:1862): HP% 80 or more and not slower: ScoreMinus3
    # unless the 50 roll jumps to the end.
    ctx = _at(ctx, high_hp)
    ctx, medium_hp = _split(c.if_hp_percent_less_than(ctx, _A, 80))
    ctx, t = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    medium_hp = medium_hp | t
    ctx, _ = _split(c.if_random_less_than(ctx, 50, 1865))
    ctx, t = _split(c.go_to(ctx))
    ctx = _score(ctx, t, -3)
    # Expert_Explosion_CheckUserMediumHP (:1868): above 50%, Expert_Explosion_TryScoreMinus1;
    # else +1 unless the 128 roll jumps, then Expert_Explosion_CheckUserLowHP.
    ctx = _at(ctx, medium_hp)
    ctx, minus1 = _split(c.if_hp_percent_greater_than(ctx, _A, 50))
    ctx, low_hp = _split(c.if_random_less_than(ctx, 128, 1870))
    ctx = c.add_to_move_score(ctx, 1)
    low_hp = low_hp | ctx.live
    # Expert_Explosion_CheckUserLowHP (:1873): 30% or less, +1 unless the 50 roll jumps.
    ctx = _at(ctx, low_hp)
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 30))
    ctx, _ = _split(c.if_random_less_than(ctx, 50, 1875))
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.go_to(ctx))  # :1877
    # Expert_Explosion_TryScoreMinus1 (:1879)
    ctx, _ = _split(c.if_random_less_than(_at(ctx, minus1), 50, 1880))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, -1))
    return ctx


def _expert_recovery(ctx: AiContext, mask) -> AiContext:
    """``Expert_Recovery`` (script.s:2640-2678)."""
    # Expert_Recovery (:2640). Expert_Recovery_Unused (:2654-2657) follows the GoTo at :2652
    # and no jump targets it: not reached.
    ctx = _at(ctx, mask)
    ctx, minus3 = _split(c.if_hp_percent_equal_to(ctx, _A, 100))
    ctx, check_hp = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx = c.add_to_move_score(ctx, -8)
    ctx, _ = _split(c.go_to(ctx))  # :2652
    # Expert_Recovery_ScoreMinus3AndEnd (:2659)
    ctx = c.add_to_move_score(_at(ctx, minus3), -3)
    ctx, _ = _split(c.go_to(ctx))
    # Expert_Recovery_CheckHP (:2663): below 70%, or the 30 roll jumps: CheckForSnatch.
    ctx = _at(ctx, check_hp)
    ctx, snatch = _split(c.if_hp_percent_less_than(ctx, _A, 70))
    ctx, t = _split(c.if_random_less_than(ctx, 30, 2665))
    snatch = snatch | t
    ctx = c.add_to_move_score(ctx, -3)
    ctx, _ = _split(c.go_to(ctx))
    # Expert_Recovery_CheckForSnatch (:2669): the player seen with a Snatch effect goes to
    # the end unless the 100 roll falls through.
    ctx = _at(ctx, snatch)
    snatch_effect = BattleEffect.STEAL_STATUS_MOVE
    ctx, plus2 = _split(c.if_move_effect_not_known(ctx, _D, snatch_effect))
    ctx, _ = _split(c.if_random_less_than(ctx, 100, 2671))
    plus2 = plus2 | ctx.live
    # Expert_Recovery_TryScorePlus2 (:2673)
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus2), 20, 2674))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 2))
    return ctx


def _expert_toxic_leech_seed(ctx: AiContext, mask) -> AiContext:
    """``Expert_ToxicLeechSeed`` (script.s:2680-2708)."""
    # Expert_ToxicLeechSeed (:2680). Its IfMoveEffectKnown reads are the attacker's own moves
    # (plan K3c Ruling 9).
    ctx = _at(ctx, mask)
    ctx, effects = _split(c.if_attacker_has_no_damaging_moves(ctx))
    ctx, target_hp = _split(c.if_hp_percent_greater_than(ctx, _A, 50))
    ctx, t = _split(c.if_random_less_than(ctx, 50, 2690))
    ctx = c.add_to_move_score(ctx, -3)
    target_hp = target_hp | t | ctx.live
    # Expert_ToxicLeechSeed_CheckTargetHP (:2693)
    ctx = _at(ctx, target_hp)
    ctx, t = _split(c.if_hp_percent_greater_than(ctx, _D, 50))
    effects = effects | t
    ctx, t = _split(c.if_random_less_than(ctx, 50, 2695))
    ctx = c.add_to_move_score(ctx, -3)
    effects = effects | t | ctx.live
    # Expert_ToxicLeechSeed_CheckMoveEffectsKnown (:2698)
    ctx = _at(ctx, effects)
    ctx, plus2 = _split(c.if_move_effect_known(ctx, _A, BattleEffect.SP_DEF_UP))
    ctx, t = _split(c.if_move_effect_known(ctx, _A, BattleEffect.PROTECT))
    plus2 = plus2 | t
    ctx, _ = _split(c.go_to(ctx))
    # Expert_ToxicLeechSeed_TryScorePlus2 (:2703)
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus2), 60, 2704))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 2))
    return ctx


def _expert_light_screen(ctx: AiContext, mask) -> AiContext:
    """``Expert_LightScreen`` (script.s:2710-2732)."""
    # Expert_LightScreen (:2710)
    ctx = _at(ctx, mask)
    ctx, minus2 = _split(c.if_hp_percent_less_than(ctx, _A, 50))
    ctx, last_move = _split(c.if_hp_percent_less_than(ctx, _A, 90))
    ctx, t = _split(c.if_random_less_than(ctx, 128, 2718))
    ctx = c.add_to_move_score(ctx, 1)
    last_move = last_move | t | ctx.live
    # Expert_LightScreen_CheckLastUsedMove (:2721)
    ctx = c.load_defender_last_used_move_class(_at(ctx, last_move))
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, c.CLASS_SPECIAL))
    ctx, _ = _split(c.if_random_less_than(ctx, 64, 2724))
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.go_to(ctx))  # :2726
    # Expert_LightScreen_ScoreMinus2 (:2728), then Expert_LightScreen_End
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, minus2), -2))
    return ctx


def _expert_status_speed_up(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusSpeedUp`` (script.s:2066-2079)."""
    # Expert_StatusSpeedUp (:2066): not slower, -3; slower, +3 unless the 70 roll jumps.
    ctx = _at(ctx, mask)
    ctx, plus3 = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx = c.add_to_move_score(ctx, -3)
    ctx, _ = _split(c.go_to(ctx))  # :2072
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus3), 70, 2075))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 3))
    return ctx


def _status_attack_up(label: str, stat: Stat, lines: tuple[int, int, int, int], n_end: int):
    """``Expert_StatusAttackUp`` (script.s:1978-2007) / ``Expert_StatusSpAttackUp``
    (:2081-2110), at ``lines`` (label, its 100, 128 and last rolls): at +3 or more (decomp
    stage 9) -1 unless the 100 roll jumps; else at exactly 100 % HP +2 unless the 128 roll
    jumps; then above 70 % HP nothing more, below 40 % -2, otherwise -2 unless the ``n_end``
    roll jumps."""
    start, r100, r128, r_end = lines

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = _at(ctx, mask)
        ctx, max_hp = _split(c.if_stat_stage_less_than(ctx, _A, stat, 9))
        ctx, t = _split(c.if_random_less_than(ctx, 100, r100))
        ctx = c.add_to_move_score(ctx, -1)
        ctx, g = _split(c.go_to(ctx))
        hp_range = t | g
        # <label>_CheckUserAtMaxHP
        ctx, t = _split(c.if_hp_percent_not_equal_to(_at(ctx, max_hp), _A, 100))
        hp_range = hp_range | t
        ctx, t = _split(c.if_random_less_than(ctx, 128, r128))
        ctx = c.add_to_move_score(ctx, 2)
        # <label>_CheckUserHPRange
        ctx = _at(ctx, hp_range | t | ctx.live)
        ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 70))  # -> _End
        ctx, minus2 = _split(c.if_hp_percent_less_than(ctx, _A, 40))
        ctx, _ = _split(c.if_random_less_than(ctx, n_end, r_end))  # -> _End
        # <label>_ScoreMinus2, then _End
        ctx = c.add_to_move_score(_at(ctx, minus2 | ctx.live), -2)
        return c.pop_or_end(ctx)

    check.__name__ = check.__qualname__ = f"_expert_status_{stat.name.lower()}_up"
    check.__doc__ = f"``{label}`` (script.s:{start})."
    return check


def _status_defense_up(label: str, stat: Stat, lines: tuple[int, ...], minus2_class: int):
    """``Expert_StatusDefenseUp`` (script.s:2009-2052) / ``Expert_StatusSpDefenseUp``
    (:2112-2155), at ``lines`` (label, its 100, 128, 200, first 60 and second 60 rolls): the
    +3-or-more -1 and the 100 %-HP +2 as in Expert_StatusAttackUp; then at 70 % HP or more
    nothing more when the 200 roll jumps; below 40 % -2; else the player's previous move
    (``LoadBattlerPreviousMove``) with power 0 goes to the second 60 roll, one of class
    ``minus2_class`` (special for Defense, physical for Sp. Def) to -2, any other to the
    first 60 roll, which falls into the second when it does not jump; -2 unless a 60 roll
    jumps."""
    start, r100, r128, r200, r60a, r60b = lines

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = _at(ctx, mask)
        ctx, max_hp = _split(c.if_stat_stage_less_than(ctx, _A, stat, 9))
        ctx, t = _split(c.if_random_less_than(ctx, 100, r100))
        ctx = c.add_to_move_score(ctx, -1)
        ctx, g = _split(c.go_to(ctx))
        high_hp = t | g
        # <label>_CheckUserAtMaxHP
        ctx, t = _split(c.if_hp_percent_not_equal_to(_at(ctx, max_hp), _A, 100))
        high_hp = high_hp | t
        ctx, t = _split(c.if_random_less_than(ctx, 128, r128))
        ctx = c.add_to_move_score(ctx, 2)
        # <label>_CheckUserHighHP
        ctx = _at(ctx, high_hp | t | ctx.live)
        ctx, medium = _split(c.if_hp_percent_less_than(ctx, _A, 70))
        ctx, _ = _split(c.if_random_less_than(ctx, 200, r200))  # -> _End
        # <label>_CheckUserMediumHP
        ctx = _at(ctx, medium | ctx.live)
        ctx, minus2 = _split(c.if_hp_percent_less_than(ctx, _A, 40))
        ctx = c.load_power_of_loaded_move(c.load_battler_previous_move(ctx, _D))
        ctx, low_hp = _split(c.if_loaded_equal_to(ctx, 0))
        ctx = c.load_defender_last_used_move_class(ctx)
        ctx, t = _split(c.if_loaded_equal_to(ctx, minus2_class))
        minus2 = minus2 | t
        ctx, _ = _split(c.if_random_less_than(ctx, 60, r60a))  # -> _End
        # <label>_UserAtLowHP
        ctx, _ = _split(c.if_random_less_than(_at(ctx, low_hp | ctx.live), 60, r60b))
        # <label>_ScoreMinus2 (_TryScoreMinus2 for Sp. Def), then _End
        ctx = c.add_to_move_score(_at(ctx, minus2 | ctx.live), -2)
        return c.pop_or_end(ctx)

    check.__name__ = check.__qualname__ = f"_expert_status_{stat.name.lower()}_up"
    check.__doc__ = f"``{label}`` (script.s:{start})."
    return check


_expert_status_atk_up = _status_attack_up(
    "Expert_StatusAttackUp", Stat.ATK, (1978, 1989, 1995, 2001), 40
)
_expert_status_spa_up = _status_attack_up(
    "Expert_StatusSpAttackUp", Stat.SPA, (2081, 2092, 2098, 2104), 70
)
_expert_status_def_up = _status_defense_up(
    "Expert_StatusDefenseUp", Stat.DEF, (2009, 2023, 2029, 2034, 2043, 2046), c.CLASS_SPECIAL
)
_expert_status_spd_up = _status_defense_up(
    "Expert_StatusSpDefenseUp", Stat.SPD, (2112, 2126, 2132, 2137, 2146, 2149), c.CLASS_PHYSICAL
)


def _expert_status_sleep(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusSleep`` (script.s:1813-1825): +1 unless the 128 roll (:1821) jumps when
    the attacker knows Dream Eater's or Nightmare's effect."""
    ctx = _at(ctx, mask)
    ctx, plus1 = _split(c.if_move_effect_known(ctx, _A, BattleEffect.RECOVER_DAMAGE_SLEEP))
    ctx, t = _split(c.if_move_effect_known(ctx, _A, BattleEffect.STATUS_NIGHTMARE))
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus1 | t), 128, 1821))
    return c.pop_or_end(c.add_to_move_score(ctx, 1))


def _status_confuse(ctx: AiContext) -> AiContext:
    """``Expert_StatusConfuse`` (script.s:2861-2877) from the live paths: the player above
    70 % HP, nothing; else -1 unless the 128 roll (:2864) jumps, then -1 at 50 % or less and
    -1 more at 30 % or less."""
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 70))  # -> _End
    ctx, t = _split(c.if_random_less_than(ctx, 128, 2864))
    ctx = c.add_to_move_score(ctx, -1)
    # Expert_StatusConfuse_CheckHP (:2867)
    ctx = _at(ctx, t | ctx.live)
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 50))
    ctx = c.add_to_move_score(ctx, -1)
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 30))
    return c.pop_or_end(c.add_to_move_score(ctx, -1))


def _expert_status_confuse(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusConfuse`` (script.s:2861-2877)."""
    return _status_confuse(_at(ctx, mask))


def _expert_swagger(ctx: AiContext, mask) -> AiContext:
    """``Expert_Swagger`` (script.s:2845-2891): with Psych Up known, -5 when the player's
    Attack stage is above decomp 3, else +3 and +2 more on the battle's first turn
    (``Expert_Swagger_PsychUp``, :2879-2891); otherwise it falls into ``Expert_Flatter``
    (:2854): +1 unless the 128 roll (:2858) jumps, then ``Expert_StatusConfuse``."""
    ctx = _at(ctx, mask)
    ctx, psych_up = _split(c.if_move_known(ctx, _A, _MOVE["Psych Up"]))
    # Expert_Flatter (:2854)
    ctx, t = _split(c.if_random_less_than(ctx, 128, 2858))
    ctx = c.add_to_move_score(ctx, 1)
    ctx = _status_confuse(_at(ctx, t | ctx.live))
    # Expert_Swagger_PsychUp (:2879)
    ctx = _at(ctx, psych_up)
    ctx, minus5 = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.ATK, 3))
    ctx = c.add_to_move_score(ctx, 3)
    ctx = c.load_turn_count(ctx)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, 0))  # -> _End
    ctx = c.add_to_move_score(ctx, 2)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Swagger_ScoreMinus5 (:2887), falling into _End
    ctx = c.add_to_move_score(_at(ctx, minus5), -5)
    return c.pop_or_end(ctx)


def _expert_status_paralyze(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusParalyze`` (script.s:2940-2954): slower than the player, +3 unless the
    20 roll (:2950) jumps; otherwise -1 at 70 % HP or less."""
    ctx = _at(ctx, mask)
    ctx, plus3 = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 70))  # -> _End
    ctx = c.add_to_move_score(ctx, -1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus3), 20, 2950))
    return c.pop_or_end(c.add_to_move_score(ctx, 3))


def _expert_bypass_accuracy(ctx: AiContext, mask) -> AiContext:
    """``Expert_BypassAccuracyMove`` (script.s:2256-2275): +1, and +1 more unless the 100 roll
    (:2271) jumps, when the player's evasion is above decomp 10 or the attacker's accuracy
    below 2; only the roll's +1 when evasion is above 8 or accuracy below 4."""
    ctx = _at(ctx, mask)
    ctx, plus1 = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.EVA, 10))
    ctx, t = _split(c.if_stat_stage_less_than(ctx, _A, Stat.ACC, 2))
    plus1 = plus1 | t
    ctx, roll = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.EVA, 8))
    ctx, t = _split(c.if_stat_stage_less_than(ctx, _A, Stat.ACC, 4))
    roll = roll | t
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_BypassAccuracyMove_ScorePlus1 (:2267), falling into _TryScorePlus1 (:2270)
    ctx = c.add_to_move_score(_at(ctx, plus1), 1)
    ctx, _ = _split(c.if_random_less_than(_at(ctx, roll | ctx.live), 100, 2271))
    return c.pop_or_end(c.add_to_move_score(ctx, 1))


def _expert_high_critical(ctx: AiContext, mask) -> AiContext:
    """``Expert_HighCritical`` (script.s:2827-2843): resisted, nothing; super effective, +1
    unless the 128 roll (:2839) jumps; neutral, the same after a first 128 roll (:2836) that
    ends it when it jumps."""
    ctx, _ = _not_resisted(_at(ctx, mask))  # -> _End
    ctx, plus1 = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.DOUBLE))
    ctx, t = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.QUADRUPLE))
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 2836))  # -> _End
    # Expert_HighCritical_TryScorePlus1 (:2838)
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus1 | t | ctx.live), 128, 2839))
    return c.pop_or_end(c.add_to_move_score(ctx, 1))


def _expert_speed_down_on_hit(ctx: AiContext, mask) -> AiContext:
    """``Expert_SpeedDownOnHit`` (script.s:2338-2350): resisted, nothing; Icy Wind, Rock Tomb
    and Mud Shot go to ``Expert_StatusSpeedDown`` (:2353-2366: not slower than the player -3;
    slower, +2 unless the 70 roll, :2362, jumps); any other move ends."""
    ctx, _ = _not_resisted(_at(ctx, mask))  # -> _End
    down = jnp.zeros_like(ctx.live)
    for name in ("Icy Wind", "Rock Tomb", "Mud Shot"):
        ctx, t = _split(c.if_move_equal_to(ctx, _MOVE[name]))
        down = down | t
    ctx = c.pop_or_end(ctx)
    return _status_speed_down(_at(ctx, down))  # Expert_StatusSpeedDown (:2353)


def _expert_facade(ctx: AiContext, mask) -> AiContext:
    """``Expert_Facade`` (script.s:4120-4127): +1 when the player (the defender, as the
    decomp has it) is poisoned, badly poisoned, burned or paralyzed (FACADE_BOOST)."""
    ctx = _at(ctx, mask)
    ctx, _ = _split(c.if_not_status(ctx, _D, c.MonCondition.FACADE_BOOST))  # -> _End
    return c.pop_or_end(c.add_to_move_score(ctx, 1))


def _expert_smelling_salts(ctx: AiContext, mask) -> AiContext:
    """``Expert_SmellingSalts`` (script.s:4165-4174): +1 when the player is paralyzed."""
    ctx = _at(ctx, mask)
    ctx, plus1 = _split(c.if_status(ctx, _D, c.MonCondition.PARALYSIS))
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    return c.pop_or_end(c.add_to_move_score(_at(ctx, plus1), 1))


def _expert_brine(ctx: AiContext, mask) -> AiContext:
    """``Expert_Brine`` (script.s:4873-4890): resisted (immune, half, quarter) -1; else, with
    the player at 50 % HP or less, +1 and +1 more unless the 128 roll (:4882) jumps."""
    ctx = _at(ctx, mask)
    ctx, minus1 = _split(c.if_move_effectiveness_equals(ctx, TypeMulti.IMMUNE))
    for cls in (TypeMulti.HALF, TypeMulti.QUARTER):
        ctx, t = _split(c.if_move_effectiveness_equals(ctx, cls))
        minus1 = minus1 | t
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 50))  # -> _End
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 4882))  # -> _End
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Brine_ScoreMinus1 (:4886)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_destiny_bond(ctx: AiContext, mask) -> AiContext:
    """``Expert_DestinyBond`` (script.s:3335-3360): -1; slower than the player, nothing more;
    otherwise at 70 % HP or less +1 unless the 128 roll (:3346) jumps, at 50 % or less +1 more
    unless :3351 jumps, at 30 % or less +2 more unless the 100 roll (:3356) jumps."""
    ctx = c.add_to_move_score(_at(ctx, mask), -1)
    ctx, _ = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))  # -> _End
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 70))  # -> _End
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3346))
    ctx = c.add_to_move_score(ctx, 1)
    # Expert_DestinyBond_CheckUserMediumHP (:3349)
    ctx, _ = _split(c.if_hp_percent_greater_than(_at(ctx, t | ctx.live), _A, 50))
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3351))
    ctx = c.add_to_move_score(ctx, 1)
    # Expert_DestinyBond_CheckUserLowHP (:3354)
    ctx, _ = _split(c.if_hp_percent_greater_than(_at(ctx, t | ctx.live), _A, 30))
    ctx, _ = _split(c.if_random_less_than(ctx, 100, 3356))
    return c.pop_or_end(c.add_to_move_score(ctx, 2))


def _hazard_expert(label: str, lines: tuple[int, int, int]):
    """``Expert_Spikes`` (script.s:3591-3606) / ``Expert_ToxicSpikes`` (:6012-6029), at
    ``lines`` (label, its 128 and 64 rolls): +1 unless the 128 roll jumps; then, with Roar or
    Whirlwind known, +1 more unless the 64 roll jumps."""
    start, r128, r64 = lines

    def check(ctx: AiContext, mask) -> AiContext:
        ctx, _ = _split(c.if_random_less_than(_at(ctx, mask), 128, r128))  # -> _End
        ctx = c.add_to_move_score(ctx, 1)
        ctx, plus1 = _split(c.if_move_known(ctx, _A, _MOVE["Roar"]))
        ctx, t = _split(c.if_move_known(ctx, _A, _MOVE["Whirlwind"]))
        ctx, _ = _split(c.go_to(ctx))  # -> _End
        # <label>_TryScorePlus1
        ctx, _ = _split(c.if_random_less_than(_at(ctx, plus1 | t), 64, r64))
        return c.pop_or_end(c.add_to_move_score(ctx, 1))

    check.__name__ = check.__qualname__ = f"_expert_{label.removeprefix('Expert_').lower()}"
    check.__doc__ = f"``{label}`` (script.s:{start})."
    return check


_expert_spikes = _hazard_expert("Expert_Spikes", (3591, 3595, 3602))
_expert_toxic_spikes = _hazard_expert("Expert_ToxicSpikes", (6012, 6018, 6025))


def _expert_dragon_dance(ctx: AiContext, mask) -> AiContext:
    """``Expert_DragonDance`` (script.s:4765-4780): slower than the player, +1 unless the 128
    roll (:4776) jumps; else at 50 % HP or less, -1 unless the 70 roll (:4771) jumps."""
    ctx = _at(ctx, mask)
    ctx, plus1 = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 50))  # -> _End
    ctx, _ = _split(c.if_random_less_than(ctx, 70, 4771))  # -> _End
    ctx = c.add_to_move_score(ctx, -1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus1), 128, 4776))
    return c.pop_or_end(c.add_to_move_score(ctx, 1))


def _status_speed_down(ctx: AiContext) -> AiContext:
    """``Expert_StatusSpeedDown`` (script.s:2353-2366) from the live paths: not slower than
    the player, -3; slower, +2 unless the 70 roll (:2362) jumps."""
    ctx, plus2 = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx = c.add_to_move_score(ctx, -3)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    ctx, _ = _split(c.if_random_less_than(_at(ctx, plus2), 70, 2362))
    return c.pop_or_end(c.add_to_move_score(ctx, 2))


def _expert_status_speed_down(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusSpeedDown`` (script.s:2353-2366)."""
    return _status_speed_down(_at(ctx, mask))


def _expert_status_attack_down(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusAttackDown`` (script.s:2277-2307): the player not at +0 Attack: -1, -1
    more at 90 % HP or less, then -2 unless the 50 roll (:2293) jumps when the player's
    Attack is at decomp 3 or less; then -2 at the player's 70 % HP or less; then, when the
    player's last move was special, -2 unless the 128 roll (:2303) jumps."""
    ctx = _at(ctx, mask)
    ctx, target_hp = _split(c.if_stat_stage_equal_to(ctx, _D, Stat.ATK, 6))
    ctx = c.add_to_move_score(ctx, -1)
    ctx, stage = _split(c.if_hp_percent_greater_than(ctx, _A, 90))
    ctx = c.add_to_move_score(ctx, -1)
    # Expert_StatusAttackDown_CheckTargetStatStage (:2291)
    ctx = _at(ctx, stage | ctx.live)
    ctx, t = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.ATK, 3))
    target_hp = target_hp | t
    ctx, t = _split(c.if_random_less_than(ctx, 50, 2293))
    ctx = c.add_to_move_score(ctx, -2)
    # Expert_StatusAttackDown_CheckTargetHP (:2296)
    ctx = _at(ctx, target_hp | t | ctx.live)
    ctx, last = _split(c.if_hp_percent_greater_than(ctx, _D, 70))
    ctx = c.add_to_move_score(ctx, -2)
    # Expert_StatusAttackDown_CheckLastUsedMove (:2300)
    ctx = c.load_defender_last_used_move_class(_at(ctx, last | ctx.live))
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, c.CLASS_SPECIAL))  # -> _End
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 2303))  # -> _End
    return c.pop_or_end(c.add_to_move_score(ctx, -2))


def _status_down_hp(label: str, stat: Stat, lines: tuple[int, int]):
    """``Expert_StatusSpDefenseDown`` (script.s:2411-2429) / ``Expert_StatusEvasionDown``
    (:2500-2518), at ``lines`` (label, its 50 roll): at 70 % HP or more and the player's stage
    above decomp 3, straight to the target's HP; else -2 unless the 50 roll jumps; then -2 at
    the player's 70 % HP or less (Roark's Expert_StatusDefenseDown's shape)."""
    start, r50 = lines

    def check(ctx: AiContext, mask) -> AiContext:
        ctx = _at(ctx, mask)
        ctx, roll = _split(c.if_hp_percent_less_than(ctx, _A, 70))
        ctx, target_hp = _split(c.if_stat_stage_greater_than(ctx, _D, stat, 3))
        # <label>_TryScoreMinus2
        ctx = _at(ctx, roll | ctx.live)
        ctx, t = _split(c.if_random_less_than(ctx, 50, r50))
        ctx = c.add_to_move_score(ctx, -2)
        # <label>_CheckTargetHP
        ctx = _at(ctx, target_hp | t | ctx.live)
        ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 70))
        return c.pop_or_end(c.add_to_move_score(ctx, -2))

    check.__name__ = check.__qualname__ = f"_expert_status_{stat.name.lower()}_down"
    check.__doc__ = f"``{label}`` (script.s:{start})."
    return check


_expert_status_spd_down = _status_down_hp("Expert_StatusSpDefenseDown", Stat.SPD, (2411, 2421))
_expert_status_eva_down = _status_down_hp("Expert_StatusEvasionDown", Stat.EVA, (2500, 2510))


def _expert_reflect(ctx: AiContext, mask) -> AiContext:
    """``Expert_Reflect`` (script.s:2893-2915): below 50 % HP -2; else at 90 % HP or more +1
    unless the 128 roll (:2901) jumps; then, when the player's last move was physical, +1
    unless the 64 roll (:2907) jumps."""
    ctx = _at(ctx, mask)
    ctx, minus2 = _split(c.if_hp_percent_less_than(ctx, _A, 50))
    ctx, last = _split(c.if_hp_percent_less_than(ctx, _A, 90))
    ctx, t = _split(c.if_random_less_than(ctx, 128, 2901))
    ctx = c.add_to_move_score(ctx, 1)
    # Expert_Reflect_CheckLastUsedMove (:2904)
    ctx = c.load_defender_last_used_move_class(_at(ctx, last | t | ctx.live))
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, c.CLASS_PHYSICAL))  # -> _End
    ctx, _ = _split(c.if_random_less_than(ctx, 64, 2907))  # -> _End
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Reflect_ScoreMinus2 (:2911)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus2), -2))


def _expert_tailwind(ctx: AiContext, mask) -> AiContext:
    """``Expert_Tailwind`` (script.s:4982-5006): nothing when the 64 roll (:4992) jumps; then
    faster than the player (its tie a read, E1 Ruling R8) or below 31 % HP, -1; above 75 %,
    +1; between, +1 unless the 64 roll (:4996) jumps."""
    ctx = _at(ctx, mask)
    ctx, _ = _split(c.if_random_less_than(ctx, 64, 4992))  # -> _End
    ctx, minus1 = _split(c.if_speed_compare_equal_to(ctx, c.COMPARE_SPEED_FASTER))
    ctx, t = _split(c.if_hp_percent_less_than(ctx, _A, 31))
    minus1 = minus1 | t
    ctx, plus1 = _split(c.if_hp_percent_greater_than(ctx, _A, 75))
    ctx, _ = _split(c.if_random_less_than(ctx, 64, 4996))  # -> _End
    # Expert_Tailwind_ScorePlus1 (:4998)
    ctx = c.add_to_move_score(_at(ctx, plus1 | ctx.live), 1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Tailwind_ScoreMinus1 (:5002), falling into _End
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_payback(ctx: AiContext, mask) -> AiContext:
    """``Expert_Payback`` (script.s:5169-5187): resisted -1; else faster than the player (its
    tie a read, E1 Ruling R8) or below 30 % HP, nothing; else +1 unless the 64 roll (:5179)
    jumps."""
    ctx, minus1 = _not_resisted(_at(ctx, mask))
    ctx, _ = _split(c.if_speed_compare_equal_to(ctx, c.COMPARE_SPEED_FASTER))  # -> _End
    ctx, _ = _split(c.if_hp_percent_less_than(ctx, _A, 30))  # -> _End
    ctx, _ = _split(c.if_random_less_than(ctx, 64, 5179))  # -> _End
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 1))
    # Expert_Payback_ScoreMinus1 (:5183)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_plus(value: int):
    """``Expert_Nightmare`` (script.s:3314-3316) and ``Expert_FakeOut`` (:4074-4076):
    ``AddToMoveScore value``."""

    def label(ctx: AiContext, mask) -> AiContext:
        return c.pop_or_end(c.add_to_move_score(_at(ctx, mask), value))

    return label


def _expert_sleep_talk(ctx: AiContext, mask) -> AiContext:
    """``Expert_SleepTalk`` (script.s:3327-3333): asleep +10, else -5."""
    ctx, t = _split(c.if_status(_at(ctx, mask), _A, c.MonCondition.SLEEP))
    ctx = _score(ctx, t, 10)
    return c.pop_or_end(c.add_to_move_score(ctx, -5))


def _expert_super_fang(ctx: AiContext, mask) -> AiContext:
    """``Expert_SuperFang`` (script.s:2800-2806): the player at 50 % HP or less, -1."""
    ctx, _ = _split(c.if_hp_percent_greater_than(_at(ctx, mask), _D, 50))  # -> _End
    return c.pop_or_end(c.add_to_move_score(ctx, -1))


def _expert_reversal(ctx: AiContext, mask) -> AiContext:
    """``Expert_Reversal`` (script.s:3362-3395): faster or tied, above 33 % HP -1, above 20 %
    nothing, below 8 % +1; slower, above 60 % -1, above 40 % nothing; then +1 unless the 100
    roll (:3388) jumps."""
    ctx = _at(ctx, mask)
    ctx, slower = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx, minus1 = _split(c.if_hp_percent_greater_than(ctx, _A, 33))
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 20))  # -> _End
    ctx, plus1 = _split(c.if_hp_percent_less_than(ctx, _A, 8))
    ctx, try_plus1 = _split(c.go_to(ctx))
    # Expert_Reversal_SlowerCheckHP (:3379)
    ctx = _at(ctx, slower)
    ctx, t = _split(c.if_hp_percent_greater_than(ctx, _A, 60))
    minus1 = minus1 | t
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _A, 40))  # -> _End
    ctx, t = _split(c.go_to(ctx))
    try_plus1 = try_plus1 | t
    # Expert_Reversal_ScorePlus1 (:3384), falling into _TryScorePlus1 (:3387)
    ctx = c.add_to_move_score(_at(ctx, plus1), 1)
    ctx = _at(ctx, try_plus1 | ctx.live)
    ctx, _ = _split(c.if_random_less_than(ctx, 100, 3388))  # -> _End
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Reversal_ScoreMinus1 (:3392)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _counter_like(label: str, other: str, cls: int, types: tuple[int, ...], lines):
    """``Expert_Counter`` (script.s:3095-3163) and ``Expert_MirrorCoat`` (:3852-3920), the same
    shape: the player asleep, infatuated or confused -1; at 30 % HP or less -1 unless the 10
    roll jumps; at 50 % or less -1 unless the 100 roll jumps; knowing the other move, +4
    unless its 100 roll jumps; else with the player's last move a damaging one, +1 unless
    not taunted or its roll jumps, then its class not ``cls`` -1, else +1 unless its roll
    jumps; with no damaging last move, +1 unless not taunted or its roll jumps, then a player
    of a type in ``types`` nothing, else unless the 50 roll jumps, the +4 try. ``lines`` are
    the rolls in line order."""
    low, half, last, cls_roll, no_last, types_roll, plus4_line = lines

    def run(ctx: AiContext, mask) -> AiContext:
        ctx = _at(ctx, mask)
        ctx, minus1 = _split(c.if_status(ctx, _D, c.MonCondition.SLEEP))
        for cond in (c.VolatileCondition.ATTRACT, c.VolatileCondition.CONFUSION):
            ctx, t = _split(c.if_volatile_status(ctx, _D, cond))
            minus1 = minus1 | t
        ctx, above = _split(c.if_hp_percent_greater_than(ctx, _A, 30))
        ctx, t = _split(c.if_random_less_than(ctx, 10, low))
        ctx = c.add_to_move_score(ctx, -1)
        # _CheckAboveHalfHP
        ctx = _at(ctx, above | t | ctx.live)
        ctx, check_last = _split(c.if_hp_percent_greater_than(ctx, _A, 50))
        ctx, t = _split(c.if_random_less_than(ctx, 100, half))
        ctx = c.add_to_move_score(ctx, -1)
        # _CheckLastUsedMove
        ctx = _at(ctx, check_last | t | ctx.live)
        ctx, plus4 = _split(c.if_move_known(ctx, _A, _MOVE[other]))
        ctx = c.load_power_of_loaded_move(c.load_battler_previous_move(ctx, _D))
        ctx, try_plus1 = _split(c.if_loaded_equal_to(ctx, 0))
        ctx, check_class = _split(c.if_target_is_not_taunted(ctx))
        ctx, t = _split(c.if_random_less_than(ctx, 100, last))
        ctx = c.add_to_move_score(ctx, 1)
        # _CheckPhysicalMove / _CheckSpecialMove
        ctx = c.load_defender_last_used_move_class(_at(ctx, check_class | t | ctx.live))
        ctx, t = _split(c.if_loaded_not_equal_to(ctx, cls))
        minus1 = minus1 | t
        ctx, _ = _split(c.if_random_less_than(ctx, 100, cls_roll))  # -> _End2
        ctx = c.add_to_move_score(ctx, 1)
        ctx, _ = _split(c.go_to(ctx))  # -> _End2
        # _TryScorePlus1
        ctx = _at(ctx, try_plus1)
        ctx, check_types = _split(c.if_target_is_not_taunted(ctx))
        ctx, t = _split(c.if_random_less_than(ctx, 100, no_last))
        ctx = c.add_to_move_score(ctx, 1)
        # _CheckOpponentTypes, falling into _TryScorePlus4
        ctx = _at(ctx, check_types | t | ctx.live)
        for target in (LoadType.DEFENDER_TYPE_1, LoadType.DEFENDER_TYPE_2):
            ctx = c.load_type_from(ctx, target)
            ctx, _ = _split(c.if_loaded_in_table(ctx, types))  # -> _End2
        ctx, _ = _split(c.if_random_less_than(ctx, 50, types_roll))  # -> _End2
        # _TryScorePlus4
        ctx = _at(ctx, plus4 | ctx.live)
        ctx, _ = _split(c.if_random_less_than(ctx, 100, plus4_line))  # -> _End
        ctx = c.pop_or_end(c.add_to_move_score(ctx, 4))
        # _ScoreMinus1, falling into _End2
        return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))

    run.__name__ = run.__qualname__ = label
    run.__doc__ = f"``{label}`` (script.s; see ``_counter_like``)."
    return run


_expert_counter = _counter_like(
    "Expert_Counter",
    "Mirror Coat",
    c.CLASS_PHYSICAL,
    c.COUNTER_PHYSICAL_TYPES,
    (3116, 3121, 3130, 3136, 3142, 3150, 3153),
)
_expert_mirror_coat = _counter_like(
    "Expert_MirrorCoat",
    "Counter",
    c.CLASS_SPECIAL,
    c.MIRROR_COAT_SPECIAL_TYPES,
    (3873, 3878, 3887, 3893, 3899, 3907, 3910),
)


def _expert_superpower(ctx: AiContext, mask) -> AiContext:
    """``Expert_Superpower`` (script.s:4467-4488): resisted, or the attacker's Attack below +0,
    -1; faster or tied above 40 % HP -1; slower at 60 % HP or more -1."""
    ctx, minus1 = _not_resisted(_at(ctx, mask))
    ctx, t = _split(c.if_stat_stage_less_than(ctx, _A, Stat.ATK, 6))
    minus1 = minus1 | t
    ctx, slower = _split(c.if_speed_compare_equal_to(ctx, COMPARE_SPEED_SLOWER))
    ctx, t = _split(c.if_hp_percent_greater_than(ctx, _A, 40))
    minus1 = minus1 | t
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Superpower_CheckUserHP (:4483), falling into _ScoreMinus1
    ctx, _ = _split(c.if_hp_percent_less_than(_at(ctx, slower), _A, 60))  # -> _End
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1 | ctx.live), -1))


def _expert_rain_dance(ctx: AiContext, mask) -> AiContext:
    """``Expert_RainDance`` (script.s:3737-3771): slower or tied (FASTER's tie a read, E1
    Ruling R8) with Swift Swim +1; else below 40 % HP -1; in hail, sun or sand +1; with Rain
    Dish +1; with Hydration and a status +1."""
    ctx = _at(ctx, mask)
    ctx, other = _split(c.if_speed_compare_equal_to(ctx, c.COMPARE_SPEED_FASTER))
    ctx = c.load_battler_ability(ctx, _A)
    ctx, plus1 = _split(c.if_loaded_equal_to(ctx, _ABILITY["Swift Swim"]))
    # Expert_RainDance_OtherChecks (:3751)
    ctx = _at(ctx, other | ctx.live)
    ctx, minus1 = _split(c.if_hp_percent_less_than(ctx, _A, 40))
    ctx = c.load_current_weather(ctx)
    for weather in (c.AiWeather.HAILING, c.AiWeather.SUNNY, c.AiWeather.SANDSTORM):
        ctx, t = _split(c.if_loaded_equal_to(ctx, int(weather)))
        plus1 = plus1 | t
    ctx = c.load_battler_ability(ctx, _A)
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Rain Dish"]))
    plus1 = plus1 | t
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Hydration"]))  # -> _End
    ctx, t = _split(c.if_status(ctx, _A, c.MonCondition.ANY))
    plus1 = plus1 | t
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_RainDance_ScorePlus1 (:3763)
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, plus1), 1))
    # Expert_RainDance_ScoreMinus1 (:3767)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_hail(ctx: AiContext, mask) -> AiContext:
    """``Expert_Hail`` (script.s:4089-4119): below 40 % HP -1; in sun, rain or sand +1, +2 more
    knowing Blizzard, +2 more with Ice Body."""
    ctx = _at(ctx, mask)
    ctx, minus1 = _split(c.if_hp_percent_less_than(ctx, _A, 40))
    ctx = c.load_current_weather(ctx)
    plus1 = jnp.zeros_like(ctx.live)
    for weather in (c.AiWeather.SUNNY, c.AiWeather.RAINING, c.AiWeather.SANDSTORM):
        ctx, t = _split(c.if_loaded_equal_to(ctx, int(weather)))
        plus1 = plus1 | t
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Hail_ScorePlus1AndCheckBlizzard (:4103)
    ctx = c.add_to_move_score(_at(ctx, plus1), 1)
    ctx, ice_body = _split(c.if_move_not_known(ctx, _A, _MOVE["Blizzard"]))
    ctx = c.add_to_move_score(ctx, 2)
    # Expert_Hail_CheckIceBody (:4108)
    ctx = c.load_battler_ability(_at(ctx, ice_body | ctx.live), _A)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, _ABILITY["Ice Body"]))  # -> _End
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 2))
    # Expert_Hail_ScoreMinus1 (:4114)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_pluck(ctx: AiContext, mask) -> AiContext:
    """``Expert_Pluck`` (script.s:4957-4980): resisted -1; on the attacker's first turn in
    battle +1 unless the 64 roll (:4968) jumps; then +1 unless the 128 roll (:4972) jumps."""
    ctx, minus1 = _not_resisted(_at(ctx, mask))
    ctx = c.load_is_first_turn_in_battle(ctx, _A)
    ctx, try_plus1 = _split(c.if_loaded_equal_to(ctx, 0))
    ctx, t = _split(c.if_random_less_than(ctx, 64, 4968))
    ctx = c.add_to_move_score(ctx, 1)
    # Expert_Pluck_TryScorePlus1 (:4971)
    ctx = _at(ctx, try_plus1 | t | ctx.live)
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 4972))  # -> _End
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 1))
    # Expert_Pluck_ScoreMinus1 (:4976)
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus1), -1))


def _expert_protect(ctx: AiContext, mask) -> AiContext:
    """``Expert_Protect`` (script.s:3502-3589): a player knowing Feint or Shadow Force, -2
    unless the 128 roll (:3541) jumps; a chain above 1, -2; the AI badly poisoned, cursed,
    perishing, infatuated, seeded or drowsy, or the player knowing a half-HP recovery or
    Defense Curl, -2 unless the AI has used Lock-On; the player under one of those six, in a
    doubles battle, the AI's Lock-On, or the 85 roll (:3564), +2; then -1 unless the 128 roll
    (:3571) jumps; then with a chain, -1 and -1 more unless :3578 jumps."""
    ctx = _at(ctx, mask)
    ctx, try_minus2 = _split(c.if_move_known(ctx, _D, _MOVE["Feint"]))
    ctx, t = _split(c.if_move_known(ctx, _D, _MOVE["Shadow Force"]))
    try_minus2 = try_minus2 | t
    ctx, check_status = _split(c.go_to(ctx))
    # Expert_Protect_TryScoreMinus2 (:3540)
    ctx = _at(ctx, try_minus2)
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3541))
    ctx = c.add_to_move_score(ctx, -2)
    # Expert_Protect_CheckStatusConditions (:3544); the first GoTo's paths join here
    ctx = _at(ctx, check_status | t | ctx.live)
    ctx = c.load_protect_chain(ctx, _A)
    ctx, minus2 = _split(c.if_loaded_greater_than(ctx, 1))
    locked = jnp.zeros_like(ctx.live)
    for check in (
        lambda x: c.if_status(x, _A, c.MonCondition.TOXIC),
        lambda x: c.if_volatile_status(x, _A, c.VolatileCondition.CURSE),
        lambda x: c.if_move_effect(x, _A, c.MoveEffectMask.PERISH_SONG),
        lambda x: c.if_volatile_status(x, _A, c.VolatileCondition.ATTRACT),
        lambda x: c.if_move_effect(x, _A, c.MoveEffectMask.LEECH_SEED),
        lambda x: c.if_move_effect(x, _A, c.MoveEffectMask.YAWN),
        lambda x: c.if_move_effect_known(x, _D, BattleEffect.RESTORE_HALF_HP),
        lambda x: c.if_move_effect_known(x, _D, BattleEffect.DEF_UP_DOUBLE_ROLLOUT_POWER),
    ):
        ctx, t = _split(check(ctx))
        locked = locked | t
    plus2 = jnp.zeros_like(ctx.live)
    for check in (
        lambda x: c.if_status(x, _D, c.MonCondition.TOXIC),
        lambda x: c.if_volatile_status(x, _D, c.VolatileCondition.CURSE),
        lambda x: c.if_move_effect(x, _D, c.MoveEffectMask.PERISH_SONG),
        lambda x: c.if_volatile_status(x, _D, c.VolatileCondition.ATTRACT),
        lambda x: c.if_move_effect(x, _D, c.MoveEffectMask.LEECH_SEED),
        lambda x: c.if_move_effect(x, _D, c.MoveEffectMask.YAWN),
        lambda x: c.if_loaded_mask(c.load_battle_type(x), c.BATTLE_TYPE_DOUBLES),
        lambda x: c.if_move_effect(x, _A, c.MoveEffectMask.LOCK_ON),
        lambda x: c.if_random_less_than(x, 85, 3564),
    ):
        ctx, t = _split(check(ctx))
        plus2 = plus2 | t
    ctx, try_minus1 = _split(c.go_to(ctx))
    # Expert_Protect_ScorePlus2 (:3567), falling into _TryScoreMinus1 (:3570)
    ctx = c.add_to_move_score(_at(ctx, plus2), 2)
    ctx = _at(ctx, try_minus1 | ctx.live)
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3571))
    ctx = c.add_to_move_score(ctx, -1)
    # Expert_Protect_CheckEmptyChain (:3574)
    ctx = c.load_protect_chain(_at(ctx, t | ctx.live), _A)
    ctx, _ = _split(c.if_loaded_equal_to(ctx, 0))  # -> _End
    ctx = c.add_to_move_score(ctx, -1)
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 3578))  # -> _End
    ctx = c.add_to_move_score(ctx, -1)
    ctx, _ = _split(c.go_to(ctx))  # -> _End
    # Expert_Protect_CheckAttackerLockedOnto (:3582), falling into _ScoreMinus2 (:3585)
    ctx, _ = _split(c.if_move_effect(_at(ctx, locked), _A, c.MoveEffectMask.LOCK_ON))
    return c.pop_or_end(c.add_to_move_score(_at(ctx, minus2 | ctx.live), -2))


def _expert_curse(ctx: AiContext, mask) -> AiContext:
    """``Expert_Curse`` (script.s:3452-3500)."""
    # Expert_Curse (:3452). A Ghost-type attacker jumps to Expert_Curse_GhostCheckHP
    # (:3467-3470 -> :3495); _refused refuses Curse on a Ghost type (Basic_CheckCurse
    # _GhostType, script.s:700, is not transcribed), so those jumps are never taken.
    ctx = _at(ctx, mask)
    for target in (LoadType.ATTACKER_TYPE_1, LoadType.ATTACKER_TYPE_2):
        ctx = c.load_type_from(ctx, target)
        ctx, _ = _split(c.if_loaded_equal_to(ctx, _TYPE["Ghost"]))
    ctx, _ = _split(c.if_stat_stage_greater_than(ctx, _A, Stat.DEF, 9))  # -> Expert_Curse_End
    ctx, high = _split(c.if_move_known(ctx, _A, _MOVE["Gyro Ball"]))
    ctx, t = _split(c.if_move_known(ctx, _A, _MOVE["Trick Room"]))
    high = high | t
    ctx, flip = _split(c.go_to(ctx))  # :3474
    # Expert_Curse_HighChanceScorePlus1 (:3476): +1 unless the 32 roll jumps past the coin.
    ctx = _at(ctx, high)
    ctx, check = _split(c.if_random_less_than(ctx, 32, 3477))
    ctx = c.add_to_move_score(ctx, 1)
    flip = flip | ctx.live
    # Expert_Curse_FlipCoinScorePlus1 (:3480)
    ctx = _at(ctx, flip)
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3481))
    ctx = c.add_to_move_score(ctx, 1)
    check = check | t | ctx.live
    # Expert_Curse_CheckDefenseStage (:3484)
    ctx = _at(ctx, check)
    ctx, _ = _split(c.if_stat_stage_greater_than(ctx, _A, Stat.DEF, 7))
    ctx, any_boosts = _split(c.if_random_less_than(ctx, 128, 3486))
    ctx = c.add_to_move_score(ctx, 1)
    any_boosts = any_boosts | ctx.live
    # Expert_Curse_CheckDefenseStageAnyBoosts (:3489)
    ctx = _at(ctx, any_boosts)
    ctx, _ = _split(c.if_stat_stage_greater_than(ctx, _A, Stat.DEF, 6))
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 3491))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 1))
    return ctx


def _expert_brick_break(ctx: AiContext, mask) -> AiContext:
    """``Expert_BrickBreak`` (script.s:4559-4569)."""
    # Expert_BrickBreak (:4559): Reflect or Light Screen on the player's side, +1.
    ctx = _at(ctx, mask)
    ctx, plus1 = _split(c.if_side_condition(ctx, _D, c.SideCondition.REFLECT))
    ctx, t = _split(c.if_side_condition(ctx, _D, c.SideCondition.LIGHT_SCREEN))
    plus1 = plus1 | t
    ctx, _ = _split(c.go_to(ctx))
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, plus1), 1))
    return ctx


def _expert_recoil_move(ctx: AiContext, mask) -> AiContext:
    """``Expert_RecoilMove`` (script.s:6289-6305)."""
    # Expert_RecoilMove (:6289): not resisted and the attacker's true ability is Rock Head
    # or Magic Guard, +1.
    ctx, _ = _not_resisted(_at(ctx, mask))
    ctx = c.load_battler_ability(ctx, _A)
    ctx, plus1 = _split(c.if_loaded_equal_to(ctx, _ABILITY["Rock Head"]))
    ctx, t = _split(c.if_loaded_equal_to(ctx, _ABILITY["Magic Guard"]))
    plus1 = plus1 | t
    ctx, _ = _split(c.go_to(ctx))
    ctx = c.pop_or_end(c.add_to_move_score(_at(ctx, plus1), 1))
    return ctx


def _expert_status_defense_down(ctx: AiContext, mask) -> AiContext:
    """``Expert_StatusDefenseDown`` (script.s:2318-2336)."""
    # Expert_StatusDefenseDown (:2318)
    ctx = _at(ctx, mask)
    ctx, roll = _split(c.if_hp_percent_less_than(ctx, _A, 70))
    ctx, target_hp = _split(c.if_stat_stage_greater_than(ctx, _D, Stat.DEF, 3))
    roll = roll | ctx.live
    # Expert_StatusDefenseDown_TryScoreMinus2 (:2327)
    ctx = _at(ctx, roll)
    ctx, t = _split(c.if_random_less_than(ctx, 50, 2328))
    ctx = c.add_to_move_score(ctx, -2)
    target_hp = target_hp | t | ctx.live
    # Expert_StatusDefenseDown_CheckTargetHP (:2331)
    ctx = _at(ctx, target_hp)
    ctx, _ = _split(c.if_hp_percent_greater_than(ctx, _D, 70))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, -2))
    return ctx


def _expert_pursuit(ctx: AiContext, mask) -> AiContext:
    """``Expert_Pursuit`` (script.s:3706-3735)."""
    # Expert_Pursuit (:3706)
    ctx = c.load_is_first_turn_in_battle(_at(ctx, mask), _A)
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
    ctx, t = _split(c.if_random_less_than(ctx, 128, 3726))
    ctx = c.add_to_move_score(ctx, 1)
    u_turn = u_turn | t | ctx.live
    # Expert_Pursuit_CheckUturn (:3729)
    ctx = _at(ctx, u_turn)
    ctx, _ = _split(c.if_move_not_known(ctx, _D, _MOVE["U-turn"]))
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 3731))
    ctx = c.pop_or_end(c.add_to_move_score(ctx, 1))
    return ctx


def _expert_stealth_rock(ctx: AiContext, mask) -> AiContext:
    """``Expert_StealthRock`` (script.s:6268-6285)."""
    # Expert_StealthRock (:6268). Expert_StealthRock_TryScorePlus1 (:6280, a 64 roll) is
    # reached only when the attacker knows Roar or Whirlwind, which are not transcribed
    # moves (ai_refusals), so the jumps below are never taken.
    ctx = _at(ctx, mask)
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 6274))
    ctx = c.add_to_move_score(ctx, 1)
    ctx, _ = _split(c.if_move_known(ctx, _A, _MOVE["Roar"]))
    ctx, _ = _split(c.if_move_known(ctx, _A, _MOVE["Whirlwind"]))
    ctx, _ = _split(c.go_to(ctx))
    return ctx


# --- The jump tables -----------------------------------------------------------------


def _active_effects(layout: AiLayout) -> frozenset[int]:
    """The battle effects whose script paths ``layout`` transcribes: its ``transcribed`` set
    in a per-slot layout (keyed by effect), else the effects of its transcribed move ids
    (``commands.MOVE_EFFECTS``). A move with any other effect is refused (``_refused``)."""
    if layout.per_slot_sites:
        return frozenset(layout.transcribed)
    return frozenset(int(c.MOVE_EFFECTS[m]) for m in layout.transcribed)


def _dispatch(ctx: AiContext, labels, layout: AiLayout) -> AiContext:
    """A flag's effect jump table (``IfCurrentMoveEffectEqualTo effect, label`` in line
    order, then ``PopOrEnd``), then each label reached, under the union of the paths of its
    entries. The table holds only ``layout``'s effects (``_active_effects``), decided at trace
    time: a label no transcribed move reaches is not traced, so one layout's labels cost
    another nothing (K3c Task 6 review m1)."""
    active = _active_effects(layout)
    reached: dict = {}
    for effect, label in labels:
        if int(effect) in active:
            ctx, t = _split(c.if_current_move_effect_equal_to(ctx, effect))
            reached[label] = reached[label] | t if label in reached else t
    ctx = c.pop_or_end(ctx)
    for label, mask in reached.items():
        ctx = label(ctx, mask)
    return ctx


# Basic_ScoreMoveEffect's entries (script.s:134-285) of the transcribed effects, in line
# order. The other effects of the transcribed moves have no entry and end at :286.
_BASIC_LABELS = (
    (BattleEffect.STATUS_SLEEP, _basic_check_cannot_sleep),  # :134
    (BattleEffect.HALVE_DEFENSE, _basic_check_cannot_explode),  # :135
    (BattleEffect.ATK_UP, _basic_check_high_atk),  # :137
    (BattleEffect.DEF_UP, _basic_check_high_def),  # :138
    (BattleEffect.SP_ATK_UP, _basic_check_high_spa),  # :140
    (BattleEffect.ATK_DOWN, _basic_check_low_attack),  # :144
    (BattleEffect.DEF_DOWN, _basic_check_defense),  # :145
    (BattleEffect.SPEED_DOWN, _basic_check_low_speed),  # :146
    (BattleEffect.EVA_DOWN, _basic_check_low_evasion),  # :150
    (BattleEffect.RESTORE_HALF_HP, _basic_check_can_recover_hp),  # :154
    (BattleEffect.STATUS_BADLY_POISON, _basic_check_cannot_poison),  # :155
    (BattleEffect.SET_LIGHT_SCREEN, _basic_check_light_screen),  # :156
    (BattleEffect.HALVE_HP, _basic_check_non_standard_damage),  # :159
    (BattleEffect.STATUS_CONFUSE, _basic_check_cannot_confuse),  # :163
    (BattleEffect.ATK_UP_2, _basic_check_high_atk),  # :164
    (BattleEffect.DEF_UP_2, _basic_check_high_def),  # :165
    (BattleEffect.SPEED_UP_2, _basic_check_speed),  # :166
    (BattleEffect.SP_ATK_UP_2, _basic_check_high_spa),  # :167
    (BattleEffect.SP_DEF_UP_2, _basic_check_high_spd),  # :168
    (BattleEffect.ATK_DOWN_2, _basic_check_low_attack),  # :171
    (BattleEffect.DEF_DOWN_2, _basic_check_defense),  # :172
    (BattleEffect.SPEED_DOWN_2, _basic_check_low_speed),  # :173
    (BattleEffect.SP_DEF_DOWN_2, _basic_check_low_spd),  # :175
    (BattleEffect.SET_REFLECT, _basic_check_reflect),  # :178
    (BattleEffect.STATUS_PARALYZE, _basic_check_cannot_paralyze),  # :180
    (BattleEffect.STATUS_LEECH_SEED, _basic_check_cannot_leech_seed),  # :183
    (BattleEffect.COUNTER, _basic_check_non_standard_damage),  # :187
    (BattleEffect.DAMAGE_WHILE_ASLEEP, _basic_check_attacker_asleep),  # :189
    (BattleEffect.USE_RANDOM_LEARNED_MOVE_SLEEP, _basic_check_attacker_asleep),  # :191
    (BattleEffect.INCREASE_POWER_WITH_LESS_HP, _basic_check_non_standard_damage),  # :192
    (BattleEffect.CURSE, _basic_check_curse),  # :196
    (BattleEffect.SET_SPIKES, _basic_check_spikes),  # :197
    (BattleEffect.ATK_UP_2_STATUS_CONFUSION, _basic_check_cannot_confuse),  # :201
    (BattleEffect.INFATUATE, _basic_check_cannot_attract),  # :202
    (BattleEffect.WEATHER_RAIN, _basic_check_rain_dance),  # :214
    (BattleEffect.MIRROR_COAT, _basic_check_non_standard_damage),  # :218
    (BattleEffect.ALWAYS_FLINCH_FIRST_TURN_ONLY, _basic_check_first_turn),  # :224
    (BattleEffect.WEATHER_HAIL, _basic_check_hail),  # :228
    (BattleEffect.STATUS_BURN, _basic_check_cannot_burn),  # :231
    (BattleEffect.LOWER_OWN_ATK_AND_DEF, _basic_check_non_standard_damage),  # :237
    (BattleEffect.ATK_DEF_UP, _basic_check_bulk_up),  # :248
    (BattleEffect.SP_ATK_SP_DEF_UP, _basic_check_calm_mind),  # :250
    (BattleEffect.ATK_SPD_UP, _basic_check_dragon_dance),  # :251
    (BattleEffect.DOUBLE_SPEED_3_TURNS, _basic_check_tailwind),  # :259
    (BattleEffect.TOXIC_SPIKES, _basic_check_toxic_spikes),  # :277
    (BattleEffect.STEALTH_ROCK, _basic_check_stealth_rock),  # :284
)
# Expert_Main's entries (script.s:1628-1808) of the transcribed effects, in line order. The
# other effects of the transcribed moves have no entry and end at :1811.
_EXPERT_LABELS = (
    (BattleEffect.STATUS_SLEEP, _expert_status_sleep),  # :1628
    (BattleEffect.RECOVER_HALF_DAMAGE_DEALT, _expert_drain_move),  # :1629
    (BattleEffect.HALVE_DEFENSE, _expert_explosion),  # :1630
    (BattleEffect.ATK_UP, _expert_status_atk_up),  # :1633
    (BattleEffect.DEF_UP, _expert_status_def_up),  # :1634
    (BattleEffect.SP_ATK_UP, _expert_status_spa_up),  # :1636
    (BattleEffect.BYPASS_ACCURACY, _expert_bypass_accuracy),  # :1640
    (BattleEffect.ATK_DOWN, _expert_status_attack_down),  # :1641
    (BattleEffect.DEF_DOWN, _expert_status_defense_down),  # :1642
    (BattleEffect.SPEED_DOWN, _expert_status_speed_down),  # :1643
    (BattleEffect.EVA_DOWN, _expert_status_eva_down),  # :1647
    (BattleEffect.RESTORE_HALF_HP, _expert_recovery),  # :1652
    (BattleEffect.STATUS_BADLY_POISON, _expert_toxic_leech_seed),  # :1653
    (BattleEffect.SET_LIGHT_SCREEN, _expert_light_screen),  # :1654
    (BattleEffect.HALVE_HP, _expert_super_fang),  # :1658
    (BattleEffect.HIGH_CRITICAL, _expert_high_critical),  # :1660
    (BattleEffect.RECOIL_QUARTER, _expert_recoil_move),  # :1661
    (BattleEffect.STATUS_CONFUSE, _expert_status_confuse),  # :1662
    (BattleEffect.ATK_UP_2, _expert_status_atk_up),  # :1663
    (BattleEffect.DEF_UP_2, _expert_status_def_up),  # :1664
    (BattleEffect.SPEED_UP_2, _expert_status_speed_up),  # :1665
    (BattleEffect.SP_ATK_UP_2, _expert_status_spa_up),  # :1666
    (BattleEffect.SP_DEF_UP_2, _expert_status_spd_up),  # :1667
    (BattleEffect.ATK_DOWN_2, _expert_status_attack_down),  # :1670
    (BattleEffect.DEF_DOWN_2, _expert_status_defense_down),  # :1671
    (BattleEffect.SPEED_DOWN_2, _expert_status_speed_down),  # :1672
    (BattleEffect.SP_DEF_DOWN_2, _expert_status_spd_down),  # :1674
    (BattleEffect.SET_REFLECT, _expert_reflect),  # :1677
    (BattleEffect.STATUS_PARALYZE, _expert_status_paralyze),  # :1679
    (BattleEffect.ATK_UP_2_STATUS_CONFUSION, _expert_swagger),  # :1680
    (BattleEffect.LOWER_SPEED_HIT, _expert_speed_down_on_hit),  # :1681
    (BattleEffect.STATUS_LEECH_SEED, _expert_toxic_leech_seed),  # :1686
    (BattleEffect.COUNTER, _expert_counter),  # :1688
    (BattleEffect.DAMAGE_WHILE_ASLEEP, _expert_plus(2)),  # :1691 Expert_Nightmare
    (BattleEffect.USE_RANDOM_LEARNED_MOVE_SLEEP, _expert_sleep_talk),  # :1693
    (BattleEffect.KO_MON_THAT_DEFEATED_USER, _expert_destiny_bond),  # :1694
    (BattleEffect.INCREASE_POWER_WITH_LESS_HP, _expert_reversal),  # :1695
    (BattleEffect.CURSE, _expert_curse),  # :1700
    (BattleEffect.PROTECT, _expert_protect),  # :1701
    (BattleEffect.SET_SPIKES, _expert_spikes),  # :1702
    (BattleEffect.HIT_BEFORE_SWITCH, _expert_pursuit),  # :1706
    (BattleEffect.WEATHER_RAIN, _expert_rain_dance),  # :1710
    (BattleEffect.MIRROR_COAT, _expert_mirror_coat),  # :1714
    (BattleEffect.ALWAYS_FLINCH_FIRST_TURN_ONLY, _expert_plus(2)),  # :1723 Expert_FakeOut
    (BattleEffect.WEATHER_HAIL, _expert_hail),  # :1726
    (BattleEffect.DOUBLE_POWER_WHEN_STATUSED, _expert_facade),  # :1729
    (BattleEffect.DOUBLE_POWER_AND_CURE_PARALYSIS, _expert_smelling_salts),  # :1731
    (BattleEffect.LOWER_OWN_ATK_AND_DEF, _expert_superpower),  # :1735
    (BattleEffect.REMOVE_SCREENS, _expert_brick_break),  # :1739
    (BattleEffect.RECOIL_THIRD, _expert_recoil_move),  # :1747
    (BattleEffect.ATK_DEF_UP, _expert_status_def_up),  # :1753
    (BattleEffect.SP_ATK_SP_DEF_UP, _expert_status_spd_up),  # :1756
    (BattleEffect.ATK_SPD_UP, _expert_dragon_dance),  # :1757
    (BattleEffect.DOUBLE_POWER_WHEN_BELOW_HALF, _expert_brine),  # :1765
    (BattleEffect.EAT_BERRY, _expert_pluck),  # :1767
    (BattleEffect.DOUBLE_SPEED_3_TURNS, _expert_tailwind),  # :1768
    (BattleEffect.DOUBLE_POWER_IF_MOVING_SECOND, _expert_payback),  # :1773
    (BattleEffect.TOXIC_SPIKES, _expert_toxic_spikes),  # :1792
    (BattleEffect.RECOIL_PARALYZE_HIT, _expert_recoil_move),  # :1802
    (BattleEffect.STEALTH_ROCK, _expert_stealth_rock),  # :1805
    (BattleEffect.RECOIL_HALF, _expert_recoil_move),  # :1806
)
# The effects _expert branches on: the per-slot guard's transcribed set
# (outcome.KAIZO_EXPERT_TRANSCRIBED, a literal) must be among them
# (tests/ai/test_kaizo_policy.py test_expert_transcribed_is_the_dispatch).
EXPERT_DISPATCH = frozenset(int(e) for e, _ in _EXPERT_LABELS)


def _risky(ctx: AiContext) -> AiContext:
    """``Risky_Main`` (script.s:6521-6534; docs/ai/kaizo_roark.md 5.4): a move whose effect
    is in ``Risky_RiskyEffects`` (``commands.RISKY_EFFECTS``, :6536-6562) gets +2 unless the
    128 roll (:6530) jumps."""
    ctx = _terminate(ctx)
    ctx = c.load_current_move_effect(ctx)
    ctx, _ = _split(c.if_loaded_not_in_table(ctx, c.RISKY_EFFECTS))  # -> Risky_Terminate
    ctx, _ = _split(c.if_random_less_than(ctx, 128, 6530))
    return c.pop_or_end(c.add_to_move_score(ctx, 2))


def _setup_first_turn(ctx: AiContext) -> AiContext:
    """``SetupFirstTurn_Main`` (script.s:6417-6433): on the battle's first turn
    (``LoadTurnCount`` 0, :6421-6422), a move whose effect is in
    ``SetupFirstTurn_SetupEffects`` (``commands.SETUP_FIRST_TURN_EFFECTS``, :6425-6426) gets
    +2 unless the 80 roll (:6429) jumps."""
    ctx = _terminate(ctx)
    ctx = c.load_turn_count(ctx)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, 0))  # -> SetupFirstTurn_Terminate
    ctx = c.load_current_move_effect(ctx)
    ctx, _ = _split(c.if_loaded_not_in_table(ctx, c.SETUP_FIRST_TURN_EFFECTS))
    ctx, _ = _split(c.if_random_less_than(ctx, 80, 6429))
    return c.pop_or_end(c.add_to_move_score(ctx, 2))


def _weather(ctx: AiContext) -> AiContext:
    """``Weather_Main`` (script.s:7979-8019): on the battle's first turn (``LoadTurnCount`` 0,
    :7983-7984), a weather move goes to its weather's label (:7987-7990) and every other move
    falls through into ``Weather_Sun`` (no ``GoTo`` after :7990; E1 Ruling R3, transcribed as
    the game has it). A label ends unless the current weather (``LoadCurrentWeather``) is
    another than its own (:7992-8010); then ``Weather_ScorePlus5`` gives +5 while the
    attacker's ``LoadIsFirstTurnInBattle`` is TRUE (:8012-8016)."""
    ctx = _terminate(ctx)
    ctx = c.load_turn_count(ctx)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, 0))  # -> Weather_Terminate
    ctx, sun = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.WEATHER_SUN))
    ctx, rain = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.WEATHER_RAIN))
    ctx, sand = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.WEATHER_SANDSTORM))
    ctx, hail = _split(c.if_current_move_effect_equal_to(ctx, BattleEffect.WEATHER_HAIL))
    sun = sun | ctx.live  # the fall-through into Weather_Sun (:7992)
    plus5 = jnp.zeros_like(ctx.live)
    for reached, weather in (
        (sun, c.AiWeather.SUNNY),  # Weather_Sun (:7992)
        (rain, c.AiWeather.RAINING),  # Weather_Rain (:7997)
        (sand, c.AiWeather.SANDSTORM),  # Weather_Sand (:8002)
        (hail, c.AiWeather.HAILING),  # Weather_Hail (:8007)
    ):
        ctx = c.load_current_weather(_at(ctx, reached))
        ctx, _ = _split(c.if_loaded_equal_to(ctx, int(weather)))  # -> Weather_Terminate
        ctx, t = _split(c.go_to(ctx))
        plus5 = plus5 | t

    # Weather_ScorePlus5 (:8012)
    ctx = c.load_is_first_turn_in_battle(_at(ctx, plus5), _A)
    ctx, _ = _split(c.if_loaded_equal_to(ctx, 0))  # FALSE -> Weather_Terminate
    return c.pop_or_end(c.add_to_move_score(ctx, 5))


def _prioritize_extremes(ctx: AiContext) -> AiContext:
    """``PrioritizeExtremes_Main`` (script.s:6499-6519): a move ``FlagMoveDamageScore
    USE_MAX_DAMAGE`` makes no comparison for (``AI_NO_COMPARISON_MADE``, :6511-6512; a status
    move or a no-calc effect, trainer_ai.c:1012-1063; an alt-power effect always compares,
    ``c.is_damaging``) gets +2 unless the 100 roll (:6515) jumps."""
    ctx = _terminate(ctx)
    ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
    ctx, _ = _split(c.if_loaded_not_equal_to(ctx, c.AI_NO_COMPARISON_MADE))
    ctx, _ = _split(c.if_random_less_than(ctx, 100, 6515))
    return c.pop_or_end(c.add_to_move_score(ctx, 2))


def _check_hp_round(ctx: AiContext, high, medium, prefix: str, line: int) -> AiContext:
    """One round of ``CheckHP_Main`` after its two ``IfHPPercentGreaterThan`` (70 to the high
    band's label in ``high``, 30 to the medium band's in ``medium``, the rest low): each band
    loads the move's effect and jumps to the round's TryScoreMinus2 when it is in the band's
    table (``c.CHECK_HP_TABLES``; IfLoadedInTable), else goes to the round's end; TryScoreMinus2
    gives -2 unless the 50 roll at ``line`` jumps, and falls into the round's end."""
    minus2 = end = jnp.zeros_like(ctx.live)
    for reached, band in ((ctx.live, "Low"), (high, "High"), (medium, "Medium")):
        ctx = c.load_current_move_effect(_at(ctx, reached))
        table = c.CHECK_HP_TABLES[f"{prefix}DiscourageAt{band}HP"]
        ctx, t = _split(c.if_loaded_in_table(ctx, table))
        minus2, end = minus2 | t, end | ctx.live  # the rest GoTo the round's end
    ctx = _at(ctx, minus2)
    ctx, rolled = _split(c.if_random_less_than(ctx, 50, line))
    ctx = c.add_to_move_score(ctx, -2)
    return _at(ctx, end | rolled | ctx.live)


def _check_hp(ctx: AiContext) -> AiContext:
    """``CheckHP_Main`` (script.s:7696-7746): ``IfTargetIsPartner`` (to TagStrategy_Partner,
    never taken in singles), then the attacker's round (bands at :7700-7701, CheckHP_Discourage*
    tables, roll :7718) and the target's round (:7724-7725, CheckHP_Target_Discourage*, roll
    :7742), each -2 for a discouraged effect unless its roll jumps."""
    # IfTargetIsPartner TagStrategy_Partner (:7697), not Terminate: never taken in singles,
    # and the doubles work must send its path to TagStrategy_Partner.
    ctx, _ = _split(c.if_target_is_partner(ctx))
    ctx, high = _split(c.if_hp_percent_greater_than(ctx, _A, 70))
    ctx, medium = _split(c.if_hp_percent_greater_than(ctx, _A, 30))
    ctx = _check_hp_round(ctx, high, medium, "CheckHP_", 7718)
    ctx, high = _split(c.if_hp_percent_greater_than(ctx, _D, 70))
    ctx, medium = _split(c.if_hp_percent_greater_than(ctx, _D, 30))
    ctx = _check_hp_round(ctx, high, medium, "CheckHP_Target_", 7742)
    return c.pop_or_end(ctx)


def _flags(layout: AiLayout) -> tuple:
    """The flag scripts the policy runs: those of ``layout.flags``' set bits, in bit order
    (``TrainerAI_MainSingles`` runs a flag's script only when its bit is set, low bit first,
    trainer_ai.c:297-308; ``FlagTable``, script.s:18-30): BASIC (bit 0), EVAL_ATTACK (1),
    EXPERT (2), SETUP_FIRST_TURN (3), RISKY (4), PRIORITIZE_EXTREMES (5), CHECK_HP (8), WEATHER
    (9). Vanilla runs 7 (roark.md 1);
    Kaizo the trainer's set (E1 Ruling
    R2), Roark's 23. A bit without a transcribed script raises (``layout.kaizo_layout``
    builds no such layout; coverage refuses the trainer)."""
    scripts = {
        BASIC: partial(_basic, layout=layout),
        EVAL_ATTACK: _eval_attack,
        EXPERT: partial(_expert, layout=layout),
        SETUP_FIRST_TURN: _setup_first_turn,
        RISKY: _risky,
        PRIORITIZE_EXTREMES: _prioritize_extremes,
        CHECK_HP: _check_hp,
        WEATHER: _weather,
    }
    out = []
    for bit in range(layout.flags.bit_length()):
        if layout.runs(bit):
            if bit not in scripts:
                raise ValueError(f"AI flag bit {bit} of {layout.flags:#x} is not transcribed")
            out.append(scripts[bit])
    return tuple(out)


# --- Refusals ----------------------------------------------------------------------

_ATTACKER_ABILITIES = ("Normalize", "Adaptability", "Scrappy", "Tinted Lens")
_DEFENDER_ABILITIES = ("Levitate", "Wonder Guard", "Filter", "Solid Rock")
_STATE_MOVE_IDS = {
    n: switching.MOVE_IDS[n] for n in ("Miracle Eye", "Roost", "Magnet Rise", "Ingrain")
}


def player_move_refusal(move: int) -> str | None:
    """Why the AI refuses a fight whose player Pokemon knows ``move`` (decomp id), or None."""
    if move in switching.PLAYER_MOVE_REFUSALS:
        return f"AI switching: {switching.PLAYER_MOVE_REFUSALS[move]}"
    for name, move_id in _STATE_MOVE_IDS.items():
        if move == move_id:
            return f"AI scoring: move {name} leaves a type-chart state the engine does not carry"
    return None


def player_ability_refusal(ability: int) -> str | None:
    """Why the AI refuses a fight whose player Pokemon has ``ability`` (decomp id), or None."""
    if ability in switching.PLAYER_ABILITY_REFUSALS:
        return f"AI switching: {switching.PLAYER_ABILITY_REFUSALS[ability]}"
    for name in _DEFENDER_ABILITIES:
        if ability == _ABILITY[name]:
            return (
                f"AI scoring: ability {name} changes IfMoveEffectivenessEquals "
                "and is not implemented"
            )
    return None


class _Refused(NamedTuple):
    move: jnp.ndarray  # bool[SLOTS, MOVES]: opponent move without transcribed script paths
    ghost_curse: jnp.ndarray  # bool[SLOTS, MOVES]: Curse by a Ghost type (its labels are not)
    expert: jnp.ndarray  # bool[SLOTS, MOVES]: its effect's Expert label is not transcribed
    calc_damage: jnp.ndarray  # bool[SLOTS, MOVES]: a TrainerAI_CalcDamage id case
    move_type: jnp.ndarray  # bool[SLOTS, MOVES]: a TrainerAI_MoveType id case
    policy: jnp.ndarray  # bool: tables of another policy than the layout's
    sites: jnp.ndarray  # bool[SLOTS]: opponent moveset with too many script-roll sites
    ability: jnp.ndarray  # bool[2, SLOTS]
    item: jnp.ndarray  # bool[2, SLOTS]
    state_move: jnp.ndarray  # bool[2, SLOTS, MOVES]
    bag: jnp.ndarray  # bool[BAG_SLOTS]: bag item other than a Potion
    scoring: jnp.ndarray  # bool: ai_scores is refused
    any: jnp.ndarray  # bool: ai_policy is refused (switching aside)


@partial(jax.jit, static_argnames="layout")
def _refused(tables: Tables, state: BattleState, layout: AiLayout = VANILLA) -> _Refused:
    present = state.species > 0
    moves = state.moves.astype(_I32)
    effects = c.move_effect(tables, moves[1])
    if layout.name == VANILLA.name:  # by move id (MOVE_EFFECTS)
        transcribed = jnp.isin(moves[1], jnp.array(sorted(layout.transcribed), _I32))
    else:  # by battle effect (outcome.KAIZO_TRANSCRIBED)
        transcribed = jnp.isin(effects, jnp.array(sorted(layout.transcribed), _I32))
    move = present[1][:, None] & ~(transcribed | (moves[1] == 0))
    # Curse by a Ghost type takes Basic_CheckCurse_GhostType (script.s:681-684 -> :700, which
    # reads IfVolatileStatus) and Expert_Curse_GhostCheckHP (:3467-3470 -> :3495), which are
    # not transcribed.
    ghost = jnp.any(state.types[1] == _TYPE["Ghost"], axis=-1)  # [SLOTS]
    ghost_curse = present[1][:, None] & (effects == BattleEffect.CURSE) & ghost[:, None]
    ghost_curse = ghost_curse & ~move & (moves[1] != 0)
    # A transcribed effect whose Expert_Main entry (script.s:1628-1808) goes to a label not
    # transcribed yet (outcome.KAIZO_EXPERT_LABELS, layout.expert_transcribed): refused.
    pending = sorted(set(KAIZO_EXPERT_LABELS) - layout.expert_transcribed)
    expert = present[1][:, None] & (moves[1] != 0) & ~move & ~ghost_curse
    if layout.per_slot_sites and pending and layout.runs(EXPERT):
        expert = expert & jnp.isin(effects, jnp.array(pending, _I32))
    else:
        expert = jnp.zeros_like(expert)
    known = present[1][:, None] & (moves[1] != 0)
    calc_damage = known & jnp.isin(moves[1], jnp.array(c.CALC_DAMAGE_MOVES, _I32))
    move_type = known & jnp.isin(moves[1], jnp.array(c.MOVE_TYPE_MOVES, _I32))
    policy = tables.ai_policy != AI_POLICIES.index(layout.name)
    if layout.per_slot_sites:  # reads by order, bounded per path (missing_read)
        sites = jnp.zeros_like(present[1])
    else:
        sites = present[1] & (jax.vmap(script_roll_count)(moves[1]) > MAX_SCRIPT_ROLLS)
    ability = decomp_abilities(tables, state)  # current abilities (trainer_ai.c:1205)
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
        | jnp.any(ghost_curse)
        | jnp.any(expert)
        | jnp.any(calc_damage)
        | jnp.any(move_type)
        | policy
        | jnp.any(sites)
        | jnp.any(ability_bad)
        | jnp.any(item_bad)
        | jnp.any(state_move)
    )
    return _Refused(
        move,
        ghost_curse,
        expert,
        calc_damage,
        move_type,
        policy,
        sites,
        ability_bad,
        item_bad,
        state_move,
        bag,
        scoring,
        scoring | jnp.any(bag),
    )


def opponent_move_refusals(
    tables: Tables, move: int, layout: AiLayout, ghost: bool = False
) -> list[str]:
    """The per-move refusals of ``_refused`` for an opponent move (decomp id) under ``layout``,
    whatever the player's party (``tools/kaizo/backlog.py`` lists them per fight), as kinds:
    ``"not transcribed"`` (its script paths, ``layout.transcribed``), ``"ghost curse"`` (Curse
    by a Ghost type, ``ghost``), ``"expert label"`` (its Expert_Main label is not transcribed
    yet, when the layout runs EXPERT), ``"calc damage"`` and ``"move type"`` (TrainerAI_CalcDamage and
    TrainerAI_MoveType move-id cases). Pinned against ``_refused`` by
    tests/ai/test_kaizo_policy.py test_opponent_move_refusals_are_refused."""
    if move == 0:
        return []
    effect = int(np.asarray(tables.move_ai_effect)[move])
    key = effect if layout.per_slot_sites else move
    if key not in layout.transcribed:
        return ["not transcribed"]
    out = []
    if ghost and effect == BattleEffect.CURSE:
        out.append("ghost curse")
    pending = set(KAIZO_EXPERT_LABELS) - layout.expert_transcribed
    if layout.per_slot_sites and layout.runs(EXPERT) and effect in pending and not out:
        out.append("expert label")
    if move in c.CALC_DAMAGE_MOVES:
        out.append("calc damage")
    if move in c.MOVE_TYPE_MOVES:
        out.append("move type")
    return out


@lru_cache(maxsize=len(AI_POLICIES))
def _policy_names(policy: int):
    """The ``Names`` of the game whose AI policy is ``AI_POLICIES[policy]``
    (``game_constants.POLICY``; Kaizo renames move and item ids, e.g. 140 is Barrage in
    vanilla and Hurricane in Kaizo), or None when that game's tables are not built or more
    than one game has the policy (the messages then give ids)."""
    from nuzline.data.game_constants import POLICY
    from nuzline.data.games import GAMES
    from nuzline.data.tables import load_names

    games = [g for g, p in POLICY.items() if p == AI_POLICIES[policy]]
    if len(games) != 1:
        return None
    try:
        return load_names(GAMES[games[0]].tables_dir)
    except OSError:
        return None


def _names_for(tables: Tables, names=None):
    """``names`` (the caller's game), else the names of ``tables``' game by its AI policy
    (``tables.ai_policy``), never another game's. None: give ids."""
    if names is not None:
        return names
    return _policy_names(int(tables.ai_policy))


def _move_name(move: int, names=None) -> str:
    """``move``'s name in ``names`` (``_names_for``), else its id."""
    return names.move_names[move] if names is not None else f"move id {move}"


def _item_name(item: int, names=None) -> str:
    return names.item_names[item] if names is not None else f"item id {item}"


def _scoring_refusals(
    tables: Tables, state: BattleState, layout: AiLayout, names=None
) -> list[str]:
    names = _names_for(tables, names)
    r = _refused(tables, state, layout)
    moves = np.asarray(state.moves)
    out: list[str] = []
    if bool(r.policy):
        policy = AI_POLICIES[int(tables.ai_policy)]
        out.append(f"ai: the tables' AI policy is {policy}, the layout's {layout.name}")
    for s, k in zip(*np.nonzero(np.asarray(r.move)), strict=True):
        out.append(
            f"ai scoring: opponent slot {s}: the script paths of move "
            f"{_move_name(int(moves[1, s, k]), names)} are not transcribed (docs/ai/roark.md 5)"
        )
    for s, k in zip(*np.nonzero(np.asarray(r.ghost_curse)), strict=True):
        out.append(
            f"ai scoring: opponent slot {s}: move {_move_name(int(moves[1, s, k]), names)} by a "
            "Ghost type goes to Basic_CheckCurse_GhostType (script.s:700) and "
            "Expert_Curse_GhostCheckHP (script.s:3495), which are not transcribed "
            "(docs/ai/kaizo_roark.md 5.1)"
        )
    effects = np.asarray(tables.move_ai_effect)
    for s, k in zip(*np.nonzero(np.asarray(r.expert)), strict=True):
        move = int(moves[1, s, k])
        label, line = KAIZO_EXPERT_LABELS[int(effects[move])]
        out.append(
            f"ai scoring: opponent slot {s}: move {_move_name(move, names)} goes to {label} "
            f"(Expert_Main, script.s:{line}), whose script path is not transcribed yet "
            "(docs/ai/kaizo_roark.md 5)"
        )
    for s, k in zip(*np.nonzero(np.asarray(r.calc_damage)), strict=True):
        out.append(
            f"ai scoring: opponent slot {s}: move {_move_name(int(moves[1, s, k]), names)} "
            "takes a TrainerAI_CalcDamage move-id case (trainer_ai.c:2885-3056) that is not "
            "transcribed"
        )
    for s, k in zip(*np.nonzero(np.asarray(r.move_type)), strict=True):
        out.append(
            f"ai scoring: opponent slot {s}: move {_move_name(int(moves[1, s, k]), names)} "
            "takes a TrainerAI_MoveType move-id case (trainer_ai.c:3132-3222) that is not "
            "transcribed"
        )
    for s in np.flatnonzero(np.asarray(r.sites)):
        out.append(
            f"ai scoring: opponent slot {s}: its moveset has more IfRandomLessThan sites "
            f"than the {MAX_SCRIPT_ROLLS} AiOutcome.script_rolls encodes"
        )
    ability = np.asarray(decomp_abilities(tables, state))
    ability_names = {v: k for k, v in _ABILITY.items()}
    for side, s in zip(*np.nonzero(np.asarray(r.ability)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: ability {ability_names[int(ability[side, s])]} "
            "changes IfMoveEffectivenessEquals and is not implemented"
        )
    items = np.asarray(state.item)
    for side, s in zip(*np.nonzero(np.asarray(r.item)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: item {_item_name(int(items[side, s]), names)} "
            "changes IfMoveEffectivenessEquals and is not implemented"
        )
    move_names = {v: k for k, v in _STATE_MOVE_IDS.items()}
    for side, s, k in zip(*np.nonzero(np.asarray(r.state_move)), strict=True):
        out.append(
            f"ai scoring: side {side} slot {s}: move {move_names[int(moves[side, s, k])]} "
            "leaves a type-chart state the engine does not carry"
        )
    return out


def _item_refusals(tables: Tables, state: BattleState, layout: AiLayout, names=None) -> list[str]:
    names = _names_for(tables, names)
    bag = np.asarray(state.trainer_items[1])
    return [
        f"item use: bag slot {k}: {_item_name(int(bag[k]), names)} is not a transcribed "
        "TrainerAI_ShouldUseItem item (only Potion)"
        for k in np.flatnonzero(np.asarray(_refused(tables, state, layout).bag))
    ]


def _layout(tables: Tables, layout: AiLayout | None) -> AiLayout:
    """``layout``, else ``layout_of(tables)`` on concrete tables, else (traced tables)
    ``VANILLA``, under which ``_Refused.policy`` refuses another policy's tables."""
    if layout is not None:
        return layout
    if isinstance(tables.ai_policy, jax.core.Tracer):
        return VANILLA
    return layout_of(tables)


def _all_refusals(tables: Tables, state: BattleState, layout: AiLayout, names=None) -> list[str]:
    return (
        _scoring_refusals(tables, state, layout, names)
        + _item_refusals(tables, state, layout, names)
        + switching_refusals(tables, state, layout.switch_sites)
    )


def ai_refusals(
    tables: Tables, state: BattleState, names=None, layout: AiLayout | None = None
) -> list[str]:
    """Every reason the AI of this (concrete) state is refused, one human-readable line
    each; empty when ``ai_policy`` is exact for it. Moves and items are named from
    ``names``, the ``Names`` of ``tables``' game (``check_coverage`` passes them; Kaizo
    renames slots), else from the names of the game whose AI policy the tables carry
    (``_names_for``), or by id when those tables are not built."""
    return _all_refusals(tables, state, _layout(tables, layout), names)


# The per-slot layout's refusals found only on a scoring path (``_scores``), after the
# static ``_refused`` checks pass: a path with more random reads than the layout's slot_reads
# (named by ``ai_path_refusals``), or a score outside the s8 moveScore's 0-127
# (_slot_move_distribution); the generic message when no read is past the bound.
_PATH_REFUSAL = (
    "the scoring path makes more random reads than the layout's slot_reads, or a score "
    "outside the s8 moveScore's 0-127 (docs/ai/kaizo_roark.md 8.2-8.4)"
)


def _check(tables: Tables, state: BattleState, layout: AiLayout, reasons) -> None:
    """Raise ``UnsupportedAi`` with ``reasons(tables, state, layout)`` when there are any;
    under a trace (jit, vmap, even of a closed-over state) the check cannot run."""
    if isinstance(_refused(tables, state, layout).any, jax.core.Tracer):
        return
    found = reasons(tables, state, layout)
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
    (``commands.defender_ability``, which ``ai_scores`` uses for every such read). A per-slot
    outcome reads each guess from its own draw (``commands.load_battler_ability``); this gives
    the ability of guess ``ai_outcome.ability_guess``."""
    return c.defender_ability(tables, state, ai_outcome.ability_guess)


def _run_flags(ctx: AiContext, layout: AiLayout, flags: tuple | None = None) -> AiContext:
    """The policy's flag scripts (``flags``, default ``_flags(layout)``; tests pass a
    prefix of them) over the four move slots."""
    ctx = ctx._replace(slot=jnp.asarray(0, _I32))
    for flag in _flags(layout) if flags is None else flags:  # trainer_ai.c:297-310
        # The slots in a loop, not unrolled: one copy of each flag script in the program.

        def eval_slot(slot, ctx, flag=flag):  # TrainerAI_EvalMoves (:483-525)
            ctx = flag(c.at_slot(ctx, slot))
            # A MOVE_NONE slot runs no command and scores 0 (:499-501).
            return ctx._replace(
                scores=ctx.scores.at[slot].set(jnp.where(ctx.move == 0, 0, ctx.scores[slot]))
            )

        ctx = jax.lax.fori_loop(0, MOVES, eval_slot, ctx)
    return ctx


def _check_outcome(ai_outcome: AiOutcome, layout: AiLayout) -> None:
    """Check that ``ai_outcome.script_rolls`` and ``ability_guess`` have ``layout``'s shapes
    (static, so also when traced)."""
    if layout.per_slot_sites:
        shapes = {"script_rolls": (MOVES, layout.slot_reads), "ability_guess": ()}
    else:
        shapes = {"script_rolls": (MAX_SCRIPT_ROLLS,), "ability_guess": ()}
    for field, shape in shapes.items():
        got = tuple(jnp.shape(getattr(ai_outcome, field)))
        if got != shape:
            raise ValueError(
                f"AiOutcome.{field} has shape {got}; the {layout.name} layout's is {shape}"
            )


def _score_context(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout
) -> AiContext:
    """The ``AiContext`` after the policy's flag scripts (refusals not applied)."""
    _check_outcome(ai_outcome, layout)
    return _run_flags(c.new_context(tables, state, ai_outcome), layout)


def _scores(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout = VANILLA
) -> jnp.ndarray:
    """``int32[4]``: the scores, -1 per slot on a refused state. A per-slot layout also
    gives -1 when a path makes more random reads than ``layout.slot_reads``
    (``AiContext.missing_read``): ``_slot_move_distribution`` is then NaN and ``_policy``
    refuses. The flat (vanilla) layout's unused column is never on a path (outcome.py
    ``_VANILLA_COLUMN``), so it is not checked and vanilla is unchanged."""
    ctx = _score_context(tables, state, ai_outcome, layout)
    refused = _refused(tables, state, layout).scoring
    if layout.per_slot_sites:
        refused = refused | (ctx.missing_read > 0)
    return jnp.where(refused, -1, ctx.scores).astype(_I32)


_scores_jit = jax.jit(_scores, static_argnames="layout")


def ai_scores(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``int32[4]``: the opponent's move scores after the policy's flag scripts (``_flags``), from
    base 100 (``TrainerAI_Init``, trainer_ai.c:223; invalid slots 0). Raises
    ``UnsupportedAi`` on a refused concrete state (a per-slot path refusal included,
    ``ai_path_refusals``); -1 per slot when traced."""
    layout = _layout(tables, layout)
    _check(tables, state, layout, _scoring_refusals)
    scores = _scores_jit(tables, state, ai_outcome, layout)
    concrete = not isinstance(scores, jax.core.Tracer)
    if layout.per_slot_sites and concrete and bool(jnp.any(scores < 0)):
        _raise_path_refusal(tables, state, layout)
    return scores


def maybe_switch(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``int32``: ``TrainerAI_ShouldSwitch`` (trainer_ai.c:3894; roark.md 11.1) under
    ``ai_outcome.switch_stop``: the switch action ``4 + j``, or ``NO_SWITCH``
    (``switching.switch_action`` over the layout's ``switch_sites``)."""
    layout = _layout(tables, layout)
    return switch_action(tables, state, ai_outcome.switch_stop, layout.switch_sites)


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


def maybe_use_item(
    tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``int32``: ``TrainerAI_ShouldUseItem`` (trainer_ai.c:4056; roark.md 12): the item
    action ``10 + k`` of the first bag slot it uses, or ``NO_ITEM``. It draws nothing.
    Both engines empty the later slots the decomp's loop also zeroes."""
    _check(tables, state, _layout(tables, layout), _item_refusals)
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


def _policy(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout = VANILLA
) -> jnp.ndarray:
    switch = maybe_switch(tables, state, ai_outcome, layout)
    item = _use_item(tables, state)
    scores = _scores(tables, state, ai_outcome, layout)
    move = _pick(tables, state, scores, ai_outcome.tiebreak)
    action = jnp.where(switch != NO_SWITCH, switch, jnp.where(item != NO_ITEM, item, move))
    # A battler locked into a move picks no command, so the AI never runs for it
    # (battle_controller_player.c:335): its action is the locked slot.
    lock = locked_slot(state, 1)
    action = jnp.where(lock >= 0, lock, action)
    refused = _refused(tables, state, layout).any
    if layout.per_slot_sites:  # a read past slot_reads (_scores); else no score < 0
        refused = refused | jnp.any(scores < 0)
    return jnp.where(refused, REFUSED_ACTION, action).astype(_I32)


_policy_jit = jax.jit(_policy, static_argnames="layout")


def ai_policy(
    tables: Tables, state: BattleState, ai_outcome: AiOutcome, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``int32``: the opponent's action in plan 1a's encoding (``TrainerAI_PickCommand``
    then ``TrainerAI_Main``; roark.md 2): a voluntary switch ``4 + j``, else a bag item
    ``10 + k``, else the top-scoring move slot ``0..3``. Raises ``UnsupportedAi`` on a
    refused concrete state; ``switching.REFUSED_ACTION`` when traced."""
    layout = _layout(tables, layout)
    _check(tables, state, layout, _all_refusals)
    action = _policy_jit(tables, state, ai_outcome, layout)
    concrete = not isinstance(action, jax.core.Tracer)
    refused = layout.per_slot_sites and concrete and int(action) == REFUSED_ACTION
    # Only a scoring path refusal raises; a caller's SWITCH_REFUSED stop stays REFUSED_ACTION.
    if refused and bool(jnp.any(_scores_jit(tables, state, ai_outcome, layout) < 0)):
        _raise_path_refusal(tables, state, layout)
    return action


def ai_action_distribution_enumerated(
    tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``float32[NUM_OPP_ACTIONS]``: the opponent's action distribution by brute force,
    ``ai_policy`` under ``vmap`` over every ``AiOutcome`` of ``enumerate_ai_outcomes``
    (3840) weighted by its probability; NaN when any outcome's action is refused. It is
    the test oracle of the factored ``ai_action_distribution`` (``tests/ai``), kept
    because it shares no reasoning with the factoring. Traced (``jax.jit``), so it never
    raises: a refused state gives NaN. The terms are summed pairwise (``_pairwise_sum``):
    a sequential float32 sum of 3840 terms drifts by about 2e-6.

    A per-slot ``layout`` (concrete inputs): every joint pattern of the four slots' reads (as
    many bits per slot as its longest path reads, ``_path_records``; at most
    ``MAX_JOINT_READS`` in all), every switch stop and every tiebreak, each weighted by the
    product of its slots' path weights (``AiContext.weight``: a read's probability comes from
    the read, ``outcome`` docstring), 0 for a pattern with a bit set past a slot's reads."""
    layout = _layout(tables, layout)
    if not layout.per_slot_sites:
        return _enumerated_flat(tables, state, layout)
    _, reads = _path_records(tables, state, layout)
    columns = np.minimum(np.asarray(reads).max(axis=0), layout.slot_reads)
    bits = [(m, k) for m in range(MOVES) for k in range(int(columns[m]))]
    if len(bits) > MAX_JOINT_READS:
        raise ValueError(f"{len(bits)} reads: more than the {MAX_JOINT_READS} enumerated")
    patterns, switch, tiebreak = (
        a.reshape(-1)
        for a in np.meshgrid(
            np.arange(2 ** len(bits)),
            np.arange(layout.switch_sites + 1),
            np.arange(TIEBREAK_MOD),
            indexing="ij",
        )
    )
    on = (patterns[:, None] >> np.arange(len(bits))) & 1  # [K, B]
    rolls = np.full((len(patterns), MOVES, layout.slot_reads), READ_DRAWS[0], np.uint8)
    for i, (m, k) in enumerate(bits):
        rolls[:, m, k] = np.asarray(READ_DRAWS, np.uint8)[on[:, i]]
    p_switch = np.asarray(
        switch_stop_probs(switch_sites(tables, state, layout.switch_sites)), np.float64
    )
    outcomes = AiOutcome(
        sim_roll=jnp.full((len(patterns), MOVES), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.zeros(len(patterns), jnp.int8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.asarray(tiebreak, jnp.int8),
        switch_stop=jnp.asarray(switch, jnp.int8),
    )
    probs = jnp.asarray(p_switch[switch] * TIEBREAK_P[tiebreak], jnp.float32)
    return _enumerated(tables, state, outcomes, probs, layout)


@partial(jax.jit, static_argnames="layout")
def _enumerated_flat(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    outcomes, probs = enumerate_ai_outcomes(tables, state)
    return _enumerated(tables, state, outcomes, probs, layout)


def _path_weight(tables, state, outcome, layout: AiLayout) -> jnp.ndarray:
    """``float32``: the probability of a per-slot outcome's reads, the product over the slots
    of their path weights, 0 when a slot's draws past its reads are not ``READ_DRAWS[0]``
    (a repeat of a shorter pattern, ``_canonical``); 1 for the flat layout."""
    if not layout.per_slot_sites:
        return jnp.float32(1)
    ctx = _score_context(tables, state, outcome, layout)
    keep = _canonical(outcome.script_rolls, ctx.reads)
    return jnp.prod(jnp.where(keep, ctx.weight, 0.0))


@partial(jax.jit, static_argnames="layout")
def _enumerated(tables, state, outcomes, probs, layout: AiLayout) -> jnp.ndarray:
    actions = jax.vmap(lambda o: _policy(tables, state, o, layout))(outcomes)
    probs = probs * jax.vmap(lambda o: _path_weight(tables, state, o, layout))(outcomes)
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


def _move_distribution(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``float32[MOVES]``: the distribution of ``TrainerAI_Main``'s move pick, over the
    only draws ``_scores`` and ``_pick`` read: ``ability_guess`` x ``script_rolls``
    (32 scorings) and, for each scoring, ``tiebreak`` through ``_pick`` itself."""
    rolls, p_rolls = script_roll_probs(state)  # [16, 4], [16]
    n = ABILITY_GUESSES * len(rolls)
    outcomes = AiOutcome(
        sim_roll=jnp.full((n, MOVES), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.repeat(jnp.arange(ABILITY_GUESSES, dtype=jnp.int8), len(rolls)),
        script_rolls=jnp.tile(jnp.asarray(rolls), (ABILITY_GUESSES, 1)),
        tiebreak=jnp.zeros(n, jnp.int8),  # _scores does not read it
        switch_stop=jnp.full(n, SWITCH_NONE, jnp.int8),  # nor this
    )
    scores = jax.vmap(lambda o: _scores(tables, state, o, layout))(outcomes)  # [32, MOVES]
    tiebreaks = jnp.arange(TIEBREAK_MOD, dtype=jnp.int8)
    picks = jax.vmap(  # [32, TIEBREAK_MOD]
        lambda sc: jax.vmap(lambda t: _pick(tables, state, sc, t))(tiebreaks)
    )(scores)
    p_tiebreak = jnp.asarray(TIEBREAK_P, jnp.float32)
    shares = jnp.einsum("t,stm->sm", p_tiebreak, jax.nn.one_hot(picks, MOVES))  # [32, MOVES]
    weight = jnp.tile(p_rolls, ABILITY_GUESSES) / ABILITY_GUESSES  # [32]
    return weight @ shares


# share[n, i]: the probability that TrainerAI_MainSingles picks the i-th of n tied slots,
# P(tiebreak % n == i) = P(RandNext() % n == i) (trainer_ai.c:338; outcome.TIEBREAK_P).
_TIE_SHARE = np.zeros((MOVES + 1, MOVES), np.float32)
for _n in range(1, MOVES + 1):
    for _t, _p in enumerate(TIEBREAK_P):
        _TIE_SHARE[_n, _t % _n] += _p
# The per-slot histograms' score range, 0..255. A score is an s8 the AI clamps at 0
# (include/battle/ai_context.h:14; trainer_ai.c:581-585, commands.add_to_move_score), so
# it is 0..127; anything else is refused (NaN) rather than ranked.
_SCORE_VALUES = 256
_SCORE_MAX = 127


def _slot_move_distribution(tables: Tables, state: BattleState, layout: AiLayout) -> jnp.ndarray:
    """``float32[MOVES]``: ``_move_distribution`` for a per-slot layout, factored per slot
    (docs/ai/kaizo_roark.md 8).

    ``TrainerAI_EvalMoves`` scores slot by slot (trainer_ai.c:483-525): slot m's score reads
    only its own ``script_rolls[m]`` and ``ability_guess[m]`` columns (each guessing read is
    its own coin, :1194-1198, so no draw is shared between slots), and
    ``FlagMoveDamageScore``'s comparison reads only the deterministic damage estimates
    (:1012-1062). So:

    1. The scoring runs once per local bit pattern p (``2**slot_reads``, bit k standing for
       the draw ``READ_DRAWS[bit]`` of the k-th read on the path, the same p in every slot):
       ``scores[p, m]`` is slot m's score with its reads p.
    2. Each slot's histogram over score values (0-255) weights pattern p by its probability
       in that slot, the product of the probabilities of the reads its path made
       (``AiContext.weight``: N/256 or 1 - N/256 for an ``IfRandomLessThan``, 1/2 for a
       guess), 0 when a bit past the path's reads is set (``_canonical``: it repeats the
       pattern with that bit clear).
    3. The pick (``_pick``, trainer_ai.c:315-338): slot m with value v wins when every
       other candidate slot ties at v or is below it; over each subset T of the other
       candidates that ties, m gets ``_TIE_SHARE[|T| + 1, rank of m in T + m]``.

    NaN when a score is outside the s8 ``moveScore``'s clamped range 0-127
    (include/battle/ai_context.h:14)."""
    scores, hist = _slot_histograms(tables, state, layout)
    has = (state.moves[1, state.active[1]] != 0).at[0].set(True)  # the pick's candidates
    equal = jnp.where(has[:, None], hist, 0.0)
    below = jnp.where(has[:, None], jnp.cumsum(hist, axis=1) - hist, 1.0)
    picked = []
    for slot in range(MOVES):
        others = [j for j in range(MOVES) if j != slot]
        win = jnp.zeros(_SCORE_VALUES, jnp.float32)
        for subset in range(2 ** len(others)):
            tied = [j for k, j in enumerate(others) if subset >> k & 1]
            term = jnp.ones(_SCORE_VALUES, jnp.float32)
            for j in others:
                term = term * (equal[j] if j in tied else below[j])
            rank = sum(j < slot for j in tied)
            win = win + term * _TIE_SHARE[len(tied) + 1, rank]
        picked.append(jnp.sum(equal[slot] * win))
    dist = jnp.stack(picked)
    dist = jnp.where(jnp.any(usable_moves(tables, state, 1)), dist, jax.nn.one_hot(0, MOVES))
    out_of_range = jnp.any((scores < 0) | (scores > _SCORE_MAX))
    return jnp.where(out_of_range, jnp.nan, dist)


def _pattern_outcomes(layout: AiLayout) -> tuple[np.ndarray, AiOutcome]:
    """``(bits int[P, R], outcomes)``: the ``AiOutcome`` of each local bit pattern p of a
    per-slot layout, ``P = 2**R`` with ``R = slot_reads``: read k of every slot draws
    ``READ_DRAWS[bit k of p]`` (``_slot_histograms``)."""
    width = layout.slot_reads
    patterns = 2**width
    bits = (np.arange(patterns)[:, None] >> np.arange(width)) & 1  # [P, R]
    draws = np.asarray(READ_DRAWS, np.uint8)[bits]
    rolls = np.broadcast_to(draws[:, None, :], (patterns, MOVES, width))
    outcomes = AiOutcome(
        sim_roll=jnp.full((patterns, MOVES), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.zeros(patterns, jnp.int8),  # unread
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.zeros(patterns, jnp.int8),  # _scores does not read it
        switch_stop=jnp.full(patterns, SWITCH_NONE, jnp.int8),  # nor this
    )
    return bits, outcomes


def _canonical(rolls, reads) -> jnp.ndarray:
    """``bool[MOVES]``: each slot's draws past its ``reads`` path reads (``rolls``,
    ``uint8[MOVES, slot_reads]``) are all ``READ_DRAWS[0]``, so the pattern is the one
    representative of its path."""
    unread = jnp.arange(rolls.shape[1]) >= jnp.asarray(reads)[:, None]
    return jnp.all(~unread | (rolls == READ_DRAWS[0]), axis=1)


@partial(jax.jit, static_argnames="layout")
def _path_records(tables: Tables, state: BattleState, layout: AiLayout):
    """``(missing_read int32[P], reads int32[P, MOVES])``: the path records (``AiContext``) of
    every scoring ``_slot_histograms`` runs: a read past ``slot_reads`` (0: none), and each
    slot's random reads."""
    _, outcomes = _pattern_outcomes(layout)

    def one(o):
        ctx = _score_context(tables, state, o, layout)
        return ctx.missing_read, ctx.reads

    return jax.vmap(one)(outcomes)


def ai_path_refusals(
    tables: Tables, state: BattleState, names=None, layout: AiLayout | None = None
) -> list[str]:
    """The refusals of a concrete per-slot state found only on a scoring path, after
    ``ai_refusals`` is empty: each slot whose path makes more random reads than
    ``layout.slot_reads`` (``AiContext.missing_read``; kaizo_roark.md 8.3), naming the slot
    and the move. Empty for the flat layout, which has no such refusal."""
    layout = _layout(tables, layout)
    if not layout.per_slot_sites:
        return []
    names = _names_for(tables, names)
    moves = np.asarray(state.moves[1, state.active[1]])
    missing, _ = _path_records(tables, state, layout)
    return [
        f"ai scoring: opponent slot {slot}: move {_move_name(int(moves[slot]), names)} makes "
        f"more than {layout.slot_reads} random reads on one path, the layout's slot_reads "
        "(docs/ai/kaizo_roark.md 8.3)"
        for slot in sorted({int(x) - 1 for x in np.asarray(missing) if x})
    ]


def _raise_path_refusal(tables: Tables, state: BattleState, layout: AiLayout) -> None:
    found = ai_path_refusals(tables, state, layout=layout)
    raise UnsupportedAi("; ".join(found) or _PATH_REFUSAL)


def _slot_histograms(
    tables: Tables, state: BattleState, layout: AiLayout
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Steps 1 and 2 of ``_slot_move_distribution``: ``(scores int32[P, MOVES],
    hist float32[MOVES, 256])``, ``hist[m, v]`` the probability that slot m scores v
    (refusals not applied)."""
    _, outcomes = _pattern_outcomes(layout)
    refused = _refused(tables, state, layout).scoring

    def one(o):
        ctx = _score_context(tables, state, o, layout)
        scores = jnp.where(refused | (ctx.missing_read > 0), -1, ctx.scores).astype(_I32)
        keep = _canonical(o.script_rolls, ctx.reads)
        return scores, jnp.where(keep, ctx.weight, 0.0)

    scores, weight = jax.vmap(one)(outcomes)  # [P, MOVES] each
    m = np.broadcast_to(np.arange(MOVES), scores.shape)
    values = jnp.clip(scores, 0, _SCORE_VALUES - 1)
    hist = (
        jnp.zeros((MOVES, _SCORE_VALUES), jnp.float32).at[m, values].add(weight)
    )  # [M, V]: P(slot m scores v)
    return scores, hist


@partial(jax.jit, static_argnames="layout")
def _distribution(tables: Tables, state: BattleState, layout: AiLayout = VANILLA) -> jnp.ndarray:
    """The exact distribution, factored as ``_policy`` decides (roark.md 2): the switch
    when ``switch_stop`` means one (``maybe_switch``, ``TrainerAI_ShouldSwitch``,
    trainer_ai.c:3894), else the item when ``_use_item`` gives one
    (``TrainerAI_ShouldUseItem``, trainer_ai.c:4056), else the move pick (``_pick``,
    trainer_ai.c:315-338). The switch reads only ``switch_stop``, the item no draw, the
    move pick only ``ability_guess``, ``script_rolls`` and ``tiebreak``, so the 3840
    outcomes of ``ai_action_distribution_enumerated`` reduce to 10 switch stops and 32
    scorings (a per-slot layout: ``_slot_move_distribution``, 2**slot_reads
    scorings). NaN exactly where the
    enumeration gives NaN: a refused state, or a refused switch layout (``switch_action``
    gives ``REFUSED_ACTION`` for any stop). The switch stops are the layout's
    ``switch_sites + 1`` (10 vanilla, 45 Kaizo)."""
    sites = layout.switch_sites
    stops = jnp.arange(sites + 1, dtype=jnp.int8)
    p_stop = switch_stop_probs(switch_sites(tables, state, sites))  # [sites + 1]
    switch = jax.vmap(lambda stop: switch_action(tables, state, stop, sites))(stops)
    switches = switch >= 0
    dist = jnp.zeros(NUM_OPP_ACTIONS, jnp.float32)
    dist = dist.at[jnp.where(switches, switch, 0)].add(jnp.where(switches, p_stop, 0))
    p_stay = jnp.sum(jnp.where(switch == NO_SWITCH, p_stop, 0))
    item = _use_item(tables, state)
    move_distribution = _slot_move_distribution if layout.per_slot_sites else _move_distribution
    moves = (
        jnp.zeros(NUM_OPP_ACTIONS, jnp.float32)
        .at[:MOVES]
        .set(move_distribution(tables, state, layout))
    )
    stay = jnp.where(item != NO_ITEM, jax.nn.one_hot(item, NUM_OPP_ACTIONS), moves)
    lock = locked_slot(state, 1)
    dist = jnp.where(lock >= 0, jax.nn.one_hot(lock, NUM_OPP_ACTIONS), dist + p_stay * stay)
    refused = _refused(tables, state, layout).any | jnp.any(switch == REFUSED_ACTION)
    return jnp.where(refused, jnp.nan, dist)


def _path_counts(tables: Tables, state: BattleState, layout: AiLayout) -> np.ndarray:
    """``int[MOVES]``: per slot of a per-slot layout, its distinct scoring paths in this
    (concrete) state: the patterns ``_canonical`` keeps, one per path."""
    _, outcomes = _pattern_outcomes(layout)
    _, reads = _path_records(tables, state, layout)
    keep = jax.vmap(_canonical)(outcomes.script_rolls, reads)  # [P, MOVES]
    return np.asarray(keep).sum(axis=0)


def ai_width(tables: Tables, state: BattleState) -> int:
    """The draw combinations that ``ai_action_distribution`` merges for this (concrete)
    state: the switch stops plus the patterns of the scoring. Flat layout: ``SWITCH_SITES
    + 1 + ABILITY_GUESSES * 2**MAX_SCRIPT_ROLLS`` (42). Per-slot layout: ``switch_sites + 1 +
    sum_slots (the slot's distinct scoring paths)``, the slots being factored (45 switch stops
    for Kaizo; ``_path_counts``). The program itself always evaluates ``2**slot_reads``
    scorings, each giving every slot's score (``_slot_move_distribution``); ``_check_width``
    bounds that against ``WIDTH_CAP`` for every state (plan K3c Ruling 12; E1 Ruling R4)."""
    layout = layout_of(tables)
    stops = layout.switch_sites + 1
    if not layout.per_slot_sites:
        return stops + ABILITY_GUESSES * 2**MAX_SCRIPT_ROLLS
    return stops + int(np.sum(_path_counts(tables, state, layout)))


def ai_flat_width(tables: Tables, state: BattleState) -> int:
    """The flat product of the same draws, reported next to ``ai_width`` and not used:
    ``2**(guess bits) * 2**(script-roll sites) * TIEBREAK_MOD * (switch stops)``, one guess
    bit for the flat layout (``ENUMERATION_WIDTH``, 3840); for a per-slot one,
    ``prod_slots (the slot's distinct paths) * TIEBREAK_MOD * (switch stops)``."""
    layout = layout_of(tables)
    if not layout.per_slot_sites:
        return ENUMERATION_WIDTH
    paths = int(np.prod(_path_counts(tables, state, layout)))
    return paths * TIEBREAK_MOD * (layout.switch_sites + 1)


def _check_width() -> None:
    """The largest ``ai_width`` of each base layout, every slot at ``2**slot_reads`` paths,
    is within ``WIDTH_CAP`` (a design-time invariant, like ``outcome.ENUMERATION_WIDTH``); a
    fight's Kaizo layout is checked when it is built (``layout.kaizo_layout``, E1 Ruling R6)."""
    from nuzline.ai.layout import LAYOUTS

    for layout in LAYOUTS:
        if layout.per_slot_sites:
            most = layout.switch_sites + 1 + MOVES * 2**layout.slot_reads
        else:
            most = layout.switch_sites + 1 + ABILITY_GUESSES * 2**MAX_SCRIPT_ROLLS
        if most > WIDTH_CAP:
            raise RuntimeError(f"the {layout.name} AI width {most} > WIDTH_CAP {WIDTH_CAP}")


_check_width()


def ai_action_distribution(
    tables: Tables, state: BattleState, layout: AiLayout | None = None
) -> jnp.ndarray:
    """``float32[NUM_OPP_ACTIONS]``: the exact probability of each opponent action over
    every ``AiOutcome``, computed factored (``_distribution``: 32 scorings instead of
    the 3840 outcomes of ``ai_action_distribution_enumerated``, which it equals; a per-slot
    layout's ``2**slot_reads``).
    Raises ``UnsupportedAi`` on a refused concrete state; NaN when traced."""
    layout = _layout(tables, layout)
    _check(tables, state, layout, _all_refusals)
    dist = _distribution(tables, state, layout)
    concrete = not isinstance(dist, jax.core.Tracer)
    if layout.per_slot_sites and concrete and bool(jnp.any(jnp.isnan(dist))):
        _raise_path_refusal(tables, state, layout)
    return dist


# --- The loose oracle (tests only) ---------------------------------------------------

LOOSE_JOINT_BITS = 18  # the loose oracle enumerates the slots jointly up to this many reads
MAX_JOINT_READS = 10  # ai_action_distribution_enumerated's per-slot joint enumeration


def _loose_run(tables: Tables, state: BattleState, bits, layout: AiLayout):
    """``(scores int32[MOVES], LooseDraws)``: the flag scripts with every ``RandNext`` read
    its own draw ``bits[slot, read]`` (``commands`` docstring), refusals not applied."""
    outcome = AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.int8(0),  # not read: every DEFENDER ability read draws its own bit
        script_rolls=jnp.zeros((MAX_SCRIPT_ROLLS,), jnp.int8),  # not read
        tiebreak=jnp.int8(0),
        switch_stop=jnp.int8(SWITCH_NONE),
    )
    no_sites = jnp.zeros((MOVES, len(c.SITE_LINES)), _I32)  # not read
    ctx = c.new_context(tables, state, outcome, no_sites, c.loose_draws(bits))
    ctx = _run_flags(ctx, layout)
    return ctx.scores, ctx.loose


def _loose_batch(tables: Tables, states: BattleState, bits, layout: AiLayout):
    """``_loose_run`` for every state (leading axis N) and bit pattern (``bits``,
    ``int8[P, MOVES, LOOSE_READS]``): scores ``[N, P, MOVES]`` and ``LooseDraws [N, P, ...]``.
    The loose oracle reads no read bound (it has no ``AiOutcome`` draws), so its program is
    compiled once per flag set and effect set, not once per fight's bound (E1 Ruling R6), and
    the patterns are padded to a power of two (zero rows, sliced off again), so a growing
    enumeration reuses a few compiled shapes instead of compiling one per count."""
    bits = jnp.asarray(bits)
    n = int(bits.shape[0])
    padded = 1 << max(n - 1, 0).bit_length()
    if padded != n:
        bits = jnp.concatenate([bits, jnp.zeros((padded - n, *bits.shape[1:]), bits.dtype)])
    out = _loose_batch_program(tables, states, bits, layout._replace(slot_reads=0))
    return jax.tree.map(lambda x: x[:, :n], out)


@partial(jax.jit, static_argnames="layout")
def _loose_batch_program(tables: Tables, states: BattleState, bits, layout: AiLayout):
    per_state = jax.vmap(lambda s, b: _loose_run(tables, s, b, layout), in_axes=(None, 0))
    return jax.vmap(lambda s: per_state(s, bits))(states)


def _loose_patterns(reads: list[list[int]]) -> np.ndarray:
    """``int8[P, MOVES, LOOSE_READS]``: pattern p sets, in each slot m, read ``reads[m][k]``
    to bit k of p, for every p below ``2**max(len)`` (padded to a power of two)."""
    width = max(len(r) for r in reads)
    patterns = np.arange(2**width)
    bits = np.zeros((len(patterns), MOVES, c.LOOSE_READS), np.int8)
    for m, slot_reads in enumerate(reads):
        for k, read in enumerate(slot_reads):
            bits[:, m, read] = patterns >> k & 1
    return bits


def _loose_reads(tables: Tables, states: BattleState, layout: AiLayout):
    """``(reads, p, kind)``: per slot, the sorted indices of the reads on some path of some
    state of ``states`` (leading axis N), found by enumerating the reads found so far until
    no new one appears (a read off every path found so far is only reached through a bit of
    one of them); and each read index's probability and ``ReadKind`` (``[MOVES,
    LOOSE_READS]``, the same for every state: the reads' order is static)."""
    reads: list[list[int]] = [[] for _ in range(MOVES)]
    while True:
        _, lo = _loose_batch(tables, states, jnp.asarray(_loose_patterns(reads)), layout)
        count = np.asarray(lo.count)
        if count.max() > c.LOOSE_READS:
            raise RuntimeError(f"{count.max()} reads in one slot > LOOSE_READS {c.LOOSE_READS}")
        live = np.asarray(lo.live).any(axis=(0, 1))  # [MOVES, LOOSE_READS]
        found = [
            sorted(set(r) | set(np.flatnonzero(live[m]).tolist())) for m, r in enumerate(reads)
        ]
        if found == reads:
            p = np.asarray(lo.p, np.float64)[0, 0]
            return reads, p, np.asarray(lo.kind)[0, 0]
        reads = found


# _LOOSE_SHARE[n, i] = P(RandNext() % n == i) over a uniform u16, computed directly
# (trainer_ai.c:338): 65536 = n * (65536 // n) + 65536 % n.
_LOOSE_SHARE = np.array(
    [
        [(65536 // n + (i < 65536 % n)) / 65536 if n and i < n else 0.0 for i in range(MOVES)]
        for n in range(MOVES + 1)
    ]
)


def _loose_pick(has: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``float64[MOVES]``: the pick of ``TrainerAI_MainSingles`` (trainer_ai.c:315-338)
    over the score rows ``scores [K, MOVES]`` with probabilities ``weights [K]``: slot 0 and
    every slot with a move are candidates, the tied top slots in slot order share by
    ``_LOOSE_SHARE``."""
    scores = np.where(has, scores, -1)  # a non-candidate never ties (scores are >= 0)
    tied = has & (scores == scores.max(axis=1, keepdims=True))
    n = tied.sum(axis=1)
    rank = np.cumsum(tied, axis=1) - 1
    share = np.where(tied, _LOOSE_SHARE[n[:, None], np.clip(rank, 0, MOVES - 1)], 0.0)
    return weights @ share


def _loose_move_distribution(
    tables: Tables, state: BattleState, layout: AiLayout, joint_bits: int = LOOSE_JOINT_BITS
) -> np.ndarray:
    """``float64[MOVES]``: the move pick when every read is its own draw. With at most
    ``joint_bits`` reads on the paths of the four slots, every combination of them is
    enumerated jointly; otherwise each slot's score distribution over its own reads, and
    the pick over the product of those (exact in loose: no two slots share a draw)."""
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], state)
    reads, p, _ = _loose_reads(tables, batch, layout)
    has = np.array(state.moves[1, state.active[1]] != 0)
    has[0] = True
    if not np.any(np.asarray(usable_moves(tables, state, 1))):
        return np.eye(MOVES)[0]  # Struggle (battle_controller_player.c:433)

    def weight(bits: np.ndarray, slot_reads: list[tuple[int, int]]) -> np.ndarray:
        w = np.ones(len(bits))
        for k, (m, r) in enumerate(slot_reads):
            on = bits >> k & 1
            w *= np.where(on == 1, p[m, r], 1 - p[m, r])
        return w

    every = [(m, r) for m in range(MOVES) for r in reads[m]]
    if len(every) <= joint_bits:
        patterns = np.arange(2 ** len(every))
        rows, weights = [], []
        for start in range(0, len(patterns), 4096):
            chunk = patterns[start : start + 4096]
            bits = np.zeros((len(chunk), MOVES, c.LOOSE_READS), np.int8)
            for k, (m, r) in enumerate(every):
                bits[:, m, r] = chunk >> k & 1
            scores, _ = _loose_batch(tables, batch, jnp.asarray(bits), layout)
            rows.append(np.asarray(scores)[0])
            weights.append(weight(chunk, every))
        return _loose_pick(has, np.concatenate(rows), np.concatenate(weights))
    support = [list(h.items()) for h in _loose_slot_scores(tables, state, layout)]
    combos = list(itertools.product(*support))
    rows = np.array([[v for v, _ in combo] for combo in combos])
    weights = np.array([np.prod([w for _, w in combo]) for combo in combos])
    return _loose_pick(has, rows, weights)


def _loose_slot_scores(
    tables: Tables, state: BattleState, layout: AiLayout
) -> list[dict[int, float]]:
    """Per move slot, the loose distribution of its score (``{score: probability}``) over
    the slot's own reads, each its own draw (refusals not applied)."""
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], state)
    reads, p, _ = _loose_reads(tables, batch, layout)
    scores, _ = _loose_batch(tables, batch, jnp.asarray(_loose_patterns(reads)), layout)
    scores = np.asarray(scores)[0]  # [P, MOVES]
    out = []
    for m in range(MOVES):
        values: dict[int, float] = {}
        for q in range(2 ** len(reads[m])):
            w = 1.0
            for k, r in enumerate(reads[m]):
                w *= p[m, r] if q >> k & 1 else 1 - p[m, r]
            values[int(scores[q, m])] = values.get(int(scores[q, m]), 0.0) + w
        out.append(values)
    return out


def ai_action_distribution_loose(
    tables: Tables,
    state: BattleState,
    layout: AiLayout | None = None,
    joint_bits: int = LOOSE_JOINT_BITS,
) -> np.ndarray:
    """``float64[NUM_OPP_ACTIONS]``: the opponent's action distribution with every decomp
    ``RandNext`` read of the scoring its own independent draw (tests only; the
    ``commands`` docstring): each ``IfRandomLessThan`` with its own N/256 (trainer_ai.c:528-538),
    each ``LoadBattlerAbility DEFENDER`` its own coin flip (:1191-1198), the tie-break
    ``RandNext() % n`` over a u16 (:338). Its reads are indexed in the static order of the
    straight-line scripts, not by order on the path, and not bounded, so it checks the gates
    of the exact layout: read order (E1 Ruling R4), the read bound, per-slot factoring. The switch, the item and the locked slot are combined as ``_distribution``
    does (roark.md 2). NaN on every state ``_distribution`` refuses: the scoring and item
    refusals (``_refused``) and the voluntary-switch ones (``switching_refusals``, where
    ``switch_sites`` would raise ``UnsupportedSwitching``). Concrete only."""
    layout = _layout(tables, layout)
    sites = layout.switch_sites
    if switching_refusals(tables, state, sites):
        return np.full(NUM_OPP_ACTIONS, np.nan)
    stops = jnp.arange(sites + 1, dtype=jnp.int8)
    switch = np.asarray(jax.vmap(lambda stop: switch_action(tables, state, stop, sites))(stops))
    if bool(_refused(tables, state, layout).any) or np.any(switch == REFUSED_ACTION):
        return np.full(NUM_OPP_ACTIONS, np.nan)
    p_stop = np.asarray(switch_stop_probs(switch_sites(tables, state, sites)), np.float64)
    dist = np.zeros(NUM_OPP_ACTIONS)
    for action, p in zip(switch, p_stop, strict=True):
        if action >= 0:
            dist[action] += p
    p_stay = float(np.sum(p_stop[switch == NO_SWITCH]))
    stay = np.zeros(NUM_OPP_ACTIONS)
    item = int(_use_item(tables, state))
    if item != NO_ITEM:
        stay[item] = 1.0
    else:
        stay[:MOVES] = _loose_move_distribution(tables, state, layout, joint_bits)
    lock = int(locked_slot(state, 1))
    if lock >= 0:
        return np.eye(NUM_OPP_ACTIONS)[lock]
    return dist + p_stay * stay
