"""Exact chance nodes: every ``Outcome`` field's law for one state and action pair,
and their joint enumeration (Stage B spec §1; plan ruling 3; K3b2 rules §10.3).

``outcome_laws`` (``jnp``, fixed shapes, jit/vmap-able) returns the integer
parameters of each field's law, exactly as ``outcome.sample_outcome`` draws it
(outcome.py:143-193), the relevance masks (``relevant``: the elements ``step`` can read
this turn in some order; ``branch_relevant``: per order branch) and the inputs of the
conditional reads (``Reads``). ``enumerate_outcomes`` (host, numpy) lists the
combinations the turn can read, with float64 probabilities computed from those
integers; every element a row does not read keeps its ``empty_outcome`` value. Values
of probability 0 are left out.

Relevance (notes/stage-b/outcomes.md §3, §7) may over-approximate: a row that turns
out not to matter steps to the same state as its neighbours and is merged after
``step``, so the masks only need to be sound, never tight. They are judged on the
turn-start state; an element is relevant when it can be read under some earlier event
of the turn:

* ``hit[s, 0]``: ``s`` uses a move (not a switch or item) with accuracy != 0
  (``step._use_move``). Only hit index 0 is read (a multi-hit move checks accuracy on
  its first hit, effects.py docstring). Its values are the interval classes of the roll
  over ``outcome.hit_thresholds`` (``hit_class_law``): two classes for one threshold
  below 100 (0 and 1, as before), one more per further threshold an earlier action can
  produce, and the single value 1 when every threshold is 100 or more.
* ``crit[s, i]``, ``roll[s, i]``: a damaging move (status moves and power 0 deal 0,
  damage.py:254), index 0, or for ``MULTI_HIT`` 0-4 (the 2-5 law) or the indices below its
  fixed count (``move_param[m, 0]``, ``effects.hit_count``). ``roll[s, 0]``
  also when a confusion self-hit is possible (``step._use_move``, step.py:503). The Pursuit
  pre-switch hit reads ``[pursuer, 0]`` (``step._pursuit_hit``, step.py:922-946), a move
  action too.
* ``secondary[s, i]``: as ``crit``, for a move with chance != 0 whose effect reads
  ``proc`` (``_PROC_EFFECTS``: the secondary stat effects, AncientPower's, secondary
  status, flinch, Fake Out, Bind, secondary confusion, Thunder Fang's paralysis). Not for a
  flinch move (``FLINCH``, ``FAKE_OUT``) of the side moving second (``subscript_flinch_mon.s``:
  ``IfMovedThisTurn`` skips it; effects._flinch), nor, moving first with one hit, for a
  secondary status (Thunder Fang's paralysis too) the target cannot take
  (``effects.can_have_status``, or Limber against paralysis; an Insomnia target is never
  skipped, its reveal runs on the proc) or a secondary confusion on a confused target
  (``subscript_confuse.s``, effects._secondary_confuse).
* ``fang_flinch[s]``: Thunder Fang's flinch roll (``FANG_PARALYZE_FLINCH``, effects._fang),
  only when ``s`` acts first in the branch (``IfMovedThisTurn``: the target has not moved,
  ``step``'s ``target_moved``) and the target has no Inner Focus as ``s`` sees it.
* ``contact_proc[d, i]``: the attacker's damaging contact move against a Static,
  Poison Point or Flame Body holder on side ``d`` (Rough Skin has no roll): its active
  Pokemon (Pursuit hits the outgoing one) or the one it switches in (effects.py:343-365);
  or against a Cute Charm holder there of the attacker's opposite gender (neither
  genderless) while the attacker is not in love at the start, or while side ``d`` switches
  (the outgoing source leaves first and clears the attacker's love, ``step._source_leaves``,
  so the incoming holder's charm can work) (effects._contact_ability: ``BtlCmd_TryAttract``
  fails otherwise; one becoming infatuated earlier in the turn only removes the read).
* ``n_hits[s]``: ``MULTI_HIT`` without a fixed count (``move_param[m, 0] == 0``).
* ``full_para[s]`` / ``thaw[s]``: ``s`` moves and is paralyzed / frozen at the start,
  or the foe moves first with a move that inflicts it and the target can take it
  (``step._use_move``: thaw step.py:467-468, full paralysis :519).
* ``infatuation[s]``: ``s`` moves and is in love at the start while the foe does not switch
  (a switch goes first and ``_source_leaves`` clears it), or the foe moves first with Attract
  and could infatuate ``s`` (opposite genders, neither genderless, ``s`` not in love yet;
  ``effects._attract``). Cute Charm infatuates the attacker during its own move, after its
  check: read only on a later turn.
* ``confusion_hit[s]``: ``s`` moves and its confusion counter is >= 2, or the foe moves
  first with a ``CONFUSE_FOE`` or ``SECONDARY_CONFUSE`` move and ``s`` is not confused
  (counter 2-5; ``step._use_move``, step.py:496-499).
* ``duration[s]``: ``s``'s move is Taunt, Bind, ``CONFUSE_FOE``, ``SECONDARY_CONFUSE`` or
  inflicts sleep (effects.py:189, :286, :297, :307).
* ``speed_tie``: the two orders ``order.action_order`` gives under ``speed_tie`` 0 and 1
  differ (it is read only there: ``player_second``'s ``RandNext() & 1``, battle_lib.c:1445).
* ``end_speed_tie``: the effective speeds after the actions can be equal
  (``step.end_of_turn_order``, step.py:1251-1261; a foe's Roar with an eligible bench
  Pokemon on this side can bring any living party member in, see ``roar_level``):
  each side's possible end speeds cover every Speed stage when a
  stage change is possible this turn (a Speed stat effect, AncientPower, Curse, a
  flinched Steadfast), paralysis when it can be inflicted, and the
  neutral speed of a fainted battler (``_faint_check`` resets stages and status). A
  White Herb holder may lose a lowered stage, a Salac Berry holder or a Pluck / Bug Bite user
  gain one, a flinched Steadfast gain one (its foe's flinch move or King's Rock), and a Lum or
  Cheri Berry holder or a Pluck / Bug Bite user lose its paralysis (over-approximate); a
  Salac holder a foe's Roar drags in may enter at +1 (it triggers on the switch-in). The weather lock: Swift Swim doubles in rain only
  ``if (NO_CLOUD_NINE)`` (battle_lib.c:1228-1237), which counts the battlers on the field
  at the turn's end with Cloud Nine / Air Lock and HP left (include/battle/common.h:63-64,
  ``COUNT_ALIVE_BATTLERS``, battle_lib.c:2951-2957); a holder can switch in or out, be
  dragged in or out, or faint during the turn, so while a living holder is in either
  party every end speed is taken with the lock on and off (``_end_speeds``). Lagging Tail
  only changes which of ``speed_tie`` / ``end_speed_tie`` is read when speeds are equal,
  never more.
* ``magnitude[s]``: ``s`` uses a ``MAGNITUDE`` move (``BtlCmd_CalcMagnitudePower``).
  ``crit``/``roll`` are not relevant for ``LEVEL_DAMAGE`` (Night Shade deals its user's
  level with no crit or roll, ``damage.fixed_damage``) nor for ``BIDE`` (the unleash is
  ``2 * storedDamage``, also fixed). A locked side's move is its locked move
  (``order.selected_move``), so the masks above read it.
* ``present[s]``: ``s`` uses Present (``BtlCmd_Present``). It joins ``crit[s, 0]`` and
  ``roll[s, 0]`` in one joint factor (``_present_reads``): the heal path reads neither.
* ``tri_attack[s]``: ``s``'s move runs Tri Attack (its own, or Nature Power's call) and its
  ``secondary[s, 0]`` is relevant (a target that can take none of burn, freeze and paralysis
  drops both). It joins ``secondary[s, 0]`` in one joint factor (``_tri_attack_reads``): read
  only on a proc, 1 + 3 rows instead of 2 * 3; the foe's thaw / full paralysis through it
  also needs the freeze / paralysis value.
* Nature Power: every mask above reads the move the action runs (``order.called_move``),
  except the status checks that read the chosen move (Disable, Taunt).
* ``protect[s]``: ``s`` uses a Protect-effect move with ``protect_count[s] > 0`` (at 0 the
  rate 0xFFFF never fails: a single value) and acts first in the branch (the battler acting
  last fails without the roll mattering, ``BtlCmd_TryProtection``). A success stops the
  foe's blockable move, so ``_protect_reads`` gates that move's own reads on it.
* ``shed_skin[d]``: side ``d``'s active or incoming Pokemon has Shed Skin, or the foe uses
  Roar, side ``d`` has an eligible bench Pokemon and a living Pokemon of side ``d`` has it
  (the drag can bring it in)
  (``step._turn_end_ability``; over-approximate: it only rolls with a status and HP).
* ``roar_level[s]``: ``s`` uses Roar / Whirlwind (``step._use_move``), its level is lower than
  its target's (the foe's active Pokemon or the one it switches in: ``outcome.roar_levels``;
  at a higher or equal level ``BattleSystem_CanWhirlwind`` draws nothing) and the target side
  has an eligible bench Pokemon (with none, ``BtlCmd_TryWhirlwind`` fails before the check).
  ``roar_target[s]``: ``s`` uses Roar and the target side has two or more eligible bench
  Pokemon (with one, the rejection loop's pick is forced). Both are the move's own reads
  (its hit, the first mover's sure flinch, a sure Protect), and ``roar_target`` also needs
  ``roar_level`` to pass (``_roar_reads``). A drag also changes what the end of the turn
  reads: ``end_speed_tie`` counts the speeds of every living Pokemon of the dragged side
  (stage 0, paralyzed or not) and ``shed_skin`` its holders (above), both only when that side
  has an eligible bench Pokemon (``roar_bench >= 1``: with none no drag happens).
* ``quick_claw[s]``: this turn's ``speedRand`` bit of side ``s`` (``step``: the end-of-turn
  order reads it with the battler on the field after the actions, and ``step.qc_kept`` stores
  it for the next turn while a living Quick Claw holder is active or can replace a fainted
  active). Relevant when side ``s`` has a living Quick Claw holder anywhere in its party:
  the active, the Pokemon it switches in, or a bench Pokemon that a replacement can send in
  (sound, not tight: the replacement needs the active to faint this turn, which is not
  judged here). The turn's own order reads ``BattleState.qc_roll`` (``action_order``), so
  every order-conditioned read above already sees Quick Claw through ``Reads.first``.
* ``flinch_item[s, i]``: King's Rock / Razor Fang's roll on hit ``i`` of ``s``'s damaging
  move with ``MOVE_FLAG_TRIGGERS_KINGS_ROCK`` while ``s``'s active Pokemon holds one
  (``effects._kings_rock``; its law is the holder's param, ``outcome.kings_rock_chance``),
  only when ``s`` acts first in the branch (``subscript_flinch_mon.s``'s ``IfMovedThisTurn``)
  and the target has no Inner Focus as ``s`` sees it, as ``fang_flinch``.
* Nothing is relevant once the battle is over (``step`` returns the state,
  ``step.step_events``, step.py:1439).

**Order branches and conditional reads** (rules §10.3 items 1-4). ``speed_tie`` is
enumerated first: each value is a branch (``Branch``) whose order is
``order.action_order``'s, with its own masks (the order-gated reads above), and the
branches' rows are concatenated. Within a branch, a ``ReadPlan`` says when each
element is read, as conditions on the values of other elements (a *joint factor* is a
set of elements linked by such conditions; it lists only the combinations that are
read). ``_move_reads`` gives the implemented mechanics': a move's own reads need its
accuracy roll to hit; a secondary duration needs the proc; a read through the first
mover's infliction needs its hit and proc; and the first mover's sure flinch stops the
second mover's move. Later mechanics add their own with ``register_reads``
(``_present_reads``, ``_protect_reads``). The
brute-force reference is ``outcome_laws(..., loose=True)``: no order or eligibility
gate and no conditional read, so its enumeration is the independent product of
every element any order can read (tests/engine/test_enumerate.py compares the two). It
reuses no gate of the exact laws: ``end_speed_tie`` is drawn whenever the battle goes on,
and the hit law is cut at every threshold variant of ``outcome.hit_thresholds``, possible
or not.

**Damage classes** (rules §10.3 item 5). A hit's ``(crit, roll)`` reaches ``step`` only
through the hit's damage (``damage_for_roll`` in ``effects.play_hit``: HP, Focus Sash,
resist berries, Bide's memory, drain, recoil and the multi-hit loop all read that
number) and the observed crit (``step._hit_events``). So pairs with the same crit flag
and the same damage step to the same state and observation, and are drawn as one class
(``damage_class_law``) whose representative ``(crit, roll)`` the row carries. The
classes come from the turn-start ``OutcomeLaws.damage``; ``OutcomeLaws.compress`` allows
them only for a hit no earlier event of the turn can change the damage inputs of
(``damage_inputs_can_change``, per order branch), of a move whose table is its damage
(not Magnitude, Present or Pursuit) and whose roll no confusion self-hit can read. The brute
force never compresses.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.damage import (
    PHYSICAL,
    ROLLS,
    SPECIAL,
    STATUS,
    decomp_abilities,
    hit_damage_table,
    move_type_of,
    weather_suppressed,
)
from nuzline.engine.effects import can_have_status, protect_blocks
from nuzline.engine.ids import (
    QUICK_CLAW_MODULUS,
    TRI_ATTACK_STATUSES,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    Stat,
    Status,
    Type,
)
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    called_move,
    effective_speed,
    is_item,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import (
    MAX_HITS,
    PRESENT_CUTS,
    PRESENT_HEAL,
    PROTECT_RATES,
    ROAR_LEVEL_VALUES,
    Outcome,
    crit_odds,
    empty_outcome,
    hit_thresholds,
    roar_inputs,
    roar_levels,
)
from nuzline.engine.outcome import (
    kings_rock_chance as outcome_kings_rock_chance,
)
from nuzline.engine.state import STAGE_MAX, STAGE_MIN, BattleState

_I32 = jnp.int32
_E = EffectId

# Pre-merge outcome combinations per (state, action pair); above it the pair is
# sampling-only (plan ruling 3).
WIDTH_CAP = 131072

# ``OutcomeLaws.duration_kind``: the acting side's ``Random`` command (outcome.py:166-174).
DURATION_DEFAULT = 0  # Random 3, 2: 2-5 (sleep, confusion and every other move)
DURATION_TRAP = 1  # Random 3, 3: 3-6 (Bind; Disable: RandNext() % 4 + 3, the same law)
DURATION_TAUNT = 2  # Random 2, 3: 3-5

# Effects whose hits read ``outcome.secondary`` through ``proc`` (effects.py:468).
_PROC_EFFECTS = (
    _E.SECONDARY_STAT_SELF,
    _E.SECONDARY_STAT_FOE,
    _E.SECONDARY_STATUS,
    _E.FLINCH,
    _E.FAKE_OUT,
    _E.RAISE_ALL_STATS_HIT,
    _E.TRAP,
    _E.SECONDARY_CONFUSE,
    _E.FANG_PARALYZE_FLINCH,  # its paralysis roll (effects._fang)
    _E.TRI_ATTACK,  # its status roll (effects._tri_attack)
)
# Effects whose flinch roll is ``outcome.secondary`` (effects._flinch) or
# ``outcome.fang_flinch`` (effects._fang): a flinched Steadfast gains Speed.
_FLINCH_EFFECTS = (_E.FLINCH, _E.FAKE_OUT, _E.FANG_PARALYZE_FLINCH)


class Reads(NamedTuple):
    """Inputs of the conditional reads (``_move_reads``), per side (``bool[2]``) unless
    noted; ``[2, 2]`` is ``[speed_tie branch, side]``."""

    first: jnp.ndarray  # int32[2]: the side acting first under speed_tie = 0, 1
    hit_gated: jnp.ndarray  # its move's reads need its accuracy roll to hit (no Pursuit)
    proc_gated: jnp.ndarray  # its duration and status infliction need a secondary proc
    flinch_sure: jnp.ndarray  # moving first, a sure hit with a proc flinches the foe
    fang: jnp.ndarray  # its flinch roll is fang_flinch (Thunder Fang), not secondary
    present: jnp.ndarray  # it uses Present: its crit and roll need a damage path (no heal)
    tri_attack: jnp.ndarray  # its move runs Tri Attack: its status draw needs the proc
    protect_sure: jnp.ndarray  # it uses Protect and reaches TryProtection whatever the rolls
    protect_blocks: jnp.ndarray  # its move is stopped by a protecting foe (protect_blocks)
    roar: jnp.ndarray  # it uses Roar: its slot pick needs the level check to pass
    confused_own: jnp.ndarray  # it moves confused (counter >= 2) at the start
    confused_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that confuses
    para_own: jnp.ndarray  # it moves paralyzed at the start
    para_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that paralyzes
    freeze_own: jnp.ndarray  # it moves frozen at the start
    freeze_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that freezes
    infatuated_own: jnp.ndarray  # it moves in love at the start (its source stays)
    infatuated_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe Attract that can work


class OutcomeLaws(NamedTuple):
    """Integer law parameters of one (state, action pair), per side, and the relevance
    masks. Fixed shapes; a batch adds a leading axis to every field."""

    hit_thresholds: jnp.ndarray  # int32[2, outcome.HIT_VARIANTS], outcome.hit_thresholds
    hit_possible: jnp.ndarray  # bool[2, outcome.HIT_VARIANTS], which of them can occur
    crit_odds: jnp.ndarray  # int32[2], outcome.crit_odds (1 in N)
    no_crit: jnp.ndarray  # bool[2], Lucky Chant on the defending side (outcome.py:157)
    secondary_chance: jnp.ndarray  # int32[2], tables.move_effect_chance of the move
    duration_kind: jnp.ndarray  # int32[2], DURATION_*
    relevant: Outcome  # bool, each field shaped like the ``Outcome`` field: any branch
    branch_relevant: Outcome  # bool, a leading axis [2]: the branch speed_tie = 0, 1
    reads: Reads  # the conditional reads' inputs
    damage: jnp.ndarray  # int32[2, 2, ROLLS]: per side, hit_damage_table at the turn start
    protect_count: jnp.ndarray  # int32[2]: the turn-start chain (protect's law, 0-3)
    roar_level_weight: jnp.ndarray  # int32[2]: of ROAR_LEVEL_VALUES, how many pass (roar_level)
    roar_bench: jnp.ndarray  # int32[2]: the eligible bench Pokemon Roar picks from (roar_target)
    compress: jnp.ndarray  # bool[2, 2, MAX_HITS]: [branch, side, hit] uses damage classes
    flinch_item_chance: jnp.ndarray  # int32[2]: the King's Rock param (outcome.kings_rock_chance)


class SamplingOnly(Exception):
    """The action pair cannot be enumerated exactly; ``reason`` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _any_of(x, values) -> jnp.ndarray:
    out = jnp.zeros(jnp.shape(x), bool)
    for v in values:
        out = out | (x == v)
    return out


def _inflicts(tables: Tables, moves, status) -> jnp.ndarray:
    """The move can inflict ``status`` on the foe (``STATUS_FOE`` / ``SECONDARY_STATUS`` /
    ``FANG_PARALYZE_FLINCH`` with ``move_param[m, 0]``, effects._inflict, effects._fang; and
    ``TRI_ATTACK`` any of ``TRI_ATTACK_STATUSES``, effects._tri_attack)."""
    effect = tables.move_effect[moves]
    status_effect = _any_of(effect, (_E.STATUS_FOE, _E.SECONDARY_STATUS, _E.FANG_PARALYZE_FLINCH))
    tri = (effect == _E.TRI_ATTACK) & (status in TRI_ATTACK_STATUSES)
    return (status_effect & (tables.move_param[moves, 0] == status)) | tri


def _holds(state: BattleState, actions, abilities) -> jnp.ndarray:
    """``bool[2]``: side ``d``'s active Pokemon, or the one it switches in this turn,
    has one of ``abilities``."""
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.ability.shape[1] - 1)
    sides = jnp.arange(2)
    now = _any_of(state.ability[sides, state.active], abilities)
    later = _any_of(state.ability[sides, incoming], abilities) & is_switch(actions)
    return now | later


def _holds_item(tables: Tables, state: BattleState, actions, items) -> jnp.ndarray:
    """``bool[2]``: as ``_holds``, for held items with one of the effects ``items``."""
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.item.shape[1] - 1)
    sides = jnp.arange(2)
    now = _any_of(tables.item_effect[state.item[sides, state.active]], items)
    later = _any_of(tables.item_effect[state.item[sides, incoming]], items) & is_switch(actions)
    return now | later


def _end_speeds(
    tables: Tables, state: BattleState, actions, side, stage_free, para_free, dragged, lock_free
):
    """``(int32[3 * (27 + 4 * SLOTS)], bool[...])``: the effective speeds ``side`` can have
    after the actions (``step.end_of_turn_order`` on the post-action state, step.py:1251-1261)
    and which of them are possible. Grid: Speed stage -6..6 x paralyzed or not of the
    Pokemon active after the actions, plus its neutral speed (a faint resets stages and
    status, ``step._faint_check``, step.py:948-992). A stage other than the start one needs ``stage_free``, a
    paralysis state other than the start one ``para_free``. Where ``dragged`` (the foe
    can Roar this side), every living party member at stage 0, paralyzed or not: the
    Pokemon the drag brings in (``InitBattleMon`` resets its stages), and a Salac Berry holder
    at +1 too (its switch-in item check can raise its Speed).

    The weather lock: every grid point is also evaluated with the lock on and with it off
    where ``lock_free`` (a Cloud Nine / Air Lock holder can enter, leave or faint this
    turn, so the lock of the turn's end can differ from the one of the turn start, which
    the plain grid point reads from the other side's turn-start battler).
    ``BattleSystem_CompareBattlerSpeed`` doubles Swift Swim in rain only
    ``if (NO_CLOUD_NINE)`` (battle_lib.c:1228-1237; include/battle/common.h:63-64)."""
    switching = is_switch(actions[side])
    mon = jnp.where(switching, actions[side] - MOVE_ACTIONS, state.active[side]).astype(
        state.active.dtype
    )
    stage0 = jnp.where(switching, 0, state.stages[side, Stat.SPE].astype(_I32))
    para0 = state.status[side, mon] == Status.PARALYSIS
    stages = jnp.repeat(jnp.arange(STAGE_MIN, STAGE_MAX + 1, dtype=_I32), 2)
    paras = jnp.tile(jnp.array([False, True]), STAGE_MAX - STAGE_MIN + 1)
    stages = jnp.concatenate([stages, jnp.zeros((1,), _I32)])
    paras = jnp.concatenate([paras, jnp.zeros((1,), bool)])
    possible = (stage_free | (stages == stage0)) & (para_free | (paras == para0))
    possible = possible | (jnp.arange(stages.shape[0]) == stages.shape[0] - 1)  # fainted
    mons = jnp.full(stages.shape, mon, state.active.dtype)
    # A foe's Roar: any living party member, at stage 0, paralyzed or not; a Salac Berry
    # holder also at +1 (its switch-in item check, hooks.on_switch_in, may raise its Speed).
    slots = jnp.arange(state.hp.shape[1])
    living = (state.species[side] != 0) & (state.hp[side] > 0)
    held = state.item[side]
    salac = (tables.item_effect[held] == ItemId.PINCH_BERRY) & (
        tables.item_param[held, 1] == Stat.SPE
    )
    n_slots = slots.shape[0]
    mons = jnp.concatenate([mons, jnp.tile(jnp.repeat(slots, 2), 2).astype(state.active.dtype)])
    stages = jnp.concatenate(
        [stages, jnp.zeros((2 * n_slots,), _I32), jnp.ones((2 * n_slots,), _I32)]
    )
    paras = jnp.concatenate([paras, jnp.tile(jnp.array([False, True]), 2 * n_slots)])
    possible = jnp.concatenate(
        [
            possible,
            jnp.asarray(dragged) & jnp.repeat(living, 2),
            jnp.asarray(dragged) & jnp.repeat(living & salac, 2),
        ]
    )
    # The weather lock: 0 = as on the field of the grid point, 1 = on, 2 = off.
    n = stages.shape[0]
    locks = jnp.repeat(jnp.arange(3), n)
    mons, stages, paras = jnp.tile(mons, 3), jnp.tile(stages, 3), jnp.tile(paras, 3)
    lock_free = jnp.asarray(lock_free, bool)
    possible = jnp.concatenate([possible, possible & lock_free, possible & lock_free])

    def speed(mon, stage, para, lock):
        s = state._replace(
            active=state.active.at[side].set(mon),
            stages=state.stages.at[side, Stat.SPE].set(stage.astype(state.stages.dtype)),
            status=state.status.at[side, mon].set(
                jnp.where(para, Status.PARALYSIS, Status.NONE).astype(state.status.dtype)
            ),
        )
        suppressed = jnp.where(lock == 0, weather_suppressed(tables, s), lock == 1)
        return effective_speed(tables, s, side, suppressed)

    return jax.vmap(speed)(mons, stages, paras, locks), possible


def _branch_masks(foe_first, first_self, base: dict) -> dict:
    """Order-dependent masks of one branch (``outcome_laws``): ``foe_first`` /
    ``first_self`` (``bool[2]``) say whether each side's foe / the side itself acts
    first."""
    b = base
    para_by_foe = b["uses_move"] & foe_first & b["para_inflict"][::-1]
    freeze_by_foe = b["uses_move"] & foe_first & b["freeze_inflict"][::-1]
    confused_by_foe = b["uses_move"] & foe_first & b["confuse_inflict"][::-1]
    infatuated_by_foe = b["uses_move"] & foe_first & b["attract_inflict"][::-1]
    confusable = b["confused_own"] | confused_by_foe
    # A flinch roll is read only when the user moves first (subscript_flinch_mon.s: the
    # target has not moved, effects._flinch). A secondary status or confusion on a target
    # that cannot take it changes nothing (status eligibility), judged on the turn-start
    # state, which is the state of the hit when the user moves first with one hit.
    status_effect = _any_of(
        b["effect"], (_E.SECONDARY_STATUS, _E.FANG_PARALYZE_FLINCH, _E.TRI_ATTACK)
    )
    blocked = first_self & (
        (status_effect & b["status_blocked"])
        | ((b["effect"] == _E.SECONDARY_CONFUSE) & b["confused_blocked"])
    )
    late_flinch = ~first_self & _any_of(b["effect"], (_E.FLINCH, _E.FAKE_OUT))
    secondary = b["secondary"] & ~(blocked | late_flinch)[:, None]
    return {
        "roll": b["crit"] | (confusable[:, None] & (b["hit_index"] == 0)),
        "secondary": secondary,
        # Tri Attack's status draw: read only through its proc (effects._tri_attack), so only
        # where that roll is; a chance-0 Tri Attack (none in either game: the decomp asserts a
        # chance, battle_lib.c:1576) would read it on every landed hit.
        "tri_attack": b["tri_attack"] & ~blocked & (secondary[:, 0] | b["certain"]),
        # Thunder Fang's flinch roll: the same order gate, and Inner Focus (effects._fang).
        "fang_flinch": b["fang_flinch"] & first_self & ~b["flinch_blocked"],
        # King's Rock / Razor Fang (effects._kings_rock): the same gates, per hit.
        "flinch_item": b["flinch_item"] & (first_self & ~b["flinch_blocked"])[:, None],
        # The battler acting last fails Protect whatever its roll (waitingBattlers == 1).
        "protect": b["protect"] & first_self,
        "full_para": b["para_own"] | para_by_foe,
        "confusion_hit": confusable,
        "thaw": b["freeze_own"] | freeze_by_foe,
        # CHECK_STATUS_STATE_ATTRACT (step._use_move): in love now, its own or a faster
        # Attract's.
        "infatuation": b["infatuated_own"] | infatuated_by_foe,
        "infatuated_by_foe": infatuated_by_foe,
        "para_by_foe": para_by_foe,
        "freeze_by_foe": freeze_by_foe,
        "confused_by_foe": confused_by_foe,
    }


# --- Damage classes: what can change a hit's damage inputs mid-turn --------------------
#
# ``damage_inputs_can_change`` must know every implemented id. Each tuple pair below
# splits one enum: *neutral* ids change none of a hit's damage inputs (everything
# ``damage._pre_random`` and ``damage._finish`` read) when they act earlier in the turn,
# except through HP damage, which the HP clause covers; *clause* ids are handled by a
# named clause of the predicate. ``tests/engine/test_damage_classes.py`` fails while an
# enum member is in neither, so a new mechanic must classify its ids (and, for a clause
# id, add the clause) before it can land.

# A foe move acting before the hit (effects.play_hit's effect functions).
INPUT_NEUTRAL_EFFECTS = (
    _E.PLAIN,
    _E.FLINCH,
    _E.HIGH_CRIT,
    _E.DRAIN,  # heals the foe
    _E.STEALTH_ROCK,
    _E.TRAP,
    _E.PURSUIT,
    _E.TAUNT,
    _E.CONFUSE_FOE,
    _E.LEECH_SEED,
    _E.MULTI_HIT,
    _E.RECOIL,  # hurts the foe
    _E.FORESIGHT,  # identifies the attacker's side; the hit reads the defender's flag
    _E.LEVEL_DAMAGE,
    _E.HIDDEN_POWER,
    _E.MAGNITUDE,
    _E.SECONDARY_CONFUSE,
    _E.DISABLE,
    _E.BIDE,
    _E.ROLLOUT,
    _E.BRICK_BREAK,  # clears the attacker's side's screens; the hit reads the defender's
    _E.RECOVER,
    _E.FAKE_OUT,
    _E.SELF_DESTRUCT,  # the foe faints: the hit has no target
    _E.FLAIL,
    _E.AQUA_RING,
    # Hurts or heals the foe (effects._present): only the foe's HP, which the HP clause
    # covers (Flail and the pinch abilities read it; nothing else in the damage does).
    _E.PRESENT,
    # Changes the user's protect_count and the turn's protecting flag: no damage input. A
    # success removes the later hit altogether (step._use_move: it does not connect), so
    # whatever (crit, roll) class the hit would have drawn, the state is the same.
    _E.PROTECT,
    # Sets the foe's infatuated, read only by its attract check, which comes before its move
    # and removes the hit altogether (step._use_move), like a flinch: no damage input.
    _E.ATTRACT,
    # Never the effect read here: the predicate reads the move each action runs
    # (order.called_move), Nature Power's call; an unresolved one (unknown terrain, refused
    # by coverage) calls MOVE_NONE and does nothing.
    _E.NATURE_POWER,
)
INPUT_CLAUSE_EFFECTS = (
    _E.STAT_SELF,  # the foe's Defense / Sp. Def
    _E.SECONDARY_STAT_SELF,
    _E.STAT_FOE,  # the attacker's Attack / Sp. Atk
    _E.SECONDARY_STAT_FOE,
    _E.RAISE_ALL_STATS_HIT,
    _E.CURSE,  # the foe's Defense
    _E.STATUS_FOE,  # the attacker's burn, or any status under Guts
    _E.SECONDARY_STATUS,
    _E.REFLECT,  # the defender's side's screens
    _E.LIGHT_SCREEN,
    _E.MUD_SPORT,  # any side's sport (damage.py :6811, :6815)
    _E.WATER_SPORT,
    # The attacker's paralysis: an input under Guts only. Its Speed change reorders the next
    # turn, not this one (order.action_order is fixed at the turn start); its flinch stops
    # the hit, which changes no input.
    _E.FANG_PARALYZE_FLINCH,
    # A burn (the attacker's physical non-Guts hit) or any status (Guts): the status clause.
    _E.TRI_ATTACK,
    # Drags the side's Pokemon out (step._use_move): the hit would be a newcomer's, but the
    # dragged battler never acts after it this turn (UpdateAfterSwitch ends its action), so
    # no such hit exists; the roar clause keeps one uncompressed anyway.
    _E.ROAR,
    # The foe's own types (effects._camouflage): the type chart of the side's later hit
    # (damage._finish) and the Rock Sp. Def boost in sand: the camouflage clause.
    _E.CAMOUFLAGE,
    # Takes the side's berry: a stat pinch berry raises the eater's stat (hooks.pluck_berry),
    # the defending one of the side's later hit: the pluck clause. Taking any other berry
    # changes no input of the side's hit, and only removes a later cure.
    _E.EAT_BERRY,
)
INPUT_NEUTRAL_ABILITIES = (
    AbilityId.NONE,
    AbilityId.STURDY,
    AbilityId.ROCK_HEAD,
    AbilityId.MOLD_BREAKER,
    AbilityId.RIVALRY,  # genders never change
    AbilityId.KEEN_EYE,
    AbilityId.SIMPLE,  # scales stages; a stage change is its own clause
    AbilityId.UNAWARE,
    AbilityId.NATURAL_CURE,  # on switch-out
    AbilityId.INNER_FOCUS,
    AbilityId.STEADFAST,  # Speed
    AbilityId.INSOMNIA,
    AbilityId.CLOUD_NINE,  # changes only when its holder faints: no hit follows
    AbilityId.COMPOUNDEYES,
    AbilityId.SHED_SKIN,  # turn end
    AbilityId.SAND_VEIL,
    AbilityId.SNOW_CLOAK,
    AbilityId.SWIFT_SWIM,
    AbilityId.LIMBER,
    AbilityId.ADAPTABILITY,
    AbilityId.SOLID_ROCK,
    # Infatuates the attacker after its hit: read at its next attract check, not a damage
    # input of a later hit of the move (effects._contact_ability).
    AbilityId.CUTE_CHARM,
    # Halves a pinch berry's param: the berry's trigger is its own item clause.
    AbilityId.GLUTTONY,
)
INPUT_CLAUSE_ABILITIES = (
    AbilityId.GUTS,  # status clause
    AbilityId.BLAZE,  # HP clause (pinch power)
    AbilityId.TORRENT,
    AbilityId.OVERGROW,
    AbilityId.FLAME_BODY,  # contact clause (burn)
    AbilityId.STATIC,  # contact clause (a status under Guts)
    AbilityId.POISON_POINT,
    AbilityId.ROUGH_SKIN,  # the attacker's HP: HP clause
    AbilityId.INTIMIDATE,  # on switch-in: switch clause
)
INPUT_NEUTRAL_ITEMS = (
    ItemId.NONE,
    ItemId.FOCUS_SASH,  # after the damage (effects.play_hit)
    ItemId.LAGGING_TAIL,
    ItemId.RESIST_BERRY,  # after the damage (effects.resist_berry_damage)
    # Changes only the order: the turn's is fixed at its start (action_order, qc_roll) and the
    # end-of-turn order (outcome.quick_claw) runs after every hit.
    ItemId.QUICK_CLAW,
    ItemId.NO_EFFECT_BERRY,  # no effect (Pluck eats it, after that hit's damage)
    # Damage inputs that stay fixed through the turn: never consumed (no implemented path
    # removes them), read with the species, move type and class (damage._item_modifiers),
    # or with the type chart (Expert Belt), whose own inputs have their clauses.
    ItemId.TYPE_BOOST,
    ItemId.LIGHT_BALL,
    ItemId.THICK_CLUB,
    ItemId.METAL_POWDER,
    ItemId.DEEP_SEA_TOOTH,
    ItemId.DEEP_SEA_SCALE,
    ItemId.SOUL_DEW,
    ItemId.ADAMANT_ORB,
    ItemId.LUSTROUS_ORB,
    ItemId.GRISEOUS_ORB,
    ItemId.MUSCLE_BAND,
    ItemId.WISE_GLASSES,
    ItemId.EXPERT_BELT,
    # The Choice lock changes only which move a later turn may choose; Band / Specs are a
    # fixed stat factor, Scarf a speed factor of the order fixed at the turn's start (the
    # end-of-turn order reads it after every hit).
    ItemId.CHOICE_BAND,
    ItemId.CHOICE_SPECS,
    ItemId.CHOICE_SCARF,
    # A fixed damage factor; its recoil (after the holder's last hit) lowers only the
    # holder's HP, which no later hit of the turn reads as a damage input but the holder's
    # own (Flail, the pinch abilities: the HP clause), and the holder has acted.
    ItemId.LIFE_ORB,
    # Like Life Orb: Shell Bell heals its holder after its last hit (effects.shell_bell_heal).
    ItemId.SHELL_BELL,
    # Confusion is no damage input (effects.play_hit), so its cure changes none.
    ItemId.PERSIM_BERRY,
    # Eaten when its holder's move starts (step._use_move): it changes only the order, fixed
    # at the turn's start, and the end-of-turn order, read after every hit.
    ItemId.CUSTAP_BERRY,
    # The end of the turn, after every hit (step.END_OF_TURN_ORDER).
    ItemId.LEFTOVERS,
    ItemId.BLACK_SLUDGE,
    ItemId.TOXIC_ORB,
    ItemId.FLAME_ORB,
    # Its contact transfer (effects.sticky_barb_transfer) only moves an item with no damage
    # effect; its damage is at the turn's end.
    ItemId.STICKY_BARB,
    # The accuracy check (outcome._rate) and the flinch roll (effects._kings_rock): a hit
    # that happens is not changed, a flinch removes the hit altogether.
    ItemId.BRIGHT_POWDER,
    ItemId.WIDE_LENS,
    ItemId.ZOOM_LENS,
    ItemId.KINGS_ROCK,
    # Only the screen's duration (effects._screen); the screen itself is its move's clause.
    ItemId.LIGHT_CLAY,
)
INPUT_CLAUSE_ITEMS = (
    ItemId.ORAN_BERRY,  # the attacker's HP: HP clause
    ItemId.SITRUS_BERRY,
    ItemId.BERRY_JUICE,
    ItemId.BIG_ROOT,
    ItemId.CHESTO_BERRY,  # item clause (status cure)
    ItemId.PECHA_BERRY,
    ItemId.LUM_BERRY,
    ItemId.CHERI_BERRY,
    ItemId.RAWST_BERRY,  # item clause (a burn cure, or any cure under Guts)
    ItemId.ASPEAR_BERRY,
    ItemId.WHITE_HERB,  # item clause (a lowered stage restored)
    ItemId.PINCH_BERRY,  # item clause (a stage raised in a pinch); pluck clause
)
_STATUS_CURES = (
    ItemId.CHERI_BERRY,
    ItemId.CHESTO_BERRY,
    ItemId.PECHA_BERRY,
    ItemId.RAWST_BERRY,
    ItemId.ASPEAR_BERRY,
    ItemId.LUM_BERRY,
)
_BURN_CURES = (ItemId.RAWST_BERRY, ItemId.LUM_BERRY)

# The attacker's own move effect, for ``OutcomeLaws.compress``: a hit of a
# ``TABLE_DAMAGE_EFFECTS`` move deals exactly ``hit_damage_table``'s entry for its
# ``(crit, roll)`` at the hit's state (``effects.play_hit`` calls ``damage_for_roll`` at the
# default ``power_mul``, which equals ``damage_table`` for these; Hidden Power, Rollout and
# Flail read their power from the state there too). Every other effect is
# ``NON_TABLE_DAMAGE_EFFECTS``: no roll (status moves, ``LEVEL_DAMAGE``, ``BIDE``), a drawn
# power (``MAGNITUDE``, ``PRESENT``) or a power the table does not see (``PURSUIT``'s
# pre-switch x2).
# An allowlist: a new effect is never compressed until it is classified here
# (``tests/engine/test_damage_classes.py`` fails while it is in neither tuple).
TABLE_DAMAGE_EFFECTS = (
    _E.PLAIN,
    _E.SECONDARY_STAT_SELF,
    _E.SECONDARY_STAT_FOE,
    _E.SECONDARY_STATUS,
    _E.FLINCH,
    _E.HIGH_CRIT,
    _E.DRAIN,
    _E.TRAP,
    _E.MULTI_HIT,
    _E.RECOIL,
    _E.HIDDEN_POWER,
    _E.SECONDARY_CONFUSE,
    _E.EAT_BERRY,
    _E.ROLLOUT,
    _E.BRICK_BREAK,
    _E.FAKE_OUT,
    _E.RAISE_ALL_STATS_HIT,
    _E.SELF_DESTRUCT,
    _E.FLAIL,
    _E.FANG_PARALYZE_FLINCH,
    _E.TRI_ATTACK,  # a plain hit at its table power; the status is after the damage
)
NON_TABLE_DAMAGE_EFFECTS = (
    _E.STAT_SELF,
    _E.STAT_FOE,
    _E.STATUS_FOE,
    _E.STEALTH_ROCK,
    _E.PURSUIT,
    _E.TAUNT,
    _E.CONFUSE_FOE,
    _E.LEECH_SEED,
    _E.WATER_SPORT,
    _E.FORESIGHT,
    _E.LEVEL_DAMAGE,
    _E.MAGNITUDE,
    _E.MUD_SPORT,
    _E.DISABLE,
    _E.BIDE,
    _E.LIGHT_SCREEN,
    _E.REFLECT,
    _E.RECOVER,
    _E.CURSE,
    _E.AQUA_RING,
    _E.PRESENT,  # its power is drawn (40/80/120, Outcome.present); the table has power 1
    _E.PROTECT,  # a status move: no damage
    _E.ATTRACT,  # a status move: no damage
    _E.CAMOUFLAGE,  # a status move: no damage
    _E.ROAR,  # a status move: no damage
    # Its table power is 0: its damage is the called move's. outcome_laws reads the called
    # move (order.called_move), whose own effect decides (Tri Attack's is a table hit), so
    # this entry only keeps an unresolved Nature Power (unknown terrain) uncompressed.
    _E.NATURE_POWER,
)


def damage_inputs_can_change(tables: Tables, state: BattleState, actions, foe_first) -> jnp.ndarray:
    """``bool[2, MAX_HITS]``: ``[side, hit]`` can deal damage other than
    ``hit_damage_table`` at the turn start says, for some ``(crit, roll)``, because an
    event earlier in the turn can change one of its damage inputs (``damage._pre_random``,
    ``damage._finish``). ``foe_first`` (``bool[2]``): the side's foe acts before it (one
    order branch). Sound, not tight: true whenever some clause below can apply.

    Inputs and the events that can change them before ``[side, hit]``:

    * **switch / item**: the foe switches (a new target, Intimidate, entry hazards,
      weather abilities) or uses a trainer item, before the side's move;
    * **foe move**: an earlier foe move with an unclassified effect (not in
      ``INPUT_NEUTRAL_EFFECTS`` / ``INPUT_CLAUSE_EFFECTS``), or one that changes: the
      side's attacking stat stage (``STAT_FOE``), the foe's defending stage
      (``STAT_SELF``, ``RAISE_ALL_STATS_HIT``, Curse's Defense), the side's status when
      it matters (a burn on a physical non-Guts move, any status under Guts, Thunder Fang's
      paralysis and Tri Attack's statuses included), the foe's screen for this category, a
      sport for this move type, or the foe's own types (Camouflage: the type chart and the
      Rock Sp. Def boost read them); or the side's active Pokemon itself (the foe's Roar
      drags it out; no hit of a dragged battler follows in Gen 4, its action ends, so this
      roar clause is conservative);
    * **HP**: Flail's power band and Overgrow / Blaze / Torrent (damage.py :6819-6837)
      read the user's HP: any earlier foe move or earlier hit of the same move;
    * **contact**: on a later hit, the foe's Flame Body burn or Static / Poison Point
      status (effects._contact_ability), when the status matters;
    * **items** (``hooks.held_item_check`` after every hit): a status cure on the side's
      Pokemon when the status matters, a White Herb on the attacker or the defender whose
      stage the hit reads can be lowered (below 0 at the start, or a stat move on that stat
      or a switch this turn), a stat pinch berry of the side on its attacking stat or of the
      foe on its defending stat;
    * **pluck**: the foe's earlier Pluck / Bug Bite eats the side's stat pinch berry of the
      foe's defending stat (``hooks.pluck_berry`` raises the eater's stage);
    * **own move**: a later hit of a move other than ``MULTI_HIT`` (none has several hits
      today), a Guts user that wakes or thaws first (its status goes to none), and
      Selfdestruct (its user's HP goes to 0 before the hit).

    Extending (Tasks 4-11): a mechanic that lets anything above (or a new input) change
    mid-turn classifies its ids in the ``INPUT_*`` tuples above and adds a clause here,
    with a brute-force equality test in ``tests/engine/test_damage_classes.py``.

    A mechanic whose *own* hit damage is not ``hit_damage_table``'s (a drawn power such as
    Present's 40/80/120, a move chosen at use time such as Nature Power's, a power
    multiplier) puts its effect in ``NON_TABLE_DAMAGE_EFFECTS``, or refines the classes
    over every variant itself. A joint factor does not turn compression off by itself:
    ``_damage_classes`` only skips a hit whose crit or roll *other* elements depend on, so a
    factor that gates crit / roll *on* another element (``present`` -> crit) leaves the hit
    compressed with the turn-start table."""
    actions = jnp.asarray(actions, _I32)
    sides = jnp.arange(2)
    mons = state.active
    live = state.over == 0
    # The move each action runs (Nature Power's call, order.called_move).
    chosen = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    moves = jnp.stack([called_move(tables, state, chosen[side]) for side in (0, 1)])
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    effect = tables.move_effect[moves]
    param0 = tables.move_param[moves, 0]
    category = tables.move_category[moves]
    move_type = jnp.stack([move_type_of(tables, state, side, moves[side]) for side in (0, 1)])
    ability = state.ability[sides, mons]
    item = tables.item_effect[state.item[sides, mons]]
    status = state.status[sides, mons]
    contact = (tables.move_flags[moves] & MoveFlag.CONTACT) != 0

    physical = category == PHYSICAL
    special = category == SPECIAL
    guts = physical & (ability == AbilityId.GUTS)  # any status counts (damage.py :6793)
    burnable = physical & (ability != AbilityId.GUTS)  # a burn halves it (:6978)
    attack_stat = jnp.where(physical, Stat.ATK, Stat.SPA)
    defense_stat = jnp.where(physical, Stat.DEF, Stat.SPD)
    pinch_ability = (
        ((move_type == Type.GRASS) & (ability == AbilityId.OVERGROW))
        | ((move_type == Type.FIRE) & (ability == AbilityId.BLAZE))
        | ((move_type == Type.WATER) & (ability == AbilityId.TORRENT))
    )
    hp_sensitive = (effect == _E.FLAIL) | pinch_ability

    held = state.item[sides, mons]
    pinch_stat = tables.item_param[held, 1]
    pinch = item == ItemId.PINCH_BERRY
    foe_effect, foe_param0 = effect[::-1], param0[::-1]
    unclassified = ~_any_of(foe_effect, INPUT_NEUTRAL_EFFECTS + INPUT_CLAUSE_EFFECTS)
    status_hit = _any_of(
        foe_effect,
        (_E.STATUS_FOE, _E.SECONDARY_STATUS, _E.FANG_PARALYZE_FLINCH, _E.TRI_ATTACK),
    )
    foe_burns = _inflicts(tables, moves[::-1], Status.BURN)
    foe_move = uses_move[::-1] & (
        unclassified
        | (_any_of(foe_effect, (_E.STAT_FOE, _E.SECONDARY_STAT_FOE)) & (foe_param0 == attack_stat))
        | (
            _any_of(foe_effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF))
            & (foe_param0 == defense_stat)
        )
        | (foe_effect == _E.RAISE_ALL_STATS_HIT)
        | ((foe_effect == _E.CURSE) & physical)
        | (status_hit & (guts | (burnable & foe_burns)))
        | ((foe_effect == _E.REFLECT) & physical)
        | ((foe_effect == _E.LIGHT_SCREEN) & special)
        | ((foe_effect == _E.MUD_SPORT) & (move_type == Type.ELECTRIC))
        | ((foe_effect == _E.WATER_SPORT) & (move_type == Type.FIRE))
        | (foe_effect == _E.CAMOUFLAGE)
        | (foe_effect == _E.ROAR)
        | ((foe_effect == _E.EAT_BERRY) & pinch & (pinch_stat == defense_stat))
        | hp_sensitive
    )
    foe_action = live & (is_switch(actions)[::-1] | is_item(actions)[::-1] | foe_move)

    # A White Herb resets its holder's lowered stages to 0 (hooks.held_item_check,
    # battle_lib.c:4875-4887). The hit reads two stages only, the attacker's Attack / Sp. Atk
    # and the defender's Defense / Sp. Def (battle_lib.c:6639-6642, damage._pre_random), so
    # the herb matters when the side's own herb can meet a lowered attacking stage or the
    # foe's a lowered defending stage: one below 0 at the turn start, or one an action this
    # turn can lower (a stat move on that stat by either side, whatever its target or sign;
    # a switch, for Intimidate, hooks.intimidate). Over-approximate: no hit of a move changes
    # a stage today (MULTI_HIT has no stat effect), and a lowering by the foe's earlier move
    # is its own clause above.
    herb = item == ItemId.WHITE_HERB
    stat_move = uses_move & _any_of(
        effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF, _E.STAT_FOE, _E.SECONDARY_STAT_FOE)
    )
    any_switch = jnp.any(live & is_switch(actions))

    def lowerable(stat) -> jnp.ndarray:
        """``bool[2]``: per side ``s``, a move this turn changes stage ``stat[s]`` (the
        stage ``s``'s hit reads) of some battler, or a Pokemon switches in."""
        moved = jnp.stack([jnp.any(stat_move & (param0 == stat[s])) for s in (0, 1)])
        return moved | any_switch

    attack_stage = state.stages[sides, attack_stat] < 0
    defense_stage = state.stages[sides[::-1], defense_stat] < 0
    herb_inputs = (herb & (attack_stage | lowerable(attack_stat))) | (
        herb[::-1] & (defense_stage | lowerable(defense_stat))
    )
    # A stat pinch berry (hooks.held_item_check) raises its holder's stage after a hit or a
    # residual: the side's own on the stage its hit reads (Attack / Sp. Atk), or the foe's
    # on its defending one (Defense / Sp. Def).
    pinch_inputs = (pinch & (pinch_stat == attack_stat)) | (
        pinch[::-1] & (pinch_stat[::-1] == defense_stat)
    )
    items = (
        herb_inputs
        | pinch_inputs
        | (guts & _any_of(item, _STATUS_CURES))
        | (burnable & _any_of(item, _BURN_CURES))
    )

    foe_ability = ability[::-1]
    contact_status = contact & (
        ((foe_ability == AbilityId.FLAME_BODY) & (guts | burnable))
        | (_any_of(foe_ability, (AbilityId.STATIC, AbilityId.POISON_POINT)) & guts)
    )
    own_hits = hp_sensitive | (effect != _E.MULTI_HIT) | contact_status
    own_pre_move = (guts & _any_of(status, (Status.SLEEP, Status.FREEZE))) | (
        effect == _E.SELF_DESTRUCT
    )

    later = jnp.arange(MAX_HITS)[None, :] >= 1
    before = jnp.asarray(foe_first, bool) & (foe_action | items)
    return own_pre_move[:, None] | before[:, None] | (later & (own_hits | items)[:, None])


def outcome_laws(
    tables: Tables, state: BattleState, actions, *, loose: bool = False
) -> OutcomeLaws:
    """The law parameters and relevance masks of ``step(tables, state, actions, .)``
    (module docstring). ``loose`` (tests only: the brute-force reference) drops every
    order and eligibility gate and every conditional read: each element any order or
    event can read is relevant in both branches, and ``reads`` gates nothing."""
    actions = jnp.asarray(actions, _I32)
    live = state.over == 0
    sides = jnp.arange(2)
    # ``chosen``: the move each action selects (its status checks: Disable, Taunt); ``moves``:
    # the move it runs (Nature Power's call, order.called_move), whose draws these are.
    chosen = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
    moves = jnp.stack([called_move(tables, state, chosen[side]) for side in (0, 1)])
    switch, item = is_switch(actions), is_item(actions)
    uses_move = live & ~switch & ~item  # bool[2]
    foe_moves = uses_move[::-1]  # the other side uses a move
    effect = tables.move_effect[moves]
    foe_move = moves[::-1]
    chance = tables.move_effect_chance[moves].astype(_I32)
    damaging = (tables.move_category[moves] != STATUS) & (tables.move_power[moves] != 0)
    # LEVEL_DAMAGE and Bide's unleash: no crit, no roll (damage.fixed_damage)
    rolled = damaging & (effect != _E.LEVEL_DAMAGE) & (effect != _E.BIDE)
    contact = (tables.move_flags[moves] & MoveFlag.CONTACT) != 0
    multi = effect == _E.MULTI_HIT
    fixed_hits = tables.move_param[moves, 0].astype(_I32)
    drawn_hits = multi & (fixed_hits == 0)  # the 2-5 law
    max_hits = jnp.where(multi, jnp.where(drawn_hits, MAX_HITS, fixed_hits), 1)
    mons = state.active
    status = state.status[sides, mons]

    hit_index = jnp.arange(MAX_HITS)[None, :]
    lands = uses_move[:, None] & (hit_index < max_hits[:, None])  # [2,5]
    hit = uses_move[:, None] & (tables.move_accuracy[moves] != 0)[:, None] & (hit_index == 0)
    crit = lands & rolled[:, None]
    secondary = lands & ((chance != 0) & _any_of(effect, _PROC_EFFECTS))[:, None]
    holders = _holds(
        state, actions, (AbilityId.STATIC, AbilityId.POISON_POINT, AbilityId.FLAME_BODY)
    )
    # BtlCmd_TryAttract (effects._attractable): different genders, neither genderless.
    gender = state.gender.astype(_I32)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.gender.shape[1] - 1)

    def lovers(a, b):
        return (a != b) & (a != 2) & (b != 2)

    attacker_gender = gender[sides, mons][::-1]  # side d's attacker: the other side's active
    # The attacker is not in love at the start, or the holder's side switches: the source
    # leaves first (step._source_leaves) and a Cute Charm switch-in can then infatuate it.
    calm = (state.infatuated[::-1] == 0) | switch
    charm_now = (state.ability[sides, mons] == AbilityId.CUTE_CHARM) & lovers(
        gender[sides, mons], attacker_gender
    )
    charm_later = (
        switch
        & (state.ability[sides, incoming] == AbilityId.CUTE_CHARM)
        & lovers(gender[sides, incoming], attacker_gender)
    )
    charm = (charm_now | charm_later) & calm
    if loose:
        charm = _holds(state, actions, (AbilityId.CUTE_CHARM,))
    holders = holders | charm
    contact_proc = (lands & (damaging & contact & holders[::-1])[:, None])[::-1]
    duration = uses_move & (
        _any_of(effect, (_E.TAUNT, _E.TRAP, _E.DISABLE, _E.CONFUSE_FOE, _E.SECONDARY_CONFUSE))
        | _inflicts(tables, moves, Status.SLEEP)
    )

    # Status eligibility of the foe's active Pokemon, the target when the user moves
    # first (a switch or an item would go first): effects._inflict (subscript_paralyze.s,
    # subscript_fall_asleep.s, can_have_status) and effects._secondary_confuse
    # (subscript_confuse.s fails on a confused target). One hit only (an earlier hit's
    # berry could cure the target). Insomnia is never a block: its "stayed awake" reveal
    # (ai_ability_known) runs on a proc; Limber's runs only for a direct status move.
    param0 = tables.move_param[moves, 0]
    attacker_ability = state.ability[sides, mons]
    ignorable = jnp.where(
        attacker_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, attacker_ability[::-1]
    )
    can_take = jnp.stack(
        [can_have_status(tables, state, 1 - side, mons[1 - side], param0[side]) for side in (0, 1)]
    )
    limber = (param0 == Status.PARALYSIS) & (ignorable == AbilityId.LIMBER)
    insomnia = (param0 == Status.SLEEP) & (ignorable == AbilityId.INSOMNIA)
    single = ~multi

    def takes(status) -> jnp.ndarray:
        return jnp.stack(
            [can_have_status(tables, state, 1 - side, mons[1 - side], status) for side in (0, 1)]
        )

    # Tri Attack: blocked when the target can take none of its three statuses (an existing
    # status; or Fire, Ice and Limber together, subscript_paralyze.s:5).
    tri_open = (
        takes(Status.BURN)
        | takes(Status.FREEZE)
        | (takes(Status.PARALYSIS) & (ignorable != AbilityId.LIMBER))
    )
    status_blocked = single & jnp.where(
        effect == _E.TRI_ATTACK, ~tri_open, ~insomnia & (~can_take | limber)
    )
    confused_blocked = single & (state.confusion[::-1] > 0)
    # subscript_flinch_mon.s:7: Inner Focus (ignorable) stops Thunder Fang's flinch.
    flinch_blocked = ignorable == AbilityId.INNER_FOCUS
    if loose:
        status_blocked = confused_blocked = flinch_blocked = jnp.zeros((2,), bool)
    confuses = _any_of(effect, (_E.CONFUSE_FOE, _E.SECONDARY_CONFUSE))
    # Attract on the foe's active Pokemon (effects._attract); in love at the start, read
    # unless the source switches out first (step._source_leaves).
    foe_calm = state.infatuated[::-1] == 0
    attract = uses_move & (effect == _E.ATTRACT)
    infatuated_own = uses_move & (state.infatuated != 0)
    if not loose:
        attract = attract & lovers(gender[sides, mons], gender[sides, mons][::-1]) & foe_calm
        infatuated_own = infatuated_own & ~switch[::-1]
    # King's Rock / Razor Fang (effects._kings_rock): the attacker's item, fixed through the turn.
    kings_rock = outcome_kings_rock_chance(tables, state)
    kings_move = (tables.move_flags[moves] & MoveFlag.KINGS_ROCK) != 0
    base = {
        "uses_move": uses_move,
        "effect": effect,
        "hit_index": hit_index,
        "crit": crit,
        "secondary": secondary,
        "status_blocked": status_blocked,
        "confused_blocked": confused_blocked,
        "fang_flinch": lands[:, 0] & (effect == _E.FANG_PARALYZE_FLINCH) & (chance != 0),
        "tri_attack": lands[:, 0] & (effect == _E.TRI_ATTACK),
        "flinch_item": lands & (damaging & kings_move & (kings_rock > 0))[:, None],
        "certain": chance == 0,
        "flinch_blocked": flinch_blocked,
        "protect": uses_move & (effect == _E.PROTECT) & (state.protect_count > 0),
        "para_own": uses_move & (status == Status.PARALYSIS),
        "freeze_own": uses_move & (status == Status.FREEZE),
        "confused_own": uses_move & (state.confusion >= 2),
        "para_inflict": uses_move & _inflicts(tables, moves, Status.PARALYSIS) & ~status_blocked,
        "freeze_inflict": uses_move & _inflicts(tables, moves, Status.FREEZE) & ~status_blocked,
        "confuse_inflict": uses_move & confuses & ~confused_blocked,
        "infatuated_own": infatuated_own,
        "attract_inflict": attract,
    }

    # Order branches: order.action_order under speed_tie = 0 and 1. speed_tie is read
    # only through the order, so it is relevant exactly when the two orders differ.
    tie_outcomes = [empty_outcome()._replace(speed_tie=jnp.int8(t)) for t in (0, 1)]
    firsts = jnp.stack([action_order(tables, state, actions, o)[0] for o in tie_outcomes])
    firsts = firsts.astype(_I32)
    if loose:
        # order.action_order: priority only between two moves; a tie only between two
        # actions of the same kind.
        priority = jnp.where(
            uses_move[0] & uses_move[1], tables.move_priority[moves].astype(_I32), 0
        )
        same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
        speed_tie = (
            live
            & same_kind
            & (priority[0] == priority[1])
            & (effective_speed(tables, state, 0) == effective_speed(tables, state, 1))
        )
        anyway = jnp.ones((2,), bool)
        branches = [_branch_masks(anyway, anyway, base)] * 2
    else:
        speed_tie = live & (firsts[0] != firsts[1])
        branches = [_branch_masks(firsts[t] == 1 - sides, firsts[t] == sides, base) for t in (0, 1)]

    own_speed = (_any_of(effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF)) & (param0 == Stat.SPE)) | (
        _any_of(effect, (_E.RAISE_ALL_STATS_HIT, _E.CURSE))
    )
    foe_speed = _any_of(effect, (_E.STAT_FOE, _E.SECONDARY_STAT_FOE)) & (param0 == Stat.SPE)
    # subscript_flinched.s:10-14: a flinched Steadfast Pokemon gains +1 Speed (its foe's
    # flinch move, or its King's Rock on a flagged damaging move).
    steadfast = _holds(state, actions, (AbilityId.STEADFAST,)) & foe_moves
    kings = (kings_rock > 0) & damaging & kings_move
    flinches = _any_of(tables.move_effect[foe_move], _FLINCH_EFFECTS) | kings[::-1]
    steadfast = steadfast & flinches
    # A White Herb restores a lowered Speed stage (hooks.held_item_check); a Salac Berry
    # raises it, held or plucked (hooks.pluck_berry: any Pluck / Bug Bite user, over-approx).
    herb = _holds_item(tables, state, actions, (ItemId.WHITE_HERB,))
    pinch_held = _holds_item(tables, state, actions, (ItemId.PINCH_BERRY,))
    plucks = uses_move & (effect == _E.EAT_BERRY)
    stage_free = (
        (uses_move & own_speed)
        | (foe_moves & foe_speed[::-1])
        | steadfast
        | herb
        | pinch_held
        | plucks
    )
    static = _holds(state, actions, (AbilityId.STATIC,))
    # A Lum or Cheri Berry cures paralysis, held or plucked (hooks.held_item_check,
    # pluck_berry).
    lum = _holds_item(tables, state, actions, (ItemId.LUM_BERRY, ItemId.CHERI_BERRY))
    para_free = (
        (foe_moves & _inflicts(tables, foe_move, Status.PARALYSIS))
        | (uses_move & damaging & contact & static[::-1])
        | lum
        | plucks
    )
    # Roar / Whirlwind (step._use_move): BtlCmd_TryWhirlwind draws nothing with no eligible
    # bench Pokemon, BattleSystem_CanWhirlwind nothing for a user of at least the target's
    # level (outcome.roar_inputs, roar_levels), and the pick is forced with one.
    roar = uses_move & (effect == _E.ROAR)
    roar_weight, roar_bench = roar_inputs(state, actions)
    user_level, target_level = roar_levels(state, actions)
    roar_level = roar & (user_level < target_level) & (roar_bench >= 1)
    roar_target = roar & (roar_bench >= 2)
    # The foe can drag this side's Pokemon out: its Roar with an eligible bench Pokemon here
    # (BtlCmd_TryWhirlwind fails at eligibleMons <= 1, battle_script.c:5285-5297, before
    # any drag).
    dragged = (roar & (roar_bench >= 1))[::-1]
    if loose:
        # The brute force: both draws whenever the side uses Roar, whatever the levels and the
        # bench (the pick's law is then over max(bench, 1) values), and every living party
        # member's speed on both sides whenever anyone uses it.
        roar_level = roar_target = roar
        dragged = jnp.broadcast_to(jnp.any(roar), (2,))
    # The weather lock of the turn's end (NO_CLOUD_NINE, include/battle/common.h:63-64)
    # counts the battlers on the field with Cloud Nine / Air Lock and HP left
    # (COUNT_ALIVE_BATTLERS, battle_lib.c:2951-2957). A living holder anywhere in either
    # party can change it this turn: by a switch in or out, a Roar's drag in or out, or a
    # faint (over-approximate; nothing in the engine revives, so a fainted holder never
    # counts again).
    abilities = decomp_abilities(tables, state)
    lock_holder = (abilities == WEATHER_ABILITY_IDS["Cloud Nine"]) | (
        abilities == WEATHER_ABILITY_IDS["Air Lock"]
    )
    lock_free = jnp.any(lock_holder & (state.species != 0) & (state.hp > 0))
    speeds = [
        _end_speeds(
            tables,
            state,
            actions,
            side,
            stage_free[side],
            para_free[side],
            dragged[side],
            lock_free,
        )
        for side in (0, 1)
    ]
    (speed0, ok0), (speed1, ok1) = speeds
    end_speed_tie = live & jnp.any(
        ok0[:, None] & ok1[None, :] & (speed0[:, None] == speed1[None, :])
    )
    if loose:
        # The brute force: the end-of-turn tie roll whenever the battle goes on, with no
        # speed gate (one binary element: at most x2).
        end_speed_tie = live
    # Shed Skin of the Pokemon on the field at the turn's end: the active or incoming one, or
    # any living one a foe's Roar can drag in.
    living = (state.species != 0) & (state.hp > 0)
    shed_party = jnp.any(living & (state.ability == AbilityId.SHED_SKIN), axis=1)
    shed_skin = live & (_holds(state, actions, (AbilityId.SHED_SKIN,)) | (dragged & shed_party))
    if loose:
        shed_skin = shed_skin | (live & jnp.any(roar) & jnp.any(shed_party))

    # Quick Claw (step.qc_kept, end_of_turn_order): a living holder anywhere in the party.
    claw = tables.item_effect[state.item] == ItemId.QUICK_CLAW
    quick_claw = live & jnp.any(claw & (state.species != 0) & (state.hp > 0), axis=1)
    if loose:
        # The brute force: both sides' draws whenever a Quick Claw is anywhere in the battle
        # (either party, any HP), holder or not. With no Quick Claw at all nothing can read
        # speedRand (battle_lib.c:1278 tests the item first), and drawing it there anyway
        # would only multiply every other brute force past the width cap.
        quick_claw = jnp.broadcast_to(live & jnp.any(claw & (state.species != 0)), (2,))

    def branch_relevant(m: dict) -> Outcome:
        return Outcome(
            speed_tie=speed_tie,
            hit=hit,
            crit=crit,
            roll=m["roll"],
            secondary=m["secondary"],
            contact_proc=contact_proc,
            n_hits=uses_move & drawn_hits,
            full_para=m["full_para"],
            confusion_hit=m["confusion_hit"],
            duration=duration,
            thaw=m["thaw"],
            flinch_item=m["flinch_item"],
            quick_claw=quick_claw,
            end_speed_tie=end_speed_tie,
            magnitude=uses_move & (effect == _E.MAGNITUDE),
            shed_skin=shed_skin,
            fang_flinch=m["fang_flinch"],
            present=uses_move & (effect == _E.PRESENT),
            protect=m["protect"],
            infatuation=m["infatuation"],
            tri_attack=m["tri_attack"],
            roar_level=roar_level,
            roar_target=roar_target,
        )

    both = zip(branch_relevant(branches[0]), branch_relevant(branches[1]), strict=True)
    per_branch = Outcome(
        *(jnp.stack([jnp.asarray(a, bool), jnp.asarray(b, bool)]) for a, b in both)
    )
    relevant = Outcome(*(jnp.any(f, axis=0) for f in per_branch))

    # The first mover's sure flinch (rules §10.3, "no move after a flinch"): its move
    # reaches the hit whatever the rolls (none of the checks of
    # BattleControllerPlayer_CheckStatusDisruption, bcp.c:2417-2728, can stop it: no
    # sleep, freeze, paralysis, confusion, flinch, Disable on it, Taunt on a power-0 move,
    # infatuation or Bide; the first mover cannot fall in love before its own check),
    # deals damage at every crit and roll (play_hit's ``effective``: the flinch needs it),
    # and the target, the foe's active Pokemon using a move, has no Inner
    # Focus as the attacker sees it (effects._flinch, effects._fang). Fake Out also needs its
    # first turn (effect_script_0158.s). Thunder Fang's roll is fang_flinch (``Reads.fang``).
    clean = ~_any_of(status, (Status.SLEEP, Status.FREEZE, Status.PARALYSIS))
    reach = (
        uses_move
        & (state.hp[sides, mons] > 0)
        & clean
        & (state.confusion == 0)
        & (state.flinched == 0)
        & ~((state.disabled_move != 0) & (state.disabled_move.astype(_I32) == chosen))
        & ~((state.taunt > 0) & (tables.move_power[chosen] == 0))
        & (state.infatuated == 0)
        & (state.bide_turns == 0)
    )
    damage = jnp.stack([hit_damage_table(tables, state, side, moves[side]) for side in (0, 1)])
    deals = jnp.all(damage > 0, axis=(1, 2))
    fake_out_ok = (effect == _E.FAKE_OUT) & (
        state.fake_out_turn.astype(_I32) == state.turn.astype(_I32)
    )
    fang = effect == _E.FANG_PARALYZE_FLINCH
    flinch_sure = (
        ((effect == _E.FLINCH) | fake_out_ok | fang)
        & reach
        & deals
        & (ignorable != AbilityId.INNER_FOCUS)
        & foe_moves
        & (chance != 0)
        & single
    )
    proc_effects = (
        _E.SECONDARY_STATUS,
        _E.SECONDARY_CONFUSE,
        _E.TRAP,
        _E.FANG_PARALYZE_FLINCH,
        _E.TRI_ATTACK,
    )
    off = jnp.zeros((2,), bool)
    reads = Reads(
        first=firsts,
        hit_gated=off if loose else uses_move & (effect != _E.PURSUIT),
        proc_gated=off if loose else uses_move & (chance != 0) & _any_of(effect, proc_effects),
        flinch_sure=off if loose else flinch_sure,
        fang=fang,
        present=off if loose else uses_move & (effect == _E.PRESENT),
        tri_attack=off if loose else uses_move & (effect == _E.TRI_ATTACK) & (chance != 0),
        protect_sure=off if loose else reach & (effect == _E.PROTECT),
        protect_blocks=off if loose else uses_move & protect_blocks(tables, moves, False),
        roar=off if loose else roar,
        confused_own=base["confused_own"],
        confused_by_foe=jnp.stack([m["confused_by_foe"] for m in branches]),
        para_own=base["para_own"],
        para_by_foe=jnp.stack([m["para_by_foe"] for m in branches]),
        freeze_own=base["freeze_own"],
        freeze_by_foe=jnp.stack([m["freeze_by_foe"] for m in branches]),
        infatuated_own=base["infatuated_own"],
        infatuated_by_foe=jnp.stack([m["infatuated_by_foe"] for m in branches]),
    )
    kind = jnp.where(
        effect == _E.TAUNT,
        DURATION_TAUNT,
        jnp.where(_any_of(effect, (_E.TRAP, _E.DISABLE)), DURATION_TRAP, DURATION_DEFAULT),
    )
    # Damage classes (rules §10.3 item 5): a hit whose damage the turn-start table gives
    # whatever happened before it (damage_inputs_can_change), of a move whose table is
    # its damage (TABLE_DAMAGE_EFFECTS, an allowlist), and whose roll is not also a
    # confusion self-hit's.
    tabled = rolled & _any_of(effect, TABLE_DAMAGE_EFFECTS)
    compress = jnp.stack(
        [
            crit
            & tabled[:, None]
            & ~damage_inputs_can_change(tables, state, actions, firsts[t] == 1 - sides)
            & ~(m["confusion_hit"][:, None] & (hit_index == 0))
            for t, m in enumerate(branches)
        ]
    )
    if loose:
        compress = jnp.zeros_like(compress)
    thresholds, possible = hit_thresholds(tables, state, actions)
    if loose:
        # The brute force: the hit law cut at every threshold variant, possible or not
        # (an extra cut only splits a class).
        possible = jnp.ones_like(possible)
    return OutcomeLaws(
        hit_thresholds=thresholds,
        hit_possible=possible,
        crit_odds=crit_odds(tables, state, actions),
        no_crit=state.lucky_chant[::-1] != 0,
        secondary_chance=chance,
        duration_kind=kind.astype(_I32),
        relevant=relevant,
        branch_relevant=per_branch,
        reads=Reads(*(jnp.asarray(f) for f in reads)),
        damage=damage.astype(_I32),
        compress=compress,
        protect_count=state.protect_count.astype(_I32),
        roar_level_weight=roar_weight,
        roar_bench=roar_bench,
        flinch_item_chance=kings_rock,
    )


# Host laws: ``(values, numerators, denominator)`` of one element, all integers, the
# same draws as ``sample_outcome`` (outcome.py:143-193).
_Law = tuple[list[int], list[int], int]


def _coin(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [1, 1], 2  # RandNext() & 1, battle_lib.c:1445 and :3364; % 2, :2621


def hit_class_law(thresholds) -> _Law:
    """The law of the accuracy roll ``u = RandNext() % 100 + 1`` (uniform on 1-100,
    battle_controller_player.c:2976) reduced to its class over ``thresholds`` (the
    possible ones; ``outcome.hit_class``): the cuts are the thresholds within 1-99, each
    class weighs its width, and its value is its smallest roll, or 0 when it misses at
    every threshold. Values in increasing order."""
    ts = sorted({min(max(int(t), 0), 100) for t in thresholds})
    cuts = [t for t in ts if 1 <= t <= 99]
    lows, highs = [1, *(c + 1 for c in cuts)], [*cuts, 100]
    law = sorted((0 if low > ts[-1] else low, high - low + 1) for low, high in zip(lows, highs))
    return [v for v, _ in law], [w for _, w in law], 100


def _hit(laws: OutcomeLaws, side: int) -> _Law:
    possible = np.asarray(laws.hit_possible[side], bool)
    return hit_class_law(np.asarray(laws.hit_thresholds[side])[possible])


def _crit(laws: OutcomeLaws, side: int) -> _Law:
    if bool(laws.no_crit[side]):
        return [0], [1], 1
    odds = int(laws.crit_odds[side])  # rand % odds == 0
    return [0, 1], [odds - 1, 1], odds


def _roll(laws: OutcomeLaws, side: int) -> _Law:
    return list(range(16)), [1] * 16, 16  # rand % 16


def _secondary(laws: OutcomeLaws, side: int) -> _Law:
    chance = int(laws.secondary_chance[side])
    if chance == 0:  # applies on every hit
        return [1], [1], 1
    chance = min(max(chance, 0), 100)  # rand % 100 < chance
    return [0, 1], [100 - chance, chance], 100


def _contact(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [7, 3], 10  # rand % 10 < 3


def _n_hits(laws: OutcomeLaws, side: int) -> _Law:
    return [2, 3, 4, 5], [3, 3, 1, 1], 8  # BtlCmd_SetMultiHit: 2-3 at 3/8, 4-5 at 1/8


def _full_para(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [3, 1], 4  # rand % 4 == 0


def _duration(laws: OutcomeLaws, side: int) -> _Law:
    kind = int(laws.duration_kind[side])
    if kind == DURATION_TAUNT:
        return [3, 4, 5], [1, 1, 1], 3
    low = 3 if kind == DURATION_TRAP else 2
    return list(range(low, low + 4)), [1] * 4, 4


def _thaw(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [4, 1], 5  # rand % 5 == 0


def _magnitude(laws: OutcomeLaws, side: int) -> _Law:
    # BtlCmd_CalcMagnitude (battle_script.c:5820-5852): rand % 100 -> level 4-10.
    return list(range(4, 11)), [5, 10, 20, 30, 20, 10, 5], 100


def _shed_skin(laws: OutcomeLaws, side: int) -> _Law:
    return [0, 1], [7, 3], 10  # RandNext() % 10 < 3, battle_lib.c:3567


def _present(laws: OutcomeLaws, side: int) -> _Law:
    # BtlCmd_Present (battle_script.c:5793-5803): RandNext() & 0xFF against 102, 178, 204.
    cuts = (0, *PRESENT_CUTS, 256)
    return list(range(PRESENT_HEAL + 1)), [b - a for a, b in itertools.pairwise(cuts)], 256


def _protect(laws: OutcomeLaws, side: int) -> _Law:
    # BtlCmd_TryProtection (battle_script.c:5167): success iff sProtectSuccessRate[n] >=
    # RandNext(), a u16: (rate + 1) of 65536 values, exactly 2^-n.
    rate = PROTECT_RATES[min(max(int(laws.protect_count[side]), 0), len(PROTECT_RATES) - 1)]
    if rate + 1 == 65536:
        return [1], [1], 1
    return [0, 1], [65536 - (rate + 1), rate + 1], 65536


def _tri_attack(laws: OutcomeLaws, side: int) -> _Law:
    # effect_script_0036.s Random 2, 3: RandNext() % 3, uniform by the % n convention (exact
    # 21846/21845/21845 of 65536).
    return [0, 1, 2], [1, 1, 1], 3


def _quick_claw(laws: OutcomeLaws, side: int) -> _Law:
    # speedRand % (100 / param) == 0 (battle_lib.c:1278-1286), uniform by the % n convention
    # (exact 13108/65536 for % 5).
    n = QUICK_CLAW_MODULUS
    return [0, 1], [n - 1, 1], n


def _roar_level(laws: OutcomeLaws, side: int) -> _Law:
    # BattleSystem_CanWhirlwind (battle_lib.c:3072-3086): RandNext() & 0xFF, exact; 1 = passes
    # (outcome.roar_level_passes).
    k = int(laws.roar_level_weight[side])
    return [0, 1], [ROAR_LEVEL_VALUES - k, k], ROAR_LEVEL_VALUES


def _roar_target(laws: OutcomeLaws, side: int) -> _Law:
    # BtlCmd_TryWhirlwind (battle_script.c:5303-5311): the rejection loop over
    # RandNext() % partyCount, uniform over the eligible bench by the % n convention (exact:
    # slot i weighs #{u16 r : r % partyCount == i}); the value is the pick's rank.
    n = max(int(laws.roar_bench[side]), 1)
    return list(range(n)), [1] * n, n


def _flinch_item(laws: OutcomeLaws, side: int) -> _Law:
    # BattleControllerPlayer_CheckExtraFlinch (battle_controller_player.c:4639):
    # RandNext() % 100 < param, uniform by the % n convention.
    chance = min(max(int(laws.flinch_item_chance[side]), 0), 100)
    return [0, 1], [100 - chance, chance], 100


_LAWS = {
    "speed_tie": _coin,
    "hit": _hit,
    "crit": _crit,
    "roll": _roll,
    "secondary": _secondary,
    "contact_proc": _contact,
    "n_hits": _n_hits,
    "full_para": _full_para,
    "confusion_hit": _coin,
    "duration": _duration,
    "thaw": _thaw,
    "flinch_item": _flinch_item,
    "quick_claw": _quick_claw,
    "end_speed_tie": _coin,
    "magnitude": _magnitude,
    "shed_skin": _shed_skin,
    "fang_flinch": _secondary,  # the same CheckEffectActivation roll at the move's chance
    "present": _present,
    "protect": _protect,
    # CHECK_STATUS_STATE_ATTRACT (bcp.c:2669-2692): RandNext() & 1, exact; 1 = immobilized.
    "infatuation": _coin,
    "tri_attack": _tri_attack,
    "roar_level": _roar_level,
    "roar_target": _roar_target,
}


@cache
def _empty() -> Outcome:
    return Outcome(*(np.asarray(f) for f in empty_outcome()))


# --- Conditional reads and joint factors -------------------------------------------------

Elem = tuple[str, tuple[int, ...]]  # (Outcome field, index), e.g. ("crit", (0, 1))
SKIPPED = -1  # a dependency's value in a row that does not draw it


class Condition(NamedTuple):
    """``test`` gets one ``int`` array per element of ``deps`` (the rows' values, or
    ``SKIPPED`` where a row does not draw that element) and returns ``bool`` per row."""

    deps: tuple[Elem, ...]
    test: Callable[..., np.ndarray]


class Branch(NamedTuple):
    """One order branch: ``speed_tie`` = ``tie`` (0 when it is not relevant), so side
    ``first`` acts first; ``relevant`` is ``OutcomeLaws.branch_relevant[tie]``."""

    tie: int
    first: int
    relevant: Outcome


class ReadPlan:
    """When each relevant element of a branch is read, as conditions on other elements.

    An element is drawn in a row when all of its *gates* hold and at least one of its
    *paths* holds (a path holds when all of its conditions do; no path declared = read
    unconditionally). Elsewhere the row keeps its ``empty_outcome`` value. Soundness
    contract for builders: every gate is a necessary condition of every read of the
    element (when it fails, ``step`` reads the element nowhere: the stepped state and
    the turn's events do not depend on its value); a builder that declares a path for an
    element declares one for every reason it is read (``outcome_laws`` relevance). A
    condition must treat ``SKIPPED`` soundly: a dependency a row does not draw is one
    ``step`` does not read in that row."""

    def __init__(self, elements: list[Elem]):
        self.elements = elements
        self._gates: dict[Elem, list[Condition]] = {e: [] for e in elements}
        self._paths: dict[Elem, dict[str, list[Condition]]] = {e: {} for e in elements}

    def __contains__(self, elem: Elem) -> bool:
        return elem in self._gates

    def gate(self, elem: Elem, *conditions: Condition) -> None:
        """AND ``conditions`` into every read of ``elem`` (ignored if not relevant)."""
        if elem in self:
            self._gates[elem].extend(c for c in conditions if c is not None)

    def path(self, elem: Elem, name: str, *conditions: Condition) -> None:
        """Declare (or extend) read path ``name`` of ``elem``: read when ``conditions``
        all hold. Paths of one element are alternatives (OR)."""
        if elem in self:
            self._paths[elem].setdefault(name, []).extend(c for c in conditions if c is not None)

    def restrict(
        self, elem: Elem, name: str, *conditions: Condition, required: bool = True
    ) -> None:
        """AND ``conditions`` into read path ``name`` of ``elem`` (ignored if ``elem`` is
        not relevant).

        It depends on builder order: it changes only the path as declared when it is called,
        so the builder that declares the path must run first (``register_reads`` order), and a
        builder registered later that adds another path to ``elem`` must apply the
        restriction itself. With ``required`` (the default) a missing path raises instead of
        letting the read escape; ``required=False`` is for an element that may be read only
        through other paths (then it is not read through ``name`` at all)."""
        if elem not in self:
            return
        if name not in self._paths[elem]:
            if required:
                raise ValueError(f"{elem} has no {name!r} read path to restrict")
            return
        self._paths[elem][name].extend(c for c in conditions if c is not None)

    def deps(self, elem: Elem) -> set[Elem]:
        conds = [*self._gates[elem], *(c for p in self._paths[elem].values() for c in p)]
        return {d for c in conds for d in c.deps if d in self and d != elem}

    def drawn(self, elem: Elem, cols: dict[Elem, np.ndarray], n: int) -> np.ndarray:
        """``bool[n]``: the rows (with the values in ``cols``) that read ``elem``."""

        def holds(cond: Condition) -> np.ndarray:
            args = [cols.get(d, np.full(n, SKIPPED, np.int16)) for d in cond.deps]
            return np.broadcast_to(np.asarray(cond.test(*args), bool), (n,))

        ok = np.ones(n, bool)
        for cond in self._gates[elem]:
            ok &= holds(cond)
        paths = self._paths[elem]
        if paths:
            any_path = np.zeros(n, bool)
            for conds in paths.values():
                this = np.ones(n, bool)
                for cond in conds:
                    this &= holds(cond)
                any_path |= this
            ok &= any_path
        return ok


ReadBuilder = Callable[[OutcomeLaws, Branch, ReadPlan], None]
_READ_BUILDERS: list[ReadBuilder] = []


def register_reads(builder: ReadBuilder) -> ReadBuilder:
    """Register a joint factor: ``builder(laws, branch, plan)`` adds gates and paths
    to ``plan`` (``ReadPlan``) for one order branch. Builders run in registration order,
    ``_move_reads`` first; ``laws`` is one numpy row of ``outcome_laws``."""
    _READ_BUILDERS.append(builder)
    return builder


def _any_proc(*values: np.ndarray) -> np.ndarray:
    assert values, "a proc condition needs at least one secondary roll"
    out = np.zeros(np.shape(values[0]), bool)
    for v in values:
        out = out | (v == 1)
    return out


def _never() -> np.ndarray:
    return np.False_


# Explicitly never read: for a read whose only trigger cannot happen in this branch.
NEVER = Condition((), _never)


def _hits_somewhere(v: np.ndarray) -> np.ndarray:
    # Class 0 misses at every threshold the move can have at its check (outcome.hit_class),
    # so the move certainly misses. SKIPPED (not read): no accuracy roll, or no move.
    return v != 0


@register_reads
def _move_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The reads of rules §10.3 items 1-3 on the implemented mechanics:

    * a move's own reads (its crits, rolls, secondary rolls, the foe's contact rolls on
      its hits, its hit count, duration, Magnitude level, Thunder Fang's flinch roll,
      Present's draw and Roar's level check and pick) need its accuracy roll to hit: every one sits behind ``play_hit``'s
      ``landed`` (``connected``, step._use_move); Pursuit's pre-switch hit reads them with
      no roll, so it is not gated (``Reads.hit_gated``). A confusion self-hit's
      ``roll[s, 0]`` is another path;
    * (secondary, duration): a ``SECONDARY_CONFUSE`` / ``SECONDARY_STATUS`` / ``TRAP``
      duration is read only under ``proc`` (effects._secondary_confuse, _inflict, _trap);
    * (secondary, the foe's thaw / full_para / confusion_hit and self-hit roll): a read
      through the first mover's infliction needs that move to hit and, for a secondary
      effect, to proc;
    * the first mover's sure flinch (``Reads.flinch_sure``) with a certain hit (a class
      that hits at every possible threshold) and a proc (its ``secondary``, or
      ``fang_flinch`` for Thunder Fang, ``Reads.fang``) stops the second mover's move:
      ``step._use_move`` checks the flinch before Disable, Taunt, confusion, paralysis,
      attract and the move, so none of those elements is read (the thaw roll comes before
      it)."""
    reads, rel = laws.reads, branch.relevant
    first, second = branch.first, 1 - branch.first

    def rel_elem(name: str, index: tuple[int, ...]) -> Elem | None:
        return (name, index) if bool(np.asarray(getattr(rel, name))[index]) else None

    def can_hit(s: int) -> Condition | None:
        h = rel_elem("hit", (s, 0))
        return None if h is None else Condition((h,), _hits_somewhere)

    def procs(s: int) -> Condition:
        """Some hit of ``s``'s move procs. A proc-gated move whose secondary roll is not
        relevant in this branch (``outcome_laws`` drops it only when the effect is
        blocked: the target cannot take the status or is confused) never applies its
        effect, so what needs the proc is never read: ``NEVER``, stated explicitly."""
        secs = tuple(e for i in range(MAX_HITS) if (e := rel_elem("secondary", (s, i))))
        return Condition(secs, _any_proc) if secs else NEVER

    def move_conds(s: int) -> list[Condition]:
        return [can_hit(s)] if bool(reads.hit_gated[s]) else []

    def inflicted(s: int) -> list[Condition]:
        """``s`` (moving first) inflicts its status or confusion on the foe."""
        conds = move_conds(s)
        return conds + [procs(s)] if bool(reads.proc_gated[s]) else conds

    t = branch.tie
    for s in (0, 1):
        foe = 1 - s
        hits = move_conds(s)
        for i in range(MAX_HITS):
            for elem in (
                ("crit", (s, i)),
                ("secondary", (s, i)),
                ("contact_proc", (foe, i)),
                ("flinch_item", (s, i)),
            ):
                plan.path(elem, "move", *hits)
            if i > 0 or bool(rel.crit[s, 0]):  # roll[s, 0]: damage, or a self-hit below
                plan.path(("roll", (s, i)), "move", *hits)
        for name in (
            "n_hits",
            "magnitude",
            "fang_flinch",
            "present",
            "protect",
            "tri_attack",
            "roar_level",
            "roar_target",
        ):
            plan.path((name, (s,)), "move", *hits)
        proc = [procs(s)] if bool(reads.proc_gated[s]) else []
        plan.path(("duration", (s,)), "move", *hits, *proc)
        for name, own, by_foe, path in (
            ("confusion_hit", reads.confused_own, reads.confused_by_foe, "confusion"),
            ("roll", reads.confused_own, reads.confused_by_foe, "confusion"),
            ("full_para", reads.para_own, reads.para_by_foe, "status"),
            ("thaw", reads.freeze_own, reads.freeze_by_foe, "status"),
            ("infatuation", reads.infatuated_own, reads.infatuated_by_foe, "status"),
        ):
            elem = (name, (s, 0)) if name == "roll" else (name, (s,))
            if bool(own[s]):
                plan.path(elem, path)
            if bool(by_foe[t, s]):
                plan.path(elem, path + "_inflicted", *inflicted(foe))

    # The first mover's flinch roll: Thunder Fang's own element, else its secondary.
    if bool(reads.fang[first]):
        secondary = rel_elem("fang_flinch", (first,))
    else:
        secondary = rel_elem("secondary", (first, 0))
    if bool(reads.flinch_sure[first]) and secondary is not None:
        h = rel_elem("hit", (first, 0))
        possible = np.asarray(laws.hit_possible[first], bool)
        sure = int(np.clip(np.asarray(laws.hit_thresholds[first])[possible], 0, 100).min())

        def not_flinched(*values: np.ndarray) -> np.ndarray:
            sec = values[-1]
            certain = (
                np.ones(np.shape(sec), bool)
                if h is None
                else ((values[0] > 0) & (values[0] <= sure))
            )
            return ~(certain & (sec == 1))

        cond = Condition(((h,) if h else ()) + (secondary,), not_flinched)
        q = second
        for i in range(MAX_HITS):
            for elem in (
                ("hit", (q, i)),
                ("crit", (q, i)),
                ("roll", (q, i)),
                ("secondary", (q, i)),
                ("contact_proc", (first, i)),
                ("flinch_item", (q, i)),
            ):
                plan.gate(elem, cond)
        for name in (
            "n_hits",
            "duration",
            "magnitude",
            "confusion_hit",
            "full_para",
            "infatuation",
            "fang_flinch",
            "present",
            "protect",
            "tri_attack",
            "roar_level",
            "roar_target",
        ):
            plan.gate((name, (q,)), cond)


def _not_heal(v: np.ndarray) -> np.ndarray:
    # SKIPPED (not drawn: the move did not reach its script) is not the heal: the damage
    # read keeps its other conditions.
    return v != PRESENT_HEAL


@register_reads
def _present_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (present, crit, roll) joint factor (rules §3(b), §10.3 item 3):
    effect_script_0122.s's heal path jumps past ``CalcCrit`` and ``CalcDamage``, so the
    move's own read of ``crit[s, 0]`` and ``roll[s, 0]`` (its "move" path) also needs
    ``present[s] != PRESENT_HEAL``: 3 * 32 + 1 rows instead of 4 * 32. A confusion
    self-hit's read of ``roll[s, 0]`` is another path and stays. Present's own crit and
    roll are never damage classes (``NON_TABLE_DAMAGE_EFFECTS``)."""
    for s in (0, 1):
        elem = ("present", (s,))
        if not bool(laws.reads.present[s]) or elem not in plan:
            continue
        damage = Condition((elem,), _not_heal)
        for name in ("crit", "roll"):
            plan.path((name, (s, 0)), "move", damage)


def _not_protected(v: np.ndarray) -> np.ndarray:
    # 1: the roll passed, so the Protect succeeded (Reads.protect_sure). SKIPPED (not
    # drawn) is not a success: the foe's reads keep their other conditions.
    return v != 1


@register_reads
def _protect_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (protect, the foe's move) joint factor (rules §2(b)): the first mover's Protect
    that certainly reaches ``BtlCmd_TryProtection`` (``Reads.protect_sure``: the checks of
    ``CheckStatusDisruption`` cannot stop it, as for the sure flinch) succeeds when its
    roll passes (always at ``protect_count`` 0), and then ``CheckMoveHitOverrides`` stops the
    second mover's blockable move (``Reads.protect_blocks``; step._use_move: it does not
    connect, after its accuracy roll, whose class is then not read either: the "protected"
    message replaces the miss). So the move's own reads (its "move" path: hit, crits, rolls,
    secondary rolls, the protecting side's contact rolls on its hits, its hit count,
    duration, Magnitude level, fang flinch, Present's draw, Roar's level check and pick) also
    need the Protect to fail.
    The second mover's status checks (sleep, thaw, confusion and its self-hit roll, full
    paralysis) come before the move and are not gated."""
    first, q = branch.first, 1 - branch.first
    reads = laws.reads
    if not (bool(reads.protect_sure[first]) and bool(reads.protect_blocks[q])):
        return
    elem = ("protect", (first,))
    cond = Condition((elem,), _not_protected) if elem in plan else NEVER
    # The accuracy roll is read for the move alone (step._use_move: the hit and the
    # "missed" event), so this is its only path.
    plan.path(("hit", (q, 0)), "move", cond)
    # _move_reads (registered first) declares a "move" path for each of these (restrict
    # raises otherwise), except roll[q, 0] when the move has no crit: then it is read only by
    # a confusion self-hit, which comes before the Protect matters.
    for i in range(MAX_HITS):
        for e in (
            ("crit", (q, i)),
            ("roll", (q, i)),
            ("secondary", (q, i)),
            ("contact_proc", (first, i)),
            ("flinch_item", (q, i)),
        ):
            optional = e == ("roll", (q, 0)) and ("crit", (q, 0)) not in plan
            plan.restrict(e, "move", cond, required=not optional)
    for name in (
        "n_hits",
        "duration",
        "magnitude",
        "fang_flinch",
        "present",
        "tri_attack",
        "roar_level",
        "roar_target",
    ):
        plan.restrict((name, (q,)), "move", cond)


def _proc(v: np.ndarray) -> np.ndarray:
    # 1: the secondary roll passed. SKIPPED (not drawn: the hit did not land) is no proc.
    return v == 1


def _tri_value(status: int) -> Callable[[np.ndarray], np.ndarray]:
    value = TRI_ATTACK_STATUSES.index(status)

    def test(v: np.ndarray) -> np.ndarray:
        # SKIPPED (not drawn: no proc) inflicts nothing.
        return v == value

    return test


@register_reads
def _tri_attack_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (tri_attack, secondary) joint factor (rules 8(b)): effect_script_0036.s draws the
    status (``Random 2, 3``) on every use, but ``effects._tri_attack`` reads it only when the
    secondary roll procs (``BattleSystem_TriggerSecondaryEffect``, battle_lib.c:1570-1586), so
    ``tri_attack[s]``'s "move" path also needs ``secondary[s, 0] == 1``: 1 + 3 rows instead of
    2 * 3. When the first mover runs Tri Attack, the second mover's thaw / full-paralysis read
    through that infliction (``Reads.freeze_by_foe`` / ``para_by_foe``, the
    "status_inflicted" path) also needs the value freeze / paralysis. A chance-0 Tri Attack
    (``Reads.tri_attack`` off) would read it on every landed hit: no restriction then."""
    for s in (0, 1):
        elem = ("tri_attack", (s,))
        if not bool(laws.reads.tri_attack[s]) or elem not in plan:
            continue
        sec = ("secondary", (s, 0))
        plan.restrict(elem, "move", Condition((sec,), _proc) if sec in plan else NEVER)
        if branch.first != s:
            continue
        for name, status in (("thaw", Status.FREEZE), ("full_para", Status.PARALYSIS)):
            cond = Condition((elem,), _tri_value(status))
            plan.restrict((name, (1 - s,)), "status_inflicted", cond, required=False)


def _passes(v: np.ndarray) -> np.ndarray:
    # 1: the level check passed. SKIPPED (not drawn: no check) fails nothing, but the builder
    # adds this condition only where roar_level is drawn whenever the pick is reached.
    return v == 1


@register_reads
def _roar_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (roar_level, roar_target) joint factor (rules §6(b)): ``BtlCmd_TryWhirlwind``
    (battle_script.c:5298-5311) runs its rejection loop only after
    ``BattleSystem_CanWhirlwind`` passed, so where the level check is drawn
    (``roar_level[s]`` relevant: a lower-levelled user), ``roar_target[s]``'s "move" path also
    needs ``roar_level[s] == 1``. Both carry the same move conditions (``_move_reads``,
    ``_protect_reads``), so the level check is drawn in every row that reaches the pick."""
    for s in (0, 1):
        level, target = ("roar_level", (s,)), ("roar_target", (s,))
        if not bool(laws.reads.roar[s]) or target not in plan or level not in plan:
            continue
        plan.restrict(target, "move", Condition((level,), _passes))


def damage_class_law(
    laws: OutcomeLaws, side: int
) -> tuple[list[int], list[int], int, list[int], list[int]]:
    """``(values, weights, denominator, crits, rolls)``: the joint law of one hit's
    ``(crit, roll)`` reduced to its damage classes. Two pairs share a class when they have
    the same crit flag (a landed crit is observed, ``step._hit_events``) and the same
    damage in ``laws.damage[side]``; a class weighs the sum of its pairs' weights over
    ``crit odds * ROLLS``, and ``crits[v]``, ``rolls[v]`` are class ``v``'s
    representative, its lowest roll. Values are class indices in crit-then-roll order."""
    table = np.asarray(laws.damage[side])
    crit_values, crit_weights, crit_den = _crit(laws, side)
    classes: dict[tuple[int, int], int] = {}
    weights: list[int] = []
    crits: list[int] = []
    rolls: list[int] = []
    for c, wc in zip(crit_values, crit_weights):
        if wc == 0:
            continue
        for r in range(ROLLS):
            key = (c, int(table[c, r]))
            if key not in classes:
                classes[key] = len(weights)
                weights.append(0)
                crits.append(c)
                rolls.append(r)
            weights[classes[key]] += wc
    return list(range(len(weights))), weights, crit_den * ROLLS, crits, rolls


class _DamageClasses(NamedTuple):
    roll: Elem  # the hit's roll element; its crit element carries the class
    law: _Law
    crits: np.ndarray  # int16[classes]: the representative's crit
    rolls: np.ndarray  # int16[classes]: the representative's roll


def _damage_classes(
    laws: OutcomeLaws, branch: Branch, plan: ReadPlan
) -> dict[Elem, _DamageClasses]:
    """The hits of this branch drawn as damage classes, keyed by their crit element:
    ``laws.compress`` allows it, and the plan reads the hit's crit and roll together
    (the same gates and paths) and no other element's read depends on either. Only that
    direction is checked: crit / roll gated on another element are still compressed, so a
    mechanic that changes the hit's damage must be kept out of ``TABLE_DAMAGE_EFFECTS``."""
    depended = set().union(*(plan.deps(e) for e in plan.elements)) if plan.elements else set()
    compress = np.asarray(laws.compress[branch.tie], bool)
    out: dict[Elem, _DamageClasses] = {}
    for side, hit in np.argwhere(compress):
        index = (int(side), int(hit))
        crit, roll = ("crit", index), ("roll", index)
        if crit not in plan or roll not in plan or {crit, roll} & depended:
            continue
        if plan._gates[crit] != plan._gates[roll] or plan._paths[crit] != plan._paths[roll]:
            continue
        values, weights, den, crits, rolls = damage_class_law(laws, index[0])
        out[crit] = _DamageClasses(
            roll, (values, weights, den), np.asarray(crits, np.int16), np.asarray(rolls, np.int16)
        )
    return out


def _branch_elements(relevant: Outcome) -> list[Elem]:
    out = []
    for name in Outcome._fields:
        if name == "speed_tie":
            continue
        mask = np.asarray(getattr(relevant, name), bool)
        for index in np.argwhere(mask) if mask.ndim else [()] * int(mask):
            out.append((name, tuple(int(i) for i in index)))
    return out


def _law(laws: OutcomeLaws, elem: Elem) -> _Law:
    name, index = elem
    values, weights, den = _LAWS[name](laws, index[0] if index else 0)
    kept = [(v, w) for v, w in zip(values, weights) if w > 0]
    return [v for v, _ in kept], [w for _, w in kept], den


def _topological(plan: ReadPlan) -> list[Elem]:
    """The plan's elements with every dependency first, else in ``Outcome`` order."""
    deps = {e: plan.deps(e) for e in plan.elements}
    done: set[Elem] = set()
    out: list[Elem] = []
    while len(out) < len(plan.elements):
        nxt = next((e for e in plan.elements if e not in done and deps[e] <= done), None)
        if nxt is None:
            raise ValueError("cyclic read conditions")
        done.add(nxt)
        out.append(nxt)
    return out


def _branch_rows(laws: OutcomeLaws, branch: Branch, budget: int, tie_weight: tuple[int, int]):
    """The rows of one branch: ``(cols, probs)``, ``cols[elem]`` the drawn values
    (``SKIPPED`` where a row does not read the element). ``tie_weight`` is the branch's
    ``(numerator, denominator)``. Raises ``SamplingOnly`` above ``budget`` rows.

    A compressed hit (``_damage_classes``) is drawn once, as its class on the crit
    element, and both elements get the class representative afterwards."""
    plan = ReadPlan(_branch_elements(branch.relevant))
    for builder in _READ_BUILDERS:
        builder(laws, branch, plan)
    classes = _damage_classes(laws, branch, plan)
    merged = {c.roll for c in classes.values()}
    order = [e for e in _topological(plan) if e not in merged]
    laws_of = {e: classes[e].law if e in classes else _law(laws, e) for e in order}
    den = tie_weight[1] * math.prod(law[2] for law in laws_of.values())
    exact = den <= 2**53
    n = 1
    cols: dict[Elem, np.ndarray] = {}
    num = np.full(1, tie_weight[0], np.int64)
    probs = np.full(1, tie_weight[0] / tie_weight[1], np.float64)
    for elem in order:
        values, weights, d = laws_of[elem]
        drawn = plan.drawn(elem, cols, n)
        sizes = np.where(drawn, len(values), 1)
        total = int(sizes.sum())
        if total > budget:
            raise SamplingOnly(
                f"the outcome combinations exceed the width cap {WIDTH_CAP} "
                f"(this branch alone reaches {total} rows with {budget} left; "
                f"{len(order) + 1} random elements)"
            )
        src = np.repeat(np.arange(n), sizes)
        pos = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        read = drawn[src]
        cols = {e: c[src] for e, c in cols.items()}
        cols[elem] = np.where(read, np.asarray(values, np.int16)[np.minimum(pos, len(values) - 1)],
                              SKIPPED).astype(np.int16)  # fmt: skip
        w = np.asarray(weights, np.int64)[np.minimum(pos, len(values) - 1)]
        if exact:
            num = num[src] * np.where(read, w, d)
        else:
            probs = probs[src] * np.where(read, w / d, 1.0)
        n = total
    if exact:
        probs = num.astype(np.float64) / float(den)
    for crit, c in classes.items():
        value = cols[crit]
        read = value != SKIPPED
        index = np.where(read, value, 0)
        cols[c.roll] = np.where(read, c.rolls[index], SKIPPED).astype(np.int16)
        cols[crit] = np.where(read, c.crits[index], SKIPPED).astype(np.int16)
    return cols, probs


def enumerate_drawn(laws: OutcomeLaws) -> tuple[Outcome, np.ndarray, Outcome]:
    """``enumerate_outcomes`` with ``drawn``: an ``Outcome`` of ``bool`` arrays shaped
    like the outcomes, true where the row reads (and enumerated) the element."""
    laws = jax.tree.map(np.asarray, laws)
    reads = laws.reads
    tied = bool(laws.relevant.speed_tie)
    branches = [
        Branch(t, int(reads.first[t]), Outcome(*(f[t] for f in laws.branch_relevant)))
        for t in ((0, 1) if tied else (0,))
    ]
    parts = []
    for branch in branches:
        budget = WIDTH_CAP - sum(len(p[1]) for p in parts)
        parts.append(_branch_rows(laws, branch, budget, (1, 2) if tied else (1, 1)))
    k = sum(len(p[1]) for p in parts)
    empty = _empty()
    fields = {name: np.repeat(f[None], k, axis=0) for name, f in zip(Outcome._fields, empty)}
    drawn = {name: np.zeros((k, *np.shape(f)), bool) for name, f in zip(Outcome._fields, empty)}
    if tied:
        fields["speed_tie"] = np.concatenate(
            [np.full(len(p[1]), b.tie, np.int8) for b, p in zip(branches, parts)]
        )
        drawn["speed_tie"][:] = True
    start = 0
    for cols, probs in parts:
        rows_ = slice(start, start + len(probs))
        for (name, index), col in cols.items():
            read = col != SKIPPED
            target = fields[name][(rows_, *index)]
            fields[name][(rows_, *index)] = np.where(read, col, target).astype(np.int8)
            drawn[name][(rows_, *index)] = read
        start += len(probs)
    probs = np.concatenate([p[1] for p in parts])
    return Outcome(**fields), probs, Outcome(**drawn)


def enumerate_outcomes(laws: OutcomeLaws) -> tuple[Outcome, np.ndarray]:
    """Every combination of the values the turn can read: an ``Outcome`` of numpy
    ``int8`` arrays with a leading axis ``K`` and ``float64[K]`` probabilities summing to
    1. ``laws`` is one (unbatched) row of ``outcome_laws``.

    ``speed_tie`` is enumerated first: each order branch (``Branch``) gets its own masks
    (``OutcomeLaws.branch_relevant``) and conditional reads (``ReadPlan``, built by the
    ``register_reads`` builders), and the branches' rows are concatenated. Within a
    branch the elements are drawn in dependency order; a row draws an element only where
    its read conditions hold, and keeps its ``empty_outcome`` value elsewhere (the
    element's whole probability goes to that one row). With no conditions this is the
    product of the relevant elements' supports, first element slowest. Each probability
    is the product of the drawn elements' integer weights over the product of all the
    branch's denominators, rounded once when that is below 2**53 (else a float64
    product). Raises ``SamplingOnly`` when ``K`` exceeds ``WIDTH_CAP``."""
    outcomes, probs, _ = enumerate_drawn(laws)
    return outcomes, probs
