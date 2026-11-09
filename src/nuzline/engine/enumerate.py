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
  also when a confusion self-hit is possible (step.py:296). The Pursuit pre-switch hit
  reads ``[pursuer, 0]`` (step.py:464-474), a move action too.
* ``secondary[s, i]``: as ``crit``, for a move with chance != 0 whose effect reads
  ``proc`` (``_PROC_EFFECTS``: the secondary stat effects, AncientPower's, secondary
  status, flinch, Fake Out, Bind, secondary confusion). Not for a flinch move (``FLINCH``,
  ``FAKE_OUT``) of the side moving second (``subscript_flinch_mon.s``: ``IfMovedThisTurn``
  skips it; effects._flinch), nor, moving first with one hit, for a secondary status the
  target cannot take (``effects.can_have_status``, or Limber against paralysis; an
  Insomnia target is never skipped, its reveal runs on the proc) or a secondary
  confusion on a confused target (``subscript_confuse.s``, effects._secondary_confuse).
* ``contact_proc[d, i]``: the attacker's damaging contact move against a Static,
  Poison Point or Flame Body holder on side ``d`` (Rough Skin has no roll): its active
  Pokemon (Pursuit hits the outgoing one) or the one it switches in (effects.py:343-365).
* ``n_hits[s]``: ``MULTI_HIT`` without a fixed count (``move_param[m, 0] == 0``).
* ``full_para[s]`` / ``thaw[s]``: ``s`` moves and is paralyzed / frozen at the start,
  or the foe moves first with a move that inflicts it and the target can take it
  (step.py:277-278, :305).
* ``confusion_hit[s]``: ``s`` moves and its confusion counter is >= 2, or the foe moves
  first with a ``CONFUSE_FOE`` or ``SECONDARY_CONFUSE`` move and ``s`` is not confused
  (counter 2-5; step.py:292-294).
* ``duration[s]``: ``s``'s move is Taunt, Bind, ``CONFUSE_FOE``, ``SECONDARY_CONFUSE`` or
  inflicts sleep (effects.py:189, :286, :297, :307).
* ``speed_tie``: the two orders ``order.action_order`` gives under ``speed_tie`` 0 and 1
  differ (it is read only there: ``player_second``'s ``RandNext() & 1``, battle_lib.c:1445).
* ``end_speed_tie``: the effective speeds after the actions can be equal
  (step.py:663-673): each side's possible end speeds cover every Speed stage when a
  stage change is possible this turn (a Speed stat effect, AncientPower, Curse, a
  flinched Steadfast), paralysis when it can be inflicted, and the
  neutral speed of a fainted battler (``_faint_check`` resets stages and status). A
  White Herb holder may lose a lowered stage and a Lum Berry holder or a Pluck / Bug Bite
  user its paralysis (over-approximate). Lagging Tail only changes which of
  ``speed_tie`` / ``end_speed_tie`` is read when speeds are equal, never more.
* ``magnitude[s]``: ``s`` uses a ``MAGNITUDE`` move (``BtlCmd_CalcMagnitudePower``).
  ``crit``/``roll`` are not relevant for ``LEVEL_DAMAGE`` (Night Shade deals its user's
  level with no crit or roll, ``damage.fixed_damage``) nor for ``BIDE`` (the unleash is
  ``2 * storedDamage``, also fixed). A locked side's move is its locked move
  (``order.selected_move``), so the masks above read it.
* ``shed_skin[d]``: side ``d``'s active or incoming Pokemon has Shed Skin
  (``step._turn_end_ability``; over-approximate: it only rolls with a status and HP).
* ``flinch_item``, ``quick_claw``: never read.
* Nothing is relevant once the battle is over (``step`` returns the state, step.py:700).

**Order branches and conditional reads** (rules §10.3 items 1-4). ``speed_tie`` is
enumerated first: each value is a branch (``Branch``) whose order is
``order.action_order``'s, with its own masks (the order-gated reads above), and the
branches' rows are concatenated. Within a branch, a ``ReadPlan`` says when each
element is read, as conditions on the values of other elements (a *joint factor* is a
set of elements linked by such conditions; it lists only the combinations that are
read). ``_move_reads`` gives the implemented mechanics': a move's own reads need its
accuracy roll to hit; a secondary duration needs the proc; a read through the first
mover's infliction needs its hit and proc; and the first mover's sure flinch stops the
second mover's move. Later mechanics add their own with ``register_reads``. The
brute-force reference is ``outcome_laws(..., loose=True)``: no order or eligibility
gate and no conditional read, so its enumeration is the independent product of
every element any order can read (tests/engine/test_enumerate.py compares the two).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine.damage import STATUS, damage_table
from nuzline.engine.effects import can_have_status
from nuzline.engine.ids import AbilityId, EffectId, ItemId, MoveFlag, Stat, Status
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    effective_speed,
    is_item,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome, crit_odds, empty_outcome, hit_thresholds
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
)


class Reads(NamedTuple):
    """Inputs of the conditional reads (``_move_reads``), per side (``bool[2]``) unless
    noted; ``[2, 2]`` is ``[speed_tie branch, side]``."""

    first: jnp.ndarray  # int32[2]: the side acting first under speed_tie = 0, 1
    hit_gated: jnp.ndarray  # its move's reads need its accuracy roll to hit (no Pursuit)
    proc_gated: jnp.ndarray  # its duration and status infliction need a secondary proc
    flinch_sure: jnp.ndarray  # moving first, a sure hit with a proc flinches the foe
    confused_own: jnp.ndarray  # it moves confused (counter >= 2) at the start
    confused_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that confuses
    para_own: jnp.ndarray  # it moves paralyzed at the start
    para_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that paralyzes
    freeze_own: jnp.ndarray  # it moves frozen at the start
    freeze_by_foe: jnp.ndarray  # bool[2, 2]: it moves after a foe move that freezes


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
    """The move inflicts ``status`` on the foe (``STATUS_FOE`` / ``SECONDARY_STATUS``
    with ``move_param[m, 0]``, effects.py:186-208)."""
    effect = tables.move_effect[moves]
    status_effect = (effect == _E.STATUS_FOE) | (effect == _E.SECONDARY_STATUS)
    return status_effect & (tables.move_param[moves, 0] == status)


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


def _end_speeds(tables: Tables, state: BattleState, actions, side, stage_free, para_free):
    """``(int32[27], bool[27])``: the effective speeds ``side`` can have after the
    actions (``end_of_turn_order`` on the post-action state, step.py:663-673) and which
    of them are possible. Grid: Speed stage -6..6 x paralyzed or not of the Pokemon
    active after the actions, plus its neutral speed (a faint resets stages and
    status, step.py:477-510). A stage other than the start one needs ``stage_free``,
    a paralysis state other than the start one ``para_free``."""
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

    def speed(stage, para):
        s = state._replace(
            active=state.active.at[side].set(mon),
            stages=state.stages.at[side, Stat.SPE].set(stage.astype(state.stages.dtype)),
            status=state.status.at[side, mon].set(
                jnp.where(para, Status.PARALYSIS, Status.NONE).astype(state.status.dtype)
            ),
        )
        return effective_speed(tables, s, side)

    speeds = jax.vmap(speed)(stages, paras)
    possible = (stage_free | (stages == stage0)) & (para_free | (paras == para0))
    fainted = jnp.arange(stages.shape[0]) == stages.shape[0] - 1
    return speeds, possible | fainted


def _branch_masks(foe_first, first_self, base: dict) -> dict:
    """Order-dependent masks of one branch (``outcome_laws``): ``foe_first`` /
    ``first_self`` (``bool[2]``) say whether each side's foe / the side itself acts
    first."""
    b = base
    para_by_foe = b["uses_move"] & foe_first & b["para_inflict"][::-1]
    freeze_by_foe = b["uses_move"] & foe_first & b["freeze_inflict"][::-1]
    confused_by_foe = b["uses_move"] & foe_first & b["confuse_inflict"][::-1]
    confusable = b["confused_own"] | confused_by_foe
    # A flinch roll is read only when the user moves first (subscript_flinch_mon.s: the
    # target has not moved, effects._flinch). A secondary status or confusion on a target
    # that cannot take it changes nothing (status eligibility), judged on the turn-start
    # state, which is the state of the hit when the user moves first with one hit.
    blocked = first_self & (
        ((b["effect"] == _E.SECONDARY_STATUS) & b["status_blocked"])
        | ((b["effect"] == _E.SECONDARY_CONFUSE) & b["confused_blocked"])
    )
    late_flinch = ~first_self & _any_of(b["effect"], (_E.FLINCH, _E.FAKE_OUT))
    return {
        "roll": b["crit"] | (confusable[:, None] & (b["hit_index"] == 0)),
        "secondary": b["secondary"] & ~(blocked | late_flinch)[:, None],
        "full_para": b["para_own"] | para_by_foe,
        "confusion_hit": confusable,
        "thaw": b["freeze_own"] | freeze_by_foe,
        "para_by_foe": para_by_foe,
        "freeze_by_foe": freeze_by_foe,
        "confused_by_foe": confused_by_foe,
    }


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
    moves = jnp.stack([selected_move(tables, state, side, actions[side])[0] for side in (0, 1)])
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
        [can_have_status(state, 1 - side, mons[1 - side], param0[side]) for side in (0, 1)]
    )
    limber = (param0 == Status.PARALYSIS) & (ignorable == AbilityId.LIMBER)
    insomnia = (param0 == Status.SLEEP) & (ignorable == AbilityId.INSOMNIA)
    single = ~multi
    status_blocked = single & ~insomnia & (~can_take | limber)
    confused_blocked = single & (state.confusion[::-1] > 0)
    if loose:
        status_blocked = confused_blocked = jnp.zeros((2,), bool)
    confuses = _any_of(effect, (_E.CONFUSE_FOE, _E.SECONDARY_CONFUSE))
    base = {
        "uses_move": uses_move,
        "effect": effect,
        "hit_index": hit_index,
        "crit": crit,
        "secondary": secondary,
        "status_blocked": status_blocked,
        "confused_blocked": confused_blocked,
        "para_own": uses_move & (status == Status.PARALYSIS),
        "freeze_own": uses_move & (status == Status.FREEZE),
        "confused_own": uses_move & (state.confusion >= 2),
        "para_inflict": uses_move & _inflicts(tables, moves, Status.PARALYSIS) & ~status_blocked,
        "freeze_inflict": uses_move & _inflicts(tables, moves, Status.FREEZE) & ~status_blocked,
        "confuse_inflict": uses_move & confuses & ~confused_blocked,
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
    # subscript_flinched.s:10-14: a flinched Steadfast Pokemon gains +1 Speed.
    steadfast = _holds(state, actions, (AbilityId.STEADFAST,)) & foe_moves
    steadfast = steadfast & _any_of(tables.move_effect[foe_move], (_E.FLINCH, _E.FAKE_OUT))
    # A White Herb restores a lowered Speed stage (hooks.held_item_check).
    herb = _holds_item(tables, state, actions, (ItemId.WHITE_HERB,))
    stage_free = (uses_move & own_speed) | (foe_moves & foe_speed[::-1]) | steadfast | herb
    static = _holds(state, actions, (AbilityId.STATIC,))
    # A Lum Berry cures paralysis, held or plucked (hooks.held_item_check, pluck_berry).
    lum = _holds_item(tables, state, actions, (ItemId.LUM_BERRY,))
    plucks = uses_move & (effect == _E.EAT_BERRY)
    para_free = (
        (foe_moves & _inflicts(tables, foe_move, Status.PARALYSIS))
        | (uses_move & damaging & contact & static[::-1])
        | lum
        | plucks
    )
    speeds = [
        _end_speeds(tables, state, actions, side, stage_free[side], para_free[side])
        for side in (0, 1)
    ]
    (speed0, ok0), (speed1, ok1) = speeds
    end_speed_tie = live & jnp.any(
        ok0[:, None] & ok1[None, :] & (speed0[:, None] == speed1[None, :])
    )

    never = jnp.zeros((2,), bool)

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
            flinch_item=never,
            quick_claw=never,
            end_speed_tie=end_speed_tie,
            magnitude=uses_move & (effect == _E.MAGNITUDE),
            shed_skin=live & _holds(state, actions, (AbilityId.SHED_SKIN,)),
        )

    both = zip(branch_relevant(branches[0]), branch_relevant(branches[1]), strict=True)
    per_branch = Outcome(
        *(jnp.stack([jnp.asarray(a, bool), jnp.asarray(b, bool)]) for a, b in both)
    )
    relevant = Outcome(*(jnp.any(f, axis=0) for f in per_branch))

    # The first mover's sure flinch (rules §10.3, "no move after a flinch"): its move
    # reaches the hit whatever the rolls (none of the checks of
    # BattleControllerPlayer_CheckStatusDisruption, bcp.c:2417-2728, can stop it: no
    # sleep, freeze, paralysis, confusion, flinch, Disable on it, Taunt on a power-0 move
    # or Bide), deals damage at every crit and roll (play_hit's ``effective``: the flinch
    # needs it), and the target, the foe's active Pokemon using a move, has no Inner
    # Focus as the attacker sees it (effects._flinch). Fake Out also needs its first turn
    # (effect_script_0158.s).
    clean = ~_any_of(status, (Status.SLEEP, Status.FREEZE, Status.PARALYSIS))
    reach = (
        uses_move
        & (state.hp[sides, mons] > 0)
        & clean
        & (state.confusion == 0)
        & (state.flinched == 0)
        & ~((state.disabled_move != 0) & (state.disabled_move.astype(_I32) == moves))
        & ~((state.taunt > 0) & (tables.move_power[moves] == 0))
        & (state.bide_turns == 0)
    )
    deals = jnp.stack(
        [
            jnp.all(damage_table(tables, state, side, moves[side], False) > 0)
            & jnp.all(damage_table(tables, state, side, moves[side], True) > 0)
            for side in (0, 1)
        ]
    )
    fake_out_ok = (effect == _E.FAKE_OUT) & (
        state.fake_out_turn.astype(_I32) == state.turn.astype(_I32)
    )
    flinch_sure = (
        ((effect == _E.FLINCH) | fake_out_ok)
        & reach
        & deals
        & (ignorable != AbilityId.INNER_FOCUS)
        & foe_moves
        & (chance != 0)
        & single
    )
    proc_effects = (_E.SECONDARY_STATUS, _E.SECONDARY_CONFUSE, _E.TRAP)
    off = jnp.zeros((2,), bool)
    reads = Reads(
        first=firsts,
        hit_gated=off if loose else uses_move & (effect != _E.PURSUIT),
        proc_gated=off if loose else uses_move & (chance != 0) & _any_of(effect, proc_effects),
        flinch_sure=off if loose else flinch_sure,
        confused_own=base["confused_own"],
        confused_by_foe=jnp.stack([m["confused_by_foe"] for m in branches]),
        para_own=base["para_own"],
        para_by_foe=jnp.stack([m["para_by_foe"] for m in branches]),
        freeze_own=base["freeze_own"],
        freeze_by_foe=jnp.stack([m["freeze_by_foe"] for m in branches]),
    )
    kind = jnp.where(
        effect == _E.TAUNT,
        DURATION_TAUNT,
        jnp.where(_any_of(effect, (_E.TRAP, _E.DISABLE)), DURATION_TRAP, DURATION_DEFAULT),
    )
    thresholds, possible = hit_thresholds(tables, state, actions)
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


def _zero(laws: OutcomeLaws, side: int) -> _Law:
    return [0], [1], 1


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
    "flinch_item": _zero,
    "quick_claw": _zero,
    "end_speed_tie": _coin,
    "magnitude": _magnitude,
    "shed_skin": _shed_skin,
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
    out = np.zeros(np.shape(values[0]) if values else (), bool)
    for v in values:
        out = out | (v == 1)
    return out


def _hits_somewhere(v: np.ndarray) -> np.ndarray:
    # Class 0 misses at every threshold the move can have at its check (outcome.hit_class),
    # so the move certainly misses. SKIPPED (not read): no accuracy roll, or no move.
    return v != 0


@register_reads
def _move_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The reads of rules §10.3 items 1-3 on the implemented mechanics:

    * a move's own reads (its crits, rolls, secondary rolls, the foe's contact rolls on
      its hits, its hit count, duration and Magnitude level) need its accuracy roll to
      hit: every one sits behind ``play_hit``'s ``landed`` (``connected``,
      step._use_move); Pursuit's pre-switch hit reads them with no roll, so it is not
      gated (``Reads.hit_gated``). A confusion self-hit's ``roll[s, 0]`` is another path;
    * (secondary, duration): a ``SECONDARY_CONFUSE`` / ``SECONDARY_STATUS`` / ``TRAP``
      duration is read only under ``proc`` (effects._secondary_confuse, _inflict, _trap);
    * (secondary, the foe's thaw / full_para / confusion_hit and self-hit roll): a read
      through the first mover's infliction needs that move to hit and, for a secondary
      effect, to proc;
    * the first mover's sure flinch (``Reads.flinch_sure``) with a certain hit (a class
      that hits at every possible threshold) and a proc stops the second mover's move:
      ``step._use_move`` checks the flinch before Disable, Taunt, confusion, paralysis
      and the move, so none of those elements is read (the thaw roll comes before it)."""
    reads, rel = laws.reads, branch.relevant
    first, second = branch.first, 1 - branch.first

    def rel_elem(name: str, index: tuple[int, ...]) -> Elem | None:
        return (name, index) if bool(np.asarray(getattr(rel, name))[index]) else None

    def can_hit(s: int) -> Condition | None:
        h = rel_elem("hit", (s, 0))
        return None if h is None else Condition((h,), _hits_somewhere)

    def procs(s: int) -> Condition:
        secs = tuple(e for i in range(MAX_HITS) if (e := rel_elem("secondary", (s, i))))
        return Condition(secs, _any_proc)

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
            for elem in (("crit", (s, i)), ("secondary", (s, i)), ("contact_proc", (foe, i))):
                plan.path(elem, "move", *hits)
            if i > 0 or bool(rel.crit[s, 0]):  # roll[s, 0]: damage, or a self-hit below
                plan.path(("roll", (s, i)), "move", *hits)
        for name in ("n_hits", "magnitude"):
            plan.path((name, (s,)), "move", *hits)
        proc = [procs(s)] if bool(reads.proc_gated[s]) else []
        plan.path(("duration", (s,)), "move", *hits, *proc)
        for name, own, by_foe, path in (
            ("confusion_hit", reads.confused_own, reads.confused_by_foe, "confusion"),
            ("roll", reads.confused_own, reads.confused_by_foe, "confusion"),
            ("full_para", reads.para_own, reads.para_by_foe, "status"),
            ("thaw", reads.freeze_own, reads.freeze_by_foe, "status"),
        ):
            elem = (name, (s, 0)) if name == "roll" else (name, (s,))
            if bool(own[s]):
                plan.path(elem, path)
            if bool(by_foe[t, s]):
                plan.path(elem, path + "_inflicted", *inflicted(foe))

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
            ):
                plan.gate(elem, cond)
        for name in ("n_hits", "duration", "magnitude", "confusion_hit", "full_para"):
            plan.gate((name, (q,)), cond)


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
    ``(numerator, denominator)``. Raises ``SamplingOnly`` above ``budget`` rows."""
    plan = ReadPlan(_branch_elements(branch.relevant))
    for builder in _READ_BUILDERS:
        builder(laws, branch, plan)
    order = _topological(plan)
    laws_of = {e: _law(laws, e) for e in order}
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
                f"more than {budget} outcome combinations exceed the width cap {WIDTH_CAP} "
                f"({len(order) + 1} random elements)"
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
