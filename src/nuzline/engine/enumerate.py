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
  (``step._use_move``). A multi-hit move checks accuracy on its first hit only (effects.py
  docstring), except Triple Kick: ``hit[s, 1]``, ``hit[s, 2]`` are its later hits' own checks
  (B13-6), read once the hit before can have landed, their classes over
  ``hit_thresholds(..., overrides=False)`` (``OutcomeLaws.hit_later_thresholds``). The values
  are the interval classes of the roll
  over ``outcome.hit_thresholds`` (``hit_class_law``): two classes for one threshold
  below 100 (0 and 1, as before), one more per further threshold an earlier action can
  produce, and the single value 1 when every threshold is 100 or more.
* ``crit[s, i]``, ``roll[s, i]``: a damaging move (status moves and power 0 deal 0,
  damage.py:254), index 0, or for ``MULTI_HIT`` 0-4 (the 2-5 law) or the indices below its
  fixed count (``move_param[m, 0]``, ``effects.hit_count``; 0-1 for a ``MoveFlag.HIT_TWICE``
  move, Twineedle, B13-1; 0-2 for ``TRIPLE_KICK``, below the eligible count for ``BEAT_UP``,
  B13-5, B13-9). ``roll[s, 0]``
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
* ``focus_band[s, i]`` (E1 Gardenia, ruling G-1): the defender's Focus Band roll on hit ``i``
  of ``s``'s non-status move (``effects.play_hit``, Pursuit's hit included) while the foe's
  active Pokemon or the one it switches in holds one (no implemented path gives one mid-turn);
  ``focus_band[s, 0]`` also when ``s`` can hurt itself in confusion (``confusion_hit`` relevant)
  holding one (``step._use_move``). Its "move" path is the move's; the self-hit's are
  ``roll[s, 0]``'s confusion paths. ``future_band[d]``: ``future_hit[d]``'s gate and a living
  holder in side ``d``'s party.
* ``chatter[s]`` (G-70): ``s``'s move is Chatter, its user is a Chatot that is not Transformed and its
  first hit lands: ``RandNext() % 100 <= chance`` (``chatter_chance``, 0 / 10 / 30 by its Chatot
  cry). The confusion counter ``duration[s]`` is then read only when it is 1
  (``_chatter_reads``), and the foe's self-hit reads through it need it too (``inflicted``).
* ``starf[d, 0]`` (F-10): the turn's Starf Berry draw, where side ``d`` can eat one this turn:
  its active Pokemon or the one it switches in holds one, or a holder of its party can enter
  (its own U-turn / Baton Pass, a foe's Roar), at its pinch or able to lose HP this turn; or ``d``
  Plucks / Bug Bites a foe holding one, or Recycles one; or a battle-start berry is still open
  (``starf_open``). Five values while the count of stats below +6 stays 5, else all 60
  (``_starf_reads``). ``starf[d, 1]``: a faint replacement's draw (five values), where a holder
  can enter at its pinch after the turn (``step.starf_kept`` keeps it).
* ``berry_confusion[s]`` (G-5): ``s`` moves and its confusion counter is ``CONFUSION_PENDING``
  at the start (a flavor berry's), or the foe moves first while ``s``'s active Pokemon, not
  confused, holds a flavor berry it dislikes (``hooks.held_item_check`` after the foe's hit):
  the draw of ``step._use_move``'s confusion check. A berry eaten after ``s``'s check leaves the
  counter pending for a later turn. The same foe-first case makes ``confusion_hit[s]`` and
  ``roll[s, 0]`` relevant (the "confusion_berry" path: the foe's move hits).
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

import collections
import itertools
import math
from collections.abc import Callable
from functools import cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import Tables
from nuzline.engine import targets
from nuzline.engine.actions import imprisoning_foe
from nuzline.engine.damage import (
    PHYSICAL,
    ROLLS,
    SPECIAL,
    STATUS,
    beat_up_damage,
    beat_up_slots,
    decomp_abilities,
    entry_ability,
    hit_damage_table,
    move_type_of,
    pinch_ability,
    weather_suppressed,
)
from nuzline.engine.effects import (
    TRIPLE_KICK_HITS,
    acupressure_valid,
    can_have_status,
    defender_side,
    ingrain_anchors,
    metronome_update,
    protect_blocks,
    psycho_shift_status,
)
from nuzline.engine.format import FORMATS, foes, format_of, party, side_of
from nuzline.engine.format import foe as _foe
from nuzline.engine.hooks import entry_damage_up, land_stats, starf_holders
from nuzline.engine.ids import (
    CALL_GRAVITY,
    CHATTER_CHANCES,
    CONFUSION_PENDING,
    FOCUS_BAND_PARAM,
    METRONOME_DRAW,
    MINIMIZE_PARAM,
    QUICK_CLAW_MODULUS,
    SLEEP_TALK_VALUES,
    SNORE_PARAM,
    SPECIES_CHATOT,
    SPECIES_SHAYMIN,
    STARF_DRAWS,
    STARF_STATS,
    STOMP_PARAM,
    TAILWIND_TURNS,
    TRACE_FIRES,
    TRI_ATTACK_STATUSES,
    UNROLLED_EFFECTS,
    WEATHER_ABILITY_IDS,
    AbilityId,
    EffectId,
    ItemId,
    MoveFlag,
    Stat,
    Status,
    Type,
    Weather,
    sleepless,
)
from nuzline.engine.order import (
    CALL_ASSIST,
    CALL_COPYCAT,
    CALL_ME_FIRST,
    CALL_METRONOME,
    CALL_NONE,
    CALL_SLEEP_TALK,
    MOVE_ACTIONS,
    action_order,
    assist_candidates,
    call_kind,
    called_move,
    copycat_call,
    disabled_state,
    dislikes_berry,
    effective_speed,
    encored_state,
    gravity_possible,
    gravity_state,
    invoked_move,
    is_item,
    is_switch,
    me_first_copy,
    metronome_weights,
    move_effects,
    sealed_state,
    selected_move,
    takeover_possible,
    truant_loafs,
    used_moves,
)
from nuzline.engine.outcome import (
    ACUPRESSURE_DRAWS,
    CRIT_DRAWS,
    MAX_HITS,
    PRESENT_CUTS,
    PRESENT_HEAL,
    PROTECT_RATES,
    ROAR_LEVEL_VALUES,
    Outcome,
    armored,
    crit_class,
    crit_variants,
    empty_outcome,
    future_thresholds,
    hit_threshold_pair,
    moved_mask,
    pivot_view,
    pursuit_armored,
    pursuit_chase,
    read_chance,
    roar_inputs,
    roar_levels,
    secondary_chance,
    self_lock,
)
from nuzline.engine.outcome import (
    kings_rock_chance as outcome_kings_rock_chance,
)
from nuzline.engine.state import STAGE_MAX, STAGE_MIN, BattleState, battler_ability, held_items

_I32 = jnp.int32
_E = EffectId

# Pre-merge outcome combinations per (state, action pair); above it the pair is
# sampling-only (plan ruling 3).
WIDTH_CAP = 131072

# ``OutcomeLaws.duration_kind``: the acting side's ``Random`` command (outcome.py:166-174).
DURATION_DEFAULT = 0  # Random 3, 2: 2-5 (sleep, confusion and every other move)
DURATION_TRAP = 1  # Random 3, 3: 3-6 (Bind, Uproar; Disable: RandNext() % 4 + 3, the same law)
DURATION_TAUNT = 2  # Random 2, 3: 3-5
DURATION_ENCORE = 3  # BtlCmd_TryEncore's RandNext() % 5 + 3: 3-7 (B10-9)

# Effects whose hits read ``outcome.secondary`` through ``proc`` (effects.py:468).
_PROC_EFFECTS = (
    _E.SECONDARY_STAT_SELF,
    _E.SECONDARY_STAT_FOE,
    _E.SECONDARY_STATUS,
    _E.RECOIL_STATUS,  # its status roll (effects._secondary_status, B10-1)
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
    """Inputs of the conditional reads (``_move_reads``), per side (``bool[B]``) unless
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
    self_lock: jnp.ndarray  # its lock move may be aimed at itself (outcome.self_lock)
    endure_sure: jnp.ndarray  # it uses Endure and reaches TryProtection whatever the rolls
    roar: jnp.ndarray  # it uses Roar: its slot pick needs the level check to pass
    confused_own: jnp.ndarray  # it moves confused (counter >= 2) at the start
    confused_by_foe: jnp.ndarray  # bool[2, B]: it moves after a foe move that confuses
    para_own: jnp.ndarray  # it moves paralyzed at the start
    para_by_foe: jnp.ndarray  # bool[2, B]: it moves after a foe move that paralyzes
    freeze_own: jnp.ndarray  # it moves frozen at the start
    freeze_by_foe: jnp.ndarray  # bool[2, B]: it moves after a foe move that freezes
    infatuated_own: jnp.ndarray  # it moves in love at the start (its source stays)
    infatuated_by_foe: jnp.ndarray  # bool[2, B]: it moves after a foe Attract that can work
    future_roll: jnp.ndarray  # its Future Sight can be set: TryFutureSight's roll[s, 0]
    crash: jnp.ndarray  # its move crashes on a miss (EffectId.CRASH): crit / roll[s, 0] then too
    # E1 Gardenia batch (G-1, G-5).
    band_move: jnp.ndarray  # bool[B, MAX_HITS]: the foe's Focus Band rolls at that hit
    band_self: jnp.ndarray  # it holds a Focus Band: a confusion self-hit rolls focus_band[s, 0]
    berry_by_foe: jnp.ndarray  # bool[2, B]: a foe move first can feed it a disliked flavor berry
    pending_own: jnp.ndarray  # it moves with a CONFUSION_PENDING counter at the start
    # G-30: it uses Present against a side with a Wonder Guard Pokemon: the draw picks the heal
    # path's SYSCTL_IGNORE_IMMUNITIES before the accuracy check, so a miss reads it too (the
    # Wonder Guard message, subscript_missed.s:107, or the plain miss).
    present_guard: jnp.ndarray
    # G-70: it uses Chatter as a Chatot (not Transformed): its confusion needs the chatter roll.
    chatter: jnp.ndarray


class OutcomeLaws(NamedTuple):
    """Integer law parameters of one (state, action pair), per side, and the relevance
    masks. Fixed shapes; a batch adds a leading axis to every field."""

    hit_thresholds: jnp.ndarray  # int32[B, outcome.HIT_VARIANTS], outcome.hit_thresholds
    hit_possible: jnp.ndarray  # bool[B, outcome.HIT_VARIANTS], which of them can occur
    # B13-6: a later Triple Kick hit's thresholds (outcome.hit_thresholds, overrides=False).
    hit_later_thresholds: jnp.ndarray  # int32[B, outcome.HIT_VARIANTS]
    hit_later_possible: jnp.ndarray  # bool[B, outcome.HIT_VARIANTS]
    hit_whole: jnp.ndarray  # bool[B]: every accuracy roll its own class (outcome.self_lock)
    # B12-4 review: the odds each side's hit can meet at its CalcCrit (outcome.crit_variants).
    crit_variants: jnp.ndarray  # int32[B, outcome.CRIT_VARIANTS], variant 0 the turn start
    crit_possible: jnp.ndarray  # bool[B, outcome.CRIT_VARIANTS]
    no_crit: jnp.ndarray  # bool[B], Lucky Chant on the defending side (outcome.py:157)
    secondary_chance: jnp.ndarray  # int32[B], tables.move_effect_chance of the move
    duration_kind: jnp.ndarray  # int32[B], DURATION_*
    relevant: Outcome  # bool, each field shaped like the ``Outcome`` field: any branch
    branch_relevant: Outcome  # bool, a leading axis [2]: the branch speed_tie = 0, 1
    reads: Reads  # the conditional reads' inputs
    # int32[B, MAX_HITS, 2, ROLLS]: per side and hit, hit_damage_table at the turn start (one
    # table for every hit but Triple Kick's power and Beat Up's member, B13-5, B13-10)
    damage: jnp.ndarray
    protect_count: jnp.ndarray  # int32[B]: the turn-start chain (protect's law, 0-3)
    roar_level_weight: jnp.ndarray  # int32[B]: of ROAR_LEVEL_VALUES, how many pass (roar_level)
    roar_bench: jnp.ndarray  # int32[B]: the eligible bench Pokemon Roar picks from (roar_target)
    compress: jnp.ndarray  # bool[2, B, MAX_HITS]: [branch, side, hit] uses damage classes
    flinch_item_chance: jnp.ndarray  # int32[B]: the King's Rock param (outcome.kings_rock_chance)
    # E2 batch 5: the invokers' call these laws assume (``call_laws``, ``call_variants``).
    call: jnp.ndarray  # int32[B]: the Outcome.call value the rows carry (Copycat: 0 / 1, none)
    branch_ok: jnp.ndarray  # bool[2]: the speed_tie branches this call is consistent with
    call_pending: jnp.ndarray  # bool[]: an invoker is used but no call was given
    # E2 batch 5: the Future Sight hit of this turn's end (outcome.future_thresholds), per defender.
    future_thresholds: jnp.ndarray  # int32[B, K]
    future_possible: jnp.ndarray  # bool[B, K]
    future_loose: jnp.ndarray  # bool[B]: every roll its own class
    # E2 batch 12 (B12-9): Acupressure's draw, per side: the count of its user's stats below +6
    # at the turn start, and whether that count can change before the move (every residue class
    # of the 420 draws its own value then).
    acupressure_count: jnp.ndarray  # int32[B]
    acupressure_loose: jnp.ndarray  # bool[B]
    # int32[B]: ids.CHATTER_CHANCES of each battler's party's Chatot cry (G-70)
    chatter_chance: jnp.ndarray
    # bool[B, 2]: F-10, the Starf Berry draw (``_starf_reads``) is read modulo a count that can
    # change before the eating: all STARF_DRAWS values; else the five of a fixed count of 5.
    starf_loose: jnp.ndarray


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


def _inflicts(tables: Tables, state: BattleState, moves, status, users=None) -> jnp.ndarray:
    """The move can inflict ``status`` on the foe (``STATUS_FOE`` / ``SECONDARY_STATUS`` /
    ``FANG_PARALYZE_FLINCH`` with ``move_param[m, 0]``, effects._inflict, effects._fang; and
    ``TRI_ATTACK`` any of ``TRI_ATTACK_STATUSES``, effects._tri_attack), Secret Power by its
    terrain's effect (``order.move_effects``). Psycho Shift (B12-17) gives its user's status
    (``effects.psycho_shift_status``): ``users`` (``int[2]``) names each move's user side; without
    it any status counts (over-approximate)."""
    fmt = format_of(state)
    effect, params = move_effects(tables, state, moves)
    status_effect = _any_of(
        effect,
        (_E.STATUS_FOE, _E.SECONDARY_STATUS, _E.RECOIL_STATUS, _E.FANG_PARALYZE_FLINCH),
    )
    tri = (effect == _E.TRI_ATTACK) & (status in TRI_ATTACK_STATUSES)
    shift = effect == _E.PSYCHO_SHIFT
    if users is not None:
        users = jnp.asarray(users, _I32)
        own = state.status[party(fmt, users), state.active[users]]
        shift = shift & (psycho_shift_status(own) == status)
    return (status_effect & (params[..., 0] == status)) | tri | shift


def _entry_abilities(tables: Tables, state: BattleState, actions) -> jnp.ndarray:
    """``int[2]``: the ability the Pokemon side ``d`` switches in this turn holds once its
    switch-in checks ran, the traced copy included (``damage.entry_ability``; the switch goes
    first, ruling B3-19); meaningful only where ``d`` switches."""
    fmt = format_of(state)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.ability.shape[1] - 1)
    return jnp.stack(
        [entry_ability(tables, state, side, incoming[side], _foe(side)) for side in range(fmt.B)]
    )


def _holds(tables: Tables, state: BattleState, actions, abilities) -> jnp.ndarray:
    """``bool[B]``: side ``d``'s active Pokemon, or the one it switches in this turn, has one
    of ``abilities``: the incoming one by its own ability or the one it traces on entry
    (``_entry_abilities``; both, so a Trace newcomer counts as Trace too)."""
    fmt = format_of(state)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.ability.shape[1] - 1)
    sides = jnp.arange(fmt.B)
    now = _any_of(battler_ability(state, sides), abilities)
    entry = _any_of(battler_ability(state, sides, incoming), abilities) | _any_of(
        _entry_abilities(tables, state, actions), abilities
    )
    return now | (entry & is_switch(actions))


def _holds_item(tables: Tables, state: BattleState, actions, items) -> jnp.ndarray:
    """``bool[B]``: as ``_holds``, for held items with one of the effects ``items``."""
    fmt = format_of(state)
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.item.shape[1] - 1)
    sides = jnp.arange(fmt.B)
    now = _any_of(tables.item_effect[state.item[party(fmt, sides), state.active]], items)
    later = _any_of(tables.item_effect[state.item[party(fmt, sides), incoming]], items) & is_switch(
        actions
    )
    return now | later


def _end_speeds(
    tables: Tables,
    state: BattleState,
    actions,
    side,
    stage_free,
    para_free,
    dragged,
    lock_free,
    weather_free=False,
    tailwind_free=False,
    unburden_free=False,
    land_free=False,
):
    """``(int32[3 * (27 + 6 * SLOTS)], bool[...])``: the effective speeds ``side`` can have
    after the actions (``step.end_of_turn_order`` on the post-action state, step.py:1251-1261)
    and which of them are possible. Grid: Speed stage -6..6 x paralyzed or not of the
    Pokemon active after the actions, plus its neutral speed (a faint resets stages and
    status, ``step._faint_check``, step.py:948-992). A stage other than the start one needs ``stage_free``, a
    paralysis state other than the start one ``para_free``. Where ``dragged`` (the foe
    can Roar this side), every living party member at stage 0, paralyzed or not: the
    Pokemon the drag brings in (``InitBattleMon`` resets its stages), and a Salac Berry holder
    at +1 and a Starf Berry holder at +2 too (their switch-in item checks can raise its Speed).

    The weather lock: every grid point is also evaluated with the lock on and with it off
    where ``lock_free`` (a Cloud Nine / Air Lock holder can enter, leave or faint this
    turn, so the lock of the turn's end can differ from the one of the turn start, which
    the plain grid point reads from the other side's turn-start battler).
    ``BattleSystem_CompareBattlerSpeed`` doubles Swift Swim in rain only
    ``if (NO_CLOUD_NINE)`` (battle_lib.c:1228-1237; include/battle/common.h:63-64). Where
    ``weather_free`` (a weather move, a switch-in weather ability or a drag can change the
    weather this turn), each grid point is also taken in rain (sun for a Chlorophyll holder,
    the same line, E2 batch 9) without the lock and with no such weather (the lock-on point's
    speed).

    Quick Feet (:1257-1261) reads any status: for a holder the grid's paralyzed points stand
    for every status (x15/10, no quartering), and the caller frees them
    (``para_free``). Where ``tailwind_free`` (the side's own Tailwind can blow this turn,
    effects._side_layers), every point is also taken with Tailwind's x2 (battle_lib.c:1274-1276).
    Where ``unburden_free`` (an Unburden holder of the side can be on the field having lost its
    item, E2 batch 9, or get one back through Recycle, B13-19), every point is also taken with
    Unburden's state toggled (:1268-1272). Where ``land_free`` (a Sky Shaymin of the side can be
    frozen and hit this turn, F-16), every point is also taken with a Sky Shaymin's stats
    recalculated on Land Forme's row (``hooks.land_stats``: base Speed 127 -> 100,
    subscript_check_shaymin_form.s:22-30, battle_io_command.c:601-604, battle_lib.c:169-193)."""
    fmt = format_of(state)
    switching = is_switch(actions[side])
    mon = jnp.where(switching, actions[side] - MOVE_ACTIONS, state.active[side]).astype(
        state.active.dtype
    )
    stage0 = jnp.where(switching, 0, state.stages[side, Stat.SPE].astype(_I32))
    quick_feet = battler_ability(state, side, mon) == AbilityId.QUICK_FEET
    para0 = jnp.where(
        quick_feet,
        state.status[party(fmt, side), mon] != Status.NONE,
        state.status[party(fmt, side), mon] == Status.PARALYSIS,
    )
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
    living = (state.species[party(fmt, side)] != 0) & (state.hp[party(fmt, side)] > 0)
    held = state.item[party(fmt, side)]
    salac = (tables.item_effect[held] == ItemId.PINCH_BERRY) & (
        tables.item_param[held, 1] == Stat.SPE
    )
    # A Starf Berry holder at +2 too (its +2 on Speed, F-10).
    starf = tables.item_effect[held] == ItemId.STARF_BERRY
    n_slots = slots.shape[0]
    mons = jnp.concatenate([mons, jnp.tile(jnp.repeat(slots, 2), 3).astype(state.active.dtype)])
    stages = jnp.concatenate(
        [
            stages,
            jnp.zeros((2 * n_slots,), _I32),
            jnp.ones((2 * n_slots,), _I32),
            jnp.full((2 * n_slots,), 2, _I32),
        ]
    )
    paras = jnp.concatenate([paras, jnp.tile(jnp.array([False, True]), 3 * n_slots)])
    possible = jnp.concatenate(
        [
            possible,
            jnp.asarray(dragged) & jnp.repeat(living, 2),
            jnp.asarray(dragged) & jnp.repeat(living & salac, 2),
            jnp.asarray(dragged) & jnp.repeat(living & starf, 2),
        ]
    )
    # The weather lock: 0 = as on the field of the grid point, 1 = on, 2 = off, 3 = off in rain.
    n = stages.shape[0]
    locks = jnp.repeat(jnp.arange(4), n)
    mons, stages, paras = jnp.tile(mons, 4), jnp.tile(stages, 4), jnp.tile(paras, 4)
    lock_free = jnp.asarray(lock_free, bool)
    weather_free = jnp.asarray(weather_free, bool)
    possible = jnp.concatenate(
        [
            possible,
            possible & (lock_free | weather_free),
            possible & lock_free,
            possible & weather_free,
        ]
    )

    # Bit 0: Tailwind as on the field, or blowing (its own move this turn); bit 1: Unburden as on
    # the field, or toggled: a holder of an item unburdened (no item, canUnburden), a holder of
    # none holding one again (Recycle, B13-19).
    n = stages.shape[0]
    winds = jnp.repeat(jnp.arange(4), n)
    mons, stages, paras, locks = (jnp.tile(x, 4) for x in (mons, stages, paras, locks))
    wind_ok = jnp.asarray(tailwind_free, bool)
    unburden_ok = jnp.asarray(unburden_free, bool)
    possible = jnp.concatenate(
        [possible, possible & wind_ok, possible & unburden_ok, possible & wind_ok & unburden_ok]
    )
    # Bit 2: Shaymin-Sky's stats as on the field, or reverted to Land Forme's (F-16).
    winds = jnp.concatenate([winds, winds + 4])
    mons, stages, paras, locks = (jnp.tile(x, 2) for x in (mons, stages, paras, locks))
    possible = jnp.concatenate([possible, possible & jnp.asarray(land_free, bool)])
    row = party(fmt, side)

    def speed(mon, stage, para, lock, wind):
        unburdened = (wind & 2) != 0
        sky = (state.species[row, mon] == SPECIES_SHAYMIN) & (state.form[row, mon] != 0)
        land = ((wind & 4) != 0) & sky
        reverted = jnp.where(
            land, land_stats(tables, state, row, mon)[Stat.ATK :], state.stats[row, mon]
        )
        # A Pokemon other than the active one comes in fresh: UpdateAfterSwitch zeroes its
        # MoveEffectsData (battle_lib.c:2101-2104) and its switch-in check arms canUnburden only
        # for an Unburden holder that still holds an item (:4433-4437); B9-13.
        incoming = mon != state.active[side]
        armed = (battler_ability(state, side, mon) == AbilityId.UNBURDEN) & (
            state.item[party(fmt, side), mon] != 0
        )
        base = jnp.where(incoming, armed.astype(_I32), state.can_unburden[side].astype(_I32))
        # The toggled item: none for a holder of one; for a holder of none, an item back
        # (Recycle, B13-19: any item it can get back acts on no speed but through Unburden's
        # heldItem test, so a nonzero stand-in, its battler's record or item 1, is exact).
        held_now = state.item[party(fmt, side), mon]
        restored = jnp.where(state.recycle_item[side] != 0, state.recycle_item[side], 1)
        toggled = jnp.where(held_now != 0, 0, restored)
        s = state._replace(
            tailwind=state.tailwind.at[side_of(side)].set(
                jnp.where((wind & 1) != 0, TAILWIND_TURNS, state.tailwind[side_of(side)]).astype(
                    state.tailwind.dtype
                )
            ),
            can_unburden=state.can_unburden.at[side].set(
                jnp.where(unburdened, 1, base).astype(state.can_unburden.dtype)
            ),
            item=state.item.at[party(fmt, side), mon].set(
                jnp.where(unburdened, toggled, state.item[party(fmt, side), mon]).astype(
                    state.item.dtype
                )
            ),
        )
        s = s._replace(
            active=state.active.at[side].set(mon),
            stats=s.stats.at[row, mon].set(reverted.astype(s.stats.dtype)),
            # A newcomer's mask starts clear (battle_lib.c:2071): no Gastro Acid bit.
            ability_suppressed=state.ability_suppressed.at[side].set(
                jnp.where(incoming, 0, state.ability_suppressed[side]).astype(
                    state.ability_suppressed.dtype
                )
            ),
            stages=s.stages.at[side, Stat.SPE].set(stage.astype(s.stages.dtype)),
            status=s.status.at[party(fmt, side), mon].set(
                jnp.where(para, Status.PARALYSIS, Status.NONE).astype(state.status.dtype)
            ),
        )
        suppressed = jnp.where(lock == 0, weather_suppressed(tables, s), lock == 1)
        chlorophyll = battler_ability(s, side, mon) == AbilityId.CHLOROPHYLL
        doubling = jnp.where(chlorophyll, Weather.SUN, Weather.RAIN)
        weather = jnp.where(lock == 3, doubling, s.weather).astype(s.weather.dtype)
        return effective_speed(tables, s, side, suppressed, weather)

    return jax.vmap(speed)(mons, stages, paras, locks, winds), possible


def _branch_masks(foe_first, first_self, base: dict) -> dict:
    """Order-dependent masks of one branch (``outcome_laws``): ``foe_first`` /
    ``first_self`` (``bool[B]``) say whether each side's foe / the side itself acts
    first."""
    b = base
    para_by_foe = b["uses_move"] & foe_first & foes(b["para_inflict"])
    freeze_by_foe = b["uses_move"] & foe_first & foes(b["freeze_inflict"])
    confused_by_foe = b["uses_move"] & foe_first & foes(b["confuse_inflict"])
    infatuated_by_foe = b["uses_move"] & foe_first & foes(b["attract_inflict"])
    # A foe moving first can bring an unconfused holder of a disliked flavor berry to half HP
    # (hooks.held_item_check, G-5): its check then reads the berry's counter.
    berry_by_foe = b["uses_move"] & foe_first & b["berry_confuse"] & foes(b["uses_move"])
    confusable = b["confused_own"] | confused_by_foe | berry_by_foe
    # A flinch roll is read only when the user moves first (subscript_flinch_mon.s: the
    # target has not moved, effects._flinch). A secondary status or confusion on a target
    # that cannot take it changes nothing (status eligibility), judged on the turn-start
    # state, which is the state of the hit when the user moves first with one hit.
    status_effect = _any_of(
        b["effect"],
        (_E.SECONDARY_STATUS, _E.RECOIL_STATUS, _E.FANG_PARALYZE_FLINCH, _E.TRI_ATTACK),
    )
    # A lock aimed at its user: its target has not moved either, whatever the order.
    first_self = first_self | b["self_lock"]
    blocked = first_self & (
        (status_effect & b["status_blocked"])
        | ((b["effect"] == _E.SECONDARY_CONFUSE) & b["confused_blocked"])
    )
    late_flinch = ~first_self & _any_of(b["effect"], (_E.FLINCH, _E.FAKE_OUT))
    secondary = b["secondary"] & ~(blocked | late_flinch)[:, None]
    return {
        "roll": b["crit"] | ((confusable | b["future_roll"])[:, None] & (b["hit_index"] == 0)),
        "secondary": secondary,
        # Tri Attack's status draw: read only through its proc (effects._tri_attack), so only
        # where that roll is; a chance-0 Tri Attack (none in either game: the decomp asserts a
        # chance, battle_lib.c:1576) would read it on every landed hit.
        "tri_attack": b["tri_attack"] & ~blocked & (secondary[:, 0] | b["certain"]),
        # Thunder Fang's flinch roll: the same order gate, and Inner Focus (effects._fang).
        "fang_flinch": b["fang_flinch"] & first_self & ~b["flinch_blocked"],
        # King's Rock / Razor Fang (effects._kings_rock): the same gates, per hit.
        "flinch_item": b["flinch_item"] & (first_self & ~b["flinch_blocked"])[:, None],
        # Focus Band (G-1): the foe's on the move's hits, the side's own on a self-hit.
        "focus_band": b["focus_band"]
        | ((confusable & b["own_band"])[:, None] & (b["hit_index"] == 0)),
        "berry_confusion": b["pending"] | berry_by_foe,
        "chatter": b["chatter"],
        "berry_by_foe": berry_by_foe,
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
    # E2 batch 13: hits that write the foe's HP only (B13-5, B13-9).
    _E.TRIPLE_KICK,
    _E.BEAT_UP,
    # B13-12..B13-17: single hits whose own power or type varies; they write the foe's HP only
    # (the HP clause), and Assurance's flag, which no table damage reads.
    _E.WEATHER_BALL,
    _E.GYRO_BALL,
    _E.ASSURANCE,
    _E.TRUMP_CARD,
    _E.PUNISHMENT,
    _E.JUDGMENT,
    # B13-20: Natural Gift writes the foe's HP and removes its user's own berry (a berry is no
    # input of damage._pre_random / _finish; the item clauses read the turn-start holder, an
    # over-approximation once it is gone).
    _E.NATURAL_GIFT,
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
    # whatever (crit, roll) class the hit would have drawn, the state is the same. Endure
    # (G-65) shares the roll and the chain; its flag changes the foe's HP only through the cap.
    _E.PROTECT,
    _E.ENDURE,
    # Sets the foe's infatuated, read only by its attract check, which comes before its move
    # and removes the hit altogether (step._use_move), like a flinch: no damage input.
    _E.ATTRACT,
    # E2 batch 4. The defender's HP or the attacker's power only (the HP clause covers HP).
    _E.HP_POWER,
    _E.WEIGHT_POWER,
    _E.HALVE_HP,
    _E.OHKO,
    _E.FALSE_SWIPE,
    # The side conditions of Tailwind (speed), Spikes and Toxic Spikes (a newcomer: the switch /
    # roar clauses), and Destiny Bond (a faint: no hit follows).
    _E.TAILWIND,
    _E.SPIKES,
    _E.TOXIC_SPIKES,
    _E.DESTINY_BOND,
    # Never read here: order.move_effects resolves it to its terrain's secondary effect.
    _E.SECRET_POWER,
    # Damages the side: its HP (the HP clause).
    _E.COUNTER,
    # Never the effect read here: the predicate reads the move each action runs
    # (order.called_move), Nature Power's call; an unresolved one (unknown terrain, refused
    # by coverage) calls MOVE_NONE and does nothing.
    _E.NATURE_POWER,
    # The invokers likewise (order.used_moves): their called move is read; one whose call
    # fails does nothing.
    _E.SLEEP_TALK,
    _E.METRONOME,
    _E.COPYCAT,
    _E.ASSIST,  # B10-11: an invoker like them
    _E.ME_FIRST,  # E3-2-3: an invoker of its target's move (order.me_first_copy)
    # E2 batch 5. Uproar starts an uproar on its user, which wakes a sleeper at its own status
    # check: the side's own status change, which ``own_pre_move`` covers for a Guts user.
    _E.UPROAR,
    # Stores a hit for a later turn's end (step._future_sight): nothing of this turn's hits.
    _E.FUTURE_SIGHT,
    # E2 batch 7. Mean Look traps the side (its switch is chosen already); Imprison seals the
    # side's move (its status check stops the move altogether, like Taunt): no damage input.
    _E.MEAN_LOOK,
    _E.IMPRISON,
    # E2 batch 8. Lock-On marks the side's battler for the foe's later moves (accuracy only);
    # Perish Song's count acts at the turn end.
    _E.LOCK_ON,
    _E.PERISH_SONG,
    # Yawn's sleep comes at the turn end, after every hit.
    _E.YAWN,
    # E2 batch 9: Follow Me draws nothing and changes no damaging move's defender at B = 2 (the foe
    # already is it); the Curse it redirects (lib.c:1714-1717) has no accuracy or damage roll. At
    # B = 4 enumeration is D11's. Helping Hand fails in singles.
    _E.FOLLOW_ME,
    _E.HELPING_HAND,
    # The user's Attack and HP only (the foe's later hit reads the user's Defense and HP: the HP
    # clause covers HP).
    _E.BELLY_DRUM,
    # B10-2: a hit, and on a miss its crash hurts the foe (the HP clause).
    _E.CRASH,
    # B10-4: the user's HP (the HP clause) and its substitute, which no table-damage input reads
    # (SmellingSalt's x2 is DOUBLE_POWER's, never compressed: NON_TABLE_DAMAGE_EFFECTS).
    _E.SUBSTITUTE,
    # B10-5: both battlers' HP only (the HP clause).
    _E.ENDEAVOR,
    _E.PAIN_SPLIT,
    # B10-7: a side condition that stops statuses (a burn's Attack is the status clause's).
    _E.SAFEGUARD,
    # B12-23: a side condition that stops stat drops (a dropped stage is the stat clause's).
    _E.MIST,
    # B10-8: a plain hit, or none once the user took a hit (its foe's damage is unchanged).
    _E.FOCUS_PUNCH,
    # B10-9: it forces the foe's later turns' move; nothing of this turn's damage.
    _E.ENCORE,
    # B12-4: its user's own crit stages (another battler's crit odds are crit_variants').
    _E.FOCUS_ENERGY,
    # B12-12..16: the user's own HP (the HP clause), status and party; Wish's heal comes at the
    # turn's field pass, after every hit.
    _E.REST,
    _E.WEATHER_HEAL,
    _E.WISH,
    _E.HEAL_BELL,
    _E.REFRESH,
    # E1 trial batch. Mimic (BtlCmd_TryMimic, battle_script.c:4476-4524) writes only its user's
    # moves, ppCur, mimickedMoveSlot and lastResortCount: none is an input of the side's hit,
    # whose move is chosen already (G-68). Chatter (effect_script_0267.s:5-7) is a plain hit
    # whose ON_HIT subscript_chatter.s only confuses, which no damage input reads (G-70).
    _E.MIMIC,
    _E.CHATTER,
    # E3 batch 1: Splash / Kaizo Thief deals no damage and writes no state but its user's
    # last move (effect_script_0085.s:5), Teleport fails (effect_script_0153.s:9-10): neither
    # changes an input of the side's hit (E3-1-1, E3-1-4).
    _E.DO_NOTHING,
    _E.FLEE,
    # E3-2-1: Torment sets a bit read only at the target's next move selection
    # (CHECK_INVALID_TORMENTED, battle_lib.c:2275-2279); no hit of this turn reads it.
    _E.TORMENT,
    # E3-2-4: Trick Room sets trickRoomActivated (battle_script.c:7863), which re-sorts the
    # action order and restarts the turn counter (battle_controller_player.c:3990-3994); in
    # singles the foe has either acted already or still acts next, and the move's user is
    # marked done, so no action changes. Its flip of the order bit is read at the next turn's
    # order (order.py), not by a hit of this one. A doubles partner's order is not a singles case.
    _E.TRICK_ROOM,
)
INPUT_CLAUSE_EFFECTS = (
    _E.STAT_SELF,  # the foe's Defense / Sp. Def; Minimize's flag (Stomp's power)
    # B10-6: it takes the foe's held item, which its later hit may read (a power item, Life Orb,
    # a Choice item): the clause below, any item held.
    _E.KNOCK_OFF,
    # The user's Attack and Speed (Dragon Dance) or Defense (Bulk Up): the stat clause.
    _E.DRAGON_DANCE,
    _E.SECONDARY_STAT_SELF,
    _E.STAT_FOE,  # the attacker's Attack / Sp. Atk
    _E.SECONDARY_STAT_FOE,
    _E.RAISE_ALL_STATS_HIT,
    _E.CURSE,  # the foe's Defense
    _E.STATUS_FOE,  # the attacker's burn, or any status under Guts
    _E.SECONDARY_STATUS,
    # B10-1: its status like SECONDARY_STATUS's (the status clause); its recoil hurts the foe
    # (the HP clause).
    _E.RECOIL_STATUS,
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
    # The weather (rain / sun on Fire and Water, the sandstorm Sp. Def boost, Solar Power):
    # the weather clause.
    _E.WEATHER,
    # SmellingSalt cures the side's paralysis (effects._cure_paralysis): the status clause
    # under Guts.
    _E.DOUBLE_POWER,
    # Raises the side's Attack (effects._swagger): the stat clause on a physical hit.
    _E.SWAGGER,
    # The roosting foe's chart (damage.chart_types): the roost clause.
    _E.ROOST,
    # The ingrained foe's chart and Levitate (damage.identified, damage.ignorable): the ingrain
    # clause.
    _E.INGRAIN,
    # Magnet Rise (G-66): the foe's Ground-immunity bits (damage.magnet_risen), the same clause.
    _E.MAGNET_RISE,
    # Gravity (G-67) lifts the Flying and Levitate immunities of the foe the side's later Ground
    # hit meets (damage.identified, damage.ignorable) and zeroes Magnet Rise's turns: the same clause.
    _E.GRAVITY,
    # E2 batch 6: the foe's U-turn / Baton Pass switches it out mid-turn (step's pause), so the
    # side's later hit meets the newcomer (its stats, types, ability, item; a Baton Pass keeps
    # the stages): the pivot clause.
    _E.U_TURN,
    _E.BATON_PASS,
    # E2 batch 12, the stat clause: Tickle lowers the side's Attack (and Defense); Captivate its
    # Sp. Atk; Haze, Psych Up, Acupressure and Power / Guard Swap change stages the side's hit
    # reads (its attacking one or the foe's defending one); Power Trick swaps the foe's Attack
    # and Defense (damage.battle_stats).
    _E.TICKLE,
    _E.CAPTIVATE,
    _E.HAZE,
    _E.PSYCH_UP,
    _E.STAGE_SWAP,
    _E.ACUPRESSURE,
    _E.POWER_TRICK,
    # B12-17: the user's status onto the side: the status clause (``status_hit``).
    _E.PSYCHO_SHIFT,
    # B13-19: the foe's Recycle gives it back an item (a berry it eats after the side's hit, a
    # White Herb, ...): the recycle clause.
    _E.RECYCLE,
    # B13-31: the foe's Embargo hides the side's held item from its later hits' damage reads
    # (Battler_HeldItem, battle_lib.c:5358): the embargo clause.
    _E.EMBARGO,
    # E1 trial batch (G-69): BtlCmd_Transform (battle_script.c:5330-5373) copies the defender's
    # BattleMon up to ``ability`` (battle_mon.h:57-84) onto its user: species (weight), Attack to
    # Sp. Def, the stat stages, both types and the ability, and reruns its switch-in checks
    # (the announce flags cleared, :5350-5358, rerun at bcp.c:3672-3673 and :3973: a copied
    # Intimidate or weather ability acts). Every input of the side's later
    # hit against it, and of its own stats the side's hit meets, can change: the transform clause.
    _E.TRANSFORM,
    # E3-2-2: Worry Seed overwrites the side's ability with Insomnia before its later hit
    # (subscript_give_target_insomnia.s:11): every ability factor of that hit, its accuracy and
    # crit odds, and a Klutz holder's item: the worry-seed clause.
    _E.WORRY_SEED,
    # E3-2-5: Skill Swap gives the side the foe's ability and the foe the side's
    # (subscript_exchange_abilities.s:21-24) before the side's later hit, and the entry ability
    # one gains acts at the move's end (``hooks.move_end_switch_in``): every ability factor of
    # that hit, its accuracy and crit odds, a Klutz holder's item, an Intimidate or Download
    # stage change of its stats: the skill-swap clause.
    _E.SKILL_SWAP,
    # E3-2-6: Gastro Acid suppresses the side's ability before its later hit
    # (subscript_suppress_target_ability.s:10; ``Battler_Ability`` reads NONE, battle_lib.c:3088-
    # 3093): every ability factor of that hit, its accuracy and crit odds, a Klutz holder's item,
    # the speed abilities of the turn's end: the gastro-acid clause.
    _E.GASTRO_ACID,
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
    # Fixed damage factors of the hit's own battlers, moves and types (damage._pre_random,
    # _finish); Solar Power reads the weather, whose changes have their clauses (the weather
    # clause, the switch clause).
    AbilityId.TECHNICIAN,
    AbilityId.RECKLESS,
    AbilityId.THICK_FAT,
    AbilityId.DRY_SKIN,  # its Fire factor; its Water absorb removes the hit
    AbilityId.SOLAR_POWER,
    AbilityId.TINTED_LENS,
    AbilityId.LEVITATE,
    AbilityId.SCRAPPY,
    # The crit odds and blocks, accuracy, speed, secondary chance and secondary blocks: no
    # damage input; a crit is the hit's own draw.
    AbilityId.SUPER_LUCK,
    AbilityId.SHELL_ARMOR,
    AbilityId.BATTLE_ARMOR,
    AbilityId.NO_GUARD,
    AbilityId.TANGLED_FEET,
    AbilityId.QUICK_FEET,
    AbilityId.SERENE_GRACE,
    AbilityId.SHIELD_DUST,
    AbilityId.TRUANT,  # removes its holder's hit
    # D2-12: no effect at B = 2 (redirects only to a battler other than the defender).
    AbilityId.LIGHTNINGROD,
    AbilityId.STORM_DRAIN,
    AbilityId.HONEY_GATHER,  # after the battle
    # E2 batch 4: fixed damage factors of the hit's own battlers (damage._pre_random,
    # _finish); Magic Guard only spares its holder HP losses (the HP clause).
    AbilityId.FILTER,
    AbilityId.SNIPER,
    AbilityId.MAGIC_GUARD,
    # E2 batch 9. Speed only (Chlorophyll, Unburden); a fixed Attack factor of the hit's own
    # attacker (Huge Power, Pure Power); Flower Gift reads the weather (its clauses) and its
    # holder's HP > 0 (a fainted holder has no hit to read it).
    AbilityId.CHLOROPHYLL,
    AbilityId.UNBURDEN,
    AbilityId.HUGE_POWER,
    AbilityId.PURE_POWER,
    AbilityId.FLOWER_GIFT,
    # B9-6: an absorbed move has no hit; Rain Dish heals at the turn end; Clear Body / Hyper
    # Cutter only remove stage drops (each drop has its own clause).
    AbilityId.VOLT_ABSORB,
    AbilityId.WATER_ABSORB,
    AbilityId.RAIN_DISH,
    AbilityId.CLEAR_BODY,
    AbilityId.HYPER_CUTTER,
    # B9-8: turn-end HP only.
    AbilityId.POISON_HEAL,
    AbilityId.BAD_DREAMS,
    # B9-11: they only gate the switch choice, made before the turn.
    AbilityId.ARENA_TRAP,
    AbilityId.SHADOW_TAG,
    AbilityId.MAGNET_PULL,
    # G-8..G-10: PP and the AI's knowledge only, no damage input.
    AbilityId.PRESSURE,
    # G-14: it only stops statuses (fewer input changes than the clauses already allow).
    AbilityId.LEAF_GUARD,
    # G-18: a fixed factor of the hit's own battlers and move through the type chart, whose
    # inputs have their clauses (the roost, ingrain, camouflage and switch clauses).
    AbilityId.WONDER_GUARD,
    # E1 Batch C: Ice Body and Speed Boost act at the turn's end, after every hit (G-45, G-48);
    # White Smoke only removes stage drops, as Clear Body (G-46); Vital Spirit blocks sleep, as
    # Insomnia (G-47); Iron Fist is a fixed power factor of the move (G-49).
    AbilityId.ICE_BODY,
    AbilityId.SPEED_BOOST,
    AbilityId.WHITE_SMOKE,
    AbilityId.VITAL_SPIRIT,
    AbilityId.IRON_FIST,
    # B13-29: Klutz hides its holder's item from every gated read (state.held_item) from the
    # start of the turn; nothing changes it mid-turn but a foe's Worry Seed (E3-2-2: the
    # worry-seed clause) or Gastro Acid (E3-2-6: the gastro-acid clause).
    AbilityId.KLUTZ,
    # E1 trial batch (G-63): a fixed Sp. Atk factor of the hit's own side's active battlers
    # (the ally's ability and HP), whose inputs have their clauses.
    AbilityId.PLUS,
    AbilityId.MINUS,
    # E1 batch F (F-1): a fixed Fire power factor of the move, and the burn's own turn-end loss.
    AbilityId.HEATPROOF,
    # F-3: Multitype's types are fixed by the held plate, which nothing can swap or remove
    # (Transform's copy has its own clause).
    AbilityId.MULTITYPE,
    # F-5: Hydration cures in the turn-end weather step, after every hit.
    AbilityId.HYDRATION,
)
INPUT_CLAUSE_ABILITIES = (
    AbilityId.GUTS,  # status clause
    AbilityId.BLAZE,  # HP clause (pinch power)
    AbilityId.TORRENT,
    AbilityId.OVERGROW,
    AbilityId.SWARM,
    AbilityId.FLAME_BODY,  # contact clause (burn)
    AbilityId.STATIC,  # contact clause (a status under Guts)
    AbilityId.POISON_POINT,
    # G-12, G-13: passes a status mid-turn: the synchronize clauses (a burn, or any status under
    # Guts, on the receiver's later hit).
    AbilityId.SYNCHRONIZE,
    # F-2: a foe's Fire move it absorbs sets the flag its own later Fire hit reads (flash_fire clause).
    AbilityId.FLASH_FIRE,
    AbilityId.ROUGH_SKIN,  # the attacker's HP: HP clause
    AbilityId.AFTERMATH,  # likewise, on its holder's faint (no later hit on it)
    AbilityId.INTIMIDATE,  # on switch-in: switch clause
    AbilityId.DOWNLOAD,  # on switch-in (G-21): switch clause (a Trace copy too), pivot clause
    # On switch-in (the switch clause): the weather, or the copied ability.
    AbilityId.TRACE,
    AbilityId.DRIZZLE,
    AbilityId.SAND_STREAM,
    AbilityId.DROUGHT,
    AbilityId.SNOW_WARNING,
    # F-6: Castform's types follow the weather, which changes mid-turn only by an earlier foe
    # weather move (weather clause) or a switch-in weather ability or Cloud Nine (switch clause).
    AbilityId.FORECAST,
)
INPUT_NEUTRAL_ITEMS = (
    ItemId.NONE,
    ItemId.FOCUS_SASH,  # after the damage (effects.play_hit)
    ItemId.FOCUS_BAND,  # likewise (G-1)
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
    # Only the weather's duration (effects._weather); the weather is its move's clause.
    ItemId.WEATHER_ROCK,
    # The crit odds (outcome.crit_variants: a Knock Off of it turns the hit's damage classes
    # off): a crit is the hit's own draw.
    ItemId.CRIT_ITEM,
    # B10-12: it skips its holder's charge and is used up; no damage input reads it (the
    # charge's own clauses are the move's: MoveFlag.CHARGE above).
    ItemId.POWER_HERB,
    # E1 Batch D. PP (G-58), the accuracy flag (G-59), Bind's count (G-57): no damage input.
    ItemId.LEPPA_BERRY,
    ItemId.MICLE_BERRY,
    ItemId.GRIP_CLAW,
    # Its count changes only at its holder's own BeforeMove, which the turn-start table already
    # applies (outcome_laws, effects.metronome_update), G-60.
    ItemId.METRONOME_ITEM,
    # A fixed input of the type chart and Levitate (never consumed; Knock Off is its own clause)
    # and a speed factor of the order fixed at the turn's start (G-56).
    ItemId.IRON_BALL,
    # E1 Batch F. Macho Brace and the Power items: the same speed factor of the order fixed at
    # the turn's start (F-7). A Lansat Berry sets its holder's Focus Energy after a foe's hit:
    # only the holder's crit odds, a variant bit of ``outcome.crit_variants`` (F-9).
    ItemId.MACHO_BRACE,
    ItemId.LANSAT_BERRY,
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
    ItemId.BERSERK_GENE,  # the same item clause (F-8); not plucked
    ItemId.STARF_BERRY,  # item clause: +2 on any of the five stats (F-10); pluck clause
    ItemId.LAVA_COOKIE,  # a status cure, as Lum Berry's (F-12); not plucked
    # Its heal is the holder's HP (HP clause); its confusion is no damage input (G-4, G-5).
    ItemId.FLAVOR_BERRY,
    # The attacker's HP (G-54) and the holder's (G-55): HP clause.
    ItemId.JABOCA_BERRY,
    ItemId.ROWAP_BERRY,
    ItemId.ENIGMA_BERRY,
)
_STATUS_CURES = (
    ItemId.CHERI_BERRY,
    ItemId.CHESTO_BERRY,
    ItemId.PECHA_BERRY,
    ItemId.RAWST_BERRY,
    ItemId.ASPEAR_BERRY,
    ItemId.LUM_BERRY,
    ItemId.LAVA_COOKIE,
)
_BURN_CURES = (ItemId.RAWST_BERRY, ItemId.LUM_BERRY, ItemId.LAVA_COOKIE)

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
    # U-turn's hit is a plain hit (effect_script_0228.s: CalcCrit, CalcDamage); the switch comes
    # after it (E2 batch 6).
    _E.U_TURN,
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
    # The defender's weight, fixed per species: a new defender is the switch clause.
    _E.WEIGHT_POWER,
    # A table hit whose secondary effect is the terrain's (never read: resolved).
    _E.SECRET_POWER,
    # Its table damage, capped after the roll (effects.play_hit) by the target's HP alone.
    _E.FALSE_SWIPE,
    # A plain hit (effect_script_0159.s: CalcCrit, CalcDamage); the uproar comes after it.
    _E.UPROAR,
    # B10-1, B10-2: plain hits (CalcCrit, CalcDamage; Reckless is the attacker's fixed ability);
    # the recoil and status come after the damage, and a crash reads the same table entry.
    _E.RECOIL_STATUS,
    _E.CRASH,
    # B10-6: CalcCrit, CalcDamage (effect_script_0188.s:6-7); the item goes after the damage.
    _E.KNOCK_OFF,
    # B10-8: CalcCrit, CalcDamage (effect_script_0170.s:8-9) unless the focus is lost first.
    _E.FOCUS_PUNCH,
    # B13-5, B13-10: their per-hit tables (``OutcomeLaws.damage[side, hit]``: Triple Kick's
    # k x step power, Beat Up's member), each its damage for the hit's (crit, roll).
    _E.TRIPLE_KICK,
    _E.BEAT_UP,
    # B13-12: Weather Ball's power and type from the weather (the weather clause; a weather
    # ability or Cloud Nine arrives with a switch, the switch clause), Judgment's type from the
    # user's held item (Knock Off, the item clause, is the only mid-turn change of it). Water
    # Spout / Eruption are FLAIL (its HP clause, hp_sensitive), B13-13.
    _E.WEATHER_BALL,
    _E.JUDGMENT,
    # G-70: CalcCrit, CalcDamage (effect_script_0267.s:6-7) at its table power; the confusion is
    # an ON_HIT subscript after the damage.
    _E.CHATTER,
)
NON_TABLE_DAMAGE_EFFECTS = (
    _E.ENCORE,  # a status move: no damage (B10-9)
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
    _E.MAGNET_RISE,  # a status move: no damage
    _E.GRAVITY,  # a status move: no damage
    _E.PRESENT,  # its power is drawn (40/80/120, Outcome.present); the table has power 1
    _E.PROTECT,  # a status move: no damage
    _E.ENDURE,  # a status move: no damage
    _E.ATTRACT,  # a status move: no damage
    _E.CAMOUFLAGE,  # a status move: no damage
    _E.ROAR,  # a status move: no damage
    _E.WEATHER,  # a status move: no damage
    # Power from the defender's HP or status, or the user's status, at the hit: not the
    # turn-start table's when either changes first (effects, the foe's own move or items).
    _E.DOUBLE_POWER,
    _E.HP_POWER,
    _E.HALVE_HP,  # no roll (damage.fixed_damage)
    _E.OHKO,  # no roll
    _E.SWAGGER,  # status moves: no damage
    _E.DRAGON_DANCE,
    _E.ROOST,
    _E.TAILWIND,
    _E.SPIKES,
    _E.TOXIC_SPIKES,
    _E.DESTINY_BOND,
    _E.COUNTER,  # the damage taken this turn, no roll
    # Its table power is 0: its damage is the called move's. outcome_laws reads the called
    # move (order.called_move), whose own effect decides (Tri Attack's is a table hit), so
    # this entry only keeps an unresolved Nature Power (unknown terrain) uncompressed.
    _E.NATURE_POWER,
    # The invokers likewise: their called move's effect decides; a failed call deals nothing.
    _E.SLEEP_TALK,
    _E.METRONOME,
    _E.COPYCAT,
    _E.ASSIST,  # B10-11: an invoker like them
    _E.ME_FIRST,  # E3-2-3: an invoker of its target's move (order.me_first_copy)
    # No damage at its use (its roll is TryFutureSight's stored damage, damage.fixed_damage).
    _E.FUTURE_SIGHT,
    _E.INGRAIN,  # status moves: no damage
    _E.BATON_PASS,
    _E.MEAN_LOOK,
    _E.IMPRISON,
    _E.LOCK_ON,
    _E.PERISH_SONG,
    _E.YAWN,
    _E.FOLLOW_ME,  # status moves: no damage (no defender change at B = 2; B = 4 is D11's)
    _E.HELPING_HAND,
    _E.BELLY_DRUM,
    _E.SUBSTITUTE,  # a status move: no damage (B10-4)
    _E.ENDEAVOR,  # no roll (damage.fixed_damage), B10-5
    _E.PAIN_SPLIT,  # a status move: no damage
    _E.SAFEGUARD,  # a status move: no damage (B10-7)
    # E2 batch 12: status moves, no damage.
    _E.TICKLE,
    _E.FOCUS_ENERGY,
    _E.HAZE,
    _E.PSYCH_UP,
    _E.STAGE_SWAP,
    _E.CAPTIVATE,
    _E.ACUPRESSURE,
    _E.POWER_TRICK,
    _E.REST,
    _E.WEATHER_HEAL,
    _E.WISH,
    _E.HEAL_BELL,
    _E.REFRESH,
    _E.PSYCHO_SHIFT,
    _E.MIST,
    # B13-14, B13-15, B13-17: a power from the speeds at the action's start, the PP after its
    # decrement (the action's move slot), the defender's raised stages or its damage this turn,
    # none of which the turn-start table can read.
    _E.GYRO_BALL,
    _E.TRUMP_CARD,
    _E.PUNISHMENT,
    _E.ASSURANCE,
    # B13-19, B13-20: Recycle deals no damage; Natural Gift's power and type are its berry's,
    # which the script removes before the hit (damage.gift_item).
    _E.RECYCLE,
    _E.NATURAL_GIFT,
    _E.EMBARGO,  # B13-31: a status move, no damage
    _E.MIMIC,  # G-68, G-69: status moves, no damage
    _E.TRANSFORM,
    _E.DO_NOTHING,  # no CalcDamage: damage 0 (damage.fixed_damage), E3-1-1
    _E.FLEE,  # a status move that fails: no damage (E3-1-4)
    _E.TORMENT,  # a status move: no damage (E3-2-1)
    _E.TRICK_ROOM,  # a status move: no damage (E3-2-4)
    _E.WORRY_SEED,  # a status move: no damage (E3-2-2)
    _E.SKILL_SWAP,  # a status move: no damage (E3-2-5)
    _E.GASTRO_ACID,  # a status move: no damage (E3-2-6)
)


def damage_inputs_can_change(
    tables: Tables, state: BattleState, actions, foe_first, moves=None
) -> jnp.ndarray:
    """``bool[B, MAX_HITS]``: ``[side, hit]`` can deal damage other than
    ``hit_damage_table`` at the turn start says, for some ``(crit, roll)``, because an
    event earlier in the turn can change one of its damage inputs (``damage._pre_random``,
    ``damage._finish``). ``foe_first`` (``bool[B]``): the side's foe acts before it (one
    order branch). Sound, not tight: true whenever some clause below can apply.

    Inputs and the events that can change them before ``[side, hit]``:

    * **switch / item**: the foe switches (a new target, Intimidate, entry hazards,
      weather abilities) or uses a trainer item, before the side's move;
    * **weather**: an earlier foe weather move (``EffectId.WEATHER``) changes the weather
      every hit reads (the switch clause covers the weather abilities and Trace);
    * **foe move**: an earlier foe move with an unclassified effect (not in
      ``INPUT_NEUTRAL_EFFECTS`` / ``INPUT_CLAUSE_EFFECTS``), or one that changes: the
      side's attacking stat stage (``STAT_FOE``), the foe's defending stage
      (``STAT_SELF``, ``RAISE_ALL_STATS_HIT``, Curse's Defense), the side's status when
      it matters (a burn on a physical non-Guts move, any status under Guts, Thunder Fang's
      paralysis and Tri Attack's statuses included), the foe's screen for this category, a
      sport for this move type, or the foe's own types (Camouflage: the type chart and the
      Rock Sp. Def boost read them), or its stats, stages, types and ability (Transform, G-69);
      or the side's active Pokemon itself (the foe's Roar
      drags it out; no hit of a dragged battler follows in Gen 4, its action ends, so this
      roar clause is conservative); a charge move (its user's semi-invulnerable bit, which a
      ``MoveFlag.DOUBLE_SEMI`` hit's power reads; Skull Bash's Defense +1), B10-12;
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
    * **recycle**: the foe's earlier Recycle gives it back an item (B13-19; any, conservative);
    * **embargo**: the foe's earlier Embargo hides the side's held item (B13-31: a holder of one
      as ``state.held_item`` reads it at the turn start);
    * **own move**: a later hit of a move other than ``MULTI_HIT``, ``TRIPLE_KICK`` and
      ``BEAT_UP`` (Twineedle's second hit, B13-1: its first can poison a Synchronize holder;
      Triple Kick's and Beat Up's own hits change no input but the HP and contact clauses',
      and their per-hit tables are ``OutcomeLaws.damage[side, hit]``, B13-5, B13-10), a Guts
      user that wakes or thaws first (its status goes to none), and Selfdestruct (its user's
      HP goes to 0 before the hit).

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
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    sides = jnp.arange(fmt.B)
    mons = state.active
    live = state.over == 0
    # The move each action runs (Nature Power's call, order.called_move; an invoker's call,
    # ``moves`` from outcome_laws).
    if moves is None:
        chosen = jnp.stack(
            [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
        )
        moves = jnp.stack([called_move(tables, state, chosen[side]) for side in range(fmt.B)])
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    effect, params = move_effects(tables, state, moves)  # Secret Power's terrain effect
    param0, param1 = params[:, 0], params[:, 1]
    category = tables.move_category[moves]
    move_type = jnp.stack([move_type_of(tables, state, side, moves[side]) for side in range(fmt.B)])
    ability = battler_ability(state, sides, mons)
    item = tables.item_effect[state.item[party(fmt, sides), mons]]
    status = state.status[party(fmt, sides), mons]
    contact = (tables.move_flags[moves] & MoveFlag.CONTACT) != 0

    physical = category == PHYSICAL
    special = category == SPECIAL
    guts = physical & (ability == AbilityId.GUTS)  # any status counts (damage.py :6793)
    burnable = physical & (ability != AbilityId.GUTS)  # a burn halves it (:6978)
    attack_stat = jnp.where(physical, Stat.ATK, Stat.SPA)
    defense_stat = jnp.where(physical, Stat.DEF, Stat.SPD)
    hp_sensitive = (effect == _E.FLAIL) | pinch_ability(move_type, ability)
    stomp = (effect == _E.FLINCH) & (params[:, 2] == STOMP_PARAM)

    held = state.item[party(fmt, sides), mons]
    gated = held_items(state)  # state.held_item, B13-29
    pinch_stat = tables.item_param[held, 1]
    pinch_berry = item == ItemId.PINCH_BERRY
    pinch = pinch_berry | (item == ItemId.BERSERK_GENE)
    starf = item == ItemId.STARF_BERRY  # +2 on any of Attack, Defense, Speed, Sp. Atk, Sp. Def
    foe_effect, foe_param0, foe_param1 = foes(effect), foes(param0), foes(param1)
    own_flags = tables.move_flags[moves].astype(_I32)
    foe_flags = foes(own_flags)
    unclassified = ~_any_of(foe_effect, INPUT_NEUTRAL_EFFECTS + INPUT_CLAUSE_EFFECTS)
    status_hit = _any_of(
        foe_effect,
        (
            _E.STATUS_FOE,
            _E.SECONDARY_STATUS,
            _E.RECOIL_STATUS,
            _E.FANG_PARALYZE_FLINCH,
            _E.TRI_ATTACK,
            _E.PSYCHO_SHIFT,  # B12-17: the user's status onto the side (foe_burns reads which)
        ),
    )
    foe_burns = _inflicts(
        tables, state, foes(moves), Status.BURN, users=tuple(_foe(b) for b in range(fmt.B))
    )
    # F-16: a Shaymin-Sky that a hit freezes becomes Land Forme (hooks.shaymin_land): the foe's
    # earlier freezing move changes the side's own stats, types and ability, and the hits of
    # any move against a Sky Shaymin frozen at the turn start (the first hit reverts it).
    sky = (
        (state.species[party(fmt, sides), mons] == SPECIES_SHAYMIN)
        & (state.form[party(fmt, sides), mons] != 0)
        & (state.transformed == 0)
    )
    foe_freezes = _inflicts(
        tables, state, foes(moves), Status.FREEZE, users=tuple(_foe(b) for b in range(fmt.B))
    )
    foe_move = foes(uses_move) & (
        unclassified
        | (sky & foe_freezes)
        | (_any_of(foe_effect, (_E.STAT_FOE, _E.SECONDARY_STAT_FOE)) & (foe_param0 == attack_stat))
        | (
            _any_of(foe_effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF))
            & (foe_param0 == defense_stat)
        )
        | (foe_effect == _E.RAISE_ALL_STATS_HIT)
        | ((foe_effect == _E.CURSE) & physical)
        | (
            (foe_effect == _E.DRAGON_DANCE)
            & ((foe_param0 == defense_stat) | (foe_param1 == defense_stat))
        )
        # Minimize sets the flag Stomp's power reads (B9-10).
        | ((foe_effect == _E.STAT_SELF) & (foes(params)[:, 2] == MINIMIZE_PARAM) & stomp)
        # Swagger / Flatter raise the side's Attack / Sp. Atk (move_param 0, B12-2).
        | ((foe_effect == _E.SWAGGER) & (foe_param0 == attack_stat))
        # E2 batch 12: Tickle lowers the side's Attack (its Defense only meets the foe's own
        # hit, which came first), Captivate its Sp. Atk; Haze, Psych Up, Acupressure and the
        # stage swaps change a stage the hit reads; Power Trick the foe's Defense.
        | ((foe_effect == _E.TICKLE) & physical)
        | ((foe_effect == _E.CAPTIVATE) & special)
        | _any_of(foe_effect, (_E.HAZE, _E.PSYCH_UP, _E.ACUPRESSURE, _E.STAGE_SWAP))
        | ((foe_effect == _E.POWER_TRICK) & physical)
        | (foe_effect == _E.ROOST)
        | (foe_effect == _E.INGRAIN)
        | (foe_effect == _E.MAGNET_RISE)
        | (foe_effect == _E.GRAVITY)
        # G-69: the foe's Transform copies the side's stats, stages, types and ability onto it.
        | (foe_effect == _E.TRANSFORM)
        # E3-2-2: the foe's Worry Seed makes the side's ability Insomnia (the worry-seed clause).
        | (foe_effect == _E.WORRY_SEED)
        # E3-2-5: the foe's Skill Swap gives the side the foe's ability (the skill-swap clause).
        | (foe_effect == _E.SKILL_SWAP)
        # E3-2-6: the foe's Gastro Acid suppresses the side's ability (the gastro-acid clause).
        | (foe_effect == _E.GASTRO_ACID)
        | _any_of(foe_effect, (_E.U_TURN, _E.BATON_PASS))
        | ((foe_effect == _E.DOUBLE_POWER) & guts)
        | (status_hit & (guts | (burnable & foe_burns)))
        | ((foe_effect == _E.REFLECT) & physical)
        | ((foe_effect == _E.LIGHT_SCREEN) & special)
        | ((foe_effect == _E.MUD_SPORT) & (move_type == Type.ELECTRIC))
        | ((foe_effect == _E.WATER_SPORT) & (move_type == Type.FIRE))
        | (foe_effect == _E.CAMOUFLAGE)
        # F-2 flash_fire: the foe's Fire move on a Flash Fire side (absorbed or not: over-
        # approximate) sets the flag that boosts the side's own later Fire hit.
        | (
            (ability == AbilityId.FLASH_FIRE)
            & (move_type == Type.FIRE)
            & (foes(move_type) == Type.FIRE)
        )
        | (foe_effect == _E.ROAR)
        | (foe_effect == _E.WEATHER)
        | ((foe_effect == _E.EAT_BERRY) & pinch_berry & (pinch_stat == defense_stat))
        | ((foe_effect == _E.EAT_BERRY) & starf)
        | ((foe_effect == _E.KNOCK_OFF) & (held != 0))  # B10-6
        # B13-19: the foe's Recycle restores an item it then holds through the side's hits.
        | (foe_effect == _E.RECYCLE)
        # B13-31: the foe's Embargo hides the side's item from its hit's reads.
        | ((foe_effect == _E.EMBARGO) & (gated != 0))
        # B10-12: a charge move sets or (its second turn, or an unlock on its disruption) clears
        # its user's semi-invulnerable bit, which a DOUBLE_SEMI hit's power reads; Skull Bash's
        # charge raises the foe's Defense.
        | (((foe_flags & MoveFlag.CHARGE) != 0) & ((own_flags & MoveFlag.DOUBLE_SEMI) != 0))
        | (((foe_flags & MoveFlag.CHARGE_DEF_UP) != 0) & physical)
        | hp_sensitive
    )
    foe_action = live & (foes(is_switch(actions)) | foes(is_item(actions)) | foe_move)

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
    # B12-3, B12-7, B12-8: Tickle and the stage swaps change move_param 0 and 1, Captivate 0;
    # Haze, Psych Up and Acupressure any stage.
    two_stat = uses_move & _any_of(effect, (_E.TICKLE, _E.STAGE_SWAP))
    # Transform copies every stage (statBoosts, battle_mon.h:75), G-69.
    any_stage = jnp.any(
        uses_move & _any_of(effect, (_E.HAZE, _E.PSYCH_UP, _E.ACUPRESSURE, _E.TRANSFORM))
    )
    stat_move = stat_move | two_stat | (uses_move & (effect == _E.CAPTIVATE))
    any_switch = jnp.any(live & is_switch(actions))

    def lowerable(stat) -> jnp.ndarray:
        """``bool[B]``: per side ``s``, a move this turn changes stage ``stat[s]`` (the
        stage ``s``'s hit reads) of some battler, or a Pokemon switches in."""
        moved = jnp.stack(
            [
                jnp.any(stat_move & (param0 == stat[s])) | jnp.any(two_stat & (param1 == stat[s]))
                for s in range(fmt.B)
            ]
        )
        return moved | any_switch | any_stage

    attack_stage = state.stages[sides, attack_stat] < 0
    defense_stage = state.stages[foes(sides), defense_stat] < 0
    herb_inputs = (herb & (attack_stage | lowerable(attack_stat))) | (
        foes(herb) & (defense_stage | lowerable(defense_stat))
    )
    # A stat pinch berry (hooks.held_item_check) raises its holder's stage after a hit or a
    # residual: the side's own on the stage its hit reads (Attack / Sp. Atk), or the foe's
    # on its defending one (Defense / Sp. Def).
    pinch_inputs = (
        (pinch & (pinch_stat == attack_stat))
        | (foes(pinch) & (foes(pinch_stat) == defense_stat))
        | starf  # attack_stat and defense_stat are among its five stats
        | foes(starf)
    )
    items = (
        herb_inputs
        | pinch_inputs
        | (guts & _any_of(item, _STATUS_CURES))
        | (burnable & _any_of(item, _BURN_CURES))
    )

    foe_ability = foes(ability)
    contact_status = contact & (
        ((foe_ability == AbilityId.FLAME_BODY) & (guts | burnable))
        | (_any_of(foe_ability, (AbilityId.STATIC, AbilityId.POISON_POINT)) & guts)
    )
    # Synchronize (effects.synchronize, G-12, G-13): a foe holder the side's hit statuses passes
    # the status back before the side's next hit; a foe holder whose contact hit met the side's
    # Static / Poison Point / Flame Body passes that status to the side before the side moves
    # (the synchronize clauses; the passed status matters as a burn, or any under Guts).
    sync_back = (foe_ability == AbilityId.SYNCHRONIZE) & (guts | burnable)
    sync_contact = (
        foes(uses_move)
        & foes(contact)
        & (foe_ability == AbilityId.SYNCHRONIZE)
        & (
            ((ability == AbilityId.FLAME_BODY) & (guts | burnable))
            | (_any_of(ability, (AbilityId.STATIC, AbilityId.POISON_POINT)) & guts)
        )
    )
    foe_action = foe_action | (live & sync_contact)
    repeat = _any_of(effect, (_E.MULTI_HIT, _E.TRIPLE_KICK, _E.BEAT_UP))
    # F-16: a later hit against a Sky Shaymin frozen at the turn start meets Land Forme.
    sky_frozen = foes(sky & (status == Status.FREEZE))
    own_hits = hp_sensitive | ~repeat | contact_status | sync_back | sky_frozen
    own_pre_move = (guts & _any_of(status, (Status.SLEEP, Status.FREEZE))) | (
        effect == _E.SELF_DESTRUCT
    )

    later = jnp.arange(MAX_HITS)[None, :] >= 1
    before = jnp.asarray(foe_first, bool) & (foe_action | items)
    return own_pre_move[:, None] | before[:, None] | (later & (own_hits | items)[:, None])


def outcome_laws(
    tables: Tables, state: BattleState, actions, call=None, taken=None, *, loose: bool = False
) -> OutcomeLaws:
    """The law parameters and relevance masks of ``step(tables, state, actions, .)``
    (module docstring). ``loose`` (tests only: the brute-force reference) drops every
    order and eligibility gate and every conditional read: each element any order or
    event can read is relevant in both branches, and ``reads`` gates nothing.

    ``call`` (``int32[B]``, one variant of ``call_variants``): the invokers' call these laws
    assume: Sleep Talk's and Metronome's ``Outcome.call`` value, which the rows carry, and
    Copycat's order (1: its foe used a move before it, so movePrev is the foe's moveTemp; 0:
    the turn-start movePrev), which keeps only the consistent ``speed_tie`` branches
    (``branch_ok``). Without ``call`` an invoker runs nothing (``call_pending``:
    ``enumerate_drawn`` refuses those laws).

    ``taken`` (``bool[B]``, a ``call_variants`` variant's tag, default none): the takeover variant
    (ruling B11-1) of each side it marks, whose chosen move a foe's Encore replaces before its
    action (BattleControllerPlayer_FightCommand, bcp.c:1866-1873). Its moveTemp is then the
    encored move (its last move) and every move-dependent law (the move run, its effect, crit
    odds, damage classes, accuracy thresholds, the call) is the encored move's, read with
    ``order.encored_state``; the order stays the chosen move's (CalcTurnOrder fixes it at the turn
    start, before any action). ``call_consistent`` keeps the rows where it happened.

    A state paused at a party list (E2 batch 6) is read through ``outcome.pivot_view``: the
    order branch is the stored one (a switch first), and the paused action's King's Rock
    (``flinch_item[pivot, 0]``, the newcomer's item, ``step._use_move``) is relevant when its hit
    damaged a target that has not moved."""
    fmt = format_of(state)
    paused_rock = _paused_rock(tables, state)
    state, actions = pivot_view(tables, state, actions)
    live = state.over == 0
    sides = jnp.arange(fmt.B)
    switch, item = is_switch(actions), is_item(actions)
    uses_move = live & ~switch & ~item  # bool[B]
    # Singles follow-up (ii): a lock move that may be aimed at its own user (outcome.self_lock).
    # Every cut below that reads the foe as the defender (its armor, immunity, Protect, status
    # eligibility, order gates of a flinch, the damage table) is lifted for such a side, whose
    # defender can also be the foe (a faster Encore or Follow Me): over-approximate, sound.
    lock = uses_move & self_lock(tables, state)
    # ``chosen``: the move each action selects (its status checks: Disable, Taunt); ``moves``:
    # the move it runs (Nature Power's and the invokers' call, order.called_move,
    # order.used_moves), whose draws these are.
    picked = jnp.stack(
        [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
    )
    # A takeover variant (B11-1): the action runs the encored move (moveTemp) from the start.
    taken = jnp.zeros((fmt.B,), bool) if taken is None else jnp.asarray(taken, bool)
    took_state = encored_state(tables, state, taken)
    chosen = jnp.where(taken, state.last_move.astype(_I32), picked)
    kinds = call_kind(tables, chosen)
    # Copycat's and Me First's calls follow the order (the foe's move before them, or not).
    copycat = uses_move & ((kinds == CALL_COPYCAT) | (kinds == CALL_ME_FIRST))
    me_first_used = uses_move & (kinds == CALL_ME_FIRST)
    if call is None:
        moves = jnp.stack([called_move(tables, state, chosen[side]) for side in range(fmt.B)])
        call = jnp.zeros((fmt.B,), _I32)
        call_pending = jnp.any(uses_move & (kinds != CALL_NONE))
    else:
        call = jnp.asarray(call, _I32)
        # A Sleep Talk variant that assumes a faster foe's Disable lands (CALL_DISABLED, ruling
        # B5-13) calls from the set with the user's last move disabled; one that assumes a faster
        # foe's Imprison (CALL_SEALED, ruling B7-13) from the set without the moves it seals.
        talking = kinds == CALL_SLEEP_TALK
        disabled = talking & (call >= CALL_DISABLED) & (call < CALL_SEALED)
        sealed = talking & (call >= CALL_SEALED) & (call < CALL_GRAVITY)
        # A variant that assumes a faster foe's Gravity lands first (CALL_GRAVITY, G-67) calls
        # from the set Gravity leaves (Sleep Talk, Metronome, Assist).
        grav = (call >= CALL_GRAVITY) & ~taken
        after_foe = copycat & (call == 1)
        disabled = disabled & ~taken
        sealed = sealed & ~taken
        moves = jnp.where(
            disabled,
            used_moves(tables, disabled_state(state, disabled), actions, call, after_foe, chosen),
            jnp.where(
                sealed,
                used_moves(tables, sealed_state(state, sealed), actions, call, after_foe, chosen),
                jnp.where(
                    taken,
                    used_moves(tables, took_state, actions, call, after_foe, chosen),
                    jnp.where(
                        grav,
                        used_moves(tables, gravity_state(state), actions, call, after_foe, chosen),
                        used_moves(tables, state, actions, call, after_foe, chosen),
                    ),
                ),
            ),
        )
        call_pending = jnp.asarray(False)
    foe_moves = foes(uses_move)  # the other side uses a move
    effect, params = move_effects(tables, state, moves)  # Secret Power's terrain effect
    foe_move = foes(moves)
    chance = read_chance(tables, moves)
    damaging = (tables.move_category[moves] != STATUS) & (tables.move_power[moves] != 0)
    # No crit, no roll for a script without CalcCrit / CalcDamage (ids.UNROLLED_EFFECTS:
    # Night Shade, Bide's unleash, Super Fang, the OHKO moves; damage.fixed_damage)
    rolled = damaging & ~_any_of(effect, UNROLLED_EFFECTS)
    contact = (tables.move_flags[moves] & MoveFlag.CONTACT) != 0
    # effects.hit_count: MULTI_HIT's count, or two for a HIT_TWICE move (Twineedle, B13-1).
    twice = ((tables.move_flags[moves].astype(_I32) & MoveFlag.HIT_TWICE) != 0) & (
        effect != _E.MULTI_HIT
    )
    # Triple Kick's 3 (B13-5), Beat Up's eligible members (damage.beat_up_slots, B13-9: nothing
    # changes a benched Pokemon's HP or status before the move, and the user is always one).
    kick = effect == _E.TRIPLE_KICK
    beat_up = effect == _E.BEAT_UP
    members = jnp.stack([beat_up_slots(state, side)[1] for side in range(fmt.B)])
    multi = (effect == _E.MULTI_HIT) | twice | kick | beat_up
    fixed_hits = jnp.where(
        twice, 2, jnp.where(kick, TRIPLE_KICK_HITS, jnp.where(beat_up, members, params[:, 0]))
    )
    drawn_hits = (effect == _E.MULTI_HIT) & (fixed_hits == 0)  # the 2-5 law
    max_hits = jnp.where(multi, jnp.where(drawn_hits, 5, fixed_hits), 1)
    mons = state.active
    status = state.status[party(fmt, sides), mons]

    hit_index = jnp.arange(MAX_HITS)[None, :]
    lands = uses_move[:, None] & (hit_index < max_hits[:, None])  # [2,5]
    # Future Sight / Doom Desire never roll for accuracy at their use (effect_script_0148.s:7,
    # SYSCTL_NONSTANDARD_ACC_CHECK); their set-up draws TryFutureSight's variance, roll[s, 0],
    # when nothing is pending on the target's battler (it cannot change before the move: only
    # this side's own Future Sight sets it, and the count drops at the turn's end).
    future = effect == _E.FUTURE_SIGHT
    future_roll = uses_move & future & (foes(state.future_timer) == 0)
    # Triple Kick's later hits roll their own check (B13-6).
    hit = (
        uses_move[:, None]
        & ((tables.move_accuracy[moves] != 0) & ~future)[:, None]
        & ((hit_index == 0) | (kick[:, None] & (hit_index < max_hits[:, None])))
    )
    crit = lands & rolled[:, None]
    secondary = lands & ((chance != 0) & _any_of(effect, _PROC_EFFECTS))[:, None]
    holders = _holds(
        tables, state, actions, (AbilityId.STATIC, AbilityId.POISON_POINT, AbilityId.FLAME_BODY)
    )
    # G-69: a side whose Transform lands before the foe's contact hit holds the foe's ability then
    # (BtlCmd_Transform copies ``ability``, battle_script.c:5346-5348, battle_mon.h:84), but not its
    # gender (a separate field, battle_mon.h after ``ability``; ``transformedGender`` is the
    # sprite's, :5338). Over-approximate: the Transform may move later or fail.
    xform = uses_move & (effect == _E.TRANSFORM)
    holders = holders | (xform & foes(holders))
    # BtlCmd_TryAttract (effects._attractable): different genders, neither genderless.
    gender = state.gender.astype(_I32)[party(fmt, jnp.arange(fmt.B))]  # each battler's party
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.gender.shape[1] - 1)

    def lovers(a, b):
        return (a != b) & (a != 2) & (b != 2)

    attacker_gender = foes(gender[sides, mons])  # side d's attacker: the other side's active
    # The attacker is not in love at the start, or the holder's side switches: the source
    # leaves first (step._source_leaves) and a Cute Charm switch-in can then infatuate it.
    calm = (foes(state.infatuated) == 0) | switch
    charm_now = (battler_ability(state, sides, mons) == AbilityId.CUTE_CHARM) & lovers(
        gender[sides, mons], attacker_gender
    )
    entering = (battler_ability(state, sides, incoming) == AbilityId.CUTE_CHARM) | (
        _entry_abilities(tables, state, actions) == AbilityId.CUTE_CHARM
    )
    charm_later = switch & entering & lovers(gender[sides, incoming], attacker_gender)
    # A Transform user copies the foe's Cute Charm (G-69) and keeps its own gender. Raw: the
    # copy is BattleMon.ability's (battle_script.c:5342-5348).
    foe_charm = foes(state.ability[party(fmt, sides), mons] == AbilityId.CUTE_CHARM)
    charm_copied = xform & foe_charm & lovers(gender[sides, mons], attacker_gender)
    charm = (charm_now | charm_later | charm_copied) & calm
    if loose:
        charm = _holds(tables, state, actions, (AbilityId.CUTE_CHARM,)) | (xform & foe_charm)
    holders = holders | charm
    contact_proc = foes(lands & (damaging & contact & foes(holders))[:, None]) | (
        lands & (damaging & contact & holders & lock)[:, None]
    )
    # Chatter (G-70): a Chatot that is not Transformed confuses (effects._chatter; over-approximate:
    # the roll may fail, the target may be confused already or behind a Substitute).
    chatot = (state.species[party(fmt, sides), mons] == SPECIES_CHATOT) & (state.transformed == 0)
    chatter_user = uses_move & (effect == _E.CHATTER) & chatot
    duration = uses_move & (
        chatter_user
        | _any_of(
            effect,
            (_E.TAUNT, _E.TRAP, _E.DISABLE, _E.CONFUSE_FOE, _E.SECONDARY_CONFUSE, _E.SWAGGER),
        )
        # Encore's RandNext() % 5 + 3 (battle_script.c:4742): missing until batch 11 (B11-9).
        | (effect == _E.ENCORE)
        | _inflicts(tables, state, moves, Status.SLEEP, users=tuple(range(fmt.B)))
        # Uproar's Random 3, 3 (subscript_uproar.s:6), only for a user not in an uproar yet
        # (effect_script_0159.s:5; only its own move can start one this turn).
        | ((effect == _E.UPROAR) & (state.uproar == 0))
    )

    # Status eligibility of the foe's active Pokemon, the target when the user moves
    # first (a switch or an item would go first): effects._inflict (subscript_paralyze.s,
    # subscript_fall_asleep.s, can_have_status) and effects._secondary_confuse
    # (subscript_confuse.s fails on a confused target). One hit only (an earlier hit's
    # berry could cure the target). Insomnia is never a block: its "stayed awake" reveal
    # (ai_ability_known) runs on a proc; Limber's runs only for a direct status move.
    param0 = params[:, 0]
    attacker_ability = battler_ability(state, sides, mons)
    ignorable = jnp.where(
        attacker_ability == AbilityId.MOLD_BREAKER, AbilityId.NONE, foes(attacker_ability)
    )
    can_take = jnp.stack(
        [
            can_have_status(tables, state, _foe(side), mons[_foe(side)], param0[side])
            for side in range(fmt.B)
        ]
    )
    limber = (param0 == Status.PARALYSIS) & (ignorable == AbilityId.LIMBER)
    insomnia = (param0 == Status.SLEEP) & sleepless(ignorable)  # Vital Spirit too (G-47)
    single = ~multi

    def takes(status) -> jnp.ndarray:
        return jnp.stack(
            [
                can_have_status(tables, state, _foe(side), mons[_foe(side)], status)
                for side in range(fmt.B)
            ]
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
    confused_blocked = single & (foes(state.confusion) > 0)
    # subscript_flinch_mon.s:7: Inner Focus (ignorable) stops Thunder Fang's flinch.
    flinch_blocked = ignorable == AbilityId.INNER_FOCUS
    if loose:
        status_blocked = confused_blocked = flinch_blocked = jnp.zeros((fmt.B,), bool)
    status_blocked, confused_blocked = status_blocked & ~lock, confused_blocked & ~lock
    flinch_blocked = flinch_blocked & ~lock
    confuses = _any_of(effect, (_E.CONFUSE_FOE, _E.SECONDARY_CONFUSE, _E.SWAGGER)) | chatter_user
    # Attract on the foe's active Pokemon (effects._attract); in love at the start, read
    # unless the source switches out first (step._source_leaves).
    foe_calm = foes(state.infatuated) == 0
    attract = uses_move & (effect == _E.ATTRACT)
    infatuated_own = uses_move & (state.infatuated != 0)
    if not loose:
        attract = attract & lovers(gender[sides, mons], foes(gender[sides, mons])) & foe_calm
        infatuated_own = infatuated_own & ~foes(switch)
    # Focus Band (effects.play_hit, step._use_move; G-1): the defender's roll on every hit of a
    # non-status move, the foe's active Pokemon or the one it switches in (no implemented path
    # gives a Focus Band mid-turn); a self-hit's roll is the user's own (_branch_masks).
    band_held = _holds_item(tables, state, actions, (ItemId.FOCUS_BAND,))
    band_move = lands & (
        (tables.move_category[moves] != STATUS) & (foes(band_held) | (band_held & lock))
    )[:, None]
    own_band = tables.item_effect[state.item[party(fmt, sides), mons]] == ItemId.FOCUS_BAND
    # A disliked flavor berry on the side's active Pokemon (hooks.berry_confuse, G-5).
    own_berry = jnp.stack(
        [
            dislikes_berry(
                tables, state, side, mons[side], state.item[party(fmt, side), mons[side]]
            )
            for side in range(fmt.B)
        ]
    )
    calm_now = state.confusion == 0
    if loose:
        calm_now = jnp.ones((fmt.B,), bool)
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
        "chatter": lands[:, 0] & chatter_user,
        "flinch_item": lands & (damaging & kings_move & (kings_rock > 0))[:, None],
        "certain": chance == 0,
        "flinch_blocked": flinch_blocked,
        "self_lock": lock,
        "protect": uses_move & _any_of(effect, (_E.PROTECT, _E.ENDURE)) & (state.protect_count > 0),
        "para_own": uses_move & (status == Status.PARALYSIS),
        "freeze_own": uses_move & (status == Status.FREEZE),
        "confused_own": uses_move & (state.confusion >= 2),
        "future_roll": future_roll,
        "para_inflict": uses_move
        & (
            (
                _inflicts(tables, state, moves, Status.PARALYSIS, users=tuple(range(fmt.B)))
                & ~status_blocked
            )
            # Synchronize (effects.synchronize, G-13): the side's holder, paralyzed by the
            # foe's Static on its contact hit, passes the paralysis to the foe.
            | (
                damaging
                & contact
                & (battler_ability(state, sides, mons) == AbilityId.SYNCHRONIZE)
                & (foes(battler_ability(state, sides, mons)) == AbilityId.STATIC)
            )
        ),
        "freeze_inflict": uses_move
        & _inflicts(tables, state, moves, Status.FREEZE, users=tuple(range(fmt.B)))
        & ~status_blocked,
        "confuse_inflict": uses_move & confuses & ~confused_blocked,
        "infatuated_own": infatuated_own,
        "attract_inflict": attract,
        "focus_band": band_move,
        "own_band": uses_move & own_band,
        "berry_confuse": uses_move & own_berry & calm_now,
        "pending": uses_move & (state.confusion == CONFUSION_PENDING),
    }

    # Order branches: order.action_order under speed_tie = 0 and 1. speed_tie is read
    # only through the order, so it is relevant exactly when the two orders differ.
    tie_outcomes = [empty_outcome(fmt)._replace(speed_tie=jnp.int8(t)) for t in (0, 1)]
    firsts = jnp.stack([action_order(tables, state, actions, o)[0] for o in tie_outcomes])
    firsts = firsts.astype(_I32)
    if loose:
        # order.action_order: priority only between two moves; a tie only between two
        # actions of the same kind.
        priority = jnp.where(
            uses_move[0] & uses_move[1], tables.move_priority[picked].astype(_I32), 0
        )
        same_kind = (switch[0] == switch[1]) & (item[0] == item[1])
        speed_tie = (
            live
            & same_kind
            & (priority[0] == priority[1])
            & (effective_speed(tables, state, 0) == effective_speed(tables, state, 1))
        )
        anyway = jnp.ones((fmt.B,), bool)
        branches = [_branch_masks(anyway, anyway, base)] * 2
    else:
        speed_tie = live & (firsts[0] != firsts[1])
        branches = [
            _branch_masks(firsts[t] == _foe(sides), firsts[t] == sides, base) for t in (0, 1)
        ]
    # Copycat's call (order.used_moves): variant 1 needs its foe to use a move before it in the
    # branch, variant 0 the opposite.
    after_foe = jnp.stack([(firsts == _foe(side)) & uses_move[_foe(side)] for side in range(fmt.B)])
    branch_ok = jnp.all(~copycat[:, None] | (after_foe == (call == 1)[:, None]), axis=0)

    own_speed = (_any_of(effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF)) & (param0 == Stat.SPE)) | (
        _any_of(effect, (_E.RAISE_ALL_STATS_HIT, _E.CURSE, _E.DRAGON_DANCE))
    )
    # E2 batch 12: Haze resets, Psych Up copies and Acupressure may raise the user's Speed; Haze
    # resets the foe's too.
    own_speed = own_speed | _any_of(effect, (_E.HAZE, _E.PSYCH_UP, _E.ACUPRESSURE))
    foe_speed = _any_of(effect, (_E.STAT_FOE, _E.SECONDARY_STAT_FOE)) & (param0 == Stat.SPE)
    foe_speed = foe_speed | (effect == _E.HAZE)
    # subscript_flinched.s:10-14: a flinched Steadfast Pokemon gains +1 Speed (its foe's
    # flinch move, or its King's Rock on a flagged damaging move).
    steadfast = _holds(tables, state, actions, (AbilityId.STEADFAST,)) & foe_moves
    kings = (kings_rock > 0) & damaging & kings_move
    flinches = _any_of(foes(effect), _FLINCH_EFFECTS) | foes(kings)
    steadfast = steadfast & flinches
    # A White Herb restores a lowered Speed stage (hooks.held_item_check); a Salac Berry
    # raises it, held or plucked (hooks.pluck_berry: any Pluck / Bug Bite user, over-approx).
    herb = _holds_item(tables, state, actions, (ItemId.WHITE_HERB,))
    pinch_held = _holds_item(tables, state, actions, (ItemId.PINCH_BERRY, ItemId.STARF_BERRY))
    plucks = uses_move & (effect == _E.EAT_BERRY)
    # B13-19: a Recycle gives its user back an item the move end's item check may use (a Salac
    # Berry, a White Herb, a Lum or Cheri Berry): free like a held one.
    recycles = uses_move & (effect == _E.RECYCLE)
    plucks = plucks | recycles
    stage_free = (
        (uses_move & own_speed)
        | (foe_moves & foes(foe_speed))
        | steadfast
        | herb
        | pinch_held
        | plucks
    )
    static = _holds(tables, state, actions, (AbilityId.STATIC,))
    sync_foe = foes(_holds(tables, state, actions, (AbilityId.SYNCHRONIZE,)))
    # A Lum or Cheri Berry cures paralysis, held or plucked (hooks.held_item_check,
    # pluck_berry).
    lum = _holds_item(
        tables, state, actions, (ItemId.LUM_BERRY, ItemId.CHERI_BERRY, ItemId.LAVA_COOKIE)
    )
    para_free = (
        (
            foe_moves
            & _inflicts(
                tables,
                state,
                foe_move,
                Status.PARALYSIS,
                users=tuple(_foe(b) for b in range(fmt.B)),
            )
        )
        # B12-12..17: Rest, Refresh, Heal Bell and Psycho Shift cure their user's paralysis.
        | (uses_move & _any_of(effect, (_E.REST, _E.REFRESH, _E.HEAL_BELL, _E.PSYCHO_SHIFT)))
        | (uses_move & damaging & contact & foes(static))
        | lum
        | plucks
        # The foe's SmellingSalt cures this side's paralysis (effect_script_0171.s:12,
        # subscript_heal_target_paralysis.s:5-6; effects._cure_paralysis) before the
        # end-of-turn sort (battle_controller_player.c:888).
        | (foe_moves & (foes(effect) == _E.DOUBLE_POWER) & (foes(params)[:, 1] == 1))
        # Quick Feet reads any status (order.effective_speed): free for a holder.
        | _holds(tables, state, actions, (AbilityId.QUICK_FEET,))
        # A paralyzed Limber holder is cured at a move's end or a switch-in
        # (hooks.recover_status_by_ability, battle_lib.c:4397): the hole a generator state with a
        # paralyzed Limber foe exposed (E1 Gardenia batch).
        | _holds(tables, state, actions, (AbilityId.LIMBER,))
        # Synchronize (effects.synchronize, G-13): a foe holder passes back the paralysis the
        # side's move gave it, or passes the paralysis the side's Static gave it.
        | (uses_move & _inflicts(tables, state, moves, Status.PARALYSIS) & sync_foe)
        | (foe_moves & foes(damaging) & foes(contact) & sync_foe & static)
    )
    # Roar / Whirlwind (step._use_move): BtlCmd_TryWhirlwind draws nothing with no eligible
    # bench Pokemon, BattleSystem_CanWhirlwind nothing for a user of at least the target's
    # level (outcome.roar_inputs, roar_levels), and the pick is forced with one.
    roar = uses_move & (effect == _E.ROAR)
    roar_weight, roar_bench = roar_inputs(state, actions)
    user_level, target_level = roar_levels(state, actions)
    # An anchored target (subscript_force_target_to_switch_or_flee.s:7, effects.ingrain_anchors)
    # stops Roar before TryWhirlwind draws (B7-16): anchored at the turn start and still so when
    # Roar runs, i.e. the target side neither switches nor uses a move that can add a bit of its
    # moveEffectsMask (Aqua Ring, the sports, Imprison, Camouflage, or an invoker's call).
    setters = (
        _E.AQUA_RING,
        _E.MAGNET_RISE,
        _E.MUD_SPORT,
        _E.WATER_SPORT,
        _E.IMPRISON,
        _E.CAMOUFLAGE,
    )
    target_steady = ~switch & ~(uses_move & (_any_of(effect, setters) | (kinds != CALL_NONE)))
    anchored = jnp.stack([ingrain_anchors(tables, state, _foe(side)) for side in range(fmt.B)])
    roar = roar & ~(anchored & foes(target_steady))
    roar_level = roar & (user_level < target_level) & (roar_bench >= 1)
    roar_target = roar & (roar_bench >= 2)
    # The foe can drag this side's Pokemon out: its Roar with an eligible bench Pokemon here
    # (BtlCmd_TryWhirlwind fails at eligibleMons <= 1, battle_script.c:5285-5297, before
    # any drag).
    dragged = foes(roar & (roar_bench >= 1))
    if loose:
        # The brute force: both draws whenever the side uses Roar, whatever the levels and the
        # bench (the pick's law is then over max(bench, 1) values), and every living party
        # member's speed on both sides whenever anyone uses it.
        roar_level = roar_target = roar
        dragged = jnp.broadcast_to(jnp.any(roar), (fmt.B,))
    # The weather lock of the turn's end (NO_CLOUD_NINE, include/battle/common.h:63-64)
    # counts the battlers on the field with Cloud Nine / Air Lock and HP left
    # (COUNT_ALIVE_BATTLERS, battle_lib.c:2951-2957). A living holder anywhere in either
    # party can change it this turn: by a switch in or out, a Roar's drag in or out, or a
    # faint (over-approximate; nothing in the engine revives, so a fainted holder never
    # counts again). Raw over the whole party: a superset of every Battler_Ability it can show.
    abilities = decomp_abilities(tables, state)
    lock_holder = (abilities == WEATHER_ABILITY_IDS["Cloud Nine"]) | (
        abilities == WEATHER_ABILITY_IDS["Air Lock"]
    )
    lock_free = jnp.any(lock_holder & (state.species != 0) & (state.hp > 0))
    # The weather can change before the turn's end: a weather move (effects._weather), or a
    # newcomer with a weather ability or Trace (which can copy one; hooks.on_switch_in): the one
    # a side switches in, or any living Pokemon of a side a foe's Roar can drag.
    setters = (
        AbilityId.DRIZZLE,
        AbilityId.SAND_STREAM,
        AbilityId.DROUGHT,
        AbilityId.SNOW_WARNING,
        AbilityId.TRACE,
    )
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.ability.shape[1] - 1)
    switch_setter = switch & _any_of(battler_ability(state, sides, incoming), setters)
    # Raw over the whole party: a superset of every Battler_Ability it can show.
    alive_setter = jnp.any(
        _any_of(state.ability, setters) & (state.species != 0) & (state.hp > 0), axis=1
    )[party(fmt, jnp.arange(fmt.B))]
    weather_free = live & jnp.any(
        (uses_move & (effect == _E.WEATHER)) | switch_setter | (dragged & alive_setter)
    )
    # F-16: a Sky Shaymin of the side, frozen now or by a foe's move, that a foe's move hits
    # becomes Land Forme mid-turn (hooks.shaymin_land): its end speed may be Land's.
    sky_mon = (
        (state.species == SPECIES_SHAYMIN)
        & (state.form != 0)
        & (state.species != 0)
        & (state.hp > 0)
    )[party(fmt, jnp.arange(fmt.B))]
    frozen_sky = jnp.any(
        sky_mon & (state.status[party(fmt, jnp.arange(fmt.B))] == Status.FREEZE), axis=1
    )
    freezes = _inflicts(
        tables, state, foes(moves), Status.FREEZE, users=tuple(_foe(b) for b in range(fmt.B))
    )
    land_free = live & foes(uses_move) & ((jnp.any(sky_mon, axis=1) & freezes) | frozen_sky)
    if loose:
        land_free = live & jnp.any(sky_mon, axis=1)
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
            weather_free,
            # The side's own Tailwind when none blows yet (effects._side_layers).
            uses_move[side] & (effect[side] == _E.TAILWIND) & (state.tailwind[side_of(side)] == 0),
            # A living Unburden holder of the side (over-approximate: it may lose its item; raw
            # over the party, a superset of every Battler_Ability it can show).
            live
            & jnp.any(
                (state.ability[party(fmt, side)] == AbilityId.UNBURDEN)
                & (state.species[party(fmt, side)] != 0)
                & (state.hp[party(fmt, side)] > 0)
            ),
            land_free[side],
        )
        for side in range(fmt.B)
    ]
    (speed0, ok0), (speed1, ok1) = speeds
    # B13-31: a foe's Embargo can hide a side's Choice Scarf (CompareBattlerSpeed's
    # Battler_HeldItemEffect, battle_lib.c:1248) before the turn end's sort, a speed the grid
    # does not follow: the tie draw is then always enumerated (over-approximate, sound).
    scarf = tables.item_effect[held_items(state)] == ItemId.CHOICE_SCARF
    embargo_speed = jnp.any(scarf & foes(uses_move & (effect == _E.EMBARGO)))
    end_speed_tie = live & (
        jnp.any(ok0[:, None] & ok1[None, :] & (speed0[:, None] == speed1[None, :])) | embargo_speed
    )
    # A Trace holder on the field at the turn's end (the active or incoming one, or one a drag
    # brings in) may have copied a speed ability (Swift Swim, Quick Feet) the grid does not
    # follow, or Shed Skin: both reads are kept (over-approximate).
    living_now = (state.species != 0) & (state.hp > 0)
    # Raw over the whole party: a superset of every Battler_Ability it can show.
    trace_party = jnp.any(living_now & (state.ability == AbilityId.TRACE), axis=1)[
        party(fmt, jnp.arange(fmt.B))
    ]
    tracer = _holds(tables, state, actions, (AbilityId.TRACE,)) | (dragged & trace_party)
    # G-69: a Transform copies its target's Speed, Speed stage and ability (BtlCmd_Transform,
    # battle_script.c:5346-5348; battle_mon.h:57-84): the two end speeds are then often equal, by
    # stats or by a copied speed ability or weather setter the grid does not follow. The tie is
    # kept (over-approximate).
    xform_used = uses_move & (effect == _E.TRANSFORM)
    # E3-2-2: a Worry Seed takes its target's speed ability (Swift Swim, Chlorophyll, Quick Feet,
    # Unburden, ...), which the grid does not follow either: the tie is kept.
    worry_used = uses_move & (effect == _E.WORRY_SEED)
    # E3-2-5: a Skill Swap moves a speed ability (and Shed Skin) between the battlers likewise.
    swap_used = uses_move & (effect == _E.SKILL_SWAP)
    # E3-2-6: a Gastro Acid suppresses its target's speed ability (Swift Swim, Chlorophyll, Quick
    # Feet, Unburden) and Shed Skin likewise (Battler_Ability reads NONE, battle_lib.c:3088-3093).
    gastro_used = uses_move & (effect == _E.GASTRO_ACID)
    end_speed_tie = end_speed_tie | (
        live & jnp.any(tracer | xform_used | worry_used | swap_used | gastro_used)
    )
    if loose:
        # The brute force: the end-of-turn tie roll whenever the battle goes on, with no
        # speed gate (one binary element: at most x2).
        end_speed_tie = live
    # Shed Skin of the Pokemon on the field at the turn's end: the active or incoming one, or
    # any living one a foe's Roar can drag in.
    living = (state.species != 0) & (state.hp > 0)
    # Raw over the whole party: a superset of every Battler_Ability it can show.
    shed_party = jnp.any(living & (state.ability == AbilityId.SHED_SKIN), axis=1)[
        party(fmt, jnp.arange(fmt.B))
    ]
    # A Transform user copies its target's Shed Skin (G-69).
    shed_holder = _holds(tables, state, actions, (AbilityId.SHED_SKIN,))
    shed_skin = live & (
        shed_holder
        | (dragged & shed_party)
        | tracer
        | (xform_used & foes(shed_holder))
        | (jnp.any(swap_used) & foes(shed_holder))  # E3-2-5: Skill Swap hands it over
    )
    if loose:
        shed_skin = shed_skin | (live & jnp.any(roar) & jnp.any(shed_party))

    # Quick Claw (step.qc_kept, end_of_turn_order): a living holder anywhere in the party.
    claw = tables.item_effect[state.item] == ItemId.QUICK_CLAW
    quick_claw = (
        live
        & jnp.any(claw & (state.species != 0) & (state.hp > 0), axis=1)[
            party(fmt, jnp.arange(fmt.B))
        ]
    )
    # Acupressure (B12-9): BtlCmd_BoostRandomStatBy2 reads the count of its defender's stats
    # below +6 at the move (battle_script.c:8943-8950; effects.defender_side, the user in
    # singles). Those stages can change before then only by the foe's action (a move, or a
    # newcomer's Intimidate; over-approximate): the count is then loose. A count of 1 with no
    # change possible draws nothing that matters.
    acu_count = jnp.stack(
        [
            jnp.sum(
                acupressure_valid(
                    state,
                    defender_side(
                        target,
                        targets.resolve(
                            tables,
                            state,
                            side,
                            moves[side],
                            targets.fight_randomizes(state, side, False),
                            state.chosen_target[side],
                            jnp.arange(fmt.B),
                            0,
                        ).defender,
                    ),
                ).astype(_I32)
            )
            for side, target in ((b, _foe(b)) for b in range(fmt.B))
        ]
    )
    acu_loose = foes(live & (uses_move | switch))
    acupressure = uses_move & (effect == _E.ACUPRESSURE) & (acu_loose | (acu_count >= 2))
    if loose:
        acupressure = uses_move & (effect == _E.ACUPRESSURE)
        acu_loose = jnp.ones_like(acu_loose)
    if loose:
        # The brute force: both sides' draws whenever a Quick Claw is anywhere in the battle
        # (either party, any HP), holder or not. With no Quick Claw at all nothing can read
        # speedRand (battle_lib.c:1278 tests the item first), and drawing it there anyway
        # would only multiply every other brute force past the width cap.
        quick_claw = jnp.broadcast_to(live & jnp.any(claw & (state.species != 0)), (fmt.B,))

    starf_draw, starf_loose = _starf_reads(
        tables, state, actions, uses_move, effect, params, dragged, loose
    )

    band_party = jnp.any(
        (tables.item_effect[state.item] == ItemId.FOCUS_BAND)
        & (state.species != 0)
        & (state.hp > 0),
        axis=1,
    )[party(fmt, jnp.arange(fmt.B))]

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
            call=jnp.zeros((fmt.B,), bool),  # stamped per call variant (enumerate_turn), not drawn
            # The Future Sight hit of this turn's end on side d's battler: its count is 1 now
            # (step._future_sight; read when the battler has HP then, over-approximate).
            future_hit=live & (state.future_timer == 1),
            # Yawn's sleep of this turn's end: a count of 1 now reaches 0 there (a Yawn of this
            # turn sets 2; over-approximate: the sleep may still be blocked).
            yawn_sleep=live & (state.yawn == 1),
            # B12-9: the side uses Acupressure (its draw decides only with two or more stats
            # below +6 at the move, or a count that can change first).
            acupressure=acupressure,
            focus_band=m["focus_band"],
            # The Future Sight hit's Focus Band roll: future_hit's gate, a living holder in the
            # party (the battler on the field at the turn's end), G-1.
            future_band=live & (state.future_timer == 1) & band_party,
            berry_confusion=m["berry_confusion"],
            chatter=m["chatter"],
            starf=starf_draw,
            # D2-4: a fight command's BattleSystem_Defender draws one bit at B = 4 (over-
            # approximate; D6-D7 tighten it with the resolver's R4 table).
            target_draw=uses_move & (fmt.B == 4),
            # D2-13 / D2-14 / D2-16: an invoker's called-move defender, a Bide release's
            # RandomOpponent and ChooseTraceTarget's coin, B = 4 only (over-approximate: any Trace
            # holder in the battle reads all its fires; D6-D7 tighten).
            call_target_draw=uses_move & (kinds != CALL_NONE) & (fmt.B == 4),
            bide_target_draw=uses_move & (state.bide_turns == 1) & (fmt.B == 4),
            trace_pick=jnp.broadcast_to(
                (jnp.any(state.ability == AbilityId.TRACE) & (fmt.B == 4))[None, None],
                (fmt.B, TRACE_FIRES),
            ),
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
    loafing = jnp.stack([truant_loafs(state, side) for side in range(fmt.B)])
    reach = (
        uses_move
        & ~loafing  # CHECK_STATUS_STATE_TRUANT
        & (state.hp[party(fmt, sides), mons] > 0)
        & clean
        & (state.confusion == 0)
        & (state.flinched == 0)
        & ~((state.disabled_move != 0) & (state.disabled_move.astype(_I32) == chosen))
        & ~((state.taunt > 0) & (tables.move_power[chosen] == 0))
        & (state.infatuated == 0)
        & (state.bide_turns == 0)
    )
    # A crash move's classes key on the damage before Wonder Guard's zero: a guarded Jump Kick
    # deals nothing but crashes for half the chart's damage (step._use_move, ruling G-19), a
    # refinement of the hit's own classes (it deals 0 at every roll).
    # The Metronome item's count is the one the side's own BeforeMove leaves (UpdateMetronomeCount,
    # bcp.c:3221, before every hit of the move; effects.metronome_update), G-60.
    counted = [metronome_update(tables, state, side, chosen[side], True) for side in range(fmt.B)]
    damage = jnp.stack(
        [
            hit_damage_table(
                tables,
                counted[side],
                side,
                moves[side],
                effect[side] != _E.CRASH,
            )
            for side in range(fmt.B)
        ]
    )
    deals = jnp.all(damage > 0, axis=(1, 2))
    # The per-hit tables (B13-5, B13-10): Triple Kick's hits 2-3 at 2x and 3x the step, Beat Up's
    # members; every other hit reads hit 0's table. The Metronome item's count is the one above
    # for every hit (G-60); Beat Up runs no CalcDamage, so no Metronome boost (B13-10).
    kicks = [
        jnp.stack(
            [
                hit_damage_table(tables, counted[side], side, moves[side], hit=k)
                for side in range(fmt.B)
            ]
        )
        for k in range(1, TRIPLE_KICK_HITS)
    ]
    rolls_ = jnp.arange(ROLLS, dtype=_I32)
    per_hit = []
    for k in range(MAX_HITS):
        beat = jnp.stack(
            [
                jnp.stack(
                    [
                        beat_up_damage(
                            tables, state, side, _foe(side), moves[side], k, crit, rolls_
                        )
                        for crit in (False, True)
                    ]
                )
                for side in range(fmt.B)
            ]
        )
        table = damage
        if 1 <= k < TRIPLE_KICK_HITS:
            table = jnp.where(kick[:, None, None], kicks[k - 1], table)
        per_hit.append(jnp.where(beat_up[:, None, None], beat, table))
    damage_hits = jnp.stack(per_hit, axis=1)
    fake_out_ok = (effect == _E.FAKE_OUT) & (
        state.fake_out_turn.astype(_I32) == state.turn.astype(_I32)
    )
    fang = effect == _E.FANG_PARALYZE_FLINCH
    # Dry Skin and Water Absorb take a Water move whole, Volt Absorb an Electric one
    # (step._use_move: no hit, no flinch).
    kinds = jnp.stack([move_type_of(tables, state, side, moves[side]) for side in range(fmt.B)])
    absorbs = (
        ((ignorable == AbilityId.DRY_SKIN) | (ignorable == AbilityId.WATER_ABSORB))
        & (kinds == Type.WATER)
    ) | ((ignorable == AbilityId.VOLT_ABSORB) & (kinds == Type.ELECTRIC))
    # Flash Fire takes a Fire move too, unless its holder is frozen (F-2; a Fire flinch move has power).
    absorbs = absorbs | (
        (ignorable == AbilityId.FLASH_FIRE) & (kinds == Type.FIRE) & (foes(status) != Status.FREEZE)
    )
    flinch_sure = (
        # Snore (FLINCH with SNORE_PARAM) fails unless its user is asleep when it acts
        # (effect_script_0092.s:5; step's ``failed``), which a wake-up this turn can undo: its
        # flinch is never sure (B10, found by test_matches_sampled_frequencies_random).
        (((effect == _E.FLINCH) & (params[:, 2] != SNORE_PARAM)) | fake_out_ok | fang)
        & reach
        & deals
        & (ignorable != AbilityId.INNER_FOCUS)
        & (ignorable != AbilityId.SHIELD_DUST)  # effects._shield_dust
        # subscript_flinch_mon.s:6: a foe behind a substitute is never flinched (its bit, or the
        # flag this single hit sets), B10-4.
        & (foes(state.substitute_hp) == 0)
        & ~absorbs
        & foe_moves
        & (chance != 0)
        & single
        & ~lock  # a lock aimed at its user flinches the user, not the foe
    )
    proc_effects = (
        _E.SECONDARY_STATUS,
        _E.RECOIL_STATUS,
        _E.SECONDARY_CONFUSE,
        _E.TRAP,
        _E.FANG_PARALYZE_FLINCH,
        _E.TRI_ATTACK,
    )
    off = jnp.zeros((fmt.B,), bool)
    reads = Reads(
        first=firsts,
        hit_gated=off if loose else uses_move & (effect != _E.PURSUIT),
        proc_gated=off if loose else uses_move & (chance != 0) & _any_of(effect, proc_effects),
        flinch_sure=off if loose else flinch_sure,
        fang=fang,
        present=off if loose else uses_move & (effect == _E.PRESENT),
        chatter=off if loose else chatter_user,
        tri_attack=off if loose else uses_move & (effect == _E.TRI_ATTACK) & (chance != 0),
        # Only Protect's success blocks the foe's move (Endure's does not: its hit still lands).
        protect_sure=off if loose else reach & (effect == _E.PROTECT),
        protect_blocks=off
        if loose
        else uses_move & protect_blocks(tables, moves, False) & ~lock,
        self_lock=lock,
        endure_sure=off if loose else reach & (effect == _E.ENDURE),
        roar=off if loose else roar,
        confused_own=base["confused_own"],
        confused_by_foe=jnp.stack([m["confused_by_foe"] for m in branches]),
        para_own=base["para_own"],
        para_by_foe=jnp.stack([m["para_by_foe"] for m in branches]),
        freeze_own=base["freeze_own"],
        freeze_by_foe=jnp.stack([m["freeze_by_foe"] for m in branches]),
        infatuated_own=base["infatuated_own"],
        infatuated_by_foe=jnp.stack([m["infatuated_by_foe"] for m in branches]),
        future_roll=future_roll,
        # B10-2: a miss or a Protect still reads hit 0's crit and roll (step._use_move's crash).
        crash=off if loose else uses_move & (effect == _E.CRASH),
        band_move=base["focus_band"],
        band_self=base["own_band"],
        berry_by_foe=jnp.stack([m["berry_by_foe"] for m in branches]),
        pending_own=base["pending"],
        present_guard=off
        if loose
        else uses_move
        & (effect == _E.PRESENT)
        & (
            # Raw over the foe's party: a superset of every Battler_Ability it can show.
            jnp.any(
                state.ability[party(fmt, _foe(jnp.arange(fmt.B)))] == AbilityId.WONDER_GUARD, axis=1
            )
            # A foe whose Transform copies the side's own Wonder Guard (G-69): the raw
            # BattleMon.ability (battle_script.c:5342-5348).
            | (foes(xform) & (state.ability[party(fmt, sides), mons] == AbilityId.WONDER_GUARD))
        ),
    )
    kind = jnp.where(
        effect == _E.TAUNT,
        DURATION_TAUNT,
        jnp.where(
            effect == _E.ENCORE,
            DURATION_ENCORE,
            jnp.where(
                _any_of(effect, (_E.TRAP, _E.DISABLE, _E.UPROAR)), DURATION_TRAP, DURATION_DEFAULT
            ),
        ),
    )
    # Damage classes (rules §10.3 item 5): a hit whose damage the turn-start table gives
    # whatever happened before it (damage_inputs_can_change), of a move whose table is
    # its damage (TABLE_DAMAGE_EFFECTS, an allowlist), and whose roll is not also a
    # confusion self-hit's.
    # A Me First user's hit carries the x1.5 its own move sets (``damage.me_first_boost``): the
    # turn-start table does not hold it (E3-2-3).
    tabled = rolled & _any_of(effect, TABLE_DAMAGE_EFFECTS) & ~me_first_used
    # The crit odds at the hit (outcome.crit_variants): a hit whose odds can change before its
    # CalcCrit draws its crit as a class of the CRIT_DRAWS values (_crit), not in damage classes.
    odds_variants, odds_possible = crit_variants(tables, state, actions, moves)
    if loose:
        odds_possible = jnp.ones_like(odds_possible)
    crit_moves = jnp.any(odds_possible & (odds_variants != odds_variants[:, :1]), axis=1)
    compress = jnp.stack(
        [
            crit
            & tabled[:, None]
            & ~crit_moves[:, None]
            & ~damage_inputs_can_change(tables, state, actions, firsts[t] == _foe(sides), moves)
            & ~(m["confusion_hit"][:, None] & (hit_index == 0))
            & ~lock[:, None]  # the table is the foe's; a lock may read the user's own
            for t, m in enumerate(branches)
        ]
    )
    if loose:
        compress = jnp.zeros_like(compress)
    thresholds, possible, later_thr = hit_threshold_pair(tables, state, actions, moves)
    later_ok = possible
    future_thr, future_ok, future_loose = future_thresholds(tables, state, actions, moves)
    if loose:
        # The brute force: the hit law cut at every threshold variant, possible or not
        # (an extra cut only splits a class), and the Future Sight roll whole.
        possible = jnp.ones_like(possible)
        later_ok = jnp.ones_like(later_ok)
        future_loose = jnp.ones_like(future_loose)
    # The paused action's King's Rock (E2 batch 6): its own read, in both branches.
    rock = (jnp.arange(MAX_HITS) == 0)[None, :] & (paused_rock & (kings_rock > 0))[:, None]
    relevant = relevant._replace(flinch_item=relevant.flinch_item | rock)
    per_branch = per_branch._replace(flinch_item=per_branch.flinch_item | rock[None])
    return OutcomeLaws(
        hit_thresholds=thresholds,
        hit_possible=possible,
        hit_later_thresholds=later_thr,
        hit_later_possible=later_ok,
        hit_whole=lock,
        crit_variants=odds_variants,
        crit_possible=odds_possible,
        # Lucky Chant, or the target's Battle Armor / Shell Armor (outcome.armored); not for a
        # lock aimed at its user, whose step reads both on the hit's defender (effects.crit_blocked).
        no_crit=(
            (state.lucky_chant[side_of(_foe(jnp.arange(fmt.B)))] != 0)
            | jnp.where(
                pursuit_chase(tables, state, actions)[0],
                pursuit_armored(tables, state),
                armored(tables, state, actions, moves),
            )
        )
        & ~lock,
        secondary_chance=secondary_chance(tables, state, moves),
        duration_kind=kind.astype(_I32),
        relevant=relevant,
        branch_relevant=per_branch,
        reads=Reads(*(jnp.asarray(f) for f in reads)),
        damage=damage_hits.astype(_I32),
        compress=compress,
        protect_count=state.protect_count.astype(_I32),
        roar_level_weight=roar_weight,
        roar_bench=roar_bench,
        flinch_item_chance=kings_rock,
        call=call,
        branch_ok=branch_ok,
        call_pending=call_pending,
        future_thresholds=future_thr,
        future_possible=future_ok,
        future_loose=future_loose,
        acupressure_count=acu_count,
        acupressure_loose=acu_loose,
        chatter_chance=jnp.asarray(CHATTER_CHANCES, _I32)[
            jnp.clip(state.chatot_cry[party(fmt, jnp.arange(fmt.B))].astype(_I32), 0, 2)
        ],
        starf_loose=starf_loose,
    )


# Effects whose stage writes are not bounded by a move's own stage param (``_starf_reads``): they
# set, copy or swap whole stages, pass them on, or give a Starf Berry back for a second eating.
_STARF_STAGE_JUMPS = (
    _E.BELLY_DRUM,
    _E.PSYCH_UP,
    _E.STAGE_SWAP,
    _E.TRANSFORM,
    _E.BATON_PASS,
    _E.RECYCLE,
)


def _starf_reads(tables, state, actions, uses_move, effect, params, dragged, loose):
    """``(relevant bool[B, 2], loose bool[B, 2])`` of ``Outcome.starf`` (F-10). Index 0, the turn's
    draw, is read where side ``d`` can eat a Starf Berry this turn: its active Pokemon or the one
    it switches in holds one, or any living holder of its party can enter (its own U-turn or Baton
    Pass, a foe's Roar), and that holder is at its pinch or can lose HP this turn (any move, a
    residual damage source, entry hazards; over-approximate); or ``d`` Plucks / Bug Bites a foe
    holding one (no HP test, battle_lib.c:5798-5817), or Recycles one; or a battle-start berry is
    still open (``starf_open``). Index 1, the faint replacement's draw, where a living holder of
    the party can be at its pinch when a replacement sends it in: at its pinch now, on the field
    this turn when it can leave, or with Spikes / Stealth Rock up or set this turn
    (``step.starf_kept``).

    The law: the draw is read modulo the count of the five stats below +6 at the eating. A fresh
    newcomer has all five below +6; an active one keeps that count when its highest of them plus
    the most this turn can add stays below +6: the side's own stat move (its stage param, 1 for
    Dragon Dance, AncientPower, Curse, 2 for Acupressure), a foe's Swagger / Flatter or raising
    stat move (its param) and 2 for the rest (Steadfast, Speed Boost, a plucked pinch berry,
    Download); no Belly Drum, Psych Up, Power / Guard Swap, Transform, Baton Pass or Recycle
    (``_STARF_STAGE_JUMPS``). Then 5 values; else ``loose``, all ``STARF_DRAWS``. The replacement's
    draw is always 5 values: a fresh battler (``step.starf_kept`` stores it modulo 5)."""
    fmt = format_of(state)
    sides = jnp.arange(fmt.B)
    rows = party(fmt, sides)
    live = state.over == 0
    switch = is_switch(actions)
    holds, pinched = starf_holders(tables, state)
    holds_b, pinched_b = holds[rows], pinched[rows]  # [B, SLOTS]
    slots = jnp.arange(state.hp.shape[1])[None, :]
    incoming = jnp.clip(actions - MOVE_ACTIONS, 0, state.hp.shape[1] - 1)
    on_field = (slots == state.active[:, None]) | (switch[:, None] & (slots == incoming[:, None]))
    pivots = uses_move & _any_of(effect, (_E.U_TURN, _E.BATON_PASS))
    enters = pivots | jnp.asarray(dragged, bool)
    candidates = (holds_b & on_field) | (enters[:, None] & holds_b)
    recycled = (
        uses_move
        & (effect == _E.RECYCLE)
        & (tables.item_effect[state.recycle_item] == ItemId.STARF_BERRY)
    )
    plucked = uses_move & (effect == _E.EAT_BERRY) & foes(jnp.any(holds_b & on_field, axis=1))
    hazards = entry_damage_up(state)
    residual = (
        (state.weather != Weather.NONE)
        | jnp.any((state.status != Status.NONE) & (state.hp > 0))
        | jnp.any(state.leech_seed != 0)
        | jnp.any(state.curse != 0)
        | jnp.any(state.trapped != 0)
        | jnp.any(state.future_timer != 0)
        | jnp.any(hazards)
    )
    hp_drop = jnp.any(uses_move) | residual
    pinch_reach = jnp.any(candidates & (pinched_b | hp_drop), axis=1)
    open_ = state.starf_open != 0
    turn = live & (pinch_reach | plucked | recycled | open_)
    # Index 1: a holder at its pinch when a replacement sends it in: one off the field, or the
    # active one if it can leave (a switch, its U-turn or Baton Pass, a foe's Roar), at its pinch
    # now, or anywhere in the party when it can lose HP first (it leaves after a hit, a hazard).
    setters = foes(uses_move & _any_of(effect, (_E.SPIKES, _E.STEALTH_ROCK)))
    leaving = switch | enters
    bench = (slots != state.active[:, None]) | leaving[:, None]
    hurt = (hazards | setters | leaving)[:, None]
    rep = live & jnp.any(bench & (pinched_b | (holds_b & hurt)), axis=1)
    # The count: fixed at 5 unless a stat can reach +6 before the eating.
    five = jnp.asarray([int(x) for x in STARF_STATS], _I32)
    highest = jnp.maximum(jnp.max(state.stages[:, five].astype(_I32), axis=1), 0)
    param1 = jnp.maximum(params[:, 1].astype(_I32), 0)
    own_gain = jnp.where(
        _any_of(effect, (_E.STAT_SELF, _E.SECONDARY_STAT_SELF)),
        param1,
        jnp.where(
            effect == _E.ACUPRESSURE,
            2,
            _any_of(effect, (_E.DRAGON_DANCE, _E.RAISE_ALL_STATS_HIT, _E.CURSE)).astype(_I32),
        ),
    )
    foe_gain = jnp.where(
        _any_of(effect, (_E.SWAGGER, _E.STAT_FOE, _E.SECONDARY_STAT_FOE)), param1, 0
    )
    gain = jnp.where(uses_move, own_gain, 0) + foes(jnp.where(uses_move, foe_gain, 0)) + 2
    jump = jnp.any(uses_move & _any_of(effect, _STARF_STAGE_JUMPS))
    turn_loose = jump | (highest + gain >= STAGE_MAX) | open_
    relevant = jnp.stack([turn, rep], axis=1)
    loose_out = jnp.stack([turn_loose, jnp.zeros_like(turn_loose)], axis=1)
    if loose:
        # The brute force: the turn's draw, all STARF_DRAWS values, wherever a living holder is
        # in the side's party (or its foe's, for a Pluck / Bug Bite) or a berry is open, and the
        # replacement's (five values: a fresh battler's count) wherever one is in the party.
        anyone = jnp.any(holds_b, axis=1)
        eats = uses_move & (effect == _E.EAT_BERRY) & foes(anyone)
        relevant = jnp.stack([live & (anyone | eats | open_ | recycled), live & anyone], axis=1)
        loose_out = jnp.stack([jnp.ones_like(live & anyone), jnp.zeros_like(anyone)], axis=1)
    return relevant, loose_out


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


def _hit(laws: OutcomeLaws, side: int, hit: int = 0) -> _Law:
    """The accuracy roll's class law: hit 0's over ``hit_thresholds``, a later Triple Kick
    hit's over ``hit_later_thresholds`` (B13-6). A lock that may be aimed at its user keeps every
    roll whole (``hit_whole``, outcome.self_lock)."""
    if bool(laws.hit_whole[side]):
        return list(range(1, 101)), [1] * 100, 100
    if hit == 0:
        possible = np.asarray(laws.hit_possible[side], bool)
        return hit_class_law(np.asarray(laws.hit_thresholds[side])[possible])
    possible = np.asarray(laws.hit_later_possible[side], bool)
    return hit_class_law(np.asarray(laws.hit_later_thresholds[side])[possible])


def _future_hit(laws: OutcomeLaws, side: int) -> _Law:
    """The Future Sight hit's CheckMoveHit roll (bcp.c:2976) on side ``side``'s battler: its
    class over ``outcome.future_thresholds``, or every roll 1-100 where ``future_loose``."""
    if bool(laws.future_loose[side]):
        return list(range(1, 101)), [1] * 100, 100
    possible = np.asarray(laws.future_possible[side], bool)
    thresholds = np.asarray(laws.future_thresholds[side])[possible]
    if thresholds.size == 0:  # no hit this turn can read it
        return [1], [1], 1
    return hit_class_law(thresholds)


def _crit(laws: OutcomeLaws, side: int) -> _Law:
    """The crit draw ``RandNext() % odds`` (battle_lib.c:7132) with the odds at the hit: with one
    possible odds N, 1 (crits) at 1/N and 0 at (N - 1)/N; with several (``crit_variants``), the
    draw r in [0, CRIT_DRAWS) reduced to its class (``outcome.crit_class``), each weighing its
    width. Values in increasing order."""
    if bool(laws.no_crit[side]):
        return [0], [1], 1
    possible = np.asarray(laws.crit_possible[side], bool)
    odds = np.asarray(laws.crit_variants[side])[possible]
    if len(set(odds.tolist())) == 1:
        n = int(odds[0])  # rand % n == 0
        return [0, 1], [n - 1, 1], n
    values = np.asarray(
        crit_class(np.arange(CRIT_DRAWS), np.asarray(laws.crit_variants[side]), possible)
    ).tolist()
    law = sorted(collections.Counter(values).items())
    return [v for v, _ in law], [w for _, w in law], CRIT_DRAWS


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
    if kind == DURATION_ENCORE:
        return [3, 4, 5, 6, 7], [1] * 5, 5
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


def _acupressure(laws: OutcomeLaws, side: int) -> _Law:
    """BtlCmd_BoostRandomStatBy2 (battle_script.c:8950): RandNext() % validStats, uniform by the
    % n convention, stored as a value in [0, ACUPRESSURE_DRAWS) read modulo the count at the move
    (``effects.acupressure_stat``, B12-9). A fixed count c gives c classes; a loose one the 420
    residues (each its own class over the counts 1-7)."""
    if bool(laws.acupressure_loose[side]):
        return list(range(ACUPRESSURE_DRAWS)), [1] * ACUPRESSURE_DRAWS, ACUPRESSURE_DRAWS
    n = max(int(laws.acupressure_count[side]), 1)
    return list(range(n)), [1] * n, n


def _chatter(laws: OutcomeLaws, side: int) -> _Law:
    # battle_script.c:8428-8430: RandNext() % 100 <= effectChance, uniform by the % n convention:
    # 1 with weight chance + 1 of 100 (G-70).
    chance = min(max(int(laws.chatter_chance[side]), 0), 99)
    return [0, 1], [99 - chance, chance + 1], 100


def _flinch_item(laws: OutcomeLaws, side: int) -> _Law:
    # BattleControllerPlayer_CheckExtraFlinch (battle_controller_player.c:4639):
    # RandNext() % 100 < param, uniform by the % n convention.
    chance = min(max(int(laws.flinch_item_chance[side]), 0), 100)
    return [0, 1], [100 - chance, chance], 100


def _focus_band(laws: OutcomeLaws, side: int) -> _Law:
    # bcp.c:3387, battle_script.c:9110: RandNext() % 100 < param, uniform by the % n convention
    # (ids.FOCUS_BAND_PARAM, the only HOLD_EFFECT_MAYBE_ENDURE param), G-1.
    return [0, 1], [100 - FOCUS_BAND_PARAM, FOCUS_BAND_PARAM], 100


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
    "chatter": _chatter,
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
    "future_hit": _future_hit,
    # subscript_fall_asleep.s Random 3, 2 at Yawn's turn end: 2-5, uniform.
    "yawn_sleep": lambda laws, side: ([2, 3, 4, 5], [1, 1, 1, 1], 4),
    "acupressure": _acupressure,
    "focus_band": _focus_band,
    "future_band": _focus_band,
    # subscript_confuse.s:34 Random 3, 2 (G-5): 2-5, uniform.
    "berry_confusion": lambda laws, side: ([2, 3, 4, 5], [1, 1, 1, 1], 4),
    # BattleSystem_Defender's RandNext() & 1 (lib.c:1653, :1700, :4187), B = 4 only (D2-4).
    "target_draw": _coin,
    # GoToMoveScript's / the Bide release's / ChooseTraceTarget's RandNext() & 1 (D2-13, D2-14,
    # D2-16), B = 4 only.
    "call_target_draw": _coin,
    "bide_target_draw": _coin,
    "trace_pick": _coin,
}


def _starf(laws: OutcomeLaws, side: int, k: int) -> _Law:
    """battle_lib.c:4864-4866: RandNext() % 5 until the stat is below +6, uniform over the
    eligible stats by the % n convention, one value in [0, 60) read modulo their count (F-10):
    the five values of a count fixed at 5, or all 60 where it can change (``_starf_reads``)."""
    if bool(laws.starf_loose[side, k]):
        return list(range(STARF_DRAWS)), [1] * STARF_DRAWS, STARF_DRAWS
    n = len(STARF_STATS)
    return list(range(n)), [1] * n, n


@cache
def _empty(b: int) -> Outcome:
    """``empty_outcome`` as numpy for ``b`` battlers (an ``Outcome`` depends on ``B`` only),
    cached per battler count."""
    return Outcome(*(np.asarray(f) for f in empty_outcome(next(f for f in FORMATS if f.B == b))))


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
    first, second = branch.first, _foe(branch.first)

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
        if bool(reads.chatter[s]):  # a Chatter confusion needs its roll (G-70)
            roll = rel_elem("chatter", (s,))
            conds = [*conds, Condition((roll,), _proc) if roll else NEVER]
        return conds + [procs(s)] if bool(reads.proc_gated[s]) else conds

    t = branch.tie
    for s in range(laws.reads.proc_gated.shape[0]):
        foe = _foe(s)
        hits = move_conds(s)
        for i in range(MAX_HITS):
            # B13-6: a later Triple Kick hit is read once the hit before can have landed, and its
            # own reads once it can land (class 0 misses at every threshold).
            own = []
            if i > 0 and (h := rel_elem("hit", (s, i))) is not None:
                before = rel_elem("hit", (s, i - 1))
                prior = [Condition((before,), _hits_somewhere)] if before else []
                plan.path(h, "move", *hits, *prior)
                own = [Condition((h,), _hits_somewhere)]
            for elem in (
                ("crit", (s, i)),
                ("secondary", (s, i)),
                ("contact_proc", (foe, i)),
                ("flinch_item", (s, i)),
            ):
                plan.path(elem, "move", *hits, *own)
            if bool(reads.self_lock[s]):  # a lock aimed at its user: its own contact holder
                plan.path(("contact_proc", (s, i)), "self", *hits, *own)
            if bool(reads.band_move[s, i]):  # the foe's Focus Band on this hit (G-1)
                plan.path(("focus_band", (s, i)), "move", *hits, *own)
            # roll[s, 0]: damage, Future Sight's stored damage, or a self-hit below.
            if i > 0 or bool(rel.crit[s, 0]) or bool(reads.future_roll[s]):
                plan.path(("roll", (s, i)), "move", *hits, *own)
        # A crash move (B10-2) reads hit 0's crit and roll whether or not it hits (its crash on
        # a miss or a Protect): a path of its own, unconditional (the flinch gate still applies).
        if bool(reads.crash[s]):
            plan.path(("crit", (s, 0)), "crash")
            plan.path(("roll", (s, 0)), "crash")
        for name in (
            "n_hits",
            "magnitude",
            "fang_flinch",
            "present",
            "protect",
            "tri_attack",
            "chatter",
            "roar_level",
            "roar_target",
            "acupressure",
        ):
            # G-30: Present's draw against a Wonder Guard side is read on a miss as well.
            guard = name == "present" and bool(reads.present_guard[s])
            plan.path((name, (s,)), "move", *([] if guard else hits))
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
        # G-1: a self-hit's Focus Band roll, read where its roll is (the confusion paths).
        # G-5: a disliked flavor berry the foe's earlier hit feeds it: the check reads the
        # berry's counter, and the self-hit's roll and coin follow; a pending counter is read
        # at the check like a confusion of its own.
        selfs = [("confusion_hit", (s,)), ("roll", (s, 0))]
        if bool(reads.band_self[s]):
            selfs.append(("focus_band", (s, 0)))
            if bool(reads.confused_own[s]):
                plan.path(("focus_band", (s, 0)), "confusion")
            if bool(reads.confused_by_foe[t, s]):
                plan.path(("focus_band", (s, 0)), "confusion_inflicted", *inflicted(foe))
        if bool(reads.pending_own[s]):
            plan.path(("berry_confusion", (s,)), "confusion")
        if bool(reads.berry_by_foe[t, s]):
            for elem in (*selfs, ("berry_confusion", (s,))):
                plan.path(elem, "confusion_berry", *move_conds(foe))

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
                ("focus_band", (q, i)),
            ):
                plan.gate(elem, cond)
        for name in (
            "n_hits",
            "duration",
            "magnitude",
            "confusion_hit",
            "berry_confusion",
            "full_para",
            "infatuation",
            "fang_flinch",
            "present",
            "protect",
            "tri_attack",
            "chatter",
            "roar_level",
            "roar_target",
            "acupressure",
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
    for s in range(laws.reads.present.shape[0]):
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
    first, q = branch.first, _foe(branch.first)
    reads = laws.reads
    if not (bool(reads.protect_sure[first]) and bool(reads.protect_blocks[q])):
        return
    elem = ("protect", (first,))
    cond = Condition((elem,), _not_protected) if elem in plan else NEVER
    # The accuracy roll is read for the move alone (step._use_move: the hit and the
    # "missed" event), so this is its only path; a later Triple Kick hit's too (B13-6).
    plan.path(("hit", (q, 0)), "move", cond)
    for i in range(1, MAX_HITS):
        plan.restrict(("hit", (q, i)), "move", cond)
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
            ("focus_band", (q, i)),
        ):
            # A Focus Band roll may be read only by a self-hit (no "move" path, G-1).
            optional = e == ("roll", (q, 0)) and ("crit", (q, 0)) not in plan
            optional = optional or e[0] == "focus_band"
            plan.restrict(e, "move", cond, required=not optional)
    for name in (
        "n_hits",
        "duration",
        "magnitude",
        "fang_flinch",
        "present",
        "tri_attack",
        "chatter",
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
def _chatter_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (chatter, duration) joint factor (G-70): ``effects._chatter`` reads the confusion counter
    only when the chatter roll passes, so ``duration[s]``'s "move" path also needs
    ``chatter[s] == 1``."""
    for s in range(laws.reads.chatter.shape[0]):
        elem, roll = ("duration", (s,)), ("chatter", (s,))
        if not bool(laws.reads.chatter[s]) or elem not in plan:
            continue
        plan.restrict(elem, "move", Condition((roll,), _proc) if roll in plan else NEVER)


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
    for s in range(laws.reads.tri_attack.shape[0]):
        elem = ("tri_attack", (s,))
        if not bool(laws.reads.tri_attack[s]) or elem not in plan:
            continue
        sec = ("secondary", (s, 0))
        plan.restrict(elem, "move", Condition((sec,), _proc) if sec in plan else NEVER)
        if branch.first != s:
            continue
        for name, status in (("thaw", Status.FREEZE), ("full_para", Status.PARALYSIS)):
            cond = Condition((elem,), _tri_value(status))
            plan.restrict((name, (_foe(s),)), "status_inflicted", cond, required=False)


def _passes(v: np.ndarray) -> np.ndarray:
    # 1: the level check passed. SKIPPED (not drawn: no check) fails nothing, but the builder
    # adds this condition only where roar_level is drawn whenever the pick is reached.
    return v == 1


@register_reads
def _endure_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (protect, focus_band) joint factor (G-65, G-1): the first mover's Endure that
    certainly reaches ``BtlCmd_TryProtection`` (``Reads.endure_sure``) sets
    ``turnFlags.enduring`` when its roll passes (always at ``protect_count`` 0), and
    ``if (DEFENDER_TURN_FLAGS.enduring == 0)`` (bcp.c:3386-3393) then skips the Focus Band roll on
    every hit of the second mover's move (``effects.play_hit``'s ``~enduring``). So
    ``focus_band[q, i]``'s "move" path also needs the Endure to fail; the self-hit's path (the
    user's own band) is not gated."""
    first, q = branch.first, _foe(branch.first)
    if not bool(laws.reads.endure_sure[first]):
        return
    elem = ("protect", (first,))
    cond = Condition((elem,), _not_protected) if elem in plan else NEVER
    for i in range(MAX_HITS):
        plan.restrict(("focus_band", (q, i)), "move", cond, required=False)


@register_reads
def _roar_reads(laws: OutcomeLaws, branch: Branch, plan: ReadPlan) -> None:
    """The (roar_level, roar_target) joint factor (rules §6(b)): ``BtlCmd_TryWhirlwind``
    (battle_script.c:5298-5311) runs its rejection loop only after
    ``BattleSystem_CanWhirlwind`` passed, so where the level check is drawn
    (``roar_level[s]`` relevant: a lower-levelled user), ``roar_target[s]``'s "move" path also
    needs ``roar_level[s] == 1``. Both carry the same move conditions (``_move_reads``,
    ``_protect_reads``), so the level check is drawn in every row that reaches the pick."""
    for s in range(laws.reads.roar.shape[0]):
        level, target = ("roar_level", (s,)), ("roar_target", (s,))
        if not bool(laws.reads.roar[s]) or target not in plan or level not in plan:
            continue
        plan.restrict(target, "move", Condition((level,), _passes))


def damage_class_law(
    laws: OutcomeLaws, side: int, hit: int = 0
) -> tuple[list[int], list[int], int, list[int], list[int]]:
    """``(values, weights, denominator, crits, rolls)``: the joint law of one hit's
    ``(crit, roll)`` reduced to its damage classes. Two pairs share a class when they have
    the same crit flag (a landed crit is observed, ``step._hit_events``) and the same
    damage in ``laws.damage[side, hit]``; a class weighs the sum of its pairs' weights over
    ``crit odds * ROLLS``, and ``crits[v]``, ``rolls[v]`` are class ``v``'s
    representative, its lowest roll. Values are class indices in crit-then-roll order."""
    table = np.asarray(laws.damage[side, hit])
    crit_values, crit_weights, crit_den = _crit(laws, side)
    # A hit whose crit odds can change is never compressed (outcome_laws' crit_moves).
    assert set(crit_values) <= {0, 1}, crit_values
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
        values, weights, den, crits, rolls = damage_class_law(laws, index[0], index[1])
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
    if name == "hit":  # B13-6: a later Triple Kick hit's own thresholds
        values, weights, den = _hit(laws, index[0], index[1])
    elif name == "starf":  # F-10: the turn's draw or the faint replacement's
        values, weights, den = _starf(laws, index[0], index[1])
    else:
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
    if bool(laws.call_pending):
        raise ValueError("an invoker is used: enumerate its calls (enumerate_turn)")
    reads = laws.reads
    tied = bool(laws.relevant.speed_tie)
    # A call variant keeps the speed_tie branches it is consistent with (Copycat's order).
    branches = [
        Branch(t, int(reads.first[t]), Outcome(*(f[t] for f in laws.branch_relevant)))
        for t in ((0, 1) if tied else (0,))
        if bool(laws.branch_ok[t])
    ]
    parts = []
    for branch in branches:
        budget = WIDTH_CAP - sum(len(p[1]) for p in parts)
        parts.append(_branch_rows(laws, branch, budget, (1, 2) if tied else (1, 1)))
    k = sum(len(p[1]) for p in parts)
    empty = _empty(np.shape(laws.relevant.full_para)[0])
    fields = {name: np.repeat(f[None], k, axis=0) for name, f in zip(Outcome._fields, empty)}
    drawn = {name: np.zeros((k, *np.shape(f)), bool) for name, f in zip(Outcome._fields, empty)}
    if tied and parts:
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
            fields[name][(rows_, *index)] = np.where(read, col, target).astype(fields[name].dtype)
            drawn[name][(rows_, *index)] = read
        start += len(probs)
    fields["call"][:] = laws.call.astype(fields["call"].dtype)
    probs = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.float64)
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


# --- The invokers' calls (E2 batch 5, rulings B5-n) ----------------------------------------

# Call values per side: Sleep Talk's 0..SLEEP_TALK_VALUES - 1, Metronome's candidate ranks (at
# most METRONOME_DRAW), Copycat's two orders.
CALL_VALUES = METRONOME_DRAW
# Sleep Talk's call variants when a faster foe's Disable (ruling B5-13) or Imprison (B7-13) can
# shrink its set within the turn: a value ``u`` in 0..SLEEP_TALK_VALUES - 1 is unsplit;
# ``u + CALL_KEEP`` assumes the turn-start set (nothing landed before the move), ``u +
# CALL_DISABLED`` the set with the user's last move disabled and ``u + CALL_SEALED`` the set
# without the moves the foe's Imprison seals. ``order.invoked_move`` takes ``call % count`` with
# ``count`` 1-4, a divisor of every offset, so ``step`` plays every encoding of ``u`` alike, on
# the live set; ``call_consistent`` keeps each row in the one variant whose assumption its turn
# met (one foe move a turn: never both).
CALL_KEEP = SLEEP_TALK_VALUES
CALL_DISABLED = 2 * SLEEP_TALK_VALUES
CALL_SEALED = 3 * SLEEP_TALK_VALUES


def _paused_rock(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``bool[B]``: the paused action's King's Rock read (``flinch_item[pivot, 0]``) is possible:
    a paused U-turn whose hit damaged its target (``ts_damaged``), with
    ``MOVE_FLAG_TRIGGERS_KINGS_ROCK``, before its target moved (``moved_mask``: also a 0-HP target, whose KO'd action never runs and whose
    King's Rock flinch needs its ``curHP``: battle_controller_player.c:4642, :3820-3826, :4430-4431;
    subscript_flinch_mon.s:5; SF2 (j)); whether the
    newcomer holds one is the law's (``flinch_item_chance``, its item through ``pivot_view``)."""
    fmt = format_of(state)
    pivot = jnp.clip(state.pivot_battler.astype(_I32) - 1, 0, fmt.B - 1)
    flag = (tables.move_flags[state.ts_used] & MoveFlag.KINGS_ROCK) != 0
    on = (state.pivot_battler != 0) & (state.ts_damaged != 0) & flag
    # The target: the other battler (singles), not yet acted.
    target_free = foes(moved_mask(state)) == 0
    return on & (jnp.arange(fmt.B) == pivot) & target_free


def _split_events(state: BattleState, events) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(disabled, sealed)`` ``bool[B]``: the Sleep Talk set changed before the side's action
    by a Disable (a move disabled at its start, none at the turn start) or an Imprison (a foe
    imprisoning at its start, none at the turn start)."""
    fmt = format_of(state)
    start_sealed = jnp.stack([imprisoning_foe(state, side) for side in range(fmt.B)])
    disabled = (jnp.asarray(events["disabled_at_move"]) != 0) & (state.disabled_move == 0)
    sealed = (jnp.asarray(events["sealed_at_move"]) != 0) & ~start_sealed
    return disabled, sealed


def gravity_events(state: BattleState, events) -> jnp.ndarray:
    """``bool[B]``: Gravity was up when the side's action started (the ``gravity_at_move`` event
    of ``step_events``) and down at the turn start: a faster foe's Gravity landed before its
    invoker call (G-67)."""
    return (jnp.asarray(events["gravity_at_move"]) != 0) & (state.gravity == 0)


def taken_events(state: BattleState, events) -> jnp.ndarray:
    """``bool[B]``: the side's chosen move was replaced by an encored move when its action
    started (the ``encored_at_move`` event of ``step_events``) and it was not encored at the turn
    start: a foe's Encore landed before its action (ruling B11-1)."""
    return (jnp.asarray(events["encored_at_move"]) != 0) & (state.encore_move == 0)


def call_consistent(
    tables: Tables, state: BattleState, actions, outcome: Outcome, events, taken=None
) -> jnp.ndarray:
    """``bool``: an outcome row of a call variant keeps its weight (rulings B5-13, B11-1). The
    variant's takeover tag ``taken`` (``bool[B]``, default none; ``call_variants``) must match the
    turn: a side that ``order.takeover_possible`` allows keeps a row only where its encored move
    replaced the chosen one (``taken_events``) exactly when the tag says so. For each Sleep Talk
    side not taken over whose call is a split variant (``CALL_KEEP`` / ``CALL_DISABLED`` /
    ``CALL_SEALED`` / ``CALL_GRAVITY``), whether a Disable, an Imprison or a Gravity changed its
    set before its action (``_split_events`` and ``gravity_events`` of ``step_events`` on
    ``state``, ``actions``, ``outcome``) matches the variant. A Metronome or Assist side not
    taken over keeps a row only where its call is a Gravity variant (``>= CALL_GRAVITY``)
    exactly when a foe's Gravity landed before it (G-67). Unsplit Sleep Talk rows always keep."""
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    taken = jnp.zeros((fmt.B,), bool) if taken is None else jnp.asarray(taken, bool)
    chosen = jnp.stack(
        [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
    )
    fight = (state.over == 0) & ~is_switch(actions) & ~is_item(actions) & ~taken
    kinds = call_kind(tables, chosen)
    talk = fight & (kinds == CALL_SLEEP_TALK)
    listing = fight & ((kinds == CALL_METRONOME) | (kinds == CALL_ASSIST))
    call = jnp.asarray(outcome.call, _I32)
    split = talk & (call >= CALL_KEEP)
    disabled, sealed = _split_events(state, events)
    landed = gravity_events(state, events)
    seen = jnp.where(
        landed,
        CALL_GRAVITY,
        jnp.where(sealed, CALL_SEALED, jnp.where(disabled, CALL_DISABLED, CALL_KEEP)),
    )
    want = jnp.where(
        call >= CALL_GRAVITY,
        CALL_GRAVITY,
        jnp.where(
            call >= CALL_SEALED,
            CALL_SEALED,
            jnp.where(call >= CALL_DISABLED, CALL_DISABLED, CALL_KEEP),
        ),
    )
    listed_ok = ~listing | ((call >= CALL_GRAVITY) == landed)
    view, view_actions = pivot_view(tables, state, actions)
    can_take = jnp.stack(
        [takeover_possible(tables, view, view_actions, side, _foe(side)) for side in range(fmt.B)]
    )
    took = taken_events(state, events)
    return (
        jnp.all(~split | (seen == want)) & jnp.all(listed_ok) & jnp.all(~can_take | (took == taken))
    )


class CallLaws(NamedTuple):
    """Each side's invoker draw before the turn (``call_laws``). Fixed shapes; a batch adds a
    leading axis."""

    kind: jnp.ndarray  # int32[B]: order.CALL_*, CALL_NONE when nothing is drawn
    weight: jnp.ndarray  # int32[B, CALL_VALUES]: the integer weight of each call value
    called: jnp.ndarray  # int32[B, CALL_VALUES]: the move each value calls (0: the call fails)
    # int32[B, CALL_VALUES]: Sleep Talk's call of each value from the set a faster foe's Disable
    # can leave (``order.disabled_state``, ruling B5-13); ``called`` where no such set can arise.
    alt: jnp.ndarray
    # int32[B, CALL_VALUES]: likewise from the set a faster foe's Imprison can leave
    # (``order.sealed_state``, ruling B7-13); ``called`` where no such set can arise.
    sealed: jnp.ndarray
    # bool[B]: a faster foe's Disable / Imprison can change the side's Sleep Talk set this turn.
    can_disable: jnp.ndarray
    can_seal: jnp.ndarray
    # The takeover variant (ruling B11-1): bool[B] a foe's Encore can replace the side's move
    # (``order.takeover_possible``), and the call law of the encored move read with the encored
    # state (``order.encored_state``): its kind, weights and calls as above.
    can_take: jnp.ndarray
    taken_kind: jnp.ndarray
    taken_weight: jnp.ndarray
    taken_called: jnp.ndarray
    # The Gravity variant (G-67): bool[B] a faster foe's Gravity can land before the side's
    # Sleep Talk, Metronome or Assist call (``order.gravity_possible``), and the call law read
    # with Gravity up (``order.gravity_state``): its weights and calls as above.
    can_gravity: jnp.ndarray
    gravity_weight: jnp.ndarray
    gravity_called: jnp.ndarray


def _side_call_law(
    tables: Tables, state: BattleState, side: int, chosen, kind, foe_chosen, foe_uses_move
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(weight int32[CALL_VALUES], called int32[CALL_VALUES])``: the call law of ``side``'s
    move ``chosen`` of invoker kind ``kind``, read with ``state`` (``call_laws``)."""
    values = jnp.arange(CALL_VALUES, dtype=_I32)
    talk_w = (values < SLEEP_TALK_VALUES).astype(_I32)
    talk = jax.vmap(lambda v: invoked_move(tables, state, side, chosen, v))(values)
    metro_all = metronome_weights(tables, state, side)
    cands = metro_all > 0
    n = jnp.sum(cands.astype(_I32))
    # The candidate of each rank (the k-th true entry), and its weight.
    rank = jnp.cumsum(cands.astype(_I32)) - 1
    ids = jnp.arange(metro_all.shape[0], dtype=_I32)
    by_rank = (
        jnp.zeros((CALL_VALUES,), _I32)
        .at[jnp.where(cands, rank, CALL_VALUES)]
        .set(ids, mode="drop")
    )
    metro_w = jnp.where(values < n, metro_all[by_rank], 0)
    copy_w = ((values == 0) | ((values == 1) & foe_uses_move)).astype(_I32)
    # Assist (B10-11): every list position weighs 1 (RandNext() % numMoves), the list fixed for
    # the turn; position v calls the v-th valid entry. An empty list draws nothing and fails
    # (battle_script.c:6438-6442): value 0 alone, calling no move.
    listed, valid = assist_candidates(tables, state, side)
    n_listed = jnp.sum(valid.astype(_I32))
    assist_w = jnp.where(n_listed > 0, values < n_listed, values == 0).astype(_I32)
    listed_rank = jnp.cumsum(valid.astype(_I32)) - 1
    assist_called = (
        jnp.zeros((CALL_VALUES,), _I32)
        .at[jnp.where(valid, listed_rank, CALL_VALUES)]
        .set(listed, mode="drop")
    )
    copy_called = jnp.where(
        values == 0,
        copycat_call(tables, state.move_prev, state),
        jnp.where(values == 1, copycat_call(tables, foe_chosen, state), 0),
    )
    # Me First (E3-2-3): value 0, its target has not acted (it copies the target's chosen move),
    # value 1, it has (the call fails).
    first_called = jnp.where(
        values == 0, me_first_copy(tables, state, _foe(side), foe_chosen, foe_uses_move), 0
    )
    weight = jnp.where(
        kind == CALL_SLEEP_TALK,
        talk_w,
        jnp.where(
            kind == CALL_METRONOME,
            metro_w,
            jnp.where(
                (kind == CALL_COPYCAT) | (kind == CALL_ME_FIRST),
                copy_w,
                jnp.where(kind == CALL_ASSIST, assist_w, (values == 0).astype(_I32)),
            ),
        ),
    )
    called = jnp.where(
        kind == CALL_SLEEP_TALK,
        talk,
        jnp.where(
            kind == CALL_METRONOME,
            by_rank,
            jnp.where(
                kind == CALL_COPYCAT,
                copy_called,
                jnp.where(
                    kind == CALL_ASSIST,
                    assist_called,
                    jnp.where(kind == CALL_ME_FIRST, first_called, 0),
                ),
            ),
        ),
    )
    return weight, called


def call_laws(tables: Tables, state: BattleState, actions) -> CallLaws:
    """The law of ``Outcome.call`` for ``step(tables, state, actions, .)``, as
    ``outcome.sample_outcome`` draws it, read with the turn-start ``state`` (a paused one
    through ``outcome.pivot_view``):

    * Sleep Talk: every value 0..``SLEEP_TALK_VALUES`` - 1 weighs 1 (BtlCmd_TrySleepTalk's
      uniform redraw, battle_script.c:4931-4933; ``order.invoked_move`` takes the allowed slot
      ``value % count``), from the turn-start set; ``alt`` from the set a faster foe's Disable can
      leave (ruling B5-13), possible while nothing is disabled, the user has a last move and its
      foe uses a move; ``sealed`` from the set a faster foe's Imprison can leave (B7-13). Drawn
      only when it can be read: the user sleeps at the start, or its foe uses an invoker or a
      move that can put it to sleep (over-approximate);
    * Metronome: each candidate rank weighs ``order.metronome_weights`` (141 or 140 of 65536)
      over the turn-start set; Assist likewise over its list (each entry 1);
    * the Gravity variant (G-67): where a faster foe's Gravity can land before a Sleep Talk,
      Metronome or Assist call (``order.gravity_possible``), the law read with Gravity up
      (``order.gravity_state``: ``Move_CanBeMetronomed``, battle_lib.c:7198-7201, and Sleep
      Talk's CHECK_INVALID_GRAVITY, :2292-2295, read Gravity when the call runs);
    * Copycat: no draw; its two orders (``outcome_laws``' ``call``) each weigh 1 and keep their
      consistent ``speed_tie`` branches (``branch_ok``), the second only when its foe uses a
      move;
    * the takeover variant (ruling B11-1): where a foe's Encore can replace the side's move
      (``order.takeover_possible``), the law of the encored move (an invoker or not) read with
      ``order.encored_state``. Move_CanBeEncored (battle_lib.c:7221-7231) leaves Metronome,
      Assist, Sleep Talk and Copycat; an encored Sleep Talk always fails
      (BattleSystem_CheckInvalidMoves, battle_lib.c:2302-2304, leaves it only itself, an invoker:
      battle_script.c:4916-4930)."""
    fmt = format_of(state)
    state, actions = pivot_view(tables, state, actions)
    live = state.over == 0
    uses_move = live & ~is_switch(actions) & ~is_item(actions)
    chosen = jnp.stack(
        [selected_move(tables, state, side, actions[side])[0] for side in range(fmt.B)]
    )
    kinds = jnp.where(uses_move, call_kind(tables, chosen), CALL_NONE)
    foe_called = foes(
        jnp.stack([called_move(tables, state, chosen[side]) for side in range(fmt.B)])
    )
    sleeps = state.status[party(fmt, jnp.arange(fmt.B)), state.active] == Status.SLEEP
    put_to_sleep = foes(uses_move) & (
        (call_kind(tables, foes(chosen)) != CALL_NONE)
        | _inflicts(tables, state, foe_called, Status.SLEEP)
    )
    kinds = jnp.where((kinds == CALL_SLEEP_TALK) & ~sleeps & ~put_to_sleep, CALL_NONE, kinds)
    values = jnp.arange(CALL_VALUES, dtype=_I32)
    can_split = (
        (kinds == CALL_SLEEP_TALK)
        & (state.disabled_move == 0)
        & (state.last_move != 0)
        & foes(uses_move)
    )
    # A foe's Imprison can land before the move while no battler of the other side imprisons yet.
    can_seal = (
        (kinds == CALL_SLEEP_TALK)
        & ~jnp.stack([imprisoning_foe(state, side) for side in range(fmt.B)])
        & foes(uses_move)
    )
    can_take = jnp.stack(
        [takeover_possible(tables, state, actions, side, _foe(side)) for side in range(fmt.B)]
    )
    encored = state.last_move.astype(_I32)
    taken_kinds = jnp.where(can_take, call_kind(tables, encored), CALL_NONE)
    taken_kinds = jnp.where(
        (taken_kinds == CALL_SLEEP_TALK) & ~sleeps & ~put_to_sleep, CALL_NONE, taken_kinds
    )
    can_gravity = jnp.stack(
        [gravity_possible(tables, state, actions, side, _foe(side)) for side in range(fmt.B)]
    ) & (kinds != CALL_NONE)
    heavy = gravity_state(state)
    weights, called, alts, seals, t_weights, t_called = [], [], [], [], [], []
    g_weights, g_called = [], []
    for side in range(fmt.B):
        foe = _foe(side)
        weight, call = _side_call_law(
            tables, state, side, chosen[side], kinds[side], chosen[foe], uses_move[foe]
        )
        weights.append(weight)
        called.append(call)
        hyp = disabled_state(state, jnp.arange(fmt.B) == side)
        alt = jax.vmap(
            lambda v, side=side, hyp=hyp: invoked_move(tables, hyp, side, chosen[side], v)
        )(values)
        shut = sealed_state(state, jnp.arange(fmt.B) == side)
        seal = jax.vmap(
            lambda v, side=side, shut=shut: invoked_move(tables, shut, side, chosen[side], v)
        )(values)
        alts.append(jnp.where(can_split[side], alt, call))
        seals.append(jnp.where(can_seal[side], seal, call))
        took = encored_state(tables, state, jnp.arange(fmt.B) == side)
        t_weight, t_call = _side_call_law(
            tables, took, side, encored[side], taken_kinds[side], chosen[foe], uses_move[foe]
        )
        t_weights.append(t_weight)
        t_called.append(t_call)
        g_weight, g_call = _side_call_law(
            tables, heavy, side, chosen[side], kinds[side], chosen[foe], uses_move[foe]
        )
        g_weights.append(g_weight)
        g_called.append(jnp.where(can_gravity[side], g_call, call))
    return CallLaws(
        kind=kinds.astype(_I32),
        weight=jnp.stack(weights),
        called=jnp.stack(called),
        alt=jnp.stack(alts),
        sealed=jnp.stack(seals),
        can_disable=can_split,
        can_seal=can_seal,
        can_take=can_take,
        taken_kind=taken_kinds.astype(_I32),
        taken_weight=jnp.stack(t_weights),
        taken_called=jnp.stack(t_called),
        can_gravity=can_gravity,
        gravity_weight=jnp.stack(g_weights),
        gravity_called=jnp.stack(g_called),
    )


def _side_variants(kind, weight, called, alt, sealed, heavy, offsets) -> list[tuple[int, int, int]]:
    """``(value, numerator, denominator)`` per call variant of one side's law
    (``call_variants``); ``heavy`` is Sleep Talk's call from the set Gravity leaves."""
    if kind in (CALL_SLEEP_TALK, CALL_METRONOME, CALL_ASSIST):
        groups: dict[tuple[int, int, int, int], list[int]] = {}
        for v in np.flatnonzero(weight):
            key = (int(called[v]), int(alt[v]), int(sealed[v]), int(heavy[v]))
            groups.setdefault(key, [int(v), 0])[1] += int(weight[v])
        den = int(weight.sum())
        out = []
        for (move, other, shut, grav), (v, w) in groups.items():
            if move == other == shut == grav:
                out.append((v, w, den))
            else:
                # One variant per set the turn can leave, each kept only where the turn met
                # its assumption (``call_consistent``).
                out += [(v + off, w, den) for off in offsets]
        return out
    if kind in (CALL_COPYCAT, CALL_ME_FIRST):
        return [(int(v), 1, 1) for v in np.flatnonzero(weight)]
    return [(0, 1, 1)]


class Variant(NamedTuple):
    """One call variant of a turn (``call_variants``): the invokers' ``call`` (``int32[B]``,
    ``Outcome.call``'s value), the takeover tag ``taken`` (``bool[B]``: the variant assumes a
    foe's Encore replaced the side's move before its action, ruling B11-1; a field beside
    ``call``, never an offset of it) and its weight ``num / den``."""

    call: np.ndarray
    taken: np.ndarray
    num: int
    den: int


def call_variants(laws: CallLaws) -> list[Variant]:
    """The call variants (``Variant``) of one (unbatched) row of ``call_laws``: the product of
    each side's values grouped by the move they call (one representative value per move, the
    smallest, with the group's summed weight). A Sleep Talk group whose values call another move
    from the Disable-shrunk set (``CallLaws.alt``) is two variants of the same weight,
    ``CALL_KEEP`` and ``CALL_DISABLED`` (ruling B5-13), and one more, ``CALL_SEALED``, where a
    faster foe's Imprison can land (B7-13), and ``CALL_GRAVITY`` where a faster foe's Gravity can
    (G-67, ``CallLaws.can_gravity``). A Metronome or Assist side whose foe's Gravity can land
    first also has the Gravity variants: the law under Gravity, values ``+ CALL_GRAVITY``, of
    total weight 1 beside the turn-start law's 1. Where a foe's Encore can replace the side's move
    (``CallLaws.can_take``), the side also has the takeover variants (``taken``): the encored
    move's own law, of total weight 1 beside the others' 1 (ruling B11-1). The rows of all these
    variants are kept only where the turn met the variant's assumption (``call_consistent``), so
    each side's variants sum to 1 over the rows kept. Copycat's orders are not a chance: each has
    weight 1 / 1 and its laws keep only the branches that agree with it."""
    laws = jax.tree.map(np.asarray, laws)
    per_side = []
    for side in range(laws.kind.shape[0]):
        offsets = [CALL_KEEP]
        offsets += [CALL_DISABLED] if bool(laws.can_disable[side]) else []
        offsets += [CALL_SEALED] if bool(laws.can_seal[side]) else []
        kind = int(laws.kind[side])
        gravity = bool(laws.can_gravity[side])
        talk = kind == CALL_SLEEP_TALK
        offsets += [CALL_GRAVITY] if gravity and talk else []
        heavy = laws.gravity_called[side] if talk else laws.called[side]
        law = (laws.weight[side], laws.called[side], laws.alt[side], laws.sealed[side], heavy)
        out = [(v, False, w, d) for v, w, d in _side_variants(kind, *law, offsets)]
        if gravity and kind in (CALL_METRONOME, CALL_ASSIST):
            heavy = laws.gravity_called[side]
            law = (laws.gravity_weight[side], heavy, heavy, heavy, heavy)
            out += [(v + CALL_GRAVITY, False, w, d) for v, w, d in _side_variants(kind, *law, [])]
        if bool(laws.can_take[side]):
            took = laws.taken_called[side]
            law = (laws.taken_weight[side], took, took, took, took)
            kind = int(laws.taken_kind[side])
            out += [(v, True, w, d) for v, w, d in _side_variants(kind, *law, [])]
        per_side.append(out)
    return [
        Variant(np.asarray([a, b], np.int32), np.asarray([ta, tb], bool), wa * wb, da * db)
        for a, ta, wa, da in per_side[0]
        for b, tb, wb, db in per_side[1]
    ]


def combine_variants(
    parts: list[tuple[Outcome, np.ndarray, int, int]],
) -> tuple[Outcome, np.ndarray]:
    """The rows of every call variant ``(outcomes, probs, numerator, denominator)`` as one
    enumeration, each variant's probabilities scaled by its weight. Raises ``SamplingOnly`` when
    the variants' rows together exceed ``WIDTH_CAP``."""
    total = sum(len(p[1]) for p in parts)
    if total > WIDTH_CAP:
        raise SamplingOnly(
            f"the outcome combinations exceed the width cap {WIDTH_CAP} "
            f"({total} rows over {len(parts)} call variants)"
        )
    outcomes = Outcome(
        *(
            np.concatenate([np.asarray(getattr(p[0], name)) for p in parts])
            for name in Outcome._fields
        )
    )
    probs = np.concatenate([p[1] * (p[2] / p[3]) for p in parts])
    return outcomes, probs


_jit_call_laws = jax.jit(call_laws)
_jit_laws = jax.jit(outcome_laws)


def enumerate_turn(tables: Tables, state: BattleState, actions) -> tuple[Outcome, np.ndarray]:
    """``enumerate_outcomes`` over every call variant (``call_variants``) of one state and action
    pair: the turn's full enumeration whatever the invokers draw, the ``Outcome.call`` of each
    row its variant's value. Raises ``SamplingOnly`` above ``WIDTH_CAP`` rows in all. Split Sleep
    Talk variants (ruling B5-13) and takeover variants (B11-1) are played (``step.turn_events``)
    and each row is kept only where ``call_consistent``."""
    from nuzline.engine.coverage import check_acid_rain  # coverage imports the engine

    check_acid_rain(state)  # a refused state is never played on (docs/kaizo FCM.md section 5)
    fmt = format_of(state)
    actions = jnp.asarray(actions, _I32)
    parts, tags = [], []
    cl = _jit_call_laws(tables, state, actions)
    talks = np.asarray(cl.kind) == CALL_SLEEP_TALK
    split = bool(np.any(np.asarray(cl.can_take)) or np.any(np.asarray(cl.can_gravity)))
    for v in call_variants(cl):
        laws = _jit_laws(tables, state, actions, jnp.asarray(v.call), jnp.asarray(v.taken))
        outcomes, probs = enumerate_outcomes(laws)
        parts.append((outcomes, probs, v.num, v.den))
        tags.append(np.broadcast_to(v.taken, (len(probs), fmt.B)))
        split = split or bool(np.any(talks & ~v.taken & (v.call >= CALL_KEEP)))
    outcomes, probs = combine_variants(parts)
    if split:
        taken = np.concatenate(tags)
        keep = np.asarray(_jit_keep()(tables, state, actions, outcomes, taken))
        outcomes = jax.tree.map(lambda f: np.asarray(f)[keep], outcomes)
        probs = probs[keep]
    return outcomes, probs


@cache
def _jit_keep():
    from nuzline.engine.step import turn_events  # step imports this module's peers only

    def keep(tables, state, actions, outcome, taken):
        events = turn_events(tables, state, actions, outcome)
        return call_consistent(tables, state, actions, outcome, events, taken)

    return jax.jit(jax.vmap(keep, in_axes=(None, None, None, 0, 0)))
