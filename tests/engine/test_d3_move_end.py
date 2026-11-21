"""D3-15: MoveEnd over four battlers (sheet D3-15 sections 1, 4.5; SF2 (a)).
``BattleControllerPlayer_MoveEnd`` (bcp.c:3962-3999) runs ``RecoverStatusByAbility`` for the
attacker and the defender (:3965-3967) and, while the battle goes on, ``TriggerEffectOnSwitch``
(:3973), whose ``SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS`` cures every battler (lib.c:4135-4148)
and whose ``..._HELD_ITEM_STATUS`` checks every battler's item (:4151-4165). A battle that ends at
the MoveEnd has run only the first two calls (CheckBattleOver, :3969, returns before :3973).
``ClearFlags`` clears the attacker's ``moveHit`` (:4389); the defender's, written by
``UpdateFlagsWhenHit`` after the faint loop, stays (:4499-4508).

Each case is the action phase of a turn (``play``: engine equal to reference, every field), the
switch case included (battler 0 switches, battler 3 cured at the switch's MoveEnd, D3-13).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.damage import confusion_damage, damage_for_roll
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS
from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import WINNER_PLAYER
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.test_d3_action_loop import (
    _all_moves,
    _set_item,
    _set_status,
    hp,
    play,
    speed_ordered,
)
from tests.engine.test_pivot import _mon

TACKLE, SWORDS, BODY_SLAM = "Tackle", "Swords Dance", "Body Slam"


def status(state, b: int) -> int:
    return int(state.status[party_of(state, b), int(state.active[b])])


def sure_secondary():
    """Every secondary effect fires and every paralyzed battler is fully paralyzed."""
    base = empty_outcome(DOUBLES)
    return base._replace(secondary=base.secondary + 1, full_para=base.full_para + 1)


def para_cure_state(tables, names, *, partner_limber: bool = True):
    """DOUBLES: battler 0 (Mold Breaker, Body Slam) paralyzes battler 1 (Limber); battler 2 is
    Limber and already paralyzed (a status its ability forbids, in the state); the others Swords
    Dance."""
    s = _all_moves(four(tables, names, DOUBLES), names, SWORDS)
    s = with_moves(s, names, 0, [BODY_SLAM])
    s = with_ability(s, 0, AbilityId.MOLD_BREAKER)
    s = with_ability(s, 1, AbilityId.LIMBER)
    if partner_limber:
        s = with_ability(s, 2, AbilityId.LIMBER)
    return _set_status(s, 2, Status.PARALYSIS)


# ---------------------------------------------------------------- the battle goes on (rule 4)


def test_move_end_cures_every_battler_not_only_the_attacker_and_defender(tables, names):
    """Battler 0's Body Slam paralyzes battler 1 (Limber, pierced by Mold Breaker); at battler 0's
    MoveEnd battler 1 (its defender) and battler 2 (neither attacker nor defender, Limber,
    paralyzed in the state) are both cured. Every paralyzed battler is fully paralyzed
    (``full_para``) in its own action, so a Swords Dance that ran proves the cure came earlier."""
    s = para_cure_state(tables, names)
    out, _ = play(
        tables, s, [0, 0, 0, 0], (0, 1, 3, 2), targets=(1, 0, 0, 0), outcome=sure_secondary()
    )
    assert status(out, 1) == Status.NONE and status(out, 2) == Status.NONE
    assert int(out.stages[1, Stat.ATK]) == 2  # battler 1 moved: cured before its action
    assert int(out.stages[2, Stat.ATK]) == 2  # battler 2 moved: cured at battler 0's MoveEnd
    # Battler 0's ClearFlags clears its own record (bcp.c:4389); battler 1's Body Slam record
    # (bcp.c:4500) is cleared at battler 1's own MoveEnd, as it acts later (bcp.c:3986, :4389).
    assert int(out.move_hit[0]) == 0 and int(out.move_hit[1]) == 0


def test_without_limber_the_paralysis_stands_and_the_action_fails(tables, names):
    """The control: battler 2 holds no Limber, so its paralysis stays and its full-paralysis
    check stops its Swords Dance."""
    s = para_cure_state(tables, names, partner_limber=False)
    out, _ = play(
        tables, s, [0, 0, 0, 0], (0, 1, 3, 2), targets=(1, 0, 0, 0), outcome=sure_secondary()
    )
    assert status(out, 2) == Status.PARALYSIS
    assert int(out.stages[2, Stat.ATK]) == 0
    assert int(out.stages[1, Stat.ATK]) == 2  # battler 1 (Limber) was cured


# ---------------------------------------------------------------- the battle ends (SF2 (a))


def test_battle_end_cures_the_attacker_and_the_defender_only(tables, names):
    """Battler 1, the last of party 1 (battler 3 has no Pokemon with HP), Body Slams battler 0
    (Limber, pierced by Mold Breaker) and faints from Life Orb's recoil in the same action: the
    battle ends at battler 1's MoveEnd. Battler 0 (its defender) is cured (bcp.c:3966-3967,
    before CheckBattleOver, :3969); battler 2 (Limber, paralyzed, neither attacker nor defender)
    is not, as TriggerEffectOnSwitch (:3973) never runs."""
    s = _all_moves(four(tables, names, DOUBLES), names, SWORDS)
    s = with_moves(s, names, 1, [BODY_SLAM])
    s = with_ability(s, 1, AbilityId.MOLD_BREAKER)
    s = with_ability(s, 0, AbilityId.LIMBER)
    s = with_ability(s, 2, AbilityId.LIMBER)
    s = _set_status(s, 2, Status.PARALYSIS)
    s = _set_item(s, 1, names.item("Life Orb"))
    s = s._replace(hp=s.hp.at[1].set(0))  # party 1: no Pokemon with HP ...
    s = with_hp(s, 1, 1)  # ... but battler 1's, at 1 HP
    out, _ = play(
        tables, s, [0, 0, 0, 0], (1, 0, 2, 3), targets=(0, 0, 0, 0), outcome=sure_secondary()
    )
    assert (int(out.over), int(out.winner)) == (1, WINNER_PLAYER)
    assert hp(out, 1) == 0
    assert status(out, 0) == Status.NONE  # cured, in the party's copy
    assert status(out, 2) == Status.PARALYSIS  # not cured


# ---------------------------------------------------------------- move_hit on the defender


def test_ko_keeps_the_defenders_move_hit_and_clears_the_attackers(tables, names):
    """Battler 0's Tackle KOs battler 1: ``move_hit[1]`` = Tackle after the actions (written after
    the faint loop), ``move_hit[0]`` = 0."""
    s = _all_moves(four(tables, names, DOUBLES), names, SWORDS)
    s = with_moves(s, names, 0, [TACKLE])
    s = with_hp(s, 1, 1)
    out, _ = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=(1, 0, 0, 0))
    assert hp(out, 1) == 0
    assert int(out.move_hit[1]) == names.move(TACKLE)
    assert int(out.move_hit[0]) == 0


def test_ko_of_a_non_foe_battler_keeps_its_move_hit(tables, names):
    """The gap D3-7 found: the defender is battler 3 (or battler 2 in MULTI), not ``foe(side)``;
    the faint's reset of its ``move_hit`` is undone by the record written after it."""
    cases = ((DOUBLES, 0, 3), (DOUBLES, 2, 1), (MULTI, 1, 2), (TWO_TRAINERS, 0, 3))
    for fmt, attacker, target in cases:
        s = _all_moves(four(tables, names, fmt), names, SWORDS)
        s = with_moves(s, names, attacker, [TACKLE])
        s = with_hp(s, target, 1)
        targets = [0, 0, 0, 0]
        targets[attacker] = target
        order = [attacker] + [b for b in range(4) if b != attacker]
        out, _ = play(tables, s, [0, 0, 0, 0], order, targets=tuple(targets))
        assert hp(out, target) == 0, (fmt, attacker, target)
        assert int(out.move_hit[target]) == names.move(TACKLE), (fmt, attacker, target)
        assert int(out.move_hit[attacker]) == 0, (fmt, attacker, target)


# ------------------------------------------------ B = 2: the item pass at MoveEnd (bcp.c:3973)
# BattleSystem_TriggerEffectOnSwitch has no format test and loops to GetMaxBattlers
# (lib.c:3656), so SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS (lib.c:4151-4165) checks both
# battlers at every MoveEnd. A confusion self-hit (bcp.c:2629-2639) and a Jump Kick crash
# (bcp.c:3325-3328, subscript_crash_on_miss.s:22) reach MoveEnd without AfterMoveEffects, so a
# berry the damage brought into range is eaten there, before the later actor acts.


def _singles(tables, names, lead0, lead1, item0):
    """SINGLES: battler 0 (Machop L40, ``item0``, max HP 100) is faster than battler 1."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Machop", [lead0], 40, item0)],
        [_mon(n, "Gyarados", [lead1], 40)],
    )
    s = s._replace(max_hp=s.max_hp.at[0, 0].set(100))
    return speed_ordered(s, (0, 1))


def _eaten_at_move_end(tables, names, s, hp_before, outcome):
    """Run battler 0's action then battler 1's; battler 0 is a Sitrus holder at ``hp_before``."""
    s = s._replace(hp=s.hp.at[0, 0].set(hp_before))
    return play(tables, s, [0, 0], (0, 1), outcome=outcome)[0]


def test_confusion_self_hit_eats_sitrus_at_its_move_end(tables, names):
    """Battler 0 hurts itself (30 + D -> 30 HP). Its Sitrus Berry (HP_PCT_RESTORE, heals
    Divide(100 * power, 100), lib.c:4983-4986) is eaten at that MoveEnd, so battler 1's Dragon
    Rage (fixed 40) leaves 30 + power - 40; eaten late, the 30 HP would faint."""
    s = _singles(tables, names, TACKLE, "Dragon Rage", "Sitrus Berry")
    s = s._replace(confusion=s.confusion.at[0].set(3))
    o = empty_outcome(SINGLES)._replace(confusion_hit=jnp.array([1, 0], jnp.int8))
    d = int(confusion_damage(tables, s, 0, int(o.roll[0, 0])))
    power = int(tables.item_param[names.item("Sitrus Berry"), 0])
    out = _eaten_at_move_end(tables, names, s, 30 + d, o)
    assert int(out.hp[0, 0]) == 30 + power - 40  # 15 with the table's power 25
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == names.item("Sitrus Berry")


def test_jump_kick_crash_eats_sitrus_at_its_move_end(tables, names):
    """The same, after a Jump Kick that missed: the crash loss C (step.py, B10-2:
    min(Divide(damage, 2), Divide(defender maxHP, 2))) leaves 30 HP; the Sitrus Berry is eaten at
    that MoveEnd and the Dragon Rage leaves 30 + power - 40."""
    s = _singles(tables, names, "Jump Kick", "Dragon Rage", "Sitrus Berry")
    base = empty_outcome(SINGLES)
    miss = base._replace(hit=base.hit.at[0].set(0))  # battler 1's Dragon Rage still hits
    would = int(damage_for_roll(tables, s, 0, names.move("Jump Kick"), False, int(miss.roll[0, 0])))
    assert would > 0
    crash = min(max(would // 2, 1), max(int(s.max_hp[1, 0]) // 2, 1))
    power = int(tables.item_param[names.item("Sitrus Berry"), 0])
    out = _eaten_at_move_end(tables, names, s, 30 + crash, miss)
    assert int(out.hp[0, 0]) == 30 + power - 40
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == names.item("Sitrus Berry")


def test_confusion_self_hit_eats_liechi_at_its_move_end(tables, names):
    """A pinch berry: Liechi (HOLD_EFFECT_PINCH_ATK_UP) triggers at curHP <= maxHP / param
    (lib.c:5196-5206; max HP 100, so 100 // param). Battler 0's self-hit leaves exactly that HP;
    the berry is eaten at its MoveEnd (Attack +1), so battler 1's Trick, which would move an
    uneaten berry, finds no item and swaps nothing."""
    s = _singles(tables, names, TACKLE, "Trick", "Liechi Berry")
    s = s._replace(confusion=s.confusion.at[0].set(3))
    o = empty_outcome(SINGLES)._replace(confusion_hit=jnp.array([1, 0], jnp.int8))
    d = int(confusion_damage(tables, s, 0, int(o.roll[0, 0])))
    limit = 100 // int(tables.item_param[names.item("Liechi Berry"), 0])
    out = _eaten_at_move_end(tables, names, s, limit + d, o)
    assert int(out.hp[0, 0]) == limit
    # Gyarados's Intimidate left battler 0 at -1 when the leads came in; Liechi raises it by 1.
    assert int(s.stages[0, Stat.ATK]) == -1
    assert int(out.stages[0, Stat.ATK]) == 0
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == names.item("Liechi Berry")
    assert int(out.item[1, 0]) == 0  # Trick moved nothing


# ---------------------------------------------------------------- a switch's MoveEnd (D3-13)


def test_a_switch_action_cures_a_sleeping_insomnia_battler_at_its_move_end(tables, names):
    """Battler 0 switches to slot 2 (order 0, 1, 3, 2); battler 3 (Insomnia) is asleep with 3 turns
    left. The switch's MoveEnd runs TriggerEffectOnSwitch (bcp.c:3973) whose forbidden-status
    state cures every battler (lib.c:4135-4148), so battler 3 acts later in the turn; without
    Insomnia it stays asleep and its Swords Dance does not run."""
    switch = MOVE_ACTIONS + 2
    for ability, cured in ((AbilityId.INSOMNIA, True), (AbilityId.OVERGROW, False)):
        s = _all_moves(four(tables, names, DOUBLES), names, SWORDS)
        s = with_ability(s, 3, ability)
        s = _set_status(s, 3, Status.SLEEP)
        p, slot = party_of(s, 3), int(s.active[3])
        s = s._replace(status_counter=s.status_counter.at[p, slot].set(3))
        out, _ = play(tables, s, [switch, 0, 0, 0], (0, 1, 3, 2))
        assert (status(out, 3) == Status.NONE) == cured
        assert int(out.stages[3, Stat.ATK]) == (2 if cured else 0)
