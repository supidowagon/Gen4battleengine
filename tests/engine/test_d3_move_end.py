"""D3-15: MoveEnd over four battlers (sheet D3-15 sections 1, 4.5; SF2 (a)).
``BattleControllerPlayer_MoveEnd`` (bcp.c:3962-3999) runs ``RecoverStatusByAbility`` for the
attacker and the defender (:3965-3967) and, while the battle goes on, ``TriggerEffectOnSwitch``
(:3973), whose ``SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS`` cures every battler (lib.c:4135-4148)
and whose ``..._HELD_ITEM_STATUS`` checks every battler's item (:4151-4165). A battle that ends at
the MoveEnd has run only the first two calls (CheckBattleOver, :3969, returns before :3973).
``ClearFlags`` clears the attacker's ``moveHit`` (:4389); the defender's, written by
``UpdateFlagsWhenHit`` after the faint loop, stays (:4499-4508).

Each case is the action phase of a turn (``play``: engine equal to reference, every field). The
switch case (battler 0 switches, battler 3 cured at the switch's MoveEnd) is D3-13's.
"""

from __future__ import annotations

from nuzline.engine.format import DOUBLES, MULTI, TWO_TRAINERS
from nuzline.engine.ids import AbilityId, Stat, Status
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import WINNER_PLAYER
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.test_d3_action_loop import _all_moves, _set_item, _set_status, hp, play

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
    assert int(out.move_hit[1]) == names.move(BODY_SLAM)  # the defender's record stays
    assert int(out.move_hit[0]) == 0  # the attacker's is cleared (ClearFlags)


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
