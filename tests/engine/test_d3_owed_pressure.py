"""D3-16 (owes D2-25): Pressure's PP cost reads the fight command's defender, before the Lightning
Rod redirect (``BattleControllerPlayer_DecrementPP``, bcp.c:2235-2270; the redirect runs later,
:3204-3206), through the four-actor step: engine equal to the reference via ``play``."""

from __future__ import annotations

from nuzline.engine.ids import AbilityId
from tests.engine.d2_states import four, with_ability, with_moves
from tests.engine.test_d3_action_loop import hp, play, pp


def state(tables, names, pressure_on):
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    s = with_moves(s, names, 0, ["Thunderbolt"])
    s = with_ability(s, 3, AbilityId.LIGHTNINGROD)  # no Pressure
    for b in pressure_on:
        s = with_ability(s, b, AbilityId.PRESSURE)
    return s


def run(tables, names, pressure_on):
    s = state(tables, names, pressure_on)
    out, _ = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=(1, 0, 0, 0))
    return s, out


def test_the_cost_is_the_chosen_defenders_and_the_hit_lands_on_the_holder(tables, names):
    s, out = run(tables, names, [1])  # Pressure on the chosen defender, battler 1
    assert pp(s, 0, 0) - pp(out, 0, 0) == 2
    assert hp(out, 3) < hp(s, 3) and hp(out, 1) == hp(s, 1)  # redirected to battler 3


def test_no_pressure_costs_one(tables, names):
    s, out = run(tables, names, [])
    assert pp(s, 0, 0) - pp(out, 0, 0) == 1
    assert hp(out, 3) < hp(s, 3)
