"""E1 trial batch (ruling G-63, docs/kaizo/e1-mechanics-ledger.md): Plus and Minus
(``BattleSystem_CalcMoveDamage``, battle_lib.c:6802-6809), each case checked against the
reference engine."""

from __future__ import annotations

from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table, side_ability_count
from nuzline.engine.ids import AbilityId
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import roark_state


def _ability(s, side, slot, ability):
    return s._replace(ability=s.ability.at[side, slot].set(ability))


def _both(tables, s, move):
    """The 16 damage rolls of ``move`` from side 0, JAX and reference (asserted equal)."""
    got = damage_table(tables, s, 0, move, False).tolist()
    assert got == ref_damage_table(to_numpy(tables), to_ref(s), 0, move, False)
    return got


def test_plus_and_minus_are_covered():
    for ability in (AbilityId.PLUS, AbilityId.MINUS):
        assert IMPLEMENTED_ABILITIES[ability] == Support.EXACT


def test_plus_minus_never_boost_in_singles_with_a_bench_partner(tables, names):
    """The only battler on the attacker's side is the attacker (the bench is not on the field),
    so a Plus attacker with a Minus on the bench (and the reverse, and Plus with Plus) deals the
    no-ability damage; physical and special moves alike (only Sp. Atk is ever scaled)."""
    s = roark_state(tables, names, player_moves=["Water Gun", "Tackle"])
    water, tackle = names.move("Water Gun"), names.move("Tackle")
    bare = _ability(s, 0, 0, AbilityId.NONE)
    base = {m: _both(tables, bare, m) for m in (water, tackle)}
    for lead, bench in (
        (AbilityId.PLUS, AbilityId.MINUS),
        (AbilityId.MINUS, AbilityId.PLUS),
        (AbilityId.PLUS, AbilityId.PLUS),
    ):
        t = _ability(_ability(s, 0, 0, lead), 0, 1, bench)
        for m in (water, tackle):
            assert _both(tables, t, m) == base[m]


def test_side_count_is_the_living_active_battlers_with_the_ability(tables, names):
    """COUNT_ALIVE_BATTLERS_OUR_SIDE (battle_lib.c:2886-2893): the asker itself counts, a fainted
    one does not, and the bench never does."""
    s = roark_state(tables, names)
    t = _ability(_ability(s, 0, 0, AbilityId.MINUS), 0, 1, AbilityId.MINUS)
    assert int(side_ability_count(t, 0, AbilityId.MINUS)) == 1  # the lead; the bench one is out
    assert int(side_ability_count(t, 0, AbilityId.PLUS)) == 0
    fainted = t._replace(hp=t.hp.at[0, 0].set(0))
    assert int(side_ability_count(fainted, 0, AbilityId.MINUS)) == 0
