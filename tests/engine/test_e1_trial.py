"""E1 trial batch (rulings G-63 .. , docs/kaizo/e1-mechanics-ledger.md): Plus and Minus
(``BattleSystem_CalcMoveDamage``, battle_lib.c:6802-6809), Endure (G-65, ``turnFlags.enduring``,
battle_controller_player.c:3386-3404), each case checked against the reference engine."""

from __future__ import annotations

import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table, side_ability_count
from nuzline.engine.ids import AbilityId, EffectId
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.engine.test_protect import _events, _hp, _o, _state


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


# --- Endure (G-65) -------------------------------------------------------------------------

ENDURE_MOVES = ("Endure", "Protect", "Scratch", "Swords Dance")


def _endure(tables, names, *, hp=None, item=None, **kw):
    """Monferno (Endure first) against Geodude; ``hp`` is the player's current (and, with an
    ``item``, maximum) HP."""
    s = _state(tables, names, mine=ENDURE_MOVES, **kw)
    if hp is not None:
        s = s._replace(hp=s.hp.at[0, 0].set(hp))
        if item:
            s = s._replace(max_hp=s.max_hp.at[0, 0].set(hp))
    if item:
        s = s._replace(item=s.item.at[0, 0].set(names.item(item)))
    return s


def test_endure_is_mapped(tables, names, kaizo_tables, kaizo_names):
    from tools.effect_map import DECOMP_EFFECT_TO_ID

    assert DECOMP_EFFECT_TO_ID[116] == (EffectId.ENDURE, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        m = n.move("Endure")
        assert int(t.move_effect[m]) == EffectId.ENDURE and int(t.move_priority[m]) == 3


def test_endure_leaves_one_hp_against_a_lethal_hit(tables, names):
    s = _endure(tables, names, hp=1)
    out = assert_matches_reference(tables, s, A("Endure", "Tackle", s, names), _o())
    assert _hp(out, 0) == 1 and int(out.protect_count[0]) == 1
    ev = _events(tables, s, A("Endure", "Tackle", s, names), _o())
    assert ev["endure_ok"].tolist() == [1, 0] and ev["protect_ok"].tolist() == [0, 0]
    assert ev["protected"].tolist() == [0, 0] and ev["hits"][1] == 1  # the hit lands
    # Without Endure the same hit faints it.
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o())
    assert _hp(out, 0) == 0


def test_a_failed_endure_roll_does_not_endure(tables, names):
    s = _endure(tables, names, hp=1, count=1)
    acts = A("Endure", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o((0, 1)))
    assert _hp(out, 0) == 0 and int(out.protect_count[0]) == 0
    ev = _events(tables, s, acts, _o((0, 1)))
    assert ev["protect_failed"].tolist() == [1, 0] and ev["endure_ok"].tolist() == [0, 0]
    out = assert_matches_reference(tables, s, acts, _o((1, 1)))  # the roll passes at chain 1
    assert _hp(out, 0) == 1 and int(out.protect_count[0]) == 2


def test_endure_acting_last_fails_whatever_the_roll(tables, names):
    s = _endure(tables, names, hp=1, opp="Protect", order="opp")
    acts = A("Endure", "Protect", s, names)
    for roll in (0, 1):
        out = assert_matches_reference(tables, s, acts, _o((roll, 1)))
        assert int(out.protect_count[0]) == 0
        ev = _events(tables, s, acts, _o((roll, 1)))
        assert ev["protect_failed"].tolist() == [1, 0] and ev["endure_ok"].tolist() == [0, 0]


def test_endure_and_protect_continue_each_others_chain(tables, names):
    for first, second in (("Endure", "Protect"), ("Protect", "Endure")):
        s = _endure(tables, names)
        out = assert_matches_reference(tables, s, A(first, "Leer", s, names), _o())
        assert int(out.protect_count[0]) == 1
        acts = A(second, "Leer", out, names)
        failed = _o((0, 1))  # chain 1: a failing roll fails the second move
        assert _events(tables, out, acts, failed)["protect_failed"][0] == 1
        assert int(assert_matches_reference(tables, out, acts, failed).protect_count[0]) == 0
        kept = assert_matches_reference(tables, out, acts, _o())
        assert int(kept.protect_count[0]) == 2
    # Another move in between resets the chain, so the roll no longer matters.
    s = _endure(tables, names)
    out = assert_matches_reference(tables, s, A("Endure", "Leer", s, names), _o())
    out = assert_matches_reference(tables, out, A("Swords Dance", "Leer", out, names), _o())
    assert int(out.protect_count[0]) == 0


def test_an_enduring_sash_holder_keeps_its_sash(tables, names):
    """bcp.c:3391-3393 sits inside ``if (enduring == 0)``: the full-HP sash is not armed, the cap
    is ``MOVE_STATUS_ENDURED`` (no item named, none removed)."""
    s = _endure(tables, names, hp=3, item="Focus Sash")
    out = assert_matches_reference(tables, s, A("Endure", "Tackle", s, names), _o())
    assert _hp(out, 0) == 1
    assert int(out.item[0, 0]) == names.item("Focus Sash") and int(out.item_consumed[0, 0]) == 0
    assert int(out.ai_item_known[0]) == 0
    # Without Endure the sash itself endures and goes.
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), _o())
    assert _hp(out, 0) == 1 and int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) != 0


def test_an_enduring_band_holder_draws_no_band_roll(tables, names):
    """bcp.c:3386-3389: the Focus Band roll is inside ``if (enduring == 0)``: the foe's
    ``focus_band`` value changes nothing, and the band is not named."""
    s = _endure(tables, names, hp=3, item="Focus Band")
    acts = A("Endure", "Tackle", s, names)
    outs = [
        assert_matches_reference(
            tables, s, acts, _o(focus_band=np.array([[0] * 5, [band] * 5], np.int8))
        )
        for band in (0, 1)
    ]
    for name, x, y in zip(outs[0]._fields, outs[0], outs[1], strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert _hp(outs[0], 0) == 1 and int(outs[0].ai_item_known[0]) == 0
    assert int(outs[0].item[0, 0]) == names.item("Focus Band")


def test_an_enduring_multi_hit_leaves_one_hp_after_every_hit(tables, names):
    s = _endure(tables, names, hp=3, opp="DoubleSlap")
    acts = A("Endure", "DoubleSlap", s, names)
    o = _o(n_hits=[2, 5])
    out = assert_matches_reference(tables, s, acts, o)
    assert _hp(out, 0) == 1
    assert _events(tables, s, acts, o)["hits"][1] == 5


def test_future_sight_is_not_endured(tables, names):
    """subscript_future_sight_damage.s:28 is CheckHoldOnWith1HP, which ignores Endure."""
    s = _endure(tables, names)
    s = s._replace(
        future_timer=s.future_timer.at[0].set(1),
        future_damage=s.future_damage.at[0].set(500),
        future_move=s.future_move.at[0].set(names.move("Future Sight")),
        future_attacker=s.future_attacker.at[0].set(2),
        future_sight=s.future_sight.at[0].set(1),
    )
    o = _o(future_hit=[1, 1])
    out = assert_matches_reference(tables, s, A("Endure", "Leer", s, names), o)
    assert _hp(out, 0) == 0


@pytest.mark.parametrize(
    ("opp", "order", "count", "item"),
    [
        ("Tackle", "player", 0, None),
        ("Tackle", "player", 1, None),
        ("Tackle", "opp", 1, None),
        ("Tackle", "player", 1, "Focus Band"),
        ("Double Kick", "player", 1, None),
    ],
)
def test_endure_enumeration_equals_bruteforce(tables, names, opp, order, count, item):
    """Endure shares Protect's roll (``Outcome.protect``) and its reach: the gated enumeration
    equals the brute-force one (the Band roll is a superset read the enduring path ignores)."""
    from tests.engine.helpers import drop_programs
    from tests.engine.test_enumerate import assert_equals_bruteforce

    drop_programs()  # the whole-engine programs of the tests above stay resident otherwise

    s = _endure(tables, names, opp=opp, order=order, count=count, opp_count=count, item=item)
    width, brute = assert_equals_bruteforce(tables, s, A("Endure", opp, s, names))
    assert width <= brute
