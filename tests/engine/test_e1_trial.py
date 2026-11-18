"""E1 trial batch (rulings G-63 .. , docs/kaizo/e1-mechanics-ledger.md): Plus and Minus
(``BattleSystem_CalcMoveDamage``, battle_lib.c:6802-6809), Endure (G-65, ``turnFlags.enduring``,
battle_controller_player.c:3386-3404), each case checked against the reference engine."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import gravity_banned, legal_actions, trapped, usable_moves
from nuzline.engine.coverage import IMPLEMENTED_ABILITIES, Support
from nuzline.engine.damage import damage_table, side_ability_count
from nuzline.engine.ids import (
    GRAVITY_MOVES,
    GRAVITY_TURNS,
    MAGNET_RISE_TURNS,
    AbilityId,
    EffectId,
    SemiKind,
    Type,
)
from nuzline.engine.order import copycat_call, metronome_candidates
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import ACT_GRAVITY
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import ref_damage_table
from tests.engine.helpers import A, assert_matches_reference, drop_programs, roark_state
from tests.engine.test_pivot import _mon, _resume
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
    drop_programs()  # the programs of the earlier tests stay resident otherwise
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
    from tests.engine.test_enumerate import assert_equals_bruteforce

    drop_programs()  # the whole-engine programs of the tests above stay resident otherwise

    s = _endure(tables, names, opp=opp, order=order, count=count, opp_count=count, item=item)
    width, brute = assert_equals_bruteforce(tables, s, A("Endure", opp, s, names))
    assert width <= brute


# --- Magnet Rise (G-66) --------------------------------------------------------------------

MAGNET_MOVES = ("Magnet Rise", "Protect", "Scratch", "Swords Dance")


def _magnet(tables, names, *, turns=0, bit=None, opp="Earthquake", **kw):
    """Monferno (Magnet Rise first) against Geodude (``opp`` first); ``turns`` / ``bit`` are
    the player's ``magnet_rise_turns`` / ``magnet_rise`` (the bit defaults to ``turns > 0``)."""
    s = _state(tables, names, mine=MAGNET_MOVES, opp=opp, **kw)
    bit = turns > 0 if bit is None else bit
    return s._replace(
        magnet_rise=s.magnet_rise.at[0].set(int(bit)),
        magnet_rise_turns=s.magnet_rise_turns.at[0].set(turns),
    )


def test_magnet_rise_is_mapped(tables, names, kaizo_tables, kaizo_names):
    from tools.effect_map import DECOMP_EFFECT_TO_ID

    assert DECOMP_EFFECT_TO_ID[252] == (EffectId.MAGNET_RISE, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Magnet Rise")]) == EffectId.MAGNET_RISE
    assert MAGNET_RISE_TURNS == 5


def test_magnet_rise_sets_the_bit_and_five_turns_then_counts_down(tables, names):
    drop_programs()  # the programs of the earlier tests stay resident otherwise
    s = _magnet(tables, names, opp="Leer")
    acts = A("Magnet Rise", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    # Set to 5 by the move, then the turn's MON_COND_CHECK_STATE_MAGNET_RISE counts it once.
    assert int(out.magnet_rise[0]) == 1 and int(out.magnet_rise_turns[0]) == MAGNET_RISE_TURNS - 1
    assert int(out.magnet_rise[1]) == 0 and int(out.magnet_rise_turns[1]) == 0
    ev = _events(tables, s, acts, _o())
    assert ev["move_failed"].tolist() == [0, 0]
    # Counted from 5 to 0 over five turn ends: the bit clears with the last one.
    left = out
    for turns in (3, 2, 1):
        left = assert_matches_reference(tables, left, A("Swords Dance", "Leer", left, names), _o())
        assert (int(left.magnet_rise[0]), int(left.magnet_rise_turns[0])) == (1, turns)
    left = assert_matches_reference(tables, left, A("Swords Dance", "Leer", left, names), _o())
    assert (int(left.magnet_rise[0]), int(left.magnet_rise_turns[0])) == (0, 0)


def test_magnet_rise_fails_with_turns_levitate_or_ingrain(tables, names):
    cases = {
        "turns": _magnet(tables, names, turns=3, opp="Leer"),
        "levitate": _ability_player(_magnet(tables, names, opp="Leer"), AbilityId.LEVITATE),
    }
    ingrained = _magnet(tables, names, opp="Leer")
    cases["ingrain"] = ingrained._replace(ingrain=ingrained.ingrain.at[0].set(1))
    # The turns, not the bit, decide: a stale bit without turns does not fail it.
    stale = _magnet(tables, names, turns=0, bit=True, opp="Leer")
    out = assert_matches_reference(tables, stale, A("Magnet Rise", "Leer", stale, names), _o())
    assert int(out.magnet_rise_turns[0]) == MAGNET_RISE_TURNS - 1
    for name, s in cases.items():
        out = assert_matches_reference(tables, s, A("Magnet Rise", "Leer", s, names), _o())
        # The script failed: nothing was set (a success would leave 4 turns and the bit).
        assert int(out.magnet_rise_turns[0]) == max(int(s.magnet_rise_turns[0]) - 1, 0), name
        assert int(out.magnet_rise[0]) == int(s.magnet_rise[0]) * int(
            out.magnet_rise_turns[0] > 0
        ), name


def _ability_player(s, ability):
    return s._replace(ability=s.ability.at[0, int(s.active[0])].set(ability))


def test_magnet_rise_blocks_ground_moves(tables, names):
    """battle_lib.c:2605-2609: MOVE_STATUS_MAGNET_RISE, part of MOVE_STATUS_IMMUNE; Iron Ball,
    Ingrain lift it, Mold Breaker does not."""
    acts = lambda s: A("Swords Dance", "Earthquake", s, names)
    hit = _magnet(tables, names, opp="Earthquake")  # no turns: a normal hit
    base = assert_matches_reference(tables, hit, acts(hit), _o())
    assert _hp(base, 0) < _hp(hit, 0)
    safe = _magnet(tables, names, turns=3, opp="Earthquake")
    out = assert_matches_reference(tables, safe, acts(safe), _o())
    assert _hp(out, 0) == _hp(safe, 0)  # the hit is an immunity
    # Mold Breaker does not lift it.
    mb = safe._replace(ability=safe.ability.at[1, int(safe.active[1])].set(AbilityId.MOLD_BREAKER))
    assert _hp(assert_matches_reference(tables, mb, acts(mb), _o()), 0) == _hp(mb, 0)
    # An Iron Ball and Ingrain (after Magnet Rise) lift it.
    iron = safe._replace(item=safe.item.at[0, 0].set(names.item("Iron Ball")))
    assert _hp(assert_matches_reference(tables, iron, acts(iron), _o()), 0) < _hp(iron, 0)
    rooted = safe._replace(ingrain=safe.ingrain.at[0].set(1))
    assert _hp(assert_matches_reference(tables, rooted, acts(rooted), _o()), 0) < _hp(rooted, 0)
    # Other types are untouched.
    tackle = _magnet(tables, names, turns=3, opp="Tackle")
    out = assert_matches_reference(tables, tackle, A("Swords Dance", "Tackle", tackle, names), _o())
    assert _hp(out, 0) < _hp(tackle, 0)


def test_a_switch_clears_magnet_rise_and_baton_pass_keeps_it(tables, names):
    n = names
    s = _magnet(tables, names, turns=3, opp="Leer")
    out = assert_matches_reference(tables, s, A(4 + 1, "Leer", s, names), _o())
    assert (int(out.magnet_rise[0]), int(out.magnet_rise_turns[0])) == (0, 0)
    # Baton Pass hands the bit and the turns over (moves.h:42-44, battle_lib.c:2112); the
    # resumed turn's end counts the newcomer's turns once.
    s = make_state(
        tables,
        [_mon(n, "Ninjask", ["Baton Pass"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Leer"], 20)],
    )
    s = s._replace(
        magnet_rise=s.magnet_rise.at[0].set(1), magnet_rise_turns=s.magnet_rise_turns.at[0].set(4)
    )
    paused = assert_matches_reference(tables, s, (0, 0), empty_outcome())
    after = _resume(tables, paused, 1)
    assert (int(after.magnet_rise[0]), int(after.magnet_rise_turns[0])) == (1, 3)
    # Spikes miss a Baton-Passed newcomer with the bit (subscript_hazards_check.s:11), and hit
    # one without it.
    spiked = s._replace(spikes=s.spikes.at[0].set(1))
    kept = _resume(tables, assert_matches_reference(tables, spiked, (0, 0), empty_outcome()), 1)
    assert int(kept.hp[0, 1]) == int(spiked.hp[0, 1])
    bare = spiked._replace(magnet_rise=spiked.magnet_rise.at[0].set(0))
    hurt = _resume(tables, assert_matches_reference(tables, bare, (0, 0), empty_outcome()), 1)
    assert int(hurt.hp[0, 1]) < int(bare.hp[0, 1])


def test_arena_trap_reads_the_turns_of_magnet_rise(tables, names):
    s = _magnet(tables, names, opp="Leer")
    s = s._replace(ability=s.ability.at[1, int(s.active[1])].set(AbilityId.ARENA_TRAP))
    s = _ability_player(s, AbilityId.NONE)
    assert bool(trapped(tables, s, 0))
    floating = s._replace(magnet_rise_turns=s.magnet_rise_turns.at[0].set(2))
    assert not bool(trapped(tables, floating, 0))
    # The bit alone (turns zeroed, as by Gravity) does not free it: BattlerIsGrounded reads the
    # turns (battle_lib.c:5503).
    bit_only = s._replace(magnet_rise=s.magnet_rise.at[0].set(1))
    assert bool(trapped(tables, bit_only, 0))


@pytest.mark.parametrize(
    ("order", "turns"),
    [("player", 3), ("opp", 3), ("player", 0)],
)
def test_magnet_rise_enumeration_equals_bruteforce(tables, names, order, turns):
    from tests.engine.test_enumerate import assert_equals_bruteforce

    drop_programs()
    s = _magnet(tables, names, turns=turns, opp="Earthquake", order=order)
    width, brute = assert_equals_bruteforce(tables, s, A("Magnet Rise", "Earthquake", s, names))
    assert width <= brute


# --- Gravity (G-67) ------------------------------------------------------------------------

GRAVITY_PLAYER = ("Gravity", "Magnet Rise", "Fly", "Scratch")
PLAIN = ("Scratch", "Leer", "Growl", "Tackle")


def _gravity(tables, names, *, turns=0, mine=GRAVITY_PLAYER, opp="Leer", **kw):
    """Monferno (``mine``) against Geodude (``opp`` first); ``turns`` is the field's Gravity."""
    s = _state(tables, names, mine=mine, opp=opp, **kw)
    return s._replace(gravity=jnp.asarray(turns, s.gravity.dtype))


def test_gravity_is_mapped_and_pinned(tables, names, kaizo_tables, kaizo_names):
    from tools.effect_map import DECOMP_EFFECT_TO_ID

    assert DECOMP_EFFECT_TO_ID[215] == (EffectId.GRAVITY, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Gravity")]) == EffectId.GRAVITY
        assert [n.move(m) for m in ("Fly", "Bounce", "Jump Kick", "Hi Jump Kick")] == list(
            GRAVITY_MOVES[:4]
        )
        assert n.move("Magnet Rise") == GRAVITY_MOVES[5]
    # id 150 (the table's Splash) is "Hidden Power Water" in Kaizo: the ban is by id.
    assert names.move("Splash") == GRAVITY_MOVES[4]
    assert kaizo_names.move_names[GRAVITY_MOVES[4]] == "Hidden Power Water"
    # sub_seq 238: Kaizo's subscript_gravity_end re-arms Gravity, vanilla's only prints.
    assert (int(tables.gravity_rearm), int(kaizo_tables.gravity_rearm)) == (0, 1)
    assert int(kaizo_tables.move_priority[kaizo_names.move("Gravity")]) == 1
    assert int(tables.move_priority[names.move("Gravity")]) == 0


def test_gravity_starts_for_five_turns_and_fails_when_up(tables, names):
    drop_programs()  # the programs of the earlier tests stay resident otherwise
    s = _gravity(tables, names)
    out = assert_matches_reference(tables, s, A("Gravity", "Leer", s, names), _o())
    # Set to 5 by the move, then the turn's field-condition pass counts it once.
    assert int(out.gravity) == GRAVITY_TURNS - 1
    up = _gravity(tables, names, turns=3)
    out = assert_matches_reference(tables, up, A("Gravity", "Leer", up, names), _o())
    assert int(out.gravity) == 2  # failed: only the countdown ran (not reset to 4)


def test_gravity_lifts_the_flying_and_levitate_immunities(tables, names):
    """battle_lib.c:2545-2549 (BasicTypeMulApplies) and :3095-3098 (Battler_Ability)."""
    flying = Type.FLYING
    base = _gravity(tables, names, opp="Earthquake", mine=("Splash", "Scratch", "Fly", "Gravity"))
    base = base._replace(types=base.types.at[0, 0].set(jnp.asarray([flying, flying], jnp.int8)))
    acts = A("Splash", "Earthquake", base, names)
    safe = assert_matches_reference(tables, base, acts, _o())
    assert _hp(safe, 0) == _hp(base, 0)  # the Ground hit is an immunity
    up = base._replace(gravity=jnp.asarray(3, base.gravity.dtype))
    # (Splash is banned under Gravity: the player uses a legal move instead.)
    hit = assert_matches_reference(tables, up, A("Scratch", "Earthquake", up, names), _o())
    assert _hp(hit, 0) < _hp(up, 0)
    # A Levitate holder: immune without Gravity, hit with it.
    lev = _ability_player(
        _gravity(tables, names, opp="Earthquake", mine=("Splash", "Scratch", "Fly", "Gravity")),
        AbilityId.LEVITATE,
    )
    assert _hp(assert_matches_reference(tables, lev, acts, _o()), 0) == _hp(lev, 0)
    lev_up = lev._replace(gravity=jnp.asarray(3, lev.gravity.dtype))
    out = assert_matches_reference(tables, lev_up, A("Scratch", "Earthquake", lev_up, names), _o())
    assert _hp(out, 0) < _hp(lev_up, 0)


def test_spikes_hit_a_flying_newcomer_under_gravity(tables, names):
    s = _gravity(tables, names, turns=3, opp="Leer", mine=PLAIN)
    flying = jnp.asarray([Type.FLYING, Type.FLYING], jnp.int8)
    s = s._replace(types=s.types.at[0, 1].set(flying), spikes=s.spikes.at[0].set(1))
    out = assert_matches_reference(tables, s, A(4 + 1, "Leer", s, names), _o())
    assert int(out.hp[0, 1]) < int(s.hp[0, 1])
    calm = s._replace(gravity=jnp.asarray(0, s.gravity.dtype))
    out = assert_matches_reference(tables, calm, A(4 + 1, "Leer", calm, names), _o())
    assert int(out.hp[0, 1]) == int(calm.hp[0, 1])  # a Flying type is off the ground


def test_gravity_grounds_a_magnet_rise_user_and_unlocks_a_fly_user(tables, names):
    """subscript_gravity_start.s:13-72: the turns go to 0 and the bit stays; a Fly user's lock and
    semi-invulnerable bit are cleared (``Battler_UnlockMoveChoice``)."""
    s = _gravity(tables, names, opp="Gravity", order="opp")
    s = s._replace(
        magnet_rise=s.magnet_rise.at[0].set(1), magnet_rise_turns=s.magnet_rise_turns.at[0].set(4)
    )
    out = assert_matches_reference(tables, s, A("Scratch", "Gravity", s, names), _o())
    assert int(out.gravity) == GRAVITY_TURNS - 1
    assert (int(out.magnet_rise[0]), int(out.magnet_rise_turns[0])) == (1, 0)
    fly = names.move("Fly")
    f = _gravity(tables, names, opp="Gravity", order="opp")
    f = f._replace(
        semi_invulnerable=f.semi_invulnerable.at[0].set(SemiKind.AIRBORNE),
        locked_move=f.locked_move.at[0].set(fly),
    )
    out = assert_matches_reference(tables, f, A("Fly", "Gravity", f, names), _o())
    assert int(out.semi_invulnerable[0]) == 0 and int(out.locked_move[0]) == 0
    assert int(out.hp[1, 0]) == int(f.hp[1, 0])  # the banned Fly did not strike


def test_the_gravity_move_ban(tables, names):
    banned = ("Fly", "Splash", "Jump Kick", "Hi Jump Kick")
    s = _gravity(tables, names, turns=3, mine=banned)
    assert not bool(jnp.any(usable_moves(tables, s, 0)))  # all invalid: Struggle (bcp.c:433)
    assert bool(legal_actions(tables, s, 0)[0])  # action 0 is Struggle
    free = s._replace(gravity=jnp.asarray(0, s.gravity.dtype))
    assert bool(jnp.all(usable_moves(tables, free, 0)))
    assert bool(jnp.all(gravity_banned(s, jnp.asarray(GRAVITY_MOVES))))
    assert not bool(jnp.any(gravity_banned(free, jnp.asarray(GRAVITY_MOVES))))
    # A move chosen before a faster foe's Gravity fails at the status check (a message only).
    c = _gravity(tables, names, opp="Gravity", order="opp")
    acts = A("Magnet Rise", "Gravity", c, names)
    out = assert_matches_reference(tables, c, acts, _o())
    assert int(out.magnet_rise[0]) == 0 and int(out.magnet_rise_turns[0]) == 0
    ev = _events(tables, c, acts, _o())
    assert int(ev["act_kind"][0]) == ACT_GRAVITY and int(ev["act_id"][0]) == names.move(
        "Magnet Rise"
    )
    assert ev["move_failed"].tolist() == [1, 0]


def test_metronome_and_copycat_never_call_a_banned_move(tables, names):
    s = _gravity(tables, names, turns=3, mine=PLAIN)
    calm = s._replace(gravity=jnp.asarray(0, s.gravity.dtype))
    for move in GRAVITY_MOVES:
        assert not bool(metronome_candidates(tables, s, 0)[move]), move
        assert int(copycat_call(tables, move, s)) == 0, move
    assert int(copycat_call(tables, names.move("Tackle"), s)) == names.move("Tackle")
    # Not every banned move is Metronome-able without Gravity, but the ones that are come back.
    back = [m for m in GRAVITY_MOVES if bool(metronome_candidates(tables, calm, 0)[m])]
    assert back, "no Gravity move is a Metronome candidate without Gravity"


def test_vanilla_gravity_ends_after_five_turn_ends(tables, names):
    s = _gravity(tables, names, turns=GRAVITY_TURNS, mine=PLAIN)
    for left in (4, 3, 2, 1, 0, 0):
        s = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), _o())
        assert int(s.gravity) == left


def test_kaizo_gravity_re_arms_for_ever(kaizo_tables, kaizo_names):
    """sub_seq 238: at 0 the counter is set to 5 again and the grounding loop runs."""
    drop_programs()  # the programs of the earlier tests stay resident otherwise
    from tests.engine.test_enumerate import _kaizo_spec

    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t,
        [_kaizo_spec(n, "Wormadam", "Leer", 20)],
        [_kaizo_spec(n, "Skitty", "Growl", 20)],
        gravity=GRAVITY_TURNS,
    )
    assert int(s.gravity) == GRAVITY_TURNS  # sub_seq 297: the fight starts with Gravity up
    s = s._replace(
        magnet_rise=s.magnet_rise.at[0].set(1), magnet_rise_turns=s.magnet_rise_turns.at[0].set(9)
    )
    seen = []
    for _turn in range(6):
        s = assert_matches_reference(t, s, A("Leer", "Growl", s, n), _o())
        seen.append(int(s.gravity))
    assert seen == [4, 3, 2, 1, GRAVITY_TURNS, 4]
    assert int(s.magnet_rise_turns[0]) == 0  # zeroed by the first start at 0 or the loop


def test_gravity_move_ids_match_the_decomp():
    """``ids.GRAVITY_MOVES`` is ``sMovesAffectedByGravity`` (battle_lib.c:3385-3392) by decomp id."""
    import re
    from pathlib import Path

    decomp = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"
    moves = (decomp / "generated" / "moves.txt").read_text().split()
    lib = (decomp / "src" / "battle" / "battle_lib.c").read_text(encoding="utf-8")
    table = lib[lib.index("sMovesAffectedByGravity[] = {") :]
    listed = re.findall(r"(MOVE_\w+),", table[: table.index("};")])
    assert [moves.index(m) for m in listed] == list(GRAVITY_MOVES)
