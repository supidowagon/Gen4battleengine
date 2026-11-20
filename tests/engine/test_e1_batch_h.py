"""E1 batch H, H-1: the Future Sight / Doom Desire hit's accuracy law where an Acid Rain write
(subscript_pursuit.s:115, OPCODE_SUB_TO_ZERO battle_script.c:3282-3287; ledger G-12) can happen:
classes over the union of every mask the hit can read (``outcome.future_thresholds``' field
axis, ``outcome._future_fields``), and the roll kept whole (``loose``) only on a battler whose hit
is due (bcp.c:1767-1769, :1782; accuracy 0 draws nothing, bcp.c:2900-2902)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.enumerate import _future_hit, hit_class_law, outcome_laws
from nuzline.engine.ids import Weather
from nuzline.engine.order import is_switch, move_effects
from nuzline.engine.outcome import acid_rain_pairs, future_thresholds, hit_class
from tests.engine.d2_states import four, with_moves
from tests.engine.helpers import A, assert_matches_reference, jit_step, sample_rows
from tests.engine.test_field_mask import _hits, _masked, _pursuit_fight, _rearming


def _pending(s, names, defender, attacker, timer=1):
    """``s`` with a Future Sight of battler ``attacker`` on battler ``defender``: count ``timer``,
    5 stored damage."""
    return s._replace(
        future_timer=s.future_timer.at[defender].set(timer),
        future_damage=s.future_damage.at[defender].set(5),
        future_move=s.future_move.at[defender].set(names.move("Future Sight")),
        future_attacker=s.future_attacker.at[defender].set(1 + attacker),
        future_sight=s.future_sight.at[defender].set(1),
    )


def _accuracy(tables, names):
    return int(tables.move_accuracy[names.move("Future Sight")])


def _law(tables, s, acts, side):
    laws = outcome_laws(tables, s, acts)
    assert not bool(laws.future_loose[side])
    return _future_hit(laws, side)


def _same(a, b):
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        assert np.array_equal(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize(
    "foe_action, start",
    [
        (0, Weather.HAIL),  # Growl: no chase
        (MOVE_ACTIONS + 1, Weather.HAIL),  # the switch's chase (subscript_switch_pokemon.s:11)
        (1, Weather.HAIL),  # U-turn's chase (subscript_attack_then_switch_out.s:54)
        (MOVE_ACTIONS + 1, Weather.NONE),  # 0 - 1 clamps to 0 at :115
    ],
)
def test_future_sight_law_is_classes_where_acid_rain_can_write(tables, names, foe_action, start):
    """T1. A due Future Sight on the foe (battler 1) while the player uses Pursuit: never kept
    whole for Acid Rain. On 0x80 the write leaves 0x7F (faintedMon 1) or 0x80 (faintedMon 0):
    no fog, no Gravity, so every row has the one threshold ``a`` (vanilla 90): the Growl row's
    law."""
    s = _pursuit_fight(
        tables, names, ("Pursuit",), ("Growl", "U-turn"), foe="Scyther", start_condition=start
    )
    s = _pending(s, names, 1, 0)
    acts = jnp.asarray([0, foe_action], jnp.int32)
    assert not bool(future_thresholds(tables, s, acts)[2][1])
    assert _law(tables, s, acts, 1) == hit_class_law([_accuracy(tables, names)])


def _fog_chase(tables, names, mask=0x8000):
    """The foe's 1 HP Geodude (battler 1) switches to the Geodude behind it while the player's
    Pursuit chases it, on ``mask``; a Future Sight of the player's is due on battler 1."""
    s = _pursuit_fight(tables, names, ("Pursuit",), ("Growl",))
    s = _pending(_masked(s.replace_hp(1, 0, 1), mask, 0), names, 1, 0)
    return s, A("Pursuit", MOVE_ACTIONS + 1, s, names)


def test_future_sight_law_on_deep_fog_covers_the_acid_rain_gravity(tables, names):
    """T2. 0x8000 (deep fog): a Pursuit KO of the switcher subtracts 1 (0x7FFF: Gravity 7, fog
    off; the turn end's count leaves 6, bcp.c:1229), so the hit reads x10/6 (bcp.c:2972-2974);
    no KO leaves the fog (x6/10, :2934); a newcomer's weather ability would clear the fog. Law
    over {a*6//10, a, a*10//6}: vanilla [1, 55, 91] / [54, 36, 10]. The step lands every roll
    after the KO and only rolls up to a*6//10 after a miss; each roll steps as its class."""
    a = _accuracy(tables, names)
    s, acts = _fog_chase(tables, names)
    assert _law(tables, s, acts, 1) == hit_class_law([a * 6 // 10, a, a * 10 // 6])
    thr, ok, _ = future_thresholds(tables, s, acts)
    for pursuit_hits, lands in ((1, lambda u: True), (0, lambda u: u <= a * 6 // 10)):
        base = _hits()._replace(hit=_hits().hit * pursuit_hits)
        missed = assert_matches_reference(tables, s, acts, base)
        assert int(missed.hp[1, 0]) == (0 if pursuit_hits else 1)
        assert int(missed.active[1]) == 1 and int(missed.hp[1, 1]) > 5  # survives the hail
        for u in (a * 6 // 10, a * 6 // 10 + 1, a, a + 1, 100):
            o = base._replace(future_hit=jnp.asarray([0, u], jnp.int8))
            out = assert_matches_reference(tables, s, acts, o)
            assert (int(out.hp[1, 1]) == int(missed.hp[1, 1]) - 5) is lands(u), (pursuit_hits, u)
            c = int(hit_class(u, thr[1], ok[1]))
            if c != u:
                o_c = o._replace(future_hit=jnp.asarray([0, c], jnp.int8))
                _same(jit_step(tables, s, acts, o_c), out)


def test_future_sight_law_on_deep_fog_covers_a_recoil_fainted_pursuer(tables, names):
    """T2 mirror. 0x8000: the player's 1 HP Monferno (battler 0) switches, the foe's 1 HP pursuer
    (Life Orb) KOs it and falls to its recoil (battle_lib.c:7357-7365): faintedMon is the
    pursuer, battler 1 (battle_script.c:1767-1770), 0x8000 - 1 = 0x7FFF. The foe's Future Sight
    due on battler 0 has the law over {a*6//10, a, a*10//6}; after the recoil it lands at a + 1."""
    a = _accuracy(tables, names)
    s = _pursuit_fight(tables, names, ("Growl",), ("Pursuit",), foe_item="Life Orb")
    s = _pending(_masked(s.replace_hp(0, 0, 1).replace_hp(1, 0, 1), 0x8000, 0), names, 0, 1)
    acts = A(MOVE_ACTIONS + 1, "Pursuit", s, names)
    assert _law(tables, s, acts, 0) == hit_class_law([a * 6 // 10, a, a * 10 // 6])
    left = []
    for u in (0, a + 1):
        o = _hits()._replace(future_hit=jnp.asarray([u, 0], jnp.int8))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
        left.append(int(out.hp[0, 1]))
    assert left[0] > 5 and left[1] == left[0] - 5


def test_future_sight_law_under_kaizo_gravity_follows_the_acid_rain_write(tables, names):
    """T3. Kaizo's Gravity rule (``gravity_rearm``, sub_seq 238, G-67) on 0x1000 (Gravity 1): no
    KO keeps it on (the count ends and re-arms, bcp.c:1227-1238), a KO of the switcher leaves
    0x0FFF (Gravity off). Law over {a, min(a*10//6, 100)}: roll a + 1 misses after the KO and
    lands without it. (The vanilla rule ends that Gravity: ``loose``.)"""
    t = _rearming(tables, "gravity_rearm")
    a = _accuracy(t, names)
    assert a < 100
    s, acts = _fog_chase(t, names, 0x1000)
    assert _law(t, s, acts, 1) == hit_class_law([a, min(a * 10 // 6, 100)])
    for pursuit_hits, lands in ((1, False), (0, True)):
        base = _hits()._replace(hit=_hits().hit * pursuit_hits)
        left = []
        for u in (0, a + 1):
            o = base._replace(future_hit=jnp.asarray([0, u], jnp.int8))
            left.append(int(assert_matches_reference(t, s, acts, o).hp[1, 1]))
        assert (left[1] == left[0] - 5) is lands


def test_only_a_due_hit_keeps_the_roll_whole(tables, names):
    """T4. An Embargo up keeps the hit's roll whole (B13-31), but only where the hit is due:
    with the count at 2 nothing is drawn this turn (bcp.c:1767-1769), so the stored value is the
    raw roll's class; at 1 it is the raw roll."""
    s = _pursuit_fight(tables, names, ("Growl",), ("Growl",))
    s = s._replace(embargo=s.embargo.at[0].set(1))
    acts = jnp.asarray([0, 0], jnp.int32)
    keys = jnp.stack([jax.random.PRNGKey(i) for i in range(8)])
    raw = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 28), (2,), 1, 101))(keys)
    for timer, loose in ((2, False), (1, True)):
        st = _pending(s, names, 1, 0, timer)
        thr, ok, whole = future_thresholds(tables, st, acts)
        assert bool(whole[1]) is loose
        stored = np.asarray(sample_rows(keys, tables, st, acts).future_hit)[:, 1]
        want = raw[:, 1] if loose else hit_class(raw[:, 1], thr[1], ok[1])
        assert np.array_equal(stored, np.asarray(want))


def test_four_battlers_keep_the_roll_whole_where_an_acid_pair_is_reachable(tables, names):
    """T5. B = 4: battler 2 switches while battler 3 (the other side) uses Pursuit. Only the pair
    [2, 3] is reachable; a due Future Sight is kept whole (a third battler can write the mask
    before the chase; the exact handling is the doubles milestone's), and without the Pursuit it
    is not."""
    s = with_moves(four(tables, names), names, 3, ("Pursuit",))
    s = _pending(s, names, 0, 1)
    acts = jnp.asarray([0, 0, MOVE_ACTIONS + 2, 0], jnp.int32)
    for pursuit in (names.move("Pursuit"), 0):
        moves = jnp.asarray([0, 0, 0, pursuit], jnp.int32)
        effect, _ = move_effects(tables, s, moves)
        uses = ~is_switch(acts)
        pairs = np.asarray(acid_rain_pairs(s, acts, uses, effect))
        want = np.zeros((4, 4), bool)
        want[2, 3] = bool(pursuit)
        assert np.array_equal(pairs, want)
        assert bool(future_thresholds(tables, s, acts, moves)[2][0]) is bool(pursuit)
