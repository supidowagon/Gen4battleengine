"""D3-11b: the Leech Seed recipient is the battler that used the move (brief D3-11b rules 1-6).

Decomp: subscript_leech_seed_start.s:14-15 (``BTLVAR_ATTACKER`` ORed into the defender's
``moveEffectsMask`` with the Leech Seed bit: ``MOVE_EFFECT_LEECH_SEED_RECIPIENT``, moves.h:8-10,
:40), battle_controller_player.c:1378-1391 (``MON_COND_CHECK_STATE_LEECH_SEED`` reads the CURRENT
occupant of the recipient's battler slot: ``battleMons[recipient].curHP``) and
subscript_leech_seed_effect.s:7-26 (the seeded battler loses ``maxHP / 8``, the recipient's Big
Root scales it, the recipient gains it). Cleared only by a switch without Baton Pass
(battle_lib.c:2071) and a faint (:2166); Baton Pass keeps both bits (:2076; moves.h:46); nothing
clears another battler's bits when the seeder leaves or faints (:2056-2098, :2168-2183).

The writer case plays the action phase (``play``: engine equal to reference in every field) and
then the end of the turn (``both``: engine equal to reference); the end-of-turn cases are ``both``
from a state that holds the seed. The switch itself is the reset of ``step._reset_battler`` and the
reference's, tested directly (a four-battler ``step`` is refused until D3-13: ``coverage``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine import step as step_module
from nuzline.engine.effects import big_root
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.format import SINGLES, format_of, party
from nuzline.engine.ids import AbilityId, Type
from nuzline.engine.state import battler_ability
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _reset_battler as ref_reset_battler
from tests.engine.d2_states import party_of, with_hp, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_end_of_turn import base, both, make_absent
from tests.strategies import supported_state


def max_hp(state, b):
    return int(state.max_hp[party_of(state, b), int(state.active[b])])


def drain(state, b):
    """D(b): ``max(maxHP // 8, 1)`` of b's active, capped at its HP."""
    return min(max(max_hp(state, b) // 8, 1), hp(state, b))


def seed(state, victim, seeder):
    """``victim`` is Leech Seeded by battler ``seeder`` (``leech_seed_recipient = 1 + seeder``)."""
    return state._replace(
        leech_seed=state.leech_seed.at[victim].set(1),
        leech_seed_recipient=state.leech_seed_recipient.at[victim].set(1 + seeder),
    )


def seeded_3_by_0(tables, names):
    """Battler 3 seeded by battler 0; battlers 0 and 2 below max HP (a gain shows)."""
    s = seed(base(tables, names), 3, 0)
    return with_hp(with_hp(s, 0, 10), 2, 10)


def non_grass(state, b):
    p, slot = party_of(state, b), int(state.active[b])
    normal = jnp.asarray([Type.NORMAL, Type.NORMAL], state.types.dtype)
    return state._replace(types=state.types.at[p, slot].set(normal))


def unchanged(out, s, battlers):
    for b in battlers:
        assert hp(out, b) == hp(s, b), b


# --- 1. the writer -----------------------------------------------------------------------------


def test_leech_seed_records_the_user_as_recipient(tables, names):
    """Battler 0 seeds battler 3, battler 2 seeds battler 1 (subscript_leech_seed_start.s:14-15);
    at the turn's end each seeded battler loses D(b) and its seeder's active gains it."""
    s = base(tables, names)
    for b in (0, 2):
        s = with_moves(s, names, b, ["Leech Seed"])
    for b in (1, 3):
        s = non_grass(with_moves(s, names, b, ["Swords Dance"]), b)
    s = with_hp(with_hp(s, 0, 10), 2, 10)
    out, _ = play(tables, s, [0, 0, 0, 0], (0, 1, 2, 3), targets=(3, 0, 1, 0))
    assert int(out.leech_seed[3]) == 1 and int(out.leech_seed_recipient[3]) == 1
    assert int(out.leech_seed[1]) == 1 and int(out.leech_seed_recipient[1]) == 3
    assert int(out.leech_seed_recipient[0]) == 0 and int(out.leech_seed_recipient[2]) == 0
    end = both(tables, out)
    assert hp(end, 3) == hp(out, 3) - drain(out, 3)
    assert hp(end, 0) == hp(out, 0) + drain(out, 3)
    assert hp(end, 1) == hp(out, 1) - drain(out, 1)
    assert hp(end, 2) == hp(out, 2) + drain(out, 1)


# --- 2-3. the recipient is the slot, partner included --------------------------------------------


def test_recipient_is_the_named_battler_not_the_opposite_number(tables, names):
    s = seeded_3_by_0(tables, names)
    out = both(tables, s)
    assert hp(out, 3) == hp(s, 3) - drain(s, 3)
    assert hp(out, 0) == hp(s, 0) + drain(s, 3)
    unchanged(out, s, (1, 2))


def test_the_seeders_partner_can_be_the_recipient(tables, names):
    """Battler 2 seeded by battler 0, its own partner (a single-target move may target the ally)."""
    s = with_hp(with_hp(seed(base(tables, names), 2, 0), 0, 10), 2, 60)
    out = both(tables, s)
    assert hp(out, 2) == 60 - drain(s, 2)
    assert hp(out, 0) == 10 + drain(s, 2)
    unchanged(out, s, (1, 3))


# --- 4-5. a seeder with no HP: no drain ------------------------------------------------------------


def test_a_fainted_seeder_not_yet_replaced_drains_nothing(tables, names):
    """The end of the turn runs before TurnEnd's ReplaceFainted (bcp.c:1848): the recipient slot
    holds a 0-HP Pokemon (bcp.c:1380), so nothing drains; the bits stay."""
    s = with_hp(seeded_3_by_0(tables, names), 0, 0)
    out = both(tables, s)
    unchanged(out, s, (0, 1, 2, 3))
    assert int(out.leech_seed[3]) == 1 and int(out.leech_seed_recipient[3]) == 1


def test_an_absent_seeder_drains_nothing(tables, names):
    s = make_absent(seeded_3_by_0(tables, names), 0)
    out = both(tables, s)
    assert hp(out, 3) == hp(s, 3)
    assert int(out.leech_seed[3]) == 1 and int(out.leech_seed_recipient[3]) == 1


# --- 6. the seeder switched out: the newcomer receives ------------------------------------------


def test_the_newcomer_of_the_seeders_slot_receives(tables, names):
    """Nothing clears the seeded battler's bits when the seeder leaves (battle_lib.c:2056-2098):
    battler 0 now holds bench slot 2 (below max HP), which gains the drain."""
    s = seeded_3_by_0(tables, names)
    s = s._replace(active=s.active.at[0].set(2), hp=s.hp.at[0, 2].set(10))
    out = both(tables, s)
    assert hp(out, 3) == hp(s, 3) - drain(s, 3)
    assert int(out.hp[0, 2]) == 10 + drain(s, 3)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert int(out.leech_seed_recipient[3]) == 1


# --- 7-8. the seeded battler leaves: both bits go, Baton Pass keeps them ---------------------------


def test_a_switch_of_the_seeded_battler_clears_both_bits(tables, names):
    s = seeded_3_by_0(tables, names)
    gone = step_module._reset_battler(s, 3, jnp.bool_(True), baton=False)
    assert int(gone.leech_seed[3]) == 0 and int(gone.leech_seed_recipient[3]) == 0
    out = both(tables, gone)
    unchanged(out, s, (0, 1, 2))  # no one is drained (or healed)


def test_baton_pass_keeps_both_bits(tables, names):
    s = seeded_3_by_0(tables, names)
    kept = step_module._reset_battler(s, 3, jnp.bool_(True), baton=True)
    assert int(kept.leech_seed[3]) == 1 and int(kept.leech_seed_recipient[3]) == 1
    rs = to_ref(s)
    ref_reset_battler(rs.battlers[3], baton=True)
    assert rs.battlers[3].leech_seed == 1 and rs.battlers[3].leech_seed_recipient == 1
    ref_reset_battler(rs.battlers[3], baton=False)
    assert rs.battlers[3].leech_seed == 0 and rs.battlers[3].leech_seed_recipient == 0
    plain = step_module._reset_battler(s, 3, jnp.bool_(True), baton=False)
    np.testing.assert_array_equal(
        np.asarray(from_ref(rs).leech_seed_recipient), np.asarray(plain.leech_seed_recipient)
    )


# --- 9. Big Root ----------------------------------------------------------------------------------


def test_the_recipients_big_root_scales_the_gain(tables, names):
    s = seeded_3_by_0(tables, names)
    s = s._replace(item=s.item.at[0, 0].set(names.item("Big Root")))
    out = both(tables, s)
    gain = drain(s, 3) * 130 // 100
    assert hp(out, 0) == min(10 + gain, max_hp(s, 0))
    assert hp(out, 3) == hp(s, 3) - drain(s, 3)


# --- 10. B = 2: every seed is the foe's, so the check is today's ------------------------------------


def _old_leech_seed_check(tables, state, b):
    """The formula before D3-11b: the recipient is the opposite number ``b ^ 1``."""
    fmt = format_of(state)
    o = b ^ 1
    mon, recipient = state.active[b], state.active[o]
    cur = state.hp[party(fmt, b), mon].astype(jnp.int32)
    guarded = battler_ability(state, b, mon) == AbilityId.MAGIC_GUARD
    on = (
        (state.leech_seed[b] != 0) & (cur > 0) & (state.hp[party(fmt, o), recipient] > 0) & ~guarded
    )
    lost = jnp.minimum(jnp.maximum(state.max_hp[party(fmt, b), mon].astype(jnp.int32) // 8, 1), cur)
    lost = jnp.where(on, lost, 0)
    gained = big_root(tables, state, o, lost)
    out = step_module._hp_change(state, b, mon, -lost)
    return step_module._hp_change(out, o, recipient, gained)


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_b2_recipient_is_the_foe_and_the_check_is_unchanged(tables, names, data):
    s = data.draw(supported_state(tables, names))
    assert format_of(s) == SINGLES
    for b in (0, 1):
        want = (1 + (b ^ 1)) * int(s.leech_seed[b])
        assert int(s.leech_seed_recipient[b]) == want
    for b in (0, 1):
        forced = s._replace(
            leech_seed=s.leech_seed.at[b].set(1),
            leech_seed_recipient=s.leech_seed_recipient.at[b].set(1 + (b ^ 1)),
        )
        new = step_module._leech_seed_check(tables, forced, b)
        old = _old_leech_seed_check(tables, forced, b)
        np.testing.assert_array_equal(np.asarray(new.hp), np.asarray(old.hp))


# --- the enumerator's Future Sight gate (bcp.c:1759-1764) -------------------------------------------


def test_an_absent_battler_with_a_stuck_future_sight_count_draws_no_hit(tables, names):
    """A battler in battlersSwitchingMask is skipped by the Future Sight pass (bcp.c:1759-1764):
    its count never fires, so ``future_hit`` / ``future_band`` are not relevant for it; a present
    battler with a count of 1 still is (B = 2: nobody is absent while the battle is live)."""
    s = base(tables, names)
    s = s._replace(future_timer=s.future_timer.at[2].set(1).at[3].set(1))
    acts = jnp.zeros(4, jnp.int32)
    live = outcome_laws(tables, s, acts).relevant
    assert bool(live.future_hit[2]) and bool(live.future_hit[3])
    gone = outcome_laws(tables, make_absent(s, 2), acts).relevant
    assert not bool(gone.future_hit[2]) and not bool(gone.future_band[2])
    assert bool(gone.future_hit[3])
