"""D3-11: the end of the turn over four battlers (sheet D3-11 rules 1-9, section 5).

Decomp: bcp.c:888-891 (the end-of-turn order is the BranchActions sort made at
``turnOrderCounter == maxBattlers``, row B of ``Outcome.sort_tie``; nothing sorts again before
the next TrainerMessage), :1085-1111 (Wish) and subscript_weather_continues.s:8-11 (the weather
loop), which do not skip a battler in ``battlersSwitchingMask``, :1305-1308 (CheckMonConditions),
:1759-1764 (Future Sight) and :1792-1797 (Perish Song), which do.

The end of the turn is played as ``step_events`` plays it after the actions: ``end_of_turn_order``,
``field_conditions``, ``end_of_turn``, ``_future_sight`` (``end_turn``), and as the reference's
``_end_turn`` does (``ref_end_turn``); every case asserts the two equal in every field. The
states are four-battler DOUBLES states: battlers 0 and 2 send from party 0, battlers 1 and 3 from
party 1; the Leech Seed recipient is the battler that used the move (``leech_seed_recipient``, D3-11b).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import absent_now
from nuzline.engine.format import SINGLES
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.order import PAIRS, player_second
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import (
    _future_sight,
    end_of_turn,
    end_of_turn_order,
    field_conditions,
)
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import (
    _end_of_turn,
    _field_conditions,
    _speed_order,
    _Turn,
)
from nuzline.reference.engine import (
    _future_sight as _ref_future_sight,
)
from tests.engine.d2_states import four, party_of, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_d3_action_loop import hp
from tests.engine.test_d3_turn_order import speeds
from tests.strategies import any_outcome, legal_action_pair, supported_state

PAIR_2_3 = PAIRS.index((2, 3))


def base(tables, names):
    s = four(tables, names)
    for b in range(4):
        s = with_moves(s, names, b, ["Tackle"])
    return s


@jax.jit
def end_turn(tables, state, outcome):
    """The end of the turn as ``step_events`` plays it after the actions."""
    sides = end_of_turn_order(tables, state, outcome)
    e = field_conditions(tables, state, sides)
    e = end_of_turn(tables, e, None, outcome, None, sides)
    e = _future_sight(tables, e, sides, outcome, jnp.zeros(state.active.shape[0], jnp.int32))
    return e


def ref_end_turn(tables, state, outcome):
    """The reference's ``_end_turn`` between the actions and ``SetupNextTurn``."""
    np_tables, rs, o = to_numpy(tables), to_ref(state), outcome_to_py(outcome)
    turn = _Turn(np_tables, o, entry_turn=rs.turn + 1, battler_count=len(rs.battlers))
    order = _speed_order(np_tables, rs, o)
    _field_conditions(turn, rs, order)
    _end_of_turn(turn, rs, order)
    _ref_future_sight(turn, rs, order)
    for sd in rs.battlers:
        sd.flinched = 0
    return from_ref(rs)


def both(tables, state, outcome=None):
    """The engine's end of the turn, asserted equal to the reference's in every field."""
    outcome = empty_outcome(_fmt(state)) if outcome is None else outcome
    out, ref = end_turn(tables, state, outcome), ref_end_turn(tables, state, outcome)
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def _fmt(state):
    from nuzline.engine.format import DOUBLES

    return DOUBLES if state.active.shape[0] == 4 else SINGLES


def tie_bits(pair: int, value: int = 1):
    bits = np.zeros(6, np.int8)
    bits[pair] = value
    return jnp.asarray(bits)


def with_sort_tie(outcome, row):
    return outcome._replace(sort_tie=outcome.sort_tie.at[4].set(row))


def poisoned(state, b):
    p, slot = party_of(state, b), int(state.active[b])
    return state._replace(status=state.status.at[p, slot].set(Status.POISON))


def make_absent(state, b):
    """Battler ``b``: 0 HP and nothing to send in (every other slot of its party is down, except
    the active slots of the party's other battlers)."""
    p = party_of(state, b)
    keep = {int(state.active[o]) for o in range(4) if o != b and party_of(state, o) == p}
    slots = jnp.arange(state.hp.shape[1])
    row = jnp.where(jnp.isin(slots, jnp.asarray(sorted(keep))), state.hp[p], 0)
    state = state._replace(hp=state.hp.at[p].set(row.astype(state.hp.dtype)))
    assert bool(absent_now(state, b))
    return state


def seeded_pair(tables, names, speed):
    """Battler 3 is Leech Seeded by battler 2 (``3 ^ 1``), battler 2 is poisoned with fewer HP than
    its poison damage (a drain heals it above that), ``speed`` the four Speed stats."""
    s = speeds(base(tables, names), speed)
    s = poisoned(s, 2)
    s = with_hp(s, 2, 5)
    return s._replace(
        leech_seed=s.leech_seed.at[3].set(1),
        leech_seed_recipient=s.leech_seed_recipient.at[3].set(3),  # seeded by battler 2
    )


def drain_and_poison(state):
    """Battler 3's drain and battler 2's poison damage (Divide(maxHP, 8), at least 1)."""
    drain = max(int(state.max_hp[party_of(state, 3), int(state.active[3])]) // 8, 1)
    poison = max(int(state.max_hp[party_of(state, 2), int(state.active[2])]) // 8, 1)
    assert drain > 5 and drain >= poison
    return drain, poison


# --- the order (rule 1) ---------------------------------------------------------------------


def test_order_is_the_sort_of_the_state_after_the_actions(tables, names):
    s = speeds(base(tables, names), (100, 300, 200, 250))
    o = empty_outcome(_fmt(s))
    assert np.asarray(end_of_turn_order(tables, s, o)).tolist() == [1, 3, 2, 0]
    # A Speed drop of the turn (battler 3 at -6) is read: the order is the sort after it.
    lowered = s._replace(stages=s.stages.at[3, Stat.SPE].set(-6))
    assert np.asarray(end_of_turn_order(tables, lowered, o)).tolist() == [1, 2, 0, 3]


def test_equal_speeds_follow_the_bit_of_the_pair_in_row_b(tables, names):
    s = speeds(base(tables, names), (300, 250, 100, 100))
    o = empty_outcome(_fmt(s))
    assert np.asarray(end_of_turn_order(tables, s, o)).tolist() == [0, 1, 2, 3]
    swapped = with_sort_tie(o, tie_bits(PAIR_2_3))
    assert np.asarray(end_of_turn_order(tables, s, swapped)).tolist() == [0, 1, 3, 2]
    # Another row of sort_tie is not the end of the turn's (rows 0-3: the acting battler's).
    other = o._replace(sort_tie=o.sort_tie.at[2].set(tie_bits(PAIR_2_3)))
    assert np.asarray(end_of_turn_order(tables, s, other)).tolist() == [0, 1, 2, 3]


def test_a_fainted_battler_goes_last(tables, names):
    s = with_hp(speeds(base(tables, names), (300, 250, 200, 100)), 0, 0)
    order = np.asarray(end_of_turn_order(tables, s, empty_outcome(_fmt(s)))).tolist()
    assert order == [1, 2, 3, 0]


# --- the mon conditions follow that order (rules 1, 3, 6) -------------------------------------


def test_faster_poisoned_seeder_faints_before_the_drain(tables, names):
    """Battler 2 (the seeder of battler 3) is faster: its poison faints it, then battler 3's Leech
    Seed does nothing (the recipient has no HP, bcp.c:1380)."""
    s = seeded_pair(tables, names, (100, 200, 300, 150))
    drain_and_poison(s)
    out = both(tables, s)
    assert hp(out, 2) == 0
    assert hp(out, 3) == hp(s, 3)


def test_slower_poisoned_seeder_is_healed_first(tables, names):
    """Battler 3 is faster: its drain heals battler 2 first, and the poison leaves it alive."""
    s = seeded_pair(tables, names, (100, 300, 200, 250))
    drain, poison = drain_and_poison(s)
    out = both(tables, s)
    assert hp(out, 3) == hp(s, 3) - drain
    assert hp(out, 2) == 5 + drain - poison


def test_equal_speed_seeder_pair_follows_the_row_b_bit(tables, names):
    """Battlers 2 and 3 at equal Speed: pair (2, 3)'s bit of row B puts battler 2 (0) or battler 3
    (1) first."""
    s = seeded_pair(tables, names, (400, 300, 100, 100))
    drain, poison = drain_and_poison(s)
    o = empty_outcome(_fmt(s))
    first_2 = both(tables, s, with_sort_tie(o, tie_bits(PAIR_2_3, 0)))
    assert hp(first_2, 2) == 0
    first_3 = both(tables, s, with_sort_tie(o, tie_bits(PAIR_2_3, 1)))
    assert hp(first_3, 2) == 5 + drain - poison


def test_every_battler_pays_its_residual(tables, names):
    """Four poisoned battlers all lose Divide(maxHP, 8): the loop runs over B positions."""
    s = speeds(base(tables, names), (100, 300, 200, 250))
    for b in range(4):
        s = poisoned(s, b)
    out = both(tables, s)
    for b in range(4):
        assert hp(out, b) == hp(s, b) - max(int(s.max_hp[party_of(s, b), int(s.active[b])]) // 8, 1)


def test_uproar_wakes_battlers_1_and_3(tables, names):
    """Battler 0 in an uproar: battlers 1 and 3, asleep, wake (rule 3)."""
    s = base(tables, names)
    s = s._replace(uproar=s.uproar.at[0].set(3))
    for b in (1, 3):
        p, slot = party_of(s, b), int(s.active[b])
        s = s._replace(
            status=s.status.at[p, slot].set(Status.SLEEP),
            status_counter=s.status_counter.at[p, slot].set(4),
        )
    out = both(tables, s)
    for b in (1, 3):
        assert int(out.status[party_of(out, b), int(out.active[b])]) == Status.NONE
    assert int(out.uproar[0]) == 2


# --- the field conditions (rule 2) --------------------------------------------------------------


def test_weather_chips_all_four_battlers(tables, names):
    """The weather loop runs over every battler of the order."""
    s = base(tables, names)._replace(weather=jnp.int8(Weather.SANDSTORM))
    s = speeds(s, (100, 300, 200, 250))
    out = both(tables, s)
    for b in range(4):
        assert hp(out, b) < hp(s, b)


def test_wish_runs_over_four_and_counts_an_absent_battler_down(tables, names):
    """Wish has no absent skip (bcp.c:1086-1100): an absent battler's count drops and nothing
    heals it; a living one at 1 heals half its max HP."""
    s = make_absent(base(tables, names), 2)
    s = with_hp(s, 3, 10)
    s = s._replace(wish_timer=s.wish_timer.at[2].set(2).at[3].set(1))
    out = both(tables, s)
    assert int(out.wish_timer[2]) == 1 and hp(out, 2) == 0
    assert int(out.wish_timer[3]) == 0 and hp(out, 3) > 10


# --- the side conditions (rule 4, 7) --------------------------------------------------------------


def _future(state, b, attacker, damage=5):
    return state._replace(
        future_timer=state.future_timer.at[b].set(1),
        future_damage=state.future_damage.at[b].set(damage),
        future_attacker=state.future_attacker.at[b].set(attacker + 1),
        future_sight=state.future_sight.at[b & 1].set(1),
    )


def test_absent_battler_keeps_its_future_sight_and_perish_counts(tables, names):
    """Battler 2 absent: its Future Sight count stays (bcp.c:1761) and its Perish Song count stays
    (bcp.c:1794); a living battler's count drops."""
    s = make_absent(base(tables, names), 2)
    s = s._replace(
        future_timer=s.future_timer.at[2].set(2).at[3].set(2),
        perish_song=s.perish_song.at[2].set(1).at[3].set(1),
        perish_song_turns=s.perish_song_turns.at[2].set(2).at[3].set(2),
    )
    out = both(tables, s)
    assert int(out.future_timer[2]) == 2 and int(out.future_timer[3]) == 1
    assert int(out.perish_song_turns[2]) == 2 and int(out.perish_song_turns[3]) == 1


def test_future_sight_hits_battlers_1_and_3_in_the_order(tables, names):
    """Both hit when the rolls allow, the faster battler first."""
    move = names.move("Future Sight")
    s = speeds(base(tables, names), (100, 300, 100, 250))
    s = _future(_future(s, 1, 0), 3, 2)
    s = s._replace(future_move=s.future_move.at[1].set(move).at[3].set(move))
    o = empty_outcome(_fmt(s))._replace(future_hit=jnp.asarray([1, 1, 1, 1], jnp.int8))
    out = both(tables, s, o)
    assert hp(out, 1) == hp(s, 1) - 5 and hp(out, 3) == hp(s, 3) - 5


def test_future_sight_second_hit_inherits_the_first_ones_miss(tables, names):
    """``moveStatusFlags`` carries from one hit of the pass to the next (rule 7): battler 1 is
    first and misses, so battler 3's hit, whose own roll would land, misses too; the other way
    round, battler 3 lands and battler 1's own miss stands."""
    move = names.move("Future Sight")
    s = _future(_future(base(tables, names), 1, 0), 3, 2)
    s = s._replace(future_move=s.future_move.at[1].set(move).at[3].set(move))
    o = empty_outcome(_fmt(s))._replace(future_hit=jnp.asarray([1, 0, 1, 1], jnp.int8))
    first_1 = both(tables, speeds(s, (100, 300, 100, 250)), o)
    assert hp(first_1, 1) == hp(s, 1) and hp(first_1, 3) == hp(s, 3)
    first_3 = both(tables, speeds(s, (100, 250, 100, 300)), o)
    assert hp(first_3, 3) == hp(s, 3) - 5 and hp(first_3, 1) == hp(s, 1)


# --- B = 2 -----------------------------------------------------------------------------------------


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_b2_end_of_turn_is_unchanged(tables, names, data):
    """B = 2: the order is the static ``player_second`` pair, the end-of-turn loops run over
    ``range(2)``, and the whole turn equals the reference on 20 drawn states."""
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome(SINGLES))
    first = player_second(tables, s, o.end_speed_tie, o.quick_claw).astype(jnp.int32)
    assert np.asarray(end_of_turn_order(tables, s, o)).tolist() == [int(first), int(first) ^ 1]
    assert_matches_reference(tables, s, acts, o)
