"""The death record and traced budget fields (plan 4b task 2): ``deaths`` per player party
slot, ``faint_budget``, ``protected``, ``replace_mul``, ``with_budget`` and
``decision_actions``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, decision_actions, legal_actions
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import LAYOUT, MAX_DEATH_BUDGET, BattleState
from nuzline.engine.step import NORMAL_POWER_MUL, needs_replacement, replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace, ref_step
from nuzline.search.playout import finish_turn, playout
from tests.engine.helpers import jit_step, roark_real_state, roark_state
from tests.strategies import any_outcome, legal_action_pair, supported_state

NEW_FIELDS = ("deaths", "faint_budget", "protected", "replace_mul")
# Fields added after them (E2 batch 6: the turn scratch, 0 between turns, docs/kaizo/
# e2-mechanics-ledger.md "Batch 6 design"): all 0 in a fresh state too.
LATER_FIELDS = tuple(
    BattleState._fields[
        BattleState._fields.index("replace_mul") + 1 : BattleState._fields.index("chosen_target")
    ]
)
# D2's fields after the scratch (ruling D2-38): `chosen_target` starts at b ^ 1 (D2-1; the
# game's 0 is never read, every battler selects on turn 1, bcp.c:507-531), `last_hit_by` at 0.
D2_FIELDS = ("chosen_target", "last_hit_by")


def test_new_fields_default_to_zero(tables, names):
    s = roark_state(tables, names)
    assert LATER_FIELDS and BattleState._fields[-len(LATER_FIELDS + D2_FIELDS) :] == (
        LATER_FIELDS + D2_FIELDS
    )
    for name in NEW_FIELDS + LATER_FIELDS + ("last_hit_by",):
        assert not np.any(np.asarray(getattr(s, name))), name
    b = np.arange(len(s.chosen_target))
    assert np.asarray(s.chosen_target).tolist() == (b ^ 1).tolist()
    start = BattleState._fields.index("deaths")
    assert BattleState._fields[start : start + 4] == NEW_FIELDS
    assert all(name == "pivot_battler" or name.startswith("ts_") for name in LATER_FIELDS)
    assert LAYOUT["deaths"] == ((6,), np.int8)
    assert LAYOUT["faint_budget"] == ((), np.int8)
    assert LAYOUT["protected"] == ((6,), np.int8)
    assert LAYOUT["replace_mul"] == ((), np.int8)
    assert MAX_DEATH_BUDGET == 3


def test_with_budget(tables, names):
    s = roark_state(tables, names).with_budget(k=2, protected=[1, 3])
    assert int(s.faint_budget) == 2
    assert np.asarray(s.protected).tolist() == [0, 1, 0, 1, 0, 0]
    assert int(roark_state(tables, names).with_budget().faint_budget) == 0
    base = roark_state(tables, names)
    for bad in (4, -1):
        with pytest.raises(ValueError, match="budget"):
            base.with_budget(k=bad)
    with pytest.raises(ValueError, match="slot 6"):
        base.with_budget(protected=[6])
    with pytest.raises(ValueError, match="slot 3 is empty"):
        roark_state(tables, names, player_count=2).with_budget(protected=[3])
    with pytest.raises(ValueError, match="slot 2 has already fainted"):
        base.replace_hp(0, 2, 0).with_budget(protected=[2])


@pytest.mark.slow
def test_faint_records_the_slot(tables, names):
    s = roark_state(tables, names, player_count=1).replace_hp(0, 0, 1)
    tackle = names.move_slot(s, 1, "Tackle")
    actions = jnp.array([0, tackle])
    out = jit_step(tables, s, actions, empty_outcome())
    assert np.asarray(out.deaths).tolist() == [1, 0, 0, 0, 0, 0]
    assert int(out.player_faints) == 1
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(s),
            tuple(int(x) for x in actions),
            outcome_to_py(empty_outcome()),
        )
    )
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(out, ref, strict=True))

    won = roark_state(tables, names).replace_hp(1, 0, 1)
    out = jit_step(tables, won, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and not np.any(np.asarray(out.deaths))
    assert int(out.player_faints) == 0


def test_decision_actions(tables, names):
    s = roark_real_state(tables, names)
    assert decision_actions(tables, s).tolist() == legal_actions(tables, s, 0).tolist()
    down = s._replace(trapped=s.trapped.at[0].set(3)).replace_hp(0, 0, 0)
    mask = decision_actions(tables, down).tolist()
    assert len(mask) == NUM_ACTIONS
    assert mask == [False] * 4 + [False] + [True] * 5


def _struggle_double_ko(tables, names, player_count):
    """The lead (1 HP, no PP: Struggle) outspeeds Geodude (1 HP, the last opponent): its
    Struggle KOs Geodude and the recoil KOs the lead."""
    s = roark_state(tables, names, player_count=player_count, opp_count=1)
    return s.replace_hp(0, 0, 1).replace_hp(1, 0, 1).with_pp(0, 0, [0, 0, 0, 0])


def test_player_faint_counts_a_death_and_the_battle_goes_on(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1)
    tackle = names.move_slot(s, 1, "Tackle")
    out = jit_step(tables, s, jnp.array([0, tackle]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.over) == 0
    assert np.asarray(out.deaths).tolist() == [1, 0, 0, 0, 0, 0]
    assert needs_replacement(out).tolist() == [1, 0]


def test_double_ko_of_last_pokemon_is_a_loss(tables, names):
    # Ruling 2: both sides out at one faint check is BATTLE_RESULT_DRAW, played as a loss.
    s = _struggle_double_ko(tables, names, player_count=1)
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.over) == 1 and int(out.winner) == 1


def test_last_opponent_with_a_non_last_player_faint_is_a_win(tables, names):
    s = _struggle_double_ko(tables, names, player_count=2)
    out = jit_step(tables, s, jnp.array([0, 0]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert int(out.over) == 1 and int(out.winner) == 2
    assert int(out.deaths[0]) == 1


@pytest.mark.slow
def test_entry_faint_needs_a_second_replacement(tables, names):
    s = roark_state(tables, names).replace_hp(0, 0, 1)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1))
    tackle = names.move_slot(s, 1, "Tackle")
    out = jit_step(tables, s, jnp.array([0, tackle]), empty_outcome())
    assert needs_replacement(out).tolist() == [1, 0]
    out = out.replace_hp(0, 3, 1)  # Starly, which Stealth Rock faints on entry
    entered = replace(tables, out, 0, 3)
    assert int(entered.deaths[3]) == 1 and int(entered.player_faints) == 2
    assert needs_replacement(entered).tolist() == [1, 0] and int(entered.over) == 0
    # finish_turn resolves both replacements (the policy picks twice).
    done, refused = finish_turn(tables, out, jnp.bool_(False), NORMAL_POWER_MUL)
    assert needs_replacement(done).tolist() == [0, 0] and not bool(refused)
    assert int(done.hp[0, int(done.active[0])]) > 0
    # A playout started at the pending state (the entry-fainted newcomer) plays no turn
    # with a fainted player Pokemon: its first action is the second replacement.
    result = playout(tables, entered, jax.random.PRNGKey(0), max_turns=1)
    assert int(result.refused) == 0 and int(result.player_faints) >= 2


def test_pursuit_ko_still_completes_the_switch(tables, names):
    # subscript_switch_pokemon.s:10-12 reaches SwitchAndUpdateMon (_026) whatever the HP,
    # so the fainted Budew is replaced by slot 1 and the battle goes on.
    s = roark_state(
        tables, names, player_species="Budew", opp_species="Cranidos", opp_moves=["Pursuit"]
    ).replace_hp(0, 0, 1)
    pursuit = names.move_slot(s, 1, "Pursuit")
    out = jit_step(tables, s, jnp.array([MOVE_ACTIONS + 1, pursuit]), empty_outcome())
    assert int(out.hp[0, 0]) == 0 and int(out.active[0]) == 1 and int(out.over) == 0
    assert int(out.deaths[0]) == 1


@pytest.mark.slow
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_player_replacement_matches_reference(tables, names, data):
    s = data.draw(supported_state(tables, names))
    active = int(s.active[0])
    if int(s.hp[0, active]) > 0:
        s = s.replace_hp(0, active, 1)  # one hit from a faint
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    after = jit_step(tables, s, acts, o)
    if not needs_replacement(after)[0]:
        return
    legal = [
        slot
        for slot in range(6)
        if int(after.species[0, slot]) != 0 and int(after.hp[0, slot]) > 0 and slot != active
    ]
    for slot in legal:
        got = replace(tables, after, 0, slot)
        want = from_ref(ref_replace(to_numpy(tables), to_ref(after), 0, slot))
        for name, a, b in zip(got._fields, got, want, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
