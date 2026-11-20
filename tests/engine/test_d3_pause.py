"""D3-10: pause and resume with the action's defender (sheet D3-10 rules 1-4, 4.6).

A U-turn or Baton Pass stops at the party list inside its action. ``battleCtx->defender`` is the
one FightCommand fixed (bcp.c:1900) and nothing rewrites it through the list or a Pursuit
(subscript_pursuit.s:5, :80, :116), so the resumed tail reads that battler: ``ts_defender``
(``1 + defender``, written at B = 4 only). The sort before the action stays in force to its
MoveEnd (``ts_speed_order``, D3-4 rule 7). The list offers the party's slots that no battler
sending from that party holds (BattleSystem_AnyReplacementMons, battle_lib.c:3125-3180, :3142-3157).

Each first half is ``play`` (the action phase, engine equal to the reference); each resume is
``resume`` below (``step._action_phase`` on the paused state, equal to ``ref_step(...,
actions_only=True)``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS, legal_replacements
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.outcome import empty_outcome, moved_mask, outcome_to_py, pivot_view
from nuzline.engine.step import _action_phase, _has_bench
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import _replacement_bench, ref_step
from tests.engine.d2_states import four, with_hp, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_pivot import _mon

IDLE = "Swords Dance"
SWITCH_2 = MOVE_ACTIONS + 2  # to party slot 2
ORDER = (0, 3, 1, 2)  # battler 0 first, so its action is the paused one
ATK = Stat.ATK
NO_TARGETS = (0, 0, 0, 0)


@jax.jit
def _engine_resume(tables, state, actions, outcome, targets):
    return _action_phase(tables, state, actions, outcome, targets)[0]


def resume(tables, paused, slot=2, outcome=None, targets=NO_TARGETS):
    """The paused turn resumed with the pivot's switch to ``slot``: the action phase on the
    engine, asserted equal to the reference's (every field but ``flinched``, which the turn's end
    clears); returns the engine state."""
    outcome = empty_outcome(DOUBLES) if outcome is None else outcome
    pivot = int(paused.pivot_battler) - 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[pivot] = MOVE_ACTIONS + slot
    out = _engine_resume(
        tables,
        paused,
        jnp.asarray(acts, jnp.int32),
        outcome,
        jnp.asarray(targets, jnp.int8),
    )
    ref = from_ref(
        ref_step(
            to_numpy(tables),
            to_ref(paused),
            tuple(acts),
            outcome_to_py(outcome),
            list(targets),
            actions_only=True,
        )
    )
    for name, x, y in zip(out._fields, out, ref, strict=True):
        if name == "flinched":
            continue
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    return out


def base(tables, names):
    """DOUBLES, every battler knows only Swords Dance."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, [IDLE])
    return s


def pivoting(tables, names, move="U-turn"):
    """Battler 0 knows ``move``, the others Swords Dance."""
    return with_moves(base(tables, names), names, 0, [move])


def encore(state, names, b: int, slot: int, move: str):
    return state._replace(
        encore_move=state.encore_move.at[b].set(names.move(move)),
        encore_slot=state.encore_slot.at[b].set(slot),
        encore_turns=state.encore_turns.at[b].set(3),
    )


def same_state(x, y):
    for name, a, b in zip(x._fields, x, y, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def scratch_zero(state):
    for name in state._fields:
        if name.startswith("ts_"):
            assert not np.asarray(getattr(state, name)).any(), name
    assert int(state.pivot_battler) == 0


# --- the kept defender ----------------------------------------------------------------------


def test_a_u_turn_that_kos_its_defender_keeps_it_through_the_list(tables, names):
    """Battler 0's U-turn KOs battler 1 (bench left in party 1): the list opens with
    ``ts_defender == 2``. The resumed tail reads battler 1 (0 HP: nothing), not a new pick: battler
    3 is untouched and no ``target_draw`` is read (a different draw gives an equal state); the later
    entries of the order run (battler 3 and 2; battler 1 fainted)."""
    s = with_hp(pivoting(tables, names), 1, 1)
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert int(paused.pivot_battler) == 1 and hp(paused, 1) == 0
    assert int(paused.ts_defender) == 2
    assert np.asarray(paused.ts_acted).tolist() == [0, 0, 0, 0]
    draw = empty_outcome(DOUBLES)._replace(target_draw=jnp.ones(4, jnp.int8))
    out = resume(tables, paused, outcome=draw)
    again = resume(tables, paused)
    same_state(out, again)
    assert hp(out, 3) == hp(s, 3)
    assert int(out.stages[3, ATK]) == 2 and int(out.stages[2, ATK]) == 2  # four actions in all
    assert int(out.stages[1, ATK]) == 0  # the KO'd battler never acted
    scratch_zero(out)


def test_an_encored_u_turn_keeps_its_first_defender_for_the_kings_rock_flinch(tables, names):
    """Battler 0, Encored into U-turn (a randomized target), hits battler 1 with the first
    ``target_draw``; the newcomer holds King's Rock. The resume's outcome would pick battler 3
    (``target_draw[0] = 1``) and rolls the flinch: battler 1 (kept, not yet moved) is flinched,
    battler 3 is not."""
    s = encore(pivoting(tables, names), names, 0, 0, "U-turn")
    s = s._replace(item=s.item.at[0, 2].set(names.item("King’s Rock")))
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert int(paused.pivot_battler) == 1 and int(paused.ts_defender) == 2
    assert hp(paused, 1) > 0
    later = empty_outcome(DOUBLES)._replace(
        target_draw=jnp.ones(4, jnp.int8),
        flinch_item=empty_outcome(DOUBLES).flinch_item.at[0, 0].set(1),
    )
    out = resume(tables, paused, outcome=later)
    assert int(out.stages[1, ATK]) == 0  # flinched: its Swords Dance never ran
    assert int(out.stages[3, ATK]) == 2


def test_baton_pass_keeps_the_user_as_its_defender(tables, names):
    """RANGE_USER: ``battleCtx->defender`` is the user (battler 0), ``ts_defender == 1``."""
    s = pivoting(tables, names, "Baton Pass")
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_defender) == 1
    out = resume(tables, paused)
    assert int(out.stages[3, ATK]) == 2 and int(out.stages[2, ATK]) == 2
    scratch_zero(out)


def test_the_sort_stays_through_the_list_and_both_scratch_fields_end_at_zero(tables, names):
    """``ts_speed_order`` is the sort before the paused action (the battlers fastest first:
    ``ORDER``), kept across the list and 0 after the turn; it is the paused action's only."""
    paused, _ = play(tables, pivoting(tables, names), [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert np.asarray(paused.ts_speed_order).tolist() == list(ORDER)
    out = resume(tables, paused)
    assert np.asarray(out.ts_speed_order).tolist() == [0, 0, 0, 0] and int(out.ts_defender) == 0
    scratch_zero(out)


def test_the_broken_substitute_flag_crosses_the_pause(tables, names):
    """F-4 through the list: battler 0's U-turn breaks battler 1's Substitute, ``ts_sub_hit[1] ==
    1`` at the pause and every entry 0 after the turn (the newcomer's Intimidate reader is D3-13's)."""
    s = pivoting(tables, names)
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(1))
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert int(paused.pivot_battler) == 1
    assert np.asarray(paused.ts_sub_hit).tolist() == [0, 1, 0, 0]
    out = resume(tables, paused)
    assert np.asarray(out.ts_sub_hit).tolist() == [0, 0, 0, 0]


def test_the_intimidate_newcomer_skips_the_flagged_foe_through_the_pause(tables, names):
    """D3-13 (``switch_in.run_states``): the party 0 slot 2 newcomer has Intimidate. Battler 0's
    U-turn breaks battler 1's Substitute: the bit is set at the pause, the newcomer's Intimidate
    skips battler 1 and lowers battler 3 (each +2 from its own Swords Dance after), and every
    entry is 0 after the turn."""
    s = pivoting(tables, names)
    s = s._replace(
        substitute_hp=s.substitute_hp.at[1].set(1),
        ability=s.ability.at[0, 2].set(int(AbilityId.INTIMIDATE)),
    )
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert np.asarray(paused.ts_sub_hit).tolist() == [0, 1, 0, 0]
    out = resume(tables, paused)
    assert int(out.stages[1, ATK]) == 2 and int(out.stages[3, ATK]) == 1
    assert np.asarray(out.ts_sub_hit).tolist() == [0, 0, 0, 0]


# --- the list --------------------------------------------------------------------------------


def with_party0_alive(state, alive):
    """Only the party 0 slots in ``alive`` keep their HP."""
    hp0 = np.asarray(state.hp[0]).copy()
    for slot in range(len(hp0)):
        if slot not in alive:
            hp0[slot] = 0
    return state._replace(hp=state.hp.at[0].set(jnp.asarray(hp0, state.hp.dtype)))


def slots(mask):
    return np.flatnonzero(np.asarray(mask)).tolist()


def test_a_partners_slot_is_not_a_replacement(tables, names):
    """DOUBLES: party 0 = slot 0 (battler 0), slot 1 (battler 2), the rest fainted: battler 0 has
    no replacement, so its U-turn does not switch and its Baton Pass fails (no pause). With slot 2
    alive both switch to slot 2 only (battler 2's slot 1 is not offered)."""
    for move in ("U-turn", "Baton Pass"):
        none = with_party0_alive(pivoting(tables, names, move), {0, 1})
        assert not bool(_has_bench(none, 0)) and slots(legal_replacements(none, 0)) == []
        out, _ = play(tables, none, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
        assert int(out.pivot_battler) == 0
        some = with_party0_alive(pivoting(tables, names, move), {0, 1, 2})
        assert bool(_has_bench(some, 0)) and slots(legal_replacements(some, 0)) == [2]
        out, _ = play(tables, some, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
        assert int(out.pivot_battler) == 1


def test_the_list_by_format(tables, names):
    """DOUBLES and TWO_TRAINERS' player side (battlers 0 and 2 share party 0): battler 2's active
    slot is excluded from battler 0's list (lib.c:3151-3157). MULTI: battler 2 sends from party 2,
    so battler 0's list is all of party 0's bench (:3142-3150); TWO_TRAINERS' enemy battlers 1 and
    3 have separate parties too. Engine equal to the reference's ``_replacement_bench``."""
    expected = {
        (DOUBLES, 0): [2, 3, 4, 5],
        (DOUBLES, 1): [2, 3, 4, 5],
        (TWO_TRAINERS, 0): [2, 3, 4, 5],
        (TWO_TRAINERS, 1): [1, 2, 3, 4, 5],
        (MULTI, 0): [1, 2, 3, 4, 5],
        (MULTI, 1): [1, 2, 3, 4, 5],
        (SINGLES, 0): [1, 2, 3, 4, 5],
    }
    for (fmt, b), want in expected.items():
        s = four(tables, names, fmt)
        assert slots(legal_replacements(s, b)) == want, (fmt, b)
        ref = to_ref(s)
        assert [i for i, ok in enumerate(_replacement_bench(ref, b)) if ok] == want, (fmt, b)
        assert bool(_has_bench(s, b))


# --- the paused view -------------------------------------------------------------------------


def test_the_moved_mask_counts_a_dragged_battler_at_every_format(tables, names):
    """``ts_dragged`` (UpdateAfterSwitch sets the dragged battler's action to MOVE_END,
    battle_lib.c:2053) is a ``moved_mask`` term at B = 4, and ``pivot_view`` gives that battler a
    switch to its own active Pokemon (no move) in place of its chosen action."""
    s = pivoting(tables, names)
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    acts = jnp.asarray([SWITCH_2, 0, 0, 0], jnp.int32)
    assert np.asarray(moved_mask(paused)).tolist() == [False] * 4
    _, viewed = pivot_view(tables, paused, acts)
    assert np.asarray(viewed).tolist()[1:] == [0, 0, 0]
    dragged = paused._replace(ts_dragged=paused.ts_dragged.at[3].set(1))
    assert np.asarray(moved_mask(dragged)).tolist() == [False, False, False, True]
    _, viewed = pivot_view(tables, dragged, acts)
    assert int(viewed[3]) == MOVE_ACTIONS + int(paused.active[3])
    assert int(viewed[1]) == 0 and int(viewed[2]) == 0


def test_the_moved_mask_terms_at_four(tables, names):
    """Each other term, at B = 4: ``ts_acted`` (the actions before the cursor) and a 0-HP battler
    (SF2 (j)); the pursuer's bit is D3-9's."""
    s = pivoting(tables, names)
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    acted = paused._replace(ts_acted=paused.ts_acted.at[2].set(1))
    assert np.asarray(moved_mask(acted)).tolist() == [False, False, True, False]
    pursued = paused._replace(ts_pursued=jnp.asarray(0b1000, paused.ts_pursued.dtype))
    assert np.asarray(moved_mask(pursued)).tolist() == [False, False, False, True]
    down = with_hp(paused, 3, 0)
    assert np.asarray(moved_mask(down)).tolist() == [False, False, False, True]


def test_singles_a_paused_view_with_a_fainted_foe_is_unchanged(tables, names):
    """B = 2: the paused U-turn that KO'd its foe (bench left): the foe is MOVE_END (SF2 (j)) and
    ``pivot_view`` gives it a switch to its own active Pokemon; ``ts_dragged`` is 0 at every
    singles pause, so the new term changes nothing."""
    from nuzline.engine.state import make_state
    from tests.engine.helpers import assert_matches_reference

    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 35)],
    )
    s = with_hp(s, 1, 1)
    paused = assert_matches_reference(tables, s, (0, 0), empty_outcome(SINGLES))
    assert int(paused.pivot_battler) == 1 and int(paused.hp[1, 0]) == 0
    assert not np.asarray(paused.ts_dragged).any()
    assert np.asarray(moved_mask(paused)).tolist() == [False, True]
    _, viewed = pivot_view(tables, paused, jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32))
    assert int(viewed[1]) == MOVE_ACTIONS + int(paused.active[1])
    assert int(paused.ts_defender) == 0 and not np.asarray(paused.ts_speed_order).any()
