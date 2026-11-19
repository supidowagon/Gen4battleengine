"""D2-1 (docs/kaizo/e2-mechanics-ledger.md, "D2: plan"): the chosen target ``chosen_target``
(``battlerActions[b][1]``, bcp.c:548, :4363-4367) written at selection, JAX and reference."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import chosen_targets
from nuzline.engine.ids import MoveTarget
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _chosen_targets, ref_step
from tests.engine.d2_states import as_ghost, four, mon, with_hp, with_moves
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon

# (move, expected range, B = 2 target of battler 0)
B2_CASES = (
    ("Tackle", MoveTarget.RANGE_SINGLE_TARGET, 1),
    ("Earthquake", MoveTarget.RANGE_ALL_ADJACENT, 1),
    ("Spikes", MoveTarget.RANGE_OPPONENT_SIDE, 1),
    ("Reflect", MoveTarget.RANGE_USER_SIDE, 1),
    ("Helping Hand", MoveTarget.RANGE_ALLY, 1),
    ("Swords Dance", MoveTarget.RANGE_USER, 0),
    ("Counter", MoveTarget.RANGE_SINGLE_TARGET_SPECIAL, 0),
    ("Sunny Day", MoveTarget.RANGE_FIELD, 0),
    ("Acupressure", MoveTarget.RANGE_USER_OR_ALLY, 0),
    ("Curse", MoveTarget.RANGE_SINGLE_TARGET, 0),  # non-Ghost: RANGE_USER (bcp.c:4344-4349)
)


def _singles(tables, names, move="Tackle", opp="Tackle"):
    s = make_state(tables, [mon(names, (move,))], [mon(names, (opp,))])
    return s


def _np_actions(*a):
    return jnp.asarray(a, jnp.int32)


def test_b2_selecting_battler_writes_the_target_of_its_range(tables, names):
    for move, rng, want in B2_CASES:
        assert int(tables.move_target[names.move(move)]) == rng, move
        s = _singles(tables, names, move)
        got = chosen_targets(tables, s, _np_actions(0, 0), None if False else jnp.zeros(2))
        assert int(got[0]) == want, move
        assert int(got[1]) == 0  # the foe's Tackle: battler 1 targets battler 0
        ref = _chosen_targets(to_numpy(tables), to_ref(s), [0, 0], [0, 0])
        assert ref == got.tolist(), move


def test_b2_ghost_curse_targets_the_foe(tables, names):
    s = as_ghost(_singles(tables, names, "Curse"), 0)
    assert int(chosen_targets(tables, s, _np_actions(0, 0), jnp.zeros(2))[0]) == 1


@pytest.mark.slow
def test_b2_step_does_not_write_chosen_target(tables, names):
    # Ruling D2-28: nothing reads it at B = 2, so the step keeps the stored b ^ 1 and a
    # Swords Dance and a Tackle state stay equal. jit, not eager: an eager step peaks far above
    # the jitted one's compile.
    fn = jax.jit(step)
    for move in ("Swords Dance", "Tackle"):
        s = _singles(tables, names, move)
        out = fn(tables, s, _np_actions(0, 0), empty_outcome())
        assert out.chosen_target.tolist() == [1, 0], move
    parked = _singles(tables, names, "Tackle")
    parked = parked._replace(chosen_target=jnp.asarray([0, 1], jnp.int8))
    out = fn(tables, parked, _np_actions(0, 0), empty_outcome())
    assert out.chosen_target.tolist() == [0, 1]


def test_b2_reference_step_does_not_write_chosen_target(tables, names):
    for move in ("Swords Dance", "Tackle"):
        out = ref_step(
            to_numpy(tables), to_ref(_singles(tables, names, move)), (0, 0),
            outcome_to_py(empty_outcome()),
        )  # fmt: skip
        assert [b.chosen_target for b in out.battlers] == [1, 0], move


@pytest.mark.slow
def test_b2_targets_are_ignored(tables, names):
    s = _singles(tables, names, "Swords Dance")
    fn = jax.jit(step)
    base = fn(tables, s, _np_actions(0, 0), empty_outcome(), None)
    for t in ([0, 0], [0, 1], [1, 0], [1, 1]):
        out = fn(tables, s, _np_actions(0, 0), empty_outcome(), jnp.asarray(t, jnp.int8))
        for name, a, b in zip(base._fields, base, out, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_b2_no_write_without_a_target_menu(tables, names):
    # chosen_target starts at b ^ 1; park it on the user and see it kept.
    s = make_state(tables, [mon(names, ("Tackle",)), mon(names)], [mon(names), mon(names)])
    s = s._replace(chosen_target=jnp.asarray([0, 1], jnp.int8))
    z = jnp.zeros(2)
    # a switch, an opponent item action
    assert chosen_targets(tables, s, _np_actions(4, 10), z).tolist() == [0, 1]
    # Struggling: no PP anywhere
    st = s._replace(pp=s.pp.at[0, 0].set(0))
    assert chosen_targets(tables, st, _np_actions(0, 10), z).tolist() == [0, 1]
    # encored (the encored slot is taken without the menu, bcp.c:442-453)
    se = s._replace(
        encore_move=s.encore_move.at[0].set(s.moves[0, 0, 0]),
        encore_slot=s.encore_slot.at[0].set(0),
    )
    assert chosen_targets(tables, se, _np_actions(0, 10), z).tolist() == [0, 1]
    # locked into a move (bcp.c:334-338)
    sl = s._replace(locked_move=s.locked_move.at[0].set(s.moves[0, 0, 0]))
    assert chosen_targets(tables, sl, _np_actions(0, 10), z).tolist() == [0, 1]
    # and the selecting case does write
    assert chosen_targets(tables, s, _np_actions(0, 10), z).tolist() == [1, 1]
    for case, acts in ((s, [4, 10]), (st, [0, 10]), (se, [0, 10]), (sl, [0, 10])):
        want = chosen_targets(tables, case, _np_actions(*acts), z).tolist()
        assert _chosen_targets(to_numpy(tables), to_ref(case), acts, [0, 0]) == want


@pytest.mark.slow  # a jitted step compile: over the fast tier's 4 GB
def test_b2_u_turn_pause_and_resume_keep_the_selection(tables, names):
    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 35)],
    )
    o = empty_outcome()
    paused = assert_matches_reference(tables, s, (0, 0), o)
    assert int(paused.pivot_battler) == 1
    written = paused.chosen_target.tolist()
    assert written == [1, 0]
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[0] = 4 + 1
    after = assert_matches_reference(tables, paused, tuple(acts), o)
    assert after.chosen_target.tolist() == written


def _four_state(tables, names):
    s = four(tables, names)
    return s._replace(chosen_target=jnp.asarray([3, 2, 1, 0], jnp.int8))


def test_b4_selection_writes_targets_and_keeps_old_values(tables, names):
    s = _four_state(tables, names)
    t = jnp.asarray([2, 3, 0, 1], jnp.int8)
    acts = _np_actions(0, 0, 0, 0)
    assert chosen_targets(tables, s, acts, t).tolist() == [2, 3, 0, 1]
    # absent battler 1 (active hp 0), battler 2 switching, battler 3 locked, battler 0 selecting
    s2 = with_hp(s, 1, 0)
    s2 = s2._replace(locked_move=s2.locked_move.at[3].set(s2.moves[1, int(s2.active[3]), 0]))
    acts2 = _np_actions(0, 0, 4, 0)
    got = chosen_targets(tables, s2, acts2, t)
    assert got.tolist() == [2, 2, 1, 0]
    ref = _chosen_targets(to_numpy(tables), to_ref(s2), [0, 0, 4, 0], [2, 3, 0, 1])
    assert ref == got.tolist()
    ref = _chosen_targets(to_numpy(tables), to_ref(s), [0, 0, 0, 0], [2, 3, 0, 1])
    assert ref == [2, 3, 0, 1]


def test_b4_helping_hand_with_the_partner_absent_writes_nothing(tables, names):
    s = _four_state(tables, names)
    s = with_moves(s, names, 0, ("Helping Hand",))
    t = jnp.asarray([2, 3, 0, 1], jnp.int8)
    acts = _np_actions(0, 0, 0, 0)
    present = chosen_targets(tables, s, acts, t)
    assert int(present[0]) == 2  # written
    gone = with_hp(s, 2, 0)
    absent = chosen_targets(tables, gone, acts, t)
    assert int(absent[0]) == 3  # the old value stays
    for case, want in ((s, present), (gone, absent)):
        ref = _chosen_targets(to_numpy(tables), to_ref(case), [0, 0, 0, 0], [2, 3, 0, 1])
        assert ref == want.tolist()


def test_b4_step_needs_targets(tables, names):
    s = four(tables, names)
    with pytest.raises(ValueError):
        step(tables, s, _np_actions(0, 0, 0, 0), empty_outcome())
