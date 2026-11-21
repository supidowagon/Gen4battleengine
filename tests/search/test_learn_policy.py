"""E3-B3-16 (GT-16): the playout's learn answer from the growth plan (``search.learn``).

The plan's answers are a batch input (``learn_answer_array``), not a ``BattleState`` field; the
answer does not depend on the ``learn_site`` that paused the battle.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.levelup import GIVE_UP, advance_learns, needs_learn
from nuzline.search.learn import learn_answer_array, learn_policy, resolve_learns
from tests.engine.helpers import roark_real_state
from tests.search.test_learn import _moves, _with_prompt


@pytest.fixture
def prompt(tables, names):
    s = roark_real_state(tables, names, player_species="Machop")
    s, slot = _with_prompt(tables, names, s)
    s = advance_learns(tables, s)
    species, form = int(s.species[0, slot]), int(s.form[0, slot])
    row = int(tables.species_form_row[species, form])
    offered = int(tables.learnset_move[row, int(s.learn_cursor[slot])])
    level = int(s.learn_level[slot])
    return s, slot, (species, form), (level, offered)


def test_a_plan_answer_replaces_that_slot(tables, names, prompt):
    s, slot, (species, form), offer = prompt
    answers = learn_answer_array(tables, [(species, form, {offer: 2})])
    assert int(learn_policy(tables, s, jnp.asarray(answers))) == 2
    out = resolve_learns(tables, s, jnp.asarray(answers))
    assert not bool(needs_learn(out))
    assert int(out.moves[0, slot, 2]) == offer[1]
    before = np.asarray(s.moves[0, slot])
    after = np.asarray(out.moves[0, slot])
    assert [int(after[k]) for k in (0, 1, 3)] == [int(before[k]) for k in (0, 1, 3)]
    assert int(out.pp[0, slot, 2]) == int(tables.move_pp[offer[1]])


def test_the_plan_giving_up_keeps_the_moves(tables, names, prompt):
    s, slot, (species, form), offer = prompt
    answers = learn_answer_array(tables, [(species, form, {offer: None})])
    assert int(learn_policy(tables, s, jnp.asarray(answers))) == GIVE_UP
    out = resolve_learns(tables, s, jnp.asarray(answers))
    assert _moves(names, out, slot) == _moves(names, s, slot)


def test_no_plan_is_the_give_up_policy_byte_for_byte(tables, names, prompt):
    s, _, (species, form), _ = prompt
    assert int(learn_policy(tables, s)) == GIVE_UP
    plain = resolve_learns(tables, s)
    base = learn_answer_array(tables, [None])
    unanswered = learn_answer_array(tables, [(species, form, {})])
    assert (base == GIVE_UP).all() and (unanswered == GIVE_UP).all()
    for answers in (base, unanswered):
        out = resolve_learns(tables, s, jnp.asarray(answers))
        for name, x, y in zip(plain._fields, plain, out, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    np.testing.assert_array_equal(np.asarray(plain.moves), np.asarray(s.moves))


def test_the_answer_ignores_the_learn_site(tables, names, prompt):
    s, _, (species, form), offer = prompt
    answers = jnp.asarray(learn_answer_array(tables, [(species, form, {offer: 1})]))
    got = {
        int(learn_policy(tables, s._replace(learn_site=jnp.int8(site)), answers))
        for site in range(8)
    }
    assert got == {1}


def test_an_hm_slot_is_never_chosen(tables, names, prompt):
    s, slot, (species, form), offer = prompt
    hm = int(np.argmax(np.asarray(tables.move_is_hm) == 1))
    assert hm and int(tables.move_is_hm[hm]) == 1
    s = s._replace(moves=s.moves.at[0, slot, 2].set(hm))
    answers = jnp.asarray(learn_answer_array(tables, [(species, form, {offer: 2})]))
    assert int(learn_policy(tables, s, answers)) == GIVE_UP
    out = resolve_learns(tables, s, answers)
    assert int(out.moves[0, slot, 2]) == hm


def test_a_bad_answer_is_refused(tables, prompt):
    _, _, (species, form), offer = prompt
    with pytest.raises(ValueError, match="move slot"):
        learn_answer_array(tables, [(species, form, {offer: 4})])
