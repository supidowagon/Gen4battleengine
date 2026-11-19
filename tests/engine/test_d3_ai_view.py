"""D3-1a: the AI's record (ability, held item, moves seen) is kept for every party-0 battler,
keyed by its active slot: battlers 0 and 2 in DOUBLES and TWO_TRAINERS, battler 0 in MULTI
(trainer_ai.c:362-375, battle_script.c:11806, battle_lib.c:2151-2153). JAX against reference,
by direct calls where ``step`` cannot reach battler 2 yet."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.effects import forget_item, know_ability, reveal_item
from nuzline.engine.format import DOUBLES, MULTI, format_of
from nuzline.engine.ids import AbilityId
from nuzline.engine.step import _forget, _record_last_move
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, with_ability, with_hp

ITEM = 17  # any nonzero id: the record copies it


def _with_item(s, b, item):
    p = format_of(s).party_of[b]
    return s._replace(item=s.item.at[p, int(s.active[b])].set(item))


def _known(s):
    return np.asarray(s.ai_ability_known).tolist()


def test_ability_tag_battler_2_doubles(tables, names):
    s = with_ability(four(tables, names, DOUBLES), 2, AbilityId.INTIMIDATE)
    slot = int(s.active[2])
    out = know_ability(s, 2, slot, True)
    r = to_ref(s)
    ref._know_ability(r, 2)
    assert _known(out)[slot] != 0
    assert _known(out) == r.ai_ability_known
    assert _known(know_ability(s, 1, int(s.active[1]), True)) == _known(s)  # the enemy: nothing


def test_ability_tag_battler_2_multi_writes_nothing(tables, names):
    s = with_ability(four(tables, names, MULTI), 2, AbilityId.INTIMIDATE)
    out = know_ability(s, 2, int(s.active[2]), True)
    r = to_ref(s)
    ref._know_ability(r, 2)
    assert _known(out) == _known(s) == r.ai_ability_known
    out0 = know_ability(with_ability(s, 0, AbilityId.INTIMIDATE), 0, int(s.active[0]), True)
    assert _known(out0)[int(s.active[0])] != 0  # battler 0 is party 0 in MULTI


def test_item_record_battler_2(tables, names):
    for fmt, kept in ((DOUBLES, True), (MULTI, False)):
        s = _with_item(four(tables, names, fmt), 2, ITEM)
        slot = int(s.active[2])
        out = reveal_item(s, 2, slot, True)
        r = to_ref(s)
        ref._reveal_item(r, 2)
        assert np.asarray(out.ai_item_known).tolist() == r.ai_item_known
        assert (int(out.ai_item_known[slot]) == ITEM) == kept
        # forget clears it again (and touches nothing it did not write)
        gone = forget_item(out, 2, slot, True)
        ref._forget_item(r, 2)
        assert np.asarray(gone.ai_item_known).tolist() == r.ai_item_known
        assert int(gone.ai_item_known[slot]) == 0


def test_switch_in_clears_battler_2_slot(tables, names):
    for fmt, cleared in ((DOUBLES, True), (MULTI, False)):
        s = four(tables, names, fmt)
        slot = int(s.active[2])
        s = s._replace(
            ai_seen_moves=s.ai_seen_moves.at[slot, 0].set(5),
            ai_ability_known=s.ai_ability_known.at[slot].set(7),
            ai_item_known=s.ai_item_known.at[slot].set(ITEM),
        )
        out = _forget(s, 2, slot)
        assert (int(out.ai_ability_known[slot]) == 0) == cleared
        assert (int(out.ai_item_known[slot]) == 0) == cleared
        assert (int(out.ai_seen_moves[slot, 0]) == 0) == cleared


def _record(tables, s, acts):
    """JAX ``_record_last_move`` and the reference's ``_record_last_moves`` on the same turn."""
    out = _record_last_move(tables, s, jnp.asarray(acts, jnp.int32))
    r = to_ref(s)
    t = to_numpy(tables)
    choices = [ref._choose(t, r, b, acts[b]) for b in range(4)]
    ref._record_last_moves(t, r, tuple(acts), choices)
    assert np.asarray(out.ai_seen_moves).tolist() == r.ai_seen_moves
    return np.asarray(out.ai_seen_moves)


def _with_last_moves(s, names):
    return s._replace(
        last_move=jnp.asarray([names.move("Tackle"), 0, names.move("Curse"), 0], s.last_move.dtype)
    )


def test_record_both_player_battlers(tables, names):
    s = _with_last_moves(four(tables, names, DOUBLES), names)
    seen = _record(tables, s, [0, MOVE_ACTIONS + 2, 0, 0])  # 1 switches, 3 fights
    assert seen[int(s.active[0]), 0] == names.move("Tackle")
    assert seen[int(s.active[2]), 0] == names.move("Curse")
    assert not seen[int(s.active[1])].any()


def test_record_skips_a_battler_without_hp(tables, names):
    s = _with_last_moves(with_hp(four(tables, names, DOUBLES), 2, 0), names)
    seen = _record(tables, s, [0, MOVE_ACTIONS + 2, 0, 0])
    assert seen[int(s.active[0]), 0] == names.move("Tackle")
    assert not seen[int(s.active[2])].any()


def test_record_needs_an_enemy_on_its_fight_path(tables, names):
    s = _with_last_moves(with_hp(four(tables, names, DOUBLES), 3, 0), names)
    seen = _record(tables, s, [0, MOVE_ACTIONS + 2, 0, 0])  # 3 absent, 1 switching
    assert not seen.any()
