"""D3-1a: the AI's record (ability, held item, moves seen) is written for every party-0 battler,
in its battler's row (sheet D45-1): battlers 0 and 2 in DOUBLES and TWO_TRAINERS, battler 0 in
MULTI (trainer_ai.c:362-375, battle_script.c:11806, battle_lib.c:2151-2153). JAX against reference,
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
    s = with_ability(s, 1, AbilityId.INTIMIDATE)
    slot = int(s.active[2])
    out = know_ability(s, 2, slot, True)
    r = to_ref(s)
    ref._know_ability(r, 2)
    assert _known(out)[2] != 0
    assert _known(out) == r.ai_ability_known
    # the enemy's row is kept at B = 4 too (sheet D45-2 rule 4)
    enemy = know_ability(s, 1, int(s.active[1]), True)
    ref._know_ability(r, 1)
    assert _known(enemy)[1] != 0 and _known(enemy)[0] == _known(enemy)[2] == 0
    assert _known(enemy) == [0, r.ai_ability_known[1], 0, 0]


def test_ability_tag_battler_2_multi_records_row_2(tables, names):
    s = with_ability(four(tables, names, MULTI), 2, AbilityId.INTIMIDATE)
    out = know_ability(s, 2, int(s.active[2]), True)
    r = to_ref(s)
    ref._know_ability(r, 2)
    assert _known(out)[2] != 0
    assert _known(out) == r.ai_ability_known
    out0 = know_ability(with_ability(s, 0, AbilityId.INTIMIDATE), 0, int(s.active[0]), True)
    assert _known(out0)[0] != 0  # battler 0 is party 0 in MULTI


def test_item_record_battler_2(tables, names):
    for fmt, kept in ((DOUBLES, True), (MULTI, False)):
        s = _with_item(four(tables, names, fmt), 2, ITEM)
        slot = int(s.active[2])
        out = reveal_item(s, 2, slot, True)
        r = to_ref(s)
        ref._reveal_item(r, 2)
        assert np.asarray(out.ai_item_known).tolist() == r.ai_item_known
        assert (int(out.ai_item_known[2]) == ITEM) == kept
        # forget clears it again (and touches nothing it did not write)
        gone = forget_item(out, 2, slot, True)
        ref._forget_item(r, 2)
        assert np.asarray(gone.ai_item_known).tolist() == r.ai_item_known
        assert int(gone.ai_item_known[2]) == 0


def test_switch_in_clears_battler_2_row(tables, names):
    """UpdateAfterSwitch clears any battler's three rows (battle_lib.c:2151-2153), D45-1 rule 3."""
    for fmt in (DOUBLES, MULTI):
        s = four(tables, names, fmt)
        s = s._replace(
            ai_seen_moves=s.ai_seen_moves.at[2, 0].set(5),
            ai_ability_known=s.ai_ability_known.at[2].set(7),
            ai_item_known=s.ai_item_known.at[2].set(ITEM),
        )
        out = _forget(s, 2, True)
        assert int(out.ai_ability_known[2]) == 0
        assert int(out.ai_item_known[2]) == 0
        assert int(out.ai_seen_moves[2, 0]) == 0


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
    assert seen[0, 0] == names.move("Tackle")
    assert seen[2, 0] == names.move("Curse")
    # the record is keyed by battler; battler 1 is the enemy, so its last move lands nowhere:
    # only the two player rows hold anything, one move each (the writes of the enemy rows come
    # with D45-4)
    rows = {0, 2}
    assert all(not seen[i].any() for i in range(seen.shape[0]) if i not in rows)
    assert all(int((seen[i] != 0).sum()) == 1 for i in rows)


def test_record_skips_a_battler_without_hp(tables, names):
    s = _with_last_moves(with_hp(four(tables, names, DOUBLES), 2, 0), names)
    seen = _record(tables, s, [0, MOVE_ACTIONS + 2, 0, 0])
    assert seen[0, 0] == names.move("Tackle")
    assert not seen[2].any()


def test_record_needs_an_enemy_on_its_fight_path(tables, names):
    s = _with_last_moves(with_hp(four(tables, names, DOUBLES), 3, 0), names)
    seen = _record(tables, s, [0, MOVE_ACTIONS + 2, 0, 0])  # 3 absent, 1 switching
    assert not seen.any()
