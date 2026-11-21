"""D45-1: the AI's record per battler (``aiContext.battlerMoves / battlerAbilities /
battlerHeldItems``, include/battle/ai_context.h:27-29): layout, the accessors, the entry clear
(battle_lib.c:2151-2153), the faint clear (:2223-2225), the writers' battler key and the three
readers in ``ai/commands.py``. Eager JAX against the reference; no jitted B = 4 ``step`` (fast
tier)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.ai import commands as c
from nuzline.data.tables import to_numpy
from nuzline.engine.effects import know_ability, reveal_item
from nuzline.engine.format import DOUBLES, FORMATS, MULTI, SINGLES, TWO_TRAINERS, format_of
from nuzline.engine.ids import AbilityId
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import (
    MOVES,
    ai_record_ability,
    ai_record_item,
    ai_record_moves,
    layout,
)
from nuzline.engine.step import _faint_check, _record_last_move, _switch_in
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.ai.fixtures import outcome
from tests.engine.d2_states import four, with_ability, with_hp
from tests.engine.helpers import roark_state

FIELDS = ("ai_seen_moves", "ai_ability_known", "ai_item_known")


def _seeded(s):
    """Every battler's three rows hold their own nonzero values (moves 10 + b, ability 7 + b,
    item 17 + b)."""
    n = format_of(s).B
    return s._replace(
        ai_seen_moves=jnp.asarray([[10 + b, 5, 0, 0] for b in range(n)], jnp.int16),
        ai_ability_known=jnp.asarray([7 + b for b in range(n)], jnp.int16),
        ai_item_known=jnp.asarray([17 + b for b in range(n)], jnp.int16),
    )


def _rows(s) -> dict:
    return {f: np.asarray(getattr(s, f)).tolist() for f in FIELDS}


def _ref_rows(r) -> dict:
    return {f: getattr(r, f) for f in FIELDS}


def _turn(tables, fmt):
    return ref._Turn(to_numpy(tables), empty_outcome(), entry_turn=1, battler_count=fmt.B)


def _only_row_cleared(before: dict, after: dict, b: int) -> None:
    for f in FIELDS:
        for row, (was, now) in enumerate(zip(before[f], after[f], strict=True)):
            zero = [0] * len(was) if isinstance(was, list) else 0
            assert now == (zero if row == b else was), (f, row)


def test_record_layout_per_battler():
    for fmt in FORMATS:
        shapes = {f: layout(fmt)[f] for f in FIELDS}
        assert shapes["ai_seen_moves"] == ((fmt.B, MOVES), np.int16), fmt
        assert shapes["ai_ability_known"] == ((fmt.B,), np.int16), fmt
        assert shapes["ai_item_known"] == ((fmt.B,), np.int16), fmt


def test_initial_rows_are_zero_at_four_battlers(tables, names):
    for fmt in (DOUBLES, TWO_TRAINERS, MULTI):
        s = four(tables, names, fmt)
        for f in FIELDS:
            assert not np.asarray(getattr(s, f)).any(), (fmt, f)


def test_accessors_read_absent_battlers_as_zero(tables, names):
    s = roark_state(tables, names)
    s = s._replace(
        ai_seen_moves=jnp.asarray([[5, 6, 0, 0], [8, 0, 0, 0]], jnp.int16),
        ai_ability_known=jnp.asarray([7, 9], jnp.int16),
        ai_item_known=jnp.asarray([17, 19], jnp.int16),
    )
    assert int(ai_record_ability(s, 2)) == 0 and int(ai_record_item(s, 3)) == 0
    assert not np.asarray(ai_record_moves(s, jnp.int32(3))).any()
    assert int(ai_record_ability(s, 0)) == 7 and int(ai_record_ability(s, jnp.int32(1))) == 9
    assert np.asarray(ai_record_moves(s, 0)).tolist() == [5, 6, 0, 0]
    assert int(ai_record_item(s, jnp.int32(0))) == 17


def test_entry_clears_the_entering_battler_only(tables, names):
    for fmt in (DOUBLES, MULTI):
        s = _seeded(four(tables, names, fmt))
        bench = 5  # no battler leads from the last slot
        out = _switch_in(tables, s, 2, bench, 1)
        r = to_ref(s)
        ref._enter(_turn(tables, fmt), r, 2, bench)
        assert _rows(out) == _ref_rows(r), fmt
        _only_row_cleared(_rows(s), _rows(out), 2)


def test_faint_clears_the_fainted_battler_only(tables, names):
    s = _seeded(four(tables, names, DOUBLES))
    out = _faint_check(tables, s, with_hp(s, 2, 0))
    r = to_ref(s)
    ref._cleanup_fainted(_turn(tables, DOUBLES), r, 2)
    assert _rows(out) == _ref_rows(r)
    _only_row_cleared(_rows(s), _rows(out), 2)
    single = _seeded(roark_state(tables, names))
    out = _faint_check(tables, single, with_hp(single, 0, 0))
    r = to_ref(single)
    ref._cleanup_fainted(_turn(tables, SINGLES), r, 0)
    assert _rows(out) == _ref_rows(r)
    _only_row_cleared(_rows(single), _rows(out), 0)


def test_write_names_the_battler_holding_the_slot(tables, names):
    s = with_ability(roark_state(tables, names), 0, AbilityId.INTIMIDATE)
    active = int(s.active[0])
    benched = next(slot for slot in range(6) if slot != active and int(s.species[0, slot]) > 0)
    s = s._replace(item=s.item.at[0, active].set(17))
    assert np.array_equal(reveal_item(s, 0, benched, True).ai_item_known, s.ai_item_known)
    assert np.array_equal(know_ability(s, 0, benched, True).ai_ability_known, s.ai_ability_known)
    assert int(reveal_item(s, 0, active, True).ai_item_known[0]) == 17
    assert int(know_ability(s, 0, active, True).ai_ability_known[0]) != 0
    assert int(know_ability(s, 0, active, False).ai_ability_known[0]) == 0


def _record(tables, s, acts) -> np.ndarray:
    """JAX ``_record_last_move`` and the reference's ``_record_last_moves`` on the same turn."""
    n = format_of(s).B
    out = _record_last_move(tables, s, jnp.asarray(acts, jnp.int32))
    r = to_ref(s)
    t = to_numpy(tables)
    choices = [ref._choose(t, r, b, acts[b]) for b in range(n)]
    ref._record_last_moves(t, r, tuple(acts), choices)
    assert np.asarray(out.ai_seen_moves).tolist() == r.ai_seen_moves
    return np.asarray(out.ai_seen_moves)


def test_record_last_move_rows(tables, names):
    tackle, curse = names.move("Tackle"), names.move("Curse")
    last = jnp.asarray([tackle, 0, curse, 0], jnp.int16)
    s = four(tables, names, DOUBLES)
    seen = _record(tables, s._replace(last_move=last.astype(s.last_move.dtype)), [0, 4 + 2, 0, 0])
    assert seen[0, 0] == tackle and seen[2, 0] == curse and not seen[[1, 3]].any()  # rows 0 and 2
    s = four(tables, names, MULTI)
    seen = _record(tables, s._replace(last_move=last.astype(s.last_move.dtype)), [0, 4 + 2, 0, 0])
    assert seen[0, 0] == tackle and not seen[1:].any()  # MULTI: row 0 only
    s = roark_state(tables, names)
    s = s._replace(last_move=s.last_move.at[0].set(tackle))
    seen = _record(tables, s, [0, 0])
    assert seen.shape == (2, 4) and seen[0, 0] == tackle and not seen[1].any()


def test_readers_take_battler_0_row(tables, names):
    tackle = names.move("Tackle")
    s = roark_state(tables, names)
    s = s._replace(active=s.active.at[0].set(3))
    s = s._replace(
        ai_ability_known=s.ai_ability_known.at[0].set(7),
        ai_item_known=s.ai_item_known.at[0].set(17),
        ai_seen_moves=s.ai_seen_moves.at[0].set(jnp.asarray([tackle, 0, 0, 0], jnp.int16)),
    )
    value, known, _ = c._defender_abilities(tables, s)
    assert int(value) == 7 and bool(known)
    ctx = c.new_context(tables, s, outcome())
    assert int(c._held_item(ctx, c.Battler.DEFENDER)) == 17
    assert tackle in np.asarray(ctx.seen_moves).tolist()
