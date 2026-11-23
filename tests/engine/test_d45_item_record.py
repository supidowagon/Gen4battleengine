"""D45-3: the AI's held-item record (``aiContext.battlerHeldItems``, ai_context.h:29) is written
for every kept battler (``state.ai_record_kept``) by the messages that name an item
(``BattleMessage_ItemTag``, battle_script.c:11726-11745) and cleared by every removal
(``BattleMon_CopyToParty``, battle_lib.c:1819-1823), any battler. Eager JAX against the reference;
no jitted B = 4 ``step`` here (fast tier)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine import hooks
from nuzline.engine.effects import consume_item, forget_item, reveal_item
from nuzline.engine.format import DOUBLES, MULTI, format_of
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import _leftovers_check, _stale_known_item, step
from nuzline.reference import engine as ref
from nuzline.reference.convert import from_ref, to_ref
from tests.engine.d2_states import four, with_hp
from tests.engine.helpers import roark_state


def _item_row(s) -> list:
    return np.asarray(s.ai_item_known).tolist()


def _with_item(s, b, item):
    p = format_of(s).party_of[b]
    return s._replace(item=s.item.at[p, int(s.active[b])].set(item))


def _turn(tables, fmt):
    return ref._Turn(to_numpy(tables), empty_outcome(), entry_turn=1, battler_count=fmt.B)


def _hurt(s, b):
    """Battler ``b``'s active Pokemon one HP below its maximum."""
    p = format_of(s).party_of[b]
    return with_hp(s, b, int(s.max_hp[p, int(s.active[b])]) - 1)


def test_leftovers_records_the_enemy(tables, names):
    leftovers = names.item("Leftovers")
    s = _hurt(_with_item(four(tables, names, DOUBLES), 3, leftovers), 3)
    out = _leftovers_check(tables, s, 3)
    r = to_ref(s)
    ref._leftovers_check(_turn(tables, DOUBLES), r, 3)
    assert _item_row(out) == [0, 0, 0, leftovers] == r.ai_item_known


def test_multi_partner_item_row_2(tables, names):
    leftovers = names.item("Leftovers")
    s = _hurt(_with_item(four(tables, names, MULTI), 2, leftovers), 2)
    out = _leftovers_check(tables, s, 2)
    r = to_ref(s)
    ref._leftovers_check(_turn(tables, MULTI), r, 2)
    assert _item_row(out) == [0, 0, leftovers, 0] == r.ai_item_known


def test_reveal_and_forget_any_battler(tables, names):
    leftovers = names.item("Leftovers")
    for fmt in (DOUBLES, MULTI):
        for b in range(4):
            s = _with_item(four(tables, names, fmt), b, leftovers)
            slot = int(s.active[b])
            want = [leftovers if i == b else 0 for i in range(4)]
            shown = reveal_item(s, b, slot, True)
            r = to_ref(s)
            ref._reveal_item(r, b)
            assert _item_row(shown) == want == r.ai_item_known, (fmt, b)
            assert _item_row(forget_item(shown, b, slot, True)) == [0] * 4
            ref._forget_item(r, b)
            assert r.ai_item_known == [0] * 4
            # RemoveItem's CopyToParty clears the record again (script.c:8981)
            gone = consume_item(shown, b, True)
            ref._reveal_item(r, b)
            ref._remove_item(r, b)
            assert _item_row(gone) == [0] * 4 == r.ai_item_known, (fmt, b)


def test_hp_berry_message_records_nothing(tables, names):
    """Sitrus Berry's message names its item through BTLSCR_MSG_TEMP (subscript_held_item_hp_restore.s:7):
    no record write, and the berry is eaten."""
    sitrus = names.item("Sitrus Berry")
    s = _with_item(four(tables, names, DOUBLES), 1, sitrus)
    slot = int(s.active[1])
    s = with_hp(s, 1, int(s.max_hp[1, slot]) // 2)
    out, triggered = hooks.held_item_check(tables, s, 1)
    r = to_ref(s)
    assert ref._held_item(_turn(tables, DOUBLES), r, 1)
    assert bool(triggered) and int(out.item[1, slot]) == 0
    assert _item_row(out) == [0] * 4 == r.ai_item_known


def test_knock_off_record_clears_at_next_copy(tables, names):
    leftovers = names.item("Leftovers")
    s = four(tables, names, DOUBLES)
    slot = int(s.active[1])
    s = s._replace(
        ai_item_known=s.ai_item_known.at[1].set(leftovers),
        knocked_off=s.knocked_off.at[1, slot].set(1),
    )
    hit = with_hp(s, 1, int(s.hp[1, slot]) - 1)
    quiet = jnp.zeros(4, bool)
    for after, fought, cleared in (
        (hit, quiet, True),  # its HP changed
        (s, quiet.at[1].set(True), True),  # it attempted a FIGHT command
        (s, quiet, False),  # neither: the record stays (B10-6)
    ):
        out = _stale_known_item(s, after, fought)
        r = to_ref(s)
        snaps = ref._stale_snapshot(r)
        if after is hit:
            r.parties[1].mons[slot].hp -= 1
        ref._stale_known_item(r, snaps, 1 if bool(fought[1]) else -1)
        want = [0, 0 if cleared else leftovers, 0, 0]
        assert _item_row(out) == want == r.ai_item_known, cleared


def test_singles_enemy_item_not_kept(tables, names):
    leftovers = names.item("Leftovers")
    s = roark_state(tables, names)
    s = _hurt(_with_item(s, 1, leftovers), 1)
    out = _leftovers_check(tables, s, 1)
    r = to_ref(s)
    ref._leftovers_check(_turn(tables, format_of(s)), r, 1)
    assert _item_row(out) == [0, 0] == r.ai_item_known
    s = _hurt(_with_item(s, 0, leftovers), 0)  # battler 0's as today
    out = _leftovers_check(tables, s, 0)
    r = to_ref(s)
    ref._leftovers_check(_turn(tables, format_of(s)), r, 0)
    assert _item_row(out) == [leftovers, 0] == r.ai_item_known


@pytest.mark.slow
def test_step_records_a_leftovers_heal_of_the_enemy(tables, names):
    """B = 4 ``step`` against ``ref_step``: battler 3 holds Leftovers below full HP; the turn-end
    heal's message records row 3 in both (subscript_restore_a_little_hp.s:10)."""
    leftovers = names.item("Leftovers")
    s = _hurt(_with_item(four(tables, names, DOUBLES), 3, leftovers), 3)
    acts = jnp.asarray([0, 0, 0, 0], jnp.int32)
    targets = [1, 0, 1, 0]
    o = empty_outcome(DOUBLES)
    out = step(tables, s, acts, o, jnp.asarray(targets, jnp.int8))
    ref_out = from_ref(
        ref.ref_step(
            to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o), targets
        )
    )
    assert _item_row(out)[3] == leftovers
    for name, a, b in zip(out._fields, out, ref_out, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
