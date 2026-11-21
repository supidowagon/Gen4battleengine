"""D45-18: ``sent_out``, the per-battler send-out mask (``BattleSystem.battleParticipantMask``,
battle_system.h:118; set by ``BattleSystem_GetBattleParticipantMask``, battle_system.c:973-976, at
both send-out emitters, battle_controller.c:282, :332; never cleared). Sheet D45-18.

Every test is fast: singles turns go through ``assert_matches_reference``, the rest call
``step._switch_in`` / ``replace`` / ``replace_round`` and the reference twins directly (no B = 4
``step`` compile).
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine import step as step_module
from nuzline.engine.actions import MOVE_ACTIONS
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import initial_arrays, make_state
from nuzline.reference import engine as ref
from nuzline.reference.convert import from_ref, to_ref
from tests.engine.d2_states import four, mon
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_d3_replacement import down
from tests.engine.test_pivot import _mon
from tests.strategies import supported_state

O = empty_outcome()


def _sent(s) -> list[int]:
    return np.asarray(s.sent_out).tolist()


def _singles(tables, names, count=3):
    members = ["Bulbasaur", "Geodude", "Scyther", "Shinx"][:count]
    return make_state(
        tables,
        [_mon(names, m, ["Tackle", "U-turn"], 30) for m in members],
        [_mon(names, m, ["Tackle"], 30) for m in members],
    )


def _same(out, rs):
    back = from_ref(rs)
    for name, x, y in zip(out._fields, out, back, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def test_start_marks_every_lead(tables, names):
    """The start's PokemonSlideIn of every battler (rule 3): ``1 << lead``."""
    members = [mon(names) for _ in range(6)]
    a = initial_arrays(SINGLES, tables, [members, members], leads=(0, 2), opp2_items=())
    assert a["sent_out"].tolist() == [1, 4]
    assert _sent(four(tables, names, DOUBLES)) == [1, 1, 2, 2]
    assert _sent(four(tables, names, TWO_TRAINERS)) == [1, 1, 2, 1]
    assert _sent(four(tables, names, MULTI)) == [1, 1, 1, 1]


def test_switch_sets_and_never_clears(tables, names):
    """Both sides switch (engine = reference); going back to a slot keeps every bit."""
    s = _singles(tables, names)
    assert _sent(s) == [1, 1]
    out = assert_matches_reference(tables, s, (MOVE_ACTIONS + 1, MOVE_ACTIONS + 2), O)
    assert _sent(out) == [0b11, 0b101]
    out = assert_matches_reference(tables, out, (MOVE_ACTIONS + 0, 0), O)
    assert _sent(out) == [0b11, 0b101]


def test_drag_marks_the_dragged_side(tables, names):
    """A drag runs ``_switch`` -> ``_switch_in`` (subscript_force_target_to_switch_or_flee.s:22):
    the dragged side's byte gains the newcomer's bit, the other side's is untouched."""
    s = _singles(tables, names, 4)
    for side, slot in ((0, 2), (1, 3)):
        out = step_module._switch_in(tables, s, side, slot, s.turn)
        rs = to_ref(s)
        turn = ref._Turn(to_numpy(tables), None, entry_turn=int(s.turn), battler_count=2)
        ref._enter(turn, rs, side, slot)
        ref._send_out_exp(rs, side)
        assert _sent(out)[side] == 1 | (1 << slot)
        assert _sent(out)[1 - side] == 1
        assert from_ref(rs).sent_out.tolist() == _sent(out)


def test_replacement_marked_even_if_hazards_faint_it(tables, names):
    """The send-out precedes the hazards check (subscript_show_party_list.s:18, :22); a faint
    leaves the bit."""
    s = _singles(tables, names)
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1)).replace_hp(0, 0, 0)
    s = s.replace_hp(0, 1, 1)
    out = step_module.replace(tables, s, 0, 1)
    assert int(out.hp[0, 1]) == 0
    assert _sent(out) == [0b11, 1]
    _same(out, ref.ref_replace(to_numpy(tables), to_ref(s), 0, 1))


def test_pivot_marks_the_pick(tables, names):
    """U-turn's pick sends the newcomer out (subscript_attack_then_switch_out.s:71 -> show_party_list)."""
    s = _singles(tables, names)
    paused = assert_matches_reference(tables, s, (1, 0), O)
    assert int(paused.pivot_battler) == 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[0] = MOVE_ACTIONS + 2
    after = assert_matches_reference(tables, paused, tuple(acts), O)
    assert int(after.active[0]) == 2
    assert _sent(after)[0] == 0b101


def test_doubles_battler_2_marks_its_own_byte(tables, names):
    """Rule 4: DOUBLES battler 2 sends from party 0 into its own byte; MULTI battler 2 is the
    partner's party. Engine ``replace_round`` against ``ref_replace_round``."""
    for fmt, party_row, expected in (
        (DOUBLES, 0, [1, 1, 0b10010, 1]),
        (MULTI, 2, [1, 1, 0b10001, 1]),
    ):
        s = four(tables, names, fmt)
        s = down(s, (party_row, int(s.active[2])))
        p = np.asarray([-1, -1, 4, -1], np.int8)
        out = step_module.replace_round(tables, s, p)
        # DOUBLES: battler 2 leads slot 1, so 1 << 1 | 1 << 4; MULTI: its lead is slot 0.
        assert _sent(out) == expected, fmt
        _same(out, ref.ref_replace_round(to_numpy(tables), to_ref(s), p.tolist()))


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_sent_out_contains_exp_mask_singles(tables, names, data):
    """Block D45-18-S's invariant: the experience mask's slots were shown by battler 0, and every
    battler's active slot was sent out."""
    s = data.draw(supported_state(tables, names))
    sent = np.asarray(s.sent_out).astype(int)
    assert int(s.exp_mask) & ~sent[0] == 0
    for b in range(2):
        assert (sent[b] >> int(s.active[b])) & 1 == 1
