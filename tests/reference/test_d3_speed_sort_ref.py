"""D3-4: the reference's ``_speed_sort`` and the two step cases of tests/engine/test_d3_speed_sort.py
through ``ref_step`` (the action phase of both engines, the whole state field by field)."""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.data.tables import to_numpy
from nuzline.engine.ids import AbilityId
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _speed_sort
from tests.engine.d2_states import with_ability, with_hp, with_moves
from tests.engine.test_d3_action_loop import hp, play
from tests.engine.test_d3_speed_sort import base, run
from tests.engine.test_d3_turn_order import ALL_BITS, item_of, speeds


def _ref_rows(tables, state, qc):
    np_tables, rs = to_numpy(tables), to_ref(state)
    return [_speed_sort(np_tables, rs, list(qc), [int(b) for b in bits]) for bits in ALL_BITS]


def _agree(tables, state, qc=(0, 0, 0, 0)):
    engine, _ = run(tables, state, qc=qc)
    assert engine.tolist() == _ref_rows(tables, state, qc)


def test_sort_matches_the_engine_over_all_64_rows(tables, names):
    s = base(tables, names)
    _agree(tables, speeds(s, (150, 150, 150, 150)))
    _agree(tables, speeds(s, (100, 300, 200, 250)))
    _agree(tables, with_hp(speeds(s, (300, 100, 100, 200)), 0, 0))
    quick = item_of(speeds(s, (300, 200, 100, 250)), 2, names.item("Quick Claw"))
    _agree(tables, quick, qc=(0, 0, 1, 0))
    custap = item_of(speeds(s, (150, 150, 150, 150)), 1, names.item("Custap Berry"))
    _agree(tables, with_hp(custap, 1, 10))
    lag = item_of(speeds(s, (300, 200, 100, 250)), 0, names.item("Lagging Tail"))
    _agree(tables, lag)
    _agree(tables, speeds(s, (100, 300, 200, 250))._replace(trick_room=jnp.int8(3)))


def test_rock_slide_after_scary_face_matches_the_reference(tables, names):
    s = speeds(base(tables, names), (300, 150, 120, 200))
    for b, m in {0: "Scary Face", 1: "Swords Dance", 2: "Rock Slide", 3: "Swords Dance"}.items():
        s = with_moves(s, names, b, [m])
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(3, 0, 1, 0))  # asserts == ref_step
    assert hp(out, 1) < hp(s, 1) and hp(out, 3) == hp(s, 3)


def test_lightning_rod_after_scary_face_matches_the_reference(tables, names):
    s = speeds(base(tables, names), (100, 300, 250, 200))
    s = with_ability(with_ability(s, 2, AbilityId.LIGHTNINGROD), 3, AbilityId.LIGHTNINGROD)
    for b, m in {0: "Thunderbolt", 1: "Scary Face", 2: "Swords Dance", 3: "Swords Dance"}.items():
        s = with_moves(s, names, b, [m])
    out, _ = play(tables, s, [0, 0, 0, 0], None, targets=(1, 2, 0, 0))
    assert hp(out, 3) < hp(s, 3) and hp(out, 2) == hp(s, 2)
