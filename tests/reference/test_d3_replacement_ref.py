"""D3-12 on the reference: ``ref_replace_round`` equals the engine's ``replace_round`` field by
field (``tests/engine/test_d3_replacement.py`` has the rules), and the reference's battler-2
reload and per-battler ``learn_pin`` equal ``levelup.pay_exp``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.format import DOUBLES, MULTI
from nuzline.engine.levelup import pay_exp
from nuzline.engine.step import replace_round
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_grant_exp, ref_replace_round
from tests.engine.d2_states import four
from tests.engine.test_d3_replacement import cell, down, leveling, picks


def assert_same(want, got):
    for name, x, y in zip(want._fields, want, got, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def both(tables, s, p):
    want = replace_round(tables, s, p)
    got = from_ref(ref_replace_round(to_numpy(tables), to_ref(s), [int(x) for x in p]))
    assert_same(want, got)
    return want


def test_ref_doubles_low_first_and_one_left(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 0), cell(s, 2), (0, 4), (0, 5))
    out = both(tables, s, picks(2, -1, -1, -1))
    both(tables, out, picks(-1, -1, 3, -1))
    one = down(four(tables, names, DOUBLES), (0, 0), (0, 1), (0, 3), (0, 4), (0, 5))
    both(tables, one, picks(2, -1, -1, -1))


def test_ref_multi_both_flagged(tables, names):
    s = four(tables, names, MULTI)
    s = down(s, cell(s, 0), cell(s, 2), (0, 4), (2, 3))
    both(tables, s, picks(3, -1, 4, -1))


def test_ref_send_in_order_and_payout_after_the_round(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 1), cell(s, 2))
    table = tables.exp_table[tables.species_exp_rate[int(s.species[0, 2])]]
    s = s._replace(
        hp=s.hp.at[1, 2].set(1).at[0, 2].set(118),
        level=s.level.at[0, 2].set(53),
        max_hp=s.max_hp.at[0, 2].set(118),
        exp=s.exp.at[0, 2].set(int(table[54]) - 1),
        item=s.item.at[0, 2].set(names.item("Exp. Share")),
        stealth_rock=jnp.ones_like(s.stealth_rock),
    )
    both(tables, s, picks(-1, 2, 2, -1))


def test_ref_hazard_faint(tables, names):
    s = down(four(tables, names, DOUBLES), (0, 0))
    s = s._replace(hp=s.hp.at[0, 2].set(1), spikes=s.spikes.at[0].set(3))
    both(tables, s, picks(2, -1, -1, -1))


def test_ref_battler_2_reload_and_pins_equal_the_engine(tables, names):
    s, after = leveling(tables, names)
    s = s._replace(
        power_trick=s.power_trick.at[0].set(1).at[2].set(1),
        transformed=s.transformed.at[2].set(1),
    )
    after = after._replace(power_trick=s.power_trick, transformed=s.transformed)
    pending = [True, False, True, False]
    selected = [1, 0, 3, 0]
    want = pay_exp(
        tables, s, after, jnp.asarray(pending), jnp.asarray(selected, jnp.int32), battler=1
    )
    ref = ref_grant_exp(
        to_numpy(tables), to_ref(after), int(after.active[1]), pending, selected, battler=1
    )
    got = from_ref(ref)
    for name in ("learn_pin", "power_trick_desync", "shadow_stats", "level", "hp", "exp"):
        np.testing.assert_array_equal(
            np.asarray(getattr(want, name)), np.asarray(getattr(got, name)), err_msg=name
        )
