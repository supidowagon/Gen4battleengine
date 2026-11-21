"""D3-13: ``ref_show_battle_mon4`` against ``switch_in.show_battle_mon4`` (the weather leads over
tie rows and the TWO_TRAINERS replacement case); the U-turn-into-Intimidate case runs through
``ref_step`` in ``tests/engine/test_d3_switch_in_order.py`` (``resume`` asserts engine = reference)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.format import TWO_TRAINERS, format_of, party
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.step import replace_round
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_show_battle_mon4
from tests.engine.d2_states import four
from tests.engine.test_d3_switch_in_order import show, weather_leads


def same(engine_state, ref_state):
    ref = from_ref(ref_state)
    for name, x, y in zip(engine_state._fields, engine_state, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


@pytest.mark.parametrize("ties", [(0,) * 6, (1,) * 6, (1, 0, 1, 0, 0, 1), (0, 1, 1, 0, 1, 0)])
@pytest.mark.parametrize("start", [False, True])
def test_four_weather_leads_match_the_engine(tables, names, ties, start):
    s = weather_leads(tables, names)
    out = show(tables, s, np.asarray(ties, np.int32), start)
    ref = ref_show_battle_mon4(to_numpy(tables), to_ref(s), list(ties), battle_start=start)
    same(out, ref)


def test_two_trainers_replacements_match_the_engine(tables, names):
    s = four(tables, names, TWO_TRAINERS)
    fmt = format_of(s)
    s = s._replace(entry_pending=jnp.zeros_like(s.entry_pending))
    for b in (1, 3):
        s = s._replace(hp=s.hp.at[party(fmt, b), int(s.active[b])].set(0))
    p1, p3 = party(fmt, 1), party(fmt, 3)
    s = s._replace(
        ability=s.ability.at[p1, 1]
        .set(int(AbilityId.DRIZZLE))
        .at[p3, 1]
        .set(int(AbilityId.INTIMIDATE)),
        stats=s.stats.at[p1, 1, Stat.SPE - 1].set(300).at[p3, 1, Stat.SPE - 1].set(100),
    )
    waiting = replace_round(tables, s, np.asarray([-1, 1, -1, 1], np.int8))
    ties = (1, 0, 0, 1, 0, 1)
    out = show(tables, waiting, np.asarray(ties, np.int32))
    ref = ref_show_battle_mon4(to_numpy(tables), to_ref(waiting), list(ties))
    same(out, ref)
