"""Hypothesis differential test: the JAX ``step`` against the plain-Python reference
``ref_step`` on random supported states, legal actions and chance outcomes. Every
disagreement found here is settled by the decomp and logged in docs/discrepancies.md."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.levelup import pay_exp
from nuzline.engine.outcome import outcome_to_py
from nuzline.engine.step import needs_replacement, replace, step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_grant_exp, ref_replace, ref_step
from tests.strategies import (
    FEMALE_ONLY,
    GENDERLESS,
    MALE_ONLY,
    any_outcome,
    legal_action_pair,
    possible_genders,
    supported_state,
)

# Compiled once: every generated state has the same shapes and dtypes (plan ruling).
jit_step = jax.jit(step)


@pytest.mark.slow
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_jax_matches_reference(tables, names, data):
    s = data.draw(supported_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    jax_out = jit_step(tables, s, acts, o)
    ref_out = from_ref(
        ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(o))
    )
    for name, a, b in zip(jax_out._fields, jax_out, ref_out):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_possible_genders_follows_the_decomp_gender_ratio():
    assert possible_genders(GENDERLESS) == (2,)
    assert possible_genders(MALE_ONLY) == (0,)
    assert possible_genders(FEMALE_ONLY) == (1,)
    assert possible_genders(127) == (0, 1)  # a mixed (50:50) ratio


# --- K3d-exp T5: the exp payout (closed form in JAX, the decomp's per-level loop in the
# reference) ------------------------------------------------------------------------------

# Every hold effect the payout reads (engine.levelup), plus no item.
EXP_ITEMS = (
    "",
    "Exp. Share",
    "Lucky Egg",
    "Soothe Bell",
    "Macho Brace",
    "Power Weight",
    "Power Bracer",
    "Power Belt",
    "Power Lens",
    "Power Band",
    "Power Anklet",
)


def _exp_party(draw, tables, names, s, *, near_level: bool):
    """``s`` with the player's exp block redrawn to reach the payout's edges: exp anywhere in
    the level or just short of the next one, EVs near the 510 / 255 caps, friendship near the
    tier limits, the payout items, Pokerus, a Luxury Ball, any participation mask (fainted
    slots included) and pending learn markers; sometimes a level-99 or level-100 Pokemon."""
    a = {name: np.asarray(getattr(s, name)).copy() for name in s._fields}
    rates, table = np.asarray(tables.species_exp_rate), np.asarray(tables.exp_table)
    for slot in np.flatnonzero(a["species"][0]):
        if draw(st.integers(0, 9)) == 0:
            a["level"][0, slot] = draw(st.sampled_from((99, 100)))
        level = int(a["level"][0, slot])
        row = table[rates[a["species"][0, slot]]]
        top = level >= 100
        lo, hi = int(row[level]), int(row[level]) if top else int(row[level + 1]) - 1
        if near_level and not top:
            lo = max(lo, hi - draw(st.integers(0, 60)))
        a["exp"][slot] = draw(st.integers(lo, hi))
        a["item"][0, slot] = names.item(draw(st.sampled_from(EXP_ITEMS)))
        evs = draw(st.sampled_from(("any", "sum", "stat")))
        if evs == "sum":
            room = draw(st.integers(0, 8))
            first = draw(st.integers(0, 5))
            a["evs"][0, slot] = 0
            a["evs"][0, slot, first] = 255
            a["evs"][0, slot, (first + 1) % 6] = 255 - room
        elif evs == "stat":
            a["evs"][0, slot, draw(st.integers(0, 5))] = draw(st.integers(248, 255))
        a["friendship"][0, slot] = draw(
            st.one_of(st.integers(92, 104), st.integers(192, 204), st.integers(246, 255))
        )
        a["pokerus"][slot] = draw(st.integers(0, 1))
        a["friend_bonus"][slot] = draw(st.integers(0, 1))
        if draw(st.integers(0, 4)) == 0:
            a["learn_level"][slot] = draw(st.integers(2, max(2, level)))
            a["learn_cursor"][slot] = draw(st.integers(0, 20))
    a["exp_mask"][...] = draw(st.integers(0, 63)) | (1 << int(a["active"][0]))
    return s._replace(**{name: jnp.asarray(a[name]) for name in s._fields})


jit_pay = jax.jit(pay_exp)


# The lazy strategy's repr carries the tables (72 kB); Hypothesis warns when it builds it.
@pytest.mark.filterwarnings("ignore:Generating overly large repr")
@pytest.mark.slow
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_pay_exp_matches_the_per_level_loop(tables, names, data):
    # The closed form (one level count, telescoped HP, friendship by tier counts) against the
    # decomp's literal per-level ShouldLevelUp loop. A foe up to level 100 pays up to
    # 255 * 100 / 7 = 3642 before the multipliers: many levels at once.
    s = data.draw(supported_state(tables, names))
    s = _exp_party(data.draw, tables, names, s, near_level=data.draw(st.booleans()))
    living = [j for j in range(6) if int(s.species[1, j]) and int(s.hp[1, j])]
    j = data.draw(st.sampled_from(living))
    s = s._replace(level=s.level.at[1, j].set(data.draw(st.integers(1, 100))))
    pending, selected = data.draw(st.booleans()), data.draw(st.integers(0, 3))
    after = s.replace_hp(1, j, 0)
    got = jit_pay(tables, s, after, jnp.asarray(pending), jnp.asarray(selected, jnp.int32))
    want = from_ref(ref_grant_exp(to_numpy(tables), to_ref(after), j, pending, selected))
    for name, a, b in zip(got._fields, got, want):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


jit_replace = jax.jit(replace)


@pytest.mark.slow
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_step_payouts_match_reference(tables, names, data):
    # KOs at every hook point: the opponent's Pokemon have little HP left (a KO by a move, a
    # recoil or Selfdestruct, Pursuit, Roar into Stealth Rock, the sandstorm / hail chip, an
    # end-of-turn residual, Stealth Rock on a replacement), and the player's are close to a
    # level-up, so the payout's level-up changes what the rest of the turn reads.
    s = data.draw(supported_state(tables, names))
    s = _exp_party(data.draw, tables, names, s, near_level=True)
    hp = np.asarray(s.hp).copy()
    for j in range(6):
        if hp[1, j] > 1 and data.draw(st.booleans()):
            hp[1, j] = data.draw(st.integers(1, min(int(hp[1, j]), 8)))
    s = s._replace(hp=jnp.asarray(hp, s.hp.dtype))
    acts = data.draw(legal_action_pair(tables, s))
    o = data.draw(any_outcome())
    jax_out = jit_step(tables, s, acts, o)
    ref_out = from_ref(
        ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(o))
    )
    for name, a, b in zip(jax_out._fields, jax_out, ref_out):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    if int(needs_replacement(jax_out)[1]):
        bench = [
            j
            for j in range(6)
            if j != int(jax_out.active[1]) and int(jax_out.species[1, j]) and int(jax_out.hp[1, j])
        ]
        slot = data.draw(st.sampled_from(bench))
        got = jit_replace(tables, jax_out, 1, slot)
        want = from_ref(ref_replace(to_numpy(tables), to_ref(jax_out), 1, slot))
        for name, a, b in zip(got._fields, got, want):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
