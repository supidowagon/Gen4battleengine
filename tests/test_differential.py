"""Hypothesis differential test: the JAX ``step`` against the plain-Python reference
``ref_step`` on random supported states, legal actions and chance outcomes. Every
disagreement found here is settled by the decomp and logged in docs/discrepancies.md."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from nuzline.data.tables import to_numpy
from nuzline.engine.coverage import four_turn_refusals
from nuzline.engine.format import format_of
from nuzline.engine.ids import MOVE_MIMIC
from nuzline.engine.levelup import advance_learns, learn, learn_options, needs_learn, pay_exp
from nuzline.engine.outcome import outcome_to_py
from nuzline.engine.step import needs_replacement, replace, step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import (
    ref_advance_learns,
    ref_grant_exp,
    ref_learn,
    ref_learn_options,
    ref_replace,
    ref_step,
)
from tests.strategies import (
    FEMALE_ONLY,
    GENDERLESS,
    MALE_ONLY,
    any_outcome,
    four_state,
    legal_action_pair,
    legal_actions_four,
    pivot_actions,
    possible_genders,
    recalc_stats,
    supported_state,
    targets_for,
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
    _step_both(tables, s, acts, o, data)


@pytest.mark.slow
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)
@given(data=st.data())
def test_jax_matches_reference_four(tables, names, data):
    """D3-1: the four-battler differential, the acceptance test of batch D3, green at every D3 commit through the
    coverage gate (``four_turn_refusals`` filters the draws the engine does not model yet; until
    D3-3 it refuses every draw)."""
    s = data.draw(four_state(tables, names))
    acts, targets = data.draw(legal_actions_four(tables, s))
    assume(not four_turn_refusals(tables, s, acts))
    o = data.draw(any_outcome(format_of(s)))
    _step_both(tables, s, acts, o, data, targets, jitted=_jit_step_four)


_jit_step_four = jax.jit(step)  # its own compile cache: the B = 4 shapes
_PAUSES: list[int] = []  # the pauses the generator reached (a coverage aid for runs)


def _step_both(tables, s, acts, o, data, targets=None, jitted=None):
    """Both engines on one turn, field for field; a turn that pauses at a U-turn's or Baton
    Pass's party list (E2 batch 6) is resumed with a drawn pick and a fresh outcome (each half
    reads its own draws), again for a second pause in the same turn. The first half uses
    ``targets`` when given (D3-1: the drawn commands' targets); later halves draw
    ``targets_for(s)`` and ``any_outcome(format_of(s))``. ``jitted``: the compiled ``step`` to
    use (default ``jit_step``)."""
    run = jit_step if jitted is None else jitted
    for _ in range(3):
        if targets is None:
            targets = data.draw(targets_for(s))  # D2-1: ignored at B = 2 but for chosen_target
        jax_out = run(tables, s, acts, o, targets)
        ref_out = from_ref(
            ref_step(
                to_numpy(tables),
                to_ref(s),
                tuple(int(x) for x in acts),
                outcome_to_py(o),
                [int(x) for x in targets],
            )
        )
        for name, a, b in zip(jax_out._fields, jax_out, ref_out, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
        if int(jax_out.pivot_battler) == 0:
            return jax_out
        _PAUSES.append(int(jax_out.pivot_battler))
        s = jax_out
        acts = data.draw(pivot_actions(s, acts))
        if format_of(s).B == 4:  # the resumed half goes through the coverage gate as well
            assume(not four_turn_refusals(tables, s, acts))
        o = data.draw(any_outcome(format_of(s)))
        targets = None
    raise AssertionError("a turn paused more than twice")


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
    slots included) and pending learn markers; sometimes a level-99 or level-100 Pokemon. The
    stats and max HP are then ``recalc_stats``' at the drawn level and EVs."""
    a = {name: np.asarray(getattr(s, name)).copy() for name in s._fields}
    rates, table = np.asarray(tables.species_exp_rate), np.asarray(tables.exp_table)
    for slot in np.flatnonzero(a["species"][0]):
        if draw(st.integers(0, 9)) == 0:
            a["level"][0, slot] = draw(st.sampled_from((99, 100)))
        level = int(a["level"][0, slot])
        # The growth rate is the party Pokemon's own: a Transformed active's shadow (G-69).
        own = bool(a["transformed"][0]) and slot == int(a["active"][0])
        row = table[rates[a["shadow_species"][0] if own else a["species"][0, slot]]]
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
    for slot in np.flatnonzero(a["species"][0]):  # the drawn level and EVs' stats
        recalc_stats(tables, a, 0, int(slot))
    return s._replace(**{name: jnp.asarray(a[name]) for name in s._fields})


jit_pay = jax.jit(pay_exp)


def _transformed_payer(tables, names):
    """gb2's falsifying draw (seed 329494325992175728685579227899638383608): the player's
    active (slot 1) is a Pikachu (25, its shadow) Transformed into a Furret (162), level 5, IVs,
    EVs and nature 0, exp 125, 2 HP of its own max 18; the foe's slot 0 is a level-62 Pikachu.
    ``recalc_stats`` must keep the max HP its own: the Furret copy's 23 made the first level's
    ``hp += newMax - oldMax`` reach 0 (pokemon.c:633-645), where the per-level loop stops adding
    HP and the telescoped HP does not (engine 6, reference 0)."""
    from tests.engine.helpers import roark_real_state

    s = roark_real_state(tables, names)
    a = {name: np.asarray(getattr(s, name)).copy() for name in s._fields}
    a["active"][0] = 1
    a["species"][0, 1], a["form"][0, 1], a["level"][0, 1], a["nature"][0, 1] = 162, 0, 5, 0
    a["ivs"][0, 1], a["evs"][0, 1], a["item"][0, 1] = 0, 0, 0
    a["hp"][0, 1], a["max_hp"][0, 1], a["friendship"][0, 1] = 2, 18, 92
    a["transformed"][0], a["shadow_species"][0], a["shadow_form"][0] = 1, 25, 0
    a["shadow_ivs"][0] = 0
    a["exp"][1], a["exp_boost"][1], a["pokerus"][1], a["friend_bonus"][1] = 125, 100, 0, 0
    a["learn_level"][1] = 0
    a["exp_mask"][...] = 1 << 1
    recalc_stats(tables, a, 0, 1)
    a["species"][1, 0], a["form"][1, 0], a["level"][1, 0] = 25, 0, 62
    return s._replace(**{name: jnp.asarray(a[name]) for name in s._fields})


# The lazy strategy's repr carries the tables (72 kB); Hypothesis warns when it builds it.
@pytest.mark.filterwarnings("ignore:Generating overly large repr")
@pytest.mark.slow
@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data(), pinned=st.none())
@example(data=None, pinned=True)  # gb2: a Transformed payer's stale max HP (G-69)
def test_pay_exp_matches_the_per_level_loop(tables, names, data, pinned):
    # The closed form (one level count, telescoped HP, friendship by tier counts) against the
    # decomp's literal per-level ShouldLevelUp loop. A foe up to level 100 pays up to
    # 255 * 100 / 7 = 3642 before the multipliers: many levels at once.
    if pinned:
        s, j, pending, selected = _transformed_payer(tables, names), 0, False, 0
    else:
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
    jax_out = _step_both(tables, s, acts, o, data)
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


# --- K3d-exp T6: the learnset resolve (JAX scan vs the decomp's Pokemon_LevelUpMove loop) ------

jit_advance = jax.jit(advance_learns)
jit_learn = jax.jit(learn)
jit_options = jax.jit(learn_options)


def _assert_same(got, want):
    for name, a, b in zip(got._fields, got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def _learn_party(draw, tables, s):
    """``s`` with pending learn markers (any level up to the Pokemon's, any cursor) and move
    lists edited toward the learnsets: empty slots (gaps included), moves of the row (so
    "already known" and duplicates happen), the volatiles and AI knowledge the options read.
    The learnset is the party Pokemon's own species: a Transformed active's is its shadow
    species and form, as the reference reads it (``_party_moves``; ruling G-69)."""
    a = {name: np.asarray(getattr(s, name)).copy() for name in s._fields}
    rows = np.asarray(tables.species_form_row)
    lm, ll = np.asarray(tables.learnset_move), np.asarray(tables.learnset_level)
    active, transformed = int(a["active"][0]), bool(a["transformed"][0])
    for slot in np.flatnonzero(a["species"][0]):
        level = int(a["level"][0, slot])
        if transformed and slot == active:
            row = rows[a["shadow_species"][0], a["shadow_form"][0]]
        else:
            row = rows[a["species"][0, slot], a["form"][0, slot]]
        learnable = [int(m) for m, lv in zip(lm[row], ll[row], strict=True) if lv]
        if level >= 2 and draw(st.integers(0, 2)):
            a["learn_level"][slot] = draw(st.integers(max(2, level - 20), level))
            # A cursor never passes the terminator (index len(learnable)).
            cursor = draw(st.integers(0, len(learnable)))
            a["learn_cursor"][slot] = draw(st.sampled_from((0, 0, 0, cursor)))
        for k in range(4):
            pick = draw(st.integers(0, 3))
            if pick == 0:
                a["moves"][0, slot, k] = a["pp"][0, slot, k] = 0
            elif pick == 1 and learnable:
                move = draw(st.sampled_from(learnable))
                a["moves"][0, slot, k] = move
                a["pp"][0, slot, k] = int(tables.move_pp[move])
        a["ai_seen_moves"][slot] = 0
        ids = [int(m) for m in a["moves"][0, slot] if m]
        picked = draw(st.lists(st.sampled_from(ids), unique=True, max_size=4)) if ids else []
        a["ai_seen_moves"][slot, : len(picked)] = picked
    side = int(a["active"][0])
    # E1 trial batch (G-68): the active Pokemon's slot may hold a Mimic copy, the party's move
    # (the learn's reads and writes) in the shadow.
    a["mimic_slots"][0] = a["shadow_moves"][0] = a["shadow_pp"][0] = 0
    flagged = [k for k in range(4) if a["moves"][0, side, k]]
    if flagged and draw(st.integers(0, 2)) == 0:
        k = draw(st.sampled_from(flagged))
        a["mimic_slots"][0] = 1 << k
        a["shadow_moves"][0, k] = draw(
            st.sampled_from((0, MOVE_MIMIC, int(a["moves"][0, side, k])))
        )
        a["shadow_pp"][0, k] = draw(st.integers(0, 10)) if a["shadow_moves"][0, k] else 0
    held = [int(m) for m in a["moves"][0, side] if m] or [0]
    for name in ("disabled_move", "encore_move", "locked_move", "choice_lock"):
        a[name][0] = draw(st.sampled_from((0, 0, *held)))
    a["learn_pin"][...] = draw(st.integers(0, 4))
    return s._replace(**{name: jnp.asarray(a[name]) for name in s._fields})


def _transformed_learner(tables, names, cursor: int):
    """The falsifying draw of the Transformed learn (G-69): the player's active (slot 0) is a
    Pidgey (16, its shadow species) Transformed into a Teddiursa (216), level 5, a learn marker
    at level 2 with ``cursor`` into Pidgey's learnset (13 rows: 13 is its terminator, where
    Teddiursa's 14-row learnset would still read a row)."""
    from tests.engine.helpers import roark_real_state

    s = roark_real_state(tables, names)
    a = {name: np.asarray(getattr(s, name)).copy() for name in s._fields}
    a["species"][0, 0], a["form"][0, 0], a["level"][0, 0] = 216, 0, 5
    a["transformed"][0], a["shadow_species"][0], a["shadow_form"][0] = 1, 16, 0
    a["learn_level"][0], a["learn_cursor"][0] = 2, cursor
    return s._replace(**{name: jnp.asarray(a[name]) for name in s._fields})


@pytest.mark.filterwarnings("ignore:Generating overly large repr")
@pytest.mark.slow
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data(), pinned=st.none())
@example(data=None, pinned=13)  # the Transformed active's own terminator (G-69)
@example(data=None, pinned=1)  # the same learner one row in: Sand Attack at level 5
def test_learn_resolve_matches_reference(tables, names, data, pinned):
    if pinned is None:
        s = _learn_party(data.draw, tables, data.draw(supported_state(tables, names)))
        _learns_match_reference(tables, s, lambda: data.draw(st.integers(0, 4)))
    else:
        _learns_match_reference(tables, _transformed_learner(tables, names, pinned))


def _learns_match_reference(tables, s, option=lambda: 4):
    """Advance ``s``'s learns and resolve each (``option()``: the pick) on jit and reference;
    both sides and the options offered must agree (tests/test_pinned_cases.py runs it too)."""
    np_tables = to_numpy(tables)
    got = jit_advance(tables, s)
    _assert_same(got, from_ref(ref_advance_learns(np_tables, to_ref(s))))
    for _ in range(6):
        if not bool(needs_learn(got)):
            break
        options = [bool(x) for x in np.asarray(jit_options(tables, got))]
        assert options == ref_learn_options(np_tables, to_ref(got)) and options[4]
        pick = option()
        want = from_ref(ref_learn(np_tables, to_ref(got), pick))
        got = jit_learn(tables, got, jnp.int32(pick))
        _assert_same(got, want)
