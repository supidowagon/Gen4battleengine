"""Mid-turn accuracy (K3b2 Task 1, docs/kaizo/k3b2-rules.md §0.2b-c): the decomp computes a
move's hit rate at its own check (``BattleControllerPlayer_CheckMoveHitAccuracy``,
battle_controller_player.c:2865-2979), with the stages of that moment, so a faster
Sand-Attack or Double Team changes a later move's hit chance in the same turn.
``outcome.hit[side, 0]`` is the roll's interval class over every threshold the move can
have this turn (``outcome.hit_thresholds``), and ``step`` compares it with the live one."""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nuzline.engine.enumerate import (
    SamplingOnly,
    enumerate_outcomes,
    hit_class_law,
    outcome_laws,
)
from nuzline.engine.ids import ItemId, Stat
from nuzline.engine.outcome import (
    Outcome,
    empty_outcome,
    hit_class,
    hit_threshold,
    sample_outcome,
)
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.strategies import legal_action_pair, outcome_law_state

CHUNK = 1024
HYPOTHESIS = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
}
_laws = jax.jit(outcome_laws)
_step = jax.jit(jax.vmap(step, in_axes=(None, None, None, 0)))
_sample = jax.jit(jax.vmap(sample_outcome, in_axes=(0, None, None, None)))


def laws_of(tables, state, actions):
    return jax.tree.map(np.asarray, _laws(tables, state, actions))


def hit_law(tables, state, actions, side) -> dict[int, int]:
    """``value -> weight`` (out of 100) of ``hit[side, 0]`` in the enumeration."""
    laws = laws_of(tables, state, actions)
    values, weights, den = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
    assert den == 100
    return dict(zip(values, weights, strict=True))


def stepped(tables, state, actions, outcomes: Outcome):
    """``step`` of every row of ``outcomes`` (numpy fields with a leading axis), in fixed
    ``CHUNK`` batches; the result's fields are numpy with the same leading axis."""
    n = len(outcomes.hit)
    parts = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        rows = Outcome(*(np.asarray(f)[pad] for f in outcomes))
        out = _step(tables, state, actions, rows)
        parts.append([np.asarray(f)[: len(idx)] for f in out])
    return [np.concatenate(fs) for fs in zip(*parts)]


def state_keys(fields) -> list[bytes]:
    n = len(fields[0])
    flat = np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in fields], axis=1
    )
    return [flat[i].tobytes() for i in range(n)]


def merged(keys, probs) -> dict[bytes, Fraction | float]:
    out: dict = {}
    for k, p in zip(keys, probs):
        out[k] = out.get(k, 0) + p
    return out


def sand_attack_first(tables, names):
    """Roark's Geodude, faster, uses Sand-Attack; the player's Monferno then Scratches
    (accuracy 100)."""
    s = roark_state(tables, names, opp_moves=["Sand-Attack"], opp_faster=True)
    return s, A("Scratch", "Sand-Attack", s, names)


def hit_probability(tables, state, actions, side) -> float:
    """The exact probability that ``side``'s move connects (the foe loses HP)."""
    outcomes, probs = enumerate_outcomes(laws_of(tables, state, actions))
    out = stepped(tables, state, actions, outcomes)
    hp = out[state._fields.index("hp")]
    foe = 1 - side
    target = int(state.active[foe])
    landed = hp[:, foe, target] < int(state.hp[foe, target])
    return math.fsum(probs[landed])


# --- the bug ---------------------------------------------------------------------


def test_sand_attack_lowers_a_later_move_this_turn(tables, names):
    s, acts = sand_attack_first(tables, names)
    # HitRateByStage[5] = 75/100 (include/data/hit_rate_stages.h) at accStages -1
    # (bcp.c:2873); Scratch's own rate was 100 at the start of the turn.
    assert int(hit_threshold(tables, s, acts)[0]) == 100
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12
    # The stage drop is there after the turn (the move hit: Sand-Attack has accuracy 100).
    out = step(tables, s, acts, empty_outcome())
    assert int(out.stages[0, Stat.ACC]) == -1

    # A slower Sand-Attack comes after Scratch: the turn-start rate stands.
    slow = roark_state(tables, names, opp_moves=["Sand-Attack"])
    assert hit_law(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) == {1: 100}
    assert (
        abs(hit_probability(tables, slow, A("Scratch", "Sand-Attack", slow, names), 0) - 1) < 1e-12
    )


def test_double_team_raises_evasion_for_a_later_move(tables, names):
    s = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    acts = A("Scratch", "Double Team", s, names)
    # evaStages = 6 - statBoosts = -1 (bcp.c:2874): HitRateByStage[5] again.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    assert abs(hit_probability(tables, s, acts, 0) - 0.75) < 1e-12


def test_class_values_against_live_threshold(tables, names):
    s, acts = sand_attack_first(tables, names)
    for value, connects in ((1, True), (75, True), (76, False), (0, False)):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        out = step(tables, s, acts, o)
        assert (int(out.hp[1, 0]) < int(s.hp[1, 0])) == connects, value
    # Without the Sand-Attack, 76 is a hit at 100.
    plain = roark_state(tables, names, opp_faster=True)
    o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(76))
    out = step(tables, plain, A("Scratch", "Leer", plain, names), o)
    assert int(out.hp[1, 0]) < int(plain.hp[1, 0])


# --- the law ---------------------------------------------------------------------


def test_hit_class_law_examples():
    assert hit_class_law([95]) == ([0, 1], [5, 95], 100)
    assert hit_class_law([100]) == ([1], [100], 100)
    assert hit_class_law([166, 100]) == ([1], [100], 100)
    assert hit_class_law([100, 75]) == ([1, 76], [75, 25], 100)
    assert hit_class_law([95, 71]) == ([0, 1, 72], [5, 71, 24], 100)
    assert hit_class_law([0]) == ([0], [100], 100)


@settings(max_examples=40, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_class_law_sums_to_one(tables, names, data):
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    for side in (0, 1):
        possible = laws.hit_possible[side]
        assert possible[0], "the turn-start threshold is always possible"
        assert int(laws.hit_thresholds[side, 0]) == int(hit_threshold(tables, s, acts)[side])
        values, weights, den = hit_class_law(laws.hit_thresholds[side][possible])
        assert sum(weights) == den == 100 and min(weights) > 0
        assert len(set(values)) == len(values)
    try:
        _, probs = enumerate_outcomes(laws)
    except SamplingOnly:
        return
    assert abs(math.fsum(probs) - 1) < 1e-12


def test_no_change_keeps_two_classes(tables, names):
    s = roark_state(tables, names)
    acts = A("Scratch", "Tackle", s, names)  # Tackle 95, Scratch 100, Leer-free turn
    assert hit_law(tables, s, acts, 1) == {0: 5, 1: 95}
    assert hit_law(tables, s, acts, 0) == {1: 100}
    # A faster Sand-Attack on the other side does not split a turn that cannot change it.
    slow = roark_state(tables, names, player_moves=["Sand-Attack"], opp_faster=True)
    acts = A("Sand-Attack", "Tackle", slow, names)
    assert hit_law(tables, slow, acts, 1) == {0: 5, 1: 95}


def test_white_herb_restores_the_stage_before_the_check(tables, names):
    s, acts = sand_attack_first(tables, names)
    mon = int(s.active[0])
    herb = int(np.flatnonzero(np.asarray(tables.item_effect) == ItemId.WHITE_HERB)[0])
    s = s._replace(item=s.item.at[0, mon].set(herb))
    # Over-approximated: the herb's reset to 0 is a threshold the move may have.
    assert hit_law(tables, s, acts, 0) == {1: 75, 76: 25}
    for value in (1, 76):
        o = empty_outcome()._replace(hit=empty_outcome().hit.at[0, 0].set(value))
        assert_matches_reference(tables, s, acts, o)


# --- both engines ----------------------------------------------------------------


def test_accuracy_classes_match_reference(tables, names):
    cases = [sand_attack_first(tables, names)]
    dt = roark_state(tables, names, opp_moves=["Double Team"], opp_faster=True)
    cases.append((dt, A("Scratch", "Double Team", dt, names)))
    # The player's Sand-Attack first, then Geodude's Tackle (95): three classes.
    sa = roark_state(tables, names, player_moves=["Sand-Attack"])
    cases.append((sa, A("Sand-Attack", "Tackle", sa, names)))
    for s, acts in cases:
        laws = laws_of(tables, s, acts)
        for side in (0, 1):
            values, _, _ = hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])
            for value in [*values, 0, 1, 100]:
                o = empty_outcome()._replace(hit=empty_outcome().hit.at[side, 0].set(value))
                assert_matches_reference(tables, s, acts, o)
    assert hit_law(tables, sa, cases[-1][1], 1) == {0: 5, 1: 71, 72: 24}


# --- sampling --------------------------------------------------------------------


def test_sampled_hit_frequencies_match_law(tables, names):
    s = roark_state(tables, names, player_moves=["Sand-Attack"])
    acts = A("Sand-Attack", "Tackle", s, names)
    law = hit_law(tables, s, acts, 1)
    assert law == {0: 5, 1: 71, 72: 24}
    n = 20_000
    keys = jax.random.split(jax.random.key(3), n)
    drawn = np.asarray(_sample(keys, tables, s, acts).hit)[:, 1, 0]
    assert set(np.unique(drawn).tolist()) <= set(law)
    for value, weight in law.items():
        p = weight / 100
        assert abs(np.mean(drawn == value) - p) < 5 * math.sqrt(p * (1 - p) / n), value


def test_hit_class_maps_every_roll():
    thresholds = jnp.array([95, 71, 100, 30], jnp.int32)
    possible = jnp.array([True, True, True, False])
    values, weights, _ = hit_class_law([95, 71, 100])
    got = [int(hit_class(u, thresholds, possible)) for u in range(1, 101)]
    assert [got.count(v) for v in values] == weights
    # Each class value decides every possible threshold exactly as its rolls do.
    for u, v in zip(range(1, 101), got, strict=True):
        for t in (95, 71, 100):
            assert (u <= t) == (0 < v <= t)


# --- soundness: classes equal the raw rolls ----------------------------------------


@settings(max_examples=30, **HYPOTHESIS)
@given(data=st.data())
def test_accuracy_classes_equal_raw_rolls(tables, names, data):
    """With every other field fixed, the class enumeration of both sides' rolls gives the
    same merged children as all 100 x 100 raw rolls ``u`` (stored as is: ``u <= t`` is
    ``0 < u <= t``). A threshold missing from ``hit_thresholds`` would split a class's
    rolls between different states and fail this."""
    s = data.draw(outcome_law_state(tables, names))
    acts = data.draw(legal_action_pair(tables, s))
    laws = laws_of(tables, s, acts)
    assume(laws.relevant.hit[:, 0].any())
    base = jax.tree.map(
        np.asarray, sample_outcome(jax.random.key(data.draw(st.integers(0, 99))), tables, s, acts)
    )

    def rows(values0, values1):
        grid = [(a, b) for a in values0 for b in values1]
        hit = np.repeat(base.hit[None], len(grid), axis=0)
        hit[:, 0, 0] = [a for a, _ in grid]
        hit[:, 1, 0] = [b for _, b in grid]
        fields = {
            name: np.repeat(np.asarray(f)[None], len(grid), axis=0)
            for name, f in zip(Outcome._fields, base)
        }
        fields["hit"] = hit
        return Outcome(**fields), grid

    laws_side = [
        dict(zip(*hit_class_law(laws.hit_thresholds[side][laws.hit_possible[side]])[:2]))
        for side in (0, 1)
    ]
    classes, grid = rows(list(laws_side[0]), list(laws_side[1]))
    probs = [Fraction(laws_side[0][a] * laws_side[1][b], 10_000) for a, b in grid]
    want = merged(state_keys(stepped(tables, s, acts, classes)), probs)
    raw, grid = rows(list(range(1, 101)), list(range(1, 101)))
    got = merged(state_keys(stepped(tables, s, acts, raw)), [Fraction(1, 10_000)] * len(grid))
    assert got == want
