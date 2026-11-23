"""E4-A-W1 helpers for W3 / W4, not wired into the policy yet: the A-3 target law and overrides,
the A-15 per-target combination, the A-12 per-battler item use. Each is a pure function with an
exact-law test against a brute force over the decomp's own draws (``RandNext() % n`` counted over
all 65536 u16 values)."""

from __future__ import annotations

import itertools
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import policy
from nuzline.ai.outcome import TIEBREAK_P
from nuzline.data.tables import BATTLE_TYPES
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.format import DOUBLES, MULTI
from nuzline.engine.ids import MOVE_CURSE, Type
from tests.engine.d2_states import four, with_hp, with_moves


def _share(n, i):
    """P(RandNext() % n == i), counted over the u16 draws (trainer_ai.c:428, :457)."""
    return Fraction(sum(1 for r in range(65536) if r % n == i), 65536)


def test_mod_laws_through_tiebreak_p():
    # the one `% 12` field read `% n` is the exact law of `% n` for n = 1..4 (A-3, A-15 s2)
    for n in range(1, 5):
        for i in range(n):
            via_field = sum(p for t, p in enumerate(TIEBREAK_P) if t % n == i)
            assert via_field == pytest.approx(float(_share(n, i)), abs=1e-12)
            assert policy._TIE_SHARE[n, i] == pytest.approx(float(_share(n, i)), abs=1e-7)
    assert [int(_share(3, i) * 65536) for i in range(3)] == [21846, 21845, 21845]


@pytest.mark.parametrize("a", [1, 3, 2])
def test_ally_rule(a):
    # ai.c:431-436: the ally a ^ 2 with a best score below 100 is -1 (99 never, 100 stays)
    for best_ally, kept in ((99, -1), (100, 100), (0, -1), (127, 127), (-1, -1)):
        best = [60, 70, 80, 90]
        best[a ^ 2] = best_ally
        out = policy._ally_rule(best, a)
        assert out.tolist() == [*[b for b in best[: a ^ 2]], kept, *best[(a ^ 2) + 1 :]]


BESTS = [
    [-1, 50, -1, -1],  # one target
    [50, 50, -1, -1],
    [50, 50, 50, -1],
    [50, 50, 50, 50],
    [30, 50, 50, 20],
    [50, 20, 30, 50],
    [-1, -1, 7, 7],
    [100, 40, 100, 100],
]


@pytest.mark.parametrize("best", BESTS)
def test_target_law_is_the_modular_draw(best):
    # the list holds the targets at the maximum in battler order, the draw is list[RandNext() %
    # count] (ai.c:440-457): the law counted over all 65536 draws, count 1 included
    top = max(best)
    listed = [t for t in range(4) if best[t] == top]
    want = np.zeros(4)
    for r in range(65536):
        want[listed[r % len(listed)]] += 1 / 65536
    np.testing.assert_allclose(np.asarray(policy._target_law(best), np.float64), want, atol=1e-7)
    # the sampled form over the `% 12` field, weighted by TIEBREAK_P, has the same law
    via_field = np.zeros(4)
    for tiebreak, p in enumerate(TIEBREAK_P):
        via_field[int(policy._pick_target(best, tiebreak))] += p
    np.testing.assert_allclose(via_field, want, atol=1e-7)


def test_target_law_three_way_tie_is_not_a_third():
    law = np.asarray(policy._target_law([50, 50, -1, 50]), np.float64)
    assert [round(float(x) * 65536) for x in law] == [21846, 21845, 0, 21845]


def _moves_state(tables, names, moves_by_battler, fmt=DOUBLES):
    s = four(tables, names, fmt)
    for b, moves in moves_by_battler.items():
        s = with_moves(s, names, b, moves)
    return s


def test_override_target(tables, names):
    # ai.c:461-469: a RANGE_USER_OR_ALLY move aimed at side 0 -> the user (side 0 literally);
    # a non-Ghost Curse -> the user, a Ghost's keeps its target.
    acu, curse = names.move("Acupressure"), MOVE_CURSE
    assert int(tables.move_target[acu]) == 10
    s = _moves_state(tables, names, {})
    over = lambda a, move, t: int(policy._override_target(tables, s, a, move, t))
    assert over(1, acu, 0) == 1 and over(1, acu, 2) == 1  # a player battler -> itself
    assert over(1, acu, 3) == 3 and over(1, acu, 1) == 1  # the ally is kept
    assert over(3, acu, 0) == 3 and over(3, acu, 2) == 3
    # MULTI's partner (battler 2, side 0): a target on side 0 goes to itself, an enemy stays
    assert over(2, acu, 0) == 2 and over(2, acu, 2) == 2
    assert over(2, acu, 1) == 1 and over(2, acu, 3) == 3
    other = names.move("Tackle")
    assert over(1, other, 0) == 0 and over(1, other, 3) == 3
    assert over(1, curse, 0) == 1 and over(3, curse, 2) == 3  # non-Ghost Curse -> self
    ghost = s._replace(
        types=s.types.at[1, 0].set(jnp.asarray([int(Type.GHOST)] * 2, s.types.dtype))
    )
    assert int(policy._override_target(tables, ghost, 1, curse, 0)) == 0


def test_pick_by_battler(tables, names):
    # `_pick` for battler b takes b's moves: battler 3 knows two moves, battler 1 four
    s = _moves_state(
        tables,
        names,
        {1: ("Tackle", "Surf", "Protect", "Curse"), 3: ("Tackle", "Surf")},
    )
    scores = jnp.asarray([50, 50, 50, 50], jnp.int32)
    assert int(policy._pick(tables, s, scores, 3)) == 3  # four candidates, 3 % 4
    assert int(policy._pick(tables, s, scores, 3, 3)) == 1  # two candidates, 3 % 2
    assert int(policy._pick(tables, s, jnp.asarray([10, 50, 50, 50], jnp.int32), 0, 3)) == 1
    # a battler's own slots only: battler 3's unknown slots never win
    assert int(policy._pick(tables, s, jnp.asarray([10, 20, 90, 90], jnp.int32), 0, 3)) == 1


def _reference_joint(hist, evaluated, a, share):
    """Brute force of ``TrainerAI_MainDoubles`` (trainer_ai.c:362-457) over every score of every
    slot of every target: ``{(target, slot): P}``, exact Fractions. ``hist[t][m]`` is
    ``{score: P}``; ``share(n, i)`` the law of ``% n``."""
    per_target = []
    for t in range(4):
        if not evaluated[t]:
            per_target.append({(-1, None): Fraction(1)})
            continue
        out = {}
        slots = [sorted(hist[t][m].items()) for m in range(4)]
        for combo in itertools.product(*slots):
            weight = Fraction(1)
            for _, p in combo:
                weight *= p
            if weight == 0:
                continue
            scores = [v for v, _ in combo]
            top = max(scores)
            tied = [m for m in range(4) if scores[m] == top]
            for rank, m in enumerate(tied):
                value = top
                if t == (a ^ 2) and value < 100:
                    value = -1
                key = (value, m)
                out[key] = out.get(key, Fraction(0)) + weight * share(len(tied), rank)
        per_target.append(out)
    law = {}
    for combo in itertools.product(*[sorted(d.items(), key=str) for d in per_target]):
        weight = Fraction(1)
        for _, p in combo:
            weight *= p
        best = [k[0] for k, _ in combo]
        top = max(best)
        listed = [t for t in range(4) if best[t] == top]
        for rank, t in enumerate(listed):
            m = combo[t][0][1]
            if m is None:
                continue
            law[(t, m)] = law.get((t, m), Fraction(0)) + weight * share(len(listed), rank)
    return law


def _random_hist(rng, values_by_target):
    hist = []
    for t in range(4):
        row = []
        for m in range(4):
            vals = values_by_target[t]
            if not vals:
                row.append({})
                continue
            raw = rng.integers(0, 5, size=len(vals))
            if raw.sum() == 0:
                raw[0] = 1
            row.append({v: Fraction(int(x), int(raw.sum())) for v, x in zip(vals, raw) if x})
        hist.append(row)
    return hist


@pytest.mark.parametrize(
    "a, evaluated, values",
    [
        (1, (1, 0, 1, 1), ((100, 101), (), (100, 101), (99, 100))),  # three targets, ally 99 / 100
        (1, (1, 0, 1, 1), ((100, 101), (), (100, 101), (100, 101))),
        (1, (1, 0, 1, 0), ((100, 101), (), (100, 101), ())),  # ally fainted: two targets
        (1, (1, 0, 0, 1), ((100, 102), (), (), (99, 100, 103))),  # one enemy and the ally
        (3, (1, 1, 1, 0), ((100, 101), (99, 100), (100, 101), ())),
        (2, (1, 1, 0, 1), ((100, 101), (100,), (), (100, 101))),  # MULTI partner: ally is 0
        (1, (0, 0, 1, 0), ((), (), (100, 101), ())),  # one target evaluated
    ],
)
def test_target_slot_law_equals_the_brute_force(a, evaluated, values):
    # A-15 section 2: the combination of the per-target laws is the exact law of (target, slot):
    # the slot pick `% n`, the ally rule, the target pick `% c` through TIEBREAK_P
    rng = np.random.default_rng(a * 10 + sum(evaluated))
    hist = _random_hist(rng, values)
    if a == 2:  # the ally of battler 2 is battler 0
        assert a ^ 2 == 0
    want = _reference_joint(hist, evaluated, a, _share)
    arr = np.zeros((4, 4, 256), np.float32)
    for t in range(4):
        for m in range(4):
            for v, p in hist[t][m].items():
                arr[t, m, v] = float(p)
    got = np.asarray(
        policy._target_slot_law(arr, np.asarray(evaluated, bool), a, jnp.ones(4, bool)),
        np.float64,
    )
    for t in range(4):
        for m in range(4):
            assert got[t, m] == pytest.approx(float(want.get((t, m), 0)), abs=2e-6), (t, m)
    assert got.sum() == pytest.approx(1.0, abs=1e-5)


def test_ally_99_never_and_100_count_law():
    # the ally at 99 is never drawn; at 100 tied with an enemy at 100 it is drawn by the count law
    point = lambda v: np.eye(256, dtype=np.float32)[v]
    hist = np.zeros((4, 4, 256), np.float32)
    for m in range(4):
        hist[0, m], hist[2, m] = point(100), point(100)
        hist[3, m] = point(99)
    ev = np.asarray([1, 0, 1, 1], bool)
    law = np.asarray(policy._target_slot_law(hist, ev, 1, jnp.ones(4, bool)), np.float64).sum(1)
    assert law[3] == 0 and law[0] == pytest.approx(0.5) and law[2] == pytest.approx(0.5)
    hist[3] = np.stack([point(100)] * 4)
    law = np.asarray(policy._target_slot_law(hist, ev, 1, jnp.ones(4, bool)), np.float64).sum(1)
    assert [round(float(x) * 65536) for x in law] == [21846, 0, 21845, 21845]


def test_written_target_law(tables, names):
    # the overrides map (target, slot) to (written target, slot): Acupressure aimed at the player
    # goes to the user, the slot is kept
    s = _moves_state(tables, names, {1: ("Tackle", "Acupressure", "Curse", "Surf")})
    law = np.zeros((4, 4))
    law[0, :] = 0.1  # target 0, every slot
    law[2, :] = 0.15
    out = np.asarray(policy._written_target_law(tables, s, 1, jnp.asarray(law)), np.float64)
    assert out.sum() == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(0.25) and out[1, 2] == pytest.approx(0.25)  # -> battler 1
    assert out[0, 0] == pytest.approx(0.1) and out[2, 3] == pytest.approx(0.15)
    assert out[0, 1] == 0 and out[0, 2] == 0


def _bagged(tables, names, fmt, items, count, hp=1):
    t = tables._replace(battle_type=BATTLE_TYPES["double" if fmt == DOUBLES else "multi"])
    s = four(t, names, fmt)
    bag = jnp.asarray(items + [0] * (4 - len(items)), s.trainer_items.dtype)
    s = s._replace(
        trainer_items=s.trainer_items.at[3].set(bag),
        trainer_item_count=s.trainer_item_count.at[3].set(count),
    )
    return t, with_hp(s, 3, hp)


def test_use_item_by_battler(tables, names):
    # A-12: the bag and HP are the battler's own: battler 3 below a quarter HP uses bag 3's Potion,
    # battler 1 (empty bag) uses none
    potion = names.item("Potion")
    t, s = _bagged(tables, names, DOUBLES, [potion], 1)
    assert int(policy._use_item(t, s, 3)) == FIRST_ITEM_ACTION
    assert int(policy._use_item(t, s, 1)) == policy.NO_ITEM
    healthy = with_hp(s, 3, int(s.max_hp[1, 1]))
    assert int(policy._use_item(t, healthy, 3)) == policy.NO_ITEM
    # Embargo of the battler stops it (:4076-4078)
    embargoed = s._replace(embargo=s.embargo.at[3].set(1))
    assert int(policy._use_item(t, embargoed, 3)) == policy.NO_ITEM


def test_use_item_alive_count_is_the_shared_party(tables, names):
    # DOUBLES: both actives are in the alive count of the shared party (:4080-4092). Slot 1's item
    # is considered only when alive <= count - 1 + 1 = 2.
    potion = names.item("Potion")
    t, s = _bagged(tables, names, DOUBLES, [0, potion], 2)
    assert int(policy._use_item(t, s, 3)) == policy.NO_ITEM  # six alive
    down = s._replace(hp=s.hp.at[1, 2:].set(0))  # battlers 1 and 3 alone alive: two
    assert int(policy._use_item(t, down, 3)) == FIRST_ITEM_ACTION + 1


def test_multi_partner_uses_no_item(tables, names):
    # MULTI's battler 2 never uses an item (:4069-4073), whatever its bag row holds
    potion = names.item("Potion")
    t, s = _bagged(tables, names, MULTI, [potion], 1)
    s = s._replace(
        trainer_items=s.trainer_items.at[2].set(s.trainer_items[3]),
        trainer_item_count=s.trainer_item_count.at[2].set(1),
    )
    s = with_hp(s, 2, 1)
    assert int(policy._use_item(t, s, 2)) == policy.NO_ITEM
    assert int(policy._use_item(t, s, 3)) == FIRST_ITEM_ACTION
