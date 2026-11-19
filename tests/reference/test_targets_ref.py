"""D2-4: the reference ``_resolve`` equals ``targets.resolve`` over every range, randomize,
chosen target, HP pattern, Follow Me state and draw bit, at B = 4 and B = 2."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.ids import MoveTarget
from nuzline.engine.state import make_state
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _resolve
from tests.engine.d2_states import four, mon, with_hp

# one move per range (the resolver reads only ``move_target`` and Curse's effect)
MOVE_PER_RANGE = {
    MoveTarget.RANGE_SINGLE_TARGET: "Tackle",
    MoveTarget.RANGE_SINGLE_TARGET_SPECIAL: "Counter",
    MoveTarget.RANGE_RANDOM_OPPONENT: "Outrage",
    MoveTarget.RANGE_ADJACENT_OPPONENTS: "Growl",
    MoveTarget.RANGE_ALL_ADJACENT: "Earthquake",
    MoveTarget.RANGE_USER: "Swords Dance",
    MoveTarget.RANGE_USER_SIDE: "Reflect",
    MoveTarget.RANGE_FIELD: "Sunny Day",
    MoveTarget.RANGE_OPPONENT_SIDE: "Spikes",
    MoveTarget.RANGE_ALLY: "Helping Hand",
    MoveTarget.RANGE_USER_OR_ALLY: "Acupressure",
    MoveTarget.RANGE_SINGLE_TARGET_ME_FIRST: "Me First",
}
ORDERS = ([0, 1, 2, 3], [3, 1, 0, 2])


def _sweep(tables, names, base, count, attackers, follow_values):
    t = to_numpy(tables)
    fn = jax.jit(
        lambda s, a, m, rnd, ch, order, bit: targets.resolve(tables, s, a, m, rnd, ch, order, bit)
    )
    checked = 0
    for alive in itertools.product((True, False), repeat=count):
        s = base
        for b, up in enumerate(alive):
            s = s if up else with_hp(s, b, 0)
        for fm in follow_values:
            sd = s._replace(ts_follow_me=jnp.asarray(fm, jnp.int8))
            ref_state = to_ref(sd)
            for (rng, move), rnd, chosen, a, bit, order in itertools.product(
                MOVE_PER_RANGE.items(),
                (False, True),
                range(count),
                attackers,
                (0, 1),
                ORDERS if count == 4 else ([0, 1], [1, 0]),
            ):
                mid = names.move(move)
                got = fn(sd, a, mid, rnd, chosen, jnp.asarray(order), bit)
                ref = _resolve(t, ref_state, a, mid, rnd, chosen, order, bit)
                want = (-1 if ref[0] is None else ref[0], ref[1], ref[2], ref[3])
                have = (int(got.defender), int(got.target_side), int(got.counter), bool(got.drew))
                assert have == want, (rng, rnd, chosen, a, bit, order, alive, fm, have, want)
                checked += 1
    return checked


@pytest.mark.fullset
def test_reference_resolve_equals_engine_at_b4(tables, names):
    base = four(tables, names)
    # Follow Me off, on side 1 (user 1 or 3), and set on the attacker's own side only
    n = _sweep(tables, names, base, 4, (0, 3), ([0, 0], [0, 2], [0, 4], [1, 0]))
    assert n > 10_000


@pytest.mark.fullset
def test_reference_resolve_equals_engine_at_b2(tables, names):
    base = make_state(tables, [mon(names)], [mon(names)])
    n = _sweep(tables, names, base, 2, (0, 1), ([0, 0], [0, 1], [1, 0]))
    assert n > 500
