"""E2 batch 4 cases the vanilla differential test does not reach (rulings B4-n,
docs/kaizo/e2-mechanics-ledger.md): Kaizo's Bide, which runs Metal Burst's effect (227) with a
single-target range, in both engines (Kaizo tables)."""

import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.ids import EffectId
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step


def _mon(names, species: str, moves, level=30, speed_iv=15) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15, 15, 15, 15, 15, speed_iv),
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


def _both(tables, state, actions, outcome):
    jax_out = step(tables, state, jnp.asarray(actions, jnp.int32), outcome)
    ref = from_ref(ref_step(to_numpy(tables), to_ref(state), actions, outcome_to_py(outcome)))
    for name, a, b in zip(jax_out._fields, jax_out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    return jax_out


def test_kaizo_bide_is_metal_burst(kaizo_tables, kaizo_names):
    """B4-18: Kaizo's Bide (effect 227, single target, accuracy 0) deals 15 / 10 of the last damage
    the user took this turn (battle_script.c:7166), with no type multiplier, and fails with none
    taken."""
    t, n = kaizo_tables, kaizo_names
    bide = n.move("Bide")
    assert int(t.move_effect[bide]) == EffectId.COUNTER
    outcome = empty_outcome()._replace(hit=jnp.ones_like(empty_outcome().hit))
    # The foe (fast) Tackles first, then the slow Bide user answers.
    s = make_state(t, [_mon(n, "Snorlax", ["Bide"], speed_iv=0)], [_mon(n, "Rattata", ["Tackle"])])
    out = _both(t, s, (0, 0), outcome)
    lost = int(s.hp[0, 0]) - int(out.hp[0, 0])
    assert lost > 0
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == lost * 15 // 10
    # Nothing taken this turn: the foe used a status move.
    s = make_state(t, [_mon(n, "Snorlax", ["Bide"], speed_iv=0)], [_mon(n, "Rattata", ["Growl"])])
    out = _both(t, s, (0, 0), outcome)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    # Kaizo's Bide is Dark-type: SYSCTL_IGNORE_TYPE_CHECKS drops the x2 against a Ghost (and STAB),
    # battle_lib.c:7544-7548.
    s = make_state(t, [_mon(n, "Snorlax", ["Bide"], speed_iv=0)], [_mon(n, "Gastly", ["Tackle"])])
    out = _both(t, s, (0, 0), outcome)
    lost = int(s.hp[0, 0]) - int(out.hp[0, 0])
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == lost * 15 // 10
