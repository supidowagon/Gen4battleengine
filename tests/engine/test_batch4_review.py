"""E2 batch 4 review follow-ups the vanilla differential test does not reach (rulings B4-19..,
docs/kaizo/e2-mechanics-ledger.md), in both engines: Immunity and Leaf Guard against Toxic Spikes
(unsupported abilities, so the generator never draws them; the hazard reads their decomp ids),
Destiny Bond taken down by Pursuit's switch hit, and Sturdy against an OHKO move."""

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.field import with_weather
from nuzline.engine.ids import Status, Weather
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_step


def _mon(names, species: str, moves, level=30, slot=0) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=slot,
        gender=0,
        friendship=70,
    )


# One compiled step for the module (every state here has the same shapes). Called eagerly,
# step compiled each of its conditionals and loops as programs of their own: 7.9-9.3 GB
# for the six steps of the Toxic Spikes test in the fast tier.
_step = jax.jit(step)


def _both(tables, state, actions, outcome=None):
    outcome = outcome if outcome is not None else empty_outcome()
    jax_out = _step(tables, state, jnp.asarray(actions, jnp.int32), outcome)
    ref = from_ref(ref_step(to_numpy(tables), to_ref(state), actions, outcome_to_py(outcome)))
    for name, a, b in zip(jax_out._fields, jax_out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    return jax_out


def test_immunity_and_leaf_guard_stop_toxic_spikes_and_reveal(tables, names):
    """subscript_poison.s:5-10 / _badly_poison.s:5-10: Immunity, or Leaf Guard in sun, stops the
    Toxic Spikes poison with "{0}'s {1} prevents poisoning!" (TAG_NICKNAME_ABILITY:
    BattleAI_SetAbility, battle_script.c:11806)."""
    t, n = tables, names
    cases = (
        ("Snorlax", 0, Weather.NONE, True),  # Immunity
        ("Tangela", 1, Weather.SUN, True),  # Leaf Guard in sun
        ("Tangela", 1, Weather.NONE, False),  # Leaf Guard without sun: poisoned, no message
    )
    for species, slot, weather, blocked in cases:
        for layers in (1, 2):
            s = make_state(
                t,
                [_mon(n, "Rattata", ["Tackle"]), _mon(n, species, ["Tackle"], slot=slot)],
                [_mon(n, "Pikachu", ["Growl"])],
            )
            s = with_weather(s, weather)._replace(toxic_spikes=s.toxic_spikes.at[0].set(layers))
            out = _both(t, s, (5, 0))
            assert int(out.active[0]) == 1
            want = Status.NONE if blocked else (Status.TOXIC if layers == 2 else Status.POISON)
            assert int(out.status[0, 1]) == want, (species, weather, layers)
            assert bool(out.ai_ability_known[0]) == bool(blocked), (species, weather, layers)


def test_pursuit_on_a_switching_destiny_bond_holder_takes_the_pursuer(tables, names):
    """B4-15: subscript_pursuit.s:85 runs the Destiny Bond faint check after Pursuit's hit on a
    switcher (subscript_faint_check_destiny_bond.s:5-7): a bond holder it KOs takes it down."""
    t, n = tables, names
    s = make_state(
        t,
        [_mon(n, "Rattata", ["Tackle"]), _mon(n, "Pidgey", ["Tackle"])],
        [_mon(n, "Houndour", ["Pursuit"], level=50)],
    )
    s = s.replace_hp(0, 0, 1)._replace(destiny_bond=s.destiny_bond.at[0].set(1))
    out = _both(t, s, (5, 0))
    assert int(out.hp[0, 0]) == 0
    assert int(out.hp[1, 0]) == 0
    # Without the bond the pursuer lives.
    s = s._replace(destiny_bond=s.destiny_bond.at[0].set(0))
    out = _both(t, s, (5, 0))
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) > 0


def test_sturdy_blocks_an_ohko_move_and_reveals(tables, names):
    """BtlCmd_TryOHKOMove (battle_script.c:4340-4341): an ignorable Sturdy sets MOVE_STATUS_STURDY;
    subscript_missed.s:67-69 prints "{0} was protected by {1}!", tagging the ability. A Ghost
    (ineffective, printed first at :59) shows no Sturdy message."""
    t, n = tables, names
    s = make_state(
        t, [_mon(n, "Geodude", ["Tackle"], slot=1)], [_mon(n, "Nidoking", ["Horn Drill"], level=40)]
    )
    out = _both(t, s, (0, 0))
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert int(out.ai_ability_known[0]) != 0
    # Rock Head (slot 0) does not block: the OHKO hits (empty_outcome's roll 1 passes).
    s = make_state(
        t, [_mon(n, "Geodude", ["Tackle"], slot=0)], [_mon(n, "Nidoking", ["Horn Drill"], level=40)]
    )
    out = _both(t, s, (0, 0))
    assert int(out.hp[0, 0]) == 0
    assert int(out.ai_ability_known[0]) == 0
