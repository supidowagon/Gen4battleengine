"""E2 batch 3 cases the random differential test does not reach (rulings B3-n,
docs/kaizo/e2-mechanics-ledger.md): the battle-start switch-in checks (``make_state``, which
only the JAX engine runs), Trace at a double KO's replacements (``replace``, not in the random
test), and Truant against a switch met by Pursuit."""

import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import to_numpy
from nuzline.engine.damage import decomp_abilities
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Weather
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.state import make_state
from nuzline.engine.step import replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace, ref_step
from tests.engine.helpers import jit_step, weather_of, weather_timer_of


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _mon(names, species: str, moves=("Tackle",), slot: int = 0, hp=None) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=20,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=slot,
        gender=0,
        friendship=70,
        current_hp=hp,
    )


def test_battle_start_weather_ability_and_trace(tables, names):
    """B3-13, B3-14: Drought sets a permanent sun at the battle-start check; a Trace lead copies
    the foe's ability (its decomp id follows) and fires the copied Drought too."""
    s = make_state(tables, [_mon(names, "Geodude")], [_mon(names, "Groudon")])
    assert int(weather_of(s)) == Weather.SUN and int(weather_timer_of(s)) == 0
    s = make_state(tables, [_mon(names, "Porygon")], [_mon(names, "Groudon")])
    assert int(s.ability[0, 0]) == AbilityId.DROUGHT
    assert int(decomp_abilities(tables, s)[0, 0]) == ABILITY_DECOMP_IDS[AbilityId.DROUGHT]
    assert int(weather_of(s)) == Weather.SUN
    # A Trace facing Trace copies nothing (ChooseTraceTarget, battle_lib.c:7750-7775).
    s = make_state(tables, [_mon(names, "Porygon")], [_mon(names, "Porygon")])
    assert s.ability[:, 0].tolist() == [AbilityId.TRACE, AbilityId.TRACE]


def test_trace_copies_the_foe_replacement_at_a_double_ko(tables, names):
    """B3-14: the player's Trace newcomer enters against a fainted foe and copies nothing; it
    copies the foe's newcomer when that one enters, in both engines; a switch-out restores it."""
    player = [_mon(names, "Geodude", hp=0), _mon(names, "Porygon")]
    opp = [_mon(names, "Geodude", hp=0), _mon(names, "Miltank")]
    s = make_state(tables, player, opp)
    s = s._replace(hp=s.hp.at[:, 0].set(0))
    t, r = tables, to_numpy(tables)
    out = replace(t, s, 0, 1)
    ref = from_ref(ref_replace(r, to_ref(s), 0, 1))
    _same(out, ref)
    assert int(out.ability[0, 1]) == AbilityId.TRACE
    out2 = replace(t, out, 1, 1)
    _same(out2, from_ref(ref_replace(r, to_ref(out), 1, 1)))
    assert int(out2.ability[0, 1]) == int(out2.ability[1, 1]) == AbilityId.THICK_FAT
    assert int(out2.ai_ability_known[1]) != 0  # the tracer's write (battle_script.c:3403)


def test_a_loafing_truant_does_not_pursue(tables, names):
    """B3-10: a Truant pursuer on its loafing turn does not hit a switcher
    (battle_script.c:6903); on its other turn it does."""
    player = [_mon(names, "Geodude"), _mon(names, "Onix")]
    opp = [_mon(names, "Slakoth", moves=("Pursuit",))]
    acts = jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)
    o = empty_outcome()
    for turn, loafs in ((1, True), (2, False)):
        s = make_state(tables, player, opp)
        s = s._replace(turn=jnp.asarray(turn, s.turn.dtype))  # leads: truant 0
        out = jit_step(tables, s, acts, o)
        ref = ref_step(to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o))
        _same(out, from_ref(ref))
        hit = int(out.hp[0, 0]) < int(s.hp[0, 0])
        assert hit != loafs, turn
