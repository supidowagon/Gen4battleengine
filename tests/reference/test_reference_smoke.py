"""Smoke tests of the plain-Python reference engine against the JAX engine."""

import jax.numpy as jnp
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.damage import damage_table
from nuzline.engine.ids import AbilityId, Status
from nuzline.engine.outcome import empty_outcome, outcome_to_py
from nuzline.engine.step import replace, step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_damage_table, ref_replace, ref_step
from tests.engine.helpers import A, roark_state


def _same(a, b):
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and (x == y).all()


def test_round_trip_exact(tables, names):
    s = roark_state(tables, names)
    back = from_ref(to_ref(s))
    for a, b in zip(s, back):
        assert (a == b).all()


def test_one_turn_matches_jax(tables, names):
    s = roark_state(tables, names)
    acts = A("Scratch", "Tackle", s, names)
    jax_out = step(tables, s, acts, empty_outcome())
    ref_in = to_ref(s)
    ref_out = ref_step(
        to_numpy(tables),
        ref_in,
        tuple(int(x) for x in acts),
        outcome_to_py(empty_outcome()),
    )
    for a, b in zip(jax_out, from_ref(ref_out)):
        assert (a == b).all()
    _same(from_ref(ref_in), s)  # the input state is not modified


def test_damage_tables_match(tables, names):
    s = roark_state(tables, names)
    mv = int(s.moves[0, 0, 0])
    assert damage_table(tables, s, 0, mv, False).tolist() == ref_damage_table(
        to_numpy(tables), to_ref(s), 0, mv, False
    )


def test_replace_matches_jax(tables, names):
    # Stealth Rock, Intimidate and a berry on a faint replacement (step.replace).
    s = roark_state(tables, names).replace_hp(1, 0, 0)
    s = s._replace(
        stealth_rock=s.stealth_rock.at[1].set(1),
        ability=s.ability.at[1, 2].set(AbilityId.INTIMIDATE),
        item=s.item.at[1, 2].set(names.item("Sitrus Berry")),
    ).replace_hp(1, 2, int(s.max_hp[1, 2]) // 2 + 1)
    _same(replace(tables, s, 1, 2), from_ref(ref_replace(to_numpy(tables), to_ref(s), 1, 2)))


def test_reference_fainted_pokemon_has_no_status(tables, names):
    # battle_controller.c:1482-1486: a Pokemon with 0 HP keeps no status condition.
    s = roark_state(tables, names).replace_hp(1, 0, 1).with_status(1, 0, Status.BURN)
    acts = A("Scratch", "Tackle", s, names)
    out = ref_step(
        to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(empty_outcome())
    )
    geodude = out.sides[1].party[0]
    assert geodude.hp == 0
    assert geodude.status == 0 and geodude.status_counter == 0


def _scenarios(tables, names):
    """The decomp rulings recorded in docs/discrepancies.md, as (state, actions, outcome)."""
    o = empty_outcome()
    faint = roark_state(tables, names).replace_hp(1, 0, 1).with_stages(1, atk=-2)
    faint = faint._replace(taunt=faint.taunt.at[1].set(3), confusion=faint.confusion.at[1].set(2))
    struggle = roark_state(tables, names).with_pp(0, 0, [0, 0, 0, 0]).replace_hp(0, 0, 1)
    struggle = struggle._replace(ability=struggle.ability.at[1, 0].set(AbilityId.STATIC))
    rocks = roark_state(
        tables,
        names,
        player_moves=["Stealth Rock"],
        opp_species="Cranidos",
        opp_moves=["Take Down"],
        opp_faster=True,
    ).replace_hp(1, 0, 1)
    cure = roark_state(
        tables, names, player_species="Budew", opp_species="Cranidos", opp_moves=["Pursuit"]
    )
    cure = cure.replace_hp(0, 0, 1).with_status(0, 0, Status.PARALYSIS)
    return {
        "faint_cleanup": (faint, A("Scratch", "Tackle", faint, names), o),
        "struggle_recoil_before_static": (
            struggle,
            A(0, "Tackle", struggle, names),
            o._replace(contact_proc=jnp.ones((2, 5), jnp.int8)),
        ),
        "stealth_rock_after_foe_fainted": (rocks, A("Stealth Rock", "Take Down", rocks, names), o),
        "natural_cure_before_pursuit": (cure, A(4 + 1, "Pursuit", cure, names), o),
    }


@pytest.mark.parametrize(
    "name",
    [
        "faint_cleanup",
        "struggle_recoil_before_static",
        "stealth_rock_after_foe_fainted",
        "natural_cure_before_pursuit",
    ],
)
def test_discrepancy_scenarios_match_jax(tables, names, name):
    s, acts, o = _scenarios(tables, names)[name]
    jax_out = step(tables, s, acts, o)
    ref_out = ref_step(to_numpy(tables), to_ref(s), tuple(int(x) for x in acts), outcome_to_py(o))
    _same(jax_out, from_ref(ref_out))
