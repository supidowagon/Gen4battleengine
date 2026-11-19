"""Roark's trainer AI: scores, chosen action and action distribution against the
hand traces of docs/ai/roark.md section 13 (tests/ai/fixtures.py)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands
from nuzline.ai.outcome import AI_SITE_MOVES
from nuzline.ai.policy import (
    NO_ITEM,
    UnsupportedAi,
    ai_action_distribution,
    ai_policy,
    ai_refusals,
    ai_scores,
    maybe_switch,
    maybe_use_item,
)
from nuzline.ai.switching import NO_SWITCH, REFUSED_ACTION
from nuzline.engine.actions import NUM_OPP_ACTIONS
from tests.ai.fixtures import (
    CRANIDOS,
    TRACED,
    know_ability,
    outcome,
    trace_a,
    trace_b,
    trace_c,
    trace_e,
    trace_f,
    versus,
)
from tests.engine.helpers import INTERMEDIATE, roark_state


@pytest.mark.slow
def test_hand_traced_scores(tables, names):
    for desc, build, ai_o, exp_scores, exp_action in TRACED:
        s = build(tables, names)
        assert ai_scores(tables, s, ai_o).tolist() == exp_scores, desc
        assert int(ai_policy(tables, s, ai_o)) == exp_action, desc


def test_distribution_sums_to_one_and_is_legal(tables, names):
    s = TRACED[0][1](tables, names)
    d = ai_action_distribution(tables, s)
    assert d.shape == (NUM_OPP_ACTIONS,) and d.dtype == jnp.float32
    assert abs(float(d.sum()) - 1.0) < 1e-5
    assert float(d[4]) == 0.0  # cannot switch to its own slot


def test_ko_move_preferred_when_available(tables, names):
    s = TRACED[0][1](tables, names).replace_hp(0, 0, 1)
    d = ai_action_distribution(tables, s)
    assert (
        float(d[:4].max()) > 0.9
    )  # a damaging move that KOs dominates under the "try to faint" flag


def _dist(tables, s):
    return np.asarray(ai_action_distribution(tables, s), np.float64)


def test_trace_a_distribution(tables, names):
    """roark.md 13, Trace A: P(0) = 3/4 (Stealth Rock's +1 at 128/256 or the tie-break),
    P(1) = 1/4."""
    d = _dist(tables, trace_a(tables, names))
    expected = np.zeros(NUM_OPP_ACTIONS)
    expected[[0, 1]] = [0.75, 0.25]
    np.testing.assert_allclose(d, expected, atol=1e-5)  # float32 sums


def test_trace_b_distribution(tables, names):
    """Trace B: a three-way tie breaks by RandNext % 3 (21846/21845/21845 of 65536)."""
    d = _dist(tables, trace_b(tables, names))
    expected = np.zeros(NUM_OPP_ACTIONS)
    expected[[0, 1, 2]] = [0.5 + 0.5 * 21846 / 65536, 0.5 * 21845 / 65536, 0.5 * 21845 / 65536]
    np.testing.assert_allclose(d, expected, atol=1e-5)  # float32 sums


def test_trace_c_always_headbutt(tables, names):
    d = _dist(tables, trace_c(tables, names))
    assert d[0] == pytest.approx(1.0, abs=1e-5)


def test_trace_f_switch_distribution(tables, names):
    """Trace F: four pass-2 sites, each stopping with p = 21846/65536."""
    q = 43690 / 65536
    d = _dist(tables, trace_f(tables, names))
    expected = np.zeros(NUM_OPP_ACTIONS)
    expected[[0, 4, 5]] = [q**4, 1 - q**2, q**2 * (1 - q**2)]
    np.testing.assert_allclose(d, expected, atol=1e-5)  # float32 sums


def test_trace_e_potion_is_certain(tables, names):
    for build in (trace_e(11), trace_e(7), trace_e(11, onix_fainted=True)):
        d = _dist(tables, build(tables, names))
        assert d[10] == pytest.approx(1.0, abs=1e-5)


# ai_action_distribution of every distinct state of TRACED, recorded at 4ccda05 (before the
# K3c policy layer): the nonzero entries as float.hex of the float32 value. Vanilla is
# frozen bit-identical (plan K3c Global Constraints, Review Focus 5).
_A1_D1_E3 = {0: "0x1.8000000000000p-1", 1: "0x1.0000000000000p-2"}
_B1_G4 = {0: "0x1.5555fe0000000p-1", 1: "0x1.5554000000000p-3", 2: "0x1.5554000000000p-3"}
_POTION = {10: "0x1.0000000000000p+0"}
FROZEN_VANILLA = {
    "A1": _A1_D1_E3,
    "A4": {1: "0x1.0000000000000p+0"},
    "B1": _B1_G4,
    "C1": {0: "0x1.0000000000000p+0"},
    "D1": _A1_D1_E3,
    "G1": {0: "0x1.6aaafe0000000p-1", 1: "0x1.aaaa000000000p-3", 2: "0x1.5554000000000p-4"},
    "G4": _B1_G4,
    "E1": _POTION,
    "E2": _POTION,
    "E3": _A1_D1_E3,
    "E4": _POTION,
    "F switch to Geodude": {
        0: "0x1.9484be0000000p-3",
        4: "0x1.1c738e0000000p-1",
        5: "0x1.f9ad0a0000000p-3",
    },
}


def test_vanilla_distribution_unchanged(tables, names):
    builds = {desc: build for desc, build, *_ in TRACED}
    assert set(FROZEN_VANILLA) <= set(builds)
    # Every distinct build of TRACED is frozen (the others repeat one of these states).
    distinct = {
        (b.__name__, tuple(c.cell_contents for c in b.__closure__ or ())) for b in builds.values()
    }
    assert len(distinct) == len(FROZEN_VANILLA)
    for desc, frozen in FROZEN_VANILLA.items():
        d = np.asarray(ai_action_distribution(tables, builds[desc](tables, names)), np.float32)
        got = {a: float(x).hex() for a, x in enumerate(d) if x != 0 or np.signbit(x)}
        assert got == frozen, desc


def test_maybe_switch_and_item_return_no_action_when_not_taken(tables, names):
    s = trace_a(tables, names)
    assert int(maybe_switch(tables, s, outcome())) == NO_SWITCH
    assert int(maybe_use_item(tables, s)) == NO_ITEM


def test_item_gate_considers_slot_1_only_with_two_alive(tables, names):
    """trainer_ai.c:4092: with 3 alive (count 2) only slot 0 is considered, so an empty
    slot 0 means no item; with 2 alive slot 1 is considered too."""
    s = trace_e(11)(tables, names)
    s = s._replace(trainer_items=s.trainer_items.at[1, 0].set(0))
    assert int(maybe_use_item(tables, s)) == NO_ITEM
    assert int(maybe_use_item(tables, s.replace_hp(1, 1, 0))) == 11


def test_no_item_when_fainted_or_bag_empty(tables, names):
    s = trace_e(7)(tables, names)
    empty = s._replace(trainer_items=jnp.zeros_like(s.trainer_items))
    assert int(maybe_use_item(tables, empty)) == NO_ITEM
    assert int(maybe_use_item(tables, s.replace_hp(1, 0, 0))) == NO_ITEM  # curHP 0 (:4106)


def test_struggle_when_no_move_is_usable(tables, names):
    s = trace_c(tables, names)
    s = s._replace(pp=s.pp.at[1, CRANIDOS].set(0))
    assert ai_scores(tables, s, outcome()).tolist() == [0, 0, 0, 0]
    assert int(ai_policy(tables, s, outcome(tiebreak=2))) == 0


def test_zero_pp_slot_still_counts_for_highest_damage(tables, names):
    """FlagMoveDamageScore compares the true moveset (trainer_ai.c:2799,
    TrainerAI_CalcAllDamage reads battleMons.moves, not PP): Headbutt at 0 PP still makes
    Pursuit NOT_HIGHEST."""
    s = trace_c(tables, names)
    s = s._replace(pp=s.pp.at[1, CRANIDOS, 0].set(0))
    assert ai_scores(tables, s, outcome()).tolist() == [0, 100, 100, 0]


def test_policy_jits_and_vmaps(tables, names):
    s = trace_c(tables, names)
    outs = jax.tree.map(lambda *x: jnp.stack(x), outcome(), outcome((0, 1, 0, 0)))
    acts = jax.jit(jax.vmap(lambda o: ai_policy(tables, s, o)))(outs)
    assert acts.tolist() == [0, 0]


def test_refuses_untranscribed_opponent_move(tables, names):
    s = trace_a(tables, names)
    s = s._replace(moves=s.moves.at[1, 0, 2].set(names.move("Ember")))
    reasons = ai_refusals(tables, s)
    assert any("Ember" in r for r in reasons), reasons
    with pytest.raises(UnsupportedAi, match="Ember"):
        ai_scores(tables, s, outcome())
    with pytest.raises(UnsupportedAi, match="Ember"):
        ai_policy(tables, s, outcome())
    with pytest.raises(UnsupportedAi, match="Ember"):
        ai_action_distribution(tables, s)


def test_refusals_name_moves_and_items_in_the_callers_game(tables, names, vanilla_manifest):
    # Final review I3: a game that renames a slot (Kaizo: Rollout is Accelerock) must not
    # see the shipped vanilla name in its refusal lines.
    from nuzline.data.tables import Names

    ember, belt = names.move("Ember"), names.item("Expert Belt")
    move_names, item_names = (
        list(vanilla_manifest["move_names"]),
        list(vanilla_manifest["item_names"]),
    )
    move_names[ember], item_names[belt] = "Fire Sting", "Belt of Doom"
    other = Names({**vanilla_manifest, "move_names": move_names, "item_names": item_names})
    s = trace_a(tables, names)
    s = s._replace(moves=s.moves.at[1, 0, 2].set(ember))
    assert any("Fire Sting" in r for r in ai_refusals(tables, s, other))
    assert not any("Ember" in r for r in ai_refusals(tables, s, other))
    assert any("Ember" in r for r in ai_refusals(tables, s))  # default: the shipped names
    s = versus(tables, names, CRANIDOS, "Starly")
    s = s._replace(item=s.item.at[1, CRANIDOS].set(belt))
    assert any("Belt of Doom" in r for r in ai_refusals(tables, s, other))


def test_refuses_effectiveness_modifiers(tables, names):
    s = versus(tables, names, CRANIDOS, "Starly")
    belt = s._replace(item=s.item.at[1, CRANIDOS].set(names.item("Expert Belt")))
    assert any("Expert Belt" in r for r in ai_refusals(tables, belt))
    solid_rock = 116  # decomp ABILITY_SOLID_ROCK; ability_slot picks a species slot, so
    # the species' ability table is edited instead
    starly = int(s.species[0, 0])
    edited = tables._replace(
        species_abilities_decomp=tables.species_abilities_decomp.at[starly, 0].set(solid_rock)
    )
    assert any("Solid Rock" in r for r in ai_refusals(edited, s))


def test_refused_state_poisons_traced_results(tables, names):
    s = trace_a(tables, names)
    s = s._replace(moves=s.moves.at[1, 0, 2].set(names.move("Ember")))
    scores = jax.jit(lambda st: ai_scores(tables, st, outcome()))(s)
    action = jax.jit(lambda st: ai_policy(tables, st, outcome()))(s)
    assert scores.tolist() == [-1, -1, -1, -1]
    assert int(action) == REFUSED_ACTION


def test_roark_real_fight_is_supported(tables, names):
    for player in (None, "Starly", "Prinplup", "Grotle", "Shinx", "Budew"):
        assert ai_refusals(tables, versus(tables, names, 0, player)) == []


@pytest.mark.fullset
def test_decomp_ids_match_the_manifest(names):
    for name, ability in commands.ABILITY_IDS.items():
        assert names.ability(name) == ability, name
    for name, move in commands.MOVE_IDS.items():
        assert names.move(name) == move, name
    for name, type_id in commands.TYPE_IDS.items():
        assert names.type(name) == type_id, name
    assert names.item("Expert Belt") == commands.ITEM_EXPERT_BELT


@pytest.mark.fullset
def test_transcribed_move_effects_match_the_decomp():
    """The script paths are transcribed for exactly the moves with script-roll sites, and
    their decomp battle effects match data/intermediate/moves.json."""
    assert set(commands.MOVE_EFFECTS) == set(AI_SITE_MOVES)
    path = INTERMEDIATE / "moves.json"
    if not path.exists():
        pytest.skip("intermediate data not extracted (run tools/extract_decomp.py)")
    moves = json.loads(path.read_text(encoding="utf-8"))
    for move, effect in commands.MOVE_EFFECTS.items():
        assert moves[move]["effect"] == effect, moves[move]["name"]


def test_transcribed_move_effects_match_tables(tables):
    for move, effect in commands.MOVE_EFFECTS.items():
        assert int(tables.move_ai_effect[move]) == effect


def test_effectiveness_class(tables, names):
    """roark.md 7.2: 40 through ApplyTypeChart with STAB, then mapped to a class."""
    cls = commands.TypeMulti
    b = trace_b(tables, names)  # Onix (Rock/Ground) vs Starly (Normal/Flying)
    assert int(commands.effectiveness_class(tables, b, names.move("Rock Throw"))) == cls.DOUBLE
    assert int(commands.effectiveness_class(tables, b, names.move("Stealth Rock"))) == cls.DOUBLE
    assert int(commands.effectiveness_class(tables, b, names.move("Screech"))) == cls.BASE
    a = trace_a(tables, names)  # Geodude vs Monferno (Fire/Fighting): x2 x0.5 with STAB
    assert int(commands.effectiveness_class(tables, a, names.move("Rock Throw"))) == cls.STAB
    ghost = a._replace(types=a.types.at[0, 0].set(jnp.asarray([7, 7], jnp.int8)))
    tackle = names.move("Tackle")
    assert int(commands.effectiveness_class(tables, ghost, tackle)) == cls.IMMUNE
    fire_flying = a._replace(types=a.types.at[0, 0].set(jnp.asarray([10, 2], jnp.int8)))
    sr = names.move("Stealth Rock")
    assert int(commands.effectiveness_class(tables, fire_flying, sr)) == cls.QUADRUPLE


def test_stealth_rock_quadruple_bonus(tables, names):
    """EvalAttack_CheckQuadEffective (script.s:6377) also scores status moves: Geodude's
    Stealth Rock against a Fire/Flying target can gain +2 (site 0, N = 80)."""
    a = trace_a(tables, names)
    s = a._replace(types=a.types.at[0, 0].set(jnp.asarray([10, 2], jnp.int8)))
    rock_throw = int(ai_scores(tables, s, outcome((1, 1, 1, 0)))[1])
    assert ai_scores(tables, s, outcome((0, 1, 1, 0))).tolist() == [102, rock_throw, 0, 0]
    assert ai_scores(tables, s, outcome((1, 1, 1, 0))).tolist() == [100, rock_throw, 0, 0]


def test_ai_reads_player_foresight(tables, names):
    """ApplyTypeChart skips the Ghost immunities for an identified defender
    (battle_lib.c:2611): Headbutt on a Dark/Ghost player is immune, then neutral."""
    cls = commands.TypeMulti
    s = versus(tables, names, CRANIDOS, "Sableye")
    headbutt = names.move("Headbutt")
    assert int(commands.effectiveness_class(tables, s, headbutt)) == cls.IMMUNE
    identified = s._replace(foresight=s.foresight.at[0].set(1))
    dark = names.type("Dark")
    pure_dark = s._replace(types=s.types.at[0, 0].set(jnp.asarray([dark, dark], jnp.int8)))
    expected = int(commands.effectiveness_class(tables, pure_dark, headbutt))
    assert expected == cls.BASE
    assert int(commands.effectiveness_class(tables, identified, headbutt)) == expected


def test_player_foresight_is_not_refused(tables, names):
    s = roark_state(tables, names)
    s = s._replace(moves=s.moves.at[0, 3, 1].set(names.move("Foresight")))
    assert ai_refusals(tables, s) == []


def test_ai_reads_the_form_row_ability(tables, names):
    # The brief's ``attacker_ability(tables, s, side=1)`` is ``commands.attacker_ability``,
    # which always reads side 1 (the AI's active Pokemon) and takes no ``side``.
    import dataclasses

    from nuzline.data.party import load_party
    from nuzline.engine.state import make_state
    from tests.engine.helpers import SAMPLE_PARTY

    sample_spec = load_party(SAMPLE_PARTY, names)[0]
    origin = dataclasses.replace(sample_spec, species=487, form=1, item=names.item("Griseous Orb"))
    s = make_state(tables, [sample_spec], [origin])
    levitate = int(np.asarray(tables.species_abilities_decomp)[501, 0])
    assert levitate == names.ability("Levitate")
    assert int(commands.attacker_ability(tables, s)) == levitate
    # The defender's true ability follows the row too, once the AI knows it.
    swapped = make_state(tables, [origin], [sample_spec])
    swapped = know_ability(tables, swapped)
    assert int(commands.defender_ability(tables, swapped, 0)) == levitate


def test_ai_guesses_an_unknown_ability_from_the_base_species(tables, names):
    # AICmd_LoadBattlerAbility guesses from SpeciesData_GetSpeciesValue(species, ...)
    # (trainer_ai.c:1191-1192), which loads the base species' row (pokemon.c:2283): the
    # player's Giratina-Origin is guessed Pressure until the AI learns its Levitate.
    import dataclasses

    from nuzline.data.party import load_party
    from nuzline.engine.state import make_state
    from tests.engine.helpers import SAMPLE_PARTY

    sample_spec = load_party(SAMPLE_PARTY, names)[0]
    origin = dataclasses.replace(sample_spec, species=487, form=1, item=names.item("Griseous Orb"))
    s = make_state(tables, [origin], [sample_spec])
    pressure, levitate = names.ability("Pressure"), names.ability("Levitate")
    assert (pressure, levitate) == (46, 26)
    for guess in (0, 1):
        assert int(commands.defender_ability(tables, s, guess)) == pressure
    known = know_ability(tables, s)
    for guess in (0, 1):
        assert int(commands.defender_ability(tables, known, guess)) == levitate
