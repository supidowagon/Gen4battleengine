"""Rough Skin, Flame Body, Limber, Adaptability and Solid Rock (K3b1 task 10).

* Rough Skin and Flame Body: ``BattleSystem_TriggerAbilityOnHit`` (battle_lib.c:4200), the
  defender's own ability (``Battler_Ability``: Mold Breaker does not matter). Rough Skin
  (:4257-4271) needs a living attacker, no Magic Guard, a contact move that dealt damage, and
  no roll: ``BattleSystem_Divide(-maxHP, 8)`` through ``subscript_rough_skin.s``, which can
  faint the attacker (``TryFaintMon``) and so ends a multi-hit move
  (``BattleControllerPlayer_LoopMultiHit``, battle_controller_player.c:3780). Flame Body
  (:4321-4336) is Static's roll with ``subscript_burn.s`` (Fire types are immune).
* Limber: ``subscript_paralyze.s:5`` (``CheckIgnorableAbility``: Mold Breaker pierces it;
  its "prevents paralysis" message, :_170, only for a direct effect) and
  ``BattleSystem_RecoverStatusByAbility`` (battle_lib.c:4397) at move end and switch-in.
* Adaptability: STAB x2 (battle_lib.c:2592-2597). Solid Rock: a super-effective hit's damage
  ``BattleSystem_Divide(damage * 3, 4)`` after the type chart (battle_lib.c:2657-2660),
  ignorable through Mold Breaker.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai.policy import ai_refusals, player_ability_refusal
from nuzline.engine.damage import ai_damage_for_roll, damage_table, divide
from nuzline.engine.effects import apply_move_effect
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.ids import AbilityId, Status, Type
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state

SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot
_jit_step = jax.jit(step)


def _ability(s, side, slot, ability):
    return s._replace(ability=s.ability.at[side, slot].set(ability))


def _procs(player=0, opp=0):
    """``contact_proc`` of the player's / the opponent's defending Pokemon, every hit."""
    rows = jnp.array([[player] * 5, [opp] * 5], jnp.int8)
    return empty_outcome()._replace(contact_proc=rows)


def _eighth(s, side, slot=0):
    return max(int(s.max_hp[side, slot]) // 8, 1)


# --- Rough Skin -----------------------------------------------------------------------


@pytest.mark.slow
def test_rough_skin_hurts_a_contact_attacker_without_a_roll(tables, names):
    s = _ability(roark_state(tables, names), 1, 0, AbilityId.ROUGH_SKIN)
    out = _jit_step(tables, s, A("Scratch", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) - _eighth(s, 0)
    assert int(out.ai_ability_known[0]) == 0  # the holder is the AI's: nothing to learn
    # A non-contact move (Ember) triggers nothing.
    out = _jit_step(tables, s, A("Ember", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])


def test_rough_skin_reveals_the_player_holder_and_ignores_mold_breaker(tables, names):
    # Cranidos (Mold Breaker) Tackles a Rough Skin player: Battler_Ability, not ignorable.
    s = roark_state(tables, names, opp_species="Cranidos")
    s = _ability(s, 0, 0, AbilityId.ROUGH_SKIN)
    out = _jit_step(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - _eighth(s, 1)
    # subscript_rough_skin.s tags BTLSCR_DEFENDER's ability (TAG_NICKNAME_ABILITY_NICKNAME).
    assert int(out.ai_ability_known[0]) == 1


def test_rough_skin_can_faint_the_attacker_and_ends_a_multi_hit_move(tables, names):
    s = roark_state(tables, names, player_moves=("Fury Swipes",))
    s = _ability(s, 1, 0, AbilityId.ROUGH_SKIN)
    acts = A("Fury Swipes", "Leer", s, names)
    full = _jit_step(tables, s, acts, empty_outcome())  # two hits (n_hits 2)
    two = int(s.hp[1, 0]) - int(full.hp[1, 0])
    assert int(full.hp[0, 0]) == int(s.hp[0, 0]) - 2 * _eighth(s, 0)
    low = s.replace_hp(0, 0, _eighth(s, 0))
    out = _jit_step(tables, low, acts, empty_outcome())
    assert int(out.hp[0, 0]) == 0  # the first hit's Rough Skin faints Monferno
    one = int(s.hp[1, 0]) - int(out.hp[1, 0])
    assert 0 < one < two  # ... and the second hit never comes (faintedMon is set)


def test_rough_skin_needs_damage_and_a_living_attacker(tables, names):
    # Scratch into a Ghost deals nothing (Gastly is not in the fixture: edit the types).
    s = _ability(roark_state(tables, names), 1, 0, AbilityId.ROUGH_SKIN)
    ghost = s._replace(types=s.types.at[1, 0].set(Type.GHOST))
    out = _jit_step(tables, ghost, A("Scratch", "Leer", ghost, names), empty_outcome())
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])


# --- Flame Body -----------------------------------------------------------------------


def test_flame_body_burns_on_the_contact_roll(tables, names):
    s = roark_state(tables, names, player_species="Grotle", player_moves=("Tackle",))
    s = _ability(s, 1, 0, AbilityId.FLAME_BODY)
    acts = A("Tackle", "Leer", s, names)
    assert int(_jit_step(tables, s, acts, _procs(opp=1)).status[0, 0]) == Status.BURN
    assert int(_jit_step(tables, s, acts, _procs(opp=0)).status[0, 0]) == Status.NONE
    # A non-contact move (Razor Leaf) never burns.
    leaf = A("Razor Leaf", "Leer", s, names)
    assert int(_jit_step(tables, s, leaf, _procs(opp=1)).status[0, 0]) == Status.NONE


def test_flame_body_cannot_burn_a_fire_type_and_reveals_the_player(tables, names):
    s = _ability(roark_state(tables, names), 1, 0, AbilityId.FLAME_BODY)  # Monferno: Fire
    out = _jit_step(tables, s, A("Scratch", "Leer", s, names), _procs(opp=1))
    assert int(out.status[0, 0]) == Status.NONE
    p = _ability(roark_state(tables, names), 0, 0, AbilityId.FLAME_BODY)
    out = _jit_step(tables, p, A("Leer", "Tackle", p, names), _procs(player=1))
    assert int(out.status[1, 0]) == Status.BURN
    assert int(out.ai_ability_known[0]) == 1  # subscript_burn.s:_177 tags the holder


@pytest.mark.slow
def test_flame_body_makes_contact_proc_relevant(tables, names):
    s = roark_state(tables, names, player_species="Grotle", player_moves=("Tackle",))
    acts = A("Tackle", "Leer", s, names)
    plain = outcome_laws(tables, s, acts).relevant.contact_proc
    flame = outcome_laws(tables, _ability(s, 1, 0, AbilityId.FLAME_BODY), acts)
    rough = outcome_laws(tables, _ability(s, 1, 0, AbilityId.ROUGH_SKIN), acts)
    assert not np.asarray(plain).any() and not np.asarray(rough.relevant.contact_proc).any()
    assert bool(flame.relevant.contact_proc[1, 0])


# --- Limber ---------------------------------------------------------------------------


def test_limber_blocks_thunder_wave_and_reveals(tables, names):
    s = roark_state(tables, names, opp_moves=("Thunder Wave",))
    s = _ability(s, 0, 0, AbilityId.LIMBER)
    out = _jit_step(tables, s, A("Leer", "Thunder Wave", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 1  # "prevents paralysis" (direct effect)
    bare = _ability(s, 0, 0, AbilityId.NONE)
    out = _jit_step(tables, bare, A("Leer", "Thunder Wave", bare, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.PARALYSIS


def test_limber_blocks_a_secondary_paralysis_silently(tables, names):
    # subscript_paralyze.s:_170 ends an INDIRECT effect with no message.
    s = roark_state(tables, names, opp_moves=("ThunderShock",))
    s = _ability(s, 0, 0, AbilityId.LIMBER)
    o = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))
    out = _jit_step(tables, s, A("Leer", "ThunderShock", s, names), o)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0


def test_limber_attacker_is_not_paralyzed_by_static(tables, names):
    # Static runs subscript_paralyze.s on the attacker; its own Limber stops it, silently.
    s = roark_state(tables, names)
    s = _ability(_ability(s, 0, 0, AbilityId.STATIC), 1, 0, AbilityId.LIMBER)
    out = _jit_step(tables, s, A("Leer", "Tackle", s, names), _procs(player=1))
    assert int(out.status[1, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 0  # no Static message either


def test_mold_breaker_pierces_limber_then_the_move_end_cures(tables, names):
    s = roark_state(tables, names, opp_species="Cranidos", opp_moves=("Thunder Wave",))
    s = _ability(s, 0, 0, AbilityId.LIMBER)
    tw = names.move("Thunder Wave")
    hit = apply_move_effect(
        tables, s, 1, tw, empty_outcome(), 0, connected=True, target_moved=False
    )
    assert int(hit.status[0, 0]) == Status.PARALYSIS  # pierced: no block, no message
    assert int(hit.ai_ability_known[0]) == 0
    out = _jit_step(tables, s, A("Leer", "Thunder Wave", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE  # RecoverStatusByAbility at MoveEnd
    assert int(out.ai_ability_known[0]) == 1  # subscript_ability_forbids_status.s


def test_limber_cures_a_paralyzed_newcomer(tables, names):
    s = _ability(roark_state(tables, names), 0, 1, AbilityId.LIMBER)
    s = s.with_status(0, 1, Status.PARALYSIS)
    out = _jit_step(tables, s, A(SWITCH_TO_1, "Leer", s, names), empty_outcome())
    assert int(out.status[0, 1]) == Status.NONE and int(out.ai_ability_known[1]) == 1


# --- Adaptability and Solid Rock ------------------------------------------------------


def _tables_of(tables, s, side, move, crit=False):
    return np.asarray(damage_table(tables, s, side, move, crit)).astype(np.int64)


def test_adaptability_doubles_stab(tables, names):
    # Ember (Fire) from Monferno into Bidoof (Normal): STAB, neutral chart.
    s = roark_state(tables, names, opp_species="Bidoof")
    ember, scratch = names.move("Ember"), names.move("Scratch")
    adapt = _ability(s, 0, 0, AbilityId.ADAPTABILITY)
    plain = _tables_of(tables, s, 0, ember)
    doubled = _tables_of(tables, adapt, 0, ember)
    # Recover the pre-STAB damage d from plain = d * 15 // 10: the doubled table is 2 * d.
    pre = [next(d for d in range(1, 999) if d * 15 // 10 == p) for p in plain]
    assert doubled.tolist() == [2 * d for d in pre]
    # A non-STAB move is unchanged.
    assert (_tables_of(tables, adapt, 0, scratch) == _tables_of(tables, s, 0, scratch)).all()


def test_solid_rock_cuts_super_effective_hits(tables, names):
    # Bubble (Water) from Prinplup into Geodude (Rock/Ground): x4.
    s = roark_state(tables, names, player_species="Prinplup")
    bubble, pound = names.move("Bubble"), names.move("Pound")
    rock = _ability(s, 1, 0, AbilityId.SOLID_ROCK)
    plain = _tables_of(tables, s, 0, bubble)
    cut = _tables_of(tables, rock, 0, bubble)
    assert cut.tolist() == [int(divide(jnp.int32(d * 3), 4)) for d in plain]
    assert (cut < plain).all()
    # Not super effective (Pound vs Rock: x0.5): unchanged.
    assert (_tables_of(tables, rock, 0, pound) == _tables_of(tables, s, 0, pound)).all()
    # A super-effective and a resisted type cancel (MOVE_STATUS_* flags, lib.c:7557-7574).
    mixed = rock._replace(
        types=rock.types.at[1, 0].set(jnp.array([Type.GRASS, Type.GROUND], jnp.int8))
    )
    base = s._replace(types=mixed.types)
    assert (_tables_of(tables, mixed, 0, bubble) == _tables_of(tables, base, 0, bubble)).all()
    # Mold Breaker ignores it (Battler_IgnorableAbility).
    breaker = _ability(rock, 0, 0, AbilityId.MOLD_BREAKER)
    unbroken = _ability(s, 0, 0, AbilityId.MOLD_BREAKER)
    assert (_tables_of(tables, breaker, 0, bubble) == _tables_of(tables, unbroken, 0, bubble)).all()


def test_solid_rock_floors_at_one_and_feeds_the_ai_estimate(tables, names):
    # Divide(1 * 3, 4) is 1 (battle_lib.c:3614-3616), not 0.
    assert int(divide(jnp.int32(3), 4)) == 1
    # TrainerAI_CalcDamage runs ApplyTypeChart (Solid Rock included) before the variance.
    s = roark_state(tables, names, opp_species="Geodude", opp_moves=("Rock Throw",))
    s = s._replace(types=s.types.at[0, 0].set(jnp.array([Type.FIRE, Type.FIRE], jnp.int8)))
    throw = names.move("Rock Throw")
    rock = _ability(s, 0, 0, AbilityId.SOLID_ROCK)
    for roll in (0, 15):
        plain = int(ai_damage_for_roll(tables, s, 1, throw, roll))
        cut = int(ai_damage_for_roll(tables, rock, 1, throw, roll))
        assert cut < plain


def test_tables_map_the_abilities_in_both_games(tables, names, kaizo_tables, kaizo_names):
    want = {
        "Rough Skin": AbilityId.ROUGH_SKIN,
        "Flame Body": AbilityId.FLAME_BODY,
        "Limber": AbilityId.LIMBER,
        "Adaptability": AbilityId.ADAPTABILITY,
        "Solid Rock": AbilityId.SOLID_ROCK,
    }
    vanilla = {"Carvanha": 0, "Magby": 0, "Glameow": 0, "Eevee": 1, "Camerupt": 1}
    for species, slot in vanilla.items():
        i = names.species(species)
        name = names.ability_names[names.species_abilities(i)[slot]]
        assert int(tables.species_abilities[i, slot]) == want[name], species
    # Survey 1.4: the Kaizo holders on Roark's and the player's side.
    kaizo = {"Lileep": 0, "Corsola": 0, "Shuckle": 0, "Sandshrew": 1, "Delibird": 0}
    kaizo |= {"Vulpix": 0, "Meowth": 1}
    for species, slot in kaizo.items():
        i = kaizo_names.species(species)
        name = kaizo_names.ability_names[kaizo_names.species_abilities(i)[slot]]
        assert name in want and int(kaizo_tables.species_abilities[i, slot]) == want[name]


def test_ai_still_refuses_what_it_does_not_score(tables, names):
    # K3c: IfMoveEffectivenessEquals and the AI's damage (TrainerAI_CalcDamage) run
    # ApplyTypeChart with the AI as attacker, so the player's Solid Rock and the AI's
    # Adaptability change them and stay refused; the AI never computes damage against its
    # own Pokemon or by the player's, so the AI's Solid Rock and the player's Adaptability
    # need nothing, and its transcribed scripts read the other three by decomp id.
    assert player_ability_refusal(names.ability("Solid Rock")) is not None
    for name in ("Rough Skin", "Flame Body", "Limber", "Adaptability"):
        assert player_ability_refusal(names.ability(name)) is None
    s = roark_state(tables, names)
    geodude = int(s.species[1, 0])
    slot = int(s.ability_slot[1, 0])
    edited = tables._replace(
        species_abilities_decomp=tables.species_abilities_decomp.at[geodude, slot].set(
            names.ability("Adaptability")
        )
    )
    assert any("Adaptability" in r for r in ai_refusals(edited, s))


# --- the reference engine -------------------------------------------------------------


def test_k3b_abilities_match_reference(tables, names):
    s = _ability(roark_state(tables, names), 1, 0, AbilityId.ROUGH_SKIN)
    for move in ("Scratch", "Ember"):
        assert_matches_reference(tables, s, A(move, "Leer", s, names), empty_outcome())
    fury = roark_state(tables, names, player_moves=("Fury Swipes",))
    fury = _ability(fury, 1, 0, AbilityId.ROUGH_SKIN)
    for hp in (int(fury.hp[0, 0]), _eighth(fury, 0)):
        low = fury.replace_hp(0, 0, hp)
        assert_matches_reference(tables, low, A("Fury Swipes", "Leer", low, names), empty_outcome())
    mb = _ability(roark_state(tables, names, opp_species="Cranidos"), 0, 0, AbilityId.ROUGH_SKIN)
    assert_matches_reference(tables, mb, A("Leer", "Tackle", mb, names), empty_outcome())

    grotle = roark_state(tables, names, player_species="Grotle", player_moves=("Tackle",))
    grotle = _ability(grotle, 1, 0, AbilityId.FLAME_BODY)
    for p in (0, 1):
        acts = A("Tackle", "Leer", grotle, names)
        assert_matches_reference(tables, grotle, acts, _procs(opp=p))
    fire = _ability(roark_state(tables, names), 1, 0, AbilityId.FLAME_BODY)
    assert_matches_reference(tables, fire, A("Scratch", "Leer", fire, names), _procs(opp=1))
    holder = _ability(roark_state(tables, names), 0, 0, AbilityId.FLAME_BODY)
    assert_matches_reference(tables, holder, A("Leer", "Tackle", holder, names), _procs(player=1))

    for opp in ("Geodude", "Cranidos"):
        tw = roark_state(tables, names, opp_species=opp, opp_moves=("Thunder Wave",))
        tw = _ability(tw, 0, 0, AbilityId.LIMBER)
        assert_matches_reference(tables, tw, A("Leer", "Thunder Wave", tw, names), empty_outcome())
    shock = roark_state(tables, names, opp_moves=("ThunderShock",))
    shock = _ability(shock, 0, 0, AbilityId.LIMBER)
    o = empty_outcome()._replace(secondary=jnp.ones((2, 5), jnp.int8))
    assert_matches_reference(tables, shock, A("Leer", "ThunderShock", shock, names), o)
    static = roark_state(tables, names)
    static = _ability(_ability(static, 0, 0, AbilityId.STATIC), 1, 0, AbilityId.LIMBER)
    assert_matches_reference(tables, static, A("Leer", "Tackle", static, names), _procs(player=1))
    bench = _ability(roark_state(tables, names), 0, 1, AbilityId.LIMBER)
    bench = bench.with_status(0, 1, Status.PARALYSIS)
    assert_matches_reference(tables, bench, A(SWITCH_TO_1, "Leer", bench, names), empty_outcome())

    adapt = _ability(roark_state(tables, names, opp_species="Bidoof"), 0, 0, AbilityId.ADAPTABILITY)
    for move in ("Ember", "Scratch"):
        for roll in (0, 15):
            o = empty_outcome()._replace(roll=jnp.full((2, 5), roll, jnp.int8))
            assert_matches_reference(tables, adapt, A(move, "Leer", adapt, names), o)
    rock = roark_state(tables, names, player_species="Prinplup")
    rock = _ability(rock, 1, 0, AbilityId.SOLID_ROCK)
    mixed = rock._replace(
        types=rock.types.at[1, 0].set(jnp.array([Type.GRASS, Type.GROUND], jnp.int8))
    )
    breaker = _ability(rock, 0, 0, AbilityId.MOLD_BREAKER)
    for st in (rock, mixed, breaker):
        for move in ("Bubble", "Pound"):
            for roll in (0, 15):
                o = empty_outcome()._replace(roll=jnp.full((2, 5), roll, jnp.int8))
                assert_matches_reference(tables, st, A(move, "Leer", st, names), o)
