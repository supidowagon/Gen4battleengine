"""H-6: the trainer AI's reads of Filter (111), Solid Rock (116), Levitate (26) and Scrappy (113)
through ``BattleSystem_ApplyTypeChart`` (battle_lib.c:2560-2679) and
``BattleSystem_CalcEffectiveness`` (:2681-2741): ``IfMoveEffectivenessEquals``
(trainer_ai.c:1311-1346), the voluntary-switch passes (:3769-3794) and ``BattleAI_PostKOSwitchIn``
(battle_lib.c:8002-8075). Vanilla tables; every state is Roark's fixture party with the
player's lead made a holder (``as_species``) and its types set by the test."""

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai import replacement
from nuzline.ai.commands import TypeMulti, effectiveness_class
from nuzline.ai.switching import (
    INEFFECTIVE,
    LEVITATED,
    SUPER_EFFECTIVE,
    SwitchSite,
    apply_type_chart,
    calc_effectiveness_flags,
    chart_flags,
    switch_sites,
    switching_refusals,
    type_chart_flags,
)
from nuzline.engine.field import gravity_count
from tests.engine.helpers import as_species, roark_state

FILTER, SOLID_ROCK = c.ABILITY_IDS["Filter"], c.ABILITY_IDS["Solid Rock"]
LEVITATE, SCRAPPY = c.ABILITY_IDS["Levitate"], c.ABILITY_IDS["Scrappy"]
GEODUDE, CRANIDOS = 0, 2  # Roark's party slots; Cranidos has Mold Breaker (roark.md section 4)


def _holder(tables, ability: int) -> tuple[int, int]:
    """``(species, ability slot)`` of the first species whose personal row holds ``ability``."""
    abilities = np.asarray(tables.species_abilities_decomp)
    rows = np.asarray(tables.species_form_row)[:, 0]
    for species in range(1, rows.shape[0]):
        for slot in (0, 1):
            if abilities[rows[species], slot] == ability:
                return species, slot
    raise AssertionError(ability)


def _setup(tables, names, *, holder, attacker=("Rock", "Ground"), defender=("Normal",) * 2):
    """The AI's Geodude (types ``attacker``) against the player's lead, a ``holder`` ability
    Pokemon of types ``defender``; no moves yet."""
    species, slot = _holder(tables, holder) if holder else (names.species("Starly"), 0)
    s = roark_state(tables, names)
    s = as_species(tables, s, 0, 0, species, ability_slot=slot)

    def types(pair):
        return jnp.asarray([names.type(t) for t in pair], s.types.dtype)

    return s._replace(
        types=s.types.at[0, 0].set(types(defender)).at[1, GEODUDE].set(types(attacker)),
        moves=s.moves.at[1, GEODUDE].set(0),
    )


def _with_move(names, s, move: str, slot: int = GEODUDE):
    return s._replace(moves=s.moves.at[1, slot, 0].set(names.move(move)))


def _cls(tables, s) -> int:
    return int(effectiveness_class(tables, s, int(s.moves[1, int(s.active[1]), 0])))


@pytest.mark.parametrize("ability", [FILTER, SOLID_ROCK])
def test_filter_and_solid_rock_scale_a_super_effective_move(tables, names, ability):
    """lib:2657-2660: a super-effective move with power loses a quarter, ``BattleSystem_Divide(
    damage * 3, 4)``, before ``IfMoveEffectivenessEquals`` maps the value (ai.c:1329-1337):
    x2 80 -> 60, STAB x2 120 -> 90 (no class), x4 160 -> 120 -> DOUBLE; STAB x4 240 -> 180."""
    base = _setup(tables, names, holder=ability, defender=("Grass", "Grass"))
    ember = _with_move(names, base, "Ember")
    assert _cls(tables, ember) == 60
    stab = ember._replace(types=ember.types.at[1, GEODUDE].set(names.type("Fire")))
    assert _cls(tables, stab) == 90
    ice = _setup(tables, names, holder=ability, defender=("Grass", "Ground"))
    ice = _with_move(names, ice, "Ice Beam")
    assert _cls(tables, ice) == TypeMulti.DOUBLE
    ice_stab = ice._replace(types=ice.types.at[1, GEODUDE].set(names.type("Ice")))
    assert _cls(tables, ice_stab) == 180
    # Neutral and resisted moves keep their classes; a move without chart effect is unscaled.
    assert _cls(tables, _with_move(names, base, "Tackle")) == TypeMulti.BASE
    assert _cls(tables, _with_move(names, base, "Water Gun")) == TypeMulti.HALF


def test_filter_is_hidden_by_mold_breaker_and_gastric_juice(tables, names):
    """``Battler_IgnorableAbility`` (lib:3108-3121): the attacker's Mold Breaker, or the
    holder's suppressed ability (``Battler_Ability``, :3088-3093), leaves the plain class."""
    s = _with_move(
        names, _setup(tables, names, holder=FILTER, defender=("Grass", "Grass")), "Ember"
    )
    assert _cls(tables, s) == 60
    suppressed = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    assert _cls(tables, suppressed) == TypeMulti.DOUBLE
    # Cranidos (Mold Breaker) active, Ember in its first slot.
    breaker = s._replace(
        active=s.active.at[1].set(CRANIDOS),
        moves=s.moves.at[1, CRANIDOS].set(s.moves[1, GEODUDE]),
        types=s.types.at[1, CRANIDOS].set(s.types[1, GEODUDE]),
    )
    assert _cls(tables, breaker) == TypeMulti.DOUBLE


@pytest.mark.parametrize("ability", [FILTER, SOLID_ROCK])
def test_filter_scales_the_post_ko_stage_2_score(tables, names, ability):
    """lib:8055-8075: ``ApplyTypeChart`` of the fainted battler's move scales the score by
    Filter's x3/4 (``BattleSystem_Divide``); the same value without Filter is the plain walk."""
    s = _with_move(
        names, _setup(tables, names, holder=ability, defender=("Grass", "Grass")), "Ember"
    )
    plain = _setup(tables, names, holder=None, defender=("Grass", "Grass"))
    plain = _with_move(names, plain, "Ember")
    for x in (7, 40, 201):
        got = int(apply_type_chart(tables, s, 1, names.move("Ember"), x))
        want = int(apply_type_chart(tables, plain, 1, names.move("Ember"), x))
        assert want == x * 2
        assert got == max(want * 3 // 4, 1)
    assert int(apply_type_chart(tables, s, 1, names.move("Ember"), 0)) == 0


def _earthquake(tables, names, **kw):
    s = _setup(tables, names, holder=LEVITATE, defender=("Steel", "Steel"))
    return _with_move(names, s, "Earthquake")


def test_levitate_makes_a_ground_move_immune(tables, names):
    """lib:2600-2603 (``LEVITATED``, part of ``MOVE_STATUS_IMMUNE``): the class is 0 although
    Ground is super effective on Steel (80 without Levitate)."""
    s = _earthquake(tables, names)
    assert _cls(tables, s) == TypeMulti.IMMUNE
    plain = _with_move(
        names, _setup(tables, names, holder=None, defender=("Steel",) * 2), "Earthquake"
    )
    assert _cls(tables, plain) == TypeMulti.DOUBLE
    # Not a Ground move: the plain class (Steel resists Water: Fire x2 would also be 80).
    assert _cls(tables, _with_move(names, s, "Fire Punch")) == TypeMulti.DOUBLE
    flags = type_chart_flags(
        tables,
        names.move("Earthquake"),
        s.types[0, 0],
        gravity_count(s),
        **chart_flags(tables, s, 1, 0)._asdict(),
    )
    assert int(flags) == LEVITATED  # no INEFFECTIVE, no SUPER_EFFECTIVE bit


@pytest.mark.parametrize("case", ["gravity", "ingrain", "iron_ball", "mold_breaker", "juice"])
def test_levitate_is_lifted(tables, names, case):
    """``Battler_Ability`` is none under Gravity (lib:3095-3098), for an ingrained holder
    (:3100-3103) and Gastric Juice (:3090-3093); an Iron Ball (:2602) and the attacker's Mold
    Breaker (:3112) lift it too: the class is the plain 80."""
    s = _earthquake(tables, names)
    if case == "gravity":
        s = s._replace(field_mask=s.field_mask | jnp.asarray(5 << 12, s.field_mask.dtype))
    elif case == "ingrain":
        s = s._replace(ingrain=s.ingrain.at[0].set(1))
    elif case == "iron_ball":
        s = s._replace(item=s.item.at[0, 0].set(names.item("Iron Ball")))
    elif case == "juice":
        s = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    else:
        s = s._replace(
            active=s.active.at[1].set(CRANIDOS),
            moves=s.moves.at[1, CRANIDOS].set(s.moves[1, GEODUDE]),
            types=s.types.at[1, CRANIDOS].set(s.types[1, GEODUDE]),
        )
    assert _cls(tables, s) == TypeMulti.DOUBLE


def test_levitate_in_calc_effectiveness(tables, names):
    """lib:2698-2703: INEFFECTIVE alone for a Ground move into the argument ``Levitate``
    unless the attacker argument is Mold Breaker, Gravity or an Iron Ball; Scrappy (:2708-2714)
    steps over the Ghost immunities, no Foresight test."""
    steel = jnp.asarray([names.type("Steel")] * 2, jnp.int8)
    ghost = jnp.asarray([names.type("Ghost")] * 2, jnp.int8)
    quake, tackle = names.move("Earthquake"), names.move("Tackle")
    flags = calc_effectiveness_flags
    assert int(flags(tables, quake, steel, 0, None, False, 0, LEVITATE)) == INEFFECTIVE
    breaker = c.ABILITY_IDS["Mold Breaker"]
    assert int(flags(tables, quake, steel, 0, None, False, breaker, LEVITATE)) == SUPER_EFFECTIVE
    assert int(flags(tables, quake, steel, 5, None, False, 0, LEVITATE)) == SUPER_EFFECTIVE
    assert int(flags(tables, quake, steel, 0, None, True, 0, LEVITATE)) == SUPER_EFFECTIVE
    assert int(flags(tables, tackle, ghost, 0)) == INEFFECTIVE
    assert int(flags(tables, tackle, ghost, 0, None, False, SCRAPPY, 0)) == 0


def test_scrappy_steps_over_the_ghost_immunities(tables, names):
    """lib:2611-2616: the AI's Scrappy (``Battler_Ability``) breaks the walk at the Ghost
    rows: Fighting on Ghost/Dark is x2 (80), on Ghost/Poison x1/2 (20); without Scrappy, or under
    Gastric Juice, a Ghost is immune (0)."""
    kangaskhan, slot = _holder(tables, SCRAPPY)
    for pair, want in (
        (("Ghost", "Dark"), TypeMulti.DOUBLE),
        (("Ghost", "Poison"), TypeMulti.HALF),
    ):
        s = _setup(tables, names, holder=None, attacker=("Normal", "Normal"), defender=pair)
        s = _with_move(names, s, "Karate Chop")
        assert _cls(tables, s) == TypeMulti.IMMUNE
        s = as_species(tables, s, 1, GEODUDE, kangaskhan, ability_slot=slot)
        assert _cls(tables, s) == want
        juiced = s._replace(ability_suppressed=s.ability_suppressed.at[1].set(1))
        assert _cls(tables, juiced) == TypeMulti.IMMUNE


def test_a_ground_move_is_never_a_keep_site_against_levitate(tables, names):
    """``AI_HasSuperEffectiveMove`` (trainer_ai.c:3582) reads ``ApplyTypeChart``: against a
    levitating Steel player Earthquake is not super effective (no keep draw); under Gravity
    it is. The layout is not refused either way."""
    s = _earthquake(tables, names)
    assert switching_refusals(tables, s) == []
    assert SwitchSite.KEEP not in switch_sites(tables, s).tolist()
    grav = s._replace(field_mask=s.field_mask | jnp.asarray(5 << 12, s.field_mask.dtype))
    assert SwitchSite.KEEP in switch_sites(tables, grav).tolist()


def test_post_ko_stage_1_and_2_read_levitate(tables, names):
    """Stage 1 (lib:8002-8010) calls ``CalcEffectiveness`` with the candidate's raw ability
    against ``Battler_Ability(defender)``: a bench member whose only super-effective move is
    Earthquake is no stage-1 pick against a levitating Steel player; stage 2 (:8066-8075)
    scores that move 0 (``MOVE_STATUS_LEVITATED`` is in ``MOVE_STATUS_IMMUNE``), so the
    Tackle member wins."""
    onix, cranidos = 1, CRANIDOS
    s = _earthquake(tables, names)
    s = s._replace(
        moves=s.moves.at[1, onix]
        .set(0)
        .at[1, onix, 0]
        .set(names.move("Earthquake"))
        .at[1, cranidos]
        .set(0)
        .at[1, cranidos, 0]
        .set(names.move("Tackle"))
    )
    plain = as_species(tables, s, 0, 0, names.species("Starly"))
    assert int(replacement.post_ko_switch_in(tables, plain)) == onix
    assert int(replacement.post_ko_switch_in(tables, s)) == cranidos
    assert (
        int(
            replacement._stage_2_score(
                tables, s, names.move("Earthquake"), names.type("Ground"), 10
            )
        )
        == 0
    )


def test_a_levitate_bench_member_is_a_pass_1_candidate(tables, names):
    """ai.c:3769-3777: a Ground ``moveHit`` into a bench member with Levitate is INEFFECTIVE
    (pass 1), so a bench member with a super-effective move of its own is a candidate site;
    the player's Mold Breaker (``Battler_Ability(moveHitBattler)``) lifts it."""
    levitate, slot = _holder(tables, LEVITATE)
    mold = c.ABILITY_IDS["Mold Breaker"]
    s = _setup(tables, names, holder=None, defender=("Normal", "Normal"))
    s = as_species(tables, s, 1, 1, levitate, ability_slot=slot)  # Onix's slot: a Levitate holder
    types = s.types.at[1, 1].set(jnp.asarray([names.type("Normal")] * 2, s.types.dtype))
    s = s._replace(
        types=types,
        moves=s.moves.at[1, 1].set(0).at[1, 1, 0].set(names.move("Wing Attack")),
        move_hit=s.move_hit.at[1].set(names.move("Earthquake")),
    )
    # The player's Normal lead is not weak to Wing Attack: make it a Grass type.
    s = s._replace(
        types=s.types.at[0, 0].set(jnp.asarray([names.type("Grass")] * 2, s.types.dtype))
    )
    assert switching_refusals(tables, s) == []
    assert SwitchSite.PASS_INEFFECTIVE in switch_sites(tables, s).tolist()
    s_mold = as_species(
        tables, s, 0, 0, _holder(tables, mold)[0], ability_slot=_holder(tables, mold)[1]
    )
    s_mold = s_mold._replace(types=s.types)
    assert SwitchSite.PASS_INEFFECTIVE not in switch_sites(tables, s_mold).tolist()
