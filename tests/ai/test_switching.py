"""The voluntary-switch site layout (docs/ai/roark.md 11.1) and its stop-to-action map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import switching
from nuzline.ai.outcome import enumerate_ai_outcomes, sample_ai_outcome
from nuzline.ai.switching import (
    INEFFECTIVE,
    NO_SWITCH,
    NOT_VERY_EFFECTIVE,
    REFUSED_ACTION,
    SUPER_EFFECTIVE,
    SWITCH_NONE,
    SWITCH_REFUSED,
    SWITCH_SITES,
    SwitchSite,
    UnsupportedSwitching,
    apply_type_chart,
    calc_effectiveness_flags,
    switch_action,
    switch_sites,
    switching_refusals,
)
from tests.engine.helpers import roark_real_state, roark_state

K = SwitchSite
CRANIDOS, ONIX = 2, 1  # Roark's party slots (roark.md section 4)


def _trace_f(tables, names):
    """roark.md 13, trace F: Cranidos (40/43, party slot 2) vs Starly (30/30), turn 2,
    moveHit[Cranidos] = Wing Attack."""
    s = roark_real_state(tables, names, player_species="Starly")
    return s._replace(
        active=s.active.at[1].set(CRANIDOS),
        hp=s.hp.at[1, CRANIDOS].set(40),
        turn=jnp.asarray(1, s.turn.dtype),
        move_hit=s.move_hit.at[1].set(names.move("Wing Attack")),
    )


def _actions(tables, s):
    stops = jnp.arange(SWITCH_SITES + 1, dtype=jnp.int8)
    return np.asarray(jax.vmap(lambda st: switch_action(tables, s, st))(stops)).tolist()


def test_decomp_ids_match_the_manifest(names):
    for name, ability in switching.ABILITY_IDS.items():
        assert names.ability(name) == ability, name
    for name, move in switching.MOVE_IDS.items():
        assert names.move(name) == move, name
    assert names.item("Iron Ball") == switching.ITEM_IRON_BALL
    assert names.move("Struggle") == switching.MOVE_STRUGGLE


def test_state_with_no_switch_sites(tables, names):
    # Geodude vs Monferno, turn 1: Rock is SE x NVE against Fire/Fighting (no flag), and
    # moveHit is none, so there is no keep draw and no party pass (roark.md 13, trace A).
    s = roark_state(tables, names)
    assert switch_sites(tables, s).tolist() == [K.NONE] * SWITCH_SITES
    assert _actions(tables, s) == [NO_SWITCH] * (SWITCH_SITES + 1)
    assert switching_refusals(tables, s) == []


def test_trace_f_layout_and_actions(tables, names):
    s = _trace_f(tables, names)
    assert int(s.hp[0, 0]) == 30 and int(s.max_hp[1, CRANIDOS]) == 43
    # Pass 2 only: Geodude's then Onix's Stealth Rock and Rock Throw.
    expected = [K.PASS_NOT_VERY_EFFECTIVE] * 4 + [K.NONE] * 5
    assert switch_sites(tables, s).tolist() == expected
    assert _actions(tables, s) == [4, 4, 5, 5] + [NO_SWITCH] * 6
    # The decision's probabilities (roark.md 13, trace F).
    outs, probs = enumerate_ai_outcomes(tables, s)
    act = np.asarray(jax.vmap(lambda st: switch_action(tables, s, st))(outs.switch_stop))
    probs = np.asarray(probs, np.float64)
    q = 43690 / 65536
    assert abs(probs[act == 4].sum() - (1 - q**2)) < 1e-6
    assert abs(probs[act == 5].sum() - q**2 * (1 - q**2)) < 1e-6
    assert abs(probs[act == NO_SWITCH].sum() - q**4) < 1e-6


def test_keep_draw_then_both_passes(tables, names):
    """Geodude (fixture moves) vs Starly, hit by Tackle; Cranidos made Ghost."""
    s = roark_state(tables, names, player_species="Starly")
    ghost = names.type("Ghost")
    s = s._replace(
        move_hit=s.move_hit.at[1].set(names.move("Tackle")),
        types=s.types.at[1, CRANIDOS].set(jnp.asarray([ghost, ghost], jnp.int8)),
    )
    # Rock Throw (slot 1) is SE against Normal/Flying: one keep draw. Normal is
    # ineffective against Ghost (pass 1: Cranidos), not very effective against
    # Rock/Ground (pass 2: Onix); each has Rock Throw and Stealth Rock, SE against Starly.
    assert (
        switch_sites(tables, s).tolist()
        == [
            K.KEEP,
            K.PASS_INEFFECTIVE,
            K.PASS_INEFFECTIVE,
            K.PASS_NOT_VERY_EFFECTIVE,
            K.PASS_NOT_VERY_EFFECTIVE,
        ]
        + [K.NONE] * 4
    )
    assert _actions(tables, s) == [NO_SWITCH, 6, 6, 5, 5] + [NO_SWITCH] * 5


def test_gates(tables, names):
    s = _trace_f(tables, names)
    none = [K.NONE] * SWITCH_SITES
    # Step 1: trapped.
    assert switch_sites(tables, s._replace(trapped=s.trapped.at[1].set(2))).tolist() == none
    # Step 2: no other living party member.
    assert switch_sites(tables, s._replace(hp=s.hp.at[1, :2].set(0))).tolist() == none
    # Steps 10-11 need a damaging moveHit.
    leer = s._replace(move_hit=s.move_hit.at[1].set(names.move("Leer")))
    assert switch_sites(tables, leer).tolist() == none
    # Step 9: four positive stages end the check before the party passes ...
    boosted = s._replace(stages=s.stages.at[1, 1:3].set(2))
    assert switch_sites(tables, boosted).tolist() == none
    # ... but after the keep draws (Starly made Ghost: Cranidos's Pursuit is SE).
    ghost = names.type("Ghost")
    vs_ghost = boosted._replace(types=s.types.at[0, 0].set(jnp.asarray([ghost, ghost], jnp.int8)))
    assert switch_sites(tables, vs_ghost).tolist() == [K.KEEP] + [K.NONE] * 8
    # A fainted bench member is skipped: only Onix's draws remain.
    no_geodude = s._replace(hp=s.hp.at[1, 0].set(0))
    assert (
        switch_sites(tables, no_geodude).tolist() == [K.PASS_NOT_VERY_EFFECTIVE] * 2 + [K.NONE] * 7
    )
    assert _actions(tables, no_geodude)[:3] == [5, 5, NO_SWITCH]


def test_effectiveness_flags_follow_the_chart_order(tables, names):
    """UpateMoveStatusForTypeMul (battle_lib.c:2781) in chart-row order."""

    def flags(move, t1, t2, gravity=0):
        types = jnp.asarray([names.type(t1), names.type(t2)], jnp.int8)
        return int(calc_effectiveness_flags(tables, names.move(move), types, gravity))

    assert flags("Rock Throw", "Normal", "Flying") == SUPER_EFFECTIVE
    assert flags("Wing Attack", "Rock", "Ground") == NOT_VERY_EFFECTIVE
    assert flags("Rock Throw", "Fire", "Fighting") == 0  # SE then NVE cancel
    assert flags("Spark", "Rock", "Ground") == INEFFECTIVE
    # Electric vs Grass (NVE, row 18) then Ground (immune, row 19): immune clears NVE.
    assert flags("Spark", "Ground", "Grass") == INEFFECTIVE
    assert flags("Mud-Slap", "Flying", "Flying") == INEFFECTIVE
    # Gravity skips the Ground -> Flying immunity (NoImmunityOverrides, :2766).
    assert flags("Mud-Slap", "Flying", "Flying", gravity=5) == 0
    assert flags("Struggle", "Ghost", "Ghost") == 0  # returns before the chart (:2687)
    # CalcEffectiveness ignores power: Stealth Rock counts.
    assert flags("Stealth Rock", "Normal", "Flying") == SUPER_EFFECTIVE


def test_apply_type_chart_stab_then_each_entry(tables, names):
    """BattleSystem_ApplyTypeChart (battle_lib.c:2560): STAB x15/10 (:2592), then each
    chart entry through BattleSystem_Divide (:7547), Struggle unchanged (:2573)."""
    s = roark_state(tables, names)  # Monferno (Fire/Fighting) vs Geodude (Rock/Ground)

    def chart(side, move, value):
        return int(apply_type_chart(tables, s, side, names.move(move), value))

    assert chart(0, "Mach Punch", 7) == 20  # STAB 10, Rock x2, Ground x1
    assert chart(0, "Ember", 40) == 30  # STAB 60, Rock x0.5, Ground x1
    assert chart(0, "Ember", 1) == 1  # Divide keeps a non-zero value at least 1
    assert chart(0, "Scratch", 40) == 20  # no STAB, Rock x0.5
    assert chart(0, "Struggle", 40) == 40
    assert chart(1, "Rock Throw", 40) == 60  # STAB 60, Fire x2, Fighting x0.5
    assert chart(1, "Tackle", 40) == 40
    assert chart(1, "Tackle", 0) == 0


def test_too_many_sites_are_refused(tables, names):
    s = _trace_f(tables, names)
    rock_throw = jnp.full((4,), names.move("Rock Throw"), jnp.int16)
    # Geodude and Onix with four Rock Throws each: 8 pass-2 sites; Cranidos with
    # Rock Throw x4 against Starly: 4 keep sites. 12 > 9.
    s = s._replace(moves=s.moves.at[1, :3].set(rock_throw))
    with pytest.raises(UnsupportedSwitching, match="12 voluntary-switch draws"):
        switch_sites(tables, s)
    with pytest.raises(UnsupportedSwitching):
        sample_ai_outcome(jax.random.PRNGKey(0), tables, s)
    with pytest.raises(UnsupportedSwitching):
        enumerate_ai_outcomes(tables, s)
    # Traced (jit/vmap): every entry is REFUSED and the refusal propagates.
    sites = jax.jit(switch_sites)(tables, s)
    assert sites.tolist() == [K.REFUSED] * SWITCH_SITES
    o = jax.jit(sample_ai_outcome)(jax.random.PRNGKey(0), tables, s)
    assert int(o.switch_stop) == SWITCH_REFUSED
    _, probs = jax.jit(enumerate_ai_outcomes)(tables, s)
    assert np.isnan(np.asarray(probs)).all()
    assert int(jax.jit(switch_action)(tables, s, jnp.int8(SWITCH_NONE))) == REFUSED_ACTION


@pytest.mark.parametrize(
    ("change", "missing"),
    [
        ("player_move", "Foresight"),
        ("opp_move", "Hidden Power"),
        ("item", "Iron Ball"),
        ("ability", "Levitate"),
        ("absorb", "Water Absorb"),
    ],
)
def test_unimplemented_modifiers_are_refused(tables, names, change, missing):
    s = roark_state(tables, names)
    if change == "player_move":
        s = s._replace(moves=s.moves.at[0, 3, 1].set(names.move("Foresight")))
    elif change == "opp_move":
        s = s._replace(moves=s.moves.at[1, 2, 3].set(names.move("Hidden Power")))
    elif change == "item":
        s = s._replace(item=s.item.at[0, 2].set(names.item("Iron Ball")))
    elif change == "ability":  # Gastly's only ability is Levitate
        s = s._replace(species=s.species.at[0, 5].set(names.species("Gastly")))
    else:  # Poliwag (Water Absorb) on Roark's bench: AI_HasAbsorbAbilityInParty
        s = s._replace(species=s.species.at[1, 1].set(names.species("Poliwag")))
    refusals = switching_refusals(tables, s)
    assert any(missing in r for r in refusals), refusals
    with pytest.raises(UnsupportedSwitching, match=missing):
        switch_sites(tables, s)
    assert jax.jit(switch_sites)(tables, s).tolist() == [K.REFUSED] * SWITCH_SITES


def test_only_ineffective_moves_is_refused(tables, names):
    """AI_OnlyIneffectiveMoves (trainer_ai.c:3361) can switch with draws not modelled."""
    s = roark_state(tables, names)
    ghost = names.type("Ghost")
    tackle, headbutt = names.move("Tackle"), names.move("Headbutt")
    s = s._replace(
        types=s.types.at[0, 0].set(jnp.asarray([ghost, ghost], jnp.int8)),
        moves=s.moves.at[1, 0].set(jnp.asarray([tackle, headbutt, 0, 0], jnp.int16)),
    )
    with pytest.raises(UnsupportedSwitching, match="AI_OnlyIneffectiveMoves"):
        switch_sites(tables, s)


def test_shadow_tag_and_arena_trap_block_the_switch(tables, names):
    s = _trace_f(tables, names)
    for species in ("Wobbuffet", "Diglett"):  # Shadow Tag; Diglett's slot 1 is Arena Trap
        slot = names.species_abilities(names.species(species)).index(
            names.ability("Shadow Tag" if species == "Wobbuffet" else "Arena Trap")
        )
        t = s._replace(
            species=s.species.at[0, 0].set(names.species(species)),
            ability_slot=s.ability_slot.at[0, 0].set(slot),
        )
        assert switch_sites(tables, t).tolist() == [K.NONE] * SWITCH_SITES, species


def test_jit_without_64bit(tables, names):
    s = _trace_f(tables, names)
    jaxpr = str(jax.make_jaxpr(switch_sites)(tables, s))
    assert "int64" not in jaxpr and "float64" not in jaxpr
    jaxpr = str(jax.make_jaxpr(switch_action)(tables, s, jnp.int8(0)))
    assert "int64" not in jaxpr and "float64" not in jaxpr
