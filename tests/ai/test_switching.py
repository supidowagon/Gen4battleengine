"""The voluntary-switch site layout (docs/ai/roark.md 11.1) and its stop-to-action map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import replacement, switching
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


@pytest.mark.fullset
def test_decomp_ids_match_the_manifest(names):
    for name, ability in switching.ABILITY_IDS.items():
        assert names.ability(name) == ability, name
    for name, move in switching.MOVE_IDS.items():
        assert names.move(name) == move, name
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
    # Step 1: trapped (:3906 VOLATILE_CONDITION_TRAPPED: Bind and Mean Look / Block / Spider Web)
    # or ingrained (:3907); an untrapped opponent's check still fires.
    assert switch_sites(tables, s._replace(trapped=s.trapped.at[1].set(2))).tolist() == none
    assert switch_sites(tables, s._replace(mean_look=s.mean_look.at[1].set(1))).tolist() == none
    assert switch_sites(tables, s._replace(ingrain=s.ingrain.at[1].set(1))).tolist() == none
    assert switch_sites(tables, s).tolist() != none
    # The player's side's trap or Ingrain gates nothing.
    free = s._replace(mean_look=s.mean_look.at[0].set(1), ingrain=s.ingrain.at[0].set(1))
    assert switch_sites(tables, free).tolist() == switch_sites(tables, s).tolist()
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
        ("player_move", "Miracle Eye"),
        ("opp_move", "Weather Ball"),
        ("ability", "Levitate"),
        ("absorb", "Water Absorb"),
    ],
)
@pytest.mark.fullset
def test_unimplemented_modifiers_are_refused(tables, names, change, missing):
    s = roark_state(tables, names)
    if change == "player_move":
        s = s._replace(moves=s.moves.at[0, 3, 1].set(names.move(missing)))
    elif change == "opp_move":
        s = s._replace(moves=s.moves.at[1, 2, 3].set(names.move(missing)))
    elif change == "ability":  # Gastly's only ability is Levitate
        s = s._replace(species=s.species.at[0, 5].set(names.species("Gastly")))
    else:  # Poliwag (Water Absorb) on Roark's bench: AI_HasAbsorbAbilityInParty
        s = s._replace(species=s.species.at[1, 1].set(names.species("Poliwag")))
    refusals = switching_refusals(tables, s)
    assert any(missing in r for r in refusals), refusals
    with pytest.raises(UnsupportedSwitching, match=missing):
        switch_sites(tables, s)
    assert jax.jit(switch_sites)(tables, s).tolist() == [K.REFUSED] * SWITCH_SITES


def test_hidden_power_move_hit_type(tables, names):
    """Review Focus 5: moveHit = the player's Hidden Power is typed by the player's active
    IVs (here Water), not Normal (TrainerAI_MoveType, trainer_ai.c:3208-3220, :3767)."""
    s = _trace_f(tables, names)
    water_ivs = jnp.asarray([30, 31, 31, 30, 31, 30], s.ivs.dtype)  # ids.Stat order
    s = s._replace(ivs=s.ivs.at[0, 0].set(water_ivs))

    def with_hit(move):
        return s._replace(move_hit=s.move_hit.at[1].set(names.move(move)))

    hidden = with_hit("Hidden Power")
    assert switching_refusals(tables, hidden) == []
    layout = switch_sites(tables, hidden).tolist()
    assert layout == switch_sites(tables, with_hit("Water Gun")).tolist()
    assert layout != switch_sites(tables, with_hit("Tackle")).tolist()
    # The AI's own Hidden Power is typed too (E1 batch 14): its active's by TrainerAI_MoveType,
    # a bench member's by Move_CalcVariableType from its own IVs (battle_lib.c:8162-8175).
    own = hidden._replace(moves=hidden.moves.at[1, 2, 3].set(names.move("Hidden Power")))
    assert switching_refusals(tables, own) == []


@pytest.mark.fullset
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
    # switch_action also traces BattleAI_PostKOSwitchIn (the Perish Song step), whose reductions
    # carry ``np.int64`` axis parameters: no 64-bit array type (``i64[`` / ``f64[``) appears.
    jaxpr = str(jax.make_jaxpr(switch_action)(tables, s, jnp.int8(0)))
    assert "i64[" not in jaxpr and "f64[" not in jaxpr and "float64" not in jaxpr


def test_party_hidden_power_takes_each_members_ivs(tables, names):
    """Move_CalcVariableType (battle_lib.c:8162-8175): each party member's Hidden Power is
    typed from its own IVs. All 31: type bits 63, 63 * 15 / 63 + 1 = 16, one more from
    TYPE_MYSTERY up: Dark (17). All 30: bits 0, 0 + 1 = 1: Fighting."""
    from nuzline.ai.switching import party_move_types

    s = _trace_f(tables, names)
    hidden = names.move("Hidden Power")
    s = s._replace(moves=s.moves.at[1, 0, 3].set(hidden).at[1, 1, 3].set(hidden))
    s = s._replace(
        ivs=s.ivs.at[1, 0]
        .set(jnp.full(6, 30, s.ivs.dtype))
        .at[1, 1]
        .set(jnp.full(6, 31, s.ivs.dtype))
    )
    types = np.asarray(party_move_types(tables, s))
    assert (types[0, 3], types[1, 3]) == (1, 17)
    other = int(s.moves[1, 0, 0])
    assert types[0, 0] == int(tables.move_type[other])


def _perishing(tables, names, turns=0):
    """Trace F's Cranidos with Perish Song under way (``perish_song_turns = turns``)."""
    s = _trace_f(tables, names)
    return s._replace(
        perish_song=s.perish_song.at[1].set(1),
        perish_song_turns=s.perish_song_turns.at[1].set(turns),
    )


def test_perish_song_ko_switches_without_a_draw(tables, names):
    """AI_PerishSongKO (trainer_ai.c:3263-3272, step 3): the AI's active with the effect at
    perishSongTurns 0 switches before every draw, to BattleAI_PostKOSwitchIn's pick
    (TrainerAI_PickCommand, :4001-4026), whatever the stop (E1 ruling G-64)."""
    s = _perishing(tables, names)
    assert switch_sites(tables, s).tolist() == [K.NONE] * SWITCH_SITES
    pick = int(replacement.post_ko_switch_in(tables, s))
    assert pick in (0, 1)
    assert _actions(tables, s) == [4 + pick] * (SWITCH_SITES + 1)
    assert switching_refusals(tables, s) == []
    # The decision is certain: the enumeration has one outcome weight.
    outs, probs = enumerate_ai_outcomes(tables, s)
    act = np.asarray(jax.vmap(lambda st: switch_action(tables, s, st))(outs.switch_stop))
    assert set(act.tolist()) == {4 + pick}
    assert abs(float(np.asarray(probs, np.float64).sum()) - 1) < 1e-5


def test_perish_song_turns_above_zero_keep_the_usual_layout(tables, names):
    usual = switch_sites(tables, _trace_f(tables, names)).tolist()
    for turns in (1, 2, 3):
        s = _perishing(tables, names, turns)
        assert switch_sites(tables, s).tolist() == usual
        assert _actions(tables, s) == _actions(tables, _trace_f(tables, names))
    # Only the AI's own battler counts: the player's Perish Song does not switch it.
    s = _trace_f(tables, names)
    mine = s._replace(perish_song=s.perish_song.at[0].set(1))
    assert _actions(tables, mine) == _actions(tables, s)


def test_perish_song_ko_follows_the_gates(tables, names):
    s = _perishing(tables, names)
    # Step 1 comes first: a trapped (Mean Look) or ingrained AI battler never switches.
    for gated in (
        s._replace(mean_look=s.mean_look.at[1].set(1)),
        s._replace(ingrain=s.ingrain.at[1].set(1)),
        s._replace(trapped=s.trapped.at[1].set(2)),
    ):
        assert _actions(tables, gated) == [NO_SWITCH] * (SWITCH_SITES + 1)
    # Step 2 (no living bench): nobody to switch to.
    alone = s._replace(hp=s.hp.at[1, :2].set(0))
    assert _actions(tables, alone) == [NO_SWITCH] * (SWITCH_SITES + 1)


def test_perish_song_ko_pick_is_post_ko_switch_in_not_the_first_slot(tables, names):
    """Geodude (slot 0) has no move that is super effective against Starly; Onix (slot 1)
    does, so BattleAI_PostKOSwitchIn's stage 1 picks slot 1 over the first living slot 0."""
    s = _perishing(tables, names)
    tackle = names.move("Tackle")
    s = s._replace(moves=s.moves.at[1, 0].set(jnp.asarray([tackle, 0, 0, 0], s.moves.dtype)))
    assert int(replacement.post_ko_switch_in(tables, s)) == 1
    assert _actions(tables, s) == [5] * (SWITCH_SITES + 1)


def test_the_players_magnet_rise_removes_the_ground_keep_draw(tables, names):
    """A Ground move that is super effective against the player leaves ``AI_HasSuperEffectiveMove``'s
    keep draw (:3588); ApplyTypeChart gives it MOVE_STATUS_MAGNET_RISE instead (no SE bit, G-66),
    unless the player is ingrained or holds an Iron Ball."""
    s = roark_state(tables, names)
    fire = names.type("Fire")
    s = s._replace(
        types=s.types.at[0, 0].set(jnp.asarray([fire, fire], jnp.int8)),
        moves=s.moves.at[1, int(s.active[1]), 1].set(names.move("Earthquake")),
    )
    assert K.KEEP in switch_sites(tables, s).tolist()
    floating = s._replace(magnet_rise_turns=s.magnet_rise_turns.at[0].set(3))
    assert K.KEEP not in switch_sites(tables, floating).tolist()
    rooted = floating._replace(ingrain=floating.ingrain.at[0].set(1))
    assert K.KEEP in switch_sites(tables, rooted).tolist()
    # The bit alone (Gravity zeroed the turns) does not: ApplyTypeChart reads the turns.
    bit = s._replace(magnet_rise=s.magnet_rise.at[0].set(1))
    assert K.KEEP in switch_sites(tables, bit).tolist()


@pytest.mark.fullset  # F-O1: asserts the refusal of Weather Ball outside the typed weathers
def test_party_variable_types_follow_move_calc_variable_type(tables, names):
    """B13-35: Move_CalcVariableType (battle_lib.c:8097-8203) of a bench member: Natural Gift
    the naturalGiftType of its party item (:8102-8104; a non-berry's is no type id,
    typeless), Judgment its plate's type (:8106-8160), Weather Ball the weather's type
    (:8177-8195); the voluntary-switch rule no longer refuses them, but refuses Weather Ball
    where its case assigns no type."""
    from nuzline.ai.replacement import post_ko_switch_in
    from nuzline.ai.switching import party_move_types
    from nuzline.engine.damage import AI_TYPELESS, judgment_type
    from nuzline.engine.ids import Type, Weather

    s = _trace_f(tables, names)
    gift, judge, ball = (names.move(m) for m in ("Natural Gift", "Judgment", "Weather Ball"))
    berry, plate, rock = (names.item(i) for i in ("Cheri Berry", "Flame Plate", "Hard Stone"))
    s = s._replace(
        moves=s.moves.at[1, 0, 3].set(gift).at[1, 1, 3].set(judge).at[1, 0, 2].set(ball),
        item=s.item.at[1, 0].set(berry).at[1, 1].set(plate),
    )
    types = np.asarray(party_move_types(tables, s))
    assert types[0, 3] == int(tables.item_gift_type[berry]) != Type.NORMAL
    assert types[1, 3] == int(judgment_type(tables, plate, Type.NORMAL)) == Type.FIRE
    assert types[0, 2] == int(tables.move_type[ball])  # no weather: refused below
    refusals = switching_refusals(tables, s)
    assert any("Weather Ball" in r for r in refusals), refusals
    assert not any("Natural Gift" in r or "Judgment" in r for r in refusals), refusals
    rock_gift = s._replace(item=s.item.at[1, 0].set(rock))
    assert int(party_move_types(tables, rock_gift)[0, 3]) == AI_TYPELESS
    rain = s._replace(
        weather=jnp.asarray(Weather.RAIN, s.weather.dtype),
        weather_timer=jnp.asarray(5, s.weather_timer.dtype),
    )
    assert int(party_move_types(tables, rain)[0, 2]) == Type.WATER
    assert switching_refusals(tables, rain) == []
    # The post-KO replacement types its candidates the same way (battle_lib.c:7998, :8051).
    fainted = s._replace(hp=s.hp.at[1, CRANIDOS].set(0))
    with pytest.raises(Exception, match="Weather Ball"):
        post_ko_switch_in(tables, fainted)
    post_ko_switch_in(tables, rain._replace(hp=rain.hp.at[1, CRANIDOS].set(0)))


def _flash_fire_bench(tables, names, *, player="Monferno"):
    """Roark's Geodude (active) with a Flash Fire holder on the bench (Vulpix in Onix's slot 1) after the
    player's Ember (moveHit, Fire with power)."""
    s = roark_real_state(tables, names, player_species=player)
    s = s._replace(species=s.species.at[1, ONIX].set(names.species("Vulpix")))  # Flash Fire
    return s._replace(move_hit=s.move_hit.at[1].set(names.move("Ember")))


def test_flash_fire_bench_draws_one_absorb_site(tables, names):
    """AI_HasAbsorbAbilityInParty (trainer_ai.c:3640-3713), Flash Fire's case: with no super-
    effective move of A (no ``RandNext() % 3`` gate, :3651) a bench Flash Fire holder after a
    Fire moveHit with power draws ``RandNext() & 1`` (:3706), which switches to it (:3707-3708)."""
    s = _flash_fire_bench(tables, names)
    assert switching_refusals(tables, s) == []
    kinds = switch_sites(tables, s).tolist()
    assert K.ABSORB in kinds
    stop = kinds.index(K.ABSORB)
    assert int(switch_action(tables, s, jnp.int8(stop))) == 4 + ONIX
    # Its own Flash Fire holds the switch (:3678): no site.
    own = s._replace(species=s.species.at[1, 0].set(names.species("Vulpix")))
    assert K.ABSORB not in switch_sites(tables, own).tolist()
    # A Water moveHit is another case (Water Absorb's), a status moveHit has no power (:3657).
    water = s._replace(move_hit=s.move_hit.at[1].set(names.move("Water Gun")))
    assert K.ABSORB not in switch_sites(tables, water).tolist()


def test_flash_fire_bench_behind_a_super_effective_move_is_refused(tables, names):
    """With a super-effective move of A the function reads ``RandNext() % 3`` (:3651) first: one
    draw shared by the bench draws, which independent sites cannot express."""
    s = _flash_fire_bench(tables, names, player="Starly")  # Rock Throw is super effective
    refusals = switching_refusals(tables, s)
    assert any("AI_HasAbsorbAbilityInParty" in r for r in refusals), refusals
    with pytest.raises(UnsupportedSwitching, match="AI_HasAbsorbAbilityInParty"):
        switch_sites(tables, s)
