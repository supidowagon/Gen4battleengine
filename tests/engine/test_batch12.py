"""E2 batch 12 (rulings B12-n, docs/kaizo/e2-mechanics-ledger.md): the moves Metronome can call
in Kaizo that coverage still refused, by family; hand cases each checked against the reference
engine (``assert_matches_reference``)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from nuzline.engine.damage import damage_for_roll
from nuzline.engine.enumerate import enumerate_turn, outcome_laws
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.outcome import ACUPRESSURE_DRAWS, crit_odds, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import assert_matches_reference
from tests.engine.test_pivot import _mon, _resume

O = empty_outcome()


def _state(tables, names, mine, theirs, level=40):
    n = names
    return make_state(
        tables, [_mon(n, mine[0], mine[1:], level)], [_mon(n, theirs[0], theirs[1:], level)]
    )


def _stages(s, side: int, **by_stat):
    stages = s.stages
    for stat, value in by_stat.items():
        stages = stages.at[side, Stat[stat]].set(value)
    return s._replace(stages=stages)


def _st(out, side: int, stat: str) -> int:
    return int(out.stages[side, Stat[stat]])


# --- B12-1: Calm Mind, Cosmic Power / Defend Order (DRAGON_DANCE with two stats) -----------


def test_calm_mind_and_cosmic_power_raise_their_two_stats(tables, names):
    """subscript_user_spatk_and_spdef_up_1_stage.s / _def_and_spdef_: the first stat then the
    second +1; one at +6 is skipped on its own."""
    s = _state(tables, names, ("Alakazam", "Calm Mind"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert (_st(out, 0, "SPA"), _st(out, 0, "SPD"), _st(out, 0, "ATK")) == (1, 1, 0)
    s = _state(tables, names, ("Clefairy", "Cosmic Power"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, _stages(s, 0, DEF=6), (0, 0), O)
    assert (_st(out, 0, "DEF"), _st(out, 0, "SPD"), _st(out, 0, "SPA")) == (6, 1, 0)


def test_cosmic_power_at_both_maxed_does_nothing(tables, names):
    """:5-6, :21-29: both at +6 is MOVE_STATUS_NO_MORE_WORK."""
    s = _state(tables, names, ("Clefairy", "Cosmic Power"), ("Bidoof", "Tackle"))
    s = _stages(s, 0, DEF=6, SPD=6)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert (_st(out, 0, "DEF"), _st(out, 0, "SPD")) == (6, 6)


# --- B12-2: Flatter -------------------------------------------------------------------------


def test_flatter_raises_sp_atk_by_one_and_confuses(tables, names):
    """subscript_flatter.s: Sp. Atk +1 (not Swagger's Attack +2), then subscript_confuse.s."""
    s = _state(tables, names, ("Pikachu", "Flatter"), ("Bidoof", "Tackle"))
    o = O._replace(duration=jnp.asarray([3, 2], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert (_st(out, 1, "SPA"), _st(out, 1, "ATK")) == (1, 0)
    assert int(out.confusion[1]) == 2  # 3 turns, one spent by the foe's own move after it
    swagger = _state(tables, names, ("Pikachu", "Swagger"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, swagger, (0, 0), o)
    assert (_st(out, 1, "ATK"), _st(out, 1, "SPA")) == (2, 0)


def test_flatter_fails_at_plus_six_sp_atk_on_a_confused_foe(tables, names):
    """subscript_flatter.s:6-8: Sp. Atk +6 and already confused fails the move; Sp. Atk +6
    alone still confuses."""
    s = _stages(_state(tables, names, ("Pikachu", "Flatter"), ("Bidoof", "Tackle")), 1, SPA=6)
    o = O._replace(duration=jnp.asarray([4, 2], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.confusion[1]) == 3  # 4 turns, one spent by the foe's move
    confused = s._replace(confusion=s.confusion.at[1].set(2))
    out = assert_matches_reference(tables, confused, (0, 0), o)
    assert _st(out, 1, "SPA") == 6


# --- B12-3: Tickle --------------------------------------------------------------------------


def test_tickle_lowers_attack_and_defense(tables, names):
    """subscript_target_atk_and_def_down_1_stage.s: Attack -1 then Defense -1."""
    s = _state(tables, names, ("Pikachu", "Tickle"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert (_st(out, 1, "ATK"), _st(out, 1, "DEF")) == (-1, -1)


def test_tickle_meets_hyper_cutter_on_attack_only(tables, names):
    """Each drop is its own BtlCmd_ChangeStatStage: Hyper Cutter blocks the Attack one
    (battle_script.c:2898), the Defense one goes through; Clear Body blocks both."""
    s = _state(tables, names, ("Pikachu", "Tickle"), ("Bidoof", "Tackle"))
    cutter = s._replace(ability=s.ability.at[1, 0].set(AbilityId.HYPER_CUTTER))
    out = assert_matches_reference(tables, cutter, (0, 0), O)
    assert (_st(out, 1, "ATK"), _st(out, 1, "DEF")) == (0, -1)
    body = s._replace(ability=s.ability.at[1, 0].set(AbilityId.CLEAR_BODY))
    out = assert_matches_reference(tables, body, (0, 0), O)
    assert (_st(out, 1, "ATK"), _st(out, 1, "DEF")) == (0, 0)


# --- B12-4: Focus Energy --------------------------------------------------------------------


def test_focus_energy_sets_the_flag_and_two_crit_stages(tables, names):
    """subscript_focus_energy.s:6; BattleSystem_CalcCriticalMulti's + 2 (battle_lib.c:7123):
    stage 2 is 1 in 4 (sCriticalStageRates)."""
    s = _state(tables, names, ("Machop", "Focus Energy", "Tackle"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.focus_energy[0]) == 1 and int(out.focus_energy[1]) == 0
    odds = crit_odds(tables, out, jnp.asarray([1, 0], jnp.int32))
    assert int(odds[0]) == 4 and int(odds[1]) == 16


def test_baton_pass_keeps_focus_energy_and_power_trick(tables, names):
    """VOLATILE_CONDITION_BATON_PASSED has Focus Energy (condition.h:72), MOVE_EFFECT_BATON_PASSED
    Power Trick (moves.h:47); a plain switch clears both."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Ninjask", ["Baton Pass"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Growl"], 20)],
    )
    s = s._replace(
        focus_energy=s.focus_energy.at[0].set(1), power_trick=s.power_trick.at[0].set(1)
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    after = _resume(tables, paused, 1)
    assert int(after.focus_energy[0]) == 1 and int(after.power_trick[0]) == 1
    switched = assert_matches_reference(tables, s, (5, 0), O)  # MOVE_ACTIONS + 1: a switch
    assert int(switched.focus_energy[0]) == 0 and int(switched.power_trick[0]) == 0


# --- B12-5, B12-6, B12-7: Haze, Psych Up, Power Swap / Guard Swap ----------------------------


def test_haze_resets_every_stage_and_focus_energy(tables, names):
    """BtlCmd_ResetAllStatChanges (battle_script.c:3534-3549): every battler."""
    s = _state(tables, names, ("Koffing", "Haze"), ("Bidoof", "Tackle"))
    s = _stages(_stages(s, 0, ATK=-2, EVA=3), 1, SPE=4, DEF=-1)
    s = s._replace(focus_energy=jnp.asarray([1, 1], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert not np.asarray(out.stages)[:, 1:].any()
    assert not np.asarray(out.focus_energy).any()


def test_psych_up_copies_the_stages_and_ors_focus_energy(tables, names):
    """BtlCmd_CopyStatStages (battle_script.c:6042-6053)."""
    s = _state(tables, names, ("Pikachu", "Psych Up"), ("Bidoof", "Tackle"))
    s = _stages(_stages(s, 0, SPE=3), 1, ATK=2, EVA=-1)
    s = s._replace(focus_energy=s.focus_energy.at[1].set(1))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert (_st(out, 0, "ATK"), _st(out, 0, "EVA"), _st(out, 0, "SPE")) == (2, -1, 0)
    assert int(out.focus_energy[0]) == 1


def test_power_swap_and_guard_swap_exchange_their_stages(tables, names):
    """subscript_exchange_atk_and_spatk_stages.s / _def_and_spdef_stages.s:5-12."""
    s = _state(tables, names, ("Pikachu", "Power Swap"), ("Bidoof", "Tackle"))
    s = _stages(_stages(s, 0, ATK=-2, SPA=1, DEF=3), 1, ATK=4, SPA=-3, DEF=-1)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert (_st(out, 0, "ATK"), _st(out, 0, "SPA"), _st(out, 0, "DEF")) == (4, -3, 3)
    assert (_st(out, 1, "ATK"), _st(out, 1, "SPA"), _st(out, 1, "DEF")) == (-2, 1, -1)
    g = _state(tables, names, ("Pikachu", "Guard Swap"), ("Bidoof", "Tackle"))
    g = _stages(_stages(g, 0, DEF=-2, SPD=1, ATK=3), 1, DEF=4, SPD=-3)
    out = assert_matches_reference(tables, g, (0, 0), O)
    assert (_st(out, 0, "DEF"), _st(out, 0, "SPD"), _st(out, 0, "ATK")) == (4, -3, 3)
    assert (_st(out, 1, "DEF"), _st(out, 1, "SPD")) == (-2, 1)


# --- B12-8: Captivate -----------------------------------------------------------------------


def _gendered(s, mine: int, theirs: int):
    return s._replace(gender=s.gender.at[0, 0].set(mine).at[1, 0].set(theirs))


def test_captivate_needs_the_opposite_gender(tables, names):
    """effect_script_0265.s:6-9: genderless or the same gender ends it; else Sp. Atk -2."""
    s = _state(tables, names, ("Pikachu", "Captivate"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, _gendered(s, 0, 1), (0, 0), O)
    assert _st(out, 1, "SPA") == -2
    for mine, theirs in ((0, 0), (1, 1), (2, 1), (0, 2)):
        out = assert_matches_reference(tables, _gendered(s, mine, theirs), (0, 0), O)
        assert _st(out, 1, "SPA") == 0, (mine, theirs)


# --- B12-9: Acupressure ---------------------------------------------------------------------


def _acu(draw: int):
    return O._replace(acupressure=jnp.asarray([draw, 0], jnp.int16))


def test_acupressure_raises_the_drawn_stat_among_those_below_six(tables, names):
    """BtlCmd_BoostRandomStatBy2: stats[RandNext() % validStats] in BATTLE_STAT order (Attack,
    Defense, Speed, Sp. Atk, Sp. Def, Accuracy, Evasion), +2."""
    s = _state(tables, names, ("Pikachu", "Acupressure"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, s, (0, 0), _acu(2))
    assert _st(out, 0, "SPE") == 2 and _st(out, 0, "SPA") == 0
    out = assert_matches_reference(tables, s, (0, 0), _acu(7 + 3))  # 10 % 7 == 3: Sp. Atk
    assert _st(out, 0, "SPA") == 2
    maxed = _stages(s, 0, ATK=6, DEF=6, SPE=6)  # valid: Sp. Atk, Sp. Def, Accuracy, Evasion
    out = assert_matches_reference(tables, maxed, (0, 0), _acu(5))  # 5 % 4 == 1: Sp. Def
    assert _st(out, 0, "SPD") == 2 and _st(out, 0, "SPA") == 0


def test_acupressure_fails_behind_its_users_substitute_or_all_maxed(tables, names):
    """effect_script_0226.s:5: CheckSubstitute on the defender, the user itself in singles; the
    BoostRandomStatBy2 jump with no stat below +6."""
    s = _state(tables, names, ("Pikachu", "Acupressure"), ("Bidoof", "Tackle"))
    subbed = s._replace(substitute_hp=s.substitute_hp.at[0].set(20))
    out = assert_matches_reference(tables, subbed, (0, 0), _acu(0))
    assert not np.asarray(out.stages)[0].any()
    full = _stages(s, 0, ATK=6, DEF=6, SPE=6, SPA=6, SPD=6, ACC=6, EVA=6)
    out = assert_matches_reference(tables, full, (0, 0), _acu(0))
    assert np.asarray(out.stages)[0, 1:].tolist() == [6] * 7


def test_acupressure_law_counts_the_stats_below_six(tables, names):
    """The draw's law: the turn-start count of stats below +6 when nothing before the move can
    change it, else the 420 residues; the rows' weights sum to 1."""
    s = _state(tables, names, ("Pikachu", "Acupressure"), ("Bidoof", "Tackle"))
    s = _stages(s, 0, ATK=6, DEF=6, SPE=6, SPA=6)
    laws = outcome_laws(tables, s, jnp.asarray([0, 0], jnp.int32))
    assert int(laws.acupressure_count[0]) == 3
    assert bool(laws.acupressure_loose[0])  # the foe acts this turn
    outcomes, probs = enumerate_turn(tables, s, jnp.asarray([0, 0], jnp.int32))
    draws = np.asarray(outcomes.acupressure)[:, 0]
    assert len(set(draws.tolist())) == ACUPRESSURE_DRAWS
    assert abs(float(np.sum(probs)) - 1.0) < 1e-9


# --- B12-10: Power Trick --------------------------------------------------------------------


def test_power_trick_toggles_and_swaps_attack_with_defense(tables, names):
    """subscript_user_swap_atk_and_def.s:5-9: the bit XORs; the user's physical hit then reads
    its Defense as Attack (damage.battle_stats)."""
    s = _state(tables, names, ("Shuckle", "Power Trick", "Tackle"), ("Bidoof", "Tackle"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.power_trick[0]) == 1
    tackle = names.move("Tackle")
    plain = int(damage_for_roll(tables, s, 0, tackle, False, 15))
    tricked = int(damage_for_roll(tables, out, 0, tackle, False, 15))
    assert tricked > plain  # Shuckle's Defense is far above its Attack
    hit = assert_matches_reference(tables, out, (1, 0), O)
    assert int(out.hp[1, 0]) - int(hit.hp[1, 0]) == tricked
    back = assert_matches_reference(tables, out, (0, 0), O)
    assert int(back.power_trick[0]) == 0


# --- B12-12: Rest -------------------------------------------------------------------------


def _hurt(s, side: int, hp: int):
    return s._replace(hp=s.hp.at[side, 0].set(hp))


def test_rest_sleeps_three_turns_and_heals_fully(tables, names):
    """subscript_rest.s: a burned user loses the burn, sleeps with a count of 3 (:32) and
    regains its max HP (:14, :36)."""
    from nuzline.engine.ids import Status

    s = _hurt(_state(tables, names, ("Snorlax", "Rest"), ("Bidoof", "Growl")), 0, 20)
    s = s.with_status(0, 0, Status.BURN)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[0, 0]) == Status.SLEEP and int(out.status_counter[0, 0]) == 3
    assert int(out.hp[0, 0]) == int(s.max_hp[0, 0])  # no burn chip: it sleeps instead


def test_rest_fails_at_full_hp_and_with_insomnia(tables, names):
    """subscript_rest.s:7-8 (Insomnia, CheckAbility), :14-15 (full HP): MOVE_STATUS_NO_MORE_WORK."""
    from nuzline.engine.ids import Status

    s = _state(tables, names, ("Snorlax", "Rest"), ("Bidoof", "Growl"))
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE
    hurt = _hurt(s, 0, 20)
    sleepless = hurt._replace(ability=hurt.ability.at[0, 0].set(AbilityId.INSOMNIA))
    out = assert_matches_reference(tables, sleepless, (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.hp[0, 0]) == 20


# --- B12-13: Morning Sun / Synthesis / Moonlight ------------------------------------------


def test_weather_heal_by_weather(tables, names):
    """BtlCmd_WeatherHPRecovery: maxHP / 2 clear, Divide(maxHP * 20, 30) in sun, Divide(maxHP, 4)
    in rain."""
    from nuzline.engine.ids import Weather

    s = _hurt(_state(tables, names, ("Bellossom", "Synthesis"), ("Bidoof", "Growl")), 0, 1)
    m = int(s.max_hp[0, 0])
    cases = ((Weather.NONE, m // 2), (Weather.SUN, m * 20 // 30), (Weather.RAIN, m // 4))
    for weather, gained in cases:
        w = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
        out = assert_matches_reference(tables, w, (0, 0), O)
        assert int(out.hp[0, 0]) == 1 + gained, weather


# --- B12-14: Wish -------------------------------------------------------------------------


def test_wish_heals_half_of_the_battler_on_the_field_after_two_turns(tables, names):
    """BtlCmd_TryWish: wishTurns 2; the field pass counts it down and heals Divide(maxHP, 2) of
    the battler then on the field (bcp.c:1085-1111); a second Wish fails while it counts."""
    s = _hurt(_state(tables, names, ("Togekiss", "Wish", "Tackle"), ("Bidoof", "Growl")), 0, 5)
    one = assert_matches_reference(tables, s, (0, 0), O)
    assert int(one.wish_timer[0]) == 1 and int(one.hp[0, 0]) == 5
    two = assert_matches_reference(tables, one, (1, 0), O)
    assert int(two.wish_timer[0]) == 0
    assert int(two.hp[0, 0]) == 5 + max(int(s.max_hp[0, 0]) // 2, 1)
    again = assert_matches_reference(tables, one, (0, 0), O)
    assert int(again.wish_timer[0]) == 0  # the failed Wish set nothing


# --- B12-15: Heal Bell / Aromatherapy -----------------------------------------------------


def test_heal_bell_cures_the_whole_party(tables, names):
    """BtlCmd_TryPartyStatusRefresh and BtlIOCmd_RefreshPartyStatus: the user and every party
    member lose their status; the foe's side keeps its own."""
    from nuzline.engine.ids import Status

    n = names
    s = make_state(
        tables,
        [_mon(n, "Chansey", ["Heal Bell"], 40), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bidoof", ["Growl"], 30)],
    )
    s = s.with_status(0, 0, Status.PARALYSIS).with_status(0, 1, Status.TOXIC)
    s = s.with_status(1, 0, Status.BURN)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.status[0, 1]) == Status.NONE
    assert int(out.status[1, 0]) == Status.BURN


# --- B12-16: Refresh ----------------------------------------------------------------------


def test_refresh_cures_poison_burn_paralysis_only(tables, names):
    """subscript_recover_psn_prz_brn.s:5-7 (MON_CONDITION_FACADE_BOOST)."""
    from nuzline.engine.ids import Status

    s = _state(tables, names, ("Pikachu", "Refresh"), ("Bidoof", "Growl"))
    out = assert_matches_reference(tables, s.with_status(0, 0, Status.BURN), (0, 0), O)
    assert int(out.status[0, 0]) == Status.NONE


# --- B12-17: Psycho Shift -----------------------------------------------------------------


def test_psycho_shift_passes_the_status_and_cures_the_user(tables, names):
    """subscript_give_target_own_status.s: the user's burn through subscript_burn.s, then the
    user cured (:28-30); a Fire defender takes no burn and the user keeps it."""
    from nuzline.engine.ids import Status, Type

    s = _state(tables, names, ("Alakazam", "Psycho Shift"), ("Bidoof", "Growl"))
    s = s.with_status(0, 0, Status.BURN)
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.status[1, 0]) == Status.BURN and int(out.status[0, 0]) == Status.NONE
    fire = s._replace(types=s.types.at[1, 0].set(jnp.asarray([Type.FIRE, Type.FIRE], jnp.int8)))
    out = assert_matches_reference(tables, fire, (0, 0), O)
    assert int(out.status[1, 0]) == Status.NONE and int(out.status[0, 0]) == Status.BURN


def test_psycho_shift_gives_sleep_with_the_drawn_count(tables, names):
    """A sleeping user's Psycho Shift (through Sleep Talk) gives sleep for Random 3, 2 turns
    (subscript_fall_asleep.s, outcome.duration of the user's side)."""
    from nuzline.engine.ids import Status

    s = _state(tables, names, ("Alakazam", "Sleep Talk", "Psycho Shift"), ("Bidoof", "Growl"))
    s = s.with_status(0, 0, Status.SLEEP, counter=3)
    o = O._replace(duration=jnp.asarray([4, 2], jnp.int8), call=jnp.asarray([1, 0], jnp.int16))
    out = assert_matches_reference(tables, s, (0, 0), o)
    assert int(out.status[1, 0]) == Status.SLEEP
    assert int(out.status[0, 0]) == Status.NONE
