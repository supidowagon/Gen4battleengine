"""The opponent's replacement after a faint: ``BattleAI_PostKOSwitchIn``
(battle_lib.c:7923; docs/ai/roark.md 11.2) and its fallback in
``Task_TrainerShowPartyMenu`` (battle_display.c:4548-4556)."""

import jax
import jax.numpy as jnp
import pytest

from nuzline.ai.policy import UnsupportedAi
from nuzline.ai.replacement import REFUSED_SLOT, post_ko_switch_in
from nuzline.engine.actions import legal_replacements
from nuzline.engine.ids import Stat
from tests.engine.helpers import roark_real_state, roark_state

GEODUDE, ONIX, CRANIDOS = 0, 1, 2


def _types(s, names, side, slot, t1, t2):
    types = jnp.array([names.type(t1), names.type(t2)], jnp.int8)
    return s._replace(types=s.types.at[side, slot].set(types))


def test_stage_2_decides_against_a_pure_fighting_type(tables, names):
    # roark.md section 14: neither Onix nor Cranidos has a move super effective against
    # Machop, so stage 2 scores their moves as used by the fainted Geodude: Headbutt
    # (neutral) beats Rock Throw (resisted). The first legal slot would be Onix.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    assert int(post_ko_switch_in(tables, s)) == CRANIDOS


def test_stage_1_skips_the_best_matchup_without_a_super_effective_move(tables, names):
    # Against Starly (Normal/Flying), Cranidos (Rock/Rock) scores 80 + 80 and Onix
    # (Rock/Ground) 80 + 0, but Cranidos's Headbutt, Pursuit and Leer are not super
    # effective, so it is discarded and Onix (Rock Throw) is sent.
    s = roark_real_state(tables, names, player_species="Starly").replace_hp(1, GEODUDE, 0)
    assert int(post_ko_switch_in(tables, s)) == ONIX


def test_stage_1_takes_the_highest_matchup_with_a_super_effective_move(tables, names):
    # The fixture moves give everyone Rock Throw: Cranidos's 160 beats Onix's 80.
    s = roark_state(tables, names, player_species="Starly").replace_hp(1, GEODUDE, 0)
    assert int(post_ko_switch_in(tables, s)) == CRANIDOS


def test_stage_1_score_wraps_as_u8(tables, names):
    # Against Fire/Flying, Cranidos scores 160 + 160 = 320, which the decomp's u8 keeps
    # as 64; Onix scores 160 + 0. Both know Rock Throw, so Onix wins (the overflow quirk).
    s = roark_state(tables, names, player_species="Starly").replace_hp(1, GEODUDE, 0)
    s = _types(s, names, 0, 0, "Fire", "Flying")
    assert int(post_ko_switch_in(tables, s)) == ONIX


def test_stage_1_ties_go_to_the_lower_slot(tables, names):
    # Cranidos fainted; Geodude and Onix are both Rock/Ground with Rock Throw: Geodude.
    s = roark_state(tables, names, player_species="Starly")
    s = s._replace(active=s.active.at[1].set(CRANIDOS)).replace_hp(1, CRANIDOS, 0)
    assert int(post_ko_switch_in(tables, s)) == GEODUDE


def _geodude_atk(s, atk):
    return s._replace(stats=s.stats.at[1, GEODUDE, Stat.ATK - 1].set(jnp.int16(atk)))


def test_stage_2_score_wraps_as_u8(tables, names):
    # Against Machop, stage 2 uses the fainted Geodude's Attack. At Atk 640 Headbutt's
    # CalcMoveDamage is 259, kept as 3 in the u8 score, while Onix's Rock Throw scores
    # 186 * 1.5 * 0.5 = 139, so the pick flips from Cranidos to Onix.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    assert int(post_ko_switch_in(tables, _geodude_atk(s, 300))) == CRANIDOS
    assert int(post_ko_switch_in(tables, _geodude_atk(s, 640))) == ONIX


def test_power_mul_scales_stage_2(tables, names):
    # Pursuit leaves powerMul at 20 for the rest of the turn (battle_script.c:6928, reset
    # only by BattleContext_Init). At Atk 400 Headbutt scores 162 against Rock Throw's 87;
    # doubled, Headbutt's 322 wraps to 66 against Rock Throw's 172.
    s = _geodude_atk(
        roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0), 400
    )
    assert int(post_ko_switch_in(tables, s, 10)) == CRANIDOS
    assert int(post_ko_switch_in(tables, s, 20)) == ONIX


def test_fallback_is_the_first_living_non_active_slot(tables, names):
    # A Ghost-type defender and Tackle-only benches: no stage 1 pick (Tackle is not super
    # effective) and every stage 2 score is 0 (Normal does not affect Ghost).
    s = roark_state(tables, names, player_species="Starly").replace_hp(1, GEODUDE, 0)
    s = _types(s, names, 0, 0, "Ghost", "Ghost")
    tackle = names.move("Tackle")
    s = s._replace(moves=s.moves.at[1, 1:3].set(jnp.array([tackle, 0, 0, 0], jnp.int16)))
    assert int(post_ko_switch_in(tables, s)) == ONIX


def test_pick_is_always_a_legal_replacement(tables, names):
    for species in ("Starly", "Machop", "Shinx", "Prinplup", "Grotle"):
        s = roark_real_state(tables, names, player_species=species).replace_hp(1, GEODUDE, 0)
        assert bool(legal_replacements(s, 1)[int(post_ko_switch_in(tables, s))])


def test_traced_matches_eager(tables, names):
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    traced = jax.jit(lambda st: post_ko_switch_in(tables, st))(s)
    assert int(traced) == int(post_ko_switch_in(tables, s))


@pytest.mark.fullset
def test_skipped_first_move_reading_a_stale_score_is_refused(tables, names):
    # Stage 2 compares the stale stage-1 score when the first candidate's slot 0 is skipped
    # (power 1: a variable-power move such as Low Kick); that read is refused.
    s = roark_real_state(tables, names, player_species="Machop").replace_hp(1, GEODUDE, 0)
    low_kick = names.move("Low Kick")
    s = s._replace(moves=s.moves.at[1, ONIX, 0].set(jnp.int16(low_kick)))
    assert int(tables.move_power[low_kick]) == 1
    with pytest.raises(UnsupportedAi, match="stage 2"):
        post_ko_switch_in(tables, s)
    assert int(jax.jit(lambda st: post_ko_switch_in(tables, st))(s)) == REFUSED_SLOT
