"""Foresight and Odor Sleuth (``effect_script_0113.s``, ``subscript_foresight.s``):
the defender is identified, so the Ghost-immunity chart rows are skipped
(``BattleSystem_ApplyTypeChart``, battle_lib.c:2611) and its positive evasion is
ignored (``BattleControllerPlayer_CheckMoveHitAccuracy``, battle_controller_player.c:2888).
``CalcEffectiveness`` (battle_lib.c:2681) does not read the flag.

The player is a Furret with Foresight, Scratch, Mach Punch and Tackle against a Sableye
(Dark/Ghost, Keen Eye).
"""

import jax.numpy as jnp
import pytest

from nuzline.engine.damage import damage_table, effectiveness
from nuzline.engine.outcome import empty_outcome, hit_threshold
from nuzline.engine.step import step
from tests.engine.helpers import A, assert_matches_reference, roark_state

MOVES = ("Foresight", "Scratch", "Mach Punch", "Tackle")


@pytest.fixture
def state(tables, names):
    return roark_state(
        tables,
        names,
        player_species="Furret",
        player_moves=MOVES,
        opp_species="Sableye",
        opp_moves=("Tackle",),
    )


def identify(s):
    return s._replace(foresight=s.foresight.at[1].set(1))


@pytest.mark.slow
def test_foresight_sets_flag(tables, names, state):
    assert int(state.foresight[1]) == 0
    out = step(tables, state, A("Foresight", "Tackle", state, names), empty_outcome())
    assert int(out.foresight[1]) == 1 and int(out.foresight[0]) == 0
    # A second use leaves it set.
    again = step(tables, out, A("Foresight", "Tackle", out, names), empty_outcome())
    assert int(again.foresight[1]) == 1


def test_foresight_needs_a_living_target(tables, names, state):
    s = state.replace_hp(1, 0, 0)
    out = step(tables, s, A("Foresight", "Tackle", s, names), empty_outcome())
    assert int(out.foresight[1]) == 0


def test_identified_ghost_is_hit(tables, names, state):
    ident = identify(state)
    scratch, punch = names.move("Scratch"), names.move("Mach Punch")
    dark, normal = names.type("Dark"), names.type("Normal")
    # Scratch (Normal) on Dark/Ghost: immune, then the damage of a neutral hit.
    neutral = state._replace(types=state.types.at[1, 0].set(jnp.asarray([dark, dark], jnp.int8)))
    assert int(damage_table(tables, state, 0, scratch, False)[15]) == 0
    expected = int(damage_table(tables, neutral, 0, scratch, False)[15])
    assert expected > 0
    assert int(damage_table(tables, ident, 0, scratch, False)[15]) == expected
    # Mach Punch (Fighting) is immune on Ghost, x2 on Dark alone.
    assert int(damage_table(tables, state, 0, punch, False)[15]) == 0
    expected = int(damage_table(tables, neutral, 0, punch, False)[15])
    assert int(damage_table(tables, ident, 0, punch, False)[15]) == expected
    assert int(effectiveness(tables, normal, ident.types[1, 0], True)) == 100
    assert int(effectiveness(tables, normal, ident.types[1, 0], False)) == 0
    assert int(effectiveness(tables, names.type("Fighting"), ident.types[1, 0], True)) == 200


def test_identified_ignores_positive_evasion(tables, names, state):
    acts = A("Tackle", "Tackle", state, names)

    def threshold(eva, identified):
        s = state.with_stages(1, eva=eva)
        return int(hit_threshold(tables, identify(s) if identified else s, acts)[0])

    assert threshold(2, True) == 95
    assert threshold(-2, True) == 157  # 95 * 166 // 100, unchanged by Foresight
    assert threshold(-1, True) == 126  # 95 * 133 // 100
    assert threshold(2, False) == 57  # 95 * 60 // 100 (HitRateByStage[4]); controller ruling


@pytest.mark.slow
def test_foresight_matches_reference(tables, names, state):
    hits = empty_outcome()._replace(hit=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, state, A("Foresight", "Tackle", state, names), hits)
    assert int(out.foresight[1]) == 1
    ident = identify(state)
    for move in ("Scratch", "Mach Punch", "Foresight"):
        assert_matches_reference(tables, ident, A(move, "Tackle", ident, names), hits)
        assert_matches_reference(tables, state, A(move, "Tackle", state, names), hits)
    dead = state.replace_hp(1, 0, 0)
    assert_matches_reference(tables, dead, A("Foresight", "Tackle", dead, names), hits)
