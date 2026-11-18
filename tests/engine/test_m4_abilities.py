"""Insomnia, Cloud Nine, Compoundeyes and Shed Skin (milestone 4a, task 9).

* Insomnia: ``subscript_fall_asleep.s`` (:_055, :_202, :_221) blocks a sleep move through
  ``CheckIgnorableAbility`` (Mold Breaker pierces it) and tags the ability;
  ``BattleSystem_RecoverStatusByAbility`` (battle_lib.c:4378) wakes a sleeper at each move
  end (battle_controller_player.c:3965-3967) and on a switch-in (battle_lib.c:4135).
* Cloud Nine: every decomp branch on it is a weather branch, so it has no engine
  behaviour while the weather is none.
* Compoundeyes: hit rate x130/100 after the stage scaling (battle_controller_player.c:2922).
* Shed Skin: the first end-of-turn check (``MON_COND_CHECK_STATE_ABILITY``, :1355) cures a
  status of a holder with HP left on ``RandNext() % 10 < 3`` (battle_lib.c:3567-3590).
"""

import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.ids import AbilityId, Status, Weather
from nuzline.engine.outcome import empty_outcome, hit_threshold
from nuzline.engine.step import _turn_end_ability, step
from tests.engine.helpers import A, assert_matches_reference, roark_state
from tests.strategies import any_outcome, legal_action_pair, supported_state

DECOMP_BATTLE = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum" / "src" / "battle"
SLEEP = Status.SLEEP
POISON = Status.POISON
SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot

_jit_step = jax.jit(step)


def shed(p=0, o=0):
    return empty_outcome()._replace(shed_skin=jnp.array([p, o], jnp.int8))


@pytest.fixture
def spinarak(tables, names):
    """The player is a Spinarak with Insomnia against a Geodude that has Hypnosis."""
    return roark_state(
        tables,
        names,
        player_species="Spinarak",
        player_ability="Insomnia",
        player_moves=("Poison Sting",),
        opp_species="Geodude",
        opp_moves=("Hypnosis",),
    )


# --- Insomnia ------------------------------------------------------------------------


@pytest.mark.slow
def test_insomnia_blocks_sleep(tables, names, spinarak):
    acts = A("Poison Sting", "Hypnosis", spinarak, names)
    out = step(tables, spinarak, acts, empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) != 0  # "stayed awake because of its Insomnia"
    # Without Insomnia the same move puts the Pokemon to sleep.
    bare = spinarak._replace(ability=spinarak.ability.at[0, 0].set(AbilityId.NONE))
    slept = step(tables, bare, acts, empty_outcome())
    assert int(slept.status[0, 0]) == SLEEP and int(slept.ai_ability_known[0]) == 0


@pytest.mark.slow
def test_mold_breaker_pierces_insomnia(tables, names):
    """The sleep lands (it unlocks a locked mover, subscript_fall_asleep.s:_176), and the
    holder wakes at that move's end (``RecoverStatusByAbility`` is not ignorable)."""

    def run(opp_species):
        s = roark_state(
            tables,
            names,
            player_species="Spinarak",
            player_ability="Insomnia",
            player_moves=("Poison Sting",),
            opp_species=opp_species,
            opp_moves=("Hypnosis",),
        )
        sting = names.move("Poison Sting")
        s = s._replace(locked_move=s.locked_move.at[0].set(sting))
        return step(tables, s, A("Poison Sting", "Hypnosis", s, names), empty_outcome())

    blocked, pierced = run("Geodude"), run("Cranidos")
    assert int(blocked.locked_move[0]) != 0 and int(pierced.locked_move[0]) == 0
    for out in (blocked, pierced):
        assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0


@pytest.mark.slow
def test_insomnia_cures_on_entry_and_move_end(tables, names, spinarak):
    # A sleeping Insomnia Pokemon switched in wakes (SWITCH_IN_CHECK_STATE_FORBIDDEN_STATUS).
    bench = spinarak._replace(ability=spinarak.ability.at[0, 1].set(AbilityId.INSOMNIA))
    bench = bench.with_status(0, 1, SLEEP, counter=3)
    out = step(tables, bench, A(SWITCH_TO_1, "Tackle", bench, names), empty_outcome())
    assert int(out.status[0, 1]) == Status.NONE and int(out.status_counter[0, 1]) == 0
    assert int(out.ai_ability_known[1]) != 0
    # An asleep Insomnia active wakes at the end of its own action (MoveEnd).
    asleep = spinarak.with_status(0, 0, SLEEP, counter=3)
    leer = A("Poison Sting", "Tackle", asleep, names)
    out = step(tables, asleep, leer, empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.status_counter[0, 0]) == 0
    assert int(out.ai_ability_known[0]) != 0
    # Other abilities keep the sleep.
    bare = asleep._replace(ability=asleep.ability.at[0, 0].set(AbilityId.NONE))
    assert int(step(tables, bare, leer, empty_outcome()).status[0, 0]) == SLEEP


# --- Cloud Nine ----------------------------------------------------------------------


@pytest.mark.slow
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_cloud_nine_is_inert(tables, names, data):
    s = data.draw(supported_state(tables, names))
    s = s._replace(weather=jnp.full_like(s.weather, Weather.NONE))
    acts = data.draw(legal_action_pair(tables, s))
    outcome = data.draw(any_outcome())
    side = data.draw(st.integers(0, 1))
    cloudy = s._replace(ability=s.ability.at[side, s.active[side]].set(AbilityId.CLOUD_NINE))
    plain = s._replace(ability=s.ability.at[side, s.active[side]].set(AbilityId.NONE))
    a, b = _jit_step(tables, cloudy, acts, outcome), _jit_step(tables, plain, acts, outcome)
    for name, x, y in zip(a._fields, a, b, strict=True):
        if name != "ability":
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


# Every site in the decomp's src/battle that reads Cloud Nine (the ability, NO_CLOUD_NINE or
# NO_WEATHER, include/battle/common.h:63-66), by file and line: each is a weather condition
# or the move animation's field conditions.
CLOUD_NINE_SITES = {
    ("battle_controller.c", 2072),  # the move animation's field conditions
    ("battle_controller_player.c", 2320),  # skip charge turn in sun
    ("battle_controller_player.c", 2915),  # Thunder in sun
    ("battle_controller_player.c", 2926),  # sandstorm accuracy
    ("battle_controller_player.c", 3024),  # Thunder in rain
    ("battle_lib.c", 1228),  # Swift Swim / Chlorophyll speed
    ("battle_lib.c", 6317),  # Castform form
    ("battle_lib.c", 6368),  # Cherrim form
    ("battle_lib.c", 6921),  # Solar Power
    ("battle_lib.c", 7046),  # rain and sun move power
    ("battle_lib.c", 7895),  # weather ability triggers
    ("battle_lib.c", 8177),  # weather ability triggers
    ("battle_script.c", 5573),  # weather damage
    ("battle_script.c", 5580),  # MOVE_EFFECT_NO_WEATHER_DAMAGE (a move-effect flag)
    ("battle_script.c", 5587),
    ("battle_script.c", 5600),
    ("battle_script.c", 5970),  # Moonlight family healing
    ("battle_script.c", 6847),  # Weather Ball
    ("battle_script.c", 9200),  # BtlCmd_CheckIgnoreWeather: jumps when Cloud Nine / Air Lock
    ("trainer_ai/trainer_ai.c", 3223),  # Weather Ball type
}
_WEATHER = re.compile(
    r"WEATHER|SUNNY|RAIN|SAND|HAIL|FOG|CASTFORM|CHERRIM|SUN\b|fieldConditions|CheckIgnore"
)


def test_cloud_nine_sites_are_weather_only():
    if not DECOMP_BATTLE.is_dir():
        pytest.skip("decomp not checked out")
    token = re.compile(r"ABILITY_CLOUD_NINE|NO_CLOUD_NINE|NO_WEATHER")
    found = {}
    for path in sorted(DECOMP_BATTLE.rglob("*.c")):
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if token.search(line):
                found[(path.relative_to(DECOMP_BATTLE).as_posix(), i + 1)] = lines[
                    max(0, i - 10) : i + 8
                ]
    assert set(found) == CLOUD_NINE_SITES
    for site, window in found.items():
        assert any(_WEATHER.search(line) for line in window), site


# --- Compoundeyes --------------------------------------------------------------------


def test_compoundeyes_accuracy(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Venonat",
        player_ability="Compoundeyes",
        player_moves=("Tackle",),
    )
    acts = A("Tackle", "Leer", s, names)
    assert int(hit_threshold(tables, s, acts)[0]) == 123  # 95 * 130 // 100
    assert int(hit_threshold(tables, s.with_stages(0, acc=-1), acts)[0]) == 92  # 71 * 130 // 100
    bare = s._replace(ability=s.ability.at[0, 0].set(AbilityId.NONE))
    assert int(hit_threshold(tables, bare, acts)[0]) == 95


# --- Shed Skin -----------------------------------------------------------------------


@pytest.fixture
def kricketot(tables, names):
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_ability="Shed Skin",
        player_moves=("Growl",),
    )
    return s.with_status(0, 0, POISON)


@pytest.mark.slow
def test_shed_skin_cures_before_poison(tables, names, kricketot):
    acts = A("Growl", "Leer", kricketot, names)
    out = step(tables, kricketot, acts, shed(p=1))
    assert int(out.status[0, 0]) == Status.NONE
    # no poison damage this turn: the cure is the first end-of-turn check
    assert int(out.hp[0, 0]) == int(kricketot.hp[0, 0])
    assert int(out.ai_ability_known[0]) != 0
    kept = step(tables, kricketot, acts, shed(p=0))
    assert int(kept.status[0, 0]) == POISON
    assert int(kept.hp[0, 0]) == int(kricketot.hp[0, 0]) - int(kricketot.max_hp[0, 0]) // 8
    assert int(kept.ai_ability_known[0]) == 0


def test_shed_skin_needs_a_living_holder_with_a_status(tables, names, kricketot):
    fainted = kricketot.replace_hp(0, 0, 0)
    out = _turn_end_ability(tables, fainted, 0, shed(p=1))
    assert int(out.status[0, 0]) == POISON and int(out.ai_ability_known[0]) == 0
    healthy = kricketot.with_status(0, 0, Status.NONE)
    out = _turn_end_ability(tables, healthy, 0, shed(p=1))
    assert int(out.ai_ability_known[0]) == 0  # no status: nothing to show


# --- the reference engine ------------------------------------------------------------


def test_abilities_match_reference(tables, names, spinarak, kricketot):
    hypnosis = A("Poison Sting", "Hypnosis", spinarak, names)
    assert_matches_reference(tables, spinarak, hypnosis, empty_outcome())
    asleep = spinarak.with_status(0, 0, SLEEP, counter=3)
    leer = A("Poison Sting", "Tackle", asleep, names)
    assert_matches_reference(tables, asleep, leer, empty_outcome())
    bench = spinarak._replace(ability=spinarak.ability.at[0, 1].set(AbilityId.INSOMNIA))
    bench = bench.with_status(0, 1, SLEEP, counter=3)
    assert_matches_reference(tables, bench, A(SWITCH_TO_1, "Tackle", bench, names), empty_outcome())
    cranidos = roark_state(
        tables,
        names,
        player_species="Spinarak",
        player_ability="Insomnia",
        player_moves=("Poison Sting",),
        opp_species="Cranidos",
        opp_moves=("Hypnosis",),
    )
    acts = A("Poison Sting", "Hypnosis", cranidos, names)
    assert_matches_reference(tables, cranidos, acts, empty_outcome())
    for p in (0, 1):
        acts = A("Growl", "Leer", kricketot, names)
        assert_matches_reference(tables, kricketot, acts, shed(p=p))
