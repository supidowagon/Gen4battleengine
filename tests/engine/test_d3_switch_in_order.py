"""D3-13: the switch-in states over one four-battler ``monSpeedOrder`` (``switch_in.run_states``,
``show_battle_mon4``; battle_lib.c:3649-4174, bcp.c:241-266, :3672-3682, :3973).

Each state loops the battlers in the order the TrainerMessage sort gave (bcp.c:251); only the
weather state's result (the last holder's weather stays) and Trace depend on that order. ``ties``
are the sort's tie bits, an input of ``show_battle_mon4``.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.format import DOUBLES, TWO_TRAINERS, format_of, party
from nuzline.engine.ids import AbilityId, Stat, Weather
from nuzline.engine.order import PAIRS
from nuzline.engine.step import replace_round, root_belief, sample_root
from nuzline.engine.switch_in import run_states, show_battle_mon4
from tests.engine.d2_states import four, party_of, with_ability
from tests.engine.test_d3_action_loop import play
from tests.engine.test_d3_pause import ATK, ORDER, pivoting, resume

HOLDERS = (
    (AbilityId.DRIZZLE, Weather.RAIN),
    (AbilityId.DROUGHT, Weather.SUN),
    (AbilityId.SAND_STREAM, Weather.SANDSTORM),
    (AbilityId.SNOW_WARNING, Weather.HAIL),
)
ZERO_TIES = np.zeros(6, np.int32)


@jax.jit
def _show(tables, state, ties, start):
    return show_battle_mon4(tables, state, ties, battle_start=start)


def show(tables, state, ties=ZERO_TIES, start=False):
    return _show(tables, state, jnp.asarray(ties, jnp.int32), jnp.asarray(start))


def speeds(state, values):
    """Each battler's active Pokemon has Speed ``values[b]``."""
    for b, v in enumerate(values):
        p, slot = party_of(state, b), int(state.active[b])
        state = state._replace(stats=state.stats.at[p, slot, Stat.SPE - 1].set(v))
    return state


def weather_leads(tables, names):
    """DOUBLES: battler b leads with ``HOLDERS[b]``'s ability; every lead waits (``entry_pending``)."""
    s = four(tables, names, DOUBLES)
    for b, (ability, _) in enumerate(HOLDERS):
        s = with_ability(s, b, ability)
    assert np.asarray(s.entry_pending).tolist() == [1, 1, 1, 1]
    return s


def last_weather(order):
    """lib.c:3739-3793 on a monSpeedOrder: the weather left is the last holder's."""
    return HOLDERS[order[-1]][1]


def transcribed_order(ties):
    """BattleSystem_SortMonSpeedOrder (lib.c:3364-3383) for four battlers of equal speed and no
    Quick Claw: a pair swaps when its tie bit is 1 (COMPARE_SPEED_TIE, lib.c:1405-1406)."""
    order = [0, 1, 2, 3]
    for (i, j), bit in zip(PAIRS, ties, strict=True):
        if bit:
            order[i], order[j] = order[j], order[i]
    return order


# ---------------------------------------------------------------- the weather state


@pytest.mark.parametrize("fastest_first", [(0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_the_slowest_holders_weather_stays_and_every_lead_is_done(tables, names, fastest_first):
    s = weather_leads(tables, names)
    values = [0] * 4
    for rank, b in enumerate(fastest_first):
        values[b] = 400 - 40 * rank
    out = show(tables, speeds(s, values))
    assert int(out.weather) == last_weather(fastest_first) and int(out.weather_timer) == 0
    assert np.asarray(out.entry_pending).tolist() == [0, 0, 0, 0]
    assert np.asarray(out.weather_announced).tolist() == [1, 1, 1, 1]
    # The player's two leads reveal their ability while the weather up is not their own (:3749).
    known = np.asarray(out.ai_ability_known)
    assert known[int(out.active[0])] != 0 and known[int(out.active[2])] != 0


def test_equal_speeds_follow_the_tie_bits_over_all_64_rows(tables, names):
    s = weather_leads(tables, names)
    for ties in itertools.product((0, 1), repeat=6):
        out = show(tables, s, np.asarray(ties, np.int32))
        assert int(out.weather) == last_weather(transcribed_order(ties)), ties


def test_the_sort_reads_the_quick_claw_draw(tables, names):
    """Battler 1 (Drought) is slower than battler 0 (Drizzle). With ``qc_roll[1]`` and a Quick Claw
    it sorts first, so battler 0's rain is the one left; without the draw battler 1's sun."""
    s = weather_leads(tables, names)
    s = speeds(s, [300, 200, 150, 100])
    p, slot = party_of(s, 1), int(s.active[1])
    s = s._replace(item=s.item.at[p, slot].set(names.item("Quick Claw")))
    assert int(show(tables, s).weather) == Weather.SUN
    drawn = s._replace(qc_roll=s.qc_roll.at[1].set(1))
    assert int(show(tables, drawn).weather) == Weather.RAIN


# ---------------------------------------------------------------- Intimidate across a pause


def intimidating(tables, names, sub_hp):
    """Battler 0 U-turns into battler 1 (Substitute with ``sub_hp``); the newcomer (party 0 slot 2)
    has Intimidate."""
    s = pivoting(tables, names)
    s = s._replace(
        substitute_hp=s.substitute_hp.at[1].set(sub_hp),
        ability=s.ability.at[0, 2].set(int(AbilityId.INTIMIDATE)),
    )
    return s


@pytest.mark.parametrize("sub_hp", [1, 5000])
def test_intimidate_skips_the_foe_whose_substitute_the_action_hit(tables, names, sub_hp):
    """Broken (1 HP) or not (5000 HP): the hit set battler 1's flag (bcp.c:3371), it crosses the
    party list as ``ts_sub_hit`` and the newcomer's Intimidate skips it (script.c:9179-9182);
    battler 3 loses 1 stage (+2 from its own Swords Dance), battler 1 and the ally none."""
    s = intimidating(tables, names, sub_hp)
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    assert np.asarray(paused.ts_sub_hit).tolist() == [0, 1, 0, 0]
    out = resume(tables, paused)
    assert int(out.stages[1, ATK]) == 2 and int(out.stages[3, ATK]) == 1
    assert int(out.stages[2, ATK]) == 2
    assert np.asarray(out.ts_sub_hit).tolist() == [0, 0, 0, 0]


def test_intimidate_without_a_substitute_hits_both_foes(tables, names):
    s = intimidating(tables, names, 0)
    paused, _ = play(tables, s, [0, 0, 0, 0], ORDER, targets=(1, 0, 0, 0))
    out = resume(tables, paused)
    assert int(out.stages[1, ATK]) == 1 and int(out.stages[3, ATK]) == 1


# ---------------------------------------------------------------- replacement then ShowBattleMon


@pytest.mark.parametrize("newcomer_1_faster", [True, False])
def test_two_trainers_replacements_run_their_states_at_the_next_show(
    tables, names, newcomer_1_faster
):
    s = four(tables, names, TWO_TRAINERS)
    fmt = format_of(s)
    s = s._replace(entry_pending=jnp.zeros_like(s.entry_pending))  # the leads' states ran
    s = speeds(s, [200, 0, 150, 0])
    for b in (1, 3):
        s = s._replace(hp=s.hp.at[party(fmt, b), int(s.active[b])].set(0))
    fast, slow = (300, 100) if newcomer_1_faster else (100, 300)
    p1, p3 = party(fmt, 1), party(fmt, 3)
    s = s._replace(
        ability=s.ability.at[p1, 1]
        .set(int(AbilityId.DRIZZLE))
        .at[p3, 1]
        .set(int(AbilityId.DROUGHT)),
        stats=s.stats.at[p1, 1, Stat.SPE - 1].set(fast).at[p3, 1, Stat.SPE - 1].set(slow),
    )
    waiting = replace_round(tables, s, np.asarray([-1, 1, -1, 1], np.int8))
    assert np.asarray(waiting.entry_pending).tolist() == [0, 1, 0, 1]
    assert int(waiting.weather) == Weather.NONE  # subscript_show_party_list.s runs no state
    out = show(tables, waiting)
    # The slower newcomer's weather stays: battler 3's sun when battler 1 is faster.
    assert int(out.weather) == (Weather.SUN if newcomer_1_faster else Weather.RAIN)
    assert np.asarray(out.entry_pending).tolist() == [0, 0, 0, 0]


# ---------------------------------------------------------------- the root


def test_sample_root_runs_the_start_states_once(tables, names):
    s = weather_leads(tables, names)
    run = jax.jit(sample_root)
    out = run(jax.random.PRNGKey(0), tables, s)
    assert np.asarray(out.entry_pending).tolist() == [0, 0, 0, 0]
    assert int(out.weather) in {int(w) for _, w in HOLDERS}
    assert int(out.war_swaps) == 0
    assert np.asarray(out.weather_announced).tolist() == [1, 1, 1, 1]


def test_root_belief_splits_a_weather_war_into_its_visible_results(tables, names):
    s = weather_leads(tables, names)
    belief = root_belief(tables, s)
    assert sum(p for _, p in belief) == pytest.approx(1.0)
    expected: dict[int, int] = {}
    for ties in itertools.product((0, 1), repeat=6):
        w = int(last_weather(transcribed_order(ties)))
        expected[w] = expected.get(w, 0) + 1
    got: dict[int, float] = {}
    for st, p in belief:
        assert np.asarray(st.entry_pending).tolist() == [0, 0, 0, 0]
        got[int(st.weather)] = got.get(int(st.weather), 0.0) + p
    assert set(got) == set(expected) and len(expected) > 1
    for w, n in expected.items():
        assert got[w] == pytest.approx(n / 64)


def test_run_states_clears_entry_pending_only_for_the_fresh(tables, names):
    s = four(tables, names, DOUBLES)
    out = run_states(tables, s, jnp.arange(4), jnp.asarray([True, False, True, False]))
    assert np.asarray(out.entry_pending).tolist() == [0, 1, 0, 1]
