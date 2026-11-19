"""D2-16: Intimidate, Download and Trace over both foes at B = 4 (``hooks.intimidate`` /
``download`` / ``trace``, ``targets.trace_target`` / ``opposing``), engine against reference.
Battlers 0, 2 are the player's, 1, 3 the opponent's; opponents(0) = (3, 1)."""

from __future__ import annotations

from nuzline.data.tables import to_numpy
from nuzline.engine import targets
from nuzline.engine.format import format_of
from nuzline.engine.hooks import download, intimidate, trace
from nuzline.engine.ids import AbilityId, Stat
from nuzline.reference import engine as ref
from nuzline.reference.convert import to_ref
from tests.engine.d2_states import four, with_ability, with_hp


def atk(s, b):
    return int(s.stages[b, Stat.ATK])


def test_opposing_static(tables, names):
    s = four(tables, names)
    assert [int(x) for x in targets.opposing(format_of(s), 0)] == [3, 1]
    assert [int(x) for x in targets.opposing(format_of(s), 3)] == [2, 0]


def test_intimidate_both_foes(tables, names):
    s = with_ability(four(tables, names), 0, AbilityId.INTIMIDATE)
    out = intimidate(s, 0)
    assert [atk(out, b) for b in range(4)] == [0, -1, 0, -1]
    r = to_ref(s)
    for b in ref.opposing(r, 0):
        ref._intimidate(r, 0, b)
    assert [r.battlers[b].stages[Stat.ATK] for b in range(4)] == [0, -1, 0, -1]


def test_intimidate_skips(tables, names):
    s = with_ability(four(tables, names), 0, AbilityId.INTIMIDATE)
    s = with_ability(s, 1, AbilityId.CLEAR_BODY)
    s = with_hp(s, 3, 0)
    out = intimidate(s, 0)
    assert [atk(out, b) for b in range(4)] == [0, 0, 0, 0]
    s = with_ability(s, 1, AbilityId.PRESSURE)
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(5))
    assert atk(intimidate(s, 0), 1) == 0  # behind a Substitute
    s = s._replace(mist=s.mist.at[1].set(3))
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(0))
    assert atk(intimidate(s, 0), 1) == 0  # the side's Mist


def _def_spd(s, b, defense, sp_def):
    """Battler ``b``'s active Pokemon has Defense ``defense`` and Sp. Def ``sp_def``."""
    p, slot = format_of(s).party_of[b], int(s.active[b])
    stats = s.stats.at[p, slot, Stat.DEF - 1].set(defense)
    return s._replace(stats=stats.at[p, slot, Stat.SPD - 1].set(sp_def))


def test_download_sums_both_foes(tables, names):
    """SWITCH_IN_CHECK_STATE_DOWNLOAD (battle_lib.c:3815-3851): Defense and Sp. Def summed, each
    through its stage, over every opposing battler with HP and no substitute; sumDef >= sumSpDef
    raises Sp. Atk, else Attack, through subscript_update_stat_stage. Each case's sum disagrees
    with one foe alone: a one-foe Download fails one of the two."""
    base = with_ability(four(tables, names), 0, AbilityId.DOWNLOAD)
    # (RIGHT foe 3, LEFT foe 1) as (Defense, Sp. Def), foe 1's Defense stage, the stat raised
    cases = (
        # 100 + 160 * 2/8 = 140 < 60 + 100 = 160: Attack; foe 3 alone (100 >= 60) gives Sp. Atk,
        # and foe 1's Defense read without its -6 stage (260 >= 160) gives Sp. Atk too
        (((100, 60), (160, 100)), -6, Stat.ATK),
        # 60 + 100 = 160 >= 100 + 40 = 140: Sp. Atk; foe 3 alone (60 < 100) gives Attack
        (((60, 100), (100, 40)), 0, Stat.SPA),
    )
    for ((d3, s3), (d1, s1)), stage, raised in cases:
        s = _def_spd(_def_spd(base, 3, d3, s3), 1, d1, s1)
        s = s._replace(stages=s.stages.at[1, Stat.DEF].set(stage))
        out = download(s, 0, targets.opposing(format_of(s), 0))
        other = Stat.SPA if raised == Stat.ATK else Stat.ATK
        assert (int(out.stages[0, raised]), int(out.stages[0, other])) == (1, 0), raised
        r = to_ref(s)
        ref._download(r, 0)
        assert (r.battlers[0].stages[raised], r.battlers[0].stages[other]) == (1, 0), raised


def test_trace_coin_picks(tables, names):
    s = four(tables, names)
    s = with_ability(s, 0, AbilityId.TRACE)
    s = with_ability(s, 3, AbilityId.INTIMIDATE)  # RIGHT slot of battler 0
    s = with_ability(s, 1, AbilityId.CLEAR_BODY)  # LEFT slot
    for coin, want in ((0, 3), (1, 1)):
        got, found = targets.trace_target(tables, s, 0, coin)
        assert (int(got), bool(found)) == (want, True)
        assert ref.trace_target(to_numpy(tables), to_ref(s), 0, coin) == want
        out, traced = trace(tables, s, 0, coin)
        assert bool(traced)
        assert int(out.ability[0, out.active[0]]) == int(
            s.ability[format_of(s).party_of[want], s.active[want]]
        )
    # one foe fainted: the other with either coin
    s2 = with_hp(s, 3, 0)
    for coin in (0, 1):
        assert int(targets.trace_target(tables, s2, 0, coin)[0]) == 1
    # neither eligible
    s4 = with_hp(with_hp(s, 3, 0), 1, 0)
    got, found = targets.trace_target(tables, s4, 0, 0)
    assert not bool(found)
    assert ref.trace_target(to_numpy(tables), to_ref(s4), 0, 0) is None
    out, traced = trace(tables, s4, 0, 0)
    assert not bool(traced) and int(out.ts_trace_fires[0]) == 0


def test_trace_fire_counted_in_step_only(tables, names):
    s = with_ability(four(tables, names), 0, AbilityId.TRACE)
    s = with_ability(s, 3, AbilityId.INTIMIDATE)
    out, _ = trace(tables, s, 0, 0, count=True)
    assert int(out.ts_trace_fires[0]) == 1
    out, _ = trace(tables, s, 0, 0)
    assert int(out.ts_trace_fires[0]) == 0
