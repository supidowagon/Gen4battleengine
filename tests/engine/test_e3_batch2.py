"""E3 batch 2 (rulings E3-B2-n, docs/kaizo/e3-ledger.md): Fling (#233, E3-B2-1), each case checked
against the reference engine.

Decomp (external/pokeplatinum): ``res/battle/scripts/effects/effect_script_0233.s``,
``res/battle/scripts/subscripts/subscript_fling.s``, ``src/battle/battle_lib.c``
(``BattleSystem_FlingItem`` :5857-6158, ``Battler_ItemFlingPower`` :5491-5498),
``src/battle/battle_script.c`` (``BtlCmd_TryFling`` :8144-8154, ``BtlCmd_RemoveItem``
:8972-8984)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.damage import (
    CHART_EYED,
    CHART_IDENTIFIED,
    damage_table,
    effectiveness,
    identified,
)
from nuzline.engine.enumerate import INPUT_CLAUSE_EFFECTS, NON_TABLE_DAMAGE_EFFECTS
from nuzline.engine.field import with_weather
from nuzline.engine.ids import (
    ABILITY_DECOMP_IDS,
    MOVE_STRUGGLE,
    AbilityId,
    EffectId,
    FieldCondition,
    FlingEffect,
    MoveFlag,
    Stat,
    Status,
    Weather,
)
from nuzline.engine.order import CALL_MIRROR_MOVE, MOVE_ACTIONS, call_kind, effective_speed
from nuzline.engine.outcome import MAX_HITS, empty_outcome, hit_threshold
from nuzline.engine.state import make_state
from nuzline.engine.step import replace as send_in
from tests.engine.helpers import (
    A,
    as_species,
    assert_matches_reference,
    roark_state,
    weather_of,
)
from tools.effect_map import DECOMP_EFFECT_TO_ID

SPE = 4


def _spec(n, species, level, moves, item=""):
    return PokemonSpec(
        species=n.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(n.move(m) for m in moves) + (0,) * (4 - len(moves)),
        item=n.item(item) if item else 0,
        ability_slot=0,
        gender=0,
        friendship=70,
    )


def _fight(t, n, item="", foe_item="", opp=("Tackle",), foe="Geodude", foe_level=40):
    """A level 20 Monferno (Fling, ``item``) against a Geodude (``foe_item``), no abilities,
    one Pokemon each; the Monferno is faster than the Geodude."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 20, ("Fling",), item)],
        [_spec(n, foe, foe_level, opp, foe_item)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    return s._replace(stats=s.stats.at[0, 0, SPE].set(200).at[1, 0, SPE].set(1))


def _hits():
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))


def _go(t, n, s, out=None, foe="Tackle"):
    return assert_matches_reference(
        t, s, A("Fling", foe, s, n), out if out is not None else _hits()
    )


def _hp(s, side):
    return int(s.hp[side, 0])


def _stage(s, side, stat):
    return int(s.stages[side, stat])


# --- mapping and the item rules ---------------------------------------------------------------


def test_fling_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 233 (effect_script_0233.s) is EffectId.FLING in both games; the fling power and
    effect of the items are the item tables' (``item_fling_power``, ``item_fling_effect``)."""
    assert DECOMP_EFFECT_TO_ID[233] == (EffectId.FLING, (0, 0, 0))
    assert int(tables.move_effect[names.move("Fling")]) == EffectId.FLING
    assert int(kaizo_tables.move_effect[kaizo_names.move("Fling")]) == EffectId.FLING
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.item_fling_power[n.item("Iron Ball")]) == 130
        assert int(t.item_fling_effect[n.item("Flame Orb")]) == FlingEffect.BURN
        assert int(t.item_fling_effect[n.item("Toxic Orb")]) == FlingEffect.BADLY_POISON
        assert int(t.item_fling_effect[n.item("Light Ball")]) == FlingEffect.PARALYZE
        assert int(t.item_fling_effect[n.item("King’s Rock")]) == FlingEffect.FLINCH


def test_fling_with_an_item_hits_with_its_power_and_loses_it(tables, names):
    """effect_script_0233.s:9-18: CalcDamage with the item's power, then RemoveItem: the user's
    item is gone and the foe took damage; a heavier item hits harder (Iron Ball 130 against
    Light Ball 30)."""
    heavy = _go(tables, names, _fight(tables, names, "Iron Ball"))
    light = _go(tables, names, _fight(tables, names, "Light Ball"))
    full = int(_fight(tables, names).hp[1, 0])
    assert int(heavy.item[0, 0]) == 0 and int(light.item[0, 0]) == 0
    assert full - _hp(heavy, 1) > full - _hp(light, 1) > 0


def test_fling_fails_without_a_usable_item(tables, names):
    """BattleSystem_FlingItem (battle_lib.c:5863-5865): no item, an item of fling power 0, or
    Embargo (``Battler_ItemFlingPower``, :5491-5498) fails; so do a raw Multitype ability and a
    Griseous Orb (effect_script_0233.s:5-6). The item is kept and the foe takes nothing."""
    t, n = tables, names
    base = _fight(t, n, "Iron Ball")
    zero = next(
        i for i in range(1, len(n.item_names)) if int(t.item_fling_power[i]) == 0 and i != 112
    )
    cases = (
        _fight(t, n),
        base._replace(item=base.item.at[0, 0].set(zero)),
        base._replace(embargo=base.embargo.at[0].set(3)),
        make_state(
            t,
            [_spec(n, "Arceus", 20, ("Fling",), "Iron Ball")],
            [_spec(n, "Geodude", 40, ("Tackle",))],
        ),  # Multitype
        _fight(t, n, "Griseous Orb"),
    )
    for s in cases:
        out = _go(t, n, s)
        assert _hp(out, 1) == _hp(s, 1)
        assert int(out.item[0, 0]) == int(s.item[0, 0])


def test_fling_misses_but_still_loses_the_item(tables, names):
    """RemoveItem (effect_script_0233.s:18) runs before TryMove's accuracy check: a miss still
    loses the item."""
    s = _fight(tables, names, "Iron Ball")
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = _go(tables, names, s, miss)
    assert int(out.item[0, 0]) == 0 and _hp(out, 1) == _hp(s, 1)


# --- the flung item's effect on the foe -------------------------------------------------------


def test_flung_orbs_and_light_ball_status_the_foe(tables, names):
    """subscript_fling.s with the INDIRECT status subscripts: Flame Orb burns, Toxic Orb badly
    poisons, Light Ball paralyzes (battle_lib.c:6045-6073)."""
    for item, status in (
        ("Flame Orb", Status.BURN),
        ("Toxic Orb", Status.TOXIC),
        ("Light Ball", Status.PARALYSIS),
    ):
        s = _fight(tables, names, item, foe="Bidoof", foe_level=20)
        out = _go(tables, names, s)
        assert int(out.status[1, 0]) == status, item


def test_flung_status_is_blocked_by_a_status_a_substitute_or_embargo(tables, names):
    """The subscript is skipped behind a Substitute (subscript_fling.s:5-7), the status
    subscripts fail on an already-statused foe, and a foe with MOVE_EFFECT_EMBARGO gets no
    script (battle_lib.c:6145-6146)."""
    base = _fight(tables, names, "Flame Orb", foe="Bidoof", foe_level=20)
    sub = base._replace(substitute_hp=base.substitute_hp.at[1].set(60))
    out = _go(tables, names, sub)
    assert int(out.status[1, 0]) == Status.NONE
    sick = base._replace(status=base.status.at[1, 0].set(Status.POISON))
    out = _go(tables, names, sick)
    assert int(out.status[1, 0]) == Status.POISON
    emb = base._replace(embargo=base.embargo.at[1].set(3))
    out = _go(tables, names, emb)
    assert int(out.status[1, 0]) == Status.NONE


def test_flung_berries_heal_and_cure_the_foe(tables, names):
    """HP_RESTORE (Oran Berry, +10), PRZ_RESTORE (Cheri Berry) and PSN_RESTORE (Pecha Berry) on
    the foe (battle_lib.c:5868-5906)."""
    s = _fight(tables, names, "Oran Berry", foe="Bidoof", foe_level=40)
    hurt = s._replace(hp=s.hp.at[1, 0].set(5))
    out = _go(tables, names, hurt)
    assert _hp(out, 1) > 5
    for berry, status in (("Cheri Berry", Status.PARALYSIS), ("Pecha Berry", Status.POISON)):
        s = _fight(tables, names, berry, foe="Bidoof", foe_level=40)
        sick = s._replace(status=s.status.at[1, 0].set(status))
        out = _go(tables, names, sick)
        assert int(out.status[1, 0]) == Status.NONE, berry
        _go(tables, names, s)  # a healthy foe: the berry does nothing, both engines agree


def test_flung_stat_berries_and_starf(tables, names):
    """ATK_UP (Liechi Berry) raises the foe's Attack one stage, +1 only below 12
    (battle_lib.c:6075-6108); Starf (RANDOM_UP2) draws a stat among those below +6 at TryFling
    (:6110-6129, ``RandNext() % 5`` redrawn while the stage is 12): by the ``% n`` convention
    (F-10) the held Starf Berry's draw of the DEFENDER, ``Outcome.starf[1, 0]`` read modulo the
    count of its stats below +6 (``hooks.starf_stat``, review b2 #4): 0 is Attack, 3 Sp. Atk, and
    with Attack at +6 the rank skips it (0 is Defense). The attacker's Acupressure draw is not
    read."""
    s = _fight(tables, names, "Liechi Berry", foe="Bidoof", foe_level=40)
    out = _go(tables, names, s)
    assert _stage(out, 1, Stat.ATK) == 1
    maxed = s._replace(stages=s.stages.at[1, Stat.ATK].set(6))
    out = _go(tables, names, maxed)
    assert _stage(out, 1, Stat.ATK) == 6
    star = _fight(tables, names, "Starf Berry", foe="Bidoof", foe_level=40)

    def draw(value, acu=0):
        out = _hits()._replace(acupressure=jnp.full(2, acu, jnp.int16))
        return out._replace(starf=out.starf.at[1, 0].set(value))

    out = _go(tables, names, star, draw(0, acu=3))
    assert _stage(out, 1, Stat.ATK) == 2
    out = _go(tables, names, star, draw(3))
    assert _stage(out, 1, Stat.SPA) == 2 and sum(_stage(out, 1, st) for st in range(1, 6)) == 2
    capped = star._replace(stages=star.stages.at[1, Stat.ATK].set(6))
    out = _go(tables, names, capped, draw(0))
    assert _stage(out, 1, Stat.DEF) == 2 and _stage(out, 1, Stat.ATK) == 6


def test_a_flung_starf_uses_up_a_draw_of_the_defenders_queue(tables, names):
    """E3-B2-22 (G-27, G-28): Fling's RANDOM_UP2 loop is its own ``RandNext() % 5``
    (battle_lib.c:6121), so it takes the defender's next queue entry (``Outcome.starf[1, 0]``)
    and the defender's own Starf Berry, eaten at its pinch after the hit, reads the next one
    (``Outcome.starf[1, 2]``): Attack from the fling, Sp. Atk from the eat (the old single draw
    would raise Attack twice). Engine equals reference."""
    s = _fight(tables, names, "Starf Berry", foe_item="Starf Berry", foe="Bidoof", foe_level=40)
    s = s._replace(hp=s.hp.at[1, 0].set(int(s.max_hp[1, 0]) // 4))
    out = _hits()
    out = out._replace(starf=out.starf.at[1, 0].set(0).at[1, 2].set(3))
    after = _go(tables, names, s, out)
    assert int(after.hp[1, 0]) > 0
    assert _stage(after, 1, Stat.ATK) == 2 and _stage(after, 1, Stat.SPA) == 2


def test_a_missed_flung_starf_still_uses_up_the_defenders_draw(tables, names):
    """The RANDOM_UP2 loop runs at TryFling (BattleSystem_FlingItem, battle_lib.c:6110-6127, from
    effect_script_0233.s:7, before CalcCrit and TryMove), so a Fling that then misses has taken the
    defender's queue entry 0 (``Outcome.starf[1, 0]``): its own Starf Berry, eaten at its pinch after
    its move, reads the next one (``starf[1, 2]``, ``ids.STARF_QUEUE``: Sp. Atk), not Attack. Engine
    equals reference."""
    s = _fight(tables, names, "Starf Berry", foe_item="Starf Berry", foe="Bidoof", foe_level=40)
    s = s._replace(hp=s.hp.at[1, 0].set(int(s.max_hp[1, 0]) // 4))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    miss = miss._replace(starf=miss.starf.at[1, 0].set(0).at[1, 2].set(3))
    after = _go(tables, names, s, miss)
    assert int(after.item[1, 0]) == 0
    assert _stage(after, 1, Stat.ATK) == 0 and _stage(after, 1, Stat.SPA) == 2


def test_a_flung_starf_raises_the_stat_picked_at_tryfling(tables, names):
    """battle_lib.c:6110-6127 picks the stat at TryFling from the stages before the hit; the hit's
    subscript_held_item_sharply_raise_stat.s:14-18 adds 2 to that stat, capped at 12, with no
    failure test. A stage raised to +6 in between (a critical hit's Anger Point,
    subscript_critical_hit.s:10-15, ONE_HIT_CRITICAL bcp.c:3472 before ONE_HIT_TRIGGER_SECONDARY
    :3484; Anger Point is not modelled) leaves the pick: the maxed stat stays at +6 and no other
    stat rises (unit test of ``hooks.fling_try`` / ``hooks.fling_effect``)."""
    from nuzline.engine.hooks import fling_effect, fling_try

    s = _fight(tables, names, "Starf Berry", foe="Bidoof", foe_level=40)
    s = s._replace(starf_roll=s.starf_roll.at[1].set(0))  # rank 0: Attack
    item = s.item[0, 0]
    tried, pick = fling_try(tables, s, 1, 0, item, True)
    assert int(pick) == Stat.ATK + 1 and int(tried.starf_taken[1]) == 1
    maxed = tried._replace(stages=tried.stages.at[1, Stat.ATK].set(6))
    out = fling_effect(tables, maxed, 1, 0, item, True, pick)
    assert np.asarray(out.stages[1]).tolist() == np.asarray(maxed.stages[1]).tolist()
    out = fling_effect(tables, tried, 1, 0, item, True, pick)
    assert _stage(out, 1, Stat.ATK) == 2
    none, nopick = fling_try(tables, s, 1, 0, item, False)
    assert int(nopick) == 0 and int(none.starf_taken[1]) == 0


def test_flung_lansat_and_white_herb(tables, names):
    """Lansat Berry sets the foe's Focus Energy (subscript_held_item_raise_crit.s:11); White
    Herb resets its lowered stages at TryFling (battle_lib.c:6029-6036), even on a miss."""
    s = _fight(tables, names, "Lansat Berry", foe="Bidoof", foe_level=40)
    out = _go(tables, names, s)
    assert int(out.focus_energy[1]) == 1
    herb = _fight(tables, names, "White Herb", foe="Bidoof", foe_level=40)
    low = herb._replace(stages=herb.stages.at[1, Stat.ATK].set(-2).at[1, Stat.DEF].set(-1))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    for out in (_go(tables, names, low), _go(tables, names, low, miss)):
        assert _stage(out, 1, Stat.ATK) == 0 and _stage(out, 1, Stat.DEF) == 0


def test_flung_leppa_restores_the_foes_pp(tables, names):
    """PP_RESTORE (Leppa Berry, +10) at TryFling on the foe's most depleted slot
    (battle_lib.c:5908-5933)."""
    s = _fight(tables, names, "Leppa Berry", foe="Bidoof", foe_level=40, opp=("Tackle", "Growl"))
    used = s._replace(pp=s.pp.at[1, 0, 1].set(1))
    out = _go(tables, names, used, foe="Growl")
    assert int(out.pp[1, 0, 1]) > int(used.pp[1, 0, 1]) - 1


def test_flung_kings_rock_flinches_a_slower_foe(tables, names):
    """FLING_EFFECT_FLINCH (King's Rock): the foe has not moved yet, so it flinches
    (subscript_flinch_mon.s:5) and its Tackle does not land."""
    s = _fight(tables, names, "King’s Rock", foe="Bidoof", foe_level=40)
    out = _go(tables, names, s)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])


# --- hit chance, crit and the enumeration -----------------------------------------------------


def test_a_flung_starf_makes_the_defenders_starf_draw_relevant(tables, names):
    """Review b2 #4: the enumeration reads ``Outcome.starf[d, 0]`` of the Fling's defender ``d``
    (``enumerate._starf_reads``), with its five-value law at a count fixed at 5, and no longer
    the flinger's Acupressure draw."""
    from nuzline.engine.enumerate import outcome_laws

    s = _fight(tables, names, "Starf Berry", foe="Bidoof", foe_level=40)
    laws = jax.tree.map(
        np.asarray, jax.jit(outcome_laws)(tables, s, A("Fling", "Tackle", s, names))
    )
    relevant = np.asarray(laws.relevant.starf)
    assert bool(relevant[1, 0]) and not bool(np.asarray(laws.relevant.acupressure)[0])
    assert not bool(np.asarray(laws.starf_loose)[1, 0])
    plain = _fight(tables, names, "Oran Berry", foe="Bidoof", foe_level=40)
    laws = jax.tree.map(
        np.asarray, jax.jit(outcome_laws)(tables, plain, A("Fling", "Tackle", plain, names))
    )
    assert not bool(np.asarray(laws.relevant.starf)[1, 0])


def test_fling_is_an_input_clause_and_a_crit_writer():
    """A foe's Fling is an input clause of the enumeration (its item's effect, the White Herb
    reset, its own item gone before the side's hit), deals no table damage, and writes the crit
    inputs (a flung Lansat Berry's Focus Energy)."""
    from nuzline.engine.outcome import CRIT_ODDS_EFFECTS

    assert EffectId.FLING in INPUT_CLAUSE_EFFECTS
    assert EffectId.FLING in NON_TABLE_DAMAGE_EFFECTS
    assert EffectId.FLING in CRIT_ODDS_EFFECTS


def test_fling_is_seen_by_hit_thresholds(tables, names):
    """A faster foe's flung White Herb resets the side's accuracy drop and a flung Micle Berry
    sets its micle flag before the side's check: the hit-class law holds both thresholds."""
    from nuzline.engine.enumerate import hit_class_law, outcome_laws

    for item in ("White Herb", "Micle Berry"):
        s = make_state(
            tables,
            [_spec(names, "Monferno", 20, ("Rock Throw",))],
            [_spec(names, "Geodude", 40, ("Fling",), item)],
        )
        s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
        s = s._replace(
            stats=s.stats.at[0, 0, SPE].set(1).at[1, 0, SPE].set(200),
            stages=s.stages.at[0, Stat.ACC].set(-2),
        )
        acts = A("Rock Throw", "Fling", s, names)
        laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
        values, _weights, den = hit_class_law(laws.hit_thresholds[0][laws.hit_possible[0]])
        assert den == 100 and len(values) > 2, item


# --- #9 Mirror Move (E3-B2-2) -----------------------------------------------------------------
#
# effect_script_0009.s -> BtlCmd_SetMirrorMove (battle_script.c:3483-3525) calls the user's
# ``moveCopied`` (``BattleState.move_copied``) when Move_CanBeEncored allows it; UpdateFlagsWhenHit
# (battle_controller_player.c:4480-4486) writes it for the defender of a move with
# MOVE_FLAG_CAN_MIRROR_MOVE whose attack message showed; a battler's switch-in or faint clears it
# (battle_lib.c:2125-2149, :2203-2220).


def _mirror_fight(
    t, n, foe=("Tackle", "Growl"), fast=True, copied="", mine=("Mirror Move", "Rock Throw")
):
    """A level 50 Monferno (``mine``) against a level 50 Geodude (``foe``); ``copied`` names the
    move the Monferno's ``moveCopied`` holds at the start."""
    s = make_state(t, [_spec(n, "Monferno", 50, mine)], [_spec(n, "Geodude", 50, foe)])
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500 if fast else 1))
    if copied:
        move = n.move(copied)
        s = s._replace(
            move_copied=s.move_copied.at[0].set(move),
            move_copied_hit=s.move_copied_hit.at[0, 1].set(move),
        )
    return s


def _mgo(t, n, s, out=None, foe="Tackle"):
    """One turn of Mirror Move against ``foe``, both engines (hits, no crits)."""
    return assert_matches_reference(
        t, s, A("Mirror Move", foe, s, n), out if out is not None else _hits()
    )


def test_mirror_move_is_mapped_and_flagged_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 9 (effect_script_0009.s) is EffectId.MIRROR_MOVE in both games and the only move
    with it; MOVE_FLAG_CAN_MIRROR_MOVE is ``MoveFlag.MIRROR_MOVE``: Tackle and Trick Room have it,
    Mirror Move and Metronome do not (so an invoker records nothing)."""
    assert DECOMP_EFFECT_TO_ID[9] == (EffectId.MIRROR_MOVE, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        effect = np.asarray(t.move_effect)
        assert list(np.flatnonzero(effect == EffectId.MIRROR_MOVE)) == [n.move("Mirror Move")]
        flags = np.asarray(t.move_flags)
        for move, has in (("Tackle", True), ("Trick Room", True), ("Mirror Move", False)):
            assert bool(flags[n.move(move)] & MoveFlag.MIRROR_MOVE) is has, move
        assert not flags[n.move("Metronome")] & MoveFlag.MIRROR_MOVE
        assert int(call_kind(t, n.move("Mirror Move"))) == CALL_MIRROR_MOVE


def test_mirror_move_calls_the_held_move_as_its_own(tables, names):
    """A fast user whose moveCopied holds Rock Throw calls it: the foe takes what the user's own
    Rock Throw deals (accuracy, type, damage of the called move, effects with the user as
    attacker), and the PP spent is Mirror Move's alone."""
    s = _mirror_fight(tables, names, copied="Rock Throw")
    out = _mgo(tables, names, s, foe="Growl")
    mine = _mirror_fight(tables, names)
    own = assert_matches_reference(tables, mine, A("Rock Throw", "Growl", mine, names), _hits())
    taken = int(s.hp[1, 0]) - _hp(out, 1)
    assert taken > 0 and taken == int(mine.hp[1, 0]) - _hp(own, 1)
    mirror = names.move_slot(s, 0, "Mirror Move")
    rock = names.move_slot(s, 0, "Rock Throw")
    assert int(out.pp[0, 0, mirror]) == int(s.pp[0, 0, mirror]) - 1
    assert int(out.pp[0, 0, rock]) == int(s.pp[0, 0, rock])


def test_mirror_move_copies_a_faster_foes_move_of_this_turn(tables, names):
    """A slower user copies the move that hit it this turn (the foe's Tackle wrote ``move_copied``
    and ``move_copied_hit`` for it before it acted): the foe takes Tackle damage, and the user's own
    call records nothing for the foe (Mirror Move has no MOVE_FLAG_CAN_MIRROR_MOVE)."""
    s = _mirror_fight(tables, names, fast=False)
    out = _mgo(tables, names, s, foe="Tackle")
    tackle = names.move("Tackle")
    assert _hp(out, 1) < int(s.hp[1, 0]) and _hp(out, 0) < int(s.hp[0, 0])
    assert int(out.move_copied[0]) == tackle and int(out.move_copied_hit[0, 1]) == tackle
    assert int(out.move_copied[1]) == 0 and not np.asarray(out.move_copied_hit[1]).any()


def test_mirror_move_fails_without_a_move_or_with_a_move_encore_refuses(tables, names):
    """BtlCmd_SetMirrorMove (:3507): nothing copied, or a copied move Move_CanBeEncored refuses
    (the battle effect of Transform, Mimic, Sketch, Mirror Move, Encore or Struggle), fails: the
    foe is untouched."""
    for copied in ("", "Encore", "Transform"):
        s = _mirror_fight(tables, names, copied=copied)
        out = _mgo(tables, names, s, foe="Growl")
        assert _hp(out, 1) == int(s.hp[1, 0]), copied


def test_flagged_moves_are_recorded_and_the_rest_are_not(tables, names):
    """UpdateFlagsWhenHit (:4480-4486): a foe's Growl (flagged) is recorded for the user, its
    Swords Dance (not flagged) is not; a miss still records (the attack message showed)."""
    for foe, recorded in (("Growl", True), ("Swords Dance", False)):
        s = _mirror_fight(tables, names, foe=(foe, "Tackle"))
        out = _mgo(tables, names, s, foe=foe)
        want = names.move(foe) if recorded else 0
        assert int(out.move_copied[0]) == want, foe
        assert int(out.move_copied_hit[0, 1]) == want, foe
    s = _mirror_fight(tables, names)
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = _mgo(tables, names, s, miss, foe="Tackle")
    assert int(out.move_copied[0]) == names.move("Tackle")


def test_a_self_defended_flagged_move_records_for_its_user(tables, names):
    """Trick Room (flagged, RANGE_FIELD) has its user as defender (BattleSystem_Defender,
    battle_lib.c:1665-1669): moveCopied[user] = Trick Room, after the foe's Growl (Trick Room's
    priority -7 puts it last) wrote Growl there."""
    s = _mirror_fight(tables, names, mine=("Trick Room", "Rock Throw"))
    out = assert_matches_reference(tables, s, A("Trick Room", "Growl", s, names), _hits())
    room = names.move("Trick Room")
    assert int(out.move_copied[0]) == room and int(out.move_copied_hit[0, 0]) == room


def test_a_switch_clears_the_copied_moves(tables, names):
    """A battler's switch-in clears its own moveCopied and moveCopiedHit row and every foe's
    moveCopied and every holder's column of it (battle_lib.c:2125-2149)."""
    s = make_state(
        tables,
        [_spec(names, "Monferno", 50, ("Mirror Move",)), _spec(names, "Bidoof", 50, ("Tackle",))],
        [_spec(names, "Geodude", 50, ("Swords Dance",))],
    )
    s = s._replace(ability=s.ability.at[:, :].set(AbilityId.NONE))
    tackle, growl = names.move("Tackle"), names.move("Growl")
    s = s._replace(
        move_copied=jnp.asarray([tackle, growl], s.move_copied.dtype),
        move_copied_hit=jnp.asarray([[0, tackle], [growl, 0]], s.move_copied_hit.dtype),
    )
    acts = jnp.asarray([4 + 1, names.move_slot(s, 1, "Swords Dance")], jnp.int32)
    out = assert_matches_reference(tables, s, acts, _hits())
    assert int(out.active[0]) == 1
    assert not np.asarray(out.move_copied).any() and not np.asarray(out.move_copied_hit).any()


def test_a_switch_a_bag_item_and_a_pursuit_record_no_copied_move(tables, names):
    """UpdateFlagsWhenHit (bcp.c:4480-4486) runs only from LoopSpreadMoves (:3837) and
    UpdateMoveBuffers (:3956), on a FIGHT's path. A switch goes to MOVE_END through CheckAnyFainted
    (SwitchCommand :1985, :4008), a bag item straight there (ItemCommand :1973), and a Pursuit on
    the switcher runs inside subscript_switch_pokemon (subscript_pursuit.s, BtlCmd_TryPursuit
    battle_script.c:6895-6950): none writes moveCopied, so the move in the slot a switch or item
    action would clip to (flagged here) records nothing, and the switch-in clears the rest
    (battle_lib.c:2125-2149)."""
    flags = np.asarray(tables.move_flags)
    s = roark_state(tables, names)
    s = s._replace(
        trainer_items=s.trainer_items.at[1].set(jnp.asarray([names.item("Dire Hit"), 0, 0, 0])),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )
    for side in (0, 1):
        assert flags[int(s.moves[side, s.active[side], 3])] & MoveFlag.MIRROR_MOVE
    # A switch and a bag item: nothing recorded.
    out = assert_matches_reference(
        tables, s, [MOVE_ACTIONS + 1, FIRST_ITEM_ACTION], empty_outcome()
    )
    assert not np.asarray(out.move_copied).any() and not np.asarray(out.move_copied_hit).any()
    # A switch, then the foe's Leer on the newcomer: Leer recorded for it, nothing for the foe.
    leer = names.move("Leer")
    out = assert_matches_reference(
        tables, s, A(MOVE_ACTIONS + 1, "Leer", s, names), empty_outcome()
    )
    assert list(np.asarray(out.move_copied)) == [leer, 0]
    assert np.asarray(out.move_copied_hit).tolist() == [[0, leer], [0, 0]]
    # A Pursuit on the switcher: nothing recorded.
    p = roark_state(tables, names, opp_moves=["Pursuit"])
    assert flags[int(p.moves[0, p.active[0], 3])] & MoveFlag.MIRROR_MOVE
    out = assert_matches_reference(
        tables, p, A(MOVE_ACTIONS + 1, "Pursuit", p, names), empty_outcome()
    )
    assert int(out.hp[0, 0]) < int(p.hp[0, 0])  # Pursuit hit the switcher
    assert not np.asarray(out.move_copied).any() and not np.asarray(out.move_copied_hit).any()


def test_mirror_move_is_an_ordered_invoker_in_the_enumeration(tables, names):
    """``call_laws``: Mirror Move is ``CALL_MIRROR_MOVE``; behind a foe that Tackles (a move that
    records for its user) value 0 is the turn-start moveCopied (Rock Throw here) and value 1 the
    foe's Tackle; each weighs 1 and ``call_variants`` keeps both. Behind a foe whose move records
    nothing for it (Swords Dance) both values call the turn-start move. It is in both effect lists
    the enumeration reads an invoker through."""
    from nuzline.engine.enumerate import (
        INPUT_NEUTRAL_EFFECTS,
        call_laws,
        call_variants,
    )

    assert EffectId.MIRROR_MOVE in INPUT_NEUTRAL_EFFECTS
    assert EffectId.MIRROR_MOVE in NON_TABLE_DAMAGE_EFFECTS
    s = _mirror_fight(tables, names, fast=False, copied="Rock Throw")
    acts = A("Mirror Move", "Tackle", s, names)
    laws = jax.tree.map(np.asarray, call_laws(tables, s, acts))
    assert int(laws.kind[0]) == CALL_MIRROR_MOVE
    assert list(laws.weight[0][:2]) == [1, 1]
    assert list(laws.called[0][:2]) == [names.move("Rock Throw"), names.move("Tackle")]
    assert len(call_variants(laws)) == 2
    quiet = _mirror_fight(tables, names, fast=False, copied="Rock Throw", foe=("Swords Dance",))
    laws = jax.tree.map(np.asarray, call_laws(tables, quiet, A("Mirror Move", 0, quiet, names)))
    assert list(laws.called[0][:2]) == [names.move("Rock Throw")] * 2


# --- #93 Conversion 2 (E3-B2-3) ---------------------------------------------------------------
#
# effect_script_0093.s -> subscript_conversion_2.s -> BtlCmd_TryConversion2 (battle_script.c:4765-
# 4820): the user's types both become a type that resists the type of the last move that landed
# on it (``conv2_move`` / ``conv2_battler`` / ``conv2_type``, written by UpdateFlagsWhenHit,
# battle_controller_player.c:4498-4536). The row is the first eligible one of a
# ``BattleSystem_RandNext() % 112`` retry loop (battle_lib.c:2994-3007), drawn as
# ``Outcome.conv2[user, recorded type]``.

NORMAL, FLYING, ROCK, GHOST, STEEL, FIRE, WATER, DARK = 0, 2, 5, 7, 8, 10, 11, 17


def _c2_fight(
    t, n, foe=("Tackle",), mine=("Conversion 2", "Swords Dance"), user="Monferno", fast=False,
    foe_species="Geodude",
):  # fmt: skip
    """A level 50 ``user`` (``mine``) against a level 50 ``foe_species`` (``foe``), no abilities;
    the user is slower unless ``fast``."""
    s = make_state(t, [_spec(n, user, 50, mine)], [_spec(n, foe_species, 50, foe)])
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    return s._replace(stats=s.stats.at[0, 0, SPE].set(500 if fast else 1))


def _c2_out(user_pick=None, kind=NORMAL, hits=True):
    """No crits; ``conv2[0, kind]`` = ``user_pick`` when given."""
    out = empty_outcome()._replace(hit=jnp.full((2, MAX_HITS), 1 if hits else 0, jnp.int8))
    if user_pick is not None:
        out = out._replace(conv2=out.conv2.at[0, kind].set(user_pick))
    return out


def _c2_go(t, n, s, foe="Tackle", out=None, me="Conversion 2"):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _c2_out())


def _record(s, side, move, attacker, kind):
    return s._replace(
        conv2_move=s.conv2_move.at[side].set(move),
        conv2_battler=s.conv2_battler.at[side].set(attacker + 1),
        conv2_type=s.conv2_type.at[side].set(kind),
    )


def _types(s, side=0):
    return np.asarray(s.types[side, 0]).tolist()


def test_conversion_2_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 93 (BATTLE_EFFECT_CONVERSION2) is EffectId.CONVERSION_2 in both games and the only
    move with it; it is an input clause of the damage classes (it rewrites the user's types like
    Camouflage) and a status move for the damage tables."""
    assert DECOMP_EFFECT_TO_ID[93] == (EffectId.CONVERSION_2, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        effect = np.asarray(t.move_effect)
        assert list(np.flatnonzero(effect == EffectId.CONVERSION_2)) == [n.move("Conversion 2")]
    assert EffectId.CONVERSION_2 in INPUT_CLAUSE_EFFECTS
    assert EffectId.CONVERSION_2 in NON_TABLE_DAMAGE_EFFECTS


def test_the_row_table_is_the_decomps(tables):
    """``damage.conv2_pairs``: the decomp's 112 rows of ``sTypeMatchupMultipliers``
    (battle_lib.c:2399-2517), with the 0xFE separator at 108, the two Foresight rows after it and
    the 0xFF end; a pair is eligible when its multiplier is 0 or 5 and it weighs 586 (rows 0-15) or
    585 of the 65536 u16 values (65536 = 112 * 585 + 16)."""
    import re
    from pathlib import Path

    import pytest

    from nuzline.engine.damage import CONV2_ROWS, conv2_pairs
    from nuzline.engine.ids import Type

    weight, row = (np.asarray(x) for x in conv2_pairs(tables))
    assert CONV2_ROWS == 112 and 112 * 585 + 16 == 65536
    source = Path(__file__).resolve().parents[2] / "external/pokeplatinum/src/battle/battle_lib.c"
    if not source.exists():
        pytest.skip("decomp not present")
    text = source.read_text(encoding="utf-8")
    body = text.split("sTypeMatchupMultipliers[][3] = {")[1].split("};")[0]
    rows = re.findall(r"\{\s*(\w+),\s*(\w+),\s*(\w+)\s*\}", body)
    assert len(rows) == 112 and rows[108][0] == "0xFE" and rows[111][0] == "0xFF"
    kinds = {f"TYPE_{x.name}": int(x) for x in Type}
    multi = {"TYPE_MULTI_IMMUNE": 0, "TYPE_MULTI_NOT_VERY_EFF": 5, "TYPE_MULTI_SUPER_EFF": 20}
    eligible = {}
    for index, (a, d, m) in enumerate(rows):
        if a in kinds and multi[m] <= 5:
            eligible[(kinds[a], kinds[d])] = index
    got = {(int(a), int(d)): int(row[a, d]) for a, d in zip(*np.nonzero(weight), strict=True)}
    assert got == eligible
    for (a, d), index in eligible.items():
        assert int(weight[a, d]) == (586 if index < 16 else 585), (a, d)


def test_the_build_checks_both_games_chart_against_the_row_modulus():
    """Review b2 #12: ``tools.build_tables.check_conv2_rows`` holds both games' charts (108 rows,
    the 0xFE separator, the two Foresight rows, the 0xFF end: ``damage.CONV2_ROWS`` = 112, the
    compiled ``% NELEMS(sTypeMatchupMultipliers)`` of battle_lib.c:2994-3007) and refuses a chart
    with a row more or a Foresight row moved."""
    import copy
    import json

    from tools.build_tables import BuildError, check_conv2_rows

    root = Path(__file__).resolve().parents[2] / "data"
    for game in ("intermediate", "intermediate_kaizo"):
        path = root / game / "types.json"
        if not path.exists():
            pytest.skip(f"{path} not built")
        types = json.loads(path.read_text(encoding="utf-8"))
        check_conv2_rows(types["chart_order"], types["foresight_immunities"])
    order, fore = types["chart_order"], types["foresight_immunities"]
    extra = copy.deepcopy(order)
    a, d = next((a, d) for a, row in enumerate(extra) for d, v in enumerate(row) if v < 0)
    extra[a][d] = max(v for row in extra for v in row) + 1
    with pytest.raises(BuildError):
        check_conv2_rows(extra, fore)
    with pytest.raises(BuildError):
        check_conv2_rows(order, fore[:1])


def test_the_pick_law_is_uniform_over_the_eligible_types(tables, names):
    """Review b2 #3: BtlCmd_TryConversion2's 1000 tries of ``RandNext() % 112`` (battle_lib.c:
    2994-3007, battle_script.c:4783-4817) are uniform per row by the ``% n`` convention (E1 F-10),
    so ``enumerate._conv2`` is uniform over the eligible types, integer weights 1 over their
    count: a Rock / Ground user against a recorded Normal move has Steel and Ghost (the Foresight
    row) left, a Fire / Fighting one Water, Rock and Dragon (not its own Fire). The eligible set is
    read here from the decomp's rows (``damage.conv2_pairs``) and the user's types."""
    from nuzline.engine.damage import conv2_pairs
    from nuzline.engine.enumerate import _conv2, outcome_laws

    weight, _ = (np.asarray(x) for x in conv2_pairs(tables))
    expected = {(NORMAL, "Geodude"): [STEEL, GHOST], (FIRE, "Monferno"): [ROCK, WATER, 16]}
    for user, kind, foe in (("Geodude", NORMAL, "Tackle"), ("Monferno", FIRE, "Ember")):
        s = _c2_fight(tables, names, foe=(foe,), user=user)
        s = _record(s, 0, names.move(foe), 1, kind)
        acts = A("Conversion 2", foe, s, names)
        laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
        values, weights, den = _conv2(laws, 0, kind)
        own = {int(x) for x in np.asarray(s.types[0, 0])}
        rows = sorted(d for d in range(18) if weight[kind, d] and d not in own)
        assert values == rows == sorted(expected[(kind, user)]), user
        assert weights == [1] * len(rows) and den == len(rows)
    # The ??? type has no rows at all: one value, the move fails.
    assert _conv2(laws, 0, 9) == ([0], [1], 1)


def test_the_user_becomes_the_drawn_resistor_of_the_last_move(tables, names):
    """A slower user is Tackled first (Normal, recorded on it), then Conversion 2 turns it into the
    drawn defending type of the Normal rows: Rock, Steel or the Foresight row's Ghost (both
    types)."""
    s = _c2_fight(tables, names)
    for pick in (ROCK, STEEL, GHOST):
        out = _c2_go(tables, names, s, out=_c2_out(pick))
        assert _types(out) == [pick, pick], pick
        assert _hp(out, 0) < int(s.hp[0, 0])  # the Tackle landed


def test_a_user_never_picks_one_of_its_own_types(tables, names):
    """MON_IS_NOT_TYPE (battle_script.c:4789-4796): a Rock / Ground user against a Normal move has
    Rock excluded (Steel and Ghost only); and a draw that is not eligible falls to the linear
    scan's first row."""
    from nuzline.engine.damage import conv2_eligible

    s = _c2_fight(tables, names, user="Geodude", foe_species="Monferno")
    assert set(np.flatnonzero(np.asarray(conv2_eligible(tables, s, 0, NORMAL)))) == {STEEL, GHOST}
    out = _c2_go(tables, names, s, out=_c2_out(ROCK))  # an ineligible draw: the first row (Steel)
    assert _types(out) == [STEEL, STEEL]


def test_a_record_is_written_for_every_landing_move_and_cleared_otherwise(tables, names):
    """UpdateFlagsWhenHit (bcp.c:4509-4533), seen through a slower user's Conversion 2: a landed
    move records (move, attacker, type); a miss, a range that clears (opponent side: Spikes) and
    a type immunity clear it (the user's Conversion 2 then fails), a self-range foe move (Swords
    Dance) leaves a stale record. Hidden Power records its IV type (``moveType``), Weather Ball the
    weather's."""
    t, n = tables, names

    def types_after(
        foe, species="Geodude", picks=(), stale=True, hits=True, weather=None, user="Monferno"
    ):
        s = _c2_fight(t, n, foe=(foe,), foe_species=species, user=user)
        if stale:
            s = _record(s, 0, n.move("Growl"), 1, NORMAL)
        if weather is not None:
            s = with_weather(s, weather)
        out = _c2_out(hits=hits)
        for kind, pick in picks:
            out = out._replace(conv2=out.conv2.at[0, kind].set(pick))
        return s, _c2_go(t, n, s, foe, out=out)

    # A landed Tackle: recorded Normal, the user was hit and its Conversion 2 reads Normal (Rock).
    s, out = types_after("Tackle", picks=((NORMAL, ROCK),))
    assert _types(out) == [ROCK, ROCK] and _hp(out, 0) < int(s.hp[0, 0])
    # A miss clears the stale record: Conversion 2 fails.
    s, out = types_after("Tackle", hits=False, picks=((NORMAL, ROCK),))
    assert _types(out) == _types(s)
    # Swords Dance (RANGE_USER) leaves the stale Normal record: Conversion 2 works on it.
    s, out = types_after("Swords Dance", picks=((NORMAL, ROCK),))
    assert _types(out) == [ROCK, ROCK]
    # Spikes (RANGE_OPPONENT_SIDE) clears it.
    s, out = types_after("Spikes", picks=((NORMAL, ROCK),))
    assert _types(out) == _types(s)
    # Hidden Power (all IVs 15: Dark): the Dark rows, drawn as Steel.
    s, out = types_after("Hidden Power", stale=False, picks=((DARK, STEEL),))
    assert _types(out) == [STEEL, STEEL]
    # Weather Ball in rain: Water, drawn as Grass.
    s, out = types_after(
        "Weather Ball",
        "Castform",
        stale=False,
        weather=Weather.RAIN,
        picks=((WATER, 12),),
        user="Snorlax",
    )
    assert _types(out) == [12, 12]
    # An Earthquake into a Flying user is a type immunity: nothing is recorded.
    s = _c2_fight(t, n, foe=("Earthquake",), user="Pidgeot")
    s = _record(s, 0, n.move("Growl"), 1, NORMAL)
    out = _c2_go(t, n, s, "Earthquake", out=_c2_out(STEEL))
    assert _types(out) == _types(s)


def test_the_users_own_move_end_clears_its_record(tables, names):
    """ClearFlags (bcp.c:4390) clears ``conv2_move`` at the holder's own MoveEnd: a record from an
    earlier turn is gone once the user acted (a faster Swords Dance against a foe that leaves
    nothing), but its battler and type stay."""
    s = _c2_fight(tables, names, foe=("Swords Dance",), mine=("Swords Dance",), fast=True)
    s = _record(s, 0, names.move("Tackle"), 1, NORMAL)
    out = _c2_go(tables, names, s, "Swords Dance", me="Swords Dance")
    assert int(out.conv2_move[0]) == 0 and int(out.conv2_battler[0]) == 2


def test_conversion_2_fails_without_a_record_and_still_shows_the_attack_message(tables, names):
    """BtlCmd_TryConversion2 (:4775-4776) with no recorded move jumps to subscript_conversion_2.s:12
    (MOVE_STATUS_FAILED), past the subscript's own attack message; CheckMoveFailure
    (bcp.c:3311-3327) then runs subscript_missed.s, whose PrintAttackMessage (:5) sets
    SYSCTL_ATTACK_MESSAGE_SHOWN (battle_script.c:1423-1434), so UpdateFlagsWhenHit writes movePrev
    (bcp.c:4499-4503, review b2 #1): the types stay (the foe's Swords Dance, a self-range move,
    left no record and the user acts last), and movePrev is Conversion 2 failed or not."""
    t, n = tables, names
    s = _c2_fight(t, n, foe=("Swords Dance",))
    failed = _c2_go(t, n, s, "Swords Dance")
    assert _types(failed) == _types(s)
    assert int(failed.move_prev) == n.move("Conversion 2")
    s = _record(s, 0, n.move("Tackle"), 1, NORMAL)
    worked = _c2_go(t, n, s, "Swords Dance", out=_c2_out(ROCK))
    assert int(worked.move_prev) == n.move("Conversion 2")
    assert _types(worked) == [ROCK, ROCK]


def test_conversion_2_fails_for_multitype_and_for_a_locked_multi_turn_move(tables, names):
    """Multitype (:4770) fails it; a recorded multi-turn move whose attacker is still locked into
    it (:4778-4782: Fly's charge turn) fails it, and an attacker that is not locked (Fly's strike
    done) lets it work (``Move_IsMultiTurn``, battle_lib.c:2975-2992)."""
    t, n = tables, names
    arceus = make_state(
        t, [_spec(n, "Arceus", 50, ("Conversion 2",))], [_spec(n, "Geodude", 50, ("Tackle",))]
    )
    arceus = _record(arceus, 0, n.move("Tackle"), 1, NORMAL)
    arceus = arceus._replace(stats=arceus.stats.at[0, 0, SPE].set(1))
    out = _c2_go(t, n, arceus, "Tackle", out=_c2_out(ROCK))
    assert _types(out) == _types(arceus)
    fly = n.move("Fly")
    s = _c2_fight(t, n, foe=("Swords Dance",), fast=True)
    s = _record(s, 0, fly, 1, FLYING)
    locked = s._replace(locked_move=s.locked_move.at[1].set(fly))
    held = _c2_go(t, n, locked, "Swords Dance", out=_c2_out(ROCK, FLYING))
    assert _types(held) == _types(s)
    free = _c2_go(t, n, s, "Swords Dance", out=_c2_out(ROCK, FLYING))
    assert _types(free) == [ROCK, ROCK]


def test_a_foes_charge_turn_records_and_locks(tables, names):
    """Fly's charge turn (a move that got past BeforeMove without MOVE_STATUS_NO_EFFECTS) records
    Fly for its target, and the foe is locked into it: the slower user's Conversion 2 fails."""
    t, n = tables, names
    s = _c2_fight(t, n, foe=("Fly",), foe_species="Pidgeot")
    out = _c2_go(t, n, s, "Fly", out=_c2_out(ROCK, FLYING))
    assert int(out.conv2_move[0]) == 0  # the user's own MoveEnd cleared it
    assert int(out.conv2_battler[0]) == 2 and int(out.conv2_type[0]) == FLYING
    assert _types(out) == _types(s)
    assert int(out.locked_move[1]) == n.move("Fly")


def test_the_types_last_until_the_battler_leaves(tables, names):
    """The new types are the battler's (``BattleMon`` types) until it leaves the field: after a
    switch out the party Pokemon has its own types again."""
    t, n = tables, names
    s = make_state(
        t,
        [_spec(n, "Monferno", 50, ("Conversion 2",)), _spec(n, "Bidoof", 50, ("Tackle",))],
        [_spec(n, "Geodude", 50, ("Tackle",))],
    )
    s = s._replace(ability=s.ability.at[:, :].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(1))
    own = _types(s)
    after = _c2_go(t, n, s, out=_c2_out(ROCK))
    assert _types(after) == [ROCK, ROCK]
    acts = jnp.asarray([4 + 1, n.move_slot(after, 1, "Tackle")], jnp.int32)
    away = assert_matches_reference(t, after, acts, _c2_out())
    assert int(away.active[0]) == 1 and _types(away) == own


def test_a_faster_user_reads_the_turn_start_record_and_a_slower_one_the_foes_move(tables, names):
    """The enumeration: ``outcome_laws`` makes ``conv2[user, t]`` relevant for the recorded type
    ``t`` the user meets: the turn-start record when it moves first, the type of the foe's move
    (which overwrites the record) when the foe moves first. The enumerated values carry the exact
    law."""
    from nuzline.engine.enumerate import _conv2, enumerate_outcomes, outcome_laws

    t, n = tables, names
    for fast, kind in ((True, NORMAL), (False, FIRE)):
        s = _c2_fight(t, n, foe=("Ember",), fast=fast, foe_species="Monferno")
        s = _record(s, 0, n.move("Tackle"), 1, NORMAL)
        acts = A("Conversion 2", "Ember", s, n)
        laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(t, s, acts))
        relevant = np.asarray(laws.relevant.conv2)
        assert relevant[0].nonzero()[0].tolist() == [kind], fast
        assert not relevant[1].any()
        outcomes, probs = enumerate_outcomes(laws)
        values, weights, den = _conv2(laws, 0, kind)
        assert abs(probs.sum() - 1) < 1e-12
        got = {}
        for row, p in zip(np.asarray(outcomes.conv2)[:, 0, kind], probs, strict=True):
            got[int(row)] = got.get(int(row), 0.0) + float(p)
        for v, w in zip(values, weights, strict=True):
            assert abs(got[v] - w / den) < 1e-9


def test_the_draw_is_an_exact_integer_pick_of_the_law(tables, names):
    """Review b2 #3: the sampler (``outcome._conv2_draw``) draws one integer in
    ``[0, CONV2_DRAWS)`` per battler and recorded type and ``damage.conv2_pick`` takes the eligible
    type of rank ``value % count``. ``CONV2_DRAWS`` is a multiple of every count 1-18, so each
    eligible type gets exactly ``CONV2_DRAWS / count`` values: the enumeration's law
    (``_conv2``) exactly. Every sampled pick is one of the law's values."""
    import math

    from nuzline.engine.damage import CONV2_DRAWS, conv2_pick
    from nuzline.engine.outcome import _conv2_draw

    assert all(CONV2_DRAWS % k == 0 for k in range(1, 19)) and CONV2_DRAWS == math.lcm(
        *range(1, 19)
    )
    t, n = tables, names
    s = _c2_fight(t, n, user="Geodude", foe_species="Monferno")
    pick = jax.jit(jax.vmap(lambda v: conv2_pick(t, s, 0, NORMAL, v)))
    first = np.asarray(pick(jnp.arange(4, dtype=jnp.int32))).tolist()
    assert first == [GHOST, STEEL, GHOST, STEEL]  # rank value % 2 in type order
    top = np.asarray(pick(jnp.asarray([CONV2_DRAWS - 2, CONV2_DRAWS - 1], jnp.int32))).tolist()
    assert top == [GHOST, STEEL]
    # No eligible row (the ??? type): 0, the move fails.
    assert int(conv2_pick(t, s, 0, 9, 7)) == 0
    drawn = np.asarray(jax.jit(_conv2_draw)(jax.random.PRNGKey(0), t, s))
    assert int(drawn[0, NORMAL]) in (STEEL, GHOST)
    assert int(drawn[0, 9]) == 0


# --- #100 Spite (E3-B2-4) ---------------------------------------------------------------------
#
# effect_script_0100.s -> subscript_reduce_target_pp.s -> BtlCmd_TrySpite (battle_script.c:4990-
# 5016): the slot of the target's movePrevByBattler (``last_move``) loses min(4, ppCur); no last
# move, a move in no slot (Struggle) or a slot at 0 PP is MOVE_STATUS_FAILED
# (subscript_reduce_target_pp.s:12-14), which still shows the attack message (subscript_missed.s:5).
# No random draw.


def _spite_fight(t, n, foe=("Tackle", "Growl"), last="Tackle", fast=True, foe_item=""):
    """A level 50 Monferno (Spite, Swords Dance) against a level 50 Geodude (``foe``) whose
    ``last_move`` is ``last`` ('' = none); the Monferno is faster unless not ``fast``."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 50, ("Spite", "Swords Dance"))],
        [_spec(n, "Geodude", 50, foe, foe_item)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500 if fast else 1))
    if last:
        s = s._replace(last_move=s.last_move.at[1].set(n.move(last)))
    return s


def _pp(s, side=1):
    return np.asarray(s.pp[side, 0]).tolist()[:2]


def _spite(t, n, s, foe="Growl", out=None):
    return assert_matches_reference(
        t, s, A("Spite", foe, s, n), out if out is not None else _hits()
    )


def test_spite_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 100 (effect_script_0100.s) is EffectId.SPITE in both games; Spite (move 180) is the
    only move with that effect; the effect is a neutral, non-damaging one of the enumeration."""
    from nuzline.engine.enumerate import INPUT_NEUTRAL_EFFECTS

    assert DECOMP_EFFECT_TO_ID[100] == (EffectId.SPITE, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move("Spite")]) == EffectId.SPITE
        assert int((np.asarray(t.move_effect) == EffectId.SPITE).sum()) == 1
    assert EffectId.SPITE in INPUT_NEUTRAL_EFFECTS
    assert EffectId.SPITE in NON_TABLE_DAMAGE_EFFECTS


def test_spite_takes_four_pp_from_the_slot_of_the_targets_last_move(tables, names):
    """BtlCmd_TrySpite (:5000-5007): the slot holding movePrevByBattler (Tackle, slot 0) loses 4;
    the other slot is untouched by it (the foe's own Growl spends its usual 1)."""
    s = _spite_fight(tables, names)
    before = _pp(s)
    out = _spite(tables, names, s)
    assert _pp(out) == [before[0] - 4, before[1] - 1]


def test_spite_takes_only_the_pp_left(tables, names):
    """dec = min(4, ppCur) (:5000-5003): 1 or 3 PP left empties the slot, 4 empties it, 5 leaves
    1; a slot already at 0 fails it (:4996-4998) and stays at 0."""
    t, n = tables, names
    s = _spite_fight(t, n)
    for left in (1, 3, 4, 5):
        low = s._replace(pp=s.pp.at[1, 0, 0].set(left))
        assert _pp(_spite(t, n, low))[0] == max(left - 4, 0)
    empty = s._replace(pp=s.pp.at[1, 0, 0].set(0))
    assert _pp(_spite(t, n, empty))[0] == 0


def test_spite_fails_without_a_usable_last_move_and_still_shows_the_attack_message(tables, names):
    """No last move (:4995, :5011-5013), Struggle (in no slot) or a move the foe holds in no slot
    (Battler_SlotForMove finds none, :4996) fail it: no PP changes. The failure skips only the
    subscript's attack message (subscript_reduce_target_pp.s:6); subscript_missed.s:5 then shows
    it (bcp.c:3311-3327, review b2 #1), so movePrev is Spite behind a slower user's sleeping foe
    (whose own failed move zeroed its last move) as after the foe's Growl."""
    t, n = tables, names
    s = _spite_fight(t, n, fast=False)
    worked = _spite(t, n, s)
    assert int(worked.move_prev) == n.move("Spite")
    sleeper = s.with_status(1, 0, Status.SLEEP, 5)
    out = _spite(t, n, sleeper)
    assert _pp(out) == _pp(sleeper) and int(out.move_prev) == n.move("Spite")
    # A faster user meets the turn-start last move: none, Struggle, a move in no slot, 0 PP left.
    fast = _spite_fight(t, n).with_status(1, 0, Status.SLEEP, 5)
    assert _pp(_spite(t, n, fast))[0] == _pp(fast)[0] - 4
    for last in (0, MOVE_STRUGGLE, n.move("Rock Throw")):
        case = fast._replace(last_move=fast.last_move.at[1].set(last))
        assert _pp(_spite(t, n, case)) == _pp(case)
    empty = fast._replace(pp=fast.pp.at[1, 0, 0].set(0))
    assert _pp(_spite(t, n, empty)) == _pp(empty)


def test_a_slower_spite_reads_the_move_the_foe_just_used(tables, names):
    """movePrevByBattler is written when the foe acts (bcp.c:3945-3949): behind a foe that Growls,
    Spite takes from Growl's slot (slot 1), not the turn-start last move (Tackle)."""
    s = _spite_fight(tables, names, fast=False)
    out = _spite(tables, names, s)
    before = _pp(s)
    assert _pp(out) == [before[0], before[1] - 1 - 4]


def test_a_miss_takes_nothing_and_a_substitute_does_not_stop_it(tables, names):
    """The direct ON_HIT runs unless NO_EFFECTS (battle_lib.c:1485-1491): a miss leaves the PP;
    a Substitute is not checked (:4990-5016), so it still takes the 4."""
    t, n = tables, names
    s = _spite_fight(t, n)
    miss = _hits()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    out = _spite(t, n, s, out=miss)
    assert _pp(out)[0] == _pp(s)[0]
    sub = s._replace(substitute_hp=s.substitute_hp.at[1].set(20))
    assert _pp(_spite(t, n, sub))[0] == _pp(s)[0] - 4


def test_a_faster_spite_empties_the_chosen_move_and_it_fails_with_no_pp(tables, names):
    """DecrementPP (bcp.c:2275-2286) finds 0 PP in the foe's chosen Tackle after a faster Spite
    emptied it: MOVE_STATUS_NO_PP, the move fails (the Monferno takes nothing); with PP to spare
    the same Tackle hits."""
    t, n = tables, names
    s = _spite_fight(t, n)
    low = s._replace(pp=s.pp.at[1, 0, 0].set(3))
    out = _spite(t, n, low, foe="Tackle")
    assert _pp(out)[0] == 0
    assert _hp(out, 0) == int(s.hp[0, 0])
    assert _hp(_spite(t, n, s, foe="Tackle"), 0) < int(s.hp[0, 0])


def test_a_leppa_berry_restores_the_emptied_slot_in_the_same_action(tables, names):
    """The after-move TriggerHeldItem on the defender (bcp.c:3694; battle_lib.c:4636-4653): the
    foe's Leppa Berry restores the first slot at 0 PP that Spite just emptied; the berry is spent."""
    t, n = tables, names
    s = _spite_fight(t, n, foe_item="Leppa Berry")
    s = s._replace(pp=s.pp.at[1, 0, 0].set(3))
    out = _spite(t, n, s)
    assert _pp(out)[0] == int(t.item_param[n.item("Leppa Berry"), 0])
    assert int(out.item[1, 0]) == 0


def test_a_mimicked_slot_loses_only_the_battlemons_pp(tables, names):
    """BattleMon_CopyToParty (:5007-5009) skips a Mimic-flagged slot (BtlIOCmd_UpdatePartyMon): the
    BattleMon's PP drops, the party's shadow of it stays."""
    t, n = tables, names
    s = _spite_fight(t, n)
    s = s._replace(
        mimic_slots=s.mimic_slots.at[1].set(1),
        shadow_moves=s.shadow_moves.at[1, 0].set(n.move("Rock Throw")),
        shadow_pp=s.shadow_pp.at[1, 0].set(15),
    )
    out = _spite(t, n, s)
    assert _pp(out)[0] == _pp(s)[0] - 4
    assert int(out.shadow_pp[1, 0]) == 15


# --- #178 Role Play (E3-B2-5) -----------------------------------------------------------------
#
# effect_script_0178.s:5-7 fails on the user's Multitype or held Griseous Orb, else runs
# subscript_copy_ability.s: it fails (:5-9, :31-32) on a miss or a semi-invulnerable defender, the
# defender's raw Wonder Guard, Multitype or no ability, the user's raw Multitype; else (:11-15) the
# user's truant becomes ``totalTurns & 1`` and its BattleMon ability the defender's raw one. No
# random draw, no Protect flag, no Substitute test.


def _rp(t, n, mine, foe, item="", foe_item=""):
    """Monferno (side 0, Role Play) holding ``item`` with ``mine`` against Geodude holding
    ``foe_item`` with ``foe``, at turn 1 (a Truant write of ``totalTurns & 1`` reads 1)."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 20, ("Role Play", "Tackle"), item)],
        [_spec(n, "Geodude", 40, ("Tackle",), foe_item)],
    )
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(1).at[1, 0, SPE].set(500))
    return s._replace(
        ability=s.ability.at[0, 0].set(mine).at[1, 0].set(foe),
        turn=jnp.asarray(1, s.turn.dtype),
    )


def _rpgo(t, n, s, out=None):
    return assert_matches_reference(
        t, s, A("Role Play", "Tackle", s, n), out if out is not None else _hits()
    )


def test_role_play_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 178 (effect_script_0178.s) is EffectId.ROLE_PLAY, a clause and a non-damaging effect
    of the enumeration, with no Protect flag."""
    assert DECOMP_EFFECT_TO_ID[178] == (EffectId.ROLE_PLAY, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        move = n.move("Role Play")
        assert int(t.move_effect[move]) == EffectId.ROLE_PLAY
        assert int(t.move_flags[move]) & MoveFlag.PROTECT == 0
    assert EffectId.ROLE_PLAY in INPUT_CLAUSE_EFFECTS
    assert EffectId.ROLE_PLAY in NON_TABLE_DAMAGE_EFFECTS


def test_role_play_copies_the_targets_ability_only_and_the_ai_learns_it(
    tables, names, kaizo_tables, kaizo_names
):
    """:14-15 copy the defender's ability onto the user, :11-13 write ``totalTurns & 1`` (1 at turn
    1) to the user's ``truant`` only; the defender keeps its ability and its truant. The AI is
    tagged with the user's new ability (battle_script.c:3401-3404)."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = _rp(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
        out = _rpgo(t, n, s)
        assert out.ability[:, 0].tolist() == [AbilityId.ROCK_HEAD, AbilityId.ROCK_HEAD]
        assert out.truant.tolist() == [1, 0]
        assert int(out.ai_ability_known[0]) == ABILITY_DECOMP_IDS[AbilityId.ROCK_HEAD]


def test_role_play_fails_on_its_listed_conditions_and_still_shows_the_attack_message(tables, names):
    """effect_script_0178.s:5-6 (the user's Multitype, a held Griseous Orb) and
    subscript_copy_ability.s:6-9 (the defender's Wonder Guard, Multitype, no ability; the user's
    Multitype): nothing is copied and truant stays. The failure (MOVE_STATUS_FAILED) reaches
    CheckMoveFailure's subscript_missed.s, whose PrintAttackMessage (:5) shows the move
    (bcp.c:3311-3327, review b2 #1): movePrev is Role Play, as on a success (the Monferno acts
    last here). The defender's Orb, the user's own Wonder Guard or no ability, and a Substitute
    do not stop it."""
    t, n = tables, names
    orb = n.item("Griseous Orb")
    base = _rp(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
    ok = _rpgo(t, n, base)
    rp = n.move("Role Play")
    assert int(ok.move_prev) == rp
    fails = {
        "wonder guard foe": base._replace(
            ability=base.ability.at[1, 0].set(AbilityId.WONDER_GUARD)
        ),
        "no ability foe": base._replace(ability=base.ability.at[1, 0].set(AbilityId.NONE)),
        "orb user": _rp(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD, item="Griseous Orb"),
    }
    for label, case in fails.items():
        out = _rpgo(t, n, case)
        assert out.ability[:, 0].tolist() == case.ability[:, 0].tolist(), label
        assert out.truant.tolist() == case.truant.tolist(), label
        assert int(out.move_prev) == rp, label
    arceus = as_species(t, base, 1, 0, n.species("Arceus"))
    out = _rpgo(t, n, arceus)  # the defender's Multitype
    assert int(out.ability[0, 0]) == int(arceus.ability[0, 0]) and int(out.move_prev) == rp
    mt_user = as_species(t, base, 0, 0, n.species("Arceus"))
    out = _rpgo(t, n, mt_user)  # the user's Multitype
    assert int(out.ability[0, 0]) == int(mt_user.ability[0, 0]) and int(out.move_prev) == rp
    for case in (
        base._replace(ability=base.ability.at[0, 0].set(AbilityId.WONDER_GUARD)),
        base._replace(ability=base.ability.at[0, 0].set(AbilityId.NONE)),
        base._replace(item=base.item.at[1, 0].set(orb)),
        base._replace(substitute_hp=base.substitute_hp.at[1].set(30)),
    ):
        out = _rpgo(t, n, case)
        assert int(out.ability[0, 0]) == AbilityId.ROCK_HEAD
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    # Accuracy 0: no roll, so a failing hit draw changes nothing (bcp.c:2901-2903).
    assert _rpgo(t, n, base, miss).ability[:, 0].tolist() == [AbilityId.ROCK_HEAD] * 2
    gone = base._replace(semi_invulnerable=base.semi_invulnerable.at[1].set(1))
    assert _rpgo(t, n, gone).ability[:, 0].tolist() == base.ability[:, 0].tolist()


def test_a_skill_swap_into_wonder_guard_fails_and_swaps_nothing(tables, names):
    """Review b2 #2: Skill Swap fails on a defender with Wonder Guard (subscript_exchange_abilities.s:7
    jumps to _156, MOVE_STATUS_FAILED) before :17-24 write the Truant parity and the abilities, so the
    Monferno keeps Blaze, the Shedinja keeps Wonder Guard and neither ``truant`` changes; against a
    Geodude the same Skill Swap exchanges the abilities. CheckMoveFailure sends the failure to
    subscript_missed.s with commandNext LOOP_FAINTED (battle_controller_player.c:3325-3328),
    skipping AfterMoveEffects. That skip has no observable effect on this path: its switch-in pass
    (bcp.c:3672) and held-item states (TriggerHeldItem :3684-3694, whose hold effects all sit in
    TriggerHeldItemOnStatus, battle_lib.c:4576 vs :4967; HELD_ITEM_STATUS :3730-3745) are repeated
    by MoveEnd's TriggerEffectOnSwitch (bcp.c:3973, battle_lib.c:4151-4165), and the others need a
    charge, an inflicted status, a hit or damage, which a failed Skill Swap does not make (:3641,
    :3665-3668, :3698-3702, :3710). Both boards are reachable: natural abilities, full HP, the
    faster Monferno by its own stats."""
    t, n = tables, names
    for foe, foe_move in (("Geodude", "Defense Curl"), ("Shedinja", "Harden")):
        s = make_state(
            t,
            [_spec(n, "Monferno", 40, ("Skill Swap",), "Sitrus Berry")],
            [_spec(n, foe, 20, (foe_move,))],
        )
        s = s._replace(turn=jnp.asarray(1, s.turn.dtype))  # :17-20 would write totalTurns & 1 = 1
        assert int(s.stats[0, 0, SPE]) > int(s.stats[1, 0, SPE])
        mine, theirs = int(s.ability[0, 0]), int(s.ability[1, 0])
        assert mine == AbilityId.BLAZE and theirs != AbilityId.NONE
        out = assert_matches_reference(t, s, A("Skill Swap", foe_move, s, n), _hits())
        got = [int(a) for a in out.ability[:, 0]]
        if foe == "Shedinja":
            assert theirs == AbilityId.WONDER_GUARD
            assert got == [mine, theirs]  # a swap would give [theirs, mine]
            assert out.truant.tolist() == [0, 0]
        else:
            assert got == [theirs, mine]
            assert out.truant.tolist() == [1, 1]
        assert int(out.item[0, 0]) == n.item("Sitrus Berry")  # sanity check: full HP, kept


def test_role_play_reads_the_raw_ability_through_gastro_acid(tables, names):
    """:14 reads BattleMon.ability: a Gastro Acid suppressed defender still gives its ability, and
    the user's own suppression bit stays (:14-15 write only the ability)."""
    t, n = tables, names
    s = _rp(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
    s = s._replace(ability_suppressed=s.ability_suppressed.at[1].set(1).at[0].set(1))
    out = _rpgo(t, n, s)
    assert int(out.ability[0, 0]) == AbilityId.ROCK_HEAD
    assert out.ability_suppressed.tolist() == [1, 1]


def test_role_play_gives_the_user_the_targets_entry_ability_once_at_the_move_end(tables, names):
    """The Intimidate the defender announced at its entry arrives with the user's flag clear:
    MoveEnd's TriggerEffectOnSwitch (bcp.c:3973) fires it, the foe's Attack -1; an Intimidate the
    user had announced already stays quiet."""
    t, n = tables, names
    s = _rp(t, n, AbilityId.BLAZE, AbilityId.INTIMIDATE)
    s = s._replace(intimidate_announced=jnp.asarray([0, 1], s.intimidate_announced.dtype))
    out = _rpgo(t, n, s)
    assert int(out.stages[1, Stat.ATK]) == -1 and int(out.stages[0, Stat.ATK]) == 0
    assert out.intimidate_announced.tolist() == [1, 1]
    spent = s._replace(intimidate_announced=jnp.asarray([1, 1], s.intimidate_announced.dtype))
    assert int(_rpgo(t, n, spent).stages[1, Stat.ATK]) == 0


def test_the_copy_ends_with_the_battler(tables, names):
    """The write is the BattleMon's (battle_lib.c:124): ``_restore_ability`` puts the party
    Pokemon's own ability back when it leaves."""
    t, n = tables, names
    two = make_state(
        t,
        [_spec(n, "Monferno", 20, ("Role Play",)), _spec(n, "Monferno", 20, ("Scratch",))],
        [_spec(n, "Geodude", 40, ("Tackle",))],
    )
    two = two._replace(ability=two.ability.at[1, 0].set(AbilityId.ROCK_HEAD))
    own = int(two.ability[0, 0])
    out = _rpgo(t, n, two)
    assert int(out.ability[0, 0]) == AbilityId.ROCK_HEAD
    switch = jnp.array([4 + 1, n.move_slot(out, 1, "Tackle")], jnp.int32)
    out2 = assert_matches_reference(t, out, switch, empty_outcome())
    assert int(out2.ability[0, 0]) == own


def test_role_play_is_seen_by_hit_thresholds_and_armor(tables, names):
    """A faster foe's Role Play gives the foe the side's ability before the side's check
    (bcp.c:2876-2947): a Sand Veil copy in a sandstorm takes Rock Throw's 90 to 72, so the law has
    a second class beside the turn start's; a Role Play user may hold the side's Battle Armor at
    the side's CalcCrit (``outcome.armored``)."""
    from nuzline.engine.outcome import armored
    from tests.engine.test_accuracy_classes import hit_law

    t, n = tables, names
    s = roark_state(t, n, player_moves=["Rock Throw"], opp_moves=["Role Play"], opp_faster=True)
    mon = int(s.active[0])
    s = with_weather(
        s._replace(ability=s.ability.at[0, mon].set(AbilityId.SAND_VEIL)), Weather.SANDSTORM
    )
    law = hit_law(t, s, A("Rock Throw", "Role Play", s, n), 0)
    assert law == {0: 10, 1: 72, 73: 18}  # a roll up to 72 hits either way; 73-90 only at 90
    s2 = s._replace(ability=s.ability.at[0, mon].set(AbilityId.BATTLE_ARMOR))
    acts = A("Rock Throw", "Role Play", s2, n)
    assert bool(armored(t, s2, acts)[0]) is False


# --- Magic Coat (#183, E3-B2-6) ----------------------------------------------------------------
# effect_script_0183.s -> BtlCmd_TrySetMagicCoat (battle_script.c:6459-6478); the reflection is
# BattleControllerPlayer_MoveStolen (battle_controller_player.c:3064-3085) at BeforeMove's
# CHECK_STOLEN, then BtlCmd_MagicCoat (battle_script.c:6487-6513): the mover's move runs with the
# Magic Coat user as attacker (``step._execute_actions``' reflect phase, ``MODE_REFLECT``).


def _coat(t, n, foe="Toxic", foe_moves=None, mine=("Magic Coat", "Tackle"), item="", foe_item=""):
    """Monferno (side 0, Magic Coat) against a faster Geodude (``foe``), no abilities: priority +4
    puts the Monferno first whatever the speeds."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 20, mine, item)],
        [_spec(n, "Geodude", 40, foe_moves or (foe,), foe_item)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    return s._replace(stats=s.stats.at[0, 0, SPE].set(1).at[1, 0, SPE].set(500))


def _cgo(t, n, s, foe="Toxic", out=None, me="Magic Coat"):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def test_magic_coat_is_mapped_and_flags_the_listed_moves(tables, names, kaizo_tables, kaizo_names):
    """Effect 183 is EffectId.MAGIC_COAT (priority +4, no flags of its own); the moves with
    MOVE_FLAG_CAN_MAGIC_COAT carry ``MoveFlag.MAGIC_COAT``: Gastro Acid and Worry Seed in both
    games, Kaizo's Disable, Encore, Spikes, Spite, Stealth Rock, Torment and Toxic Spikes only
    there (moves.json)."""
    assert DECOMP_EFFECT_TO_ID[183] == (EffectId.MAGIC_COAT, (0, 0, 0))
    both = ("Toxic", "Will-O-Wisp", "Gastro Acid", "Worry Seed", "Growl", "Leech Seed")
    kaizo_only = ("Disable", "Encore", "Spikes", "Spite", "Stealth Rock", "Torment", "Toxic Spikes")
    for t, nn, kaizo in ((tables, names, False), (kaizo_tables, kaizo_names, True)):
        coat = nn.move("Magic Coat")
        assert int(t.move_effect[coat]) == EffectId.MAGIC_COAT
        assert (
            int(t.move_flags[coat]) & MoveFlag.MAGIC_COAT == 0 and int(t.move_priority[coat]) == 4
        )
        for move in both:
            assert int(t.move_flags[nn.move(move)]) & MoveFlag.MAGIC_COAT, move
        for move in kaizo_only:
            flagged = bool(int(t.move_flags[nn.move(move)]) & MoveFlag.MAGIC_COAT)
            assert flagged == kaizo, move
        assert not int(t.move_flags[nn.move("Tackle")]) & MoveFlag.MAGIC_COAT


def test_magic_coat_reflects_a_move_aimed_at_its_user(tables, names):
    """bcp.c:3064-3085: Toxic aimed at the Magic Coat user, who went first, runs with the user as
    attacker (Toxic's accuracy, effect and target): the Geodude is badly poisoned, the Monferno
    is not. The flag is consumed (:3071) and gone at the turn's end; the Geodude's Toxic cost its
    PP (the first pass's DecrementPP) and is its last move (:3073-3075), the battle's movePrev."""
    t, n = tables, names
    toxic = n.move("Toxic")
    s = _coat(t, n)
    out = _cgo(t, n, s)
    assert int(out.status[0, 0]) == Status.NONE and int(out.status[1, 0]) == Status.TOXIC
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
    assert int(out.last_move[1]) == toxic and int(out.move_prev) == toxic
    assert out.magic_coat.tolist() == [0, 0]
    # Without the Magic Coat the Toxic poisons the Monferno.
    plain = _cgo(t, n, s, me="Tackle")
    assert int(plain.status[0, 0]) == Status.TOXIC and int(plain.status[1, 0]) == Status.NONE


def _misses(side):
    """Every roll class hits (``_hits``) except ``side``'s own, which misses."""
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8).at[side].set(0))


def test_magic_coat_reflects_only_flagged_moves_and_only_when_it_goes_first(tables, names):
    """MoveStolen tests MOVE_FLAG_CAN_MAGIC_COAT (:3070): a Tackle hits the user; a switch or a
    move of higher priority leaves nothing to reflect; the flag never outlives the turn."""
    t, n = tables, names
    s = _coat(t, n, "Tackle")
    out = _cgo(t, n, s, "Tackle")
    assert int(out.hp[0, 0]) < int(s.hp[0, 0]) and int(out.hp[1, 0]) == int(s.hp[1, 0])
    assert out.magic_coat.tolist() == [0, 0]
    # TrySetMagicCoat fails when no other battler's action is pending (:6471): the Helping Hand
    # (priority +5) goes first, so the user acts last and a Toxic could not be reflected.
    last = _coat(t, n, "Helping Hand")
    assert _cgo(t, n, last, "Helping Hand").magic_coat.tolist() == [0, 0]
    two = make_state(
        t,
        [_spec(n, "Monferno", 20, ("Magic Coat",))],
        [_spec(n, "Geodude", 40, ("Toxic",)), _spec(n, "Geodude", 40, ("Tackle",))],
    )
    out = assert_matches_reference(t, two, jnp.array([0, 4 + 1], jnp.int32), _hits())
    assert int(out.active[1]) == 1 and out.magic_coat.tolist() == [0, 0]


def test_a_reflected_move_rolls_the_holders_accuracy_not_the_movers(tables, names):
    """The move runs with the holder as attacker (BtlCmd_MagicCoat swaps them, :6487-6513): the
    holder's roll decides it, whatever the mover would have rolled."""
    t, n = tables, names
    s = _coat(t, n)
    assert int(_cgo(t, n, s, out=_misses(0)).status[1, 0]) == Status.NONE
    assert int(_cgo(t, n, s, out=_misses(1)).status[1, 0]) == Status.TOXIC


def test_a_reflected_move_meets_the_movers_defences(tables, names):
    """The reflected move is an ordinary move against the mover: its Substitute stops Toxic."""
    t, n = tables, names
    s = _coat(t, n)
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(30))
    out = _cgo(t, n, s)
    assert int(out.status[1, 0]) == Status.NONE and int(out.status[0, 0]) == Status.NONE


def test_a_reflection_costs_the_holder_a_pp_when_the_mover_has_pressure(tables, names):
    """BattleSystem_DecPPForPressure(attacker = the holder, defender = the mover) (bcp.c:3083,
    battle_lib.c:7390-7397): one PP from the holder's Magic Coat slot while the MOVER has
    Pressure; the mover's own Toxic costs 1 (the holder has none), and a holder's Pressure costs
    the mover's Toxic 2 and the holder nothing more."""
    t, n = tables, names
    s = _coat(t, n, mine=("Tackle", "Magic Coat"))
    pressure = s._replace(ability=s.ability.at[1, 0].set(AbilityId.PRESSURE))
    out = _cgo(t, n, pressure, me="Magic Coat")
    assert int(out.pp[0, 0, 1]) == int(s.pp[0, 0, 1]) - 2 and int(out.pp[0, 0, 0]) == int(
        s.pp[0, 0, 0]
    )
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1
    base = _cgo(t, n, s, me="Magic Coat")
    assert int(base.pp[0, 0, 1]) == int(s.pp[0, 0, 1]) - 1
    holder = s._replace(ability=s.ability.at[0, 0].set(AbilityId.PRESSURE))
    out = _cgo(t, n, holder, me="Magic Coat")
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 2
    assert int(out.pp[0, 0, 1]) == int(s.pp[0, 0, 1]) - 1


def test_a_reflection_writes_no_move_records_but_the_movers_own(tables, names):
    """SYSCTL_REUSE_LAST_MOVE (bcp.c:3077) keeps UpdateFlagsWhenHit from writing the holder's
    moveHit, moveCopied and conversion2 (:4481, :4498, :4509); the mover's movePrevByBattler and
    movePrev are written (:3073-3075) and its Choice lock stays. Unreflected, Toxic writes all."""
    t, n = tables, names
    toxic = n.move("Toxic")
    s = _coat(t, n, foe_item="Choice Band")
    out = _cgo(t, n, s)
    assert int(out.move_hit[0]) == 0 and int(out.move_copied[0]) == 0
    assert not np.asarray(out.move_copied_hit[0]).any() and int(out.conv2_move[0]) == 0
    assert int(out.last_move[1]) == toxic and int(out.choice_lock[1]) == toxic
    plain = _cgo(t, n, s, me="Tackle")
    assert int(plain.move_copied[0]) == toxic and int(plain.move_copied_hit[0, 1]) == toxic


def test_the_metronome_count_follows_the_passes(tables, names):
    """BeforeMove's end (bcp.c:3221) runs in the holder's pass only: the Metronome item holder
    counts the reflected move (moveTemp), the mover (whose first pass returned) counts nothing."""
    t, n = tables, names
    toxic = n.move("Toxic")
    s = _coat(t, n, item="Metronome", foe_item="Metronome")
    out = _cgo(t, n, s)
    assert int(out.metronome_move[0]) == toxic and int(out.metronome_move[1]) == 0


def test_magic_coat_reflects_the_effects_that_land_on_the_mover(tables, names):
    """Both games' flagged moves land on the MOVER when reflected (the holder is the attacker):
    Growl (a spread range: the defender is the mover, :6499-6500), Leech Seed, Swagger, Worry Seed
    (Insomnia), Gastro Acid (E3-2-2, E3-2-6: Owed 1)."""
    t, n = tables, names
    out = _cgo(t, n, _coat(t, n, "Growl"), "Growl")
    assert int(out.stages[1, Stat.ATK]) == -1 and int(out.stages[0, Stat.ATK]) == 0
    out = _cgo(t, n, _coat(t, n, "Leech Seed"), "Leech Seed")
    assert int(out.leech_seed[1]) != 0 and int(out.leech_seed[0]) == 0
    out = _cgo(t, n, _coat(t, n, "Swagger"), "Swagger")
    assert int(out.stages[1, Stat.ATK]) == 2 and int(out.confusion[1]) > 0
    assert int(out.stages[0, Stat.ATK]) == 0 and int(out.confusion[0]) == 0
    out = _cgo(t, n, _coat(t, n, "Worry Seed"), "Worry Seed")
    assert int(out.ability[1, 0]) == AbilityId.INSOMNIA and int(out.ability[0, 0]) == AbilityId.NONE
    out = _cgo(t, n, _coat(t, n, "Gastro Acid"), "Gastro Acid")
    assert int(out.ability_suppressed[1]) == 1 and int(out.ability_suppressed[0]) == 0


def test_magic_coats_doubles_paths_follow_the_decomp(tables, names):
    """Review b2 #9. BtlCmd_MagicCoat (battle_script.c:6487-6507): the holder pass's defender is
    the living Follow Me user of the MOVER's side (``target`` is read while the mover is the
    attacker), else the mover; the range branch and the Lightning Rod / Storm Drain branch give
    the mover (``targets.magic_coat_defender``; reference ``_follow_me_user``). At B = 2 it is the
    mover. BtlCmd_TrySetMagicCoat fails on ``waitingBattlers == 1`` (bs.c:6470-6471), the count
    over every battler of BranchActions (bcp.c:881-886): the reference's ``_waiting`` counts the
    battlers whose action is not MOVE_END."""
    from nuzline.engine.format import side_of
    from nuzline.engine.targets import NO_DEFENDER, follow_me_user, magic_coat_defender
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _follow_me_user, _Turn, _waiting
    from tests.engine.d2_states import four, with_hp

    t, n = tables, names
    s = four(t, n)
    assert int(magic_coat_defender(s, 1)) == 1  # no Follow Me: the mover
    fm = s._replace(ts_follow_me=s.ts_follow_me.at[side_of(1)].set(1 + 3))
    assert int(follow_me_user(fm, side_of(1))) == 3
    assert int(magic_coat_defender(fm, 1)) == 3  # the mover's partner draws it
    assert int(magic_coat_defender(fm, 0)) == 0  # the other side's Follow Me does not
    assert _follow_me_user(to_ref(fm), side_of(1)) == 3
    gone = with_hp(fm, 3, 0)
    assert int(follow_me_user(gone, side_of(1))) == NO_DEFENDER
    assert int(magic_coat_defender(gone, 1)) == 1
    assert _follow_me_user(to_ref(gone), side_of(1)) is None
    single = roark_state(t, n, player_moves=["Magic Coat"], opp_moves=["Growl"])
    assert int(magic_coat_defender(single, 1)) == 1 and int(magic_coat_defender(single, 0)) == 0
    turn = _Turn(t, empty_outcome(), entry_turn=1, battler_count=4)
    turn.done = [False, True, False, True]
    assert _waiting(turn) == 2
    turn.done = [True, True, False, True]
    assert _waiting(turn) == 1


@pytest.mark.slow  # 49 s, peak 8.17 GB, +4 GB over a loaded step (fast tier: 60 s, 2 GB above a loaded step); run on the pod
def test_kaizos_reflectable_moves_are_reflected_too(kaizo_tables, kaizo_names):
    """Owed 1 (E3-1-3, E3-2-1) and E3-B2-4's ruling 1: Kaizo's Disable, Encore, Spite, Torment and
    the hazards carry CAN_MAGIC_COAT and meet the Magic Coat with the holder as attacker (the
    mover is the defender, the hazards' range included): the mover's Disable disables its own
    Disable (bs.c:4581-4600), its Encore fails (``Move_CanBeEncored``, lib.c:7221-7231: Encore is
    its own banned move), its Spite cuts its own Spite's PP by 4, Torment torments it and the
    hazards land on its own side."""
    t, n = kaizo_tables, kaizo_names
    s = _coat(t, n, foe_moves=("Disable", "Tackle"))
    out = _cgo(t, n, s, "Disable")
    assert int(out.disabled_move[1]) == n.move("Disable") and int(out.disabled_move[0]) == 0
    s = _coat(t, n, foe_moves=("Encore", "Tackle"))
    out = _cgo(t, n, s, "Encore")
    assert out.encore_move.tolist() == [0, 0]
    s = _coat(t, n, foe_moves=("Spite", "Tackle"))
    s = s._replace(pp=s.pp.at[1, 0, 0].set(10))  # Kaizo's Spite has 3 PP
    out = _cgo(t, n, s, "Spite")
    assert int(out.pp[1, 0, 0]) == 10 - 1 - 4
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
    out = _cgo(t, n, _coat(t, n, foe_moves=("Torment", "Tackle")), "Torment")
    assert out.torment.tolist() == [0, 1]
    for move, field in (
        ("Spikes", "spikes"),
        ("Stealth Rock", "stealth_rock"),
        ("Toxic Spikes", "toxic_spikes"),
    ):
        out = _cgo(t, n, _coat(t, n, foe_moves=(move, "Tackle")), move)
        assert int(getattr(out, field)[1]) == 1 and int(getattr(out, field)[0]) == 0, move
    # Vanilla has none of them flagged: the Spite of the foe is no reflected move there.


# --- Lucky Chant (#240, E3-B2-7) ---------------------------------------------------------------
# effect_script_0240.s: fails with SIDE_CONDITION_LUCKY_CHANT on the user's side (:5, :9-11), else
# an indirect ON_HIT subscript_lucky_chant_start.s sets the counter (bits 12-14) to 5 (:5-6); the
# field-condition pass counts it down (battle_controller_player.c:1062-1083); the defender's side
# conditions switch the crit off (BattleSystem_CalcCriticalMulti, battle_lib.c:7137, after the
# draw :7132; battle_script.c:2158-2163).


def _lc(t, n, fast=False, foe="Tackle"):
    """Monferno (side 0, Lucky Chant, Tackle) against a Geodude (``foe``), no abilities; the
    Monferno is the faster one only if ``fast`` (vanilla Lucky Chant has priority 0)."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 30, ("Lucky Chant", "Tackle"))],
        [_spec(n, "Geodude", 20, (foe,))],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    fast_spe, slow_spe = (500, 1) if fast else (1, 500)
    return s._replace(stats=s.stats.at[0, 0, SPE].set(fast_spe).at[1, 0, SPE].set(slow_spe))


def _crits(*sides):
    """Every hit lands; the crit draw of each side in ``sides`` is 1 (a crit at any odds)."""
    crit = jnp.zeros((2, MAX_HITS), jnp.int8)
    for side in sides:
        crit = crit.at[side].set(1)
    return empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8), crit=crit)


def _lcgo(t, n, s, me="Lucky Chant", out=None):
    return assert_matches_reference(
        t, s, A(me, "Tackle", s, n), out if out is not None else _hits()
    )


def _lost(s, out, side):
    return int(s.hp[side, 0]) - int(out.hp[side, 0])


def test_lucky_chant_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 240 (effect_script_0240.s) is EffectId.LUCKY_CHANT in both games and Lucky Chant
    (move 381) the only move with it; priority 0 in vanilla, +2 in Kaizo (moves.json); the effect
    is a clause (the foe's earlier Lucky Chant blocks a later crit) and non-damaging."""
    assert DECOMP_EFFECT_TO_ID[240] == (EffectId.LUCKY_CHANT, (0, 0, 0))
    for t, nn, priority in ((tables, names, 0), (kaizo_tables, kaizo_names, 2)):
        move = nn.move("Lucky Chant")
        assert int(t.move_effect[move]) == EffectId.LUCKY_CHANT
        assert int((np.asarray(t.move_effect) == EffectId.LUCKY_CHANT).sum()) == 1
        assert int(t.move_priority[move]) == priority
        assert int(t.move_accuracy[move]) == 0
    assert EffectId.LUCKY_CHANT in INPUT_CLAUSE_EFFECTS
    assert EffectId.LUCKY_CHANT in NON_TABLE_DAMAGE_EFFECTS


def test_lucky_chant_sets_five_turns_on_the_users_side_only(tables, names):
    """subscript_lucky_chant_start.s:5-6 ORs 0x5000: the counter is 5 on the user's side, and the
    turn's field-condition pass counts it down once (4 left); the foe's side has none."""
    t, n = tables, names
    out = _lcgo(t, n, _lc(t, n))
    assert out.lucky_chant.tolist() == [4, 0]


def test_lucky_chant_fails_while_it_is_up_and_counts_down(tables, names):
    """effect_script_0240.s:5, :9-11: with the condition up the move fails and sets nothing (2 left
    becomes 1, not 4); the pass ends a counter of 1 at the turn's end (:1069)."""
    t, n = tables, names
    s = _lc(t, n)
    up = s._replace(lucky_chant=s.lucky_chant.at[0].set(2))
    assert _lcgo(t, n, up).lucky_chant.tolist() == [1, 0]
    last = s._replace(lucky_chant=s.lucky_chant.at[0].set(1))
    assert _lcgo(t, n, last).lucky_chant.tolist() == [0, 0]
    # The foe's own side holding it does not stop the user's: it sets 5 (4 after the pass).
    other = s._replace(lucky_chant=s.lucky_chant.at[1].set(3))
    assert _lcgo(t, n, other).lucky_chant.tolist() == [4, 2]


def test_a_faster_lucky_chant_blocks_the_foes_crit_the_same_turn(tables, names):
    """The Monferno's Lucky Chant goes first, so the Geodude's CalcCrit meets the condition: a crit
    draw (battle_lib.c:7132, spent either way) does nothing and the Tackle deals what a no-crit
    Tackle does; a slower Lucky Chant comes too late and the crit lands (more damage)."""
    t, n = tables, names
    fast = _lc(t, n, fast=True)
    blocked = _lcgo(t, n, fast, out=_crits(1))
    plain = _lcgo(t, n, fast, out=_crits())
    assert _lost(fast, blocked, 0) == _lost(fast, plain, 0) > 0
    slow = _lc(t, n, fast=False)
    late = _lcgo(t, n, slow, out=_crits(1))
    assert _lost(slow, late, 0) > _lost(slow, _lcgo(t, n, slow, out=_crits()), 0)
    assert late.lucky_chant.tolist() == [4, 0]  # it still stands for the next turns


def test_lucky_chant_protects_its_side_only_and_through_its_last_turn(tables, names):
    """Crits against the Monferno's side are off while its counter is up, including the turn the
    pass ends it (a counter of 1 counts down after the actions); the Monferno's own crit against
    the unprotected Geodude still lands; a Tackle at an unprotected Monferno crits."""
    t, n = tables, names
    s = _lc(t, n, fast=True)
    for left in (1, 3, 5):
        up = s._replace(lucky_chant=s.lucky_chant.at[0].set(left))
        crit = _lcgo(t, n, up, me="Tackle", out=_crits(0, 1))
        plain = _lcgo(t, n, up, me="Tackle", out=_crits(0))
        assert _lost(up, crit, 0) == _lost(up, plain, 0) > 0
        assert _lost(up, crit, 1) > _lost(up, _lcgo(t, n, up, me="Tackle", out=_crits(1)), 1)
    bare = _lcgo(t, n, s, me="Tackle", out=_crits(1))
    assert _lost(s, bare, 0) > _lost(s, _lcgo(t, n, s, me="Tackle", out=_crits()), 0)


def test_kaizos_priority_puts_lucky_chant_before_a_faster_foe(kaizo_tables, kaizo_names):
    """Kaizo's +2 priority: a slower Lucky Chant still goes first and blocks the faster foe's crit
    this turn (the same-turn case above, by priority)."""
    t, n = kaizo_tables, kaizo_names
    s = _lc(t, n, fast=False)
    blocked = _lcgo(t, n, s, out=_crits(1))
    assert _lost(s, blocked, 0) == _lost(s, _lcgo(t, n, s, out=_crits()), 0) > 0
    assert blocked.lucky_chant.tolist() == [4, 0]


def test_a_lucky_chant_in_the_turn_changes_the_damage_inputs_of_the_foes_hit(tables, names):
    """enumerate.damage_inputs_can_change: the foe's Lucky Chant before the side's hit can change
    that hit's damage (a crit becomes none), so the turn-start damage classes are not used for it;
    not when the side's foe uses another move, or when its side has the condition at the turn
    start (the turn-start table has no crit then, outcome_laws' ``no_crit``)."""
    from nuzline.engine.enumerate import damage_inputs_can_change, outcome_laws

    t, n = tables, names
    s = _lc(t, n, fast=True)
    foe_first = jnp.array([False, True])
    acts = A("Lucky Chant", "Tackle", s, n)
    assert bool(damage_inputs_can_change(t, s, acts, foe_first)[1, 0])
    tackle = A("Tackle", "Tackle", s, n)
    assert not bool(damage_inputs_can_change(t, s, tackle, foe_first)[1, 0])
    up = s._replace(lucky_chant=s.lucky_chant.at[0].set(3))
    assert not bool(damage_inputs_can_change(t, up, acts, foe_first)[1, 0])
    assert bool(outcome_laws(t, up, acts).no_crit[1])
    assert not bool(outcome_laws(t, s, acts).no_crit[1])


# --- Charge (#174, E3-B2-8) ---------------------------------------------------------------------
# effect_script_0174.s: an indirect ON_HIT subscript_charge.s: MOVE_EFFECT_CHARGE and chargedTurns
# = 2 on the user (:5-6), then Sp. Def +1 (:11-13); the turn-end pass drops the counter by 1 and
# the bit goes at 0 (battle_controller_player.c:1626-1630); the user's Electric movePower doubles
# while it is set (battle_lib.c:6690-6692).


def _ch(t, n, mine=("Charge", "Thunderbolt", "Tackle"), foe_moves=("Tackle",), fast=True):
    """Monferno (side 0, ``mine``) against a Bidoof (``foe_moves``) with 1000 HP, no abilities; the
    Monferno is the faster one only if ``fast``."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 30, mine)],
        [_spec(n, "Bidoof", 20, foe_moves)],
    )
    s = s._replace(
        ability=s.ability.at[:, 0].set(AbilityId.NONE),
        hp=s.hp.at[1, 0].set(1000),
        max_hp=s.max_hp.at[1, 0].set(1000),
    )
    fast_spe, slow_spe = (500, 1) if fast else (1, 500)
    return s._replace(stats=s.stats.at[0, 0, SPE].set(fast_spe).at[1, 0, SPE].set(slow_spe))


def _chgo(t, n, s, me="Charge", foe="Tackle", out=None):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def _charged(s, turns):
    return s._replace(charged_turns=s.charged_turns.at[0].set(turns))


def _roll(value):
    return _hits()._replace(roll=jnp.full((2, MAX_HITS), value, jnp.int8))


def test_charge_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 174 (effect_script_0174.s) is ``STAT_SELF`` (Sp. Def +1) with ``CHARGE_PARAM``; Charge
    (move 268) is the only move with it; accuracy 0; PP 20 in vanilla and 1 in Kaizo (moves.json);
    the field is a volatile that Baton Pass drops and the Ingrain test reads."""
    from nuzline.engine.effects import MOVE_EFFECT_MASK_FIELDS
    from nuzline.engine.ids import CHARGE_PARAM
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    assert DECOMP_EFFECT_TO_ID[174] == (EffectId.STAT_SELF, (Stat.SPD, 1, CHARGE_PARAM))
    for t, nn, pp in ((tables, names, 20), (kaizo_tables, kaizo_names, 1)):
        move = nn.move("Charge")
        assert int(t.move_effect[move]) == EffectId.STAT_SELF
        assert np.asarray(t.move_param[move]).tolist() == [Stat.SPD, 1, CHARGE_PARAM]
        assert int((np.asarray(t.move_param)[:, 2] == CHARGE_PARAM).sum()) == 1
        assert int(t.move_accuracy[move]) == 0
        assert int(t.move_pp[move]) == pp
    assert "charged_turns" in VOLATILE_FIELDS and "charged_turns" not in BATON_PASSED_FIELDS
    assert "charged_turns" in MOVE_EFFECT_MASK_FIELDS


def test_charge_sets_two_turns_and_raises_sp_def_and_the_pass_counts_it_down(tables, names):
    """subscript_charge.s:5-6, :11-13: chargedTurns = 2 and Sp. Def +1 on the user only; the turn's
    pass (bcp.c:1626-1630) leaves 1, the next turn's pass 0 (the bit goes), so a Charge used on
    turn T is up for T + 1 and gone for T + 2."""
    t, n = tables, names
    s = _ch(t, n)
    one = _chgo(t, n, s)
    assert one.charged_turns.tolist() == [1, 0]
    assert one.stages[:, Stat.SPD].tolist() == [1, 0]
    two = _chgo(t, n, one, me="Tackle")
    assert two.charged_turns.tolist() == [0, 0]
    assert two.stages[:, Stat.SPD].tolist() == [1, 0]
    # A second Charge while up sets the counter to 2 again (the script has no fail test).
    again = _chgo(t, n, one)
    assert again.charged_turns.tolist() == [1, 0]
    assert again.stages[:, Stat.SPD].tolist() == [2, 0]


def test_charge_sets_the_counter_even_when_sp_def_cannot_rise(tables, names):
    """UPDATE_STAT_STAGE fails with a message at +6 (:11-13) but the counter was set first (:5-6)
    and the move is not a failure."""
    t, n = tables, names
    s = _ch(t, n)
    s = s._replace(stages=s.stages.at[0, Stat.SPD].set(6))
    out = _chgo(t, n, s)
    assert out.charged_turns.tolist() == [1, 0]
    assert int(out.stages[0, Stat.SPD]) == 6


def test_charge_raises_sp_def_before_a_slower_foes_special_move(tables, names):
    """The same-turn case: the +1 lands before the slower foe's Water Gun, so it deals less than
    against a user that used a move that raises nothing (Leer lowers the foe's Defense)."""
    t, n = tables, names
    s = _ch(t, n, mine=("Charge", "Leer"), foe_moves=("Water Gun",))
    charged = _chgo(t, n, s, foe="Water Gun")
    plain = _chgo(t, n, s, me="Leer", foe="Water Gun")
    assert _lost(s, charged, 0) < _lost(s, plain, 0)
    assert int(charged.stages[0, Stat.SPD]) == 1


def test_a_faster_foe_hits_before_the_charge_lands(tables, names):
    """A faster foe's special move meets the unraised Sp. Def (the same damage as against a user
    that did nothing); the counter is set all the same."""
    t, n = tables, names
    s = _ch(t, n, mine=("Charge", "Leer"), foe_moves=("Water Gun",), fast=False)
    charged = _chgo(t, n, s, foe="Water Gun")
    plain = _chgo(t, n, s, me="Leer", foe="Water Gun")
    assert _lost(s, charged, 0) == _lost(s, plain, 0) > 0
    assert charged.charged_turns.tolist() == [1, 0]


def test_a_charged_electric_move_deals_double_power_and_is_not_consumed(tables, names):
    """battle_lib.c:6690-6692: the user's Electric movePower x2 after powerMul (so the same as
    ``power_mul=20`` here), at either counter value; the hit does not consume it (the pass does)."""
    from nuzline.engine.damage import damage_for_roll

    t, n = tables, names
    s = _ch(t, n)
    move = n.move("Thunderbolt")
    base = int(damage_for_roll(t, s, 0, move, False, 7))
    doubled = int(damage_for_roll(t, s, 0, move, False, 7, power_mul=20))
    assert doubled > base
    for left in (1, 2):
        assert int(damage_for_roll(t, _charged(s, left), 0, move, False, 7)) == doubled
    out = _chgo(t, n, _charged(s, 1), me="Thunderbolt", out=_roll(7))
    assert _lost(s, out, 1) == doubled
    assert out.charged_turns.tolist() == [0, 0]  # the pass, not the hit, ended it
    plain = _chgo(t, n, s, me="Thunderbolt", out=_roll(7))
    assert _lost(s, plain, 1) == base


def test_charge_doubles_electric_only_and_only_the_chargers_own_moves(tables, names):
    """Only a move whose type is Electric (``moveType``): Tackle is unchanged, a charged foe's
    counter changes nothing of the user's damage, and the foe's own Electric move is doubled by
    its own counter."""
    from nuzline.engine.damage import damage_for_roll

    t, n = tables, names
    s = _ch(t, n)
    tackle = n.move("Tackle")
    assert int(damage_for_roll(t, _charged(s, 2), 0, tackle, False, 7)) == int(
        damage_for_roll(t, s, 0, tackle, False, 7)
    )
    foe_charged = s._replace(charged_turns=s.charged_turns.at[1].set(2))
    bolt = n.move("Thunderbolt")
    assert int(damage_for_roll(t, foe_charged, 0, bolt, False, 7)) == int(
        damage_for_roll(t, s, 0, bolt, False, 7)
    )
    shock = n.move("ThunderShock")
    assert int(damage_for_roll(t, foe_charged, 1, shock, False, 7)) > int(
        damage_for_roll(t, s, 1, shock, False, 7)
    )


def test_charge_comes_before_technician(tables, names):
    """battle_lib.c:6690-6702: Charge doubles Thunder Shock's 40 to 80 and Technician (power <= 60)
    then no longer applies: the damage is that of an 80-power hit, not 120."""
    from nuzline.engine.damage import damage_for_roll

    t, n = tables, names
    s = _ch(t, n)
    s = s._replace(ability=s.ability.at[0, 0].set(AbilityId.TECHNICIAN))
    shock = n.move("ThunderShock")
    assert int(t.move_power[shock]) == 40
    got = int(damage_for_roll(t, _charged(s, 1), 0, shock, False, 7))
    assert got == int(damage_for_roll(t, s, 0, shock, False, 7, in_power=80))
    assert got > int(damage_for_roll(t, s, 0, shock, False, 7))  # 40 x 1.5 = 60 alone


def test_the_ai_estimate_reads_the_charge(tables, names):
    """TrainerAI_CalcDamage -> BattleSystem_CalcMoveDamage (trainer_ai.c:3082-3091) reads the
    attacker's mask like the hit does."""
    from nuzline.engine.damage import ai_damage_for_roll

    t, n = tables, names
    s = _ch(t, n)
    move = n.move("Thunderbolt")
    base = int(ai_damage_for_roll(t, s, 0, move, 7))
    assert int(ai_damage_for_roll(t, _charged(s, 1), 0, move, 7)) > base


def test_a_switch_clears_the_charge(tables, names):
    """The mask and the data are zeroed with the volatiles (battle_lib.c:2071, :2101-2104;
    moves.h:46-48 does not list MOVE_EFFECT_CHARGE): a plain switch drops it."""
    t, n = tables, names
    two = make_state(
        t,
        [_spec(n, "Monferno", 30, ("Charge", "Tackle")), _spec(n, "Geodude", 30, ("Tackle",))],
        [_spec(n, "Bidoof", 20, ("Tackle",))],
    )
    two = two._replace(
        stats=two.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1),
        charged_turns=two.charged_turns.at[0].set(2),
    )
    out = assert_matches_reference(t, two, jnp.array([4 + 1, 0], jnp.int32), empty_outcome())
    assert int(out.charged_turns[0]) == 0


def test_a_charged_ingrain_user_no_longer_anchors_against_roar(tables, names):
    """subscript_force_target_to_switch_or_flee.s:7 compares the whole mask: MOVE_EFFECT_CHARGE
    next to Ingrain is not exactly Ingrain (``effects.ingrain_anchors``)."""
    from nuzline.engine.effects import ingrain_anchors

    t, n = tables, names
    s = _ch(t, n)
    rooted = s._replace(ingrain=s.ingrain.at[0].set(1))
    assert bool(ingrain_anchors(t, rooted, 0))
    assert not bool(ingrain_anchors(t, _charged(rooted, 1), 0))


# --- Grudge (#194, E3-B2-9) ---------------------------------------------------------------------
# effect_script_0194.s sets MOVE_EFFECT_GRUDGE on the user (:6); the user's next move attempt
# clears it (bcp.c:2427). A defender that fainted holding it to a living attacker of the other
# side (subscript_faint_check_destiny_bond.s:28, subscript_attack_then_switch_out.s:40) takes that
# attacker's selected move slot to 0 PP (BtlCmd_TryGrudge, battle_script.c:6743-6763), but never
# for Struggle.


def _gr(t, n, foe=("Growl", "Tackle"), foe_item="", mine=("Grudge", "Tackle"), hp=1):
    """Monferno (side 0, ``mine``, ``hp`` HP, a Geodude behind it) against a Bidoof (``foe``);
    the Monferno is the faster one, no abilities."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 50, mine), _spec(n, "Geodude", 50, ("Tackle",))],
        [_spec(n, "Bidoof", 50, foe, foe_item)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE), hp=s.hp.at[0, 0].set(hp))
    return s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))


def _grgo(t, n, s, me="Grudge", foe="Tackle", out=None):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def _foe_pp(s):
    return np.asarray(s.pp[1, 0]).tolist()[:2]


def test_grudge_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 194 (effect_script_0194.s) is EffectId.GRUDGE in both games; Grudge (move 288) is the
    only move with it; a neutral, non-damaging effect of the enumeration; the field is a volatile
    that Baton Pass drops (moves.h:46-48) and the Ingrain test reads."""
    from nuzline.engine.effects import MOVE_EFFECT_MASK_FIELDS
    from nuzline.engine.enumerate import INPUT_NEUTRAL_EFFECTS
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    assert DECOMP_EFFECT_TO_ID[194] == (EffectId.GRUDGE, (0, 0, 0))
    for t, nn in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[nn.move("Grudge")]) == EffectId.GRUDGE
        assert int((np.asarray(t.move_effect) == EffectId.GRUDGE).sum()) == 1
        assert int(t.move_accuracy[nn.move("Grudge")]) == 0
    assert EffectId.GRUDGE in INPUT_NEUTRAL_EFFECTS and EffectId.GRUDGE in NON_TABLE_DAMAGE_EFFECTS
    assert "grudge" in VOLATILE_FIELDS and "grudge" not in BATON_PASSED_FIELDS
    assert "grudge" in MOVE_EFFECT_MASK_FIELDS


def test_grudge_sets_the_bit_and_a_repeat_does_not_fail(tables, names):
    """effect_script_0194.s:6 sets the bit; the :5 fail never runs, because the user's own move
    attempt cleared the bit first (bcp.c:2427): Grudge over a set bit sets it again."""
    t, n = tables, names
    s = _gr(t, n, hp=100)
    out = _grgo(t, n, s, foe="Growl")
    assert out.grudge.tolist() == [1, 0]
    again = _grgo(t, n, out, foe="Growl")
    assert again.grudge.tolist() == [1, 0]


def test_a_ko_zeroes_the_ko_moves_slot_only(tables, names):
    """subscript_faint_check_destiny_bond.s:28 TryGrudge: the foe's Tackle (slot 1) knocks out the
    Grudge user; slot 1 goes to 0 PP, slot 0 keeps what it had."""
    t, n = tables, names
    s = _gr(t, n)
    before = _foe_pp(s)
    out = _grgo(t, n, s)
    assert int(out.hp[0, 0]) == 0
    assert _foe_pp(out) == [before[0], 0]
    # without the bit (a user of another move) the Tackle only costs its usual PP
    plain = _grgo(t, n, _gr(t, n), me="Tackle")
    assert int(plain.hp[0, 0]) == 0 and _foe_pp(plain) == [before[0], before[1] - 1]


def test_the_users_next_move_attempt_clears_grudge(tables, names):
    """CHECK_STATUS_START (bcp.c:2426-2427) clears the bit at any move attempt: a user holding it
    from an earlier turn that uses Tackle is not avenged."""
    t, n = tables, names
    s = _gr(t, n)
    s = s._replace(grudge=s.grudge.at[0].set(1))
    out = _grgo(t, n, s, me="Tackle")
    assert out.grudge.tolist() == [0, 0]
    assert _foe_pp(out) == [_foe_pp(s)[0], _foe_pp(s)[1] - 1]


def test_a_user_that_cannot_move_still_loses_the_bit(tables, names):
    """The clear is the first state of the check, before sleep stops the move: a sleeping user
    holding the bit has it gone and the foe's KO takes no PP beyond its own."""
    t, n = tables, names
    s = _gr(t, n).with_status(0, 0, Status.SLEEP, 5)
    s = s._replace(grudge=s.grudge.at[0].set(1))
    out = _grgo(t, n, s, me="Tackle")
    assert out.grudge.tolist() == [0, 0]
    assert _foe_pp(out) == [_foe_pp(s)[0], _foe_pp(s)[1] - 1]


def test_a_faint_outside_a_move_takes_no_pp(tables, names):
    """Residual damage faints with onlyFaint TRUE (bcp.c:940, :4004, :4722): no TryGrudge. A poisoned
    Grudge user at 1 HP faints at the turn end; the foe's Growl cost only its own PP."""
    t, n = tables, names
    s = _gr(t, n).with_status(0, 0, Status.POISON)
    out = _grgo(t, n, s, foe="Growl")
    assert int(out.hp[0, 0]) == 0
    assert _foe_pp(out) == [_foe_pp(s)[0] - 1, _foe_pp(s)[1]]


def test_a_destiny_bond_ko_leaves_no_attacker_for_grudge(tables, names):
    """subscript_faint_check_destiny_bond.s:5-25 takes the attacker down first, then :28 TryGrudge
    finds its HP 0 (battle_script.c:6751): nothing is zeroed. A switching holder of both bits (a
    switch is no move attempt) knocked out by a Pursuit: the pursuer takes its Pursuit's PP only."""
    from nuzline.engine.actions import MOVE_ACTIONS

    t, n = tables, names
    s = _gr(t, n, foe=("Growl", "Pursuit"))
    s = s._replace(grudge=s.grudge.at[0].set(1), destiny_bond=s.destiny_bond.at[0].set(1))
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 1), _hits())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert _foe_pp(out) == [_foe_pp(s)[0], _foe_pp(s)[1] - 1]


def test_an_invokers_own_slot_is_zeroed_not_the_called_moves(tables, names):
    """ATTACKER_MOVE_SLOT is the slot picked at command selection: a sleeping Bidoof's Sleep Talk
    (slot 0) calling Tackle (slot 1) loses Sleep Talk's PP slot."""
    t, n = tables, names
    s = _gr(t, n, foe=("Sleep Talk", "Tackle")).with_status(1, 0, Status.SLEEP, 5)
    before = _foe_pp(s)
    out = _grgo(t, n, s, foe="Sleep Talk", out=_hits()._replace(call=jnp.zeros(2, jnp.int16)))
    assert int(out.hp[0, 0]) == 0
    assert _foe_pp(out) == [0, before[1]]


def test_struggle_is_never_avenged(tables, names):
    """BtlCmd_TryGrudge needs moveTemp != MOVE_STRUGGLE (battle_script.c:6752); the slot write
    is skipped whatever PP the slot holds (``effects.try_grudge``), and a dead attacker keeps it."""
    from nuzline.engine.effects import try_grudge

    t, n = tables, names
    s = _gr(t, n)
    s = s._replace(hp=s.hp.at[0, 0].set(0))
    spared = try_grudge(s, 1, 0, 1, 0, True, True, True)
    assert _foe_pp(spared) == _foe_pp(s)
    zeroed = try_grudge(s, 1, 0, 1, 0, True, False, True)
    assert _foe_pp(zeroed) == [_foe_pp(s)[0], 0]
    dead = s._replace(hp=s.hp.at[1, 0].set(0))
    assert _foe_pp(try_grudge(dead, 1, 0, 1, 0, True, False, True)) == _foe_pp(dead)


def test_a_u_turn_ko_zeroes_the_users_slot_before_it_switches(tables, names):
    """subscript_attack_then_switch_out.s:40 TryGrudge (after the Destiny Bond block, before the
    switch): the foe's U-turn knocks the Grudge user out and loses its slot's PP, then pauses at
    its party list."""
    t, n = tables, names
    s = make_state(
        t,
        [_spec(n, "Monferno", 50, ("Grudge",)), _spec(n, "Geodude", 50, ("Tackle",))],
        [
            _spec(n, "Bidoof", 50, ("Growl", "U-turn")),
            _spec(n, "Geodude", 50, ("Tackle", "Rock Throw")),
        ],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE), hp=s.hp.at[0, 0].set(1))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))
    out = _grgo(t, n, s, foe="U-turn")
    assert int(out.hp[0, 0]) == 0 and int(out.pivot_battler) == 2
    assert _foe_pp(out) == [_foe_pp(s)[0], 0]


def test_the_resumed_u_turn_zeroes_the_newcomers_slot_too(tables, names):
    """Review b2 #5: after :40 zeroed the U-turn user's slot, the defender stays flagged
    (SYSCTL_MON_FAINTED, battle_script.c:1770-1771, cleared only by PlayFaintAnimation :1799) with
    its Grudge bit (cleared at its faint cleanup, battle_lib.c:2166): the paused state keeps it
    (``ts_grudge``), and the resumed action's LOOP_FAINTED (bcp.c:3820-3823) runs
    subscript_faint_check_destiny_bond.s:28's TryGrudge with the attacker battler's newcomer:
    moveSlot[attacker] (common.h:8) is still U-turn's slot, so the newcomer's Rock Throw (its
    slot 1) loses its PP, its Tackle not; both engines agree (the resume) and the scratch is 0
    after it. Without the bit the newcomer keeps its PP."""
    t, n = tables, names
    s = make_state(
        t,
        [_spec(n, "Monferno", 50, ("Grudge",)), _spec(n, "Geodude", 50, ("Tackle",))],
        [
            _spec(n, "Bidoof", 50, ("Growl", "U-turn")),
            _spec(n, "Geodude", 50, ("Tackle", "Rock Throw")),
        ],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE), hp=s.hp.at[0, 0].set(1))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))
    paused = _grgo(t, n, s, foe="U-turn")
    assert int(paused.pivot_battler) == 2 and np.asarray(paused.ts_grudge).tolist() == [1, 0]
    full = np.asarray(s.pp[1, 1]).tolist()[:2]

    def resume(state):
        acts = np.asarray(state.ts_actions).astype(int).tolist()
        acts[1] = MOVE_ACTIONS + 1
        return assert_matches_reference(t, state, tuple(acts), _hits())

    out = resume(paused)
    assert int(out.active[1]) == 1 and np.asarray(out.pp[1, 1]).tolist()[:2] == [full[0], 0]
    assert np.asarray(out.ts_grudge).tolist() == [0, 0]
    plain = resume(paused._replace(ts_grudge=jnp.zeros_like(paused.ts_grudge)))
    assert np.asarray(plain.pp[1, 1]).tolist()[:2] == full


def test_the_reference_turn_sizes_the_grudge_scratch_per_battler(tables, names):
    """Review b2-fix #1: ``_write_scratch`` writes ``turn.grudge_kept`` to ``ts_grudge`` on every
    pivot pause through ``_put`` (strict per battler), so a four-battler turn's list is four long
    from ``__post_init__``, not only after a U-turn pause replaced it."""
    from nuzline.data.tables import to_numpy
    from nuzline.reference.convert import to_ref
    from nuzline.reference.engine import _put, _Turn
    from tests.engine.d2_states import four

    s = to_ref(four(tables, names))
    turn = _Turn(to_numpy(tables), None, entry_turn=1, battler_count=4)
    assert turn.grudge_kept == [0, 0, 0, 0]
    _put(s, "ts_grudge", turn.grudge_kept)
    assert [bt.ts_grudge for bt in s.battlers] == [0, 0, 0, 0]


def test_a_pursuit_ko_of_a_switching_grudge_user_zeroes_the_pursuers_slot(tables, names):
    """subscript_pursuit.s:85 runs the same faint check: a Grudge holder at 1 HP switching out to
    a Pursuit zeroes the pursuer's chosen slot (here also the one Pursuit spent). The bit stays
    from an earlier turn: a switch is no move attempt."""
    from nuzline.engine.actions import MOVE_ACTIONS

    t, n = tables, names
    s = _gr(t, n, foe=("Growl", "Pursuit"))
    s = s._replace(grudge=s.grudge.at[0].set(1))
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 1), _hits())
    assert int(out.hp[0, 0]) == 0
    assert _foe_pp(out) == [_foe_pp(s)[0], 0]


def test_a_pursuit_grudge_zeroes_the_chosen_slot_not_the_pp_slot(tables, names):
    """Pursuit in slots 0 and 2, slot 2 chosen: BtlCmd_TryPursuit spends the PP of the first slot
    holding the move (battle_script.c:6913-6917, Battler_SlotForMove) and writes attacker but never
    moveSlot (:6926); TryGrudge zeroes ATTACKER_MOVE_SLOT (:6753-6754, common.h:8), the slot chosen
    at selection (bcp.c:526). So slot 0 loses one PP and slot 2 goes to 0."""
    from nuzline.engine.actions import MOVE_ACTIONS

    t, n = tables, names
    s = _gr(t, n, foe=("Pursuit", "Growl", "Pursuit"))
    s = s._replace(grudge=s.grudge.at[0].set(1))
    full = np.asarray(s.pp[1, 0]).tolist()[:3]
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 2), _hits())
    assert int(out.hp[0, 0]) == 0
    assert np.asarray(out.pp[1, 0]).tolist()[:3] == [full[0] - 1, full[1], 0]


@pytest.mark.parametrize("ability", ["Aftermath", "Rough Skin"])
def test_a_pursuit_ko_runs_grudge_before_the_contact_ability_that_faints_the_pursuer(
    tables, names, ability
):
    """subscript_pursuit.s:85 (FAINT_CHECK_DESTINY_BOND, whose :28 TryGrudge needs the pursuer's
    HP, battle_script.c:6751) runs before TriggerAbilityOnHit (:86-87): the switcher's Aftermath
    (battle_lib.c:4358-4373, maxHP/4) or Rough Skin (:4257-4271, maxHP/8) faints the 1-HP pursuer
    after its Pursuit slot is zeroed, so the slot ends at 0 PP."""
    from nuzline.engine.actions import MOVE_ACTIONS

    t, n = tables, names
    s = _gr(t, n, foe=("Growl", "Pursuit"))
    held = AbilityId.AFTERMATH if ability == "Aftermath" else AbilityId.ROUGH_SKIN
    s = s._replace(grudge=s.grudge.at[0].set(1), ability=s.ability.at[0, 0].set(held))
    s = s._replace(hp=s.hp.at[1, 0].set(1))
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 1), _hits())
    assert int(out.hp[0, 0]) == 0 and int(out.hp[1, 0]) == 0
    assert _foe_pp(out) == [_foe_pp(s)[0], 0]


def test_the_attackers_leppa_misses_the_zero_until_the_turn_end(tables, names):
    """The attacker's after-move held-item check (bcp.c:3687) runs before LOOP_FAINTED, so Leppa
    finds no empty slot at the move; the end-of-turn check (bcp.c:1363) restores the zeroed slot
    and spends the berry."""
    t, n = tables, names
    s = _gr(t, n, foe_item="Leppa Berry")
    out = _grgo(t, n, s)
    assert int(out.item[1, 0]) == 0
    assert _foe_pp(out)[1] == int(t.item_param[n.item("Leppa Berry"), 0])


def test_a_mimicked_slot_loses_only_the_battlemons_pp_to_grudge(tables, names):
    """BattleMon_CopyToParty skips a Mimic-flagged slot: the KO-ing battler's BattleMon PP goes to
    0, the party's shadow of the slot stays."""
    t, n = tables, names
    s = _gr(t, n)
    s = s._replace(
        mimic_slots=s.mimic_slots.at[1].set(2),
        shadow_moves=s.shadow_moves.at[1, 1].set(n.move("Rock Throw")),
        shadow_pp=s.shadow_pp.at[1, 1].set(15),
    )
    out = _grgo(t, n, s)
    assert _foe_pp(out)[1] == 0 and int(out.shadow_pp[1, 1]) == 15


def test_a_switch_clears_grudge_and_the_ingrain_test_reads_it(tables, names):
    """Cleared with the other volatiles (battle_lib.c:2071); the whole-mask Roar test
    (subscript_force_target_to_switch_or_flee.s:7) no longer finds an Ingrain-only mask."""
    from nuzline.engine.actions import MOVE_ACTIONS
    from nuzline.engine.effects import ingrain_anchors

    t, n = tables, names
    s = _gr(t, n, hp=100)
    s = s._replace(grudge=s.grudge.at[0].set(1))
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 0), _hits())
    assert out.grudge.tolist() == [0, 0]
    rooted = s._replace(ingrain=s.ingrain.at[0].set(1))
    assert not bool(ingrain_anchors(t, rooted, 0))
    assert bool(ingrain_anchors(t, rooted._replace(grudge=rooted.grudge.at[0].set(0)), 0))


# --- Miracle Eye (#216, E3-B2-10) ---------------------------------------------------------------
# effect_script_0216.s: an indirect ON_HIT subscript_miracle_eye.s (:5 MOVE_EFFECT_MIRACLE_EYE on
# the defender: no failure, no Substitute test; run unless MOVE_STATUS_NO_EFFECTS, battle_lib.c:
# 1524-1529) and SYSCTL_IGNORE_IMMUNITIES (inert: power 0 skips CheckTypeChart, bcp.c:2352-2357).
# MON_IS_IDENTIFIED (common.h:48-49) ignores the defender's positive evasion stage (bcp.c:2888-2890)
# and BasicTypeMulApplies skips the immune Dark rows (battle_lib.c:2551-2555).

_ME_DECOMP = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"


def _me(t, n, foe="Mightyena", foe_moves=("Tackle",), fast=True, foe_level=40):
    """Monferno (side 0: Miracle Eye, Psychic, Tackle, Earthquake) against ``foe`` (``foe_moves``),
    no abilities, one Pokemon each; the Monferno is the faster one only if ``fast``."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 40, ("Miracle Eye", "Psychic", "Tackle", "Earthquake"))],
        [_spec(n, foe, foe_level, foe_moves)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    fast_spe, slow_spe = (500, 1) if fast else (1, 500)
    return s._replace(stats=s.stats.at[0, 0, SPE].set(fast_spe).at[1, 0, SPE].set(slow_spe))


def _eyed(s, side=1):
    return s._replace(miracle_eye=s.miracle_eye.at[side].set(1))


def _me_roll(value):
    """Every hit lands on a roll of ``value`` (``RandNext() % 100 + 1``, the accuracy check)."""
    return empty_outcome()._replace(hit=jnp.full((2, MAX_HITS), value, jnp.int8))


def _mego(t, n, s, me="Miracle Eye", foe="Tackle", out=None):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def test_miracle_eye_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 216 (effect_script_0216.s) is EffectId.MIRACLE_EYE in both games and Miracle Eye
    (move 357) the only move with it: power 0, accuracy 0 (no roll), AI effect 216; its
    SYSCTL_IGNORE_IMMUNITIES is in ``IGNORES_TYPE_CHECKS``; a neutral, non-damaging effect of the
    enumeration; a volatile that Baton Pass drops (moves.h:46-48) and the Ingrain test reads."""
    from nuzline.engine.effects import IGNORES_TYPE_CHECKS, MOVE_EFFECT_MASK_FIELDS
    from nuzline.engine.enumerate import INPUT_NEUTRAL_EFFECTS
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    assert DECOMP_EFFECT_TO_ID[216] == (EffectId.MIRACLE_EYE, (0, 0, 0))
    for t, nn in ((tables, names), (kaizo_tables, kaizo_names)):
        move = nn.move("Miracle Eye")
        assert move == 357 and int(t.move_effect[move]) == EffectId.MIRACLE_EYE
        assert int((np.asarray(t.move_effect) == EffectId.MIRACLE_EYE).sum()) == 1
        assert int(t.move_accuracy[move]) == 0 and int(t.move_power[move]) == 0
        assert int(t.move_ai_effect[move]) == 216
        assert int(t.move_flags[move]) & MoveFlag.PROTECT
    assert EffectId.MIRACLE_EYE in IGNORES_TYPE_CHECKS
    assert EffectId.MIRACLE_EYE in INPUT_NEUTRAL_EFFECTS
    assert EffectId.MIRACLE_EYE in NON_TABLE_DAMAGE_EFFECTS
    assert "miracle_eye" in VOLATILE_FIELDS and "miracle_eye" not in BATON_PASSED_FIELDS
    assert "miracle_eye" in MOVE_EFFECT_MASK_FIELDS


def test_the_decomp_scripts_are_as_the_sheet_says():
    """effect_script_0216.s sets the indirect subscript and SYSCTL_IGNORE_IMMUNITIES; the
    subscript sets the bit on BTLSCR_DEFENDER without a jump (no failure, no Substitute test);
    battle_lib.c:2551-2553 and common.h:48-49 read it; bcp.c:2888 ignores positive evasion."""
    if not _ME_DECOMP.exists():
        pytest.skip("decomp not present")
    effect = (_ME_DECOMP / "res/battle/scripts/effects/effect_script_0216.s").read_text("utf-8")
    assert "MOVE_SIDE_EFFECT_ON_HIT|MOVE_SUBSCRIPT_PTR_MIRACLE_EYE" in effect
    assert "SYSCTL_IGNORE_IMMUNITIES" in effect
    sub = (_ME_DECOMP / "res/battle/scripts/subscripts/subscript_miracle_eye.s").read_text("utf-8")
    assert (
        "UpdateMonData OPCODE_FLAG_ON, BTLSCR_DEFENDER, BATTLEMON_MOVE_EFFECTS_MASK, "
        "MOVE_EFFECT_MIRACLE_EYE" in sub
    )
    assert "Jump" not in sub and "Substitute" not in sub and "If" not in sub
    lib = (_ME_DECOMP / "src/battle/battle_lib.c").read_text("utf-8").splitlines()
    assert "MOVE_EFFECT_MIRACLE_EYE" in lib[2550]
    assert "TYPE_DARK" in lib[2551] and "TYPE_MULTI_IMMUNE" in lib[2552]
    common = (_ME_DECOMP / "include/battle/common.h").read_text("utf-8").splitlines()
    assert "VOLATILE_CONDITION_FORESIGHT" in common[47] and "MOVE_EFFECT_MIRACLE_EYE" in common[48]
    bcp = (_ME_DECOMP / "src/battle/battle_controller_player.c").read_text("utf-8").splitlines()
    assert "MON_IS_IDENTIFIED(defender) && evaStages < 0" in bcp[2887]


def test_miracle_eye_sets_the_bit_on_the_target_and_a_repeat_keeps_it(tables, names):
    """subscript_miracle_eye.s:5 flags BTLSCR_DEFENDER; it never fails, so a second use over a set
    bit leaves it set (no fail branch, unlike Grudge's); the user's own bit stays clear."""
    t, n = tables, names
    s = _me(t, n)
    out = _mego(t, n, s)
    assert out.miracle_eye.tolist() == [0, 1]
    assert _mego(t, n, out).miracle_eye.tolist() == [0, 1]
    # Foresight's flag is another bit: not set.
    assert out.foresight.tolist() == [0, 0]


def test_miracle_eye_goes_through_a_substitute(tables, names):
    """The subscript has no Substitute test (battle_lib.c:1524-1529 checks only NO_EFFECTS): a
    target behind a Substitute is still identified."""
    t, n = tables, names
    s = _me(t, n)
    sub = s._replace(substitute_hp=s.substitute_hp.at[1].set(40))
    out = _mego(t, n, sub)
    assert out.miracle_eye.tolist() == [0, 1] and int(out.substitute_hp[1]) == 40


def test_protect_and_a_semi_invulnerable_target_stop_it(tables, names):
    """NO_EFFECTS (Protect, the semi-invulnerable miss) stops the ON_HIT subscript
    (battle_lib.c:1524-1529): a faster Protect, and a target mid-Fly, are not identified."""
    from nuzline.engine.ids import SemiKind

    t, n = tables, names
    protected = _mego(t, n, _me(t, n, foe_moves=("Protect",), fast=False), foe="Protect")
    assert protected.miracle_eye.tolist() == [0, 0]
    base = _me(t, n, foe="Pidgey", foe_moves=("Fly",))
    up = base._replace(
        semi_invulnerable=base.semi_invulnerable.at[1].set(SemiKind.AIRBORNE),
        locked_move=base.locked_move.at[1].set(n.move("Fly")),
    )
    assert _mego(t, n, up, foe="Fly").miracle_eye.tolist() == [0, 0]
    # The same foe not in the air yet: the faster Miracle Eye lands before its Fly charges.
    assert _mego(t, n, base, foe="Fly").miracle_eye.tolist() == [0, 1]


def test_an_identified_target_ignores_positive_evasion_in_every_accuracy_read(tables, names):
    """bcp.c:2888-2890: evasion +2 counts as 0 for the bit's holder (hit_threshold, the future-hit
    stage sum, the live check in ``step`` and its reference); a negative stage still counts."""
    from nuzline.engine.outcome import stage_total

    t, n = tables, names
    s = _me(t, n)
    acts = A("Tackle", "Tackle", s, n)

    def threshold(eva, eyed):
        x = s.with_stages(1, eva=eva)
        return int(hit_threshold(t, _eyed(x) if eyed else x, acts)[0])

    assert threshold(2, True) == 95
    assert threshold(2, False) == 57  # 95 * 60 // 100 (HitRateByStage[4])
    assert threshold(-2, True) == 157  # 95 * 166 // 100, unchanged by the bit
    assert threshold(-1, True) == 126  # 95 * 133 // 100
    assert threshold(0, True) == threshold(0, False) == 95
    # The stage sum (``stage_total``): 6 + (-evasion) + accuracy, clamped to 0-12.
    x = s.with_stages(1, eva=2)
    assert int(stage_total(t, x, 0, 1, x.active[1], 0, 2)) == 4
    assert int(stage_total(t, _eyed(x), 0, 1, x.active[1], 0, 2)) == 6
    assert int(stage_total(t, _eyed(x), 0, 1, x.active[1], 0, -2)) == 8
    # The attacker's own bit does not matter: only the defender's.
    assert int(hit_threshold(t, _eyed(x, 0), acts)[0]) == 57
    # The live check: a roll of 80 misses behind +2 evasion, hits once the target is identified.
    miss = _mego(t, n, x, me="Tackle", out=_me_roll(80))
    hit = _mego(t, n, _eyed(x), me="Tackle", out=_me_roll(80))
    assert int(miss.hp[1, 0]) == int(x.hp[1, 0])
    assert int(hit.hp[1, 0]) < int(x.hp[1, 0])


def test_the_dark_immunity_is_skipped_for_an_identified_defender(tables, names):
    """battle_lib.c:2551-2555 BasicTypeMulApplies drops the Dark rows with multiplier 0: Psychic
    hits a Dark Pokemon at the other type's multiplier (pure Dark x1, Dark/Poison x2, Dark/Flying
    x1) while the bit is up, and not before; the Ground row of a Flying defender and the Ghost
    rows stay as they were (a separate flag from Foresight's rows after the 0xFE separator)."""
    t, n = tables, names
    psychic = n.move("Psychic")
    for foe in ("Mightyena", "Drapion", "Honchkrow"):
        s = _me(t, n, foe=foe)
        types = [int(x) for x in np.asarray(s.types[1, 0])]
        assert n.type("Dark") in types
        rest = [x for x in types if x != n.type("Dark")]
        other = rest[0] if rest else n.type("Normal")  # the other type, or a neutral one
        compare = s._replace(types=s.types.at[1, 0].set(jnp.asarray([other, other], jnp.int8)))
        want = int(damage_table(t, compare, 0, psychic, False)[15])
        assert int(damage_table(t, s, 0, psychic, False)[15]) == 0, foe
        assert want > 0, foe
        assert int(damage_table(t, _eyed(s), 0, psychic, False)[15]) == want, foe
    # The unit view: effectiveness(), tenths.
    psy, dark, ground = n.type("Psychic"), n.type("Dark"), n.type("Ground")
    flying, ghost, normal = n.type("Flying"), n.type("Ghost"), n.type("Normal")
    assert int(effectiveness(t, psy, jnp.asarray([dark, dark]), 0)) == 0
    assert int(effectiveness(t, psy, jnp.asarray([dark, dark]), CHART_EYED)) == 100
    assert int(effectiveness(t, psy, jnp.asarray([dark, ghost]), CHART_EYED)) == 100
    poison = n.type("Poison")
    assert int(effectiveness(t, psy, jnp.asarray([poison, dark]), CHART_EYED)) == 200
    # Only Dark rows with multiplier 0: Ground on Flying, Normal on Ghost stay immune.
    assert int(effectiveness(t, ground, jnp.asarray([flying, flying]), CHART_EYED)) == 0
    assert int(effectiveness(t, normal, jnp.asarray([ghost, ghost]), CHART_EYED)) == 0
    # Fighting on Dark (x2) and other Psychic rows (Steel x0.5) are unchanged.
    fighting, steel = n.type("Fighting"), n.type("Steel")
    assert int(effectiveness(t, fighting, jnp.asarray([dark, dark]), CHART_EYED)) == 200
    assert int(effectiveness(t, psy, jnp.asarray([steel, steel]), CHART_EYED)) == 50
    # The state flags: Miracle Eye alone sets CHART_EYED, never CHART_IDENTIFIED (Foresight's bit).
    plain = _me(t, n)
    flags = int(identified(t, _eyed(plain), 0, 1))
    assert flags & CHART_EYED and not flags & CHART_IDENTIFIED
    assert not int(identified(t, plain, 0, 1)) & CHART_EYED


def test_a_psychic_hit_on_an_identified_dark_foe_deals_damage_in_a_step(tables, names):
    """The same through ``step`` and its reference: a Psychic into a Dark foe does nothing, and
    with Miracle Eye in effect the foe loses HP, the turn after the Miracle Eye lands too; a
    Sableye (Dark/Ghost) is still immune to Normal (Foresight's rows, not this bit's)."""
    t, n = tables, names
    s = _me(t, n)
    nothing = _mego(t, n, s, me="Psychic")
    assert int(nothing.hp[1, 0]) == int(s.hp[1, 0])
    hurt = _mego(t, n, _eyed(s), me="Psychic")
    assert int(hurt.hp[1, 0]) < int(s.hp[1, 0])
    first = _mego(t, n, s)
    second = _mego(t, n, first, me="Psychic")
    assert int(second.hp[1, 0]) < int(first.hp[1, 0])
    sab = _eyed(_me(t, n, foe="Sableye"))
    assert int(_mego(t, n, sab, me="Tackle").hp[1, 0]) == int(sab.hp[1, 0])
    assert int(_mego(t, n, sab, me="Psychic").hp[1, 0]) < int(sab.hp[1, 0])


def test_a_switch_clears_miracle_eye_and_the_ingrain_test_reads_it(tables, names):
    """Cleared with the other volatiles (battle_lib.c:2071, :2166); the whole-mask Roar test
    (subscript_force_target_to_switch_or_flee.s:7) no longer finds an Ingrain-only mask while it
    is set."""
    from nuzline.engine.actions import MOVE_ACTIONS
    from nuzline.engine.effects import ingrain_anchors

    t, n = tables, names
    s = make_state(
        t,
        [_spec(n, "Monferno", 40, ("Tackle",))],
        [_spec(n, "Mightyena", 40, ("Tackle",)), _spec(n, "Geodude", 40, ("Tackle",))],
    )
    s = _eyed(s._replace(ability=s.ability.at[:, :].set(AbilityId.NONE)))
    out = assert_matches_reference(t, s, (0, MOVE_ACTIONS + 1), _hits())
    assert out.miracle_eye.tolist() == [0, 0]
    rooted = _eyed(s, 0)._replace(ingrain=s.ingrain.at[0].set(1))
    assert not bool(ingrain_anchors(t, rooted, 0))
    assert bool(ingrain_anchors(t, rooted._replace(miracle_eye=rooted.miracle_eye * 0), 0))


# --- Last Resort (#246, E3-B2-11) --------------------------------------------------------------
# effect_script_0246.s: TryLastResort (:5; MOVE_STATUS_FAILED at :10-11), then CalcCrit and
# CalcDamage (:6-7). BtlCmd_TryLastResort (battle_script.c:7567-7579) fails the move when the
# user's lastResortCount is under numMoves - 1 or it knows fewer than 2 moves (numMoves =
# Battler_CountMoves, battle_lib.c:3461-3471). BattleSystem_UpdateLastResort (battle_lib.c:3443-3459)
# records moveTemp at the end of BeforeMove (battle_controller_player.c:3209-3218).

MOVE_LAST_RESORT = 387


def _lr(t, n, moves=("Last Resort", "Tackle"), used=(), sleeping=False, count=None):
    """Monferno (side 0, ``moves``) against a slower Geodude that only curls up (it never hurts
    the Monferno), no abilities; the Monferno's use history is ``used`` (move names; ``count``
    overrides the length). ``sleeping`` puts the Monferno to sleep (counter 3)."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 30, moves)],
        [_spec(n, "Geodude", 40, ("Defense Curl",))],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))
    ids = [n.move(m) for m in used]
    s = s._replace(
        last_resort_count=s.last_resort_count.at[0].set(len(ids) if count is None else count),
        last_resort_moves=s.last_resort_moves.at[0].set(
            jnp.asarray(ids + [0] * (4 - len(ids)), jnp.int16)
        ),
    )
    if sleeping:
        s = s._replace(
            status=s.status.at[0, 0].set(Status.SLEEP),
            status_counter=s.status_counter.at[0, 0].set(3),
        )
    return s


def _lrgo(t, n, s, me="Last Resort", foe="Defense Curl", out=None):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def _history(s, side=0):
    """``(count, listed ids)`` of ``side``'s Last Resort history."""
    count = int(s.last_resort_count[side])
    return count, np.asarray(s.last_resort_moves[side]).tolist()[:count]


def _foe_hp(s):
    return int(s.hp[1, 0])


def test_last_resort_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 246 (effect_script_0246.s) is EffectId.LAST_RESORT in both games and Last Resort
    (move 387, 130 BP Normal, accuracy 100, PP 5) the only move with it; it is a plain hit that can
    fail, classed as Focus Punch is (enumerate's two allowlists); Struggle is 165."""
    from nuzline.engine.enumerate import INPUT_NEUTRAL_EFFECTS, TABLE_DAMAGE_EFFECTS

    assert DECOMP_EFFECT_TO_ID[246] == (EffectId.LAST_RESORT, (0, 0, 0))
    for t, nn in ((tables, names), (kaizo_tables, kaizo_names)):
        move = nn.move("Last Resort")
        assert move == MOVE_LAST_RESORT
        assert int(t.move_effect[move]) == EffectId.LAST_RESORT
        assert int((np.asarray(t.move_effect) == EffectId.LAST_RESORT).sum()) == 1
        assert int(t.move_power[move]) == 130 and int(t.move_accuracy[move]) == 100
        assert int(t.move_pp[move]) == 5 and int(t.move_type[move]) == nn.type("Normal")
        assert nn.move("Struggle") == MOVE_STRUGGLE
    assert EffectId.LAST_RESORT in INPUT_NEUTRAL_EFFECTS
    assert EffectId.LAST_RESORT in TABLE_DAMAGE_EFFECTS


def test_last_resort_fails_until_the_other_move_was_used(tables, names):
    """TryLastResort (battle_script.c:7573-7574), two moves: the count must reach 1. The first use
    fails (no damage) but spends its PP (DecrementPP precedes the script) and records nothing
    (battle_lib.c:3445: Last Resort itself); the Tackle records itself; then Last Resort hits and
    still records nothing."""
    t, n = tables, names
    s = _lr(t, n)
    first = _lrgo(t, n, s)
    assert _foe_hp(first) == _foe_hp(s)
    assert int(first.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
    assert _history(first) == (0, [])
    tackled = _lrgo(t, n, first, me="Tackle")
    assert _history(tackled) == (1, [n.move("Tackle")])
    hit = _lrgo(t, n, tackled)
    assert _foe_hp(hit) < _foe_hp(tackled)
    assert _history(hit) == (1, [n.move("Tackle")])


def test_last_resort_alone_never_works(tables, names):
    """numMoves < 2 fails (:7574) whatever the count: the count of 3 is set by hand (a one-move
    user's history cannot reach it, Last Resort not being recorded)."""
    t, n = tables, names
    for count in (0, 3):
        s = _lr(t, n, moves=("Last Resort",), count=count)
        assert _foe_hp(_lrgo(t, n, s)) == _foe_hp(s), count


def test_last_resort_counts_each_other_move_once_and_needs_all_of_them(tables, names):
    """Four moves need a count of 3 (:7573). Ids are counted, not uses (:3451-3455): a repeated
    Tackle records nothing; Leer makes 2 (the Last Resort still fails), Growl 3 (it works)."""
    t, n = tables, names
    s = _lr(t, n, moves=("Last Resort", "Tackle", "Leer", "Growl"))
    again = _lrgo(t, n, _lrgo(t, n, s, me="Tackle"), me="Tackle")
    assert _history(again) == (1, [n.move("Tackle")])
    two = _lrgo(t, n, again, me="Leer")
    assert _history(two) == (2, [n.move("Tackle"), n.move("Leer")])
    assert _foe_hp(_lrgo(t, n, two)) == _foe_hp(two)
    three = _lrgo(t, n, two, me="Growl")
    assert _history(three)[0] == 3
    assert _foe_hp(_lrgo(t, n, three)) < _foe_hp(three)


def test_a_miss_still_counts(tables, names):
    """The record is made before the accuracy check (BeforeMove's tail, bcp.c:3209-3218): a Tackle
    that misses counts."""
    t, n = tables, names
    s = _lr(t, n)
    off = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    miss = _lrgo(t, n, s, me="Tackle", out=off)
    assert _foe_hp(miss) == _foe_hp(s)
    assert _history(miss) == (1, [n.move("Tackle")])


def test_struggle_is_recorded_by_its_own_id(tables, names):
    """A user with no PP left struggles: moveTemp is MOVE_STRUGGLE (bcp.c:1867), which the list
    takes (it is not Last Resort)."""
    t, n = tables, names
    s = _lr(t, n)
    dry = s._replace(pp=s.pp.at[0, 0].set(0))
    out = _lrgo(t, n, dry, me=0)
    assert _history(out) == (1, [MOVE_STRUGGLE])


def test_a_full_list_records_nothing_more(tables, names):
    """lastResortCount == 4 returns first (battle_lib.c:3446-3447): the list and the count stay;
    and 4 >= 3 lets a Last Resort through."""
    t, n = tables, names
    full = _lr(t, n, moves=("Last Resort", "Tackle", "Leer", "Growl"))
    ids = [n.move("Tackle"), n.move("Leer"), n.move("Growl"), MOVE_STRUGGLE]
    full = full._replace(
        last_resort_count=full.last_resort_count.at[0].set(4),
        last_resort_moves=full.last_resort_moves.at[0].set(jnp.asarray(ids, jnp.int16)),
    )
    out = _lrgo(t, n, full, me="Tackle")
    assert _history(out) == (4, ids)
    assert _foe_hp(_lrgo(t, n, full)) < _foe_hp(full)


def test_no_pp_and_a_sleepers_move_record_nothing(tables, names):
    """Only a move that gets through BeforeMove records: one with no PP (MOVE_STATUS_NO_PP is a
    NO_EFFECTS status, bcp.c:2285, :3209) and a sleeper's disrupted move (CheckStatusDisruption
    returns first) do not."""
    t, n = tables, names
    s = _lr(t, n)
    empty = s._replace(pp=s.pp.at[0, 0, 1].set(0))
    out = _lrgo(t, n, empty, me=1)
    assert _history(out) == (0, [])
    asleep = _lr(t, n, sleeping=True)
    out = _lrgo(t, n, asleep, me="Tackle")
    assert _history(out) == (0, []) and int(out.status[0, 0]) == Status.SLEEP


def test_an_invoker_is_recorded_not_the_move_it_calls(tables, names):
    """Sleep Talk calls Last Resort (moveTemp stays Sleep Talk, :3445-3458): the history holds
    Sleep Talk, so the called Last Resort (two moves, count 1) works and is not recorded."""
    t, n = tables, names
    s = _lr(t, n, moves=("Sleep Talk", "Last Resort"), sleeping=True)
    out = _lrgo(t, n, s, me="Sleep Talk", out=_hits()._replace(call=jnp.zeros(2, jnp.int16)))
    assert _history(out) == (1, [n.move("Sleep Talk")])
    assert _foe_hp(out) < _foe_hp(s)


def test_mimic_copying_last_resort_and_transform_zero_the_count(tables, names):
    """battle_script.c:4514-4516: a Mimic that copies Last Resort zeroes the count (Mimic itself
    was recorded before its script); a Mimic of another move does not; BtlCmd_Transform zeroes it
    (:5340)."""
    t, n = tables, names
    s = _lr(t, n, moves=("Mimic", "Tackle"), used=("Tackle",))
    copies = s._replace(last_move=s.last_move.at[1].set(MOVE_LAST_RESORT))
    out = _lrgo(t, n, copies, me="Mimic")
    assert int(out.moves[0, 0, 0]) == MOVE_LAST_RESORT  # the copy landed
    assert _history(out)[0] == 0
    other = s._replace(last_move=s.last_move.at[1].set(n.move("Leer")))
    out = _lrgo(t, n, other, me="Mimic")
    assert int(out.moves[0, 0, 0]) == n.move("Leer")
    assert _history(out) == (2, [n.move("Tackle"), n.move("Mimic")])
    trans = _lr(t, n, moves=("Transform", "Tackle"), used=("Tackle",))
    out = _lrgo(t, n, trans, me="Transform")
    assert int(out.transformed[0]) == 1 and _history(out)[0] == 0


def test_a_reflected_move_is_recorded_for_the_magic_coat_holder_only(tables, names):
    """The holder's second pass reaches BeforeMove's tail with moveTemp the mover's move
    (bcp.c:3196-3221, as the Metronome count does): the holder lists Magic Coat, then the Toxic;
    the mover's first pass returned at CHECK_STOLEN (:3064-3085), so its history stays empty."""
    t, n = tables, names
    out = _cgo(t, n, _coat(t, n))
    assert _history(out, 0) == (2, [n.move("Magic Coat"), n.move("Toxic")])
    assert _history(out, 1) == (0, [])


def test_a_switch_clears_the_history_and_baton_pass_does_not_keep_it(tables, names):
    """moveEffectsData is zeroed whole at a switch and a faint (battle_lib.c:2100-2104, :2166) and
    Baton Pass keeps only the fields listed at :2107-2115: the fields are ``VOLATILE_FIELDS``
    members, not ``BATON_PASSED_FIELDS``."""
    from nuzline.engine.actions import MOVE_ACTIONS
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    t, n = tables, names
    for field in ("last_resort_count", "last_resort_moves"):
        assert field in VOLATILE_FIELDS and field not in BATON_PASSED_FIELDS
    s = make_state(
        t,
        [_spec(n, "Monferno", 30, ("Last Resort", "Tackle")), _spec(n, "Geodude", 30, ("Tackle",))],
        [_spec(n, "Geodude", 40, ("Defense Curl",))],
    )
    s = s._replace(ability=s.ability.at[:, :].set(AbilityId.NONE))
    s = s._replace(
        last_resort_count=s.last_resort_count.at[0].set(1),
        last_resort_moves=s.last_resort_moves.at[0].set(
            jnp.asarray([n.move("Tackle"), 0, 0, 0], jnp.int16)
        ),
    )
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 0), _hits())
    assert _history(out) == (0, []) and np.asarray(out.last_resort_moves[0]).tolist() == [0] * 4


def test_the_history_is_per_battler_and_indexed_by_the_battler(tables, names):
    """Doubles-ready (docs/doubles-state-design.md section 5): both fields lead with the battler
    axis, ``last_resort_update`` and ``last_resort_fails`` take the battler as an argument, and a
    write for side 1 leaves side 0's history alone."""
    from nuzline.engine.effects import last_resort_fails, last_resort_update
    from nuzline.engine.state import _BATTLER, _KEYS

    t, n = tables, names
    assert _KEYS["last_resort_count"][0] == _BATTLER
    assert _KEYS["last_resort_moves"][0] == (*_BATTLER, 4)
    s = _lr(t, n, used=("Tackle",))
    two = jnp.asarray([n.move("Defense Curl"), 5, 0, 0], s.moves.dtype)
    s = s._replace(moves=s.moves.at[1, 0].set(two))
    out = last_resort_update(s, 1, n.move("Defense Curl"), True)
    assert _history(out, 0) == (1, [n.move("Tackle")])
    assert _history(out, 1) == (1, [n.move("Defense Curl")])
    assert not bool(last_resort_fails(out, 1)) and bool(last_resort_fails(s, 1))
    # Last Resort itself, the cap and a repeat: nothing is written (battle_lib.c:3445-3455).
    same = last_resort_update(out, 1, MOVE_LAST_RESORT, True)
    assert _history(same, 1) == _history(out, 1)
    for other in (2, 3, 4, 6):
        out = last_resort_update(out, 1, other, True)
    assert _history(out, 1) == (4, [n.move("Defense Curl"), 2, 3, 4])
    assert _history(last_resort_update(out, 1, 6, True), 1) == _history(out, 1)


def test_last_resort_decomp_lines_are_as_cited():
    """The lines the ledger and the code cite for Last Resort (E3-B2-11): the script's order, the
    command's test, the recorder, its call at BeforeMove's tail, and the three zeroing sites."""
    if not _ME_DECOMP.exists():
        pytest.skip("decomp not present")
    effect = (_ME_DECOMP / "res/battle/scripts/effects/effect_script_0246.s").read_text("utf-8")
    body = [line.strip() for line in effect.splitlines() if line.strip()]
    assert body[1:6] == ["_000:", "TryLastResort _004", "CalcCrit", "CalcDamage", "End"]
    assert "MOVE_STATUS_FAILED" in body[-2]
    bs = (_ME_DECOMP / "src/battle/battle_script.c").read_text("utf-8").splitlines()
    assert "BtlCmd_TryLastResort" in bs[7566]
    assert "Battler_CountMoves" in bs[7570]
    assert "lastResortCount < numMoves - 1" in bs[7572] and "numMoves < 2" in bs[7573]
    assert "BattleScript_Iter(battleCtx, jumpOnFail)" in bs[7574]
    for number in (4514, 4876):  # Mimic and Sketch learning Last Resort
        assert "msgMoveTemp == MOVE_LAST_RESORT" in bs[number - 1], number
        assert "lastResortCount = 0" in bs[number], number
    assert "lastResortCount = 0" in bs[5339]  # BtlCmd_Transform
    lib = (_ME_DECOMP / "src/battle/battle_lib.c").read_text("utf-8").splitlines()
    assert "BattleSystem_UpdateLastResort" in lib[3442]
    assert "moveTemp == MOVE_LAST_RESORT" in lib[3444] and "lastResortCount == 4" in lib[3445]
    assert "lastResortMoves[i] == battleCtx->moveTemp" in lib[3451]
    assert "lastResortMoves[i] = battleCtx->moveTemp" in lib[3456]
    assert "lastResortCount++" in lib[3457]
    assert "Battler_CountMoves" in lib[3460] and "MOVE_NONE" in lib[3464]
    bcp = (_ME_DECOMP / "src/battle/battle_controller_player.c").read_text("utf-8").splitlines()
    assert "MOVE_STATUS_NO_EFFECTS" in bcp[3208] and "BattleSystem_UpdateLastResort" in bcp[3217]
    assert "battleCtx->moveTemp = MOVE_STRUGGLE" in bcp[1866]


# --- Feint (#223, E3-B2-13) --------------------------------------------------------------------
# effect_script_0223.s: TryFeint (:5; MOVE_STATUS_FAILED at :11-13), an ON_HIT indirect
# subscript_feint.s (:6), CalcCrit, CalcDamage (:7-8). BtlCmd_TryFeint (battle_script.c:7518-7528)
# fails the move unless the defender's turnFlags.protecting is set; subscript_feint.s:20 clears it
# after a hit that is not MOVE_STATUS_NO_EFFECTS (battle_lib.c:1524-1529). Feint is move 364, +2;
# Protect and Detect are +3, so in singles a successful Protect is always earlier in the turn.

MOVE_FEINT = 364


def _fe(
    t, n, foe=("Protect", "Tackle"), mine=("Feint", "Tackle"), foe_count=0, foe_species="Geodude"
):
    """Monferno (side 0, ``mine``) against a level 40 ``foe_species`` (``foe``), no abilities; the
    foe's Protect chain starts at ``foe_count``."""
    s = make_state(
        t,
        [_spec(n, "Monferno", 30, mine)],
        [_spec(n, foe_species, 40, foe)],
    )
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(1).at[1, 0, SPE].set(500))
    return s._replace(protect_count=s.protect_count.at[1].set(foe_count))


def _fego(t, n, s, me="Feint", foe="Protect", out=None):
    ones = jnp.ones(2, jnp.int8)
    out = out if out is not None else _hits()._replace(protect=ones, future_hit=ones)
    return assert_matches_reference(t, s, A(me, foe, s, n), out)


def _pending_future_sight(n, s, damage=50):
    """``s`` with a delayed hit of the player due on the foe at this turn's end, whose stored move
    is a Tackle. Future Sight and Doom Desire have no MOVE_FLAG_CAN_PROTECT (res/moves/*/data.json),
    so the end-of-turn Protect test (bcp.c:3007) never stops either and the flag Feint clears has
    no reader in a singles turn (the sheet's section 1); a stored Tackle, which a Protect does
    stop, is what makes the flag observable."""
    return s._replace(
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(damage),
        future_move=s.future_move.at[1].set(n.move("Tackle")),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
    )


def test_feint_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 223 (effect_script_0223.s) is EffectId.FEINT in both games and Feint (move 364,
    Normal physical, accuracy 100, PP 10, +2, no flags: it is not blocked by Protect, which is its
    point) the only move with it; vanilla 50 BP, Kaizo 80 BP. It is a plain hit that can fail,
    classed as Last Resort is (enumerate's two allowlists)."""
    from nuzline.engine.enumerate import INPUT_NEUTRAL_EFFECTS, TABLE_DAMAGE_EFFECTS

    assert DECOMP_EFFECT_TO_ID[223] == (EffectId.FEINT, (0, 0, 0))
    for t, nn, power in ((tables, names, 50), (kaizo_tables, kaizo_names, 80)):
        move = nn.move("Feint")
        assert move == MOVE_FEINT
        assert int(t.move_effect[move]) == EffectId.FEINT
        assert int((np.asarray(t.move_effect) == EffectId.FEINT).sum()) == 1
        assert int(t.move_power[move]) == power and int(t.move_accuracy[move]) == 100
        assert int(t.move_pp[move]) == 10 and int(t.move_type[move]) == nn.type("Normal")
        assert int(t.move_priority[move]) == 2 and int(t.move_category[move]) == 0
        assert not int(t.move_flags[move]) & (MoveFlag.PROTECT | MoveFlag.CONTACT)
        assert int(t.move_priority[nn.move("Protect")]) == 3
    assert EffectId.FEINT in INPUT_NEUTRAL_EFFECTS and EffectId.FEINT in TABLE_DAMAGE_EFFECTS


def test_feint_fails_against_a_foe_that_did_not_protect(tables, names):
    """TryFeint (:5, battle_script.c:7523): no turnFlags.protecting, MOVE_STATUS_FAILED: no damage,
    but the PP is spent (DecrementPP precedes the script) and the foe's Tackle still lands."""
    t, n = tables, names
    s = _fe(t, n)
    out = _fego(t, n, s, foe="Tackle")
    assert _foe_hp(out) == _foe_hp(s)
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_feint_hits_through_a_successful_protect(tables, names):
    """A Protect that succeeded earlier this turn sets turnFlags.protecting (battle_script.c:5174-
    5177): Feint (no MOVE_FLAG_CAN_PROTECT, so CheckMoveHitOverrides does not stop it, bcp.c:3007-
    3014) damages the foe, where a Tackle of the same Monferno is stopped. Detect is the same."""
    t, n = tables, names
    s = _fe(t, n, foe=("Protect", "Detect"))
    out = _fego(t, n, s)
    assert _foe_hp(out) < _foe_hp(s)
    assert int(out.protect_count[1]) == 1  # the Protect counted
    assert _foe_hp(_fego(t, n, s, me="Tackle")) == _foe_hp(s)
    assert _foe_hp(_fego(t, n, s, foe="Detect")) < _foe_hp(s)


def test_feint_fails_after_a_failed_protect_and_after_endure(tables, names):
    """A Protect whose chain roll fails (count 1, roll 0) sets no flag, so Feint fails; Endure sets
    turnFlags.enduring instead of protecting (battle_script.c:5174-5177), so Feint fails against
    it as well."""
    t, n = tables, names
    s = _fe(t, n, foe_count=1)
    failed = _fego(t, n, s, out=_hits()._replace(protect=jnp.zeros(2, jnp.int8)))
    assert _foe_hp(failed) == _foe_hp(s) and int(failed.protect_count[1]) == 0
    assert _foe_hp(_fego(t, n, s)) < _foe_hp(s)  # the same chain, the roll passing
    endure = _fe(t, n, foe=("Endure", "Tackle"))
    assert _foe_hp(_fego(t, n, endure, foe="Endure")) == _foe_hp(endure)


def test_a_feint_hit_clears_protecting_for_the_rest_of_the_turn(tables, names):
    """subscript_feint.s:20 clears the defender's flag, so a delayed hit due at this turn's end
    (CheckMoveHitOverrides, bcp.c:3005-3011; see ``_pending_future_sight``) lands on the foe that
    protected: the foe loses the Feint and the hit; with a Tackle the Protect stands and stops
    it."""
    t, n = tables, names
    s = _pending_future_sight(n, _fe(t, n))
    out = _fego(t, n, s)
    plain = _fego(t, n, _fe(t, n))
    assert _foe_hp(out) < _foe_hp(plain) < _foe_hp(s)
    assert _foe_hp(_fego(t, n, s, me="Tackle")) == _foe_hp(s)


def test_a_feint_that_does_not_hit_leaves_protecting_up(tables, names):
    """The clear is an ON_HIT subscript, run unless MOVE_STATUS_NO_EFFECTS (battle_lib.c:1524-
    1529): a Feint that misses (MOVE_STATUS_MISSED) and one that is ineffective against a Ghost
    (MOVE_STATUS_INEFFECTIVE) leave the Protect up, and the pending hit is still stopped."""
    t, n = tables, names
    s = _pending_future_sight(n, _fe(t, n))
    off = _hits()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8), protect=jnp.ones(2, jnp.int8))
    missed = _fego(t, n, s, out=off)
    assert _foe_hp(missed) == _foe_hp(s)
    ghost = _pending_future_sight(n, _fe(t, n, foe_species="Gastly"))
    assert _foe_hp(_fego(t, n, ghost)) == _foe_hp(ghost)


def test_shadow_force_clears_protecting_with_its_attack_turn(tables, names):
    """Shadow Force's script runs subscript_feint.s as its ON_HIT effect too (effect_script_0272.s:
    21): an attack turn that hits (a Power Herb skips the charge) goes through the Protect (no
    MOVE_FLAG_CAN_PROTECT) and clears the flag, so the pending hit lands; a Ghost-immune Normal
    foe cannot be told apart, so the Geodude is used. Its charge turn clears nothing."""
    t, n = tables, names
    herb = n.item("Power Herb")
    base = _fe(t, n, mine=("Shadow Force", "Tackle"))
    base = base._replace(item=base.item.at[0, 0].set(herb))
    pending = _pending_future_sight(n, base)
    out = _fego(t, n, pending, me="Shadow Force")
    alone = _fego(t, n, base, me="Shadow Force")
    assert _foe_hp(out) < _foe_hp(alone) < _foe_hp(base)
    charging = _pending_future_sight(n, _fe(t, n, mine=("Shadow Force", "Tackle")))
    assert _foe_hp(_fego(t, n, charging, me="Shadow Force")) == _foe_hp(charging)


def test_a_called_feint_reads_the_same_flag(tables, names):
    """Sleep Talk (priority 0) calls Feint, whose TryFeint reads the foe's turnFlags.protecting
    the same way: the Protect (+3) is earlier, so the called Feint hits; with no Protect it
    fails."""
    t, n = tables, names
    s = _fe(t, n, mine=("Sleep Talk", "Feint"))
    s = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP), status_counter=s.status_counter.at[0, 0].set(3)
    )
    out = _hits()._replace(protect=jnp.ones(2, jnp.int8), call=jnp.zeros(2, jnp.int16))
    assert _foe_hp(_fego(t, n, s, me="Sleep Talk", out=out)) < _foe_hp(s)
    assert _foe_hp(_fego(t, n, s, me="Sleep Talk", foe="Tackle", out=out)) == _foe_hp(s)


@pytest.mark.slow  # 157 s (fast tier: 60 s, 2 GB above a loaded step); run on the pod
def test_feint_enumeration_matches_the_brute_force(tables, names):
    """The exact enumeration (Feint a table-damage, input-neutral effect; the foe's chain roll a
    read of the Protect) equals the brute-force one, at chain 2 (the roll is drawn)."""
    from tests.engine.test_enumerate import assert_equals_bruteforce

    t, n = tables, names
    s = _fe(t, n, foe_count=2)
    assert_equals_bruteforce(t, s, A("Feint", "Protect", s, n))


def test_feint_takes_the_targets_flag_as_an_input(tables, names):
    """Doubles-ready (docs/doubles-state-design.md section 5, rules 1 and 5): ``_use_move`` takes
    ``protecting`` (``bool[B]``, every battler's flag; D3) as an argument and reads its defender's
    (``turnFlags[defender].protecting``, bcp.c:3007); a Feint hit zeroes the target's entry, never
    the mover's: with the flag given and no Protect in the turn the Feint hits; withdrawn, it
    fails."""
    import inspect

    from nuzline.engine import step as step_module

    assert "protecting" in inspect.signature(step_module._use_move).parameters
    t, n = tables, names
    s = _fe(t, n)
    out = _fego(t, n, s)
    assert _foe_hp(out) < _foe_hp(s) and int(out.protect_count[0]) == 0


def test_feint_decomp_lines_are_as_cited():
    """The lines the ledger and the code cite for Feint (E3-B2-13): the script, the command, the
    subscript's clear, and the shared ON_HIT trigger."""
    if not _ME_DECOMP.exists():
        pytest.skip("decomp not present")
    effect = (_ME_DECOMP / "res/battle/scripts/effects/effect_script_0223.s").read_text("utf-8")
    body = [line.strip() for line in effect.splitlines() if line.strip()]
    assert body[1:7] == [
        "_000:",
        "TryFeint _008",
        (
            "UpdateVar OPCODE_SET, BTLVAR_SIDE_EFFECT_FLAGS_INDIRECT, MOVE_SIDE_EFFECT_ON_HIT|"
            "MOVE_SUBSCRIPT_PTR_FEINT"
        ),
        "CalcCrit",
        "CalcDamage",
        "End",
    ]
    assert "MOVE_STATUS_FAILED" in body[-2]
    sub = (_ME_DECOMP / "res/battle/scripts/subscripts/subscript_feint.s").read_text("utf-8")
    assert "SetTurnFlag BTLSCR_DEFENDER, TURN_FLAG_PROTECTING, 0" in sub
    bs = (_ME_DECOMP / "src/battle/battle_script.c").read_text("utf-8").splitlines()
    assert "BtlCmd_TryFeint" in bs[7517] and "DEFENDER_TURN_FLAGS.protecting == FALSE" in bs[7522]
    assert "BattleScript_Iter(battleCtx, jumpOnFail)" in bs[7523]
    lib = (_ME_DECOMP / "src/battle/battle_lib.c").read_text("utf-8").splitlines()
    assert "BattleSystem_TriggerSecondaryEffect" in lib[1518]
    assert "MOVE_SIDE_EFFECT_ON_HIT" in lib[1523] and "MOVE_STATUS_NO_EFFECTS" in lib[1526]


# --- Defog (#258, E3-B2-14) and the Metronome close-out (#83) ------------------------------------
# effect_script_0258.s:5: a direct ON_HIT subscript_defog.s (``BattleSystem_TriggerPrimaryEffect``,
# battle_lib.c:1481-1491): the defender's evasion -1 through subscript_update_stat_stage.s (:19-20),
# then its side's Reflect, Light Screen, Mist, Safeguard, Spikes, Toxic Spikes and Stealth Rock and
# the field's deep fog cleared (:21-69).

_SIDE_FIELDS = (
    ("reflect", 5),
    ("light_screen", 5),
    ("mist", 5),
    ("safeguard", 5),
    ("spikes", 2),
    ("toxic_spikes", 1),
    ("stealth_rock", 1),
)


# Every condition but Mist, which stops the evasion drop (the tests that need a drop leave it out).
_NO_MIST = tuple(f for f in _SIDE_FIELDS if f[0] != "mist")


def _dg(t, n, *, fog=False, side=1, fields=_NO_MIST, mine=("Defog", "Tackle"), foe=("Tackle",)):
    """Monferno (side 0, Defog) against a Geodude (``foe``), no abilities, the Monferno faster;
    every condition of ``fields`` is up on ``side``'s side (its value), and the deep fog when
    ``fog``."""
    s = make_state(t, [_spec(n, "Monferno", 20, mine)], [_spec(n, "Geodude", 40, foe)])
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))
    for field, value in fields:
        s = s._replace(**{field: getattr(s, field).at[side].set(value)})
    if fog:
        s = with_weather(s, Weather.FOG)
    return s


def _slower(s):
    """``s`` with the Defog user (side 0) slower than the foe."""
    return s._replace(stats=s.stats.at[0, 0, SPE].set(1).at[1, 0, SPE].set(500))


def _dgo(t, n, s, me="Defog", foe="Tackle", out=None):
    return assert_matches_reference(t, s, A(me, foe, s, n), out if out is not None else _hits())


def _no_conditions(s, side):
    return all(int(getattr(s, field)[side]) == 0 for field, _ in _SIDE_FIELDS)


def test_defog_is_mapped_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """Effect 258 (effect_script_0258.s) is EffectId.DEFOG with the evasion drop's params in both
    games and Defog (move 432: Flying status, accuracy 0, PROTECT and MIRROR_MOVE) the only move
    with it; PP 15 in vanilla, 1 in Kaizo (moves.json). It is a non-damaging clause effect: the
    foe's earlier Defog can clear the fog Weather Ball and SolarBeam read."""
    assert DECOMP_EFFECT_TO_ID[258] == (EffectId.DEFOG, (Stat.EVA, -1, 0))
    for t, nn, pp in ((tables, names, 15), (kaizo_tables, kaizo_names, 1)):
        move = nn.move("Defog")
        assert move == 432
        assert int(t.move_effect[move]) == EffectId.DEFOG
        assert int((np.asarray(t.move_effect) == EffectId.DEFOG).sum()) == 1
        assert int(t.move_accuracy[move]) == 0 and int(t.move_pp[move]) == pp
        assert int(t.move_type[move]) == nn.type("Flying") and int(t.move_category[move]) == 2
        flags = int(t.move_flags[move])
        assert flags & MoveFlag.PROTECT and flags & MoveFlag.MIRROR_MOVE
        assert not flags & MoveFlag.NO_METRONOME
        assert tuple(int(x) for x in np.asarray(t.move_param[move])) == (Stat.EVA, -1, 0)
    assert EffectId.DEFOG in INPUT_CLAUSE_EFFECTS and EffectId.DEFOG in NON_TABLE_DAMAGE_EFFECTS


def test_defog_lowers_evasion_and_clears_the_targets_side_only(tables, names):
    """subscript_defog.s:19-69: the foe's evasion -1 and its side's seven conditions and hazards
    gone; the user's own side keeps every one (the clears name the defender's side) and the
    user's evasion is unchanged."""
    t, n = tables, names
    theirs = _dgo(t, n, _dg(t, n))
    assert _stage(theirs, 1, Stat.EVA) == -1 and _stage(theirs, 0, Stat.EVA) == 0
    assert _no_conditions(theirs, 1)
    out = _dgo(t, n, _dg(t, n, side=0, fields=_SIDE_FIELDS))
    assert _stage(out, 1, Stat.EVA) == -1
    # The turn's pass counts a screen down; the hazards (layers) stay as they were.
    for field, value in _SIDE_FIELDS[4:]:
        assert int(getattr(out, field)[0]) == value, field
    for field, _ in _SIDE_FIELDS[:4]:
        assert int(getattr(out, field)[0]) > 0, field


def test_mist_stops_the_drop_but_is_cleared_after_it(tables, names):
    """BtlCmd_ChangeStatStage (battle_script.c:2877-2882) meets Mist before the subscript clears it
    (:33-36): the evasion stays 0 and the Mist is gone."""
    t, n = tables, names
    out = _dgo(t, n, _dg(t, n, fields=(("mist", 5),)))
    assert _stage(out, 1, Stat.EVA) == 0 and int(out.mist[1]) == 0
    assert _stage(_dgo(t, n, _dg(t, n, fields=())), 1, Stat.EVA) == -1


def test_clear_body_and_stage_minimum_stop_the_drop_not_the_clears(tables, names):
    """The drop is stopped by the target's Clear Body or White Smoke as the user sees it (:2883-
    2884: a Mold Breaker user pierces it) and by a stage already at -6 (:2904); the subscript goes
    on and clears the side either way."""
    t, n = tables, names
    s = _dg(t, n)
    for ability in (AbilityId.CLEAR_BODY, AbilityId.WHITE_SMOKE):
        held = s._replace(ability=s.ability.at[1, 0].set(ability))
        out = _dgo(t, n, held)
        assert _stage(out, 1, Stat.EVA) == 0 and _no_conditions(out, 1), ability
        pierced = held._replace(ability=held.ability.at[0, 0].set(AbilityId.MOLD_BREAKER))
        assert _stage(_dgo(t, n, pierced), 1, Stat.EVA) == -1, ability
    low = s._replace(stages=s.stages.at[1, Stat.EVA].set(-6))
    out = _dgo(t, n, low)
    assert _stage(out, 1, Stat.EVA) == -6 and _no_conditions(out, 1)


def test_defog_behind_a_substitute_fails_but_still_clears(tables, names):
    """subscript_update_stat_stage.s:43-45: a drop stopped by the Substitute sets MOVE_STATUS_FAILED
    (``defog_stage_fails``), yet the clears run (subscript_defog.s:21-69). With a condition to clear
    the attack message was shown (:5-13: movePrev is Defog, the slower mover's); with nothing to
    clear CheckMoveFailure's subscript_missed.s:5 shows it (bcp.c:3311-3327, review b2 #1), so
    movePrev is Defog too; Mist stops the drop first (no failure, the message of the stage
    subscript shows)."""
    t, n = tables, names

    def behind(s):
        return s._replace(substitute_hp=s.substitute_hp.at[1].set(30))

    out = _dgo(t, n, behind(_slower(_dg(t, n))))
    assert _stage(out, 1, Stat.EVA) == 0 and _no_conditions(out, 1)
    assert int(out.move_prev) == n.move("Defog")
    bare = behind(_slower(_dg(t, n, fields=())))
    assert int(_dgo(t, n, bare).move_prev) == n.move("Defog")  # subscript_missed.s:5
    open_ = bare._replace(substitute_hp=bare.substitute_hp.at[1].set(0))
    assert int(_dgo(t, n, open_).move_prev) == n.move("Defog")
    misted = behind(_slower(_dg(t, n, fields=(("mist", 5),))))
    assert int(_dgo(t, n, misted).move_prev) == n.move("Defog")


def test_defog_clears_the_deep_fog(tables, names):
    """subscript_defog.s:64-65: FIELD_CONDITION_DEEP_FOG off, the field has no weather; another
    weather is not touched, nor are the other bits of a mask that holds fog and a rain bit (FCM
    3.1: only bit 15 goes, ``field.defog_clear``)."""
    t, n = tables, names
    out = _dgo(t, n, _dg(t, n, fog=True, fields=()))
    assert int(weather_of(out)) == Weather.NONE
    assert int(out.field_mask) & int(FieldCondition.DEEP_FOG) == 0
    rain = with_weather(_dg(t, n, fields=()), Weather.RAIN)
    assert int(weather_of(_dgo(t, n, rain))) == Weather.RAIN
    fog = _dg(t, n, fog=True, fields=())
    both = fog._replace(field_mask=fog.field_mask | int(FieldCondition.RAINING_PERM))
    out = _dgo(t, n, both)
    assert int(out.field_mask) & int(FieldCondition.WEATHER) == int(FieldCondition.RAINING_PERM)


def test_defog_is_seen_by_hit_thresholds_and_the_damage_clauses(tables, names):
    """A faster foe's Defog clears the fog before the side's check: Rock Throw's 90 is 54 under
    deep fog (x6/10, bcp.c:2935-2937) and 90 after, so the law has a second class
    (``outcome._threshold_grid``); the foe's Defog under fog is a clause of
    ``damage_inputs_can_change`` (Weather Ball, SolarBeam), not without fog."""
    from nuzline.engine.enumerate import damage_inputs_can_change
    from tests.engine.test_accuracy_classes import hit_law

    t, n = tables, names
    s = roark_state(t, n, player_moves=["Rock Throw"], opp_moves=["Defog"], opp_faster=True)
    fog = with_weather(s, Weather.FOG)
    acts = A("Rock Throw", "Defog", fog, n)
    assert hit_law(t, fog, acts, 0) == {0: 10, 1: 54, 55: 36}
    assert hit_law(t, s, acts, 0) == {0: 10, 1: 90}
    # The Defog user is side 0 here, going before side 1's hit (the Lucky Chant case's layout).
    d = roark_state(t, n, player_moves=["Defog"], opp_moves=["Rock Throw"])
    dfog = with_weather(d, Weather.FOG)
    dacts = A("Defog", "Rock Throw", dfog, n)
    foe_first = jnp.array([False, True])
    assert bool(damage_inputs_can_change(t, dfog, dacts, foe_first)[1, 0])
    assert not bool(damage_inputs_can_change(t, d, dacts, foe_first)[1, 0])


def test_defog_takes_its_target_side_as_an_argument(tables, names):
    """Doubles-ready (docs/doubles-state-design.md section 5, rules 1, 2 and 6): the failure
    helper takes the defender's side, and the clears index ``side_of(target)``
    for each field, never a computed foe: a Defog whose user's side holds every condition too
    clears the target's only."""
    import inspect

    from nuzline.engine import effects as effects_module

    assert "target_side" in inspect.signature(effects_module.defog_stage_fails).parameters
    t, n = tables, names
    both = _dg(t, n)
    for field, value in _SIDE_FIELDS:
        both = both._replace(**{field: getattr(both, field).at[0].set(value)})
    out = _dgo(t, n, both)
    assert _no_conditions(out, 1)
    assert int(out.spikes[0]) == 2 and int(out.stealth_rock[0]) == 1


def test_defog_decomp_lines_are_as_cited():
    """The lines the ledger and the code cite for Defog (E3-B2-14): the script, the clears in order,
    the stat change's failure branch and the primary-effect trigger."""
    if not _ME_DECOMP.exists():
        pytest.skip("decomp not present")
    effect = (_ME_DECOMP / "res/battle/scripts/effects/effect_script_0258.s").read_text("utf-8")
    assert (
        "UpdateVar OPCODE_SET, BTLVAR_SIDE_EFFECT_FLAGS_DIRECT, MOVE_SIDE_EFFECT_ON_HIT|"
        "MOVE_SIDE_EFFECT_TO_DEFENDER|MOVE_SUBSCRIPT_PTR_DEFOG"
    ) in effect
    sub = (_ME_DECOMP / "res/battle/scripts/subscripts/subscript_defog.s").read_text("utf-8")
    cleared = [
        line.split(",")[2].strip() for line in sub.splitlines() if "CHECK_SIDE_COND_CLEAR" in line
    ]
    assert cleared == [
        "SIDE_COND_REFLECT_TURNS",
        "SIDE_COND_LIGHT_SCREEN_TURNS",
        "SIDE_COND_MIST_TURNS",
        "SIDE_COND_SAFEGUARD_TURNS",
        "SIDE_COND_SPIKES_LAYERS",
        "SIDE_COND_TOXIC_SPIKES_LAYERS",
    ]
    assert "BTLVAR_SIDE_CONDITIONS_DEFENDER, SIDE_CONDITION_STEALTH_ROCK" in sub
    assert "UpdateVar OPCODE_FLAG_OFF, BTLVAR_FIELD_CONDITIONS, FIELD_CONDITION_DEEP_FOG" in sub
    stat_call = sub.index("BATTLE_SUBSCRIPT_UPDATE_STAT_STAGE")
    assert stat_call < sub.index("SIDE_COND_REFLECT_TURNS, _065")
    update = (_ME_DECOMP / "res/battle/scripts/subscripts/subscript_update_stat_stage.s").read_text(
        "utf-8"
    )
    assert "ChangeStatStage _041, _059, _060" in update
    assert "BTLVAR_MOVE_STATUS_FLAGS, MOVE_STATUS_FAILED" in update.split("_060:")[1]
    lib = (_ME_DECOMP / "src/battle/battle_lib.c").read_text("utf-8").splitlines()
    assert "BattleSystem_TriggerPrimaryEffect" in lib[1480]
    assert "MOVE_SIDE_EFFECT_ON_HIT" in lib[1484] and "MOVE_STATUS_NO_EFFECTS" in lib[1488]


# The Metronome close-out (#83, E3-B2-14): no code of Metronome's own changes; every move it can
# call in Kaizo is modelled, so coverage no longer refuses its holder.


@pytest.mark.fullset
def test_every_kaizo_metronome_call_is_modelled(kaizo_tables, kaizo_names):
    """BtlCmd_Metronome (battle_script.c:4535-4562) calls any of ids 1-467 that
    ``Move_CanBeMetronomed`` allows and the user does not know: ``coverage.move_effect_line``
    refuses a holder while one of them is refused, so with Kaizo's table every callable move
    has a modelled effect and a Metronome holder passes the per-move check."""
    from nuzline.engine.coverage import (
        IMPLEMENTED_EFFECTS,
        Support,
        metronome_calls,
        move_effect_line,
    )

    t, nn = kaizo_tables, kaizo_names
    spec = _spec(nn, "Clefairy", 30, ("Metronome",))
    calls = metronome_calls(t, spec)
    assert len(calls) > 400
    effect = np.asarray(t.move_effect)
    unmodelled = [
        nn.move_names[m]
        for m in calls
        if IMPLEMENTED_EFFECTS.get(EffectId(int(effect[m])), Support.NONE) == Support.NONE
    ]
    assert unmodelled == []
    support, _ = move_effect_line(t, nn, nn.move("Metronome"), 0, spec.moves)
    assert support != Support.NONE


@pytest.mark.fullset
def test_a_kaizo_metronome_holder_passes_the_party_check(kaizo_tables, kaizo_names):
    """The per-party coverage check lists nothing for a Kaizo Metronome holder."""
    from nuzline.engine.coverage import _check_party, _Collector

    t, nn = kaizo_tables, kaizo_names
    c = _Collector()
    _check_party(c, t, nn, "Player", [_spec(nn, "Clefairy", 30, ("Metronome", "Tackle"))])
    assert not any("Metronome" in line for line in c.report.missing)


def test_the_laws_keep_a_struggle_fixed_at_the_selection_through_a_learn_pause(tables, names):
    """Review b2 #7: Struggle is fixed at the selection (``turnFlags.struggling``, bcp.c:433; a
    locked battler's ``ppDecremented``, :335), so a learn prompt that refills the PP of a
    battler whose action is still to come does not undo it: ``step._execute_actions`` reads
    ``ts_struggling`` / ``ts_locked`` / ``ts_move`` at a pause, and ``order.selected_move`` (every
    law's and the sampler's read, ``outcome.used_move``) now does the same; a free slot is
    re-read (FightCommand, bcp.c:1889-1895). Unpaused, the live PP decide."""
    from nuzline.engine.levelup import LEARN_MOVE_END
    from nuzline.engine.order import selected_move, used_move

    t, n = tables, names
    s = roark_state(t, n, player_moves=["Tackle"], opp_moves=["Tackle"])
    acts = A("Tackle", "Tackle", s, n)
    tackle = n.move("Tackle")
    assert [int(x) for x in selected_move(t, s, 0, acts[0])] == [tackle, 0]
    paused = s._replace(
        learn_site=jnp.asarray(LEARN_MOVE_END, s.learn_site.dtype),
        ts_actions=jnp.asarray(acts, s.ts_actions.dtype),
        ts_move=s.ts_move.at[0].set(MOVE_STRUGGLE),
        ts_struggling=s.ts_struggling.at[0].set(1),
    )
    assert [int(x) for x in selected_move(t, paused, 0, acts[0])] == [MOVE_STRUGGLE, 1]
    assert int(used_move(t, paused, 0, acts[0])) == MOVE_STRUGGLE
    # A lock fixed at the selection runs its move (ts_move), not the slot's.
    growl = n.move("Growl")
    locked = paused._replace(
        ts_struggling=paused.ts_struggling.at[0].set(0),
        ts_locked=paused.ts_locked.at[0].set(1),
        ts_move=paused.ts_move.at[0].set(growl),
    )
    assert [int(x) for x in selected_move(t, locked, 0, acts[0])] == [growl, 0]
    free = locked._replace(ts_locked=locked.ts_locked.at[0].set(0))
    assert [int(x) for x in selected_move(t, free, 0, acts[0])] == [tackle, 0]


def test_metronome_calling_defog_runs_defogs_script_with_metronomes_pp(kaizo_tables, kaizo_names):
    """Metronome's call of Defog (rank ``call % count`` among ``metronome_calls``) runs
    subscript_defog.s on the foe's side; the PP stays Metronome's (``effects.metronome_update``);
    both engines agree."""
    from nuzline.engine.coverage import metronome_calls

    t, nn = kaizo_tables, kaizo_names
    spec = _spec(nn, "Clefairy", 30, ("Metronome", "Tackle"))
    s = make_state(t, [spec], [_spec(nn, "Geodude", 40, ("Tackle",))])
    s = s._replace(ability=s.ability.at[:, 0].set(AbilityId.NONE))
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(500).at[1, 0, SPE].set(1))
    s = s._replace(reflect=s.reflect.at[1].set(5), stealth_rock=s.stealth_rock.at[1].set(1))
    rank = metronome_calls(t, spec).index(nn.move("Defog"))
    out = _hits()._replace(call=jnp.asarray([rank, 0], jnp.int16))
    after = assert_matches_reference(t, s, A("Metronome", "Tackle", s, nn), out)
    assert int(after.reflect[1]) == 0 and int(after.stealth_rock[1]) == 0
    assert int(after.stages[1, Stat.EVA]) == -1
    assert int(after.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1


# --- Gyro Ball's AI case and monSpeedValues (#360 / effect 219, E3-B2-15) --------------------------
# battle_lib.c:1372-1373: BattleSystem_CompareBattlerSpeed stores both speeds in
# BattleContext.monSpeedValues, which BtlCmd_CalcGyroBallPower (battle_script.c:7123-7131) and
# TrainerAI_CalcDamage's Gyro Ball case (trainer_ai.c:2995-3003) read. Its writers: the sort after
# the switch-in effects at the battle start (bcp.c:263) and, in BranchActions (bcp.c:888), before
# each action and once more after the last one; the AI's own compares write it inside a decision
# (tests/ai/test_kaizo_policy.py). ``BattleState.speed_rec``; each case against the reference.


def _speeds(t, s):
    return [int(effective_speed(t, s, b)) for b in range(2)]


def _rec(s):
    return [int(x) for x in s.speed_rec]


def _sp(t, n, species, *moves, level=40, ability_slot=0):
    """One Pokemon of ``species`` at ``level`` with ``moves`` (no item)."""
    return PokemonSpec(
        species=n.species(species),
        level=level,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(n.move(m) for m in moves) + (0,) * (4 - len(moves)),
        item=0,
        ability_slot=ability_slot,
        gender=0,
        friendship=70,
    )


def test_monspeedvalues_is_speed_rec_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """The field exists per battler and ``battle_start`` fills it with the battlers' speeds."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        s = make_state(t, [_sp(t, n, "Monferno", "Tackle")], [_sp(t, n, "Geodude", "Tackle")])
        assert s.speed_rec.shape == (2,) and s.speed_rec.dtype == jnp.int32
        assert _rec(s) == _speeds(t, s) and min(_rec(s)) > 0


def test_the_battle_start_record_follows_the_weather_and_ability(tables, names):
    """bcp.c:263 sorts after the switch-in effects: a Swift Swim lead in the start weather rain
    has its speed doubled in the record (CompareBattlerSpeed :1228-1238)."""
    t, n = tables, names
    mine, foe = _sp(t, n, "Monferno", "Tackle"), _sp(t, n, "Goldeen", "Tackle")
    s = make_state(t, [mine], [foe], start_condition=Weather.RAIN)
    assert int(s.ability[1, 0]) == AbilityId.SWIFT_SWIM
    plain = make_state(t, [mine], [foe])
    assert _rec(s)[1] == 2 * _rec(plain)[1] and _rec(s)[0] == _rec(plain)[0]
    assert _rec(s) == _speeds(t, s)


def test_a_turn_ends_with_the_speeds_of_the_last_sort(tables, names):
    """Speed Boost raises the foe's Speed at the end of the turn, after the sort that follows the
    last action (bcp.c:888-891): the record still holds the speed before it; the next turn's
    sorts overwrite it."""
    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Monferno", "Tackle")],
        [_sp(t, n, "Ninjask", "Tackle", level=30)],
    )
    assert int(s.ability[1, 0]) == AbilityId.SPEED_BOOST
    before = _speeds(t, s)
    s1 = _go2(t, n, s)
    assert _stage(s1, 1, Stat.SPE) == 1  # the end-of-turn boost happened
    assert _rec(s1) == before and _speeds(t, s1)[1] > before[1]
    s2 = _go2(t, n, s1)
    assert _rec(s2)[1] == _speeds(t, s1)[1] and _stage(s2, 1, Stat.SPE) == 2


def _go2(t, n, s, me="Tackle", foe="Tackle"):
    return assert_matches_reference(t, s, A(me, foe, s, n), _hits())


def test_a_change_during_the_turn_is_in_the_record(tables, names):
    """Agility (+2 Speed) is the first action: the sorts after it (before the second action and
    after the last, bcp.c:888) see the new speed."""
    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Monferno", "Agility", "Tackle", level=30)],
        [_sp(t, n, "Geodude", "Tackle", level=30)],
    )
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(300).at[1, 0, SPE].set(20))
    s1 = _go2(t, n, s, me="Agility")
    assert _stage(s1, 0, Stat.SPE) == 2
    assert _rec(s1) == _speeds(t, s1) and _rec(s1)[0] > _speeds(t, s)[0]


def test_a_faint_stops_the_writes_and_a_replacement_keeps_its_predecessors(tables, names):
    """CompareBattlerSpeed writes nothing when a battler has no HP (battle_lib.c:1204-1211): the
    sorts after the knock-out leave the record at the speeds of the sort before the fatal action,
    and the newcomer's slot holds the fainted Pokemon's value until the next sort."""
    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Monferno", "Tackle", level=50)],
        [_sp(t, n, "Geodude", "Tackle", level=5), _sp(t, n, "Ninjask", "Tackle", level=60)],
    )
    s = s._replace(stats=s.stats.at[0, 0, SPE].set(300).at[1, 0, SPE].set(20))
    before = _speeds(t, s)
    s1 = _go2(t, n, s)
    assert int(s1.hp[1, 0]) == 0 and int(s1.hp[0, 0]) > 0
    assert _rec(s1) == before
    s2 = send_in(t, s1, 1, 1)
    assert int(s2.active[1]) == 1
    assert _rec(s2) == before and _speeds(t, s2)[1] != before[1]
    s3 = _go2(t, n, s2)  # the next turn's sorts write the newcomer's speed
    assert int(s3.hp[0, 0]) > 0 and int(s3.hp[1, 1]) > 0
    assert _rec(s3)[1] == _speeds(t, s2)[1] and _rec(s3)[0] == _speeds(t, s2)[0]


def test_the_record_after_a_switch_turn_is_the_newcomers(tables, names):
    """BranchActions sorts before every action, a switch included, and after the last one
    (bcp.c:888): after a turn with a switch the record holds the newcomer's speed."""
    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Monferno", "Tackle"), _sp(t, n, "Ninjask", "Tackle")],
        [_sp(t, n, "Geodude", "Tackle", level=30)],
    )
    before = _speeds(t, s)
    s1 = assert_matches_reference(t, s, jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32), _hits())
    assert int(s1.active[0]) == 1
    assert _rec(s1) == _speeds(t, s1) and _rec(s1)[0] != before[0]


def test_a_weather_war_at_the_start_writes_the_record_when_it_settles(tables, names):
    """The battle start's sort (bcp.c:263) read the weather the engine's order leaves; a tie left
    open in ``war_swaps`` is settled at the root and the record is written again with the final
    weather (``step.resolve_war``), at turn 0 only: a replacement's war has no sort after it."""
    from nuzline.engine.step import resolve_war

    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Groudon", "Tackle", level=50)],
        [_sp(t, n, "Kyogre", "Tackle", level=50)],
    )
    assert int(s.war_swaps) != 0
    swim = AbilityId.SWIFT_SWIM
    s = s._replace(ability=s.ability.at[1, 0].set(swim))  # a Swift Swim speed reads the weather
    seen = {}
    for tie in (0, 1):
        after = resolve_war(s, tie, t)
        seen[tie] = _rec(after)
        assert _rec(after) == _speeds(t, after)
        assert int(after.war_swaps) == 0
    assert seen[0][1] != seen[1][1]  # rain doubles the Swift Swim speed, sun does not
    later = resolve_war(s._replace(turn=jnp.asarray(1, s.turn.dtype)), 0, t)
    assert _rec(later) == _rec(s)
    assert _rec(resolve_war(s, 0)) == _rec(s)  # no tables: nothing to compute with


# --- #251 Aqua Ring (E3-B2-20) ------------------------------------------------------------------


def test_baton_pass_keeps_aqua_ring_and_the_newcomer_heals_at_the_turns_end(tables, names):
    """MOVE_EFFECT_AQUA_RING is in MOVE_EFFECT_BATON_PASSED (moves.h:46-48), so battle_lib.c:2079
    keeps it on the replacement, whose end-of-turn ``MON_COND_CHECK_STATE_AQUA_RING`` (bcp.c:1332)
    heals ``Divide(maxHP, 16)``; a plain switch clears the bit (:2071)."""
    from nuzline.engine.state import BATON_PASSED_FIELDS

    assert "aqua_ring" in BATON_PASSED_FIELDS
    t, n = tables, names
    s = make_state(
        t,
        [_sp(t, n, "Ninjask", "Baton Pass", level=30), _sp(t, n, "Monferno", "Tackle", level=20)],
        [_sp(t, n, "Geodude", "Tackle", level=20)],
    )
    s = s.replace_hp(0, 1, 20)

    def baton(state):
        acts = A("Baton Pass", "Tackle", state, n)
        hits = empty_outcome()._replace(hit=jnp.ones((2, MAX_HITS), jnp.int8))
        paused = assert_matches_reference(t, state, acts, hits)
        assert int(paused.pivot_battler) == 1
        resume = np.asarray(paused.ts_actions).astype(int).tolist()
        resume[0] = MOVE_ACTIONS + 1
        return assert_matches_reference(t, paused, tuple(resume), empty_outcome())

    ringed = baton(s._replace(aqua_ring=s.aqua_ring.at[0].set(1)))
    control = baton(s)
    assert int(ringed.active[0]) == 1 and ringed.aqua_ring.tolist() == [1, 0]
    assert control.aqua_ring.tolist() == [0, 0]
    heal = max(int(s.max_hp[0, 1]) // 16, 1)
    assert int(ringed.hp[0, 1]) - int(control.hp[0, 1]) == heal
    plain = s._replace(aqua_ring=s.aqua_ring.at[0].set(1))
    switch = jnp.array([MOVE_ACTIONS + 1, n.move_slot(s, 1, "Tackle")], jnp.int32)
    assert assert_matches_reference(t, plain, switch, empty_outcome()).aqua_ring.tolist() == [0, 0]


# --- E3-B2-21: the mid-turn learn prompt ----------------------------------------------------------
# The payout runs the whole GetExpTask, every level's learn prompt included, inside itself
# (battle_script.c:9882; MoveEnd bcp.c:3968 then the tail :3973, the field conditions :941, a
# battler's turn-end block :1296): ``step`` pauses at the prompt (``learn_site``), the answer is
# ``levelup.learn``, and the same ``step`` resumes the turn. Both engines agree on every field at
# the pause and after the resume (``assert_matches_reference``).


def _learner_moves():
    return ("Bubble", "Roar", "Tackle", "Water Gun")


def _prinplup(t, n, level=18):
    """A Prinplup one exp short of the next level (BubbleBeam at 19, vanilla learnset)."""
    from tests.engine.test_levelup import _base, _mon

    return _mon(
        n,
        "Prinplup",
        level,
        moves=_learner_moves(),
        ivs=(3,) * 6,
        exp=_base(t, n, "Prinplup", level + 1) - 1,
    )


def _learn_state(t, n, opp, **kw):
    from tests.engine.test_levelup import _mon

    return make_state(t, [_prinplup(t, n), _mon(n, "Machop", 20)], opp, **kw)


def _resume_learn(t, state, option=None):
    """The state answered (``option``: give up when None) and resumed through both engines."""
    from nuzline.engine.levelup import GIVE_UP, learn

    answered = jax.jit(learn)(t, state, jnp.int32(GIVE_UP if option is None else option))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    return answered, assert_matches_reference(t, answered, jnp.zeros(2, jnp.int32), miss)


def test_midturn_learn_pauses_at_the_move_end_payout(tables, names):
    # Prinplup's Bubble KOs the 1-HP Geodude: the payout levels it to 19 and the BubbleBeam prompt
    # comes inside MoveEnd (bcp.c:3968), before Geodude's own action could be skipped or the
    # next action run. The pause holds the turn at the action's position (ts_cursor).
    from nuzline.engine.levelup import LEARN_MOVE_END, needs_learn
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = _learn_state(tables, names, opp).replace_hp(1, 0, 1)
    paused = assert_matches_reference(tables, s, A("Bubble", "Tackle", s, names), empty_outcome())
    assert int(paused.learn_site) == LEARN_MOVE_END and bool(needs_learn(paused))
    assert int(paused.level[0, 0]) == 19 and int(paused.hp[1, 0]) == 0
    _, done = _resume_learn(tables, paused)
    assert int(done.learn_site) == 0 and int(done.turn) == int(s.turn) + 1
    assert not bool(needs_learn(done))


def test_midturn_learn_field_payout_pauses_before_gravity_and_the_turn_end(tables, names):
    # Hail faints the 1-HP Geodude in the field conditions (bcp.c:941): the payout and its prompt
    # come there, with the turn end's order fixed at the last BranchActions (ts_eot, :888-891).
    from nuzline.engine.levelup import LEARN_FIELD
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = _learn_state(tables, names, opp, start_condition=Weather.HAIL).replace_hp(1, 0, 1)
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    paused = assert_matches_reference(tables, s, A("Tackle", "Tackle", s, names), miss)
    assert int(paused.learn_site) == LEARN_FIELD and int(paused.level[0, 0]) == 19
    _, done = _resume_learn(tables, paused)
    assert int(done.learn_site) == 0


def test_midturn_learn_field_prompt_with_two_weather_groups_follows_the_later_chip(tables, names):
    # A corrupted mask with sand and hail set (FCM 3.1): the SANDSTORM state's chip faints the 1-HP
    # Bidoof and the payout levels Prinplup at the loop head (bcp.c:938-944); the HAILING state then
    # chips Prinplup again on its new max HP before the pause (ledger E3-B2-21 ruling L4: the
    # game's prompt between the two states gives the same state in singles).
    from nuzline.engine.ids import FieldCondition
    from nuzline.engine.levelup import LEARN_FIELD, needs_learn
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Bidoof", 12), _mon(names, "Bonsly", 15)]
    s = _learn_state(tables, names, opp).replace_hp(1, 0, 1)
    mask = int(FieldCondition.SANDSTORM_PERM | FieldCondition.HAILING_PERM)
    s = s._replace(field_mask=jnp.asarray(mask, s.field_mask.dtype))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    paused = assert_matches_reference(tables, s, A("Tackle", "Tackle", s, names), miss)
    assert int(paused.learn_site) == LEARN_FIELD and bool(needs_learn(paused))
    assert int(paused.level[0, 0]) == 19 and int(paused.hp[1, 0]) == 0
    hp0, max0, max1 = int(s.hp[0, 0]), int(s.max_hp[0, 0]), int(paused.max_hp[0, 0])
    assert int(paused.hp[0, 0]) == hp0 - max0 // 16 + (max1 - max0) - max1 // 16
    _, done = _resume_learn(tables, paused)
    assert int(done.learn_site) == 0


def test_midturn_learn_battler_block_payout_pauses_in_the_turn_end_order(tables, names):
    # Burn faints the 1-HP Geodude in its end-of-turn block (bcp.c:1296): the prompt comes after
    # that block, and the resume runs the other battlers' blocks in the stored order.
    from nuzline.engine.levelup import LEARN_MON
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    s = _learn_state(tables, names, opp).replace_hp(1, 0, 1)
    s = s._replace(status=s.status.at[1, 0].set(int(Status.BURN)))
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    paused = assert_matches_reference(tables, s, A("Tackle", "Tackle", s, names), miss)
    assert int(paused.learn_site) == LEARN_MON and int(paused.level[0, 0]) == 19
    _, done = _resume_learn(tables, paused)
    assert int(done.learn_site) == 0


@pytest.mark.parametrize("answer", ["give up", "replace slot 3"])
def test_midturn_learn_pending_action_rereads_the_replaced_slot(tables, names, answer):
    # Geodude is faster and Selfdestructs; the payout levels Prinplup before its selected Water
    # Gun (slot 3) runs. FightCommand reads moves[moveSlot] when the action runs (bcp.c:1889-1895):
    # replacing slot 3 runs BubbleBeam and spends its PP; giving up keeps Water Gun.
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Geodude", 12, moves=("Selfdestruct",)), _mon(names, "Bonsly", 15)]
    s = _learn_state(tables, names, opp)
    s = s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(int(s.stats[0, 0, Stat.SPE - 1]) + 1))
    paused = assert_matches_reference(
        tables, s, A("Water Gun", "Selfdestruct", s, names), empty_outcome()
    )
    assert int(paused.learn_site) == 1 and int(paused.ts_actions[0]) == 3
    bubble_beam = names.move("BubbleBeam")
    answered, done = _resume_learn(tables, paused, None if answer == "give up" else 3)
    if answer == "give up":
        assert int(done.moves[0, 0, 3]) == names.move("Water Gun")
    else:
        assert int(done.moves[0, 0, 3]) == bubble_beam
        assert int(done.pp[0, 0, 3]) < int(answered.pp[0, 0, 3])


def test_midturn_learn_resume_starts_the_next_action_at_power_mul_10(tables, names):
    # Geodude is faster and Selfdestructs; the payout pauses Prinplup's MoveEnd at the BubbleBeam
    # prompt. The resume runs that MoveEnd's tail, whose BattleContext_Init sets powerMul to 10
    # (bcp.c:3998, battle_lib.c:1980); Prinplup's Baton Pass then stops at its party list, where
    # the foe's replacement AI reads that powerMul (``ts_pick_mul``, battle_script.c:6928 is the
    # only 20).
    from nuzline.engine.levelup import LEARN_MOVE_END
    from tests.engine.test_levelup import _base, _mon

    learner = _mon(
        names,
        "Prinplup",
        18,
        moves=("Bubble", "Roar", "Tackle", "Baton Pass"),
        ivs=(3,) * 6,
        exp=_base(tables, names, "Prinplup", 19) - 1,
    )
    opp = [_mon(names, "Geodude", 12, moves=("Selfdestruct",)), _mon(names, "Bonsly", 15)]
    s = make_state(tables, [learner, _mon(names, "Machop", 20)], opp)
    s = s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(int(s.stats[0, 0, Stat.SPE - 1]) + 1))
    paused = assert_matches_reference(
        tables, s, A("Baton Pass", "Selfdestruct", s, names), empty_outcome()
    )
    assert int(paused.learn_site) == LEARN_MOVE_END and int(paused.hp[1, 0]) == 0
    _, listed = _resume_learn(tables, paused)
    assert int(listed.pivot_battler) == 1 and int(listed.learn_site) == 0
    assert int(listed.ts_pick_mul) == 10


def test_a_party_list_resume_keeps_the_turns_speed_rand_draw(tables, names):
    # speedRand is drawn once per turn (bcp.c:853-857): the foe's Quick Claw activates (it moves
    # first), Scyther's U-turn stops at its party list, and the resume, played with a draw of 0,
    # still reads the turn's draw (``ts_qc``): the turn end keeps it for the next turn's readers
    # (``qc_roll``, the foe an active Quick Claw holder).
    from tests.engine.test_levelup import _mon

    s = make_state(
        tables,
        [_mon(names, "Scyther", 50, moves=("U-turn",)), _mon(names, "Geodude", 30)],
        [_mon(names, "Tangela", 70, item=names.item("Quick Claw"))],
    )
    turn_draw = empty_outcome()._replace(quick_claw=jnp.asarray([0, 1], jnp.int8))
    paused = assert_matches_reference(tables, s, (0, 0), turn_draw)
    assert int(paused.pivot_battler) == 1 and np.asarray(paused.ts_qc).tolist() == [0, 1]
    done = assert_matches_reference(tables, paused, (MOVE_ACTIONS + 1, 0), empty_outcome())
    assert int(done.pivot_battler) == 0 and int(done.active[0]) == 1
    assert np.asarray(done.qc_roll).tolist() == [0, 1]
    assert np.asarray(done.ts_qc).tolist() == [0, 0]


_COUNTED = ("reflect", "light_screen", "mist", "safeguard", "tailwind", "lucky_chant")


def _with_counters(s, temp_rain):
    """``s`` with every turn-end counter at 3: the six side timers on both sides, Gravity and
    Trick Room (the mask's 3-bit counters), the player's Taunt and Perish Song, and a TEMP rain's
    ``weather_turns`` when ``temp_rain``."""
    from nuzline.engine.ids import (
        FIELD_CONDITION_GRAVITY_SHIFT,
        FIELD_CONDITION_TRICK_ROOM_SHIFT,
        FieldCondition,
    )

    three = {name: jnp.full_like(getattr(s, name), 3) for name in _COUNTED}
    mask = int(s.field_mask) | 3 << FIELD_CONDITION_GRAVITY_SHIFT
    mask |= 3 << FIELD_CONDITION_TRICK_ROOM_SHIFT
    if temp_rain:
        mask |= int(FieldCondition.RAINING_TEMP)
        three["weather_turns"] = jnp.asarray(3, s.weather_turns.dtype)
    return s._replace(
        **three,
        field_mask=jnp.asarray(mask, s.field_mask.dtype),
        taunt=s.taunt.at[0].set(3),
        perish_song=s.perish_song.at[0].set(1),
        perish_song_turns=s.perish_song_turns.at[0].set(3),
    )


def _counted_once(done, temp_rain):
    from nuzline.engine.ids import FIELD_CONDITION_GRAVITY_SHIFT, FIELD_CONDITION_TRICK_ROOM_SHIFT

    for name in _COUNTED:  # CheckFieldConditions :947-1083
        assert np.asarray(getattr(done, name)).tolist() == [2, 2], name
    mask = int(done.field_mask)
    assert mask >> FIELD_CONDITION_GRAVITY_SHIFT & 7 == 2  # :1227-1238
    assert mask >> FIELD_CONDITION_TRICK_ROOM_SHIFT & 7 == 2  # :1822-1830
    assert int(done.taunt[0]) == 2  # CheckMonConditions
    assert int(done.perish_song_turns[0]) == 2  # CheckSideConditions
    if temp_rain:
        assert int(done.weather_turns) == 2  # :1121


@pytest.mark.parametrize("site", ["move end", "field", "mon"])
def test_midturn_learn_resume_counts_every_turn_end_counter_once(tables, names, site):
    # A MoveEnd pause comes before the turn end, which the resume runs whole; a pause in the field
    # pass (LEARN_FIELD, after the weather states) resumes with Gravity alone; one in a battler's
    # block (LEARN_MON) resumes with the later blocks and the side conditions. Each counter of the
    # turn end drops exactly once over the pause and the resume (bcp.c:932-1254, :1289, :1746).
    from nuzline.engine.levelup import LEARN_FIELD, LEARN_MON, LEARN_MOVE_END
    from tests.engine.test_levelup import _mon

    opp = [_mon(names, "Geodude", 12), _mon(names, "Bonsly", 15)]
    miss = empty_outcome()._replace(hit=jnp.zeros((2, MAX_HITS), jnp.int8))
    temp_rain = site != "field"
    kw = {"start_condition": Weather.HAIL} if site == "field" else {}
    s = _with_counters(_learn_state(tables, names, opp, **kw).replace_hp(1, 0, 1), temp_rain)
    if site == "mon":
        s = s._replace(status=s.status.at[1, 0].set(int(Status.BURN)))
    acts, o, want = A("Tackle", "Tackle", s, names), miss, {"field": LEARN_FIELD, "mon": LEARN_MON}
    if site == "move end":
        acts, o = A("Bubble", "Tackle", s, names), empty_outcome()
    paused = assert_matches_reference(tables, s, acts, o)
    assert int(paused.learn_site) == want.get(site, LEARN_MOVE_END)
    _, done = _resume_learn(tables, paused)
    assert int(done.learn_site) == 0 and int(done.turn) == int(s.turn) + 1
    _counted_once(done, temp_rain)
