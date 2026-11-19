"""E3 batch 2 (rulings E3-B2-n, docs/kaizo/e3-ledger.md): Fling (#233, E3-B2-1), each case checked
against the reference engine.

Decomp (external/pokeplatinum): ``res/battle/scripts/effects/effect_script_0233.s``,
``res/battle/scripts/subscripts/subscript_fling.s``, ``src/battle/battle_lib.c``
(``BattleSystem_FlingItem`` :5857-6158, ``Battler_ItemFlingPower`` :5491-5498),
``src/battle/battle_script.c`` (``BtlCmd_TryFling`` :8144-8154, ``BtlCmd_RemoveItem``
:8972-8984)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.spec import PokemonSpec
from nuzline.engine.enumerate import INPUT_CLAUSE_EFFECTS, NON_TABLE_DAMAGE_EFFECTS
from nuzline.engine.ids import AbilityId, EffectId, FlingEffect, Stat, Status
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import A, assert_matches_reference
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
    (:6110-6129): ``Outcome.acupressure`` 0 is the first (Attack), +2."""
    s = _fight(tables, names, "Liechi Berry", foe="Bidoof", foe_level=40)
    out = _go(tables, names, s)
    assert _stage(out, 1, Stat.ATK) == 1
    maxed = s._replace(stages=s.stages.at[1, Stat.ATK].set(6))
    out = _go(tables, names, maxed)
    assert _stage(out, 1, Stat.ATK) == 6
    star = _fight(tables, names, "Starf Berry", foe="Bidoof", foe_level=40)
    out = _go(tables, names, star, _hits()._replace(acupressure=jnp.zeros(2, jnp.int16)))
    assert _stage(out, 1, Stat.ATK) == 2
    out = _go(tables, names, star, _hits()._replace(acupressure=jnp.asarray([3, 3], jnp.int16)))
    assert sum(_stage(out, 1, st) for st in range(5)) == 2


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
