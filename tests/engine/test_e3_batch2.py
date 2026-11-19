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
from nuzline.engine.ids import AbilityId, EffectId, FlingEffect, MoveFlag, Stat, Status
from nuzline.engine.order import CALL_MIRROR_MOVE, call_kind
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
