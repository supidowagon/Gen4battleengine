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
from nuzline.engine.ids import (
    ABILITY_DECOMP_IDS,
    MOVE_STRUGGLE,
    AbilityId,
    EffectId,
    FlingEffect,
    MoveFlag,
    Stat,
    Status,
    Weather,
)
from nuzline.engine.order import CALL_MIRROR_MOVE, call_kind
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import make_state
from tests.engine.helpers import A, as_species, assert_matches_reference, roark_state
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


def test_the_pick_law_is_the_retry_loop(tables, names):
    """``enumerate._conv2`` (the law the enumeration draws) is the 1000-try loop of
    BtlCmd_TryConversion2 followed by the linear scan, computed here try by try with fractions:
    a Rock / Ground user against a recorded Normal move has Steel and Ghost left, a Fire / Fighting
    one the Fire rows (Fire, Water, Rock, Dragon). The values are on the 2**-52 grid."""
    from fractions import Fraction

    from nuzline.engine.damage import conv2_pairs
    from nuzline.engine.enumerate import CONV2_DENOMINATOR, _conv2, outcome_laws

    weight, row = (np.asarray(x) for x in conv2_pairs(tables))
    for user, kind, foe in (("Geodude", NORMAL, "Tackle"), ("Monferno", FIRE, "Ember")):
        s = _c2_fight(tables, names, foe=(foe,), user=user)
        s = _record(s, 0, names.move(foe), 1, kind)
        acts = A("Conversion 2", foe, s, names)
        laws = jax.tree.map(np.asarray, jax.jit(outcome_laws)(tables, s, acts))
        values, weights, den = _conv2(laws, 0, kind)
        assert den == CONV2_DENOMINATOR and sum(weights) == den
        own = {int(x) for x in np.asarray(s.types[0, 0])}
        rows = sorted((int(row[kind, d]), d) for d in range(18) if weight[kind, d] and d not in own)
        w = {d: int(weight[kind, d]) for _, d in rows}
        # The loop, 1000 times: a try wins with its row's weight, else the next try; then the scan.
        mass, left = {d: Fraction(0) for d in w}, Fraction(1)
        for _ in range(1000):
            for d, x in w.items():
                mass[d] += left * Fraction(x, 65536)
            left *= 1 - Fraction(sum(w.values()), 65536)
        mass[rows[0][1]] += left
        for d, x in zip(values, weights, strict=True):
            assert abs(Fraction(x, den) - mass[d]) < Fraction(1, 2**50), (user, d)
        assert set(values) == set(w)
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
            s = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
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


def test_conversion_2_fails_without_a_record_and_skips_the_attack_message(tables, names):
    """BtlCmd_TryConversion2 (:4775-4776) with no recorded move jumps to subscript_conversion_2.s:12
    (MOVE_STATUS_FAILED) before the attack message: the types stay and movePrev is 0 (the foe's
    Swords Dance, a self-range move, left no record and the user acts last), while a success shows
    it (movePrev is Conversion 2)."""
    t, n = tables, names
    s = _c2_fight(t, n, foe=("Swords Dance",))
    failed = _c2_go(t, n, s, "Swords Dance")
    assert _types(failed) == _types(s)
    assert int(failed.move_prev) == 0
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


def test_the_draw_is_sampled_from_the_law(tables, names):
    """``outcome.sample_outcome`` draws ``conv2`` from ``damage.conv2_probs`` per battler and type:
    over many keys the Rock user's picks are Steel (row 1, 586 u16 values) and Ghost (the Foresight
    row 110, 585) only, in the law's proportion."""
    from nuzline.engine.damage import conv2_probs
    from nuzline.engine.outcome import sample_outcome

    t, n = tables, names
    s = _c2_fight(t, n, user="Geodude", foe_species="Monferno")
    law = np.asarray(conv2_probs(t, s, 0, NORMAL))
    assert set(np.flatnonzero(law)) == {STEEL, GHOST}
    assert abs(law.sum() - 1) < 1e-6 and law[STEEL] > law[GHOST]
    acts = A("Conversion 2", "Tackle", s, n)
    draw = jax.jit(jax.vmap(lambda k: sample_outcome(k, t, s, acts, False).conv2[0, NORMAL]))
    picks = np.asarray(draw(jax.random.split(jax.random.PRNGKey(0), 4000)))
    assert set(np.unique(picks)) == {STEEL, GHOST}
    assert abs((picks == STEEL).mean() - law[STEEL]) < 0.04


# --- #100 Spite (E3-B2-4) ---------------------------------------------------------------------
#
# effect_script_0100.s -> subscript_reduce_target_pp.s -> BtlCmd_TrySpite (battle_script.c:4990-
# 5016): the slot of the target's movePrevByBattler (``last_move``) loses min(4, ppCur); no last
# move, a move in no slot (Struggle) or a slot at 0 PP is MOVE_STATUS_FAILED
# (subscript_reduce_target_pp.s:12-14) and the attack message is skipped. No random draw.


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


def test_spite_fails_without_a_usable_last_move_and_skips_the_attack_message(tables, names):
    """No last move (:4995, :5011-5013), Struggle (in no slot) or a move the foe holds in no slot
    (Battler_SlotForMove finds none, :4996) fail it: no PP changes. The attack message
    (subscript_reduce_target_pp.s:6, after TrySpite) is skipped, so movePrev stays 0 behind a
    slower user's sleeping foe (whose own failed move zeroed it and zeroed its last move), where
    the foe's Growl first lets Spite succeed and show itself."""
    t, n = tables, names
    s = _spite_fight(t, n, fast=False)
    worked = _spite(t, n, s)
    assert int(worked.move_prev) == n.move("Spite")
    sleeper = s.with_status(1, 0, Status.SLEEP, 5)
    out = _spite(t, n, sleeper)
    assert _pp(out) == _pp(sleeper) and int(out.move_prev) == 0
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


def test_role_play_fails_on_its_listed_conditions_and_skips_the_attack_message(tables, names):
    """effect_script_0178.s:5-6 (the user's Multitype, a held Griseous Orb) and
    subscript_copy_ability.s:6-9 (the defender's Wonder Guard, Multitype, no ability; the user's
    Multitype): nothing is copied, truant stays, and the attack message after the failure jump is
    skipped, so the battle's movePrev stays empty (the Monferno acts last here); a success
    records the move. The defender's Orb, the user's own Wonder Guard or no ability, and a Substitute do not stop it."""
    t, n = tables, names
    orb = n.item("Griseous Orb")
    base = _rp(t, n, AbilityId.BLAZE, AbilityId.ROCK_HEAD)
    ok = _rpgo(t, n, base)
    assert int(ok.move_prev) == n.move("Role Play")
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
        assert int(out.move_prev) == 0, label
    arceus = as_species(t, base, 1, 0, n.species("Arceus"))
    out = _rpgo(t, n, arceus)  # the defender's Multitype
    assert int(out.ability[0, 0]) == int(arceus.ability[0, 0]) and int(out.move_prev) == 0
    mt_user = as_species(t, base, 0, 0, n.species("Arceus"))
    out = _rpgo(t, n, mt_user)  # the user's Multitype
    assert int(out.ability[0, 0]) == int(mt_user.ability[0, 0]) and int(out.move_prev) == 0
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
    s = s._replace(
        ability=s.ability.at[0, mon].set(AbilityId.SAND_VEIL),
        weather=jnp.asarray(Weather.SANDSTORM, s.weather.dtype),
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


def test_a_pursuit_ko_of_a_switching_grudge_user_zeroes_the_pursuers_slot(tables, names):
    """subscript_pursuit.s:85 runs the same faint check: a Grudge holder at 1 HP switching out to
    a Pursuit zeroes the pursuer's slot (the one Pursuit spent). The bit stays from an earlier
    turn: a switch is no move attempt."""
    from nuzline.engine.actions import MOVE_ACTIONS

    t, n = tables, names
    s = _gr(t, n, foe=("Growl", "Pursuit"))
    s = s._replace(grudge=s.grudge.at[0].set(1))
    out = assert_matches_reference(t, s, (MOVE_ACTIONS + 1, 1), _hits())
    assert int(out.hp[0, 0]) == 0
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
