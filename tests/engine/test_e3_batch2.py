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
