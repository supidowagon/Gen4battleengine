"""E1 Gardenia batch (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): hand cases, each checked
against the reference engine (``assert_matches_reference``).

Family A: held items that draw. Focus Band (``HOLD_EFFECT_MAYBE_ENDURE``, G-1 .. G-3) and the
flavor berries Figy, Wiki, Mago, Aguav, Iapapa (``HOLD_EFFECT_HP_RESTORE_SPICY`` .. ``_SOUR``,
G-4 .. G-6). Family B: the ability Pressure (G-8 .. G-11). Family C: the status guards
Synchronize and Leaf Guard (G-12 .. G-17). Family D: Wonder Guard and Download, with the AI's
reads of Wonder Guard (G-18 .. G-23).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import FIRST_ITEM_ACTION, legal_actions
from nuzline.engine.damage import ai_damage_for_roll, damage_table
from nuzline.engine.enumerate import enumerate_drawn, outcome_laws
from nuzline.engine.field import with_weather
from nuzline.engine.ids import (
    CONFUSION_PENDING,
    FOCUS_BAND_PARAM,
    NATURE_DISLIKED_FLAVOR,
    AbilityId,
    Flavor,
    ItemId,
    SemiKind,
    Stat,
    Status,
    Weather,
)
from nuzline.engine.order import (
    MOVE_ACTIONS,
    action_order,
    battler_selected_move,
    fight_move,
    me_first_copy,
    with_selection,
)
from nuzline.engine.outcome import MAX_HITS, empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import _pursuer_move, replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    roark_state,
    weather_of,
    weather_timer_of,
)
from tests.engine.test_pivot import _mon

DECOMP = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"
O = empty_outcome()
SWITCH_TO_1 = MOVE_ACTIONS + 1
ADAMANT, MODEST = 3, 15  # decomp nature ids: Adamant dislikes Dry, Modest likes it


def _item(s, side, slot, item, names):
    return s._replace(item=s.item.at[side, slot].set(names.item(item) if item else 0))


def _full(s, side, slot, hp):
    return s._replace(hp=s.hp.at[side, slot].set(hp), max_hp=s.max_hp.at[side, slot].set(hp))


def _band(o, side, hits):
    """``o`` with ``focus_band[side, i] = 1`` for each ``i`` in ``hits``."""
    band = np.zeros((2, MAX_HITS), np.int8)
    for i in hits:
        band[side, i] = 1
    return o._replace(focus_band=jnp.asarray(band))


# --- data ---------------------------------------------------------------------------------


def test_tables_map_the_family_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """G-1, G-4: Focus Band (param 10, ``FOCUS_BAND_PARAM``) and the five flavor berries
    (param 8, column 1 their ``Flavor``) in both games."""
    flavors = {
        "Figy Berry": Flavor.SPICY,
        "Wiki Berry": Flavor.DRY,
        "Mago Berry": Flavor.SWEET,
        "Aguav Berry": Flavor.BITTER,
        "Iapapa Berry": Flavor.SOUR,
    }
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        band = n.item("Focus Band")
        assert int(t.item_effect[band]) == ItemId.FOCUS_BAND
        assert int(t.item_param[band, 0]) == FOCUS_BAND_PARAM
        for berry, flavor in flavors.items():
            i = n.item(berry)
            assert int(t.item_effect[i]) == ItemId.FLAVOR_BERRY, berry
            assert int(t.item_param[i, 0]) == 8, berry
            assert int(t.item_param[i, 1]) == flavor, berry


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_nature_flavor_table_is_the_decomps(names):
    """G-4: ``NATURE_DISLIKED_FLAVOR`` is the -1 entries of ``sNatureFlavorAffinities``
    (src/pokemon.c), by the decomp's nature order, and ``Flavor`` is include/constants/flavor.h."""
    src = (DECOMP / "src/pokemon.c").read_text(encoding="utf-8")
    start = src.index("sNatureFlavorAffinities[][5]")
    body = src[start : src.index("};", start)]
    rows = re.findall(r"\[NATURE_(\w+)\] = \{(.*?)\}", body, re.DOTALL)
    assert [r[0].lower() for r in rows] == [n.lower() for n in names.nature_names[:25]]
    order = [f.name for f in Flavor]
    for (nature, row), disliked in zip(rows, NATURE_DISLIKED_FLAVOR, strict=True):
        values = dict(re.findall(r"\[FLAVOR_(\w+)\] = (-?\d)", row))
        minus = [f for f in order if values[f] == "-1"]
        assert minus == ([] if disliked == -1 else [Flavor(disliked).name]), nature
    header = (DECOMP / "include/constants/flavor.h").read_text(encoding="utf-8")
    assert re.findall(r"^\s+FLAVOR_(\w+)", header, re.MULTILINE)[:5] == order


# --- Focus Band ------------------------------------------------------------------------------


def test_focus_band_holds_on_only_when_its_roll_passes(tables, names):
    """G-1: a lethal hit leaves 1 HP when ``focus_band[attacker, hit]`` passed, the band stays
    held and the player's is revealed to the AI; a failed roll lets the holder faint."""
    s = roark_state(tables, names)
    s = _full(_item(s, 1, 0, "Focus Band", names), 1, 0, 5)  # Mach Punch deals more than 5
    acts = A("Mach Punch", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, _band(O, 0, [0]))
    assert int(out.hp[1, 0]) == 1 and int(out.item[1, 0]) == names.item("Focus Band")
    assert int(out.item_consumed[1, 0]) == 0
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.hp[1, 0]) == 0
    # Below full HP it works the same (no full-HP test, bcp.c:3387).
    low = s.replace_hp(1, 0, 3)
    assert int(assert_matches_reference(tables, low, acts, _band(O, 0, [0])).hp[1, 0]) == 1
    # The player's band: the "hung on" message tags it (ai_item_known).
    p = _full(_item(roark_state(tables, names), 0, 0, "Focus Band", names), 0, 0, 2)
    out = assert_matches_reference(tables, p, A("Leer", "Tackle", p, names), _band(O, 1, [0]))
    assert int(out.hp[0, 0]) == 1
    assert int(out.ai_item_known[0]) == names.item("Focus Band")


def test_focus_band_multi_hit_rolls_per_hit(tables, names):
    """G-2: a roll on every hit; a passed roll sets focusItemActivated for the rest of the
    move, so a non-lethal first hit's roll still saves the holder from the second."""
    s = roark_state(tables, names, player_moves=["Double Kick"])
    kick = names.move("Double Kick")
    d = int(damage_table(tables, s, 0, kick, False)[15])
    acts = A("Double Kick", "Leer", s, names)
    two = _full(_item(s, 1, 0, "Focus Band", names), 1, 0, d + 2)
    # The second (lethal) hit's own roll.
    assert int(assert_matches_reference(tables, two, acts, _band(O, 0, [1])).hp[1, 0]) == 1
    # Only the first, non-lethal hit's roll: the flag lasts the move.
    assert int(assert_matches_reference(tables, two, acts, _band(O, 0, [0])).hp[1, 0]) == 1
    # Neither: it faints.
    assert int(assert_matches_reference(tables, two, acts, O).hp[1, 0]) == 0
    # DoubleSlap (6 damage a hit) for 5 hits into a 3-HP holder: the first roll saves it and
    # the flag caps every later hit; a roll on a later hit comes too late.
    slap = roark_state(tables, names, player_moves=["DoubleSlap"], opp_species="Shinx")
    slap = _full(_item(slap, 1, 0, "Focus Band", names), 1, 0, 3)
    five = O._replace(n_hits=jnp.array([5, 2], jnp.int8))
    acts = A("DoubleSlap", "Leer", slap, names)
    out = assert_matches_reference(tables, slap, acts, _band(five, 0, [0]))
    assert int(out.hp[1, 0]) == 1 and int(out.item[1, 0]) == names.item("Focus Band")
    assert int(assert_matches_reference(tables, slap, acts, _band(five, 0, [3])).hp[1, 0]) == 0


def test_focus_band_confusion_self_hit(tables, names):
    """G-3: subscript_hurt_self_in_confusion.s's CheckHoldOnWith1HP on the user, with
    ``focus_band[side, 0]`` (its move does not run)."""
    s = _full(_item(roark_state(tables, names), 0, 0, "Focus Band", names), 0, 0, 2)
    s = s._replace(confusion=s.confusion.at[0].set(3))
    hurt = O._replace(confusion_hit=jnp.array([1, 0], jnp.int8))
    acts = A("Scratch", "Leer", s, names)
    assert int(assert_matches_reference(tables, s, acts, hurt).hp[0, 0]) == 0
    out = assert_matches_reference(tables, s, acts, _band(hurt, 0, [0]))
    assert int(out.hp[0, 0]) == 1 and int(out.item[0, 0]) == names.item("Focus Band")


def test_focus_band_pursuit(tables, names):
    """G-3: subscript_pursuit.s's CheckHoldOnWith1HP on the switcher, ``focus_band[pursuer, 0]``."""
    s = roark_state(tables, names, opp_moves=["Pursuit"])
    s = _full(_item(s, 0, 0, "Focus Band", names), 0, 0, 3)
    acts = A(SWITCH_TO_1, "Pursuit", s, names)
    assert int(assert_matches_reference(tables, s, acts, O).hp[0, 0]) == 0
    out = assert_matches_reference(tables, s, acts, _band(O, 1, [0]))
    assert int(out.hp[0, 0]) == 1 and int(out.active[0]) == 1


def test_focus_band_future_sight(tables, names):
    """G-3: subscript_future_sight_damage.s:28's CheckHoldOnWith1HP, ``future_band[battler]``."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Slowbro", ["Growl"], 50)],
        [_mon(n, "Geodude", ["Growl"], 20, "Focus Band")],
    )
    s = s._replace(
        future_timer=s.future_timer.at[1].set(1),
        future_damage=s.future_damage.at[1].set(500),
        future_move=s.future_move.at[1].set(n.move("Future Sight")),
        future_attacker=s.future_attacker.at[1].set(1),
        future_sight=s.future_sight.at[1].set(1),
    )
    o = O._replace(future_hit=jnp.asarray([1, 1], jnp.int8))
    assert int(assert_matches_reference(tables, s, (0, 0), o).hp[1, 0]) == 0
    band = o._replace(future_band=jnp.asarray([0, 1], jnp.int8))
    out = assert_matches_reference(tables, s, (0, 0), band)
    assert int(out.hp[1, 0]) == 1


# --- flavor berries --------------------------------------------------------------------------


def _wiki_state(tables, names, nature, hp):
    s = roark_state(tables, names)
    s = _full(_item(s, 0, 0, "Wiki Berry", names), 0, 0, 80).replace_hp(0, 0, hp)
    return s._replace(nature=s.nature.at[0, 0].set(nature))


def test_wiki_berry_heals_and_confuses_a_disliking_nature(tables, names):
    """G-4, G-5: at ``curHP <= maxHP / 2`` the berry heals ``Divide(maxHP, 8)``; an Adamant
    holder (dislikes Dry) is then confused, its counter pending until its confusion check
    draws ``berry_confusion``."""
    s = _wiki_state(tables, names, ADAMANT, 40)
    acts = A("Leer", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, O)  # the turn end's USE_ITEM check
    assert int(out.hp[0, 0]) == 40 + 10
    assert int(out.item[0, 0]) == 0 and int(out.item_consumed[0, 0]) == names.item("Wiki Berry")
    assert int(out.confusion[0]) == CONFUSION_PENDING
    # The next turn's check draws the counter (4), drops it (3) and hurts itself on the coin.
    nxt = O._replace(berry_confusion=jnp.asarray([4, 2], jnp.int8))
    later = assert_matches_reference(tables, out, acts, nxt)
    assert int(later.confusion[0]) == 3
    hurt = nxt._replace(confusion_hit=jnp.asarray([1, 0], jnp.int8))
    later = assert_matches_reference(tables, out, acts, hurt)
    assert int(later.confusion[0]) == 3 and int(later.hp[0, 0]) < 50


def test_wiki_berry_does_not_confuse_a_liking_nature(tables, names):
    """G-4: a Modest holder (likes Dry) is healed, not confused; above half HP nothing."""
    s = _wiki_state(tables, names, MODEST, 40)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), O)
    assert int(out.hp[0, 0]) == 50 and int(out.confusion[0]) == 0
    high = _wiki_state(tables, names, ADAMANT, 41)
    out = assert_matches_reference(tables, high, A("Leer", "Leer", high, names), O)
    assert int(out.hp[0, 0]) == 41 and int(out.item[0, 0]) == names.item("Wiki Berry")


def test_wiki_berry_after_a_hit_confuses_before_the_holders_move(tables, names):
    """G-5: the foe's faster hit drops the holder to half; it eats the berry after that hit
    (AfterMoveEffects' DEFENDER_ITEM) and its own confusion check that turn draws the
    counter."""
    s = roark_state(tables, names, opp_faster=True)
    s = _full(_item(s, 0, 0, "Wiki Berry", names), 0, 0, 200)
    s = s._replace(nature=s.nature.at[0, 0].set(ADAMANT))
    tackle = int(damage_table(tables, s, 1, names.move("Tackle"), False)[15])
    s = s.replace_hp(0, 0, 100 + tackle)  # Tackle leaves exactly 100 = maxHP / 2
    o = O._replace(berry_confusion=jnp.asarray([2, 2], jnp.int8))
    out = assert_matches_reference(tables, s, A("Scratch", "Tackle", s, names), o)
    assert int(out.confusion[0]) == 1  # drawn 2, dropped to 1 at the check
    assert int(out.hp[0, 0]) == 100 + 200 // 8
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # it still moved (no self-hit)


_laws = jax.jit(outcome_laws)


def _berry_boards(tables, names):
    """E2-ML-1 (test_enumerate.py::test_mask_sound falsifier A): an Adamant Wiki Berry holder at
    ``maxHP / 2`` whose foe acts first with a bag Potion, or with a Leer that may miss. Every
    action ends in MoveEnd, whose TriggerEffectOnSwitch (bcp.c:1973, :3973) runs the held-item
    pass over every battler (battle_lib.c:4151-4165): the berry is eaten there and the holder's
    own check draws its counter and the self-hit coin."""
    s = roark_state(tables, names, opp_faster=True)
    s = _full(_item(s, 0, 0, "Wiki Berry", names), 0, 0, 200).replace_hp(0, 0, 100)
    s = s._replace(nature=s.nature.at[0, 0].set(ADAMANT))
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]) - 10)
    potion = A("Scratch", FIRST_ITEM_ACTION, s, names)
    leer = A("Scratch", "Leer", s, names)
    return s, potion, leer


def _drawn(tables, s, acts):
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
    return laws, enumerate_drawn(laws)[2]


def test_a_berry_in_range_is_read_after_any_earlier_action(tables, names):
    s, potion, leer = _berry_boards(tables, names)
    for acts in (potion, leer):
        laws, drawn = _drawn(tables, s, acts)
        for name in ("confusion_hit", "berry_confusion"):
            assert bool(getattr(laws.relevant, name)[0]), name
            assert np.asarray(getattr(drawn, name))[:, 0].all(), name  # a missed Leer too
        assert np.asarray(drawn.roll)[:, 0, 0].all()
    # Out of range, a hidden berry (Embargo: Battler_HeldItem, battle_lib.c:5358-5360) or the
    # holder moving first: no read.
    for board, acts in (
        (s.replace_hp(0, 0, 101), potion),
        (s._replace(embargo=s.embargo.at[0].set(3)), potion),
    ):
        laws = jax.tree.map(np.asarray, _laws(tables, board, acts))
        assert not bool(laws.relevant.confusion_hit[0])
    slow = s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(1))
    laws = jax.tree.map(np.asarray, _laws(tables, slow, leer))
    assert not bool(laws.relevant.confusion_hit[0])


@pytest.mark.slow  # two mask-sound steps of the boards (one compiled step program)
def test_a_berry_in_range_boards_are_mask_sound(tables, names):
    from tests.engine.test_enumerate import _assert_mask_sound

    s, potion, leer = _berry_boards(tables, names)
    arbitrary = O._replace(
        confusion_hit=jnp.asarray([1, 1], jnp.int8),
        berry_confusion=jnp.asarray([4, 4], jnp.int8),
    )
    for acts in (potion, leer):
        _assert_mask_sound(tables, s, np.asarray(acts), arbitrary, 0)


def _encored_lock_board(tables, names):
    """E2-ML-2 (test_enumerate.py::test_mask_sound_with_a_trainer_item falsifier, explore seed 7):
    the player's lead locked into Bide (bide_turns 2: it stores this turn) and encored into
    Scratch, which its encored slot holds; the foe uses a bag Potion. FightCommand runs the
    encored move before the lock (bcp.c:1869-1873 before :1885-1888), so Scratch hits with its
    crit and roll."""
    n = names
    s = roark_state(tables, names, player_moves=["Supersonic", "Scratch"])
    scratch = n.move("Scratch")
    s = s._replace(
        locked_move=s.locked_move.at[0].set(n.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(2),
        encore_move=s.encore_move.at[0].set(scratch),
        encore_slot=s.encore_slot.at[0].set(1),
        encore_turns=s.encore_turns.at[0].set(2),
        last_move=s.last_move.at[0].set(scratch),
    )
    return s, A(0, FIRST_ITEM_ACTION, s, names)


def test_a_turn_start_encore_runs_over_a_lock_in_the_laws(tables, names):
    s, acts = _encored_lock_board(tables, names)
    assert int(fight_move(tables, s, 0, 0)) == names.move("Scratch")
    laws, drawn = _drawn(tables, s, acts)
    assert bool(laws.relevant.crit[0, 0]) and bool(laws.relevant.roll[0, 0])
    assert np.asarray(drawn.roll)[:, 0, 0].all()
    # Without the Encore the lock runs Bide, which stores this turn: no crit, no roll.
    bare = s._replace(encore_move=s.encore_move.at[0].set(0))
    laws = jax.tree.map(np.asarray, _laws(tables, bare, acts))
    assert not bool(laws.relevant.roll[0, 0])


@pytest.mark.slow  # one mask-sound step of the board
def test_a_turn_start_encore_over_a_lock_is_mask_sound(tables, names):
    from tests.engine.test_enumerate import _assert_mask_sound

    s, acts = _encored_lock_board(tables, names)
    arbitrary = O._replace(
        crit=jnp.ones((2, MAX_HITS), jnp.int8), roll=jnp.zeros((2, MAX_HITS), jnp.int8)
    )
    _assert_mask_sound(tables, s, np.asarray(acts), arbitrary, 0)


def _unheld_encore_board(tables, names):
    """E2-ML-3: the Bide-locked lead of ``_encored_lock_board`` whose encored slot 1 no longer
    holds the encored move (Tackle; the slot holds Leer). FightCommand runs moves[encoredMoveSlot]
    and ends the encore (bcp.c:1874-1883) before the lock (:1885-1888): Leer, not Bide."""
    s, acts = _encored_lock_board(tables, names)
    s = s._replace(
        moves=s.moves.at[0, 0, 1].set(names.move("Leer")),
        encore_move=s.encore_move.at[0].set(names.move("Tackle")),
    )
    return s, acts


def test_an_unheld_encore_runs_its_slot_over_a_lock_in_the_laws(tables, names):
    s, acts = _unheld_encore_board(tables, names)
    assert int(fight_move(tables, s, 0, 0)) == names.move("Leer")
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
    assert bool(laws.relevant.hit[0, 0])  # Leer's accuracy roll; Bide has none


@pytest.mark.slow  # two steps against the reference
def test_an_unheld_encore_runs_its_slot_over_a_lock_in_both_engines(tables, names):
    s, acts = _unheld_encore_board(tables, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.stages[1, Stat.DEF]) == int(s.stages[1, Stat.DEF]) - 1  # Leer ran
    assert int(out.encore_move[0]) == 0 and int(out.encore_turns[0]) == 0  # the encore ended
    # A held encore runs the encored move over the lock (E2-ML-2): Scratch hits.
    s2, acts2 = _encored_lock_board(tables, names)
    out2 = assert_matches_reference(tables, s2, acts2, O)
    assert int(out2.hp[1, 0]) < int(s2.hp[1, 0])


def _locked_order_board(tables, names):
    """E2-ML-5: the player's slower lead locked into Rollout (a lock its slot 1 holds) whose
    moveSlot is still slot 0, Quick Attack (a level-up wrote it over the slot it last selected).
    CalcTurnOrder reads moves[moveSlot]'s priority (battle_lib.c:1375-1397), so the player moves
    first; the foe Tackles."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Geodude", ["Quick Attack", "Rollout"], 20)],
        [_mon(n, "Starly", ["Tackle"], 30)],
    )
    s = s._replace(
        locked_move=s.locked_move.at[0].set(n.move("Rollout")),
        rollout_count=s.rollout_count.at[0].set(1),
        move_slot=s.move_slot.at[0].set(0),
    )
    return s, jnp.asarray([0, 0], jnp.int32)


def test_a_locked_battler_orders_by_its_move_slot(tables, names):
    s, acts = _locked_order_board(tables, names)
    assert int(action_order(tables, s, acts, O)[0]) == 0  # Quick Attack's +1
    lock_slot = s._replace(move_slot=s.move_slot.at[0].set(1))
    assert int(action_order(tables, lock_slot, acts, O)[0]) == 1  # Rollout's 0: the faster foe


@pytest.mark.slow  # two steps against the reference
def test_a_locked_battler_orders_by_its_move_slot_in_both_engines(tables, names):
    s, acts = _locked_order_board(tables, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.move_slot[0]) == 0  # a locked battler keeps its moveSlot (bcp.c:333-337)
    assert int(out.move_slot[1]) == 0  # the foe's FIGHT writes its chosen slot (:526)


def _pivot_board(tables, names):
    n = names
    return make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn", "Tackle"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle", "Encore"], 30)],
    )


def _resume_after(tables, paused, slot):
    side = int(paused.pivot_battler) - 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[side] = MOVE_ACTIONS + slot
    return assert_matches_reference(tables, paused, tuple(acts), O)


@pytest.mark.slow  # a pause and its resume against the reference
def test_an_unheld_encore_u_turn_over_a_lock_resumes_with_its_move(tables, names):
    """E2-ML-4: the player locked into Bide and encored into Tackle, whose encored slot 0 now
    holds U-turn: FightCommand runs U-turn (bcp.c:1874-1883) and ends the Encore; at the list the
    pause keeps that moveTemp (``ts_move_temp``) and the resumed tail's UpdateMoveBuffers writes it
    on the newcomer (bcp.c:3917-3946), not the lock's Bide."""
    n = names
    s = _pivot_board(tables, names)
    s = s._replace(
        locked_move=s.locked_move.at[0].set(n.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(2),
        encore_move=s.encore_move.at[0].set(n.move("Tackle")),
        encore_slot=s.encore_slot.at[0].set(0),
        encore_turns=s.encore_turns.at[0].set(2),
    )
    paused = assert_matches_reference(tables, s, (0, 0), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_move_temp) == n.move("U-turn")
    after = _resume_after(tables, paused, 1)
    assert int(after.last_move[0]) == n.move("U-turn")


@pytest.mark.slow  # a pause and its resume against the reference
def test_a_mid_turn_encore_takeover_of_u_turn_resumes_with_it(tables, names):
    """E2-ML-4: the faster foe encores the player into its last move, U-turn, which replaces the
    chosen Tackle (bcp.c:1869-1873); the resumed tail writes U-turn (moveTemp), in both engines."""
    n = names
    s = _pivot_board(tables, names)
    s = s._replace(
        last_move=s.last_move.at[0].set(n.move("U-turn")),
        stats=s.stats.at[1, 0, Stat.SPE - 1].set(200),
    )
    paused = assert_matches_reference(tables, s, (1, 1), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_move_temp) == n.move("U-turn")
    after = _resume_after(tables, paused, 1)
    assert int(after.last_move[0]) == n.move("U-turn")


def test_an_encored_pursuer_with_an_item_draws_its_pursuit_in_the_laws(tables, names):
    """E2-ML-6: battler 1 Encored into Pursuit picks a bag Potion while battler 0 switches;
    TryPursuit takes the held Encore's move (battle_script.c:6905-6909), so its hit's crit and
    roll are read."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 35)],
    )
    s = s._replace(
        encore_move=s.encore_move.at[1].set(n.move("Pursuit")),
        encore_slot=s.encore_slot.at[1].set(0),
        encore_turns=s.encore_turns.at[1].set(3),
        last_move=s.last_move.at[1].set(n.move("Pursuit")),
        trainer_items=s.trainer_items.at[1, 0].set(n.item("Potion")),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )
    acts = jnp.asarray([MOVE_ACTIONS + 1, FIRST_ITEM_ACTION], jnp.int32)
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
    assert bool(laws.relevant.crit[1, 0]) and bool(laws.relevant.roll[1, 0])


def _carried_board(tables, names):
    """E2-ML-8: the foe locked into Rollout (slot 0) whose last selection's TEMP_VALUE is still
    2, Pursuit (slot 1); the player switches its lead out (party slot 1)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Rollout", "Pursuit", "Tackle"], 35)],
    )
    s = s._replace(
        locked_move=s.locked_move.at[1].set(n.move("Rollout")),
        rollout_count=s.rollout_count.at[1].set(1),
        selected_temp=s.selected_temp.at[1].set(2),
    )
    return s, jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)


def test_a_locked_battler_reads_its_carried_selected_move(tables, names):
    """E2-ML-8: a locked battler picks no command (bcp.c:333-337), so its TEMP_VALUE stays its
    last selection's and Battler_SelectedMove (battle_lib.c:2858-2868) reads that slot's move, not
    the lock's: Pursuit's (battle_script.c:6909) and Me First's (:7296) read."""
    n = names
    s, acts = _carried_board(tables, names)
    sel = with_selection(tables, s, acts)
    assert int(sel.selected_temp[1]) == 2  # kept by the lock
    assert int(sel.selected_temp[0]) == 1  # the switch writes its party slot (bcp.c:595)
    assert int(battler_selected_move(sel, 1, True)) == n.move("Pursuit")
    assert int(battler_selected_move(sel, 1, False)) == 0  # not a FIGHT: MOVE_NONE
    has, move = _pursuer_move(sel, 1, acts[1], n.move("Rollout"), False)
    assert bool(has) and int(move) == n.move("Pursuit")
    tackle = sel._replace(selected_temp=sel.selected_temp.at[1].set(3))
    assert int(me_first_copy(tables, tackle, 1, True)) == n.move("Tackle")
    none = sel._replace(selected_temp=sel.selected_temp.at[1].set(0))
    assert int(me_first_copy(tables, none, 1, True)) == 0  # MOVE_NONE has no power (:7300)


@pytest.mark.slow  # a step against the reference
def test_a_locked_pursuer_pursues_with_its_carried_move_in_both_engines(tables, names):
    """E2-ML-8: the foe locked into Rollout carries Pursuit's TEMP_VALUE, so TryPursuit reads
    Pursuit (battle_script.c:6909) and chases the player's switch, spending its PP (:6914), in
    both engines."""
    s, acts = _carried_board(tables, names)
    out = assert_matches_reference(tables, s, tuple(int(a) for a in acts), O)
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # the Pursuit hit the switcher
    assert int(out.pp[1, 0, 1]) == int(s.pp[1, 0, 1]) - 1
    assert int(out.selected_temp[1]) == 2


@pytest.mark.slow  # a step against the reference
def test_dead_selection_values_are_canonical_in_both_engines(tables, names):
    """E2-ML-9: at the turn boundary an unrun pick leaves no trace. The sleeping player picks
    Growl, which does not run; it can pick next turn with a usable move, so its moveSlot and
    TEMP_VALUE are rewritten before any read (bcp.c:445-447, :525-526): both 0. The foe has no PP
    left (STRUGGLING_ALL, bcp.c:433-442): its carried TEMP_VALUE stays readable (Rage check,
    bcp.c:844; TryPursuit, battle_script.c:6909) and is kept; its moveSlot has no reader under
    Struggle (battle_lib.c:1382-1383, battle_script.c:6752): 0."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle", "Growl"], 35)],
        [_mon(n, "Houndour", ["Pursuit", "Tackle"], 30)],
    )
    s = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
        pp=s.pp.at[1, 0, :2].set(0),
        move_slot=s.move_slot.at[1].set(1),
        selected_temp=s.selected_temp.at[1].set(2),
    )
    out = assert_matches_reference(tables, s, (1, 0), O)
    assert int(out.status[0, 0]) == Status.SLEEP  # the pick never ran
    assert [int(out.move_slot[0]), int(out.selected_temp[0])] == [0, 0]
    assert [int(out.move_slot[1]), int(out.selected_temp[1])] == [0, 2]


def _struggling_pursuer_board(tables, names):
    """E2-ML-8, review 5 #2: the foe's only move, Pursuit, is disabled, so its FIGHT Struggles
    (STRUGGLING_ALL, bcp.c:433-434) and keeps the TEMP_VALUE 1 of its last Pursuit; TryPursuit has
    no Disable or Struggle test (battle_script.c:6899-6916), so it chases the player's switch."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 35)],
    )
    s = s._replace(
        disabled_move=s.disabled_move.at[1].set(n.move("Pursuit")),
        disable_turns=s.disable_turns.at[1].set(3),
        selected_temp=s.selected_temp.at[1].set(1),
    )
    return s, jnp.asarray([MOVE_ACTIONS + 1, 0], jnp.int32)


def test_a_carried_pursuit_draws_its_pursuit_in_the_laws(tables, names):
    """Review 5 #2: on the locked (``_carried_board``, Rollout) and the Struggling pursuer, step
    hits with the carried Pursuit (``step._pursuer_move``); ``outcome.pursuit_chase`` gives the
    laws that move, so their damage table is Pursuit's, not the lock's or Struggle's."""
    from nuzline.engine.damage import hit_damage_table
    from nuzline.engine.outcome import pursuit_chase

    pursuit = names.move("Pursuit")
    for board, old in (
        (_carried_board, "Rollout"),
        (_struggling_pursuer_board, "Struggle"),
    ):
        s, acts = board(tables, names)
        fires, pursues, move = (np.asarray(x) for x in pursuit_chase(tables, s, acts))
        assert fires.tolist() == [False, True] and pursues.tolist() == [False, True]
        assert int(move[1]) == pursuit
        laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
        assert bool(laws.relevant.crit[1, 0]) and bool(laws.relevant.roll[1, 0])
        want = np.asarray(hit_damage_table(tables, s, 1, pursuit))
        stale = np.asarray(hit_damage_table(tables, s, 1, names.move(old)))
        assert not np.array_equal(want, stale)  # the board tells the two moves apart
        np.testing.assert_array_equal(laws.damage[1, 0], want, err_msg=old)


@pytest.mark.slow  # two mask-sound steps (one compiled step program)
def test_a_carried_pursuit_is_mask_sound(tables, names):
    """Review 5 #2: the locked and the Struggling carried-Pursuit boards are mask-sound."""
    from tests.engine.test_enumerate import _assert_mask_sound

    arbitrary = O._replace(
        crit=jnp.ones((2, MAX_HITS), jnp.int8), roll=jnp.zeros((2, MAX_HITS), jnp.int8)
    )
    for board in (_carried_board, _struggling_pursuer_board):
        s, acts = board(tables, names)
        _assert_mask_sound(tables, s, np.asarray(acts), arbitrary, 0)


@pytest.mark.slow  # compiles the action phase (one engine)
def test_turn_muls_read_this_turns_selection(tables, names):
    """Review 5 #1: the search's turn powerMul (``playout.turn_muls``, through
    ``step.replacement_power_mul``) reads this turn's TEMP_VALUE, as step does: the foe picks
    Pursuit while its last selection was Tackle (20), and Tackle after a Pursuit selection (10).
    Step's own value is the chase it plays (``shown["pursuit"]``: TryPursuit,
    battle_script.c:6884-6935, from the switch, subscript_switch_pokemon.s:11), the condition its
    ``_action_phase`` gives ``replacement_power_mul`` on the selected state. ``ts_power_mul`` is
    written only at a stop (``_execute_actions``' stop dict), so this unpaused turn leaves it 0.
    A raw-state ``turn_muls`` (84278d86) reads the previous TEMP_VALUE: 10 then 20."""
    from nuzline.engine.step import NORMAL_POWER_MUL, PURSUIT_POWER_MUL, _action_phase
    from nuzline.search.playout import turn_muls

    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Tackle", "Pursuit"], 35)],
    )
    phase = jax.jit(_action_phase)
    for last_temp, foe_action, want in ((1, 1, PURSUIT_POWER_MUL), (2, 0, NORMAL_POWER_MUL)):
        board = s._replace(selected_temp=s.selected_temp.at[1].set(last_temp))
        acts = jnp.asarray([MOVE_ACTIONS + 1, foe_action], jnp.int32)
        shown = phase(tables, board, acts, O)[1]
        chased = bool(int(shown["pursuit"]))
        stepped = PURSUIT_POWER_MUL if chased else NORMAL_POWER_MUL
        assert stepped == want
        assert int(turn_muls(tables, board, acts)[0]) == stepped


def test_selection_temps_of_items_switches_and_struggle(tables, names):
    """E2-ML-8 (review 5): ``selection_temps`` per command. A switch writes its party slot,
    0, 1 or 5 (bcp.c:595); a bag item writes 0 (its BattleItemUse, :565: the stated assumption);
    a FIGHT 1 + its slot (:525), under an Encore 0 (:447); a Struggling FIGHT keeps the carried
    value (:433-434)."""
    from nuzline.engine.order import selection_temps

    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 30 + k) for k in range(6)],
        [_mon(n, "Houndour", ["Tackle", "Pursuit", "Leer"], 35)],
    )
    s = s._replace(
        active=s.active.at[0].set(2),
        selected_temp=jnp.asarray([3, 3], s.selected_temp.dtype),
        trainer_items=s.trainer_items.at[1, 0].set(n.item("Potion")),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )

    def temps(board, a0, a1):
        return [int(x) for x in selection_temps(tables, board, jnp.asarray([a0, a1], jnp.int32))]

    assert temps(s, MOVE_ACTIONS + 0, 2) == [0, 3]
    assert temps(s, MOVE_ACTIONS + 1, FIRST_ITEM_ACTION) == [1, 0]
    assert temps(s, MOVE_ACTIONS + 5, 0) == [5, 1]
    encored = s._replace(
        encore_move=s.encore_move.at[1].set(n.move("Leer")),
        encore_slot=s.encore_slot.at[1].set(2),
        encore_turns=s.encore_turns.at[1].set(2),
    )
    assert temps(encored, 0, 2)[1] == 0
    struggling = s._replace(pp=s.pp.at[1, 0].set(0))  # no usable move: Struggle
    assert temps(struggling, 0, 0) == [1, 3]


def test_a_slot_five_switch_carries_the_iv_word(tables, names):
    """E2-ML-8 (review 5): TEMP_VALUE 5 (a switch to party slot 5) makes Battler_SelectedMove read
    ``moves[4]``, the low 16 bits of the IV bitfield after the four move slots (battle_mon.h:64-72):
    HP IV 4, Attack IV 7, Defense IV 0 and an even Speed IV give 4 | 7 << 5 = 228, Pursuit; an odd
    Speed IV sets bit 15, an id past the move table: MOVE_NONE (the stated assumption). Both
    engines read it alike."""
    from nuzline.engine.order import past_moves_word, selection_temps
    from nuzline.reference.engine import _selected_move

    n = names
    pursuit = n.move("Pursuit")
    assert pursuit == 228  # the decomp's MOVE_PURSUIT
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 30 + k) for k in range(6)],
        [_mon(n, "Houndour", ["Tackle"], 35)],
    )
    switch = jnp.asarray([MOVE_ACTIONS + 5, 0], jnp.int32)
    assert int(selection_temps(tables, s, switch)[0]) == 5
    ivs = [0] * 6
    ivs[Stat.HP], ivs[Stat.ATK], ivs[Stat.DEF], ivs[Stat.SPE] = 4, 7, 0, 30
    board = s._replace(
        active=s.active.at[0].set(5),
        ivs=s.ivs.at[0, 5].set(jnp.asarray(ivs, s.ivs.dtype)),
        selected_temp=s.selected_temp.at[0].set(5),
    )
    assert int(past_moves_word(board, 0)) == pursuit
    assert int(battler_selected_move(board, 0, True)) == pursuit
    assert _selected_move(to_ref(board), 0) == pursuit
    ivs[Stat.SPE] = 31
    odd = board._replace(ivs=board.ivs.at[0, 5].set(jnp.asarray(ivs, s.ivs.dtype)))
    assert int(battler_selected_move(odd, 0, True)) == 0
    assert _selected_move(to_ref(odd), 0) == 0


def test_a_locked_battler_orders_by_its_move_slot_in_the_laws(tables, names):
    """E2-ML-5 (review 5): the laws' loose branch reads CalcTurnOrder's priority through
    ``order_moves`` (moves[moveSlot], battle_lib.c:1375-1397): at equal speed the locked player's
    slot 0, Quick Attack (+1), against the foe's Tackle reads no speed tie; slot 1, Rollout (0),
    does."""
    s, acts = _locked_order_board(tables, names)
    s = s._replace(stats=s.stats.at[1, 0, Stat.SPE - 1].set(int(s.stats[0, 0, Stat.SPE - 1])))
    loose = jax.jit(outcome_laws, static_argnames=("loose",))

    def tie(board):
        return bool(np.asarray(loose(tables, board, acts, loose=True).relevant.speed_tie))

    assert not tie(s)
    assert tie(s._replace(move_slot=s.move_slot.at[0].set(1)))


def test_a_four_battler_berry_out_of_range_needs_another_move(tables, names):
    """E2-ML-7 (review 5): at B = 4 a Wiki Berry holder above half HP comes into range before an
    earlier MoveEnd only through another battler's move: with the other three switching, its
    turn-start berry check is not read; with them using Tackle, it is."""
    from nuzline.engine.format import DOUBLES
    from tests.engine.d2_states import four, party_of

    s = four(tables, names, DOUBLES)
    p, slot = party_of(s, 0), int(s.active[0])
    s = s._replace(
        item=s.item.at[p, slot].set(names.item("Wiki Berry")),
        nature=s.nature.at[p, slot].set(ADAMANT),
        stats=s.stats.at[p, slot, Stat.SPE - 1].set(1),  # it acts last
    )
    moves = jnp.asarray([1, 0, 0, 0], jnp.int32)  # Swords Dance; the others Tackle
    switches = jnp.asarray([1, MOVE_ACTIONS + 2, MOVE_ACTIONS + 2, MOVE_ACTIONS + 3], jnp.int32)
    read = [
        bool(np.any(np.asarray(_laws(tables, s, a).reads.berry_start)[:, 0]))
        for a in (moves, switches)
    ]
    assert read == [True, False]


def _carrying(state, names, b: int):
    """Battler ``b`` knows only Pursuit, disabled: its FIGHT Struggles (bcp.c:433-434) and keeps
    the TEMP_VALUE 1 of its last Pursuit, which TryPursuit reads with no Disable or Struggle test
    (battle_script.c:6899-6916, battle_lib.c:2858-2868)."""
    from tests.engine.d2_states import with_moves

    state = with_moves(state, names, b, ["Pursuit"])
    return state._replace(
        disabled_move=state.disabled_move.at[b].set(names.move("Pursuit")),
        disable_turns=state.disable_turns.at[b].set(3),
        selected_temp=state.selected_temp.at[b].set(1),
    )


def _idle_four(tables, names):
    """DOUBLES, every battler knows only Swords Dance (no crit, no roll of its own)."""
    from nuzline.engine.format import DOUBLES
    from tests.engine.d2_states import four, with_moves

    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, ["Swords Dance"])
    return s


def _assert_chase_widened(laws, b: int):
    """Battler ``b``'s laws read its chase beside its own move (``outcome.chase_widening``): the
    crit at hit 0 and its roll, classed over every stage's odds, no damage classes, no hit gate."""
    from nuzline.engine.outcome import CRIT_WRITE_VARIANTS

    assert bool(laws.relevant.crit[b, 0]) and bool(laws.relevant.roll[b, 0])
    assert bool(np.all(laws.crit_possible[b, CRIT_WRITE_VARIANTS:]))
    assert not bool(np.any(laws.compress[:, b, 0]))
    assert not bool(laws.reads.hit_gated[b])
    assert not bool(laws.no_crit[b])


def test_b4_a_diagonal_switcher_meets_a_carried_pursuit_in_the_laws(tables, names):
    """Review 6 #1: at B = 4 TryPursuit scans every battler of the other side (battle_script.c:
    6899-6904), so battler 0, Struggling with a carried Pursuit, chases battler 3's switch
    (``_foe(0)`` is 1). It may chase or Struggle at its own turn: both moves' reads are relevant
    (``chase_widening``'s ``both``). Struggle alone already makes its crit and roll relevant; what
    the old laws lacked is the chase's crit classes (every stage's odds) and the lifted hit gate
    (the chase has no accuracy check, subscript_pursuit.s:8-12)."""
    from nuzline.engine.outcome import chase_widening, pursuit_chase

    s = _carrying(_idle_four(tables, names), names, 0)
    acts = jnp.asarray([0, 0, 0, MOVE_ACTIONS + 2], jnp.int32)
    fires, pursues, move = (np.asarray(x) for x in pursuit_chase(tables, s, acts))
    assert fires.tolist() == [True, False, False, False]
    assert pursues.tolist() == [False] * 4  # a FIGHT of another move: its own move may run
    assert int(move[0]) == names.move("Struggle")
    both, _, chase = (np.asarray(x) for x in chase_widening(tables, s, acts, move, fires))
    assert both.tolist() == [True, False, False, False]
    assert int(chase[0]) == names.move("Pursuit")
    _assert_chase_widened(jax.tree.map(np.asarray, _laws(tables, s, acts)), 0)


def test_b4_the_chain_reaches_the_second_pursuer_in_the_laws(tables, names):
    """Review 6 #1: battler 0 switches; battler 1 (Struggling, carried Pursuit) is the first
    pursuer and battler 3 the next while the switcher lives (subscript_pursuit.s:79-82 -> :8).
    Battler 3 is Encored into Pursuit and picks a switch: it chases or runs no move, so its laws
    are Pursuit's (``pursues``); before, ``fires[3]`` tested battler 2 (``_foe(3)``) and its
    draws were not read."""
    from nuzline.engine.outcome import pursuit_chase
    from tests.engine.d2_states import with_moves

    s = _carrying(_idle_four(tables, names), names, 1)
    s = with_moves(s, names, 3, ["Pursuit"])
    s = s._replace(
        encore_move=s.encore_move.at[3].set(names.move("Pursuit")),
        encore_slot=s.encore_slot.at[3].set(0),
        encore_turns=s.encore_turns.at[3].set(3),
    )
    acts = jnp.asarray([MOVE_ACTIONS + 2, 0, 0, MOVE_ACTIONS + 2], jnp.int32)
    fires, pursues, move = (np.asarray(x) for x in pursuit_chase(tables, s, acts))
    assert fires.tolist() == [False, True, False, True]
    assert pursues.tolist() == [False, False, False, True]
    assert int(move[3]) == names.move("Pursuit")
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
    _assert_chase_widened(laws, 1)
    assert bool(laws.relevant.crit[3, 0]) and bool(laws.relevant.roll[3, 0])


def test_a_u_turn_meets_a_carried_pursuit_in_the_laws(tables, names):
    """Review 6 #1: U-turn calls the same TryPursuit (subscript_attack_then_switch_out.s:54). The
    foe, Struggling with a carried Pursuit, chases the player's U-turn when it has not acted, or
    Struggles first: both moves' reads are relevant (B = 2)."""
    from nuzline.engine.outcome import chase_widening, pursuit_chase

    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 30)],
    )
    s = _carrying(s, names, 1)
    acts = jnp.asarray([0, 0], jnp.int32)
    fires, _, move = pursuit_chase(tables, s, acts)
    assert not bool(np.any(np.asarray(fires)))  # no switch action
    both = np.asarray(chase_widening(tables, s, acts, move, fires)[0])
    assert both.tolist() == [False, True]
    _assert_chase_widened(jax.tree.map(np.asarray, _laws(tables, s, acts)), 1)


def test_a_u_turn_chase_by_a_carried_pursuit_reaches_acid_rain(tables, names):
    """Review 7 #1: the Struggling carried-Pursuit foe chases the player's U-turn
    (subscript_attack_then_switch_out.s:54); a KO there runs subscript_pursuit.s:115 on a nonzero
    mask (:39 -> _166). Acid Rain's pursuer mask takes the battler whose TryPursuit move is a
    Pursuit (``chase_pursuers``), not only a battler running one: Struggle's effect alone
    misses the pair."""
    from nuzline.engine.ids import FieldCondition
    from nuzline.engine.order import move_effects
    from nuzline.engine.outcome import acid_rain_reachable, chase_pursuers

    n = names
    s = make_state(
        tables,
        [_mon(n, "Scyther", ["U-turn"], 30), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 30)],
    )
    s = _carrying(s, names, 1)
    s = s._replace(field_mask=jnp.asarray(int(FieldCondition.SANDSTORM_TEMP), s.field_mask.dtype))
    acts = jnp.asarray([0, 0], jnp.int32)
    moves = jnp.asarray([n.move("U-turn"), n.move("Struggle")], jnp.int32)
    effect, _ = move_effects(tables, s, moves)
    uses = jnp.asarray([True, True])
    pursuers = chase_pursuers(tables, s, acts)
    assert np.asarray(pursuers).tolist() == [False, True]
    assert not bool(acid_rain_reachable(s, acts, uses, effect))  # Struggle's effect alone
    assert bool(acid_rain_reachable(s, acts, uses, effect, pursuers))
    # End to end (review 8): a due Future Sight on the foe reads the mask at the turn's end
    # (bcp.c:2926-2937, :2972-2974). Deep fog alone minus the pursuer's index 1 (:115's
    # sub_to_zero) is 0x7FFF: fog off, the Gravity counter on, so the hit's grid opens that
    # field (``_future_fields``); without the carried TEMP_VALUE no chase, no such field.
    from nuzline.engine.outcome import future_thresholds
    from tests.engine.test_e1_batch_h import _pending

    fog = _pending(
        s._replace(field_mask=jnp.asarray(int(FieldCondition.DEEP_FOG), s.field_mask.dtype)),
        names,
        1,
        0,
    )
    plain = fog._replace(selected_temp=fog.selected_temp.at[1].set(0))  # nothing carried
    # future_thresholds returns (values bool[B, 101], loose bool[B]): the defender's possible
    # thresholds (E4 future_values, a3113932). Gravity on with fog off adds the clipped 100.
    chased = np.asarray(future_thresholds(tables, fog, acts, moves)[0][1])
    unchased = np.asarray(future_thresholds(tables, plain, acts, moves)[0][1])
    assert bool(np.all(chased | ~unchased)) and bool(np.any(chased & ~unchased))


def test_b4_a_switch_chase_is_cut_only_by_the_chased_battlers_armor(tables, names):
    """Review 7 (should-fix): battler 0 chooses Pursuit and diagonal battler 3 switches; battler 1
    (``_foe(0)``) holds Shell Armor, battler 3 does not. The chase's defender is the switcher
    (battle_script.c:6927), so the crit is not cut (the old cut read battler 1's armor)."""
    from tests.engine.d2_states import with_ability, with_moves

    s = with_moves(_idle_four(tables, names), names, 0, ["Pursuit"])
    s = with_ability(s, 1, AbilityId.SHELL_ARMOR)
    acts = jnp.asarray([0, 0, 0, MOVE_ACTIONS + 2], jnp.int32)
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts))
    assert not bool(laws.no_crit[0])
    assert bool(laws.relevant.crit[0, 0])
    shelled = with_ability(s, 3, AbilityId.SHELL_ARMOR)  # every battler it may meet is armored
    assert bool(np.asarray(_laws(tables, shelled, acts).no_crit)[0])


def test_b4_a_takeover_pursuit_meets_the_u_turn_users_armor(tables, names):
    """Review 7 #2: battler 1 chose Tackle; in the takeover variant (B11-1) battler 0's faster
    Encore makes its last move, Pursuit, its encoredMove, and battler 2's U-turn then meets it
    (TryPursuit's Encore branch, battle_script.c:6905-6907). Battler 0, its own target, holds
    Shell Armor; the U-turn user does not: the takeover row's crit is not cut."""
    from nuzline.engine.outcome import chase_widening
    from tests.engine.d2_states import with_ability, with_moves

    n = names
    s = _idle_four(tables, names)
    s = with_moves(s, names, 0, ["Encore"])
    s = with_moves(s, names, 1, ["Tackle", "Pursuit"])
    s = with_moves(s, names, 2, ["U-turn"])
    s = with_ability(s, 0, AbilityId.SHELL_ARMOR)
    s = s._replace(last_move=s.last_move.at[1].set(n.move("Pursuit")))
    acts = jnp.asarray([0, 0, 0, 0], jnp.int32)
    taken = jnp.asarray([False, True, False, False])
    moves = jnp.asarray(
        [n.move("Encore"), n.move("Pursuit"), n.move("U-turn"), n.move("Swords Dance")], jnp.int32
    )
    fires = jnp.zeros(4, bool)
    both, armor, _ = (np.asarray(x) for x in chase_widening(tables, s, acts, moves, fires, taken))
    assert not bool(armor[1]) and not bool(both[1])  # the chase move is the run move
    laws = jax.tree.map(np.asarray, _laws(tables, s, acts, None, taken))
    assert not bool(laws.no_crit[1])


@pytest.mark.slow  # a step against the reference
def test_a_takeover_trump_card_reads_its_move_slot_pp(tables, names):
    """Review 5 #3: the faster foe encores the player into its last move, Trump Card, over its
    chosen Tackle (slot 1): FightCommand changes moveTemp, not moveSlot (bcp.c:1869-1873), so
    CalcTrumpCardPower reads ppCur[ATTACKER_MOVE_SLOT] (battle_script.c:7237), Tackle's 1 PP:
    80, not Trump Card's 4 after its decrement (40); in both engines."""
    from tests.engine.test_batch13 import _dfr, _loss

    n = names
    s = make_state(
        tables,
        [_mon(n, "Eevee", ["Trump Card", "Tackle"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle", "Encore"], 30)],
    )
    s = s.with_pp(0, 0, [5, 1, 0, 0])
    s = s._replace(
        last_move=s.last_move.at[0].set(n.move("Trump Card")),
        stats=s.stats.at[1, 0, Stat.SPE - 1].set(200),
    )
    m = n.move("Trump Card")
    assert _dfr(tables, s, m, 80) != _dfr(tables, s, m, 40)
    out = assert_matches_reference(tables, s, (1, 1), O)
    assert int(out.encore_move[0]) == m
    assert int(out.pp[0, 0, 0]) == 4 and int(out.pp[0, 0, 1]) == 1
    assert _loss(s, out) == _dfr(tables, s, m, 80)


@pytest.mark.slow  # a step against the reference
def test_an_unheld_encore_counter_drains_its_move_slot(tables, names):
    """Review 5 #3: the player locked into Bide (moveSlot 2, Growl's, from its last selection) and
    encored into Tackle, whose encored slot 1 now holds Counter: FightCommand runs Counter and ends
    the Encore (bcp.c:1874-1883). The faster Pressure foe's Tackle lets it pass, and
    BattleSystem_DecPPForPressure (battle_script.c:4646; battle_lib.c:7390-7395) takes one more
    PP from ppCur[moveSlot]: Growl's, in both engines (the action's slot 0 and the Encore slot 1
    keep theirs from it)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Eevee", ["Supersonic", "Counter", "Growl"], 30)],
        [_mon(n, "Bulbasaur", ["Tackle"], 30)],
    )
    s = _pressure(s, 1)._replace(
        locked_move=s.locked_move.at[0].set(n.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(2),
        encore_move=s.encore_move.at[0].set(n.move("Tackle")),
        encore_slot=s.encore_slot.at[0].set(1),
        encore_turns=s.encore_turns.at[0].set(2),
        move_slot=s.move_slot.at[0].set(2),
        stats=s.stats.at[1, 0, Stat.SPE - 1].set(200),
    )
    out = assert_matches_reference(tables, s, (0, 0), O)
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # Counter hit back
    assert int(out.pp[0, 0, 2]) == int(s.pp[0, 0, 2]) - 1  # Growl: the Pressure drain
    assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0])


@pytest.mark.slow  # a pause and its resume against the reference
def test_a_takeover_u_turn_leaves_a_choice_newcomer_unlocked(tables, names):
    """E2-ML-4 (review 5): the mid-turn takeover of ``test_a_mid_turn_encore_takeover_of_u_turn_
    resumes_with_it`` with a Choice Band newcomer: the resumed tail's UpdateMoveBuffers reads
    moveTemp, U-turn, whose SYSCTL_UTURN_ACTIVE exempts the Choice lock (bcp.c:3923-3929), not
    the chosen Tackle, which would lock the newcomer into it."""
    n = names
    s = make_state(
        tables,
        [
            _mon(n, "Scyther", ["U-turn", "Tackle"], 30),
            _mon(n, "Geodude", ["Tackle"], 30, "Choice Band"),
        ],
        [_mon(n, "Bulbasaur", ["Tackle", "Encore"], 30)],
    )
    s = s._replace(
        last_move=s.last_move.at[0].set(n.move("U-turn")),
        stats=s.stats.at[1, 0, Stat.SPE - 1].set(200),
    )
    paused = assert_matches_reference(tables, s, (1, 1), O)
    assert int(paused.pivot_battler) == 1 and int(paused.ts_move_temp) == n.move("U-turn")
    after = _resume_after(tables, paused, 1)
    assert int(after.active[0]) == 1 and int(after.choice_lock[0]) == 0


def test_a_klutz_lansat_at_its_pinch_reads_its_crit_after_a_gastro_acid(tables, names):
    """E2-ML-7 (manager CLASS 7): a Klutz Lopunny holding a Lansat Berry at a quarter of its HP;
    the faster foe's Gastro Acid takes the Klutz, and that action's MoveEnd pass eats the berry
    (TriggerHeldItemOnStatus over every battler, battle_lib.c:4151-4165, :5261-5271; the item
    read through Battler_HeldItem, :5352-5363), so Lopunny's Tackle can crit at the raised
    stage. Above its pinch with no HP-lowering foe move: no variant."""
    from nuzline.engine.outcome import CRIT_WRITE_VARIANTS, crit_variants

    s = roark_state(
        tables,
        names,
        player_species="Lopunny",
        player_ability="Klutz",
        player_moves=["Tackle"],
        opp_moves=["Gastro Acid"],
        opp_faster=True,
    )
    s = _item(s, 0, 0, "Lansat Berry", names)
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 4)
    acts = A("Tackle", "Gastro Acid", s, names)
    w = CRIT_WRITE_VARIANTS

    def opened(board):
        odds, ok = (np.asarray(x) for x in crit_variants(tables, board, acts))
        return bool(np.any(ok[0, 1:w] & (odds[0, 1:w] != odds[0, 0])))

    assert opened(s)
    assert not opened(s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 4 + 1))


def test_own_tempo_and_safeguard_stop_the_berry_confusion(tables, names):
    """G-5: subscript_confuse.s:6-8 for a held item: the holder's own Own Tempo, or Safeguard
    on its side, ends the script with no draw; the heal stays."""
    s = _wiki_state(tables, names, ADAMANT, 40)
    acts = A("Leer", "Leer", s, names)
    guarded = s._replace(safeguard=s.safeguard.at[0].set(3))
    out = assert_matches_reference(tables, guarded, acts, O)
    assert int(out.hp[0, 0]) == 50 and int(out.confusion[0]) == 0
    n = names
    tempo = make_state(
        tables,
        [_mon(n, "Smeargle", ["Tackle"], 30, "Wiki Berry")],  # ability slot 0: Own Tempo
        [_mon(n, "Geodude", ["Tackle"], 20)],
    )
    tempo = tempo._replace(nature=tempo.nature.at[0, 0].set(ADAMANT))
    tempo = tempo.replace_hp(0, 0, int(tempo.max_hp[0, 0]) // 2)
    out = assert_matches_reference(tables, tempo, (0, 0), O)
    assert int(out.confusion[0]) == 0 and int(out.item[0, 0]) == 0


def test_pluck_eats_a_wiki_berry(tables, names):
    """G-6: Bug Bite eats the foe's Wiki Berry (PLUCK_EFFECT_HP_RESTORE_DRY): the eater heals
    ``Divide(maxHP, 8)`` below full HP and, disliking Dry, is confused (pending); at full HP it
    only eats it."""
    s = roark_state(tables, names, player_moves=["Bug Bite"])
    s = _item(s, 1, 0, "Wiki Berry", names)
    s = _full(s, 0, 0, 80).replace_hp(0, 0, 70)
    s = s._replace(nature=s.nature.at[0, 0].set(ADAMANT))
    acts = A("Bug Bite", "Leer", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.item[1, 0]) == 0 and int(out.item_consumed[1, 0]) == names.item("Wiki Berry")
    assert int(out.hp[0, 0]) == 80 and int(out.confusion[0]) == CONFUSION_PENDING
    full = s.replace_hp(0, 0, 80)
    out = assert_matches_reference(tables, full, acts, O)
    assert int(out.item[1, 0]) == 0 and int(out.confusion[0]) == 0


def test_a_replacement_eats_its_berry_with_a_pending_counter(tables, names):
    """G-5: a faint replacement's switch-in item check (``replace``, outside any turn) leaves
    the counter pending; both engines agree."""
    s = roark_state(tables, names)
    s = _full(_item(s, 0, 1, "Wiki Berry", names), 0, 1, 80).replace_hp(0, 1, 30)
    s = s._replace(nature=s.nature.at[0, 1].set(ADAMANT)).replace_hp(0, 0, 0)
    out = replace(tables, s, 0, 1)
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(s), 0, 1))
    for name, x, y in zip(out._fields, out, ref, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)
    assert int(out.hp[0, 1]) == 40 and int(out.confusion[0]) == CONFUSION_PENDING


# --- Pressure (G-8 .. G-11) ---------------------------------------------------------------


def _pressure(s, side):
    """``s`` with ``side``'s active battler holding Pressure (its ``BattleMon.ability``)."""
    mon = int(s.active[side])
    return s._replace(ability=s.ability.at[side, mon].set(AbilityId.PRESSURE))


def _pp(s, side, move, names):
    return int(s.pp[side, int(s.active[side]), names.move_slot(s, side, move)])


def test_tables_map_pressure_in_both_games(tables, names, kaizo_tables, kaizo_names):
    """G-8: every species slot with the decomp's ABILITY_PRESSURE (46) carries
    ``AbilityId.PRESSURE`` in both games' tables, and no other slot does."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        decomp = np.asarray(t.species_abilities_decomp) == n.ability("Pressure")
        engine = np.asarray(t.species_abilities) == AbilityId.PRESSURE
        assert decomp.any() and (decomp == engine).all()
    absol = names.species("Absol")
    assert AbilityId.PRESSURE in [int(a) for a in np.asarray(tables.species_abilities[absol])]


@pytest.mark.parametrize(
    ("move", "cost"),
    [
        ("Tackle", 2),  # RANGE_SINGLE_TARGET: the defender's Pressure (bcp.c:2261-2266)
        ("Growl", 2),  # RANGE_ADJACENT_OPPONENTS: COUNT_ALIVE_BATTLERS_THEIR_SIDE
        ("Earthquake", 2),  # RANGE_ALL_ADJACENT: COUNT_ALIVE_BATTLERS_EXCEPT_ME
        ("Sunny Day", 2),  # RANGE_FIELD: likewise, its defender is the user
        ("Spikes", 2),  # RANGE_OPPONENT_SIDE
        ("Imprison", 2),  # RANGE_USER, counted by move id (:2239-2240)
        ("Swords Dance", 1),  # RANGE_USER
        ("Reflect", 1),  # RANGE_USER_SIDE
        ("Counter", 1),  # RANGE_SINGLE_TARGET_SPECIAL: the user is the defender; it fails here
    ],
)
def test_pressure_pp_cost_by_range(tables, names, move, cost):
    """G-8: BattleControllerPlayer_DecrementPP's cost against a Pressure foe, by moveTemp's range;
    without Pressure every move costs 1, and the user's own Pressure never counts."""
    s = roark_state(tables, names, player_moves=[move])
    before = _pp(s, 0, move, names)
    acts = A(move, "Leer", s, names)
    out = assert_matches_reference(tables, _pressure(s, 1), acts, O)
    assert before - _pp(out, 0, move, names) == cost
    out = assert_matches_reference(tables, s, acts, O)
    assert before - _pp(out, 0, move, names) == 1
    out = assert_matches_reference(tables, _pressure(s, 0), acts, O)
    assert before - _pp(out, 0, move, names) == 1


def test_pressure_runs_a_move_out(tables, names):
    """G-8: with 1 PP left the cost of 2 stops at 0 (bcp.c:2276-2280; the move still runs), and
    with 2 PP left one use empties the slot, so the move can no longer be chosen."""
    s = _pressure(roark_state(tables, names, player_moves=["Tackle"]), 1)
    slot = names.move_slot(s, 0, "Tackle")
    acts = A("Tackle", "Leer", s, names)
    for left in (1, 2):
        low = s._replace(pp=s.pp.at[0, 0, slot].set(left))
        out = assert_matches_reference(tables, low, acts, O)
        assert int(out.pp[0, 0, slot]) == 0
        assert int(out.hp[1, 0]) < int(s.hp[1, 0])  # the move ran
        assert not bool(legal_actions(tables, out, 0)[slot])


def test_pressure_counter_spends_one_more(tables, names):
    """G-9: a Counter that passes runs BattleSystem_DecPPForPressure on its new defender
    (battle_script.c:4646; battle_lib.c:7390-7397): 1 at DecrementPP, 1 more for the Pressure
    foe; a Counter that fails spends only the 1."""
    s = roark_state(tables, names, player_moves=["Counter"], opp_faster=True)
    before = _pp(s, 0, "Counter", names)
    acts = A("Counter", "Tackle", s, names)
    out = assert_matches_reference(tables, _pressure(s, 1), acts, O)
    assert before - _pp(out, 0, "Counter", names) == 2
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])
    out = assert_matches_reference(tables, s, acts, O)
    assert before - _pp(out, 0, "Counter", names) == 1
    # With 1 PP left, nothing remains for the second decrement.
    slot = names.move_slot(s, 0, "Counter")
    low = _pressure(s, 1)._replace(pp=s.pp.at[0, 0, slot].set(1))
    assert int(assert_matches_reference(tables, low, acts, O).pp[0, 0, slot]) == 0


def test_pressure_pursuit_on_a_switching_holder(tables, names):
    """G-9: Pursuit's hit before a switch spends 1 PP, and 1 more while PP is left when the
    switching battler has Pressure (battle_script.c:6915-6922)."""
    s = roark_state(tables, names, opp_moves=["Pursuit"])
    before = _pp(s, 1, "Pursuit", names)
    acts = A(SWITCH_TO_1, "Pursuit", s, names)
    out = assert_matches_reference(tables, _pressure(s, 0), acts, O)
    assert before - _pp(out, 1, "Pursuit", names) == 2
    out = assert_matches_reference(tables, s, acts, O)
    assert before - _pp(out, 1, "Pursuit", names) == 1


def test_pressure_switch_in_reveals_it_to_the_ai(tables, names):
    """G-10: SWITCH_IN_CHECK_STATE_PRESSURE's subscript_pressure.s tags the holder's ability
    (BattleMessage_AbilityTag, battle_script.c:11798-11809): the AI knows a player's Pressure
    lead from the battle start and a Pressure newcomer from its switch-in."""
    n = names
    lead = make_state(tables, [_mon(n, "Absol", ["Tackle"])], [_mon(n, "Geodude", ["Tackle"])])
    assert int(lead.ability[0, 0]) == AbilityId.PRESSURE
    assert int(lead.ai_ability_known[0]) != 0
    s = make_state(
        tables,
        [_mon(n, "Geodude", ["Tackle"]), _mon(n, "Absol", ["Tackle"])],
        [_mon(n, "Geodude", ["Tackle"])],
    )
    assert int(s.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, (SWITCH_TO_1, 0), O)
    assert int(out.active[0]) == 1 and int(out.ai_ability_known[0]) != 0


# --- Status guards: Synchronize and Leaf Guard (G-12 .. G-17) -----------------------------


def _spec(names, species: str, moves, level=30, item="", slot=0):
    """``_mon`` with an ability slot."""
    return dataclasses.replace(_mon(names, species, moves, level, item), ability_slot=slot)


def _ability(s, side, ability, slot=None):
    """``s`` with ``side``'s party ``slot`` (the active one by default) holding ``ability``."""
    mon = int(s.active[side]) if slot is None else slot
    return s._replace(ability=s.ability.at[side, mon].set(ability))


def _weather(s, weather):
    return with_weather(s, weather)


def _status(s, side, slot, status):
    return s._replace(status=s.status.at[side, slot].set(status))


def _proc(o, field, side, hit=0):
    """``o`` with ``field[side, hit] = 1`` (a per-hit roll that passes)."""
    arr = np.asarray(getattr(o, field)).copy()
    arr[side, hit] = 1
    return o._replace(**{field: jnp.asarray(arr)})


def _sync_duel(tables, names, foe_move, foe="Rattata", player_item=""):
    """The player's Abra (Synchronize, slot 0) with Growl and Tackle against a level-30 foe
    using ``foe_move``."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Abra", ["Growl", "Tackle"], 30, player_item)],
        [_spec(n, foe, [foe_move, "Growl"], 30)],
    )
    assert int(s.ability[0, 0]) == AbilityId.SYNCHRONIZE
    return s


def test_tables_map_leaf_guard_and_synchronize_in_both_games(
    tables, names, kaizo_tables, kaizo_names
):
    """G-12, G-14: every species slot with the decomp's ABILITY_SYNCHRONIZE (28) /
    ABILITY_LEAF_GUARD (102) carries the engine id in both games' tables, and no other does."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for name, ability in (
            ("Synchronize", AbilityId.SYNCHRONIZE),
            ("Leaf Guard", AbilityId.LEAF_GUARD),
        ):
            decomp = np.asarray(t.species_abilities_decomp) == n.ability(name)
            engine = np.asarray(t.species_abilities) == ability
            assert decomp.any() and (decomp == engine).all(), name


@pytest.mark.parametrize(
    ("move", "held", "passed"),
    [
        ("Thunder Wave", Status.PARALYSIS, Status.PARALYSIS),
        ("Glare", Status.PARALYSIS, Status.PARALYSIS),
        ("Will-O-Wisp", Status.BURN, Status.BURN),
        ("PoisonPowder", Status.POISON, Status.POISON),
        ("Toxic", Status.TOXIC, Status.POISON),  # any poison runs subscript_poison.s
    ],
)
def test_synchronize_passes_a_status_moves_status_back(tables, names, move, held, passed):
    """G-12: the defender whose Synchronize is the sideEffectMon passes its new status to the
    attacker (battle_lib.c:4499-4504, :4514-4520); a bad poison passes as a poison. Its message
    tags the holder's ability (subscript_poison.s:57, _burn.s:68, _paralyze.s:44): the AI learns
    a player's Synchronize. Another ability passes nothing."""
    s = _sync_duel(tables, names, move)
    acts = A("Growl", move, s, names)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.status[0, 0]) == held
    assert int(out.status[1, 0]) == passed
    assert int(out.ai_ability_known[0]) != 0
    out = assert_matches_reference(tables, _ability(s, 0, AbilityId.INNER_FOCUS), acts, O)
    assert int(out.status[0, 0]) == held and int(out.status[1, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 0


@pytest.mark.parametrize(
    ("ability", "status"),
    [
        (AbilityId.STATIC, Status.PARALYSIS),
        (AbilityId.POISON_POINT, Status.POISON),
        (AbilityId.FLAME_BODY, Status.BURN),
    ],
)
def test_synchronize_passes_a_contact_abilitys_status(tables, names, ability, status):
    """G-12: a Synchronize attacker that a Static-like ability statused (TriggerAbilityOnHit sets
    sideEffectMon to the attacker, battle_lib.c:4225) passes the status to the holder
    (:4506-4511)."""
    s = _ability(_sync_duel(tables, names, "Growl"), 1, ability)
    o = _proc(O, "contact_proc", 1)
    out = assert_matches_reference(tables, s, A("Tackle", "Growl", s, names), o)
    assert int(out.status[0, 0]) == status
    assert int(out.status[1, 0]) == status
    assert int(out.ai_ability_known[0]) != 0


def test_synchronize_after_kings_rock_passes_nothing(tables, names):
    """G-13: CheckExtraFlinch makes the defender the sideEffectMon once the attacker's King's
    Rock passes its own tests (bcp.c:4636-4643), after the contact ability: the paralyzed
    Synchronize attacker is no longer the sideEffectMon and passes nothing."""
    s = _sync_duel(tables, names, "Growl", player_item="King’s Rock")
    s = _ability(s, 1, AbilityId.STATIC)
    acts = A("Tackle", "Growl", s, names)
    o = _proc(O, "contact_proc", 1)
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.status[1, 0]) == Status.PARALYSIS  # no King's Rock roll: it passes
    out = assert_matches_reference(tables, s, acts, _proc(o, "flinch_item", 0))
    assert int(out.status[0, 0]) == Status.PARALYSIS
    assert int(out.status[1, 0]) == Status.NONE


def test_synchronize_passes_no_sleep_or_freeze(tables, names):
    """G-12: a sleep or a freeze picks no script (battle_lib.c:4514-4520) and never toggles the
    flag (subscript_fall_asleep.s, subscript_freeze.s)."""
    s = _sync_duel(tables, names, "Spore")
    out = assert_matches_reference(tables, s, A("Growl", "Spore", s, names), O)
    assert int(out.status[0, 0]) == Status.SLEEP and int(out.status[1, 0]) == Status.NONE
    s = _sync_duel(tables, names, "Ice Beam")
    o = _proc(O, "secondary", 1)
    out = assert_matches_reference(tables, s, A("Growl", "Ice Beam", s, names), o)
    assert int(out.status[0, 0]) == Status.FREEZE and int(out.status[1, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 0


def test_synchronize_receiver_checks(tables, names):
    """G-12: the pass is an ABILITY side effect: a receiver already statused or of an immune
    type takes nothing and no message is shown; its substitute and its side's Safeguard do not
    stop it (subscript_paralyze.s:20, :26: both skipped for SIDE_EFFECT_TYPE_ABILITY)."""
    s = _sync_duel(tables, names, "Thunder Wave")
    acts = A("Growl", "Thunder Wave", s, names)
    out = assert_matches_reference(tables, _status(s, 1, 0, Status.BURN), acts, O)
    assert int(out.status[0, 0]) == Status.PARALYSIS and int(out.status[1, 0]) == Status.BURN
    assert int(out.ai_ability_known[0]) == 0
    guarded = s._replace(
        safeguard=s.safeguard.at[1].set(3), substitute_hp=s.substitute_hp.at[1].set(20)
    )
    out = assert_matches_reference(tables, guarded, acts, O)
    assert int(out.status[1, 0]) == Status.PARALYSIS
    s = _sync_duel(tables, names, "Toxic", foe="Grimer")  # a Poison type
    out = assert_matches_reference(tables, s, A("Growl", "Toxic", s, names), O)
    assert int(out.status[0, 0]) == Status.TOXIC and int(out.status[1, 0]) == Status.NONE
    assert int(out.ai_ability_known[0]) == 0


def test_synchronize_pass_meets_leaf_guard_in_sun(tables, names):
    """G-12, G-14: the receiver's own Leaf Guard in sun stops the passed status silently
    (subscript_paralyze.s:6-8, :_170); out of sun it takes it."""
    s = _ability(_sync_duel(tables, names, "Thunder Wave"), 1, AbilityId.LEAF_GUARD)
    acts = A("Growl", "Thunder Wave", s, names)
    out = assert_matches_reference(tables, _weather(s, Weather.SUN), acts, O)
    assert int(out.status[0, 0]) == Status.PARALYSIS and int(out.status[1, 0]) == Status.NONE
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.status[1, 0]) == Status.PARALYSIS


def test_synchronize_after_roar_drags_into_toxic_spikes(tables, names):
    """G-13: Roar's drag runs the hazards in the move script; Toxic Spikes poisons a Synchronize
    newcomer (BtlCmd_CheckToxicSpikes makes it the sideEffectMon, battle_script.c:7640-7641)
    and AfterMoveEffects' Synchronize step passes the poison to the Roar user, before the
    switch-in effects (bcp.c:3665-3672): a Pecha Berry cures the newcomer only after the pass."""
    n = names
    for item, kept in (("", Status.POISON), ("Pecha Berry", Status.NONE)):
        s = make_state(
            tables,
            [_spec(n, "Rattata", ["Roar"], 30)],
            [_spec(n, "Geodude", ["Growl"], 20), _spec(n, "Abra", ["Growl"], 20, item)],
        )
        s = s._replace(toxic_spikes=s.toxic_spikes.at[1].set(1))
        out = assert_matches_reference(tables, s, A("Roar", "Growl", s, n), O)
        assert int(out.active[1]) == 1
        assert int(out.status[1, 1]) == kept
        assert int(out.status[0, 0]) == Status.POISON


def test_synchronize_after_u_turn_into_toxic_spikes(tables, names):
    """G-13: U-turn clears the Synchronize flag before its list
    (subscript_attack_then_switch_out.s:69); the newcomer's Toxic Spikes poison sets it, and the
    resumed tail passes the poison (two layers: the bad poison as a poison) to the defender;
    the AI learns the player's Synchronize."""
    n = names
    for layers, own in ((1, Status.POISON), (2, Status.TOXIC)):
        s = make_state(
            tables,
            [_spec(n, "Scyther", ["U-turn"], 30), _spec(n, "Abra", ["Growl"], 30)],
            [_spec(n, "Rattata", ["Growl"], 40), _spec(n, "Rattata", ["Growl"], 30)],
        )
        s = s._replace(toxic_spikes=s.toxic_spikes.at[0].set(layers))
        paused = assert_matches_reference(tables, s, A("U-turn", "Growl", s, n), O)
        assert int(paused.pivot_battler) == 1
        acts = np.asarray(paused.ts_actions).astype(int).tolist()
        acts[0] = MOVE_ACTIONS + 1
        out = assert_matches_reference(tables, paused, tuple(acts), O)
        assert int(out.active[0]) == 1 and int(out.status[0, 1]) == own
        assert int(out.status[1, 0]) == Status.POISON
        assert int(out.ai_ability_known[0]) != 0


@pytest.mark.parametrize(
    ("move", "status"),
    [
        ("Thunder Wave", Status.PARALYSIS),
        ("Will-O-Wisp", Status.BURN),
        ("PoisonPowder", Status.POISON),
        ("Toxic", Status.TOXIC),
        ("Spore", Status.SLEEP),
    ],
)
def test_leaf_guard_in_sun_blocks_a_status_move(tables, names, move, status):
    """G-14: CheckIgnoreWeather, then FIELD_CONDITION_SUNNY, then an ignorable Leaf Guard
    (subscript_paralyze.s:6-8, _burn.s:21-23, _poison.s:14-16, _badly_poison.s:32-34,
    _fall_asleep.s:23-25): in sun the status move is ineffective and its message tags the
    ability (the AI learns it). Out of sun, under a Cloud Nine (BtlCmd_CheckIgnoreWeather,
    battle_script.c:9195-9206) or against a Mold Breaker attacker the status lands."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Tangela", ["Growl"], 30, slot=1)],
        [_spec(n, "Rattata", [move, "Growl"], 30)],
    )
    assert int(s.ability[0, 0]) == AbilityId.LEAF_GUARD
    acts = A("Growl", move, s, n)
    sunny = _weather(s, Weather.SUN)
    out = assert_matches_reference(tables, sunny, acts, O)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.status[0, 0]) == status and int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, _ability(sunny, 1, AbilityId.MOLD_BREAKER), acts, O)
    assert int(out.status[0, 0]) == status
    locked = make_state(
        tables,
        [_spec(n, "Tangela", ["Growl"], 30, slot=1)],
        [_spec(n, "Psyduck", [move, "Growl"], 30, slot=1)],  # Cloud Nine
    )
    locked_acts = A("Growl", move, locked, n)
    out = assert_matches_reference(tables, _weather(locked, Weather.SUN), locked_acts, O)
    assert int(out.status[0, 0]) == status


def test_leaf_guard_in_sun_stops_a_secondary_silently(tables, names):
    """G-14: an indirect status (Ember's burn) on a Leaf Guard holder in sun ends silently
    (subscript_burn.s:_264: SIDE_EFFECT_TYPE_INDIRECT jumps to the end): no burn, no reveal."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Tangela", ["Growl"], 40, slot=1)],
        [_spec(n, "Rattata", ["Ember", "Growl"], 20)],
    )
    acts = A("Growl", "Ember", s, n)
    o = _proc(O, "secondary", 1)
    out = assert_matches_reference(tables, _weather(s, Weather.SUN), acts, o)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.status[0, 0]) == Status.BURN


def test_leaf_guard_in_sun_stops_static_on_its_attacker(tables, names):
    """G-14: the attacker's own Leaf Guard in sun stops the paralysis of the defender's Static
    (subscript_paralyze.s:6-8 on the attacker, an ABILITY side effect: silent)."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Tangela", ["Tackle"], 20, slot=1)],
        [_spec(n, "Pikachu", ["Growl"], 30)],
    )
    acts = A("Tackle", "Growl", s, n)
    o = _proc(O, "contact_proc", 1)
    out = assert_matches_reference(tables, _weather(s, Weather.SUN), acts, o)
    assert int(out.status[0, 0]) == Status.NONE
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.status[0, 0]) == Status.PARALYSIS


# --- Wonder Guard and Download (G-18 .. G-23) ----------------------------------------------


def _guard_duel(tables, names, foe_moves, foe="Rattata", player="Squirtle"):
    """The player's level-30 ``player`` (Squirtle: Water) given Wonder Guard (its
    ``BattleMon.ability``), Tackle and Growl, against a level-30 ``foe`` knowing ``foe_moves``."""
    n = names
    s = make_state(
        tables,
        [_spec(n, player, ["Tackle", "Growl"], 30)],
        [_spec(n, foe, list(foe_moves), 30)],
    )
    return _ability(s, 0, AbilityId.WONDER_GUARD)


def test_tables_map_wonder_guard_and_download_in_both_games(
    tables, names, kaizo_tables, kaizo_names
):
    """G-18, G-21: every species slot with the decomp's ABILITY_WONDER_GUARD (25) /
    ABILITY_DOWNLOAD (88) carries the engine id in both games' tables, and no other does;
    Fire Fang is the only BATTLE_EFFECT_FLINCH_BURN_HIT (273) move."""
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for name, ability in (
            ("Wonder Guard", AbilityId.WONDER_GUARD),
            ("Download", AbilityId.DOWNLOAD),
        ):
            decomp = np.asarray(t.species_abilities_decomp) == n.ability(name)
            engine = np.asarray(t.species_abilities) == ability
            assert decomp.any() and (decomp == engine).all(), name
        fangs = np.flatnonzero(np.asarray(t.move_ai_effect) == 273).tolist()
        assert fangs == [n.move("Fire Fang")]


@pytest.mark.parametrize(
    ("move", "guarded"),
    [
        ("Water Gun", True),  # Water on Water: resisted
        ("Tackle", True),  # neutral
        ("Bubble", True),  # resisted, its secondary effect never runs
        ("ThunderShock", False),  # super effective: MOVE_STATUS_SUPER_EFFECTIVE set
        ("Fire Fang", False),  # resisted, but never on its damaging turn (lib.c:7600)
        ("Ice Fang", True),  # resisted: BATTLE_EFFECT_FLINCH_FREEZE_HIT is not listed
        ("Night Shade", True),  # fixed damage, power 1: Ghost on Water is neutral
        ("Seismic Toss", True),
        ("Thunder Wave", False),  # power 0: never guarded (lib.c:2654)
        ("Growl", False),
    ],
)
def test_wonder_guard_blocks_what_is_not_super_effective(tables, names, move, guarded):
    """G-18: ApplyTypeChart's MOVE_STATUS_WONDER_GUARD (battle_lib.c:2649-2654) stops a damaging
    move that is not super effective: no damage, no secondary effect; the message of
    subscript_missed.s:107-110 tags the holder (G-20). A super-effective hit, Fire Fang and the
    status moves are untouched."""
    s = _guard_duel(tables, names, [move, "Growl"])
    acts = A("Growl", move, s, names)
    o = _proc(O, "secondary", 1)._replace(fang_flinch=jnp.ones((2,), jnp.int8))
    out = assert_matches_reference(tables, s, acts, o)
    hp, after = int(s.hp[0, 0]), int(out.hp[0, 0])
    if guarded:
        assert after == hp and int(out.status[0, 0]) == Status.NONE
        assert int(out.stages[0, Stat.SPE]) == 0  # Bubble's Speed drop never ran
        assert int(out.ai_ability_known[0]) != 0
    else:
        assert int(out.ai_ability_known[0]) == 0
        if move == "Thunder Wave":
            assert int(out.status[0, 0]) == Status.PARALYSIS
        elif move == "Growl":
            assert int(out.stages[0, Stat.ATK]) == -1
        else:
            assert after < hp


def test_wonder_guard_mold_breaker_and_the_miss(tables, names):
    """G-18, G-20: an attacker's Mold Breaker hides the ability (Battler_IgnorableAbility):
    the hit lands and nothing is revealed. A miss still reaches subscript_missed.s:107 (the
    plain-miss message is :157): the AI learns the ability."""
    s = _guard_duel(tables, names, ["Tackle", "Growl"])
    acts = A("Growl", "Tackle", s, names)
    out = assert_matches_reference(tables, _ability(s, 1, AbilityId.MOLD_BREAKER), acts, O)
    assert int(out.hp[0, 0]) < int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 0
    missed = O._replace(hit=O.hit.at[1, 0].set(0))
    out = assert_matches_reference(tables, s, acts, missed)
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) != 0


def test_wonder_guard_jump_kick_crashes_on_the_charts_damage(tables, names):
    """G-19: a guarded Jump Kick ends in subscript_missed.s, whose crash
    (subscript_crash_on_miss.s) reads battleCtx->damage as ApplyTypeChart left it, not 0: the
    user loses min(Divide(damage, 2), Divide(defender maxHP, 2)); the holder takes nothing."""
    s = _guard_duel(tables, names, ["Jump Kick", "Growl"], foe="Hitmonlee")
    acts = A("Growl", "Jump Kick", s, names)
    out = assert_matches_reference(tables, s, acts, O)
    kick = names.move("Jump Kick")
    raw = int(damage_table(tables, s, 1, kick, False, wonder_guard=False)[15])
    assert raw > 0 and int(damage_table(tables, s, 1, kick, False)[15]) == 0
    loss = min(max(raw // 2, 1), max(int(s.max_hp[0, 0]) // 2, 1))
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])
    assert int(s.hp[1, 0]) - int(out.hp[1, 0]) == loss


def test_wonder_guard_skips_the_after_move_items(tables, names):
    """G-19: a guarded move skips AfterMoveEffects (CheckMoveFailure, bcp.c:3324-3327); both
    engines agree on the crashing attacker holding a Sitrus Berry."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Tackle", "Growl"], 30)],
        [_spec(n, "Hitmonlee", ["Jump Kick", "Growl"], 30, "Sitrus Berry")],
    )
    s = _ability(s, 0, AbilityId.WONDER_GUARD)
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]) // 2 + 2)
    out = assert_matches_reference(tables, s, A("Growl", "Jump Kick", s, n), O)
    assert int(out.hp[1, 0]) != int(s.hp[1, 0])


def test_wonder_guard_shedinja_reveals_and_falls_to_a_super_effective_hit(tables, names):
    """G-18: Shedinja (Bug / Ghost, 1 HP) shrugs off a neutral Water Gun and a Ghost-immune
    Tackle (both bits: subscript_missed.s tests WONDER_GUARD first, :107, so the ability shows),
    and faints to Ember (Fire on Bug, super effective)."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Shedinja", ["Growl"], 30), _spec(n, "Squirtle", ["Growl"], 30)],
        [_spec(n, "Rattata", ["Water Gun", "Tackle", "Ember"], 30)],
    )
    assert int(s.ability[0, 0]) == AbilityId.WONDER_GUARD
    for move in ("Water Gun", "Tackle"):
        out = assert_matches_reference(tables, s, A("Growl", move, s, n), O)
        assert int(out.hp[0, 0]) == 1 and int(out.ai_ability_known[0]) != 0
    out = assert_matches_reference(tables, s, A("Growl", "Ember", s, n), O)
    assert int(out.hp[0, 0]) == 0


def _download_lead(tables, names, foe, slot=1):
    """Porygon (Trace / Download: ``slot`` 1 is Download) leading against ``foe``."""
    n = names
    return make_state(
        tables, [_spec(n, "Porygon", ["Tackle"], 30, slot=slot)], [_spec(n, foe, ["Tackle"], 30)]
    )


def test_download_at_the_battle_start(tables, names):
    """G-21: SWITCH_IN_CHECK_STATE_DOWNLOAD (battle_lib.c:3814-3851) at ShowBattleMon: the foe's
    Defense against its Sp. Def (each through its stage): Defense >= Sp. Def raises Sp. Attack,
    else Attack, by one; no AI reveal (the message is built in C, battle_script.c:2854)."""
    for foe in ("Geodude", "Tentacool"):
        s = _download_lead(tables, names, foe)
        assert int(s.ability[0, 0]) == AbilityId.DOWNLOAD
        defense, sp_def = int(s.stats[1, 0, Stat.DEF - 1]), int(s.stats[1, 0, Stat.SPD - 1])
        assert defense != sp_def
        raised = Stat.SPA if defense >= sp_def else Stat.ATK
        other = Stat.ATK if raised == Stat.SPA else Stat.SPA
        assert int(s.stages[0, raised]) == 1 and int(s.stages[0, other]) == 0, foe
        assert int(s.ai_ability_known[0]) == 0
    # Trace copies a foe's Download first (SWITCH_IN_CHECK_STATE_TRACE, :3718), then it fires.
    n = names
    s = make_state(
        tables,
        [_spec(n, "Porygon", ["Tackle"], 30, slot=0)],
        [_spec(n, "Porygon", ["Tackle"], 30, slot=1)],
    )
    assert int(s.ability[0, 0]) == AbilityId.DOWNLOAD
    assert int(s.stages[0, Stat.SPA]) + int(s.stages[0, Stat.ATK]) == 1
    assert int(s.stages[1, Stat.SPA]) + int(s.stages[1, Stat.ATK]) == 1


def test_download_on_a_switch_in_reads_the_stages_and_the_substitute(tables, names):
    """G-21: a switched-in Download holder reads the foe's staged Defense and Sp. Def (a -6
    Defense turns Geodude's ratio round: Attack); a foe behind a substitute counts 0, so
    nothing rises."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Tackle"], 30), _spec(n, "Porygon", ["Tackle"], 30, slot=1)],
        [_spec(n, "Geodude", ["Tackle"], 30)],
    )
    acts = (SWITCH_TO_1, 0)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.stages[0, Stat.SPA]) == 1 and int(out.stages[0, Stat.ATK]) == 0
    low = s._replace(stages=s.stages.at[1, Stat.DEF].set(-6))
    out = assert_matches_reference(tables, low, acts, O)
    assert int(out.stages[0, Stat.ATK]) == 1 and int(out.stages[0, Stat.SPA]) == 0
    behind = s._replace(substitute_hp=s.substitute_hp.at[1].set(10))
    out = assert_matches_reference(tables, behind, acts, O)
    assert int(out.stages[0, Stat.ATK]) == 0 and int(out.stages[0, Stat.SPA]) == 0


def test_download_after_a_double_ko_fires_on_the_opposing_newcomer(tables, names):
    """G-22: the player's Download newcomer enters against a fainted foe (no sum, no raise);
    ``replacement_pass`` fires it once the opponent's newcomer is in. At a single replacement it
    fires on entry, once."""
    from nuzline.search.playout import replacement_pass
    from tests.search.fixtures import at_replacement

    s = at_replacement(roark_state(tables, names), opp_down=True)
    s = _ability(s, 0, AbilityId.DOWNLOAD, slot=4)
    after, refused = replacement_pass(tables, s, 4)
    assert not bool(refused)
    foe = int(after.active[1])
    assert int(after.hp[1, foe]) > 0
    defense = int(after.stats[1, foe, Stat.DEF - 1])
    sp_def = int(after.stats[1, foe, Stat.SPD - 1])
    raised = Stat.SPA if defense >= sp_def else Stat.ATK
    assert int(after.stages[0, raised]) == 1
    assert int(after.stages[0, Stat.SPA]) + int(after.stages[0, Stat.ATK]) == 1
    single = _ability(at_replacement(roark_state(tables, names)), 0, AbilityId.DOWNLOAD, slot=4)
    after, refused = replacement_pass(tables, single, 4)
    assert not bool(refused)
    assert int(after.stages[0, Stat.SPA]) + int(after.stages[0, Stat.ATK]) == 1


def test_ai_reads_wonder_guard(tables, names):
    """G-23: IfMoveEffectivenessEquals and TrainerAI_CalcDamage see MOVE_STATUS_IMMUNE for a
    guarded move (the true ability; trainer_ai.c:1339, :3106), not for a super-effective one;
    the AI no longer refuses a Wonder Guard player."""
    from nuzline.ai import commands, policy

    n = names
    s = make_state(
        tables,
        [_spec(n, "Shedinja", ["Growl"], 30)],
        [_spec(n, "Geodude", ["Tackle", "Rock Throw", "Leer"], 12)],
    )
    tackle, rock = n.move("Tackle"), n.move("Rock Throw")
    assert int(commands.effectiveness_class(tables, s, rock)) == commands.TypeMulti.DOUBLE
    assert int(commands.ai_damage(tables, s, rock, 15)) > 0
    water = jnp.asarray([n.type("Water")] * 2, s.types.dtype)
    s_water = s._replace(types=s.types.at[0, 0].set(water))
    assert int(commands.effectiveness_class(tables, s_water, tackle)) == commands.TypeMulti.IMMUNE
    assert int(commands.ai_damage(tables, s_water, tackle, 15)) == 0
    assert not any("Wonder Guard" in r for r in policy.ai_refusals(tables, s, n))


def test_ai_switches_against_wonder_guard(tables, names):
    """G-23: AI_CannotDamageWonderGuard (trainer_ai.c:3286-3348): the player's Shedinja, no
    super-effective move on the AI's active: each bench Pokemon's super-effective move draws
    ``RandNext() % 3 < 2`` (43691 of 65536), the first success switching to it. With a
    super-effective move on the active there is no such draw (its keep draw instead)."""
    from nuzline.ai.outcome import switch_stop_probs
    from nuzline.ai.switching import NO_SWITCH, SwitchSite, switch_action, switch_sites

    n = names
    s = make_state(
        tables,
        [_spec(n, "Shedinja", ["Growl"], 30)],
        [
            _spec(n, "Geodude", ["Tackle", "Leer"], 12),
            _spec(n, "Onix", ["Tackle", "Rock Throw"], 12),
            _spec(n, "Cranidos", ["Headbutt"], 14),
        ],
    )
    sites = switch_sites(tables, s).tolist()
    assert sites == [SwitchSite.WONDER_GUARD] + [SwitchSite.NONE] * 8
    assert int(switch_action(tables, s, jnp.int8(0))) == MOVE_ACTIONS + 1
    assert int(switch_action(tables, s, jnp.int8(9))) == NO_SWITCH
    p = np.asarray(switch_stop_probs(switch_sites(tables, s)), np.float64)
    assert abs(p[0] - 43691 / 65536) < 1e-6 and abs(p[-1] - 21845 / 65536) < 1e-6
    keep = s._replace(moves=s.moves.at[1, 0, 1].set(n.move("Rock Throw")))
    assert switch_sites(tables, keep).tolist() == [SwitchSite.KEEP] + [SwitchSite.NONE] * 8


def test_ai_bench_wonder_guard_is_immune_to_the_last_hit(tables, names):
    """G-23: the passes' first CalcEffectiveness has the AI's bench Pokemon as the defender
    (trainer_ai.c:3769-3777): a bench Shedinja is INEFFECTIVE against a neutral moveHit (Water
    Gun) through Wonder Guard, so its super-effective move is a pass-1 draw; against a Mold
    Breaker hitter (Battler_Ability of moveHitBattler) it is not."""
    from nuzline.ai.switching import SwitchSite, switch_sites

    n = names
    for slot, expected in ((0, SwitchSite.PASS_INEFFECTIVE), (1, SwitchSite.NONE)):
        s = make_state(
            tables,
            [_spec(n, "Pinsir", ["Water Gun"], 30, slot=slot)],
            [_spec(n, "Geodude", ["Tackle"], 12), _spec(n, "Shedinja", ["Aerial Ace"], 30)],
        )
        s = s._replace(move_hit=s.move_hit.at[1].set(n.move("Water Gun")))
        assert switch_sites(tables, s).tolist()[0] == expected, slot


def test_wonder_guard_pursuit_on_a_switching_holder(tables, names):
    """G-20: Pursuit's hit on a switching Wonder Guard holder (subscript_switch_pokemon.s:11;
    subscript_pursuit.s:17, :75-77) ends in subscript_missed.s: no damage, and its :54-56 message
    tags the ability (BattleMessage_AbilityTag, battle_script.c:11806). The switch then runs
    SwitchAndUpdateMon (subscript_switch_pokemon.s:22; BattleSystem_UpdateAfterSwitch,
    battle_lib.c:2152), which clears battler 0's ability record: the AI does not keep the tag."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Tackle"], 30), _spec(n, "Abra", ["Tackle"], 30)],
        [_spec(n, "Rattata", ["Pursuit"], 30)],
    )
    s = _ability(s, 0, AbilityId.WONDER_GUARD)
    out = assert_matches_reference(tables, s, A(SWITCH_TO_1, "Pursuit", s, n), O)
    assert int(out.active[0]) == 1
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 0


# --- the follow-ups (G-24 .. G-26) ---------------------------------------------------------


def test_synchronize_u_turn_user_fainted_by_pursuit_passes_its_status(tables, names):
    """G-24: a Synchronize U-turn user paralyzed by the foe's Static (TriggerAbilityOnHit,
    subscript_attack_then_switch_out.s:7) that falls to the foe's Pursuit (:54) ends the script at
    :55, before the flag clear (:69): AfterMoveEffects' Synchronize step reads its
    BattleMon.status, which the faint left (only the party copy is zeroed,
    battle_controller.c:1481-1486), and passes the paralysis to the pursuer."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Abra", ["U-turn"], 30), _spec(n, "Abra", ["Growl"], 30)],
        [_spec(n, "Rattata", ["Pursuit"], 30)],
    )
    s = _ability(s, 1, AbilityId.STATIC)._replace(hp=s.hp.at[0, 0].set(1))
    acts = A("U-turn", "Pursuit", s, n)
    out = assert_matches_reference(tables, s, acts, _proc(O, "contact_proc", 1))
    assert int(out.hp[0, 0]) == 0 and int(out.status[0, 0]) == Status.NONE
    assert int(out.status[1, 0]) == Status.PARALYSIS
    assert int(out.ai_ability_known[0]) != 0
    # Without the Static paralysis nothing passes.
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.hp[0, 0]) == 0 and int(out.status[1, 0]) == Status.NONE


def test_synchronize_newcomer_fainted_by_entry_hazards_passes_its_poison(tables, names):
    """G-24: a U-turn's Synchronize newcomer poisoned by Toxic Spikes and fainted by Stealth Rock
    in the list (subscript_show_party_list.s:22-24; the hazards check runs Toxic Spikes first,
    subscript_hazards_check.s:14-41) keeps its BattleMon.status: the resumed tail's Synchronize
    step passes the poison to the defender. A Roar's dragged newcomer likewise."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Scyther", ["U-turn"], 30), _spec(n, "Abra", ["Growl"], 30)],
        [_spec(n, "Rattata", ["Growl"], 40), _spec(n, "Rattata", ["Growl"], 30)],
    )
    s = s._replace(
        toxic_spikes=s.toxic_spikes.at[0].set(1),
        stealth_rock=s.stealth_rock.at[0].set(1),
        hp=s.hp.at[0, 1].set(1),
    )
    paused = assert_matches_reference(tables, s, A("U-turn", "Growl", s, n), O)
    assert int(paused.pivot_battler) == 1
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[0] = MOVE_ACTIONS + 1
    out = assert_matches_reference(tables, paused, tuple(acts), O)
    assert int(out.hp[0, 1]) == 0 and int(out.status[0, 1]) == Status.NONE
    assert int(out.status[1, 0]) == Status.POISON
    s = make_state(
        tables,
        [_spec(n, "Rattata", ["Roar"], 30)],
        [_spec(n, "Geodude", ["Growl"], 20), _spec(n, "Abra", ["Growl"], 20)],
    )
    s = s._replace(
        toxic_spikes=s.toxic_spikes.at[1].set(1),
        stealth_rock=s.stealth_rock.at[1].set(1),
        hp=s.hp.at[1, 1].set(1),
    )
    out = assert_matches_reference(tables, s, A("Roar", "Growl", s, n), O)
    assert int(out.hp[1, 1]) == 0 and int(out.status[0, 0]) == Status.POISON


def _standoff(tables, names, player_moves=("Tackle",)):
    """Porygon (Trace) against Porygon (Trace): neither copies (ChooseTraceTarget skips Trace,
    battle_lib.c:7750-7775), both stay armed; the opponent's bench Geodude holds Super Luck."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Porygon", list(player_moves), 30)],
        [_spec(n, "Porygon", ["Tackle"], 30), _spec(n, "Geodude", ["Tackle"], 30)],
    )
    assert int(s.ability[0, 0]) == AbilityId.TRACE and int(s.ability[1, 0]) == AbilityId.TRACE
    return _ability(s, 1, AbilityId.SUPER_LUCK, slot=1)


def test_trace_standoff_copies_the_next_newcomer_before_the_holders_move(tables, names):
    """G-25: an armed Trace holder copies the foe's switch-in at its switch-in checks
    (SWITCH_IN_CHECK_STATE_TRACE, battle_lib.c:3718-3737), before its own move: the step writes
    the copy and the laws read it (Super Luck: crit 1 in 8, ``outcome.pivot_view``). A Pursuit on
    the switch hits before it (subscript_switch_pokemon.s:11): its reads see Trace."""
    from nuzline.engine.outcome import pivot_view
    from tests.engine.test_enumerate import laws_of as outcome_laws  # jitted, not eager

    n = names
    s = _standoff(tables, names)
    acts = A("Tackle", SWITCH_TO_1, s, n)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.ability[0, 0]) == AbilityId.SUPER_LUCK
    view, _ = pivot_view(tables, s, acts)
    assert int(view.ability[0, 0]) == AbilityId.SUPER_LUCK
    assert int(outcome_laws(tables, s, acts).crit_variants[0, 0]) == 8
    assert int(outcome_laws(tables, s, A("Tackle", "Tackle", s, n)).crit_variants[0, 0]) == 16
    s = _standoff(tables, names, ("Pursuit",))
    acts = A("Pursuit", SWITCH_TO_1, s, n)
    assert int(pivot_view(tables, s, acts)[0].ability[0, 0]) == AbilityId.TRACE
    assert int(outcome_laws(tables, s, acts).crit_variants[0, 0]) == 16
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.ability[0, 0]) == AbilityId.SUPER_LUCK


def test_trace_standoff_copies_a_u_turn_newcomer(tables, names):
    """G-25: the foe's U-turn pick enters before the holder moves: at the paused list the laws'
    view gives the holder the copy (``pivot_view``), and the resume writes it."""
    from tests.engine.test_enumerate import laws_of as outcome_laws  # jitted, not eager

    n = names
    s = make_state(
        tables,
        [_spec(n, "Porygon", ["Tackle"], 20)],
        [_spec(n, "Porygon", ["U-turn"], 40), _spec(n, "Geodude", ["Tackle"], 30)],
    )
    s = _ability(s, 1, AbilityId.SUPER_LUCK, slot=1)
    paused = assert_matches_reference(tables, s, A("Tackle", "U-turn", s, n), O)
    assert int(paused.pivot_battler) == 2
    acts = np.asarray(paused.ts_actions).astype(int).tolist()
    acts[1] = MOVE_ACTIONS + 1
    assert int(outcome_laws(tables, paused, jnp.asarray(acts)).crit_variants[0, 0]) == 8
    out = assert_matches_reference(tables, paused, tuple(acts), O)
    assert int(out.ability[0, 0]) == AbilityId.SUPER_LUCK


def _war(tables, names, levels, items=("", "")):
    """Groudon (Drought) leading against Kyogre (Drizzle), both base Speed 90: equal levels tie."""
    n = names
    return make_state(
        tables,
        [_spec(n, "Groudon", ["Tackle"], levels[0], items[0])],
        [_spec(n, "Kyogre", ["Tackle"], levels[1], items[1])],
    )


def test_weather_war_at_the_battle_start_leaves_the_slower_leads_weather(tables, names):
    """G-26: the weather state runs over monSpeedOrder (battle_lib.c:3740-3790), sorted at
    TrainerMessage (battle_controller_player.c:251, battle_lib.c:3364-3382): the second lead's
    weather stays. A slower player's Drought runs its script last (the AI learns it)."""
    fast = _war(tables, names, (50, 40))
    assert int(weather_of(fast)) == Weather.RAIN and int(fast.war_swaps) == 0
    assert int(fast.ai_ability_known[0]) != 0  # its own script ran first
    slow = _war(tables, names, (40, 50))
    assert int(weather_of(slow)) == Weather.SUN and int(weather_timer_of(slow)) == 0
    assert int(slow.war_swaps) == 0 and int(slow.ai_ability_known[0]) != 0


def test_weather_war_tie_and_quick_claw_are_root_draws(tables, names):
    """G-26: an exact speed tie swaps on RandNext() & 1 (battle_lib.c:1445) and a lead's Quick
    Claw reads the battle-start speedRand (:1278-1286, BattleContext_InitCounters :2024): the
    order waits in ``war_swaps`` for the root chance node (``root_belief``, ``sample_root``)."""
    import jax

    from nuzline.engine.step import root_belief, sample_root

    tie = _war(tables, names, (50, 50))
    assert int(tie.war_swaps) == 0b10101010 and int(weather_of(tie)) == Weather.RAIN
    belief = root_belief(tables, tie)
    assert sorted((int(weather_of(b)), p) for b, p in belief) == sorted(
        [(int(Weather.SUN), 0.5), (int(Weather.RAIN), 0.5)]
    )
    assert all(int(b.war_swaps) == 0 for b, _ in belief)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    drawn = {int(weather_of(sample_root(k, tables, tie))) for k in keys}
    assert drawn == {Weather.SUN, Weather.RAIN}
    # The slower player's Quick Claw: activated (1/5) it sorts first and the rain stays.
    claw = _war(tables, names, (40, 50), ("Quick Claw", ""))
    assert int(claw.war_swaps) != 0
    belief = {(int(b.qc_roll[0]), int(weather_of(b))): p for b, p in root_belief(tables, claw)}
    assert belief == pytest.approx({(1, Weather.RAIN): 0.2, (0, Weather.SUN): 0.8})


def test_root_draws_take_a_host_state(tables, names):
    """A NumPy-backed state (``prep``'s and the CLI's, ``to_numpy`` of a state) at the root
    chance node: ``root_belief`` and ``sample_root`` settle its weather war as they do a JAX
    state's (the tie's ``reveal_ability`` writes with ``.at``)."""
    import jax
    import numpy as np

    from nuzline.engine.step import root_belief, sample_root

    tie = _war(tables, names, (50, 50))
    host = jax.tree.map(np.asarray, tie)
    assert isinstance(host.ai_ability_known, np.ndarray)

    def outcomes(state):
        belief = root_belief(tables, state)
        return sorted((int(weather_of(b)), int(b.ai_ability_known[0]), p) for b, p in belief)

    assert outcomes(host) == outcomes(tie) and len(outcomes(tie)) == 2
    key = jax.random.PRNGKey(3)
    assert int(weather_of(sample_root(key, tables, host))) == int(
        weather_of(sample_root(key, tables, tie))
    )


def _double_ko(tables, names, levels):
    """Geodude leads, both down; the player's bench Groudon (Drought, level ``levels[0]``) and
    the opponent's only bench Pokemon Kyogre (Drizzle, ``levels[1]``) come in together."""
    from tests.search.fixtures import at_replacement

    n = names
    s = make_state(
        tables,
        [_spec(n, "Geodude", ["Tackle"], 30), _spec(n, "Groudon", ["Tackle"], levels[0])],
        [_spec(n, "Geodude", ["Tackle"], 30), _spec(n, "Kyogre", ["Tackle"], levels[1])],
    )
    return at_replacement(s, opp_down=True)


def test_weather_war_at_a_double_ko(tables, names):
    """G-26: the replacements' switch-in states run together at ShowBattleMon, over the order
    TrainerMessage sorted on the newcomers (battle_controller_player.c:251, the turn's speedRand):
    the slower newcomer's weather stays; a tie waits in ``war_swaps``, a chance node of the
    tree's replacement children (``expand_replacements``) and of the playout's next turn."""
    import jax

    from nuzline.engine.step import resolve_war
    from nuzline.search.expand import expand_replacements
    from nuzline.search.playout import replacement_pass, settle_war
    from nuzline.search.rows import take

    for levels, weather in (((50, 40), Weather.RAIN), ((40, 50), Weather.SUN)):
        after, refused = replacement_pass(tables, _double_ko(tables, names, levels), 1)
        assert not bool(refused) and int(after.active[1]) == 1
        assert int(weather_of(after)) == weather and int(after.war_swaps) == 0
    s = _double_ko(tables, names, (50, 50))
    after, _ = replacement_pass(tables, s, 1)
    assert int(after.war_swaps) != 0 and int(weather_of(after)) == Weather.RAIN
    assert int(weather_of(resolve_war(after, 1))) == Weather.SUN
    assert int(weather_of(resolve_war(after, 0))) == Weather.RAIN
    keys = jax.random.split(jax.random.PRNGKey(1), 32)
    assert {int(weather_of(settle_war(after, k))) for k in keys} == {Weather.SUN, Weather.RAIN}
    batch = jax.tree.map(lambda f: np.asarray(f)[None], s)
    store, children = expand_replacements(tables, batch, [(0, MOVE_ACTIONS + 1)])
    kids = children[0]
    assert [k.prob for k in kids] == [0.5, 0.5]
    weathers = {int(weather_of(take(store, np.asarray([k.state_row])))[0]) for k in kids}
    assert weathers == {Weather.SUN, Weather.RAIN}


# --- The merge with E2's charge turn (G-27, G-28) -------------------------------------------


def test_wonder_guard_tests_a_charge_moves_strike_not_its_charge(tables, names):
    """G-27: the charge turn sets SYSCTL_FIRST_OF_MULTI_TURN, so CheckTypeChart never runs
    (bcp.c:2356): no Wonder Guard flag, no reveal. The strike runs
    subscript_charge_move_cleanup.s (SYSCTL_LAST_OF_MULTI_TURN, :7) before TryMove's chart
    (bcp.c:3214-3216), so MoveIsOnDamagingTurn holds (battle_lib.c:7588-7604): a neutral Dig is
    guarded, deals nothing and tags the holder. A Power Herb's skip strikes the same turn
    (effect_script_0256.s:12-26). The AI's own estimate runs after BattleContext_Init, where
    the flag is clear: it sees the hit."""
    n = names
    s = _guard_duel(tables, names, ["Dig", "Growl"])
    acts = A("Growl", "Dig", s, n)
    one = assert_matches_reference(tables, s, acts, O)
    assert int(one.locked_move[1]) == n.move("Dig")
    assert int(one.semi_invulnerable[1]) == SemiKind.UNDERGROUND
    assert int(one.ai_ability_known[0]) == 0 and int(one.hp[0, 0]) == int(s.hp[0, 0])
    two = assert_matches_reference(tables, one, acts, O)
    assert int(two.locked_move[1]) == 0 and int(two.semi_invulnerable[1]) == 0
    assert int(two.hp[0, 0]) == int(s.hp[0, 0]) and int(two.ai_ability_known[0]) != 0
    dig = n.move("Dig")
    assert int(damage_table(tables, s, 1, dig, False)[15]) == 0
    assert int(damage_table(tables, s, 1, dig, False, wonder_guard=False)[15]) > 0
    assert int(ai_damage_for_roll(tables, s, 1, dig, 15)) > 0
    herb = _item(s, 1, 0, "Power Herb", n)
    out = assert_matches_reference(tables, herb, acts, O)
    assert int(out.item[1, 0]) == 0 and int(out.locked_move[1]) == 0
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) != 0
    # Without Wonder Guard the same strike lands.
    plain = _ability(one, 0, AbilityId.NONE)
    out = assert_matches_reference(tables, plain, acts, O)
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_wonder_guard_holder_underground_is_missed_not_revealed(tables, names):
    """G-28: a move that misses a semi-invulnerable Wonder Guard holder still runs the chart
    (bcp.c:3263-3277), but subscript_missed.s:11 sends MOVE_STATUS_SEMI_INVULNERABLE to the
    miss message (:157) before the Wonder Guard one (:107): the AI learns nothing."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Dig", "Growl"], 50)],
        [_spec(n, "Rattata", ["Tackle", "Growl"], 10)],
    )
    s = _ability(s, 0, AbilityId.WONDER_GUARD)
    out = assert_matches_reference(tables, s, A("Dig", "Tackle", s, n), O)
    assert int(out.semi_invulnerable[0]) == SemiKind.UNDERGROUND
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 0


def test_pressure_charges_a_charge_move_once(tables, names):
    """G-29: the charge turn's BattleControllerPlayer_DecrementPP takes 1 plus the Pressure
    foe's 1 (bcp.c:2235-2280); the locked second turn cannot pick a command
    (Battler_CanPickCommand, battle_lib.c:6224-6235: VOLATILE_CONDITION_MOVE_LOCKED), so
    ppDecremented is set (bcp.c:334-335) and it spends nothing, Pressure or not."""
    s = _pressure(roark_state(tables, names, player_moves=["Dig"]), 1)
    before = _pp(s, 0, "Dig", names)
    acts = A("Dig", "Leer", s, names)
    one = assert_matches_reference(tables, s, acts, O)
    assert before - _pp(one, 0, "Dig", names) == 2
    assert int(one.locked_move[0]) == names.move("Dig")
    two = assert_matches_reference(tables, one, acts, O)
    assert _pp(two, 0, "Dig", names) == _pp(one, 0, "Dig", names)
    assert int(two.locked_move[0]) == 0


# --- review fixes (G-30 ..) -------------------------------------------------------------------


def _present(o, side, kind):
    return o._replace(present=o.present.at[side].set(kind))


def test_present_heal_skips_wonder_guard(tables, names):
    """G-30: Present's heal path sets SYSCTL_IGNORE_IMMUNITIES (effect_script_0122.s:13) before
    TryMove's chart, which then does not run (bcp.c:2350-2356): no MOVE_STATUS_WONDER_GUARD, no
    subscript_missed.s, no reveal; the heal lands and AfterMoveEffects run. Its damage path is a
    Normal hit, never super effective: guarded and revealed, as is a miss on that path (G-20); a
    missed heal shows only the miss."""
    from nuzline.engine.outcome import PRESENT_HEAL

    s = _guard_duel(tables, names, ["Present", "Growl"])
    s = s.replace_hp(0, 0, int(s.max_hp[0, 0]) // 2)
    acts = A("Growl", "Present", s, names)
    out = assert_matches_reference(tables, s, acts, _present(O, 1, PRESENT_HEAL))
    assert int(out.hp[0, 0]) > int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, acts, _present(O, 1, 2))
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) != 0
    missed = O._replace(hit=O.hit.at[1, 0].set(0))
    out = assert_matches_reference(tables, s, acts, _present(missed, 1, PRESENT_HEAL))
    assert int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, acts, _present(missed, 1, 0))
    assert int(out.ai_ability_known[0]) != 0


def test_present_heal_runs_the_after_move_items(tables, names):
    """G-30: the heal path's AfterMoveEffects run on a Wonder Guard holder (no
    subscript_missed.s): the healer's Sitrus Berry check happens."""
    from nuzline.engine.outcome import PRESENT_HEAL

    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Tackle", "Growl"], 30)],
        [_spec(n, "Rattata", ["Present", "Growl"], 30, "Sitrus Berry")],
    )
    s = _ability(s, 0, AbilityId.WONDER_GUARD)
    s = s.replace_hp(1, 0, int(s.max_hp[1, 0]) // 2 - 1)
    acts = A("Growl", "Present", s, n)
    out = assert_matches_reference(tables, s, acts, _present(O, 1, PRESENT_HEAL))
    assert int(out.item[1, 0]) == 0 and int(out.hp[1, 0]) > int(s.hp[1, 0])


def test_present_vs_wonder_guard_enumerates_its_draw_on_a_miss(tables, names):
    """G-30: against a player's Wonder Guard the Present draw decides the reveal on a miss too
    (the script runs before the accuracy check, bcp.c:3214), so the exact enumeration reads it
    there (``Reads.present_guard``) and equals the brute force."""
    from tests.engine.test_enumerate import assert_equals_bruteforce

    s = _guard_duel(tables, names, ["Present", "Growl"])
    assert_equals_bruteforce(tables, s, A("Growl", "Present", s, names))


def test_synchronize_pass_to_a_fainted_statused_receiver_fails(tables, names):
    """G-31: the receiver's status test reads BattleMon.status (subscript_paralyze.s:41-42), which
    a faint keeps (battle_controller.c:1481-1486): the AI's burned Static U-turn user that the
    player's Synchronize Pursuit KOs (Static paralyzes the pursuer, subscript_pursuit.s:86) is the
    receiver of the MODE_PURSUIT tail's pass, still burned: the pass fails silently (no reveal;
    Umbreon, slower, Synchronize).
    Unburned, the pass lands on the fainted battler: the message tags Synchronize."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Umbreon", ["Pursuit"], 35)],
        [_spec(n, "Rattata", ["U-turn"], 50), _spec(n, "Rattata", ["Growl"], 30)],
    )
    assert int(s.ability[0, 0]) == AbilityId.SYNCHRONIZE
    s = _ability(s, 1, AbilityId.STATIC)._replace(hp=s.hp.at[1, 0].set(1))
    burned = _status(s, 1, 0, Status.BURN)
    acts = A("Pursuit", "U-turn", s, n)
    o = _proc(O, "contact_proc", 1)
    out = assert_matches_reference(tables, burned, acts, o)
    assert int(out.hp[1, 0]) == 0 and int(out.status[0, 0]) == Status.PARALYSIS
    assert int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.hp[1, 0]) == 0 and int(out.status[0, 0]) == Status.PARALYSIS
    assert int(out.ai_ability_known[0]) != 0


def test_synchronize_flag_toggles_twice_in_one_hit(tables, names):
    """G-12 (test gap): one hit whose secondary paralyzes the defender (subscript_paralyze.s:50-55:
    the flag toggles on) and whose contact meets the defender's Static (toggles it off again,
    TriggerAbilityOnHit, sideEffectMon the attacker): the step finds the flag clear. The
    Synchronize attacker's pass could only reach the paralyzed defender anyway, so both engines
    agree and nothing more lands."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Abra", ["ThunderPunch", "Growl"], 30)],
        [_spec(n, "Rattata", ["Growl"], 30)],
    )
    s = _ability(s, 1, AbilityId.STATIC)
    o = _proc(_proc(O, "secondary", 0), "contact_proc", 1)
    out = assert_matches_reference(tables, s, A("ThunderPunch", "Growl", s, n), o)
    assert int(out.status[1, 0]) == Status.PARALYSIS
    assert int(out.status[0, 0]) == Status.PARALYSIS


def test_synchronize_runs_after_every_hit_of_a_multi_hit_move(tables, names):
    """G-12 (test gap): AfterMoveEffects' Synchronize step runs after each hit (bcp.c:3665-3670,
    inside the multi-hit loop): Double Kick's second hit meets the foe's Static
    (TriggerAbilityOnHit, sideEffectMon the Synchronize attacker), and the step after that hit
    passes the paralysis back. (Twineedle, the one multi-hit move with a status secondary, is
    tests/engine/test_batch13.py's, B13-1.)"""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Abra", ["Double Kick", "Growl"], 30)],
        [_spec(n, "Rattata", ["Growl"], 30)],
    )
    s = _ability(s, 1, AbilityId.STATIC)
    for hit in (0, 1):
        o = _proc(O, "contact_proc", 1, hit)
        out = assert_matches_reference(tables, s, A("Double Kick", "Growl", s, n), o)
        assert int(out.status[0, 0]) == Status.PARALYSIS, hit
        assert int(out.status[1, 0]) == Status.PARALYSIS, hit
        assert int(out.ai_ability_known[0]) != 0


def test_pressure_with_a_fainted_foe(tables, names):
    """G-8 (test gap): the foe's Selfdestruct faints it before the player moves. Tackle then
    has no defender (FightCommand: BATTLER_NONE for a fainted single target), so
    DecrementPP's Pressure test (bcp.c:2238) is skipped: 1 PP. Spikes (RANGE_OPPONENT_SIDE)
    keeps its defender, but COUNT_ALIVE_BATTLERS_THEIR_SIDE counts no fainted holder: 1 PP."""
    n = names
    for move in ("Tackle", "Spikes"):
        s = make_state(
            tables,
            [_spec(n, "Snorlax", [move], 50)],
            [_spec(n, "Electrode", ["Selfdestruct"], 20), _spec(n, "Geodude", ["Tackle"], 20)],
        )
        s = _pressure(s, 1)
        before = _pp(s, 0, move, names)
        out = assert_matches_reference(tables, s, A(move, "Selfdestruct", s, names), O)
        assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) > 0
        assert before - _pp(out, 0, move, names) == 1, move


def _hazard_double_ko(tables, names, levels, opp="Kyogre"):
    """A double KO (both Geodude leads down) whose player newcomer, Charmander at 1 HP, falls to
    Stealth Rock: the player's next newcomer is Groudon (Drought, level ``levels[0]``), the
    opponent's only bench Pokemon ``opp`` (level ``levels[1]``)."""
    from tests.search.fixtures import at_replacement

    n = names
    s = make_state(
        tables,
        [
            _spec(n, "Geodude", ["Tackle"], 30),
            _spec(n, "Charmander", ["Tackle"], 30),
            _spec(n, "Groudon", ["Tackle"], levels[0]),
        ],
        [_spec(n, "Geodude", ["Tackle"], 30), _spec(n, opp, ["Tackle"], levels[1])],
    )
    s = s._replace(stealth_rock=s.stealth_rock.at[0].set(1), hp=s.hp.at[0, 1].set(1))
    return at_replacement(s, opp_down=True)


def test_double_ko_newcomer_felled_by_hazards_leaves_the_foes_states_waiting(tables, names):
    """G-32: the list sends both newcomers in with the hazards only (subscript_show_party_list.s:
    22-24); the player's falls to Stealth Rock, so TurnEnd re-enters ReplaceFainted for the
    player (battle_controller_player.c:1846-1858) before any ShowBattleMon: the opponent's
    Kyogre has not set its rain. With the player's next newcomer both run ShowBattleMon's states
    together (:251-263): the slower one's weather stays (G-26), not the later entry's."""
    from nuzline.engine.step import needs_replacement
    from nuzline.search.playout import replacement_pass

    for levels, weather in (((50, 40), Weather.RAIN), ((40, 50), Weather.SUN)):
        s = _hazard_double_ko(tables, names, levels)
        first, refused = replacement_pass(tables, s, 1)
        assert not bool(refused)
        assert int(first.hp[0, 1]) == 0 and int(first.active[1]) == 1
        assert int(weather_of(first)) == Weather.NONE
        assert np.asarray(first.entry_pending).tolist() == [0, 1]
        assert int(needs_replacement(first)[0]) == 1
        second, refused = replacement_pass(tables, first, 2)
        assert not bool(refused) and int(second.active[0]) == 2
        assert int(weather_of(second)) == weather and int(weather_timer_of(second)) == 0
        assert np.asarray(second.entry_pending).tolist() == [0, 0]
    # A tie (Groudon and Kyogre, both Speed 102 at level 50): the sort at TrainerMessage
    # (battle_controller_player.c:251) swaps on RandNext() & 1 (battle_lib.c:1445), so the war
    # stays open in war_swaps for the next draw, as at an ordinary double KO.
    from nuzline.engine.step import resolve_war

    first, _ = replacement_pass(tables, _hazard_double_ko(tables, names, (50, 50)), 1)
    second, refused = replacement_pass(tables, first, 2)
    assert not bool(refused) and int(second.war_swaps) == 0b10101010
    assert int(weather_of(resolve_war(second, 1))) == Weather.SUN
    assert int(weather_of(resolve_war(second, 0))) == Weather.RAIN


def test_double_ko_waiting_intimidate_meets_the_next_newcomer(tables, names):
    """G-32: the waiting opposing newcomer's Intimidate fires at ShowBattleMon against the
    player's next newcomer (battle_lib.c:3795-3808); before, the engine had fired it at its
    entry against the fainted Charmander (nothing). Its held-item check waits too."""
    from nuzline.search.playout import replacement_pass

    s = _hazard_double_ko(tables, names, (50, 40), opp="Rattata")
    s = _ability(s, 1, AbilityId.INTIMIDATE, slot=1)
    first, _ = replacement_pass(tables, s, 1)
    assert int(first.entry_pending[1]) == 1 and int(first.stages[0, Stat.ATK]) == 0
    second, _ = replacement_pass(tables, first, 2)
    assert int(second.stages[0, Stat.ATK]) == -1
    assert int(second.entry_pending[1]) == 0


def test_replace_without_effects_matches_the_reference(tables, names):
    """G-32: ``replace(..., effects=False)`` runs the list alone (the hazards, the faint check)
    and marks a living newcomer's switch-in states as waiting, in both engines."""
    s = _hazard_double_ko(tables, names, (50, 40))
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1))
    out = replace(tables, s, 1, 1, False)
    ref = from_ref(ref_replace(to_numpy(tables), to_ref(s), 1, 1, False))
    for name in out._fields:
        mine, theirs = np.asarray(getattr(out, name)), np.asarray(getattr(ref, name))
        np.testing.assert_array_equal(mine, theirs, name)
    assert int(out.entry_pending[1]) == 1 and int(weather_of(out)) == Weather.NONE
    assert int(out.hp[1, 1]) < int(s.hp[1, 1])  # Stealth Rock hit it


def test_weather_war_tie_is_a_root_chance_node_the_player_sees(tables, names, monkeypatch):
    """G-34: the battle-start weather war's tie (battle_lib.c:1445, sorted at TrainerMessage,
    battle_controller_player.c:251) is drawn before command selection, and the weather shows:
    Stage B's root is a chance node over what the player sees, each outcome its own belief node.
    A real Stage B solve with a leaf that pays Tackle in sun and Growl in rain: each outcome's
    best is worth 1, so the root's value is the weighted best, 1, where one belief node over
    both weathers gives 1/2. A Quick Claw read stays hidden within an outcome; here it decides
    the weather (1/5 rain). ``expand`` runs at 1024-row calls instead of ``CHUNK`` (16384): a
    row's result depends on neither the call size nor the device count (expand's module
    docstring), and the default call size took this solve to a 16.5 GB sampled peak."""
    import json

    import jax

    from nuzline.engine.step import root_belief
    from nuzline.search import expand as expand_mod
    from nuzline.search import stage_b
    from nuzline.search.leaves import LeafValue
    from nuzline.search.plan_tree import render, to_json
    from nuzline.search.stage_b import ROOT_CHANCE, SolveSettings, result_from_dict, result_to_dict

    monkeypatch.setattr(expand_mod, "CHUNK", 1024)
    n = names
    s = make_state(
        tables,
        [_spec(n, "Groudon", ["Tackle", "Growl"], 50)],
        [_spec(n, "Kyogre", ["Tackle"], 50)],
    )
    assert int(s.war_swaps) != 0
    tackle, growl = n.move_slot(s, 0, "Tackle"), n.move_slot(s, 0, "Growl")
    full = int(s.pp[0, 0, tackle])

    def leaf(states, weights, pairs=None):
        sun = weather_of(states) == Weather.SUN
        tackled = np.asarray(states.pp)[:, 0, 0, tackle] < full
        return [LeafValue(10 * int(w), 10, 10 * int(w), 10, False) for w in sun == tackled]

    settings = SolveSettings(max_depth=1, budget_playouts=20, n_depth0=10, epsilon=0.0)
    key = jax.random.PRNGKey(0)
    result = stage_b.solve(tables, n, s, key, settings, leaf_fn=leaf)
    got = sorted((o.label, o.prob) for o in result.outcomes)
    assert got == [("weather rain", 0.5), ("weather sun", 0.5)]
    best = {o.label: o.result.ranking[0] for o in result.outcomes}
    assert {k: r.action for k, r in best.items()} == {"weather sun": tackle, "weather rain": growl}
    assert all(r.estimate == pytest.approx(1.0) for r in best.values())
    [row] = result.ranking
    assert row.action == ROOT_CHANCE and row.estimate == pytest.approx(1.0)
    assert result.plan.kind == "chance" and len(result.plan.children[ROOT_CHANCE]) == 2
    plans = {o.label: p for o, p in zip(result.outcomes, result.plan.children[ROOT_CHANCE])}
    assert {k: p.action for k, p in plans.items()} == {"weather sun": tackle, "weather rain": growl}
    assert all(p.obs == o.obs for o, p in zip(result.outcomes, result.plan.children[ROOT_CHANCE]))
    # One belief node over both weathers (the lower bound the chance root replaces): 1/2.
    one = stage_b._solve_belief(tables, n, s, root_belief(tables, s), key, settings, leaf, 0.0)
    assert one.outcomes == () and one.ranking[0].estimate == pytest.approx(0.5)
    assert result_from_dict(result_to_dict(result)) == result
    assert any("weather sun" in line for line in render(result, n, 1, root=s))
    json.dumps(to_json(result, n, root=s))
    # The slower player's Quick Claw (G-26): activated (1/5) it sorts first and the rain stays.
    claw = _war(tables, n, (40, 50), ("Quick Claw", ""))
    result = stage_b.solve(tables, n, claw, key, settings, leaf_fn=leaf)
    got = {o.label: o.prob for o in result.outcomes}
    assert got == pytest.approx({"weather rain": 0.2, "weather sun": 0.8})
    # No war: one belief node, the solve's own result.
    plain = _war(tables, n, (50, 40))
    assert stage_b.solve(tables, n, plain, key, settings, leaf_fn=leaf).outcomes == ()


@pytest.mark.parametrize(
    ("player_item", "foe_ability", "mine", "theirs"),
    [
        ("", AbilityId.STATIC, "Tackle", "Spark"),  # Synchronize: contact and secondary passes
        ("", AbilityId.POISON_POINT, "Growl", "Thunder Wave"),  # a status move's pass
        ("Focus Band", AbilityId.INNER_FOCUS, "Growl", "Tackle"),  # the holder's band roll per hit
    ],
)
def test_synchronize_and_focus_band_reads_equal_bruteforce(
    tables, names, player_item, foe_ability, mine, theirs
):
    """Review check (G-1, G-12): the exact enumeration's conditional reads of the Synchronize
    passes and the Focus Band rolls give the brute-force product's merged children."""
    from tests.engine.test_enumerate import assert_equals_bruteforce

    n = names
    s = make_state(
        tables,
        [_spec(n, "Abra", ["Growl", "Tackle"], 30, player_item)],
        [_spec(n, "Rattata", ["Spark", "Thunder Wave", "Tackle"], 30)],
    )
    s = _ability(s, 1, foe_ability)
    if player_item:
        s = s.replace_hp(0, 0, 3)
    assert_equals_bruteforce(tables, s, A(mine, theirs, s, n))
