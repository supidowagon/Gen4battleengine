"""E1 Gardenia batch (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): hand cases, each checked
against the reference engine (``assert_matches_reference``).

Family A: held items that draw. Focus Band (``HOLD_EFFECT_MAYBE_ENDURE``, G-1 .. G-3) and the
flavor berries Figy, Wiki, Mago, Aguav, Iapapa (``HOLD_EFFECT_HP_RESTORE_SPICY`` .. ``_SOUR``,
G-4 .. G-6). Family B: the ability Pressure (G-8 .. G-11).
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import legal_actions
from nuzline.engine.damage import damage_table
from nuzline.engine.ids import (
    CONFUSION_PENDING,
    FOCUS_BAND_PARAM,
    NATURE_DISLIKED_FLAVOR,
    AbilityId,
    Flavor,
    ItemId,
)
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import replace
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace
from tests.engine.helpers import A, assert_matches_reference, roark_state
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
    band = np.zeros((2, 5), np.int8)
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
    assert int(lead.ai_ability_known[0]) == 1
    s = make_state(
        tables,
        [_mon(n, "Geodude", ["Tackle"]), _mon(n, "Absol", ["Tackle"])],
        [_mon(n, "Geodude", ["Tackle"])],
    )
    assert int(s.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, (SWITCH_TO_1, 0), O)
    assert int(out.active[0]) == 1 and int(out.ai_ability_known[1]) == 1
