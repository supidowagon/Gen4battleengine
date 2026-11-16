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

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import legal_actions
from nuzline.engine.damage import ai_damage_for_roll, damage_table
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


# --- Status guards: Synchronize and Leaf Guard (G-12 .. G-17) -----------------------------


def _spec(names, species: str, moves, level=30, item="", slot=0):
    """``_mon`` with an ability slot."""
    return dataclasses.replace(_mon(names, species, moves, level, item), ability_slot=slot)


def _ability(s, side, ability, slot=None):
    """``s`` with ``side``'s party ``slot`` (the active one by default) holding ``ability``."""
    mon = int(s.active[side]) if slot is None else slot
    return s._replace(ability=s.ability.at[side, mon].set(ability))


def _weather(s, weather):
    return s._replace(weather=jnp.asarray(weather, s.weather.dtype))


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
    assert int(out.ai_ability_known[0]) == 1
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
    assert int(out.ai_ability_known[0]) == 1


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
        assert int(out.ai_ability_known[1]) == 1


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
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 1
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
        assert int(out.ai_ability_known[0]) == 1
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
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 1


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
        assert int(out.hp[0, 0]) == 1 and int(out.ai_ability_known[0]) == 1
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
    """G-20: Pursuit's hit on a switching Wonder Guard holder (subscript_pursuit.s:11, :17,
    :154) ends in subscript_missed.s: no damage, and its :107-110 message tags the ability."""
    n = names
    s = make_state(
        tables,
        [_spec(n, "Squirtle", ["Tackle"], 30), _spec(n, "Abra", ["Tackle"], 30)],
        [_spec(n, "Rattata", ["Pursuit"], 30)],
    )
    s = _ability(s, 0, AbilityId.WONDER_GUARD)
    out = assert_matches_reference(tables, s, A(SWITCH_TO_1, "Pursuit", s, n), O)
    assert int(out.active[0]) == 1
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 1


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
    assert int(out.ai_ability_known[0]) == 1
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
    from nuzline.engine.enumerate import outcome_laws
    from nuzline.engine.outcome import pivot_view

    n = names
    s = _standoff(tables, names)
    acts = A("Tackle", SWITCH_TO_1, s, n)
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.ability[0, 0]) == AbilityId.SUPER_LUCK
    view, _ = pivot_view(tables, s, acts)
    assert int(view.ability[0, 0]) == AbilityId.SUPER_LUCK
    assert int(outcome_laws(tables, s, acts).crit_odds[0]) == 8
    assert int(outcome_laws(tables, s, A("Tackle", "Tackle", s, n)).crit_odds[0]) == 16
    s = _standoff(tables, names, ("Pursuit",))
    acts = A("Pursuit", SWITCH_TO_1, s, n)
    assert int(pivot_view(tables, s, acts)[0].ability[0, 0]) == AbilityId.TRACE
    assert int(outcome_laws(tables, s, acts).crit_odds[0]) == 16
    out = assert_matches_reference(tables, s, acts, O)
    assert int(out.ability[0, 0]) == AbilityId.SUPER_LUCK


def test_trace_standoff_copies_a_u_turn_newcomer(tables, names):
    """G-25: the foe's U-turn pick enters before the holder moves: at the paused list the laws'
    view gives the holder the copy (``pivot_view``), and the resume writes it."""
    from nuzline.engine.enumerate import outcome_laws

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
    assert int(outcome_laws(tables, paused, jnp.asarray(acts)).crit_odds[0]) == 8
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
    assert int(fast.weather) == Weather.RAIN and int(fast.war_swaps) == 0
    assert int(fast.ai_ability_known[0]) == 1  # its own script ran first
    slow = _war(tables, names, (40, 50))
    assert int(slow.weather) == Weather.SUN and int(slow.weather_timer) == 0
    assert int(slow.war_swaps) == 0 and int(slow.ai_ability_known[0]) == 1


def test_weather_war_tie_and_quick_claw_are_root_draws(tables, names):
    """G-26: an exact speed tie swaps on RandNext() & 1 (battle_lib.c:1445) and a lead's Quick
    Claw reads the battle-start speedRand (:1278-1286, BattleContext_InitCounters :2024): the
    order waits in ``war_swaps`` for the root chance node (``root_belief``, ``sample_root``)."""
    import jax

    from nuzline.engine.step import root_belief, sample_root

    tie = _war(tables, names, (50, 50))
    assert int(tie.war_swaps) == 0b10101010 and int(tie.weather) == Weather.RAIN
    belief = root_belief(tables, tie)
    assert sorted((int(b.weather), p) for b, p in belief) == sorted(
        [(int(Weather.SUN), 0.5), (int(Weather.RAIN), 0.5)]
    )
    assert all(int(b.war_swaps) == 0 for b, _ in belief)
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    drawn = {int(sample_root(k, tables, tie).weather) for k in keys}
    assert drawn == {Weather.SUN, Weather.RAIN}
    # The slower player's Quick Claw: activated (1/5) it sorts first and the rain stays.
    claw = _war(tables, names, (40, 50), ("Quick Claw", ""))
    assert int(claw.war_swaps) != 0
    belief = {(int(b.qc_roll[0]), int(b.weather)): p for b, p in root_belief(tables, claw)}
    assert belief == pytest.approx({(1, Weather.RAIN): 0.2, (0, Weather.SUN): 0.8})


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
        assert int(after.weather) == weather and int(after.war_swaps) == 0
    s = _double_ko(tables, names, (50, 50))
    after, _ = replacement_pass(tables, s, 1)
    assert int(after.war_swaps) != 0 and int(after.weather) == Weather.RAIN
    assert int(resolve_war(after, 1).weather) == Weather.SUN
    assert int(resolve_war(after, 0).weather) == Weather.RAIN
    keys = jax.random.split(jax.random.PRNGKey(1), 32)
    assert {int(settle_war(after, k).weather) for k in keys} == {Weather.SUN, Weather.RAIN}
    batch = jax.tree.map(lambda f: np.asarray(f)[None], s)
    store, children = expand_replacements(tables, batch, [(0, MOVE_ACTIONS + 1)])
    kids = children[0]
    assert [k.prob for k in kids] == [0.5, 0.5]
    weathers = {int(take(store, np.asarray([k.state_row])).weather[0]) for k in kids}
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
    assert int(two.hp[0, 0]) == int(s.hp[0, 0]) and int(two.ai_ability_known[0]) == 1
    dig = n.move("Dig")
    assert int(damage_table(tables, s, 1, dig, False)[15]) == 0
    assert int(damage_table(tables, s, 1, dig, False, wonder_guard=False)[15]) > 0
    assert int(ai_damage_for_roll(tables, s, 1, dig, 15)) > 0
    herb = _item(s, 1, 0, "Power Herb", n)
    out = assert_matches_reference(tables, herb, acts, O)
    assert int(out.item[1, 0]) == 0 and int(out.locked_move[1]) == 0
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 1
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
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.ai_ability_known[0]) == 1
    missed = O._replace(hit=O.hit.at[1, 0].set(0))
    out = assert_matches_reference(tables, s, acts, _present(missed, 1, PRESENT_HEAL))
    assert int(out.ai_ability_known[0]) == 0
    out = assert_matches_reference(tables, s, acts, _present(missed, 1, 0))
    assert int(out.ai_ability_known[0]) == 1


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
    assert int(out.ai_ability_known[0]) == 1


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
    not a supported effect in these tables.)"""
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
        assert int(out.ai_ability_known[0]) == 1


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
        assert int(first.weather) == Weather.NONE
        assert np.asarray(first.entry_pending).tolist() == [0, 1]
        assert int(needs_replacement(first)[0]) == 1
        second, refused = replacement_pass(tables, first, 2)
        assert not bool(refused) and int(second.active[0]) == 2
        assert int(second.weather) == weather and int(second.weather_timer) == 0
        assert np.asarray(second.entry_pending).tolist() == [0, 0]


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
    assert int(out.entry_pending[1]) == 1 and int(out.weather) == Weather.NONE
    assert int(out.hp[1, 1]) < int(s.hp[1, 1])  # Stealth Rock hit it


def test_weather_war_tie_is_a_root_chance_node_the_player_sees(tables, names, monkeypatch):
    """G-34: the battle-start weather war's tie (battle_lib.c:1445, sorted at TrainerMessage,
    battle_controller_player.c:251) is drawn before command selection, and the weather shows:
    Stage B's root is a chance node over what the player sees, each outcome its own belief node
    (``_solve_belief``, stubbed here: Tackle wins in sun, Growl in rain). The root's value is
    the weighted best, 1 (one belief node over both gave 1/2). A Quick Claw read stays hidden
    within an outcome; here it decides the weather (1/5 rain)."""
    import json

    import jax

    from nuzline.search import stage_b
    from nuzline.search.plan_tree import render, to_json
    from nuzline.search.stage_b import (
        ROOT_CHANCE,
        ActionValue,
        PlanNode,
        SolveResult,
        SolveSettings,
        result_from_dict,
        result_to_dict,
    )

    n = names
    s = make_state(
        tables,
        [_spec(n, "Groudon", ["Tackle", "Growl"], 50)],
        [_spec(n, "Kyogre", ["Tackle"], 50)],
    )
    tackle, growl = n.move_slot(s, 0, "Tackle"), n.move_slot(s, 0, "Growl")
    seen = []

    def one(action, value):
        return ActionValue(action, f"a{action}", value, value, value, value, 0.0, 0.0, 0.0, False)

    def fake(tables_, names_, state, belief, key, settings, leaf_fn, start):
        weathers = {int(b.weather) for b, _ in belief}
        assert len(weathers) == 1 and abs(sum(w for _, w in belief) - 1) < 1e-12
        weather = weathers.pop()
        seen.append(weather)
        best = tackle if weather == Weather.SUN else growl
        rows = [one(best, 1.0), one(tackle + growl - best, 0.0)]
        plan = PlanNode(None, 1.0, best, {best: 1.0}, {}, "decision")
        return SolveResult(rows, {}, 1, plan, 10, 0.0, settings, 10)

    monkeypatch.setattr(stage_b, "_solve_belief", fake)
    settings = SolveSettings(max_depth=1)
    result = stage_b.solve(tables, n, s, jax.random.PRNGKey(0), settings)
    assert sorted(seen) == sorted([int(Weather.SUN), int(Weather.RAIN)])
    got = sorted((o.label, o.prob) for o in result.outcomes)
    assert got == [("weather rain", 0.5), ("weather sun", 0.5)]
    [row] = result.ranking
    assert row.action == ROOT_CHANCE and row.estimate == pytest.approx(1.0)
    assert result.plan.kind == "chance" and len(result.plan.children[ROOT_CHANCE]) == 2
    assert result_from_dict(result_to_dict(result)) == result
    assert any("weather sun" in line for line in render(result, n, 1, root=s))
    json.dumps(to_json(result, n, root=s))
    # The slower player's Quick Claw (G-26): activated (1/5) it sorts first and the rain stays.
    claw = _war(tables, n, (40, 50), ("Quick Claw", ""))
    result = stage_b.solve(tables, n, claw, jax.random.PRNGKey(0), settings)
    got = {o.label: o.prob for o in result.outcomes}
    assert got == pytest.approx({"weather rain": 0.2, "weather sun": 0.8})
    # No war: one belief node, the solve's own result.
    plain = _war(tables, n, (50, 40))
    assert stage_b.solve(tables, n, plain, jax.random.PRNGKey(0), settings).outcomes == ()


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
