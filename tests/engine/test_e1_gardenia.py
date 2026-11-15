"""E1 Gardenia batch (rulings G-n, docs/kaizo/e1-mechanics-ledger.md): hand cases, each checked
against the reference engine (``assert_matches_reference``).

Family A: held items that draw. Focus Band (``HOLD_EFFECT_MAYBE_ENDURE``, G-1 .. G-3) and the
flavor berries Figy, Wiki, Mago, Aguav, Iapapa (``HOLD_EFFECT_HP_RESTORE_SPICY`` .. ``_SOUR``,
G-4 .. G-6). Family B: the ability Pressure (G-8 .. G-11). Family C: the status guards
Synchronize and Leaf Guard (G-12 .. G-17).
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
from nuzline.engine.damage import damage_table
from nuzline.engine.ids import (
    CONFUSION_PENDING,
    FOCUS_BAND_PARAM,
    NATURE_DISLIKED_FLAVOR,
    AbilityId,
    Flavor,
    ItemId,
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
