"""Focus Sash, Big Root, Berry Juice, Chesto / Pecha / Lum Berry, White Herb and Lagging Tail
(K3b1 task 11).

* Focus Sash (``HOLD_EFFECT_ENDURE``): ``BattleControllerPlayer_UpdateHP`` (bcp.c:3386-3404)
  sets ``focusItemActivated`` for a damaging hit taken at full HP; while it is set a hit that
  would KO leaves 1 HP. The flag lives in ``selfTurnFlags`` until ``BattleContext_Init`` at
  the action's end (bcp.c:3998, battle_lib.c:2014), so a multi-hit move that met a full-HP
  holder is capped on every later hit too. ``BtlCmd_CheckHoldOnWith1HP`` (script.c:
  9098-9124) is the same full-HP check for the confusion self-hit and Pursuit.
  ``subscript_move_followup_message.s:12-19`` removes the sash only when it saved the holder.
* Chesto / Pecha / Lum / White Herb: ``BattleSystem_TriggerHeldItem`` (battle_lib.c:4576) and
  ``..._OnStatus`` (:4967), the call sites of Oran / Sitrus. Pecha cures toxic
  (``MON_CONDITION_ANY_POISON``), Lum every status and confusion, White Herb every lowered
  stage. Pluck / Bug Bite eat them by their pluck effects (``BattleSystem_PluckBerry``,
  :5539) and leave Berry Juice (``PLUCK_EFFECT_NONE``, no berry, :5833).
* Big Root: ``subscript_drain_half_damage_dealt.s`` / ``subscript_leech_seed_effect.s``,
  ``heal * (100 + 30) / 100`` after the base amount, no minimum.
* Lagging Tail: ``BattleSystem_CompareBattlerSpeed`` (battle_lib.c:1412-1421), also for the
  end-of-turn order (``SortMonSpeedOrder``).
"""

import jax
import jax.numpy as jnp

from nuzline.engine.damage import damage_table
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.ids import AbilityId, ItemId, Stat, Status
from nuzline.engine.order import action_order
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import (
    WINNER_OPPONENT,
    WINNER_PLAYER,
    end_of_turn_order,
    step,
)
from nuzline.prep.variants import hp_cuts
from tests.engine.helpers import A, assert_matches_reference, roark_state, sample_party

SWITCH_TO_1 = 4 + 1  # order.MOVE_ACTIONS + party slot
_jit_step = jax.jit(step)


def _item(s, side, slot, item, names):
    return s._replace(item=s.item.at[side, slot].set(names.item(item)))


def _full(s, side, slot, hp):
    """Max HP and HP both ``hp``."""
    return s._replace(hp=s.hp.at[side, slot].set(hp), max_hp=s.max_hp.at[side, slot].set(hp))


def _sash(s, side, names, hp):
    return _full(_item(s, side, 0, "Focus Sash", names), side, 0, hp)


def _consumed(out, side, item, names, slot=0):
    return int(out.item[side, slot]) == 0 and int(out.item_consumed[side, slot]) == names.item(item)


def test_tables_map_the_items_in_both_games(tables, names, kaizo_tables, kaizo_names):
    want = {
        "Focus Sash": (ItemId.FOCUS_SASH, 0),
        "Big Root": (ItemId.BIG_ROOT, 30),
        "Berry Juice": (ItemId.BERRY_JUICE, 20),
        "Chesto Berry": (ItemId.CHESTO_BERRY, 0),
        "Pecha Berry": (ItemId.PECHA_BERRY, 0),
        "Lum Berry": (ItemId.LUM_BERRY, 0),
        "White Herb": (ItemId.WHITE_HERB, 0),
        "Lagging Tail": (ItemId.LAGGING_TAIL, 0),
    }
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        for item, (effect, param) in want.items():
            assert int(t.item_effect[n.item(item)]) == effect, item
            assert int(t.item_param[n.item(item), 0]) == param, item


# --- Focus Sash ---------------------------------------------------------------------


def test_focus_sash_leaves_one_hp_and_is_consumed(tables, names):
    s = _sash(roark_state(tables, names), 1, names, 5)  # Mach Punch deals more than 5
    out = _jit_step(tables, s, A("Mach Punch", "Leer", s, names), empty_outcome())
    assert int(out.hp[1, 0]) == 1 and _consumed(out, 1, "Focus Sash", names)
    # Below full HP it does nothing and stays.
    low = s.replace_hp(1, 0, 4)
    out = _jit_step(tables, low, A("Mach Punch", "Leer", low, names), empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.item_consumed[1, 0]) == 0
    # A hit that does not KO activates it but leaves it held (no MOVE_STATUS_ENDURED_ITEM).
    big = _sash(roark_state(tables, names), 1, names, 200)
    out = _jit_step(tables, big, A("Mach Punch", "Leer", big, names), empty_outcome())
    assert 1 < int(out.hp[1, 0]) < 200 and int(out.item[1, 0]) == names.item("Focus Sash")


def test_focus_sash_crit(tables, names):
    s = _sash(roark_state(tables, names), 1, names, 5)
    crit = empty_outcome()._replace(crit=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, s, A("Mach Punch", "Leer", s, names), crit)
    assert int(out.hp[1, 0]) == 1 and _consumed(out, 1, "Focus Sash", names)


def test_focus_sash_multi_hit_flag_lasts_the_move(tables, names):
    # Only a hit taken at full HP activates the sash; the activation (focusItemActivated)
    # lasts for the rest of the move, so later hits are capped as well.
    s = roark_state(tables, names, player_moves=["Double Kick"])
    kick = names.move("Double Kick")
    d = int(damage_table(tables, s, 0, kick, False)[15])  # empty_outcome: roll 15, no crit
    acts = A("Double Kick", "Leer", s, names)
    # Full HP, the first hit does not KO, the second would: capped at 1, sash consumed.
    two = _sash(s, 1, names, d + 2)
    out = assert_matches_reference(tables, two, acts, empty_outcome())
    assert int(out.hp[1, 0]) == 1 and _consumed(out, 1, "Focus Sash", names)
    # Full HP, the first hit KOs: 1 HP, then the second deals 0 (the flag is still set).
    one = _sash(s, 1, names, d)
    out = assert_matches_reference(tables, one, acts, empty_outcome())
    assert int(out.hp[1, 0]) == 1 and _consumed(out, 1, "Focus Sash", names)
    # DoubleSlap (2-5 law, 6 damage at roll 15) for 5 hits into a 3-HP holder: it ends at
    # 1 HP; the sash is removed once, at the first hit's followup message.
    slap = roark_state(tables, names, player_moves=["DoubleSlap"], opp_species="Shinx")
    slap = _sash(slap, 1, names, 3)
    five = empty_outcome()._replace(n_hits=jnp.array([5, 2], jnp.int8))
    out = assert_matches_reference(tables, slap, A("DoubleSlap", "Leer", slap, names), five)
    assert int(out.hp[1, 0]) == 1 and _consumed(out, 1, "Focus Sash", names)
    # Not at full HP at the first hit: never activated, the holder faints.
    below = _sash(s, 1, names, d + 2).replace_hp(1, 0, d + 1)
    out = assert_matches_reference(tables, below, acts, empty_outcome())
    assert int(out.hp[1, 0]) == 0 and int(out.item_consumed[1, 0]) == 0


def test_focus_sash_confusion_self_hit(tables, names):
    s = _sash(roark_state(tables, names), 0, names, 2)
    s = s._replace(confusion=s.confusion.at[0].set(3))
    hurt = empty_outcome()._replace(confusion_hit=jnp.array([1, 0], jnp.int8))
    acts = A("Scratch", "Leer", s, names)
    plain = _item(s, 0, 0, "", names)
    assert int(assert_matches_reference(tables, plain, acts, hurt).hp[0, 0]) == 0
    out = assert_matches_reference(tables, s, acts, hurt)
    assert int(out.hp[0, 0]) == 1 and _consumed(out, 0, "Focus Sash", names)


def test_focus_sash_pursuit(tables, names):
    s = roark_state(tables, names, opp_moves=["Pursuit"])
    s = _sash(s, 0, names, 3)  # the player's lead switches out into Pursuit
    acts = A(SWITCH_TO_1, "Pursuit", s, names)
    plain = _item(s, 0, 0, "", names)
    assert int(assert_matches_reference(tables, plain, acts, empty_outcome()).hp[0, 0]) == 0
    out = assert_matches_reference(tables, s, acts, empty_outcome())
    assert int(out.hp[0, 0]) == 1 and _consumed(out, 0, "Focus Sash", names)
    assert int(out.active[0]) == 1


# --- status berries and White Herb ----------------------------------------------------


def test_chesto_wakes_a_sleeper(tables, names):
    s = _item(roark_state(tables, names, opp_moves=["Hypnosis"]), 0, 0, "Chesto Berry", names)
    out = assert_matches_reference(tables, s, A("Leer", "Hypnosis", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and _consumed(out, 0, "Chesto Berry", names)
    # Asleep at the turn start: the sleep check stops the move, the end-of-turn
    # USE_ITEM check (bcp.c:1362) wakes it.
    asleep = s.with_status(0, 0, Status.SLEEP, 3)
    acts = A("Scratch", "Stealth Rock", asleep, names)
    out = assert_matches_reference(tables, asleep, acts, empty_outcome())
    assert int(out.hp[1, 0]) == int(asleep.hp[1, 0])  # did not move
    assert int(out.status[0, 0]) == Status.NONE and _consumed(out, 0, "Chesto Berry", names)


def test_pecha_cures_toxic(tables, names):
    s = _item(roark_state(tables, names, opp_moves=["Toxic"]), 0, 0, "Pecha Berry", names)
    out = assert_matches_reference(tables, s, A("Leer", "Toxic", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.status_counter[0, 0]) == 0
    assert _consumed(out, 0, "Pecha Berry", names) and int(out.hp[0, 0]) == int(s.hp[0, 0])
    # A poisoned holder is cured at USE_ITEM, before the poison check: no damage.
    poisoned = s.with_status(0, 0, Status.POISON)
    acts = A("Leer", "Stealth Rock", s, names)
    out = assert_matches_reference(tables, poisoned, acts, empty_outcome())
    assert int(out.hp[0, 0]) == int(s.hp[0, 0]) and int(out.status[0, 0]) == Status.NONE
    # Pecha does not cure sleep.
    asleep = s.with_status(0, 0, Status.SLEEP, 3)
    out = assert_matches_reference(tables, asleep, acts, empty_outcome())
    assert int(out.status[0, 0]) == Status.SLEEP and int(out.item[0, 0]) == names.item(
        "Pecha Berry"
    )


def test_lum_cures_confusion(tables, names):
    s = _item(roark_state(tables, names, opp_moves=["Confuse Ray"]), 0, 0, "Lum Berry", names)
    out = assert_matches_reference(tables, s, A("Leer", "Confuse Ray", s, names), empty_outcome())
    assert int(out.confusion[0]) == 0 and _consumed(out, 0, "Lum Berry", names)
    # Status and confusion at once (subscript_held_item_multi_restore.s): both cured.
    both = s.with_status(0, 0, Status.BURN)._replace(confusion=s.confusion.at[0].set(3))
    out = assert_matches_reference(tables, both, A("Leer", "Tackle", both, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.confusion[0]) == 0
    assert _consumed(out, 0, "Lum Berry", names)
    # Any status: paralysis from Thunder Wave.
    tw = _item(roark_state(tables, names, opp_moves=["Thunder Wave"]), 0, 0, "Lum Berry", names)
    out = assert_matches_reference(
        tables, tw, A("Leer", "Thunder Wave", tw, names), empty_outcome()
    )
    assert int(out.status[0, 0]) == Status.NONE and _consumed(out, 0, "Lum Berry", names)


def test_white_herb_restores_lowered_stages(tables, names):
    s = _item(roark_state(tables, names), 0, 0, "White Herb", names)
    s = s.with_stages(0, atk=2)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    # Leer's Defense -1 is undone; the raised Attack stays.
    assert int(out.stages[0, Stat.DEF]) == 0 and int(out.stages[0, Stat.ATK]) == 2
    assert _consumed(out, 0, "White Herb", names)
    # Nothing lowered: the herb stays.
    out = assert_matches_reference(tables, s, A("Leer", "Tackle", s, names), empty_outcome())
    assert int(out.item[0, 0]) == names.item("White Herb")


def test_white_herb_after_intimidate_on_switch_in(tables, names):
    s = roark_state(tables, names)
    s = s._replace(ability=s.ability.at[0, 4].set(AbilityId.INTIMIDATE))
    s = _item(s, 1, 0, "White Herb", names)
    out = assert_matches_reference(tables, s, A(4 + 4, "Leer", s, names), empty_outcome())
    assert int(out.stages[1, Stat.ATK]) == 0 and _consumed(out, 1, "White Herb", names)


# --- Pluck / Bug Bite ---------------------------------------------------------------


def test_bug_bite_leaves_berry_juice_and_eats_status_berries(tables, names):
    s = roark_state(tables, names, player_moves=["Bug Bite"])
    juice = _item(s, 1, 0, "Berry Juice", names).replace_hp(0, 0, 10)
    out = assert_matches_reference(
        tables, juice, A("Bug Bite", "Leer", juice, names), empty_outcome()
    )
    assert int(out.item[1, 0]) == names.item("Berry Juice") and int(out.hp[0, 0]) == 10
    # A paralyzed eater is cured by the foe's Lum Berry; its own item is not touched.
    lum = _item(_item(s, 1, 0, "Lum Berry", names), 0, 0, "Oran Berry", names)
    lum = lum.with_status(0, 0, Status.PARALYSIS)
    out = assert_matches_reference(tables, lum, A("Bug Bite", "Leer", lum, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and _consumed(out, 1, "Lum Berry", names)
    assert int(out.item[0, 0]) == names.item("Oran Berry")


def test_pluck_flag_keeps_the_eaters_own_berry(tables, names):
    # A healthy eater plucks a Pecha Berry: no restore subscript runs, so
    # SELF_TURN_FLAG_PLUCK_BERRY stays set (battle_lib.c:5848). Static then paralyzes the
    # eater (contact), and its own Lum Berry cures it at AFTER_MOVE_EFFECT_ATTACKER_ITEM, but
    # subscript_pluck_check.s skips RemoveItem: the Lum Berry stays.
    s = roark_state(tables, names, player_moves=["Bug Bite"], opp_species="Shinx")
    s = s._replace(ability=s.ability.at[1, 0].set(AbilityId.STATIC))
    s = _item(_item(s, 1, 0, "Pecha Berry", names), 0, 0, "Lum Berry", names)
    procs = empty_outcome()._replace(contact_proc=jnp.ones((2, 5), jnp.int8))
    out = assert_matches_reference(tables, s, A("Bug Bite", "Leer", s, names), procs)
    assert _consumed(out, 1, "Pecha Berry", names)
    assert int(out.status[0, 0]) == Status.NONE
    assert int(out.item[0, 0]) == names.item("Lum Berry") and int(out.item_consumed[0, 0]) == 0


# --- Berry Juice and Big Root ---------------------------------------------------------


def test_berry_juice_heals_twenty_at_half(tables, names):
    s = _item(roark_state(tables, names), 0, 0, "Berry Juice", names)
    half = int(s.max_hp[0, 0]) // 2
    s = s.replace_hp(0, 0, half)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.hp[0, 0]) == min(half + 20, int(s.max_hp[0, 0]))
    assert _consumed(out, 0, "Berry Juice", names)


def test_big_root_boosts_drain(tables, names):
    s = roark_state(tables, names, player_species="Grotle", player_moves=["Absorb"])
    s = s.replace_hp(0, 0, 20)
    plain = _item(s, 0, 0, "", names)
    root = _item(s, 0, 0, "Big Root", names)
    acts = A("Absorb", "Leer", s, names)
    base = assert_matches_reference(tables, plain, acts, empty_outcome())
    boosted = assert_matches_reference(tables, root, acts, empty_outcome())
    dealt = int(s.hp[1, 0]) - int(base.hp[1, 0])
    heal = max(dealt // 2, 1)
    assert int(base.hp[0, 0]) == 20 + heal
    assert int(boosted.hp[0, 0]) == 20 + heal * 130 // 100


def test_big_root_boosts_leech_seed(tables, names):
    s = roark_state(tables, names, player_moves=["Leech Seed"])
    s = _item(s, 0, 0, "Big Root", names).replace_hp(0, 0, 10)
    s = s._replace(leech_seed=s.leech_seed.at[1].set(1))
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    lost = max(int(s.max_hp[1, 0]) // 8, 1)
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) - lost
    assert int(out.hp[0, 0]) == 10 + lost * 130 // 100


# --- Lagging Tail ---------------------------------------------------------------------


def test_lagging_tail_moves_last_within_priority(tables, names):
    s = roark_state(tables, names, player_moves=["Quick Attack"])
    assert int(s.stats[0, 0, Stat.SPE - 1]) > int(s.stats[1, 0, Stat.SPE - 1])
    tail = _item(s, 0, 0, "Lagging Tail", names)
    # Leer vs Leer: the faster holder goes second, so the opponent's Leer lands first.
    leers = A("Leer", "Leer", s, names)

    def order(state):
        return _order(tables, state, leers)

    assert order(s).tolist() == [0, 1] and order(tail).tolist() == [1, 0]
    # Priority still decides first: Quick Attack (+1) goes first.
    assert _order(tables, tail, A("Quick Attack", "Leer", s, names)).tolist() == [0, 1]
    # Both hold it: speed reversed, the slower first.
    both = _item(tail, 1, 0, "Lagging Tail", names)
    assert order(both).tolist() == [1, 0]
    slow = both._replace(stats=both.stats.at[1, 0, Stat.SPE - 1].set(999))
    assert order(slow).tolist() == [0, 1]


def _order(tables, s, acts):
    return action_order(tables, s, acts, empty_outcome())


def test_lagging_tail_end_of_turn_order(tables, names):
    # Both last Pokemon poisoned at 1 HP: the battler checked first faints first and its
    # side loses (bcp.c:1289, monSpeedOrder). The faster player goes first unless it holds
    # a Lagging Tail.
    s = roark_state(tables, names, player_count=1, opp_count=1)
    s = s.with_status(0, 0, Status.POISON).with_status(1, 0, Status.POISON)
    s = s.replace_hp(0, 0, 1).replace_hp(1, 0, 1)
    assert end_of_turn_order(tables, s, empty_outcome()).tolist() == [0, 1]
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.winner) == WINNER_OPPONENT
    tail = _item(s, 0, 0, "Lagging Tail", names)
    assert end_of_turn_order(tables, tail, empty_outcome()).tolist() == [1, 0]
    out = assert_matches_reference(tables, tail, A("Leer", "Leer", tail, names), empty_outcome())
    assert int(out.winner) == WINNER_PLAYER


# --- relevance, prep -------------------------------------------------------------------


def test_lum_cure_makes_the_end_speed_tie_relevant(tables, names):
    # A paralyzed Lum holder is cured after its move: its end-of-turn speed is its
    # unparalyzed one, which here equals the foe's.
    s = _item(roark_state(tables, names, equal_speed=True), 0, 0, "Lum Berry", names)
    s = s.with_status(0, 0, Status.PARALYSIS)
    laws = outcome_laws(tables, s, A("Leer", "Leer", s, names))
    assert bool(laws.relevant.end_speed_tie)
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE


def test_focus_sash_hp_cut(tables, names):
    mon = sample_party(names, override_item={0: "Focus Sash"})[0]
    plain = sample_party(names)[0]
    max_hp = 40
    assert max_hp - 1 in hp_cuts(tables, mon, max_hp)
    assert max_hp - 1 not in hp_cuts(tables, plain, max_hp)


# --- both engines -----------------------------------------------------------------------


def test_k3b_items_match_reference(tables, names):
    base = roark_state(tables, names, player_moves=["Bug Bite", "Double Kick", "Absorb"])
    o = empty_outcome()
    for item in ("Focus Sash", "Big Root", "Berry Juice", "Chesto Berry", "Pecha Berry",
                 "Lum Berry", "White Herb", "Lagging Tail"):  # fmt: skip
        for side in (0, 1):
            s = _item(base, side, 0, item, names)
            for hp in (int(s.max_hp[side, 0]), 3, int(s.max_hp[side, 0]) // 2):
                st = s.replace_hp(side, 0, hp)
                for move in ("Bug Bite", "Double Kick", "Absorb", "Scratch"):
                    for opp in ("Tackle", "Leer"):
                        assert_matches_reference(tables, st, A(move, opp, st, names), o)
            sick = (
                s.with_status(side, 0, Status.TOXIC)
                ._replace(confusion=s.confusion.at[side].set(2))
                .with_stages(side, spe=-2)
            )
            assert_matches_reference(tables, sick, A("Scratch", "Tackle", sick, names), o)
            assert_matches_reference(tables, sick, A(SWITCH_TO_1, "Tackle", sick, names), o)


def test_make_state_cures_a_berry_holders_status(tables, names):
    # SWITCH_IN_CHECK_STATE_HELD_ITEM_STATUS (battle_lib.c:4151) at the battle start.
    from dataclasses import replace

    party = sample_party(names, override_item={0: "Pecha Berry"})
    party[0] = replace(party[0], status=int(Status.POISON))
    opp = sample_party(names)[:1]
    s = make_state(tables, party, opp)
    assert int(s.status[0, 0]) == Status.NONE and int(s.item_consumed[0, 0]) == names.item(
        "Pecha Berry"
    )
