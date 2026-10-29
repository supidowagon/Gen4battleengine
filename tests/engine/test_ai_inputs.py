"""Engine inputs the trainer AI reads (docs/ai/roark.md sections 5.3, 8.2, 11.1, 12).

* ``trainer_item_count``: ``aiContext.trainerItemCounts``, the bag size at battle
  start. An item action also zeroes the later slots ``TrainerAI_ShouldUseItem``
  (trainer_ai.c:4056) considers after a use: no ``break``, gate
  ``i == 0 || aliveMons <= trainerItemCounts - i + 1`` (:4092).
* ``fake_out_turn``: ``fakeOutTurnNumber``, set to ``totalTurns + 1`` by every
  switch-in (``BattleSystem_UpdateAfterSwitch``, battle_lib.c:2117) and faint
  (``BattleSystem_CleanupFaintedMon``, :2195); ``LoadIsFirstTurnInBattle``
  (trainer_ai.c:2487) is TRUE when ``fakeOutTurnNumber >= totalTurns``.
* ``move_hit``: ``moveHit``, written by ``UpdateFlagsWhenHit``
  (battle_controller_player.c:4500-4507) and cleared at the battler's own move end
  (``ClearFlags``, :4389), on switch-in (battle_lib.c:2121) and on a faint (:2199).
* ``species_abilities_decomp`` / ``ability_slot``: the decomp ability id of every
  Pokemon, also for abilities the engine does not implement.
"""

import jax.numpy as jnp
import numpy as np

from nuzline.data.tables import to_numpy
from nuzline.engine.actions import FIRST_ITEM_ACTION
from nuzline.engine.ids import AbilityId, Status
from nuzline.engine.outcome import empty_outcome, hit_threshold, outcome_to_py
from nuzline.engine.step import replace, step
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_replace, ref_step
from tests.engine.helpers import roark_state

POTION = 17  # decomp ITEM_POTION
USE_SLOT_0, USE_SLOT_1 = FIRST_ITEM_ACTION, FIRST_ITEM_ACTION + 1
SWITCH_TO_1 = 4 + 1
SPE = 4  # stats[..., 4] is Speed


def _same(a, b):
    for name, x, y in zip(a._fields, a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


def _both(tables, s, acts, o=None):
    """``step`` and ``ref_step`` agree; returns the JAX result."""
    o = empty_outcome() if o is None else o
    out = step(tables, s, jnp.asarray(acts, jnp.int32), o)
    ref = ref_step(to_numpy(tables), to_ref(s), tuple(int(a) for a in acts), outcome_to_py(o))
    _same(out, from_ref(ref))
    return out


def _both_replace(tables, s, side, slot):
    out = replace(tables, s, side, slot)
    _same(out, from_ref(ref_replace(to_numpy(tables), to_ref(s), side, slot)))
    return out


def _bag(s, items, count):
    return s._replace(
        trainer_items=s.trainer_items.at[1].set(jnp.asarray(items, jnp.int16)),
        trainer_item_count=s.trainer_item_count.at[1].set(count),
    )


def _player_faster(s):
    return s._replace(stats=s.stats.at[1, 0, SPE].set(1))


# --- trainer_item_count and the bag loop ---------------------------------------------


def test_make_state_records_the_bag_size(tables, names):
    s = roark_state(tables, names)
    assert s.trainer_item_count.dtype == jnp.int8
    assert s.trainer_item_count.tolist() == [0, 2]  # trainerItemCounts, bcp.c:4796-4812


def test_two_potions_both_spent_with_two_alive(tables, names):
    # aliveMons = 2 <= 2 - 1 + 1: slot 1 is considered after slot 0's use and zeroed too,
    # but only one Potion heals (roark.md section 12, quirk 5; trace E2).
    s = roark_state(tables, names, opp_count=2)
    max_hp = int(s.max_hp[1, 0])
    s = s.replace_hp(1, 0, max_hp - 30)
    out = _both(tables, s, [SWITCH_TO_1, USE_SLOT_0])
    assert out.trainer_items[1].tolist() == [0, 0, 0, 0]
    assert int(out.hp[1, 0]) == max_hp - 10
    assert out.trainer_item_count.tolist() == [0, 2]  # never decremented


def test_second_slot_untouched_with_three_alive(tables, names):
    # aliveMons = 3 > 2 - 1 + 1: slot 1 is not considered, so it keeps its Potion.
    s = roark_state(tables, names).replace_hp(1, 0, 5)
    out = _both(tables, s, [SWITCH_TO_1, USE_SLOT_0])
    assert out.trainer_items[1].tolist() == [0, POTION, 0, 0]
    # Using slot 1 itself spends only slot 1.
    out = _both(tables, s, [SWITCH_TO_1, USE_SLOT_1])
    assert out.trainer_items[1].tolist() == [POTION, 0, 0, 0]
    # A full bag of four with 3 alive: slots i with 3 <= 4 - i + 1 (i <= 2) are zeroed.
    out = _both(tables, _bag(s, [POTION] * 4, 4), [SWITCH_TO_1, USE_SLOT_0])
    assert out.trainer_items[1].tolist() == [0, 0, 0, POTION]


def test_bag_gate_counts_alive_mons_at_command_selection(tables, names):
    # Three in the party, one fainted: aliveMons = 2, so slot 1 goes too.
    s = roark_state(tables, names).replace_hp(1, 2, 0).replace_hp(1, 0, 5)
    out = _both(tables, s, [SWITCH_TO_1, USE_SLOT_0])
    assert out.trainer_items[1].tolist() == [0, 0, 0, 0]


# --- fake_out_turn ---------------------------------------------------------------------


def test_fake_out_turn_set_on_every_switch_in(tables, names):
    base = roark_state(tables, names)
    assert base.fake_out_turn.dtype == jnp.int16
    assert base.fake_out_turn.tolist() == [0, 0]  # leads: BattleContext zeroed, bcp.c:180
    s = base._replace(turn=jnp.int16(7))  # decomp totalTurns during this turn

    # A voluntary switch, each side: totalTurns + 1.
    out = _both(tables, s, [SWITCH_TO_1, 0])
    assert out.fake_out_turn.tolist() == [8, 0] and int(out.turn) == 8
    out = _both(tables, s, [0, SWITCH_TO_1])
    assert out.fake_out_turn.tolist() == [0, 8]

    # A switch that Pursuit hits first.
    pursuit = roark_state(tables, names, opp_moves=["Pursuit"])._replace(turn=jnp.int16(7))
    out = _both(tables, pursuit, [SWITCH_TO_1, 0])
    assert int(out.hp[0, 0]) < int(pursuit.hp[0, 0])  # Pursuit hit the outgoing Pokemon
    assert out.fake_out_turn.tolist() == [8, 0]

    # A faint (CleanupFaintedMon) and its replacement, which runs in TurnEnd before
    # totalTurns++ (bcp.c:1844-1852): both write the turn's totalTurns + 1.
    ko = _player_faster(s).replace_hp(1, 0, 1)
    fainted = _both(tables, ko, [names.move_slot(ko, 0, "Scratch"), 0])
    assert int(fainted.hp[1, 0]) == 0 and int(fainted.turn) == 8
    assert int(fainted.fake_out_turn[1]) == 8
    entered = _both_replace(tables, fainted, 1, 1)
    assert int(entered.fake_out_turn[1]) == 8


def test_first_turn_check_is_true_only_on_the_first_decision(tables, names):
    # LoadIsFirstTurnInBattle (trainer_ai.c:2487): TRUE when fake_out_turn >= turn.
    s = roark_state(tables, names)._replace(turn=jnp.int16(7))
    out = _both(tables, s, [0, SWITCH_TO_1])
    assert int(out.fake_out_turn[1]) >= int(out.turn)  # its first decision: TRUE
    later = _both(tables, out, [0, 0])
    assert int(later.fake_out_turn[1]) < int(later.turn)  # the next one: FALSE
    assert int(s.fake_out_turn[1]) < int(s.turn)  # a lead at turn 7: FALSE


# --- move_hit --------------------------------------------------------------------------


def test_move_hit_tracks_last_hit_move(tables, names):
    s = _player_faster(roark_state(tables, names))
    scratch, tackle = names.move("Scratch"), names.move("Tackle")
    acts = [names.move_slot(s, 0, "Scratch"), names.move_slot(s, 1, "Tackle")]
    # The player moves first: the opponent's own move end clears what hit it
    # (ClearFlags, bcp.c:4389), while the player keeps the Tackle that hit it after.
    out = _both(tables, s, acts)
    assert out.move_hit.dtype == jnp.int16
    assert out.move_hit.tolist() == [tackle, 0]
    # The opponent moves first: it keeps the Scratch into its next decision.
    fast = roark_state(tables, names, opp_faster=True)
    assert _both(tables, fast, acts).move_hit.tolist() == [0, scratch]
    # A status move aimed at the foe is recorded too (moveHit holds moveCur).
    leer = [names.move_slot(fast, 0, "Leer"), acts[1]]
    assert _both(tables, fast, leer).move_hit.tolist() == [0, names.move("Leer")]


def test_move_hit_survives_the_faint_it_causes(tables, names):
    # LOOP_FAINTED (CleanupFaintedMon) runs before UpdateFlagsWhenHit (bcp.c:3837), so a
    # KO'd defender holds the move until its replacement clears it (battle_lib.c:2121).
    s = _player_faster(roark_state(tables, names)).replace_hp(1, 0, 1)
    out = _both(tables, s, [names.move_slot(s, 0, "Scratch"), 0])
    assert int(out.hp[1, 0]) == 0 and int(out.move_hit[1]) == names.move("Scratch")
    assert int(_both_replace(tables, out, 1, 1).move_hit[1]) == 0


def test_move_hit_resets(tables, names):
    scratch, tackle = names.move("Scratch"), names.move("Tackle")
    s = roark_state(tables, names, opp_faster=True).replace_hp(1, 0, 5)
    s = s._replace(move_hit=jnp.asarray([tackle, scratch], jnp.int16))
    # A switch-in clears the newcomer's (battle_lib.c:2121); an item use ends with the
    # user's move end (ItemCommand -> MOVE_END, bcp.c:1973), which clears its own.
    out = _both(tables, s, [SWITCH_TO_1, USE_SLOT_0])
    assert out.move_hit.tolist() == [0, 0]
    # A move that never starts (full paralysis) leaves no record on its target, and its
    # user's move end still clears the Tackle that hit it first.
    fast = roark_state(tables, names, opp_faster=True)
    para = fast._replace(status=fast.status.at[0, 0].set(Status.PARALYSIS))
    o = empty_outcome()._replace(full_para=jnp.asarray([1, 0], jnp.int8))
    acts = [names.move_slot(fast, 0, "Scratch"), names.move_slot(fast, 1, "Tackle")]
    assert _both(tables, para, acts, o).move_hit.tolist() == [0, 0]
    assert _both(tables, para, acts).move_hit.tolist() == [0, scratch]


def test_pursuer_keeps_its_move_hit(tables, names):
    # TryPursuit ends the pursuer's action (battlerActions = MOVE_END, battle_script.c:
    # 6931) without its own move end, so nothing clears what last hit it.
    scratch = names.move("Scratch")
    s = roark_state(tables, names, opp_moves=["Pursuit"])
    s = s._replace(move_hit=jnp.asarray([0, scratch], jnp.int16))
    assert _both(tables, s, [SWITCH_TO_1, 0]).move_hit.tolist() == [0, scratch]
    tackle = names.move_slot(s, 1, "Tackle")
    assert _both(tables, s, [SWITCH_TO_1, tackle]).move_hit.tolist() == [names.move("Tackle"), 0]


# --- decomp ability ids ----------------------------------------------------------------


def test_ability_slot_and_decomp_ability_ids(tables, names):
    decomp = tables.species_abilities_decomp
    assert decomp.dtype == jnp.int16 and decomp.shape == tables.species_abilities.shape
    rows = np.asarray(decomp).tolist()
    assert rows == [list(names.species_abilities(i)) for i in range(len(rows))]
    shinx = names.species("Shinx")
    assert int(decomp[shinx, 1]) == names.ability("Intimidate")
    # An ability the engine does not implement still has its decomp id.
    voltorb = names.species("Voltorb")
    assert int(tables.species_abilities[voltorb, 0]) == AbilityId.UNSUPPORTED
    assert int(decomp[voltorb, 0]) == names.ability("Soundproof")

    s = roark_state(tables, names, player_species="Shinx", player_ability="Intimidate")
    assert s.ability_slot.dtype == jnp.int8
    assert int(s.ability_slot[0, 0]) == 1 and int(s.ability[0, 0]) == AbilityId.INTIMIDATE
    assert int(decomp[s.species[0, 0], s.ability_slot[0, 0]]) == names.ability("Intimidate")
    assert s.ability_slot[1].tolist() == [0] * 6  # build_trainer_party: ability slot 0


# --- hit threshold of an item action ---------------------------------------------------


def test_item_action_has_no_accuracy(tables, names):
    # Slot 3 holds Rock Throw (accuracy 90); item action 10 must not read it as a move.
    s = roark_state(tables, names, opp_moves=["Tackle", "Stealth Rock", "Leer", "Rock Throw"])
    assert int(hit_threshold(tables, s, jnp.array([0, 3]))[1]) == 90
    assert int(hit_threshold(tables, s, jnp.array([0, USE_SLOT_0]))[1]) == 100
