"""D3-9: Pursuit over four battlers (sheet D3-9 sections 1, 4.5, 4.6). ``BtlCmd_TryPursuit``
(battle_script.c:6884-6953) scans ``i = 0 .. maxBattlers - 1`` in index order (:6899, ``break``
:6934; never by speed) for the first battler of the other side that is not MOVE_END, alive, not
asleep or frozen, not loafing, whose move (its Encore move when the Encore slot holds it, else the
selected one: :6905-6909) has the Pursuit effect and PP in the first slot holding it (:6910-6916);
subscript_pursuit.s:79-82 loops to the next one while the switcher lives. SF2 (e) landed (E2-ML-6):
in every format an Encored battler pursues whatever it chose.

Each ``play`` case is the engine's action phase asserted equal to the reference's (every field).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.engine.actions import FIRST_ITEM_ACTION, MOVE_ACTIONS
from nuzline.engine.format import DOUBLES
from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Stat
from nuzline.engine.outcome import empty_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import ACT_MOVE, next_pursuer, pursuit_fires
from tests.engine.d2_states import four, party_of, with_ability, with_hp, with_moves
from tests.engine.helpers import A, assert_matches_reference
from tests.engine.test_d3_action_loop import hp, play, pp
from tests.engine.test_pivot import _mon

PURSUIT, IDLE = "Pursuit", "Swords Dance"
SWITCH_2 = MOVE_ACTIONS + 2  # to party slot 2 (the benches hold six)


def idle_state(tables, names):
    """DOUBLES, every battler knows only Swords Dance."""
    s = four(tables, names, DOUBLES)
    for b in range(4):
        s = with_moves(s, names, b, [IDLE])
    return s


def pursuers(tables, names, *battlers):
    s = idle_state(tables, names)
    for b in battlers:
        s = with_moves(s, names, b, [PURSUIT])
    return s


def mask(*battlers):
    return jnp.asarray([b in battlers for b in range(4)])


def encore(state, names, b: int, slot: int, move: str):
    """Battler ``b`` is Encored into the move in ``slot`` (the slot holds it)."""
    return state._replace(
        encore_move=state.encore_move.at[b].set(names.move(move)),
        encore_slot=state.encore_slot.at[b].set(slot),
        encore_turns=state.encore_turns.at[b].set(3),
    )


def choices_of(tables, state, actions):
    from nuzline.engine.order import selected_move

    picks = [selected_move(tables, state, b, actions[b]) for b in range(4)]
    return jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks])


def scan(tables, state, switcher, actions, move_end=()):
    """``next_pursuer`` on ``state`` as ``step._action_phase`` hands it to the action loop: the
    selection written (``with_selection``: moveSlot and TEMP_VALUE, E2-ML-5, E2-ML-8), which
    TryPursuit's ``Battler_SelectedMove`` reads (battle_lib.c:2858-2868). The in-loop scan never
    re-selects, so a raw turn-start state finds no FIGHT's move there."""
    from nuzline.engine.order import with_selection

    acts = jnp.asarray(actions, jnp.int32)
    state = with_selection(tables, state, acts)
    found, who = next_pursuer(
        tables, state, switcher, acts, choices_of(tables, state, acts), mask(*move_end)
    )
    return bool(found), int(who)


# --- the scan -------------------------------------------------------------------------------


def test_the_scan_takes_the_lowest_battler_not_the_fastest(tables, names):
    """Battler 0 switches, battlers 1 and 3 chose Pursuit: 1 first, then 3 once 1 is MOVE_END
    (battle_script.c:6899, :6934), whatever the speeds."""
    s = pursuers(tables, names, 1, 3)
    s = s._replace(stats=s.stats.at[1, 1, Stat.SPE - 1].set(10).at[1, 0, Stat.SPE - 1].set(500))
    acts = [SWITCH_2, 0, 0, 0]
    assert scan(tables, s, 0, acts) == (True, 1)
    assert scan(tables, s, 0, acts, move_end=(1,)) == (True, 3)
    assert scan(tables, s, 0, acts, move_end=(1, 3)) == (False, 1)


def test_the_switchers_partner_never_pursues(tables, names):
    """Battler 1 switches: battlers 0 and 2 (the other side) pursue, 0 first; battler 3, the
    switcher's partner, does not (:6904)."""
    s = pursuers(tables, names, 0, 2, 3)
    acts = [0, SWITCH_2, 0, 0]
    assert scan(tables, s, 1, acts) == (True, 0)
    assert scan(tables, s, 1, acts, move_end=(0,)) == (True, 2)
    assert scan(tables, s, 1, acts, move_end=(0, 2))[0] is False
    only_partner = pursuers(tables, names, 3)
    assert scan(tables, only_partner, 1, acts)[0] is False


def test_the_scan_asks_for_pp_alive_sleep_and_a_pursuit_move(tables, names):
    from nuzline.engine.ids import Status

    s = pursuers(tables, names, 1, 3)
    acts = [SWITCH_2, 0, 0, 0]
    p1, slot1 = party_of(s, 1), int(s.active[1])
    drained = s._replace(pp=s.pp.at[p1, slot1, 0].set(0))
    assert scan(tables, drained, 0, acts) == (True, 3)  # PP of the first slot holding it
    asleep = s._replace(status=s.status.at[p1, slot1].set(Status.SLEEP))
    assert scan(tables, asleep, 0, acts) == (True, 3)
    other = with_moves(s, names, 1, ["Tackle"])
    assert scan(tables, other, 0, acts) == (True, 3)
    assert scan(tables, with_hp(s, 1, 0), 0, acts) == (True, 3)


def test_pursuit_fires_reads_the_turn_start_state(tables, names):
    s = pursuers(tables, names, 3)
    acts = jnp.asarray([SWITCH_2, 0, 0, 0], jnp.int32)
    ch = choices_of(tables, s, acts)
    assert bool(pursuit_fires(tables, s, 0, acts, ch))
    stay = jnp.asarray([0, 0, 0, 0], jnp.int32)
    assert not bool(pursuit_fires(tables, s, 0, stay, choices_of(tables, s, stay)))


# --- the hits --------------------------------------------------------------------------------


def test_both_foes_pursue_and_the_lower_battler_hits_first(tables, names):
    """Battler 0 (Rough Skin) switches; battlers 1 (1 HP) and 3 chose Pursuit, targets anywhere,
    battler 3 the faster: battler 1 hits first (index order) and falls to the Rough Skin, then
    battler 3 hits as well (the switcher lived) and takes the Rough Skin too. Each spent one PP; both are MOVE_END, so neither
    acts at its own turn (battler 2, their chosen target, is untouched); the switch completes."""
    s = with_hp(pursuers(tables, names, 1, 3), 1, 1)
    s = with_ability(s, 0, AbilityId.ROUGH_SKIN)
    out, shown = play(tables, s, [SWITCH_2, 0, 0, 0], (0, 3, 1, 2), targets=(0, 2, 0, 2))
    assert hp(out, 1) == 0  # battler 1 (1 HP) fell to the Rough Skin
    assert hp(out, 3) == hp(s, 3) - hp(s, 3) // 8 == 98  # battler 3 took it too (lib.c:4257-4268)
    assert pp(out, 1, 0) == pp(s, 1, 0) - 1 and pp(out, 3, 0) == pp(s, 3, 0) - 1
    assert hp(out, 2) == hp(s, 2)  # no Pursuit reached its own target
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # the recalled switcher took both hits
    assert int(out.active[0]) == 2
    assert int(shown["pursuit"]) == 1
    for b in (1, 3):
        assert int(shown["act_kind"][b]) == ACT_MOVE and int(shown["hits"][b]) == 1
    assert int(out.field_mask) == int(s.field_mask)  # no Acid Rain (subscript_pursuit.s:115)


def test_the_second_pursuer_meets_the_switchers_ability_not_its_partners(tables, names):
    """As the both-foes case, but the Rough Skin is on battler 2 (the switcher's partner): neither
    pursuer reads it, the defender is the switcher (battle_script.c:6926-6928)."""
    s = with_ability(pursuers(tables, names, 1, 3), 2, AbilityId.ROUGH_SKIN)
    out, _ = play(tables, s, [SWITCH_2, 0, 0, 0], (0, 3, 1, 2), targets=(0, 2, 0, 2))
    assert hp(out, 1) == hp(s, 1) and hp(out, 3) == hp(s, 3)
    assert hp(out, 2) == hp(s, 2)
    assert int(out.active[0]) == 2


def test_the_first_ko_ends_the_chain(tables, names):
    """The switcher has 1 HP: battler 1's Pursuit KOs it, so battler 3 does not pursue (the KO
    path takes ``_166`` and ends, subscript_pursuit.s:39, :84-120): it acts at its own turn and
    hits its target, battler 2."""
    s = pursuers(tables, names, 1, 3)
    s = s._replace(hp=s.hp.at[0, 0].set(1))
    out, _ = play(tables, s, [SWITCH_2, 0, 0, 0], (0, 3, 1, 2), targets=(0, 2, 0, 2))
    assert int(out.hp[0, 0]) == 0 and int(out.active[0]) == 2
    assert pp(out, 1, 0) == pp(s, 1, 0) - 1  # the pursuer
    assert pp(out, 3, 0) == pp(s, 3, 0) - 1 and hp(out, 2) < hp(s, 2)  # battler 3's own turn


def test_each_pursuer_pays_two_pp_against_pressure(tables, names):
    s = with_ability(pursuers(tables, names, 1, 3), 0, AbilityId.PRESSURE)
    out, _ = play(tables, s, [SWITCH_2, 0, 0, 0], (0, 1, 3, 2), targets=(0, 2, 0, 2))
    assert pp(out, 1, 0) == pp(s, 1, 0) - 2 and pp(out, 3, 0) == pp(s, 3, 0) - 2


def test_an_encored_pursuer_pursues_whatever_it_chose(tables, names):
    """Battler 1's Encore slot holds Pursuit while it chose slot 0 (Swords Dance): it pursues
    (battle_script.c:6905-6907), its Swords Dance never runs."""
    s = with_moves(idle_state(tables, names), names, 1, [IDLE, PURSUIT])
    s = encore(s, names, 1, 1, PURSUIT)
    out, _ = play(tables, s, [SWITCH_2, 0, 0, 0], (0, 1, 2, 3), targets=(0, 2, 0, 2))
    assert pp(out, 1, 1) == pp(s, 1, 1) - 1 and pp(out, 1, 0) == pp(s, 1, 0)
    assert int(out.stages[1, Stat.ATK]) == int(s.stages[1, Stat.ATK])
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_an_encored_pursuer_that_chose_an_item_pursues_and_its_item_is_not_used(tables, names):
    """Battler 3 Encored into Pursuit picks a bag item; battler 0 switches first (battler order
    between an ITEM and a switch, bcp.c:754-761): battler 3 pursues, its item never runs, the bag
    slot is spent."""
    s = with_moves(idle_state(tables, names), names, 3, [PURSUIT])
    s = encore(s, names, 3, 0, PURSUIT)
    s = with_hp(s, 3, hp(s, 3) - 10)
    potion = names.item("Potion")
    s = s._replace(
        trainer_items=s.trainer_items.at[3, 0].set(potion),
        trainer_item_count=s.trainer_item_count.at[3].set(1),
    )
    out, _ = play(tables, s, [SWITCH_2, 0, 0, FIRST_ITEM_ACTION], (0, 3, 1, 2))
    assert pp(out, 3, 0) == pp(s, 3, 0) - 1
    assert hp(out, 3) == hp(s, 3)  # the Potion was not used
    assert int(out.trainer_items[3, 0]) == 0  # spent at selection
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_an_encored_item_runs_first_when_its_battler_comes_before_the_switch(tables, names):
    """Battler 1 Encored picks an item, battler 2 switches: items and switches go in battler
    order, so battler 1's item runs first (it is MOVE_END) and no Pursuit fires."""
    s = with_moves(idle_state(tables, names), names, 1, [PURSUIT])
    s = encore(s, names, 1, 0, PURSUIT)
    s = with_hp(s, 1, hp(s, 1) - 10)
    s = s._replace(
        trainer_items=s.trainer_items.at[1, 0].set(names.item("Potion")),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )
    out, _ = play(tables, s, [0, FIRST_ITEM_ACTION, SWITCH_2, 0], (1, 2, 0, 3))
    assert hp(out, 1) > hp(s, 1)  # the Potion healed
    assert pp(out, 1, 0) == pp(s, 1, 0)  # no Pursuit
    assert int(out.active[2]) == 2


def test_an_encored_switch_slower_than_the_switch_pursues_and_stays_in(tables, names):
    """Battler 3 Encored into Pursuit picks a switch, slower than battler 0's switch (two
    switches go by speed, bcp.c:772-785): battler 3 pursues and stays in."""
    s = with_moves(idle_state(tables, names), names, 3, [PURSUIT])
    s = encore(s, names, 3, 0, PURSUIT)
    out, _ = play(tables, s, [SWITCH_2, 0, 0, SWITCH_2], (0, 3, 1, 2))
    assert int(out.active[3]) == int(s.active[3])  # its switch was never made
    assert int(out.active[0]) == 2
    assert pp(out, 3, 0) == pp(s, 3, 0) - 1
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])


def test_the_switchers_partner_uses_pursuit_at_its_own_turn(tables, names):
    """Battler 1 switches: battlers 0 and 2 pursue, 0 first; battler 3 (the partner) does not,
    and hits its target at its own turn."""
    s = pursuers(tables, names, 0, 2, 3)
    s = with_ability(s, 1, AbilityId.ROUGH_SKIN)
    s = with_hp(s, 0, 1)
    out, _ = play(tables, s, [0, SWITCH_2, 0, 0], (1, 0, 2, 3), targets=(3, 0, 3, 2))
    assert hp(out, 0) == 0  # battler 0 (1 HP) hit first and fell to the Rough Skin
    assert pp(out, 0, 0) == pp(s, 0, 0) - 1 and pp(out, 2, 0) == pp(s, 2, 0) - 1
    assert pp(out, 3, 0) == pp(s, 3, 0) - 1  # 3 acts at its own turn
    # battler 2: the Rough Skin as the second pursuer, then battler 3's own-turn Pursuit
    assert hp(out, 2) < hp(s, 2) - hp(s, 2) // 8
    assert hp(out, 2) == 79


def test_u_turn_is_met_by_both_foes_before_the_list(tables, names):
    """Battler 2's U-turn at battler 3 (the player side's, so the pursuers are battlers 1 and
    3): both hit battler 2 before the party list, the turn pauses with both bits in
    ``ts_pursued`` (1 first, in index order)."""
    s = with_moves(pursuers(tables, names, 1, 3), names, 2, ["U-turn"])
    s = s._replace(hp=s.hp.at[0, 1].set(s.hp[0, 1]))
    out, shown = play(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), targets=(0, 0, 3, 0))
    assert int(out.pivot_battler) == 3
    assert int(out.ts_pursued) == 0b1010
    assert pp(out, 1, 0) == pp(s, 1, 0) - 1 and pp(out, 3, 0) == pp(s, 3, 0) - 1
    assert hp(out, 2) < hp(s, 2)
    assert int(shown["pursuit"]) == 1
    assert int(out.ts_pick_mul) == 20


def test_a_felled_u_turn_users_rough_skin_stays_in_its_record(tables, names):
    """Battler 1's Pursuit fells battler 2 (1 HP, Rough Skin) at its U-turn: FAINT_MON clears its
    record (subscript_pursuit.s:85, battle_script.c:1803, battle_lib.c:2223-2225), then the Rough
    Skin message writes it (:86-87, battle_script.c:11805-11806) until the replacement enters."""
    s = with_moves(pursuers(tables, names, 1, 3), names, 2, ["U-turn"])
    s = with_hp(with_ability(s, 2, AbilityId.ROUGH_SKIN), 2, 1)
    out, _ = play(tables, s, [0, 0, 0, 0], (2, 0, 1, 3), targets=(0, 0, 3, 0))
    assert hp(out, 2) == 0 and int(out.pivot_battler) == 0
    assert int(out.ai_ability_known[2]) == ABILITY_DECOMP_IDS[AbilityId.ROUGH_SKIN]


# --- singles (B = 2) ----------------------------------------------------------------------


def test_singles_an_encored_pursuer_with_an_item_pursues(tables, names):
    """SF2 (e), E2-ML-6: TryPursuit takes a held Encore's move before Battler_SelectedMove
    (battle_script.c:6905-6909) with no command test, so battler 1 Encored into Pursuit that
    picked a bag item pursues battler 0's switch (items and switches run in battler order,
    bcp.c:754-768) and its item action is then MOVE_END (:6931)."""
    n = names
    s = make_state(
        tables,
        [_mon(n, "Bulbasaur", ["Tackle"], 35), _mon(n, "Geodude", ["Tackle"], 30)],
        [_mon(n, "Houndour", ["Pursuit"], 35)],
    )
    s = encore(s, n, 1, 0, PURSUIT)
    s = s._replace(
        trainer_items=s.trainer_items.at[1, 0].set(n.item("Potion")),
        trainer_item_count=s.trainer_item_count.at[1].set(1),
    )
    switch, use = MOVE_ACTIONS + 1, FIRST_ITEM_ACTION
    out = assert_matches_reference(tables, s, (switch, use), empty_outcome())
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])  # the Pursuit hit the switcher
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1
    fight = A("Tackle", "Pursuit", s, n)
    out = assert_matches_reference(tables, s, (switch, int(fight[1])), empty_outcome())
    assert int(out.pp[1, 0, 0]) == int(s.pp[1, 0, 0]) - 1  # a FIGHT still pursues
