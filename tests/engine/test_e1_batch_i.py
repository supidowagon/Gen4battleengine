"""E1 batch I.

I-9: ``hooks.order_swaps`` needs no dead-battler line of its own (the sort's short-circuit sits in
``order.compare_speed``).
I-6: the cures of ``BattleSystem_RecoverStatusByAbility`` reveal no ability.
I-3: Sketch (move effect 95), the player side: ``moveSketched`` and ``BtlCmd_TrySketch``.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.actions import MOVE_ACTIONS, usable_moves
from nuzline.engine.hooks import ALL_SWAPS, REVEALS_ON, STATUS_CURE, order_swaps
from nuzline.engine.ids import (
    MOVE_SKETCH,
    SKETCH_FORBIDDEN,
    AbilityId,
    EffectId,
    MoveFlag,
    Status,
)
from nuzline.engine.outcome import empty_outcome
from tests.engine.helpers import A, assert_matches_reference, drop_programs, roark_state


def _down(s, side):
    return s._replace(hp=s.hp.at[side, s.active[side]].set(0))


@pytest.mark.parametrize("equal_speed", [False, True])
@pytest.mark.parametrize("opp_faster", [False, True])
def test_order_swaps_puts_a_fainted_battler_last(tables, names, equal_speed, opp_faster):
    """``BattleSystem_CompareBattlerSpeed`` (battle_lib.c:1204-1211): a fainted battler is SLOWER
    than a living one for every draw (tie bit, Quick Claw bits), whatever the speeds."""
    s = roark_state(tables, names, equal_speed=equal_speed, opp_faster=opp_faster)
    assert int(order_swaps(tables, _down(s, 0))) == ALL_SWAPS  # the player's second, every draw
    assert int(order_swaps(tables, _down(s, 1))) == 0  # the player's first, every draw
    assert order_swaps(tables, s).dtype == jnp.int32


# --- I-6: the cures of BattleSystem_RecoverStatusByAbility reveal nothing -------------------


def test_status_cure_reveals_are_shed_skin_and_hydration_alone():
    """``BattleMessage_AbilityTag`` returns ``msgAbilityTemp`` for ``BTLSCR_MSG_TEMP`` without
    ``BattleAI_SetAbility`` (battle_script.c:11801-11802), and ``RecoverStatusByAbility``'s
    script tags that (subscript_ability_forbids_status.s:6); Shed Skin's and Hydration's script
    tags a battler (subscript_ability_restore_status.s:8: reveal)."""
    holders = {a for a, kinds in REVEALS_ON.items() if STATUS_CURE in kinds}
    assert holders == {AbilityId.SHED_SKIN, AbilityId.HYDRATION}


@pytest.mark.parametrize(
    "species, ability, status",
    [
        ("Spinarak", "Insomnia", Status.SLEEP),
        ("Mankey", "Vital Spirit", Status.SLEEP),
        ("Persian", "Limber", Status.PARALYSIS),
    ],
)
def test_a_cure_at_move_end_leaves_the_ai_in_the_dark(tables, names, species, ability, status):
    """The statused holder is cured at the MoveEnd (battle_lib.c:4397-4410) and the AI's
    ``battlerAbilities`` stays unwritten (I-6): engine == reference."""
    s = roark_state(
        tables,
        names,
        player_species=species,
        player_ability=ability,
        player_moves=("Leer",),
        opp_moves=("Leer",),
    )
    s = s._replace(
        status=s.status.at[0, 0].set(status),
        status_counter=s.status_counter.at[0, 0].set(3 if status == Status.SLEEP else 0),
    )
    out = assert_matches_reference(tables, s, A("Leer", "Leer", s, names), empty_outcome())
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) == 0


def test_shed_skin_still_reveals_its_cure(tables, names):
    """subscript_ability_restore_status.s:8 tags ``BTLSCR_MSG_BATTLER_TEMP``: the AI learns it."""
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_ability="Shed Skin",
        player_moves=("Growl",),
        opp_moves=("Leer",),
    )
    s = s.with_status(0, 0, Status.POISON)
    shed = empty_outcome()._replace(shed_skin=jnp.array([1, 0], jnp.int8))
    out = assert_matches_reference(tables, s, A("Growl", "Leer", s, names), shed)
    assert int(out.status[0, 0]) == Status.NONE and int(out.ai_ability_known[0]) != 0


# --- I-3: Sketch (move effect 95, player side) ----------------------------------------------

SPE = 4  # stats[..., 0..4] are Atk..Spe
SKETCH_MOVES = ("Sketch", "Tackle", "Swords Dance", "Protect")


def _sketcher(tables, names, *, foe=("Rock Throw",), order="opp", mine=SKETCH_MOVES):
    """Smeargle (``mine``, Sketch first) against Geodude (``foe`` moves first), no abilities.
    ``order`` "opp": the foe is faster, so its move of this turn is what Sketch copies."""
    s = roark_state(
        tables, names, player_species="Smeargle", player_moves=list(mine), opp_moves=list(foe)
    )
    s = s._replace(ability=s.ability.at[0, int(s.active[0])].set(AbilityId.NONE))
    s = s._replace(ability=s.ability.at[1, int(s.active[1])].set(AbilityId.NONE))
    speed = int(s.stats[0, 0, SPE])
    foe_speed = {"player": speed - 1, "opp": speed + 1}[order]
    return s._replace(stats=s.stats.at[1, 0, SPE].set(foe_speed))


def _o():
    return empty_outcome()


def test_sketch_is_mapped_and_pinned(tables, names, kaizo_tables, kaizo_names):
    from tools.effect_map import DECOMP_EFFECT_TO_ID

    assert DECOMP_EFFECT_TO_ID[95] == (EffectId.SKETCH, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        m = n.move("Sketch")
        assert m == MOVE_SKETCH and int(t.move_effect[m]) == EffectId.SKETCH
        assert int(t.move_accuracy[m]) == 0 and int(t.move_power[m]) == 0
        assert int(t.move_pp[m]) == 1
        # Neither Protect nor Magic Coat nor Mirror Move touches it (no such flag).
        assert not int(t.move_flags[m]) & (
            MoveFlag.PROTECT | MoveFlag.MIRROR_MOVE | MoveFlag.MAGIC_COAT
        )


@pytest.mark.fullset
def test_sketch_forbidden_ids_match_the_decomp():
    """``ids.SKETCH_FORBIDDEN`` is ``BtlCmd_TrySketch``'s three moves (battle_script.c:4848-4850).
    A decomp pin: ``fullset``, as the other decomp-reading pins (test_e1_batch_d.py
    ``test_speed_halving_hold_effects_match_the_decomp``), and skipped where the decomp is not
    cloned (run locally then, CLAUDE.md "After merging main")."""
    decomp = Path(__file__).resolve().parents[2] / "external" / "pokeplatinum"
    if not decomp.exists():
        pytest.skip("external/pokeplatinum not cloned")
    moves = (decomp / "generated" / "moves.txt").read_text().split()
    source = (decomp / "src" / "battle" / "battle_script.c").read_text(encoding="utf-8")
    body = source[source.rindex("static BOOL BtlCmd_TrySketch") :]
    body = body[: body.index("return FALSE;")]
    listed = re.findall(r"moveSketched\[battleCtx->defender\] == (MOVE_\w+)", body)
    assert listed == ["MOVE_STRUGGLE", "MOVE_SKETCH", "MOVE_CHATTER", "MOVE_NONE"]
    assert [moves.index(m) for m in listed[:3]] == list(SKETCH_FORBIDDEN)
    assert MOVE_SKETCH == moves.index("MOVE_SKETCH")


def test_a_slower_sketch_copies_the_foes_move_for_good(tables, names):
    """The foe's Rock Throw is its ``moveSketched`` (UpdateMoveBuffers, bcp.c:3952-3954); the first
    Sketch slot takes it with the table PP, no shadow and no ``mimic_slots`` bit
    (battle_script.c:4862-4879): the party keeps it after a switch and at the battle's end."""
    drop_programs()
    s = _sketcher(tables, names)
    out = assert_matches_reference(tables, s, A("Sketch", "Rock Throw", s, names), _o())
    rock = names.move("Rock Throw")
    assert out.moves[0, 0].tolist() == [rock, *s.moves[0, 0, 1:].tolist()]
    assert int(out.pp[0, 0, 0]) == int(tables.move_pp[rock])  # table PP: Sketch's own PP is spent
    assert int(out.mimic_slots[0]) == 0 and out.shadow_moves[0].tolist() == [0] * 4
    assert int(out.sketch_move[1]) == rock and int(out.sketch_move[0]) == names.move("Sketch")
    # A switch-out and back: the BattleMon is reloaded from the party, which holds the copy.
    away = assert_matches_reference(
        tables, out, A(MOVE_ACTIONS + 1, "Rock Throw", out, names), _o()
    )
    back = assert_matches_reference(tables, away, A(MOVE_ACTIONS, "Rock Throw", away, names), _o())
    assert int(back.moves[0, 0, 0]) == rock
    assert int(away.sketch_move[0]) == 0  # UpdateAfterSwitch (battle_lib.c:2130)
    # The battle's end (the foe's last Pokemon faints) leaves the party with it.
    last = out._replace(hp=out.hp.at[1].set(jnp.asarray([1, 0, 0, 0, 0, 0], out.hp.dtype)))
    last = last._replace(species=last.species.at[1, 1:].set(0))
    done = assert_matches_reference(tables, last, A("Tackle", "Rock Throw", last, names), _o())
    assert int(done.over) == 1 and int(done.moves[0, 0, 0]) == rock


def test_sketch_is_kept_when_no_attack_message_shows(tables, names):
    """A foe that is fully paralyzed shows no message: ``movePrevByBattler`` becomes MOVE_NONE
    (bcp.c:3945-3949) but ``moveSketched`` keeps Rock Throw (:3952-3954)."""
    rock = names.move("Rock Throw")
    s = _sketcher(tables, names)
    s = s._replace(
        sketch_move=s.sketch_move.at[1].set(rock),
        last_move=s.last_move.at[1].set(rock),
        status=s.status.at[1, 0].set(Status.PARALYSIS),
    )
    para = _o()._replace(full_para=jnp.asarray([0, 1], jnp.int8))
    out = assert_matches_reference(tables, s, A("Sketch", "Rock Throw", s, names), para)
    assert int(out.last_move[1]) == 0 and int(out.sketch_move[1]) == rock
    assert int(out.moves[0, 0, 0]) == rock


def test_sketch_fails_on_none_forbidden_known_substitute_or_transformed(tables, names):
    """``BtlCmd_TrySketch``'s failures (battle_script.c:4847-4860, subscript_sketch.s:5, :15): the
    slot stays Sketch and only its PP is spent."""
    base = _sketcher(tables, names, order="player")
    cases = {"no move yet": base}
    for name in ("Struggle", "Sketch", "Chatter"):
        cases[name] = base._replace(sketch_move=base.sketch_move.at[1].set(names.move(name)))
    cases["known"] = base._replace(sketch_move=base.sketch_move.at[1].set(names.move("Tackle")))
    seen = names.move("Rock Throw")
    sub = base._replace(sketch_move=base.sketch_move.at[1].set(seen))
    cases["substitute"] = sub._replace(substitute_hp=sub.substitute_hp.at[1].set(20))
    cases["transformed"] = sub._replace(transformed=sub.transformed.at[0].set(1))
    for name, s in cases.items():
        out = assert_matches_reference(tables, s, A("Sketch", "Leer", s, names), _o())
        assert out.moves[0, 0].tolist() == s.moves[0, 0].tolist(), name
        assert int(out.pp[0, 0, 0]) == int(s.pp[0, 0, 0]) - 1, name


def test_a_sketched_last_resort_zeroes_the_count(tables, names):
    """battle_script.c:4876-4878."""
    s = _sketcher(tables, names, foe=("Last Resort", "Tackle"), order="player")
    s = s._replace(
        last_resort_count=s.last_resort_count.at[0].set(2),
        sketch_move=s.sketch_move.at[1].set(names.move("Last Resort")),
    )
    out = assert_matches_reference(tables, s, A("Sketch", "Tackle", s, names), _o())
    assert int(out.moves[0, 0, 0]) == names.move("Last Resort")
    assert int(out.last_resort_count[0]) == 0


def test_sleep_talk_can_call_sketch(tables, names):
    """Sleep Talk's pool excludes only the moves of its own list (battle_script.c:4916-4925), not
    Sketch: the called Sketch writes the same permanent copy."""
    s = _sketcher(tables, names, mine=("Sleep Talk", "Sketch", "Tackle", "Protect"), order="player")
    s = s._replace(
        status=s.status.at[0, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[0, 0].set(3),
        sketch_move=s.sketch_move.at[1].set(names.move("Rock Throw")),
    )
    # Rank 1 of Sleep Talk's eligible moves (the slots but its own and the empty ones) is Sketch's
    # neighbor in the list: draw every rank, and one of them must call Sketch.
    results = []
    for k in range(3):
        called = _o()._replace(call=jnp.asarray([k, 0], jnp.int16))
        out = assert_matches_reference(tables, s, A("Sleep Talk", "Leer", s, names), called)
        results.append(out.moves[0, 0].tolist())
    rock = names.move("Rock Throw")
    sketched = [r for r in results if rock in r]
    assert sketched, results
    for r in sketched:
        assert MOVE_SKETCH not in r  # the Sketch slot took the move


def test_switch_in_and_faint_clear_sketch_move(tables, names):
    """``UpdateAfterSwitch`` (battle_lib.c:2130) and ``CleanupFaintedMon`` (:2208): the volatile
    field leaves with the battler, not through Baton Pass."""
    from nuzline.engine.state import BATON_PASSED_FIELDS, VOLATILE_FIELDS

    assert "sketch_move" in VOLATILE_FIELDS and "sketch_move" not in BATON_PASSED_FIELDS
    s = _sketcher(tables, names, order="player")
    s = s._replace(sketch_move=s.sketch_move.at[1].set(names.move("Rock Throw")))
    out = assert_matches_reference(tables, s, A("Tackle", MOVE_ACTIONS + 1, s, names), _o())
    assert int(out.sketch_move[1]) == 0  # the foe switched out


def test_a_faint_clears_sketch_move(tables, names):
    """``BattleSystem_CleanupFaintedMon`` clears ``moveSketched`` (battle_lib.c:2208): the foe's
    last Pokemon faints with Rock Throw as its ``moveSketched``, no switch-in follows, and the
    field is 0 at the battle's end."""
    s = _sketcher(tables, names, order="player")
    s = s._replace(
        sketch_move=s.sketch_move.at[1].set(names.move("Rock Throw")),
        hp=s.hp.at[1].set(jnp.asarray([1, 0, 0, 0, 0, 0], s.hp.dtype)),
        species=s.species.at[1, 1:].set(0),
    )
    out = assert_matches_reference(tables, s, A("Tackle", "Rock Throw", s, names), _o())
    assert int(out.over) == 1 and int(out.hp[1, 0]) == 0
    assert int(out.sketch_move[1]) == 0


def test_a_shown_move_that_misses_is_sketched(tables, names):
    """subscript_missed.s:5 runs PrintAttackMessage, which sets SYSCTL_ATTACK_MESSAGE_SHOWN
    (battle_script.c:1432): UpdateMoveBuffers writes ``moveSketched`` (bcp.c:3952-3954) for a
    move that missed. The slower Sketch copies the missed Rock Throw."""
    s = _sketcher(tables, names)
    miss = _o()._replace(hit=_o().hit.at[1, 0].set(0))
    out = assert_matches_reference(tables, s, A("Sketch", "Rock Throw", s, names), miss)
    rock = names.move("Rock Throw")
    assert int(out.hp[0, 0]) == int(s.hp[0, 0])  # it missed
    assert int(out.sketch_move[1]) == rock and int(out.moves[0, 0, 0]) == rock


def test_the_choice_lock_on_sketch_is_released_next_turn(tables, names):
    """UpdateMoveBuffers locks a Choice holder into ``moveTemp`` = Sketch (bcp.c:3921-3930); Sketch
    then replaces itself, so CHECK_INVALID_CHOICE_ITEM finds no slot for the lock and clears it
    (battle_lib.c:2307-2309): next turn every slot is usable, and the move used locks anew."""
    s = _sketcher(tables, names)
    s = s._replace(
        item=s.item.at[0, 0].set(names.item("Choice Scarf")),
        stats=s.stats.at[1, 0, SPE].set(999),  # the foe stays first despite the Scarf
    )
    out = assert_matches_reference(tables, s, A("Sketch", "Rock Throw", s, names), _o())
    rock = names.move("Rock Throw")
    assert int(out.moves[0, 0, 0]) == rock and int(out.choice_lock[0]) == MOVE_SKETCH
    assert usable_moves(tables, out, 0).tolist() == [True] * 4
    nxt = assert_matches_reference(tables, out, A("Tackle", "Rock Throw", out, names), _o())
    assert int(nxt.choice_lock[0]) == names.move("Tackle")


def test_coverage_grades_the_moves_a_sketch_can_copy(tables, names):
    """``check_coverage`` runs every move of the opposing party a Sketch holder can copy
    (``sketch_calls``) through ``move_effect_line`` as a "Sketch move effect" line: with Rock
    Throw unmodelled, the Smeargle's line refuses it; a Pokemon without Sketch gets no such line."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_coverage, sketch_calls
    from tests.engine.helpers import roark_party, roark_trainer, sample_party, unmodelled

    opp = roark_party(names)
    smeargle = replace(
        sample_party(names)[0],
        species=names.species("Smeargle"),
        moves=(MOVE_SKETCH, names.move("Tackle"), 0, 0),
    )
    rock = names.move("Rock Throw")
    calls = sketch_calls(tables, list(smeargle.moves), opp)
    assert rock in calls and names.move("Tackle") not in calls
    assert not set(calls) & set(SKETCH_FORBIDDEN)
    ok = check_coverage(tables, names, [smeargle], roark_trainer(), opp, 0)
    assert not any("Sketch move effect" in m for m in ok.missing)
    broken = unmodelled(tables, rock)
    r = check_coverage(broken, names, [smeargle], roark_trainer(), opp, 0)
    assert any("Sketch move effect" in m and "SMEARGLE" in m.upper() for m in r.missing)
    plain = check_coverage(broken, names, sample_party(names)[:1], roark_trainer(), opp, 0)
    assert not any("Sketch move effect" in m for m in plain.missing)


# --- I-3 gate fix: a Sketch by an imprisoner shrinks the foe's Sleep Talk set ---------------


def _talker_under_sketch(tables, names, order):
    """Smeargle (Sketch, Imprison, Spore, Protect) imprisons at the turn start (it imprisoned
    the foe's previous Pokemon: the bit stays, battle_lib.c:3316-3338 reads only the bit and
    its live moves) against a sleeping Geodude (Sleep Talk, Tackle, Ember, Thunder) that shares
    no move with it. The Geodude's ``moveSketched`` is Ember and its ``movePrevByBattler``
    MOVE_NONE (an Ember that showed its message and failed, bcp.c:3944-3954), so no Disable
    hypothesis splits the set. ``order`` "player": the Smeargle moves first."""
    s = _sketcher(
        tables,
        names,
        foe=("Sleep Talk", "Tackle", "Ember", "Thunder"),
        order=order,
        mine=("Sketch", "Imprison", "Spore", "Protect"),
    )
    ember = names.move("Ember")
    return s._replace(
        status=s.status.at[1, 0].set(Status.SLEEP),
        status_counter=s.status_counter.at[1, 0].set(5),
        imprison=s.imprison.at[0].set(1),
        sketch_move=s.sketch_move.at[1].set(ember),
        last_move=s.last_move.at[1].set(0),
    )


@pytest.mark.slow  # pod e1-igf-a4fe340: 701 s, 46 GB peak
def test_a_faster_imprisoner_sketch_seals_its_copy_for_sleep_talk(tables, names):
    """The decomp's draw: the Smeargle's Sketch puts Ember in its first slot (battle_script.c:
    4868), Sleep Talk then flags every move an imprisoner knows (BtlCmd_TrySleepTalk, :4927;
    CHECK_INVALID_IMPRISONED, battle_lib.c:2287-2290, Move_Imprisoned's live ``moves``,
    :3316-3338) and redraws uniformly over the rest (:4931-4933): of the 12 call values, 6 call
    Tackle and 6 Thunder, none Ember. Engine == reference on each. A pin of the decomp's draw,
    not a regression test: ``step`` played ``call % count`` on the live set before the gate fix
    too. The regression test is ``test_the_call_law_follows_an_imprisoners_sketch[player]``."""
    from nuzline.engine.step import step_events

    s = _talker_under_sketch(tables, names, "player")
    acts = A("Sketch", "Sleep Talk", s, names)
    seen = []
    for u in range(12):
        o = _o()._replace(call=jnp.asarray([0, u], jnp.int16))
        out = assert_matches_reference(tables, s, acts, o)
        assert int(out.moves[0, 0, 0]) == names.move("Ember")
        seen.append(int(step_events(tables, s, acts, o)[1]["called"][1]))
    assert seen.count(names.move("Tackle")) == 6 and seen.count(names.move("Thunder")) == 6


@pytest.mark.slow  # pod e1-igf-a4fe340: 545 s + 374 s, 16 GB peak
@pytest.mark.parametrize("order", ["player", "opp"])
def test_the_call_law_follows_an_imprisoners_sketch(tables, names, order):
    """``enumerate_turn`` gives the decomp's weights: after a faster imprisoner's Sketch of Ember
    Tackle and Thunder 1/2 each (the ``CALL_SKETCHED`` variant, ``order.sketched_state``); before
    a slower one the turn-start set, 1/3 each. The turn-start law alone gave Tackle 2/3 in the
    first case. Each kept call value plays engine == reference."""
    import jax

    from nuzline.engine.enumerate import CALL_SKETCHED, enumerate_turn
    from nuzline.engine.step import turn_events
    from tests.engine.helpers import padded_rows

    s = _talker_under_sketch(tables, names, order)
    acts = A("Sketch", "Sleep Talk", s, names)
    outcomes, probs = enumerate_turn(tables, s, acts)
    assert abs(float(sum(probs)) - 1.0) < 1e-12
    program = jax.jit(jax.vmap(turn_events, in_axes=(None, None, None, 0)))
    called = np.asarray(padded_rows(program, tables, s, acts, outcomes)["called"])[:, 1]
    law = {int(m): float(probs[called == m].sum()) for m in np.unique(called)}
    moves = ("Tackle", "Thunder") if order == "player" else ("Tackle", "Ember", "Thunder")
    assert set(law) == {names.move(m) for m in moves}
    for m in moves:
        assert abs(law[names.move(m)] - 1 / len(moves)) < 1e-12, law
    calls = np.asarray(outcomes.call)[:, 1]
    assert np.any(calls >= CALL_SKETCHED) == (order == "player")
    for value in np.unique(calls):
        row = int(np.flatnonzero(calls == value)[0])
        assert_matches_reference(tables, s, acts, jax.tree.map(lambda f, r=row: f[r], outcomes))


def _slow_smeargle(names, *moves):
    """A level-5 Smeargle (Speed 13, below ``GYRO_SPEED_MIN``) that knows Sketch."""
    from dataclasses import replace

    from tests.engine.helpers import sample_party

    ids = [MOVE_SKETCH, *(names.move(m) for m in moves)]
    return replace(
        sample_party(names)[0],
        species=names.species("Smeargle"),
        level=5,
        moves=tuple(ids + [0] * (4 - len(ids))),
    )


@pytest.mark.parametrize("copier", ["Transform", "Mimic", None])
def test_a_sketch_can_copy_a_party_move_a_foe_copied(tables, names, copier):
    """A foe's Transform (battle_script.c:5346-5348) or Mimic (:4476-4524) puts a move of the
    player's party in its slot, so its ``moveSketched`` (bcp.c:3952-3954) can be a move the
    Smeargle does not know: ``sketch_calls`` lists the party's moves then, and the Smeargle's own
    Gyro Ball line (``gyro_speed_line``, Speed 13) is graded. Without a copier it is not."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_coverage, sketch_calls
    from tests.engine.helpers import roark_party, roark_trainer, sample_party

    gyro = names.move("Gyro Ball")
    smeargle = _slow_smeargle(names, "Tackle")
    mate = replace(sample_party(names)[1], moves=(gyro, 0, 0, 0))
    opp = roark_party(names)
    if copier is not None:
        opp[0] = replace(opp[0], moves=(names.move(copier), *opp[0].moves[1:]))
    calls = sketch_calls(tables, list(smeargle.moves), opp, [gyro])
    assert (gyro in calls) == (copier is not None)
    report = check_coverage(tables, names, [smeargle, mate], roark_trainer(), opp, 0)
    line = [m for m in report.missing if "SMEARGLE" in m.upper() and "Gyro Ball with Speed" in m]
    assert bool(line) == (copier is not None), report.missing


def test_a_sketched_metronome_calls_are_graded_for_the_holder(tables, names):
    """A foe's Metronome is a move a Sketch can copy; the copied Metronome then calls any move
    ``Move_CanBeMetronomed`` allows (battle_script.c:4535-4562) for the Smeargle, Gyro Ball
    included: its own Speed-13 Gyro line is graded (``fight_calls`` reads only known moves)."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_coverage, sketch_calls
    from tests.engine.helpers import roark_party, roark_trainer

    metronome = names.move("Metronome")
    smeargle = _slow_smeargle(names, "Tackle")
    opp = roark_party(names)
    opp[0] = replace(opp[0], moves=(metronome, *opp[0].moves[1:]))
    assert metronome in sketch_calls(tables, list(smeargle.moves), opp)
    report = check_coverage(tables, names, [smeargle], roark_trainer(), opp, 0)
    assert any("SMEARGLE" in m.upper() and "Gyro Ball with Speed" in m for m in report.missing), (
        report.missing
    )


@pytest.mark.parametrize("side", ["player", "opp"])
def test_a_known_metronomes_calls_reach_the_rom_change_check(tables, names, vanilla_manifest, side):
    """A known Metronome can call any move ``Move_CanBeMetronomed`` allows (BtlCmd_Metronome,
    battle_script.c:4535-4562), so a Kaizo ROM-change condition on such a move refuses the fight
    (``rom_change_refusals`` through ``reached_moves``), on either side, as a gained one's calls
    do; without the Metronome nothing reaches it. Gyro Ball is a call (no NO_METRONOME flag)."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_opponent, reached_moves, rom_change_refusals
    from tests.engine.helpers import kaizo_names_with, roark_party, roark_trainer

    gyro, metronome = names.move("Gyro Ball"), names.move("Metronome")
    assert gyro in reached_moves(tables, [metronome]) and gyro not in reached_moves(tables, [1])
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [gyro]}])
    roark, plain = roark_trainer(), roark_party(names)
    holder = replace(plain[0], moves=(metronome, *plain[0].moves[1:]))
    with_holder = [holder, *plain[1:]]
    assert rom_change_refusals(tables, kaizo, roark, plain) == []
    assert rom_change_refusals(tables, kaizo, roark, with_holder) == ["c"]
    opp, player = (with_holder, ()) if side == "opp" else (plain, [holder])
    report = check_opponent(tables, kaizo, roark, opp, 0, player)
    assert any(m.startswith("Kaizo ROM: c is not modelled") for m in report.missing), report.missing


def test_a_metronome_reads_the_terrain_through_its_calls(tables, names):
    """Metronome can call Secret Power (no NO_METRONOME flag), whose script reads the terrain
    (``TERRAIN_EFFECTS``): ``terrain_refusal`` refuses a Metronome alone, as prep asks it per
    move, against a trainer of unknown terrain, and names Secret Power."""
    from dataclasses import replace

    from nuzline.engine.coverage import TERRAIN_UNKNOWN, terrain_refusal
    from tests.engine.helpers import roark_trainer

    unplaced = replace(roark_trainer(), terrain=TERRAIN_UNKNOWN, terrains=())
    line = terrain_refusal(tables, names, unplaced, [names.move("Metronome")])
    assert line is not None and "Secret Power" in line
    assert terrain_refusal(tables, names, unplaced, [names.move("Tackle")]) is None
    assert terrain_refusal(tables, names, roark_trainer(), [names.move("Metronome")]) is None


def test_a_gained_metronomes_calls_stay_on_the_holders_lines(tables, names, monkeypatch):
    """A sketched Metronome's calls are the holder's: the fight's terrain check reads them
    (``extra_calls``), but the AI's Mimic and Transform pool (``foe_moves``) does not. Mimic
    copies ``movePrevByBattler`` (battle_script.c:4481), the Metronome itself (bcp.c:3945-3946),
    which Move_CanBeMimicked refuses (battle_lib.c:7182-7191). Only a called lock move, its
    next turn's ``moveTemp`` (bcp.c:1885-1888), joins that pool."""
    from dataclasses import replace

    from nuzline.engine import coverage
    from tests.engine.helpers import roark_party, roark_trainer

    seen: dict[str, list[int]] = {}
    terrain = coverage.terrain_refusal
    pool = coverage.foe_moves

    def _terrain(tables_, names_, trainer, moves):
        seen["terrain"] = list(moves)
        return terrain(tables_, names_, trainer, moves)

    def _pool(tables_, player, opp_, extra=()):
        seen["pool"] = list(extra)
        return pool(tables_, player, opp_, extra)

    monkeypatch.setattr(coverage, "terrain_refusal", _terrain)
    monkeypatch.setattr(coverage, "foe_moves", _pool)
    smeargle = _slow_smeargle(names, "Tackle")
    gyro, outrage = names.move("Gyro Ball"), names.move("Outrage")
    plain = roark_party(names)
    for calls in ((), (gyro,)):
        coverage.check_opponent(tables, names, roark_trainer(), plain, 0, [smeargle], (), calls)
        assert (gyro in seen["terrain"]) == bool(calls)
    opp = list(plain)
    opp[0] = replace(opp[0], moves=(names.move("Metronome"), names.move("Mimic"), 0, 0))
    coverage.check_coverage(tables, names, [smeargle], roark_trainer(), opp, 0)
    assert gyro not in seen["pool"] and outrage in seen["pool"]


@pytest.mark.parametrize("transform", [True, False])
def test_a_sketch_can_copy_a_party_move_a_foe_copycat_transformed_into(tables, names, transform):
    """A foe's Copycat calls ``movePrev`` when ``Move_CanBeMetronomed`` allows it
    (battle_script.c:7333-7336), Transform included (sCannotMetronomeMoves, battle_lib.c:
    7152-7177): after the player's Transform it becomes the player's active and carries a party
    move in its slot, so ``sketch_calls`` lists the party's moves and the Smeargle's own Gyro
    Ball line is graded. Without a Transform in the player's party Copycat copies none."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_coverage, sketch_calls
    from tests.engine.helpers import roark_party, roark_trainer, sample_party

    gyro = names.move("Gyro Ball")
    smeargle = _slow_smeargle(names, "Tackle")
    mate_moves = (gyro, names.move("Transform") if transform else 0, 0, 0)
    mate = replace(sample_party(names)[1], moves=mate_moves)
    opp = roark_party(names)
    opp[0] = replace(opp[0], moves=(names.move("Copycat"), *opp[0].moves[1:]))
    calls = sketch_calls(tables, list(smeargle.moves), opp, [m for m in mate_moves if m])
    assert (gyro in calls) == transform
    report = check_coverage(tables, names, [smeargle, mate], roark_trainer(), opp, 0)
    line = [m for m in report.missing if "SMEARGLE" in m.upper() and "Gyro Ball with Speed" in m]
    assert bool(line) == transform, report.missing


@pytest.mark.parametrize("invoker", ["Metronome", "Copycat", "Mirror Move", "Me First", None])
def test_a_sketch_can_copy_a_lock_move_a_foe_called(tables, names, invoker):
    """A called lock or charge move (Outrage: subscript_thrash.s:8; Fly: subscript_vanish_on_
    charge_turn.s:10) is ``moveLockedInto = moveCur``, the called move (battle_lib.c:1831,
    battle_script.c:2118); the next turn ``moveTemp`` is it (bcp.c:1885-1888) and so the foe's
    ``moveSketched`` (:3952-3954). ``sketch_calls`` lists every lock move for a foe with an
    invoker that calls a move it does not know, none without one (``locked_calls``)."""
    from dataclasses import replace

    from nuzline.engine.coverage import locked_calls, sketch_calls
    from tests.engine.helpers import roark_party

    smeargle = _slow_smeargle(names, "Tackle")
    opp = roark_party(names)
    if invoker is not None:
        opp[0] = replace(opp[0], moves=(names.move(invoker), *opp[0].moves[1:]))
    calls = sketch_calls(tables, list(smeargle.moves), opp)
    locks = {names.move(m) for m in ("Outrage", "Fly", "Rollout", "Bide", "Uproar")}
    assert locks <= set(locked_calls(tables, [names.move("Metronome")]))
    assert names.move("Hyper Beam") not in locked_calls(tables, [names.move("Metronome")])
    assert (locks <= set(calls)) == (invoker is not None)
    if invoker is None:
        assert not locks & set(calls)


def test_a_metronome_called_lock_move_is_graded_for_the_sketch_holder(tables, names):
    """A foe's Metronome-called Outrage is a move its next turn's Sketch can copy: with Outrage
    unmodelled the Smeargle gets its "Sketch move effect" line; against a party with no invoker
    it gets none."""
    from dataclasses import replace

    from nuzline.engine.coverage import check_coverage
    from tests.engine.helpers import roark_party, roark_trainer, unmodelled

    outrage = names.move("Outrage")
    broken = unmodelled(tables, outrage)
    smeargle = _slow_smeargle(names, "Tackle")
    plain = roark_party(names)
    opp = list(plain)
    opp[0] = replace(opp[0], moves=(names.move("Metronome"), *opp[0].moves[1:]))
    for party, want in ((opp, True), (plain, False)):
        report = check_coverage(broken, names, [smeargle], roark_trainer(), party, 0)
        line = [
            m
            for m in report.missing
            if "Sketch move effect" in m and "SMEARGLE" in m.upper() and "Outrage" in m
        ]
        assert bool(line) == want, report.missing


def test_a_reflected_move_is_the_movers_sketch_move(tables, names):
    """A Magic Coat reflection: LoopSpreadMoves swaps the attacker back to the mover
    (``magicCoatMon``, bcp.c:3831-3835) before UpdateMoveBuffers writes moveSketched (:3952-3954),
    so the mover's is its reflected Leer and the holder's stays its own Magic Coat. The next
    turn's Sketch copies the Leer. Engine == reference."""
    s = _sketcher(
        tables,
        names,
        foe=("Leer",),
        order="player",
        mine=("Magic Coat", "Sketch", "Tackle", "Protect"),
    )
    out = assert_matches_reference(tables, s, A("Magic Coat", "Leer", s, names), _o())
    leer = names.move("Leer")
    assert int(out.sketch_move[1]) == leer
    assert int(out.sketch_move[0]) == names.move("Magic Coat")
    nxt = assert_matches_reference(tables, out, A("Sketch", "Leer", out, names), _o())
    assert leer in nxt.moves[0, 0].tolist()


def test_a_sketched_slot_keeps_its_pp_ups(tables, names):
    """TrySketch writes the slot's move and ``ppCur`` = table PP (battle_script.c:4868-4870), not
    its ``ppUps``. The slot's max PP (MoveTable_CalcMaxPP, move_table.c:25-34) is the copy's table
    PP with the slot's PP Ups, none on a Sketch slot (PP Up needs a max PP of 5 or more,
    item_use_pokemon.c:20, :187). Engine == reference."""
    from nuzline.engine.hooks import max_pp

    s = _sketcher(tables, names)
    out = assert_matches_reference(tables, s, A("Sketch", "Rock Throw", s, names), _o())
    rock = names.move("Rock Throw")
    assert out.pp_ups[0, 0].tolist() == s.pp_ups[0, 0].tolist()
    assert int(max_pp(tables, out, 0, 0)[0]) == int(tables.move_pp[rock]) == int(out.pp[0, 0, 0])
