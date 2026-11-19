"""Roar and Whirlwind (K3b2 Task 11, ``EffectId.ROAR``; docs/kaizo/k3b2-rules.md §6).

Decomp (external/pokeplatinum):

* ``res/battle/scripts/effects/effect_script_0028.s``: ``SIDE_EFFECT_FLAGS_DIRECT =
  ON_HIT | FORCE_TARGET_TO_SWITCH_OR_FLEE``. ``BattleSystem_TriggerPrimaryEffect``
  (src/battle/battle_lib.c:1481-1510) runs an ON_HIT effect only without
  ``MOVE_STATUS_NO_EFFECTS`` (a miss, Protect).
* ``res/battle/scripts/subscripts/subscript_force_target_to_switch_or_flee.s``: semi-invulnerable,
  Suction Cups, Ingrain, the doubles-AI battle type, ``TryWhirlwind`` (fail: ``_074``,
  ``MOVE_STATUS_FAILED``), the attack message, ``TryRestoreStatusOnSwitch BTLSCR_DEFENDER``
  (Natural Cure), ``DeletePokemon``, ``SwitchAndUpdateMon BTLSCR_FORCED_OUT``, "was dragged out!",
  ``BATTLE_SUBSCRIPT_HAZARDS_CHECK``. No Pursuit, no faint call.
* ``BtlCmd_TryWhirlwind`` (src/battle/battle_script.c:5241-5320): eligible = species, not an
  egg, HP > 0; ``eligibleMons <= maxActiveMons`` fails before any draw; then
  ``BattleSystem_CanWhirlwind`` (battle_lib.c:3072-3086); then ``RandNext() % max`` until an
  eligible slot other than the active one.
* ``BattleSystem_UpdateAfterSwitch`` (battle_lib.c:2040-2155): the newcomer's action becomes
  MOVE_END (:2053), volatiles and move effects of the slot are cleared, ATTRACT and Bind sourced
  from the outgoing battler end, ``fakeOutTurnNumber = totalTurns + 1``, moveHit and
  movePrevByBattler NONE, the AI forgets the battler's moves, ability and item.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace as replace_spec
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.engine.coverage import check_coverage
from nuzline.engine.enumerate import (
    _LAWS,
    _READ_BUILDERS,
    INPUT_CLAUSE_EFFECTS,
    NON_TABLE_DAMAGE_EFFECTS,
    damage_inputs_can_change,
    enumerate_drawn,
)
from nuzline.engine.ids import AbilityId, EffectId, MoveFlag, Stat, Status, Type, Weather
from nuzline.engine.observe import OBS_FIELDS, observe
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import (
    MAX_HITS,
    Outcome,
    empty_outcome,
    hit_threshold,
    roar_level_passes,
    roar_level_weight,
    sample_outcome,
)
from nuzline.engine.state import make_state
from nuzline.engine.step import _VOLATILE, needs_replacement, replace, step, step_events
from nuzline.search.plan_tree import label
from tests.engine.helpers import (
    A,
    assert_matches_reference,
    roark_party,
    roark_state,
    roark_trainer,
    sample_party,
)
from tests.engine.test_enumerate import (
    _kaizo_spec,
    assert_equals_bruteforce,
    check_frequencies,
    children,
    laws_of,
    sampled,
)
from tools.effect_map import DECOMP_EFFECT_NAMES, DECOMP_EFFECT_TO_ID

DECOMP = Path("external/pokeplatinum")
_SPE = 4  # stats[..., 0..4] are Atk..Spe
SWITCH = MOVE_ACTIONS + 1  # to party slot 1
ROAR, WHIRLWIND = 46, 18  # decomp move ids


def _state(tables, names, *, mine="Roar", opp="Tackle", order="opp", opp_count=3):
    """The sample party's lead (L14; ``mine`` first) against Geodude (L12; ``opp`` first,
    then the fixture moves), Onix (L12) and Cranidos (L14). ``order``: who is faster."""
    s = roark_state(tables, names, player_moves=[mine], opp_moves=[opp], opp_count=opp_count)
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    return s._replace(stats=s.stats.at[1, 0, _SPE].set(foe))


def _o(level=(1, 1), target=(0, 0), **fields) -> Outcome:
    o = empty_outcome()._replace(
        roar_level=jnp.asarray(level, jnp.int8), roar_target=jnp.asarray(target, jnp.int8)
    )
    return o._replace(**{k: jnp.asarray(v, jnp.int8) for k, v in fields.items()})


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


def _level(s, side, slot, level):
    return s._replace(level=s.level.at[side, slot].set(level))


def _faint(s, side, *slots):
    hp = s.hp
    for slot in slots:
        hp = hp.at[side, slot].set(0)
    return s._replace(hp=hp)


# --- mapping, data and decomp ----------------------------------------------------------------


@pytest.mark.parametrize("game", ["vanilla", "kaizo"])
def test_effect_28_maps_roar_and_whirlwind(game, tables, kaizo_tables):
    t = tables if game == "vanilla" else kaizo_tables
    assert DECOMP_EFFECT_NAMES[28] == "BATTLE_EFFECT_FORCE_SWITCH"
    assert DECOMP_EFFECT_TO_ID[28] == (EffectId.ROAR, (0, 0, 0))
    for move in (ROAR, WHIRLWIND):
        assert int(t.move_effect[move]) == EffectId.ROAR
        assert int(t.move_accuracy[move]) == 100  # rolls: evasion and Sand Veil count
        assert int(t.move_priority[move]) == -6
        assert int(t.move_flags[move]) & MoveFlag.PROTECT
    assert int(t.move_flags[ROAR]) & MoveFlag.SOUND  # sSoundMoves (battle_lib.c:3473-3475)
    assert not int(t.move_flags[WHIRLWIND]) & MoveFlag.SOUND


def _function(source: str, name: str) -> str:
    # The definition, not the prototype: the signature line followed by its brace.
    pattern = re.escape(name) + r"\(BattleSystem \*battleSys[^;{]*\)\n\{"
    start = re.search(pattern, source).end() - 1
    return source[start : source.index("\n}\n", start)]


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_match_the_engine():
    scripts = DECOMP / "res" / "battle" / "scripts"
    effect = (scripts / "effects" / "effect_script_0028.s").read_text()
    assert "MOVE_SIDE_EFFECT_ON_HIT|MOVE_SUBSCRIPT_PTR_FORCE_TARGET_TO_SWITCH_OR_FLEE" in effect
    sub = (scripts / "subscripts" / "subscript_force_target_to_switch_or_flee.s").read_text()
    steps = [
        "MOVE_STATUS_SEMI_INVULNERABLE",
        "ABILITY_SUCTION_CUPS",
        "MOVE_EFFECT_INGRAIN",
        "TryWhirlwind _074",
        "TryRestoreStatusOnSwitch BTLSCR_DEFENDER",
        "SwitchAndUpdateMon BTLSCR_FORCED_OUT",
        "BATTLE_SUBSCRIPT_HAZARDS_CHECK",
    ]
    where = [sub.index(step_) for step_ in steps]
    assert where == sorted(where)
    assert "PURSUIT" not in sub and "FAINT" not in sub
    assert (
        "_074:\n    UpdateVar OPCODE_FLAG_ON, BTLVAR_MOVE_STATUS_FLAGS, MOVE_STATUS_FAILED" in sub
    )

    script_c = (DECOMP / "src" / "battle" / "battle_script.c").read_text()
    whirl = _function(script_c, "static BOOL BtlCmd_TryWhirlwind")
    order = [
        whirl.index("eligibleMons <= maxActiveMons"),
        whirl.index("BattleSystem_CanWhirlwind(battleSys, battleCtx)"),
        whirl.index("BattleSystem_RandNext(battleSys) % max"),
    ]
    assert order == sorted(order)
    assert "i == selectedSlot1 || i == selectedSlot2" in whirl
    assert "MON_DATA_HP, NULL) == 0" in whirl
    lib = (DECOMP / "src" / "battle" / "battle_lib.c").read_text()
    can = _function(lib, "BOOL BattleSystem_CanWhirlwind")
    assert "ATTACKING_MON.level >= DEFENDING_MON.level" in can
    flat = re.sub(r"\s+", " ", can)
    assert (
        "(((BattleSystem_RandNext(battleSys) & 0xFF) * (ATTACKING_MON.level + DEFENDING_MON.level))"
        " >> 8) + 1" in flat
    )
    assert "tmp > battleCtx->battleMons[battleCtx->defender].level / 4" in flat
    after = _function(lib, "void BattleSystem_UpdateAfterSwitch")
    assert (
        "battleCtx->battlerActions[battler][BATTLE_ACTION_PICK_COMMAND] = BATTLE_CONTROL_MOVE_END"
        in after
    )
    sounds = lib[lib.index("sSoundMoves") : lib.index("};", lib.index("sSoundMoves"))]
    assert "MOVE_ROAR" in sounds and "MOVE_WHIRLWIND" not in sounds


# --- the level check -------------------------------------------------------------------------


def test_level_law_is_exact():
    for la in range(1, 101, 3):
        for ld in range(1, 101, 7):
            want = sum(((r * (la + ld)) >> 8) + 1 > ld // 4 for r in range(256))
            assert int(roar_level_weight(la, ld)) == want, (la, ld)
    # Vulpix L16 against Roark's L13-16 (rules §6(a)) never draws; a L5 user against L12.
    assert int(roar_level_weight(5, 12)) == 210  # r * 17 >> 8 >= 3: r >= 46
    assert bool(roar_level_passes(46, 5, 12)) and not bool(roar_level_passes(45, 5, 12))


# --- behaviour (both engines) ----------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("rank,slot", [(0, 1), (1, 2)])
def test_roar_drags_a_bench_pokemon_in(tables, names, rank, slot):
    s = _state(tables, names)
    acts = A("Roar", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o(target=(rank, 0)))
    assert int(out.active[1]) == slot
    ev = _events(tables, s, acts, _o(target=(rank, 0)))
    assert ev["dragged"].tolist() == [0, slot + 1]
    assert ev["act_id"][0] == ROAR
    assert int(out.move_hit[1]) == ROAR  # UpdateFlagsWhenHit after the drag (bcp.c:4499)
    assert int(out.fake_out_turn[1]) == int(s.turn) + 1  # UpdateAfterSwitch (lib.c:2116)
    assert int(out.last_move[0]) == ROAR


def test_the_pick_skips_fainted_slots(tables, names):
    s = _faint(_state(tables, names), 1, 1)  # Onix fainted: only Cranidos is eligible
    acts = A("Roar", "Tackle", s, names)
    for rank in (0, 3):  # a forced pick; a rank past the bench is clipped
        assert int(assert_matches_reference(tables, s, acts, _o(target=(rank, 0))).active[1]) == 2


@pytest.mark.parametrize("opp_count,fainted", [(1, ()), (3, (1, 2))])
def test_roar_fails_without_an_eligible_bench(tables, names, opp_count, fainted):
    s = _faint(_state(tables, names, opp_count=opp_count), 1, *fainted)
    s = _level(s, 0, 0, 5)  # a lower level: still no draw (TryWhirlwind fails first)
    acts = A("Roar", "Tackle", s, names)
    outs = [assert_matches_reference(tables, s, acts, _o(level=(lv, 0))) for lv in (0, 1)]
    for out in outs:
        assert int(out.active[1]) == 0
    assert _events(tables, s, acts, _o())["dragged"].tolist() == [0, 0]
    assert all(np.array_equal(a, b) for a, b in zip(outs[0], outs[1]))


def test_the_level_check(tables, names):
    s = _level(_state(tables, names), 0, 0, 5)  # L5 against Geodude L12
    acts = A("Roar", "Tackle", s, names)
    assert int(assert_matches_reference(tables, s, acts, _o(level=(0, 0))).active[1]) == 0
    assert int(assert_matches_reference(tables, s, acts, _o(level=(1, 0))).active[1]) == 1
    # At an equal or higher level the check draws nothing: roar_level 0 still drags.
    for lv in (12, 30):
        s = _level(_state(tables, names), 0, 0, lv)
        assert int(assert_matches_reference(tables, s, acts, _o(level=(0, 0))).active[1]) == 1


def test_a_miss_and_protect_stop_it(tables, names):
    s = _state(tables, names)
    acts = A("Roar", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o(hit=[[0] * MAX_HITS, [1] * MAX_HITS]))
    assert int(out.active[1]) == 0
    assert _events(tables, s, acts, _o(hit=[[0] * MAX_HITS, [1] * MAX_HITS]))[
        "missed"
    ].tolist() == [1, 0]
    # Accuracy 100 still rolls: Sand Veil in sand and the target's evasion count.
    sand = s._replace(
        weather=jnp.asarray(Weather.SANDSTORM, s.weather.dtype),
        ability=s.ability.at[1, 0].set(AbilityId.SAND_VEIL),
    )
    assert int(hit_threshold(tables, sand, acts)[0]) == 80
    evasive = s._replace(stages=s.stages.at[1, Stat.EVA].set(1))
    assert int(hit_threshold(tables, evasive, acts)[0]) == 75
    s = _state(tables, names, opp="Protect")
    acts = A("Roar", "Protect", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    assert int(out.active[1]) == 0
    assert _events(tables, s, acts, _o())["protected"].tolist() == [1, 0]


def test_the_outgoing_pokemon_leaves_everything_behind(tables, names):
    s = _state(tables, names)
    t = int(s.active[1])
    s = s._replace(
        stages=s.stages.at[1, Stat.DEF].set(2).at[1, Stat.EVA].set(-1),
        confusion=s.confusion.at[1].set(3),
        taunt=s.taunt.at[1].set(2),
        disabled_move=s.disabled_move.at[1].set(names.move("Leer")),
        disable_turns=s.disable_turns.at[1].set(2),
        leech_seed=s.leech_seed.at[1].set(1),
        trapped=s.trapped.at[1].set(3).at[0].set(4),  # Bind both ways
        trap_move=s.trap_move.at[1].set(20).at[0].set(20),
        infatuated=s.infatuated.at[0].set(1).at[1].set(1),  # in love both ways
        types=s.types.at[1, t].set(jnp.asarray([Type.NORMAL, Type.NORMAL], s.types.dtype)),
        status=s.status.at[1, t].set(Status.TOXIC),
        status_counter=s.status_counter.at[1, t].set(4),
    )
    acts = A("Roar", "Tackle", s, names)
    o = _o(full_para=(0, 0), infatuation=(0, 0), confusion_hit=(0, 0))
    out = assert_matches_reference(tables, s, acts, o)
    assert int(out.active[1]) == 1
    for name in _VOLATILE:  # the newcomer's slot starts clean; moveHit is Roar (below)
        if name != "move_hit":
            assert int(np.asarray(getattr(out, name))[1]) == 0, name
    assert int(out.move_hit[1]) == ROAR
    assert np.asarray(out.stages[1]).tolist() == [0] * 8
    # Sourced from the outgoing Geodude: the Roar user's Bind and love end (lib.c:2087-2096).
    assert int(out.trapped[0]) == 0 and int(out.infatuated[0]) == 0
    # Camouflage types restored (InitBattleMon re-reads the party), the toxic counter lost.
    assert (
        np.asarray(out.types[1, t]).tolist()
        == np.asarray(
            tables.species_types[int(tables.species_form_row[int(s.species[1, t]), 0])]
        ).tolist()
    )
    assert int(out.status[1, t]) == Status.TOXIC and int(out.status_counter[1, t]) == 0


def test_natural_cure_on_the_dragged_out_pokemon(tables, names):
    s = _state(tables, names)
    s = s._replace(
        ability=s.ability.at[1, 0].set(AbilityId.NATURAL_CURE),
        status=s.status.at[1, 0].set(Status.BURN),
    )
    out = assert_matches_reference(tables, s, A("Roar", "Tackle", s, names), _o())
    assert int(out.status[1, 0]) == Status.NONE and int(out.active[1]) == 1


def test_the_dragged_in_pokemon_does_not_act(tables, names):
    # Roar against Roar (both -6): the faster one drags; the newcomer's action is MOVE_END.
    s = _state(tables, names, opp="Roar", order="player")
    acts = A("Roar", "Roar", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    assert int(out.active[1]) == 1 and int(out.active[0]) == 0
    assert np.array_equal(np.asarray(out.pp[1]), np.asarray(s.pp[1]))
    ev = _events(tables, s, acts, _o())
    assert ev["act_kind"][1] == 0 and ev["shown"][1] == 0
    assert int(out.move_hit[1]) == ROAR  # kept: the skipped battler's MoveEnd clears nothing


def test_quick_claw_puts_a_roar_first_only_among_minus_six(tables, names):
    s = _state(tables, names, opp="Roar", order="opp")
    claw = names.item("Quick Claw")
    s = s._replace(item=s.item.at[0, 0].set(claw), qc_roll=s.qc_roll.at[0].set(1))
    out = assert_matches_reference(tables, s, A("Roar", "Roar", s, names), _o())
    assert int(out.active[1]) == 1 and int(out.active[0]) == 0  # the slower holder first
    # Against a priority-0 move the claw does not help: priority is compared first
    # (battle_lib.c:1401): Tackle hits, then Roar.
    s2 = _state(tables, names, order="opp")._replace(item=s.item, qc_roll=s.qc_roll)
    ev = _events(tables, s2, A("Roar", "Tackle", s2, names), _o())
    assert int(ev["order"][0]) == 1


def test_stealth_rock_and_intimidate_on_the_newcomer(tables, names):
    s = _state(tables, names)
    s = s._replace(
        stealth_rock=s.stealth_rock.at[1].set(1),
        ability=s.ability.at[1, 1].set(AbilityId.INTIMIDATE),
    )
    acts = A("Roar", "Tackle", s, names)
    out = assert_matches_reference(tables, s, acts, _o())
    assert int(out.active[1]) == 1
    assert int(out.hp[1, 1]) < int(s.hp[1, 1])  # Stealth Rock
    assert int(out.stages[0, Stat.ATK]) == -1  # Intimidate on the Roar user


def test_a_stealth_rock_faint_on_the_newcomer(tables, names):
    s = _state(tables, names)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1), hp=s.hp.at[1, 1].set(1))
    out = assert_matches_reference(tables, s, A("Roar", "Tackle", s, names), _o())
    assert int(out.active[1]) == 1 and int(out.hp[1, 1]) == 0
    assert int(needs_replacement(out)[1]) == 1 and int(out.over) == 0
    assert int(out.move_hit[1]) == ROAR  # written after the faint loop (bcp.c:3820-3837)


def test_a_dragged_in_quick_claw_holder_keeps_its_draw(tables, names):
    s = _state(tables, names)
    s = s._replace(item=s.item.at[1, 1].set(names.item("Quick Claw")))
    out = assert_matches_reference(tables, s, A("Roar", "Tackle", s, names), _o(quick_claw=(0, 1)))
    assert int(out.active[1]) == 1 and np.asarray(out.qc_roll).tolist() == [0, 1]


def test_the_ai_forgets_a_dragged_in_player_pokemon(tables, names):
    s = _state(tables, names, mine="Tackle", opp="Roar")
    s = s._replace(
        ai_seen_moves=s.ai_seen_moves.at[1].set(jnp.asarray([10, 20, 30, 40], jnp.int16)),
        ai_ability_known=s.ai_ability_known.at[1].set(1),
    )
    out = assert_matches_reference(tables, s, A("Tackle", "Roar", s, names), _o())
    assert int(out.active[0]) == 1
    assert np.asarray(out.ai_seen_moves[1]).tolist() == [0] * 4
    assert int(out.ai_ability_known[1]) == 0


@pytest.mark.slow
def test_observation_and_label(tables, names):
    s = _state(tables, names)
    acts = A("Roar", "Tackle", s, names)
    o = _o(target=(1, 0))
    after = step(tables, s, acts, o)
    obs = np.asarray(observe(tables, s, acts, o, after))
    sl = dict(OBS_FIELDS)["dragged"]
    assert obs[sl].tolist() == [0, 3]
    text = label(obs.tobytes(), names)
    assert "you used Roar" in text and "was dragged out" in text


def _sp(names, s, side, slot):
    return names.species_names[int(s.species[side, slot])]


@pytest.mark.slow
def test_label_names_the_actor_and_tells_a_drag_from_a_send_out(tables, names):
    # Geodude Tackles, then the player's Roar drags Cranidos in (review round, minor 2): the
    # label names Geodude as the actor and says nothing of a send-out.
    s = _state(tables, names)
    acts = A("Roar", "Tackle", s, names)
    o = _o(target=(1, 0))
    after = step(tables, s, acts, o)
    obs = np.asarray(observe(tables, s, acts, o, after)).tobytes()
    geodude, cranidos = _sp(names, s, 1, 0), _sp(names, s, 1, 2)
    text = label(obs, names, prev_opp=0, prev_mine=0)
    assert text.startswith(f"{geodude} used Tackle, you used Roar, {cranidos} was dragged out")
    assert "sent out" not in text
    assert label(obs, names).startswith("the foe used Tackle")  # no prev_opp: no name
    # The opponent's Roar drags the player's slot 1 in: no "you sent in" either.
    s = _state(tables, names, mine="Tackle", opp="Roar", order="player")
    acts = A("Tackle", "Roar", s, names)
    o = _o()
    after = step(tables, s, acts, o)
    obs = np.asarray(observe(tables, s, acts, o, after)).tobytes()
    text = label(obs, names, prev_opp=0, prev_mine=0)
    assert f"{_sp(names, s, 0, 1)} was dragged out" in text and "sent in" not in text
    # A Stealth Rock faint of the dragged-in Pokemon, then the replacement.
    s = _state(tables, names)
    s = s._replace(stealth_rock=s.stealth_rock.at[1].set(1), hp=s.hp.at[1, 1].set(1))
    acts = A("Roar", "Tackle", s, names)
    o = _o()
    after = step(tables, s, acts, o)
    shown = replace(tables, after, 1, 2)
    obs = np.asarray(observe(tables, s, acts, o, shown)).tobytes()
    text = label(obs, names, prev_opp=0, prev_mine=0)
    onix = _sp(names, s, 1, 1)
    assert f"{onix} was dragged out" in text and f"{onix} fainted" in text
    assert f"sent out {cranidos}" in text


def test_reference_agreement_over_draws(tables, names):
    setups = []
    for order in ("player", "opp", "tie"):
        for mine, opp in (("Roar", "Tackle"), ("Roar", "Roar"), ("Tackle", "Roar")):
            setups.append(_state(tables, names, mine=mine, opp=opp, order=order))
    for s in list(setups):
        setups.append(_level(_level(s, 0, 0, 5), 1, 0, 5))
        setups.append(_faint(s, 1, 1))
    acts = jnp.zeros(2, jnp.int32)  # each lead's slot 0
    for s in setups:
        for lv in ((0, 0), (1, 1), (0, 1)):
            for tg in ((0, 0), (1, 4), (4, 1)):
                for tie in (0, 1):
                    assert_matches_reference(tables, s, acts, _o(lv, tg, speed_tie=tie))


# --- relevance and laws ----------------------------------------------------------------------


def _rel(tables, s, acts, loose=False):
    laws = laws_of(tables, s, acts, loose=loose)
    return (
        np.asarray(laws.relevant.roar_level).tolist(),
        np.asarray(laws.relevant.roar_target).tolist(),
    )


@pytest.mark.parametrize("bench", [0, 1, 2, 3])
@pytest.mark.parametrize("level", [10, 14, 20])  # the Roar user's, against the player's L14
def test_relevance_by_level_and_bench(tables, names, bench, level):
    # Geodude Roars the player, whose bench is cut to ``bench`` living Pokemon.
    s = _state(tables, names, mine="Tackle", opp="Roar")
    s = _faint(_level(s, 1, 0, level), 0, *range(1 + bench, 6))
    acts = A("Tackle", "Roar", s, names)
    level_rel, target_rel = _rel(tables, s, acts)
    assert level_rel == [False, level < 14 and bench >= 1]
    assert target_rel == [False, bench >= 2]
    assert _rel(tables, s, acts, loose=True) == ([False, True], [False, True])
    laws = laws_of(tables, s, acts)
    assert int(laws.roar_bench[1]) == bench
    assert _LAWS["roar_target"](laws, 1)[0] == list(range(max(bench, 1)))


def test_target_switching_in_sets_the_level_and_the_bench(tables, names):
    # Geodude (L12) switches to Cranidos (L14) first: a L13 user's Roar checks against L14.
    s = _level(_state(tables, names), 0, 0, 13)
    acts = A("Roar", MOVE_ACTIONS + 2, s, names)
    laws = laws_of(tables, s, acts)
    assert np.asarray(laws.relevant.roar_level).tolist() == [True, False]
    assert int(laws.roar_level_weight[0]) == int(roar_level_weight(13, 14))
    assert int(laws.roar_bench[0]) == 2  # Geodude and Onix


def test_marginals_match_the_decomp(tables, names):
    s = _level(_state(tables, names), 0, 0, 5)
    acts = A("Roar", "Tackle", s, names)
    outcomes, probs, drawn = enumerate_drawn(laws_of(tables, s, acts))
    lv, tg = np.asarray(outcomes.roar_level)[:, 0], np.asarray(outcomes.roar_target)[:, 0]
    dl, dt = np.asarray(drawn.roar_level)[:, 0], np.asarray(drawn.roar_target)[:, 0]
    hit = Fraction(100, 100)  # Roar is 100 % against a stage-0 Geodude out of sand
    p_pass = Fraction(210, 256)
    assert math.fsum(probs[dl & (lv == 1)]) == pytest.approx(float(hit * p_pass), abs=1e-15)
    # The pick is drawn only after a passing level check.
    assert (dl[dt]).all() and (lv[dt] == 1).all()
    for k in (0, 1):
        got = math.fsum(probs[dt & (tg == k)])
        assert got == pytest.approx(float(hit * p_pass / 2), abs=1e-15)


@pytest.mark.slow
def test_enumeration_matches_sampled_frequencies(tables, names):
    s = _level(_state(tables, names), 0, 0, 5)
    acts = A("Roar", "Tackle", s, names)
    n = 20000
    check_frequencies(children(tables, s, acts), sampled(tables, s, acts, n), n)


@pytest.mark.slow
def test_sampled_draws(tables, names):
    s = _level(_state(tables, names), 0, 0, 5)
    acts = A("Roar", "Tackle", s, names)
    keys = jax.random.split(jax.random.key(1), 4000)
    o = jax.vmap(sample_outcome, in_axes=(0, None, None, None))(keys, tables, s, acts)
    level = np.asarray(o.roar_level)[:, 0]
    target = np.asarray(o.roar_target)[:, 0]
    assert abs(level.mean() - 210 / 256) < 5 * math.sqrt(0.18 / 4000)
    assert set(np.unique(target).tolist()) == {0, 1}
    assert abs(target.mean() - 0.5) < 5 * math.sqrt(0.25 / 4000)


@pytest.mark.fullset
def test_roar_builder_is_registered():
    assert "_roar_reads" in [b.__name__ for b in _READ_BUILDERS]


# --- damage classes --------------------------------------------------------------------------


def test_damage_classes_classify_roar(tables, names):
    assert EffectId.ROAR in INPUT_CLAUSE_EFFECTS
    assert EffectId.ROAR in NON_TABLE_DAMAGE_EFFECTS
    s = _state(tables, names, mine="Tackle", opp="Roar")
    acts = A("Tackle", "Roar", s, names)
    can = np.asarray(damage_inputs_can_change(tables, s, acts, np.array([True, False])))
    assert can[0].all()  # the roar clause: the foe's earlier Roar


def _minus_six(tables, names, move):
    """``tables`` with ``move`` at priority -6 (Roar's): a damaging move after a Roar."""
    priority = tables.move_priority.at[names.move(move)].set(-6)
    return tables._replace(move_priority=priority)


@pytest.mark.slow
@pytest.mark.parametrize("bench", ["full", "none"])
def test_a_hit_after_a_roar_bruteforce(tables, names, bench):
    # A constructed Tackle at -6 after the faster foe's Roar: dragged, it never runs; with no
    # bench the Roar fails and the Tackle hits uncompressed (the roar clause).
    t = _minus_six(tables, names, "Tackle")
    s = _state(t, names, mine="Tackle", opp="Roar", order="opp")
    s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    if bench == "none":
        s = _faint(s, 0, 1, 2, 3, 4, 5)
    acts = A("Tackle", "Roar", s, names)
    assert int(_events(t, s, acts, _o())["order"][0]) == 1
    assert_equals_bruteforce(t, s, acts)


# --- brute force: gated == loose -------------------------------------------------------------


def _setup(tables, names, case):
    """(state, actions) of one brute-force case (the rules §6 / brief list)."""
    kind, *args = case
    if kind == "level_bench":  # the foe's Roar: its level vs the player's L14, bench 0-3
        level, bench = args
        s = _state(tables, names, mine="Tackle", opp="Roar")
        s = _faint(_level(s, 1, 0, level), 0, *range(1 + bench, 6))
        return s, A("Tackle", "Roar", s, names)
    s = _state(tables, names, order=args[0] if args else "opp")
    acts = A("Roar", "Tackle", s, names)
    if kind == "protect":
        s = _state(tables, names, opp="Protect", order="opp")
        s = s._replace(protect_count=s.protect_count.at[1].set(args[1]))
        return s, A("Roar", "Protect", s, names)
    if kind == "infatuated_user":  # the Roar user is in love: immobilized or not
        s = _level(s._replace(infatuated=s.infatuated.at[0].set(1)), 0, 0, 5)
    elif kind == "paralysed_user":
        s = _level(s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS)), 0, 0, 5)
    elif kind == "quick_claw_in":
        s = s._replace(item=s.item.at[1, 1].set(names.item("Quick Claw")))
    elif kind == "intimidate_in":
        s = s._replace(ability=s.ability.at[1, 2].set(AbilityId.INTIMIDATE))
    elif kind == "natural_cure_out":
        s = s._replace(
            ability=s.ability.at[1, 0].set(AbilityId.NATURAL_CURE),
            status=s.status.at[1, 0].set(Status.BURN),
        )
    elif kind == "shed_skin_in":  # read at the turn's end only if the drag brings it in
        s = s._replace(
            ability=s.ability.at[1, 2].set(AbilityId.SHED_SKIN),
            status=s.status.at[1, 2].set(Status.POISON),
        )
        if "no_bench" in args:  # Onix fainted, the holder too: no drag, nothing read
            s = _faint(s, 1, 1, 2)
    elif kind == "speed_tie_in":  # the newcomer ties the Roar user's speed at the turn's end
        s = s._replace(stats=s.stats.at[1, 2, _SPE].set(s.stats[0, 0, _SPE]))
    elif kind == "drag_needs_bench":
        # Geodude at four times the Roar user's Speed (tied only if paralyzed, which the
        # drag set adds) and ``bench`` living bench Pokemon at the user's Speed: no drag
        # without one, and end_speed_tie is not read then. The end order shows in the state:
        # the seeded Roar user's drain heals a poisoned full-HP newcomer before or after
        # its poison damage (the end-of-turn checks run per battler, step.END_OF_TURN_ORDER).
        (bench,) = args[1:]
        spe = s.stats[0, 0, _SPE]
        s = s._replace(
            stats=s.stats.at[1, 0, _SPE].set(4 * spe).at[1, 1:3, _SPE].set(spe),
            status=s.status.at[1, 1:3].set(Status.POISON),
            leech_seed=s.leech_seed.at[0].set(1),
        )
        s = _faint(s, 1, *range(1 + bench, 3))
        return s, A("Roar", "Tackle", s, names)
    elif kind == "infatuated_target":  # Geodude in love with the Roar user, then dragged out
        s = s._replace(infatuated=s.infatuated.at[1].set(1))
    elif kind == "roar_vs_roar":
        s = _state(tables, names, opp="Roar", order=args[0])
        s = _level(_level(s, 0, 0, 13), 1, 0, 13)
        if len(args) > 1:  # the slower side's Quick Claw activates
            slow = 0 if args[0] == "opp" else 1
            s = s._replace(
                item=s.item.at[slow, 0].set(names.item("Quick Claw")),
                qc_roll=s.qc_roll.at[slow].set(1),
            )
        return s, A("Roar", "Roar", s, names)
    elif kind == "switch_target":  # the foe switches first: Roar can drag Geodude back
        s = _level(s, 0, 0, 13)
        return s, A("Roar", MOVE_ACTIONS + 2, s, names)
    return s, acts


BRUTE_CASES = (
    [("level_bench", lv, b) for lv in (10, 14, 20) for b in (0, 1, 2, 3)]
    + [("protect", "opp", c) for c in (0, 1)]
    + [("infatuated_user", "opp"), ("paralysed_user", "opp")]
    + [("quick_claw_in", "opp"), ("intimidate_in", "opp"), ("natural_cure_out", "opp")]
    + [("shed_skin_in", "opp"), ("speed_tie_in", "opp"), ("infatuated_target", "opp")]
    + [("shed_skin_in", "opp", "no_bench")]
    + [("roar_vs_roar", o) for o in ("player", "opp", "tie")]
    + [("roar_vs_roar", "opp", "claw"), ("switch_target", "opp")]
    + [("drag_needs_bench", "opp", b) for b in (0, 1, 2)]
)


@pytest.mark.parametrize("case", BRUTE_CASES, ids=lambda c: "-".join(map(str, c)))
def test_gated_equals_bruteforce(tables, names, case):
    s, acts = _setup(tables, names, case)
    width, brute = assert_equals_bruteforce(tables, s, acts)
    assert width <= brute


@pytest.mark.parametrize("bench", [0, 1, 2])
def test_a_drag_widens_the_end_order_only_with_a_bench(tables, names, bench):
    # Task 12: a foe's Roar adds the dragged side's speeds to end_speed_tie only when a drag
    # can happen (roar_bench >= 1, BtlCmd_TryWhirlwind, battle_script.c:5285-5297).
    s, acts = _setup(tables, names, ("drag_needs_bench", "opp", bench))
    laws = laws_of(tables, s, acts)
    assert int(laws.roar_bench[0]) == bench
    assert bool(laws.relevant.end_speed_tie) == (bench >= 1)
    assert bool(laws_of(tables, s, acts, loose=True).relevant.end_speed_tie)


def test_no_shed_skin_read_without_a_bench(tables, names):
    # The same gate guards shed_skin's drag clause. With no eligible bench the only living
    # Pokemon of the side is the Roar's target, whose own Shed Skin the holder clause reads
    # anyway, so the gate cannot change shed_skin (a fainted holder is never living).
    s, acts = _setup(tables, names, ("shed_skin_in", "opp", "no_bench"))
    laws = laws_of(tables, s, acts)
    assert int(laws.roar_bench[0]) == 0
    assert not bool(laws.relevant.shed_skin[1])


@pytest.mark.parametrize("level", [12, 16])
def test_kaizo_vulpix_roar_bruteforce(kaizo_tables, kaizo_names, level):
    # Rules §6(a): Vulpix (L16 in the save) Roars Roark's Cranidos (L14) with Gible and
    # Lileep behind it; at L12 the level check is drawn.
    t, n = kaizo_tables, kaizo_names
    vulpix = _kaizo_spec(n, "Vulpix", "Roar", level)
    foes = [
        _kaizo_spec(n, "Cranidos", "Headbutt", 14),
        _kaizo_spec(n, "Gible", "Headbutt", 15),
        _kaizo_spec(n, "Lileep", "Headbutt", 13),
    ]
    s = make_state(t, [vulpix], foes)
    acts = jnp.zeros(2, jnp.int32)
    laws = laws_of(t, s, acts)
    assert bool(np.asarray(laws.relevant.roar_level)[0]) == (level < 14)
    assert bool(np.asarray(laws.relevant.roar_target)[0])
    width, brute = assert_equals_bruteforce(t, s, acts)
    assert width <= brute


# --- coverage --------------------------------------------------------------------------------


@pytest.mark.fullset
def test_roar_is_supported_and_its_blockers_refused(tables, names):
    base = sample_party(names)
    mon = replace_spec(base[0], moves=(names.move("Roar"), *base[0].moves[1:]))
    report = check_coverage(tables, names, [mon, *base[1:]], roark_trainer(), roark_party(names), 0)
    assert not any("ROAR" in m.upper() for m in report.missing)
    # Suction Cups (Lileep, vanilla) and Soundproof (Voltorb) are unsupported abilities.
    for species, ability in (("Lileep", "Suction Cups"), ("Voltorb", "Soundproof")):
        sp = names.species(species)
        k = [names.ability_names[a] for a in names.species_abilities(sp)].index(ability)
        foe = replace_spec(roark_party(names)[0], species=sp, form=0, ability_slot=k)
        report = check_coverage(
            tables, names, [mon, *base[1:]], roark_trainer(), [foe, *roark_party(names)[1:]], 0
        )
        assert any(m.endswith(f"{ability} (ability UNSUPPORTED)") for m in report.missing)
    # Ingrain is modelled since E2 batch 7 (its anchoring: effects.ingrain_anchors, B7-5).
    assert int(tables.move_effect[names.move("Ingrain")]) == EffectId.INGRAIN
