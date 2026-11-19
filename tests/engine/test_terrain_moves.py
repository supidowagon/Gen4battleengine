"""Camouflage and Nature Power (K3b2 Task 10, ``EffectId.CAMOUFLAGE``, ``NATURE_POWER`` and the
``TRI_ATTACK`` it calls on BUILDING terrain; docs/kaizo/k3b2-rules.md §7, §8).

Decomp (external/pokeplatinum):

* ``include/data/terrain/to_move.h`` ``sTerrainMove`` and ``to_type.h``
  ``sTerrainCamouflageType``, indexed by ``min(terrain, TERRAIN_SPECIAL)``
  (``include/constants/battle/terrain.h``: ``TERRAIN_SPECIAL`` is ``TERRAIN_AARON``).
* ``effect_script_0213.s`` -> ``BtlCmd_TryCamouflage`` (src/battle/battle_script.c:7997-8022):
  Multitype fails, else the terrain's type is written to ``type1`` and ``type2`` unless the
  user already has it (``MON_IS_NOT_TYPE``, include/battle/common.h:54).
* ``effect_script_0173.s``: ``GetTerrainMove`` (battle_script.c:8036-8048) then
  ``GoToMoveScript FALSE`` (:2111-2134): ``moveCur = msgMoveTemp``, the defender re-picked with
  ``randomize`` (BattleSystem_Defender, battle_lib.c:1683-1705: the living foe in singles, else
  BATTLER_NONE -> subscript_no_target). ``moveTemp`` (command selection, bcp.c:1861-1898) stays
  Nature Power: ``movePrevByBattler`` (bcp.c:3945-3949), Disable and the AI's seen moves.
  ``moveHit[defender] = moveCur`` (bcp.c:4499-4500).
* ``effect_script_0036.s`` (Tri Attack): ``Random 2, 3`` (``RandNext() % 3 + 3``,
  battle_script.c:3200-3210: burn, freeze, paralysis) before ``CalcCrit``; the secondary is the
  probabilistic branch of ``BattleSystem_TriggerSecondaryEffect`` (battle_lib.c:1570-1586).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.trainer import TERRAIN_NAMES, TERRAIN_UNKNOWN
from nuzline.engine.coverage import terrain_call, terrain_refusal
from nuzline.engine.enumerate import (
    _LAWS,
    INPUT_CLAUSE_EFFECTS,
    NON_TABLE_DAMAGE_EFFECTS,
    TABLE_DAMAGE_EFFECTS,
    damage_inputs_can_change,
    enumerate_drawn,
    enumerate_outcomes,
)
from nuzline.engine.ids import (
    TERRAIN_MOVES,
    TERRAIN_SPECIAL,
    TERRAIN_TYPES,
    TRI_ATTACK_STATUSES,
    AbilityId,
    EffectId,
    MoveFlag,
    Status,
    Type,
)
from nuzline.engine.observe import OBS_FIELDS, observe
from nuzline.engine.order import called_move
from nuzline.engine.outcome import Outcome, empty_outcome, sample_outcome
from nuzline.engine.state import make_state
from nuzline.engine.step import ACT_MOVE, ACT_TAUNTED, replace, step_events
from tests.engine.helpers import A, as_species, assert_matches_reference, jit_step, roark_state
from tests.engine.test_enumerate import _kaizo_spec, assert_equals_bruteforce, laws_of
from tools.effect_map import DECOMP_EFFECT_TO_ID

DECOMP = Path("external/pokeplatinum")
NP, CAMO, TRI = "Nature Power", "Camouflage", "Tri Attack"
BUILDING = TERRAIN_NAMES.index("TERRAIN_BUILDING")
GRASS = TERRAIN_NAMES.index("TERRAIN_GRASS")
MOUNTAIN = TERRAIN_NAMES.index("TERRAIN_MOUNTAIN")
ROARK = 246
_SPE = 4  # stats[..., 0..4] are Atk..Spe
BURN, FREEZE, PARA = 0, 1, 2  # Outcome.tri_attack values


def _ability(s, side, ability):
    return s._replace(ability=s.ability.at[side, int(s.active[side])].set(ability))


def _state(
    tables,
    names,
    *,
    player=(NP,),
    opp=("Tackle",),
    order="player",
    terrain=BUILDING,
    lucky=True,
):
    """The sample party's lead (``player`` moves, ability none) against Cranidos (``opp``
    moves, Sturdy instead of Mold Breaker) on ``terrain``; ``order`` is who is faster
    ("player", "opp") or "tie"; ``lucky`` blocks every crit (Lucky Chant on both sides: a
    smaller brute force)."""
    s = roark_state(
        tables,
        names,
        player_moves=list(player),
        opp_species="Cranidos",
        opp_moves=list(opp),
    )
    s = _ability(_ability(s, 1, AbilityId.STURDY), 0, AbilityId.NONE)
    spe = int(s.stats[0, 0, _SPE])
    foe = {"player": spe - 1, "opp": spe + 1, "tie": spe}[order]
    s = s._replace(
        stats=s.stats.at[1, 0, _SPE].set(foe), terrain=jnp.asarray(terrain, s.terrain.dtype)
    )
    if lucky:
        s = s._replace(lucky_chant=s.lucky_chant.at[:].set(5))
    return s


def _o(*, tri=0, secondary=0, hit=1, roll=15, side=0) -> Outcome:
    o = empty_outcome()
    return o._replace(
        tri_attack=o.tri_attack.at[side].set(tri),
        secondary=o.secondary.at[side, 0].set(secondary),
        hit=o.hit.at[side, 0].set(hit),
        roll=o.roll.at[side, 0].set(roll),
    )


def _events(tables, s, acts, o) -> dict:
    _, events = jax.jit(step_events)(tables, s, acts, o)
    return {k: np.asarray(v) for k, v in events.items()}


def _types(s, side):
    return [int(t) for t in s.types[side, int(s.active[side])]]


# --- the decomp's tables and scripts --------------------------------------------------------


def _decomp_ids(name: str, prefix: str) -> dict[str, int]:
    lines = (DECOMP / "generated" / f"{name}.txt").read_text("utf-8").split()
    return {n: i for i, n in enumerate(lines) if n.startswith(prefix)}


def _terrain_table(header: str, prefix: str) -> list[str]:
    text = (DECOMP / "include" / "data" / "terrain" / header).read_text("utf-8")
    rows = re.findall(r"\[(TERRAIN_\w+)\]\s*=\s*(\w+)", text)
    assert [t for t, _ in rows] == [*TERRAIN_NAMES[:TERRAIN_SPECIAL], "TERRAIN_SPECIAL"]
    return [v for _, v in rows if v.startswith(prefix)]


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
@pytest.mark.fullset
def test_terrain_tables_are_the_decomps():
    moves = _decomp_ids("moves", "MOVE_")
    assert [moves[m] for m in _terrain_table("to_move.h", "MOVE_")] == list(TERRAIN_MOVES)
    types = {f"TYPE_{t.name}": int(t) for t in Type}
    assert [types[t] for t in _terrain_table("to_type.h", "TYPE_")] == [
        int(t) for t in TERRAIN_TYPES
    ]
    terrain_h = (DECOMP / "include/constants/battle/terrain.h").read_text("utf-8")
    assert "#define TERRAIN_SPECIAL TERRAIN_AARON" in terrain_h
    assert TERRAIN_NAMES[TERRAIN_SPECIAL] == "TERRAIN_AARON"
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8")
    # Both readers clamp past TERRAIN_SPECIAL (TERRAIN_MAX, 24, included).
    assert script.count("if (terrain > TERRAIN_SPECIAL) {\n        terrain = TERRAIN_SPECIAL;") >= 3


@pytest.mark.skipif(not DECOMP.exists(), reason="decomp not cloned")
def test_scripts_do_what_the_engine_does():
    effects = DECOMP / "res/battle/scripts/effects"
    tri = [ln.strip() for ln in (effects / "effect_script_0036.s").read_text("utf-8").splitlines()]
    assert tri[tri.index("_000:") + 1 :][:6] == [
        "Random 2, 3",
        "UpdateVar OPCODE_ADD, BTLVAR_CALC_TEMP, 0x80000000",
        "UpdateVarFromVar OPCODE_SET, BTLVAR_SIDE_EFFECT_FLAGS_INDIRECT, BTLVAR_CALC_TEMP",
        "CalcCrit",
        "CalcDamage",
        "End",
    ]
    ptrs = (DECOMP / "generated/battle_move_subscript_ptrs.txt").read_text("utf-8").split()
    assert [ptrs[3 + k] for k in range(3)] == [
        "MOVE_SUBSCRIPT_PTR_BURN",
        "MOVE_SUBSCRIPT_PTR_FREEZE",
        "MOVE_SUBSCRIPT_PTR_PARALYZE",
    ]
    assert TRI_ATTACK_STATUSES == (Status.BURN, Status.FREEZE, Status.PARALYSIS)
    script = (DECOMP / "src/battle/battle_script.c").read_text("utf-8")
    assert "battleCtx->calcTemp = (BattleSystem_RandNext(battleSys) % range) + offset;" in script
    np_ = (effects / "effect_script_0173.s").read_text("utf-8")
    assert "GetTerrainMove" in np_ and "GoToMoveScript FALSE" in np_
    camo = (effects / "effect_script_0213.s").read_text("utf-8")
    assert "TryCamouflage _016" in camo and "MOVE_STATUS_FAILED" in camo
    assert "ATTACKING_MON.type1 = type;\n        ATTACKING_MON.type2 = type;" in script
    assert "battleCtx->moveCur = battleCtx->msgMoveTemp;" in script


def test_effects_map_in_both_games(tables, names, kaizo_tables, kaizo_names):
    assert DECOMP_EFFECT_TO_ID[36] == (EffectId.TRI_ATTACK, (0, 0, 0))
    assert DECOMP_EFFECT_TO_ID[173] == (EffectId.NATURE_POWER, (0, 0, 0))
    assert DECOMP_EFFECT_TO_ID[213] == (EffectId.CAMOUFLAGE, (0, 0, 0))
    for t, n in ((tables, names), (kaizo_tables, kaizo_names)):
        assert int(t.move_effect[n.move(TRI)]) == EffectId.TRI_ATTACK
        assert int(t.move_effect[n.move(NP)]) == EffectId.NATURE_POWER
        assert int(t.move_effect[n.move(CAMO)]) == EffectId.CAMOUFLAGE


@pytest.mark.parametrize(("game", "power", "chance"), [("vanilla", 80, 20), ("kaizo", 95, 30)])
def test_tri_attack_data_from_the_tables(game, power, chance, request):
    t = request.getfixturevalue("tables" if game == "vanilla" else "kaizo_tables")
    n = request.getfixturevalue("names" if game == "vanilla" else "kaizo_names")
    m = n.move(TRI)
    assert (int(t.move_power[m]), int(t.move_effect_chance[m])) == (power, chance)
    assert int(t.move_accuracy[m]) == 100 and int(t.move_type[m]) == Type.NORMAL
    flags = int(t.move_flags[m])
    assert flags & MoveFlag.PROTECT and not flags & MoveFlag.CONTACT
    nature = n.move(NP)  # Nature Power: power 0 (Taunt), accuracy 0, no CAN_PROTECT
    assert (int(t.move_power[nature]), int(t.move_accuracy[nature])) == (0, 0)
    assert not int(t.move_flags[nature]) & MoveFlag.PROTECT


@pytest.mark.parametrize("game", ["vanilla", "kaizo"])
def test_roark_fight_calls_tri_attack_and_camouflages_normal(game, request):
    """Roark's fight is on BUILDING terrain (K3b2 Task 9, both games): there Nature Power calls
    Tri Attack and Camouflage gives Normal, read off the terrain tables."""
    if game == "vanilla":
        roark = next(t for t in request.getfixturevalue("trainers") if t.id == ROARK)
        names = request.getfixturevalue("names")
    else:
        roark = request.getfixturevalue("kaizo_roark")
        names = request.getfixturevalue("kaizo_names")
    assert roark.terrain == BUILDING
    assert names.move_names[terrain_call(roark.terrain)] == TRI
    assert TERRAIN_TYPES[roark.terrain] == Type.NORMAL


# --- Nature Power in step ----------------------------------------------------------------------


@pytest.mark.slow
def test_nature_power_is_tri_attack_from_its_script_on(tables, names):
    s = _state(tables, names, player=(NP, TRI))
    o = _o(roll=7)
    np_out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), o)
    tri_out = assert_matches_reference(tables, s, A(TRI, "Tackle", s, names), o)
    assert int(np_out.hp[1, 0]) == int(tri_out.hp[1, 0]) < int(s.hp[1, 0])  # Tri Attack's damage
    # Nature Power's own: its PP, movePrevByBattler; the foe's moveHit is the called move.
    np_slot, tri_slot = names.move_slot(s, 0, NP), names.move_slot(s, 0, TRI)
    assert int(np_out.pp[0, 0, np_slot]) == int(s.pp[0, 0, np_slot]) - 1
    assert int(np_out.pp[0, 0, tri_slot]) == int(s.pp[0, 0, tri_slot])
    assert int(np_out.last_move[0]) == names.move(NP)
    ev = _events(tables, s, A(NP, "Tackle", s, names), o)
    assert ev["act_kind"][0] == ACT_MOVE and ev["act_id"][0] == names.move(NP)
    assert ev["called"][0] == names.move(TRI) and ev["called"][1] == 0
    # The foe's record before its own move ends: step a faster foe that has not acted.
    s2 = _state(tables, names, player=(NP,), opp=("Leer",), order="player")
    out = jit_step(tables, s2, A(NP, "Leer", s2, names), o)
    assert int(out.last_move[0]) == names.move(NP)


def test_move_hit_records_the_called_move(tables, names):
    # The foe acts second; its moveHit after the player's action is cleared only at its own
    # MoveEnd, so read it through a Nature Power that KOs: the KO'd foe keeps the record.
    s = _state(tables, names).replace_hp(1, 0, 1)
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o())
    assert int(out.hp[1, 0]) == 0 and int(out.move_hit[1]) == names.move(TRI)


def test_accuracy_is_the_called_moves(tables, names):
    # Nature Power has accuracy 0 (never misses) but Tri Attack's roll runs: class 0 misses.
    s = _state(tables, names)
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o(hit=0))
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    ev = _events(tables, s, A(NP, "Tackle", s, names), _o(hit=0))
    assert ev["missed"][0] == 1 and ev["called"][0] == names.move(TRI)
    assert int(out.last_move[0]) == names.move(NP)  # a miss still succeeded (bcp.c:3212)


def test_protect_blocks_the_called_move(tables, names):
    s = _state(tables, names, opp=("Protect",), order="opp")
    out = assert_matches_reference(tables, s, A(NP, "Protect", s, names), _o(secondary=1))
    assert int(out.hp[1, 0]) == int(s.hp[1, 0]) and int(out.status[1, 0]) == Status.NONE
    assert _events(tables, s, A(NP, "Protect", s, names), _o())["protected"][0] == 1


def test_taunt_stops_nature_power(tables, names):
    # Nature Power's own power is 0: CHECK_STATUS_STATE_TAUNT fails it (no PP, no call).
    s = _state(tables, names, player=(NP, TRI))
    s = s._replace(taunt=s.taunt.at[0].set(3))
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o())
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    ev = _events(tables, s, A(NP, "Tackle", s, names), _o())
    assert ev["act_kind"][0] == ACT_TAUNTED and ev["called"][0] == 0


def test_disable_disables_nature_power(tables, names):
    # movePrevByBattler is Nature Power: the foe's later Disable disables Nature Power.
    s = _state(tables, names, opp=("Disable",), order="player")
    out = assert_matches_reference(tables, s, A(NP, "Disable", s, names), _o())
    assert int(out.disabled_move[0]) == names.move(NP)


def test_nature_power_without_a_target(tables, names):
    # The foe fainted earlier this turn: Nature Power's own defender is its user, so the
    # no-target check passes (PP spent, movePrevByBattler = Nature Power); GoToMoveScript
    # finds no defender for Tri Attack: subscript_no_target, and no moveHit on the foe.
    s = _state(tables, names, opp=("Explosion",), order="opp")
    ghost = jnp.asarray([Type.GHOST] * 2, s.types.dtype)  # Explosion cannot touch a Ghost
    s = s._replace(move_hit=s.move_hit.at[1].set(33), types=s.types.at[0, 0].set(ghost))
    out = assert_matches_reference(tables, s, A(NP, "Explosion", s, names), _o())
    assert int(out.hp[1, 0]) == 0 and int(out.hp[0, 0]) > 0
    slot = names.move_slot(s, 0, NP)
    assert int(out.pp[0, 0, slot]) == int(s.pp[0, 0, slot]) - 1
    assert int(out.last_move[0]) == names.move(NP)
    ev = _events(tables, s, A(NP, "Explosion", s, names), _o())
    assert ev["no_target"][0] == 1 and ev["called"][0] == names.move(TRI)


@pytest.mark.parametrize("terrain", [TERRAIN_SPECIAL, len(TERRAIN_NAMES) - 1])
def test_special_and_max_terrains_clamp(tables, names, terrain):
    s = _state(tables, names, terrain=terrain)
    assert int(called_move(tables, s, names.move(NP))) == names.move(TRI)
    assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o(secondary=1, tri=PARA))


def test_other_terrains_call_their_move(tables, names):
    s = _state(tables, names, terrain=MOUNTAIN)
    assert int(called_move(tables, s, names.move(NP))) == names.move("Rock Slide")
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o(secondary=1))
    assert int(out.flinched[1]) == 0  # flinch cleared at the turn end; the hit dealt damage
    assert int(out.hp[1, 0]) < int(s.hp[1, 0])


def test_unknown_terrain_never_indexes_a_table(tables, names):
    # TERRAIN_UNKNOWN (-1): the guard calls MOVE_NONE (no -1 index into sTerrainMove, whose
    # last row a -1 would read in JAX); coverage and trace_to_state refuse such a fight.
    s = _state(tables, names, player=(NP, CAMO), terrain=TERRAIN_UNKNOWN)
    assert int(called_move(tables, s, names.move(NP))) == 0
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), _o())
    assert int(out.hp[1, 0]) == int(s.hp[1, 0])
    out = assert_matches_reference(tables, s, A(CAMO, "Tackle", s, names), _o())
    assert _types(out, 0) == _types(s, 0)


# --- Tri Attack's status ---------------------------------------------------------------------


@pytest.mark.parametrize(("tri", "status"), list(enumerate(TRI_ATTACK_STATUSES)))
def test_tri_attack_status_on_a_proc(tables, names, tri, status):
    s = _state(tables, names, opp=("Leer",), order="opp")  # the foe moved: no turn-order reads
    out = assert_matches_reference(tables, s, A(NP, "Leer", s, names), _o(secondary=1, tri=tri))
    assert int(out.status[1, 0]) == status
    out = assert_matches_reference(tables, s, A(NP, "Leer", s, names), _o(secondary=0, tri=tri))
    assert int(out.status[1, 0]) == Status.NONE


@pytest.mark.parametrize(
    ("types", "tri", "status"),
    [
        ((Type.FIRE, Type.FIRE), BURN, Status.NONE),  # subscript_burn.s: Fire
        ((Type.FIRE, Type.FIRE), FREEZE, Status.FREEZE),
        ((Type.ICE, Type.ICE), FREEZE, Status.NONE),  # subscript_freeze.s: Ice
        ((Type.ELECTRIC, Type.ELECTRIC), PARA, Status.PARALYSIS),  # Gen 4: no Electric check
        ((Type.GHOST, Type.GHOST), PARA, Status.NONE),  # Normal into Ghost: no damage
    ],
)
def test_tri_attack_immunities(tables, names, types, tri, status):
    s = _state(tables, names, opp=("Leer",), order="opp")
    s = s._replace(types=s.types.at[1, 0].set(jnp.asarray(types, s.types.dtype)))
    out = assert_matches_reference(tables, s, A(NP, "Leer", s, names), _o(secondary=1, tri=tri))
    assert int(out.status[1, 0]) == status


def test_tri_attack_blocked_by_status_and_limber(tables, names):
    s = _state(tables, names, opp=("Leer",), order="opp")
    poisoned = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    out = assert_matches_reference(
        tables, poisoned, A(NP, "Leer", s, names), _o(secondary=1, tri=PARA)
    )
    assert int(out.status[1, 0]) == Status.POISON
    limber = _ability(s, 1, AbilityId.LIMBER)
    out = assert_matches_reference(
        tables, limber, A(NP, "Leer", s, names), _o(secondary=1, tri=PARA)
    )
    assert int(out.status[1, 0]) == Status.NONE


def test_freeze_then_the_slower_foe_reads_its_thaw(tables, names):
    s = _state(tables, names, order="player")
    frozen = _o(secondary=1, tri=FREEZE)
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), frozen)
    assert int(out.status[1, 0]) == Status.FREEZE and int(out.hp[0, 0]) == int(s.hp[0, 0])
    thawed = frozen._replace(thaw=frozen.thaw.at[1].set(1))
    out = assert_matches_reference(tables, s, A(NP, "Tackle", s, names), thawed)
    assert int(out.status[1, 0]) == Status.NONE and int(out.hp[0, 0]) < int(s.hp[0, 0])


# --- Camouflage --------------------------------------------------------------------------------


def test_camouflage_writes_the_terrain_type_and_fails_on_a_match(tables, names):
    s = _state(tables, names, player=(CAMO,), opp=("Leer",))
    assert Type.NORMAL not in _types(s, 0)
    out = assert_matches_reference(tables, s, A(CAMO, "Leer", s, names), _o())
    assert _types(out, 0) == [Type.NORMAL, Type.NORMAL]
    grass = _state(tables, names, player=(CAMO,), opp=("Leer",), terrain=GRASS)
    out = assert_matches_reference(tables, grass, A(CAMO, "Leer", grass, names), _o())
    assert _types(out, 0) == [Type.GRASS, Type.GRASS]
    # Already of that type (either slot): MON_IS_NOT_TYPE fails, nothing written.
    half = s._replace(
        types=s.types.at[0, 0].set(jnp.asarray([Type.FIRE, Type.NORMAL], s.types.dtype))
    )
    out = assert_matches_reference(tables, half, A(CAMO, "Leer", s, names), _o())
    assert _types(out, 0) == [Type.FIRE, Type.NORMAL]


def test_camouflage_fails_for_multitype(tables, names):
    # BtlCmd_TryCamouflage (:8001): Multitype, read through the decomp ability. GRASS terrain,
    # so that the type test (Arceus is Normal) is not what fails it.
    s = _state(tables, names, player=(CAMO,), opp=("Leer",), terrain=GRASS)
    arceus = names.species("Arceus")
    s = as_species(tables, s, 0, 0, arceus)
    out = assert_matches_reference(tables, s, A(CAMO, "Leer", s, names), _o())
    assert _types(out, 0) == _types(s, 0)
    plain = as_species(tables, s, 0, 0, names.species("Kecleon"))
    out = assert_matches_reference(tables, plain, A(CAMO, "Leer", s, names), _o())
    assert _types(out, 0) == [Type.GRASS, Type.GRASS]


def test_camouflage_types_are_restored_on_leave(tables, names):
    s = _state(tables, names, player=(CAMO,), opp=("Leer",))
    party = _types(s, 0)
    camo = assert_matches_reference(tables, s, A(CAMO, "Leer", s, names), _o())
    assert _types(camo, 0) == [Type.NORMAL, Type.NORMAL]
    # A switch out restores them.
    out = assert_matches_reference(tables, camo, A(5, "Leer", camo, names), _o())
    assert [int(t) for t in out.types[0, 0]] == party
    # A faint restores them (the foe's KO): the fainted battler's party types.
    weak = camo.replace_hp(0, 0, 1)
    out = assert_matches_reference(tables, weak, A(CAMO, "Tackle", weak, names), _o(side=1))
    assert int(out.hp[0, 0]) == 0 and [int(t) for t in out.types[0, 0]] == party
    # The replacement path (step.replace) restores nothing more: still the party types.
    back = replace(tables, out, 0, 1)
    assert [int(t) for t in back.types[0, 0]] == party


def test_camouflage_changes_the_foes_type_chart_and_its_own_stab(tables, names):
    # A Ghost Camouflaging into Normal is hit by the foe's Tackle the same turn.
    s = _state(tables, names, player=(CAMO, NP), opp=("Tackle",), order="player")
    ghost = s._replace(types=s.types.at[0, 0].set(jnp.asarray([Type.GHOST] * 2, s.types.dtype)))
    out = assert_matches_reference(tables, ghost, A(CAMO, "Tackle", s, names), _o())
    assert int(out.hp[0, 0]) < int(s.hp[0, 0])
    plain = assert_matches_reference(tables, ghost, A(NP, "Tackle", s, names), _o())
    assert int(plain.hp[0, 0]) == int(s.hp[0, 0])  # without it, Tackle cannot touch a Ghost
    # Next turn Nature Power's Tri Attack has Normal STAB.
    after = assert_matches_reference(tables, out, A(NP, "Leer", out, names), _o())
    stab = int(out.hp[1, 0]) - int(after.hp[1, 0])
    base = int(out.hp[1, 0]) - int(
        jit_step(tables, ghost._replace(hp=out.hp), A(NP, "Leer", out, names), _o()).hp[1, 0]
    )
    assert stab > base


@pytest.mark.slow
def test_observation_shows_types_and_the_call(tables, names):
    s = _state(tables, names, player=(CAMO, NP), opp=("Leer",))
    out = jit_step(tables, s, A(CAMO, "Leer", s, names), _o())
    obs = np.asarray(observe(tables, s, A(CAMO, "Leer", s, names), _o(), out))
    fields = {name: obs[sl] for name, sl in OBS_FIELDS}
    assert fields["active_types"][:2].tolist() == [Type.NORMAL, Type.NORMAL]
    obs = np.asarray(observe(tables, s, A(NP, "Leer", s, names), _o(), out))
    assert {name: obs[sl] for name, sl in OBS_FIELDS}["called"].tolist() == [names.move(TRI), 0]


# --- laws and enumeration ----------------------------------------------------------------------


@pytest.mark.slow
def test_tri_attack_law_and_draw(tables, names):
    assert _LAWS["tri_attack"](None, 0) == ([0, 1, 2], [1, 1, 1], 3)
    s = _state(tables, names)
    acts = A(NP, "Tackle", s, names)
    keys = jax.random.split(jax.random.key(3), 6000)
    draws = jax.vmap(lambda k: sample_outcome(k, tables, s, acts).tri_attack)(keys)
    counts = np.bincount(np.asarray(draws)[:, 0], minlength=3) / 6000
    assert np.all(np.abs(counts - 1 / 3) < 5 * math.sqrt(2 / 9 / 6000))
    # From fold_in(key, 21): every earlier field keeps its samples (test_outcome).
    o = sample_outcome(jax.random.key(3), tables, s, acts)
    want = jax.random.randint(jax.random.fold_in(jax.random.key(3), 21), (2,), 0, 3)
    assert np.asarray(o.tri_attack).tolist() == np.asarray(want).tolist()


def test_tri_attack_is_drawn_only_on_a_proc(tables, names):
    s = _state(tables, names, opp=("Leer",))
    laws = laws_of(tables, s, A(NP, "Leer", s, names))
    assert bool(np.asarray(laws.relevant.tri_attack)[0])
    outcomes, _, drawn = enumerate_drawn(laws)
    sec = np.asarray(outcomes.secondary)[:, 0, 0]
    tri_read = np.asarray(drawn.tri_attack)[:, 0]
    assert (tri_read == (np.asarray(drawn.secondary)[:, 0, 0] & (sec == 1))).all()
    tri = np.asarray(outcomes.tri_attack)[:, 0]
    assert set(tri[tri_read].tolist()) == {BURN, FREEZE, PARA}
    # The slower Cranidos's thaw / full paralysis is read only through the matching status.
    assert (np.asarray(drawn.thaw)[:, 1] == (tri_read & (tri == FREEZE))).all()
    assert (np.asarray(drawn.full_para)[:, 1] == (tri_read & (tri == PARA))).all()


def test_statused_target_reads_no_tri_attack(tables, names):
    s = _state(tables, names, opp=("Leer",))
    s = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    laws = laws_of(tables, s, A(NP, "Leer", s, names))
    assert not np.asarray(laws.relevant.tri_attack).any()
    assert not np.asarray(laws.relevant.secondary)[0].any()


def test_compression_and_camouflage(tables, names):
    # Nature Power's hit is its call's table hit: compressed. A faster foe's Camouflage
    # changes the target's types first: not compressed (damage_inputs_can_change).
    assert EffectId.NATURE_POWER in NON_TABLE_DAMAGE_EFFECTS
    assert EffectId.TRI_ATTACK in TABLE_DAMAGE_EFFECTS
    assert {EffectId.CAMOUFLAGE, EffectId.TRI_ATTACK} <= set(INPUT_CLAUSE_EFFECTS)
    s = _state(tables, names, opp=("Leer",), lucky=False)
    assert np.asarray(laws_of(tables, s, A(NP, "Leer", s, names)).compress)[:, 0, 0].all()
    s = _state(tables, names, opp=(CAMO,), order="opp", lucky=False)
    acts = A(NP, CAMO, s, names)
    assert np.asarray(damage_inputs_can_change(tables, s, acts, np.array([True, False])))[0, 0]
    assert not np.asarray(laws_of(tables, s, acts).compress)[:, 0, 0].any()


# The probes of the brute force: (setup, opponent move, order).
SETUPS = {
    "plain": ("Tackle", "Leer"),
    "statused": ("Tackle",),  # an existing status blocks every Tri Attack status
    "fire": ("Tackle",),  # burn blocked, freeze and paralysis open
    "ghost": ("Tackle",),  # Normal into Ghost: no damage, no secondary
    "limber": ("Tackle",),
    "protect": ("Protect",),
    "miss": ("Tackle",),  # accuracy -1: the call's roll can miss
    "faint": ("Tackle",),  # the foe KOs the Nature Power user first
    "frozen": ("Tackle",),  # a frozen foe thaws or not, then Tri Attack
    "camouflage": (CAMO,),  # the foe Camouflages (Rock -> Normal) first: type chart
}
ORDERS = ("player", "opp", "tie")
CASES = [(setup, opp, order) for setup, opps in SETUPS.items() for opp in opps for order in ORDERS]


@pytest.mark.slow
@pytest.mark.parametrize(("setup", "opp", "order"), CASES)
def test_nature_power_equals_bruteforce(tables, names, setup, opp, order):
    """The gated enumeration (the (tri_attack, secondary) joint factor, the thaw / full
    paralysis read through the called Tri Attack, the hit and Protect gates, damage classes
    of the called move) equals the loose brute force merged by state and observation."""
    s = _state(tables, names, opp=(opp,), order=order)
    if setup == "statused":
        s = s._replace(status=s.status.at[1, 0].set(Status.POISON))
    elif setup == "fire":
        s = s._replace(types=s.types.at[1, 0].set(jnp.asarray([Type.FIRE] * 2, s.types.dtype)))
    elif setup == "ghost":
        s = s._replace(types=s.types.at[1, 0].set(jnp.asarray([Type.GHOST] * 2, s.types.dtype)))
    elif setup == "limber":
        s = _ability(s, 1, AbilityId.LIMBER)
    elif setup == "miss":
        s = s.with_stages(0, acc=-1)
    elif setup == "faint":
        s = s.replace_hp(0, 0, 1)
    elif setup == "frozen":
        s = s._replace(status=s.status.at[1, 0].set(Status.FREEZE))
    width, brute = assert_equals_bruteforce(tables, s, A(NP, opp, s, names))
    if setup in ("plain", "fire", "limber", "miss"):
        assert width < brute


@pytest.mark.parametrize("order", ORDERS)
def test_camouflage_then_a_hit_equals_bruteforce(tables, names, order):
    # The player (a Ghost) Camouflages into Normal: the foe's Tackle then hits it (type chart),
    # and with crits on the foe's Tackle stays uncompressed only when it comes after.
    s = _state(tables, names, player=(CAMO,), opp=("Tackle",), order=order, lucky=False)
    s = s._replace(
        types=s.types.at[0, 0].set(jnp.asarray([Type.GHOST] * 2, s.types.dtype)),
        lucky_chant=s.lucky_chant.at[1].set(5),
    )
    assert_equals_bruteforce(tables, s, A(CAMO, "Tackle", s, names))


@pytest.mark.parametrize("np_first", [True, False])
def test_kaizo_nature_power_on_roarks_terrain(kaizo_tables, kaizo_names, kaizo_roark, np_first):
    # Kaizo tables (Tri Attack 95 BP, 30 %) on Roark's own terrain, against Cranidos's Leer.
    t, n = kaizo_tables, kaizo_names
    s = make_state(
        t,
        [_kaizo_spec(n, "Kecleon", NP, 20)],
        [_kaizo_spec(n, "Cranidos", "Leer", 14)],
        terrain=kaizo_roark.terrain,
    )
    spe = int(s.stats[0, 0, _SPE])
    s = s._replace(
        stats=s.stats.at[1, 0, _SPE].set(spe - 1 if np_first else spe + 1),
        lucky_chant=s.lucky_chant.at[:].set(5),
        ability=s.ability.at[:, 0].set(AbilityId.STURDY),
    )
    width, brute = assert_equals_bruteforce(t, s, jnp.zeros(2, jnp.int32))
    assert width < brute
    outcomes, probs = enumerate_outcomes(laws_of(t, s, jnp.zeros(2, jnp.int32)))
    burned = np.asarray(outcomes.tri_attack)[:, 0] == BURN
    procs = np.asarray(outcomes.secondary)[:, 0, 0] == 1
    assert abs(math.fsum(probs[procs & burned]) - 0.30 / 3) < 1e-12


# --- coverage ----------------------------------------------------------------------------------


def test_coverage_refuses_an_unsupported_call(tables, names, trainers):
    """Nature Power is refused where its terrain's call has an unsupported effect. Earthquake
    (TERRAIN_PLAIN's call) is supported since E2 batch 4 (B4-8), so the branch is shown with its
    effect made unsupported."""
    roark = next(t for t in trainers if t.id == ROARK)
    assert terrain_refusal(tables, names, roark, [names.move(NP)]) is None
    plain = roark.__class__(**{**roark.__dict__, "terrain": TERRAIN_NAMES.index("TERRAIN_PLAIN")})
    assert terrain_refusal(tables, names, plain, [names.move(NP), names.move(CAMO)]) is None
    quake = names.move("Earthquake")
    gone = tables._replace(move_effect=tables.move_effect.at[quake].set(EffectId.UNSUPPORTED))
    line = terrain_refusal(gone, names, plain, [names.move(NP), names.move(CAMO)])
    assert line is not None and "Earthquake" in line
    assert terrain_refusal(gone, names, plain, [names.move(CAMO)]) is None
