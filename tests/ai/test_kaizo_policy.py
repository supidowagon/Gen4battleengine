"""The BASIC and EVAL_ATTACK paths of Kaizo Roark's moves (K3c Task 5; docs/ai/kaizo_roark.md
5.1-5.2, the traces of section 13).

Decomp paths are relative to ``external/pokeplatinum``: ``script.s`` is
``src/battle/trainer_ai/script.s``, ``trainer_ai.c`` is ``src/battle/trainer_ai/trainer_ai.c``
and ``battle_lib.c`` is ``src/battle/battle_lib.c``.

The hand-traced decisions give the four move scores after the BASIC and EVAL_ATTACK flag
scripts only (``_two_flag_scores``): the EXPERT and RISKY paths of these effects are K3c
Task 6, so a full ``ai_scores`` would still lack them. Each trace's script lines are in
kaizo_roark.md 13. Facts a trace takes as input (an effectiveness class, which damage
estimate is the highest, whether a move kills) are asserted first, so a data surprise fails
there and not in the score.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands as c
from nuzline.ai import policy
from nuzline.ai.commands import BattleEffect, TypeMulti
from nuzline.ai.layout import KAIZO
from nuzline.ai.outcome import (
    KAIZO_EXPERT_LABELS,
    SIM_ROLL_MAX,
    SWITCH_NONE,
    AiOutcome,
    slot_roll_sites,
    slot_site_table,
)
from nuzline.engine.damage import ai_damage_for_roll
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.state import MOVES, make_state
from tests.ai.test_kaizo_outcome import (
    KAIZO_ALL_EXPERT,
    _assert_exact_equals_loose,
    _bulky,
    _with_player,
)

DECOMP = Path(__file__).resolve().parents[2] / "external/pokeplatinum"

# Kaizo Roark's party order and move ids (data/intermediate_kaizo/trainers.json, trainer 246;
# moves.json), checked by test_roark_moves_and_effects.
BONSLY, LILEEP, GIBLE, CORSOLA, SHUCKLE, CRANIDOS = range(6)
STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK = 446, 120, 280, 205
SLUDGE_BOMB, GIGA_DRAIN, ANCIENTPOWER, RECOVER = 188, 202, 246, 105
DRAGONBREATH, FLAMETHROWER, THUNDER_FANG, EARTH_POWER = 225, 53, 422, 414
WATER_PULSE, LIGHT_SCREEN = 352, 113
TOXIC, CURSE, SAND_TOMB = 92, 174, 328
HEAD_SMASH, ZEN_HEADBUTT, CRUNCH, ROCK_POLISH = 457, 428, 242, 397
ROARK_MOVES = {  # Kaizo id: (name, battle effect)
    STEALTH_ROCK: ("Stealth Rock", 266),
    SELFDESTRUCT: ("Selfdestruct", 7),
    BRICK_BREAK: ("Brick Break", 186),
    ACCELEROCK: ("Accelerock", 0),
    SLUDGE_BOMB: ("Sludge Bomb", 2),
    GIGA_DRAIN: ("Giga Drain", 3),
    ANCIENTPOWER: ("AncientPower", 140),
    RECOVER: ("Recover", 32),
    DRAGONBREATH: ("DragonBreath", 6),
    FLAMETHROWER: ("Flamethrower", 4),
    THUNDER_FANG: ("Thunder Fang", 275),
    EARTH_POWER: ("Earth Power", 72),
    WATER_PULSE: ("Water Pulse", 76),
    LIGHT_SCREEN: ("Light Screen", 35),
    TOXIC: ("Toxic", 33),
    CURSE: ("Curse", 109),
    SAND_TOMB: ("Sand Tomb", 261),
    HEAD_SMASH: ("Head Smash", 269),
    ZEN_HEADBUTT: ("Zen Headbutt", 31),
    CRUNCH: ("Crunch", 69),
    ROCK_POLISH: ("Rock Polish", 52),
}
DAMP, WATER_ABSORB, SIMPLE, MOLD_BREAKER = 6, 11, 86, 104  # commands.ABILITY_IDS
WOOPER = 194


# --- States, outcomes, scores ------------------------------------------------------------


def _state(t, party, mon: int, moves, bench=(CRANIDOS,), players: int = 2, hp: int = 400):
    """Kaizo Roark's ``mon`` with ``moves`` (``bench`` behind it), against ``players`` copies
    of Roark's Gible as the player's party, the active one at ``hp``/``hp`` (no transcribed
    move kills 400)."""
    opp = [replace(party[mon], moves=tuple(moves), pp=None)] + [party[b] for b in bench]
    s = make_state(t, [party[GIBLE]] * players, opp)
    return _bulky(s, hp)


def _outcome(t, s, guess: int = 0, jumps=()) -> AiOutcome:
    """The per-slot outcome where the ``(slot, line)`` sites of ``jumps`` jump and no other
    does; ``guess`` is the ability guess."""
    rolls = np.zeros((MOVES, KAIZO.slot_sites), np.int8)
    moves = np.asarray(s.moves[1, s.active[1]])
    for slot, line in jumps:
        _, lines = slot_site_table(t, KAIZO, int(moves[slot]))
        (k,) = np.flatnonzero(np.asarray(lines) == line)
        rolls[slot, k] = 1
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.asarray(guess, jnp.int8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.asarray(0, jnp.int8),
        switch_stop=jnp.asarray(SWITCH_NONE, jnp.int8),
    )


@partial(jax.jit, static_argnames="flags")
def _flag_scores(t, s, o, flags: int = 2):
    ctx = c.new_context(t, s, o, slot_roll_sites(t, s, KAIZO))
    return policy._run_flags(ctx, KAIZO, policy._flags(KAIZO)[:flags]).scores


def _two_flag_scores(t, s, guess: int = 0, jumps=()) -> list[int]:
    """The four move scores after BASIC and EVAL_ATTACK (trainer_ai.c:297-310), refusals
    not applied."""
    return [int(x) for x in _flag_scores(t, s, _outcome(t, s, guess, jumps))]


def _effectiveness(t, s, move) -> int:
    return int(c.effectiveness_class(t, s, move))


def _damage(t, s) -> list[int]:
    """The AI's damage estimate of each slot (``TrainerAI_CalcAllDamage``, 0 for a no-calc
    or power-0 move)."""
    return [int(x) for x in c._slot_damage(t, s, jnp.full((MOVES,), SIM_ROLL_MAX))]


def _with_abilities(t, species: int, abilities):
    """``t`` with ``species``' two abilities replaced (a synthetic species row)."""
    rows = t.species_abilities_decomp.at[species].set(
        jnp.asarray(abilities, t.species_abilities_decomp.dtype)
    )
    return t._replace(species_abilities_decomp=rows)


def _named(t, names, s, species: str, slot: int = 0):
    return _with_player(t, s, names.species(species), slot)


# --- The data the paths key on ---------------------------------------------------------


def test_roark_moves_and_effects(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Kaizo Roark's moves, by Kaizo id, with the battle effect the AI reads
    (``MOVE_DATA(move).effect``, ``tables.move_ai_effect``; trainer_ai.c:1441): every one is
    transcribed now, and the effect set is what the plan lists."""
    t, n = kaizo_tables, kaizo_names
    effect = np.asarray(t.move_ai_effect)
    known = {m for mon in kaizo_roark_party for m in mon.moves if m}
    assert known == set(ROARK_MOVES)
    for move, (name, eff) in ROARK_MOVES.items():
        assert n.move_names[move] == name
        assert int(effect[move]) == eff, name
        assert eff in KAIZO.transcribed, name
    species = [n.species_names[m.species] for m in kaizo_roark_party]
    assert species == ["Bonsly", "Lileep", "Gible", "Corsola", "Shuckle", "Cranidos"]
    # Accelerock has priority +1 but effect HIT, not PRIORITY_1 (plan K3c Ruling 8).
    assert int(t.move_priority[ACCELEROCK]) == 1 and int(effect[ACCELEROCK]) == 0


def test_roark_moves_refused_only_for_their_expert_labels(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    """Every effect of Kaizo Roark's moves is transcribed (no "script paths ... are not
    transcribed" line). The moves whose Expert_Main entry goes to a label K3c Task 6
    transcribes are refused by name of that label; with those labels counted as transcribed
    (``KAIZO_ALL_EXPERT``), no move line is left."""
    t, n, party = kaizo_tables, kaizo_names, kaizo_roark_party
    s = make_state(t, [party[GIBLE]], list(party))
    reasons = [r for r in policy.ai_refusals(t, s, n) if "opponent slot" in r]
    assert not any("script paths of move" in r for r in reasons), reasons
    expected = set()
    for slot, mon in enumerate(party):
        for move in mon.moves:
            effect = ROARK_MOVES[move][1]
            if effect in KAIZO_EXPERT_LABELS:
                label, line = KAIZO_EXPERT_LABELS[effect]
                expected.add(
                    f"ai scoring: opponent slot {slot}: move {n.move_names[move]} goes to "
                    f"{label} (Expert_Main, script.s:{line}), whose script path is not "
                    "transcribed yet (docs/ai/kaizo_roark.md 5)"
                )
    assert set(reasons) == expected and len(expected) == 10
    rest = policy.ai_refusals(t, s, n, layout=KAIZO_ALL_EXPERT)
    assert not any("opponent slot" in r for r in rest), rest


def test_expert_label_table_is_the_dispatch():
    """``KAIZO_EXPERT_LABELS`` is read off ``Expert_Main``'s dispatch table (script.s:1628-1808,
    ``IfCurrentMoveEffectEqualTo BATTLE_EFFECT_<effect>, <label>``): each entry's line holds
    its effect and label, and every transcribed effect with an entry is either there or one
    whose label policy._expert transcribes (DEF_DOWN, DEF_DOWN_2, HIT_BEFORE_SWITCH,
    STEALTH_ROCK). After K3c Task 5 no label of the table is transcribed."""
    script = DECOMP / "src/battle/trainer_ai/script.s"
    effects = DECOMP / "generated/move_battle_effects.txt"
    if not script.exists() or not effects.exists():
        pytest.skip("decomp not present")
    lines = script.read_text(encoding="utf-8").splitlines()
    names = effects.read_text(encoding="utf-8").split()
    dispatch = {}
    for number in range(1628, 1809):
        words = lines[number - 1].replace(",", " ").split()
        if words[:1] == ["IfCurrentMoveEffectEqualTo"]:
            dispatch.setdefault(names.index(words[1]), (words[2], number))
    for effect, entry in KAIZO_EXPERT_LABELS.items():
        assert dispatch[effect] == entry, effect
    old = {
        BattleEffect.DEF_DOWN,
        BattleEffect.DEF_DOWN_2,
        BattleEffect.HIT_BEFORE_SWITCH,
        BattleEffect.STEALTH_ROCK,
    }
    assert set(dispatch) & KAIZO.transcribed == set(KAIZO_EXPERT_LABELS) | old
    assert KAIZO.expert_transcribed == frozenset()


def test_expert_guard_refuses_a_0b111_kaizo_trainer(
    kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party, monkeypatch
):
    """Coordinator ruling (K3c Task 5): a Kaizo trainer with the EXPERT flag but not RISKY
    (synthetic: Roark with ai_flags 0b111) holding Recover is refused by name of
    Expert_Recovery (Expert_Main's RESTORE_HALF_HP entry, script.s:1652), never scored without
    it; once RESTORE_HALF_HP is in the transcribed Expert set the refusal is gone."""
    from nuzline.ai import layout as layout_module
    from nuzline.engine.coverage import check_coverage

    t, n = kaizo_tables, kaizo_names
    trainer = replace(kaizo_roark, ai_flags=0b111)
    opp = [replace(kaizo_roark_party[CORSOLA], moves=(RECOVER, ACCELEROCK, 0, 0), pp=None)]
    player = [kaizo_roark_party[GIBLE]]
    line = (
        "ai scoring: opponent slot 0: move Recover goes to Expert_Recovery (Expert_Main, "
        "script.s:1652), whose script path is not transcribed yet (docs/ai/kaizo_roark.md 5)"
    )
    report = check_coverage(t, n, player, trainer, opp, Weather.NONE)
    assert not any("AI_FLAG" in m for m in report.missing), report.missing
    assert any(line in m for m in report.missing), report.missing
    opened = KAIZO._replace(expert_transcribed=frozenset({BattleEffect.RESTORE_HALF_HP}))
    monkeypatch.setattr(layout_module, "LAYOUTS", (layout_module.VANILLA, opened))
    report = check_coverage(t, n, player, trainer, opp, Weather.NONE)
    assert not any("Expert_" in m or "ai scoring" in m for m in report.missing), report.missing


def test_battle_effect_ids_are_the_decomps():
    """``commands.BattleEffect``: line n + 1 of generated/move_battle_effects.txt is
    ``BATTLE_EFFECT_<name>``, id n."""
    path = DECOMP / "generated/move_battle_effects.txt"
    if not path.exists():
        pytest.skip("decomp not present")
    lines = path.read_text(encoding="utf-8").split()
    for member in BattleEffect:
        assert lines[member.value] == f"BATTLE_EFFECT_{member.name}", member


def test_no_species_splits_the_toxic_reads(tables, kaizo_tables):
    """Basic_CheckCannotPoison reads the defender's ability twice (script.s:529, :539), two
    coin flips in the decomp (trainer_ai.c:1191-1198) and one guess in the exact layout
    (Ruling 6). Both reads can change Toxic's score only for a species whose two abilities
    are Hydration (:540, in rain) and one of Immunity, Magic Guard, Poison Heal or Leaf Guard
    (:530-534, Leaf Guard in sun). No species of either game has such a pair, in any
    weather (test_one_scalar_guess_is_exact_for_roark checks the state's weather only)."""
    first = {c.ABILITY_IDS[a] for a in ("Immunity", "Magic Guard", "Poison Heal", "Leaf Guard")}
    hydration = c.ABILITY_IDS["Hydration"]
    for t in (tables, kaizo_tables):
        for row in np.asarray(t.species_abilities_decomp).tolist():
            assert not (hydration in row and first & set(row)), row


def test_site_rows_of_roark_effects():
    """The per-slot site table (outcome.EFFECT_SITE_LINES) of Task 5: every effect reaches
    EvalAttack's quad roll (script.s:6383); HALVE_DEFENSE first reaches
    EvalAttack_MaybeDeprioritize's 51 roll (:6365 -> :6374). No BASIC label of these effects
    reads IfRandomLessThan (script.s:133-286 and the labels of 5.1)."""
    rows = dict(KAIZO.effect_sites)
    for _, eff in ROARK_MOVES.values():
        assert (6383, 80) in rows[eff], eff
    assert rows[BattleEffect.HALVE_DEFENSE][:2] == ((6374, 51), (6383, 80))


# --- BASIC: Basic_CheckCannotExplode (script.s:297-319) ----------------------------------


@pytest.mark.parametrize(
    ("bench", "players", "jump", "expected"),
    [
        # B1: Bonsly not last: CountAlivePartyBattlers ATTACKER = 1 -> :318 PopOrEnd.
        ((CRANIDOS,), 2, 1, [100, 100, 100, 99]),
        ((CRANIDOS,), 2, 0, [100, 98, 100, 99]),
        # B2: Bonsly last, the player not: :313 ScoreMinus10.
        ((), 2, 1, [100, 90, 100, 99]),
        ((), 2, 0, [100, 88, 100, 99]),
        # B3: both last: :316 ScoreMinus1; Stealth Rock -10 too (:1546-1547).
        ((), 1, 1, [90, 99, 100, 99]),
        ((), 1, 0, [90, 97, 100, 99]),
    ],
)
def test_selfdestruct_last_mon(kaizo_tables, kaizo_roark_party, bench, players, jump, expected):
    t = kaizo_tables
    moves = (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK)  # Roark's Bonsly
    s = _state(t, kaizo_roark_party, BONSLY, moves, bench, players)
    # Inputs: Normal vs Dragon/Ground is no immunity and not QUADRUPLE; Brick Break is the
    # highest damage, Selfdestruct estimates 0 (no-calc, trainer_ai.c:31-46), nothing kills.
    assert _effectiveness(t, s, SELFDESTRUCT) not in (TypeMulti.IMMUNE, TypeMulti.QUADRUPLE)
    damage = _damage(t, s)
    assert damage[1] == 0 and damage[2] > damage[3] > 0 and max(damage) < 400
    jumps = ((1, 6374),) if jump else ()
    assert _two_flag_scores(t, s, jumps=jumps) == expected


def test_selfdestruct_damp_guess(tables, names, kaizo_tables, kaizo_roark_party):
    """A Damp guess (script.s:304-305). No Kaizo species has Damp (data/intermediate_kaizo
    species.json, through ``species_abilities_decomp``), so the row is synthetic: vanilla
    Wooper's [Damp, Water Absorb] on Kaizo Wooper (Water/Ground, Water Absorb only)."""
    t = kaizo_tables
    abilities = np.asarray(t.species_abilities_decomp)
    assert not (abilities[: t.species_form_row.shape[0]] == DAMP).any()
    assert np.asarray(tables.species_abilities_decomp)[WOOPER].tolist() == [DAMP, WATER_ABSORB]
    assert names.species("Wooper") == WOOPER
    t2 = _with_abilities(t, WOOPER, [DAMP, WATER_ABSORB])
    moves = (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK)
    s = _with_player(t2, _state(t2, kaizo_roark_party, BONSLY, moves), WOOPER)
    damage = _damage(t2, s)
    assert damage[2] > damage[3] > 0 and max(damage) < 400
    jump = ((1, 6374),)
    # Guess 0, Damp: :305 ScoreMinus10.
    assert _two_flag_scores(t2, s, 0, jump) == [100, 90, 100, 99]
    # Guess 1, Water Absorb: Basic_CheckLastMon, Bonsly has Cranidos: no change. Brick Break
    # and Accelerock take Basic_CheckWaterAbsorption (:74 -> :86), not Water: no change.
    assert _two_flag_scores(t2, s, 1, jump) == [100, 100, 100, 99]
    # The ability is known (ai_ability_known): Wooper's true slot 0, Damp, whatever the guess.
    known = s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))
    assert _two_flag_scores(t2, known, 1, jump) == [100, 90, 100, 99]


def test_selfdestruct_mold_breaker_skips_damp(kaizo_tables, kaizo_roark_party):
    """An attacker with Mold Breaker jumps to Basic_CheckLastMon (script.s:302-303) before
    the Damp read: Kaizo Cranidos in ability slot 1 (Mold Breaker)."""
    t = _with_abilities(kaizo_tables, WOOPER, [DAMP, WATER_ABSORB])
    party = list(kaizo_roark_party)
    party[CRANIDOS] = replace(party[CRANIDOS], ability_slot=1)
    assert int(np.asarray(t.species_abilities_decomp)[party[CRANIDOS].species, 1]) == MOLD_BREAKER
    s = _state(t, party, CRANIDOS, (SELFDESTRUCT, CRUNCH, 0, 0), bench=(BONSLY,))
    s = _with_player(t, s, WOOPER)
    for guess in (0, 1):
        assert _two_flag_scores(t, s, guess, ((0, 6374),)) == [100, 100, 0, 0]
        assert _two_flag_scores(t, s, guess) == [98, 100, 0, 0]


def test_selfdestruct_immune(kaizo_tables, kaizo_names, kaizo_roark_party):
    """:299: IMMUNE gives ScoreMinus10 and ends BASIC. Selfdestruct (Normal) into a Ghost
    type (battle_lib.c sTypeMatchupMultipliers, Normal x Ghost immune)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, (SELFDESTRUCT, 0, 0, 0))
    s = _named(t, n, s, "Gastly")
    assert _effectiveness(t, s, SELFDESTRUCT) == TypeMulti.IMMUNE
    assert _two_flag_scores(t, s, jumps=((0, 6374),)) == [90, 0, 0, 0]


# --- BASIC: Basic_CheckCanRecoverHP (script.s:511-517) -----------------------------------


def test_recover_at_full_hp(kaizo_tables, kaizo_roark_party):
    """Lileep [Sludge Bomb, Giga Drain, AncientPower, Recover]: at 100% HP Recover scores -8
    (IfHPPercentNotEqualTo ATTACKER 100, :513-514); one HP lower it is 99% (trainer_ai.c:641,
    integer division) and unchanged. Giga Drain is the highest damage, so the other two
    hits get -1 (script.s:6361-6362)."""
    t = kaizo_tables
    moves = (SLUDGE_BOMB, GIGA_DRAIN, ANCIENTPOWER, RECOVER)
    s = _state(t, kaizo_roark_party, LILEEP, moves)
    damage = _damage(t, s)
    assert damage[1] > max(damage[0], damage[2]) > 0 and max(damage) < 400
    for move in moves:
        assert _effectiveness(t, s, move) != TypeMulti.QUADRUPLE
    assert _two_flag_scores(t, s) == [99, 100, 99, 92]
    hurt = s.replace_hp(1, 0, int(s.max_hp[1, 0]) - 1)
    assert _two_flag_scores(t, hurt) == [99, 100, 99, 100]


# --- BASIC: Basic_CheckCannotPoison (script.s:519-547) -----------------------------------


@pytest.mark.parametrize(
    ("player", "change", "expected"),
    [
        ("Gible", None, 100),  # Dragon/Ground, Sand Veil/Rough Skin, no status: :547
        ("Mawile", None, 90),  # Steel type 1: :521-522
        ("Scizor", None, 90),  # Steel type 2: :524-525
        ("Tentacool", None, 90),  # Poison type 2: :524, :526
        ("Gible", "paralysis", 90),  # IfStatus DEFENDER MON_CONDITION_ANY: :545
        ("Gible", "safeguard", 90),  # IfSideCondition DEFENDER SAFEGUARD: :546
        ("Gible", "sun", 100),  # weather alone changes nothing without Leaf Guard / Hydration
    ],
)
def test_toxic_cannot_poison(
    kaizo_tables, kaizo_names, kaizo_roark_party, player, change, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, SHUCKLE, (TOXIC, 0, 0, 0), bench=(BONSLY,))
    if player != "Gible":
        s = _named(t, n, s, player)
    if change == "paralysis":
        s = s.with_status(0, 0, Status.PARALYSIS)
    elif change == "safeguard":
        s = s._replace(safeguard=s.safeguard.at[0].set(5))
    elif change == "sun":
        s = s._replace(weather=jnp.asarray(Weather.SUN, s.weather.dtype))
    assert _effectiveness(t, s, TOXIC) != TypeMulti.QUADRUPLE
    assert _two_flag_scores(t, s) == [expected, 0, 0, 0]


@pytest.mark.parametrize(
    ("player", "weather", "slot", "ability"),
    [
        ("Zangoose", Weather.NONE, 1, "Immunity"),  # :530
        ("Kadabra", Weather.NONE, 0, "Magic Guard"),  # :531
        ("Lickilicky", Weather.NONE, 0, "Poison Heal"),  # :532
        ("Grotle", Weather.SUN, 0, "Leaf Guard"),  # :534-536
        ("Whiscash", Weather.RAIN, 0, "Hydration"),  # :539-542
    ],
)
def test_toxic_ability_guess(
    kaizo_tables, kaizo_names, kaizo_roark_party, player, weather, slot, ability
):
    """``LoadBattlerAbility DEFENDER`` (:529, :539) guesses between the species' two
    abilities (trainer_ai.c:1191-1198): the guess of ``ability`` scores Toxic -10, the
    other ability does not. Leaf Guard needs sun and Hydration rain (LoadCurrentWeather,
    trainer_ai.c:1414-1439); without them the guess changes nothing."""
    t, n = kaizo_tables, kaizo_names
    species = n.species(player)
    assert int(np.asarray(t.species_abilities_decomp)[species, slot]) == c.ABILITY_IDS[ability]
    s = _state(t, kaizo_roark_party, SHUCKLE, (TOXIC, 0, 0, 0), bench=(BONSLY,))
    s = _with_player(t, s, species)
    assert _effectiveness(t, s, TOXIC) != TypeMulti.QUADRUPLE
    wet = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
    assert _two_flag_scores(t, wet, slot) == [90, 0, 0, 0]
    assert _two_flag_scores(t, wet, 1 - slot) == [100, 0, 0, 0]
    if weather != Weather.NONE:
        assert _two_flag_scores(t, s, slot) == [100, 0, 0, 0]


# --- BASIC: Basic_CheckAlreadyUnderLightScreen (script.s:549-552) ------------------------


def test_light_screen_up(kaizo_tables, kaizo_roark_party):
    """Corsola [Recover, Light Screen]: Light Screen on the AI's side (IfSideCondition
    ATTACKER) scores -8 (:551 ScoreMinus8); on the player's side it changes nothing. Corsola
    at full HP also gives Recover -8 (:513-514)."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, CORSOLA, (RECOVER, LIGHT_SCREEN, 0, 0), bench=(BONSLY,))
    assert _effectiveness(t, s, LIGHT_SCREEN) != TypeMulti.QUADRUPLE
    up = s._replace(light_screen=s.light_screen.at[1].set(5))
    assert _two_flag_scores(t, up) == [92, 92, 0, 0]
    assert _two_flag_scores(t, s) == [92, 100, 0, 0]
    theirs = s._replace(light_screen=s.light_screen.at[0].set(5))
    assert _two_flag_scores(t, theirs) == [92, 100, 0, 0]


# --- BASIC: Basic_CheckHighStatStage_Speed (script.s:363-372) ----------------------------

_CRANIDOS_MOVES = (HEAD_SMASH, ZEN_HEADBUTT, CRUNCH, ROCK_POLISH)


def _cranidos(t, party):
    s = _state(t, party, CRANIDOS, _CRANIDOS_MOVES, bench=(BONSLY,))
    # Inputs: Head Smash is no-calc (RECOIL_HALF, trainer_ai.c:44) and estimates 0; Crunch
    # out-damages Zen Headbutt; no move is QUADRUPLE against Gible.
    damage = _damage(t, s)
    assert damage[0] == 0 and damage[2] > damage[1] > 0 and max(damage) < 400
    for move in _CRANIDOS_MOVES:
        assert _effectiveness(t, s, move) != TypeMulti.QUADRUPLE
    return s


@pytest.mark.parametrize(
    ("speed", "trick_room", "expected"),
    [
        (0, 0, [100, 99, 100, 100]),
        (5, 0, [100, 99, 100, 100]),
        (6, 0, [100, 99, 100, 90]),  # IfStatStageEqualTo ATTACKER SPEED 12: :371
        (0, 3, [100, 99, 100, 90]),  # IfFieldConditionsMask TRICK_ROOM: :364
    ],
)
def test_rock_polish(kaizo_tables, kaizo_roark_party, speed, trick_room, expected):
    """Cranidos (Rock Head, not Simple: :366 jumps to _NoSimple) [Head Smash, Zen Headbutt,
    Crunch, Rock Polish]. Head Smash gets neither -1 nor a kill bonus (no-calc)."""
    t = kaizo_tables
    s = _cranidos(t, kaizo_roark_party)
    s = s._replace(
        stages=s.stages.at[1, Stat.SPE].set(speed),
        trick_room=jnp.asarray(trick_room, s.trick_room.dtype),
    )
    assert _two_flag_scores(t, s) == expected


@pytest.mark.parametrize(("speed", "expected"), [(2, 100), (3, 90)])
def test_rock_polish_simple(kaizo_tables, kaizo_roark_party, speed, expected):
    """A Simple attacker (synthetic Cranidos row [Simple, Mold Breaker]) scores -10 above
    decomp stage 8 (IfStatStageGreaterThan ATTACKER SPEED 8, :367): engine +3."""
    t = _with_abilities(kaizo_tables, kaizo_roark_party[CRANIDOS].species, [SIMPLE, MOLD_BREAKER])
    s = _state(t, kaizo_roark_party, CRANIDOS, (ROCK_POLISH, 0, 0, 0), bench=(BONSLY,))
    s = s._replace(stages=s.stages.at[1, Stat.SPE].set(speed))
    assert _two_flag_scores(t, s) == [expected, 0, 0, 0]


# --- BASIC: Basic_CheckCurse (script.s:679-698) ------------------------------------------

_SHUCKLE_MOVES = (TOXIC, ACCELEROCK, CURSE, SAND_TOMB)  # Roark's Shuckle


@pytest.mark.parametrize(
    ("atk", "def_", "expected"),
    [
        (0, 0, [100, 99, 100, 100]),
        (6, 0, [100, 99, 90, 100]),  # IfStatStageEqualTo ATTACKER ATTACK 12: :696 ScoreMinus10
        (0, 6, [100, 99, 92, 100]),  # IfStatStageEqualTo ATTACKER DEFENSE 12: :697 ScoreMinus8
        (6, 6, [100, 99, 90, 100]),  # the Attack test comes first
        (2, 2, [100, 99, 100, 100]),
    ],
)
def test_curse(kaizo_tables, kaizo_roark_party, atk, def_, expected):
    """Shuckle is Bug/Rock, not Ghost (:681-684 fall through), and Solid Rock, not Simple
    (:689 jumps to _NoSimple). Sand Tomb out-damages Accelerock (-1)."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, SHUCKLE, _SHUCKLE_MOVES, bench=(BONSLY,))
    assert np.asarray(s.types[1, 0]).tolist() == [6, 5]  # TYPE_BUG, TYPE_ROCK
    damage = _damage(t, s)
    assert damage[3] > damage[1] > 0 and max(damage) < 400
    for move in _SHUCKLE_MOVES:
        assert _effectiveness(t, s, move) != TypeMulti.QUADRUPLE
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(atk).at[1, Stat.DEF].set(def_))
    assert _two_flag_scores(t, s) == expected


@pytest.mark.parametrize(("atk", "def_", "expected"), [(2, 2, 100), (3, 0, 90), (0, 3, 90)])
def test_curse_simple(kaizo_tables, kaizo_roark_party, atk, def_, expected):
    """A Simple attacker: -10 when Attack or Defense is above decomp stage 8 (:690-691)."""
    t = _with_abilities(kaizo_tables, kaizo_roark_party[SHUCKLE].species, [SIMPLE, SIMPLE])
    s = _state(t, kaizo_roark_party, SHUCKLE, (CURSE, 0, 0, 0), bench=(BONSLY,))
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(atk).at[1, Stat.DEF].set(def_))
    assert _two_flag_scores(t, s) == [expected, 0, 0, 0]


def test_ghost_curse_refused(kaizo_tables, kaizo_names, kaizo_roark_party):
    """A Ghost-type Curse user takes Basic_CheckCurse_GhostType (:681-684 -> :700), which is
    not transcribed: refused. (``KAIZO_ALL_EXPERT``: KAIZO also refuses Curse for its Expert
    label until K3c Task 6.)"""
    t, n, all_expert = kaizo_tables, kaizo_names, KAIZO_ALL_EXPERT
    s = _state(t, kaizo_roark_party, SHUCKLE, _SHUCKLE_MOVES, bench=(BONSLY,))
    assert not any("Curse" in r for r in policy.ai_refusals(t, s, n, layout=all_expert))
    ghost = s._replace(types=s.types.at[1, 0, 1].set(7))  # TYPE_GHOST
    reasons = policy.ai_refusals(t, ghost, n, layout=all_expert)
    assert any("move Curse are not transcribed" in r for r in reasons), reasons


# --- EVAL_ATTACK (script.s:6353-6415) -----------------------------------------------------


def test_accelerock_kill_bonus_is_plus_4(kaizo_tables, kaizo_roark_party):
    """Ruling 8: Accelerock kills (IfCurrentMoveKills, :6358 -> :6387), its effect is HIT,
    not PRIORITY_1 (:6400), so it takes GoTo EvalAttack_ScorePlus4 (:6401 -> :6411): +4,
    not +6, whatever its +1 priority."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (ACCELEROCK, 0, 0, 0), hp=1)
    assert _damage(t, s)[0] >= 1
    assert _two_flag_scores(t, s) == [104, 0, 0, 0]


def test_priority_1_kill_bonus_is_plus_6(kaizo_tables, kaizo_roark_party):
    """The :6400 branch itself: a move whose effect is PRIORITY_1 (synthetic: Accelerock
    given effect 103, which is not transcribed, so only the two-flag scores are read) that
    kills takes EvalAttack_ScorePlus2 (:6408), which falls into ScorePlus4: +6."""
    t = kaizo_tables
    t103 = t._replace(move_ai_effect=t.move_ai_effect.at[ACCELEROCK].set(BattleEffect.PRIORITY_1))
    s = _state(t103, kaizo_roark_party, BONSLY, (ACCELEROCK, 0, 0, 0), hp=1)
    assert _two_flag_scores(t103, s) == [106, 0, 0, 0]


@pytest.mark.parametrize(("jump", "expected"), [(1, [100, 104, 0, 0]), (0, [98, 104, 0, 0])])
def test_selfdestruct_never_kills(kaizo_tables, kaizo_roark_party, jump, expected):
    """Selfdestruct against a 1-HP player: its real damage kills, but IfCurrentMoveKills
    computes no damage for a no-calc effect (trainer_ai.c:1539-1554), so it takes no kill
    bonus and walks EvalAttack_MaybeDeprioritize (:6365, :6374-6375) as usual. Accelerock
    in the next slot kills: +4."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (SELFDESTRUCT, ACCELEROCK, 0, 0), hp=1)
    assert int(ai_damage_for_roll(t, s, 1, SELFDESTRUCT, SIM_ROLL_MAX)) >= 1
    assert _damage(t, s)[0] == 0
    jumps = ((0, 6374),) if jump else ()
    assert _two_flag_scores(t, s, jumps=jumps) == expected


def test_head_smash_never_kills(kaizo_tables, kaizo_roark_party):
    """Head Smash (RECOIL_HALF, no-calc, trainer_ai.c:44) against a 1-HP player: no kill
    bonus and no -1 (AI_NO_COMPARISON_MADE); Crunch kills: +4."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, CRANIDOS, (HEAD_SMASH, CRUNCH, 0, 0), bench=(BONSLY,), hp=1)
    assert int(ai_damage_for_roll(t, s, 1, HEAD_SMASH, SIM_ROLL_MAX)) >= 1
    assert _two_flag_scores(t, s) == [100, 104, 0, 0]


@pytest.mark.parametrize(("jump", "expected"), [(0, 102), (1, 100)])
def test_status_move_quad_roll(kaizo_tables, kaizo_names, kaizo_roark_party, jump, expected):
    """EvalAttack_CheckQuadEffective (:6379) applies to status moves too: Rock Polish (Rock)
    into Moltres (Fire/Flying, Rock x2 each, battle_lib.c:2483, :2487) is QUADRUPLE, and the
    80 roll (:6383) gives +2 unless it jumps."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, CRANIDOS, (ROCK_POLISH, 0, 0, 0), bench=(BONSLY,))
    s = _named(t, n, s, "Moltres")
    assert _effectiveness(t, s, ROCK_POLISH) == TypeMulti.QUADRUPLE
    jumps = ((0, 6383),) if jump else ()
    assert _two_flag_scores(t, s, jumps=jumps) == [expected, 0, 0, 0]


# --- Exact against loose on the new paths -------------------------------------------------


def _loose_cases(t, party, names):
    sd = _state(t, party, BONSLY, (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK), ())
    toxic = _named(t, names, _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,)), "Zangoose")
    curse = toxic._replace(stages=toxic.stages.at[1, Stat.ATK].set(6))
    rain = _named(t, names, _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,)), "Whiscash")
    rain = rain._replace(weather=jnp.asarray(Weather.RAIN, rain.weather.dtype))
    lileep = _state(t, party, LILEEP, (SLUDGE_BOMB, GIGA_DRAIN, ANCIENTPOWER, RECOVER))
    corsola = _state(t, party, CORSOLA, (RECOVER, WATER_PULSE, LIGHT_SCREEN, ACCELEROCK))
    corsola = corsola._replace(light_screen=corsola.light_screen.at[1].set(5))
    cranidos = _cranidos(t, party)
    cranidos = cranidos._replace(stages=cranidos.stages.at[1, Stat.SPE].set(6))
    kill = _state(t, party, BONSLY, (SELFDESTRUCT, ACCELEROCK, 0, 0), hp=1)
    return {
        "selfdestruct_last": sd,
        "toxic_immunity_guess": toxic,
        "curse_plus_6": curse,
        "toxic_hydration_rain": rain,
        "recover_full": lileep,
        "light_screen_up": corsola,
        "rock_polish_plus_6": cranidos,
        "kill": kill,
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    [
        "selfdestruct_last",
        "toxic_immunity_guess",
        "curse_plus_6",
        "toxic_hydration_rain",
        "recover_full",
        "light_screen_up",
        "rock_polish_plus_6",
        "kill",
    ],
)
def test_new_paths_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, case):
    """On the traced states, the exact distribution (per-slot sites, one guess) equals the
    loose oracle (every IfRandomLessThan and LoadBattlerAbility DEFENDER its own draw), and
    so does each slot's score distribution: the new sites (:6374) and reads (:305, :529,
    :539) are read where the decomp reads them."""
    t = kaizo_tables
    s = _loose_cases(t, kaizo_roark_party, kaizo_names)[case]
    assert policy.ai_refusals(t, s, kaizo_names, layout=KAIZO_ALL_EXPERT) == []
    _assert_exact_equals_loose(t, s, KAIZO_ALL_EXPERT)
