"""The paths of Kaizo Roark's moves: BASIC and EVAL_ATTACK (K3c Task 5), EXPERT and RISKY
(K3c Task 6); docs/ai/kaizo_roark.md 5, the traces of section 13.

Decomp paths are relative to ``external/pokeplatinum``: ``script.s`` is
``src/battle/trainer_ai/script.s``, ``trainer_ai.c`` is ``src/battle/trainer_ai/trainer_ai.c``
and ``battle_lib.c`` is ``src/battle/battle_lib.c``.

The Task 5 traces (K1-K11) give the four move scores after the BASIC and EVAL_ATTACK flag
scripts only (``_two_flag_scores``); the Task 6 traces (E1-E9) give them after all four of
the Kaizo policy's flag scripts (``_all_flag_scores``). Each trace's script lines are in
kaizo_roark.md 13. Facts a trace takes as input (an effectiveness class, which damage
estimate is the highest, whether a move kills, which Pokemon is faster) are asserted first,
so a data surprise fails there and not in the score.
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
    KAIZO_EXPERT_TRANSCRIBED,
    READ_DRAWS,
    SIM_ROLL_MAX,
    SWITCH_NONE,
    AiOutcome,
)
from nuzline.engine.damage import ai_damage_for_roll
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.order import (
    COMPARE_SPEED_FASTER,
    COMPARE_SPEED_SLOWER,
    compare_speed,
    effective_speed,
)
from nuzline.engine.state import MOVES, make_state
from tests.ai.test_kaizo_outcome import _assert_exact_equals_loose, _bulky, _with_player

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
ROCK_HEAD = 69
WOOPER = 194
# Kaizo ids of moves the Task 6 traces add (names checked where used).
PROTECT, TRICK_ROOM, GYRO_BALL, NATURE_POWER = 182, 433, 360, 267
QUICK_CLAW = 217  # Kaizo item id


# --- States, outcomes, scores ------------------------------------------------------------


def _state(t, party, mon: int, moves, bench=(CRANIDOS,), players: int = 2, hp: int = 400):
    """Kaizo Roark's ``mon`` with ``moves`` (``bench`` behind it), against ``players`` copies
    of Roark's Gible as the player's party, the active one at ``hp``/``hp`` (no transcribed
    move kills 400)."""
    opp = [replace(party[mon], moves=tuple(moves), pp=None)] + [party[b] for b in bench]
    s = make_state(t, [party[GIBLE]] * players, opp)
    return _bulky(s, hp)


def _outcome(t, s, guess: int = 0, jumps=()) -> AiOutcome:
    """The per-slot outcome where the ``(slot, line)`` ``IfRandomLessThan`` reads of ``jumps``
    jump and no other does, and every guessing read takes ability slot ``guess``. The reads'
    order on each slot's path (E1 Ruling R4) is the loose oracle's (``LooseDraws``: every read
    in static order, its kind and line) restricted to the reads on the path; the speed tie,
    which the exact layout does not draw (Ruling 3), takes 0 in both, and so does a defender
    ability read that does not guess."""
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], s)
    bits = np.zeros((1, MOVES, c.LOOSE_READS), np.int8)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), KAIZO)
    kind, line = np.asarray(lo.kind)[0, 0], np.asarray(lo.line)[0, 0]  # static order
    roll = kind == c.ReadKind.ROLL
    # A LoadBattlerAbility DEFENDER draws only when the decision guesses (trainer_ai.c:1191-1198).
    ability = (kind == c.ReadKind.ABILITY) & bool(c.ability_guessed(t, s))
    for slot, at in jumps:
        bits[0, slot] |= (roll[slot] & (line[slot] == at)).astype(np.int8)
    bits[0] |= (ability * guess).astype(np.int8)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), KAIZO)
    live = np.asarray(lo.live)[0, 0]
    rolls = np.full((MOVES, KAIZO.slot_reads), READ_DRAWS[0], np.uint8)
    for slot in range(MOVES):
        on_path = np.flatnonzero(live[slot] & (roll[slot] | ability[slot]))
        for k, i in enumerate(on_path):
            # A jump draws below N; ability slot 0 is RandNext() & 1 set (trainer_ai.c:1195).
            taken = bits[0, slot, i] == 1 if roll[slot, i] else guess == 0
            rolls[slot, k] = READ_DRAWS[int(taken)]
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.asarray(0, jnp.int8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.asarray(0, jnp.int8),
        switch_stop=jnp.asarray(SWITCH_NONE, jnp.int8),
    )


@partial(jax.jit, static_argnames="flags")
def _flag_scores(t, s, o, flags: int = 2):
    ctx = c.new_context(t, s, o)
    return policy._run_flags(ctx, KAIZO, policy._flags(KAIZO)[:flags]).scores


def _two_flag_scores(t, s, guess: int = 0, jumps=()) -> list[int]:
    """The four move scores after BASIC and EVAL_ATTACK (trainer_ai.c:297-310), refusals
    not applied."""
    return [int(x) for x in _flag_scores(t, s, _outcome(t, s, guess, jumps))]


def _all_flag_scores(t, s, guess: int = 0, jumps=()) -> list[int]:
    """The four move scores after the Kaizo policy's four flag scripts, BASIC, EVAL_ATTACK,
    EXPERT and RISKY (trainer_ai.c:297-310), refusals not applied."""
    return [int(x) for x in _flag_scores(t, s, _outcome(t, s, guess, jumps), 4)]


def _speeds(s, player: int | None = None, opp: int | None = None):
    """``s`` with the actives' Speed stats set (``stats[..., 4]`` is Speed)."""
    if player is not None:
        s = s._replace(stats=s.stats.at[0, int(s.active[0]), Stat.SPE - 1].set(player))
    if opp is not None:
        s = s._replace(stats=s.stats.at[1, int(s.active[1]), Stat.SPE - 1].set(opp))
    return s


def _ai_compare(t, s) -> int:
    """``BattleSystem_CompareBattlerSpeed(the AI, the player, TRUE)`` with the tie bit 0
    (trainer_ai.c:1110; plan K3c Ruling 3), an input the traces assert."""
    return int(compare_speed(t, s, 1, 0, s.qc_roll))


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


def test_roark_moves_are_all_accepted(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Since K3c Task 6 every effect of Kaizo Roark's moves is transcribed with its Expert
    label: no opponent move of his party is refused, and the guard's pending set is empty."""
    t, n, party = kaizo_tables, kaizo_names, kaizo_roark_party
    s = make_state(t, [party[GIBLE]], list(party))
    reasons = [r for r in policy.ai_refusals(t, s, n) if "opponent slot" in r]
    assert reasons == []
    roark_labels = {e for _, e in ROARK_MOVES.values()} & set(KAIZO_EXPERT_LABELS)
    assert roark_labels <= KAIZO.expert_transcribed


def test_expert_label_table_is_the_dispatch():
    """``KAIZO_EXPERT_LABELS`` is read off ``Expert_Main``'s dispatch table (script.s:1628-1808,
    ``IfCurrentMoveEffectEqualTo BATTLE_EFFECT_<effect>, <label>``): each entry's line holds
    its effect and label, and every transcribed effect with an entry is either there or one
    whose label the vanilla transcription has (DEF_DOWN, DEF_DOWN_2, HIT_BEFORE_SWITCH,
    STEALTH_ROCK). Since K3c Task 6 every label of the table is transcribed."""
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
    assert KAIZO.expert_transcribed == KAIZO_EXPERT_TRANSCRIBED


def _dispatched(t, s, flag, layout) -> set[int]:
    """The effects ``flag`` (``policy._basic`` or ``policy._expert``) branches on under
    ``layout``: the ``IfCurrentMoveEffectEqualTo`` commands it runs, recorded while it runs
    once (the dispatch is decided before any value, so one state shows it)."""
    seen = set()
    original = c.if_current_move_effect_equal_to

    def record(ctx, effect):
        seen.add(int(effect))
        return original(ctx, effect)

    ctx = c.at_slot(c.new_context(t, s, _outcome(t, s)), 0)
    c.if_current_move_effect_equal_to = record
    try:
        flag(ctx, layout)
    finally:
        c.if_current_move_effect_equal_to = original
    return seen


def test_expert_transcribed_is_the_dispatch(
    kaizo_tables, kaizo_names, kaizo_roark_party, monkeypatch
):
    """Task 6 review I1. (a) The guard's transcribed set (``outcome.KAIZO_EXPERT_TRANSCRIBED``,
    a literal) is exactly the effects of ``KAIZO_EXPERT_LABELS`` that ``policy._expert``
    branches on, recorded from a run of it (and ``policy.EXPERT_DISPATCH`` is that set of
    branches). (b) An entry added to the table but not to the literal is refused: a synthetic
    Expert_Main entry for FLINCH_HIT (an effect that has none, script.s:1628-1808) refuses
    Cranidos's Zen Headbutt by its label, though FLINCH_HIT is transcribed. (c) Adding that
    entry to outcome.py's table literal leaves KAIZO_EXPERT_TRANSCRIBED unchanged."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, CRANIDOS, _CRANIDOS_MOVES, bench=(BONSLY,))
    branches = _dispatched(t, s, policy._expert, KAIZO)
    assert branches == policy.EXPERT_DISPATCH
    # (c) The literal does not follow the table: outcome.py's own source, with an entry added
    # to the KAIZO_EXPERT_LABELS literal (as a later task would), still leaves it out.
    from nuzline.ai import outcome as outcome_module

    source = Path(outcome_module.__file__).read_text(encoding="utf-8")
    anchor = "KAIZO_EXPERT_LABELS: dict[int, tuple[str, int]] = {\n"
    assert source.count(anchor) == 1
    grown = source.replace(anchor, anchor + "    31: ('Expert_Synthetic', 9999),\n")
    copy: dict = {"__name__": "outcome_with_a_new_entry"}
    exec(compile(grown, "outcome_with_a_new_entry", "exec"), copy)  # noqa: S102
    assert 31 in copy["KAIZO_EXPERT_LABELS"]
    assert copy["KAIZO_EXPERT_TRANSCRIBED"] == KAIZO_EXPERT_TRANSCRIBED
    # A table entry may lead its label (K3c Task 6 re-review B): an effect in the table that
    # _expert does not branch on is pending, refused by the guard, so only these two hold.
    assert KAIZO_EXPERT_TRANSCRIBED == set(KAIZO_EXPERT_LABELS) & branches
    assert KAIZO_EXPERT_TRANSCRIBED <= branches
    assert policy.ai_refusals(t, s, n) == []
    flinch = BattleEffect.FLINCH_HIT
    assert int(t.move_ai_effect[ZEN_HEADBUTT]) == flinch and flinch in KAIZO.transcribed
    table = {**KAIZO_EXPERT_LABELS, int(flinch): ("Expert_Synthetic", 9999)}
    monkeypatch.setattr(policy, "KAIZO_EXPERT_LABELS", table)
    with jax.disable_jit():  # _refused is jitted with the table as a trace-time constant
        reasons = policy.ai_refusals(t, s, n)
    line = (
        "ai scoring: opponent slot 0: move Zen Headbutt goes to Expert_Synthetic (Expert_Main, "
        "script.s:9999), whose script path is not transcribed yet (docs/ai/kaizo_roark.md 5)"
    )
    assert reasons == [line]


def test_dispatch_is_static_per_layout(kaizo_tables, kaizo_roark_party):
    """Task 6 review m1: the jump tables hold only the layout's effects, decided at trace
    time, so the vanilla program traces none of the Kaizo labels. Vanilla's transcribed moves
    have the effects HIT, DEF_DOWN, FLINCH_HIT, DEF_DOWN_2, HIT_BEFORE_SWITCH and STEALTH_ROCK
    (commands.MOVE_EFFECTS): BASIC branches on DEF_DOWN, DEF_DOWN_2 and STEALTH_ROCK (:145,
    :172, :284), EXPERT on those four with entries (:1642, :1671, :1706, :1805)."""
    from nuzline.ai.layout import VANILLA

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, CRANIDOS, _CRANIDOS_MOVES, bench=(BONSLY,))
    e = BattleEffect
    assert _dispatched(t, s, policy._basic, VANILLA) == {e.DEF_DOWN, e.DEF_DOWN_2, e.STEALTH_ROCK}
    vanilla_expert = {e.DEF_DOWN, e.DEF_DOWN_2, e.HIT_BEFORE_SWITCH, e.STEALTH_ROCK}
    assert _dispatched(t, s, policy._expert, VANILLA) == vanilla_expert


def test_basic_label_table_is_the_dispatch():
    """K3c Task 6 re-review A: ``policy._BASIC_LABELS`` is read off ``Basic_ScoreMoveEffect``
    (script.s:133-286, ``IfCurrentMoveEffectEqualTo BATTLE_EFFECT_<effect>, <label>``).
    - Every row's effect has an entry there, its function's label (the first name of its
      docstring) is the entry's label, and the rows are in the entries' line order.
    - Every effect a layout transcribes (``policy._active_effects``) that has an entry has a
      row. So a new transcribed effect with a BASIC label (Sandstorm :200, Will-O-Wisp's
      STATUS_BURN :231, ...) fails here until its label is written."""
    from nuzline.ai.layout import LAYOUTS

    script = DECOMP / "src/battle/trainer_ai/script.s"
    effects = DECOMP / "generated/move_battle_effects.txt"
    if not script.exists() or not effects.exists():
        pytest.skip("decomp not present")
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[132] == "Basic_ScoreMoveEffect:" and lines[285].split() == ["PopOrEnd"]
    names = effects.read_text(encoding="utf-8").split()
    dispatch = {}
    for number in range(134, 286):
        words = lines[number - 1].replace(",", " ").split()
        assert words[:1] == ["IfCurrentMoveEffectEqualTo"], number
        dispatch.setdefault(names.index(words[1]), (words[2], number))
    rows = [(int(effect), label) for effect, label in policy._BASIC_LABELS]
    for effect, label in rows:
        assert effect in dispatch, effect
        assert label.__doc__.split("``")[1] == dispatch[effect][0], (effect, label.__name__)
    assert [dispatch[e][1] for e, _ in rows] == sorted(dispatch[e][1] for e, _ in rows)
    for layout in LAYOUTS:
        needed = policy._active_effects(layout) & set(dispatch)
        assert needed <= {e for e, _ in rows}, (layout.name, needed - {e for e, _ in rows})


def test_expert_guard_refuses_an_untranscribed_label(
    kaizo_tables, kaizo_names, kaizo_roark, kaizo_roark_party, monkeypatch
):
    """The guard stays for a future effect (coordinator ruling, K3c Task 5): under a layout
    whose Expert set lacks RESTORE_HALF_HP (synthetic), Kaizo Roark (flags 23) with a
    Recover user is refused by name of Expert_Recovery (Expert_Main's entry, script.s:1652),
    never scored without it; under the real layout coverage refuses nothing of his AI."""
    from nuzline.ai import layout as layout_module
    from nuzline.engine.coverage import check_coverage

    t, n = kaizo_tables, kaizo_names
    opp = [replace(kaizo_roark_party[CORSOLA], moves=(RECOVER, ACCELEROCK, 0, 0), pp=None)]
    player = [kaizo_roark_party[GIBLE]]
    line = (
        "ai scoring: opponent slot 0: move Recover goes to Expert_Recovery (Expert_Main, "
        "script.s:1652), whose script path is not transcribed yet (docs/ai/kaizo_roark.md 5)"
    )
    report = check_coverage(t, n, player, kaizo_roark, opp, Weather.NONE)
    assert not any("AI_FLAG" in m or "ai scoring" in m for m in report.missing), report.missing
    recover = BattleEffect.RESTORE_HALF_HP
    closed = KAIZO._replace(expert_transcribed=KAIZO.expert_transcribed - {recover})
    monkeypatch.setattr(layout_module, "LAYOUTS", (layout_module.VANILLA, closed))
    report = check_coverage(t, n, player, kaizo_roark, opp, Weather.NONE)
    assert any(line in m for m in report.missing), report.missing


def test_battle_effect_ids_are_the_decomps():
    """``commands.BattleEffect``: line n + 1 of generated/move_battle_effects.txt is
    ``BATTLE_EFFECT_<name>``, id n."""
    path = DECOMP / "generated/move_battle_effects.txt"
    if not path.exists():
        pytest.skip("decomp not present")
    lines = path.read_text(encoding="utf-8").split()
    for member in BattleEffect:
        assert lines[member.value] == f"BATTLE_EFFECT_{member.name}", member


def test_kaizo_refusals_use_the_kaizo_names(names, kaizo_tables, kaizo_names, kaizo_roark_party):
    """K3c final review: a refusal message without the caller's names names the move in
    the tables' own game, never in the shipped vanilla tables (Kaizo renames ids)."""
    t = kaizo_tables
    hurricane = 140
    assert names.move_names[hurricane] == "Barrage"
    assert kaizo_names.move_names[hurricane] == "Hurricane"
    s = _state(t, kaizo_roark_party, BONSLY, [STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, hurricane])
    reasons = policy.ai_refusals(t, s)
    assert reasons == policy.ai_refusals(t, s, kaizo_names), reasons
    assert any("Hurricane" in r for r in reasons) and not any("Barrage" in r for r in reasons)
    with pytest.raises(policy.UnsupportedAi, match="Hurricane"):
        policy.ai_action_distribution(t, s)


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


def test_toxic_poison_type_1(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Task 5 review m4: Poison as the defender's type 1 (script.s:521, :523), with a type 2
    that is neither Steel nor Poison, so only the :523 compare scores -10: Kaizo Zubat
    [Poison, Flying], one ability (Inner Focus), so no guess."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, SHUCKLE, (TOXIC, 0, 0, 0), bench=(BONSLY,))
    s = _named(t, n, s, "Zubat")
    assert np.asarray(s.types[0, 0]).tolist() == [3, 2]  # TYPE_POISON, TYPE_FLYING
    assert _effectiveness(t, s, TOXIC) != TypeMulti.QUADRUPLE
    assert _two_flag_scores(t, s) == [90, 0, 0, 0]


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
    """A Ghost-type Curse user takes Basic_CheckCurse_GhostType (:681-684 -> :700) and
    Expert_Curse_GhostCheckHP (:3467-3470 -> :3495), which are not transcribed: refused by
    their names (Task 5 review m1), and only for that move."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, SHUCKLE, _SHUCKLE_MOVES, bench=(BONSLY,))
    assert policy.ai_refusals(t, s, n) == []
    ghost = s._replace(types=s.types.at[1, 0, 1].set(7))  # TYPE_GHOST
    reasons = [r for r in policy.ai_refusals(t, ghost, n) if "opponent slot" in r]
    line = (
        "ai scoring: opponent slot 0: move Curse by a Ghost type goes to "
        "Basic_CheckCurse_GhostType (script.s:700) and Expert_Curse_GhostCheckHP "
        "(script.s:3495), which are not transcribed (docs/ai/kaizo_roark.md 5.1)"
    )
    assert reasons == [line]


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


# --- EXPERT (script.s:1623-1811) and RISKY (script.s:6521-6562): K3c Task 6 ---------------
#
# The traces E1-E9 of kaizo_roark.md 13: the four scores after all four flag scripts.
# Speeds are the actives' Speed stats (no modifier applies): the player's Gible 20, Bonsly
# 12, Lileep 15, Corsola 19, Shuckle 11, Cranidos 28; each trace asserts its compare.

_SD_ONLY_EXPERT = ((0, 6374), (0, 6530))  # EvalAttack's -2 and Risky's +2 not taken
_LILEEP_MOVES = (SLUDGE_BOMB, GIGA_DRAIN, ANCIENTPOWER, RECOVER)
_RESISTED = (TypeMulti.IMMUNE, TypeMulti.HALF, TypeMulti.QUARTER)


def _bonsly_explosion(t, party):
    """Bonsly [Selfdestruct, Brick Break], Cranidos behind it, against two Gibles: the
    attacker has another Pokemon, so Basic_CheckLastMon ends with no change (:311 -> :318);
    Brick Break is the only damage estimate and kills nothing."""
    s = _state(t, party, BONSLY, (SELFDESTRUCT, BRICK_BREAK, 0, 0))
    for move in (SELFDESTRUCT, BRICK_BREAK):
        assert _effectiveness(t, s, move) == TypeMulti.BASE
    damage = _damage(t, s)
    assert damage[0] == 0 and 0 < damage[1] < 400
    return s


@pytest.mark.parametrize(
    ("hp", "player_speed", "evasion", "claw", "jumps", "expected"),
    [
        # 100% (44/44), slower: :1863 falls through, :1864 jumps to :1868; above 50%:
        # Expert_Explosion_TryScoreMinus1, -1 unless :1880 jumps.
        (44, None, 0, False, (), 99),
        (44, None, 0, False, ((0, 1880),), 100),
        # 100%, faster (player Speed 10): :1865, GoTo ScoreMinus3 unless it jumps.
        (44, 10, 0, False, (), 97),
        (44, 10, 0, False, ((0, 1865),), 100),
        # The same with the player's Quick Claw activated: a lone claw goes first
        # (battle_lib.c:1408-1411), so the AI is SLOWER: -1 as at 100% slower.
        (44, 10, 0, True, (), 99),
        # 70% (31/44): :1863 jumps to :1868; above 50%: -1.
        (31, None, 0, False, (), 99),
        # 50% (22/44): :1869 falls through, +1 unless :1870 jumps; above 30%: End (:1874).
        (22, None, 0, False, (), 101),
        (22, None, 0, False, ((0, 1870),), 100),
        # 29% (13/44): +1 (:1871), then 30% or less: +1 unless :1875 jumps (:1875-1876).
        (13, None, 0, False, (), 102),
        (13, None, 0, False, ((0, 1870),), 101),
        (13, None, 0, False, ((0, 1870), (0, 1875)), 100),
        # The player's evasion +1 (decomp 7): -1 (:1856-1857), then as at 100% slower.
        (44, None, 1, False, (), 98),
        # +4 (decomp 10): -1, then -1 unless :1859 jumps (:1858-1860).
        (44, None, 4, False, (), 97),
        (44, None, 4, False, ((0, 1859),), 98),
    ],
)
def test_explosion(
    kaizo_tables, kaizo_names, kaizo_roark_party, hp, player_speed, evasion, claw, jumps, expected
):
    """E1, Expert_Explosion (script.s:1841-1884), with :6374 and :6530 jumping so the
    Selfdestruct score moves by Expert alone. Brick Break: Expert_BrickBreak without screens
    (:4561-4563): 100."""
    t, n = kaizo_tables, kaizo_names
    assert n.item_names[QUICK_CLAW] == "Quick Claw"
    s = _bonsly_explosion(t, kaizo_roark_party)
    s = _speeds(s, player_speed).replace_hp(1, 0, hp)
    s = s._replace(stages=s.stages.at[0, Stat.EVA].set(evasion))
    if claw:
        s = s._replace(
            item=s.item.at[0, 0].set(QUICK_CLAW), qc_roll=jnp.asarray([1, 0], s.qc_roll.dtype)
        )
    slower = player_speed is None or claw
    assert _ai_compare(t, s) == (COMPARE_SPEED_SLOWER if slower else COMPARE_SPEED_FASTER)
    assert _all_flag_scores(t, s, jumps=_SD_ONLY_EXPERT + jumps) == [expected, 100, 0, 0]


def test_explosion_all_flags(kaizo_tables, kaizo_roark_party):
    """E1 with no roll jumping: Selfdestruct -2 (:6374-6375), -1 (:1880-1881), +2 (Risky
    :6530-6531): 99; with Risky's roll jumping, 97."""
    t = kaizo_tables
    s = _bonsly_explosion(t, kaizo_roark_party)
    assert _all_flag_scores(t, s) == [99, 100, 0, 0]
    assert _all_flag_scores(t, s, jumps=((0, 6530),)) == [97, 100, 0, 0]


@pytest.mark.parametrize(
    ("hp", "player_speed", "jumps", "expected"),
    [
        # 100% (44/44): Recover -8 (:513-514), then Expert :2649 -> :2659, -3.
        (44, None, (), [99, 100, 101, 89]),
        # AncientPower's Risky roll jumps (:6530): no +2.
        (44, None, ((2, 6530),), [99, 100, 99, 89]),
        # 97% (43/44), slower: :2650 -> :2663; not below 70%: -3 unless :2665 jumps.
        (43, None, (), [99, 100, 101, 97]),
        # :2665 jumps to :2669: no STEAL_STATUS_MOVE seen (:2670) -> :2673, +2 unless :2674
        # jumps.
        (43, None, ((3, 2665),), [99, 100, 101, 102]),
        (43, None, ((3, 2665), (3, 2674)), [99, 100, 101, 100]),
        # 50% (22/44), slower: below 70% (:2664) -> :2669 -> +2. Recover when slower.
        (22, None, (), [99, 100, 101, 102]),
        # 97%, faster (player Speed 10): -8 (:2651).
        (43, 10, (), [99, 100, 101, 92]),
    ],
)
def test_recover(kaizo_tables, kaizo_roark_party, hp, player_speed, jumps, expected):
    """E2, Lileep: Sludge Bomb and AncientPower are not the highest damage (-1, :6362).
    Giga Drain (Grass, STAB) is not resisted by Gible, so Expert_DrainMove ends (:1832).
    AncientPower gets Risky's +2 (:6530-6531) unless it jumps."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, LILEEP, _LILEEP_MOVES)
    damage = _damage(t, s)
    assert damage[1] > max(damage[0], damage[2]) > 0 and max(damage) < 400
    for move in _LILEEP_MOVES:
        assert _effectiveness(t, s, move) != TypeMulti.QUADRUPLE
    assert _effectiveness(t, s, GIGA_DRAIN) not in _RESISTED
    s = _speeds(s, player_speed).replace_hp(1, 0, hp)
    slower = player_speed is None
    assert _ai_compare(t, s) == (COMPARE_SPEED_SLOWER if slower else COMPARE_SPEED_FASTER)
    assert _all_flag_scores(t, s, jumps=jumps) == expected


@pytest.mark.parametrize(("jump", "expected"), [(0, 97), (1, 100)])
def test_giga_drain_resisted(kaizo_tables, kaizo_names, kaizo_roark_party, jump, expected):
    """E3, Expert_DrainMove (:1827-1836): Giga Drain into Zubat (Poison/Flying, each halves
    Grass, battle_lib.c sTypeMatchupMultipliers) is resisted: -3 unless :1835 jumps."""
    t, n = kaizo_tables, kaizo_names
    s = _named(t, n, _state(t, kaizo_roark_party, LILEEP, (GIGA_DRAIN, 0, 0, 0)), "Zubat")
    assert _effectiveness(t, s, GIGA_DRAIN) in _RESISTED
    assert 0 < _damage(t, s)[0] < 400
    jumps = ((0, 1835),) if jump else ()
    assert _all_flag_scores(t, s, jumps=jumps) == [expected, 0, 0, 0]


@pytest.mark.parametrize(
    ("moves", "shuckle_hp", "player_hp", "jumps", "expected"),
    [
        # Both above 50%: :2689 and :2694 jump; no SP_DEF_UP or PROTECT move: End (:2701).
        # Curse +3 (E7).
        (_SHUCKLE_MOVES, 39, 400, (), [100, 99, 103, 100]),
        # Shuckle 48% (19/39): -3 unless :2690 jumps.
        (_SHUCKLE_MOVES, 19, 400, (), [97, 99, 103, 100]),
        (_SHUCKLE_MOVES, 19, 400, ((0, 2690),), [100, 99, 103, 100]),
        # The player at 50% (200/400): -3 unless :2695 jumps.
        (_SHUCKLE_MOVES, 39, 200, (), [97, 99, 103, 100]),
        (_SHUCKLE_MOVES, 19, 200, (), [94, 99, 103, 100]),
        # No damaging move (Toxic and Curse have power 0): :2688 jumps past both HP checks.
        ((TOXIC, CURSE, 0, 0), 19, 200, (), [100, 103, 0, 0]),
    ],
)
def test_toxic(kaizo_tables, kaizo_roark_party, moves, shuckle_hp, player_hp, jumps, expected):
    """E4, Expert_ToxicLeechSeed (:2680-2708), Shuckle against Gible. Accelerock is
    out-damaged by Sand Tomb (-1, :6362)."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, SHUCKLE, moves, bench=(BONSLY,))
    power = [int(t.move_power[m]) for m in (TOXIC, ACCELEROCK, CURSE, SAND_TOMB)]
    assert power[0] == power[2] == 0 and power[1] > 0 and power[3] > 0
    if moves == _SHUCKLE_MOVES:
        damage = _damage(t, s)
        assert damage[3] > damage[1] > 0 and max(damage) < 200
    s = s.replace_hp(1, 0, shuckle_hp).replace_hp(0, 0, player_hp)
    assert _all_flag_scores(t, s, jumps=jumps) == expected


@pytest.mark.parametrize(("jump", "expected"), [(0, 102), (1, 100)])
def test_toxic_with_protect(kaizo_tables, kaizo_names, kaizo_roark_party, jump, expected):
    """E4: the attacker knows a PROTECT move (:2700; synthetic: Protect, untranscribed, so only
    Toxic's slot is read) -> :2703, +2 unless :2704 jumps. No Kaizo move has SP_DEF_UP
    (:2699), so the other read is not fixtured with data."""
    t, n = kaizo_tables, kaizo_names
    assert n.move_names[PROTECT] == "Protect"
    assert int(t.move_ai_effect[PROTECT]) == BattleEffect.PROTECT
    effects = np.asarray(t.move_ai_effect)
    assert not (effects == BattleEffect.SP_DEF_UP).any()
    s = _state(t, kaizo_roark_party, SHUCKLE, (TOXIC, PROTECT, CURSE, SAND_TOMB), bench=(BONSLY,))
    jumps = ((0, 2704),) if jump else ()
    assert _all_flag_scores(t, s, jumps=jumps)[0] == expected


@pytest.mark.parametrize(
    ("hp", "last", "jumps", "expected"),
    [
        # 100% (46/46): Recover -8 and -3 (E2). Light Screen: not below 90%, +1 unless :2718
        # jumps; the player's last move (none: MOVE_NONE, physical) is not special: End.
        (46, 0, (), [89, 101]),
        (46, 0, ((1, 2718),), [89, 100]),
        # The player's last move Flamethrower (special): +1 unless :2724 jumps.
        (46, FLAMETHROWER, (), [89, 102]),
        (46, FLAMETHROWER, ((1, 2724),), [89, 101]),
        # Nature Power: its own class, status (kaizo_roark.md 7.4), not the called move's.
        (46, NATURE_POWER, (), [89, 101]),
        # 86% (40/46): below 90% skips :2718. Recover, slower: not below 70%, -3 (:2666).
        (40, FLAMETHROWER, (), [97, 101]),
        # 47% (22/46): Light Screen -2 (:2716 -> :2728); Recover below 70% -> +2 (:2675).
        (22, FLAMETHROWER, (), [102, 98]),
    ],
)
def test_light_screen(kaizo_tables, kaizo_names, kaizo_roark_party, hp, last, jumps, expected):
    """E5, Expert_LightScreen (:2710-2732), Corsola [Recover, Light Screen] against Gible
    (Corsola slower: 19 against 20). LoadDefenderLastUsedMoveClass reads the player's last
    used move (trainer_ai.c:2050-2054); CLASS_SPECIAL is 1 (generated/move_classes.txt)."""
    t, n = kaizo_tables, kaizo_names
    assert (n.move_names[FLAMETHROWER], n.move_names[NATURE_POWER]) == (
        "Flamethrower",
        "Nature Power",
    )
    assert int(t.move_category[FLAMETHROWER]) == c.CLASS_SPECIAL
    assert int(t.move_category[NATURE_POWER]) == 2  # CLASS_STATUS
    s = _state(t, kaizo_roark_party, CORSOLA, (RECOVER, LIGHT_SCREEN, 0, 0), bench=(BONSLY,))
    assert _effectiveness(t, s, LIGHT_SCREEN) != TypeMulti.QUADRUPLE
    s = s.replace_hp(1, 0, hp)
    s = s._replace(last_move=s.last_move.at[0].set(last))
    assert _ai_compare(t, s) == COMPARE_SPEED_SLOWER
    assert _all_flag_scores(t, s, jumps=jumps) == [*expected, 0, 0]


@pytest.mark.parametrize(
    ("player", "ability_slot", "player_speed", "trick_room", "jumps", "expected"),
    [
        # Gible: Head Smash (Rock) is resisted by Ground: End (:6294). Rock Polish: Cranidos
        # (28) is faster: -3 (:2071).
        ("Gible", 0, None, 0, (), [100, 99, 100, 97]),
        # Rattata (Normal): not resisted; Cranidos's true ability Rock Head: +1 (:6297).
        ("Rattata", 0, None, 0, (), [101, 99, 100, 97]),
        # Cranidos in ability slot 1, Mold Breaker: neither Rock Head nor Magic Guard: no +1.
        ("Rattata", 1, None, 0, (), [100, 99, 100, 97]),
        # The player at Speed 30: Cranidos is slower, +3 unless :2075 jumps.
        ("Gible", 0, 30, 0, (), [100, 99, 100, 103]),
        ("Gible", 0, 30, 0, ((3, 2075),), [100, 99, 100, 100]),
        # Trick Room: BASIC -10 (:364); the compare reverses (battle_lib.c:1432-1439), so the
        # faster Cranidos is SLOWER: +3.
        ("Gible", 0, None, 3, (), [100, 99, 100, 93]),
    ],
)
def test_head_smash_and_rock_polish(
    kaizo_tables,
    kaizo_names,
    kaizo_roark_party,
    player,
    ability_slot,
    player_speed,
    trick_room,
    jumps,
    expected,
):
    """E6, Expert_RecoilMove (:6289-6305) and Expert_StatusSpeedUp (:2066-2079), Cranidos
    [Head Smash, Zen Headbutt, Crunch, Rock Polish]. Head Smash is no-calc (E: K8); Crunch
    out-damages Zen Headbutt (-1)."""
    t, n = kaizo_tables, kaizo_names
    party = list(kaizo_roark_party)
    party[CRANIDOS] = replace(party[CRANIDOS], ability_slot=ability_slot)
    ability = int(np.asarray(t.species_abilities_decomp)[party[CRANIDOS].species, ability_slot])
    assert ability == (ROCK_HEAD, MOLD_BREAKER)[ability_slot]
    s = _state(t, party, CRANIDOS, _CRANIDOS_MOVES, bench=(BONSLY,))
    if player != "Gible":
        s = _named(t, n, s, player)
    s = _speeds(s, player_speed)
    s = s._replace(trick_room=jnp.asarray(trick_room, s.trick_room.dtype))
    damage = _damage(t, s)
    assert damage[0] == 0 and damage[2] > damage[1] > 0 and max(damage) < 400
    for move in _CRANIDOS_MOVES:
        assert _effectiveness(t, s, move) != TypeMulti.QUADRUPLE
    assert (_effectiveness(t, s, HEAD_SMASH) in _RESISTED) == (player == "Gible")
    slower = player_speed is not None or trick_room
    assert _ai_compare(t, s) == (COMPARE_SPEED_SLOWER if slower else COMPARE_SPEED_FASTER)
    assert _all_flag_scores(t, s, jumps=jumps) == expected


@pytest.mark.parametrize(
    ("side", "screen", "expected"),
    [(None, None, 100), (0, "reflect", 101), (0, "light_screen", 101), (1, "reflect", 100)],
)
def test_brick_break_screens(kaizo_tables, kaizo_roark_party, side, screen, expected):
    """E7, Expert_BrickBreak (:4559-4568): Reflect or Light Screen on the player's side
    (IfSideCondition DEFENDER) gives +1; a screen on the AI's own side does not. Accelerock
    is out-damaged (-1)."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (BRICK_BREAK, ACCELEROCK, 0, 0))
    if screen is not None:
        s = s._replace(**{screen: getattr(s, screen).at[side].set(5)})
    damage = _damage(t, s)
    assert damage[0] > damage[1] > 0 and max(damage) < 400
    assert _all_flag_scores(t, s) == [expected, 99, 0, 0]


@pytest.mark.parametrize(
    ("def_", "expected"),
    [
        # Defense 0 (decomp 6): +1 (:3481-3482), +1 (:3486-3487), +1 (:3491-3492).
        (0, 103),
        # +1 (decomp 7): :3490 (7 > 6) skips the third.
        (1, 102),
        # +2 (decomp 8): :3485 (8 > 7) leaves only the coin. Curse at +2 Defense.
        (2, 101),
        # +3 (decomp 9): :3471 tests > 9, so the coin still runs.
        (3, 101),
        # +4 (decomp 10): :3471 ends Expert_Curse.
        (4, 100),
        # +6 (decomp 12): BASIC -8 (:697), and Expert_Curse ends.
        (6, 92),
    ],
)
def test_curse_defense_stage(kaizo_tables, kaizo_roark_party, def_, expected):
    """E8, Expert_Curse (:3452-3500), Shuckle (Bug/Rock, not Ghost: :3467-3470 fall
    through), knowing neither Gyro Ball nor Trick Room (:3472-3474 -> :3480)."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, SHUCKLE, _SHUCKLE_MOVES, bench=(BONSLY,))
    s = s._replace(stages=s.stages.at[1, Stat.DEF].set(def_))
    assert _all_flag_scores(t, s) == [100, 99, expected, 100]


@pytest.mark.parametrize("known", ["Trick Room", "Gyro Ball"])
@pytest.mark.parametrize(("jumps", "expected"), [((), 104), (((2, 3477),), 102)])
def test_curse_gyro_ball_or_trick_room(
    kaizo_tables, kaizo_names, kaizo_roark_party, known, jumps, expected
):
    """E8: IfMoveKnown ATTACKER Gyro Ball or Trick Room (:3472-3473; synthetic, in slot 1,
    untranscribed, so only Curse's slot is read) -> :3476: +1 unless the 32 roll jumps, and a
    jump skips the coin too (:3477 -> :3484). Then +1, +1 at Defense 0."""
    t, n = kaizo_tables, kaizo_names
    move = {"Trick Room": TRICK_ROOM, "Gyro Ball": GYRO_BALL}[known]
    assert n.move_names[move] == known and c.MOVE_IDS[known] == move
    s = _state(t, kaizo_roark_party, SHUCKLE, (TOXIC, move, CURSE, SAND_TOMB), bench=(BONSLY,))
    assert _all_flag_scores(t, s, jumps=jumps)[2] == expected


@pytest.mark.parametrize(
    ("mon", "moves", "slot"),
    [(BONSLY, (SELFDESTRUCT, BRICK_BREAK, 0, 0), 0), (LILEEP, _LILEEP_MOVES, 2)],
)
@pytest.mark.parametrize("jump", [0, 1])
def test_risky(kaizo_tables, kaizo_roark_party, mon, moves, slot, jump):
    """E9, Risky_Main (:6521-6534): LoadCurrentMoveEffect, IfLoadedNotInTable
    Risky_RiskyEffects. Selfdestruct (HALVE_DEFENSE, :6538) and AncientPower
    (RAISE_ALL_STATS_HIT, :6550) get +2 unless :6530 jumps; no other effect of these movesets
    is in the table. Measured against the three flags before it."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, mon, moves)
    jumps = ((slot, 6530),) if jump else ()
    three = [int(x) for x in _flag_scores(t, s, _outcome(t, s, jumps=jumps), 3)]
    four = _all_flag_scores(t, s, jumps=jumps)
    delta = [0] * MOVES
    delta[slot] = 0 if jump else 2
    assert [b - a for a, b in zip(three, four, strict=True)] == delta


def test_risky_is_part_of_the_policy(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Task 5 review m2: RISKY is part of the Kaizo policy itself (``policy._flags``,
    ``layout.risky``), not only of coverage, so a caller of ``ai_refusals`` and ``ai_scores``
    alone never scores Selfdestruct or AncientPower without Risky's roll (:6530)."""
    from nuzline.ai.layout import VANILLA

    t, n = kaizo_tables, kaizo_names
    names = [getattr(f, "func", f).__name__ for f in policy._flags(KAIZO)]
    assert names == ["_basic", "_eval_attack", "_expert", "_risky"]
    assert len(policy._flags(VANILLA)) == 3
    cases = (
        (BONSLY, (SELFDESTRUCT, BRICK_BREAK, 0, 0), 0, [99, 100, 0, 0], 97),
        (LILEEP, _LILEEP_MOVES, 2, [99, 100, 101, 89], 99),
    )
    for mon, moves, slot, plain, rolled in cases:
        s = _state(t, kaizo_roark_party, mon, moves)
        assert policy.ai_refusals(t, s, n) == []
        assert [int(x) for x in policy.ai_scores(t, s, _outcome(t, s))] == plain
        scores = policy.ai_scores(t, s, _outcome(t, s, jumps=((slot, 6530),)))
        assert int(scores[slot]) == rolled


# --- Exact against loose on the new paths -------------------------------------------------


def _loose_cases(t, party, names):
    sd = _state(t, party, BONSLY, (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK), ())
    toxic = _named(t, names, _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,)), "Zangoose")
    curse = toxic._replace(stages=toxic.stages.at[1, Stat.ATK].set(6))
    rain = _named(t, names, _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,)), "Whiscash")
    rain = rain._replace(weather=jnp.asarray(Weather.RAIN, rain.weather.dtype))
    lileep = _state(t, party, LILEEP, _LILEEP_MOVES)
    corsola = _state(t, party, CORSOLA, (RECOVER, WATER_PULSE, LIGHT_SCREEN, ACCELEROCK))
    corsola = corsola._replace(light_screen=corsola.light_screen.at[1].set(5))
    cranidos = _cranidos(t, party)
    cranidos = cranidos._replace(stages=cranidos.stages.at[1, Stat.SPE].set(6))
    kill = _state(t, party, BONSLY, (SELFDESTRUCT, ACCELEROCK, 0, 0), hp=1)
    # K3c Task 6: states on the Expert paths with their rolls reached.
    explosion = _bonsly_explosion(t, party).replace_hp(1, 0, 13)
    evasion = _bonsly_explosion(t, party)
    evasion = evasion._replace(stages=evasion.stages.at[0, Stat.EVA].set(4))
    toxic_low = _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,)).replace_hp(1, 0, 19)
    toxic_low = toxic_low.replace_hp(0, 0, 200)
    screen = _state(t, party, CORSOLA, (RECOVER, WATER_PULSE, LIGHT_SCREEN, ACCELEROCK))
    screen = screen._replace(last_move=screen.last_move.at[0].set(FLAMETHROWER))
    recover = _state(t, party, LILEEP, _LILEEP_MOVES).replace_hp(1, 0, 43)
    curse_def = _state(t, party, SHUCKLE, _SHUCKLE_MOVES, (BONSLY,))
    curse_def = curse_def._replace(stages=curse_def.stages.at[1, Stat.DEF].set(1))
    drain = _named(t, names, _state(t, party, LILEEP, _LILEEP_MOVES), "Zubat")
    head_smash = _named(t, names, _cranidos(t, party), "Rattata")
    brick = _state(t, party, BONSLY, (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK))
    brick = brick._replace(reflect=brick.reflect.at[0].set(5))
    return {
        "selfdestruct_last": sd,
        "toxic_immunity_guess": toxic,
        "curse_plus_6": curse,
        "toxic_hydration_rain": rain,
        "recover_full": lileep,
        "light_screen_up": corsola,
        "rock_polish_plus_6": cranidos,
        "kill": kill,
        "explosion_low_hp": explosion,
        "explosion_evasion_4": evasion,
        "toxic_low_hp": toxic_low,
        "light_screen_special": screen,
        "recover_slower": recover,
        "curse_defense_1": curse_def,
        "giga_drain_resisted": drain,
        "head_smash_rock_head": head_smash,
        "brick_break_reflect": brick,
    }


_LOOSE_CASES = [
    "selfdestruct_last",
    "toxic_immunity_guess",
    "curse_plus_6",
    "toxic_hydration_rain",
    "recover_full",
    "light_screen_up",
    "rock_polish_plus_6",
    "kill",
    "explosion_low_hp",
    "explosion_evasion_4",
    "toxic_low_hp",
    "light_screen_special",
    "recover_slower",
    "curse_defense_1",
    "giga_drain_resisted",
    "head_smash_rock_head",
    "brick_break_reflect",
]


@pytest.mark.slow
@pytest.mark.parametrize("case", _LOOSE_CASES)
def test_new_paths_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, case):
    """On the traced states, the exact distribution (per-slot reads in path order) equals the
    loose oracle (every IfRandomLessThan and LoadBattlerAbility DEFENDER its own draw), and
    so does each slot's score distribution: the rolls of Tasks 5 (:6374) and 6 (the Expert
    labels' and :6530) and the reads (:305, :529, :539) are read where the decomp reads
    them. The real KAIZO layout accepts every state."""
    t = kaizo_tables
    s = _loose_cases(t, kaizo_roark_party, kaizo_names)[case]
    assert policy.ai_refusals(t, s, kaizo_names) == []
    _assert_exact_equals_loose(t, s, KAIZO)


def _equal_speed_cases(t, party):
    """``(state, slot)``: equal Speed at each compare on Roark's paths (script.s:1864,
    :2070, :2650), the slot whose label reads it; and the Quick Claw case."""
    explosion = _bonsly_explosion(t, party)  # 100%: :1863 falls through to :1864
    claw = _speeds(explosion, 10)._replace(
        item=explosion.item.at[0, 0].set(QUICK_CLAW),
        qc_roll=jnp.asarray([1, 0], explosion.qc_roll.dtype),
    )
    recover = _state(t, party, LILEEP, _LILEEP_MOVES).replace_hp(1, 0, 43)  # :2649 no
    return {
        "explosion_1864": (_speeds(explosion, 12), 0),
        "speed_up_2070": (_speeds(_cranidos(t, party), 28), 3),
        "recovery_2650": (_speeds(recover, 15), 3),
        "explosion_quick_claw": (claw, 0),
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", ["explosion_1864", "speed_up_2070", "recovery_2650", "explosion_quick_claw"]
)
def test_exact_equals_loose_at_equal_speed(kaizo_tables, kaizo_names, kaizo_roark_party, case):
    """Carried from Task 4 (Ruling 3): at equal speed CompareBattlerSpeed draws RandNext() & 1
    and gives FASTER or TIE, never SLOWER (battle_lib.c:1401-1447), so the exact layout drops
    the draw. The loose oracle draws it (ReadKind.SPEED_TIE) at each of the three compares,
    and the two agree. The Quick Claw case: Bonsly is faster, but the player's lone activated
    claw makes it SLOWER (:1408-1411) before any speed compare, so no tie is drawn."""
    t, n = kaizo_tables, kaizo_names
    s, slot = _equal_speed_cases(t, kaizo_roark_party)[case]
    tied = case != "explosion_quick_claw"
    assert (int(effective_speed(t, s, 0)) == int(effective_speed(t, s, 1))) == tied
    assert policy.ai_refusals(t, s, n) == []
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], s)
    reads, _, kind = policy._loose_reads(t, batch, KAIZO)
    assert any(kind[slot, r] == c.ReadKind.SPEED_TIE for r in reads[slot]) == tied
    _assert_exact_equals_loose(t, s, KAIZO)
