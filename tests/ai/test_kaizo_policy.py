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
from nuzline.ai.outcome import (
    KAIZO_EXPERT_LABELS,
    KAIZO_EXPERT_TRANSCRIBED,
    READ_DRAWS,
    SIM_ROLL_MAX,
    SWITCH_NONE,
    AiOutcome,
    no_calc_draws,
)
from nuzline.engine import damage
from nuzline.engine.damage import ai_damage_for_roll
from nuzline.engine.ids import Stat, Status, Weather
from nuzline.engine.order import (
    COMPARE_SPEED_FASTER,
    COMPARE_SPEED_SLOWER,
    compare_speed,
    effective_speed,
)
from nuzline.engine.state import MOVES, make_state
from tests.ai.fixtures import know_ability
from tests.ai.test_kaizo_outcome import (
    KAIZO,
    KAIZO_ALL,
    _assert_exact_equals_loose,
    _bulky,
    _with_player,
    layout_for,
)
from tests.engine.helpers import as_species

DECOMP = Path(__file__).resolve().parents[2] / "external/pokeplatinum"

# Kaizo Roark's party order and move ids (data/intermediate_kaizo/trainers.json, trainer 246;
# moves.json), checked by test_roark_moves_and_effects.
BONSLY, LILEEP, GIBLE, CORSOLA, SHUCKLE, CRANIDOS = range(6)

# The flag sets of the batches' exact-equals-loose sweeps. The fast tier runs their union
# (0x11F: BASIC, EVAL_ATTACK, EXPERT, SETUP_FIRST_TURN, RISKY, CHECK_HP), which traces every
# label either set does; each set alone, the per-set read bound and order, runs in the slow
# tier (owed test work after K3c/E1: one compile instead of two per sweep).
SWEEP_FLAGS = [
    pytest.param(0x17, marks=pytest.mark.slow),
    pytest.param(0x10F, marks=pytest.mark.slow),
    0x11F,
]
# The heavy exact-equals-loose sweeps: their first state at 0x11F is fast (one per family keeps
# exact == loose and no refusal in the fast tier); every state at every flag set is slow.
SWEEP_ONE = [
    pytest.param(0x11F, 1, id="0x11F-one"),
    pytest.param(0x17, None, id="0x17", marks=pytest.mark.slow),
    pytest.param(0x10F, None, id="0x10F", marks=pytest.mark.slow),
    pytest.param(0x11F, None, id="0x11F", marks=pytest.mark.slow),
]
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


def _outcome(t, s, guess: int = 0, jumps=(), layout=KAIZO, tie: int = 0) -> AiOutcome:
    """The per-slot outcome where the ``(slot, line)`` ``IfRandomLessThan`` reads of ``jumps``
    jump and no other does, every guessing read takes ability slot ``guess``, and every drawn
    speed tie (a FASTER or TIE compare at equal speed, E1 Ruling R8) takes bit ``tie`` (1:
    TIE). The reads' order on each slot's path (E1 Ruling R4) is the loose oracle's
    (``LooseDraws``: every read in static order, its kind and line) restricted to the reads on
    the path; a SLOWER compare's tie, which the exact layout does not draw (Ruling 3), takes 0
    in both, and so does a defender ability read that does not guess."""
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], s)
    bits = np.zeros((1, MOVES, c.LOOSE_READS), np.int8)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), layout)
    kind, line = np.asarray(lo.kind)[0, 0], np.asarray(lo.line)[0, 0]  # static order
    roll = kind == c.ReadKind.ROLL
    # A LoadBattlerAbility DEFENDER draws only when the decision guesses (trainer_ai.c:1191-1198).
    ability = (kind == c.ReadKind.ABILITY) & bool(c.ability_guessed(t, s))
    drawn = kind == c.ReadKind.SPEED_TIE_DRAWN
    for slot, at in jumps:
        bits[0, slot] |= (roll[slot] & (line[slot] == at)).astype(np.int8)
    bits[0] |= (ability * guess).astype(np.int8)
    bits[0] |= (drawn * tie).astype(np.int8)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), layout)
    live = np.asarray(lo.live)[0, 0]
    rolls = np.full((MOVES, layout.slot_reads), READ_DRAWS[0], np.uint8)
    for slot in range(MOVES):
        on_path = np.flatnonzero(live[slot] & (roll[slot] | ability[slot] | drawn[slot]))
        for k, i in enumerate(on_path):
            # A jump draws below N; ability slot 0 is RandNext() & 1 set (trainer_ai.c:1195);
            # a tie is RandNext() & 1 set (battle_lib.c:1445-1446).
            if roll[slot, i]:
                taken = bits[0, slot, i] == 1
            elif drawn[slot, i]:
                taken = tie == 1
            else:
                taken = guess == 0
            rolls[slot, k] = READ_DRAWS[int(taken)]
    return AiOutcome(
        sim_roll=jnp.full((MOVES,), SIM_ROLL_MAX, jnp.int8),
        ability_guess=jnp.asarray(0, jnp.int8),
        script_rolls=jnp.asarray(rolls),
        tiebreak=jnp.asarray(0, jnp.int8),
        switch_stop=jnp.asarray(SWITCH_NONE, jnp.int8),
        calc_draws=no_calc_draws((), rolls.shape[1]),
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
    return [int(x) for x in c._slot_damage(t, s, jnp.full((MOVES,), SIM_ROLL_MAX)).value[:, 0]]


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


@pytest.mark.fullset
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
    assert set(dispatch) & KAIZO_ALL.transcribed == set(KAIZO_EXPERT_LABELS) | old
    assert KAIZO_ALL.expert_transcribed == KAIZO_EXPERT_TRANSCRIBED


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


@pytest.mark.slow
@pytest.mark.fullset
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
    branches = _dispatched(t, s, policy._expert, KAIZO_ALL)
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


@pytest.mark.fullset
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


@pytest.mark.fullset
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
    # Each fight's layout comes from ai.layout.kaizo_layout (E1 Rulings R6, R7), which reads
    # layout.KAIZO through a cache: patch the base layout and clear the cache around it.
    base = layout_module.KAIZO
    closed = base._replace(expert_transcribed=base.expert_transcribed - {recover})
    monkeypatch.setattr(layout_module, "KAIZO", closed)
    layout_module._with.cache_clear()
    try:
        report = check_coverage(t, n, player, kaizo_roark, opp, Weather.NONE)
    finally:
        monkeypatch.undo()
        layout_module._with.cache_clear()
    assert any(line in m for m in report.missing), report.missing


@pytest.mark.fullset
def test_battle_effect_ids_are_the_decomps():
    """``commands.BattleEffect``: line n + 1 of generated/move_battle_effects.txt is
    ``BATTLE_EFFECT_<name>``, id n."""
    path = DECOMP / "generated/move_battle_effects.txt"
    if not path.exists():
        pytest.skip("decomp not present")
    lines = path.read_text(encoding="utf-8").split()
    for member in BattleEffect:
        name = c.BATTLE_EFFECT_DECOMP_NAMES.get(member, member.name)
        assert lines[member.value] == f"BATTLE_EFFECT_{name}", member


@pytest.mark.fullset
def test_kaizo_refusals_use_the_kaizo_names(names, kaizo_tables, kaizo_names, kaizo_roark_party):
    """K3c final review: a refusal message without the caller's names names the move in
    the tables' own game, never in the shipped vanilla tables (Kaizo renames ids)."""
    t = kaizo_tables
    # Kaizo's Mystical Fire (vanilla Bind's id, LOWER_SP_ATK_HIT): outside this test layout's
    # transcribed set (K3C_EFFECTS); Hurricane (Barrage's id) was the example until its effect
    # was transcribed (E2 batch 13, B13-23).
    mystical_fire = 20
    assert names.move_names[mystical_fire] == "Bind"
    assert kaizo_names.move_names[mystical_fire] == "Mystical Fire"
    moves = [STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, mystical_fire]
    s = _state(t, kaizo_roark_party, BONSLY, moves)
    reasons = policy.ai_refusals(t, s)
    assert reasons == policy.ai_refusals(t, s, kaizo_names), reasons
    assert any("Mystical Fire" in r for r in reasons) and not any("Bind" in r for r in reasons)
    with pytest.raises(policy.UnsupportedAi, match="Mystical Fire"):
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
    known = know_ability(t2, s)
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


@pytest.mark.parametrize(
    ("case", "basic", "expert"),
    [
        # Basic_CheckCurse_GhostType (script.s:700-705) / Expert_Curse_GhostCheckHP (:3495-3500)
        ("plain", 0, 0),  # not cursed, no Magic Guard; above 80 % HP the check ends
        ("hurt", 0, -1),  # AI at 50 % HP: -1
        ("cursed", -10, 0),  # :702 the player already cursed
        ("magic guard", -10, 0),  # :704 the player's Magic Guard
    ],
)
def test_ghost_curse(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic, expert):
    """A Ghost-type Curse user takes the Ghost branches (:681-684, :3467-3470)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, SHUCKLE, _SHUCKLE_MOVES, bench=(BONSLY,))
    s = s._replace(types=s.types.at[1, 0, 1].set(7))  # TYPE_GHOST
    if case == "hurt":
        s = _hp(s, 1, 50)
    if case == "cursed":
        s = s._replace(curse=s.curse.at[0].set(1))
    if case == "magic guard":
        mg = c.ABILITY_IDS["Magic Guard"]
        t = _with_abilities(t, int(s.species[0, 0]), [mg, mg])
    assert policy.ai_refusals(t, s, n) == []
    assert _delta(t, s, 0x2, 0x3)[2] == basic
    assert _delta(t, s, 0x3, 0x7)[2] == expert


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
    given effect 103; only the two-flag scores are read) that
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


# --- Flag sets (E1 Ruling R2) ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("flags", "scripts"),
    [
        (0x1, ["_basic"]),
        (0x5, ["_basic", "_expert"]),
        (0x7, ["_basic", "_eval_attack", "_expert"]),
        (0x17, ["_basic", "_eval_attack", "_expert", "_risky"]),
        (0xD, ["_basic", "_expert", "_setup_first_turn"]),
        (0x1F, ["_basic", "_eval_attack", "_expert", "_setup_first_turn", "_risky"]),
        (0x207, ["_basic", "_eval_attack", "_expert", "_weather"]),
        (0x20F, ["_basic", "_eval_attack", "_expert", "_setup_first_turn", "_weather"]),
        (0x27, ["_basic", "_eval_attack", "_expert", "_prioritize_extremes"]),
        (0x10F, ["_basic", "_eval_attack", "_expert", "_setup_first_turn", "_check_hp"]),
    ],
)
def test_flag_set_runs_its_scripts(kaizo_tables, kaizo_roark_party, flags, scripts):
    """TrainerAI_MainSingles runs a flag's script only when its bit is set, in bit order
    (trainer_ai.c:297-308; FlagTable, script.s:18-30). Under set 0x7 Selfdestruct and
    AncientPower get no Risky +2 (:6530), and their scores are the three flags' scores."""
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (SELFDESTRUCT, BRICK_BREAK, 0, 0))
    layout = layout_for(t, flags, s)
    assert [getattr(f, "func", f).__name__ for f in policy._flags(layout)] == scripts
    o = _outcome(t, s)
    ctx = policy._run_flags(c.new_context(t, s, o), layout)
    prefix = {0x1: 1, 0x7: 3, 0x17: 4}
    if flags in prefix:
        assert [int(x) for x in ctx.scores] == [
            int(x) for x in _flag_scores(t, s, o, prefix[flags])
        ]


@pytest.mark.fullset
def test_opponent_move_refusals_are_refused(kaizo_tables, kaizo_roark_party):
    """``policy.opponent_move_refusals`` (the backlog's per-move AI lines) agrees with
    ``_refused`` for every Kaizo move in an opponent slot: a move has a kind exactly when
    ``_refused`` refuses it by move."""
    t = kaizo_tables
    base = _state(t, kaizo_roark_party, BONSLY, (STEALTH_ROCK, 0, 0, 0))
    n_moves = int(t.move_ai_effect.shape[0])
    refused = jax.jit(
        lambda m: policy._refused(
            t, base._replace(moves=base.moves.at[1, 0, 1].set(m.astype(base.moves.dtype))), KAIZO
        )
    )
    for move in range(1, n_moves):
        r = refused(jnp.int32(move))
        by_move = any(
            bool(np.asarray(f)[0, 1]) for f in (r.move, r.expert, r.calc_damage, r.move_type)
        )
        kinds = policy.opponent_move_refusals(t, move, KAIZO)
        assert bool(kinds) == by_move, (move, kinds)


# --- SETUP_FIRST_TURN and WEATHER (E1 batch 2b) ----------------------------------------------


@partial(jax.jit, static_argnames="layout")
def _layout_scores(t, s, o, layout):
    return policy._run_flags(c.new_context(t, s, o), layout).scores


def _delta(t, s, base: int, flags: int, jumps=(), tie: int = 0) -> list[int]:
    """Per slot, the score under the flag set ``flags`` minus the score under ``base`` (its
    first flags), with the ``(slot, line)`` rolls of ``jumps`` jumping and no other, and every
    drawn speed tie taking bit ``tie`` (``_outcome``)."""
    lo, hi = layout_for(t, base, s), layout_for(t, flags, s)
    before = _layout_scores(t, s, _outcome(t, s, layout=lo, tie=tie), lo)
    after = _layout_scores(t, s, _outcome(t, s, jumps=jumps, layout=hi, tie=tie), hi)
    return [int(b) - int(a) for a, b in zip(before, after, strict=True)]


def _path_lines(t, s, layout) -> list[set[int]]:
    """Per slot, the script.s lines of the random reads on the path where no read jumps."""
    batch = jax.tree.map(lambda a: jnp.asarray(a)[None], s)
    bits = np.zeros((1, MOVES, c.LOOSE_READS), np.int8)
    _, lo = policy._loose_batch(t, batch, jnp.asarray(bits), layout)
    live, line = np.asarray(lo.live)[0, 0], np.asarray(lo.line)[0, 0]
    return [set(line[m][live[m]].tolist()) for m in range(MOVES)]


def _turn(s, turn: int):
    return s._replace(turn=jnp.asarray(turn, s.turn.dtype))


# Rock Polish (SPEED_UP_2, script.s:6457), Light Screen (SET_LIGHT_SCREEN, :6451) and Curse
# (:6475) are in SetupFirstTurn_SetupEffects; Stealth Rock's effect is not.
_SETUP_MOVES = (ROCK_POLISH, LIGHT_SCREEN, CURSE, STEALTH_ROCK)


@pytest.mark.fullset
def test_setup_first_turn_effects_are_the_decomps():
    """``commands.SETUP_FIRST_TURN_EFFECTS`` is SetupFirstTurn_SetupEffects (script.s:6435-6497)
    entry by entry, through generated/move_battle_effects.txt (line n + 1 is id n)."""
    script, ids = (
        DECOMP / "src/battle/trainer_ai/script.s",
        DECOMP / "generated/move_battle_effects.txt",
    )
    if not script.exists() or not ids.exists():
        pytest.skip("decomp not present")
    names = ids.read_text(encoding="utf-8").split()
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[6434] == "SetupFirstTurn_SetupEffects:"
    entries = []
    for line in lines[6435:]:
        name = line.split()[1]
        if name == "TABLE_END":
            break
        entries.append(names.index(name))
    assert tuple(entries) == c.SETUP_FIRST_TURN_EFFECTS


@pytest.mark.parametrize("turn", [0, 1])
@pytest.mark.parametrize("jump", [0, 1])
def test_setup_first_turn(kaizo_tables, kaizo_roark_party, turn, jump):
    """SetupFirstTurn_Main (script.s:6417-6433): on totalTurns 0 (LoadTurnCount,
    trainer_ai.c:931-935) a move whose effect is in the table gets +2 unless the 80 roll
    (:6429) jumps; on a later turn, and for another effect, no roll and no score. Against
    BASIC, EVAL_ATTACK and EXPERT (set 0x7 -> 0xf)."""

    t = kaizo_tables
    s = _turn(_state(t, kaizo_roark_party, BONSLY, _SETUP_MOVES), turn)
    jumps = tuple((slot, 6429) for slot in range(3)) if jump else ()
    plus = 2 if turn == 0 and not jump else 0
    assert _delta(t, s, 0x7, 0xF, jumps) == [plus, plus, plus, 0]
    rolls = [6429 in lines for lines in _path_lines(t, s, layout_for(t, 0xF, s))]
    assert rolls == [turn == 0] * 3 + [False]
    _assert_exact_equals_loose(t, s, layout_for(t, 0xF, s))


@pytest.mark.parametrize(
    ("weather", "turn", "expected"),
    [
        (Weather.NONE, 0, [5, 5, 5, 5]),
        (Weather.SANDSTORM, 0, [5, 5, 5, 5]),
        # The fall-through (E1 Ruling R3): a non-weather move runs Weather_Sun and ends.
        (Weather.SUN, 0, [0, 0, 0, 5]),
        (Weather.RAIN, 0, [5, 5, 5, 0]),
        (Weather.HAIL, 0, [5, 5, 5, 5]),
        (Weather.NONE, 1, [0, 0, 0, 0]),
    ],
)
@pytest.mark.parametrize("own", [False, True])
def test_weather(kaizo_tables, kaizo_names, kaizo_roark_party, weather, turn, expected, own):
    """Weather_Main (script.s:7979-8019), against set 0x7 (-> 0x207): on totalTurns 0 a
    weather move ends when its own weather is up (Weather_Sun/Rain, :7992-8000) and every other
    move falls through into Weather_Sun (no GoTo after :7990), so it ends under sun; otherwise
    Weather_ScorePlus5 gives +5 (the lead's LoadIsFirstTurnInBattle is TRUE, :8014-8016).
    Slots: Stealth Rock, Rock Polish, Sunny Day (WEATHER_SUN), Rain Dance (WEATHER_RAIN);
    refusals are not applied (the weather moves are unmodelled). With ``own``, Sandstorm
    (WEATHER_SANDSTORM) and Hail (WEATHER_HAIL) take slots 2 and 3, each ending under its own
    weather (Weather_Sand, Weather_Hail, :8002-8010)."""
    t, n = kaizo_tables, kaizo_names
    names = ("Sandstorm", "Hail") if own else ("Sunny Day", "Rain Dance")
    effects = (
        (BattleEffect.WEATHER_SANDSTORM, BattleEffect.WEATHER_HAIL)
        if own
        else (BattleEffect.WEATHER_SUN, BattleEffect.WEATHER_RAIN)
    )
    first, second = (n.move_names.index(x) for x in names)
    effect = np.asarray(t.move_ai_effect)
    assert (effect[first], effect[second]) == effects
    if own:  # own weather of each slot: sand for slot 2, hail for slot 3
        own_weather = {Weather.SANDSTORM: 2, Weather.HAIL: 3}
        expected = [0] * 4 if turn else [0 if weather == Weather.SUN else 5] * 2 + [5, 5]
        if weather in own_weather:
            expected[own_weather[weather]] = 0
    s = _state(t, kaizo_roark_party, BONSLY, (STEALTH_ROCK, ROCK_POLISH, first, second))
    s = _turn(s, turn)._replace(weather=jnp.asarray(weather, s.weather.dtype))
    assert _delta(t, s, 0x7, 0x207) == expected


@pytest.mark.parametrize("flags", [0xD, 0xF, 0x1F, 0x207, 0x20F])
@pytest.mark.parametrize("turn", [0, 1])
def test_exact_equals_loose_with_setup_and_weather(kaizo_tables, kaizo_roark_party, flags, turn):
    """Exact == loose under the 2b flag sets, on the first turn and after it: Rock Polish,
    Curse, Selfdestruct and Light Screen (three in SetupFirstTurn_SetupEffects, Selfdestruct
    with the longest path, kaizo_roark.md 8.3) against a guessed Gible and Tentacool."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t = kaizo_tables
    moves = (ROCK_POLISH, CURSE, SELFDESTRUCT, LIGHT_SCREEN)
    for player in (GIBLE_SPECIES, 72):
        s = _state(t, kaizo_roark_party, BONSLY, moves, hp=200)
        s = _turn(_with_player(t, s, player), turn)
        _assert_exact_equals_loose(t, s, layout_for(t, flags, s))


# --- PRIORITIZE_EXTREMES and CHECK_HP (E1 batch 2c) ------------------------------------------


@pytest.mark.parametrize("jump", [0, 1])
def test_prioritize_extremes(kaizo_tables, kaizo_roark_party, jump):
    """PrioritizeExtremes_Main (script.s:6499-6519), against set 0x7 (-> 0x27): a move
    FlagMoveDamageScore makes no comparison for (AI_NO_COMPARISON_MADE, :6511-6512;
    trainer_ai.c:1012-1063: power <= 1 or a no-calc effect) gets +2 unless the 100 roll (:6515)
    jumps; a damaging move makes no read. Stealth Rock (status) and Selfdestruct
    (HALVE_DEFENSE, no-calc) roll; Brick Break and Accelerock do not."""

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (STEALTH_ROCK, SELFDESTRUCT, BRICK_BREAK, ACCELEROCK))
    jumps = ((0, 6515), (1, 6515)) if jump else ()
    plus = 0 if jump else 2
    assert _delta(t, s, 0x7, 0x27, jumps) == [plus, plus, 0, 0]
    rolls = [6515 in lines for lines in _path_lines(t, s, layout_for(t, 0x27, s))]
    assert rolls == [True, True, False, False]
    _assert_exact_equals_loose(t, s, layout_for(t, 0x27, s))


@pytest.mark.fullset
def test_check_hp_tables_are_the_decomps():
    """``commands.CHECK_HP_TABLES`` is CheckHP_Main's six tables (script.s:7748-7977), label by
    label and entry by entry, through generated/move_battle_effects.txt (line n + 1 is id n);
    CheckHP_Target_DiscourageAtHighHP is empty."""
    script = DECOMP / "src/battle/trainer_ai/script.s"
    ids = DECOMP / "generated/move_battle_effects.txt"
    if not script.exists() or not ids.exists():
        pytest.skip("decomp not present")
    names = ids.read_text(encoding="utf-8").split()
    tables, label = {}, None
    for line in script.read_text(encoding="utf-8").splitlines()[7747:7977]:
        words = line.split()
        if line.endswith(":"):
            label = line[:-1]
            tables[label] = []
        elif words and words[0] == "TableEntry" and words[1] != "TABLE_END":
            tables[label].append(names.index(words[1]))
    assert {k: tuple(v) for k, v in tables.items()} == c.CHECK_HP_TABLES
    assert c.CHECK_HP_TABLES["CheckHP_Target_DiscourageAtHighHP"] == ()


def _hp(s, side: int, hp: int):
    """``s`` with side ``side``'s active at ``hp``/100 HP."""
    return s._replace(hp=s.hp.at[side, 0].set(hp), max_hp=s.max_hp.at[side, 0].set(100))


# The CheckHP bands (script.s:7700-7701, :7724-7725): IfHPPercentGreaterThan 70, then 30, on
# curHP * 100 / maxHP (trainer_ai.c:603), so 71 is high, 70 and 31 medium, 30 low.
# Slots: Selfdestruct (HALVE_DEFENSE 7), Rock Polish (SPEED_UP_2 52), Recover (RESTORE_HALF_HP
# 32), Accelerock (HIT 0). Their -2s, from the tables (:7748-7977):
_ATTACKER_BAND = {  # CheckHP_DiscourageAt{High,Medium,Low}HP
    71: [-2, 0, -2, 0],  # High: 7, 32
    70: [-2, -2, 0, 0],  # Medium: 7, 52
    31: [-2, -2, 0, 0],
    30: [0, -2, 0, 0],  # Low: 52
}
_TARGET_BAND = {  # CheckHP_Target_DiscourageAt{High,Medium,Low}HP
    71: [0, 0, 0, 0],  # High: empty
    70: [0, -2, 0, 0],  # Medium: 52
    31: [0, -2, 0, 0],
    30: [-2, -2, 0, 0],  # Low: 7, 52
}


@pytest.mark.parametrize("attacker", sorted(_ATTACKER_BAND))
@pytest.mark.parametrize("target", sorted(_TARGET_BAND))
def test_check_hp(kaizo_tables, kaizo_roark_party, attacker, target):
    """CheckHP_Main (script.s:7696-7746), against set 0x7 (-> 0x107): the attacker's HP band
    picks its table, a listed effect -2 unless :7718 jumps; then the target's (the player's
    true HP) band, -2 unless :7742 jumps. With both rolls jumping, no change."""
    t = kaizo_tables
    moves = (SELFDESTRUCT, ROCK_POLISH, RECOVER, ACCELEROCK)
    s = _hp(_hp(_state(t, kaizo_roark_party, BONSLY, moves), 1, attacker), 0, target)
    expected = [a + b for a, b in zip(_ATTACKER_BAND[attacker], _TARGET_BAND[target], strict=True)]
    assert _delta(t, s, 0x7, 0x107) == expected
    jumps = tuple((slot, line) for slot in range(MOVES) for line in (7718, 7742))
    assert _delta(t, s, 0x7, 0x107, jumps) == [0] * MOVES


@pytest.mark.parametrize("flags", [0x27, 0x10F, 0x107])
@pytest.mark.parametrize(("attacker", "target"), [(100, 100), (50, 20), (20, 60)])
def test_exact_equals_loose_with_extremes_and_check_hp(
    kaizo_tables, kaizo_roark_party, flags, attacker, target
):
    """Exact == loose under the 2c flag sets, across the CheckHP bands of both sides: Rock
    Polish, Curse, Selfdestruct and Recover against a guessed Gible and Tentacool, on the first
    turn (0x10f also runs SETUP_FIRST_TURN)."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t = kaizo_tables
    moves = (ROCK_POLISH, CURSE, SELFDESTRUCT, RECOVER)
    for player in (GIBLE_SPECIES, 72):
        s = _state(t, kaizo_roark_party, BONSLY, moves)
        s = _hp(_hp(_with_player(t, s, player), 1, attacker), 0, target)
        _assert_exact_equals_loose(t, s, layout_for(t, flags, s))


# --- Effects with no Basic or Expert label (E1 batch 3) --------------------------------------


@pytest.mark.parametrize("flags", [0x17, 0x27, 0x10F, 0x20F])
@pytest.mark.parametrize(
    "moves",
    [
        # DOUBLE_DAMAGE_DIG, PRIORITY_1 (nothing kills here: its :6400 bonus is
        # test_priority_1_kill_bonus_is_plus_6), CALL_RANDOM_MOVE
        # (Risky_RiskyEffects, :6543; a status move), MAKE_GLOBAL_TARGET (status)
        ("Earthquake", "Quick Attack", "Metronome", "Follow Me"),
        # MULTI_HIT, HIT_TWICE, THUNDER (no Expert_Thunder: :1718 tests another effect),
        # LOWER_ACCURACY_HIT
        ("Rock Blast", "Double Kick", "Thunder", "Mud-Slap"),
    ],
)
def test_unlabelled_effects_exact_equals_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, flags, moves
):
    """The effects with no Basic_ScoreMoveEffect or Expert_Main entry (outcome.KAIZO_TRANSCRIBED,
    batch 3) are scored, not refused, and exact == loose under Roark's set, PRIORITIZE_EXTREMES,
    CHECK_HP and SETUP_FIRST_TURN with WEATHER, against a guessed Gible and Tentacool, Bonsly at
    25 % HP (CheckHP's low band). Uproar's move-id read in Basic's Soundproof check (:127) has
    no Kaizo case: no Kaizo species has Soundproof."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    ids = tuple(n.move_names.index(m) for m in moves)
    for player in (GIBLE_SPECIES, 72):
        s = _state(t, kaizo_roark_party, BONSLY, ids)
        layout = layout_for(t, flags, s)
        s = _hp(_with_player(t, s, player), 1, 25)
        assert policy.ai_refusals(t, s, n, layout) == []
        _assert_exact_equals_loose(t, s, layout)


# --- Stat raises (E1 batch 4) ----------------------------------------------------------------


def _ids(n, *names):
    return tuple(n.move_names.index(m) if m else 0 for m in names)


@pytest.mark.parametrize(
    ("stage", "hp", "jumps", "expected"),
    [
        # Expert_StatusAttackUp (script.s:1978-2007), Swords Dance (ATK_UP_2, :1663):
        (3, 100, (), -1),  # +3 (decomp 9): -1 unless :1989 jumps; above 70 % nothing more
        (3, 100, (1989,), 0),
        (0, 100, (), 2),  # 100 % HP: +2 unless :1995 jumps
        (0, 100, (1995,), 0),
        (0, 70, (), -2),  # 40-70 %: -2 unless :2001 (N 40) jumps
        (0, 70, (2001,), 0),
        (0, 39, (2001,), -2),  # below 40 %: -2, no roll
        (3, 50, (), -3),  # both
    ],
)
def test_status_attack_up(kaizo_tables, kaizo_names, kaizo_roark_party, stage, hp, jumps, expected):
    """EXPERT's score for Swords Dance (BASIC and EVAL_ATTACK first: set 0x3 -> 0x7)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Swords Dance", "Accelerock", "", ""))
    s = _hp(s, 1, hp)._replace(stages=s.stages.at[1, Stat.ATK].set(stage))
    got = _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))
    assert got == [expected, 0, 0, 0]


@pytest.mark.parametrize(
    ("hp", "last", "jumps", "expected"),
    [
        # Expert_StatusDefenseUp (script.s:2009-2052), Iron Defense (DEF_UP_2, :1664), stage 0:
        (100, "", (), 0),  # +2 (:2029); then :2034 falls through; no move (power 0): -2 (:2046)
        (100, "", (2034,), 2),  # :2034 jumps: nothing more
        (50, "Flamethrower", (), -2),  # the player's special move: -2, no roll
        (50, "Tackle", (), -2),  # physical: :2043, then falls into :2046
        (50, "Tackle", (2043,), 0),
        (50, "Tackle", (2046,), 0),
        (50, "Leer", (2046,), 0),  # power 0: straight to :2046
        (50, "Leer", (2043,), -2),  # power 0 never reads :2043 (2c review M1)
        (30, "Tackle", (), -2),  # below 40 %: -2, no roll
    ],
)
def test_status_defense_up(kaizo_tables, kaizo_names, kaizo_roark_party, hp, last, jumps, expected):
    """EXPERT's score for Iron Defense against the player's previous move
    (LoadBattlerPreviousMove, trainer_ai.c:1068-1076; LoadPowerOfLoadedMove, :2526-2530;
    LoadDefenderLastUsedMoveClass, :2050-2054)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Iron Defense", "Accelerock", "", ""))
    move = n.move_names.index(last) if last else 0
    s = _hp(s, 1, hp)._replace(last_move=s.last_move.at[0].set(move))
    got = _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))
    assert got == [expected, 0, 0, 0]


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_stat_raises_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for the batch 4 raises, the first Kaizo move of each effect (ATK_UP,
    DEF_UP, SP_ATK_UP, ATK_UP_2; DEF_UP_2, SP_ATK_UP_2, SP_DEF_UP_2), at several stages and
    HP, against a guessed Gible."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    effect = np.asarray(t.move_ai_effect)
    pool = {}
    for m in range(1, effect.shape[0]):
        pool.setdefault(int(effect[m]), m)
    groups = ((10, 11, 13, 50), (51, 53, 54, 0))
    for moves in groups:
        ids = tuple(pool[e] if e else 0 for e in moves)
        for hp, stage in ((100, 0), (55, 3), (35, 6)):
            s = _state(t, kaizo_roark_party, BONSLY, ids)
            layout = layout_for(t, flags, s)
            s = _hp(_with_player(t, s, GIBLE_SPECIES), 1, hp)
            s = s._replace(stages=s.stages.at[1, 1:].set(stage))
            assert policy.ai_refusals(t, s, n, layout) == []
            _assert_exact_equals_loose(t, s, layout)


@pytest.mark.parametrize(
    ("last", "jumps", "expected"),
    [
        # Expert_StatusSpDefenseUp (script.s:2112-2155), Amnesia (SP_DEF_UP_2, :1667), stage 0,
        # 50 % HP: the player's physical move -> -2 with no roll (:2145); a special one rolls
        # :2146 and falls into :2149.
        ("Tackle", (2146, 2149), -2),
        ("Flamethrower", (), -2),
        ("Flamethrower", (2146,), 0),
        ("Flamethrower", (2149,), 0),
    ],
)
def test_status_sp_defense_up(kaizo_tables, kaizo_names, kaizo_roark_party, last, jumps, expected):
    """EXPERT's score for Amnesia: the Sp. Def label sends a physical previous move to -2
    (batch 4 review M2)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Amnesia", "Accelerock", "", ""))
    s = _hp(s, 1, 50)._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    got = _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))
    assert got == [expected, 0, 0, 0]


@pytest.mark.parametrize(
    ("stage", "simple", "expected"),
    [
        # Basic_CheckHighStatStage_SpAttack (script.s:374-382), Nasty Plot (SP_ATK_UP_2):
        (6, False, -10),  # +6 (decomp 12): -10
        (5, False, 0),
        (3, True, -10),  # Simple: above decomp 8, -10
        (2, True, 0),
    ],
)
def test_basic_high_stat_stage(
    kaizo_tables, kaizo_names, kaizo_roark_party, stage, simple, expected
):
    """BASIC's score for a stat raise at the top of its stage (set 0x2 -> 0x3; batch 4 review
    M2)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Nasty Plot", "Accelerock", "", ""))
    s = s._replace(stages=s.stages.at[1, Stat.SPA].set(stage))
    if simple:  # a species whose ability slot 0 is Simple (the AI reads the species')
        abilities = np.asarray(t.species_abilities_decomp)
        rows = np.asarray(t.species_form_row)[:, 0]
        species = next(
            sp
            for sp in range(1, rows.shape[0])
            if abilities[rows[sp], 0] == c.ABILITY_IDS["Simple"]
        )
        s = as_species(t, s, 1, 0, species)
    assert _delta(t, s, 0x2, 0x3)[0] == expected


# --- Status moves (E1 batch 6) ---------------------------------------------------------------


def _species_with(t, ability: int) -> int:
    """A species whose ability slot 0 is ``ability`` (decomp id) in the Kaizo tables."""
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    return next(sp for sp in range(1, rows.shape[0]) if abilities[rows[sp], 0] == ability)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("plain", 0),
        ("statused", -10),  # :629 IfStatus MON_CONDITION_ANY
        ("safeguard", -10),  # :630
        ("volt_absorb", -10),  # Thunder Wave only: :623-626, the player's ability known
    ],
)
def test_basic_check_cannot_paralyze(kaizo_tables, kaizo_names, kaizo_roark_party, case, expected):
    """BASIC's score for Thunder Wave (Basic_CheckCannotParalyze, script.s:612-631; set
    0x2 -> 0x3), against a Tentacool whose ability the AI knows."""

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Thunder Wave", "Accelerock", "", ""))
    s = _with_player(t, s, 72, 0, known=True)  # Tentacool: not Ground, so not immune (:614)
    if case == "statused":
        s = s._replace(status=s.status.at[0, 0].set(Status.POISON))
    elif case == "safeguard":
        s = s._replace(safeguard=s.safeguard.at[0].set(5))
    elif case == "volt_absorb":
        s = _with_player(t, s, _species_with(t, c.ABILITY_IDS["Volt Absorb"]), 0, known=True)
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("attacker", "defender", "expected"),
    [
        (c.Gender.MALE, c.Gender.FEMALE, 0),
        (c.Gender.FEMALE, c.Gender.MALE, 0),
        (c.Gender.MALE, c.Gender.MALE, -10),
        (c.Gender.FEMALE, c.Gender.FEMALE, -10),
        (c.Gender.NONE, c.Gender.FEMALE, -10),  # :739 GoTo ScoreMinus10
        (c.Gender.MALE, c.Gender.NONE, -10),
    ],
)
def test_basic_check_cannot_attract(
    kaizo_tables, kaizo_names, kaizo_roark_party, attacker, defender, expected
):
    """BASIC's score for Attract (Basic_CheckCannotAttract, script.s:731-752; LoadGender,
    trainer_ai.c:2470-2477)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Attract", "Accelerock", "", ""))
    s = s._replace(gender=s.gender.at[1, 0].set(attacker).at[0, 0].set(defender))
    assert _delta(t, s, 0x2, 0x3)[0] == expected


def test_basic_check_cannot_confuse(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Already confused: -5 (script.s:599, IfVolatileStatus CONFUSION); not: 0."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Confuse Ray", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    s = s._replace(confusion=s.confusion.at[0].set(2))
    assert _delta(t, s, 0x2, 0x3)[0] == -5


@pytest.mark.parametrize(
    ("hp", "jumps", "expected"),
    [
        # Expert_StatusConfuse (script.s:2861-2877), the player's HP:
        (71, (), 0),  # above 70 %: nothing
        (60, (), -1),  # -1 unless :2864 jumps
        (60, (2864,), 0),
        (50, (), -2),  # and -1 at 50 % or less
        (30, (), -3),  # and -1 more at 30 % or less
        (30, (2864,), -2),
    ],
)
def test_status_confuse(kaizo_tables, kaizo_names, kaizo_roark_party, hp, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Confuse Ray", "Accelerock", "", ""))
    s = _hp(s, 0, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("atk", "turn", "expected"),
    [
        # Expert_Swagger_PsychUp (script.s:2879-2891): Psych Up known.
        (-2, 1, -5),  # the player's Attack above decomp 3 (engine -3): -5
        (-3, 1, 3),  # else +3
        (-3, 0, 5),  # and +2 on the battle's first turn
    ],
)
def test_swagger_with_psych_up(kaizo_tables, kaizo_names, kaizo_roark_party, atk, turn, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Swagger", "Psych Up", "", ""))
    s = _turn(s._replace(stages=s.stages.at[0, Stat.ATK].set(atk)), turn)
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_status_moves_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for the batch 6 status moves, against a guessed Gible and Tentacool, at
    two HP levels, the player confused or not."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    groups = (
        ("Thunder Wave", "Hypnosis", "Confuse Ray", "Leech Seed"),
        ("Swagger", "Attract", "Will-O-Wisp", "Accelerock"),
    )
    for moves in groups:
        for player in (GIBLE_SPECIES, 72):
            for hp, confused in ((100, 0), (35, 2)):
                s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
                s = _hp(_with_player(t, s, player), 0, hp)
                s = s._replace(confusion=s.confusion.at[0].set(confused))
                layout = layout_for(t, flags, s)
                assert policy.ai_refusals(t, s, n, layout) == []
                _assert_exact_equals_loose(t, s, layout)


# --- Expert-only hits (E1 batch 7) -----------------------------------------------------------


@pytest.mark.parametrize(
    ("eva", "jumps", "expected"),
    [
        # Expert_BypassAccuracyMove (script.s:2256-2275), Aerial Ace, the player's evasion:
        (0, (), 0),  # neither threshold: nothing
        (3, (), 1),  # above decomp 8: +1 unless :2271 jumps
        (3, (2271,), 0),
        (5, (), 2),  # above decomp 10: +1, then the roll's +1
        (5, (2271,), 1),
        # the attacker's accuracy (:2262, :2264): below decomp 4 (engine -3) the roll's +1,
        # below 2 (engine -5) +1 and the roll's +1
        ("acc-3", (), 1),
        ("acc-5", (), 2),
    ],
)
def test_bypass_accuracy(kaizo_tables, kaizo_names, kaizo_roark_party, eva, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Aerial Ace", "Accelerock", "", ""))
    if isinstance(eva, str):
        s = s._replace(stages=s.stages.at[1, Stat.ACC].set(int(eva.removeprefix("acc"))))
    else:
        s = s._replace(stages=s.stages.at[0, Stat.EVA].set(eva))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("player", "jumps", "expected"),
    [
        # Expert_HighCritical (script.s:2827-2843), Cross Chop (Fighting):
        ("Tentacool", (), 0),  # Water/Poison: Fighting is resisted: nothing
        ("Geodude", (), 1),  # Rock/Ground: super effective: +1 unless :2839 jumps
        ("Geodude", (2839,), 0),
        ("Gible", (), 1),  # Dragon/Ground: neutral: :2836, then :2839
        ("Gible", (2836,), 0),
        ("Geodude", (2836,), 1),  # super effective skips :2836 (:2834-2835; review m1)
    ],
)
def test_high_critical(kaizo_tables, kaizo_names, kaizo_roark_party, player, jumps, expected):
    from tests.ai.test_kaizo_outcome import _with_player

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Cross Chop", "Accelerock", "", ""))
    s = _with_player(t, s, n.species_names.index(player), 0, known=True)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("move", "status", "hp", "expected"),
    [
        ("Facade", Status.BURN, 100, 1),  # Expert_Facade (:4120-4127): the player's status
        ("Facade", Status.SLEEP, 100, 0),
        ("SmellingSalt", Status.PARALYSIS, 100, 1),  # Expert_SmellingSalts (:4165-4174)
        ("SmellingSalt", Status.POISON, 100, 0),
        ("Brine", Status.NONE, 51, 0),  # Expert_Brine (:4873-4890): above 50 %, nothing
        ("Brine", Status.NONE, 50, 2),  # +1 and +1 unless :4882 jumps
    ],
)
def test_status_and_hp_boosts(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, status, hp, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _hp(s._replace(status=s.status.at[0, 0].set(status)), 0, hp)
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize(
    ("move", "slower", "expected"),
    [
        # Expert_SpeedDownOnHit (script.s:2338-2350) -> Expert_StatusSpeedDown (:2353-2366):
        ("Icy Wind", True, 2),  # slower: +2 unless :2362 jumps
        ("Icy Wind", False, -3),
        ("Rock Tomb", True, 2),
        ("Mud Shot", False, -3),
        ("BubbleBeam", True, 0),  # LOWER_SPEED_HIT but not one of the three: nothing
        ("Icy Wind:2362", True, 0),  # the 70 roll jumps
    ],
)
def test_speed_down_on_hit(kaizo_tables, kaizo_names, kaizo_roark_party, move, slower, expected):
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    move, _, line = move.partition(":")
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    # a player none of them is resisted by: Gible for Icy Wind and Mud Shot, Tentacool for
    # Rock Tomb and BubbleBeam
    target = GIBLE_SPECIES if move in ("Icy Wind", "Mud Shot") else 72
    s = _with_player(t, s, target, 0, known=True)
    s = _speeds(s, player=500 if slower else 1, opp=100)
    jumps = ((0, int(line)),) if line else ()
    assert _delta(t, s, 0x3, 0x7, jumps)[0] == expected


def test_brine_resisted(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Expert_Brine (script.s:4877-4887): resisted (Water into a Water/Poison Tentacool), -1
    whatever the HP (review m1)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Brine", "Accelerock", "", ""))
    s = _hp(_with_player(t, s, 72, 0, known=True), 0, 30)
    assert _delta(t, s, 0x3, 0x7)[0] == -1


@pytest.mark.parametrize(
    ("attacker", "player", "expected"),
    [
        (None, "Pikachu", 1),  # Bonsly's Rock Head, not resisted: +1 (:6289-6298)
        (("Kadabra", 0, "Magic Guard"), "Pikachu", 1),  # :6297
        (("Cranidos", 1, "Mold Breaker"), "Pikachu", 0),  # neither ability: nothing
        (None, "Gible", 0),  # resisted (Fire into Dragon): nothing
    ],
)
def test_flare_blitz_recoil_move(
    kaizo_tables, kaizo_names, kaizo_roark_party, attacker, player, expected
):
    """E2 batch 10 (B10-1): RECOIL_BURN_HIT goes to Expert_RecoilMove (script.s:1796), the
    label RECOIL_PARALYZE_HIT reaches (:1802). [Flare Blitz, Accelerock]."""
    t, n = kaizo_tables, kaizo_names
    abilities = np.asarray(t.species_abilities_decomp)
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Flare Blitz", "Accelerock", "", ""))
    s = _with_player(t, s, n.species(player), 0, known=True)
    if attacker:
        name, slot, ability = attacker
        species = n.species(name)
        assert int(abilities[species, slot]) == c.ABILITY_IDS[ability]
        s = as_species(t, s, 1, 0, species, slot)
    else:
        bonsly = kaizo_roark_party[BONSLY]
        assert int(abilities[bonsly.species, bonsly.ability_slot]) == ROCK_HEAD
    assert _delta(t, s, 0x3, 0x7)[:2] == [expected, 0]


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_expert_only_hits_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for one Kaizo move of each batch 7 effect, against a guessed Gible and
    Tentacool, at full and low HP, with the player's evasion up."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    effect = np.asarray(t.move_ai_effect)
    pool = {}
    for m in range(1, effect.shape[0]):
        pool.setdefault(int(effect[m]), m)
    pool[70] = n.move_names.index("Icy Wind")  # one that reaches Expert_StatusSpeedDown
    groups = ((17, 43, 48, 70), (169, 171, 198, 221), (262, 253, 0, 0))
    for moves in groups:
        ids = tuple(pool[e] if e else 0 for e in moves)
        for player in (GIBLE_SPECIES, 72):
            for hp in (100, 40):
                s = _state(t, kaizo_roark_party, BONSLY, ids)
                s = _hp(_with_player(t, s, player), 0, hp)
                s = s._replace(stages=s.stages.at[0, Stat.EVA].set(3))
                layout = layout_for(t, flags, s)
                assert policy.ai_refusals(t, s, n, layout) == []
                _assert_exact_equals_loose(t, s, layout)


# Batch 6 review M1: the branches the first status tests left open.


@pytest.mark.parametrize(
    ("move", "attacker", "expected"),
    [
        ("Thunder Wave", None, -10),  # Volt Absorb known: -10 (:623-626)
        ("Glare", None, 0),  # only Thunder Wave reads Motor Drive / Volt Absorb (:620)
        ("Thunder Wave", "Mold Breaker", 0),  # Mold Breaker skips them (:619)
    ],
)
def test_paralyze_absorb_branches(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, attacker, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    volt = _species_with(t, c.ABILITY_IDS["Volt Absorb"])
    assert int(np.asarray(t.species_types)[volt][0]) != 4  # not Ground: no type immunity
    s = _with_player(t, s, volt, 0, known=True)
    if attacker:
        species = _species_with(t, c.ABILITY_IDS[attacker])
        s = as_species(t, s, 1, 0, species)
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("move", "case", "expected"),
    [
        ("Hypnosis", "statused", -10),  # Basic_CheckCannotSleep (:288-295)
        ("Hypnosis", "safeguard", -10),
        ("Hypnosis", "Insomnia", -10),
        ("Leech Seed", "seeded", -10),  # Basic_CheckCannotLeechSeed (:639-648)
        ("Leech Seed", "grass", -10),
        ("Leech Seed", "plain", 0),
        ("Will-O-Wisp", "fire", -10),  # Basic_CheckCannotBurn (:879-890)
        # (Water Veil, :882, is no Kaizo species' ability)
        ("Will-O-Wisp", "plain", 0),
        ("Confuse Ray", "Own Tempo", -10),  # Basic_CheckCannotConfuse (:597-605)
        ("Confuse Ray", "safeguard", -10),
        ("Attract", "attracted", -10),  # :733, IfVolatileStatus ATTRACT
    ],
)
def test_basic_cannot_status(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _with_player(t, s, 72, 0, known=True)  # Tentacool: Water/Poison
    s = s._replace(gender=s.gender.at[1, 0].set(c.Gender.MALE).at[0, 0].set(c.Gender.FEMALE))
    if case == "statused":
        s = s._replace(status=s.status.at[0, 0].set(Status.BURN))
    elif case == "safeguard":
        s = s._replace(safeguard=s.safeguard.at[0].set(5))
    elif case == "seeded":
        s = s._replace(leech_seed=s.leech_seed.at[0].set(1))
    elif case == "attracted":
        s = s._replace(infatuated=s.infatuated.at[0].set(1))
    elif case in ("grass", "fire"):
        kind = c.TYPE_IDS["Grass" if case == "grass" else "Fire"]
        s = s._replace(types=s.types.at[0, 0].set(jnp.asarray([kind, kind], s.types.dtype)))
    elif case != "plain":  # a species with the ability in either slot
        abilities = np.asarray(t.species_abilities_decomp)
        rows = np.asarray(t.species_form_row)[:, 0]
        species, slot = next(
            (sp, k)
            for sp in range(1, rows.shape[0])
            for k in (0, 1)
            if abilities[rows[sp], k] == c.ABILITY_IDS[case]
        )
        s = _with_player(t, s, species, slot, known=True)
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("hp", "jumps", "expected"),
    [
        # Swagger without Psych Up: Expert_Flatter's +1 unless :2858 jumps, then
        # Expert_StatusConfuse on the player's HP (:2854-2877).
        (100, (), 1),
        (100, (2858,), 0),
        (40, (), -1),  # +1, -1 (:2864 does not jump), -1 at 50 % or less
    ],
)
def test_swagger_without_psych_up(
    kaizo_tables, kaizo_names, kaizo_roark_party, hp, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _hp(_state(t, kaizo_roark_party, BONSLY, _ids(n, "Swagger", "Accelerock", "", "")), 0, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(("known", "jumps", "expected"), [(1, (), 1), (1, (1821,), 0), (0, (), 0)])
def test_expert_status_sleep(kaizo_tables, kaizo_names, kaizo_roark_party, known, jumps, expected):
    """Expert_StatusSleep (script.s:1813-1825): +1 unless :1821 jumps when the attacker knows
    Dream Eater's effect (IfMoveEffectKnown RECOVER_DAMAGE_SLEEP, :1816)."""
    t, n = kaizo_tables, kaizo_names
    third = "Dream Eater" if known else "Accelerock"
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Hypnosis", "Accelerock", third, ""))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("slower", "own_hp", "jumps", "expected"),
    [
        # Expert_StatusParalyze (script.s:2940-2954):
        (True, 100, (), 3),  # slower: +3 unless :2950 jumps
        (True, 100, (2950,), 0),
        (False, 100, (), 0),  # not slower, own HP above 70 %: nothing
        (False, 70, (), -1),  # own HP 70 % or less: -1 (:2945, the attacker's)
    ],
)
def test_expert_status_paralyze(
    kaizo_tables, kaizo_names, kaizo_roark_party, slower, own_hp, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Thunder Wave", "Accelerock", "", ""))
    s = _with_player(t, s, 72, 0, known=True)
    s = _hp(_speeds(s, player=500 if slower else 1, opp=100), 1, own_hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


# --- Setup moves, hazards and Destiny Bond (E1 batch 9) --------------------------------------


@pytest.mark.parametrize(
    ("move", "stages", "expected"),
    [
        # Basic_CheckBulkUp (script.s:965-979) / _CalmMind (:986-1000) / _DragonDance
        # (:1002-1019), no Simple: first stat at +6 -10, second at +6 -8
        ("Bulk Up", {Stat.ATK: 6}, -10),
        ("Bulk Up", {Stat.DEF: 6}, -8),
        ("Bulk Up", {Stat.ATK: 5, Stat.DEF: 5}, 0),
        ("Calm Mind", {Stat.SPD: 6}, -8),
        ("Dragon Dance", {Stat.SPE: 6}, -8),
        ("Dragon Dance", {}, 0),
    ],
)
def test_basic_double_stat_stage(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, stages, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    for stat, stage in stages.items():
        s = s._replace(stages=s.stages.at[1, stat].set(stage))
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        # Basic_CheckBulkUp with Simple (script.s:967-970): either stat above decomp 8 (+3), -10
        ({Stat.ATK: 3}, -10),
        ({Stat.DEF: 3}, -10),
        ({Stat.ATK: 2, Stat.DEF: 2}, 0),
    ],
)
def test_basic_double_stat_stage_simple(
    kaizo_tables, kaizo_names, kaizo_roark_party, stages, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Bulk Up", "Accelerock", "", ""))
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    species = next(
        sp for sp in range(1, rows.shape[0]) if abilities[rows[sp], 0] == c.ABILITY_IDS["Simple"]
    )
    s = as_species(t, s, 1, 0, species)
    for stat, stage in stages.items():
        s = s._replace(stages=s.stages.at[1, stat].set(stage))
    assert _delta(t, s, 0x2, 0x3)[0] == expected


def test_basic_dragon_dance_trick_room(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckDragonDance gives -10 under Trick Room first (:1004)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Dragon Dance", "Accelerock", "", ""))
    s = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    assert _delta(t, s, 0x2, 0x3)[0] == -10


@pytest.mark.parametrize(
    ("move", "layers", "alone", "expected"),
    [
        # Basic_CheckSpikes (:707-713) / _ToxicSpikes (:1445-1453): full layers or no other
        # living Pokemon on the player's side, -10
        ("Spikes", 3, False, -10),
        ("Spikes", 2, False, 0),
        ("Spikes", 0, True, -10),
        ("Toxic Spikes", 2, False, -10),
        ("Toxic Spikes", 1, False, 0),
    ],
)
def test_basic_hazards(kaizo_tables, kaizo_names, kaizo_roark_party, move, layers, alone, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(
        t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""), players=1 if alone else 2
    )
    field = "spikes" if move == "Spikes" else "toxic_spikes"
    s = s._replace(**{field: getattr(s, field).at[0].set(layers)})
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("roar", "jumps", "expected"),
    [
        # Expert_Spikes (:3591-3606): +1 unless :3595 jumps; with Roar known +1 more unless
        # :3602 jumps
        (False, (), 1),
        (False, (3595,), 0),
        (True, (), 2),
        (True, (3602,), 1),
    ],
)
def test_expert_spikes(kaizo_tables, kaizo_names, kaizo_roark_party, roar, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    second = "Roar" if roar else "Accelerock"
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Spikes", second, "", ""))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("second", "jumps", "expected"),
    [
        # Expert_StealthRock (:6268-6285), the hazard labels' shape: +1 unless :6274 jumps;
        # with Roar or Whirlwind known (transcribed since E1 batch 17) +1 more unless :6281
        # jumps (merge check on ff97cd0)
        ("Accelerock", (), 1),
        ("Roar", (), 2),
        ("Roar", (6281,), 1),
        ("Whirlwind", (), 2),
        ("Whirlwind", (6281,), 1),
        ("Roar", (6274,), 0),
    ],
)
def test_expert_stealth_rock(kaizo_tables, kaizo_names, kaizo_roark_party, second, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Stealth Rock", second, "", ""))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("slower", "hp", "jumps", "expected"),
    [
        # Expert_DragonDance (:4765-4780)
        (True, 100, (), 1),  # slower: +1 unless :4776 jumps
        (True, 100, (4776,), 0),
        (False, 100, (), 0),  # not slower, above 50 %: nothing
        (False, 50, (), -1),  # at 50 % or less: -1 unless :4771 jumps
        (False, 50, (4771,), 0),
    ],
)
def test_expert_dragon_dance(
    kaizo_tables, kaizo_names, kaizo_roark_party, slower, hp, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Dragon Dance", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=500 if slower else 1, opp=100), 1, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("slower", "hp", "jumps", "expected"),
    [
        # Expert_DestinyBond (:3335-3360): -1 first
        (True, 20, (), -1),  # slower: nothing more
        (False, 71, (), -1),  # above 70 %: nothing more
        (False, 60, (), 0),  # +1 unless :3346 jumps
        (False, 40, (), 1),  # and +1 unless :3351 jumps
        (False, 30, (), 3),  # and +2 unless :3356 jumps
        (False, 30, (3346, 3351, 3356), -1),
    ],
)
def test_expert_destiny_bond(
    kaizo_tables, kaizo_names, kaizo_roark_party, slower, hp, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Destiny Bond", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=500 if slower else 1, opp=100), 1, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch9_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for the batch 9 moves, against a guessed Gible and Tentacool, at two HP
    levels, the player's side with a Spikes layer."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    groups = (
        ("Bulk Up", "Calm Mind", "Dragon Dance", "Accelerock"),
        ("Spikes", "Toxic Spikes", "Destiny Bond", "Accelerock"),
    )
    for moves in groups:
        for player in (GIBLE_SPECIES, 72):
            for hp in (100, 40):
                s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
                s = _hp(_with_player(t, s, player), 1, hp)
                s = s._replace(spikes=s.spikes.at[0].set(1))
                layout = layout_for(t, flags, s)
                assert policy.ai_refusals(t, s, n, layout) == []
                _assert_exact_equals_loose(t, s, layout)


# --- Stat drops and Reflect (E1 batch 10) ----------------------------------------------------


def _move_of(t, effect: int) -> int:
    """The first Kaizo move with battle effect ``effect``."""
    return 1 + int(np.flatnonzero(np.asarray(t.move_ai_effect)[1:] == effect)[0])


def _known_ability(t, s, ability: int):
    """``s`` with the player's active a species with ``ability`` in either slot, known."""
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    species, slot = next(
        (sp, k)
        for sp in range(1, rows.shape[0])
        for k in (0, 1)
        if abilities[rows[sp], k] == ability
    )
    return _with_player(t, s, species, slot, known=True)


@pytest.mark.parametrize(
    ("effect", "case", "expected"),
    [
        # Basic_CheckLowStatStage_Attack (:428-432) then Basic_CheckClearBodyEffect (:469-473)
        (18, "plain", 0),
        (18, "stage", -10),  # the player's Attack at -6
        (18, "Hyper Cutter", -10),
        (18, "Clear Body", -10),
        # Basic_CheckLowStatStage_Speed (:438-443): Trick Room, -6, Speed Boost
        (20, "trick_room", -10),
        (20, "Speed Boost", -10),
        (20, "plain", 0),
        # Basic_CheckLowStatStage_Evasion (:462-467): No Guard on either side
        (24, "No Guard", -10),
        (24, "plain", 0),
        # Basic_CheckLowStatStage_SpDefense (:449-451)
        (62, "White Smoke", -10),
        # Basic_CheckAlreadyUnderReflect (:607-610)
        (65, "reflect", -8),
        (65, "plain", 0),
    ],
)
def test_basic_drops_and_reflect(kaizo_tables, kaizo_roark_party, effect, case, expected):
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (_move_of(t, effect), ACCELEROCK, 0, 0))
    s = _with_player(t, s, 72, 0, known=True)  # Tentacool, its abilities not in play
    stat = {18: Stat.ATK, 20: Stat.SPE, 24: Stat.EVA, 62: Stat.SPD}.get(effect)
    if case == "stage":
        s = s._replace(stages=s.stages.at[0, stat].set(-6))
    elif case == "trick_room":
        s = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    elif case == "reflect":
        s = s._replace(reflect=s.reflect.at[1].set(5))
    elif case != "plain":
        s = _known_ability(t, s, c.ABILITY_IDS[case])
    assert _delta(t, s, 0x2, 0x3)[0] == expected


def test_check_battler_ability_unknown(kaizo_tables, kaizo_roark_party):
    """CheckBattlerAbility (trainer_ai.c:1236-1246): a player of two abilities whose ability
    the AI has not learned, one of them Speed Boost, is AI_UNKNOWN, so
    Basic_CheckLowStatStage_Speed gives no -10 for it (:441-442); known, it does."""
    t = kaizo_tables
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    boost = c.ABILITY_IDS["Speed Boost"]
    species, slot = next(
        (sp, k)
        for sp in range(1, rows.shape[0])
        for k in (0, 1)
        if abilities[rows[sp], k] == boost and abilities[rows[sp], 1 - k] not in (0, boost)
    )
    s = _state(t, kaizo_roark_party, BONSLY, (_move_of(t, 20), ACCELEROCK, 0, 0))
    unknown = _with_player(t, s, species, slot, known=False)
    known = _with_player(t, s, species, slot, known=True)
    assert _delta(t, unknown, 0x2, 0x3)[0] == 0
    assert _delta(t, known, 0x2, 0x3)[0] == -10


@pytest.mark.parametrize(
    ("own_hp", "player_hp", "stage", "last", "jumps", "expected"),
    [
        # Expert_StatusAttackDown (script.s:2277-2307)
        (100, 100, 0, "", (), 0),  # the player's Attack at +0: straight to its HP, above 70 %
        (100, 100, -1, "", (), -1),  # -1; decomp 5 is above 3: no roll
        (80, 100, -1, "", (), -2),  # and -1 more at 90 % or less
        (100, 100, -4, "", (), -3),  # decomp 2: the 50 roll's -2 (:2293)
        (100, 100, -4, "", (2293,), -1),
        (100, 60, 0, "", (), -2),  # the player at 70 % or less: -2
        (100, 100, 0, "Flamethrower", (), -2),  # its last move special: -2 unless :2303 jumps
        (100, 100, 0, "Flamethrower", (2303,), 0),
    ],
)
def test_expert_status_attack_down(
    kaizo_tables, kaizo_roark_party, own_hp, player_hp, stage, last, jumps, expected, kaizo_names
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, (_move_of(t, 18), ACCELEROCK, 0, 0))
    s = _hp(_hp(s, 1, own_hp), 0, player_hp)
    s = s._replace(stages=s.stages.at[0, Stat.ATK].set(stage))
    if last:
        s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("effect", "own_hp", "player_hp", "jumps", "expected"),
    [
        # Expert_StatusSpDefenseDown (:2411-2429) / _EvasionDown (:2500-2518)
        (62, 100, 100, (), 0),  # 70 % or more and the player's stage above decomp 3: its HP
        (62, 50, 100, (), -2),  # below 70 %: -2 unless the 50 roll jumps
        (62, 50, 100, (2421,), 0),
        (62, 50, 50, (), -4),  # and -2 at the player's 70 % or less
        (24, 50, 50, (2510,), -2),
    ],
)
def test_expert_status_down_hp(
    kaizo_tables, kaizo_roark_party, effect, own_hp, player_hp, jumps, expected
):
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (_move_of(t, effect), ACCELEROCK, 0, 0))
    s = _hp(_hp(s, 1, own_hp), 0, player_hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("hp", "last", "jumps", "expected"),
    [
        # Expert_Reflect (script.s:2893-2915)
        (40, "Tackle", (), -2),  # below 50 %: -2
        (100, "Tackle", (), 2),  # 90 % or more: +1 unless :2901 jumps; physical: +1 unless :2907
        (100, "Tackle", (2901, 2907), 0),
        (70, "Flamethrower", (), 0),  # 50-89 %, special: nothing
        (70, "Tackle", (), 1),
    ],
)
def test_expert_reflect(kaizo_tables, kaizo_names, kaizo_roark_party, hp, last, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, (_move_of(t, 65), ACCELEROCK, 0, 0))
    s = _hp(s, 1, hp)._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch10_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for the batch 10 effects, against a guessed Gible and Tentacool, at two
    HP levels, the player's Attack down one stage."""
    from tests.ai.test_kaizo_outcome import GIBLE as GIBLE_SPECIES

    t, n = kaizo_tables, kaizo_names
    for effects in ((18, 20, 24, 58), (60, 62, 65, 0)):
        ids = tuple(_move_of(t, e) if e else 0 for e in effects)
        for player in (GIBLE_SPECIES, 72):
            for hp in (100, 40):
                s = _state(t, kaizo_roark_party, BONSLY, ids)
                s = _hp(_with_player(t, s, player), 1, hp)
                s = s._replace(stages=s.stages.at[0, Stat.ATK].set(-1))
                layout = layout_for(t, flags, s)
                assert policy.ai_refusals(t, s, n, layout) == []
                _assert_exact_equals_loose(t, s, layout)


# --- Tailwind and Payback: COMPARE_SPEED_FASTER (E1 batch 11, Ruling R8) ---------------------


@pytest.mark.parametrize(
    ("case", "expected"), [("plain", 0), ("trick_room", -10), ("tailwind", -10)]
)
def test_basic_check_tailwind(kaizo_tables, kaizo_names, kaizo_roark_party, case, expected):
    """Basic_CheckTailwind (script.s:1128-1133)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Tailwind", "Accelerock", "", ""))
    if case == "trick_room":
        s = s._replace(trick_room=jnp.asarray(3, s.trick_room.dtype))
    elif case == "tailwind":
        s = s._replace(tailwind=s.tailwind.at[1].set(3))
    assert _delta(t, s, 0x2, 0x3)[0] == expected
    # the player's own Tailwind is not the AI's side (:1131 reads AI_BATTLER_ATTACKER)
    if case == "plain":
        s = s._replace(tailwind=s.tailwind.at[0].set(3))
        assert _delta(t, s, 0x2, 0x3)[0] == 0


@pytest.mark.parametrize(
    ("tie", "jumps", "expected"),
    [
        # Expert_Tailwind at equal speed (script.s:4993; battle_lib.c:1445-1446): the tie's
        # RandNext() & 1 is the slot's read after :4992 (E1 Ruling R8)
        (0, (), -1),  # bit 0: FASTER, -1
        (1, (), 1),  # bit 1: TIE, not faster; above 75 % HP, +1
        (1, (4992,), 0),  # the first roll jumps before the compare: nothing, no tie drawn
    ],
)
def test_expert_tailwind_tie(kaizo_tables, kaizo_names, kaizo_roark_party, tie, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Tailwind", "Accelerock", "", ""))
    s = _speeds(s, player=100, opp=100)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps), tie=tie)[0] == expected


def test_check_battler_ability_guess(kaizo_tables, kaizo_roark_party):
    """CheckBattlerAbility on an unlearned player ability (trainer_ai.c:1236-1251): two
    abilities, neither the one asked, answers ability 1, so AI_NOT_HAVE (:1242-1243); one
    ability answers that one (:1247-1251), AI_HAVE when it is the one asked."""
    t = kaizo_tables
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    boost = c.ABILITY_IDS["Speed Boost"]
    two = next(
        sp
        for sp in range(1, rows.shape[0])
        if 0 not in abilities[rows[sp]][:2]
        and boost not in abilities[rows[sp]][:2]
        and abilities[rows[sp], 0] != abilities[rows[sp], 1]
    )
    one = next(
        sp
        for sp in range(1, rows.shape[0])
        if abilities[rows[sp], 0] != 0 and abilities[rows[sp], 1] == 0
    )
    s = _state(t, kaizo_roark_party, BONSLY, (ACCELEROCK, 0, 0, 0))

    def ask(species, expected):
        p = _with_player(t, s, species, 0, known=False)
        ctx = c.at_slot(c.new_context(t, p, _outcome(t, p)), 0)
        return int(c.check_battler_ability(ctx, c.Battler.DEFENDER, expected).calc_temp)

    assert ask(two, boost) == c.AI_NOT_HAVE
    assert ask(one, int(abilities[rows[one], 0])) == c.AI_HAVE
    assert ask(one, boost if abilities[rows[one], 0] != boost else 1) == c.AI_NOT_HAVE


@pytest.mark.parametrize(
    ("speed", "hp", "jumps", "expected"),
    [
        # Expert_Tailwind (script.s:4982-5006); the AI's Speed 100 against the player's
        ("slower", 100, (4992,), 0),  # the first 64 roll jumps: nothing
        ("slower", 100, (), 1),  # above 75 %: +1
        ("slower", 50, (), 1),  # 31-75 %: +1 unless :4996 jumps
        ("slower", 50, (4996,), 0),
        ("slower", 20, (), -1),  # below 31 %: -1
        ("faster", 100, (), -1),  # faster: -1
    ],
)
def test_expert_tailwind(kaizo_tables, kaizo_names, kaizo_roark_party, speed, hp, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Tailwind", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=500 if speed == "slower" else 1, opp=100), 1, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("player", "speed", "hp", "jumps", "expected"),
    [
        # Expert_Payback (script.s:5169-5187), Payback (Dark)
        ("Machop", "slower", 100, (), -1),  # Fighting resists Dark: -1
        ("Tentacool", "slower", 100, (), 1),  # +1 unless :5179 jumps
        ("Tentacool", "slower", 100, (5179,), 0),
        ("Tentacool", "faster", 100, (), 0),  # faster: nothing
        ("Tentacool", "slower", 29, (), 0),  # below 30 %: nothing
    ],
)
def test_expert_payback(
    kaizo_tables, kaizo_names, kaizo_roark_party, player, speed, hp, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Payback", "Accelerock", "", ""))
    s = _with_player(t, s, n.species_names.index(player), 0, known=True)
    s = _hp(_speeds(s, player=500 if speed == "slower" else 1, opp=100), 1, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
@pytest.mark.parametrize("speeds", [(100, 100), (500, 100), (1, 100)])
def test_speed_faster_exact_equals_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, flags, speeds
):
    """Exact == loose for Tailwind and Payback, whose labels compare for COMPARE_SPEED_FASTER:
    at equal speed the tie's RandNext() & 1 decides (battle_lib.c:1437-1438 under Trick Room, :1445-1446 plain) and is a read
    of the exact layout (E1 Ruling R8); slower and faster too."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Tailwind", "Payback", "Accelerock", ""))
    s = _with_player(t, s, 72, 0, known=True)
    for hp in (100, 50):
        s2 = _hp(_speeds(s, player=speeds[0], opp=speeds[1]), 1, hp)
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 batch 12: sleep moves, Fake Out, Pluck, fixed-damage and counter moves, weather ------


def _asleep(s, side: int):
    return s._replace(status=s.status.at[side, 0].set(Status.SLEEP))


@pytest.mark.parametrize(
    ("move", "asleep", "basic", "expert"),
    [
        # Basic_CheckAttackerAsleep (script.s:660-663); Expert_SleepTalk (:3327-3333),
        # Expert_Nightmare (:3314-3316)
        ("Sleep Talk", False, -8, -5),
        ("Sleep Talk", True, 0, 10),
        ("Snore", False, -8, 2),
        ("Snore", True, 0, 2),
    ],
)
def test_sleep_moves(kaizo_tables, kaizo_names, kaizo_roark_party, move, asleep, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if asleep:
        s = _asleep(s, 1)
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize(("first", "basic"), [(True, 0), (False, -10)])
def test_fake_out(kaizo_tables, kaizo_names, kaizo_roark_party, first, basic):
    """Basic_CheckFirstTurnInBattle (script.s:829-833); Expert_FakeOut (:4074-4076) +2."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Fake Out", "Accelerock", "", ""))
    if not first:
        s = _turn(s, 3)
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == 2


@pytest.mark.parametrize(
    ("player_hp", "expected"),
    [(100, 0), (51, 0), (50, -1)],  # Expert_SuperFang (script.s:2800-2806)
)
def test_expert_super_fang(kaizo_tables, kaizo_names, kaizo_roark_party, player_hp, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Super Fang", "Accelerock", "", ""))
    assert _delta(t, _hp(s, 0, player_hp), 0x3, 0x7)[0] == expected


def test_basic_non_standard_damage_immune(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckNonStandardDamageOrChargeTurn (script.s:575): Counter (Fighting in Kaizo's
    move table) against a Ghost is immune, -10; against Gible, nothing."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Counter", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    ghost = _with_player(t, s, n.species_names.index("Gastly"), 0, known=True)
    assert _delta(t, ghost, 0x2, 0x3)[0] == -10


@pytest.mark.parametrize(
    ("speed", "hp", "jumps", "expected"),
    [
        # Expert_Reversal (script.s:3362-3395); the AI at Speed 100
        ("faster", 100, (), -1),  # above 33 %: -1
        ("faster", 25, (), 0),  # 21-33 %: nothing
        ("faster", 10, (), 1),  # 8-20 %: +1 unless :3388 jumps
        ("faster", 10, (3388,), 0),
        ("faster", 5, (), 2),  # below 8 %: +1, then the try
        ("slower", 70, (), -1),  # above 60 %: -1
        ("slower", 50, (), 0),  # 41-60 %: nothing
        ("slower", 30, (), 1),  # 40 % or less: the try
    ],
)
def test_expert_reversal(kaizo_tables, kaizo_names, kaizo_roark_party, speed, hp, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Reversal", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 1, hp)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("speed", "hp", "stage", "expected"),
    [
        # Expert_Superpower (script.s:4467-4488): Kaizo's Close Combat (effect 182, Fighting,
        # neutral on Gible; Kaizo's Superpower has effect 48)
        ("faster", 100, 0, -1),  # faster above 40 %: -1
        ("faster", 40, 0, 0),
        ("slower", 100, 0, -1),  # slower at 60 % or more: -1
        ("slower", 59, 0, 0),
        ("faster", 40, -1, -1),  # Attack below +0: -1
    ],
)
def test_expert_superpower(
    kaizo_tables, kaizo_names, kaizo_roark_party, speed, hp, stage, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Close Combat", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 1, hp)
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(stage))
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize(
    ("turn", "jumps", "expected"),
    [
        # Expert_Pluck (script.s:4957-4980), Flying against Gible (neutral)
        (0, (), 2),  # first turn: +1 unless :4968 jumps, then +1 unless :4972 jumps
        (0, (4968,), 1),
        (0, (4968, 4972), 0),
        (3, (), 1),  # a later turn: only the :4972 try
    ],
)
def test_expert_pluck(kaizo_tables, kaizo_names, kaizo_roark_party, turn, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _turn(_state(t, kaizo_roark_party, BONSLY, _ids(n, "Pluck", "Accelerock", "", "")), turn)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("move", "last", "case", "jumps", "expected"),
    [
        # Expert_Counter (script.s:3095-3163) / Expert_MirrorCoat (:3852-3920), the AI at
        # 100 % HP unless "low hp"
        ("Counter", "", "asleep", (), -1),
        ("Counter", "Tackle", "", (), 1),  # not taunted: no :3130 try; physical +1 unless :3136
        ("Counter", "Tackle", "", (3136,), 0),
        ("Counter", "Tackle", "taunted", (), 2),  # the :3130 +1 too
        ("Counter", "Water Gun", "", (), -1),  # a special last move: -1
        ("Counter", "", "", (), 0),  # no last move; Gible is Ground, a physical type: nothing
        ("Counter", "", "Psyduck", (), 4),  # a Water player: the +4 try (:3150, :3153)
        ("Counter", "", "Psyduck", (3150,), 0),
        ("Counter", "", "low hp", (), -2),  # 20 % HP: :3116 and :3121 each -1
        ("Mirror Coat", "Water Gun", "", (), 1),
        ("Mirror Coat", "Tackle", "", (), -1),
        ("Mirror Coat", "", "", (), 0),  # Gible is Dragon, a special type (:3929): nothing
        ("Mirror Coat", "", "Gastly", (), 4),  # Ghost/Poison, neither special: the +4 try
    ],
)
def test_expert_counter_mirror_coat(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, last, case, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _hp(s, 1, 20 if case == "low hp" else 100)
    if case == "asleep":
        s = _asleep(s, 0)
    if last:
        s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    if case == "taunted":
        s = s._replace(taunt=s.taunt.at[0].set(3))
    if case in ("Psyduck", "Gastly"):
        s = _with_player(t, s, n.species_names.index(case), 0, known=True)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("move", "weather", "hp", "basic", "expert"),
    [
        # Basic_CheckRainDance (script.s:786-801) / Expert_RainDance (:3737-3771), the AI
        # faster (no Swift Swim read); Basic_CheckHail (:853-871) / Expert_Hail (:4089-4119)
        ("Rain Dance", Weather.NONE, 100, 0, 0),
        ("Rain Dance", Weather.SUN, 100, 0, 1),
        ("Rain Dance", Weather.RAIN, 100, -8, 0),
        ("Rain Dance", Weather.NONE, 30, 0, -1),
        ("Hail", Weather.NONE, 100, 0, 0),
        ("Hail", Weather.SUN, 100, 0, 1),
        ("Hail", Weather.HAIL, 100, -8, 0),
        ("Hail", Weather.NONE, 30, 0, -1),
    ],
)
def test_weather_moves(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, weather, hp, basic, expert
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _hp(_speeds(s, player=1, opp=100), 1, hp)
    s = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch12_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose on a fight of batch 12's rolling labels (Counter, Mirror Coat, Pluck,
    Reversal), at equal speed and not, with the player's last move physical, special or
    none."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Counter", "Mirror Coat", "Pluck", "Reversal"))
    for player_speed, last, hp in ((100, "Tackle", 25), (1, "Water Gun", 100), (500, "", 60)):
        s2 = _hp(_speeds(s, player=player_speed, opp=100), 1, hp)
        move = n.move_names.index(last) if last else 0
        s2 = s2._replace(last_move=s2.last_move.at[0].set(move))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


@pytest.mark.parametrize(
    ("case", "jumps", "expected"),
    [
        # Expert_Protect (script.s:3502-3589), the AI's Bonsly with Protect against Gible
        ("plain", (), -1),  # no +2 (the 85 roll :3564 does not jump), -1 (:3571)
        ("plain", (3564,), 1),  # +2, -1
        ("plain", (3571,), 0),
        ("chain 1", (), -3),  # -1, then with a chain -1 and -1 more unless :3578 jumps
        ("chain 1", (3578,), -2),
        ("chain 2", (), -2),  # a chain above 1: -2 and end
        ("ai toxic", (), -2),  # badly poisoned, no Lock-On: -2 and end
        ("ai toxic lock on", (), 0),  # with the AI's Lock-On: end
        ("ai curse", (), -2),
        ("ai yawn", (), -2),
        ("player toxic", (), 1),  # +2, -1
        ("player curse", (), 1),
        ("player perish", (), 1),
        ("player yawn", (), 1),
        ("doubles", (), 1),  # LoadBattleType & BATTLE_TYPE_DOUBLES (:3561-3562)
        ("ai lock on", (), 1),  # :3563
    ],
)
def test_expert_protect(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Protect", "Accelerock", "", ""))
    side = 1 if case.startswith("ai") else 0
    if case.startswith("chain"):
        s = s._replace(protect_count=s.protect_count.at[1].set(int(case[-1])))
    if "toxic" in case:
        s = s._replace(status=s.status.at[side, 0].set(Status.TOXIC))
    if "curse" in case:
        s = s._replace(curse=s.curse.at[side].set(1))
    if "perish" in case:
        s = s._replace(
            perish_song=s.perish_song.at[side].set(1),
            perish_song_turns=s.perish_song_turns.at[side].set(2),
        )
    if "yawn" in case:
        s = s._replace(yawn=s.yawn.at[side].set(1))
    if "lock on" in case:
        s = s._replace(lock_on=s.lock_on.at[1].set(2), lock_on_target=s.lock_on_target.at[1].set(1))
    if case == "doubles":
        t = t._replace(battle_type=np.int32(3))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_protect_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for Protect with no chain, a chain of 1, and the player badly
    poisoned."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Protect", "Accelerock", "Headbutt", ""))
    for chain, toxic in ((0, False), (1, False), (1, True)):
        s2 = s._replace(protect_count=s.protect_count.at[1].set(chain))
        if toxic:
            s2 = s2._replace(status=s2.status.at[0, 0].set(Status.TOXIC))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 batch 14: Basic-only alt-power effects ----------------------------------------------


@pytest.mark.parametrize(
    ("move", "player", "expected"),
    [
        # Basic_Main (script.s:62-68) and Basic_CheckNonStandardDamageOrChargeTurn
        # (:572-585): a damaging immune move already scores -10 at Basic_CheckForImmunity
        ("Seismic Toss", "Shuppet", -10),  # Fighting against a Ghost: immune
        ("Seismic Toss", "Gible", 0),
        ("Night Shade", "Bidoof", -10),  # Ghost against a Normal: immune
        ("Grass Knot", "Gible", 0),
        ("Dragon Rage", "Gible", 0),
        ("SonicBoom", "Shuppet", -10),  # Normal against a Ghost
        ("Fury Cutter", "Gible", 0),
        # Present (122, power 1, not alt-power: not damaging) skips Basic_CheckForImmunity's
        # -10 (:62-63) and reaches this label, the one Kaizo case it decides: Normal on a Ghost
        ("Present", "Shuppet", -10),
        ("Present", "Gible", 0),
    ],
)
def test_basic_alt_power_effects(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, player, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if player != "Gible":
        s = _with_player(t, s, n.species_names.index(player), 0, known=True)
    assert policy.ai_refusals(t, s, n, layout_for(t, 0x17, s)) == []
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch14_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose on a fight of the batch 14 effects, whose damage the AI estimates
    through TrainerAI_CalcDamage's cases (engine.damage.ai_case_power / ai_case_fixed) or its
    default, at full HP and low HP."""
    t, n = kaizo_tables, kaizo_names
    moves = _ids(n, "Grass Knot", "Seismic Toss", "Dragon Rage", "Fury Cutter")
    s = _state(t, kaizo_roark_party, BONSLY, moves)
    for hp in (400, 30):
        s2 = s._replace(hp=s.hp.at[0, 0].set(hp))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


def _all_ivs(s, side: int, iv: int):
    return s._replace(ivs=s.ivs.at[side, 0].set(jnp.full(6, iv, s.ivs.dtype)))


def test_hidden_power_ai_type_and_power(kaizo_tables, kaizo_names, kaizo_roark_party):
    """TrainerAI_MoveType's and TrainerAI_CalcDamage's MOVE_HIDDEN_POWER cases
    (trainer_ai.c:3208-3220, :2973-2993): with every IV 31, type bits 63 give
    63 * 15 / 63 + 1 = 16, one more from TYPE_MYSTERY up: 17, Dark; power bits 63 give
    63 * 40 / 63 + 30 = 70. Dark is super effective on Shuppet (Ghost), so the
    effectiveness class is DOUBLE, where the table type (Normal) would be immune."""
    t, n = kaizo_tables, kaizo_names
    hp = n.move_names.index("Hidden Power")
    s = _all_ivs(_state(t, kaizo_roark_party, BONSLY, (hp, ACCELEROCK, 0, 0)), 1, 31)
    s = _with_player(t, s, n.species_names.index("Shuppet"), 0, known=True)
    assert int(damage.ai_move_type(t, s, 1, hp)) == c.TYPE_IDS["Dark"]
    assert int(damage.ai_move_type(t, s, 1, ACCELEROCK)) == int(t.move_type[ACCELEROCK])
    assert int(damage.ai_case_power(t, s, 1, hp)) == 70
    assert int(c.effectiveness_class(t, s, hp)) == TypeMulti.DOUBLE
    assert policy.ai_refusals(t, s, n, layout_for(t, 0x17, s)) == []
    _assert_exact_equals_loose(t, s, layout_for(t, 0x17, s))


@pytest.mark.parametrize(
    ("case", "basic"),
    [
        # Basic_CheckOHKOWouldFail (script.s:554-564), Kaizo Guillotine (Normal)
        ("same level", 0),
        ("player higher", -10),  # IfLevel CHECK_LOWER_THAN_TARGET (:563)
        ("ghost", -10),  # immune (:556)
        ("sturdy", -10),  # a known Sturdy player (:559-560)
    ],
)
def test_basic_ohko(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Guillotine", "Accelerock", "", ""))
    s = s._replace(level=s.level.at[0, 0].set(s.level[1, 0]))
    if case == "player higher":
        s = s._replace(level=s.level.at[0, 0].set(s.level[1, 0] + 1))
    elif case == "ghost":
        s = _with_player(t, s, n.species_names.index("Shuppet"), 0, known=True)
    elif case == "sturdy":  # Kaizo has no Sturdy species: a synthetic row for Gible
        sturdy = c.ABILITY_IDS["Sturdy"]
        t = _with_abilities(t, int(s.species[0, 0]), [sturdy, sturdy])
        s = know_ability(t, s)
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(("jumps", "expected"), [((), 1), ((2794,), 0)])
def test_expert_ohko(kaizo_tables, kaizo_names, kaizo_roark_party, jumps, expected):
    """Expert_OHKOMove (script.s:2792-2798): +1 unless the 192 roll (:2794) jumps."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Guillotine", "Accelerock", "", ""))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


# --- E1 batch 15: U-turn ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "jumps", "expected"),
    [
        # Expert_UTurn (script.s:5085-5145), Bonsly's U-turn (Bug, 80) against Gible (neutral;
        # Accelerock is resisted, so no super-effective move), the AI faster, the player at 100 %
        ("weak bench", (), -2),  # no member out-damages it: -2 unless :5119 jumps, and end
        ("weak bench", (5119,), 3),  # then 75 % +1, 50 % +1, faster +1
        ("strong bench", (), 3),  # a bench Earthquake (100) beats U-turn as Bonsly: to the HP
        ("strong bench", (5130, 5134), 1),  # only the faster +1
        ("no bench", (), 0),  # no other living Pokemon (:5104-5105)
        ("resisted", (), -1),  # Charmander (Fire) resists Bug (:5101-5103)
    ],
)
def test_expert_u_turn(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "U-turn", "Accelerock", "", ""))
    s = _speeds(s, player=1, opp=100)
    bench_moves = (
        _ids(n, "Earthquake", "", "", "")
        if case == "strong bench"
        else _ids(n, "Tackle", "", "", "")
    )
    s = s._replace(moves=s.moves.at[1, 1].set(jnp.asarray(bench_moves, s.moves.dtype)))
    if case == "no bench":
        s = s._replace(hp=s.hp.at[1, 1].set(0))
    elif case == "resisted":
        s = _with_player(t, s, n.species_names.index("Charmander"), 0, known=True)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


def test_party_member_deals_more_damage_as_the_active(kaizo_tables, kaizo_names, kaizo_roark_party):
    """IfPartyMemberDealsMoreDamage (trainer_ai.c:2106-2174) calculates the party member's
    moves as the AI's active battler: the bench's Earthquake beats U-turn by Bonsly's own
    Attack, and a weaker bench move (Tackle) does not."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "U-turn", "Accelerock", "", ""))
    ctx = c.at_slot(c.new_context(t, s, _outcome(t, s)), 0)
    for bench, jumps in (("Earthquake", True), ("Tackle", False)):
        s2 = s._replace(
            moves=s.moves.at[1, 1].set(jnp.asarray(_ids(n, bench, "", "", ""), s.moves.dtype))
        )
        ctx2 = c.at_slot(c.new_context(t, s2, _outcome(t, s2)), 0)
        assert bool(c.if_party_member_deals_more_damage(ctx2, c.USE_MAX_DAMAGE).jump) == jumps, (
            bench
        )
    assert not bool(c.if_has_super_effective_move(ctx).jump)  # U-turn neutral, Accelerock resisted


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_u_turn_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    """Exact == loose for U-turn at equal speed (its FASTER tie) and not, against a Grass
    player (super effective: the :5114 roll) and Gible."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "U-turn", "Accelerock", "", ""))
    for player, speed in (("Gible", 100), ("Turtwig", 1)):
        s2 = _speeds(s, player=speed, opp=100)
        if player != "Gible":
            s2 = _with_player(t, s2, n.species_names.index(player), 0, known=True)
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


def test_party_member_damage_ties_as_the_active(kaizo_tables, kaizo_names, kaizo_roark_party):
    """A tie decides (batch 15 review): Cranidos's Mega Punch (Normal, 80, physical, no STAB
    for either Rock Pokemon, neutral on Gible) calculated as Bonsly equals Bonsly's U-turn
    (Bug, 80, neutral), so the strict ``>`` (trainer_ai.c:2165) does not jump; calculated
    with Cranidos's own Attack (base 125 against Bonsly's 80) it would."""
    t, n = kaizo_tables, kaizo_names
    u_turn, mega_punch = n.move_names.index("U-turn"), n.move_names.index("Mega Punch")
    s = _state(t, kaizo_roark_party, BONSLY, (u_turn, 0, 0, 0))
    s = s._replace(moves=s.moves.at[1, 1].set(jnp.asarray((mega_punch, 0, 0, 0), s.moves.dtype)))
    as_active = int(c.ai_damage(t, s, mega_punch, c.SIM_ROLL_MAX))
    assert as_active == int(c.ai_damage(t, s, u_turn, c.SIM_ROLL_MAX)) > 0  # the tie
    ctx = c.at_slot(c.new_context(t, s, _outcome(t, s)), 0)
    assert not bool(c.if_party_member_deals_more_damage(ctx, c.USE_MAX_DAMAGE).jump)


# --- E1 batch 16: Gardenia's Sunny Day, Safeguard, Blizzard, Belly Drum, Endeavor, ----------
# --- Focus Punch, Pain Split -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "basic"),
    [("plain", 0), ("safeguard", -8)],  # Basic_CheckAlreadyUnderSafeguard (script.s:754-757)
)
def test_basic_safeguard(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Safeguard", "Accelerock", "", ""))
    if case == "safeguard":
        s = s._replace(safeguard=s.safeguard.at[1].set(3))
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(
    ("weather", "hp", "basic", "expert"),
    [
        # Basic_CheckSunnyDay (script.s:803-820) / Expert_SunnyDay (:3773-3801)
        (Weather.NONE, 100, 0, 0),
        (Weather.SUN, 100, -8, 0),
        (Weather.RAIN, 100, 0, 1),
        (Weather.NONE, 30, 0, -1),
    ],
)
def test_sunny_day(kaizo_tables, kaizo_names, kaizo_roark_party, weather, hp, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Sunny Day", "Accelerock", "", ""))
    s = _hp(s, 1, hp)._replace(weather=jnp.asarray(weather, s.weather.dtype))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize(
    ("hp", "attack", "basic", "expert"),
    [
        # Basic_CheckBellyDrum (script.s:335-337, falling into :343) / Expert_BellyDrum
        # (:3803-3812)
        (100, 0, 0, 0),
        (50, 0, -10, -2),
        (80, 0, 0, -2),
        (100, 6, -10, 0),  # Attack at +6: Basic_CheckHighStatStage_Attack's -10
    ],
)
def test_belly_drum(kaizo_tables, kaizo_names, kaizo_roark_party, hp, attack, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Belly Drum", "Accelerock", "", ""))
    s = _hp(s, 1, hp)._replace(stages=s.stages.at[1, Stat.ATK].set(attack))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize(
    ("move", "speed", "own", "player", "expected"),
    [
        # Expert_Endeavor (script.s:4583-4608): player below 70 % -1; faster above 40 % -1,
        # else +1; slower above 50 % -1, else +1
        ("Endeavor", "faster", 100, 100, -1),
        ("Endeavor", "faster", 30, 100, 1),
        ("Endeavor", "slower", 60, 100, -1),
        ("Endeavor", "slower", 50, 100, 1),
        ("Endeavor", "faster", 30, 50, -1),
        # equal speed: SLOWER never holds at a tie (battle_lib.c:1401-1447; no tie read,
        # plan K3c Ruling 3), so the faster branch: above 40 % -1
        ("Endeavor", "tied", 45, 100, -1),
        # Expert_PainSplit (:3287-3312): player below 80 %; slower above 60 %
        ("Pain Split", "faster", 30, 100, 1),
        ("Pain Split", "faster", 100, 100, -1),
        ("Pain Split", "slower", 70, 100, -1),
        ("Pain Split", "slower", 50, 100, 1),
        ("Pain Split", "faster", 30, 70, -1),
    ],
)
def test_hp_trade(kaizo_tables, kaizo_names, kaizo_roark_party, move, speed, own, player, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _speeds(s, player={"faster": 1, "tied": 100}.get(speed, 500), opp=100)
    s = _hp(_hp(s, 1, own), 0, player)
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize(
    ("case", "jumps", "expected"),
    [
        # Expert_FocusPunch (script.s:4129-4163), Fighting against Gible (neutral)
        ("first turn", (), 0),  # its first turn: nothing
        ("later turn", (), 1),  # +1 unless the 200 roll (:4148) jumps
        ("later turn", (4148,), 0),
        ("asleep", (), 1),
        ("confused", (), 1),  # +1 unless the 100 roll (:4157) jumps
        ("confused", (4157,), 0),
        ("infatuated", (), 1),  # :4144, the same try
        ("substitute", (), 5),  # behind its own Substitute: +5
    ],
)
def test_expert_focus_punch(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Focus Punch", "Accelerock", "", ""))
    if case != "first turn":
        s = _turn(s, 3)
    if case == "asleep":
        s = s._replace(status=s.status.at[0, 0].set(Status.SLEEP))
    elif case == "confused":
        s = s._replace(confusion=s.confusion.at[0].set(2))
    elif case == "infatuated":
        s = s._replace(infatuated=s.infatuated.at[0].set(1))
    elif case == "substitute":
        s = s._replace(substitute_hp=s.substitute_hp.at[1].set(10))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(("jumps", "expected"), [((), -2), ((6374,), 0)])
def test_eval_attack_focus_punch(kaizo_tables, kaizo_names, kaizo_roark_party, jumps, expected):
    """EvalAttack_Main (script.s:6366): Focus Punch takes EvalAttack_MaybeDeprioritize, -2
    unless the 51 roll (:6374) jumps; as a no-calc move it makes no damage comparison."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Focus Punch", "Accelerock", "", ""))
    assert _delta(t, s, 0x1, 0x3, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("player", "weather", "jumps", "expected"),
    [
        # Expert_Blizzard (script.s:6216-6234), Kaizo Blizzard (Ice, effect 260)
        ("Gible", Weather.NONE, (), 0),
        ("Gible", Weather.HAIL, (), 1),
        ("Charmander", Weather.NONE, (), -3),  # Fire resists Ice: -3 unless :6229 jumps
        ("Charmander", Weather.NONE, (6229,), 0),
    ],
)
def test_expert_blizzard(
    kaizo_tables, kaizo_names, kaizo_roark_party, player, weather, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Blizzard", "Accelerock", "", ""))
    s = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
    if player != "Gible":
        s = _with_player(t, s, n.species_names.index(player), 0, known=True)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch16_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    t, n = kaizo_tables, kaizo_names
    s = _state(
        t, kaizo_roark_party, BONSLY, _ids(n, "Focus Punch", "Blizzard", "Pain Split", "Sunny Day")
    )
    for hp, turn in ((100, 0), (30, 3)):
        s2 = _turn(_hp(s, 1, hp), turn)
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 batch 17: Roar / Whirlwind, Substitute, Dream Eater ---------------------------------


@pytest.mark.parametrize(
    ("case", "jumps", "basic", "expert"),
    [
        # Basic_CheckCanForceSwitch (script.s:499-509) / Expert_ForceSwitch (:2574-2611)
        ("fresh", (), 0, -3),  # in 0 turns, no hazard, no stage above +2: -3
        ("long", (), 0, 4),  # in more than 3 turns: +2 (:2603), +2 (:2607)
        ("long", (2603,), 0, 2),
        ("spikes", (), 0, 2),  # Spikes on the player's side: the second try only
        ("spikes", (2607,), 0, 0),
        ("boosted", (), 0, 2),  # the player's Attack at +3 (decomp 9), above decomp 8
        ("toxic spikes", (), 0, 2),  # Toxic Spikes on the player's side (:2593)
        ("alone", (), -10, -3),  # no other living Pokemon on the player's side
    ],
)
def test_force_switch(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Roar", "Accelerock", "", ""))
    if case == "long":
        s = _turn(s, 5)
    elif case == "spikes":
        s = s._replace(spikes=s.spikes.at[0].set(1))
    elif case == "toxic spikes":
        s = s._replace(toxic_spikes=s.toxic_spikes.at[0].set(1))
    elif case == "boosted":
        s = s._replace(stages=s.stages.at[0, Stat.ATK].set(3))
    elif case == "alone":
        s = s._replace(hp=s.hp.at[0, 1].set(0))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


@pytest.mark.parametrize(
    ("own", "speed", "last", "jumps", "basic", "expert"),
    [
        # Basic_CheckCannotSubstitute (script.s:633-637) / Expert_Substitute (:2976-3042)
        (100, "faster", "", (), 0, 0),  # above 90 %: straight to the last move: none
        (100, "faster", "Toxic", (), 0, 1),  # a status move, the player unstatused
        (100, "faster", "Toxic", (3038,), 0, 0),
        (100, "slower", "Toxic", (), 0, 0),  # slower: nothing (:3014)
        (80, "faster", "", (), 0, -1),  # above 70 %: one try (:3010)
        (60, "faster", "", (), 0, -2),  # above 50 %: two (:3006, :3010)
        (40, "faster", "", (), 0, -3),  # three (:3002, :3006, :3010)
        (20, "faster", "", (), -10, -3),  # below 26 %: Basic -10
        # the last-move table (:3017-3039)
        (100, "faster", "Confuse Ray", (), 0, 1),  # a confusing move, the player unconfused
        (100, "faster", "Confuse Ray confused", (), 0, 0),
        (100, "faster", "Leech Seed", (), 0, 1),  # unseeded: falls into TryScorePlus1
        (100, "faster", "Leech Seed seeded", (), 0, 0),
        # knowing Focus Punch (:2994-2996): +1 unless the 96 roll (:2995) jumps
        (100, "faster", "focus punch", (), 0, 1),
        (100, "faster", "focus punch", (2995,), 0, 0),
    ],
)
def test_substitute(
    kaizo_tables, kaizo_names, kaizo_roark_party, own, speed, last, jumps, basic, expert
):
    t, n = kaizo_tables, kaizo_names
    second = "Focus Punch" if last == "focus punch" else ""
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Substitute", "Accelerock", second, ""))
    s = _hp(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 1, own)
    if last.endswith("confused"):
        s = s._replace(confusion=s.confusion.at[0].set(2))
    if last.endswith("seeded"):
        s = s._replace(leech_seed=s.leech_seed.at[0].set(1))
    move = last.removesuffix(" confused").removesuffix(" seeded")
    if move and move != "focus punch":
        s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index(move)))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


def test_basic_substitute_already_behind_one(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckCannotSubstitute (script.s:635): behind a Substitute already, -8."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Substitute", "Accelerock", "", ""))
    s = s._replace(substitute_hp=s.substitute_hp.at[1].set(10))
    assert _delta(t, s, 0x2, 0x3)[0] == -8


@pytest.mark.parametrize(
    ("asleep", "jumps", "basic", "expert"),
    [
        # Basic_CheckDreamEater (script.s:329-333) / Expert_DreamEater (:1886-1905), Kaizo
        # Dream Eater (Psychic) against Gible (neutral)
        (False, (), -8, 0),
        (True, (), 0, 3),  # +3 unless the 51 roll (:1897) jumps
        (True, (1897,), 0, 0),
    ],
)
def test_dream_eater(kaizo_tables, kaizo_names, kaizo_roark_party, asleep, jumps, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Dream Eater", "Accelerock", "", ""))
    if asleep:
        s = s._replace(status=s.status.at[0, 0].set(Status.SLEEP))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_batch17_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags, states):
    t, n = kaizo_tables, kaizo_names
    s = _state(
        t, kaizo_roark_party, BONSLY, _ids(n, "Roar", "Substitute", "Dream Eater", "Accelerock")
    )
    for hp, turn, last in ((100, 5, "Toxic"), (40, 0, ""))[:states]:
        s2 = _turn(_hp(s, 1, hp), turn)
        if last:
            s2 = s2._replace(last_move=s2.last_move.at[0].set(n.move_names.index(last)))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


@pytest.mark.parametrize(
    ("own", "speed"), [(100, "faster"), (60, "faster"), (30, "slower"), (30, "faster")]
)
def test_roost_scores_as_recover(kaizo_tables, kaizo_names, kaizo_roark_party, own, speed):
    """Roost (214) takes Recover's labels, Basic_CheckCanRecoverHP and Expert_Recovery
    (script.s:253, :1758); the roosting turn flag is cleared before command selection
    (battle_lib.c:2229-2231), so its BASIC and EXPERT scores are Recover's."""
    t, n = kaizo_tables, kaizo_names

    def scored(move):
        s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
        s = _hp(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 1, own)
        return _delta(t, s, 0x2, 0x3)[0], _delta(t, s, 0x3, 0x7)[0]

    assert scored("Roost") == scored("Recover")


# --- E1 prep batch P1 (on E2's state): Metal Burst, Future Sight, ACC_DOWN, Minimize, -------
# --- Mean Look, Knock Off, Imprison --------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "basic"),
    [
        # Basic_CheckMetalBurst (script.s:1160-1182), Kaizo Metal Burst (Steel)
        ("faster", -10),  # the AI faster: -10 (:1179)
        ("slower", 0),
        ("shiny player", -10),  # the player known to hold a Shiny Stone (:1169)
        ("shiny self", 0),  # the AI holds one: straight to _Terminate (:1176)
    ],
)
def test_basic_metal_burst(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Metal Burst", "Accelerock", "", ""))
    s = _speeds(s, player=500 if case == "slower" else 1, opp=100)
    if case == "shiny player":
        s = s._replace(ai_item_known=s.ai_item_known.at[0].set(c.ITEM_SHINY_STONE))
    elif case == "shiny self":
        s = s._replace(item=s.item.at[1, 0].set(c.ITEM_SHINY_STONE))
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(
    ("own", "asleep", "jumps", "expected"),
    [
        # Expert_MetalBurst (script.s:5029-5083), no last move by the player
        (100, False, (), 1),  # +1 unless the 192 roll (:5062) jumps; then not taunted: end
        (100, False, (5062,), 0),
        (20, False, (), -1),  # -1 (:5053), -1 (:5058), +1 (:5062)
        (100, True, (), -1),  # the player asleep: -1
    ],
)
def test_expert_metal_burst(
    kaizo_tables, kaizo_names, kaizo_roark_party, own, asleep, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Metal Burst", "Accelerock", "", ""))
    s = _hp(s, 1, own)
    if asleep:
        s = s._replace(status=s.status.at[0, 0].set(Status.SLEEP))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(("side", "basic"), [(None, 0), (0, -12), (1, -12)])
def test_basic_future_sight(kaizo_tables, kaizo_names, kaizo_roark_party, side, basic):
    """Basic_CheckFutureSight (script.s:823-827): Future Sight pending on either side, -12."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Future Sight", "Accelerock", "", ""))
    if side is not None:
        s = s._replace(future_sight=s.future_sight.at[side].set(1))
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(("jumps", "expected"), [((), 4), ((6405,), 0)])
def test_eval_attack_future_sight_kill(
    kaizo_tables, kaizo_names, kaizo_roark_party, jumps, expected
):
    """EvalAttack_ApplyKillBonuses (script.s:6396 -> :6403): a killing Future Sight takes
    +4 unless the 170 roll (:6405) jumps."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Future Sight", "Accelerock", "", ""))
    s = _hp(s, 0, 1)
    assert _delta(t, s, 0x1, 0x3, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("case", "basic", "expert"),
    [
        # Basic_CheckLowStatStage_Accuracy (script.s:453-460) / Expert_StatusAccuracyDown
        # (:2431-2498), Sand-Attack against Gible
        ("plain", 0, 0),  # both at full HP: to the HP ranges, above 70 %: end
        ("floored", -10, 0),  # the player's accuracy at -6
        ("ingrain", 0, 1),  # the AI ingrained: +1 unless :2473 jumps
        ("cursed", 0, 2),  # the player cursed: +2 unless :2484 jumps
        ("hurt", 0, -1),  # the AI at 50 %: -1 (:2453), then its range ends (player acc +0)
    ],
)
def test_accuracy_down(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Sand-Attack", "Accelerock", "", ""))
    if case == "floored":
        s = s._replace(stages=s.stages.at[0, Stat.ACC].set(-6))
    elif case == "ingrain":
        s = s._replace(ingrain=s.ingrain.at[1].set(1))
    elif case == "cursed":
        s = s._replace(curse=s.curse.at[0].set(1))
    elif case == "hurt":
        s = _hp(s, 1, 50)
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize(
    "move", ["Minimize", "Double Team"]
)  # EVA_UP_2_MINIMIZE (108), EVA_UP (16)
@pytest.mark.parametrize(
    ("evasion", "simple", "jumps", "basic", "expert"),
    [
        # Basic_CheckHighStatStage_Evasion (script.s:407-418) / Expert_StatusEvasionUp
        # (:2184-2253): at full HP +3 unless :2204 jumps, then the HP ranges end it
        (0, False, (), 0, 3),
        (0, False, (2204,), 0, 0),
        (6, False, (), -10, 2),  # +6: Basic -10; Expert +3, then -1 unless :2209 jumps
        (3, True, (), -10, None),  # Simple: above +2 is already the top (:412-413)
    ],
)
def test_evasion_up(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, evasion, simple, jumps, basic, expert
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = s._replace(stages=s.stages.at[1, Stat.EVA].set(evasion))
    if simple:  # a species whose ability slot 0 is Simple (the AI reads the species')
        abilities = np.asarray(t.species_abilities_decomp)
        rows = np.asarray(t.species_form_row)[:, 0]
        species = next(
            sp
            for sp in range(1, rows.shape[0])
            if abilities[rows[sp], 0] == c.ABILITY_IDS["Simple"]
        )
        s = as_species(t, s, 1, 0, species)
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    if expert is not None:
        assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


@pytest.mark.parametrize(
    ("case", "jumps", "basic", "expert"),
    [
        # Basic_CheckMeanLook (script.s:674-677) / Expert_BindingMove (:2808-2825)
        ("plain", (), 0, 0),
        ("trapped", (), -10, 0),
        ("toxic", (), 0, 1),  # the player badly poisoned: +1 unless :2821 jumps
        ("toxic", (2821,), 0, 0),
    ],
)
def test_mean_look(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Mean Look", "Accelerock", "", ""))
    if case == "trapped":
        s = s._replace(mean_look=s.mean_look.at[0].set(1))
    elif case == "toxic":
        s = s._replace(status=s.status.at[0, 0].set(Status.TOXIC))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


@pytest.mark.parametrize(
    ("case", "jumps", "basic", "expert"),
    [
        # Basic_CheckCanRemoveItem (script.s:898-904) / Expert_KnockOff (:4571-4581)
        ("first turn", (), 0, 0),  # its first turn ends the Expert label
        ("later", (), 0, 1),  # +1 unless :4577 jumps
        ("later", (4577,), 0, 0),
        ("later no item", (), -10, 1),  # the player holds nothing (its true item)
    ],
)
def test_knock_off(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Knock Off", "Accelerock", "", ""))
    if case.startswith("later"):
        s = _turn(s, 3)
    if case.endswith("no item"):
        s = s._replace(item=s.item.at[0, 0].set(0))
    else:
        s = s._replace(item=s.item.at[0, 0].set(c.ITEM_SHINY_STONE))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


@pytest.mark.parametrize(
    ("case", "basic", "expert"),
    [
        # Basic_CheckCanImprison (script.s:917-921) / Expert_Imprison (:4634-4642)
        ("later", 0, 2),  # not its first turn: +2 unless :4638 jumps
        ("imprisoning", -10, 2),
        ("imprisoned", -10, 2),
    ],
)
def test_imprison(kaizo_tables, kaizo_names, kaizo_roark_party, case, basic, expert):
    t, n = kaizo_tables, kaizo_names
    s = _turn(_state(t, kaizo_roark_party, BONSLY, _ids(n, "Imprison", "Accelerock", "", "")), 3)
    if case == "imprisoning":
        s = s._replace(imprison=s.imprison.at[1].set(1))
    elif case == "imprisoned":
        s = s._replace(imprisoned=s.imprisoned.at[0].set(1))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == expert


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_p1_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags, states):
    t, n = kaizo_tables, kaizo_names
    s = _state(
        t, kaizo_roark_party, BONSLY, _ids(n, "Metal Burst", "Sand-Attack", "Minimize", "Knock Off")
    )
    for own, turn in ((100, 0), (40, 3))[:states]:
        s2 = _turn(_hp(s, 1, own), turn)
        s2 = s2._replace(ingrain=s2.ingrain.at[1].set(1))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 prep batch P2 (on E2's state): Copycat, Ingrain, the chart's Ingrain and Roost -------


@pytest.mark.parametrize(
    ("turn", "speed", "basic"),
    [
        # Basic_CheckCopycat (script.s:1383-1390)
        (0, "faster", -10),
        (0, "slower", 0),
        (3, "faster", 0),  # not the battle's first turn: _Terminate
    ],
)
def test_basic_copycat(kaizo_tables, kaizo_names, kaizo_roark_party, turn, speed, basic):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Copycat", "Accelerock", "", ""))
    s = _turn(_speeds(s, player=1 if speed == "faster" else 500, opp=100), turn)
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(
    ("speed", "last", "jumps", "expected"),
    [
        # Expert_Copycat (script.s:5552-5580), Bonsly's Accelerock its best against Gible
        ("slower", "", (), -1),  # MOVE_NONE: not more, not encouraged: -1 unless :5576
        ("slower", "", (5576,), 0),
        ("slower", "Earthquake", (), 0),  # Gible's Earthquake on itself out-damages: _End
        ("faster", "Earthquake", (), 2),  # +2 unless :5568 jumps
        ("faster", "Earthquake", (5568,), 0),
        ("faster", "Toxic", (), 2),  # encouraged: +2 unless :5563 jumps
        ("faster", "Toxic", (5563,), 0),
        ("faster", "Growl", (), -1),  # neither: on to _CheckMoveEncouraged
    ],
)
def test_expert_copycat(kaizo_tables, kaizo_names, kaizo_roark_party, speed, last, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Copycat", "Accelerock", "", ""))
    s = _turn(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 3)
    if last:
        s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


def test_battler_deals_more_damage_counts_a_status_move(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    """IfBattlerDealsMoreDamage (trainer_ai.c:2187-2240) calculates the player's last move
    with no damaging-move test: CalcMoveDamage's ``+ 2`` through the chart (battle_lib.c:7078),
    the player attacking itself. Against an AI with no damaging move (best 0) MOVE_NONE (a
    physical Normal move of power 0, neutral on Gible) deals 2 and jumps; Toxic, Poison into
    Gible's Ground, deals 1 and jumps; with Accelerock the AI's best is above both."""
    t, n = kaizo_tables, kaizo_names
    toxic = n.move_names.index("Toxic")
    for moves, jumps in (
        (("Copycat", "Sand-Attack", "", ""), True),
        (("Copycat", "Accelerock", "", ""), False),
    ):
        s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
        for last, dealt in ((0, 2), (toxic, 1)):
            s2 = s._replace(last_move=s.last_move.at[0].set(last))
            got = ai_damage_for_roll(t, s2, 0, last, c.SIM_ROLL_MAX, def_side=0, ivs=s2.ivs[1, 0])
            assert int(got) == dealt, last
            ctx = c.at_slot(c.new_context(t, s2, _outcome(t, s2)), 0)
            assert (
                bool(c.if_battler_deals_more_damage(ctx, c.Battler.DEFENDER, c.USE_MAX_DAMAGE).jump)
                == jumps
            )


def test_a_status_move_takes_no_type_chart_bonus(kaizo_tables, kaizo_names, kaizo_roark_party):
    """ApplyTypeMultiplier sets the effectiveness flags only for a move with table power
    (battle_lib.c:2627, :7568), and Tinted Lens needs it too (:2667-2669): Gible's Toxic on
    itself (Poison into Ground, 2 -> 1) stays 1 with Tinted Lens, review of prep P2."""
    from nuzline.engine.ids import AbilityId

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Copycat", "Accelerock", "", ""))
    toxic = n.move_names.index("Toxic")
    for ability in (int(s.ability[0, 0]), AbilityId.TINTED_LENS):
        s2 = s._replace(ability=s.ability.at[0, 0].set(ability))
        got = ai_damage_for_roll(t, s2, 0, toxic, c.SIM_ROLL_MAX, def_side=0, ivs=s2.ivs[1, 0])
        assert int(got) == 1, ability


@pytest.mark.parametrize(("ingrained", "basic"), [(False, 0), (True, -10)])
def test_ingrain(kaizo_tables, kaizo_names, kaizo_roark_party, ingrained, basic):
    """Basic_CheckAlreadyIngrained (script.s:906-909) -10 when ingrained; Expert_Ingrain
    (:4463-4465) changes nothing."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Ingrain", "Accelerock", "", ""))
    s = s._replace(ingrain=s.ingrain.at[1].set(int(ingrained)))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == 0


def test_ai_chart_reads_ingrain_and_roosting(kaizo_tables, kaizo_names, kaizo_roark_party):
    """The AI's ApplyTypeChart against the player (BasicTypeMulApplies, battle_lib.c:
    2534-2543): an ingrained Flying player loses its Ground immunity, a roosting one every
    Flying row (Rock no longer super effective)."""
    from nuzline.ai.switching import apply_type_chart

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Accelerock", "", "", ""))
    s = _with_player(t, s, n.species_names.index("Starly"), 0, known=True)
    earthquake, accelerock = n.move_names.index("Earthquake"), n.move_names.index("Accelerock")
    assert int(c.effectiveness_class(t, s, earthquake)) == TypeMulti.IMMUNE
    rooted = s._replace(ingrain=s.ingrain.at[0].set(1))
    assert int(c.effectiveness_class(t, rooted, earthquake)) != TypeMulti.IMMUNE
    roosted = s._replace(roosting=s.roosting.at[0].set(1))
    flying = int(apply_type_chart(t, s, 1, accelerock, 40))
    assert int(apply_type_chart(t, roosted, 1, accelerock, 40)) * 2 == flying


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_p2_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags, states):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Copycat", "Accelerock", "Ingrain", ""))
    for speed, turn, last in ((100, 0, ""), (1, 3, "Toxic"), (500, 3, "Earthquake"))[:states]:
        s2 = _turn(_speeds(s, player=speed, opp=100), turn)
        if last:
            s2 = s2._replace(last_move=s2.last_move.at[0].set(n.move_names.index(last)))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 prep batch P3 (on E2's state): Baton Pass, AI_FLAG_BATON_PASS -------------------------


@pytest.mark.parametrize(("alone", "basic"), [(False, 0), (True, -10)])
def test_basic_baton_pass(kaizo_tables, kaizo_names, kaizo_roark_party, alone, basic):
    """Basic_CheckBatonPass (script.s:780-784): no other living party member, -10."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Baton Pass", "Accelerock", "", ""))
    if alone:
        s = s._replace(hp=s.hp.at[1, 1].set(0))
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(
    ("stage", "speed", "own", "jumps", "expected"),
    [
        # Expert_BatonPass (script.s:3651-3704), the attacker's Attack stage
        (0, "faster", 100, (), -2),
        (3, "faster", 50, (), 2),  # +3, faster at 60 % or less: +2 unless :3680 jumps
        (3, "faster", 50, (3680,), 0),
        (3, "faster", 100, (), 0),
        (3, "slower", 65, (), 2),  # slower at 70 % or less
        (2, "faster", 100, (), -2),  # +2, faster above 60 %
        (2, "faster", 50, (), 0),
        (2, "slower", 50, (), 0),  # slower below 70 %: _End
        (2, "slower", 100, (), -2),
    ],
)
def test_expert_baton_pass(
    kaizo_tables, kaizo_names, kaizo_roark_party, stage, speed, own, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Baton Pass", "Accelerock", "", ""))
    s = _hp(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 1, own)
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(stage))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(
    ("turn", "own", "stage", "last", "expected"),
    [
        # BatonPass_Main (script.s:6564-6635): Swords Dance, Accelerock (damaging: nothing),
        # Baton Pass, Protect, the attacker knowing Baton Pass
        (0, 100, 0, "", [5, 0, -2, 2]),  # first turn: +5; Baton Pass -2; Protect +2
        (3, 100, 2, "", [1, 0, 2, 2]),  # +1; Attack +2: Baton Pass +2
        (3, 50, 0, "Protect", [-10, 0, 0, -2]),  # below 60 %: -10; after its Protect -2
    ],
)
def test_baton_pass_flag(
    kaizo_tables, kaizo_names, kaizo_roark_party, turn, own, stage, last, expected
):
    t, n = kaizo_tables, kaizo_names
    moves = _ids(n, "Swords Dance", "Accelerock", "Baton Pass", "Protect")
    s = _hp(_turn(_state(t, kaizo_roark_party, BONSLY, moves), turn), 1, own)
    s = s._replace(stages=s.stages.at[1, Stat.ATK].set(stage))
    if last:
        s = s._replace(last_move=s.last_move.at[1].set(n.move_names.index(last)))
    assert _delta(t, s, 0x1, 0x41) == expected


@pytest.mark.parametrize(("jumps", "expected"), [((), 4), (((0, 6577),), 0), (((0, 6591),), 0)])
def test_baton_pass_flag_without_baton_pass(
    kaizo_tables, kaizo_names, kaizo_roark_party, jumps, expected
):
    """Without a Baton Pass effect the 80 roll (:6577) can end it; then any other status move
    +3 unless the 20 roll (:6591) jumps, falling into _SetupAtHighHP (+1 later on)."""
    t, n = kaizo_tables, kaizo_names
    s = _turn(_state(t, kaizo_roark_party, BONSLY, _ids(n, "Sand-Attack", "Accelerock", "", "")), 3)
    assert _delta(t, s, 0x1, 0x41, jumps)[0] == expected


@pytest.mark.parametrize(
    ("flags", "states"),
    [
        pytest.param(0x57, 1, id="0x57-one"),
        pytest.param(0x57, None, id="0x57", marks=pytest.mark.slow),
    ],
)
def test_p3_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags, states):
    t, n = kaizo_tables, kaizo_names
    moves = _ids(n, "Swords Dance", "Sand-Attack", "Baton Pass", "Protect")
    s = _state(t, kaizo_roark_party, BONSLY, moves)
    for own, turn, stage in ((100, 0, 0), (50, 3, 3), (100, 3, 2))[:states]:
        s2 = _turn(_hp(s, 1, own), turn)
        s2 = s2._replace(stages=s2.stages.at[1, Stat.ATK].set(stage))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E1 prep batch P4 (Gardenia): Helping Hand, SolarBeam ------------------------------------


def test_basic_helping_hand(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckHelpingHand (script.s:892-896): a singles battle, -10."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Helping Hand", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == -10


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        # Expert_ChargeTurnNoInvuln (script.s:3933-3972), SolarBeam (Grass) against Gible
        ("plain", 0),
        ("sun", 2),
        ("power herb", 2),
        ("low hp", -1),  # 38 % HP or less
        ("resisted", -2),  # against a Grass player: half damage
    ],
)
def test_expert_solar_beam(kaizo_tables, kaizo_names, kaizo_roark_party, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "SolarBeam", "Accelerock", "", ""))
    if case == "sun":
        s = s._replace(weather=jnp.asarray(Weather.SUN, s.weather.dtype))
    elif case == "power herb":
        s = s._replace(item=s.item.at[1, 0].set(c.ITEM_POWER_HERB))
    elif case == "low hp":
        s = _hp(s, 1, 30)
    elif case == "resisted":
        s = _with_player(t, s, n.species_names.index("Turtwig"), 0, known=True)
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_p4_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags, states):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "SolarBeam", "Helping Hand", "Accelerock", ""))
    for own, weather in ((100, Weather.SUN), (30, Weather.NONE))[:states]:
        s2 = _hp(s, 1, own)._replace(weather=jnp.asarray(weather, s.weather.dtype))
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E2 batch 10: the charge-turn moves (script.s:3933-4072) -----------------------------

POWER_HERB = c.ITEM_POWER_HERB


def _charge_state(t, n, party, move: str, player: str = "Gible", herb: bool = False):
    """Bonsly (Rock) with ``move`` and Accelerock, holding nothing or (``herb``) a Power Herb,
    slower than the player and at full HP, against ``player``."""
    s = _state(t, party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if player != "Gible":
        s = _with_player(t, s, n.species(player), 0, known=True)
    s = s._replace(item=s.item.at[1, 0].set(POWER_HERB if herb else 0))
    return _speeds(s, player=500, opp=1)


def _seen_protect(t, n, s):
    """``s`` with the player's active knowing Protect, seen by the AI."""
    protect = n.move_names.index("Protect")
    assert int(t.move_ai_effect[protect]) == BattleEffect.PROTECT
    return s._replace(
        moves=s.moves.at[0, 0, 3].set(protect), ai_seen_moves=s.ai_seen_moves.at[0, 0].set(protect)
    )


def test_charge_turn_kaizo_moves(kaizo_tables, kaizo_names):
    """The Kaizo moves the tests below use carry the decomp effects (Tables.move_ai_effect),
    and Power Herb is item 271 (generated/items.txt:272)."""
    t, n = kaizo_tables, kaizo_names
    for move, effect in (
        ("Razor Wind", BattleEffect.CHARGE_TURN_HIGH_CRIT),
        ("SolarBeam", BattleEffect.SKIP_CHARGE_TURN_IN_SUN),
        ("Fly", BattleEffect.FLY),
        ("Shadow Force", BattleEffect.SHADOW_FORCE),
        ("Lock-On", BattleEffect.NEXT_ATTACK_ALWAYS_HITS),
    ):
        assert int(t.move_ai_effect[n.move_names.index(move)]) == effect, move
    path = DECOMP / "generated/items.txt"
    if path.exists():
        assert path.read_text(encoding="utf-8").split()[POWER_HERB] == "ITEM_POWER_HERB"


@pytest.mark.parametrize(
    ("move", "player", "herb", "protect", "hp", "weather", "expected"),
    [
        # Expert_ChargeTurnNoInvuln (script.s:3933-3973), Razor Wind (:1657):
        ("Razor Wind", "Gible", False, False, 100, Weather.NONE, 0),
        ("Razor Wind", "Gible", True, False, 100, Weather.NONE, 2),  # Power Herb (:3956)
        ("Razor Wind", "Gible", False, True, 100, Weather.NONE, -2),  # Protect seen (:3964)
        ("Razor Wind", "Gible", False, False, 38, Weather.NONE, -1),  # 38 % HP or less
        ("Razor Wind", "Gible", False, False, 39, Weather.NONE, 0),  # above 38 %: nothing
        ("Razor Wind", "Gible", True, True, 30, Weather.NONE, 2),  # the herb ends it
        ("Razor Wind", "Geodude", True, False, 100, Weather.NONE, -2),  # resisted first
        ("Razor Wind", "Gible", False, False, 100, Weather.SUN, 0),  # sun: SolarBeam only
        # SolarBeam (:1716; the :1719 Expert_Thunder entry is never reached):
        ("SolarBeam", "Gible", False, False, 100, Weather.SUN, 2),  # :3949-3952
        ("SolarBeam", "Gible", False, True, 30, Weather.SUN, 2),  # sun ends it
        ("SolarBeam", "Gible", True, False, 100, Weather.NONE, 2),  # no sun: the herb
        ("SolarBeam", "Gible", False, True, 100, Weather.RAIN, -2),
        ("SolarBeam", "Charmander", False, False, 100, Weather.SUN, -2),  # resisted (Grass)
    ],
)
def test_charge_turn_no_invuln(
    kaizo_tables,
    kaizo_names,
    kaizo_roark_party,
    move,
    player,
    herb,
    protect,
    hp,
    weather,
    expected,
):
    """E2 batch 10: Expert_ChargeTurnNoInvuln, no random read."""
    t, n = kaizo_tables, kaizo_names
    s = _charge_state(t, n, kaizo_roark_party, move, player, herb)
    if protect:
        s = _seen_protect(t, n, s)
    s = _hp(s, 1, hp)._replace(weather=jnp.asarray(weather, s.weather.dtype))
    if player != "Gible":
        resisted = (TypeMulti.HALF, TypeMulti.QUARTER)
        assert _effectiveness(t, s, n.move_names.index(move)) in resisted
    assert _delta(t, s, 0x3, 0x7)[:2] == [expected, 0]


def test_charge_turn_basic(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Razor Wind's Basic_Main entry (script.s:158) is Basic_CheckNonStandardDamageOrChargeTurn
    (:572-585): a neutral hit against no Wonder Guard, nothing."""
    t, n = kaizo_tables, kaizo_names
    s = _charge_state(t, n, kaizo_roark_party, "Razor Wind")
    assert _delta(t, s, 0x2, 0x3)[0] == 0


@pytest.mark.parametrize(
    ("case", "jumps", "expected"),
    [
        # Expert_ChargeTurnWithInvuln (script.s:3992-4012), Fly (:1721):
        ("plain", (), 0),  # slower, nothing on the player: the end
        ("herb", (), 2),  # Expert_ChargeTurnNoInvuln_ScorePlus2 (:4009 -> :3959)
        ("protect", (), -1),  # :4010-4012
        ("herb protect", (), 2),
        ("resisted", (), 1),  # Expert_ShadowForce's +1 on a resisted move (:4017-4019, :4064)
        ("toxic", (), 1),  # :4028 -> TryScorePlus1, +1 unless :4058 jumps
        ("toxic", ((0, 4058),), 0),
        ("curse", (), 1),  # :4029
        ("seeded", (), 1),  # :4030
        ("sand", (), 1),  # a Rock attacker in sand (:4036-4040, the table :4068-4072)
        ("sand", ((0, 4058),), 0),
        ("hail", (), 0),  # no Ice type (:4043-4048): the speed compare, slower
        ("faster", (), 1),  # :4050-4055: not slower, last move not NEXT_ATTACK_ALWAYS_HITS
        ("faster", ((0, 4058),), 0),
        ("faster lock-on", (), 0),  # the player's last move Lock-On
    ],
)
def test_charge_turn_with_invuln(
    kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, expected
):
    """E2 batch 10: Expert_ChargeTurnWithInvuln, falling into Expert_ShadowForce."""
    t, n = kaizo_tables, kaizo_names
    player = "Geodude" if case == "resisted" else "Gible"
    s = _charge_state(t, n, kaizo_roark_party, "Fly", player, herb="herb" in case)
    if "protect" in case:
        s = _seen_protect(t, n, s)
    if case == "toxic":
        s = s._replace(status=s.status.at[0, 0].set(Status.TOXIC))
    if case == "curse":
        s = s._replace(curse=s.curse.at[0].set(1))
    if case == "seeded":
        s = s._replace(leech_seed=s.leech_seed.at[0].set(1))
    weather = {"sand": Weather.SANDSTORM, "hail": Weather.HAIL}.get(case, Weather.NONE)
    s = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
    if case.startswith("faster"):
        s = _speeds(s, player=1, opp=500)
    if case == "faster lock-on":
        s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index("Lock-On")))
    if case == "resisted":
        assert _effectiveness(t, s, n.move_names.index("Fly")) == TypeMulti.HALF
    assert _delta(t, s, 0x3, 0x7, jumps)[:2] == [expected, 0]


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        # Expert_ShadowForce (script.s:4014-4066), Shadow Force (:1808):
        ("plain", 0),
        ("herb", 1),  # Expert_ChargeTurnWithInvuln_ScorePlus1AndEnd (:4020 -> :4023)
        ("protect", 0),  # no Protect check (Shadow Force goes through it)
        ("immune", 1),  # :4017 -> Expert_ChargeTurnWithInvuln_ScorePlus1 (:4064)
        ("herb toxic", 1),  # the herb ends it before the roll
    ],
)
def test_shadow_force(kaizo_tables, kaizo_names, kaizo_roark_party, case, expected):
    """E2 batch 10: Expert_ShadowForce."""
    t, n = kaizo_tables, kaizo_names
    player = "Rattata" if case == "immune" else "Gible"
    s = _charge_state(t, n, kaizo_roark_party, "Shadow Force", player, herb="herb" in case)
    if case == "protect":
        s = _seen_protect(t, n, s)
    if "toxic" in case:
        s = s._replace(status=s.status.at[0, 0].set(Status.TOXIC))
    if case == "immune":
        assert _effectiveness(t, s, n.move_names.index("Shadow Force")) == TypeMulti.IMMUNE
    assert _delta(t, s, 0x3, 0x7)[:2] == [expected, 0]


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_charge_turn_exact_equals_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, flags, states
):
    """Exact == loose for the charge-turn moves, on paths through the :4058 roll."""
    t, n = kaizo_tables, kaizo_names
    s = _state(
        t, kaizo_roark_party, BONSLY, _ids(n, "Razor Wind", "SolarBeam", "Fly", "Shadow Force")
    )
    for weather, faster in ((Weather.SANDSTORM, False), (Weather.SUN, True))[:states]:
        s2 = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
        s2 = _speeds(s2, player=1, opp=500) if faster else _speeds(s2, player=500, opp=1)
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E2 batch 11 (B11-4): Encore -------------------------------------------------------------


@pytest.mark.fullset
def test_encore_encouraged_effects_are_the_decomps():
    """``commands.ENCORE_ENCOURAGED_EFFECTS`` is Expert_Encore_EncouragedMoveEffects
    (script.s:3202-3285) entry by entry, through generated/move_battle_effects.txt."""
    script, ids = (
        DECOMP / "src/battle/trainer_ai/script.s",
        DECOMP / "generated/move_battle_effects.txt",
    )
    if not script.exists() or not ids.exists():
        pytest.skip("decomp not present")
    names = ids.read_text(encoding="utf-8").split()
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[3201] == "Expert_Encore_EncouragedMoveEffects:"
    entries = []
    for line in lines[3202:]:
        name = line.split()[1]
        if name == "TABLE_END":
            break
        entries.append(names.index(name))
    assert tuple(entries) == c.ENCORE_ENCOURAGED_EFFECTS


@pytest.mark.parametrize(("turns", "basic"), [(0, 0), (3, -8)])
def test_basic_cannot_encore(kaizo_tables, kaizo_names, kaizo_roark_party, turns, basic):
    """Basic_CheckCannotEncore (script.s:655-658): IfBattlerUnderEffect CHECK_ENCORE reads the
    player's encoredTurns (trainer_ai.c:1839-1843), -8."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Encore", "Accelerock", "", ""))
    s = s._replace(encore_turns=s.encore_turns.at[0].set(turns))
    assert _delta(t, s, 0x2, 0x3)[0] == basic


@pytest.mark.parametrize(
    ("case", "jumps", "expected"),
    [
        # Expert_Encore (script.s:3177-3200)
        ("disabled", (), 3),  # :3185 -> TryScorePlus3: +3 unless :3192 jumps
        ("disabled", (3192,), 0),
        ("slower", (), -2),  # :3186
        ("plain hit", (), -2),  # the player's last move a HIT: not in the table (:3189)
        ("swords dance", (), 3),  # ATK_UP_2 is in the table
        ("swords dance", (3192,), 0),
    ],
)
def test_expert_encore(kaizo_tables, kaizo_names, kaizo_roark_party, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Encore", "Accelerock", "", ""))
    s = _speeds(s, player=500, opp=1) if case == "slower" else _speeds(s, player=1, opp=500)
    last = "Swords Dance" if case == "swords dance" else "Tackle"
    s = s._replace(last_move=s.last_move.at[0].set(n.move_names.index(last)))
    if case == "disabled":
        # Disabled goes to the roll before the speed test: slower here.
        s = _speeds(s, player=500, opp=1)._replace(disable_turns=s.disable_turns.at[0].set(2))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.parametrize(("flags", "states"), SWEEP_ONE)
def test_batch11_encore_exact_equals_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, flags, states
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Encore", "Accelerock", "", ""))
    for disabled, faster in ((0, True), (2, False), (0, False))[:states]:
        s2 = s._replace(disable_turns=s.disable_turns.at[0].set(disabled))
        s2 = _speeds(s2, player=1, opp=500) if faster else _speeds(s2, player=500, opp=1)
        layout = layout_for(t, flags, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E2 batch 12 (B12-11): the stat family's AI ---------------------------------------------


def _stage(s, side: int, stat, value: int):
    """``s`` with ``side``'s ``stat`` stage at engine ``value`` (decomp ``value + 6``)."""
    return s._replace(stages=s.stages.at[side, stat].set(value))


@pytest.mark.parametrize(
    ("move", "case", "expected"),
    [
        # Basic_CheckStatStageImbalance (script.s:475-497): -10 with nothing to undo or copy.
        ("Haze", "neutral", -10),
        ("Haze", "own drop", 0),
        ("Psych Up", "player boost", 0),
        ("Psych Up", "neutral", -10),
        # Basic_CheckAlreadyPumpedUp (:592-595).
        ("Focus Energy", "pumped", -10),
        ("Focus Energy", "neutral", 0),
        # Basic_CheckAlreadyUnderMist (:587-590): the AI's side under Mist -8 (B12-23).
        ("Mist", "misted", -8),
        ("Mist", "player misted", 0),
        ("Mist", "neutral", 0),
        # Basic_CheckTickle (:933-947): Attack at -6 -10, Defense at -6 -8.
        ("Tickle", "player atk -6", -10),
        ("Tickle", "player def -6", -8),
        # Basic_CheckCosmicPower (:949-963): Defense +6 -10, Sp. Def +6 -8.
        ("Cosmic Power", "own def +6", -10),
        ("Cosmic Power", "own spd +6", -8),
        # Basic_CheckAcupressure (:1135-1147): any stage at +6 -10.
        ("Acupressure", "own eva +6", -10),
        ("Acupressure", "neutral", 0),
        # Basic_CheckPowerTrick (:1358-1361).
        ("Power Trick", "tricked", -10),
        ("Power Trick", "neutral", 0),
        # Basic_CheckPowerSwap (:1392-1404): -10 unless the player is ahead on Attack or Sp. Atk.
        ("Power Swap", "neutral", -10),
        ("Power Swap", "player boost", 0),
        ("Guard Swap", "player def ahead", 0),
        ("Guard Swap", "neutral", -10),
        # Basic_CheckCaptivate (:1508-1538): same gender -10.
        ("Captivate", "same gender", -10),
        ("Captivate", "opposite", 0),
    ],
)
def test_batch12_basic(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if case == "own drop":
        s = _stage(s, 1, Stat.SPE, -1)
    elif case == "player boost":
        s = _stage(s, 0, Stat.ATK, 2)
    elif case == "pumped":
        s = s._replace(focus_energy=s.focus_energy.at[1].set(1))
    elif case in ("misted", "player misted"):
        s = s._replace(mist=s.mist.at[1 if case == "misted" else 0].set(2))
    elif case == "player atk -6":
        s = _stage(s, 0, Stat.ATK, -6)
    elif case == "player def -6":
        s = _stage(s, 0, Stat.DEF, -6)
    elif case == "own def +6":
        s = _stage(s, 1, Stat.DEF, 6)
    elif case == "own spd +6":
        s = _stage(s, 1, Stat.SPD, 6)
    elif case == "own eva +6":
        s = _stage(s, 1, Stat.EVA, 6)
    elif case == "tricked":
        s = s._replace(power_trick=s.power_trick.at[1].set(1))
    elif case == "player def ahead":
        s = _stage(s, 0, Stat.DEF, 1)
    elif case in ("same gender", "opposite"):
        mine = 0 if case == "same gender" else 1
        s = s._replace(gender=s.gender.at[0, 0].set(0).at[1, 0].set(mine))
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("move", "case", "jumps", "expected"),
    [
        # Expert_Haze (script.s:2520-2564)
        ("Haze", "neutral", (), -1),  # neither list: -1 unless :2555 jumps
        ("Haze", "neutral", (2555,), 0),
        ("Haze", "own +3", (), -3 - 1),  # :2541 does not jump: -3; then -1
        ("Haze", "player +3", (), 3),  # :2560: +3 unless it jumps
        ("Haze", "player +3", (2560,), 0),
        # Expert_PsychUp (:3814-3850)
        ("Psych Up", "neutral", (), -2),
        ("Psych Up", "player +3", (), 1),  # the AI's Attack below 7 (neutral)
        # Expert_Flatter (:2854-2859), then Expert_StatusConfuse with the player above 70 %.
        ("Flatter", "neutral", (), 1),
        ("Flatter", "neutral", (2858,), 0),
        # Expert_Acupressure (:5008-5027): full HP: +1 unless :5019 jumps.
        ("Acupressure", "neutral", (), 1),
        ("Acupressure", "neutral", (5019,), 0),
        ("Acupressure", "low hp", (), -1),
        # Expert_PowerTrick (:5442-5469)
        ("Power Trick", "neutral", (), 1),
        ("Power Trick", "neutral", (5454,), 0),
        ("Power Trick", "low hp", (), -2),
        # Expert_PowerSwap (:5632-5764): Attack diff 4 (HighDiff), Sp. Atk diff 0: TryScorePlus3.
        ("Power Swap", "player atk +4", (), 3),
        ("Power Swap", "player atk +4", (5749,), 2),
        ("Power Swap", "player atk +4", (5749, 5754, 5759), 0),
        ("Power Swap", "neutral", (), 0),  # diff 0 / 0: NoDiff, no entry: nothing
        # Expert_GuardSwap (:5766-5898): Defense diff 1 (LowDiff), Sp. Def diff 0: Plus1.
        ("Guard Swap", "player def +1", (), 1),
        ("Guard Swap", "player def +1", (5893,), 0),
        # Expert_Captivate (:6236-6266): the player's Sp. Atk at neutral and its HP full; its
        # last move none, whose class is physical (LoadDefenderLastUsedMoveClass): -1 unless
        # :6262 jumps.
        ("Captivate", "neutral", (), -1),
        ("Captivate", "neutral", (6262,), 0),
        # Sp. Atk at -4 (decomp 2): -1 (the AI's HP full skips the second), -2 unless :6252
        # jumps, then the last move's -1.
        ("Captivate", "player spa -4", (), -1 - 2 - 1),
        ("Captivate", "player spa -4", (6252, 6262), -1),
    ],
)
def test_batch12_expert(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = s._replace(gender=s.gender.at[0, 0].set(0).at[1, 0].set(1))
    if case == "own +3":
        s = _stage(s, 1, Stat.SPA, 3)
    elif case == "player +3":
        s = _stage(s, 0, Stat.ATK, 3)
    elif case == "low hp":
        s = _hp(s, 1, 30)
    elif case == "player atk +4":
        s = _stage(s, 0, Stat.ATK, 4)
    elif case == "player def +1":
        s = _stage(s, 0, Stat.DEF, 1)
    elif case == "player spa -4":
        s = _stage(s, 0, Stat.SPA, -4)
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.slow  # 242 s, 5.7 GB peak at 0x11F on the merged tip (fast tier: 60 s, 4 GB)
@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch12_stat_family_exact_equals_loose(
    kaizo_tables, kaizo_names, kaizo_roark_party, flags
):
    t, n = kaizo_tables, kaizo_names
    for moves in (
        ("Haze", "Psych Up", "Flatter", "Tickle"),
        ("Acupressure", "Power Trick", "Power Swap", "Guard Swap"),
        ("Captivate", "Focus Energy", "Cosmic Power", "Calm Mind"),
    ):
        s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
        for own, stage in ((100, 3), (40, -2)):
            s2 = _stage(_stage(_hp(s, 1, own), 0, Stat.ATK, stage), 1, Stat.DEF, -stage)
            layout = layout_for(t, flags, s2)
            assert policy.ai_refusals(t, s2, n, layout) == []
            _assert_exact_equals_loose(t, s2, layout)


@pytest.mark.parametrize(
    ("attacker", "target", "expected"),
    [(100, 100, 0), (70, 100, -2), (100, 30, -2), (70, 30, -4)],
)
def test_mist_check_hp(kaizo_tables, kaizo_names, kaizo_roark_party, attacker, target, expected):
    """Mist (PREVENT_STAT_REDUCTION, the Kaizo ROM's effect for the move, ``move_ai_effect``)
    is in CheckHP_DiscourageAtMediumHP (script.s:7782) and CheckHP_Target_DiscourageAtLowHP
    (:7938), not the high-HP ones: -2 per band met unless :7718 / :7742 jumps (B12-23)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Mist", "Accelerock", "", ""))
    assert int(t.move_ai_effect[n.move_names.index("Mist")]) == BattleEffect.PREVENT_STAT_REDUCTION
    s = _hp(_hp(s, 1, attacker), 0, target)
    assert _delta(t, s, 0x7, 0x107)[0] == expected
    assert _delta(t, s, 0x7, 0x107, ((0, 7718), (0, 7742)))[0] == 0


@pytest.mark.parametrize(
    "states", [pytest.param(1, id="one"), pytest.param(None, id="all", marks=pytest.mark.slow)]
)
def test_mist_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, states):
    """Under Roark's flag set (0x17), misted or not: scored, not refused, exact == loose."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Mist", "Accelerock", "", ""))
    for s2 in (s, s._replace(mist=s.mist.at[1].set(2)))[:states]:
        layout = layout_for(t, 0x17, s2)
        assert policy.ai_refusals(t, s2, n, layout) == []
        _assert_exact_equals_loose(t, s2, layout)


# --- E2 batch 12 (B12-18): the healing family's AI ------------------------------------------


@pytest.mark.parametrize(
    ("move", "case", "expected"),
    [
        # Basic_CheckCanRefreshStatus (script.s:923-926)
        ("Refresh", "neutral", -10),
        ("Refresh", "burned", 0),
        ("Refresh", "asleep", -10),
        # Basic_CheckCanPsychoShift (:1301-1345)
        ("Psycho Shift", "neutral", -10),
        ("Psycho Shift", "burned", 0),
        ("Psycho Shift", "burned, player statused", -10),
        ("Psycho Shift", "poisoned", -10),  # poison onto a Steel player (:1320-1321)
        # Basic_CheckCanRecoverHP (:210 -> :538)
        ("Morning Sun", "neutral", -8),
        ("Morning Sun", "hurt", 0),
    ],
)
def test_batch12_healing_basic(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if "burned" in case:
        s = s._replace(status=s.status.at[1, 0].set(Status.BURN))
    if case == "asleep":
        s = s._replace(status=s.status.at[1, 0].set(Status.SLEEP))
    if "player statused" in case:
        s = s._replace(status=s.status.at[0, 0].set(Status.PARALYSIS))
    if case == "poisoned":
        s = s._replace(status=s.status.at[1, 0].set(Status.POISON))
        s = _with_player(t, s, n.species_names.index("Magnemite"), 0, known=True)  # Steel
    if case == "hurt":
        s = _hp(s, 1, 40)
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("move", "case", "jumps", "expected"),
    [
        # Expert_Rest (script.s:2745-2790): faster at full HP -8.
        ("Rest", "faster full", (), -8),
        # Faster at 30 %: CheckForSnatch: the player shows no Snatch: +3 unless :2786 jumps.
        ("Rest", "faster low", (), 3),
        ("Rest", "faster low", (2786,), 0),
        # Faster at 60 % (above 50): -3.
        ("Rest", "faster 60", (), -3),
        # Faster at 45 %: -3 unless :2766 jumps (then +3).
        ("Rest", "faster 45", (), -3),
        ("Rest", "faster 45", (2766,), 3),
        # Slower at 65 % (between 60 and 70): the :2775 roll.
        ("Rest", "slower 65", (), -3),
        ("Rest", "slower 65", (2775,), 3),
        # Expert_HealBell (:3398-3406): no status anywhere -5; a statused party member: 0.
        ("Heal Bell", "neutral", (), -5),
        ("Heal Bell", "bench statused", (), 0),
        # Expert_Synthesis (:2628-2638): rain -2, then Expert_Recovery (full HP: -3).
        ("Synthesis", "rain", (), -2 - 3),
        ("Synthesis", "neutral", (), -3),
        # Expert_Refresh (:4644-4653): the player below 50 % -1.
        ("Refresh", "player low", (), -1),
        ("Refresh", "neutral", (), 0),
        # Expert_PsychoShift (:5296-5306): statused AI: +1 unless :5301 jumps.
        ("Psycho Shift", "burned", (), 1),
        ("Psycho Shift", "burned", (5301,), 0),
        ("Psycho Shift", "neutral", (), -10),
    ],
)
def test_batch12_healing_expert(
    kaizo_tables, kaizo_names, kaizo_roark_party, move, case, jumps, expected
):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _speeds(s, player=500, opp=1) if "slower" in case else _speeds(s, player=1, opp=500)
    hp = {"faster low": 30, "faster 60": 60, "faster 45": 45, "slower 65": 65}.get(case)
    if hp is not None:
        s = _hp(s, 1, hp)
    if case == "bench statused":
        s = s._replace(status=s.status.at[1, 1].set(Status.BURN))
    if case == "rain":
        s = s._replace(weather=jnp.asarray(Weather.RAIN, s.weather.dtype))
    if case == "player low":
        s = _hp(s, 0, 40)
    if case == "burned":
        s = s._replace(status=s.status.at[1, 0].set(Status.BURN))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.slow  # 146 s, 5.1 GB peak at 0x11F on the merged tip (fast tier: 60 s, 4 GB)
@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch12_healing_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    t, n = kaizo_tables, kaizo_names
    for moves in (("Rest", "Heal Bell", "Synthesis", "Wish"), ("Refresh", "Psycho Shift", "", "")):
        s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
        for own, status, faster in ((100, Status.NONE, True), (45, Status.BURN, False)):
            s2 = _hp(s, 1, own)._replace(status=s.status.at[1, 0].set(status))
            s2 = _speeds(s2, player=1, opp=500) if faster else _speeds(s2, player=500, opp=1)
            layout = layout_for(t, flags, s2)
            assert policy.ai_refusals(t, s2, n, layout) == []
            _assert_exact_equals_loose(t, s2, layout)


# --- E1 trial batch: Perish Song (G-64) ------------------------------------------------------


@pytest.mark.parametrize(("perishing", "basic"), [(False, 0), (True, -10)])
def test_perish_song(kaizo_tables, kaizo_names, kaizo_roark_party, perishing, basic):
    """Basic_CheckPerishSong (script.s:720-723): -10 when the player already has the effect;
    no Expert entry for ALL_FAINT_3_TURNS (:1628-1808)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Perish Song", "Accelerock", "", ""))
    s = s._replace(perish_song=s.perish_song.at[0].set(int(perishing)))
    assert _delta(t, s, 0x2, 0x3)[0] == basic
    assert _delta(t, s, 0x3, 0x7)[0] == 0


# --- E1 trial batch: Nature Power -------------------------------------------------------------


@pytest.mark.parametrize(("jumps", "harass"), [((), 2), ((8030,), 0)])
def test_nature_power(kaizo_tables, kaizo_names, kaizo_roark_party, jumps, harass):
    """NATURE_POWER (173) has no Basic_ScoreMoveEffect or Expert_Main entry: the AI scores the
    slot's own status move (power 0). Its only mention is Harrassment_Effects (script.s:8058):
    +2 unless the :8030 roll jumps."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Nature Power", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    assert _delta(t, s, 0x3, 0x7)[0] == 0
    assert _delta(t, s, 0x5, 0x405, tuple((0, line) for line in jumps))[0] == harass


# --- E1 trial batch: Endure (G-65) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("own_hp", "jumps", "expert"),
    [
        # Expert_Endure (script.s:3633-3649): below 4 % -1; below 35 % +1 unless the :3645 roll
        # jumps; 35 % or more falls through into Expert_Endure_ScoreMinus1 (:3640), -1.
        (100, (), -1),
        (35, (), -1),
        (34, (), 1),
        (34, (3645,), 0),
        (20, (), 1),
        (20, (3645,), 0),
        (4, (), 1),
        (3, (), -1),
        (1, (3645,), -1),
    ],
)
def test_endure(kaizo_tables, kaizo_names, kaizo_roark_party, own_hp, jumps, expert):
    """SURVIVE_WITH_1_HP (116): no Basic entry (:134-285), Expert_Main :1704."""
    t, n = kaizo_tables, kaizo_names
    s = _hp(
        _state(t, kaizo_roark_party, BONSLY, _ids(n, "Endure", "Accelerock", "", "")), 1, own_hp
    )
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


# --- E1 trial batch: Magnet Rise (G-66) -------------------------------------------------------


def _levitate_species(t):
    """A species whose ability slot 0 is Levitate (the AI reads the species')."""
    abilities = np.asarray(t.species_abilities_decomp)
    rows = np.asarray(t.species_form_row)[:, 0]
    return next(
        sp for sp in range(1, rows.shape[0]) if abilities[rows[sp], 0] == c.ABILITY_IDS["Levitate"]
    )


@pytest.mark.parametrize("case", ["plain", "bit", "levitate", "flying"])
def test_magnet_rise_basic(kaizo_tables, kaizo_names, kaizo_roark_party, case):
    """Basic_CheckMagnetRise (script.s:1461-1474): the attacker with the bit (:1463), Levitate
    (:1467) or a Flying type (:1470-1473) scores -10."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Magnet Rise", "Accelerock", "", ""))
    if case == "bit":
        s = s._replace(magnet_rise=s.magnet_rise.at[1].set(1))
    if case == "levitate":
        s = as_species(t, s, 1, 0, _levitate_species(t))
    if case == "flying":
        s = s._replace(types=s.types.at[1, 0, 1].set(2))  # TYPE_FLYING
    assert _delta(t, s, 0x2, 0x3)[0] == (0 if case == "plain" else -10)


@pytest.mark.parametrize(
    ("own_hp", "ground", "known", "jumps", "expert"),
    [
        # Expert_MagnetRise (script.s:6090-6119): below 50 % HP ends; a known Earthquake / Earth
        # Power / Fissure +1; a Ground-type player +1, else +1 unless the :6113 roll jumps.
        (40, True, True, (), 0),
        (49, False, True, (), 0),
        (100, True, True, (), 2),
        (100, True, False, (), 1),
        (100, False, True, (), 2),
        (100, False, True, (6113,), 1),
        (100, False, False, (), 1),
        (100, False, False, (6113,), 0),
        (50, False, False, (), 1),
    ],
)
def test_magnet_rise_expert(
    kaizo_tables, kaizo_names, kaizo_roark_party, own_hp, ground, known, jumps, expert
):
    t, n = kaizo_tables, kaizo_names
    s = _hp(
        _state(t, kaizo_roark_party, BONSLY, _ids(n, "Magnet Rise", "Accelerock", "", "")),
        1,
        own_hp,
    )
    if not ground:  # Roark's Gible is Dragon / Ground
        s = s._replace(types=s.types.at[0, 0].set(jnp.zeros(2, s.types.dtype)))  # Normal
    if known:
        s = s._replace(
            moves=s.moves.at[0, 0, 3].set(n.move("Earthquake")),
            ai_seen_moves=s.ai_seen_moves.at[0, 0].set(n.move("Earthquake")),
        )
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


def test_the_players_magnet_rise_is_immune_to_the_ai_ground_move(
    kaizo_tables, kaizo_names, kaizo_roark_party
):
    """MOVE_STATUS_MAGNET_RISE (battle_lib.c:2605-2609, moves.h:87): IfMoveEffectivenessEquals
    reads IMMUNE, the damage estimate is 0, and the walk is skipped (only STAB stays)."""
    from nuzline.ai.switching import MAGNET_RISE, apply_type_chart, type_chart_flags

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Earthquake", "Accelerock", "", ""))
    s = s._replace(types=s.types.at[0, 0].set(jnp.asarray([10, 10], s.types.dtype)))  # Fire
    quake, rock = n.move("Earthquake"), n.move("Accelerock")
    assert int(c.effectiveness_class(t, s, quake)) == TypeMulti.DOUBLE  # Ground vs Fire
    floating = s._replace(magnet_rise_turns=s.magnet_rise_turns.at[0].set(3))
    assert int(c.effectiveness_class(t, floating, quake)) == TypeMulti.IMMUNE
    assert int(c.effectiveness_class(t, floating, rock)) == int(c.effectiveness_class(t, s, rock))
    # The value keeps STAB only (Bonsly is Rock, Earthquake Ground: no STAB): 40 stays 40.
    assert int(apply_type_chart(t, floating, 1, quake, 40)) == 40
    assert int(apply_type_chart(t, s, 1, quake, 40)) == 80
    flags = int(
        type_chart_flags(t, quake, floating.types[0, 0], 0, False, None, False, False, True)
    )
    assert flags == MAGNET_RISE
    # Ingrain or an Iron Ball (the player's) lifts it for the AI as for the battle.
    for lifted in (
        floating._replace(ingrain=floating.ingrain.at[0].set(1)),
        floating._replace(item=floating.item.at[0, 0].set(n.item("Iron Ball"))),
    ):
        assert int(c.effectiveness_class(t, lifted, quake)) == TypeMulti.DOUBLE
    # The damage estimate (TrainerAI_CalcDamage, trainer_ai.c:3106) is zero.
    from nuzline.engine.damage import ai_damage_for_roll

    assert int(ai_damage_for_roll(t, floating, 1, quake, 100)) == 0
    assert int(ai_damage_for_roll(t, s, 1, quake, 100)) > 0


# --- E1 trial batch: Gravity (G-67) -------------------------------------------------------------


@pytest.mark.parametrize("up", [False, True])
def test_gravity_basic(kaizo_tables, kaizo_names, kaizo_roark_party, up):
    """Basic_CheckGravityActive (script.s:1026-1029): Gravity already up, -10."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Gravity", "Accelerock", "", ""))
    s = s._replace(gravity=jnp.asarray(3 if up else 0, s.gravity.dtype))
    assert _delta(t, s, 0x2, 0x3)[0] == (-10 if up else 0)


@pytest.mark.parametrize(
    ("case", "own_hp", "jumps", "expert"),
    [
        # Expert_Gravity (script.s:4782-4803): a Levitate / Magnet Rise / Flying player goes to
        # Expert_Gravity_TryScorePlus1 whatever the HP (+1 unless the :4799 roll jumps); else
        # below 60 % HP ends, and the :4795 roll jumps to TryScorePlus1 or the script ends.
        ("flying", 100, (), 1),
        ("flying", 100, (4799,), 0),
        ("flying", 30, (), 1),
        ("magnet", 100, (), 1),
        ("magnet", 100, (4799,), 0),
        ("levitate", 100, (), 1),
        ("levitate", 100, (4799,), 0),
        ("ground", 50, (), 0),
        ("ground", 50, (4795,), 0),
        ("ground", 80, (), 0),
        ("ground", 80, (4795,), 1),
        ("ground", 80, (4795, 4799), 0),
    ],
)
def test_gravity_expert(kaizo_tables, kaizo_names, kaizo_roark_party, case, own_hp, jumps, expert):
    t, n = kaizo_tables, kaizo_names
    s = _hp(
        _state(t, kaizo_roark_party, BONSLY, _ids(n, "Gravity", "Accelerock", "", "")), 1, own_hp
    )
    if case == "flying":
        s = s._replace(types=s.types.at[0, 0].set(jnp.asarray([2, 2], s.types.dtype)))
    if case == "magnet":
        s = s._replace(magnet_rise=s.magnet_rise.at[0].set(1))
    if case == "levitate":
        lev = c.ABILITY_IDS["Levitate"]
        t = _with_abilities(t, int(s.species[0, 0]), [lev, lev])
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expert


# --- E2 batch 13 (B13-23..27): the variable-power and secondary hits, Recycle --------------


def _item(s, side: int, item: int, slot: int = 0):
    return s._replace(item=s.item.at[side, slot].set(item))


@pytest.mark.parametrize(
    ("move", "case", "expected"),
    [
        # Basic_CheckCanRecycle (script.s:911-915): nothing to recycle -10.
        ("Recycle", "neutral", -10),
        ("Recycle", "recycle Lum", 0),
        # Basic_CheckNaturalGift (:1053-1059): not a berry -10; a berry the player is not
        # immune to (Cheri, Fire, against Gible), nothing.
        ("Natural Gift", "neutral", -10),
        ("Natural Gift", "hold Leftovers", -10),
        ("Natural Gift", "hold Cheri Berry", 0),
        # Basic_CheckNonStandardDamageOrChargeTurn (:572-585) against a non-Wonder Guard player.
        ("Trump Card", "neutral", 0),
        ("Punishment", "neutral", 0),
        # Basic_CheckEmbargo (:1184-1197, B13-32): the player under Embargo -10; else its
        # battler's recycle item outside the Frontier, nothing.
        ("Embargo", "neutral", 0),
        ("Embargo", "player embargoed", -10),
        ("Embargo", "player recycle Lum", 0),
    ],
)
def test_batch13_basic(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    if case == "recycle Lum":
        s = s._replace(recycle_item=s.recycle_item.at[1].set(n.item_names.index("Lum Berry")))
    elif case == "player recycle Lum":
        s = s._replace(recycle_item=s.recycle_item.at[0].set(n.item_names.index("Lum Berry")))
    elif case == "player embargoed":
        s = s._replace(embargo=s.embargo.at[0].set(3))
    elif case.startswith("hold "):
        s = _item(s, 1, n.item_names.index(case[5:]))
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("move", "case", "jumps", "expected"),
    [
        # Expert_HighCritical (script.s:2827-2843) by Blaze Kick (:1748), Cross Poison (:1754):
        # resisted (Fire, Poison against Gible), nothing; neutral (Snorlax) +1 unless a roll
        # (:2836, :2839) jumps.
        ("Blaze Kick", "faster", (), 0),
        ("Cross Poison", "snorlax", (), 1),
        ("Cross Poison", "snorlax", (2836,), 0),
        ("Blaze Kick", "snorlax", (2839,), 0),
        # Expert_WaterSpout (:4610-4632): Water against Gible is neutral.
        ("Water Spout", "faster", (), 0),
        ("Water Spout", "faster, player 40", (), -1),
        ("Water Spout", "slower, player 60", (), -1),
        ("Water Spout", "slower, player 80", (), 0),
        # Expert_WakeUpSlap (:4829-4847)
        ("Wake-Up Slap", "faster", (), 0),
        ("Wake-Up Slap", "player asleep", (), 1),
        # Expert_GyroBall (:4869-4871): no score change.
        ("Gyro Ball", "faster", (), 0),
        # Expert_Assurance (:5189-5216): faster, nothing; slower, :5204 then :5212.
        ("Assurance", "faster", (), 0),
        ("Assurance", "slower", (), 0),
        ("Assurance", "slower", (5204,), 1),
        ("Assurance", "slower", (5204, 5212), 0),
        # Expert_TrumpCard (:5308-5359) by the move's PP left.
        ("Trump Card", "pp 1", (), 3),
        ("Trump Card", "pp 2", (), 2),
        ("Trump Card", "pp 2", (5347,), 1),
        ("Trump Card", "pp 3", (), 1),
        ("Trump Card", "pp 3", (5347,), 0),
        ("Trump Card", "faster", (), 0),
        ("Trump Card", "player eva +3", (), 1),
        ("Trump Card", "player eva +5", (), 2),
        ("Trump Card", "own acc -3", (), 1),
        # Expert_Punishment (:5900-5951): each entered label adds unless its roll jumps and
        # falls into the next one.
        ("Punishment", "faster", (), 0),
        ("Punishment", "player +3", (), 1),
        ("Punishment", "player +3", (5947,), 0),
        ("Punishment", "player +5", (), 2 + 1),
        ("Punishment", "player +7", (), 4 + 3 + 2 + 1),
        ("Punishment", "player +7", (5935, 5943), 3 + 1),
        # Expert_Recycle (:4517-4534)
        ("Recycle", "faster", (), -2),
        ("Recycle", "recycle Oran", (), -2),
        ("Recycle", "recycle Lum", (), 1),
        ("Recycle", "recycle Lum", (4526,), 0),
        # Expert_Embargo (:5223-5229, B13-32): +1 unless the 128 roll (:5225) jumps.
        ("Embargo", "faster", (), 1),
        ("Embargo", "faster", (5225,), 0),
    ],
)
def test_batch13_expert(kaizo_tables, kaizo_names, kaizo_roark_party, move, case, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    s = _speeds(s, player=500, opp=1) if "slower" in case else _speeds(s, player=1, opp=500)
    if case == "snorlax":
        s = _with_player(t, s, n.species_names.index("Snorlax"), 0, known=True)
    for hp in (40, 60, 80):
        if f"player {hp}" in case:
            s = _hp(s, 0, hp)
    if case == "player asleep":
        s = s._replace(status=s.status.at[0, 0].set(Status.SLEEP))
    if case.startswith("pp "):
        s = s._replace(pp=s.pp.at[1, 0, 0].set(int(case[3:])))
    elif move == "Trump Card":  # Kaizo's Trump Card has 1 PP: a synthetic 4 left
        s = s._replace(pp=s.pp.at[1, 0, 0].set(4))
    if case == "player eva +3":
        s = _stage(s, 0, Stat.EVA, 3)
    elif case == "player eva +5":
        s = _stage(s, 0, Stat.EVA, 5)
    elif case == "own acc -3":
        s = _stage(s, 1, Stat.ACC, -3)
    elif case == "player +3":
        s = _stage(s, 0, Stat.ATK, 3)
    elif case == "player +5":
        s = _stage(_stage(s, 0, Stat.ATK, 3), 0, Stat.SPE, 2)
    elif case == "player +7":
        s = _stage(_stage(s, 0, Stat.ATK, 6), 0, Stat.DEF, 1)
    elif case.startswith("recycle "):
        item = n.item_names.index(case[8:] + " Berry")
        s = s._replace(recycle_item=s.recycle_item.at[1].set(item))
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


@pytest.mark.fullset
def test_batch13_refusals_lifted(kaizo_tables, kaizo_names, kaizo_roark_party):
    """B13-23..27: the batch's effects are transcribed (no "not transcribed" or "expert
    label"); Natural Gift and Judgment take modelled id cases; Gyro Ball's TrainerAI_CalcDamage
    case stays refused (monSpeedValues), and Weather Ball's TrainerAI_MoveType case is refused
    exactly where it assigns no type (trainer_ai.c:3222-3240)."""
    from nuzline.ai.layout import ALLIN

    t, n = kaizo_tables, kaizo_names
    for name in (
        "Twineedle", "Triple Kick", "Rock Wrecker", "Hurricane", "Beat Up", "Recycle",
        "Water Spout", "Blaze Kick", "Weather Ball", "Cross Poison", "Wake-Up Slap",
        "Natural Gift", "Assurance", "Trump Card", "Punishment", "Judgment", "Embargo",
    ):  # fmt: skip
        kinds = policy.opponent_move_refusals(t, n.move_names.index(name), ALLIN)
        assert kinds == (["move type"] if name == "Weather Ball" else []), (name, kinds)
    gyro = n.move_names.index("Gyro Ball")
    assert policy.opponent_move_refusals(t, gyro, ALLIN) == ["calc damage"]
    moves = _ids(n, "Weather Ball", "Natural Gift", "Judgment", "")
    s = _state(t, kaizo_roark_party, BONSLY, moves)
    r = policy._refused(t, s, KAIZO)
    assert bool(r.move_type[0, 0]) and not np.asarray(r.move_type)[0, 1:].any()
    assert not np.asarray(r.calc_damage).any()
    for weather, refused in (
        (Weather.RAIN, False),
        (Weather.SUN, False),
        (Weather.SANDSTORM, False),
        (Weather.HAIL, False),
        (Weather.FOG, True),
    ):
        s2 = s._replace(weather=jnp.asarray(weather, s.weather.dtype))
        assert bool(policy._refused(t, s2, KAIZO).move_type[0, 0]) == refused, weather


@pytest.mark.slow  # 307 s, 5.0 GB peak at 0x11F (fast tier: 60 s, 4 GB)
@pytest.mark.parametrize("flags", SWEEP_FLAGS)
def test_batch13_exact_equals_loose(kaizo_tables, kaizo_names, kaizo_roark_party, flags):
    t, n = kaizo_tables, kaizo_names
    lum = n.item_names.index("Lum Berry")
    for moves in (
        ("Recycle", "Water Spout", "Wake-Up Slap", "Assurance"),
        ("Trump Card", "Punishment", "Cross Poison", "Blaze Kick"),
        ("Twineedle", "Triple Kick", "Beat Up", "Hurricane"),
        ("Embargo", "Recycle", "Natural Gift", "Assurance"),  # B13-32
        ("Disable", "Hammer Arm", "Thief", "Mist Ball"),  # E3-1-1..E3-1-5
        ("Skill Swap", "Worry Seed", "Accelerock", ""),  # E3-2-5
    ):
        s = _state(t, kaizo_roark_party, BONSLY, _ids(n, *moves))
        s = s._replace(recycle_item=s.recycle_item.at[1].set(lum))
        for faster, stage in ((True, 3), (False, 6)):
            s2 = _stage(s, 0, Stat.ATK, stage)
            s2 = _speeds(s2, player=1, opp=500) if faster else _speeds(s2, player=500, opp=1)
            s2 = s2._replace(pp=s2.pp.at[1, 0, 0].set(2))
            layout = layout_for(t, flags, s2)
            assert policy.ai_refusals(t, s2, n, layout) == []
            _assert_exact_equals_loose(t, s2, layout)


# --- E3 batch 1: Disable, Teleport, Hammer Arm; Mist Ball and Thief have no label (E3-1-1..5) --


def _player_last(s, move: int):
    """``s`` with the player's ``movePrevByBattler`` (``last_move[0]``) set."""
    return s._replace(last_move=s.last_move.at[0].set(move))


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        # Basic_CheckCannotDisable (script.s:650-653): IfBattlerUnderEffect CHECK_DISABLE reads
        # disabledTurns (trainer_ai.c:1833-1836), so the last disabled turn (turns 0, the move
        # still set) is not seen.
        ("plain", 0),
        ("disabled", -8),
        ("last turn", 0),
    ],
)
def test_disable_basic(kaizo_tables, kaizo_names, kaizo_roark_party, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Disable", "Accelerock", "", ""))
    if case != "plain":
        s = s._replace(
            disabled_move=s.disabled_move.at[0].set(n.move("Tackle")),
            disable_turns=s.disable_turns.at[0].set(3 if case == "disabled" else 0),
        )
    assert _delta(t, s, 0x2, 0x3)[0] == expected


@pytest.mark.parametrize(
    ("case", "last", "jumps", "expected"),
    [
        # Expert_Disable (script.s:3075-3093): slower than the player, nothing; else the power of
        # the player's last move (none reads 0): above 0 +1, else -1 unless the :3089 roll
        # (N 100) jumps. Kaizo's Thief has power 40, so it reads as a damaging move.
        ("slower", "Tackle", (), 0),
        ("slower", "Leer", (), 0),
        ("faster", "Tackle", (), 1),
        ("faster", "Thief", (), 1),
        ("faster", "Leer", (), -1),
        ("faster", "Leer", (3089,), 0),
        ("faster", "", (), -1),
        ("faster", "", (3089,), 0),
        ("faster", "Tackle", (3089,), 1),
    ],
)
def test_disable_expert(kaizo_tables, kaizo_names, kaizo_roark_party, case, last, jumps, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Disable", "Accelerock", "", ""))
    s = _speeds(s, player=500, opp=1) if case == "slower" else _speeds(s, player=1, opp=500)
    s = _player_last(s, n.move(last) if last else 0)
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    assert _delta(t, s, 0x3, 0x7, tuple((0, line) for line in jumps))[0] == expected


def test_teleport_is_minus_ten_with_no_expert_entry(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_ScoreMoveEffect :221: FLEE_FROM_WILD_BATTLE jumps straight to ScoreMinus10 (no test,
    no roll); Expert_Main has no entry for it."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Teleport", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == -10
    assert _delta(t, s, 0x3, 0x7)[0] == 0


_POISON, _FLYING, _BUG, _GHOST = 3, 2, 6, 7


@pytest.mark.parametrize(
    ("typing", "case", "expected"),
    [
        # Expert_HammerArm (script.s:4849-4867): immune, half or quarter damage -1; else slower
        # than the player +1 (the SLOWER compare reads no tie); else nothing. No roll.
        (None, "slower", 1),
        (None, "faster", 0),
        ((_POISON, _POISON), "slower", -1),  # Fighting against Poison: half
        ((_FLYING, _BUG), "faster", -1),  # quarter
        ((_GHOST, _GHOST), "slower", -1),  # immune
    ],
)
def test_hammer_arm_expert(kaizo_tables, kaizo_names, kaizo_roark_party, typing, case, expected):
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Hammer Arm", "Accelerock", "", ""))
    s = _speeds(s, player=500, opp=1) if case == "slower" else _speeds(s, player=1, opp=500)
    if typing:
        s = s._replace(types=s.types.at[0, 0].set(jnp.asarray(typing, s.types.dtype)))
    assert _delta(t, s, 0x3, 0x7)[0] == expected


@pytest.mark.parametrize("move", ["Mist Ball", "Mystical Fire", "Thief"])
def test_label_less_moves_score_through_the_damage_path_only(
    kaizo_tables, kaizo_names, kaizo_roark_party, move
):
    """LOWER_SP_ATK_HIT (71) and DO_NOTHING (85) have no Basic or Expert entry (script.s): the
    refusal is lifted and Expert adds nothing. Kaizo's Thief keeps its 40-power damage reads
    (trainer_ai.c:31-45 does not list effect 85, E3-1-2), so it stays a damaging move for the
    AI."""
    from nuzline.ai.layout import ALLIN

    t, n = kaizo_tables, kaizo_names
    assert policy.opponent_move_refusals(t, n.move(move), ALLIN) == []
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, move, "Accelerock", "", ""))
    assert _delta(t, s, 0x3, 0x7)[0] == 0
    assert bool(c.is_damaging(t, n.move(move)))


# --- E3 batch 1 round 2: Torment, Worry Seed, Me First, Trick Room (E3-2-1 .. E3-2-4) ---------


def test_torment_basic(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckTorment (script.s:874-877): the player already tormented, -10, else nothing; no
    Expert entry (E3-2-1)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Torment", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    tormented = s._replace(torment=s.torment.at[0].set(1))
    assert _delta(t, tormented, 0x2, 0x3)[0] == -10
    assert _delta(t, tormented, 0x3, 0x7)[0] == 0


def test_worry_seed_basic(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckWorrySeed (script.s:1428-1443): the player's known ability Truant, Insomnia or
    Vital Spirit -10; asleep and (as far as the AI has seen) without Sleep Talk and Snore -10;
    else nothing (E3-2-2)."""
    from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId, Status

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Worry Seed", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0
    for ability in (AbilityId.TRUANT, AbilityId.INSOMNIA, AbilityId.VITAL_SPIRIT):
        known = s._replace(
            ability=s.ability.at[0, 0].set(ability),
            ai_ability_known=s.ai_ability_known.at[0].set(ABILITY_DECOMP_IDS[ability]),
        )
        assert _delta(t, known, 0x2, 0x3)[0] == -10, ability
    asleep = s._replace(status=s.status.at[0, 0].set(Status.SLEEP))
    assert _delta(t, asleep, 0x2, 0x3)[0] == -10
    assert _delta(t, asleep, 0x3, 0x7)[0] != -10  # Expert adds, never the Basic -10


def test_worry_seed_expert(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Expert_WorrySeed (script.s:5973-5993), the AI above 50 % HP: the 128 roll (:5984) not
    jumping +1, then the 64 roll (:5988) not jumping +1; each jump drops its +1 (E3-2-2)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Worry Seed", "Accelerock", "", ""))
    assert _delta(t, s, 0x3, 0x7)[0] == 2
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5984),))[0] == 1
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5988),))[0] == 1
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5984), (0, 5988)))[0] == 0


def test_me_first_expert(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Expert_MeFirst (script.s:5519-5550): slower than the player -2 (and nothing else); faster,
    the player's last move not a status one: +1 for the 128 roll (:5538) not jumping and +1 for
    the 64 roll (:5542), plus +1 first when the player out-damages the AI's best and the 32 roll
    (:5532) does not jump (E3-2-3). Risky_RiskyEffects lists the effect (:6560)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Me First", "Accelerock", "", ""))
    slow = s._replace(stats=s.stats.at[1, 0, 4].set(1))
    assert _delta(t, slow, 0x3, 0x7)[0] == -2
    fast = s._replace(stats=s.stats.at[1, 0, 4].set(500).at[0, 0, 4].set(1))
    assert _delta(t, fast, 0x3, 0x7, jumps=((0, 5538), (0, 5532)))[0] == 0
    assert _delta(t, fast, 0x3, 0x7, jumps=((0, 5538),))[0] in (0, 1)
    assert _delta(t, fast, 0x3, 0x7)[0] in (2, 3)
    assert c.BattleEffect.USE_MOVE_FIRST in c.RISKY_EFFECTS


def test_trick_room_basic(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckTrickRoom (script.s:1501-1506): faster than the player, or a speed tie (read as
    faster, :1503), -10; slower nothing (E3-2-4)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Trick Room", "Accelerock", "", ""))
    assert _delta(t, _speeds(s, player=1, opp=500), 0x2, 0x3)[0] == -10
    assert _delta(t, _speeds(s, player=500, opp=1), 0x2, 0x3)[0] == 0
    assert _delta(t, _speeds(s, player=100, opp=100), 0x2, 0x3, tie=1)[0] == -10


def test_trick_room_expert(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Expert_TrickRoom (script.s:6190-6214): faster -1; slower +3 unless the 64 roll (:6210)
    jumps; the AI at 30 % HP or less with no living party member left scores nothing (E3-2-4)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Trick Room", "Accelerock", "", ""))
    slow, fast = _speeds(s, player=500, opp=1), _speeds(s, player=1, opp=500)
    assert _delta(t, fast, 0x3, 0x7)[0] == -1
    assert _delta(t, slow, 0x3, 0x7)[0] == 3
    assert _delta(t, slow, 0x3, 0x7, jumps=((0, 6210),))[0] == 0
    # Low HP: a living bench member still goes to the speed check, none ends the script.
    low = slow._replace(hp=slow.hp.at[1, 0].set(1))
    assert _delta(t, low, 0x3, 0x7)[0] == 3
    alone = low._replace(hp=low.hp.at[1, 1].set(0))
    assert _delta(t, alone, 0x3, 0x7)[0] == 0


# --- E3-2-5 Skill Swap (#191) ----------------------------------------------------------------


def _swap_delta(t, s, guess: int = 0, jumps=()) -> list[int]:
    """The Expert flag's score change (0x7 over 0x3) per slot, as ``_delta`` with the defender
    ability guess ``guess`` (the species coin, trainer_ai.c:1191-1198) taken."""
    lo, hi = layout_for(t, 0x3, s), layout_for(t, 0x7, s)
    before = _layout_scores(t, s, _outcome(t, s, guess=guess, layout=lo), lo)
    after = _layout_scores(t, s, _outcome(t, s, guess=guess, jumps=jumps, layout=hi), hi)
    return [int(b) - int(a) for a, b in zip(before, after, strict=True)]


def test_change_user_ability_tables_are_the_decomps(kaizo_tables, kaizo_names):
    """``commands.CHANGE_USER_ABILITY_DESIRABLE`` is Expert_ChangeUserAbility_DesirableAbilities
    (script.s:4435-4461) entry by entry, and Skill Swap is in the Encore table twice
    (script.s:3229, :3256)."""
    names = kaizo_names.ability_names
    spelled = [names[i].upper().replace(" ", "_") for i in c.CHANGE_USER_ABILITY_DESIRABLE]
    assert len(set(c.CHANGE_USER_ABILITY_DESIRABLE)) == 25
    assert c.ENCORE_ENCOURAGED_EFFECTS.count(BattleEffect.SWITCH_ABILITIES) == 2
    script = DECOMP / "src/battle/trainer_ai/script.s"
    if not script.exists():
        pytest.skip("decomp not present")
    lines = script.read_text(encoding="utf-8").splitlines()
    start = lines.index("Expert_ChangeUserAbility_DesirableAbilities:")
    entries = []
    for line in lines[start + 1 :]:
        name = line.split()[1]
        if name == "TABLE_END":
            break
        entries.append(name.removeprefix("ABILITY_"))
    assert spelled == entries


def test_change_user_ability_expert(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Expert_ChangeUserAbility (script.s:4415-4433): the AI's own desirable ability -1; else the
    player's desirable one +2 unless the 50 roll (:4429) jumps; neither falls through to -1
    (the script's comment says +0, the code does not). An unknown player ability guesses its
    species' slot (the coin): slot 0 desirable +2, slot 1 not -1 (E3-2-5)."""
    from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Skill Swap", "Accelerock", "", ""))

    def known(state, ability):
        return state._replace(
            ability=state.ability.at[0, 0].set(ability),
            ai_ability_known=state.ai_ability_known.at[0].set(ABILITY_DECOMP_IDS[ability]),
        )

    good, plain = known(s, AbilityId.SPEED_BOOST), known(s, AbilityId.KEEN_EYE)
    assert _delta(t, good, 0x3, 0x7)[0] == 2
    assert _delta(t, good, 0x3, 0x7, jumps=((0, 4429),))[0] == 0
    assert _delta(t, plain, 0x3, 0x7)[0] == -1
    # The AI's own desirable ability decides first, whatever the player holds.
    own = s._replace(ability=s.ability.at[1, 0].set(AbilityId.HUGE_POWER))
    assert _delta(t, known(own, AbilityId.SPEED_BOOST), 0x3, 0x7)[0] == -1
    assert _delta(t, known(own, AbilityId.KEEN_EYE), 0x3, 0x7)[0] == -1
    # The coin: the player's species has Speed Boost in slot 0 and Keen Eye in slot 1.
    t2 = _with_abilities(t, int(s.species[0, 0]), [3, 51])
    assert _swap_delta(t2, s, guess=0)[0] == 2
    assert _swap_delta(t2, s, guess=0, jumps=((0, 4429),))[0] == 0
    assert _swap_delta(t2, s, guess=1)[0] == -1


def test_gastro_acid_tables_are_the_decomps(kaizo_tables, kaizo_names):
    """Gastro Acid is BATTLE_EFFECT_SUPRESS_ABILITY, 239 (generated/move_battle_effects.txt line
    240), in Basic_ScoreMoveEffect (script.s:269), Expert_Main (:1782) and the Encore table once
    (:3276); Basic_CheckGastroAcid's seven abilities are the decomp's ids (E3-2-6)."""
    assert BattleEffect.SUPRESS_ABILITY == 239
    assert c.ENCORE_ENCOURAGED_EFFECTS.count(BattleEffect.SUPRESS_ABILITY) == 1
    names = kaizo_names.ability_names
    for name in (
        "Multitype",
        "Truant",
        "Slow Start",
        "Stench",
        "Run Away",
        "Pickup",
        "Honey Gather",
    ):
        assert names[c.ABILITY_IDS[name]].upper() == name.upper()
    script = DECOMP / "src/battle/trainer_ai/script.s"
    if not script.exists():
        pytest.skip("decomp not present")
    lines = script.read_text(encoding="utf-8").splitlines()
    assert "BATTLE_EFFECT_SUPRESS_ABILITY" in lines[268] and "Basic_CheckGastroAcid" in lines[268]
    assert "BATTLE_EFFECT_SUPRESS_ABILITY" in lines[1781] and "Expert_GastroAcid" in lines[1781]
    assert lines[5480].strip() == "IfRandomLessThan 64, Expert_GastroAcid_End"
    assert lines[5483].strip() == "IfRandomLessThan 128, Expert_GastroAcid_ContinueHPCheck"
    start = lines.index("Basic_CheckGastroAcid:")
    entries = []
    for line in lines[start + 1 :]:
        if line.strip() == "PopOrEnd":
            break
        if line.strip().startswith("IfLoadedEqualTo"):
            entries.append(line.split()[1].removeprefix("ABILITY_").rstrip(","))
    assert entries == [
        "MULTITYPE", "TRUANT", "SLOW_START", "STENCH", "RUN_AWAY", "PICKUP", "HONEY_GATHER"
    ]  # fmt: skip


def test_gastro_acid_basic(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Basic_CheckGastroAcid (script.s:1363-1376): the player already suppressed (``IfMoveEffect``)
    -10, once (the script ends there, so a known Truant adds nothing, and LoadBattlerAbility would
    read NONE anyway); else a known Multitype, Truant, Slow Start, Stench, Run Away, Pickup or
    Honey Gather -10; any other ability nothing (E3-2-6)."""
    from nuzline.engine.ids import ABILITY_DECOMP_IDS, AbilityId

    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Gastro Acid", "Accelerock", "", ""))
    assert _delta(t, s, 0x2, 0x3)[0] == 0

    def known(state, ability, decomp):
        return state._replace(
            ability=state.ability.at[0, 0].set(ability),
            ai_ability_known=state.ai_ability_known.at[0].set(decomp),
        )

    # Each ability as the AI knows it (the decomp id of its record, ``ai_ability_known``).
    for decomp in (121, 54, 112, 1, 50, 53, 118):
        assert _delta(t, known(s, AbilityId.UNSUPPORTED, decomp), 0x2, 0x3)[0] == -10, decomp
    keen = known(s, AbilityId.KEEN_EYE, ABILITY_DECOMP_IDS[AbilityId.KEEN_EYE])
    assert _delta(t, keen, 0x2, 0x3)[0] == 0
    off = s._replace(ability_suppressed=s.ability_suppressed.at[0].set(1))
    assert _delta(t, off, 0x2, 0x3)[0] == -10
    both = known(off, AbilityId.TRUANT, ABILITY_DECOMP_IDS[AbilityId.TRUANT])
    assert _delta(t, both, 0x2, 0x3)[0] == -10


@pytest.mark.parametrize(
    ("percent", "plain", "skipped"),
    [
        # Expert_GastroAcid (script.s:5471-5494, the code, not its comment): +1 after the 64 roll
        # (:5481); above 70 % HP it ends; else the 128 roll (:5484) not jumping -1; above 50 %
        # ends, else -1; above 30 % ends, else -1.
        (100, 1, 1),
        (71, 1, 1),
        (70, 0, 1),
        (51, 0, 1),
        (50, -1, 0),
        (31, -1, 0),
        (30, -2, -1),
        (1, -2, -1),
    ],
)
def test_gastro_acid_expert(kaizo_tables, kaizo_names, kaizo_roark_party, percent, plain, skipped):
    """Expert_GastroAcid with both rolls: no jump gives ``plain``, the 128 roll (:5484) jumping
    gives ``skipped`` (it drops the first -1), the 64 roll (:5481) jumping gives 0 at any HP
    (E3-2-6)."""
    t, n = kaizo_tables, kaizo_names
    s = _state(t, kaizo_roark_party, BONSLY, _ids(n, "Gastro Acid", "Accelerock", "", ""))
    s = s._replace(hp=s.hp.at[0, 0].set(int(s.max_hp[0, 0]) * percent // 100))
    assert _delta(t, s, 0x3, 0x7)[0] == plain
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5484),))[0] == skipped
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5481),))[0] == 0
    assert _delta(t, s, 0x3, 0x7, jumps=((0, 5481), (0, 5484)))[0] == 0
