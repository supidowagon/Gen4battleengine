"""E1 Roark close batch (rulings G-35 onward, docs/kaizo/e1-mechanics-ledger.md): hand cases.

Family 1: the battle terrain of the Roark split's fights whose moves read it (G-35, G-36), and
the clowns' (G-38).
Family 2: AI_FLAG_HARRASSMENT (G-37).
Family 3: TrainerAI_CalcDamage's draws, Psywave's and Magnitude's id cases (G-39).
"""

from __future__ import annotations

import pytest

from nuzline.data.trainer import TERRAIN_NAMES
from nuzline.engine.coverage import TERRAIN_EFFECTS, terrain_refusal

# --- Family 1: battle terrain (G-35, G-36) -------------------------------------------------

# Kaizo trainer ids -> the terrain tools/terrain.py narrows their battle to (G-35): Barry #2
# (247-249) starts in the Route 203 coord event's rectangle (Route203_CoordEvent_Rival), Lass
# Kaitlin (323) on her sight and talk tiles, Camper Zackary (377, Gardenia split) on his.
NARROWED = {
    247: "TERRAIN_PLAIN",
    248: "TERRAIN_PLAIN",
    249: "TERRAIN_PLAIN",
    323: "TERRAIN_GRASS",
    377: "TERRAIN_MOUNTAIN",
}


def _moves(trainer) -> list[int]:
    return [m for mon in trainer.party for m in (mon.moves or ()) if m]


@pytest.mark.parametrize("tid, name", sorted(NARROWED.items()))
def test_roark_split_terrain_readers_have_their_narrowed_terrain(
    kaizo_tables, kaizo_names, kaizo_trainers, tid, name
):
    t = next(x for x in kaizo_trainers if x.id == tid)
    assert TERRAIN_NAMES[t.terrain] == name and t.terrains == (TERRAIN_NAMES.index(name),)
    moves = _moves(t)
    effects = {int(kaizo_tables.move_ai_effect[m]) for m in moves}
    assert effects & set(TERRAIN_EFFECTS)  # Secret Power reads it (coverage.TERRAIN_EFFECTS)
    assert terrain_refusal(kaizo_tables, kaizo_names, t, moves) is None


@pytest.mark.parametrize("tid", [252, 254])
def test_clowns_fight_on_jubilife_buildings(kaizo_trainers, tid):
    """G-38 (closing G-36): Clowns Ronald (252) and Moe (254) are fought only in Jubilife City.
    Route 202's script file carries their battles as entries no Route 202 event runs, and its
    C-started tag battle's variable partner operand names a partner, not them
    (scrcmd_trainer.c:146-161), so tools/terrain.py drops Route 202 and the data gives
    TERRAIN_BUILDING (rebuilt with tools.extract_rom / tools.build_tables --game kaizo)."""
    t = next(x for x in kaizo_trainers if x.id == tid)
    building = TERRAIN_NAMES.index("TERRAIN_BUILDING")
    assert t.terrain == building and t.terrains == (building,)


# --- Family 2: AI_FLAG_HARRASSMENT (G-37) ----------------------------------------------------

LEER, THUNDER_WAVE = 43, 86  # Kaizo ids: DEF_DOWN (19), STATUS_PARALYZE (67)


def test_harrassment_effects_are_the_decomps():
    """``commands.HARRASSMENT_EFFECTS`` is Harrassment_Effects (script.s:8036-8071) entry by
    entry, through generated/move_battle_effects.txt (line n + 1 is id n)."""
    from nuzline.ai import commands as c
    from tests.ai.test_kaizo_policy import DECOMP

    script = DECOMP / "src/battle/trainer_ai/script.s"
    ids = DECOMP / "generated/move_battle_effects.txt"
    if not script.exists() or not ids.exists():
        pytest.skip("decomp not present")
    names = ids.read_text(encoding="utf-8").split()
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[8035] == "Harrassment_Effects:"
    assert lines[8029].split() == ["IfRandomLessThan", "128,", "Harrassment_Terminate"]
    flag_table = lines.index("FlagTable:")
    assert "Harrassment_Main" in lines[flag_table + 1 + 10]  # AI_FLAG_HARRASSMENT, bit 10
    entries = []
    for line in lines[8036:]:
        name = line.split()[1]
        if name == "TABLE_END":
            break
        entries.append(names.index(name))
    assert tuple(entries) == c.HARRASSMENT_EFFECTS


@pytest.mark.parametrize("jump", [0, 1])
def test_harrassment(kaizo_tables, kaizo_roark_party, jump):
    """Harrassment_Main (script.s:8021-8034): Leer (DEF_DOWN, :8039) and Thunder Wave
    (STATUS_PARALYZE, :8048) get +2 unless the 128 roll (:8030) jumps; Stealth Rock and Rock
    Polish (effects outside the table) neither roll nor score. Against BASIC, EVAL_ATTACK and
    EXPERT (set 0x7 -> 0x407); exact equals loose under the new set."""
    from tests.ai.test_kaizo_outcome import _assert_exact_equals_loose, layout_for
    from tests.ai.test_kaizo_policy import (
        BONSLY,
        ROCK_POLISH,
        STEALTH_ROCK,
        _delta,
        _path_lines,
        _state,
    )

    t = kaizo_tables
    effect = [int(t.move_ai_effect[m]) for m in (LEER, THUNDER_WAVE)]
    assert effect == [19, 67]
    s = _state(t, kaizo_roark_party, BONSLY, (STEALTH_ROCK, LEER, THUNDER_WAVE, ROCK_POLISH))
    jumps = ((1, 8030), (2, 8030)) if jump else ()
    plus = 0 if jump else 2
    assert _delta(t, s, 0x7, 0x407, jumps) == [0, plus, plus, 0]
    rolls = [8030 in lines for lines in _path_lines(t, s, layout_for(t, 0x407, s))]
    assert rolls == [False, True, True, False]
    _assert_exact_equals_loose(t, s, layout_for(t, 0x407, s))


# --- Family 3: TrainerAI_CalcDamage's draws (G-39) --------------------------------------------

TRIPLE_AXEL, BULLDOZE, COPYCAT = 149, 222, 383  # Kaizo ids: MOVE_PSYWAVE's, MOVE_MAGNITUDE's


def test_calc_draw_laws():
    """The laws of TrainerAI_CalcDamage's two draws over a uniform u16 RandNext()
    (battle_system.c:1308-1312): Psywave ``RandNext() % 11`` (trainer_ai.c:3015), 65536 = 11 *
    5957 + 9, so classes 0-8 have 5958 values and 9-10 have 5957; Magnitude ``RandNext() % 100``
    bucketed below 5, 15, 35, 65, 85, 95 (:3030-3046), 65536 = 100 * 655 + 36, so residues 0-35
    have 656 values and 36-99 have 655: 5 * 656, 10 * 656, 20 * 656, 656 + 29 * 655, 20 * 655,
    10 * 655 and 5 * 655. Another move draws nothing (class 0)."""
    import jax.numpy as jnp
    import numpy as np

    from nuzline.ai.outcome import CALC_DRAW_P, CALC_MAGNITUDE, CALC_PSYWAVE, calc_draw_class

    psywave = [5958] * 9 + [5957] * 2
    magnitude = [5 * 656, 10 * 656, 20 * 656, 656 + 29 * 655, 20 * 655, 10 * 655, 5 * 655]
    assert (CALC_DRAW_P[CALC_PSYWAVE] * 65536).round().astype(int).tolist() == psywave
    got = (CALC_DRAW_P[CALC_MAGNITUDE] * 65536).round().astype(int).tolist()
    assert got == magnitude + [0] * 4
    r = jnp.arange(65536)
    assert np.array_equal(np.asarray(calc_draw_class(TRIPLE_AXEL, r)), np.arange(65536) % 11)
    bucket = np.asarray(calc_draw_class(BULLDOZE, r))
    edges = [0, 5, 15, 35, 65, 85, 95, 100]
    for c in range(7):
        assert set(np.flatnonzero(bucket == c) % 100) == set(range(edges[c], edges[c + 1]))
    assert not np.asarray(calc_draw_class(LEER, r)).any()


def test_psywave_and_magnitude_cases(kaizo_tables, kaizo_roark_party):
    """MOVE_PSYWAVE's case (trainer_ai.c:3014-3016): ``level * (draw + 5) / 10`` as fixed damage
    (no STAB or chart multiplier, :3081-3104), and at level 1 a draw below 5 gives 0, which
    goes to CalcMoveDamage at the table power (:3073). MOVE_MAGNITUDE's case (:3028-3048) gives
    the bucket's power 10, 30, 50, 70, 90, 110, 150 to CalcMoveDamage, the type staying the
    table's: the battle's damage with that table power at the top roll (the AI applies the
    variance last, here 100)."""
    import numpy as np

    from nuzline.engine.damage import ai_damage_for_roll, damage_for_roll
    from tests.ai.test_kaizo_policy import BONSLY, _state

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (TRIPLE_AXEL, BULLDOZE, 0, 0))
    level = int(s.level[1, 0])
    for c in range(11):
        assert int(ai_damage_for_roll(t, s, 1, TRIPLE_AXEL, 15, draw=c)) == level * (c + 5) // 10
    low = s._replace(level=s.level.at[1, 0].set(1))
    table = int(damage_for_roll(t, low, 1, TRIPLE_AXEL, False, 15))
    for c in range(11):
        fixed = (c + 5) // 10
        assert int(ai_damage_for_roll(t, low, 1, TRIPLE_AXEL, 15, draw=c)) == (fixed or table)
    for c, power in enumerate((10, 30, 50, 70, 90, 110, 150)):
        t2 = t._replace(move_power=np.asarray(t.move_power).copy())
        t2.move_power[BULLDOZE] = power
        battle = int(damage_for_roll(t2, s, 1, BULLDOZE, False, 15))
        assert int(ai_damage_for_roll(t, s, 1, BULLDOZE, 15, draw=c)) == battle, power


@pytest.mark.parametrize(("move", "hp"), [(BULLDOZE, 40), (BULLDOZE, 60), (TRIPLE_AXEL, 14)])
def test_eval_attack_reads_each_estimate_once(kaizo_tables, kaizo_roark_party, move, hp):
    """EvalAttack_Main (script.s:6353-6363) under EVAL_ATTACK alone: IfCurrentMoveKills (:6358,
    trainer_ai.c:1560) and FlagMoveDamageScore (:6361, :1041) each estimate the move with a
    fresh draw. Slot 0 scores 104 when it kills (+4, :6411), else 99 when Accelerock or Headbutt
    out-damages it (-1, :6362), else 100, or 102 for a quadruple chart unless :6383 jumps
    (Triple Axel's Ice into Gible's Dragon/Ground): the two draws are
    independent reads, P(kill) = sum_c P(c) [hp <= d_c], P(not highest) = sum_c P(c)
    [max(others) > d_c]. Exact equals loose (each draw its own digit) and the brute-force
    enumeration; the path makes the two reads (G-39)."""
    import numpy as np

    from nuzline.ai import commands as c
    from nuzline.ai import policy
    from nuzline.ai.outcome import CALC_DRAW_P, calc_draw_kind
    from tests.ai.test_kaizo_outcome import (
        ACCELEROCK,
        HEADBUTT,
        _assert_exact_equals_loose,
        _exact_slot_scores,
        layout_for,
    )
    from tests.ai.test_kaizo_policy import BONSLY, _state

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (move, ACCELEROCK, HEADBUTT, 0), hp=hp)
    layout = layout_for(t, 0x2, s)
    d = np.array([int(c.ai_damage_for_roll(t, s, 1, move, 15, draw=k)) for k in range(11)])
    law = CALC_DRAW_P[int(calc_draw_kind(move))]
    others = max(int(c.ai_damage(t, s, m, c.SIM_ROLL_MAX)) for m in (ACCELEROCK, HEADBUTT))
    p_kill = float(law @ (hp <= d))
    p_low = float(law @ (others > d))
    assert 0 < p_kill < 1 or 0 < p_low < 1  # the draws matter here
    hist = _exact_slot_scores(t, s, layout)[0]
    rest = (1 - p_kill) * (1 - p_low)
    want = {104: p_kill, 99: (1 - p_kill) * p_low, 100: rest}
    quad = int(c.effectiveness_class(t, s, move)) == c.TypeMulti.QUADRUPLE  # Ice into Gible
    if quad:
        want.update({100: rest * 80 / 256, 102: rest * 176 / 256})  # :6383 unless it jumps
    for score, p in want.items():
        assert hist[score] == pytest.approx(p, abs=1e-6), score
    _assert_exact_equals_loose(t, s, layout)
    enum = np.asarray(policy.ai_action_distribution_enumerated(t, s, layout), np.float64)
    np.testing.assert_allclose(enum, np.asarray(policy._distribution(t, s, layout)), atol=1e-6)
    _, reads = policy._path_records(t, s, layout)
    assert int(np.asarray(reads)[:, 0].max()) == 2 + quad <= layout.slot_reads  # + :6383


def test_basic_draws_are_no_read(kaizo_tables, kaizo_roark_party):
    """Basic_Main's FlagMoveDamageScore (script.s:62, :112) is compared only with
    AI_NO_COMPARISON_MADE, which the move alone decides (trainer_ai.c:1035-1036), and calcTemp
    is loaded anew (:67, :117): its draws change nothing, so the exact layout makes no read
    for them, while the loose oracle draws each one (two on Bulldoze's slot's path) and still
    equals it."""
    import numpy as np

    from nuzline.ai import policy
    from nuzline.ai.commands import ReadKind
    from tests.ai.test_kaizo_outcome import (
        ACCELEROCK,
        HEADBUTT,
        _assert_exact_equals_loose,
        _one,
        layout_for,
    )
    from tests.ai.test_kaizo_policy import BONSLY, _state

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (BULLDOZE, ACCELEROCK, HEADBUTT, 0), hp=40)
    layout = layout_for(t, 0x1, s)
    _, reads = policy._path_records(t, s, layout)
    lo_reads, _, kind = policy._loose_reads(t, _one(s), layout)
    calc = [r for r in lo_reads[0] if kind[0, r] == ReadKind.CALC_MAGNITUDE]
    assert len(calc) == 2
    assert int(np.asarray(reads)[:, 0].max()) == len(lo_reads[0]) - len(calc)
    _assert_exact_equals_loose(t, s, layout)


def test_party_member_estimates_draw(kaizo_tables, kaizo_roark_party):
    """Expert_UTurn's IfPartyMemberDealsMoreDamage (script.s:5118; trainer_ai.c:2106-2174):
    Cranidos behind Bonsly knows Bulldoze, estimated as Bonsly (its stats and level) with a
    fresh Magnitude draw (:2156). Its jump is one read, ``1 - sum_v P(A = v) P(M <= v)``; exact
    equals loose and the sampled AI (raw RandNext draws, ``AiOutcome.calc_draws``) matches."""
    import jax
    import numpy as np
    from scipy.stats import chisquare

    from nuzline.ai import policy
    from nuzline.ai.outcome import sample_ai_outcome
    from nuzline.engine.actions import NUM_OPP_ACTIONS
    from tests.ai.test_kaizo_outcome import (
        ACCELEROCK,
        HEADBUTT,
        _assert_exact_equals_loose,
        layout_for,
    )
    from tests.ai.test_kaizo_policy import BONSLY, CRANIDOS, _state

    t = kaizo_tables
    u_turn = 369
    s = _state(t, kaizo_roark_party, BONSLY, (u_turn, HEADBUTT, 0, 0), bench=(CRANIDOS,), hp=300)
    s = s._replace(moves=s.moves.at[1, 1].set(np.array([BULLDOZE, ACCELEROCK, 0, 0])))
    layout = layout_for(t, 0x7, s)
    _assert_exact_equals_loose(t, s, layout)
    dist = np.asarray(policy._distribution(t, s, layout), np.float64)
    keys = jax.random.split(jax.random.PRNGKey(3), 20_000)
    act = jax.jit(
        jax.vmap(lambda k: policy._policy(t, s, sample_ai_outcome(k, t, s, layout), layout))
    )(keys)
    counts = np.bincount(np.asarray(act), minlength=NUM_OPP_ACTIONS)
    support = dist > 0
    assert counts[~support].sum() == 0
    expected = dist[support] / dist[support].sum() * counts.sum()
    assert chisquare(counts[support], expected).pvalue > 1e-3


@pytest.mark.parametrize("speed", ["slower", "faster"])
def test_battler_estimate_draws(kaizo_tables, kaizo_roark_party, speed):
    """Expert_Copycat's IfBattlerDealsMoreDamage (script.s:5560, :5573; trainer_ai.c:2187-2240):
    the player's last move is Triple Axel, estimated with no damaging test and a fresh Psywave
    draw at the player's level (:2228, ``battleMons[battler].level``), against the AI's best
    (:2210). Its jump is one read, P(B > A) = sum_c P(c) [b_c > A]; slower, the path reads it
    once (:5573), faster up to twice (:5560, then :5573). Exact equals loose."""
    import numpy as np

    from nuzline.ai import commands as c
    from nuzline.ai.outcome import CALC_DRAW_P, CALC_PSYWAVE
    from tests.ai.test_kaizo_outcome import ACCELEROCK, _assert_exact_equals_loose, layout_for
    from tests.ai.test_kaizo_policy import BONSLY, _speeds, _state, _turn

    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, (COPYCAT, ACCELEROCK, 0, 0))
    s = _turn(_speeds(s, player=1 if speed == "faster" else 500, opp=100), 3)
    s = s._replace(level=s.level.at[0, 0].set(10), last_move=s.last_move.at[0].set(TRIPLE_AXEL))
    best = int(c.ai_damage(t, s, ACCELEROCK, c.SIM_ROLL_MAX))
    ivs = s.ivs[1, 0]
    b = np.array(
        [
            int(c.ai_damage_for_roll(t, s, 0, TRIPLE_AXEL, 15, def_side=0, ivs=ivs, draw=k))
            for k in range(11)
        ]
    )
    assert (b == 10 * (np.arange(11) + 5) // 10).all()  # fixed: the player's level 10
    p = float(CALC_DRAW_P[CALC_PSYWAVE] @ (b > best))
    assert 0 < p < 1, (best, b)
    _assert_exact_equals_loose(t, s, layout_for(t, 0x7, s))


@pytest.mark.parametrize(
    ("command", "moves", "hp"),
    [
        ("kills", (BULLDOZE, "accelerock", "headbutt", 0), 40),
        ("kills", (TRIPLE_AXEL, "accelerock", "headbutt", 0), 14),
        ("highest", (TRIPLE_AXEL, BULLDOZE, "accelerock", "headbutt"), 400),
    ],
)
def test_raw_draw_classes_score_the_exact_p_yes(
    kaizo_tables, kaizo_roark_party, command, moves, hp
):
    """A sampled per-slot outcome scores a damage command from its raw ``RandNext()`` draws
    (``AiOutcome.calc_draws``; G-39): for each class of each drawing estimate (one raw draw of
    that class, in the decomp's draw order), the command's yes/no, and the read's weight is the
    exact ``p_yes`` or ``1 - p_yes``. Summed over the classes' joint law, the yeses give that
    ``p_yes``. IfCurrentMoveKills (trainer_ai.c:1560) draws once; FlagMoveDamageScore over
    Triple Axel and Bulldoze (:1041) twice, Psywave's (slot 0) first."""
    import itertools

    import jax
    import jax.numpy as jnp
    import numpy as np

    from nuzline.ai import commands as c
    from nuzline.ai.outcome import (
        CALC_DRAW_P,
        CALC_DRAWS,
        calc_draw_class,
        calc_draw_kind,
        sample_ai_outcome,
    )
    from nuzline.engine.state import MOVES
    from tests.ai.test_kaizo_outcome import ACCELEROCK, HEADBUTT, layout_for
    from tests.ai.test_kaizo_policy import BONSLY, _state

    names = {"accelerock": ACCELEROCK, "headbutt": HEADBUTT}
    moves = tuple(names.get(m, m) for m in moves)
    t = kaizo_tables
    s = _state(t, kaizo_roark_party, BONSLY, moves, hp=hp)
    layout = layout_for(t, 0x7, s)
    base = sample_ai_outcome(jax.random.PRNGKey(0), t, s, layout)

    @jax.jit
    def run(raw):
        calc = jnp.zeros((MOVES, layout.slot_reads, CALC_DRAWS), jnp.uint16)
        outcome = base._replace(calc_draws=calc.at[0, 0, : raw.shape[0]].set(raw))
        ctx = c.at_slot(c.new_context(t, s, outcome), 0)
        if command == "kills":
            ctx = c.if_current_move_kills(ctx, c.USE_MAX_DAMAGE)
            yes = ctx.jump
        else:
            ctx = c.flag_move_damage_score(ctx, c.USE_MAX_DAMAGE)
            yes = ctx.calc_temp == c.AI_MOVE_IS_HIGHEST_DAMAGE
        return yes, ctx.weight[0], ctx.reads[0]

    drawing = [m for m in (moves[:1] if command == "kills" else moves) if m]
    drawing = [m for m in drawing if int(calc_draw_kind(m))]
    every = np.arange(65536)
    raws, laws = [], []
    for m in drawing:
        cls = np.asarray(calc_draw_class(m, jnp.asarray(every)))
        law = CALC_DRAW_P[int(calc_draw_kind(m))]
        support = np.flatnonzero(law > 0)
        raws.append([int(np.flatnonzero(cls == k)[0]) for k in support])
        laws.append(law[support])
    total, p_yes = 0.0, []
    for combo in itertools.product(*(range(len(r)) for r in raws)):
        raw = jnp.asarray([raws[i][j] for i, j in enumerate(combo)], jnp.uint16)
        yes, weight, reads = run(raw)
        assert int(reads) == 1  # one read for the command, whatever its draws
        p = float(weight) if bool(yes) else 1.0 - float(weight)
        p_yes.append(p)
        total += float(np.prod([laws[i][j] for i, j in enumerate(combo)])) * bool(yes)
    assert max(p_yes) - min(p_yes) < 1e-6, p_yes
    p = p_yes[0]
    assert 0 < p < 1  # the draws decide
    assert total == pytest.approx(p, abs=1e-6)


def test_drawing_moves_are_not_refused_and_widen_the_bound(kaizo_tables, kaizo_trainers):
    """Triple Axel and Bulldoze are no longer refused (``CALC_DAMAGE_MOVES``), and a fight with
    one reads the damage commands (read_bounds.DRAWN_READ_BOUNDS): Leader Gardenia (315, flags
    0xF, Bulldoze on Torterra) goes from 5 reads to 6 (EvalAttack's FlagMoveDamageScore and
    IfCurrentMoveKills on Bulldoze's own path). Every Kaizo fight's scoring stays within
    WIDTH_CAP (``layout.kaizo_layout``; the widest drawing fights, Barry #6 479-481, read 11)."""
    from nuzline.ai import commands as c
    from nuzline.ai.layout import KAIZO, fight_slot_reads, move_read_rows
    from nuzline.ai.policy import opponent_move_refusals
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.enumerate import WIDTH_CAP
    from tests.conftest import KAIZO_INTERMEDIATE

    t = kaizo_tables
    assert TRIPLE_AXEL not in c.CALC_DAMAGE_MOVES and BULLDOZE not in c.CALC_DAMAGE_MOVES
    for move in (TRIPLE_AXEL, BULLDOZE):
        assert "calc damage" not in opponent_move_refusals(t, move, KAIZO)
    data = IntermediateData(KAIZO_INTERMEDIATE)
    gardenia = next(x for x in kaizo_trainers if x.id == 315)
    moves = [m for p in build_trainer_party(gardenia, data) for m in p.moves if m]
    assert BULLDOZE in moves
    assert fight_slot_reads(move_read_rows(t, moves), gardenia.ai_flags) == 6
    without = [m for m in moves if m != BULLDOZE]
    assert fight_slot_reads(move_read_rows(t, without), gardenia.ai_flags) == 5
    most = 0
    for trainer in kaizo_trainers:
        if not trainer.split:
            continue
        party = build_trainer_party(trainer, data)
        rows = move_read_rows(t, [m for p in party for m in p.moves if m])
        most = max(most, fight_slot_reads(rows, trainer.ai_flags))
    assert KAIZO.switch_sites + 1 + 4 * 2**most <= WIDTH_CAP
