"""The per-game AI policy layer (K3c Task 2): ``nuzline.ai.layout`` and the corrections
that need no new script label.

Decomp paths are relative to ``external/pokeplatinum``: ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``, ``script.s`` is ``src/battle/trainer_ai/script.s``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.ai import commands, policy
from nuzline.ai.layout import (
    KAIZO,
    VANILLA,
    AiLayout,
    fight_slot_reads,
    kaizo_layout,
    layout_of,
)
from nuzline.ai.outcome import no_calc_draws
from nuzline.ai.policy import ai_policy, ai_refusals
from nuzline.ai.switching import REFUSED_ACTION
from nuzline.engine.state import make_state
from tests.ai.fixtures import GEODUDE, know_ability, outcome, trace_a, versus
from tests.conftest import K3C_EFFECTS

SELFDESTRUCT, HEAD_SMASH, CRUNCH = 120, 457, 242
RETURN = 216  # Kaizo name of the decomp's MOVE_RETURN
WATER_GUN = 55


def _kaizo_mon(party, species: str, names):
    """The Kaizo Roark party member of ``species``."""
    return next(m for m in party if m.species == names.species(species))


def test_layout_of_reads_tables(tables, kaizo_tables):
    assert isinstance(VANILLA, AiLayout) and isinstance(KAIZO, AiLayout)
    assert layout_of(tables) is VANILLA
    # the kaizo_tables fight transcribes the K3c effects only (E1 Ruling R7)
    assert layout_of(kaizo_tables) == KAIZO._replace(
        transcribed=KAIZO.transcribed & K3C_EFFECTS,
        expert_transcribed=KAIZO.expert_transcribed & K3C_EFFECTS,
    )
    assert int(tables.ai_policy) == 0 and int(kaizo_tables.ai_policy) == 1
    with pytest.raises(TypeError, match="layout_of"):
        jax.jit(lambda t: layout_of(t).water2_ability)(tables)


def test_layout_per_flag_set(tables, kaizo_tables):
    """E1 Rulings R2, R6 and R7: the Kaizo layout is the fight's: its trainer's flag set
    (``Tables.ai_flags``), and its opponents' moves (``Tables.ai_moves``): their read bound
    under that set (``fight_slot_reads``) and their transcribed effects only; refused while
    either is unset; vanilla keeps its one set whatever the flags. Each layout is its own
    static argument with its own program-cache key."""
    from nuzline.ai.layout import fight_effects, fight_layout, fight_slot_reads, move_read_rows
    from nuzline.data.tables import AI_FLAGS_UNSET, AI_MOVES_UNSET

    t = kaizo_tables
    roark = layout_of(t)
    assert roark == fight_layout(t, KAIZO.flags, t.ai_moves)
    assert roark.slot_reads == KAIZO.slot_reads == 7 and roark.flags == KAIZO.flags
    assert roark.transcribed == KAIZO.transcribed & fight_effects(t, t.ai_moves)
    assert roark.transcribed < KAIZO.transcribed and roark.key.startswith("kaizo-0x17-r7-e")
    rows = move_read_rows(t, t.ai_moves)
    sets = (0x1, 0x5, 0x7, 0x17, 0xD, 0xF, 0x1F, 0x207, 0x20F, 0x27, 0x10F)
    for flags in sets:
        layout = layout_of(t._replace(ai_flags=jnp.int32(flags)))
        reads = fight_slot_reads(rows, flags)
        assert layout == roark._replace(flags=flags, slot_reads=reads)
        assert layout.key.startswith(f"kaizo-{flags:#x}-r{reads}-e")
    assert len({kaizo_layout(f, 7).key for f in sets} | {VANILLA.key}) == len(sets) + 1
    assert kaizo_layout(0x17, 7) == KAIZO and kaizo_layout(0x17, 6).key != KAIZO.key
    with pytest.raises(ValueError, match="not set"):
        layout_of(t._replace(ai_flags=AI_FLAGS_UNSET))
    with pytest.raises(ValueError, match="not set"):
        layout_of(t._replace(ai_moves=AI_MOVES_UNSET))
    with pytest.raises(ValueError, match="WIDTH_CAP"):
        kaizo_layout(0x17, 15)
    assert layout_of(tables._replace(ai_flags=jnp.int32(0x17))) is VANILLA


def test_tag_strategy_is_a_kaizo_flag_bit(kaizo_tables):
    """E4-A0: bit 7 is transcribed (``policy._tag_strategy``) and bounded (``read_bounds``); a
    bit-7 fight has a layout whose read bound is the walk's: a Follow Me and a plain hit read at
    most 2 (the plain hit's highest roll and effectiveness roll)."""
    from nuzline.ai.layout import KAIZO_FLAG_BITS, fight_layout

    assert 7 in KAIZO_FLAG_BITS
    layout = fight_layout(kaizo_tables, 0x87, np.asarray([33, 266, 106]))
    assert layout.flags == 0x87 and layout.key.startswith("kaizo-0x87-r")
    only = fight_layout(kaizo_tables, 1 << 7, np.asarray([33, 266, 106]))
    assert only.slot_reads == 2


def test_layout_values():
    # script.s:78 compares Levitate (26) in vanilla, Dry Skin (87) in Kaizo (overlay14 word
    # 0x022249CC, data/kaizo/rom_diff.toml).
    assert (VANILLA.water2_ability, KAIZO.water2_ability) == (26, 87)
    assert VANILLA.transcribed == frozenset(commands.MOVE_EFFECTS)
    # Kaizo transcribes the vanilla-transcribed moves' effects and, from K3c Task 5, those of
    # Kaizo Roark's moves (tests/ai/test_kaizo_policy.py test_roark_moves_and_effects).
    assert frozenset(int(e) for e in commands.MOVE_EFFECTS.values()) < KAIZO.transcribed
    assert (VANILLA.flags, KAIZO.flags) == (0b111, 0b10111)  # Roark's ai_flags 7 and 23


def test_is_damaging_excludes_no_calc_effects(kaizo_tables, kaizo_names, kaizo_roark_party):
    """sNoDamageCalcMoveEffects (trainer_ai.c:31-46) holds HALVE_DEFENSE (7) and
    RECOIL_HALF (269): FlagMoveDamageScore (:1035-1036) and TrainerAI_CalcAllDamage
    (:2827-2828) compute no damage for them, whatever their power."""
    t = kaizo_tables
    assert int(t.move_ai_effect[SELFDESTRUCT]) == 7 and int(t.move_ai_effect[HEAD_SMASH]) == 269
    assert not bool(commands.is_damaging(t, SELFDESTRUCT))
    assert not bool(commands.is_damaging(t, HEAD_SMASH))
    assert bool(commands.is_damaging(t, CRUNCH))
    cranidos = _kaizo_mon(kaizo_roark_party, "Cranidos", kaizo_names)
    gible = _kaizo_mon(kaizo_roark_party, "Gible", kaizo_names)
    s = make_state(t, [gible], [cranidos])
    moves = jnp.array([SELFDESTRUCT, HEAD_SMASH, CRUNCH, 0], s.moves.dtype)
    s = s._replace(moves=s.moves.at[1, 0].set(moves))
    damage = commands._slot_damage(t, s, jnp.full((4,), commands.SIM_ROLL_MAX)).value[:, 0]
    assert int(damage[0]) == 0 and int(damage[1]) == 0 and int(damage[2]) > 0
    assert int(commands.ai_damage(t, s, HEAD_SMASH, commands.SIM_ROLL_MAX)) == 0


def test_kaizo_highest_damage_ignores_head_smash(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Review Focus 1: Cranidos [Head Smash, Zen Headbutt, Crunch, Rock Polish]. Head Smash
    (power 150) gets AI_NO_COMPARISON_MADE and does not count in the comparison; the
    stronger of Zen Headbutt and Crunch is AI_MOVE_IS_HIGHEST_DAMAGE (trainer_ai.c:1035-1061)."""
    t, n = kaizo_tables, kaizo_names
    cranidos = _kaizo_mon(kaizo_roark_party, "Cranidos", n)
    assert [n.move_names[m] for m in cranidos.moves] == [
        "Head Smash",
        "Zen Headbutt",
        "Crunch",
        "Rock Polish",
    ]
    gible = _kaizo_mon(kaizo_roark_party, "Gible", n)
    s = make_state(t, [gible], [cranidos])
    ctx = commands.new_context(t, s, outcome())
    flags = [
        int(commands.flag_move_damage_score(commands.at_slot(ctx, k), 0).calc_temp)
        for k in range(4)
    ]
    damage = [int(commands.ai_damage(t, s, m, commands.SIM_ROLL_MAX)) for m in cranidos.moves]
    assert damage[0] == 0 and damage[3] == 0 and min(damage[1:3]) > 0
    best = 1 if damage[1] >= damage[2] else 2
    expected = [commands.AI_NO_COMPARISON_MADE] * 4
    expected[best] = commands.AI_MOVE_IS_HIGHEST_DAMAGE
    expected[3 - best] = (
        commands.AI_MOVE_IS_HIGHEST_DAMAGE
        if damage[1] == damage[2]
        else commands.AI_NOT_HIGHEST_DAMAGE
    )
    assert flags == expected


def _dry_skin_state(tables, names):
    """Roark's Geodude, Water Gun in slot 0, against a Croagunk whose Dry Skin the AI knows."""
    s = versus(tables, names, GEODUDE, "Croagunk")
    croagunk = names.species("Croagunk")
    dry_skin = 1
    assert names.species_abilities(croagunk)[dry_skin] == names.ability("Dry Skin") == 87
    return know_ability(
        tables,
        s._replace(
            moves=s.moves.at[1, GEODUDE, 0].set(WATER_GUN),
            pp=s.pp.at[1, GEODUDE, 0].set(25),
            ability_slot=s.ability_slot.at[0, 0].set(dry_skin),
            ability=s.ability.at[0, 0].set(tables.species_abilities[croagunk, dry_skin]),
        ),
    )


def test_dry_skin_compare_per_policy(tables, names):
    """Basic_CheckForImmunity's last compare (script.s:78) branches to
    Basic_CheckWaterAbsorption2 (:106-109): a Water move scores -12 against Dry Skin in
    Kaizo. Vanilla compares Levitate again there, which never matches. The Kaizo layout holds
    the fight's own effects (E1 Ruling R7): another effect's label may make a FASTER compare's
    per-slot read (E1 Ruling R8), which this flat outcome has no place for."""
    s = _dry_skin_state(tables, names)
    ctx = commands.at_slot(commands.new_context(tables, s, outcome()), 0)
    own = {int(tables.move_ai_effect[m]) for m in np.asarray(s.moves[1, 0]) if m}
    kaizo = kaizo_layout(KAIZO.flags, KAIZO.slot_reads, own)
    scores = {
        layout.name: int(policy._basic(ctx, layout).scores[0]) - 100 for layout in (VANILLA, kaizo)
    }
    assert scores == {"vanilla": 0, "kaizo": -12}


def test_kaizo_refuses_an_unmodelled_id_case(
    kaizo_tables, kaizo_names, kaizo_roark_party, monkeypatch
):
    """A TrainerAI_CalcDamage move-id case (trainer_ai.c:2885-3071) the AI damage does not
    model is refused by name, whatever the move's effect. Synthetic, so the test does not
    depend on which cases are modelled: Tackle's id added to ``commands.CALC_DAMAGE_MOVES``.
    Kaizo Return (216, effect 0) takes the modelled MOVE_RETURN case (:3018): not refused."""
    t, n = kaizo_tables, kaizo_names
    cranidos = _kaizo_mon(kaizo_roark_party, "Cranidos", n)
    gible = _kaizo_mon(kaizo_roark_party, "Gible", n)
    s = make_state(t, [gible], [cranidos])
    tackle = n.move_names.index("Tackle")
    monkeypatch.setattr(commands, "CALC_DAMAGE_MOVES", (*commands.CALC_DAMAGE_MOVES, tackle))
    with jax.disable_jit():  # _refused reads the tuple at trace time
        reasons = ai_refusals(t, s._replace(moves=s.moves.at[1, 0, 3].set(tackle)), n)
        assert any("Tackle" in r and "TrainerAI_CalcDamage" in r for r in reasons), reasons
        ret = s._replace(moves=s.moves.at[1, 0, 3].set(RETURN))
        assert int(t.move_ai_effect[RETURN]) == 0
        reasons = ai_refusals(t, ret, n)
    assert not any("Return" in r and "TrainerAI_CalcDamage" in r for r in reasons), reasons


def test_move_type_id_case_refused(tables, names, monkeypatch):
    """A TrainerAI_MoveType move-id case (trainer_ai.c:3132-3222) the AI does not model is
    refused by name. Synthetic (Tackle's id added to ``commands.MOVE_TYPE_MOVES``), so the
    test does not depend on which cases are modelled."""
    s = trace_a(tables, names)
    tackle = names.move("Tackle")
    monkeypatch.setattr(commands, "MOVE_TYPE_MOVES", (*commands.MOVE_TYPE_MOVES, tackle))
    s = s._replace(moves=s.moves.at[1, GEODUDE, 2].set(tackle))
    with jax.disable_jit():
        reasons = ai_refusals(tables, s)
    assert any("Tackle" in r and "TrainerAI_MoveType" in r for r in reasons), reasons


def test_traced_tables_without_a_layout_refuse_another_policy(tables, names):
    """Under a trace ``layout_of`` cannot read ``tables``: a caller that passes no layout
    gets the vanilla one, and a game of another policy is refused, not misread."""
    s = trace_a(tables, names)
    kaizo_like = tables._replace(
        ai_policy=jnp.asarray(1, tables.ai_policy.dtype),
        ai_flags=jnp.int32(KAIZO.flags),
        # the fight's moves: the trace state's opponents' (E1 Rulings R6, R7)
        ai_moves=np.asarray(s.moves[1]).ravel().astype(np.int16),
    )
    # Concrete tables resolve their own layout: the Kaizo policy's outcome is per slot.
    reads = layout_of(kaizo_like).slot_reads
    per_slot = outcome()._replace(
        script_rolls=jnp.full((4, reads), 254, jnp.uint8), calc_draws=no_calc_draws((), reads)
    )
    assert int(jax.jit(lambda st: ai_policy(kaizo_like, st, per_slot))(s)) == 0
    traced = jax.jit(lambda t: ai_policy(t, s, outcome()))
    assert int(traced(tables)) == 0
    assert int(traced(kaizo_like)) == REFUSED_ACTION


def test_program_caches_key_on_the_flag_set():
    # Two Kaizo flag sets share layout.name; a program cache keyed on it would serve one
    # set's compiled AI to the other (E1 batch 2a review I1). Keys use layout.key.
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "nuzline"
    bad = [
        f"{p.relative_to(src)}:{i}"
        for p in src.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if '"layout": layout.name' in line
    ]
    assert not bad, bad


@pytest.mark.fullset
def test_every_kaizo_fight_fits_the_width_cap(kaizo_tables, kaizo_trainers):
    """E1 Ruling R6: every Kaizo trainer's fight, under the transcribed bits of its own AI
    flags, has a read bound whose layout passes WIDTH_CAP (``kaizo_layout`` raises otherwise).
    Synthetic flag sets can pass it (every flag with ACC_DOWN bounds at 17); no trainer does."""
    from nuzline.ai.layout import KAIZO_FLAG_BITS, fight_slot_reads, move_read_rows
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from tests.conftest import KAIZO_INTERMEDIATE

    inter = IntermediateData(KAIZO_INTERMEDIATE)
    mask = sum(1 << b for b in KAIZO_FLAG_BITS)
    for trainer in kaizo_trainers:
        party = build_trainer_party(trainer, inter)
        moves = np.array([m for p in party for m in p.moves], np.int32)
        flags = trainer.ai_flags & mask
        reads = fight_slot_reads(move_read_rows(kaizo_tables, moves), flags)
        kaizo_layout(flags, reads)  # raises past WIDTH_CAP


# --- E4-A W2 (A-15 section 5, A-10 sections 2.7-2.9): the B = 4 read bound and the sampled mark ---

TAG = 1 << 7
B4_FORMATS = ("double", "double_two_trainer", "multi")


def test_fight_slot_reads_b4(kaizo_tables):
    """``fight_slot_reads(.., b4=True)`` is the most over the move rows and both target kinds
    (A-15 section 5). The B = 4 rows are ``[n, 2, 11]``, enemy then ally. Hand rows: a damaging
    Future Sight (effect 148) on an enemy reads 9 in Tag Strategy: before its block, TryScoreMinus's
    R@6666 or the SE roll R@6722 after CheckIfHighestDamageWithPartner's R@6694 (s:6658-6730), one
    each way, then the block's three rank calls of two reads each and R@7173 (s:7150-7193): 2 + 7.
    On an ally the partner part never reaches that block (s:6638); the row is keyed by effect, not
    type, so it holds the one roll an Electric / Water move of that effect may make in the absorb
    ladders (s:7343-7345, rolls at :7383-7438): 1. A Helping Hand (effect 176) on an ally
    reads 3 in TagStrategy_Partner (s:7582-7599), and CHECK_HP (bit 8) enters the same part again
    (s:7697), so bits 7 and 8 together make 6."""
    from nuzline.ai.layout import move_read_rows

    effect = np.asarray(kaizo_tables.move_ai_effect)
    future_sight = 1 + int(np.flatnonzero(effect[1:] == 148)[0])
    helping_hand = 1 + int(np.flatnonzero(effect[1:] == 176)[0])
    rows = move_read_rows(kaizo_tables, [future_sight, helping_hand], b4=True)
    assert rows.shape == (2, 2, 11)
    assert (rows[0, 0, 7], rows[0, 1, 7]) == (9, 1)
    assert (rows[1, 1, 7], rows[1, 1, 8]) == (3, 3)
    assert fight_slot_reads(rows, TAG, b4=True) == int(rows[..., 7].max()) == 9
    assert fight_slot_reads(rows[1:, 1:], TAG | 1 << 8, b4=True) == 6
    assert fight_slot_reads(rows, 0, b4=True) == 1  # no flag: at least 1


def test_b4_layout_key_and_allin4():
    """``AiLayout.key`` adds ``-b4`` for a B = 4 layout, a B = 2 layout's key is unchanged;
    ``ALLIN4`` is ALLIN with the B = 4 maximum."""
    from nuzline.ai.layout import ALLIN, ALLIN4, ALLIN_READS, ALLIN_READS_B4

    exact = kaizo_layout(0x97, 7)
    b4 = kaizo_layout(0x97, 7, b4=True)
    assert b4.key == exact.key + "-b4" and b4.b4 and not exact.b4
    assert ALLIN4.b4 and ALLIN4.slot_reads == ALLIN_READS_B4 >= ALLIN_READS
    assert ALLIN4.flags == ALLIN.flags and ALLIN4.runtime_flags and ALLIN4.key.endswith("-run-b4")


def test_beyond_cap_marks_or_refuses():
    """A-10 section 2.9: a B = 4 bound past WIDTH_CAP gives a sampled layout (N and base seed in the
    key); the refuse setting raises as B = 2 does; a B = 2 layout never samples; the seeds repeat."""
    from nuzline.ai.layout import BEYOND_CAP_REFUSE, SAMPLE_N, sample_seeds

    exact = kaizo_layout(0x97, 14, b4=True)  # 65597 scorings: the edge, still exact
    assert not exact.sampled and exact.sample_n == 0
    marked = kaizo_layout(0x97, 18, b4=True, sample_seed=5)
    assert marked.sampled and marked.sample_n == SAMPLE_N and marked.sample_seed == 5
    assert marked.key.endswith(f"-b4-s{SAMPLE_N}x5") and marked.slot_reads == 18
    with pytest.raises(ValueError, match="WIDTH_CAP"):
        kaizo_layout(0x97, 18, b4=True, beyond_cap=BEYOND_CAP_REFUSE)
    with pytest.raises(ValueError, match="WIDTH_CAP"):
        kaizo_layout(0x97, 18)
    with pytest.raises(ValueError, match="B = 4"):
        kaizo_layout(0x97, 7, beyond_cap="sample")
    a = sample_seeds(marked, 3)
    assert a.shape == (SAMPLE_N,) and (a == sample_seeds(marked, 3)).all()
    assert not (sample_seeds(marked, 4) == a).all()
    assert not (sample_seeds(kaizo_layout(0x97, 18, b4=True, sample_seed=6), 3) == a).all()
    with pytest.raises(ValueError):
        sample_seeds(exact, 0)


def _party_moves(trainer, inter):
    from nuzline.data.trainer import build_trainer_party

    return [int(m) for p in build_trainer_party(trainer, inter) for m in p.moves if int(m) > 0]


@pytest.mark.fullset
def test_b4_split_width(kaizo_tables, kaizo_trainers):
    """A-10 section 2.7 (ledger E4-16): the most, over every real split B = 4 fight (DOUBLES,
    TWO_TRAINERS, MULTI per partner variant) and every AI battler in it (its own flags with Tag
    Strategy, its party's moves, the whole fight's moves deciding the drawing rows), of
    ``fight_slot_reads(.., b4=True)`` is 14: 65597 scorings, half WIDTH_CAP, so every one is
    exact. A change that adds a read to any path fails this. The multi partner (battler 2) has the
    player's battler 0 as ``GetPartner`` (battle_system.c:394-411): its second
    ``CheckIfHighestDamageWithPartner`` sweep reads moves no layout knows, which may draw
    (trainer_ai.c:2414, :3014-3048), so it takes the drawn rows (``partner_unknown``); with them the
    maximum stays 14 (partner 612's fights reach it; _scratch/e4/width/partner_drawn_b4.py)."""
    from nuzline.ai.layout import BEYOND_CAP_REFUSE, KAIZO_FLAG_BITS, move_read_rows
    from nuzline.data.trainer import IntermediateData
    from tests.conftest import KAIZO_INTERMEDIATE

    inter = IntermediateData(KAIZO_INTERMEDIATE)
    mask = sum(1 << b for b in KAIZO_FLAG_BITS)
    by_id = {t.id: t for t in kaizo_trainers}
    fights = set()
    for t in kaizo_trainers:
        if t.format not in B4_FORMATS:
            continue
        if t.format == "double" or t.opponent_2 is None:
            enemies = (t.id,)
        else:
            enemies = tuple(sorted({t.id, t.opponent_2}))
        partners = [None]
        if t.format == "multi":
            partners = [p.trainer for p in (t.partner or ())] or [None]
        for p in partners:
            fights.add((t.format, enemies, p))
    cache = {}

    def moves_of(tid):
        if tid not in cache:
            cache[tid] = _party_moves(by_id[tid], inter)
        return cache[tid]

    best, in_split = 0, 0
    for _, enemies, partner in sorted(fights, key=repr):
        if not any(by_id[e].split for e in enemies):
            continue
        in_split += 1
        battlers = [*enemies, *([partner] if partner is not None else [])]
        pool = [m for tid in battlers for m in moves_of(tid)]
        for i, tid in enumerate(battlers):
            flags = (by_id[tid].ai_flags & mask) | TAG
            # the multi partner (battler 2, last): its GetPartner is the player's battler 0
            unknown = partner is not None and i == len(battlers) - 1
            rows = move_read_rows(
                kaizo_tables, moves_of(tid), b4=True, pool=pool, partner_unknown=unknown
            )
            reads = fight_slot_reads(rows, flags, b4=True)
            kaizo_layout(flags, reads, b4=True, beyond_cap=BEYOND_CAP_REFUSE)  # exact: no raise
            best = max(best, reads)
    assert in_split > 100  # the measurement's split fights (_scratch/e4/width/report.md)
    assert best == 14


def test_b4_partner_unknown_takes_drawn_rows(kaizo_tables, kaizo_names):
    """A battler whose ``GetPartner`` is the player's (MULTI battler 2, battle_system.c:394-411)
    estimates its partner's unknown moves in ``CheckIfHighestDamageWithPartner``'s second sweep
    (trainer_ai.c:2414), which may draw (Kaizo Triple Axel at Psywave's id, :3014-3015): its rows are
    those of a fight whose pool holds a drawing move that is not its own."""
    from nuzline.ai.layout import move_read_rows
    from nuzline.engine.damage import AI_PSYWAVE

    t, n = kaizo_tables, kaizo_names
    moves = [n.move(m) for m in ("Tackle", "Earthquake", "Swords Dance", "Protect")]
    plain = move_read_rows(t, moves, b4=True)
    unknown = move_read_rows(t, moves, b4=True, partner_unknown=True)
    drawn = move_read_rows(t, moves, b4=True, pool=[AI_PSYWAVE])
    assert np.array_equal(unknown, drawn)
    assert (unknown >= plain).all() and (unknown > plain).any()


def _copier_fight(tables, trainers, names, trainer_id, foe, fellow=()):
    """Trainer ``trainer_id``'s ``fight_tables`` with the player side's moves ``foe`` (names) and
    the ally's ``fellow`` (ids) in the copier pool: (tables, held moves, B = 4 layout builder)."""
    from nuzline.ai.layout import fight_layout, fight_tables
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from tests.conftest import KAIZO_INTERMEDIATE

    trainer = trainers[trainer_id]
    party = build_trainer_party(trainer, IntermediateData(KAIZO_INTERMEDIATE))
    t = fight_tables(
        tables,
        trainer,
        party,
        foe_moves=[names.move(n) for n in foe],
        fellow_moves=fellow,
    )
    held = [int(m) for m in np.asarray(t.ai_moves) if m > 0]
    return t, held, lambda **kw: fight_layout(t, trainer.ai_flags, held, b4=True, **kw)


def test_copier_pool_holds_the_fellow_battlers_moves(kaizo_tables, kaizo_trainers, kaizo_names):
    """A-10 section 2.8: at B = 4 a Transform / Mimic user's pool is every battler it can target,
    its ally's party moves included (``fight_tables(.., fellow_moves=..)``), each once, after its
    own; a party without a copier gets no pool."""
    from nuzline.ai.layout import copies_moves, fight_tables

    n = kaizo_names
    _, held, _ = _copier_fight(
        kaizo_tables, kaizo_trainers, n, 455, ["Future Sight"], fellow=[n.move("Surf"), 0]
    )
    assert copies_moves(kaizo_tables, held)
    assert n.move("Future Sight") in held and held.count(n.move("Surf")) == 1
    plain = next(t for t in kaizo_trainers if t.format == "single")
    t = fight_tables(
        kaizo_tables, plain, [], foe_moves=[n.move("Surf")], fellow_moves=[n.move("Surf")]
    )
    assert n.move("Surf") not in [int(m) for m in np.asarray(t.ai_moves)]


@pytest.mark.fullset
def test_copier_ditto_future_sight_layout(kaizo_tables, kaizo_trainers, kaizo_names):
    """A-10 section 2.10 item 5, layout part (the scoring part is W2's other builder): trainer 372's
    Ditto (Transform) against a player Future Sight keeps an exact B = 4 layout, bound <= 14. Real
    trainer rows by ``KAIZO_TRANSCRIBED``: an effect that becomes transcribed adds reads (``fullset``)."""
    _, _, layout = _copier_fight(kaizo_tables, kaizo_trainers, kaizo_names, 372, ["Future Sight"])
    lay = layout()
    assert not lay.sampled and lay.slot_reads <= 14 and lay.b4


@pytest.mark.fullset
def test_copier_mr_mime_future_sight_layout(kaizo_tables, kaizo_trainers, kaizo_names):
    """A-10 section 2.10 item 6, layout part: trainer 455's Mr. Mime (Mimic) against a player
    Future Sight is exact, bound <= 14. ``fullset`` as the Ditto case."""
    _, _, layout = _copier_fight(kaizo_tables, kaizo_trainers, kaizo_names, 455, ["Future Sight"])
    lay = layout()
    assert not lay.sampled and lay.slot_reads <= 14 and lay.b4


def test_copier_beyond_cap_sampled(kaizo_tables, kaizo_trainers, kaizo_names):
    """A-10 section 2.10 item 7: trainer 455 (flags 0x7ff) against a player Sand Attack: a Mimicked
    Sand Attack reads past the cap (the sheet measured 18). The default layout is sampled and
    marked with its N and base seed; the refuse setting raises."""
    from nuzline.ai.layout import BEYOND_CAP_REFUSE, SAMPLE_N

    _, _, layout = _copier_fight(kaizo_tables, kaizo_trainers, kaizo_names, 455, ["Sand-Attack"])
    lay = layout(sample_seed=455)
    assert lay.sampled and lay.slot_reads >= 15 and lay.b4
    assert (lay.sample_n, lay.sample_seed) == (SAMPLE_N, 455)
    with pytest.raises(ValueError, match="WIDTH_CAP"):
        layout(beyond_cap=BEYOND_CAP_REFUSE)


def test_exact_path_refuses_a_sampled_layout():
    """A sampled B = 4 layout (bound past WIDTH_CAP, A-10 section 2.9) has no exact distribution:
    ``policy._exact_layout`` raises before ``4 * 2**slot_reads`` scorings are built; an exact B = 4
    layout passes."""
    sampled = kaizo_layout(0x87, 18, b4=True)
    assert sampled.sampled
    with pytest.raises(ValueError, match="sampled"):
        policy._exact_layout(sampled)
    exact = kaizo_layout(0x87, 14, b4=True)
    assert not exact.sampled and policy._exact_layout(exact) is exact
