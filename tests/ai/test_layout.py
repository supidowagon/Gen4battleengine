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
from nuzline.ai.layout import KAIZO, VANILLA, AiLayout, kaizo_layout, layout_of
from nuzline.ai.policy import ai_policy, ai_refusals
from nuzline.ai.switching import REFUSED_ACTION
from nuzline.engine.state import make_state
from tests.ai.fixtures import GEODUDE, outcome, trace_a, versus
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
    damage = commands._slot_damage(t, s, jnp.full((4,), commands.SIM_ROLL_MAX))
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
    return s._replace(
        moves=s.moves.at[1, GEODUDE, 0].set(WATER_GUN),
        pp=s.pp.at[1, GEODUDE, 0].set(25),
        ability_slot=s.ability_slot.at[0, 0].set(dry_skin),
        ability=s.ability.at[0, 0].set(tables.species_abilities[croagunk, dry_skin]),
        ai_ability_known=s.ai_ability_known.at[0].set(1),
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
    per_slot = outcome()._replace(script_rolls=jnp.full((4, reads), 254, jnp.uint8))
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


def test_every_kaizo_fight_fits_the_width_cap(kaizo_tables, kaizo_trainers):
    """E1 Ruling R6: every Kaizo trainer's fight, under the transcribed bits of its own AI
    flags, has a read bound whose layout passes WIDTH_CAP (``kaizo_layout`` raises otherwise).
    Synthetic flag sets can pass it (every flag with ACC_DOWN bounds at 17); no trainer does
    (the widest, Jason 82 and Danny 725, bound at 14)."""
    from nuzline.ai.layout import KAIZO_FLAG_BITS, fight_slot_reads, move_read_rows
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from tests.conftest import KAIZO_INTERMEDIATE

    inter = IntermediateData(KAIZO_INTERMEDIATE)
    mask = sum(1 << b for b in KAIZO_FLAG_BITS)
    widest = 0
    for trainer in kaizo_trainers:
        party = build_trainer_party(trainer, inter)
        moves = np.array([m for p in party for m in p.moves], np.int32)
        flags = trainer.ai_flags & mask
        reads = fight_slot_reads(move_read_rows(kaizo_tables, moves), flags)
        kaizo_layout(flags, reads)  # raises past WIDTH_CAP
        widest = max(widest, reads)
    assert widest == 14
