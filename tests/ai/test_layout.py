"""The per-game AI policy layer (K3c Task 2): ``nuzline.ai.layout`` and the corrections
that need no new script label.

Decomp paths are relative to ``external/pokeplatinum``: ``trainer_ai.c`` is
``src/battle/trainer_ai/trainer_ai.c``, ``script.s`` is ``src/battle/trainer_ai/script.s``.
"""

import jax
import jax.numpy as jnp
import pytest

from nuzline.ai import commands, policy
from nuzline.ai.layout import KAIZO, VANILLA, AiLayout, kaizo_layout, layout_of
from nuzline.ai.policy import ai_policy, ai_refusals
from nuzline.ai.switching import REFUSED_ACTION
from nuzline.engine.state import make_state
from tests.ai.fixtures import GEODUDE, outcome, trace_a, versus

SELFDESTRUCT, HEAD_SMASH, CRUNCH = 120, 457, 242
TRIPLE_AXEL, RETURN = 149, 216  # Kaizo names of the decomp's MOVE_PSYWAVE and MOVE_RETURN
WATER_GUN = 55


def _kaizo_mon(party, species: str, names):
    """The Kaizo Roark party member of ``species``."""
    return next(m for m in party if m.species == names.species(species))


def test_layout_of_reads_tables(tables, kaizo_tables):
    assert isinstance(VANILLA, AiLayout) and isinstance(KAIZO, AiLayout)
    assert layout_of(tables) is VANILLA
    assert layout_of(kaizo_tables) is KAIZO
    assert int(tables.ai_policy) == 0 and int(kaizo_tables.ai_policy) == 1
    with pytest.raises(TypeError, match="layout_of"):
        jax.jit(lambda t: layout_of(t).water2_ability)(tables)


def test_layout_per_flag_set(tables, kaizo_tables):
    """E1 Ruling R2: the Kaizo layout is the fight's trainer's flag set (``Tables.ai_flags``),
    refused while unset; vanilla keeps its one set whatever the flags. Each set is its own
    static layout with its own program-cache key."""
    from nuzline.data.tables import AI_FLAGS_UNSET

    assert layout_of(kaizo_tables) is kaizo_layout(0b10111) == KAIZO
    for flags in (0x1, 0x5, 0x7, 0x17):
        layout = layout_of(kaizo_tables._replace(ai_flags=jnp.int32(flags)))
        assert layout == KAIZO._replace(flags=flags) and layout.key == f"kaizo-{flags:#x}"
    assert len({kaizo_layout(f).key for f in (0x1, 0x5, 0x7, 0x17)} | {VANILLA.key}) == 5
    with pytest.raises(ValueError, match="not set"):
        layout_of(kaizo_tables._replace(ai_flags=AI_FLAGS_UNSET))
    with pytest.raises(ValueError, match="transcribes only"):
        kaizo_layout(0xF)  # SETUP_FIRST_TURN, not transcribed yet
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
    Kaizo. Vanilla compares Levitate again there, which never matches."""
    s = _dry_skin_state(tables, names)
    ctx = commands.at_slot(commands.new_context(tables, s, outcome()), 0)
    scores = {
        layout.name: int(policy._basic(ctx, layout).scores[0]) - 100 for layout in (VANILLA, KAIZO)
    }
    assert scores == {"vanilla": 0, "kaizo": -12}


def test_kaizo_refuses_alt_power_and_id_cases(kaizo_tables, kaizo_names, kaizo_roark_party):
    """Kaizo Triple Axel (149) has effect 88, in sAltPowerMoveEffects (trainer_ai.c:48-61),
    and its id is MOVE_PSYWAVE's, a TrainerAI_CalcDamage id case (:3014). Kaizo Return
    (216) has effect 0 but still hits the MOVE_RETURN case (:3018)."""
    t, n = kaizo_tables, kaizo_names
    cranidos = _kaizo_mon(kaizo_roark_party, "Cranidos", n)
    gible = _kaizo_mon(kaizo_roark_party, "Gible", n)
    s = make_state(t, [gible], [cranidos])
    axel = s._replace(moves=s.moves.at[1, 0, 3].set(TRIPLE_AXEL))
    reasons = ai_refusals(t, axel, n)
    assert any("Triple Axel" in r and "sAltPowerMoveEffects" in r for r in reasons), reasons
    assert any("Triple Axel" in r and "TrainerAI_CalcDamage" in r for r in reasons), reasons
    ret = s._replace(moves=s.moves.at[1, 0, 3].set(RETURN))
    assert int(t.move_ai_effect[RETURN]) == 0
    reasons = ai_refusals(t, ret, n)
    assert any("Return" in r and "TrainerAI_CalcDamage" in r for r in reasons), reasons
    assert not any("sAltPowerMoveEffects" in r for r in reasons), reasons


def test_move_type_id_case_refused(tables, names):
    """TrainerAI_MoveType's id cases (trainer_ai.c:3132-3222): Weather Ball's AI type is not
    its table type."""
    s = trace_a(tables, names)
    s = s._replace(moves=s.moves.at[1, GEODUDE, 2].set(names.move("Weather Ball")))
    reasons = ai_refusals(tables, s)
    assert any("Weather Ball" in r and "TrainerAI_MoveType" in r for r in reasons), reasons


def test_traced_tables_without_a_layout_refuse_another_policy(tables, names):
    """Under a trace ``layout_of`` cannot read ``tables``: a caller that passes no layout
    gets the vanilla one, and a game of another policy is refused, not misread."""
    s = trace_a(tables, names)
    kaizo_like = tables._replace(
        ai_policy=jnp.asarray(1, tables.ai_policy.dtype), ai_flags=jnp.int32(KAIZO.flags)
    )
    # Concrete tables resolve their own layout: the Kaizo policy's outcome is per slot.
    per_slot = outcome()._replace(
        script_rolls=jnp.full((4, KAIZO.slot_reads), 254, jnp.uint8),
    )
    assert int(jax.jit(lambda st: ai_policy(kaizo_like, st, per_slot))(s)) == 0
    traced = jax.jit(lambda t: ai_policy(t, s, outcome()))
    assert int(traced(tables)) == 0
    assert int(traced(kaizo_like)) == REFUSED_ACTION
