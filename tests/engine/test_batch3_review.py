"""E2 batch 3 review follow-ups (rulings B3-18 .. B3-22, docs/kaizo/e2-mechanics-ledger.md):
the battle-start switch-in checks over both leads (``make_state``, which only the JAX engine
runs), the turn-start laws of a Trace newcomer, and the per-candidate refusals in prep."""

import numpy as np
import pytest

from nuzline.data.spec import PokemonSpec
from nuzline.engine.coverage import check_coverage
from nuzline.engine.enumerate import outcome_laws
from nuzline.engine.ids import AbilityId, Stat, Weather
from nuzline.engine.order import MOVE_ACTIONS
from nuzline.engine.outcome import hit_threshold
from nuzline.engine.state import make_state
from nuzline.prep.pool import Candidate, check_pool
from nuzline.prep.problem import usable_pool
from nuzline.prep.states import make_states
from tests.engine.helpers import roark_party, roark_trainer


def _mon(names, species: str, moves=("Tackle",), slot: int = 0) -> PokemonSpec:
    ids = [names.move(m) for m in moves]
    return PokemonSpec(
        species=names.species(species),
        level=20,
        nature=0,
        ivs=(15,) * 6,
        evs=(0,) * 6,
        moves=tuple(ids + [0] * (4 - len(ids))),
        item=0,
        ability_slot=slot,
        gender=0,
        friendship=70,
    )


def test_a_traced_intimidate_fires_once_at_battle_start(tables, names):
    """B3-18: an Intimidate lead against a Trace lead. Every lead traces before any Intimidate
    runs (battle_lib.c:3718-3737 before :3795-3808), and each battler's Intimidate is announced
    once: both sides end at -1 Attack, in make_state and in prep's batched make_states."""
    player, opp = [_mon(names, "Growlithe")], [_mon(names, "Porygon")]
    s = make_state(tables, player, opp)
    assert int(s.ability[1, 0]) == AbilityId.INTIMIDATE
    assert s.stages[:, Stat.ATK].tolist() == [-1, -1]
    batch = make_states(tables, [player], opp)
    for name, x, y in zip(s._fields, s, batch, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y)[0], err_msg=name)


def test_a_trace_newcomer_holds_the_copy_at_the_foes_accuracy_check(tables, names):
    """B3-19: a Trace Pokemon switching in against a Sand Veil attacker in sand copies Sand
    Veil before the attacker's check, which is then x80/100 (bcp.c:2927-2928)."""
    player = [_mon(names, "Sandshrew")]
    opp = [_mon(names, "Geodude"), _mon(names, "Porygon")]
    s = make_state(tables, player, opp, start_condition=Weather.SANDSTORM)
    assert int(s.ability[0, 0]) == AbilityId.SAND_VEIL
    accuracy = int(tables.move_accuracy[names.move("Tackle")])
    actions = np.array([0, MOVE_ACTIONS + 1], np.int32)
    assert int(hit_threshold(tables, s, actions)[0]) == accuracy * 80 // 100


def test_a_trace_newcomer_copying_static_makes_the_contact_roll_relevant(tables, names):
    """B3-19: a Trace Pokemon switching in against a Static contact attacker copies Static, so
    the attacker's contact hit rolls it (contact_proc on the newcomer's side is read)."""
    player = [_mon(names, "Pikachu")]
    opp = [_mon(names, "Geodude"), _mon(names, "Porygon")]
    s = make_state(tables, player, opp)
    assert int(s.ability[0, 0]) == AbilityId.STATIC
    laws = outcome_laws(tables, s, np.array([0, MOVE_ACTIONS + 1], np.int32))
    assert bool(laws.relevant.contact_proc[1, 0])


def _candidate(spec: PokemonSpec, slot: int = 0) -> Candidate:
    return Candidate(("party", 0, slot), f"party slot {slot}", spec, 0, (), ())


@pytest.mark.fullset
def test_prep_keeps_a_weather_war_and_refuses_a_trace_copy_the_ai_refuses(tables, names):
    """B3-20, B3-21, G-26: a Drought candidate against a Drizzle opponent is kept (the weather
    war is modelled); a Trace candidate is kept unless a copy it can make is one the AI refuses
    on the player's side (Normalize, policy ``player_ability_refusal``), in the usable pool and the
    pool check alike."""
    opp = [_mon(names, "Kyogre")]
    trainer = roark_trainer()
    groudon, porygon = _candidate(_mon(names, "Groudon")), _candidate(_mon(names, "Porygon"), 1)
    kept, lines = usable_pool(tables, names, [groudon, porygon], opp, trainer)
    assert groudon in kept and porygon in kept
    assert not any("weather" in line for line in lines)
    report = check_pool(tables, names, [groudon, porygon], opp, trainer)
    assert not any("weather" in m for m in report.missing)
    assert not any("Trace" in m for m in report.missing)
    normalize_opp = [_mon(names, "Skitty", slot=1)]
    kept, lines = usable_pool(tables, names, [porygon], normalize_opp, trainer)
    assert kept == []
    assert any("party slot 1" in line and "Trace copying Normalize" in line for line in lines)
    report = check_pool(tables, names, [porygon], normalize_opp, trainer)
    assert any("party slot 1" in m and "Trace copying Normalize" in m for m in report.missing)


@pytest.mark.fullset
def test_coverage_reads_the_trace_copy_for_the_ai(tables, names):
    """B3-21: the AI reads the copy (LoadBattlerAbility, trainer_ai.c:1170-1209), so a Trace
    holder is refused only for a copy the AI refuses on the holder's side: Normalize copied by
    the player's Porygon (refused on either side), Tinted Lens
    copied by the trainer's (an attacker ability, refused on the AI's side only)."""
    trainer = roark_trainer()

    def trace_lines(player, opp):
        r = check_coverage(tables, names, player, trainer, opp, 0)
        return [m for m in r.missing if "Trace copying" in m]

    porygon = _mon(names, "Porygon")
    assert trace_lines([porygon], roark_party(names)) == []
    assert trace_lines([_mon(names, "Geodude")], [porygon]) == []
    # Copies of a Pokemon behind the lead (the lead's own copy is in the turn-0 state already,
    # whose refusals list it once).
    geodude = _mon(names, "Geodude")
    lines = trace_lines([porygon], [geodude, _mon(names, "Skitty", slot=1)])
    assert lines and all("Trace copying Normalize" in m and "side 0" in m for m in lines)
    lines = trace_lines([geodude, _mon(names, "Venomoth", slot=1)], [porygon])
    assert lines and all("Trace copying Tinted Lens" in m and "side 1" in m for m in lines)
    r = check_coverage(tables, names, [porygon], trainer, [_mon(names, "Skitty", slot=1)], 0)
    assert sum("Trace copying Normalize" in m for m in r.missing) == 1


def test_a_trace_newcomer_copying_battle_armor_is_never_crit(tables, names):
    """B3-19: the crit law sees the copy too (outcome.armored): a Trace Pokemon switching in
    against a Battle Armor attacker cannot be crit by it (battle_lib.c:7135-7136)."""
    from nuzline.engine.outcome import armored

    player = [_mon(names, "Kabuto", slot=1)]
    opp = [_mon(names, "Geodude"), _mon(names, "Porygon")]
    s = make_state(tables, player, opp)
    assert int(s.ability[0, 0]) == AbilityId.BATTLE_ARMOR
    assert bool(armored(tables, s, np.array([0, MOVE_ACTIONS + 1], np.int32))[0])
