"""The hand-traced AI decisions of docs/ai/roark.md section 13 (Traces A-G).

Each entry of ``TRACED`` is ``(description, build, outcome, expected_scores,
expected_action)``: ``build(tables, names)`` gives the state, ``outcome`` the AI's
random draws, ``expected_scores`` what ``ai_scores`` returns and ``expected_action``
what ``ai_policy`` returns, both from the trace tables in roark.md.

The states are Roark's real party and movesets (``roark_real_state``: Geodude, Onix,
Cranidos built from the trainer data, bag ``[17, 17]``) against the sample party with
the named Pokemon as the lead. The traces set the opponent's active slot, HP, turn and
``move_hit`` directly; "first turn" means ``fake_out_turn`` 0 >= ``turn`` 0 (roark.md
5.3). Unreached script-roll sites are given as 0 (roark.md 13).
"""

from __future__ import annotations

import jax.numpy as jnp

from nuzline.ai.outcome import SWITCH_NONE, AiOutcome, no_calc_draws
from nuzline.engine.state import BattleState
from tests.engine.helpers import roark_real_state

GEODUDE, ONIX, CRANIDOS = 0, 1, 2  # Roark's party slots (roark.md section 4)


def outcome(rolls=(0, 0, 0, 0), tiebreak=0, ability_guess=0, switch_stop=SWITCH_NONE):
    """An ``AiOutcome`` with ``sim_roll`` fixed at roll index 15 (100 %, roark.md 8.1)."""
    i8 = jnp.int8
    return AiOutcome(
        sim_roll=jnp.full((4,), 15, i8),
        ability_guess=jnp.asarray(ability_guess, i8),
        script_rolls=jnp.asarray(rolls, i8),
        tiebreak=jnp.asarray(tiebreak, i8),
        switch_stop=jnp.asarray(switch_stop, i8),
        calc_draws=no_calc_draws(),
    )


def versus(
    tables,
    names,
    opp_slot: int,
    player: str | None = None,
    *,
    opp_hp: int | None = None,
    player_hp: int | None = None,
    turn: int = 0,
) -> BattleState:
    """Roark's ``opp_slot`` against the sample party led by ``player``."""
    s = roark_real_state(tables, names, player_species=player)
    s = s._replace(active=s.active.at[1].set(opp_slot), turn=jnp.asarray(turn, s.turn.dtype))
    if opp_hp is not None:
        s = s.replace_hp(1, opp_slot, opp_hp)
    if player_hp is not None:
        s = s.replace_hp(0, 0, player_hp)
    return s


def trace_a(tables, names):
    """Geodude (32/32) vs Monferno (44/44), turn 1."""
    return versus(tables, names, GEODUDE)


def trace_a4(tables, names):
    s = trace_a(tables, names)
    return s._replace(stealth_rock=s.stealth_rock.at[0].set(1))


def trace_b(tables, names):
    """Onix (31/31) vs Starly (30/30), turn 1."""
    return versus(tables, names, ONIX, "Starly")


def trace_c(tables, names):
    """Cranidos (43/43) vs Starly (30/30), turn 1."""
    return versus(tables, names, CRANIDOS, "Starly")


def trace_d(tables, names):
    """Cranidos (30/43) vs Monferno (30/44), turn 1."""
    return versus(tables, names, CRANIDOS, opp_hp=30, player_hp=30)


def trace_g(tables, names):
    """Onix (31/31) vs Voltorb (L14, default moves, true ability Static, unrevealed)."""
    s = versus(tables, names, ONIX, "Voltorb")
    voltorb = names.species("Voltorb")
    static = 1  # Voltorb's species ability slot of Static (Soundproof, Static)
    assert names.species_abilities(voltorb)[static] == names.ability("Static")
    return s._replace(
        ability_slot=s.ability_slot.at[0, 0].set(static),
        ability=s.ability.at[0, 0].set(tables.species_abilities[voltorb, static]),
    )


def trace_g4(tables, names):
    s = trace_g(tables, names)
    return s._replace(ai_ability_known=s.ai_ability_known.at[0].set(1))


def trace_e(geodude_hp: int, onix_fainted: bool = False):
    """Geodude (``geodude_hp``/32) vs Monferno (32/44) on turn 2, moveHit none."""

    def build(tables, names):
        s = versus(tables, names, GEODUDE, opp_hp=geodude_hp, player_hp=32, turn=1)
        return s.replace_hp(1, ONIX, 0) if onix_fainted else s

    return build


def trace_f(tables, names):
    """Cranidos (40/43, the lead in party slot 2) vs Starly (30/30), turn 2,
    moveHit[Cranidos] = Wing Attack."""
    s = versus(tables, names, CRANIDOS, "Starly", opp_hp=40, turn=1)
    return s._replace(move_hit=s.move_hit.at[1].set(names.move("Wing Attack")))


TRACED = [
    ("A1", trace_a, outcome(), [101, 100, 0, 0], 0),
    ("A2", trace_a, outcome((0, 0, 1, 0), tiebreak=0), [100, 100, 0, 0], 0),
    ("A3", trace_a, outcome((0, 0, 1, 0), tiebreak=1), [100, 100, 0, 0], 1),
    ("A4", trace_a4, outcome(), [91, 100, 0, 0], 1),
    ("B1", trace_b, outcome(), [101, 100, 100, 0], 0),
    ("B2", trace_b, outcome((0, 0, 1, 0), tiebreak=0), [100, 100, 100, 0], 0),
    ("B3", trace_b, outcome((0, 0, 1, 0), tiebreak=1), [100, 100, 100, 0], 1),
    ("B4", trace_b, outcome((0, 0, 1, 0), tiebreak=5), [100, 100, 100, 0], 2),
    ("B keep draw", trace_b, outcome(switch_stop=0), [101, 100, 100, 0], 0),
    ("C1", trace_c, outcome(), [104, 100, 100, 0], 0),
    ("C2", trace_c, outcome((0, 1, 0, 0)), [104, 99, 100, 0], 0),
    ("D1", trace_d, outcome(tiebreak=0), [100, 100, 96, 0], 0),
    ("D2", trace_d, outcome(tiebreak=1), [100, 100, 96, 0], 1),
    ("D3", trace_d, outcome((0, 1, 0, 1)), [100, 99, 98, 0], 0),
    ("D4", trace_d, outcome((0, 1, 0, 0)), [100, 99, 96, 0], 0),
    ("G1", trace_g, outcome(ability_guess=0), [101, 100, 90, 0], 0),
    ("G2", trace_g, outcome((0, 0, 1, 0), tiebreak=1, ability_guess=0), [100, 100, 90, 0], 1),
    ("G3", trace_g, outcome((0, 0, 1, 0), tiebreak=2, ability_guess=1), [100, 100, 100, 0], 2),
    ("G4", trace_g4, outcome(ability_guess=0), [101, 100, 100, 0], 0),
    ("E1", trace_e(11), outcome(), [101, 100, 0, 0], 10),
    ("E2", trace_e(11, onix_fainted=True), outcome(), [101, 100, 0, 0], 10),
    ("E3", trace_e(12), outcome(), [101, 100, 0, 0], 0),
    ("E4", trace_e(7), outcome(), [101, 100, 0, 0], 10),
    ("F switch to Geodude", trace_f, outcome(switch_stop=0), [104, 99, 100, 0], 4),
    ("F switch to Geodude (2nd)", trace_f, outcome(switch_stop=1), [104, 99, 100, 0], 4),
    ("F switch to Onix", trace_f, outcome(switch_stop=2), [104, 99, 100, 0], 5),
    ("F no switch", trace_f, outcome(), [104, 99, 100, 0], 0),
]
