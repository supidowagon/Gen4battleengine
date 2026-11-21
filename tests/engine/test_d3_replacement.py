"""D3-12: the turn-end replacement round over four battlers (``ReplaceFainted``, bcp.c:4070-4167;
``replace_round``), and the owed reload of battler 2's level-up (battle_script.c:9716-9720,
:9860-9864; ``levelup._reloaders``).

``four`` states: every Pokemon is a level-50 Bulbasaur; the battlers lead from the first slots of
their parties (DOUBLES: battler 0 slot 0, battler 2 slot 1). ``down`` sets Pokemon to 0 HP."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.actions import absent_now, replacement_bench
from nuzline.engine.damage import stealth_rock_loss
from nuzline.engine.format import DOUBLES, MULTI, SINGLES, TWO_TRAINERS, format_of, party
from nuzline.engine.levelup import pay_exp
from nuzline.engine.step import needs_replacement, replace_round
from nuzline.reference.convert import to_ref
from nuzline.reference.engine import _awaits_replacement
from tests.engine.d2_states import four
from tests.strategies import supported_state


def down(s, *cells):
    """``s`` with the party cells ``(party, slot)`` at 0 HP."""
    hp = s.hp
    for p, j in cells:
        hp = hp.at[p, j].set(0)
    return s._replace(hp=hp)


def picks(*values):
    return np.asarray(values, np.int8)


def flags(s) -> list[int]:
    return np.asarray(needs_replacement(s)).tolist()


def bench(s, b) -> list[int]:
    return [int(x) for x in np.asarray(replacement_bench(s, b))]


def cell(s, b) -> tuple[int, int]:
    """Battler ``b``'s ``(party, active slot)``."""
    return party(format_of(s), b), int(s.active[b])


# ---------------------------------------------------------------- DOUBLES: low first


def test_doubles_both_down_serves_battler_0_first(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 0), cell(s, 2), (0, 4), (0, 5))  # slots 2 and 3 are left
    assert flags(s) == [1, 0, 0, 0]  # bcp.c:4083-4086: battler 2 waits for the next entry
    out = replace_round(tables, s, picks(2, -1, -1, -1))
    assert int(out.active[0]) == 2
    assert flags(out) == [0, 0, 1, 0]
    assert bench(out, 2) == [0, 0, 0, 1, 0, 0]  # battler 0's newcomer is on the field
    assert int(out.entry_pending[0]) == 1  # its switch-in states wait for ShowBattleMon (D3-13)
    out2 = replace_round(tables, out, picks(-1, -1, 3, -1))
    assert flags(out2) == [0, 0, 0, 0] and int(out2.active[2]) == 3


def test_doubles_one_pokemon_left_leaves_battler_2_absent(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 0), cell(s, 2), (0, 3), (0, 4), (0, 5))  # only slot 2 is left
    assert flags(s) == [1, 0, 0, 0]
    assert not bool(absent_now(s, 2))  # its bench still holds slot 2
    out = replace_round(tables, s, picks(2, -1, -1, -1))
    assert flags(out) == [0, 0, 0, 0]
    assert bool(absent_now(out, 2)) and bool(out.over == 0)


def test_a_pick_outside_the_bench_is_refused(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 0), cell(s, 2))
    with pytest.raises(ValueError):
        replace_round(tables, s, picks(1, -1, -1, -1))  # battler 2's active slot (display.c:4481)
    with pytest.raises(ValueError):
        replace_round(tables, s, picks(-1, -1, 2, -1))  # battler 2 is not flagged this round
    with pytest.raises(ValueError):
        replace_round(tables, s, picks(0, -1, -1, -1))  # its own fainted slot


def test_replace_round_is_the_four_battler_api(tables, names):
    s = down(four(tables, names, SINGLES), (0, 0))
    with pytest.raises(ValueError):
        replace_round(tables, s, picks(1, -1))


# ---------------------------------------------------------------- MULTI, TWO_TRAINERS


def test_multi_flags_both_with_their_own_benches(tables, names):
    s = four(tables, names, MULTI)
    s = down(s, cell(s, 0), cell(s, 2), (0, 4), (2, 3))
    assert flags(s) == [1, 0, 1, 0]  # bcp.c:4118-4140: each looks at its own party
    out = replace_round(tables, s, picks(3, -1, 4, -1))
    assert int(out.active[0]) == 3 and int(out.active[2]) == 4
    assert flags(out) == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        replace_round(tables, s, picks(4, -1, -1, -1))  # party 0's slot 4 is down


def test_two_trainers_enemy_battlers_1_and_3_together(tables, names):
    s = four(tables, names, TWO_TRAINERS)
    s = down(s, cell(s, 1), cell(s, 3))  # parties 1 and 2
    assert flags(s) == [0, 1, 0, 1]  # bcp.c:4118-4140
    out = replace_round(tables, s, picks(-1, 2, -1, 3))
    assert int(out.active[1]) == 2 and int(out.active[3]) == 3
    assert flags(out) == [0, 0, 0, 0]


def test_two_trainers_player_battlers_low_first(tables, names):
    s = four(tables, names, TWO_TRAINERS)
    s = down(s, cell(s, 0), cell(s, 2))  # the shared party 0
    assert flags(s) == [1, 0, 0, 0]
    out = replace_round(tables, s, picks(2, -1, -1, -1))
    assert flags(out) == [0, 0, 1, 0]


# ---------------------------------------------------------------- send-in order, payout timing


def test_payout_waits_for_the_round_and_hazards_read_the_pre_level_max_hp(tables, names):
    """Battler 1's newcomer (party 1 slot 2, 1 HP) faints on Stealth Rock; battler 2's newcomer
    (party 0 slot 2: an Exp. Share holder one exp point below level 54) enters after it in the
    same round. Its Stealth Rock loss divides the pre-level max HP; the level-up comes after
    (subscript_show_party_list.s:9-27, then TurnEnd's AnyExpPayout, bcp.c:1846)."""
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 1), cell(s, 2))  # battlers 1 and 2; battlers 3 and 0 stand
    table = tables.exp_table[tables.species_exp_rate[int(s.species[0, 2])]]
    s = s._replace(
        hp=s.hp.at[1, 2].set(1).at[0, 2].set(118),
        level=s.level.at[0, 2].set(53),
        max_hp=s.max_hp.at[0, 2].set(118),  # 14 = 118 // 8; level 54's 120 would give 15
        exp=s.exp.at[2].set(int(table[54]) - 1),
        item=s.item.at[0, 2].set(names.item("Exp. Share")),
        stealth_rock=jnp.ones_like(s.stealth_rock),
    )
    assert flags(s) == [0, 1, 1, 0]
    loss = int(stealth_rock_loss(tables, s, 2, 2))
    out = replace_round(tables, s, picks(-1, 2, 2, -1))
    assert int(out.hp[1, 2]) == 0  # the first newcomer fainted on the rocks
    assert int(out.level[0, 2]) == 54  # its payout levelled the holder after the round
    assert loss != int(stealth_rock_loss(tables, out, 2, 2))  # the max HP did grow into 15
    assert int(out.hp[0, 2]) == int(out.max_hp[0, 2]) - loss  # the pre-level 1/8 was taken
    assert flags(out) == [0, 1, 0, 0]  # battler 1 is flagged again (its bench is left)


def test_a_hazard_faint_flags_the_battler_again(tables, names):
    s = four(tables, names, DOUBLES)
    s = down(s, cell(s, 0))
    s = s._replace(hp=s.hp.at[0, 2].set(1), spikes=s.spikes.at[0].set(3))
    assert flags(s) == [1, 0, 0, 0]
    out = replace_round(tables, s, picks(2, -1, -1, -1))
    assert int(out.hp[0, 2]) == 0 and int(out.player_faints) == 1
    assert int(out.entry_pending[0]) == 0  # no newcomer waits
    assert flags(out) == [1, 0, 0, 0]  # battler 0 again: slots 3-5 are left


# ---------------------------------------------------------------- B = 2


def _singles_needs_replacement_before_d3_12(state):
    """The B = 2 body of ``needs_replacement`` before D3-12."""
    fmt = format_of(state)
    slots = jnp.arange(state.hp.shape[1])
    active_down = state.hp[party(fmt, jnp.arange(fmt.B)), state.active] == 0
    rows = party(fmt, jnp.arange(fmt.B))
    bench_ = (
        (state.species[rows] != 0)
        & (state.hp[rows] > 0)
        & (slots[None, :] != state.active[:, None])
    )
    return (active_down & jnp.any(bench_, axis=1) & (state.over == 0)).astype(jnp.int8)


@settings(max_examples=20, deadline=None, suppress_health_check=list(HealthCheck))
@given(data=st.data())
def test_singles_needs_replacement_is_unchanged(tables, names, data):
    s = data.draw(supported_state(tables, names))
    for side in (0, 1):
        if data.draw(st.booleans()):
            s = s._replace(hp=s.hp.at[side, int(s.active[side])].set(0))
    np.testing.assert_array_equal(
        np.asarray(needs_replacement(s)), np.asarray(_singles_needs_replacement_before_d3_12(s))
    )
    ref = to_ref(s)
    for b in range(2):
        assert bool(needs_replacement(s)[b]) == (not ref.over and _awaits_replacement(ref, b))


# ---------------------------------------------------------------- owed from D3-8b: battler 2's reload


def leveling(tables, names, fmt=DOUBLES):
    """``(before, after)``: party 0 slots 0 and 1 (battlers 0 and 2) are one exp point below
    level 51 and the foe of battler 1 faints in ``after``."""
    s = four(tables, names, fmt)
    table = tables.exp_table[tables.species_exp_rate[int(s.species[0, 0])]]
    s = s._replace(exp=s.exp.at[0].set(int(table[51]) - 1).at[1].set(int(table[51]) - 1))
    return s, s._replace(hp=s.hp.at[1, 0].set(0))


def reloading(tables, names):
    """``leveling`` with Power Trick on battlers 0 and 2 and battler 2 Transformed: its
    ``shadow_*`` hold party slot 1's own species, form and IVs (G-69, ``party_identity``)."""
    s, after = leveling(tables, names)
    s = s._replace(
        power_trick=s.power_trick.at[0].set(1).at[2].set(1),
        transformed=s.transformed.at[2].set(1),
        shadow_species=s.shadow_species.at[2].set(s.species[0, 1]),
        shadow_form=s.shadow_form.at[2].set(s.form[0, 1]),
        shadow_ivs=s.shadow_ivs.at[2].set(s.ivs[0, 1]),
    )
    keep = ("power_trick", "transformed", "shadow_species", "shadow_form", "shadow_ivs")
    return s, after._replace(**{k: getattr(s, k) for k in keep})


def test_battler_2s_level_up_reloads_battler_2(tables, names):
    s, after = reloading(tables, names)
    out = pay_exp(tables, s, after, battler=1)
    assert int(out.level[0, 0]) == 51 and int(out.level[0, 1]) == 51
    # ReloadPokemon for expBattler 2 (battle_script.c:9716-9720, :9860-9864): the Attack / Defense
    # come unswapped (the Power Trick bit stays) and a Transformed battler's shadow follows.
    assert int(out.power_trick_desync[0]) == 1 and int(out.power_trick_desync[2]) == 1
    np.testing.assert_array_equal(np.asarray(out.shadow_stats[2]), np.asarray(out.stats[0, 1]))


def test_multi_partner_is_not_reloaded(tables, names):
    """MULTI's battler 2 draws from party 2: party 0 slot 0 (battler 0's) is the only reload."""
    s, after = leveling(tables, names, MULTI)
    s = s._replace(power_trick=s.power_trick.at[2].set(1))
    after = after._replace(power_trick=s.power_trick)
    out = pay_exp(tables, s, after, battler=1)
    assert int(out.level[0, 0]) == 51
    assert int(out.power_trick_desync[2]) == 0  # its active slot 0 is party 2's, not leveled
