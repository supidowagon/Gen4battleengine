"""Coverage of experience and in-battle level-ups (K3d-exp design §4, task T7): the moves a
player Pokemon can learn by levelling up in the fight, and the refusals of §4.2-4.7.

Hand numbers (vanilla tables): Roark's Geodude L12 has exp yield 73, so
``E = 73 * 12 // 7 = 125`` (battle_script.c:2216-2217) and the per-opponent bound of
``coverage.reachable_level`` is ``max(E, 2) * 3/2 * exp_boost / 100 + 1``:
``125 * 3 * 2 * 100 // 400 + 1 = 188``. Starly grows at rate 3 (medium slow), whose level-17
base is 3120."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.games import KAIZO
from nuzline.data.tables import Names
from nuzline.engine.coverage import (
    EXP_ITEM_HOLD_EFFECTS,
    FRIENDSHIP_POWER_EFFECTS,
    IMPLEMENTED_EFFECTS,
    ITEM_MOVING_EFFECTS,
    RELOAD_STATE_EFFECTS,
    Support,
    check_coverage,
    forced_learn_moves,
    learnable_moves,
    reachable_learn_moves,
    reachable_level,
)
from nuzline.engine.ids import EffectId
from nuzline.engine.levelup import pay_exp
from nuzline.engine.state import make_state
from tests.engine.helpers import kaizo_names_with, roark_party, roark_trainer, sample_party

STARLY_SLOT = 3  # tests/fixtures/sample_party.toml
L17 = 3120  # exp_table[3][17]
GEODUDE_BOUND = 188


def _starly(names, party=None, **changes):
    """The sample party with its Starly (slot 3) at level 16 and ``changes``."""
    party = list(party or sample_party(names))
    party[STARLY_SLOT] = replace(party[STARLY_SLOT], level=16, **changes)
    return party


def _geodude(names):
    return [roark_party(names)[0]]


def _fainted(names):
    """Roark's party, every member at 0 HP: no opponent can faint, so nothing is paid."""
    return [replace(m, current_hp=0) for m in roark_party(names)]


def _missing(tables, names, player, opp, trainer=None):
    return check_coverage(tables, names, player, trainer or roark_trainer(), opp, 0).missing


def _lines(missing, kind):
    return [line for line in missing if f"({kind} UNSUPPORTED)" in line]


# --- §4.1: the reachable level and the learnable moves ----------------------------------


def test_reachable_level_is_the_payout_bound(tables, names):
    geodude = _geodude(names)
    starly = _starly(names, exp=L17 - GEODUDE_BOUND)[STARLY_SLOT]
    assert reachable_level(tables, starly, geodude) == 17
    assert reachable_level(tables, replace(starly, exp=L17 - GEODUDE_BOUND - 1), geodude) == 16
    # Every opponent that can faint adds its bound: Onix L12 (yield 108, E = 185) adds 278,
    # Cranidos L14 (yield 99, E = 198) adds 298; 2535 + 188 + 278 + 298 = 3299 >= 3120.
    base = replace(starly, exp=None)
    assert reachable_level(tables, base, roark_party(names)) == 17
    assert reachable_level(tables, base, geodude) == 16
    fainted = [replace(m, current_hp=0) for m in roark_party(names)]
    assert reachable_level(tables, base, fainted) == 16
    assert reachable_level(tables, replace(starly, level=100, exp=None), geodude) == 100


def test_reachable_level_bound_with_lucky_egg_and_boost(tables, names):
    """Lucky Egg and a foreign non-OT Pokemon: ``125 * 3 * 3 * 170 // 400 + 1 = 479``.
    An unresolved boost (0) counts as 170: ``125 * 3 * 2 * 170 // 400 + 1 = 319``."""
    geodude = _geodude(names)
    egg = names.item("Lucky Egg")
    starly = _starly(names, exp=L17 - 479, item=egg, exp_boost=170)[STARLY_SLOT]
    assert reachable_level(tables, starly, geodude) == 17
    assert reachable_level(tables, replace(starly, exp=L17 - 480), geodude) == 16
    unresolved = replace(starly, item=0, exp_boost=0)
    assert reachable_level(tables, replace(unresolved, exp=L17 - 319), geodude) == 17
    assert reachable_level(tables, replace(unresolved, exp=L17 - 320), geodude) == 16


def test_the_bound_covers_the_engine_payout(tables, names):
    """The engine's payout to a lone participant (``levelup.pay_exp``) never exceeds the
    bound: 125 -> Lucky Egg 187 -> trainer 280 -> boost 170: 476 <= 479."""
    egg = names.item("Lucky Egg")
    starly = _starly(names, exp=2535, item=egg, exp_boost=170)[STARLY_SLOT]
    before = make_state(tables, [starly], _geodude(names))
    after = before._replace(hp=before.hp.at[1, 0].set(0))
    paid = pay_exp(tables, before, after)
    gained = int(paid.exp[0]) - 2535
    assert gained == 476
    assert gained + 1 <= 479  # the bound's own +1 slack


def test_reachable_learn_moves_reads_the_form_row_levels(tables, names):
    """Starly learns Double Team at 13 and Endeavor at 17 (vanilla learnset)."""
    geodude = _geodude(names)
    starly = _starly(names, exp=L17 - GEODUDE_BOUND)[STARLY_SLOT]
    assert reachable_learn_moves(tables, starly, geodude) == [names.move("Endeavor")]
    assert reachable_learn_moves(tables, replace(starly, exp=L17 - 189), geodude) == []
    young = replace(starly, level=12, exp=None)
    assert names.move("Double Team") in reachable_learn_moves(tables, young, roark_party(names))


def _gap(names):
    """Starly's moves with its last slot empty: a level-up move goes there unasked."""
    return (names.move("Tackle"), names.move("Growl"), names.move("Quick Attack"), 0)


@pytest.fixture
def endeavor_unsupported(monkeypatch):
    """These tests need an unsupported move Starly learns at L17: Endeavor was one until E2
    batch 10 (B10-5) modelled it, so the registry entry is turned off for them."""
    monkeypatch.setitem(IMPLEMENTED_EFFECTS, EffectId.ENDEAVOR, Support.NONE)


def test_a_learnable_unsupported_move_is_refused(tables, names, endeavor_unsupported):
    """Endeavor (UNSUPPORTED here) is reached only through the level-up in battle, and with an
    empty slot it is learned without a prompt (pokemon.c:3852-3873, ruling D10)."""
    geodude = _geodude(names)
    assert names.move("Endeavor") not in sample_party(names)[STARLY_SLOT].moves
    line = "Player STARLY (slot 3): Endeavor (level-up in battle move effect UNSUPPORTED)"
    gap = _gap(names)
    near = _starly(names, exp=L17 - GEODUDE_BOUND, moves=gap)
    assert line in _missing(tables, names, near, geodude)
    assert forced_learn_moves(tables, near[STARLY_SLOT], geodude) == [names.move("Endeavor")]
    far = _starly(names, exp=L17 - GEODUDE_BOUND - 1, moves=gap)
    assert _missing(tables, names, far, geodude) == []


def test_a_prompt_only_unsupported_learnable_is_masked_not_refused(
    tables, names, endeavor_unsupported
):
    """Ruling D10 (the user's Kaizo Roark fight: Luvdisc, Lucky Chant at L17, four moves):
    with every slot full Endeavor is only a prompt (battle_script.c:10016-10018), which the
    learn options mask (``levelup.learn_options``) and the player can give up, so the fight
    stays covered. Endeavor's effect is set UNSUPPORTED in these tables (B10-5 modelled it; the
    learn options mask by the table's effect, ``coverage.learn_masked``)."""
    endeavor = names.move("Endeavor")
    tables = tables._replace(
        move_effect=jnp.asarray(tables.move_effect).at[endeavor].set(int(EffectId.UNSUPPORTED))
    )
    geodude = _geodude(names)
    full = _starly(names, exp=L17 - GEODUDE_BOUND)
    starly = full[STARLY_SLOT]
    assert all(starly.moves)
    assert reachable_learn_moves(tables, starly, geodude) == [endeavor]
    assert forced_learn_moves(tables, starly, geodude) == []
    assert learnable_moves(tables, starly, geodude) == []
    assert _missing(tables, names, full, geodude) == []


def test_the_sample_party_against_roark_learns_only_supported_moves(tables, names):
    learned = {
        names.move_names[m]
        for mon in sample_party(names)
        for m in reachable_learn_moves(tables, mon, roark_party(names))
    }
    assert learned == {"Peck", "Double Team", "Mega Drain"}
    assert _missing(tables, names, sample_party(names), roark_party(names)) == []


def test_a_learnable_move_reaches_a_kaizo_rom_change(tables, names, vanilla_manifest):
    """The learnable list widens ``rom_change_refusals`` (a supported move, Double Team)."""
    team = _starly(names)
    team[STARLY_SLOT] = replace(team[STARLY_SLOT], level=12, exp=None)
    kaizo = kaizo_names_with(vanilla_manifest, [{"moves": [names.move("Double Team")]}])
    rom = "Kaizo ROM: c is not modelled and this fight reaches it"
    assert any(rom in line for line in _missing(tables, kaizo, team, roark_party(names)))
    assert not any(rom in line for line in _missing(tables, kaizo, team, _fainted(names)))


# --- §4.2-4.7 --------------------------------------------------------------------------


def test_the_effect_ids_are_the_decomp_moves(tables, names):
    """generated/move_battle_effects.txt (line N is id N - 1), read through each move's
    res/moves/<move>/data.json effect."""
    effect = np.asarray(tables.move_ai_effect)
    by_effect = {
        105: ("Thief", "Covet"),
        177: ("Trick", "Switcheroo"),
        188: ("Knock Off",),
        233: ("Fling",),
        121: ("Return",),
        123: ("Frustration",),
        57: ("Transform",),
        82: ("Mimic",),
        238: ("Power Trick",),
    }
    for eid, moves in by_effect.items():
        assert all(int(effect[names.move(m)]) == eid for m in moves), moves
    assert set(ITEM_MOVING_EFFECTS) == {105, 177, 188, 233}
    assert set(FRIENDSHIP_POWER_EFFECTS) == {121, 123}
    assert set(RELOAD_STATE_EFFECTS) == {57, 82, 238}
    hold = np.asarray(tables.item_hold_effect)
    for item in ("Lucky Egg", "Exp. Share", "Soothe Bell", "Macho Brace", "Power Weight"):
        assert int(hold[names.item(item)]) in EXP_ITEM_HOLD_EFFECTS, item


@pytest.mark.parametrize("game", ["tables", "kaizo_tables"])
def test_the_exp_sensitive_moves_are_unsupported(request, game):
    """Every move of §4.2, §4.4 and §4.5 is an UNSUPPORTED effect in both games but Knock Off
    (188, E2 batch 10, B10-6), so a Pokemon that knows one is refused on that line already; the
    exp lines are the guard for an implemented one (Knock Off's: exp_refusals' §4.2 line, its own
    test below)."""
    tables = request.getfixturevalue(game)
    effect, ai_effect = np.asarray(tables.move_effect), np.asarray(tables.move_ai_effect)
    eids = (
        *(e for e in ITEM_MOVING_EFFECTS if e != 188),
        *FRIENDSHIP_POWER_EFFECTS,
        *RELOAD_STATE_EFFECTS,
    )
    for move in np.flatnonzero(np.isin(ai_effect, eids)):
        engine = EffectId(int(effect[move]))
        assert IMPLEMENTED_EFFECTS.get(engine, Support.NONE) == Support.NONE, int(move)


def test_an_unresolved_exp_boost_is_refused_below_level_100(tables, names):
    """§4.3: ``exp_boost == 0`` (``SaveMon.to_spec`` without a trainer). The payout would pay
    ``t * 0 / 100 = 0`` exp yet still add EVs; coverage refuses the fight instead."""
    party = sample_party(names)
    party[0] = replace(party[0], exp_boost=0)
    lines = _lines(_missing(tables, names, party, roark_party(names)), "experience")
    assert lines == [
        (
            "Player MONFERNO (slot 0): exp multiplier unresolved (original trainer and language "
            "unknown: read the save with its trainer) (experience UNSUPPORTED)"
        )
    ]
    party[0] = replace(party[0], level=100, stats=None, current_hp=None, exp=None)
    assert _lines(_missing(tables, names, party, roark_party(names)), "experience") == []


def test_an_exp_item_with_an_item_moving_move_is_refused(tables, names):
    """§4.2 (ruling D9): the payout reads the party item (battle_script.c:2209, :9706,
    :10226; pokemon.c:2649)."""
    party = sample_party(names)
    party[0] = replace(party[0], item=names.item("Lucky Egg"))
    opp = roark_party(names)
    thief = [replace(opp[0], moves=(names.move("Thief"), *opp[0].moves[1:]))] + opp[1:]
    want = (
        "Player MONFERNO (slot 0): Lucky Egg is read from the party by the exp payout, and "
        "Thief can move held items in battle (experience UNSUPPORTED)"
    )
    assert _lines(_missing(tables, names, party, thief), "experience") == [want]
    assert _lines(_missing(tables, names, party, opp), "experience") == []
    plain = sample_party(names)
    assert _lines(_missing(tables, names, plain, thief), "experience") == []


def test_an_egg_location_with_a_friendship_reader_is_refused(tables, names):
    """§4.4 (ruling D6): only when the Pokemon can level and a Return / Frustration is in
    the fight (pokemon.c:2668-2670)."""
    opp = roark_party(names)
    ret = [replace(opp[0], moves=(names.move("Return"), *opp[0].moves[1:]))] + opp[1:]
    party = sample_party(names)
    party[0] = replace(party[0], egg_location=1)
    want = (
        "Player MONFERNO (slot 0): egg location 1 may add 1 friendship per level-up, which "
        "Return reads (experience UNSUPPORTED)"
    )
    assert _lines(_missing(tables, names, party, ret), "experience") == [want]
    assert _lines(_missing(tables, names, party, opp), "experience") == []
    no_egg = sample_party(names)
    assert _lines(_missing(tables, names, no_egg, ret), "experience") == []
    capped = list(party)
    capped[0] = replace(party[0], level=100, stats=None, current_hp=None, exp=None)
    assert _lines(_missing(tables, names, capped, ret), "experience") == []


@pytest.mark.parametrize("move", ["Transform", "Mimic", "Power Trick"])
def test_a_reload_state_move_is_refused_when_a_level_up_is_reachable(tables, names, move):
    """§4.5: ``BattleSystem_ReloadPokemon`` (battle_lib.c:173-191)."""
    party = sample_party(names)
    party[0] = replace(party[0], moves=(names.move(move), *party[0].moves[1:]))
    lines = _lines(_missing(tables, names, party, roark_party(names)), "experience")
    assert lines == [
        (
            f"Player MONFERNO (slot 0): a level-up reloads the battler, which {move} changes "
            "(battle_lib.c:173-191) (experience UNSUPPORTED)"
        )
    ]
    assert _lines(_missing(tables, names, party, _fainted(names)), "experience") == []


def test_junk_pp_ups_in_an_empty_slot_are_refused(tables, names):
    """§4.6: a move learned into the slot inherits its PP Ups (pokemon.c:3929-3930)."""
    party = sample_party(names)
    mon = party[0]
    party[0] = replace(mon, moves=(*mon.moves[:3], 0), pp=None, pp_ups=(3, 0, 0, 2))
    lines = _lines(_missing(tables, names, party, roark_party(names)), "experience")
    assert lines == [
        (
            "Player MONFERNO (slot 0): empty move slot 3 has 2 PP Ups, which a move learned "
            "there inherits (experience UNSUPPORTED)"
        )
    ]
    party[0] = replace(party[0], pp_ups=(3, 0, 0, 0))
    assert _lines(_missing(tables, names, party, roark_party(names)), "experience") == []


def test_out_of_scope_formats_are_refused(tables, names):
    """§4.7: ``expBattler = 2`` exists only in doubles (battle_script.c:9716-9720); every
    format but singles is refused by ``IMPLEMENTED_FORMATS``."""
    for fmt in ("double", "double_two_trainer", "multi"):
        trainer = replace(roark_trainer(), format=fmt)
        missing = _missing(tables, names, sample_party(names), roark_party(names), trainer)
        assert any("(battle format UNSUPPORTED)" in line for line in missing), fmt


def test_kaizo_names_keep_the_learnable_scan(tables, vanilla_manifest, names, endeavor_unsupported):
    """The scan runs on the tables, whatever the game's names (no game gate)."""
    kaizo = Names({**vanilla_manifest, "game": KAIZO, "unclassified_rom_changes": 0})
    near = _starly(names, exp=L17 - GEODUDE_BOUND, moves=_gap(names))
    missing = _missing(tables, kaizo, near, _geodude(names))
    assert any("Endeavor (level-up in battle move effect UNSUPPORTED)" in m for m in missing)


def test_make_state_takes_the_new_spec_fields(tables, names):
    """``egg_location`` and ``pp_ups`` are coverage inputs only: the state ignores them."""
    party = sample_party(names)
    plain = make_state(tables, party, roark_party(names))
    marked = [replace(m, egg_location=7, pp_ups=(1, 1, 1, 1)) for m in party]
    other = make_state(tables, marked, roark_party(names))
    for a, b in zip(plain, other, strict=True):
        assert bool(jnp.all(a == b))


def test_knock_off_meets_the_exp_item_line(tables, names):
    """§4.2 (ruling D9) for Knock Off (B10-6): an exp-item holder in a fight where Knock Off can be
    used is refused, as the payout reads the party item (a knocked-off item stays on the party,
    battle_io_command.c:590-592) while the battle copy holds nothing."""
    from nuzline.engine.coverage import exp_refusals

    starly = replace(sample_party(names)[STARLY_SLOT], item=names.item("Lucky Egg"))
    lines = exp_refusals(tables, names, starly, [], [names.move("Knock Off")], False)
    assert any("Knock Off can move held items in battle" in line for line in lines), lines
