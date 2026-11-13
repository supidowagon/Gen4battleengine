"""K3d-exp T6: the learnset resolve in both engines (design §1.5, §3.2).

Decomp (external/pokeplatinum):

* ``Pokemon_LevelUpMove`` (src/pokemon.c:3935-3966) scans the whole list of the species and
  form for entries of exactly the current level, from an index that restarts at 0 for each
  level (``GET_EXP_LEARNSET_INDEX = 0``, src/battle/battle_script.c:9882).
* ``BoxPokemon_AddMove`` (src/pokemon.c:3852-3873) walks the move slots in order: the first
  empty slot learns, a slot already holding the move stops the walk ("already known"), else
  the player is prompted (battle_script.c:10008-10018).
* The prompt: replace a slot (PP Ups 0, base PP; battle_script.c:10162-10187) or give up
  (:10132-10135); either way the scan continues at the same level. HM moves cannot be
  forgotten (battle_party.c:1724-1735, item.c:306-315).

Every JAX result is compared with the reference engine's (``ref_advance_learns``,
``ref_learn``, ``ref_learn_options``), all fields.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuzline.data.tables import to_numpy
from nuzline.engine.ids import EffectId
from nuzline.engine.levelup import (
    GIVE_UP,
    advance_learns,
    learn,
    learn_options,
    needs_learn,
)
from nuzline.engine.state import make_state
from nuzline.reference.convert import from_ref, to_ref
from nuzline.reference.engine import ref_advance_learns, ref_learn, ref_learn_options
from tests.engine.test_levelup import _base, _ko, _mon, _same, _with

_adv = jax.jit(advance_learns)
_answer = jax.jit(learn)
_opts = jax.jit(learn_options)


def _advance(tables, s):
    out = _adv(tables, s)
    _same(out, from_ref(ref_advance_learns(to_numpy(tables), to_ref(s))))
    return out


def _learn(tables, s, option: int):
    out = _answer(tables, s, jnp.int32(option))
    _same(out, from_ref(ref_learn(to_numpy(tables), to_ref(s), option)))
    return out


def _options(tables, s) -> list[bool]:
    got = [bool(x) for x in np.asarray(_opts(tables, s))]
    assert got == ref_learn_options(to_numpy(tables), to_ref(s))
    return got


def _state(tables, names, *player):
    return make_state(tables, list(player), [_mon(names, "Geodude", 12)])


def _set_moves(tables, names, s, slot: int, *moves: str):
    """Slot ``slot``'s moves (``""`` = empty) at full base PP."""
    ids = [names.move(m) for m in moves]
    pp = [int(tables.move_pp[m]) if m else 0 for m in ids]
    return s._replace(
        moves=s.moves.at[0, slot].set(jnp.asarray(ids, s.moves.dtype)),
        pp=s.pp.at[0, slot].set(jnp.asarray(pp, s.pp.dtype)),
    )


def _mark(s, slot: int, level: int, cursor: int = 0):
    return _with(s, learn_level={slot: level}, learn_cursor={slot: cursor})


def _moves(names, s, slot: int = 0) -> list[str]:
    return [names.move_names[int(m)] if int(m) else "" for m in np.asarray(s.moves[0, slot])]


def _prompted_move(tables, names, s, slot: int) -> str:
    row = int(tables.species_form_row[int(s.species[0, slot]), int(s.form[0, slot])])
    return names.move_names[int(tables.learnset_move[row, int(s.learn_cursor[slot])])]


# --- the scan and AddMove ----------------------------------------------------------------------


def test_an_unsupported_offered_move_is_masked(tables, names):
    """Ruling D10: Starly (four moves) prompted for Endeavor at 17, an UNSUPPORTED effect:
    every replacement is masked, giving up stays (coverage's ``learnable_moves`` relies on
    it). Prompted for Double Team at 13 (supported), the replacements are open."""
    starly = _mon(names, "Starly", 17, moves=("Tackle", "Growl", "Quick Attack", "Wing Attack"))
    s = _advance(tables, _mark(_state(tables, names, starly), 0, 17))
    assert bool(needs_learn(s)) and _prompted_move(tables, names, s, 0) == "Endeavor"
    assert _options(tables, s) == [False, False, False, False, True]
    s = _advance(tables, _mark(_state(tables, names, starly), 0, 13))
    assert _prompted_move(tables, names, s, 0) == "Double Team"
    assert _options(tables, s) == [True, True, True, True, True]


def test_auto_learn_takes_the_first_empty_slot_in_slot_order(tables, names):
    # Magikarp: Splash 1, Tackle 15, Flail 30. At L15 Tackle goes into the first empty slot,
    # slot 1 (the gap), with Tackle's base PP 35.
    s = _state(tables, names, _mon(names, "Magikarp", 15))
    s = _mark(_set_moves(tables, names, s, 0, "Splash", "", "Harden", ""), 0, 15)
    out = _advance(tables, s)
    assert _moves(names, out) == ["Splash", "Tackle", "Harden", ""]
    assert int(out.pp[0, 0, 1]) == int(tables.move_pp[names.move("Tackle")]) == 35
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (0, 0)
    assert not bool(needs_learn(out))


def test_add_move_stops_at_an_empty_slot_before_a_copy(tables, names):
    # AddMove's walk stops at the first slot that is empty OR holds the move: an empty slot 1
    # comes before the Tackle in slot 2, so Tackle is learned a second time (pokemon.c:
    # 3857-3866).
    s = _state(tables, names, _mon(names, "Magikarp", 15))
    s = _mark(_set_moves(tables, names, s, 0, "Splash", "", "Tackle", ""), 0, 15)
    assert _moves(names, _advance(tables, s)) == ["Splash", "Tackle", "Tackle", ""]


def test_an_already_known_move_is_skipped(tables, names):
    s = _state(tables, names, _mon(names, "Magikarp", 15))
    s = _mark(_set_moves(tables, names, s, 0, "Splash", "Tackle", "", ""), 0, 15)
    out = _advance(tables, s)
    assert _moves(names, out) == ["Splash", "Tackle", "", ""]
    assert int(out.learn_level[0]) == 0


def test_unsorted_row_354_learns_per_level_from_index_0(tables, names):
    # Vanilla row 354 (Banette) lists Screech at level 8 (index 4) before Night Shade at
    # level 5 (index 5). Banette L4 (fast rate: L8 base 409, L9 base 583) with 408 exp KOs
    # Bonsly: 145 * 150 / 100 = 217, 625 exp, L9 in one payout. Level 5's scan finds Night
    # Shade at index 5; level 8's restarts at 0 and finds Screech at index 4. A cursor carried
    # across levels (6 after level 5) or a scan that stops at the first higher level (index 4,
    # level 8, at level 5) would miss one of them.
    banette = _mon(names, "Banette", 4, moves=("Knock Off",))
    s = make_state(tables, [banette], [_mon(names, "Bonsly", 15), _mon(names, "Geodude", 12)])
    assert (_base(tables, names, "Banette", 8), _base(tables, names, "Banette", 9)) == (409, 583)
    s = _with(s, exp={0: 408})
    paid = _ko(tables, s)
    assert int(paid.level[0, 0]) == 9 and int(paid.exp[0]) == 625
    assert (int(paid.learn_level[0]), int(paid.learn_cursor[0])) == (5, 0)
    out = _advance(tables, paid)
    assert _moves(names, out) == ["Knock Off", "Night Shade", "Screech", ""]
    assert int(out.learn_level[0]) == 0


def test_a_level_without_entries_moves_on_to_the_next(tables, names):
    # Magikarp L30 with a marker at 14: levels 14..29 have nothing but Tackle at 15 (known),
    # level 30 auto-learns Flail into the empty slot.
    s = _state(tables, names, _mon(names, "Magikarp", 30))
    s = _mark(_set_moves(tables, names, s, 0, "Splash", "Tackle", "", ""), 0, 14)
    out = _advance(tables, s)
    assert _moves(names, out) == ["Splash", "Tackle", "Flail", ""]
    assert int(out.learn_level[0]) == 0


# --- prompts ------------------------------------------------------------------------------------


def _bulbasaur(tables, names, *moves):
    # Bulbasaur learns PoisonPowder (index 4) and Sleep Powder (index 5) at level 13.
    s = _state(tables, names, _mon(names, "Bulbasaur", 13), _mon(names, "Magikarp", 15))
    return _mark(_set_moves(tables, names, s, 0, *moves), 0, 13)


def test_a_full_move_list_stops_at_a_prompt(tables, names):
    s = _bulbasaur(tables, names, "Tackle", "Growl", "Leech Seed", "Vine Whip")
    out = _advance(tables, s)
    assert bool(needs_learn(out))
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (13, 4)
    assert _prompted_move(tables, names, out, 0) == "PoisonPowder"
    assert _moves(names, out) == ["Tackle", "Growl", "Leech Seed", "Vine Whip"]
    # advance_learns at a prompt changes nothing.
    _same(_advance(tables, out), out)


def test_giving_up_continues_the_same_level(tables, names):
    s = _advance(tables, _bulbasaur(tables, names, "Tackle", "Growl", "Leech Seed", "Vine Whip"))
    out = _learn(tables, s, GIVE_UP)
    assert (int(out.learn_level[0]), int(out.learn_cursor[0])) == (13, 5)
    assert _prompted_move(tables, names, out, 0) == "Sleep Powder"
    done = _learn(tables, out, GIVE_UP)
    assert not bool(needs_learn(done))
    assert _moves(names, done) == ["Tackle", "Growl", "Leech Seed", "Vine Whip"]


def test_the_resumed_scan_depends_on_the_choice(tables, names):
    # Bulbasaur knows Sleep Powder. Replacing Tackle with PoisonPowder (base PP 35, PP Ups
    # 0) leaves Sleep Powder "already known"; replacing Sleep Powder itself prompts it again.
    s = _advance(tables, _bulbasaur(tables, names, "Tackle", "Growl", "Sleep Powder", "Vine Whip"))
    kept = _learn(tables, s, 0)
    assert _moves(names, kept) == ["PoisonPowder", "Growl", "Sleep Powder", "Vine Whip"]
    assert int(kept.pp[0, 0, 0]) == int(tables.move_pp[names.move("PoisonPowder")]) == 35
    assert not bool(needs_learn(kept))
    swapped = _learn(tables, s, 2)
    assert _moves(names, swapped) == ["Tackle", "Growl", "PoisonPowder", "Vine Whip"]
    assert bool(needs_learn(swapped))
    assert _prompted_move(tables, names, swapped, 0) == "Sleep Powder"


def test_the_lowest_slot_is_prompted_first(tables, names):
    # Slot 0 (Bulbasaur) is prompted; slot 1 (Magikarp) waits with its marker, then learns.
    s = _bulbasaur(tables, names, "Tackle", "Growl", "Leech Seed", "Vine Whip")
    s = _mark(_set_moves(tables, names, s, 1, "Splash", "", "", ""), 1, 15)
    out = _advance(tables, s)
    assert int(out.learn_level[0]) == 13 and int(out.learn_level[1]) == 15
    out = _learn(tables, _learn(tables, out, GIVE_UP), GIVE_UP)
    assert not bool(needs_learn(out))
    assert _moves(names, out, 1) == ["Splash", "Tackle", "", ""]


# --- learn_options --------------------------------------------------------------------------------


def _prompt(tables, names, **fields):
    s = _advance(tables, _bulbasaur(tables, names, "Tackle", "Growl", "Leech Seed", "Vine Whip"))
    return s._replace(**{k: jnp.asarray(v, getattr(s, k).dtype) for k, v in fields.items()})


def test_every_option_is_open_by_default_and_giving_up_always(tables, names):
    assert _options(tables, _prompt(tables, names)) == [True] * 5


def test_an_hm_cannot_be_forgotten(tables, names):
    s = _set_moves(tables, names, _prompt(tables, names), 0, "Tackle", "Cut", "Leech Seed", "")
    s = s._replace(moves=s.moves.at[0, 0, 3].set(names.move("Vine Whip")))
    assert _options(tables, s) == [True, False, True, True, True]


@pytest.mark.parametrize(
    ("field", "move"),
    [("disabled_move", "Tackle"), ("locked_move", "Growl"), ("choice_lock", "Leech Seed")],
)
def test_the_active_learner_keeps_its_held_moves(tables, names, field, move):
    s = _prompt(tables, names)
    s = s._replace(**{field: getattr(s, field).at[0].set(names.move(move))})
    want = [m != move for m in ("Tackle", "Growl", "Leech Seed", "Vine Whip")]
    assert _options(tables, s) == [*want, True]
    # On the bench (slot 1 active) the battler's volatiles are not the learner's.
    bench = s._replace(active=s.active.at[0].set(1))
    assert _options(tables, bench) == [True] * 5


def test_a_seen_slot_and_the_pinned_slot_are_masked(tables, names):
    s = _prompt(tables, names)
    seen = s._replace(ai_seen_moves=s.ai_seen_moves.at[0, 1].set(1))
    assert _options(tables, seen) == [True, False, True, True, True]
    pinned = s._replace(learn_pin=jnp.int8(4))
    assert _options(tables, pinned) == [True, True, True, False, True]
    assert _options(tables, pinned._replace(active=s.active.at[0].set(1))) == [True] * 5


@pytest.mark.parametrize("game", ["tables", "kaizo_tables"])
@pytest.mark.parametrize("move", ["Sleep Talk", "Assist"])
def test_moves_that_read_the_whole_move_list_are_unsupported(request, game, move):
    # Sleep Talk (BtlCmd_TrySleepTalk, battle_script.c:4907) and Assist (BtlCmd_TryAssist,
    # :6405) read the move list when they run, which a deferred learn could change (design
    # §3.3 (a)): they stay UNSUPPORTED, so coverage refuses them, also as learnable moves.
    tables = request.getfixturevalue(game)
    names = request.getfixturevalue("names" if game == "tables" else "kaizo_names")
    assert int(tables.move_effect[names.move(move)]) == EffectId.UNSUPPORTED


def test_kaizo_row_359_scans_to_its_last_column(kaizo_tables, kaizo_names):
    """T6 review Minor 4: Kaizo Absol (row 359) is the one 21-entry row, with no end marker
    in the tables; its last entry is Shadow Sneak at 99. A prompt there, then the give-up,
    runs the cursor past the last column (both engines)."""
    tables, names = kaizo_tables, kaizo_names
    absol = _mon(names, "Absol", 99, moves=("Bite", "Slash", "Pursuit", "Megahorn"))
    s = _advance(tables, _mark(_state(tables, names, absol), 0, 99))
    assert int(tables.species_form_row[names.species("Absol"), 0]) == 359
    assert int(s.learn_cursor[0]) == 20 and _prompted_move(tables, names, s, 0) == "Shadow Sneak"
    out = _learn(tables, s, GIVE_UP)
    assert int(out.learn_level[0]) == 0 and not bool(needs_learn(out))
