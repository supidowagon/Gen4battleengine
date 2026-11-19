"""Convert an emulator battle trace into engine states, actions and outcomes.

``tools/trace_battle.lua`` records a raw trace (JSON lines) in DeSmuME. This module
bundles it with the player's party and the trainer id into one JSON file
(``tests/golden/*.json``: extracted battle state only, never ROM bytes), and
``trace_to_states`` turns a bundle into a ``ReplayTurn`` per turn (state, actions,
outcome, the player's replacement slot and the move-learn choices) for
``tests/test_golden.py``. The formats, the RAM layout and the inference rules are in
``docs/tracing.md``.

Per turn, the before-state is the engine's replay of the previous turn with every field
the trace records (``COMPARED_FIELDS``) written over it. The fields the trace lacks are
carried forward from the engine. The actions come from PP drops and active-slot
changes. Hits, crits and damage rolls come from the ``hit``/``move_end`` records,
matched against ``damage_for_roll``. The remaining chance fields are found by
replaying every candidate through ``apply_turn`` (``step`` and the turn-end
replacements) and keeping the one that best matches the next record. A draw that shows
only on a later turn (a secondary confusion, a trap's duration) is refitted by rewinding
to an earlier turn's tied candidates (``_refit``).

Quick Claw's hidden ``speedRand`` bit (``BattleState.qc_roll``, which the trace does not
record) is fitted per turn as a variant of the turn's *start* state, for a side whose
active Pokemon holds a Quick Claw: the value carried from the previous turn first, so it
is kept unless the other value matches the record better. That bit is the previous turn's
draw (``Outcome.quick_claw``), so a changed value is written back into the previous
turn's outcome and that turn is replayed again; on turn 0 it is the battle-start draw.

CLI: ``uv run python -m tools.trace_to_state --trace <raw.jsonl> --out
tests/golden/roark_01.json (--party tests/fixtures/golden_party_roark_01.toml | --save
<save.dsv>) [--game vanilla|kaizo] [--trainer roark | --trainer-id N]``. ``--save`` takes
the player's party from a save (``SaveMon.to_spec``), so no party file is typed in.
``--game`` picks the data set (tables, trainers) the bundle is built and converted with;
the bundle records it as ``"game"`` (a bundle without it is vanilla).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from nuzline.data.games import VANILLA, get_game
from nuzline.data.spec import PokemonSpec
from nuzline.data.tables import Names, Tables
from nuzline.data.trainer import Trainer
from nuzline.engine.actions import FIRST_ITEM_ACTION, legal_actions, usable_trainer_items
from nuzline.engine.damage import (
    PRESENT_HEAL,
    PRESENT_TABLE,
    confusion_damage,
    damage_for_roll,
)
from nuzline.engine.format import foe as foe_of
from nuzline.engine.format import format_of, party
from nuzline.engine.ids import AbilityId, EffectId, ItemId, Stat, Status, Weather
from nuzline.engine.levelup import GIVE_UP, advance_learns, learn, learn_options, needs_learn
from nuzline.engine.order import MOVE_ACTIONS, effective_speed, used_move
from nuzline.engine.outcome import MAX_HITS, Outcome, empty_outcome, read_chance
from nuzline.engine.state import LAYOUT, MOVES, SLOTS, BattleState, make_state
from nuzline.engine.step import needs_replacement, replace, step
from nuzline.saves.save import SaveFile
from nuzline.saves.status import decode_status

INTERMEDIATE = Path(__file__).resolve().parents[1] / "data" / "intermediate"
# /2 (K3d-exp): party entries carry exp, exp_boost, pokerus and friend_bonus (and, from T7,
# the coverage inputs egg_location and pp_ups). A /1 bundle, or a /2 one written before T7,
# still loads: its entries lack them and get the PokemonSpec defaults.
BUNDLE_FORMAT = "nuzline-golden-trace/2"
BUNDLE_FORMATS = ("nuzline-golden-trace/1", BUNDLE_FORMAT)

# The BattleState fields a ``turn`` record fills; tests/test_golden.py compares them.
COMPARED_FIELDS: tuple[str, ...] = (
    "species",
    "level",
    "hp",
    "max_hp",
    "status",
    "status_counter",
    "item",
    "moves",
    "pp",
    "active",
    "stages",
    "weather",
)
_START_FIELDS = ("species", "level", "max_hp", "moves", "item")

# BattleMon.statBoosts order (generated/battle_stats.txt): HP, Atk, Def, Spe, SpA, SpD,
# Acc, Eva; 6 is neutral.
STAGE_ORDER = (Stat.HP, Stat.ATK, Stat.DEF, Stat.SPE, Stat.SPA, Stat.SPD, Stat.ACC, Stat.EVA)
STAGE_NEUTRAL = 6
MOVE_STATUS_MISSED = 1 << 0  # include/constants/battle/moves.h
MOVE_STATUS_FAILED = 1 << 6  # include/constants/battle/moves.h
MOVE_STATUS_PROTECTED = 1 << 15  # include/constants/battle/moves.h
COMMAND_RESULT = 41  # BATTLE_CONTROL_RESULT
NO_ACTIONS = (-1, -1)  # the last turn: the trace ends before any choice

# FIELD_CONDITION_* (TEMP | PERM pairs) and FIELD_CONDITION_DEEP_FOG
_WEATHER_BITS = (
    (0x3, Weather.RAIN),
    (0xC, Weather.SANDSTORM),
    (0x30, Weather.SUN),
    (0xC0, Weather.HAIL),
    (1 << 15, Weather.FOG),
)

_MAX_CANDIDATES = 4096
# Learn-move prompts answered in one turn (``apply_turn``'s ``learn_choices``, padded with -1
# to this length so that one compile serves every count).
MAX_LEARNS = 8
_NO_LEARNS = np.full(MAX_LEARNS, -1, np.int32)  # learn_array(()) on the host
_CHUNK = 16  # candidates replayed per jitted call (one compile)
_I8, _I32 = jnp.int8, jnp.int32


@functools.cache
def _outcome_dtypes() -> tuple:
    """Each ``Outcome`` field's engine dtype (``empty_outcome``'s: int8, ``call`` int16)."""
    return tuple(f.dtype for f in empty_outcome())


def _engine_outcome(fields) -> Outcome:
    """An ``Outcome`` of ``fields`` (in ``Outcome._fields`` order), each in its engine dtype:
    every replay passes the same dtypes, so it runs the one ``_replay_chunk`` program."""
    return Outcome(*(jnp.asarray(f, d) for f, d in zip(fields, _outcome_dtypes(), strict=True)))


class TraceError(ValueError):
    """A trace that cannot be converted."""


class TraceMismatch(TraceError):
    """The trace contradicts the party file, the trainer data or the engine."""


class AmbiguousTrace(TraceError):
    """More than one realized outcome fits the trace; ``candidates`` lists them."""

    def __init__(self, message: str, candidates: list):
        super().__init__(f"{message}; candidates {candidates}")
        self.candidates = candidates


@dataclasses.dataclass(frozen=True)
class Turn:
    """One converted turn. ``roll_candidates[(side, hit)]`` lists every ``(crit, roll)``
    that fits that hit (the lowest is used); ``mismatched`` names the
    ``COMPARED_FIELDS`` on which the best replay still differs from the next record.
    ``player_slot`` is the party slot the player sent in for a fainted active Pokemon at
    the turn's end (``apply_turn``), -1 for none. ``learn_choices`` are the player's answers
    to the turn's learn-move prompts (``levelup.learn`` options: 0-3 replace that move slot,
    ``GIVE_UP``), in the engine's order: by player slot ascending (``levelup.needs_learn``),
    each slot's prompts in learnset order (``_fit_learns``)."""

    state: BattleState
    actions: tuple[int, int]
    outcome: Outcome
    roll_candidates: dict[tuple[int, int], list[tuple[int, int]]]
    mismatched: tuple[str, ...]
    player_slot: int = -1
    learn_choices: tuple[int, ...] = ()


class ReplayTurn(NamedTuple):
    """One turn as ``tests/test_golden.py`` replays it: ``apply_turn`` of ``state`` with
    ``actions``, ``outcome``, ``player_slot`` and ``learn_choices`` (the opponent's
    replacement is the next turn's active slot)."""

    state: BattleState
    actions: tuple[int, int]
    outcome: Outcome
    player_slot: int
    learn_choices: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _Fit:
    """One replayed candidate of a turn: its index in ``_Turn.candidate_list``, start
    ``qc_roll``, actions, outcome fields, replayed state (replacements made), mismatched
    fields, ``needs_replacement`` after the step, the player's replacement slot and the
    learn choices (``_fit_learns``; empty until fitted)."""

    index: int
    qc: tuple[int, int]
    actions: tuple[int, int]
    outcome: dict
    state: BattleState
    mismatched: tuple[str, ...]
    need: tuple[bool, bool]
    player_slot: int
    learn_choices: tuple[int, ...] = ()


def decode_weather(mask: int) -> Weather:
    """``ids.Weather`` from ``fieldConditionsMask``."""
    for bits, weather in _WEATHER_BITS:
        if mask & bits:
            return weather
    return Weather.NONE


def resolve_roll(candidates: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """The ``(crit, roll)`` used for a hit that ``candidates`` all fit. They all give
    the same damage, so the lowest is used. A crit is an observable game event,
    however, so candidates that disagree on it raise ``AmbiguousTrace``."""
    candidates = list(candidates)
    if len({crit for crit, _ in candidates}) > 1:
        raise AmbiguousTrace("both a crit and a non-crit roll fit", candidates)
    return min(candidates)


def initial_state(
    tables: Tables,
    player: list[PokemonSpec],
    trainer: Trainer,
    leads: tuple[int, int],
    game: str = VANILLA,
    terrain: int | None = None,
) -> BattleState:
    """The fight's turn-0 state: ``player`` against the trainer's built party (from
    ``game``'s intermediates), bag, start weather and battle terrain, with the given active
    slots. The terrain is the trainer's (``Trainer.terrain``) unless ``terrain`` gives the
    one the traced battle started on (``battle_terrain``). A move of either party that reads
    the terrain (``coverage.TERRAIN_EFFECTS``) raises ``TraceError`` when the terrain is not
    known: trace conversion runs no coverage, and no terrain table may be read at -1."""
    from nuzline.data.trainer import IntermediateData, build_trainer_party
    from nuzline.engine.coverage import TERRAIN_EFFECTS, start_weather

    g = get_game(game)
    opp = build_trainer_party(trainer, IntermediateData(g.intermediate_dir, g.id))
    terrain = battle_terrain(trainer, terrain)
    if terrain < 0:
        effect = np.asarray(tables.move_ai_effect)
        reading = sorted(
            {
                int(m)
                for s in (*player, *opp)
                for m in s.moves
                if m and int(effect[m]) in TERRAIN_EFFECTS
            }
        )
        if reading:
            raise TraceError(
                f"moves {reading} read the battle terrain, which the trainer's maps do not "
                "settle (Trainer.terrains); pass --terrain"
            )
    return make_state(
        tables,
        player,
        opp,
        start_weather(trainer),
        opp_items=trainer.items,
        leads=leads,
        terrain=terrain,
    )


def battle_terrain(trainer: Trainer, given: int | None) -> int:
    """The traced battle's terrain: ``given`` (a ``TERRAIN_NAMES`` id, from the bundle's
    ``terrain``: the trace does not record it, and a trainer whose maps give several
    terrains, ``Trainer.terrains``, fights on the one where the battle started), else the
    trainer's ``terrain``. ``TraceError`` for a ``given`` terrain the trainer's maps cannot
    give."""
    from nuzline.data.trainer import TERRAIN_NAMES

    if given is None:
        return trainer.terrain
    if not 0 <= given < len(TERRAIN_NAMES):
        raise TraceError(f"terrain {given} is not a TERRAIN_* id")
    if trainer.terrains and given not in trainer.terrains:
        maps = ", ".join(TERRAIN_NAMES[t] for t in trainer.terrains if t >= 0)
        raise TraceError(f"{TERRAIN_NAMES[given]} is not a terrain of the trainer's maps ({maps})")
    return given


def _fields(record: dict) -> dict[str, np.ndarray]:
    """The ``COMPARED_FIELDS`` of a ``turn`` record, shaped and typed as in ``LAYOUT``."""
    a = {name: np.zeros(*LAYOUT[name]) for name in COMPARED_FIELDS}
    for side, data in enumerate(record["sides"]):
        if len(data["party"]) > SLOTS:
            raise TraceError(f"side {side}: {len(data['party'])} Pokemon in one party")
        for slot, mon in enumerate(data["party"]):
            for name in ("species", "level", "hp", "max_hp", "item"):
                a[name][side, slot] = mon[name]
            a["moves"][side, slot] = mon["moves"]
            a["pp"][side, slot] = mon["pp"]
            a["status"][side, slot], a["status_counter"][side, slot] = decode_status(mon["status"])
        a["active"][side] = data["active"]
        a["stages"][side] = _stage_row(data["stages"])
    a["weather"][...] = decode_weather(record["field"])
    # The tracer reads an active battler's status from the BattleMon
    # (tools/trace_battle.lua:182-183). The decomp zeroes the status only in the party copy
    # of a Pokemon with no HP (BattleController_EmitUpdatePartyMon, battle_controller.c:
    # 1481-1486), the value the engine models (step._faint_check). A battler that faints
    # at the battle's end is never replaced, so its BattleMon keeps the bits: zero them.
    fainted = a["hp"] == 0
    a["status"][fainted] = 0
    a["status_counter"][fainted] = 0
    return a


def _stage_row(raw: Sequence[int]) -> np.ndarray:
    row = np.zeros(LAYOUT["stages"][0][1], LAYOUT["stages"][1])
    for i, stat in enumerate(STAGE_ORDER):
        if stat != Stat.HP:
            row[stat] = raw[i] - STAGE_NEUTRAL
    return row


def _overlay(state: BattleState, fields: dict[str, np.ndarray]) -> BattleState:
    return state._replace(**{name: jnp.asarray(value) for name, value in fields.items()})


def _check_start(state: BattleState, fields: dict[str, np.ndarray]) -> None:
    for name in _START_FIELDS:
        built, traced = np.asarray(getattr(state, name)), fields[name]
        diff = np.argwhere(built != traced)
        if diff.size:
            side, slot = (int(i) for i in diff[0][:2])
            who = "the party file" if side == 0 else "the trainer data"
            raise TraceMismatch(
                f"turn 0: {name} of side {side} slot {slot} is {traced[side, slot].tolist()} "
                f"in the trace but {built[side, slot].tolist()} in {who}"
            )


def _party_of(state: BattleState, b: int) -> int:
    """The party index (the leading axis of the per-party fields) battler ``b`` sends from."""
    return int(party(format_of(state), b))


def _hit_state(state: BattleState, record: dict) -> BattleState:
    """The turn's start state with a ``hit`` record's active slots, HP, status and
    stages written over it."""
    active = np.asarray(record["active"], LAYOUT["active"][1])
    hp, status = np.asarray(state.hp).copy(), np.asarray(state.status).copy()
    counter, stages = np.asarray(state.status_counter).copy(), np.asarray(state.stages).copy()
    for b in range(format_of(state).B):
        p = _party_of(state, b)
        hp[p, active[b]] = record["hp"][b]
        status[p, active[b]], counter[p, active[b]] = decode_status(record["status"][b])
        stages[b] = _stage_row(record["stages"][b])
    return state._replace(
        active=jnp.asarray(active),
        hp=jnp.asarray(hp),
        status=jnp.asarray(status),
        status_counter=jnp.asarray(counter),
        stages=jnp.asarray(stages),
    )


@jax.jit
def _damage_rows(tables, state, side, move, crit, power_mul, present=PRESENT_TABLE):
    """The 16 rolls' damage; ``present`` is Present's drawn power (``Outcome.present``)."""
    rolls = jnp.arange(16, dtype=_I32)
    return jax.vmap(
        lambda r: damage_for_roll(
            tables, state, side, move, crit, r, power_mul=power_mul, present=present
        )
    )(rolls)


@jax.jit
def _confusion_rows(tables, state, side):
    return jax.vmap(lambda r: confusion_damage(tables, state, side, r))(jnp.arange(16))


def learn_array(choices: Sequence[int]) -> jnp.ndarray:
    """``int32[MAX_LEARNS]``: a turn's ``learn_choices`` padded with -1 (no answer)."""
    if len(choices) > MAX_LEARNS:
        raise TraceError(f"{len(choices)} learn-move prompts in one turn (at most {MAX_LEARNS})")
    return jnp.asarray([*choices, *[-1] * (MAX_LEARNS - len(choices))], _I32)


def apply_turn(
    tables, state, actions, outcome, player_slot, opp_slot, learn_choices=()
) -> BattleState:
    """One traced turn: ``step``; the learnset scans of the levels gained
    (``levelup.advance_learns``) with ``learn_choices`` answering the learn-move prompts in
    order (``levelup.learn``; a tuple, or ``learn_array``'s padded array, -1 = no answer);
    then the replacement of each side's fainted active Pokemon (``needs_replacement`` after
    the step) by ``replace`` into the recorded slot, the player's (``player_slot``) before
    the opponent's (``opp_slot``); a slot below 0 is no replacement. Host or jit (the slots
    and choices may be traced).

    The decomp's ``BattleControllerPlayer_TurnEnd`` runs the exp payout, the battle-over
    check and then ``ReplaceFainted`` (battle_controller_player.c:1846-1848), whose loop
    takes the battlers in order, the player's (battler 0) first (:4078-4130). Each
    replacement runs ``subscript_show_party_list.s``: the send-out, the entry hazards and
    a faint check (:18-24), which ``replace`` models (step.py:1514-1536). A player faint
    is not terminal (``needs_replacement``, step.py:1504-1511). The learnset work the
    engine defers out of ``step`` comes before the replacements (``nuzline.engine.levelup``
    module docstring)."""
    if not isinstance(learn_choices, jnp.ndarray):
        learn_choices = learn_array(learn_choices)
    return _apply_turn(tables, state, actions, outcome, player_slot, opp_slot, learn_choices)[0]


def _answer_prompts(tables, state, choices):
    """``advance_learns``, then each answer of ``choices`` (``int32[MAX_LEARNS]``, -1 none)
    given to the pending prompt (``levelup.learn``) while one is pending."""
    state = advance_learns(tables, state)

    def answer(s, c):
        r = learn(tables, s, jnp.maximum(c, 0))
        do = (c >= 0) & needs_learn(s)
        return jax.tree.map(lambda x, y: jnp.where(do, x, y), r, s), None

    return jax.lax.scan(answer, state, choices)[0]


def _apply_turn(tables, state, actions, outcome, player_slot, opp_slot, choices):
    """``(apply_turn, needs_replacement(...) != 0 after the step (bool[B]), the state after
    the learn answers and before the replacements)``."""
    s = step(tables, state, jnp.asarray(actions, _I32), outcome)
    need = needs_replacement(s) != 0
    s = learned = _answer_prompts(tables, s, choices)
    for side, slot in ((0, player_slot), (1, opp_slot)):
        slot = jnp.asarray(slot, _I32)
        r = replace(tables, s, side, jnp.maximum(slot, 0))
        do = need[side] & (slot >= 0)
        s = jax.tree.map(lambda x, y, do=do: jnp.where(do, x, y), r, s)
    return s, need, learned


@jax.jit
def _replay_chunk(tables, state, qcs, actions, outcomes, player_slot, opp_slot, choices):
    def one(q, a, o):
        s = state._replace(qc_roll=q)
        return _apply_turn(tables, s, a, o, player_slot, opp_slot, choices)

    return jax.vmap(one)(qcs, actions, outcomes)


def _replay(tables, state, qcs, actions, outcomes, player_slot, opp_slot, choices):
    """``_apply_turn`` for each candidate from ``state`` with that candidate's start
    ``qc_roll`` (``qcs``), all with the learn answers ``choices``. The candidates go in
    ``_CHUNK``-row calls of ``_replay_chunk`` (fewer rows padded with copies of the first,
    dropped: a row's result does not depend on the others), so a one-candidate replay (a
    learn fit, a Quick Claw redraw, ``replay_turn``) with the same dtypes runs the chunk's
    program rather than a one-row ``step`` program (which keeps over 2 GB resident).
    ``ValueError`` with no candidate."""
    n = len(qcs)
    if n == 0:
        raise ValueError("_replay: no candidate to replay")
    rows = (qcs, actions, outcomes)
    if n % _CHUNK:
        pad = np.concatenate([np.arange(n), np.zeros(_CHUNK - n % _CHUNK, np.int64)])
        rows = jax.tree.map(lambda x: jnp.asarray(x)[pad], rows)
    if len(rows[0]) == _CHUNK:
        out = _replay_chunk(tables, state, *rows, player_slot, opp_slot, choices)
        return out if n == _CHUNK else jax.tree.map(lambda x: x[:n], out)
    parts = [
        _replay_chunk(
            tables,
            state,
            *jax.tree.map(lambda x, i=i: x[i : i + _CHUNK], rows),
            player_slot,
            opp_slot,
            choices,
        )
        for i in range(0, len(rows[0]), _CHUNK)
    ]
    return jax.tree.map(lambda *xs: jnp.concatenate(xs)[:n], *parts)


def replay_turn(tables, state, actions, outcome, player_slot, opp_slot, learn_choices=()):
    """``apply_turn``'s state, computed as one ``_replay`` row: ``qc_roll`` as int8, the
    actions and slots as int32 and each ``Outcome`` field in its engine dtype, the dtypes the
    converter's candidate calls pass, so a test that converts a trace and replays it runs
    the converter's program instead of compiling a ``jit(apply_turn)`` beside it."""
    if not isinstance(learn_choices, jnp.ndarray):
        learn_choices = learn_array(learn_choices)
    out = _replay(
        tables,
        state,
        jnp.asarray(state.qc_roll, _I8)[None],
        jnp.asarray(actions, _I32)[None],
        jax.tree.map(lambda x: x[None], _engine_outcome(outcome)),
        jnp.asarray(player_slot, _I32),
        jnp.asarray(opp_slot, _I32),
        learn_choices,
    )
    return jax.tree.map(lambda x: x[0], out[0])


def _holds_quick_claw(tables: Tables, state: BattleState, side: int, mon: int) -> bool:
    item = int(state.item[_party_of(state, side), mon])
    return int(tables.item_effect[item]) == ItemId.QUICK_CLAW


def _mismatches(
    result: BattleState, after: dict[str, np.ndarray], open_learns: bool = False
) -> np.ndarray:
    """``bool[N, len(COMPARED_FIELDS)]``: where each replayed row differs from ``after``.
    With ``open_learns`` (the per-turn fit, which replays no learn answer; ``_fit_learns``
    then fits them), the moves and PP of a player slot with learnset work left are not
    compared."""
    wrong = np.zeros((len(result.hp), len(COMPARED_FIELDS)), bool)
    pending = np.asarray(result.learn_level) > 0  # [N, SLOTS]
    for f, name in enumerate(COMPARED_FIELDS):
        got = np.asarray(getattr(result, name))
        if open_learns and name in ("moves", "pp") and pending.any():
            open_ = np.zeros(got.shape, bool)
            open_[:, 0] = pending[:, :, None]
            got = np.where(open_, after[name][None], got)
        axes = tuple(range(1, got.ndim))
        wrong[:, f] = ~np.all(got == after[name][None], axis=axes)
    return wrong


class _Turn:
    """Inference for one turn: ``state`` before it, ``after`` the next record's fields,
    ``events`` the records in between."""

    def __init__(self, tables, state, after, events, index):
        self.tables, self.state, self.after, self.index = tables, state, after, index
        self.events = events
        self.candidates: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._candidate_list: list | None = None
        self._ties: list[_Fit] | None = None
        self.chosen = -1  # the ``candidate_list`` index of the accepted fit

    def fail(self, message: str) -> TraceMismatch:
        return TraceMismatch(f"turn {self.index}: {message}")

    def _used_slots(self, side: int) -> np.ndarray:
        """The move slots of the side's start-of-turn active Pokemon whose PP dropped. A slot
        whose move a learn-move prompt replaced this turn (its move differs in the next
        record) shows the new move's PP, not a drop: it counts as used when the side's
        ``hit`` or ``move_end`` records name its start move."""
        mon, p = int(self.state.active[side]), _party_of(self.state, side)
        start = np.asarray(self.state.moves[p, mon])
        same = start == self.after["moves"][p, mon]
        drop = np.asarray(self.state.pp[p, mon], np.int32) - self.after["pp"][p, mon]
        used = np.flatnonzero((drop > 0) & same)
        if used.size == 0 and not same.all():
            named = {
                e.get("move")
                for e in self.events
                if e["kind"] in ("hit", "move_end") and e["attacker"] == side
            }
            used = np.asarray([k for k in np.flatnonzero(~same) if int(start[k]) in named], int)
        if used.size > 1:
            raise self.fail(f"side {side}: PP dropped in move slots {used.tolist()}")
        return used

    def _roared(self, side: int) -> bool:
        """The foe used Roar / Whirlwind this turn (a PP drop on one), so the side's active
        Pokemon may have been dragged out rather than switched."""
        foe = foe_of(side)
        used = self._used_slots(foe)
        if used.size == 0:
            return False
        p = _party_of(self.state, foe)
        move = int(self.state.moves[p, int(self.state.active[foe]), int(used[0])])
        return int(self.tables.move_effect[move]) == EffectId.ROAR

    def action_choices(self, side: int) -> list[int]:
        state, after = self.state, self.after
        mon = int(state.active[side])
        used = self._used_slots(side)
        if used.size == 1:
            return [int(used[0])]
        legal = np.asarray(legal_actions(self.tables, state, side))
        roared = self._roared(side)
        alive = int(after["hp"][_party_of(state, side), mon]) > 0
        if int(after["active"][side]) != mon and alive and not roared:
            return [MOVE_ACTIONS + int(after["active"][side])]
        if roared:
            # A foe's Roar drags the side's Pokemon out (step._use_move) and the newcomer does
            # not act: a move that never ran, or a switch the drag undid, possibly dragging the
            # start Pokemon back so that the active slot looks unchanged. Every legal action.
            choices = [int(a) for a in np.flatnonzero(legal)]
        else:
            # It did not get to use a move: every legal move.
            choices = [int(a) for a in np.flatnonzero(legal[:MOVE_ACTIONS])]
        if side == 1:  # and the trainer's bag
            items = np.asarray(usable_trainer_items(self.tables, state, 1))
            choices += [FIRST_ITEM_ACTION + int(k) for k in np.flatnonzero(items)]
        return choices

    def fit(self, side: int, hit: int, record: dict, table: dict[int, list[int]]):
        """The ``(crit, roll)`` of a hit whose damage by roll is ``table[crit]``."""
        cands = [
            (crit, r)
            for crit, damages in table.items()
            for r, damage in enumerate(damages)
            if damage == record["damage"]
        ]
        if not cands:
            raise self.fail(
                f"side {side} hit {hit} with move {record['move']} did damage "
                f"{record['damage']}, which fits no roll of the engine's table {table}"
            )
        self.candidates[(side, hit)] = cands
        try:
            return resolve_roll(cands)
        except AmbiguousTrace as err:
            raise AmbiguousTrace(
                f"turn {self.index}: side {side} hit {hit}: both a crit and a non-crit roll fit",
                err.candidates,
            ) from None

    def _protect_roll(self, o: dict, side: int, action: int) -> None:
        """``o["protect"][side]`` for a Protect / Detect action: BtlCmd_TryProtection
        (battle_script.c:5167-5186) sets MOVE_STATUS_FAILED on a failure, which the move's
        ``move_end`` record keeps; a success does not (1 replays it; a chain of 0 or a user
        acting last ignores the roll). tools/trace_battle.lua writes a ``move_end`` at every
        MOVE_END with the context's ``attacker`` and ``moveCur``, which a switch or a bag item
        leaves stale, so only records naming this side and this move count. A side that never
        reached the move (flinched, fully paralyzed) has none: the roll stays unread. Several
        that disagree on the flag raise."""
        move = int(used_move(self.tables, self.state, side, action))
        if int(self.tables.move_effect[move]) != EffectId.PROTECT:
            return
        ends = [
            e
            for e in self.events
            if e["kind"] == "move_end" and e["attacker"] == side and e.get("move") == move
        ]
        failed = {bool(e["flags"] & MOVE_STATUS_FAILED) for e in ends}
        if len(failed) > 1:
            raise self.fail(
                f"side {side}'s Protect matches {len(ends)} move_end records that disagree on "
                "MOVE_STATUS_FAILED"
            )
        if failed:
            o["protect"][side] = 0 if failed.pop() else 1

    def damage_outcome(self, actions: list[list[int]]) -> dict[str, np.ndarray]:
        """``empty_outcome`` with the hits, crits and rolls of the trace's records."""
        o = {name: np.asarray(v).copy() for name, v in empty_outcome()._asdict().items()}
        for side in range(format_of(self.state).B):
            hits = [
                e
                for e in self.events
                if e["kind"] == "hit" and e["attacker"] == side and e["defender"] == foe_of(side)
            ]
            self_hits = [
                e
                for e in self.events
                if e["kind"] == "hit" and e["attacker"] == side and e["defender"] == side
            ]
            missed = any(
                e["kind"] == "move_end"
                and e["attacker"] == side
                and e["flags"] & MOVE_STATUS_MISSED
                for e in self.events
            )
            known_move = len(actions[side]) == 1 and actions[side][0] < MOVE_ACTIONS
            if (hits or missed) and not known_move:
                raise self.fail(f"side {side} has move records but used no PP")
            if missed:
                o["hit"][side] = 0
            if known_move:
                self._protect_roll(o, side, actions[side][0])
            if hits:
                move = int(used_move(self.tables, self.state, side, actions[side][0]))
                o["n_hits"][side] = min(max(len(hits), 2), MAX_HITS)
            for i, e in enumerate(hits[:MAX_HITS]):
                if e["move"] != move:
                    raise self.fail(f"side {side} hit {i} is move {e['move']}, chose {move}")
                hs = _hit_state(self.state, e)
                other = actions[foe_of(side)]
                pursuit = (
                    int(self.tables.move_effect[move]) == EffectId.PURSUIT
                    and len(other) == 1
                    and MOVE_ACTIONS <= other[0] < FIRST_ITEM_ACTION
                    and e["active"][foe_of(side)] == int(self.state.active[foe_of(side)])
                )
                power_mul = 20 if pursuit else 10
                crits = (0, 1) if e["crit"] is None else (int(e["crit"]),)
                # Present: a hit record is its damage path; its power 40/80/120 is drawn
                # (Outcome.present 0-2), so the lowest power with a fitting roll is used
                # (equal damage gives the same state). Its heal leaves no hit record
                # (``variants``).
                present = int(self.tables.move_effect[move]) == EffectId.PRESENT
                for k in range(PRESENT_HEAL) if present else (PRESENT_TABLE,):
                    table = {
                        crit: np.asarray(
                            _damage_rows(self.tables, hs, side, move, crit != 0, power_mul, k)
                        ).tolist()
                        for crit in crits
                    }
                    fits = any(e["damage"] in rows for rows in table.values())
                    if fits or k in (PRESENT_TABLE, PRESENT_HEAL - 1):
                        break
                if present:
                    o["present"][side] = k
                o["crit"][side, i], o["roll"][side, i] = self.fit(side, i, e, table)
            if self_hits:
                hs = _hit_state(self.state, self_hits[0])
                o["confusion_hit"][side] = 1
                table = {0: np.asarray(_confusion_rows(self.tables, hs, side)).tolist()}
                _, o["roll"][side, 0] = self.fit(side, 0, self_hits[0], table)
        return o

    def variants(self, actions: tuple[int, int], base: dict) -> list[dict]:
        """Every value of the chance fields the trace does not record that can matter
        in this state with these actions (docs/tracing.md)."""
        tables, state = self.tables, self.state
        options: list[tuple[str, tuple, list]] = []  # (field, index, values)
        if _tied(tables, state):
            options.append(("speed_tie", (), [0, 1]))
        if any(_tied(tables, s) for s in self._end_of_turn_states(actions)):
            options.append(("end_speed_tie", (), [0, 1]))
        connected = [
            bool(base["hit"][s, 0]) and self._acted(s) for s in range(format_of(self.state).B)
        ]
        fixed: list[tuple[str, tuple, int]] = []
        for side in range(format_of(self.state).B):
            # This turn's speedRand bit orders the end of the turn (step.end_of_turn_order)
            # when the battler on the field then holds a Quick Claw; the next turn's fit of
            # its start qc_roll can still rewrite it (convert_bundle).
            end_mon = int(state.active[side])
            if MOVE_ACTIONS <= actions[side] < FIRST_ITEM_ACTION:
                end_mon = actions[side] - MOVE_ACTIONS
            if _holds_quick_claw(tables, state, side, end_mon):
                options.append(("quick_claw", (side,), [0, 1]))
            statuses = self._statuses(side)
            if Status.PARALYSIS in statuses:
                options.append(("full_para", (side,), [0, 1]))
            if Status.FREEZE in statuses:
                options.append(("thaw", (side,), [0, 1]))
            if self._may_be_in_love(side, actions):
                options.append(("infatuation", (side,), [0, 1]))
            if actions[side] >= MOVE_ACTIONS or not connected[side]:
                continue
            move = int(used_move(tables, state, side, actions[side]))
            effect = int(tables.move_effect[move])
            chance = int(read_chance(tables, move))
            per_hit = (side, slice(None))
            if chance == 0:
                fixed.append(("secondary", per_hit, 1))  # applies on every hit
            else:
                options.append(("secondary", per_hit, [0, 1]))
            hit_record = any(
                e["kind"] == "hit" and e["attacker"] == side and e["defender"] == foe_of(side)
                for e in self.events
            )
            if effect == EffectId.PRESENT and not hit_record:
                # Present connected with no hit record: its heal (no UPDATE_HP damage), or a
                # damage path into an immunity; the replay matches the target's HP gain.
                options.append(("present", (side,), [0, PRESENT_HEAL]))
            if effect == EffectId.TRI_ATTACK:
                # Tri Attack's status (Outcome.tri_attack): shows only as the target's status,
                # read only when the secondary procs. Without a proc every value replays to the
                # same state, so the first candidate (0) is kept: the recorded value is then 0
                # and means nothing (ties go to the first candidate; ``secondary`` is listed
                # first, so each of its values meets 0 before 1 and 2).
                options.append(("tri_attack", (side,), [0, 1, 2]))
            if effect == EffectId.ROAR:
                # Roar's level check (a user below its target's level) and its pick among the
                # target side's eligible bench (Outcome.roar_level, roar_target): they show as
                # the foe's new active Pokemon, or none on a failure.
                options += self._roar_options(side, actions)
            if effect == EffectId.FANG_PARALYZE_FLINCH:
                # Thunder Fang's second roll (Outcome.fang_flinch): a flinch shows only as the
                # target's missing move.
                options.append(("fang_flinch", (side,), [0, 1]))
            # The contact abilities with a roll (effects._contact_ability) of the defender:
            # the target's start-of-turn active (Pursuit hits the outgoing one) or the
            # Pokemon it switches in.
            target = foe_of(side)
            defenders = [int(state.active[target])]
            if MOVE_ACTIONS <= actions[target] < FIRST_ITEM_ACTION:
                defenders.append(actions[target] - MOVE_ACTIONS)
            rolled = (
                AbilityId.STATIC,
                AbilityId.POISON_POINT,
                AbilityId.FLAME_BODY,
                AbilityId.CUTE_CHARM,
            )
            target_party = _party_of(state, target)
            if any(int(state.ability[target_party, mon]) in rolled for mon in defenders):
                options.append(("contact_proc", (target, slice(None)), [0, 1]))
            sleep = (
                effect in (EffectId.STATUS_FOE, EffectId.SECONDARY_STATUS)
                and int(tables.move_param[move, 0]) == Status.SLEEP
            )
            if effect == EffectId.TAUNT:
                options.append(("duration", (side,), [3, 4, 5]))
            elif effect == EffectId.TRAP:
                options.append(("duration", (side,), [3, 4, 5, 6]))
            elif sleep or effect in (EffectId.CONFUSE_FOE, EffectId.SECONDARY_CONFUSE):
                # A secondary confusion runs the same subscript_confuse.s, whose
                # ``Random 3, 2`` (:34) draws rand % 4 + 2 (BtlCmd_Random, bs.c:3200-3207).
                # It may show only on a later turn, as a self-hit (convert_bundle's refit).
                options.append(("duration", (side,), [2, 3, 4, 5]))
        out = []
        for values in itertools.product(*(vals for _, _, vals in options)):
            o = {k: v.copy() for k, v in base.items()}
            for name, index, value in fixed:
                o[name][index] = value
            for (name, index, _), value in zip(options, values):
                o[name][index] = value
            out.append(o)
        return out

    def _roar_options(self, side: int, actions: tuple[int, int]) -> list[tuple[str, tuple, list]]:
        """The ``roar_level`` / ``roar_target`` values to replay for the side's Roar: its target
        is the foe's active Pokemon or the one the foe switches in (a switch goes first)."""
        state, foe = self.state, foe_of(side)
        target = int(state.active[foe])
        if MOVE_ACTIONS <= actions[foe] < FIRST_ITEM_ACTION:
            target = actions[foe] - MOVE_ACTIONS
        own, foes_party = _party_of(state, side), _party_of(state, foe)
        bench = [
            k
            for k in range(SLOTS)
            if k != target and int(state.species[foes_party, k]) and int(state.hp[foes_party, k])
        ]
        out: list[tuple[str, tuple, list]] = []
        if int(state.level[own, int(state.active[side])]) < int(state.level[foes_party, target]):
            out.append(("roar_level", (side,), [1, 0]))
        if len(bench) >= 2:
            out.append(("roar_target", (side,), list(range(len(bench)))))
        return out

    def _statuses(self, side: int) -> set[int]:
        """The statuses the side's start-of-turn active mon has at any point the trace
        shows: at the start, in a ``hit`` record and in the next record. A status
        inflicted before the mon moves (Stun Spore, Static) needs its ``full_para`` or
        ``thaw`` draw on the same turn."""
        mon, p = int(self.state.active[side]), _party_of(self.state, side)
        statuses = {int(self.state.status[p, mon]), int(self.after["status"][p, mon])}
        for e in self.events:
            if e["kind"] == "hit" and e["active"][side] == mon:
                statuses.add(int(decode_status(e["status"][side])[0]))
        return statuses

    def _may_be_in_love(self, side: int, actions: tuple[int, int]) -> bool:
        """The side uses a move and its attract check can run (CHECK_STATUS_STATE_ATTRACT,
        battle_controller_player.c:2669-2692): it is in love at the start (``infatuated``,
        carried by the engine: the trace has no volatile bits) or its foe uses Attract."""
        if actions[side] >= MOVE_ACTIONS:
            return False
        if int(self.state.infatuated[side]) != 0:
            return True
        foe = foe_of(side)
        if actions[foe] >= MOVE_ACTIONS:
            return False
        move = int(used_move(self.tables, self.state, foe, actions[foe]))
        return int(self.tables.move_effect[move]) == EffectId.ATTRACT

    def _end_of_turn_states(self, actions: tuple[int, int]) -> list[BattleState]:
        """States whose speeds the end-of-turn order may compare: the start state, with
        the switches made, and with the next record's status and stages over that."""
        active = np.asarray(self.state.active).copy()
        for side in range(format_of(self.state).B):
            if MOVE_ACTIONS <= actions[side] < FIRST_ITEM_ACTION:
                active[side] = actions[side] - MOVE_ACTIONS
        switched = self.state._replace(active=jnp.asarray(active))
        shown = {name: self.after[name] for name in ("status", "stages")}
        # The Pokemon on the field in the next record: a Roar's drag (or a replacement).
        ended = self.state._replace(active=jnp.asarray(self.after["active"]))
        return [self.state, switched, _overlay(switched, shown), _overlay(ended, shown)]

    def _acted(self, side: int) -> bool:
        return any(e["kind"] == "move_end" and e["attacker"] == side for e in self.events) or any(
            e["kind"] == "hit" and e["attacker"] == side for e in self.events
        )

    def qc_variants(self) -> list[tuple[int, int]]:
        """The start ``qc_roll`` values to replay: for a side whose active Pokemon holds a
        Quick Claw both bits, the carried value first (so ties keep it); else 0."""
        out = []
        for side in range(format_of(self.state).B):
            carried = int(self.state.qc_roll[side])
            if _holds_quick_claw(self.tables, self.state, side, int(self.state.active[side])):
                out.append([carried, 1 - carried])
            else:
                out.append([0])
        return [tuple(q) for q in itertools.product(*out)]

    def candidate_list(self) -> list[tuple[tuple, tuple[int, int], dict]]:
        """Every ``(start qc_roll, actions, outcome fields)`` to replay, in tie order."""
        if self._candidate_list is None:
            choices = [self.action_choices(0), self.action_choices(1)]
            base = self.damage_outcome(choices)
            self._candidate_list = [
                (qc, pair, o)
                for qc in self.qc_variants()
                for pair in itertools.product(*choices)
                for o in self.variants(pair, base)
            ]
            if len(self._candidate_list) > _MAX_CANDIDATES:
                raise TraceError(
                    f"turn {self.index}: {len(self._candidate_list)} candidates to replay"
                )
        return self._candidate_list

    def _replays(self):
        """Per chunk of ``candidate_list``: its start index, the chunk, the replayed states,
        ``needs_replacement`` after the step and the mismatched-field rows."""
        candidates = self.candidate_list()
        slots = [
            jnp.asarray(int(self.after["active"][side]), _I32)
            for side in range(format_of(self.state).B)
        ]
        for start in range(0, len(candidates), _CHUNK):
            chunk = candidates[start : start + _CHUNK]
            padded = chunk + [chunk[0]] * (_CHUNK - len(chunk))
            qcs = jnp.asarray([qc for qc, _, _ in padded], _I8)
            actions = jnp.asarray([pair for _, pair, _ in padded], _I32)
            outcomes = _engine_outcome(
                np.stack([o[name] for _, _, o in padded]) for name in Outcome._fields
            )
            result, need, _ = _replay(
                self.tables, self.state, qcs, actions, outcomes, *slots, _NO_LEARNS
            )
            wrong = _mismatches(result, self.after, open_learns=True)
            yield start, chunk, result, np.asarray(need), wrong

    def _fit(self, start, chunk, result, need, wrong, k) -> _Fit:
        qc, pair, outcome = chunk[k]
        player_need = bool(need[k, 0])
        return _Fit(
            index=start + k,
            qc=tuple(int(q) for q in qc),
            actions=(int(pair[0]), int(pair[1])),
            outcome=outcome,
            state=jax.tree.map(lambda x: x[k], result),
            mismatched=tuple(n for n, w in zip(COMPARED_FIELDS, wrong[k]) if w),
            need=(player_need, bool(need[k, 1])),
            player_slot=int(self.after["active"][0]) if player_need else -1,
        )

    def infer(self) -> _Fit:
        """The best candidate: the first with the fewest mismatched fields."""
        best = None
        for start, chunk, result, need, wrong in self._replays():
            score = wrong.sum(axis=1)[: len(chunk)]
            k = int(np.argmin(score))
            if best is None or score[k] < best[0]:
                best = (int(score[k]), self._fit(start, chunk, result, need, wrong, k))
            if best[0] == 0:
                break
        return best[1]

    def ties(self) -> list[_Fit]:
        """Every candidate that matches the next record on every compared field, in
        ``candidate_list`` order (cached: the start state and the record do not change)."""
        if self._ties is None:
            self._ties = []
            for start, chunk, result, need, wrong in self._replays():
                for k in np.flatnonzero(~wrong[: len(chunk)].any(axis=1)):
                    self._ties.append(self._fit(start, chunk, result, need, wrong, int(k)))
        return self._ties

    def unexplained_self_hit(self) -> bool:
        """A confusion self-hit record (attacker == defender) of a side that is not confused
        at the turn's start: its confusion was drawn on an earlier turn and not fitted."""
        return any(
            e["kind"] == "hit"
            and e["attacker"] == e["defender"]
            and int(self.state.confusion[e["attacker"]]) == 0
            for e in self.events
        )


def _tied(tables: Tables, state: BattleState) -> bool:
    return int(effective_speed(tables, state, 0)) == int(effective_speed(tables, state, 1))


def _records(bundle: dict) -> tuple[list[dict], list[list[dict]]]:
    turns, events = [], []
    for record in bundle["records"]:
        if record["kind"] == "turn":
            turns.append(record)
            events.append([])
        elif not turns:
            raise TraceError(f"a {record['kind']!r} record comes before the first turn record")
        elif record["kind"] in ("hit", "move_end"):
            events[-1].append(record)
        else:
            raise TraceError(f"unknown record kind {record['kind']!r}")
    if not turns:
        raise TraceError("the trace has no turn record")
    return turns, events


def _spec(d: dict) -> PokemonSpec:
    return PokemonSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def _trainer(trainer_id: int, game: str = VANILLA) -> Trainer:
    from nuzline.cli import resolve_trainer
    from nuzline.data.trainer import load_trainers

    trainers = load_trainers(get_game(game).intermediate_dir)
    return resolve_trainer(trainers, trainer_id=trainer_id)


def convert_bundle(bundle: dict, tables: Tables, names: Names) -> list[Turn]:
    """Every turn of a bundle (module docstring). The last ``Turn`` is the final record,
    with actions ``NO_ACTIONS``. A player faint is followed by the player's replacement
    into the next record's active slot (``apply_turn``); the trace ends at the battle's
    end. A turn the best candidate does not explain is refitted from an earlier turn's
    tied candidates (``_refit``)."""
    del names  # ids throughout; kept for the plan's signature
    if bundle.get("format") not in BUNDLE_FORMATS:
        raise TraceError(f"not a {' or '.join(BUNDLE_FORMATS)} bundle")
    game = bundle.get("game", VANILLA)
    trainer = _trainer(bundle["trainer_id"], game)
    player = [_spec(d) for d in bundle["party"]]
    records, events = _records(bundle)
    first = _fields(records[0])
    leads = (int(first["active"][0]), int(first["active"][1]))
    try:
        state = initial_state(tables, player, trainer, leads, game, bundle.get("terrain"))
    except ValueError as err:  # make_state or the opponent's party build (e.g. a bad exp)
        if isinstance(err, TraceError):
            raise
        raise TraceError(f"the turn-0 state (the bundle's party or the trainer's): {err}") from None
    _check_start(state, first)
    state = _overlay(state, first)
    turns: list[Turn] = []
    fits: list[_Turn] = []
    for i in range(len(records) - 1):
        inference = _Turn(tables, state, _fields(records[i + 1]), events[i], i)
        state = _accept(tables, turns, fits, inference, inference.infer())
        if not _explained(turns[-1], inference):
            state = _refit(tables, turns, fits) or state
        if int(state.over):
            break
    turns.append(Turn(state, NO_ACTIONS, empty_outcome(), {}, ()))
    return turns


_LOOK_BACK = 5  # turns a refit may rewind


def _explained(turn: Turn, inference: _Turn) -> bool:
    """The turn's replay matches the next record and no self-hit lacks its confusion."""
    return turn.mismatched == () and not inference.unexplained_self_hit()


def _accept(tables: Tables, turns: list[Turn], fits: list[_Turn], inference: _Turn, fit: _Fit):
    """Append ``fit`` as the turn ``inference`` describes; return the next turn's start
    state. A changed start ``qc_roll`` is the previous turn's draw (or the battle start's),
    written back into that turn (``_redraw_quick_claw``). ``TraceError`` for a turn the
    replay with one replacement per side cannot give: a double KO, or a player Pokemon at
    0 HP in the record that the replay keeps alive (a newcomer that fainted on entry)."""
    state, after, i = inference.state, inference.after, inference.index
    fit = _fit_learns(tables, inference, fit)
    if int(fit.state.exp_refused):
        raise TraceError(
            f"turn {i}: a payout inside the opponent's replacement left learn-move work "
            "(BattleState.exp_refused, design §3.3 window (c))"
        )
    if fit.need[0] and fit.need[1]:
        raise TraceError(f"turn {i}: both sides need a replacement (a double KO)")
    if fit.need[0]:
        alive = np.flatnonzero((np.asarray(fit.state.hp[0]) > 0) & (after["hp"][0] == 0))
        if alive.size:
            raise TraceError(
                f"turn {i}: player slot {int(alive[0])} has 0 HP in the record but not in "
                f"the replay with the replacement into slot {fit.player_slot}: a newcomer "
                "that fainted on entry (two replacements in one turn) is not replayed"
            )
    if fit.qc != tuple(int(q) for q in np.asarray(state.qc_roll)):
        state = state._replace(qc_roll=jnp.asarray(fit.qc, _I8))
        if turns:
            turns[-1] = _redraw_quick_claw(tables, turns[-1], fits[-1].after, fit.qc)
    inference.chosen = fit.index
    outcome = _engine_outcome(fit.outcome[name] for name in Outcome._fields)
    turns.append(
        Turn(
            state,
            fit.actions,
            outcome,
            inference.candidates,
            fit.mismatched,
            fit.player_slot,
            fit.learn_choices,
        )
    )
    fits.append(inference)
    return _overlay(fit.state, after)


def _fit_learns(tables: Tables, inference: _Turn, fit: _Fit) -> _Fit:
    """``fit`` with its learn-move prompts answered (design §5.4, phase 2). The per-turn fit
    replays no answer and leaves the moves and PP of a slot with a pending prompt
    uncompared (``_mismatches``); here, from that candidate, the answers are searched depth
    first, prompt by prompt in the engine's order (``Turn.learn_choices``), each over giving
    up and every move slot whose move is not an HM (the game's only rule:
    ``CheckSelectedMoveIsHM``, src/battle_sub_menus/battle_party.c:1724-1735;
    ``Item_IsHMMove``, src/item.c:306-315), not ``learn_options``' search masks. The first
    sequence whose replay matches the next record on every compared field wins (else the
    one with the fewest mismatched fields): giving up keeps the moves (bs.c:10132-10135), a
    replacement puts the new move in the slot at its base PP (bs.c:10169-10171), and either
    way the scan goes on (bs.c:10183-10187), possibly to another prompt (bs.c:10007-10018).
    Answers that give the same moves and PP replay the same records, so the first is kept.

    ``levelup.learn`` applies every answer, but after a mid-turn payout it is exact only for
    the answers ``learn_options`` offers (``levelup``'s module docstring). ``TraceError`` for
    an answer it masks (``_learn_refusal``), and for a move the engine does not support
    learned by an answer or into an empty slot (ruling D10)."""
    if not bool(needs_learn(fit.state)):
        _check_learned(tables, inference, fit.state, ())
        return fit
    after = inference.after
    qcs = jnp.asarray([fit.qc], _I8)
    actions = jnp.asarray([fit.actions], _I32)
    outcome = _engine_outcome(fit.outcome[n][None] for n in Outcome._fields)
    slots = (jnp.asarray(fit.player_slot, _I32), jnp.asarray(int(after["active"][1]), _I32))

    def run(choices):
        out = _replay(tables, inference.state, qcs, actions, outcome, *slots, learn_array(choices))
        return jax.tree.map(lambda x: x[0], out)

    hm = np.asarray(tables.move_is_hm)
    best: list = []

    def search(prefix: tuple[int, ...]) -> bool:
        final, need, learned = run(prefix)
        if bool(needs_learn(learned)):
            if len(prefix) == MAX_LEARNS:
                raise inference.fail(f"more than {MAX_LEARNS} learn-move prompts")
            slot = int(np.argmax(np.asarray(learned.learn_level) > 0))
            moves = np.asarray(learned.moves[0, slot])
            options = (GIVE_UP, *(k for k in range(MOVES) if not hm[moves[k]]))
            return any(search((*prefix, c)) for c in options)
        wrong = _mismatches(jax.tree.map(lambda x: x[None], final), after)[0]
        if not best or wrong.sum() < best[0][0]:
            best[:] = [(int(wrong.sum()), prefix, final, need)]
        return not wrong.any()

    search(())
    _, choices, final, need = best[0]
    wrong = _mismatches(jax.tree.map(lambda x: x[None], final), after)[0]
    if not wrong.any():  # refuse only a path that reproduces the record; else report it
        _check_learned(tables, inference, final, choices)
        for n in range(len(choices)):
            _learn_refusal(tables, inference, run(choices[:n])[2], choices[n])
    return dataclasses.replace(
        fit,
        state=final,
        mismatched=tuple(name for name, w in zip(COMPARED_FIELDS, wrong) if w),
        need=(bool(need[0]), bool(need[1])),
        learn_choices=choices,
    )


def _learn_refusal(tables: Tables, inference: _Turn, state: BattleState, choice: int) -> None:
    """``TraceError`` when ``choice`` answers ``state``'s prompt with an option
    ``levelup.learn_options`` masks. The converter fits it (it happened), but after a mid-turn
    payout the engine does not replay it exactly (``levelup``'s module docstring), for the
    active learner:

    * the slot selected this turn (``learn_pin``, ruling U1): the game re-reads
      ``moves[moveSlot]`` when the move runs (battle_controller_player.c:1891-1895), the engine
      runs the move it selected;
    * the disabled or the encored move (ruling D2): the end-of-turn checks end Disable and
      Encore once the move is no longer known (battle_controller_player.c:1565-1576,
      :1591-1602), the engine runs them before the resolve.

    None of them once the battle is over: the payout that ended it runs no later turn end or
    move (bcp.c:3965-3969), so ``learn_options`` offers them. The locked and the choice-locked
    move are read by id and replay exactly (offered by
    ``learn_options``); a move the AI has seen keeps its id in ``ai_seen_moves``
    (``TrainerAI_RecordLastMove``, src/battle/trainer_ai/trainer_ai.c:2705-2716, G-68). The
    D10 mask (an unsupported offered move) is refused first, by ``_check_learned``."""
    if choice == GIVE_UP or bool(learn_options(tables, state)[choice]):
        return
    slot = int(np.argmax(np.asarray(state.learn_level) > 0))
    move = int(state.moves[0, slot, choice])
    if int(state.learn_pin) == choice + 1:
        why = "the slot selected this turn, re-read when its move runs (bcp.c:1891-1895)"
    elif move == int(state.disabled_move[0]):
        why = "the disabled move, whose Disable the turn's end would end (bcp.c:1565-1576)"
    else:  # the encored move: the converter fits no HM, and D10 is refused first
        why = "the encored move, whose Encore the turn's end would end (bcp.c:1591-1602)"
    raise inference.fail(
        f"the player replaced move {move} (party slot {slot}, move slot {choice}) at a "
        f"learn-move prompt: {why}"
    )


def _check_learned(
    tables: Tables, inference: _Turn, state: BattleState, choices: tuple[int, ...]
) -> None:
    """``TraceError`` when the player's party ends the turn with a move the engine does not
    support that it did not start with: learned into an empty slot or by a prompt's answer
    (ruling D10; a prompt answered by giving up is replayed)."""
    unsupported = np.asarray(tables.move_effect) == EffectId.UNSUPPORTED
    start, end = np.asarray(inference.state.moves[0]), np.asarray(state.moves[0])
    for slot in range(SLOTS):
        new = [int(m) for m in end[slot] if m and m not in start[slot] and unsupported[m]]
        if new:
            raise inference.fail(
                f"party slot {slot} learned move {new[0]}, which the engine does not support "
                f"(ruling D10; learn choices {list(choices)})"
            )


def _refit(tables: Tables, turns: list[Turn], fits: list[_Turn]) -> BattleState | None:
    """Rewind for the last turn ``j``, which its best candidate does not explain
    (``_explained``): some draws show only on a later turn (a secondary confusion's
    duration as a self-hit, a Bind's duration as a residual that stops), and each turn is
    fitted against the next record only. For ``i = j-1 .. j-5`` (latest first) and each
    tied candidate of turn ``i`` other than the chosen one (``_Turn.ties``, in candidate
    order), replay turn ``i`` with it and infer turns ``i+1 .. j`` again as usual (Quick
    Claw write-backs included). The first that explains every turn ``i .. j`` (without
    worsening turn ``i-1``) replaces ``turns``/``fits`` in place, and its next start state
    is returned; ``None`` keeps the original. Skipping ties that replay to a state already
    tried (``_fit_key``) only saves time; the no-worsening guard is a conservative
    check. Neither has its own test, nor has the five-turn bound
    (``_LOOK_BACK``, docs/tracing.md "Known limits")."""
    j = len(turns) - 1
    for i in range(j - 1, max(j - 1 - _LOOK_BACK, -1), -1):
        seen = {_fit_key(f) for f in fits[i].ties() if f.index == fits[i].chosen}
        for alt in fits[i].ties():
            key = _fit_key(alt)
            if alt.index == fits[i].chosen or key in seen:
                continue  # the chosen one, or one replaying to a state already tried
            seen.add(key)
            trial = _trial(tables, turns, fits, i, alt)
            if trial is None:
                continue
            new_turns, new_fits, state = trial
            if i and len(new_turns[i - 1].mismatched) > len(turns[i - 1].mismatched):
                continue
            turns[:], fits[:] = new_turns, new_fits
            return state
    return None


def _trial(tables: Tables, turns: list[Turn], fits: list[_Turn], i: int, alt: _Fit):
    """``_refit``'s trial of ``alt`` for turn ``i``: ``(turns, fits, next start state)``
    when the replay explains every turn ``i .. len(turns)-1``, else ``None`` (also when a
    re-inferred turn raises ``TraceError``: a trial the replay cannot give is discarded
    like one that mismatches). ``turns``/``fits`` are not changed."""
    new_turns, new_fits = turns[:i], fits[:i]
    trial = copy.copy(fits[i])
    # Turn i's start qc_roll as it stands now: an earlier refit may have rewritten it and
    # redrawn turn i-1, so ``_accept`` must compare ``alt`` with that value.
    trial.state = turns[i].state
    try:
        state = _accept(tables, new_turns, new_fits, trial, alt)
        if not _explained(new_turns[-1], trial):
            return None
        for k in range(i + 1, len(turns)):
            inference = _Turn(tables, state, fits[k].after, fits[k].events, k)
            state = _accept(tables, new_turns, new_fits, inference, inference.infer())
            if not _explained(new_turns[-1], inference):
                return None
    except TraceError:
        return None
    # A later turn's Quick Claw write-back replays (and may unexplain) the turn before it.
    if not all(_explained(t, f) for t, f in zip(new_turns[i:], new_fits[i:])):
        return None
    return new_turns, new_fits, state


def _fit_key(fit: _Fit) -> tuple:
    """Fits with equal keys replay to the same state from the same start ``qc_roll``."""
    return (fit.qc, *(np.asarray(x).tobytes() for x in jax.tree.leaves(fit.state)))


def _redraw_quick_claw(tables: Tables, turn: Turn, after: dict, qc: tuple) -> Turn:
    """``turn`` with its draw ``Outcome.quick_claw`` set to the next turn's fitted start
    ``qc_roll`` (the same ``speedRand``, bcp.c:853-857), replayed again with the turn's
    own replacements and learn choices: its ``mismatched`` is recomputed (this turn's
    end-of-turn order reads the draw)."""
    outcome = turn.outcome._replace(quick_claw=jnp.asarray(qc, _I8))
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None], outcome)
    result, _, _ = _replay(
        tables,
        turn.state,
        jnp.asarray([np.asarray(turn.state.qc_roll)], _I8),
        jnp.asarray([turn.actions], _I32),
        batch,
        jnp.asarray(turn.player_slot, _I32),
        jnp.asarray(int(after["active"][1]), _I32),
        learn_array(turn.learn_choices),
    )
    wrong = _mismatches(result, after)[0]
    mismatched = tuple(n for n, w in zip(COMPARED_FIELDS, wrong) if w)
    return dataclasses.replace(turn, outcome=outcome, mismatched=mismatched)


def convert(path: Path, tables: Tables, names: Names) -> list[Turn]:
    """``convert_bundle`` of the bundle at ``path``."""
    return convert_bundle(json.loads(Path(path).read_text(encoding="utf-8")), tables, names)


def trace_to_states(path: Path, tables: Tables, names: Names) -> list[ReplayTurn]:
    """A ``ReplayTurn`` per turn of the bundle at ``path``; the last entry is the final
    state with ``NO_ACTIONS`` and an empty outcome."""
    return [
        ReplayTurn(t.state, t.actions, t.outcome, t.player_slot, t.learn_choices)
        for t in convert(path, tables, names)
    ]


def bundle_trace(
    raw: Path,
    party: Path | SaveFile,
    names: Names,
    trainer_id: int,
    game: str = VANILLA,
    terrain: int | None = None,
) -> dict:
    """A raw trace plus the player's party, the trainer id and the game (and, when given, the
    battle's ``terrain``: ``battle_terrain``). ``party`` is a
    party TOML or a loaded save, whose party becomes the specs (``SaveMon.to_spec`` with
    ``game``'s tables and the save's trainer, which resolves ``exp_boost``; the save is
    read in place, never copied); an egg there is left out with a note on stderr, as
    ``nuzline fight --save`` does (an egg never enters a battle)."""
    from nuzline.data.party import load_party

    game = get_game(game).id
    if isinstance(party, SaveFile):
        from nuzline.data.games import load_game

        tables, _ = load_game(get_game(game))
        specs = []
        for slot, mon in enumerate(party.party):
            if mon.is_egg:
                print(f"party slot {slot}: egg left out", file=sys.stderr)
            else:
                specs.append(mon.to_spec(tables, party.trainer))
    else:
        specs = load_party(party, names)
    lines = Path(raw).read_text(encoding="utf-8").splitlines()
    bundle = {
        "format": BUNDLE_FORMAT,
        "game": game,
        "trainer_id": trainer_id,
        "party": [dataclasses.asdict(m) for m in specs],
        "records": [json.loads(line) for line in lines if line.strip()],
    }
    if terrain is not None:
        bundle["terrain"] = terrain
    return bundle


def main(argv: list[str] | None = None) -> int:
    from nuzline.cli import resolve_trainer
    from nuzline.data.games import load_game
    from nuzline.data.trainer import load_trainers
    from nuzline.saves.save import load_save

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", required=True, type=Path, help="raw trace (JSON lines)")
    parser.add_argument("--out", required=True, type=Path, help="bundle to write")
    mine = parser.add_mutually_exclusive_group(required=True)
    mine.add_argument("--party", type=Path, help="the player's party TOML")
    mine.add_argument("--save", type=Path, help="a save (.sav/.dsv) whose party is used")
    parser.add_argument("--game", default="vanilla", help="vanilla (default) or kaizo")
    who = parser.add_mutually_exclusive_group()
    who.add_argument("--trainer", default="roark", help="trainer name (a Leader is preferred)")
    who.add_argument("--trainer-id", type=int, help="exact trainer id")
    parser.add_argument(
        "--terrain",
        help="the battle terrain the trace started on (e.g. GRASS), for a trainer whose maps "
        "give several",
    )
    args = parser.parse_args(argv)

    game = get_game(args.game)
    trainer = resolve_trainer(
        load_trainers(game.intermediate_dir), name=args.trainer, trainer_id=args.trainer_id
    )
    tables, names = load_game(game)
    party = args.party if args.save is None else load_save(args.save, game=game.id)
    try:
        terrain = None if args.terrain is None else _terrain_id(args.terrain)
        bundle = bundle_trace(args.trace, party, names, trainer.id, game.id, terrain)
        turns = convert_bundle(bundle, tables, names)
    except TraceError as err:
        print(f"trace_to_state: error: {err}", file=sys.stderr)
        return 1
    for i, t in enumerate(turns[:-1]):
        rolls = {f"{s}.{h}": c for (s, h), c in t.roll_candidates.items()}
        print(f"turn {i}: actions {t.actions} rolls {rolls} mismatched {list(t.mismatched)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(turns)} turn records, trainer {trainer.id}")
    return 0


def _terrain_id(name: str) -> int:
    """``GRASS`` or ``TERRAIN_GRASS`` (any case) -> its ``TERRAIN_NAMES`` id."""
    from nuzline.data.trainer import TERRAIN_NAMES

    key = name.strip().upper()
    key = key if key.startswith("TERRAIN_") else f"TERRAIN_{key}"
    if key not in TERRAIN_NAMES:
        raise TraceError(f"unknown terrain {name!r}")
    return TERRAIN_NAMES.index(key)


if __name__ == "__main__":
    raise SystemExit(main())
