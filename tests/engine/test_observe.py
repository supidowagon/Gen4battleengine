"""Player observations (``engine.observe``): the decomp's HP bar, what an observation
shows and hides, effect-gated chance flags (plan ruling 1) and the information sets
they form, on which the player's legal actions must be constant (notes/stage-b
observe.md, critic.md B.1, C1, C5)."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nuzline.engine.actions import legal_actions, legal_opponent_actions, legal_replacements
from nuzline.engine.damage import STATUS
from nuzline.engine.enumerate import SamplingOnly, enumerate_turn
from nuzline.engine.ids import AbilityId, Status, Weather
from nuzline.engine.observe import (
    OBS_FIELDS,
    OBS_SIZE,
    hp_pixels,
    observation,
    observe,
    replacement_observation,
)
from nuzline.engine.order import (
    MOVE_ACTIONS,
    effective_speed,
    is_switch,
    selected_move,
)
from nuzline.engine.outcome import MAX_HITS, Outcome, empty_outcome
from nuzline.engine.state import BattleState
from nuzline.engine.step import PURSUIT_POWER_MUL, pursuit_fires, step_events
from nuzline.search.playout import replacement_pass, send_replacements
from tests.engine.helpers import BATCH, A, jit_step, roark_real_state, roark_state
from tests.search.fixtures import (
    at_replacement,
    bind_1v1,
    confuse_ray_1v1,
    shinx_vs_cranidos,
    spore_1v1,
    taunt_1v1,
)
from tests.strategies import any_outcome, legal_action_pair, supported_state

CHUNK = BATCH  # the suite's one padded row count: one compiled shape per program
HYPOTHESIS = {
    "deadline": None,
    "suppress_health_check": [
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
}
FIELDS = dict(OBS_FIELDS)


def _turn_row(tables, before, actions, outcome):
    """One expanded turn as Stage B plays it (playout.py:177-194): ``step``, then the
    opponent's forced replacement; its observation, the player's legal actions in the
    child, and the stepped state and observation through ``step_events`` (the path that
    projects ``step``'s own events, ``observation``), to check against ``step`` and
    ``observe``."""
    picks = [selected_move(tables, before, side, actions[side]) for side in (0, 1)]
    choices = (jnp.stack([p[0] for p in picks]), jnp.stack([p[1] for p in picks]))
    pursued = pursuit_fires(tables, before, 0, actions, choices)
    power_mul = jnp.where(pursued, PURSUIT_POWER_MUL, 10)
    stepped = jit_step(tables, before, actions, outcome)
    after, refused = send_replacements(tables, stepped, is_switch(actions[1]), power_mul)
    obs = observe(tables, before, actions, outcome, after)
    with_events, events = step_events(tables, before, actions, outcome)
    via_events = (with_events, observation(events, after))
    return after, refused, obs, legal_actions(tables, after, 0), (stepped, obs), via_events


_turn = jax.jit(jax.vmap(_turn_row, in_axes=(None, 0, 0, 0)))
_observe = jax.jit(jax.vmap(observe, in_axes=(None, 0, 0, 0, 0)))


def _np(tree):
    return jax.tree.map(np.asarray, tree)


def _stack(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)


def _take(tree, idx):
    return jax.tree.map(lambda f: np.asarray(f)[idx], tree)


def _chunked(fn, tables, args, n):
    """``fn`` over ``n`` stacked rows in ``CHUNK`` batches (padding = copies of row 0)."""
    outs = []
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        pad = np.concatenate([idx, np.full(CHUNK - len(idx), idx[0])])
        out = _np(fn(tables, *(_take(a, pad) for a in args)))
        outs.append(_take(out, slice(0, len(idx))))
    return jax.tree.map(lambda *xs: np.concatenate(xs), *outs)


def turns(tables, befores: BattleState, actions, outcomes: Outcome):
    """``_turn_row`` over stacked rows."""
    return _chunked(_turn, tables, (befores, actions, outcomes), len(actions))


def obs_rows(tables, rows):
    """``observe`` of each ``(before, actions, outcome, after)``."""
    stacked = [_stack([r[i] for r in rows]) for i in range(4)]
    return _chunked(_observe, tables, stacked, len(rows))


def one_turn(tables, before, actions, outcome):
    """``(after, obs)`` of a single turn."""
    out = turns(tables, _stack([before]), np.asarray([actions], np.int32), _stack([outcome]))
    return _take(out[0], 0), out[2][0]


def with_outcome(**fields) -> Outcome:
    """``empty_outcome`` with the given ``(side, value)`` / ``(side, hit, value)`` /
    scalar overrides."""
    out = {name: np.asarray(f).copy() for name, f in zip(Outcome._fields, empty_outcome())}
    for name, changes in fields.items():
        if np.ndim(changes) == 0:
            out[name][...] = changes
            continue
        for change in changes:
            *index, value = change
            out[name][tuple(index)] = value
    return Outcome(**out)


def field(obs, name):
    return np.asarray(obs)[..., FIELDS[name]]


def state_key(state) -> bytes:
    return b"".join(np.ascontiguousarray(np.asarray(f)).tobytes() for f in state)


def row_keys(arrays) -> list[bytes]:
    """Per-row bytes of stacked numpy arrays (a state or an observation batch)."""
    leaves = jax.tree.leaves(arrays)
    n = len(leaves[0])
    flat = np.concatenate(
        [np.ascontiguousarray(f).reshape(n, -1).view(np.uint8) for f in leaves], axis=1
    )
    return [flat[i].tobytes() for i in range(n)]


def set_field(state: BattleState, name: str, index, value) -> BattleState:
    arr = np.asarray(getattr(state, name)).copy()
    arr[index] = value
    return state._replace(**{name: jnp.asarray(arr)})


# --- the HP bar -------------------------------------------------------------------


def decomp_pixels(hp: np.ndarray, max_hp: np.ndarray) -> np.ndarray:
    """``FillCells`` (healthbox.c:1559-1590) at rest (``*temp`` = the HP, in 1/256 HP
    when ``max < 48``), with the one-pixel floor for a living Pokemon (:1585-1587)."""
    offset = 48  # 8 * HEALTHBOX_HP_CELL_COUNT (healthbox.c:61, :1571)
    small = max_hp < offset
    pixels = np.where(small, ((hp << 8) * offset // max_hp) >> 8, hp * offset // max_hp)
    return np.where((pixels == 0) & (hp > 0), 1, pixels)


def test_hp_pixels_matches_decomp():
    for hp, max_hp, want in ((0, 10, 0), (1, 999, 1), (999, 999, 48), (24, 48, 24), (47, 48, 47)):
        assert int(hp_pixels(hp, max_hp)) == want
    max_hp = np.repeat(np.arange(1, 1000), np.arange(2, 1001))
    hp = np.concatenate([np.arange(m + 1) for m in range(1, 1000)])
    got = np.asarray(hp_pixels(jnp.asarray(hp, jnp.int32), jnp.asarray(max_hp, jnp.int32)))
    assert (got == decomp_pixels(hp.astype(np.int64), max_hp.astype(np.int64))).all()


# --- what is visible ---------------------------------------------------------------


def big_opponent(tables, names, hp=999):
    """1v1 Roark fixture with the opponent's max HP 999 (pixels ~20.8 HP wide)."""
    s = roark_state(tables, names, player_count=1, opp_count=1)
    s = set_field(s, "max_hp", (1, 0), 999)
    return set_field(s, "hp", (1, 0), hp)


@pytest.mark.slow
def test_opponent_hp_within_pixel_equal(tables, names):
    s = big_opponent(tables, names)
    acts = np.asarray(A("Scratch", "Tackle", s, names))
    out = empty_outcome()

    def after(hp):
        return set_field(s, "hp", (1, 0), hp)

    obs = obs_rows(tables, [(s, acts, out, after(h)) for h in (500, 510, 530)])
    assert [int(hp_pixels(h, 999)) for h in (500, 510, 530)] == [24, 24, 25]
    assert (obs[0] == obs[1]).all()
    assert (obs[0] != obs[2]).any()
    assert len(obs[0]) == OBS_SIZE


def test_player_exact_hp_splits(tables, names):
    s = roark_state(tables, names)
    acts = np.asarray(A("Scratch", "Tackle", s, names))
    hp = int(s.hp[0, 0])
    rows = [(s, acts, empty_outcome(), set_field(s, "hp", (0, 0), h)) for h in (hp, hp - 1)]
    obs = obs_rows(tables, rows)
    assert (field(obs[0], "player_hp") != field(obs[1], "player_hp")).any()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("build", "move", "counter"),
    [
        (spore_1v1, "Spore", "status_counter"),
        (confuse_ray_1v1, "Confuse Ray", "confusion"),
        (taunt_1v1, "Taunt", "taunt"),
        (bind_1v1, "Bind", "trapped"),
    ],
)
def test_hidden_counters_grouped(tables, names, build, move, counter):
    s = build(tables, names)
    acts = np.asarray(A(move, "Tackle", s, names))
    afters, obs = [], []
    for duration in (2, 3, 4, 5) if move != "Taunt" else (3, 4, 5):
        if move == "Bind":
            duration += 1  # Random 3, 3: 3-6
        after, o = one_turn(tables, s, acts, with_outcome(duration=[(0, duration)]))
        afters.append(after)
        obs.append(o)
    index = (1, 0) if counter == "status_counter" else 1  # Geodude's
    values = {int(np.asarray(getattr(a, counter))[index]) for a in afters}
    assert len(values) == len(afters) and 0 not in values  # the counters really differ
    assert all((o == obs[0]).all() for o in obs)
    # Counter 0 versus non-zero in ``after`` is visible.
    a = afters[0]
    if counter == "status_counter":
        cleared = a._replace(
            status=jnp.asarray(np.asarray(a.status) * 0),
            status_counter=jnp.asarray(np.asarray(a.status_counter) * 0),
        )
    else:
        cleared = set_field(a, counter, 1, 0)
    o = obs_rows(tables, [(s, acts, with_outcome(duration=[(0, 2)]), x) for x in (a, cleared)])
    assert (o[0] != o[1]).any()


VISIBLE_CLASSES = (
    "disabled move",
    "locked move",
    "foresight",
    "curled",
    "bide",
    "mud sport",
    "aqua ring",
    "stat stage",
    "weather",
    "stealth rock",
    "screen",
    "item consumed",
    "ability reveal",
    "thaw",
    "full paralysis",
    "flinch",
    "confusion self-hit",
    "switch target",
    "replacement",
    "turn order",
    # E2 batch 7 (B7-n): their cases were in visible_cases, not in this list.
    "ingrain",
    "mean look",
    "imprison",
    "roosting",
)


def visible_cases(tables, names):
    """Message class -> a pair from one parent that differs only in that class: either
    ``(field, state, actions, outcomes)`` (the two turns are played; ``actions`` one
    pair or a list of two) or ``(field, rows)`` (two ``(before, actions, outcome,
    after)`` rows whose ``after`` differs in one visible field: no M1 move sets weather
    or screens). ``field`` is the ``OBS_FIELDS`` entry that must tell them apart."""
    base = roark_state(tables, names)
    sc_tk = np.asarray(A("Scratch", "Tackle", base, names))
    plain = with_outcome()
    played, _ = one_turn(tables, base, sc_tk, plain)

    def projected(fld, change):
        return fld, [(base, sc_tk, plain, played), (base, sc_tk, plain, change(played))]

    leer = base.with_stages(0, eva=1)  # Leer can miss
    static = set_field(base, "ability", (0, 0), AbilityId.STATIC)  # Tackle makes contact
    headbutt = roark_state(tables, names, player_moves=["Headbutt"])
    tie = roark_state(tables, names, equal_speed=True)
    berry = int(played.item[0, 2])  # Grotle's Sitrus Berry
    switches = [np.asarray([4 + 1, 0], np.int32), np.asarray([4 + 2, 0], np.int32)]
    return {
        "disabled move": projected("disabled_move", lambda a: set_field(a, "disabled_move", 1, 33)),
        "locked move": projected("locked_move", lambda a: set_field(a, "locked_move", 0, 33)),
        "foresight": projected("foresight", lambda a: set_field(a, "foresight", 1, 1)),
        "curled": projected("curled", lambda a: set_field(a, "curled", 0, 1)),
        "bide": projected("bide_turns", lambda a: set_field(a, "bide_turns", 1, 2)),
        # Mud Sport is per battler: one side's sport differs from the other's.
        "mud sport": projected("mud_sport", lambda a: set_field(a, "mud_sport", 1, 1)),
        "aqua ring": projected("aqua_ring", lambda a: set_field(a, "aqua_ring", 1, 1)),
        # E2 batch 7: "planted its roots!", "can no longer escape!", "sealed the opponent's
        # move(s)!", and a Roost that healed (kept at a pause or a pending replacement).
        "ingrain": projected("ingrain", lambda a: set_field(a, "ingrain", 1, 1)),
        "mean look": projected("mean_look", lambda a: set_field(a, "mean_look", 0, 1)),
        "imprison": projected("imprison", lambda a: set_field(a, "imprison", 1, 1)),
        "roosting": projected("roosting", lambda a: set_field(a, "roosting", 0, 1)),
        "stat stage": (
            "stages",
            leer,
            np.asarray(A("Scratch", "Leer", leer, names)),
            [plain, with_outcome(hit=[(1, 0, 0)])],
        ),
        "weather": projected("weather", lambda a: a._replace(weather=jnp.int8(Weather.RAIN))),
        "stealth rock": projected("stealth_rock", lambda a: set_field(a, "stealth_rock", 0, 1)),
        "screen": projected("reflect", lambda a: set_field(a, "reflect", 1, 5)),
        "item consumed": projected(
            "item",
            lambda a: set_field(set_field(a, "item", (0, 2), 0), "item_consumed", (0, 2), berry),
        ),
        "ability reveal": (
            "contact_ability",
            static,
            sc_tk,
            [plain, with_outcome(contact_proc=[(0, 0, 1)])],
        ),
        "thaw": (
            "thawed",
            base.with_status(1, 0, Status.FREEZE),
            sc_tk,
            [plain, with_outcome(thaw=[(1, 1)])],
        ),
        "full paralysis": (
            "full_para",
            base.with_status(1, 0, Status.PARALYSIS),
            sc_tk,
            [plain, with_outcome(full_para=[(1, 1)])],
        ),
        "flinch": (
            "flinched",
            headbutt,
            np.asarray(A("Headbutt", "Tackle", headbutt, names)),
            [plain, with_outcome(secondary=[(0, 0, 1)])],
        ),
        "confusion self-hit": (
            "self_hit",
            set_field(base, "confusion", 1, 3),
            sc_tk,
            [plain, with_outcome(confusion_hit=[(1, 1)])],
        ),
        "switch target": ("act_id", base, switches, [plain, plain]),
        "replacement": (
            "active",
            set_field(base, "hp", (1, 0), 1),
            sc_tk,
            [plain, with_outcome(hit=[(0, 0, 0)])],
        ),
        "turn order": (
            "order",
            tie,
            np.asarray(A("Scratch", "Tackle", tie, names)),
            [plain, with_outcome(speed_tie=1)],
        ),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", VISIBLE_CLASSES)
def test_visible_message_classes(tables, names, name):
    cases = visible_cases(tables, names)
    assert set(cases) == set(VISIBLE_CLASSES)
    fld, *rest = cases[name]
    if len(rest) == 1:
        obs = obs_rows(tables, rest[0])
    else:
        s, acts, outcomes = rest
        acts = acts if isinstance(acts, list) else [acts, acts]
        obs = turns(tables, _stack([s, s]), np.stack(acts), _stack(outcomes))[2]
    assert (field(obs[0], fld) != field(obs[1], fld)).any(), name


@pytest.mark.slow
def test_hidden_new_counters_grouped(tables, names):
    # Disable's turn count and Bide's stored damage are not on screen (spec §4.2 "hidden
    # counters"): the observation shows only the Disabled move and that the foe is Biding.
    s = roark_state(tables, names)
    acts = np.asarray(A("Scratch", "Tackle", s, names))
    played, _ = one_turn(tables, s, acts, with_outcome())
    played = set_field(set_field(played, "disabled_move", 1, 33), "bide_turns", 1, 2)
    rows = [
        (
            s,
            acts,
            with_outcome(),
            set_field(set_field(played, "disable_turns", 1, n), "bide_damage", 1, d),
        )
        for n, d in ((1, 10), (4, 250))
    ]
    obs = obs_rows(tables, rows)
    assert (obs[0] == obs[1]).all()
    cleared = set_field(set_field(played, "disable_turns", 1, 0), "disabled_move", 1, 0)
    obs = obs_rows(tables, [rows[0], (s, acts, with_outcome(), cleared)])
    assert (obs[0] != obs[1]).any()


def _equal_obs(tables, s, acts, a: Outcome, b: Outcome):
    out = turns(tables, _stack([s, s]), np.stack([acts, acts]), _stack([a, b]))
    return (out[2][0] == out[2][1]).all(), out


def test_ineffective_draws_invisible(tables, names):
    base = roark_state(tables, names)
    sc_tk = np.asarray(A("Scratch", "Tackle", base, names))
    cases = [
        ("full_para, not paralyzed", base, sc_tk, with_outcome(full_para=[(1, 1)])),
        ("thaw, not frozen", base, sc_tk, with_outcome(thaw=[(0, 1), (1, 1)])),
        ("confusion_hit, not confused", base, sc_tk, with_outcome(confusion_hit=[(0, 1), (1, 1)])),
        ("crit on a missed hit", base, sc_tk, None),
        (
            "crit on a status move",
            base,
            np.asarray(A("Leer", "Leer", base, names)),
            with_outcome(crit=[(0, 0, 1), (1, 0, 1)]),
        ),
        ("speed_tie, speeds differ", base, sc_tk, with_outcome(speed_tie=1)),
    ]
    for name, s, acts, changed in cases:
        if changed is None:  # both moves miss; the crit flags vary
            plain = with_outcome(hit=[(0, 0, 0), (1, 0, 0)])
            changed = with_outcome(hit=[(0, 0, 0), (1, 0, 0)], crit=[(0, 0, 1), (1, 0, 1)])
        else:
            plain = empty_outcome()
        equal, out = _equal_obs(tables, s, acts, plain, changed)
        assert equal, name

    # Spark paralyzes Cranidos (faster) after it has acted: its full_para has no effect.
    s = shinx_vs_cranidos(tables, names)
    assert int(s.stats[1, 0, 4]) > int(s.stats[0, 0, 4])
    acts = np.asarray(A("Spark", "Tackle", s, names))
    para = with_outcome(secondary=[(0, 0, 1)])
    equal, out = _equal_obs(
        tables, s, acts, para, with_outcome(secondary=[(0, 0, 1)], full_para=[(1, 1)])
    )
    assert int(out[0].status[0, 1, 0]) == Status.PARALYSIS
    assert equal, "full_para after the paralyzed side acted"

    # Crit on hit indices after the KO (multi-hit) and beyond the hit count.
    s = set_field(roark_state(tables, names, player_moves=["Spike Cannon"]), "hp", (1, 0), 1)
    acts = np.asarray(A("Spike Cannon", "Tackle", s, names))
    plain = with_outcome(n_hits=[(0, 5)])
    late = with_outcome(n_hits=[(0, 5)], crit=[(0, i, 1) for i in range(1, MAX_HITS)])
    equal, _ = _equal_obs(tables, s, acts, plain, late)
    assert equal, "crit after the KO"
    s = roark_state(tables, names, player_moves=["Spike Cannon"])
    plain = with_outcome(n_hits=[(0, 2)])
    late = with_outcome(n_hits=[(0, 2)], crit=[(0, i, 1) for i in range(2, MAX_HITS)])
    equal, _ = _equal_obs(tables, s, np.asarray(A("Spike Cannon", "Tackle", s, names)), plain, late)
    assert equal, "crit beyond the hit count"

    # speed_tie with equal speeds when only one side runs a move (a switch goes first).
    s = roark_state(tables, names, equal_speed=True)
    acts = np.asarray([4 + 1, 0], np.int32)
    equal, _ = _equal_obs(tables, s, acts, empty_outcome(), with_outcome(speed_tie=1))
    assert equal, "speed_tie with a switch"


def _gate_free(tables, s, acts, outcome: Outcome, name, index) -> bool:
    """Element ``name[index]`` of ``outcome`` has no possible effect, judged from the
    rules alone (not from ``observe``): the fields whose every visible effect changes
    the state, and ``crit``/``hit``/``speed_tie`` where the rules make them inert."""
    if name in (
        "roll",
        "secondary",
        "contact_proc",
        "n_hits",
        "full_para",
        "confusion_hit",
        "duration",
        "thaw",
        "end_speed_tie",
        "flinch_item",
        "quick_claw",
    ):
        return True
    if name == "speed_tie":
        return int(effective_speed(tables, s, 0)) != int(effective_speed(tables, s, 1))
    side = index[0]
    if acts[side] >= MOVE_ACTIONS:
        return name in ("hit", "crit")
    move = int(selected_move(tables, s, side, acts[side])[0])
    if name == "hit":
        return int(tables.move_accuracy[move]) == 0
    status_move = int(tables.move_category[move]) == STATUS
    missed = int(outcome.hit[side, 0]) == 0 and int(tables.move_accuracy[move]) != 0
    return status_move or missed


_ranges = {
    "roll": 16,
    "n_hits": 6,
    "duration": 7,
}


@pytest.mark.slow
@settings(max_examples=25, **HYPOTHESIS)
@given(data=st.data())
def test_ineffective_draws_invisible_enumerated(tables, names, data):
    which = data.draw(st.sampled_from(["roark", "spore", "shinx"]))
    if which == "roark":
        s = roark_real_state(tables, names)
    elif which == "spore":
        s = spore_1v1(tables, names)
    else:
        s = shinx_vs_cranidos(tables, names, slower_opp=data.draw(st.booleans()))
    acts = np.asarray(data.draw(legal_action_pair(tables, s)))
    try:
        outcomes, _ = enumerate_turn(tables, s, acts)  # every invoker call variant
    except SamplingOnly:
        return
    k = len(outcomes.hit)
    rng = np.random.default_rng(data.draw(st.integers(0, 2**31 - 1)))
    picked = rng.choice(k, min(k, CHUNK // 2), replace=False)
    base = _take(outcomes, picked)
    changed = {name: np.asarray(f).copy() for name, f in zip(Outcome._fields, base)}
    gated = np.zeros(len(picked), bool)
    for row in range(len(picked)):
        name = Outcome._fields[rng.integers(len(Outcome._fields))]
        arr = changed[name]
        index = tuple(int(rng.integers(n)) for n in arr.shape[1:])
        old = _take(base, row)
        gated[row] = _gate_free(tables, s, acts, old, name, index)
        lo = 2 if name in ("n_hits", "duration") else 0
        arr[(row, *index)] = rng.integers(lo, _ranges.get(name, 2))
    both = jax.tree.map(lambda a, b: np.concatenate([np.asarray(a), b]), base, Outcome(**changed))
    n = len(picked)
    out = turns(tables, _stack([s] * (2 * n)), np.tile(acts, (2 * n, 1)), both)
    keys = row_keys(_take(out[0], slice(None)))
    for row in range(n):
        if gated[row] and keys[row] == keys[n + row]:
            assert (out[2][row] == out[2][n + row]).all()


def test_crit_and_miss_visible(tables, names):
    s = set_field(roark_state(tables, names), "hp", (1, 0), 1)  # any hit KOs
    acts = np.asarray(A("Scratch", "Tackle", s, names))
    equal, out = _equal_obs(tables, s, acts, empty_outcome(), with_outcome(crit=[(0, 0, 1)]))
    assert state_key(_take(out[0], 0)) == state_key(_take(out[0], 1))  # the same after
    assert not equal and (field(out[2][0], "crit") != field(out[2][1], "crit")).any()

    s = roark_state(tables, names)
    acts = np.asarray(A("Scratch", "Rock Throw", s, names))
    equal, out = _equal_obs(tables, s, acts, empty_outcome(), with_outcome(hit=[(1, 0, 0)]))
    assert not equal and (field(out[2][0], "missed") != field(out[2][1], "missed")).any()


def test_unrun_choice_invisible(tables, names):
    s = set_field(roark_state(tables, names), "hp", (1, 0), 1)  # Monferno is faster
    rows = [np.asarray(A("Scratch", m, s, names)) for m in ("Tackle", "Rock Throw", "Leer")]
    out = turns(tables, _stack([s] * 3), np.stack(rows), _stack([empty_outcome()] * 3))
    keys = row_keys(out[0])
    assert int(out[0].active[0, 1]) != 0  # Geodude fainted before acting, replaced
    assert keys[0] == keys[1] == keys[2]
    assert (out[2][0] == out[2][1]).all() and (out[2][0] == out[2][2]).all()


# --- information sets --------------------------------------------------------------


def legal_pairs(tables, state):
    player = np.flatnonzero(np.asarray(legal_actions(tables, state, 0)))
    opp = np.flatnonzero(np.asarray(legal_opponent_actions(tables, state)))
    return [(int(a), int(b)) for a in player for b in opp]


def grouped_children(tables, parents, pairs, max_k=None):
    """``obs bytes -> {child state bytes: legal-action bytes}`` over every enumerated
    outcome of each parent's pairs (refused replacements left out). ``max_k`` skips
    pairs with more outcome rows."""
    groups: dict[bytes, dict[bytes, bytes]] = {}
    for s in parents:
        for pair in pairs(s):
            acts = np.asarray(pair, np.int32)
            try:
                outcomes, _ = enumerate_turn(tables, s, acts)
            except SamplingOnly:
                continue
            k = len(outcomes.hit)
            if max_k is not None and k > max_k:
                continue
            befores = jax.tree.map(lambda f, k=k: np.broadcast_to(f, (k, *np.shape(f))), _np(s))
            after, refused, obs, legal, direct, via_events = turns(
                tables, befores, np.tile(acts, (k, 1)), outcomes
            )
            check_events_path(direct, via_events)
            for o, child, lg, bad in zip(row_keys(obs), row_keys(after), row_keys(legal), refused):
                if not bad:
                    groups.setdefault(o, {})[child] = lg
    return groups


def check_events_path(direct, via_events):
    """``step_events`` gives ``step``'s state, and ``observation`` of its events gives
    ``observe``'s vector."""
    for name, a, b in zip(BattleState._fields, direct[0], via_events[0]):
        assert (np.asarray(a) == np.asarray(b)).all(), name
    assert (np.asarray(direct[1]) == np.asarray(via_events[1])).all()


def assert_constant(groups):
    for children in groups.values():
        assert len(set(children.values())) == 1


def locked_bide(tables, names, turns=2):
    """Kricketot (``turns`` Bide turns left, 30 damage stored) against Geodude and Onix."""
    s = roark_state(
        tables,
        names,
        player_species="Kricketot",
        player_moves=("Bide", "Growl"),
        opp_species="Geodude",
        opp_moves=("Tackle", "Leer"),
        player_count=2,
        opp_count=2,
    )
    return s._replace(
        locked_move=s.locked_move.at[0].set(names.move("Bide")),
        bide_turns=s.bide_turns.at[0].set(turns),
        bide_damage=s.bide_damage.at[0].set(30),
    )


def test_legality_constant_on_equal_obs(tables, names):
    s = roark_real_state(tables, names)
    groups = grouped_children(tables, [s], lambda p: legal_pairs(tables, p))
    assert groups
    assert_constant(groups)

    # Spore across durations: one information set, several states.
    s = spore_1v1(tables, names)
    pair = [tuple(int(x) for x in A("Spore", "Tackle", s, names))]
    groups = grouped_children(tables, [s], lambda p: pair)
    assert_constant(groups)
    assert max(len(c) for c in groups.values()) >= 2

    # A rolled hit on a 999-max-HP opponent: HP values inside one pixel.
    s = big_opponent(tables, names)
    pair = [tuple(int(x) for x in A("Scratch", "Tackle", s, names))]
    groups = grouped_children(tables, [s], lambda p: pair)
    assert_constant(groups)
    assert max(len(c) for c in groups.values()) >= 2

    # A player locked into Bide (Review Focus 2): the lock is on screen, so the legal
    # actions (the locked slot alone) are constant over every enumerated outcome.
    parents = [locked_bide(tables, names, turns) for turns in (2, 1)]
    for s in parents:
        legal = np.asarray(legal_actions(tables, s, 0))
        assert np.flatnonzero(legal).tolist() == [names.move_slot(s, 0, "Bide")]
    groups = grouped_children(tables, parents, lambda p: legal_pairs(tables, p))
    assert groups
    assert_constant(groups)


def test_legality_disable_splits_on_move_groups_on_turns(tables, names):
    """A Disabled move splits information sets (``disabled_move`` is on screen); its
    turn count does not, and the legal actions after the tick agree within a set."""
    base = roark_state(
        tables, names, player_count=1, opp_count=1, player_moves=("Tackle", "Scratch", "Leer")
    )
    scratch, other = names.move("Scratch"), names.move("Leer")
    parents = [
        set_field(set_field(base, "disabled_move", 0, move), "disable_turns", 0, turns)
        for move, turns in ((scratch, 2), (scratch, 3), (other, 2))
    ]
    acts = np.asarray(A("Tackle", "Tackle", base, names), np.int32)
    groups = grouped_children(tables, parents, lambda p: [tuple(int(x) for x in acts)])
    assert_constant(groups)
    # The two Scratch parents differ only in the hidden counter: one observation, two states.
    assert max(len(c) for c in groups.values()) >= 2
    obs = {k for k in groups}
    assert len(obs) >= 2  # the other disabled move is a different observation
    # A counter at 0 clears the move at the end of the turn: visible.
    ends = set_field(set_field(base, "disabled_move", 0, scratch), "disable_turns", 0, 0)
    still = set_field(set_field(base, "disabled_move", 0, scratch), "disable_turns", 0, 1)
    out = turns(
        tables, _stack([ends, still]), np.stack([acts, acts]), _stack([empty_outcome()] * 2)
    )
    assert field(out[2][0], "disabled_move").tolist() == [0, 0]
    assert field(out[2][1], "disabled_move").tolist() == [scratch, 0]


def test_legality_splits_on_taunt(tables, names):
    """Two states of one information set differing only in the player's hidden Taunt
    counter (1 or 2): after the end-of-turn tick, Leer is legal in one child only, and
    only ``taunt`` in the observation tells them apart."""
    base = roark_state(tables, names, player_count=1, opp_count=1)
    parents = [set_field(base, "taunt", 0, t) for t in (1, 2)]
    acts = np.asarray(A("Scratch", "Tackle", base, names), np.int32)
    out = turns(tables, _stack(parents), np.stack([acts, acts]), _stack([empty_outcome()] * 2))
    _, _, obs, legal = out[:4]
    assert (legal[0] != legal[1]).any()
    assert (obs[0] != obs[1]).any()
    diff = np.flatnonzero(obs[0] != obs[1])
    assert set(diff) <= set(range(OBS_SIZE)[FIELDS["taunt"]])
    # The same for Bind: freed at 0 (switches legal) or still trapped. Bind's damage
    # makes the player's HP differ as well, so ``trapped`` is not the only witness.
    base = roark_state(tables, names, opp_count=1)
    bind = names.move("Bind")
    parents = [set_field(set_field(base, "trapped", 0, t), "trap_move", 0, bind) for t in (1, 2)]
    out = turns(tables, _stack(parents), np.stack([acts, acts]), _stack([empty_outcome()] * 2))
    _, _, obs, legal = out[:4]
    assert (legal[0] != legal[1]).any()
    assert (field(obs[0], "trapped") != field(obs[1], "trapped")).any()


@settings(max_examples=5, **HYPOTHESIS)
@given(data=st.data())
def test_legality_constant_on_equal_obs_random(tables, names, data):
    # Pairs above 4 chunks of outcome rows are left out for runtime.
    s = data.draw(supported_state(tables, names))
    groups = grouped_children(tables, [s], lambda p: legal_pairs(tables, p), max_k=4 * CHUNK)
    assert_constant(groups)


@pytest.mark.slow
@settings(max_examples=40, **HYPOTHESIS)
@given(data=st.data())
def test_step_events_match_step_and_observe(tables, names, data):
    """``step_events`` is ``step`` plus its events, and ``observation`` of those events
    is ``observe`` (the path Stage B's expansion uses to step once)."""
    s = data.draw(supported_state(tables, names))
    acts = np.asarray(data.draw(legal_action_pair(tables, s)))
    outcome = _np(data.draw(any_outcome()))
    out = turns(tables, _stack([s]), acts[None], _stack([outcome]))
    check_events_path(out[4], out[5])


@pytest.mark.slow
def test_magnitude_level_is_observed_only_when_shown(tables, names):
    """ "Magnitude N!" (subscript_magnitude.s) tells the player the level: two draws that
    differ only in it are different observations when the hit lands, and the same one when
    the target is immune (no message, no damage)."""

    def observe_levels(opp):
        s = roark_state(
            tables,
            names,
            player_species="Geodude",
            player_moves=("Magnitude",),
            opp_species=opp,
            opp_moves=("Leer",),
        )
        acts = A("Magnitude", "Leer", s, names)
        out = []
        for level in (4, 10):
            o = empty_outcome()._replace(magnitude=jnp.full((2,), level, jnp.int8))
            after = jit_step(tables, s, acts, o)
            out.append(np.asarray(observe(tables, s, acts, o, after)))
        return out

    low, high = observe_levels("Bidoof")
    assert field(low, "magnitude").tolist() == [4, 0] and field(high, "magnitude").tolist() == [
        10,
        0,
    ]
    low, high = observe_levels("Staravia")
    assert (low == high).all()


def test_replacement_observation_shows_both_newcomers(tables, names):
    s = at_replacement(roark_state(tables, names), opp_down=True)
    slot = int(np.flatnonzero(np.asarray(legal_replacements(s, 0)))[0])
    after, refused = replacement_pass(tables, s, slot)
    assert not bool(refused)
    assert int(after.active[1]) != int(s.active[1])
    obs = np.asarray(replacement_observation(after))
    assert obs.shape == (OBS_SIZE,)
    assert field(obs, "active").tolist() == np.asarray(after.active).tolist()
    assert field(obs, "active").tolist() == [slot, int(after.active[1])]
    assert field(obs, "act_kind").tolist() == [0, 0]
    assert field(obs, "order").tolist() == [-1]
    assert field(obs, "pursuit").tolist() == [0]
    assert not field(obs, "crit").any()
    for name in ("act_id", "magnitude"):
        assert not field(obs, name).any(), name


# --- the player's party after a payout (K3d-exp T7, design §3.5) -----------------------


def _player_obs_diff(a: BattleState, b: BattleState) -> set[str]:
    """The observation fields in which ``a`` and ``b`` (seen as after a replacement) differ."""
    oa = np.asarray(replacement_observation(a))
    ob = np.asarray(replacement_observation(b))
    return {name for name, sl in OBS_FIELDS if not np.array_equal(oa[sl], ob[sl])}


def test_a_level_up_is_observed(tables, names):
    """Two histories in which the same KO pays the same exp: one levels Prinplup, one does
    not. The level, stats, max HP, HP and learn prompt split them, and the exp shows the
    amount ("gained N Exp. Points!", battle_script.c:9738)."""
    from nuzline.engine.levelup import pay_exp
    from nuzline.engine.state import make_state
    from tests.engine.helpers import roark_party, sample_party

    prinplup = sample_party(names)[1]
    rate = int(np.asarray(tables.species_exp_rate)[prinplup.species])
    table = np.asarray(tables.exp_table)[rate]

    def paid(exp):
        before = make_state(tables, [replace(prinplup, exp=int(exp))], roark_party(names))
        return pay_exp(tables, before, before._replace(hp=before.hp.at[1, 0].set(0)))

    short, base = table[prinplup.level + 1] - 1, table[prinplup.level]
    leveled, flat = paid(short), paid(base)
    assert (
        int(leveled.level[0, 0]) == prinplup.level + 1 and int(flat.level[0, 0]) == prinplup.level
    )
    diff = _player_obs_diff(leveled, flat)
    assert {"player_level", "player_stats", "player_max_hp", "player_exp"} <= diff
    assert diff <= {
        "player_level",
        "player_stats",
        "player_max_hp",
        "player_hp",
        "player_exp",
        "player_moves",
        "player_pp",
        "learn_level",
        "learn_cursor",
    }


def test_max_hp_through_evs_alone_is_observed(tables, names):
    """Two states that differ only in max HP, through the HP EVs a payout adds (Finding 21):
    same HP, same other stats. Only ``player_max_hp`` tells them apart."""
    from nuzline.engine.state import make_state
    from tests.engine.helpers import roark_party, sample_party

    mon = replace(sample_party(names)[0], stats=None, current_hp=20)
    evs = list(mon.evs)
    states = []
    for hp_ev in range(0, 256, 4):
        evs[0] = hp_ev
        states.append(make_state(tables, [replace(mon, evs=tuple(evs))], roark_party(names)))
        if int(states[-1].max_hp[0, 0]) != int(states[0].max_hp[0, 0]):
            break
    a, b = states[0], states[-1]
    assert int(a.max_hp[0, 0]) != int(b.max_hp[0, 0])
    assert np.array_equal(np.asarray(a.stats), np.asarray(b.stats))
    assert int(a.hp[0, 0]) == int(b.hp[0, 0]) == 20
    assert _player_obs_diff(a, b) == {"player_max_hp"}


def test_unscanned_learn_work_at_the_battle_end_is_no_prompt(tables, names):
    """Ruling D8 keeps the learn work of a battle-ending KO, which no scan reaches: it may be
    unscanned work, not a prompt (T6 review hand-off). Once ``over`` is set the learn fields
    read 0; while the battle goes on, a (resolved) prompt is shown."""
    state = roark_state(tables, names)
    marked = state._replace(
        learn_level=state.learn_level.at[0].set(18), learn_cursor=state.learn_cursor.at[0].set(2)
    )
    assert _player_obs_diff(marked, state) == {"learn_level", "learn_cursor"}
    over = state._replace(over=jnp.int8(1), winner=jnp.int8(0))
    assert (
        _player_obs_diff(
            over._replace(learn_level=marked.learn_level, learn_cursor=marked.learn_cursor), over
        )
        == set()
    )
