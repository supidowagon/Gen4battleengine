"""Stage A playouts (docs/design.md §4.3): play a battle on to its end from a state.

Each turn the player picks ``playout_policy``'s action, the opponent ``ai_policy``'s
under a sampled ``AiOutcome``, the turn's chance events come from ``sample_outcome``
and ``step`` plays it; a fainted opponent is then replaced (``send_replacements``) and
a fainted player Pokemon (Set mode: no switch is offered before the faint, so the player
chooses after it) by ``replacement_policy`` (``finish_turn``, ``replacement_pass``). A
playout ends at the end of the battle (``BattleState.over``), after ``max_turns`` turns,
or when the AI meets a refused mechanic.

``playout_policy`` is the player's default policy, a search heuristic (it is not the
trainer AI, which follows the decomp): a one-ply expectimax over the player's legal
actions against the opponent's exact action distribution (``ai_action_distribution``).
Each pair of actions is valued ``(1 - death_cost * risk) * (1 + fraction of the
target's current HP removed)``, ``risk`` the chance the player's Pokemon faints this turn
and ``death_cost`` what that faint costs against the faint budget (plan ruling 6: 1 with no
budget left, for a protected Pokemon or the last one, 1/2 in a failed state (over the budget
already, ``deaths.over_budget``) unless it is the last one, else ``1 / (deaths left + 1)``;
``death_cost``), from the exact non-critical damage tables (16 rolls) and
the accuracy of both moves: the player is at risk only if the opponent's move KOs, and the
opponent's move does not run when the player acts first (``action_order``, a speed tie
counted as 1/2) and KOs. A switch is valued against the incoming Pokemon (stages reset);
crits, multi-hit counts, secondary effects, switch-in effects and end-of-turn damage are
left out. ``key`` breaks exact ties only.

Set-mode replacements (ruling 2, ``BattleControllerPlayer_ReplaceFainted``,
battle_controller_player.c:4070-4155): when both actives are down the AI picks against the
player's *fainted* battler (``BtlCmd_ShowParty`` opens both menus first,
``Task_TrainerShowPartyMenu``, battle_display.c:4543-4556), the player enters first
(``GoToIfAnySwitches``, battle_script.c:2537) and then the opponent; the turn's ``powerMul``
waits in ``BattleState.replace_mul``. ``replacement_policy`` is a search heuristic like
``playout_policy`` (ruling 7). A playout started at a pending player replacement resolves
it first, drawing no key and counting no turn.

The opponent's replacement follows ``BattleAI_PostKOSwitchIn`` (``nuzline.ai.replacement``,
docs/ai/roark.md 11.2), with ``powerMul`` 20 on a turn where its Pursuit hit the
player's switch. A replacement in a turn the AI switched is refused (roark.md 14, item
(3): ``Task_TrainerShowPartyMenu`` reads the stale ``aiSwitchedPartySlot``).

A root's hidden draws (``root_state``): the battle-start ``speedRand``, which decides a
lead's Quick Claw on turn 1 (``BattleState.qc_roll``, ``step.sample_root``), is a chance
node. Stage A draws it per playout at its root (``stage_a.rank_actions``, and
``leaves.evaluate_leaves(root=True)`` for the prep screen) from ``fold_in(key, ROOT_FOLD)``,
so the playout's own key stream is unchanged; a state below a root carries the value
``step`` stored.

The trainer AI's policy (``nuzline.ai.layout``): ``tables`` are traced here, so the layout
cannot be read from them. Every function that runs the AI takes it as the static
``layout`` (default ``VANILLA``, under which another policy's tables are refused, never
misread); the program builders (``stage_a.rank_actions``, ``leaves``) resolve it with
``layout_of`` on the concrete tables.

Refusals cannot raise under ``jit``: a playout that meets one stops with
``PlayoutResult.refused = 1`` and ``won = REFUSED_WON`` (-1; the turn is not played), so
summing ``won`` over refused playouts cannot pass for losses. Callers must exclude
refused playouts or refuse loudly; ``refusal_reason`` replays one playout outside
``jit`` and names the refused mechanic for the message.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from nuzline.ai.layout import VANILLA, AiLayout, layout_of
from nuzline.ai.outcome import sample_ai_outcome
from nuzline.ai.policy import (
    UnsupportedAi,
    ai_action_distribution,
    ai_path_refusals,
    ai_policy,
    ai_refusals,
)
from nuzline.ai.replacement import post_ko_switch_in
from nuzline.data.tables import Tables
from nuzline.engine.actions import (
    MOVE_ACTIONS,
    NUM_ACTIONS,
    NUM_OPP_ACTIONS,
    legal_actions,
    legal_replacements,
)
from nuzline.engine.damage import damage_table
from nuzline.engine.hooks import download, intimidate
from nuzline.engine.ids import AbilityId, Stat
from nuzline.engine.order import action_order, is_switch, used_move
from nuzline.engine.outcome import empty_outcome, hit_threshold, sample_outcome
from nuzline.engine.state import SLOTS, BattleState
from nuzline.engine.step import (
    NORMAL_POWER_MUL,
    WINNER_PLAYER,
    needs_replacement,
    replace,
    replacement_power_mul,
    sample_root,
    step,
)
from nuzline.search.deaths import over_budget
from nuzline.search.learn import learn_reason, resolve_learns, settle_learns, settle_reason

__all__ = [
    "REFUSED_WON",
    "PlayoutResult",
    "batched_playout",
    "death_cost",
    "finish_turn",
    "playout",
    "playout_from_action",
    "playout_policy",
    "refusal_reason",
    "replacement_pass",
    "replacement_policy",
    "replacement_refusal",
    "root_state",
    "send_replacements",
]

_I32, _F32 = jnp.int32, jnp.float32
REFUSED_WON = -1  # PlayoutResult.won of a refused playout
ROOT_FOLD = 0x0C1A  # fold_in data of a playout key's root draws (root_state)


def root_state(tables: Tables, state: BattleState, key) -> BattleState:
    """``state`` with its hidden root draws made (module docstring): ``step.sample_root``
    on ``fold_in(key, ROOT_FOLD)``, ``key`` being the playout's own key."""
    return sample_root(jax.random.fold_in(key, ROOT_FOLD), tables, state)


class PlayoutResult(NamedTuple):
    won: jnp.ndarray  # int8[], 1 = the player won the battle, REFUSED_WON = refused
    player_faints: jnp.ndarray  # int8[]
    turns: jnp.ndarray  # int16[], turns played
    refused: jnp.ndarray  # int8[], 1 = stopped at a refused mechanic (no result)
    deaths: jnp.ndarray  # int8[6], the final state's per-slot death record
    ended: jnp.ndarray  # int8[], 1 = the battle is over (not cut off by max_turns)
    hp_left: jnp.ndarray  # float32[], player party HP / max HP at the end (refused: 0)
    pp_left: jnp.ndarray  # float32[], player party PP / full PP of its moves (refused: 0)


def _resources_left(tables: Tables, state: BattleState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(hp_left, pp_left)`` (``float32``): the player party's HP over its max HP, and its
    PP over the full PP of the moves it knows (an empty slot counts 0). hp_left/pp_left are
    reported by the prep search, not ranked."""
    hp = jnp.sum(state.hp[0].astype(_F32)) / jnp.maximum(jnp.sum(state.max_hp[0].astype(_F32)), 1)
    full = jnp.sum(tables.move_pp[state.moves[0]].astype(_F32))
    pp = jnp.sum(state.pp[0].astype(_F32)) / jnp.maximum(full, 1.0)
    return hp.astype(_F32), pp.astype(_F32)


def _select(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


def _facing(state: BattleState, side, action) -> BattleState:
    """``state`` with ``side``'s switch ``action`` made (the incoming Pokemon active,
    stages reset), as the other side's move will meet it; unchanged otherwise."""
    switching = is_switch(action)
    slot = jnp.where(switching, action - MOVE_ACTIONS, state.active[side])
    stages = jnp.where(switching, 0, state.stages[side])
    return state._replace(
        active=state.active.at[side].set(slot.astype(state.active.dtype)),
        stages=state.stages.at[side].set(stages.astype(state.stages.dtype)),
    )


def _move_odds(tables: Tables, state: BattleState, side, actions, hit):
    """``(P(KO), expected fraction of the target's HP removed)`` of ``side``'s action."""
    action = actions[side]
    target = _facing(state, 1 - side, actions[1 - side])
    move = used_move(tables, state, side, action)  # Nature Power: its call
    damage = damage_table(tables, target, side, move, False)  # [16] non-critical rolls
    hp = target.hp[1 - side, target.active[1 - side]].astype(_I32)
    hp_ = jnp.maximum(hp, 1).astype(_F32)
    p = jnp.where(action < MOVE_ACTIONS, hit, 0.0)
    ko = p * jnp.mean((damage >= hp).astype(_F32))
    removed = p * jnp.mean(jnp.minimum(damage, hp).astype(_F32)) / hp_
    return ko, removed


def death_cost(state: BattleState, slot) -> jnp.ndarray:
    """``float32``: what the faint of the player's ``slot`` costs the objective (plan
    ruling 6). 1 when it is the player's only living Pokemon (its faint is the whiteout);
    1/2 in a failed state (``deaths.over_budget``: the line already counts for nothing,
    so a faint is a mild loss, not a total one); 1 when ``slot`` is protected or no
    budget is left; else ``1 / (r + 1)`` with ``r`` the deaths left. With ``faint_budget``
    0 and nothing failed it is 1, today's value."""
    living = (state.species[0] > 0) & (state.hp[0] > 0)
    left = state.faint_budget.astype(_I32) - state.player_faints.astype(_I32)
    spare = 1.0 / (jnp.maximum(left, 0).astype(_F32) + 1.0)
    cost = jnp.where((state.protected[slot] != 0) | (left <= 0), 1.0, spare)
    cost = jnp.where(over_budget(state), 0.5, cost)
    return jnp.where(jnp.sum(living) <= 1, 1.0, cost).astype(_F32)


def _pair_value(tables: Tables, state: BattleState, a, b) -> jnp.ndarray:
    actions = jnp.stack([a, b]).astype(_I32)
    # A policy heuristic, not an exact law: the turn-start threshold on purpose.
    hit = jnp.clip(hit_threshold(tables, state, actions), 0, 100).astype(_F32) / 100
    ko_player, removed = _move_odds(tables, state, 0, actions, hit[0])
    ko_opp, _ = _move_odds(tables, state, 1, actions, hit[1])
    ties = [empty_outcome()._replace(speed_tie=jnp.int8(tie)) for tie in (0, 1)]
    player_first = [action_order(tables, state, actions, o)[0] == 0 for o in ties]
    first = jnp.mean(jnp.stack(player_first).astype(_F32))
    risk = ko_opp * (1.0 - first * ko_player)
    # The slot that takes the opponent's move: the incoming one for a switch.
    slot = jnp.where(is_switch(a), a - MOVE_ACTIONS, state.active[0]).astype(_I32)
    return (1.0 - death_cost(state, slot) * risk) * (1.0 + removed)


@jax.jit
def _choose(tables: Tables, state: BattleState, q, key, allowed=None) -> jnp.ndarray:
    """``int32``: the best legal action against ``q``; ``allowed`` (``bool[NUM_ACTIONS]``,
    ``None`` = all) narrows the legal actions, and is ignored when it leaves none."""
    legal = legal_actions(tables, state, 0)
    if allowed is not None:
        narrowed = legal & allowed
        legal = jnp.where(jnp.any(narrowed), narrowed, legal)
    mine, theirs = jnp.arange(NUM_ACTIONS), jnp.arange(NUM_OPP_ACTIONS)
    row = jax.vmap(lambda a, b: _pair_value(tables, state, a, b), in_axes=(None, 0))
    values = jax.vmap(lambda a: row(a, theirs))(mine)  # [NUM_ACTIONS, NUM_OPP_ACTIONS]
    score = jnp.where(legal, values @ q, -jnp.inf)
    tied = score == jnp.max(score)
    return jax.random.choice(key, NUM_ACTIONS, p=tied / jnp.sum(tied)).astype(_I32)


def _policy(
    tables: Tables, state: BattleState, key, layout: AiLayout = VANILLA, allowed=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(action, refused)``: ``playout_policy``'s action, and whether the opponent's
    distribution it was chosen against is refused (traced, a refused distribution is
    NaN; the action is then meaningless and the caller must refuse the playout).
    ``allowed`` narrows the player's actions as in ``_choose`` (``None`` = all)."""
    q = ai_action_distribution(tables, state, layout)
    refused = jnp.any(jnp.isnan(q))
    return _choose(tables, state, jnp.where(refused, 0.0, q), key, allowed), refused


def playout_policy(
    tables: Tables, state: BattleState, key, layout: AiLayout = VANILLA
) -> jnp.ndarray:
    """``int32``: the player's action with the highest expected value (module docstring),
    a uniformly random one among exact ties; illegal actions score -inf. Raises
    ``UnsupportedAi`` on a concrete state whose AI is refused; traced, the action on a
    refused distribution is meaningless (``_run`` refuses the playout instead)."""
    return _policy(tables, state, key, layout)[0]


@jax.jit
def send_replacements(
    tables: Tables, state: BattleState, ai_switched, power_mul
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, refused)``: the opponent's fainted active Pokemon replaced by
    ``post_ko_switch_in`` (``power_mul`` is the turn's ``battleCtx->powerMul``), again
    while each newcomer faints on entry (Stealth Rock), as ``TurnEnd`` re-enters
    ``ReplaceFainted``. At a pass where the player's active is down too, it stops before
    anyone enters (the player enters first, ``replacement_pass``), keeping ``power_mul``
    in ``replace_mul`` when the opponent's active is down as well (else 0). ``refused``
    (``bool``) when an opponent replacement is needed after the AI switched this turn
    (``ai_switched``) or ``post_ko_switch_in`` refuses, also when it stops; the state is
    then left as it was at that point."""

    def once(_, carry):  # a loop, not unrolled: one copy of the replacement rule
        state, refused = carry
        need = needs_replacement(state)
        stop = need[0] == 1
        slot = post_ko_switch_in(tables, state, power_mul)
        bad = (need[1] == 1) & (jnp.asarray(ai_switched) | (slot < 0))
        replaced = replace(tables, state, 1, jnp.maximum(slot, 0))
        opp_down = state.hp[1, state.active[1]] == 0
        held = state._replace(
            replace_mul=jnp.where(opp_down, power_mul, 0).astype(state.replace_mul.dtype)
        )
        nxt = _select(stop, held, _select((need[1] == 1) & ~bad, replaced, state))
        return nxt, refused | bad

    return jax.lax.fori_loop(0, SLOTS - 1, once, (state, jnp.bool_(False)))


def _carried_mul(state: BattleState) -> jnp.ndarray:
    """The turn's ``powerMul`` at a player replacement: ``replace_mul``, else normal."""
    return jnp.where(state.replace_mul > 0, state.replace_mul, NORMAL_POWER_MUL).astype(_I32)


@jax.jit
def replacement_pass(tables: Tables, state: BattleState, slot) -> tuple[BattleState, jnp.ndarray]:
    """``(state, refused)``: one pass of ``ReplaceFainted`` at a player replacement
    (module docstring): the opponent's slot is picked on this state (against the fainted
    battler) when its active is down, the player's ``slot`` (a ``legal_replacements(state,
    0)`` slot) enters, then the opponent's, ``replace_mul`` is cleared and
    ``send_replacements`` settles what the entries leave (an opponent newcomer fainting on
    Stealth Rock). ``refused`` (``bool``) when the opponent's pick is refused.

    A player newcomer that entered while the opponent's active was down (a double KO) has
    not announced its Intimidate (``replace`` found no living foe):
    ``subscript_show_party_list.s`` only sends every switcher in with the hazards and a
    faint check, and ``ShowBattleMon``'s ``BattleSystem_TriggerEffectOnSwitch``
    (battle_controller_player.c:256, ``SWITCH_IN_CHECK_STATE_INTIMIDATE``,
    battle_lib.c:3795-3808) fires it later against the living opposing newcomer. So it
    fires once the opponent's side is settled (the two Intimidates touch different
    battlers, so their speed order does not matter). A Trace newcomer that only copies
    Intimidate when the opponent's newcomer enters fires it there (``hooks.on_switch_in``):
    the deferred one is the newcomer's own Intimidate at its entry.

    Download (``hooks.download``, ``SWITCH_IN_CHECK_STATE_DOWNLOAD``, battle_lib.c:3814-3851,
    after the Intimidate state and before the held-item state, ruling G-22) likewise: the
    player newcomer's fires once the opponent's side is settled, reading the opposing newcomer's
    Defense and Sp. Def at stage 0 (a fresh battler; only its own held-item check, a later
    state, can have raised them since). Each opposing newcomer's Download, run at its entry,
    reads the player newcomer's two stats at stage 0 too: their stages are set aside for the
    opponent's entries (nothing an opposing switch-in does touches them) and put back after."""
    mul = _carried_mul(state)
    opp_needs = needs_replacement(state)[1] == 1
    opp_slot = post_ko_switch_in(tables, state, mul)
    bad = opp_needs & (opp_slot < 0)
    entered = replace(tables, state, 0, slot)
    opp_down = state.hp[1, state.active[1]] == 0
    own = entered.ability[0, entered.active[0]]
    unannounced = opp_down & (own == AbilityId.INTIMIDATE)
    download_unannounced = opp_down & (own == AbilityId.DOWNLOAD)
    kept = entered.stages[0]
    fresh = entered.stages.at[0, Stat.DEF].set(0).at[0, Stat.SPD].set(0)
    entered = entered._replace(stages=jnp.where(opp_down, fresh, entered.stages))
    both = replace(tables, entered, 1, jnp.maximum(opp_slot, 0))
    entered = _select((needs_replacement(entered)[1] == 1) & ~bad, both, entered)
    entered = entered._replace(replace_mul=jnp.zeros_like(entered.replace_mul))
    settled, refused = send_replacements(tables, entered, False, mul)
    put_back = (
        settled.stages.at[0, Stat.DEF].set(kept[Stat.DEF]).at[0, Stat.SPD].set(kept[Stat.SPD])
    )
    settled = settled._replace(stages=jnp.where(opp_down, put_back, settled.stages))
    settled = intimidate(settled, 0, unannounced & (settled.over == 0))
    settled = download(settled, 0, (1,), download_unannounced & (settled.over == 0), staged=False)
    return settled, refused | bad


@jax.jit
def replacement_policy(tables: Tables, state: BattleState) -> jnp.ndarray:
    """``int32``: the party slot the player sends in at a replacement (ruling 7). Each
    legal slot scores the best, over its usable moves, of the mean over the opponent
    active's legal moves of ``_pair_value`` with that Pokemon active (stages reset)
    against the opponent's active, or against the slot the AI sends in when its active is
    down too; ties go to the lowest slot. No key, no AI distribution."""
    opp_down = state.hp[1, state.active[1]] == 0
    o = jnp.maximum(post_ko_switch_in(tables, state, _carried_mul(state)), 0)
    versus = _facing(state, 1, jnp.where(opp_down, MOVE_ACTIONS + o, 0))
    moves = jnp.arange(MOVE_ACTIONS)

    def score(c):
        mine = _facing(versus, 0, MOVE_ACTIONS + c)
        mine_ok = legal_actions(tables, mine, 0)[:MOVE_ACTIONS]
        theirs_ok = legal_actions(tables, mine, 1)[:MOVE_ACTIONS]
        row = jax.vmap(lambda a, b: _pair_value(tables, mine, a, b), in_axes=(None, 0))
        values = jax.vmap(lambda a: row(a, moves))(moves)  # [MOVE_ACTIONS, MOVE_ACTIONS]
        mean = (values * theirs_ok).sum(axis=1) / jnp.maximum(theirs_ok.sum(), 1)
        return jnp.max(jnp.where(mine_ok, mean, -jnp.inf))

    scores = jax.vmap(score)(jnp.arange(SLOTS))
    scores = jnp.where(legal_replacements(state, 0), scores, -jnp.inf)
    return jnp.argmax(scores).astype(_I32)  # first maximum: the lower slot


@jax.jit
def finish_turn(
    tables: Tables, state: BattleState, ai_switched, power_mul
) -> tuple[BattleState, jnp.ndarray]:
    """``(state, refused)``: ``send_replacements``, then while the player needs a
    replacement (``SLOTS - 1`` passes at most) ``replacement_pass`` with
    ``replacement_policy``'s slot. The turn's learn-move prompts are resolved first
    (``resolve_learns``), as ``TurnEnd`` runs them before ``ReplaceFainted``."""
    state = resolve_learns(tables, state)
    state, refused = send_replacements(tables, state, ai_switched, power_mul)
    state, refused = _replace_while_needed(tables, state, refused, SLOTS - 1)
    return state, refused | (state.exp_refused != 0)


def _replace_while_needed(tables: Tables, state: BattleState, refused, passes):
    """``(state, refused)``: while the player needs a replacement (``passes`` at most)
    ``replacement_pass`` with ``replacement_policy``'s slot. A ``while_loop``, so a batch
    in which nobody needs one runs no pass at all."""

    def cond(carry):
        state, _, n = carry
        return (needs_replacement(state)[0] == 1) & (n < passes)

    def body(carry):
        state, refused, n = carry
        passed, bad = replacement_pass(tables, state, replacement_policy(tables, state))
        return passed, refused | bad, n + 1

    state, refused, _ = jax.lax.while_loop(cond, body, (state, refused, jnp.int32(0)))
    return state, refused


def pivot_pick(tables: Tables, state: BattleState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(actions, refused)`` that resume ``state`` paused at a party list (E2 batch 6): the
    pivoting side's switch to its pick, the other side's action as chosen. The opponent's pick is
    ``BattleAI_PostKOSwitchIn`` on the paused state with the list's powerMul (``ts_pick_mul``;
    ``aiSwitchedPartySlot`` is 6 there, ``Task_TrainerShowPartyMenu``, battle_display.c:
    4543-4556), refused as at a replacement; the player's is ``replacement_policy``'s (a search
    heuristic)."""
    pivot = jnp.clip(state.pivot_battler.astype(_I32) - 1, 0, 1)
    opp = post_ko_switch_in(tables, state, state.ts_pick_mul.astype(_I32))
    slot = jnp.where(pivot == 1, jnp.maximum(opp, 0), replacement_policy(tables, state))
    actions = state.ts_actions.astype(_I32).at[pivot].set(MOVE_ACTIONS + slot)
    return actions, (state.pivot_battler == 2) & (opp < 0)


def turn_muls(tables: Tables, state: BattleState, actions) -> tuple[jnp.ndarray, jnp.ndarray]:
    """``(power_mul, ai_switched)`` of a turn from ``state``: ``replacement_power_mul`` and the
    opponent's switch; at a pause (E2 batch 6) the turn start's (``ts_power_mul``,
    ``ts_switched[1]``): the resume's actions are switch-shaped."""
    actions = jnp.asarray(actions, _I32)
    paused = state.pivot_battler != 0
    power_mul = jnp.where(
        paused, state.ts_power_mul.astype(_I32), replacement_power_mul(tables, state, actions)
    )
    ai_switched = jnp.where(paused, state.ts_switched[1] != 0, is_switch(actions[1]))
    return power_mul, ai_switched


def turn_inputs(tables: Tables, state: BattleState, actions):
    """``(actions, refused, power_mul, ai_switched)`` of a turn from ``state`` (``expand``'s row
    program): ``turn_muls``, and at the opponent's own list its pick replaces its action
    (``pivot_pick``, ``refused`` when the AI refuses it)."""
    actions = jnp.asarray(actions, _I32)
    picked, bad = pivot_pick(tables, state)
    opp_list = state.pivot_battler == 2
    actions = jnp.where(opp_list, picked, actions)
    power_mul, ai_switched = turn_muls(tables, state, actions)
    return actions, opp_list & bad, power_mul, ai_switched


def step_through(tables: Tables, state: BattleState, actions, first_key, key, pick=-1):
    """``(state, refused)``: the turn from ``state`` to its end. Each pass of one loop makes the
    pick when the state is paused at a party list (``pivot_pick``; at the first pass ``pick``, a
    switch action, overrides the player's own), draws ``sample_outcome`` (``first_key`` at the
    first pass, then from ``key``: each half reads its own draws) and plays ``step``; it loops
    while ``step`` pauses (at most twice a turn). One ``step``, one ``sample_outcome`` and one
    ``pivot_pick`` in the program. ``refused``: the AI refused its pick."""

    def cond(carry):
        s, _, _, refused, n = carry
        return (n == 0) | ((s.pivot_battler != 0) & ~refused & (n < 3))

    def body(carry):
        s, a, k, refused, n = carry
        paused = s.pivot_battler != 0
        picked, bad = pivot_pick(tables, s)
        mine = (s.pivot_battler == 1) & (n == 0) & (pick >= MOVE_ACTIONS)
        picked = jnp.where(mine, picked.at[0].set(pick), picked)
        a = jnp.where(paused, picked, a)
        k, k_half = jax.random.split(k)
        k_half = jnp.where(n == 0, first_key, k_half)
        played = step(tables, s, a, sample_outcome(k_half, tables, s, a))
        return played, a, k, refused | (paused & bad), n + 1

    init = (state, jnp.asarray(actions, _I32), key, jnp.bool_(False), jnp.int32(0))
    state, _, _, refused, _ = jax.lax.while_loop(cond, body, init)
    return state, refused


def _play(
    tables: Tables, state: BattleState, player_action, key, layout: AiLayout = VANILLA, pick=-1
):
    """One turn after the player's choice, before the opponent's replacement:
    ``(state after step, ai_refused, ai_switched, power_mul)``. A pause at a party list is
    resumed in the turn (``step_through``). From a paused state (a playout's root) the turn is
    its resume: ``pick`` (a switch action) is the player's pick at its own list, else
    ``pivot_pick``'s; the AI's action is not used."""
    k_ai, k_turn = jax.random.split(key)
    ai_outcome = sample_ai_outcome(k_ai, tables, state, layout)
    opp_action = ai_policy(tables, state, ai_outcome, layout)
    paused = state.pivot_battler != 0
    ai_refused = (opp_action < 0) & ~paused  # switching.REFUSED_ACTION
    actions = jnp.stack([player_action, jnp.maximum(opp_action, 0)]).astype(_I32)
    power_mul, ai_switched = turn_muls(tables, state, actions)
    played, pick_refused = step_through(
        tables, state, actions, k_turn, jax.random.fold_in(k_turn, 1), pick
    )
    return played, ai_refused | pick_refused, ai_switched, power_mul


def _turn(
    tables: Tables, state: BattleState, player_action, key, layout: AiLayout = VANILLA, pick=-1
):
    """One turn after the player's choice: ``(state, refused)``."""
    played, refused, ai_switched, power_mul = _play(tables, state, player_action, key, layout, pick)
    played, replace_refused = finish_turn(tables, played, ai_switched, power_mul)
    refused = refused | replace_refused
    return _select(refused, state, played), refused


def _fixed(turns, first_action):
    return (turns == 0) & (first_action >= 0)


def _first_action(turns, first_action, chosen):
    return jnp.where(_fixed(turns, first_action), first_action, chosen)


def _resolve_pending(tables: Tables, state: BattleState, first_action):
    """``(state, first_action, refused)``: a pending player replacement resolved first
    (a switch ``first_action`` chooses the slot, else ``replacement_policy``); the policy
    then chooses from the first turn on (``first_action`` -1). A pending learn-move prompt
    and a pending opponent replacement are settled before the player's (``settle_learns``)."""
    state, learn_refused = settle_learns(tables, state, send_replacements, _carried_mul)
    pending = needs_replacement(state)[0] == 1
    chosen = jnp.where(
        first_action >= MOVE_ACTIONS,
        first_action - MOVE_ACTIONS,
        replacement_policy(tables, state),
    )
    passed, refused = replacement_pass(tables, state, chosen.astype(_I32))
    state = _select(pending, passed, state)
    refused = (pending & refused) | learn_refused
    # A newcomer that faints on entry (Stealth Rock) leaves it pending again: finish_turn's
    # loop, with this pass counted in its SLOTS - 1.
    state, refused = _replace_while_needed(tables, state, refused, SLOTS - 2)
    return state, jnp.where(pending, -1, first_action).astype(_I32), refused


def _run(
    tables: Tables,
    state: BattleState,
    key,
    max_turns: int,
    first_action,
    layout: AiLayout = VANILLA,
):
    # A root paused at a party list (E2 batch 6) has nothing pending before its resume: its
    # fainted actives wait for the turn's end.
    paused = state.pivot_battler != 0
    settled, settled_first, settled_refused = _resolve_pending(tables, state, first_action)
    state = _select(paused, state, settled)
    first_action = jnp.where(paused, first_action, settled_first)
    refused0 = ~paused & settled_refused

    def cond(carry):
        s, _, turns, refused = carry
        return (s.over == 0) & (turns < max_turns) & ~refused

    def body(carry):
        s, key, turns, _ = carry
        key, k_policy, k_turn = jax.random.split(key, 3)
        chosen, q_refused = _policy(tables, s, k_policy, layout)
        # A refused distribution refuses the turn whenever the player's choice uses it.
        q_refused = q_refused & ~_fixed(turns, first_action)
        action = _first_action(turns, first_action, chosen)
        # A paused root (E2 batch 6) resumes first: the rest of its turn, not a turn of its own.
        resumed = s.pivot_battler != 0
        q_refused = q_refused & ~resumed
        pick = jnp.where(_fixed(turns, first_action), first_action, -1)
        played, refused = _turn(tables, s, action.astype(_I32), k_turn, layout, pick)
        refused = refused | q_refused
        s = _select(q_refused, s, played)
        return s, key, turns + jnp.where(refused | resumed, 0, 1), refused

    init = (state, key, jnp.int32(0), refused0)
    s, _, turns, refused = jax.lax.while_loop(cond, body, init)
    won = (s.over != 0) & (s.winner == WINNER_PLAYER)
    hp_left, pp_left = _resources_left(tables, s)
    return PlayoutResult(
        won=jnp.where(refused, REFUSED_WON, won).astype(jnp.int8),
        player_faints=s.player_faints.astype(jnp.int8),
        turns=turns.astype(jnp.int16),
        refused=refused.astype(jnp.int8),
        deaths=s.deaths.astype(jnp.int8),
        ended=(s.over != 0).astype(jnp.int8),
        hp_left=jnp.where(refused, 0.0, hp_left).astype(_F32),
        pp_left=jnp.where(refused, 0.0, pp_left).astype(_F32),
    )


@partial(jax.jit, static_argnames=("max_turns", "layout"))
def playout(
    tables: Tables, state: BattleState, key, max_turns: int = 100, layout: AiLayout = VANILLA
) -> PlayoutResult:
    """Play from ``state`` with ``playout_policy`` for the player (module docstring); a
    pending player replacement is resolved first by ``replacement_policy``."""
    return _run(tables, state, key, max_turns, jnp.int32(-1), layout)


@partial(jax.jit, static_argnames=("max_turns", "layout"))
def playout_from_action(
    tables: Tables,
    state: BattleState,
    action,
    key,
    max_turns: int = 100,
    layout: AiLayout = VANILLA,
) -> PlayoutResult:
    """``playout`` with the player's first action fixed to ``action`` (a legal action; at
    a pending player replacement ``MOVE_ACTIONS + slot`` is that replacement and the policy
    chooses from the first turn on);
    the same ``key`` gives every first action the same draws (common random numbers)."""
    return _run(tables, state, key, max_turns, jnp.asarray(action, _I32), layout)


@partial(jax.jit, static_argnames=("max_turns", "layout"))
def batched_playout(
    tables: Tables, states: BattleState, keys, max_turns: int = 100, layout: AiLayout = VANILLA
):
    """``playout`` over a leading batch axis of ``states`` and ``keys`` (``jax.vmap``,
    ``tables`` shared); ``max_turns`` and ``layout`` are static."""
    return jax.vmap(lambda s, k: playout(tables, s, k, max_turns, layout))(states, keys)


def _opponent_reason(tables: Tables, state: BattleState, ai_switched, power_mul) -> str | None:
    """Why the opponent's replacement on the concrete ``state`` is refused, if it is needed."""
    if int(needs_replacement(state)[1]) != 1:
        return None
    if bool(ai_switched):
        return (
            "replacement: the opponent's Pokemon fainted in a turn its AI switched; "
            "Task_TrainerShowPartyMenu reads the stale aiSwitchedPartySlot "
            "(docs/ai/roark.md 14, item 3); not implemented"
        )
    try:
        post_ko_switch_in(tables, state, power_mul)
    except UnsupportedAi as e:
        return str(e)
    return None


def _send_reason(tables: Tables, state: BattleState, ai_switched, power_mul):
    """``(reason, state)``: ``send_replacements`` on the concrete ``state``, outside
    ``jit``; ``reason`` is why it refuses, or ``None``."""
    for _ in range(SLOTS - 1):
        reason = _opponent_reason(tables, state, ai_switched, power_mul)
        if reason:
            return reason, state
        need = needs_replacement(state)
        if int(need[0]) == 1:
            held = power_mul if int(state.hp[1, state.active[1]]) == 0 else 0
            return None, state._replace(replace_mul=jnp.asarray(held, state.replace_mul.dtype))
        if int(need[1]) != 1:
            break
        state = replace(tables, state, 1, post_ko_switch_in(tables, state, power_mul))
    return None, state


def _pass_reason(tables: Tables, state: BattleState, slot: int):
    """``(reason, state)``: ``replacement_pass`` on the concrete ``state``, outside ``jit``."""
    mul = int(_carried_mul(state))
    reason = _opponent_reason(tables, state, False, mul)
    if reason:
        return reason, state
    opp_slot = (
        post_ko_switch_in(tables, state, mul) if int(needs_replacement(state)[1]) == 1 else None
    )
    state = replace(tables, state, 0, slot)
    if opp_slot is not None and int(needs_replacement(state)[1]) == 1:
        state = replace(tables, state, 1, opp_slot)
    state = state._replace(replace_mul=jnp.zeros_like(state.replace_mul))
    return _send_reason(tables, state, False, mul)


def replacement_refusal(tables: Tables, state: BattleState, ai_switched, power_mul) -> str | None:
    """Why ``finish_turn`` refuses the concrete ``state`` (a state after ``step``), or
    ``None`` when it does not; mirrors it outside ``jit``."""
    reason, state = _send_reason(tables, state, ai_switched, power_mul)
    for _ in range(SLOTS - 1):
        if reason or int(needs_replacement(state)[0]) != 1:
            return reason
        reason, state = _pass_reason(tables, state, int(replacement_policy(tables, state)))
    return reason


_play_jit = jax.jit(_play, static_argnames="layout")
_turn_jit = jax.jit(_turn, static_argnames="layout")
_policy_jit = jax.jit(_policy, static_argnames="layout")


def refusal_reason(
    tables: Tables,
    state: BattleState,
    action,
    key,
    max_turns: int = 100,
    layout: AiLayout | None = None,
) -> str | None:
    """Why ``playout_from_action(tables, state, action, key, max_turns, layout)`` is
    refused, or ``None`` when it is not: the playout is replayed turn by turn outside the
    loop (the same key splits) and the refused turn's AI (``ai_refusals``, then the
    per-slot scoring path's ``ai_path_refusals``, also when the
    player's choice met a refused distribution) or replacement (``replacement_refusal``) is
    asked on its concrete state (after the turn's learns, as ``finish_turn``); a pending
    learn and a pending player replacement are replayed first (``settle_reason``, as
    ``_resolve_pending``).
    ``layout`` defaults to the concrete ``tables``' own (``layout_of``)."""
    layout = layout_of(tables) if layout is None else layout
    first = jnp.asarray(action, _I32)
    reason, state = settle_reason(tables, state, _send_reason, _carried_mul)
    if reason:
        return reason
    if int(needs_replacement(state)[0]) == 1:
        slot = int(first) - MOVE_ACTIONS if int(first) >= MOVE_ACTIONS else None
        if slot is None:
            slot = int(replacement_policy(tables, state))
        reason, state = _pass_reason(tables, state, slot)
        for _ in range(SLOTS - 2):
            if reason or int(needs_replacement(state)[0]) != 1:
                break
            reason, state = _pass_reason(tables, state, int(replacement_policy(tables, state)))
        if reason:
            return reason
        first = jnp.int32(-1)
    for turns in range(max_turns):
        if int(state.over) != 0:
            return None
        key, k_policy, k_turn = jax.random.split(key, 3)
        chosen, q_refused = _policy_jit(tables, state, k_policy, layout)
        q_refused = bool(q_refused) and not bool(_fixed(turns, first))
        player_action = _first_action(turns, first, chosen).astype(_I32)
        played, refused = _turn_jit(tables, state, player_action, k_turn, layout)
        if q_refused or bool(refused):
            reasons = ai_refusals(tables, state, layout=layout) or ai_path_refusals(
                tables, state, layout=layout
            )
            if reasons:
                return "; ".join(reasons)
            if q_refused:
                return "the opponent's action distribution is refused"
            played, _, ai_switched, power_mul = _play_jit(
                tables, state, player_action, k_turn, layout
            )
            return learn_reason(
                finish_turn(tables, played, ai_switched, power_mul)[0]
            ) or replacement_refusal(tables, resolve_learns(tables, played), ai_switched, power_mul)
        state = played
    return None
