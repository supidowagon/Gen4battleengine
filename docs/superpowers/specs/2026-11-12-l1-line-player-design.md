# Milestone L1: a line-shaped player for hard fights — design

Status: design by the managing session's brief, 2026-11-12. Parent: `docs/design.md` §8 item 4a
(line quality). Order agreed with the user on 2026-11-12: this milestone first, then an
exact-solve experiment, then mid-fight level-ups (K3d), then prep speed and breadth.

## 1. The problem, measured

On Kaizo Roark (trainer 246) from the pre-Roark save, `nuzline prep` ran 8 h 50 m and reported
0.3 % deathless, 1.4 % any win, 95 % whiteout. The user then beat the fight in game with one death.
Their line, scripted as "one answer per opposing Pokemon" and run in the engine 200 times, wins
72 % and wins with at most one death 42.5 %. `playout_policy` with the same five Pokemon wins
8-9 %, 2 % with at most one death. The engine matches the game; the simulated player is what is
wrong. `playout_policy` values one turn only. It never takes a hit to set up a switch, never
sacrifices, never uses a status move.

## 2. What L1 builds

1. **A line**: per opposing Pokemon, which of the player's Pokemon answers it and a small rule for
   what it does. Where a line has no answer, the existing `playout_policy` and
   `replacement_policy` decide. Only the player's choices change: the trainer AI, `step` and the
   chance model are not touched.
2. **A search over lines** for a given team, scored by playouts, so the tool finds a good line
   itself.
3. **Two commands**: `nuzline line odds` scores a line the user gives; `nuzline line find`
   searches for one.

The word is "line", not "plan": `search.plan_tree` already means the Stage B tree.

**Not in L1.** Solve-from-here (the next milestone). Using a line inside `prep` or as the Stage B
leaf player (see §8). Choosing the team, the lead, levels or items: the team is an input.
Mid-fight level-ups and experience (K3d). Any change to a battle rule.

## 3. The line

A line has one optional rule per opposing party slot (0-5). A rule has:

| field    | meaning                                                                              |
|----------|--------------------------------------------------------------------------------------|
| `use`    | the player's party slot that answers this foe                                        |
| `move`   | the answer's move slot used against it                                               |
| `first`  | optional move slot: used instead of `move` while the foe has no major status         |
| `via`    | optional party slot: when the answer must switch in on a turn, go through this one   |
| `from`   | optional party slot: `via` applies only when this slot is the active one             |
| `wait`   | optional flag: the answer never spends a turn switching in; see below                |

A turn's choice, with `foe` the opponent's active slot and `me` the player's active slot:

1. No rule for `foe`, or its `use` has fainted: the fallback decides (`playout_policy`).
2. `me == use`: play `first` if it is set and the foe's `status` is 0, else `move`. If `first` is
   not legal this turn (no PP, Taunt, Disable), play `move`.
3. `me != use`, `wait` set: the fallback decides among the active Pokemon's moves only (no
   switch). The active Pokemon fights on; the answer comes in on the free switch after a faint.
4. `me != use`, `via` set and alive, `me != via`, and (`from` unset or `me == from`): switch to
   `via`.
5. Otherwise switch to `use`.

Whenever the action so chosen is not legal (`legal_actions`: trapped, locked into a move, no PP),
the fallback decides that turn. This is why a rule can never make a playout illegal.

A replacement after the player's Pokemon faints (Set mode): the foe is the opponent's active
Pokemon, or, when it is down too, the slot `post_ko_switch_in` will send (as `replacement_policy`
already reads it). If that foe's rule has a living `use` that may enter, it enters; otherwise
`replacement_policy` decides. `via` and `wait` do not apply: the switch is free.

Why these fields. `use`/`move` is the user's "one answer per Pokemon". `first` is Toxic-then-attack.
`via`/`from` is the user's Wormadam -> Rhyhorn -> Delibird pivot against Gible: a hard switch that
routes the opponent's hit onto a Pokemon that takes it well. `wait` is the Set-mode sacrifice: let
the active Pokemon fall and bring the answer in for free. None of these is a battle rule; all are
player choices, so nothing here needs a decomp citation. The rules the choices are checked against
(`legal_actions`, `legal_replacements`) already carry theirs.

`first` is kept while the foe has no major status, so on a foe that cannot be statused it would
repeat until its PP runs out. The search only offers `first` where it scores well; a user-written
line gets what it says. A counted opener ("Swords Dance twice") is left for later.

### In code

`src/nuzline/search/line.py`. A `Line` is a NamedTuple of six `int32[6]` arrays indexed by opposing
slot (`use`, `move`, `first`, `via`, `via_from`, `wait`; -1 = unset, `wait` 0/1), so a batch of
lines is data: one compiled program plays any line, and `vmap` runs many lines on the same keys.

- `line_action(tables, state, line) -> (action, mode)`: `mode` 0 = fallback, 1 = the rule's action,
  2 = fallback restricted to moves (`wait`).
- `line_replacement(tables, state, line) -> slot`.
- `line_playout(tables, state, line, key, max_turns, layout) -> PlayerResult`: `playout._run`'s loop
  with the two choices above. `PlayerResult` is `PlayoutResult`'s fields plus `fallback_turns`
  (turns the fallback decided). It draws the root's hidden values first (`root_state`), like Stage
  A's root playouts, and uses the same key splits per turn as `playout`, so with an empty line it
  returns exactly what `playout` returns from the same root and key. That equality is a test.
- `playout._choose` gains a mask of allowed actions (default: all) for mode 2. With the default it
  must return what it returns today (the K0 regression and playout tests pin this).

A refused playout stays refused (`refused = 1`, `won = REFUSED_WON`) and the caller refuses loudly,
as Stage A does.

## 4. Scoring a line

`evaluate_lines(tables, state, lines, n, key) -> list[PlayerOdds]` in `src/nuzline/search/player_eval.py`:
every line against the same `n` keys (common random numbers), chunked and sharded over the host
devices like `stage_a.rank_actions`, through `cached_export`. `PlayerOdds` holds, from `n` playouts:
the ladder P(win with at most k deaths) for k = 0..3 and P(win), each with its Wilson interval
(`stage_a.wilson`), the death distribution (`deaths.DeathDistribution`), mean turns and the share of
turns the fallback decided.

A refused playout (one that meets a mechanic the engine deliberately does not model) is never
counted as a win or a loss. `line odds` and `line find` leave it out, so every rate is over the
playouts counted, and print how many were left out and which mechanic (replayed outside `jit`).
On Kaizo Roark this happens about once in 2000 playouts on some lines: Roark's AI switches and
the incoming Pokemon faints the same turn (`docs/ai/roark.md` 14, item 3). The search never
chooses a rule whose playouts are refused more than 1 % of the time.

The state's `faint_budget` is the `--deaths K` the user passes. It changes how the fallback weighs a
faint (`death_cost`) and which rung is the headline; the whole ladder is always shown. The user's
measurement was made with K = 3, so the reproduction is run with `--deaths 3`.

## 5. Searching for a line

`find_line(tables, names, state, key, settings) -> LineSearchResult` in
`src/nuzline/search/line_search.py`. Host-side Python around `evaluate_lines`.

**Candidates per foe.** `use`: every living party slot. `move`: each of its moves that can damage
that foe (`damage_table` at the turn-0 state has a positive roll). `first`: none, or one of its
power-0 moves whose effect inflicts a major status (read from the tables' effect ids, not from a
list of names). Entry: plain, `wait`, or `via` each other living slot (`from` unset). And "no
rule". `from` is not searched: it only narrows `via`, and the user can write it.

**Score.** Wins with at most K deaths, then wins, then fewer total deaths, compared on the same
keys. A loss is never traded for fewer deaths (prep spec §2).

**Procedure** (revised 2026-11-12 after the first measurement on Kaizo Roark: the first version
tried entries only for the Pokemon that won with a plain switch-in, so the user's pivot against
Gible was never scored; three independent designs and a judge chose this one). Coordinate ascent
with common random numbers:

1. Start from the empty line (pure fallback), or from the user's rules when the file has some.
2. A sweep visits each foe: odd sweeps in party order, even sweeps in reverse, so the first foes
   are not always decided against default play on the rest. A visit races "no rule" and every
   rule of who x what x entry (plain, `wait`, `via` each other living Pokemon) with the rest of
   the line fixed: all on 64 keys, the best 16 on 192 more, the best 4 on 256 more (ranked by
   wins with at most k deaths plus wins, on cumulative common keys, the incumbent always in);
   then the single best and the incumbent on 768 fresh keys, where it replaces the incumbent only
   if it scores strictly better. A winning `via` is then polished over `from`.
3. Sweeps repeat until one changes nothing, 12 sweeps, or the time budget.
4. The line at the end of every sweep is kept; `restarts` "kicks" reset two random foes of the
   best line and ascend again; the kept lines are compared on 4000 fresh keys (`pick_n`), both to
   choose a kick's start and at the end. Kept lines are often a point or two apart, which 768
   keys cannot separate (measured on Kaizo Roark: a 768-key pick kept a line tied with the
   user's although an earlier kept line scored 47.5 % to 45.75 % on 20 000 shared keys).
5. The chosen line is scored once more on `final_n` keys the search never saw. Only these numbers
   are reported: picking the best of many noisy scores flatters the pick, a fresh sample does not.

Defaults (`LineSearchSettings`): race ((64, all), (192, 16), (256, 4)), `accept_n` 768,
`pick_n` 4000, `max_sweeps` 12, `restarts` 1, `final_n` 2000, `budget_seconds` None. A sweep costs about 73 000
playouts (about 200 s on 20 CPU devices).

The search is a heuristic: it finds a good line, not the best one, and its report says so. It
never calls a fight "solved".

## 6. Commands

Both take `--game`, `--trainer`/`--trainer-id`, `--deaths K`, `--protect`, `--seed`, `--devices`,
and the team from either `--party P.toml` (a complete party file, as `rank` takes) or `--save S`
with the line file's `[[team]]` table. Both run `require_coverage` first and exit 2 on a refused
fight or playout.

    nuzline line odds --game kaizo --save S --trainer-id 246 --line roark.toml --deaths 3 [--n 2000]
    nuzline line find --game kaizo --save S --trainer-id 246 --line team.toml --deaths 1 \
        [--out found.toml] [--budget SECONDS] [--restarts N] [--final-n 2000]

`odds` prints the team, the line, then the ladder: one row per rung 0..3 and one for any win, each
with its 95 % interval, plus the death distribution, mean turns and the fallback share. `find`
prints its progress per sweep to stderr, then the found line as TOML and the same ladder from the
hold-out sample, and writes the line (with the team it was given) to `--out`. With `find`, rules
already in the file are the search's first start.

The line file (TOML):

    [[team]]                 # battle order; the first entry leads. Only with --save.
    pokemon = "Wormadam"     # species, nickname, party:N or box:B:S (the labels of `save show`)
    moves = ["Tackle", "Protect", "Leech Life", "Toxic"]   # optional: replaces the moves, full PP
    item = "Oran Berry"      # optional: replaces the held item ("" for none)

    [[rule]]
    against = "Gible"        # the opponent's species, or slot:N (0-based) when it has two
    use = "Delibird"         # a team member, by the name it was listed under or its species
    move = "Powder Snow"
    first = "Toxic"          # optional
    via = "Rhyhorn"          # optional
    from = "Wormadam"        # optional, only with via
    wait = false             # optional, not with via

Every name is checked when the file is read; an unknown or ambiguous name, a move the Pokemon does
not know, `from` without `via`, `wait` with `via`, or two rules for one foe is a one-line error and
exit 1. A `moves` or `item` override is the user's claim that the TM or item is available: the tool
does not check the bag here (prep does). `src/nuzline/search/line_text.py` reads and writes this
format.

## 7. Done means

- On the pre-Roark save with the user's five Pokemon and items (Prinplup with Quick Claw, Wormadam
  with Toxic, Rhyhorn, Delibird, Sandshrew with Oran Berry), `line odds` on the user's line
  reproduces 72 % wins and 42.5 % with at most one death within sampling error (the 200-playout
  figures carry about ±6 points).
- `line find` on the same team returns a line whose hold-out odds are at least the user's line's on
  both figures, the two measured on the same keys.
- Both are in a committed benchmark (`benchmarks/line_roark.py`) with the numbers and the found line
  recorded in `benchmarks/README.md`, and a slow test that skips without the save pins the
  reproduction loosely.
- The full suite passes on the branch tip.

The save is never copied or committed; tests and the benchmark read its path from
`data/local.toml` (`kaizo_save_pre_roark`) and skip without it.

## 7a. Pluggable players and the scoreboard (added 2026-11-12 by the managing session)

The user's end goal is a player that works lines out itself (exact solve, look-ahead search, later
a learned network), so that lines a human would not find are possible. The line player is the
first step and the baseline. So:

- **One interface for a player** (`search/player.py`): `act(tables, state, params, key, layout) ->
  (action, refused, fallback)` and `replace(tables, state, params, key, layout) -> (slot, refused)`. `playout_policy`
  (`GREEDY`) and the line player (`LINE`, whose params are a `Line`; a user's scripted line and a
  found line are both just params) implement it. The playout loop (`player_playout`) knows only
  the interface, so a search-based or learned player drops in without touching it.
- **A scoreboard** (`python -m benchmarks.players`): every registered player on a fixed list of
  fights and setups (`benchmarks/fights.toml`; first entry Kaizo Roark from the pre-Roark save with
  the user's five), the same seeds for every player, win rate and win rate with at most k deaths
  with intervals. The table for greedy, the user's scripted line and the found line is recorded in
  `benchmarks/README.md`.

No new search method comes with this.

## 8. What L1 leaves, and why

- **`prep` and Stage B still use `playout_policy` at their leaves.** A line belongs to one team, so
  prep would need a line search per setup, and Stage B leaves start mid-fight where a per-foe line
  found from turn 0 is not obviously right. That is design work for the solve-from-here milestone,
  not a small wiring job.
- **The search space is the rule table of §3.** No counted setup moves, no rule keyed on HP or on
  what the opponent just did, no lead choice.
- **The search's power per visit is modest, and it rarely "converges".** In simulation a rule 4
  points better than about 97 others is kept and accepted in one foe visit only 20-45 % of the
  time, depending on how strongly the rules' outcomes share their random draws; repeating the
  visit every sweep is what finds it. A challenger no better than the incumbent is accepted about
  half the time, so a sweep with no change is rare and real runs stop at the time budget or the
  sweep cap. The best line seen is kept, so the drift costs time, not the answer. An accept margin
  or a stop on "no improvement for N sweeps" is the first thing to try if a later fight needs a
  faster search.
- **A player has no memory between turns.** `act` sees the state, its params and a key. A
  recurrent learned player, or a search that reuses its tree from turn to turn, needs the loop to
  carry a player state; that is added when the first such player is built.
- **Level-ups during the fight are not modelled** (K3d). The user's Pokemon gained levels mid-fight
  in the recording; the engine's 72 % does not include that.
