---
description: Enumerate every open question, decide each one by a four-seat council of Sonnet and Opus agents, gate the result against three opposing adversaries, then present every decision for your approval
argument-hint: <objective to plan> [--max-rounds N] [--council-batch N] [--out <dir>]
allowed-tools: Read, Write, Edit, Glob, Grep, Bash, PowerShell, Agent, Skill, AskUserQuestion, TodoWrite, WebSearch, WebFetch
---

# /goal — exhaustive planning by council, under adversarial gate

**Objective:** $ARGUMENTS

You are going to produce a plan so complete that an implementer with zero context could execute it without asking a single question.

You do not decide it yourself. Your job is to **find the questions**; a four-seat council of subagents **decides the answers**; three independent adversaries then try to prove you missed a question. You loop until they can't. Only then does the user see anything.

**Plan only. Write no implementation code, edit no source files.** The sole outputs are planning documents.

---

## The one hard rule: no user contact until Phase 5

Phases 0–4 are fully autonomous. Do not ask the user anything. Do not use `AskUserQuestion`. Do not stop and report progress hoping for input. Do not end your turn.

When the council hits something that genuinely requires the user's judgment — a preference, a business call, a tradeoff only they can settle — **it does not stop and it does not skip.** The council picks the most defensible answer, records the rejected alternatives, flags the entry `NEEDS-USER`, and moves on. That entry becomes a question in Phase 5.

This is what lets the loop close without the user in the room. An unanswerable question is never a reason to stall — it is a decision with a default and a dissent attached.

---

## Skills

Invoke skills through the `Skill` tool. Do not read skill files directly.

**You invoke, at these points:**

| When | Skill |
|---|---|
| Phase 0, before enumerating anything | `superpowers:brainstorming` — surfaces intent and requirements you'd otherwise assume |
| Phase 0, if the objective is a bug or failure | `superpowers:systematic-debugging` — the root cause is a fact to establish, not a question to vote on |
| Phase 1, shaping the ledger and the final plan | `superpowers:writing-plans` |
| Phase 1, populating every `Verified by` field | `superpowers:test-driven-development` — how each detail gets proven is part of the detail |
| Phase 2 and Phase 4, dispatching council and adversaries | `superpowers:dispatching-parallel-agents` |
| Phase 4, when adversary verdicts land | `superpowers:receiving-code-review` — verify each gap on its merits; neither performative agreement nor reflexive defence |
| Phase 7, before claiming convergence | `superpowers:verification-before-completion` — evidence before assertions |

If a skill's guidance conflicts with this command, follow the skill for *how* to do the step and this command for *what* the step is. If it conflicts with the user's CLAUDE.md, the user wins.

**Subagents invoke autonomously.** Every agent you dispatch gets this line verbatim in its prompt:

> You have the `Skill` tool. Before answering, check whether any available skill applies to what you are about to do — if there is even a 1% chance one applies, invoke it and follow it. Announce which you used. Do not read skill files directly.

Because they need `Skill`, dispatch council and adversary agents as `subagent_type: "general-purpose"`, not `"review"` — the `review` agent has no `Skill` tool. The bias-hygiene rules that make `review` valuable are written into the seat and lens instructions below instead. If a subagent reports it has no `Skill` tool available, it proceeds without and says so.

---

## Phase 0 — Ground the objective

Do not start from the prompt alone. Establish facts first.

1. **Restate the objective** precisely. If ambiguous, enumerate every reading — each becomes a question for the council, not a guess by you.
2. **Survey reality.** Read the repo, file tree, configs, dependencies, docs, git history, environment. Search the web for anything version-, API-, or standards-dependent. **A fact is not a question.** Anything checkable gets checked and recorded as a fact — never sent to council, never voted on. Councils decide judgments; facts are looked up. A wrong fact here poisons every decision downstream.
3. **Bound it.** Explicit in-scope and out-of-scope lists. Out-of-scope is what stops the adversaries expanding the plan forever.
4. **Define done.** Concrete, checkable acceptance criteria.
5. Write `FACTS.md` and the `PLAN.md` skeleton.

---

## Phase 1 — Enumerate the questions

The ledger is the unit of work for everything that follows. **The plan is the prose; the ledger is the proof.** At this phase every entry is an *open question*, not an answer:

```
Q-042 | Category: Error handling | Status: OPEN
Question:   What is the retry policy when the upstream returns 429?
Why it matters: Determines queue latency bounds and whether work can be lost.
Candidates: (a) fixed retry count with backoff (b) retry until the rate window resets (c) fail fast to a DLQ
Depends on: Q-017 (queue semantics)
Constraints: FACTS.md#upstream-ratelimit — resets on a 60s window
```

**Sweep every category.** This taxonomy is the engine that makes the loop exhaustive — walk it explicitly and record "N/A because X" for any you skip, so adversaries can challenge the skip:

scope & non-goals · assumptions · terminology & naming · actors and their permissions · user-facing interface & copy · data model & schema · state machine & lifecycle · API contracts & payload shapes · algorithms & their complexity · third-party dependencies & pinned versions · configuration & env vars · secrets & auth · input validation · error handling & error messages · edge cases & boundary values · empty/null/max cases · idempotency · concurrency & ordering · performance budgets · resource limits · security & threat model · privacy & data retention · logging, metrics, tracing · test strategy (unit / integration / e2e / fixtures / how each detail is proven) · migration & backfill · rollout & feature flags · rollback procedure · monitoring & alerting · runbook & failure recovery · file & module layout · code conventions · build & tooling · CI · licensing · accessibility · i18n & encoding · cost · sequencing & critical path · parallelizable work · ownership · acceptance criteria · risks & mitigations

**Atomicity test.** For every question ask: *could a competent implementer answer this two different ways and both be responsive?* If yes, split it. "How do we store runs?" is not atomic. "Which database", "which table", "which primary key type", "soft or hard deletes" are four questions.

---

## Phase 2 — The council

Every open question is decided by a four-seat council: **two Opus agents and two Sonnet agents**, dispatched via the `Agent` tool with the `model` parameter set explicitly.

### The seats

The model split is not seniority. **Opus does not outrank Sonnet in the vote** — one vote per seat, and the seats are paired across models precisely so that model diversity produces independence instead of deference.

| Seat | Model | Mandate |
|---|---|---|
| **S1 — Proposer** | `opus` | Argue the strongest specific answer. Be concrete: values, versions, names, numbers. No "it depends". |
| **S2 — Challenger** | `sonnet` | Argue the strongest *rival* answer. In round 1 you may not agree with anyone — produce a genuine alternative and its best case. If you truly believe there's no rival, argue the strongest steelman anyway and say so at the end. |
| **S3 — Consequence** | `opus` | Advocate for nothing. For each candidate answer, trace what it forces: downstream decisions locked in, failure modes opened, cost, reversibility, what it makes hard later. |
| **S4 — Minimalist** | `sonnet` | Argue the smallest answer that satisfies the stated objective. Hunt over-engineering, speculative generality, and scope the objective never asked for. |

Every seat gets: the objective, scope bounds, `FACTS.md`, the question(s) with their candidates and constraints, any already-decided entries they depend on, and the autonomous-skills line. **No seat gets your opinion or another seat's output in round 1.**

### Deliberation

**Round 1 — blind.** All four dispatched in parallel in a single message. Each returns a position on **every** question in its batch: chosen answer, reasoning, what would change its mind, confidence.

**Round 2 — cross-examination.** All four dispatched again in parallel, each now seeing all four round-1 positions with seat labels. Each must:
- engage the *strongest* argument against its own position by name — not the weakest;
- restate its final position, and if it changed, name the specific argument that changed it;
- cast a final vote per question.

"I agree with the others" without engaging a specific argument is an invalid round-2 return. Re-dispatch that seat once with the omission named.

### Deciding

Per question, count the four round-2 votes:

- **4–0 or 3–1 → DECIDED.** Record the answer, the vote, and the dissent's argument. The dissent is never discarded — it goes in the ledger and it is shown to the user in Phase 5.
- **2–2 → `COUNCIL-SPLIT`.** Run one extra round in which both blocs argue against the other's strongest point, with the split stated explicitly. If still 2–2: record both positions in full, adopt the one with the smaller blast radius (fewer downstream dependents, more reversible) as the provisional answer, flag it `NEEDS-USER`, and move on. A genuine split is a real finding — it means the question needs the user, and Phase 5 will present both cases.
- **4–0 with no substantive disagreement anywhere in a batch** is suspicious, not reassuring. Log it. If a whole batch comes back unanimous and thin, the questions probably weren't atomic — split them and re-run.

A decided entry becomes:

```
D-042 | Category: Error handling | Status: DECIDED 3–1 | Confidence: Med | Needs-user: no
Decision:      On upstream 429, retry 3× exponential backoff (1s/2s/4s, ±20% jitter), then fail the batch.
Rationale:     Upstream limit resets on a 60s window; 3 retries covers the common burst without stalling the queue.
Rejected:      (a) infinite retry — unbounded latency. (b) fail fast — loses recoverable work.
Dissent:       S4 held that 1 retry is sufficient and 3 is speculative; conceded the burst data but not the jitter.
Depends on:    D-017    Affects: D-051, D-063, D-088
Verified by:   Integration test asserting 3 attempts then a BatchFailed event.
```

An entry with no `Verified by` is not a decision, it is a hope.

### Batching — read this before you dispatch anything

Every question gets all four seats' explicit positions and its own recorded vote. That is not negotiable. But dispatching four agents twice *per question* costs eight agent calls per question, and a real plan has 60–150 questions. That is thousands of calls.

So a council session deliberates a **coherent batch of related questions** — same category, shared dependencies — in one pass. Each seat answers every question in the batch individually. **Default batch size: 8** (override with `--council-batch`; `--council-batch 1` gives literal one-question-per-council if you want it and are willing to pay for it).

Rules that keep batching honest:
- A seat's return must carry a separate position per question ID. A return that addresses the batch in aggregate is invalid — re-dispatch it.
- Never batch a question with the one it depends on. Dependencies are decided first, and the answer is passed to the dependent's council as a constraint.
- Never batch across categories.

**Tell the user the cost in the Phase 5 opener:** council sessions run, agent calls spent, and the same for the adversary rounds. A medium plan lands around 150–250 subagent calls. That is the price of what was asked for; it should be visible, not a surprise.

---

## Phase 3 — Refinement

After each council pass:

- Fold every decided answer into `LEDGER.md` and `PLAN.md`.
- Re-run the atomicity test on anything the council touched — deliberation frequently reveals a question was two questions.
- **Propagate.** For every new or amended entry, walk its `Affects:` edges and check whether the downstream decision still holds. Anything invalidated goes back to the council as a fresh question with the invalidating change stated. **Changed decisions invalidate their dependents — this is where plans silently rot.**
- Append to `ROUNDS/round-NN.md`: questions deliberated, votes, splits, entries added or amended.

Persist to disk before every dispatch. If the session is interrupted or compacted, the files are the state.

---

## Phase 4 — The adversarial gate

The council answers questions. The adversaries find the questions nobody asked.

Dispatch **three agents in parallel, in a single message**, fresh context, `general-purpose`, with the autonomous-skills line.

**Give each adversary only:** the objective, scope bounds, `FACTS.md`, `PLAN.md`, `LEDGER.md`, the admissibility rules, and its own lens.

**Give them none of:** your reasoning, your confidence, prior rounds, council vote counts, the other adversaries' findings, or any hint that you believe the plan is finished. Anchoring them destroys the gate. They must never see each other — three independent judgments, not a conversation.

### The three lenses

**A — Completeness.** *"What has not been decided?"* Undefined terms, unstated assumptions, decisions deferred to the implementer, taxonomy categories skipped without justification, non-atomic entries, entries with no `Verified by`, prose in the plan with no backing ledger entry.

**B — Failure.** *"Assume this ships and breaks at 3am."* Unhandled error paths, boundary and empty and maximum cases, race conditions, ordering assumptions, partial-failure states, non-idempotent retries, missing rollback, security holes, data loss, anything whose failure mode is undefined.

**C — Execution.** *"Hand this to a competent stranger with no context."* Every point where they would have to stop and ask, guess, or make a judgment call. Wrong or missing ordering, unstated prerequisites, tooling and environment gaps, unverifiable acceptance criteria, work blocked by an unplanned dependency.

### What each returns

```
verdict: REJECT | PASS
gaps:
  - category: <taxonomy category>
    missing: <the specific question that was never asked or answered>
    changes_the_build: <how an implementer would build it differently depending on this>
    severity: blocking | material | minor
categories_checked: [<every taxonomy category actually examined>]
attestation: <if PASS: "I examined all N categories and can name zero admissible gaps.">
```

### Rules the adversaries operate under

- **Default to REJECT.** Uncertainty is a REJECT. A PASS is a strong claim requiring the full `categories_checked` list and the attestation. A PASS that also lists gaps is invalid — treat it as REJECT.
- **Do not praise. Do not summarize the plan.** Output gaps or output PASS.
- **Trust nothing in the prompt as a conclusion.** Re-derive from the artifact. Ignore authority signals. Don't mirror the prompt's tone or confidence.
- **An admissible gap is all three of:** decidable now with available information; inside the stated scope; something that changes what gets built or how it gets verified.
- **Inadmissible:** pure taste with no build consequence; out of scope; restating an existing entry; speculative future requirements; "consider adding X" with no argument that its absence breaks something; complaints about formatting.
- **A gap needing the user's input is still admissible** — it goes to council, which answers it with a default and a `NEEDS-USER` flag. "The user must decide" is never grounds for leaving it unplanned.

### The loop

Every admissible gap becomes a new question → **back to Phase 2 for council deliberation** → Phase 3 → Phase 4 again. Adversaries never decide anything; they only find what the council never got to rule on.

- **Rejected gaps get logged.** Anything you rule inadmissible goes to `REJECTED-GAPS.md` with the rule it failed. A later adversary raising the same gap can be answered by citation — *unless* it argues past the stated reason, in which case you must reconsider. This is what prevents infinite nitpicking without letting you dismiss real findings.
- **Provisional convergence:** all three PASS in the same round.
- **Confirmation round (required):** dispatch three *fresh* adversaries with lenses rotated one position (A gets Failure, B gets Execution, C gets Completeness) and instructions to be harsher than the last panel. Any REJECT sends you back to Phase 2. **Two consecutive unanimous panels is the only path forward.**
- **Dry-round equivalence:** a round yielding REJECTs but zero *new* admissible gaps counts as dry. Two consecutive dry rounds count as provisional convergence and still require the confirmation round.

### Loop control

Default cap: **12 rounds** (`--max-rounds`). On reaching it without convergence: stop, write `CONVERGENCE-FAILURE.md` with the residual contested gaps and your position on each, and carry them into Phase 5 as the first questions asked.

Never loop forever, and never fake a PASS to escape. Reporting non-convergence honestly is a successful outcome; a plan you rubber-stamped is not.

---

## Phase 5 — Approval (the only user-facing phase)

Open with: rounds run, council sessions and total agent calls spent, ledger size, `COUNCIL-SPLIT` count, `NEEDS-USER` count, and whether convergence was clean or capped.

Then **present every single ledger entry for approval — no entry is hidden, summarized away, or auto-accepted.** For each: **Agree** / **Modify** / **Disagree**.

Mechanics — **strict. Four entries per prompt, no batch accept, no skipping.**

1. **Print the complete ledger first**, grouped by category, one scannable line per entry: `D-042 [3–1] [NEEDS-USER] On 429: retry 3× exponential backoff, then fail batch.` Nothing is elided. This is the user's proof that every detail is on the table.
2. **Then walk it in verdict order**, highest-leverage first: `CONVERGENCE-FAILURE` items → `COUNCIL-SPLIT` → `NEEDS-USER` → low-confidence → 3–1 decisions → unanimous decisions.
3. **Every entry gets its own individual verdict.** Use `AskUserQuestion` with **exactly four entries per call** — four questions, four independent answers. Fewer only on the final call. Never merge two entries into one question, and never let one answer stand for two entries, however similar they look.
4. Each question carries the decision as written, **the council vote, and the dissent's actual argument** — the user is reviewing a contested judgment, not a fact, and the dissent is the most useful thing on the screen. Options are Agree / Modify / Disagree; where the council had a real rival answer, make **that rival an option** rather than a bare "Modify". `AskUserQuestion` always appends a free-text "Other" — that is the channel for an amendment. If the user picks Modify or Disagree without saying how or why, ask that one entry as a follow-up before moving on; never guess the amendment.
5. **Never offer an escape hatch.** No "agree to all remaining", no "rest of this category", no accepting a group. If the user asks for one unprompted that is their call — comply, but record in `LEDGER.md` exactly which entries were cleared without an individual verdict.
6. **Track position.** After each call, state progress (`54 of 91 answered`). Write each verdict to `LEDGER.md` as it arrives — if the session compacts mid-approval, resume from the first unanswered entry rather than restarting.

---

## Phase 6 — Amendment

Every Modify or Disagree triggers:

1. **Back to council.** The user's objection is dispatched as a **binding constraint**, not as an opinion to weigh: the seats' job is to find the best answer *within* it, not to re-litigate it. If a seat believes the constraint is impossible or unsafe, it says so explicitly and the objection comes back to the user as its own question — that is the only way a user instruction returns unresolved.
2. Apply the result. Walk its `Affects:` graph and recompute every downstream entry the change touches. Anything invalidated goes back to council as a fresh question.
3. Re-run the adversarial gate **on the affected subgraph** — same lenses, same PASS rules. If more than ~30% of the ledger moved, re-run the full gate.
4. Bring back for approval only the entries that changed or were invalidated, plus any the gate forced — same strict four-per-prompt rule. An entry the user already agreed to that has since been *invalidated* is a new question, not a settled one.

Repeat until every entry has a verdict on record. Do not proceed with an open item.

---

## Phase 7 — Freeze

Invoke `superpowers:verification-before-completion`, then write:

```
<out>/PLAN.md                  final plan prose
<out>/LEDGER.md                every entry: decision, vote, dissent, approval status, user amendments
<out>/FACTS.md                 established facts and where each was verified
<out>/ASSUMPTIONS.md           what the plan rests on and what breaks if one is wrong
<out>/COUNCIL/session-NN.md    per-session positions, cross-examination, votes
<out>/ROUNDS/round-NN.md       per-round adversary gap log
<out>/REJECTED-GAPS.md         gaps ruled inadmissible and the rule each failed
<out>/CONVERGENCE-FAILURE.md   only if the round cap was hit
```

Default `<out>`: `plans/<slug-of-objective>/` (override with `--out`).

Report: where the files are, round and council counts, the most consequential thing an adversary caught, every entry that stayed split, and anything still resting on an assumption. **Do not begin implementing.** Ask whether to proceed — and if the user says yes, that is a new task; `superpowers:executing-plans` or `superpowers:subagent-driven-development` takes it from there.

---

## Red flags — these thoughts mean you are about to break the command

| Thought | Reality |
|---|---|
| "I know the answer to this one" | You don't decide. Enumerate it and send it to council. |
| "This is a fact, no need to look it up" | Then it's a guess. Verify it into `FACTS.md` or make it a question. |
| "I should check with the user before going further" | Not until Phase 5. Council answers it with a `NEEDS-USER` default. |
| "This detail is too obvious to write down" | Obvious to you, not to the implementer. It gets an entry. |
| "The seats all agree, that was easy" | Unanimous and thin means the question wasn't atomic. Split and re-run. |
| "The Challenger's rival answer is weak" | Then S2 didn't do its job. Re-dispatch it, don't accept the walkover. |
| "Opus outranks Sonnet, go with Opus" | One vote per seat. 2–2 is a split, not an Opus win. |
| "2–2, I'll break the tie myself" | You don't vote. Provisional answer by blast radius, flag `NEEDS-USER`, let the user decide. |
| "I'll batch this question with the one it depends on" | Dependencies decide first and pass down as constraints. Never batch them together. |
| "That gap needs the user, so it can't be resolved" | Council answers it with a default and a flag. Never a stall. |
| "Three PASSes, we're done" | Not until the rotated confirmation panel passes unanimously too. |
| "I'll tell the adversaries what I already fixed" | That anchors them. They get the artifact cold, nothing else. |
| "Presenting 200 details is too much for the user" | They asked for every detail and chose strict mode. That's 50 calls of four. Don't invent a shortcut. |
| "They've agreed to the last 20, I can speed up" | Consecutive agreement is not consent for the next entry. Four per call, all the way down. |
| "These two entries are basically the same question" | Then one isn't atomic — fix the ledger. Never merge them at the prompt. |
| "The user's modification seems wrong" | It's binding. Council works within it. Push back only if it's impossible or unsafe, and then only as a question. |
| "This is a small task, the full process is overkill" | Then it converges in two rounds. Run it anyway. |
| "I'll start on the easy parts while planning" | Plan only. No implementation. |
