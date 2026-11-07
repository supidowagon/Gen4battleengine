"""Exhaustive check of the prep funnel on Roark from the user's save (plan 4c Task 12, spec 8).

    uv run python -m benchmarks.exhaustive_roark --save "$NUZLINE_USER_SAVE" [--resume FILE]
                                                 [--devices N] [--out FILE]
                                                 [--recheck VALUES [--seeds 0,1,2,3,4]]

The pool is the save's party plus the first Venonat (box 1 slot 1) and Geodude (box 1 slot 4)
(ruling 14); the level cap is the configured one, no bag items, full HP. The valued setups are
``restricted_setups`` plus ``pruned_setups`` (the prune run with ``relearner="all"``). For each
rung ``k`` in (0, 1): the funnel's pick (``relearner="designated"``, no TMs, items, entry HP or
free level-up replacements (``levelled=False``: the valued space has none), ``max_depth=1``; the
problem's Heart Scales are pinned to at most one); phase 1 values every setup by Stage B depth 1
at ``budget_playouts=2000``; phase 2 re-solves at precision 0.015 the setups whose phase-1
interval overlaps the best's, the pick, and every pruned setup whose phase-1 interval reaches
above the pick's; ``verdict`` on phase 2. The run checkpoints after every setup (``--resume``
continues across nights; the
checkpoint is tied to the problem, settings, tables and code like a prep checkpoint) and writes
``benchmarks/exhaustive_roark.json`` (``--out``). A failed verdict is a finding, not an error:
the exit code is 0 either way.

``--recheck VALUES`` values nothing else: it runs the funnel fresh for each rung and each seed
of ``--seeds`` (``PRNGKey(seed)``) and judges each pick with ``recheck_verdict`` against the
stored values of an earlier exhaustive checkpoint (read as raw JSON, so a checkpoint of an older
version or fingerprint still serves; it must hold the same setup count). The pick's value is its
stored phase-2 value, else its phase-1 one, else (``"solved"``) a fresh solve at precision 0.015.
It writes ``benchmarks/exhaustive_roark_recheck.json`` (``--out``), never
``exhaustive_roark.json``; a failed verdict is again a finding, with exit code 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE / "exhaustive_roark.json"
RECHECK = HERE / "exhaustive_roark_recheck.json"
RESULTS = HERE / "results"
BOXED = (("Venonat", 0, 1), ("Geodude", 0, 4))  # (species, 0-based box, slot): ruling 14
PHASE1_PLAYOUTS = 2000
PHASE2_PRECISION = 0.015
ETA_AFTER = 20  # setups


def _value_json(v) -> list[float]:
    return [v.estimate, v.ci_low, v.ci_high, v.e_deaths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save", type=Path, default=os.environ.get("NUZLINE_USER_SAVE"))
    parser.add_argument("--resume", type=Path, default=None, help="continue from this checkpoint")
    parser.add_argument("--devices", type=int, default=os.cpu_count())
    parser.add_argument("--out", type=Path, default=None, help=f"default {RESULT.name}")
    parser.add_argument(
        "--recheck",
        type=Path,
        default=None,
        metavar="VALUES",
        help="judge fresh funnel picks against this exhaustive checkpoint's stored values;"
        f" values nothing else (default --out {RECHECK.name})",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="--recheck: the funnel seeds")
    args = parser.parse_args(argv)
    if args.out is None:
        args.out = RESULT if args.recheck is None else RECHECK
    try:
        args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        parser.error(f"--seeds must be comma-separated integers, not {args.seeds!r}")
    if not args.save:
        parser.error("--save (or NUZLINE_USER_SAVE) is required")
    args.save = Path(args.save)

    from nuzline.cli import INTERMEDIATE, configure_cpu_devices, resolve_trainer

    configure_cpu_devices(args.devices)  # before jax is imported

    import jax

    from benchmarks.rank_timing import machine
    from nuzline import jaxcache
    from nuzline.data import tables as tables_module
    from nuzline.data.games import VANILLA
    from nuzline.data.learnsets import load_learnsets
    from nuzline.data.tables import load_names, load_tables
    from nuzline.data.trainer import IntermediateData, build_trainer_party, load_trainers
    from nuzline.engine.coverage import start_weather
    from nuzline.prep.caps import default_cap
    from nuzline.prep.checkpoint import (
        CHECKPOINT_VERSION,
        fingerprint,
        read_checkpoint,
        write_checkpoint,
    )
    from nuzline.prep.exhaustive import (
        Value,
        exhaustive_values,
        pruned_setups,
        restricted_setups,
        verdict,
    )
    from nuzline.prep.pool import pool_candidates
    from nuzline.prep.problem import build_problem
    from nuzline.prep.search import PrepSettings, run_prep, setup_label
    from nuzline.saves import load_save

    table_version = jaxcache.table_version(tables_module.DEFAULT)
    jaxcache.enable_compile_cache(table_version + jaxcache.code_version(), VANILLA)
    tables, names = load_tables(), load_names()
    trainer = resolve_trainer(load_trainers(INTERMEDIATE), name="roark", trainer_id=None)
    cap = default_cap(trainer)
    save = load_save(args.save, tables=tables, names=names)
    learnsets = load_learnsets()
    pool, _ = pool_candidates(save, tables, names, learnsets, cap)
    keep = {(box, slot) for _, box, slot in BOXED}
    exclude = [
        f"box:{c.location[1] + 1}:{c.location[2]}"
        for c in pool
        if c.location[0] == "box" and c.location[1:] not in keep
    ]
    opp = build_trainer_party(trainer, IntermediateData(INTERMEDIATE))
    base, warnings = build_problem(
        tables,
        names,
        save,
        trainer,
        opp,
        start_weather(trainer),
        learnsets,
        cap=cap,
        exclude=exclude,
    )
    for line in warnings:
        print(line)
    # The spec's space allows one relearner change and the bag holds one Heart Scale.
    base = replace(
        base,
        resources=replace(base.resources, heart_scales=min(1, base.resources.heart_scales)),
    )
    boxed = [c for c in base.candidates if c.location[0] == "box"]
    for species, box, slot in BOXED:
        if not any(
            c.location[1:] == (box, slot)
            and names.species_names[c.spec.species].casefold() == species.casefold()
            for c in boxed
        ):
            raise SystemExit(f"box {box + 1} slot {slot} is not a usable {species}: wrong save?")
    pool_labels = [c.label for c in base.candidates]
    print(f"pool ({len(pool_labels)}): " + "; ".join(pool_labels))

    restricted = restricted_setups(tables, base)
    extra, pairs = pruned_setups(tables, base)
    setups = restricted + extra
    pruned_ids = set(range(len(restricted), len(setups)))
    print(f"setups: {len(restricted)} restricted + {len(extra)} pruned = {len(setups)}")
    note = None
    if not pairs:
        note = (
            "prune dropped nothing on this save: the pruned-setup criterion is vacuous here"
            " (the check rests on test_pruned_setups_are_valued)"
        )
        print(note)

    settings = PrepSettings(
        max_depth=1,
        relearner="designated",
        tms=False,
        items=False,
        entry_hp=False,
        levelled=False,
        max_team=6,
    )
    if args.recheck is not None:
        return _recheck(args, tables, names, base, setups, pruned_ids, settings)
    print_ = fingerprint(base, settings, 0, table_version, jaxcache.code_version())
    funnel: dict = {"setups": len(setups), "rungs": {}}
    sink = args.resume or RESULTS / f"exhaustive_roark_{datetime.now():%Y%m%d_%H%M%S}.ckpt.json"
    if args.resume is not None:
        funnel = read_checkpoint(args.resume, print_)["funnel"]
        if funnel["setups"] != len(setups):
            raise SystemExit(f"checkpoint {args.resume} was written for another setup count")
    print(f"checkpoint: {sink}")

    def save_state() -> None:
        write_checkpoint(
            sink, {"version": CHECKPOINT_VERSION, "fingerprint": print_, "funnel": funnel}
        )

    started = time.perf_counter()
    rungs_out = {}
    for k in (0, 1):
        prob = replace(base, deaths=k)
        rung = funnel["rungs"].setdefault(
            str(k), {"pick": None, "phase1": {}, "phase2": {}, "seconds": {}}
        )

        def add_seconds(phase: str, t0: float, rung=rung) -> None:
            rung["seconds"][phase] = rung["seconds"].get(phase, 0.0) + time.perf_counter() - t0

        if rung["pick"] is None:
            t0 = time.perf_counter()
            result = run_prep(tables, names, prob, settings, jax.random.PRNGKey(0))
            pick = result.chosen
            if pick not in setups:
                raise SystemExit(f"rung {k}: the funnel's pick is outside the valued space")
            rung["pick"] = setups.index(pick)
            add_seconds("funnel", t0)
            save_state()
        pick = rung["pick"]
        print(f"rung {k}: funnel pick #{pick}: {setup_label(names, prob, setups[pick])}")

        def run_phase(name: str, indices, key, rung=rung, k=k, prob=prob, **kw) -> dict[int, Value]:
            subset = [setups[i] for i in indices]
            done = {
                j: Value(*rung[name][str(i)]) for j, i in enumerate(indices) if str(i) in rung[name]
            }
            t_phase = time.perf_counter()
            count = [0]
            t_first = [time.perf_counter()]

            def on_value(j: int, v: Value) -> None:
                rung[name][str(indices[j])] = _value_json(v)
                save_state()
                count[0] += 1
                if count[0] == ETA_AFTER:
                    per = (time.perf_counter() - t_first[0]) / ETA_AFTER
                    left = len(indices) - len(done) - ETA_AFTER
                    print(
                        f"rung {k} {name}: {per:.1f} s per setup, "
                        f"{left} left, ETA {per * left / 3600:.1f} h"
                    )
                elif count[0] % 100 == 0:
                    print(f"rung {k} {name}: {len(rung[name])}/{len(indices)} setups")

            values = exhaustive_values(
                tables,
                names,
                prob,
                subset,
                key,
                depth=1,
                done=done,
                on_value=on_value,
                **kw,
            )
            add_seconds(name, t_phase)
            save_state()
            return {indices[j]: v for j, v in values.items()}

        phase1 = run_phase(
            "phase1",
            range(len(setups)),
            jax.random.fold_in(jax.random.PRNGKey(1), k),
            budget_playouts=PHASE1_PLAYOUTS,
        )
        best1 = min(
            (i for i in phase1 if i not in pruned_ids), key=lambda i: (-phase1[i].estimate, i)
        )
        overlapping = {
            i
            for i, v in phase1.items()
            if i not in pruned_ids
            and v.ci_high >= phase1[best1].ci_low
            and v.ci_low <= phase1[best1].ci_high
        }
        above = {i for i in pruned_ids if phase1[i].ci_high > phase1[pick].ci_high}
        chosen = sorted(overlapping | {pick} | above)
        print(f"rung {k}: phase 2 on {len(chosen)} setups ({len(above)} pruned)")
        phase2 = run_phase(
            "phase2",
            chosen,
            jax.random.fold_in(jax.random.PRNGKey(2), k),
            precision=PHASE2_PRECISION,
        )
        local = {g: j for j, g in enumerate(chosen)}
        out = verdict(
            {local[g]: v for g, v in phase2.items()},
            local[pick],
            [local[g] for g in chosen if g in pruned_ids],
        )
        back = chosen.__getitem__

        def labelled(i: int, prob=prob, phase2=phase2) -> dict:
            return {
                "index": i,
                "setup": setup_label(names, prob, setups[i]),
                "value": _value_json(phase2[i]),
            }

        rungs_out[str(k)] = {
            "best": labelled(back(out.best)),
            "pick": labelled(pick),
            "pick_ties_best": out.pick_ties_best,
            "passed": out.passed,
            "pruned_beating_pick": [labelled(back(j)) for j in out.beaten_by_pruned],
            "phase_seconds": rung["seconds"],
        }
        print(
            f"rung {k}: {'PASSED' if out.passed else 'FAILED'}: best {phase2[back(out.best)]}, "
            f"pick {phase2[pick]}, pruned beating the pick: {len(out.beaten_by_pruned)}"
        )

    doc = {
        "save_sha256": hashlib.sha256(args.save.read_bytes()).hexdigest(),
        "pool": pool_labels,
        "setups": len(setups),
        "restricted_setups": len(restricted),
        "pruned_setups": len(extra),
        "pruned_note": note,
        "value_fields": ["estimate", "ci_low", "ci_high", "e_deaths"],
        "rungs": rungs_out,
        "passed": all(r["passed"] for r in rungs_out.values()),
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "machine": machine(),
        "devices": jax.device_count(),
    }
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {'PASSED' if doc['passed'] else 'FAILED'}")
    return 0


def _plain(obj):
    """A dataclass (or nested tuples of one) as JSON-ready plain values."""
    if dataclasses.is_dataclass(obj):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_plain(v) for v in obj]
    return obj


def _recheck(args, tables, names, base, setups, pruned_ids, settings) -> int:
    """``--recheck``: fresh funnel picks for rungs 0 and 1 and each seed, judged against the
    stored values of ``args.recheck``; writes ``args.out`` and returns 0."""
    import jax

    from benchmarks.rank_timing import machine
    from nuzline.prep.exhaustive import (
        Value,
        exhaustive_values,
        recheck_verdict,
        stored_value,
    )
    from nuzline.prep.search import run_prep, setup_label

    # Raw JSON: the stored file may be an older checkpoint version with an older fingerprint.
    doc = json.loads(args.recheck.read_text(encoding="utf-8"))
    if doc["funnel"]["setups"] != len(setups):
        raise SystemExit(
            f"{args.recheck} holds {doc['funnel']['setups']} setups, this save {len(setups)}"
        )
    runs = []
    for k in (0, 1):
        prob = replace(base, deaths=k)
        rung = doc["funnel"]["rungs"][str(k)]
        for seed in args.seeds:
            t0 = time.perf_counter()
            result = run_prep(tables, names, prob, settings, jax.random.PRNGKey(seed))
            seconds = time.perf_counter() - t0
            if result.chosen not in setups:
                raise SystemExit(f"rung {k} seed {seed}: the pick is outside the valued space")
            pick = setups.index(result.chosen)
            found = stored_value(rung, pick)
            if found is None:
                key = jax.random.fold_in(jax.random.PRNGKey(3), k)
                solved = exhaustive_values(
                    tables, names, prob, [setups[pick]], key, depth=1, precision=PHASE2_PRECISION
                )
                found = ("solved", solved[0])
            src, value = found
            v = recheck_verdict(rung, pick, value, pruned_ids)
            best_value = Value(*rung["phase2"][str(v.best)])
            print(
                f"rung {k} seed {seed}: {'PASS' if v.passed else 'FAIL'} pick #{pick} ({src})"
                f" {value} vs best #{v.best} {best_value}, funnel {seconds:.1f} s,"
                f" {result.stats.playouts} playouts"
            )
            finalists = []
            for f in result.finalists:
                top = f.result.ranking[0] if f.result.ranking else None
                index = setups.index(f.setup) if f.setup in setups else None
                stored = stored_value(rung, index) if index is not None else None
                finalists.append(
                    {
                        "index": index,
                        "estimate": top.estimate if top else None,
                        "ci_low": top.ci_low if top else None,
                        "ci_high": top.ci_high if top else None,
                        "stored": [stored[0], _value_json(stored[1])] if stored else None,
                    }
                )
            runs.append(
                {
                    "k": k,
                    "seed": seed,
                    "pick": pick,
                    "setup": setup_label(names, prob, setups[pick]),
                    "source": src,
                    "value": _value_json(value),
                    "best": v.best,
                    "best_value": _value_json(best_value),
                    "passed": v.passed,
                    "funnel_seconds": seconds,
                    "playouts": result.stats.playouts,
                    "finalists": finalists,
                }
            )
    out = {
        "values_file": str(args.recheck),
        "runs": runs,
        "passed": all(r["passed"] for r in runs),
        "machine": machine(),
        "devices": jax.device_count(),
        "settings": _plain(settings),
    }
    args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    passed = sum(r["passed"] for r in runs)
    print(f"wrote {args.out}: {passed}/{len(runs)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
