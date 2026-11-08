"""Benchmark harness: run real trials and report measured rates with CIs.

Two experiments run over the same seeded dataset of synthetic graphs:

1. **Replay determinism** -- does re-executing a recorded run reproduce its
   per-step state-hash sequence? Compared: journal-based replay (this runtime)
   vs naive re-run without a journal (baseline).
2. **Resume correctness** -- after an injected crash, does the recovered run
   match the uninterrupted reference in *both* final state and the external
   write ledger (exactly-once)? Compared: write-ahead checkpointing with
   per-step fsync and with group-commit, vs re-run-from-scratch (baseline).

No numbers are invented: every rate is the fraction of real trials that passed.
"""

from __future__ import annotations

import random
from typing import Any

from .executor import STRATEGIES, determinism_trial, reference_run, resume_run
from .generate import build_graph, dataset_checksum, generate_dataset, initial_state
from .stats import bootstrap_ci, permutation_test_paired, proportion, wilson_ci


def _rate_block(outcomes: list[int], *, boot_seed: int) -> dict[str, Any]:
    n = len(outcomes)
    succ = sum(outcomes)
    return {
        "n": n,
        "successes": succ,
        "rate": proportion(succ, n),
        "bootstrap_ci": bootstrap_ci(outcomes, seed=boot_seed).as_dict(),
        "wilson_ci": wilson_ci(succ, n).as_dict(),
    }


def run_benchmark(
    *,
    base_seed: int = 20260101,
    n_graphs: int = 400,
    bootstrap_resamples: int = 5000,
    permutations: int = 20000,
) -> dict[str, Any]:
    specs = generate_dataset(base_seed, n_graphs)
    checksum = dataset_checksum(specs)
    crash_rng = random.Random(base_seed ^ 0x5DEECE66D)

    det_logged: list[int] = []
    det_naive: list[int] = []
    det_had_read: list[int] = []

    resume_outcomes: dict[str, list[int]] = {s: [] for s in STRATEGIES}
    resume_extra: dict[str, list[int]] = {s: [] for s in STRATEGIES}

    for i, spec in enumerate(specs):
        graph = build_graph(spec)
        init = initial_state()
        run_seed = base_seed * 31 + i

        det = determinism_trial(graph, init, run_seed)
        det_logged.append(int(det.logged_ok))
        det_naive.append(int(det.naive_ok))
        det_had_read.append(int(det.had_read))

        reference = reference_run(graph, init, run_seed)
        n_steps = reference.length
        crash_step = crash_rng.randrange(0, n_steps)
        for strat in STRATEGIES:
            out = resume_run(graph, init, run_seed, reference, crash_step, strat)
            resume_outcomes[strat].append(int(out.correct))
            resume_extra[strat].append(out.extra_node_executions)

    # --- assemble report ---
    replay = {
        "logged_replay": _rate_block(det_logged, boot_seed=1),
        "naive_replay": _rate_block(det_naive, boot_seed=2),
        "graphs_with_external_read": sum(det_had_read),
        "permutation_logged_vs_naive": permutation_test_paired(det_logged, det_naive, permutations=permutations),
    }

    per_strategy: dict[str, Any] = {}
    for idx, strat in enumerate(STRATEGIES):
        block = _rate_block(resume_outcomes[strat], boot_seed=10 + idx)
        extras = resume_extra[strat]
        block["mean_extra_node_executions"] = sum(extras) / len(extras) if extras else 0.0
        per_strategy[strat] = block

    headline = permutation_test_paired(
        resume_outcomes["wal-fsync"], resume_outcomes["rerun"], permutations=permutations
    )

    return {
        "config": {
            "base_seed": base_seed,
            "n_graphs": n_graphs,
            "bootstrap_resamples": bootstrap_resamples,
            "permutations": permutations,
            "strategies": list(STRATEGIES),
        },
        "dataset": {"n_graphs": n_graphs, "checksum": checksum},
        "replay_determinism": replay,
        "resume_correctness": {
            "per_strategy": per_strategy,
            "headline": {
                "comparison": "wal-fsync vs rerun",
                "permutation": headline,
            },
        },
        "notes": [
            "Rates are fractions of real trials; no value is hand-set.",
            "Bootstrap CIs collapse to a point for all-success/all-failure cells; "
            "the Wilson interval is reported alongside for those.",
            "'extra_node_executions' is work redone beyond the uninterrupted run "
            "(a cost metric, not a correctness metric).",
        ],
    }
