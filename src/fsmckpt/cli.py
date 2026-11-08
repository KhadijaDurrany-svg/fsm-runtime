"""Command-line interface for fsmckpt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .bench import run_benchmark
from .executor import reference_run
from .generate import build_graph, generate_spec, initial_state
from .manifest import build_manifest


def _cmd_bench(args: argparse.Namespace) -> int:
    results = run_benchmark(
        base_seed=args.seed,
        n_graphs=args.graphs,
        bootstrap_resamples=args.resamples,
        permutations=args.permutations,
    )
    manifest = build_manifest(results["dataset"]["checksum"], results["config"])

    _print_summary(results, manifest)

    if args.no_write:
        print("\n[--no-write] results not persisted.")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'manifest.json'}")
    return 0


def _fmt(interval: dict[str, float]) -> str:
    return f"{interval['point']:.3f} [{interval['low']:.3f}, {interval['high']:.3f}]"


def _print_summary(results: dict[str, Any], manifest: dict[str, Any]) -> None:
    print("=== fsmckpt benchmark ===")
    print(f"python {manifest['python']} | git {manifest['git_sha']}")
    print(f"dataset: {results['dataset']['n_graphs']} graphs | checksum {results['dataset']['checksum'][:16]}")

    rd = results["replay_determinism"]
    logged = rd["logged_replay"]
    naive = rd["naive_replay"]
    print("\n-- replay determinism (identical per-step state hashes) --")
    print(f"  logged replay : {_fmt(logged['bootstrap_ci'])}  (wilson {_fmt(logged['wilson_ci'])})")
    print(f"  naive  replay : {_fmt(naive['bootstrap_ci'])}  (wilson {_fmt(naive['wilson_ci'])})")
    print(f"  permutation p : {rd['permutation_logged_vs_naive']['p_value']:.2e}")

    rc = results["resume_correctness"]["per_strategy"]
    print("\n-- resume correctness (final state AND exactly-once ledger) --")
    for strat, block in rc.items():
        print(
            f"  {strat:<11}: {_fmt(block['bootstrap_ci'])}  "
            f"(wilson {_fmt(block['wilson_ci'])})  "
            f"redo+={block['mean_extra_node_executions']:.1f}"
        )
    head = results["resume_correctness"]["headline"]["permutation"]
    print(f"  headline {results['resume_correctness']['headline']['comparison']}: p = {head['p_value']:.2e}")


def _cmd_demo(args: argparse.Namespace) -> int:
    spec = generate_spec(args.seed)
    graph = build_graph(spec)
    trace = reference_run(graph, initial_state(), args.seed)
    print(f"graph {spec.name}: {len(spec.nodes)} nodes, run length {trace.length}")
    print(f"final state hash: {trace.final_hash}")
    print(f"ledger entries  : {len(trace.ledger.entries)}")
    print("state-hash trajectory (first 6):")
    for h in trace.hash_sequence[:6]:
        print(f"  {h[:16]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fsmckpt", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bench", help="run the determinism + resume benchmark")
    b.add_argument("--seed", type=int, default=20260101)
    b.add_argument("--graphs", type=int, default=400, help="number of synthetic graphs")
    b.add_argument("--resamples", type=int, default=5000, help="bootstrap resamples")
    b.add_argument("--permutations", type=int, default=20000)
    b.add_argument("--out", default="benchmarks", help="output directory")
    b.add_argument("--no-write", action="store_true", help="print only, do not persist")
    b.set_defaults(func=_cmd_bench)

    d = sub.add_parser("demo", help="run one synthetic graph and print its trace")
    d.add_argument("--seed", type=int, default=1)
    d.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
