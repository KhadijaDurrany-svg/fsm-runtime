# fsmckpt

A tiny **typed finite-state graph executor** with **deterministic checkpointing
and replay**. Define a workflow as typed nodes and conditional edges, run it to
completion with a checkpoint after every step, and **resume after a crash with
exactly-once semantics** — then benchmark, honestly, how much that buys you over
naively re-running from scratch.

Everything is offline, seeded and reproducible. The only third-party runtime
dependency is `pydantic` (for the typed state); the checkpoint log, effect
journal, graph executor, synthetic graph generator and statistics are all
implemented here.

## Why this exists / what it measures

A durable executor earns its keep only when a run touches things that cannot be
recomputed from a seed — a nonce, a clock, an external API response, or a
non-idempotent write. `fsmckpt` isolates that nondeterminism behind one
`ExternalSource` and then asks two concrete questions over a corpus of seeded
synthetic graphs:

1. **Replay determinism** — can a recorded run be re-executed to the *same*
   per-step state hashes? Compared: journal-based replay (this runtime) vs a
   naive re-run with no journal.
2. **Resume correctness** — after an injected crash, does the recovered run
   match the uninterrupted reference in **both** final state **and** the
   external write ledger (exactly-once)? Compared: write-ahead checkpointing
   (per-step fsync and group-commit) vs re-run-from-scratch.

## Results

Real numbers from `fsmckpt bench` (400 seeded graphs, base seed `20260101`,
5000 bootstrap resamples, 20000 permutations; Python 3.13, this machine).
Reproduce with `fsmckpt bench`; the dataset checksum is recorded in
`benchmarks/manifest.json`.

### Replay determinism — identical per-step state-hash sequences

378 / 400 graphs actually executed an external read.

| strategy          | correct | rate  | bootstrap 95% CI | Wilson 95% CI    |
|-------------------|---------|-------|------------------|------------------|
| journal replay    | 400/400 | 1.000 | [1.000, 1.000]\* | [0.990, 1.000]   |
| naive replay      |  86/400 | 0.215 | [0.172, 0.255]   | [0.178, 0.258]   |

Paired sign-flip permutation test, journal vs naive: mean difference **0.785**,
**p ≈ 5.0e-05** (at the test's resolution of 1/20001; the samples are
completely separated so the true p is smaller).

Naive replay reproduces the state hashes only when no read ever feeds back into
the state — e.g. a write-only external call (`send → id`) changes the ledger but
not the state. Those 86 graphs are exactly that case; every graph with a
state-affecting read diverges.

### Resume correctness — final state **and** exactly-once write ledger

A crash is injected at a uniformly-random step boundary of each run; the run is
then recovered under each strategy. `redo+` is the mean number of node
re-executions beyond the uninterrupted run (a cost metric, not correctness).

| strategy    | correct | rate  | bootstrap 95% CI | Wilson 95% CI    | redo+ |
|-------------|---------|-------|------------------|------------------|-------|
| wal-fsync   | 400/400 | 1.000 | [1.000, 1.000]\* | [0.990, 1.000]   | 0.0   |
| wal-group4  | 242/400 | 0.605 | [0.557, 0.652]   | [0.556, 0.652]   | 1.4   |
| wal-group8  | 134/400 | 0.335 | [0.287, 0.383]   | [0.291, 0.383]   | 3.3   |
| rerun       |  82/400 | 0.205 | [0.168, 0.242]   | [0.168, 0.247]   | 5.9   |

Headline paired permutation test, **wal-fsync vs rerun**: mean difference
**0.795**, **p ≈ 5.0e-05** (resolution-limited; samples fully separated).

\* **On the all-success rows the bootstrap CI collapses to a single point** —
resampling a vector of all 1s can only produce 1.0. That is why the Wilson score
interval is reported beside it; `[0.990, 1.000]` is the honest uncertainty on a
400/400 observation. `wal-fsync` at 1.000 is a *designed guarantee* (resume from
the exact crash boundary never re-executes completed work), verified across the
corpus and by a Hypothesis property test — not a surprising empirical win. The
informative empirical variance lives in the group-commit and rerun rows.

## How this differs

**From the reference it was built next to**
([`langgraph-fullstack-python`](https://github.com/daniellopez882/langgraph-fullstack-python),
another developer's project): that is a full-stack LLM chatbot — a LangGraph
ReAct agent behind a FastHTML/SSE streaming UI, talking to model providers over
the network. `fsmckpt` shares only the broad "graph of steps" vocabulary. It has
no LLM, no server, no UI, no network and **no dependency on langgraph**; it is a
library about the *execution semantics* — typed state, checkpoints, replay and
crash recovery — that such frameworks assume but rarely benchmark in isolation.

**From an async DAG task scheduler** (a separate project): that work is about
*parallel scheduling* — critical-path ordering of concurrent tasks. `fsmckpt` is
deliberately **sequential and single-threaded**: one node runs at a time, chosen
by deterministic routing, and the entire contribution is the
state-machine + checkpoint/replay/exactly-once layer. There is no concurrency,
no scheduler and no task graph here.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # runtime dep is just pydantic>=2
```

## Use

```bash
fsmckpt bench                # run both experiments, write benchmarks/*.json
fsmckpt bench --no-write     # print the summary only
fsmckpt bench --graphs 200 --seed 7
fsmckpt demo --seed 3        # run one synthetic graph and print its trace
```

As a library:

```python
from fsmckpt import reference_run, resume_run
from fsmckpt.generate import build_graph, generate_spec, initial_state

graph = build_graph(generate_spec(seed=1))
ref = reference_run(graph, initial_state(), seed=1)
out = resume_run(graph, initial_state(), 1, ref, crash_step=2, strategy="wal-fsync")
assert out.correct and out.final_hash == ref.final_hash
```

## Layout

```
src/fsmckpt/
  state.py        typed WorkflowState + canonical, order-insensitive hashing
  graph.py        Node / Edge / Graph, routing, structural validation
  effects.py      ExternalSource, EffectJournal (WAL), SideEffectLedger, context
  checkpoint.py   Checkpoint record + integrity-checked CheckpointStore
  executor.py     forward stepping, reference run, replay, crash + resume
  generate.py     seeded synthetic graph generator + dataset checksum
  stats.py        bootstrap CI, Wilson interval, paired permutation test
  bench.py        benchmark harness (the two experiments)
  manifest.py     git sha / python / dataset-checksum manifest
  cli.py          argparse CLI: `bench`, `demo`
tests/            70 tests (pytest + Hypothesis), incl. replay-determinism property
docs/adr/         3 architecture decision records
```

## How the benchmark stays honest

- Every rate is a fraction of **real** trials; nothing is hand-set. Re-running
  `fsmckpt bench` reproduces the dataset checksum and the integer success counts.
- Nondeterminism is modelled, not faked away: re-executing a step genuinely
  returns a different external read (see `docs/adr/0002`), so the baselines fail
  for a real reason rather than by construction.
- Correctness for resume is judged on **both** state and the external ledger,
  because deterministic recomputation can get the state right while still
  duplicating side effects — that gap is the whole point.
- All-success cells get a Wilson interval, and the bootstrap degeneracy is
  called out rather than hidden behind a fake `[1,1]`.

## Known gaps

- **`wal-fsync` is a guarantee, so its empirical rate is 1.000.** I tried to
  break it — loops, reads that feed state, writes, forward skips, every crash
  boundary in the Hypothesis property test — and it holds, which is the expected
  outcome for a correct resume-from-boundary design, not evidence of a rigged
  benchmark. The bootstrap CI on that row is therefore degenerate (hence Wilson).
- The permutation p-values are **resolution-limited** at 1/20001 ≈ 5.0e-05: the
  distributions are completely separated, so the test can only report "smaller
  than the smallest value I can resolve." Raising `--permutations` lowers the
  floor but will not change the conclusion.
- Exactly-once is achieved by **skipping** completed work on resume, not by a
  dedup-on-replay table. A design that re-executes but suppresses already-emitted
  writes via a durable dedup set would be a strictly different (and interesting)
  comparison; it is not implemented.
- The external-nondeterminism model is a faithful *stand-in* for nonces, clocks
  and API reads, not a proof about any particular real source. Non-serializable
  state is out of scope: nodes must keep state JSON-canonicalizable.
- The generator produces small graphs (5–14 nodes) with bounded loops. Very
  large or deeply cyclic workflows are not exercised.
