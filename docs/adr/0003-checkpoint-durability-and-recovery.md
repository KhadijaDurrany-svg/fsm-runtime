# ADR 0003: Checkpoint durability and recovery strategies

## Status
Accepted.

## Context
"Resume after failure" means different things depending on how durable the
checkpoint log is. We wanted the benchmark to expose that spectrum honestly
rather than claim a single number.

## Decision
A checkpoint carries the state snapshot, its hash, the set of steps whose writes
have committed, a checksum, and an explicit commit marker. `is_intact()`
requires both the marker and a matching checksum, and `latest_valid()` walks
backward to the newest intact record -- so a torn or bit-rotted final write is
transparently skipped.

We benchmark three recovery strategies against the re-run baseline:
- **wal-fsync** -- a checkpoint is durably flushed after every step, so recovery
  resumes from the exact crash boundary and the completed prefix is never
  re-executed. This is a *correctness guarantee*, and the benchmark verifies it
  holds across the whole dataset and a Hypothesis property test.
- **wal-group{4,8}** -- checkpoints are flushed in batches (group commit). A
  crash loses the un-flushed tail, forcing re-execution of up to `G-1` steps;
  those re-executed steps draw fresh external reads and re-apply writes, so a
  fraction of resumes are incorrect. This is the realistic durability/throughput
  trade-off.
- **rerun** (baseline) -- no checkpoints; recovery restarts from scratch,
  re-reading and re-writing the entire completed prefix.

Correctness is judged on *both* the final state hash and the external write
ledger (exactly-once), because deterministic recomputation can get the state
right while still duplicating side effects.

## Consequences
- The headline comparison (`wal-fsync` vs `rerun`) is a guarantee vs a baseline;
  the interesting empirical variance lives in the group-commit rows, which have
  non-degenerate confidence intervals.
- Because `wal-fsync` is all-success on the dataset, its bootstrap CI collapses
  to a point; we report the Wilson interval alongside and flag the degeneracy.
- Exactly-once here is achieved by *skipping* completed work on resume, not by a
  dedup table; a dedup-on-replay design is noted as future work.
