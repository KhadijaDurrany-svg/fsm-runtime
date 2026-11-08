# ADR 0002: Modeling nondeterminism offline and reproducibly

## Status
Accepted.

## Context
A durable executor only earns its keep when execution touches things that
cannot be recomputed from a seed: a nonce, a clock, an external API response, or
a non-idempotent write. But the benchmark must be fully offline and
reproducible -- no real clock, no real network, identical results on every
machine.

## Decision
All nondeterminism is funneled through `ExternalSource`. A read of a logical key
`(step, node, kind)` returns a value derived from `sha256(seed, step, node,
kind, attempt)`, where `attempt` is the number of times that exact key has
*already been read on this source*. The first physical execution of a step sees
`attempt=0`; re-executing the same step sees `attempt=1`, and so on.

This single rule captures the thing that matters: **re-running a step yields a
different external value**, exactly as a real nonce or clock would. Two
independently-constructed sources with the same seed agree on every *first*
read (the shared recorded history) and diverge only on re-execution.

Two mechanisms recover from that nondeterminism:
- **replay** restores past reads from a durable journal (`EffectJournal`), so a
  recorded run can be re-derived bit-for-bit;
- **resume** restores a checkpoint and skips the completed prefix entirely, so
  those steps are never re-read.

## Consequences
- The benchmark is deterministic given a master seed, yet the baselines
  genuinely fail, because re-execution genuinely changes external reads.
- Reads that feed back into state break *state* determinism under naive replay;
  reads that only feed a write (e.g. "send, get id") do not change the state
  hash but still corrupt the write ledger. Both failure modes are measured, in
  the replay and the resume experiments respectively.
- This is a *model*, not a proof about any specific real clock or RNG; the
  "Known gaps" section of the README says so plainly.
