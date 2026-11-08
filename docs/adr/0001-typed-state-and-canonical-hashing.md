# ADR 0001: Typed state with canonical hashing

## Status
Accepted.

## Context
The whole project rests on being able to say "two runs produced the same state."
If that comparison is flaky -- because a dict rehashed, a set iterated in a
different order, or a float formatted differently -- then every determinism and
resume claim downstream is meaningless.

## Decision
State is a `pydantic` v2 model (`WorkflowState`) with `extra="forbid"` so typos
in node code fail loudly. Equality is defined by a **canonical encoding**:
`canonical_bytes` normalizes recursively (sorted dict keys, sets emitted as a
sorted tagged list, `-0.0` folded to `0.0`) and serializes to compact,
ASCII-escaped JSON; the state hash is the sha256 of those bytes.

We hash the canonical *content*, not pydantic's `model_dump_json`, because we
want stability across pydantic minor versions and across unordered containers.

## Consequences
- Two semantically-equal states always hash identically; ordered fields
  (`trail`) still distinguish order, which is intentional.
- Nodes must keep state JSON-serializable. That is a mild constraint but keeps
  checkpoints trivially durable.
- A dedicated property test asserts dict-order invariance so a future refactor
  cannot silently reintroduce order sensitivity.
