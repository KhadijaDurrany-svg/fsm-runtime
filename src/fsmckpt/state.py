"""Typed workflow state with canonical, stable hashing.

The executor needs two things from a state object:

1. It must be a typed, validated structure (we use a ``pydantic`` v2 model).
2. It must serialize to a *canonical* byte string so that two states that are
   semantically equal always produce the same hash, regardless of dict/set
   insertion order or float formatting quirks.

State hashes are the backbone of the determinism and resume benchmarks: a run
is "replay-deterministic" when the sequence of per-step state hashes is
bit-identical across executions, and a resume is "correct" when its final state
hash matches the uninterrupted reference.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def canonical_bytes(obj: Any) -> bytes:
    """Serialize a plain Python object to canonical JSON bytes.

    Keys are sorted, separators are fixed, and non-ASCII is escaped so the
    encoding is stable across interpreters and locales. ``set``/``frozenset``
    are sorted by their canonical encoding before being emitted as lists, so
    membership-equal sets hash identically regardless of iteration order.
    """

    def _normalize(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return _normalize(value.model_dump(mode="python"))
        if isinstance(value, dict):
            # Sort by key so insertion order never leaks into the hash.
            return {str(k): _normalize(value[k]) for k in sorted(value, key=str)}
        if isinstance(value, (set, frozenset)):
            encoded = sorted(_normalize(v) for v in value)  # sort normalized members for a stable order
            return {"__set__": encoded}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        if isinstance(value, float):
            # Round-trip through repr to avoid platform float-formatting drift,
            # and normalize -0.0 to 0.0 so the two hash identically.
            if value == 0.0:
                return 0.0
            return value
        return value

    normalized = _normalize(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_obj(obj: Any) -> str:
    """Return the hex sha256 of an object's canonical encoding."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


class WorkflowState(BaseModel):
    """Typed state threaded through a finite-state workflow.

    The fields are generic enough to express the synthetic workloads the
    generator produces while still exercising ordered (``trail``), unordered
    (``tallies``) and scalar accumulation, which is where canonicalization
    actually matters.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    step_count: int = Field(default=0, ge=0)
    accumulator: int = 0
    tallies: dict[str, int] = Field(default_factory=dict)
    trail: list[str] = Field(default_factory=list)
    checksum: int = 0
    branch_visits: dict[str, int] = Field(default_factory=dict)

    def state_hash(self) -> str:
        """Stable sha256 of this state's canonical encoding."""
        return hash_obj(self.model_dump(mode="python"))

    def copy_state(self) -> WorkflowState:
        """Return a deep, independent copy (so node functions stay pure)."""
        return self.model_copy(deep=True)
