"""Checkpoint records and an integrity-checked checkpoint store.

A checkpoint captures the workflow position *between* steps: how many steps have
completed, which node runs next, a snapshot of the typed state, and the set of
steps whose non-idempotent writes have already been committed (the exactly-once
dedup marker). Every checkpoint carries a checksum and an explicit commit
marker; a crash during a write can leave the latest record torn (bad checksum or
missing marker), and :meth:`CheckpointStore.latest_valid` transparently falls
back to the last intact record.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .state import WorkflowState, canonical_bytes


@dataclass
class Checkpoint:
    step: int
    next_node: str
    state: WorkflowState
    state_hash: str
    completed_writes: frozenset[int]
    committed: bool = False
    checksum: str = ""

    def compute_checksum(self) -> str:
        payload = canonical_bytes(
            {
                "step": self.step,
                "next_node": self.next_node,
                "state": self.state.model_dump(mode="python"),
                "state_hash": self.state_hash,
                "completed_writes": sorted(self.completed_writes),
            }
        )
        return hashlib.sha256(payload).hexdigest()

    def seal(self) -> Checkpoint:
        """Finalize: set the checksum and the commit marker (an atomic commit)."""
        self.checksum = self.compute_checksum()
        self.committed = True
        return self

    def is_intact(self) -> bool:
        return self.committed and self.checksum == self.compute_checksum()


@dataclass
class CheckpointStore:
    """Append-only store of checkpoints with integrity-aware recovery."""

    records: list[Checkpoint] = field(default_factory=list)

    def save(self, cp: Checkpoint) -> None:
        self.records.append(cp)

    def latest_valid(self) -> Checkpoint | None:
        """Return the highest-step intact checkpoint, or ``None`` if there is none."""
        for cp in reversed(self.records):
            if cp.is_intact():
                return cp
        return None

    def tear_last_write(self) -> None:
        """Simulate a torn final write: drop the latest record's commit marker."""
        if self.records:
            self.records[-1].committed = False

    def corrupt_last_write(self) -> None:
        """Simulate bit-rot: keep the marker but invalidate the checksum."""
        if self.records:
            self.records[-1].checksum = "deadbeef"

    def __len__(self) -> int:
        return len(self.records)
