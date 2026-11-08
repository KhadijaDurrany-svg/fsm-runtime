"""Nondeterministic external effects, their journal, and the exactly-once ledger.

This module isolates the *only* source of nondeterminism in the system. Node
transition logic is a pure function of ``(state, recorded_effects)``; everything
that cannot be recomputed from a seed lives here.

Two kinds of effect exist:

* **reads** -- values a node pulls from the outside world (a nonce, a clock, an
  external API response). A fresh physical execution of the same node returns a
  *different* value; we model that with an execution-attempt-dependent hash.
  Reads are *journaled*: the first execution records them, and any later replay
  reads them back from the journal rather than re-drawing.
* **writes** -- non-idempotent actions on the outside world (append to a ledger,
  enqueue a message). Replaying a recorded read makes recomputation
  deterministic, but it does **not** undo a write: re-executing a completed node
  applies its write a second time. Avoiding that duplicate is exactly what
  checkpoint-based resume buys over re-running from scratch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

_MASK64 = (1 << 64) - 1


def _mix(*parts: object) -> int:
    """Deterministic 64-bit mix of its arguments via sha256."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class ExternalSource:
    """A stand-in for genuinely non-reproducible external reads.

    ``run_seed`` identifies the logical workflow run. Each ``(step, node, kind)``
    carries its own *physical execution count*: the first time it is read the
    value reflects attempt 0, the second time attempt 1, and so on. Re-executing
    a step therefore yields a *different* value -- faithfully modelling a nonce,
    clock or external API read that cannot be reproduced -- which is exactly why
    a naive re-run cannot reproduce the original's reads and why a durable
    journal (replay) or checkpoint skip (resume) is needed.

    ``attempt`` is an additional base salt; two independently-constructed
    sources with the same seed and base produce identical *first* reads for a
    key (the shared recorded history), diverging only on re-execution.
    """

    run_seed: int
    attempt: int = 0
    _counts: dict[tuple[int, str, str], int] = field(default_factory=dict)

    def read(self, step: int, node: str, kind: str) -> int:
        key = (step, node, kind)
        n = self._counts.get(key, 0)
        self._counts[key] = n + 1
        return _mix("read", self.run_seed, step, node, kind, self.attempt + n) & _MASK64


@dataclass
class EffectJournal:
    """Write-ahead log of recorded external reads, keyed by ``(step, node, kind)``.

    ``durable`` holds entries guaranteed to survive a crash. ``buffered`` holds
    entries written but not yet flushed; a crash discards them. ``fsync`` moves
    buffered entries into ``durable``. This models the real durability/perf
    tension: flushing per effect is safe but slow, batching is fast but risks
    losing the un-flushed tail.
    """

    durable: dict[str, int] = field(default_factory=dict)
    buffered: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def key(step: int, node: str, kind: str) -> str:
        return f"{step}:{node}:{kind}"

    def record(self, step: int, node: str, kind: str, value: int) -> None:
        self.buffered[self.key(step, node, kind)] = value

    def fsync(self) -> None:
        self.durable.update(self.buffered)
        self.buffered.clear()

    def lookup(self, step: int, node: str, kind: str) -> int | None:
        k = self.key(step, node, kind)
        if k in self.buffered:
            return self.buffered[k]
        return self.durable.get(k)

    def durable_snapshot(self) -> dict[str, int]:
        return dict(self.durable)

    def crash(self) -> None:
        """Simulate a crash: everything not yet fsync'd is lost."""
        self.buffered.clear()

    @classmethod
    def from_durable(cls, durable: dict[str, int]) -> EffectJournal:
        return cls(durable=dict(durable), buffered={})


@dataclass
class SideEffectLedger:
    """The external world's record of non-idempotent write actions.

    Each entry is ``(step, node, payload)``. Exactly-once means this ledger,
    after any crash/recovery, equals the ledger of an uninterrupted run.
    """

    entries: list[tuple[int, str, int]] = field(default_factory=list)

    def emit(self, step: int, node: str, payload: int) -> None:
        self.entries.append((step, node, payload))

    def signature(self) -> tuple[tuple[int, str, int], ...]:
        return tuple(self.entries)

    def clone(self) -> SideEffectLedger:
        return SideEffectLedger(entries=list(self.entries))


class ExecutionContext:
    """Handed to node functions; the only channel to the outside world.

    A read is served from the journal if present (replay), otherwise drawn from
    the external source and journaled (first execution). Writes go to both the
    ledger and are gated by the caller's exactly-once dedup set.
    """

    def __init__(
        self,
        step: int,
        node: str,
        source: ExternalSource,
        journal: EffectJournal,
        ledger: SideEffectLedger,
    ) -> None:
        self.step = step
        self.node = node
        self._source = source
        self._journal = journal
        self._ledger = ledger
        self._read_count = 0

    def read_int(self, kind: str = "r") -> int:
        """Return an external read, replayed from the journal when available."""
        tag = f"{kind}{self._read_count}"
        self._read_count += 1
        cached = self._journal.lookup(self.step, self.node, tag)
        if cached is not None:
            return cached
        value = self._source.read(self.step, self.node, tag)
        self._journal.record(self.step, self.node, tag, value)
        return value

    def emit(self, payload: int) -> None:
        """Apply a non-idempotent write to the external ledger."""
        self._ledger.emit(self.step, self.node, payload)
