"""The finite-state executor: forward stepping, replay, and crash recovery.

The executor walks a :class:`~fsmckpt.graph.Graph` one node at a time, threading
a typed state and writing a checkpoint after every step. On top of the single
forward pass it provides three capabilities the benchmark exercises:

* **reference run** -- an uninterrupted execution recording the per-step state
  hashes, the read journal, and the external side-effect ledger.
* **replay** -- re-execute a recorded run, either from its journal (faithful) or
  without it (naive), used to measure replay determinism.
* **resume** -- recover from an injected crash under a chosen durability
  strategy, used to measure exactly-once resume correctness.

Nodes are executed via an :class:`~fsmckpt.effects.ExecutionContext`; the only
nondeterminism is the external reads that context mediates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checkpoint import Checkpoint
from .effects import EffectJournal, ExecutionContext, ExternalSource, SideEffectLedger
from .graph import END, Graph
from .state import WorkflowState

# Hard ceiling so a malformed cyclic graph can never hang the executor.
MAX_STEPS = 10_000


class ExecutionError(RuntimeError):
    pass


@dataclass
class StepRecord:
    step: int
    node: str
    state_hash: str
    did_read: bool
    did_write: bool


@dataclass
class RunTrace:
    """Everything observable about one execution."""

    init_hash: str
    steps: list[StepRecord]
    final_hash: str
    ledger: SideEffectLedger
    journal: EffectJournal
    checkpoints: list[Checkpoint]
    node_executions: int

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def hash_sequence(self) -> list[str]:
        return [self.init_hash] + [s.state_hash for s in self.steps]

    @property
    def ledger_signature(self) -> tuple[tuple[int, str, int], ...]:
        return self.ledger.signature()


class _TrackingContext(ExecutionContext):
    """Execution context that notes whether a step touched the outside world."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.read_happened = False
        self.write_happened = False

    def read_int(self, kind: str = "r") -> int:
        self.read_happened = True
        return super().read_int(kind)

    def emit(self, payload: int) -> None:
        self.write_happened = True
        super().emit(payload)


def _run_forward(
    graph: Graph,
    state: WorkflowState,
    source: ExternalSource,
    journal: EffectJournal,
    ledger: SideEffectLedger,
    *,
    start_step: int,
    start_node: str,
    stop_after: int | None = None,
) -> RunTrace:
    """Execute from ``start_node``/``start_step`` until END (or ``stop_after``).

    ``stop_after`` is a boundary step index: when the step with that index
    finishes, execution halts (used to model a crash at a step boundary). The
    returned trace's ``steps`` only covers the steps executed in *this* pass.
    """
    current = start_node
    step = start_step
    steps: list[StepRecord] = []
    checkpoints: list[Checkpoint] = []
    completed_writes: set[int] = set()
    node_executions = 0
    init_hash = state.state_hash()

    while current != END:
        if step >= MAX_STEPS:
            raise ExecutionError("step budget exceeded; graph is likely non-terminating")
        node = graph.nodes[current]
        ctx = _TrackingContext(step, current, source, journal, ledger)
        node.transition(state, ctx)
        state.step_count = step + 1
        node_executions += 1
        if ctx.write_happened:
            completed_writes.add(step)
        next_node = node.next_node(state)
        sh = state.state_hash()
        steps.append(
            StepRecord(
                step=step,
                node=current,
                state_hash=sh,
                did_read=ctx.read_happened,
                did_write=ctx.write_happened,
            )
        )
        cp = Checkpoint(
            step=step,
            next_node=next_node,
            state=state.copy_state(),
            state_hash=sh,
            completed_writes=frozenset(completed_writes),
        ).seal()
        checkpoints.append(cp)
        if stop_after is not None and step == stop_after:
            # A crash at this boundary: stop before advancing.
            return RunTrace(init_hash, steps, sh, ledger, journal, checkpoints, node_executions)
        current = next_node
        step += 1

    final_hash = state.state_hash()
    return RunTrace(init_hash, steps, final_hash, ledger, journal, checkpoints, node_executions)


def reference_run(graph: Graph, init: WorkflowState, seed: int) -> RunTrace:
    """Run uninterrupted, journaling every external read (the golden trace)."""
    source = ExternalSource(run_seed=seed, attempt=0)
    journal = EffectJournal()
    ledger = SideEffectLedger()
    state = init.copy_state()
    trace = _run_forward(graph, state, source, journal, ledger, start_step=0, start_node=graph.start)
    journal.fsync()
    return trace


def replay_from_journal(graph: Graph, init: WorkflowState, seed: int, journal_durable: dict[str, int]) -> RunTrace:
    """Faithful replay: every read is served from the recorded journal."""
    source = ExternalSource(run_seed=seed, attempt=99)  # must never actually be used
    journal = EffectJournal.from_durable(journal_durable)
    ledger = SideEffectLedger()
    state = init.copy_state()
    return _run_forward(graph, state, source, journal, ledger, start_step=0, start_node=graph.start)


def replay_naive(graph: Graph, init: WorkflowState, source: ExternalSource) -> RunTrace:
    """Naive replay: no journal, reads are re-drawn from ``source``.

    ``source`` is shared with the reference run, so its per-key execution counts
    have advanced -- a re-drawn read is a *fresh* physical read and will differ
    from the reference whenever a read occurred.
    """
    journal = EffectJournal()
    ledger = SideEffectLedger()
    state = init.copy_state()
    return _run_forward(graph, state, source, journal, ledger, start_step=0, start_node=graph.start)


# --- resume strategies -------------------------------------------------------

STRATEGIES = ("wal-fsync", "wal-group4", "wal-group8", "rerun")


def _resume_boundary(strategy: str, crash_step: int) -> int:
    """Index of the checkpoint to resume from (``-1`` means from the start)."""
    if strategy == "rerun":
        return -1
    if strategy == "wal-fsync":
        return crash_step
    if strategy.startswith("wal-group"):
        g = int(strategy.removeprefix("wal-group"))
        # Checkpoints are flushed every ``g`` steps; the un-flushed tail is lost.
        flushed = ((crash_step + 1) // g) * g - 1
        return flushed
    raise ValueError(f"unknown strategy {strategy!r}")


@dataclass
class ResumeOutcome:
    strategy: str
    crash_step: int
    final_hash: str
    ledger_signature: tuple[tuple[int, str, int], ...]
    node_executions: int
    correct: bool
    extra_node_executions: int = field(default=0)


def resume_run(
    graph: Graph,
    init: WorkflowState,
    seed: int,
    reference: RunTrace,
    crash_step: int,
    strategy: str,
) -> ResumeOutcome:
    """Execute a crash-at-``crash_step`` scenario and recover with ``strategy``.

    The physical reads of this scenario come from a *fresh* source with the same
    seed, so the first execution of any step reproduces the reference read; only
    steps that are physically re-executed draw a new (differing) value.
    """
    source = ExternalSource(run_seed=seed, attempt=0)
    journal = EffectJournal()
    ledger = SideEffectLedger()
    state = init.copy_state()

    # Partial run up to and including the crash boundary.
    partial = _run_forward(
        graph,
        state,
        source,
        journal,
        ledger,
        start_step=0,
        start_node=graph.start,
        stop_after=crash_step,
    )
    node_execs = partial.node_executions

    resume_from = _resume_boundary(strategy, crash_step)
    if resume_from < 0:
        resume_state = init.copy_state()
        resume_step = 0
        resume_node = graph.start
    else:
        cp = partial.checkpoints[resume_from]
        resume_state = cp.state.copy_state()
        resume_step = cp.step + 1
        resume_node = cp.next_node

    # Recovery pass continues on the SAME source/ledger (so re-executed steps are
    # physically fresh reads, and prior writes remain in the ledger).
    if resume_node == END:
        recovery = RunTrace(
            resume_state.state_hash(),
            [],
            resume_state.state_hash(),
            ledger,
            journal,
            [],
            0,
        )
    else:
        recovery = _run_forward(
            graph,
            resume_state,
            source,
            journal,
            ledger,
            start_step=resume_step,
            start_node=resume_node,
        )
    node_execs += recovery.node_executions

    final_hash = recovery.final_hash
    correct = final_hash == reference.final_hash and ledger.signature() == reference.ledger_signature
    return ResumeOutcome(
        strategy=strategy,
        crash_step=crash_step,
        final_hash=final_hash,
        ledger_signature=ledger.signature(),
        node_executions=node_execs,
        correct=correct,
        extra_node_executions=node_execs - reference.node_executions,
    )


@dataclass
class DeterminismOutcome:
    logged_ok: bool
    naive_ok: bool
    had_read: bool


def determinism_trial(graph: Graph, init: WorkflowState, seed: int) -> DeterminismOutcome:
    """Compare journal-based replay vs naive re-run against a reference run.

    The reference and the naive replay share one external source, so the naive
    replay's reads are the *next* physical draw for each key and will differ
    whenever a read occurred -- exactly the condition under which unlogged replay
    loses determinism.
    """
    src = ExternalSource(run_seed=seed, attempt=0)
    journal = EffectJournal()
    ledger = SideEffectLedger()
    ref = _run_forward(
        graph,
        init.copy_state(),
        src,
        journal,
        ledger,
        start_step=0,
        start_node=graph.start,
    )
    journal.fsync()
    had_read = any(s.did_read for s in ref.steps)

    logged = replay_from_journal(graph, init, seed, journal.durable_snapshot())
    logged_ok = logged.hash_sequence == ref.hash_sequence

    naive = replay_naive(graph, init, src)  # shares src -> fresh (attempt >=1) reads
    naive_ok = naive.hash_sequence == ref.hash_sequence
    return DeterminismOutcome(logged_ok=logged_ok, naive_ok=naive_ok, had_read=had_read)
