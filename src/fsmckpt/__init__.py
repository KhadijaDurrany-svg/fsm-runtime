"""fsmckpt -- a tiny typed finite-state graph executor with deterministic
checkpointing and replay.

Define a workflow as typed nodes and conditional edges, run it to completion
with a checkpoint after every step, and resume after a crash with exactly-once
semantics. The public API mirrors the three things the benchmark measures:
building graphs, running them, and recovering them.
"""

from __future__ import annotations

from .checkpoint import Checkpoint, CheckpointStore
from .effects import EffectJournal, ExecutionContext, ExternalSource, SideEffectLedger
from .executor import (
    ResumeOutcome,
    RunTrace,
    determinism_trial,
    reference_run,
    replay_from_journal,
    replay_naive,
    resume_run,
)
from .graph import END, START, Edge, Graph, GraphError, Node
from .state import WorkflowState, canonical_bytes, hash_obj

__version__ = "0.1.0"

__all__ = [
    "END",
    "START",
    "Checkpoint",
    "CheckpointStore",
    "Edge",
    "EffectJournal",
    "ExecutionContext",
    "ExternalSource",
    "Graph",
    "GraphError",
    "Node",
    "ResumeOutcome",
    "RunTrace",
    "SideEffectLedger",
    "WorkflowState",
    "__version__",
    "canonical_bytes",
    "determinism_trial",
    "hash_obj",
    "reference_run",
    "replay_from_journal",
    "replay_naive",
    "resume_run",
]
