"""Typed finite-state graph: nodes, conditional edges, and validation.

A :class:`Graph` is a set of named nodes plus a ``START`` entry node. Each node
applies a transition to the (typed) state via an :class:`ExecutionContext`, then
routing picks the next node from the node's ordered edge list. Reaching ``END``
stops the run. Routing is a pure, deterministic function of the state, so the
only nondeterminism in a run comes from :mod:`fsmckpt.effects`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .effects import ExecutionContext
from .state import WorkflowState

START = "__start__"
END = "__end__"

Transition = Callable[[WorkflowState, ExecutionContext], None]
Predicate = Callable[[WorkflowState], bool]


@dataclass
class Edge:
    """A conditional routing edge. ``predicate=None`` is the default (fallback)."""

    target: str
    predicate: Predicate | None = None

    def matches(self, state: WorkflowState) -> bool:
        return self.predicate is None or self.predicate(state)


@dataclass
class Node:
    """A named state transition with ordered outgoing edges."""

    name: str
    transition: Transition
    edges: list[Edge] = field(default_factory=list)

    def next_node(self, state: WorkflowState) -> str:
        """Deterministically select the next node name given the state."""
        for edge in self.edges:
            if edge.matches(state):
                return edge.target
        raise GraphError(f"node {self.name!r} has no matching edge and no default")


class GraphError(ValueError):
    """Raised when a graph is structurally invalid."""


@dataclass
class Graph:
    """A validated finite-state graph."""

    nodes: dict[str, Node]
    start: str
    name: str = "graph"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.start not in self.nodes:
            raise GraphError(f"start node {self.start!r} is not defined")
        if END in self.nodes:
            raise GraphError("END is reserved and must not be a defined node")
        for node in self.nodes.values():
            if not node.edges:
                raise GraphError(f"node {node.name!r} has no outgoing edges")
            # Exactly one unconditional (default) edge, and it must be last.
            defaults = [i for i, e in enumerate(node.edges) if e.predicate is None]
            if len(defaults) != 1:
                raise GraphError(f"node {node.name!r} must have exactly one default edge")
            if defaults[0] != len(node.edges) - 1:
                raise GraphError(f"node {node.name!r} default edge must be listed last")
            for edge in node.edges:
                if edge.target != END and edge.target not in self.nodes:
                    raise GraphError(f"node {node.name!r} points to undefined node {edge.target!r}")
        self._check_reachability()

    def _check_reachability(self) -> None:
        seen: set[str] = set()
        stack = [self.start]
        reaches_end = False
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for edge in self.nodes[cur].edges:
                if edge.target == END:
                    reaches_end = True
                else:
                    stack.append(edge.target)
        unreachable = set(self.nodes) - seen
        if unreachable:
            raise GraphError(f"unreachable nodes: {sorted(unreachable)}")
        if not reaches_end:
            raise GraphError("END is not reachable from start")
