from __future__ import annotations

import pytest

from fsmckpt.effects import ExecutionContext
from fsmckpt.graph import END, Edge, Graph, GraphError, Node
from fsmckpt.state import WorkflowState


def _noop(state: WorkflowState, ctx: ExecutionContext) -> None:
    state.accumulator += 1


def _linear(n: int) -> Graph:
    nodes = {}
    for i in range(n):
        target = f"n{i + 1}" if i + 1 < n else END
        nodes[f"n{i}"] = Node(f"n{i}", _noop, [Edge(target=target)])
    return Graph(nodes=nodes, start="n0")


def test_linear_graph_validates():
    g = _linear(3)
    assert g.start == "n0"
    assert len(g.nodes) == 3


def test_missing_start_rejected():
    with pytest.raises(GraphError):
        Graph(nodes={"n0": Node("n0", _noop, [Edge(END)])}, start="missing")


def test_node_without_edges_rejected():
    with pytest.raises(GraphError):
        Graph(nodes={"n0": Node("n0", _noop, [])}, start="n0")


def test_node_without_default_edge_rejected():
    bad = Node("n0", _noop, [Edge(END, predicate=lambda s: False)])
    with pytest.raises(GraphError):
        Graph(nodes={"n0": bad}, start="n0")


def test_default_edge_must_be_last():
    bad = Node(
        "n0",
        _noop,
        [Edge(END), Edge(END, predicate=lambda s: True)],
    )
    with pytest.raises(GraphError):
        Graph(nodes={"n0": bad}, start="n0")


def test_unreachable_node_rejected():
    nodes = {
        "n0": Node("n0", _noop, [Edge(END)]),
        "orphan": Node("orphan", _noop, [Edge(END)]),
    }
    with pytest.raises(GraphError):
        Graph(nodes=nodes, start="n0")


def test_end_not_reachable_rejected():
    # n0 -> n1 -> n0 loop, END never reachable.
    nodes = {
        "n0": Node("n0", _noop, [Edge("n1")]),
        "n1": Node("n1", _noop, [Edge("n0")]),
    }
    with pytest.raises(GraphError):
        Graph(nodes=nodes, start="n0")


def test_end_as_node_name_rejected():
    with pytest.raises(GraphError):
        Graph(nodes={END: Node(END, _noop, [Edge(END)])}, start=END)


def test_pointing_to_undefined_node_rejected():
    with pytest.raises(GraphError):
        Graph(nodes={"n0": Node("n0", _noop, [Edge("ghost")])}, start="n0")


def test_next_node_routing_picks_first_match():
    node = Node(
        "n0",
        _noop,
        [
            Edge("a", predicate=lambda s: s.accumulator > 10),
            Edge("b", predicate=lambda s: s.accumulator > 0),
            Edge("c"),
        ],
    )
    assert node.next_node(WorkflowState(accumulator=20)) == "a"
    assert node.next_node(WorkflowState(accumulator=5)) == "b"
    assert node.next_node(WorkflowState(accumulator=0)) == "c"
