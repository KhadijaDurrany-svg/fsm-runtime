"""Seeded synthetic graph generator.

Graphs are built around a forward spine (``n0 -> n1 -> ... -> END``) so every
graph is guaranteed to terminate and to reach END. On top of the spine we add
two kinds of conditional edge:

* **skip** edges jump forward (never backward), and
* **loop** edges jump backward but are gated by a per-node visit cap stored in
  the state, so the number of iterations is bounded.

Each node is assigned an *op* that determines how it mutates the typed state and
whether it performs an external read, a non-idempotent write, or both. The op
mix controls how often the naive baselines fail, which keeps the measured rates
away from the degenerate 0/1 ends.

Everything is derived from an integer seed, and :func:`dataset_checksum` hashes
the full spec list so a run is reproducible and verifiable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .graph import END, Edge, Graph, Node
from .state import WorkflowState, hash_obj

OPS = ("incr", "read_add", "tally", "emit", "read_emit", "noop")


@dataclass
class BranchSpec:
    kind: str  # "skip" | "loop"
    mod: int
    rem: int
    target: str
    cap: int = 0


@dataclass
class NodeSpec:
    name: str
    op: str
    const: int
    tally_key: str
    branches: list[BranchSpec] = field(default_factory=list)
    default: str = END


@dataclass
class GraphSpec:
    name: str
    nodes: list[NodeSpec]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "nodes": [
                {
                    "name": n.name,
                    "op": n.op,
                    "const": n.const,
                    "tally_key": n.tally_key,
                    "default": n.default,
                    "branches": [
                        {"kind": b.kind, "mod": b.mod, "rem": b.rem, "target": b.target, "cap": b.cap}
                        for b in n.branches
                    ],
                }
                for n in self.nodes
            ],
        }


def _make_transition(spec: NodeSpec):
    op = spec.op
    const = spec.const
    key = spec.tally_key
    name = spec.name

    def transition(state: WorkflowState, ctx) -> None:  # type: ignore[no-untyped-def]
        state.branch_visits[name] = state.branch_visits.get(name, 0) + 1
        state.trail.append(name)
        if op == "incr":
            state.accumulator += const
        elif op == "noop":
            state.accumulator += 0
        elif op == "tally":
            state.tallies[key] = state.tallies.get(key, 0) + 1
        elif op == "read_add":
            r = ctx.read_int("a")
            state.accumulator += r % 1000
        elif op == "emit":
            ctx.emit((state.accumulator + const) % 100_000)
        elif op == "read_emit":
            r = ctx.read_int("w")
            ctx.emit(r % 100_000)
        else:  # pragma: no cover - guarded by generator
            raise ValueError(f"unknown op {op!r}")
        # Fold something position-dependent into the checksum every step.
        state.checksum = (state.checksum * 1_000_003 + state.accumulator + len(state.trail)) % (2**61 - 1)

    return transition


def build_graph(spec: GraphSpec) -> Graph:
    nodes: dict[str, Node] = {}
    for ns in spec.nodes:
        edges: list[Edge] = []
        for b in ns.branches:
            if b.kind == "skip":
                edges.append(Edge(target=b.target, predicate=_skip_pred(b.mod, b.rem)))
            else:  # loop
                edges.append(Edge(target=b.target, predicate=_loop_pred(ns.name, b.mod, b.rem, b.cap)))
        edges.append(Edge(target=ns.default))  # default must be last
        nodes[ns.name] = Node(name=ns.name, transition=_make_transition(ns), edges=edges)
    return Graph(nodes=nodes, start=spec.nodes[0].name, name=spec.name)


def _skip_pred(mod: int, rem: int):
    def pred(state: WorkflowState) -> bool:
        return state.accumulator % mod == rem

    return pred


def _loop_pred(name: str, mod: int, rem: int, cap: int):
    def pred(state: WorkflowState) -> bool:
        return state.branch_visits.get(name, 0) < cap and state.accumulator % mod == rem

    return pred


def generate_spec(seed: int, min_nodes: int = 5, max_nodes: int = 14) -> GraphSpec:
    rng = random.Random(seed)
    m = rng.randint(min_nodes, max_nodes)
    names = [f"n{i}" for i in range(m)]
    nodes: list[NodeSpec] = []
    for i, name in enumerate(names):
        op = rng.choice(OPS)
        ns = NodeSpec(
            name=name,
            op=op,
            const=rng.randint(1, 50),
            tally_key=rng.choice(("x", "y", "z")),
            default=names[i + 1] if i + 1 < m else END,
        )
        # Optional forward skip to i+2.
        if i + 2 < m and rng.random() < 0.30:
            ns.branches.append(
                BranchSpec(kind="skip", mod=rng.randint(2, 5), rem=rng.randint(0, 1), target=names[i + 2])
            )
        # Optional bounded backward loop to an earlier node.
        if i >= 2 and rng.random() < 0.25:
            back = names[rng.randint(max(0, i - 3), i - 1)]
            ns.branches.append(
                BranchSpec(kind="loop", mod=rng.randint(2, 4), rem=0, target=back, cap=rng.randint(1, 3))
            )
        nodes.append(ns)
    return GraphSpec(name=f"g{seed}", nodes=nodes)


def generate_dataset(base_seed: int, count: int) -> list[GraphSpec]:
    return [generate_spec(base_seed * 1_000_003 + i) for i in range(count)]


def dataset_checksum(specs: list[GraphSpec]) -> str:
    return hash_obj([s.as_dict() for s in specs])


def initial_state() -> WorkflowState:
    return WorkflowState()
