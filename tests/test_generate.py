from __future__ import annotations

from fsmckpt.generate import (
    build_graph,
    dataset_checksum,
    generate_dataset,
    generate_spec,
    initial_state,
)


def test_generated_graph_is_valid():
    for seed in range(30):
        g = build_graph(generate_spec(seed))  # Graph.__post_init__ validates
        assert g.nodes


def test_generation_is_deterministic():
    a = generate_spec(123).as_dict()
    b = generate_spec(123).as_dict()
    assert a == b


def test_different_seeds_usually_differ():
    specs = {dataset_checksum([generate_spec(s)]) for s in range(10)}
    assert len(specs) > 1


def test_dataset_checksum_stable():
    d1 = generate_dataset(1000, 20)
    d2 = generate_dataset(1000, 20)
    assert dataset_checksum(d1) == dataset_checksum(d2)


def test_dataset_checksum_changes_with_seed():
    assert dataset_checksum(generate_dataset(1, 10)) != dataset_checksum(generate_dataset(2, 10))


def test_generated_graphs_terminate():
    from fsmckpt.executor import reference_run

    for seed in range(25):
        g = build_graph(generate_spec(seed))
        trace = reference_run(g, initial_state(), seed)
        assert 1 <= trace.length < 10_000


def test_dataset_size():
    assert len(generate_dataset(5, 50)) == 50
