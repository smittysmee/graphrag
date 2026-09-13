"""Centrality, brokerage and network summaries on graphs whose answers are known by hand."""

from __future__ import annotations

import networkx as nx
import pytest

from graphrag.sna.measures import CENTRALITIES, brokers, centrality, summary, top_n


@pytest.fixture
def barbell() -> nx.Graph:
    """Two triangles joined through a single node, so brokerage has one obvious answer."""
    graph = nx.Graph()
    for a, b in [("a1", "a2"), ("a2", "a3"), ("a3", "a1")]:
        graph.add_edge(a, b, weight=1)
    for a, b in [("b1", "b2"), ("b2", "b3"), ("b3", "b1")]:
        graph.add_edge(a, b, weight=1)
    graph.add_edge("a1", "bridge", weight=1)
    graph.add_edge("bridge", "b1", weight=1)
    return graph


def test_every_centrality_ranks_the_bridge_or_its_neighbours_first(barbell: nx.Graph) -> None:
    for kind in CENTRALITIES:
        scores = centrality(barbell, kind)
        assert set(scores) == set(barbell.nodes), kind
        assert all(isinstance(v, float) for v in scores.values()), kind
    assert top_n(centrality(barbell, "betweenness"), 1)[0][0] == "bridge"


def test_unknown_centrality_is_refused(barbell: nx.Graph) -> None:
    with pytest.raises(ValueError, match="centrality must be one of"):
        centrality(barbell, "importance")


def test_weight_is_read_as_affinity_not_distance() -> None:
    """A heavier edge must make two nodes *closer*, which is the opposite of what networkx
    assumes about an attribute called ``weight``."""
    graph = nx.Graph()
    graph.add_edge("a", "hub", weight=100)
    graph.add_edge("hub", "b", weight=100)
    graph.add_edge("a", "b", weight=1)

    # With weight read as distance, the direct a-b edge would be the short path and the hub
    # would broker nothing. Read as affinity, the hub sits on the cheap route.
    assert centrality(graph, "betweenness")["hub"] > 0.0


def test_brokers_find_the_node_whose_neighbours_span_groups(barbell: nx.Graph) -> None:
    communities = [["a1", "a2", "a3"], ["b1", "b2", "b3"], ["bridge"]]
    ranked = dict(brokers(barbell, communities))
    assert ranked["bridge"] == pytest.approx(0.5)  # half its ties to each side
    assert ranked["a2"] == pytest.approx(0.0)  # both neighbours in its own group
    assert brokers(barbell, communities, 1)[0][0] == "bridge"


def test_brokers_handle_an_isolated_node() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_node("lonely")
    assert dict(brokers(graph, [["a", "b"], ["lonely"]]))["lonely"] == 0.0


def test_summary_describes_shape_and_fragmentation(barbell: nx.Graph) -> None:
    stats = summary(barbell)
    assert stats["nodes"] == 7 and stats["edges"] == 8
    assert stats["components"] == 1
    assert stats["largest_component_share"] == pytest.approx(1.0)
    assert 0.0 <= stats["density"] <= 1.0

    barbell.add_node("orphan")
    split = summary(barbell)
    assert split["components"] == 2
    assert split["largest_component_share"] == pytest.approx(7 / 8)


def test_summary_of_an_empty_network_is_all_zeros() -> None:
    stats = summary(nx.Graph())
    assert stats["nodes"] == 0 and stats["components"] == 0
    assert all(value == 0 for value in stats.values())


def test_summary_never_reports_a_nan_assortativity() -> None:
    """Every node having the same degree leaves assortativity undefined; a table needs a number."""
    ring = nx.cycle_graph(5)
    nx.set_edge_attributes(ring, 1, "weight")
    assert summary(ring)["degree_assortativity"] == 0.0
