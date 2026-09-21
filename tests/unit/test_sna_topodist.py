"""Topological distances (ch. 48), on data with a known answer.

Every expected number below is derived independently of :mod:`graphrag.sna.topodist` -- by
closed-form linear algebra (the spectral distance), by hand-tracing the seven NetSimile features
over a star graph's two node roles, or by exact rational arithmetic on the fast-belief-propagation
matrix (DeltaCon) -- and only then checked against what the module returns.
"""

from __future__ import annotations

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.generators import barabasi_albert, erdos_renyi_gnp
from graphrag.sna.topodist import (
    align_by_id,
    compare_topology,
    delta_con,
    fuse_networks,
    netsimile_distance,
    netsimile_signature,
    portrait_divergence,
    render_topology,
    spectral_distance,
)


def _triangle_pair(offset: int = 0) -> nx.Graph:
    """Two disjoint triangles, node ids starting at ``offset``."""
    graph = nx.Graph()
    for base in (offset, offset + 3):
        nodes = [base, base + 1, base + 2]
        graph.add_edges_from([(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])])
    return graph


def _clique(nodes: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]


def _disjoint_cliques(sizes: list[int], offset: int = 0) -> nx.Graph:
    graph = nx.Graph()
    n = offset
    for size in sizes:
        nodes = list(range(n, n + size))
        graph.add_edges_from(_clique(nodes))
        n += size
    return graph


# ----------------------------------------------------------------------------- isomorphism, size


def test_the_alignment_free_distances_are_zero_between_a_graph_and_a_relabelled_copy() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    relabelled = nx.relabel_nodes(graph, {0: "w", 1: "x", 2: "y", 3: "z"})
    assert spectral_distance(graph, relabelled).distance == pytest.approx(0.0, abs=1e-9)
    assert netsimile_distance(graph, relabelled).distance == pytest.approx(0.0, abs=1e-9)
    assert portrait_divergence(graph, relabelled).divergence == pytest.approx(0.0, abs=1e-9)


def test_delta_con_is_zero_between_a_graph_and_itself() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    result = delta_con(graph, graph)
    assert result.distance == pytest.approx(0.0, abs=1e-9)
    assert result.similarity == pytest.approx(1.0, abs=1e-9)


def test_every_distance_is_positive_once_an_edge_is_added() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    heavier = graph.copy()
    heavier.add_edge(1, 3)
    assert spectral_distance(graph, heavier).distance > 0
    assert netsimile_distance(graph, heavier).distance > 0
    assert portrait_divergence(graph, heavier).divergence > 0
    assert delta_con(graph, heavier).distance > 0


# ----------------------------------------------------------------------------- §48.1 spectral


def test_spectral_distance_of_a_four_node_path_and_star_matches_the_hand_computed_spectrum() -> (
    None
):
    """The symmetric normalised Laplacian spectrum of ``P4`` is exactly ``{0, 1/2, 3/2, 2}`` and
    of the 4-node star ``S4`` exactly ``{0, 1, 1, 2}`` -- both standard closed forms, not read off
    this module. Discarding the trivial 0 leaves ``[0.5, 1.5, 2]`` against ``[1, 1, 2]``, an
    L2 distance of ``sqrt(0.25 + 0.25 + 0) = sqrt(0.5)``."""
    path = nx.path_graph(4)
    star = nx.star_graph(3)
    result = spectral_distance(path, star)
    assert result.distance == pytest.approx(np.sqrt(0.5), abs=1e-9)
    assert result.dims == 3


def test_spectral_distance_pads_the_shorter_spectrum_rather_than_truncating_the_longer() -> None:
    small = nx.path_graph(3)
    big = nx.path_graph(3)
    big.add_edges_from([(2, 3), (3, 4)])
    result = spectral_distance(small, big)
    assert result.dims == big.number_of_nodes() - 1


def test_spectral_distance_refuses_a_graph_with_fewer_than_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least two nodes"):
        spectral_distance(nx.Graph([(0, 1)]), nx.empty_graph(1))


# ----------------------------------------------------------------------------- §48.1 NetSimile


def test_netsimile_signature_on_a_star_matches_the_hand_computed_moments() -> None:
    """A star with centre ``c`` and four leaves has two node roles. By hand: the centre has
    degree 4, clustering 0, mean neighbour degree 1, mean neighbour clustering 0, 4 edges in its
    egonet (the whole graph), 0 edges leaving it, and 0 two-hop neighbours (its neighbours have no
    other neighbours). Each leaf has degree 1, clustering 0, mean neighbour degree 4 (its only
    neighbour is the centre), mean neighbour clustering 0, 1 edge in its egonet (itself-centre),
    3 edges leaving it (the centre's other three spokes), and 3 two-hop neighbours (the other
    three leaves, reached through the centre)."""
    star = nx.star_graph(4)  # node 0 is the centre; 1..4 are leaves
    signature = netsimile_signature(star)
    assert signature.shape == (35,)

    degree = [4.0, 1.0, 1.0, 1.0, 1.0]  # centre, then the four leaves
    expected_degree_moments = [
        float(np.median(degree)),
        float(np.mean(degree)),
        float(np.std(degree)),
    ]
    assert signature[:3] == pytest.approx(expected_degree_moments, abs=1e-9)

    # clustering is 0 for every node of a star (no triangles) -- a constant feature, so this
    # module reports skew and kurtosis as 0.0 rather than scipy's nan.
    assert signature[5:10] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-9)

    avg_neighbor_degree = [1.0, 4.0, 4.0, 4.0, 4.0]
    assert signature[10] == pytest.approx(float(np.median(avg_neighbor_degree)), abs=1e-9)
    assert signature[11] == pytest.approx(float(np.mean(avg_neighbor_degree)), abs=1e-9)

    egonet_edges = [4.0, 1.0, 1.0, 1.0, 1.0]
    assert signature[20] == pytest.approx(float(np.median(egonet_edges)), abs=1e-9)
    assert signature[21] == pytest.approx(float(np.mean(egonet_edges)), abs=1e-9)

    leaving = [0.0, 3.0, 3.0, 3.0, 3.0]
    assert signature[25] == pytest.approx(float(np.median(leaving)), abs=1e-9)
    assert signature[26] == pytest.approx(float(np.mean(leaving)), abs=1e-9)

    two_hop = [0.0, 3.0, 3.0, 3.0, 3.0]
    assert signature[30] == pytest.approx(float(np.median(two_hop)), abs=1e-9)
    assert signature[31] == pytest.approx(float(np.mean(two_hop)), abs=1e-9)


def test_netsimile_distance_is_zero_between_two_stars_of_the_same_size() -> None:
    a = nx.star_graph(4)
    b = nx.relabel_nodes(nx.star_graph(4), {0: "hub", 1: "l1", 2: "l2", 3: "l3", 4: "l4"})
    assert netsimile_distance(a, b).distance == pytest.approx(0.0, abs=1e-9)


def test_netsimile_signature_refuses_an_empty_graph() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        netsimile_signature(nx.Graph())


# ----------------------------------------------------------------------------- §48.1 DeltaCon


def test_delta_con_affinity_follows_the_fast_belief_propagation_closed_form() -> None:
    """The 3-node path ``0-1-2`` against the same path missing its ``(1, 2)`` edge. Exact
    rational arithmetic (Gaussian elimination on ``[I + eps^2 D - eps A]`` by hand, ``eps = 1 /
    (1 + max degree)``) gives the affinity matrices' own ``(0, 0)`` entry as ``909/920`` for the
    full path (``eps = 1/3``) and ``20/21`` for the path missing the edge (``eps = 1/2``); this
    recomputes both and checks the module's output against them."""
    full = nx.Graph()
    full.add_edges_from([(0, 1), (1, 2)])
    missing = nx.Graph()
    missing.add_nodes_from([0, 1, 2])
    missing.add_edge(0, 1)

    result = delta_con(full, missing)
    assert result.epsilon_a == pytest.approx(1.0 / 3.0)
    assert result.epsilon_b == pytest.approx(1.0 / 2.0)
    assert result.affinity_a[0, 0] == pytest.approx(float(Fraction(909, 920)), abs=1e-9)
    assert result.affinity_b[0, 0] == pytest.approx(float(Fraction(20, 21)), abs=1e-9)
    assert result.distance > 0


def test_delta_con_notes_when_the_two_graphs_share_no_node_id() -> None:
    a = nx.Graph([(0, 1)])
    b = nx.relabel_nodes(nx.Graph([(0, 1)]), {0: "a", 1: "b"})
    result = delta_con(a, b)
    assert result.alignment.overlap == 0.0
    assert "share no node id" in result.alignment.note


def test_align_by_id_reports_partial_overlap() -> None:
    a = nx.Graph([(1, 2), (2, 3)])
    b = nx.Graph([(3, 4), (4, 5), (5, 6), (6, 7)])
    alignment = align_by_id(a, b)
    assert alignment.shared == (3,)
    assert alignment.nodes == (1, 2, 3, 4, 5, 6, 7)
    assert alignment.overlap == pytest.approx(1 / 7)
    assert "only" in alignment.note


def test_align_by_id_handles_node_ids_of_different_types() -> None:
    a = nx.Graph([(0, 1)])
    b = nx.relabel_nodes(nx.Graph([(0, 1)]), {0: "a", 1: "b"})
    alignment = align_by_id(a, b)
    assert alignment.overlap == 0.0
    assert set(alignment.nodes) == {0, 1, "a", "b"}


# ------------------------------------------------------------------- §48.1 portrait divergence


def test_portrait_divergence_is_zero_between_two_same_size_disjoint_clique_graphs() -> None:
    a = _triangle_pair()
    b = _triangle_pair(offset=100)
    assert portrait_divergence(a, b).divergence == pytest.approx(0.0, abs=1e-9)


def test_portrait_divergence_is_larger_between_different_sized_disjoint_cliques() -> None:
    same_reference = _triangle_pair()
    same = _triangle_pair(offset=100)
    different = _disjoint_cliques([3, 4], offset=200)
    d_same = portrait_divergence(same_reference, same).divergence
    d_different = portrait_divergence(same_reference, different).divergence
    assert d_same == pytest.approx(0.0, abs=1e-9)
    assert d_different > d_same


def test_portrait_divergence_refuses_an_empty_graph() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        portrait_divergence(nx.Graph(), nx.Graph([(0, 1)]))


# ----------------------------------------------------------------------------- against random


def test_a_scale_free_graph_is_closer_to_another_draw_than_to_an_erdos_renyi_graph() -> None:
    """Seeded so the ordering is reproducible: two draws of the same Barabasi-Albert generator
    against an Erdos-Renyi graph matched on node count and edge count, checked under every one of
    the four distances."""
    n = 30
    first = barabasi_albert(n, 2, seed=1)
    second = barabasi_albert(n, 2, seed=2)
    density = 2 * first.number_of_edges() / (n * (n - 1))
    random_graph = erdos_renyi_gnp(n, density, seed=3)

    same_generator = compare_topology(first, second)
    against_random = compare_topology(first, random_graph)

    assert same_generator.spectral.distance < against_random.spectral.distance
    assert same_generator.netsimile.distance < against_random.netsimile.distance
    assert same_generator.delta_con.distance < against_random.delta_con.distance
    assert same_generator.portrait.divergence < against_random.portrait.divergence


# ----------------------------------------------------------------------------- gathered report


def test_compare_topology_gathers_all_four_and_render_topology_names_them() -> None:
    a = nx.star_graph(4)
    b = nx.path_graph(5)
    result = compare_topology(a, b)
    lines = render_topology(result)
    text = "\n".join(lines)
    assert "## Topological distances" in text
    assert "spectral distance" in text
    assert "NetSimile" in text
    assert "DeltaCon similarity" in text
    assert "portrait divergence" in text


def test_render_topology_discloses_the_size_confound_beside_the_padded_rows() -> None:
    """The spectral-distance and portrait-divergence rows pad the smaller graph up to the
    larger's shape, so a reader has to be told *next to the number* that part of what it reads
    is size, not shape -- not only in a guide rule or a docstring."""
    smaller = nx.star_graph(4)  # 5 nodes
    larger = nx.path_graph(9)  # 9 nodes
    lines = render_topology(compare_topology(smaller, larger))
    spectral_row = next(line for line in lines if line.startswith("| spectral distance"))
    portrait_row = next(line for line in lines if line.startswith("| portrait divergence"))
    assert "inflated by size alone, 5 vs 9 nodes" in spectral_row
    assert "inflated by size alone, 5 vs 9 nodes" in portrait_row

    same_size = render_topology(compare_topology(nx.star_graph(4), nx.path_graph(5)))
    spectral_row_same = next(line for line in same_size if line.startswith("| spectral distance"))
    portrait_row_same = next(line for line in same_size if line.startswith("| portrait divergence"))
    assert "inflated by size alone" not in spectral_row_same
    assert "inflated by size alone" not in portrait_row_same


# ----------------------------------------------------------------------------- §48.3 fusion


def test_fuse_networks_averages_weight_across_observations_and_keeps_it_above_the_threshold() -> (
    None
):
    a = nx.Graph()
    a.add_edge("x", "y", weight=3)
    b = nx.Graph()
    b.add_edge("x", "y", weight=1)
    b.add_edge("y", "z", weight=2)

    fused = fuse_networks([a, b])
    assert fused["x"]["y"]["weight"] == pytest.approx(2.0)
    assert fused["y"]["z"]["weight"] == pytest.approx(1.0)

    thresholded = fuse_networks([a, b], min_weight=1.5)
    assert thresholded.has_edge("x", "y")
    assert not thresholded.has_edge("y", "z")


def test_fuse_networks_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one graph"):
        fuse_networks([])
