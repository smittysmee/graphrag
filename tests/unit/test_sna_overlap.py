"""Chapter 38's overlapping coverage, held to answers that existed before the code did.

Every claim asserted here comes from one of three places, never from "the function returned
something":

* **The book's own worked mechanics.** Two triangles sharing an edge (k-1 = 2 nodes) percolate
  into one k=3 community; sharing only a node does not (p. 548-549). A shared node's own report
  belongs in more than one community once its edges say so.
* **A planted graph whose answer is known before it is built.** Three 6-node cliques, each
  overlapping its neighbour in exactly two nodes -- one short of what k=4 percolation needs to
  merge them -- recovers all three memberships, shared nodes included. Two dense communities
  glued by a single bridging node recovers the bridge inside more than one link community
  (Figure 38.12's own claim: "node 4 belongs to three communities"). A planted overlap region
  built strictly denser than the two communities it sits between fires the §38.7 "boundary is
  the core" reading and nothing else.
* **An arithmetic identity.** Overlapping NMI of a cover against itself is exactly 1; against an
  independently reshuffled cover of the same sizes it is close to 0, which is what independence
  gives it (§38.1's own caution that it is never exactly 0).
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence

import networkx as nx
import pytest

from graphrag.sna.overlap import (
    DEFAULT_JACCARD_THRESHOLD,
    build_overlap_report,
    ego_splitting,
    flatten_cover,
    k_clique_communities,
    link_clustering,
    overlap_paradox,
    overlap_payload,
    overlapping_nmi,
    render_overlap,
)


def _clique_edges(nodes: Sequence[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(nodes, 2))


def _clique_graph(*cliques: Sequence[str]) -> nx.Graph:
    graph = nx.Graph()
    for clique in cliques:
        graph.add_edges_from(_clique_edges(clique))
    return graph


# ------------------------------------------------------------------ §38.3, k-clique percolation


def test_two_triangles_sharing_an_edge_percolate_into_one_community() -> None:
    """p. 548-549: sharing k-1 = 2 nodes (an edge) merges two 3-cliques into one community."""
    graph = _clique_graph(["1", "2", "3"], ["2", "3", "4"])
    result = k_clique_communities(graph, k=3)
    assert result.communities == [["1", "2", "3", "4"]]


def test_two_triangles_sharing_only_a_node_stay_separate() -> None:
    """Sharing one node, short of k-1 = 2, does not percolate: two communities, not one."""
    graph = _clique_graph(["1", "2", "3"], ["3", "4", "5"])
    result = k_clique_communities(graph, k=3)
    assert sorted(result.communities) == [["1", "2", "3"], ["3", "4", "5"]]


def test_k_clique_percolation_recovers_a_planted_overlap_including_the_shared_nodes() -> None:
    """Three 6-cliques, each sharing exactly 2 nodes with its neighbour -- one short of what
    k=4 percolation needs (k-1 = 3) -- so all three stay separate and every shared node is
    correctly reported as a member of both of the cliques it was planted into."""
    a = [f"a{i}" for i in range(6)]
    b = a[-2:] + [f"b{i}" for i in range(4)]
    c = b[-2:] + [f"c{i}" for i in range(4)]
    graph = _clique_graph(a, b, c)
    result = k_clique_communities(graph, k=4)
    communities = sorted(sorted(community) for community in result.communities)
    assert communities == [sorted(a), sorted(b), sorted(c)]
    # the shared nodes are members of both cliques they were planted into
    assert set(a) & set(b) == {"a4", "a5"}
    for shared in set(a) & set(b):
        assert sum(1 for community in result.communities if shared in community) == 2
    assert result.too_low_degree == []
    assert result.uncovered == []


def test_a_node_below_k_minus_one_degree_can_never_join_and_is_named() -> None:
    """p. 550: "a node with degree equal to one cannot be part of a 3-clique." """
    graph = _clique_graph(["1", "2", "3"])
    graph.add_edge("3", "4")  # node 4 has degree 1, below k-1=2 for k=3
    result = k_clique_communities(graph, k=3)
    assert result.communities == [["1", "2", "3"]]
    assert result.too_low_degree == ["4"]
    assert result.uncovered == []


def test_k_clique_percolation_refuses_k_below_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        k_clique_communities(nx.cycle_graph(5), k=1)


# ------------------------------------------------------------------ §38.5, link clustering


def test_link_clustering_puts_a_bridge_node_in_more_than_one_community() -> None:
    """A single node joining two dense cliques by one edge each ends up covered by more than one
    community -- Figure 38.12's own claim, made about a bridging node's several link
    communities, not about being folded into the two big ones."""
    a = [f"x{i}" for i in range(5)]
    b = [f"y{i}" for i in range(5)]
    graph = _clique_graph(a, b)
    graph.add_edge("bridge", a[0])
    graph.add_edge("bridge", b[0])
    result = link_clustering(graph)
    counts: dict[str, int] = {}
    for community in result.communities:
        for node in community:
            counts[node] = counts.get(node, 0) + 1
    assert counts["bridge"] >= 2
    # the two cliques themselves come back whole, each its own community
    community_sets = [set(c) for c in result.communities]
    assert set(a) in community_sets
    assert set(b) in community_sets
    assert result.partition_density > 0


def test_link_clustering_needs_at_least_two_edges() -> None:
    with pytest.raises(ValueError, match="at least two edges"):
        link_clustering(nx.Graph([("a", "b")]))


def test_link_clustering_refuses_a_network_too_large_for_a_dense_similarity_matrix() -> None:
    from graphrag.sna import overlap as overlap_module

    graph = nx.gnm_random_graph(200, overlap_module.MAX_LINK_CLUSTERING_EDGES + 50, seed=1)
    graph = nx.relabel_nodes(graph, str)
    with pytest.raises(ValueError, match="dense similarity matrix"):
        link_clustering(graph)


# ------------------------------------------------------------------ §38.5, ego-splitting


def test_ego_splitting_places_a_two_component_ego_network_into_two_communities() -> None:
    """§38.5's own known answer: a node whose ego network (minus itself) has two components is
    placed in two communities, one anchored on each side.

    At the book's own exercise-3 default threshold (0.1) a small toy graph like this one has
    single-node overlaps between the two sides' candidates (1 shared node out of a 7-node union)
    that clear 0.1 and over-merge everything into one community -- an artefact of Jaccard on
    small sets, not a defect of the method. A higher threshold, still well inside "ignoring
    singletons", separates the two sides cleanly.
    """
    graph = _clique_graph(["p0", "p1", "p2"], ["q0", "q1", "q2"])
    graph.add_edges_from([("u", "p0"), ("u", "p1"), ("u", "q0"), ("u", "q1")])
    result = ego_splitting(graph, seed=1, jaccard_threshold=0.2)
    communities = sorted(sorted(c) for c in result.communities)
    assert communities == [
        ["p0", "p1", "p2", "u"],
        ["q0", "q1", "q2", "u"],
    ]
    assert sum(1 for c in result.communities if "u" in c) == 2


def test_ego_splitting_at_the_book_default_threshold_still_runs() -> None:
    """§38.9 exercise 3's own number (0.1) is the default; it must not raise, whatever it merges."""
    graph = _clique_graph(["p0", "p1", "p2"], ["q0", "q1", "q2"])
    graph.add_edges_from([("u", "p0"), ("u", "p1"), ("u", "q0"), ("u", "q1")])
    result = ego_splitting(graph, seed=1)
    assert result.jaccard_threshold == DEFAULT_JACCARD_THRESHOLD
    assert result.communities  # something was found, whatever it merged into


def test_ego_splitting_refuses_an_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="jaccard_threshold"):
        ego_splitting(nx.cycle_graph(5), jaccard_threshold=1.0)


# ------------------------------------------------------------------ §38.1, overlapping NMI


def _random_cover(nodes: Sequence[str], sizes: Sequence[int], seed: int) -> list[list[str]]:
    shuffled = list(nodes)
    random.Random(seed).shuffle(shuffled)
    cover: list[list[str]] = []
    start = 0
    for size in sizes:
        cover.append(shuffled[start : start + size])
        start += size
    return cover


def test_overlapping_nmi_of_a_cover_with_itself_is_one() -> None:
    nodes = [f"n{i}" for i in range(60)]
    cover = _random_cover(nodes, [20, 20, 20], seed=1)
    assert overlapping_nmi(cover, cover) == pytest.approx(1.0)


def test_overlapping_nmi_against_an_independent_reshuffling_is_near_zero() -> None:
    """§38.1's own caution: it is never exactly 0 for independent vectors -- but is far below the
    self-comparison's 1.0, the qualitative claim the exercise (§38.9.2) asks for."""
    nodes = [f"n{i}" for i in range(80)]
    cover_a = _random_cover(nodes, [20, 20, 20, 20], seed=2)
    values = [
        overlapping_nmi(cover_a, _random_cover(nodes, [20, 20, 20, 20], seed=100 + trial))
        for trial in range(5)
    ]
    assert max(values) < 0.15
    assert all(value >= 0.0 for value in values)


def test_overlapping_nmi_refuses_an_empty_cover() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        overlapping_nmi([], [["a", "b"]])


# ------------------------------------------------------------------ §38.7, the overlap paradox


def test_the_overlap_paradox_fires_when_the_shared_region_is_denser_than_either_community() -> None:
    """Figure 38.15: the shared nodes are a complete clique on their own and fully bipartite-tied
    into each side, so the overlap (density 1.0) outranks both communities (density 2/3) --
    "the boundary is the core"."""
    shared = [f"s{i}" for i in range(4)]
    left = [f"a{i}" for i in range(6)]
    right = [f"b{i}" for i in range(6)]
    graph = nx.Graph()
    graph.add_edges_from(_clique_edges(shared))
    for node in left:
        for s in shared:
            graph.add_edge(node, s)
    for node in right:
        for s in shared:
            graph.add_edge(node, s)
    community_a = shared + left
    community_b = shared + right
    paradox = overlap_paradox(graph, [community_a, community_b])
    assert len(paradox.rows) == 1
    row = paradox.rows[0]
    assert row.overlap_size == 4
    assert row.overlap_density == pytest.approx(1.0)
    assert row.density_a == pytest.approx(row.density_b)
    assert row.overlap_density > row.density_a
    assert "boundary is the core" in row.verdict
    assert paradox.denser_than_both == 1
    assert paradox.sparser_than_both == 0


def test_the_overlap_paradox_has_nothing_to_check_below_two_shared_nodes() -> None:
    graph = _clique_graph(["1", "2", "3"], ["3", "4", "5"])
    paradox = overlap_paradox(graph, [["1", "2", "3"], ["3", "4", "5"]])
    assert paradox.rows == []
    assert paradox.denser_than_both == 0


# ------------------------------------------------------------------ flattening for chapter 36


def test_flattening_keeps_every_node_in_its_largest_community_and_says_what_was_lost() -> None:
    """Node "c" is in both communities; the 4-node one is larger, so it keeps "c" and the
    3-node one loses it -- flattening favours the larger community, never the first-listed."""
    cover = [["a", "b", "c"], ["c", "d", "e", "f"]]
    partition, notes = flatten_cover(_clique_graph(*cover), cover)
    assert sorted(partition) == [["a", "b"], ["c", "d", "e", "f"]]
    assert any("1 of" in note and "more than one community" in note for note in notes)


def test_flattening_a_disjoint_cover_changes_nothing() -> None:
    partition, notes = flatten_cover(nx.Graph(), [["a", "b"], ["c", "d"]])
    assert sorted(partition) == [["a", "b"], ["c", "d"]]
    assert any("changed nothing" in note for note in notes)


# ------------------------------------------------------------------ the full report


def test_build_overlap_report_runs_all_three_methods_and_prints_every_section() -> None:
    graph = _clique_graph(["1", "2", "3", "4"], ["4", "5", "6", "7"])
    graph.graph["frame"] = "a planted overlap, for a test."
    for method in ("clique", "link", "ego"):
        report = build_overlap_report(graph, method, seed=3)
        text = "\n".join(render_overlap(report))
        assert "## Overlapping coverage" in text
        assert "**Sampling frame.** a planted overlap, for a test." in text
        assert "## Community evaluation" in text  # ch. 36's battery, reused whole
        assert "### The overlap paradox (§38.7)" in text
        assert "### Against this network's own disjoint answer (§38.1)" in text
        payload = overlap_payload(report)
        assert payload["communities"] == report.communities
        assert payload["evaluation"]["communities"] == len(report.flattened)
        assert "paradox" in payload


def test_build_overlap_report_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        build_overlap_report(nx.cycle_graph(5), "nope")


def test_an_empty_network_still_produces_a_report_that_says_so() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    report = build_overlap_report(graph, "clique")
    assert report.communities == []
    assert math.isnan(report.overlap_nmi_vs_baseline)
    text = "\n".join(render_overlap(report))
    assert "nothing below is a measurement" in text
