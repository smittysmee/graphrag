"""Centrality, brokerage and network summaries on graphs whose answers are known by hand."""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.export import describe as describe_network
from graphrag.sna.matrices import adjacency, eigenpairs
from graphrag.sna.measures import (
    CENTRALITIES,
    CENTRALITY_MEANING,
    CHEAP_CENTRALITIES,
    CLUSTERING_KINDS,
    DIRECTED_CENTRALITIES,
    EXPENSIVE_CENTRALITIES,
    FLATTENED,
    brokers,
    centralities_for,
    centrality,
    centrality_meaning,
    cliques,
    clustering,
    density_note,
    density_report,
    directed_notes,
    independent_set,
    local_clustering_summary,
    reciprocity,
    render_density,
    summary,
    top_n,
    undirected_view,
)
from tests.legendary import karate_club, les_miserables


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


def test_cheap_and_expensive_centralities_partition_both_batteries() -> None:
    """ATL-F2: every centrality this module can compute is exactly one of cheap or expensive,
    for both the undirected and the directed battery, and no name is in both."""
    cheap, expensive = set(CHEAP_CENTRALITIES), set(EXPENSIVE_CENTRALITIES)
    assert not (cheap & expensive)
    assert cheap | expensive == set(CENTRALITIES) | set(DIRECTED_CENTRALITIES)


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


# ------------------------------------------------------------- directed networks (ch. 6, §10.3)


@pytest.fixture
def messages() -> nx.DiGraph:
    """The Atlas's own reciprocity example (§10.3, Figure 10.6), rebuilt by its counts.

    Five connected pairs, two of which carry both edges, which the book reads as "reciprocity is
    2/5, or [...] the probability of a connection to be reciprocated is 40%".
    """
    graph = nx.DiGraph()
    for source, target in [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d")]:
        graph.add_edge(source, target, weight=1)
    graph.add_edge("d", "e", weight=1)
    graph.add_edge("e", "a", weight=1)
    return graph


def test_reciprocity_counts_pairs_the_way_the_book_does(messages: nx.DiGraph) -> None:
    """Pairs reciprocated over pairs connected, which is not edges reciprocated over edges."""
    assert messages.number_of_edges() == 7
    assert reciprocity(messages) == pytest.approx(0.4)  # 2 of 5 connected pairs
    assert nx.overall_reciprocity(messages) == pytest.approx(4 / 7)  # the other convention
    assert reciprocity(nx.DiGraph([("a", "b"), ("b", "a")])) == pytest.approx(1.0)
    assert reciprocity(nx.DiGraph([("a", "b")])) == 0.0
    assert reciprocity(nx.DiGraph()) == 0.0
    assert reciprocity(nx.Graph([("a", "b")])) == 0.0  # undefined undirected: every edge is both


def test_summary_of_a_directed_network_reports_reciprocity_and_weak_components(
    messages: nx.DiGraph,
) -> None:
    stats = summary(messages)
    assert stats["reciprocity"] == pytest.approx(0.4)
    assert stats["nodes"] == 5 and stats["edges"] == 7
    messages.add_node("orphan")
    assert summary(messages)["components"] == 2  # weakly connected: direction ignored
    assert "reciprocity" not in summary(nx.cycle_graph(4))
    assert summary(nx.DiGraph())["reciprocity"] == 0.0


def test_in_and_out_degree_are_different_questions() -> None:
    """A star everything points at: one node has all the in-degree and none of the out."""
    graph = nx.DiGraph()
    for name in ("a", "b", "c"):
        graph.add_edge(name, "hub", weight=2)
    assert centralities_for(graph) == DIRECTED_CENTRALITIES
    assert centralities_for(nx.Graph()) == CENTRALITIES
    assert centrality(graph, "in_degree")["hub"] == pytest.approx(1.0)  # 3 of 3 possible
    assert centrality(graph, "out_degree")["hub"] == 0.0
    assert centrality(graph, "in_degree")["a"] == 0.0
    assert centrality(graph, "out_degree")["a"] == pytest.approx(1 / 3)
    assert centrality(graph, "weighted_in_degree")["hub"] == pytest.approx(6.0)
    assert centrality(graph, "weighted_out_degree")["a"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="defined only on a directed network"):
        centrality(nx.Graph([("a", "b")]), "in_degree")


def test_flattening_sums_the_two_directions_and_says_so(messages: nx.DiGraph) -> None:
    """A measure that cannot read direction must lose it visibly, not silently (§6.2)."""
    flat, note = undirected_view(messages)
    assert not flat.is_directed()
    assert flat.number_of_edges() == 5  # the five connected pairs
    assert flat["a"]["b"]["weight"] == 2  # a -> b and b -> a, one passage each
    assert flat["c"]["d"]["weight"] == 1
    assert "flattened undirected view" in note
    same, no_note = undirected_view(nx.Graph([("a", "b")]))
    assert no_note == "" and same.number_of_edges() == 1
    assert directed_notes(messages) and not directed_notes(nx.Graph())


def test_the_measures_that_cannot_read_direction_still_answer(messages: nx.DiGraph) -> None:
    """Every centrality in the directed battery returns a score for every node."""
    for kind in DIRECTED_CENTRALITIES:
        scores = centrality(messages, kind)
        assert set(scores) == set(messages.nodes), kind
    assert dict(brokers(messages, [["a", "b"], ["c", "d", "e"]]))["b"] > 0.0


def test_total_degree_stays_available_on_a_directed_network_and_says_what_it_is() -> None:
    """The rank tables order by total degree, so it must not be refused -- only labelled."""
    graph = nx.DiGraph()
    for name in ("a", "b", "c"):
        graph.add_edge(name, "hub", weight=2)
    assert centrality(graph, "degree")["hub"] == pytest.approx(1.0)  # 3 of 3, in plus out
    assert centrality(graph, "weighted_degree")["hub"] == pytest.approx(6.0)
    assert "in + out" in centrality_meaning("weighted_degree", graph)
    assert "in + out" in centrality_meaning("degree", graph)
    assert centrality_meaning("degree", nx.Graph()) == CENTRALITY_MEANING["degree"]


def test_closeness_on_a_directed_network_is_the_incoming_one() -> None:
    """networkx measures closeness along arriving paths, and the caption has to say so.

    On the path a -> b -> c, c is reachable from both others and reaches nobody, so it leads;
    reversing the graph is how the other question is asked, and it puts a on top instead.
    """
    path = nx.DiGraph()
    nx.add_path(path, ["a", "b", "c"])
    nx.set_edge_attributes(path, 1, "weight")
    incoming = centrality(path, "closeness")
    assert incoming["c"] > incoming["b"] > incoming["a"] == 0.0
    outgoing = centrality(path.reverse(), "closeness")
    assert outgoing["a"] > outgoing["b"] > outgoing["c"] == 0.0
    assert "being reachable" in centrality_meaning("closeness", path)
    assert centrality_meaning("closeness", nx.Graph()) == CENTRALITY_MEANING["closeness"]


def test_the_summary_conventions_of_a_directed_network_are_reported(
    messages: nx.DiGraph,
) -> None:
    """Two numbers change definition without changing name, so a note has to carry them."""
    notes = directed_notes(messages)
    assert len(notes) == 2
    assert "flattened undirected view" in notes[0]
    assert "out-in coefficient" in notes[1] and "Fagiolo" in notes[1]


def test_flattening_leaves_an_undirected_graph_exactly_as_it_was(barbell: nx.Graph) -> None:
    """The node-order contract of §8.1 holds through this ticket: same graph, same object.

    ``undirected_view`` returns the argument itself when there is nothing to flatten, so the
    eigenvector an undirected network produces is bit for bit the one ``matrices.eigenpairs``
    produces from its adjacency -- no copy, no re-ordering, no new number.
    """
    same, note = undirected_view(barbell)
    assert same is barbell and note == ""

    nodes = list(barbell.nodes)
    matrix, order = adjacency(barbell, nodes=nodes, weight="weight")
    _, vectors = eigenpairs(matrix, k=1, largest=True)
    principal = np.abs(vectors[:, 0])
    expected = dict(zip(order, principal / np.linalg.norm(principal), strict=True))
    assert centrality(barbell, "eigenvector") == {n: float(v) for n, v in expected.items()}


# ------------------------------------------------- density, clustering, cliques (ch. 12)


def test_the_karate_club_reproduces_its_published_clustering_coefficients() -> None:
    """The two numbers every paper quotes for this graph, to four decimal places.

    Transitivity 0.2557 and average clustering 0.5706 are the published values for the 78-edge
    karate club -- the copy ``tests.legendary`` pins and the one Newman and Girvan used. They
    are also the gap §12.2 exists to warn about: the same graph is "26% clustered" or "57%
    clustered" depending only on which coefficient you print.
    """
    graph, known = karate_club()
    assert known.nodes == 34 and known.edges == 78
    assert clustering(graph, "global") == pytest.approx(0.2557, abs=5e-5)
    assert clustering(graph, "average") == pytest.approx(0.5706, abs=5e-5)
    # The same numbers straight from networkx, so a future version changing one is caught here
    # rather than in a report.
    assert clustering(graph, "global") == pytest.approx(nx.transitivity(graph))
    assert clustering(graph, "average") == pytest.approx(nx.average_clustering(graph))

    local = clustering(graph, "local")
    assert isinstance(local, dict) and set(local) == set(graph.nodes)
    assert local[0] == pytest.approx(nx.clustering(graph, 0))
    assert sum(local.values()) / 34 == pytest.approx(0.5706, abs=5e-5)


def test_a_complete_graph_scores_one_on_every_coefficient() -> None:
    """K_n is the ceiling of §12.3: every triad closes, and the whole graph is one clique."""
    graph = nx.complete_graph(5)
    nx.set_edge_attributes(graph, 3, "weight")
    assert clustering(graph, "global") == pytest.approx(1.0)
    assert clustering(graph, "average") == pytest.approx(1.0)
    assert clustering(graph, "local") == dict.fromkeys(range(5), 1.0)
    # Onnela normalises by the heaviest weight, so a graph of equal weights also reads 1.0.
    assert clustering(graph, "weighted") == dict.fromkeys(range(5), 1.0)
    assert nx.density(graph) == pytest.approx(1.0)

    found = cliques(graph)
    assert found.total == 1 and found.largest_size == 5
    assert found.largest == [[0, 1, 2, 3, 4]] and found.sizes == {5: 1}
    # No two nodes are unconnected, so an independent set can hold exactly one of them (§12.4).
    assert len(independent_set(graph)) == 1


def test_a_star_closes_no_triad_anywhere() -> None:
    """The other extreme: a hub with 5 neighbours who have nothing to do with each other.

    Every local coefficient is 0 and so is the transitivity, because the 10 triads centred on
    the hub all stay open -- five structural holes' worth (§12.2). The maximal cliques are the
    edges, and the leaves are the maximum independent set.
    """
    graph = nx.star_graph(5)
    nx.set_edge_attributes(graph, 1, "weight")
    assert clustering(graph, "local") == dict.fromkeys(range(6), 0.0)
    assert clustering(graph, "average") == 0.0
    assert clustering(graph, "global") == 0.0

    found = cliques(graph)
    assert found.sizes == {2: 5} and found.largest_size == 2
    assert independent_set(graph) == [1, 2, 3, 4, 5]


def test_average_and_global_clustering_disagree_on_a_planted_hub() -> None:
    """The book's warning, planted so both numbers are known before the code runs (§12.2).

    Five triangles sharing one hub (the windmill graph F_5, 11 nodes). Every leaf has degree 2
    and a coefficient of 1; the hub has 10 neighbours, so it opens C(10,2) = 45 triads and
    closes 5 of them, for 1/9. The average is therefore (10 + 1/9) / 11 = 91/99 = 0.9192, while
    the global coefficient counts the whole network at once: 3 x 5 triangles over 45 + 10 = 55
    triads = 3/11 = 0.2727. One graph, two coefficients, a gap of 0.65.
    """
    graph = nx.windmill_graph(5, 3)
    nx.set_edge_attributes(graph, 1, "weight")
    assert graph.number_of_nodes() == 11
    assert clustering(graph, "average") == pytest.approx(91 / 99)
    assert clustering(graph, "global") == pytest.approx(3 / 11)

    report = density_report(graph)
    assert "hub-heavy case" in report.clustering_note
    assert "neither is the other" in report.clustering_note
    section = "\n".join(render_density(report))
    assert "**Average clustering** 0.9192" in section
    assert "**global clustering** 0.2727" in section


def test_les_miserables_largest_maximal_clique_is_ten_characters() -> None:
    """The students of the ABC barricade: the largest maximal clique of this graph has 10 nodes.

    Two maximal cliques reach that size, both built from the Friends of the ABC (Enjolras,
    Combeferre, Courfeyrac, Feuilly, Bahorel, Joly, Bossuet, Gavroche plus Grantaire and
    Prouvaire in one, Mabeuf and Marius in the other). Recomputed from ``networkx`` below so
    the number in this docstring is checked rather than asserted twice.
    """
    graph, known = les_miserables()
    assert known.nodes == 77 and known.edges == 254
    expected = max(len(clique) for clique in nx.find_cliques(graph))
    assert expected == 10

    found = cliques(graph)
    assert found.largest_size == 10
    assert found.largest_count == 2 and len(found.largest) == 2
    assert "Enjolras" in found.largest[0] and "Gavroche" in found.largest[0]
    assert found.total == len(list(nx.find_cliques(graph)))
    assert not found.truncated and "complete" in found.note
    # Every maximal clique is a clique: all its members are pairwise joined.
    for clique in found.largest:
        assert all(graph.has_edge(u, v) for u, v in itertools.combinations(clique, 2))


def test_k_cliques_are_the_complete_subgraphs_of_exactly_that_size() -> None:
    """§12.3's other count: a 5-clique contains C(5,3) = 10 triangles, none of them maximal."""
    found = cliques(nx.complete_graph(5), k=3)
    assert found.k == 3 and found.k_clique_count == 10
    assert found.largest_size == 5  # the maximal count is unaffected by asking for k
    assert all(len(clique) == 3 for clique in found.k_cliques)
    with pytest.raises(ValueError, match="k must be at least 1"):
        cliques(nx.complete_graph(4), k=0)


def test_the_clique_walk_says_when_its_budget_stopped_it() -> None:
    """Exponential enumeration needs a budget, and a truncated count must say it is a bound.

    The bound is Moon and Moser's, not the chapter's, so the note cites them: §12.3 defines the
    objects and says nothing about what listing them costs.
    """
    found = cliques(nx.les_miserables_graph(), limit=3)
    assert found.truncated and found.total == 3
    assert "lower bound" in found.note and "Moon and Moser" in found.note
    assert found.total < len(list(nx.find_cliques(nx.les_miserables_graph())))


def test_the_budget_bounds_the_whole_k_walk_and_not_only_its_finds() -> None:
    """A k-clique walk reaches size k last, so a budget on the finds alone bounds nothing.

    ``enumerate_all_cliques`` visits every node, then every edge, then every triangle. Counting
    only the size-k results against the budget would leave all that work unbudgeted -- on a
    few-hundred-node graph, seconds of it. The budget therefore counts every clique visited,
    and when it runs out before the walk reaches k the note says the count is a lower bound
    that can be 0 on a graph that has such cliques.
    """
    graph = nx.complete_graph(12)  # 220 triangles, 495 4-cliques: plenty to run out of budget
    generous = cliques(graph, k=4)
    assert not generous.truncated and generous.k_clique_count == 495

    starved = cliques(graph, k=4, limit=20)  # 12 nodes and 66 edges come before any 4-clique
    assert starved.truncated and starved.k_clique_count == 0
    assert "can be 0 on a graph that has them" in starved.note


@pytest.mark.parametrize(
    "graph",
    [karate_club()[0], les_miserables()[0], nx.star_graph(6), nx.cycle_graph(7)],
    ids=["karate", "les-miserables", "star", "cycle"],
)
def test_an_independent_set_holds_no_edge_and_admits_no_further_node(graph: nx.Graph) -> None:
    """Both halves of §12.4's definition: independent, and maximal.

    Independent: no pair inside the set is joined. Maximal: every node outside it has a
    neighbour inside, so nothing can be added. It is still only an approximation to the
    *maximum* independent set, which is why the report calls it a lower bound.
    """
    chosen = independent_set(graph)
    assert chosen
    assert not any(graph.has_edge(u, v) for u, v in itertools.combinations(chosen, 2))
    inside = set(chosen)
    for node in graph.nodes:
        if node not in inside:
            assert inside & set(graph.neighbors(node)), node
    assert independent_set(graph) == chosen  # deterministic: a report cannot print a number
    # that moves between runs


def test_density_is_printed_against_n_because_it_is_meaningless_without_it() -> None:
    """§12.1's argument, as its own figure states it: same average degree, different densities.

    Figure 12.1 puts a 3-node network of average degree 2 (density 1.0, every possible edge)
    beside a ~650-node network of the same average degree (density 0.0031). The note has to
    carry the node count for exactly that reason.
    """
    triangle = nx.cycle_graph(3)
    assert nx.density(triangle) == pytest.approx(1.0)
    assert "3 of the 3 edges 3 nodes could carry" in density_note(triangle)
    assert "only readable against n (§12.1)" in density_note(triangle)

    big = nx.random_regular_graph(2, 650, seed=1)  # 650 nodes, average degree 2, as in the book
    assert nx.density(big) == pytest.approx(0.0031, abs=5e-5)
    assert "650 of the 210,925 edges 650 nodes could carry" in density_note(big)

    assert "undefined for 1 node(s)" in density_note(nx.Graph([("only", "only")]))
    assert "|V|(|V|-1)" in density_note(nx.DiGraph([("a", "b")]))


def test_a_self_loop_is_not_one_of_the_pairs_a_density_counts() -> None:
    """§12.1 bans self loops from the denominator twice, so they cannot be in the numerator.

    Four nodes can carry 4*3/2 = 6 edges. A graph of five real edges and one self loop has to
    read "5 of the 6", never "6 of the 6": the loop is not one of the six pairs.
    """
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 2)
    graph.add_edge(1, 1)
    assert graph.number_of_edges() == 6  # networkx counts the loop as an edge
    note = density_note(graph)
    assert "5 of the 6 edges 4 nodes could carry" in note
    assert "1 self loop(s) are left out" in note

    report = density_report(graph)
    assert report.edges == 5 and report.density == pytest.approx(5 / 6)
    assert "self loop(s) are left out" not in density_note(nx.cycle_graph(4))


def test_the_summary_carries_transitivity_beside_the_weighted_average() -> None:
    """Three clustering numbers, three definitions; the table has to name which it printed."""
    graph, _ = karate_club()
    stats = summary(graph)
    assert stats["transitivity"] == pytest.approx(0.2557, abs=5e-5)
    # ``average_clustering`` in the table is Onnela's *weighted* mean, which on a weighted graph
    # is neither of the two published numbers -- hence the docstring and the report's third row.
    assert stats["average_clustering"] == pytest.approx(
        nx.average_clustering(graph, weight="weight")
    )
    assert stats["average_clustering"] != pytest.approx(stats["transitivity"])
    assert summary(nx.Graph())["transitivity"] == 0.0


def test_local_clustering_summary_describes_the_distribution_behind_the_mean() -> None:
    """A mean of local coefficients is a mean (§3.1), so the report prints its spread too."""
    graph, _ = karate_club()
    described = local_clustering_summary(graph)
    assert described.count == 34
    assert described.mean == pytest.approx(0.5706, abs=5e-5)
    assert described.minimum == 0.0 and described.maximum == pytest.approx(1.0)
    assert described.median == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least one node"):
        local_clustering_summary(nx.Graph())


def test_unknown_clustering_kinds_are_refused() -> None:
    with pytest.raises(ValueError, match="clustering kind must be one of"):
        clustering(nx.cycle_graph(4), "transitive")
    assert set(CLUSTERING_KINDS) == {"local", "average", "global", "weighted"}


def test_chapter_twelve_flattens_a_directed_network_and_says_so(messages: nx.DiGraph) -> None:
    """networkx reads a directed triangle as transitivity 0.0, which is worse than refusing.

    Clustering, cliques and independent sets are all defined on undirected graphs (§12.1), so
    they run on the flattened view and the section prints the §6.2 note next to them rather
    than publishing a silent zero. The density is the exception and keeps its directed
    denominator; the next test holds the report to saying which is which.
    """
    triangle = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    nx.set_edge_attributes(triangle, 1, "weight")
    assert nx.transitivity(triangle) == 0.0  # the trap
    assert clustering(triangle, "global") == pytest.approx(1.0)
    assert summary(triangle)["transitivity"] == pytest.approx(1.0)

    report = density_report(messages)
    assert report.flattened == FLATTENED
    section = "\n".join(render_density(report))
    assert "flattened undirected view" in section
    assert report.nodes == 5 and report.edges == 7  # the directed counts, as the table has them
    assert cliques(messages).largest_size == 2  # no directed cycle closes an undirected triangle


def test_the_density_of_a_directed_network_is_the_directed_one_and_the_section_says_so(
    messages: nx.DiGraph,
) -> None:
    """One number in the section is not flattened, and the claim printed has to match.

    §12.1 gives a directed network |V|(|V|-1) possible edges, twice the undirected count,
    because (u, v) and (v, u) are two edges. Flattening first would halve the denominator and
    describe a network nobody built, so the density stays directed: 7 of the 20 ordered pairs
    5 nodes could carry. The flattened copy has 5 edges of a possible 10 -- the same 0.5 here,
    which is a coincidence of this graph, so the test pins the counts and not only the ratio.
    """
    report = density_report(messages)
    assert report.edges == 7 and report.density == pytest.approx(7 / 20)
    assert "7 of the 20 edges 5 nodes could carry (|V|(|V|-1))" in report.density_note

    section = "\n".join(render_density(report))
    assert "**The density is not**" in section
    assert "The clustering coefficients, the cliques and the independent set above" in section
    # And the weighted row must stop claiming to be the summary table's number, which on a
    # directed network is Fagiolo's rather than Onnela-on-the-flattened-view.
    assert "The summary table's weighted average clustering is **not** this number" in section
    assert summary(messages)["average_clustering"] == pytest.approx(
        nx.average_clustering(messages, weight="weight")
    )
    undirected_section = "\n".join(render_density(density_report(nx.karate_club_graph())))
    assert "This is the number the summary table above calls weighted average clustering" in (
        undirected_section
    )


def test_a_two_mode_network_says_its_zeros_are_structural() -> None:
    """A two-mode network has no triangle and no clique above an edge, by construction (§6.4).

    Planted: three speakers, three entities, every speaker joined to every entity. Every
    clustering coefficient is 0 and every maximal clique is one of the nine edges -- neither is
    a finding about the corpus, and a report that prints them as findings is wrong. §12.3's
    object for this case is the biclique (this graph is one 3,3-clique), which is not computed.
    Projecting is the remedy: the three speakers all share entities, so the projection is a
    triangle and its transitivity is 1.
    """
    graph = nx.complete_bipartite_graph(3, 3)
    for node in graph:
        graph.nodes[node]["mode"] = "speaker" if node < 3 else "entity"
    nx.set_edge_attributes(graph, 1, "weight")
    assert describe_network(graph).bipartite

    report = density_report(graph)
    assert report.two_mode
    assert report.average_clustering == 0.0 and report.global_clustering == 0.0
    assert "by construction, not as a finding" in report.clustering_note
    assert "--project" in report.clustering_note

    assert report.cliques.largest_size == 2 and report.cliques.sizes == {2: 9}
    section = "\n".join(render_density(report))
    assert "biclique" in section and "3,3" not in section  # the object is named, not computed
    assert "no three nodes of a two-mode network are mutually adjacent" in section

    projected = nx.projected_graph(graph, [0, 1, 2])
    assert clustering(projected, "global") == pytest.approx(1.0)
    assert not density_report(nx.karate_club_graph()).two_mode


def test_the_density_section_reaches_the_report_with_its_frame_and_its_n() -> None:
    """Every number in the section is printed beside n, the frame, the null and the chapter."""
    from graphrag.graph.memory_store import InMemoryGraphStore
    from graphrag.sna.analysis import render_markdown, run_analysis, to_payload

    graph, known = karate_club()
    analysis = run_analysis(
        InMemoryGraphStore(),
        graph,
        persona_id="test-pm",
        network="speakers",
        method="louvain",
        seed=1,
        runs=3,
        samples=5,
    )
    report = render_markdown(analysis)
    assert "## Density" in report
    assert "Atlas ch. 12, over the whole network: n = 34 nodes and 78 edges." in report
    assert known.frame in report
    assert "No null model" in report
    assert "only readable against n (§12.1)" in report
    assert "**Average clustering** 0.5706" in report and "0.2557 (transitivity)" in report
    assert "the largest maximal clique has 5 nodes" in report

    payload = to_payload(analysis)["density"]
    assert payload["chapter"] == "Atlas ch. 12"
    assert payload["nodes"] == 34 and payload["cliques"]["largest_size"] == 5
    assert payload["global_clustering"] == pytest.approx(0.2557, abs=5e-5)
    assert payload["independent_set"]["size"] == len(analysis.density.independent_set)  # type: ignore[union-attr]


def test_an_empty_network_has_no_density_section() -> None:
    """Nothing in chapter 12 is defined over no nodes, so the section is absent, not zeroed."""
    from graphrag.graph.memory_store import InMemoryGraphStore
    from graphrag.sna.analysis import render_markdown, run_analysis

    analysis = run_analysis(
        InMemoryGraphStore(),
        nx.Graph(),
        persona_id="test-pm",
        network="speakers",
        method="louvain",
        seed=1,
    )
    assert analysis.density is None
    assert "## Density" not in render_markdown(analysis)
