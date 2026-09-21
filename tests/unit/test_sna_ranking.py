"""Chapter 14: the rest of the centrality battery, Freeman's centralization, and rank stability.

Every number asserted here is either published (the Medici lead every centrality table ever
printed of the Florentine marriage network; the karate club's core decomposition is what
``networkx``'s own O(m) implementation returns and is stable across both circulating copies of
the graph) or known before the code runs, because the graph was planted to have it: a star is
the most centralized network there is, a cycle is the least, and a directed graph built with one
node everybody points at has one authority.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graphrag.sna.analysis import (
    RANKING_SAMPLES,
    STABLE_SHARE,
    ranking_report,
    ranking_stability,
    render_ranking,
)
from graphrag.sna.measures import (
    CENTRALITIES,
    DIRECTED_CENTRALITIES,
    STAR_SPREAD,
    Centralization,
    centrality,
    centrality_meaning,
    centralization,
    centralization_table,
    core_shells,
    coreness,
    harmonic,
    hits,
    k_truss,
    reach,
    star_spread,
    top_n,
    truss_number,
)
from tests.legendary import florentine_families, karate_club


def _weighted(graph: nx.Graph, weight: float = 1.0) -> nx.Graph:
    """The same graph with an explicit weight on every edge, as the builders here produce."""
    copy = graph.copy()
    for u, v in copy.edges():
        copy[u][v]["weight"] = weight
    return copy


@pytest.fixture
def star() -> nx.Graph:
    """Seven nodes around one: §14.8's maximum, and the network every centralization divides by."""
    return _weighted(nx.star_graph(6))


@pytest.fixture
def cycle() -> nx.Graph:
    """Seven nodes in a ring: every node identical, so no centrality has anything to rank."""
    return _weighted(nx.cycle_graph(7))


@pytest.fixture
def hub_and_authority() -> nx.DiGraph:
    """Three hubs pointing at two authorities, and one stray voter for the first of them.

    The answer is planted: ``a1`` is pointed at by all three hubs *and* by ``x``, ``a2`` only by
    the three, so ``a1`` is the authority; the three hubs point at both authorities and ``x``
    points at one, so ``x`` must be the weaker hub. §14.5's recursion -- "a good hub is a hub
    that points to good authorities" -- is exactly what separates them.
    """
    graph = nx.DiGraph()
    for hub in ("h1", "h2", "h3"):
        for authority in ("a1", "a2"):
            graph.add_edge(hub, authority, weight=1.0)
    graph.add_edge("x", "a1", weight=1.0)
    return graph


# ------------------------------------------------------------------ the rankings themselves


def test_the_medici_lead_every_ranking_the_literature_ranks_them_first() -> None:
    """§53.4's standard example: the Medici top every centrality on the marriage network.

    The harmonic ranking is the new one here and it agrees, which is the check that matters:
    a measure that disagreed with closeness on a *connected* network would be wrong, since the
    only thing §14.6 changes is how unreachable pairs are counted and there are none here.
    """
    graph, known = florentine_families()
    assert known.top_degree == "Medici" and known.top_betweenness == "Medici"
    for kind in ("degree", "betweenness", "closeness", "eigenvector", "harmonic"):
        assert top_n(centrality(graph, kind), 1)[0][0] == "Medici", kind
    # And harmonic is the chapter's raw sum, not an average: 9.5 over the 14 other families.
    assert harmonic(graph)["Medici"] == pytest.approx(9.5)


def test_reach_within_two_hops_covers_a_star_and_says_nothing(star: nx.Graph) -> None:
    """§14.3's own objection: on one connected component every node reaches everything.

    A leaf of a star is one hop from the centre and two from every other leaf, so two hops cover
    the network from anywhere in it and the ranking is a column of ones -- which is why the
    chapter says reach "doesn't make much sense for undirected networks", and why §14.8's
    centralization for it is undefined here.
    """
    assert set(reach(star).values()) == {1.0}
    assert set(reach(star, hops=None).values()) == {1.0}
    # One hop is a different question, and the centre is the only node that answers it.
    one_hop = reach(star, hops=1)
    assert one_hop[0] == 1.0
    assert one_hop[1] == pytest.approx(1 / 6)


def test_reach_follows_the_edges_the_way_they_point(hub_and_authority: nx.DiGraph) -> None:
    """The chapter's wheel: a sink reaches nothing, whatever points at it."""
    scores = reach(hub_and_authority)
    assert scores["a1"] == 0.0 and scores["a2"] == 0.0
    assert scores["h1"] == pytest.approx(2 / 5)
    assert scores["x"] == pytest.approx(1 / 5)


def test_reach_refuses_a_radius_of_zero(star: nx.Graph) -> None:
    with pytest.raises(ValueError, match="at least one hop"):
        reach(star, hops=0)


def test_harmonic_is_finite_where_closeness_needs_the_caveat() -> None:
    """§14.6's whole point: 1/infinity = 0, so a network in pieces still has one ranking.

    Two disjoint edges and an isolate. Closeness is not comparable across the components -- in
    networkx's wf-improved form every node of both components scores the same 0.25, which says
    only "one neighbour out of four possible", not which of them is better placed -- while
    harmonic gives each connected node its single reachable neighbour, 1.0, and the isolate 0.0
    without being asked to divide by an infinity.
    """
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("c", "d", weight=1.0)
    graph.add_node("e")
    scores = harmonic(graph)
    assert scores == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 0.0}
    assert all(
        value == pytest.approx(0.25)
        for node, value in centrality(graph, "closeness").items()
        if node != "e"
    )
    assert centrality(graph, "closeness")["e"] == 0.0


def test_hits_finds_the_planted_hub_and_authority(hub_and_authority: nx.DiGraph) -> None:
    """§14.5 solved as the two eigenvectors, normalised the way Figure 14.11 normalises them."""
    hubs, authorities = hits(hub_and_authority)
    assert max(hubs.values()) == pytest.approx(1.0)
    assert max(authorities.values()) == pytest.approx(1.0)
    # The authority is the node the good hubs agree on; the stray voter is the weaker hub.
    assert authorities["a1"] > authorities["a2"] > 0.0
    assert hubs["h1"] > hubs["x"] > 0.0
    # An authority is not a hub and a hub is not an authority, on a graph with no reciprocity.
    assert authorities["h1"] == 0.0 and hubs["a1"] == 0.0
    # And the dispatch reaches the same two vectors by name.
    assert centrality(hub_and_authority, "hits_authority") == authorities
    assert centrality(hub_and_authority, "hits_hub") == hubs


def test_hits_is_refused_on_an_undirected_network(star: nx.Graph) -> None:
    """The §14.5 degeneracy: A^T A and A A^T are the same matrix, so there is one vector."""
    with pytest.raises(ValueError, match="only on a directed network"):
        hits(star)
    # Asked for by name through the battery, it refuses with the same §14.5 reason rather than
    # with "unknown centrality", which would send a reader looking for a spelling mistake.
    with pytest.raises(ValueError, match="Use the eigenvector centrality"):
        centrality(star, "hits_hub")


def test_the_karate_clubs_core_decomposition_is_four_deep() -> None:
    """§14.7 on the graph §53.4 calls the first one everything is tried on.

    The peel runs 1, 2, 3, 4 and stops: ten of the 34 members survive into the 4-core, and they
    are the instructor, the officer and the people who met both sides. The shells are the reason
    the report prints them -- four distinct values over 34 nodes is not a ranking.
    """
    graph, _ = karate_club()
    numbers = coreness(graph)
    assert max(numbers.values()) == 4.0
    assert {node for node, value in numbers.items() if value == 4.0} == {
        0,
        1,
        2,
        3,
        7,
        8,
        13,
        30,
        32,
        33,
    }
    assert core_shells(graph) == {1: 1, 2: 11, 3: 12, 4: 10}
    # Weights are ignored, because §14.7 counts connections and not their strength.
    plain = nx.karate_club_graph()
    for u, v in plain.edges():
        plain[u][v]["weight"] = 9.0
    assert coreness(plain) == numbers


def test_the_truss_is_the_edge_side_of_the_core_and_agrees_with_networkx() -> None:
    """Cross-checked against ``nx.k_truss``, and against the k-core it must sit inside."""
    graph, _ = karate_club()
    numbers = truss_number(graph)
    assert max(numbers.values()) == 5
    assert (
        k_truss(graph, 4).number_of_edges()
        == nx.k_truss(nx.karate_club_graph(), 4).number_of_edges()
    )
    for k in (2, 3, 4, 5, 6):
        truss = k_truss(graph, k)
        expected = {tuple(sorted(edge)) for edge in nx.k_truss(nx.karate_club_graph(), k).edges()}
        assert {tuple(sorted(edge)) for edge in truss.edges()} == expected
        assert {edge for edge, value in numbers.items() if value >= k} == expected
        # Every k-truss is inside the (k-1)-core: an edge in k-2 triangles needs k-1 neighbours.
        cores = coreness(graph)
        assert all(cores[node] >= k - 1 for node in truss.nodes)
    assert k_truss(graph, 6).number_of_edges() == 0


def test_a_truss_below_two_is_refused() -> None:
    graph, _ = karate_club()
    with pytest.raises(ValueError, match="at least 2"):
        k_truss(graph, 1)


def test_the_new_rankings_are_in_the_battery_the_report_prints() -> None:
    """A measure nobody runs is not built: these have to be in the lists ``analyze`` walks."""
    assert {"harmonic", "reach", "coreness"} <= set(CENTRALITIES)
    assert {"harmonic", "reach", "coreness", "hits_hub", "hits_authority"} <= set(
        DIRECTED_CENTRALITIES
    )
    assert "hits_hub" not in CENTRALITIES and "hits_authority" not in CENTRALITIES


def test_every_ranking_carries_the_caption_the_report_prints_under_it() -> None:
    """A ranking without a caption is a column of numbers, and would crash the renderer."""
    undirected, directed = nx.Graph(), nx.DiGraph()
    for kind in CENTRALITIES:
        assert centrality_meaning(kind, undirected), kind
    for kind in DIRECTED_CENTRALITIES:
        assert centrality_meaning(kind, directed), kind
    # Three of them mean something else once the edges point, and say so rather than keeping the
    # undirected line: harmonic and closeness become incoming, reach is the only outgoing one.
    for kind in ("harmonic", "reach", "coreness"):
        assert centrality_meaning(kind, directed) != centrality_meaning(kind, undirected), kind
    assert "reads *out* of a node" in centrality_meaning("reach", directed)


# --------------------------------------------------------------------- centralization (§14.8)


def test_a_star_is_fully_centralized_and_a_cycle_is_not_at_all(
    star: nx.Graph, cycle: nx.Graph
) -> None:
    """The two ends of §14.8's scale, on the two graphs that define them.

    "A network cannot get more centralized than that": the star is the denominator, so its own
    ratio is 1 by construction for every measure. A cycle gives every node the same score by
    symmetry, so the numerator is 0 and no measure can find a centre.
    """
    for kind in ("degree", "closeness", "betweenness", "harmonic", "eigenvector", "pagerank"):
        assert centralization(star, kind) == pytest.approx(1.0), kind
        assert centralization(cycle, kind) == pytest.approx(0.0, abs=1e-9), kind


def test_the_closed_form_star_denominators_match_a_computed_star() -> None:
    """Every denominator read from a table has to equal the one read off a star that was built.

    Four of them exist because a closed form is in print for this package's normalisation; the
    other two, reach and coreness, exist because computing them would be quadratic on the large
    networks that still ask for them. Both kinds are checked here against the real thing.
    """
    for nodes in (3, 5, 9, 15, 34):
        built = _weighted(nx.star_graph(nodes - 1))
        for kind in (*STAR_SPREAD, "reach", "coreness"):
            scores = centrality(built, kind)
            peak = max(scores.values())
            computed = sum(peak - value for value in scores.values())
            assert star_spread(nodes, kind) == pytest.approx(computed), (nodes, kind)
        # Directed, §14.8's "largest theoretical sum" is the best-oriented star, which for reach
        # is the out-star: its centre commands everything and its leaves command nothing.
        out_star = nx.DiGraph((0, leaf, {"weight": 1.0}) for leaf in range(1, nodes))
        assert star_spread(nodes, "reach", directed=True) == pytest.approx(
            _spread_of(centrality(out_star, "reach"))
        )


def _spread_of(scores: dict[str, float]) -> float:
    """§14.8's numerator, written out here so the test does not borrow the implementation."""
    peak = max(scores.values())
    return sum(peak - value for value in scores.values())


def test_centralization_is_undefined_where_a_star_is_not_the_maximum(star: nx.Graph) -> None:
    """Two rankings a star cannot maximise, and the report says so instead of dividing by zero."""
    for kind in ("coreness", "reach"):
        with pytest.raises(ValueError, match="denominator"):
            centralization(star, kind)
    with pytest.raises(ValueError, match="at least 3 nodes"):
        centralization(nx.path_graph(2), "degree")


def test_the_eigenvector_centralization_is_refused_on_a_network_in_pieces() -> None:
    """§14.8's "usually a star" fails for the eigenvector as soon as the network fragments.

    The principal eigenvector of a disconnected adjacency matrix lives on one component, and
    normalising it to unit length sends every node outside that component to ~0. The observed
    sum of differences then *exceeds* the star's -- this graph reads 5.8357 against a star's
    5.4330, a ratio of 1.07 -- which would print as a network more centralized than a star.
    Every other ranking in the battery stays inside [0, 1] on the same graph, so the refusal is
    scoped to the one measure that breaks.
    """
    graph = nx.gnp_random_graph(12, 0.2, seed=0)
    for u, v in graph.edges():
        graph[u][v]["weight"] = 1.0
    assert nx.number_connected_components(graph) == 6
    # The number the refusal exists to suppress, computed here rather than taken on trust:
    spread = _spread_of(centrality(graph, "eigenvector"))
    star = star_spread(graph.number_of_nodes(), "eigenvector")
    assert spread == pytest.approx(5.8357, abs=5e-5)
    assert star == pytest.approx(5.4330, abs=5e-5)
    assert spread / star == pytest.approx(1.0741, abs=5e-5)

    with pytest.raises(ValueError, match="is in 6 pieces"):
        centralization(graph, "eigenvector")
    row = {item.kind: item for item in centralization_table(graph)}["eigenvector"]
    assert row.value is None
    assert "PageRank's teleportation" in row.note  # §14.4's own answer to the same problem
    # Every other row on this fragmented graph is a ratio, and a ratio inside the scale.
    for kind, other in row_values(centralization_table(graph)).items():
        assert 0.0 <= other <= 1.0, kind

    # Connected, it is defined again and behaves: a star still reads exactly 1.0.
    connected = nx.path_graph(6)
    for u, v in connected.edges():
        connected[u][v]["weight"] = 1.0
    assert 0.0 <= centralization(connected, "eigenvector") <= 1.0


def row_values(rows: list[Centralization]) -> dict[str, float]:
    """The rows that carry a ratio, by kind, so a test can assert over all of them at once."""
    return {row.kind: row.value for row in rows if row.value is not None}


def test_centralization_does_not_move_when_the_weights_are_rescaled() -> None:
    """Why it is computed on the binary view: a star has no weights to be compared with.

    Doubling every weight halves every distance and doubles every closeness, so a centralization
    taken on the weighted graph would double while the shape of the network stood still.
    """
    graph = _weighted(nx.barbell_graph(4, 2), weight=1.0)
    heavy = _weighted(nx.barbell_graph(4, 2), weight=7.0)
    for kind in ("degree", "closeness", "betweenness", "harmonic", "eigenvector"):
        assert centralization(graph, kind) == pytest.approx(centralization(heavy, kind)), kind


def test_the_karate_club_is_about_forty_percent_centralized() -> None:
    """Freeman's own formula, computed by hand beside the function's answer.

    sum(d_max - d_v) / ((|V|-1)(|V|-2)) over the raw degrees is the closed form for degree
    centralization; it has to agree with the ratio taken over normalised degrees, because
    dividing both sums by the same n-1 cannot change a ratio.
    """
    graph, _ = karate_club()
    degrees = [degree for _, degree in graph.degree()]
    peak, nodes = max(degrees), graph.number_of_nodes()
    by_hand = sum(peak - degree for degree in degrees) / ((nodes - 1) * (nodes - 2))
    assert centralization(graph, "degree") == pytest.approx(by_hand)
    assert by_hand == pytest.approx(0.3996, abs=5e-5)


def test_the_centralization_table_covers_every_ranking_and_says_why_two_are_missing() -> None:
    graph, _ = karate_club()
    rows = {row.kind: row for row in centralization_table(graph)}
    assert set(rows) == set(CENTRALITIES)
    assert rows["coreness"].value is None and "denominator" in rows["coreness"].note
    assert rows["reach"].value is None
    # The weighted degree is the degree on the binary view, and the row says so rather than
    # letting a reader take it for a second opinion.
    assert rows["weighted_degree"].value == pytest.approx(rows["degree"].value or 0.0)
    assert "the weighted degree is the degree" in rows["weighted_degree"].note


# ------------------------------------------------------------------------- ranking stability


def test_the_karate_leaders_survive_the_bootstrap() -> None:
    """Nodes 33 and 0 are the officer and the instructor: no draw of this club loses them.

    Fifty edge-sampled resamples, each holding 80% of the nodes; both leaders stay in the top two
    of every one of them, which is the property a report needs before it prints a name.
    """
    graph, known = karate_club()
    assert known.top_degree == 33
    result = ranking_stability(graph, "degree", samples=50, seed=7, top_k=2)
    assert result.samples == 50
    assert result.retention[33] >= STABLE_SHARE
    assert result.retention[0] >= STABLE_SHARE
    assert result.stable == [33, 0] and result.unstable == []
    assert result.spearman_mean > 0.9
    assert result.spearman_low <= result.spearman_mean <= result.spearman_high
    # Kendall's tau runs smaller than Spearman on the same data (§3.4) and is not the same scale.
    assert result.kendall_mean < result.spearman_mean
    assert "§29.1's edge sampling" in result.note


def test_the_tail_of_the_same_ranking_is_not_stable() -> None:
    """The reason the section exists: the leaders hold and the tail is a coin flip."""
    graph, _ = karate_club()
    result = ranking_stability(graph, "degree", samples=30, seed=3, top_k=20)
    assert result.retention[33] == 1.0
    assert result.unstable, "a 20-deep ranking over 34 nodes cannot be stable all the way down"
    assert min(result.retention.values()) < STABLE_SHARE


def test_the_degree_preserving_null_holds_the_degree_ranking_fixed() -> None:
    """A caveat the report has to print, not a bug: §19.1's swap preserves every degree.

    So the degree ranking correlates with itself at exactly 1.0 over any number of rewirings,
    and this resampling only says something about the path- and walk-based rankings.
    """
    graph, _ = karate_club()
    result = ranking_stability(graph, "degree", samples=5, seed=11, method="configuration")
    assert result.samples == 5
    assert result.spearman_mean == pytest.approx(1.0)
    assert "by construction" in result.note
    walked = ranking_stability(graph, "betweenness", samples=5, seed=11, method="configuration")
    assert walked.spearman_mean < 1.0


def test_a_backbone_comparison_runs_once_because_it_is_deterministic() -> None:
    graph, _ = karate_club()
    result = ranking_stability(graph, "betweenness", samples=9, seed=1, method="backbone")
    assert result.requested == 1 and result.samples == 1
    assert result.spearman_low == result.spearman_high == result.spearman_mean
    assert "deterministic" in result.note


def test_ranking_stability_refuses_what_it_cannot_resample() -> None:
    graph, _ = karate_club()
    with pytest.raises(ValueError, match="method must be one of"):
        ranking_stability(graph, "degree", method="jackknife")
    with pytest.raises(ValueError, match="top_k must be at least 1"):
        ranking_stability(graph, "degree", top_k=0)


def test_the_section_prints_its_frame_its_n_its_null_and_its_chapter() -> None:
    """Every report section carries the four things a reader needs to check it."""
    graph, known = karate_club()
    report = ranking_report(graph, samples=5, seed=4)
    text = "\n".join(render_ranking(report))
    assert "## Ranking stability" in text
    assert "Atlas ch. 14" in text
    assert f"n = {known.nodes:,} nodes and {known.edges:,} edges" in text
    assert known.frame in text
    assert "5 x `bootstrap-edges`" in text
    assert "Seed: 4." in text
    assert "### Centralization (§14.8)" in text
    assert "1.0 is a star" in text
    assert "Shells (§14.7)" in text
    for kind in CENTRALITIES:
        assert f"| {kind} |" in text


def test_the_report_never_takes_more_resamples_than_its_budget() -> None:
    """``--samples 50`` is the null model's budget; this section recomputes the whole battery."""
    graph, _ = karate_club()
    report = ranking_report(graph, samples=500, seed=2)
    assert report.samples == RANKING_SAMPLES
    assert all(row.samples <= RANKING_SAMPLES for row in report.stability)
    assert report.skipped == []
    assert "inside the 500-node budget" in report.note


def test_a_network_with_no_edges_says_so_rather_than_reporting_zeros() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    result = ranking_stability(graph, "degree", samples=4, seed=1)
    assert result.samples == 0
    assert "No usable resample" in result.note
