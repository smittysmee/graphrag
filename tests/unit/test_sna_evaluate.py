"""Chapter 36's battery, held to answers that existed before the code did.

Every number asserted here comes from one of four places, and none of them is "the function
returned something":

* **The book's own worked examples.** §36.2 computes four measures by hand for the community in
  its Figure 36.9 -- conductance 12/(2x13 + 12), Maximum-ODF 2/3, Average-ODF 0.319, Flake-ODF
  2/7 -- and two conductances for Figure 36.7, 4/((2x21) + 4) and 7/((2x10) + 7). Both figures
  are rebuilt here from the degrees the text gives and every number is asserted to the digit.
* **A legendary graph with a published ground truth.** The karate club partitioned by Zachary's
  two factions: modularity 0.3582, and the best two-way split at 0.3718 differing from the
  factions by exactly two traded members, one of them node 8 -- the member Zachary explains
  himself, and the reason a higher modularity is not a better recovery of the truth.
* **A planted graph whose answer is known before it is built.** k equal disconnected cliques
  reach modularity 1 - 1/k with conductance 0 and internal density 1, and a 4 x 30 planted
  partition is recovered at NMI above 0.9.
* **An arithmetic identity.** A partition whose labels were dealt out at random predicts no link
  better than chance, so its AUC sits at 0.5 and its modularity at zero.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import render_markdown, run_analysis, to_payload
from graphrag.sna.cluster import louvain
from graphrag.sna.evaluate import (
    MEASURES,
    adjusted_mutual_information,
    evaluate_partition,
    evaluation_payload,
    link_prediction_score,
    render_evaluation,
    resolution_limit,
)
from graphrag.sna.generators import planted_partition
from tests.legendary import karate_club, unweighted


@pytest.fixture
def karate() -> nx.Graph:
    """The 78-edge karate club without its weights: the copy the published numbers belong to."""
    graph, _ = karate_club()
    plain = unweighted(graph)
    plain.graph["frame"] = "34 members of one university karate club, 1970-1972."
    return plain


def _factions(graph: nx.Graph) -> list[list[int]]:
    """Zachary's two clubs, recorded independently of the edges."""
    return [
        sorted(n for n, data in graph.nodes(data=True) if data["club"] == club)
        for club in ("Mr. Hi", "Officer")
    ]


# ------------------------------------------------------------------ §36.2, the book's own hands


def _figure_36_9() -> tuple[nx.Graph, list[list[str]]]:
    """Figure 36.9's community, rebuilt from the degrees p. 523 quotes for it.

    The text gives the out-degree fraction of each of the seven nodes clockwise from node 1:
    ``2/3, 3/5, 0, 4/10, 1/5, 1/5, 1/6``. That fixes both the degree of every node and how many
    of its edges leave the community, and therefore the internal degree sequence
    ``(1, 2, 4, 6, 4, 4, 5)`` -- 26 endpoints, so the 13 internal edges the text also quotes --
    and the 12 boundary edges. Any graph with those numbers gives the book's four values, so the
    internal structure is Havel-Hakimi's and every boundary edge goes to a node of its own.
    """
    inside = [6, 5, 4, 4, 4, 2, 1]
    leaving = [4, 1, 0, 1, 1, 3, 2]
    community = nx.havel_hakimi_graph(inside)
    graph = nx.relabel_nodes(community, {node: f"c{node}" for node in community})
    outside = 0
    for node, out in zip(sorted(graph, key=lambda n: -graph.degree(n)), leaving, strict=True):
        for _ in range(out):
            graph.add_edge(node, f"x{outside}")
            outside += 1
    members = sorted(n for n in graph if n.startswith("c"))
    return graph, [members, *([n] for n in sorted(graph) if n.startswith("x"))]


def test_figure_36_9_reproduces_all_four_of_the_books_hand_computed_values() -> None:
    """p. 523 works conductance and the three ODF variants by hand for one community."""
    graph, partition = _figure_36_9()
    scores = evaluate_partition(graph, partition, link_prediction=False)
    row = scores.per_community[0]
    assert row.size == 7
    assert (row.internal_edges, row.boundary_edges) == (13, 12)
    # "conductance -- which is 12/(2 x (13) + 12) ~ 0.316"
    assert row.conductance == pytest.approx(12 / 38)
    # "Thus, the Maximum-ODF is f(C) = 2/3"
    assert row.max_odf == pytest.approx(2 / 3)
    # "f(C) = (2/3 + 3/5 + 0 + 4/10 + 1/5 + 1/5 + 1/6)/7 ~ 0.319"
    assert row.average_odf == pytest.approx(
        (2 / 3 + 3 / 5 + 0 + 4 / 10 + 1 / 5 + 1 / 5 + 1 / 6) / 7
    )
    assert row.average_odf == pytest.approx(0.319, abs=5e-4)
    # "we only have two nodes with more links going outside the community than inside"
    assert row.flake_odf == pytest.approx(2 / 7)
    # and the average ODF is "awfully close, but not quite" the conductance
    assert row.average_odf != pytest.approx(row.conductance)
    assert abs(row.average_odf - row.conductance) < 0.005


@pytest.mark.parametrize(
    ("clique", "boundary", "expected"),
    [(7, 4, 4 / 46), (5, 7, 7 / 27)],
    ids=["figure-36.7a", "figure-36.7b"],
)
def test_figure_36_7_conductances(clique: int, boundary: int, expected: float) -> None:
    """Both cliques of p. 519, whose point is that conductance ignores internal density.

    *"Here, both communities are cliques, the denset possible structure. But, since the one in
    Figure 36.7(b) also has a lot of connections to the rest of the network, the resulting
    conductance is almost three times higher."*
    """
    graph = nx.relabel_nodes(nx.complete_graph(clique), {n: f"c{n}" for n in range(clique)})
    members = sorted(graph)
    for i in range(boundary):
        graph.add_edge(members[i % clique], f"x{i}")
    partition = [members, *([n] for n in sorted(graph) if n.startswith("x"))]
    scores = evaluate_partition(graph, partition, link_prediction=False)
    row = scores.per_community[0]
    assert row.internal_edges == clique * (clique - 1) // 2
    assert row.boundary_edges == boundary
    assert row.conductance == pytest.approx(expected)
    assert row.internal_density == pytest.approx(1.0)  # both are cliques and it cannot tell them
    assert row.triangle_participation == pytest.approx(1.0)


# ------------------------------------------------------------------ §36.1, the karate club


def test_the_karate_factions_score_the_modularity_they_are_published_with(
    karate: nx.Graph,
) -> None:
    """The ground-truth split, and the conductance of each faction, pinned to the digit."""
    scores = evaluate_partition(karate, _factions(karate), link_prediction=False)
    assert (scores.nodes, scores.edges, scores.communities) == (34, 78, 2)
    assert scores.modularity == pytest.approx(0.3582, abs=5e-5)
    assert not scores.weighted  # the published numbers are for the unweighted copy
    assert scores.max_modularity == pytest.approx(0.5)  # 1 - 1/2, what two cliques would reach

    mr_hi, officer = scores.per_community
    assert (mr_hi.internal_edges, mr_hi.boundary_edges) == (35, 11)
    assert (officer.internal_edges, officer.boundary_edges) == (32, 11)
    assert mr_hi.conductance == pytest.approx(11 / (2 * 35 + 11))
    assert officer.conductance == pytest.approx(11 / (2 * 32 + 11))
    assert mr_hi.conductance == pytest.approx(0.1358, abs=5e-5)
    assert officer.conductance == pytest.approx(0.1467, abs=5e-5)
    # coverage counts edges, so the 11 edges that cross the split are the whole of the gap
    assert scores.averages["coverage"] == pytest.approx(67 / 78)


def test_the_best_two_way_split_is_the_factions_with_two_members_traded(karate: nx.Graph) -> None:
    """0.3718 against the factions' 0.3582, and the gap is two members, one of them node 8.

    The highest modularity a two-way split of this graph reaches is 0.3718, and the partition
    that reaches it is the two factions with nodes 8 and 9 traded. Zachary explains node 8
    himself -- he sparred with the officers and joined the instructor's club because of a black
    belt test three weeks away -- so this is the difference between a ground truth and a method's
    answer, in two nodes, and the reason a higher modularity is not a better recovery of the
    truth.
    """
    factions = [set(faction) for faction in _factions(karate)]
    traded = [sorted((factions[0] - {8}) | {9}), sorted((factions[1] - {9}) | {8})]
    best = evaluate_partition(karate, traded, link_prediction=False)
    assert best.modularity == pytest.approx(0.3718, abs=5e-5)
    assert (
        best.modularity
        > evaluate_partition(
            karate, [sorted(f) for f in factions], link_prediction=False
        ).modularity
    )
    greedy = nx.community.greedy_modularity_communities(karate, cutoff=2, best_n=2)
    assert sorted(sorted(c) for c in greedy) == sorted(traded)
    assert set(traded[0]) ^ factions[0] == {8, 9}


def test_louvain_on_the_karate_club_agrees_with_the_factions_only_in_part(
    karate: nx.Graph,
) -> None:
    """§36.4 over a real partition: the agreement is moderate, and the adjusted indices are lower.

    Louvain finds four communities where the ground truth has two, so this is the chapter's own
    warning in one assertion: the NMI is inflated by the extra groups and the chance-corrected
    numbers sit below it.
    """
    found = louvain(karate, seed=11, runs=5)
    truth = {node: data["club"] for node, data in karate.nodes(data=True)}
    scores = evaluate_partition(karate, found.communities, truth=truth, link_prediction=False)
    assert scores.truth is not None
    assert scores.truth.truth_communities == 2
    assert scores.truth.found_communities == len(found.communities) >= 3
    assert 0.4 < scores.truth.nmi < 0.7
    assert scores.truth.adjusted_rand_index < scores.truth.nmi
    assert scores.truth.variation_of_information > 0.0


def test_the_factions_compared_with_themselves_agree_perfectly(karate: nx.Graph) -> None:
    """The identity case of §36.4: every index at its ceiling and VI at zero bits."""
    factions = _factions(karate)
    truth = {node: data["club"] for node, data in karate.nodes(data=True)}
    scores = evaluate_partition(karate, factions, truth=truth, link_prediction=False)
    assert scores.truth is not None
    assert scores.truth.nmi == pytest.approx(1.0)
    assert scores.truth.adjusted_mutual_information == pytest.approx(1.0)
    assert scores.truth.adjusted_rand_index == pytest.approx(1.0)
    assert scores.truth.variation_of_information == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ planted answers


@pytest.mark.parametrize("k", [2, 3, 5])
def test_k_equal_cliques_reach_the_maximum_modularity_of_one_minus_one_over_k(k: int) -> None:
    """The ceiling: k equal, mutually disconnected cliques, with nothing to conduct out of."""
    graph = nx.disjoint_union_all([nx.complete_graph(6) for _ in range(k)])
    graph = nx.relabel_nodes(graph, {n: f"n{n:02d}" for n in graph})
    partition = [sorted(component) for component in nx.connected_components(graph)]
    scores = evaluate_partition(graph, partition, link_prediction=False)
    assert scores.modularity == pytest.approx(1.0 - 1.0 / k)
    assert scores.modularity == pytest.approx(scores.max_modularity)
    for row in scores.per_community:
        assert row.boundary_edges == 0
        assert row.conductance == pytest.approx(0.0)
        assert row.internal_density == pytest.approx(1.0)
        assert row.max_odf == pytest.approx(0.0)
        assert row.triangle_participation == pytest.approx(1.0)
    assert scores.averages["coverage"] == pytest.approx(1.0)
    assert scores.averages["performance"] == pytest.approx(1.0)


def test_a_planted_partition_is_recovered_and_sits_above_the_resolution_limit() -> None:
    """Four blocks of thirty, with the ground truth on the nodes before anything runs (§18.2)."""
    graph = planted_partition(4, 30, 0.35, 0.02, seed=5)
    truth = {node: int(data["community"]) for node, data in graph.nodes(data=True)}
    found = louvain(graph, seed=5, runs=5)
    scores = evaluate_partition(graph, found.communities, truth=truth, link_prediction=False)
    assert scores.truth is not None
    assert scores.truth.nmi > 0.9
    assert scores.truth.adjusted_mutual_information > 0.9
    assert scores.truth.adjusted_rand_index > 0.9
    # every community is far above sqrt(2m), so none of them is at the resolution limit's mercy
    assert scores.resolution_limit == pytest.approx(math.sqrt(2 * graph.number_of_edges()))
    assert scores.flagged == []
    assert all(row.internal_edges > scores.resolution_limit for row in scores.per_community)


def test_three_node_communities_on_a_two_thousand_edge_graph_are_flagged() -> None:
    """§36.1's resolution limit: below sqrt(2|E|) internal edges, merging raises modularity."""
    blob = nx.gnm_random_graph(100, 2000, seed=5)
    graph = nx.relabel_nodes(blob, {n: f"b{n:03d}" for n in blob})
    for name in ("p", "q"):
        for a, b in ((0, 1), (1, 2), (0, 2)):
            graph.add_edge(f"{name}{a}", f"{name}{b}")
    partition = [
        sorted(n for n in graph if n.startswith("b")),
        [f"p{i}" for i in range(3)],
        [f"q{i}" for i in range(3)],
    ]
    scores = evaluate_partition(graph, partition, link_prediction=False)
    assert scores.edges == 2006
    assert scores.resolution_limit == pytest.approx(math.sqrt(2 * 2006))
    assert scores.resolution_limit > 63
    assert scores.flagged == [1, 2]  # the two triangles, three internal edges each
    text = "\n".join(render_evaluation(scores))
    assert "2 of 3 communities have fewer than sqrt(2|E|) = 63.3 internal edges" in text
    assert "#2 (3 edges)" in text and "#3 (3 edges)" in text


# ------------------------------------------------------------------ §36.3, link prediction


def test_the_factions_predict_held_out_edges_and_shuffled_labels_do_not(karate: nx.Graph) -> None:
    """A real partition beats its own shuffle; the shuffle sits on 0.5, which is chance."""
    prediction = link_prediction_score(karate, _factions(karate), share=0.3, seed=7)
    assert prediction is not None
    assert prediction.held_out == 23 and prediction.non_edges == 23
    assert prediction.auc > 0.7
    assert prediction.null.null_mean == pytest.approx(0.5, abs=0.1)
    assert prediction.auc > prediction.null.null_mean


def test_a_partition_with_the_labels_dealt_out_at_random_predicts_nothing(
    karate: nx.Graph,
) -> None:
    """The arithmetic identity: shuffled labels carry no information about where an edge is.

    Averaged over twenty shuffles of the same faction sizes the AUC has to sit on 0.5 -- the
    value a predictor that knows nothing reads, because every pair it scores is a tie -- and the
    modularity has to sit at zero or a little under it, which is what §36.1 says a partition
    holding no more edges than chance would give it scores. It lands a little under because a
    shuffle scatters the hubs, and a group of high-degree nodes is *expected* many edges: the
    book's own negative case, a partition whose members connect to each other less than chance.
    """
    rng = random.Random(4)
    nodes = sorted(karate)
    aucs: list[float] = []
    modularities: list[float] = []
    for _ in range(20):
        rng.shuffle(nodes)
        partition = [sorted(nodes[:17]), sorted(nodes[17:])]
        scores = evaluate_partition(karate, partition, share=0.3, nulls=0, seed=7)
        modularities.append(scores.modularity)
        assert scores.link_prediction is not None
        aucs.append(scores.link_prediction.auc)
    assert sum(aucs) / len(aucs) == pytest.approx(0.5, abs=0.06)
    assert -0.06 < sum(modularities) / len(modularities) < 0.01
    assert max(modularities) < 0.2  # nowhere near the 0.3582 the real split scores
    assert min(modularities) < 0  # and a negative modularity is a reading, not a bug (p. 512)


def test_a_network_too_small_to_hold_edges_out_says_so_instead_of_guessing() -> None:
    """Below twenty edges a tenth of them is one pair, and an AUC over one pair is noise."""
    graph = nx.cycle_graph(8)
    assert link_prediction_score(graph, [sorted(graph)], seed=1) is None
    scores = evaluate_partition(graph, [sorted(graph)])
    assert scores.link_prediction is None
    assert "Not run" in "\n".join(render_evaluation(scores))


# ------------------------------------------------------------------ §36.4, the chance correction


def test_two_independent_labellings_score_a_positive_nmi_and_a_negative_ami() -> None:
    """Figure 36.12: ten elements, three values, drawn independently -- NMI 0.09, AMI -0.22.

    *"Yet, if you calculate their NMI values, you're going to obtain around 0.09: a non-zero
    mutual information from vectors that literally have nothing to do with each other. This is
    not good."* The exact draw is not the book's, so the two values are asserted for what they
    are rather than to the digit: NMI above zero, AMI below it.
    """
    rng = random.Random(3)
    a = [rng.randrange(3) for _ in range(10)]
    b = [rng.randrange(3) for _ in range(10)]
    from graphrag.sna.stats import normalized_mutual_information

    assert normalized_mutual_information(a, b) > 0.0
    assert adjusted_mutual_information(a, b) < 0.0
    assert adjusted_mutual_information(a, a) == pytest.approx(1.0)


def test_adjusted_mutual_information_refuses_what_it_cannot_compare() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        adjusted_mutual_information([], [])
    with pytest.raises(ValueError, match="same elements"):
        adjusted_mutual_information([1, 2], [1, 2, 3])


# ------------------------------------------------------------------ the contract of the battery


def test_an_overlapping_partition_is_refused_and_points_at_chapter_38(karate: nx.Graph) -> None:
    """§36.1 p. 515: the standard definition works on disjoint partitions only."""
    with pytest.raises(ValueError, match=r"ch\. 38"):
        evaluate_partition(karate, [[0, 1, 2], [2, 3, 4]])


def test_a_partition_that_covers_part_of_the_network_is_scored_on_that_part(
    karate: nx.Graph,
) -> None:
    """A clustering that dropped nodes for want of features still has communities to measure."""
    factions = _factions(karate)
    partial = [faction[:10] for faction in factions]
    scores = evaluate_partition(karate, partial, link_prediction=False)
    assert scores.nodes == 20
    assert any("covers 20 of the network's 34 nodes" in note for note in scores.notes)


def test_an_undefined_measure_is_a_dash_and_a_null_never_a_zero() -> None:
    """A singleton community has no internal density: the report must not print 0.0 for it."""
    graph = nx.star_graph(5)
    partition = [[0, 1, 2, 3, 4], [5]]
    scores = evaluate_partition(graph, partition, link_prediction=False)
    singleton = scores.per_community[1]
    assert math.isnan(singleton.internal_density)
    assert evaluation_payload(scores)["per_community"][1]["internal_density"] is None
    assert "| - |" in "\n".join(render_evaluation(scores))


def test_a_directed_network_is_flattened_and_the_section_says_so() -> None:
    """§6.2: the standard modularity of p. 515 is defined on undirected graphs."""
    graph = nx.DiGraph()
    for a, b in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3)):
        graph.add_edge(a, b)
    scores = evaluate_partition(graph, [[0, 1, 2], [3, 4, 5]], link_prediction=False)
    assert any("flatten" in note for note in scores.notes)
    assert scores.edges == 7


def test_every_measure_declares_what_it_wants_and_which_size_pushes_it(karate: nx.Graph) -> None:
    """The chapter's argument is the table, not the values: no measure may be printed bare."""
    assert {measure.wants for measure in MEASURES} == {"minimise", "maximise"}
    scores = evaluate_partition(karate, _factions(karate), link_prediction=False)
    text = "\n".join(render_evaluation(scores))
    for measure in MEASURES:
        assert measure.name in text
        assert measure.favours in text
        assert measure.section in text
        assert measure.key in scores.averages
    payload = evaluation_payload(scores)
    assert {row["key"] for row in payload["measures"]} == {m.key for m in MEASURES}
    assert payload["implements"] == "Atlas §36.1-36.4 (community evaluation)"


def test_the_resolution_limit_is_a_property_of_the_network_not_the_partition(
    karate: nx.Graph,
) -> None:
    assert resolution_limit(karate) == pytest.approx(math.sqrt(156))
    assert resolution_limit(nx.Graph()) == 0.0


# ------------------------------------------------------------------ the section in a report


def _analysis(graph: nx.Graph, **kwargs: object):  # type: ignore[no-untyped-def]
    graph.graph.update(min_weight=1, frame="A generated network, for a test.")
    return run_analysis(
        InMemoryGraphStore(),
        graph,
        persona_id="test-pm",
        network="speakers",
        seed=3,
        runs=2,
        samples=5,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("method", ["louvain", "kmeans"])
def test_the_section_is_printed_for_every_method_that_produces_a_partition(method: str) -> None:
    """A K-means clustering of a graph is still a partition of it, and still has a modularity."""
    graph = planted_partition(3, 12, 0.6, 0.03, seed=2)
    graph = nx.relabel_nodes(graph, {n: f"n{n:02d}" for n in graph})
    analysis = _analysis(graph, method=method, features="spectral")
    assert analysis.evaluation is not None
    text = render_markdown(analysis)
    assert "## Community evaluation" in text
    assert "**Implements.** §36.1-36.4" in text
    assert "**Sampling frame.** A generated network, for a test." in text
    assert "### Modularity (§36.1)" in text
    assert "### The partition as a link predictor (§36.3)" in text
    assert "No ground truth was given" in text  # §36.4 on a corpus network

    payload = to_payload(analysis)["evaluation"]
    assert payload["communities"] == len(analysis.groups)
    assert payload["nodes"] == sum(payload["sizes"])
    assert payload["resolution_limit"] == pytest.approx(math.sqrt(2 * analysis.evaluation.edges))
    assert payload["link_prediction"]["seed"] == 3
    assert "same community sizes" in payload["link_prediction"]["null"]


def test_an_empty_network_has_no_partition_and_so_no_section() -> None:
    analysis = _analysis(nx.Graph(), method="louvain")
    assert analysis.evaluation is None
    assert "## Community evaluation" not in render_markdown(analysis)
    assert "evaluation" not in to_payload(analysis)
