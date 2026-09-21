"""Chapter 27's backbones, held to answers that exist before the code runs.

Four kinds of known answer are used here, and none of them is "the function returned something":

* **Arithmetic recomputed independently.** The noise-corrected p-value of one edge of a five-node
  planted graph is summed from ``math.comb`` in the test, with the trials, the success
  probability and the expectation written out in the comment (§27.6).
* **Properties the method is defined by.** A maximum spanning tree has ``n - 1`` edges per
  component and no non-tree edge heavier than the lightest edge on the tree path it closes
  (§13.4); a planar maximally filtered graph is planar and holds ``3(n - 2)`` edges on a graph
  dense enough to give it that many (§13.4); a convex skeleton's every biconnected component is
  a clique (§27.4); a doubly stochastic matrix's rows and columns sum to 1 (§27.2); every
  shortest-path tree of a star uses every edge, and each edge of a triangle is in two of its
  three trees (§27.3).
* **The book's own claim about two methods, planted so it can fail.** On a hub-and-spoke network
  with one dense clique bolted on, the disparity filter keeps every one of the hub's spokes and
  the noise-corrected filter keeps none of them, while both keep the clique -- p. 391's "high
  centralization, broad degree distributions, and weak communities" and p. 393's "NC overweights
  peripheries and communities", made into an assertion.
* **Continuity.** ``naive`` is the ``--min-weight`` prune it replaces, edge for edge, on the
  networks the shared corpus fixture builds.
"""

from __future__ import annotations

import itertools
from math import comb

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.backbone import (
    _BY_NAME,
    BACKBONES,
    SALIENCE_THRESHOLD,
    _degree_stratified_sample,
    _naive_top_note,
    backbone,
    compare_backbones,
    disparity_p,
    doubly_stochastic_scores,
    high_salience,
    noise_corrected_p,
    prune_below,
    render_backbones,
)
from graphrag.sna.compare import compare_windows, render_comparison
from graphrag.sna.export import (
    build_network,
    entity_co_mention,
    speaker_co_participation,
)
from graphrag.sna.export import (
    describe as describe_network,
)
from tests.legendary import les_miserables

# --------------------------------------------------------------------------- planted graphs


@pytest.fixture
def five_nodes() -> nx.Graph:
    """Five nodes whose strengths and total weight can be added up by eye.

    Edges: a-b 6, a-c 2, b-c 3, c-d 1, d-e 4. Strengths: a=8, b=9, c=6, d=5, e=4, which sum to
    32 -- twice the total edge weight of 16, as they must.
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 6), ("a", "c", 2), ("b", "c", 3), ("c", "d", 1), ("d", "e", 4)]
    )
    return graph


@pytest.fixture
def star() -> nx.Graph:
    """A hub with four leaves of different weights: the smallest asymmetry there is."""
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("hub", "a", 5), ("hub", "b", 4), ("hub", "c", 3), ("hub", "d", 2)]
    )
    return graph


#: The planted network behind p. 391. One hub with 20 spokes of weight 4; each spoke also holds
#: six villagers of weight 1, so a spoke's strength is 10 and the hub edge is 40% of it; and one
#: triangle of weight 6 whose members each hold ten villagers of their own. The hub is joined to
#: the triangle by a single edge of weight 1 so the network is one component.
HUB_SPOKES = 20
HUB_WEIGHT = 4
SPOKE_VILLAGERS = 6
CLIQUE_WEIGHT = 6
CLIQUE_VILLAGERS = 10


@pytest.fixture
def hub_and_clique() -> nx.Graph:
    """The mobility network of §27.5 in miniature: New York, its small towns, and a neighbourhood.

    Both filters are run over this at alpha 0.05 and they disagree in the way the book says
    they will. The p-values are exact rational arithmetic, so nothing here is near a boundary by
    luck:

    * from a spoke, the hub edge has disparity p = (1 - 4/10)^6 = 0.046656, which clears 0.05;
    * from the hub, the same edge has p = (1 - 4/81)^20 = 0.3632, which does not.

    One success out of two attempts is enough for the disparity filter (p. 390), so every spoke
    survives on the small town's vote, exactly as Franklington saves its link to New York.
    """
    graph = nx.Graph()
    for spoke in range(HUB_SPOKES):
        graph.add_edge("hub", f"s{spoke}", weight=HUB_WEIGHT)
        for villager in range(SPOKE_VILLAGERS):
            graph.add_edge(f"s{spoke}", f"v{spoke}-{villager}", weight=1)
    clique = ["c0", "c1", "c2"]
    for u, v in itertools.combinations(clique, 2):
        graph.add_edge(u, v, weight=CLIQUE_WEIGHT)
    for position, node in enumerate(clique):
        for villager in range(CLIQUE_VILLAGERS):
            graph.add_edge(node, f"w{position}-{villager}", weight=1)
    graph.add_edge("hub", "c0", weight=1)
    return graph


def clique_edges() -> list[tuple[str, str]]:
    """The three edges of the planted triangle."""
    return list(itertools.combinations(("c0", "c1", "c2"), 2))


def spoke_edges() -> list[tuple[str, str]]:
    """The twenty hub-to-spoke edges of the planted network."""
    return [("hub", f"s{spoke}") for spoke in range(HUB_SPOKES)]


# ----------------------------------------------------------------------- §27.6 noise corrected


def test_the_noise_corrected_p_value_is_the_binomial_tail_recomputed_by_hand(
    five_nodes: nx.Graph,
) -> None:
    """§27.6 for the edge a-b of the five-node graph, summed here from ``math.comb``.

    The arithmetic, from p. 392:

    * trials = the total edge weight summed over ordered pairs = 2 * (6+2+3+1+4) = **32**;
    * successes = the weight of a-b = **6**;
    * success probability = (strength of a) * (strength of b) / trials^2
      = (6+2) * (6+3) / 32^2 = 72 / 1024 = **0.0703125**;
    * so the null expects 32 * 0.0703125 = **2.25** units of weight on this edge, and the
      p-value is P(X >= 6) = 1 - sum_{k=0..5} C(32, k) p^k (1-p)^(32-k).
    """
    trials, successes, probability = 32, 6, 72 / 1024
    assert abs(trials * probability - 2.25) < 1e-12
    by_hand = 1.0 - sum(
        comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
        for k in range(successes)
    )
    assert by_hand == pytest.approx(0.02243, abs=5e-6)  # significant at 0.05, not at 0.01
    assert noise_corrected_p(five_nodes, "a", "b") == pytest.approx(by_hand, rel=1e-9)


def test_noise_corrected_is_symmetric_and_the_disparity_filter_is_not(
    five_nodes: nx.Graph, star: nx.Graph, hub_and_clique: nx.Graph
) -> None:
    """p. 392: NC's three ingredients "are the same in the perspective of u and v"; DF's are not."""
    for u, v in five_nodes.edges():
        assert noise_corrected_p(five_nodes, u, v) == noise_corrected_p(five_nodes, v, u)

    # The star's leaves have one neighbour each, so the formula's exponent is zero and their
    # side of the test is undefined: reported as 1.0, never significant. The hub's side is a
    # real number, so the two disagree on every edge of the star.
    assert disparity_p(star, "a", "hub") == 1.0
    assert disparity_p(star, "hub", "a") == pytest.approx((1 - 5 / 14) ** 3)
    assert disparity_p(star, "hub", "a") != disparity_p(star, "a", "hub")

    # Where both endpoints have a distribution the disagreement is the method's point: the small
    # town keeps the edge (0.0467) that the hub votes to delete (0.3632).
    assert disparity_p(hub_and_clique, "s0", "hub") == pytest.approx(0.6**6)
    assert disparity_p(hub_and_clique, "hub", "s0") == pytest.approx((1 - 4 / 81) ** 20)
    assert (
        disparity_p(hub_and_clique, "s0", "hub") < 0.05 < disparity_p(hub_and_clique, "hub", "s0")
    )


def test_noise_corrected_refuses_weights_that_are_not_counts(five_nodes: nx.Graph) -> None:
    """p. 392: "NC works only for discrete counts as edge weights"."""
    five_nodes["a"]["b"]["weight"] = 6.5
    with pytest.raises(ValueError, match="count weights"):
        backbone(five_nodes, "noise-corrected")


# --------------------------------------------------------- §27.5 against §27.6, the p. 391 bias


def test_the_disparity_filter_keeps_the_hub_and_noise_corrected_does_not(
    hub_and_clique: nx.Graph,
) -> None:
    """The book's own claim, planted: DF centralises, NC keeps the community (pp. 391-393)."""
    disparity = backbone(hub_and_clique, "disparity", alpha=0.05)
    corrected = backbone(hub_and_clique, "noise-corrected", alpha=0.05)

    # DF: one success out of two attempts, so every spoke survives on the small node's vote.
    assert disparity.degree("hub") == HUB_SPOKES
    for u, v in spoke_edges():
        assert disparity.has_edge(u, v)
    # NC: both nodes have to agree, and the hub votes to delete every one of them.
    for u, v in spoke_edges():
        assert not corrected.has_edge(u, v)
    assert "hub" not in corrected  # nothing it was joined to survived, so it is not a node here

    # Both keep the dense corner: the clique edges beat every null in play.
    for u, v in clique_edges():
        assert disparity.has_edge(u, v)
        assert corrected.has_edge(u, v)

    # And the consequence the book names: DF's backbone is a star with a triangle attached,
    # NC's is the periphery. Centralisation is the difference, not a detail of it.
    assert disparity.number_of_edges() < corrected.number_of_edges()
    assert max(dict(disparity.degree()).values()) == HUB_SPOKES
    assert max(dict(corrected.degree()).values()) <= CLIQUE_VILLAGERS + 2


def test_a_correction_can_only_take_edges_away(hub_and_clique: nx.Graph) -> None:
    """§3.3: a backbone tests every edge at once, so the family can be corrected."""
    uncorrected = backbone(hub_and_clique, "noise-corrected", alpha=0.05)
    for method in ("bonferroni", "holm", "fdr_bh"):
        adjusted = backbone(hub_and_clique, "noise-corrected", alpha=0.05, correction=method)
        assert set(adjusted.edges()) <= set(uncorrected.edges()), method
        assert adjusted.graph["backbone_correction"] == method
    strict = backbone(hub_and_clique, "noise-corrected", alpha=0.05, correction="bonferroni")
    assert strict.number_of_edges() < uncorrected.number_of_edges()


# ------------------------------------------------------------------------------- §27.1 naive


@pytest.mark.parametrize("min_weight", [2, 3])
def test_naive_is_the_min_weight_prune_it_replaces(
    layered: InMemoryGraphStore, min_weight: int
) -> None:
    """Continuity: the naive method keeps exactly the edges ``--min-weight`` has always kept."""
    for build in (speaker_co_participation, entity_co_mention):
        whole = build(layered, "test-layers", min_weight=1)
        pruned = prune_below(build(layered, "test-layers", min_weight=1), min_weight)
        filtered = backbone(whole, "naive", threshold=min_weight)
        assert set(filtered.edges()) == set(pruned.edges())
        assert {(u, v, d["score"]) for u, v, d in filtered.edges(data=True)} == {
            (u, v, float(d["weight"])) for u, v, d in pruned.edges(data=True)
        }


def test_the_naive_row_of_the_comparison_carries_the_page_383_arithmetic(
    hub_and_clique: nx.Graph,
) -> None:
    """p. 383: most edges sit at the smallest weight, so the mildest threshold is not mild."""
    rows = {row.method: row for row in compare_backbones(hub_and_clique, methods=["naive"])}
    naive = rows["naive"]
    # 151 edges of weight 1 out of 174: 87% of the network sits at the floor, and the
    # smallest hard threshold available (4, the next distinct weight) removes every one of them.
    assert "87% of the edges sit at the smallest weight (1)" in naive.note
    assert "smallest possible hard threshold (4)" in naive.note
    assert naive.edges == 174 - 151


# --------------------------------------------------------------------- §27.2 doubly stochastic


def test_the_doubly_stochastic_scores_are_doubly_stochastic_when_they_converge() -> None:
    """§27.2: alternate row and column normalisation until both sum to 1 -- when it can."""
    complete = nx.complete_graph(4)
    nx.set_edge_attributes(complete, 1.0, "weight")
    scores, converged = doubly_stochastic_scores(complete)
    assert converged
    # Every node of K4 has three identical edges, so each entry must be exactly 1/3.
    assert {round(value, 9) for value in scores.values()} == {round(1 / 3, 9)}

    ring = nx.cycle_graph(6)
    nx.set_edge_attributes(ring, 1.0, "weight")
    scores, converged = doubly_stochastic_scores(ring)
    assert converged
    assert set(scores.values()) == {0.5}  # two edges per node, so each carries half the row

    filtered = backbone(ring, "doubly-stochastic")
    # Slater's rule: the highest threshold that keeps the component count. A ring cannot lose an
    # edge without the six nodes ceasing to be a ring, and all six scores tie, so all six stay.
    assert filtered.number_of_edges() == 6
    assert nx.number_connected_components(filtered) == 1


def test_sparse_doubly_stochastic_scores_agree_with_dense() -> None:
    """ATL-F2: above ``sparse_above`` the Sinkhorn normalisation runs on a sparse matrix instead
    of a dense one (§27.2), and the two must agree -- the sparse path is exact arithmetic here
    (diagonal scaling), not an iterative approximation like ``matrices.eigenpairs``'s, so the
    tolerance is tight."""
    graph, _ = les_miserables()
    dense_scores, dense_converged = doubly_stochastic_scores(graph)
    sparse_scores, sparse_converged = doubly_stochastic_scores(graph, sparse_above=1)
    assert sparse_converged == dense_converged
    assert dense_scores.keys() == sparse_scores.keys()
    for edge, value in dense_scores.items():
        assert sparse_scores[edge] == pytest.approx(value, abs=1e-9)

    # And on a network small enough to converge cleanly, forcing sparse below its own default
    # threshold changes nothing about the result.
    ring = nx.cycle_graph(6)
    nx.set_edge_attributes(ring, 1.0, "weight")
    dense_scores, dense_converged = doubly_stochastic_scores(ring)
    sparse_scores, sparse_converged = doubly_stochastic_scores(ring, sparse_above=1)
    assert sparse_converged and dense_converged
    for edge, value in dense_scores.items():
        assert sparse_scores[edge] == pytest.approx(value, abs=1e-9)


def test_the_two_mode_network_is_told_why_it_cannot_be_doubly_stochastic(
    layered: InMemoryGraphStore,
) -> None:
    """p. 386: "you cannot apply the doubly stochastic backboning to bipartite networks"."""
    two_mode = build_network(layered, "speakers-entities", "test-layers", min_weight=1)
    assert describe_network(two_mode).bipartite
    _, converged = doubly_stochastic_scores(two_mode)
    assert not converged
    note = backbone(two_mode, "doubly-stochastic").graph["backbone_note"]
    assert "two-mode" in note
    assert "|V1| = |V2|" in note
    assert "sparse networks" not in note  # the reason is the shape, not this corpus's size

    # A one-mode network that fails gets the other reason, which is the one that can be argued
    # with: another corpus might converge.
    graph, _ = les_miserables()
    assert not describe_network(graph).bipartite
    assert "sparse networks" in backbone(graph, "doubly-stochastic").graph["backbone_note"]


def test_a_normalisation_that_cannot_converge_says_so() -> None:
    """p. 385: "this solution cannot be always applied" on a sparse real network."""
    graph, _ = les_miserables()
    _, converged = doubly_stochastic_scores(graph)
    assert not converged  # 77 nodes, so the matrix has no positive diagonal to converge onto
    filtered = backbone(graph, "doubly-stochastic")
    assert "did not converge" in filtered.graph["backbone_note"]
    assert "did not converge" in filtered.graph["backbone"]


# ---------------------------------------------------------------------- §27.3 high salience


def _bridge_barbell(size: int) -> tuple[nx.Graph, tuple[str, str]]:
    """Two size-``size`` cliques joined by exactly one edge: ``("l0", "r0")`` is the bridge.

    A bridge is a cut edge, so *every* node's shortest-path tree -- whichever side it starts on
    -- has to cross it to reach the other side. Its full-network §27.3 score is therefore exactly
    1.0, not approximately, and any non-empty sample of sources, from either side, sees it in
    every one of its trees too: this is the known answer a sampled salience is checked against.
    """
    left = nx.complete_graph([f"l{i}" for i in range(size)])
    right = nx.complete_graph([f"r{i}" for i in range(size)])
    graph = nx.union(left, right)
    nx.set_edge_attributes(graph, 1.0, "weight")
    bridge = ("l0", "r0")
    graph.add_edge(*bridge, weight=1.0)
    return graph, bridge


def _edge_value(scores: dict[tuple[str, str], float], u: str, v: str) -> float:
    """Read an edge's score whichever orientation the backbone canonicalised it to."""
    return scores[(u, v)] if (u, v) in scores else scores[(v, u)]


def test_high_salience_counts_the_shortest_path_trees_an_edge_is_in(star: nx.Graph) -> None:
    """§27.3: sum one shortest-path tree per node, and score each edge by its share of them."""
    # Every path in a star runs through the hub, so every tree holds every edge.
    scores, note = high_salience(star)
    assert set(scores.values()) == {1.0}
    assert note == ""  # below the sampling threshold: nothing was sampled, and it says so

    triangle = nx.Graph()
    triangle.add_weighted_edges_from([("a", "b", 1), ("b", "c", 1), ("a", "c", 1)])
    # a's tree is {ab, ac}, b's is {ab, bc}, c's is {ac, bc}: each edge is in two of three.
    scores, _ = high_salience(triangle)
    assert sorted(scores.values()) == pytest.approx([2 / 3, 2 / 3, 2 / 3])

    barbell = nx.Graph()
    barbell.add_weighted_edges_from(
        [("a", "b", 3), ("a", "c", 3), ("b", "c", 3), ("c", "d", 1), ("d", "e", 3), ("d", "f", 3)]
    )
    scores, _ = high_salience(barbell)
    bridge = scores[("c", "d")] if ("c", "d") in scores else scores[("d", "c")]
    assert bridge == 1.0  # the only way across, so no tree can do without it
    assert backbone(barbell, "high-salience").has_edge("c", "d")


def test_high_salience_with_explicit_sources_matches_hand_counting() -> None:
    """``sources`` overrides the automatic choice and scales by ``len(sources)``, not ``n``."""
    triangle = nx.Graph()
    triangle.add_weighted_edges_from([("a", "b", 1), ("b", "c", 1), ("a", "c", 1)])
    # a's tree alone is {ab, ac}: both score 1.0 from a's single tree, bc scores 0.0.
    scores, note = high_salience(triangle, sources=["a"])
    assert scores == {("a", "b"): 1.0, ("a", "c"): 1.0, ("b", "c"): 0.0}
    assert "1 of 3" in note
    assert "sources given by the caller" in note


def test_high_salience_sampled_bridge_matches_the_full_answer_at_any_sample_size() -> None:
    """An identity check, not a convergence one: a cut edge is in *every* source's tree, sampled
    or not, so the sampled score equals the full one exactly at every sample size -- it is the
    stratified mean's weights summing to 1 that makes this hold at every ``k``, not the sample
    happening to be large. See ``test_high_salience_sampled_scores_converge_...`` below for an
    edge whose full score is not already known before the code runs."""
    graph, bridge = _bridge_barbell(6)
    full_scores, full_note = high_salience(graph)
    assert full_note == ""
    assert _edge_value(full_scores, *bridge) == 1.0

    for k in (1, 2, 4, 8, graph.number_of_nodes()):
        scores, note = high_salience(graph, sample=k, seed=3)
        assert _edge_value(scores, *bridge) == pytest.approx(1.0)
        if k < graph.number_of_nodes():
            assert f"{k:,} of {graph.number_of_nodes():,}" in note
        else:
            assert note == ""  # sample >= n is the unsampled, full computation


def test_high_salience_sampled_scores_converge_to_the_full_answer_as_the_sample_grows() -> None:
    """A genuine convergence check, on an edge whose full-network score is not a foregone
    conclusion (unlike a bridge's exact 1.0): a seeded Barabasi-Albert graph's edge closest to
    0.5 -- the middle of §27.3's empty "horns" valley, so a genuinely uncertain one -- and the
    mean absolute error of its sampled score against the full computation, averaged over ten
    seeds so one unlucky draw cannot decide it, at a small sample size against a large one.
    """
    graph = nx.barabasi_albert_graph(300, 3, seed=7)
    nx.set_edge_attributes(graph, 1.0, "weight")
    full_scores, _ = high_salience(graph)
    edge = min(full_scores, key=lambda e: abs(full_scores[e] - 0.5))
    truth = full_scores[edge]

    def mean_absolute_error(sample: int, seeds: int = 10) -> float:
        errors = [
            abs(_edge_value(high_salience(graph, sample=sample, seed=seed)[0], *edge) - truth)
            for seed in range(seeds)
        ]
        return sum(errors) / len(errors)

    small_sample, large_sample = mean_absolute_error(10), mean_absolute_error(160)
    assert large_sample < small_sample * 0.6  # a real drop, not sampling noise


def test_high_salience_samples_automatically_above_the_threshold() -> None:
    """Above ``sample_above`` and with no explicit ``sample``, the draw happens on its own."""
    graph, bridge = _bridge_barbell(6)  # 12 nodes
    scores, note = high_salience(graph, seed=1, sample_above=4)
    assert len(_degree_stratified_sample(graph, 4, 1)[0]) == 4
    assert "4 of 12" in note
    assert "degree-stratified" in note
    assert _edge_value(scores, *bridge) == pytest.approx(1.0)

    # Below the (overridden) threshold, nothing is sampled and the note says so.
    _, unsampled_note = high_salience(graph, seed=1, sample_above=graph.number_of_nodes())
    assert unsampled_note == ""


def test_high_salience_below_threshold_is_unchanged_by_atl_f2(star: nx.Graph) -> None:
    """Rule: identical numbers where nothing was sampled. A network below
    :data:`~graphrag.sna.backbone.SALIENCE_SAMPLE_ABOVE_NODES` gets exactly what this function
    always returned, node for node."""
    scores, note = high_salience(star)
    assert note == ""
    assert scores == dict.fromkeys(scores, 1.0)


def test_degree_stratified_sample_spans_the_degree_distribution() -> None:
    """The sample is not a plain uniform draw: its top band always contributes a max-degree node
    (§29.1's own complaint about uniform samples missing a hub), is deterministic given a seed,
    has no duplicates, and never exceeds the population. Its weights are each band's population
    share and sum to exactly 1, which is what makes the stratified mean unbiased.

    ``_bridge_barbell(10)`` has exactly two nodes of the network's maximum degree -- the bridge's
    two endpoints -- so a top band of size 2 (``k=10`` over 20 nodes) is exactly that pair, and
    the one node it draws is always one of the two, whichever the seed picks.
    """
    graph, _ = _bridge_barbell(10)  # two 10-cliques joined by one bridge: 20 nodes
    top_degree = max(d for _, d in graph.degree())
    sample, weights = _degree_stratified_sample(graph, 10, seed=0)
    assert len(sample) == len(set(sample)) == 10
    assert set(weights) == set(sample)
    assert max(graph.degree(node) for node in sample) == top_degree
    # 20 nodes over 10 evenly-sized bands: every band is 2 nodes, so every weight is 2/20 = 0.1.
    assert weights == {node: pytest.approx(0.1) for node in sample}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert _degree_stratified_sample(graph, 10, seed=0)[0] == sample  # deterministic given a seed
    assert _degree_stratified_sample(graph, 0, seed=0) == ([], {})
    filled, filled_weights = _degree_stratified_sample(graph, 999, seed=0)
    assert len(filled) == graph.number_of_nodes()
    assert sum(filled_weights.values()) == pytest.approx(1.0)


def test_degree_stratified_sample_weights_are_unbiased_when_bands_split_unevenly() -> None:
    """The case the review names: ``n`` that does not divide evenly by ``k``. A plain ``1 / k``
    per draw would be biased here -- some bands are bigger than others -- but the size-weighted
    mean is exact regardless: drawing every node (one band each of size 1) recovers the true
    population mean exactly, node for node, which is the known answer this checks against."""
    graph, _ = _bridge_barbell(7)  # 14 nodes, an awkward divisor for several sample sizes
    n = graph.number_of_nodes()
    for k in (3, 4, 5, 6, 9, 11, 13):
        sample, weights = _degree_stratified_sample(graph, k, seed=1)
        assert len(sample) == k
        assert sum(weights.values()) == pytest.approx(1.0)
        # Every drawn node's weight is its own band's size over n, an integer number of nodes.
        for weight in weights.values():
            assert (weight * n) == pytest.approx(round(weight * n))

    sample, weights = _degree_stratified_sample(graph, n, seed=1)
    assert sorted(sample, key=str) == sorted(graph.nodes, key=str)
    assert weights == dict.fromkeys(sample, pytest.approx(1.0 / n))


# --------------------------------------------------------------------------- §27.4 convex


def is_tree_of_cliques(graph: nx.Graph) -> bool:
    """§27.4's target shape: every biconnected component complete, so the blocks form a tree."""
    return all(
        all(graph.has_edge(u, v) for u, v in itertools.combinations(block, 2))
        for block in nx.biconnected_components(graph)
    )


def test_convex_reduction_leaves_a_tree_of_cliques() -> None:
    """p. 389: "stitching together cliques" is what a convex network is."""
    bowtie = nx.Graph()
    for triangle in (("a", "b", "x"), ("c", "d", "x")):
        for u, v in itertools.combinations(triangle, 2):
            bowtie.add_edge(u, v, weight=2)
    # A bowtie is already two cliques joined at a cut vertex, so nothing has to go.
    assert backbone(bowtie, "convex").number_of_edges() == 6

    square = nx.Graph()
    square.add_weighted_edges_from([("a", "b", 9), ("b", "c", 8), ("c", "d", 7), ("d", "a", 6)])
    # A 4-cycle is not convex: the shortest path between two opposite corners leaves the
    # subgraph. The lightest edge is the one that goes, leaving a path -- which is a tree.
    reduced = backbone(square, "convex")
    assert reduced.number_of_edges() == 3
    assert not reduced.has_edge("d", "a")
    assert nx.is_tree(reduced)

    graph, known = les_miserables()
    skeleton = backbone(graph, "convex")
    assert is_tree_of_cliques(skeleton)
    assert skeleton.number_of_nodes() == known.nodes  # no node is dropped by this one
    assert nx.number_connected_components(skeleton) == nx.number_connected_components(graph)


# ------------------------------------------------------------------ §13.4 spanning tree, PMFG


def test_the_maximum_spanning_tree_is_a_tree_and_no_edge_beats_it() -> None:
    """§13.4: n-1 edges, and the cycle property that makes it maximal."""
    graph, known = les_miserables()
    tree = backbone(graph, "mst")
    assert tree.number_of_nodes() == known.nodes
    assert tree.number_of_edges() == known.nodes - 1  # one component, so n - 1
    assert nx.is_tree(tree)
    total = sum(float(w) for _, _, w in tree.edges(data="weight"))
    assert total == pytest.approx(
        sum(float(w) for _, _, w in nx.maximum_spanning_tree(graph).edges(data="weight"))
    )
    # Maximality, checked without building another tree: no edge left out is heavier than the
    # lightest edge on the tree path it would close, or swapping the two would improve the tree.
    for u, v, weight in graph.edges(data="weight"):
        if tree.has_edge(u, v):
            continue
        path = nx.shortest_path(tree, u, v)
        lightest = min(float(tree[a][b]["weight"]) for a, b in itertools.pairwise(path))
        assert float(weight) <= lightest


def test_the_planar_filtered_graph_is_planar_and_holds_three_n_minus_two_edges() -> None:
    """§13.4: "a planar maximally filtered graph must have 3(|V|-2) or fewer edges" (p. 203)."""
    complete = nx.complete_graph(8)
    for position, (u, v) in enumerate(complete.edges()):
        complete[u][v]["weight"] = position + 1  # distinct weights, so the greedy is unambiguous
    filtered = backbone(complete, "pmfg")
    assert nx.check_planarity(filtered, counterexample=False)[0]
    assert filtered.number_of_edges() == 3 * (8 - 2)
    # It is built heaviest-first, so it holds the maximum spanning tree it starts from.
    tree = nx.maximum_spanning_tree(complete)
    assert set(tree.edges()) <= set(filtered.edges()) | {(v, u) for u, v in filtered.edges()}

    graph, _ = les_miserables()
    real = backbone(graph, "pmfg")
    assert nx.check_planarity(real, counterexample=False)[0]
    assert real.number_of_edges() <= 3 * (graph.number_of_nodes() - 2)


# ------------------------------------------------------------------------- the one interface


@pytest.mark.parametrize("method", BACKBONES)
def test_every_survivor_carries_the_number_it_survived_on(
    method: str, hub_and_clique: nx.Graph
) -> None:
    """A backbone that cannot say why an edge is there is a filter nobody can check."""
    filtered = backbone(hub_and_clique, method, alpha=0.05, threshold=None, seed=7)
    assert filtered.number_of_edges() > 0
    for _, _, data in filtered.edges(data=True):
        assert ("p_value" in data) ^ ("score" in data)
        assert "weight" in data  # the original weight survives beside it
    if "p_value" in next(iter(filtered.edges(data=True)))[2]:
        for _, _, data in filtered.edges(data=True):
            assert 0.0 <= data["p_value"] <= 0.05
            assert data["p_adjusted"] >= data["p_value"] - 1e-12


@pytest.mark.parametrize("method", BACKBONES)
def test_the_run_is_recorded_on_the_graph(method: str, five_nodes: nx.Graph) -> None:
    """ATL-ENT-4: method, level, null model and what it removed, in ``graph.graph``."""
    filtered = backbone(five_nodes, method, alpha=0.5, seed=3)
    assert filtered.graph["backbone_method"] == method
    assert filtered.graph["backbone_edges_before"] == 5
    assert filtered.graph["backbone_edges_after"] == filtered.number_of_edges()
    assert filtered.graph["backbone_nodes_before"] == 5
    assert filtered.graph["backbone_seed"] == "3"
    assert filtered.graph["backbone_null"]
    sentence = filtered.graph["backbone"]
    assert method in sentence
    assert "of 5 edges survive" in sentence
    assert "Null model:" in sentence
    assert five_nodes.number_of_edges() == 5  # the original is never touched


def test_a_node_the_filter_stranded_goes_but_a_node_that_arrived_alone_stays() -> None:
    """p. 382: backboning "wants to keep as many [nodes] as possible"."""
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1), ("b", "c", 5)])
    graph.add_node("alone")
    dropped = backbone(graph, "naive", threshold=5)
    assert "a" not in dropped  # its only edge went, so it has nothing left to describe
    assert "alone" in dropped  # nothing of its was removed, so removing it would be our doing
    kept = backbone(graph, "naive", threshold=5, keep_isolates=True)
    assert "a" in kept and kept.degree("a") == 0


def test_a_threshold_below_one_is_still_one_edge_per_node() -> None:
    """n is a count of edges, so the note can never say "0 strongest"."""
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 3), ("b", "c", 2)])
    for asked in (0, -4, 0.5):
        filtered = backbone(graph, "naive-top", threshold=asked)
        assert filtered.graph["backbone_threshold"] == 1.0
        assert filtered.number_of_edges() == 2  # each node's strongest, unioned
        assert "0 strongest" not in _naive_top_note(filtered)
        assert "its 1 strongest edge(s)" in _naive_top_note(filtered)


def test_a_threshold_replaces_a_structural_default_and_the_record_says_which() -> None:
    """--threshold is the structural methods' companion to --alpha (docs/SNA.md)."""
    graph, _ = les_miserables()
    default = backbone(graph, "high-salience")
    assert default.graph["backbone_threshold"] == SALIENCE_THRESHOLD
    strict = backbone(graph, "high-salience", threshold=0.9)
    assert strict.graph["backbone_threshold"] == 0.9
    assert strict.number_of_edges() < default.number_of_edges()
    assert "threshold 0.9" in strict.graph["backbone"]

    # It is ignored by the two that cut on alpha, and the sentence says alpha rather than a
    # level that did no work.
    statistical = backbone(graph, "noise-corrected", alpha=0.05, threshold=0.9)
    assert statistical.graph["backbone_alpha"] == 0.05
    assert "alpha 0.05" in statistical.graph["backbone"]
    assert "threshold" not in statistical.graph["backbone"]
    assert set(statistical.edges()) == set(backbone(graph, "noise-corrected", alpha=0.05).edges())

    # And the comparison passes one level to every structural row.
    rows = {row.method: row for row in compare_backbones(graph, threshold=0.9, methods=["naive"])}
    # A level the caller chose is not the smallest hard threshold, and the note must not say it is:
    # 0.9 keeps every weight-1 edge, so the row's survival and its sentence agree.
    assert rows["naive"].edge_share == 1.0
    assert "the threshold asked for (0.9) keeps that tier" in rows["naive"].note
    assert "smallest possible hard threshold (0.9)" not in rows["naive"].note


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "hard-threshold"}, "backbone method must be one of"),
        ({"method": "naive", "alpha": 0.0}, "significance level"),
        ({"method": "naive", "alpha": 1.5}, "significance level"),
        ({"method": "naive", "correction": "sidak"}, "correction must be"),
    ],
)
def test_backbone_refuses_what_it_cannot_do(
    kwargs: dict[str, object], message: str, five_nodes: nx.Graph
) -> None:
    with pytest.raises(ValueError, match=message):
        backbone(five_nodes, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------- directed networks (27.5, 27.6, 6.2)


@pytest.fixture
def reciprocal() -> nx.DiGraph:
    """Four directed edges, two of them a reciprocal pair, with every sum small enough to add up.

    a->b 6, b->a 2, a->c 1, c->b 3. Out-strengths: a=7, b=2, c=3. In-strengths: a=2, b=9, c=1.
    The total weight is 12, which on a directed network is already the ordered-pair sum, so the
    binomial's trials are 12 and not 24.
    """
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([("a", "b", 6), ("b", "a", 2), ("a", "c", 1), ("c", "b", 3)])
    return graph


@pytest.fixture
def senders_and_receivers() -> nx.DiGraph:
    """The mobility story of p. 391 with direction: one edge each side of §27.5's "either".

    ``u -> v`` (weight 10) is two thirds of u's out-strength but only half of v's in-strength,
    so the sender finds it significant and the receiver does not. ``x -> y`` (weight 10) is the
    mirror image: unremarkable among x's two out-edges, dominant among y's six in-edges.
    """
    graph = nx.DiGraph()
    graph.add_edge("u", "v", weight=10)
    for leaf in range(5):
        graph.add_edge("u", f"u{leaf}", weight=1)  # u: out-strength 15 over 6 out-edges
    graph.add_edge("w", "v", weight=10)  # v: in-strength 20 over 2 in-edges
    graph.add_edge("x", "y", weight=10)
    graph.add_edge("x", "z", weight=10)  # x: out-strength 20 over 2 out-edges
    for leaf in range(5):
        graph.add_edge(f"y{leaf}", "y", weight=1)  # y: in-strength 15 over 6 in-edges
    return graph


def test_the_noise_corrected_null_is_directed_when_the_network_is(reciprocal: nx.DiGraph) -> None:
    """p. 392: "the u, v edge is different from the v, u edge", so the two p-values differ."""
    trials = 12  # the ordered-pair sum is the edge sum on a digraph: 6 + 2 + 1 + 3
    assert sum(w for _, _, w in reciprocal.edges(data="weight")) == trials

    def tail(successes: int, probability: float) -> float:
        return 1.0 - sum(
            comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
            for k in range(successes)
        )

    # a -> b: out-strength(a) * in-strength(b) / trials^2 = 7 * 9 / 144, so the null expects
    # 12 * 0.4375 = 5.25 of the 6 units observed -- unremarkable.
    forward = tail(6, 63 / 144)
    assert noise_corrected_p(reciprocal, "a", "b") == pytest.approx(forward, rel=1e-9)
    assert forward == pytest.approx(0.43776, abs=5e-6)

    # b -> a: out-strength(b) * in-strength(a) / trials^2 = 2 * 2 / 144, so the null expects
    # 0.333 and 2 were observed -- the same pair of nodes, the other way, and significant.
    backward = tail(2, 4 / 144)
    assert noise_corrected_p(reciprocal, "b", "a") == pytest.approx(backward, rel=1e-9)
    assert backward == pytest.approx(0.04233, abs=5e-6)
    assert backward < 0.05 < forward

    # The direction is the whole difference: undirected, the same four edges give one number
    # per pair, because both endpoints' strengths are then their totals.
    undirected = reciprocal.to_undirected()
    assert noise_corrected_p(undirected, "a", "b") == noise_corrected_p(undirected, "b", "a")

    # An edge that does not run that way is not an edge, even when the reverse exists.
    with pytest.raises(KeyError, match="not joined"):
        noise_corrected_p(reciprocal, "c", "a")


def test_the_directed_disparity_filter_asks_the_sender_then_the_receiver(
    senders_and_receivers: nx.DiGraph,
) -> None:
    """p. 391: significant "either when compared to the out-connections … or … the in-connection
    weights of the node receiving it"."""
    graph = senders_and_receivers
    # u -> v from u's out side: (1 - 10/15)^5 = (1/3)^5 = 0.004115. From v's in side:
    # (1 - 10/20)^1 = 0.5. The sender keeps it; the receiver would have deleted it.
    assert disparity_p(graph, "u", "v") == pytest.approx((1 / 3) ** 5)
    assert disparity_p(graph, "v", "u") == pytest.approx(0.5)
    # x -> y is the mirror: (1 - 10/20)^1 = 0.5 from x, (1 - 10/15)^5 from y.
    assert disparity_p(graph, "x", "y") == pytest.approx(0.5)
    assert disparity_p(graph, "y", "x") == pytest.approx((1 / 3) ** 5)

    filtered = backbone(graph, "disparity", alpha=0.05)
    assert filtered.is_directed()
    assert filtered.has_edge("u", "v")  # kept on the sender's vote alone
    assert filtered.has_edge("x", "y")  # kept on the receiver's vote alone
    # and the edge records both votes, so the one that kept it is visible
    assert filtered["u"]["v"]["p_value"] == pytest.approx((1 / 3) ** 5)
    assert filtered["u"]["v"]["p_other"] == pytest.approx(0.5)
    assert filtered["x"]["y"]["p_value"] == pytest.approx((1 / 3) ** 5)
    assert filtered["x"]["y"]["p_other"] == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["naive", "naive-top", "disparity", "noise-corrected"])
def test_the_directed_capable_methods_keep_the_direction(
    method: str, reciprocal: nx.DiGraph
) -> None:
    """A backbone of a digraph is a digraph, and the reverse edge is a separate decision."""
    filtered = backbone(reciprocal, method, alpha=0.5, keep_isolates=True)
    assert filtered.is_directed()
    assert set(filtered.edges()) <= set(reciprocal.edges())
    assert filtered.graph["backbone_method"] == method


@pytest.mark.parametrize("method", ["doubly-stochastic", "high-salience", "convex", "mst", "pmfg"])
def test_an_undirected_only_method_refuses_a_digraph_and_says_what_the_book_has(
    method: str, reciprocal: nx.DiGraph
) -> None:
    """A refusal that only says 'unsupported' teaches nothing; each quotes its own reason."""
    with pytest.raises(ValueError, match="is undirected") as caught:
        backbone(reciprocal, method)
    message = str(caught.value)
    assert method in message
    assert _BY_NAME[method].directed_note in message
    assert "noise-corrected" in message  # and what to use instead


def test_the_comparison_of_a_digraph_runs_what_applies_and_says_why_for_the_rest(
    reciprocal: nx.DiGraph,
) -> None:
    """§27's table over a directed network: four rows of numbers, five rows of reasons."""
    rows = {row.method: row for row in compare_backbones(reciprocal, alpha=0.5)}
    assert [name for name, row in rows.items() if row.applicable] == [
        "naive",
        "naive-top",
        "disparity",
        "noise-corrected",
    ]
    for name in ("doubly-stochastic", "high-salience", "convex", "mst", "pmfg"):
        assert not rows[name].applicable
        assert _BY_NAME[name].directed_note in rows[name].note

    report = render_backbones(reciprocal, list(rows.values()), alpha=0.5)
    assert "weakly connected component(s)" in report  # 27.2's rule reads them weakly here
    assert "Not run on this network" in report


# --------------------------------------------------------------------- §27's own comparison


def test_the_comparison_reports_every_method_with_its_null(hub_and_clique: nx.Graph) -> None:
    """p. 393: the methods "give very different results", so the report is the evidence."""
    rows = compare_backbones(hub_and_clique, alpha=0.05)
    assert [row.method for row in rows] == list(BACKBONES)
    for row in rows:
        assert row.applicable
        assert row.null and row.section
        assert 0.0 < row.edge_share <= 1.0
        assert row.weights is not None
    kept = {row.method: row.edges for row in rows}
    assert kept["mst"] < kept["pmfg"]  # a tree is inside the planar graph that contains it

    report = render_backbones(hub_and_clique, rows, alpha=0.05)
    assert "chapter 27" in report
    assert "Sampling frame:" in report
    assert f"{hub_and_clique.number_of_edges():,} edges" in report
    for row in rows:
        assert f"`{row.method}`" in report
        assert row.null in report


def test_a_method_that_refuses_the_network_is_a_row_not_a_gap(five_nodes: nx.Graph) -> None:
    five_nodes["a"]["b"]["weight"] = 6.5
    rows = {row.method: row for row in compare_backbones(five_nodes)}
    assert not rows["noise-corrected"].applicable
    assert "count weights" in rows["noise-corrected"].note
    assert rows["naive"].applicable
    assert "Not run on this network" in render_backbones(five_nodes, list(rows.values()))


# ------------------------------------------------------------------------ end to end, in place


def test_build_network_applies_the_backbone_and_the_frame_names_it(
    layered: InMemoryGraphStore,
) -> None:
    """The frame is where a reader finds out the network was filtered (ch. 27)."""
    plain = build_network(layered, "entities", "test-layers", min_weight=1)
    filtered = build_network(
        layered, "entities", "test-layers", min_weight=1, backbone="noise-corrected", alpha=0.5
    )
    assert filtered.number_of_edges() < plain.number_of_edges()
    assert filtered.graph["network"] == "entities"
    assert filtered.graph["backbone_method"] == "noise-corrected"
    frame = filtered.graph["frame"]
    assert "noise-corrected" in frame and "§27.6" in frame
    assert "describes the edges that survived that test" in frame
    for _, _, data in filtered.edges(data=True):
        assert data["p_value"] <= 0.5
    assert "backbone" not in plain.graph  # nothing is applied unless it was asked for
    assert "§27" not in plain.graph["frame"]


# --------------------------------------------------------------- §27.1's other naive strategy


def test_naive_top_keeps_each_nodes_strongest_edges_and_fixes_the_minimum_degree() -> None:
    """p. 384: "simply pick the top n strongest connections for each node"."""
    graph = nx.Graph()
    # A hub with four edges of distinct weights, and a lonely pair nothing else touches. The
    # top-1 answer is planted: the hub keeps its 9, each leaf keeps its only edge, and the pair
    # keeps its own edge -- so the union is every edge that is somebody's strongest.
    graph.add_weighted_edges_from(
        [("hub", "a", 9), ("hub", "b", 5), ("hub", "c", 3), ("hub", "d", 1), ("x", "y", 2)]
    )
    top = backbone(graph, "naive-top")
    assert set(top.edges()) == set(graph.edges())  # every edge is the strongest one somebody has
    assert min(degree for _, degree in top.degree()) == 1

    # A clique's nodes all have somebody stronger, so n decides how much of it survives.
    clique = nx.Graph()
    for position, (u, v) in enumerate(itertools.combinations("abcde", 2)):
        clique.add_edge(u, v, weight=position + 1)  # distinct weights: the answer is unique
    assert clique.number_of_edges() == 10
    for n in (1, 2, 3):
        filtered = backbone(clique, "naive-top", threshold=n)
        # The union of five nodes' top-n lists, so at most 5n edges and never fewer than n.
        assert filtered.number_of_edges() <= 5 * n
        assert min(degree for _, degree in filtered.degree()) >= n
        assert filtered.graph["backbone_threshold"] == n
    # n = 4 is every edge of a 5-clique, since every node has exactly four.
    assert backbone(clique, "naive-top", threshold=4).number_of_edges() == 10


def test_the_naive_top_row_carries_the_minimum_degree_objection(hub_and_clique: nx.Graph) -> None:
    """The crime is measured on the result, not only asserted (p. 384)."""
    rows = {row.method: row for row in compare_backbones(hub_and_clique, methods=["naive-top"])}
    note = rows["naive-top"].note
    assert "minimum degree of this backbone is 1" in note
    assert "heinous crime against the God of power law degree distributions" in note


# ------------------------------------------------------- the backbone reaches both comparisons


def test_compare_windows_filters_both_builds_with_the_same_backbone(
    layered: InMemoryGraphStore,
) -> None:
    """Two backbones would be two networks, so `sna compare` applies one to both builds."""
    plain = compare_windows(
        layered,
        "test-layers",
        "entities",
        min_weight=1,
        where={"region": "north"},
        where2={"region": "south"},
        runs=2,
        seed=1,
    )
    filtered = compare_windows(
        layered,
        "test-layers",
        "entities",
        min_weight=1,
        where={"region": "north"},
        where2={"region": "south"},
        backbone="noise-corrected",
        alpha=0.5,
        runs=2,
        seed=1,
    )
    for window, unfiltered in ((filtered.a, plain.a), (filtered.b, plain.b)):
        assert window.graph.graph["backbone_method"] == "noise-corrected"
        assert window.graph.number_of_edges() <= unfiltered.graph.number_of_edges()
        for _, _, data in window.graph.edges(data=True):
            assert data["p_value"] <= 0.5

    # And it bites on both sides: over the whole corpus, where the same three co-mention edges
    # are built twice, the same alpha takes the same two away from each.
    whole = compare_windows(layered, "test-layers", "entities", min_weight=1, runs=2, seed=1)
    cut = compare_windows(
        layered,
        "test-layers",
        "entities",
        min_weight=1,
        backbone="noise-corrected",
        alpha=0.5,
        runs=2,
        seed=1,
    )
    assert whole.a.graph.number_of_edges() == whole.b.graph.number_of_edges() == 3
    assert cut.a.graph.number_of_edges() == cut.b.graph.number_of_edges() == 1

    # Both frames name it, because each window is a sample of its own and the report prints
    # one frame per window.
    report = render_comparison(filtered)
    assert report.count("noise-corrected") >= 2
    assert "§27.6" in filtered.a.graph.graph["frame"]
    assert "§27.6" in filtered.b.graph.graph["frame"]
