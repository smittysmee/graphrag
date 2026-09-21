"""Chapter 35, held to answers that existed before the code did.

Every partition here goes through :func:`graphrag.sna.evaluate.evaluate_partition`, per the
wave's own rule (ATL-36), and every claim is checked against a graph whose right answer is known
by construction:

* **Two disjoint cliques.** No edge connects them, so label percolation (§35.3) and Infomap's map
  equation (§35.2) have nothing to disagree about: any partition that splits a clique or merges
  the two costs strictly more (a code length above 0, a lower share of matching labels) than the
  one that keeps them apart, which is the known answer both are checked against.
* **A planted two-block graph** (moderate within-block density, sparse cross density, a fixed
  seed so it is reproducible): Walktrap's own distance (§35.2) is checked directly -- two nodes of
  one block must be closer than one node from each -- and the plain stochastic blockmodel's own
  likelihood (§35.1) is checked to prefer the planted partition over a random relabelling of the
  same block sizes.
* **A planted hub-in-a-block graph**: two blocks of otherwise low, even degree, each with one hub
  connected to nearly all of its own block and a slice of the other. §35.1's own point (p. 495)
  is that a plain, non-degree-corrected fit cannot explain a hub's degree except by giving it a
  block of its own, while a degree-corrected fit does not need to -- checked here as which block
  each hub ends up sharing a majority with, at a fixed ``k=2``, empirically confirmed stable
  across five seeds before being written down as a test.
* **Three planted snapshots**, one clique pair unchanged and one clique split at the second
  window: §35.4's own taxonomy (Figures 35.9-35.11) applied to a case whose split is not in
  doubt, matching the transitions this produces label for label.
* **A planted ring of cliques bridged by single edges** (the same shape ``test_sna_cluster.py``
  uses for Louvain): growing §35.5's local community from a seed in one clique must stop at that
  clique's boundary, and growing it inside a single isolated clique with nothing to bridge to must
  return the whole thing.
"""

from __future__ import annotations

import random

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.cluster import (
    LouvainResult,
    _binary_matrix,
    _block_weights,
    _plain_score,
    _sbm_free_parameters,
    infomap_communities,
    label_propagation,
    local_community,
    louvain,
    louvain_levels,
    map_equation,
    match_communities,
    sbm_communities,
    walktrap,
    walktrap_distance,
)
from graphrag.sna.evaluate import evaluate_partition


def _clique(nodes: list[str]) -> nx.Graph:
    graph = nx.Graph()
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            graph.add_edge(a, b, weight=1)
    return graph


@pytest.fixture
def two_cliques() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(_clique([f"a{i}" for i in range(6)]).edges(data=True))
    graph.add_edges_from(_clique([f"b{i}" for i in range(6)]).edges(data=True))
    return graph


@pytest.fixture
def bridged_cliques() -> tuple[nx.Graph, list[list[str]]]:
    blocks = [
        [f"n{i}" for i in range(0, 8)],
        [f"n{i}" for i in range(8, 16)],
        [f"n{i}" for i in range(16, 24)],
    ]
    graph = nx.Graph()
    for block in blocks:
        graph.add_edges_from(_clique(block).edges(data=True))
    graph.add_edge("n7", "n8", weight=1)
    graph.add_edge("n15", "n16", weight=1)
    return graph, blocks


def _planted_two_block(
    seed: int = 2, n: int = 9, p_in: float = 0.6, p_cross: float = 0.02
) -> tuple[nx.Graph, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    a = [f"a{i}" for i in range(n)]
    b = [f"b{i}" for i in range(n)]
    graph.add_nodes_from(a + b)
    for block in (a, b):
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                if rng.random() < p_in:
                    graph.add_edge(block[i], block[j], weight=1)
    for x in a:
        for y in b:
            if rng.random() < p_cross:
                graph.add_edge(x, y, weight=1)
    return graph, a, b


def _hub_two_block(cross: int = 6) -> tuple[nx.Graph, set[str], set[str]]:
    """Two blocks of nine ring-connected regulars plus one hub each; the hubs also carry
    ``cross`` edges into the other block, which is enough (empirically, across seeds 1-5) to make
    the plain Bernoulli fit peel both hubs into a block of their own, and not enough to move the
    degree-corrected one."""
    graph = nx.Graph()
    a = [f"a{i}" for i in range(9)]
    b = [f"b{i}" for i in range(9)]
    hub_a, hub_b = "hubA", "hubB"
    for block in (a, b):
        for i in range(len(block)):
            graph.add_edge(block[i], block[(i + 1) % len(block)], weight=1)
    for node in a:
        graph.add_edge(hub_a, node, weight=1)
    for node in b:
        graph.add_edge(hub_b, node, weight=1)
    for i in range(cross):
        graph.add_edge(hub_a, b[i], weight=1)
        graph.add_edge(hub_b, a[i], weight=1)
    return graph, set(a) | {hub_a}, set(b) | {hub_b}


def _hub_majority_block(fit_communities: list[list[str]], block: set[str], hub: str) -> bool:
    membership = {node: i for i, community in enumerate(fit_communities) for node in community}
    regulars = [membership[n] for n in block if n != hub]
    majority = max(set(regulars), key=regulars.count)
    return membership[hub] == majority


# ----------------------------------------------------------------------------- edgeless fix


def test_louvain_on_nodes_with_no_edges_returns_singletons_not_a_crash() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    result = louvain(graph, seed=1, runs=5)
    assert sorted(result.communities) == [["a"], ["b"], ["c"]]
    assert result.modularity == 0.0
    assert result.stability == 1.0
    assert len(result.seeds) == 5


def test_louvain_levels_on_nodes_with_no_edges_does_not_raise() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    dendrogram = louvain_levels(graph, seed=1)
    assert sorted(dendrogram.communities) == [["a"], ["b"], ["c"]]
    assert dendrogram.modularity == 0.0


# ------------------------------------------------------------------- §35.3 label propagation


def test_label_propagation_recovers_two_disjoint_cliques(two_cliques: nx.Graph) -> None:
    result = label_propagation(two_cliques, seed=1, runs=8)
    assert sorted(sorted(c) for c in result.communities) == [
        [f"a{i}" for i in range(6)],
        [f"b{i}" for i in range(6)],
    ]
    assert result.stability == 1.0
    scores = evaluate_partition(two_cliques, result.communities, seed=1)
    assert scores.modularity > 0.4


# ----------------------------------------------------------------------------- §35.2 random walks


def test_map_equation_prefers_the_correct_partition_over_the_whole_network(
    two_cliques: nx.Graph,
) -> None:
    """The whole network pays full entropy over all 12 equally visited nodes (``log2(12)``, no
    community prefixes to shorten anything), while the correct partition pays only ``log2(6)`` --
    entropy inside a six-node clique, no exit code at all since nothing crosses between them."""
    whole = map_equation(two_cliques, [list(two_cliques.nodes)])
    correct = map_equation(two_cliques, [[f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]])
    assert whole == pytest.approx(np.log2(12))
    assert correct == pytest.approx(np.log2(6))
    assert correct < whole


def test_infomap_recovers_two_disjoint_cliques(two_cliques: nx.Graph) -> None:
    result = infomap_communities(two_cliques, seed=1, runs=5)
    assert sorted(sorted(c) for c in result.communities) == [
        [f"a{i}" for i in range(6)],
        [f"b{i}" for i in range(6)],
    ]
    assert result.code_length == pytest.approx(np.log2(6))
    assert result.stability == 1.0
    scores = evaluate_partition(two_cliques, result.communities, seed=1)
    assert scores.modularity > 0.4


def test_infomap_refuses_a_network_above_its_node_guard() -> None:
    graph = nx.gnm_random_graph(250, 400, seed=1)
    with pytest.raises(ValueError, match="refuses a network"):
        infomap_communities(graph, max_nodes=200)


def test_walktrap_distance_is_smaller_within_a_block_than_across() -> None:
    graph, a, b = _planted_two_block(seed=2)
    distance, order = walktrap_distance(graph, steps=4)
    index = {node: i for i, node in enumerate(order)}
    within = distance[index[a[0]], index[a[1]]]
    across = distance[index[a[0]], index[b[0]]]
    assert within < across


def test_walktrap_recovers_the_planted_blocks() -> None:
    graph, a, b = _planted_two_block(seed=2)
    dendrogram = walktrap(graph, steps=4)
    assert len(dendrogram.communities) == 2
    assert {frozenset(c) for c in dendrogram.communities} == {frozenset(a), frozenset(b)}
    scores = evaluate_partition(graph, dendrogram.communities, seed=1)
    assert scores.modularity > 0.2


def test_walktrap_refuses_a_network_above_its_node_guard() -> None:
    graph = nx.gnm_random_graph(50, 80, seed=1)
    with pytest.raises(ValueError, match="refuses a network"):
        walktrap(graph, max_nodes=10)


# ----------------------------------------------------------------------------- §35.1 SBM


def test_sbm_plain_likelihood_prefers_the_planted_partition_over_a_random_one() -> None:
    graph, a, _b = _planted_two_block(seed=3)
    order = sorted(graph.nodes)
    truth = np.array([0 if n in a else 1 for n in order])
    shuffled = truth.copy()
    random.Random(1).shuffle(shuffled)
    binary = _binary_matrix(graph, order)
    e_truth, sizes_truth = _block_weights(binary, truth, 2)
    e_random, sizes_random = _block_weights(binary, shuffled, 2)
    assert _plain_score(e_truth, sizes_truth, 2) > _plain_score(e_random, sizes_random, 2)


def test_sbm_fit_recovers_the_planted_blocks_and_is_evaluable() -> None:
    graph, a, b = _planted_two_block(seed=2)
    fit = sbm_communities(graph, k=2, degree_corrected=True, seed=1, restarts=6)
    assert {frozenset(c) for c in fit.communities} == {frozenset(a), frozenset(b)}
    assert fit.implementation in {
        "spectral-initialised greedy likelihood fit",
        "graph-tool (minimize_blockmodel_dl)",
    }
    scores = evaluate_partition(graph, fit.communities, seed=1)
    assert scores.modularity > 0.2


def test_sbm_free_parameters_charges_the_degree_corrected_model_for_its_own_thetas() -> None:
    """Karrer-Newman's per-node ``theta``, one per node minus one constraint per block, is a cost
    the plain Bernoulli model does not pay (§35.1, p. 495) -- so the two models' BIC penalties at
    the same ``k`` must differ, and by exactly ``n - k``."""
    plain = _sbm_free_parameters(k=2, n=18, degree_corrected=False)
    corrected = _sbm_free_parameters(k=2, n=18, degree_corrected=True)
    assert plain == 2 * 3 / 2  # one connection probability per unordered block pair: 3
    assert corrected == plain + (18 - 2)


def test_sbm_automatic_k_selection_reports_a_bic_row_per_candidate() -> None:
    """No other test reaches ``sbm_communities(..., k=None)``; this is the path
    :func:`_sbm_free_parameters` was fixed for, so it is held to the exact BIC it should report."""
    graph, _a, _b = _planted_two_block(seed=2)
    n = graph.number_of_nodes()
    fit = sbm_communities(
        graph, k=None, k_range=(2, 3, 4), degree_corrected=True, seed=1, restarts=3
    )
    assert fit.k in {2, 3, 4}
    assert [row["k"] for row in fit.choices] == [2.0, 3.0, 4.0]
    for row in fit.choices:
        expected_params = _sbm_free_parameters(int(row["k"]), n, degree_corrected=True)
        expected_bic = -2 * row["log_likelihood"] + expected_params * np.log(n)
        assert row["bic"] == pytest.approx(expected_bic)
    best_row = min(fit.choices, key=lambda row: row["bic"])
    assert fit.k == int(best_row["k"])


def test_degree_corrected_sbm_keeps_the_hub_with_its_block_where_plain_sbm_does_not() -> None:
    graph, block_a, block_b = _hub_two_block()
    plain = sbm_communities(graph, k=2, degree_corrected=False, seed=1, restarts=8)
    dc = sbm_communities(graph, k=2, degree_corrected=True, seed=1, restarts=8)
    assert not _hub_majority_block(plain.communities, block_a, "hubA")
    assert _hub_majority_block(dc.communities, block_a, "hubA")
    assert _hub_majority_block(dc.communities, block_b, "hubB")
    evaluate_partition(graph, dc.communities, seed=1)  # every partition goes through ch. 36


def test_sbm_refuses_a_network_above_its_node_guard() -> None:
    graph = nx.gnm_random_graph(20, 30, seed=1)
    with pytest.raises(ValueError, match="refuses a network"):
        sbm_communities(graph, k=2, max_nodes=10)


# ----------------------------------------------------------------------------- §35.4 temporal


def test_temporal_matching_reports_a_split_and_then_stability() -> None:
    def snapshot_one() -> nx.Graph:
        graph = nx.Graph()
        graph.add_edges_from(_clique([f"a{i}" for i in range(8)]).edges(data=True))
        graph.add_edges_from(_clique([f"b{i}" for i in range(8)]).edges(data=True))
        return graph

    def snapshot_two() -> nx.Graph:
        graph = nx.Graph()
        graph.add_edges_from(_clique([f"a{i}" for i in range(4)]).edges(data=True))
        graph.add_edges_from(_clique([f"a{i}" for i in range(4, 8)]).edges(data=True))
        graph.add_edges_from(_clique([f"b{i}" for i in range(8)]).edges(data=True))
        return graph

    snapshots = [("t1", snapshot_one()), ("t2", snapshot_two()), ("t3", snapshot_two())]
    tracked = match_communities(snapshots, seed=1, runs=5)
    assert tracked.method == "louvain"

    first_to_second = [t for t in tracked.transitions if t.from_window == "t1"]
    splits = [t for t in first_to_second if t.kind == "split"]
    continues = [t for t in first_to_second if t.kind == "continue"]
    assert len(splits) == 2
    assert {t.from_label for t in splits} == {splits[0].from_label}  # both halves of one label
    assert len(continues) == 1

    second_to_third = [t for t in tracked.transitions if t.from_window == "t2"]
    assert len(second_to_third) == 3
    assert all(t.kind == "continue" and t.jaccard == pytest.approx(1.0) for t in second_to_third)

    for partition in tracked.partitions:
        evaluate_partition(nx.compose_all([g for _, g in snapshots]), partition, seed=1)


def test_temporal_matching_needs_at_least_two_snapshots() -> None:
    with pytest.raises(ValueError, match="at least two"):
        match_communities([("t1", nx.Graph())])


# ----------------------------------------------------------------------------- §35.5 local


def test_local_community_from_a_bridged_clique_returns_that_clique_and_stops(
    bridged_cliques: tuple[nx.Graph, list[list[str]]],
) -> None:
    graph, blocks = bridged_cliques
    result = local_community(graph, blocks[0][0], seed=1)
    assert sorted(result.members) == sorted(blocks[0])
    assert result.stopped == "local modularity optimum"
    scores = evaluate_partition(graph, [result.members], seed=1)
    assert scores.communities == 1


def test_local_community_in_an_isolated_clique_returns_the_whole_clique() -> None:
    graph = _clique([f"c{i}" for i in range(6)])
    result = local_community(graph, "c0", seed=1)
    assert sorted(result.members) == [f"c{i}" for i in range(6)]
    assert result.stopped == "explored the whole component"
    assert result.local_modularity == pytest.approx(1.0)


def test_local_community_respects_max_size(
    bridged_cliques: tuple[nx.Graph, list[list[str]]],
) -> None:
    graph, blocks = bridged_cliques
    result = local_community(graph, blocks[0][0], max_size=4, seed=1)
    assert len(result.members) <= 4
    assert result.stopped in {"reached the exploration limit", "local modularity optimum"}


def test_local_community_raises_on_a_seed_outside_the_network() -> None:
    graph = _clique(["a", "b", "c"])
    with pytest.raises(ValueError, match="not in the network"):
        local_community(graph, "z")


# ----------------------------------------------------------------------------- ATL-37 wiring sanity


def test_directed_communities_result_type_matches_louvains() -> None:
    """``analysis.py`` reuses the ``louvain_result`` field for ``--method directed``; this holds
    it to the type that field is declared with."""
    from graphrag.sna.cluster import directed_communities

    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "a", weight=1)
    result = directed_communities(graph, seed=1, runs=2)
    assert isinstance(result, LouvainResult)
