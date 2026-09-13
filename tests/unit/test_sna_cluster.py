"""Louvain, K-means, Gaussian mixtures and the null model, on data with a planted answer."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.cluster import (
    choose_k_gmm,
    choose_k_kmeans,
    compare_partitions,
    gmm,
    kmeans,
    louvain,
    null_model_modularity,
    spectral_embedding,
)

PLANTED = [
    ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7"],
    ["n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15"],
    ["n16", "n17", "n18", "n19", "n20", "n21", "n22", "n23"],
]


@pytest.fixture
def planted() -> nx.Graph:
    """Three cliques joined by one edge each: the communities are not in doubt."""
    graph = nx.Graph()
    for block in PLANTED:
        for i, a in enumerate(block):
            for b in block[i + 1 :]:
                graph.add_edge(a, b, weight=3)
    graph.add_edge("n7", "n8", weight=1)
    graph.add_edge("n15", "n16", weight=1)
    return graph


@pytest.fixture
def blobs() -> np.ndarray:
    """Three well-separated clouds of points in two dimensions."""
    rng = np.random.default_rng(7)
    centres = np.array([[0.0, 0.0], [12.0, 0.0], [6.0, 12.0]])
    return np.vstack([centre + rng.normal(0, 0.4, size=(30, 2)) for centre in centres])


# ----------------------------------------------------------------------------- Louvain


def test_louvain_recovers_the_planted_communities(planted: nx.Graph) -> None:
    result = louvain(planted, seed=1, runs=10)
    nodes = list(planted.nodes)
    agreement = compare_partitions(result.labels(nodes), _planted_labels(nodes))
    assert agreement["adjusted_rand_index"] > 0.9
    assert result.modularity > 0.4
    assert result.stability > 0.9  # every seed found the same thing
    assert len(result.seeds) == 10 and len(result.modularities) == 10


def test_louvain_is_reproducible_from_a_seed(planted: nx.Graph) -> None:
    first = louvain(planted, seed=42, runs=3)
    second = louvain(planted, seed=42, runs=3)
    assert first.communities == second.communities
    assert first.modularities == second.modularities


def test_louvain_resolution_changes_how_many_groups_it_finds(planted: nx.Graph) -> None:
    coarse = louvain(planted, resolution=0.2, seed=1, runs=3)
    fine = louvain(planted, resolution=2.0, seed=1, runs=3)
    assert len(coarse.communities) <= len(fine.communities)


def test_louvain_on_an_empty_graph_returns_nothing_rather_than_raising() -> None:
    result = louvain(nx.Graph())
    assert result.communities == [] and result.stability == 1.0


# ----------------------------------------------------------------------------- null model


def test_null_model_separates_planted_structure_from_a_random_graph(planted: nx.Graph) -> None:
    structured = louvain(planted, seed=1, runs=3)
    against_null = null_model_modularity(planted, structured.communities, samples=30, seed=1)
    assert against_null.samples > 0
    assert against_null.z_score > 3.0
    assert "beyond chance" in against_null.verdict

    random_graph = nx.gnm_random_graph(24, planted.number_of_edges(), seed=5)
    nx.set_edge_attributes(random_graph, 1, "weight")
    found = louvain(random_graph, seed=1, runs=3)
    noise = null_model_modularity(random_graph, found.communities, samples=30, seed=1)
    # Louvain always finds *something*; the null model is what says it means nothing.
    assert noise.z_score < against_null.z_score
    assert noise.z_score < 3.0


def test_null_model_reports_that_a_tiny_network_is_not_testable() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    result = null_model_modularity(graph, [["a", "b"]], samples=5, seed=1)
    assert result.samples == 0
    assert "not testable" in result.verdict


# ----------------------------------------------------------------------------- vectors


def test_spectral_embedding_is_aligned_to_the_node_order(planted: nx.Graph) -> None:
    embedded = spectral_embedding(planted, dims=4, seed=1)
    assert embedded.shape == (planted.number_of_nodes(), 4)
    assert len(list(planted.nodes)) == embedded.shape[0]


def test_spectral_embedding_clamps_dims_to_the_graph_size() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "c", weight=1)
    assert spectral_embedding(graph, dims=50, seed=1).shape == (3, 2)


def test_kmeans_and_choose_k_find_three_blobs(blobs: np.ndarray) -> None:
    choice = choose_k_kmeans(blobs, range(2, 8), seed=1)
    assert choice.best_silhouette_k == 3
    assert choice.elbow_k == 3
    result = kmeans(blobs, 3, seed=1)
    assert len(set(result.labels)) == 3
    assert result.silhouette > 0.8
    assert sorted(np.bincount(result.labels).tolist()) == [30, 30, 30]


def test_gmm_finds_three_blobs_and_reports_soft_membership(blobs: np.ndarray) -> None:
    choice = choose_k_gmm(blobs, range(1, 7), seed=1)
    assert choice.best_bic_k == 3
    mixture = gmm(blobs, 3, seed=1)
    assert mixture.converged
    assert len(set(mixture.labels)) == 3
    assert len(mixture.responsibilities) == len(blobs)
    assert all(len(row) == 3 for row in mixture.responsibilities)
    assert sum(mixture.responsibilities[0]) == pytest.approx(1.0)
    # A point at the heart of a blob belongs almost entirely to one component.
    assert max(mixture.responsibilities[0]) > 0.99


def test_gmm_rejects_an_unknown_covariance(blobs: np.ndarray) -> None:
    with pytest.raises(ValueError, match="covariance must be one of"):
        gmm(blobs, 2, covariance="banana", seed=1)


def test_kmeans_and_gmm_are_reproducible_from_a_seed(blobs: np.ndarray) -> None:
    assert kmeans(blobs, 3, seed=9).labels == kmeans(blobs, 3, seed=9).labels
    assert gmm(blobs, 3, seed=9).labels == gmm(blobs, 3, seed=9).labels


def test_spectral_then_kmeans_recovers_the_planted_communities(planted: nx.Graph) -> None:
    """The documented bridge from edges to vectors has to actually work."""
    nodes = list(planted.nodes)
    labels = kmeans(spectral_embedding(planted, dims=3, seed=1), 3, seed=1).labels
    agreement = compare_partitions(labels, _planted_labels(nodes))
    assert agreement["adjusted_rand_index"] > 0.9


def test_compare_partitions_scores_identical_and_random_labellings() -> None:
    a = [0, 0, 1, 1, 2, 2]
    assert compare_partitions(a, a)["adjusted_rand_index"] == pytest.approx(1.0)
    assert compare_partitions(a, a)["normalized_mutual_information"] == pytest.approx(1.0)
    shuffled = [2, 2, 0, 0, 1, 1]  # same grouping, different names
    assert compare_partitions(a, shuffled)["adjusted_rand_index"] == pytest.approx(1.0)
    unrelated = compare_partitions(a, [0, 1, 0, 1, 0, 1])
    assert unrelated["adjusted_rand_index"] < 0.2


def _planted_labels(nodes: list[str]) -> list[int]:
    truth = {node: index for index, block in enumerate(PLANTED) for node in block}
    return [truth[node] for node in nodes]


def _blob_with_detached_pairs(blob: int, pairs: int) -> nx.Graph:
    graph = nx.Graph()
    for i in range(blob):
        for j in range(i + 1, blob):
            graph.add_edge(f"b{i}", f"b{j}", weight=1)
    for p in range(pairs):
        graph.add_edge(f"p{p}a", f"p{p}b", weight=1)
    graph.graph.update(min_weight=1, frame="a test network")
    return graph


@pytest.mark.parametrize("method", ["kmeans", "louvain"])
def test_a_grouping_that_only_found_the_components_says_so(method: str) -> None:
    """A near-perfect silhouette on a fragmented network usually means nothing was learned.

    Forty nodes in one clique plus five detached pairs: every method separates the pairs, which
    looks like a clean result and says nothing at all about the forty.
    """
    from graphrag.graph.memory_store import InMemoryGraphStore
    from graphrag.sna.analysis import run_analysis

    analysis = run_analysis(
        InMemoryGraphStore(),
        _blob_with_detached_pairs(40, 5),
        persona_id="test-pm",
        network="speakers",
        method=method,  # type: ignore[arg-type]
        k=None if method == "louvain" else 6,
        seed=1,
        runs=3,
        samples=5,
    )
    assert max(len(g) for g in analysis.groups) == 40
    assert any("mostly separates disconnected pieces" in note for note in analysis.notes)


def test_a_grouping_that_splits_the_main_component_stays_quiet() -> None:
    """The same shape below the threshold must not carry the warning."""
    from graphrag.graph.memory_store import InMemoryGraphStore
    from graphrag.sna.analysis import run_analysis

    analysis = run_analysis(
        InMemoryGraphStore(),
        _blob_with_detached_pairs(30, 4),  # 30 of 38 nodes, under the 80% threshold
        persona_id="test-pm",
        network="speakers",
        method="kmeans",
        k=5,
        seed=1,
    )
    assert not any("disconnected pieces" in note for note in analysis.notes)
