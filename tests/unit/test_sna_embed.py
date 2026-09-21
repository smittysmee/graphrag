"""The §42.1 embedding contract, spectral embeddings and pooling, on data with a known answer."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.cluster import gmm, kmeans, sbm_communities
from graphrag.sna.cluster import spectral_embedding as _cluster_spectral_embedding
from graphrag.sna.embed import (
    Embedding,
    deepwalk,
    metapath2vec,
    node2vec,
    node2vec_scores,
    pool,
    spectral,
)
from graphrag.sna.experiment import evaluate_predictor, holdout, random_scores
from graphrag.sna.matrices import eigenpairs, laplacian


def _two_cliques(sizes: tuple[int, int]) -> tuple[nx.Graph, list[str]]:
    """Two disjoint cliques of the given sizes, joined by no edge at all."""
    graph = nx.Graph()
    order: list[str] = []
    for block, size in enumerate(sizes):
        nodes = [f"g{block}n{i}" for i in range(size)]
        order += nodes
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                graph.add_edge(u, v, weight=1)
    return graph, order


def _asymmetric_graph() -> tuple[nx.Graph, list[str]]:
    """A small connected graph with a simple (non-degenerate) spectrum: no two nodes are
    interchangeable under any automorphism, so its spectral embedding has no rotation ambiguity,
    only the sign ambiguity :func:`graphrag.sna.embed.spectral` already canonicalises."""
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("A", "C")])
    return graph, ["A", "B", "C", "D", "E"]


def _clique_with_detached_pairs(blob: int, pairs: int) -> nx.Graph:
    """A ``blob``-node clique plus ``pairs`` detached two-node components: the regression fixture.
    The clique's internal eigenspace is exactly ``(blob - 1)``-fold degenerate (a clique has no
    internal structure to separate), which is what makes a dense, deterministic eigensolver's
    choice within it an artefact rather than a reading of the network."""
    graph = nx.Graph()
    for i in range(blob):
        for j in range(i + 1, blob):
            graph.add_edge(f"b{i}", f"b{j}", weight=1)
    for p in range(pairs):
        graph.add_edge(f"p{p}a", f"p{p}b", weight=1)
    return graph


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _planted_two_mode() -> nx.Graph:
    """Two speaker groups of three, two entities each, sharing no entity at all: a speaker-entity
    two-mode graph that is really two disjoint complete bipartite pieces, so a
    speaker-entity-speaker walk from either group can never reach the other."""
    graph = nx.Graph()
    groups = {
        "a": (["sa0", "sa1", "sa2"], ["ea0", "ea1"]),
        "b": (["sb0", "sb1", "sb2"], ["eb0", "eb1"]),
    }
    for speakers, entities in groups.values():
        for speaker in speakers:
            graph.add_node(speaker, mode="speaker")
        for entity in entities:
            graph.add_node(entity, mode="entity")
        for speaker in speakers:
            for entity in entities:
                graph.add_edge(speaker, entity, weight=1)
    return graph


def _twin_pair_graph() -> tuple[nx.Graph, list[str]]:
    """Two cliques of five plus one pair of "twin" nodes (``t1``, ``t2``) wired to the same three
    anchors and not to each other -- the minimal structural symmetry §42.3's degenerate-eigenspace
    caveat describes at its smallest: swapping the twins is a graph automorphism, so their
    Laplacian rows are related by a permutation and the antisymmetric direction between them ties
    an eigenvalue against a neighbour closely enough to trigger :func:`spectral`'s eigengap
    truncation at ``dims=4``."""
    graph, order = _two_cliques((5, 5))
    anchors = order[:3]
    for twin in ("t1", "t2"):
        for anchor in anchors:
            graph.add_edge(twin, anchor, weight=1)
    return graph, [*order, "t1", "t2"]


# ----------------------------------------------------------------------------- Embedding contract


def test_embedding_rejects_a_row_count_that_does_not_match_the_node_order() -> None:
    with pytest.raises(ValueError, match="one row per node"):
        Embedding(vectors=np.zeros((2, 2)), nodes=["a", "b", "c"], method="test", provenance={})


def test_embedding_rejects_a_repeated_node() -> None:
    with pytest.raises(ValueError, match="must not repeat"):
        Embedding(vectors=np.zeros((2, 2)), nodes=["a", "a"], method="test", provenance={})


def test_embedding_dims_and_vector_lookup() -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0]])
    embedding = Embedding(vectors=vectors, nodes=["a", "b"], method="test", provenance={})
    assert embedding.dims == 2
    assert list(embedding.vector("a")) == [1.0, 2.0]
    assert list(embedding.vector("b")) == [3.0, 4.0]
    with pytest.raises(KeyError):
        embedding.vector("z")


# ----------------------------------------------------------------------------- spectral (§42.3)


def test_spectral_separates_two_disjoint_cliques_linearly() -> None:
    """The known answer: two cliques with no edge between them embed to two distinct constants
    in the first non-trivial dimension, because that dimension is (up to the trivial discard) an
    indicator of which clique a node is in."""
    graph, order = _two_cliques((3, 5))
    embedding = spectral(graph, dims=1, nodes=order)
    values = embedding.vectors[:, 0]
    first_clique, second_clique = values[:3], values[3:]
    # Every node agrees with its own clique and disagrees with the other one.
    assert np.allclose(first_clique, first_clique[0])
    assert np.allclose(second_clique, second_clique[0])
    assert not np.isclose(first_clique[0], second_clique[0])


def test_spectral_values_equal_matrices_eigenpairs_on_the_named_laplacian() -> None:
    """§42.3 names the matrix: the symmetric normalised Laplacian. The contract's vectors are
    exactly its eigenvectors (up to the sign :func:`graphrag.sna.embed.spectral` canonicalises,
    which does not change what the numbers mean, only which of the two equally valid signs a
    solver returned)."""
    graph, order = _asymmetric_graph()
    embedding = spectral(graph, dims=2, nodes=order)
    sym_laplacian, _ = laplacian(graph, nodes=order, kind="symmetric")
    values, vectors = eigenpairs(sym_laplacian, largest=False)
    assert np.allclose(np.abs(embedding.vectors), np.abs(vectors[:, 1:3]), atol=1e-8)
    assert embedding.provenance["eigenvalues"] == pytest.approx(values[1:3].tolist())


def test_spectral_refuses_a_dims_the_graph_cannot_support() -> None:
    graph, order = _asymmetric_graph()
    with pytest.raises(ValueError, match="dims <= 4"):
        spectral(graph, dims=10, nodes=order)


def test_spectral_is_deterministic_regardless_of_seed() -> None:
    """Unlike the ``scikit-learn`` solver this replaces, ``numpy.linalg.eigh`` has no random
    initialisation, so ``seed`` is recorded for the caller's report but changes nothing."""
    graph, order = _asymmetric_graph()
    first = spectral(graph, dims=2, nodes=order, seed=1)
    second = spectral(graph, dims=2, nodes=order, seed=99)
    assert np.array_equal(first.vectors, second.vectors)


def test_spectral_flattens_a_directed_graph_rather_than_raising() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "c", weight=1)
    embedding = spectral(graph, dims=1)
    assert embedding.vectors.shape == (3, 1)


def test_spectral_does_not_truncate_a_clean_cut() -> None:
    """The negative case: five non-trivial dimensions is exactly what six components (one clique,
    five pairs) offer before the clique's own degenerate eigenspace starts, so nothing is cut."""
    graph = _clique_with_detached_pairs(40, 5)
    order = list(graph.nodes)
    embedding = spectral(graph, dims=5, nodes=order)
    assert embedding.dims == 5
    assert embedding.provenance["dims_kept"] == embedding.provenance["dims_requested"] == 5
    assert embedding.provenance["eigengap_eigenvalue"] is None


def test_spectral_truncates_at_an_eigengap_rather_than_cutting_a_degenerate_block() -> None:
    """The regression this fixes: requesting more than the six components' five separating
    dimensions cuts into the 40-node clique's 39-fold degenerate internal eigenspace, which has
    no canonical basis, so every dimension in that block is dropped rather than an arbitrary few
    of them."""
    graph = _clique_with_detached_pairs(40, 5)
    order = list(graph.nodes)
    embedding = spectral(graph, dims=8, nodes=order)
    assert embedding.dims == 5
    assert embedding.provenance["dims_requested"] == 8
    assert embedding.provenance["dims_kept"] == 5
    assert embedding.provenance["eigengap_eigenvalue"] == pytest.approx(40 / 39, rel=1e-6)


def test_spectral_keeps_every_asked_dimension_when_components_outnumber_it() -> None:
    """The regression the smoke test found: a network in more components than ``dims + 1`` has a
    zero eigenvalue repeated past the cut, which used to read as a tied block and collapse the
    embedding to one arbitrary column. The zero block has a canonical basis -- one indicator per
    component -- so the cut is honoured and each kept column names one smaller component."""
    graph = _clique_with_detached_pairs(10, 11)
    order = list(graph.nodes)
    embedding = spectral(graph, dims=4, nodes=order)
    assert embedding.dims == 4
    assert embedding.provenance["eigengap_eigenvalue"] is None
    assert embedding.provenance["components"] == 12
    assert embedding.provenance["component_dims"] == 4
    components = [set(part) for part in nx.connected_components(graph)]
    for column in embedding.vectors.T:
        support = {order[i] for i in np.flatnonzero(np.abs(column) > 1e-12)}
        assert support in components
        assert np.all(column >= 0)
    # The largest component is the discarded one, so it sits at the origin of every column.
    clique_rows = [i for i, node in enumerate(order) if node.startswith("b")]
    assert np.allclose(embedding.vectors[clique_rows], 0.0)


def test_spectral_component_distances_survive_a_node_reordering() -> None:
    """Which of several equal-size components gets which column follows node order, but the
    geometry a clusterer reads -- every pairwise distance -- does not."""
    graph = _clique_with_detached_pairs(10, 11)
    order = sorted(graph.nodes)
    reordered = list(reversed(order))
    first = spectral(graph, dims=4, nodes=order)
    second = spectral(graph, dims=4, nodes=reordered)
    position = {node: i for i, node in enumerate(reordered)}
    aligned = second.vectors[[position[node] for node in order]]

    def distances(vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)

    # Four of the eleven pairs get a column either way, just not the same four, so compare the
    # distribution of distances rather than the matrix entry by entry.
    assert np.allclose(
        np.sort(distances(first.vectors).ravel()), np.sort(distances(aligned).ravel())
    )


def test_spectral_reads_inside_components_once_the_component_block_is_used_up() -> None:
    """Past ``components - 1`` dimensions the embedding continues into the non-zero spectrum,
    so the columns after the component indicators carry structure inside a component."""
    graph = nx.path_graph(6)
    graph.add_edge(10, 11)
    graph.add_edge(20, 21)
    graph = nx.relabel_nodes(graph, {n: f"n{n}" for n in graph.nodes})
    order = list(graph.nodes)
    embedding = spectral(graph, dims=3, nodes=order)
    assert embedding.dims == 3
    assert embedding.provenance["component_dims"] == 2
    inside = embedding.vectors[:, 2]
    path_rows = [order.index(f"n{i}") for i in range(6)]
    # The Fiedler vector of a path splits it in the middle: one half positive, the other
    # negative, and nothing outside the path.
    path_values = inside[path_rows]
    assert np.all(np.sign(path_values[:3]) == np.sign(path_values[0]))
    assert np.all(np.sign(path_values[3:]) == -np.sign(path_values[0]))
    other_rows = [i for i in range(len(order)) if i not in path_rows]
    assert np.allclose(inside[other_rows], 0.0)


def test_spectral_truncation_never_drops_below_one_dimension() -> None:
    """A pure clique's entire non-trivial spectrum but the last dimension is one degenerate
    block: asking for less than the graph's maximum still cuts into it, and the floor keeps at
    least one dimension rather than none."""
    graph = nx.complete_graph(6)
    graph = nx.relabel_nodes(graph, {i: f"n{i}" for i in range(6)})
    order = [f"n{i}" for i in range(6)]
    embedding = spectral(graph, dims=3, nodes=order)
    assert embedding.dims == 1
    assert embedding.provenance["dims_kept"] == 1


def test_gmm_does_not_crash_on_a_graph_with_twin_nodes() -> None:
    """Spot check: the blockmodel restart path (``cluster.spectral_embedding``) and GMM must
    keep working when the eigengap truncation triggers on ordinary clustering input, not only
    inside :func:`graphrag.sna.embed.spectral` itself."""
    graph, order = _twin_pair_graph()
    embedding = _cluster_spectral_embedding(graph, dims=4, seed=1)
    assert embedding.shape[0] == len(order)
    result = gmm(embedding, 2, seed=1)
    assert len(result.labels) == len(order)


def test_sbm_communities_does_not_crash_on_a_graph_with_twin_nodes() -> None:
    """Spot check: ``sbm_communities``'s restart loop (cluster.py) calls
    ``spectral_embedding`` once per restart; it must not choke on a graph whose eigengap
    truncation kicks in."""
    graph, order = _twin_pair_graph()
    fit = sbm_communities(graph, k=2, seed=1, restarts=3)
    assert sum(len(block) for block in fit.communities) == len(order)


# ----------------------------------------------------------------------------- pool (§42.4)


def test_pool_mean_sum_max_on_a_hand_computed_three_node_example() -> None:
    vectors = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    embedding = Embedding(vectors=vectors, nodes=["n0", "n1", "n2"], method="test", provenance={})
    assert list(pool(embedding, "mean")) == pytest.approx([3.0, 4.0])
    assert list(pool(embedding, "sum")) == pytest.approx([9.0, 12.0])
    assert list(pool(embedding, "max")) == pytest.approx([5.0, 6.0])


def test_pool_rejects_an_unknown_kind() -> None:
    embedding = Embedding(vectors=np.zeros((1, 1)), nodes=["a"], method="test", provenance={})
    with pytest.raises(ValueError, match="kind must be one of"):
        pool(embedding, "banana")  # type: ignore[arg-type]


def test_pool_rejects_an_embedding_with_no_nodes() -> None:
    embedding = Embedding(vectors=np.zeros((0, 2)), nodes=[], method="test", provenance={})
    with pytest.raises(ValueError, match="at least one node"):
        pool(embedding)


def test_pool_of_isomorphic_graphs_agrees_and_an_extra_edge_changes_it() -> None:
    graph, order = _asymmetric_graph()
    mapping = {"A": "X", "B": "Y", "C": "Z", "D": "W", "E": "V"}
    relabelled = nx.relabel_nodes(graph, mapping)
    relabelled_order = [mapping[node] for node in order]

    pooled_original = pool(spectral(graph, dims=2, nodes=order), "mean")
    pooled_relabelled = pool(spectral(relabelled, dims=2, nodes=relabelled_order), "mean")
    assert np.allclose(pooled_original, pooled_relabelled)

    with_extra_edge = graph.copy()
    with_extra_edge.add_edge("B", "D", weight=1)
    pooled_extra = pool(spectral(with_extra_edge, dims=2, nodes=order), "mean")
    assert not np.allclose(pooled_original, pooled_extra)


# --------------------------------------------------------- random-walk embeddings (ch. 43)


def test_deepwalk_separates_two_disjoint_cliques_linearly() -> None:
    """The known answer: with no edge between the two cliques, no walk from one ever reaches the
    other, so K-means with k=2 on the embedding recovers the two cliques exactly and every
    within-clique cosine similarity beats every across-clique one."""
    graph, order = _two_cliques((6, 6))
    embedding = deepwalk(
        graph, dims=4, walks_per_node=6, walk_length=10, epochs=4, seed=3, nodes=order
    )
    result = kmeans(embedding.vectors, 2, seed=3)
    first, second = set(result.labels[:6]), set(result.labels[6:])
    assert len(first) == 1
    assert len(second) == 1
    assert first != second

    within = [
        _cos(embedding.vector(order[i]), embedding.vector(order[j]))
        for i in range(6)
        for j in range(6)
        if i != j
    ]
    across = [
        _cos(embedding.vector(order[i]), embedding.vector(order[j]))
        for i in range(6)
        for j in range(6, 12)
    ]
    assert min(within) > max(across)


def test_deepwalk_equals_node2vec_at_p_equals_q_equals_one() -> None:
    """§43.2, p. 625: 'DeepWalk is equal to Node2Vec when we set p = q = 1'."""
    graph, order = _two_cliques((4, 4))
    budget = {"walks_per_node": 5, "walk_length": 8, "epochs": 3}
    from_deepwalk = deepwalk(graph, dims=3, seed=9, nodes=order, **budget)
    from_node2vec = node2vec(graph, dims=3, p=1.0, q=1.0, seed=9, nodes=order, **budget)
    assert np.array_equal(from_deepwalk.vectors, from_node2vec.vectors)


def test_node2vec_same_seed_reproduces_the_same_vectors() -> None:
    graph, order = _two_cliques((4, 4))
    budget = {"walks_per_node": 5, "walk_length": 8, "epochs": 3}
    first = node2vec(graph, dims=3, p=1.5, q=0.5, seed=5, nodes=order, **budget)
    second = node2vec(graph, dims=3, p=1.5, q=0.5, seed=5, nodes=order, **budget)
    assert np.array_equal(first.vectors, second.vectors)


def test_node2vec_bfs_like_keeps_the_two_bells_more_separated_than_dfs_like() -> None:
    """§43.2: high ``q`` biases the walk toward shared neighbours (BFS-like, staying local);
    high ``p`` only discourages backtracking, letting the walk wander across the bridge more
    freely (DFS-like). Measured directly on the walks themselves (not shown here, ATL-43's own
    exploration), a BFS-like walk (``p=1, q=50``) spends about 3% of its steps on the barbell's
    other bell against about 35% for a DFS-like one (``p=50, q=1``) -- the DFS-like walk crosses
    the bridge *more* often but, once across, is far less likely to bounce straight back (a low
    ``p`` no longer discourages leaving), so it dwells in long, unbroken stretches on one side
    while the BFS-like walk's rare crossings are short round trips. A window no wider than that
    dwell length reads mostly single-bell context either way; the embedding is asserted with a
    ``window`` of 2, well under either walk's dwell length, so the resulting separation tracks
    the walks' own bias rather than an accident of how the window happens to straddle a crossing.
    The ordering is asserted over several seeds, not a fixed value, exactly as the brief asks."""
    graph = nx.barbell_graph(5, 0)
    bell = {node: (0 if node < 5 else 1) for node in graph.nodes}

    def separation(p: float, q: float, seed: int) -> float:
        embedding = node2vec(
            graph,
            dims=4,
            p=p,
            q=q,
            walks_per_node=8,
            walk_length=16,
            window=2,
            epochs=4,
            seed=seed,
        )
        same, cross = [], []
        for u in graph.nodes:
            for v in graph.nodes:
                if u == v:
                    continue
                sim = _cos(embedding.vector(u), embedding.vector(v))
                (same if bell[u] == bell[v] else cross).append(sim)
        return float(np.mean(same) - np.mean(cross))

    for seed in (1, 2, 3):
        bfs_like = separation(p=1.0, q=50.0, seed=seed)
        dfs_like = separation(p=50.0, q=1.0, seed=seed)
        assert bfs_like > dfs_like, f"seed {seed}: bfs-like {bfs_like} <= dfs-like {dfs_like}"


def test_node2vec_gives_an_isolated_node_a_row_and_counts_it_visited() -> None:
    """Every node seeds its own walk (``for start in range(n)``), so even a node with no
    neighbours appears in its own (length-1) walk and is never counted as unvisited -- unlike
    :func:`metapath2vec`, whose walks seed only from one type."""
    graph, order = _two_cliques((4, 4))
    graph.add_node("isolated")
    order = [*order, "isolated"]
    embedding = node2vec(
        graph, dims=2, nodes=order, seed=1, walks_per_node=5, walk_length=8, epochs=3
    )
    assert embedding.vectors.shape == (9, 2)
    assert embedding.provenance["unvisited_nodes"] == 0


def test_node2vec_on_an_empty_graph_returns_an_empty_embedding() -> None:
    embedding = node2vec(nx.Graph(), dims=3)
    assert embedding.vectors.shape == (0, 3)
    assert embedding.nodes == []


def test_node2vec_rejects_dims_below_one() -> None:
    graph, order = _two_cliques((3, 3))
    with pytest.raises(ValueError, match="dims must be at least 1"):
        node2vec(graph, dims=0, nodes=order)


def test_node2vec_rejects_a_non_positive_p_or_q() -> None:
    graph, order = _two_cliques((3, 3))
    with pytest.raises(ValueError, match="p and q must be positive"):
        node2vec(graph, dims=2, p=0.0, nodes=order)
    with pytest.raises(ValueError, match="p and q must be positive"):
        node2vec(graph, dims=2, q=-1.0, nodes=order)


def test_node2vec_refuses_a_graph_over_max_nodes() -> None:
    graph, order = _two_cliques((3, 3))
    with pytest.raises(ValueError, match="needs at most"):
        node2vec(graph, dims=2, nodes=order, max_nodes=3)


def test_node2vec_flattens_a_directed_graph_rather_than_raising() -> None:
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1)
    graph.add_edge("b", "c", weight=1)
    embedding = node2vec(graph, dims=1, walks_per_node=3, walk_length=5, seed=1)
    assert embedding.vectors.shape == (3, 1)


def test_node2vec_scores_beats_the_random_baseline_on_planted_communities() -> None:
    """§43.4's link-prediction application: cosine of the embedding, measured through
    :func:`graphrag.sna.experiment.evaluate_predictor` on a network where community, not degree,
    predicts the missing edges -- the same fixture shape ``test_sna_predict.py`` uses for chapter
    23's neighbourhood scorers, at the same size (two 15-node communities). Even at that size a
    13-positive test set lets the random scorer wander well off 0.5, so the ordering is checked
    against the largest of several draws of it rather than the constant."""
    graph = nx.relabel_nodes(
        nx.planted_partition_graph(2, 15, 0.6, 0.05, seed=3), lambda n: f"n{n}"
    )
    split = holdout(graph, share=0.1, seed=1)
    report = evaluate_predictor(
        split,
        lambda train: node2vec_scores(
            train, dims=6, walks_per_node=8, walk_length=15, epochs=5, window=3, seed=1
        ),
        name="node2vec",
        k=10,
    )
    baselines = [
        evaluate_predictor(split, lambda train, s=s: random_scores(train, seed=s), k=10).auc
        for s in (1, 2, 3)
    ]
    assert report.auc > 0.75
    assert report.auc > max(baselines)


# ------------------------------------------------------------------------- metapath2vec (§43.3)


def test_metapath2vec_separates_two_disjoint_speaker_groups() -> None:
    """The known answer: two speaker groups sharing no entity, so a speaker-entity-speaker walk
    from either group can never reach the other, and every within-group cosine beats every
    across-group one."""
    graph = _planted_two_mode()
    embedding = metapath2vec(graph, dims=4, walks_per_node=8, walk_length=10, epochs=4, seed=11)
    group_a = ["sa0", "sa1", "sa2"]
    group_b = ["sb0", "sb1", "sb2"]
    within = [
        _cos(embedding.vector(u), embedding.vector(v)) for u in group_a for v in group_a if u != v
    ]
    across = [_cos(embedding.vector(u), embedding.vector(v)) for u in group_a for v in group_b]
    assert min(within) > max(across)


def test_metapath2vec_reports_an_entity_no_walk_ever_reaches() -> None:
    graph = _planted_two_mode()
    graph.add_node("orphan_entity", mode="entity")
    embedding = metapath2vec(graph, dims=2, walks_per_node=5, walk_length=6, seed=1)
    assert embedding.provenance["unvisited_nodes"] == 1
    assert len(embedding.nodes) == 11


def test_metapath2vec_refuses_a_metapath_that_does_not_close() -> None:
    graph = _planted_two_mode()
    with pytest.raises(ValueError, match="starts and ends with the same type"):
        metapath2vec(graph, dims=2, metapath=("speaker", "entity"))


def test_metapath2vec_refuses_a_graph_with_no_mode_attribute() -> None:
    graph, order = _two_cliques((3, 3))
    with pytest.raises(ValueError, match="carry a 'mode' attribute"):
        metapath2vec(graph, dims=2, nodes=order)


def test_metapath2vec_refuses_when_no_node_starts_the_metapath() -> None:
    graph = _planted_two_mode()
    with pytest.raises(ValueError, match="no node carries mode"):
        metapath2vec(graph, dims=2, metapath=("nobody", "entity", "nobody"))
