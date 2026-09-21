"""The matrix representations of chapter 8, held to answers the book states in advance.

Three kinds of known answer are used here, and none of them is "the function returned something":

* **Spectral theorems.** The combinatorial Laplacian's smallest eigenvalue is 0 with multiplicity
  equal to the number of connected components (§8.4); the symmetric normalised Laplacian's
  eigenvalues lie in [0, 2], reaching 2 only for a bipartite component; a stochastic matrix's
  rows sum to 1 and its leading eigenvalue is 1 (§8.2).
* **Identities between representations.** The oriented incidence matrix rebuilds the Laplacian as
  ``B @ B.T`` and the line graph as ``B.T @ B - 2I`` (§8.3); ``B @ B.T`` on a bipartite incidence
  matrix rebuilds the shared-neighbour weights that ``export.bipartite_projection`` counts in
  Python (§8.3, §26.1).
* **Published results on legendary graphs.** Zachary's karate club has 34 nodes, 78 edges and one
  component, and node 33 -- the club president -- leads its adjacency eigenvector; the Davis
  Southern Women network is 18 women by 14 events.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.export import bipartite_projection
from graphrag.sna.matrices import (
    adjacency,
    dense,
    edge_incidence,
    eigenpairs,
    incidence,
    laplacian,
    nmf,
    node_order,
    project,
    stochastic,
    svd,
)

#: (left, right) memberships whose projection can be counted by hand: a and b share x and y,
#: c shares only y with each of them.
PLANTED_PAIRS = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y"), ("c", "y")]


@pytest.fixture
def karate() -> nx.Graph:
    """Zachary's karate club: one component, 34 nodes, 78 edges, and a published ranking."""
    return nx.karate_club_graph()


@pytest.fixture
def two_components() -> nx.Graph:
    """A 4-clique and a 5-clique with nothing between them: the component count is planted."""
    return nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(5))


@pytest.fixture
def southern_women() -> list[tuple[str, str]]:
    """The Davis Southern Women two-mode edge list, as (woman, event) memberships."""
    graph = nx.davis_southern_women_graph()
    return [(u, v) for u, v in graph.edges if graph.nodes[u]["bipartite"] == 0]


# ----------------------------------------------------------------------- the node-order contract


def test_node_order_defaults_to_sorted_ids_whatever_the_insertion_order() -> None:
    first = nx.Graph()
    first.add_edges_from([("z", "a"), ("a", "m")])
    second = nx.Graph()
    second.add_edges_from([("m", "a"), ("a", "z")])
    assert node_order(first) == node_order(second) == ["a", "m", "z"]
    assert np.array_equal(dense(adjacency(first)[0]), dense(adjacency(second)[0]))


def test_an_explicit_node_order_lays_the_rows_out_that_way() -> None:
    graph = nx.Graph([("a", "b")])
    matrix, order = adjacency(graph, nodes=["b", "a"])
    assert order == ["b", "a"]
    assert matrix[0, 1] == 1.0 and matrix[0, 0] == 0.0


def test_a_node_order_may_be_a_subset_but_not_a_stranger_or_a_repeat() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c")])
    matrix, order = adjacency(graph, nodes=["a", "b"])
    assert order == ["a", "b"] and matrix.shape == (2, 2)
    with pytest.raises(ValueError, match="not in the graph"):
        adjacency(graph, nodes=["a", "zz"])
    with pytest.raises(ValueError, match="must not repeat"):
        adjacency(graph, nodes=["a", "a"])


# ----------------------------------------------------------------------------- §8.1 adjacency


def test_the_karate_adjacency_is_the_published_graph(karate: nx.Graph) -> None:
    matrix, nodes = adjacency(karate, weight=None)
    assert nodes == list(range(34))
    assert matrix.shape == (34, 34)
    assert matrix.sum() == 2 * 78  # §8.1: the entries sum to twice the edge count
    assert np.array_equal(matrix, matrix.T)  # undirected, so symmetric
    assert np.array_equal(np.diag(matrix), np.zeros(34))  # no self-loops, so a free diagonal
    assert np.array_equal(matrix.sum(axis=1), np.array([d for _, d in karate.degree()]))


def test_the_weighted_adjacency_carries_the_weights() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=6)
    matrix, _ = adjacency(graph)
    assert matrix[0, 1] == 6.0
    unweighted, _ = adjacency(graph, weight=None)
    assert unweighted[0, 1] == 1.0


def test_the_sparse_adjacency_holds_the_same_numbers(karate: nx.Graph) -> None:
    packed, nodes = adjacency(karate, sparse=True)
    spread, same_nodes = adjacency(karate)
    assert nodes == same_nodes
    assert np.allclose(dense(packed), spread)


# ----------------------------------------------------------------------------- §8.2 stochastic


def test_row_stochastic_rows_sum_to_one(karate: nx.Graph) -> None:
    matrix, _ = stochastic(karate, weight=None)
    assert np.allclose(matrix.sum(axis=1), np.ones(34))
    # §8.2's worked example: a node of degree five gives each of its edges probability 0.2.
    degree_five = next(n for n, d in karate.degree() if d == 5)
    row = matrix[degree_five]
    assert np.allclose(sorted(set(np.round(row, 10))), [0.0, 0.2])


def test_column_stochastic_columns_sum_to_one(karate: nx.Graph) -> None:
    matrix, _ = stochastic(karate, orientation="column", weight=None)
    assert np.allclose(matrix.sum(axis=0), np.ones(34))
    row_wise, _ = stochastic(karate, orientation="row", weight=None)
    assert np.allclose(matrix, row_wise.T)  # the two conventions are each other's transpose


def test_a_stochastic_matrix_is_not_symmetric_even_though_the_graph_is() -> None:
    """§8.2: u and v normalise by different degrees, so A.T != A for an undirected graph."""
    graph = nx.Graph([("hub", "a"), ("hub", "b"), ("a", "b")])
    matrix, nodes = stochastic(graph, weight=None)
    hub, a = nodes.index("hub"), nodes.index("a")
    assert matrix[hub, a] == pytest.approx(0.5)
    assert matrix[a, hub] == pytest.approx(0.5)
    graph.add_edge("hub", "c")
    lopsided, nodes = stochastic(graph, weight=None)
    hub, a = nodes.index("hub"), nodes.index("a")
    assert lopsided[hub, a] == pytest.approx(1 / 3)
    assert lopsided[a, hub] == pytest.approx(0.5)


def test_the_leading_eigenvalue_of_a_stochastic_matrix_is_one(karate: nx.Graph) -> None:
    """§8.2: no eigenvalue of a stochastic adjacency matrix is ever above 1."""
    matrix, _ = stochastic(karate, weight=None)
    values, _ = eigenpairs(matrix, largest=True)
    assert values[0] == pytest.approx(1.0)
    assert values.max() <= 1.0 + 1e-9


def test_an_isolated_node_leaves_its_stochastic_row_at_zero() -> None:
    """A walker on a node with no edges has nowhere to go, so the row cannot be scaled to 1."""
    graph = nx.Graph([("a", "b")])
    graph.add_node("lonely")
    matrix, nodes = stochastic(graph, weight=None)
    assert matrix[nodes.index("lonely")].sum() == 0.0


def test_powers_of_the_stochastic_matrix_are_walks_of_that_length() -> None:
    """§8.2: the nth matrix power holds the transition probabilities of a walk of n steps, and
    its rows still sum to 1. On a path a-b-c, two steps from a must land on a or c, never on b.
    """
    graph = nx.Graph([("a", "b"), ("b", "c")])
    matrix, nodes = stochastic(graph, weight=None)
    two = np.linalg.matrix_power(matrix, 2)
    assert np.allclose(two.sum(axis=1), np.ones(3))
    a, b, c = (nodes.index(n) for n in ("a", "b", "c"))
    assert two[a, b] == pytest.approx(0.0)
    assert two[a, a] == pytest.approx(0.5) and two[a, c] == pytest.approx(0.5)


def test_the_sparse_stochastic_matrix_holds_the_same_numbers(karate: nx.Graph) -> None:
    for orientation in ("row", "column"):
        packed, _ = stochastic(karate, orientation=orientation, sparse=True)
        spread, _ = stochastic(karate, orientation=orientation)
        assert np.allclose(dense(packed), spread), orientation


def test_an_unknown_orientation_is_refused(karate: nx.Graph) -> None:
    with pytest.raises(ValueError, match="orientation must be"):
        stochastic(karate, orientation="diagonal")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- §8.3 incidence


def test_the_incidence_matrix_of_a_two_mode_edge_list() -> None:
    matrix, left, right = incidence(PLANTED_PAIRS)
    assert left == ["a", "b", "c"] and right == ["x", "y"]
    assert np.array_equal(matrix, np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]))
    assert np.array_equal(matrix.sum(axis=1), np.array([2.0, 2.0, 1.0]))


def test_a_repeated_membership_still_counts_once() -> None:
    matrix, _, _ = incidence([*PLANTED_PAIRS, ("a", "x"), ("a", "x")])
    assert matrix.max() == 1.0


def test_explicit_axes_align_the_incidence_matrix() -> None:
    matrix, left, right = incidence(PLANTED_PAIRS, left=["c", "a"], right=["y", "x"])
    assert left == ["c", "a"] and right == ["y", "x"]
    assert np.array_equal(matrix, np.array([[1.0, 0.0], [1.0, 1.0]]))


def test_an_explicit_incidence_axis_may_not_repeat_or_invent_a_node() -> None:
    """A duplicated row and a row for a node in no pair are both indistinguishable from a node
    with no memberships, so they are refused on both axes, as ``node_order`` refuses them."""
    with pytest.raises(ValueError, match="must not repeat"):
        incidence(PLANTED_PAIRS, left=["a", "a", "b"])
    with pytest.raises(ValueError, match="must not repeat"):
        incidence(PLANTED_PAIRS, right=["x", "x"])
    with pytest.raises(ValueError, match="not in the left of pairs"):
        incidence(PLANTED_PAIRS, left=["a", "zz"])
    with pytest.raises(ValueError, match="not in the right of pairs"):
        incidence(PLANTED_PAIRS, right=["x", "zz"])


def test_project_reproduces_the_planted_shared_neighbour_counts() -> None:
    """§26.1: the (u, v) cell of B @ B.T is |N(u) ∩ N(v)|, the simple weight."""
    matrix, left, _ = incidence(PLANTED_PAIRS)
    projected = project(matrix, "left")
    index = {node: position for position, node in enumerate(left)}
    assert projected[index["a"], index["b"]] == 2.0  # a and b share x and y
    assert projected[index["a"], index["c"]] == 1.0  # only y
    assert np.array_equal(np.diag(projected), np.zeros(3))  # §26.1 zeroes the degree diagonal


def test_project_and_bipartite_projection_agree_on_a_planted_graph() -> None:
    assert _projected_weights(PLANTED_PAIRS, "left") == _counted_weights(PLANTED_PAIRS, "left")
    assert _projected_weights(PLANTED_PAIRS, "right") == _counted_weights(PLANTED_PAIRS, "right")


def test_project_and_bipartite_projection_agree_on_the_southern_women(
    southern_women: list[tuple[str, str]],
) -> None:
    """The legendary two-mode network: 18 women, 14 events, 89 memberships, and a projection
    with 139 edges between women and 66 between events -- asserted so that the comparison
    below cannot pass by finding nothing on either side."""
    matrix, women, events = incidence(southern_women)
    assert (len(women), len(events)) == (18, 14)
    assert matrix.sum() == 89.0
    for side, edges in (("left", 139), ("right", 66)):
        projected = _projected_weights(southern_women, side)
        assert len(projected) == edges
        assert projected == _counted_weights(southern_women, side)
    assert max(_projected_weights(southern_women, "left").values()) == 7.0


def test_the_edge_incidence_matrix_has_two_endpoints_per_column(karate: nx.Graph) -> None:
    """§8.3: a simple graph's incidence column sums to 2; only a hypergraph's can exceed it."""
    matrix, nodes, edges = edge_incidence(karate)
    assert matrix.shape == (34, 78) and len(nodes) == 34 and len(edges) == 78
    assert np.array_equal(matrix.sum(axis=0), np.full(78, 2.0))


def test_the_oriented_incidence_matrix_rebuilds_the_laplacian(karate: nx.Graph) -> None:
    """§8.3: columns sum to zero, and B @ B.T is the combinatorial Laplacian."""
    oriented, nodes, _ = edge_incidence(karate, oriented=True)
    assert np.allclose(oriented.sum(axis=0), np.zeros(78))
    expected, same_nodes = laplacian(karate, weight=None)
    assert nodes == same_nodes
    assert np.allclose(oriented @ oriented.T, expected)
    # §8.3: B.T @ B - 2I is the adjacency of the line graph.
    line = oriented.T @ oriented
    np.fill_diagonal(line, 0.0)
    assert np.abs(line).sum() / 2 == nx.line_graph(karate).number_of_edges()


def test_the_sparse_incidence_and_projection_hold_the_same_numbers(
    southern_women: list[tuple[str, str]],
) -> None:
    packed, _, _ = incidence(southern_women, sparse=True)
    spread, _, _ = incidence(southern_women)
    assert np.allclose(dense(packed), spread)
    assert np.allclose(dense(project(packed, "right")), project(spread, "right"))


def test_an_unknown_projection_side_is_refused() -> None:
    matrix, _, _ = incidence(PLANTED_PAIRS)
    with pytest.raises(ValueError, match="side must be"):
        project(matrix, "middle")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- §8.4 Laplacian


def test_the_combinatorial_laplacian_is_degrees_minus_adjacency(karate: nx.Graph) -> None:
    matrix, nodes = laplacian(karate, weight=None)
    assert np.allclose(matrix.sum(axis=1), np.zeros(34))  # §8.4: every row sums to zero
    assert np.allclose(matrix.sum(axis=0), np.zeros(34))
    assert np.array_equal(np.diag(matrix), np.array([float(karate.degree(n)) for n in nodes]))
    assert matrix[0, 1] == -1.0  # -1 per edge


def test_the_laplacian_diagonal_holds_the_strength_of_a_weighted_network() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=6)
    matrix, _ = laplacian(graph)
    assert np.array_equal(np.diag(matrix), np.array([6.0, 6.0]))
    assert np.allclose(matrix.sum(axis=1), np.zeros(2))


def test_zero_is_the_smallest_laplacian_eigenvalue_on_the_karate_club(karate: nx.Graph) -> None:
    """§8.4: lambda_1 = 0, and its multiplicity is the number of components -- here, one."""
    matrix, _ = laplacian(karate, weight=None)
    values, _ = eigenpairs(matrix, largest=False)
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values.min() > -1e-9  # positive semi-definite (§8.4)
    assert sum(abs(value) < 1e-9 for value in values) == nx.number_connected_components(karate) == 1
    assert values[1] > 1e-9  # the algebraic connectivity of a connected graph is positive


def test_the_zero_eigenvalue_counts_the_components_of_a_planted_graph(
    two_components: nx.Graph,
) -> None:
    values, _ = eigenpairs(laplacian(two_components, weight=None)[0], largest=False)
    assert sum(abs(value) < 1e-9 for value in values) == 2
    assert nx.number_connected_components(two_components) == 2


def test_an_isolated_node_counts_as_its_own_component_in_every_laplacian() -> None:
    graph = nx.Graph([("a", "b")])
    graph.add_node("lonely")
    for kind in ("combinatorial", "symmetric", "random-walk"):
        values, _ = eigenpairs(laplacian(graph, kind=kind, weight=None)[0], largest=False)
        assert sum(abs(value) < 1e-9 for value in values) == 2, kind


def test_the_symmetric_laplacian_eigenvalues_lie_between_zero_and_two(karate: nx.Graph) -> None:
    matrix, _ = laplacian(karate, kind="symmetric", weight=None)
    assert np.allclose(matrix, matrix.T)
    values, _ = eigenpairs(matrix, largest=False)
    assert values.min() >= -1e-9
    assert values.max() <= 2.0 + 1e-9
    assert values[0] == pytest.approx(0.0, abs=1e-9)


def test_only_a_bipartite_graph_reaches_the_upper_bound_of_two() -> None:
    """The top of [0, 2] is attained exactly when a component is bipartite."""
    bipartite, _ = laplacian(nx.complete_bipartite_graph(3, 4), kind="symmetric", weight=None)
    assert eigenpairs(bipartite, largest=True)[0][0] == pytest.approx(2.0)
    triangle, _ = laplacian(nx.complete_graph(3), kind="symmetric", weight=None)
    assert eigenpairs(triangle, largest=True)[0][0] < 2.0


def test_the_random_walk_laplacian_is_one_minus_the_transition_matrix(karate: nx.Graph) -> None:
    """L_rw = I - P, so it shares the symmetric form's eigenvalues but not its symmetry."""
    walk, _ = laplacian(karate, kind="random-walk", weight=None)
    transition, _ = stochastic(karate, orientation="row", weight=None)
    assert np.allclose(walk, np.eye(34) - transition)
    assert not np.allclose(walk, walk.T)
    symmetric, _ = laplacian(karate, kind="symmetric", weight=None)
    assert np.allclose(
        np.sort(eigenpairs(walk, largest=False)[0]),
        np.sort(eigenpairs(symmetric, largest=False)[0]),
    )


def test_the_sparse_laplacians_hold_the_same_numbers(karate: nx.Graph) -> None:
    for kind in ("combinatorial", "symmetric", "random-walk"):
        packed, _ = laplacian(karate, kind=kind, sparse=True)
        spread, _ = laplacian(karate, kind=kind)
        assert np.allclose(dense(packed), spread), kind


def test_dense_and_sparse_hold_the_same_numbers_on_a_500_node_generator_graph() -> None:
    """ATL-ENT-3's own known answer, named beside the karate-club fixtures above: a graph well
    above the size those fixtures test at must still give bit-for-bit identical adjacency,
    stochastic and Laplacian matrices between the dense and the sparse builder -- unlike
    ``eigenpairs``, neither of these is iterative, so ``==`` is the right assertion, not
    ``allclose``."""
    graph = nx.gnm_random_graph(500, 2000, seed=3)
    dense_adjacency, dense_nodes = adjacency(graph, weight=None)
    sparse_adjacency, sparse_nodes = adjacency(graph, weight=None, sparse=True)
    assert sparse_nodes == dense_nodes
    assert np.array_equal(dense(sparse_adjacency), dense_adjacency)

    for orientation in ("row", "column"):
        dense_p, _ = stochastic(graph, orientation=orientation, weight=None)
        sparse_p, _ = stochastic(graph, orientation=orientation, weight=None, sparse=True)
        assert np.array_equal(dense(sparse_p), dense_p), orientation

    for kind in ("combinatorial", "symmetric", "random-walk"):
        dense_l, _ = laplacian(graph, kind=kind, weight=None)
        sparse_l, _ = laplacian(graph, kind=kind, weight=None, sparse=True)
        assert np.array_equal(dense(sparse_l), dense_l), kind


def test_an_unknown_laplacian_is_refused(karate: nx.Graph) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        laplacian(karate, kind="signed")  # type: ignore[arg-type]


def test_a_directed_graph_is_refused_rather_than_given_one_of_its_two_laplacians() -> None:
    """§8.4: a directed node has an indegree and an outdegree, so it has two Laplacians. Left
    alone the code would build the outdegree one silently -- asymmetric, columns not summing to
    zero -- and its spectrum can come out real, so nothing downstream would notice."""
    directed = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "c")])
    # DiGraph subclasses Graph, so the type annotation alone stops nothing.
    assert isinstance(directed, nx.Graph)
    with pytest.raises(ValueError, match="undefined for a directed graph"):
        laplacian(directed)
    undirected, _ = laplacian(directed.to_undirected())
    assert np.allclose(undirected.sum(axis=0), np.zeros(3))
    # The adjacency and the stochastic matrix of a directed graph are well defined, so they stay.
    matrix, nodes = adjacency(directed, weight=None)
    assert not np.array_equal(matrix, matrix.T)
    assert matrix[nodes.index("a"), nodes.index("b")] == 1.0
    assert matrix[nodes.index("b"), nodes.index("a")] == 0.0
    assert np.allclose(stochastic(directed, weight=None)[0].sum(axis=1), np.ones(3))


# ----------------------------------------------------------------------------- §5.5 eigenpairs


def test_the_leading_karate_eigenvector_puts_node_33_on_top(karate: nx.Graph) -> None:
    """The published answer for Zachary's club: node 33 (the president) leads the adjacency
    eigenvector, with node 0 (the instructor) next."""
    matrix, nodes = adjacency(karate, weight=None)
    values, vectors = eigenpairs(matrix, k=1, largest=True)
    assert values[0] == pytest.approx(6.7256977276, abs=1e-8)
    ranked = sorted(zip(nodes, np.abs(vectors[:, 0]), strict=True), key=lambda p: -p[1])
    assert [node for node, _ in ranked[:2]] == [33, 0]


def test_eigenpairs_pairs_each_value_with_its_column() -> None:
    matrix = np.diag([3.0, 1.0, 2.0])
    values, vectors = eigenpairs(matrix, largest=True)
    assert np.allclose(values, [3.0, 2.0, 1.0])
    for index, value in enumerate(values):
        assert np.allclose(matrix @ vectors[:, index], value * vectors[:, index])


def test_eigenpairs_keeps_only_k_and_sorts_the_way_it_is_asked() -> None:
    matrix = np.diag([3.0, 1.0, 2.0])
    values, vectors = eigenpairs(matrix, k=2, largest=False)
    assert np.allclose(values, [1.0, 2.0])
    assert vectors.shape == (3, 2)


def test_eigenpairs_of_an_empty_matrix_is_empty() -> None:
    values, vectors = eigenpairs(np.zeros((0, 0)))
    assert values.shape == (0,) and vectors.shape == (0, 0)


def test_eigenpairs_refuses_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        eigenpairs(np.zeros((2, 3)))


def test_eigenpairs_refuses_a_matrix_with_complex_eigenvalues() -> None:
    """§8.4: a directed graph's Laplacian has complex eigenvalues, and dropping the imaginary
    part silently is a decision the caller has to make, not this function."""
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="complex eigenvalues"):
        eigenpairs(rotation)


# --------------------------------------------------------------------------- §5.6 factorisation


def test_svd_reconstructs_the_matrix_it_decomposed() -> None:
    matrix, _, _ = incidence(PLANTED_PAIRS)
    u, singular_values, vt = svd(matrix)
    assert np.all(np.diff(singular_values) <= 1e-12)  # descending
    assert np.allclose(u @ np.diag(singular_values) @ vt, matrix)


def test_svd_truncates_to_k_and_clamps_it() -> None:
    matrix, _, _ = incidence(PLANTED_PAIRS)
    u, singular_values, vt = svd(matrix, k=1)
    assert u.shape == (3, 1) and singular_values.shape == (1,) and vt.shape == (1, 2)
    assert svd(matrix, k=99)[1].shape == (2,)


def test_svd_of_a_rank_one_matrix_has_one_singular_value() -> None:
    """Every row a multiple of [1, 2]: one component explains the matrix exactly."""
    matrix = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    singular_values = svd(matrix)[1]
    assert singular_values[0] > 0.0
    assert singular_values[1] == pytest.approx(0.0, abs=1e-12)


def test_nmf_recovers_two_planted_non_negative_parts() -> None:
    """Six rows built from two archetypes; NMF must find them without a negative entry."""
    parts = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    loadings = np.array([[2.0, 0.0], [3.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0], [0.0, 3.0]])
    matrix = loadings @ parts
    w, h = nmf(matrix, 2, seed=1)
    assert w.shape == (6, 2) and h.shape == (2, 4)
    assert w.min() >= 0.0 and h.min() >= 0.0
    assert np.allclose(w @ h, matrix, atol=1e-6)
    # Each row loads on exactly one part, as it was built to.
    assert [int(np.argmax(row)) for row in w] == [0, 0, 0, 1, 1, 1]


def test_nmf_is_reproducible_from_a_seed() -> None:
    matrix, _, _ = incidence(PLANTED_PAIRS)
    first = nmf(matrix, 2, seed=7)
    second = nmf(matrix, 2, seed=7)
    assert np.allclose(first[0], second[0]) and np.allclose(first[1], second[1])


def test_nmf_refuses_a_matrix_it_is_undefined_for(karate: nx.Graph) -> None:
    matrix, _ = laplacian(karate, weight=None)
    with pytest.raises(ValueError, match="non-negative"):
        nmf(matrix, 2)
    adjacency_matrix, _ = adjacency(karate, weight=None)
    with pytest.raises(ValueError, match="k must be between"):
        nmf(adjacency_matrix, 99)


# --------------------------------------------------------------- ATL-ENT-3: sparse eigenpairs


def test_sparse_eigenpairs_agree_with_dense_on_the_karate_laplacian(karate: nx.Graph) -> None:
    sparse_matrix, _ = laplacian(karate, kind="combinatorial", weight=None, sparse=True)
    dense_matrix, _ = laplacian(karate, kind="combinatorial", weight=None)

    small_values, small_vectors = eigenpairs(sparse_matrix, k=2, largest=False, sparse=True)
    dense_small_values, _ = eigenpairs(dense_matrix, largest=False)
    assert small_values == pytest.approx(dense_small_values[:2], abs=1e-6)
    assert small_values[0] == pytest.approx(0.0, abs=1e-8)  # §8.4: one component, one zero

    large_values, _ = eigenpairs(sparse_matrix, k=2, largest=True, sparse=True)
    dense_large_values, _ = eigenpairs(dense_matrix, largest=True)
    assert large_values == pytest.approx(dense_large_values[:2], abs=1e-6)
    assert small_vectors.shape == (34, 2)


def test_sparse_eigenpairs_on_a_500_node_generator_graph_matches_dense() -> None:
    """The known-answer generator graph ATL-ENT-3's own ticket names: dense and sparse must
    agree (to a numerical tolerance -- the sparse solver is iterative) on a graph well above
    :data:`~graphrag.sna.matrices.SPARSE_ABOVE_NODES`."""
    graph = nx.gnm_random_graph(500, 1800, seed=7)
    giant = graph.subgraph(max(nx.connected_components(graph), key=len))
    dense_matrix, _ = laplacian(giant, kind="combinatorial", weight=None)
    sparse_matrix, _ = laplacian(giant, kind="combinatorial", weight=None, sparse=True)

    dense_values, dense_vectors = eigenpairs(dense_matrix, k=3, largest=False)
    sparse_values, sparse_vectors = eigenpairs(sparse_matrix, k=3, largest=False, sparse=True)
    assert sparse_values == pytest.approx(dense_values, abs=1e-6)

    # The eigenvector for a simple (non-repeated) eigenvalue is unique up to sign (§5.5), so
    # compare magnitudes rather than the vectors themselves.
    assert np.abs(np.abs(sparse_vectors[:, 1]) - np.abs(dense_vectors[:, 1])).max() < 1e-4


def test_sparse_eigenpairs_needs_k(karate: nx.Graph) -> None:
    matrix, _ = laplacian(karate, sparse=True)
    with pytest.raises(ValueError, match="needs k"):
        eigenpairs(matrix, sparse=True)


def test_sparse_eigenpairs_refuses_k_too_close_to_n(karate: nx.Graph) -> None:
    matrix, _ = laplacian(karate, sparse=True)  # 34 nodes
    with pytest.raises(ValueError, match="1 <= k < n - 1"):
        eigenpairs(matrix, k=33, sparse=True)
    with pytest.raises(ValueError, match="1 <= k < n - 1"):
        eigenpairs(matrix, k=0, sparse=True)


def test_sparse_eigenpairs_refuses_an_asymmetric_matrix() -> None:
    directed = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "a")])
    matrix, _ = adjacency(directed, weight=None, sparse=True)
    with pytest.raises(ValueError, match="needs a symmetric matrix"):
        eigenpairs(matrix, k=1, sparse=True)


def test_sparse_eigenpairs_refuses_a_non_square_matrix() -> None:
    incidence_matrix, _, _ = incidence(PLANTED_PAIRS, sparse=True)
    with pytest.raises(ValueError, match="square matrix"):
        eigenpairs(incidence_matrix, k=1, sparse=True)


# ----------------------------------------------------------------------------- helpers


def _projected_weights(pairs: list[tuple[str, str]], side: str) -> dict[tuple[str, str], float]:
    """Edge weights from ``project(incidence(...))``, keyed by node pair."""
    matrix, left, right = incidence(pairs)
    nodes = left if side == "left" else right
    projected = project(matrix, side)  # type: ignore[arg-type]
    return {
        (nodes[i], nodes[j]): float(projected[i, j])
        for i in range(len(nodes))
        for j in range(i + 1, len(nodes))
        if projected[i, j] > 0
    }


def _counted_weights(pairs: list[tuple[str, str]], side: str) -> dict[tuple[str, str], float]:
    """The same weights as ``export.bipartite_projection`` counts them in Python."""
    graph = bipartite_projection(pairs, side)  # type: ignore[arg-type]
    return {(u, v) if u < v else (v, u): float(w) for u, v, w in graph.edges(data="weight")}
