"""Chapter 11's random walks, held to answers worked out before the code ran.

Every assertion here comes from one of four places, and none of them is "the function returned
something":

* **First-step equations solved by hand.** The three-node chain of §11.3 is small enough to write
  the recurrence out: from an end node the walker has one move, from the middle it has two, and
  the pair of equations gives 4 and 3 exactly. The arithmetic is in the test.
* **Closed forms for graphs whose structure is a formula.** The effective resistance between two
  nodes ``d`` apart on a cycle of ``n`` is ``d(n-d)/n``, because the two arcs are resistors in
  parallel; on a tree it is the hop count, because there is only one route; the stationary
  distribution of an undirected graph is ``k/2|E|``; the non-backtracking matrix of a
  ``k``-regular graph has every row summing to ``k-1``, so its spectral radius is ``k-1``.
* **Identities the book states between two quantities.** ``C = 2|E|Ω`` (§11.4), ``C = H + Hᵀ``
  (§11.3), the Random Target Lemma's ``Σ_v π_v H[u][v] = Σ_n 1/(1-λ_n)`` with no ``u`` on the
  right (§11.3), max-flow equals min-cut (§11.5), and the conserved mean of each consensus
  dynamics (§11.6). Each is checked by computing both sides two different ways.
* **Published results on legendary graphs.** The karate club's degree ranking, which §11.1 says
  *is* its stationary distribution, and its pendant node, which §11.5 says is what a global
  minimum cut will find on a real network.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.matrices import laplacian
from graphrag.sna.walks import (
    Consensus,
    commute_distance,
    commute_matrix,
    commute_time,
    consensus,
    consensus_time,
    directed_edges,
    effective_resistance,
    fiedler_cut,
    fiedler_vector,
    global_mincut,
    hitting_time,
    hitting_times,
    laplacian_pseudoinverse,
    mincut,
    non_backtracking_matrix,
    non_backtracking_radius,
    random_target_time,
    resistance_matrix,
    stationary_distribution,
    volume,
)
from tests.legendary import karate_club, les_miserables, unweighted


@pytest.fixture
def chain() -> nx.Graph:
    """§11.3's own example: the three-node chain, degrees (1, 2, 1), 2|E| = 4.

    *"How long does it take to get from node 1 to node 2? Well, node 1 has only one connection
    and it goes to node 2, so it will always take one step. But to get from node 2 to node 1, you
    only have a 50% chance of doing it in one step."*
    """
    return nx.path_graph(3)


@pytest.fixture
def barbell() -> nx.Graph:
    """Two 5-cliques joined by exactly one edge: the cut between them is 1 by construction."""
    return nx.barbell_graph(5, 0)


@pytest.fixture
def two_triangles() -> nx.Graph:
    """Two triangles with nothing between them: two components, planted."""
    return nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))


# --------------------------------------------------------- §11.1 the stationary distribution


def test_stationary_distribution_of_the_three_node_chain_is_a_quarter_a_half_a_quarter(
    chain: nx.Graph,
) -> None:
    """§11.1: π is the normalized degree. Degrees are (1, 2, 1) and 2|E| = 4, so π = (.25, .5,
    .25) -- no eigensolver involved."""
    distribution, order = stationary_distribution(chain)
    assert order == [0, 1, 2]
    assert distribution == pytest.approx([0.25, 0.5, 0.25])
    assert distribution.sum() == pytest.approx(1.0)


def test_the_stationary_distribution_is_the_fixed_point_of_the_transition_matrix(
    chain: nx.Graph,
) -> None:
    """The defining equation of §11.1, ``πP = π``, checked against the closed form that skipped
    it entirely."""
    from graphrag.sna.matrices import dense, stochastic

    distribution, order = stationary_distribution(chain)
    transition, _ = stochastic(chain, order, orientation="row")
    assert distribution @ dense(transition) == pytest.approx(distribution)


def test_the_stationary_distribution_ranks_the_karate_club_by_degree() -> None:
    """§11.1 says π *is* degree, so its ranking must be the published degree ranking: node 33
    (the club president, 17 ties) ahead of node 0 (Mr. Hi, 16)."""
    graph, known = karate_club()
    distribution, order = stationary_distribution(unweighted(graph), weight=None)
    by_share = sorted(zip(order, distribution, strict=True), key=lambda pair: -pair[1])
    ranked = [node for node, _ in by_share]
    assert ranked[0] == known.top_degree == 33
    assert ranked[1] == 0
    degrees = dict(unweighted(graph).degree())
    assert distribution == pytest.approx([degrees[node] / (2 * known.edges) for node in order])


def test_a_weighted_stationary_distribution_uses_strength_not_degree() -> None:
    """Two edges of weight 1 and 3 on a chain: the middle node holds half the strength either
    way, but the ends split 1:3 rather than 1:1."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "c", weight=3.0)
    distribution, order = stationary_distribution(graph)
    assert order == ["a", "b", "c"]
    assert distribution == pytest.approx([1 / 8, 4 / 8, 3 / 8])


def test_a_disconnected_graph_mixes_its_components_in_proportion_to_their_degree(
    two_triangles: nx.Graph,
) -> None:
    """§11.1: two components are *"two different networks"*. Both triangles hold half the degree,
    so every node gets 1/6, and renormalising a component's entries gives that component's own
    walk (1/3 each)."""
    distribution, _ = stationary_distribution(two_triangles)
    assert distribution == pytest.approx([1 / 6] * 6)
    first = distribution[:3] / distribution[:3].sum()
    assert first == pytest.approx([1 / 3] * 3)


def test_the_stationary_distribution_of_a_directed_cycle_is_uniform() -> None:
    """A directed cycle has no closed form to fall back on, so this is the power iteration of
    §11.1. Every node has exactly one way in and one way out, so π must be uniform."""
    cycle = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
    distribution, order = stationary_distribution(cycle)
    assert order == [0, 1, 2, 3]
    assert distribution == pytest.approx([0.25] * 4)


def test_the_sparse_directed_power_iteration_agrees_with_the_dense_one() -> None:
    """ATL-ENT-3: above ``sparse_above`` the directed path multiplies a sparse ``P`` instead of
    a dense one (:func:`graphrag.sna.walks._directed_stationary`); the numbers must be identical
    to the dense path, not merely close, since neither side is iterative here."""
    cycle = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
    dense_distribution, dense_order = stationary_distribution(cycle)
    sparse_distribution, sparse_order = stationary_distribution(cycle, sparse_above=1)
    assert sparse_order == dense_order
    assert sparse_distribution == pytest.approx(dense_distribution)


def test_the_sparse_directed_power_iteration_matches_on_a_500_node_generator_graph() -> None:
    """ATL-ENT-3's own known-answer graph, for the one walk quantity that iterates: a directed
    cycle through every node guarantees no sink and strong connectivity, so π exists and this
    reads the same to machine precision whichever form ``P`` took while getting there."""
    cycle_edges = [(i, (i + 1) % 500) for i in range(500)]
    extra = nx.gnm_random_graph(500, 1500, seed=5, directed=True).edges
    graph = nx.DiGraph(cycle_edges + list(extra))
    dense_distribution, dense_order = stationary_distribution(graph)
    sparse_distribution, sparse_order = stationary_distribution(graph, sparse_above=1)
    assert sparse_order == dense_order
    assert sparse_distribution == pytest.approx(dense_distribution, abs=1e-9)


def test_a_directed_graph_with_a_sink_has_no_stationary_distribution() -> None:
    """§11.1's reason, in the error: a walker that reaches a node with no out-edge never leaves,
    so probability drains instead of converging. The fix the section names is the teleport."""
    with_sink = nx.DiGraph([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="no out-edge") as raised:
        stationary_distribution(with_sink)
    assert "pagerank" in str(raised.value)


def test_a_directed_graph_that_is_not_strongly_connected_has_no_single_answer() -> None:
    """Two cycles joined one way: the walker falls into the second and cannot return, so the
    limit depends on where it started."""
    trap = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 3), (3, 2)])
    with pytest.raises(ValueError, match="not strongly connected"):
        stationary_distribution(trap)


# ----------------------------------------------------------- §11.2 non-backtracking walks


def test_the_non_backtracking_matrix_has_one_row_per_edge_direction(chain: nx.Graph) -> None:
    """§11.2 builds *"one row/column per edge direction"*, so an undirected graph with 2 edges
    gives a 4 by 4 matrix, and the rows follow ``directed_edges``."""
    matrix, edges = non_backtracking_matrix(chain)
    assert edges == [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert matrix.shape == (4, 4)
    assert directed_edges(chain) == edges


def test_the_non_backtracking_matrix_of_a_triangle_is_a_permutation() -> None:
    """Each arrival in a triangle has exactly one legal continuation, so every row sums to 1,
    the block diagonal is zero (§11.2: *"a one in the diagonal is exactly a backtracking
    move"*) and the matrix is not symmetric."""
    matrix, edges = non_backtracking_matrix(nx.complete_graph(3))
    assert len(edges) == 6
    assert matrix.sum(axis=1) == pytest.approx([1.0] * 6)
    assert np.diag(matrix) == pytest.approx([0.0] * 6)
    assert not np.allclose(matrix, matrix.T)


def test_the_leading_eigenvalue_is_one_for_a_triangle_and_two_for_k4() -> None:
    """On a ``k``-regular graph every row of NB sums to ``k - 1``, so the spectral radius is
    ``k - 1``: 1 for the 2-regular triangle, 2 for the 3-regular K4."""
    assert non_backtracking_radius(nx.complete_graph(3)) == pytest.approx(1.0)
    assert non_backtracking_radius(nx.complete_graph(4)) == pytest.approx(2.0)
    assert non_backtracking_radius(nx.complete_graph(6)) == pytest.approx(4.0)


def test_a_walk_into_a_pendant_node_has_no_continuation_at_all() -> None:
    """The row for an edge arriving at a degree-1 node is all zero: the only move left is the
    backtrack, and §11.2 forbids it. On a tree that makes NB nilpotent, so its radius is 0."""
    star = nx.star_graph(2)
    matrix, edges = non_backtracking_matrix(star)
    arrivals = dict(zip(edges, matrix.sum(axis=1), strict=True))
    assert arrivals[(0, 1)] == 0.0
    assert arrivals[(1, 0)] == 1.0
    assert non_backtracking_radius(star) == pytest.approx(0.0)


def test_the_non_backtracking_matrix_refuses_to_build_a_square_too_big_to_hold() -> None:
    with pytest.raises(ValueError, match="max_edges"):
        non_backtracking_matrix(nx.complete_graph(6), max_edges=4)


# --------------------------------------------------------------------- §11.3 hitting time


def test_hitting_times_on_the_three_node_chain_match_the_first_step_equations(
    chain: nx.Graph,
) -> None:
    """Solved by hand from §11.3's recurrence ``H[u][v] = 1 + (1/k_u) Σ_z H[z][v]``.

    Write a, b, c for nodes 0, 1, 2. From a there is one move, to b::

        H(a->c) = 1 + H(b->c)

    From b the walker steps to c with probability 1/2 (arriving, 0 more steps) and to a with
    probability 1/2 (from where it must return to b)::

        H(b->c) = 1 + (1/2)(0) + (1/2) H(a->c)

    Substituting the first into the second: H(a->c) = 1 + 1 + H(a->c)/2, so H(a->c)/2 = 2 and
    **H(a->c) = 4**, **H(b->c) = 3**. By the mirror symmetry of the chain H(c->a) = 4 and
    H(a->b) = 1 -- a has one edge and it goes to b -- which is the *"super intuitive"* pair
    §11.3 works through.
    """
    matrix, order = hitting_times(chain)
    assert order == [0, 1, 2]
    assert matrix == pytest.approx(np.array([[0.0, 1.0, 4.0], [3.0, 0.0, 3.0], [4.0, 1.0, 0.0]]))
    assert hitting_time(chain, 0, 2) == pytest.approx(4.0)
    assert hitting_time(chain, 1, 2) == pytest.approx(3.0)
    assert hitting_time(chain, 0, 1) == pytest.approx(1.0)


def test_hitting_time_is_asymmetric_and_commute_time_is_not(chain: nx.Graph) -> None:
    """§11.3: H depends on the degree of the *destination*, so one step out and three back. The
    commute time is their sum and is the same either way round."""
    assert hitting_time(chain, 0, 1) == pytest.approx(1.0)
    assert hitting_time(chain, 1, 0) == pytest.approx(3.0)
    assert commute_time(chain, 0, 1) == pytest.approx(4.0)
    assert commute_time(chain, 1, 0) == pytest.approx(4.0)


def test_the_commute_time_of_the_chain_ends_is_two_m_times_the_resistance(
    chain: nx.Graph,
) -> None:
    """The ticket's arithmetic, and §11.4's identity: the chain has 2|E| = 4 and the two ends are
    2 ohms apart (two 1-ohm resistors in series), so C = 4 x 2 = 8. And directly: 4 steps there
    plus 4 back."""
    assert volume(chain) == pytest.approx(4.0)
    assert effective_resistance(chain, 0, 2) == pytest.approx(2.0)
    assert commute_time(chain, 0, 2) == pytest.approx(8.0)
    assert commute_time(chain, 0, 2) == pytest.approx(
        hitting_time(chain, 0, 2) + hitting_time(chain, 2, 0)
    )


def test_the_random_target_lemma_does_not_depend_on_where_you_started(chain: nx.Graph) -> None:
    """§11.3: ``Σ_v π_v H[u][v] = Σ_{n>=2} 1/(1 - λ_n)``, and *"the right hand side has no trace
    of u"*.

    On the chain the eigenvalues of N are (1, 0, -1), so the right side is 1/(1-0) + 1/(1+1) =
    1.5. The left side, from the hitting times above: for u = a, 0(.25) + 1(.5) + 4(.25) = 1.5;
    for u = b, 3(.25) + 0(.5) + 3(.25) = 1.5; for u = c, 4(.25) + 1(.5) + 0(.25) = 1.5. The two
    formulas are computed by different routes -- one from the Laplacian pseudoinverse, one from
    the eigenvalues of the normalised adjacency -- so agreeing is a statement about both.
    """
    assert random_target_time(chain) == pytest.approx(1.5)
    times, order = hitting_times(chain)
    distribution, same_order = stationary_distribution(chain)
    assert order == same_order
    assert times @ distribution == pytest.approx([1.5, 1.5, 1.5])


def test_the_random_target_lemma_holds_on_the_karate_club() -> None:
    """The same identity on a graph nobody can solve by hand, which is what makes it a check on
    the implementation rather than on the arithmetic."""
    graph = unweighted(karate_club()[0])
    times, order = hitting_times(graph, weight=None)
    distribution, _ = stationary_distribution(graph, weight=None)
    constant = random_target_time(graph, weight=None)
    assert times @ distribution == pytest.approx([constant] * len(order))


def test_hitting_time_across_two_components_is_infinite(two_triangles: nx.Graph) -> None:
    """A walker cannot reach what it cannot reach, so the expected number of steps is infinite
    rather than large."""
    matrix, _ = hitting_times(two_triangles)
    assert math.isinf(matrix[0, 3])
    assert matrix[0, 1] == pytest.approx(2.0)


def test_hitting_times_refuse_a_component_bigger_than_the_guard() -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        hitting_times(nx.complete_graph(10), max_nodes=5)


def test_hitting_times_refuse_a_directed_graph() -> None:
    """§11.3's route runs through the Laplacian, which §8.4 will not give a directed graph."""
    with pytest.raises(ValueError, match="undirected"):
        hitting_times(nx.DiGraph([(0, 1), (1, 0)]))


# -------------------------------------------------------------- §11.4 effective resistance


@pytest.mark.parametrize("hops", [1, 2, 3, 4])
def test_resistance_on_a_path_is_the_hop_count(hops: int) -> None:
    """A tree has exactly one route between any two nodes, so the resistors are in series and Ω
    is the hop count -- the one case where §11.4's metric tells you nothing a shortest path did
    not."""
    path = nx.path_graph(6)
    assert effective_resistance(path, 0, hops) == pytest.approx(float(hops))
    assert nx.shortest_path_length(path, 0, hops) == hops


@pytest.mark.parametrize("size", [5, 7, 8])
def test_resistance_on_a_cycle_is_the_parallel_arc_formula(size: int) -> None:
    """Two nodes ``d`` apart on C_n are joined by arcs of ``d`` and ``n - d`` unit resistors in
    parallel, so ``Ω = d(n-d)/n`` -- always **below** the shortest path ``min(d, n-d)``, which
    is §11.4's point about counting every route rather than one."""
    cycle = nx.cycle_graph(size)
    matrix, order = resistance_matrix(cycle)
    for distance in range(1, size):
        expected = distance * (size - distance) / size
        assert matrix[order.index(0), order.index(distance)] == pytest.approx(expected)
        assert expected <= min(distance, size - distance)


def test_the_book_writes_the_pseudoinverse_with_a_matrix_of_ones_and_it_is_the_same_number() -> (
    None
):
    """§11.4 computes ``Γ = (L + 1/|V| · 1)†`` where this computes ``L†``. They differ by
    ``J/|V|`` everywhere, and that difference cancels in ``Γuu + Γvv - 2Γuv`` because
    ``1/|V| + 1/|V| - 2/|V| = 0``. Both routes, on the karate club, entry by entry."""
    graph = unweighted(karate_club()[0])
    combinatorial, order = laplacian(graph, weight=None)
    size = len(order)
    gamma = np.linalg.pinv(np.asarray(combinatorial) + np.ones((size, size)) / size)
    diagonal = np.diag(gamma)
    book = diagonal[:, np.newaxis] + diagonal[np.newaxis, :] - 2.0 * gamma
    ours, same_order = resistance_matrix(graph, weight=None)
    assert same_order == order
    assert ours == pytest.approx(book, abs=1e-9)


def test_the_pseudoinverse_satisfies_the_moore_penrose_identity() -> None:
    """§11.4: *"it holds that L L† L = L"*, which is what makes the dagger stand in for an
    inverse the singular Laplacian does not have."""
    graph = unweighted(karate_club()[0])
    combinatorial, order = laplacian(graph, weight=None)
    dagger, same_order = laplacian_pseudoinverse(graph, weight=None)
    assert same_order == order
    grid = np.asarray(combinatorial)
    assert grid @ dagger @ grid == pytest.approx(grid, abs=1e-9)


def test_commute_time_is_two_m_times_resistance_everywhere_on_les_miserables() -> None:
    """§11.4's ``C = 2|E|Ω``, on a weighted network with 77 nodes, checked against the hitting
    times computed for the same graph: ``C == H + Hᵀ`` as well."""
    graph, _ = les_miserables()
    resistances, order = resistance_matrix(graph)
    commutes, same_order = commute_matrix(graph)
    times, _ = hitting_times(graph)
    assert same_order == order
    assert commutes == pytest.approx(volume(graph) * resistances)
    assert commutes == pytest.approx(times + times.T)
    assert commute_distance(graph)[0] == pytest.approx(np.sqrt(commutes))


def test_resistance_is_a_metric_where_a_shortest_path_on_a_weighted_graph_is_not() -> None:
    """§11.4 calls Ω *"a proper metric"*: symmetric, zero on the diagonal, and obeying the
    triangle inequality for every triple. Checked exhaustively on the karate club."""
    graph = unweighted(karate_club()[0])
    matrix, order = resistance_matrix(graph, weight=None)
    assert matrix == pytest.approx(matrix.T)
    assert np.diag(matrix) == pytest.approx(np.zeros(len(order)))
    # Ω[i][k] <= Ω[i][j] + Ω[j][k] for every i, k and every detour j, all 39,304 of them.
    direct = matrix[:, :, np.newaxis]
    detour = matrix[:, np.newaxis, :] + matrix[np.newaxis, :, :]
    assert np.all(direct <= detour + 1e-9)


def test_adding_one_edge_moves_the_resistance_less_than_it_moves_the_shortest_path() -> None:
    """§11.4's argument for preferring Ω on noisy data, on the section's own example: the 6-cycle
    of figure 11.7, then the edge 1-6 added as in figure 11.8. The shortest path between those
    two falls by a factor of 3 (from 3 hops to 1) and the resistance by 2.5 (1.5 to 0.6) -- the
    book's numbers."""
    ring = nx.cycle_graph(6)
    far = (0, 3)
    before_path = nx.shortest_path_length(ring, *far)
    before_resistance = effective_resistance(ring, *far)
    assert before_path == 3
    assert before_resistance == pytest.approx(1.5)
    ring.add_edge(*far)
    assert nx.shortest_path_length(ring, *far) == 1
    assert effective_resistance(ring, *far) == pytest.approx(0.6)
    assert before_path / 1 == pytest.approx(3.0)
    assert before_resistance / 0.6 == pytest.approx(2.5)


def test_a_weighted_edge_is_a_resistor_of_one_over_its_weight() -> None:
    """Weights are conductances here, because our weights are affinities and a walker crosses a
    heavy edge more often. Two edges of weight 2 in series are 1/2 + 1/2 = 1 ohm apart."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=2.0)
    graph.add_edge("b", "c", weight=2.0)
    assert effective_resistance(graph, "a", "c") == pytest.approx(1.0)
    assert effective_resistance(graph, "a", "c", weight=None) == pytest.approx(2.0)


def test_resistance_across_two_components_is_an_open_circuit(two_triangles: nx.Graph) -> None:
    matrix, _ = resistance_matrix(two_triangles)
    assert math.isinf(matrix[0, 3])
    assert matrix[0, 1] == pytest.approx(2 / 3)


# ------------------------------------------------------------------------ §11.5 the mincut


def test_the_minimum_cut_between_two_cliques_is_the_one_edge_joining_them(
    barbell: nx.Graph,
) -> None:
    """Two 5-cliques joined by a single edge: the cut is 1 by construction, it equals the maximum
    flow (max-flow min-cut), and it puts each clique on its own side."""
    cut = mincut(barbell, 0, 9)
    assert cut.value == pytest.approx(1.0)
    assert cut.value == pytest.approx(
        nx.maximum_flow_value(
            nx.Graph((u, v, {"capacity": 1.0}) for u, v in barbell.edges), 0, 9, capacity="capacity"
        )
    )
    assert cut.source_side == frozenset({0, 1, 2, 3, 4})
    assert cut.target_side == frozenset({5, 6, 7, 8, 9})


def test_the_resistance_is_never_below_one_over_the_cut(barbell: nx.Graph) -> None:
    """§11.4 against §11.5: every route between the two nodes crosses the cut, so the conductance
    between them is at most the cut's capacity and ``Ω >= 1 / cut``. A network expensive to cut
    is cheap to walk across."""
    cut = mincut(barbell, 0, 9)
    assert effective_resistance(barbell, 0, 9) >= 1.0 / cut.value
    dense_graph = nx.complete_graph(6)
    dense_cut = mincut(dense_graph, 0, 5)
    assert dense_cut.value == pytest.approx(5.0)
    assert effective_resistance(dense_graph, 0, 5) >= 1.0 / dense_cut.value


def test_the_global_minimum_cut_of_a_real_network_is_a_pendant_node() -> None:
    """§11.5's warning, made concrete: *"the best way to solve the 2-cut problem is to put in one
    group a node with degree equal to one"*. On the karate club the exact global cut is 1, and
    the side it isolates is node 11 -- the member with a single tie."""
    graph = unweighted(karate_club()[0])
    cut = global_mincut(graph, weight=None)
    assert cut.value == pytest.approx(1.0)
    small = min((cut.source_side, cut.target_side), key=len)
    assert small == frozenset({11})
    assert graph.degree(11) == 1


def test_the_fiedler_cut_is_balanced_and_never_cheaper_than_the_exact_one(
    barbell: nx.Graph,
) -> None:
    """§11.5's spectral approximation: the sign of the Fiedler vector splits the barbell into its
    two cliques, cutting the one edge. It is an approximation, so its cut can only be worse than
    ``global_mincut``'s -- and on the karate club it is, because the exact answer is a pendant
    node and this one is balanced."""
    spectral = fiedler_cut(barbell)
    assert spectral.value == pytest.approx(1.0)
    assert {frozenset(spectral.source_side), frozenset(spectral.target_side)} == {
        frozenset({0, 1, 2, 3, 4}),
        frozenset({5, 6, 7, 8, 9}),
    }
    graph = unweighted(karate_club()[0])
    approximate = fiedler_cut(graph, weight=None)
    assert approximate.value >= global_mincut(graph, weight=None).value
    assert min(len(approximate.source_side), len(approximate.target_side)) > 1


def test_the_fiedler_vector_ranks_nodes_by_how_far_they_are_from_the_cut(
    barbell: nx.Graph,
) -> None:
    """§11.5: *"the absolute value tells us how embedded the node is in the group, or how far
    from the cut it is"*. The two nodes the bridge touches must be the least embedded of their
    own cliques."""
    vector, order = fiedler_vector(barbell)
    loadings = dict(zip(order, np.abs(vector), strict=True))
    assert loadings[4] == min(loadings[node] for node in (0, 1, 2, 3, 4))
    assert loadings[5] == min(loadings[node] for node in (5, 6, 7, 8, 9))


def test_the_fiedler_vector_is_undefined_on_a_disconnected_network(
    two_triangles: nx.Graph,
) -> None:
    with pytest.raises(ValueError, match="disconnected"):
        fiedler_vector(two_triangles)


def test_the_sparse_fiedler_vector_agrees_with_the_dense_one(barbell: nx.Graph) -> None:
    """ATL-ENT-3: ``sparse_above`` routes this through ``matrices.eigenpairs(..., sparse=True)``
    below the module's own default threshold, on the same barbell the dense test above reads by
    hand -- the sign of an eigenvector is arbitrary (§5.5), so this compares magnitudes."""
    dense_vector, dense_order = fiedler_vector(barbell)
    sparse_vector, sparse_order = fiedler_vector(barbell, sparse_above=1)
    assert sparse_order == dense_order
    assert np.abs(sparse_vector) == pytest.approx(np.abs(dense_vector), abs=1e-6)


def test_a_cut_between_two_components_costs_nothing(two_triangles: nx.Graph) -> None:
    assert mincut(two_triangles, 0, 3).value == pytest.approx(0.0)
    assert global_mincut(two_triangles).value == pytest.approx(0.0)


def test_an_unweighted_graph_does_not_read_as_infinite_capacity(barbell: nx.Graph) -> None:
    """A regression guard with teeth: ``networkx`` treats a *missing* capacity attribute as
    infinite, so handing it a graph with no weights raises ``NetworkXUnbounded`` instead of
    returning the cut that is obviously there."""
    assert "weight" not in next(iter(barbell.edges(data=True)))[2]
    assert mincut(barbell, 0, 9, weight=None).value == pytest.approx(1.0)


# ---------------------------------------------------------------------- §11.6 consensus


def test_the_laplacian_dynamics_converges_to_the_plain_mean(chain: nx.Graph) -> None:
    """§11.6's continuous diffusion conserves the plain sum, because ``1ᵀL = 0``, so opinions
    (0, 1, 2) must settle on 1.0 -- every node counting the same."""
    result = consensus(chain, {0: 0.0, 1: 1.0, 2: 2.0}, dynamics="laplacian")
    assert isinstance(result, Consensus)
    assert result.converged and result.agreed
    assert result.limit == pytest.approx(1.0)
    assert result.values == pytest.approx([1.0, 1.0, 1.0], abs=1e-8)
    assert result.note == ""


def test_the_random_walk_dynamics_converges_to_the_degree_weighted_mean(chain: nx.Graph) -> None:
    """§11.6's discrete DeGroot dynamics conserves ``π · x`` instead, so the middle node's
    opinion counts double: (1/4)(0) + (1/2)(1) + (1/4)(2) = 1.0 here, and on a graph whose
    degrees differ it parts company with the plain mean.

    The lazy walk is used because the chain is bipartite; see the next test for why.
    """
    result = consensus(chain, {0: 0.0, 1: 1.0, 2: 2.0}, lazy=True)
    assert result.limit == pytest.approx(1.0)

    star = nx.star_graph(3)  # centre 0 with degree 3, three leaves with degree 1
    opinions = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    weighted = consensus(star, opinions, lazy=True)
    plain = consensus(star, opinions, dynamics="laplacian")
    assert weighted.limit == pytest.approx(3 / 6)  # the centre carries 3 of the 6 degree
    assert plain.limit == pytest.approx(1 / 4)  # every node counts once
    assert weighted.converged and plain.converged
    assert weighted.values == pytest.approx([0.5] * 4, abs=1e-8)


def test_the_discrete_dynamics_oscillates_forever_on_a_bipartite_network(
    chain: nx.Graph,
) -> None:
    """The periodicity of §11.1 in its dynamic form: P has an eigenvalue of -1 on a bipartite
    graph, so (0, 1, 0) becomes (1, 0, 1) becomes (0, 1, 0) and never settles. ``lazy=True``
    iterates ``(I + P)/2``, which cannot oscillate and has the same limit."""
    stuck = consensus(chain, {0: 0.0, 1: 1.0, 2: 0.0}, max_steps=50)
    assert not stuck.converged
    assert "bipartite" in stuck.note
    assert stuck.values == pytest.approx([0.0, 1.0, 0.0])
    fixed = consensus(chain, {0: 0.0, 1: 1.0, 2: 0.0}, lazy=True)
    assert fixed.converged and fixed.agreed
    assert fixed.limit == pytest.approx(0.5)


def test_two_components_reach_two_consensuses_and_never_one(two_triangles: nx.Graph) -> None:
    """§11.6: a network with separate components *"cannot reach a unique consensus; every
    connected component will reach its own consensus independently"*. Each triangle settles on
    its own mean -- (0+1+2)/3 = 1 and (3+4+5)/3 = 4 -- and the two never meet."""
    result = consensus(two_triangles, {node: float(node) for node in two_triangles}, "laplacian")
    assert result.converged
    assert not result.agreed
    assert result.limit is None
    assert "components" in result.note
    final = result.by_node()
    assert [final[node] for node in (0, 1, 2)] == pytest.approx([1.0] * 3, abs=1e-8)
    assert [final[node] for node in (3, 4, 5)] == pytest.approx([4.0] * 3, abs=1e-8)


def test_consensus_on_the_karate_club_lands_on_the_mean_its_dynamics_conserves() -> None:
    """A graph with a wide degree distribution, where the two means genuinely differ: the
    degree-weighted one is pulled towards whatever the hubs started with."""
    graph = unweighted(karate_club()[0])
    opinions = {node: float(node % 2) for node in graph}
    degrees = dict(graph.degree())
    expected = sum(degrees[n] * opinions[n] for n in graph) / sum(degrees.values())
    walk = consensus(graph, opinions, weight=None)
    plain = consensus(graph, opinions, dynamics="laplacian", weight=None)
    assert walk.converged and walk.agreed
    assert walk.limit == pytest.approx(expected)
    assert walk.values == pytest.approx([expected] * 34, abs=1e-8)
    assert plain.limit == pytest.approx(sum(opinions.values()) / 34)
    assert walk.limit != pytest.approx(plain.limit, abs=1e-3)


def test_consensus_accepts_opinions_as_a_sequence_in_node_order(chain: nx.Graph) -> None:
    listed = consensus(chain, [0.0, 1.0, 2.0], dynamics="laplacian")
    mapped = consensus(chain, {0: 0.0, 1: 1.0, 2: 2.0}, dynamics="laplacian")
    assert listed.values == pytest.approx(mapped.values)
    with pytest.raises(ValueError, match="one opinion per node"):
        consensus(chain, [0.0, 1.0])
    with pytest.raises(ValueError, match="no starting opinion"):
        consensus(chain, {0: 0.0, 1: 1.0})


def test_an_unknown_dynamics_is_refused(chain: nx.Graph) -> None:
    with pytest.raises(ValueError, match="dynamics must be one of"):
        consensus(chain, [0.0, 1.0, 2.0], dynamics="brownian")  # type: ignore[arg-type]


def test_the_consensus_time_is_one_over_the_algebraic_connectivity(barbell: nx.Graph) -> None:
    """§11.6 reads ``1/λ2`` as how long the network takes to agree, and §11.5 says λ2 is small
    when the network has a thin cut. The barbell, joined by one edge, is therefore slow; a clique
    of the same size is fast. Infinite when the network is in pieces."""
    assert consensus_time(barbell) == pytest.approx(
        1.0 / float(np.linalg.eigvalsh(np.asarray(laplacian(barbell)[0]))[1])
    )
    assert consensus_time(barbell) > consensus_time(nx.complete_graph(10))
    assert math.isinf(consensus_time(nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))))


def test_the_sparse_consensus_time_agrees_with_the_dense_one(barbell: nx.Graph) -> None:
    """ATL-ENT-3: the same ``sparse_above`` routing as ``fiedler_vector``, same barbell."""
    assert consensus_time(barbell, sparse_above=1) == pytest.approx(consensus_time(barbell))


def test_a_slow_network_to_cut_is_a_slow_network_to_agree(barbell: nx.Graph) -> None:
    """§11.6's mechanism for §11.5, watched happening: after roughly ``1/λ2`` steps the two sides
    of the cut hold their own opinions, and the sign of the Fiedler vector says which side a node
    is on before the network as a whole agrees."""
    opinions = {node: (1.0 if node < 5 else 0.0) for node in barbell}
    early = consensus(barbell, opinions, steps=int(consensus_time(barbell)) + 1, lazy=True)
    final = early.by_node()
    assert not early.agreed
    assert min(final[node] for node in range(5)) > max(final[node] for node in range(5, 10))
    vector, order = fiedler_vector(barbell)
    sides = {node: value >= 0 for node, value in zip(order, vector, strict=True)}
    assert len({sides[node] for node in range(5)}) == 1
    assert len({sides[node] for node in range(5, 10)}) == 1
