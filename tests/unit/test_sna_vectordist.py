"""Chapter 47's node vector distances, held to answers worked out before the code ran.

Every assertion here comes from one of three places, and none of them is "the function returned
something":

* **First-step equations solved by hand.** The 3-node path's Laplacian pseudoinverse is inverted
  by hand from ``L + J/n`` (§8.4, §11.4) and checked against ``numpy.linalg.pinv`` -- see
  ``test_the_laplacian_pseudoinverse_of_a_three_node_path_matches_the_hand_computed_matrix``.
* **Closed forms for graphs whose structure is a formula.** Effective resistance on a tree is the
  hop count (already the convention ``tests/legendary.py`` and ``test_sna_walks.py`` hold
  ``graphrag.sna.walks`` to), so on a path, the generalized Euclidean distance between two unit
  masses is ``√(hop count)``; the earth-mover distance between unit masses at the two ends of a
  length-``n`` path is exactly ``n``; the Dirichlet-energy diagnostic of a vector constant on
  every edge's two endpoints is the hand-summed ``Σ w(x_u - x_v)²`` over the edges that actually
  differ, and ``graph_fourier_distance`` -- p. 692's own ``‖L(p-q)‖₂`` -- is checked against the
  same hand-multiplied ``L`` applied to that difference.
* **Algebraic identities.** Pearson-style self- and negation-correlation (``rho(x,x)=1``,
  ``rho(x,-x)=-1``) follow from the quadratic form cancelling exactly, checked here rather than
  merely asserted; the plain Euclidean distance between "mass swapped between two far nodes" and
  "mass swapped between two adjacent nodes" is provably identical (both are a swap of one unit
  between two 0/1 indicator vectors), which is §47.2's own point about what Euclidean cannot see.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.matrices import dense, laplacian, node_order
from graphrag.sna.vectordist import (
    MAX_SUPPORT,
    NodeVector,
    correlation_distance,
    cosine_distance,
    earth_mover_distance,
    euclidean_distance,
    generalized_euclidean_distance,
    graph_fourier_distance,
    graph_fourier_smoothness,
    network_correlation,
    network_variance,
    resolve_vector,
    shortest_path_linkage_distance,
)
from graphrag.sna.walks import laplacian_pseudoinverse

# ----------------------------------------------------------------------------- §47.1 baselines


def test_euclidean_distance_is_the_straight_line_rope() -> None:
    assert euclidean_distance(np.array([4.0, 5.0]), np.array([1.0, 1.0])) == pytest.approx(5.0)


def test_cosine_distance_is_zero_on_the_same_ray_and_one_at_a_right_angle() -> None:
    # p. 682's own point: two points on the same ray are cosine-adjacent however far apart.
    assert cosine_distance(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(0.0)
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_cosine_distance_is_undefined_for_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_distance(np.array([0.0, 0.0]), np.array([1.0, 0.0]))


def test_correlation_distance_is_zero_for_perfect_agreement_and_two_for_perfect_reversal() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert correlation_distance(x, x).distance == pytest.approx(0.0, abs=1e-9)
    assert correlation_distance(x, -x).distance == pytest.approx(2.0, abs=1e-9)


# ------------------------------------------------------------------- §47.2 generalized Euclidean


def test_the_laplacian_pseudoinverse_of_a_three_node_path_matches_the_hand_computed_matrix() -> (
    None
):
    """``L = [[1,-1,0],[-1,2,-1],[0,-1,1]]``, inverted by hand via ``(L + J/3)^-1 - J/3``.

    ``(L + J/3)^-1 = (1/27) [[24,6,-3],[6,15,6],[-3,6,24]]`` (adjugate over determinant, both
    computed by hand); subtracting ``J/3 = (9/27) [[1,1,1],[1,1,1],[1,1,1]]`` gives
    ``L† = (1/27) [[15,-3,-12],[-3,6,-3],[-12,-3,15]]``, i.e. ``(1/9) [[5,-1,-4],[-1,2,-1],
    [-4,-1,5]]``. Every row sums to 0, the signature of a pseudoinverse whose kernel is the
    all-ones vector.
    """
    path = nx.path_graph(3)
    order = node_order(path)
    pinv, returned_order = laplacian_pseudoinverse(path, order, weight=None)
    assert returned_order == [0, 1, 2]
    hand = np.array([[5.0, -1.0, -4.0], [-1.0, 2.0, -1.0], [-4.0, -1.0, 5.0]]) / 9.0
    np.testing.assert_allclose(pinv, hand, atol=1e-9)
    assert np.allclose(pinv.sum(axis=1), 0.0, atol=1e-9)


def test_generalized_euclidean_discriminates_far_from_near_mass_while_euclidean_cannot() -> None:
    """§47.2's own point (Figure 47.5), on the path 0-1-2-3.

    Swapping a unit of mass between the two *far* ends (0 and 3) and swapping it between two
    *adjacent* nodes (0 and 1) are both, to plain Euclidean, "one unit moved between two 0/1
    indicator entries" -- the distance is ``√2`` either way. The generalized Euclidean distance
    is ``√Ω(u,v)``, and on a tree ``Ω`` is the hop count, so it reads ``√3`` for the far swap and
    ``√1 = 1`` for the near one: strictly larger for the farther pair, which plain Euclidean
    could never say.
    """
    path = nx.path_graph(4)
    order = node_order(path)
    pinv, _ = laplacian_pseudoinverse(path, order, weight=None)
    far_p, far_q = np.array([1.0, 0, 0, 0]), np.array([0, 0, 0, 1.0])
    near_p, near_q = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])

    assert euclidean_distance(far_p, far_q) == pytest.approx(math.sqrt(2))
    assert euclidean_distance(near_p, near_q) == pytest.approx(math.sqrt(2))
    assert euclidean_distance(far_p, far_q) == pytest.approx(euclidean_distance(near_p, near_q))

    far = generalized_euclidean_distance(pinv, far_p, far_q)
    near = generalized_euclidean_distance(pinv, near_p, near_q)
    assert far == pytest.approx(math.sqrt(3))
    assert near == pytest.approx(1.0)
    assert far > near


def test_generalized_euclidean_across_components_is_each_sides_own_spread_not_zero() -> None:
    """Two 2-node components (single edges): L-dagger's cross term is 0, but each side's own
    diagonal term is not, so the distance is neither 0 nor an error -- see the docstring caveat.

    A single edge has ``Ω=1``, so each component's own ``L†[u,u] = Ω/4 = 0.25`` (a single-edge
    graph's pseudoinverse is ``[[0.25,-0.25],[-0.25,0.25]]``, checked directly). The generalized
    Euclidean between one unit of mass on either side is then ``√(0.25 + 0.25) = √0.5``.
    """
    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))  # nodes 0-1 and 2-3
    order = node_order(graph)
    pinv, _ = laplacian_pseudoinverse(graph, order, weight=None)
    assert pinv[0, 2] == pytest.approx(0.0)  # no cross term between components
    p, q = np.array([1.0, 0, 0, 0]), np.array([0, 0, 1.0, 0])
    assert generalized_euclidean_distance(pinv, p, q) == pytest.approx(math.sqrt(0.5))


# ------------------------------------------------------------- §47.3 shortest-path / earth-mover


@pytest.mark.parametrize("length", [2, 3, 5])
def test_earth_mover_distance_between_unit_masses_at_the_two_ends_of_a_path_is_its_length(
    length: int,
) -> None:
    path = nx.path_graph(length + 1)  # nodes 0..length
    assert earth_mover_distance(path, {0: 1.0}, {length: 1.0}) == pytest.approx(float(length))


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_every_linkage_agrees_with_earth_mover_for_two_unit_node_vectors(linkage: str) -> None:
    """With one unit-mass node on each side there is only one possible pairing, so every strategy
    of §47.3 -- greedy single, greedy complete, the weighted average, and the LP optimum -- must
    agree: they all move the one unit of mass along the one path there is, 3 hops long."""
    path = nx.path_graph(6)
    p, q = {1: 1.0}, {4: 1.0}
    assert shortest_path_linkage_distance(path, p, q, linkage) == pytest.approx(3.0)
    assert earth_mover_distance(path, p, q) == pytest.approx(3.0)


def test_shortest_path_distances_return_infinity_rather_than_raise_when_disconnected() -> None:
    graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
    p, q = {0: 1.0}, {2: 1.0}
    assert math.isinf(shortest_path_linkage_distance(graph, p, q, "single"))
    assert math.isinf(shortest_path_linkage_distance(graph, p, q, "complete"))
    assert math.isinf(shortest_path_linkage_distance(graph, p, q, "average"))
    assert math.isinf(earth_mover_distance(graph, p, q))


def test_shortest_path_distance_refuses_an_oversized_support() -> None:
    star = nx.star_graph(MAX_SUPPORT + 5)
    p = dict.fromkeys(range(MAX_SUPPORT + 5), 1.0)
    q = dict.fromkeys(range(MAX_SUPPORT + 5), 1.0)
    with pytest.raises(ValueError, match="max_support"):
        shortest_path_linkage_distance(star, p, q, "single")
    with pytest.raises(ValueError, match="max_support"):
        earth_mover_distance(star, p, q)


def test_average_linkage_rescales_the_lighter_vector_before_averaging() -> None:
    """p. 688's rule: the lighter vector is scaled up so both sides carry the same total mass.

    ``p = {0: 1.0}``, lighter than ``q = {3: 2.0}``, is scaled to ``{0: 2.0}`` first (§47.3:
    "if q had a lower sum, we transform it"). With a single pairing on each side the
    average-linkage formula ``Σ p_u q_v |P_u,v| / Σp`` reduces to ``q_v * hops`` once ``p``'s
    total equals ``q``'s: ``2.0 * 3 = 6.0``, the *equalised* mass carried across the one path,
    not the raw hop count of 3.0 the unit-mass case above gets.
    """
    path = nx.path_graph(4)  # 0-1-2-3, hop distance 3 apart
    p, q = {0: 1.0}, {3: 2.0}  # q carries twice p's mass
    assert shortest_path_linkage_distance(path, p, q, "average") == pytest.approx(6.0)


# ------------------------------------------------------------------- §47.4 graph Fourier transform


def test_a_vector_constant_on_each_of_two_disjoint_cliques_is_zero_under_both_readings() -> None:
    """This case does *not* discriminate ``graph_fourier_distance`` (p. 692's ``‖L(p-q)‖``) from
    ``graph_fourier_smoothness`` (the Dirichlet energy diagnostic, ``xᵀLx``): ``Lx = 0`` for any
    vector constant on every connected component, because each node's row of ``L`` sums its own
    degree against its neighbours' shared value, which cancels exactly. Both formulas therefore
    read 0 here; ``test_the_two_formulas_disagree_on_an_alternating_vector`` is the case that
    tells them apart.
    """
    graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    order = node_order(graph)
    x = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])  # constant within each clique
    zero = np.zeros(6)
    assert graph_fourier_smoothness(graph, x, order) == pytest.approx(0.0, abs=1e-9)
    assert graph_fourier_distance(graph, x, zero, order) == pytest.approx(0.0, abs=1e-9)


def test_the_two_formulas_disagree_on_an_alternating_vector() -> None:
    """Clique {0,1,2} of ``nx.complete_graph(3)`` at ``x = (1,-1,1)``, clique {3,4,5} constant at
    5 (so it contributes nothing to either quantity, per the test above).

    **Dirichlet energy** (``graph_fourier_smoothness``, the diagnostic, *not* ch. 47's formula):
    ``xᵀLx = Σ_(u,v)∈E (x_u-x_v)²`` over clique {0,1,2}'s three edges: ``(0,1)`` contributes
    ``(1-(-1))²=4``, ``(0,2)`` contributes ``(1-1)²=0``, ``(1,2)`` contributes ``(-1-1)²=4``,
    total 8.

    **``graph_fourier_distance`` (p. 692's own formula, ‖L(p-q)‖₂, here p=x, q=0 so p-q=x).**
    ``L`` for a 3-clique is ``[[2,-1,-1],[-1,2,-1],[-1,-1,2]]`` (degree 2 on the diagonal, -1 for
    every edge). By hand: ``(Lx)_0 = 2(1) - (-1)(-1) - (-1)(1) = 2 - 1 + 1 = 2``,
    ``(Lx)_1 = -(-1)(1) + 2(-1) - (-1)(1) = 1 - 2 + 1 = -2``... expanded fully,
    ``Lx = (2, -4, 2)`` on the first clique and ``(0, 0, 0)`` on the constant second one, so
    ``‖Lx‖² = 2² + (-4)² + 2² = 4 + 16 + 4 = 24`` and ``‖Lx‖ = √24``.

    The two numbers -- 8 and √24 ≈ 4.899 -- are different quantities on purpose: this is exactly
    the case ``test_a_vector_constant_on_each_of_two_disjoint_cliques_is_zero_under_both_readings``
    cannot show, because both vanish there.
    """
    graph = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    order = node_order(graph)
    x = np.array([1.0, -1.0, 1.0, 5.0, 5.0, 5.0])
    zero = np.zeros(6)
    assert graph_fourier_smoothness(graph, x, order) == pytest.approx(8.0)
    assert graph_fourier_distance(graph, x, zero, order) == pytest.approx(math.sqrt(24.0))


def test_graph_fourier_distance_is_the_norm_of_the_laplacian_applied_to_the_difference() -> None:
    """p. 692's formula directly, on a path graph, independent of the Dirichlet-energy
    diagnostic."""
    graph = nx.path_graph(3)
    order = node_order(graph)
    p, q = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    matrix, _ = laplacian(graph, order, kind="combinatorial")
    expected = float(np.linalg.norm(dense(matrix) @ (p - q)))
    assert graph_fourier_distance(graph, p, q, order) == pytest.approx(expected)


# ----------------------------------------------------------------- §47.5 network-space statistics


def test_network_variance_is_zero_concentrated_and_grows_spread_on_a_path() -> None:
    path = nx.path_graph(3)  # 0-1-2, Ω(0,1)=1, Ω(1,2)=1, Ω(0,2)=2 (series resistors)
    order = node_order(path)
    concentrated = np.array([1.0, 0.0, 0.0])
    spread = np.array([1.0, 1.0, 1.0])  # normalised internally to (1/3, 1/3, 1/3)
    var_concentrated = network_variance(path, concentrated, order)
    var_spread = network_variance(path, spread, order)
    assert var_concentrated == pytest.approx(0.0, abs=1e-9)
    # by hand: 0.5 * (1/9) * (Ω(0,1)^2 + Ω(1,0)^2 + Ω(1,2)^2 + Ω(2,1)^2 + Ω(0,2)^2 + Ω(2,0)^2)
    #        = 0.5 * (1/9) * (1 + 1 + 1 + 1 + 4 + 4) = 0.5 * 12/9 = 2/3
    assert var_spread == pytest.approx(2.0 / 3.0)
    assert var_spread > var_concentrated


def test_network_variance_needs_positive_total_mass() -> None:
    path = nx.path_graph(3)
    order = node_order(path)
    with pytest.raises(ValueError, match="positive total mass"):
        network_variance(path, np.array([0.0, 0.0, 0.0]), order)


def test_network_correlation_is_one_with_itself_and_minus_one_with_its_negation() -> None:
    path = nx.path_graph(5)
    order = node_order(path)
    x = np.array([1.0, 2.0, 0.5, 3.0, -1.0])
    assert network_correlation(path, x, x, order) == pytest.approx(1.0)
    assert network_correlation(path, x, -x, order) == pytest.approx(-1.0)


def test_network_correlation_is_undefined_for_a_constant_vector() -> None:
    path = nx.path_graph(4)
    order = node_order(path)
    x = np.array([1.0, 2.0, 0.5, 3.0])
    constant = np.full(4, 7.0)
    with pytest.raises(ValueError, match="constant on every connected component"):
        network_correlation(path, x, constant, order)


# ----------------------------------------------------------------------------- resolve_vector


def test_node_vector_array_defaults_absent_nodes_to_zero_and_present_counts_only_seen_ones() -> (
    None
):
    vector = NodeVector(spec="attr:x", values={"a": 3.0, "c": 1.0}, source="test")
    order = ["a", "b", "c"]
    np.testing.assert_array_equal(vector.array(order), np.array([3.0, 0.0, 1.0]))
    assert vector.present(order) == 2


def test_resolve_vector_refuses_a_spec_it_does_not_understand() -> None:
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="not a vector spec"):
        resolve_vector("nonsense:thing", graph)


def test_resolve_vector_reads_a_centrality_directly_off_the_graph() -> None:
    graph = nx.path_graph(3)
    vector = resolve_vector("centrality:degree", graph)
    assert vector.spec == "centrality:degree"
    assert set(vector.values) == {0, 1, 2}


def test_resolve_vector_reads_a_built_in_numeric_attribute() -> None:
    graph = nx.Graph()
    graph.add_node("a", mentions=5)
    graph.add_node("b", mentions=0)
    vector = resolve_vector("attr:mentions", graph)
    assert vector.values == {"a": 5.0, "b": 0.0}


def test_resolve_vector_centrality_rejects_an_unknown_kind_with_the_valid_list() -> None:
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="centrality must be one of"):
        resolve_vector("centrality:nonsense", graph)


def test_resolve_vector_stance_requires_the_entities_network() -> None:
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="--network entities"):
        resolve_vector("stance:praise", graph, network="speakers")


def test_resolve_vector_stance_needs_a_store_and_a_persona() -> None:
    graph = nx.path_graph(3)
    with pytest.raises(ValueError, match="needs a persona and a store"):
        resolve_vector("stance:praise", graph, network="entities")
