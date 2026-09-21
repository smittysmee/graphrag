"""Chapter 40 over hand-planted multilayer networks, so every answer is known before it runs.

Every :class:`graphrag.sna.layers.Multilayer` here is built by hand -- never through the store --
because the point of this file is graphs whose communities, redundancy and modularity can be
computed with a pencil first.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

from graphrag.sna.layers import Multilayer
from graphrag.sna.multilayer import (
    AGREEMENT_HIGH,
    analyze_multilayer_communities,
    complementarity,
    layer_agreement,
    layer_by_layer_communities,
    multilayer_modularity_value,
    omega_sweep,
    optimize_multilayer_modularity,
    redundancy,
)


def _ml(graphs: dict[str, nx.Graph], omega: float = 1.0) -> Multilayer:
    nodes = tuple(sorted({node for graph in graphs.values() for node in graph.nodes}))
    return Multilayer(
        persona_id="test",
        network="entities",
        layering="stance",
        names=tuple(graphs),
        graphs=graphs,
        nodes=nodes,
        omega=omega,
        frame="a hand-planted network for testing",
    )


def _clique(nodes: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            graph.add_edge(a, b, weight=3)
    return graph


# ----------------------------------------------------------------------------- shared planting


def _two_layers_same_communities() -> Multilayer:
    """Two layers, each two triangles: the same communities hold in both (§40.1-40.3)."""
    layer = nx.Graph()
    for group in (["a1", "a2", "a3"], ["b1", "b2", "b3"]):
        for i, u in enumerate(group):
            for v in group[i + 1 :]:
                layer.add_edge(u, v, weight=3)
    return _ml({"l1": layer.copy(), "l2": layer.copy()})


# ----------------------------------------------------------------------------- §40.1-40.3


def test_all_three_methods_recover_the_same_planted_communities() -> None:
    ml = _two_layers_same_communities()
    report = analyze_multilayer_communities(ml, threshold=2, seed=1, runs=5, samples=5)

    # §40.1: flattening the two identical layers sums the weights but keeps the same boundary.
    flat_groups = {frozenset(c) for c in report.flatten_result.communities}
    assert flat_groups == {frozenset(["a1", "a2", "a3"]), frozenset(["b1", "b2", "b3"])}

    # §40.2: each layer's own Louvain recovers the same split, so the two per-layer communities
    # overlap fully (6 of 6 members) and merge into one multilayer community per group.
    lbl = report.layer_by_layer
    merged_groups = {mlc.members for mlc in lbl.multilayer_communities}
    assert frozenset(["a1", "a2", "a3"]) in merged_groups
    assert frozenset(["b1", "b2", "b3"]) in merged_groups
    assert not lbl.unaffiliated

    # §40.3: the same two groups come back inside *each* layer of the supra-adjacency, at the
    # default omega. The two layers need not share a *label number* for the same group -- that
    # is the pillar/flat distinction the omega sweep tests separately -- only the same members.
    by_layer_label: dict[tuple[str, int], set[str]] = {}
    for (node, layer), label in report.supra.node_labels.items():
        by_layer_label.setdefault((layer, label), set()).add(node)
    per_layer_groups: dict[str, set[frozenset[str]]] = {"l1": set(), "l2": set()}
    for (layer, _label), members in by_layer_label.items():
        per_layer_groups[layer].add(frozenset(members))
    expected = {frozenset(["a1", "a2", "a3"]), frozenset(["b1", "b2", "b3"])}
    assert per_layer_groups["l1"] == expected
    assert per_layer_groups["l2"] == expected

    # §40.5: identical layers agree perfectly, so flattening is what gets recommended.
    assert report.agreement.mean_agreement == pytest.approx(1.0)
    assert report.agreement.recommended == "flatten"


def test_omega_zero_decomposes_into_the_two_layers_own_partitions() -> None:
    """Layer 1 splits {A, B}; layer 2 splits {A, C}. At omega=0 each layer answers alone."""
    a = ["a1", "a2", "a3"]
    b = ["b1", "b2", "b3"]
    c = ["c1", "c2", "c3"]
    layer1 = nx.compose(_clique(a), _clique(b))
    layer1.add_nodes_from(c)  # present in the network, isolated in this layer
    layer2 = nx.compose(_clique(a), _clique(c))
    layer2.add_nodes_from(b)
    ml = _ml({"l1": layer1, "l2": layer2})

    result = optimize_multilayer_modularity(ml, omega=0.0, seed=3)
    # No coupling can move a node, so layer 1's supra-nodes group exactly as layer 1's own edges
    # would (A and B, two communities) and layer 2's group exactly as its own edges would (A and
    # C, two different communities), independently of each other.
    assert result.node_labels[("a1", "l1")] == result.node_labels[("a2", "l1")]
    assert result.node_labels[("a1", "l1")] == result.node_labels[("a3", "l1")]
    assert result.node_labels[("b1", "l1")] == result.node_labels[("b2", "l1")]
    assert result.node_labels[("a1", "l1")] != result.node_labels[("b1", "l1")]
    assert result.node_labels[("a1", "l2")] == result.node_labels[("a2", "l2")]
    assert result.node_labels[("c1", "l2")] == result.node_labels[("c2", "l2")]
    assert result.node_labels[("a1", "l2")] != result.node_labels[("c1", "l2")]


def test_a_large_omega_forces_pillar_communities() -> None:
    a = ["a1", "a2", "a3"]
    b = ["b1", "b2", "b3"]
    c = ["c1", "c2", "c3"]
    layer1 = nx.compose(_clique(a), _clique(b))
    layer1.add_nodes_from(c)
    layer2 = nx.compose(_clique(a), _clique(c))
    layer2.add_nodes_from(b)
    ml = _ml({"l1": layer1, "l2": layer2})

    result = optimize_multilayer_modularity(ml, omega=1000.0, seed=3)
    assert result.consistency == 1.0


def test_the_omega_sweep_shows_consistency_rising_with_omega() -> None:
    a = ["a1", "a2", "a3"]
    b = ["b1", "b2", "b3"]
    c = ["c1", "c2", "c3"]
    layer1 = nx.compose(_clique(a), _clique(b))
    layer1.add_nodes_from(c)
    layer2 = nx.compose(_clique(a), _clique(c))
    layer2.add_nodes_from(b)
    ml = _ml({"l1": layer1, "l2": layer2}, omega=0.0)

    rows = {row.omega: row for row in omega_sweep(ml, omegas=(0.0, 0.1, 1.0, 100.0), seed=3)}
    assert rows[0.0].consistency < rows[100.0].consistency
    assert rows[100.0].consistency == 1.0


def test_supra_modularity_matches_the_mucha_formula_by_hand() -> None:
    """Two nodes, two layers, one edge in layer 1 -- the whole B matrix fits on a napkin."""
    layer1 = nx.Graph([("u", "v")])
    layer2 = nx.Graph()
    layer2.add_nodes_from(["u", "v"])
    ml = _ml({"l1": layer1, "l2": layer2}, omega=1.0)

    order = [("u", "l1"), ("v", "l1"), ("u", "l2"), ("v", "l2")]
    from graphrag.sna.multilayer import modularity_matrix

    matrix, computed_order, two_mu = modularity_matrix(ml, omega=1.0, gamma=1.0)
    assert computed_order == order
    assert two_mu == pytest.approx(6.0)  # 2 (the one layer-1 edge) + 4 (four coupling entries)

    one_community = dict.fromkeys(order, 0)
    assert multilayer_modularity_value(matrix, order, one_community, two_mu) == pytest.approx(
        4.0 / 6.0
    )

    pillars = {("u", "l1"): 0, ("u", "l2"): 0, ("v", "l1"): 1, ("v", "l2"): 1}
    assert multilayer_modularity_value(matrix, order, pillars, two_mu) == pytest.approx(3.0 / 6.0)

    # The true optimum on this tiny network is the single community (4/6), which beats the
    # pillar split (3/6) by capturing the real layer-1 edge and both coupling terms at once. The
    # single-level local mover documented in optimize_multilayer_modularity's own docstring does
    # not find it from singleton starts: from (u, l1), joining (v, l1) (B = 0.5) and joining
    # (u, l2) (B = 1.0, the coupling) are both on the table at once; the greedy move takes the
    # larger one first, and the community that results can never later discover that merging
    # everything was better. It converges to the pillar split instead -- exactly the local
    # optimum this module's own docstring admits to.
    best = optimize_multilayer_modularity(ml, omega=1.0, seed=1)
    assert best.modularity == pytest.approx(3.0 / 6.0)


# ----------------------------------------------------------------------------- §40.4 density


def test_redundancy_matches_the_books_own_worked_example() -> None:
    """Figure 40.11a (p. 582): five nodes, three layers, redundancy 18/30 = 0.6."""
    k4 = _clique(["n1", "n2", "n3", "n4"])
    k4.add_node("n5")  # in the network, isolated in every layer
    ml = _ml({"l1": k4.copy(), "l2": k4.copy(), "l3": k4.copy()})
    result = redundancy(ml, ["n1", "n2", "n3", "n4", "n5"])
    assert (result.numerator, result.denominator) == (18, 30)
    assert result.value == pytest.approx(0.6)


def test_complementarity_variety_and_exclusivity_match_the_books_own_example() -> None:
    """Figure 40.11b (p. 582-583): variety 1.0, exclusivity 9/10, by construction below."""
    red = nx.Graph([("n1", "n2"), ("n1", "n3"), ("n1", "n4"), ("n1", "n5"), ("n2", "n3")])
    green = nx.Graph([("n1", "n2"), ("n2", "n4"), ("n2", "n5"), ("n3", "n4"), ("n3", "n5")])
    blue = nx.Graph([("n4", "n5")])
    for graph in (red, green, blue):
        graph.add_nodes_from(["n1", "n2", "n3", "n4", "n5"])
    ml = _ml({"red": red, "green": green, "blue": blue})

    community = ["n1", "n2", "n3", "n4", "n5"]
    result = complementarity(ml, community)
    assert result.edge_counts == {"red": 5, "green": 5, "blue": 1}
    assert result.variety == pytest.approx(1.0)
    assert result.exclusivity == pytest.approx(0.9)
    # Homogeneity is this module's own definition (see DEPARTURE_HOMOGENEITY), computed here by
    # the same formula the module uses rather than the book's unreproducible 0.33.
    sigma = math.sqrt(((5 - 11 / 3) ** 2 + (5 - 11 / 3) ** 2 + (1 - 11 / 3) ** 2) / 3)
    sigma_max = 11 * math.sqrt(2) / 3
    assert result.homogeneity == pytest.approx(1 - sigma / sigma_max)

    rd = redundancy(ml, community)
    assert (rd.numerator, rd.denominator) == (11, 30)


def test_density_is_undefined_for_a_community_of_one_or_a_single_layer_network() -> None:
    ml = _ml({"only": nx.Graph([("n1", "n2")])})
    assert math.isnan(redundancy(ml, ["n1"]).value)
    assert math.isnan(complementarity(ml, ["n1", "n2"]).variety)  # only one layer


# ----------------------------------------------------------------------------- §40.5


def test_the_recommendation_flips_with_what_the_layers_share() -> None:
    triangle_a = ["1", "2", "3"]
    triangle_b = ["4", "5", "6"]
    agreeing = _ml(
        {
            "l1": nx.compose(_clique(triangle_a), _clique(triangle_b)),
            "l2": nx.compose(_clique(triangle_a), _clique(triangle_b)),
        }
    )
    agreement = layer_agreement(agreeing, seed=1, runs=5)
    assert agreement.mean_agreement == pytest.approx(1.0)
    assert agreement.recommended == "flatten"
    assert agreement.mean_agreement >= AGREEMENT_HIGH

    # Layer 2 re-groups the same six nodes into an orthogonal split: {1,4,5} and {2,3,6}. The
    # adjusted Rand index between two independent bisections of six elements this way is -1/9.
    scrambled = _ml(
        {
            "l1": nx.compose(_clique(triangle_a), _clique(triangle_b)),
            "l2": nx.compose(_clique(["1", "4", "5"]), _clique(["2", "3", "6"])),
        }
    )
    disagreement = layer_agreement(scrambled, seed=1, runs=5)
    assert disagreement.mean_agreement == pytest.approx(-1.0 / 9.0)
    assert disagreement.recommended == "layer-by-layer"
    assert disagreement.mean_agreement < AGREEMENT_HIGH


# ----------------------------------------------------------------------------- §40.2 edge cases


def test_a_layer_community_can_belong_to_two_maximal_sets_at_once() -> None:
    """One layer-community bridging two others, the way Figure 40.2's C1L2 does (p. 574).

    Three per-layer communities, each a single clique so each layer has exactly one Louvain
    community: X = {1..5}, Y = {3..7}, Z = {5..9}. |X n Y| = |{3,4,5}| = 3 and |Y n Z| =
    |{5,6,7}| = 3, both at the threshold, while |X n Z| = |{5}| = 1 is below it -- so the overlap
    graph is the path X-Y-Z, not a triangle, and its maximal cliques are {X, Y} and {Y, Z}: Y (and
    only node 5, X n Y n Z) sits in both.
    """
    ml = _ml(
        {
            "l1": _clique([str(i) for i in range(1, 6)]),
            "l2": _clique([str(i) for i in range(3, 8)]),
            "l3": _clique([str(i) for i in range(5, 10)]),
        }
    )
    result = layer_by_layer_communities(ml, threshold=3, seed=1, runs=5)
    merged = {mlc.members for mlc in result.multilayer_communities}
    assert merged == {frozenset(["3", "4", "5"]), frozenset(["5", "6", "7"])}
    assert set(result.unaffiliated) == {"1", "2", "8", "9"}
    assert "more than one multilayer community" in result.overlap_note
    assert result.overlap_note.startswith("1 node(s)")
