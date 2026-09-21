"""Chapter 33, held to answers that existed before the code did.

``tests/legendary`` has no directed classic -- every graph §53.4 lists is undirected -- so the
fixtures here are planted digraphs whose hierarchy was written down before it was measured:

* **A perfect binary tree of depth three.** 15 nodes, 14 arcs. Its GRC is arithmetic a reader
  can redo: the root reaches 14/14, the two nodes below it 6/14, the four below them 2/14, the
  eight leaves 0, so ``GRC = (0 + 2*(8/14) + 4*(12/14) + 8*1) / 14 = 176/196 = 0.8980``. That is
  the 0.89 the book reads off Figure 33.6 (p. 468), the network it calls "a pretty darn perfect
  hierarchy" that GRC still refuses to score 1. Its cycle share is 0, its agony 0 and its
  arborescence score 1, all three by construction.
* **A directed cycle.** Every node reaches every other, so every local reach is equal and GRC is
  0; every arc is inside the one strongly connected component, so the cycle share is 1 and
  §33.2's flow hierarchy is 0 -- *"any directed network composed by a single strongly connected
  component has a hierarchicalness of zero by definition"* (p. 466).
* **A DAG with one back-edge.** The back-edge climbs k levels, so §33.5 charges it ``k + 1``
  (p. 470) and that is the whole agony of the network.
* **Agony against its own dual.** Minimum agony is the linear-programming dual of a maximum
  circulation with unit capacities: the dual of ``min sum_e z_e`` s.t. ``r_u - r_v - z_e <= -1``
  is ``max sum_e f_e`` over circulations ``0 <= f <= 1``, and both are integral. So for any
  digraph, the agony computed here must equal ``networkx``'s min-cost flow answer to a problem
  this module never solves. That identity is Gupte et al.'s (2011) and is what makes the exact
  claim checkable rather than self-referential.
* **A planted two-boss network.** One node with two incoming arcs of different weight: the
  maximum spanning arborescence must keep the heavy one and drop the light one, which names the
  exact edge set the answer is.
* **A random tournament**, which has no hierarchy by construction: with an arc between every
  pair in a random direction it is strongly connected with high probability, so GRC is 0 and
  every degree-preserving rewiring of it scores the same -- the null model must find nothing.
"""

from __future__ import annotations

import json
import math
import random

import networkx as nx
import pytest

from graphrag.sna.hierarchy import (
    agony,
    analyse_hierarchy,
    arborescence,
    cycle_share,
    global_reach_centrality,
    hierarchy_payload,
    hierarchy_type,
    layered_layout,
    local_reach,
    render_hierarchy,
)


@pytest.fixture
def tree() -> nx.DiGraph:
    """A perfect binary tree of depth 3, pointing away from the root: §33.4's arborescence."""
    return nx.balanced_tree(2, 3, create_using=nx.DiGraph)


@pytest.fixture
def ring() -> nx.DiGraph:
    """A directed cycle over four nodes: one strongly connected component and nothing else."""
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])


@pytest.fixture
def back_edge() -> nx.DiGraph:
    """A four-node chain with one arc back to its top, three levels up (§33.5, Figure 33.8).

    Plus a shortcut ``a -> c``, which is what makes the layered ranking strictly better than the
    flat one: flattening the chain would charge the shortcut too. Without it the two are tied,
    since the agony of a bare cycle is its length whichever way you rank it.
    """
    return nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")])


def tournament(size: int, seed: int) -> nx.DiGraph:
    """Every pair joined once, in a direction a seeded coin picked. No planted hierarchy."""
    rng = random.Random(seed)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(size))
    for left in range(size):
        for right in range(left + 1, size):
            graph.add_edge(*((left, right) if rng.random() < 0.5 else (right, left)))
    return graph


def max_circulation(graph: nx.DiGraph) -> int:
    """The dual of §33.5's program: the most arcs an edge-disjoint set of cycles can cover.

    Each arc gets capacity 1 and cost -1, every node demands nothing, and the minimum-cost
    circulation is therefore the maximum number of arc-units that can flow round the network.
    By linear-programming duality this equals the minimum agony; ``networkx`` solves it with the
    network simplex, which shares no code with the linear program under test.
    """
    flow = nx.DiGraph()
    flow.add_nodes_from(graph.nodes(), demand=0)
    for u, v in graph.edges():
        if u != v:
            flow.add_edge(u, v, capacity=1, weight=-1)
    cost, _ = nx.network_simplex(flow)
    return -int(cost)


# ------------------------------------------------------------------- §33.1 the classification


def test_the_tree_is_an_arborescence_and_says_which_checks_decided_it(tree: nx.DiGraph) -> None:
    kind = hierarchy_type(tree)
    assert kind.name == "arborescence"
    assert kind.acyclic and kind.roots == 1 and kind.many_bosses == 0
    assert kind.reciprocity == 0.0 and kind.weak_components == 1
    assert "no bosses" in kind.sentence
    # Every check is printed, not only the verdict.
    assert any("in-degree 0" in reason for reason in kind.reasons)
    assert any("Reciprocity" in reason for reason in kind.reasons)


def test_the_shapes_the_chapter_separates_are_separated(ring: nx.DiGraph) -> None:
    """A DAG that is not a tree is not an arborescence, and a knot is not a DAG."""
    assert hierarchy_type(ring).name == "strongly connected"
    # Two bosses for one worker: acyclic, so §33.2 calls it perfect and §33.1 does not.
    two_bosses = nx.DiGraph([("r", "a"), ("r", "b"), ("a", "c"), ("b", "c")])
    assert hierarchy_type(two_bosses).name == "directed acyclic"
    assert hierarchy_type(two_bosses).many_bosses == 1
    # Two separate trees: every node has at most one boss, but there is no single head.
    forest = nx.DiGraph([("r", "a"), ("s", "b")])
    assert hierarchy_type(forest).name == "arborescence forest"
    assert hierarchy_type(nx.DiGraph()).name == "empty"
    # A cycle with a tail is cyclic without being one component.
    tailed = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c")])
    assert hierarchy_type(tailed).name == "cyclic"


def test_every_measure_refuses_an_undirected_network(tree: nx.DiGraph) -> None:
    """§33.1, p. 465: the chapter assumes direction, and without it each number is degenerate."""
    flat = nx.Graph(tree)
    for call in (
        hierarchy_type,
        cycle_share,
        global_reach_centrality,
        agony,
        arborescence,
        layered_layout,
        local_reach,
        analyse_hierarchy,
    ):
        with pytest.raises(ValueError, match="directed"):
            call(flat)


# ------------------------------------------------------------------------------ §33.2 cycles


def test_the_tree_has_no_edge_on_a_cycle_and_the_ring_has_nothing_else(
    tree: nx.DiGraph, ring: nx.DiGraph
) -> None:
    clean = cycle_share(tree)
    assert clean.on_cycles == 0
    assert clean.share == 0.0
    assert clean.flow_hierarchy == 1.0
    assert clean.knots == 0

    knotted = cycle_share(ring)
    assert knotted.on_cycles == 4
    assert knotted.share == 1.0
    assert knotted.flow_hierarchy == 0.0
    assert knotted.knots == 1 and knotted.largest == 4


def test_the_flow_hierarchy_is_the_share_of_arcs_outside_the_knots() -> None:
    """A planted network: a 3-cycle, a node feeding it and two hanging off it.

    Ten arcs, three of them inside the one strongly connected component, so §33.2's flow
    hierarchy is 7/10 -- the same count the book gets by condensing and summing the condensed
    weights (p. 466), reached without the bookkeeping.
    """
    graph = nx.DiGraph(
        [
            ("x", "y"),
            ("y", "z"),
            ("z", "x"),  # the knot
            ("in", "x"),
            ("in", "y"),
            ("x", "out1"),
            ("y", "out2"),
            ("out1", "out3"),
            ("out2", "out3"),
            ("out3", "out4"),
        ]
    )
    cycles = cycle_share(graph)
    assert cycles.edges == 10
    assert cycles.on_cycles == 3
    assert cycles.flow_hierarchy == pytest.approx(0.7)
    assert cycles.knots == 1 and cycles.largest == 3
    # networkx computes the same ratio from the other direction, and never sees this code.
    assert cycles.flow_hierarchy == pytest.approx(float(nx.flow_hierarchy(graph)))


def test_a_self_loop_is_an_edge_on_a_cycle_and_is_kept_out_of_everything_else() -> None:
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "a")])
    cycles = cycle_share(graph)
    assert cycles.self_loops == 1
    assert cycles.on_cycles == 1
    # The loop cannot be given a level that makes it free, so it is excluded rather than charged.
    assert agony(graph).total == 0.0
    assert arborescence(graph).edges == 2


# --------------------------------------------------------------- §33.3 global reach centrality


def test_the_local_reach_of_a_tree_is_the_share_of_the_network_below_each_node(
    tree: nx.DiGraph,
) -> None:
    """§14.3 as §33.3 uses it: the fraction of the other 14 nodes reachable from each."""
    reach = local_reach(tree)
    assert reach[0] == pytest.approx(14 / 14)
    assert reach[1] == pytest.approx(6 / 14)
    assert reach[3] == pytest.approx(2 / 14)
    assert reach[7] == 0.0


def test_the_perfect_binary_tree_scores_the_books_089(tree: nx.DiGraph) -> None:
    """p. 468: GRC on Figure 33.6 is 0.89, and a depth-3 binary tree is 176/196 = 0.898."""
    reach = global_reach_centrality(tree)
    assert reach.grc == pytest.approx(176 / 196)
    assert round(reach.grc, 2) == 0.90
    assert reach.maximum == 1.0
    assert reach.heads == (0,) and reach.head_count == 1


def test_a_star_scores_one_and_a_directed_cycle_scores_zero(ring: nx.DiGraph) -> None:
    """The two ends of §33.3's scale, both of which the chapter names (pp. 467-468)."""
    star = nx.DiGraph([("centre", f"leaf{i}") for i in range(6)])
    assert global_reach_centrality(star).grc == pytest.approx(1.0)

    cycle = global_reach_centrality(ring)
    assert cycle.grc == pytest.approx(0.0)
    assert cycle.head_count == 4  # everybody reaches everybody: no head at all
    assert global_reach_centrality(nx.DiGraph([(1, 2)]).subgraph([1])).nodes == 1


def test_grc_is_undefined_below_two_nodes() -> None:
    one = nx.DiGraph()
    one.add_node("alone")
    assert math.isnan(global_reach_centrality(one).grc)


# ----------------------------------------------------------------------- §33.4 arborescences


def test_the_tree_is_already_an_arborescence_and_nothing_is_dropped(tree: nx.DiGraph) -> None:
    result = arborescence(tree)
    assert result.spanning and result.refusal == ""
    assert result.roots == (0,)
    assert result.kept == result.edges == 14
    assert result.score == 1.0
    assert result.dropped == ()


def test_one_boss_per_node_keeps_the_heaviest_arc_into_each_node() -> None:
    """A planted two-boss network whose answer is fixed by the weights, not by a tie-break."""
    graph = nx.DiGraph()
    graph.add_edge("r", "a", weight=1)
    graph.add_edge("r", "b", weight=1)
    graph.add_edge("a", "c", weight=5)
    graph.add_edge("b", "c", weight=1)
    result = arborescence(graph)
    assert result.spanning
    assert result.roots == ("r",)
    assert set(result.tree.edges()) == {("r", "a"), ("r", "b"), ("a", "c")}
    assert [(u, v) for u, v, _ in result.dropped] == [("b", "c")]
    assert result.score == pytest.approx(3 / 4)
    assert result.weight_share == pytest.approx(7 / 8)


def test_a_cycle_loses_exactly_one_arc(ring: nx.DiGraph) -> None:
    """Every node has a boss, so one arc must go: §33.4's score is then 3/4."""
    result = arborescence(ring)
    assert result.spanning
    assert result.kept == 3 and result.score == pytest.approx(0.75)
    assert len(result.roots) == 1


def test_no_spanning_arborescence_is_refused_with_the_reason() -> None:
    """Two pieces, so no node reaches every other: the result is the forest of p. 469."""
    forest = nx.DiGraph([("r", "a"), ("r", "b"), ("s", "c"), ("s", "d"), ("c", "d")])
    result = arborescence(forest)
    assert not result.spanning
    assert "No spanning arborescence exists" in result.refusal
    assert "weakly connected pieces" in result.refusal
    assert set(result.roots) == {"r", "s"}
    assert result.kept == 4 and result.score == pytest.approx(4 / 5)

    # And the other symptom: one piece, but two nodes nobody points at.
    two_heads = nx.DiGraph([("r", "x"), ("s", "x")])
    assert "in-degree 0" in arborescence(two_heads).refusal


# -------------------------------------------------------------------------------- §33.5 agony


def test_a_dag_has_no_agony_and_a_back_edge_costs_its_climb(back_edge: nx.DiGraph) -> None:
    """§33.5, p. 470: an arc that climbs k levels costs k + 1, and a DAG costs nothing."""
    chain = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d")])
    assert agony(chain).total == 0.0
    assert agony(chain).backward_edges == 0

    # The same chain closed by d -> a, which climbs the three levels of the chain: 3 + 1.
    closed = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    result = agony(closed)
    assert result.total == 4.0
    assert result.exact and result.lower_bound == pytest.approx(4.0)
    assert result.total == max_circulation(closed)
    assert agony(back_edge).total == 4.0  # the shortcut is free under the layered ranking


def test_two_back_edges_of_different_span_are_priced_differently() -> None:
    """Figure 33.8: one edge against the flow, two structures, two agonies."""
    short = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("b", "a")])
    long = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert agony(short).total == 2.0  # one level apart: 1 - 0 + 1
    assert agony(long).total == 4.0  # three levels apart: 3 - 0 + 1
    assert agony(short).total == max_circulation(short)
    assert agony(long).total == max_circulation(long)


def test_the_exact_agony_equals_its_own_dual_on_networks_this_module_never_saw() -> None:
    """Gupte et al.'s duality, checked against ``networkx``'s min-cost flow on five digraphs."""
    for seed in range(5):
        graph = nx.gnp_random_graph(14, 0.25, seed=seed, directed=True)
        assert agony(graph).total == max_circulation(graph), seed


def test_the_heuristic_is_an_upper_bound_and_says_so(back_edge: nx.DiGraph) -> None:
    """Above the exact-program guard the descent runs instead, and the report is told."""
    result = agony(back_edge, max_exact_edges=1)
    assert not result.exact
    assert math.isnan(result.lower_bound)
    assert result.total >= agony(back_edge).total
    assert "upper bound" in result.note


def test_agony_names_the_arcs_it_charged(back_edge: nx.DiGraph) -> None:
    """The total is the finding; which arcs pay it is one optimum among several.

    Many rankings reach the same minimum -- flattening a whole cycle costs exactly as much as
    laying it out and paying for one arc -- so this asserts what duality guarantees at *every*
    optimum instead of the ranking this solver happened to return: the charges sum to the total,
    and every charged arc lies inside a strongly connected component, because the dual optimum
    is a circulation and a circulation decomposes into cycles.
    """
    result = agony(back_edge)
    assert result.backward_edges >= 1
    assert sum(cost for _, _, cost in result.backward) == result.total
    knots = {
        node: index
        for index, group in enumerate(nx.strongly_connected_components(back_edge))
        for node in group
    }
    assert all(knots[u] == knots[v] for u, v, _ in result.backward)
    assert result.per_edge == pytest.approx(result.total / back_edge.number_of_edges())


# --------------------------------------------------------------------- §33.6 drawing the layers


def test_the_tree_layers_by_depth_with_no_arc_pointing_up(tree: nx.DiGraph) -> None:
    layout = layered_layout(tree)
    assert [len(layer) for layer in layout.layers] == [1, 2, 4, 8]
    assert layout.layers[0] == (0,)
    assert layout.upward_edges == 0
    assert layout.dummies == ()
    assert layout.positions[0] == (0.0, -0.0)
    # Every arc goes down exactly one layer, so y falls by one across each of them.
    for u, v in tree.edges():
        assert layout.positions[v][1] == layout.positions[u][1] - 1


def test_every_arc_points_down_except_the_ones_agony_counts(back_edge: nx.DiGraph) -> None:
    """§33.6, p. 472: a drawing laid out by agony levels shows exactly what agony charged for."""
    ranks = agony(back_edge).ranks
    layout = layered_layout(back_edge, levels=ranks)
    charged = {(u, v) for u, v in back_edge.edges() if ranks[u] - ranks[v] + 1 > 0}
    assert set(layout.upward) == charged
    assert layout.upward_edges == len(charged) >= 1
    assert ("d", "a") in charged  # the arc that closes the cycle is charged at this optimum


def test_a_long_arc_gets_a_waypoint_on_every_layer_it_crosses() -> None:
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")])
    layout = layered_layout(graph)
    assert layout.depth == 4
    assert [(dummy.source, dummy.target, dummy.layer) for dummy in layout.dummies] == [
        ("a", "d", 1),
        ("a", "d", 2),
    ]


def test_the_layering_refuses_levels_that_do_not_cover_every_node(tree: nx.DiGraph) -> None:
    with pytest.raises(ValueError, match="cover every node"):
        layered_layout(tree, levels={0: 0})


def test_a_knot_is_the_only_thing_the_default_layering_cannot_send_downward() -> None:
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("c", "b"), ("c", "d")])
    layout = layered_layout(graph)
    # b and c are one strongly connected component, so they share a layer and neither arc
    # between them can point down. Nothing else in the network is upward.
    assert set(layout.upward) == {("b", "c"), ("c", "b")}
    assert layout.positions["b"][1] == layout.positions["c"][1]


# ------------------------------------------------------------------------ the report and null


def test_the_report_carries_the_frame_the_n_the_null_and_the_chapter(tree: nx.DiGraph) -> None:
    tree.graph["frame"] = "A planted tree."
    report = analyse_hierarchy(tree, samples=10, seed=7)
    text = "\n".join(render_hierarchy(report))
    assert "**Sampling frame.** A planted tree." in text
    assert "**n.** 15 nodes and 14 arcs" in text
    assert "configuration model" in text
    assert "**Implements.** §33.1" in text
    for section in ("§33.2", "§33.3", "§33.4", "§33.5", "§33.6"):
        assert section in text
    assert json.loads(json.dumps(hierarchy_payload(report)))["type"]["name"] == "arborescence"


def test_a_random_tournament_is_no_more_hierarchical_than_its_degrees() -> None:
    """No planted hierarchy, so the null must find nothing: |z| small on every score."""
    graph = tournament(12, seed=3)
    report = analyse_hierarchy(graph, samples=50, seed=11)
    assert report.samples > 0
    assert report.reach.grc == pytest.approx(0.0)  # strongly connected: everyone reaches everyone
    assert report.grc_null is not None and abs(report.grc_null.z) < 2
    assert report.flow_null is not None and abs(report.flow_null.z) < 2
    assert report.agony_null is not None and abs(report.agony_null.z) < 2


def test_a_planted_hierarchy_beats_its_own_degree_sequence_where_the_null_has_room() -> None:
    """A tree of out-degree 3 against its rewirings -- and what the null cannot see.

    GRC separates the tree from its rewirings by six null standard deviations. The other three
    scores cannot separate them at all, and that is a property of the null rather than of the
    tree: rewiring holds every in-degree fixed, so a rewiring of a network where no node has two
    bosses is another network where no node has two bosses. The flow hierarchy stays 1, the
    agony stays 0 and the arborescence score stays 1 in every sample. The report says so, since
    a z of 0 there means the null had no room to move.
    """
    planted = nx.balanced_tree(3, 3, create_using=nx.DiGraph)
    report = analyse_hierarchy(planted, samples=50, seed=5)
    assert report.samples >= 5
    assert report.grc_null is not None
    assert report.grc_null.observed > report.grc_null.null_mean
    assert report.grc_null.z > 3
    for locked in (report.flow_null, report.agony_null, report.arborescence_null):
        assert locked is not None
        assert locked.null_std == 0.0
        assert locked.observed == locked.null_mean


def test_the_report_says_when_no_null_could_be_built() -> None:
    """§19.1's directed swap rotates three arcs; two arcs have no rotation."""
    tiny = nx.DiGraph([("a", "b"), ("b", "c")])
    report = analyse_hierarchy(tiny, samples=10, seed=1)
    assert report.samples == 0
    text = "\n".join(render_hierarchy(report))
    assert "No null model could be built" in text
    assert "Not tested" in text
