"""Chapter 10 and §13.1-13.3, held to answers that existed before the code did.

Where each number comes from:

* **The karate club, measured in print.** Diameter 5, average path length 2.408, radius 3, 45
  triangles, 45 independent cycles. The first three are the standard published description of
  Zachary's graph (the 78-edge copy ``networkx`` ships, see ``tests/legendary``); the last two
  are arithmetic a reader can redo -- ``trace(A³)/6`` and ``|E| - |V| + c = 78 - 34 + 1``.
* **Identities the chapter states between two quantities.** ``A²_uu = k_u`` (p. 156),
  ``trace(A³)/6`` against ``networkx.triangles`` (§10.1-10.2), ``mutual/(mutual+asymmetric)``
  against :func:`graphrag.sna.measures.reciprocity` (§10.3), and the histogram of §13.3 summing
  to ``|V|(|V|-1)`` on a connected graph.
* **Planted graphs whose answer was written down first.** A digraph built as a three-node core
  with one node feeding it and two hanging off it, so its strongly connected components, its
  condensation and its in- and out-components are known by construction; the book's own
  reciprocity figure (five connected pairs, two reciprocated, 2/5); a path graph, whose
  traversal orders §13.1 predicts.
* **The estimate against the exact answer** on a 300-node graph, which is the only claim the
  sampled mode makes: the same mean, and a diameter that is a lower bound.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.measures import reciprocity, summary
from graphrag.sna.paths import (
    MAX_EXACT_NODES,
    analyse_paths,
    bfs_tree,
    closed_walks,
    components,
    cycle_space,
    dfs_tree,
    dyad_census,
    exploration_order,
    path_lengths,
    paths_payload,
    render_paths,
    triangle_count,
    walk_counts,
)
from tests.legendary import karate_club, les_miserables


@pytest.fixture
def core_and_tails() -> nx.DiGraph:
    """A planted digraph whose §10.4 answer is known before anything is computed.

    ``a -> b -> c -> a`` is one strongly connected component of three. ``in`` feeds it and is
    reached by nothing, so it is the in-component. ``c -> out -> tail`` hangs off it, so those
    two are the out-component. ``lonely`` has no edge at all, so it is its own weak component.
    Five strong components, two weak ones, and a condensation three hops deep.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [("a", "b"), ("b", "c"), ("c", "a"), ("in", "a"), ("c", "out"), ("out", "tail")]
    )
    graph.add_node("lonely")
    return graph


@pytest.fixture
def figure_10_6() -> nx.DiGraph:
    """§10.3's worked example: five connected pairs, two of them reciprocated, so 2/5 (p. 159)."""
    graph = nx.DiGraph()
    graph.add_edges_from(
        [("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "d"), ("d", "e"), ("e", "a")]
    )
    return graph


# ------------------------------------------------------------------ §10.1 walks and matrices


def test_a_power_of_the_adjacency_counts_walks_and_its_diagonal_is_the_degree() -> None:
    """§10.1: ``A^n_uv`` is the number of walks of length n, and ``A²_uu = k_u`` (pp. 155-156).

    The path graph 1-2-3-4 has exactly one walk of length two from 1 to 3 (via 2) and none from
    1 to 4, which is the book's "no way to go from 1 to 7 in two hops" case.
    """
    graph = nx.path_graph(4)
    squared, order = walk_counts(graph, 2)
    assert order == [0, 1, 2, 3]
    assert squared[0][2] == 1.0  # one walk: 0 -> 1 -> 2
    assert squared[0][3] == 0.0  # no walk of length two at all
    assert closed_walks(graph, 2) == dict(graph.degree())

    # And the karate club, where node 33 has degree 17: the special case of p. 156 at scale.
    club, _ = karate_club()
    assert closed_walks(club, 2) == dict(club.degree())

    # (I + A)^n counts walks of length n or less (p. 156): the self loop lets the walker wait.
    at_most, _ = walk_counts(graph, 2, at_most=True)
    assert at_most[0][1] == 2.0  # wait then step, or step then wait
    assert at_most[0][2] == 1.0


def test_a_self_loop_is_dropped_before_any_power_is_taken() -> None:
    """§10.1's readings hold for a matrix "binary and its diagonal is set to zero" (p. 155).

    ``a-b`` with a loop at ``a``: ``a`` has degree 3 in networkx's counting (the loop counts
    twice), and a raw ``A²`` would put 2 on its diagonal -- neither the degree nor anything
    else. Zeroing the diagonal first makes ``A²_uu = k_u`` true of the simple graph the section
    is talking about, which is the only graph the identity was ever about.
    """
    graph = nx.Graph([("a", "b")])
    graph.add_edge("a", "a")
    assert dict(graph.degree()) == {"a": 3, "b": 1}  # networkx counts a loop twice

    assert closed_walks(graph, 2) == {"a": 1, "b": 1}  # the degrees of the loop-free graph
    squared, order = walk_counts(graph, 2)
    assert order == ["a", "b"]
    assert np.array_equal(squared, np.eye(2))  # no walk of length two between a and b

    # The loop is not a triangle and not a cycle either (§10.2: a cycle is a path).
    assert triangle_count(graph) == 0
    assert cycle_space(graph).dimension == 0 and cycle_space(graph).kind == "tree"


def test_walk_counts_refuses_a_negative_length_and_an_oversized_matrix() -> None:
    graph = nx.path_graph(4)
    with pytest.raises(ValueError, match="negative length"):
        walk_counts(graph, -1)
    with pytest.raises(ValueError, match="dense 4 by 4"):
        walk_counts(graph, 2, max_nodes=3)
    identity, _ = walk_counts(graph, 0)
    assert np.array_equal(identity, np.eye(4))


def test_the_trace_of_a_cubed_over_six_is_the_triangle_count() -> None:
    """§10.1's closed-walk reading, divided by the six closed walks each triangle produces.

    The karate club has 45 triangles, and the two ways of counting them -- the matrix trace and
    ``networkx.triangles`` -- have to agree, because the fallback above the matrix-size limit is
    the second one.
    """
    club, _ = karate_club()
    assert triangle_count(club) == 45
    assert triangle_count(club, max_nodes=10) == 45  # the networkx branch: the same number
    assert sum(nx.triangles(club).values()) // 3 == 45
    # Per node, the same identity one step earlier: A³_uu counts the closed walks of length
    # three through u, which is twice the number of triangles u sits in (two directions).
    per_node = closed_walks(club, 3)
    assert per_node == {node: 2 * count for node, count in nx.triangles(club).items()}
    assert sum(per_node.values()) == 6 * 45

    assert triangle_count(nx.complete_graph(4)) == 4
    assert triangle_count(nx.path_graph(5)) == 0


# ----------------------------------------------------------------------------- §10.2 cycles


def test_the_cycle_space_of_the_karate_club_has_dimension_78_minus_34_plus_1() -> None:
    """§10.2 plus the standard count of independent cycles: ``|E| - |V| + c``."""
    club, _ = karate_club()
    space = cycle_space(club)
    assert space.dimension == 78 - 34 + 1 == 45
    assert space.kind == "cyclic"
    assert space.triangles == 45
    assert len(space.cycles) == 20  # TOP_NODES of the 45 basis cycles, shortest first
    assert sum(space.lengths.values()) == 45
    assert min(space.lengths) == 3
    assert all(len(cycle) <= len(space.cycles[-1]) for cycle in space.cycles)


def test_the_cycle_space_names_the_structures_of_figure_10_5() -> None:
    """The zoo of §10.2 (pp. 156-158): tree, forest, arborescence, directed tree, DAG, cyclic."""
    tree = nx.balanced_tree(2, 2)
    assert cycle_space(tree).kind == "tree"
    assert cycle_space(tree).dimension == 0

    forest = nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))
    assert cycle_space(forest).kind == "forest"

    # An arborescence: one root of in-degree zero, every other node of in-degree one (p. 158).
    arborescence = nx.DiGraph([("r", "a"), ("r", "b"), ("a", "c"), ("a", "d")])
    assert cycle_space(arborescence).kind == "arborescence"

    # Figure 10.5(c): the same tree with "that little pesky edge at the bottom" reversed, so a
    # node has in-degree two and the root is no longer unique. Still a directed tree.
    pesky = nx.DiGraph([("r", "a"), ("r", "b"), ("a", "c"), ("d", "c")])
    assert cycle_space(pesky).kind == "directed tree"

    dag = nx.DiGraph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    assert cycle_space(dag).kind == "directed acyclic"
    assert cycle_space(dag).dimension == 1  # acyclic with the directions, cyclic without them

    assert cycle_space(nx.DiGraph([("a", "b"), ("b", "a")])).kind == "directed cyclic"
    assert cycle_space(nx.Graph()).kind == "empty"


def test_a_large_network_gets_its_dimension_but_not_its_basis() -> None:
    """The dimension is three counts; the basis is as many cycles as the dimension says."""
    club, _ = karate_club()
    space = cycle_space(club, max_edges=10)
    assert space.dimension == 45 and space.triangles == 45
    assert space.cycles == () and space.lengths == {}
    assert "not enumerated" in space.note


# ------------------------------------------------------------------------- §10.3 reciprocity


def test_the_dyad_census_reproduces_the_books_own_reciprocity_figure(
    figure_10_6: nx.DiGraph,
) -> None:
    """§10.3, p. 159: five connected pairs, two reciprocated, "reciprocity is 2/5, or 40%"."""
    census = dyad_census(figure_10_6)
    assert (census.mutual, census.asymmetric) == (2, 3)
    assert census.connected == 5
    assert census.reciprocity == pytest.approx(0.4)
    # The third box the ratio hides: five nodes give ten unordered pairs, five of them empty.
    assert census.null == 5
    assert census.pairs == 5 * 4 // 2


def test_the_census_agrees_with_the_summary_reciprocity_on_every_directed_graph(
    figure_10_6: nx.DiGraph, core_and_tails: nx.DiGraph
) -> None:
    """Two implementations of §10.3's definition must not be allowed to drift apart."""
    for graph in (figure_10_6, core_and_tails):
        census = dyad_census(graph)
        assert census.reciprocity == pytest.approx(reciprocity(graph))
        assert census.reciprocity == pytest.approx(summary(graph)["reciprocity"])
        assert census.mutual + census.asymmetric + census.null == census.pairs


def test_a_dyad_census_is_refused_on_an_undirected_network() -> None:
    with pytest.raises(ValueError, match="defined on a directed network"):
        dyad_census(nx.path_graph(4))


# ------------------------------------------------------------------------- §10.4 components


def test_the_planted_digraph_has_the_components_it_was_built_with(
    core_and_tails: nx.DiGraph,
) -> None:
    """§10.4: strong components respect the directions, weak ones ignore them (pp. 161-162)."""
    report = components(core_and_tails)
    assert report.directed
    assert report.weak.kind == "weakly connected"
    assert report.weak.sizes == (6, 1)  # the lonely node is its own weak component
    assert report.weak.singletons == 1
    assert report.strong is not None
    assert report.strong.sizes == (3, 1, 1, 1, 1)  # the core, and four nodes on no cycle
    assert set(report.strong.largest) == {"a", "b", "c"}

    condensation = report.condensation
    assert condensation is not None
    assert condensation.nodes == 5 and condensation.edges == 3
    # in -> core -> out -> tail: four components end to end, so three edges between them.
    assert condensation.depth == 3
    assert condensation.core == 3
    assert condensation.in_component == 1  # "in"
    assert condensation.out_component == 2  # "out", "tail"
    assert condensation.other == 1  # "lonely"


def test_an_undirected_network_gets_one_component_reading(core_and_tails: nx.DiGraph) -> None:
    report = components(core_and_tails.to_undirected())
    assert not report.directed
    assert report.weak.kind == "connected" and report.weak.sizes == (6, 1)
    assert report.strong is None and report.condensation is None
    club, _ = karate_club()
    assert components(club).weak.giant_share == 1.0


# ------------------------------------------------------------------------ §13.1 exploration


def test_breadth_first_visits_every_neighbour_of_the_root_before_going_deeper() -> None:
    """§13.1's contrast (pp. 191-192), on a graph where the two orders cannot coincide.

    The star-of-paths: the root has three neighbours, each of which leads away down a chain.
    BFS takes all three neighbours first; DFS, using the section's own last-in-first-out queue,
    follows the last neighbour it pushed to the end of its chain, and the section's consequence
    holds -- "the very first neighbor of the root node will be the last node to be explored".
    """
    graph = nx.Graph()
    for branch in ("a", "b", "c"):
        graph.add_edges_from([("root", f"{branch}1"), (f"{branch}1", f"{branch}2")])

    breadth = exploration_order(graph, "root", "bfs")
    assert breadth[:4] == ["root", "a1", "b1", "c1"]
    assert set(breadth[4:]) == {"a2", "b2", "c2"}

    depth = exploration_order(graph, "root", "dfs")
    assert depth[0] == "root"
    assert depth.index("c2") < depth.index("a1")  # down one branch before starting another
    # The section's own consequence: of the root's three neighbours, the first one pushed is the
    # last one popped, because the queue is last-in-first-out.
    assert depth.index("c1") < depth.index("b1") < depth.index("a1")
    # And where a neighbour leads nowhere -- a plain star -- it is the very last node explored.
    assert exploration_order(nx.star_graph(3), 0, "dfs") == [0, 3, 2, 1]

    assert sorted(breadth) == sorted(depth) == sorted(graph.nodes)


def test_a_traversal_stops_at_the_component_and_follows_directions(
    core_and_tails: nx.DiGraph,
) -> None:
    """§10.4: a walk "cannot use edges pointing to the direction opposite" of its travel."""
    assert exploration_order(core_and_tails, "in") == ["in", "a", "b", "c", "out", "tail"]
    assert exploration_order(core_and_tails, "tail") == ["tail"]  # nothing leaves it
    assert exploration_order(core_and_tails, "lonely") == ["lonely"]
    with pytest.raises(ValueError, match="not a node"):
        exploration_order(core_and_tails, "nobody")
    with pytest.raises(ValueError, match="must be one of"):
        exploration_order(core_and_tails, "in", "astrology")  # type: ignore[arg-type]


def test_the_two_trees_span_what_the_traversal_reached() -> None:
    """The wrappers of §13.1: a BFS tree puts every node at its shortest-path distance."""
    graph = nx.path_graph(5)
    breadth = bfs_tree(graph, 0)
    assert breadth.number_of_edges() == 4
    assert nx.shortest_path_length(breadth, 0, 4) == nx.shortest_path_length(graph, 0, 4) == 4
    assert bfs_tree(graph, 0, depth_limit=2).number_of_nodes() == 3
    assert set(dfs_tree(graph, 0).nodes) == set(graph.nodes)


# --------------------------------------------------------------- §13.2-13.3 shortest paths


def test_the_karate_club_has_the_path_lengths_that_were_published_for_it() -> None:
    """Diameter 5, average path length 2.408, radius 3 (§13.3's three numbers, on the graph
    every one of them is quoted for)."""
    club, _ = karate_club()
    lengths = path_lengths(club)
    assert lengths.diameter == 5
    assert lengths.mean == pytest.approx(2.408199643, abs=1e-6)
    assert lengths.radius == 3
    assert lengths.median == 2
    assert not lengths.estimated and lengths.unreachable_pairs == 0
    # The centre and the periphery are what networkx computes independently of this module.
    assert set(lengths.center) == set(nx.center(club))
    assert set(lengths.periphery) == set(nx.periphery(club))
    assert lengths.eccentricity == pytest.approx(
        {node: float(value) for node, value in nx.eccentricity(club).items()}
    )


def test_the_histogram_holds_every_ordered_pair_of_a_connected_network() -> None:
    """§13.3: "The number of total shortest paths is |V|(|V|-1)" when everything is reachable."""
    graph, known = les_miserables()
    lengths = path_lengths(graph)
    assert lengths.nodes == known.nodes == 77
    assert sum(lengths.histogram.values()) == 77 * 76 == lengths.reachable_pairs
    assert lengths.unreachable_pairs == 0
    # §13.3: "the number of paths of length one is twice the number of edges".
    assert lengths.histogram[1] == 2 * known.edges
    assert lengths.mean == pytest.approx(nx.average_shortest_path_length(graph))
    assert lengths.diameter == nx.diameter(graph)


def test_a_disconnected_network_is_measured_on_its_giant_component_and_says_so() -> None:
    """§13.3's convention: unreachable pairs are infinite, so the giant component is measured."""
    graph = nx.disjoint_union(nx.path_graph(6), nx.path_graph(2))
    graph.add_node("island")
    lengths = path_lengths(graph)
    assert lengths.nodes == 6 and lengths.network_nodes == 9
    assert lengths.diameter == 5 and lengths.unreachable_pairs == 0
    assert "3 node(s) sit outside this component" in lengths.note
    assert "The diameter of the whole network is infinite." in lengths.note
    assert "6 of 9 nodes (66.7%)" in lengths.frame


def test_a_directed_network_counts_the_pairs_no_directed_path_joins(
    core_and_tails: nx.DiGraph,
) -> None:
    """A giant *weak* component still holds unreachable ordered pairs; they are not averaged in.

    The 6-node weak component has 30 ordered pairs. Eighteen are joined by a directed path (the
    in-node reaches all five others, the three core nodes reach four or five each, ``out``
    reaches one) and the other twelve are not, which the section counts rather than calls zero.
    """
    lengths = path_lengths(core_and_tails)
    assert lengths.component_kind == "weakly connected"
    assert lengths.reachable_pairs == 18
    assert lengths.unreachable_pairs == 30 - 18 == 12
    assert lengths.diameter == 5  # in -> a -> b -> c -> out -> tail
    assert "not joined by any directed path" in lengths.note
    assert "a node that reaches almost nothing looks central" in lengths.note
    # "tail" reaches nobody, so it has no eccentricity at all rather than an eccentricity of 0.
    assert "tail" not in lengths.eccentricity
    assert lengths.eccentricity["in"] == 5


def test_a_network_with_no_edges_reports_no_distance_at_all() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c"])
    lengths = path_lengths(graph)
    assert lengths.mean == 0.0 and lengths.diameter == 0.0
    assert lengths.reachable_pairs == 0 and lengths.unreachable_pairs == 6
    assert "infinite distance" in lengths.note


def test_a_weighted_length_is_a_cost_and_has_no_hop_histogram() -> None:
    """§13.2's weighted case (fig. 13.5b) and §6.3's warning about which way a weight runs."""
    graph = nx.Graph()
    graph.add_edge("a", "b", cost=1.0)
    graph.add_edge("b", "c", cost=1.0)
    graph.add_edge("a", "c", cost=5.0)
    hops = path_lengths(graph)
    costs = path_lengths(graph, weight="cost")
    assert hops.diameter == 1 and hops.histogram == {1: 6}
    assert costs.diameter == 2.0  # a -> b -> c costs 2, which beats the direct edge's 5
    assert costs.histogram == {}
    assert "not hop counts" in costs.note and "affinity" in costs.note


def test_a_negative_weight_is_refused_with_the_reason_section_13_2_gives() -> None:
    """§13.2, pp. 196-197: over a negative edge the shortest *walk* is unbounded.

    The book's own figure 13.8(a): going back and forth over the negative edge finds an
    equivalent path every time, so there is "an infinite loop of shorter and shorter paths
    without ever reaching the destination". Dijkstra cannot tell a path from a walk, so the
    question is refused rather than answered with a number.
    """
    graph = nx.Graph()
    graph.add_edge("a", "b", cost=2.0)
    graph.add_edge("b", "c", cost=-1.0)
    with pytest.raises(ValueError, match="infinite loop of shorter and shorter paths"):
        path_lengths(graph, weight="cost")
    # Hops are unaffected: a negative weight cannot make an edge count as fewer than one edge.
    assert path_lengths(graph).diameter == 2


def test_the_estimate_agrees_with_the_exact_answer_on_a_three_hundred_node_graph() -> None:
    """The one claim the sampled mode makes: same mean, and a diameter that is a lower bound.

    A Watts-Strogatz small world of 300 nodes, built with a fixed seed so the graph itself is
    fixed. The exact answer solves all 89,700 ordered pairs; the estimate solves 60 sources'
    worth of them. The mean must land close, and the diameter must not exceed the true one --
    that is what "lower bound" means, and it is the only direction the error can go.
    """
    graph = nx.watts_strogatz_graph(300, 6, 0.1, seed=7)
    exact = path_lengths(graph)
    assert not exact.estimated and exact.nodes == 300

    estimate = path_lengths(graph, max_exact_nodes=100, sources=60, seed=11)
    assert estimate.estimated and estimate.sources == 60
    assert estimate.mean == pytest.approx(exact.mean, rel=0.05)
    assert estimate.median == exact.median
    assert estimate.diameter <= exact.diameter
    assert estimate.radius >= exact.radius
    assert "lower bound" in estimate.note and "seed 11" in estimate.note
    assert sum(estimate.histogram.values()) == estimate.reachable_pairs == 60 * 299

    # And it is reproducible: the same seed draws the same sources.
    assert path_lengths(graph, max_exact_nodes=100, sources=60, seed=11).mean == estimate.mean


# ----------------------------------------------------------------------------- the section


def test_the_section_prints_its_frame_its_n_its_null_and_its_chapter() -> None:
    """The four things every report section in this package has to carry."""
    club, _ = karate_club()
    report = analyse_paths(club, seed=3)
    text = "\n".join(render_paths(report))
    assert "## Paths and components" in text
    assert "**Chapter.** Atlas ch. 10" in text and "§13.1-13.3" in text
    assert "**Sampling frame.** 34 members of one university karate club" in text
    assert "**n.** 34 nodes and 78 edges" in text
    assert "**Null model.** None:" in text
    assert "| average path length | 2.4082 |" in text
    assert "| diameter | 5 |" in text
    assert "| radius | 3 |" in text
    assert "| 5 | 16 | 1.4% |" in text  # the histogram's last bar: the diameter's own pairs
    assert "45 triangle(s)" in text and "dimension 45" in text
    assert "A walk is not a path" in text
    assert "Reciprocity and the dyad census" not in text  # undirected: no census to print


def test_the_section_prints_the_census_and_the_condensation_on_a_directed_network(
    core_and_tails: nx.DiGraph,
) -> None:
    report = analyse_paths(core_and_tails, seed=3)
    text = "\n".join(render_paths(report))
    assert "| weakly connected | 2 |" in text and "| strongly connected | 5 |" in text
    assert "**The DAG of strongly connected components.**" in text
    # The depth is edges; the sentence must not call three edges "three components deep".
    assert "longest chain crosses 3 edge(s), so 4 component(s) end to end" in text
    assert "in-component (1 nodes that can reach it" in text
    assert "### Reciprocity and the dyad census (§10.3)" in text
    assert "| mutual (both edges) | 0 |" in text
    payload = paths_payload(report)
    assert payload["components"]["strongly connected"]["largest"] == 3
    assert payload["condensation"]["depth"] == 3
    assert payload["dyads"]["reciprocity"] == 0.0
    assert payload["cycles"]["kind"] == "directed cyclic"
    assert payload["path_lengths"]["unreachable_pairs"] == 12


def test_the_small_world_line_prints_both_numbers_and_claims_neither() -> None:
    """§13.3 makes the claim; the number it is checked against belongs to a later chapter."""
    club, _ = karate_club()
    world = analyse_paths(club, seed=3).small_world
    assert world.observed == pytest.approx(2.4082, abs=1e-3)
    assert world.expected == pytest.approx(math.log(34) / math.log(2 * 78 / 34), abs=1e-6)
    assert "it is an expectation, not a test" in world.sentence
    assert "ch. 17" in world.sentence

    # Undefined where the logarithm is: a single edge has mean degree 1, so ln <k> is zero.
    undefined = analyse_paths(nx.Graph([("a", "b")]), seed=3).small_world
    assert math.isnan(undefined.expected)
    assert "undefined here" in undefined.sentence


def test_the_default_bound_is_the_one_the_module_documents() -> None:
    """A change to the estimate threshold has to be a deliberate edit, not a drift."""
    assert MAX_EXACT_NODES == 1_000
