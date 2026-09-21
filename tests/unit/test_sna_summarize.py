"""Chapter 46's four techniques, held to answers worked out by hand before the code ran.

* **Figure 46.2, reproduced by hand.** Two planted 4-cliques joined by one edge, grouped by
  community: the meta-network is two super nodes of internal density 1 and one super edge of
  weight 1. Grouped instead by an attribute that splits each clique in half and crosses both,
  the internal and between densities are the same fraction (1/3 and 9/16) counted by hand from
  the thirteen edges the construction gives.
* **Figure 46.4's inequality, on a graph whose block structure is known before it is built.** The
  two-clique network compresses to fewer bits than its own edge list; a graph with no block
  structure at the same node/edge count and the same fixed grouping costs more bits than the
  two-clique network does, because every one of its blocks needs corrections the clique network's
  diagonal blocks do not.
* **A star, for simplification and the influence summary.** Degree centrality ranks the hub above
  every leaf, so keeping one node keeps the hub. An independent cascade seeded at the hub
  saturates the star in one step; seeded at a leaf, it takes two -- the leaf has to reach the hub
  before the hub can reach the other leaves -- so the hub ranks first.
* **A partition dealt out by hand for the attribute grouping**, checked against
  `graphrag.sna.attributes.attribute_sources`' own vocabulary (`OWN`, `INHERITED`).
"""

from __future__ import annotations

import json
import math

import networkx as nx
import pytest

from graphrag.sna.export import INHERITED, OWN, attr_key, attr_source_key
from graphrag.sna.summarize import (
    FRAME,
    aggregate,
    compress,
    groups_for,
    influence_summary,
    render_summary,
    simplify_by_importance,
    summarize,
    summary_payload,
)

# --------------------------------------------------------------------------- planted graphs


def _clique(nodes: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]


@pytest.fixture
def two_cliques() -> nx.Graph:
    """Two K4 cliques, {a,b,c,d} and {e,f,g,h}, joined by exactly one edge, d-e.

    13 edges total: six inside each clique, one between them. Every count below is checked by
    hand in the module docstring above.
    """
    graph = nx.Graph()
    left, right = ["a", "b", "c", "d"], ["e", "f", "g", "h"]
    graph.add_edges_from(_clique(left))
    graph.add_edges_from(_clique(right))
    graph.add_edge("d", "e")
    graph.graph["frame"] = "two planted 4-cliques joined by one edge"
    assert graph.number_of_edges() == 13
    return graph


@pytest.fixture
def star() -> nx.Graph:
    """A hub with three leaves -- small enough that a cascade run is instant either way."""
    graph = nx.Graph()
    graph.add_edges_from([("hub", "leaf_a"), ("hub", "leaf_b"), ("hub", "leaf_c")])
    graph.graph["frame"] = "one hub, three leaves"
    return graph


# --------------------------------------------------------------------------- §46.1 aggregation


def test_aggregate_two_cliques_by_community(two_cliques: nx.Graph) -> None:
    """Fig 46.2's own shape: two dense blocks joined by one edge become two super nodes of
    density 1 and one super edge of weight 1."""
    groups = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    result = aggregate(two_cliques, groups, ["left", "right"], by="community")

    assert [node.size for node in result.supernodes] == [4, 4]
    assert [node.internal_edges for node in result.supernodes] == [6, 6]
    assert [node.possible_edges for node in result.supernodes] == [6, 6]
    assert all(node.internal_density == 1.0 for node in result.supernodes)

    assert len(result.superedges) == 1
    edge = result.superedges[0]
    assert edge.edges == 1
    assert edge.possible == 16
    assert edge.density == pytest.approx(1 / 16)
    assert result.left_out == 0
    assert result.frame == "two planted 4-cliques joined by one edge"


def test_aggregate_by_attribute_crossing_both_cliques(two_cliques: nx.Graph) -> None:
    """An attribute that splits each clique in half and crosses both: densities computed by hand
    from the thirteen edges of `two_cliques` (module docstring works the arithmetic out)."""
    x_group, y_group = ["a", "c", "e", "g"], ["b", "d", "f", "h"]
    result = aggregate(two_cliques, [x_group, y_group], ["x", "y"], by="attr:half")

    by_label = {node.label: node for node in result.supernodes}
    assert by_label["x"].internal_edges == 2  # a-c, e-g
    assert by_label["y"].internal_edges == 2  # b-d, f-h
    assert by_label["x"].internal_density == pytest.approx(1 / 3)
    assert by_label["y"].internal_density == pytest.approx(1 / 3)

    assert len(result.superedges) == 1
    between = result.superedges[0]
    assert between.edges == 9
    assert between.possible == 16
    assert between.density == pytest.approx(9 / 16)


def test_aggregate_rejects_overlapping_groups(two_cliques: nx.Graph) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        aggregate(two_cliques, [["a", "b"], ["b", "c"]], ["one", "two"], by="community")


def test_aggregate_left_out_and_directed_flag(two_cliques: nx.Graph) -> None:
    result = aggregate(two_cliques, [["a", "b", "c", "d"]], ["left"], by="community", left_out=4)
    assert result.left_out == 4
    assert any("carried no value" in note for note in result.notes)
    assert result.directed is False


# --------------------------------------------------------------------------- §46.2 compression


def test_compress_two_cliques_beats_its_own_edge_list(two_cliques: nx.Graph) -> None:
    """Two cliques joined by one edge, worked out by hand against `_list_bits`'s per-correction
    code (`log2(possible)` per wrong pair, *not* the combinatorial `log2(C(possible, wrong))`
    that is symmetric under complementation and would make the "connected" declaration free):

    * both diagonal blocks (4-cliques, possible=C(4,2)=6) are declared connected with 0 wrong
      pairs, so they cost 0 correction bits each;
    * the one off-diagonal block (possible=4*4=16) has 1 actual edge (the bridge), which is far
      below half of 16, so it is declared "not connected" and that single edge is the one wrong
      pair: 1 * log2(16) = 4.0 correction bits;
    * partition_bits = log2(C(8,4)) (which of the 8 nodes are "left") = log2(70) ~= 6.1293;
    * model_bits = log2(C(3,2)) (2 of the 3 group-pairs -- both diagonals -- declared connected)
      = log2(3) ~= 1.5850;
    * total_bits = 6.1293 + 1.5850 + 4.0 ~= 11.7142, against raw_bits = log2(C(28,13)) ~= 25.1582
      for the plain edge list with no grouping at all.
    """
    groups = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    result = compress(two_cliques, groups, ["left", "right"], by="community")

    assert result.groups == 2
    assert result.possible_pairs == 3  # (left,left), (right,right), (left,right)
    assert result.declared_pairs == 2  # both cliques declared "connected"; the bridge is not
    assert result.mistakes == 1  # the single bridge edge is the one correction needed

    assert result.partition_bits == pytest.approx(math.log2(70))
    assert result.model_bits == pytest.approx(math.log2(3))
    assert result.correction_bits == pytest.approx(1 * math.log2(16))
    assert result.total_bits == pytest.approx(math.log2(70) + math.log2(3) + math.log2(16))
    assert result.raw_bits == pytest.approx(math.log2(37_442_160))  # C(28, 13)
    assert result.total_bits < result.raw_bits
    assert result.ratio < 1.0


def test_compress_no_block_structure_costs_more_than_two_cliques(two_cliques: nx.Graph) -> None:
    """A graph with the same node and edge count as `two_cliques` but no block structure at the
    same fixed 4/4 split costs more bits than the two-clique network does, at that same split:
    every block is close to half-full, so `_list_bits` charges close to `possible * log2(possible)`
    bits per block rather than the near-zero cost the two-clique network's diagonal blocks pay."""
    clique_groups = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    clique_result = compress(two_cliques, clique_groups, ["left", "right"], by="community")

    random_graph = nx.gnm_random_graph(8, 13, seed=7)
    random_graph = nx.relabel_nodes(random_graph, {i: str(i) for i in random_graph.nodes})
    nodes = sorted(random_graph.nodes, key=int)
    flat_groups = [nodes[:4], nodes[4:]]
    random_result = compress(random_graph, flat_groups, ["one", "two"], by="community")

    assert random_result.total_bits > clique_result.total_bits


def test_compress_flattens_a_directed_graph() -> None:
    directed = nx.DiGraph()
    directed.add_edges_from([("a", "b"), ("b", "a"), ("b", "c")])
    result = compress(directed, [["a", "b"], ["c"]], ["one", "two"], by="community")
    assert any("flattened" in note or "computed" in note for note in result.notes)


# --------------------------------------------------------------------------- §46.3 simplification


def test_simplify_keeps_the_hub_of_a_star(star: nx.Graph) -> None:
    result = simplify_by_importance(star, method="degree", keep=1)

    assert result.keep == 1
    kept = [row for row in result.ranked if row.kept]
    assert [row.node for row in kept] == ["hub"]
    assert result.ranked[0].node == "hub"
    # The three leaves tie on degree, so the order the ranking gives is the id order.
    dropped = [row.node for row in result.ranked if not row.kept]
    assert dropped == sorted(dropped)
    assert result.nodes_after == 1
    assert result.edges_after == 0


def test_simplify_share_default_and_rejects_both_keep_and_share(star: nx.Graph) -> None:
    default = simplify_by_importance(star)  # neither keep nor share: SIMPLIFY_DEFAULT_SHARE
    assert default.keep == round(0.5 * 4)

    with pytest.raises(ValueError, match="keep or share"):
        simplify_by_importance(star, keep=1, share=0.5)


def test_simplify_rejects_unknown_method(star: nx.Graph) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        simplify_by_importance(star, method="not-a-centrality", keep=1)


# --------------------------------------------------------------------------- §46.4 influence


def test_influence_summary_ranks_the_hub_first(star: nx.Graph) -> None:
    """A deterministic cascade (beta=1): from the hub every leaf is infected in one step; from a
    leaf, the hub is reached in step 1 and the other leaves only in step 2. The hub is therefore
    the faster spreader and ranks first, exactly the seed SPINE's argument would keep the edges
    of."""
    result = influence_summary(
        star, candidates=["hub", "leaf_a"], beta=1.0, attempts=1, runs=1, seed=0
    )

    by_node = {row.node: row for row in result.ranked}
    assert by_node["hub"].steps_to_saturate == pytest.approx(1.0)
    assert by_node["leaf_a"].steps_to_saturate == pytest.approx(2.0)
    assert result.ranked[0].node == "hub"
    assert by_node["hub"].reach_share == pytest.approx(1.0)
    assert by_node["hub"].saturated_share == pytest.approx(1.0)
    assert result.kept_edges == result.total_edges  # a tree: every edge is somebody's only path
    assert result.caveat  # SPREAD_FRAME is always carried


def test_influence_summary_empty_graph() -> None:
    result = influence_summary(nx.Graph())
    assert result.ranked == []
    assert result.candidates == 0


# --------------------------------------------------------------------------- attribute grouping


def test_groups_for_attribute_reports_left_out_and_borrowed_share() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("n1", "n2"), ("n2", "n3"), ("n3", "n4")])
    graph.nodes["n1"][attr_key("region")] = "north"
    graph.nodes["n1"][attr_source_key("region")] = OWN
    graph.nodes["n2"][attr_key("region")] = "north"
    graph.nodes["n2"][attr_source_key("region")] = OWN
    graph.nodes["n3"][attr_key("region")] = "south"
    graph.nodes["n3"][attr_source_key("region")] = INHERITED
    # n4 carries no value for "region" at all.

    groups, labels, left_out, borrowed_share, note = groups_for(graph, "attr:region")

    assert labels == ["north", "south"]
    assert sorted(groups[labels.index("north")]) == ["n1", "n2"]
    assert groups[labels.index("south")] == ["n3"]
    assert left_out == 1
    assert borrowed_share == pytest.approx(1 / 3)
    assert note == ""


def test_groups_for_rejects_bad_by() -> None:
    with pytest.raises(ValueError, match=r"community.*attr:<key>"):
        groups_for(nx.Graph(), "nonsense")
    with pytest.raises(ValueError, match=r"attr:<key>"):
        groups_for(nx.Graph(), "attr:")


# --------------------------------------------------------------------------- the whole report


def test_summarize_runs_the_community_evaluation_battery(two_cliques: nx.Graph) -> None:
    """Wave 4's rule: every partition `sna summarize` produces is scored by chapter 36's battery."""
    report = summarize(two_cliques, by="community", simplify_share=0.5, seed=0)

    assert report.evaluation is not None
    assert report.evaluation.communities == 2
    assert report.evaluation.modularity > 0

    rendered = "\n".join(render_summary(report))
    assert "## Community evaluation" in rendered
    assert "## Aggregation (§46.1)" in rendered
    assert "## Compression (§46.2)" in rendered
    assert "## Simplification (§46.3)" in rendered
    assert "## Influence-based summary (§46.4)" in rendered
    assert "backboning" in rendered  # the chapter's own distinction, in FRAME


def test_summarize_threads_resolution_into_the_evaluation(two_cliques: nx.Graph) -> None:
    """A non-default `resolution` must reach `evaluate_partition`, or modularity at 0.5 is
    scored as if the partition had been found at chapter 36's default of 1.0 (§36.1, p. 517:
    modularity at one resolution is a different function from modularity at another)."""
    report = summarize(two_cliques, by="community", resolution=0.5, seed=0)

    assert report.evaluation is not None
    assert report.evaluation.resolution == 0.5

    # Cross-checked against the battery run directly at both resolutions: modularity itself
    # (unlike the resolution field) need not differ on every graph, so the field is what proves
    # the parameter was threaded through, and the direct call proves `summarize` did not just
    # relabel a resolution=1.0 run.
    from graphrag.sna.evaluate import evaluate_partition

    groups = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    direct = evaluate_partition(two_cliques, groups, resolution=0.5, seed=0)
    assert report.evaluation.modularity == pytest.approx(direct.modularity)
    assert report.evaluation.modularity != pytest.approx(
        evaluate_partition(two_cliques, groups, resolution=1.0, seed=0).modularity
    )


def test_summarize_frame_states_the_backboning_distinction() -> None:
    assert "backboning" in FRAME
    assert "merge" in FRAME or "super node" in FRAME


def test_summary_payload_is_json_serialisable(two_cliques: nx.Graph) -> None:
    report = summarize(two_cliques, by="community", seed=0)
    payload = summary_payload(report)
    encoded = json.dumps(payload)  # raises on nan or non-serialisable content
    assert json.loads(encoded)["by"] == "community"
    assert math.isfinite(payload["compression"]["total_bits"])


def test_summarize_by_attribute_with_no_groups_raises() -> None:
    graph = nx.Graph()
    graph.add_edge("n1", "n2")
    with pytest.raises(ValueError, match="no groups"):
        summarize(graph, by="attr:region")
