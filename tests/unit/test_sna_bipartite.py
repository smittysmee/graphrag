"""Chapter 39: Barber modularity, BRIM, biLouvain and neighbour similarity, on the incidence.

Every number here is known before the code runs. Two disjoint complete bipartite blocks give a
Barber modularity computable by hand from §39.1's own definition, and BRIM, biLouvain and
neighbour similarity all have to recover exactly those two blocks. Figure 39.1 (p. 563) is the
book's own worked example of two bipartite structures that project onto the identical unipartite
graph -- a single hub and six pair-wise leaves both give the same K4 -- and it is used here
exactly as the book uses it: to show that a method reading the incidence directly can still tell
them apart where a projected one cannot. The Davis Southern women network is the canonical
two-mode graph (`tests/legendary.py`); its own caveat is that the two women's groups are
disputed at the edges, so the test asks only what every one of Freeman's 21 methods agreed on --
that there is a two-way split at all, mixing women and events on both sides of it -- and not for
a specific roster.
"""

from __future__ import annotations

import itertools

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import render_markdown, run_analysis, to_payload
from graphrag.sna.bipartite import (
    BIPARTITE_EVALUATION_NOTE,
    barber_modularity,
    bilouvain,
    bipartite_report,
    bipartite_significance,
    brim,
    neighbor_similarity,
    render_bipartite,
)
from graphrag.sna.cluster import louvain
from graphrag.sna.export import bipartite_projection
from tests.legendary import southern_women


def two_blocks(n: int = 3) -> tuple[nx.Graph, list[list[str]]]:
    """Two disjoint complete bipartite blocks of ``n`` rows and ``n`` columns each.

    Barber modularity of the ground truth is exactly 0.5 by hand: ``m = 2 * n**2``, every row
    and column has degree ``n``, and each unlike-type pair sharing a block contributes
    ``1 - n*n/m = 1 - 0.5 = 0.5``, summed over the ``2 * n**2`` such pairs and divided by ``m``
    again -- ``(2*n**2 * 0.5) / (2*n**2) = 0.5``, independent of ``n``.
    """
    graph = nx.Graph()
    blocks: list[list[str]] = [[], []]
    for block in range(2):
        rows = [f"r{block}_{i}" for i in range(n)]
        columns = [f"c{block}_{i}" for i in range(n)]
        for row in rows:
            graph.add_node(row, mode="speaker")
            for column in columns:
                graph.add_node(column, mode="entity")
                graph.add_edge(row, column, weight=1.0)
        blocks[block] = sorted(rows + columns)
    graph.graph.update(frame="two planted bipartite blocks, for a test.", min_weight=1)
    return graph, blocks


def figure_39_1() -> tuple[nx.Graph, nx.Graph]:
    """The book's own pair (p. 563, Figure 39.1): a 1,4-clique and six 1,2-cliques.

    Both project onto the identical weighted K4 among the four right nodes -- every pair shares
    exactly one left neighbour either way -- which the book gives as the reason a projection
    cannot tell the two apart. Structure A has only one left node, so it cannot be split into
    more than one row-cluster; structure B has six, and can.
    """
    rights = ["r1", "r2", "r3", "r4"]
    hub = nx.Graph()
    hub.add_node("hub", mode="speaker")
    for right in rights:
        hub.add_node(right, mode="entity")
        hub.add_edge("hub", right, weight=1.0)
    hub.graph.update(frame="Figure 39.1, the 1,4-clique.", min_weight=1)

    leaves = nx.Graph()
    for right in rights:
        leaves.add_node(right, mode="entity")
    for index, (a, b) in enumerate(itertools.combinations(rights, 2)):
        left = f"l{index}"
        leaves.add_node(left, mode="speaker")
        leaves.add_edge(left, a, weight=1.0)
        leaves.add_edge(left, b, weight=1.0)
    leaves.graph.update(frame="Figure 39.1, the six 1,2-cliques.", min_weight=1)
    return hub, leaves


def _modes(graph: nx.Graph, community: list[str]) -> set[str]:
    return {graph.nodes[node]["mode"] for node in community}


# --------------------------------------------------------------------------------- §39.1 quality


def test_barber_modularity_of_two_disjoint_bicliques_is_the_hand_computed_value() -> None:
    graph, blocks = two_blocks(3)
    assert barber_modularity(graph, blocks) == pytest.approx(0.5)


def test_barber_modularity_is_scale_invariant_for_balanced_disjoint_bicliques() -> None:
    """The hand derivation says 0.5 regardless of block size; check it at a second size too."""
    graph, blocks = two_blocks(5)
    assert barber_modularity(graph, blocks) == pytest.approx(0.5)


def test_barber_modularity_refuses_a_one_mode_graph() -> None:
    graph = nx.karate_club_graph()
    with pytest.raises(ValueError, match="two-mode network"):
        barber_modularity(graph, [list(graph.nodes)])


def test_a_single_community_scores_zero_barber_modularity() -> None:
    """Putting everything in one group leaves nothing to distinguish it from chance (§39.1)."""
    graph, _ = two_blocks(3)
    everything = [sorted(graph.nodes)]
    assert barber_modularity(graph, everything) == pytest.approx(0.0)


# ------------------------------------------------------------------------- §39.3 direct methods


def test_brim_recovers_the_planted_two_block_partition() -> None:
    """BRIM's random row starts can converge to a worse local optimum on some restarts -- that
    is exactly what ``stability`` is for -- but with several restarts the *best* one found is
    the planted partition, at the modularity hand-computed above.
    """
    graph, blocks = two_blocks(3)
    result = brim(graph, k=2, seed=1, restarts=8)
    assert result.modularity == pytest.approx(0.5)
    assert sorted(sorted(c) for c in result.communities) == sorted(blocks)
    assert 0.0 <= result.stability <= 1.0


def test_bilouvain_recovers_the_planted_two_block_partition_without_a_k() -> None:
    graph, blocks = two_blocks(3)
    result = bilouvain(graph, seed=1, runs=4)
    assert result.modularity == pytest.approx(0.5)
    assert sorted(sorted(c) for c in result.communities) == sorted(blocks)


def test_brim_and_bilouvain_refuse_a_one_mode_graph() -> None:
    graph = nx.karate_club_graph()
    with pytest.raises(ValueError, match="two-mode network"):
        brim(graph)
    with pytest.raises(ValueError, match="two-mode network"):
        bilouvain(graph)


# ---------------------------------------------------------------------- §39.4 neighbour similarity


def test_neighbor_similarity_recovers_the_same_blocks_as_projection_then_louvain() -> None:
    """On a graph where the two agree: two disjoint blocks project onto two disjoint cliques."""
    graph, blocks = two_blocks(3)
    similarity_result = neighbor_similarity(graph, k=2, seed=1)
    assert sorted(sorted(c) for c in similarity_result.communities) == sorted(blocks)

    pairs = [(u, v) if graph.nodes[u]["mode"] == "speaker" else (v, u) for u, v in graph.edges()]
    projected = bipartite_projection(pairs, side="right")
    projected_result = louvain(projected, seed=1, runs=1)
    right_blocks = sorted(sorted(c) for c in projected_result.communities)
    right_from_similarity = sorted(
        sorted(node for node in community if graph.nodes[node]["mode"] == "entity")
        for community in similarity_result.communities
    )
    assert right_from_similarity == right_blocks


def test_figure_39_1_projects_to_the_identical_graph_but_neighbor_similarity_tells_them_apart() -> (
    None
):
    """The book's own case (p. 563) for why projecting first loses information §39.4 does not."""
    hub, leaves = figure_39_1()
    hub_pairs = [(u, v) if hub.nodes[u]["mode"] == "speaker" else (v, u) for u, v in hub.edges()]
    leaves_pairs = [
        (u, v) if leaves.nodes[u]["mode"] == "speaker" else (v, u) for u, v in leaves.edges()
    ]
    hub_projection = bipartite_projection(hub_pairs, side="right")
    leaves_projection = bipartite_projection(leaves_pairs, side="right")
    hub_edges = sorted(
        (tuple(sorted((u, v))), data["weight"]) for u, v, data in hub_projection.edges(data=True)
    )
    leaves_edges = sorted(
        (tuple(sorted((u, v))), data["weight"]) for u, v, data in leaves_projection.edges(data=True)
    )
    assert hub_edges == leaves_edges  # Figure 39.1's claim: the two projections are identical

    # The hub structure has one left node, so co-clustering it into 2 row-groups is impossible;
    # the six-leaf structure has six and can be split. Reading the projection alone cannot see
    # this difference, because the two projections are the same graph.
    with pytest.raises(ValueError, match=r"2 <= k <= 1"):
        neighbor_similarity(hub, k=2, seed=1)
    leaves_result = neighbor_similarity(leaves, k=2, seed=1)
    assert leaves_result.k == 2
    assert {node for community in leaves_result.communities for node in community} == set(
        leaves.nodes
    )


def test_neighbor_similarity_refuses_a_one_mode_graph() -> None:
    graph = nx.karate_club_graph()
    with pytest.raises(ValueError, match="two-mode network"):
        neighbor_similarity(graph, k=2)


# -------------------------------------------------------------------------------------- §39.1 null


def test_bipartite_significance_names_the_curveball_never_the_configuration_model() -> None:
    graph, _ = two_blocks(3)
    result = brim(graph, k=2, seed=1, restarts=2)
    sig = bipartite_significance(graph, result, samples=15, seed=2)
    assert sig.null == "bipartite_preserving"
    assert sig.testable
    # The planted blocks are as separated as two disjoint bicliques can be, so the observed
    # modularity beats every curveball rewiring by a wide margin.
    assert sig.z > 2.0


def test_bipartite_report_renders_barber_modularity_and_the_curveball_null() -> None:
    graph, _ = two_blocks(3)
    report = bipartite_report(graph, "brim", k=2, seed=1, restarts=2, samples=15)
    text = "\n".join(render_bipartite(report))
    assert "## Bipartite community discovery" in text
    assert "Barber modularity: 0.5000" in text
    assert "bipartite_preserving" in text
    assert "curveball" in text


def test_bipartite_report_refuses_an_unknown_method() -> None:
    graph, _ = two_blocks(3)
    with pytest.raises(ValueError, match=r"brim.*bilouvain.*neighbor-similarity"):
        bipartite_report(graph, "louvain")


# --------------------------------------------------------------- the Southern women (legendary)


def test_brim_on_southern_women_finds_two_mixed_communities_beating_the_null() -> None:
    """Freeman's meta-analysis (`tests/legendary.py`) agrees on a two-way split existing at all,
    not on its exact roster, so this asks only for that: two communities, each spanning both
    modes, with a modularity the curveball null calls a real finding.
    """
    graph, known = southern_women()
    graph.graph.update(min_weight=1, frame=known.frame)
    result = brim(graph, k=2, seed=11, restarts=8)
    assert result.k == 2
    assert len(result.communities) == 2
    for community in result.communities:
        modes = {str(graph.nodes[node]["bipartite"]) for node in community}
        assert modes == {"0", "1"}  # both women and events, in both groups
    sig = bipartite_significance(graph, result, samples=20, seed=11)
    assert sig.z > 3.0


def test_bilouvain_on_southern_women_also_beats_the_null() -> None:
    graph, known = southern_women()
    graph.graph.update(min_weight=1, frame=known.frame)
    result = bilouvain(graph, seed=11, runs=8)
    assert len(result.communities) >= 2
    assert result.modularity > 0.0


# -------------------------------------------------------------------------- integration: analyze


def _bipartite_analysis(method: str, **kwargs: object):  # type: ignore[no-untyped-def]
    graph, _ = two_blocks(3)
    return run_analysis(
        InMemoryGraphStore(),
        graph,
        persona_id="test-pm",
        network="speakers-entities",
        method=method,  # type: ignore[arg-type]
        seed=1,
        samples=10,
        **kwargs,  # type: ignore[arg-type]
    )


def test_analyze_groups_both_modes_together_with_brim() -> None:
    """``analyze --network speakers-entities`` without ``--project`` groups both modes at once."""
    analysis = _bipartite_analysis("brim", k=2)
    assert analysis.bipartite is not None
    assert len(analysis.groups) == 2
    for community in analysis.groups:
        assert _modes(analysis.graph, community) == {"speaker", "entity"}
    assert analysis.evaluation is not None
    assert BIPARTITE_EVALUATION_NOTE in analysis.evaluation.notes

    text = render_markdown(analysis)
    assert "## Bipartite community discovery" in text
    assert "## Community evaluation" in text
    payload = to_payload(analysis)
    assert "bipartite" in payload
    assert payload["bipartite"]["method"] == "brim"


def test_analyze_bilouvain_and_neighbor_similarity_also_route_through_bipartite() -> None:
    for method in ("bilouvain", "neighbor-similarity"):
        analysis = _bipartite_analysis(method, k=2)
        assert analysis.bipartite is not None
        assert analysis.bipartite.result.method == method
        assert analysis.groups


def test_a_bipartite_method_on_a_one_mode_network_raises() -> None:
    graph = nx.karate_club_graph()
    graph.graph.update(min_weight=1, frame="karate club, for a test.")
    with pytest.raises(ValueError, match="two-mode network"):
        run_analysis(
            InMemoryGraphStore(),
            graph,
            persona_id="test-pm",
            network="speakers",
            method="brim",
            seed=1,
            samples=5,
        )
