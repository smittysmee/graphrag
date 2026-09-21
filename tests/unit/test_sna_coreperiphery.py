"""Chapter 32: core-periphery, the rich club, the tension with communities, and nestedness.

Every number here is known before the code runs. Two graphs are planted so that the answer is
arithmetic rather than opinion -- a clique core with a periphery that touches only the core, and
two blocks that touch each other barely -- and the correlation the first of them must produce is
computed by hand in :func:`test_the_planted_core_periphery_correlation_is_the_hand_computed_one`
from the four counts of §32.1's phi coefficient. The rest come from the legendary graphs: the
karate club, whose two hubs are the rivals the club split between and are famously *not*
connected to each other, and the Southern women, the canonical two-mode matrix and a classic
nestedness example.

The NODF extremes are definitional: a perfectly nested triangular matrix is 100 and a
checkerboard is 0, and a third matrix small enough to compute by hand pins the arithmetic in
between.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import render_markdown, run_analysis, to_payload
from graphrag.sna.cluster import louvain
from graphrag.sna.coreperiphery import (
    _discrepancy,
    _nodf,
    _phi,
    community_correlation,
    continuous_coreness,
    core_periphery_report,
    core_periphery_tension,
    discrete_core,
    nestedness,
    render_core_periphery,
    rich_club,
    two_mode_sides,
)
from graphrag.sna.matrices import adjacency
from graphrag.sna.null import configuration
from graphrag.sna.stats import pearson
from tests.legendary import karate_club, southern_women, unweighted

CORE = 8
PERIPHERY = 40
ATTACHED = 6


@pytest.fixture
def planted_core() -> tuple[nx.Graph, list[str]]:
    """A clique of 8 with 40 periphery nodes attached only to the core: Figure 32.1, with gaps.

    The core is complete (28 edges) and every peripheral node picks 6 of the 8 core nodes, so
    the core-periphery block is 75% full and the periphery-periphery block is empty by
    construction -- which is the one thing the discrete model insists on. Nothing about the
    answer is emergent: the core is the eight nodes whose ids start with ``c``.
    """
    rng = random.Random(7)
    graph = nx.Graph()
    core = [f"c{i}" for i in range(CORE)]
    graph.add_edges_from((u, v) for i, u in enumerate(core) for v in core[i + 1 :])
    for index in range(PERIPHERY):
        for node in rng.sample(core, ATTACHED):
            graph.add_edge(f"p{index:02d}", node)
    graph.graph.update(frame="A planted core-periphery network, for a test.", min_weight=1)
    return graph, core


@pytest.fixture
def two_blocks() -> nx.Graph:
    """Two dense blocks of 20 joined by a handful of edges: Figure 32.4b, planted.

    The community ideal is the right one here by construction, so the tension check has to
    choose it; if it chose the core-periphery ideal on this graph the check would be useless.
    """
    graph = nx.planted_partition_graph(2, 20, 0.5, 0.02, seed=5)
    graph.graph.update(frame="A planted two-block network, for a test.", min_weight=1)
    return graph


def _analysis(graph: nx.Graph, method: str = "louvain", **kwargs: object):  # type: ignore[no-untyped-def]
    return run_analysis(
        InMemoryGraphStore(),
        graph,
        persona_id="test-pm",
        network="speakers",
        method=method,  # type: ignore[arg-type]
        seed=3,
        runs=2,
        samples=10,
        **kwargs,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------- §32.1 the discrete model


def test_the_planted_core_is_recovered_exactly(planted_core: tuple[nx.Graph, list[str]]) -> None:
    """The eight clique members, and nobody else, with the correlation §32.1 scores it by."""
    graph, core = planted_core
    fitted = discrete_core(graph, seed=11)
    assert fitted.core == sorted(core)
    assert fitted.size == CORE
    assert fitted.correlation > 0.8
    # Figure 32.1's empty block, which is what made the fit possible.
    assert fitted.periphery_edges == 0
    assert fitted.core_edges == CORE * (CORE - 1) // 2
    assert fitted.boundary_edges == PERIPHERY * ATTACHED


def test_the_planted_core_periphery_correlation_is_the_hand_computed_one(
    planted_core: tuple[nx.Graph, list[str]],
) -> None:
    """phi from the four counts of p. 452, computed here in the test and matched to the fit.

    n = 48, so 1,128 pairs; m = 28 + 40*6 = 268 edges; the ideal marks every pair with a core
    node, which is 1,128 - 40*39/2 = 348 of them, and every edge is inside that mark because the
    periphery block is empty. That gives 0.8357..., and the fit has no freedom to beat it.
    """
    graph, _ = planted_core
    pairs, edges, marked = 1128, 268, 348
    expected = (pairs * edges - marked * edges) / math.sqrt(
        marked * (pairs - marked) * edges * (pairs - edges)
    )
    assert expected == pytest.approx(0.8357, abs=5e-5)
    assert discrete_core(graph, seed=11).correlation == pytest.approx(expected)


def test_the_closed_form_correlation_is_the_pearson_of_the_two_matrices() -> None:
    """The four-count phi is the same number as correlating the matrices entry by entry (p. 452).

    §32.1 says to take "the Pearson correlation coefficient between A and D". The search needs
    that in constant time per candidate, so it is computed from counts; this holds the shortcut
    to the long way round on the karate club, over the ideal pattern of its own fitted core.
    """
    graph = unweighted(karate_club()[0])
    fitted = discrete_core(graph, seed=5)
    matrix, nodes = adjacency(graph, nodes=sorted(graph.nodes, key=str), weight=None)
    inside = np.array([node in set(fitted.core) for node in nodes])
    ideal = (inside[:, None] | inside[None, :]).astype(float)
    mask = ~np.eye(len(nodes), dtype=bool)
    long_way = pearson(matrix[mask].tolist(), ideal[mask].tolist()).coefficient
    assert fitted.correlation == pytest.approx(long_way)


def test_the_correlation_refuses_the_all_core_answer_the_sum_would_choose() -> None:
    """§32.2's degenerate optimum: every node in the core makes the ideal constant.

    "Maximizing the quality function would imply to put all nodes in the same core, sacrificing
    the defining characteristic of a core." The sum cannot see that; the correlation is
    undefined there, which is why p. 452 offers it.
    """
    assert math.isnan(_phi(pairs=1128, edges=268, ideal_pairs=1128, ideal_edges=268))
    assert math.isnan(_phi(pairs=1128, edges=268, ideal_pairs=0, ideal_edges=0))


def test_an_edgeless_network_has_no_core() -> None:
    """Nothing to correlate with, so the fit says so rather than returning an arbitrary set."""
    fitted = discrete_core(nx.empty_graph(10), seed=1)
    assert fitted.core == []
    assert math.isnan(fitted.correlation)
    assert "too few nodes or edges" in fitted.method


# ----------------------------------------------------------------- §32.1 the continuous model


def test_the_coreness_vector_puts_the_planted_core_on_top(
    planted_core: tuple[nx.Graph, list[str]],
) -> None:
    """Every core node scores above every peripheral one, and the k-core agrees here."""
    graph, core = planted_core
    fitted = continuous_coreness(graph, core)
    assert min(fitted.coreness[node] for node in core) > max(
        value for node, value in fitted.coreness.items() if node not in set(core)
    )
    assert max(fitted.coreness.values()) == pytest.approx(1.0)
    assert fitted.correlation > 0.5
    # On a planted graph this clean the a priori answer happens to agree; the report prints the
    # agreement precisely because it usually does not (§32.1, p. 450).
    assert fitted.kcore_agreement == pytest.approx(1.0)
    assert sorted(fitted.shells) == [PERIPHERY // 40 * ATTACHED, CORE - 1]


def test_the_k_core_and_the_fitted_core_disagree_on_the_karate_club() -> None:
    """§32.1 keeps k-core out of the chapter because it answers a different question (p. 450).

    The karate club's deepest shell holds ten nodes; the discrete fit of this chapter picks a
    different set, and the Jaccard between them says how different.
    """
    graph = unweighted(karate_club()[0])
    fitted = discrete_core(graph, seed=5)
    continuous = continuous_coreness(graph, fitted.core)
    assert continuous.shells[max(continuous.shells)] == 10
    assert 0.0 < continuous.kcore_agreement < 1.0


# ------------------------------------------------------------------------ §32.1 the rich club


def test_the_rich_club_ratio_is_one_on_a_graph_drawn_from_its_own_null() -> None:
    """rho(k) ~ 1 when the network *is* a configuration-model graph: the null of itself.

    The sharpest available check on the normalisation, because the answer is 1 by construction
    at every threshold. The two lowest thresholds are 1 exactly -- the club is then the whole
    network, whose node and edge counts the null holds fixed -- and the rest scatter around it
    with the sample size.
    """
    karate = unweighted(karate_club()[0])
    drawn = next(iter(configuration(karate, 1, seed=99, weights=False)))
    club = rich_club(drawn, samples=50, seed=4)
    assert club.samples == 50
    ratios = [row.rho for row in club.rows if row.nodes >= 5]
    assert club.rows[0].rho == pytest.approx(1.0)
    assert all(0.75 < ratio < 1.35 for ratio in ratios)
    assert sum(ratios) / len(ratios) == pytest.approx(1.0, abs=0.15)


def test_the_karate_club_has_no_rich_club() -> None:
    """Its two hubs are the rivals the club split between, and they are not connected.

    Zachary's node 0 (the instructor) and node 33 (the president) have the two largest degrees
    and no edge between them, which is the whole story of the network. So at the thresholds
    where the club is just the hubs, phi falls instead of rising, and the normalised curve is
    below 1: the hubs are *less* connected to each other than their degrees imply.
    """
    graph = unweighted(karate_club()[0])
    assert not graph.has_edge(0, 33)
    club = rich_club(graph, samples=50, seed=4)
    assert not club.sustained
    assert "No rich club" in club.verdict
    top = [row for row in club.rows if row.nodes >= 5][-1]
    assert top.rho < 1.0


def test_a_rich_club_curve_stops_where_the_density_stops_being_defined() -> None:
    """phi(k) needs two nodes above the threshold; below that there is no row, not a zero."""
    graph = unweighted(karate_club()[0])
    club = rich_club(graph, samples=5, seed=4)
    assert club.rows
    assert all(row.nodes >= 2 for row in club.rows)
    assert [row.k for row in club.rows] == sorted({row.k for row in club.rows})


def test_a_forced_hub_clique_beats_the_degree_preserving_null() -> None:
    """rho(k) > 1, sustained, when the hubs are made to over-connect on purpose.

    A plain degree-skewed graph is not enough to produce a rich club: §32.1's own point (p. 450)
    is that a broad degree distribution alone already buys the hubs a high chance of meeting, so
    a naive "some nodes have many edges" construction reproduces the null almost exactly (see
    :func:`test_the_rich_club_ratio_is_one_on_a_graph_drawn_from_its_own_null`). To plant an
    answer the null cannot reach, this takes an Erdos-Renyi graph and forces its 15 highest-degree
    nodes into a clique -- edges the degree-preserving null has no reason to place between
    *those* nodes rather than any other pair with the same degrees. The result is deterministic
    for this seed: rho climbs past 1 well before the club narrows to the forced clique itself,
    and stays there.
    """
    graph = nx.gnp_random_graph(200, 0.04, seed=7)
    degree = dict(graph.degree())
    hubs = sorted(degree, key=lambda node: -degree[node])[:15]
    graph.add_edges_from(
        (hubs[i], hubs[j]) for i in range(len(hubs)) for j in range(i + 1, len(hubs))
    )
    club = rich_club(graph, samples=60, seed=4)
    assert club.samples == 60
    assert club.sustained
    assert "Rich club:" in club.verdict
    top = next(row for row in club.rows if row.nodes == len(hubs))
    assert top.phi == pytest.approx(1.0)
    assert top.rho > 2.0
    tail = [row for row in club.rows if row.nodes <= 30 and row.nodes >= len(hubs)]
    assert all(row.rho > 1.0 for row in tail)


# ------------------------------------------------------------- §32.2 the tension with communities


def test_a_core_periphery_network_is_not_read_as_communities(
    planted_core: tuple[nx.Graph, list[str]],
) -> None:
    """§32.2, on the graph it describes: Louvain still returns groups, and they mean nothing.

    Modularity on this network is low but not zero -- the periphery has to be cut somewhere --
    which is exactly the trap the chapter warns about. The ideal-pattern comparison is not
    fooled: the core-periphery pattern reproduces the adjacency an order of magnitude better.
    """
    graph, _ = planted_core
    partition = louvain(graph, seed=3, runs=5)
    assert partition.modularity < 0.2
    tension = core_periphery_tension(graph, partition.communities, samples=10, seed=5)
    assert tension.verdict == "core-periphery"
    assert tension.core_correlation > 0.8
    assert tension.community_correlation < 0.2
    assert tension.core_size == CORE
    assert "periphery" in tension.reading
    # This network is dense enough that no degree-preserving rewiring fits in the sampler's
    # budget, which the reading has to say rather than print a z-score of zero.
    assert tension.samples == 0
    assert "No degree-preserving rewiring" in tension.reading


def test_a_two_block_network_is_read_as_communities(two_blocks: nx.Graph) -> None:
    """The mirror case, with the null available: the community ideal wins and beats its null."""
    partition = louvain(two_blocks, seed=3, runs=5)
    assert partition.modularity > 0.3
    tension = core_periphery_tension(two_blocks, partition.communities, samples=20, seed=5)
    assert tension.verdict == "communities"
    assert tension.community_correlation > tension.core_correlation
    assert tension.samples == 20
    assert tension.community_excess > 0.1
    assert "blend is always different" in tension.reading


def test_the_community_ideal_is_the_mirror_of_the_core_ideal(two_blocks: nx.Graph) -> None:
    """Figure 32.4b as a pattern: 1 inside a group, 0 across. One group marks every pair, so it
    is constant and has no correlation -- the same refusal the all-core partition gets."""
    blocks = [sorted(n for n in two_blocks if n < 20), sorted(n for n in two_blocks if n >= 20)]
    assert community_correlation(two_blocks, blocks) > 0.4
    assert math.isnan(community_correlation(two_blocks, [sorted(two_blocks.nodes)]))
    assert math.isnan(community_correlation(two_blocks, [[node] for node in two_blocks]))


def test_the_tension_check_stays_quiet_on_two_disconnected_cliques() -> None:
    """The other extreme of Figure 32.4: two complete cliques with no edge between them at all.

    This is the community ideal made exact -- every pair inside a clique is joined, no pair
    across them is, so ``community_correlation`` is 1.0 by construction. Any single core the
    discrete model tries to fit has to leave the *other* clique's edges in the periphery block,
    which Figure 32.1 forbids, so its correlation is far below the community one. The tension
    check has to prefer the communities here, not the core, or it would fire on every network
    with more than one dense component.
    """
    graph = nx.disjoint_union(nx.complete_graph(12), nx.complete_graph(12))
    partition = louvain(graph, seed=3, runs=5)
    assert len(partition.communities) == 2
    tension = core_periphery_tension(graph, partition.communities, samples=20, seed=5)
    assert tension.verdict == "communities"
    assert tension.community_correlation == pytest.approx(1.0)
    assert tension.core_correlation < 0.5
    assert "reads as communities" in tension.reading


# ------------------------------------------------------------------------- §32.4 nestedness


def test_nodf_is_100_on_a_perfect_nest_and_0_on_a_checkerboard() -> None:
    """The two ends of the scale, both definitional (§32.4, Figure 32.10).

    The upper-triangular matrix of p. 459 is the ideal: every smaller row is a subset of every
    larger one. The checkerboard has every row total equal, so no pair has decreasing fill and
    nothing can be nested inside anything.
    """
    perfect = np.triu(np.ones((8, 8)))
    assert _nodf(perfect) == (100.0, 100.0, 100.0)
    assert _discrepancy(perfect) == 0

    checkerboard = (np.indices((8, 8)).sum(axis=0) % 2).astype(float)
    assert _nodf(checkerboard) == (0.0, 0.0, 0.0)


def test_nodf_matches_a_matrix_small_enough_to_do_by_hand() -> None:
    """Three rows, three columns, every paired overlap computed in the docstring.

    Rows [1,1,1], [1,0,1], [0,1,0] have totals 3, 2, 1. Pair (1,2): the smaller row's two ones
    are both in the larger, so 100. Pair (1,3): its one is in the larger, so 100. Pair (2,3):
    they share nothing, so 0. The rows give 200/3 = 66.67. Every column totals 2, so no column
    pair has decreasing fill and the columns give 0. NODF is (200 + 0)/(3 + 3) = 33.33.
    """
    matrix = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    nodf, rows, columns = _nodf(matrix)
    assert rows == pytest.approx(200.0 / 3.0)
    assert columns == pytest.approx(0.0)
    assert nodf == pytest.approx(100.0 / 3.0)


def test_the_southern_women_are_nested_but_not_beyond_their_own_margins() -> None:
    """§32.4's caveat on the canonical example, against 100 curveball samples.

    The matrix is nested -- NODF near 49, far from the 0 a checkerboard scores -- and the
    fixed-fixed null produces the same value. That is the objection the chapter reports: "other
    authors suggest that nestedness could arise simply from the degree distribution ... and that
    there are fewer nested system than we originally thought" (p. 458). A report that printed
    the 49 alone would be claiming a structure these margins already imply.
    """
    graph, known = southern_women()
    result = nestedness(graph, samples=100, seed=13)
    assert result.rows == 18
    assert result.columns == 14
    assert result.nodf == pytest.approx(48.58, abs=0.5)
    assert result.discrepancy == 32
    assert result.samples == 100
    assert result.significance is not None
    assert result.significance.null_mean == pytest.approx(result.nodf, abs=2.0)
    assert abs(result.significance.z) < 2.0
    assert known.nodes == graph.number_of_nodes()


def test_nestedness_refuses_a_one_mode_network() -> None:
    """NODF is defined on an incidence matrix, and the karate club has no second mode (§6.4)."""
    with pytest.raises(ValueError, match="two-mode network"):
        nestedness(unweighted(karate_club()[0]))


def test_the_modes_are_read_off_the_nodes_and_never_inferred() -> None:
    """A 2-colourable one-mode graph is still one mode: §6.4's point, and ``describe``'s."""
    assert two_mode_sides(nx.cycle_graph(6)) is None
    sides = two_mode_sides(southern_women()[0])
    assert sides is not None
    assert (len(sides[2]), len(sides[3])) == (18, 14)


# ----------------------------------------------------------------------------- the report


def test_the_section_is_in_the_markdown_and_the_payload(
    planted_core: tuple[nx.Graph, list[str]],
) -> None:
    """Chapter 32 in the analyze report, with its frame, its n, its null and its section."""
    graph, core = planted_core
    analysis = _analysis(graph)
    assert analysis.coreperiphery is not None
    text = render_markdown(analysis)
    assert "## Core-periphery" in text
    assert "**Implements.** §32.1" in text
    assert "§32.2 (the tension with communities)" in text
    assert "**Sampling frame.** A planted core-periphery network" in text
    assert "**n.** 48 node(s), 268 edge(s)" in text
    assert "### Rich club" in text
    assert "### Core-periphery against communities" in text
    assert "reads as core-periphery" in text
    payload = to_payload(analysis)["core_periphery"]
    assert payload["discrete"]["core"] == sorted(core)
    assert payload["discrete"]["correlation"] > 0.8
    assert payload["tension"]["verdict"] == "core-periphery"
    assert payload["rich_club"]["rows"]
    assert "nestedness" not in payload


def test_the_report_never_prints_a_z_score_it_did_not_measure(
    planted_core: tuple[nx.Graph, list[str]],
) -> None:
    """A dense core has no degree-preserving rewiring, and 0.00 would read as "average"."""
    graph, _ = planted_core
    report = core_periphery_report(graph, louvain(graph, seed=3, runs=2).communities, samples=5)
    text = "\n".join(render_core_periphery(report, graph))
    assert "| core-periphery (core of 8) | 0.8357 | - | - | - | 0 |" in text
    assert "0 rewiring(s) for the rich club" in text


def test_the_nestedness_subsection_appears_on_a_two_mode_network() -> None:
    """§32.4's section, and the note that the one-mode models read differently here."""
    graph, _ = southern_women()
    graph.graph.update(min_weight=1)
    analysis = _analysis(graph)
    assert analysis.coreperiphery is not None
    assert analysis.coreperiphery.nested is not None
    text = render_markdown(analysis)
    assert "## Nestedness" in text
    assert "§32.4 (nestedness)" in text
    assert "the fixed-fixed null, both margins" in text
    assert "This network is two-mode" in text
    payload = to_payload(analysis)["core_periphery"]["nestedness"]
    assert payload["rows"] == 18
    assert payload["nodf"] == pytest.approx(48.58, abs=0.5)


def test_a_clustering_report_gets_the_models_but_no_verdict(two_blocks: nx.Graph) -> None:
    """The models are the network's; §32.2's comparison needs a partition Louvain found."""
    analysis = _analysis(two_blocks, "kmeans", k=2)
    assert analysis.coreperiphery is not None
    assert analysis.coreperiphery.tension is None
    text = render_markdown(analysis)
    assert "## Core-periphery" in text
    assert "### Core-periphery against communities" not in text
    assert "the model comparison did not run" in text
