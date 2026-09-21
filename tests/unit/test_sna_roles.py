"""Chapter 15's node roles, held to answers that existed before the code did.

Every assertion here comes from one of four places, and none of them is "the function returned
something":

* **The book's own worked example.** §15.2 computes the structural equivalence of the pair in its
  Figure 15.6 twice, by hand: *"three common neighbors out of four possible, thus their structural
  equivalence is 0.75"* and *"the Pearson correlation coefficient of nodes 1 and 2 in Figure 15.6
  is around 0.7"*. That graph is built here edge for edge and both numbers are asserted, along
  with the cosine the section names but does not compute.
* **A planted graph whose roles are known before it is built.** Three blocks, each a star with a
  path through its leaves, and one node joined to all three: the block hubs must come out
  provincial (all their connection at home, well above their module's mean) and the bridge must
  come out a connector (its ties spread evenly over three modules, P = 2/3).
* **A legendary graph with published structure.** The karate club, partitioned by the two factions
  Zachary recorded independently of the edges: nodes 0 and 33 are the instructor and the
  president, and nothing else in the graph is a hub of its faction; node 11 is a leaf with its
  single tie at home, which is what R1 means.
* **A second implementation.** ``simrank`` is checked against ``networkx.simrank_similarity`` on
  the karate club, and separately against the fixed-point equation it is supposed to solve.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from graphrag.sna.roles import (
    ATLAS_ROLE,
    GA_THRESHOLDS,
    ROLE_NAMES,
    Blockmodel,
    RoleThresholds,
    blockmodel,
    build_roles_report,
    classify,
    guimera_amaral,
    image_matrix,
    participation,
    participation_ceiling,
    regular_equivalence,
    regular_similarity,
    render_roles,
    roles_payload,
    same_role_never_cooccur,
    similarity,
    simrank,
    structural_similarity,
    within_module_degree,
)
from tests.legendary import karate_club, unweighted


def figure_15_6() -> nx.Graph:
    """The graph of the Atlas's Figures 15.5(b) and 15.6 (pp. 229-230).

    The book prints the two adjacency rows it compares: node 1 is ``(0, 0, 1, 1, 1, 1)`` and node
    2 is ``(0, 0, 0, 1, 1, 1)``. So node 1 is joined to 3, 4, 5 and 6, node 2 to 4, 5 and 6, and
    the two are **not** joined to each other -- which is the section's whole point about
    structural equivalence.
    """
    return nx.Graph([(1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6)])


def planted_blocks() -> tuple[nx.Graph, list[list[str]]]:
    """Three blocks, each a hub with nine leaves in a path, plus one node joined to all three.

    Known before the code runs: each ``hub{b}`` has nine internal ties where its leaves have two
    or three, so its within-module degree is far above its module's mean; and ``bridge`` has
    exactly one tie into each of the three modules, so ``P = 1 - 3 * (1/3)^2 = 2/3``, which is
    above the 0.62 that separates an ordinary node from a connector and below the 0.80 that makes
    one kinless.
    """
    graph = nx.Graph()
    blocks: list[list[str]] = []
    for block in range(3):
        hub = f"hub{block}"
        members = [f"n{block}_{i}" for i in range(9)]
        blocks.append([hub, *members])
        for member in members:
            graph.add_edge(hub, member)
        for left, right in itertools.pairwise(members):
            graph.add_edge(left, right)
    for block in range(3):
        graph.add_edge("bridge", f"n{block}_0")
    partition = [[*blocks[0], "bridge"], blocks[1], blocks[2]]
    return graph, partition


# ----------------------------------------------------------------- §15.1 the two coordinates


def test_the_karate_factions_make_the_instructor_and_the_president_the_only_hubs() -> None:
    """Zachary's two factions, and the two people the club split between (§15.1).

    The partition is the ground truth Zachary recorded, not a partition a method found, so the
    roles here are about the club and not about Louvain. Node 0 is Mr. Hi and node 33 the
    officer; both sit more than 2.5 standard deviations above their own faction's internal
    degree and neither reaches the 0.30 participation that would make them connectors, so both
    are provincial hubs -- hubs of one circle, which is exactly what a club that split in two
    has. Weights are dropped first: the published numbers for this graph are the unweighted ones
    (``tests.legendary.unweighted``), and z and P are both computed in edge weight.
    """
    graph, known = karate_club()
    plain = unweighted(graph)
    factions = [list(known.groups["Mr. Hi"]), list(known.groups["Officer"])]

    table = guimera_amaral(plain, factions)
    by_node = {row.node: row for row in table.rows}

    assert [row.node for row in table.hubs()] == [0, 33]
    assert by_node[0].within_module_degree == pytest.approx(3.3465, abs=5e-4)
    assert by_node[33].within_module_degree == pytest.approx(3.1363, abs=5e-4)
    assert by_node[0].role == "R5" and by_node[0].role_name == "provincial hub"
    assert by_node[33].role == "R5"
    assert by_node[0].atlas_role == "core"

    # Node 11 is the club's one leaf: a single tie, and it is to Mr. Hi, inside its own faction.
    assert plain.degree(11) == 1
    assert by_node[11].participation == 0.0
    assert by_node[11].role == "R1" and by_node[11].atlas_role == "periphery"

    # Two factions cap P at 0.5, below the 0.62 a connector needs, so there can be none.
    assert table.ceiling == 0.5
    assert table.counts()["R3"] == 0 and table.counts()["R4"] == 0
    assert sum(table.counts().values()) == known.nodes


def test_a_planted_bridge_is_a_connector_and_the_block_hubs_are_provincial() -> None:
    """The planted graph of :func:`planted_blocks`, whose answer is arithmetic (§15.1)."""
    graph, partition = planted_blocks()
    table = guimera_amaral(graph, partition)
    by_node = {row.node: row for row in table.rows}

    bridge = by_node["bridge"]
    assert bridge.participation == pytest.approx(2 / 3)
    assert bridge.within_module_degree < GA_THRESHOLDS.hub  # it has one tie at home, not nine
    assert bridge.role == "R3" and bridge.role_name == "non-hub connector"
    assert bridge.atlas_role == "broker"

    for block in range(3):
        hub = by_node[f"hub{block}"]
        assert hub.participation == 0.0, "every one of the hub's ties is inside its own block"
        assert hub.within_module_degree > GA_THRESHOLDS.hub
        assert hub.role == "R5" and hub.atlas_role == "core"

    # Three modules raise the ceiling above the connector threshold, which two never could.
    assert table.ceiling == pytest.approx(2 / 3)
    assert table.counts()["R5"] == 3 and table.counts()["R3"] == 1


def test_participation_is_the_broker_score_the_rest_of_the_package_already_prints() -> None:
    """§15.1's P and ``measures.brokers`` must be one number, not two that nearly agree."""
    from graphrag.sna.measures import brokers

    graph, partition = planted_blocks()
    mine = participation(graph, partition)
    theirs = dict(brokers(graph, partition, graph.number_of_nodes()))
    assert mine == theirs


def test_a_uniform_module_has_no_spread_so_it_can_hold_no_hub() -> None:
    """z is undefined when every member has the same internal degree; 0.0 says "exactly average".

    A four-clique is the extreme case: each node has three internal ties and the module's
    standard deviation is zero. Reporting 0.0 rather than a NaN keeps the table printable, and
    the consequence -- that no member of such a module can ever be a hub -- is a property of the
    module, which is why the report prints it as a sentence rather than an empty hub list.
    """
    clique = nx.complete_graph(4)
    scores = within_module_degree(clique, [list(clique.nodes)])
    assert set(scores.values()) == {0.0}
    table = guimera_amaral(clique, [list(clique.nodes)])
    assert table.hubs() == []
    assert table.counts()["R1"] == 4  # every tie at home, P = 0


def test_the_participation_ceiling_is_one_minus_one_over_m() -> None:
    assert participation_ceiling(1) == 0.0
    assert participation_ceiling(2) == 0.5
    assert participation_ceiling(4) == 0.75
    # Two modules cannot reach the connector cut point; four can.
    assert participation_ceiling(2) < GA_THRESHOLDS.peripheral < participation_ceiling(4)


def test_every_region_of_the_plane_has_a_name_in_both_vocabularies() -> None:
    """The seven regions, and the four words chapter 15 uses for them (pp. 226-227)."""
    assert list(ROLE_NAMES) == ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    assert set(ATLAS_ROLE) == set(ROLE_NAMES)
    assert set(ATLAS_ROLE.values()) == {"broker", "gatekeeper", "core", "periphery"}
    cut = GA_THRESHOLDS
    assert classify(0.0, 0.0, cut) == "R1"
    assert classify(0.5, 0.0, cut) == "R2"
    assert classify(0.7, 0.0, cut) == "R3"
    assert classify(0.9, 0.0, cut) == "R4"
    assert classify(0.0, 3.0, cut) == "R5"
    assert classify(0.5, 3.0, cut) == "R6"
    assert classify(0.9, 3.0, cut) == "R7"


def test_the_thresholds_can_be_moved_and_the_table_says_which_were_used() -> None:
    """They are Guimera and Amaral's, not the Atlas's, so they are a parameter (see the module)."""
    graph, partition = planted_blocks()
    assert len(guimera_amaral(graph, partition).hubs()) == 3  # the three block hubs, at z ~= 2.95

    strict = RoleThresholds(hub=5.0)
    table = guimera_amaral(graph, partition, thresholds=strict)
    assert table.thresholds is strict
    assert table.hubs() == [], "no node is five standard deviations above its module"

    lenient = guimera_amaral(graph, partition, thresholds=RoleThresholds(hub=-0.3))
    assert len(lenient.hubs()) > 3


# --------------------------------------------------------------------- §15.2 node similarity


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        # "three common neighbors out of four possible, thus their structural equivalence is 0.75"
        ("jaccard", 0.75),
        # 3 / sqrt(4 * 3), the measure the section names without computing it
        ("cosine", 0.8660254),
        # "the Pearson correlation coefficient of nodes 1 and 2 [...] is around 0.7"
        ("pearson", 0.7071068),
    ],
)
def test_the_books_own_figure_15_6_pair_scores_what_the_book_says(
    method: str, expected: float
) -> None:
    scores = structural_similarity(figure_15_6(), method)
    assert scores.between(1, 2) == pytest.approx(expected, abs=1e-6)
    assert scores.between(1, 2) == scores.between(2, 1)
    assert scores.between(1, 1) == 1.0


def test_structural_equivalence_is_about_neighbours_not_about_being_connected() -> None:
    """A star's leaves share every neighbour, share no edge, and are indistinguishable (p. 229).

    This is the chapter's Figure 15.5(a) in its cleanest form. All three measures put the leaves
    at 1.0, the graph has no edge between any two of them, and they therefore turn up in the
    same-role-never-together list -- which is the report section §15.2 earns.
    """
    star = nx.star_graph(5)  # node 0 at the centre, 1..5 as leaves
    for method in ("jaccard", "cosine", "pearson"):
        scores = structural_similarity(star, method)
        assert scores.between(1, 2) == pytest.approx(1.0)
        assert not star.has_edge(1, 2)

    scores = structural_similarity(star, "jaccard")
    table = guimera_amaral(star, [list(star.nodes)])
    pairs = same_role_never_cooccur(star, table, scores, minimum=0.9)
    assert {(pair.u, pair.v) for pair in pairs} == {
        (u, v) for u in range(1, 6) for v in range(u + 1, 6)
    }
    assert all(pair.similarity == pytest.approx(1.0) for pair in pairs)
    assert all(pair.role == "R1" for pair in pairs), "every leaf's single tie is inside the module"
    # The centre is not in the list: it shares an edge with every leaf and no neighbour with any.
    assert all("0" not in (str(pair.u), str(pair.v)) for pair in pairs)


def test_two_isolated_nodes_are_not_called_indistinguishable() -> None:
    """Equal and empty neighbour sets score 0.0, not 1.0 (see ``structural_similarity``).

    The convention matters on a corpus network: an entity nobody wrote about twice has an empty
    row, and the alternative convention would put every such pair at the top of the table.
    """
    graph = nx.Graph()
    graph.add_nodes_from(["alone", "also-alone"])
    graph.add_edge("a", "b")
    for method in ("jaccard", "cosine", "pearson"):
        assert structural_similarity(graph, method).between("alone", "also-alone") == 0.0


def test_pearson_is_the_one_that_can_go_negative() -> None:
    """Two nodes whose neighbourhoods avoid each other correlate below zero (§15.2, p. 230)."""
    graph = nx.Graph([("u", "a"), ("u", "b"), ("v", "c"), ("v", "d")])
    assert structural_similarity(graph, "pearson").between("u", "v") < 0
    assert structural_similarity(graph, "jaccard").between("u", "v") == 0.0


def test_simrank_matches_networkx_on_the_karate_club() -> None:
    """Two implementations of the same recursion, on the graph everything is tried on first.

    ``networkx`` reads edge weights into its transition matrix and §15.2's vectors are of zeros
    and ones, so the comparison is made on the unweighted copy, where the two definitions
    coincide. The tolerance is 1e-5 rather than something tighter for a reason that is
    networkx's and not ours: its loop stops on ``np.allclose``, whose default *relative*
    tolerance is 1e-5 and cannot be passed in, so its fixed point is only ever that good. Our own
    iteration is held to the fixed-point equation itself below, at 1e-12.
    """
    plain = unweighted(karate_club()[0])
    mine = simrank(plain, decay=0.8, iterations=500, tolerance=1e-12)
    theirs = nx.simrank_similarity(
        plain, importance_factor=0.8, max_iterations=1000, tolerance=1e-12
    )
    worst = max(abs(mine.between(u, v) - theirs[u][v]) for u in plain.nodes for v in plain.nodes)
    assert worst < 1e-5
    assert mine.parameters["decay"] == 0.8
    assert mine.parameters["converged"] == 1.0


def test_simrank_solves_the_equation_the_book_writes_down() -> None:
    """``score(u, v) = gamma * sum over the two neighbourhoods / (k_u k_v)``, p. 338.

    Checked directly, node pair by node pair, against the matrix the function returned: the
    recursion is re-applied by hand and has to leave every off-diagonal entry where it was.
    """
    graph = nx.karate_club_graph()
    plain = unweighted(graph)
    scores = simrank(plain, decay=0.7, iterations=500, tolerance=1e-14)
    for u in plain.nodes:
        for v in plain.nodes:
            if u == v:
                assert scores.between(u, u) == 1.0
                continue
            total = sum(
                scores.between(a, b) for a in plain.neighbors(u) for b in plain.neighbors(v)
            )
            expected = 0.7 * total / (plain.degree(u) * plain.degree(v))
            assert scores.between(u, v) == pytest.approx(expected, abs=1e-12)


def test_the_regular_recursion_makes_the_ceo_look_like_the_intern() -> None:
    """The book's own warning about ``sigma = alpha A sigma A + I``, checked (p. 232).

    *"If you run this formula on the network in Figure 15.8 you get that nodes 1, 4, 5 and 6 are
    all similar -- in intuitive terms, you'd get that CEOs are similar to interns because they
    both connect to middle managers [...] node 1 is as similar to node 4 as nodes 2 and 3 are to
    each other."* On a clean two-level hierarchy that is exactly what comes out: the root is
    about as similar to a leaf as the two middle managers are to each other, and it is not
    similar at all to the managers it actually connects to, which sit an odd number of hops away.
    """
    hierarchy = nx.Graph([(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)])
    scores = regular_similarity(hierarchy)
    assert scores.between(1, 4) == pytest.approx(scores.between(2, 3), abs=0.05)
    assert scores.between(1, 4) > 0.15
    assert scores.between(1, 2) == pytest.approx(0.0), "an odd number of hops contributes nothing"
    assert scores.parameters["alpha"] < scores.parameters["convergence_limit"]
    assert scores.parameters["converged"] == 1.0


def test_the_regular_recursion_refuses_an_alpha_it_cannot_converge_for() -> None:
    """§15.2 says ``alpha < 1``; the series needs ``alpha < 1/lambda1^2``, which is smaller."""
    graph = nx.karate_club_graph()
    with pytest.raises(ValueError, match="converges only for alpha"):
        regular_similarity(graph, alpha=0.9)


def test_the_regular_matrix_says_it_was_normalised_where_the_number_is_read() -> None:
    """The cosine normalisation is this package's step and not the book's, so it is disclosed.

    Not only in the docstring: the caption the report prints under the table comes from
    ``Similarity.meaning``, and that is where a reader meets the number.
    """
    scores = regular_similarity(nx.karate_club_graph())
    assert "cosine-normalised" in scores.meaning
    assert "this package adds" in scores.meaning
    assert np.allclose(np.diag(scores.matrix), 1.0), "normalising is what puts 1 on the diagonal"
    assert scores.matrix.min() >= 0.0 and scores.matrix.max() <= 1.0

    graph, partition = planted_blocks()
    report = build_roles_report(
        graph,
        persona_id="test-pm",
        network="entities",
        communities=partition,
        method="regular",
    )
    assert "cosine-normalised" in render_roles(report)


def test_the_similarity_dispatcher_offers_exactly_the_five_and_refuses_the_rest() -> None:
    graph = figure_15_6()
    for method in ("jaccard", "cosine", "pearson", "simrank", "regular"):
        assert similarity(graph, method).method == method
    with pytest.raises(ValueError, match="similarity must be one of"):
        similarity(graph, "euclidean")


def test_a_similarity_matrix_is_refused_before_it_is_too_big_to_hold() -> None:
    graph = nx.path_graph(30)
    with pytest.raises(ValueError, match="max_nodes=10"):
        structural_similarity(graph, "jaccard", max_nodes=10)


# -------------------------------------------------------------------------- §15.2 positions


def test_regular_equivalence_recovers_two_planted_blocks_and_their_image_matrix() -> None:
    """Two disjoint 4-cliques: the image matrix must be the identity, exactly (§15.2, p. 231).

    Nothing is approximate here. The blocks are complete and unjoined, so the density inside each
    position is 1.0 and between them 0.0, and any positioning that finds anything else has not
    found the planted structure.
    """
    graph = nx.Graph()
    for names in (["a", "b", "c", "d"], ["e", "f", "g", "h"]):
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                graph.add_edge(left, right)

    found = regular_equivalence(graph, 2)
    assert isinstance(found, Blockmodel)
    assert sorted(found.positions, key=lambda p: p[0]) == [
        ["a", "b", "c", "d"],
        ["e", "f", "g", "h"],
    ]
    assert np.allclose(found.image, np.eye(2))
    assert found.sizes == [4, 4]


def test_the_image_matrix_reads_a_planted_core_periphery_off_the_blocks() -> None:
    """A core joined to everything and a periphery joined only to the core.

    The image is ``[[1, 1], [1, 0]]`` before the code runs: the core is complete, every core-
    periphery pair is joined, and no two peripheral nodes are. That pattern is the one chapter 32
    will call core-periphery; here it is only a demonstration that the image says what it claims.
    """
    graph = nx.Graph()
    core, periphery = ["c1", "c2", "c3"], ["p1", "p2", "p3"]
    for i, left in enumerate(core):
        for right in core[i + 1 :]:
            graph.add_edge(left, right)
    for left in core:
        for right in periphery:
            graph.add_edge(left, right)

    assert np.allclose(image_matrix(graph, [core, periphery]), [[1.0, 1.0], [1.0, 0.0]])

    found = blockmodel(graph, structural_similarity(graph, "jaccard"), 2)
    assert sorted(found.positions, key=lambda p: p[0]) == [core, periphery]
    assert np.allclose(found.image, [[1.0, 1.0], [1.0, 0.0]])


def test_a_blockmodel_refuses_a_k_it_cannot_produce() -> None:
    graph = nx.path_graph(4)
    scores = structural_similarity(graph, "jaccard")
    with pytest.raises(ValueError, match="k must be between 1 and the 4 nodes"):
        blockmodel(graph, scores, 5)
    assert blockmodel(graph, scores, 1).k == 1


# ------------------------------------------------------------------------------- the report


def test_the_report_prints_the_frame_the_partition_the_null_and_the_chapter() -> None:
    """Every discipline the programme requires of a section, in one run (§15.1, §15.2)."""
    graph, partition = planted_blocks()
    graph.graph["frame"] = "a planted network, built for this test"
    report = build_roles_report(
        graph,
        persona_id="test-pm",
        network="entities",
        communities=partition,
        partition_source="the planted blocks",
        method="jaccard",
        k=3,
    )
    text = render_roles(report)

    assert "chapter 15 (node" in text
    assert "a planted network, built for this test" in text
    assert f"n:** {graph.number_of_nodes():,} nodes" in text
    assert "Partition:** the planted blocks" in text
    assert "Null model:** none" in text
    assert "provincial hub" in text and "non-hub connector" in text
    assert "§15.3" in text, "the section that is deliberately absent has to say that it is"
    assert "automorphic" in text, "and so does the middle rung of §15.2"
    assert "Positions and the image matrix" in text
    # §15.1 warns about its own vocabulary on p. 226, and the report prints the same warning:
    # "core" here is a hub of one module, not a member of chapter 32's core.
    assert "not to be confused with the core-periphery mesoscale structure" in text
    assert "Chapter 32" in text

    payload = roles_payload(report)
    assert payload["implements"].startswith("Atlas ch. 15")
    assert payload["modules"] == 3
    assert payload["counts"]["R5"] == 3
    assert payload["thresholds"]["participation_ceiling"] == pytest.approx(2 / 3)
    assert payload["blockmodel"]["k"] == 3
    assert len(payload["roles"]) == graph.number_of_nodes()
    assert {row["role"] for row in payload["roles"]} <= set(ROLE_NAMES)


def test_the_report_finds_its_own_partition_and_says_which_one() -> None:
    """With no partition handed in, Louvain provides one and the report names it (§15.1)."""
    graph, _ = planted_blocks()
    report = build_roles_report(
        graph, persona_id="test-pm", network="entities", seed=7, runs=3, method="simrank"
    )
    assert report.partition_source.startswith("Louvain at resolution 1")
    assert "stability ARI" in report.partition_source
    text = render_roles(report)
    assert "decay=0.8" in text, "SimRank's decay is a choice, so the report prints it"


def test_a_two_module_run_says_that_no_connector_was_possible() -> None:
    """The ceiling is a property of the partition and the report refuses to let it read as one
    of the network (§15.1)."""
    graph = nx.barbell_graph(5, 0)
    partition = [list(range(5)), list(range(5, 10))]
    report = build_roles_report(
        graph, persona_id="test-pm", network="entities", communities=partition
    )
    assert any("cannot exceed 0.50" in note for note in report.notes)
    assert "That is the partition talking, not the network." in " ".join(report.notes)
