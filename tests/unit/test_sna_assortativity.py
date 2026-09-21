"""Quantitative assortativity, held to answers known before the code ran (Atlas ch. 31).

Four kinds of fixture, each chosen for where its answer comes from.

*The karate club.* Its degree assortativity is a published number, -0.4756, and ``networkx``
computes it by the mixing-matrix formula while this module builds the two endpoint vectors of
p. 443 and takes a Pearson coefficient. The two must agree to floating point, which is the whole
claim of §31.1's first strategy. Its friendship paradox is known the same way: mean degree
4.588, mean neighbour degree 9.610, 29 of 34 members out-done by their friends.

*Graphs whose answer is a theorem.* A star is perfectly disassortative, r = -1, because every
edge joins the one hub to a leaf. A single clique has a constant degree and therefore no
coefficient at all -- a variable that does not vary cannot covary -- and p. 443's "there is only
one way to achieve perfect degree assortativiy... each connected component is a clique" is
checked on two cliques of different sizes, which reach exactly +1.

*A planted numeric attribute.* Two blocks wired densely inside and sparsely across, with values
drawn near 0 in one block and near 100 in the other: the correlation over the edges has to be
large and the permutation null has to be nowhere near it. The same graph with the same numbers
dealt out in a fixed order is the negative case, and it has to come back near zero.

*The corpus fixtures.* ``mentions`` and ``documents`` are numbers the builders write, so the
end-to-end path through ``sna analyze --by`` is checked on the planted corpus rather than on a
hand-built graph, and the borrowed-value refusal is checked on a graph built the way
``export._label_attributes`` builds one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.extract.attributes import AttributeTable, load_attributes
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import run_analysis
from graphrag.sna.assortativity import (
    BUILTIN_NUMERIC,
    _render_correlation,
    _render_curve,
    attribute_by_degree,
    default_null,
    degree_correlations,
    friendship_paradox,
    is_numeric,
    neighbour_average_curve,
    numeric_assortativity,
    numeric_payload,
    numeric_report,
    numeric_values,
    render_degree_correlations,
    render_numeric,
)
from graphrag.sna.attributes import CONFOUNDED_SHARE
from graphrag.sna.export import INHERITED, OWN, attr_key, attr_source_key
from tests.legendary import karate_club, unweighted

runner = CliRunner()

SEED = 11

#: Published for the karate club and reproducible from the graph ``networkx`` ships: the degree
#: assortativity coefficient of the 78-edge copy.
KARATE_R = -0.4756


def named(graph: nx.Graph) -> nx.Graph:
    """The same graph with string node ids, which is what every corpus network has."""
    return nx.relabel_nodes(graph, {node: str(node) for node in graph}, copy=True)


def two_blocks(size: int = 12, *, crossing: int = 2) -> nx.Graph:
    """Two dense blocks joined by a few edges, each node carrying its block's number.

    The answer is known before the code runs: values near 0 sit with values near 0 and values
    near 100 with values near 100, so the correlation over the edges is close to 1 and no
    shuffle of the same numbers can reach it.
    """
    graph = nx.Graph()
    for block, base in (("low", 0.0), ("high", 100.0)):
        nodes = [f"{block}-{i}" for i in range(size)]
        graph.add_nodes_from(nodes)
        for a, b in ((x, y) for x in nodes for y in nodes if x < y):
            graph.add_edge(a, b, weight=1)
        for index, node in enumerate(nodes):
            graph.nodes[node][attr_key("score")] = str(base + index)
            graph.nodes[node][attr_source_key("score")] = OWN
    for i in range(crossing):
        graph.add_edge(f"low-{i}", f"high-{i}", weight=1)
    return graph


def dealt(graph: nx.Graph, values: list[str]) -> nx.Graph:
    """The same graph with the numbers dealt out in a fixed order, so nothing is left to luck."""
    copy = graph.copy()
    for node, value in zip(sorted(copy.nodes), values, strict=True):
        copy.nodes[node][attr_key("score")] = value
    return copy


# ------------------------------------------------------- §31.1 the endpoint correlation


def test_the_endpoint_pearson_is_newmans_degree_assortativity_on_the_karate_club() -> None:
    """p. 443's two vectors and networkx's mixing matrix are the same coefficient."""
    graph = named(unweighted(karate_club()[0]))
    result = numeric_assortativity(graph, "degree", samples=0, seed=SEED)

    assert result.r is not None
    assert result.r == pytest.approx(nx.degree_assortativity_coefficient(graph), abs=1e-12)
    assert result.r == pytest.approx(KARATE_R, abs=5e-5)
    # "each edge contributes two entries to this vector -- unless your network is directed"
    assert result.pairs == 2 * graph.number_of_edges() == 156
    assert result.nodes == 34
    assert result.rho is not None and result.rho < 0


def test_a_star_is_perfectly_disassortative() -> None:
    """Every edge joins the one hub to a leaf, so r is exactly -1 and nothing else."""
    result = numeric_assortativity(named(nx.star_graph(8)), "degree", samples=0)

    assert result.r == pytest.approx(-1.0)
    assert result.pairs == 16


def test_two_cliques_of_different_sizes_are_perfectly_assortative() -> None:
    """p. 443: perfect assortativity is exactly the case where every component is a clique."""
    graph = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(9))
    result = numeric_assortativity(named(graph), "degree", samples=0)

    assert result.r == pytest.approx(1.0)


def test_one_clique_has_no_coefficient_because_the_degree_does_not_vary() -> None:
    result = numeric_assortativity(named(nx.complete_graph(6)), "degree", samples=0)

    assert result.r is None
    assert "does not vary" in result.undefined
    assert "Not measurable" in result.verdict


def test_a_planted_numeric_attribute_correlates_and_beats_its_permutation_null() -> None:
    graph = two_blocks()
    result = numeric_assortativity(graph, "score", permutations=100, seed=SEED)

    assert result.r is not None and result.r > 0.5
    assert result.null is not None and result.null.testable
    assert result.null.z > 5
    assert result.null.p_value < 0.05
    assert result.null_model == "permutation"


def test_the_same_graph_with_the_numbers_dealt_out_scores_near_zero() -> None:
    """The measure has to be able to say no, or it is not a measure."""
    graph = two_blocks()
    order = [str(float(i % 2) * 100 + i) for i in range(graph.number_of_nodes())]
    result = numeric_assortativity(dealt(graph, order), "score", permutations=100, seed=SEED)

    assert result.r is not None and abs(result.r) < 0.2
    assert result.null is not None and abs(result.null.z) < 3


def test_the_degree_is_read_against_a_rewiring_and_an_attribute_against_a_shuffle() -> None:
    """p. 445: for the degree the values *are* the graph, so only a rewiring is a null."""
    assert default_null("degree") == "configuration"
    assert default_null("score") == "permutation"
    graph = named(unweighted(karate_club()[0]))
    result = numeric_assortativity(graph, "degree", samples=30, seed=SEED)

    assert result.null is not None and result.null.null == "configuration"
    # The structural disassortativity the chapter warns about: a rewiring of these degrees is
    # already negative, so the observation is only a finding by its distance from that.
    assert result.null.null_mean < 0
    assert result.null.z < -1.5


def test_an_unknown_null_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="null must be one of"):
        numeric_assortativity(two_blocks(), "score", null="astrology")


# ------------------------------------------------------ §31.1 the neighbour-average curve


def test_the_neighbour_average_curve_aggregates_by_value_and_fits_a_negative_exponent() -> None:
    """Figure 31.3 on the karate club: one point per degree, and a falling curve."""
    graph = named(unweighted(karate_club()[0]))
    curve = neighbour_average_curve(graph, "degree")

    degrees = sorted({d for _, d in graph.degree()})
    assert [point.value for point in curve.points] == [float(d) for d in degrees]
    assert sum(point.nodes for point in curve.points) == 34
    # The node of degree 17 is the officer, whose neighbours are mostly leaves.
    assert curve.points[-1].neighbour_mean < curve.points[0].neighbour_mean
    assert curve.slope is not None and curve.slope < 0
    assert 0.0 <= curve.r_squared <= 1.0
    assert "disassortative" in curve.reading


def test_the_curve_rises_on_a_planted_assortative_network() -> None:
    """Two cliques of different sizes: the bigger a node's degree, the bigger its neighbours'."""
    graph = named(nx.disjoint_union_all([nx.complete_graph(n) for n in (3, 5, 8, 13, 21)]))
    curve = neighbour_average_curve(graph, "degree")

    assert curve.slope is not None and curve.slope > 0
    assert "assortative" in curve.reading


def test_a_curve_with_too_few_points_says_so_instead_of_fitting() -> None:
    curve = neighbour_average_curve(named(nx.complete_graph(5)), "degree")

    assert curve.slope is None
    assert "at least" in curve.undefined
    assert len(curve.points) == 1


# --------------------------------------------------------------- §31.2 friendship paradox


def test_the_friendship_paradox_holds_on_the_karate_club_with_the_published_numbers() -> None:
    """p. 446-447: "for the average node, its degree is lower than the average degree of their
    neighbors". Every number here is recomputable from the graph networkx ships."""
    graph = named(unweighted(karate_club()[0]))
    paradox = friendship_paradox(graph)

    assert paradox.nodes == 34 and paradox.isolated == 0
    assert paradox.mean_value == pytest.approx(2 * 78 / 34)
    assert paradox.mean_value == pytest.approx(4.5882, abs=1e-4)
    assert paradox.mean_neighbour_value == pytest.approx(9.6102, abs=1e-4)
    # <k^2>/<k>, the exact form of the inequality.
    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    assert paradox.edge_weighted_value == pytest.approx(float((degrees**2).mean() / degrees.mean()))
    assert paradox.edge_weighted_value == pytest.approx(7.7692, abs=1e-4)
    assert paradox.outnumbered == 29
    assert paradox.share == pytest.approx(29 / 34)
    assert paradox.holds
    assert "<k^2>/<k>" in paradox.statement


def test_a_regular_network_is_the_one_escape_from_the_paradox() -> None:
    """p. 447: "the only way to escape such paradox is by having a network whose degree
    distributes mostly regularly"."""
    paradox = friendship_paradox(named(nx.cycle_graph(10)))

    assert paradox.mean_value == pytest.approx(2.0)
    assert paradox.mean_neighbour_value == pytest.approx(2.0)
    assert paradox.edge_weighted_value == pytest.approx(2.0)
    assert paradox.excess == pytest.approx(0.0)
    assert paradox.outnumbered == 0
    assert not paradox.holds


@pytest.mark.parametrize(
    "graph",
    [
        nx.barabasi_albert_graph(60, 2, seed=SEED),
        nx.karate_club_graph(),
        nx.les_miserables_graph(),
        nx.florentine_families_graph(),
    ],
    ids=["preferential attachment", "karate", "les miserables", "florentine"],
)
def test_the_inequality_is_a_theorem_not_an_observation(graph: nx.Graph) -> None:
    """Both means beat the mean degree on every network, because both are the same identity."""
    paradox = friendship_paradox(named(graph))

    assert paradox.edge_weighted_value >= paradox.mean_value
    assert paradox.mean_neighbour_value >= paradox.mean_value


# -------------------------------------------------------- §31.3 the attribute by degree


def test_the_degree_bands_partition_the_nodes_and_describe_the_attribute_in_each() -> None:
    graph = two_blocks()
    bins = attribute_by_degree(graph, "score")

    assert sum(b.nodes for b in bins) == graph.number_of_nodes()
    assert [b.summary.count for b in bins] == [b.nodes for b in bins]
    assert all(b.low <= b.high for b in bins)
    # Power-of-two bands, in order, with no overlap.
    assert [(b.low, b.high) for b in bins] == sorted({(b.low, b.high) for b in bins})


def test_the_bands_are_powers_of_two_and_isolated_nodes_get_their_own() -> None:
    graph = nx.star_graph(9)
    graph.add_node("alone")
    bins = attribute_by_degree(named(graph), "degree")

    assert [(b.low, b.high, b.nodes) for b in bins] == [(0, 0, 1), (1, 1, 9), (8, 15, 1)]
    assert sum(b.nodes for b in bins) == 11


def test_no_values_means_no_bands_rather_than_an_error() -> None:
    assert attribute_by_degree(named(nx.path_graph(4)), "nothing") == ()


# ------------------------------------------------------------------ reading the values


def test_the_builtin_counts_and_the_degree_are_numbers_and_a_label_is_not() -> None:
    graph = two_blocks()
    graph.nodes["low-0"]["mentions"] = 4

    assert set(BUILTIN_NUMERIC) == {"degree", "mentions", "documents", "chunks"}
    assert is_numeric(graph, "degree")
    assert is_numeric(graph, "mentions")
    assert is_numeric(graph, "score")
    assert numeric_values(graph, "mentions") == {"low-0": 4.0}
    assert len(numeric_values(graph, "degree")) == graph.number_of_nodes()


def test_a_vocabulary_decides_a_key_that_looks_numeric_but_is_a_code(tmp_path: Path) -> None:
    """A closed list of numerals is a set of labels, and the declaration is what says so."""
    path = tmp_path / "facets.yaml"
    path.write_text(
        "attributes:\n  band:\n    values: ['1', '2']\n  tenure_years:\n    type: number\n",
        encoding="utf-8",
    )
    table = load_attributes(path)
    graph = nx.Graph()
    graph.add_edge("a", "b")
    for node, value in (("a", "1"), ("b", "2")):
        graph.nodes[node][attr_key("band")] = value
        graph.nodes[node][attr_key("tenure_years")] = value

    assert not is_numeric(graph, "band", table)
    assert is_numeric(graph, "tenure_years", table)
    # Undeclared and numeric-looking: the values decide, which is the documented fallback.
    assert is_numeric(graph, "band")


def test_a_key_with_one_non_number_in_it_is_not_numeric() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b")
    graph.nodes["a"][attr_key("score")] = "3.5"
    graph.nodes["b"][attr_key("score")] = "later"

    assert not is_numeric(graph, "score")


def test_a_graph_whose_nodes_are_not_strings_is_measured_the_same_way() -> None:
    """Every legendary graph and half of networkx's generators label their nodes with ints."""
    graph = nx.karate_club_graph()
    result = numeric_assortativity(graph, "degree", samples=0)

    assert result.r == pytest.approx(KARATE_R, abs=5e-5)
    assert friendship_paradox(graph).outnumbered == 29
    assert neighbour_average_curve(graph, "degree").slope is not None
    assert sum(b.nodes for b in attribute_by_degree(graph, "degree")) == 34


# ----------------------------------------------------------------------- the whole section


def test_the_numeric_report_covers_the_chapter_and_renders_every_section() -> None:
    report = numeric_report(two_blocks(), "score", permutations=50, samples=10, seed=SEED)
    text = "\n".join(render_numeric(report))

    assert report.labelled == 24 and report.unlabelled == 0
    assert report.correlation.r is not None and report.correlation.r > 0.5
    assert report.curve.slope is not None
    assert report.paradox.nodes == 24
    assert sum(b.nodes for b in report.bins) == 24
    assert "## By attribute: score (quantitative)" in text
    assert "### Assortativity over the edge endpoints" in text
    assert "### The neighbour-average curve" in text
    assert "### The score paradox" in text
    assert "### The attribute against the degree" in text
    # Frame, n, null and chapter beside the numbers, on every section.
    assert text.count("**Sampling frame.**") >= 4
    assert text.count("**n.**") >= 4
    assert text.count("**Null model.**") >= 4
    assert text.count("**Implements.** §31") >= 4
    payload = numeric_payload(report)
    assert payload["correlation"]["pearson"] == report.correlation.r
    assert json.dumps(payload)


def test_a_borrowed_number_is_refused_the_verdict_as_a_borrowed_label_is() -> None:
    """The confound and its threshold are the categorical module's; the currency is different."""
    graph = two_blocks()
    borrowed = sorted(graph.nodes)[: int(len(graph) * (CONFOUNDED_SHARE + 0.2))]
    for node in borrowed:
        graph.nodes[node][attr_source_key("score")] = INHERITED
        graph.nodes[node]["documents"] = 1
    report = numeric_report(graph, "score", permutations=20, samples=5, seed=SEED)

    assert report.inherited == len(borrowed)
    assert report.inherited_share > CONFOUNDED_SHARE
    assert report.confounded
    assert "Not a reading of this attribute" in report.verdict
    assert any("carry no value of their own" in note for note in report.notes)
    assert any("single document" in note for note in report.notes)


def test_a_builtin_count_carries_the_warning_that_it_counts_the_corpus() -> None:
    graph = two_blocks()
    for index, node in enumerate(sorted(graph.nodes)):
        graph.nodes[node]["documents"] = index + 1
    report = numeric_report(graph, "documents", permutations=20, samples=5, seed=SEED)

    assert report.source == "node data"
    assert not report.sources
    assert any("counted from the same memberships" in note for note in report.notes)


def test_a_key_no_node_carries_says_so_and_names_what_the_network_does_carry() -> None:
    report = numeric_report(two_blocks(), "height", permutations=5, samples=0)

    assert report.labelled == 0
    assert report.correlation.r is None
    assert any("No node in this network carries a number" in note for note in report.notes)
    assert any("degree" in note for note in report.notes)


def test_degree_correlations_are_computed_on_every_network() -> None:
    graph = named(unweighted(karate_club()[0]))
    report = degree_correlations(graph, samples=20, seed=SEED)

    assert report.nodes == 34 and report.edges == 78
    assert report.correlation.r == pytest.approx(KARATE_R, abs=5e-5)
    assert report.paradox.outnumbered == 29
    assert report.curve.slope is not None
    assert "disassortative" in report.verdict


def test_a_section_with_no_coefficient_does_not_blame_the_null() -> None:
    """A 10-cycle has a perfectly good configuration null and nothing to test with it.

    The two absences are different: "the null produced no sample" is a statement about the
    network being too small to rewire, and saying it where the truth is "the degree is constant
    so no null was ever asked for" would send a reader looking for the wrong fix.
    """
    cycle = degree_correlations(named(nx.cycle_graph(10)), samples=20, seed=SEED)
    text = "\n".join(render_degree_correlations(cycle))

    assert cycle.correlation.r is None
    assert "**Null model.** None was run" in text
    assert "too small" not in text
    assert "does not vary" in text

    # And the other absence still says what it is.
    tiny = numeric_assortativity(named(nx.path_graph(3)), "degree", samples=0)
    tiny_text = "\n".join(_render_correlation(tiny, "### x"))
    assert tiny.r is not None
    assert "produced no usable sample" in tiny_text


def test_the_curve_reports_how_many_points_a_fit_could_have_used() -> None:
    """The n line and the sentence under it have to count the same points."""
    curve = neighbour_average_curve(named(nx.complete_graph(5)), "degree")
    text = "\n".join(_render_curve(curve))

    assert curve.slope is None
    assert curve.fitted_points == 1
    assert "the fit ran on the 1 that" in text
    assert "1 point(s) sit at a positive value" in text


def test_an_undeclared_numeric_key_says_that_nothing_declared_it(tmp_path: Path) -> None:
    """The third branch of is_numeric is a guess, so the section prints it as one."""
    path = tmp_path / "facets.yaml"
    path.write_text("attributes:\n  score:\n    type: number\n", encoding="utf-8")
    declared = numeric_report(
        two_blocks(), "score", permutations=10, samples=5, seed=SEED, table=load_attributes(path)
    )
    guessed = numeric_report(two_blocks(), "score", permutations=10, samples=5, seed=SEED)

    assert declared.declared and declared.source == "declared attribute"
    assert not any("nothing declares" in note for note in declared.notes)

    assert not guessed.declared
    assert guessed.source == "undeclared attribute, read as a number"
    note = next(n for n in guessed.notes if "declares `score`" in n)
    assert "every value of it parsed as a number" in note.lower()
    assert "`values:`" in note
    assert guessed.source in "\n".join(render_numeric(guessed))
    assert numeric_payload(guessed)["declared"] is False


def test_by_degree_is_not_printed_twice(layered: InMemoryGraphStore) -> None:
    """`## Degree correlations` is already chapter 31 over the degree, on every report."""
    from graphrag.sna.analysis import render_markdown
    from graphrag.sna.export import build_network

    analysis = run_analysis(
        layered,
        build_network(layered, "entities", "test-layers", min_weight=1),
        persona_id="test-layers",
        network="entities",
        method="louvain",
        by="degree",
        permutations=10,
        samples=5,
        runs=2,
        seed=1,
    )
    text = render_markdown(analysis)

    assert analysis.numeric is None
    assert analysis.degree_correlations is not None
    # As a heading once; the note below quotes the heading, which is not a second section.
    assert text.count("\n## Degree correlations\n") == 1
    assert "## By attribute: degree" not in text
    assert any(
        "asks for the section every report already prints" in note for note in analysis.notes
    )


def test_an_unknown_null_on_a_numeric_key_is_named_rather_than_called_bipartite(
    layered: InMemoryGraphStore,
) -> None:
    """`--null configuration` is not the bipartite null, so the refusal must not say it is."""
    from graphrag.sna.export import build_network

    graph = build_network(layered, "entities", "test-layers", min_weight=1)
    with pytest.raises(ValueError, match="null must be one of"):
        run_analysis(
            layered,
            graph,
            persona_id="test-layers",
            network="entities",
            method="louvain",
            by="documents",
            null="configuration",
            permutations=5,
            samples=2,
            runs=2,
            seed=1,
        )


# ---------------------------------------------------------------------- end to end


def test_analyze_routes_a_numeric_key_to_chapter_31_and_a_label_to_chapter_30(
    layered: InMemoryGraphStore,
) -> None:
    from graphrag.sna.export import build_network

    graph = build_network(layered, "entities", "test-layers", min_weight=1)
    numeric = run_analysis(
        layered,
        graph,
        persona_id="test-layers",
        network="entities",
        method="louvain",
        by="mentions",
        permutations=20,
        samples=5,
        runs=2,
        seed=1,
    )
    categorical = run_analysis(
        layered,
        graph,
        persona_id="test-layers",
        network="entities",
        method="louvain",
        by="region",
        permutations=20,
        samples=5,
        runs=2,
        seed=1,
    )

    assert numeric.numeric is not None and numeric.attribute is None
    assert numeric.numeric.key == "mentions"
    assert categorical.attribute is not None and categorical.numeric is None
    # Chapter 31 over the degree is printed either way.
    assert numeric.degree_correlations is not None
    assert categorical.degree_correlations is not None


def test_the_bipartite_null_is_refused_on_a_numeric_key_rather_than_ignored(
    layered: InMemoryGraphStore,
) -> None:
    from graphrag.sna.export import build_network

    graph = build_network(layered, "entities", "test-layers", min_weight=1)
    with pytest.raises(ValueError, match="categorical --by"):
        run_analysis(
            layered,
            graph,
            persona_id="test-layers",
            network="entities",
            method="louvain",
            by="documents",
            null="bipartite",
            permutations=5,
            samples=2,
            runs=2,
            seed=1,
        )


def test_sna_analyze_by_a_builtin_count_prints_the_numeric_section(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    out = tmp_path / "by-mentions.md"
    as_json = tmp_path / "by-mentions.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--by",
            "mentions",
            "--permutations",
            "20",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## By attribute: mentions (quantitative)" in text
    assert "## Degree correlations" in text
    assert "counted from the same memberships" in text
    payload = json.loads(as_json.read_text())
    assert payload["numeric_attribute"]["key"] == "mentions"
    assert payload["degree_correlations"]["chapter"].startswith("Atlas ch. 31")


def test_the_degree_section_of_a_report_names_its_null_and_its_chapter(
    layered: InMemoryGraphStore,
) -> None:
    from graphrag.sna.analysis import render_markdown
    from graphrag.sna.export import build_network

    analysis = run_analysis(
        layered,
        build_network(layered, "entities", "test-layers", min_weight=1),
        persona_id="test-layers",
        network="entities",
        method="louvain",
        samples=5,
        runs=2,
        seed=1,
    )
    text = render_markdown(analysis)

    assert "## Degree correlations" in text
    assert "### Degree assortativity r" in text
    assert "**Implements.** §31.1" in text
    assert math.isfinite(analysis.summary["degree_assortativity"])


def test_the_summary_still_carries_the_single_number_it_always_did() -> None:
    """`summary()["degree_assortativity"]` is unchanged, and the new section agrees with it."""
    from graphrag.sna.measures import summary

    graph = named(unweighted(karate_club()[0]))
    report = degree_correlations(graph, samples=0)

    assert report.correlation.r == pytest.approx(summary(graph)["degree_assortativity"], abs=1e-9)


def test_an_empty_vocabulary_leaves_the_decision_to_the_values() -> None:
    graph = two_blocks()

    assert is_numeric(graph, "score", AttributeTable())
