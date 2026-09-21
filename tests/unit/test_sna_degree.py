"""Degree, its distribution and the power-law test, held to answers known before the code ran.

Four kinds of fixture appear here, and each is chosen for what its answer is known from.

*Planted distributions.* ``numpy``'s ``zipf`` draws from the discrete power law itself, so the
exponent and the lower bound are known exactly and the estimator can be asked to recover them.
Conditioning such a sample on ``k >= 5`` leaves a discrete power law with the same exponent and
``xmin = 5`` -- that is what the normalisation of p(k) = k^-alpha / zeta(alpha, xmin) means --
so the ``xmin`` sweep has a right answer too.

*Generative network models.* Preferential attachment produces a degree distribution whose
exponent is 3 asymptotically, which is the textbook case a fit must not miss; an Erdos-Renyi
graph produces a Poisson one, which is the case a fit must not call scale free. Both are the
book's own contrast (§9.2 against ch. 16).

*The legendary graphs.* The karate club is here as the negative case §9.4 cares about: 34 nodes
is not enough to fit anything, and a procedure that returns an exponent for it is broken.

*The corpus fixtures.* The ``relations`` network of the layered persona, where the in-degree
sequence has to agree node for node with the directed centrality ATL-06 landed, because two
modules computing the same quantity differently is how a report ends up with two answers.

Bootstrap counts are kept small (20-60 resamples) where the assertion is about a fit rather than
about a p-value; the two p-value assertions state the count next to the number, because a
bootstrap p is only as precise as the resampling bought.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import DEGREE_BOOTSTRAP, app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.degree import (
    CSN_LEVEL,
    DEFAULT_BOOTSTRAP,
    KS_LEVEL,
    MIN_DEGREE_SAMPLE,
    VERDICTS,
    compare_tails,
    degree_distribution,
    degree_payload,
    degree_report,
    degree_sequence,
    fit_power_law,
    render_degree,
)
from graphrag.sna.export import entity_relations
from graphrag.sna.guide import READING_RULES
from graphrag.sna.measures import centrality
from graphrag.sna.sampling import sample
from graphrag.sna.stats import likelihood_ratio
from tests.legendary import karate_club

runner = CliRunner()


@pytest.fixture
def messages() -> nx.DiGraph:
    """Three nodes pointing at one hub, weighted 2 each: the star of §9.1's Figure 9.3."""
    graph = nx.DiGraph()
    for name in ("a", "b", "c"):
        graph.add_edge(name, "hub", weight=2.0)
    return graph


# --------------------------------------------------------------------- §9.1 degree variants


def test_the_degree_variants_are_the_ones_the_chapter_defines(messages: nx.DiGraph) -> None:
    """In-degree counts arrow heads, out-degree arrow tails, strength sums the weights."""
    # node_order is the sorted ids: a, b, c, hub.
    assert degree_sequence(messages, "in") == [0, 0, 0, 3]
    assert degree_sequence(messages, "out") == [1, 1, 1, 0]
    assert degree_sequence(messages, "degree") == [1, 1, 1, 3]  # total, as §9.1 defines it
    assert degree_sequence(messages, "weighted") == [2.0, 2.0, 2.0, 6.0]
    assert degree_sequence(messages, "weighted_in") == [0.0, 0.0, 0.0, 6.0]
    assert degree_sequence(messages, "weighted_out") == [2.0, 2.0, 2.0, 0.0]
    # "the nth positions of the two sequences refer to the same node" (§9.1): in + out = degree.
    halves = zip(degree_sequence(messages, "in"), degree_sequence(messages, "out"), strict=True)
    assert [i + o for i, o in halves] == degree_sequence(messages, "degree")


def test_a_direction_is_needed_before_in_and_out_are_different_questions() -> None:
    with pytest.raises(ValueError, match="defined only on a directed network"):
        degree_sequence(nx.cycle_graph(4), "in")
    with pytest.raises(ValueError, match="kind must be one of"):
        degree_sequence(nx.cycle_graph(4), "popularity")


def test_an_unweighted_strength_reduces_to_the_count() -> None:
    """§9.1: the advantage of node strength is that it is the degree when weights are all one."""
    graph = nx.star_graph(4)  # no weight attribute anywhere
    assert degree_sequence(graph, "weighted") == [float(d) for d in degree_sequence(graph)]


def test_a_two_mode_network_has_one_degree_sequence_per_mode() -> None:
    """§9.1: both sequences sum to |E|, and they are not comparable with each other."""
    graph = nx.Graph()
    for speaker in ("s1", "s2"):
        graph.add_node(speaker, mode="speaker")
    for entity in ("e1", "e2", "e3"):
        graph.add_node(entity, mode="entity")
    graph.add_edges_from([("s1", "e1"), ("s1", "e2"), ("s1", "e3"), ("s2", "e1")])

    speakers = degree_sequence(graph, mode="speaker")
    entities = degree_sequence(graph, mode="entity")
    assert speakers == [3, 1]
    assert entities == [2, 1, 1]
    assert sum(speakers) == sum(entities) == graph.number_of_edges()


def test_the_in_degree_sequence_agrees_with_the_directed_centrality(
    related: InMemoryGraphStore,
) -> None:
    """Two modules, one quantity: §9.1's count and ATL-06's n-1 normalised version of it."""
    graph = entity_relations(related, "test-layers")
    assert graph.is_directed()
    scale = graph.number_of_nodes() - 1
    scores = centrality(graph, "in_degree")
    expected = [round(scores[node] * scale) for node in sorted(graph.nodes)]
    assert degree_sequence(graph, "in") == expected
    assert sum(degree_sequence(graph, "in")) == graph.number_of_edges()
    assert degree_sequence(graph, "in") != degree_sequence(graph, "out")  # not a balanced graph


# ----------------------------------------------------------------- §9.2 degree distributions


def test_the_ccdf_starts_at_one_and_never_rises() -> None:
    """The shape §9.2 says to read: p(K >= k), a function rather than a scattergram."""
    karate, _ = karate_club()
    distribution = degree_distribution(degree_sequence(karate))

    assert distribution.n == 34
    assert distribution.ccdf[0] == pytest.approx(1.0)
    assert all(a >= b for a, b in pairwise(distribution.ccdf))
    assert sum(distribution.pmf) == pytest.approx(1.0)
    assert sum(distribution.counts) == 34
    # The karate club's 34 nodes carry 78 edges, so the mean degree is 2 * 78 / 34.
    values = degree_sequence(karate)
    assert sum(values) == 2 * 78
    # And the CCDF at the smallest observed degree is the whole network by construction.
    assert distribution.values[0] == min(values)


def test_the_ccdf_reads_off_the_book_s_own_example() -> None:
    """§9.3: 'half of the network has a degree equal to or greater than two'."""
    # Four nodes of degree 1 and four of degree 2 or more: p(K >= 2) is exactly 0.5.
    distribution = degree_distribution([1, 1, 1, 1, 2, 3, 4, 9])
    at_two = distribution.ccdf[distribution.values.index(2.0)]
    assert at_two == pytest.approx(0.5)


def test_the_log_bins_grow_and_still_hold_every_node() -> None:
    """§9.2's power binning: first bin one wide, each next 10% wider, nothing lost."""
    distribution = degree_distribution([1, 1, 2, 2, 2, 3, 5, 8, 13, 40])
    widths = [b.high - b.low for b in distribution.bins]
    assert widths[0] == pytest.approx(1.0)
    assert all(b >= a for a, b in pairwise(widths))
    assert sum(b.count for b in distribution.bins) == distribution.n
    # A bin's density divides by its width, which is what makes wide and narrow bins comparable.
    for one in distribution.bins:
        assert one.density == pytest.approx(one.count / (distribution.n * (one.high - one.low)))


def test_a_distribution_of_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one node"):
        degree_distribution([])
    with pytest.raises(ValueError, match="factor must be greater than 1"):
        degree_distribution([1, 2, 3], factor=1.0)


# ------------------------------------------------------------ §9.4 fitting and testing


def test_a_planted_power_law_recovers_its_exponent_and_its_xmin() -> None:
    """Drawn from a zeta with alpha = 2.5, so both parameters are known before the fit runs."""
    draws = np.random.default_rng(1).zipf(2.5, 20_000)
    whole = fit_power_law(draws, bootstrap=40, seed=1)
    assert whole is not None
    assert whole.alpha == pytest.approx(2.5, abs=0.15)
    assert whole.xmin == pytest.approx(1.0, abs=1.0)
    assert whole.n_tail == whole.n == 20_000
    # Data really drawn from a power law clear the bar the verdict uses with room to spare:
    # p = 0.976 at 40 resamples, where the resolution is 1/41, so this is not a coin flip.
    assert whole.p_value > CSN_LEVEL
    assert whole.plausible and not whole.rejected

    # Conditioning on k >= 5 leaves the same power law with xmin = 5: the sweep has to find it.
    conditioned = fit_power_law(draws[draws >= 5], bootstrap=20, seed=1)
    assert conditioned is not None
    assert conditioned.alpha == pytest.approx(2.5, abs=0.15)
    assert conditioned.xmin == pytest.approx(5.0, abs=1.0)


def test_a_network_whose_degrees_are_a_planted_power_law_reads_as_scale_free() -> None:
    """The positive case for the word, on a graph whose degree sequence is known to be one.

    The configuration model wires a given degree sequence without changing it, so this network's
    degrees *are* 2,000 draws from a zeta with alpha = 2.5. It is the one network in this file
    that should come back "scale-free", and it does: alpha 2.48, p = 0.443.
    """
    draws = np.random.default_rng(4).zipf(2.5, 2_000)
    if sum(draws) % 2:  # a degree sequence has to sum to an even number of edge endpoints
        draws[0] += 1
    graph = nx.configuration_model(list(draws), seed=4)
    report = degree_report(graph, bootstrap=40, seed=1)

    assert report.fit is not None
    assert report.fit.alpha == pytest.approx(2.5, abs=0.15)
    assert report.fit.plausible
    assert report.verdict == "scale-free"
    assert f"p > {CSN_LEVEL}" in report.reading or "Clauset" in report.reading
    # A multigraph counts connections, not neighbours, and §9.1 asks for that to be said.
    assert any("k_u >= |N_u|" in note for note in report.notes)


def test_preferential_attachment_is_fitted_near_its_generating_exponent() -> None:
    """The textbook scale-free network: p(k) ~ k^-3 asymptotically (§9.3, ch. 16)."""
    graph = nx.barabasi_albert_graph(3000, 3, seed=1)
    report = degree_report(graph, bootstrap=60, seed=1)

    fit = report.fit
    assert fit is not None
    assert 2.5 <= fit.alpha <= 3.5  # measured 2.696 with xmin = 4 over 1,801 nodes
    assert fit.n_tail > MIN_DEGREE_SAMPLE
    # 60 resamples, seed 1: p = 0.066. That does not reject the fit outright, and it does not
    # reach Clauset, Shalizi and Newman's 0.1 either, which is the bar the verdict uses. The
    # band exists because a BA graph's degrees follow 2m(m+1)/(k(k+1)(k+2)) -- a *shifted* power
    # law -- and a KS test over 1,801 tail nodes is strong enough to see the difference.
    assert KS_LEVEL < fit.p_value <= CSN_LEVEL
    assert not fit.rejected
    assert not fit.plausible
    assert report.verdict == "heavy-tailed, power law not rejected but evidence weak"
    assert report.verdict != "scale-free"
    assert "cumulative advantage" in report.reading  # the lognormal was not excluded

    # What the data do support, in the chapter's own terms: broad, exponential ruled out,
    # lognormal not excluded.
    assert report.summary.heavy_tailed
    exponential = report.comparison("exponential")
    assert exponential is not None
    assert exponential.favours == "power law"
    lognormal = report.comparison("lognormal")
    assert lognormal is not None
    assert lognormal.favours == "neither"


def test_a_random_graph_is_not_scale_free() -> None:
    """Erdos-Renyi degrees are Poisson, and §9.4 rules a power law out once an exponential fits."""
    graph = nx.erdos_renyi_graph(3000, 0.004, seed=1)
    report = degree_report(graph, bootstrap=60, seed=1)

    assert report.verdict != "scale-free"
    assert report.verdict == "exponential/Poisson-like"
    fit = report.fit
    assert fit is not None
    exponential = report.comparison("exponential")
    assert exponential is not None
    # Either the bootstrap rejects the fit or the exponential is significantly better; here it
    # is the second, which is the stronger statement the chapter asks for first.
    assert fit.rejected or (exponential.significant and exponential.ratio < 0)
    assert exponential.favours == "exponential"
    assert fit.at_bound  # the tail is steeper than any exponent in the search range
    assert "definitely not a power law" in report.reading


def test_the_karate_club_is_too_small_to_test() -> None:
    """34 nodes. §9.4's procedure needs more, and a fit returned here would be arithmetic."""
    karate, known = karate_club()
    assert known.nodes == 34
    report = degree_report(karate, bootstrap=20, seed=1)

    assert report.fit is None
    assert report.comparisons == ()
    assert report.verdict == "too few nodes to test"
    assert f"at least {MIN_DEGREE_SAMPLE}" in report.reading
    # It still describes the network, and §3.1's caveat about the mean is printed with it.
    assert report.summary.heavy_tailed
    section = "\n".join(render_degree(report))
    assert "No fit." in section
    assert "Quote the median" in report.summary.caveat


def test_a_tail_of_one_value_is_refused_rather_than_fitted_perfectly() -> None:
    """A regular graph fits a power law exactly at xmin = 59, and that is the bug to avoid.

    Every node of the complete graph has degree 59, so the empirical CDF of any tail is a single
    step and the KS distance to any fitted curve is zero. A procedure that reports the perfect
    fit and its p-value of 1 would call the densest possible network scale free.
    """
    graph = nx.complete_graph(60)
    report = degree_report(graph, bootstrap=10, seed=1)

    assert report.fit is None
    assert report.verdict == "too few nodes in the tail to test"
    assert "distinct degrees" in report.reading
    assert report.verdict != "scale-free"


def test_the_discrete_estimator_refuses_fractional_strengths() -> None:
    """§9.4's estimator is over counts; a strength of 2.5 is not a count of anything."""
    with pytest.raises(ValueError, match="discrete estimator"):
        fit_power_law([1.0, 2.5, 3.0] * 30)

    graph = nx.barabasi_albert_graph(200, 2, seed=1)
    for u, v in graph.edges:
        graph[u][v]["weight"] = 1.5
    report = degree_report(graph, "weighted", bootstrap=10, seed=1)
    assert report.fit is None
    assert report.verdict == "not counts, so the discrete test does not apply"
    assert any("not whole numbers" in note for note in report.notes)
    assert report.continuous_fits  # §3.2's continuous fits are still printed for context


def test_comparing_tails_needs_a_fit_to_compare() -> None:
    assert compare_tails([1, 2, 3]) == ()
    draws = np.random.default_rng(3).zipf(2.5, 2_000)
    fit = fit_power_law(draws, bootstrap=10, seed=1)
    assert fit is not None
    comparisons = compare_tails(draws, fit)
    assert [c.other for c in comparisons] == ["lognormal", "exponential"]
    # Both alternatives are fitted on the same tail, which is what makes the ratio a comparison.
    assert all(set(c.params) for c in comparisons)


def test_the_likelihood_ratio_prefers_the_model_that_fits_and_says_when_it_cannot_tell() -> None:
    """The Vuong form §9.4 needs: normalised by its own spread, two-sided, never a verdict."""
    same = [math.log(0.1)] * 50
    assert likelihood_ratio(same, same) == (0.0, 0.0, 1.0)

    better = [math.log(0.2)] * 50
    ratio, statistic, p_value = likelihood_ratio(better, same)
    assert ratio > 0
    # Every point prefers the first model by the same amount, so the difference has no spread
    # and the normalised statistic is undefined: a preference nobody can size is not a finding.
    assert (statistic, p_value) == (0.0, 1.0)

    rng = np.random.default_rng(5)
    noisy_first = list(rng.normal(0.5, 0.1, 400))
    noisy_second = list(rng.normal(0.0, 0.1, 400))
    ratio, statistic, p_value = likelihood_ratio(noisy_first, noisy_second)
    assert ratio > 0 and statistic > 2 and p_value < 0.01

    with pytest.raises(ValueError, match="same observations"):
        likelihood_ratio([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least one observation"):
        likelihood_ratio([], [])


def test_the_bootstrap_is_seeded_and_refuses_a_count_of_nothing() -> None:
    draws = np.random.default_rng(2).zipf(2.5, 1_000)
    first = fit_power_law(draws, bootstrap=20, seed=4)
    second = fit_power_law(draws, bootstrap=20, seed=4)
    assert first is not None and second is not None
    assert first.p_value == second.p_value
    assert first.bootstrap == 20 and first.seed == 4
    with pytest.raises(ValueError, match="at least 1 resample"):
        fit_power_law(draws, bootstrap=0)


# ------------------------------------------------------------------------ the report


def test_the_report_prints_n_the_frame_the_null_model_and_the_chapter() -> None:
    graph = nx.barabasi_albert_graph(400, 2, seed=2)
    graph.graph["frame"] = "Every entity two passages named together."
    report = degree_report(graph, bootstrap=20, seed=1)
    section = "\n".join(render_degree(report))

    assert "## Degree" in section
    assert "Atlas ch. 9" in section and "Clauset" in section
    assert "Every entity two passages named together." in section
    assert "n = 400 nodes" in section
    assert "**Null model.**" in section and "synthetic data sets" in section
    assert "p(K >= k)" in section
    assert f"**{report.verdict}.**" in section

    payload = degree_payload(report)
    assert payload["nodes"] == 400
    assert payload["kind"] == "degree"
    assert payload["power_law"]["bootstrap"] == 20
    assert payload["power_law"]["reject_level"] == KS_LEVEL
    # The bar the verdict rests on is in the payload, not only in the prose.
    assert payload["power_law"]["plausible_level"] == CSN_LEVEL
    assert payload["power_law"]["verdict_level"] == CSN_LEVEL
    assert payload["power_law"]["plausible"] is (payload["power_law"]["p_value"] > CSN_LEVEL)
    assert len(payload["distribution"]["ccdf"]) == len(payload["distribution"]["values"])
    assert payload["verdict"] == report.verdict
    assert json.loads(json.dumps(payload))["chapter"].startswith("Atlas ch. 9")


def test_a_sampled_network_reports_the_rwrw_correction_and_says_so() -> None:
    """§29.3: a walk lands on a node in proportion to its degree, so the raw curve is tilted."""
    population = nx.barabasi_albert_graph(500, 3, seed=1)
    population.graph["frame"] = "A planted network."
    sampled = sample(population, "random-walk", 200, seed=1)

    report = degree_report(sampled, bootstrap=20, seed=1)
    assert report.sample is not None and report.sample["method"] == "random-walk"
    assert report.reweighted
    assert sum(report.reweighted.values()) == pytest.approx(1.0)
    # The correction is an estimate of the population's distribution, so it puts more mass on
    # the low degrees than the walk's own sample did.
    smallest = min(report.reweighted)
    sampled_pmf = dict(zip(report.distribution.values, report.distribution.pmf, strict=True))
    assert report.reweighted[smallest] > sampled_pmf[float(smallest)]

    section = "\n".join(render_degree(report))
    assert "Corrected for the sampler (§29.3)" in section
    assert "random-walk sample" in section
    assert "corrected p(k)" in section


def test_a_two_mode_network_is_fitted_once_per_mode_and_never_as_a_mixture() -> None:
    """§9.1: the two sequences are not comparable, so their union is not a distribution to fit.

    The planted network is deliberately the shape a speakers-entities projection has: 60
    speakers with a few dozen edges each, 300 entities with a handful, so the two modes have
    means an order of magnitude apart. Fitting the union puts xmin in the gap *between* the
    modes and reports the gap as a tail.
    """
    graph = nx.Graph()
    rng = np.random.default_rng(2)
    for i in range(60):
        graph.add_node(f"s{i}", mode="speaker")
    for j in range(300):
        graph.add_node(f"e{j}", mode="entity")
    for i in range(60):
        for j in rng.choice(300, size=int(rng.integers(5, 60)), replace=False):
            graph.add_edge(f"s{i}", f"e{j}")

    report = degree_report(graph, bootstrap=20, seed=1)
    assert report.bipartite
    assert report.fit is None and report.comparisons == ()
    assert report.verdict == "two modes, measured separately"
    assert "not comparable" in report.reading

    assert [row.mode for row in report.modes] == ["entity", "speaker"]
    entities, speakers = report.modes
    assert entities.nodes == 300 and speakers.nodes == 60
    # Both sequences sum to |E| over different node counts, so the means differ by construction.
    assert entities.mean * entities.nodes == pytest.approx(speakers.mean * speakers.nodes)
    assert speakers.mean > 3 * entities.mean
    # Each mode carries its own verdict, and the whole-network reading names them.
    for row in report.modes:
        assert row.verdict in VERDICTS
        assert f"{row.mode} ({row.nodes:,} nodes" in report.reading

    section = "\n".join(render_degree(report))
    assert "### Mode `speaker` on its own (§9.1)" in section
    assert "### Mode `entity` on its own (§9.1)" in section

    payload = degree_payload(report)
    assert payload["power_law"] is None
    assert [m["mode"] for m in payload["modes"]] == ["entity", "speaker"]
    assert all("verdict" in m and "distribution" in m for m in payload["modes"])


def test_a_flattened_multilayer_network_says_the_layer_question_is_elsewhere() -> None:
    """§9.1's multilayer degree is per layer, and this distribution cannot see a layer."""
    graph = nx.barabasi_albert_graph(60, 2, seed=3)
    graph.graph["layers"] = "stance"
    report = degree_report(graph, bootstrap=5, seed=1)
    assert any("per layer" in note and "--layers" in note for note in report.notes)


def test_an_empty_network_has_no_distribution_to_report() -> None:
    report = degree_report(nx.Graph(), bootstrap=5, seed=1)
    assert report.nodes == 0
    assert report.fit is None
    assert any("empty" in note for note in report.notes)
    assert "## Degree" in "\n".join(render_degree(report))


def test_a_directed_report_says_the_degree_is_the_total(messages: nx.DiGraph) -> None:
    report = degree_report(messages, bootstrap=5, seed=1)
    assert report.directed
    assert any("in + out" in note for note in report.notes)


def test_the_guide_carries_the_chapter_s_cautions() -> None:
    section = next((rules for heading, rules in READING_RULES if heading.startswith("Degree")), ())
    names = [rule.name for rule in section]
    assert "A straight line in log-log space is not a power law" in names
    assert "The mean of a heavy tail is not the typical node" in names
    assert "A sampled degree distribution is the sampler's" in names
    assert "The bin size is a choice, and a choice can show you a pattern" in names
    # The section count is test_sna_guide's to assert; this test owns only the degree section.
    assert len(names) == 7


def test_the_cli_default_bootstrap_matches_the_modules() -> None:
    assert DEGREE_BOOTSTRAP == DEFAULT_BOOTSTRAP


# --------------------------------------------------------------------------- the CLI


def test_sna_degree_prints_the_section_and_writes_it(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    report = tmp_path / "degree.md"
    payload = tmp_path / "degree.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--bootstrap",
            "5",
            "--seed",
            "1",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "## Degree" in result.output
    assert "too few nodes to test" in result.output  # three entities

    written = report.read_text(encoding="utf-8")
    assert "p(K >= k)" in written
    body = json.loads(payload.read_text(encoding="utf-8"))
    assert body["kind"] == "degree"
    assert body["nodes"] == 3
    assert body["power_law"] is None
    assert body["verdict"] == "too few nodes to test"


def test_sna_degree_takes_the_directed_variants_and_rejects_what_it_cannot_serve(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    directed = runner.invoke(
        app,
        [
            "sna",
            "degree",
            "test-layers",
            "--network",
            "relations",
            "--kind",
            "in",
            "--bootstrap",
            "5",
        ],
    )
    assert directed.exit_code == 0, directed.output
    assert "arrow heads" in directed.output

    undirected = runner.invoke(
        app, ["sna", "degree", "test-layers", "--network", "entities", "--kind", "out"]
    )
    assert undirected.exit_code == 2
    assert "defined only on a directed network" in undirected.output

    unknown = runner.invoke(app, ["sna", "degree", "test-layers", "--kind", "popularity"])
    assert unknown.exit_code == 2
    assert "--kind must be one of" in unknown.output

    nothing = runner.invoke(app, ["sna", "degree", "test-layers", "--bootstrap", "0"])
    assert nothing.exit_code == 2
    assert "--bootstrap must be at least 1" in nothing.output


def test_sna_degree_can_measure_a_sample_and_says_it_did(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The sampled path: the frame says which sampler, and §29.3's correction is printed."""
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--sample",
            "random-walk",
            "--sample-size",
            "2",
            "--bootstrap",
            "5",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Sampled with random-walk to 2 of 3 nodes" in result.output
    assert "Corrected for the sampler (§29.3)" in result.output

    missing_size = runner.invoke(app, ["sna", "degree", "test-layers", "--sample", "bfs"])
    assert missing_size.exit_code == 2
    assert "--sample needs --sample-size" in missing_size.output


def test_the_analyze_report_carries_the_degree_section(
    cli_context: AppContext, layered: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The section is driven by the same functions, so a reader of a report sees chapter 9."""
    out = tmp_path / "analysis.md"
    as_json = tmp_path / "analysis.json"
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
            "--seed",
            "1",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    written = out.read_text(encoding="utf-8")
    assert "## Degree" in written
    assert "too few nodes to test" in written
    body = json.loads(as_json.read_text(encoding="utf-8"))
    assert body["degree"]["verdict"] == "too few nodes to test"
    assert body["degree"]["nodes"] == 3
