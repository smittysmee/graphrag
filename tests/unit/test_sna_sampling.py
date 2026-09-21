"""Sampling a network, held to the directions chapter 29 states for each method.

The population here is ``nx.barabasi_albert_graph(500, 3, seed=1)``: 500 nodes, 1,491 edges,
mean degree 5.964, maximum degree 67. It is used rather than a legendary graph because the
chapter's claims are claims about a *broad degree distribution* -- "the degree distribution is
emphatically not distributed normally" (§29.1) -- and preferential attachment is where that
distribution comes from. The karate club and Les Misérables appear below where the assertion is
about a known network rather than about a tail.

Every bias assertion is a **direction**, because a direction is what the book states. The
threshold each one uses is written next to it with the measured value that motivated it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.guide import READING_RULES
from graphrag.sna.sampling import (
    BIASES,
    CRAWL_DEGREE,
    SAMPLE_METHODS,
    bias_report,
    completion,
    render_sample,
    reweighted_degree_distribution,
    sample,
    sample_payload,
    sample_record,
)
from tests.legendary import karate_club, les_miserables

runner = CliRunner()

#: Seeds every "across seeds" assertion runs over. Ten is enough for the mean of a 100-node
#: sample to settle and few enough to keep the file fast.
SEEDS = tuple(range(1, 11))

#: The methods the book says oversample hubs, and whose returned node set is the object that
#: claim is about. ``edge`` is not here: it collects every neighbour of the endpoints it drew,
#: and the neighbours of a hub are overwhelmingly low-degree, so the draw's upward bias and the
#: neighbourhood's downward one mix in the returned set. ``ties`` is the same draw stopped at
#: the endpoints, which is where §29.1's prediction can be tested.
HUB_BIASED = ("ties", "bfs", "snowball", "forest-fire", "random-walk")

#: Every sampler except the two that, by the book's own definition, return the seeds *and* the
#: neighbourhood collected around them: ``edge`` (§29.1's plain edge induction) and ``induced``
#: with ``neighbors=True``.
EXACT_SIZE = tuple(m for m in SAMPLE_METHODS if m != "edge")


@pytest.fixture(scope="module")
def population() -> nx.Graph:
    """A preferential-attachment network: the broad degree distribution ch. 29 argues about."""
    return nx.barabasi_albert_graph(500, 3, seed=1)


def mean_selected_degree(population: nx.Graph, sampled: nx.Graph) -> float:
    """The mean degree **in the population** of the nodes a sampler chose.

    This is the measure the book's degree claims are about: it isolates which nodes were picked
    from the arithmetic of cutting the edges that leave the sample, which lowers every sample's
    internal mean whatever the method.
    """
    return statistics.mean(population.degree(node) for node in sampled)


def mean_degree(graph: nx.Graph) -> float:
    return 2 * graph.number_of_edges() / graph.number_of_nodes()


# --------------------------------------------------------------------- the samplers


@pytest.mark.parametrize("method", EXACT_SIZE)
def test_every_sampler_returns_exactly_the_size_asked_for(
    population: nx.Graph, method: str
) -> None:
    sampled = sample(population, method, 100, seed=1)
    assert sampled.number_of_nodes() == 100
    # The induced subgraph keeps every population edge between two sampled nodes.
    assert sampled.number_of_edges() == population.subgraph(sampled.nodes).number_of_edges()
    assert all(node in population for node in sampled)


@pytest.mark.parametrize("method", SAMPLE_METHODS)
def test_a_seed_makes_a_sample_reproducible_and_a_different_seed_moves_it(
    population: nx.Graph, method: str
) -> None:
    first = sample(population, method, 60, seed=7)
    again = sample(population, method, 60, seed=7)
    other = sample(population, method, 60, seed=8)
    assert set(first) == set(again)
    assert sorted(first.edges) == sorted(again.edges)
    assert set(first) != set(other)


@pytest.mark.parametrize("method", SAMPLE_METHODS)
def test_the_sample_records_what_it_is_a_sample_of(population: nx.Graph, method: str) -> None:
    sampled = sample(population, method, 40, seed=3)
    record = sample_record(sampled)
    assert record is not None
    assert record["method"] == method
    assert record["size"] == 40
    assert record["seed"] == 3
    assert record["population_nodes"] == 500
    assert record["population_edges"] == 1491
    assert record["nodes"] == sampled.number_of_nodes()
    assert record["seeds"] <= 40
    assert record["chapter"] == 29
    assert record["section"] == BIASES[method].section
    assert not record["truncated"]
    # It survives GraphML, which is the only lossless format that carries graph attributes.
    assert isinstance(sampled.graph["sample"], str)
    assert "Sampled with" in sampled.graph["frame"]
    assert f"{sampled.number_of_nodes()} of 500 nodes" in sampled.graph["frame"]


@pytest.mark.parametrize("method", SAMPLE_METHODS)
def test_a_sample_larger_than_the_network_is_the_network_and_says_so(method: str) -> None:
    """The karate club has 34 nodes; asking for 100 cannot produce 100, and the record says it."""
    graph, answers = karate_club()
    sampled = sample(graph, method, 100, seed=1)
    record = sample_record(sampled)
    assert record is not None
    assert sampled.number_of_nodes() <= answers.nodes
    if method in {"edge", "ties"}:
        # Edge sampling can only ever reach a node that has an edge; the club has no isolates.
        assert sampled.number_of_nodes() == answers.nodes
    assert record["truncated"] is True


def test_an_empty_network_and_a_bad_method_are_refused() -> None:
    with pytest.raises(ValueError, match="nothing to take a sample of"):
        sample(nx.Graph(), "bfs", 5)
    with pytest.raises(ValueError, match="method must be one of"):
        sample(nx.path_graph(10), "astrology", 5)
    with pytest.raises(ValueError, match="at least 1 node"):
        sample(nx.path_graph(10), "bfs", 0)


def test_a_sampler_refuses_a_parameter_that_belongs_to_another_sampler() -> None:
    """``p`` is forest fire's burning probability; on a walk it would be silently ignored."""
    with pytest.raises(ValueError, match="random-walk takes restart; got p"):
        sample(nx.path_graph(10), "random-walk", 5, seed=1, p=0.5)
    with pytest.raises(ValueError, match="k >= 1"):
        sample(nx.path_graph(10), "snowball", 5, seed=1, k=0)
    with pytest.raises(ValueError, match="burning probability"):
        sample(nx.path_graph(10), "forest-fire", 5, seed=1, p=0.0)


def test_snowball_takes_at_most_k_connections_from_the_node_it_asked(population: nx.Graph) -> None:
    """§29.2: 'she might have more than k friends, but we only take k'.

    The cap is on what a node reveals, not on the degree of the sample: 'a node might be
    mentioned by more than k neighbors', so a node's degree in a snowball sample can exceed k
    and the test asserts the mechanism instead -- a smaller k yields a sparser sample of the
    same size.
    """
    tight = sample(population, "snowball", 100, seed=1, k=1)
    loose = sample(population, "snowball", 100, seed=1, k=8)
    assert tight.number_of_edges() < loose.number_of_edges()


def test_edge_collects_the_neighbourhood_and_ties_stops_at_the_endpoints() -> None:
    """§29.1's two edge methods, told apart by what they do after the draw.

    Plain edge induction *"select[s] edges at random and you collect all their direct
    neighbors"*, so on a 500-node preferential-attachment network a draw of 100 endpoints drags
    in about four hundred nodes. TIES, the *"more sophisticated"* variant, stops at the
    endpoints and induces, so it returns the hundred that were asked for -- and the endpoints it
    drew are a subset of what plain edge induction would have kept.
    """
    graph = nx.barabasi_albert_graph(500, 3, seed=1)
    neighbourhood = sample(graph, "edge", 100, seed=1)
    endpoints = sample(graph, "ties", 100, seed=1)
    assert endpoints.number_of_nodes() == 100
    assert neighbourhood.number_of_nodes() > 300
    assert set(endpoints).issubset(set(neighbourhood))
    record = sample_record(neighbourhood)
    assert record is not None
    assert record["seeds"] == 100
    assert record["nodes"] == neighbourhood.number_of_nodes()


def test_induced_can_collect_the_neighbourhood_the_section_describes(
    population: nx.Graph,
) -> None:
    """§29.1's node-induced sample adds the seeds' neighbours, so it is bigger than its seeds."""
    plain = sample(population, "induced", 20, seed=1)
    with_neighbours = sample(population, "induced", 20, seed=1, neighbors=True)
    assert plain.number_of_nodes() == 20
    assert with_neighbours.number_of_nodes() > 20
    assert set(plain).issubset(set(with_neighbours))
    record = sample_record(with_neighbours)
    assert record is not None
    assert record["seeds"] == 20
    assert record["nodes"] == with_neighbours.number_of_nodes()


# --------------------------------------------------------------------- §29.4 the biases


@pytest.mark.parametrize("method", HUB_BIASED)
def test_the_hub_seeking_samplers_oversample_hubs_at_every_seed(
    population: nx.Graph, method: str
) -> None:
    """§29.1-29.3: edge sampling, the BFS family and the vanilla walk all land on hubs.

    Threshold: the sampled nodes' mean population degree is at least 1.1x the population's at
    every seed. The smallest ratio measured over these ten seeds is 1.17 (forest fire, which
    burns half the neighbours it is offered); BFS, snowball, edge and the walk all sit above
    1.40.
    """
    expected = mean_degree(population)
    ratios = [
        mean_selected_degree(population, sample(population, method, 100, seed=seed)) / expected
        for seed in SEEDS
    ]
    assert min(ratios) > 1.1, f"{method} ratios: {ratios}"
    assert BIASES[method].degree == "up"


def test_induced_sampling_has_no_degree_bias_in_the_mean(population: nx.Graph) -> None:
    """§29.1: uniform random nodes are fair in the mean -- what they miss is the tail.

    Threshold: the mean over ten seeds sits within 15% of the population's mean degree; the
    measured value is 0.97. A single seed is not held to that (the range over these ten is
    0.81-1.17), because the degree distribution is heavy-tailed and one hub moves a 100-node
    mean a long way -- which is the sense in which the section says a random sample is 'unlikely
    to fairly represent the hubs'.
    """
    expected = mean_degree(population)
    ratios = [
        mean_selected_degree(population, sample(population, "induced", 100, seed=seed)) / expected
        for seed in SEEDS
    ]
    assert abs(statistics.mean(ratios) - 1.0) < 0.15, f"induced ratios: {ratios}"
    # And the tail is what it loses: the population's maximum degree is 67.
    maxima = [
        max(population.degree(n) for n in sample(population, "induced", 100, seed=seed))
        for seed in SEEDS
    ]
    assert statistics.mean(maxima) < max(d for _, d in population.degree())


def test_metropolis_hastings_takes_most_of_the_walks_degree_bias_away(
    population: nx.Graph,
) -> None:
    """§29.3: the k_v/k_u rule is the fix for the vanilla walk's degree bias.

    What it does **not** do, at this sample size, is remove it. The uniform stationary
    distribution the section promises is a statement about where the chain spends its steps; the
    object built here is the set of distinct nodes a finite crawl has discovered, and a hub is
    discovered early whatever the acceptance rule, because every one of its many neighbours can
    propose it. So the assertion is the one the sampler can support: the correction cuts the
    walk's excess mean degree by a third on average, and never makes it worse.

    Measured over these ten seeds: the walk's excess is 0.52 of the population mean, the
    corrected walk's 0.32, a cut of 38%.
    """
    expected = mean_degree(population)
    walk = [
        mean_selected_degree(population, sample(population, "random-walk", 100, seed=s)) / expected
        for s in SEEDS
    ]
    corrected = [
        mean_selected_degree(population, sample(population, "metropolis-hastings", 100, seed=s))
        / expected
        for s in SEEDS
    ]
    assert statistics.mean(corrected) < statistics.mean(walk)
    excess_walk = statistics.mean(walk) - 1.0
    excess_corrected = statistics.mean(corrected) - 1.0
    assert excess_corrected < 0.75 * excess_walk
    assert BIASES["metropolis-hastings"].degree == "none"


def test_a_breadth_first_crawl_inflates_clustering_and_induced_sampling_shatters_the_network(
    population: nx.Graph,
) -> None:
    """§29.2 on BFS ('we would overestimate it') and §29.1 on node induction.

    The population's average clustering is 0.047 and it is one connected component. A BFS sample
    of 100 nodes scores 0.137 and stays connected; an induced sample of 100 scores near zero and
    falls into pieces, the largest holding 55%, 18% and 15% of the sample over the first three
    seeds.
    """
    crawled = sample(population, "bfs", 100, seed=1)
    picked = sample(population, "induced", 100, seed=1)
    assert nx.average_clustering(crawled) > nx.average_clustering(population)
    assert nx.number_connected_components(crawled) == 1
    assert nx.number_connected_components(picked) > 1
    largest = max(len(c) for c in nx.connected_components(picked))
    assert largest / picked.number_of_nodes() < 0.8


def test_forest_fire_estimates_clustering_better_than_the_bfs_it_varies(
    population: nx.Graph,
) -> None:
    """§29.2's claim for forest fire is comparative, and so is the assertion.

    *"The advantage of Forest Fire is usually linked with a proper estimation of the clustering
    coefficient of the network, since with a BFS we would overestimate it -- because we fully
    explore the neighborhood of nodes."* Better than BFS is not the same as equal to the
    population, so the sampler's clustering row is printed and not graded, and what is tested is
    the comparison the book actually makes -- and it makes it as a tendency ("usually linked
    with"), so the assertion is one too. Over ten seeds the forest fire sample misses the
    population's 0.047 by 0.055 on average against BFS's 0.139, and it is the closer of the two
    at nine of the ten; seed 10 is the one where a fire happened to burn through a dense corner.
    """
    expected = nx.average_clustering(population)
    fire = [
        abs(nx.average_clustering(sample(population, "forest-fire", 100, seed=s)) - expected)
        for s in SEEDS
    ]
    crawl = [
        abs(nx.average_clustering(sample(population, "bfs", 100, seed=s)) - expected) for s in SEEDS
    ]
    assert statistics.mean(fire) < 0.5 * statistics.mean(crawl)
    assert sum(f < c for f, c in zip(fire, crawl, strict=True)) >= 8
    assert BIASES["forest-fire"].clustering == ""
    assert BIASES["bfs"].clustering == "up"


@pytest.mark.parametrize("method", ["induced", "ties", "bfs", "snowball", "forest-fire"])
def test_the_bias_report_grades_each_sampler_against_the_prediction(
    population: nx.Graph, method: str
) -> None:
    """The report's verdict for the degree row agrees with the book for these five samplers."""
    sampled = sample(population, method, 100, seed=1)
    report = bias_report(population, sampled)
    degree_row = report.checks[0]
    assert degree_row.predicted == BIASES[method].degree
    assert degree_row.agrees is True, f"{method}: observed {degree_row.observed}"
    assert report.population_nodes == 500
    assert report.sample_nodes == 100
    assert report.share == pytest.approx(0.2)
    # The rows the chapter makes no prediction about are printed and not graded.
    ungraded = {check.measure for check in report.checks if check.agrees is None}
    assert {"mean degree inside the sample", "density"} <= ungraded
    assert all(check.predicted == "" for check in report.checks if check.measure in ungraded)


def test_the_bias_report_finds_nothing_wrong_with_a_sample_of_everything() -> None:
    """Les Misérables sampled to its own size is itself, so nothing can have moved.

    The only fixed point a bias report has: n = N, every measure is the population's own, and
    Spearman between a node's degree in the sample and in the population is exactly 1. The one
    row that comes back as a disagreement is the honest one -- §29.1 predicts that induced
    sampling breaks the network into components, and a sample of everything did not.
    """
    graph, answers = les_miserables()
    sampled = sample(graph, "induced", answers.nodes, seed=1)
    report = bias_report(graph, sampled)
    assert report.sample_nodes == answers.nodes
    assert report.sample_edges == answers.edges
    assert all(check.observed == "none" for check in report.checks)
    assert [check.measure for check in report.disagreements] == ["largest component share"]
    assert report.rank_agreement is not None
    assert report.rank_agreement.coefficient == pytest.approx(1.0)


def test_a_sample_of_another_network_is_refused() -> None:
    graph, _ = karate_club()
    # A path over nodes 100-119, so none of its nodes could be mistaken for a club member.
    sampled = sample(nx.path_graph(range(100, 120)), "bfs", 5, seed=1)
    with pytest.raises(ValueError, match="not a sample of this network"):
        bias_report(graph, sampled)
    with pytest.raises(ValueError, match="carries no record"):
        bias_report(graph, graph.copy())


# --------------------------------------------------------------------- §29.3 re-weighting


def test_the_reweighting_reproduces_the_books_worked_example() -> None:
    """§29.3, p. 422, in the book's own numbers.

    100 sampled nodes: 50 of degree 1, 20 of degree 2, 10 of degree 3, 8 of degree 4, 7 of
    degree 5, 5 of degree 6. The denominator is 50/1 + 20/2 + 10/3 + 8/4 + 7/5 + 5/6 = 67.5666,
    the numerator for i = 2 is 20 x 1/2 = 10, and p_2 is 0.148 -- against the 0.20 the raw
    sample showed.
    """
    degrees = [1] * 50 + [2] * 20 + [3] * 10 + [4] * 8 + [5] * 7 + [6] * 5
    corrected = reweighted_degree_distribution(degrees)
    denominator = 50 / 1 + 20 / 2 + 10 / 3 + 8 / 4 + 7 / 5 + 5 / 6
    assert denominator == pytest.approx(67.5666, abs=1e-4)
    assert corrected[2] == pytest.approx(10 / denominator, abs=1e-9)
    assert corrected[2] == pytest.approx(0.148, abs=0.0005)
    assert sum(corrected.values()) == pytest.approx(1.0)
    # The correction pushes probability down the degree axis, which is the whole point.
    assert corrected[1] > 50 / 100
    assert corrected[6] < 5 / 100
    assert reweighted_degree_distribution([0, 0]) == {}


def test_the_reweighting_recovers_a_planted_distribution_from_a_degree_biased_draw() -> None:
    """A planted population, sampled in proportion to degree, is put back where it came from.

    The population is 900 nodes of degree 1 and 100 of degree 10. A random walk lands on a node
    in proportion to its degree (§11.1), so drawing that way gives 900/1900 of the degree-1 nodes
    and 1000/1900 of the degree-10 ones -- 47% against 53%, where the truth is 90/10. RWRW is
    the arithmetic that undoes exactly that tilt, and here it recovers 0.9 and 0.1 to three
    decimal places.
    """
    drawn = [1] * 900 + [10] * 1000
    corrected = reweighted_degree_distribution(drawn)
    assert corrected[1] == pytest.approx(0.9, abs=0.001)
    assert corrected[10] == pytest.approx(0.1, abs=0.001)


# --------------------------------------------------------------------- §29.5 completion


def test_completion_counts_the_edges_the_crawl_saw_leaving_a_planted_sample() -> None:
    """A planted answer: on a 10-cycle, an induced sample of any 4 nodes has known loose ends.

    Every node of a cycle has degree 2, so a sample of k nodes has 2k edge endpoints, of which
    twice the number of edges inside the sample stay in. The sampler probed each selected node,
    so ``known_missing_edges`` is not an estimate and must equal that difference exactly -- which
    is also the number of edges the population has between the sample and the rest of the cycle.
    """
    cycle = nx.cycle_graph(10)
    sampled = sample(cycle, "induced", 4, seed=2)
    estimate = completion(sampled)
    cut = nx.cut_size(cycle, set(sampled.nodes))
    assert estimate.probed == 4
    assert estimate.unprobed == 0
    assert estimate.known_missing_edges == cut
    assert estimate.known_missing_edges == 8 - 2 * sampled.number_of_edges()
    assert estimate.estimated_missing_edges == float(cut)
    assert estimate.completeness == pytest.approx(
        sampled.number_of_edges() / (sampled.number_of_edges() + cut)
    )


def test_a_sample_of_the_whole_network_is_complete() -> None:
    """Nothing was missed when nothing was left out: the karate club, sampled entire."""
    graph, answers = karate_club()
    sampled = sample(graph, "induced", answers.nodes, seed=1)
    estimate = completion(sampled)
    assert estimate.known_missing_edges == 0
    assert estimate.estimated_missing_edges == 0.0
    assert estimate.completeness == pytest.approx(1.0)
    assert estimate.probes == ()


def test_completion_ranks_unprobed_nodes_and_never_the_ones_already_probed(
    population: nx.Graph,
) -> None:
    """§29.5: probing a node whose neighbours you have already seen 'won't help you'."""
    sampled = sample(population, "bfs", 100, seed=1)
    estimate = completion(sampled)
    probed = {node for node, value in sampled.nodes(data=CRAWL_DEGREE) if value is not None}
    assert 0 < estimate.probed < 100
    assert estimate.unprobed == 100 - estimate.probed
    assert estimate.ratio is not None and estimate.ratio > 1.0
    assert [probe.node for probe in estimate.probes]
    assert not {probe.node for probe in estimate.probes} & {str(node) for node in probed}
    # The ranking is by expected new information, so the scores fall.
    scores = [probe.score for probe in estimate.probes]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < estimate.completeness < 1.0  # type: ignore[operator]


def test_a_snowball_sample_cannot_estimate_what_it_missed(population: nx.Graph) -> None:
    """§29.2 again: a respondent who names k friends has not told you their degree.

    Nothing in a snowball sample carries a true degree, so there is nothing to scale an estimate
    against, and the report says that rather than printing a number with nothing behind it.
    """
    sampled = sample(population, "snowball", 60, seed=1)
    estimate = completion(sampled)
    assert estimate.probed == 0
    assert estimate.ratio is None
    assert estimate.completeness is None
    assert "never learns any node's true degree" in estimate.sentence


def test_neighbor_reservoir_keeps_the_sample_in_one_piece_and_keeps_swapping(
    population: nx.Graph,
) -> None:
    """§29.3's first condition: *"we want our sample to be a single connected component."*

    An induced sample of the same size of the same network falls into forty-odd pieces, and a
    walk's sample into two to four. NRS's swap phase refuses any exchange that would leave more
    components behind than it found, so its sample is one piece at every seed -- and the phase
    does run: sixty to a hundred and ten swaps are accepted, of the 600 attempts a 60-node core
    makes at the default rounds=10, before the acceptance probability |V'|/i falls to one in
    eleven.
    """
    rounds_budget = 10 * 60  # rounds * |V'| attempts, of which only some are accepted
    for seed in (1, 2, 3):
        sampled = sample(population, "neighbor-reservoir", 60, seed=seed)
        record = sample_record(sampled)
        assert record is not None
        assert sampled.number_of_nodes() == 60
        assert nx.number_connected_components(sampled) == 1
        assert 0 < record["swaps"] < rounds_budget
        assert nx.number_connected_components(sample(population, "induced", 60, seed=seed)) > 10
    with pytest.raises(ValueError, match="rounds >= 1"):
        sample(population, "neighbor-reservoir", 10, seed=1, rounds=0)


def test_neighbor_reservoir_will_not_swap_out_the_node_that_holds_the_sample_together() -> None:
    """A planted articulation point, which §29.3's connectivity condition can never remove.

    The network: two five-cliques joined only through one node ``x``, each clique wearing six
    fringe nodes so the reservoir has something in it. Any path from one clique to the other runs
    through ``x``, so a sample holding nodes of both cliques holds ``x`` -- and can never stop
    holding it, because removing it would leave two components where there was one. That is the
    mechanism the book credits for NRS's realistic clustering: *"the v with higher clustering
    have higher probability to be replaced, because by removing them it is more likely that the
    graph will stay connected"*, and ``x``, whose two neighbours are not joined to each other,
    has a local clustering of zero and is the one node that is never replaced.

    Of thirty seeds, seventeen produce a sample spanning both cliques; every one of them holds
    x.
    """
    graph = nx.Graph()
    left = [f"L{i}" for i in range(5)]
    right = [f"R{i}" for i in range(5)]
    for clique in (left, right):
        for index, node in enumerate(clique):
            for other in clique[index + 1 :]:
                graph.add_edge(node, other)
    graph.add_edge("L0", "x")
    graph.add_edge("R0", "x")
    for i in range(6):
        graph.add_edge(left[i % 5], f"lf{i}")
        graph.add_edge(right[i % 5], f"rf{i}")
    assert "x" in set(nx.articulation_points(graph))
    assert nx.clustering(graph, "x") == 0.0

    spanning = 0
    for seed in range(30):
        sampled = sample(graph, "neighbor-reservoir", 8, seed=seed)
        assert nx.number_connected_components(sampled) == 1
        if any(n in sampled for n in left) and any(n in sampled for n in right):
            spanning += 1
            assert "x" in sampled, f"seed {seed} spans both cliques without the bridge"
    assert spanning > 10, f"only {spanning} of 30 samples spanned both cliques"


# --------------------------------------------------------------------- the report


def test_the_report_prints_its_frame_its_n_its_null_model_and_its_chapter(
    population: nx.Graph,
) -> None:
    sampled = sample(population, "random-walk", 100, seed=1)
    bias = bias_report(population, sampled)
    estimate = completion(sampled)
    text = render_sample(bias, estimate)
    assert "**Sampling frame.**" in text
    assert "100 of 500 nodes" in text
    assert "**Null model.** None:" in text
    assert "Atlas ch. 29" in text
    assert "§29.5" in text
    assert "stationary distribution" in text  # the prediction being graded
    # Every row carries its note, which is what makes an ungraded row readable.
    for check in bias.checks:
        assert check.note in text

    payload = sample_payload(bias, estimate)
    assert payload["chapter"] == 29
    assert payload["method"] == "random-walk"
    assert payload["population"] == {"nodes": 500, "edges": 1491}
    assert payload["sample"]["nodes"] == 100
    assert payload["completion"]["completeness"] is not None
    assert payload["checks"][0]["measure"].startswith("mean degree")
    assert json.dumps(payload)  # the payload is JSON, not merely dict-shaped


def test_the_guide_carries_the_sampling_rules() -> None:
    headings = [heading for heading, _ in READING_RULES]
    assert "Sampling (sna sample / analyze --sample)" in headings
    names = [
        rule.name for heading, rules in READING_RULES for rule in rules if "Sampling" in heading
    ]
    # §29.4's own warning: a faster-looking source can hand you a smaller sample.
    assert "Throughput is not speed, and a page is not a neighbourhood" in names
    # The section count is test_sna_guide's to assert; this test only owns the sampling section.
    assert any("Sampling" in heading for heading, _ in READING_RULES)


# --------------------------------------------------------------------- the CLI


@pytest.fixture
def entities_network(cli_context: AppContext, layered: InMemoryGraphStore) -> AppContext:
    """The layered corpus's entity network: Alpha, Beta and Gamma, co-mentioned in a triangle."""
    return cli_context


def test_sna_sample_writes_a_sample_and_prints_the_bias_report(
    entities_network: AppContext, tmp_path: Path
) -> None:
    target = tmp_path / "sample.json"
    as_json = tmp_path / "bias.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "sample",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--method",
            "bfs",
            "--size",
            "2",
            "--seed",
            "1",
            "--out",
            str(target),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 of 3 nodes" in result.output
    assert "Sampling (bfs, §29.2)" in result.output
    assert "Completion estimate" in result.output

    written = json.loads(target.read_text())
    assert len(written["nodes"]) == 2
    record = json.loads(written["graph"]["sample"])
    assert record["method"] == "bfs"
    assert record["population_nodes"] == 3
    assert "Sampled with bfs to 2 of 3 nodes" in written["graph"]["frame"]

    payload = json.loads(as_json.read_text())
    assert payload["chapter"] == 29
    assert payload["population"]["nodes"] == 3
    assert payload["null_model"].startswith("none")


def test_sna_sample_can_write_graphml_which_is_where_the_record_has_to_survive(
    entities_network: AppContext, tmp_path: Path
) -> None:
    """The record is a JSON string precisely so this format can carry it (§53.2)."""
    target = tmp_path / "sample.graphml"
    result = runner.invoke(
        app,
        [
            "sna",
            "sample",
            "test-layers",
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--method",
            "induced",
            "--size",
            "2",
            "--seed",
            "1",
            "--out",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    back = nx.read_graphml(target)
    assert json.loads(back.graph["sample"])["method"] == "induced"


def test_sna_sample_rejects_a_method_and_a_parameter_it_cannot_serve(
    entities_network: AppContext, tmp_path: Path
) -> None:
    out = str(tmp_path / "x.json")
    unknown = runner.invoke(
        app, ["sna", "sample", "test-layers", "--method", "vibes", "--size", "2", "--out", out]
    )
    assert unknown.exit_code == 2
    assert "--method must be one of" in unknown.output

    malformed = runner.invoke(
        app,
        ["sna", "sample", "test-layers", "--size", "2", "--param", "k", "--out", out],
    )
    assert malformed.exit_code == 2
    assert "must look like name=value" in malformed.output

    wrong = runner.invoke(
        app,
        [
            "sna",
            "sample",
            "test-layers",
            "--method",
            "bfs",
            "--size",
            "2",
            "--param",
            "p=0.3",
            "--out",
            out,
        ],
    )
    assert wrong.exit_code == 2
    assert "bfs takes no parameters" in wrong.output


def test_sna_analyze_on_a_sample_says_so_in_the_frame(
    entities_network: AppContext, tmp_path: Path
) -> None:
    out = tmp_path / "sampled.md"
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
            "--sample",
            "random-walk",
            "--sample-size",
            "2",
            "--seed",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    report = out.read_text()
    assert "Sampled with random-walk to 2 of 3 nodes" in report
    assert "stationary distribution" in report
    assert "§29.5" in report


def test_sna_analyze_without_a_sample_reports_the_whole_network(
    entities_network: AppContext, tmp_path: Path
) -> None:
    """The default is unchanged: no sampling unless it was asked for."""
    out = tmp_path / "whole.md"
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
        ],
    )
    assert result.exit_code == 0, result.output
    report = out.read_text()
    assert "Sampled with" not in report
    assert "| nodes | 3 |" in report

    missing_size = runner.invoke(
        app,
        ["sna", "analyze", "test-layers", "--sample", "bfs", "--out", str(tmp_path / "x.md")],
    )
    assert missing_size.exit_code == 2
    assert "--sample needs --sample-size" in missing_size.output
