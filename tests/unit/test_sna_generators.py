"""The synthetic networks of Atlas ch. 16-18, and chapter 17's "observed versus random" section.

Two kinds of known answer are asserted here, and nothing else is.

The first is the book's own arithmetic: a G(n,p) graph has clustering p and mean degree p(n-1)
(§16.2, §16.5), the karate club has clustering 0.5706 and an average path length of 2.408 against
a random graph's 0.139 and 2.315, and ``ln n / ln k̄`` is a formula either the code or the test
gets wrong, never both by accident.

The second is what each model was *built* to produce, which is how the book itself argues: a
small-world graph must come out clustered with a nearly constant degree distribution, a
preferential-attachment graph must come out with hubs a Poisson would never reach, a planted
partition must give its planted communities back to Louvain. A generator that returned a plausible
graph with none of those properties would pass a "did it return a graph" test and fail these.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import pytest

from graphrag.sna.cluster import louvain
from graphrag.sna.generators import (
    GENERATORS,
    PATH_SOURCES,
    against_random,
    against_random_payload,
    barabasi_albert,
    caveman,
    configuration_model,
    erdos_renyi_gnm,
    erdos_renyi_gnp,
    expected_properties,
    lfr_benchmark,
    planted_partition,
    poisson_degrees,
    random_geometric,
    render_against_random,
    watts_strogatz,
)
from graphrag.sna.null import configuration
from graphrag.sna.stats import compare_partitions
from tests.legendary import karate_club

# Every generator under one call, so reproducibility and the registry can be swept over all of
# them. The parameters are small enough to run twice in a unit test and large enough for the
# property each model is famous for to be visible.
CALLS = {
    "erdos_renyi_gnp": lambda seed: erdos_renyi_gnp(120, 0.05, seed=seed),
    "erdos_renyi_gnm": lambda seed: erdos_renyi_gnm(120, 300, seed=seed),
    "caveman": lambda seed: caveman(6, 6),
    "watts_strogatz": lambda seed: watts_strogatz(120, 6, 0.1, seed=seed),
    "barabasi_albert": lambda seed: barabasi_albert(120, 3, seed=seed),
    "configuration_model": lambda seed: configuration_model([3] * 40 + [2] * 40, seed=seed),
    "planted_partition": lambda seed: planted_partition(4, 20, 0.4, 0.02, seed=seed),
    "lfr_benchmark": lambda seed: lfr_benchmark(
        150, mu=0.1, average_degree=8, min_community=25, max_community=60, seed=seed
    ),
    "random_geometric": lambda seed: random_geometric(120, 0.2, seed=seed),
}


def test_every_generator_is_in_the_registry_with_a_sentence() -> None:
    """A model a report can produce but not describe is a number with no caption."""
    assert set(GENERATORS) == set(CALLS)
    for name, sentence in GENERATORS.items():
        assert "§1" in sentence, f"{name} does not cite the section it comes from"
        assert len(sentence) > 80


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_generator_is_seeded_and_reproducible(name: str) -> None:
    """The same seed twice is the same graph, and the graph says what made it."""
    first = CALLS[name](7)
    second = CALLS[name](7)
    assert sorted(first.edges()) == sorted(second.edges())
    assert first.graph["model"] == name
    assert "synthetic" in first.graph["frame"]
    assert "Nothing in it was observed" in first.graph["frame"]
    assert not list(nx.selfloop_edges(first))


@pytest.mark.parametrize("name", ["erdos_renyi_gnp", "watts_strogatz", "barabasi_albert"])
def test_a_different_seed_is_a_different_graph(name: str) -> None:
    """Reproducible is not constant: these are random models and the seed is what fixes them."""
    assert sorted(CALLS[name](7).edges()) != sorted(CALLS[name](8).edges())


# ----------------------------------------------------------------------------- the models


def test_a_random_graph_has_the_clustering_and_the_degree_the_chapter_derives() -> None:
    """§16.5: clustering = p. §16.2: mean degree = p(n-1). Both to within sampling noise."""
    graph = erdos_renyi_gnp(1000, 0.01, seed=11)
    degrees = [d for _, d in graph.degree()]
    assert sum(degrees) / len(degrees) == pytest.approx(0.01 * 999, rel=0.1)
    assert nx.average_clustering(graph) == pytest.approx(0.01, abs=0.005)
    # And the G(n,m) form of the same graph is the same graph in the book's sense (§16.1).
    twin = erdos_renyi_gnm(1000, graph.number_of_edges(), seed=11)
    assert nx.average_clustering(twin) == pytest.approx(0.01, abs=0.005)


def test_a_small_world_is_clustered_and_a_random_graph_of_the_same_size_is_not() -> None:
    """§17.2's point: rewiring a few edges keeps the triangles and collapses the distances."""
    small_world = watts_strogatz(500, 6, 0.05, seed=7)
    p = 2 * small_world.number_of_edges() / (500 * 499)
    clustering = nx.average_clustering(small_world)
    assert clustering > 0.5  # the lattice's 0.6, barely dented by 5% rewiring
    assert clustering > 40 * p  # where a random graph of the same size would sit
    # "Short" on the scale that matters: a 500-node ring lattice of degree 6 averages about
    # n/(2k) = 42 hops, and rewiring one edge in twenty brings that down to the order of ln n.
    path = nx.average_shortest_path_length(small_world)
    assert path < 2 * math.log(500)
    assert path < 0.25 * (500 / 6)


def test_preferential_attachment_produces_hubs_a_poisson_would_not() -> None:
    """§17.3: the hubs of a real network are far above "the highest degrees you'll find in a
    random network", and that is what the model was built to reproduce."""
    graph = barabasi_albert(1000, 3, seed=7)
    degrees = [d for _, d in graph.degree()]
    expected = expected_properties(graph)
    assert max(degrees) > 5 * expected.max_degree  # 118 against a Poisson's 15
    assert max(degrees) > 15 * expected.mean_degree
    # ...and the clustering the chapter says it does *not* produce.
    assert nx.average_clustering(graph) < 0.1


def test_the_configuration_model_keeps_the_degree_sequence_it_can_and_says_what_it_dropped() -> (
    None
):
    """§18.1: collapsing to a simple graph loses a few edges, always downward, and says so."""
    karate, _ = karate_club()
    sequence = [d for _, d in karate.degree()]
    graph = configuration_model(sequence, seed=3)
    assert graph.number_of_nodes() == karate.number_of_nodes()
    assert not list(nx.selfloop_edges(graph))
    assert not graph.is_multigraph()
    realised = sorted(d for _, d in graph.degree())
    assert all(a <= b for a, b in zip(realised, sorted(sequence), strict=True))
    dropped = graph.graph["dropped_edges"]
    assert dropped == sum(sequence) // 2 - graph.number_of_edges()
    assert dropped > 0  # this sequence's tail makes parallel edges, which is §18.1's own caveat


def test_a_planted_partition_hands_its_communities_back_to_louvain() -> None:
    """§18.2: the answer is known before the method runs, which is what a benchmark is for."""
    graph = planted_partition(4, 30, 0.35, 0.02, seed=11)
    planted = [graph.nodes[node]["community"] for node in sorted(graph.nodes)]
    assert sorted(set(planted)) == [0, 1, 2, 3]
    result = louvain(graph, seed=1, runs=3)
    found = {node: index for index, group in enumerate(result.communities) for node in group}
    agreement = compare_partitions(planted, [found[node] for node in sorted(graph.nodes)])
    assert agreement["normalized_mutual_information"] > 0.9
    assert agreement["adjusted_rand_index"] > 0.9


def test_an_lfr_benchmark_respects_its_mixing_parameter_and_its_planted_communities() -> None:
    """§18.2: mu is "the share of edges that a node has to nodes that are not part of its own
    cave", and at a low mu the planted partition is recoverable."""
    graph = lfr_benchmark(
        200, mu=0.05, average_degree=8, min_community=25, max_community=60, seed=7
    )
    planted = {node: graph.nodes[node]["community"] for node in graph}
    assert graph.graph["communities"] == len(set(planted.values())) == 4
    external = sum(1 for u, v in graph.edges() if planted[u] != planted[v])
    assert external / graph.number_of_edges() == pytest.approx(0.05, abs=0.1)
    result = louvain(graph, seed=1, runs=3)
    found = {node: index for index, group in enumerate(result.communities) for node in group}
    nodes = sorted(graph.nodes)
    agreement = compare_partitions([planted[n] for n in nodes], [found[n] for n in nodes])
    assert agreement["normalized_mutual_information"] > 0.9


def test_a_random_geometric_graph_is_clustered_with_long_paths() -> None:
    """§18.3: space forces triangles and forbids shortcuts, which is the opposite trade to §17.2."""
    graph = random_geometric(300, 0.12, seed=7)
    report = against_random(graph, samples=10, seed=2)
    assert report.clustered and not report.short
    assert report.clustering_ratio > 10
    assert report.path_ratio > 2
    assert all("pos" in graph.nodes[node] for node in graph)


def test_the_caveman_graph_is_the_clustering_extreme_with_the_diameter_to_match() -> None:
    """§17.1: caves are cliques, so the clustering is near 1 and the ring makes the paths long."""
    graph = caveman(8, 8)
    report = against_random(graph, samples=10, seed=2)
    assert nx.average_clustering(graph) > 0.85  # a clique of 8 is 1.0, the two emissaries less
    assert report.clustered and not report.short and not report.broad


def test_the_books_own_exercise_gives_its_two_graphs_two_different_words() -> None:
    """§17.5 exercise 1, which is what §17.2 exists to demonstrate.

    "Generate a connected caveman graph with 10 cliques, each with 10 nodes. Generate a small
    world graph with 100 nodes, each connected to 8 of their neighbors. Add shortcuts for each
    edge with probability of .05. The two graphs have approximately the same number of edges.
    Compare their clustering coefficients and their average path lengths." Both are clustered --
    that is the half the two models share -- and the report has to separate them on the other
    half: "differently from cavemen, this time we have short paths". A threshold that called
    both of them long would make the section blind to the one contrast the chapter is built on.
    """
    small_world = against_random(watts_strogatz(100, 8, 0.05, seed=7), samples=10, seed=2)
    caves = against_random(caveman(10, 10), samples=10, seed=2)
    assert small_world.clustered and caves.clustered
    assert small_world.short
    assert not caves.short
    assert small_world.path_ratio < caves.path_ratio


# ----------------------------------------------------------------------------- the closed forms


def test_the_closed_forms_on_the_karate_club_are_the_chapters_arithmetic() -> None:
    """Every field of §16.1-16.5 for a graph whose n and m are published (§53.4)."""
    karate, known = karate_club()
    expected = expected_properties(karate)
    assert (expected.nodes, expected.edges) == (known.nodes, known.edges)
    assert expected.p == pytest.approx(2 * 78 / (34 * 33))  # 0.13904
    assert expected.p == pytest.approx(0.1390, abs=5e-5)
    assert expected.clustering == expected.p  # §16.5
    assert expected.mean_degree == pytest.approx(2 * 78 / 34)  # 4.588
    assert expected.degree_variance == expected.mean_degree  # Poisson, §16.2
    assert expected.path_length == pytest.approx(math.log(34) / math.log(4.588235), rel=1e-6)
    assert expected.path_length == pytest.approx(2.3147, abs=5e-4)
    assert expected.giant_threshold_p == pytest.approx(1 / 34)  # §16.3
    assert expected.connected_threshold_p == pytest.approx(math.log(34) / 34)
    assert expected.above_giant_threshold and expected.expected_connected
    assert expected.giant_component_share > 0.98
    assert expected.max_degree == 9  # a Poisson(4.59) puts fewer than one node of 34 above 9


def test_a_thin_graph_has_no_expected_path_length_and_no_giant_component() -> None:
    """§16.3-16.4: below k̄ = 1 there is no giant component and ln k̄ is not a divisor."""
    scattered = nx.Graph()
    scattered.add_nodes_from(range(12))
    scattered.add_edges_from([(0, 1), (2, 3), (4, 5)])  # k̄ = 0.5, well under the transition
    expected = expected_properties(scattered)
    assert expected.mean_degree < 1.0
    assert expected.path_length is None
    assert expected.giant_component_share == 0.0
    assert not expected.above_giant_threshold


def test_the_poisson_degree_distribution_is_a_distribution_at_the_mean_degree() -> None:
    """§16.2's substitution: the parameters "all depend on a single parameter: k̄"."""
    pmf = poisson_degrees(4.588235, 60)
    assert sum(pmf) == pytest.approx(1.0, abs=1e-9)
    assert sum(k * p for k, p in enumerate(pmf)) == pytest.approx(4.588235, rel=1e-6)
    assert pmf.index(max(pmf)) == 4
    assert poisson_degrees(0.0, 3) == [1.0, 0.0, 0.0, 0.0]
    assert poisson_degrees(2.0, -1) == []


def test_expected_properties_flattens_a_directed_network() -> None:
    """§16.1's p counts unordered pairs, so a mutual pair is one edge here (§6.2)."""
    directed = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c")])
    expected = expected_properties(directed)
    assert (expected.nodes, expected.edges) == (3, 2)
    assert expected.p == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------- against random


def test_the_karate_club_against_random_is_clustered_short_and_broad() -> None:
    """The known answer the whole section rests on: 0.5706 and 2.408 against 0.139 and 2.315."""
    karate, _ = karate_club()
    report = against_random(karate, samples=30, seed=5, filters="min_weight=1")
    assert report.clustering == pytest.approx(0.5706, abs=5e-4)
    assert report.path_length == pytest.approx(2.4082, abs=5e-4)  # exact: 34 nodes, 34 sources
    assert not report.path_estimated
    assert report.path_sources == report.path_nodes == 34
    assert report.expected.clustering == pytest.approx(0.1390, abs=5e-4)
    assert report.clustering_ratio == pytest.approx(0.5706 / 0.13904, rel=1e-3)
    assert report.degrees.maximum == 17
    assert report.dispersion == pytest.approx(14.5952 / 4.588235, rel=1e-3)
    assert report.clustered and report.short and report.broad
    assert report.clustering_null.samples == 30
    assert report.clustering_null.null == "configuration"
    assert report.clustering_null.z > 2  # not explained by the degrees either
    assert report.component_expected is None  # the karate club is connected


def test_against_random_is_reproducible_and_says_what_it_held_fixed() -> None:
    karate, _ = karate_club()
    first = against_random(karate, samples=10, seed=5)
    second = against_random(karate, samples=10, seed=5)
    assert first.clustering_null.null_mean == second.clustering_null.null_mean
    assert first.path_null.null_mean == second.path_null.null_mean
    assert first.verdict == second.verdict


def test_a_random_graph_scores_near_its_own_expectation_on_all_three() -> None:
    """The control: chapter 17's three comparisons all come out at 1x on a G(n,m) graph."""
    report = against_random(erdos_renyi_gnm(400, 1600, seed=3), samples=10, seed=3)
    assert report.clustering_ratio == pytest.approx(1.0, abs=0.5)
    assert report.path_ratio == pytest.approx(1.0, abs=0.2)
    assert report.dispersion == pytest.approx(1.0, abs=0.3)
    assert not report.clustered and report.short and not report.broad
    assert "not clustered" in report.verdict and "not broad" in report.verdict


def test_a_small_world_beats_both_nulls_on_clustering_and_a_hub_graph_does_not() -> None:
    """The two nulls disagree exactly where the book says they should.

    Watts-Strogatz clustering is structural, so it survives keeping the degrees; Barabasi-Albert
    clustering is what a few enormous hubs produce, so rewiring the same degrees reproduces it
    and the z collapses. Printing only the G(n,p) ratio would call both of them clustered.
    """
    small_world = against_random(watts_strogatz(300, 6, 0.05, seed=7), samples=20, seed=1)
    hubs = against_random(barabasi_albert(300, 3, seed=7), samples=20, seed=1)
    assert small_world.clustering_ratio > 10 and small_world.clustering_null.z > 5
    assert small_world.clustered and not small_world.broad
    assert hubs.clustering_ratio > 2  # looks clustered against G(n,p)
    assert hubs.clustering_null.z < 2  # and is not, once the degrees are held fixed
    assert not hubs.clustered
    assert hubs.broad and hubs.short


def test_a_caller_can_hand_in_the_null_family_it_already_drew() -> None:
    """The report draws §19.1's rewirings once and spends them on two sections (ch. 17, ch. 36).

    Handing the same family in has to produce the same numbers as drawing it here would, or the
    saving would have changed the answer. The family is consumed lazily, so what is passed is an
    iterator and what comes back counts however many samples arrived.
    """
    karate, _ = karate_club()
    drawn = against_random(karate, samples=8, seed=5)
    shared = against_random(
        karate,
        seed=5,
        samples=999,  # ignored: the family was supplied
        rewirings=configuration(karate, 8, seed=random.Random(5), weights=False),
    )
    assert shared.samples == drawn.samples == 8
    assert shared.clustering_null.null_mean == pytest.approx(drawn.clustering_null.null_mean)
    assert shared.path_null.null_mean == pytest.approx(drawn.path_null.null_mean)
    assert shared.verdict == drawn.verdict


def test_a_fragmented_network_reports_the_path_length_on_its_largest_component() -> None:
    """§16.4 has no answer for a distance between components, so the frame has to shrink."""
    graph = nx.Graph()
    nx.add_cycle(graph, [f"a{i}" for i in range(8)])
    nx.add_cycle(graph, [f"b{i}" for i in range(4)])
    report = against_random(graph, samples=10, seed=1)
    assert report.path_nodes == 8
    assert report.nodes == 12
    assert report.component_expected is not None
    assert report.component_expected.nodes == 8
    assert "largest component" in report.verdict
    assert any("more than one piece" in note for note in report.notes)


def test_a_network_with_no_edges_makes_none_of_the_three_claims() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(12))
    report = against_random(graph, samples=5, seed=1)
    assert not report.clustered and not report.short and not report.broad
    assert report.path_length is None
    assert "None of chapter 17's three comparisons" in report.verdict
    assert "no path length to average" in report.verdict
    assert "no edges" in report.verdict
    assert render_against_random(report)  # and it still renders rather than raising


def test_a_network_too_small_to_rewire_says_there_was_no_null() -> None:
    """§19.1 has no move to make on three nodes, and the report says so rather than printing 0."""
    triangle = nx.Graph([("a", "b"), ("b", "c"), ("c", "a")])
    report = against_random(triangle, samples=20, seed=1)
    assert report.samples == 0
    assert not report.clustering_null.testable
    assert any("No null sample could be built" in note for note in report.notes)
    assert "with no null sample" in report.verdict


def test_a_directed_network_is_flattened_and_the_section_says_so() -> None:
    """The same statement the ``--by`` section makes (§6.2): direction is invisible to ch. 16."""
    directed = nx.DiGraph()
    directed.add_edges_from([("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "b")])
    report = against_random(directed, samples=5, seed=1)
    assert report.nodes == 4
    assert report.edges == 4  # the mutual pair is one undirected edge
    assert "flattened undirected view" in report.flattened
    assert any("flattened undirected view" in note for note in report.notes)


def test_the_path_length_is_estimated_above_the_source_budget_and_says_that_too() -> None:
    """Above :data:`PATH_SOURCES` the number is an estimate, and an estimate has to be labelled."""
    report = against_random(erdos_renyi_gnm(400, 1600, seed=3), samples=3, seed=3)
    assert report.path_estimated
    assert report.path_sources == PATH_SOURCES
    assert report.path_nodes == 400
    exact = nx.average_shortest_path_length(erdos_renyi_gnm(400, 1600, seed=3))
    assert report.path_length == pytest.approx(exact, rel=0.05)
    assert any("estimate" in note for note in report.notes)


def test_the_section_prints_its_frame_its_n_its_nulls_and_its_chapter() -> None:
    karate, _ = karate_club()
    report = against_random(karate, samples=12, seed=5, filters="min_weight=2, source=x")
    text = "\n".join(render_against_random(report))
    assert "## Against random" in text
    assert "**Sampling frame.** The 34 nodes and 78 edges" in text
    assert "Filters: min_weight=2, source=x." in text
    assert "**n.** 12 degree-preserving rewirings" in text
    assert "**Null models.**" in text and "erdos_renyi" in text and "configuration" in text
    assert "**Implements.** §16.2-16.5" in text and "§17.1-17.3" in text
    assert "| average clustering (§17.1) |" in text
    assert "held fixed by construction" in text
    assert "Giant component (§16.3)" in text
    assert "not scale-free" in text


def test_the_payload_carries_both_sides_of_every_comparison() -> None:
    karate, _ = karate_club()
    payload = against_random_payload(against_random(karate, samples=8, seed=5))
    assert payload["observed"]["average_clustering"] == pytest.approx(0.5706, abs=5e-4)
    assert payload["erdos_renyi"]["average_clustering"] == pytest.approx(0.1390, abs=5e-4)
    assert payload["erdos_renyi"]["nodes"] == 34
    assert payload["erdos_renyi_largest_component"] is None
    assert payload["configuration_null"]["average_clustering"]["null"] == "configuration"
    assert payload["configuration_null"]["average_path_length"]["tail"] == "left"
    assert payload["ratios"]["clustering"] > 4
    assert payload["clustered"] and payload["short"] and payload["broad"]
    assert len(payload["erdos_renyi"]["degree_pmf"]) == 18  # 0..17, the observed maximum
