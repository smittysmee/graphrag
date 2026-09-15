"""Filtering a network by a node attribute, and measuring whether it divides along one.

Two kinds of data are used, for two kinds of claim. The planted corpus in ``conftest`` carries
an invented ``region`` attribute on its documents and its speakers, which is what the filters are
checked against. The measures are checked against graphs built here, where the answer is known
before the code runs: two cliques labelled by clique must score high, and the same graph with the
labels shuffled must score near zero.
"""

from __future__ import annotations

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.analysis import run_analysis
from graphrag.sna.attributes import analyse_attribute, attribute_labels, render_attribute
from graphrag.sna.export import attr_count_key, attr_key, build_network, matches

SEED = 11


def two_cliques(size: int = 8) -> nx.Graph:
    """Two cliques joined by one edge, each node labelled with the clique it is in."""
    graph = nx.Graph()
    for side in ("north", "south"):
        nodes = [f"{side}-{i}" for i in range(size)]
        graph.add_nodes_from(nodes)
        for a, b in ((x, y) for x in nodes for y in nodes if x < y):
            graph.add_edge(a, b, weight=1)
    graph.add_edge("north-0", "south-0", weight=1)
    for node in graph.nodes:
        graph.nodes[node][attr_key("region")] = node.split("-")[0]
        graph.nodes[node][attr_count_key("region")] = 1
    return graph


def relabelled(graph: nx.Graph, order: list[str]) -> nx.Graph:
    """The same graph with the labels dealt out in a fixed order, so nothing is left to luck."""
    copy = graph.copy()
    for node, value in zip(sorted(copy.nodes), order, strict=True):
        copy.nodes[node][attr_key("region")] = value
    return copy


# ----------------------------------------------------------------------------- the measures


def test_an_attribute_that_matches_the_communities_scores_high_on_every_measure() -> None:
    report = analyse_attribute(two_cliques(), "region", seed=SEED, permutations=50, samples=20)

    assert report.counts == {"north": 8, "south": 8}
    assert report.labelled == 16 and report.unlabelled == 0
    assert report.assortativity is not None and report.assortativity > 0.9
    assert report.assortativity_z > 3
    assert report.within_share > 0.95
    # The attribute partition is the community structure: it scores what Louvain scores.
    assert report.louvain_modularity is not None
    assert report.modularity == pytest.approx(report.louvain_modularity, abs=0.01)
    assert report.modularity_z > 2
    assert "tracks a real division" in report.verdict


def test_the_same_graph_with_shuffled_labels_scores_near_zero() -> None:
    """The measure has to be able to say no, or it is not a measure."""
    shuffled = relabelled(two_cliques(), ["north", "south"] * 8)

    report = analyse_attribute(shuffled, "region", seed=SEED, permutations=50, samples=20)

    assert report.counts == {"north": 8, "south": 8}  # same labels, dealt differently
    assert report.assortativity is not None and abs(report.assortativity) < 0.2
    assert abs(report.assortativity_z) < 2
    assert report.modularity < 0.05
    assert "does not divide along this attribute" in report.verdict
    assert report.louvain_modularity is not None
    assert report.modularity < report.louvain_modularity


def test_nodes_with_no_value_are_outside_every_measure_and_are_counted() -> None:
    graph = two_cliques()
    graph.add_node("untagged-1")
    graph.add_edge("untagged-1", "north-1", weight=1)

    report = analyse_attribute(graph, "region", seed=SEED, permutations=20, samples=10)

    assert report.total == 17 and report.labelled == 16 and report.unlabelled == 1
    assert sum(report.counts.values()) == 16  # not a third value
    assert any("carry no value" in note for note in report.notes)


def test_an_attribute_no_node_carries_says_so_instead_of_computing_anything() -> None:
    report = analyse_attribute(two_cliques(), "colour", seed=SEED)

    assert report.counts == {} and report.assortativity is None
    assert any("No node in this network carries 'colour'" in note for note in report.notes)
    assert "Not measurable" in report.verdict


def test_a_network_too_small_to_measure_reports_its_counts_and_stops() -> None:
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1)
    for node, value in (("a", "north"), ("b", "south")):
        graph.nodes[node][attr_key("region")] = value

    report = analyse_attribute(graph, "region", seed=SEED)

    assert report.counts == {"north": 1, "south": 1}
    assert report.assortativity is None
    assert any("too little to measure" in note for note in report.notes)


def test_a_mixed_node_keeps_the_majority_value_and_is_counted_as_mixed() -> None:
    graph = two_cliques()
    graph.nodes["north-1"][attr_count_key("region")] = 2

    report = analyse_attribute(graph, "region", seed=SEED, permutations=20, samples=10)

    assert report.mixed == 1
    assert any("disagreed about this attribute" in note for note in report.notes)


def test_the_clusters_are_compared_with_the_attribute_against_a_random_baseline(
    memory_store: InMemoryGraphStore,
) -> None:
    graph = two_cliques()
    groups = [
        sorted(n for n in graph if n.startswith("north")),
        sorted(n for n in graph if n.startswith("south")),
    ]

    report = analyse_attribute(
        graph, "region", groups=groups, seed=SEED, permutations=20, samples=10
    )

    assert report.agreement["adjusted_rand_index"] == pytest.approx(1.0)
    assert report.agreement["baseline_adjusted_rand_index"] < 0.2
    assert report.agreement["nodes"] == 16
    rendered = "\n".join(render_attribute(report))
    assert "random baseline" in rendered
    assert "### The clusters against the attribute" in rendered


def test_the_rendered_section_leads_with_the_counts_and_names_the_nulls() -> None:
    text = "\n".join(
        render_attribute(analyse_attribute(two_cliques(), "region", seed=SEED, permutations=20))
    )

    assert "## By attribute: region" in text
    assert "hypothesis about where this network divides, not a finding" in text
    assert "| value | nodes | share of labelled |" in text
    assert "label shuffles" in text
    assert "degree-preserving rewirings scoring the same partition" in text


def test_attribute_labels_reads_only_the_nodes_that_carry_a_value() -> None:
    graph = two_cliques()
    graph.add_node("untagged-1")
    labels = attribute_labels(graph, "region")
    assert "untagged-1" not in labels and len(labels) == 16


# ----------------------------------------------------------------------------- the filter


def test_matches_needs_every_pair_and_treats_a_missing_key_as_no_match() -> None:
    assert matches({"region": "north"}, None)
    assert matches({"region": "north", "team": "blue"}, {"region": "north"})
    assert not matches({"region": "north"}, {"region": "south"})
    assert not matches({"team": "blue"}, {"region": "north"})
    assert not matches({}, {"region": "north"})


def test_the_speaker_network_filters_on_the_speakers_own_attributes(
    layered: InMemoryGraphStore,
) -> None:
    every = build_network(layered, "speakers", "test-layers")
    assert set(every.nodes) == {"ana", "bo", "cy", "dot"}

    north = build_network(layered, "speakers", "test-layers", where={"region": "north"})

    assert set(north.nodes) == {"ana"}  # bo and cy are south; dot was never tagged
    assert north.nodes["ana"][attr_key("region")] == "north"
    assert "Restricted to nodes attributed region=north" in north.graph["frame"]
    assert north.graph["where"] == "region=north"


def test_the_entity_network_filters_on_the_attributes_of_the_passages_document(
    layered: InMemoryGraphStore,
) -> None:
    north = build_network(
        layered, "entities", "test-layers", min_weight=1, where={"region": "north"}
    )
    south = build_network(
        layered, "entities", "test-layers", min_weight=1, where={"region": "south"}
    )

    # North is posts 1 and 3: Alpha with Beta, then Beta with Gamma.
    assert {north.nodes[n]["name"] for n in north.nodes} == {"Alpha", "Beta", "Gamma"}
    assert north.has_edge("product:alpha", "product:beta")
    assert not north.has_edge("product:alpha", "product:gamma")
    # South is posts 2 and 4, which complain about Alpha and Gamma together, twice.
    assert south["product:alpha"]["product:gamma"]["weight"] == 2
    assert "product:beta" not in south


def test_an_entity_node_carries_the_value_most_of_its_documents_have(
    layered: InMemoryGraphStore,
) -> None:
    graph = build_network(layered, "entities", "test-layers", min_weight=1)

    # Beta is named only in the two north posts; Alpha is in one north and two south ones.
    assert graph.nodes["product:beta"][attr_key("region")] == "north"
    assert graph.nodes["product:beta"][attr_count_key("region")] == 1
    assert graph.nodes["product:alpha"][attr_key("region")] == "south"
    assert graph.nodes["product:alpha"][attr_count_key("region")] == 2


def test_the_topic_network_recomputes_over_the_documents_the_filter_leaves(
    layered: InMemoryGraphStore,
) -> None:
    north = build_network(layered, "topics", "test-layers", min_weight=1, where={"region": "north"})

    assert set(north.nodes) == {"pricing", "service"}  # the two north posts' topics
    assert north["pricing"]["service"]["weight"] == 2
    assert north.nodes["service"][attr_key("region")] == "north"


def test_the_two_mode_network_needs_both_ends_to_match(layered: InMemoryGraphStore) -> None:
    graph = build_network(layered, "speakers-entities", "test-layers", where={"region": "south"})

    assert {n for n in graph.nodes if graph.nodes[n]["mode"] == "speaker"} == {"bo", "cy"}
    assert "ana" not in graph  # a north speaker, even though she names the same entities
    assert graph.has_edge("bo", "product:gamma")


def test_a_filter_that_matches_nothing_gives_an_empty_network_rather_than_everything(
    layered: InMemoryGraphStore,
) -> None:
    graph = build_network(layered, "speakers", "test-layers", where={"region": "east"})
    assert graph.number_of_nodes() == 0


def test_analyze_adds_the_attribute_section_when_asked_and_not_otherwise(
    layered: InMemoryGraphStore,
) -> None:
    graph = build_network(layered, "entities", "test-layers", min_weight=1)

    plain = run_analysis(
        layered, graph, persona_id="test-layers", network="entities", method="louvain", seed=SEED
    )
    by_region = run_analysis(
        layered,
        graph,
        persona_id="test-layers",
        network="entities",
        method="louvain",
        by="region",
        permutations=20,
        samples=10,
        seed=SEED,
    )

    assert plain.attribute is None
    assert by_region.attribute is not None
    assert by_region.attribute.key == "region"
    assert by_region.attribute.counts == {"south": 2, "north": 1}
