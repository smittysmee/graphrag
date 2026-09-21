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
from graphrag.sna.attributes import (
    analyse_attribute,
    attribute_labels,
    attribute_payload,
    render_attribute,
)
from graphrag.sna.export import (
    INHERITED,
    OWN,
    attr_count_key,
    attr_key,
    attr_source_key,
    build_network,
    matches,
)

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


# ----------------------------------------------------------------------- where the labels came from


def inherited(graph: nx.Graph) -> nx.Graph:
    """The same two cliques, but with every label marked as borrowed from documents."""
    copy = graph.copy()
    for node in copy.nodes:
        copy.nodes[node][attr_source_key("region")] = INHERITED
        copy.nodes[node]["documents"] = 1
    return copy


def test_a_label_the_node_carries_itself_is_measured_as_a_finding() -> None:
    report = analyse_attribute(two_cliques(), "region", seed=SEED, permutations=50, samples=20)

    assert report.own == 16 and report.inherited == 0
    assert not report.confounded
    assert "tracks a real division" in report.verdict


def test_a_borrowed_label_is_reported_as_circular_however_well_it_scores() -> None:
    """The case the whole provenance key exists for.

    These are the same two cliques that score above 0.9 with a z-score over 3 -- the strongest
    result the measure can produce. Borrowed from the documents the edges came from, that score
    is a restatement of how the labels were assigned, and the shuffle null cannot subtract it:
    it destroys the correlation, so the null sits at zero and the z-score goes up rather than
    down. The verdict has to say so instead of reporting the number.
    """
    report = analyse_attribute(
        inherited(two_cliques()), "region", seed=SEED, permutations=50, samples=20
    )

    assert report.assortativity is not None and report.assortativity > 0.9
    assert report.assortativity_z > 3  # the null does not catch it, which is the point
    assert report.inherited == 16 and report.own == 0
    assert report.inherited_share == 1.0 and report.confounded
    assert report.single_document == 16
    assert "Not a reading of this attribute" in report.verdict
    assert "took their value from the very documents the edges were drawn from" in report.verdict


def test_a_mostly_intrinsic_attribute_is_still_read_as_a_finding() -> None:
    """The confound is a matter of degree: a few borrowed labels do not void the measure."""
    graph = two_cliques()
    for node in ("north-1", "south-1"):
        graph.nodes[node][attr_source_key("region")] = INHERITED

    report = analyse_attribute(graph, "region", seed=SEED, permutations=50, samples=20)

    assert report.inherited == 2 and report.own == 14
    assert not report.confounded
    assert "tracks a real division" in report.verdict
    assert any("carry no value of their own" in note for note in report.notes)


def test_the_rendered_section_says_where_the_labels_came_from() -> None:
    text = "\n".join(
        render_attribute(
            analyse_attribute(
                inherited(two_cliques()), "region", seed=SEED, permutations=20, samples=10
            )
        )
    )

    assert "Recorded about the node itself: 0" in text
    assert "Borrowed from the node's documents: 16" in text
    assert "from a single document" in text


def test_the_payload_carries_the_provenance_so_a_json_reader_sees_it_too() -> None:
    payload = attribute_payload(
        analyse_attribute(
            inherited(two_cliques()), "region", seed=SEED, permutations=20, samples=10
        )
    )

    assert payload["own"] == 0 and payload["inherited"] == 16
    assert payload["confounded"] is True
    assert payload["single_document"] == 16
    assert payload["sources"] == {INHERITED: 16}


# ------------------------------------------------------------------- entities with their own values


def test_an_entity_with_its_own_value_keeps_it_against_the_documents_that_name_it(
    layered: InMemoryGraphStore,
) -> None:
    """Alpha sits in one north document and two south ones, so the majority says south.

    Given a value of its own, the node stops being a summary of where it was mentioned and
    starts being a property of the thing -- which is the difference between an assortativity
    that describes the projection and one that describes the entities.
    """
    layered.set_entity_attributes("test-layers", "product:alpha", {"region": "north"})

    graph = build_network(layered, "entities", "test-layers", min_weight=1)

    assert graph.nodes["product:alpha"][attr_key("region")] == "north"
    assert graph.nodes["product:alpha"][attr_count_key("region")] == 1
    assert graph.nodes["product:alpha"][attr_source_key("region")] == OWN
    # Beta was never given one, so it still borrows, and says so.
    assert graph.nodes["product:beta"][attr_key("region")] == "north"
    assert graph.nodes["product:beta"][attr_source_key("region")] == INHERITED


def test_the_entity_filter_prefers_the_entitys_own_value_over_its_documents(
    layered: InMemoryGraphStore,
) -> None:
    """An entity tagged north is north in every passage that names it, south documents included.

    Alpha is named in posts 1, 2 and 4, of which only post 1 is north. Borrowing, it would be a
    south entity and ``region=north`` would keep one of its three mentions. Carrying its own
    value, all three survive the filter and it is out of the south network altogether -- an
    entity, not a summary of where it was mentioned.

    Its neighbours are unaffected: Gamma has no value of its own, so it still borrows from the
    south documents it sits in and the filter drops it, which is why no Alpha-Gamma edge appears
    here. An edge needs both of its endpoints to survive.
    """
    layered.set_entity_attributes("test-layers", "product:alpha", {"region": "north"})

    north = build_network(
        layered, "entities", "test-layers", min_weight=1, where={"region": "north"}
    )
    south = build_network(
        layered, "entities", "test-layers", min_weight=1, where={"region": "south"}
    )

    assert north.nodes["product:alpha"]["documents"] == 3
    assert north.nodes["product:alpha"]["mentions"] == 3
    assert not north.has_edge("product:alpha", "product:gamma")  # Gamma still borrows, so it went
    assert "product:alpha" not in south  # and Alpha has left the south network entirely


def test_a_speakers_value_is_its_own_in_every_network_it_appears_in(
    layered: InMemoryGraphStore,
) -> None:
    """A speaker is what an attribution pass said it is, so nothing about it is borrowed."""
    speakers = build_network(layered, "speakers", "test-layers")
    two_mode = build_network(layered, "speakers-entities", "test-layers")

    assert speakers.nodes["ana"][attr_source_key("region")] == OWN
    assert two_mode.nodes["ana"][attr_source_key("region")] == OWN
    assert two_mode.nodes["product:beta"][attr_source_key("region")] == INHERITED
