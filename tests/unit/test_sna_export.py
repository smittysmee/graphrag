"""Building networks out of a store, and the bipartite projection both of them share."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport
from graphrag.sna.export import (
    bipartite_projection,
    build_network,
    ego,
    entity_co_mention,
    speaker_co_participation,
    speaker_entity_bipartite,
    topic_co_occurrence,
    write_graph,
)


def test_bipartite_projection_counts_shared_partners() -> None:
    pairs = [
        ("ana", "doc-1"),
        ("bo", "doc-1"),
        ("ana", "doc-2"),
        ("bo", "doc-2"),
        ("cy", "doc-2"),
        ("cy", "doc-3"),
    ]
    left = bipartite_projection(pairs, side="left")
    assert left["ana"]["bo"]["weight"] == 2  # doc-1 and doc-2
    assert left["bo"]["cy"]["weight"] == 1
    assert not left.has_edge("ana", "cy") or left["ana"]["cy"]["weight"] == 1

    right = bipartite_projection(pairs, side="right")
    assert right["doc-1"]["doc-2"]["weight"] == 2  # ana and bo appear in both
    assert right["doc-2"]["doc-3"]["weight"] == 1  # only cy


def test_speaker_network_from_the_sample_corpus(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    graph = speaker_co_participation(memory_store, "test-pm")
    assert graph.graph["network"] == "speakers"
    assert graph.graph.get("frame")
    # The host appears in all three episodes, each guest in one, so the host is the only hub.
    assert graph.number_of_nodes() == 4
    assert graph.nodes["Lenny Rachitsky"]["documents"] == 3
    assert graph.nodes["Ada North"]["documents"] == 1
    assert graph.nodes["Ada North"]["chunks"] >= 1
    assert graph.degree("Lenny Rachitsky") == 3
    assert not graph.has_edge("Ada North", "Ben Oduya")
    for guest in ("Ada North", "Ben Oduya", "Cleo Vance"):
        assert graph["Lenny Rachitsky"][guest]["weight"] == 1


def test_speaker_network_is_scoped_by_persona_and_source(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    assert speaker_co_participation(memory_store, "other-persona").number_of_nodes() == 0
    assert speaker_co_participation(memory_store, "test-pm", "nope").number_of_nodes() == 0
    scoped = speaker_co_participation(memory_store, "test-pm", "test-podcast")
    assert scoped.number_of_nodes() == 4


def test_entity_network_weighs_shared_passages(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    doc_ids = sorted(memory_store.document_ids("test-pm"))
    passages = [memory_store.document_chunks(doc_id, 0, 1)[0] for doc_id in doc_ids]
    entities = [
        Entity(id="metric:retention", name="Retention", type="metric"),
        Entity(id="concept:onboarding", name="Onboarding", type="concept"),
        Entity(id="concept:pricing", name="Pricing", type="concept"),
    ]
    mentions = [
        Mention(chunk_id=passages[0].id, entity_id="metric:retention"),
        Mention(chunk_id=passages[0].id, entity_id="concept:onboarding"),
        Mention(chunk_id=passages[1].id, entity_id="metric:retention"),
        Mention(chunk_id=passages[1].id, entity_id="concept:onboarding"),
        Mention(chunk_id=passages[1].id, entity_id="concept:pricing"),
    ]
    memory_store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))

    graph = entity_co_mention(memory_store, "test-pm", min_weight=2)
    # Retention and Onboarding share two passages; Pricing shares only one with each, so
    # min_weight=2 removes its edges and leaves it isolated, which prunes it out.
    assert set(graph.nodes) == {"metric:retention", "concept:onboarding"}
    assert graph["metric:retention"]["concept:onboarding"]["weight"] == 2
    assert graph.nodes["metric:retention"]["label"] == "Retention"
    assert graph.nodes["metric:retention"]["mentions"] == 2
    assert graph.nodes["metric:retention"]["documents"] == 2

    loose = entity_co_mention(memory_store, "test-pm", min_weight=1)
    assert "concept:pricing" in loose
    assert loose["concept:pricing"]["metric:retention"]["weight"] == 1

    typed = entity_co_mention(memory_store, "test-pm", min_weight=1, types=["concept"])
    assert set(typed.nodes) == {"concept:onboarding", "concept:pricing"}


def test_topic_network_uses_the_stored_cooccurrence(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    graph = topic_co_occurrence(memory_store, "test-pm", min_weight=1)
    assert graph.number_of_edges() > 0
    assert "retention" in graph
    assert graph.nodes["retention"]["documents"] == 2
    heavy = topic_co_occurrence(memory_store, "test-pm", min_weight=99)
    assert heavy.number_of_edges() == 0


def test_build_network_dispatches_and_rejects_unknown_names(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    for name in ("speakers", "entities", "topics"):
        assert build_network(memory_store, name, "test-pm").graph["network"] == name  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="network must be one of"):
        build_network(memory_store, "people", "test-pm")  # type: ignore[arg-type]


def test_networks_are_built_in_a_stable_node_order(
    ingested: IngestReport, memory_store: InMemoryGraphStore
) -> None:
    """A `--seed` only buys reproducibility if the graph itself is identical every time.

    Louvain shuffles the node list and networkx builds its matrices in insertion order, so
    nodes arriving from a set (whose iteration order varies between processes) silently make
    the same command return different communities and different eigenvector scores.
    """
    doc_ids = sorted(memory_store.document_ids("test-pm"))
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="metric:retention", name="Retention", type="metric"),
                Entity(id="concept:onboarding", name="Onboarding", type="concept"),
            ],
            mentions=[
                Mention(chunk_id=memory_store.document_chunks(doc_id, 0, 1)[0].id, entity_id=e)
                for doc_id in doc_ids
                for e in ("metric:retention", "concept:onboarding")
            ],
        )
    )
    for name in ("speakers", "entities", "topics"):
        nodes = list(build_network(memory_store, name, "test-pm", min_weight=1).nodes)
        assert nodes == sorted(nodes), name
        assert nodes  # a sorted empty list would pass the assertion above vacuously


def test_ego_takes_a_neighbourhood() -> None:
    graph = nx.Graph()
    nx.add_path(graph, ["a", "b", "c", "d"])
    assert set(ego(graph, "b", radius=1)) == {"a", "b", "c"}
    assert set(ego(graph, "b", radius=2)) == {"a", "b", "c", "d"}
    with pytest.raises(KeyError, match="not in this network"):
        ego(graph, "zz")


def test_write_graph_supports_graphml_and_json(tmp_path: Path) -> None:
    graph = nx.Graph(network="speakers", frame="a sample")
    graph.add_edge("ana", "bo", weight=3)
    graph.nodes["ana"]["documents"] = 2

    graphml = write_graph(graph, tmp_path / "net" / "g.graphml")
    assert graphml.exists()
    back = nx.read_graphml(graphml)
    assert back["ana"]["bo"]["weight"] == 3

    as_json = write_graph(graph, tmp_path / "g.json")
    payload = json.loads(as_json.read_text())
    assert {n["id"] for n in payload["nodes"]} == {"ana", "bo"}

    with pytest.raises(ValueError, match="unsupported export format"):
        write_graph(graph, tmp_path / "g.dot")


# ------------------------------------------------------------- the annotation and date layers


def test_a_date_window_keeps_only_speakers_with_a_dated_post_in_it(
    layered: InMemoryGraphStore,
) -> None:
    """An undated post is outside every window rather than assumed to be inside one."""
    everyone = speaker_co_participation(layered, "test-layers")
    assert set(everyone.nodes) == {"ana", "bo", "cy", "dot"}

    early = speaker_co_participation(layered, "test-layers", since="2025-01-01", until="2025-03-01")
    assert set(early.nodes) == {"ana", "bo"}  # dot is undated, cy posted in June
    late = speaker_co_participation(layered, "test-layers", since="2025-06-01")
    assert set(late.nodes) == {"ana", "cy"}
    assert (
        speaker_co_participation(layered, "test-layers", since="2030-01-01").number_of_nodes() == 0
    )
    assert "2025-06-01" in late.graph["frame"]


def test_a_window_reaches_the_entity_network_through_the_dated_passages(
    layered: InMemoryGraphStore,
) -> None:
    """The entity network has no date of its own, so it inherits the documents' dates."""
    whole = entity_co_mention(layered, "test-layers", min_weight=1)
    assert set(whole.nodes) == {"product:alpha", "product:beta", "product:gamma"}

    first_half = entity_co_mention(layered, "test-layers", min_weight=1, until="2025-03-01")
    assert first_half["product:alpha"]["product:beta"]["weight"] == 1  # post 1
    assert first_half["product:alpha"]["product:gamma"]["weight"] == 1  # post 2
    assert not first_half.has_edge("product:beta", "product:gamma")

    second_half = entity_co_mention(layered, "test-layers", min_weight=1, since="2025-06-01")
    assert second_half["product:beta"]["product:gamma"]["weight"] == 1  # post 3
    assert not second_half.has_edge("product:alpha", "product:beta")


def test_a_stance_filter_produces_a_different_network_not_a_thinner_one(
    layered: InMemoryGraphStore,
) -> None:
    """Praise joins Alpha to Beta; complaint joins Alpha to Gamma. The partition flips."""
    praise = entity_co_mention(layered, "test-layers", min_weight=1, stances=["praise"])
    assert set(praise.nodes) == {"product:alpha", "product:beta"}
    assert praise["product:alpha"]["product:beta"]["weight"] == 1

    complaint = entity_co_mention(layered, "test-layers", min_weight=1, stances=["complaint"])
    assert set(complaint.nodes) == {"product:alpha", "product:gamma"}
    assert complaint["product:alpha"]["product:gamma"]["weight"] == 2  # posts 2 and 4

    # The two are not complements of the unfiltered network: nothing annotated `substitution`
    # exists, so asking for it returns an empty network rather than "everything else".
    assert (
        entity_co_mention(layered, "test-layers", 1, stances=["substitution"]).number_of_nodes()
        == 0
    )
    both = entity_co_mention(layered, "test-layers", min_weight=1, stances=["praise", "complaint"])
    assert set(both.nodes) == {"product:alpha", "product:beta", "product:gamma"}
    assert "praise or complaint" in both.graph["frame"]


def test_a_facet_filter_keeps_only_the_passages_about_that_function(
    layered: InMemoryGraphStore,
) -> None:
    quoting = entity_co_mention(layered, "test-layers", min_weight=1, facets=["quoting"])
    assert quoting["product:alpha"]["product:beta"]["weight"] == 1  # post 1
    assert quoting["product:beta"]["product:gamma"]["weight"] == 1  # post 3
    assert not quoting.has_edge("product:alpha", "product:gamma")

    outage = entity_co_mention(layered, "test-layers", min_weight=1, facets=["outage"])
    assert set(outage.nodes) == {"product:alpha", "product:gamma"}
    assert outage["product:alpha"]["product:gamma"]["weight"] == 2

    speakers = speaker_co_participation(layered, "test-layers", facets=["outage"])
    assert set(speakers.nodes) == {"bo", "cy"}  # ana wrote only quoting passages
    assert "outage" in speakers.graph["frame"]


def test_a_facet_recomputes_the_topic_network_over_the_surviving_documents(
    layered: InMemoryGraphStore,
) -> None:
    """A stored aggregate cannot be filtered, so the filtered network is recomputed."""
    quoting = topic_co_occurrence(layered, "test-layers", min_weight=1, facets=["quoting"])
    assert set(quoting.edges) == {("pricing", "service")}
    assert quoting["pricing"]["service"]["weight"] == 2  # posts 1 and 3
    assert quoting.nodes["pricing"]["documents"] == 2

    outage = topic_co_occurrence(layered, "test-layers", min_weight=1, facets=["outage"])
    assert set(outage.edges) == {("outages", "service")}
    assert "outage" in outage.graph["frame"]


def test_the_bipartite_network_weighs_speaker_to_entity_edges_in_documents(
    layered: InMemoryGraphStore,
) -> None:
    graph = speaker_entity_bipartite(layered, "test-layers")
    assert graph.graph["network"] == "speakers-entities"
    assert set(graph.nodes) == {"ana", "bo", "cy", "product:alpha", "product:beta", "product:gamma"}
    assert graph.nodes["ana"]["mode"] == "speaker"
    assert graph.nodes["product:alpha"]["mode"] == "entity"
    assert graph.nodes["product:alpha"]["label"] == "Alpha"
    # ana named Beta in two documents and Alpha in one, so the edges differ in weight.
    assert graph["ana"]["product:beta"]["weight"] == 2
    assert graph["ana"]["product:alpha"]["weight"] == 1
    assert not graph.has_edge("ana", "dot")  # dot's post mentions nothing
    assert graph.nodes["ana"]["partners"].startswith("Beta")


def test_projecting_the_bipartite_network_counts_the_nodes_two_share(
    layered: InMemoryGraphStore,
) -> None:
    """Projected weights are co-membership counts, and they inflate: read them as such."""
    speakers = speaker_entity_bipartite(layered, "test-layers", project="speakers")
    assert set(speakers.nodes) == {"ana", "bo", "cy"}
    assert speakers["bo"]["cy"]["weight"] == 2  # both named Alpha and Gamma
    assert speakers["ana"]["bo"]["weight"] == 2  # Alpha and Gamma again
    assert speakers.nodes["ana"]["mode"] == "speaker"
    assert "Beta" in speakers.nodes["ana"]["partners"]

    entities = speaker_entity_bipartite(layered, "test-layers", project="entities")
    assert entities["product:alpha"]["product:gamma"]["weight"] == 3  # ana, bo and cy
    assert entities["product:alpha"]["product:beta"]["weight"] == 1  # only ana
    assert entities.nodes["product:gamma"]["mode"] == "entity"
    assert "ana" in entities.nodes["product:beta"]["partners"]

    heavy = speaker_entity_bipartite(layered, "test-layers", min_weight=2, project="entities")
    assert set(heavy.nodes) == {"product:alpha", "product:gamma"}


def test_the_bipartite_network_takes_the_same_filters_as_the_others(
    layered: InMemoryGraphStore,
) -> None:
    complaints = speaker_entity_bipartite(layered, "test-layers", stances=["complaint"])
    assert set(complaints.nodes) == {"bo", "cy", "product:alpha", "product:gamma"}
    early = speaker_entity_bipartite(layered, "test-layers", until="2025-03-01")
    assert set(early.nodes) == {"ana", "bo", "product:alpha", "product:beta", "product:gamma"}
    quoting = speaker_entity_bipartite(
        layered, "test-layers", facets=["quoting"], project="speakers"
    )
    assert set(quoting.nodes) == {"ana"}


def test_build_network_rejects_filters_the_chosen_network_cannot_answer(
    layered: InMemoryGraphStore,
) -> None:
    with pytest.raises(ValueError, match="--stance reads the annotation"):
        build_network(layered, "speakers", "test-layers", stances=["praise"])
    with pytest.raises(ValueError, match="--stance reads the annotation"):
        build_network(layered, "topics", "test-layers", stances=["praise"])
    with pytest.raises(ValueError, match="--project applies to"):
        build_network(layered, "entities", "test-layers", project="speakers")
    assert build_network(layered, "speakers-entities", "test-layers").number_of_nodes() == 6
