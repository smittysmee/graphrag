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
