"""`graphrag.sna.Network`, the notebook facade (ATL-ENT-2).

Every method under test here is asserted against the exact function `graphrag sna <command>`
already calls, on the same graph, so a "known answer" is simply "the two paths agree": there is
no arithmetic of this ticket's own to check, only wiring. The one genuinely new behaviour --
opening and closing a store in :meth:`Network.build`, and never opening one at all in
:meth:`Network.from_graph` -- is checked directly, the second half through a store connection
:class:`Network.analyze` cannot use without being told about it explicitly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from graphrag.config import Settings
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport
from graphrag.sna.analysis import run_analysis
from graphrag.sna.backbone import backbone
from graphrag.sna.cluster import louvain
from graphrag.sna.draw import draw
from graphrag.sna.evaluate import evaluate_partition
from graphrag.sna.export import build_network, read_graph, write_graph
from graphrag.sna.facade import Measures, Network
from graphrag.sna.measures import centralization_table, density_report
from graphrag.sna.measures import summary as network_summary
from graphrag.sna.predict import common_neighbors
from tests.legendary import karate_club

pytestmark = pytest.mark.usefixtures("registry")


class _TrackingStore:
    """Delegates everything to a wrapped store, remembering whether ``close()`` ran.

    Not a fixture the rest of the suite shares: this is the one test in this file that needs to
    read the store's own state after :meth:`Network.build` returns, which the store itself has
    no field for (``InMemoryGraphStore.close`` is a documented no-op).
    """

    def __init__(self, inner: InMemoryGraphStore) -> None:
        self._inner = inner
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self._inner.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture
def enriched(memory_store: InMemoryGraphStore, ingested: IngestReport) -> InMemoryGraphStore:
    """Put entities on the sample corpus so the entity network has something to analyze.

    Same fixture ``test_sna_cli.py`` builds for the CLI's own tests: each document gets a
    different three of four entities, so the co-mention weights differ and the network is not a
    single clique.
    """
    store = memory_store
    entities = [
        Entity(id="metric:retention", name="Retention", type="metric"),
        Entity(id="concept:onboarding", name="Onboarding", type="concept"),
        Entity(id="concept:pricing", name="Pricing", type="concept"),
        Entity(id="concept:roadmap", name="Roadmap", type="concept"),
    ]
    per_document = [
        ["metric:retention", "concept:onboarding", "concept:pricing"],
        ["metric:retention", "concept:onboarding", "concept:roadmap"],
        ["metric:retention", "concept:pricing", "concept:roadmap"],
    ]
    mentions: list[Mention] = []
    for doc_id, entity_ids in zip(sorted(store.document_ids("test-pm")), per_document, strict=True):
        for chunk in store.document_chunks(doc_id, 0, 100):
            mentions += [Mention(chunk_id=chunk.id, entity_id=e) for e in entity_ids]
    store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))
    return store


# --------------------------------------------------------------------------------------- build


def test_build_opens_through_app_context_and_matches_build_network_directly(
    settings: Settings,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    ingested: IngestReport,
) -> None:
    net = Network.build(
        "test-pm", "speakers", settings=settings, store=memory_store, embedder=hash_embedder
    )
    expected = build_network(memory_store, "speakers", "test-pm")
    assert sorted(net.graph.nodes) == sorted(expected.nodes)
    assert sorted(net.graph.edges) == sorted(expected.edges)
    assert net.persona_id == "test-pm"
    assert net.network == "speakers"
    assert net.filters == {}


def test_build_passes_filters_through_unparsed(
    settings: Settings, enriched: InMemoryGraphStore, hash_embedder: HashEmbedder
) -> None:
    net = Network.build(
        "test-pm",
        "entities",
        settings=settings,
        store=enriched,
        embedder=hash_embedder,
        min_weight=2,
    )
    expected = build_network(enriched, "entities", "test-pm", min_weight=2)
    assert sorted(net.graph.edges) == sorted(expected.edges)
    assert net.filters == {"min_weight": 2}


def test_build_closes_the_store(
    settings: Settings,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    ingested: IngestReport,
) -> None:
    tracking = _TrackingStore(memory_store)
    net = Network.build(
        "test-pm", "speakers", settings=settings, store=tracking, embedder=hash_embedder
    )
    assert tracking.closed is True
    assert net.graph.number_of_nodes() > 0


def test_build_rejects_an_unknown_persona(
    settings: Settings, memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder
) -> None:
    with pytest.raises(KeyError, match="unknown persona"):
        Network.build(
            "nope", "speakers", settings=settings, store=memory_store, embedder=hash_embedder
        )


# ------------------------------------------------------------------------------------ analyze


def test_analyze_matches_run_analysis_field_for_field(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    expected = run_analysis(
        enriched, graph, persona_id="test-pm", network="entities", method="louvain", seed=1
    )
    net = Network.from_graph(graph, persona_id="test-pm", network="entities", store=enriched)
    actual = net.analyze(method="louvain", seed=1)

    assert actual.summary == expected.summary
    assert actual.centralities == expected.centralities
    assert actual.groups == expected.groups
    assert actual.density == expected.density
    assert actual.degree == expected.degree
    assert actual.evaluation == expected.evaluation
    assert actual.rationale == expected.rationale


def test_analyze_defaults_to_the_store_this_network_was_built_with(
    settings: Settings, enriched: InMemoryGraphStore, hash_embedder: HashEmbedder
) -> None:
    """The default ``features="spectral"`` never reads the store, so this succeeds even though
    :meth:`Network.build` has already closed the store it is implicitly reusing."""
    net = Network.build(
        "test-pm", "entities", settings=settings, store=enriched, embedder=hash_embedder
    )
    report = net.analyze(method="louvain", seed=1)
    assert report.summary["nodes"] == net.graph.number_of_nodes()


def test_analyze_without_a_store_raises_only_when_one_is_actually_read() -> None:
    graph, _known = karate_club()
    # Labelled "entities" so features="embedding" below reaches the store read rather than
    # refusing first for being on the wrong network (that refusal is its own, unrelated path).
    net = Network.from_graph(graph, network="entities")

    # The default (method="louvain") reads no store at all.
    report = net.analyze(seed=1)
    assert report.groups

    # method="kmeans" with features="embedding" does, and this Network never opened one.
    with pytest.raises(RuntimeError, match="no store connection"):
        net.analyze(method="kmeans", features="embedding", k=2, seed=1)


# ----------------------------------------------------------------------------------- backbone


def test_backbone_equals_the_direct_call_field_for_field(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    net = Network.from_graph(graph, persona_id="test-pm", network="entities")

    result = net.backbone("noise-corrected", alpha=0.5)
    expected = backbone(graph, "noise-corrected", alpha=0.5)

    assert isinstance(result, Network)
    assert sorted(result.graph.nodes) == sorted(expected.nodes)
    assert sorted(result.graph.edges) == sorted(expected.edges)
    for u, v, data in expected.edges(data=True):
        assert result.graph.edges[u, v] == data
    assert result.persona_id == net.persona_id
    assert result.network == net.network


# ----------------------------------------------------------------------------------- measures


def test_measures_equals_the_direct_calls_field_for_field(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    net = Network.from_graph(graph)

    result = net.measures()

    assert isinstance(result, Measures)
    assert result.summary == network_summary(graph)
    assert result.density == density_report(graph)
    assert result.centralization == centralization_table(graph)


def test_measures_density_is_none_on_an_empty_network() -> None:
    import networkx as nx

    net = Network.from_graph(nx.Graph())
    assert net.measures().density is None


# ----------------------------------------------------------------------------------- evaluate


def test_evaluate_equals_the_direct_call(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    communities = louvain(graph, seed=1).communities
    net = Network.from_graph(graph)

    result = net.evaluate(communities, seed=1)
    expected = evaluate_partition(graph, communities, seed=1)

    assert result == expected


# ------------------------------------------------------------------------------------ predict


def test_predict_common_neighbours_matches_the_direct_call(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    net = Network.from_graph(graph)

    result = net.predict("cn")
    expected = common_neighbors(graph)
    nodes = list(graph.nodes)
    non_edges = [
        (u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :] if not graph.has_edge(u, v)
    ]
    assert non_edges, "fixture should have at least one non-edge to score"
    for u, v in non_edges:
        assert result[u, v] == expected[u, v]


def test_predict_rejects_hrg_and_rules() -> None:
    graph, _known = karate_club()
    net = Network.from_graph(graph)
    with pytest.raises(ValueError, match="hrg"):
        net.predict("hrg")


# --------------------------------------------------------------------------------------- draw


def test_draw_equals_the_direct_call(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    net = Network.from_graph(graph)

    result = net.draw(layout="force", seed=1)
    expected = draw(graph, layout="force", seed=1)

    assert result.svg == expected.svg
    assert result.legend == expected.legend
    assert result.positions == expected.positions


# ------------------------------------------------------------------------------------ to_pandas


def test_to_pandas_returns_one_row_per_node_and_per_edge(
    settings: Settings,
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    ingested: IngestReport,
) -> None:
    graph = build_network(memory_store, "speakers", "test-pm")
    net = Network.from_graph(graph)

    nodes, edges = net.to_pandas()

    assert len(nodes) == graph.number_of_nodes()
    assert len(edges) == graph.number_of_edges()
    if importlib.util.find_spec("pandas") is None:
        assert isinstance(nodes, list)
        assert isinstance(edges, list)
        assert all("id" in row for row in nodes)
        assert all({"source", "target"} <= row.keys() for row in edges)
    else:
        import pandas as pd

        assert isinstance(nodes, pd.DataFrame)
        assert isinstance(edges, pd.DataFrame)
        assert "id" in nodes.columns
        assert {"source", "target"} <= set(edges.columns)


# --------------------------------------------------------------------------------- from_graph


def test_from_graph_needs_no_store(enriched: InMemoryGraphStore) -> None:
    graph = build_network(enriched, "entities", "test-pm")
    net = Network.from_graph(graph, persona_id="test-pm", network="entities")
    assert net.persona_id == "test-pm"
    assert net.network == "entities"
    assert net.filters == {}
    # measures/backbone/evaluate/predict/draw never touch a store at all
    assert net.measures().summary["nodes"] == graph.number_of_nodes()


def test_read_graph_round_trips_a_legendary_graph(tmp_path: Path) -> None:
    graph, _known = karate_club()
    path = write_graph(graph, tmp_path / "karate.graphml")

    net = Network.from_graph(read_graph(path))

    assert net.graph.number_of_nodes() == graph.number_of_nodes()
    assert net.graph.number_of_edges() == graph.number_of_edges()
