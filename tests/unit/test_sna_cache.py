"""The on-disk network cache (ATL-ENT-3), held to known answers about when it hits, when it
misses, and what a caller who asked to skip it gets instead.

Every filter-and-identity combination below is chosen to be a *different* cache entry, and the
"skips the store" tests use a store that counts its own calls rather than a mock, so a hit is
proven by the store never having run the read at all -- not by the graph merely looking right.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pytest
from networkx.utils import graphs_equal

from graphrag.config import Settings
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import (
    Enrichment,
    Entity,
    EntityChunk,
    Mention,
    PersonaSpec,
    SnapshotManifest,
    SourceSpec,
)
from graphrag.pipeline import IngestPipeline
from graphrag.sna.cache import (
    NO_SNAPSHOT_IDENTITY,
    NetworkCache,
    build_network_cached,
    cache_key,
    snapshot_identity,
)


class _CountingStore(InMemoryGraphStore):
    """An ``InMemoryGraphStore`` that counts calls to ``entity_chunk_pairs``, the read
    ``entity_co_mention`` makes for every entity network -- so a cache hit can be proven to
    never reach it, not merely inferred from the graph it returns."""

    def __init__(self) -> None:
        super().__init__()
        self.entity_chunk_pairs_calls = 0

    def entity_chunk_pairs(
        self, persona_id: str, source_id: str | None = None, types: Sequence[str] | None = None
    ) -> list[EntityChunk]:
        self.entity_chunk_pairs_calls += 1
        return super().entity_chunk_pairs(persona_id, source_id, types)


@pytest.fixture
def counting_store() -> _CountingStore:
    return _CountingStore()


@pytest.fixture
def entity_graph_store(
    counting_store: _CountingStore,
    sample_corpus: Path,
    persona: PersonaSpec,
    transcript_source: SourceSpec,
    hash_embedder: HashEmbedder,
) -> _CountingStore:
    """A counting store with a real, ingested corpus plus two mentioned entities sharing two
    passages, which is enough for ``entity_co_mention`` to build a non-empty network.

    Ingests into ``counting_store`` directly rather than through the shared ``ingested``
    fixture, which ingests into a different store (``memory_store``) than the one this test
    needs to count calls on.
    """
    IngestPipeline(counting_store, hash_embedder, doc_batch=2).ingest(
        sample_corpus, persona, transcript_source
    )
    doc_ids = sorted(counting_store.document_ids("test-pm"))
    passages = [counting_store.document_chunks(doc_id, 0, 1)[0] for doc_id in doc_ids]
    entities = [
        Entity(id="metric:retention", name="Retention", type="metric"),
        Entity(id="concept:onboarding", name="Onboarding", type="concept"),
    ]
    mentions = [
        Mention(chunk_id=passages[0].id, entity_id="metric:retention"),
        Mention(chunk_id=passages[0].id, entity_id="concept:onboarding"),
        Mention(chunk_id=passages[1].id, entity_id="metric:retention"),
        Mention(chunk_id=passages[1].id, entity_id="concept:onboarding"),
    ]
    counting_store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))
    return counting_store


def _manifest(
    persona_id: str, *, source_commit: str | None, chunk_count: int = 3
) -> SnapshotManifest:
    return SnapshotManifest(
        persona_id=persona_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        embedding_model="hash-test",
        embedding_dim=8,
        document_count=1,
        chunk_count=chunk_count,
        source_commit=source_commit,
    )


def _write_manifest(snapshots_dir: Path, manifest: SnapshotManifest) -> None:
    target = snap.snapshot_dir(snapshots_dir, manifest.persona_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / snap.MANIFEST).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------- snapshot_identity


def test_snapshot_identity_is_the_constant_when_there_is_no_manifest(tmp_path: Path) -> None:
    assert snapshot_identity(tmp_path, "no-such-persona") == NO_SNAPSHOT_IDENTITY


def test_snapshot_identity_reads_the_source_commit_when_the_manifest_has_one(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, _manifest("p1", source_commit="abc123"))
    assert snapshot_identity(tmp_path, "p1") == "commit:abc123"


def test_snapshot_identity_falls_back_to_a_manifest_hash_without_a_commit(tmp_path: Path) -> None:
    _write_manifest(tmp_path, _manifest("p1", source_commit=None, chunk_count=10))
    first = snapshot_identity(tmp_path, "p1")
    assert first.startswith("manifest:")
    # Re-exporting with a different chunk count is a different identity, even with no commit.
    _write_manifest(tmp_path, _manifest("p1", source_commit=None, chunk_count=11))
    assert snapshot_identity(tmp_path, "p1") != first


# ----------------------------------------------------------------------------- cache_key


def test_cache_key_changes_with_persona_network_filters_or_identity() -> None:
    base = cache_key("p1", "entities", "commit:aaa", {"min_weight": 2})
    assert cache_key("p2", "entities", "commit:aaa", {"min_weight": 2}) != base
    assert cache_key("p1", "speakers", "commit:aaa", {"min_weight": 2}) != base
    assert cache_key("p1", "entities", "commit:bbb", {"min_weight": 2}) != base
    assert cache_key("p1", "entities", "commit:aaa", {"min_weight": 3}) != base
    assert cache_key("p1", "entities", "commit:aaa", {"min_weight": 2}) == base


def test_cache_key_does_not_care_whether_a_filter_arrived_as_a_list_a_tuple_or_a_set() -> None:
    by_list = cache_key("p1", "entities", "id", {"types": ["a", "b"]})
    by_tuple = cache_key("p1", "entities", "id", {"types": ("a", "b")})
    assert by_list == by_tuple


def test_cache_key_does_not_care_about_dict_filter_key_order() -> None:
    a = cache_key("p1", "entities", "id", {"min_weight": 1, "types": ["x"]})
    b = cache_key("p1", "entities", "id", {"types": ["x"], "min_weight": 1})
    assert a == b


# ----------------------------------------------------------------------------- NetworkCache


def test_network_cache_round_trips_a_graph_exactly(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    graph = nx.Graph(network="entities")
    graph.add_node("a", label="Alpha", weight=1)
    graph.add_node("b", label="Beta", weight=2)
    graph.add_edge("a", "b", weight=3)

    cache.put("p1", "key-1", graph)
    restored = cache.get("p1", "key-1")

    assert restored is not None
    assert graphs_equal(restored, graph)
    assert restored.graph == graph.graph  # graphs_equal ignores graph-level attributes


def test_network_cache_get_of_a_missing_key_is_none(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    assert cache.get("p1", "nope") is None


def test_network_cache_get_of_a_corrupted_file_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    path = cache._path("p1", "key-1")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a pickle")
    assert cache.get("p1", "key-1") is None


def test_concurrent_writers_of_one_key_do_not_collide(tmp_path: Path) -> None:
    """Two `sna` commands building the same network at once used to share one temp path, so the
    second rename failed with FileNotFoundError; every writer now has a temp file of its own."""
    from concurrent.futures import ThreadPoolExecutor

    cache = NetworkCache(tmp_path / "cache")
    graph = nx.gnm_random_graph(400, 4000, seed=1)

    def write(_: int) -> None:
        cache.put("p1", "same-key", graph)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(write, i) for i in range(200)]:
            future.result()  # re-raises a collision, if there was one

    restored = cache.get("p1", "same-key")
    assert restored is not None and graphs_equal(restored, graph)
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_clear_sweeps_another_writers_temp_file_without_failing(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    cache.put("p1", "a", nx.Graph())
    stray = cache._path("p1", "a").with_name("a.99999-deadbeef.pickle.tmp")
    stray.write_bytes(b"half-written")
    assert cache.clear("p1") == 1
    assert not stray.exists()
    cache.put("p1", "a", nx.path_graph(3))  # the next write lands as usual
    assert cache.get("p1", "a") is not None


def test_clear_removes_every_cached_file_and_reports_how_many(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    cache.put("p1", "a", nx.Graph())
    cache.put("p1", "b", nx.Graph())
    assert cache.clear() == 2
    assert cache.get("p1", "a") is None
    assert list(cache.directory.rglob("*.pickle")) == []
    assert cache.clear() == 0  # clearing an already-empty (or absent) directory is not an error


def test_clear_of_one_persona_leaves_another_alone(tmp_path: Path) -> None:
    cache = NetworkCache(tmp_path / "cache")
    cache.put("p1", "a", nx.Graph())
    cache.put("p2", "a", nx.Graph())
    assert cache.clear("p1") == 1
    assert cache.get("p1", "a") is None
    assert cache.get("p2", "a") is not None
    assert cache.clear("p1") == 0  # nothing left for p1; p2's entry is not touched by naming p1
    assert cache.get("p2", "a") is not None


# ----------------------------------------------------------------------------- build_network_cached


def test_a_cache_hit_never_touches_the_store(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    first = build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    calls_after_miss = entity_graph_store.entity_chunk_pairs_calls
    assert calls_after_miss >= 1
    assert first.number_of_nodes() == 2

    second = build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls == calls_after_miss  # store untouched
    assert set(second.nodes) == set(first.nodes)
    assert second.graph["network"] == "entities"


def test_a_different_filter_misses_the_cache(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings, min_weight=1)
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings, min_weight=2)
    assert entity_graph_store.entity_chunk_pairs_calls > calls_after_first


def test_a_different_persona_misses_the_cache(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls
    build_network_cached(entity_graph_store, "entities", "other-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls > calls_after_first


def test_no_cache_flag_bypasses_the_cache_entirely(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    build_network_cached(
        entity_graph_store, "entities", "test-pm", settings=settings, no_cache=True
    )
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls
    build_network_cached(
        entity_graph_store, "entities", "test-pm", settings=settings, no_cache=True
    )
    assert entity_graph_store.entity_chunk_pairs_calls == calls_after_first + 1
    assert list(settings.sna_cache_dir.rglob("*.pickle")) == []


def test_settings_sna_cache_false_disables_caching_globally(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    off = settings.model_copy(update={"sna_cache": False})
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=off)
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=off)
    assert entity_graph_store.entity_chunk_pairs_calls == calls_after_first + 1


def test_a_changed_snapshot_identity_misses_the_cache(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    _write_manifest(settings.snapshots_dir, _manifest("test-pm", source_commit="aaa"))
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls

    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls == calls_after_first  # same commit: a hit

    _write_manifest(settings.snapshots_dir, _manifest("test-pm", source_commit="bbb"))
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls > calls_after_first  # new commit: a miss


# ----------------------------------------------------------------------------- persona_fingerprint


def test_persona_fingerprint_counts_documents_chunks_mentions_and_attributed_nodes(
    entity_graph_store: _CountingStore,
) -> None:
    # 3 documents (the sample corpus), 2 mentioned entities each shared across 2 passages: 4
    # mention edges (see `entity_graph_store`), nothing attributed yet.
    documents, chunks, mentions, attributed = entity_graph_store.persona_fingerprint("test-pm")
    assert (documents, mentions, attributed) == (3, 4, 0)
    assert chunks >= 3  # exact chunk count depends on how the sample corpus is split

    entity_graph_store.set_entity_attributes("test-pm", "metric:retention", {"sector": "growth"})
    _, _, _, attributed_after = entity_graph_store.persona_fingerprint("test-pm")
    assert attributed_after == 1

    entity_graph_store.set_document_attributes(
        next(iter(entity_graph_store.document_ids("test-pm"))), {"region": "north"}
    )
    _, _, _, attributed_after_doc = entity_graph_store.persona_fingerprint("test-pm")
    assert attributed_after_doc == 2


def test_persona_fingerprint_is_all_zero_for_an_unknown_persona(
    entity_graph_store: _CountingStore,
) -> None:
    assert entity_graph_store.persona_fingerprint("no-such-persona") == (0, 0, 0, 0)


def test_a_fingerprint_change_misses_the_cache_even_with_no_snapshot_on_disk(
    entity_graph_store: _CountingStore, settings: Settings
) -> None:
    """The second line of defence (the ticket's review finding): nothing here calls `clear()`,
    the persona has no committed snapshot (`snapshot_identity` alone would never change), and the
    cache still misses because `persona_fingerprint`'s attributed-node count moved."""
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    calls_after_first = entity_graph_store.entity_chunk_pairs_calls

    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls == calls_after_first  # nothing moved: a hit

    entity_graph_store.set_entity_attributes("test-pm", "concept:onboarding", {"sector": "core"})
    build_network_cached(entity_graph_store, "entities", "test-pm", settings=settings)
    assert entity_graph_store.entity_chunk_pairs_calls > calls_after_first  # attributed_nodes moved
