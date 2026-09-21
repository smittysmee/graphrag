from pathlib import Path

import numpy as np
import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import PersonaSpec
from graphrag.pipeline import IngestReport


def test_export_then_load_into_fresh_store(
    tmp_path: Path,
    ingested: IngestReport,
    memory_store: InMemoryGraphStore,
    persona: PersonaSpec,
    hash_embedder: HashEmbedder,
) -> None:
    root = tmp_path / "snapshots"
    manifest = snap.export_snapshot(
        memory_store,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=hash_embedder.dim,
        source_commit="abc",
    )
    assert manifest.chunk_count == ingested.chunks
    assert (root / persona.id / "embeddings.npy").exists()
    assert snap.list_snapshots(root)[0].persona_id == persona.id

    fresh = InMemoryGraphStore()
    progress: list[str] = []
    loaded = snap.load_snapshot(
        fresh,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=hash_embedder.dim,
        on_progress=progress.append,
    )
    assert loaded.document_count == 3
    assert fresh.stats().chunks == memory_store.stats().chunks
    assert progress[-1].endswith("loaded")
    q = hash_embedder.embed_query("leaky bucket")
    assert fresh.vector_search(q, 1)[0].chunk_id == memory_store.vector_search(q, 1)[0].chunk_id
    assert fresh.related_topics("retention")
    for cid, vec in memory_store.embeddings.items():
        assert np.allclose(vec, fresh.embeddings[cid])


def test_load_refuses_other_embedding_model(
    tmp_path: Path, ingested: IngestReport, memory_store: InMemoryGraphStore, persona: PersonaSpec
) -> None:
    root = tmp_path / "snapshots"
    snap.export_snapshot(memory_store, persona, root, embedding_model="hash-test", embedding_dim=64)
    with pytest.raises(snap.SnapshotError, match="GRAPHRAG_EMBEDDING_MODEL=hash-test"):
        snap.load_snapshot(
            InMemoryGraphStore(),
            persona,
            root,
            embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=384,
        )


def test_missing_manifest(tmp_path: Path, persona: PersonaSpec) -> None:
    with pytest.raises(snap.SnapshotError):
        snap.read_manifest(tmp_path / "nope")
    assert snap.list_snapshots(tmp_path / "nope") == []


def test_export_preserves_source_commit(
    tmp_path: Path, ingested: IngestReport, memory_store: InMemoryGraphStore, persona: PersonaSpec
) -> None:
    root = tmp_path / "snapshots"
    snap.export_snapshot(
        memory_store,
        persona,
        root,
        embedding_model="hash-test",
        embedding_dim=64,
        source_commit="abc",
    )
    again = snap.export_snapshot(
        memory_store, persona, root, embedding_model="hash-test", embedding_dim=64
    )
    assert again.source_commit == "abc"


def test_speaker_dates_stances_facets_and_attributes_ride_the_snapshot(
    tmp_path: Path,
    thread_document: str,
    memory_store: InMemoryGraphStore,
    docs_persona: PersonaSpec,
    hash_embedder: HashEmbedder,
) -> None:
    """The layers a teammate would otherwise have to rebuild after loading a snapshot.

    The Neo4j store is held to the same assertions in ``tests/integration/test_neo4j_store.py``.
    """
    from graphrag.models import Enrichment, Entity, Mention

    chunks = memory_store.document_chunks(thread_document, 0, 100)
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(
                    id="concept:handbook",
                    name="Handbook",
                    type="concept",
                    aliases=["hand book"],
                )
            ],
            mentions=[Mention(chunk_id=chunks[0].id, entity_id="concept:handbook")],
        )
    )
    memory_store.attach_speaker(
        thread_document, chunks[0].id, "quill-maker", posted_at="2025-02-03", role="op", score=12
    )
    memory_store.annotate_mention(thread_document, chunks[0].id, "Handbook", "complaint")
    memory_store.annotate_chunk(thread_document, chunks[0].id, ["handover"])
    memory_store.set_document_attributes(thread_document, {"region": "north"})
    memory_store.set_speaker_attributes("test-docs", "quill-maker", {"region": "south"})
    memory_store.set_entity_attributes("test-docs", "concept:handbook", {"region": "north"})

    root = tmp_path / "snapshots"
    snap.export_snapshot(
        memory_store,
        docs_persona,
        root,
        embedding_model="hash-test",
        embedding_dim=hash_embedder.dim,
    )
    fresh = InMemoryGraphStore()
    snap.load_snapshot(
        fresh, docs_persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )

    loaded = fresh.document_chunks(thread_document, 0, 1)[0]
    assert loaded.facets == ["handover"]
    assert loaded.speakers == ["quill-maker"]
    assert [(p.posted_at, p.role, p.score) for p in loaded.speaker_posts] == [
        ("2025-02-03", "op", 12)
    ]
    assert [p.speaker for p in fresh.speaker_document_pairs("test-docs", since="2025-02-01")] == [
        "quill-maker"
    ]
    stance = fresh.mention_stances("test-docs")[0]
    assert (stance.name, stance.stance) == ("Handbook", "complaint")
    assert fresh.entities["concept:handbook"].aliases == ["hand book"]
    # Node attributes ride too: the document's on the document, the speaker's and the entity's
    # in files of their own, because both of those nodes are shared between personas.
    assert fresh.document_attributes("test-docs") == {thread_document: {"region": "north"}}
    assert fresh.speaker_attributes("test-docs") == {"quill-maker": {"region": "south"}}
    assert fresh.entity_attributes("test-docs") == {"concept:handbook": {"region": "north"}}
    row = next(r for r in fresh.speaker_document_pairs("test-docs"))
    assert row.speaker_attributes == {"region": "south"}
    assert row.document_attributes == {"region": "north"}


def test_punctuated_entity_ids_survive_the_round_trip(
    tmp_path: Path,
    ingested: IngestReport,
    memory_store: InMemoryGraphStore,
    persona: PersonaSpec,
    hash_embedder: HashEmbedder,
) -> None:
    """A product and its "+" variant are two nodes in the snapshot as well as in the graph.

    Ids are carried verbatim, not re-derived on load, so a teammate who loads the snapshot gets
    the same two nodes rather than whichever one the slug would have collapsed them onto.
    """
    from graphrag.models import Enrichment, Entity, Mention

    docs = sorted(memory_store.document_ids(persona.id))
    first, second = (memory_store.document_chunks(d, 0, 1)[0].id for d in docs[:2])
    ids = [Entity.make_id("Lumenta", "product"), Entity.make_id("Lumenta+", "product")]
    assert ids == ["product:lumenta", "product:lumenta-plus"]
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id=ids[0], name="Lumenta", type="product"),
                Entity(id=ids[1], name="Lumenta+", type="product"),
            ],
            mentions=[
                Mention(chunk_id=first, entity_id=ids[0]),
                Mention(chunk_id=second, entity_id=ids[1]),
            ],
        )
    )

    root = tmp_path / "snapshots"
    snap.export_snapshot(
        memory_store, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    fresh = InMemoryGraphStore()
    snap.load_snapshot(
        fresh, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )

    loaded = fresh.enrichment_for_persona(persona.id)
    assert sorted(e.id for e in loaded.entities) == ids
    assert {(e.id, e.name) for e in loaded.entities} == {
        ("product:lumenta", "Lumenta"),
        ("product:lumenta-plus", "Lumenta+"),
    }
    assert len(loaded.mentions) == 2
