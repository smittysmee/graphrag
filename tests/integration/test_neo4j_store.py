from pathlib import Path

import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.neo4j_store import Neo4jGraphStore
from graphrag.models import Enrichment, Entity, Mention, PersonaSpec, Relation
from graphrag.pipeline import IngestPipeline
from graphrag.retrieve.search import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture
def it_persona(persona: PersonaSpec) -> PersonaSpec:
    return persona.model_copy(update={"id": "it-pm"})


def test_end_to_end_on_neo4j(
    clean_store: Neo4jGraphStore,
    sample_corpus: Path,
    it_persona: PersonaSpec,
    hash_embedder: HashEmbedder,
    tmp_path: Path,
) -> None:
    report = IngestPipeline(clean_store, hash_embedder).ingest(
        sample_corpus, it_persona, it_persona.sources[0]
    )
    assert report.documents == 3

    stats = clean_store.stats()
    assert stats.per_persona["it-pm"] == {"documents": 3, "chunks": report.chunks}

    retriever = Retriever(clean_store, hash_embedder)
    hits = retriever.search("retention curve leaky bucket", persona_id="it-pm", k=3, expand=1)
    assert hits and "leaky bucket" in hits[0].chunk.text
    assert "vector" in hits[0].methods and "fulltext" in hits[0].methods

    assert clean_store.fulltext_search("roadmap (review)", 3, "it-pm")  # lucene specials escaped
    assert clean_store.related_topics("retention", "it-pm")
    ids = clean_store.document_ids("it-pm")
    assert len(ids) == 3
    assert clean_store.document_ids("it-pm", "test-podcast") == ids
    assert clean_store.document_ids("it-pm", "absent") == set()
    assert clean_store.enriched_document_ids("it-pm", "test-podcast") == set()  # nothing extracted
    doc = clean_store.list_documents("it-pm", speaker="Ada North")[0]
    assert clean_store.document_chunks(doc.id, 0, 1)[0].ordinal == 0
    assert clean_store.list_speakers("it-pm")[0].speaker == "Lenny Rachitsky"

    # enrichment round trip
    chunk_id = clean_store.document_chunks(doc.id, 0, 1)[0].id
    clean_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="metric:retention", name="Retention", type="metric"),
                Entity(id="concept:onboarding", name="Onboarding"),
            ],
            mentions=[Mention(chunk_id=chunk_id, entity_id="metric:retention")],
            relations=[
                Relation(
                    source_id="concept:onboarding",
                    target_id="metric:retention",
                    type="IMPROVES",
                    chunk_id=chunk_id,
                )
            ],
        )
    )
    assert [e.name for e in clean_store.entities_for_chunks([chunk_id])] == ["Retention"]
    assert clean_store.enriched_doc_ids("it-pm") == {doc.id}
    # same contract the in-memory store is held to in tests/unit/test_pipeline_and_store.py
    assert clean_store.enriched_document_ids("it-pm", "test-podcast") == {doc.id}
    assert clean_store.enriched_document_ids("it-pm", "absent") == set()
    assert clean_store.enriched_document_ids("other-persona", "test-podcast") == set()
    assert len(clean_store.enrichment_for_persona("it-pm").relations) == 1

    # snapshot export -> delete -> load
    root = tmp_path / "snapshots"
    manifest = snap.export_snapshot(
        clean_store, it_persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    assert manifest.chunk_count == report.chunks and manifest.entity_count == 2
    clean_store.delete_persona("it-pm")
    assert "it-pm" not in clean_store.stats().per_persona
    snap.load_snapshot(
        clean_store, it_persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    assert clean_store.stats().per_persona["it-pm"]["chunks"] == report.chunks
    assert retriever.search("retention curve", persona_id="it-pm", k=1)

    # read-only cypher guard
    rows = clean_store.run_readonly_cypher(
        "MATCH (d:Document {persona_id: $pid}) RETURN count(d) AS n", {"pid": "it-pm"}
    )
    assert rows[0]["n"] == 3
    with pytest.raises(PermissionError):
        clean_store.run_readonly_cypher("MATCH (n) DETACH DELETE n")
