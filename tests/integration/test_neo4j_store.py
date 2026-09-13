from pathlib import Path

import pytest

from graphrag.embed.hashing import HashEmbedder
from graphrag.graph import snapshot as snap
from graphrag.graph.neo4j_store import Neo4jGraphStore
from graphrag.models import Enrichment, Entity, Mention, PersonaSpec, Relation, SourceSpec
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


def test_attribution_writes_speakers_that_survive_a_snapshot(
    clean_store: Neo4jGraphStore,
    sample_corpus: Path,
    thread_source: SourceSpec,
    hash_embedder: HashEmbedder,
    tmp_path: Path,
) -> None:
    """The other half of ``tests/unit/test_pipeline_and_store.py``'s attach_speaker contract.

    A thread loaded as prose reaches the graph with no ``Speaker`` node at all, so everything
    asserted here is what ``graphrag attribution-import`` put there.
    """
    persona = PersonaSpec(id="it-threads", name="IT Threads", sources=[thread_source])
    report = IngestPipeline(clean_store, hash_embedder).ingest(
        sample_corpus, persona, thread_source
    )
    assert report.documents == 1 and report.chunks > 1
    doc_id = next(iter(clean_store.document_ids("it-threads", "threads")))
    assert clean_store.attributed_document_ids("it-threads", "threads") == set()

    chunks = clean_store.document_chunks(doc_id, 0, 100)
    clean_store.attach_speaker(doc_id, chunks[0].id, "quill-maker")
    clean_store.attach_speaker(doc_id, chunks[-1].id, "ledger-ann")
    clean_store.attach_speaker(doc_id, chunks[0].id, "quill-maker")  # idempotent

    # same contract the in-memory store is held to in tests/unit/test_pipeline_and_store.py
    assert clean_store.attributed_document_ids("it-threads", "threads") == {doc_id}
    assert clean_store.attributed_document_ids("it-threads", "absent") == set()
    assert clean_store.attributed_document_ids("other-persona", "threads") == set()
    document = clean_store.get_document(doc_id)
    assert document is not None and document.speakers == ["quill-maker", "ledger-ann"]
    assert clean_store.document_chunks(doc_id, 0, 1)[0].speakers == ["quill-maker"]
    assert {s.speaker for s in clean_store.list_speakers("it-threads")} == {
        "quill-maker",
        "ledger-ann",
    }
    spoke = clean_store.run_readonly_cypher(
        "MATCH (:Speaker)-[r:SPOKE]->(c:Chunk {doc_id: $doc}) RETURN count(r) AS n", {"doc": doc_id}
    )
    assert spoke[0]["n"] == 2  # one per (speaker, passage), not one per call

    # a passage that is not this document's matches nothing, so nothing is written
    clean_store.attach_speaker(doc_id, "no-such-chunk", "ghost")
    ghost = clean_store.run_readonly_cypher(
        "MATCH (s:Speaker {name: 'ghost'}) RETURN count(s) AS n"
    )
    assert ghost[0]["n"] == 0

    # the speakers ride the snapshot, because they are written to the `speakers` properties too
    root = tmp_path / "snapshots"
    snap.export_snapshot(
        clean_store, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    clean_store.delete_persona("it-threads")
    snap.load_snapshot(
        clean_store, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    assert clean_store.attributed_document_ids("it-threads", "threads") == {doc_id}
    assert clean_store.document_chunks(doc_id, 0, 1)[0].speakers == ["quill-maker"]


def test_network_reads_and_mean_embeddings_on_neo4j(
    clean_store: Neo4jGraphStore,
    sample_corpus: Path,
    it_persona: PersonaSpec,
    hash_embedder: HashEmbedder,
) -> None:
    """The other half of ``tests/unit/test_pipeline_and_store.py``'s network-read contract.

    Everything ``graphrag sna`` builds comes through these four methods, so the Cypher behind
    them has to agree with the in-memory store rather than merely return something.
    """
    import numpy as np

    from graphrag.sna.export import entity_co_mention, speaker_co_participation

    IngestPipeline(clean_store, hash_embedder).ingest(
        sample_corpus, it_persona, it_persona.sources[0]
    )

    pairs = clean_store.speaker_document_pairs("it-pm")
    assert [(p.speaker, p.doc_id) for p in pairs] == sorted((p.speaker, p.doc_id) for p in pairs)
    host = [p for p in pairs if p.speaker == "Lenny Rachitsky"]
    assert len(host) == 3 and all(p.chunks >= 1 for p in host)
    assert clean_store.speaker_document_pairs("it-pm", "test-podcast") == pairs
    assert clean_store.speaker_document_pairs("it-pm", "absent") == []
    assert clean_store.speaker_document_pairs("other-persona") == []

    graph = speaker_co_participation(clean_store, "it-pm")
    assert graph.number_of_nodes() == 4
    assert graph.nodes["Lenny Rachitsky"]["documents"] == 3
    assert graph.degree("Lenny Rachitsky") == 3

    doc = clean_store.list_documents("it-pm", speaker="Ada North")[0]
    chunk_id = clean_store.document_chunks(doc.id, 0, 1)[0].id
    clean_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="metric:retention", name="Retention", type="metric"),
                Entity(id="concept:onboarding", name="Onboarding", type="concept"),
            ],
            mentions=[
                Mention(chunk_id=chunk_id, entity_id="metric:retention"),
                Mention(chunk_id=chunk_id, entity_id="concept:onboarding"),
            ],
        )
    )
    mentions = clean_store.entity_chunk_pairs("it-pm")
    assert {m.entity_id for m in mentions} == {"metric:retention", "concept:onboarding"}
    assert all(m.chunk_id == chunk_id and m.doc_id == doc.id for m in mentions)
    assert [m.entity_id for m in clean_store.entity_chunk_pairs("it-pm", types=["metric"])] == [
        "metric:retention"
    ]
    assert clean_store.entity_chunk_pairs("it-pm", "absent") == []
    assert clean_store.entity_chunk_pairs("other-persona") == []
    co_mention = entity_co_mention(clean_store, "it-pm", min_weight=1)
    assert co_mention["metric:retention"]["concept:onboarding"]["weight"] == 1

    edges = clean_store.topic_edges("it-pm", min_weight=1)
    assert edges and all(e.source < e.target and e.weight >= 1 for e in edges)
    assert clean_store.topic_edges("it-pm", min_weight=99) == []
    assert clean_store.topic_edges("other-persona") == []

    doc_ids, matrix = clean_store.mean_embeddings("it-pm")
    assert doc_ids == sorted(clean_store.document_ids("it-pm"))
    assert matrix.shape == (3, hash_embedder.dim)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    keys, entity_matrix = clean_store.mean_embeddings("it-pm", level="entity")
    assert keys == ["concept:onboarding", "metric:retention"]
    assert entity_matrix.shape == (2, hash_embedder.dim)
    assert clean_store.mean_embeddings("other-persona")[0] == []
    with pytest.raises(ValueError, match="level must be"):
        clean_store.mean_embeddings("it-pm", level="chunk")


def test_dated_speaker_edges_annotations_and_aliases_on_neo4j(
    clean_store: Neo4jGraphStore,
    sample_corpus: Path,
    thread_source: SourceSpec,
    hash_embedder: HashEmbedder,
    tmp_path: Path,
) -> None:
    """The other half of the contract the in-memory store is held to in the unit tests.

    Covers ``tests/unit/test_pipeline_and_store.py``'s dated-speaker and annotation assertions,
    ``tests/unit/test_aliases.py``'s merge assertions, and the snapshot round trip for all three.
    """
    persona = PersonaSpec(id="it-layers", name="IT Layers", sources=[thread_source])
    IngestPipeline(clean_store, hash_embedder).ingest(sample_corpus, persona, thread_source)
    doc_id = next(iter(clean_store.document_ids("it-layers", "threads")))
    chunks = clean_store.document_chunks(doc_id, 0, 100)

    # --- dated speaker edges
    clean_store.attach_speaker(
        doc_id, chunks[0].id, "quill-maker", posted_at="2025-02-03", role="op", score=12
    )
    clean_store.attach_speaker(
        doc_id, chunks[-1].id, "ledger-ann", posted_at="2025-02-06", role="reply"
    )
    # a second call for the same speaker and passage replaces the record rather than adding one
    clean_store.attach_speaker(doc_id, chunks[0].id, "quill-maker", posted_at="2025-02-03")

    posts = clean_store.document_chunks(doc_id, 0, 1)[0].speaker_posts
    assert [(p.speaker, p.posted_at, p.role, p.score) for p in posts] == [
        ("quill-maker", "2025-02-03", None, None)
    ]
    edges = clean_store.run_readonly_cypher(
        "MATCH (:Speaker)-[r:SPOKE]->(c:Chunk {doc_id: $doc}) "
        "RETURN r.posted_at AS posted_at, r.role AS role ORDER BY posted_at",
        {"doc": doc_id},
    )
    assert [(e["posted_at"], e["role"]) for e in edges] == [
        ("2025-02-03", None),
        ("2025-02-06", "reply"),
    ]
    everyone = clean_store.speaker_document_pairs("it-layers")
    assert [p.speaker for p in everyone] == ["ledger-ann", "quill-maker"]
    early = clean_store.speaker_document_pairs("it-layers", until="2025-02-04")
    assert [(p.speaker, p.chunks) for p in early] == [("quill-maker", 1)]
    assert [
        p.speaker for p in clean_store.speaker_document_pairs("it-layers", since="2025-02-04")
    ] == ["ledger-ann"]
    assert clean_store.speaker_document_pairs("it-layers", since="2030-01-01") == []

    # --- annotations
    clean_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id="concept:handbook", name="Handbook", type="concept"),
                Entity(id="concept:hand-book", name="hand book", type="concept"),
            ],
            mentions=[
                Mention(chunk_id=chunks[0].id, entity_id="concept:handbook"),
                Mention(chunk_id=chunks[-1].id, entity_id="concept:hand-book"),
            ],
        )
    )
    assert clean_store.annotated_document_ids("it-layers", "threads") == set()
    clean_store.annotate_mention(doc_id, chunks[0].id, "Handbook", "complaint")
    clean_store.annotate_chunk(doc_id, chunks[0].id, ["handover", "access"])
    clean_store.annotate_chunk(doc_id, chunks[0].id, ["handover"])  # idempotent
    clean_store.annotate_mention(doc_id, chunks[0].id, "Nobody", "praise")  # no such mention

    assert clean_store.annotated_document_ids("it-layers", "threads") == {doc_id}
    assert clean_store.annotated_document_ids("it-layers", "absent") == set()
    assert clean_store.annotated_document_ids("other-persona", "threads") == set()
    stance = clean_store.mention_stances("it-layers")[0]
    assert (stance.name, stance.stance, stance.doc_id) == ("Handbook", "complaint", doc_id)
    assert stance.speakers == ["quill-maker"]
    assert len(clean_store.mention_stances("it-layers")) == 1
    assert clean_store.chunk_facets("it-layers")[0].facets == ["handover", "access"]
    assert clean_store.chunk_facets("it-layers", "absent") == []
    assert clean_store.mention_stances("other-persona") == []

    rows = clean_store.entity_mention_rows("it-layers")
    assert [(r.entity_id, r.chunk_id) for r in rows] == sorted(
        (r.entity_id, r.chunk_id) for r in rows
    )
    annotated = [r for r in rows if r.stance is not None]
    assert [(r.name, r.stance, r.speakers) for r in annotated] == [
        ("Handbook", "complaint", ["quill-maker"])
    ]
    # The unannotated mention is still a row, carrying no stance rather than a neutral one.
    assert [r.stance for r in rows if r.stance is None] == [None]
    assert [r.type for r in rows] == ["concept", "concept"]
    assert clean_store.entity_mention_rows("it-layers", types=["person"]) == []
    assert clean_store.entity_mention_rows("it-layers", "absent") == []
    assert clean_store.entity_mention_rows("other-persona") == []

    # --- the entity index the annotation fallback resolves a name against
    index = clean_store.persona_entities("it-layers")
    assert [e.id for e in index] == ["concept:hand-book", "concept:handbook"]
    assert clean_store.persona_entities("other-persona") == []
    clean_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id="concept:handbook", name="Handbook", type="concept")],
            mentions=[Mention(chunk_id=chunks[-1].id, entity_id="concept:handbook")],
        )
    )
    assert "concept:handbook" in {e.id for e in clean_store.entities_for_chunks([chunks[-1].id])}

    # --- aliases
    moved = clean_store.merge_entities("it-layers", "Handbook", ["hand book", "HANDBOOK"])

    assert moved == 1
    canonical = clean_store.entities_for_chunks([chunks[-1].id])[0]
    assert (canonical.id, canonical.name) == ("concept:handbook", "Handbook")
    assert canonical.aliases == ["hand book"]
    gone = clean_store.run_readonly_cypher(
        "MATCH (e:Entity {id: 'concept:hand-book'}) RETURN count(e) AS n"
    )
    assert gone[0]["n"] == 0
    assert clean_store.merge_entities("it-layers", "Handbook", ["hand book"]) == 0  # idempotent

    # --- all three ride the snapshot
    root = tmp_path / "snapshots"
    snap.export_snapshot(
        clean_store, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )
    clean_store.delete_persona("it-layers")
    snap.load_snapshot(
        clean_store, persona, root, embedding_model="hash-test", embedding_dim=hash_embedder.dim
    )

    loaded = clean_store.document_chunks(doc_id, 0, 1)[0]
    assert loaded.facets == ["handover", "access"]
    assert [(p.speaker, p.posted_at) for p in loaded.speaker_posts] == [
        ("quill-maker", "2025-02-03")
    ]
    assert [
        p.speaker for p in clean_store.speaker_document_pairs("it-layers", since="2025-02-01")
    ] == ["ledger-ann", "quill-maker"]
    assert clean_store.mention_stances("it-layers")[0].stance == "complaint"
    assert clean_store.entities_for_chunks([chunks[0].id])[0].aliases == ["hand book"]
