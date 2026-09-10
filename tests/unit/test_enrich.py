from graphrag.extract.llm import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    enrich_documents,
    result_to_enrichment,
    windows,
)
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk, PersonaSpec
from graphrag.pipeline import IngestReport
from graphrag.retrieve.search import Retriever


class FakeExtractionClient:
    """Returns entities that appear in the text; records every call."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, *, persona_name: str, document_title: str, text: str) -> ExtractionResult:
        self.calls.append(document_title)
        entities = []
        if "retention" in text.lower():
            entities.append(
                ExtractedEntity(name="Retention", type="metric", description="Users who stay.")
            )
        if "onboarding" in text.lower():
            entities.append(ExtractedEntity(name="Onboarding", type="concept"))
        relations = []
        if len(entities) == 2:
            relations.append(
                ExtractedRelation(
                    source="Onboarding",
                    target="Retention",
                    type="improves",
                    evidence="Onboarding is the lever that moves early retention",
                )
            )
        return ExtractionResult(entities=entities, relations=relations)


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"d#{i}", doc_id="d", persona_id="p", ordinal=i, text=text)


def test_windows_respect_max_chars() -> None:
    chunks = [_chunk(i, "x" * 100) for i in range(5)]
    assert [len(w) for w in windows(chunks, 250)] == [2, 2, 1]


def test_result_to_enrichment_maps_names_and_anchors() -> None:
    window = [_chunk(0, "Retention matters."), _chunk(1, "Onboarding drives retention.")]
    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Retention", type="metric"),
            ExtractedEntity(name="Onboarding"),
            ExtractedEntity(name="  "),
        ],
        relations=[
            ExtractedRelation(source="onboarding", target="Retention", type="drives up!"),
            ExtractedRelation(source="Nope", target="Retention", type="x"),
        ],
    )
    enr = result_to_enrichment(result, window)
    assert {e.id for e in enr.entities} == {"metric:retention", "concept:onboarding"}
    assert {(m.chunk_id, m.entity_id) for m in enr.mentions} == {
        ("d#0", "metric:retention"),
        ("d#1", "metric:retention"),
        ("d#1", "concept:onboarding"),
    }
    assert len(enr.relations) == 1 and enr.relations[0].type == "DRIVES_UP"


def test_enrich_documents_writes_store_and_surfaces_in_context(
    ingested: IngestReport,
    memory_store: InMemoryGraphStore,
    persona: PersonaSpec,
    retriever: Retriever,
) -> None:
    client = FakeExtractionClient()
    docs = list(memory_store.iter_documents(persona.id))
    total = enrich_documents(memory_store, client, persona, docs[:2], max_chars=2000)
    assert len(client.calls) >= 2
    assert memory_store.stats().entities >= 1
    assert memory_store.enriched_doc_ids(persona.id)
    assert total.mentions
    pack = retriever.context("onboarding and retention", persona=persona)
    assert any(e.name == "Retention" for e in pack.entities)
    assert "Named entities" in pack.to_prompt()
    exported = memory_store.enrichment_for_persona(persona.id)
    assert {e.id for e in exported.entities} >= {"metric:retention"}


def test_match_chunks_tiers() -> None:
    from graphrag.extract.llm import match_chunks

    window = [_chunk(0, "Lenny asks about pricing."), _chunk(1, "Retention matters more.")]
    assert match_chunks("Retention", window) == (["d#1"], "exact")
    assert match_chunks("Lenny Rachitsky", window) == (["d#0"], "loose")
    assert match_chunks("Growth loops", window) == (["d#0"], "none")
