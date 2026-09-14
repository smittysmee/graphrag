import json
from pathlib import Path

from graphrag.extract.llm import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    enrich_documents,
    result_to_enrichment,
    windows,
)
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk, Enrichment, Entity, PersonaSpec
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


def test_import_reports_a_name_whose_id_another_entity_already_holds(
    ingested: IngestReport, memory_store: InMemoryGraphStore, tmp_path: Path
) -> None:
    """One `collision:` line per name the graph declined to rename, per file.

    The import still succeeds: the mentions are keyed on the id and they landed. What the line
    says is that a reviewer now has to choose -- a name whose id differs, or an entry in the
    persona's ``aliases.yaml`` declaring the two spellings one thing.
    """
    from graphrag.extract.importer import import_extraction_file

    doc = memory_store.list_documents("test-pm", speaker="Ada North")[0]
    memory_store.upsert_enrichment(
        Enrichment(entities=[Entity(id="product:lumenta", name="Lumenta", type="product")])
    )
    file = tmp_path / "ada.json"
    file.write_text(
        json.dumps(
            {
                "doc_id": doc.id,
                "entities": [
                    {"name": "Lumenta", "type": "product"},
                    # An older file that spelled the name without the punctuation carrying it.
                    {"name": "lumenta ", "type": "product"},
                ],
                "relations": [],
            }
        )
    )

    clean = import_extraction_file(memory_store, file)
    assert clean.ok and clean.collisions == ()

    # Punctuation the id rule does not spell out still folds away, which is exactly the case
    # the store's guard is the second line of defence for.
    rival = tmp_path / "rival.json"
    rival.write_text(
        json.dumps(
            {
                "doc_id": doc.id,
                "entities": [{"name": "Lumenta!", "type": "product"}],
                "relations": [],
            }
        )
    )
    assert Entity.make_id("Lumenta!", "product") == "product:lumenta"

    dry = import_extraction_file(memory_store, rival, dry_run=True)
    assert dry.ok and dry.collisions == ()  # a dry run writes nothing, so nothing collided

    written = import_extraction_file(memory_store, rival)
    assert written.ok
    assert written.collisions == ("Lumenta! kept as Lumenta",)
    assert memory_store.entities["product:lumenta"].name == "Lumenta"
