"""Optional Claude-powered entity/relationship extraction (``graphrag enrich``).

The Anthropic client sits behind ``ExtractionClient`` so the pipeline is unit-tested with a fake
and never needs a key or network. The real client uses structured outputs so the response is a
validated ``ExtractionResult``; server-side refusal fallbacks are enabled by default.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from graphrag.graph.store import GraphStore
from graphrag.models import (
    Chunk,
    Document,
    Enrichment,
    Entity,
    EntityType,
    Mention,
    PersonaSpec,
    Relation,
)

SYSTEM_PROMPT = """You build a knowledge graph from source text for a "{persona}" persona.
Extract the specific people, companies, products, frameworks, named concepts, books, metrics and
regulations that the text actually discusses (not passing mentions), and the relationships the
text states between them. Use canonical names (e.g. "Jobs to Be Done", not "JTBD framework").
Keep descriptions to one sentence grounded in the text. Evidence must be a short paraphrase."""


class ExtractedEntity(BaseModel):
    name: str
    type: EntityType = "concept"
    description: str = ""


class ExtractedRelation(BaseModel):
    source: str
    target: str
    type: str = Field(description="short verb phrase in UPPER_SNAKE_CASE, e.g. CREATED, WORKS_AT")
    evidence: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class DocumentExtraction(ExtractionResult):
    """One document's extraction as produced by an agent (``graphrag enrich-import``)."""

    doc_id: str


class ExtractionClient(Protocol):
    def extract(self, *, persona_name: str, document_title: str, text: str) -> ExtractionResult: ...


class AnthropicExtractionClient:
    """Structured-output extraction with the Anthropic SDK (model from GRAPHRAG_ENRICH_MODEL)."""

    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self._model = model

    def extract(self, *, persona_name: str, document_title: str, text: str) -> ExtractionResult:
        response = self._client.beta.messages.parse(
            model=self._model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT.format(persona=persona_name),
            messages=[
                {
                    "role": "user",
                    "content": f'Source: "{document_title}"\n\n{text}',
                }
            ],
            output_format=ExtractionResult,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return ExtractionResult()
        return response.parsed_output


def windows(chunks: Sequence[Chunk], max_chars: int) -> list[list[Chunk]]:
    """Group consecutive chunks into windows of at most ``max_chars`` characters."""
    out: list[list[Chunk]] = []
    current: list[Chunk] = []
    size = 0
    for chunk in chunks:
        n = len(chunk.text)
        if current and size + n > max_chars:
            out.append(current)
            current, size = [], 0
        current.append(chunk)
        size += n
    if current:
        out.append(current)
    return out


MIN_TOKEN_LEN = 4


def match_chunks(name: str, window: Sequence[Chunk]) -> tuple[list[str], str]:
    """Chunks that mention ``name``. Returns ``(chunk_ids, tier)`` with tier ``exact`` (verbatim,
    case-insensitive), ``loose`` (every significant token of the name occurs in the chunk, e.g.
    "Lenny Rachitsky" vs a transcript that only says "Lenny") or ``none`` (first chunk)."""
    lowered = name.lower()
    exact = [c.id for c in window if lowered in c.text.lower()]
    if exact:
        return exact, "exact"
    tokens = [
        t
        for t in "".join(ch if ch.isalnum() else " " for ch in lowered).split()
        if len(t) >= MIN_TOKEN_LEN
    ]
    if tokens:
        loose = [c.id for c in window if any(t in c.text.lower() for t in tokens)]
        if loose:
            return loose, "loose"
    return [window[0].id], "none"


def result_to_enrichment(result: ExtractionResult, window: Sequence[Chunk]) -> Enrichment:
    """Map model output onto graph objects; a mention is attached to every chunk naming the entity
    (see ``match_chunks``; the first chunk is the fallback so nothing is lost)."""
    entities: dict[str, Entity] = {}
    name_to_id: dict[str, str] = {}
    mentions: list[Mention] = []
    for item in result.entities:
        name = item.name.strip()
        if not name:
            continue
        entity_id = Entity.make_id(name, item.type)
        entities.setdefault(
            entity_id,
            Entity(id=entity_id, name=name, type=item.type, description=item.description.strip()),
        )
        name_to_id[name.lower()] = entity_id
        hits, _tier = match_chunks(name, window)
        mentions.extend(Mention(chunk_id=cid, entity_id=entity_id) for cid in hits)

    relations: list[Relation] = []
    for rel in result.relations:
        src = name_to_id.get(rel.source.strip().lower())
        dst = name_to_id.get(rel.target.strip().lower())
        if not src or not dst or src == dst:
            continue
        rel_type = "".join(ch if ch.isalnum() else "_" for ch in rel.type.strip().upper()).strip(
            "_"
        )
        anchor = next(
            (c.id for c in window if rel.evidence and rel.evidence[:40].lower() in c.text.lower()),
            window[0].id,
        )
        relations.append(
            Relation(
                source_id=src,
                target_id=dst,
                type=rel_type or "RELATED_TO",
                evidence=rel.evidence.strip(),
                chunk_id=anchor,
            )
        )
    return Enrichment(entities=list(entities.values()), mentions=mentions, relations=relations)


def enrich_documents(
    store: GraphStore,
    client: ExtractionClient,
    persona: PersonaSpec,
    documents: Iterable[Document],
    *,
    max_chars: int = 6000,
    on_progress: Callable[[str], None] | None = None,
) -> Enrichment:
    """Run extraction over each document's chunks and write the result to the store."""
    total = Enrichment()
    for doc in documents:
        chunks = store.document_chunks(doc.id, 0, 100_000)
        for window in windows(chunks, max_chars):
            text = "\n\n".join(c.text for c in window)
            result = client.extract(persona_name=persona.name, document_title=doc.title, text=text)
            part = result_to_enrichment(result, window)
            store.upsert_enrichment(part)
            total.entities.extend(part.entities)
            total.mentions.extend(part.mentions)
            total.relations.extend(part.relations)
        if on_progress:
            on_progress(f"enriched {doc.title}")
    return total
