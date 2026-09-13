"""Domain models shared by ingestion, storage, retrieval, the CLI and the MCP server."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from graphrag.textutil import slugify

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

LoaderName = Literal["transcripts", "documents"]
SearchMode = Literal["hybrid", "vector", "fulltext"]
Scalar = str | int | float | bool | None


# ----------------------------------------------------------------------------- personas


class SourceSpec(BaseModel):
    """Where a persona's grounding text comes from."""

    id: str
    kind: Literal["git", "local"] = "local"
    url: str | None = None
    path: str | None = None
    loader: LoaderName = "documents"
    glob: str = "**/*.md"
    description: str = ""

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not SLUG_RE.match(value):
            msg = f"source id must be a slug, got {value!r}"
            raise ValueError(msg)
        return value


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    expand_neighbors: int = Field(default=1, ge=0, le=3)
    mode: SearchMode = "hybrid"


class PersonaSpec(BaseModel):
    """A persona = a role prompt + the sources that ground it + when in the SDLC it applies."""

    id: str
    name: str
    description: str = ""
    role_prompt: str = ""
    voice: list[str] = Field(default_factory=list)
    sdlc_stages: list[str] = Field(default_factory=list)
    sources: list[SourceSpec] = Field(default_factory=list)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tags: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not SLUG_RE.match(value):
            msg = f"persona id must be a slug, got {value!r}"
            raise ValueError(msg)
        return value


# ----------------------------------------------------------------------------- corpus


class Turn(BaseModel):
    """One unit of parsed input text: a speaker turn in a transcript or a paragraph in a doc."""

    text: str
    speaker: str | None = None
    start_ts: str | None = None
    start_seconds: int | None = None


class Document(BaseModel):
    id: str
    persona_id: str
    source_id: str
    title: str
    path: str
    url: str | None = None
    published: date | None = None
    description: str = ""
    speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    metadata: dict[str, Scalar] = Field(default_factory=dict)
    word_count: int = 0


class Chunk(BaseModel):
    id: str
    doc_id: str
    persona_id: str
    ordinal: int
    text: str
    speaker: str | None = None
    speakers: list[str] = Field(default_factory=list)
    start_ts: str | None = None
    start_seconds: int | None = None
    url: str | None = None
    word_count: int = 0


class LoadedDocument(BaseModel):
    """A document plus its parsed turns, before chunking."""

    document: Document
    turns: list[Turn]


# ----------------------------------------------------------------------------- enrichment

EntityType = Literal[
    "person", "company", "product", "framework", "concept", "book", "metric", "regulation", "other"
]


class Entity(BaseModel):
    id: str
    name: str
    type: EntityType = "concept"
    description: str = ""

    @staticmethod
    def make_id(name: str, entity_type: str) -> str:
        return f"{entity_type}:{slugify(name)}"


class Mention(BaseModel):
    chunk_id: str
    entity_id: str


class Relation(BaseModel):
    source_id: str
    target_id: str
    type: str
    evidence: str = ""
    chunk_id: str


class Enrichment(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ----------------------------------------------------------------------------- retrieval


class ScoredChunk(BaseModel):
    chunk_id: str
    score: float


class ChunkHit(BaseModel):
    chunk: Chunk
    document: Document
    score: float
    methods: list[str] = Field(default_factory=list)
    neighbors: list[Chunk] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)

    def citation(self) -> str:
        who = f"{self.chunk.speaker} in " if self.chunk.speaker else ""
        when = f" @ {self.chunk.start_ts}" if self.chunk.start_ts else ""
        link = (
            f" <{self.chunk.url or self.document.url}>"
            if (self.chunk.url or self.document.url)
            else ""
        )
        return f'{who}"{self.document.title}"{when}{link}'

    def passage(self) -> str:
        parts = [n.text for n in self.neighbors if n.ordinal < self.chunk.ordinal]
        parts.append(self.chunk.text)
        parts.extend(n.text for n in self.neighbors if n.ordinal > self.chunk.ordinal)
        return "\n\n".join(parts)


class TopicCount(BaseModel):
    topic: str
    count: int


class RelatedTopic(BaseModel):
    topic: str
    weight: int


class SpeakerCount(BaseModel):
    speaker: str
    documents: int


class ContextPack(BaseModel):
    """Everything an agent needs to answer a question *as* a persona, with citations."""

    query: str
    persona_id: str | None
    persona_name: str | None = None
    role_prompt: str | None = None
    hits: list[ChunkHit] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)

    def to_prompt(self) -> str:
        lines: list[str] = []
        if self.persona_name:
            lines.append(f"# Persona: {self.persona_name}")
        if self.role_prompt:
            lines.append(self.role_prompt.strip())
            lines.append("")
        lines.append(f"# Grounding for: {self.query}")
        lines.append(
            "Answer using the sources below. Cite them inline as [n]. If the sources do not cover"
            " the question, say so rather than inventing detail."
        )
        lines.append("")
        for i, hit in enumerate(self.hits, start=1):
            lines.append(f"## [{i}] {hit.citation()}")
            lines.append(hit.passage().strip())
            lines.append("")
        if self.topics:
            lines.append("Related topics: " + ", ".join(self.topics))
        if self.entities:
            lines.append(
                "Named entities: " + ", ".join(f"{e.name} ({e.type})" for e in self.entities)
            )
        return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------------------- stats / snapshot


class GraphStats(BaseModel):
    personas: int = 0
    sources: int = 0
    documents: int = 0
    chunks: int = 0
    speakers: int = 0
    topics: int = 0
    entities: int = 0
    per_persona: dict[str, dict[str, int]] = Field(default_factory=dict)


class SnapshotManifest(BaseModel):
    version: int = 1
    persona_id: str
    created_at: datetime
    embedding_model: str
    embedding_dim: int
    document_count: int
    chunk_count: int
    entity_count: int = 0
    sources: list[SourceSpec] = Field(default_factory=list)
    source_commit: str | None = None
    tool_version: str = ""
