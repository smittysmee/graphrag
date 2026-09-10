"""Idempotent Neo4j schema (constraints + indexes). Each statement runs on its own: Bolt does not
accept ``;``-separated batches in one call."""

from __future__ import annotations

VECTOR_INDEX = "chunk_embedding"
FULLTEXT_INDEX = "chunk_text"
DOCUMENT_FULLTEXT_INDEX = "document_text"


def schema_statements(dim: int) -> list[str]:
    if dim <= 0:
        msg = "embedding dim must be positive"
        raise ValueError(msg)
    return [
        "CREATE CONSTRAINT persona_id IF NOT EXISTS FOR (p:Persona) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT source_key IF NOT EXISTS FOR (s:Source) REQUIRE s.key IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT speaker_name IF NOT EXISTS FOR (s:Speaker) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE INDEX chunk_persona IF NOT EXISTS FOR (c:Chunk) ON (c.persona_id)",
        "CREATE INDEX chunk_doc_ordinal IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.ordinal)",
        "CREATE INDEX document_persona IF NOT EXISTS FOR (d:Document) ON (d.persona_id)",
        (
            f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: "
            f"{int(dim)}, `vector.similarity_function`: 'cosine'}}}}"
        ),
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]",
        (
            f"CREATE FULLTEXT INDEX {DOCUMENT_FULLTEXT_INDEX} IF NOT EXISTS "
            "FOR (d:Document) ON EACH [d.title, d.description]"
        ),
    ]


LUCENE_SPECIALS = set('+-&|!(){}[]^"~*?:\\/')


def lucene_escape(query: str) -> str:
    """Escape Lucene operators so user text is searched literally (terms OR-ed by default)."""
    out: list[str] = []
    for ch in query:
        if ch in LUCENE_SPECIALS:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out).strip()
