"""Neo4j Community implementation of ``GraphStore`` (driver 6.x, ``execute_query``)."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Sequence
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np
from neo4j import GraphDatabase, RoutingControl

from graphrag.config import Neo4jSettings
from graphrag.embed.base import Matrix, Vector
from graphrag.graph.schema import FULLTEXT_INDEX, VECTOR_INDEX, lucene_escape, schema_statements
from graphrag.models import (
    Chunk,
    Document,
    Enrichment,
    Entity,
    GraphStats,
    Mention,
    PersonaSpec,
    RelatedTopic,
    Relation,
    ScoredChunk,
    SpeakerCount,
    TopicCount,
)

BATCH = 500
# Server notifications ("label does not exist" on an empty graph, etc.) are noise for a CLI.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
WRITE_KEYWORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL\s*\{|LOAD\s+CSV|"
    r"apoc\.(create|merge|refactor|load|periodic|export|import|cypher\.run)|db\.create|"
    r"dbms\.|db\.index\.vector\.createNodeIndex)\b",
    re.IGNORECASE,
)


def _chunk_props(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "persona_id": chunk.persona_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "speaker": chunk.speaker,
        "speakers": chunk.speakers,
        "start_ts": chunk.start_ts,
        "start_seconds": chunk.start_seconds,
        "url": chunk.url,
        "word_count": chunk.word_count,
    }


def _chunk_from_node(node: dict[str, Any]) -> Chunk:
    return Chunk(
        id=node["id"],
        doc_id=node["doc_id"],
        persona_id=node["persona_id"],
        ordinal=int(node["ordinal"]),
        text=node["text"],
        speaker=node.get("speaker"),
        speakers=list(node.get("speakers") or []),
        start_ts=node.get("start_ts"),
        start_seconds=node.get("start_seconds"),
        url=node.get("url"),
        word_count=int(node.get("word_count") or 0),
    )


def _doc_props(doc: Document) -> dict[str, Any]:
    return {
        "persona_id": doc.persona_id,
        "source_id": doc.source_id,
        "title": doc.title,
        "path": doc.path,
        "url": doc.url,
        "published": doc.published.isoformat() if doc.published else None,
        "description": doc.description,
        "speakers": doc.speakers,
        "topics": doc.topics,
        "metadata_json": json.dumps(doc.metadata, sort_keys=True),
        "word_count": doc.word_count,
    }


def _doc_from_node(node: dict[str, Any]) -> Document:
    published = node.get("published")
    return Document(
        id=node["id"],
        persona_id=node["persona_id"],
        source_id=node["source_id"],
        title=node["title"],
        path=node.get("path") or "",
        url=node.get("url"),
        published=date.fromisoformat(published) if published else None,
        description=node.get("description") or "",
        speakers=list(node.get("speakers") or []),
        topics=list(node.get("topics") or []),
        metadata=json.loads(node.get("metadata_json") or "{}"),
        word_count=int(node.get("word_count") or 0),
    )


class Neo4jGraphStore:
    def __init__(self, settings: Neo4jSettings) -> None:
        self._driver = GraphDatabase.driver(
            settings.uri, auth=(settings.user, settings.password.get_secret_value())
        )
        self._db = settings.database

    # ------------------------------------------------------------- helpers
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        result = self._driver.execute_query(query, parameters_=params, database_=self._db)
        return [record.data() for record in result.records]

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        result = self._driver.execute_query(
            query, parameters_=params, database_=self._db, routing_=RoutingControl.READ
        )
        return [record.data() for record in result.records]

    def verify_connectivity(self) -> None:
        self._driver.verify_connectivity()

    # ------------------------------------------------------------- lifecycle
    def ensure_schema(self, dim: int) -> None:
        for statement in schema_statements(dim):
            self._run(statement)
        self._run("CALL db.awaitIndexes(300)")

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------- writes
    def upsert_persona(self, persona: PersonaSpec) -> None:
        self._run(
            """
            MERGE (p:Persona {id: $id})
            SET p.name = $name, p.description = $description, p.sdlc_stages = $stages,
                p.spec_json = $spec
            """,
            id=persona.id,
            name=persona.name,
            description=persona.description,
            stages=persona.sdlc_stages,
            spec=persona.model_dump_json(),
        )
        self._run(
            """
            MATCH (p:Persona {id: $persona_id})
            UNWIND $sources AS src
            MERGE (s:Source {key: $persona_id + ':' + src.id})
            SET s.id = src.id, s.url = src.url, s.kind = src.kind, s.loader = src.loader,
                s.description = src.description
            MERGE (p)-[:GROUNDED_BY]->(s)
            """,
            persona_id=persona.id,
            sources=[s.model_dump() for s in persona.sources],
        )

    def upsert_documents(self, documents: Sequence[Document]) -> None:
        for i in range(0, len(documents), BATCH):
            rows = [{"id": d.id, "props": _doc_props(d)} for d in documents[i : i + BATCH]]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (d:Document {id: row.id})
                SET d += row.props
                WITH d, row
                OPTIONAL MATCH (s:Source {key: row.props.persona_id + ':' + row.props.source_id})
                FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END | MERGE (s)-[:CONTAINS]->(d))
                WITH d, row
                UNWIND row.props.speakers AS name
                MERGE (sp:Speaker {name: name})
                MERGE (d)-[:FEATURES]->(sp)
                """,
                rows=rows,
            )
            self._run(
                """
                UNWIND $rows AS row
                MATCH (d:Document {id: row.id})
                UNWIND row.props.topics AS topic
                MERGE (t:Topic {name: topic})
                MERGE (d)-[:ABOUT]->(t)
                """,
                rows=rows,
            )

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Matrix) -> None:
        if len(chunks) != len(embeddings):
            msg = f"{len(chunks)} chunks but {len(embeddings)} embeddings"
            raise ValueError(msg)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vecs = embeddings[i : i + BATCH]
            rows = [
                {"id": c.id, "props": _chunk_props(c), "embedding": [float(x) for x in v]}
                for c, v in zip(batch, vecs, strict=True)
            ]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c += row.props
                WITH c, row
                CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
                WITH c, row
                MATCH (d:Document {id: row.props.doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                WITH c, row
                UNWIND row.props.speakers AS name
                MERGE (s:Speaker {name: name})
                MERGE (s)-[:SPOKE]->(c)
                """,
                rows=rows,
            )
        pairs = [
            [a.id, b.id]
            for a, b in pairwise(chunks)
            if a.doc_id == b.doc_id and b.ordinal == a.ordinal + 1
        ]
        for i in range(0, len(pairs), BATCH):
            self._run(
                """
                UNWIND $pairs AS pair
                MATCH (a:Chunk {id: pair[0]}), (b:Chunk {id: pair[1]})
                MERGE (a)-[:NEXT]->(b)
                """,
                pairs=pairs[i : i + BATCH],
            )

    def upsert_topic_cooccurrence(self, pairs: Sequence[tuple[str, str, int]]) -> None:
        rows = [{"a": a, "b": b, "w": w} for a, b, w in pairs]
        for i in range(0, len(rows), BATCH):
            self._run(
                """
                UNWIND $rows AS row
                MERGE (a:Topic {name: row.a})
                MERGE (b:Topic {name: row.b})
                MERGE (a)-[r:CO_OCCURS]-(b)
                SET r.weight = row.w
                """,
                rows=rows[i : i + BATCH],
            )

    def upsert_enrichment(self, enrichment: Enrichment) -> None:
        if enrichment.entities:
            self._run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                SET e.name = row.name, e.type = row.type,
                    e.description = CASE WHEN row.description <> '' THEN row.description
                                         ELSE coalesce(e.description, '') END
                """,
                rows=[e.model_dump() for e in enrichment.entities],
            )
        if enrichment.mentions:
            self._run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.chunk_id}), (e:Entity {id: row.entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                rows=[m.model_dump() for m in enrichment.mentions],
            )
        if enrichment.relations:
            self._run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {id: row.source_id}), (b:Entity {id: row.target_id})
                MERGE (a)-[r:RELATED_TO {type: row.type, chunk_id: row.chunk_id}]->(b)
                SET r.evidence = row.evidence
                """,
                rows=[r.model_dump() for r in enrichment.relations],
            )

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        self._run(
            """
            UNWIND $ids AS id
            MATCH (d:Document {id: id})
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE d, c
            """,
            ids=list(doc_ids),
        )

    def delete_persona(self, persona_id: str) -> None:
        # Batched deletes need an implicit (auto-commit) transaction; execute_query is explicit.
        with self._driver.session(database=self._db) as session:
            session.run(
                "MATCH (c:Chunk {persona_id: $pid}) "
                "CALL { WITH c DETACH DELETE c } IN TRANSACTIONS OF 1000 ROWS",
                pid=persona_id,
            ).consume()
        self._run("MATCH (d:Document {persona_id: $pid}) DETACH DELETE d", pid=persona_id)
        self._run(
            "MATCH (p:Persona {id: $pid}) OPTIONAL MATCH (p)-[:GROUNDED_BY]->(s:Source) "
            "DETACH DELETE p, s",
            pid=persona_id,
        )
        self._run("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")
        self._run("MATCH (s:Speaker) WHERE NOT (s)--() DELETE s")
        self._run("MATCH (t:Topic) WHERE NOT (t)<-[:ABOUT]-() DETACH DELETE t")

    # ------------------------------------------------------------- search
    def vector_search(
        self, vector: Vector, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]:
        candidates = k * 4 if persona_id else k
        rows = self._read(
            f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $candidates, $vector)
            YIELD node, score
            WHERE $persona_id IS NULL OR node.persona_id = $persona_id
            RETURN node.id AS chunk_id, score
            ORDER BY score DESC LIMIT $k
            """,
            candidates=candidates,
            vector=[float(x) for x in np.asarray(vector, dtype=np.float32)],
            persona_id=persona_id,
            k=k,
        )
        return [ScoredChunk(chunk_id=r["chunk_id"], score=float(r["score"])) for r in rows]

    def fulltext_search(
        self, query: str, k: int, persona_id: str | None = None
    ) -> list[ScoredChunk]:
        escaped = lucene_escape(query)
        if not escaped:
            return []
        rows = self._read(
            f"""
            CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $q, {{limit: $candidates}})
            YIELD node, score
            WHERE $persona_id IS NULL OR node.persona_id = $persona_id
            RETURN node.id AS chunk_id, score
            ORDER BY score DESC LIMIT $k
            """,
            q=escaped,
            candidates=k * 4 if persona_id else k,
            persona_id=persona_id,
            k=k,
        )
        return [ScoredChunk(chunk_id=r["chunk_id"], score=float(r["score"])) for r in rows]

    # ------------------------------------------------------------- reads
    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        rows = self._read(
            "MATCH (c:Chunk) WHERE c.id IN $ids RETURN c {.*, embedding: null} AS c",
            ids=list(chunk_ids),
        )
        by_id = {r["c"]["id"]: _chunk_from_node(r["c"]) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def neighbors(self, chunk_id: str, hops: int = 1) -> list[Chunk]:
        if hops <= 0:
            return []
        rows = self._read(
            """
            MATCH (c:Chunk {id: $id})
            MATCH (n:Chunk {doc_id: c.doc_id})
            WHERE n.id <> c.id AND abs(n.ordinal - c.ordinal) <= $hops
            RETURN n {.*, embedding: null} AS n ORDER BY n.ordinal
            """,
            id=chunk_id,
            hops=hops,
        )
        return [_chunk_from_node(r["n"]) for r in rows]

    def get_document(self, doc_id: str) -> Document | None:
        rows = self._read("MATCH (d:Document {id: $id}) RETURN d {.*} AS d", id=doc_id)
        return _doc_from_node(rows[0]["d"]) if rows else None

    def list_documents(
        self,
        persona_id: str | None = None,
        *,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 50,
    ) -> list[Document]:
        rows = self._read(
            """
            MATCH (d:Document)
            WHERE ($persona_id IS NULL OR d.persona_id = $persona_id)
              AND ($topic IS NULL OR $topic IN d.topics)
              AND ($speaker IS NULL OR $speaker IN d.speakers)
            RETURN d {.*} AS d
            ORDER BY d.published DESC, d.title
            LIMIT $limit
            """,
            persona_id=persona_id,
            topic=topic,
            speaker=speaker,
            limit=limit,
        )
        return [_doc_from_node(r["d"]) for r in rows]

    def document_chunks(self, doc_id: str, start: int = 0, count: int = 5) -> list[Chunk]:
        rows = self._read(
            """
            MATCH (c:Chunk {doc_id: $doc_id})
            WHERE c.ordinal >= $start AND c.ordinal < $end
            RETURN c {.*, embedding: null} AS c ORDER BY c.ordinal
            """,
            doc_id=doc_id,
            start=start,
            end=start + count,
        )
        return [_chunk_from_node(r["c"]) for r in rows]

    def list_topics(self, persona_id: str | None = None, limit: int = 50) -> list[TopicCount]:
        rows = self._read(
            """
            MATCH (d:Document)-[:ABOUT]->(t:Topic)
            WHERE $persona_id IS NULL OR d.persona_id = $persona_id
            RETURN t.name AS topic, count(d) AS count ORDER BY count DESC, topic LIMIT $limit
            """,
            persona_id=persona_id,
            limit=limit,
        )
        return [TopicCount(topic=r["topic"], count=int(r["count"])) for r in rows]

    def related_topics(
        self, topic: str, persona_id: str | None = None, limit: int = 10
    ) -> list[RelatedTopic]:
        rows = self._read(
            """
            MATCH (t:Topic {name: $topic})-[r:CO_OCCURS]-(o:Topic)
            WHERE $persona_id IS NULL
               OR EXISTS { MATCH (d:Document {persona_id: $persona_id})-[:ABOUT]->(o) }
            RETURN o.name AS topic, r.weight AS weight ORDER BY weight DESC, topic LIMIT $limit
            """,
            topic=topic,
            persona_id=persona_id,
            limit=limit,
        )
        return [RelatedTopic(topic=r["topic"], weight=int(r["weight"])) for r in rows]

    def list_speakers(self, persona_id: str | None = None, limit: int = 50) -> list[SpeakerCount]:
        rows = self._read(
            """
            MATCH (d:Document)-[:FEATURES]->(s:Speaker)
            WHERE $persona_id IS NULL OR d.persona_id = $persona_id
            RETURN s.name AS speaker, count(d) AS documents
            ORDER BY documents DESC, speaker LIMIT $limit
            """,
            persona_id=persona_id,
            limit=limit,
        )
        return [SpeakerCount(speaker=r["speaker"], documents=int(r["documents"])) for r in rows]

    def list_personas(self) -> list[PersonaSpec]:
        rows = self._read("MATCH (p:Persona) RETURN p.spec_json AS spec ORDER BY p.id")
        return [PersonaSpec.model_validate_json(r["spec"]) for r in rows]

    def entities_for_chunks(self, chunk_ids: Sequence[str]) -> list[Entity]:
        rows = self._read(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.id IN $ids
            RETURN DISTINCT e {.*} AS e ORDER BY e.name
            """,
            ids=list(chunk_ids),
        )
        return [Entity.model_validate(r["e"]) for r in rows]

    def enriched_doc_ids(self, persona_id: str) -> set[str]:
        rows = self._read(
            """
            MATCH (c:Chunk {persona_id: $pid})-[:MENTIONS]->(:Entity)
            RETURN DISTINCT c.doc_id AS doc_id
            """,
            pid=persona_id,
        )
        return {r["doc_id"] for r in rows}

    # ------------------------------------------------------------- bulk
    def iter_documents(self, persona_id: str) -> Iterator[Document]:
        rows = self._read(
            "MATCH (d:Document {persona_id: $pid}) RETURN d {.*} AS d ORDER BY d.id",
            pid=persona_id,
        )
        for r in rows:
            yield _doc_from_node(r["d"])

    def iter_chunks(self, persona_id: str) -> Iterator[tuple[Chunk, Vector]]:
        skip = 0
        page = 2000
        while True:
            rows = self._read(
                """
                MATCH (c:Chunk {persona_id: $pid})
                RETURN c {.*} AS c ORDER BY c.doc_id, c.ordinal SKIP $skip LIMIT $page
                """,
                pid=persona_id,
                skip=skip,
                page=page,
            )
            for r in rows:
                node = r["c"]
                vec = np.asarray(node.pop("embedding") or [], dtype=np.float32)
                yield _chunk_from_node(node), vec
            if len(rows) < page:
                return
            skip += page

    def enrichment_for_persona(self, persona_id: str) -> Enrichment:
        mentions = [
            Mention(chunk_id=r["chunk_id"], entity_id=r["entity_id"])
            for r in self._read(
                """
                MATCH (c:Chunk {persona_id: $pid})-[:MENTIONS]->(e:Entity)
                RETURN c.id AS chunk_id, e.id AS entity_id ORDER BY chunk_id, entity_id
                """,
                pid=persona_id,
            )
        ]
        relations = [
            Relation.model_validate(r)
            for r in self._read(
                """
                MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
                MATCH (c:Chunk {id: r.chunk_id, persona_id: $pid})
                RETURN a.id AS source_id, b.id AS target_id, r.type AS type,
                       coalesce(r.evidence, '') AS evidence, r.chunk_id AS chunk_id
                ORDER BY source_id, target_id, type
                """,
                pid=persona_id,
            )
        ]
        entity_ids = sorted(
            {m.entity_id for m in mentions}
            | {e for r in relations for e in (r.source_id, r.target_id)}
        )
        entities = [
            Entity.model_validate(r["e"])
            for r in self._read(
                "MATCH (e:Entity) WHERE e.id IN $ids RETURN e {.*} AS e ORDER BY e.id",
                ids=entity_ids,
            )
        ]
        return Enrichment(entities=entities, mentions=mentions, relations=relations)

    # ------------------------------------------------------------- misc
    def stats(self) -> GraphStats:
        counts = {
            label: int(self._read(f"MATCH (n:{label}) RETURN count(n) AS n")[0]["n"])
            for label in ("Persona", "Source", "Document", "Chunk", "Speaker", "Topic", "Entity")
        }
        per: dict[str, dict[str, int]] = {}
        for r in self._read(
            """
            MATCH (p:Persona)
            OPTIONAL MATCH (d:Document {persona_id: p.id})
            WITH p, count(d) AS documents
            OPTIONAL MATCH (c:Chunk {persona_id: p.id})
            RETURN p.id AS id, documents, count(c) AS chunks
            """
        ):
            per[r["id"]] = {"documents": int(r["documents"]), "chunks": int(r["chunks"])}
        return GraphStats(
            personas=counts["Persona"],
            sources=counts["Source"],
            documents=counts["Document"],
            chunks=counts["Chunk"],
            speakers=counts["Speaker"],
            topics=counts["Topic"],
            entities=counts["Entity"],
            per_persona=per,
        )

    def run_readonly_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if WRITE_KEYWORDS.search(query):
            msg = "only read-only Cypher is allowed (MATCH / RETURN / WITH / CALL db.index.*)"
            raise PermissionError(msg)
        with self._driver.session(database=self._db, default_access_mode="READ") as session:
            result = session.run(query, params or {})
            return [record.data() for record in result][:500]
