"""Neo4j Community implementation of ``GraphStore`` (driver 6.x, ``execute_query``)."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import date
from itertools import pairwise
from typing import Any

import numpy as np
from neo4j import GraphDatabase, RoutingControl

from graphrag.config import Neo4jSettings
from graphrag.embed.base import Matrix, Vector
from graphrag.graph.schema import FULLTEXT_INDEX, VECTOR_INDEX, lucene_escape, schema_statements
from graphrag.graph.store import AttributeKind
from graphrag.graph.vectors import stack_means
from graphrag.models import (
    ENTITY_TYPES,
    Chunk,
    ChunkFacets,
    Document,
    Enrichment,
    Entity,
    EntityChunk,
    EntityCollision,
    EntityMention,
    GraphStats,
    Mention,
    MentionStance,
    PersonaSpec,
    RelatedTopic,
    Relation,
    RelationRow,
    ScoredChunk,
    SpeakerCount,
    SpeakerDocument,
    SpeakerPost,
    Stance,
    TopicCount,
    TopicEdge,
    merge_entity,
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


#: ``" ".join(name.split()).lower()`` written in Cypher, so two spellings of an entity are
#: compared here exactly as :func:`graphrag.extract.aliases.fold_name` compares them in Python.
FOLD_ENTITY_NAME = (
    "reduce(s = '', w IN split(toLower(trim(e.name)), ' ') | "
    "CASE WHEN w = '' THEN s WHEN s = '' THEN w ELSE s + ' ' + w END)"
)


def _fold_name(name: str) -> str:
    """The Python half of :data:`FOLD_ENTITY_NAME`: lower case, whitespace collapsed."""
    return " ".join(name.split()).casefold()


def _chunk_props(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "persona_id": chunk.persona_id,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "speaker": chunk.speaker,
        "speakers": chunk.speakers,
        # A list of maps cannot be a node property, and the SPOKE edges these rebuild are not
        # in the snapshot, so the passage carries them as JSON the way a Document carries its
        # metadata.
        "speaker_posts_json": json.dumps([p.model_dump() for p in chunk.speaker_posts]),
        "facets": chunk.facets,
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
        speaker_posts=[
            SpeakerPost.model_validate(p)
            for p in json.loads(node.get("speaker_posts_json") or "[]")
        ],
        facets=list(node.get("facets") or []),
        start_ts=node.get("start_ts"),
        start_seconds=node.get("start_seconds"),
        url=node.get("url"),
        word_count=int(node.get("word_count") or 0),
    )


#: How a persona's speaker attributes are named on the shared ``Speaker`` node. The node is
#: keyed on the name alone, so two personas can hold the same handle; a reading written for one
#: corpus is not a fact about the other, and the prefix is what keeps them apart. Keys are
#: slugs (:data:`graphrag.extract.attributes.KEY_RE`) and persona ids are slugs, so the double
#: underscore never occurs inside either half and the split back is unambiguous.
ATTR_PREFIX = "attr__"


def _attr_prefix(persona_id: str) -> str:
    return f"{ATTR_PREFIX}{persona_id}__"


def _scoped_attributes(props: Mapping[str, Any], persona_id: str) -> dict[str, str]:
    """This persona's attributes out of a shared node's properties.

    ``Speaker`` and ``Entity`` nodes are both shared between personas and both carry their
    attributes as persona-prefixed properties, so both are read the same way.
    """
    prefix = _attr_prefix(persona_id)
    return {
        key[len(prefix) :]: str(value)
        for key, value in props.items()
        if key.startswith(prefix) and value is not None
    }


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
        # Carried as JSON for the reason `metadata_json` is: a map cannot be a node property.
        # Written on every upsert, so a re-ingest drops what the annotation layer put here
        # exactly as it drops the stances, and `sync --refresh-annotations` puts it back.
        "attributes_json": json.dumps(doc.attributes, sort_keys=True),
        "word_count": doc.word_count,
    }


def _mention_row(mention: Mention) -> dict[str, Any]:
    """One mention as Cypher parameters, with ``unknown`` written as a null.

    ``unknown`` is the absence of a tier rather than a fourth measurement of one, so it travels
    as ``NULL`` and the write is a ``coalesce`` -- exactly how the stance behaves. That is what
    lets a re-import of an old extraction file, or of a snapshot written before tiers existed,
    land on an edge that already carries a tier without erasing it. Reads put ``unknown`` back.
    """
    row = mention.model_dump()
    row["tier"] = None if mention.tier == "unknown" else mention.tier
    return row


def _attributes(raw: Any) -> dict[str, str]:
    """A node's ``attributes_json`` as a plain mapping of strings."""
    loaded = json.loads(raw or "{}")
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items() if value is not None}


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
        attributes=_attributes(node.get("attributes_json")),
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
                {
                    "id": c.id,
                    "props": _chunk_props(c),
                    "embedding": [float(x) for x in v],
                    "posts": [p.model_dump() for p in c.speaker_posts],
                }
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
            # The post's date, role and score live on the edge, so loading a snapshot has to put
            # them back; the passage carried them across as JSON.
            dated = [row for row in rows if row["posts"]]
            if dated:
                self._run(
                    """
                    UNWIND $rows AS row
                    MATCH (c:Chunk {id: row.id})
                    UNWIND row.posts AS post
                    MERGE (s:Speaker {name: post.speaker})
                    MERGE (s)-[r:SPOKE]->(c)
                    SET r.posted_at = post.posted_at, r.role = post.role, r.score = post.score
                    """,
                    rows=dated,
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

    def _entities_by_id(self, ids: set[str]) -> dict[str, tuple[Entity, int]]:
        """The entity nodes these ids name and how many passages mention each, as models.

        Ids nobody holds are simply absent. The mention count rides along because
        ``merge_entity`` needs it to tell a node that stands for something from one a re-ingest
        emptied out. ``type`` is guarded rather than trusted: a node edited by hand in the
        browser can carry anything, and an import of hundreds of files should not stop on one
        bad property.
        """
        return {
            row["id"]: (
                Entity(
                    id=row["id"],
                    name=row["name"],
                    type=row["type"] if row["type"] in ENTITY_TYPES else "other",
                    description=row["description"],
                    aliases=row["aliases"],
                ),
                int(row["mentions"]),
            )
            for row in self._read(
                """
                MATCH (e:Entity) WHERE e.id IN $ids
                RETURN e.id AS id, coalesce(e.name, '') AS name,
                       coalesce(e.type, 'other') AS type,
                       coalesce(e.description, '') AS description,
                       coalesce(e.aliases, []) AS aliases,
                       count { (e)<-[:MENTIONS]-() } AS mentions
                """,
                ids=sorted(ids),
            )
        }

    def upsert_enrichment(self, enrichment: Enrichment) -> list[EntityCollision]:
        collisions: list[EntityCollision] = []
        if enrichment.entities:
            # Read the nodes these ids already name before writing any of them: an id whose node
            # is called something else is a slug collision, and the node keeps its name. The
            # merge itself is done in Python by ``merge_entity`` so both stores decide alike,
            # which is why the write below sets the resolved values outright.
            held = self._entities_by_id({e.id for e in enrichment.entities})
            rows: list[dict[str, Any]] = []
            for entity in enrichment.entities:
                found = held.get(entity.id)
                resolved, collision = merge_entity(
                    None if found is None else found[0],
                    entity,
                    held_mentions=None if found is None else found[1],
                )
                # so a second row for this id sees the resolved node, still with its own count
                held[entity.id] = (resolved, 0 if found is None else found[1])
                if collision is not None:
                    collisions.append(collision)
                    continue
                rows.append(resolved.model_dump())
            if rows:
                self._run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {id: row.id})
                    SET e.name = row.name, e.type = row.type,
                        e.description = row.description, e.aliases = row.aliases
                    """,
                    rows=rows,
                )
        if enrichment.mentions:
            self._run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.chunk_id}), (e:Entity {id: row.entity_id})
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.stance = coalesce(row.stance, m.stance),
                    m.tier = coalesce(row.tier, m.tier)
                """,
                rows=[_mention_row(m) for m in enrichment.mentions],
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
        return collisions

    def merge_entities(self, persona_id: str, canonical: str, aliases: Sequence[str]) -> int:
        """Fold every alias spelling of ``canonical`` into one node. Returns mentions moved.

        Scoped to one persona: entity nodes are shared, so another persona's mentions of an
        alias spelling stay where they are and the alias node survives. Relations are scoped the
        same way, through the passage each one is anchored to.
        """
        spellings = sorted({_fold_name(a) for a in aliases if a.strip()} | {_fold_name(canonical)})
        found = self._read(
            f"""
            MATCH (e:Entity) WHERE {FOLD_ENTITY_NAME} IN $spellings
            OPTIONAL MATCH (c:Chunk {{persona_id: $pid}})-[:MENTIONS]->(e)
            RETURN e.id AS id, e.name AS name, coalesce(e.type, 'other') AS type,
                   coalesce(e.description, '') AS description, coalesce(e.aliases, []) AS aliases,
                   count(c) AS mentions
            """,
            spellings=spellings,
            pid=persona_id,
        )
        if not found:
            return 0
        folded = _fold_name(canonical)
        best = max(
            found, key=lambda r: (_fold_name(r["name"]) == folded, int(r["mentions"]), r["id"])
        )
        target_id = Entity.make_id(canonical, best["type"])
        recorded = sorted(
            {a for row in found for a in row["aliases"]}
            | {a.strip() for a in aliases if a.strip() and _fold_name(a) != folded}
        )
        self._run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name, e.type = $type,
                e.description = CASE WHEN coalesce(e.description, '') <> '' THEN e.description
                                     ELSE $description END,
                e.aliases = $aliases
            """,
            id=target_id,
            name=canonical,
            type=best["type"],
            description=best["description"],
            aliases=recorded,
        )
        stale = [row["id"] for row in found if row["id"] != target_id]
        if not stale:
            return 0
        group = [*stale, target_id]

        # A relation between two spellings of one thing says nothing once they are one node.
        self._run(
            """
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            WHERE a.id IN $group AND b.id IN $group
            MATCH (:Chunk {id: r.chunk_id, persona_id: $pid})
            DELETE r
            """,
            group=group,
            pid=persona_id,
        )
        self._run(
            """
            MATCH (old:Entity)-[r:RELATED_TO]->(other:Entity)
            WHERE old.id IN $stale AND NOT other.id IN $group
            MATCH (:Chunk {id: r.chunk_id, persona_id: $pid})
            MATCH (target:Entity {id: $target_id})
            MERGE (target)-[n:RELATED_TO {type: r.type, chunk_id: r.chunk_id}]->(other)
            SET n.evidence = CASE WHEN coalesce(n.evidence, '') <> '' THEN n.evidence
                                  ELSE coalesce(r.evidence, '') END
            DELETE r
            """,
            stale=stale,
            group=group,
            pid=persona_id,
            target_id=target_id,
        )
        self._run(
            """
            MATCH (other:Entity)-[r:RELATED_TO]->(old:Entity)
            WHERE old.id IN $stale AND NOT other.id IN $group
            MATCH (:Chunk {id: r.chunk_id, persona_id: $pid})
            MATCH (target:Entity {id: $target_id})
            MERGE (other)-[n:RELATED_TO {type: r.type, chunk_id: r.chunk_id}]->(target)
            SET n.evidence = CASE WHEN coalesce(n.evidence, '') <> '' THEN n.evidence
                                  ELSE coalesce(r.evidence, '') END
            DELETE r
            """,
            stale=stale,
            group=group,
            pid=persona_id,
            target_id=target_id,
        )
        # The tier of the surviving mention is the better-supported of the two, which is
        # :func:`graphrag.models.strongest_tier` written in Cypher: the four tiers sort
        # strongest-first alphabetically, so the smaller string is the stronger evidence.
        # ``tests/unit/test_sna_uncertain.py`` pins that ordering, because this statement and
        # nothing else depends on it.
        moved = self._run(
            """
            MATCH (c:Chunk {persona_id: $pid})-[m:MENTIONS]->(old:Entity)
            WHERE old.id IN $stale
            MATCH (target:Entity {id: $target_id})
            MERGE (c)-[n:MENTIONS]->(target)
            SET n.stance = coalesce(n.stance, m.stance),
                n.tier = CASE
                    WHEN m.tier IS NULL THEN n.tier
                    WHEN n.tier IS NULL THEN m.tier
                    WHEN m.tier < n.tier THEN m.tier
                    ELSE n.tier END
            DELETE m
            RETURN count(*) AS moved
            """,
            stale=stale,
            pid=persona_id,
            target_id=target_id,
        )
        # Only a node nothing holds any more: another persona may still mention this spelling.
        self._run(
            "MATCH (e:Entity) WHERE e.id IN $stale AND NOT (e)--() DELETE e",
            stale=stale,
        )
        return int(moved[0]["moved"]) if moved else 0

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
        self.delete_orphan_entities(persona_id)
        self._run("MATCH (s:Speaker) WHERE NOT (s)--() DELETE s")
        self._run("MATCH (t:Topic) WHERE NOT (t)<-[:ABOUT]-() DETACH DELETE t")

    def delete_orphan_entities(self, persona_id: str, *, dry_run: bool = False) -> int:
        """Drop the entity nodes nothing mentions any more. Returns how many went.

        Scoped the way ``merge_entities`` is: entity nodes are shared, so a node another
        persona's passage still relates to survives even with no mention on it. A relation whose
        passage is gone holds nothing, so it does not keep its endpoints alive; the RELATED_TO
        edges of a node that goes are detached with it.
        """
        match = """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-()
              AND NOT EXISTS {
                MATCH (e)-[r:RELATED_TO]-()
                MATCH (c:Chunk {id: r.chunk_id}) WHERE c.persona_id <> $pid
              }
        """
        if dry_run:
            rows = self._read(match + "RETURN count(e) AS removed", pid=persona_id)
        else:
            rows = self._run(
                match + "WITH collect(e) AS doomed "
                "FOREACH (e IN doomed | DETACH DELETE e) "
                "RETURN size(doomed) AS removed",
                pid=persona_id,
            )
        return int(rows[0]["removed"]) if rows else 0

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

    def document_ids(self, persona_id: str, source_id: str | None = None) -> set[str]:
        rows = self._read(
            """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE $source_id IS NULL OR d.source_id = $source_id
            RETURN d.id AS id
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return {r["id"] for r in rows}

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

    def persona_entities(self, persona_id: str) -> list[Entity]:
        rows = self._paged(
            """
            MATCH (c:Chunk {persona_id: $persona_id})-[:MENTIONS]->(e:Entity)
            RETURN DISTINCT e {.*} AS e ORDER BY e.id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
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

    def enriched_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        rows = self._read(
            """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE d.source_id = $source_id
              AND EXISTS { MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity) }
            RETURN d.id AS id
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return {r["id"] for r in rows}

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
            Mention(
                chunk_id=r["chunk_id"],
                entity_id=r["entity_id"],
                stance=r["stance"],
                tier=r["tier"],
            )
            for r in self._read(
                """
                MATCH (c:Chunk {persona_id: $pid})-[m:MENTIONS]->(e:Entity)
                RETURN c.id AS chunk_id, e.id AS entity_id, m.stance AS stance,
                       coalesce(m.tier, 'unknown') AS tier
                ORDER BY chunk_id, entity_id
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

    def persona_fingerprint(self, persona_id: str) -> tuple[int, int, int, int]:
        """See the protocol docstring. One round trip: six independent ``CALL {}`` subqueries,
        each guaranteed exactly one output row (every ``MATCH`` inside is ``OPTIONAL``, and the
        two-pattern speaker search collects into a list and counts that, rather than returning
        rows to count outside -- a ``CALL {}`` subquery that itself returns zero rows drops the
        correlated outer row, which silently zeroed every column here during development, not
        only the one the empty subquery belonged to)."""
        rows = self._read(
            """
            CALL {
                OPTIONAL MATCH (d:Document {persona_id: $persona_id})
                RETURN count(d) AS documents
            }
            CALL {
                OPTIONAL MATCH (c:Chunk {persona_id: $persona_id})
                RETURN count(c) AS chunks
            }
            CALL {
                OPTIONAL MATCH (:Chunk {persona_id: $persona_id})-[:MENTIONS]->(e:Entity)
                RETURN count(e) AS mentions
            }
            CALL {
                OPTIONAL MATCH (d:Document {persona_id: $persona_id})
                WHERE d.attributes_json IS NOT NULL AND d.attributes_json <> '{}'
                RETURN count(d) AS attributed_documents
            }
            CALL {
                OPTIONAL MATCH (:Chunk {persona_id: $persona_id})-[:MENTIONS]->(e:Entity)
                WHERE any(key IN keys(e) WHERE key STARTS WITH $prefix)
                RETURN count(DISTINCT e) AS attributed_entities
            }
            CALL {
                OPTIONAL MATCH (:Document {persona_id: $persona_id})-[:FEATURES]->(s1:Speaker)
                WHERE any(key IN keys(s1) WHERE key STARTS WITH $prefix)
                WITH collect(DISTINCT s1) AS featured
                OPTIONAL MATCH (:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(:Chunk)
                    <-[:SPOKE]-(s2:Speaker)
                WHERE any(key IN keys(s2) WHERE key STARTS WITH $prefix)
                WITH featured, collect(DISTINCT s2) AS spoke
                UNWIND (featured + spoke) AS s
                RETURN count(DISTINCT s) AS attributed_speakers
            }
            RETURN documents, chunks, mentions,
                   attributed_documents + attributed_entities + attributed_speakers
                       AS attributed_nodes
            """,
            persona_id=persona_id,
            prefix=_attr_prefix(persona_id),
        )
        row = rows[0]
        return (
            int(row["documents"]),
            int(row["chunks"]),
            int(row["mentions"]),
            int(row["attributed_nodes"]),
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

    # ------------------------------------------------------------- attribution
    def attach_speaker(
        self,
        doc_id: str,
        chunk_id: str,
        speaker: str,
        *,
        posted_at: str | None = None,
        role: str | None = None,
        score: int | None = None,
    ) -> None:
        """Record that ``speaker`` wrote the passage ``chunk_id`` of document ``doc_id``.

        Writes the same shape the transcripts path writes on upsert -- a ``Speaker`` node, a
        ``SPOKE`` edge to the passage and a ``FEATURES`` edge from the document -- plus the
        ``speakers`` properties those edges are rebuilt from, so an attributed document survives
        a snapshot export and load. Every ``MERGE`` makes re-import a no-op. A ``chunk_id`` that
        is not a passage of ``doc_id`` matches nothing and writes nothing.

        When the post is dated, given a role or scored, those go on the ``SPOKE`` edge and on
        the passage's ``speaker_posts``, which is what carries them into a snapshot. A second
        call for the same speaker and passage replaces that record: one speaker holds one
        passage once, however many times a file says so.
        """
        self._run(
            """
            MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
            MERGE (s:Speaker {name: $speaker})
            MERGE (s)-[r:SPOKE]->(c)
            MERGE (d)-[:FEATURES]->(s)
            SET r.posted_at = $posted_at, r.role = $role, r.score = $score,
                c.speakers = CASE
                    WHEN $speaker IN coalesce(c.speakers, []) THEN c.speakers
                    ELSE coalesce(c.speakers, []) + $speaker END,
                d.speakers = CASE
                    WHEN $speaker IN coalesce(d.speakers, []) THEN d.speakers
                    ELSE coalesce(d.speakers, []) + $speaker END
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            speaker=speaker,
            posted_at=posted_at,
            role=role,
            score=score,
        )
        if posted_at is None and role is None and score is None:
            return
        self._run(
            """
            MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
            SET c.speaker_posts_json = $posts
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            posts=json.dumps(self._merged_posts(doc_id, chunk_id, speaker, posted_at, role, score)),
        )

    def _merged_posts(
        self,
        doc_id: str,
        chunk_id: str,
        speaker: str,
        posted_at: str | None,
        role: str | None,
        score: int | None,
    ) -> list[dict[str, Any]]:
        """The passage's post records with this speaker's replaced, read back before writing."""
        rows = self._read(
            """
            MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
            RETURN coalesce(c.speaker_posts_json, '[]') AS posts
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
        held = json.loads(rows[0]["posts"]) if rows else []
        post = SpeakerPost(speaker=speaker, posted_at=posted_at, role=role, score=score)
        return [p for p in held if p.get("speaker") != speaker] + [post.model_dump()]

    def attributed_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        rows = self._read(
            """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE d.source_id = $source_id AND EXISTS { MATCH (d)-[:FEATURES]->(:Speaker) }
            RETURN d.id AS id
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return {r["id"] for r in rows}

    # ------------------------------------------------------------- node attributes
    def set_speaker_attributes(
        self, persona_id: str, speaker: str, attributes: Mapping[str, str]
    ) -> list[str]:
        """Record what this persona's corpus says the speaker is. Returns the keys it refused.

        Held as prefixed properties on the shared ``Speaker`` node rather than on an edge to
        the ``Persona``: a node with no passages left is deleted by the orphan sweep in
        ``delete_persona``, and an attribute must not be the thing that keeps it alive.

        Read then written, in two statements, because first-value-wins needs the held value and
        Cypher cannot build a map from a computed list without APOC. The import path is a single
        writer, so nothing races between the two. The write matches rather than merges the node:
        a speaker no passage records is not a speaker, and an attribute is not a reason to keep
        one alive.
        """
        wanted = {key: value for key, value in attributes.items() if value}
        if not wanted:
            return []
        rows = self._read(
            "MATCH (s:Speaker {name: $speaker}) RETURN properties(s) AS props", speaker=speaker
        )
        held = _scoped_attributes(rows[0]["props"], persona_id) if rows else {}
        refused = [key for key, value in wanted.items() if held.get(key, value) != value]
        fresh = {
            _attr_prefix(persona_id) + key: value
            for key, value in wanted.items()
            if key not in refused
        }
        if fresh:
            self._run(
                "MATCH (s:Speaker {name: $speaker}) SET s += $props",
                speaker=speaker,
                props=fresh,
            )
        return refused

    def set_entity_attributes(
        self, persona_id: str, entity_id: str, attributes: Mapping[str, str]
    ) -> list[str]:
        """Record what this persona's corpus says the entity is. Returns the keys it refused.

        Held as persona-prefixed properties on the shared ``Entity`` node, for the reason
        :meth:`set_speaker_attributes` gives: the node is shared between personas, and "which
        sector this company is in" is a reading of one corpus rather than a fact about the
        world.

        Read then written, in two statements, because first-value-wins needs the held value.
        The write matches rather than merges: an entity no passage mentions is not an entity,
        and an attribute must not be the thing that keeps one alive past the orphan sweep.
        """
        wanted = {key: value for key, value in attributes.items() if value}
        if not wanted:
            return []
        rows = self._read(
            "MATCH (e:Entity {id: $entity_id}) RETURN properties(e) AS props", entity_id=entity_id
        )
        if not rows:
            return list(wanted)
        held = _scoped_attributes(rows[0]["props"], persona_id)
        refused = [key for key, value in wanted.items() if held.get(key, value) != value]
        fresh = {
            _attr_prefix(persona_id) + key: value
            for key, value in wanted.items()
            if key not in refused
        }
        if fresh:
            self._run(
                "MATCH (e:Entity {id: $entity_id}) SET e += $props",
                entity_id=entity_id,
                props=fresh,
            )
        return refused

    def set_document_attributes(self, doc_id: str, attributes: Mapping[str, str]) -> None:
        """Merge attributes into the document, the incoming value winning per key."""
        rows = self._read(
            "MATCH (d:Document {id: $doc_id}) RETURN coalesce(d.attributes_json, '{}') AS held",
            doc_id=doc_id,
        )
        if not rows:
            return
        merged = {**_attributes(rows[0]["held"]), **{k: v for k, v in attributes.items() if v}}
        self._run(
            "MATCH (d:Document {id: $doc_id}) SET d.attributes_json = $json",
            doc_id=doc_id,
            json=json.dumps(merged, sort_keys=True),
        )

    def clear_attributes(self, persona_id: str, kind: AttributeKind) -> int:
        """Forget this persona's values of one kind; return how many nodes held one.

        Speaker and entity values are persona-prefixed properties on shared nodes, so only the
        keys under this persona's prefix go: setting a property to null in ``SET +=`` removes it,
        which names each key without needing APOC's dynamic property removal. A document belongs
        to one persona and keeps its values in one property, so that property goes whole.
        """
        if kind == "document":
            rows = self._read(
                """
                MATCH (d:Document {persona_id: $persona_id})
                WHERE d.attributes_json IS NOT NULL
                RETURN count(d) AS n
                """,
                persona_id=persona_id,
            )
            self._run(
                "MATCH (d:Document {persona_id: $persona_id}) REMOVE d.attributes_json",
                persona_id=persona_id,
            )
            return int(rows[0]["n"]) if rows else 0
        label = "Speaker" if kind == "speaker" else "Entity"
        held = self._read(
            f"""
            MATCH (n:{label})
            WITH n, [key IN keys(n) WHERE key STARTS WITH $prefix] AS held
            WHERE size(held) > 0
            RETURN elementId(n) AS node, held
            """,
            prefix=_attr_prefix(persona_id),
        )
        rows = [{"node": r["node"], "nulls": dict.fromkeys(r["held"])} for r in held]
        if rows:
            self._run(
                f"UNWIND $rows AS row MATCH (n:{label}) WHERE elementId(n) = row.node "
                "SET n += row.nulls",
                rows=rows,
            )
        return len(rows)

    def speaker_attributes(self, persona_id: str) -> dict[str, dict[str, str]]:
        """Every speaker of this persona that carries attributes, with what it carries."""
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})
            CALL {
              WITH d
              MATCH (d)-[:FEATURES]->(s:Speaker) RETURN s
              UNION
              WITH d
              MATCH (d)-[:HAS_CHUNK]->(:Chunk)<-[:SPOKE]-(s:Speaker) RETURN s
            }
            WITH DISTINCT s
            RETURN s.name AS speaker, properties(s) AS props
            ORDER BY speaker
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
        )
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            attrs = _scoped_attributes(row["props"], persona_id)
            if attrs:
                out[row["speaker"]] = attrs
        return out

    def document_attributes(self, persona_id: str) -> dict[str, dict[str, str]]:
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE d.attributes_json IS NOT NULL AND d.attributes_json <> '{}'
            RETURN d.id AS doc_id, d.attributes_json AS attrs
            ORDER BY doc_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
        )
        return {row["doc_id"]: _attributes(row["attrs"]) for row in rows}

    def entity_attributes(self, persona_id: str) -> dict[str, dict[str, str]]:
        """Every entity this persona's passages mention that carries attributes.

        Scoped through the mentions rather than through the node, exactly as
        :meth:`persona_entities` is: the node is shared, so "the entities of this persona" only
        ever means the ones its passages name.
        """
        rows = self._paged(
            """
            MATCH (c:Chunk {persona_id: $persona_id})-[:MENTIONS]->(e:Entity)
            WITH DISTINCT e
            RETURN e.id AS entity_id, properties(e) AS props
            ORDER BY entity_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
        )
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            attrs = _scoped_attributes(row["props"], persona_id)
            if attrs:
                out[row["entity_id"]] = attrs
        return out

    # ------------------------------------------------------------- annotation
    def annotate_mention(self, doc_id: str, chunk_id: str, entity: str, stance: Stance) -> None:
        """Set ``stance`` on the ``MENTIONS`` edge from this passage to ``entity``.

        The entity is looked up among the ones the passage already mentions, by id or by folded
        name: a stance is a reading of an edge that exists, so an entity nobody extracted here
        matches nothing and writes nothing rather than creating the edge it would need.
        """
        self._run(
            f"""
            MATCH (:Document {{id: $doc_id}})-[:HAS_CHUNK]->(c:Chunk {{id: $chunk_id}})
            MATCH (c)-[m:MENTIONS]->(e:Entity)
            WHERE e.id = $entity OR {FOLD_ENTITY_NAME} = $folded
            SET m.stance = $stance
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            entity=entity.strip(),
            folded=_fold_name(entity),
            stance=stance,
        )

    def annotate_chunk(self, doc_id: str, chunk_id: str, facets: Sequence[str]) -> None:
        """Add ``facets`` to the passage's ``facets`` list, deduplicated and order-preserving."""
        self._run(
            """
            MATCH (:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {id: $chunk_id})
            SET c.facets = reduce(
                held = coalesce(c.facets, []), f IN $facets |
                CASE WHEN f IN held THEN held ELSE held + f END)
            """,
            doc_id=doc_id,
            chunk_id=chunk_id,
            facets=list(facets),
        )

    def annotated_document_ids(self, persona_id: str, source_id: str) -> set[str]:
        # A document counts as annotated when a passage carries a facet, a mention carries a
        # stance, or the document itself carries an attribute. That third case covers an
        # annotation file with nothing but a top-level `attributes` block and no `annotations`
        # entries: it wrote `d.attributes_json`, and without this clause that import looks
        # identical to one that never ran.
        rows = self._read(
            """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE d.source_id = $source_id
              AND (EXISTS { MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                            WHERE size(coalesce(c.facets, [])) > 0 }
                   OR EXISTS { MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[m:MENTIONS]->(:Entity)
                               WHERE m.stance IS NOT NULL }
                   OR (d.attributes_json IS NOT NULL AND d.attributes_json <> '{}'))
            RETURN d.id AS id
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return {r["id"] for r in rows}

    def mention_stances(self, persona_id: str, source_id: str | None = None) -> list[MentionStance]:
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[m:MENTIONS]->(e:Entity)
            WHERE m.stance IS NOT NULL AND ($source_id IS NULL OR d.source_id = $source_id)
            RETURN e.id AS entity_id, e.name AS name, c.id AS chunk_id, d.id AS doc_id,
                   m.stance AS stance, coalesce(c.speakers, []) AS speakers
            ORDER BY entity_id, chunk_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return [MentionStance.model_validate(r) for r in rows]

    def chunk_facets(self, persona_id: str, source_id: str | None = None) -> list[ChunkFacets]:
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk)
            WHERE size(coalesce(c.facets, [])) > 0
              AND ($source_id IS NULL OR d.source_id = $source_id)
            RETURN c.id AS chunk_id, d.id AS doc_id, c.facets AS facets
            ORDER BY chunk_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return [ChunkFacets.model_validate(r) for r in rows]

    # ------------------------------------------------------------- network analysis
    def _paged(self, query: str, **params: Any) -> Iterator[dict[str, Any]]:
        """Run a read-only query in ``BATCH``-row pages. The query must end in SKIP/LIMIT."""
        skip = 0
        while True:
            rows = self._read(query, skip=skip, page=BATCH, **params)
            yield from rows
            if len(rows) < BATCH:
                return
            skip += BATCH

    def speaker_document_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[SpeakerDocument]:
        # A speaker belongs to a document either because the document credits them (FEATURES)
        # or because they hold one of its passages (SPOKE); ingestion writes both, but an
        # attribution pass can land one before the other, so take the union.
        query = """
            MATCH (d:Document {persona_id: $persona_id})
            WHERE $source_id IS NULL OR d.source_id = $source_id
            CALL {
              WITH d
              MATCH (d)-[:FEATURES]->(s:Speaker) RETURN s
              UNION
              WITH d
              MATCH (d)-[:HAS_CHUNK]->(:Chunk)<-[:SPOKE]-(s:Speaker) RETURN s
            }
            WITH d, s
            OPTIONAL MATCH (s)-[:SPOKE]->(c:Chunk {doc_id: d.id})
            WITH d, s, count(c) AS chunks
            RETURN s.name AS speaker, d.id AS doc_id, chunks, properties(s) AS props,
                   coalesce(d.attributes_json, '{}') AS doc_attrs
            ORDER BY speaker, doc_id
            SKIP $skip LIMIT $page
            """
        if since is not None or until is not None:
            # A window is answered from the dated SPOKE edges alone: a document credit carries no
            # date, and an undated post is left out rather than assumed to fall in range.
            query = """
                MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk)
                MATCH (s:Speaker)-[r:SPOKE]->(c)
                WHERE ($source_id IS NULL OR d.source_id = $source_id)
                  AND r.posted_at IS NOT NULL
                  AND ($since IS NULL OR r.posted_at >= $since)
                  AND ($until IS NULL OR r.posted_at <= $until)
                WITH d, s, count(c) AS chunks
                RETURN s.name AS speaker, d.id AS doc_id, chunks, properties(s) AS props,
                       coalesce(d.attributes_json, '{}') AS doc_attrs
                ORDER BY speaker, doc_id
                SKIP $skip LIMIT $page
                """
        rows = self._paged(
            query, persona_id=persona_id, source_id=source_id, since=since, until=until
        )
        return [
            SpeakerDocument(
                speaker=r["speaker"],
                doc_id=r["doc_id"],
                chunks=int(r["chunks"]),
                speaker_attributes=_scoped_attributes(r["props"], persona_id),
                document_attributes=_attributes(r["doc_attrs"]),
            )
            for r in rows
        ]

    def entity_chunk_pairs(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityChunk]:
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[m:MENTIONS]->(e:Entity)
            WHERE ($source_id IS NULL OR d.source_id = $source_id)
              AND ($types IS NULL OR e.type IN $types)
            RETURN e.id AS entity_id, e.name AS name, coalesce(e.type, 'other') AS type,
                   c.id AS chunk_id, d.id AS doc_id, coalesce(m.tier, 'unknown') AS tier
            ORDER BY entity_id, chunk_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            source_id=source_id,
            types=list(types) if types else None,
        )
        return [EntityChunk.model_validate(r) for r in rows]

    def topic_edges(self, persona_id: str, min_weight: int = 1) -> list[TopicEdge]:
        rows = self._paged(
            """
            MATCH (a:Topic)-[r:CO_OCCURS]-(b:Topic)
            WHERE a.name < b.name AND coalesce(r.weight, 1) >= $min_weight
              AND EXISTS { MATCH (:Document {persona_id: $persona_id})-[:ABOUT]->(a) }
              AND EXISTS { MATCH (:Document {persona_id: $persona_id})-[:ABOUT]->(b) }
            RETURN a.name AS source, b.name AS target, coalesce(r.weight, 1) AS weight
            ORDER BY source, target
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            min_weight=min_weight,
        )
        return [
            TopicEdge(source=r["source"], target=r["target"], weight=int(r["weight"])) for r in rows
        ]

    def relation_rows(self, persona_id: str, source_id: str | None = None) -> list[RelationRow]:
        """The directed ``RELATED_TO`` edges this persona's passages state.

        The scope runs through the passage each relation is anchored to, for the reason
        :meth:`enrichment_for_persona` scopes through it: the ``Entity`` nodes are shared between
        personas, so a relation another persona's passage stated is not this persona's to
        export. The route differs, though, and deliberately -- that method matches the passage
        on ``Chunk.persona_id``, this one reaches it from its ``Document``, because the document
        is what carries ``source_id`` and a relation whose passage a re-ingest removed then
        comes back not at all rather than unscoped.
        """
        rows = self._paged(
            """
            MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
            MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk {id: r.chunk_id})
            WHERE $source_id IS NULL OR d.source_id = $source_id
            RETURN a.id AS source_id, a.name AS source_name,
                   coalesce(a.type, 'other') AS source_type,
                   b.id AS target_id, b.name AS target_name,
                   coalesce(b.type, 'other') AS target_type,
                   r.type AS type, c.id AS chunk_id, d.id AS doc_id
            ORDER BY source_id, target_id, type, chunk_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            source_id=source_id,
        )
        return [RelationRow.model_validate(r) for r in rows]

    def mean_embeddings(self, persona_id: str, level: str = "document") -> tuple[list[str], Matrix]:
        if level == "document":
            query = """
                MATCH (c:Chunk {persona_id: $persona_id})
                RETURN c.doc_id AS key, c.embedding AS embedding
                ORDER BY c.doc_id, c.ordinal
                SKIP $skip LIMIT $page
                """
        elif level == "entity":
            query = """
                MATCH (c:Chunk {persona_id: $persona_id})-[:MENTIONS]->(e:Entity)
                RETURN e.id AS key, c.embedding AS embedding
                ORDER BY e.id, c.id
                SKIP $skip LIMIT $page
                """
        else:
            msg = f"level must be 'document' or 'entity', got {level!r}"
            raise ValueError(msg)
        sums: dict[str, np.ndarray] = {}
        counts: Counter[str] = Counter()
        for row in self._paged(query, persona_id=persona_id):
            raw = row["embedding"]
            if not raw:
                continue
            vec = np.asarray(raw, dtype=np.float32)
            key = row["key"]
            sums[key] = sums.get(key, np.zeros_like(vec)) + vec
            counts[key] += 1
        return stack_means(sums, counts)

    def entity_mention_rows(
        self,
        persona_id: str,
        source_id: str | None = None,
        types: Sequence[str] | None = None,
    ) -> list[EntityMention]:
        rows = self._paged(
            """
            MATCH (d:Document {persona_id: $persona_id})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (c)-[m:MENTIONS]->(e:Entity)
            WHERE ($source_id IS NULL OR d.source_id = $source_id)
              AND ($types IS NULL OR e.type IN $types)
            RETURN e.id AS entity_id, e.name AS name, coalesce(e.type, 'other') AS type,
                   c.id AS chunk_id, d.id AS doc_id, m.stance AS stance,
                   coalesce(c.speakers, []) AS speakers,
                   coalesce(d.attributes_json, '{}') AS doc_attrs,
                   coalesce(m.tier, 'unknown') AS tier
            ORDER BY entity_id, chunk_id
            SKIP $skip LIMIT $page
            """,
            persona_id=persona_id,
            source_id=source_id,
            types=list(types) if types else None,
        )
        return [
            EntityMention(
                entity_id=r["entity_id"],
                name=r["name"],
                type=r["type"],
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                stance=r["stance"],
                speakers=list(r["speakers"]),
                document_attributes=_attributes(r["doc_attrs"]),
                tier=r["tier"],
            )
            for r in rows
        ]
