# Architecture

```
files ──loader──▶ LoadedDocument ──chunker──▶ Chunk[] ──embedder──▶ float32[n,dim]
                                                   │                      │
                                                   └────── GraphStore.upsert_* ─────▶ Neo4j
                                                                                        │
  MCP / CLI ──▶ Retriever.search ──▶ vector_search + fulltext_search ──RRF──▶ hits ──expand──┘
                                                                     └──▶ ContextPack.to_prompt()
```

## Modules
- `ingest/` — `transcript.py` (front-matter + `Name (HH:MM:SS):` turns), `documents.py`
  (md/txt/pdf paragraphs), `chunker.py` (pack turns to ~300 words, split long turns on sentences),
  `loaders.py` (registry keyed by `SourceSpec.loader`).
- `embed/` — `Embedder` Protocol; `fastembed_embedder` (in-process ONNX), `http_embedder`
  (OpenAI-compatible client), `server.py` (the matching server), `hashing` (tests).
- `graph/` — `GraphStore` Protocol; `neo4j_store` (driver 6, `execute_query`, batched `UNWIND`
  upserts, `db.create.setNodeVectorProperty`), `memory_store` (executable spec for tests),
  `schema.py` (idempotent constraints/indexes), `snapshot.py` (portable export/load).
- `retrieve/` — hybrid search with reciprocal-rank fusion, neighbour expansion, context packs.
- `extract/` — `topics.py` (co-occurrence graph), `llm.py` (Claude structured extraction behind
  an `ExtractionClient` Protocol).
- `personas/` — YAML registry, brief builder, skill exporter.
- `app.py` composition root; `cli.py` (Typer); `mcp_server.py` (FastMCP).

## Graph schema
Nodes `Persona`, `Source`, `Document`, `Chunk` (with `embedding`), `Speaker`, `Topic`, `Entity`.
Edges `GROUNDED_BY`, `CONTAINS`, `HAS_CHUNK`, `NEXT`, `SPOKE`, `FEATURES`, `ABOUT`,
`CO_OCCURS{weight}`, `MENTIONS`, `RELATED_TO{type,evidence,chunk_id}`.
Indexes: vector `chunk_embedding` (cosine), full-text `chunk_text`, `document_text`, b-tree on
`persona_id` and `(doc_id, ordinal)`.

## Snapshots
`data/snapshots/<persona>/{manifest.json, documents.jsonl.gz, chunks.jsonl.gz, embeddings.npy,
enrichment.json.gz}`. The manifest pins the embedding model + dim; loading with another model is
refused because query vectors would live in a different space.

## Compute placement
Only the embedder is heavy. `GRAPHRAG_EMBEDDING_BACKEND` selects in-process vs HTTP; the same
image serves `/v1/embeddings` (`--profile embed`) so a GPU host runs identical code.
