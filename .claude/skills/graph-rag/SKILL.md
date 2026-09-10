---
name: graph-rag
description: Set up, query and extend the graph-rag knowledge graph (Neo4j + FastMCP). Use when an agent needs grounded, cited answers from a persona (e.g. product leader), when asked to "use the knowledge graph", "ground this in real sources", "which persona applies", or when adding a new persona or ingesting new documents.
---

# graph-rag: persona-grounded knowledge graph

Everything runs in Docker. The graph snapshot is committed, so setup is a load, not a rebuild.

**Repo location:** wherever you cloned this repo. Every `make` and `docker compose` command
below runs from that directory; when you are working in another project, `cd` there first. If you
do not know where it is, ask the user for the path once and reuse it for the session.

## 1. Initialise (first time on a machine)
```bash
make setup          # hooks, images, Neo4j, load data/snapshots/*, MCP server on :8765
make doctor         # verifies Neo4j, embedder backend/model/dim, snapshot compatibility
```
Claude Code picks the server up from `.mcp.json` (`graphrag` → http://localhost:8765/mcp).

## 2. Use it from an agent
1. `list_personas` → pick one; `recommend_personas(sdlc_stage)` if unsure.
2. `persona_brief(persona_id)` → adopt as operating instructions (or use the
   `assume_persona` prompt).
3. `context(query, persona_id)` → prompt-ready passages with citations; answer with [n] cites.
4. Dig deeper with `search`, `documents`, `read_document`, `topics`, `related_topics`,
   `speakers`; `cypher` for read-only graph queries.

Per-persona skills live in `.claude/skills/persona-<id>/` (regenerate with
`graphrag persona export-skill --all`).

## 3. CLI equivalents (agentic use without MCP)
```bash
docker compose run --rm graphrag graphrag context "how do I know I have PMF" -p product-leader
docker compose run --rm graphrag graphrag search "retro format" -p product-leader -k 5 --json
```

## 4. Add knowledge
- New persona: `graphrag persona new "Name" -d "..."`, then follow the generated README.
- New documents for an existing persona: drop files under `data/raw/<persona>/<source>/`, add a
  `sources:` entry, `make ingest PERSONA=<id> SRC=data/raw/<id>`, commit `data/snapshots/<id>/`.
- Optional entity layer, agent-driven: read a document, write `data/enrichment/<doc>.json` as
  `{doc_id, entities:[{name,type,description}], relations:[{source,target,type,evidence}]}`
  (types: person|company|product|framework|concept|book|metric|regulation|other; evidence is a
  short paraphrase), then
  `docker compose run --rm graphrag graphrag enrich-import <id> /app/data/enrichment/<doc>.json`.
  Unattended alternative with an API key: `make enrich PERSONA=<id> LIMIT=10`.

## 5. Where compute runs
Embeddings default to an in-process ONNX model. To use a GPU box (e.g. a GB10) run
`make embed-up` there (add `-f docker-compose.gpu.yml`), and set
`GRAPHRAG_EMBEDDING_BACKEND=http GRAPHRAG_EMBEDDING_BASE_URL=http://<host>:8766/v1` here.
The model must match the snapshot's manifest; `make doctor` checks.
