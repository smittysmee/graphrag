# CLAUDE.md

Persona-grounded Graph RAG: text → chunks → embeddings → Neo4j graph → FastMCP server + CLI.
Snapshots in `data/snapshots/` make a built graph portable: commit one only if you own the source
content (see NOTICE.md), otherwise each user ingests locally.

## Commands (all run in Docker)
- `make setup` — first-time: hooks, images, Neo4j, load snapshots, MCP on :8765
- `make check` — ruff format check, ruff lint, mypy --strict, unit tests (this is the pre-commit hook)
- `make test-integration` — against a throwaway Neo4j (pre-push hook)
- `make sync PERSONA=<id>` — ingest what is missing and re-import that source's enrichment JSON
- `make ingest PERSONA=<id> SRC=<path> [SOURCE=<id>]` / `make enrich` / `make search Q="..."` / `make context`
- `make doctor` — Neo4j, embedder placement (local vs http), snapshot compatibility

## Layout
`src/graphrag/` — `config.py` (pydantic-settings, the only place env is read), `models.py`,
`ingest/` (loaders + chunker), `embed/` (Embedder Protocol: fastembed | http | hash, plus the
embed server), `graph/` (GraphStore Protocol: neo4j_store, memory_store, snapshot),
`extract/` (topic co-occurrence; optional Claude enrichment), `retrieve/` (hybrid RRF search,
context packs), `personas/` (registry, brief, skill export), `pipeline.py`, `cli.py`,
`mcp_server.py`, `app.py` (composition root).
`hooks/` (host-side Claude Code hooks, stdlib only; shell wrappers in `.claude/hooks/`).

## Data facts worth knowing
- `product-leader` is grounded in the ChatPRD Lenny's Podcast archive, pinned as a git submodule
  at `data/raw/product-leader`. It has **303 folders but only 291 unique episodes**: 12 are the
  same recording archived twice. The loader drops them by hashing the body text, so re-ingesting
  is safe and the count will not drift.
- Roughly 18 transcripts carry **another guest's front-matter title**. The loader detects a
  "| Other Name" credit that does not match the guest, substitutes "<guest> on <channel>", and
  keeps the original in `metadata.archive_title`. Trust the body, not the front-matter.
- Some transcripts misspell names ("Ethan Malik" for Ethan Mollick). Extraction deliberately uses
  the **transcript's surface form** so mentions anchor to real passages, with the correction noted
  in the entity description.
- The entity layer is built by Claude Code subagents via the `graph-rag-enrich` skill rather than
  the API. Each run writes one JSON file per document under `data/enrichment/`, which is the
  source of truth and can be re-imported at any time.

## Rules
- Config only via `graphrag.config.Settings`; never read `os.environ` elsewhere.
- Dependencies are injected (`AppContext`, Protocols). Tests use fixtures and fakes
  (`InMemoryGraphStore`, `HashEmbedder`, `httpx.MockTransport`, fake extraction client) — never
  `unittest.mock.patch`.
- Modify files in place; do not create `_v2`/`_enhanced` copies.
- Every Neo4j statement runs on its own (no `;`-batches). Schema is idempotent (`IF NOT EXISTS`).
- Snapshot manifests record the embedding model; loading with a different model must fail loudly.
- Keep individual snapshot files under 45 MB (pre-commit enforces this).
- The embedding model is pinned in `vendor/model.lock`. Changing it invalidates every committed
  snapshot, so re-ingest before shipping such a change.
- Never `docker compose down -v` on the default profile: it destroys the Neo4j volume and the
  model cache. Use `make down`. (Losing the graph is recoverable via `make setup`; it is just slow.)
- Enrichment: prefer the agent-driven path (`graphrag enrich-import` with JSON the agent wrote);
  the SDK path (`graphrag enrich`, `claude-opus-5`, structured outputs, refusal fallbacks) is for
  unattended runs and stays behind `ExtractionClient`.
