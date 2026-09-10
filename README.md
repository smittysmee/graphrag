# graph-rag

Persona-grounded Graph RAG you can drop into any project as an MCP server or CLI.

Ingest text → parse → chunk → embed → build a graph in **Neo4j Community** → query it through a
**FastMCP** server (`.mcp.json` is pre-wired for Claude Code) or the `graphrag` CLI.
Knowledge is organised into **personas**: a role prompt plus the real sources that ground it.

```bash
git clone https://github.com/smittysmee/graphrag && cd graphrag
make setup                              # images, model, Neo4j, MCP server on :8765

# then either import a persona someone sent you...
make persona-import FILE=their-persona.tar.gz
# ...or build one from source
make sources && make ingest PERSONA=product-leader SRC=data/raw/product-leader

make context Q="how do I know we have product-market fit" PERSONA=product-leader
```

> **No content ships with this repo.** You bring your own sources and build the graph locally.
> The included `product-leader` persona is an example configuration pointing at a public
> transcript archive that you fetch yourself. See [NOTICE.md](NOTICE.md) before redistributing
> anything you build with it.

**[Read the usage guide](docs/USAGE.md)** for the tools, workflows and troubleshooting.

## What you get

| Surface | How |
|---|---|
| MCP server | `http://localhost:8765/mcp` — tools `list_personas`, `persona_brief`, `context`, `search`, `documents`, `read_document`, `topics`, `related_topics`, `speakers`, `recommend_personas`, `cypher`, `stats`; prompt `assume_persona` |
| CLI | `graphrag setup / doctor / ingest / snapshot / search / context / persona / enrich / serve / stats` |
| Claude skills | `.claude/skills/graph-rag/` (platform) and `.claude/skills/persona-<id>/` (one per persona) |
| Portable graph | `data/snapshots/<persona>/` — documents, chunks, float32 embeddings, entities. Built locally; commit it only if you own the source content |

## Quick start

Requirements: Docker Desktop. Nothing else is installed on the host.

```bash
make init      # asks a few questions, writes .env, runs setup
make doctor    # sanity check
```

Ingesting the example archive takes a few hours on CPU, or minutes if you point the embedder at a
GPU box (see below). Everything is reproducible: the source archive is pinned as a git submodule
and the embedding model is pinned by revision and SHA-256 in `vendor/model.lock`. Once built, the
snapshot in `data/snapshots/` restores into an empty database in well under a minute, so the Neo4j
volume is disposable.

Then in Claude Code (this repo has `.mcp.json`): *"Use the graphrag MCP: adopt the
product-leader persona and tell me how to run a roadmap review."*

## Personas

```
personas/<id>/persona.yaml    role prompt, voice, sdlc_stages, sources, retrieval config
data/snapshots/<id>/          the ingested graph for that persona (committed)
.claude/skills/persona-<id>/  generated skill so an agent can assume the persona
```

- `product-leader` — an example persona wired to a public podcast archive. Ingest it to see the
  whole pipeline end to end: 291 unique episodes, ~13,300 passages, and an optional entity layer.
Add your own with `graphrag persona new`; see [PERSONAS.md](docs/PERSONAS.md).

Add one: `docker compose run --rm graphrag graphrag persona new "Name" -d "..."`, drop files under
`data/raw/<id>/<source>/`, `make ingest PERSONA=<id> SRC=data/raw/<id>`, commit the snapshot.

## Where the compute runs

Embeddings are the only heavy step. By default an ONNX model (`BAAI/bge-small-en-v1.5`, 384-d)
runs inside the container: free, offline, nothing leaves your machine. To run it on a GPU box on
your network (e.g. a GB10):

```bash
# on the GPU box
EMBED_GPU=1 GRAPHRAG_EMBEDDING_CUDA=true \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile embed up -d --build embed

# on your machine (.env)
GRAPHRAG_EMBEDDING_BACKEND=http
GRAPHRAG_EMBEDDING_BASE_URL=http://gb10.local:8766/v1
```

The `http` backend speaks the OpenAI `/v1/embeddings` format, so Ollama, vLLM,
text-embeddings-inference or llama.cpp work too. Query vectors must come from the same model as
the snapshot; `make doctor` refuses mismatches.

## Development

Everything runs in Docker; `make hooks` (done by `make setup`) enables versioned git hooks:
pre-commit runs ruff + mypy --strict + unit tests, pre-push adds Neo4j integration tests.

```bash
make check              # what the pre-commit hook runs
make test-integration   # what the pre-push hook runs
make shell              # bash in the dev container
make lsp                # python-lsp-server over stdio for your editor
```

## Docs

| Doc | What's in it |
|---|---|
| [USAGE.md](docs/USAGE.md) | **Start here.** Tools, workflows, CLI, troubleshooting |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Pipeline, graph schema, snapshot format |
| [PERSONAS.md](docs/PERSONAS.md) | Persona spec and lifecycle |
| [SDLC.md](docs/SDLC.md) | Which persona applies at which stage |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Testing conventions, LSP setup |
| [NOTICE.md](NOTICE.md) | Attribution, third-party licenses, what you may redistribute |

## License

MIT for this project's code — see [LICENSE](LICENSE). Ingested content is **not** covered by it;
read [NOTICE.md](NOTICE.md) first.

## Optional: entity enrichment

Enrichment adds `Entity` nodes (people, companies, products, frameworks, concepts, books,
metrics, regulations) with `MENTIONS` edges from passages and `RELATED_TO` edges between entities.
The deterministic graph (documents, passages, speakers, topics, topic co-occurrence) never needs it.

Two ways to produce it:

- **Agent-driven (no API key).** Any agent, including Claude Code, reads a transcript and writes
  `data/enrichment/<doc>.json` (`{doc_id, entities[], relations[]}`; see `DocumentExtraction` in
  `src/graphrag/extract/llm.py`). Then:
  `docker compose run --rm graphrag graphrag enrich-import product-leader /app/data/enrichment/*.json`
  which anchors mentions to passages by name match, writes the graph, and re-exports the snapshot.
- **Unattended (API key).** `make enrich PERSONA=product-leader LIMIT=10` calls `claude-opus-5`
  with structured outputs; needs `ANTHROPIC_API_KEY` in `.env`.

