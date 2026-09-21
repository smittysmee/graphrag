# CLAUDE.md

Persona-grounded Graph RAG: text → chunks → embeddings → Neo4j graph → FastMCP server + CLI.
Snapshots in `data/snapshots/` make a built graph portable: commit one only if you own the source
content (see NOTICE.md), otherwise each user ingests locally.

## Commands (all run in Docker)
- `make setup` — first-time: hooks, images, Neo4j, load snapshots, MCP on :8765
- `make check` — ruff format check, ruff lint, mypy --strict, unit tests (this is the pre-commit hook)
- `make test-integration` — against a throwaway Neo4j (pre-push hook)
- `make sync PERSONA=<id> [REFRESH=1]` — ingest what is missing and re-import that source's
  sidecars; `REFRESH=1` re-reads every extraction, attribution and annotation file, which is how
  a rewritten sidecar (node attributes, say) reaches a graph that already has that layer, and
  first clears the persona's stored attribute values so a corrected file replaces the old one;
  it takes an advisory per-persona lock first, so a second concurrent sync refuses instead
  of racing the first (`--force-lock` overrides a lock that looks abandoned)
- `make ingest PERSONA=<id> SRC=<path> [SOURCE=<id>]` / `make enrich` / `make search Q="..."` / `make context`
- `make doctor` — Neo4j, embedder placement (local vs http), snapshot compatibility
- `make bench` — the `slow` timing budget against the largest loaded persona (`make check` skips it)

## Layout
`src/graphrag/` — `config.py` (pydantic-settings, the only place env is read), `models.py`,
`ingest/` (loaders + chunker), `embed/` (Embedder Protocol: fastembed | http | hash, plus the
embed server), `graph/` (GraphStore Protocol: neo4j_store, memory_store, snapshot),
`extract/` (topic co-occurrence; optional Claude enrichment; `attributes.py` holds the node
attribute vocabulary a persona declares in `facets.yaml`), `retrieve/` (hybrid RRF search,
context packs), `personas/` (registry, brief, skill export), `sna/` (network export,
centrality, Louvain/K-means/GMM with stability and null-model checks, plus `attributes.py`:
whether a network divides along a node attribute, against a permutation and a rewiring null,
and whether the labels were independent of the edges in the first place; `stats.py`,
`matrices.py` and `null.py` are the primitives every later test calls -- named null models
under one `significance()` contract, ERGM by pseudo-likelihood; `sampling.py` and `layers.py`
hold the Atlas ch. 29 samplers and the ch. 7 multilayer, hypergraph and dynamic readings;
`backbone.py` the ch. 27 filters, `walks.py` the ch. 11 random-walk quantities, `degree.py`
the ch. 9 degree distributions and the power-law test, `paths.py` ch. 10 and §13.1–13.3,
`projection.py` the ch. 26 weightings, `uncertain.py` the ch. 28 edge probabilities,
`generators.py` the ch. 16–18 synthetic graphs, `roles.py` the ch. 15 roles and equivalences;
`measures.py` carries the ch. 12 density block and the ch. 14 rankings and centralization;
`assortativity.py` the ch. 31 numeric assortativity; `ego.py` and `ties.py` the ch. 30
homophily readings -- ego networks, EI/Coleman index, weak-tie bridging and the contagion
caveat; `hierarchy.py` the ch. 33 flow hierarchy -- type classification, cycle share, global
reach centrality, maximum spanning arborescence, exact agony -- refusing an undirected network;
`highorder.py` the ch. 34 simplicial complex and memory (second-order) network readings;
`experiment.py` the ch. 25 holdouts and link-prediction evaluation every prediction reports
through; `evaluate.py` the ch. 36 battery that scores every partition a report produces; `robustness.py`
the ch. 22 removal curves, cascades and interdependent coupling; `motifs.py` the ch. 41 triad
census, motif profile and frequent-subgraph mining; `spread.py` the ch. 20–21 epidemic and
complex-contagion what-ifs, immunisation and driver nodes; `coreperiphery.py` the ch. 32
coreness, rich club, nestedness and the tension check every grouping report runs; `cluster.py`
also carries the ch. 35 partitions -- blockmodels, map equation, Walktrap, label propagation,
temporal matching, local communities -- and the ch. 37 dendrograms, HRG fit and directed communities; `overlap.py` the ch. 38
overlapping covers; `multilayer.py` the ch. 40 multilayer communities over `layers.py`;
`bipartite.py` the ch. 39 two-mode community discovery; `predict.py` the ch. 23 link-prediction
scorers and the hypotheses they rank, evaluated through `experiment.py` first, plus the ch. 24
signed and cross-layer readings; `embed.py` the ch. 42 embedding contract, spectral embedding and
pooling and the ch. 43 random-walk embeddings; `draw.py` the ch. 49–51 drawings, SVG and HTML; `vectordist.py` the ch. 47 distances between two
node vectors on one network; `summarize.py` the ch. 46 meta-networks, compression and influence
summaries; `topodist.py` the ch. 48 distances between two networks; `gnn.py` the ch. 44–45 graph convolutions
and attention behind the optional `gnn` extra, never a mandatory dependency; `facade.py` the notebook `Network`
over the same functions the CLI calls; `cache.py` the per-persona network cache every `sna`
command reads through, cleared by every write path; `provenance.py` the block every report ends
with and `--cite`; `arguments.py` the one parser for the filter arguments the CLI and the MCP tools
share),
`pipeline.py`, `cli.py`,
`mcp_server.py` (the search and context tools, and every `sna` command as `sna_<command>`
returning markdown and payload), `app.py` (composition root).
`hooks/` (host-side Claude Code hooks, stdlib only; shell wrappers in `.claude/hooks/`).
`docs/ATLAS_PLAN.md` is the programme that takes `sna/` to full coverage of *The Atlas for the
Aspiring Network Scientist*: one ticket per chapter, waves, and the orchestration (Opus
`atlas-implementer` / `atlas-reviewer` agents, briefed from `docs/templates/ATLAS_TICKET.md`).
`scripts/atlas_chapter.py <n>` prints a chapter of the book, which is the source of truth for
every ticket.
`docs/ATLAS_INDEX.md` is generated by `scripts/atlas_index.py` (one row per chapter and section to
the modules, commands and rules that cite it; a test holds it verbatim) and `docs/SNA_GLOSSARY.md`
maps the book's glossary to the commands.
`extract/layers.py` + `graphrag layers check <persona> [--all|--doc-id|--file]` reports which
sidecars a captured document has on disk, which layers the graph holds, and what each importer
would leave loose; the `PostToolUse` capture-contract hook names the same contract after a write.

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
- Node attributes are written by the sidecar that owns the node: an extraction file says what an
  **entity** is, an attribution file what a **speaker** is, an annotation file what a
  **document** is. Speaker and entity nodes are shared between personas, so both are stored
  per-persona, first value written wins, and a second file that disagrees is reported rather
  than allowed to overwrite.
- An entity's attributes are a property of the thing, so they hold in every document that names
  it. That is what makes SNA over entities possible: an entity with no value of its own borrows
  the majority of its documents, and since the edges come from those same documents, any
  assortativity that follows is circular — the label shuffle inflates it rather than catching
  it. `sna analyze --by` records the provenance per node and refuses the verdict when most
  labels were borrowed, so tag entities in the extraction files rather than reading the number.
