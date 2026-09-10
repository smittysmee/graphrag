# How to use graph-rag

A persona is a role prompt plus the real sources that ground it. You ask a question, the system
retrieves the passages that actually answer it, and the agent replies in that persona's voice with
citations you can click through to the source.

This repo ships **no content**. It ships the machinery plus one example persona, **Product
Leader**, wired to a public podcast archive you fetch yourself. Ingesting it yields roughly 291
episodes and 13,300 passages, with an optional entity layer on top. Read [NOTICE.md](../NOTICE.md)
before redistributing anything you build.

---

## 1. First run

You need Docker Desktop. Nothing else is installed on your machine.

```bash
git clone https://github.com/smittysmee/graphrag
cd graphrag
make init
```

`make init` asks at most three things — what you want to do first, where embeddings should run,
and how the model should get here — then writes `.env` and runs the rest. It skips any question it
can answer itself: no download question when the cache is already warm or you chose a remote
embedder. `./scripts/init.sh --yes` takes every default silently, for scripted installs.

The `.env` it writes is rendered from the pydantic settings models, so its keys cannot drift from
what the app actually reads. `graphrag config show` prints the effective configuration with
secrets redacted; `graphrag config init` writes one directly.

To drive it manually instead, copy `.env.example` to `.env` and run `make setup`.

`make ingest` is the slow part: chunking and embedding the archive takes a few hours on CPU, or
minutes against a GPU box (see section 6). You only do it once. It writes a snapshot to
`data/snapshots/`, and reloading that into an empty database later takes well under a minute.

**Nothing downloads model weights unless you ask.** `make setup` only reports whether the pinned
model is present. If it is missing, setup tells you how to supply it and carries on.

At runtime the guarantee is enforced, not merely deferred: the fastembed backend marks the Hugging
Face stack offline and refuses to construct a model that is not already cached, failing with
instructions rather than fetching silently mid-query. `GRAPHRAG_EMBEDDING_ALLOW_DOWNLOAD` defaults
to `false`.

Three ways to get the model onto a machine, in order of least network:

```bash
make persona-import FILE=bundle.tar.gz   # a bundle exported --with-model carries it
make model-import FILE=model.tar.gz      # a cache tarball from `make model-export`
make model                               # explicit opt-in: ~64 MB from Hugging Face
```

`make model` is the only command that will ever fetch it, and only because you typed it. The check
also skips itself entirely when `GRAPHRAG_EMBEDDING_BACKEND` is not `fastembed`, since an `http`
backend embeds on another machine and never touches a local model.

Check it:

```bash
make doctor     # Neo4j reachable, embedder placement, snapshot compatibility
make stats      # node counts, overall and per persona
```

Two useful side doors: the Neo4j browser at `http://localhost:7474` (user `neo4j`, password from
your `.env`), and `make logs` to tail the server.

---

## 2. Using it from Claude Code

This is the main path. The server is registered under the name `graphrag`, and four skills are
installed, so you rarely need to name a tool yourself.

**Just ask in words.** Phrasing that mentions the persona pulls the right skill:

> As a product leader, how should we sequence a roadmap when two bets both look urgent?

> What do experienced operators say about knowing you have product-market fit?

> Ground this PRD section in real practice and cite your sources.

Behind the scenes the agent adopts the persona brief, calls the context tool, and answers from
retrieved passages with `[n]` citations. Each citation carries the speaker, the episode, and a
timestamped link to the exact moment in the source video.

**When you want to steer the retrieval yourself**, name the tool:

> Use the graphrag `search` tool for "pricing experiments", persona product-leader, k=15.

**If the server is not responding**, the agent will tell you. Run `make up` in the repo and retry.

---

## 3. The tools

Thirteen tools, in the order you would normally reach for them.

### Orienting

| Tool | Use it to |
|---|---|
| `list_personas` | See which personas exist, their sources and SDLC stages |
| `recommend_personas` | Ask which persona fits a stage: discovery, requirements, testing, compliance |
| `persona_brief` | Get the role prompt plus live grounding stats; adopt this before answering |
| `stats` | Node counts, per-persona breakdown, and which embedder is configured |

### Answering a question

| Tool | Use it to |
|---|---|
| `context` | **The main one.** Returns prompt-ready cited passages for a question |
| `search` | Raw hybrid search when you want more or fewer hits, or a specific mode |

`context` accepts a query, an optional persona, and optional `k` and `expand`. It returns both a
formatted prompt and structured hits, so an agent can either paste it or reason over the parts.
`expand` pulls in neighbouring passages, which helps when an answer straddles a topic change.

`search` takes a `mode`: `hybrid` by default, or `vector` for pure semantic similarity, or
`fulltext` for exact keyword matching. Reach for `fulltext` when you know the phrase, and `vector`
when you only know the idea.

### Reading the corpus

| Tool | Use it to |
|---|---|
| `documents` | List episodes, filtered by topic or speaker |
| `get_document` | Full metadata for one episode |
| `read_document` | Page through an episode's passages in order |
| `topics` | Topics ranked by how many episodes carry them |
| `related_topics` | What co-occurs with a topic, strongest first |
| `speakers` | Voices ranked by episode count |

`read_document` is how you go deeper after a promising hit, rather than dumping a whole transcript
into context. It pages by passage ordinal.

### The escape hatch

`cypher` runs read-only graph queries when the shaped tools do not fit. Writes are rejected and
results are capped. It is the right tool for questions about structure rather than content:

```cypher
// Which operators discuss both pricing and onboarding?
MATCH (d:Document)-[:ABOUT]->(t:Topic)
WHERE t.name IN ['pricing', 'onboarding']
WITH d, count(DISTINCT t) AS hits WHERE hits = 2
RETURN d.title, d.speakers LIMIT 20

// Most cross-cited entities
MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
RETURN e.name, e.type, count(DISTINCT c.doc_id) AS episodes
ORDER BY episodes DESC LIMIT 15
```

The graph's shape: `Persona` is grounded by `Source`, which contains `Document`, which has
`Chunk`. Chunks link to the next chunk in sequence, to the `Speaker` who said them, and to any
`Entity` they mention. Documents link to `Topic`, and topics co-occur with weights. Entities relate
to each other with a type and short evidence.

---

## 4. Using it from the command line

Everything the tools do is also a command, which is handy for scripting and for agents without MCP.

```bash
# a cited context pack, printed as prompt-ready text
make context Q="how do I know we have product-market fit" PERSONA=product-leader

# raw search, five hits
make search Q="roadmap review" PERSONA=product-leader K=5

# machine-readable
docker compose run --rm graphrag graphrag search "pricing" -p product-leader -k 5 --json
docker compose run --rm graphrag graphrag context "retention" -p product-leader --json

# personas
docker compose run --rm graphrag graphrag persona list
docker compose run --rm graphrag graphrag persona brief product-leader
docker compose run --rm graphrag graphrag persona recommend discovery
```

---

## 5. Common workflows

### Research a question properly

Start with `context`. If the passages feel thin, widen with `search` at a higher `k`, or come at it
sideways: find the topic with `topics`, list its episodes with `documents`, then read the promising
one with `read_document`. Ending with a `cypher` query is often how you notice a pattern across
episodes that no single passage states.

### Add a new persona

```bash
docker compose run --rm graphrag graphrag persona new "Support Engineer" -d "Answers from our runbooks"
```

That scaffolds `personas/<id>/persona.yaml` and a README with the runbook. Then:

1. Put documents under `data/raw/<id>/<source>/`. Markdown, plain text and PDF are supported.
2. Add a `sources:` entry to `persona.yaml`, choosing the `documents` loader for prose or
   `transcripts` for timestamped speaker turns.
3. `make ingest PERSONA=<id> SRC=data/raw/<id>`
4. Commit `personas/<id>` and `data/snapshots/<id>` so teammates get it from `make setup`.
5. `docker compose run --rm graphrag graphrag persona export-skill <id>` writes the Claude skill.

Fill in the persona's `role_prompt`, `voice` and `sdlc_stages` by hand. Those are what make the
persona sound like someone rather than like a search engine.

### Add the entity layer

Entities are optional. Retrieval works without them; they earn their place when you want to ask
graph questions or keep an agent in the persona's vocabulary.

Ask Claude Code to "enrich the remaining episodes for `<persona>`". The `graph-rag-enrich` skill
handles it: agents read each document and write one JSON file per document, then

```bash
docker compose run --rm graphrag graphrag enrich-import <persona> /app/data/enrichment/*.json --dry-run
docker compose run --rm graphrag graphrag enrich-import <persona> /app/data/enrichment/*.json
```

The dry run reports names that do not occur in any passage and relations pointing at undeclared
entities, so you can fix the JSON before writing. Import is idempotent, so re-running everything is
cheap and safe.

There is also an unattended path that calls the API directly, `make enrich PERSONA=<id> LIMIT=10`,
which needs `ANTHROPIC_API_KEY` in `.env`. The agent path needs no key and was used for the
existing corpus.

### Share a persona with someone

A persona is the unit you hand over: the role prompt, its sources, the built graph and the entity
layer, in one file.

```bash
make persona-export PERSONA=product-leader            # → data/exports/product-leader-persona.tar.gz
make persona-export PERSONA=product-leader COMPACT=1 WITH_MODEL=1
```

Bundles land in `data/exports/`, which is gitignored. (Only `./data` is mounted into the
container, so a bundle written anywhere else would vanish when the command exits; the CLI warns
if you try.)

It prints what went in and which embedder built it. Send the file however you like. On their side:

```bash
make persona-import FILE=product-leader-persona.tar.gz
```

That unpacks the persona, restores the graph into their Neo4j, and writes the Claude skill so the
persona is immediately usable. On a real 291-episode graph the whole import takes about 40 seconds,
against roughly four hours to rebuild it from source.

Options worth knowing:

| Flag | Effect |
|---|---|
| `--no-graph` | Ship the definition only; the recipient ingests their own copy of the sources |
| `--no-enrichment` | Leave out the entity-extraction JSON |
| `--compact` | Store vectors as float16: about a third smaller, no measurable ranking change |
| `--with-model` | Also carry the embedding model, making the bundle fully self-contained |
| `--notes "..."` | A short message shown to the recipient on import |
| `--overwrite` | On import, replace a persona of the same id that already exists |
| `--load` | On import, load into Neo4j immediately instead of just unpacking |

Or via make: `make persona-export PERSONA=id` and `make persona-import FILE=bundle.tar.gz`.

**About size.** Bundles are gzipped already, but most of the weight is vectors, and vectors barely
compress: they are high-entropy floats, so gzip only reaches about 90% of raw and xz is no better
for nine times the CPU. Halving their width is worth far more than any codec, which is what
`--compact` does by storing float16. Measured over 300 real queries against the example persona,
the top result never changed and worst-case top-10 overlap was 9 of 10, for a cosine delta around
2e-4. Neo4j widens them back on load.

| Export | Size |
|---|---|
| default | 29 MB |
| `--compact` | 20 MB |
| `--compact --with-model` | 81 MB |
| `--with-model` | 86 MB |

**About the embedding model.** The graph's vectors are meaningless without the model that produced
them, so import refuses a mismatch up front rather than querying a foreign vector space. By default
the bundle leaves the model out and the recipient runs `make model`, which fetches the pinned build
and verifies its checksum. That keeps bundles small, and one cached model serves every persona.

Add `--with-model` when the recipient has no network, or when you want an archival copy that will
still work if the model ever disappears upstream. It grows the bundle by about 63 MB (roughly 29 MB
to 86 MB for the example persona) and makes it completely self-contained: import restores the model
into their cache, skipping anything they already have. Verified end to end into an empty cache, with
the restored blob matching the pinned SHA-256.

Two things a bundle never carries: **Docker images**, since `make setup` builds those, and **the raw
source files**, because the graph already holds what retrieval needs.

One caution: a bundle with a graph contains your source text. Only share graphs built from content
you own or are licensed to redistribute — see `NOTICE.md`. Use `--no-graph` to share just the
persona design.

### Refresh a source

```bash
git submodule update --remote data/raw/product-leader
make ingest PERSONA=product-leader SRC=data/raw/product-leader
git add data/raw/product-leader data/snapshots/product-leader && git commit
```

Re-ingesting replaces a document rather than merging into it, so edits and deletions land cleanly.

---

## 6. Where the compute runs

Embedding is the only expensive step. By default it runs inside the container on CPU: free,
offline, and nothing leaves your machine. Building the existing corpus took about three and a half
hours that way. Querying is unaffected, since each query embeds a single short string.

To move it to a GPU box on your network, run the server there:

```bash
# on the GPU host (e.g. a GB10)
EMBED_GPU=1 GRAPHRAG_EMBEDDING_CUDA=true \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile embed up -d --build embed
```

and point this machine at it in `.env`:

```
GRAPHRAG_EMBEDDING_BACKEND=http
GRAPHRAG_EMBEDDING_BASE_URL=http://gb10.local:8766/v1
```

The wire format is the OpenAI embeddings API, so Ollama, vLLM, text-embeddings-inference and
llama.cpp all work as alternatives. `make doctor` reports the active backend and round-trip
latency.

One hard rule: query vectors must come from the same model as the snapshot. The manifest records
the model and dimension, and loading refuses a mismatch rather than silently returning nonsense.

---

## 7. Troubleshooting

**The agent says the server is unavailable.** Run `make up`. Both containers restart automatically
after a reboot, but not if Docker itself is stopped.

**`make doctor` reports an embedder mismatch.** Your configured model differs from the snapshot's.
Either restore the pinned values from `.env.example`, or re-ingest with the new model.

**Search returns nothing.** Check `make stats` shows a non-zero chunk count. If it is zero, the
graph is empty: `docker compose run --rm graphrag graphrag snapshot load --all`.

**A commit is rejected.** The pre-commit hook runs formatting, linting, strict type checking and
unit tests in Docker, and blocks files over 45 MB. Run `make check` to see the failure, `make fmt`
to auto-fix formatting.

**The graph seems corrupted.** Delete it and reload; the database holds nothing that is not in the
repo. `docker compose down && docker volume rm graph-rag_neo4j_data && make setup`.

**Air-gapped machine.** Carry the model across: `make model-export FILE=model.tar.gz` on a
connected machine, then `make model-import FILE=model.tar.gz` on the target.

---

## 8. Where it fits your work

The server is knowledge infrastructure, available in every stage. Personas are what you switch.
`docs/SDLC.md` maps stages to personas in detail; the short version is that the product persona
covers discovery, requirements, prioritisation, design review and retros, while a domain persona
covers requirements, test scenarios, compliance review and support content.

Ask `recommend_personas` with a stage name when you are not sure which to reach for.
