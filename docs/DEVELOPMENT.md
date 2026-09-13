# Development

All tooling runs in Docker; the host needs only Docker Desktop and git.

| Task | Command |
|---|---|
| Quality gate (pre-commit) | `make check` → ruff format --check, ruff check, mypy --strict, pytest |
| Integration tests (pre-push) | `make test-integration` (throwaway Neo4j on tmpfs) |
| Auto-format | `make fmt` |
| Shell | `make shell` |
| Enable hooks | `make hooks` (sets `core.hooksPath=.githooks`) |

## Testing conventions
- **Fixtures over patching.** `tests/conftest.py` provides `sample_corpus`, `persona`,
  `hash_embedder`, `memory_store`, `pipeline`, `ingested`, `retriever`, `settings`, `registry`,
  `app_context`, `cli_context`. External systems are replaced by injected fakes:
  `InMemoryGraphStore`, `HashEmbedder`, `httpx.MockTransport`, a fake `ExtractionClient`.
- Integration tests are marked `integration` and skip themselves when Neo4j is unreachable.
- `embedding`-marked tests would download the real model; none exist by default.

## LSP
The dev image includes `python-lsp-server` with `pylsp-mypy` and `python-lsp-ruff`.
Editors that spawn a stdio language server can use:

```
command: docker
args: ["compose", "--profile", "dev", "run", "--rm", "-i", "lsp"]
```

(`make lsp` runs the same.) Ruff's own server is also available: `docker compose --profile dev run
--rm -i dev ruff server`. Paths inside the container are `/app/src` and `/app/tests`; mount-aware
editors (VS Code Dev Containers, JetBrains Docker interpreters) map them automatically.

## Adding a store or embedder
Implement the Protocol (`graph/store.py`, `embed/base.py`), register it in `AppContext.build` /
`build_embedder`, and add it to the fixture matrix. The in-memory store doubles as the contract:
integration tests run the same scenarios against Neo4j.

## Claude Code hooks
`.claude/settings.json` wires three hooks that keep an agent session aware of the graph. Each is
a thin `.claude/hooks/*.sh` wrapper around a module in `src/graphrag/hooks/`.

- **`SessionStart` → `session-card.sh`** prints a status card: every persona with its document
  and chunk counts, whether the MCP server answers, the five newest files under `data/raw/`, and
  how many of them the graph does not have yet.
- **`UserPromptSubmit` → `route-prompt.sh`** detects that a prompt belongs to a persona and
  injects a one-paragraph nudge to call `context(query, persona_id=...)` first, plus up to three
  matching document titles. Routing only: it never injects passages.
- **`Stop` → `uningested.sh`** names any files under `data/raw/` that the graph does not contain
  and tells you to run `make sync PERSONA=<id>`, which is the one command that fixes it: it
  re-ingests only the sources that are behind and re-imports their extraction JSON.

**The router has no keyword list.** Vocabulary is derived from the graph at runtime: the persona
id and name (weight 3), its `tags` (2), and its top 60 topics from the MCP `topics` tool (1).
Topics are cached per persona under `data/cache/hooks/topics-<persona>.json` with a 24 hour TTL,
so a live prompt never waits on the server; a stale cache is used when the server is down. The
cache is gitignored. Adding a persona or re-running extraction changes routing on its own.

**Host requirements.** Unlike everything else here, hooks run on the host, not in Docker. They
need only a `python3` on PATH (3.10+) and use the standard library only, so
`src/graphrag/hooks/` and anything it imports must stay dependency-free — that is why `slugify`
lives in `src/graphrag/textutil.py` rather than with the loaders. The wrappers exit 0 when no
`python3` is found. The MCP server is reached over plain `urllib` at
`http://localhost:8765/mcp` (`GRAPHRAG_MCP_PORT` overrides the port).

**Fail silent.** Every hook wraps `main()` in a catch-all and returns 0 on any error, so a broken
hook, a missing snapshot or a down server degrades the output instead of blocking the session.
Network calls time out after 2 s; with the server down a hook finishes in well under 1 s and
falls back to the committed snapshot, labelling the numbers as such. No hook ever blocks a turn.

**Turning them off.** Set `"disableAllHooks": true` in `.claude/settings.local.json` (untracked),
or delete the entry from `.claude/settings.json`. For a single run, pass
`--settings '{"disableAllHooks": true}'`.

A subagent can pull the same card into its own context by declaring a `SubagentStart` hook in its
frontmatter that runs `session-card.sh --json`; the wrapper echoes the incoming `hook_event_name`
back as `hookSpecificOutput.hookEventName`, so the one script serves both events.
