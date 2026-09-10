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
