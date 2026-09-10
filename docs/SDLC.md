# Where graph-rag sits in the SDLC

graph-rag is **knowledge infrastructure**, not a stage-specific tool. The MCP server runs
continuously in the development environment; *personas* are what get switched on per stage.
"A domain persona" below means one you define for your own field, grounded in your own documents.

| Stage | What agents use it for | Personas (`sdlc_stages`) |
|---|---|---|
| Discovery | Interview synthesis, "what do experienced operators say about X", opportunity framing | product-leader |
| Requirements / PRD | Grounded PRD sections, acceptance criteria drawn from real domain rules, terminology | product-leader, a domain persona |
| Prioritisation / Roadmap | RICE-style arguments with cited precedent, retention-vs-growth trade-offs | product-leader |
| Design review | Reviewer persona that challenges the problem statement and onboarding flows | product-leader |
| Implementation | `context()` inline for domain questions while coding; `cypher()` for data questions | any |
| Testing | Domain personas author realistic scenarios and edge cases from real rules | a domain persona |
| Release / Compliance | Checks against cited policy or regulatory guidance | a domain persona |
| Operations / Support | Support-answer drafting with citations to source documents | a domain persona |
| Retrospective | "What would a seasoned PM ask about this launch" | product-leader |

## When to expose it

- **Always on** in dev: `make up` starts Neo4j and the MCP server; `.mcp.json` wires Claude Code.
- **Per stage**: an agent calls `recommend_personas(stage)` (or `graphrag persona recommend`) and
  loads the matching skill. Stages are declared in each `persona.yaml`.
- **In CI**: not required. The graph is data; CI only verifies code (`make check`) and, on push,
  the Neo4j integration tests.

## Growing it

Each new persona is a folder + a committed snapshot. The team pulls, runs `make setup`, and the new
persona is available to every agent. Enrichment (`make enrich`) is an optional, paid layer that
adds entities/relations for graph-style questions ("which operators discuss both pricing and
onboarding?").
