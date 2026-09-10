# Personas

A persona is the unit of knowledge sharing: `personas/<id>/persona.yaml` + a committed snapshot.

```yaml
id: product-leader
name: Product Leader
description: one paragraph shown in list_personas
role_prompt: |          # becomes the system-style instruction in persona_brief / assume_persona
  You are ...
voice: [short style bullets]
sdlc_stages: [discovery, requirements, ...]   # see graphrag persona recommend
sources:
  - id: lennys-podcast
    kind: git | local
    url: ...                # informational for git sources
    path: episodes          # sub-folder under the ingest root
    loader: transcripts | documents
    glob: "*/transcript.md"
retrieval: {top_k: 8, expand_neighbors: 1, mode: hybrid}
```

## Lifecycle
1. `graphrag persona new "Name"` scaffolds the folder and a README runbook.
2. Put source files under `data/raw/<id>/...` (not committed) and list them in `sources:`.
3. `make ingest PERSONA=<id> SRC=data/raw/<id>` → graph + `data/snapshots/<id>/`.
4. Optionally add entities/relations: have an agent write `data/enrichment/<doc>.json` and run
   `graphrag enrich-import <id> ...` (or `make enrich PERSONA=<id>` with an API key).
5. Commit `personas/<id>` and `data/snapshots/<id>`; teammates get it via `make setup`.
6. `graphrag persona export-skill <id>` regenerates `.claude/skills/persona-<id>/SKILL.md`.

## How an agent assumes a persona
`persona_brief(id)` returns the role prompt, voice, live grounding stats (documents, strongest
topics, most-cited voices) and a four-step answering protocol. `context(query, id)` then supplies
cited passages; the agent answers in-voice with `[n]` citations and says when sources are silent.

## Bundled personas
- `product-leader` — the worked example. Everything else is yours to define.
