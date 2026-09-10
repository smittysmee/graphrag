"""Write a Claude Code skill that makes an agent assume a persona through the MCP server."""

from __future__ import annotations

from pathlib import Path

from graphrag.models import PersonaSpec


def skill_markdown(persona: PersonaSpec, brief: str) -> str:
    stages = ", ".join(persona.sdlc_stages) or "any"
    description = (
        f'Act as the "{persona.name}" persona, grounded in the graph-rag knowledge graph '
        f"(MCP server `graphrag`). Use during {stages} work, or whenever a question would "
        f"benefit from this persona's real sources. Triggers on phrases like "
        f'"as a {persona.name.lower()}", "what would {persona.name.lower()} say", "{persona.id}".'
    )
    return f"""---
name: persona-{persona.id}
description: {description}
---

# Persona: {persona.name}

This skill turns the `graphrag` MCP server into a grounded persona. Do not answer from memory:
every substantive claim must trace back to a retrieved passage.

## Setup (once per session)
1. Ensure the MCP server is reachable: call `stats` on the `graphrag` server. If it fails, run
   `make setup` in the graph-rag repo (starts Neo4j, loads the committed snapshot, starts MCP).
2. Call `persona_brief("{persona.id}")` and adopt it as your operating instructions.

## Answering
1. `context(query="<user question>", persona_id="{persona.id}")` — returns cited passages.
2. If the pack is thin, widen with `search(query, persona_id="{persona.id}", k=15)`, or browse
   `topics("{persona.id}")` / `documents(persona_id="{persona.id}", topic=...)`.
3. Read more of a promising source with `read_document(doc_id, start, count)`.
4. Answer in the persona's voice, citing [n] from the context pack. State clearly when the
   sources are silent.

## Brief
{brief}"""


def export_persona_skill(persona: PersonaSpec, brief: str, skills_dir: Path) -> Path:
    target = skills_dir / f"persona-{persona.id}" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(skill_markdown(persona, brief), encoding="utf-8")
    return target
