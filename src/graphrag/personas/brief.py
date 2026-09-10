"""A persona brief = the system-prompt-sized text an agent reads to *assume* the persona."""

from __future__ import annotations

from graphrag.graph.store import GraphStore
from graphrag.models import PersonaSpec


def build_brief(persona: PersonaSpec, store: GraphStore | None = None, *, top: int = 12) -> str:
    lines: list[str] = [f"# {persona.name}", ""]
    if persona.description:
        lines += [persona.description.strip(), ""]
    lines += ["## Role", persona.role_prompt.strip() or "(no role prompt set)", ""]
    if persona.voice:
        lines += ["## Voice", *[f"- {v}" for v in persona.voice], ""]
    if persona.sdlc_stages:
        lines += ["## Use in the SDLC", "Stages: " + ", ".join(persona.sdlc_stages), ""]

    lines += ["## Grounding"]
    for src in persona.sources:
        where = src.url or src.path or src.id
        lines.append(f"- {src.id}: {src.description or where} ({src.loader})")
    if store is not None:
        stats = store.stats().per_persona.get(persona.id, {})
        docs = stats.get("documents", 0)
        chunks = stats.get("chunks", 0)
        lines.append(f"- Indexed: {docs} documents, {chunks} passages")
        topics = store.list_topics(persona.id, limit=top)
        if topics:
            lines.append(
                "- Strongest topics: " + ", ".join(f"{t.topic} ({t.count})" for t in topics)
            )
        speakers = store.list_speakers(persona.id, limit=top)
        if speakers:
            lines.append("- Most-cited voices: " + ", ".join(s.speaker for s in speakers))
    lines += [
        "",
        "## How to answer as this persona",
        "1. Call `context(query, persona_id)` before answering anything substantive.",
        "2. Reason from the returned passages; quote or paraphrase with a citation [n].",
        "3. Prefer the persona's own vocabulary and frameworks over generic advice.",
        "4. When the sources do not cover the question, say so and answer generically, labelled.",
    ]
    return "\n".join(lines).rstrip() + "\n"
