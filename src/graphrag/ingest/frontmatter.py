"""Minimal YAML front-matter splitting (``---`` fenced block at the top of a markdown file)."""

from __future__ import annotations

from typing import Any

import yaml

FENCE = "---"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)``. Files without front-matter yield ``({}, text)``."""
    if not text.startswith(FENCE):
        return {}, text
    end = text.find(f"\n{FENCE}", len(FENCE))
    if end == -1:
        return {}, text
    raw = text[len(FENCE) : end]
    body = text[end + len(FENCE) + 1 :]
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body.lstrip("\n")
