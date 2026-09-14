"""Minimal YAML front-matter splitting (``---`` fenced block at the top of a markdown file)."""

from __future__ import annotations

from typing import Any

import yaml

FENCE = "---"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(metadata, body)``. Files without front-matter yield ``({}, text)``.

    A fence that never closes, or one whose YAML does not parse, gets the same fallback: this is
    the loader's own best-effort split, used at ingest time on files nobody has necessarily
    checked yet, so it never raises. A `documents` source is checked before it reaches here
    (:func:`graphrag.ingest.validate.validate_files`, run by ``ingest`` and ``sync``), which is
    where a malformed fence is reported as a problem naming the file; this function's job is only
    to not crash a caller that has not made that check, such as ``sync`` computing which document
    ids a source would produce.
    """
    if not text.startswith(FENCE):
        return {}, text
    end = text.find(f"\n{FENCE}", len(FENCE))
    if end == -1:
        return {}, text
    raw = text[len(FENCE) : end]
    body = text[end + len(FENCE) + 1 :]
    try:
        loaded = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body.lstrip("\n")
