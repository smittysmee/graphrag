"""Small text helpers with no third-party dependencies.

Kept import-light on purpose: ``graphrag.hooks`` runs on the host with a bare ``python3``
and the standard library only, so anything it needs must live outside the pydantic models.
"""

from __future__ import annotations

import re

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Lower-case, ASCII-ish slug safe for ids, folder names and Cypher parameters."""
    value = value.strip().lower()
    value = _NON_SLUG.sub("-", value)
    return value.strip("-") or "untitled"
