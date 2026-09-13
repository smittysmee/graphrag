"""SessionStart hook: a short status card of the graphrag knowledge graph.

Tells whoever just started a session (or a subagent whose frontmatter runs this hook) what
personas exist, how big each one is, whether the MCP server answers, which raw files are
newest, and -- when :mod:`graphrag.hooks.uningested` is importable -- how many raw files the
graph does not know about yet. Nothing here does retrieval; it is a map, not an answer.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphrag.hooks import project
from graphrag.hooks.mcp_client import McpClient, McpUnavailable, connect, default_url

__all__ = ["main", "render_card"]

MAX_NEWEST_FILES = 5
# ``datetime.UTC`` (the alias ruff/UP017 wants) is 3.11+; the host may still run 3.10.
_UTC = timezone.utc  # noqa: UP017


def render_card(
    root: Path,
    client: McpClient | None,
    now: datetime,
    uningested_line: str | None = None,
) -> str:
    """Render the plain-text status card. Pure: no I/O beyond the given ``client``."""
    personas = project.load_personas(root)
    manifests = project.load_manifests(root)
    live_stats = _live_per_persona(client)

    lines = ["graphrag knowledge graph", ""]
    ids = sorted(set(personas) | set(manifests))
    if not ids:
        lines.append("no personas found under personas/")
    for pid in ids:
        lines.append(_persona_line(pid, personas.get(pid), manifests.get(pid), live_stats, now))

    lines.append("")
    lines.append(_server_line(client))

    newest = project.newest_raw_files(root, MAX_NEWEST_FILES)
    if newest:
        lines.append("")
        lines.append("newest raw files:")
        lines.extend(f"  {_newest_entry(root, path, mtime)}" for path, mtime in newest)

    if uningested_line:
        lines.append("")
        lines.append(uningested_line)

    lines.append("")
    lines.append(
        "Use the graphrag MCP tools (context, search, read_document); "
        "persona skills are under .claude/skills/."
    )
    return "\n".join(lines)


def _newest_entry(root: Path, path: Path, mtime: float) -> str:
    date = datetime.fromtimestamp(mtime, tz=_UTC).strftime("%Y-%m-%d")
    return f"{project.rel_path(root, path)} ({date})"


def _live_per_persona(client: McpClient | None) -> dict[str, dict[str, int]] | None:
    """``stats().graph.per_persona`` from the live server, or ``None`` when it is down."""
    if client is None:
        return None
    try:
        payload = client.call_tool("stats", {})
    except McpUnavailable:
        return None
    if not isinstance(payload, dict):
        return None
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        return None
    per_persona = graph.get("per_persona")
    return per_persona if isinstance(per_persona, dict) else {}


def _persona_line(
    pid: str,
    persona: project.PersonaInfo | None,
    manifest: project.Manifest | None,
    live_stats: dict[str, dict[str, int]] | None,
    now: datetime,
) -> str:
    name = persona.name if persona is not None else pid
    live = live_stats.get(pid) if live_stats is not None else None
    if live is not None:
        counts = f"{_count(live, 'documents')} documents, {_count(live, 'chunks')} chunks"
    elif manifest is not None:
        counts = f"{manifest.document_count} documents, {manifest.chunk_count} chunks (snapshot)"
    elif live_stats is not None:
        counts = "not ingested"
    else:
        counts = "unknown (no snapshot; server not reachable)"
    snapshot = f", snapshot {manifest.created_date}" if manifest and manifest.created_date else ""
    return f"- {pid} ({name}): {counts}{snapshot}"


def _count(counts: dict[str, int], key: str) -> int:
    value = counts.get(key, 0)
    return value if isinstance(value, int) else 0


def _server_line(client: McpClient | None) -> str:
    if client is None:
        return f"MCP server: not reachable at {default_url()} (start with `make setup`)"
    return f"MCP server: up at {client.url}"


def _read_stdin_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _uningested_line(root: Path, client: McpClient | None) -> str | None:
    """The "not yet ingested" summary, when :mod:`graphrag.hooks.uningested` is available."""
    try:
        from graphrag.hooks.uningested import find_uningested, summary_line
    except ImportError:
        return None
    try:
        line = summary_line(find_uningested(root, client))
    except Exception:  # a broken sibling module must not take this hook down
        return None
    return line or None


def _emit(card: str, event_name: str, as_json: bool) -> None:
    if not as_json:
        print(card)
        return
    output = {
        "hookSpecificOutput": {
            "hookEventName": event_name or "SessionStart",
            "additionalContext": card,
        }
    }
    print(json.dumps(output))


def main() -> int:
    """Read the SessionStart stdin payload, print the card, and always return 0."""
    try:
        payload = _read_stdin_payload()
        cwd = payload.get("cwd")
        start = Path(cwd) if isinstance(cwd, str) and cwd else None
        root = project.find_root(start)
        if root is None:
            return 0
        client = connect()
        card = render_card(root, client, datetime.now(tz=_UTC), _uningested_line(root, client))
        event_name = payload.get("hook_event_name")
        _emit(card, event_name if isinstance(event_name, str) else "", "--json" in sys.argv[1:])
    except Exception:  # a status card must never break a session start
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
