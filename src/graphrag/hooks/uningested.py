"""Stop hook: notice when files under ``data/raw`` have not reached the graph.

Writing a note under ``data/raw/<persona>/...`` does nothing until someone runs
``make sync``; this hook checks after each turn so nobody has to remember. It is
read-only and fail-silent: ``main`` always exits 0, and it stays quiet when nothing is
missing or when this ``Stop`` itself is a hook continuation (``stop_hook_active``).

Two kinds of comparison, matching how the loaders behave (see ``graphrag.ingest.loaders``):

* a ``documents`` source is compared file by file, since each raw file maps to one
  predictable document id (:func:`graphrag.hooks.project.expected_doc_id`);
* a ``transcripts`` source dedupes archived re-uploads by body hash, so a raw-file count
  above the graph's document count is normal, not a finding. Only a source with raw files
  but *zero* matching documents in the graph is reported.

A third check covers the entity layer: a document that is in the graph but has no extraction
JSON under ``data/enrichment`` is searchable yet invisible to entity and relation queries. That
gap is agent work (the ``graph-rag-enrich`` skill), so the hook names it rather than fixing it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.hooks import project
from graphrag.hooks.mcp_client import McpClient, McpUnavailable, connect
from graphrag.textutil import sanitize_inline

__all__ = [
    "Finding",
    "Report",
    "Unenriched",
    "find_uningested",
    "main",
    "render_message",
    "summary_line",
]

_PAGE_SIZE = 500
_MAX_PAGES = 40  # guards against an unbounded loop if the server misbehaves
_MAX_NAMED = 3
_MAX_MESSAGE = 500
# Persona ids, source ids and raw file paths all reach a rendered line. They come off the
# filesystem, so each is flattened before it is spliced into a message a session reads.
_MAX_ID_LEN = 60
_MAX_FILE_LEN = 100


@dataclass(frozen=True)
class Finding:
    """One persona/source pair that has raw files the graph does not know about."""

    persona_id: str
    source_id: str
    loader: str
    missing_files: tuple[str, ...] = ()  # ``documents`` loader: rel paths not in the graph
    raw_count: int = 0  # ``transcripts`` loader: files found when the graph has none

    @property
    def file_count(self) -> int:
        """How many raw files this finding is about, for totals and the summary line."""
        return len(self.missing_files) if self.missing_files else self.raw_count


@dataclass(frozen=True)
class Unenriched:
    """Documents in the graph for one persona/source that have no extraction JSON yet."""

    persona_id: str
    source_id: str
    doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Report:
    """What :func:`find_uningested` found, plus enough to build the sync command."""

    source: str  # "graph" (the live server answered) or "snapshot" (it was down)
    findings: tuple[Finding, ...] = ()
    unenriched: tuple[Unenriched, ...] = ()

    @property
    def total_files(self) -> int:
        """Total raw files across every finding."""
        return sum(f.file_count for f in self.findings)

    @property
    def by_persona(self) -> dict[str, int]:
        """Total raw files per persona, in the order personas were first found."""
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.persona_id] = counts.get(finding.persona_id, 0) + finding.file_count
        return counts

    @property
    def total_unenriched(self) -> int:
        """Total ingested documents that have no extraction JSON."""
        return sum(len(u.doc_ids) for u in self.unenriched)

    @property
    def unenriched_by_persona(self) -> dict[str, int]:
        """Un-enriched document counts per persona, in the order personas were first found."""
        counts: dict[str, int] = {}
        for item in self.unenriched:
            counts[item.persona_id] = counts.get(item.persona_id, 0) + len(item.doc_ids)
        return counts

    @property
    def is_clean(self) -> bool:
        """True when there is nothing to report at all."""
        return self.total_files == 0 and self.total_unenriched == 0


def find_uningested(root: Path, client: McpClient | None) -> Report:
    """Raw files that are not represented in the graph, persona by persona.

    Uses the live graph (paged Cypher on ``Document.id``) when ``client`` answers; otherwise
    falls back to the committed snapshot and labels the report ``source="snapshot"``. A
    source whose check itself fails (a transient server error) is skipped rather than guessed
    at, so it never produces a false finding.
    """
    personas = project.load_personas(root)
    source_kind = "graph" if client is not None else "snapshot"
    snapshot_cache: dict[str, set[str]] = {}
    enriched = project.enriched_doc_ids(root)
    findings: list[Finding] = []
    unenriched: list[Unenriched] = []

    for pid in sorted(personas):
        persona = personas[pid]
        for source in persona.sources:
            files = project.raw_files(root, persona, source)
            if not files:
                continue
            try:
                existing = _existing_ids(root, client, persona, source, snapshot_cache)
            except McpUnavailable:
                continue

            gap = tuple(sorted(existing - enriched))
            if gap:
                unenriched.append(Unenriched(pid, source.id, gap))

            if source.loader == "transcripts":
                if not existing:
                    findings.append(Finding(pid, source.id, source.loader, raw_count=len(files)))
                continue

            base = project.source_base(root, persona, source)
            missing = tuple(
                project.rel_path(base, path)
                for path in files
                if project.expected_doc_id(root, persona, source, path) not in existing
            )
            if missing:
                findings.append(Finding(pid, source.id, source.loader, missing_files=missing))

    return Report(source=source_kind, findings=tuple(findings), unenriched=tuple(unenriched))


def _existing_ids(
    root: Path,
    client: McpClient | None,
    persona: project.PersonaInfo,
    source: project.SourceInfo,
    snapshot_cache: dict[str, set[str]],
) -> set[str]:
    """Document ids already known for this persona/source, live or from the snapshot."""
    prefix = f"{persona.id}:{source.id}:"
    if client is not None:
        return _graph_doc_ids(client, prefix)
    if persona.id not in snapshot_cache:
        snapshot_cache[persona.id] = project.snapshot_doc_ids(root, persona)
    return {doc_id for doc_id in snapshot_cache[persona.id] if doc_id.startswith(prefix)}


def _graph_doc_ids(client: McpClient, prefix: str) -> set[str]:
    """Every ``Document.id`` under ``prefix``, paged 500 rows at a time."""
    ids: set[str] = set()
    skip = 0
    query = (
        "MATCH (d:Document) WHERE d.id STARTS WITH $prefix "
        "RETURN d.id ORDER BY d.id SKIP $skip LIMIT 500"
    )
    for _ in range(_MAX_PAGES):
        params = {"prefix": prefix, "skip": skip}
        rows = client.call_tool("cypher", {"query": query, "params": params})
        if not isinstance(rows, list) or not rows:
            break
        for row in rows:
            doc_id = _row_id(row)
            if doc_id:
                ids.add(doc_id)
        if len(rows) < _PAGE_SIZE:
            break
        skip += _PAGE_SIZE
    return ids


def _row_id(row: Any) -> str:
    if not isinstance(row, dict) or not row:
        return ""
    value = next(iter(row.values()))
    return value if isinstance(value, str) else ""


def _persona(pid: str) -> str:
    """A persona id, flattened for display."""
    return sanitize_inline(pid, _MAX_ID_LEN)


def _counts(by_persona: dict[str, int]) -> str:
    """``pid: n`` pairs with every id flattened to one printable line."""
    return ", ".join(f"{_persona(pid)}: {n}" for pid, n in by_persona.items())


def summary_line(report: Report) -> str:
    """One short line for the session card; empty when nothing is missing.

    Every persona id in the line goes through :func:`graphrag.textutil.sanitize_inline`, since
    this line is embedded in the session card that a model reads as context.
    """
    if report.is_clean:
        return ""
    clauses: list[str] = []
    total = report.total_files
    if total > 0:
        plural = "file" if total == 1 else "files"
        clauses.append(f"not yet ingested: {total} {plural} ({_counts(report.by_persona)})")
    missing = report.total_unenriched
    if missing > 0:
        plural = "document" if missing == 1 else "documents"
        parts = _counts(report.unenriched_by_persona)
        clauses.append(f"without entity extraction: {missing} {plural} ({parts})")
    suffix = " (vs committed snapshot)" if report.source == "snapshot" else ""
    return "; ".join(clauses) + suffix


def render_message(report: Report) -> str | None:
    """The Stop-hook ``systemMessage`` (under 500 characters), or ``None`` when nothing missing.

    File names, source ids and persona ids are all filesystem-derived, so each is flattened by
    :func:`graphrag.textutil.sanitize_inline` before it reaches the message.
    """
    if report.is_clean:
        return None
    enrich = _enrich_clause(report)
    total = report.total_files
    if total <= 0:
        return _fit(f"graphrag:{enrich}")
    entries = _entries(report)
    plural = "file" if total == 1 else "files"
    against = " (vs committed snapshot)" if report.source == "snapshot" else ""
    sync_cmd = f"`make sync PERSONA={_persona(_primary(report))}`"
    message = ""
    for shown_n in (_MAX_NAMED, 2, 1, 0):
        shown = entries[:shown_n]
        more = len(entries) - len(shown)
        names = ", ".join(shown)
        if more > 0:
            names = f"{names} (+{more} more)" if names else f"(+{more} more)"
        message = (
            f"graphrag: {total} raw {plural} not in the graph{against}: {names}. "
            f"Run {sync_cmd}.{enrich}"
        )
        if len(message) <= _MAX_MESSAGE:
            return message
    return _fit(message)


def _fit(message: str) -> str:
    if len(message) <= _MAX_MESSAGE:
        return message
    return message[: _MAX_MESSAGE - 1] + "…"


def _enrich_clause(report: Report) -> str:
    """`` N ingested documents have no entity extraction (...); run the graph-rag-enrich skill``."""
    missing = report.total_unenriched
    if missing <= 0:
        return ""
    by_persona = report.unenriched_by_persona
    parts = _counts(by_persona)
    verb = "document has" if missing == 1 else "documents have"
    persona = min(by_persona, key=lambda pid: (-by_persona[pid], pid))
    return (
        f" {missing} ingested {verb} no entity extraction ({parts}); "
        f"run the graph-rag-enrich skill for {_persona(persona)}."
    )


def _entries(report: Report) -> list[str]:
    """File and source names for the message, each flattened to one printable line."""
    names: list[str] = []
    for finding in report.findings:
        if finding.missing_files:
            names.extend(sanitize_inline(f, _MAX_FILE_LEN) for f in finding.missing_files)
        else:
            source = sanitize_inline(finding.source_id, _MAX_ID_LEN)
            names.append(f"{source}/ (0 of {finding.raw_count} ingested)")
    return names


def _primary(report: Report) -> str:
    """The persona with the most missing files (ties broken by id), for the sync command."""
    by_persona = report.by_persona
    return min(by_persona, key=lambda pid: (-by_persona[pid], pid))


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


def main() -> int:
    """Read the ``Stop`` stdin payload; print a ``systemMessage`` when files are un-ingested."""
    try:
        payload = _read_stdin_payload()
        if payload.get("stop_hook_active"):
            return 0
        cwd = payload.get("cwd")
        start = Path(cwd) if isinstance(cwd, str) and cwd else None
        root = project.find_root(start)
        if root is None:
            return 0
        client = connect()
        message = render_message(find_uningested(root, client))
        if message:
            print(json.dumps({"systemMessage": message}))
    except Exception:  # a broken hook must never fail a turn
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
