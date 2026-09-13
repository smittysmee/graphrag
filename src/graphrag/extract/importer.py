"""Import entity/relation extractions an agent wrote as JSON.

The agent-driven enrichment path writes one ``{doc_id, entities, relations}`` file per
document. Two callers read those files: ``graphrag enrich-import`` and :mod:`graphrag.sync`,
which re-imports a source's files after a re-ingest (re-ingesting deletes a document's chunks,
and mentions hang off chunks). Both go through :func:`import_extraction_file` so a file lands
in the graph the same way whoever asked for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graphrag.extract.llm import DocumentExtraction, match_chunks, result_to_enrichment
from graphrag.graph.store import GraphStore

__all__ = ["ImportResult", "import_extraction_file", "read_doc_id"]

#: Enough to cover any single document; the store pages, so this is one read either way.
_ALL_CHUNKS = 100_000


@dataclass(frozen=True)
class ImportResult:
    """One extraction file's outcome, including the names a reviewer should look at.

    ``loose`` names were matched by token rather than verbatim, ``unmatched`` ones occur in no
    passage at all (their mention falls back to the first chunk), and ``dangling`` relations
    point at an entity the file never declared.
    """

    path: Path
    doc_id: str = ""
    error: str = ""
    entities: int = 0
    mentions: int = 0
    relations: int = 0
    loose: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()
    dangling: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error


def import_extraction_file(store: GraphStore, path: Path, *, dry_run: bool = False) -> ImportResult:
    """Validate one extraction file and upsert it into the graph.

    Bad input never raises: an unreadable file, invalid JSON or a ``doc_id`` the graph does not
    know comes back as a result whose ``error`` is set, so a caller importing hundreds of files
    can report them all instead of stopping at the first.
    """
    try:
        payload = DocumentExtraction.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ImportResult(path=path, error=f"invalid: {exc}")
    chunks = store.document_chunks(payload.doc_id, 0, _ALL_CHUNKS)
    if not chunks:
        return ImportResult(
            path=path, doc_id=payload.doc_id, error=f"unknown document {payload.doc_id}"
        )
    tiers = {e.name: match_chunks(e.name.strip(), chunks)[1] for e in payload.entities}
    names = {e.name.strip().lower() for e in payload.entities}
    enrichment = result_to_enrichment(payload, chunks)
    if not dry_run:
        store.upsert_enrichment(enrichment)
    return ImportResult(
        path=path,
        doc_id=payload.doc_id,
        entities=len(enrichment.entities),
        mentions=len(enrichment.mentions),
        relations=len(enrichment.relations),
        loose=tuple(name for name, tier in tiers.items() if tier == "loose"),
        unmatched=tuple(name for name, tier in tiers.items() if tier == "none"),
        dangling=tuple(
            f"{r.source} -> {r.target}"
            for r in payload.relations
            if r.source.strip().lower() not in names or r.target.strip().lower() not in names
        ),
    )


def read_doc_id(path: Path) -> str:
    """The ``doc_id`` an extraction file claims, or an empty string when it cannot be read.

    Used to work out which source a file belongs to without trusting the directory it sits in.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    doc_id = data.get("doc_id")
    return doc_id.strip() if isinstance(doc_id, str) else ""
