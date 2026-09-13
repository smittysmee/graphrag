"""Bring one persona's graph back in step with the files on disk.

Three facts make "just re-ingest it" a poor instruction. A persona can have several sources,
so the caller has to know which one changed. Re-ingesting replaces a document rather than
merging into it, which deletes its chunks, and entity mentions hang off chunks -- so every
extraction file for that source has to be re-imported afterwards or the entity layer quietly
thins out. And nothing on disk says which of those two things is needed.

:func:`sync_persona` works it out instead: it asks the loaders which document ids the raw files
would produce, compares them with the ids the store already holds, and for each source that is
behind, re-ingests it and re-imports its extraction files. A source that is already complete is
left alone, so running this when nothing changed costs one query per source and writes nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.embed.base import Embedder
from graphrag.extract.importer import import_extraction_file, read_doc_id
from graphrag.graph.store import GraphStore
from graphrag.ingest.loaders import load_source
from graphrag.models import PersonaSpec, SourceSpec
from graphrag.pipeline import IngestPipeline

__all__ = [
    "SourceReport",
    "SyncReport",
    "UnknownSourceError",
    "expected_document_ids",
    "index_extractions",
    "missing_document_ids",
    "summary_lines",
    "sync_persona",
]

Progress = Callable[[str], None]


class UnknownSourceError(LookupError):
    """Raised when ``--source`` names something the persona does not define."""


@dataclass(frozen=True)
class SourceReport:
    """What syncing one source found and, unless this was a dry run, did about it."""

    source_id: str
    missing: tuple[str, ...] = ()  # document ids the raw files imply but the graph lacks
    ingested_documents: int = 0
    ingested_chunks: int = 0
    enrichment_files: int = 0  # extraction files re-imported (or, dry run, that would be)
    enrichment_errors: tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        return bool(self.missing)


@dataclass(frozen=True)
class SyncReport:
    """The per-source outcome for one persona."""

    persona_id: str
    sources: tuple[SourceReport, ...] = field(default_factory=tuple)
    dry_run: bool = False

    @property
    def stale_sources(self) -> tuple[SourceReport, ...]:
        return tuple(s for s in self.sources if s.stale)

    @property
    def missing_total(self) -> int:
        return sum(len(s.missing) for s in self.sources)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(err for s in self.sources for err in s.enrichment_errors)


def expected_document_ids(raw_root: Path, persona: PersonaSpec, source: SourceSpec) -> list[str]:
    """The document ids ingesting ``source`` would produce, in loader order.

    Runs the loader rather than re-deriving ids from paths, so the transcripts loader's
    body-hash dedupe (an episode archived twice yields one document) and its deterministic
    ``-2`` suffix for colliding ids are both respected. Files with no text at all are left out,
    because the pipeline skips them too and they would otherwise look permanently missing.
    """
    return [
        loaded.document.id for loaded in load_source(raw_root, source, persona.id) if loaded.turns
    ]


def missing_document_ids(
    store: GraphStore, raw_root: Path, persona: PersonaSpec, source: SourceSpec
) -> list[str]:
    """Document ids the raw files imply that the graph does not hold, in loader order."""
    present = store.document_ids(persona.id, source.id)
    return [
        doc_id
        for doc_id in expected_document_ids(raw_root, persona, source)
        if doc_id not in present
    ]


def index_extractions(enrichment_root: Path) -> dict[str, list[Path]]:
    """Every extraction file under ``enrichment_root``, grouped by ``<persona>:<source>:``.

    Grouped on the ``doc_id`` inside each file rather than on the directory it sits in: the
    corpus holds both the per-source layout (``<root>/<persona>/<source>/<doc>.json``) and an
    older flat one, and the id is what actually ties a file to a source. Files that cannot be
    read, or that carry no ``doc_id``, are left out -- ``enrich-import`` is where they get
    reported.
    """
    grouped: dict[str, list[Path]] = {}
    if not enrichment_root.is_dir():
        return grouped
    for path in sorted(enrichment_root.rglob("*.json")):
        if not path.is_file():
            continue
        doc_id = read_doc_id(path)
        persona_id, _, rest = doc_id.partition(":")
        source_id, sep, _ = rest.partition(":")
        if not persona_id or not sep or not source_id:
            continue
        grouped.setdefault(f"{persona_id}:{source_id}:", []).append(path)
    return grouped


def sync_persona(
    persona: PersonaSpec,
    *,
    store: GraphStore,
    embedder: Callable[[], Embedder],
    raw_root: Path,
    enrichment_root: Path,
    source_id: str | None = None,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> SyncReport:
    """Ingest whatever ``raw_root`` holds that the graph does not, source by source.

    ``embedder`` is a factory rather than an ``Embedder`` so a dry run, or a run where every
    source is already complete, never loads model weights.
    """
    say = progress or (lambda _msg: None)
    extractions = index_extractions(enrichment_root)
    pipeline: IngestPipeline | None = None
    reports: list[SourceReport] = []

    for source in _select(persona, source_id):
        missing = missing_document_ids(store, raw_root, persona, source)
        files = extractions.get(f"{persona.id}:{source.id}:", [])
        if not missing:  # the summary reports this; no progress noise for a no-op
            reports.append(SourceReport(source_id=source.id))
            continue
        if dry_run:
            reports.append(
                SourceReport(
                    source_id=source.id, missing=tuple(missing), enrichment_files=len(files)
                )
            )
            continue
        if pipeline is None:
            pipeline = IngestPipeline(store, embedder(), progress=say)
        say(f"{source.id}: {len(missing)} documents missing, re-ingesting")
        ingested = pipeline.ingest(raw_root, persona, source)
        imported, errors = _reimport(store, files, say)
        reports.append(
            SourceReport(
                source_id=source.id,
                missing=tuple(missing),
                ingested_documents=ingested.documents,
                ingested_chunks=ingested.chunks,
                enrichment_files=imported,
                enrichment_errors=errors,
            )
        )
    return SyncReport(persona_id=persona.id, sources=tuple(reports), dry_run=dry_run)


def _select(persona: PersonaSpec, source_id: str | None) -> list[SourceSpec]:
    if source_id is None:
        return list(persona.sources)
    matches = [s for s in persona.sources if s.id == source_id]
    if not matches:
        known = ", ".join(s.id for s in persona.sources) or "(none)"
        msg = f"unknown source {source_id!r} for persona {persona.id}; known sources: {known}"
        raise UnknownSourceError(msg)
    return matches


def _reimport(store: GraphStore, files: list[Path], say: Progress) -> tuple[int, tuple[str, ...]]:
    """Re-import a source's extraction files; re-ingesting dropped their mentions."""
    if files:
        say(f"re-importing {len(files)} extraction files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_extraction_file(store, path)
        if result.ok:
            imported += 1
        else:
            errors.append(f"{path.name}: {result.error}")
    return imported, tuple(errors)


def summary_lines(report: SyncReport) -> list[str]:
    """One line per source, plus a closing line. Pure, so it is what the tests assert on."""
    lines: list[str] = []
    for src in report.sources:
        if not src.stale:
            lines.append(f"{src.source_id}: up to date")
        elif report.dry_run:
            lines.append(
                f"{src.source_id}: {len(src.missing)} documents missing; would re-ingest and "
                f"re-import {src.enrichment_files} extraction files"
            )
        else:
            lines.append(
                f"{src.source_id}: {len(src.missing)} missing -> ingested "
                f"{src.ingested_documents} documents / {src.ingested_chunks} chunks, "
                f"re-imported {src.enrichment_files} extraction files"
            )
        lines.extend(f"  {err}" for err in src.enrichment_errors)
    if report.missing_total == 0:
        lines.append(f"{report.persona_id}: nothing to sync")
    elif report.dry_run:
        lines.append(
            f"{report.persona_id}: {report.missing_total} documents would be ingested "
            f"across {len(report.stale_sources)} source(s)"
        )
    else:
        lines.append(
            f"{report.persona_id}: synced {len(report.stale_sources)} source(s), "
            f"{report.missing_total} documents were missing"
        )
    return lines
