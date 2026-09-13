"""Bring one persona's graph back in step with the files on disk.

Three facts make "just re-ingest it" a poor instruction. A persona can have several sources,
so the caller has to know which one changed. Re-ingesting replaces a document rather than
merging into it, which deletes its chunks, and entity mentions hang off chunks -- so every
extraction file for that source has to be re-imported afterwards or the entity layer quietly
thins out. And nothing on disk says which of those two things is needed.

:func:`sync_persona` works it out instead: it asks the loaders which document ids the raw files
would produce, compares them with the ids the store already holds, and for each source that is
behind, re-ingests it and re-imports its extraction files.

A source that is up to date still gets one more check, because ingesting and extracting are
separate steps: a document can sit in the graph with no entities at all, because its extraction
file was written after it was ingested or because importing that file failed. Those files are
imported too, so "up to date" means the entity layer is complete rather than only the documents.
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
    enrichment_files: int = 0  # extraction files imported (or, dry run, that would be)
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
    def backfilled_files(self) -> int:
        """Extraction files imported for sources that needed no re-ingest, or that would be."""
        return sum(s.enrichment_files for s in self.sources if not s.stale)

    @property
    def wrote(self) -> bool:
        """Whether the graph changed, which is what decides if a new snapshot is worth taking."""
        return not self.dry_run and any(s.stale or s.enrichment_files for s in self.sources)

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
        if dry_run:
            pending = _pending(store, persona, source, files, reingested=bool(missing))
            reports.append(
                SourceReport(
                    source_id=source.id, missing=tuple(missing), enrichment_files=len(pending)
                )
            )
            continue
        documents = chunks = 0
        if missing:
            if pipeline is None:
                pipeline = IngestPipeline(store, embedder(), progress=say)
            say(f"{source.id}: {len(missing)} documents missing, re-ingesting")
            ingested = pipeline.ingest(raw_root, persona, source)
            documents, chunks = ingested.documents, ingested.chunks
        # After the re-ingest, so the documents it replaced count as needing their entities back.
        pending = _pending(store, persona, source, files, reingested=bool(missing))
        imported, errors = _reimport(store, pending, say, reingested=bool(missing))
        reports.append(
            SourceReport(
                source_id=source.id,
                missing=tuple(missing),
                ingested_documents=documents,
                ingested_chunks=chunks,
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


def _pending(
    store: GraphStore,
    persona: PersonaSpec,
    source: SourceSpec,
    files: list[Path],
    *,
    reingested: bool,
) -> list[Path]:
    """The extraction files this source needs put into the graph now.

    A re-ingested source needs all of them: re-ingesting replaced its documents, which deleted
    the chunks its mentions hung off. A source that was left alone needs only the files whose
    document is in the graph with no entity at all -- extraction JSON written after the document
    was ingested, or whose import failed at the time. Files naming a document the graph does not
    hold are left out of that second set on purpose: on an untouched source that means the raw
    file is gone, and reporting it on every run would be noise rather than news.
    """
    if reingested:
        return files
    unenriched = store.document_ids(persona.id, source.id) - store.enriched_document_ids(
        persona.id, source.id
    )
    if not unenriched:
        return []
    return [path for path in files if read_doc_id(path) in unenriched]


def _reimport(
    store: GraphStore, files: list[Path], say: Progress, *, reingested: bool
) -> tuple[int, tuple[str, ...]]:
    """Put a source's pending extraction files back into the graph."""
    if files:
        what = "re-importing" if reingested else "importing"
        say(f"{what} {len(files)} extraction files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_extraction_file(store, path)
        if result.ok:
            imported += 1
        else:
            errors.append(f"{path.name}: {result.error}")
    return imported, tuple(errors)


def _backfill_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about documents that had no entities, if anything."""
    if not src.enrichment_files:
        return ""
    verb = "would import" if dry_run else "imported"
    return f", {verb} {src.enrichment_files} extraction files for documents without entities"


def summary_lines(report: SyncReport) -> list[str]:
    """One line per source, plus a closing line. Pure, so it is what the tests assert on."""
    lines: list[str] = []
    for src in report.sources:
        if not src.stale:
            lines.append(f"{src.source_id}: up to date{_backfill_clause(src, report.dry_run)}")
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
    if report.missing_total == 0 and report.backfilled_files == 0:
        lines.append(f"{report.persona_id}: nothing to sync")
    elif report.missing_total == 0:
        verb = "would import" if report.dry_run else "imported"
        lines.append(
            f"{report.persona_id}: nothing to ingest, {verb} {report.backfilled_files} "
            f"extraction files for documents without entities"
        )
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
