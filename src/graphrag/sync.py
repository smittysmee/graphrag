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

Attribution files ride along on exactly the same reasoning. ``SPOKE`` edges hang off chunks, so a
re-ingest drops them; and a document loaded without speaker turns can sit in the graph with no
speaker at all until its attribution file lands. Both cases are handled beside the extraction
ones, against :func:`graphrag.extract.attribution.import_attribution_file`. Annotation files --
the stance on a mention, the facets on a passage -- are a third layer with the same two triggers,
and hang off chunks for the same reason.

The persona's alias table is applied once at the end, after every file has landed. An import can
only canonicalise the names inside the file it is reading; folding two spellings that arrived in
different files into one node is a whole-graph operation, so it runs when the graph is whole.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.embed.base import Embedder
from graphrag.extract.aliases import EMPTY_ALIASES, AliasReport, AliasTable, apply_alias_table
from graphrag.extract.annotations import EntityIndex, FacetTable, import_annotation_file
from graphrag.extract.attribution import import_attribution_file
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
    "index_annotations",
    "index_attributions",
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
    attribution_files: int = 0  # attribution files imported (or, dry run, that would be)
    attribution_errors: tuple[str, ...] = ()
    annotation_files: int = 0  # annotation files imported (or, dry run, that would be)
    annotation_errors: tuple[str, ...] = ()

    @property
    def stale(self) -> bool:
        return bool(self.missing)


@dataclass(frozen=True)
class SyncReport:
    """The per-source outcome for one persona."""

    persona_id: str
    sources: tuple[SourceReport, ...] = field(default_factory=tuple)
    dry_run: bool = False
    aliases: AliasReport | None = None

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
    def backfilled_attribution_files(self) -> int:
        """Attribution files imported for sources that needed no re-ingest, or that would be."""
        return sum(s.attribution_files for s in self.sources if not s.stale)

    @property
    def backfilled_annotation_files(self) -> int:
        """Annotation files imported for sources that needed no re-ingest, or that would be."""
        return sum(s.annotation_files for s in self.sources if not s.stale)

    @property
    def wrote(self) -> bool:
        """Whether the graph changed, which is what decides if a new snapshot is worth taking."""
        moved = self.aliases.mentions_moved if self.aliases else 0
        return not self.dry_run and (
            bool(moved)
            or any(
                s.stale or s.enrichment_files or s.attribution_files or s.annotation_files
                for s in self.sources
            )
        )

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            err
            for s in self.sources
            for err in (*s.enrichment_errors, *s.attribution_errors, *s.annotation_errors)
        )


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


def index_attributions(attribution_root: Path | None) -> dict[str, list[Path]]:
    """Every attribution file under ``attribution_root``, grouped by ``<persona>:<source>:``.

    The same grouping as :func:`index_extractions`, and for the same reason: what ties a file to
    a source is the ``doc_id`` inside it. ``None`` means the caller has no attribution directory,
    which is not a finding.
    """
    return index_extractions(attribution_root) if attribution_root else {}


def index_annotations(annotation_root: Path | None) -> dict[str, list[Path]]:
    """Every annotation file under ``annotation_root``, grouped by ``<persona>:<source>:``."""
    return index_extractions(annotation_root) if annotation_root else {}


def sync_persona(
    persona: PersonaSpec,
    *,
    store: GraphStore,
    embedder: Callable[[], Embedder],
    raw_root: Path,
    enrichment_root: Path,
    attribution_root: Path | None = None,
    annotation_root: Path | None = None,
    aliases: AliasTable = EMPTY_ALIASES,
    facets: FacetTable | None = None,
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
    attributions = index_attributions(attribution_root)
    annotations = index_annotations(annotation_root)
    pipeline: IngestPipeline | None = None
    reports: list[SourceReport] = []

    for source in _select(persona, source_id):
        missing = missing_document_ids(store, raw_root, persona, source)
        key = f"{persona.id}:{source.id}:"
        files = extractions.get(key, [])
        voices = attributions.get(key, [])
        readings = annotations.get(key, [])
        if dry_run:
            pending = _pending(store, persona, source, files, reingested=bool(missing))
            waiting = _pending_attribution(store, persona, source, voices, reingested=bool(missing))
            unread = _pending_annotation(store, persona, source, readings, reingested=bool(missing))
            reports.append(
                SourceReport(
                    source_id=source.id,
                    missing=tuple(missing),
                    enrichment_files=len(pending),
                    attribution_files=len(waiting),
                    annotation_files=len(unread),
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
        imported, errors = _reimport(store, pending, say, aliases, reingested=bool(missing))
        # Likewise for speakers: a re-ingest deleted the chunks their SPOKE edges hung off.
        waiting = _pending_attribution(store, persona, source, voices, reingested=bool(missing))
        attributed, voice_errors = _reimport_attribution(
            store, waiting, say, reingested=bool(missing)
        )
        # And for the readings, which hang off the same chunks and the mentions on them.
        unread = _pending_annotation(store, persona, source, readings, reingested=bool(missing))
        annotated, reading_errors = _reimport_annotation(
            store, unread, say, persona.id, aliases, facets, reingested=bool(missing)
        )
        reports.append(
            SourceReport(
                source_id=source.id,
                missing=tuple(missing),
                ingested_documents=documents,
                ingested_chunks=chunks,
                enrichment_files=imported,
                enrichment_errors=errors,
                attribution_files=attributed,
                attribution_errors=voice_errors,
                annotation_files=annotated,
                annotation_errors=reading_errors,
            )
        )
    # Last, and over the whole persona: one file can only canonicalise its own names, so two
    # spellings that arrived in different files are folded together once everything has landed.
    alias_report = None
    if aliases and not dry_run:
        alias_report = apply_alias_table(store, persona.id, aliases)
        if alias_report.applied:
            say(
                f"folded {alias_report.mentions_moved} mentions onto "
                f"{len(alias_report.applied)} canonical names"
            )
    return SyncReport(
        persona_id=persona.id, sources=tuple(reports), dry_run=dry_run, aliases=alias_report
    )


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
    store: GraphStore, files: list[Path], say: Progress, aliases: AliasTable, *, reingested: bool
) -> tuple[int, tuple[str, ...]]:
    """Put a source's pending extraction files back into the graph."""
    if files:
        what = "re-importing" if reingested else "importing"
        say(f"{what} {len(files)} extraction files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_extraction_file(store, path, aliases=aliases)
        if result.ok:
            imported += 1
        else:
            errors.append(f"{path.name}: {result.error}")
    return imported, tuple(errors)


def _pending_attribution(
    store: GraphStore,
    persona: PersonaSpec,
    source: SourceSpec,
    files: list[Path],
    *,
    reingested: bool,
) -> list[Path]:
    """The attribution files this source needs put into the graph now.

    The reasoning is :func:`_pending`'s, one layer over: a re-ingested source needs all of them,
    because re-ingesting deleted the chunks its ``SPOKE`` edges hung off. A source that was left
    alone needs only the files whose document carries no speaker at all -- a thread whose
    attribution JSON was written after it was ingested, or whose import failed at the time.
    """
    if reingested:
        return files
    unattributed = store.document_ids(persona.id, source.id) - store.attributed_document_ids(
        persona.id, source.id
    )
    if not unattributed:
        return []
    return [path for path in files if read_doc_id(path) in unattributed]


def _reimport_attribution(
    store: GraphStore, files: list[Path], say: Progress, *, reingested: bool
) -> tuple[int, tuple[str, ...]]:
    """Put a source's pending attribution files back into the graph.

    A file whose anchors all came loose still counts as imported: it was read and applied, and
    the loose anchors are ``attribution-import``'s report to make, not this one's.
    """
    if files:
        what = "re-importing" if reingested else "importing"
        say(f"{what} {len(files)} attribution files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_attribution_file(store, path)
        if result.ok:
            imported += 1
        else:
            errors.append(f"{path.name}: {result.error}")
    return imported, tuple(errors)


def _pending_annotation(
    store: GraphStore,
    persona: PersonaSpec,
    source: SourceSpec,
    files: list[Path],
    *,
    reingested: bool,
) -> list[Path]:
    """The annotation files this source needs put into the graph now.

    :func:`_pending`'s reasoning again: a re-ingested source needs all of them, because the
    stance lives on a mention and the facets on a passage, and re-ingesting deleted both. A
    source that was left alone needs only the files whose document carries no stance and no
    facet -- annotation JSON written after the document was ingested, or whose import failed.
    """
    if reingested:
        return files
    unannotated = store.document_ids(persona.id, source.id) - store.annotated_document_ids(
        persona.id, source.id
    )
    if not unannotated:
        return []
    return [path for path in files if read_doc_id(path) in unannotated]


def _reimport_annotation(
    store: GraphStore,
    files: list[Path],
    say: Progress,
    persona_id: str,
    aliases: AliasTable,
    facets: FacetTable | None,
    *,
    reingested: bool,
) -> tuple[int, tuple[str, ...]]:
    """Put a source's pending annotation files back into the graph.

    A file whose anchors all came loose still counts as imported, for the reason
    :func:`_reimport_attribution` gives: the loose anchors are ``annotations-import``'s report.
    The entity index is built once here, after the extraction files for this source have landed,
    so the fallback sees every entity this run put in the graph.
    """
    if not files:
        return 0, ()
    what = "re-importing" if reingested else "importing"
    say(f"{what} {len(files)} annotation files")
    known = EntityIndex.build(store, persona_id)
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_annotation_file(store, path, aliases=aliases, facets=facets, entities=known)
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


def _attribution_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about documents that had no speakers, if anything.

    Empty unless there were attribution files, so a corpus that uses none reads exactly as it
    did before the attribution layer existed.
    """
    if not src.attribution_files:
        return ""
    verb = "would import" if dry_run else "imported"
    return f", {verb} {src.attribution_files} attribution files for documents without speakers"


def _annotation_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about documents that had no stance or facet, if anything.

    Empty unless there were annotation files, so a corpus that uses none reads exactly as it did
    before the annotation layer existed.
    """
    if not src.annotation_files:
        return ""
    verb = "would import" if dry_run else "imported"
    return f", {verb} {src.annotation_files} annotation files for documents without annotations"


def _reimported_clause(src: SourceReport) -> str:
    """The other layers of a stale source's line, each left out when there is none."""
    parts = []
    if src.attribution_files:
        parts.append(f"{src.attribution_files} attribution files")
    if src.annotation_files:
        parts.append(f"{src.annotation_files} annotation files")
    return f" and {' and '.join(parts)}" if parts else ""


def _closing_line(report: SyncReport) -> str:
    """The last line: what the run did overall, in the terms the run was about."""
    if report.missing_total and report.dry_run:
        return (
            f"{report.persona_id}: {report.missing_total} documents would be ingested "
            f"across {len(report.stale_sources)} source(s)"
        )
    if report.missing_total:
        return (
            f"{report.persona_id}: synced {len(report.stale_sources)} source(s), "
            f"{report.missing_total} documents were missing"
        )
    verb = "would import" if report.dry_run else "imported"
    clauses = []
    if report.backfilled_files:
        clauses.append(
            f"{verb} {report.backfilled_files} extraction files for documents without entities"
        )
    if report.backfilled_attribution_files:
        clauses.append(
            f"{verb} {report.backfilled_attribution_files} attribution files for documents "
            f"without speakers"
        )
    if report.backfilled_annotation_files:
        clauses.append(
            f"{verb} {report.backfilled_annotation_files} annotation files for documents "
            f"without annotations"
        )
    if not clauses:
        return f"{report.persona_id}: nothing to sync"
    return f"{report.persona_id}: nothing to ingest, " + ", ".join(clauses)


def summary_lines(report: SyncReport) -> list[str]:
    """One line per source, plus a closing line. Pure, so it is what the tests assert on."""
    lines: list[str] = []
    for src in report.sources:
        if not src.stale:
            lines.append(
                f"{src.source_id}: up to date{_backfill_clause(src, report.dry_run)}"
                f"{_attribution_clause(src, report.dry_run)}"
                f"{_annotation_clause(src, report.dry_run)}"
            )
        elif report.dry_run:
            lines.append(
                f"{src.source_id}: {len(src.missing)} documents missing; would re-ingest and "
                f"re-import {src.enrichment_files} extraction files{_reimported_clause(src)}"
            )
        else:
            lines.append(
                f"{src.source_id}: {len(src.missing)} missing -> ingested "
                f"{src.ingested_documents} documents / {src.ingested_chunks} chunks, "
                f"re-imported {src.enrichment_files} extraction files{_reimported_clause(src)}"
            )
        lines.extend(
            f"  {err}"
            for err in (*src.enrichment_errors, *src.attribution_errors, *src.annotation_errors)
        )
    if report.aliases and report.aliases.applied:
        lines.append(
            f"aliases: folded {report.aliases.mentions_moved} mentions onto "
            f"{len(report.aliases.applied)} canonical names"
        )
    lines.append(_closing_line(report))
    return lines
