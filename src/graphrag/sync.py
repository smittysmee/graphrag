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

Both of those triggers ask whether a layer is *absent*, which is the wrong question after a
sidecar is rewritten: an attribution file that gains posted dates, or an annotation file that
gains facets, names a document that already has speakers or stances, so nothing would re-import
it and the new fields would never reach the graph. ``refresh_attribution`` and
``refresh_annotations`` answer that case by re-importing every sidecar of that kind the persona
has, whatever the graph already holds, and each source's line says so. A tagging pass that adds
node attributes to files the graph already has speakers and stances for is exactly that case, so
those two flags are how attributes reach the graph; what the vocabulary refuses, and what a
second file contradicts, is printed as the run goes past.

The persona's alias table is applied once at the end, after every file has landed. An import can
only canonicalise the names inside the file it is reading; folding two spellings that arrived in
different files into one node is a whole-graph operation, so it runs when the graph is whole.

A `documents` source behind on documents is checked against the corpus contract
(:func:`graphrag.ingest.validate.validate_files`) before it is re-ingested, exactly as
``graphrag ingest`` checks one -- a bad capture is invisible once it is a chunk. A source that
fails is refused rather than re-ingested, and reported that way, while every other source keeps
going, including this one's own backfill of documents already in the graph. ``allow_invalid``
skips the check, as it does for ``ingest``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.embed.base import Embedder
from graphrag.extract.aliases import EMPTY_ALIASES, AliasReport, AliasTable, apply_alias_table
from graphrag.extract.annotations import EntityIndex, FacetTable, import_annotation_file
from graphrag.extract.attributes import EMPTY_ATTRIBUTES, AttributeTable
from graphrag.extract.attribution import import_attribution_file
from graphrag.extract.importer import import_extraction_file, read_doc_id
from graphrag.graph.store import GraphStore
from graphrag.ingest.loaders import iter_source_files, load_source
from graphrag.ingest.validate import DOCUMENT_SUFFIXES, ValidationReport, validate_files
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
    attribution_refreshed: bool = False  # every attribution file was re-imported, not only gaps
    annotation_refreshed: bool = False  # every annotation file was re-imported, not only gaps
    invalid: bool = False  # a `documents` source with missing docs whose captures broke the
    # corpus contract, so the re-ingest that would have filled `missing` was refused
    invalid_files: int = 0  # captured files checked when refusing
    invalid_documents: int = 0  # documents the refusal left un-ingested
    problems: tuple[str, ...] = ()  # validate_corpus-style problem lines, for a refused source

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
    def invalid_sources(self) -> tuple[SourceReport, ...]:
        return tuple(s for s in self.sources if s.invalid)

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
    def refreshed_attribution(self) -> bool:
        """Whether this run re-imported attribution files it was not asked to by a gap."""
        return any(s.attribution_refreshed for s in self.sources)

    @property
    def refreshed_annotations(self) -> bool:
        """Whether this run re-imported annotation files it was not asked to by a gap."""
        return any(s.annotation_refreshed for s in self.sources)

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
    attributes: AttributeTable = EMPTY_ATTRIBUTES,
    source_id: str | None = None,
    refresh_attribution: bool = False,
    refresh_annotations: bool = False,
    allow_invalid: bool = False,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> SyncReport:
    """Ingest whatever ``raw_root`` holds that the graph does not, source by source.

    ``embedder`` is a factory rather than an ``Embedder`` so a dry run, or a run where every
    source is already complete, never loads model weights.

    ``refresh_attribution`` and ``refresh_annotations`` re-import every sidecar of that kind
    rather than only the ones filling a gap, which is what a rewritten file needs: it names a
    document that already has the layer, so nothing else would pick the new fields up.

    A `documents` source that is missing documents is checked against the corpus contract
    (:func:`graphrag.ingest.validate.validate_files`) before it is re-ingested, exactly as
    ``graphrag ingest`` checks one -- a bad capture is invisible once it is a chunk. A source that
    fails is refused rather than re-ingested (reported on its :class:`SourceReport` as
    ``invalid``), while every other source, and this source's own backfill of already-ingested
    documents missing an entity, speaker or annotation layer, still runs. ``allow_invalid`` skips
    the check, as it does for ``ingest``.
    """
    say = progress or (lambda _msg: None)
    extractions = index_extractions(enrichment_root)
    attributions = index_attributions(attribution_root)
    annotations = index_annotations(annotation_root)
    pipeline: IngestPipeline | None = None
    reports: list[SourceReport] = []

    for source in _select(persona, source_id):
        found_missing = missing_document_ids(store, raw_root, persona, source)
        # Checked before anything is re-ingested, and only when there is something to re-ingest:
        # an already-complete source is left alone, exactly as `ingest` never revalidates one.
        invalid_report = (
            _validate_documents_source(raw_root, source)
            if found_missing and not allow_invalid
            else None
        )
        invalid = invalid_report is not None and not invalid_report.ok
        problems: tuple[str, ...] = ()
        invalid_files = 0
        if invalid_report is not None and invalid:
            problems = tuple(invalid_report.problem_lines())
            invalid_files = sum(invalid_report.documents_per_source.values())
        # A refused source is treated as though nothing were missing: no re-ingest happens, so
        # none of the layers a re-ingest would have dropped need re-importing either. Its already-
        # ingested documents still get whatever backfill they are due, below.
        missing = [] if invalid else found_missing
        if invalid:
            say(
                f"{source.id}: refusing to ingest -- {len(problems)} problem(s) in "
                f"{invalid_files} captured file(s); fix the files or pass --allow-invalid"
            )
        key = f"{persona.id}:{source.id}:"
        files = extractions.get(key, [])
        voices = attributions.get(key, [])
        readings = annotations.get(key, [])
        if dry_run:
            pending = _pending(store, persona, source, files, reingested=bool(missing))
            waiting = _pending_attribution(
                store,
                persona,
                source,
                voices,
                reingested=bool(missing),
                refresh=refresh_attribution,
            )
            unread = _pending_annotation(
                store,
                persona,
                source,
                readings,
                reingested=bool(missing),
                refresh=refresh_annotations,
            )
            reports.append(
                SourceReport(
                    source_id=source.id,
                    missing=tuple(missing),
                    enrichment_files=len(pending),
                    attribution_files=len(waiting),
                    annotation_files=len(unread),
                    attribution_refreshed=refresh_attribution,
                    annotation_refreshed=refresh_annotations,
                    invalid=invalid,
                    invalid_files=invalid_files,
                    invalid_documents=len(found_missing) if invalid else 0,
                    problems=problems,
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
            if ingested.orphans_removed:
                say(f"{source.id}: removed {ingested.orphans_removed} orphaned entities")
        # After the re-ingest, so the documents it replaced count as needing their entities back.
        pending = _pending(store, persona, source, files, reingested=bool(missing))
        imported, errors = _reimport(store, pending, say, aliases, reingested=bool(missing))
        # Likewise for speakers: a re-ingest deleted the chunks their SPOKE edges hung off.
        waiting = _pending_attribution(
            store, persona, source, voices, reingested=bool(missing), refresh=refresh_attribution
        )
        attributed, voice_errors = _reimport_attribution(
            store, waiting, say, attributes, reingested=bool(missing) or refresh_attribution
        )
        # And for the readings, which hang off the same chunks and the mentions on them.
        unread = _pending_annotation(
            store, persona, source, readings, reingested=bool(missing), refresh=refresh_annotations
        )
        annotated, reading_errors = _reimport_annotation(
            store,
            unread,
            say,
            persona.id,
            aliases,
            facets,
            attributes,
            reingested=bool(missing) or refresh_annotations,
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
                attribution_refreshed=refresh_attribution,
                annotation_refreshed=refresh_annotations,
                invalid=invalid,
                invalid_files=invalid_files,
                invalid_documents=len(found_missing) if invalid else 0,
                problems=problems,
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


def _validate_documents_source(raw_root: Path, source: SourceSpec) -> ValidationReport | None:
    """The corpus contract check ``ingest`` applies to a `documents` source, run here too.

    ``None`` for any other loader: a `transcripts` source's front matter belongs to the archive
    that produced it, exactly as ``ingest`` leaves it unchecked.
    """
    if source.loader != "documents":
        return None
    base = raw_root / source.path if source.path else raw_root
    files = [
        p for p in iter_source_files(raw_root, source) if p.suffix.lower() in DOCUMENT_SUFFIXES
    ]
    return validate_files(files, root=base)


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
    """Put a source's pending extraction files back into the graph.

    An id collision is said out loud rather than counted as an error: the file was imported and
    its mentions landed, but one name is now hanging off a node called something else, and a
    sync that swallowed that would be the silence this reporting exists to break.
    """
    if files:
        what = "re-importing" if reingested else "importing"
        say(f"{what} {len(files)} extraction files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_extraction_file(store, path, aliases=aliases)
        if result.ok:
            imported += 1
            for collision in result.collisions:
                say(f"{path.name}: collision: {collision}")
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
    refresh: bool = False,
) -> list[Path]:
    """The attribution files this source needs put into the graph now.

    The reasoning is :func:`_pending`'s, one layer over: a re-ingested source needs all of them,
    because re-ingesting deleted the chunks its ``SPOKE`` edges hung off. A source that was left
    alone needs only the files whose document carries no speaker at all -- a thread whose
    attribution JSON was written after it was ingested, or whose import failed at the time.

    ``refresh`` is the caller saying the files themselves changed, which no state in the graph
    can reveal: a file that gained posted dates names a document that already has speakers. Every
    file whose document the graph holds is then re-imported. Files naming a document the graph
    does not hold stay out, for the reason :func:`_pending` gives -- on a source nothing
    re-ingested, that is a raw file that is gone, and it would be reported on every run.
    """
    if reingested:
        return files
    present = store.document_ids(persona.id, source.id)
    if refresh:
        return [path for path in files if read_doc_id(path) in present]
    unattributed = present - store.attributed_document_ids(persona.id, source.id)
    if not unattributed:
        return []
    return [path for path in files if read_doc_id(path) in unattributed]


def _reimport_attribution(
    store: GraphStore,
    files: list[Path],
    say: Progress,
    attributes: AttributeTable,
    *,
    reingested: bool,
) -> tuple[int, tuple[str, ...]]:
    """Put a source's pending attribution files back into the graph.

    A file whose anchors all came loose still counts as imported: it was read and applied, and
    the loose anchors are ``attribution-import``'s report to make, not this one's. What a file
    says about its speakers is not the importer's to keep quiet about, though: an invalid
    attribute and a value two files disagree on are both said out loud here, because a refresh
    run is exactly where a rewritten sidecar's attributes land and nothing else would show them.
    """
    if files:
        what = "re-importing" if reingested else "importing"
        say(f"{what} {len(files)} attribution files")
    imported = 0
    errors: list[str] = []
    for path in files:
        result = import_attribution_file(store, path, attributes=attributes)
        if result.ok:
            imported += 1
            for problem in result.attribute_problems:
                say(f"{path.name}: attribute invalid: {problem}")
            for conflict in result.conflicts:
                say(f"{path.name}: {conflict}")
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
    refresh: bool = False,
) -> list[Path]:
    """The annotation files this source needs put into the graph now.

    :func:`_pending`'s reasoning again: a re-ingested source needs all of them, because the
    stance lives on a mention and the facets on a passage, and re-ingesting deleted both. A
    source that was left alone needs only the files whose document carries no stance and no
    facet -- annotation JSON written after the document was ingested, or whose import failed.

    ``refresh`` means the files changed, which :func:`_pending_attribution` explains: a reading
    added to a document that already carries one is invisible to the gap test.
    """
    if reingested:
        return files
    present = store.document_ids(persona.id, source.id)
    if refresh:
        return [path for path in files if read_doc_id(path) in present]
    unannotated = present - store.annotated_document_ids(persona.id, source.id)
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
    attributes: AttributeTable,
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
        result = import_annotation_file(
            store, path, aliases=aliases, facets=facets, attributes=attributes, entities=known
        )
        if result.ok:
            imported += 1
            for problem in result.attribute_problems:
                say(f"{path.name}: attribute invalid: {problem}")
        else:
            errors.append(f"{path.name}: {result.error}")
    return imported, tuple(errors)


def _backfill_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about documents that had no entities, if anything."""
    if not src.enrichment_files:
        return ""
    verb = "would import" if dry_run else "imported"
    return f", {verb} {src.enrichment_files} extraction files for documents without entities"


def _layer_phrase(count: int, layer: str, without: str, *, dry_run: bool, refreshed: bool) -> str:
    """``imported 3 attribution files for documents without speakers``, or the refresh wording.

    The two are worth telling apart in the line, because they answer different questions. The
    backfill wording says a gap was filled; the refresh wording says every file was read again
    whether or not the graph had a gap, which is what a rewritten sidecar needs and what a
    reader would otherwise mistake for the corpus having lost its speakers.
    """
    if refreshed:
        return f"{'would re-import' if dry_run else 're-imported'} {count} {layer} files"
    verb = "would import" if dry_run else "imported"
    return f"{verb} {count} {layer} files for documents without {without}"


def _attribution_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about its attribution files, if anything.

    Empty unless there were attribution files, so a corpus that uses none reads exactly as it
    did before the attribution layer existed.
    """
    if not src.attribution_files:
        return ""
    phrase = _layer_phrase(
        src.attribution_files,
        "attribution",
        "speakers",
        dry_run=dry_run,
        refreshed=src.attribution_refreshed,
    )
    return f", {phrase}"


def _annotation_clause(src: SourceReport, dry_run: bool) -> str:
    """What an up-to-date source did about its annotation files, if anything.

    Empty unless there were annotation files, so a corpus that uses none reads exactly as it did
    before the annotation layer existed.
    """
    if not src.annotation_files:
        return ""
    phrase = _layer_phrase(
        src.annotation_files,
        "annotation",
        "annotations",
        dry_run=dry_run,
        refreshed=src.annotation_refreshed,
    )
    return f", {phrase}"


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
            _layer_phrase(
                report.backfilled_attribution_files,
                "attribution",
                "speakers",
                dry_run=report.dry_run,
                refreshed=report.refreshed_attribution,
            )
        )
    if report.backfilled_annotation_files:
        clauses.append(
            _layer_phrase(
                report.backfilled_annotation_files,
                "annotation",
                "annotations",
                dry_run=report.dry_run,
                refreshed=report.refreshed_annotations,
            )
        )
    if not clauses:
        return f"{report.persona_id}: nothing to sync"
    return f"{report.persona_id}: nothing to ingest, " + ", ".join(clauses)


def summary_lines(report: SyncReport) -> list[str]:
    """One line per source, plus a closing line. Pure, so it is what the tests assert on."""
    lines: list[str] = []
    for src in report.sources:
        if src.invalid:
            lines.append(
                f"{src.source_id}: refused -- {len(src.problems)} problem(s) in "
                f"{src.invalid_files} captured file(s), {src.invalid_documents} document(s) "
                "not ingested; fix the files or pass --allow-invalid"
                f"{_backfill_clause(src, report.dry_run)}"
                f"{_attribution_clause(src, report.dry_run)}"
                f"{_annotation_clause(src, report.dry_run)}"
            )
            lines.extend(f"  {problem}" for problem in src.problems)
        elif not src.stale:
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
