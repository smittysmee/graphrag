"""Check that a captured document carries every layer the corpus expects, and that each lands.

A document reaches the graph through passes that are written at different times by different
people: the raw file is ingested, an extraction file names its entities, an attribution file
names the voices inside it, an annotation file records what its passages say and what they are
about. Nothing ties the four together, so the usual failure is silent -- a note is written, the
graph holds the text, and every question that needs entities, speakers or stance comes back
empty months later without anyone noticing that three of the four passes never happened.

This module answers the question directly, for one document or for a whole persona: which
sidecars exist on disk, which layers the graph actually holds, and whether each sidecar still
imports cleanly. Both halves are needed, because they fail apart. A sidecar on disk that was
never imported leaves the graph empty; a graph layer with no sidecar cannot survive a re-ingest,
which deletes the chunks that mentions, ``SPOKE`` edges, stances and facets all hang off.

Every sidecar found is re-run through its own importer with ``dry_run=True``, so this never
writes and never differs from what a real import would do. What those runs report as unplaced --
an entity name that occurs in no passage, a post anchor that matches nothing, an annotation
pointing at an entity the passage does not name -- is collected here as *loose*, counted by
reason, because each reason sends a reviewer to a different file.

Node attributes ride the same check. All three sidecars may say what kind of thing they are
about -- an extraction file what kind of entity, an attribution file what kind of speaker, an
annotation file what kind of document -- and a key or value the persona's vocabulary does not
declare is counted here as a loose entry with its own reason, ``attribute invalid``: nothing is
written, so a tagging pass that invented a value fails visibly rather than half-landing. The
three are kept apart because each sends a reviewer to a different file.

Nothing here knows about any particular corpus: the persona's own ``persona.yaml`` supplies the
sources, and its ``aliases.yaml`` and ``facets.yaml`` supply the vocabulary the importers check
against.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.extract.aliases import EMPTY_ALIASES, AliasTable
from graphrag.extract.annotations import (
    LOOSE_REASONS,
    EntityIndex,
    FacetTable,
    import_annotation_file,
)
from graphrag.extract.attributes import EMPTY_ATTRIBUTES, AttributeTable
from graphrag.extract.attribution import import_attribution_file
from graphrag.extract.importer import import_extraction_file, read_doc_id
from graphrag.graph.store import GraphStore
from graphrag.models import PersonaSpec, SourceSpec
from graphrag.textutil import slugify

__all__ = [
    "ATTRIBUTE_INVALID",
    "LAYERS",
    "LOOSE_LABELS",
    "DocumentCheck",
    "GraphLayers",
    "LayerReport",
    "SidecarCheck",
    "SourceNotFoundError",
    "check_documents",
    "doc_id_for_file",
    "index_sidecars",
    "read_graph_layers",
    "source_base",
    "summary_lines",
]

#: The layers a captured document can carry, in the order the contract asks for them.
LAYERS: tuple[str, ...] = ("extraction", "attribution", "annotation")

#: A key or value the persona's vocabulary does not declare. Both sidecar kinds can carry one,
#: and both mean the same thing -- a tagger invented a key or a value, and it was dropped -- so
#: the reason is the same in each. The label names the sidecar to open, which is not the same:
#: one is a post in an attribution file, the other the top of an annotation file.
ATTRIBUTE_INVALID = "attribute invalid"

#: Why one entry of a sidecar did not land, keyed ``<layer>/<reason>``. The reasons are kept
#: apart rather than summed because each sends a reviewer somewhere else: a non-verbatim entity
#: name is a fix in the extraction file, an anchor that matched nothing is a paraphrased quote,
#: an entity the persona has never heard of is a spelling for the alias table.
LOOSE_LABELS: dict[str, str] = {
    "extraction/loose": "entity name not verbatim in any passage",
    "extraction/unmatched": "entity name in no passage",
    "extraction/attribute": f"entity {ATTRIBUTE_INVALID}",
    "attribution/anchor": "post anchor not found",
    "attribution/attribute": f"speaker {ATTRIBUTE_INVALID}",
    **{f"annotation/{reason}": text for reason, text in LOOSE_REASONS.items()},
    "annotation/attribute": f"document {ATTRIBUTE_INVALID}",
}

#: How much of a name or an anchor is echoed when an entry is reported as loose.
_LABEL = 80


class SourceNotFoundError(LookupError):
    """No source of the persona claims the given raw file."""


# ----------------------------------------------------------------------------- results


@dataclass(frozen=True)
class SidecarCheck:
    """One sidecar file: where it is, what a dry run made of it, and what came loose."""

    layer: str
    path: Path
    error: str = ""
    summary: str = ""
    loose: tuple[tuple[str, str], ...] = ()  # (``<layer>/<reason>``, the entry)

    @property
    def ok(self) -> bool:
        return not self.error and not self.loose


@dataclass(frozen=True)
class DocumentCheck:
    """One document: the sidecars it has, and the layers the graph holds for it."""

    doc_id: str
    source_id: str = ""
    in_graph: bool = False
    mentions: bool = False
    speakers: bool = False
    annotations: bool = False
    #: How many document attributes the graph holds for this id. An annotation file can carry
    #: nothing but a top-level ``attributes`` block and no ``annotations`` entries, which sets
    #: this without ever touching a chunk facet or a mention stance, so the table prints it
    #: alongside ``annotations`` rather than folding it into that one flag.
    attribute_count: int = 0
    sidecars: Mapping[str, SidecarCheck] = field(default_factory=dict)

    def sidecar(self, layer: str) -> SidecarCheck | None:
        return self.sidecars.get(layer)

    @property
    def has_extraction(self) -> bool:
        return "extraction" in self.sidecars

    @property
    def loose(self) -> tuple[tuple[str, str], ...]:
        return tuple(item for check in self.sidecars.values() for item in check.loose)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(f"{c.path.name}: {c.error}" for c in self.sidecars.values() if c.error)

    def graph_holds(self, layer: str) -> bool:
        """Whether the graph carries this layer for this document."""
        return {
            "extraction": self.mentions,
            "attribution": self.speakers,
            "annotation": self.annotations,
        }.get(layer, False)

    def state(self, layer: str) -> str:
        """``ok``, ``file``, ``graph`` or ``-``: where this layer is, and where it is not.

        ``file`` means a sidecar exists that the graph does not reflect, which is what an import
        that never ran looks like. ``graph`` means the graph holds the layer with no sidecar
        behind it, which a re-ingest would silently drop (transcripts are the honest case: their
        loader parses speaker turns, so the speakers were never written by hand).
        """
        on_disk = layer in self.sidecars
        in_graph = self.graph_holds(layer)
        if on_disk and in_graph:
            return "ok"
        if on_disk:
            return "file"
        return "graph" if in_graph else "-"


@dataclass(frozen=True)
class LayerReport:
    """What :func:`check_documents` found for one persona."""

    persona_id: str
    documents: tuple[DocumentCheck, ...] = ()

    @property
    def without_extraction(self) -> tuple[str, ...]:
        """Documents with no extraction JSON at all -- the one sidecar every document needs."""
        return tuple(d.doc_id for d in self.documents if not d.has_extraction)

    @property
    def not_in_graph(self) -> tuple[str, ...]:
        return tuple(d.doc_id for d in self.documents if not d.in_graph)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(err for d in self.documents for err in d.errors)

    @property
    def loose_totals(self) -> dict[str, int]:
        """Loose entries counted by reason, in ``LOOSE_LABELS`` order; zero reasons included."""
        totals = dict.fromkeys(LOOSE_LABELS, 0)
        for document in self.documents:
            for reason, _entry in document.loose:
                totals[reason] = totals.get(reason, 0) + 1
        return totals

    @property
    def total_loose(self) -> int:
        return sum(self.loose_totals.values())

    def present(self, layer: str) -> int:
        """How many checked documents have this layer's sidecar on disk."""
        return sum(1 for d in self.documents if layer in d.sidecars)

    def in_graph(self, layer: str) -> int:
        """How many checked documents have this layer in the graph."""
        return sum(1 for d in self.documents if d.graph_holds(layer))

    @property
    def ok(self) -> bool:
        """True when every listed document has extraction JSON and nothing came loose."""
        return not self.without_extraction and not self.errors and self.total_loose == 0


# ----------------------------------------------------------------------------- disk


def index_sidecars(root: Path | None, persona_id: str) -> dict[str, Path]:
    """Sidecars under ``root`` belonging to ``persona_id``, keyed by the ``doc_id`` inside them.

    Keyed on the id inside each file rather than the directory it sits in, for the reason
    :func:`graphrag.sync.index_extractions` gives: a corpus holds both the per-source layout and
    an older flat one, and the id is what actually ties a file to a document. ``None``, or a
    directory that is not there, means the caller keeps no such layer, which is not a finding.
    """
    out: dict[str, Path] = {}
    if root is None or not root.is_dir():
        return out
    prefix = f"{persona_id}:"
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        doc_id = read_doc_id(path)
        if doc_id.startswith(prefix):
            out.setdefault(doc_id, path)
    return out


def source_base(raw_root: Path, source: SourceSpec) -> Path:
    """The directory a source's ``glob`` is evaluated against (mirrors the loaders)."""
    return raw_root / source.path if source.path else raw_root


def doc_id_for_file(persona: PersonaSpec, raw_root: Path, path: Path) -> tuple[str, SourceSpec]:
    """The id the persona's loaders would give ``path``, and the source that claims it.

    The same two rules the loaders use, chosen by the source's loader: a ``documents`` source
    slugifies the file's path under the source, without its suffix; a ``transcripts`` source
    slugifies the containing folder, because one folder is one recording. The persona's sources
    decide which applies, so nothing here is specific to a corpus.

    A file under two sources belongs to the more specific one (the deepest base directory).
    Raises :class:`SourceNotFoundError` when no source of this persona covers the path.
    """
    target = _resolve(path)
    best: tuple[int, SourceSpec, Path] | None = None
    for source in persona.sources:
        base = _resolve(source_base(raw_root, source))
        try:
            rel = target.relative_to(base)
        except ValueError:
            continue
        depth = len(base.parts)
        if best is None or depth > best[0]:
            best = (depth, source, rel)
    if best is None:
        bases = [str(_resolve(source_base(raw_root, s))) for s in persona.sources]
        known = ", ".join(bases) or "(none)"
        msg = f"no source of {persona.id} covers {target}; sources live under {known}"
        raise SourceNotFoundError(msg)
    _depth, source, rel = best
    if source.loader == "transcripts":
        folder = rel.parent.name
        slug = slugify(folder if folder not in ("", ".") else rel.stem)
    else:
        slug = slugify(str(rel.with_suffix("")))
    return f"{persona.id}:{source.id}:{slug}", source


def _resolve(path: Path) -> Path:
    """``path`` made absolute against the working directory, with ``.`` and ``..`` folded out.

    The target and the source bases go through the same call because they arrive in different
    shapes and comparing them as they come is meaningless. A base is built from settings, which
    keep ``data/raw`` relative to wherever the command runs -- ``/app`` in the container -- while
    the target is whatever the caller typed: ``data/raw/...`` from a shell, or an absolute path
    from a hook or an editor. Anchoring both to the working directory makes either form answer.
    """
    return Path(_normalise(path if path.is_absolute() else Path.cwd() / path))


def _normalise(path: Path) -> str:
    """``path`` with ``..`` and ``.`` resolved, without touching the filesystem.

    ``Path.resolve`` would follow symlinks and needs the file to exist; a check can legitimately
    be asked about a path that was just written, or one behind a mounted link.
    """
    parts: list[str] = []
    for part in path.parts:
        if part == "..":
            if parts and parts[-1] not in ("", "/"):
                parts.pop()
        elif part != ".":
            parts.append(part)
    return str(Path(*parts)) if parts else "."


# ----------------------------------------------------------------------------- graph


@dataclass(frozen=True)
class GraphLayers:
    """Which documents the graph holds for a persona, and which layers it holds for each."""

    documents: frozenset[str] = frozenset()
    mentions: frozenset[str] = frozenset()
    speakers: frozenset[str] = frozenset()
    annotations: frozenset[str] = frozenset()
    #: How many document attributes the graph holds, for every document of the persona that has
    #: any (not just the ones in ``documents``) -- cheap to read whole, since it is one call.
    attribute_counts: Mapping[str, int] = field(default_factory=dict)


def read_graph_layers(
    store: GraphStore, persona: PersonaSpec, source_id: str | None = None
) -> GraphLayers:
    """Read the persona's document ids and its three layer sets, one source at a time.

    Four set reads per source rather than one read per document: the store already answers each
    of these per source, and a persona with three hundred documents would otherwise be three
    hundred round trips. Document attribute counts are read once for the whole persona, since
    ``document_attributes`` is not scoped to a source.
    """
    sources = [s for s in persona.sources if source_id is None or s.id == source_id]
    documents: set[str] = set()
    mentions: set[str] = set()
    speakers: set[str] = set()
    annotations: set[str] = set()
    for source in sources:
        documents |= store.document_ids(persona.id, source.id)
        mentions |= store.enriched_document_ids(persona.id, source.id)
        speakers |= store.attributed_document_ids(persona.id, source.id)
        annotations |= store.annotated_document_ids(persona.id, source.id)
    attribute_counts = {
        doc_id: len(attrs) for doc_id, attrs in store.document_attributes(persona.id).items()
    }
    return GraphLayers(
        documents=frozenset(documents),
        mentions=frozenset(mentions),
        speakers=frozenset(speakers),
        annotations=frozenset(annotations),
        attribute_counts=attribute_counts,
    )


# ----------------------------------------------------------------------------- the check


def check_documents(
    store: GraphStore,
    persona: PersonaSpec,
    *,
    enrichment_root: Path,
    attribution_root: Path | None = None,
    annotation_root: Path | None = None,
    doc_ids: Sequence[str] | None = None,
    source_id: str | None = None,
    aliases: AliasTable = EMPTY_ALIASES,
    facets: FacetTable | None = None,
    attributes: AttributeTable = EMPTY_ATTRIBUTES,
) -> LayerReport:
    """Check every layer of ``doc_ids``, or of the whole persona when they are not given.

    Nothing is written: each sidecar is run through its own importer with ``dry_run=True``, so
    what this reports is exactly what an import would do. An id the graph does not hold is still
    checked and still reported, because a sidecar written for a document that was never ingested
    is a common way for a capture to look finished and be empty.
    """
    layers = read_graph_layers(store, persona, source_id)
    found = {
        "extraction": index_sidecars(enrichment_root, persona.id),
        "attribution": index_sidecars(attribution_root, persona.id),
        "annotation": index_sidecars(annotation_root, persona.id),
    }
    wanted = _wanted_ids(layers, found, doc_ids, persona.id, source_id)
    known: EntityIndex | None = None
    checks: list[DocumentCheck] = []
    for doc_id in wanted:
        sidecars: dict[str, SidecarCheck] = {}
        for layer in LAYERS:
            path = found[layer].get(doc_id)
            if path is None:
                continue
            if layer == "annotation" and known is None:
                known = EntityIndex.build(store, persona.id)
            sidecars[layer] = _dry_run(store, layer, path, aliases, facets, attributes, known)
        checks.append(
            DocumentCheck(
                doc_id=doc_id,
                source_id=_source_of(doc_id),
                in_graph=doc_id in layers.documents,
                mentions=doc_id in layers.mentions,
                speakers=doc_id in layers.speakers,
                annotations=doc_id in layers.annotations,
                attribute_count=layers.attribute_counts.get(doc_id, 0),
                sidecars=sidecars,
            )
        )
    return LayerReport(persona_id=persona.id, documents=tuple(checks))


def _wanted_ids(
    layers: GraphLayers,
    found: Mapping[str, Mapping[str, Path]],
    doc_ids: Sequence[str] | None,
    persona_id: str,
    source_id: str | None,
) -> list[str]:
    """The ids to check: the ones asked for, or every id the graph or the sidecars know.

    Sidecar ids are included in the ``--all`` case on purpose: a file written for a document
    that was never ingested is invisible to a check that only walks the graph, and that is the
    case worth catching.
    """
    if doc_ids is not None:
        return list(dict.fromkeys(doc_ids))
    everything = set(layers.documents)
    for index in found.values():
        everything |= set(index)
    prefix = f"{persona_id}:{source_id}:" if source_id else f"{persona_id}:"
    return sorted(doc_id for doc_id in everything if doc_id.startswith(prefix))


def _source_of(doc_id: str) -> str:
    """The source id inside ``<persona>:<source>:<slug>``, or an empty string."""
    parts = doc_id.split(":")
    return parts[1] if len(parts) >= 3 else ""


def _dry_run(
    store: GraphStore,
    layer: str,
    path: Path,
    aliases: AliasTable,
    facets: FacetTable | None,
    attributes: AttributeTable,
    known: EntityIndex | None,
) -> SidecarCheck:
    """Run one sidecar through its own importer without writing, and record what came loose."""
    if layer == "extraction":
        result = import_extraction_file(
            store, path, dry_run=True, aliases=aliases, attributes=attributes
        )
        if not result.ok:
            return SidecarCheck(layer=layer, path=path, error=result.error)
        return SidecarCheck(
            layer=layer,
            path=path,
            summary=(
                f"{result.entities} entities, {result.mentions} mentions, "
                f"{result.relations} relations"
                + (f", {result.attributes} attributes" if result.attributes else "")
                + (f", {len(result.dangling)} dangling" if result.dangling else "")
            ),
            loose=tuple(("extraction/loose", _short(name)) for name in result.loose)
            + tuple(("extraction/unmatched", _short(name)) for name in result.unmatched)
            + tuple(("extraction/attribute", _short(entry)) for entry in result.attribute_problems),
        )
    if layer == "attribution":
        posts = import_attribution_file(store, path, dry_run=True, attributes=attributes)
        if not posts.ok:
            return SidecarCheck(layer=layer, path=path, error=posts.error)
        return SidecarCheck(
            layer=layer,
            path=path,
            summary=f"{posts.attached}/{posts.posts} posts, {len(posts.speakers)} speakers"
            + (f", {posts.attributes} attributes" if posts.attributes else ""),
            loose=tuple(("attribution/anchor", _short(entry)) for entry in posts.loose)
            + tuple(("attribution/attribute", _short(entry)) for entry in posts.attribute_problems),
        )
    readings = import_annotation_file(
        store,
        path,
        dry_run=True,
        aliases=aliases,
        facets=facets,
        attributes=attributes,
        entities=known,
    )
    if not readings.ok:
        return SidecarCheck(layer=layer, path=path, error=readings.error)
    return SidecarCheck(
        layer=layer,
        path=path,
        summary=(
            f"{readings.applied}/{readings.annotations} annotations, "
            f"{readings.stances} stances, {readings.facets} facets"
            + (f", {readings.attributes} attributes" if readings.attributes else "")
            + (
                f", {len(readings.unknown_facets)} unknown facets"
                if readings.unknown_facets
                else ""
            )
        ),
        loose=tuple((f"annotation/{item.reason}", _short(item.detail)) for item in readings.loose)
        + tuple(("annotation/attribute", _short(entry)) for entry in readings.attribute_problems),
    )


def _short(text: str) -> str:
    return text if len(text) <= _LABEL else text[: _LABEL - 1] + "…"


# ----------------------------------------------------------------------------- reporting


def summary_lines(report: LayerReport) -> list[str]:
    """The lines printed under the table. Pure, so this is what the tests assert on."""
    lines = [f"{report.persona_id}: {len(report.documents)} documents checked"]
    for layer in LAYERS:
        lines.append(
            f"  {layer}: {report.present(layer)} sidecars on disk, "
            f"{report.in_graph(layer)} in the graph"
        )
    missing = report.without_extraction
    if missing:
        lines.append(f"no extraction JSON: {len(missing)} ({_names(missing)})")
    absent = report.not_in_graph
    if absent:
        lines.append(f"not in the graph: {len(absent)} ({_names(absent)})")
    for err in report.errors:
        lines.append(f"error: {err}")
    totals = report.loose_totals
    if report.total_loose:
        lines.append(f"loose entries: {report.total_loose}")
        lines.extend(
            f"  {count} {LOOSE_LABELS[reason]}"
            for reason, count in totals.items()
            if count and reason in LOOSE_LABELS
        )
    lines.append("every layer complete" if report.ok else "layers incomplete")
    return lines


def _names(ids: Iterable[str], limit: int = 3) -> str:
    listed = list(ids)
    shown = ", ".join(listed[:limit])
    more = len(listed) - min(len(listed), limit)
    return f"{shown} (+{more} more)" if more else shown
