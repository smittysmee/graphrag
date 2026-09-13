"""Import passage annotations an agent wrote as JSON: what a passage says, and what it is about.

Extraction records that a passage mentions something. It does not record whether the passage is
praising it, complaining about it, or saying it was replaced by something else -- so "most
discussed" is the only question the entity layer can answer, and every reading of the corpus
that depends on sentiment or on which function of a subject is under discussion has to be done
by hand, again, each time.

The annotation layer fills that in the way attribution fills in speakers. An agent reads one
document and writes one file naming the passages it wants to mark, each with a verbatim
anchor::

    {"doc_id": "<persona>:<source>:<slug>",
     "annotations": [
       {"anchor": "verbatim six to twenty words of the passage",
        "entity": "Entity name as written in that passage",
        "stance": "praise | complaint | substitution | neutral",
        "facets": ["quoting", "outage"],
        "note": "optional short paraphrase, never quoted back as evidence"}
     ]}

``anchor`` is required; ``entity`` is optional, and ``stance`` needs one, because a stance with
nothing to be about is not a reading. Anchors are placed exactly as attribution places them: the
first passage containing the anchor, matched case-insensitively with whitespace folded.

The entity is then checked against the graph rather than trusted. If the passage already
mentions it, the stance goes on that edge. If it does not, the annotation is not thrown away on
the spot: extraction and annotation read a document separately, so a passage can name a thing
the extraction pass did not list there, or listed under another spelling. When the persona
already has an entity of that name and one of its spellings occurs in the passage, the mention
is created and the stance goes on it. When the persona has no such entity, or none of its
spellings is in the passage, the annotation is reported as loose and skipped -- creating
entities is the extraction pass's job, and a mention the passage does not support would be a
reading of nothing.

Skipped annotations are counted by reason, because the three send a reviewer to different
files: an anchor that matched nothing is a paraphrased quote, an entity the persona has never
heard of is a spelling for the alias table or a gap in extraction, and an entity missing from
the passage is an annotation pointing at the wrong one.

When the persona keeps a ``facets.yaml``, facets outside it are reported and dropped, per
annotation, so one invented slug costs that slug rather than the file.

Stance goes on the ``MENTIONS`` edge, facets go on the passage; both ride the snapshot. Re-import
is a no-op, which is what lets :mod:`graphrag.sync` re-run these files after a re-ingest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, StringConstraints, model_validator

from graphrag.extract.aliases import EMPTY_ALIASES, AliasTable, fold_name
from graphrag.extract.attribution import find_anchor, fold_passage
from graphrag.graph.store import GraphStore
from graphrag.models import Enrichment, Entity, Mention, Stance
from graphrag.textutil import sanitize_inline, slugify

__all__ = [
    "FACET_FILE",
    "LOOSE_REASONS",
    "Annotation",
    "AnnotationResult",
    "DocumentAnnotations",
    "EntityIndex",
    "FacetError",
    "FacetTable",
    "LooseAnnotation",
    "LooseReason",
    "import_annotation_file",
    "load_facets",
    "load_persona_facets",
    "loose_totals",
]

#: Why an annotation was skipped, and what a reviewer does about each one. The three are worth
#: telling apart: they send you to a different file. An anchor that matched nothing is a
#: paraphrased quote to fix in the annotation; an entity the persona has never heard of is a
#: spelling to add to the alias table, or a document the extraction pass has not covered; an
#: entity whose name is nowhere in the passage is an annotation pointing at the wrong passage.
LooseReason = Literal["anchor", "unknown-entity", "entity-absent"]

LOOSE_REASONS: dict[LooseReason, str] = {
    "anchor": "anchor not found",
    "unknown-entity": "entity unknown to the persona",
    "entity-absent": "entity not in the passage",
}

#: Where a persona keeps the functions its corpus talks about, beside ``persona.yaml``.
FACET_FILE = "facets.yaml"

#: Enough to cover any single document; the store pages, so this is one read either way.
_ALL_CHUNKS = 100_000
#: How much of an anchor or a name is echoed when an annotation is reported as loose.
_LABEL = 80

#: A blank anchor is a mistake worth failing the file for, not a silent skip.
Required = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FacetError(ValueError):
    """The facet file is unreadable or malformed."""


@dataclass(frozen=True)
class FacetTable:
    """The facets a persona recognises, and one sentence saying what each one covers.

    An absent file is not an empty table: it means the persona has not declared its vocabulary
    yet, and every facet is accepted. :func:`load_persona_facets` returns ``None`` for that case
    so the two are never confused.
    """

    definitions: Mapping[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.definitions)

    def known(self, facet: str) -> bool:
        return facet in self.definitions


def load_facets(path: Path) -> FacetTable:
    """Read one ``facets.yaml``: ``facets: {slug: one sentence}``."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"cannot read facet file {path}: {exc}"
        raise FacetError(msg) from exc
    if data is None:
        return FacetTable()
    if not isinstance(data, Mapping):
        msg = f"{path}: expected a mapping with a 'facets' key, got {type(data).__name__}"
        raise FacetError(msg)
    raw = data.get("facets") or {}
    if not isinstance(raw, Mapping):
        msg = f"{path}: 'facets' must be a mapping of slug to a one-sentence description"
        raise FacetError(msg)
    definitions: dict[str, str] = {}
    for key, description in raw.items():
        facet = slugify(str(key))
        if not facet:
            msg = f"{path}: a facet key is blank"
            raise FacetError(msg)
        definitions[facet] = str(description or "").strip()
    return FacetTable(definitions=definitions)


def load_persona_facets(personas_dir: Path, persona_id: str) -> FacetTable | None:
    """The persona's facet vocabulary, or ``None`` when it declares none and anything goes."""
    path = personas_dir / persona_id / FACET_FILE
    return load_facets(path) if path.is_file() else None


class Annotation(BaseModel):
    """One reading of one passage: where it is, what it is about, and what it says."""

    anchor: Required
    entity: str = ""
    stance: Stance | None = None
    facets: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _stance_needs_an_entity(self) -> Annotation:
        if self.stance is not None and not self.entity.strip():
            msg = "stance requires an entity: a stance about nothing is not a reading"
            raise ValueError(msg)
        return self

    @property
    def slugs(self) -> list[str]:
        """The facets as slugs, deduplicated, in the order the file lists them."""
        out: list[str] = []
        for facet in self.facets:
            slug = slugify(facet)
            if slug and slug not in out:
                out.append(slug)
        return out


class DocumentAnnotations(BaseModel):
    """One document's annotations as produced by an agent (``graphrag annotations-import``)."""

    doc_id: str
    annotations: list[Annotation] = Field(default_factory=list)


@dataclass(frozen=True)
class EntityIndex:
    """Every entity a persona already has, keyed by every spelling it answers to.

    Built once per run rather than per file: the annotation pass never creates an entity, so
    what the persona knows does not change while its files are being read.
    """

    by_key: Mapping[str, Entity] = field(default_factory=dict)

    @classmethod
    def build(cls, store: GraphStore, persona_id: str) -> EntityIndex:
        keys: dict[str, Entity] = {}
        for entity in store.persona_entities(persona_id):
            for key in (
                entity.id,
                fold_name(entity.name),
                *(fold_name(alias) for alias in entity.aliases),
            ):
                keys.setdefault(key, entity)
        return cls(by_key=keys)

    def find(self, name: str) -> Entity | None:
        return self.by_key.get(fold_name(name)) or self.by_key.get(name.strip())


@dataclass(frozen=True)
class LooseAnnotation:
    """One skipped annotation: why it was skipped, and enough of it to find it by."""

    reason: LooseReason
    detail: str

    def __str__(self) -> str:
        return f"{LOOSE_REASONS[self.reason]}: {self.detail}"


@dataclass(frozen=True)
class AnnotationResult:
    """One annotation file's outcome, including what a reviewer should look at.

    ``applied`` counts annotations that reached the graph, ``stances`` and ``facets`` count what
    they wrote and ``created`` the mentions the fallback added. ``loose`` holds the annotations
    that were skipped, each carrying the reason it was, and ``unknown_facets`` the slugs dropped
    for being outside the persona's declared vocabulary.
    """

    path: Path
    doc_id: str = ""
    error: str = ""
    annotations: int = 0
    applied: int = 0
    stances: int = 0
    facets: int = 0
    created: int = 0
    entities: tuple[str, ...] = ()
    loose: tuple[LooseAnnotation, ...] = ()
    unknown_facets: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def loose_by_reason(self) -> dict[LooseReason, int]:
        return {
            reason: sum(1 for item in self.loose if item.reason == reason)
            for reason in LOOSE_REASONS
        }


def import_annotation_file(
    store: GraphStore,
    path: Path,
    *,
    dry_run: bool = False,
    aliases: AliasTable = EMPTY_ALIASES,
    facets: FacetTable | None = None,
    entities: EntityIndex | None = None,
) -> AnnotationResult:
    """Validate one annotation file and write its stances and facets into the graph.

    Bad input never raises: an unreadable file, invalid JSON or a ``doc_id`` the graph does not
    know comes back as a result whose ``error`` is set, so a caller importing hundreds of files
    reports them all instead of stopping at the first.

    ``entities`` is the persona's entity index. A caller reading many files should build it once
    and pass it; left out, it is built from the store the first time an annotation needs it.
    """
    try:
        payload = DocumentAnnotations.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return AnnotationResult(path=path, error=f"invalid: {exc}")
    chunks = store.document_chunks(payload.doc_id, 0, _ALL_CHUNKS)
    if not chunks:
        return AnnotationResult(
            path=path, doc_id=payload.doc_id, error=f"unknown document {payload.doc_id}"
        )

    folded = [(c.id, fold_passage(c.text)) for c in chunks]
    passages = dict(folded)
    mentioned: dict[str, dict[str, str]] = {}
    known = entities
    named: list[str] = []
    loose: list[LooseAnnotation] = []
    unknown: list[str] = []
    applied = stances = facet_writes = created = 0

    for item in payload.annotations:
        chunk_id = find_anchor(item.anchor, folded)
        if chunk_id is None:
            loose.append(LooseAnnotation("anchor", sanitize_inline(item.anchor, _LABEL)))
            continue
        keep = [f for f in item.slugs if facets is None or facets.known(f)]
        unknown.extend(f for f in item.slugs if f not in keep)
        if not item.stance and not keep:
            # Nothing is being asserted, so there is nothing to write and nothing to report.
            continue
        name = ""
        if item.entity.strip():
            canonical = aliases.canonical(item.entity)
            name = _resolve_entity(store, chunk_id, canonical, mentioned)
            if not name:
                if known is None:
                    known = EntityIndex.build(store, chunks[0].persona_id)
                entity = known.find(canonical)
                where = (
                    f"{sanitize_inline(item.entity, _LABEL)} at "
                    f"{sanitize_inline(item.anchor, _LABEL)}"
                )
                if entity is None:
                    loose.append(LooseAnnotation("unknown-entity", where))
                    continue
                if not _occurs_in(entity, passages[chunk_id]):
                    loose.append(LooseAnnotation("entity-absent", where))
                    continue
                created += 1
                name = entity.name
                # The passage mentions it now, so a later annotation in this file resolves.
                mentioned[chunk_id][fold_name(entity.name)] = entity.name
                mentioned[chunk_id][entity.id] = entity.name
                if not dry_run:
                    _add_mention(store, chunk_id, entity, item.stance)
        applied += 1
        if item.stance and name:
            stances += 1
            if name not in named:
                named.append(name)
            if not dry_run:
                store.annotate_mention(payload.doc_id, chunk_id, name, item.stance)
        if keep:
            facet_writes += len(keep)
            if not dry_run:
                store.annotate_chunk(payload.doc_id, chunk_id, keep)

    return AnnotationResult(
        path=path,
        doc_id=payload.doc_id,
        annotations=len(payload.annotations),
        applied=applied,
        stances=stances,
        facets=facet_writes,
        created=created,
        entities=tuple(named),
        loose=tuple(loose),
        unknown_facets=tuple(dict.fromkeys(unknown)),
    )


def _resolve_entity(
    store: GraphStore, chunk_id: str, name: str, cache: dict[str, dict[str, str]]
) -> str:
    """The stored name of an entity this passage already mentions, or an empty string.

    Tried first, because a stance attaches to a ``MENTIONS`` edge and an edge that is already
    there needs nothing invented. Ids are accepted alongside names, so a caller that already
    knows the node can say so. When this comes back empty the fallback takes over.
    """
    if chunk_id not in cache:
        cache[chunk_id] = {
            key: entity.name
            for entity in store.entities_for_chunks([chunk_id])
            for key in (fold_name(entity.name), entity.id)
        }
    return cache[chunk_id].get(fold_name(name)) or cache[chunk_id].get(name.strip(), "")


def _occurs_in(entity: Entity, passage: str) -> bool:
    """Whether the passage names this entity, under its own name or any spelling folded into it.

    What bounds the fallback. Extraction and annotation read a document separately, so a passage
    can name a thing the extraction pass did not list there, or listed under another spelling,
    and the reading is still true -- but the evidence has to be the passage rather than the
    annotation asserting it. Matched on the folded text, so case and whitespace do not decide it.
    """
    return any(
        fold_passage(spelling) in passage
        for spelling in (entity.name, *entity.aliases)
        if spelling.strip()
    )


def _add_mention(store: GraphStore, chunk_id: str, entity: Entity, stance: Stance | None) -> None:
    """Attach an entity the persona already has to a passage that names it, stance and all.

    Written through ``upsert_enrichment`` rather than a write of its own, so the edge lands
    exactly as an extraction file's would and a re-import is the same no-op.
    """
    store.upsert_enrichment(
        Enrichment(
            entities=[entity],
            mentions=[Mention(chunk_id=chunk_id, entity_id=entity.id, stance=stance)],
        )
    )


def loose_totals(results: Iterable[AnnotationResult]) -> dict[LooseReason, int]:
    """Skipped annotations across a run, counted by reason, in ``LOOSE_REASONS`` order."""
    totals: dict[LooseReason, int] = dict.fromkeys(LOOSE_REASONS, 0)
    for result in results:
        for item in result.loose:
            totals[item.reason] += 1
    return totals


def loose_summary(totals: Mapping[LooseReason, int]) -> str:
    """``18 anchor not found, 190 entity unknown to the persona`` -- reasons that happened."""
    return ", ".join(
        f"{count} {LOOSE_REASONS[reason]}" for reason, count in totals.items() if count
    )
