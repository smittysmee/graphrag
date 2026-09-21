"""Import entity/relation extractions an agent wrote as JSON.

The agent-driven enrichment path writes one ``{doc_id, entities, relations}`` file per
document. Two callers read those files: ``graphrag enrich-import`` and :mod:`graphrag.sync`,
which re-imports a source's files after a re-ingest (re-ingesting deletes a document's chunks,
and mentions hang off chunks). Both go through :func:`import_extraction_file` so a file lands
in the graph the same way whoever asked for it.

An entity may carry attributes of its own, beside its description::

    {"name": "Acme", "type": "company", "attributes": {"sector": "fintech"}}

They describe the *thing*, not the document that names it, which is what makes them usable for
network analysis: an entity's attributes are the same whichever passage mentions it, so an
attribute measured over the entity network is a property of the nodes rather than of the
documents the edges were drawn from. :mod:`graphrag.sna.export` says why that distinction
decides whether an assortativity number means anything.

They are checked against the persona's vocabulary (:mod:`graphrag.extract.attributes`) and
written onto the ``Entity`` node, with the speaker layer's rule for disagreement: one entity is
named by many documents, so many files can claim it, the first value written stands, and a
second file that says something else is reported as a conflict rather than overwriting. Which
of two readings is right is a question for whoever wrote them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from graphrag.extract.aliases import EMPTY_ALIASES, AliasTable, apply_aliases, fold_name
from graphrag.extract.attributes import EMPTY_ATTRIBUTES, AttributeTable
from graphrag.extract.llm import (
    DocumentExtraction,
    ExtractedEntity,
    match_chunks,
    result_to_enrichment,
)
from graphrag.graph.store import GraphStore
from graphrag.models import Entity
from graphrag.textutil import sanitize_inline

__all__ = ["ImportResult", "import_extraction_file", "read_doc_id"]

#: Enough to cover any single document; the store pages, so this is one read either way.
_ALL_CHUNKS = 100_000
#: How much of a name or value is echoed when a claim about an entity is reported as conflicting.
_LABEL = 80


@dataclass(frozen=True)
class ImportResult:
    """One extraction file's outcome, including the names a reviewer should look at.

    ``loose`` names were matched by token rather than verbatim, ``unmatched`` ones occur in no
    passage at all (their mention falls back to the first chunk), and ``dangling`` relations
    point at an entity the file never declared. ``renamed`` names were folded onto a canonical
    spelling by the persona's alias table.

    ``collisions`` are the names whose id already belonged to a node called something else --
    ``<incoming> kept as <existing>``. The graph was not renamed and the mentions still landed,
    so the fix is a human one: give the entity a name whose id differs, or say in the persona's
    ``aliases.yaml`` that the two spellings are one thing.

    ``attributes`` counts the entity attributes written, ``attribute_problems`` the keys the
    persona's vocabulary refused, and ``attribute_conflicts`` the ones a stored value already
    contradicts. The last two are different findings, exactly as they are in the attribution
    layer: a problem is a file to fix, a conflict is two documents disagreeing about one entity,
    and only the people who wrote them can say which is right.
    """

    path: Path
    doc_id: str = ""
    error: str = ""
    entities: int = 0
    mentions: int = 0
    relations: int = 0
    attributes: int = 0
    loose: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()
    dangling: tuple[str, ...] = ()
    renamed: tuple[str, ...] = ()
    collisions: tuple[str, ...] = ()
    attribute_problems: tuple[str, ...] = ()
    attribute_conflicts: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error


def import_extraction_file(
    store: GraphStore,
    path: Path,
    *,
    dry_run: bool = False,
    aliases: AliasTable = EMPTY_ALIASES,
    attributes: AttributeTable = EMPTY_ATTRIBUTES,
) -> ImportResult:
    """Validate one extraction file and upsert it into the graph.

    Bad input never raises: an unreadable file, invalid JSON or a ``doc_id`` the graph does not
    know comes back as a result whose ``error`` is set, so a caller importing hundreds of files
    can report them all instead of stopping at the first.

    ``aliases`` is applied after the passages have been matched, so an entity is still found by
    the spelling its passage uses and only the node it lands on is canonical. A file imported
    with a table therefore never creates the alias node that ``graphrag aliases apply`` exists
    to clean up. Entity attributes follow the name onto the canonical node, for the same reason:
    a claim about "JTBD framework" is a claim about whatever node that spelling folds onto.

    ``attributes`` is the persona's vocabulary. A key or value it does not declare is reported
    and dropped, per key, so one invented value costs that key rather than the file.
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
    enrichment = apply_aliases(result_to_enrichment(payload, chunks), aliases)
    collisions = () if dry_run else tuple(c.line() for c in store.upsert_enrichment(enrichment))
    claimed, problems = _claimed_attributes(payload.entities, aliases, attributes)
    written, conflicts = _write_attributes(store, chunks[0].persona_id, claimed, dry_run=dry_run)
    return ImportResult(
        path=path,
        doc_id=payload.doc_id,
        entities=len(enrichment.entities),
        mentions=len(enrichment.mentions),
        relations=len(enrichment.relations),
        attributes=written,
        loose=tuple(name for name, tier in tiers.items() if tier == "loose"),
        unmatched=tuple(name for name, tier in tiers.items() if tier == "none"),
        dangling=tuple(
            f"{r.source} -> {r.target}"
            for r in payload.relations
            if r.source.strip().lower() not in names or r.target.strip().lower() not in names
        ),
        renamed=tuple(
            f"{e.name} -> {aliases.canonical(e.name)}"
            for e in payload.entities
            if fold_name(aliases.canonical(e.name)) != fold_name(e.name)
        ),
        collisions=collisions,
        attribute_problems=tuple(problems),
        attribute_conflicts=tuple(conflicts),
    )


def landing_id(name: str, kind: str, aliases: AliasTable) -> str:
    """The id of the node an extracted entity ends up on, alias table included.

    The same two steps :func:`graphrag.extract.llm.result_to_enrichment` and
    :func:`graphrag.extract.aliases.apply_aliases` take between them: the id is built from the
    name the passage uses, unless the alias table folds that spelling onto a different one, in
    which case it is built from the canonical name. Attributes are keyed by this rather than by
    the surface form so a claim about an alias lands on the node the mentions landed on.
    """
    surface = name.strip()
    canonical = aliases.canonical(surface)
    folded = fold_name(canonical) != fold_name(surface)
    return Entity.make_id(canonical if folded else surface, kind)


def _claimed_attributes(
    entities: Sequence[ExtractedEntity], aliases: AliasTable, table: AttributeTable
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """What this one file claims about each entity, and the keys the vocabulary refused.

    Two entries of one file that fold onto the same node are folded here too, first value
    winning, so a document naming a thing twice under two spellings does not race with itself.
    A disagreement inside one file is reported by the same path as one across files: it goes
    into the store, the store keeps what it holds, and the refusal comes back as a conflict.
    """
    claimed: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for item in entities:
        if not item.name.strip() or not item.attributes:
            continue
        kept, refused = table.check(item.attributes)
        problems.extend(refused)
        held = claimed.setdefault(landing_id(item.name, item.type, aliases), {})
        for key, value in kept.items():
            held.setdefault(key, value)
    return claimed, problems


def _write_attributes(
    store: GraphStore, persona_id: str, claimed: dict[str, dict[str, str]], *, dry_run: bool
) -> tuple[int, list[str]]:
    """Put this file's entity attributes on the nodes, and report what was already contradicted.

    The stored values are read once, before anything is written, so a dry run reports exactly
    the conflicts a real import would: the store keeps the first value and returns the keys it
    refused, and the read supplies the value it kept them at, for the report. This is
    :func:`graphrag.extract.attribution._write_attributes` for entities, kept beside its own
    importer rather than shared, because the two differ in what they key on and in what a
    refusal is called in the report.
    """
    if not claimed:
        return 0, []
    held = store.entity_attributes(persona_id)
    conflicts: list[str] = []
    written = 0
    for entity_id, incoming in claimed.items():
        if not incoming:
            continue
        stored = held.get(entity_id, {})
        refused = (
            [key for key, value in incoming.items() if stored.get(key, value) != value]
            if dry_run
            else store.set_entity_attributes(persona_id, entity_id, incoming)
        )
        # A refusal names a conflict only when there is a held value to name. The other way a
        # key comes back refused is an entity the graph does not hold, which cannot happen on
        # the path above -- the entities were upserted a few lines earlier -- and is not a
        # disagreement between two readings if it ever does.
        conflicts += [
            f"attribute conflict: {sanitize_inline(entity_id, _LABEL)} {key} "
            f"{sanitize_inline(stored[key], _LABEL)} vs {sanitize_inline(incoming[key], _LABEL)}"
            for key in refused
            if key in incoming and key in stored
        ]
        written += len(incoming) - len(refused)
    return written, conflicts


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
