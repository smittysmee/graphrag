"""One entity, one node: fold the spellings a corpus uses into the canonical name.

Extraction deliberately keeps the surface form a passage uses, so a mention anchors to real
text. The cost is that one thing arrives as several nodes -- an abbreviation, a former name, a
typo -- and every count over it is low by however many spellings nobody thought to look for.

A persona answers that with ``personas/<persona>/aliases.yaml``::

    aliases:
      Canonical Name: [Alias One, alias two, "Alias, with comma"]

Matching is case-insensitive and whitespace-folded on both sides, and a name may appear under
one canonical only -- two canonicals claiming the same alias is a contradiction the loader
raises on rather than resolving by file order.

The table is applied in two places and one command. At extraction import the names are mapped
before they are written, so new files never create the alias node in the first place; the same
happens at annotation import, so an annotation may name the entity the way its passage does.
``graphrag aliases apply`` fixes what is already in the graph, through
:meth:`GraphStore.merge_entities`, and :mod:`graphrag.sync` runs it after an import pass.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from graphrag.graph.store import GraphStore
from graphrag.models import Enrichment, Entity, Mention, Relation

__all__ = [
    "ALIAS_FILE",
    "EMPTY_ALIASES",
    "AliasError",
    "AliasReport",
    "AliasTable",
    "MergeResult",
    "alias_lines",
    "apply_alias_table",
    "apply_aliases",
    "fold_name",
    "load_aliases",
    "load_persona_aliases",
    "preview_alias_table",
]

#: Where a persona keeps its table, beside ``persona.yaml``.
ALIAS_FILE = "aliases.yaml"


class AliasError(ValueError):
    """The alias file is unreadable, malformed, or claims one alias for two canonicals."""


def fold_name(name: str) -> str:
    """The form two spellings are compared in: lower case, with runs of whitespace collapsed."""
    return " ".join(name.split()).casefold()


@dataclass(frozen=True)
class AliasTable:
    """Canonical names and the spellings that should fold into each one.

    ``groups`` keeps the file's order so a report reads the way the file does; ``canonical_for``
    is the lookup, keyed on the folded form of both the aliases and the canonical itself.
    """

    groups: tuple[tuple[str, tuple[str, ...]], ...] = ()
    canonical_for: Mapping[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.groups)

    def canonical(self, name: str) -> str:
        """The canonical spelling of ``name``, or ``name`` unchanged when the table is silent."""
        return self.canonical_for.get(fold_name(name), name)

    def aliases_for(self, canonical: str) -> tuple[str, ...]:
        """The alias spellings listed under ``canonical``, empty when it is not a canonical."""
        folded = fold_name(canonical)
        return next((a for c, a in self.groups if fold_name(c) == folded), ())


EMPTY_ALIASES = AliasTable()


def load_aliases(path: Path) -> AliasTable:
    """Read one ``aliases.yaml``. Raises :class:`AliasError` on anything it cannot trust."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"cannot read alias file {path}: {exc}"
        raise AliasError(msg) from exc
    if data is None:
        return EMPTY_ALIASES
    if not isinstance(data, Mapping):
        msg = f"{path}: expected a mapping with an 'aliases' key, got {type(data).__name__}"
        raise AliasError(msg)
    raw = data.get("aliases") or {}
    if not isinstance(raw, Mapping):
        msg = f"{path}: 'aliases' must be a mapping of canonical name to a list of spellings"
        raise AliasError(msg)

    groups: list[tuple[str, tuple[str, ...]]] = []
    owner: dict[str, str] = {}
    for canonical, spellings in raw.items():
        name = str(canonical).strip()
        if not name:
            msg = f"{path}: a canonical name is blank"
            raise AliasError(msg)
        if isinstance(spellings, str) or not isinstance(spellings, Iterable):
            msg = f"{path}: aliases for {name!r} must be a list, got {type(spellings).__name__}"
            raise AliasError(msg)
        seen: list[str] = []
        for spelling in spellings:
            alias = str(spelling).strip()
            if not alias:
                msg = f"{path}: {name!r} lists a blank alias"
                raise AliasError(msg)
            folded = fold_name(alias)
            if folded == fold_name(name):
                continue  # a canonical listed among its own aliases is harmless, not news
            held = owner.get(folded)
            if held is not None and fold_name(held) != fold_name(name):
                msg = f"{path}: alias {alias!r} is claimed by both {held!r} and {name!r}"
                raise AliasError(msg)
            owner[folded] = name
            if alias not in seen:
                seen.append(alias)
        groups.append((name, tuple(seen)))

    for name, _ in groups:
        folded = fold_name(name)
        held = owner.get(folded)
        if held is not None and fold_name(held) != folded:
            msg = f"{path}: {name!r} is both a canonical name and an alias of {held!r}"
            raise AliasError(msg)
        owner[folded] = name
    return AliasTable(groups=tuple(groups), canonical_for=owner)


def load_persona_aliases(personas_dir: Path, persona_id: str) -> AliasTable:
    """The persona's table, or an empty one when it keeps no alias file. Absence is not news."""
    path = personas_dir / persona_id / ALIAS_FILE
    return load_aliases(path) if path.is_file() else EMPTY_ALIASES


def apply_aliases(enrichment: Enrichment, table: AliasTable) -> Enrichment:
    """Rewrite an extraction onto canonical names before it is written to the graph.

    Applied after the passages have been matched, never before: matching looks for the name the
    passage actually uses, and canonicalising first would send every renamed entity to the
    fallback passage. The surface form is not lost -- it is recorded on the canonical entity's
    ``aliases``, which is where a reader goes to find out why a node is named what it is.
    """
    if not table:
        return enrichment
    remap: dict[str, str] = {}
    entities: dict[str, Entity] = {}
    for entity in enrichment.entities:
        canonical = table.canonical(entity.name)
        if fold_name(canonical) == fold_name(entity.name):
            mapped = entity
        else:
            mapped = entity.model_copy(
                update={
                    "id": Entity.make_id(canonical, entity.type),
                    "name": canonical,
                    "aliases": sorted({*entity.aliases, entity.name.strip()}),
                }
            )
        remap[entity.id] = mapped.id
        held = entities.get(mapped.id)
        entities[mapped.id] = _combine(held, mapped) if held is not None else mapped

    mentions: dict[tuple[str, str], Mention] = {}
    for mention in enrichment.mentions:
        entity_id = remap.get(mention.entity_id, mention.entity_id)
        key = (mention.chunk_id, entity_id)
        seen = mentions.get(key)
        stance = (seen.stance if seen is not None else None) or mention.stance
        mentions[key] = Mention(chunk_id=mention.chunk_id, entity_id=entity_id, stance=stance)

    relations: dict[tuple[str, str, str, str], Relation] = {}
    for relation in enrichment.relations:
        source = remap.get(relation.source_id, relation.source_id)
        target = remap.get(relation.target_id, relation.target_id)
        if source == target:
            continue  # two spellings of one thing cannot be related to each other
        moved = relation.model_copy(update={"source_id": source, "target_id": target})
        relations.setdefault((source, target, moved.type, moved.chunk_id), moved)

    return Enrichment(
        entities=list(entities.values()),
        mentions=list(mentions.values()),
        relations=list(relations.values()),
    )


def _combine(held: Entity, incoming: Entity) -> Entity:
    """Two spellings that folded onto one id, as a single entity."""
    return held.model_copy(
        update={
            "description": held.description or incoming.description,
            "aliases": sorted({*held.aliases, *incoming.aliases}),
        }
    )


@dataclass(frozen=True)
class MergeResult:
    """What folding one canonical's spellings together did, or would do."""

    canonical: str
    aliases: tuple[str, ...] = ()
    mentions_moved: int = 0


@dataclass(frozen=True)
class AliasReport:
    """One persona's alias pass. ``applied`` is the part worth printing."""

    persona_id: str
    merges: tuple[MergeResult, ...] = ()
    dry_run: bool = False

    @property
    def applied(self) -> tuple[MergeResult, ...]:
        return tuple(m for m in self.merges if m.mentions_moved)

    @property
    def mentions_moved(self) -> int:
        return sum(m.mentions_moved for m in self.merges)


def apply_alias_table(store: GraphStore, persona_id: str, table: AliasTable) -> AliasReport:
    """Fold every group in ``table`` together in the graph, one canonical at a time."""
    merges = [
        MergeResult(
            canonical=canonical,
            aliases=spellings,
            mentions_moved=store.merge_entities(persona_id, canonical, spellings),
        )
        for canonical, spellings in table.groups
    ]
    return AliasReport(persona_id=persona_id, merges=tuple(merges))


def preview_alias_table(store: GraphStore, persona_id: str, table: AliasTable) -> AliasReport:
    """What :func:`apply_alias_table` would move, counted without writing anything.

    Counts mentions the way the merge does -- per persona, per alias spelling present in the
    graph -- from the same bipartite read the entity network is built from.
    """
    counts: Counter[str] = Counter(
        fold_name(row.name) for row in store.entity_chunk_pairs(persona_id)
    )
    merges = [
        MergeResult(
            canonical=canonical,
            aliases=spellings,
            mentions_moved=sum(
                counts[fold_name(alias)]
                for alias in spellings
                if fold_name(alias) != fold_name(canonical)
            ),
        )
        for canonical, spellings in table.groups
    ]
    return AliasReport(persona_id=persona_id, merges=tuple(merges), dry_run=True)


def alias_lines(report: AliasReport) -> list[str]:
    """One line per canonical that moved something, plus a closing line. Pure, so tests read it."""
    verb = "would move" if report.dry_run else "moved"
    lines = [
        f"{m.canonical}: {verb} {m.mentions_moved} mentions from {len(m.aliases)} spellings"
        for m in report.applied
    ]
    if not report.applied:
        lines.append(f"{report.persona_id}: no alias spellings found in the graph")
    else:
        lines.append(
            f"{report.persona_id}: {verb} {report.mentions_moved} mentions onto "
            f"{len(report.applied)} canonical names"
        )
    return lines
