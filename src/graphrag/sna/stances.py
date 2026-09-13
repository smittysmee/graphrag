"""What the corpus says about a thing, not merely how often it names it.

The entity network answers "discussed alongside". Once an annotation pass has marked mentions,
a second question becomes answerable: of the passages that name this thing, how many praise it,
how many complain about it, who wrote each kind, which function of the thing they were talking
about, and what the passages actually say.

This module is that report. It reads the annotation layer only -- ``mention_stances`` and
``chunk_facets`` -- so every number in it is a count of annotations, never of mentions. That
distinction is the one a reader gets wrong first: an entity with two complaints and no praise
has two complaints and an unknown amount of everything else, because nobody annotated the rest.

The signed co-mention table at the end is the pairwise view: entities that turn up in the same
praised passage against entities that turn up in the same complained-about passage. Two things
praised together and two things complained about together are different findings, and a single
"co-mention" number hides which one you have.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from graphrag.extract.aliases import fold_name
from graphrag.graph.store import GraphStore
from graphrag.models import MentionStance, Stance
from graphrag.sna.export import STANCES
from graphrag.textutil import sanitize_inline

__all__ = [
    "EntityStances",
    "SignedPair",
    "StanceReport",
    "build_stance_report",
    "render_stances",
]

#: How much of a passage is quoted as evidence. Long enough to be a quotation, short enough
#: that a report of fifty entities stays readable.
QUOTE = 280
#: How many names a "who said it" cell lists before it gives up and counts the rest.
TOP_NAMES = 6
#: How many pairs the signed co-mention table shows.
TOP_PAIRS = 25


@dataclass(frozen=True)
class EntityStances:
    """One entity's annotated mentions, grouped by what they say."""

    entity_id: str
    name: str
    counts: dict[Stance, int] = field(default_factory=dict)
    speakers: dict[Stance, list[tuple[str, int]]] = field(default_factory=dict)
    facets: dict[Stance, list[tuple[str, int]]] = field(default_factory=dict)
    quotes: dict[Stance, list[str]] = field(default_factory=dict)
    documents: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class SignedPair:
    """Two entities and how many passages of each stance name both of them."""

    left: str
    right: str
    shared: dict[Stance, int] = field(default_factory=dict)

    def count(self, stance: Stance) -> int:
        return self.shared.get(stance, 0)


@dataclass(frozen=True)
class StanceReport:
    """Everything ``graphrag sna stances`` produced, before it is rendered."""

    persona_id: str
    source_id: str = ""
    facets: tuple[str, ...] = ()
    entities: tuple[EntityStances, ...] = ()
    pairs: tuple[SignedPair, ...] = ()
    annotations: int = 0
    quotes_per_stance: int = 3
    missing: tuple[str, ...] = ()

    @property
    def stances_present(self) -> tuple[Stance, ...]:
        """The stances that actually occur, in the fixed order, so columns stay stable."""
        seen = {s for entity in self.entities for s in entity.counts}
        return tuple(s for s in STANCES if s in seen)


def build_stance_report(
    store: GraphStore,
    persona_id: str,
    *,
    source_id: str | None = None,
    entities: Sequence[str] = (),
    facets: Sequence[str] = (),
    quotes_per_stance: int = 3,
) -> StanceReport:
    """Read the annotation layer for one persona and group it by entity and by stance.

    ``entities`` names the entities to keep, matched the way aliases are matched: case
    insensitive with whitespace folded, against the stored name. A name that matches nothing is
    reported rather than silently dropped, because a typo and an entity with no annotations look
    identical in the output otherwise.
    """
    rows = store.mention_stances(persona_id, source_id)
    facet_of = {r.chunk_id: list(r.facets) for r in store.chunk_facets(persona_id, source_id)}
    if facets:
        wanted_facets = {f.strip() for f in facets if f.strip()}
        rows = [r for r in rows if wanted_facets & set(facet_of.get(r.chunk_id, ()))]
    asked = {fold_name(name) for name in entities if name.strip()}
    if asked:
        rows = [r for r in rows if fold_name(r.name) in asked]
    missing = tuple(
        name
        for name in entities
        if name.strip() and fold_name(name) not in {fold_name(r.name) for r in rows}
    )

    text_of = {c.id: c.text for c in store.get_chunks(sorted({r.chunk_id for r in rows}))}
    grouped: dict[str, list[tuple[Stance, str, list[str], list[str]]]] = defaultdict(list)
    names: dict[str, str] = {}
    documents: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        names[row.entity_id] = row.name
        documents[row.entity_id].add(row.doc_id)
        grouped[row.entity_id].append(
            (row.stance, row.chunk_id, list(row.speakers), facet_of.get(row.chunk_id, []))
        )

    built = [
        _one_entity(entity_id, names[entity_id], items, text_of, documents, quotes_per_stance)
        for entity_id, items in grouped.items()
    ]
    built.sort(key=lambda e: (-e.total, e.name))
    return StanceReport(
        persona_id=persona_id,
        source_id=source_id or "",
        facets=tuple(facets),
        entities=tuple(built),
        pairs=_signed_pairs(rows_by_chunk(rows)),
        annotations=len(rows),
        quotes_per_stance=quotes_per_stance,
        missing=missing,
    )


def _one_entity(
    entity_id: str,
    name: str,
    items: Sequence[tuple[Stance, str, list[str], list[str]]],
    text_of: dict[str, str],
    documents: dict[str, set[str]],
    quotes_per_stance: int,
) -> EntityStances:
    """Fold one entity's annotated mentions into counts, names, facets and quotations."""
    counts: Counter[Stance] = Counter()
    speakers: dict[Stance, Counter[str]] = defaultdict(Counter)
    facets: dict[Stance, Counter[str]] = defaultdict(Counter)
    quotes: dict[Stance, list[str]] = {}
    for stance, chunk_id, who, marks in items:
        counts[stance] += 1
        speakers[stance].update(who)
        facets[stance].update(marks)
        passage = sanitize_inline(text_of.get(chunk_id, ""), QUOTE)
        if passage and quotes_per_stance:
            held = quotes.setdefault(stance, [])
            if len(held) < quotes_per_stance and passage not in held:
                held.append(passage)
    return EntityStances(
        entity_id=entity_id,
        name=name,
        counts=dict(counts),
        speakers={s: c.most_common() for s, c in speakers.items()},
        facets={s: c.most_common() for s, c in facets.items()},
        quotes=quotes,
        documents=len(documents.get(entity_id, ())),
    )


def rows_by_chunk(rows: Sequence[MentionStance]) -> dict[tuple[str, Stance], set[str]]:
    """Passage and stance to the entity names annotated that way in it.

    Keyed on the passage rather than the document: two entities complained about in the same
    post share a complaint, while two complained about in the same thread a week apart do not.
    """
    grouped: dict[tuple[str, Stance], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(row.chunk_id, row.stance)].add(row.name)
    return grouped


def _signed_pairs(grouped: dict[tuple[str, Stance], set[str]]) -> tuple[SignedPair, ...]:
    """Entity pairs that share a passage, counted separately for each stance."""
    tallies: dict[tuple[str, str], Counter[Stance]] = defaultdict(Counter)
    for (_, stance), names in grouped.items():
        for left, right in combinations(sorted(names), 2):
            tallies[(left, right)][stance] += 1
    pairs = [
        SignedPair(left=left, right=right, shared=dict(counts))
        for (left, right), counts in tallies.items()
    ]
    pairs.sort(key=lambda p: (-sum(p.shared.values()), p.left, p.right))
    return tuple(pairs)


# ----------------------------------------------------------------------------- rendering


def _names(pairs: Sequence[tuple[str, int]]) -> str:
    """A "who" or "which facet" cell: the heaviest few, then a count of the rest."""
    if not pairs:
        return "-"
    shown = pairs[:TOP_NAMES]
    rest = len(pairs) - len(shown)
    text = ", ".join(f"{name} ({count})" for name, count in shown)
    return text + (f", +{rest} more" if rest else "")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_stances(report: StanceReport) -> str:
    """The stance report as a markdown note, ready to drop into a corpus."""
    columns = report.stances_present
    filters = [f"persona={report.persona_id}"]
    if report.source_id:
        filters.append(f"source={report.source_id}")
    if report.facets:
        filters.append(f"facets={', '.join(report.facets)}")
    lines = [
        f"# Stances: {report.persona_id}",
        "",
        f"**Sampling frame.** {report.annotations:,} annotated mentions over "
        f"{len(report.entities):,} entities. Filters: {', '.join(filters)}. Every count here is "
        "a count of annotations, not of mentions: a passage nobody annotated is absent from this "
        "report, so an entity with no complaints has no annotated complaints rather than no "
        "complaints.",
        "",
    ]
    if not report.entities:
        lines += ["No annotated mentions matched these filters.", ""]
    if report.missing:
        lines += [
            "> Named but not found in the annotated mentions: "
            + ", ".join(sanitize_inline(name, 60) for name in report.missing)
            + ". Check the spelling against the entity's stored name, or the alias table.",
            "",
        ]

    if report.entities and columns:
        lines += ["## By entity", ""]
        lines += _table(
            ["entity", *columns, "total", "documents"],
            [
                [
                    entity.name,
                    *[str(entity.counts.get(stance, 0)) for stance in columns],
                    str(entity.total),
                    str(entity.documents),
                ]
                for entity in report.entities
            ],
        )

    for entity in report.entities:
        lines += [f"### {entity.name}", ""]
        for stance in columns:
            count = entity.counts.get(stance, 0)
            if not count:
                continue
            lines += [
                f"**{stance}** — {count} passage(s).",
                "",
                f"- who: {_names(entity.speakers.get(stance, []))}",
                f"- facets: {_names(entity.facets.get(stance, []))}",
                "",
            ]
            lines += [f"> {quote}" for quote in entity.quotes.get(stance, [])]
            lines += [""]

    if report.pairs and columns:
        lines += [
            "## Signed co-mention",
            "",
            "Entities that share a passage, counted per stance. Praised together and complained "
            "about together are different relationships; a pair heavy in one column and empty "
            "in the other is the interesting case.",
            "",
        ]
        lines += _table(
            ["entity", "entity", *columns],
            [
                [pair.left, pair.right, *[str(pair.count(stance)) for stance in columns]]
                for pair in report.pairs[:TOP_PAIRS]
            ],
        )

    lines += [
        "## Caveats",
        "",
        "- Report n. These are annotations an agent wrote while reading, not a survey: the "
        "denominator is what was read, not what exists.",
        "- Absence of a stance is absence of an annotation. Do not read an empty complaint "
        "column as satisfaction.",
        "- A stance sits on a mention, so it says what one passage said about one thing. "
        "Aggregating stances into a score for the entity invents a precision the passages do "
        "not have.",
        "- The quotations are the evidence. If a count cannot be traced to a passage that reads "
        "the way the count claims, the annotation is wrong, not the passage.",
    ]
    return "\n".join(lines).rstrip() + "\n"
