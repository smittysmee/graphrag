"""Reproducibility and citation for every `sna` report (entitlement ticket ATL-ENT-4).

A report a `sna` command prints is a statistic, and a statistic without its recipe cannot be
checked or repeated: which method computed it, with which parameters and which random seed,
against which null, by which version of this tool, over which committed snapshot of the corpus,
and which chapter of the book it is answering. :class:`Provenance` is that recipe, one instance
per report; :func:`make_provenance` builds it from the exact argument values the calling command
already has in scope -- never re-derived from the report after the fact, so a report's own
numbers and its provenance block can never disagree about what produced them.

Every `sna` command calls :func:`make_provenance` once, then :func:`render_provenance` to append
the block to its markdown and :func:`provenance_payload` to set the ``"provenance"`` key of its
JSON payload -- the same :class:`Provenance` both times, so the two representations of one report
always agree. `--cite` (CLI only) additionally appends :func:`render_references`, one line per
chapter in :attr:`Provenance.chapters`, after the book's own citation.

Chapter titles and page ranges below are read from the book's own outline
(``scripts/atlas_chapter.py toc``, v2, Coscia, free at <https://www.networkatlas.eu/>) rather
than retyped from memory, and cover all 56 chapters so a report from any command resolves.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from graphrag import __version__
from graphrag.config import Settings
from graphrag.graph.snapshot import SnapshotError, read_manifest, snapshot_dir

#: (title, first page, last page), in the book's own chapter order (the Introduction is
#: chapter 1). Titles are the outline's own wording, which is not always the paraphrase
#: `docs/ATLAS_PLAN.md` uses for its coverage matrix row.
CHAPTER_TITLES: dict[int, tuple[str, int, int]] = {
    1: ("Introduction", 7, 21),
    2: ("Probability Theory", 22, 38),
    3: ("Statistics", 39, 54),
    4: ("Machine Learning", 55, 69),
    5: ("Linear Algebra", 70, 90),
    6: ("Basic Graphs", 91, 102),
    7: ("Extended Graphs", 103, 118),
    8: ("Matrices", 119, 133),
    9: ("Degree", 134, 153),
    10: ("Paths & Walks", 154, 164),
    11: ("Random Walks", 165, 178),
    12: ("Density", 179, 190),
    13: ("Shortest Paths", 191, 206),
    14: ("Node Ranking", 207, 224),
    15: ("Node Roles", 225, 236),
    16: ("Random Graphs", 237, 246),
    17: ("Understanding Network Properties", 247, 257),
    18: ("Generating Realistic Data", 258, 273),
    19: ("Evaluating Statistical Significance", 274, 285),
    20: ("Epidemics", 286, 298),
    21: ("Complex Contagion", 299, 313),
    22: ("Catastrophic Failures", 314, 326),
    23: ("For Simple Graphs", 327, 343),
    24: ("For Multilayer Graphs", 344, 355),
    25: ("Designing an Experiment", 356, 367),
    26: ("Bipartite Projections", 368, 380),
    27: ("Network Backboning", 381, 394),
    28: ("Uncertainty & Measurement Error", 395, 411),
    29: ("Network Sampling", 412, 429),
    30: ("Homophily", 430, 440),
    31: ("Quantitative Assortativity", 441, 449),
    32: ("Core-Periphery", 450, 461),
    33: ("Hierarchies", 462, 473),
    34: ("High-Order Dynamics", 474, 489),
    35: ("Graph Partitions", 490, 509),
    36: ("Community Evaluation", 510, 530),
    37: ("Hierarchical Community Discovery", 531, 542),
    38: ("Overlapping Coverage", 543, 560),
    39: ("Bipartite Community Discovery", 561, 571),
    40: ("Multilayer Community Discovery", 572, 587),
    41: ("Frequent Subgraph Mining", 588, 605),
    42: ("Shallow Graph Learning", 606, 621),
    43: ("Random Walk Embeddings", 622, 635),
    44: ("Message-Passing & Graph Convolution", 636, 653),
    45: ("Deep Graph Learning Models", 654, 667),
    46: ("Graph Summarization", 668, 678),
    47: ("Node Vector Distance", 679, 696),
    48: ("Topological Distances", 697, 711),
    49: ("Node Visual Attributes", 712, 726),
    50: ("Edge Visual Attributes", 727, 734),
    51: ("Network Layouts", 735, 756),
    52: ("Network Science Applications", 757, 769),
    53: ("Data & Tools", 770, 788),
    54: ("Glossary", 789, 798),
    55: ("Most Common Abbreviations", 799, 802),
    56: ("Bibliography", 803, 916),
}

#: The book's own citation, printed once by `--cite` ahead of the per-chapter lines.
BOOK_CITATION = (
    "Coscia, M. *The Atlas for the Aspiring Network Scientist*, v2 (916 pp.). "
    "Free at https://www.networkatlas.eu/."
)

#: Every `sna` command's primary Atlas coverage (chapter numbers), used when a command does not
#: narrow the list itself from the flags a particular run set. Keyed by the CLI command name
#: (hyphens, matching `@sna_app.command("...")`); `guide` and `cache *` are not reports and are
#: not in this table.
COMMAND_CHAPTERS: dict[str, tuple[int, ...]] = {
    "sample": (29,),
    "export": (6, 7, 53),
    "analyze": (9, 12, 14, 19, 26, 30, 31, 35, 36, 42),
    "stances": (24,),
    "compare": (7, 35, 48),
    "layers": (7,),
    "multilayer-communities": (7, 40),
    "projections": (26,),
    "walks": (2, 11),
    "distance": (47,),
    "backbone": (13, 27),
    "summarize": (46,),
    "degree": (9,),
    "roles": (15,),
    "ego": (30,),
    "community": (35, 36, 37, 38, 39),
    "overlap": (38,),
    "hierarchy": (33,),
    "highorder": (34,),
    "predict-eval": (25,),
    "predict": (23, 24),
    "complete": (44, 45),
    "robustness": (22,),
    "motifs": (41,),
    "spread": (20, 21),
    "draw": (49, 50, 51),
}


def chapter_citation(chapter: int) -> str:
    """One `--cite` line for ``chapter``, or a note that this package does not know it.

    The unknown case is not expected to fire from any `sna` command -- every entry in
    :data:`COMMAND_CHAPTERS` is a key of :data:`CHAPTER_TITLES` -- but a bad chapter number is
    reported rather than raised, since a citation line is not worth crashing a report over.
    """
    entry = CHAPTER_TITLES.get(chapter)
    if entry is None:
        return f"Chapter {chapter} (page range not on file)."
    title, first, last = entry
    return f"Chapter {chapter} -- {title} (pp. {first}-{last})."


@dataclass(frozen=True)
class Provenance:
    """One report's recipe: what produced these numbers, and how to reproduce them.

    :attr:`parameters` and :attr:`seed` are the exact values the calling command already held --
    see :func:`make_provenance`, which is the only place this class is normally constructed --
    never re-read off the report afterwards, so this can never drift from what the report's own
    numbers were actually computed from.
    """

    method: str
    """The `sna` command name, e.g. ``"degree"``."""
    parameters: dict[str, Any]
    """Every argument the command was called with, apart from where it wrote its output."""
    seed: int | None
    tool_version: str
    null_model: str | None
    """The name of the null this report's headline structural claim was tested against, when the
    report has one obvious primary null; ``None`` when the command has no single such null, or
    makes several claims each against its own (those stay in the report's own prose -- this is a
    summary field, not a replacement)."""
    null_model_samples: int | None
    chapters: tuple[int, ...]
    """Atlas chapters this report answers; drives `--cite`."""
    generated_at: str
    snapshot_commit: str | None
    """The corpus submodule's git commit recorded by the persona's committed snapshot
    (:func:`graphrag.graph.snapshot.export_snapshot`'s ``source_commit``), or ``None`` for a
    persona with no snapshot on disk -- an in-memory store in a test, or one still being
    ingested."""


NO_SOURCE_COMMIT = "unknown (no source commit recorded for this persona)"
"""What the provenance line says when :func:`snapshot_commit_for` finds nothing. A persona can
have a committed snapshot and still record no ``source_commit`` -- one whose corpus is not a git
submodule -- so the line names the missing commit, not a missing snapshot."""


def snapshot_commit_for(settings: Settings, persona_id: str) -> str | None:
    """The committed snapshot's ``source_commit`` for ``persona_id``, or ``None`` if unknown.

    Mirrors :func:`graphrag.sna.cache.snapshot_identity`'s own read of the manifest, but returns
    the commit alone rather than a cache key -- a persona never exported, or exported before a
    commit was recorded, both report ``None`` rather than raising.
    """
    try:
        manifest = read_manifest(snapshot_dir(settings.snapshots_dir, persona_id))
    except SnapshotError:
        return None
    return manifest.source_commit or None


def make_provenance(
    *,
    method: str,
    parameters: Mapping[str, Any],
    chapters: Sequence[int],
    seed: int | None = None,
    null_model: str | None = None,
    null_model_samples: int | None = None,
    snapshot_commit: str | None = None,
    generated_at: str | None = None,
) -> Provenance:
    """Build one report's :class:`Provenance` from the values the calling command already has.

    ``parameters`` and ``seed`` should be the command's own local variables, not anything read
    back off the report the command went on to build -- see the module docstring. ``generated_at``
    defaults to now (UTC, second precision, ISO 8601); a caller never needs to pass it except a
    test that wants a fixed clock.
    """
    return Provenance(
        method=method,
        parameters=dict(parameters),
        seed=seed,
        tool_version=__version__,
        null_model=null_model,
        null_model_samples=null_model_samples,
        chapters=tuple(chapters),
        generated_at=generated_at or datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        snapshot_commit=snapshot_commit,
    )


def _jsonable(value: Any) -> Any:
    """``value`` with every `Path` turned to `str`, recursively through dict/list/tuple.

    Parameters can carry a filter value that started life as a CLI `Path` option (none currently
    do, but a future one might); this keeps :func:`provenance_payload` valid JSON either way
    without every command having to remember to convert it.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def render_provenance(provenance: Provenance) -> list[str]:
    """The `## Provenance` block: one line per field, in the order a reader would ask for them."""
    chapters = ", ".join(str(c) for c in provenance.chapters) or "none"
    null = (
        f"{provenance.null_model} ({provenance.null_model_samples:,} samples)"
        if provenance.null_model and provenance.null_model_samples
        else (provenance.null_model or "none for this report's headline claim")
    )
    lines = [
        "## Provenance",
        "",
        f"- method: `{provenance.method}` (Atlas ch. {chapters})",
        # `sort_keys`: valid JSON rather than a `dict` repr, and stable regardless of the order
        # a command happened to build its own `parameters` dict in -- the CLI and an MCP tool
        # rarely declare the same options in the same order, and an unsorted repr would make
        # otherwise-identical markdown disagree on that alone.
        f"- parameters: `{json.dumps(_jsonable(provenance.parameters), sort_keys=True)}`",
        f"- seed: {provenance.seed if provenance.seed is not None else 'not fixed'}",
        f"- null model: {null}",
        f"- tool version: graphrag {provenance.tool_version}",
        f"- generated: {provenance.generated_at}",
        f"- snapshot commit: {provenance.snapshot_commit or NO_SOURCE_COMMIT}",
        "",
    ]
    return lines


def provenance_payload(provenance: Provenance) -> dict[str, Any]:
    """The `"provenance"` payload key: the same fields :func:`render_provenance` printed."""
    return {
        "method": provenance.method,
        "parameters": _jsonable(provenance.parameters),
        "seed": provenance.seed,
        "tool_version": provenance.tool_version,
        "null_model": provenance.null_model,
        "null_model_samples": provenance.null_model_samples,
        "chapters": list(provenance.chapters),
        "generated_at": provenance.generated_at,
        "snapshot_commit": provenance.snapshot_commit,
    }


def render_references(chapters: Iterable[int]) -> list[str]:
    """The `## References` block `--cite` appends: the book once, then one line per chapter.

    Chapters are de-duplicated and printed in ascending order regardless of the order a report
    touched them in, since a bibliography is read by chapter number, not by narrative order.
    """
    lines = ["## References", "", BOOK_CITATION, ""]
    for chapter in sorted(set(chapters)):
        lines.append(f"- {chapter_citation(chapter)}")
    lines.append("")
    return lines


__all__ = [
    "BOOK_CITATION",
    "CHAPTER_TITLES",
    "COMMAND_CHAPTERS",
    "Provenance",
    "chapter_citation",
    "make_provenance",
    "provenance_payload",
    "render_provenance",
    "render_references",
    "snapshot_commit_for",
]
