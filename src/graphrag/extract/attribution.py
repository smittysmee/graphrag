"""Import post-level speaker attribution an agent wrote as JSON.

Documents that arrive through the ``documents`` loader carry no speaker turns. A captured
discussion thread is one body of text, so the graph holds it with no ``Speaker`` nodes at all and
every speaker-shaped question about it comes back empty. Transcripts do not have the problem:
their loader parses ``Name (00:00:00):`` turns and the pipeline writes the speakers from those.

The fix mirrors enrichment. An agent reads one document and writes one JSON file naming the posts
it contains, each with a verbatim anchor; this module finds the passage each anchor lands in and
attaches the speaker to it::

    {"doc_id": "<persona>:<source>:<slug>",
     "posts": [{"speaker": "handle or name",
                "anchor": "verbatim opening words of the post, 6 to 20 words",
                "role": "op" | "reply",
                "date": "YYYY-MM-DD" | null,
                "score": <int> | null,
                "attributes": {"region": "north"}}]}

``role``, ``date`` and ``score`` are validated and passed through to the ``SPOKE`` edge, so a
speaker network can be cut to a time window without going back to the files.

``attributes`` describe the speaker rather than the post, and go on the ``Speaker`` node after
being checked against the persona's vocabulary (:mod:`graphrag.extract.attributes`); a key or
value outside it is reported and dropped, per key. Two posts that disagree about one speaker do
not fight over the node: the first value written stands and the second is reported as
``attribute conflict: <speaker> <key> <kept> vs <incoming>``, because which of two readings is
right is a question for whoever wrote them, not for whichever file was imported last.

Anchors are matched case-insensitively with whitespace folded, against the document's passages in
order, and the first passage containing one wins. A post whose anchor matches nothing is reported
as ``loose`` and skipped: an agent that paraphrased one opening line should cost that line, not
the file.

Two callers read these files: ``graphrag attribution-import`` and :mod:`graphrag.sync`, which
re-imports a source's files after a re-ingest. Re-ingesting deletes a document's chunks, and
``SPOKE`` edges hang off chunks exactly as entity mentions do.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from graphrag.extract.attributes import EMPTY_ATTRIBUTES, AttributeTable
from graphrag.graph.store import GraphStore
from graphrag.textutil import sanitize_inline

__all__ = [
    "AttributedPost",
    "AttributionResult",
    "DocumentAttribution",
    "find_anchor",
    "fold_passage",
    "import_attribution_file",
    "report_lines",
]

#: Enough to cover any single document; the store pages, so this is one read either way.
_ALL_CHUNKS = 100_000
#: How much of a speaker or anchor is echoed when a post is reported as loose.
_LABEL = 80

#: Blank-only speakers and anchors are a mistake worth failing the file for, not silent skips.
Required = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AttributedPost(BaseModel):
    """One post inside a document: who wrote it, and words to find it by."""

    speaker: Required
    anchor: Required
    role: Literal["op", "reply"] = "reply"
    date: dt.date | None = None
    score: int | None = None
    #: What this post says the speaker *is*, not what the post is. Written onto the speaker.
    attributes: dict[str, str] = Field(default_factory=dict)


class DocumentAttribution(BaseModel):
    """One document's posts as produced by an agent (``graphrag attribution-import``)."""

    doc_id: str
    posts: list[AttributedPost] = Field(default_factory=list)


@dataclass(frozen=True)
class AttributionResult:
    """One attribution file's outcome, including what a reviewer should look at.

    ``attached`` counts posts placed in a passage, ``speakers`` names the distinct people that
    produced, and ``loose`` holds the posts whose anchor occurs in no passage of the document.
    A loose post is skipped rather than guessed at, so nothing is attributed to the wrong voice.

    ``attributes`` counts the speaker attributes written, ``attribute_problems`` the keys the
    persona's vocabulary refused, and ``conflicts`` the ones a stored value already contradicts.
    The last two are different findings: a problem is a file to fix, a conflict is two readings
    of one speaker that disagree, and only the file's author can say which is right.
    """

    path: Path
    doc_id: str = ""
    error: str = ""
    posts: int = 0
    attached: int = 0
    attributes: int = 0
    speakers: tuple[str, ...] = ()
    loose: tuple[str, ...] = ()
    attribute_problems: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error


def fold_passage(text: str) -> str:
    """Normalise for anchor matching: no invisible characters, single spaces, lower case.

    The same flattening :func:`graphrag.textutil.sanitize_inline` performs, at a limit no real
    text can reach, so an anchor copied out of a raw file still matches a passage whose text the
    ingest hardening already stripped of hidden characters.
    """
    return sanitize_inline(text, max(len(text), 1)).lower()


def find_anchor(anchor: str, folded: list[tuple[str, str]]) -> str | None:
    """The id of the first passage containing ``anchor``, or ``None`` when no passage does.

    Shared with the annotation layer, which places its passages exactly the same way.
    """
    needle = fold_passage(anchor)
    if not needle:
        return None
    return next((chunk_id for chunk_id, text in folded if needle in text), None)


def report_lines(result: AttributionResult) -> list[str]:
    """One file's report: its own line, then the anchors that matched nothing, under it.

    One block on one stream, for the reason :func:`graphrag.extract.annotations.report_lines`
    gives: a count on stdout and its loose anchors on stderr are two streams with no order
    between them, so a loose line can surface under the wrong document's line. Plain text, no
    markup, because each loose line quotes an anchor an agent wrote.
    """
    head = (
        f"{result.doc_id}: {result.attached}/{result.posts} posts attached, "
        f"{len(result.speakers)} speakers"
        + (f", {result.attributes} speaker attributes" if result.attributes else "")
        + (f"; {len(result.loose)} loose anchors" if result.loose else "")
        + (
            f"; {len(result.attribute_problems)} attribute problems"
            if result.attribute_problems
            else ""
        )
        + (f"; {len(result.conflicts)} attribute conflicts" if result.conflicts else "")
    )
    return [
        head,
        *(f"  loose: {post}" for post in result.loose),
        *(f"  attribute invalid: {problem}" for problem in result.attribute_problems),
        *(f"  {conflict}" for conflict in result.conflicts),
    ]


def import_attribution_file(
    store: GraphStore,
    path: Path,
    *,
    dry_run: bool = False,
    attributes: AttributeTable = EMPTY_ATTRIBUTES,
) -> AttributionResult:
    """Validate one attribution file and attach its speakers to the passages they wrote.

    Bad input never raises: an unreadable file, invalid JSON or a ``doc_id`` the graph does not
    know comes back as a result whose ``error`` is set, so a caller importing hundreds of files
    can report them all instead of stopping at the first. Re-importing the same file changes
    nothing, because the store merges both the speaker and its edges.
    """
    try:
        payload = DocumentAttribution.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return AttributionResult(path=path, error=f"invalid: {exc}")
    chunks = store.document_chunks(payload.doc_id, 0, _ALL_CHUNKS)
    if not chunks:
        return AttributionResult(
            path=path, doc_id=payload.doc_id, error=f"unknown document {payload.doc_id}"
        )
    folded = [(c.id, fold_passage(c.text)) for c in chunks]
    speakers: list[str] = []
    loose: list[str] = []
    problems: list[str] = []
    claimed: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    attached = 0
    for post in payload.posts:
        kept, refused = attributes.check(post.attributes)
        problems.extend(refused)
        chunk_id = find_anchor(post.anchor, folded)
        if chunk_id is None:
            loose.append(
                f"{sanitize_inline(post.speaker, _LABEL)}: {sanitize_inline(post.anchor, _LABEL)}"
            )
            continue
        attached += 1
        if post.speaker not in speakers:
            speakers.append(post.speaker)
        # Attributes ride on a post that placed: an anchor nobody can find is not evidence of
        # anything, including of who the speaker is.
        _claim(claimed.setdefault(post.speaker, {}), post.speaker, kept, conflicts)
        if not dry_run:
            store.attach_speaker(
                payload.doc_id,
                chunk_id,
                post.speaker,
                posted_at=post.date.isoformat() if post.date else None,
                role=post.role,
                score=post.score,
            )
    written, stored_conflicts = _write_attributes(
        store, chunks[0].persona_id, claimed, dry_run=dry_run
    )
    return AttributionResult(
        path=path,
        doc_id=payload.doc_id,
        posts=len(payload.posts),
        attached=attached,
        attributes=written,
        speakers=tuple(speakers),
        loose=tuple(loose),
        attribute_problems=tuple(problems),
        conflicts=tuple(conflicts + stored_conflicts),
    )


def _conflict(speaker: str, key: str, kept: str, incoming: str) -> str:
    """The one line a disagreement produces, wherever the kept value came from."""
    return (
        f"attribute conflict: {sanitize_inline(speaker, _LABEL)} {key} "
        f"{sanitize_inline(kept, _LABEL)} vs {sanitize_inline(incoming, _LABEL)}"
    )


def _claim(
    held: dict[str, str], speaker: str, incoming: dict[str, str], conflicts: list[str]
) -> None:
    """Fold one post's attributes into what this file already claims about its speaker.

    First value written wins, inside a file exactly as across files: a thread where one post
    says one thing and a later post says another is a disagreement to report, not a race for
    whichever line the reader reached last.
    """
    for key, value in incoming.items():
        if key in held and held[key] != value:
            conflicts.append(_conflict(speaker, key, held[key], value))
        else:
            held.setdefault(key, value)


def _write_attributes(
    store: GraphStore, persona_id: str, claimed: dict[str, dict[str, str]], *, dry_run: bool
) -> tuple[int, list[str]]:
    """Put this file's speaker attributes on the nodes, and report what was already contradicted.

    The stored values are read once, before anything is written, so a dry run reports exactly
    the conflicts a real import would: the store keeps the first value and returns the keys it
    refused, and the read supplies the value it kept them at, for the report.
    """
    if not claimed:
        return 0, []
    held = store.speaker_attributes(persona_id)
    conflicts: list[str] = []
    written = 0
    for speaker, incoming in claimed.items():
        if not incoming:
            continue
        stored = held.get(speaker, {})
        refused = (
            [key for key, value in incoming.items() if stored.get(key, value) != value]
            if dry_run
            else store.set_speaker_attributes(persona_id, speaker, incoming)
        )
        conflicts += [
            _conflict(speaker, key, stored.get(key, ""), incoming[key])
            for key in refused
            if key in incoming
        ]
        written += len(incoming) - len(refused)
    return written, conflicts
