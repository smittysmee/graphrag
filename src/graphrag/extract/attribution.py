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
                "score": <int> | null}]}

``role``, ``date`` and ``score`` are validated and kept in the file as provenance. Only who spoke
in which passage reaches the graph, because that is all :meth:`GraphStore.attach_speaker` writes.

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

from graphrag.graph.store import GraphStore
from graphrag.textutil import sanitize_inline

__all__ = [
    "AttributedPost",
    "AttributionResult",
    "DocumentAttribution",
    "import_attribution_file",
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
    """

    path: Path
    doc_id: str = ""
    error: str = ""
    posts: int = 0
    attached: int = 0
    speakers: tuple[str, ...] = ()
    loose: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.error


def _fold(text: str) -> str:
    """Normalise for anchor matching: no invisible characters, single spaces, lower case.

    The same flattening :func:`graphrag.textutil.sanitize_inline` performs, at a limit no real
    text can reach, so an anchor copied out of a raw file still matches a passage whose text the
    ingest hardening already stripped of hidden characters.
    """
    return sanitize_inline(text, max(len(text), 1)).lower()


def _find_anchor(anchor: str, folded: list[tuple[str, str]]) -> str | None:
    """The id of the first passage containing ``anchor``, or ``None`` when no passage does."""
    needle = _fold(anchor)
    if not needle:
        return None
    return next((chunk_id for chunk_id, text in folded if needle in text), None)


def import_attribution_file(
    store: GraphStore, path: Path, *, dry_run: bool = False
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
    folded = [(c.id, _fold(c.text)) for c in chunks]
    speakers: list[str] = []
    loose: list[str] = []
    attached = 0
    for post in payload.posts:
        chunk_id = _find_anchor(post.anchor, folded)
        if chunk_id is None:
            loose.append(
                f"{sanitize_inline(post.speaker, _LABEL)}: {sanitize_inline(post.anchor, _LABEL)}"
            )
            continue
        attached += 1
        if post.speaker not in speakers:
            speakers.append(post.speaker)
        if not dry_run:
            store.attach_speaker(payload.doc_id, chunk_id, post.speaker)
    return AttributionResult(
        path=path,
        doc_id=payload.doc_id,
        posts=len(payload.posts),
        attached=attached,
        speakers=tuple(speakers),
        loose=tuple(loose),
    )
