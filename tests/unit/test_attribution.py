"""``graphrag attribution-import``: anchoring posts to passages and attaching their speakers.

Everything runs against the in-memory store and the sample thread from ``tests/conftest.py``,
which the documents loader reads as prose and so leaves with no speakers at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphrag.extract.attribution import import_attribution_file, report_lines
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk
from tests.conftest import THREAD_POSTS

# The opening words of the first and last posts. The last post is in a later passage than the
# first, so an import that ignored the anchor and took passage zero would fail these tests.
FIRST = THREAD_POSTS[0][2][:60]
LAST = THREAD_POSTS[-1][2][:60]


def write_attribution(path: Path, doc_id: str, posts: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"doc_id": doc_id, "posts": posts}), encoding="utf-8")
    return path


def post(speaker: str, anchor: str, **extra: object) -> dict[str, object]:
    return {"speaker": speaker, "anchor": anchor, **extra}


def chunk_with(store: InMemoryGraphStore, doc_id: str, text: str) -> Chunk:
    """The passage of ``doc_id`` containing ``text``, found the way a reader would."""
    return next(c for c in store.document_chunks(doc_id, 0, 100) if text in c.text)


def test_each_post_lands_on_the_passage_its_anchor_is_in(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [
            post("quill-maker", FIRST, role="op", date="2025-02-03", score=12),
            post("ledger-ann", LAST, role="reply", date="2025-02-06", score=None),
        ],
    )

    result = import_attribution_file(memory_store, file)

    assert result.ok
    assert (result.posts, result.attached, result.loose) == (2, 2, ())
    assert result.speakers == ("quill-maker", "ledger-ann")
    opening = chunk_with(memory_store, thread_document, FIRST)
    closing = chunk_with(memory_store, thread_document, LAST)
    assert opening.id != closing.id
    assert opening.speakers == ["quill-maker"]
    assert closing.speakers == ["ledger-ann"]
    document = memory_store.documents[thread_document]
    assert document.speakers == ["quill-maker", "ledger-ann"]
    assert {s.speaker for s in memory_store.list_speakers("test-docs")} == {
        "quill-maker",
        "ledger-ann",
    }


def test_anchors_match_with_different_spacing_and_case(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """An agent re-typing an anchor should not have to reproduce the source's whitespace."""
    sloppy = "   " + FIRST.upper().replace(" ", "\n  ") + "\n"
    file = write_attribution(
        tmp_path / "thread.json", thread_document, [post("quill-maker", sloppy)]
    )

    result = import_attribution_file(memory_store, file)

    assert result.attached == 1 and result.loose == ()


def test_an_anchor_in_no_passage_is_loose_and_costs_only_its_own_post(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """A paraphrased opening line must never be attributed to a guessed passage."""
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [
            post("ledger-ann", "something this thread never says anywhere at all"),
            post("quill-maker", FIRST),
        ],
    )

    result = import_attribution_file(memory_store, file)

    assert result.ok  # loose anchors are a report, not a failure
    assert (result.posts, result.attached) == (2, 1)
    assert result.speakers == ("quill-maker",)
    assert len(result.loose) == 1 and result.loose[0].startswith("ledger-ann: something")
    assert memory_store.documents[thread_document].speakers == ["quill-maker"]
    # One block on one stream: a loose anchor printed apart from its own file's line can be read
    # as belonging to the file above it, which sends a reviewer to the wrong JSON.
    assert report_lines(result) == [
        f"{thread_document}: 1/2 posts attached, 1 speakers; 1 loose anchors",
        "  loose: ledger-ann: something this thread never says anywhere at all",
    ]


def test_re_importing_the_same_file_changes_nothing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """`make sync` re-imports these files, so a second pass must not duplicate a speaker."""
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [post("quill-maker", FIRST), post("quill-maker", THREAD_POSTS[2][2][:60])],
    )

    first = import_attribution_file(memory_store, file)
    before = memory_store.documents[thread_document]
    second = import_attribution_file(memory_store, file)

    assert first.attached == second.attached == 2
    assert first.speakers == second.speakers == ("quill-maker",)  # counted once, attached twice
    assert memory_store.documents[thread_document] == before
    assert memory_store.stats().speakers == 1
    assert chunk_with(memory_store, thread_document, FIRST).speakers == ["quill-maker"]


def test_a_dry_run_reports_without_writing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    file = write_attribution(
        tmp_path / "thread.json",
        thread_document,
        [post("quill-maker", FIRST), post("nobody", "not in the thread")],
    )

    result = import_attribution_file(memory_store, file, dry_run=True)

    assert (result.attached, len(result.loose)) == (1, 1)
    assert memory_store.documents[thread_document].speakers == []
    assert memory_store.stats().speakers == 0


def test_a_document_the_graph_does_not_hold_is_an_error_not_a_raise(
    memory_store: InMemoryGraphStore, tmp_path: Path
) -> None:
    file = write_attribution(tmp_path / "x.json", "test-docs:threads:gone", [post("a", "b")])

    result = import_attribution_file(memory_store, file)

    assert not result.ok
    assert "unknown document test-docs:threads:gone" in result.error


def test_bad_json_and_bad_posts_are_reported_per_file(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """One malformed file must not stop a run over hundreds of them."""
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    assert import_attribution_file(memory_store, broken).error.startswith("invalid:")

    blank = write_attribution(tmp_path / "blank.json", thread_document, [post("  ", FIRST)])
    assert import_attribution_file(memory_store, blank).error.startswith("invalid:")

    undated = write_attribution(
        tmp_path / "undated.json", thread_document, [post("a", FIRST, date="last tuesday")]
    )
    assert import_attribution_file(memory_store, undated).error.startswith("invalid:")

    assert import_attribution_file(memory_store, tmp_path / "absent.json").error.startswith(
        "invalid:"
    )
    assert memory_store.stats().speakers == 0


def test_a_file_with_no_posts_is_valid_and_does_nothing(
    memory_store: InMemoryGraphStore, thread_document: str, tmp_path: Path
) -> None:
    """An agent that found no attributable posts should say so, not write a file that fails."""
    file = write_attribution(tmp_path / "empty.json", thread_document, [])

    result = import_attribution_file(memory_store, file)

    assert result.ok and (result.posts, result.attached, result.speakers) == (0, 0, ())
    assert memory_store.documents[thread_document].speakers == []
