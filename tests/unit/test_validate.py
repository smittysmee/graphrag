"""The captured-corpus contract: front-matter, body length and provenance consistency."""

from __future__ import annotations

import json
from pathlib import Path

from graphrag.ingest.validate import (
    MAX_BODY_WORDS,
    MERGED_MANIFEST,
    PROVENANCE_DIR,
    merge_provenance,
    parse_captured,
    validate_corpus,
)

GOOD_META = {
    "title": "How practitioners pick a scheduling tool",
    "fetched_at": "2026-01-15T10:00:00Z",
    "fetched_by": "researcher-1",
    "retrieval": "web_fetch",
    "content_fidelity": "verbatim",
    "document_type": "forum_thread",
    "category": "tooling",
    "source_url": "https://example.invalid/threads/1",
}


def body(words: int) -> str:
    return " ".join(f"word{n}" for n in range(words))


def write_capture(root: Path, name: str, *, meta: dict[str, object], words: int = 120) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    front = "\n".join(f"{k}: {v}" for k, v in meta.items())
    path.write_text(f"---\n{front}\n---\n\n{body(words)}\n", encoding="utf-8")
    return path


def codes(root: Path) -> list[str]:
    return [p.code for p in validate_corpus(root).problems]


def test_parse_captured_splits_front_matter_from_body() -> None:
    meta, text = parse_captured("---\ntitle: A\n---\n\nhello world\n")
    assert meta == {"title": "A"} and text.strip() == "hello world"


def test_a_well_formed_capture_has_no_problems(tmp_path: Path) -> None:
    write_capture(tmp_path, "threads/one.md", meta=GOOD_META)
    report = validate_corpus(tmp_path)
    assert report.ok, report.problem_lines()
    assert report.documents == 1
    assert report.documents_per_source == {"threads": 1}
    assert report.document_types == {"forum_thread": 1}


def test_a_missing_required_key_is_reported(tmp_path: Path) -> None:
    write_capture(
        tmp_path, "threads/one.md", meta={k: v for k, v in GOOD_META.items() if k != "fetched_by"}
    )
    report = validate_corpus(tmp_path)
    assert codes(tmp_path) == ["missing-keys"]
    assert "fetched_by" in report.problem_lines()[0]


def test_an_oversized_body_is_reported(tmp_path: Path) -> None:
    write_capture(tmp_path, "threads/long.md", meta=GOOD_META, words=MAX_BODY_WORDS + 1)
    report = validate_corpus(tmp_path)
    assert codes(tmp_path) == ["too-long"]
    assert f"TOO LONG ({MAX_BODY_WORDS + 1} words): threads/long.md" in report.problem_lines()


def test_a_near_empty_body_is_reported(tmp_path: Path) -> None:
    write_capture(tmp_path, "threads/stub.md", meta=GOOD_META, words=9)
    assert codes(tmp_path) == ["near-empty"]


def test_unparsable_front_matter_is_one_problem_not_seven(tmp_path: Path) -> None:
    """A file with no fence is missing every key, but the reviewer only needs the first line."""
    (tmp_path / "loose.md").write_text(f"# no front-matter\n\n{body(80)}\n", encoding="utf-8")
    assert codes(tmp_path) == ["frontmatter"]


def test_broken_yaml_in_front_matter_is_reported_as_front_matter(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text(
        f"---\ntitle: [unclosed\n---\n\n{body(80)}\n", encoding="utf-8"
    )
    assert codes(tmp_path) == ["frontmatter"]


def test_a_fetched_document_needs_a_source_url(tmp_path: Path) -> None:
    write_capture(
        tmp_path, "one.md", meta={k: v for k, v in GOOD_META.items() if k != "source_url"}
    )
    assert codes(tmp_path) == ["no-source-url"]


def test_a_research_note_needs_no_source_url(tmp_path: Path) -> None:
    meta = {k: v for k, v in GOOD_META.items() if k != "source_url"}
    meta["document_type"] = "research_note"
    write_capture(tmp_path, "notes/wave-1.md", meta=meta)
    assert validate_corpus(tmp_path).ok


def test_provenance_lines_must_parse_and_point_at_a_file_that_exists(tmp_path: Path) -> None:
    write_capture(tmp_path, "threads/one.md", meta=GOOD_META)
    manifest = tmp_path / PROVENANCE_DIR / "researcher-1.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"file": "threads/one.md", "status": "ok"}),
                json.dumps({"file": "threads/gone.md", "status": "ok"}),
                json.dumps({"file": "threads/blocked.md", "status": "failed"}),
                "{not json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    report = validate_corpus(tmp_path)
    assert sorted(p.code for p in report.problems) == ["bad-json", "missing-file"]
    assert len(report.provenance) == 3  # the unparsable line is reported, not kept


def test_merge_writes_one_manifest_and_never_reads_its_own_output(tmp_path: Path) -> None:
    write_capture(tmp_path, "threads/one.md", meta=GOOD_META)
    folder = tmp_path / PROVENANCE_DIR
    folder.mkdir()
    for who in ("researcher-1", "researcher-2"):
        (folder / f"{who}.jsonl").write_text(
            json.dumps({"file": "threads/one.md", "fetched_by": who}) + "\n", encoding="utf-8"
        )
    out = merge_provenance(tmp_path, validate_corpus(tmp_path).provenance)
    assert out == folder / MERGED_MANIFEST
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    # A second pass must still see two records, not four.
    assert len(validate_corpus(tmp_path).provenance) == 2
