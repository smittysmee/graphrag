"""``docs/SNA_FOUNDATIONS.md`` is where every "Doc" row of the Atlas coverage matrix lands.

The plan promises that nothing in the book is passed over: a chapter is either a merged ticket or
a section of the foundations document with the reason it is not code. These tests hold the two
files to that promise, so a chapter cannot be quietly dropped from the programme by deleting a
paragraph.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FOUNDATIONS = REPO / "docs" / "SNA_FOUNDATIONS.md"
PLAN = REPO / "docs" / "ATLAS_PLAN.md"

#: The seven sections the ticket (ATL-01) commissions, in order.
HEADINGS = (
    "## How the Atlas approaches complexity, and how this package does",
    "## Probability",
    "## Statistics",
    "## Machine learning",
    "## Linear algebra",
    "## Documented, not built",
    "## Reading order for a network scientist new to this repository",
)

#: ``pp. 22-29``, ``p. 7``: a page citation into the book's v2 PDF.
PAGE_CITATION = re.compile(r"\bpp?\.\s*\d+")


def _foundations() -> str:
    return FOUNDATIONS.read_text(encoding="utf-8")


def _documented_chapters() -> list[int]:
    """Chapter numbers whose disposition in the plan's coverage matrix is labelled "Doc".

    The matrix is one markdown row per chapter: number, title, what exists today, disposition,
    owner ticket. A disposition *labelled* "Doc" — at the start of the cell or after a semicolon,
    in either case ("Doc:", "Doc + build:", "Doc (§13.5):", "doc: notation") — promises a section
    of this document, whether or not the same chapter also has code behind it: chapter 2 and
    chapter 5 have both.

    The label is what is matched, not the word. Two Build rows mention a document that belongs to
    their own ticket's output rather than to this file — chapter 42's "the 'what an embedding is'
    doc" and chapter 51's "§51.5 case studies as doc patterns" — and neither has to resolve here,
    because ATL-42 and ATL-49 write them. If one of those tickets later hands its prose to this
    file, the plan's disposition is what changes first, and this test then demands the section.
    """
    label = re.compile(r"(?:^|[;,]\s)doc\b(?=\s*[:+(])", re.IGNORECASE)
    chapters: list[int] = []
    for line in PLAN.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        if label.search(cells[3]):
            chapters.append(int(cells[0]))
    return chapters


def _sections() -> dict[str, str]:
    """The body of each top-level section, keyed by its heading line."""
    bodies: dict[str, str] = {}
    current = ""
    for line in _foundations().splitlines():
        if line.startswith("## "):
            current = line.strip()
            bodies[current] = ""
        elif current:
            bodies[current] += line + "\n"
    return bodies


@pytest.mark.parametrize("heading", HEADINGS)
def test_every_commissioned_section_exists(heading: str) -> None:
    assert heading in _foundations().splitlines()


def test_the_sections_are_in_the_order_the_ticket_lists_them() -> None:
    text = _foundations()
    positions = [text.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)


def test_the_plan_still_has_documented_chapters_to_check() -> None:
    """A guard on the parser: if the matrix format changes, the test below must not pass empty."""
    chapters = _documented_chapters()
    assert len(chapters) >= 5
    assert 1 in chapters  # the introduction, this file's own opening section
    assert 5 in chapters  # "doc: notation", lowercase, still a label
    assert 13 in chapters  # classic combinatorial problems
    assert 45 in chapters  # transformers and deep generative models
    assert 42 not in chapters  # "the 'what an embedding is' doc" is ATL-42's own output
    assert 51 not in chapters  # "§51.5 case studies as doc patterns" is ATL-49's


@pytest.mark.parametrize("chapter", _documented_chapters())
def test_every_documented_chapter_is_named_by_number(chapter: int) -> None:
    """Every "Doc" row resolves to a mention in the foundations file, by chapter or section."""
    mention = re.compile(rf"(?:§|ch\.\s*|chapters?\s+){chapter}\b", re.IGNORECASE)
    assert mention.search(_foundations()), f"chapter {chapter} is documented nowhere"


@pytest.mark.parametrize("heading", HEADINGS)
def test_every_section_cites_a_page_of_the_book(heading: str) -> None:
    """A disposition without a page number is an opinion; the book is the source of truth."""
    body = _sections()[heading]
    assert PAGE_CITATION.search(body), f"{heading} cites no page of the book"


def test_the_sections_that_replace_code_name_what_would_change_that() -> None:
    """A "documented, not built" reason is only honest if it says what would overturn it."""
    body = _sections()["## Documented, not built"]
    assert body.count("What would earn a ticket") >= 3


def test_the_docs_table_and_the_network_analysis_guide_point_at_the_foundations() -> None:
    for doc in (REPO / "README.md", REPO / "docs" / "SNA.md"):
        text = doc.read_text(encoding="utf-8")
        assert "SNA_FOUNDATIONS.md" in text, f"{doc.name} does not link the foundations"
