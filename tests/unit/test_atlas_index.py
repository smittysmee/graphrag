"""`docs/ATLAS_INDEX.md` is generated; this holds the committed copy to that generation verbatim.

The same discipline `test_sna_guide.py` applies to `render_guide()`'s block in `docs/SNA.md`, but
the generator here lives in `scripts/atlas_index.py`, which is not part of the `graphrag` package
(ATL-52 owns it, not `sna/`), so it is loaded from its file path rather than imported by name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "atlas_index.py"


def _load_atlas_index() -> ModuleType:
    """Import ``scripts/atlas_index.py`` from its path, the way `python scripts/atlas_index.py`
    would run it, rather than requiring a package `scripts` never declares."""
    spec = importlib.util.spec_from_file_location("atlas_index", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("atlas_index", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def atlas_index() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.skip("scripts/ is not mounted in this container")
    module = _load_atlas_index()
    if not module.atlas_chapter.CACHE.is_file():
        pytest.skip(
            f"{module.atlas_chapter.CACHE} is not cached locally; a unit test must not download "
            "it over the network. Run `python scripts/atlas_chapter.py toc` once (host-side) to "
            "cache it."
        )
    return module


def test_the_committed_index_is_the_generated_one(atlas_index: ModuleType) -> None:
    """Guards against a hand edit drifting away from the scan that produces this file.

    If this fails, do not edit `docs/ATLAS_INDEX.md` by hand: run
    `python scripts/atlas_index.py` and commit its output.
    """
    committed = (REPO / "docs" / "ATLAS_INDEX.md").read_text(encoding="utf-8")
    assert committed == atlas_index.render_index()


def test_every_chapter_and_section_of_the_book_gets_a_row(atlas_index: ModuleType) -> None:
    rows = atlas_index.build_rows()
    pdf = atlas_index.atlas_chapter.ensure_pdf()
    chapters = atlas_index.atlas_chapter.read_outline(pdf)
    expected = len(chapters) + sum(len(chapter.sections) for chapter in chapters)
    assert len(rows) == expected
    # Chapter 54 (Glossary) has no subsections in the outline, but is still one row.
    assert any(row["chapter"] == "54" and row["section"] == "" for row in rows)


def test_a_module_that_cites_a_section_is_credited_on_that_row(atlas_index: ModuleType) -> None:
    """Known answer: `sna/backbone.py` cites §27.6 (noise-corrected) and names `sna backbone`."""
    rows = {row["section"] or row["chapter"]: row for row in atlas_index.build_rows()}
    row = rows["27.6"]
    assert "backbone.py" in row["modules"]
    assert "sna backbone" in row["commands"]


def test_a_rule_that_cites_a_section_is_credited_by_name_not_by_file(
    atlas_index: ModuleType,
) -> None:
    """Known answer: the guide's noise-corrected rule cites §27.6 by name, not as `guide.py`."""
    rows = {row["section"] or row["chapter"]: row for row in atlas_index.build_rows()}
    row = rows["27.6"]
    assert "guide.py" not in row["modules"]
    assert row["rules"] != "—"


def test_a_declined_section_is_documented_and_not_relabelled_by_a_passing_mention(
    atlas_index: ModuleType,
) -> None:
    """Known answer from `docs/SNA_FOUNDATIONS.md`: §13.5 is declined and links there; §13.1,
    which that same "documented, not built" paragraph mentions only in passing to contrast it,
    is built by `paths.py` and must not be relabelled "documented" by that passing mention."""
    rows = {row["section"] or row["chapter"]: row for row in atlas_index.build_rows()}
    declined = rows["13.5"]
    assert declined["modules"] == ""
    assert "SNA_FOUNDATIONS.md" in declined["documented"]
    built = rows["13.1"]
    assert "paths.py" in built["modules"]
    assert built["documented"] == ""


def test_a_citation_past_the_outlines_own_sections_falls_back_to_the_chapter(
    atlas_index: ModuleType,
) -> None:
    """Known answer: the code cites §22.6 and §22.6.1, but this printing's chapter 22 outline
    stops at §22.4, so both credit the chapter-22 row rather than inventing a section 22.6."""
    index: dict[str, tuple] = {}
    pdf = atlas_index.atlas_chapter.ensure_pdf()
    for chapter in atlas_index.atlas_chapter.read_outline(pdf):
        index[str(chapter.number)] = (chapter, None)
        for section in chapter.sections:
            index[section.number] = (chapter, section)
    assert "22.6" not in index
    assert atlas_index.resolve("22.6", index) == "22"
    assert atlas_index.resolve("22.6.1", index) == "22"


def test_slugify_matches_the_anchor_this_index_actually_links(atlas_index: ModuleType) -> None:
    """Known answer: the real heading text in ``docs/SNA_FOUNDATIONS.md`` (with its em-dash)."""
    assert atlas_index.slugify("§13.5 — classic combinatorial problems") == (
        "135--classic-combinatorial-problems"
    )
    assert atlas_index.slugify("Machine learning") == "machine-learning"


def test_check_mode_exits_zero_against_the_committed_file(atlas_index: ModuleType) -> None:
    assert atlas_index.main(["--check"]) == 0
