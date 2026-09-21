"""`docs/SNA_GLOSSARY.md` is hand-written, but every term the book's own glossary and
abbreviation chapters list must appear in it somewhere -- this reads the book, not a copy of its
term list kept here, so a reprint that adds a term fails this test rather than going unnoticed.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "atlas_chapter.py"
GLOSSARY = REPO / "docs" / "SNA_GLOSSARY.md"

#: A term whose entry in the book starts with a symbol rather than a capital letter (an em-dash
#: macron over ``k``, the Greek ``Omega``) or a lower-case word (``n, m-Clique``), which the
#: extraction regexes below cannot pull out of the running text on their own.
GLOSSARY_TERM_RE = re.compile(r"^([A-Z][A-Za-z0-9’' \-]{1,40}?): ", re.MULTILINE)  # noqa: RUF001
ABBREVIATION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9,¯’' \-]{0,15}?): ", re.MULTILINE)  # noqa: RUF001
MANUAL_GLOSSARY_TERMS = ("n, m-Clique",)
MANUAL_ABBREVIATIONS = ("k̄", "Ω")  # k-bar (average degree), Omega (effective resistance)


def _load_atlas_chapter() -> ModuleType:
    spec = importlib.util.spec_from_file_location("atlas_chapter", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("atlas_chapter", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def atlas_chapter() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.skip("scripts/ is not mounted in this container")
    module = _load_atlas_chapter()
    if not module.CACHE.is_file():
        pytest.skip(
            f"{module.CACHE} is not cached locally; a unit test must not download it over the "
            "network. Run `python scripts/atlas_chapter.py toc` once (host-side) to cache it."
        )
    return module


def _chapter_text(atlas_chapter: ModuleType, number: int) -> str:
    pdf = atlas_chapter.ensure_pdf()
    chapters = atlas_chapter.read_outline(pdf)
    chapter = next(c for c in chapters if c.number == number)
    return atlas_chapter.text_of(pdf, chapter.first, chapter.last)


def test_every_glossary_term_appears_in_the_repos_glossary(atlas_chapter: ModuleType) -> None:
    text = _chapter_text(atlas_chapter, 54)
    terms = sorted(set(GLOSSARY_TERM_RE.findall(text)) | set(MANUAL_GLOSSARY_TERMS))
    assert len(terms) >= 100, "the extraction regex found suspiciously few terms; check it first"
    glossary = GLOSSARY.read_text(encoding="utf-8")
    missing = [term for term in terms if term not in glossary]
    assert not missing, f"{len(missing)} glossary term(s) missing from {GLOSSARY}: {missing}"


def test_every_abbreviation_appears_in_the_repos_glossary(atlas_chapter: ModuleType) -> None:
    text = _chapter_text(atlas_chapter, 55)
    symbols = sorted(set(ABBREVIATION_RE.findall(text)) | set(MANUAL_ABBREVIATIONS))
    assert len(symbols) >= 30, "the extraction regex found suspiciously few symbols; check it first"
    glossary = GLOSSARY.read_text(encoding="utf-8")
    missing = [symbol for symbol in symbols if symbol not in glossary]
    assert not missing, f"{len(missing)} abbreviation(s) missing from {GLOSSARY}: {missing}"


def test_the_glossary_names_the_three_declined_foundations_sections() -> None:
    """Every "not built" row has to point somewhere, not just say so."""
    glossary = GLOSSARY.read_text(encoding="utf-8")
    assert "SNA_FOUNDATIONS.md" in glossary
    assert "not built" in glossary.lower()
