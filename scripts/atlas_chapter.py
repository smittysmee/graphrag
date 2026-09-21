#!/usr/bin/env python3
"""Read a chapter of *The Atlas for the Aspiring Network Scientist* (Coscia, v2) as plain text.

Usage:
    python scripts/atlas_chapter.py toc                 # every part, chapter, section, with pages
    python scripts/atlas_chapter.py 27                  # chapter 27 (Network Backboning), in full
    python scripts/atlas_chapter.py 27 --section 27.6   # one section of it
    python scripts/atlas_chapter.py "Core-Periphery"    # by title, case-insensitive
    python scripts/atlas_chapter.py 27 --pages          # page range only

The book is the source of truth for everything under ``docs/ATLAS_PLAN.md``. An agent working a
ticket reads its chapter through this script before writing a line, and cites section numbers in
the docstrings it writes, so a reviewer can hold the code to the text rather than to memory.

The PDF (27 MB, free, https://www.networkatlas.eu/files/sna_book.pdf) is downloaded once into
``data/cache/atlas/``, which is git-ignored. Page ranges are read from the PDF's own outline, so
a new printing with shifted pages still resolves; chapter numbers are assigned in outline order,
which matches the book's own numbering (the Introduction is chapter 1).
"""

from __future__ import annotations

import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "atlas" / "sna_book.pdf"
URL = "https://www.networkatlas.eu/files/sna_book.pdf"

#: Outline entries that are chapter furniture rather than sections.
FURNITURE = {"Summary", "Exercises"}
#: Outline titles that open a part rather than a chapter: a roman numeral, a space, a title.
PART_RE = re.compile(r"^(?:[IVX]+) ")


@dataclass
class Section:
    number: str
    title: str
    first: int  # 1-based page
    last: int = 0


@dataclass
class Chapter:
    number: int
    title: str
    part: str
    first: int
    last: int = 0
    sections: list[Section] = field(default_factory=list)


def ensure_pdf() -> Path:
    if CACHE.is_file():
        return CACHE
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {CACHE}", file=sys.stderr)
    with urllib.request.urlopen(URL, timeout=300) as response, CACHE.open("wb") as out:
        out.write(response.read())
    return CACHE


def read_outline(path: Path) -> list[Chapter]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chapters: list[Chapter] = []
    part = ""
    counter = 0

    def page_of(item: object) -> int:
        return int(reader.get_destination_page_number(item)) + 1  # type: ignore[arg-type]

    def walk(items: list[object], parent: str) -> None:
        """``parent`` is what the enclosing entry was: ``root``, ``part`` or ``chapter``.

        pypdf flattens the outline into a list where an entry's children follow it as a nested
        list, so the kind of an entry is decided by its parent, not by its depth: a chapter is a
        child of a part (or a top-level entry that is not a part, like the Introduction), and a
        section is a child of a chapter. The Introduction's own sections sit at the same depth
        as Part I's chapters, which is why depth alone gets the numbering wrong.
        """
        nonlocal part, counter
        index = 0
        while index < len(items):
            item = items[index]
            children = items[index + 1] if index + 1 < len(items) else None
            index += 2 if isinstance(children, list) else 1
            if isinstance(item, list):
                continue
            title = str(item.title).strip()  # type: ignore[attr-defined]
            if parent == "root" and PART_RE.match(title):
                part = title
                kind = "part"
            elif parent in {"root", "part"}:
                counter += 1
                chapters.append(Chapter(counter, title, part, page_of(item)))
                kind = "chapter"
            elif parent == "chapter" and title not in FURNITURE and chapters:
                chapter = chapters[-1]
                number = f"{chapter.number}.{len(chapter.sections) + 1}"
                chapter.sections.append(Section(number, title, page_of(item)))
                kind = "section"
            else:
                kind = "section"
            if isinstance(children, list):
                walk(children, kind)

    walk(list(reader.outline), "root")
    total = len(reader.pages)
    for index, chapter in enumerate(chapters):
        chapter.last = chapters[index + 1].first - 1 if index + 1 < len(chapters) else total
        for s_index, section in enumerate(chapter.sections):
            if s_index + 1 < len(chapter.sections):
                section.last = chapter.sections[s_index + 1].first - 1
            else:
                section.last = chapter.last
    return chapters


def find(chapters: list[Chapter], key: str) -> Chapter:
    if key.isdigit():
        number = int(key)
        for chapter in chapters:
            if chapter.number == number:
                return chapter
        raise SystemExit(f"no chapter {number}; run `toc` to list them")
    wanted = key.strip().lower()
    for chapter in chapters:
        if chapter.title.lower() == wanted:
            return chapter
    for chapter in chapters:
        if wanted in chapter.title.lower():
            return chapter
    raise SystemExit(f"no chapter titled {key!r}; run `toc` to list them")


def text_of(path: Path, first: int, last: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number in range(first, last + 1):
        raw = reader.pages[number - 1].extract_text() or ""
        pages.append(f"\n----- page {number} -----\n{raw}")
    return "".join(pages)


def print_toc(chapters: list[Chapter]) -> None:
    part = None
    for chapter in chapters:
        if chapter.part != part:
            part = chapter.part
            print(f"\n{part or 'Front matter'}")
        print(f"  {chapter.number:>2}  {chapter.title}  (pp. {chapter.first}-{chapter.last})")
        for section in chapter.sections:
            print(
                f"        {section.number:<6} {section.title}  (pp. {section.first}-{section.last})"
            )


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    pdf = ensure_pdf()
    chapters = read_outline(pdf)
    if argv[0] == "toc":
        print_toc(chapters)
        return 0
    chapter = find(chapters, argv[0])
    first, last, label = chapter.first, chapter.last, f"{chapter.number} {chapter.title}"
    if "--section" in argv:
        wanted = argv[argv.index("--section") + 1]
        section = next((s for s in chapter.sections if s.number == wanted), None)
        if section is None:
            raise SystemExit(f"chapter {chapter.number} has no section {wanted}")
        first, last, label = section.first, section.last, f"{section.number} {section.title}"
    if "--pages" in argv:
        print(f"{label}: pp. {first}-{last}")
        return 0
    print(f"===== {label} (pp. {first}-{last}) =====")
    print(text_of(pdf, first, last))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
