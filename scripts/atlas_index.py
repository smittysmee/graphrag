#!/usr/bin/env python3
"""Generate ``docs/ATLAS_INDEX.md``: a chapter-by-chapter table from the book's own table of
contents to what this package does for it.

Usage:
    python scripts/atlas_index.py            # (re)write docs/ATLAS_INDEX.md
    python scripts/atlas_index.py --check     # exit 1 if the committed file is stale, write nothing
    python scripts/atlas_index.py --print     # print the generated text instead of writing it

One row per chapter and per section of *The Atlas for the Aspiring Network Scientist* (Coscia,
v2), read from the PDF's own outline through :mod:`atlas_chapter` -- the same source of truth
every Atlas ticket reads before writing a line, so a new printing with shifted pages still
resolves. Each row is filled in by scanning, not by hand:

* every ``src/graphrag/sna/*.py`` module (its docstrings *and* its code -- a stray ``§`` outside
  a docstring would be a bug worth surfacing, not one worth hiding) for a ``§n`` or ``§n.m``
  citation, crediting the module to every section it names;
* the same modules for a backtick-quoted ``sna <word>`` command, credited to whatever it cites
  in the same file (a command is rarely scoped to one section, so this is coarser than the
  module credit and says so in the legend);
* :mod:`graphrag.sna.guide`'s ``METHOD_RULES``, ``NETWORK_RULES`` and ``READING_RULES`` --
  imported, not text-scanned, so a rule's prose can move without this script drifting -- crediting
  each named rule to every section its own text cites;
* ``docs/SNA_FOUNDATIONS.md``'s headings (of any level, most recent one wins) for the same
  citation, crediting a section with no module or rule to whichever heading covers it as
  "documented, not built", with a link.

A row with none of the four is not a bug: chapters 1-5 are the book's own background, several
sections are pure narrative (a chapter's opening page before its first ``§``), and a handful of
this book's own within-chapter citations (e.g. ``§22.6.1``) run past what its outline lists for
that chapter -- those are credited to the chapter row itself rather than invented as a section
that does not exist in the outline, and the legend at the top of the generated file says so.

``tests/unit/test_atlas_index.py`` regenerates this file and holds the committed copy to it,
verbatim -- the same discipline ``sna/guide.py``'s ``render_guide()`` block gets from
``docs/SNA.md`` and the SNA skill. If that test fails, do not hand-edit ``docs/ATLAS_INDEX.md``:
run this script and commit its output.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNA_DIR = ROOT / "src" / "graphrag" / "sna"
OUTPUT = ROOT / "docs" / "ATLAS_INDEX.md"
FOUNDATIONS = ROOT / "docs" / "SNA_FOUNDATIONS.md"

# scripts/ holds no __init__.py, so atlas_chapter.py is a sibling module found on sys.path, and
# graphrag.sna.guide is read from source rather than the installed package (there may not be one).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import atlas_chapter  # noqa: E402

from graphrag.sna.guide import METHOD_RULES, NETWORK_RULES, READING_RULES  # noqa: E402

SECTION_RE = re.compile(r"§(\d+(?:\.\d+)?)")
#: A negative lookbehind keeps this off ``graphrag.sna import matrices`` -- a real import
#: statement, not a CLI command -- where a plain word boundary would still match.
COMMAND_RE = re.compile(r"(?<![\w.])sna [a-z][a-z-]*")
#: Modules that hold the rules and the CLI wiring, not analyses of their own; scanning them as
#: ordinary modules would double-count every citation already credited through the rule objects
#: or would credit "cli.py" with every command it merely dispatches.
NOT_A_MODULE = {"__init__.py", "guide.py"}


def slugify(heading: str) -> str:
    """A GitHub-flavoured markdown heading anchor: lowercase, spaces to hyphens, punctuation gone.

    GitHub does not collapse repeated hyphens, so this does not either -- matching it exactly
    matters here, because a wrong anchor is a link a reader clicks and lands nowhere useful.
    """
    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\- ]+", "", slug)
    return slug.replace(" ", "-")


def module_citations() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``section -> {module filenames}`` and ``section -> {sna commands}`` cited in each module.

    A command found anywhere in a module is credited to every section that module cites: the
    book's chapter is usually bigger than any one flag, and the alternative -- trying to pair a
    specific command mention with the nearest ``§`` -- would invent a precision the docstrings do
    not carry (a module like ``backbone.py`` names ``sna backbone`` once and six sections
    throughout).
    """
    modules: dict[str, set[str]] = defaultdict(set)
    commands: dict[str, set[str]] = defaultdict(set)
    for path in sorted(SNA_DIR.glob("*.py")):
        if path.name in NOT_A_MODULE:
            continue
        text = path.read_text(encoding="utf-8")
        sections = set(SECTION_RE.findall(text))
        if not sections:
            continue
        cmds = {m.group(0) for m in COMMAND_RE.finditer(text)}
        for section in sections:
            modules[section].add(path.name)
            commands[section] |= cmds
    return modules, commands


def guide_citations() -> dict[str, set[str]]:
    """``section -> {rule names}`` for every named rule in ``sna/guide.py`` that cites it.

    Reads the rule *objects*, not the file's text, so a rewording inside a rule's prose can never
    leave this index quoting a stale sentence -- only the ``§`` citations inside that prose, and
    the rule's own ``name``, matter here.
    """
    out: dict[str, set[str]] = defaultdict(set)

    def scan(name: str, obj: object) -> None:
        text = " ".join(str(getattr(obj, field.name)) for field in fields(obj))
        for section in SECTION_RE.findall(text):
            out[section].add(name)

    for rule in METHOD_RULES:
        scan(rule.name, rule)
    for rule in NETWORK_RULES:
        scan(rule.name, rule)
    for _heading, rules in READING_RULES:
        for rule in rules:
            scan(rule.name, rule)
    return out


def foundations_citations() -> dict[str, str]:
    """``section -> markdown link`` into whichever ``docs/SNA_FOUNDATIONS.md`` heading covers it.

    Attributes every ``§`` -- including one named in the heading itself, which is where the file's
    "Documented, not built" subsections state their own disposition (``### §13.5 -- classic
    combinatorial problems``) -- to the most recent heading of any level (headings nest, so a
    citation under a ``###`` is credited to that ``###``, not to the ``##`` around it). A section
    that is genuinely built keeps citing itself from its own module or rule, so
    :func:`build_rows` only shows this link where that is the *only* thing that cites the row;
    a passing mention inside a "documented, not built" paragraph of a section the codebase does
    build elsewhere (``§45.2``'s own text cites ``§19.2``'s ERGM, which ``null.py`` also cites)
    is exactly the case that guard exists for.
    """
    text = FOUNDATIONS.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    heading = ""
    link = ""
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            link = f"SNA_FOUNDATIONS.md#{slugify(heading)}"
        for section in SECTION_RE.findall(line):
            out.setdefault(section, f"[{heading}]({link})")
    return out


def resolve(
    citation: str, index: dict[str, tuple[atlas_chapter.Chapter, atlas_chapter.Section | None]]
) -> str:
    """The row key a raw ``§`` citation maps onto: the exact section if the outline has it, else
    its chapter -- never an invented section the book's own table of contents does not list."""
    if citation in index:
        return citation
    chapter_number = citation.split(".")[0]
    return chapter_number if chapter_number in index else ""


def build_rows() -> list[dict[str, str]]:
    pdf = atlas_chapter.ensure_pdf()
    chapters = atlas_chapter.read_outline(pdf)
    index: dict[str, tuple[atlas_chapter.Chapter, atlas_chapter.Section | None]] = {}
    for chapter in chapters:
        index[str(chapter.number)] = (chapter, None)
        for section in chapter.sections:
            index[section.number] = (chapter, section)

    raw_modules, raw_commands = module_citations()
    raw_rules = guide_citations()
    raw_documented = foundations_citations()

    modules: dict[str, set[str]] = defaultdict(set)
    commands: dict[str, set[str]] = defaultdict(set)
    rules: dict[str, set[str]] = defaultdict(set)
    documented: dict[str, str] = {}
    for citation, names in raw_modules.items():
        key = resolve(citation, index)
        if key:
            modules[key] |= names
    for citation, names in raw_commands.items():
        key = resolve(citation, index)
        if key:
            commands[key] |= names
    for citation, names in raw_rules.items():
        key = resolve(citation, index)
        if key:
            rules[key] |= names
    for citation, link in raw_documented.items():
        key = resolve(citation, index)
        if key and key not in documented:
            documented[key] = link

    def row_of(key: str, title: str, pages: str, chapter: str, section: str) -> dict[str, str]:
        # "documented, not built" is a disposition for a row nothing *else* cites -- a section
        # this codebase also builds is not "documented instead", it is documented *and* built,
        # which the Modules/Rules columns already say, so a stray mention of it inside some other
        # section's "documented, not built" prose (§45.2's own text cites §19.2's ERGM, which
        # `null.py` builds) must not relabel it here.
        row_modules = modules.get(key, set())
        row_rules = rules.get(key, set())
        return {
            "chapter": chapter,
            "section": section,
            "title": title,
            "pages": pages,
            "modules": ", ".join(sorted(row_modules)),
            "commands": ", ".join(sorted(commands.get(key, set()))),
            "rules": ", ".join(sorted(row_rules)),
            "documented": "" if row_modules or row_rules else documented.get(key, ""),
        }

    rows: list[dict[str, str]] = []
    for chapter in chapters:
        rows.append(
            row_of(
                str(chapter.number),
                chapter.title,
                f"{chapter.first}-{chapter.last}",
                str(chapter.number),
                "",
            )
        )
        for section in chapter.sections:
            rows.append(
                row_of(
                    section.number,
                    section.title,
                    f"{section.first}-{section.last}",
                    str(chapter.number),
                    section.number,
                )
            )
    return rows


def cell(value: str) -> str:
    return value if value else "—"


def render_index() -> str:
    rows = build_rows()
    lines = [
        "# The Atlas index",
        "",
        "Generated by `scripts/atlas_index.py` from the book's own table of contents "
        "(`atlas_chapter.read_outline`), the `§n.m` citations in every `src/graphrag/sna/*.py` "
        "module docstring, the named rules in `graphrag.sna.guide`, and the headings of "
        "`docs/SNA_FOUNDATIONS.md`. **Do not hand-edit this file** -- "
        "`python scripts/atlas_index.py` regenerates it, and "
        "`tests/unit/test_atlas_index.py` holds the committed copy to that output verbatim.",
        "",
        "A blank row (`—` in every column but the title) is not a gap in this index; it is a "
        "chapter or section the book's own outline lists that no module cites by `§` number, "
        "which is either background prose (chapters 1-5 read the book's own conventions, not a "
        "network), the page before a chapter's first `§`, or a citation this codebase makes "
        "against the chapter as a whole (see the note below). Cross-check a blank row against "
        "`docs/ATLAS_PLAN.md`'s coverage matrix before concluding nothing answers it.",
        "",
        "A handful of citations in the code name a sub-point past what this printing's own "
        "outline lists for that chapter (its table of contents runs out at a lower number, e.g. "
        "a module cites `§22.6` where the outline's chapter 22 stops at `§22.4`). Those are "
        "credited to the chapter row, not invented as a section row the book does not list.",
        "",
        "| Ch. | § | Title | Pages | Modules | Commands | Rules | Documented |",
        "|---:|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {chapter} | {section} | {title} | {pages} | {modules} | {commands} | {rules} | "
            "{documented} |".format(
                chapter=row["chapter"] if not row["section"] else "",
                section=row["section"] or "—",
                title=row["title"],
                pages=row["pages"],
                modules=cell(row["modules"]),
                commands=cell(row["commands"]),
                rules=cell(row["rules"]),
                documented=row["documented"] or "—",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    text = render_index()
    if "--print" in argv:
        print(text)
        return 0
    if "--check" in argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.is_file() else ""
        if current != text:
            print(
                f"{OUTPUT} is stale; run `python scripts/atlas_index.py` and commit it.",
                file=sys.stderr,
            )
            return 1
        return 0
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
