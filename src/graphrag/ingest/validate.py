"""Check a captured corpus before it becomes a graph.

A captured document is written by hand (or by an agent) long before anything reads it back, and
every way it can be wrong is silent. Front-matter that does not parse loses the whole file's
metadata; a missing ``fetched_at`` or ``retrieval`` key means nobody can later say where the text
came from or how faithful it is; a body of nine words produces a chunk that matches every query
weakly and answers none of them; a body of twelve thousand words is a page that was captured
whole instead of the passage that mattered. None of that raises anything at ingest time. It just
quietly lowers the quality of every answer the persona gives, months later, with no trace back to
the file that caused it.

So the contract is checked once, up front, against the files on disk:

* the front-matter parses as YAML and is a mapping;
* it carries every key in :data:`REQUIRED_KEYS`;
* anything that is not a ``research_note`` or ``internal_context`` also carries a ``source_url``,
  because a document fetched from somewhere has to say from where;
* the body is between :data:`MIN_BODY_WORDS` and :data:`MAX_BODY_WORDS` words.

Alongside the documents sits a provenance manifest: one JSON object per line per captured file,
written by whoever captured it. Those are checked too -- each line parses, and each line that
claims a successful capture names a file that exists -- and can be merged into a single
``provenance/provenance.jsonl`` for the corpus.

Nothing here knows about any particular corpus or market. It takes a root directory, or an
explicit list of files, and reports :class:`Problem` records; the caller decides whether a
problem is fatal.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "MAX_BODY_WORDS",
    "MERGED_MANIFEST",
    "MIN_BODY_WORDS",
    "PROVENANCE_DIR",
    "REQUIRED_KEYS",
    "URL_EXEMPT_TYPES",
    "FrontMatterError",
    "Problem",
    "ValidationReport",
    "iter_corpus_files",
    "merge_provenance",
    "parse_captured",
    "read_provenance",
    "validate_corpus",
    "validate_document",
    "validate_file",
    "validate_files",
]

#: Front-matter keys every captured document must carry.
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "title",
        "fetched_at",
        "fetched_by",
        "retrieval",
        "content_fidelity",
        "document_type",
        "category",
    }
)

#: Document types written by the researcher rather than fetched, so they need no ``source_url``.
URL_EXEMPT_TYPES: frozenset[str] = frozenset({"research_note", "internal_context"})

#: A body shorter than this is a stub: it chunks into something that matches weakly and answers
#: nothing. A body longer than this is a whole site section captured instead of the passage.
MIN_BODY_WORDS = 40
MAX_BODY_WORDS = 6000

#: Where the per-researcher manifests live, and the name of the merged one.
PROVENANCE_DIR = "provenance"
MERGED_MANIFEST = "provenance.jsonl"

#: Manifests that are outputs rather than inputs, so merging never reads its own result back in.
_GENERATED_MANIFESTS: frozenset[str] = frozenset({MERGED_MANIFEST, "doc-ids.jsonl"})

#: Suffixes a captured document can have. Other files under the root are left alone.
DOCUMENT_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown"})

_FENCE = "---"


class FrontMatterError(ValueError):
    """The file does not open with a parsable ``---`` fenced YAML mapping."""


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with one file, named the way a reviewer needs to hear it.

    ``where`` is relative to the corpus root (``provenance/alex.jsonl:12`` for a manifest line),
    so a report can be pasted into a message and still point at something.
    """

    where: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.detail}: {self.where}" if self.detail else self.where


@dataclass(slots=True)
class ValidationReport:
    """Everything one pass over a corpus found: the problems, and what it counted on the way."""

    root: Path
    problems: list[Problem] = field(default_factory=list)
    documents_per_source: Counter[str] = field(default_factory=Counter)
    document_types: Counter[str] = field(default_factory=Counter)
    provenance: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing is wrong. This is what a caller gates on."""
        return not self.problems

    @property
    def documents(self) -> int:
        return sum(self.documents_per_source.values())

    def extend(self, problems: Iterable[Problem]) -> None:
        self.problems.extend(problems)

    def problem_lines(self) -> list[str]:
        return [str(p) for p in self.problems]

    def summary_lines(self) -> list[str]:
        """The three counting lines, printed after the problems."""
        return [
            f"documents per source: {dict(self.documents_per_source)}",
            f"document types: {dict(self.document_types)}",
            f"provenance records: {len(self.provenance)}; problems: {len(self.problems)}",
        ]


def parse_captured(text: str) -> tuple[dict[str, Any], str]:
    """Split a captured document into ``(front-matter, body)``.

    Unlike :func:`graphrag.ingest.frontmatter.split_frontmatter`, which treats a file with no
    front-matter as a body with empty metadata, this raises: a captured document without
    front-matter has lost its provenance and is a problem to report, not a default to apply.
    """
    if not text.startswith(_FENCE):
        msg = "no front-matter fence"
        raise FrontMatterError(msg)
    end = text.find(f"\n{_FENCE}", len(_FENCE))
    if end == -1:
        msg = "front-matter fence never closes"
        raise FrontMatterError(msg)
    try:
        loaded = yaml.safe_load(text[len(_FENCE) : end])
    except yaml.YAMLError as exc:
        msg = f"yaml error: {exc}"
        raise FrontMatterError(msg) from exc
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        msg = "front-matter is not a mapping"
        raise FrontMatterError(msg)
    meta: dict[str, Any] = loaded
    return meta, text[end + len(_FENCE) + 1 :]


def validate_document(meta: dict[str, Any], body: str, *, where: str) -> list[Problem]:
    """Check one already-parsed document against the contract."""
    problems: list[Problem] = []
    missing = REQUIRED_KEYS - set(meta)
    if missing:
        problems.append(Problem(where, "missing-keys", f"MISSING {sorted(missing)}"))
    words = len(body.split())
    if words > MAX_BODY_WORDS:
        problems.append(Problem(where, "too-long", f"TOO LONG ({words} words)"))
    if words < MIN_BODY_WORDS:
        problems.append(Problem(where, "near-empty", f"NEAR-EMPTY ({words} words)"))
    if meta.get("document_type") not in URL_EXEMPT_TYPES and not meta.get("source_url"):
        problems.append(Problem(where, "no-source-url", "NO source_url"))
    return problems


def validate_file(path: Path, *, root: Path) -> tuple[list[Problem], dict[str, Any]]:
    """Check one file on disk. Returns its problems and its front-matter (empty if unparsable)."""
    where = _relative(path, root)
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        meta, body = parse_captured(text)
    except FrontMatterError as exc:
        return [Problem(where, "frontmatter", f"BAD front-matter ({exc})")], {}
    return validate_document(meta, body, where=where), meta


def iter_corpus_files(root: Path) -> list[Path]:
    """Every captured document under ``root``, in sorted order, skipping the manifests."""
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in DOCUMENT_SUFFIXES
        and _relative(path, root).split("/")[0] != PROVENANCE_DIR
    ]


def validate_files(files: Sequence[Path], *, root: Path) -> ValidationReport:
    """Check an explicit list of files. Used by ``graphrag ingest`` for one source's files."""
    report = ValidationReport(root=root)
    for path in files:
        problems, meta = validate_file(path, root=root)
        report.extend(problems)
        rel = _relative(path, root)
        report.documents_per_source[rel.split("/")[0]] += 1
        if meta:
            report.document_types[str(meta.get("document_type"))] += 1
    return report


def read_provenance(root: Path) -> tuple[list[dict[str, Any]], list[Problem]]:
    """Read every per-researcher manifest under ``root/provenance``.

    A record that claims a successful capture has to name a file that is actually there: the
    common failure is a manifest line written for a document that was renamed or never saved.
    """
    records: list[dict[str, Any]] = []
    problems: list[Problem] = []
    for manifest in sorted((root / PROVENANCE_DIR).glob("*.jsonl")):
        if manifest.name in _GENERATED_MANIFESTS:
            continue
        lines = manifest.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            where = f"{PROVENANCE_DIR}/{manifest.name}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems.append(Problem(where, "bad-json", "BAD JSON"))
                continue
            if not isinstance(record, dict):
                problems.append(Problem(where, "bad-json", "BAD JSON (not an object)"))
                continue
            named = record.get("file")
            if record.get("status", "ok") == "ok" and named and not (root / str(named)).exists():
                problems.append(
                    Problem(
                        f"{where} -> {named}", "missing-file", "PROVENANCE points at missing file"
                    )
                )
            records.append(record)
    return records, problems


def merge_provenance(root: Path, records: Sequence[dict[str, Any]]) -> Path:
    """Write every record into one ``provenance/provenance.jsonl`` and return its path."""
    out = root / PROVENANCE_DIR / MERGED_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return out


def validate_corpus(root: Path) -> ValidationReport:
    """Check every captured document under ``root`` and the provenance manifests beside them."""
    report = validate_files(iter_corpus_files(root), root=root)
    records, problems = read_provenance(root)
    report.provenance = records
    report.extend(problems)
    return report


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
