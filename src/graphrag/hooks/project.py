"""Read the repository from the host: personas, snapshots, raw corpus, expected document ids.

The container-side code answers these questions through pydantic models and ``Settings``; a
hook cannot import either, so this module re-reads the same files with the standard library
and deliberately narrow, read-only dataclasses. PyYAML is used when the host's ``python3``
happens to have it and a small best-effort parser covers the persona keys otherwise.
"""

from __future__ import annotations

import gzip
import heapq
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

from graphrag.textutil import slugify

__all__ = [
    "DOCUMENT_SUFFIXES",
    "Manifest",
    "PersonaInfo",
    "SourceInfo",
    "cache_dir",
    "enriched_doc_ids",
    "expected_doc_id",
    "find_root",
    "load_manifests",
    "load_persona",
    "load_personas",
    "newest_raw_files",
    "persona_ids",
    "raw_files",
    "raw_root",
    "rel_path",
    "snapshot_doc_ids",
    "source_base",
]

PERSONAS_DIR = "personas"
PERSONA_FILE = "persona.yaml"
RAW_DIR = ("data", "raw")
SNAPSHOT_DIR = ("data", "snapshots")
CACHE_DIR = ("data", "cache", "hooks")
MANIFEST_FILE = "manifest.json"
DOCUMENTS_FILE = "documents.jsonl.gz"

#: Suffixes the ``documents`` loader accepts (mirrors ``graphrag.ingest.loaders``).
DOCUMENT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".pdf"})
_PRUNE = frozenset({".git", "__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"})
_MAX_WALK_FILES = 20000

_yaml_safe_load: Callable[[str], Any] | None
try:
    from yaml import safe_load as _imported_safe_load
except ImportError:  # PyYAML is not guaranteed to exist on the host's python3.
    _yaml_safe_load = None
else:
    _yaml_safe_load = _imported_safe_load


# ----------------------------------------------------------------------------- data


@dataclass(frozen=True)
class SourceInfo:
    """One entry of a persona's ``sources:`` list, or of a snapshot manifest."""

    id: str
    kind: str = "local"
    url: str | None = None
    path: str = ""
    loader: str = "documents"
    glob: str = "**/*.md"
    description: str = ""


@dataclass(frozen=True)
class PersonaInfo:
    """``personas/<id>/persona.yaml``, reduced to what a hook needs."""

    id: str
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    sources: tuple[SourceInfo, ...] = ()


@dataclass(frozen=True)
class Manifest:
    """``data/snapshots/<persona>/manifest.json``."""

    persona_id: str
    created_at: str = ""
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0
    embedding_model: str = ""
    sources: tuple[SourceInfo, ...] = ()
    path: Path = field(default_factory=Path)

    @property
    def created_date(self) -> str:
        """The ``YYYY-MM-DD`` part of ``created_at``, or an empty string."""
        return self.created_at[:10] if len(self.created_at) >= 10 else ""


# ----------------------------------------------------------------------------- layout


def find_root(start: Path | None = None) -> Path | None:
    """The repository root: ``$CLAUDE_PROJECT_DIR`` when it looks right, else walk up.

    A hook's stdin ``cwd`` is not guaranteed to be the root, so callers pass it as ``start``
    and fall back to the process working directory.
    """
    env = os.environ.get("CLAUDE_PROJECT_DIR", "").strip()
    if env:
        candidate = Path(env).expanduser()
        if _is_root(candidate):
            return candidate.resolve()
    here = (start or Path.cwd()).expanduser()
    try:
        here = here.resolve()
    except OSError:
        return None
    for directory in (here, *here.parents):
        if _is_root(directory):
            return directory
    return None


def _is_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / PERSONAS_DIR).is_dir()


def raw_root(root: Path, persona: PersonaInfo | str) -> Path:
    """``<root>/data/raw/<persona>`` -- the path ``make ingest`` passes as SRC."""
    return root.joinpath(*RAW_DIR, _persona_id(persona))


def snapshot_root(root: Path, persona: PersonaInfo | str) -> Path:
    return root.joinpath(*SNAPSHOT_DIR, _persona_id(persona))


def cache_dir(root: Path) -> Path:
    """``<root>/data/cache/hooks``, created on demand. Gitignored via ``data/cache/``."""
    path = root.joinpath(*CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_path(root: Path, path: Path) -> str:
    """``path`` relative to ``root`` when possible, for display."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _persona_id(persona: PersonaInfo | str) -> str:
    return persona if isinstance(persona, str) else persona.id


# ----------------------------------------------------------------------------- personas


def persona_ids(root: Path) -> list[str]:
    """Ids of every persona directory that holds a ``persona.yaml``, sorted."""
    base = root / PERSONAS_DIR
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / PERSONA_FILE).is_file())


def load_persona(root: Path, persona_id: str) -> PersonaInfo:
    """Read one ``persona.yaml``. Raises ``FileNotFoundError`` when it is not there."""
    path = root / PERSONAS_DIR / persona_id / PERSONA_FILE
    text = path.read_text(encoding="utf-8", errors="replace")
    return _persona_from_mapping(_load_yaml(text), persona_id)


def load_personas(root: Path) -> dict[str, PersonaInfo]:
    """Every readable persona, keyed by id. Unreadable ones are skipped, not raised."""
    out: dict[str, PersonaInfo] = {}
    for pid in persona_ids(root):
        try:
            out[pid] = load_persona(root, pid)
        except (OSError, ValueError):  # a malformed persona must not take a hook down
            continue
    return out


def _persona_from_mapping(data: Mapping[str, Any], persona_id: str) -> PersonaInfo:
    pid = _text(data.get("id")) or persona_id
    return PersonaInfo(
        id=pid,
        name=_text(data.get("name")) or pid,
        description=" ".join(_text(data.get("description")).split()),
        tags=tuple(t for t in (_text(v) for v in _as_list(data.get("tags"))) if t),
        sources=tuple(
            _source_from_mapping(item) for item in _as_list(data.get("sources")) if _has_id(item)
        ),
    )


def _source_from_mapping(data: Any) -> SourceInfo:
    mapping: Mapping[str, Any] = data if isinstance(data, Mapping) else {}
    url = _text(mapping.get("url"))
    return SourceInfo(
        id=_text(mapping.get("id")),
        kind=_text(mapping.get("kind")) or "local",
        url=url or None,
        path=_text(mapping.get("path")),
        loader=_text(mapping.get("loader")) or "documents",
        glob=_text(mapping.get("glob")) or "**/*.md",
        description=" ".join(_text(mapping.get("description")).split()),
    )


def _has_id(item: Any) -> bool:
    return isinstance(item, Mapping) and bool(_text(item.get("id")))


def _text(value: Any) -> str:
    if value is None or isinstance(value, (list, dict)):
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


# ----------------------------------------------------------------------------- snapshots


def load_manifests(root: Path) -> dict[str, Manifest]:
    """Every committed snapshot manifest, keyed by persona id."""
    base = root.joinpath(*SNAPSHOT_DIR)
    if not base.is_dir():
        return {}
    out: dict[str, Manifest] = {}
    for child in sorted(base.iterdir()):
        manifest_file = child / MANIFEST_FILE
        if not manifest_file.is_file():
            continue
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # a truncated snapshot must not take a hook down
            continue
        if not isinstance(data, Mapping):
            continue
        pid = _text(data.get("persona_id")) or child.name
        out[pid] = Manifest(
            persona_id=pid,
            created_at=_text(data.get("created_at")),
            document_count=_int(data.get("document_count")),
            chunk_count=_int(data.get("chunk_count")),
            entity_count=_int(data.get("entity_count")),
            embedding_model=_text(data.get("embedding_model")),
            sources=tuple(
                _source_from_mapping(item)
                for item in _as_list(data.get("sources"))
                if _has_id(item)
            ),
            path=child,
        )
    return out


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def snapshot_doc_ids(root: Path, persona: PersonaInfo | str) -> set[str]:
    """Document ids inside the committed ``documents.jsonl.gz``. Empty when unreadable."""
    path = snapshot_root(root, persona) / DOCUMENTS_FILE
    out: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                doc_id = _doc_id_of(line)
                if doc_id:
                    out.add(doc_id)
    except (OSError, EOFError, gzip.BadGzipFile):  # no snapshot, or a partial one
        return out
    return out


ENRICHMENT_DIR = "data/enrichment"


def enriched_doc_ids(root: Path) -> set[str]:
    """``doc_id`` of every extraction JSON under ``data/enrichment``, whatever the layout.

    The agent-driven enrichment path writes one ``{doc_id, entities, relations}`` file per
    document, flat or nested per persona/source; the id inside the file is what counts, not
    its path. Unreadable files are skipped, so this never raises.
    """
    base = root / ENRICHMENT_DIR
    out: set[str] = set()
    if not base.is_dir():
        return out
    for path in base.rglob("*.json"):
        try:
            with path.open(encoding="utf-8", errors="replace") as fh:
                record = json.load(fh)
        except (OSError, ValueError):
            continue
        doc_id = record.get("doc_id") if isinstance(record, dict) else None
        if isinstance(doc_id, str) and doc_id.strip():
            out.add(doc_id.strip())
    return out


def _doc_id_of(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    try:
        record = json.loads(line)
    except ValueError:
        return ""
    return _text(record.get("id")) if isinstance(record, Mapping) else ""


# ----------------------------------------------------------------------------- raw corpus


def source_base(root: Path, persona: PersonaInfo | str, source: SourceInfo) -> Path:
    """The directory a source's ``glob`` is evaluated against (mirrors the loaders)."""
    base = raw_root(root, persona)
    return base / source.path if source.path else base


def raw_files(root: Path, persona: PersonaInfo | str, source: SourceInfo) -> list[Path]:
    """Files the ingest pipeline would pick up for this source, in the loader's order."""
    base = source_base(root, persona, source)
    if not base.is_dir():
        return []
    try:
        files = sorted(p for p in base.glob(source.glob) if p.is_file())
    except (OSError, ValueError):  # a malformed glob must not take a hook down
        return []
    if source.loader == "documents":
        files = [p for p in files if p.suffix.lower() in DOCUMENT_SUFFIXES]
    return files


def expected_doc_id(root: Path, persona: PersonaInfo | str, source: SourceInfo, path: Path) -> str:
    """The id the ``documents`` loader gives ``path``.

    ``<persona>:<source>:<slug of the path under the source, without its suffix>``, matching
    ``graphrag.ingest.documents.load_document``. Only meaningful for the ``documents`` loader:
    the ``transcripts`` loader slugifies the containing folder and drops duplicate bodies, so
    callers compare it at the count level instead.
    """
    base = source_base(root, persona, source)
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = Path(path.name)
    return f"{_persona_id(persona)}:{source.id}:{slugify(str(rel.with_suffix('')))}"


def newest_raw_files(root: Path, n: int = 5) -> list[tuple[Path, float]]:
    """The ``n`` most recently modified corpus files under ``data/raw``, newest first."""
    if n <= 0:
        return []
    return heapq.nlargest(n, _walk_raw(root), key=lambda item: item[1])


def _walk_raw(root: Path) -> Iterator[tuple[Path, float]]:
    base = root.joinpath(*RAW_DIR)
    if not base.is_dir():
        return
    seen = 0
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _PRUNE and not d.startswith("."))
        for name in filenames:
            if Path(name).suffix.lower() not in DOCUMENT_SUFFIXES or name.startswith("."):
                continue
            path = Path(dirpath) / name
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            seen += 1
            if seen > _MAX_WALK_FILES:
                return
            yield path, mtime


# ----------------------------------------------------------------------------- yaml


def _load_yaml(text: str) -> Mapping[str, Any]:
    if _yaml_safe_load is not None:
        try:
            data = _yaml_safe_load(text)
        except Exception:
            data = None
        if isinstance(data, Mapping):
            return data
    data = parse_simple_yaml(text)
    return data if isinstance(data, Mapping) else {}


class _Line(NamedTuple):
    indent: int
    text: str
    skip: bool


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
_BLOCK_MARKERS = frozenset({"|", "|-", "|+", ">", ">-", ">+"})


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """A best-effort YAML reader for persona files, used when PyYAML is missing.

    Covers what ``persona.yaml`` actually uses: scalars, ``[a, b]`` and ``- item`` lists,
    lists of mappings, nested mappings, and ``|``/``>`` block scalars. It is not a YAML
    implementation: anchors, flow mappings and multi-document files are not supported.
    """
    lines = [_line_of(raw) for raw in text.replace("\r\n", "\n").split("\n")]
    start = _next(lines, 0)
    if start >= len(lines):
        return {}
    value, _ = _parse_map(lines, start, lines[start].indent)
    return value


def _line_of(raw: str) -> _Line:
    expanded = raw.expandtabs(2).rstrip()
    stripped = expanded.strip()
    skip = not stripped or stripped.startswith("#") or stripped in ("---", "...")
    return _Line(len(expanded) - len(expanded.lstrip(" ")), expanded, skip)


def _next(lines: Sequence[_Line], i: int) -> int:
    while i < len(lines) and lines[i].skip:
        i += 1
    return i


def _parse_map(lines: Sequence[_Line], i: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while True:
        i = _next(lines, i)
        if i >= len(lines):
            return out, i
        line = lines[i]
        if line.indent != indent:
            return out, i
        content = line.text.strip()
        if content.startswith("-"):
            return out, i
        key, sep, rest = content.partition(":")
        key = key.strip().strip("\"'")
        if not sep or not key:
            return out, i
        rest = _strip_comment(rest.strip())
        if rest in _BLOCK_MARKERS:
            out[key], i = _parse_block_scalar(lines, i + 1, indent, rest)
        elif rest:
            out[key] = _scalar(rest)
            i += 1
        else:
            out[key], i = _parse_child(lines, i + 1, indent)


def _parse_child(lines: Sequence[_Line], i: int, indent: int) -> tuple[Any, int]:
    """The value of a ``key:`` with nothing after the colon: a nested block, or nothing."""
    j = _next(lines, i)
    if j >= len(lines):
        return None, j
    child = lines[j]
    is_item = child.text.strip().startswith("-")
    if child.indent > indent or (is_item and child.indent == indent):
        if is_item:
            return _parse_list(lines, j, child.indent)
        return _parse_map(lines, j, child.indent)
    return None, i


def _parse_list(lines: Sequence[_Line], i: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while True:
        i = _next(lines, i)
        if i >= len(lines) or lines[i].indent != indent:
            return out, i
        content = lines[i].text.strip()
        if not content.startswith("-"):
            return out, i
        item = _strip_comment(content[1:].strip())
        if _is_mapping_entry(item):
            value, consumed = _parse_map(_rebase(lines, i, indent, item), 0, indent + 2)
            out.append(value)
            i += consumed
        elif item:
            out.append(_scalar(item))
            i += 1
        else:
            value, i = _parse_child(lines, i + 1, indent)
            out.append(value)


def _rebase(lines: Sequence[_Line], i: int, indent: int, item: str) -> list[_Line]:
    """Re-present ``- key: value`` as a mapping line indented two past the dash.

    Index ``k`` of the result maps to index ``i + k`` of ``lines``, so the caller can advance
    by whatever ``_parse_map`` consumed.
    """
    pad = " " * (indent + 2)
    return [_Line(indent + 2, pad + item, False), *lines[i + 1 :]]


def _is_mapping_entry(item: str) -> bool:
    key, sep, _ = item.partition(":")
    return bool(sep) and bool(_KEY_RE.match(key.strip()))


def _parse_block_scalar(
    lines: Sequence[_Line], i: int, indent: int, marker: str
) -> tuple[str, int]:
    body: list[str] = []
    base: int | None = None
    while i < len(lines):
        line = lines[i]
        if not line.text.strip():
            body.append("")
            i += 1
            continue
        if line.indent <= indent:
            break
        if base is None:
            base = line.indent
        body.append(line.text[base:] if len(line.text) > base else line.text.strip())
        i += 1
    while body and not body[-1]:
        body.pop()
    text = _fold(body) if marker.startswith(">") else "\n".join(body)
    return (text if marker.endswith("-") else text + "\n") if text else "", i


def _fold(body: Iterable[str]) -> str:
    paragraphs: list[list[str]] = [[]]
    for line in body:
        if line.strip():
            paragraphs[-1].append(line.strip())
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def _strip_comment(value: str) -> str:
    if value.startswith(("'", '"')):
        return value
    head, sep, _ = value.partition(" #")
    return head.rstrip() if sep else value


def _scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_scalar(part) for part in inner.split(",") if part.strip()] if inner else []
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "~", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
