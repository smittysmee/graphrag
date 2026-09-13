"""PostToolUse hook: say what a freshly written corpus document still owes the graph.

Writing a file under ``data/raw/<persona>/<source>/`` is the first of four passes, and the only
one that leaves visible evidence. The text reaches the graph on the next ``make sync``; its
entities, its speakers and what its passages say reach the graph only if somebody also writes
the three sidecar files, and nothing about the document announces that they are missing. The
usual result is a corpus that looks complete and answers entity, speaker and stance questions
with silence.

So this hook fires after ``Write``/``Edit``/``MultiEdit``, works out whether the path is a
corpus document, and if it is, states the document id the loaders will give it, the sidecar
files it should produce, and the one command that checks all of them. Which sidecars are named
is decided from the file itself and from the persona's own declared vocabulary, never from a
built-in list: attribution when the text carries attributed posts, annotation when the text uses
names the persona's ``aliases.yaml`` or ``facets.yaml`` declares.

Fail-silent like the other hooks here: any error, an unreadable file, or stdin that is not the
shape expected exits 0 with no output. A hook that fires after every edit must never wedge a
session, and a contract nobody can read is better than a turn that cannot finish.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.hooks import project
from graphrag.textutil import sanitize_inline, slugify

__all__ = [
    "CaptureTarget",
    "build_context",
    "build_output",
    "describe",
    "main",
    "resolve_target",
]

#: Tools that write a file. ``Edit`` and ``MultiEdit`` carry ``file_path`` exactly as ``Write``
#: does, and a document is just as un-captured after a rewrite as after a first draft.
WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit"})

#: Where each layer's sidecar lives, under the repository root.
LAYER_DIRS: dict[str, tuple[str, ...]] = {
    "extraction": ("data", "enrichment"),
    "attribution": ("data", "attribution"),
    "annotation": ("data", "annotations"),
}

_ALIAS_FILE = "aliases.yaml"
_FACET_FILE = "facets.yaml"

#: At most this much of a document is read to decide which layers it needs. Two passes over a
#: few hundred kilobytes is well inside the hook's budget, and no real note is longer.
_MAX_BYTES = 400_000
#: A post is only evidence of an attributed document when the pattern repeats.
_MIN_POSTS = 2
#: Shorter vocabulary terms match inside unrelated words too often to be evidence.
_MIN_TERM = 4
#: Filesystem-derived ids reach a line a model reads, so each is flattened to this width.
_MAX_ID_LEN = 60

# A post opening: a handle on its own, bold or plain, followed by a reporting verb or a colon
# ("**handle** wrote on 2025-02-03:", "handle said:"), or a transcript turn with a timestamp
# ("Name (00:00:00):"). Deliberately shallow: this decides which sidecar to name, not who spoke.
_POST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:\*\*|__)(?P<name>[^*_\n]{1,60}?)(?:\*\*|__)\s*"
        r"(?:wrote|posted|said|replied|commented|:)",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?P<name>[^\n(]{1,60}?)\s*\(\d{1,2}:\d{2}(?::\d{2})?\)\s*:",
        re.MULTILINE,
    ),
)


@dataclass(frozen=True)
class CaptureTarget:
    """A written path that is a corpus document, and everything the contract says about it."""

    root: Path
    path: Path
    persona_id: str
    source_id: str
    doc_id: str
    slug: str
    layers: tuple[str, ...]  # the layers this document is expected to produce
    reasons: Mapping[str, str] = None  # type: ignore[assignment]

    def sidecar(self, layer: str) -> Path:
        """``data/<layer dir>/<persona>/<source>/<slug>.json`` under the repository root."""
        return self.root.joinpath(*LAYER_DIRS[layer], self.persona_id, self.source_id).joinpath(
            f"{self.slug}.json"
        )


# ----------------------------------------------------------------------------- resolution


def resolve_target(payload: Mapping[str, Any]) -> CaptureTarget | None:
    """The capture target this payload describes, or ``None`` when it describes none.

    ``None`` covers a tool that does not write, a payload missing the fields this hook needs, a
    path outside ``data/raw``, a path whose suffix no loader reads, and a first path segment
    that is not a persona this repository defines.
    """
    if payload.get("tool_name") not in WRITE_TOOLS:
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw_path = tool_input.get("file_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    cwd = payload.get("cwd")
    root = project.find_root(Path(cwd) if isinstance(cwd, str) and cwd else None)
    if root is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return _target_for(root, path)


def _target_for(root: Path, path: Path) -> CaptureTarget | None:
    raw_base = root.joinpath(*project.RAW_DIR)
    try:
        rel = path.relative_to(raw_base)
    except ValueError:
        return None
    if path.suffix.lower() not in project.DOCUMENT_SUFFIXES or len(rel.parts) < 2:
        return None
    persona_id = rel.parts[0]
    try:
        persona = project.load_persona(root, persona_id)
    except (OSError, ValueError):
        return None
    placed = _place(root, persona, path)
    if placed is None:
        return None
    source, slug = placed
    body = _read_body(path)
    layers, reasons = _expected_layers(root, persona_id, body)
    return CaptureTarget(
        root=root,
        path=path,
        persona_id=persona.id,
        source_id=source.id,
        doc_id=f"{persona.id}:{source.id}:{slug}",
        slug=slug,
        layers=layers,
        reasons=reasons,
    )


def _place(
    root: Path, persona: project.PersonaInfo, path: Path
) -> tuple[project.SourceInfo, str] | None:
    """The source that covers ``path`` and the slug its loader would give it.

    The loaders' own two rules, chosen by the source's ``loader``: a ``documents`` source
    slugifies the path under the source without its suffix, a ``transcripts`` source slugifies
    the containing folder because one folder is one recording. A path covered by two sources
    belongs to the more specific one, i.e. the deeper base directory.
    """
    best: tuple[int, project.SourceInfo, Path] | None = None
    for source in persona.sources:
        base = project.source_base(root, persona, source)
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        depth = len(base.parts)
        if best is None or depth > best[0]:
            best = (depth, source, rel)
    if best is None:
        return None
    _depth, source, rel = best
    if source.loader == "transcripts":
        folder = rel.parent.name
        return source, slugify(folder if folder not in ("", ".") else rel.stem)
    return source, slugify(str(rel.with_suffix("")))


def _read_body(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return fh.read(_MAX_BYTES)
    except OSError:
        return ""


# ----------------------------------------------------------------------------- which layers


def _expected_layers(
    root: Path, persona_id: str, body: str
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Which sidecars this document should produce, and the reason for each answer.

    Extraction is expected of every document. Attribution is expected when the text carries
    attributed posts, which is the case the layer exists for: a thread arrives as one body of
    prose and the loader parses no speaker turns from it. Annotation is expected when the text
    uses vocabulary the persona itself has declared, so a persona that keeps neither an alias
    table nor a facet file never has the layer suggested to it.
    """
    reasons = {"extraction": "every document carries one"}
    layers = ["extraction"]

    posts = _count_posts(body)
    if posts >= _MIN_POSTS:
        reasons["attribution"] = f"the text carries {posts} attributed posts"
        layers.append("attribution")
    else:
        reasons["attribution"] = "no attributed posts in the text"

    terms = _vocabulary(root, persona_id)
    hit = _first_term(body, terms) if terms else ""
    if hit:
        # The term comes out of a YAML file on disk and is about to be spliced into text a
        # model reads as context, so it is flattened like every other filesystem-derived string.
        reasons["annotation"] = (
            f"the text uses declared vocabulary, such as {sanitize_inline(hit, _MAX_ID_LEN)!r}"
        )
        layers.append("annotation")
    elif not terms:
        reasons["annotation"] = f"{persona_id} declares no alias or facet vocabulary"
    else:
        reasons["annotation"] = "the text uses none of the persona's declared vocabulary"
    return tuple(layers), reasons


def _count_posts(body: str) -> int:
    """How many distinct post openings the text carries. Zero for ordinary prose."""
    names: set[str] = set()
    for pattern in _POST_PATTERNS:
        for match in pattern.finditer(body):
            name = " ".join(match.group("name").split()).lower()
            if name and len(name) <= 60:
                names.add(name)
    return len(names)


def _vocabulary(root: Path, persona_id: str) -> set[str]:
    """Every name and facet word the persona declares, folded, long enough to be evidence."""
    base = root / project.PERSONAS_DIR / persona_id
    terms: set[str] = set()
    aliases = _read_yaml(base / _ALIAS_FILE).get("aliases")
    if isinstance(aliases, Mapping):
        for canonical, spellings in aliases.items():
            for name in (canonical, *(spellings if isinstance(spellings, list) else [])):
                folded = " ".join(str(name).split()).lower()
                if len(folded) >= _MIN_TERM:
                    terms.add(folded)
    facets = _read_yaml(base / _FACET_FILE).get("facets")
    if isinstance(facets, Mapping):
        for key in facets:
            for word in slugify(str(key)).split("-"):
                if len(word) >= _MIN_TERM:
                    terms.add(word)
    return terms


def _first_term(body: str, terms: set[str]) -> str:
    """The first declared term the text uses as a whole word, or an empty string."""
    folded = " ".join(body.split()).lower()
    for term in sorted(terms, key=lambda t: (-len(t), t)):
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded):
            return term
    return ""


def _read_yaml(path: Path) -> Mapping[str, Any]:
    """One persona YAML file, read with PyYAML when the host has it and by hand otherwise."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:  # PyYAML is not guaranteed to exist on the host's python3.
        from yaml import safe_load
    except ImportError:
        pass
    else:
        try:
            data = safe_load(text)
        except Exception:
            data = None
        if isinstance(data, Mapping):
            return data
    parsed = project.parse_simple_yaml(text)
    return parsed if isinstance(parsed, Mapping) else {}


# ----------------------------------------------------------------------------- the message


def describe(target: CaptureTarget) -> str:
    """The contract for one document, as statements about what the corpus holds and expects.

    Written as facts rather than as commands on purpose: injected context that reads like an
    out-of-band instruction trips a model's prompt-injection defences and gets surfaced to the
    user instead of used. Every id in it is filesystem-derived, so each is flattened first.
    """
    persona = sanitize_inline(target.persona_id, _MAX_ID_LEN)
    rel = project.rel_path(target.root, target.path)
    lines = [
        f"graph-rag capture contract. {sanitize_inline(rel, 120)} is a corpus document for "
        f"persona {persona}, source {sanitize_inline(target.source_id, _MAX_ID_LEN)}. Ingesting "
        f"it produces document id {sanitize_inline(target.doc_id, 3 * _MAX_ID_LEN)}.",
        "The corpus expects these sidecar files for it:",
    ]
    for layer in LAYER_DIRS:
        expected = "expected" if layer in target.layers else "not indicated"
        state = " (already on disk)" if target.sidecar(layer).is_file() else ""
        lines.append(
            f"- {layer}, {expected} ({target.reasons.get(layer, '')}): "
            f"{project.rel_path(target.root, target.sidecar(layer))}{state}"
        )
    lines += [
        "The formats and the procedure are in the graph-rag-capture skill at "
        ".claude/skills/graph-rag-capture/SKILL.md.",
        "The command that verifies every layer of this document is:",
        f"  docker compose run --rm -T graphrag graphrag layers check {persona} "
        f"--file {sanitize_inline(rel, 120)}",
        "A document counts as captured when that check exits 0; until then its layers are "
        "incomplete and the graph answers entity, speaker and stance questions about it with "
        "nothing.",
    ]
    return "\n".join(lines)


def build_context(payload: Mapping[str, Any]) -> str | None:
    """The text to inject for this payload, or ``None`` when the path is not a corpus document."""
    target = resolve_target(payload)
    return describe(target) if target is not None else None


def build_output(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``PostToolUse`` JSON for this payload, or ``None`` to say nothing at all."""
    context = build_context(payload)
    if context is None:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    }


def _read_stdin_payload() -> dict[str, Any]:
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def main() -> int:
    """Read the ``PostToolUse`` stdin payload; print the capture contract when one applies."""
    try:
        output = build_output(_read_stdin_payload())
        if output is not None:
            print(json.dumps(output))
    except Exception:  # a broken hook must never fail a tool call
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
