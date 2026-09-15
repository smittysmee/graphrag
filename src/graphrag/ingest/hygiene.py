"""Ingest hygiene: strip what a reader cannot see, flag what a reader should look at.

Corpus files are untrusted input: web captures, archived transcripts, notes written by agents
that read the web. The two jobs here are deliberately kept apart.

``clean_text``
    Removes characters and markup that are invisible in a rendered document but reach a model
    verbatim: hidden code points, control characters, HTML comments, ``<script>`` and ``<style>``
    blocks. Everything else survives byte for byte, Markdown included, and running it twice
    changes nothing.

``suspicious_spans``
    Reports injection-shaped text without touching it. Ingest never refuses a file on this
    evidence: the flags are advisory and a human decides what they mean.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Hidden code points, by class. Zero-width joiners and the byte-order mark can hide a whole
# sentence between two visible words; the bidi overrides can reverse one on screen while the
# underlying text says something else.
ZERO_WIDTH = "\u200b-\u200f\u2060-\u2064\ufeff"
BIDI_OVERRIDE = "\u202a-\u202e"
# C0 and C1 controls apart from tab (\x09) and newline (\x0a). Carriage return is in the range,
# so CRLF input is normalised to LF.
CONTROL = "\x00-\x08\x0b-\x1f\x7f-\x9f"

HIDDEN_CLASSES: tuple[tuple[str, str], ...] = (
    ("zero-width", f"[{ZERO_WIDTH}]"),
    ("bidi-override", f"[{BIDI_OVERRIDE}]"),
    ("control-character", f"[{CONTROL}]"),
)

MARKUP_BLOCKS: tuple[tuple[str, str], ...] = (
    ("html-comment", r"<!--.*?-->"),
    ("script-block", r"<script\b[^>]*>.*?</script\s*>"),
    ("style-block", r"<style\b[^>]*>.*?</style\s*>"),
)

# Phrases that read as an instruction to a model rather than as prose. One tuple so the list is
# easy to extend: add a ``(kind, pattern)`` pair, nothing else changes. Matching is
# case-insensitive and ``^`` anchors to a line start.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("injection-phrase", r"ignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+instructions"),
    ("injection-phrase", r"disregard\s+(?:your|the)\s+(?:rules|instructions)"),
    ("injection-phrase", r"you\s+are\s+now\b"),
    ("injection-phrase", r"\bsystem\s+prompt\b"),
    ("injection-phrase", r"do\s+not\s+tell\s+the\s+user"),
    ("injection-phrase", r"\bas\s+an\s+ai\b"),
    ("chat-token", r"^assistant:\s"),
    ("chat-token", r"\[/?INST\]"),
    ("chat-token", r"<\|im_(?:start|end)\|>"),
)

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
EXCERPT_CHARS = 80
MAX_SPANS = 50  # an adversarial file should not be able to produce an unbounded report

_LEAD_CHARS = 20
_MAX_PASSES = 3  # nested markup can reveal a new block once the inner one is gone

_HIDDEN = re.compile(f"[{ZERO_WIDTH}{BIDI_OVERRIDE}{CONTROL}]")
_WHITESPACE = re.compile(r"\s+")
_MARKUP = re.compile(
    "|".join(pattern for _kind, pattern in MARKUP_BLOCKS), re.IGNORECASE | re.DOTALL
)
_HIDDEN_BY_CLASS = tuple((kind, re.compile(pattern)) for kind, pattern in HIDDEN_CLASSES)
_MARKUP_BY_CLASS = tuple(
    (kind, re.compile(pattern, re.IGNORECASE | re.DOTALL)) for kind, pattern in MARKUP_BLOCKS
)
_INJECTION = tuple(
    (kind, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for kind, pattern in INJECTION_PATTERNS
)


@dataclass(frozen=True)
class Suspicion:
    """One reason a human might want to read a source file before trusting it."""

    kind: str
    excerpt: str
    offset: int


def clean_text(text: str) -> str:
    """Remove hidden characters and invisible markup, preserving everything else verbatim.

    Idempotent: ``clean_text(clean_text(t)) == clean_text(t)``.
    """
    return _HIDDEN.sub("", _strip_markup(text))


def _strip_markup(text: str) -> str:
    for _ in range(_MAX_PASSES):
        stripped = _MARKUP.sub("", text)
        if stripped == text:
            break
        text = stripped
    return text


def suspicious_spans(text: str) -> list[Suspicion]:
    """Everything worth flagging in ``text``, in document order.

    Hidden-character classes are reported once each: a file with four hundred zero-width spaces
    is one finding, not four hundred. Markup blocks and injection phrases are reported per
    occurrence, capped at ``MAX_SPANS``.
    """
    found: list[Suspicion] = []
    for kind, pattern in _HIDDEN_BY_CLASS:
        match = pattern.search(text)
        if match:
            found.append(Suspicion(kind, _excerpt(text, match.start()), match.start()))
    for kind, pattern in (*_MARKUP_BY_CLASS, *_INJECTION):
        found.extend(
            Suspicion(kind, _excerpt(text, m.start()), m.start()) for m in pattern.finditer(text)
        )
    found.sort(key=lambda s: (s.offset, s.kind))
    return found[:MAX_SPANS]


def summarize(suspicions: Sequence[Suspicion]) -> str:
    """A one-line reason for a report: the kinds found, then the first excerpt."""
    if not suspicions:
        return ""
    counts = Counter(s.kind for s in suspicions)  # insertion order is document order
    kinds = ", ".join(f"{kind} x{n}" if n > 1 else kind for kind, n in counts.items())
    first = next((s.excerpt for s in suspicions if s.excerpt), "")
    return f"{kinds}: {first}" if first else kinds


def scan_file(path: Path) -> list[Suspicion]:
    """Suspicions in one source file; none when it is binary, missing or unreadable.

    PDFs are skipped on purpose: the text pypdf extracts is littered with control characters,
    so scanning them would flag every file and teach the reader to ignore the flags.
    """
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return suspicious_spans(text)


def _excerpt(text: str, start: int) -> str:
    begin = max(0, start - _LEAD_CHARS)
    return _one_line(text[begin : begin + EXCERPT_CHARS])[:EXCERPT_CHARS]


def _one_line(text: str) -> str:
    return _WHITESPACE.sub(" ", _HIDDEN.sub("", text)).strip()
