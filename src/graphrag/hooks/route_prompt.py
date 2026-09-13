"""UserPromptSubmit hook: notice a prompt belongs to a persona and say so.

This is routing, not retrieval: it never quotes graph passages, only names a persona, the
``context()`` call to make, and up to three document titles that look relevant. A prompt
scores against a small per-persona vocabulary built from the persona's id, name, tags, and its
top topics (cached under ``data/cache/hooks/`` with a day's TTL so a live prompt never waits on
a topics() round trip more often than that). Slash commands, short prompts, and pasted
code/stack traces are skipped in silence -- there is nothing to route them to.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphrag.hooks import project
from graphrag.hooks.mcp_client import McpClient, McpUnavailable, connect

__all__ = [
    "MAX_PERSONAS",
    "MAX_TITLES",
    "MIN_SCORE",
    "MIN_WORDS",
    "TOPICS_LIMIT",
    "TOPIC_CACHE_TTL",
    "PersonaMatch",
    "VocabTerm",
    "build_vocabulary",
    "is_skippable",
    "looks_like_code",
    "main",
    "render_persona",
    "route",
    "score_persona",
]

MIN_WORDS = 4
MIN_TOKEN_LEN = 3
TOPICS_LIMIT = 60
TOPIC_CACHE_TTL = 24 * 60 * 60  # seconds
MAX_PERSONAS = 2
MIN_SCORE = 3
MAX_TITLES = 3
MAX_PARAGRAPH = 700
MAX_TITLE_LEN = 90
MAX_MATCHED_SHOWN = 6

ID_NAME_WEIGHT = 3
TAG_WEIGHT = 2
TOPIC_WEIGHT = 1

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "how",
        "what",
        "when",
        "where",
        "why",
        "who",
        "which",
        "should",
        "would",
        "could",
        "can",
        "do",
        "does",
        "did",
        "we",
        "you",
        "they",
        "he",
        "she",
        "them",
        "our",
        "your",
        "their",
        "about",
        "into",
        "if",
        "than",
        "then",
        "so",
        "not",
        "no",
        "yes",
        "just",
        "like",
        "also",
        "more",
        "most",
        "some",
        "any",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
        "get",
        "got",
        "all",
        "new",
    }
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_CODE_MARKERS = ("```", "traceback (most recent call last)", "syntaxerror")
_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |import |from \S+ import|@\w+|#include|public |private |func "
    r"|\}|\{|;\s*$|=>|->\s*\w|\$\s*\w|at \S+\(.*\)|file \"[^\"]+\", line \d+)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------- skipping


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _word_count(text: str) -> int:
    return len(text.split())


def looks_like_code(prompt: str) -> bool:
    """A cheap heuristic: fenced blocks, tracebacks, or lines that read like source."""
    lowered = prompt.lower()
    if any(marker in lowered for marker in _CODE_MARKERS):
        return True
    lines = [ln for ln in prompt.splitlines() if ln.strip()]
    if len(lines) > 1:
        code_lines = sum(1 for ln in lines if _CODE_LINE_RE.search(ln))
        if code_lines >= 2 or code_lines >= max(1, len(lines) // 2):
            return True
    symbols = sum(prompt.count(c) for c in "{}[]<>;")
    words = _word_count(prompt)
    return words > 0 and symbols >= 4 and symbols >= words


def is_skippable(prompt: str) -> bool:
    """Slash commands, short prompts, and code-ish pastes have nothing to route to."""
    stripped = prompt.strip()
    if not stripped or stripped.startswith("/"):
        return True
    if _word_count(stripped) < MIN_WORDS:
        return True
    return looks_like_code(stripped)


# ----------------------------------------------------------------------------- vocabulary


@dataclass(frozen=True)
class VocabTerm:
    """One entry a prompt can match: a single word, or a phrase matched as a substring."""

    text: str
    weight: int
    identity: bool
    phrase: bool


def _normalized_words(raw: str) -> list[str]:
    return [w for w in _tokens(raw) if len(w) >= MIN_TOKEN_LEN and w not in STOPWORDS]


def build_vocabulary(persona: project.PersonaInfo, topics: Sequence[str]) -> list[VocabTerm]:
    """Id and name tokens (weight 3, identity), tags (2), top topics (1).

    Multi-word tags and topics are kept as phrases (matched as a substring of the prompt with
    word boundaries); single words are matched by membership in the prompt's token set. The
    first source to claim a word wins, so an id/name token never gets diluted to weight 1 by
    also showing up as a topic.
    """
    seen: dict[str, VocabTerm] = {}

    def add_words(raw: str, weight: int, *, identity: bool) -> None:
        for word in _normalized_words(raw):
            seen.setdefault(word, VocabTerm(word, weight, identity, phrase=False))

    def add_phrase(raw: str, weight: int, *, identity: bool) -> None:
        words = _normalized_words(raw)
        if not words:
            return
        text = " ".join(words)
        seen.setdefault(text, VocabTerm(text, weight, identity, phrase=len(words) > 1))

    add_words(persona.id, ID_NAME_WEIGHT, identity=True)
    add_words(persona.name, ID_NAME_WEIGHT, identity=True)
    for tag in persona.tags:
        add_phrase(tag, TAG_WEIGHT, identity=False)
    for topic in topics:
        add_phrase(topic, TOPIC_WEIGHT, identity=False)
    return list(seen.values())


# ----------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class PersonaMatch:
    persona: project.PersonaInfo
    score: int
    matched: tuple[str, ...]
    identity_hit: bool

    @property
    def qualifies(self) -> bool:
        return self.identity_hit or self.score >= MIN_SCORE


def _matches(term: VocabTerm, prompt_word_set: frozenset[str], padded_prompt: str) -> bool:
    if term.phrase:
        return f" {term.text} " in padded_prompt
    return term.text in prompt_word_set


def score_persona(
    persona: project.PersonaInfo,
    vocab: Sequence[VocabTerm],
    prompt_word_set: frozenset[str],
    padded_prompt: str,
) -> PersonaMatch:
    """Sum the weight of every vocabulary term the prompt matches."""
    matched: list[str] = []
    score = 0
    identity_hit = False
    for term in vocab:
        if _matches(term, prompt_word_set, padded_prompt):
            score += term.weight
            matched.append(term.text)
            identity_hit = identity_hit or term.identity
    return PersonaMatch(
        persona=persona, score=score, matched=tuple(matched), identity_hit=identity_hit
    )


# ----------------------------------------------------------------------------- topics cache


def _topics_cache_path(root: Path, persona_id: str) -> Path:
    return project.cache_dir(root) / f"topics-{persona_id}.json"


def _read_topics_cache(path: Path, now: float) -> tuple[list[str] | None, list[str] | None]:
    """``(fresh, stale)``: ``fresh`` is set only when the cache is inside the TTL."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    fetched_at = data.get("fetched_at")
    raw_topics = data.get("topics")
    if not isinstance(fetched_at, (int, float)) or not isinstance(raw_topics, list):
        return None, None
    topics = [t for t in raw_topics if isinstance(t, str)]
    if now - float(fetched_at) <= TOPIC_CACHE_TTL:
        return topics, topics
    return None, topics


def _write_topics_cache(path: Path, topics: list[str], now: float) -> None:
    with contextlib.suppress(OSError):
        path.write_text(json.dumps({"fetched_at": now, "topics": topics}), encoding="utf-8")


def _topic_names(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    names: list[str] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("topic"), str) and item["topic"]:
            names.append(item["topic"])
        elif isinstance(item, str) and item:
            names.append(item)
    return names


def _topics_for(persona_id: str, root: Path, client: McpClient | None, now: float) -> list[str]:
    """Top topics for a persona, cached for a day; a live miss falls back to a stale cache."""
    cache_path = _topics_cache_path(root, persona_id)
    fresh, stale = _read_topics_cache(cache_path, now)
    if fresh is not None:
        return fresh
    if client is not None:
        try:
            raw = client.call_tool("topics", {"persona_id": persona_id, "limit": TOPICS_LIMIT})
        except McpUnavailable:
            raw = None
        if raw is not None:
            topics = _topic_names(raw)
            _write_topics_cache(cache_path, topics, now)
            return topics
    return stale or []


# ----------------------------------------------------------------------------- search


def _titles_for(persona_id: str, prompt: str, client: McpClient | None) -> list[str]:
    if client is None:
        return []
    try:
        raw = client.call_tool("search", {"query": prompt, "persona_id": persona_id, "k": 5})
    except McpUnavailable:
        return []
    if not isinstance(raw, list):
        return []
    titles: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if isinstance(title, str) and title and title not in titles:
            titles.append(title)
        if len(titles) >= MAX_TITLES:
            break
    return titles


# ----------------------------------------------------------------------------- rendering


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def render_persona(
    persona: project.PersonaInfo,
    manifest: project.Manifest | None,
    match: PersonaMatch,
    titles: Sequence[str],
) -> str:
    """One paragraph, under 700 characters, naming the persona, the matches, and doc titles."""
    matched_display = ", ".join(match.matched[:MAX_MATCHED_SHOWN]) or "its focus"
    doc_count = manifest.document_count if manifest is not None else 0
    text = (
        f"This prompt looks like a {persona.name} question (matched: {matched_display}). "
        f"Before answering, call context(query, persona_id='{persona.id}') on the graphrag "
        f"MCP server; {doc_count} documents are indexed."
    )
    if titles:
        shown = [_shorten(t, MAX_TITLE_LEN) for t in titles[:MAX_TITLES]]
        text += f" Possibly relevant: {'; '.join(shown)}."
    return _shorten(text, MAX_PARAGRAPH)


# ----------------------------------------------------------------------------- orchestration


def route(root: Path, prompt: str, client: McpClient | None, now: float) -> str:
    """The whole pure pipeline: skip check, vocabulary, scoring, search, rendering."""
    if is_skippable(prompt):
        return ""
    personas = project.load_personas(root)
    if not personas:
        return ""
    manifests = project.load_manifests(root)
    prompt_word_set = frozenset(_tokens(prompt))
    padded_prompt = " " + " ".join(_tokens(prompt)) + " "

    candidates: list[PersonaMatch] = []
    for persona in sorted(personas.values(), key=lambda p: p.id):
        topics = _topics_for(persona.id, root, client, now)
        vocab = build_vocabulary(persona, topics)
        match = score_persona(persona, vocab, prompt_word_set, padded_prompt)
        if match.qualifies:
            candidates.append(match)

    if not candidates:
        return ""
    candidates.sort(key=lambda m: (-m.score, m.persona.id))
    chosen = candidates[:MAX_PERSONAS]

    paragraphs = [
        render_persona(
            m.persona, manifests.get(m.persona.id), m, _titles_for(m.persona.id, prompt, client)
        )
        for m in chosen
    ]
    return "\n\n".join(paragraphs)


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
    """Read the UserPromptSubmit stdin payload, print a routing nudge, and always return 0."""
    try:
        payload = _read_stdin_payload()
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return 0
        cwd = payload.get("cwd")
        start = Path(cwd) if isinstance(cwd, str) and cwd else None
        root = project.find_root(start)
        if root is None:
            return 0
        client = connect()
        output = route(root, prompt, client, time.time())
        if output:
            print(output)
    except Exception:  # a routing nudge must never break a prompt submission
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
