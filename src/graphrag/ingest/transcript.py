"""Loader for timestamped speaker-turn transcripts (Lenny's Podcast archive format).

File layout::

    ---
    guest: Bob Moesta
    title: ...
    youtube_url: https://www.youtube.com/watch?v=...
    publish_date: 2025-02-23
    keywords: [growth, retention]
    ---
    # Title
    ## Transcript

    Bob Moesta (00:00:00):
    paragraph text...

    Lenny Rachitsky (00:01:12):
    ...
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from graphrag.ingest.frontmatter import split_frontmatter
from graphrag.models import Document, LoadedDocument, Scalar, Turn
from graphrag.textutil import slugify

TURN_HEADER = re.compile(r"^(?P<speaker>[^\n(]{1,80}?)\s\((?P<ts>\d{1,2}:\d{2}:\d{2})\):\s*$")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s")


PERSON_NAME = re.compile(r"^[A-Z][\w'.\-]+( [A-Z][\w'.\-]+){1,3}$")
# "... | First Last (Company)" / "... | First Last" credits at the end of archive titles
TITLE_NAME_SEGMENT = re.compile(
    r"\|\s*(?P<name>[A-Z][\w'.\-]+(?: [A-Z][\w'.\-]+){1,3})\s*(?:[(,]|$)"
)
NAME_STOPWORDS = {"jr", "sr", "dr", "live", "the", "and"}


def _fold(text: str) -> str:
    """Lower-case and strip accents so "Lütke" matches "Lutke"."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(ch)
    )


def guest_names(guest: str) -> list[str]:
    """Split "A, B and C 2.0" into ["A", "B", "C"], dropping return-visit suffixes."""
    cleaned = re.sub(r"\s+\d+\.\d+$", "", guest.strip())
    return [n.strip() for n in re.split(r",|\+|&|\band\b", cleaned) if n.strip()]


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", _fold(name)) if t not in NAME_STOPWORDS]


def _token_matches(token: str, candidates: list[str]) -> bool:
    """Exact, prefix (truncated folder names: "shriva"/"shrivastava") or fuzzy (transcription
    spellings: "hardimen"/"hardiman") match of one name token against credit tokens."""
    return any(
        c == token
        or c.startswith(token)
        or token.startswith(c)
        or difflib.SequenceMatcher(None, token, c).ratio() >= 0.8
        for c in candidates
    )


def title_mentions_guest(title: str, guest: str) -> bool:
    """False only when the title carries a "| Person Name" credit for someone who is not the
    guest: that pattern means the archive paired another episode's title with this file.
    Titles without a person credit (acronyms like "HackAPrompt CEO" are not one), and
    non-person guests, always count as mentioned. A guest matches a credit when every token of
    the guest's name matches a credit token, so "Kim Scott" is not "Scott Wu"."""
    names = [n for n in guest_names(guest) if PERSON_NAME.match(n)]
    name_credits = [
        m.group("name")
        for m in TITLE_NAME_SEGMENT.finditer(title)
        if not re.search(r"\b[A-Z]{2,}\b", m.group("name"))
    ]
    if not names or not name_credits:
        return True
    credit_tokens = [t for c in name_credits for t in _name_tokens(c)]
    for name in names:
        tokens = _name_tokens(name)
        if tokens and all(_token_matches(t, credit_tokens) for t in tokens):
            return True
    return False


def timestamp_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, seconds = parts[-3:]
    return hours * 3600 + minutes * 60 + seconds


def parse_turns(body: str) -> list[Turn]:
    """Parse ``Speaker (HH:MM:SS):`` blocks into turns; untimed prose becomes speaker-less turns."""
    turns: list[Turn] = []
    speaker: str | None = None
    ts: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            turns.append(
                Turn(
                    text=text,
                    speaker=speaker,
                    start_ts=ts,
                    start_seconds=timestamp_to_seconds(ts) if ts else None,
                )
            )
        buffer.clear()

    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        match = TURN_HEADER.match(line)
        if match:
            flush()
            speaker = match.group("speaker").strip()
            ts = match.group("ts")
            continue
        if MARKDOWN_HEADING.match(line):
            continue
        buffer.append(line)
    flush()
    return turns


def _scalar(value: Any) -> Scalar:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _parse_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def load_transcript(path: Path, *, root: Path, persona_id: str, source_id: str) -> LoadedDocument:
    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    turns = parse_turns(body)

    rel = path.relative_to(root)
    folder = rel.parent.name or rel.stem
    doc_slug = slugify(folder if folder not in ("", ".") else rel.stem)
    doc_id = f"{persona_id}:{source_id}:{doc_slug}"

    guest = str(meta.get("guest") or "").strip()
    title = str(meta.get("title") or guest or rel.stem).strip()
    if guest and not title_mentions_guest(title, guest):
        # The archive sometimes pairs a guest with another episode's title; the guest field
        # and the body are reliable, the title is not. Keep the original for reference.
        meta["archive_title"] = title
        title = f"{guest} on {meta.get('channel') or 'the podcast'}"
    url = meta.get("youtube_url")
    keywords_raw = meta.get("keywords") or []
    topics = sorted({slugify(str(k)).replace("-", " ") for k in keywords_raw if str(k).strip()})

    speakers: list[str] = []
    for turn in turns:
        if turn.speaker and turn.speaker not in speakers:
            speakers.append(turn.speaker)
    if guest and guest not in speakers:
        speakers.insert(0, guest)

    passthrough = (
        "video_id",
        "duration",
        "duration_seconds",
        "view_count",
        "channel",
        "guest",
        "archive_title",
    )
    metadata: dict[str, Scalar] = {k: _scalar(meta[k]) for k in passthrough if k in meta}

    document = Document(
        id=doc_id,
        persona_id=persona_id,
        source_id=source_id,
        title=title,
        path=str(rel),
        url=str(url) if url else None,
        published=_parse_date(meta.get("publish_date")),
        description=str(meta.get("description") or "").strip(),
        speakers=speakers,
        topics=topics,
        metadata=metadata,
        word_count=sum(len(t.text.split()) for t in turns),
    )
    return LoadedDocument(document=document, turns=turns)


def deep_link(url: str | None, start_seconds: int | None) -> str | None:
    """YouTube deep link to a timestamp; other URLs are returned unchanged."""
    if not url:
        return None
    if start_seconds is None or ("youtube.com" not in url and "youtu.be" not in url):
        return url
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}t={start_seconds}s"
