"""Small text helpers with no third-party dependencies.

Kept import-light on purpose: ``graphrag.hooks`` runs on the host with a bare ``python3``
and the standard library only, so anything it needs must live outside the pydantic models.
"""

from __future__ import annotations

import re

__all__ = ["entity_slug", "fold_name", "sanitize_inline", "slugify"]

_NON_SLUG = re.compile(r"[^a-z0-9]+")
#: Punctuation that carries meaning inside a name, spelled out before the slug drops it.
#: Space-padded so each becomes its own word: ``C++`` reads as ``c-plus-plus``, not
#: ``cplusplus``. Additions are not free -- they change the id of every entity holding the
#: character, and there is no migration -- so this stays the short list of characters that
#: actually distinguish one product or language from another.
_ID_WORDS = {"+": " plus ", "&": " and ", "#": " sharp "}
_ID_PUNCT = re.compile("[" + re.escape("".join(_ID_WORDS)) + "]")

# Zero-width joiners/marks, the bidirectional overrides, the invisible-maths operators and the
# byte-order mark. All of them can render as nothing while changing what a human reads, so they
# have no business in a string a hook splices into a model's instruction slot.
_HIDDEN = re.compile("[​-‏‪-‮⁠-⁤﻿]")
# C0 and C1 controls, minus the five whitespace ones (tab, newline, vertical tab, form feed,
# carriage return) which ``_WHITESPACE`` folds into ordinary spaces instead of deleting.
_CONTROL = re.compile("[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_WHITESPACE = re.compile(r"\s+")

ELLIPSIS = "…"


def slugify(value: str) -> str:
    """Lower-case, ASCII-ish slug safe for ids, folder names and Cypher parameters."""
    value = value.strip().lower()
    value = _NON_SLUG.sub("-", value)
    return value.strip("-") or "untitled"


def entity_slug(value: str) -> str:
    """Slug for an entity id, keeping the punctuation that tells two names apart.

    :func:`slugify` drops every non-alphanumeric character, so ``Acme+`` and ``Acme``, ``C++``
    and ``C``, ``A&B`` and ``AB`` all land on one slug -- and therefore on one entity id, where
    the second import of the pair used to rename the first one's node and swallow its mentions.

    The rule: before slugifying, spell out the three characters that carry meaning inside a
    name, each padded with spaces so it becomes a word of its own::

        Acme+ -> acme-plus       C++ -> c-plus-plus       A&B -> a-and-b
        Acme  -> acme            C   -> c                 AB  -> ab

    Deterministic and context-free: the same name always gives the same slug, whatever else is
    in the graph. Other punctuation still folds away -- ``Acme!`` and ``Acme`` share a slug --
    which is why ``upsert_enrichment`` refuses to rename an existing node when an incoming name
    lands on its id, and reports the collision instead of resolving it.

    :func:`slugify` itself is deliberately untouched: document, topic and persona ids are built
    from its exact output and are committed in snapshots.
    """
    return slugify(_ID_PUNCT.sub(lambda m: _ID_WORDS[m.group()], value))


def fold_name(value: str) -> str:
    """The form two spellings are compared in: lower case, with runs of whitespace collapsed."""
    return " ".join(value.split()).casefold()


def sanitize_inline(text: str, limit: int) -> str:
    """Flatten untrusted text to one short, printable line.

    For any string that came out of the graph or off the filesystem and is about to be
    interpolated into text a model reads as instructions: a document title, a topic name, a
    file name. Strips control characters and invisible/bidi code points, folds every run of
    whitespace into a single space, trims, and caps the result at ``limit`` characters with a
    trailing ellipsis. The result is always at most ``limit`` characters and contains no line
    breaks, so it cannot escape the sentence that hosts it.

    This is a formatting guard, not a content filter: a title that reads like an instruction
    survives as words. Callers quote it and label it so the model knows it is a name.
    """
    if limit <= 0:
        return ""
    text = _HIDDEN.sub("", text)
    text = _CONTROL.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + ELLIPSIS
