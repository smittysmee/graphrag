"""The properties a persona's nodes can carry, and the vocabulary that keeps them honest.

Extraction says what a passage names, attribution says who wrote it and annotation says what it
says. None of them says what *kind* of node this is: which part of a market a speaker sells in,
which organisation they say they work for, which half of a corpus a document belongs to. Without
that, "do these two populations form two networks" cannot be measured at all -- it can only be
counted by hand, once, by whoever happens to know the corpus.

An attribute fills that in. It is one short key and one short value, written by the same agent
that writes the other sidecars: ``{"region": "north"}`` on a post, so the speaker carries it, or
on an annotation file, so the document does. Nothing here infers one; an attribute is a claim the
document supports, and a tagger that cannot point at the sentence should leave the key out.

The vocabulary lives beside the facet vocabulary, in ``personas/<persona>/facets.yaml``::

    attributes:
      region:
        values: [north, south]
        description: Which half of the network the speaker belongs to.
      team:
        description: The team a speaker states they are on, as written.

A key with ``values`` is closed: a value outside the list is reported and dropped, per entry, so
one invented value costs that entry rather than the file. A key without ``values`` is free text,
which is the right shape for something like an organisation name that nobody can enumerate in
advance. When the file declares no ``attributes:`` section at all, the table is empty and every
key and value is accepted -- the same rule the facet table uses, and for the same reason: an
absent vocabulary means the persona has not declared one yet, not that it declared nothing.

The table is read once per run and handed to the importers, which is why it lives here rather
than inside either of them: ``graphrag.extract.attribution`` and ``graphrag.extract.annotations``
both validate against it, and the annotation module already imports the attribution one.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = [
    "ATTRIBUTES_KEY",
    "EMPTY_ATTRIBUTES",
    "KEY_RE",
    "VOCABULARY_FILE",
    "AttributeSpec",
    "AttributeTable",
    "AttributeTableError",
    "load_attributes",
    "load_persona_attributes",
]

#: Where a persona declares what its corpus talks about: the facet vocabulary and this one.
VOCABULARY_FILE = "facets.yaml"

#: The section of that file this module reads.
ATTRIBUTES_KEY = "attributes"

#: What an attribute key may look like. Kept narrow on purpose: a key becomes a column in every
#: report and a property name in the graph, so a space or a capital in one file and not in the
#: next would silently make two attributes out of one.
KEY_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class AttributeTableError(ValueError):
    """The vocabulary file is unreadable or malformed."""


@dataclass(frozen=True)
class AttributeSpec:
    """One declared attribute key: what it means, and what it is allowed to hold."""

    key: str
    values: tuple[str, ...] = ()
    description: str = ""

    @property
    def closed(self) -> bool:
        """Whether the key enumerates its values, so anything else is a mistake."""
        return bool(self.values)


@dataclass(frozen=True)
class AttributeTable:
    """The attribute keys a persona recognises, and the values each one allows.

    An empty table is not an empty vocabulary: it means the persona declared none, so every key
    and value is accepted. That is what :func:`load_persona_attributes` returns for a persona
    with no ``facets.yaml`` and for one whose file has no ``attributes:`` section, because the
    two say the same thing -- nobody has written the vocabulary down yet.
    """

    keys: Mapping[str, AttributeSpec] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.keys)

    def known(self, key: str) -> bool:
        return key in self.keys

    def spec(self, key: str) -> AttributeSpec | None:
        return self.keys.get(key)

    def check_attribute(self, key: str, value: str) -> str | None:
        """What is wrong with this pair, in words, or ``None`` when nothing is.

        The return is a problem string rather than an exception because one bad entry is not a
        bad file: the importers report it, drop that key and keep going, exactly as they do for
        a facet outside the declared vocabulary.
        """
        if not KEY_RE.match(key):
            return f"{key}: not a usable attribute key (lower case, digits, '-' and '_')"
        if not value.strip():
            return f"{key}: blank value"
        if not self.keys:
            return None
        spec = self.keys.get(key)
        if spec is None:
            return f"{key}: not declared in {VOCABULARY_FILE}"
        if spec.closed and value not in spec.values:
            return f"{key}={value}: not one of {', '.join(spec.values)}"
        return None

    def check(self, attributes: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
        """Split a sidecar's attributes into the ones to write and the problems to report."""
        kept: dict[str, str] = {}
        problems: list[str] = []
        for key, value in attributes.items():
            problem = self.check_attribute(key, value)
            if problem is None:
                kept[key] = value.strip()
            else:
                problems.append(problem)
        return kept, problems


#: What a persona that declares no vocabulary gets: anything goes.
EMPTY_ATTRIBUTES = AttributeTable()


def load_attributes(path: Path) -> AttributeTable:
    """Read the ``attributes:`` section of one ``facets.yaml``.

    A missing section is an empty table, not an error: the same file carries the facet
    vocabulary, and a persona that declares facets and no attributes is a normal persona.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"cannot read attribute vocabulary {path}: {exc}"
        raise AttributeTableError(msg) from exc
    if data is None:
        return EMPTY_ATTRIBUTES
    if not isinstance(data, Mapping):
        msg = f"{path}: expected a mapping, got {type(data).__name__}"
        raise AttributeTableError(msg)
    raw = data.get(ATTRIBUTES_KEY) or {}
    if not isinstance(raw, Mapping):
        msg = f"{path}: '{ATTRIBUTES_KEY}' must be a mapping of key to its declaration"
        raise AttributeTableError(msg)
    keys: dict[str, AttributeSpec] = {}
    for key, declaration in raw.items():
        name = str(key).strip()
        if not KEY_RE.match(name):
            msg = f"{path}: attribute key {key!r} must be lower case letters, digits, '-' or '_'"
            raise AttributeTableError(msg)
        keys[name] = _spec(path, name, declaration)
    return AttributeTable(keys=keys)


def _spec(path: Path, key: str, declaration: object) -> AttributeSpec:
    """One key's declaration: ``{values: [...], description: ...}``, or just a description."""
    if declaration is None:
        return AttributeSpec(key=key)
    if isinstance(declaration, str):
        return AttributeSpec(key=key, description=declaration.strip())
    if not isinstance(declaration, Mapping):
        msg = f"{path}: attribute {key!r} must be a mapping or a description string"
        raise AttributeTableError(msg)
    raw = declaration.get("values")
    if raw is None:
        values: tuple[str, ...] = ()
    elif isinstance(raw, list):
        values = tuple(str(v).strip() for v in raw if str(v).strip())
        if not values:
            msg = f"{path}: attribute {key!r} declares an empty 'values' list"
            raise AttributeTableError(msg)
    else:
        msg = f"{path}: attribute {key!r} must declare 'values' as a list"
        raise AttributeTableError(msg)
    return AttributeSpec(
        key=key, values=values, description=str(declaration.get("description") or "").strip()
    )


def load_persona_attributes(personas_dir: Path, persona_id: str) -> AttributeTable:
    """The persona's attribute vocabulary, or the empty one that accepts anything."""
    path = personas_dir / persona_id / VOCABULARY_FILE
    return load_attributes(path) if path.is_file() else EMPTY_ATTRIBUTES
