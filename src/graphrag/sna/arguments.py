"""Filter arguments the CLI and the MCP tools parse into one canonical shape (ATL-F1).

The CLI takes `--types a,b` as one comma string and `--where key=value` repeated; an MCP tool
caller already holds a list and a mapping, because a tool call is JSON, not a shell line. Before
this module each surface recorded its own raw form in :func:`graphrag.sna.provenance.
make_provenance`'s ``parameters``, so the same query run through the CLI and through a tool
disagreed about what `types`/`where`/a sampler's `--param` even *were* -- a string here, a list
there; a list of ``"k=v"`` strings here, a dict there -- even though both built the identical
network. The parsers below take either surface's native input and return the one shape every
caller then uses both to build the network *and* to record in `parameters`, so a report's
provenance no longer depends on which surface asked for it.

Every function here raises `ValueError` on malformed input rather than exiting or returning a
structured error -- neither belongs in a module with no `typer`/`fastmcp` import. `cli.py`
catches it and turns it into `err.print` + `typer.Exit(2)`; `mcp_server.py` catches it and turns
it into `{"error": ...}`, the same split `_parse_k_range`/`_parse_seeds` already use for the same
reason (see `mcp_server.py`'s module-level comment on those two).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "DEFAULT_MAX_SECONDS",
    "parse_max_seconds",
    "parse_sample_params",
    "parse_types",
    "parse_where",
]

DEFAULT_MAX_SECONDS = 300.0
"""The wall-clock budget `sna analyze --uncertain` and the `sna_analyze` tool give the uncertain
section when the caller names none (ATL-F2's `--max-seconds`). Without one, fifty realisations on
a 1,300-node entity network ran past fifteen minutes, so the unbounded
default was a hang rather than a choice. The report already prints the budget and how many
realisations each measure got inside it, so a run the budget cut short says so beside its
intervals. `0` asks for no budget. The library functions (`run_analysis`,
`graphrag.sna.uncertain.uncertainty_report`) keep `None` as their default: a caller of those has
chosen its own limits, and this default belongs to the two surfaces a person types at."""


def parse_max_seconds(value: float | None) -> float | None:
    """The budget to run with: ``None`` (no budget) for ``0``, a negative value or ``None``."""
    return value if value is not None and value > 0 else None


def parse_types(value: str | Sequence[str] | None) -> list[str] | None:
    """Entity types into the canonical ``list[str]`` every network builder takes.

    The CLI's ``--network entities --types a,b`` is one comma-separated string; an MCP tool's
    ``types`` argument is already a list. Both go through here: blanks are dropped either way, so
    a CLI ``--types "a, ,b"`` and a tool's ``types=["a", "", "b"]`` both become ``["a", "b"]`` and
    therefore record the identical `parameters["types"]` no matter which surface asked.
    """
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else value
    cleaned = [item.strip() for item in items if item.strip()]
    return cleaned or None


def parse_where(
    value: Sequence[str] | Mapping[str, str] | None, option: str = "--where"
) -> dict[str, str] | None:
    """``--where key=value`` (repeatable) or an MCP ``dict``, into the canonical ``dict[str, str]``.

    A key given twice is refused rather than resolved -- a node holds one value per key, so
    ``--where region=north --where region=south`` can only ever match nothing, and silently
    keeping the last one would answer a question nobody asked. A mapping already has unique keys
    by construction, so that check only fires for the CLI's repeated-string form; an empty
    mapping or an empty list of strings both collapse to ``None``, matching each other and every
    filter this module returns for "nothing was asked for".

    ``option`` names the flag in a refusal message (``--where`` or ``--where2``, for `sna
    compare`'s second population) -- purely cosmetic, never read back afterward.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        cleaned = {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip()}
        return cleaned or None
    where: dict[str, str] = {}
    for item in value:
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or not key or not raw:
            raise ValueError(f"{option} must look like key=value, got {item!r}")
        if key in where:
            raise ValueError(
                f"{option} {key} given twice; a node holds one value per key, so no node could "
                "match both"
            )
        where[key] = raw
    return where or None


def parse_sample_params(value: Sequence[str] | Mapping[str, Any] | None) -> dict[str, Any]:
    """``--param name=value`` (repeatable) or an MCP ``dict``, into the sampler's own parameters.

    ``true``/``false`` become booleans, digits become numbers, everything else stays a string;
    the sampler itself refuses a name it does not take, which is what turns a typo into a
    refusal rather than a sample taken by a method nobody asked for. An MCP caller's mapping is
    already in this shape and is returned as a plain ``dict`` copy, unexamined -- coercing an
    already-typed ``bool``/``int``/``float`` back through string parsing would be the one way to
    *lose* information the CLI's own ``"true"``/``"3"`` strings never carried in the first place.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    params: dict[str, Any] = {}
    for item in value:
        key, sep, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if not sep or not key or not raw:
            raise ValueError(f"--param must look like name=value, got {item!r}")
        if raw.lower() in {"true", "false"}:
            params[key] = raw.lower() == "true"
        else:
            try:
                params[key] = int(raw) if raw.lstrip("-").isdigit() else float(raw)
            except ValueError:
                params[key] = raw
    return params
