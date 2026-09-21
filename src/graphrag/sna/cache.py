"""An on-disk cache of built networks (entitlement ATL-ENT-3).

The same command, run twice with the same filters against the same snapshot, should not re-run
the same Cypher and rebuild the same graph a second time: :func:`graphrag.sna.export.build_network`
is exactly the read that costs the most on the largest persona (13k chunks), and every command in
`cli.py` that builds a network calls it once per invocation. This module memoises that call on
disk, keyed by everything that could make it return a different graph. It prints nothing and
raises nothing the caller did not already risk; the report a caller prints is free to say whether
the graph it got was a hit or a miss.

**What goes in the key.** The persona, the network kind, and every filter
:func:`~graphrag.sna.export.build_network` was called with -- the whole ``**filters`` mapping,
not a curated subset, so a filter this module has never heard of still busts the cache rather
than silently serving a stale graph the next one introduces. Beside those sits two different
readings of "has the data changed": the committed snapshot's own identity
(:func:`snapshot_identity`) and a cheap fingerprint read straight from the store
(:func:`~graphrag.graph.store.GraphStore.persona_fingerprint`). Neither is redundant with the
other, and neither is redundant with the explicit invalidation below:

* **Explicit, exact, and free of false misses**: every command in `cli.py` that writes to a
  persona's graph -- ingest, sync, enrich, the sidecar importers, a snapshot load, an alias
  merge, an orphan prune -- clears that persona's entries as its last step (grep `cli.py` for
  ``NetworkCache(ctx.settings.sna_cache_dir).clear(``). This is the fast path: it costs nothing
  on a read, and a persona nothing wrote to keeps every entry it had.
* **The fingerprint, as a second line of defence**: a write this module has no hook for -- Cypher
  run by hand against Neo4j, a script that imports `graphrag.graph.neo4j_store` directly, a test
  that mutates a store without going through the CLI -- leaves no `clear()` call to catch it.
  `persona_fingerprint` is read on *every* build, hit or miss, and folded into the key, so a
  changed document, chunk or mention count, or a changed count of nodes carrying an attribute
  property (a property-only reimport that adds or corrects a value without adding a document or
  a mention), still misses even when nothing ever called `clear()`. Measured at ~0.1-0.6s on this
  repository's largest persona (13k chunks) -- see `tests/benchmarks/test_sna_budget.py` -- which
  is cheap next to the build it is deciding whether to skip.
* **What neither catches**: a write that changes *values* without changing any of the four counts
  the fingerprint reads (an attribute correction to a key a node already carried a value for,
  say) is invisible to both defences until the persona's cache is cleared some other way --
  `graphrag sna cache clear`, `--no-cache`, or the persona's next write that does move a count.

**What does not go in the key.** The ``GraphStore`` object itself: two different stores that
answer to the same persona id, filters and identity get the same cached graph. That is correct
for this repository's own arrangement, where a persona's Neo4j data *is* what the committed
snapshot holds (CLAUDE.md: "the graph snapshot is committed ... so teammates load rather than
rebuild") -- and it is exactly why ``--no-cache`` exists on every command that calls
:func:`build_network_cached`, for whoever is not in that situation (a scratch store built for a
one-off test, say, that shares a persona id with something real).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from graphrag.config import Settings
from graphrag.graph.snapshot import SnapshotError, read_manifest, snapshot_dir
from graphrag.graph.store import GraphStore
from graphrag.sna.export import build_network

#: The cache's own on-disk format, bumped when this module changes what it writes (not when a
#: network's shape changes) -- so an old version's files are treated as a miss instead of being
#: unpickled and misread.
CACHE_VERSION = 1

#: What a persona with no committed snapshot hashes to: no `source_commit` and no manifest to
#: hash, but that should not turn caching off, only stop it from tracking the one thing it has
#: nothing to compare -- the filters and the persona id still drive the key.
NO_SNAPSHOT_IDENTITY = "no-snapshot"

#: The pickle a `.get` could not read: truncated by a crash mid-write, from a future format this
#: version does not know, or simply not a graph. Every one of these is "rebuild it", not "raise".
_UNREADABLE: tuple[type[Exception], ...] = (
    pickle.UnpicklingError,
    EOFError,
    AttributeError,
    ImportError,
    ValueError,
)


def snapshot_identity(snapshots_dir: Path, persona_id: str) -> str:
    """The identity a cache entry is invalidated against: this persona's committed snapshot.

    ``manifest.source_commit`` when the snapshot's own export recorded one -- set by
    ``graphrag snapshot export`` from the repository's ``HEAD`` at export time
    (:func:`graphrag.cli._git_commit`), so this is literally "keyed by ... snapshot commit" in
    the ticket's own words. A snapshot exported before a commit was recorded, or built by a
    caller that never passed one, hashes the whole manifest instead (model, dims, the document
    and chunk counts, ``created_at``) so a re-export of the same persona still changes the key
    even without a commit to point at.

    :data:`NO_SNAPSHOT_IDENTITY` for a persona with no manifest on disk at all -- an in-memory
    store built for a test, or a persona still being ingested and never exported. Caching still
    works there, keyed on the persona, the network and the filters alone; what it cannot do is
    notice that the underlying data changed without a new export, which is the one thing this
    function reads to know.
    """
    try:
        manifest = read_manifest(snapshot_dir(snapshots_dir, persona_id))
    except SnapshotError:
        return NO_SNAPSHOT_IDENTITY
    if manifest.source_commit:
        return f"commit:{manifest.source_commit}"
    digest = hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()
    return f"manifest:{digest[:16]}"


def _normalise(value: Any) -> Any:
    """A filter value as something ``json.dumps(..., sort_keys=True)`` renders identically
    regardless of which collection type a caller happened to pass.

    ``build_network``'s filters arrive as lists, tuples or a plain ``dict`` depending on which
    CLI option produced them; two calls that mean the same filter must hash to the same key
    however the caller's own code built the argument, or the cache misses on a distinction that
    is not one the graph reflects.
    """
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalise(item) for item in value]
    return value


def cache_key(persona_id: str, network: str, identity: str, filters: Mapping[str, Any]) -> str:
    """The cache key for one ``build_network`` call: a hash of everything that could change its
    answer.

    Two calls collide only when the persona, the network kind, every filter and the snapshot
    identity all agree -- exactly the four things :func:`build_network_cached`'s docstring names.
    A ``sha256`` over the JSON rather than the tuple itself, so the key is a fixed-length string
    safe to use as a filename regardless of what a filter value happens to contain.
    """
    payload = {
        "cache_version": CACHE_VERSION,
        "persona": persona_id,
        "network": network,
        "snapshot": identity,
        "filters": _normalise(dict(filters)),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class NetworkCache:
    """One directory of cached networks, one subdirectory per persona, each entry a pickled
    ``networkx`` graph named by its key.

    Pickle, not GraphML: a built network carries Python values GraphML cannot round-trip exactly
    (``None``, mixed-type node attributes, the tuples some report payloads leave on an edge), and
    this cache only ever reads back what it itself wrote, so GraphML's portability buys nothing
    here that pickling every attribute exactly does not already give for free. A cache entry is
    never read from, or written to, anywhere but this class, which is what makes trusting the
    pickle safe: nothing outside this process's own prior run ever populates the directory.

    Partitioned by persona on disk, not only in the key's own hash, so that :meth:`clear` can
    empty one persona's entries -- what every write path in `cli.py` calls -- without listing and
    re-hashing every file in the directory to find the ones that belong to it.
    """

    directory: Path

    def _path(self, persona_id: str, key: str) -> Path:
        return self.directory / persona_id / f"{key}.pickle"

    def get(self, persona_id: str, key: str) -> nx.Graph | None:
        """The cached graph for ``persona_id``/``key``, or ``None`` on a miss -- including a
        file that exists but will not unpickle, which is treated as though it were never
        written."""
        path = self._path(persona_id, key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                graph = pickle.load(handle)  # noqa: S301 -- this class wrote every file it reads
        except _UNREADABLE:
            return None
        return graph if isinstance(graph, nx.Graph) else None

    def put(self, persona_id: str, key: str, graph: nx.Graph) -> None:
        """Write ``graph`` under ``persona_id``/``key``, atomically: a reader never sees a
        half-written file.

        Each write has a temp file of its own, named for the process and a random suffix, so two
        commands building the same network at once no longer write to one shared temp path and
        race to rename it (the second rename used to fail with ``FileNotFoundError``). Both
        renames now land, the last one winning with an identical graph. A ``clear()`` that
        sweeps this write's temp file away mid-flight loses the write and nothing else: the cache
        is an optimisation, and the next build writes it again.
        """
        path = self._path(persona_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}-{secrets.token_hex(4)}.pickle.tmp")
        try:
            with tmp.open("wb") as handle:
                pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
        except FileNotFoundError:
            tmp.unlink(missing_ok=True)

    def clear(self, persona_id: str | None = None) -> int:
        """Delete cached networks: every persona's when ``persona_id`` is ``None`` (what
        ``graphrag sna cache clear`` does), or just one persona's (what every write path in
        `cli.py` does as its last step). Returns how many files that was; clearing a persona
        with nothing cached, or the whole directory when it does not exist yet, is not an error.
        """
        root = self.directory / persona_id if persona_id is not None else self.directory
        if not root.exists():
            return 0
        removed = 0
        for path in root.rglob("*.pickle"):
            try:
                path.unlink()
            except FileNotFoundError:
                continue  # another process's clear() got there first
            removed += 1
        for stray in root.rglob("*.pickle.tmp"):
            stray.unlink(missing_ok=True)
        return removed


def _identity(snapshots_dir: Path, store: GraphStore, persona_id: str) -> str:
    """Everything the key reads to notice the persona's data changed, combined into one string:
    the committed snapshot's own identity (:func:`snapshot_identity`) and a live fingerprint
    (``store.persona_fingerprint``) read fresh on every call. See the module docstring for why
    both -- and why neither is a substitute for the explicit ``clear()`` calls in `cli.py`.
    """
    documents, chunks, mentions, attributed = store.persona_fingerprint(persona_id)
    return (
        f"{snapshot_identity(snapshots_dir, persona_id)}"
        f"|fingerprint:{documents}-{chunks}-{mentions}-{attributed}"
    )


def build_network_cached(
    store: GraphStore,
    network: str,
    persona_id: str,
    *,
    settings: Settings,
    no_cache: bool = False,
    **filters: Any,
) -> nx.Graph:
    """:func:`graphrag.sna.export.build_network`, through the on-disk cache.

    Same dispatch, same filters, same graph back -- every positional and keyword argument
    ``build_network`` takes is accepted here as a keyword and forwarded unchanged -- except that
    a hit is served from ``settings.sna_cache_dir`` without touching ``store`` for the network
    itself, and a miss is written there after being built. Every command in `cli.py` that builds
    a network calls this instead of ``build_network`` directly, which is what makes ``--no-cache``
    one flag defined once (:data:`graphrag.cli.NO_CACHE_OPTION`) rather than reimplemented per
    command, and what makes every write path's ``clear()`` call the one place invalidation lives.

    Caching is off entirely -- every call falls straight through to ``build_network``, reading
    nothing but the network itself -- when ``settings.sna_cache`` is ``False`` or ``no_cache`` is
    ``True`` for this one call; neither path reads or writes the cache directory, and neither
    reads ``store.persona_fingerprint``.
    """
    if not settings.sna_cache or no_cache:
        return build_network(store, network, persona_id, **filters)
    cache = NetworkCache(settings.sna_cache_dir)
    identity = _identity(settings.snapshots_dir, store, persona_id)
    key = cache_key(persona_id, network, identity, filters)
    cached = cache.get(persona_id, key)
    if cached is not None:
        return cached
    graph = build_network(store, network, persona_id, **filters)
    cache.put(persona_id, key, graph)
    return graph


__all__: Sequence[str] = (
    "CACHE_VERSION",
    "NO_SNAPSHOT_IDENTITY",
    "NetworkCache",
    "build_network_cached",
    "cache_key",
    "snapshot_identity",
)
