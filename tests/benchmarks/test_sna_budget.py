"""ATL-ENT-3's timing budget: how long the commands this ticket routed through sparse matrices
and the on-disk cache take on the largest persona this repository ships, on the machine running
the test, right now.

**Sampling frame.** Whichever persona's committed snapshot (`data/snapshots/<persona>/
manifest.json`) reports the most chunks -- 13k for `product-leader` at the time this ticket was
written -- read against the live store that snapshot was loaded into. Skipped, not failed, when
there is no snapshot to read or the store holds nothing for the persona it names: a machine that
has not run `make setup` (or a CI job with no Neo4j) has nothing to time, and a timing budget
that fails on an empty database would be testing the fixture, not the code.

**Null model: none.** A wall-clock budget is not a statistical claim -- there is no null a
build time is compared against, only a stated ceiling generous enough to hold on the reference
machine this ticket was built on (a commodity laptop, Docker Desktop, one worker) with headroom
for a slower one, and tight enough that a regression back to the dense path trips it. Every
number this test prints is followed by the budget it was held to, in the same table.

**What this is not.** A performance regression gate meant to run on every commit -- it needs a
live Neo4j with a persona actually loaded, which `make check`'s pytest invocation does not
provide, so it is marked ``slow`` and left out of the ``-m`` expression there (see the Makefile).
Run it explicitly with ``make bench`` or ``pytest -m slow tests/benchmarks``.

**What each row exercises.** ``export`` is :func:`graphrag.sna.export.build_network`, the read
every ``sna`` command pays before it does anything else, timed once with ATL-ENT-3's cache off
(a cold Cypher read + graph assembly) and once through :func:`graphrag.sna.cache.
build_network_cached` on a hit (should not touch the store at all). ``pagerank`` and ``degree``
are :func:`graphrag.sna.measures.centrality`'s two cheapest kinds -- betweenness and closeness are
deliberately not here, since their O(VE) cost on a corpus-sized network is the reason those
commands warn about ``--max-nodes`` elsewhere rather than something ATL-ENT-3 changed. ``walks``
covers :func:`graphrag.sna.walks.stationary_distribution` (closed form, no eigensolver at all),
:func:`graphrag.sna.walks.fiedler_vector` and :func:`graphrag.sna.walks.consensus_time` (both
routed through :func:`graphrag.sna.matrices.eigenpairs`'s sparse path above
:data:`graphrag.sna.matrices.SPARSE_ABOVE_NODES` nodes) -- the three walk quantities this ticket
actually changed the algorithm for, run on the network's largest connected component since both
are undefined on a disconnected one.

**ATL-F2's rows.** ``sna backbone``'s eight-method comparison and ``sna analyze --uncertain``
with 200 samples each ran over two hours on a 7,000-node entity network; these four rows are
this network's version of that regression. ``backbone: high-salience`` times
:func:`graphrag.sna.backbone.high_salience` at its own default -- a degree-stratified sample of
:data:`graphrag.sna.backbone.SALIENCE_SAMPLE_ABOVE_NODES` sources once the network passes that
many nodes, rather than one Dijkstra per node. ``backbone: doubly-stochastic`` times
:func:`graphrag.sna.backbone.doubly_stochastic_scores` at its own default, sparse above
:data:`graphrag.sna.matrices.SPARSE_ABOVE_NODES` nodes. ``uncertain: cheap centralities`` times
200 realisations of pagerank (:data:`graphrag.sna.measures.CHEAP_CENTRALITIES`), the default
``--uncertain`` now samples; ``uncertain: --max-seconds budget`` times the same 200 requested
realisations of betweenness (:data:`graphrag.sna.measures.EXPENSIVE_CENTRALITIES`, the kind
``--uncertain-expensive`` opts back in) under a five-second budget, which is the row that would
have hung for the better part of the original two hours without one.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import networkx as nx
import pytest

from graphrag.config import load_settings
from graphrag.graph import snapshot as snap
from graphrag.graph.neo4j_store import Neo4jGraphStore
from graphrag.sna.backbone import backbone
from graphrag.sna.cache import NetworkCache, build_network_cached
from graphrag.sna.export import build_network
from graphrag.sna.measures import centrality
from graphrag.sna.uncertain import EDGE_PROBABILITY, node_expectation
from graphrag.sna.walks import consensus_time, fiedler_vector, stationary_distribution

pytestmark = pytest.mark.slow

#: Wall-clock ceilings, in seconds, generous enough to hold in Docker on a laptop with the
#: sparse routing this ticket adds; a regression back to always-dense would blow the walks rows
#: first, since those are exactly the ones an O(n^3) dense eigendecomposition makes quadratic in
#: memory and cubic in time on a network this size.
BUDGETS: dict[str, float] = {
    "export (cold, no cache)": 60.0,
    "export (cache miss, writes)": 60.0,
    "export (cache hit)": 5.0,
    "centrality: degree": 20.0,
    "centrality: pagerank": 30.0,
    "walks: stationary_distribution": 10.0,
    "walks: fiedler_vector (giant component)": 60.0,
    "walks: consensus_time (giant component)": 60.0,
    # ATL-F2: a regression back to one Dijkstra per node, or a dense Sinkhorn matrix, would blow
    # these two first -- that is the whole point of the two rows. Measured 137.7s and 2.5s on
    # this repository's `product-leader` persona (3,281 nodes -- just over
    # SALIENCE_SAMPLE_ABOVE_NODES, so the sampled row's saving here is modest; at the 7,000-node
    # scale the ticket names, the same fixed sample size is a much larger share cut) on a machine
    # busy running several other tickets' containers at once, hence the headroom.
    "backbone: high-salience (sampled)": 240.0,
    "backbone: doubly-stochastic (sparse)": 30.0,
    # 200 realisations of the cheapest centrality in the battery, the default --uncertain now
    # samples: measured 16.2s on the same persona and machine.
    "uncertain: cheap centralities (200 samples)": 90.0,
    # This row's ceiling is deliberately loose and is not the real assertion: exact betweenness
    # on a corpus-sized network can cost more per *single* call than the whole budget (measured
    # 79s for one call on this persona), and --max-seconds can only stop the *next* realisation
    # from starting, not pre-empt one already running -- see the docstring's own caveat. The
    # timing here only has to catch the budget being ignored outright (200 full realisations
    # would run for hours); the real, load-independent check is the realisation count asserted
    # below.
    "uncertain: --max-seconds budget (expensive, 5s cap)": 600.0,
}

#: The wall-clock budget the last row above is held to (ATL-F2): --max-seconds itself, not the
#: BUDGETS ceiling, which only has to hold a little longer to allow for the loop's own overhead.
UNCERTAIN_MAX_SECONDS = 5.0


@pytest.fixture(scope="module")
def largest_persona() -> Iterator[tuple[Neo4jGraphStore, str]]:
    """The store and the id of whichever committed snapshot has the most chunks, or a skip."""
    settings = load_settings()
    manifests = snap.list_snapshots(settings.snapshots_dir)
    if not manifests:
        pytest.skip(f"no committed snapshot under {settings.snapshots_dir}; run `make setup`")
    largest = max(manifests, key=lambda manifest: manifest.chunk_count)
    store = Neo4jGraphStore(settings.neo4j)
    try:
        store.verify_connectivity()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable at {settings.neo4j.uri}: {exc}")
    loaded = {persona.id for persona in store.list_personas()}
    if largest.persona_id not in loaded:
        store.close()
        pytest.skip(
            f"persona {largest.persona_id!r} has a snapshot but is not loaded; run "
            "`make snapshot-load`"
        )
    yield store, largest.persona_id
    store.close()


def _giant_component(graph: nx.Graph) -> nx.Graph:
    """The largest connected component, undirected -- what `fiedler_vector`/`consensus_time`
    need, since both are undefined on a disconnected network (§11.5, §11.6)."""
    flat = graph.to_undirected() if graph.is_directed() else graph
    members = max(nx.connected_components(flat), key=len)
    return nx.Graph(flat.subgraph(members))


def test_sna_command_budgets(
    largest_persona: tuple[Neo4jGraphStore, str], capsys: pytest.CaptureFixture[str]
) -> None:
    store, persona_id = largest_persona
    settings = load_settings()
    cache = NetworkCache(settings.sna_cache_dir)
    cache.clear()  # a stale entry from a previous run must not shortcut the "cold" row

    rows: list[tuple[str, float]] = []

    started = time.perf_counter()
    graph = build_network(store, "entities", persona_id)
    rows.append(("export (cold, no cache)", time.perf_counter() - started))

    if graph.number_of_nodes() == 0:
        pytest.skip(f"persona {persona_id!r} has an empty entity network; nothing to time")

    started = time.perf_counter()
    build_network_cached(store, "entities", persona_id, settings=settings)
    rows.append(("export (cache miss, writes)", time.perf_counter() - started))

    started = time.perf_counter()
    build_network_cached(store, "entities", persona_id, settings=settings)
    rows.append(("export (cache hit)", time.perf_counter() - started))

    started = time.perf_counter()
    centrality(graph, "degree")
    rows.append(("centrality: degree", time.perf_counter() - started))

    started = time.perf_counter()
    centrality(graph, "pagerank")
    rows.append(("centrality: pagerank", time.perf_counter() - started))

    started = time.perf_counter()
    stationary_distribution(graph)
    rows.append(("walks: stationary_distribution", time.perf_counter() - started))

    giant = _giant_component(graph)
    started = time.perf_counter()
    fiedler_vector(giant)
    rows.append(("walks: fiedler_vector (giant component)", time.perf_counter() - started))

    started = time.perf_counter()
    consensus_time(giant)
    rows.append(("walks: consensus_time (giant component)", time.perf_counter() - started))

    # ATL-F2: high-salience by a sampled set of sources above SALIENCE_SAMPLE_ABOVE_NODES nodes,
    # instead of one Dijkstra per node.
    started = time.perf_counter()
    backbone(graph, "high-salience", seed=0)
    rows.append(("backbone: high-salience (sampled)", time.perf_counter() - started))

    # ATL-F2: the Sinkhorn normalisation through a sparse matrix above SPARSE_ABOVE_NODES nodes,
    # instead of a dense n x n one.
    started = time.perf_counter()
    backbone(graph, "doubly-stochastic")
    rows.append(("backbone: doubly-stochastic (sparse)", time.perf_counter() - started))

    # ATL-F2: --uncertain's realisations, budgeted. Every edge is given a probability -- this
    # network's own, from entity_co_mention's tiers, would do as well for timing, but a uniform
    # one guarantees every realisation actually differs from the observed graph rather than
    # collapsing onto the certain-network fast path if this persona's edges all priced at 1.0.
    nx.set_edge_attributes(graph, 0.9, EDGE_PROBABILITY)

    started = time.perf_counter()
    node_expectation(graph, lambda world: centrality(world, "pagerank"), samples=200, seed=0)
    rows.append(("uncertain: cheap centralities (200 samples)", time.perf_counter() - started))

    # Betweenness on a corpus-sized network is exactly the case the ATL-ENT-3 docstring above
    # already warns is too slow to benchmark unbudgeted -- a single call here can itself take
    # longer than the whole budget, so the timing row is a loose sanity ceiling (catches the
    # budget being ignored outright) and the real assertion is on the realisation count: with
    # 200 requested and a 5-second budget, only a handful can possibly finish.
    started = time.perf_counter()
    expensive = node_expectation(
        graph,
        lambda world: centrality(world, "betweenness"),
        samples=200,
        seed=0,
        max_seconds=UNCERTAIN_MAX_SECONDS,
    )
    rows.append(
        ("uncertain: --max-seconds budget (expensive, 5s cap)", time.perf_counter() - started)
    )
    ran = next(iter(expensive.values())).samples

    cache.clear()

    with capsys.disabled():
        print(
            f"\nATL-ENT-3/ATL-F2 timing budget -- persona={persona_id!r}, "
            f"nodes={graph.number_of_nodes():,}, edges={graph.number_of_edges():,}, "
            f"giant_component={giant.number_of_nodes():,} nodes"
        )
        print(f"{'command':42}{'seconds':>10}{'budget':>10}")
        for name, seconds in rows:
            print(f"{name:42}{seconds:10.3f}{BUDGETS[name]:10.1f}")
        print(
            f"uncertain: --max-seconds actually ran {ran} of 200 requested realisations "
            f"(budget {UNCERTAIN_MAX_SECONDS:g}s)"
        )

    over_budget = [(name, seconds) for name, seconds in rows if seconds > BUDGETS[name]]
    assert not over_budget, (
        f"over budget on persona {persona_id!r} "
        f"({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges): "
        + ", ".join(
            f"{name}={seconds:.3f}s > {BUDGETS[name]:.1f}s" for name, seconds in over_budget
        )
    )
    # The load-independent check: whatever a single betweenness call costs on this machine right
    # now, a 5-second budget against 200 requested realisations must stop it far short of all of
    # them, or --max-seconds is doing nothing.
    assert ran < 50, (
        f"--max-seconds {UNCERTAIN_MAX_SECONDS:g}s let {ran} of 200 realisations run; "
        "the budget should have stopped it far short of that"
    )
