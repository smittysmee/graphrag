"""Motifs and frequent subgraph mining, on graphs whose answer is known before the code runs.

Four kinds of check, one per method the chapter builds.

*The census (§41.2).* A four-node directed graph drawn by hand -- one reciprocated pair, one
arc, one node touching nothing -- has exactly one triple in each of four of the sixteen MAN
classes and none in the rest, and its undirected flattening has exactly one in each of three of
the four undirected classes. Both censuses are exhaustive, so counted by hand they sum to
``C(4, 3) = 4``, and that is what is asserted rather than a total read back from the code.

*The profile (§41.2, ATL-19).* A directed graph built from several disjoint feed-forward loops,
chained just enough that a degree-preserving rewiring has a move to make, is planted so that the
feed-forward class (``030T``) is over-represented against :func:`graphrag.sna.null.configuration`
-- the ATL-19 primitive this module is required to draw its nulls from, never a private shuffle.

*Isomorphism (§41.3).* A path relabelled is still that path; a path and a star of the same size
are not. :func:`canonical_form` and :func:`isomorphic` are asked the same question two ways and
must agree.

*Mining (§41.4, §41.5).* A triangle planted in exactly ``k`` of ``n`` per-document entity graphs
is found by :func:`transactional_mining` with support exactly ``k`` -- support counts documents,
not occurrences, so the two open edges of each planted triangle do not inflate it. A star mined
inside one network gets the minimum image support of §41.5: with only one node able to play the
centre, the pattern's support is 1 however many leaves it has, which is the chapter's own
example (p. 603) and the case :func:`single_graph_mining` exists to get right where a naive
occurrence count would not.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk, Document, Enrichment, Entity, Mention
from graphrag.sna.motifs import (
    TRIAD_CODES,
    canonical_form,
    document_graphs,
    isomorphic,
    mni_support,
    motif_counts,
    motif_profile,
    single_graph_mining,
    transactional_mining,
    triad_census,
)
from graphrag.sna.null import HOLDS_FIXED

SEED = 7


# ------------------------------------------------------------------------------ §41.2 the census


def test_directed_triad_census_matches_a_hand_count() -> None:
    """One reciprocated pair, one arc, one untouched node -- four nodes, four triples, by hand.

    ``a<->b`` is a mutual dyad, ``b->c`` an asymmetric one, ``a-c`` and every pair with ``d``
    null. Sorting the four triples of ``{a, b, c, d}`` by eye:

    - ``{a, b, c}``: M=1 (ab), A=1 (bc), N=1 (ac), and the arc leaves the mutual pair -- ``111U``.
    - ``{a, b, d}``: M=1 (ab), N=2 (ad, bd) -- ``102``.
    - ``{a, c, d}``: no edge at all among the three -- ``003``.
    - ``{b, c, d}``: A=1 (bc), N=2 (bd, cd) -- ``012``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    graph.add_edge("b", "c")

    census = triad_census(graph)
    assert census.directed is True
    assert census.triples == math.comb(4, 3) == 4
    expected = dict.fromkeys(TRIAD_CODES, 0)
    expected.update({"003": 1, "012": 1, "102": 1, "111U": 1})
    assert census.counts == expected
    assert sum(census.counts.values()) == census.triples
    # `connected` drops the three disconnected classes and keeps every other one, so only
    # 111U carries a nonzero count and every other connected class reads 0.
    dropped = {"003", "012", "102"}
    assert census.connected == {code: expected[code] for code in expected if code not in dropped}
    assert census.connected["111U"] == 1


def test_undirected_triad_census_and_clustering_match_the_same_hand_count() -> None:
    """The same edges, flattened: two of the four undirected classes, and clustering 0.

    ``a-b`` and ``b-c`` give one open triad (``{a, b, c}``) and two single-edge triples
    (``{a, b, d}``, ``{b, c, d}``), with ``{a, c, d}`` empty -- again four triples, none of them
    a triangle, so the global clustering coefficient (§12.2) is defined and is exactly 0.
    """
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")

    census = triad_census(graph)
    assert census.directed is False
    assert census.triples == 4
    assert census.counts == {"empty": 1, "one edge": 2, "open triad": 1, "triangle": 0}
    assert sum(census.counts.values()) == census.triples
    assert census.global_clustering() == pytest.approx(0.0)


def test_triad_census_raises_on_a_graph_with_no_triad_to_close() -> None:
    """A graph with an edge but no third node to close a triangle over has no clustering."""
    graph = nx.Graph([("a", "b")])
    with pytest.raises(ValueError, match="no triad"):
        triad_census(graph).global_clustering()


def test_motif_counts_refuses_a_size_it_does_not_offer() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c")])
    with pytest.raises(ValueError, match="size must be one of"):
        motif_counts(graph, size=5)


# --------------------------------------------------------------- §41.2 / ATL-19 the null and z


def _chained_feed_forward_loops(k: int) -> nx.DiGraph:
    """``k`` disjoint feed-forward loops (``a_i -> b_i``, ``b_i -> c_i``, ``a_i -> c_i``), joined
    tail to head (``c_i -> a_{i+1}``) so the degree sequence admits a directed edge swap at all --
    fully disjoint triangles are too rigid for ``nx.directed_edge_swap`` to find a move in."""
    graph = nx.DiGraph()
    for i in range(k):
        a, b, c = f"a{i}", f"b{i}", f"c{i}"
        graph.add_edge(a, b)
        graph.add_edge(b, c)
        graph.add_edge(a, c)
    for i in range(k):
        graph.add_edge(f"c{i}", f"a{(i + 1) % k}")
    return graph


def test_feed_forward_loop_is_overexpressed_against_the_configuration_null() -> None:
    """Ten planted feed-forward loops, tested against ATL-19's ``configuration`` null (§19.1).

    The planted network has exactly ten ``030T`` triples -- one per loop -- and a
    degree-preserving rewiring keeps every in- and out-degree but not which nodes closed a
    triangle, so the null count is expected to run well below ten. ``profile.null`` is asserted
    against :data:`graphrag.sna.null.HOLDS_FIXED`, the ATL-19 primitives' own sentence, rather
    than a copy: the module must draw this null from there and not from a private shuffle.
    """
    graph = _chained_feed_forward_loops(10)
    profile = motif_profile(graph, size=3, samples=200, seed=SEED)
    assert profile.samples > 0, "the chained loops must admit at least one directed edge swap"
    assert profile.null == HOLDS_FIXED["configuration"]

    row = next(r for r in profile.rows if r.key == "030T")
    assert row.count == 10
    assert row.test.null == "configuration"
    assert row.z > 5.0
    assert row in profile.over(2.0)


# --------------------------------------------------------------------------- §41.3 isomorphism


def test_canonical_form_and_isomorphic_agree_on_a_relabelled_and_a_different_shape() -> None:
    """A path relabelled is the path; a path and a star of the same size are not (§41.3)."""
    path = nx.path_graph(4)
    relabelled = nx.relabel_nodes(path, {0: "w", 1: "x", 2: "y", 3: "z"})
    star = nx.star_graph(3)  # also 4 nodes, 3 edges, but centred rather than strung out

    assert canonical_form(path) == canonical_form(relabelled)
    assert isomorphic(path, relabelled) is True

    assert canonical_form(path) != canonical_form(star)
    assert isomorphic(path, star) is False


def test_isomorphic_refuses_to_compare_across_directedness() -> None:
    with pytest.raises(ValueError, match="directed"):
        isomorphic(nx.DiGraph([("a", "b")]), nx.Graph([("a", "b")]))


def test_canonical_form_refuses_above_its_node_bound() -> None:
    with pytest.raises(ValueError, match="n!"):
        canonical_form(nx.complete_graph(9), max_nodes=8)


# ---------------------------------------------------------------------------- §41.4 mining


def _entity_document_store(n: int, k: int) -> InMemoryGraphStore:
    """``n`` one-passage documents: the first ``k`` name three entities X/Y/Z (a triangle once
    projected), the rest name two entities P/Q (a single edge). Written straight into the store
    so :func:`document_graphs` is exercised end to end, the way ``sna motifs --mine`` calls it."""
    store = InMemoryGraphStore()
    persona = "test-motifs"
    documents = []
    chunks = []
    mentions = []
    entities: dict[str, Entity] = {}
    for i in range(n):
        doc_id = f"doc-{i}"
        documents.append(
            Document(id=doc_id, persona_id=persona, source_id="src", title=doc_id, path=doc_id)
        )
        chunk_id = f"{doc_id}-c0"
        chunks.append(Chunk(id=chunk_id, doc_id=doc_id, persona_id=persona, ordinal=0, text="x"))
        names = [("x", "X"), ("y", "Y"), ("z", "Z")] if i < k else [("p", "P"), ("q", "Q")]
        for slug, label in names:
            entity_id = f"e-{slug}{i}"
            entities[entity_id] = Entity(id=entity_id, name=label, type="concept")
            mentions.append(Mention(chunk_id=chunk_id, entity_id=entity_id))
    store.upsert_documents(documents)
    store.upsert_chunks(chunks, np.zeros((len(chunks), 0), dtype=np.float32))
    store.upsert_enrichment(Enrichment(entities=list(entities.values()), mentions=mentions))
    return store


def test_transactional_mining_finds_a_planted_triangle_with_support_k() -> None:
    """A triangle planted in exactly 3 of 6 per-document graphs is found with support 3.

    Support counts *graphs*, not occurrences (§41.4, p. 598): each planted triangle contributes
    three edges and three connected two-edge subgraphs, but only ever one document, so the open
    triad's support equals the triangle's rather than tripling it.
    """
    n, k = 6, 3
    store = _entity_document_store(n, k)
    graphs = document_graphs(store, "test-motifs")
    assert len(graphs) == n

    mining = transactional_mining(graphs, min_support=1, max_size=3)
    assert mining.kind == "transactional"
    assert mining.graphs == n

    by_name = {pattern.name: pattern for pattern in mining.patterns}
    assert by_name["triangle"].support == k
    assert by_name["triangle"].size == 3
    assert by_name["open triad"].support == k
    # Every document holds at least one edge, planted or not, so "edge" has support n.
    assert by_name["edge"].support == n


def test_transactional_mining_refuses_bad_parameters() -> None:
    with pytest.raises(ValueError, match="min_support"):
        transactional_mining([nx.Graph([("a", "b")])], min_support=0)
    with pytest.raises(ValueError, match="max_size"):
        transactional_mining([nx.Graph([("a", "b")])], max_size=0)


# ---------------------------------------------------------------------------- §41.5 mining


def test_minimum_image_support_of_a_star_is_one_however_many_leaves() -> None:
    """The chapter's own worked case (p. 603): only the centre can ever play the centre's role.

    A four-leaf star has four two-step paths running through it, but every one of them maps its
    middle node onto the same single node of the network, so the minimum image support of the
    two-step path pattern is 1 -- not 4, which a naive occurrence count would give.
    """
    star = nx.star_graph(4)  # centre 0, leaves 1..4
    two_step_path = nx.path_graph(3)  # u - v - w
    assert mni_support(star, two_step_path) == 1


def test_minimum_image_support_of_an_empty_pattern_is_undefined() -> None:
    with pytest.raises(ValueError, match="no edges"):
        mni_support(nx.star_graph(3), nx.Graph())


def test_single_graph_mining_gives_the_windmill_star_support_one() -> None:
    """Four triangles sharing one hub: the hub is the only node that can be a 3-star's centre.

    Each blade's two spoke-role nodes (the pattern's own two other roles) range over all eight
    outer nodes, and the hub itself is one of every triangle's three roles, so the triangle's
    minimum image support is the number of outer nodes, 9 -- 4 blades of 2 each, plus the hub's
    own role reaching every blade. What that count cannot be is unbounded: a 4-edge star made of
    the hub and any four of its spokes can only ever be imaged with the *hub* in the centre role,
    so its minimum image support is 1 (§41.5, p. 603) regardless of how many ways to pick the
    leaves there are -- the same reading as :func:`mni_support` above, now reached through the
    level-wise search ``sna motifs --mine`` actually runs.
    """
    graph = nx.Graph()
    hub = "hub"
    for i in range(4):
        a, b = f"a{i}", f"b{i}"
        graph.add_edge(hub, a)
        graph.add_edge(hub, b)
        graph.add_edge(a, b)

    mining = single_graph_mining(graph, min_support=1, max_size=3)
    assert mining.kind == "single-graph"
    by_name = {pattern.name: pattern for pattern in mining.patterns}
    assert by_name["triangle"].support == 9
    assert by_name["star"].support == 1


def test_single_graph_mining_refuses_above_its_node_bound() -> None:
    graph = nx.path_graph(5)
    with pytest.raises(ValueError, match="bounded at"):
        single_graph_mining(graph, max_nodes=4)
