"""Chapter 34 over planted data and over the book's own figures.

Every number asserted here is known before the code runs. The §34.1 answers come from the Atlas
itself: Figure 7.9 (p. 113) and Figure 34.2 (p. 475) are the same picture, the book lists its
facets and works out two of its generalized degrees in prose, so the complex built from those
five facets has to reproduce them. The §34.2 and §34.3 answers come from planted passage
sequences short enough to solve by hand, and the working is written out in each test. The
Southern women complex is checked against an independent brute-force recount of the same
attendance table (``tests/legendary.py``), never against a previous run of this code.
"""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import HIGHORDER_DAMPING, HIGHORDER_MAX_DIM, app
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk, Document, Enrichment, Entity, Mention, PersonaSpec, SourceSpec
from graphrag.personas.registry import PersonaRegistry
from graphrag.sna.guide import READING_RULES
from graphrag.sna.highorder import (
    DEFAULT_DAMPING,
    DEFAULT_MAX_DIM,
    Passage,
    PassageSequence,
    build_memory_report,
    build_simplicial_report,
    closure,
    highorder_reports,
    memory_payload,
    memory_stationary,
    passage_sequences,
    render_memory,
    render_simplicial,
    second_order_network,
    simplicial_clustering,
    simplicial_complex,
    simplicial_degree,
    simplicial_degrees,
    simplicial_incidence,
    simplicial_payload,
    transition_id,
    transition_network,
)
from graphrag.sna.layers import Hypergraph, build_hypergraph, clique_expansion
from tests.legendary import southern_women

runner = CliRunner()

# The facets of the simplicial complex the Atlas draws twice -- Figure 7.9 (p. 113) and, with the
# generalized degrees worked out on it, Figure 34.2 (p. 475). §7.3 lists them itself: "The facets
# in the simplicial complex of Figure 7.9 are: [{1, 2}, {1, 3}, {2, 3, 4}, {4, 5}, {4, 6, 7, 8}]".
BOOK_FACETS: tuple[tuple[str, ...], ...] = (
    ("1", "2"),
    ("1", "3"),
    ("2", "3", "4"),
    ("4", "5"),
    ("4", "6", "7", "8"),
)


def book_complex(max_dim: int = 3) -> Hypergraph:
    """Figure 7.9 as a hypergraph, one hyperedge per facet, ready to be closed downward."""
    return build_hypergraph(
        ((member, f"facet-{index}") for index, facet in enumerate(BOOK_FACETS) for member in facet),
        persona_id="atlas-figure",
        member_noun="nodes",
        edge_noun="facet",
    )


def planted(letters: str, doc_id: str = "doc-1", start: int = 0) -> PassageSequence:
    """One document whose passages each name exactly one entity, in the order given."""
    return PassageSequence(
        doc_id=doc_id,
        passages=tuple(
            Passage(chunk_id=f"{doc_id}#{start + index}", ordinal=start + index, entities=(letter,))
            for index, letter in enumerate(letters)
        ),
    )


def weights(graph: nx.DiGraph) -> dict[tuple[str, str], int]:
    """Every directed edge as ``{(u, v): weight}``, so a table can be asserted whole."""
    return {(u, v): int(w) for u, v, w in graph.edges(data="weight")}


# ----------------------------------------------------------------------------- §34.1 the complex


def test_the_closure_reproduces_the_facets_the_book_lists() -> None:
    """§7.3, p. 113: the facets of Figure 7.9, and the dimension and purity it states.

    *"The facets in the simplicial complex of Figure 7.9 are: [{1, 2}, {1, 3}, {2, 3, 4}, {4, 5},
    {4, 6, 7, 8}]"*, *"the simplicial complex in Figure 7.9 has dimension 3"*, and it is *"not
    pure because it has facets of dimension three, two and one"*. A facet is a simplex that is a
    face of no larger one, so the 2-simplex {2, 3, 4} stays a facet while the triangle {4, 6, 7}
    inside the tetrahedron does not.
    """
    complex_ = simplicial_complex(book_complex())

    assert complex_.facets == BOOK_FACETS
    assert complex_.dimension == 3
    assert complex_.pure is False

    # The closure of {4, 6, 7, 8} alone: 4 nodes, 6 edges, 4 triangles, 1 tetrahedron (§7.3,
    # p. 112, which counts exactly those "faces of the simplex"). Adding the rest of the figure:
    # 8 nodes; 6 + 1 (4-5) + 3 ({2,3,4}) + 2 ({1,2} and {1,3}) = 12 edges; 4 + 1 = 5 triangles.
    assert [len(complex_.simplices(dim)) for dim in range(4)] == [8, 12, 5, 1]


def test_the_generalized_degrees_are_the_two_the_chapter_works_out() -> None:
    """§34.1, p. 475: the only two values of ``k_(d,m)`` the book prints for its figure.

    *"The k2,0 of node 4 is four. That's because we're looking at d = 2, i.e. 2-simplices -- also
    known as triangles --, on an m = 0 face -- which is a node. Node 4 gets three 2-simplices
    from its 3-simplex connecting it to nodes 6, 7, and 8, and another 2-simplex with nodes 2 and
    3. On the other hand, the k2,1 of edge (4, 8) is two. The 1-simplex -- i.e. edge -- (4, 8)
    participates in two 2-simplices -- i.e. triangles -- with nodes 6 and 7."*
    """
    complex_ = simplicial_complex(book_complex())

    assert simplicial_degree(complex_, "4", 2) == 4
    assert simplicial_degree(complex_, ("4", "8"), 2) == 2
    # The incidence of p. 476 is k_(d,d-1) on a face one dimension below: for the edge (4, 8)
    # that is the same two triangles, and two is already above the 0 or 1 a manifold allows.
    assert simplicial_incidence(complex_, ("4", "8")) == 2

    by_node = simplicial_degrees(complex_, 2, of=0)
    assert by_node[("4",)] == 4
    assert by_node[("6",)] == by_node[("7",)] == by_node[("8",)] == 3
    assert by_node[("5",)] == 0  # 4-5 is a facet of its own: an edge in no triangle at all
    assert simplicial_degrees(complex_, 2, of=1)[("4", "8")] == 2

    with pytest.raises(ValueError, match="m < d"):
        simplicial_degree(complex_, ("4", "8"), 1)


def test_the_incidence_of_a_bare_node_is_its_edges_not_its_characters() -> None:
    """A node id is one face, not a sequence of its letters (§34.1's k_(d,d-1) on an m-face).

    ``simplicial_degree`` has always read a bare string as one node. The incidence takes the same
    type for the same reason: "alpha" read as a sequence would ask about a 4-simplex over the
    letters a, l, p, h -- a face the complex does not hold -- and answer 0, which is a wrong
    answer rather than an error.
    """
    hyper = build_hypergraph(
        [
            ("alpha", "p1"),
            ("beta", "p1"),
            ("gamma", "p1"),
            ("alpha", "p2"),
            ("delta", "p2"),
        ],
        member_noun="entities",
        edge_noun="passage",
    )
    complex_ = simplicial_complex(hyper)

    # Alpha is in three edges (beta, gamma from p1; delta from p2) and one filled triangle.
    assert simplicial_incidence(complex_, "alpha") == 3
    assert simplicial_degree(complex_, "alpha", 1) == 3
    assert simplicial_incidence(complex_, ("alpha",)) == 3
    assert simplicial_incidence(complex_, ("alpha", "beta")) == 1


def test_the_only_unfilled_triangle_of_the_books_figure_is_the_one_nobody_wrote() -> None:
    """The closure of Figure 7.9: six triangles in the 1-skeleton, five of them simplices.

    {1, 2} and {1, 3} are facets and {2, 3} comes from the 2-simplex {2, 3, 4}, so 1, 2 and 3 are
    pairwise joined -- but no facet holds all three, which is §34.1's own observation about its
    Figure 34.3 (*"nodes 1, 3, and 4 make up a triangle but not a simplex"*) happening here. The
    other five triangles are {2, 3, 4} and the four faces of the tetrahedron.
    """
    complex_ = simplicial_complex(book_complex())
    result = closure(complex_)

    assert result.triangles == 6
    assert result.filled == 5
    assert result.unfilled == (("1", "2", "3"),)
    assert simplicial_clustering(complex_) == pytest.approx(5 / 6)


def test_max_dim_truncates_the_closure_and_the_frame_says_so() -> None:
    """Cutting the closure at dimension 2 drops the tetrahedron and keeps its four triangles."""
    cut = simplicial_complex(book_complex(), max_dim=2)

    assert cut.dimension == 2
    assert cut.simplices(3) == ()
    assert len(cut.simplices(2)) == 5
    assert cut.truncated == 1  # only {4, 6, 7, 8} names more than three nodes
    assert "1 facet(s) named more than 3 nodes" in cut.frame
    # The facets are what the corpus stated, so they are unaffected by where the closure stopped.
    assert cut.facets == BOOK_FACETS

    with pytest.raises(ValueError, match="past the"):
        simplicial_complex(book_complex(), max_faces=3)


def test_the_planted_passages_fill_one_triangle_and_no_other() -> None:
    """Three passages naming {A, B, C}, {A, B} and {C, D}: the ticket's own planted answer.

    The first passage is a 2-simplex, the other two are 1-simplices. So A is in two edges and one
    filled triangle, and the 1-skeleton -- A-B, A-C, B-C, C-D -- holds exactly one triangle,
    which is filled. Nothing is left open, which is the case the closure exists to distinguish
    from the one below it.
    """
    hyper = build_hypergraph(
        [("A", "p1"), ("B", "p1"), ("C", "p1"), ("A", "p2"), ("B", "p2"), ("C", "p3"), ("D", "p3")],
        member_noun="entities",
        edge_noun="passage",
    )
    complex_ = simplicial_complex(hyper)

    assert complex_.simplices(2) == (("A", "B", "C"),)
    assert simplicial_degree(complex_, "A", 1) == 2
    assert simplicial_degree(complex_, "A", 2) == 1
    assert closure(complex_) == closure(complex_)  # a frozen dataclass of counts, not a graph
    assert closure(complex_).triangles == 1
    assert closure(complex_).unfilled == ()
    assert simplicial_clustering(complex_) == 1.0


def test_a_skeleton_with_no_triangle_has_no_clustering() -> None:
    """Two passages naming a pair each: three edges, no triple, so the ratio is undefined."""
    hyper = build_hypergraph([("A", "p1"), ("B", "p1"), ("B", "p2"), ("C", "p2")])
    complex_ = simplicial_complex(hyper)

    assert closure(complex_).triangles == 0
    with pytest.raises(ValueError, match="undefined"):
        simplicial_clustering(complex_)


def test_the_southern_women_complex_matches_a_recount_of_the_attendance_table() -> None:
    """The published two-mode graph closed into a complex, checked against a brute-force recount.

    The 14 events are the simplices and the 18 women their members. Every number below is
    recomputed here straight from ``nx.davis_southern_women_graph`` with set arithmetic, so the
    assertion compares two independent implementations of §7.3's closure rather than trusting
    this one. The interesting number is the closure: three women can be pairwise co-attendees
    without any single event holding all three, and on these data most triples are exactly that.
    """
    graph, known = southern_women()
    women = set(known.groups["women"])
    events = {
        event: frozenset(n for n in graph.neighbors(event) if n in women)
        for event in known.groups["events"]
    }
    hyper = build_hypergraph(
        [(woman, event) for event, attendees in events.items() for woman in attendees],
        member_noun="women",
        edge_noun="event",
    )
    complex_ = simplicial_complex(hyper, max_dim=2)

    expected_triples = {
        triple for attendees in events.values() for triple in combinations(sorted(attendees), 3)
    }
    expected_pairs = {
        pair for attendees in events.values() for pair in combinations(sorted(attendees), 2)
    }
    assert len(complex_.simplices(0)) == 18
    assert set(complex_.simplices(1)) == expected_pairs
    assert set(complex_.simplices(2)) == expected_triples

    # The 1-skeleton is the clique expansion of the same hyperedges with the weights dropped,
    # which is the claim SimplicialComplex.skeleton makes and §7.3's "any network is a skeleton
    # of one or more simplicial complexes" read from the other end.
    skeleton = complex_.skeleton()
    assert {tuple(sorted(edge)) for edge in clique_expansion(hyper).edges} == expected_pairs
    assert {tuple(sorted(edge)) for edge in skeleton.edges} == expected_pairs

    expected_open = {
        triple
        for triple in combinations(sorted(women), 3)
        if all(skeleton.has_edge(u, v) for u, v in combinations(triple, 2))
        and triple not in expected_triples
    }
    result = closure(complex_)
    assert set(result.unfilled) == expected_open
    assert result.triangles == len(expected_triples) + len(expected_open)
    assert 0.0 < result.ratio < 1.0

    # Evelyn and Theresa shared seven events, so their edge sits in many filled triangles; Olivia
    # and Flora attended two events and only those two (see tests/legendary.py), so their edge
    # sits in exactly the triangles those two events' other attendees make.
    pair = {"Olivia Carleton", "Flora Price"}
    shared = [attendees for attendees in events.values() if pair <= attendees]
    others = {woman for attendees in shared for woman in attendees} - pair
    assert simplicial_incidence(complex_, ("Flora Price", "Olivia Carleton")) == len(others)


# ------------------------------------------------------------------- §34.2 the memory network


def test_the_planted_document_gives_the_transitions_the_sequence_shows() -> None:
    """The ticket's planted sequence [A][B][C][A][B][D], solved by reading it left to right.

    Consecutive pairs: A-B, B-C, C-A, A-B, B-D -- four distinct transitions, one of them seen
    twice. Consecutive triples: A-B-C, B-C-A, C-A-B, A-B-D -- so the memory network joins
    ``A -> B`` to ``B -> C`` once and to ``B -> D`` once, which is the whole point of the
    structure: the same node ``A -> B`` continues two different ways and the weights say how
    often each.
    """
    sequences = [planted("ABCABD")]
    memory = second_order_network(sequences)

    assert sorted(memory.nodes) == [
        transition_id("A", "B"),
        transition_id("B", "C"),
        transition_id("B", "D"),
        transition_id("C", "A"),
    ]
    assert weights(memory) == {
        (transition_id("A", "B"), transition_id("B", "C")): 1,
        (transition_id("A", "B"), transition_id("B", "D")): 1,
        (transition_id("B", "C"), transition_id("C", "A")): 1,
        (transition_id("C", "A"), transition_id("A", "B")): 1,
    }
    assert memory.nodes[transition_id("A", "B")]["occurrences"] == 2
    assert memory.nodes[transition_id("A", "B")]["source"] == "A"
    assert memory.nodes[transition_id("A", "B")]["target"] == "B"
    # B -> D is the last transition of the document: a node with no continuation, kept because
    # dropping it would quietly shorten the document by a passage.
    assert memory.out_degree(transition_id("B", "D")) == 0

    plain = transition_network(sequences)
    assert weights(plain) == {("A", "B"): 2, ("B", "C"): 1, ("C", "A"): 1, ("B", "D"): 1}


def test_a_document_boundary_is_not_a_transition() -> None:
    """Two documents [A][B] and [C][D] give A -> B and C -> D, and never B -> C."""
    sequences = [planted("AB", doc_id="doc-1"), planted("CD", doc_id="doc-2")]
    plain = transition_network(sequences)

    assert sorted(plain.edges) == [("A", "B"), ("C", "D")]
    assert not plain.has_edge("B", "C")
    assert second_order_network(sequences).number_of_edges() == 0
    assert plain.graph["documents"] == 2
    assert plain.graph["occurrences"] == 2


def test_a_gap_in_the_passage_order_breaks_the_chain_instead_of_bridging_it() -> None:
    """Passages 0, 1 and 3 of one document: the missing passage 2 is a break, not a step.

    Passage 2 named nothing, or a filter removed it. Either way the corpus never put B next to C,
    so the chain stops and the report counts the break rather than inventing the transition.
    """
    sequence = PassageSequence(
        doc_id="doc-1",
        passages=(
            Passage(chunk_id="c0", ordinal=0, entities=("A",)),
            Passage(chunk_id="c1", ordinal=1, entities=("B",)),
            Passage(chunk_id="c3", ordinal=3, entities=("C",)),
        ),
    )
    plain = transition_network([sequence])

    assert sorted(plain.edges) == [("A", "B")]
    assert plain.graph["breaks"] == 1
    assert plain.graph["occurrences"] == 1


def test_an_entity_carried_into_the_next_passage_is_not_a_step() -> None:
    """[A][A,B][B] gives A -> B twice and no self-loop: the repeats are continuations.

    Passage 1 names both, so there are two steps into a different thing -- A (passage 0) to B
    (passage 1), and A (passage 1) to B (passage 2) -- while A into passage 1 and B out of it are
    the same entity twice and are dropped rather than turned into ``A -> A``.
    """
    sequence = PassageSequence(
        doc_id="doc-1",
        passages=(
            Passage(chunk_id="c0", ordinal=0, entities=("A",)),
            Passage(chunk_id="c1", ordinal=1, entities=("A", "B")),
            Passage(chunk_id="c2", ordinal=2, entities=("B",)),
        ),
    )
    plain = transition_network([sequence])

    assert weights(plain) == {("A", "B"): 2}
    assert plain.graph["carried"] == 2  # A into passage 1, B out of it into passage 2
    assert not list(nx.selfloop_edges(plain))


# ------------------------------------------------------------------- §34.3 the walk with a memory


def test_the_memory_walk_differs_from_the_first_order_one_on_a_hand_solved_document() -> None:
    """The planted sequence [A][B][C][B][A][D][A][B][C], both walks solved as fractions.

    **First order.** The transitions are A-B, B-C, C-B, B-A, A-D, D-A, A-B, B-C, so the weighted
    digraph is A->B 2, B->C 2, C->B 1, B->A 1, A->D 1, D->A 1 and the row-stochastic P is
    A: B 2/3, D 1/3; B: C 2/3, A 1/3; C: B 1; D: A 1. Solving pi = pi P by hand:
    pi_A = pi_B/3 + pi_D, pi_D = pi_A/3, hence pi_B = 2 pi_A; pi_C = 2 pi_B/3 = 4 pi_A/3. With
    pi_A (1 + 2 + 4/3 + 1/3) = 1 that is **A 3/14, B 3/7, C 2/7, D 1/14**.

    **Second order.** The triples are ABC, BCB, CBA, BAD, ADA, DAB, ABC, so the memory network is
    the single directed cycle A->B => B->C => C->B => B->A => A->D => D->A => A->B (the first
    edge carries weight 2 because ABC occurs twice; a cycle's stationary distribution does not
    depend on the weights). Six nodes on one cycle means **1/6 each**, and folding by the
    transition's target -- standing on ``u -> v`` is standing on v -- gives B 1/6 + 1/6, A 1/6 +
    1/6, C 1/6, D 1/6, i.e. **A 1/3, B 1/3, C 1/6, D 1/6**.

    So the two disagree, and they disagree by more than a rounding: first order ranks C (2/7)
    above A (3/14), while the walker that remembers where it came from puts A (1/3) above C
    (1/6). That is §34.3's rank change, on a document a reader can check with a pencil.
    """
    sequences = [planted("ABCBADABC")]
    plain = transition_network(sequences)
    memory = second_order_network(sequences)

    assert weights(plain) == {
        ("A", "B"): 2,
        ("B", "C"): 2,
        ("C", "B"): 1,
        ("B", "A"): 1,
        ("A", "D"): 1,
        ("D", "A"): 1,
    }
    assert memory.number_of_nodes() == 6
    assert nx.is_strongly_connected(memory)

    walk = memory_stationary(memory, plain, damping=None)
    assert walk.first_order == pytest.approx(
        {
            "A": float(Fraction(3, 14)),
            "B": float(Fraction(3, 7)),
            "C": float(Fraction(2, 7)),
            "D": float(Fraction(1, 14)),
        }
    )
    assert walk.memory == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 6, "D": 1 / 6})

    moves = {move.entity: move for move in walk.moves}
    assert (moves["C"].first_order_rank, moves["C"].memory_rank) == (2, 3)
    assert (moves["A"].first_order_rank, moves["A"].memory_rank) == (3, 1)
    assert moves["A"].moved == 2 and moves["C"].moved == -1
    assert "no teleport" in walk.method


def test_a_single_cycle_makes_both_walks_agree_because_it_is_a_cycle() -> None:
    """[A][B][C][A][B][C][A]: memory changes nothing here, and the reason is narrow.

    The first-order network is the 3-cycle A->B->C->A with weight 2 on each edge, and the memory
    network is the 3-cycle of its transitions, A->B => B->C => C->A => A->B. A directed cycle has
    a uniform stationary distribution whatever its weights, so both come out at 1/3 and folding
    the second back onto entities reproduces the first.

    That is a property of the *cycle*, not a law about closed documents: the next test closes a
    document on the entity it began with and the two walks still disagree. What would make them
    agree in general is balance in the memory network itself -- every transition continued as
    often as it is entered -- which this sequence does not have either (C->A is entered once and
    left once, but A->B is entered once and left twice).
    """
    sequences = [planted("ABCABCA")]
    walk = memory_stationary(
        second_order_network(sequences), transition_network(sequences), damping=None
    )

    assert walk.first_order == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert walk.memory == pytest.approx(walk.first_order)
    assert all(move.moved == 0 for move in walk.moves)


def test_closing_a_document_on_its_first_entity_does_not_make_the_walks_agree() -> None:
    """[A][B][A][B][A][C][B][A]: closed on A, nothing repeated, and the two walks still differ.

    **First order.** Transitions AB, BA, AB, BA, AC, CB, BA give A->B 2, B->A 3, A->C 1, C->B 1,
    so P is A: B 2/3, C 1/3; B: A 1; C: B 1. Then pi_A = pi_B, pi_C = pi_A/3, and
    pi_B = (2/3)pi_A + pi_C = pi_A checks out; with pi_A(1 + 1 + 1/3) = 1 that is
    **A 3/7, B 3/7, C 1/7**. Every entity is left as often as it is entered -- the document ends
    where it began -- which is exactly why the first-order chain settles on the corpus's own
    frequencies.

    **Second order.** The triples ABA, BAB, ABA, BAC, ACB, CBA give A->B => B->A (2),
    B->A => A->B, B->A => A->C, A->C => C->B, C->B => B->A. Writing pi(B->A) = x: A->B and A->C
    each take half of x, C->B equals A->C, and B->A is fed by A->B and C->B, x/2 + x/2 = x. With
    x + 3(x/2) = 1 that is pi(B->A) = 2/5 and 1/5 each for the other three, and folding by the
    transition's target gives **A 2/5, B 2/5, C 1/5**.

    So C rises from 1/7 to 1/5 with no teleport anywhere: the memory chain is *not* balanced by a
    closed entity sequence, because A->B is entered once and left twice while B->A is entered
    twice and left once. The ranking happens not to move -- A and B are tied in both walks, and
    the tolerance in `_ranks` keeps them tied rather than separating them by the last bits of the
    iteration -- which is itself worth asserting: the memory walk can change every number without
    changing the order.
    """
    sequences = [planted("ABABACBA")]
    plain = transition_network(sequences)
    memory = second_order_network(sequences)
    walk = memory_stationary(memory, plain, damping=None)

    assert weights(plain) == {("A", "B"): 2, ("B", "A"): 3, ("A", "C"): 1, ("C", "B"): 1}
    assert sorted(memory.nodes) == [
        transition_id("A", "B"),
        transition_id("A", "C"),
        transition_id("B", "A"),
        transition_id("C", "B"),
    ]
    assert walk.first_order == pytest.approx({"A": 3 / 7, "B": 3 / 7, "C": 1 / 7})
    assert walk.memory == pytest.approx({"A": 2 / 5, "B": 2 / 5, "C": 1 / 5})
    assert walk.memory["C"] > walk.first_order["C"]
    # Tied entities stay tied: the two walks differ in every value and in no rank.
    assert {move.entity: move.moved for move in walk.moves} == {"A": 0, "B": 0, "C": 0}


def test_the_damped_walk_is_run_on_both_networks_or_on_neither() -> None:
    """A corpus whose memory network is not strongly connected: the undamped walk has no answer.

    §34.2 says HON-style structures *"tend to transform into weakly connected graphs, or even not
    connected"* (p. 483), and [A][B][C][A][B][D] is that in miniature: the walker reaching
    ``B -> D`` can never leave. The damped walk of §11.1 exists there and is what the report runs
    by default, on both networks, so the two numbers are the same quantity.
    """
    sequences = [planted("ABCABD")]
    plain = transition_network(sequences)
    memory = second_order_network(sequences)

    assert not nx.is_strongly_connected(memory)
    with pytest.raises(ValueError, match=r"no out-edge|not strongly connected"):
        memory_stationary(memory, plain, damping=None)

    walk = memory_stationary(memory, plain, damping=0.85)
    assert sum(walk.memory.values()) == pytest.approx(1.0)
    assert sum(walk.first_order.values()) == pytest.approx(1.0)
    assert walk.damping == 0.85
    assert "damping 0.85" in walk.method
    with pytest.raises(ValueError, match="damping must be"):
        memory_stationary(memory, plain, damping=0.0)


def test_a_corpus_with_nothing_consecutive_is_refused_rather_than_answered() -> None:
    """One passage per document: no transition at all, so neither walk is defined."""
    sequences = [planted("A", doc_id="doc-1"), planted("B", doc_id="doc-2")]
    plain = transition_network(sequences)
    memory = second_order_network(sequences)

    with pytest.raises(ValueError, match="no first-order transitions"):
        memory_stationary(memory, plain)

    with pytest.raises(ValueError, match="no second-order paths"):
        memory_stationary(
            second_order_network([planted("AB")]), transition_network([planted("AB")])
        )


def test_the_second_order_network_refuses_a_corpus_larger_than_its_guard() -> None:
    sequences = [planted("ABCABD")]
    with pytest.raises(ValueError, match="past the 2 limit"):
        second_order_network(sequences, max_transitions=2)


# ----------------------------------------------------------------------------- reports and store


def test_both_reports_print_their_frame_their_n_and_the_chapter() -> None:
    hyper = build_hypergraph(
        [("A", "p1"), ("B", "p1"), ("C", "p1"), ("A", "p2"), ("B", "p2")],
        persona_id="test",
        member_noun="entities",
        edge_noun="passage",
        labels={"A": "Alpha", "B": "Beta", "C": "Gamma"},
    )
    simplicial = build_simplicial_report(hyper)
    memory = build_memory_report([planted("ABCBADABC")], persona_id="test", damping=0.85)

    structure = "\n".join(render_simplicial(simplicial))
    assert "Atlas" in structure and "§34.1" in structure
    assert "Sampling frame:" in structure and "Null model: none" in structure
    assert "**n:** 3 node(s)" in structure
    assert "Alpha" in structure  # labels, not ids, in the table

    dynamics = "\n".join(render_memory(memory))
    assert "§34.2" in dynamics and "§34.3" in dynamics
    assert "Sampling frame:" in dynamics and "**Null model:**" in dynamics
    assert "damping 0.85" in dynamics
    assert "4 entities and 6 transitions first-order" in dynamics

    structure_payload = simplicial_payload(simplicial)
    assert structure_payload["closure"]["triangles"] == 1
    assert structure_payload["null_model"].startswith("none")
    assert structure_payload["faces"]["2"] == 1

    dynamics_payload = memory_payload(memory)
    assert dynamics_payload["second_order"]["nodes"] == 6
    assert dynamics_payload["walk"]["damping"] == 0.85
    assert {row["entity"] for row in dynamics_payload["walk"]["entities"]} == set("ABCD")


PERSONA = "test-highorder"
SOURCE = "posts"
#: Two documents. The first is the hand-solved sequence of the walk test above, one entity per
#: passage; the second is two passages naming one more pair, so a boundary has two sides.
PLANTED_DOCUMENTS: dict[str, str] = {"doc-1": "ABCBADABC", "doc-2": "CD"}
NAMES = {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}


@pytest.fixture
def sequenced(
    memory_store: InMemoryGraphStore,
    hash_embedder: HashEmbedder,
    registry: PersonaRegistry,
) -> InMemoryGraphStore:
    """The planted documents above in the in-memory store, with real chunk ordinals."""
    registry.save(
        PersonaSpec(
            id=PERSONA,
            name="Test High Order",
            role_prompt="You answer from captured posts.",
            sources=[SourceSpec(id=SOURCE, kind="local", path="posts", loader="documents")],
        )
    )
    documents: list[Document] = []
    chunks: list[Chunk] = []
    mentions: list[Mention] = []
    for slug, letters in PLANTED_DOCUMENTS.items():
        doc_id = f"{PERSONA}:{SOURCE}:{slug}"
        documents.append(
            Document(
                id=doc_id,
                persona_id=PERSONA,
                source_id=SOURCE,
                title=slug,
                path=f"posts/{slug}.md",
                word_count=len(letters) * 4,
            )
        )
        for ordinal, letter in enumerate(letters):
            chunk_id = f"{doc_id}#{ordinal:04d}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    persona_id=PERSONA,
                    ordinal=ordinal,
                    text=f"passage {ordinal} about {NAMES[letter]}",
                    word_count=4,
                )
            )
            mentions.append(Mention(chunk_id=chunk_id, entity_id=f"entity:{letter}"))
    memory_store.upsert_documents(documents)
    memory_store.upsert_chunks(chunks, hash_embedder.embed_documents([c.text for c in chunks]))
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[
                Entity(id=f"entity:{key}", name=name, type="product") for key, name in NAMES.items()
            ],
            mentions=mentions,
        )
    )
    return memory_store


def test_the_sequences_come_off_the_store_in_the_documents_own_order(
    sequenced: InMemoryGraphStore,
) -> None:
    """``passage_sequences`` reads the chunker's ordinal, so the order is the document's."""
    sequences = passage_sequences(sequenced, PERSONA)

    assert [s.doc_id for s in sequences] == [
        f"{PERSONA}:{SOURCE}:doc-1",
        f"{PERSONA}:{SOURCE}:doc-2",
    ]
    assert [len(s.passages) for s in sequences] == [9, 2]
    assert [p.ordinal for p in sequences[0].passages] == list(range(9))
    assert [p.entities for p in sequences[0].passages] == [
        (f"entity:{letter}",) for letter in "ABCBADABC"
    ]


def test_the_store_backed_walk_is_the_hand_solved_one_and_stops_at_the_boundary(
    sequenced: InMemoryGraphStore,
) -> None:
    """The same answer as the hand-solved test, now read through the store, plus the boundary.

    doc-1 is [A][B][C][B][A][D][A][B][C] and doc-2 is [C][D], so the first-order network gains
    one edge, C -> D, and no transition ever crosses from doc-1's last passage to doc-2's first.
    Gamma to Delta is a dead end and Gamma is entered once more, which is exactly the boundary
    effect the module says is where the two walks part company.
    """
    simplicial, memory = highorder_reports(sequenced, PERSONA)

    plain = memory.first_order
    assert weights(plain)[("entity:C", "entity:D")] == 1
    assert not plain.has_edge("entity:C", "entity:C")
    assert memory.second_order.graph["documents"] == 2
    assert memory.second_order.graph["breaks"] == 0

    # Every passage names one entity, so the complex is 4 isolated 0-simplices: no passage put
    # two things together, and a hypergraph of singletons closes to nothing above dimension 0.
    assert [len(simplicial.complex_.simplices(dim)) for dim in range(3)] == [4, 0, 0]
    assert closure(simplicial.complex_).triangles == 0

    # doc-2's C -> D has no continuation, so the memory network is not strongly connected and the
    # undamped walk of the hand-solved test is unavailable here: the damped one is the default.
    with pytest.raises(ValueError, match=r"not strongly connected|no out-edge"):
        memory_stationary(memory.second_order, plain, damping=None)


# ----------------------------------------------------------------------------- the guide, the CLI


def test_the_guide_carries_the_chapters_cautions() -> None:
    section = next(
        (rules for heading, rules in READING_RULES if heading.startswith("High-order")), ()
    )
    names = [rule.name for rule in section]
    assert "A hypergraph is not a simplicial complex: closure is a claim" in names
    assert "A memory network's node is a transition, not an entity" in names
    assert "A first-order walk forgets where it came from; report both" in names
    assert "Consecutive passages are an ordering the corpus imposes, not a causal one" in names
    # The section count is test_sna_guide's to assert; this test owns only this section.
    assert len(names) == 4


def test_sna_highorder_prints_both_halves_of_chapter_34(
    cli_context: AppContext, sequenced: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The command over the planted documents, with both reports written out.

    Each passage there names one entity, so the complex is four isolated nodes -- no passage put
    two things together, which is itself a finding the frame has to state -- and the memory
    network is the hand-solved one plus doc-2's dead end. Both halves have to print their frame,
    their n, their null model and the chapter beside the numbers.
    """
    out = tmp_path / "highorder.md"
    as_json = tmp_path / "highorder.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "highorder",
            PERSONA,
            "--max-dim",
            "3",
            "--out",
            str(out),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "34.1" in text and "34.2" in text and "34.3" in text
    assert text.count("Sampling frame:") == 2
    assert "Null model: none" in text and "**Null model:**" in text
    assert "damping 0.85" in text
    # The walk table names entities by label, never by the id the store keys them on.
    assert "Beta" in text and "Delta" in text
    assert "entity:B" not in text

    data = json.loads(as_json.read_text(encoding="utf-8"))
    assert data["simplicial"]["implements"].startswith("Atlas ")
    assert data["simplicial"]["faces"]["0"] == 4
    assert data["simplicial"]["closure"]["triangles"] == 0
    assert data["memory"]["documents"] == 2
    assert data["memory"]["second_order"]["nodes"] == 7  # six from doc-1, plus Gamma -> Delta
    assert data["memory"]["walk"]["damping"] == 0.85
    assert {row["entity"] for row in data["memory"]["walk"]["entities"]} == {
        "entity:A",
        "entity:B",
        "entity:C",
        "entity:D",
    }


def test_sna_highorder_refuses_what_it_cannot_serve(
    cli_context: AppContext, sequenced: InMemoryGraphStore
) -> None:
    """Chapter 34's structures are read off the passage hypergraph, so `--network` has one value;
    a closure below dimension 0 is not a complex; and a walk needs a teleport a walker could use.
    """
    assert runner.invoke(app, ["sna", "highorder", PERSONA, "--network", "speakers"]).exit_code == 2
    assert runner.invoke(app, ["sna", "highorder", PERSONA, "--damping", "0"]).exit_code == 2
    assert runner.invoke(app, ["sna", "highorder", PERSONA, "--max-dim", "-1"]).exit_code == 2

    # --exact on a corpus whose memory network is not strongly connected: refused, not fudged.
    exact = runner.invoke(app, ["sna", "highorder", PERSONA, "--exact"])
    assert exact.exit_code == 2


def test_the_cli_defaults_for_the_high_order_command_match_the_module() -> None:
    """Same reason as the walks and roles guards: a signature default is evaluated at import
    time, so the CLI repeats the numbers rather than importing the module to read them."""
    assert HIGHORDER_MAX_DIM == DEFAULT_MAX_DIM
    assert HIGHORDER_DAMPING == DEFAULT_DAMPING
