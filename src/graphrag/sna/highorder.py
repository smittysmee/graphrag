"""High-order dynamics over a corpus (Atlas ch. 34): passages as simplices, and a walk that
remembers where it came from.

Every other network in this package is single-order: an edge joins two nodes and a walker on it
*"only knows"* the node it stands on. Chapter 34 says that can miss the story -- *"if you only
look at the structure you might miss part of the story. This is especially true if you're
interested in knowing how different agents act in it"* (p. 474) -- and gives three ways out. Two
are structural, one is algorithmic. This module builds one of each.

**§34.1, simplicial complexes.** A passage naming Alpha, Beta and Gamma is one relation between
three things, which :func:`graphrag.sna.layers.hypergraph` already keeps as a hyperedge. The
difference between that hyperedge and a *simplex* is downward closure: *"a simplex of four
logically also contains all lower-level simplices, which are taken into account in the
analysis"*, while *"a hyperedge with four nodes only contains those four nodes"* (§7.3, p. 112).
:func:`simplicial_complex` performs that closure, and everything §34.1 measures follows from it:
the generalized degree ``k_{d,m}``, *"the number of d dimensional simplices incident on an
m-face"* (p. 475), the incidence ``k_{d,d-1}`` that decides whether a complex is a manifold
(p. 476), and -- because the closure turns a claim into a number -- how many triangles of the
1-skeleton are *not* filled. That last one is §34.1's own warning made measurable: cliques to
simplices and simplices to cliques *"are not commutative"* (Figure 34.10, p. 481), so treating
every triangle of the entity network as a simplex would invent co-mentions the corpus never made.

**§34.2, memory in the structure.** *"If you want to really model the traveler's behavior, you
need some sort of memory, you need to know they arrived from u into v"* (p. 474). The section
gives two constructions: the High Order Network, which splits a node into one meta-node per
path that reaches it and mixes orders freely, and the **memory network** of Rosvall et al.,
where *"all nodes represent transitions of the same order"* and second-order dynamics are *"a
line graph"* of the original (pp. 483-484). :func:`second_order_network` builds the memory
network: its nodes are ordered entity transitions ``A -> B``, its edges are the observed paths
``A -> B -> C``, and the sequence they are read off is the passages of one document in reading
order. §34.2's own footnote applies to it: the adjacency matrix of a second-order memory network
*"is a non-backtracking matrix (Section 11.2), provided that the memory network has no
self-loops"*, which :func:`graphrag.sna.walks.non_backtracking_matrix` is -- with the one
difference stated in :func:`second_order_network`.

**§34.3, memory in the algorithm.** *"The alternative approach ... is to leave your structure
alone and to embed the high order logic directly into your algorithm"* (p. 484). Here the two
approaches meet: a memoryless walk on the memory network *is* the second-order walk (p. 484,
*"the simple memoryless Markov processes on the memory network describe the high-order
processes"*), so :func:`memory_stationary` runs §11.1's walk on both networks and folds the
second-order answer back onto entities. What comes out is the comparison §34.3 is about: where a
walker with memory spends its time, next to where a walker without memory does, and which
entities change rank between them.

**What makes the two walks agree, and what does not.** A walk reproduces the corpus's own
frequencies only when the chain it runs on is *flow-balanced* by the data: every state left as
often as it is entered. Those are two different conditions at the two orders, and the second is
much stronger. A document that ends on the entity it started from balances the **first-order**
chain, because then every entity is departed as often as it is arrived at. It does **not**
balance the **memory** chain, whose states are transitions: a document's first transition is
never reached as the second leg of a path and its last is never left as the first leg, and in
between a transition can be continued more often than it is entered. So a closed entity sequence
proves nothing about agreement. ``[A][B][A][B][A][C][B][A]`` closes on A, repeats nothing, and
still gives ``(A 3/7, B 3/7, C 1/7)`` without memory against ``(A 2/5, B 2/5, C 1/5)`` with it,
both undamped and with no teleport in sight.

That is not a defect in either walk: it is §34.3's point, that *"the next step is not dependent
exclusively on the fact we're in v"* (p. 474). It does mean the difference below has three
sources, and a reader should not attribute it to one: second-order flow imbalance **inside** a
document, the boundaries where documents start and stop, and -- when it is on -- the teleport,
which injects its mass over transitions in one walk and over entities in the other. The default
here is the **damped** walk (§11.1's teleport, which is what Rosvall et al. compare and what
PageRank is), the damping factor is printed beside every number, and ``damping=None`` asks for
the undamped π and raises when the chain has none.

**What chapter 34 describes and this module deliberately does not build.**

- *Third and higher orders (§34.2, p. 484).* *"You can model third order dynamics by making a
  more complicated version of a line graph, in which nodes stand in for paths of length three"*,
  and a memory network's defining property is that *"all nodes represent transitions of the same
  order"*. This module builds order two only. The reason is the section's own cost argument --
  each order multiplies the node count by the branching of the sequences, and the paper it cites
  stopped at five on data with longer dependencies because *"the increase in complexity simply
  wasn't worth it"* -- and a corpus-specific one: a third-order node is three consecutive
  passages that each named something and its edges need four, which on a chunked transcript is a
  thin sample before it is anything else.
  The construction generalises (:func:`_paths` would widen to four-passage windows) and the
  order is recorded on the graph as ``graph.graph["order"]`` so a later ticket can raise it
  without changing what a reader of an existing report was told.
- *High Order Networks (§34.2, Figures 34.11-34.12).* A HON mixes orders in one structure and is
  *"a bit more flexible"* than a memory network, at the cost the section names: 7 nodes and 6
  edges became 17 and 16, the result *"can become really unwieldy"*, and it fragments into five
  components where the original had one. A corpus entity network is already large; the memory
  network answers the same second-order question at a fixed order, and that is the trade the
  section tells you to make.
- *Synchronization, percolation and epidemics on the complex (§34.1).* Three dynamical processes
  the section explains *"briefly"* to *"give you a taste"*. They are simulations over a complex,
  with parameters -- a synchronization speed, two infection probabilities -- that no corpus
  states; spreading over this package's networks is ATL-20/ATL-22's chapter, not this one.
- *Manifolds and the growth models (§34.1, p. 479).* Growing a manifold by saturating links is a
  *generative* model (chapter 17's question), and a passage complex is not a manifold anyway:
  :func:`SimplicialReport` prints the incidence that proves it, because a pair of entities named
  together in twenty passages is a 1-face with incidence far above the 0 or 1 a manifold allows.
- *d-connected components (§34.1, p. 476).* Defined there to define manifolds. With no manifold
  to define, the notion has no consumer here; the 0-connected components are
  :mod:`graphrag.sna.paths` on the 1-skeleton and are printed by every ``analyze`` report.
- *The motif dictionary and the high-order cut (§34.3, p. 485).* ``φ(S, M)``, the normalized cut
  counting motifs instead of edges, is community discovery: it belongs with ATL-35's methods and
  ATL-41's motif census, and the community half of §34.3 -- Rosvall's map equation over a memory
  network -- belongs with Infomap in ATL-35. This module builds the walk and documents the
  community consequence rather than implementing a second partitioner.
- *Continuous-time and t-step transitions (§34.3, p. 486).* ``T_t = e^{-t D^{-1} L}`` is the same
  random walk with time made continuous. It is a one-liner over
  :func:`graphrag.sna.matrices.laplacian` and belongs to chapter 11's module, not here; nothing
  in this report asks a question about ``t``.
- *Tensor representations (§34.3, p. 487).* The section's own explanations are *"brief, but
  overly simplistic"* pointers to graph-matching papers. There is no corpus question here that a
  tensor answers and a second-order network does not.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import combinations
from typing import Any

import networkx as nx

from graphrag.graph.store import GraphStore
from graphrag.models import EntityChunk, EntityMention
from graphrag.sna.export import entity_passage_rows
from graphrag.sna.layers import Hypergraph, hyperedge_sizes, hypergraph
from graphrag.sna.stats import Summary
from graphrag.sna.walks import stationary_distribution

__all__ = [
    "DEFAULT_DAMPING",
    "DEFAULT_MAX_DIM",
    "MAX_FACES",
    "MAX_TRANSITIONS",
    "MAX_TRIANGLES",
    "RANK_TOLERANCE",
    "TRANSITION_SEP",
    "Closure",
    "Face",
    "MemoryReport",
    "MemoryWalk",
    "Passage",
    "PassageSequence",
    "RankMove",
    "SimplicialComplex",
    "SimplicialReport",
    "build_memory_report",
    "build_simplicial_report",
    "closure",
    "highorder_reports",
    "memory_payload",
    "memory_stationary",
    "passage_sequences",
    "render_memory",
    "render_simplicial",
    "second_order_network",
    "simplicial_clustering",
    "simplicial_complex",
    "simplicial_degree",
    "simplicial_degrees",
    "simplicial_incidence",
    "simplicial_payload",
    "split_transition",
    "transition_id",
    "transition_network",
]

#: One simplex: the sorted node ids it joins. A ``d``-simplex holds ``d + 1`` of them (§7.3).
Face = tuple[str, ...]

#: How far the downward closure goes by default. 3 keeps tetrahedra -- the largest simplex §7.3
#: draws -- and stops the closure of a passage naming forty entities from producing 91,390 faces
#: nobody will read. The report prints how many hyperedges were larger than this.
DEFAULT_MAX_DIM = 3
#: The most faces :func:`simplicial_complex` will materialise before refusing. The closure of a
#: hyperedge of size ``k`` has ``sum_{d<=max_dim} C(k, d+1)`` faces, which is a fourth-power
#: growth in ``k`` at ``max_dim=3``: one passage naming 60 entities is half a million faces on
#: its own.
MAX_FACES = 500_000
#: The most triangles :func:`closure` will enumerate. Triangle enumeration on the 1-skeleton is
#: the cost of the closure number, and a dense corpus skeleton has many more triangles than edges.
MAX_TRIANGLES = 500_000
#: The most distinct transitions a second-order network will hold. Its node count is the number
#: of ordered pairs the corpus put in consecutive passages, and its edge count grows faster
#: still, so a command over an unfiltered persona stops here rather than appearing to hang.
MAX_TRANSITIONS = 100_000
#: What separates the two halves of a second-order node id. A node of the memory network is a
#: transition, and its id says so on sight rather than needing the node attributes read.
TRANSITION_SEP = " -> "
#: The default damping factor of the walk (§11.1's teleport). Both walks use the same one, or
#: neither does; see :func:`memory_stationary` for why the default is not "no teleport".
DEFAULT_DAMPING = 0.85
#: How close two probabilities have to be to count as one rank. Neither walk is computed to
#: better than this -- the undamped one stops at a total-variation change of 1e-12 and the damped
#: one at networkx's own tolerance -- so a pair separated by less is a tie, and printing it as a
#: rank change would report the arithmetic rather than the corpus.
RANK_TOLERANCE = 1e-9
#: How many rows the rank-change table prints.
TOP_MOVES = 25
#: How many unfilled triangles the closure section lists.
TOP_OPEN = 15

SIMPLICIAL_FRAME = (
    "Read as a simplicial complex (Atlas §34.1, §7.3): each of {edges} {edge_noun}(s) is a "
    "simplex over the {nodes} {member_noun} it names, and the complex is the downward closure of "
    "those simplices up to dimension {max_dim} -- every sub-group of a {edge_noun}'s members is "
    "a face of it, which is exactly what a hypergraph does not give you (§7.3, p. 112). "
    "{truncated} Every caution about the network it came from holds here: a face exists because "
    "somebody wrote a {edge_noun} naming those {member_noun} together, so this describes what "
    "was written down and not what is true. Null model: none. A face is a deterministic "
    "function of the {edge_noun}s, so "
    "there is nothing to test against a rewiring; what carries a null here is the 1-skeleton, "
    "and `sna analyze` is where it is tested."
)
MEMORY_FRAME = (
    "Read as a second-order memory network (Atlas §34.2): a node is an ordered transition "
    "A -> B, not an entity, and an edge is a path A -> B -> C the corpus actually shows. The "
    "sequence is the passages of one document in reading order, and a document boundary ends a "
    "sequence: {documents} document(s) contributed {passages} passage(s) with at least one "
    "entity, {breaks} place(s) where the next passage was not the next one in the document "
    "(unmentioned or filtered out) broke the chain, and {occurrences} adjacency(ies) of "
    "consecutive passages produced {transitions} distinct transition(s) over {entities} "
    "entities. {carried} continuation(s) of the same entity into the next passage are not "
    "transitions and were dropped. One adjacency contributes a transition per pair of entities "
    "across it, so a rich passage weighs quadratically more than a sparse one -- ten entities "
    "beside ten is 100 transitions from a single adjacency, the same inflation a clique "
    "expansion applies to a large hyperedge (§7.3). Null model: none. Both walks below are "
    "deterministic "
    "functions of these sequences; what a null could test is whether the passage order matters "
    "at all, which would be a shuffle of the passages and is not run here."
)


# ----------------------------------------------------------------------------- §34.1 simplices


@dataclass(frozen=True)
class SimplicialComplex:
    """A set of simplices closed downward: every face of a simplex is in it (§7.3, §34.1).

    ``faces[d]`` holds the ``d``-simplices, sorted, each as the tuple of node ids it joins: a
    0-simplex is a node, a 1-simplex an edge, a 2-simplex a filled triangle, *"and you can see
    where this is going"* (§7.3, p. 112). ``facets`` are the simplices that are a face of no
    larger one -- *"the sequence of facets of a simplicial complex fully distinguishes it from
    any other"* (p. 113) -- and are computed before ``max_dim`` truncates anything, so they say
    what the corpus stated even where the closure was cut short.
    """

    persona_id: str
    member_noun: str
    edge_noun: str
    faces: Mapping[int, tuple[Face, ...]]
    facets: tuple[Face, ...]
    labels: Mapping[str, str] = field(default_factory=dict)
    max_dim: int = DEFAULT_MAX_DIM
    truncated: int = 0
    """How many hyperedges were larger than ``max_dim + 1`` nodes, so their top simplex is not
    in ``faces``. Their faces up to ``max_dim`` are."""
    frame: str = ""

    @property
    def dimension(self) -> int:
        """The complex's dimension: *"the largest dimension of the simplices it contains"* (§7.3).

        -1 for the empty complex, which has no simplices at all. Truncated by ``max_dim``, and
        ``truncated`` says whether that happened.
        """
        return max((d for d, faces in self.faces.items() if faces), default=-1)

    @property
    def pure(self) -> bool:
        """Whether every facet has the same dimension (§7.3, p. 113's *"pure simplicial complex"*).

        A corpus complex is pure only when every passage named the same number of entities,
        which is the uniform-hypergraph condition §7.3 says nothing about a passage fixes.
        """
        return len({len(facet) for facet in self.facets}) <= 1

    def simplices(self, dim: int) -> tuple[Face, ...]:
        """The ``dim``-simplices, or an empty tuple when the complex holds none."""
        return self.faces.get(dim, ())

    def skeleton(self) -> nx.Graph:
        """The 1-skeleton: the plain network you get *"if you were to ignore all simplices"* (§7.3).

        Nodes are the 0-simplices and edges the 1-simplices, unweighted, because a face either
        is in the complex or is not. This is the same edge set as
        :func:`graphrag.sna.layers.clique_expansion` of the hypergraph it came from, with the
        weights dropped -- which is §7.3's *"any network is a skeleton of one or more simplicial
        complexes"* read from the other end.
        """
        graph = nx.Graph()
        for (node,) in self.simplices(0):
            graph.add_node(node, label=self.labels.get(node, node), mode=self.member_noun)
        graph.add_edges_from(self.simplices(1))
        graph.graph.update(
            network="entities",
            persona_id=self.persona_id,
            unit="faces of the complex",
            frame=f"{self.frame} This is its 1-skeleton, unweighted.",
        )
        return graph


def simplicial_complex(
    hyper: Hypergraph, *, max_dim: int = DEFAULT_MAX_DIM, max_faces: int = MAX_FACES
) -> SimplicialComplex:
    """The downward closure of a hypergraph's hyperedges: every passage's entity set as a simplex.

    This is the operation §34.1 describes at the end of its first section -- *"you can make any
    arbitrary network into a simplicial complex ... consider every clique in the network as a
    simplex"* (p. 481) -- applied one step earlier, to the hyperedges rather than to the cliques
    of their expansion. Doing it from the hyperedges is what makes the result *the corpus's*
    complex: a passage that named three entities is a 2-simplex because a passage said so, while
    a triangle of the expanded entity network may be three separate passages that never once
    named all three together. §34.1 is explicit that the two are not the same object -- the
    operations are *"not commutative"*, and Figure 34.10 (p. 481) draws the clique of four that
    was really two 2-simplices -- and :func:`closure` is the count of how far apart they are here.

    ``max_dim`` cuts the closure off: with the default 3 a passage naming ten entities
    contributes its 10 nodes, 45 edges, 120 triangles and 210 tetrahedra but not its 9-simplex,
    and ``truncated`` counts the passages that happened to. Raises ``ValueError`` past
    ``max_faces`` rather than materialise a closure nobody can read; filter the corpus (by type,
    facet or window) or lower ``max_dim``.

    The complex is undefined on nothing in particular: a hypergraph with no hyperedges gives an
    empty complex of dimension -1, which is a fact about the filter that produced it and is
    reported rather than raised.
    """
    if max_dim < 0:
        msg = f"max_dim must be at least 0 -- a 0-simplex is a node -- got {max_dim}"
        raise ValueError(msg)
    groups = [tuple(sorted(set(hyper.members[edge]))) for edge in hyper.hyperedges]
    total = sum(_closure_size(len(group), max_dim) for group in groups)
    if total > max_faces:
        msg = (
            f"this closure would hold about {total:,} faces, past the {max_faces:,} limit: the "
            f"faces of a {hyper.edge_noun} naming k {hyper.member_noun} grow as k^(max_dim+1), "
            f"and the largest here names {max((len(g) for g in groups), default=0)}. Lower "
            "--max-dim, or filter the corpus first (--types, --facet, --since/--until)"
        )
        raise ValueError(msg)
    by_dim: dict[int, set[Face]] = {d: set() for d in range(max_dim + 1)}
    for group in groups:
        for dim in range(min(max_dim, len(group) - 1) + 1):
            by_dim[dim].update(combinations(group, dim + 1))
    truncated = sum(1 for group in groups if len(group) - 1 > max_dim)
    frame = SIMPLICIAL_FRAME.format(
        edges=len(hyper.hyperedges),
        nodes=len(hyper.nodes),
        edge_noun=hyper.edge_noun,
        member_noun=hyper.member_noun,
        max_dim=max_dim,
        truncated=(
            f"{truncated} {hyper.edge_noun}(s) named more than {max_dim + 1} {hyper.member_noun} "
            "and so are in the complex only through their faces of that size and below."
            if truncated
            else f"No {hyper.edge_noun} named more than {max_dim + 1} {hyper.member_noun}, so "
            "nothing was truncated."
        ),
    )
    return SimplicialComplex(
        persona_id=hyper.persona_id,
        member_noun=hyper.member_noun,
        edge_noun=hyper.edge_noun,
        faces={dim: tuple(sorted(faces)) for dim, faces in by_dim.items()},
        facets=_facets(groups),
        labels=dict(hyper.labels),
        max_dim=max_dim,
        truncated=truncated,
        frame=frame,
    )


def _closure_size(members: int, max_dim: int) -> int:
    """How many faces the downward closure of one hyperedge of ``members`` nodes produces."""
    return sum(_choose(members, dim + 1) for dim in range(min(max_dim, members - 1) + 1))


def _choose(n: int, k: int) -> int:
    """``C(n, k)``, spelled out so the guard above reads as the arithmetic §7.3 describes."""
    from math import comb

    return comb(n, k) if 0 <= k <= n else 0


def _facets(groups: Sequence[Face]) -> tuple[Face, ...]:
    """The maximal simplices: *"a simplex that is not a face of any other larger simplex"* (§7.3).

    Compared only against the groups that share a member, through a posting list, so a corpus of
    thousands of passages does not become a quadratic scan of all pairs.
    """
    unique = sorted({group for group in groups if group}, key=lambda g: (-len(g), g))
    postings: dict[str, list[int]] = {}
    kept: list[Face] = []
    for group in unique:
        members = set(group)
        candidates = min((postings.get(node, []) for node in group), key=len, default=[])
        if any(members < set(kept[index]) for index in candidates):
            continue
        for node in group:
            postings.setdefault(node, []).append(len(kept))
        kept.append(group)
    return tuple(sorted(kept))


def simplicial_degree(complex_: SimplicialComplex, face: str | Sequence[str], dim: int) -> int:
    """``k_{d,m}``: how many ``dim``-simplices are incident on ``face`` (§34.1, p. 475).

    *"With k_{d,m} we can indicate the number of d dimensional simplices incident on an m-face --
    with m < d"*, where ``m`` is this face's own dimension: one fewer than the number of nodes it
    holds. A node is an ``m = 0`` face, so ``simplicial_degree(cx, "alpha", 2)`` is the number of
    filled triangles Alpha sits in, and the book's own worked example is its Figure 34.2, where
    *"the k_{2,0} of node 4 is four"* and *"the k_{2,1} of edge (4, 8) is two"*.

    A single string is read as a node. The count is 0 for a face the complex does not hold, which
    is not the same as a face that holds no simplices and is why a report prints the two apart.
    Raises ``ValueError`` when ``dim`` is not above the face's own dimension -- ``k_{d,m}`` is
    defined for ``m < d``, and ``k_{2,2}`` would be "is this triangle itself", not a degree.
    """
    wanted = (face,) if isinstance(face, str) else tuple(sorted(set(face)))
    own = len(wanted) - 1
    if dim <= own:
        msg = (
            f"k_(d,m) is defined for m < d (§34.1): this face is an m={own} face, so d must be "
            f"above {own}, got d={dim}"
        )
        raise ValueError(msg)
    members = set(wanted)
    return sum(1 for simplex in complex_.simplices(dim) if members.issubset(simplex))


def simplicial_degrees(complex_: SimplicialComplex, dim: int, *, of: int = 0) -> dict[Face, int]:
    """Every ``k_{dim,of}`` at once: the generalized degree of each ``of``-face (§34.1, p. 475).

    ``of=0`` gives the distribution over nodes and ``of=1`` the distribution over edges, which is
    the pair §34.5's first exercise asks for. Faces of dimension ``of`` that are in no
    ``dim``-simplex are present with 0, so the distribution is over the complex's faces rather
    than over the ones that happened to score.
    """
    if of >= dim:
        msg = f"k_(d,m) is defined for m < d (§34.1): got d={dim}, m={of}"
        raise ValueError(msg)
    counts: dict[Face, int] = dict.fromkeys(complex_.simplices(of), 0)
    for simplex in complex_.simplices(dim):
        for sub in combinations(simplex, of + 1):
            if sub in counts:
                counts[sub] += 1
    return counts


def simplicial_incidence(complex_: SimplicialComplex, face: str | Sequence[str]) -> int:
    """``k_{d,d-1}``, the *"special generalized degree that we call 'incidence'"* (§34.1, p. 476).

    How many simplices one dimension above this face are incident on it: for an edge, how many
    filled triangles contain it. *"Incidence is important to define manifolds"* -- a manifold
    needs every face's incidence to be 0 or 1 -- so the maximum of this over the 1-faces is the
    one number that says a corpus complex is not a manifold, and the report prints it.

    A bare string is one node, as in :func:`simplicial_degree`, and its incidence is the number
    of edges it is in. Reading it as a sequence of its characters instead would silently answer a
    question about a face that does not exist, which is why both functions take the same type.
    """
    wanted = (face,) if isinstance(face, str) else tuple(face)
    return simplicial_degree(complex_, wanted, len(set(wanted)))


@dataclass(frozen=True)
class Closure:
    """How many triangles of the 1-skeleton the passages actually filled (§34.1, Figure 34.10)."""

    triangles: int
    """Triangles in the 1-skeleton: three nodes joined pairwise by 1-simplices."""
    filled: int
    """Of those, the ones that are also a 2-simplex -- one passage named all three."""
    unfilled: tuple[Face, ...]
    """The rest, sorted: three entities pairwise co-mentioned, never all three together."""

    @property
    def ratio(self) -> float:
        """``filled / triangles``. Undefined with no triangles; :func:`simplicial_clustering`
        raises there rather than return a 0 that reads like "nothing closes"."""
        return self.filled / self.triangles if self.triangles else 0.0


def closure(complex_: SimplicialComplex, *, max_triangles: int = MAX_TRIANGLES) -> Closure:
    """Which triangles of the 1-skeleton are 2-simplices and which are only triangles (§34.1).

    The chapter's caution, turned into a count. §34.1 ends its first section by noting that you
    *can* promote every clique of a network to a simplex, and immediately warns that this is not
    the inverse of flattening a complex: the two operations *"are not commutative. If you apply
    them one after the other, you're not going to go back to your original simplicial complex"*,
    and Figure 34.10 (p. 481) draws a four-clique that was two 2-simplices and not a 3-simplex.
    On a corpus the same gap has a plain reading: an unfilled triangle is three entities the
    corpus paired up in three different passages and never named together in one.

    Returns the triangle count, how many are filled, and the unfilled ones sorted. Raises
    ``ValueError`` past ``max_triangles``, because enumerating them is the cost here and a dense
    skeleton has far more triangles than edges.
    """
    skeleton = complex_.skeleton()
    neighbours = {node: set(skeleton.neighbors(node)) - {node} for node in skeleton}
    filled = set(complex_.simplices(2))
    found = 0
    unfilled: list[Face] = []
    for u, v in sorted(tuple(sorted(edge)) for edge in skeleton.edges()):
        for w in sorted(neighbours[u] & neighbours[v]):
            if w <= v:
                continue
            found += 1
            if found > max_triangles:
                msg = (
                    f"this 1-skeleton holds more than {max_triangles:,} triangles, so the "
                    "closure would cost more than it tells you. Filter the corpus first "
                    "(--types, --facet, --since/--until)"
                )
                raise ValueError(msg)
            triangle = (u, v, w)
            if triangle not in filled:
                unfilled.append(triangle)
    return Closure(triangles=found, filled=found - len(unfilled), unfilled=tuple(unfilled))


def simplicial_clustering(
    complex_: SimplicialComplex, *, max_triangles: int = MAX_TRIANGLES
) -> float:
    """The fraction of the 1-skeleton's triangles that are filled 2-simplices.

    **What the book defines and what this adds.** §34.1 gives the generalized degree a formula
    and gives the clique/simplex mismatch a figure, but it names no clustering coefficient for a
    complex. The definition used here is the standard *simplicial closure* ratio -- among the
    triples of nodes that are pairwise joined in the 1-skeleton, the share that also appear
    together in one simplex (Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie and
    Jon Kleinberg. Simplicial closure and higher-order link prediction. *PNAS*, 115(48):E11221,
    2018). It is the higher-order analogue of §12.1's clustering coefficient, which is the same
    ratio with "appear together in one simplex" replaced by "are joined by an edge", and it
    measures exactly what Figure 34.10 warns about.

    Undefined on a skeleton with no triangles -- there is no fraction of nothing -- and raises
    ``ValueError`` there instead of returning 0, because 0 would read as "nothing closes".
    """
    result = closure(complex_, max_triangles=max_triangles)
    if not result.triangles:
        msg = (
            "simplicial clustering is undefined: the 1-skeleton holds no triangle, so there is "
            "no triple that could have been filled (§34.1)"
        )
        raise ValueError(msg)
    return result.ratio


# ----------------------------------------------------------------------- §34.1 report


@dataclass(frozen=True)
class SimplicialReport:
    """Everything one simplicial pass produced, before it is rendered."""

    persona_id: str
    complex_: SimplicialComplex
    closure: Closure
    node_degrees: dict[Face, int]
    """``k_{2,0}`` per node: how many filled triangles each entity sits in."""
    edge_degrees: dict[Face, int]
    """``k_{2,1}`` per edge: how many filled triangles each co-mentioned pair sits in."""
    sizes: Summary | None = None
    """The hyperedge-size distribution the complex was closed from, or ``None`` with no passages."""
    generated_at: str = ""

    def label(self, node: str) -> str:
        return self.complex_.labels.get(node, node)


def build_simplicial_report(
    hyper: Hypergraph, *, max_dim: int = DEFAULT_MAX_DIM, max_faces: int = MAX_FACES
) -> SimplicialReport:
    """Close a hypergraph into a complex and measure everything §34.1 measures on it."""
    complex_ = simplicial_complex(hyper, max_dim=max_dim, max_faces=max_faces)
    return SimplicialReport(
        persona_id=hyper.persona_id,
        complex_=complex_,
        closure=closure(complex_),
        node_degrees=simplicial_degrees(complex_, 2, of=0),
        edge_degrees=simplicial_degrees(complex_, 2, of=1),
        sizes=hyperedge_sizes(hyper) if hyper.hyperedges else None,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_simplicial(report: SimplicialReport) -> list[str]:
    """The §34.1 half of the report as markdown lines."""
    complex_ = report.complex_
    incidences = report.edge_degrees
    worst = max(incidences.values(), default=0)
    lines = [
        f"## High-order structure: {report.persona_id} (Atlas §34.1)",
        "",
        "**Implements:** *The Atlas for the Aspiring Network Scientist*, §34.1 (high order with "
        "simplicial complexes, pp. 475-481), on the terminology of §7.3 (pp. 112-113).",
        "",
        f"**Sampling frame:** {complex_.frame}",
        "",
        f"**n:** {len(complex_.simplices(0)):,} node(s), {len(complex_.facets):,} facet(s), "
        f"dimension {complex_.dimension} (truncated at {complex_.max_dim}), "
        f"{'pure' if complex_.pure else 'not pure'}.",
        "",
        "### Faces by dimension",
        "",
        "A *d*-simplex joins *d+1* nodes, and the complex holds every face of every simplex in "
        "it -- that downward closure is the whole difference between this and the hypergraph it "
        "was built from (§7.3, p. 112).",
        "",
    ]
    lines += _rows(
        ["dimension", "joins", "what it is", "count"],
        [
            [
                str(dim),
                f"{dim + 1} node(s)",
                ["node", "edge", "filled triangle", "tetrahedron"][dim]
                if dim < 4
                else f"{dim}-simplex",
                f"{len(complex_.simplices(dim)):,}",
            ]
            for dim in range(complex_.max_dim + 1)
        ],
    )
    if report.sizes is not None:
        lines += [
            f"The {complex_.edge_noun}s they were closed from name a mean of "
            f"{report.sizes.mean:.2f} {complex_.member_noun} (median "
            f"{report.sizes.median:.1f}, max {report.sizes.maximum:.0f}); a "
            f"{complex_.edge_noun} naming two contributes no triangle at all, which is why the "
            "counts above fall away so fast.",
            "",
        ]
    lines += [
        "### Generalized degree (§34.1, p. 475)",
        "",
        '`k_(d,m)` is *"the number of d dimensional simplices incident on an m-face"*. The two '
        "the chapter works through are `k_(2,0)`, how many filled triangles a node is in, and "
        "`k_(2,1)`, how many a given edge is in.",
        "",
    ]
    lines += _rows(
        ["rank", complex_.member_noun, "k(2,0): filled triangles it is in"],
        [
            [str(rank + 1), report.label(face[0]), f"{count:,}"]
            for rank, (face, count) in enumerate(_top(report.node_degrees, TOP_MOVES))
        ]
        or [["-", "no node is in a filled triangle", "0"]],
    )
    lines += [
        f"**Incidence** (`k_(d,d-1)`, p. 476): the highest any 1-face reaches is {worst:,}. A "
        "manifold needs every face's incidence to be 0 or 1, so this complex is "
        f"{'not a manifold' if worst > 1 else 'compatible with a manifold on this count alone'}"
        " -- which is expected and not a defect: a pair of entities named together in many "
        "passages is a face in many triangles. The growth models of §34.1 (p. 479) need a "
        "manifold and are therefore not run.",
        "",
        "### Closure: which triangles the passages actually filled",
        "",
        "§34.1 warns that promoting cliques to simplices and flattening simplices to cliques "
        '*"are not commutative"* (Figure 34.10, p. 481). Here that gap is a count: an unfilled '
        "triangle is three entities this corpus paired up in three separate passages and never "
        "once named together.",
        "",
    ]
    closed = report.closure
    ratio = f"{closed.ratio:.3f}" if closed.triangles else "undefined"
    lines += _rows(
        ["triangles in the 1-skeleton", "filled (2-simplices)", "unfilled", "closure"],
        [[f"{closed.triangles:,}", f"{closed.filled:,}", f"{len(closed.unfilled):,}", ratio]],
    )
    if closed.unfilled:
        lines += [
            f"The first {min(TOP_OPEN, len(closed.unfilled))} unfilled triangle(s), which is "
            "what an analysis that treated every triangle of the entity network as a simplex "
            "would have invented:",
            "",
        ]
        lines += [
            "- " + " + ".join(report.label(node) for node in triangle)
            for triangle in closed.unfilled[:TOP_OPEN]
        ]
        lines += [""]
    return lines


def _top(degrees: Mapping[Face, int], limit: int) -> list[tuple[Face, int]]:
    """The highest-scoring faces, ties broken by id so two runs print the same table."""
    ranked = [row for row in degrees.items() if row[1] > 0]
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]


def simplicial_payload(report: SimplicialReport) -> dict[str, Any]:
    """The §34.1 half as plain JSON-able data."""
    complex_ = report.complex_
    return {
        "persona_id": report.persona_id,
        "generated_at": report.generated_at,
        "implements": "Atlas §34.1 (simplicial complexes), on §7.3's terminology",
        "frame": complex_.frame,
        "null_model": "none: a face is a deterministic function of the passages",
        "max_dim": complex_.max_dim,
        "dimension": complex_.dimension,
        "pure": complex_.pure,
        "truncated_hyperedges": complex_.truncated,
        "faces": {str(dim): len(complex_.simplices(dim)) for dim in range(complex_.max_dim + 1)},
        "facets": len(complex_.facets),
        "max_incidence": max(report.edge_degrees.values(), default=0),
        "closure": {
            "triangles": report.closure.triangles,
            "filled": report.closure.filled,
            "unfilled": len(report.closure.unfilled),
            "ratio": report.closure.ratio if report.closure.triangles else None,
            "definition": (
                "filled triangles over triangles of the 1-skeleton (simplicial closure; not the "
                "book's own formula -- §34.1 gives the mismatch a figure, not a coefficient)"
            ),
            "examples": [list(triangle) for triangle in report.closure.unfilled[:TOP_OPEN]],
        },
        "simplicial_degree": [
            {"node": face[0], "label": report.label(face[0]), "k_2_0": count}
            for face, count in _top(report.node_degrees, TOP_MOVES)
        ],
    }


# ------------------------------------------------------------------- §34.2 memory in the structure


@dataclass(frozen=True)
class Passage:
    """One passage of one document, and the entities it named, in the document's own order."""

    chunk_id: str
    ordinal: int
    entities: tuple[str, ...]


@dataclass(frozen=True)
class PassageSequence:
    """One document's passages in reading order: the sequence the memory network is read off.

    Only the passages that named at least one entity are held. ``ordinal`` is the chunker's
    position of the passage in its document, so two passages are *consecutive* exactly when
    their ordinals differ by one -- a passage that named nothing, or that a filter removed, is a
    gap and breaks the chain rather than being silently bridged.
    """

    doc_id: str
    passages: tuple[Passage, ...]


def transition_id(source: str, target: str) -> str:
    """The id of the memory network's node for the transition ``source -> target`` (§34.2)."""
    return f"{source}{TRANSITION_SEP}{target}"


def split_transition(node: str) -> tuple[str, str]:
    """The two entities behind a memory-network node id. The inverse of :func:`transition_id`."""
    source, _, target = node.partition(TRANSITION_SEP)
    return source, target


def passage_sequences(
    store: GraphStore,
    persona_id: str,
    *,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    batch: int = 5_000,
) -> list[PassageSequence]:
    """The corpus as one ordered entity sequence per document (§34.2's *"sequences"* input).

    The same rows the entity network is projected from -- :func:`entity_passage_rows`, so every
    filter means what it means everywhere else -- put back in the order the document was written
    in. The order itself comes from the chunker's ``ordinal``, read through
    ``GraphStore.get_chunks`` in batches of ``batch`` ids so one command is a handful of
    statements rather than one per passage.

    What an order is here, and what it is not: passage *i+1* follows passage *i* because a
    transcript ran that way or a thread was posted that way. Nothing makes it a cause, and
    nothing makes the two ends of a document adjacent to anything outside it.
    """
    rows: list[EntityChunk] | list[EntityMention] = entity_passage_rows(
        store,
        persona_id,
        source_id,
        types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        where=where,
    )
    named: dict[str, set[str]] = {}
    for row in rows:
        named.setdefault(row.chunk_id, set()).add(row.entity_id)
    chunk_ids = sorted(named)
    positions: dict[str, tuple[str, int]] = {}
    for start in range(0, len(chunk_ids), max(batch, 1)):
        for chunk in store.get_chunks(chunk_ids[start : start + max(batch, 1)]):
            positions[chunk.id] = (chunk.doc_id, chunk.ordinal)
    by_document: dict[str, list[Passage]] = {}
    for chunk_id in chunk_ids:
        placed = positions.get(chunk_id)
        if placed is None:
            continue
        doc_id, ordinal = placed
        by_document.setdefault(doc_id, []).append(
            Passage(chunk_id=chunk_id, ordinal=ordinal, entities=tuple(sorted(named[chunk_id])))
        )
    return [
        PassageSequence(
            doc_id=doc_id,
            passages=tuple(sorted(passages, key=lambda p: (p.ordinal, p.chunk_id))),
        )
        for doc_id, passages in sorted(by_document.items())
    ]


def _steps(sequences: Iterable[PassageSequence]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Every observed transition, with the counters a frame has to print.

    One entry per (passage *i*, passage *i+1*, entity of *i*, entity of *i+1*) with the two
    entities different. ``carried`` counts the pairs dropped for being the same entity twice --
    a thing mentioned again in the next passage is a continuation, not a step from one thing to
    another -- and ``breaks`` counts the places where the next passage held nothing to step to.
    """
    steps: list[tuple[str, str]] = []
    counts = {"documents": 0, "passages": 0, "breaks": 0, "occurrences": 0, "carried": 0}
    for sequence in sequences:
        counts["documents"] += 1
        counts["passages"] += len(sequence.passages)
        for first, second in zip(sequence.passages, sequence.passages[1:], strict=False):
            if second.ordinal - first.ordinal != 1:
                counts["breaks"] += 1
                continue
            counts["occurrences"] += 1
            for source in first.entities:
                for target in second.entities:
                    if source == target:
                        counts["carried"] += 1
                    else:
                        steps.append((source, target))
    return steps, counts


def _paths(sequences: Iterable[PassageSequence]) -> Iterator[tuple[str, str, str]]:
    """Every observed ``A -> B -> C``: three consecutive passages, one entity taken from each.

    A generator rather than a list: a passage naming ten entities beside two others naming ten
    contributes a thousand paths on its own, and they are only ever counted into edge weights.
    """
    for sequence in sequences:
        passages = sequence.passages
        for first, second, third in zip(passages, passages[1:], passages[2:], strict=False):
            if second.ordinal - first.ordinal != 1 or third.ordinal - second.ordinal != 1:
                continue
            for a in first.entities:
                for b in second.entities:
                    if a == b:
                        continue
                    for c in third.entities:
                        if b != c:
                            yield (a, b, c)


def transition_network(
    sequences: Sequence[PassageSequence], *, labels: Mapping[str, str] | None = None
) -> nx.DiGraph:
    """The **first-order** network the memory network is the second order of (§34.2).

    Nodes are entities, and a directed edge ``A -> B`` weighs how many times the corpus named A
    in a passage and B in the next one. This is not the co-mention network: that one joins
    entities named in the *same* passage and has no direction, while this one is the one-step
    transition table a memoryless walker reads -- *"all we can say is that all steps from the
    central node are equally likely"* (p. 474). Both walks in :func:`memory_stationary` need it,
    one to run on and one to be compared against.
    """
    steps, counts = _steps(sequences)
    graph = nx.DiGraph()
    for source, target in steps:
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)
    for node in graph:
        graph.nodes[node]["label"] = (labels or {}).get(node, node)
    graph.graph.update(
        network="entity-transitions",
        unit="consecutive passages",
        order=1,
        frame=MEMORY_FRAME.format(
            entities=graph.number_of_nodes(),
            transitions=graph.number_of_edges(),
            **counts,
        ),
        **counts,
    )
    return graph


def second_order_network(
    sequences: Sequence[PassageSequence],
    *,
    labels: Mapping[str, str] | None = None,
    max_transitions: int = MAX_TRANSITIONS,
) -> nx.DiGraph:
    """The memory network of §34.2: a node is a transition, an edge is a path.

    *"To model second-order dynamics we can create a line graph ... The edges of the original
    graph become the nodes of the network and they are connected to each other if the original
    edges shared a node"* (pp. 483-484). Here the original edges are the transitions
    :func:`transition_network` holds, so a node of this graph is ``A -> B`` and an edge joins
    ``A -> B`` to ``B -> C``, weighted by how many times three consecutive passages of one
    document actually showed that path. The weights are the point: a line graph would join every
    ``A -> B`` to every ``B -> C``, and what makes this a *memory* network is that it only joins
    the ones the corpus walked, in the proportion it walked them.

    Two consequences of that, both the section's own. First, this is where §34.2's remark about
    §11.2 lands: *"the adjacency matrix of a memory network of second order is a non-backtracking
    matrix ... provided that the memory network has no self-loops"*. Ours is not quite
    :func:`graphrag.sna.walks.non_backtracking_matrix`, and in a way worth knowing: Hashimoto's
    operator forbids ``A -> B -> A`` by construction, while a corpus really does return to a
    thing after mentioning another, so those edges are kept. Delete them and this is exactly the
    non-backtracking operator restricted to the observed transitions. Second, *"HON networks tend
    to transform into weakly connected graphs, or even not connected"* (p. 483) -- the same is
    true here, which is why :func:`memory_stationary` damps by default.

    Every observed transition is a node even when no path continued it, so the last transition of
    a document appears with no out-edge. That is the data, and dropping it would quietly shorten
    every document by one passage. Raises ``ValueError`` past ``max_transitions`` distinct
    transitions.
    """
    steps, counts = _steps(sequences)
    distinct = set(steps)
    if len(distinct) > max_transitions:
        msg = (
            f"this corpus produces {len(distinct):,} distinct transitions, past the "
            f"{max_transitions:,} limit: a second-order network has one node per ordered pair of "
            "entities in consecutive passages and more edges still. Filter first (--types, "
            "--facet, --since/--until)"
        )
        raise ValueError(msg)
    graph = nx.DiGraph()
    occurrences: dict[tuple[str, str], int] = {}
    for step in steps:
        occurrences[step] = occurrences.get(step, 0) + 1
    known = labels or {}
    for (source, target), count in sorted(occurrences.items()):
        graph.add_node(
            transition_id(source, target),
            source=source,
            target=target,
            label=f"{known.get(source, source)}{TRANSITION_SEP}{known.get(target, target)}",
            occurrences=count,
        )
    paths = 0
    for a, b, c in _paths(sequences):
        paths += 1
        u, v = transition_id(a, b), transition_id(b, c)
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
        else:
            graph.add_edge(u, v, weight=1)
    graph.graph.update(
        network="entity-transitions",
        unit="observed A -> B -> C paths",
        order=2,
        paths=paths,
        frame=MEMORY_FRAME.format(
            entities=len({node for step in distinct for node in step}),
            transitions=len(distinct),
            **counts,
        ),
        **counts,
    )
    return graph


# ------------------------------------------------------------------- §34.3 memory in the algorithm


@dataclass(frozen=True)
class RankMove:
    """One entity, where each of the two walks puts it, and how far apart that is (§34.3)."""

    entity: str
    label: str
    first_order: float
    memory: float
    first_order_rank: int
    memory_rank: int

    @property
    def moved(self) -> int:
        """Places gained by remembering: positive when the memory walk ranks it higher."""
        return self.first_order_rank - self.memory_rank


@dataclass(frozen=True)
class MemoryWalk:
    """Where a walker spends its time with and without a memory, over the same entities (§34.3)."""

    first_order: dict[str, float]
    memory: dict[str, float]
    moves: tuple[RankMove, ...]
    damping: float | None
    method: str
    """Which walk was run, in words, because the answer is a different quantity for each."""

    @property
    def moved(self) -> tuple[RankMove, ...]:
        """The entities whose rank changed, furthest first."""
        return tuple(sorted(self.moves, key=lambda m: (-abs(m.moved), -m.memory, m.entity)))


def memory_stationary(
    second_order: nx.DiGraph,
    first_order: nx.DiGraph,
    *,
    damping: float | None = DEFAULT_DAMPING,
) -> MemoryWalk:
    """π of the walker with a memory, folded back onto entities, beside π of the one without.

    §34.3's question, in the form §34.2 makes available: *"once you have built your memory
    network with the desired order, then the simple memoryless Markov processes on the memory
    network describe the high-order processes"* (p. 484). So this runs the ordinary §11.1 walk on
    both graphs and folds the second-order answer back -- the walker's probability of *standing
    on* entity ``v`` is the total probability of standing on any transition that ends in ``v``,
    because a memory node ``u -> v`` means "in v, having come from u".

    **Damping, and why it is on by default.** With ``damping=None`` both walks are the plain
    stationary distribution, and both raise when the chain has none: a node with no out-edge or a
    graph that is not strongly connected, which §34.2 warns a memory network usually is not
    (*"HON networks tend to transform into weakly connected graphs, or even not connected"*,
    p. 483). With a damping factor both are the teleporting walk §11.1 calls *"the bells and
    whistles"* and PageRank is -- the walk Rosvall et al. compare across orders, and the one that
    always exists. Both walks are damped or neither is, because the two numbers are only
    comparable when they are the same quantity, and the factor is reported beside them.

    **Read the difference, not the level.** The two agree only when the *memory* chain is
    flow-balanced by the corpus -- every transition continued as often as it is entered -- which
    a document that closes on the entity it opened with does **not** give you: that closes the
    first-order chain, while the memory chain still opens on a transition nothing reached and
    ends on one nothing continued. So a difference here has three possible sources and is worth
    attributing to none of them on sight: second-order imbalance inside a document, the document
    boundaries, and the teleport, which injects its mass over transitions in one walk and over
    entities in the other. A rank change says the corpus ordered its passages in a way that
    first-order weights cannot express; it does not say the memory walk is more accurate about
    anything. Ranks are assigned with a tolerance (:data:`RANK_TOLERANCE`), because two
    distributions this close are two ways of computing one number and a tie broken at the
    fifteenth decimal is not a finding.

    Raises ``ValueError`` when either graph has no edges: there is nothing for a walker to do.
    """
    if not first_order.number_of_edges():
        msg = (
            "no first-order transitions: no document has two consecutive passages that name two "
            "different entities, so neither walk is defined (§34.2)"
        )
        raise ValueError(msg)
    if not second_order.number_of_edges():
        msg = (
            "no second-order paths: no document has three consecutive passages naming entities, "
            "so the memory network has nodes but no edges and its walk is not defined (§34.2)"
        )
        raise ValueError(msg)
    plain = _walk(first_order, damping)
    upper = _walk(second_order, damping)
    folded: dict[str, float] = dict.fromkeys(first_order.nodes, 0.0)
    for node, value in upper.items():
        target = str(second_order.nodes[node].get("target", split_transition(node)[1]))
        folded[target] = folded.get(target, 0.0) + value
    entities = sorted(set(plain) | set(folded))
    plain = {entity: plain.get(entity, 0.0) for entity in entities}
    folded = {entity: folded.get(entity, 0.0) for entity in entities}
    ranks_plain = _ranks(plain)
    ranks_memory = _ranks(folded)
    labels = {node: str(first_order.nodes[node].get("label", node)) for node in first_order}
    moves = tuple(
        RankMove(
            entity=entity,
            label=labels.get(entity, entity),
            first_order=plain[entity],
            memory=folded[entity],
            first_order_rank=ranks_plain[entity],
            memory_rank=ranks_memory[entity],
        )
        for entity in entities
    )
    method = (
        "the plain stationary distribution of §11.1, no teleport"
        if damping is None
        else f"the damped (teleporting) walk of §11.1, damping {damping:g}, on both networks"
    )
    return MemoryWalk(first_order=plain, memory=folded, moves=moves, damping=damping, method=method)


def _walk(graph: nx.DiGraph, damping: float | None) -> dict[str, float]:
    """π on one directed graph, damped or not, as a dict over its nodes."""
    if damping is None:
        values, order = stationary_distribution(graph, weight="weight")
        return {str(node): float(value) for node, value in zip(order, values, strict=True)}
    if not 0.0 < damping <= 1.0:
        msg = f"damping must be above 0 and at most 1 (§11.1's teleport), got {damping}"
        raise ValueError(msg)
    scores = nx.pagerank(graph, alpha=damping, weight="weight")
    return {str(node): float(value) for node, value in scores.items()}


def _ranks(values: Mapping[str, float], tolerance: float = RANK_TOLERANCE) -> dict[str, int]:
    """Competition ranks, 1 highest, ties sharing the better rank so a tie is not a move.

    Two probabilities within ``tolerance`` are one rank. Without that, a pair of entities a
    symmetric corpus puts at exactly the same probability is separated by whatever the power
    iteration left in the last bits, and the report prints a rank change that is a property of
    floating-point arithmetic rather than of the corpus.
    """
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, int] = {}
    previous: float | None = None
    rank = 0
    for index, (node, value) in enumerate(ordered, start=1):
        if previous is None or value < previous - tolerance:
            rank = index
            previous = value
        ranks[node] = rank
    return ranks


@dataclass(frozen=True)
class MemoryReport:
    """Everything one §34.2/§34.3 pass produced, before it is rendered."""

    persona_id: str
    first_order: nx.DiGraph
    second_order: nx.DiGraph
    walk: MemoryWalk
    sequences: int
    generated_at: str = ""


def build_memory_report(
    sequences: Sequence[PassageSequence],
    *,
    persona_id: str = "",
    labels: Mapping[str, str] | None = None,
    damping: float | None = DEFAULT_DAMPING,
    max_transitions: int = MAX_TRANSITIONS,
) -> MemoryReport:
    """Build both networks from the document sequences and run both walks over them."""
    plain = transition_network(sequences, labels=labels)
    memory = second_order_network(sequences, labels=labels, max_transitions=max_transitions)
    return MemoryReport(
        persona_id=persona_id,
        first_order=plain,
        second_order=memory,
        walk=memory_stationary(memory, plain, damping=damping),
        sequences=len(sequences),
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def render_memory(report: MemoryReport) -> list[str]:
    """The §34.2/§34.3 half of the report as markdown lines."""
    plain, memory, walk = report.first_order, report.second_order, report.walk
    components = nx.number_weakly_connected_components(memory)
    strong = nx.is_strongly_connected(memory) if memory.number_of_nodes() else False
    lines = [
        f"## High-order dynamics: {report.persona_id} (Atlas §34.2, §34.3)",
        "",
        "**Implements:** *The Atlas for the Aspiring Network Scientist*, §34.2 (embedding memory "
        "into the structure, pp. 481-484) and §34.3 (embedding it into the algorithm, "
        "pp. 484-487), over §11.1's random walk.",
        "",
        f"**Sampling frame:** {memory.graph.get('frame', 'not recorded')}",
        "",
        f"**n:** {plain.number_of_nodes():,} entities and {plain.number_of_edges():,} "
        f"transitions first-order; {memory.number_of_nodes():,} transitions and "
        f"{memory.number_of_edges():,} paths second-order, in {components:,} weakly connected "
        f"component(s), {'strongly connected' if strong else 'not strongly connected'}.",
        "",
        "### The memory network (§34.2)",
        "",
        "A node here is a transition, not an entity: `A -> B` means *in B, having arrived from "
        "A*. §34.2 builds it as a line graph of the first-order network, and the edges are the "
        "`A -> B -> C` paths the corpus actually shows, weighted by how often -- which is the "
        "whole difference between this and the plain line graph. The book's own note applies: "
        "the adjacency matrix of a second-order memory network is a non-backtracking matrix "
        "(§11.2) once the backtracking steps are removed; here they are kept, because a corpus "
        "does come back to a thing after naming another.",
        "",
    ]
    lines += _rows(
        ["", "nodes", "edges", "what a node is", "what an edge is"],
        [
            [
                "first order",
                f"{plain.number_of_nodes():,}",
                f"{plain.number_of_edges():,}",
                "an entity",
                "A in a passage, B in the next",
            ],
            [
                "second order",
                f"{memory.number_of_nodes():,}",
                f"{memory.number_of_edges():,}",
                "a transition A -> B",
                "an observed path A -> B -> C",
            ],
        ],
    )
    lines += [
        "### Where the two walkers spend their time (§34.3)",
        "",
        f"Both distributions are {walk.method}. The second-order one is folded back onto "
        "entities by summing over the transitions that end in each: standing on `u -> v` is "
        "standing on v. **Read the difference, not the level.** The two coincide only when the "
        "memory network's own flow is balanced by the corpus -- every transition continued as "
        "often as it is entered -- which even a document that ends on the entity it began with "
        "does not give, because its first transition is reached by nothing and its last "
        "continues into nothing. A difference therefore has three sources and belongs to none "
        "of them on sight: second-order imbalance inside a document, the document boundaries, "
        "and the teleport, which puts its mass on transitions in one walk and on entities in "
        "the other.",
        "",
        "**Null model:** none. Both walks are deterministic functions of the passage order; a "
        "null for this would shuffle the passages within their documents, which is a different "
        "question (is the order informative at all?) and is not run here.",
        "",
    ]
    moved = sorted(walk.moves, key=lambda m: (-m.memory, m.entity))[:TOP_MOVES]
    lines += _rows(
        ["entity", "first-order π", "memory π", "rank", "with memory", "moved"],
        [
            [
                move.label,
                f"{move.first_order:.4f}",
                f"{move.memory:.4f}",
                str(move.first_order_rank),
                str(move.memory_rank),
                f"{move.moved:+d}",
            ]
            for move in moved
        ]
        or [["this corpus put no entity in a transition", "", "", "", "", ""]],
    )
    lines += [
        f"The {min(TOP_MOVES, len(walk.moves)):,} highest by the memory walk, of "
        f"{len(walk.moves):,}. "
        f"{sum(1 for move in walk.moves if move.moved):,} of {len(walk.moves):,} entities change "
        "rank. What a change says is that the corpus ordered its passages in a way first-order "
        "weights cannot express; it does not say the memory walk is more accurate about "
        "anything, and neither ranking is evidence about the world outside what was written "
        "down.",
        "",
        "Community discovery over this network -- the other half of §34.3's effect, and Rosvall "
        "et al.'s own result -- is not run here: the map equation over a memory network is "
        "Infomap's, and it belongs with the other partitioners rather than in this report.",
        "",
    ]
    return lines


def memory_payload(report: MemoryReport) -> dict[str, Any]:
    """The §34.2/§34.3 half as plain JSON-able data."""
    walk = report.walk
    return {
        "persona_id": report.persona_id,
        "generated_at": report.generated_at,
        "implements": "Atlas §34.2 (memory network) and §34.3 (memory-aware walk)",
        "frame": report.second_order.graph.get("frame", ""),
        "null_model": "none: both walks are deterministic functions of the passage order",
        "documents": report.sequences,
        "first_order": {
            "nodes": report.first_order.number_of_nodes(),
            "edges": report.first_order.number_of_edges(),
        },
        "second_order": {
            "nodes": report.second_order.number_of_nodes(),
            "edges": report.second_order.number_of_edges(),
            "paths": int(report.second_order.graph.get("paths", 0)),
            "weakly_connected_components": nx.number_weakly_connected_components(
                report.second_order
            ),
        },
        "walk": {
            "method": walk.method,
            "damping": walk.damping,
            "entities": [
                {
                    "entity": move.entity,
                    "label": move.label,
                    "first_order": move.first_order,
                    "memory": move.memory,
                    "first_order_rank": move.first_order_rank,
                    "memory_rank": move.memory_rank,
                    "moved": move.moved,
                }
                for move in walk.moved
            ],
        },
    }


# ----------------------------------------------------------------------------- the whole command


def highorder_reports(
    store: GraphStore,
    persona_id: str,
    *,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    max_dim: int = DEFAULT_MAX_DIM,
    damping: float | None = DEFAULT_DAMPING,
) -> tuple[SimplicialReport, MemoryReport]:
    """Both halves of chapter 34 over one persona, from one set of filters.

    The two read the same passages through the same filters and answer different questions about
    them: the complex asks what was named *together*, the memory network what was named *after*.
    """
    hyper = hypergraph(
        store,
        persona_id,
        source_id=source_id,
        types=types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        where=where,
    )
    sequences = passage_sequences(
        store,
        persona_id,
        source_id=source_id,
        types=types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        where=where,
    )
    return (
        build_simplicial_report(hyper, max_dim=max_dim),
        build_memory_report(sequences, persona_id=persona_id, labels=hyper.labels, damping=damping),
    )
