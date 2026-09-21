"""The small shapes a network is built out of, and how often they are there (Atlas ch. 41).

Chapter 41 asks one question in two directions. Top-down: *"Networks will tend to have
overexpressed connection patterns. Nodes and edges will form different shapes much more -- or
less -- often than what you'd expect if the connections were random"* (p. 589), so take a shape
you already care about and ask whether this network has more of it than chance. Bottom-up:
*"Sometimes, you need a 'bottom-up' approach: you want an algorithm telling you the frequencies
of all possible simple network motifs"* (p. 596), so let the data name the shapes. Both are here,
and both are counts of subgraphs rather than measurements of nodes.

**§41.2, and what a motif count is worth on its own: nothing.** The chapter gives the procedure
in four steps (p. 590): *"First, you count how many times each motif appears in your network.
Second, you define a null model of your network, keeping its relevant properties fixed -- maybe
just the degree distribution. Third, you count the expected number of occurrences of the motifs
in the null model. Finally, you compare with your observation, so that you can build an idea of
the statistical significance of the motif."* Steps one and three are :func:`motif_counts`, step
two is :func:`graphrag.sna.null.configuration` -- §19.1's degree-preserving rewiring, in-degree
and out-degree separately when the network is directed -- and step four is
:func:`graphrag.sna.null.significance`. :func:`motif_profile` runs all four and returns the
z-score per shape, plus the normalised *significance profile* of Milo et al., which the chapter
cites as its own first reference (§41.2, n. 2) though it does not print the normalisation; that
last step is named as this module's addition wherever it appears.

**The census, and where the book counts three nodes.** Chapter 41 never fixes a vocabulary for
the three-node shapes, so this module borrows the one the field uses and the one the book's own
earlier chapters set up. §12.2 names the two connected undirected ones -- *"A set of three
connected nodes is a triad"*, and Figure 12.4's caption, *"The two possible connected network
patterns involving three nodes"* (p. 182), a triad and a triangle -- and :func:`triad_census`
adds the two disconnected ones so that the four classes sum to every triple of nodes, exactly as
§10.3's dyad census (:func:`graphrag.sna.paths.dyad_census`) adds the null dyad so the three
classes sum to every pair. On a directed network the same census has sixteen classes, the MAN
codes of Holland and Leinhardt -- whom the book cites in §12.2 (n. 2) for the same object one
size down -- and ``networkx`` counts them; :data:`TRIAD_MEANING` says what each one is in words.
The naming is therefore **the field's, not the Atlas's**, and is written down here rather than
assumed.

**§41.3, isomorphism, exactly rather than approximately.** *"Graph isomorphism is the search of a
function which maps each node of a graph to each node of the other graph, such that they have the
same neighbors -- identically mapped nodes"* (p. 591), and *"subgraph isomorphism is an
NP-complete problem"* (p. 591). The chapter gives two exact-vs-approximate strategies:
Weisfeiler-Lehman and DFS codes on the approximate side, VF2 on the exact one. :func:`isomorphic`
is VF2, through ``networkx``, which is the chapter's own recommendation (*"the current practical
state of the art to solve graph isomorphism is the VF2 algorithm"*, p. 594) including the cheap
pre-checks it opens with (*"two graphs cannot be isomorphic if they have a different number of
nodes or a different number of edges"*, p. 594). :func:`canonical_form` is the labelling the
mining below needs, and it is **not** the book's minimum DFS code: the book is explicit that DFS
codes *"provide an approximate solution to the graph isomorphism problem, so you might get it
wrong sometimes"* (p. 600), and an approximate canonical form would make a support count quietly
wrong. Since every pattern this module mines is bounded to a handful of nodes, exhaustive
relabelling is affordable and exact, so that is what is done, up to
:data:`MAX_CANONICAL_NODES`. What is bought is certainty; what is paid is the bound.

**§41.4, transactional mining: a database of small graphs.** *"So we have a database of many
different graphs and we ask in how many graphs the motif appears. This is our definition of
support"* (p. 598). This corpus has such a database and did not have to invent one:
:func:`document_graphs` reads one entity graph per document, the same passages
:func:`graphrag.sna.export.entity_co_mention` projects into a single network, kept apart instead
of pooled. :func:`transactional_mining` is then the apriori/gSpan family in its bounded form:
grow a pattern one edge at a time, canonicalise, count the *documents* holding an isomorphic
copy, and never extend a pattern that already failed, because *"a larger graph can at most be as
frequent as the least frequent of its subgraphs"* (p. 598) -- the anti-monotonicity of footnote
30, *"it can only stay constant or shrink as your set grows in size"* (p. 596).

**§41.5, single-graph mining: the support that does not double-count.** On one large network the
transactional definition collapses -- *"That number is always going to be either zero -- the
pattern doesn't appear -- or one"* (p. 601) -- and the naive count of occurrences breaks the
pruning rule that made the search feasible: the chapter's Figure 41.13 has a motif whose
extension appears *more* often than the motif itself, which *"is unacceptable, because it breaks
the anti-monotonicity rule"* (p. 601). Of the three repairs the chapter offers -- ego networks,
harmful overlap, minimum image support -- :func:`single_graph_mining` implements the third,
which is the one the chapter recommends because the first two *"have to solve the maximum
independent set problem for every motif"* (p. 603). MNI is exactly the chapter's image table:
*"what matters is that we do not re-use the same node in the network to play the same role in the
motif [...] the support of the motif is the minimum number of distinct row values"* (p. 603).

**What is deliberately not here.** Weisfeiler-Lehman (§41.3, pp. 591-593) is not implemented:
this module needs an exact answer on tiny graphs, where exhaustive relabelling is cheaper than a
colour refinement that *"sometimes fails"* (p. 593), and the place WL earns its keep is the
message-passing chapter it is introduced as a preview of. Association rules and their lift
(§41.4, pp. 597-598) belong to the link-prediction chapters the book itself points at (*"that is
why I introduced it as graph association rule mining in the link prediction chapters"*, p. 598).
Node and edge *labels* on a pattern -- the ``(id, id, label, label, label)`` quintuple of a DFS
code (p. 600) -- are not mined: two documents never share an entity id in the interesting
patterns, so a labelled pattern would have support one by construction, and every pattern here is
an unlabelled topology. Five-node counting by the clever methods the chapter cites (n. 11,
Escape) is not here either; the bound is :data:`MOTIF_SIZES` and the budget is
:data:`MAX_SUBGRAPHS`, and both refuse rather than run for a week.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from graphrag.graph.store import GraphStore
from graphrag.sna.export import entity_passage_rows
from graphrag.sna.matrices import Node
from graphrag.sna.null import HOLDS_FIXED, Significance, configuration, significance

__all__ = [
    "CONNECTED_TRIADS",
    "MAX_CANONICAL_NODES",
    "MAX_MINING_NODES",
    "MAX_SUBGRAPHS",
    "MOTIF_NAMES",
    "MOTIF_SAMPLES",
    "MOTIF_SIZES",
    "TRIAD_CODES",
    "TRIAD_MEANING",
    "UNDIRECTED_TRIADS",
    "Mining",
    "MotifProfile",
    "MotifReport",
    "MotifRow",
    "Pattern",
    "TriadCensus",
    "build_motifs_report",
    "canonical_form",
    "canonical_graph",
    "document_graphs",
    "isomorphic",
    "mni_support",
    "motif_counts",
    "motif_name",
    "motif_profile",
    "motifs_payload",
    "render_motifs",
    "single_graph_mining",
    "transactional_mining",
    "triad_census",
]

#: The sixteen directed triad classes, in the MAN order ``networkx`` returns them. The code
#: counts the **M**\ utual, **A**\ symmetric and **N**\ ull dyads inside the triple, and the
#: letter separates the classes that share those three counts: ``D`` down, ``U`` up, ``C``
#: cyclic, ``T`` transitive. Not the Atlas's vocabulary -- chapter 41 names no three-node
#: classes at all -- but Holland and Leinhardt's, whom §12.2 cites (n. 2) for the same census
#: one size down.
TRIAD_CODES: tuple[str, ...] = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
)

#: The three directed classes whose triples are not connected even ignoring direction. A motif
#: is a shape, and three nodes with no path between them are not one, so the profile of §41.2
#: leaves them out while the census of all sixteen keeps them (they are most of the triples in
#: any sparse network, which is the point of printing the census).
DISCONNECTED_TRIADS: tuple[str, ...] = ("003", "012", "102")

#: The thirteen connected directed triads: the ones a motif profile is over.
CONNECTED_TRIADS: tuple[str, ...] = tuple(c for c in TRIAD_CODES if c not in DISCONNECTED_TRIADS)

#: What each directed triad is, in words, so a report never prints a bare code. ``a``, ``b`` and
#: ``c`` are the three nodes; ``->`` is one arc and ``<->`` a reciprocated pair.
TRIAD_MEANING: dict[str, str] = {
    "003": "no edges at all",
    "012": "one arc, a -> b, and a node neither touches",
    "102": "one reciprocated pair, a <-> b, and a node neither touches",
    "021D": "one node pointing at two others: a <- b -> c (an out-star)",
    "021U": "two nodes pointing at one: a -> b <- c (an in-star)",
    "021C": "a chain: a -> b -> c",
    "111D": "a reciprocated pair with an arc pointing into it: a <-> b <- c",
    "111U": "a reciprocated pair with an arc pointing out of it: a <-> b -> c",
    "030T": (
        "the feed-forward loop: a -> b, b -> c and a -> c, so the chain is short-circuited. "
        "The motif §41.2's own references were written about (n. 3, the E. coli transcription "
        "network)"
    ),
    "030C": "the three-cycle: a -> b -> c -> a, with nothing reciprocated",
    "201": "two reciprocated pairs sharing a node: a <-> b <-> c",
    "120D": "one node pointing at two that reciprocate each other: a <- b -> c, a <-> c",
    "120U": "two nodes pointing at one, reciprocating each other: a -> b <- c, a <-> c",
    "120C": "a chain whose ends reciprocate: a -> b -> c, a <-> c",
    "210": "the complete triad minus one arc",
    "300": "all three pairs reciprocated: the complete triad",
}

#: The four undirected triad classes, summing to every triple of nodes. The middle two are the
#: only ones §12.2 draws: *"the two possible connected network patterns involving three nodes"*
#: (Figure 12.4, p. 182), which the book calls a triad and a triangle. The other two are added
#: here for the same reason §10.3's census keeps the null dyad -- a ratio whose denominator is
#: invisible is half a number.
UNDIRECTED_TRIADS: tuple[str, ...] = ("empty", "one edge", "open triad", "triangle")

#: How many nodes :func:`canonical_form` will relabel exhaustively. The cost is ``n!`` string
#: builds, so 8 nodes is 40,320 of them (tenths of a second) and 12 would be 479 million. Every
#: pattern this module mines is far below the bound; the bound exists so that a caller who hands
#: in a whole network gets an error rather than a hang.
MAX_CANONICAL_NODES: int = 8

#: The motif sizes :func:`motif_counts` will enumerate, in nodes. Three is the census, and is the
#: only size offered on a directed network: there are 199 connected directed graphs on four
#: nodes and enumerating them is the specialist job §41.2's footnote 11 points at, not this one.
MOTIF_SIZES: tuple[int, ...] = (3, 4)

#: How many connected subgraphs an enumeration will look at before refusing. §41.2 says plainly
#: that *"finding motifs in a large network is a hard problem"* (p. 590); this is where that
#: stops being an abstraction. A dense network has combinatorially many four-node subgraphs, and
#: a report that takes an hour is not a report, so the enumerator raises and names the filters
#: (``--min-weight``, ``--backbone``, an ego network) that make the question askable.
MAX_SUBGRAPHS: int = 250_000

#: Above this many nodes single-graph mining is refused rather than attempted (§41.5). Every
#: level of it enumerates every connected edge set of that size in the whole network, which is
#: the cost the chapter's three support definitions are all trying to make bearable.
MAX_MINING_NODES: int = 500

#: Null rewirings a motif profile draws unless told otherwise. The same default ``sna analyze``
#: uses for its own nulls, repeated rather than imported so that the two can be argued about
#: separately; §19.1 fixes no number, and a z-score is printed beside its sample count so a
#: reader can see what bought it.
MOTIF_SAMPLES: int = 50

#: The largest pattern, in **edges**, that mining will grow to unless told otherwise. gSpan grows
#: a pattern one edge at a time (§41.4), so the bound is naturally in edges: 4 reaches every
#: four-node shape and the five-node square-with-a-tail §41.2 asks about by name.
DEFAULT_MAX_SIZE: int = 4

#: How many documents (or, for §41.5, how many distinct images of a node) a pattern needs before
#: it is reported. §41.4 leaves the threshold to the analyst -- *"you establish a support
#: threshold: if a subset fails to occur in that many sets, then you don't want to see it"*
#: (p. 596) -- so this is a convention, printed on every report that uses it.
DEFAULT_MIN_SUPPORT: int = 3


def _plain(graph: nx.Graph) -> nx.Graph:
    """The structure only: no weights, no attributes, no self-loops, same directedness.

    A motif is a shape (§41.2), so everything that is not the shape is dropped before counting.
    Self-loops go because none of the three-node classes has a place for one and every count
    here is over *distinct* nodes.
    """
    bare: nx.Graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    bare.add_nodes_from(graph.nodes)
    bare.add_edges_from((u, v) for u, v in graph.edges if u != v)
    return bare


# --------------------------------------------------------------------------- §41.3 isomorphism


def isomorphic(a: nx.Graph, b: nx.Graph) -> bool:
    """Are these two graphs the same graph, ignoring how the nodes are named (§41.3)?

    The chapter's definition: *"graph isomorphism is the search of a function which maps each
    node of a graph to each node of the other graph, such that they have the same neighbors --
    identically mapped nodes"* (p. 591). ``networkx`` answers it with VF2, which is the
    algorithm §41.3 names as *"the current practical state of the art"* (p. 594), backtracking
    over partial matches and opening with the cheap refusals the chapter lists (*"two graphs
    cannot be isomorphic if they have a different number of nodes or a different number of
    edges"*).

    Structure only: weights, node ids and every attribute are ignored, because the question is
    about topology. Two graphs of different directedness are never the same graph, and asking is
    a mistake rather than an answer, so that raises ``ValueError``.

    The complexity the chapter warns about (*"subgraph isomorphism is an NP-complete problem"*,
    p. 591) is real but is about the *subgraph* question and about adversarial inputs; deciding
    plain isomorphism on the handful-of-nodes patterns this module compares is instant.
    """
    if a.is_directed() != b.is_directed():
        msg = (
            "a directed graph and an undirected one are never isomorphic: an arc carries "
            "information an edge does not (§6.2), so compare like with like"
        )
        raise ValueError(msg)
    return bool(nx.is_isomorphic(_plain(a), _plain(b)))


def _canonical_pairs(n: int, *, directed: bool) -> list[tuple[int, int]]:
    """The cells of the adjacency matrix a canonical string reads, in a fixed order."""
    if directed:
        return [(i, j) for i in range(n) for j in range(n) if i != j]
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def canonical_form(graph: nx.Graph, *, max_nodes: int = MAX_CANONICAL_NODES) -> str:
    """A string that is equal for two graphs exactly when they are isomorphic (§41.3).

    §41.4 needs a *canonical labelling*: something to key a pattern by, so that counting how
    often a shape occurs is a dictionary lookup rather than a quadratic pile of isomorphism
    tests. The book's answer is the minimum DFS code (p. 600), reached by sorting every DFS
    exploration of the graph and taking the smallest -- *"Two graphs with the same minimum DFS
    code are isomorphic"*.

    **This is not that, and the difference is deliberate.** The same page warns that *"DFS codes
    provide an approximate solution to the graph isomorphism problem, so you might get it wrong
    sometimes. However, it is an occurrence rare enough not to impact you in practice most of the
    times"* (p. 600). A support count built on an approximate key is wrong in a way nothing
    downstream can see, so this takes the exact route the same chapter offers in the other
    direction: try **every** relabelling and keep the lexicographically smallest adjacency
    string. That is the definition of a canonical form rather than an approximation of one, and
    it costs ``n!``, which is why it is bounded at ``max_nodes``
    (:data:`MAX_CANONICAL_NODES`) and why the bound is affordable: every pattern grown here has
    at most ``max_size + 1`` nodes.

    The string is ``u`` or ``d`` for undirected or directed, the node count, and the bits of the
    upper triangle (undirected) or of every off-diagonal cell (directed), read in the order
    :func:`_canonical_pairs` fixes. :func:`canonical_graph` turns it back into a graph, so a
    report can print a pattern as an edge list without keeping the subgraph it came from.

    Isolated nodes count: two graphs with the same edges and different numbers of isolates get
    different forms, because they are different graphs. Everything else -- weights, attributes,
    self-loops, the node ids -- is ignored (:func:`_plain`).
    """
    bare = _plain(graph)
    nodes = list(bare.nodes)
    n = len(nodes)
    if n > max_nodes:
        msg = (
            f"canonical_form() relabels exhaustively, which costs n!, so it is bounded at "
            f"{max_nodes} nodes and this graph has {n}. It is meant for the small patterns of "
            "§41.4, not for a whole network; use isomorphic() to compare two large graphs"
        )
        raise ValueError(msg)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [[False] * n for _ in range(n)]
    for u, v in bare.edges:
        adjacency[index[u]][index[v]] = True
        if not bare.is_directed():
            adjacency[index[v]][index[u]] = True
    pairs = _canonical_pairs(n, directed=bare.is_directed())
    best = ""
    for perm in itertools.permutations(range(n)):
        bits = "".join("1" if adjacency[perm[i]][perm[j]] else "0" for i, j in pairs)
        if not best or bits < best:
            best = bits
    return f"{'d' if bare.is_directed() else 'u'}{n}:{best}"


def canonical_graph(canonical: str) -> nx.Graph:
    """The graph a :func:`canonical_form` string stands for, on nodes ``0..n-1``.

    The inverse of :func:`canonical_form` up to isomorphism, which is all a canonical form
    promises: ``canonical_form(canonical_graph(s)) == s`` for any ``s`` the former produced.
    This is how a mined pattern is printed as an edge list (§41.4) without carrying the subgraph
    it was first seen in.
    """
    head, _, bits = canonical.partition(":")
    directed = head.startswith("d")
    n = int(head[1:])
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(n))
    pairs = _canonical_pairs(n, directed=directed)
    if len(bits) != len(pairs):
        msg = f"{canonical!r} is not a canonical form: {len(bits)} bits for {len(pairs)} cells"
        raise ValueError(msg)
    for bit, (i, j) in zip(bits, pairs, strict=True):
        if bit == "1":
            graph.add_edge(i, j)
    return graph


def _named(name_to_graph: Mapping[str, nx.Graph]) -> dict[str, str]:
    return {canonical_form(graph): name for name, graph in name_to_graph.items()}


def _square_with_a_tail() -> nx.Graph:
    """The shape §41.2 asks about by name: *"a square with a dangling edge"* (p. 590)."""
    graph = nx.cycle_graph(4)
    graph.add_edge(0, 4)
    return graph


def _tailed_triangle() -> nx.Graph:
    """A triangle with one pendant edge: the four-node shape a projection makes constantly."""
    graph = nx.complete_graph(3)
    graph.add_edge(0, 3)
    return graph


def _diamond() -> nx.Graph:
    """``K4`` minus one edge: two triangles sharing an edge."""
    graph = nx.complete_graph(4)
    graph.remove_edge(2, 3)
    return graph


#: The undirected shapes small enough to have a name people use, keyed by canonical form. Two of
#: them are the book's own (*"A triangle is a motif, a square is a motif"*, p. 590, and the
#: square with a dangling edge of the same page); the rest are the field's ordinary names for the
#: connected graphlets on three and four nodes, and a pattern this table does not know is printed
#: as its edge list and nothing else.
MOTIF_NAMES: dict[str, str] = _named(
    {
        "edge": nx.path_graph(2),
        "open triad": nx.path_graph(3),
        "triangle": nx.complete_graph(3),
        "path": nx.path_graph(4),
        "star": nx.star_graph(3),
        "square": nx.cycle_graph(4),
        "tailed triangle": _tailed_triangle(),
        "diamond": _diamond(),
        "clique (K4)": nx.complete_graph(4),
        "square with a tail": _square_with_a_tail(),
    }
)


def motif_name(key: str) -> str:
    """The words for a motif key: a MAN code's meaning, a canonical form's ordinary name, or "".

    A report never prints a bare key. ``030T`` is a feed-forward loop and ``u4:111000`` is a
    path; a shape with no ordinary name gets its edge list instead, which is what
    :func:`canonical_graph` is for.
    """
    if key in TRIAD_MEANING:
        return TRIAD_MEANING[key]
    if key in UNDIRECTED_TRIADS:
        return key
    return MOTIF_NAMES.get(key, "")


# ---------------------------------------------------------------------------- §41.2 the census


@dataclass(frozen=True)
class TriadCensus:
    """Every triple of nodes in the network, sorted into the classes it can fall into.

    The three-node counterpart of §10.3's dyad census, and read the same way: the classes are
    exhaustive, so they sum to ``C(n, 3)`` and a count means nothing until the total is beside
    it. On a sparse network almost every triple is empty, which is why the profile
    (:func:`motif_profile`) drops the disconnected classes and this does not.
    """

    directed: bool
    nodes: int
    counts: dict[str, int]

    @property
    def triples(self) -> int:
        """Every unordered triple of distinct nodes, ``C(n, 3)``. The classes sum to it."""
        return math.comb(self.nodes, 3)

    @property
    def connected(self) -> dict[str, int]:
        """The classes whose triples hold at least one path between all three nodes."""
        skip = set(DISCONNECTED_TRIADS) if self.directed else {"empty", "one edge"}
        return {key: count for key, count in self.counts.items() if key not in skip}

    def global_clustering(self) -> float:
        """§12.2's clustering coefficient, read straight off the undirected census.

        *"The global clustering coefficient (CC) is simply CC = 3 x #Triangles / #Triads"*
        (p. 182), where the book's ``#Triads`` counts the open triads *centred on each node*: a
        triangle closes three of them, which is why the numerator is multiplied by three. In
        this census a triple with three edges is one ``triangle`` and a triple with two is one
        ``open triad``, so the book's denominator is ``open triad + 3 x triangle`` and the
        identity is exact rather than approximate.

        Undefined on a directed network -- §12.2's coefficient is about undirected triads -- and
        undefined when there is no triad at all to close, both of which raise ``ValueError``
        rather than return zero: a network with no triads has no clustering, not clustering 0.
        """
        if self.directed:
            msg = (
                "§12.2's clustering coefficient counts undirected triads; flatten the network "
                "first (§6.2) and take the census of the flattened view"
            )
            raise ValueError(msg)
        triangles = self.counts["triangle"]
        triads = self.counts["open triad"] + 3 * triangles
        if not triads:
            msg = "no triad in this network, so there is nothing for a triangle to close (§12.2)"
            raise ValueError(msg)
        return 3 * triangles / triads


def triad_census(graph: nx.Graph) -> TriadCensus:
    """Sort every triple of nodes into its class: sixteen directed, four undirected (§41.2).

    §41.2 tells you to *"count how many times each motif appears in your network"* (p. 590) and
    leaves the enumeration of the three-node shapes to the literature it cites. Directed, that
    enumeration is Holland and Leinhardt's sixteen MAN types, which ``networkx`` counts exactly;
    undirected it is four, of which §12.2 draws the two connected ones as *"the two possible
    connected network patterns involving three nodes"* (Figure 12.4, p. 182) -- a triad and a
    triangle.

    **What this adds to the book.** The two disconnected classes. §12.2 has no use for them
    because it is computing a ratio; a census does, because the classes have to be exhaustive
    for a count to be readable -- ``triangle: 41`` is a different finding in a network of 50
    triples and in one of 50 million. :attr:`TriadCensus.triples` is that denominator and the
    counts sum to it.

    The undirected counts are arithmetic rather than enumeration, which is why this is cheap on
    a large network: triangles come from ``networkx``, the open triads are
    ``sum_i C(k_i, 2) - 3T`` (§12.2's own triad count, minus the three each triangle closes),
    and the two disconnected classes follow from ``m(n - 2)``, the number of (edge, third node)
    pairs. Self-loops and weights are ignored throughout: a motif is a shape.
    """
    bare = _plain(graph)
    n = bare.number_of_nodes()
    if bare.is_directed():
        counts = (
            dict.fromkeys(TRIAD_CODES, 0)
            if n < 3
            else {code: int(nx.triadic_census(bare)[code]) for code in TRIAD_CODES}
        )
        return TriadCensus(directed=True, nodes=n, counts=counts)
    triangles = sum(nx.triangles(bare).values()) // 3
    two_paths = sum(math.comb(degree, 2) for _, degree in bare.degree()) - 3 * triangles
    edges = bare.number_of_edges()
    one_edge = edges * (n - 2) - 2 * two_paths - 3 * triangles
    counts = {
        "empty": math.comb(n, 3) - one_edge - two_paths - triangles,
        "one edge": one_edge,
        "open triad": two_paths,
        "triangle": triangles,
    }
    return TriadCensus(directed=False, nodes=n, counts=counts)


def _connected_node_sets(graph: nx.Graph, size: int) -> Iterator[tuple[Node, ...]]:
    """Every connected set of ``size`` nodes, each exactly once (ESU).

    Wernicke's enumeration: fix an order on the nodes, and grow a subgraph only with neighbours
    that come after the node it started from and that do not already touch it, so each set is
    reached from exactly one root by exactly one path. Direction is ignored while growing --
    a directed triple is a motif when its three nodes hang together at all, which is what makes
    the thirteen connected triads thirteen.

    Not the book's: §41.2 leaves "count how many times each motif appears" to the specialist
    counters it cites (n. 11-12) and this is the ordinary one. It raises above
    :data:`MAX_SUBGRAPHS` rather than run on.
    """
    order = {node: i for i, node in enumerate(graph.nodes)}
    neighbours: dict[Node, set[Node]] = {
        node: {other for other in nx.all_neighbors(graph, node) if other != node}
        for node in graph.nodes
    }
    seen = 0
    for root in graph.nodes:
        stack: list[tuple[tuple[Node, ...], set[Node]]] = [
            ((root,), {n for n in neighbours[root] if order[n] > order[root]})
        ]
        while stack:
            chosen, extension = stack.pop()
            if len(chosen) == size:
                seen += 1
                if seen > MAX_SUBGRAPHS:
                    msg = (
                        f"more than {MAX_SUBGRAPHS:,} connected subgraphs of {size} nodes: "
                        "§41.2 says finding motifs in a large network is a hard problem (p. 590) "
                        "and this is that. Narrow the network first -- --min-weight, a backbone "
                        "(ch. 27), or an ego network -- and ask again"
                    )
                    raise ValueError(msg)
                yield chosen
                continue
            pool = set(extension)
            touching = {node for member in chosen for node in neighbours[member]}
            while pool:
                node = pool.pop()
                grown = (*chosen, node)
                fresh = {
                    other
                    for other in neighbours[node]
                    if order[other] > order[root] and other not in touching and other not in chosen
                }
                stack.append((grown, pool | fresh))


def motif_counts(graph: nx.Graph, size: int = 3) -> dict[str, int]:
    """How many connected subgraphs of each shape this network holds (§41.2, step one).

    Keyed by the MAN code on a directed network of triples, and by :func:`canonical_form`
    otherwise; :func:`motif_name` turns either into words. Only *connected* shapes are counted,
    because a motif is a connection pattern and three nodes with nothing between them are not
    one -- the full census, disconnected classes and all, is :func:`triad_census`.

    Three nodes is the only size a directed network is offered: there are 199 connected directed
    graphs on four nodes, and enumerating them is the specialist problem §41.2 hands to its
    references (n. 11), not a thing to do in passing. Undirected, four nodes is also available
    and is where the shapes the chapter names by hand live -- *"A triangle is a motif, a square
    is a motif"* (p. 590).

    The counts are of *induced* subgraphs: the four nodes of a square are a square only if the
    diagonals are absent, and the same four nodes with a diagonal are a diamond instead. That is
    the convention the triad census uses (its sixteen classes are induced and exhaustive), so
    the two halves of a report agree. It is also what makes the classes comparable against a
    null: the counts of all shapes of one size sum to the number of connected subgraphs of that
    size, and nothing is counted twice.
    """
    if size not in MOTIF_SIZES:
        msg = f"size must be one of {MOTIF_SIZES}, got {size}"
        raise ValueError(msg)
    bare = _plain(graph)
    if bare.is_directed():
        if size != 3:
            msg = (
                "a directed motif census is offered at three nodes only: four-node directed "
                "shapes number 199 and counting them is §41.2's specialist problem (n. 11). "
                "Flatten the network (§6.2) to ask about four-node shapes"
            )
            raise ValueError(msg)
        census = triad_census(bare)
        return {code: census.counts[code] for code in CONNECTED_TRIADS}
    if size == 3:
        census = triad_census(bare)
        return {
            canonical_form(nx.path_graph(3)): census.counts["open triad"],
            canonical_form(nx.complete_graph(3)): census.counts["triangle"],
        }
    counts: dict[str, int] = {}
    for nodes in _connected_node_sets(bare, size):
        key = canonical_form(bare.subgraph(nodes))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# -------------------------------------------------------------------------- §41.2 the profile


@dataclass(frozen=True)
class MotifRow:
    """One shape, its count, and how that count reads against the null (§41.2, step four)."""

    key: str
    name: str
    count: int
    edges: tuple[tuple[int, int], ...]
    """The shape as an edge list on nodes ``0..n-1``, so a reader can draw it."""
    test: Significance
    profile: float
    """The normalised significance profile ``z / sqrt(sum z^2)``: this shape's share of the
    network's total deviation from the null. Not the book's -- see :func:`motif_profile`."""

    @property
    def z(self) -> float:
        return self.test.z

    @property
    def p_value(self) -> float:
        return self.test.p_value


@dataclass(frozen=True)
class MotifProfile:
    """Every shape of one size, counted, and tested against a degree-preserving null (§41.2)."""

    size: int
    directed: bool
    rows: tuple[MotifRow, ...]
    null: str
    samples: int
    """How many null networks were actually built -- not how many were asked for. A network too
    small or too constrained to rewire yields fewer, or none (§19.1)."""
    asked: int
    subgraphs: int
    """How many connected subgraphs of this size the observed network holds: the n of §41.2."""
    note: str = ""

    def over(self, threshold: float = 2.0) -> tuple[MotifRow, ...]:
        """The shapes whose z-score clears ``threshold``: §41.2's overexpressed patterns."""
        return tuple(row for row in self.rows if row.z >= threshold)

    def under(self, threshold: float = -2.0) -> tuple[MotifRow, ...]:
        """The shapes the null has more of than the network does."""
        return tuple(row for row in self.rows if row.z <= threshold)


def motif_profile(
    graph: nx.Graph,
    *,
    size: int = 3,
    samples: int = MOTIF_SAMPLES,
    seed: int | random.Random | None = None,
) -> MotifProfile:
    """§41.2's four steps, run: count, rewire, count again, compare.

    *"First, you count how many times each motif appears in your network. Second, you define a
    null model of your network, keeping its relevant properties fixed -- maybe just the degree
    distribution. Third, you count the expected number of occurrences of the motifs in the null
    model. Finally, you compare with your observation, so that you can build an idea of the
    statistical significance of the motif"* (p. 590). The null is
    :func:`graphrag.sna.null.configuration`, which is the *"maybe just the degree distribution"*
    the sentence offers: every node keeps its degree, and on a directed network its in-degree and
    out-degree separately, by §19.1's swaps. Nothing else is held fixed, so an overexpressed
    triangle here has beaten the degree sequence and not "chance".

    The test is two-tailed. §41.2 is interested in shapes that appear *"much more -- or less --
    often than what you'd expect"* (p. 589), and a motif the network avoids is as much a finding
    as one it repeats.

    ``profile`` is the **significance profile**: each z divided by the root of the summed
    squares, so the vector has length one and two networks of different sizes can be compared on
    shape alone. The book does not define it; it is the normalisation of Milo et al., whose motif
    paper §41.2 cites as its own first reference (n. 2) and whose follow-up (*Superfamilies of
    evolved and designed networks*, Science 2004) introduced this vector. It is named as an
    addition wherever it is printed.

    **An identity worth knowing before reading an undirected profile.** The number of open triads
    is ``sum_i C(k_i, 2) - 3T``, and a degree-preserving null holds the first term exactly fixed.
    So on an undirected network the open triad's z-score is *minus* the triangle's, to the digit,
    and the two rows are one finding rather than two. That is a property of the null, not of the
    network, and the report says so.

    A network too small or too constrained for the null to have a move yields no samples at all;
    that is reported rather than raised (every ``z`` is then 0.0 and the caveat says the test did
    not happen), because a count with no test is still a description.
    """
    observed = motif_counts(graph, size)
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)  # noqa: S311
    drawn: list[dict[str, int]] = [
        motif_counts(sample, size)
        for sample in configuration(_plain(graph), samples, seed=rng, weights=False)
    ]
    keys = sorted(set(observed) | {key for counts in drawn for key in counts})
    tests = {
        key: significance(
            float(observed.get(key, 0)),
            [float(counts.get(key, 0)) for counts in drawn],
            null="configuration",
            tail="two",
        )
        for key in keys
    }
    norm = math.sqrt(sum(test.z**2 for test in tests.values()))
    rows = tuple(
        MotifRow(
            key=key,
            name=motif_name(key),
            count=observed.get(key, 0),
            edges=_pattern_edges(key),
            test=tests[key],
            profile=tests[key].z / norm if norm else 0.0,
        )
        for key in keys
    )
    return MotifProfile(
        size=size,
        directed=graph.is_directed(),
        rows=rows,
        null=HOLDS_FIXED["configuration"],
        samples=len(drawn),
        asked=samples,
        subgraphs=sum(observed.values()),
        note=_profile_note(graph, drawn, samples),
    )


def _profile_note(graph: nx.Graph, drawn: Sequence[object], asked: int) -> str:
    """What a reader has to know about this particular run before reading its z-scores."""
    if not drawn:
        return (
            "No null network could be built, so no shape below was tested: the network is too "
            "small, or too constrained, for a degree-preserving swap to have a move to make "
            "(§19.1). The counts are a description and nothing more."
        )
    if len(drawn) < asked:
        return (
            f"Only {len(drawn)} of the {asked} rewirings asked for could be built: the degree "
            "sequence admits few swaps, so the null distribution is narrower than the sample "
            "count suggests and the z-scores are optimistic (§19.1)."
        )
    if not graph.is_directed():
        return (
            "Undirected, the open triad and the triangle are one finding: the degree sequence "
            "fixes `sum_i C(k_i, 2) = open triads + 3 x triangles`, and the null holds that "
            "sum exactly, so the open triad's z is the triangle's with the sign reversed. That "
            "is the null talking, not the network."
        )
    return ""


def _pattern_edges(key: str) -> tuple[tuple[int, int], ...]:
    """The shape a motif key stands for, as an edge list on ``0..n-1``. Empty for a MAN code.

    A directed triad is named by its code and drawn by :data:`TRIAD_MEANING` in words; every
    other key is a canonical form and can be decoded exactly.
    """
    if key in TRIAD_MEANING or key in UNDIRECTED_TRIADS:
        return ()
    return tuple(sorted((int(u), int(v)) for u, v in canonical_graph(key).edges))


# -------------------------------------------------------------- §41.4/§41.5 frequent subgraphs


@dataclass(frozen=True)
class Pattern:
    """One frequent subgraph: its shape, and how many things in the frame hold it (§41.4)."""

    canonical: str
    name: str
    nodes: int
    edges: tuple[tuple[int, int], ...]
    support: int

    @property
    def size(self) -> int:
        """The pattern's size in edges, which is what mining grows one at a time (§41.4)."""
        return len(self.edges)

    def as_edge_list(self) -> str:
        """The pattern as ``0-1, 1-2, 0-2``, which is how a report prints a shape it has no
        name for."""
        return ", ".join(f"{u}-{v}" for u, v in self.edges)


@dataclass(frozen=True)
class Mining:
    """One frequent-subgraph run: the patterns, and the support definition that produced them."""

    kind: str
    """``transactional`` (§41.4) or ``single-graph`` (§41.5)."""
    support_means: str
    """What a support of 5 means, in words. The two kinds count different things and a number
    without this sentence beside it is not readable."""
    min_support: int
    max_size: int
    frame: str
    """What was mined: how many graphs, drawn from what."""
    graphs: int
    patterns: tuple[Pattern, ...]
    levels: int
    """How many sizes were reached before the support threshold killed every candidate. The
    anti-monotonic pruning of §41.4 is what makes this stop early rather than at ``max_size``."""
    note: str = ""

    def of_size(self, size: int) -> tuple[Pattern, ...]:
        return tuple(pattern for pattern in self.patterns if pattern.size == size)

    @property
    def largest(self) -> int:
        return max((pattern.size for pattern in self.patterns), default=0)


Edge = tuple[Node, Node]


def _connected_edge_sets(graph: nx.Graph, size: int) -> list[frozenset[Edge]]:
    """Every connected set of ``size`` edges in ``graph``, each exactly once.

    The search space §41.4 explores, in its data-driven form: rather than generating candidate
    patterns and testing them, grow edge sets that are actually present and canonicalise what
    comes out. Two edges are adjacent when they share a node, so a "connected" edge set is one
    whose edge-induced subgraph is connected -- which is the only kind of pattern this module
    mines, because a disconnected motif is two motifs.

    Refuses above :data:`MAX_SUBGRAPHS`, for the reason §41.2 gives on p. 590.
    """
    edges = [(u, v) for u, v in graph.edges if u != v]
    current: set[frozenset[Edge]] = {frozenset({edge}) for edge in edges}
    _within_budget(current, size)
    incident: dict[Node, list[Edge]] = {}
    for edge in edges:
        incident.setdefault(edge[0], []).append(edge)
        incident.setdefault(edge[1], []).append(edge)
    for _ in range(size - 1):
        grown: set[frozenset[Edge]] = set()
        for chosen in current:
            for node in {node for edge in chosen for node in edge}:
                grown |= {chosen | {edge} for edge in incident.get(node, ()) if edge not in chosen}
            _within_budget(grown, size)
        current = grown
        if not current:
            break
    return sorted(current, key=lambda chosen: sorted(map(str, chosen)))


def _within_budget(found: set[frozenset[Edge]], size: int) -> None:
    """Stop an enumeration that has left the realm of the answerable (§41.2, p. 590)."""
    if len(found) > MAX_SUBGRAPHS:
        msg = (
            f"more than {MAX_SUBGRAPHS:,} connected subgraphs of up to {size} edges: §41.2 says "
            "finding motifs in a large network is a hard problem (p. 590) and this is that. "
            "Narrow the network -- --min-weight, a backbone (ch. 27), an ego network -- or "
            "lower --max-size, and ask again"
        )
        raise ValueError(msg)


def _edge_graph(edges: frozenset[Edge], *, directed: bool) -> nx.Graph:
    """The subgraph an edge set induces -- its edges, and only the nodes they touch."""
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    return graph


def _survives_apriori(
    edges: frozenset[Edge], frequent: Mapping[str, Pattern], *, directed: bool
) -> bool:
    """Is every connected one-edge-smaller piece of this candidate already frequent (§41.4)?

    The pruning rule the chapter's footnote 30 states -- *"the support function is
    anti-monotonic: it can only stay constant or shrink as your set grows in size"* -- and p. 598
    repeats for graphs: *"a larger graph can at most be as frequent as the least frequent of its
    subgraphs"*. So a candidate with an infrequent sub-pattern cannot be frequent and is never
    counted, which is the whole economy of apriori and of gSpan.
    """
    for edge in edges:
        smaller = edges - {edge}
        piece = _edge_graph(smaller, directed=directed)
        joined = nx.is_weakly_connected(piece) if directed else nx.is_connected(piece)
        if joined and canonical_form(piece) not in frequent:
            return False
    return True


def transactional_mining(
    graphs: Sequence[nx.Graph],
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_size: int = DEFAULT_MAX_SIZE,
    frame: str = "",
) -> Mining:
    """Shapes that recur across a *database* of small graphs, and in how many of them (§41.4).

    *"So we have a database of many different graphs and we ask in how many graphs the motif
    appears. This is our definition of support"* (p. 598). Support therefore counts **graphs**,
    not occurrences: a document whose entity graph holds four triangles contributes one to a
    triangle's support, which is exactly what makes the count anti-monotonic and the search
    prunable.

    The enumerator is gSpan-shaped rather than gSpan: patterns grow one edge at a time (§41.4),
    every candidate is keyed by a canonical labelling, and a candidate with an infrequent
    sub-pattern is dropped before it is ever counted. Two deliberate departures from the paper
    the chapter follows. First, the canonical key is :func:`canonical_form`, which is exact,
    rather than the minimum DFS code, which p. 600 says is *"an approximate solution to the graph
    isomorphism problem"*. Second, candidates are generated **from the data** -- the connected
    edge sets the graphs actually contain -- rather than from a pattern lattice, which costs the
    memory of one level at a time and buys the support count for free, since a pattern's support
    is the number of graphs its key was seen in.

    Patterns are unlabelled topologies. The book's DFS quintuple carries node and edge labels
    (p. 600), and they are dropped here on purpose: the node labels available are entity ids,
    two documents rarely name the same three entities, and a labelled pattern would have support
    one by construction. What comes back is a shape -- "three things discussed pairwise in one
    document" -- not a statement about which things.

    ``max_size`` is in edges, and is a refusal rather than a hint: nothing above it is grown, and
    the report says so beside the largest pattern found. An empty database, or one whose graphs
    have no edges, mines nothing and says so rather than raising.
    """
    if min_support < 1:
        msg = f"min_support counts graphs, so it is at least 1; got {min_support}"
        raise ValueError(msg)
    if max_size < 1:
        msg = f"max_size is a pattern's size in edges, so it is at least 1; got {max_size}"
        raise ValueError(msg)
    directed = any(graph.is_directed() for graph in graphs)
    found: list[Pattern] = []
    frequent: dict[str, Pattern] = {}
    levels = 0
    for size in range(1, max_size + 1):
        holders: dict[str, set[int]] = {}
        for index, graph in enumerate(graphs):
            bare = _plain(graph)
            for edges in _connected_edge_sets(bare, size):
                if size > 1 and not _survives_apriori(edges, frequent, directed=directed):
                    continue
                holders.setdefault(
                    canonical_form(_edge_graph(edges, directed=directed)), set()
                ).add(index)
        level = {
            key: _pattern(key, len(indices))
            for key, indices in sorted(holders.items())
            if len(indices) >= min_support
        }
        if not level:
            break
        levels = size
        found += sorted(level.values(), key=lambda p: (-p.support, p.canonical))
        frequent = level
    return Mining(
        kind="transactional",
        support_means=(
            "the number of graphs in the database that hold at least one copy of the pattern, "
            "however many copies each holds (§41.4, p. 598)"
        ),
        min_support=min_support,
        max_size=max_size,
        frame=frame or f"{len(graphs)} graphs",
        graphs=len(graphs),
        patterns=tuple(found),
        levels=levels,
        note=_mining_note(found, max_size, levels),
    )


def _pattern(canonical: str, support: int) -> Pattern:
    shape = canonical_graph(canonical)
    return Pattern(
        canonical=canonical,
        name=MOTIF_NAMES.get(canonical, ""),
        nodes=shape.number_of_nodes(),
        edges=tuple(sorted((int(u), int(v)) for u, v in shape.edges)),
        support=support,
    )


def _mining_note(patterns: Sequence[Pattern], max_size: int, levels: int) -> str:
    if not patterns:
        return (
            "Nothing cleared the support threshold, not even a single edge. That is a finding "
            "about the threshold as much as about the data: §41.4 leaves it to the analyst "
            "(p. 596) and it is the first thing to move."
        )
    if levels >= max_size:
        return (
            f"The search stopped at the {max_size}-edge bound rather than at the support "
            "threshold, so there may be larger frequent patterns this run never looked at. "
            "Raise --max-size to see them, and expect the cost to climb quickly (p. 590)."
        )
    return (
        f"The search stopped on its own at {levels + 1} edges: no candidate of that size had a "
        "frequent sub-pattern left to grow from, which is the anti-monotonicity of p. 598 doing "
        "its job."
    )


def mni_support(graph: nx.Graph, pattern: nx.Graph) -> int:
    """§41.5's minimum image support: how often a pattern occurs without re-using a node's role.

    *"In this definition, what matters is that we do not re-use the same node in the network to
    play the same role in the motif. In practice, we look at which node in the network we use to
    map each node in the motif. To do so, we build an 'image' table [...] Thus, the support of
    the motif is the minimum number of distinct row values"* (p. 603). So: find every occurrence
    of the pattern in the network, note for each node of the *pattern* which network nodes ever
    play it, and take the smallest of those counts.

    That is the definition and this is exactly it: the occurrences are the connected edge sets of
    the network isomorphic to the pattern, the mapping of each occurrence onto the pattern's
    nodes comes from VF2 (§41.3), and every mapping counts -- a pattern with a symmetry has
    several, and the chapter's image table (Figure 41.15) is built the same way.

    What it buys is the thing §41.5 spends its whole length on: *"a motif can only occur at most
    as much as the least frequent of its sub-motifs"* (p. 601) is true of this count and false of
    the naive one, so the apriori pruning still works. What it costs is intuition -- MNI can be
    smaller than the number of occurrences you can see with your eyes, and the chapter's own
    Figure 41.13 is the example: one red node in the data means support 1 for anything that needs
    a red node, however many ways the rest of the shape can be drawn around it.

    Zero when the pattern does not occur. Undefined for an empty pattern, which raises.
    """
    if pattern.number_of_edges() == 0:
        msg = "the minimum image support of a pattern with no edges is not defined (§41.5)"
        raise ValueError(msg)
    directed = graph.is_directed()
    target = canonical_form(pattern)
    images: dict[Node, set[Node]] = {node: set() for node in pattern.nodes}
    for edges in _connected_edge_sets(_plain(graph), pattern.number_of_edges()):
        occurrence = _edge_graph(edges, directed=directed)
        if canonical_form(occurrence) != target:
            continue
        matcher = (
            nx.algorithms.isomorphism.DiGraphMatcher(occurrence, pattern)
            if directed
            else nx.algorithms.isomorphism.GraphMatcher(occurrence, pattern)
        )
        for mapping in matcher.isomorphisms_iter():
            for node, role in mapping.items():
                images[role].add(node)
    return min((len(nodes) for nodes in images.values()), default=0)


def single_graph_mining(
    graph: nx.Graph,
    *,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_size: int = DEFAULT_MAX_SIZE,
    max_nodes: int = MAX_MINING_NODES,
    frame: str = "",
) -> Mining:
    """Shapes that recur *inside* one network, counted so that the count cannot grow (§41.5).

    On one network the transactional support of §41.4 is useless -- *"That number is always going
    to be either zero -- the pattern doesn't appear -- or one"* (p. 601) -- and counting
    occurrences instead breaks the rule the search depends on: the chapter's Figure 41.13 shows a
    motif whose extension occurs twice while the motif itself occurs once, which *"is
    unacceptable, because it breaks the anti-monotonicity rule"* (p. 601). The support used here
    is the third of the chapter's three repairs, the minimum image support (:func:`mni_support`),
    chosen for the reason the chapter gives: simple and harmful overlap *"have to solve the
    maximum independent set problem for every motif we search in a possibly very large overlap
    graph, which is a hard problem"* (p. 603).

    The search is the same level-wise growth :func:`transactional_mining` runs, with the same
    exact canonical key and the same apriori pruning; only the support changes. Ego-network
    support -- the chapter's first repair, *"the number of nodes seeing the pattern around them"*
    (p. 601) -- is not offered: it is transactional mining over
    :func:`graphrag.sna.export.ego`, and a caller who wants it can build the ego networks and
    call the transactional function, which is what it would do anyway.

    Refuses above ``max_nodes``. Every level enumerates every connected edge set of its size in
    the whole network, and on a corpus network of thousands of nodes that is the hard problem of
    p. 590 with no bound on it; narrow the network first.
    """
    if graph.number_of_nodes() > max_nodes:
        msg = (
            f"single-graph mining enumerates every connected subgraph of the network at every "
            f"size, so it is bounded at {max_nodes} nodes and this one has "
            f"{graph.number_of_nodes():,} (§41.5). Narrow it first -- --min-weight, a backbone "
            "(ch. 27), or an ego network -- or mine the per-document graphs instead (§41.4)"
        )
        raise ValueError(msg)
    if min_support < 1:
        msg = (
            f"min_support counts distinct images of a node, so it is at least 1; got {min_support}"
        )
        raise ValueError(msg)
    directed = graph.is_directed()
    bare = _plain(graph)
    found: list[Pattern] = []
    frequent: dict[str, Pattern] = {}
    levels = 0
    for size in range(1, max_size + 1):
        images: dict[str, dict[Node, set[Node]]] = {}
        for edges in _connected_edge_sets(bare, size):
            if size > 1 and not _survives_apriori(edges, frequent, directed=directed):
                continue
            occurrence = _edge_graph(edges, directed=directed)
            key = canonical_form(occurrence)
            shape = canonical_graph(key)
            table = images.setdefault(key, {node: set() for node in shape.nodes})
            matcher = (
                nx.algorithms.isomorphism.DiGraphMatcher(occurrence, shape)
                if directed
                else nx.algorithms.isomorphism.GraphMatcher(occurrence, shape)
            )
            for mapping in matcher.isomorphisms_iter():
                for node, role in mapping.items():
                    table[role].add(node)
        level = {
            key: _pattern(key, min(len(nodes) for nodes in table.values()))
            for key, table in sorted(images.items())
            if min(len(nodes) for nodes in table.values()) >= min_support
        }
        if not level:
            break
        levels = size
        found += sorted(level.values(), key=lambda p: (-p.support, p.canonical))
        frequent = level
    return Mining(
        kind="single-graph",
        support_means=(
            "the minimum image support: for each node of the pattern, how many different network "
            "nodes ever play that role, and then the smallest of those counts (§41.5, p. 603). "
            "It is not the number of occurrences, and it is deliberately smaller"
        ),
        min_support=min_support,
        max_size=max_size,
        frame=frame or f"one network of {graph.number_of_nodes():,} nodes",
        graphs=1,
        patterns=tuple(found),
        levels=levels,
        note=_mining_note(found, max_size, levels),
    )


# ------------------------------------------------------------------- the database this corpus has


def document_graphs(
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
) -> list[nx.Graph]:
    """One entity graph per document: the database §41.4's transactional mining needs.

    *"For instance, your graph database could contain thousands of different chemical compounds,
    and you want to find the most common substructures among all those molecules"* (p. 600). This
    corpus has the same shape of data and did not have to be bent into it: a document is a
    self-contained conversation, and the entities it names, joined when a passage named both, are
    a small graph. What ``--network entities`` does is pool all of them into one; this keeps them
    apart, which is what makes "in how many documents" a meaningful question.

    The rows, the filters and the attribute resolution are
    :func:`graphrag.sna.export.entity_passage_rows`'s, so a document graph here holds exactly the
    mentions the pooled network would have drawn from the same document. Every document with at
    least one entity mention is returned, including those with a single entity and therefore no
    edges: they are part of the denominator, and dropping them would inflate every support.

    Each graph carries ``doc_id`` and a frame on ``graph.graph``, and every node carries its
    ``label``, so a pattern found across documents can be traced back to them.
    """
    rows = entity_passage_rows(
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
    passages: dict[str, dict[str, set[str]]] = {}
    labels: dict[str, str] = {}
    for row in rows:
        passages.setdefault(row.doc_id, {}).setdefault(row.chunk_id, set()).add(row.entity_id)
        labels[row.entity_id] = row.name
    built: list[nx.Graph] = []
    for doc_id in sorted(passages):
        graph = nx.Graph()
        for entities in passages[doc_id].values():
            graph.add_nodes_from(entities)
            for u, v in itertools.combinations(sorted(entities), 2):
                graph.add_edge(u, v, weight=graph.get_edge_data(u, v, {}).get("weight", 0) + 1)
        for node in graph.nodes:
            graph.nodes[node]["label"] = labels.get(node, node)
        graph.graph.update(
            network="entities",
            persona_id=persona_id,
            doc_id=doc_id,
            unit="passages shared",
            frame=(
                f"The entities named in document {doc_id}: {graph.number_of_nodes()} of them, "
                f"joined when one of its {len(passages[doc_id])} passages named both."
            ),
        )
        built.append(graph)
    return built


# ------------------------------------------------------------------------------------ the report


@dataclass(frozen=True)
class MotifReport:
    """One ``sna motifs`` run: the census, the profile, and whatever mining was asked for."""

    persona_id: str
    network: str
    graph: nx.Graph
    census: TriadCensus
    profile: MotifProfile | None
    transactional: Mining | None
    single: Mining | None
    generated_at: str
    notes: list[str] = field(default_factory=list)

    def label(self, node: Node) -> str:
        data = self.graph.nodes.get(node, {})
        return str(data.get("label", node))


def build_motifs_report(
    graph: nx.Graph,
    *,
    persona_id: str,
    network: str,
    size: int = 3,
    samples: int = MOTIF_SAMPLES,
    seed: int | None = None,
    documents: Sequence[nx.Graph] | None = None,
    mine: bool = False,
    min_support: int = DEFAULT_MIN_SUPPORT,
    max_size: int = DEFAULT_MAX_SIZE,
    max_nodes: int = MAX_MINING_NODES,
) -> MotifReport:
    """One run of chapter 41 over one network: §41.2 always, §41.4 and §41.5 when asked.

    The census and the profile are cheap and always computed. Mining is opt-in because it is the
    hard problem of p. 590 and its cost is not predictable from the node count alone;
    ``documents`` is the database §41.4 mines over (:func:`document_graphs`) and, when it is not
    supplied, the transactional section is absent rather than faked from the pooled network,
    which would give every pattern support one.
    """
    notes: list[str] = []
    census = triad_census(graph)
    profile: MotifProfile | None = None
    try:
        profile = motif_profile(graph, size=size, samples=samples, seed=seed)
    except ValueError as exc:
        notes.append(f"No motif profile was built: {exc}")
    transactional: Mining | None = None
    single: Mining | None = None
    if mine:
        if documents is not None:
            transactional = transactional_mining(
                documents,
                min_support=min_support,
                max_size=max_size,
                frame=(
                    f"{len(documents)} documents of persona {persona_id}, one entity graph each, "
                    "built from the same passages the pooled entity network projects"
                ),
            )
        single = single_graph_mining(
            graph,
            min_support=min_support,
            max_size=max_size,
            max_nodes=max_nodes,
            frame=(
                f"the {network} network of persona {persona_id}: "
                f"{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges"
            ),
        )
    return MotifReport(
        persona_id=persona_id,
        network=network,
        graph=graph,
        census=census,
        profile=profile,
        transactional=transactional,
        single=single,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
    )


def _rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _census_section(report: MotifReport) -> list[str]:
    census = report.census
    graph = report.graph
    lines = [
        "## The three-node census (§41.2)",
        "",
        f"**Sampling frame:** {graph.graph.get('frame', 'not recorded')}",
        "",
        f"**n:** {census.triples:,} triples of nodes, out of {census.nodes:,} nodes and "
        f"{graph.number_of_edges():,} edges. The classes are exhaustive and sum to that.",
        "",
        "**Null model:** none in this table -- it is a count, not a claim. The claims are in the "
        "profile below, against a degree-preserving null.",
        "",
    ]
    if census.directed:
        lines += _rows(
            ["code", "triples", "share", "what it is"],
            [
                [
                    f"`{code}`",
                    f"{census.counts[code]:,}",
                    f"{census.counts[code] / census.triples:.4f}" if census.triples else "--",
                    TRIAD_MEANING[code],
                ]
                for code in TRIAD_CODES
            ],
        )
        lines += [
            "The codes are Holland and Leinhardt's -- **M**utual, **A**symmetric and **N**ull "
            "dyads inside the triple, then `D` down, `U` up, `C` cyclic, `T` transitive -- and "
            "not chapter 41's, which names no three-node classes. Chapter 12 cites the same "
            "authors (n. 2) for the dyad census one size down, and `sna analyze` prints that one "
            "(§10.3).",
            "",
        ]
        return lines
    lines += _rows(
        ["class", "triples", "share", "what it is"],
        [
            [
                f"`{name}`",
                f"{census.counts[name]:,}",
                f"{census.counts[name] / census.triples:.4f}" if census.triples else "--",
                meaning,
            ]
            for name, meaning in (
                ("empty", "no edge between any of the three"),
                ("one edge", "one pair joined, the third node apart"),
                ("open triad", "two edges: the triad of §12.2's Figure 12.4(a)"),
                ("triangle", "all three pairs joined: Figure 12.4(b)"),
            )
        ],
    )
    try:
        clustering = census.global_clustering()
    except ValueError as exc:
        lines += [f"No clustering coefficient: {exc}", ""]
        return lines
    lines += [
        f"The last two classes are §12.2's two: *\"the two possible connected network patterns "
        f'involving three nodes"* (p. 182). Read off them, the global clustering coefficient is '
        f"`3 x {census.counts['triangle']:,} / "
        f"({census.counts['open triad']:,} + 3 x {census.counts['triangle']:,})` = "
        f"**{clustering:.4f}** -- the same number `sna analyze` prints, from the same two counts.",
        "",
    ]
    return lines


def _profile_section(report: MotifReport, profile: MotifProfile) -> list[str]:
    lines = [
        f"## Motif profile at {profile.size} nodes (§41.2)",
        "",
        f"**Sampling frame:** {report.graph.graph.get('frame', 'not recorded')}",
        "",
        f"**n:** {profile.subgraphs:,} connected subgraphs of {profile.size} nodes, over "
        f"{report.graph.number_of_nodes():,} nodes and {report.graph.number_of_edges():,} edges.",
        "",
        f"**Null model:** `configuration` -- {profile.null}. "
        f"{profile.samples:,} rewirings were built of the {profile.asked:,} asked for. The test "
        "is two-tailed, because §41.2 is as interested in a shape the network avoids as in one "
        "it repeats (p. 589).",
        "",
        "`count` is the observed number of induced connected subgraphs of that shape; `z` is how "
        "many null standard deviations away that is; `SP` is the significance profile -- the "
        "z-vector scaled to length one, so two networks of different sizes can be compared on "
        "shape alone. **The SP is not the book's**: it is the normalisation of Milo et al., "
        "whose motif paper §41.2 cites as its first reference (n. 2) and whose 1994-2004 "
        "follow-up introduced this vector.",
        "",
    ]
    lines += _rows(
        ["motif", "count", "null mean", "null sd", "z", "p", "SP", "what it is"],
        [
            [
                f"`{row.key}`",
                f"{row.count:,}",
                f"{row.test.null_mean:,.1f}",
                f"{row.test.null_std:,.1f}",
                f"{row.z:+.2f}",
                f"{row.p_value:.4f}" if row.test.testable else "--",
                f"{row.profile:+.3f}",
                row.name or (f"edges {', '.join(f'{u}-{v}' for u, v in row.edges)}"),
            ]
            for row in sorted(profile.rows, key=lambda r: -r.count)
        ],
    )
    over = profile.over()
    under = profile.under()
    lines += [
        (
            "Overexpressed (z >= 2): "
            + ", ".join(f"`{row.key}`" for row in over)
            + ". Underexpressed (z <= -2): "
            + (", ".join(f"`{row.key}`" for row in under) or "none")
            + "."
        )
        if over or under
        else "No shape is more than two null standard deviations from the null mean.",
        "",
    ]
    if profile.note:
        lines += [profile.note, ""]
    caveats = {row.test.caveat for row in profile.rows if row.test.caveat}
    lines += sorted(caveats)
    if caveats:
        lines.append("")
    return lines


def _mining_section(mining: Mining, heading: str) -> list[str]:
    lines = [
        heading,
        "",
        f"**Sampling frame:** {mining.frame}",
        "",
        f"**n:** {mining.graphs:,} graph(s) mined, patterns grown to at most {mining.max_size} "
        f"edge(s), {len(mining.patterns):,} pattern(s) at or above a support of "
        f"{mining.min_support}.",
        "",
        "**Null model:** none. A support is a count, and §41.4 sets its threshold by hand "
        "(p. 596); a frequent pattern is not thereby a surprising one. The section above is "
        "where a shape is tested against a null.",
        "",
        f"**Support means** {mining.support_means}.",
        "",
    ]
    if not mining.patterns:
        return [*lines, mining.note, ""]
    lines += _rows(
        ["pattern", "nodes", "edges", "support", "edge list"],
        [
            [
                pattern.name or "--",
                str(pattern.nodes),
                str(pattern.size),
                f"{pattern.support:,}",
                f"`{pattern.as_edge_list()}`",
            ]
            for pattern in mining.patterns
        ],
    )
    return [*lines, mining.note, ""]


def render_motifs(report: MotifReport) -> str:
    """The motifs report as markdown: what shapes are here, and which of them are surprising."""
    lines = [
        f"# Motifs and frequent subgraphs: {report.persona_id} / {report.network}",
        "",
        "**Implements:** *The Atlas for the Aspiring Network Scientist*, chapter 41 -- §41.2 "
        "(network motifs, counted and tested against a null), §41.3 (graph isomorphism, by VF2 "
        "and by exhaustive canonical labelling), §41.4 (transactional mining over a database of "
        "small graphs) and §41.5 (single-graph mining with the minimum image support). §41.1 is "
        "the machine-learning framing and is documented rather than built.",
        "",
    ]
    lines += _census_section(report)
    if report.profile is not None:
        lines += _profile_section(report, report.profile)
    if report.transactional is not None:
        lines += _mining_section(
            report.transactional, "## Frequent subgraphs across documents (§41.4)"
        )
    if report.single is not None:
        lines += _mining_section(report.single, "## Frequent subgraphs inside this network (§41.5)")
    lines += [
        "## Reading this",
        "",
        "- A motif is a count against a null, never a count. §41.2's procedure has four steps "
        'and the first one is only the first: *"you compare with your observation, so that you '
        'can build an idea of the statistical significance of the motif"* (p. 590). The null '
        "here holds the degrees fixed, so an overexpressed shape has beaten the degree sequence "
        "and not chance.",
        "- These networks are projections, and a projection makes triangles. Every network here "
        "but `relations` joins two nodes because a document or a passage held both, so a passage "
        "naming three things is a triangle by construction and a census of them counts the "
        "corpus's writing habits as much as its subject. The hypergraph (`sna layers`) is where "
        "that trio is one relation rather than three edges.",
        "- Support counts containers, not copies. Across documents (§41.4) a support of 5 means "
        "five documents held the shape at least once, not that it occurred five times; inside "
        "one network (§41.5) it is the minimum image support, which is deliberately smaller than "
        "the number of occurrences so that a larger pattern can never look more frequent than "
        "the smaller one inside it (p. 601).",
        "- A shape is not a mechanism. The feed-forward loop is overexpressed in transcription "
        "networks because it computes something (§41.2, n. 3); a feed-forward loop among "
        "extracted relations means three sentences were written, and what it computes is a "
        "question for the passages behind the edges.",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines).rstrip() + "\n"


def _mining_payload(mining: Mining) -> dict[str, Any]:
    return {
        "kind": mining.kind,
        "support_means": mining.support_means,
        "min_support": mining.min_support,
        "max_size": mining.max_size,
        "frame": mining.frame,
        "graphs": mining.graphs,
        "levels": mining.levels,
        "null_model": "none: a support is a count with a threshold chosen by hand (§41.4)",
        "patterns": [
            {
                "canonical": pattern.canonical,
                "name": pattern.name,
                "nodes": pattern.nodes,
                "edges": [list(edge) for edge in pattern.edges],
                "support": pattern.support,
            }
            for pattern in mining.patterns
        ],
        "note": mining.note,
    }


def motifs_payload(report: MotifReport) -> dict[str, Any]:
    """The same report as plain JSON-able data."""
    payload: dict[str, Any] = {
        "persona_id": report.persona_id,
        "network": report.network,
        "generated_at": report.generated_at,
        "implements": "Atlas ch. 41 (frequent subgraph mining), §41.2-§41.5",
        "frame": report.graph.graph.get("frame", ""),
        "nodes": report.graph.number_of_nodes(),
        "edges": report.graph.number_of_edges(),
        "census": {
            "directed": report.census.directed,
            "triples": report.census.triples,
            "counts": dict(report.census.counts),
        },
        "notes": report.notes,
    }
    if not report.census.directed and report.census.counts["open triad"]:
        payload["census"]["global_clustering"] = report.census.global_clustering()
    if report.profile is not None:
        profile = report.profile
        payload["profile"] = {
            "size": profile.size,
            "subgraphs": profile.subgraphs,
            "null_model": profile.null,
            "samples": profile.samples,
            "asked": profile.asked,
            "tail": "two",
            "note": profile.note,
            "motifs": [
                {
                    "key": row.key,
                    "name": row.name,
                    "edges": [list(edge) for edge in row.edges],
                    "count": row.count,
                    "null_mean": row.test.null_mean,
                    "null_std": row.test.null_std,
                    "z": row.z,
                    "p_value": row.p_value,
                    "significance_profile": row.profile,
                    "caveat": row.test.caveat,
                }
                for row in profile.rows
            ],
        }
    if report.transactional is not None:
        payload["transactional"] = _mining_payload(report.transactional)
    if report.single is not None:
        payload["single_graph"] = _mining_payload(report.single)
    return payload
