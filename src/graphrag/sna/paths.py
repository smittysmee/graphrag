"""Paths, walks, cycles and components (ch. 10), and how far apart a network is (§13.1-13.3).
Returns data; builds no graphs except the two exploration trees, and prints nothing.

**Walk, path, cycle: three words the chapter is careful about.** A *walk* is "a sequence of nodes
with the property that each node in the sequence is adjacent to the node next to it" and it may
reuse nodes and edges as often as it likes; a *path* is "a walk that does not repeat nodes nor
edges"; a *cycle* is a path that begins and ends at the same node (pp. 154-156). The length of
any of them is the number of edges crossed. Everything in this module is one of those three and
says which, because the counts are wildly different: the karate club has 45 triangles and 45
independent cycles, and an unbounded number of closed walks.

**What is here, by section.**

``walk_counts`` / ``closed_walks`` / ``triangle_count`` (§10.1)
    ``A^k`` counts walks, not paths: *"if A^n_uv = 0, then there are no walks of length n going
    from u to v [...] If, instead, A^n_uv > 0, then the A^n_uv is exactly the number of such
    walks"* (p. 155). The diagonal counts closed walks, and ``A²_uu`` is the degree (p. 156).
    All three hold only for the matrix §10.1 asks for -- *"binary and its diagonal is set to
    zero"* -- so these build that matrix rather than trust the caller to pass it: weights are
    dropped and self loops with them.

``cycle_space`` (§10.2)
    How many independent cycles the network holds, a few of the short ones, and which entry of
    the chapter's zoo the network is: tree, forest, directed tree, arborescence, directed
    acyclic, cyclic (pp. 156-158).

``dyad_census`` (§10.3)
    The three states an unordered pair of nodes can be in on a directed network -- both edges,
    one edge, no edge -- from which the chapter's reciprocity falls out as ``mutual / (mutual +
    asymmetric)``. :func:`graphrag.sna.measures.reciprocity` is the same number and this module
    re-derives it rather than importing it, so a test can hold the two together.

``components`` (§10.4)
    Weak and strong components, the giant one, the isolates, the DAG of strongly connected
    components, and the in- and out-components of the chapter's last subsection (pp. 159-163).

``exploration_order`` / ``bfs_tree`` / ``dfs_tree`` (§13.1)
    The two traversals, written the way the section writes them: the same algorithm with a
    first-in-first-out queue and with a last-in-first-out one (pp. 191-192).

``path_lengths`` (§13.2-13.3)
    The distribution of shortest-path lengths, its mean and median, the diameter, the radius,
    and each node's eccentricity, with the centre and the periphery they define.

**Length is a count of edges unless you ask otherwise.** §13.2 defines the shortest path as the
one "crossing the fewest possible number of edges", so ``weight=None`` -- hops -- is the default
everywhere here, and the ``§13.3`` histogram only exists in that case, because a sum of costs is
not a count of edges. Passing a ``weight`` is supported and switches to Dijkstra, but our edge
weights are *affinities* (a weight of 6 means "shared six documents"), so passing ``"weight"``
asks for the paths that cross the *strongest* ties and calls them long. §6.3's distinction is
the caller's to make: pass an attribute that is a cost, the way ``measures`` derives ``1 /
weight`` for betweenness and closeness.

**"Infinite" is not a number you average.** §13.3 fixes the convention -- *"Nodes in different
components are unreachable, and thus we say that their shortest path length is infinite. Thus, a
network with more than one connected component has an infinite diameter. Usually, in these
cases, what you want to look at is the diameter of the giant connected component."* So
:func:`path_lengths` measures the giant component and says, in the report, what it left out. On
a **directed** network that remedy needs one more step the book does not take: a giant *weak*
component can still hold ordered pairs that no directed path joins, so the average is taken over
the reachable ordered pairs and the unreachable ones are counted and printed rather than
dropped silently.

**What the book defines and what this adds.** §13.3 defines the path-length distribution, the
diameter and the average; the *eccentricity* of a node, and the radius, centre and periphery it
defines, are the standard companions of a diameter and are added here with that said. §10.2
defines the cycle and the acyclic graph; the *dimension* of the cycle space, ``|E| - |V| + c``,
is the standard count of independent cycles and is added the same way. §10.3 defines reciprocity
from connected pairs; the *dyad census* is the three-way split that number is a ratio of, and is
added because reciprocity alone cannot distinguish a sparse network where everything is
reciprocated from a dense one where nothing is.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.matrices import Matrix, Node, adjacency, dense, node_order
from graphrag.sna.measures import undirected_view

Traversal = Literal["bfs", "dfs"]

#: The two traversals of §13.1, by the name this module calls them.
TRAVERSALS: tuple[str, ...] = ("bfs", "dfs")

#: Above this many nodes in the component being measured, :func:`path_lengths` stops solving all
#: pairs and estimates from a seeded sample of sources instead. All-pairs shortest paths cost one
#: traversal per node -- ``O(|V||E|)`` -- and a corpus entity network carries tens of thousands of
#: nodes, so the exact answer on one would take longer than the rest of the report put together.
MAX_EXACT_NODES: int = 1_000

#: How many source nodes the estimate draws when the component is larger than that. Each source
#: contributes ``n - 1`` ordered pairs, so 200 sources of a 20,000-node component is a sample of
#: four million pairs: plenty for the mean, and nowhere near enough to be sure of the diameter,
#: which is why the estimate reports the diameter as a lower bound.
ESTIMATE_SOURCES: int = 200

#: Above this many nodes, the ``A^k`` of §10.1 is refused rather than built. The matrix is dense
#: and square, so 2,000 nodes is a 4-million-cell array and twice that is 16 million.
MAX_MATRIX_NODES: int = 2_000

#: Above this many edges, the cycle *basis* is not enumerated. Its dimension is free -- it is
#: ``|E| - |V| + c``, three counts -- but the basis itself holds that many cycles, and listing
#: half a million of them to print ten is not work worth doing.
MAX_BASIS_EDGES: int = 10_000

#: How many nodes a centre or periphery list prints before it is truncated, and how many basis
#: cycles are kept. The same 20 the reports use for a centrality ranking.
TOP_NODES: int = 20


def _sorted(nodes: Iterable[Node]) -> list[Node]:
    """Node ids in a reproducible order, whatever they are.

    This package's own networks are keyed by strings and the book's legendary graphs by
    integers, so ``sorted`` is right for both and neither is allowed to depend on the order the
    graph happened to be built in. A graph mixing the two -- which nothing here builds, but a
    graph read back from a file can be -- falls back to ordering by the text of the id, because
    an ordering that raises is worse than an ordering that is arbitrary but stable.

    ``measures._sorted_nodes`` is the same rule, arrived at independently for chapter 12's
    member lists. Neither module imports the other's private helper; if a third needs it, the
    two should become one.
    """
    listed = list(nodes)
    try:
        return sorted(listed)
    except TypeError:
        return sorted(listed, key=str)


def _loopless(graph: nx.Graph) -> nx.Graph:
    """``graph`` with its self loops removed, keeping its direction and everything else.

    §10.1 states the precondition for every reading it goes on to make: *"If A is binary and its
    diagonal is set to zero, then A^n can tell us lots of interesting things"* (p. 155). A self
    loop puts a 1 on that diagonal and each of those readings then fails quietly -- ``A²_uu``
    stops being the degree, ``trace(A³)`` counts walks that went round a loop instead of round a
    triangle, and ``(I + A)^k`` double-counts the waiting step it adds. So the diagonal is
    zeroed here, once, for every function in this module that reads a matrix power.

    The graph is copied only when there is a loop to drop, so the common case costs nothing.
    """
    loops = list(nx.selfloop_edges(graph))
    if not loops:
        return graph
    simple = graph.copy()
    simple.remove_edges_from(loops)
    return simple


def _simple_undirected(graph: nx.Graph) -> nx.Graph:
    """The undirected, self-loop-free view the cycle and triangle counts are defined on (§6.2).

    Self loops go for §10.1's reason (see :func:`_loopless`), and §10.2 adds its own: a cycle is
    a path, and a path may not repeat a node, so a loop is not a cycle of length one. Directions
    are flattened for the reason §12.1 gives for its own objects -- a cycle basis is an
    undirected construction, and ``A -> B`` beside ``B -> A`` is §10.3's reciprocated pair
    rather than a cycle of length two.
    """
    flat, _ = undirected_view(graph)
    return _loopless(nx.Graph(flat))


# ----------------------------------------------------------------- §10.1 walks and matrices


def walk_counts(
    graph: nx.Graph,
    k: int,
    nodes: Sequence[Node] | None = None,
    at_most: bool = False,
    max_nodes: int = MAX_MATRIX_NODES,
) -> tuple[Matrix, list[Node]]:
    """``A^k``: how many walks of length exactly ``k`` join each pair of nodes (§10.1).

    *"If, instead, A^n_uv > 0, then the A^n_uv is exactly the number of such walks"* (p. 155).
    These are **walks**, not paths: a walk may cross the same edge as often as it likes, so on
    the book's own figure there are two walks of length two from node 1 to node 3 and there is
    no path of length two at all between most pairs. The count of *paths* of a given length is a
    different and much harder quantity, and nothing here computes it.

    ``at_most=True`` gives ``(I + A)^k``, which §10.1 reads as the number of walks of length
    ``k`` *or less*: *"the diagonal represents self loops. If it is set to 1, the walker in u can
    choose to follow the self loop to u an arbitrary number of times before reaching v"*
    (p. 156).

    **The matrix §10.1 requires is built here rather than assumed of the caller.** The section
    opens with the precondition -- *"If A is binary and its diagonal is set to zero, then A^n
    can tell us lots of interesting things"* (p. 155) -- so this reads ``weight=None`` in
    :func:`graphrag.sna.matrices.adjacency` (weighted, the entries would be sums of products of
    weights, which count nothing) and drops self loops first, exactly as :func:`triangle_count`
    does. A loop left in place breaks the readings silently: ``a-b`` with a loop at ``a`` would
    give ``A²_aa = 2`` against a degree of 3. Direction is **not** touched, so ``A^k`` on a
    digraph counts the walks that respect it.

    Returns ``(matrix, nodes)`` under the node-order contract of chapter 8, so row ``i`` is
    ``nodes[i]``; ``nodes`` may name a subset, and dropping loops never changes which nodes
    exist.

    Undefined for a negative ``k``, which raises; ``k = 0`` is the identity, the one walk of
    length zero from each node to itself. Refused above ``max_nodes`` nodes, because the product
    is dense.
    """
    if k < 0:
        msg = f"a walk cannot have negative length, got k={k}"
        raise ValueError(msg)
    order = node_order(graph, nodes)
    if len(order) > max_nodes:
        msg = (
            f"A^k is a dense {len(order)} by {len(order)} matrix, above the {max_nodes}-node "
            "limit this module will build. Raise max_nodes if you have the memory, or ask for "
            "the quantity you wanted directly (closed_walks, triangle_count)"
        )
        raise ValueError(msg)
    matrix = dense(adjacency(_loopless(graph), nodes=order, weight=None)[0])
    if at_most:
        matrix = matrix + np.eye(len(order))
    return np.linalg.matrix_power(matrix, k), order


def closed_walks(graph: nx.Graph, k: int, max_nodes: int = MAX_MATRIX_NODES) -> dict[Node, int]:
    """How many closed walks of length ``k`` each node takes part in: the diagonal of ``A^k``
    (§10.1).

    *"A^n_uu is the number of loops or closed walks of length n -- walks starting and ending in
    the same node -- to which u participates. For n = 2 we have a special case: this value is
    equal to the degree -- or A²_uu = k_u [...] you could consider A^n_uu as a sort of
    generalized degree"* (p. 156). ``k = 3`` gives **twice** the number of triangles the node
    sits in -- each triangle can be walked round in two directions, and ``u`` is the only
    starting point a *diagonal* entry considers. The six belongs one level up, to the trace:
    summed over every node, each triangle is counted from each of its three corners in each of
    two directions, which is why :func:`triangle_count` divides ``trace(A³)`` by six and not by
    two.

    A closed walk is not a cycle. The length-2 closed walk that gives a node its degree crosses
    one edge and comes straight back, which §10.2's cycle definition forbids. Self loops are
    dropped before the power is taken (see :func:`walk_counts`), because §10.1's readings all
    assume a zero diagonal.
    """
    matrix, order = walk_counts(graph, k, max_nodes=max_nodes)
    diagonal = np.diagonal(dense(matrix))
    return {node: round(float(value)) for node, value in zip(order, diagonal, strict=True)}


def triangle_count(graph: nx.Graph, max_nodes: int = MAX_MATRIX_NODES) -> int:
    """How many triangles the network holds: ``trace(A³) / 6`` (§10.1, §10.2).

    §10.1 gives the reading and this adds the arithmetic: ``A³_uu`` counts the closed walks of
    length three through ``u``, each triangle produces six of them across the whole trace (three
    nodes to start at, two directions to go round), so the trace over six is the number of
    triangles -- the shortest cycle §10.2 admits, since a cycle may not repeat a node.

    Direction is ignored: a triangle here is a triangle of the flattened undirected view (§6.2).
    On a directed network the eight orientations of a triangle are eight different motifs, which
    is chapter 32's question and not this one's -- and is the same distinction
    ``measures.summary`` records for the Fagiolo clustering coefficient.

    The trace is computed from the matrix while the matrix is small enough to build, and from
    ``networkx.triangles`` above ``max_nodes``, where a dense cube is not worth its memory. They
    are the same number by the identity above, which is what the unit test asserts; nothing about
    the result depends on which branch ran.
    """
    flat = _simple_undirected(graph)
    if flat.number_of_nodes() > max_nodes:
        return int(sum(nx.triangles(flat).values())) // 3
    cube, _ = walk_counts(flat, 3, max_nodes=max_nodes)
    return round(float(np.trace(dense(cube))) / 6.0)


# ----------------------------------------------------------------------------- §10.2 cycles


#: The chapter's zoo of cyclic and acyclic structures (pp. 156-158, fig. 10.5), by the name it
#: gives each one. ``cyclic`` and ``directed cyclic`` are the ordinary case; the rest are the
#: structures a network has to earn.
CYCLE_KINDS: tuple[str, ...] = (
    "tree",
    "forest",
    "arborescence",
    "directed tree",
    "directed acyclic",
    "cyclic",
    "directed cyclic",
    "empty",
)


@dataclass(frozen=True)
class CycleSpace:
    """How many independent cycles a network holds, some of the short ones, and what it is."""

    #: ``|E| - |V| + c``: the number of independent cycles, the dimension of the cycle space.
    #: Zero exactly when the network is a forest. See :func:`cycle_space` for why this, and not
    #: the number of cycles, is the number worth printing.
    dimension: int
    #: The entry of §10.2's zoo this network is, one of :data:`CYCLE_KINDS`.
    kind: str
    #: Up to :data:`TOP_NODES` cycles from one basis, shortest first. Empty when the network is
    #: acyclic, and empty with a note when it was too large to enumerate.
    cycles: tuple[tuple[Node, ...], ...] = ()
    #: How many basis cycles there are of each length; empty when the basis was not enumerated.
    lengths: dict[int, int] = field(default_factory=dict)
    #: How many triangles the network holds (§10.1's ``trace(A³)/6``), which is the count of
    #: cycles of length three and the only cycle length counted exactly here.
    triangles: int = 0
    #: What the numbers above could not be, and why.
    note: str = ""


def cycle_space(
    graph: nx.Graph, max_edges: int = MAX_BASIS_EDGES, listed: int = TOP_NODES
) -> CycleSpace:
    """The independent cycles of a network, and which of §10.2's structures it is.

    §10.2 defines the cycle -- *"a path that begins and ends with the same node [...] we don't
    have any repeated nodes nor edges"* (p. 156) -- and the structures that have none: *"Trees
    are simple graphs with no cycles"* (p. 157), a forest being a graph whose every component is
    a tree. On a directed network it adds the zoo of fig. 10.5: a directed acyclic graph, a
    directed tree (acyclic even ignoring the directions, ``|V| - 1`` edges), and an arborescence,
    *"a directed tree in which all nodes have in-degree of one, except the root"* (p. 158).

    **What this adds to the section.** The *number* of cycles in a network is not a number worth
    printing -- it is exponential in the network's size, and the karate club has millions -- so
    what is reported is the dimension of the cycle space, ``|E| - |V| + c`` with ``c`` the number
    of connected components: the number of *independent* cycles, one per edge that closes a loop
    when the network is grown from a spanning forest. Adding one edge to a tree adds exactly one,
    which is the sense in which it counts the loops.

    A **basis** is enumerated below ``max_edges`` and the ``listed`` shortest of its cycles are
    kept. A cycle basis is not unique, so the shortest cycle *in this basis* is not necessarily
    the shortest cycle in the network: read the list as examples, not as extremes. The exception
    is ``triangles``, which is counted exactly from §10.1's trace.

    Direction is flattened for the dimension and the basis (§6.2), because a cycle basis is an
    undirected construction -- ``A -> B`` beside ``B -> A`` is one undirected loop, and §10.3
    calls that a reciprocated pair rather than a cycle of length two. ``kind`` is the one field
    that does read direction, since that is what fig. 10.5 is about.
    """
    if graph.number_of_nodes() == 0:
        return CycleSpace(dimension=0, kind="empty", note="The network has no nodes.")
    flat = _simple_undirected(graph)
    parts = nx.number_connected_components(flat)
    dimension = flat.number_of_edges() - flat.number_of_nodes() + parts
    kind = _cycle_kind(graph, flat, dimension)
    triangles = triangle_count(flat)
    if flat.number_of_edges() > max_edges:
        return CycleSpace(
            dimension=dimension,
            kind=kind,
            triangles=triangles,
            note=(
                f"The cycle basis was not enumerated: {flat.number_of_edges():,} undirected "
                f"edges is above the {max_edges:,} this module lists for. The dimension and the "
                "triangle count above are exact; only the example cycles are missing."
            ),
        )
    basis = [tuple(cycle) for cycle in nx.cycle_basis(flat)]
    basis.sort(key=lambda cycle: (len(cycle), [str(node) for node in cycle]))
    return CycleSpace(
        dimension=dimension,
        kind=kind,
        cycles=tuple(basis[:listed]),
        lengths=dict(sorted(Counter(len(cycle) for cycle in basis).items())),
        triangles=triangles,
        note=(
            ""
            if len(basis) <= listed
            else f"{len(basis):,} basis cycles in all; the {listed} shortest are listed."
        ),
    )


def _cycle_kind(graph: nx.Graph, flat: nx.Graph, dimension: int) -> str:
    """Which entry of fig. 10.5 this network is. See :func:`cycle_space`."""
    connected = nx.is_connected(flat) if flat.number_of_nodes() else False
    if not graph.is_directed():
        if dimension > 0:
            return "cyclic"
        return "tree" if connected else "forest"
    if not nx.is_directed_acyclic_graph(graph):
        return "directed cyclic"
    if dimension > 0 or not connected:
        return "directed acyclic"
    roots = [node for node, degree in graph.in_degree() if degree == 0]
    others = [degree for node, degree in graph.in_degree() if degree != 0]
    if len(roots) == 1 and all(degree == 1 for degree in others):
        return "arborescence"
    return "directed tree"


# ------------------------------------------------------------------------- §10.3 reciprocity


@dataclass(frozen=True)
class DyadCensus:
    """The three states an unordered pair of distinct nodes can be in, on a directed network."""

    #: Pairs with both edges: the reciprocated ones, §10.3's "cycles of length two".
    mutual: int
    #: Pairs with exactly one edge.
    asymmetric: int
    #: Pairs with no edge at all.
    null: int

    @property
    def connected(self) -> int:
        """§10.3's "connected pairs": pairs of nodes with at least one edge between them."""
        return self.mutual + self.asymmetric

    @property
    def pairs(self) -> int:
        """Every unordered pair of distinct nodes, ``n(n-1)/2``. The three counts sum to it."""
        return self.mutual + self.asymmetric + self.null

    @property
    def reciprocity(self) -> float:
        """§10.3's reciprocity: reciprocated pairs over connected pairs, 0.0 with no pairs.

        The same number :func:`graphrag.sna.measures.reciprocity` returns, and the reason to
        print the census beside it: 0.4 over five connected pairs and 0.4 over five thousand are
        the same ratio and not the same finding.
        """
        return self.mutual / self.connected if self.connected else 0.0


def dyad_census(graph: nx.Graph) -> DyadCensus:
    """Mutual, asymmetric and null pairs (§10.3), the split reciprocity is a ratio of.

    §10.3 counts pairs and not edges: *"we count the number of connected pairs of the network:
    pairs of nodes with at least one edge between them [...] Then we count the number of
    connected pairs that have both possible edges between them [...] Reciprocity is simply the
    second count over the first one"* (p. 159). Its worked example -- five connected pairs, two
    of them reciprocated -- reads 2/5, which this census gives as ``mutual=2, asymmetric=3``.

    **What this adds.** The census names the third box the ratio hides: the pairs with no edge
    at all. Reciprocity is ``mutual / (mutual + asymmetric)`` and says nothing about how many
    pairs were available to connect, so a report that prints the ratio without the census has
    printed half a number.

    Self-loops are not pairs of nodes and are counted in none of the three boxes, which is the
    convention :func:`graphrag.sna.measures.reciprocity` already uses.

    Undefined for an undirected network, where every pair with an edge is mutual by construction
    and the census is a restatement of the density. It raises rather than return that.
    """
    if not graph.is_directed():
        msg = (
            "a dyad census is defined on a directed network: undirected, every connected pair "
            "is mutual by construction and the census restates the edge count (§10.3)"
        )
        raise ValueError(msg)
    pairs: dict[frozenset[Node], tuple[Node, Node]] = {}
    for u, v in graph.edges:
        if u != v:
            pairs.setdefault(frozenset((u, v)), (u, v))
    mutual = sum(1 for u, v in pairs.values() if graph.has_edge(v, u))
    n = graph.number_of_nodes()
    return DyadCensus(
        mutual=mutual,
        asymmetric=len(pairs) - mutual,
        null=n * (n - 1) // 2 - len(pairs),
    )


# ------------------------------------------------------------------- §10.4 connected components


@dataclass(frozen=True)
class ComponentSet:
    """One way of cutting a network into components, and the giant one §10.4 expects to find."""

    #: ``connected``, ``weakly connected`` or ``strongly connected``.
    kind: str
    count: int
    #: The component sizes, largest first.
    sizes: tuple[int, ...]
    #: The members of the largest component, sorted.
    largest: tuple[Node, ...]
    #: How many components hold a single node. Under ``connected`` and ``weakly connected`` that
    #: is the isolates -- nodes with no edge at all. Under ``strongly connected`` it is every
    #: node that lies on no cycle, which on a near-acyclic network is most of them (§10.2: a
    #: node of degree one can be in no cycle, so it is its own strong component).
    singletons: int

    @property
    def giant(self) -> int:
        return self.sizes[0] if self.sizes else 0

    @property
    def giant_share(self) -> float:
        """The share of nodes in the largest component. §10.4: *"the vast majority of real
        networks host most of their nodes in a single connected component"*, which is a claim
        this number either supports or does not."""
        total = sum(self.sizes)
        return self.giant / total if total else 0.0


@dataclass(frozen=True)
class Condensation:
    """The DAG of strongly connected components, and §10.4's bow-tie around the biggest one."""

    nodes: int
    edges: int
    #: The longest chain through the DAG, measured **in edges**: a chain of four components has
    #: a depth of three. §10.4's office passing a document from desk to desk, once every cycle
    #: has been collapsed to a point -- so the number of components end to end is ``depth + 1``,
    #: and the report prints both rather than let the two be confused.
    depth: int
    #: The size of the largest strongly connected component -- §10.4's "core of the office",
    #: which "works on documents together, by passing them to each other multiple times".
    core: int
    #: Nodes outside the core that can reach it: §10.4's **in-component**, "positioned 'before'
    #: the core. Documents pass through it and they are put in the core".
    in_component: int
    #: Nodes outside the core the core can reach: the **out-component**, which the core "outputs
    #: them into".
    out_component: int
    #: Everything else: nodes that neither reach the core nor are reached by it, including every
    #: node outside the core's weak component.
    other: int


@dataclass(frozen=True)
class ComponentReport:
    """Every component reading of one network (§10.4)."""

    directed: bool
    #: The connected components of an undirected network, or the weakly connected ones of a
    #: directed network: *"take a directed network, ignore edge directions, and look for
    #: connected components"* (p. 162).
    weak: ComponentSet
    #: The strongly connected components, on a directed network only: the sets where "any u can
    #: reach any v, and vice versa" respecting the directions (p. 161).
    strong: ComponentSet | None = None
    condensation: Condensation | None = None


def _component_set(kind: str, groups: Iterable[set[Node]]) -> ComponentSet:
    """One component reading, sorted largest first. See :func:`components`."""
    members = sorted((_sorted(group) for group in groups), key=lambda g: (-len(g), str(g[0])))
    sizes = tuple(len(group) for group in members)
    return ComponentSet(
        kind=kind,
        count=len(members),
        sizes=sizes,
        largest=tuple(members[0]) if members else (),
        singletons=sum(1 for size in sizes if size == 1),
    )


def components(graph: nx.Graph) -> ComponentReport:
    """The components of a network: weak and strong on a digraph, connected on an undirected one
    (§10.4).

    §10.4 defines a connected component as a subgraph "whose nodes can be reached from one
    another by following the edges of the network", and splits the idea in two once the edges
    point. **Strong**: "u must be able to contact v, and vice versa" respecting the directions,
    so a strongly connected component is exactly a set of nodes any two of which lie on a common
    cycle (p. 161). **Weak**: the components you get by ignoring the directions (p. 162). The
    chapter's own figure 10.11(b) is the case that makes the distinction worth drawing -- one
    weak component, many strong ones -- and that is what a corpus ``relations`` network looks
    like too.

    The **condensation** is the standard companion this adds: collapse each strongly connected
    component to a point and what is left is a DAG, because any cycle between two of them would
    have merged them. Its depth is how many layers of components a message crosses, and its
    largest node is the core of §10.4's last subsection, around which that subsection's
    in-component (everything that can reach the core) and out-component (everything the core can
    reach) are counted here by name.

    **The spectral reading of the same fact is chapter 8's, and already exists.** §10.4 closes
    by counting components without traversing anything: *"the number of eigenvalues equal to one
    is the number of components in the graph"* for the stochastic adjacency, with the leading
    eigenvectors saying which node is in which, and *"similar properties hold for the
    Laplacian"*, whose *smallest* eigenvectors do the same job. Both matrices and their solver
    are :func:`graphrag.sna.matrices.stochastic`, :func:`graphrag.sna.matrices.laplacian` --
    whose docstring states the zero-multiplicity property this section points at -- and
    :func:`graphrag.sna.matrices.eigenpairs`, landed by ATL-08, so nothing here re-derives it.
    This function traverses instead, for two reasons: it needs the members and not only the
    count, and "an eigenvalue equal to one" is a floating-point judgement where a traversal is
    not.

    :func:`graphrag.sna.measures.summary` already reports the weak count and the giant share for
    every network; this is where the strong ones and the flow structure live, because they cost
    a second pass and mean nothing on the four undirected networks.
    """
    if graph.is_directed():
        strong_groups = list(nx.strongly_connected_components(graph))
        return ComponentReport(
            directed=True,
            weak=_component_set("weakly connected", nx.weakly_connected_components(graph)),
            strong=_component_set("strongly connected", strong_groups),
            condensation=_condensation(graph, strong_groups),
        )
    return ComponentReport(
        directed=False,
        weak=_component_set("connected", nx.connected_components(graph)),
    )


def _condensation(graph: nx.Graph, groups: Sequence[set[Node]]) -> Condensation | None:
    """The DAG of strongly connected components and the bow-tie round its core. See
    :func:`components`."""
    if not groups:
        return None
    dag = nx.condensation(graph, scc=groups)
    core = max(groups, key=len)
    seed = _sorted(core)[0]
    ahead = nx.ancestors(graph, seed) - core
    behind = nx.descendants(graph, seed) - core
    return Condensation(
        nodes=dag.number_of_nodes(),
        edges=dag.number_of_edges(),
        depth=int(nx.dag_longest_path_length(dag)),
        core=len(core),
        in_component=len(ahead),
        out_component=len(behind),
        other=graph.number_of_nodes() - len(core) - len(ahead) - len(behind),
    )


# ------------------------------------------------------------------------ §13.1 exploration


def exploration_order(graph: nx.Graph, source: Node, method: Traversal = "bfs") -> list[Node]:
    """The order §13.1's traversal visits the nodes in, starting at ``source``.

    The section describes both traversals as one algorithm with one difference, and this is
    written that way. *"In Depth First Search (DFS), you start by picking a root node. Then you
    put its neighbors in a Last-In-First-Out queue. You pick the last neighbor you added (you
    'pop' the queue) and you perform the same operation: you add its neighbors to the queue,
    making sure you don't add nodes you already explored"* (p. 191); BFS is *"practically
    speaking the exact same algorithm as DFS, with a tiny change. Rather than putting the
    neighbors of the root node in a Last-In-First-Out queue, you put them into a First-In-First-
    Out queue"* (p. 192).

    The consequence the section draws, and the one a test can hold this to: under DFS *"the very
    first neighbor of the root node will be the last node to be explored -- unless you encounter
    it again as the neighbor of some other node down the exploration tree"*, while BFS *"tends to
    make a gradient from the origin node to the farthest nodes in the network"* -- every
    neighbour of the root before any neighbour of a neighbour.

    Neighbours are taken in sorted order so that two runs on the same graph give the same list;
    ``networkx`` uses insertion order instead, which makes its traversal a function of how the
    graph was built. On a directed network the traversal follows the out-edges only, which is
    §10.4's rule that a walk "cannot use edges pointing to the direction opposite of the one they
    want to go".

    Returns the reachable nodes only. Everything in another component is not visited, which is
    the fact :func:`components` measures.
    """
    if method not in TRAVERSALS:
        msg = f"method must be one of {', '.join(TRAVERSALS)}, got {method!r}"
        raise ValueError(msg)
    if source not in graph:
        msg = f"source {source!r} is not a node of this network"
        raise ValueError(msg)
    seen: set[Node] = {source}
    queue: list[Node] = [source]
    order: list[Node] = []
    while queue:
        node = queue.pop() if method == "dfs" else queue.pop(0)
        order.append(node)
        for neighbour in _sorted_neighbours(graph, node):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return order


def _sorted_neighbours(graph: nx.Graph, node: Node) -> list[Node]:
    """This node's neighbours in a reproducible order, whatever its ids are."""
    return _sorted(graph.neighbors(node))


def bfs_tree(graph: nx.Graph, source: Node, depth_limit: int | None = None) -> nx.DiGraph:
    """The breadth-first exploration tree from ``source`` (§13.1).

    A thin wrapper on ``networkx``. Every node in it sits at its shortest-path distance from the
    source, because BFS reaches a node the first time by the fewest hops -- which is why §13.2
    says *"BFS finds shortest paths only for undirected unweighted graphs"*, and why the
    unweighted branch of :func:`path_lengths` is a BFS per source. It is a *tree*, so it holds
    one path per node and throws away the other shortest paths of equal length that §13.2 warns
    are common.

    ``depth_limit`` stops the exploration at that many hops, which is the ego network of that
    radius. Use :func:`exploration_order` when the question is the order rather than the tree:
    ``networkx`` explores in the graph's own adjacency order, which is insertion order.
    """
    return nx.bfs_tree(graph, source, depth_limit=depth_limit)


def dfs_tree(graph: nx.Graph, source: Node, depth_limit: int | None = None) -> nx.DiGraph:
    """The depth-first exploration tree from ``source`` (§13.1).

    A thin wrapper on ``networkx``. Unlike the BFS tree this says nothing about distance: DFS
    follows one branch to its end before backtracking, so a node one hop from the source can
    appear twenty edges down the tree. §13.2 is explicit that neither traversal solves the
    shortest-path problem in general -- *"the problem of exploring the graph via BFS or DFS is
    that they are not optimal"* -- so a DFS tree is for reachability and structure, never for
    distance.

    ``networkx`` recurses where §13.1 describes a stack, which visits the same nodes and can
    order them differently; :func:`exploration_order` is the section's own algorithm.
    """
    return nx.dfs_tree(graph, source, depth_limit=depth_limit)


# ----------------------------------------------------- §13.2-13.3 shortest paths and lengths


@dataclass(frozen=True)
class PathLengths:
    """The shortest-path length distribution of one component, and what it leaves out."""

    #: What was measured: which component, how big, and whether all pairs were solved.
    frame: str
    #: ``connected`` or ``weakly connected``: which kind of component this ran on.
    component_kind: str
    #: Nodes in the component measured.
    nodes: int
    #: Nodes in the whole network, so the share measured is visible.
    network_nodes: int
    #: Ordered pairs the sources actually reached.
    reachable_pairs: int
    #: Ordered pairs from the same sources that no path joins. §13.3 calls their length infinite;
    #: they are excluded from every average here and counted instead.
    unreachable_pairs: int
    #: Length -> number of ordered pairs at that length, for hop counts only.
    histogram: dict[int, int]
    mean: float
    median: float
    std: float
    #: The longest shortest path §13.3 calls the diameter. A **lower bound** when ``estimated``.
    diameter: float
    #: The smallest eccentricity. An **upper bound** when ``estimated``.
    radius: float
    #: Eccentricity per source: the distance from it to the furthest node it can reach.
    eccentricity: dict[Node, float]
    #: The sources whose eccentricity equals the radius, and those whose equals the diameter.
    center: tuple[Node, ...]
    periphery: tuple[Node, ...]
    #: True when the numbers come from a sample of sources rather than from every node.
    estimated: bool
    sources: int
    seed: int | None
    weight: str | None
    #: What the numbers are not, in the words the report prints.
    note: str


def path_lengths(
    graph: nx.Graph,
    weight: str | None = None,
    max_exact_nodes: int = MAX_EXACT_NODES,
    sources: int = ESTIMATE_SOURCES,
    seed: int | None = None,
) -> PathLengths:
    """The distribution of shortest-path lengths on the largest component (§13.2-13.3).

    §13.3: *"you have the path length on the x axis and the number of paths of a given length on
    the y axis"*, and from that distribution come the two numbers the section names. The
    **diameter** is the longest shortest path, *"the worst case for reachability in the network
    [...] the measure of the maximum possible separation between two nodes"*; the **average path
    length** is ``APL = Σ|P_uv| / (|V|(|V|-1))``, *"the expected length of the shortest path
    between two nodes picked at random"*.

    **Which nodes.** §13.3 fixes the convention for a disconnected network -- unreachable pairs
    have infinite length, so the diameter of such a network is infinite and *"what you want to
    look at is the diameter of the giant connected component"*. This measures that component and
    ``frame`` says which it is and what share of the network it holds. On a directed network the
    giant *weak* component still holds ordered pairs no directed path joins, so those are counted
    into ``unreachable_pairs`` and excluded from the mean: an average that included them would be
    infinite, and an average that dropped them silently would be a different quantity with the
    same name.

    **Which length.** ``weight=None`` -- the number of edges, §13.2's own definition -- is the
    default, and it is the only case with a ``histogram``, because a sum of edge costs is not a
    count of hops. A ``weight`` switches to Dijkstra and leaves the histogram empty. Read §6.3
    before passing one: our weights are affinities, so ``weight="weight"`` treats the strongest
    ties as the longest and is almost certainly not what you meant.

    **Above ``max_exact_nodes``** the all-pairs work is replaced by a seeded sample of
    ``sources`` source nodes, and ``estimated`` is set. The mean and the histogram are then
    unbiased estimates over the sampled ordered pairs; the diameter is a **lower** bound and the
    radius an **upper** bound, because an unsampled node may be further from everything than any
    sampled one and may also be more central than all of them. The centre and the periphery are
    then subsets of the sample and are labelled as such. Everything about the estimate is
    reproducible from ``seed``.

    **A negative weight is refused.** §13.2 spends its last page on why (pp. 196-197): with a
    negative edge *"by going back and forth over a negative weight we can find an equivalent
    path. At that point, we can be stuck in an infinite loop of shorter and shorter paths
    without ever reaching the destination"*, and on a directed network a *negative cycle* -- one
    whose edge weights sum below zero -- does the same. The section's instruction is to
    *"explicitly say that you're looking for paths, not walks"*, and Dijkstra, which is what
    runs here, cannot: it assumes distances only grow. Rather than return a number that answers
    no question, this raises and quotes the reason. Nothing this package builds carries a
    negative weight, so no report can reach it; a caller deriving costs by subtraction can.

    Undefined, and returned as zeros with a note, for a network with no nodes or no edges: there
    is no pair to be any distance apart.
    """
    _refuse_negative_weights(graph, weight)
    kind = "weakly connected" if graph.is_directed() else "connected"
    total = graph.number_of_nodes()
    groups = list(
        nx.weakly_connected_components(graph)
        if graph.is_directed()
        else nx.connected_components(graph)
    )
    if not groups or graph.number_of_edges() == 0:
        return _empty_lengths(kind, total, weight, seed)
    biggest = max(groups, key=len)
    component = graph.subgraph(biggest)
    order = _sorted(component.nodes)
    estimated = len(order) > max_exact_nodes
    if estimated:
        rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
        chosen = _sorted(rng.sample(order, min(sources, len(order))))
    else:
        chosen = order
    distances: Counter[float] = Counter()
    eccentricity: dict[Node, float] = {}
    reachable = 0
    for source in chosen:
        lengths = _single_source(component, source, weight)
        reached = {node: value for node, value in lengths.items() if node != source}
        reachable += len(reached)
        for value in reached.values():
            distances[float(value)] += 1
        if reached:
            # A source that reaches nothing has no eccentricity: §13.3 calls the distance to
            # every node it cannot reach infinite, and the maximum of an empty set is not 0.
            eccentricity[source] = max(reached.values())
    unreachable = len(chosen) * (len(order) - 1) - reachable
    return _lengths_from(
        kind=kind,
        component=component,
        order=order,
        chosen=chosen,
        network_nodes=total,
        distances=distances,
        eccentricity=eccentricity,
        reachable=reachable,
        unreachable=unreachable,
        estimated=estimated,
        weight=weight,
        seed=seed,
    )


def _refuse_negative_weights(graph: nx.Graph, weight: str | None) -> None:
    """Raise if an edge carries a negative ``weight``. See :func:`path_lengths` for why (§13.2)."""
    if weight is None:
        return
    offenders = [(u, v) for u, v, w in graph.edges(data=weight, default=0.0) if float(w) < 0.0]
    if offenders:
        u, v = offenders[0]
        msg = (
            f"{len(offenders)} edge(s) carry a negative {weight!r}, the first being "
            f"{u!r}-{v!r}. §13.2 (pp. 196-197): going back and forth over a negative edge gives "
            "'an infinite loop of shorter and shorter paths without ever reaching the "
            "destination', and a negative cycle does the same on a directed network, so the "
            "shortest *walk* is unbounded and only the shortest *path* is defined. Dijkstra "
            "cannot make that distinction. Transform the weights into non-negative costs first"
        )
        raise ValueError(msg)


def _single_source(graph: nx.Graph, source: Node, weight: str | None) -> dict[Node, float]:
    """Shortest-path lengths from one source: BFS when the length is hops, Dijkstra otherwise.

    The branch is on the *length*, not on the graph. ``weight=None`` means the length is a hop
    count, and a BFS answers that on a directed graph as readily as on an undirected one,
    because it follows whatever edges it is handed. A ``weight`` means the length is a sum of
    costs, which a BFS cannot accumulate, so Dijkstra runs. §13.2's figure 13.6 puts Dijkstra
    under both the weighted and the directed column, but its directed column is about respecting
    the directions, which the BFS here already does. Floyd-Warshall, the all-pairs entry of the
    same table, is not used at all -- it is ``O(|V|³)`` and dense, and repeating a single-origin
    solve per node is cheaper on every network this package builds.
    """
    if weight is None:
        return {
            n: float(d)
            for n, d in dict(nx.single_source_shortest_path_length(graph, source)).items()
        }
    return {
        n: float(d)
        for n, d in nx.single_source_dijkstra_path_length(graph, source, weight=weight).items()
    }


def _empty_lengths(kind: str, total: int, weight: str | None, seed: int | None) -> PathLengths:
    """The degenerate answer for a network with no edge to walk along."""
    return PathLengths(
        frame="No edges, so no pair of nodes is any distance apart.",
        component_kind=kind,
        nodes=0,
        network_nodes=total,
        reachable_pairs=0,
        unreachable_pairs=total * (total - 1),
        histogram={},
        mean=0.0,
        median=0.0,
        std=0.0,
        diameter=0.0,
        radius=0.0,
        eccentricity={},
        center=(),
        periphery=(),
        estimated=False,
        sources=0,
        seed=seed,
        weight=weight,
        note=(
            "Every pair is unreachable, which §13.3 calls an infinite distance. There is no "
            "average and no diameter to report."
        ),
    )


def _lengths_from(
    *,
    kind: str,
    component: nx.Graph,
    order: Sequence[Node],
    chosen: Sequence[Node],
    network_nodes: int,
    distances: Counter[float],
    eccentricity: dict[Node, float],
    reachable: int,
    unreachable: int,
    estimated: bool,
    weight: str | None,
    seed: int | None,
) -> PathLengths:
    """Assemble the distribution's statistics. See :func:`path_lengths` for what each means."""
    mean, median, std = _moments(distances)
    diameter = max(eccentricity.values(), default=0.0)
    radius = min(eccentricity.values(), default=0.0)
    share = len(order) / network_nodes if network_nodes else 0.0
    frame = (
        f"the largest {kind} component of this network: {len(order):,} of {network_nodes:,} "
        f"nodes ({share:.1%}) and {component.number_of_edges():,} edges, measured from "
        f"{len(chosen):,} source node(s) over {reachable:,} reachable ordered pair(s)."
    )
    notes: list[str] = []
    if estimated:
        notes.append(
            f"Estimated from {len(chosen):,} sources drawn at random with seed {seed}, because "
            f"the component has more than {MAX_EXACT_NODES:,} nodes. The mean and the histogram "
            "are estimates over the sampled ordered pairs; the diameter is a lower bound and the "
            "radius an upper bound, since an unsampled node can be further from everything than "
            "any sampled one, and the centre and periphery are drawn from the sample only."
        )
    if network_nodes > len(order):
        notes.append(
            f"{network_nodes - len(order):,} node(s) sit outside this component and are at "
            "infinite distance from every node in it (§13.3), so they are in none of these "
            "numbers. The diameter of the whole network is infinite."
        )
    if unreachable:
        notes.append(
            f"{unreachable:,} ordered pair(s) inside the component are not joined by any "
            "directed path, so they are excluded from the average rather than counted as "
            "infinite (§13.3). That also means the radius, the centre and the periphery are "
            "not §13.3's, which assume every node reaches every other: an eccentricity here is "
            "the distance to the furthest node that source *can* reach, so a node that reaches "
            "almost nothing looks central, and a source that reaches nothing at all has no "
            "eccentricity and is in neither list. The components table above says how much of "
            "the network is strongly connected, which is where these three are the book's."
        )
    if weight is not None:
        notes.append(
            f"Lengths are sums of the {weight!r} edge attribute, not hop counts, so there is no "
            "§13.3 histogram. Check that the attribute is a cost and not an affinity (§6.3)."
        )
    return PathLengths(
        frame=frame,
        component_kind=kind,
        nodes=len(order),
        network_nodes=network_nodes,
        reachable_pairs=reachable,
        unreachable_pairs=unreachable,
        histogram=(
            {int(length): count for length, count in sorted(distances.items())}
            if weight is None
            else {}
        ),
        mean=mean,
        median=median,
        std=std,
        diameter=diameter,
        radius=radius,
        eccentricity=eccentricity,
        center=tuple(n for n in chosen if eccentricity.get(n) == radius),
        periphery=tuple(n for n in chosen if eccentricity.get(n) == diameter),
        estimated=estimated,
        sources=len(chosen),
        seed=seed,
        weight=weight,
        note=" ".join(notes),
    )


def _moments(distances: Counter[float]) -> tuple[float, float, float]:
    """Mean, median and standard deviation of a distribution held as a tally.

    Computed from the tally rather than from an expanded sample, because the sample has one
    entry per ordered pair -- a million of them on a thousand-node component -- and the three
    numbers need only its counts. (This is why :func:`graphrag.sna.stats.describe` is not called
    here: it takes a sequence, and materialising that sequence is the expensive part.)
    """
    total = sum(distances.values())
    if not total:
        return 0.0, 0.0, 0.0
    mean = sum(length * count for length, count in distances.items()) / total
    variance = sum(count * (length - mean) ** 2 for length, count in distances.items()) / total
    seen = 0
    median = 0.0
    for length, count in sorted(distances.items()):
        seen += count
        if seen >= (total + 1) / 2:
            median = length
            break
    return mean, median, math.sqrt(variance)


# ----------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class SmallWorld:
    """The observed average path length beside the number the "small world" claim compares it to."""

    observed: float
    nodes: int
    mean_degree: float
    #: ``ln n / ln <k>``: how long paths would be if the network were random with this size and
    #: this mean degree. ``nan`` when the mean degree is 1 or less, where the logarithm is not
    #: positive and the formula says nothing.
    expected: float
    sentence: str


@dataclass(frozen=True)
class PathReport:
    """Everything the ``Paths and components`` section of a report prints."""

    frame: str
    nodes: int
    edges: int
    directed: bool
    components: ComponentReport
    lengths: PathLengths
    cycles: CycleSpace
    small_world: SmallWorld
    dyads: DyadCensus | None = None


def analyse_paths(
    graph: nx.Graph,
    weight: str | None = None,
    max_exact_nodes: int = MAX_EXACT_NODES,
    sources: int = ESTIMATE_SOURCES,
    seed: int | None = None,
) -> PathReport:
    """Measure chapter 10 and §13.1-13.3 over one network, for the report section.

    Runs :func:`components`, :func:`path_lengths`, :func:`cycle_space` and -- on a directed
    network only -- :func:`dyad_census`, and computes the small-world comparison number. Nothing
    here is a test of anything: every number is a count or a distance in the observed network,
    and the section says so where a reader would otherwise look for a null model.

    **Why the distances are here and not in the summary table.** The ticket for this chapter
    asked for the path-length numbers "in ``summary`` and the report", and they are only in the
    report. :func:`graphrag.sna.measures.summary` is called at the top of every analysis, every
    window comparison and every sampling bias report, and it takes neither a seed nor a size
    bound; a diameter is an all-pairs solve, ``O(|V||E|)``, which on a corpus entity network
    costs more than the rest of the report together. Putting it there would either make those
    three commands slow or make ``summary`` quietly approximate. Here the number can carry the
    three things that make it readable -- which component it was measured on, how many sources
    it was measured from, and the seed -- which a flat ``dict[str, float]`` has nowhere to put.
    ``summary``'s own fields and their definitions are unchanged.
    """
    lengths = path_lengths(
        graph, weight=weight, max_exact_nodes=max_exact_nodes, sources=sources, seed=seed
    )
    return PathReport(
        frame=str(graph.graph.get("frame", "")),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        directed=graph.is_directed(),
        components=components(graph),
        lengths=lengths,
        cycles=cycle_space(graph),
        small_world=_small_world(graph, lengths),
        dyads=dyad_census(graph) if graph.is_directed() else None,
    )


def _small_world(graph: nx.Graph, lengths: PathLengths) -> SmallWorld:
    """The observed average against ``ln n / ln <k>``. See :class:`SmallWorld`.

    §13.3 is where the claim is made -- *"the diameter and APL typically grow sublinearly in
    terms of number of nodes"*, the six-degrees story, Facebook's 3.57 over a billion people --
    but the section states it and does not give a formula to check it against. ``ln n / ln <k>``
    is the expected path length of a random network of the same size and mean degree, and it is
    chapter 16's, not §13.3's, so this prints the pair and stops. A comparison that is a finding
    needs the null model chapter 17 asks for -- a random graph with the same ``n`` and ``m``,
    measured the same way -- which is a different section of the report and is not run here.
    """
    flat, _ = undirected_view(graph)
    component = flat.subgraph(
        max(nx.connected_components(flat), key=len) if flat.number_of_edges() else flat.nodes
    )
    n = component.number_of_nodes()
    mean_degree = 2 * component.number_of_edges() / n if n else 0.0
    expected = math.log(n) / math.log(mean_degree) if n > 1 and mean_degree > 1 else math.nan
    if math.isnan(expected):
        sentence = (
            "The comparison number ln n / ln <k> is undefined here: the largest component has a "
            "mean degree of 1 or less, so the logarithm it divides by is not positive."
        )
    else:
        verdict = "shorter than" if lengths.mean < expected else "no shorter than"
        sentence = (
            f"Average path length {lengths.mean:.3f} against ln {n} / ln {mean_degree:.3f} = "
            f"{expected:.3f}, the expected length in a random network of the same size and mean "
            f"degree: {verdict} random. §13.3 makes the small-world claim ('the diameter and APL "
            "typically grow sublinearly in terms of number of nodes') without giving a number to "
            "check it against; this is the usual one, and it is an expectation, not a test. A "
            "test needs a random graph with this network's own n and m, measured the same way "
            "(ch. 17), which is not run in this section."
        )
    return SmallWorld(
        observed=lengths.mean,
        nodes=n,
        mean_degree=mean_degree,
        expected=expected,
        sentence=sentence,
    )


def _num(value: float) -> str:
    """A number for a report table: integers with thousands separators, the rest to 4 places."""
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.4f}"


def _names(nodes: Sequence[Node], limit: int = TOP_NODES) -> str:
    """Up to ``limit`` node ids, with the count of those left off."""
    if not nodes:
        return "-"
    shown = ", ".join(str(node) for node in nodes[:limit])
    return shown + (f", +{len(nodes) - limit} more" if len(nodes) > limit else "")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_paths(report: PathReport) -> list[str]:
    """The ``Paths and components`` section as markdown lines.

    Prints the four things a reader needs beside every number -- the sampling frame, the n, the
    null model (there is none here, and why) and the chapter each subsection implements -- and
    then the numbers.
    """
    lengths = report.lengths
    lines: list[str] = [
        "## Paths and components",
        "",
        "**Chapter.** Atlas ch. 10 -- §10.1 walks and matrix powers, §10.2 cycles, §10.3 "
        "reciprocity, §10.4 connected components -- and §13.1-13.3: graph exploration, shortest "
        "paths, and the path-length distribution.",
        "",
        "**Sampling frame.** "
        + (f"{report.frame} " if report.frame else "")
        + f"Path lengths were measured on {lengths.frame}",
        "",
        f"**n.** {report.nodes:,} nodes and {report.edges:,} edges in the network; "
        f"{lengths.nodes:,} nodes and {lengths.reachable_pairs:,} reachable ordered pairs in the "
        f"component measured"
        + (f", estimated from {lengths.sources:,} sources." if lengths.estimated else "."),
        "",
        "**Null model.** None: every number in this section is a count or a distance in the "
        "network as observed, not a comparison against anything. The small-world line below "
        "prints the expectation for a random network of the same size beside the observed "
        "average; testing that difference needs a null model (ch. 17) and is not done here.",
        "",
    ]
    lines += _render_components(report)
    lines += _render_lengths(lengths)
    lines += _render_cycles(report.cycles)
    if report.dyads is not None:
        lines += _render_dyads(report.dyads)
    return lines


def _render_components(report: PathReport) -> list[str]:
    """The component tables (§10.4)."""
    sets = [report.components.weak]
    if report.components.strong is not None:
        sets.append(report.components.strong)
    out = ["### Components (§10.4)", ""]
    out += _table(
        ["kind", "count", "largest", "share of nodes", "components of size 1"],
        [
            [
                part.kind,
                f"{part.count:,}",
                f"{part.giant:,}",
                f"{part.giant_share:.1%}",
                f"{part.singletons:,}",
            ]
            for part in sets
        ],
    )
    giant_share = report.components.weak.giant_share
    out += [
        '§10.4 expects a giant component: *"the vast majority of real networks host most of '
        f'their nodes in a single connected component"*. Here it holds {giant_share:.1%} of '
        "them.",
        "",
    ]
    condensation = report.components.condensation
    if condensation is not None:
        out += [
            f"**The DAG of strongly connected components.** {condensation.nodes:,} components "
            f"joined by {condensation.edges:,} edges; its longest chain crosses "
            f"{condensation.depth:,} edge(s), so {condensation.depth + 1:,} component(s) end to "
            "end. Collapsing every strongly connected component to a point always leaves a DAG, "
            "because a cycle between two of them would have made them one. Round the largest of "
            f"them -- §10.4's core, {condensation.core:,} node(s) -- sit the in-component "
            f"({condensation.in_component:,} nodes that can reach it and are not in it) and the "
            f"out-component ({condensation.out_component:,} nodes it can reach); "
            f"{condensation.other:,} node(s) do neither.",
            "",
        ]
    return out


def _render_lengths(lengths: PathLengths) -> list[str]:
    """The path-length distribution, the diameter and the radius (§13.2-13.3)."""
    bound = " (lower bound)" if lengths.estimated else ""
    upper = " (upper bound)" if lengths.estimated else ""
    rows = [
        ["average path length", _num(lengths.mean)],
        ["median path length", _num(lengths.median)],
        ["standard deviation", _num(lengths.std)],
        ["diameter" + bound, _num(lengths.diameter)],
        ["radius" + upper, _num(lengths.radius)],
        ["reachable ordered pairs", f"{lengths.reachable_pairs:,}"],
        ["unreachable ordered pairs", f"{lengths.unreachable_pairs:,}"],
    ]
    lines = ["### Path lengths (§13.2-13.3)", "", *_table(["measure", "value"], rows)]
    if lengths.histogram:
        total = sum(lengths.histogram.values())
        lines += _table(
            ["length (hops)", "ordered pairs", "share"],
            [
                [str(length), f"{count:,}", f"{count / total:.1%}"]
                for length, count in sorted(lengths.histogram.items())
            ],
        )
    lines += [
        'The diameter is one pair of nodes: §13.3 calls it *"the worst case for reachability"*, '
        "not a typical distance. The average is the typical one, and the histogram above is what "
        "both of them summarise.",
        "",
        f"**Centre** (eccentricity = radius): {_names(lengths.center)}.",
        "",
        f"**Periphery** (eccentricity = diameter): {_names(lengths.periphery)}.",
        "",
    ]
    if lengths.note:
        lines += [f"> {lengths.note}", ""]
    return lines


def _render_cycles(cycles: CycleSpace) -> list[str]:
    """Walks, triangles and the cycle space (§10.1-10.2)."""
    lines = [
        "### Walks and cycles (§10.1-10.2)",
        "",
        f"This network is **{cycles.kind}** in the terms of §10.2 (p. 157-158). It holds "
        f"{cycles.triangles:,} triangle(s) -- the shortest cycle there is, counted exactly as "
        "§10.1's trace(A³)/6 -- and its cycle space has dimension "
        f"{cycles.dimension:,}: that many independent cycles, |E| - |V| + components, one for "
        "each edge that closes a loop when the network is grown from a spanning forest.",
        "",
        "A walk is not a path (§10.1): A^k counts walks, which may cross the same edge as often "
        "as they like, so the number of walks between two nodes grows without limit while the "
        "number of paths does not. Nothing in this report counts paths of a given length.",
        "",
    ]
    if cycles.lengths:
        lines += _table(
            ["basis cycle length", "cycles"],
            [[str(length), f"{count:,}"] for length, count in sorted(cycles.lengths.items())],
        )
        lines += [
            "A cycle basis is not unique, so the shortest cycle in *this* basis need not be the "
            "shortest cycle in the network. Read the lengths as a description of one basis.",
            "",
        ]
    if cycles.note:
        lines += [f"> {cycles.note}", ""]
    return lines


def _render_dyads(dyads: DyadCensus) -> list[str]:
    """The dyad census and the reciprocity it is the split behind (§10.3)."""

    def share(count: int) -> str:
        return f"{count / dyads.pairs:.2%}" if dyads.pairs else "-"

    return [
        "### Reciprocity and the dyad census (§10.3)",
        "",
        *_table(
            ["dyad state", "pairs", "share of all pairs"],
            [
                ["mutual (both edges)", f"{dyads.mutual:,}", share(dyads.mutual)],
                ["asymmetric (one edge)", f"{dyads.asymmetric:,}", share(dyads.asymmetric)],
                ["null (no edge)", f"{dyads.null:,}", share(dyads.null)],
            ],
        ),
        f"Reciprocity is {dyads.reciprocity:.4f}: {dyads.mutual:,} reciprocated pair(s) over "
        f"{dyads.connected:,} connected pair(s), which is §10.3's own definition (p. 159) and "
        "the number the summary table prints. The census is here because the ratio hides the "
        "third box: the same reciprocity over five connected pairs and over five thousand is not "
        "the same finding.",
        "",
    ]


def paths_payload(report: PathReport) -> dict[str, Any]:
    """The same section as plain JSON-able data."""
    lengths = report.lengths
    payload: dict[str, Any] = {
        "frame": lengths.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "directed": report.directed,
        "components": {
            part.kind: {
                "count": part.count,
                "sizes": list(part.sizes[:TOP_NODES]),
                "largest": part.giant,
                "giant_share": part.giant_share,
                "singletons": part.singletons,
            }
            for part in [report.components.weak, report.components.strong]
            if part is not None
        },
        "path_lengths": {
            "component_kind": lengths.component_kind,
            "nodes": lengths.nodes,
            "network_nodes": lengths.network_nodes,
            "reachable_pairs": lengths.reachable_pairs,
            "unreachable_pairs": lengths.unreachable_pairs,
            "histogram": {str(length): count for length, count in lengths.histogram.items()},
            "mean": lengths.mean,
            "median": lengths.median,
            "std": lengths.std,
            "diameter": lengths.diameter,
            "radius": lengths.radius,
            "center": [str(node) for node in lengths.center[:TOP_NODES]],
            "periphery": [str(node) for node in lengths.periphery[:TOP_NODES]],
            "estimated": lengths.estimated,
            "sources": lengths.sources,
            "seed": lengths.seed,
            "weight": lengths.weight,
            "note": lengths.note,
        },
        "cycles": {
            "dimension": report.cycles.dimension,
            "kind": report.cycles.kind,
            "triangles": report.cycles.triangles,
            "lengths": {str(k): v for k, v in report.cycles.lengths.items()},
            "note": report.cycles.note,
        },
        "small_world": {
            "observed": report.small_world.observed,
            "nodes": report.small_world.nodes,
            "mean_degree": report.small_world.mean_degree,
            "expected": report.small_world.expected,
            "sentence": report.small_world.sentence,
        },
    }
    if report.components.condensation is not None:
        condensation = report.components.condensation
        payload["condensation"] = {
            "nodes": condensation.nodes,
            "edges": condensation.edges,
            "depth": condensation.depth,
            "core": condensation.core,
            "in_component": condensation.in_component,
            "out_component": condensation.out_component,
            "other": condensation.other,
        }
    if report.dyads is not None:
        payload["dyads"] = {
            "mutual": report.dyads.mutual,
            "asymmetric": report.dyads.asymmetric,
            "null": report.dyads.null,
            "connected": report.dyads.connected,
            "reciprocity": report.dyads.reciprocity,
        }
    return payload
