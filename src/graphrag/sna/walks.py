"""Random walks on a network (ch. 11) and the quantities a walker defines: where it ends up,
how long it takes to get somewhere, what that costs as an electrical resistance, where the
network is thin enough to cut, and what everybody agrees on in the end. Returns arrays and
numbers; builds no graphs and prints nothing.

**What a random walk is.** §2.7 calls it "the classical Markov process in network science": the
walker's next node depends on the node it stands on and on nothing it did before. That
memorylessness is the whole of the chapter's arithmetic -- it is what makes the one-step
transition matrix enough to describe a walk of any length -- and it is also the assumption
ATL-34's second-order networks exist to break.

**The conventions come from ATL-08 and are not re-derived here.**
:func:`graphrag.sna.matrices.stochastic` with ``orientation="row"`` *is* the transition matrix
``P`` of §8.2: a walker stands on a *row* and crosses to a *column*, so every row sums to 1.
:func:`graphrag.sna.matrices.laplacian` with ``kind="random-walk"`` is ``I - P``, and the
``combinatorial`` one is the ``L = D - A`` that the resistances below invert. Every function
here returns its numbers next to the node order that indexes them, the same
``(matrix, nodes)`` contract the matrix module publishes.

**Weights are conductances.** Our edge weights are affinities -- a weight of 6 means "shared six
documents" -- and a walker crosses a heavier edge more often, ``P[u][v] = w(u,v) / k_u`` with
``k_u`` the strength. That is the same reading §11.4 needs: in the electrical picture the weight
is the *conductance* of the edge, so one edge on its own has resistance ``1/w``. This is
``measures._with_distance``'s ``1 / weight`` arriving from the other side, and it means nothing
here has to be inverted by the caller. ``weight=None`` gives the book's unweighted network.

**What is here, by section.**

``stationary_distribution`` (§11.1)
    Where an infinitely long walk ends up. On an undirected graph it is not an eigenproblem at
    all: *"π is quite literally the normalized degree of the nodes: the degree divided by the sum
    of all degrees (2|E|)"*. On a directed graph it is the leading **left** eigenvector of ``P``
    and has to be iterated for, with the two things §11.1 warns can go wrong -- a component the
    walker cannot leave, and a node it cannot leave at all.

``non_backtracking_matrix`` / ``non_backtracking_radius`` (§11.2)
    Hashimoto's edge-to-edge operator: two rows and columns per undirected edge, one per
    direction, with a 1 wherever a walker may continue without turning round.

``hitting_time`` / ``hitting_times`` / ``commute_time`` (§11.3)
    How many steps the walk takes to arrive, and how many to arrive and come back. Asymmetric
    and symmetric respectively.

``effective_resistance`` / ``resistance_matrix`` / ``commute_matrix`` / ``commute_distance``
(§11.4)
    The same numbers as a circuit, where ``C = 2|E|Ω`` ties them back to §11.3. Ω is a proper
    metric, which shortest-path distance is not, and it moves less than shortest paths do when
    one edge appears or vanishes.

``mincut`` / ``global_mincut`` / ``fiedler_cut`` / ``fiedler_vector`` (§11.5)
    Where the network is thinnest, exactly (by max-flow) and spectrally (by the sign of the
    Fiedler vector, which is the method §11.5 actually describes).

``consensus`` / ``consensus_time`` (§11.6)
    The averaging dynamics, in both of the forms §11.6 names: the discrete one driven by ``P``
    and the continuous one driven by ``L``. They converge to **different** means, which is the
    one thing about this section that is easy to get wrong.

**What is deliberately not here.** §11.2's list of other matrix representations -- cycle, cut-set,
path and distance, modularity matrices -- is a list of things a reader should know exist, and
each belongs to the ticket that needs it (the modularity matrix to ATL-36, the distance matrix
to ATL-10). §11.5's k-cut by the ``k-1`` smallest Laplacian eigenvectors is
:func:`graphrag.sna.cluster.spectral_embedding`, which already does it and is where it stays --
with one difference a reader of figure 11.10 should know: the section plots eigenvectors of the
combinatorial Laplacian and that function embeds with the symmetric normalised one (§8.4), so
the coordinates differ while the cut they suggest is the same idea.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.matrices import (
    SPARSE_ABOVE_NODES,
    Node,
    adjacency,
    dense,
    eigenpairs,
    laplacian,
    node_order,
    stochastic,
)

Dynamics = Literal["random-walk", "laplacian"]

#: The two consensus dynamics §11.6 describes, by the name this module calls them.
DYNAMICS: tuple[str, ...] = ("random-walk", "laplacian")

#: How large a connected component this module will pseudo-invert before refusing. The
#: pseudoinverse of §11.4 is a dense ``O(n^3)`` factorisation of one component's Laplacian, and a
#: corpus entity network can carry tens of thousands of nodes, so the default stops a command
#: that would otherwise appear to hang. Every function that uses it takes ``max_nodes`` and says
#: what raising it costs.
MAX_NODES: int = 2_000

#: How many directed edges the non-backtracking matrix of §11.2 will build for. That matrix is
#: ``2|E|`` by ``2|E|``, so it grows with the square of the edge count: 2,000 directed edges is a
#: 4-million-cell array, and twice that is 16 million.
MAX_DIRECTED_EDGES: int = 2_000


# ----------------------------------------------------------------------------- shared checks


#: Why each caller refuses a directed graph. The Laplacian quantities are undefined there
#: (§8.4); the cuts and the consensus are *scope*: an s-t cut is well defined on a digraph by
#: max-flow, and the book poses §11.5's 2-cut on an undirected graph, so a directed cut is a
#: different question that this module does not answer rather than one it cannot.
_DIRECTED_REASONS: dict[str, str] = {
    "laplacian": (
        "§11.3 and §11.4 build it from the combinatorial Laplacian, which §8.4 refuses on a "
        "directed graph because a directed node has two degrees"
    ),
    "cut": (
        "§11.5 poses the minimum cut on an undirected graph and its Ω ≥ 1/cut reading needs "
        "the resistance, which is undirected; a directed s-t cut is well defined by max-flow "
        "but is a different question from the one this module answers, so it is out of scope "
        "here rather than undefined"
    ),
    "consensus": (
        "§11.6's averaging assumes the walk is irreducible and aperiodic, which a directed "
        "chain need not be, so the dynamics need not converge at all"
    ),
}


def _undirected_or_raise(graph: nx.Graph, what: str, *, because: str = "laplacian") -> None:
    """Refuse a directed graph, giving the reason that applies to *this* caller.

    ``because`` picks the sentence from :data:`_DIRECTED_REASONS`: the Laplacian quantities are
    undefined on a digraph, the cuts and the consensus are refused for scope, and the message
    says which so a reader is not told that Stoer-Wagner needs a Laplacian.
    """
    if graph.is_directed():
        msg = (
            f"{what} is defined here for an undirected graph only: {_DIRECTED_REASONS[because]}. "
            "Take measures.undirected_view(graph) if flattening is what you meant, and say in "
            "the report that you did"
        )
        raise ValueError(msg)


def _strengths(graph: nx.Graph, order: Sequence[Node], weight: str | None) -> np.ndarray:
    """The weighted degree of each node in ``order``, read off the adjacency so that a weighted
    and an unweighted call agree on what ``k_u`` means."""
    matrix, _ = adjacency(graph, order, weight=weight)
    return np.asarray(dense(matrix).sum(axis=1), dtype=np.float64)


def volume(
    graph: nx.Graph, nodes: Sequence[Node] | None = None, weight: str | None = "weight"
) -> float:
    """``2|E|``: the sum of every degree, which is the constant §11.1 divides by and §11.4
    multiplies the resistance matrix by.

    On a weighted network this is the sum of the *strengths*, twice the total edge weight rather
    than twice the edge count, because that is the normaliser that keeps ``k_v / 2|E|`` a
    probability distribution. Zero for a graph with no edges, where nothing in this module is
    defined.
    """
    order = node_order(graph, nodes)
    return float(_strengths(graph, order, weight).sum())


# ----------------------------------------------------------------- §11.1 stationary distribution


def stationary_distribution(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    tolerance: float = 1e-12,
    max_steps: int = 10_000,
    sparse_above: int = SPARSE_ABOVE_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """π (§11.1): the probability of standing on each node after an infinitely long random walk.

    Returns ``(pi, nodes)`` under the module's node-order contract, with ``pi`` summing to 1.
    The number means: run a walker forever and it spends ``pi[i]`` of its steps on ``nodes[i]``,
    whatever node it started from. It is defined by ``πP = π`` -- π is the leading *left*
    eigenvector of the transition matrix, for the eigenvalue 1 that every stochastic matrix has.

    **Undirected: this is degree, and nothing else.** §11.1 is blunt about it -- *"π is quite
    literally the normalized degree of the nodes: the degree divided by the sum of all degrees
    (2|E|)"* -- so the closed form ``k_v / 2|E|`` is what this computes, with no iteration and no
    eigensolver. On a weighted graph ``k_v`` is the strength. A ranking by π is therefore a
    ranking by degree with a different scale on the axis, and reporting it as a separate finding
    is reporting degree twice. The teleporting version, which is *not* degree, is PageRank:
    ``measures.centrality(graph, "pagerank")``, which §11.1 says is *"almost exactly"* this
    *"plus/minus some bells and whistles"*.

    **Disconnected: one distribution per component, and they are not comparable.** §11.1 walks
    through this case: no power of ``P`` ever carries mass across, so *"we end up with two
    stationary distributions, one for one component, and one for the other ... effectively
    telling you something about two different networks"*. The closed form still answers, and what
    it returns is those distributions mixed in proportion to each component's share of the total
    degree; divide a component's entries by their own sum to recover the walk that actually runs
    inside it. An isolated node gets 0, which is §11.1's own reading of the zeros in the
    eigenvector: there is no edge a walker could use to arrive.

    **Directed: iterated, and refused when the iteration would be meaningless.** There is no
    closed form -- the in-degree is not the answer -- so this runs the power iteration §11.1
    describes as ``A^30``, left-multiplying a uniform vector by ``P`` until it stops moving by
    more than ``tolerance`` in total variation. Two things stop that being a distribution, and
    both raise rather than return a number:

    * a node with **no out-edges**, where the walker arrives and the walk ends. Its row of ``P``
      sums to 0 rather than 1 (§8.2), so probability drains out of the system on every step and
      the limit is the zero vector.
    * a graph that is **not strongly connected**, where §11.1's two-components argument applies
      with a sharper edge than it does undirected: the walker can fall into a trap it cannot
      leave, so the limit depends on where it started and there is no single π.

    In both cases the fix is the same one §11.1 names, and it is a different measure rather than
    a repair of this one: PageRank's teleport gives every node a small chance of jumping
    anywhere, which makes the chain irreducible and aperiodic by construction. Use
    ``measures.centrality(graph, "pagerank")``; do not reach for it by patching this.

    A chain that is irreducible but **periodic** -- a directed cycle is the clean example -- has
    exactly one π, but ``P^n`` oscillates instead of converging to it. Starting the iteration
    from the uniform vector finds the fixed point of a directed cycle immediately, because
    uniform already is it; where it does not, this raises on ``max_steps`` and names the lazy
    walk (iterate ``(I + P)/2``, same π) as the standard fix.

    ``sparse_above`` is ATL-ENT-3's scale knob for the directed (iterated) path: above this many
    nodes, ``P`` is built and multiplied sparsely rather than densified first. It does not change
    any number the closed-form undirected path returns, only when the directed one stops paying
    for zeros it never needed to store; see :func:`_directed_stationary`.
    """
    order = node_order(graph, nodes)
    if not order:
        return np.zeros(0, dtype=np.float64), order
    if not graph.is_directed():
        strengths = _strengths(graph, order, weight)
        total = float(strengths.sum())
        if total <= 0.0:
            msg = (
                "the stationary distribution is undefined on a network with no edges: a walker "
                "has nowhere to be (§11.1)"
            )
            raise ValueError(msg)
        return strengths / total, order
    return _directed_stationary(graph, order, weight, tolerance, max_steps, sparse_above), order


def _directed_stationary(
    graph: nx.Graph,
    order: Sequence[Node],
    weight: str | None,
    tolerance: float,
    max_steps: int,
    sparse_above: int = SPARSE_ABOVE_NODES,
) -> np.ndarray:
    """π of a directed chain by power iteration on ``P``. See :func:`stationary_distribution`.

    ATL-ENT-3: above ``sparse_above`` nodes, ``P`` is built and iterated as a ``scipy.sparse``
    matrix rather than densified first -- ``distribution @ P`` costs one multiply per non-zero
    entry either way, so the only thing a dense ``P`` buys here is memory this function then
    wastes holding zeros. The row-sum check below and every step of the loop read the same
    numbers whichever form ``P`` takes, which is what the sparse and dense fixtures being
    asserted equal in the tests actually holds this to.
    """
    use_sparse = len(order) > sparse_above
    matrix, _ = stochastic(graph, order, orientation="row", weight=weight, sparse=use_sparse)
    row_sums = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    dangling = [node for node, total in zip(order, row_sums, strict=True) if total <= 0.0]
    if dangling:
        named = ", ".join(str(node) for node in dangling[:5])
        msg = (
            f"no stationary distribution: {len(dangling)} node(s) have no out-edge ({named}), so "
            "a walker that arrives there never leaves and probability drains away on every step "
            "instead of converging (§11.1, §8.2). The fix §11.1 names is the teleport, which is a "
            "different measure: measures.centrality(graph, 'pagerank')"
        )
        raise ValueError(msg)
    if not nx.is_strongly_connected(graph.subgraph(order)):
        msg = (
            "no single stationary distribution: this directed network is not strongly connected, "
            "so §11.1's two-component argument applies -- the walker can reach a part it cannot "
            "leave, and the limit depends on where it started. Analyse one strongly connected "
            "component at a time, or use the teleport: measures.centrality(graph, 'pagerank')"
        )
        raise ValueError(msg)
    size = len(order)
    distribution = np.full(size, 1.0 / size, dtype=np.float64)
    for _ in range(max_steps):
        moved = np.asarray(distribution @ matrix, dtype=np.float64).ravel()
        moved = moved / float(moved.sum())
        if float(np.abs(moved - distribution).sum()) < tolerance:
            return np.asarray(moved, dtype=np.float64)
        distribution = moved
    msg = (
        f"the power iteration did not settle in {max_steps} steps: this chain is almost certainly "
        "periodic, so P^n oscillates rather than converging even though exactly one π exists "
        "(§11.1). Iterate the lazy walk (I + P)/2, which has the same π and cannot oscillate, or "
        "use measures.centrality(graph, 'pagerank')"
    )
    raise ValueError(msg)


# ------------------------------------------------------------------ §11.2 non-backtracking walks


def directed_edges(graph: nx.Graph, nodes: Sequence[Node] | None = None) -> list[tuple[Node, Node]]:
    """The edge directions of ``graph``, in the order that indexes the non-backtracking matrix.

    §11.2 builds its matrix with *"one row/column per edge direction"*, treating *"an undirected
    graph as a directed one with perfect reciprocity"*, so an undirected edge appears twice, once
    each way. Sorted by the node order, so the layout is a function of the data alone in the same
    way :func:`graphrag.sna.matrices.node_order` makes the adjacency matrix's layout one.

    Self-loops are dropped. ``NB[uv, vz] = 1 if u != z`` has no honest answer for ``u -> u``:
    every continuation both is and is not a backtrack, and the block-diagonal zero §11.2 relies
    on stops meaning what it means.
    """
    order = node_order(graph, nodes)
    index = {node: position for position, node in enumerate(order)}
    if graph.is_directed():
        pairs = [(u, v) for u, v in graph.edges if u in index and v in index and u != v]
    else:
        pairs = [
            direction
            for u, v in graph.edges
            if u in index and v in index and u != v
            for direction in ((u, v), (v, u))
        ]
    return sorted(pairs, key=lambda pair: (index[pair[0]], index[pair[1]]))


def non_backtracking_matrix(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    max_edges: int = MAX_DIRECTED_EDGES,
) -> tuple[np.ndarray, list[tuple[Node, Node]]]:
    """Hashimoto's non-backtracking matrix ``NB`` (§11.2), and the edge directions indexing it.

    A walk is non-backtracking when the walker may not *"re-use the same edge twice in a row"*.
    That is not a property of the node it stands on -- it depends on the edge it arrived by -- so
    the matrix cannot be node-by-node. §11.2 gives it a row and a column per edge direction:

        ``NB[(u,v), (v,z)] = 1 if u != z, else 0``

    Read a row as "I got into ``v`` from ``u``; where may I go next?". Every legal continuation
    leaves ``v``, and the one continuation forbidden is the one back to ``u``. Returns a dense
    binary array and the ``[(u, v), ...]`` list its rows and columns follow, so entry ``(i, j)``
    is the transition from ``edges[i]`` to ``edges[j]``.

    Three properties §11.2 states, and which the tests hold this to. It is ``2|E| x 2|E|`` for an
    undirected graph. It is **not symmetric** -- the section's own example: going red-to-blue
    permits blue-to-purple, but going blue-to-purple does not permit red-to-blue. And the block
    diagonal is zero, because *"a one in the diagonal is exactly a backtracking move"*.

    Row sums are ``k_v - 1`` rather than ``k_v``: on a ``k``-regular graph every row sums to
    ``k - 1``, which is where :func:`non_backtracking_radius` gets its answer. A row for an edge
    into a degree-1 node is **all zero** -- a walker that runs into a pendant node has no legal
    move at all, and the honest matrix says so rather than letting it turn round.

    Undefined, and raised on, above ``max_edges`` directions: the array is dense and quadratic in
    the edge count, so the guard is the difference between a slow command and one that exhausts
    memory. The unweighted structure is all §11.2 uses, so edge weights are ignored here.
    """
    edges = directed_edges(graph, nodes)
    if len(edges) > max_edges:
        msg = (
            f"non_backtracking_matrix would build a {len(edges)} by {len(edges)} dense array "
            f"({len(edges) ** 2:,} cells) for {len(edges)} edge directions, above max_edges="
            f"{max_edges}. §11.2's matrix is quadratic in the edge count; raise max_edges only "
            "with the memory to hold the square"
        )
        raise ValueError(msg)
    size = len(edges)
    matrix = np.zeros((size, size), dtype=np.float64)
    starting: dict[Node, list[int]] = {}
    for position, (u, _) in enumerate(edges):
        starting.setdefault(u, []).append(position)
    for row, (u, v) in enumerate(edges):
        for column in starting.get(v, ()):
            if edges[column][1] != u:
                matrix[row, column] = 1.0
    return matrix, edges


def non_backtracking_radius(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    max_edges: int = MAX_DIRECTED_EDGES,
) -> float:
    """The largest eigenvalue of the non-backtracking matrix, in absolute value (§11.2).

    This is the growth rate of non-backtracking walks: the number of them of length ``n`` grows
    like ``radius ** n``. On a ``k``-regular graph it is exactly ``k - 1``, since every row of
    ``NB`` sums to ``k - 1`` -- a triangle gives 1 (one legal continuation per arrival) and
    ``K4`` gives 2.

    **Why §11.2 says it is worth having.** The section lists what the non-backtracking matrix is
    used for, and the first entry is *"fixing eigenvector centrality degeneration"* (Martin,
    Zhang and Newman 2014; §14.4). The failure it fixes is this: the leading eigenvector of ``A``
    can *localise*, piling nearly all of its weight onto one hub and its immediate neighbours,
    because the hub and a neighbour keep reinforcing each other across the same edge -- a
    two-step walk out and back. The non-backtracking operator forbids exactly that step, so the
    hub cannot vote for itself through its own neighbour, and the centrality it defines stays
    spread over the network. The section's other three uses: community detection in sparse
    graphs, where ``NB``'s spectrum keeps a detectability signal ``A``'s loses (Krzakala et al.
    2013, Part X); percolation (Karrer, Newman, Zdeborová 2014, ch. 22); and counting motifs
    (Torres et al. 2019, ch. 41).

    ``NB`` is not symmetric, so its eigenvalues may be complex and this returns the **spectral
    radius** -- the largest modulus -- rather than a largest eigenvalue that might not be real.
    That is also why it does not go through :func:`graphrag.sna.matrices.eigenpairs`, which
    refuses a complex spectrum by design (§8.4). On a connected graph that is not a tree or a
    cycle the leading eigenvalue is real and positive by Perron-Frobenius, and the modulus is it.

    Zero on a **tree** and on any graph with no cycle: ``NB`` is nilpotent there, because every
    non-backtracking walk on a tree runs into a leaf and stops. Zero on an empty graph. Both are
    correct answers rather than failures, and both mean "no non-backtracking walk survives".
    """
    matrix, edges = non_backtracking_matrix(graph, nodes, max_edges=max_edges)
    if not edges:
        return 0.0
    return float(np.abs(np.linalg.eigvals(matrix)).max())


# ------------------------------------------------- §11.3-11.4 hitting time, effective resistance


def _pseudoinverses(
    graph: nx.Graph,
    order: Sequence[Node],
    weight: str | None,
    max_nodes: int,
) -> tuple[list[list[int]], list[np.ndarray], np.ndarray]:
    """One Moore-Penrose pseudoinverse of the combinatorial Laplacian per connected component.

    Returns ``(blocks, inverses, strengths)``: the positions of each component in ``order``, that
    component's ``L†``, and every node's strength. Component-wise rather than whole-graph because
    the numbers of §11.3 and §11.4 are undefined *between* components -- the walker never arrives
    -- while inside each one they are perfectly well defined, and §11.1 has already said that a
    disconnected graph is two networks rather than one bad one.
    """
    _undirected_or_raise(graph, "this quantity")
    view = graph.subgraph(order)
    components = [
        sorted(position for position, node in enumerate(order) if node in group)
        for group in nx.connected_components(view)
    ]
    largest = max((len(block) for block in components), default=0)
    if largest > max_nodes:
        msg = (
            f"the largest connected component here has {largest:,} nodes, above max_nodes="
            f"{max_nodes:,}. §11.4's formula pseudo-inverts that component's Laplacian densely, "
            "which costs O(n^3) time and n^2 memory; raise max_nodes deliberately, or take a "
            "sample first (graphrag.sna.sampling)"
        )
        raise ValueError(msg)
    strengths = _strengths(graph, order, weight)
    inverses: list[np.ndarray] = []
    for block in components:
        members = [order[position] for position in block]
        matrix, _ = laplacian(view, members, kind="combinatorial", weight=weight)
        inverses.append(np.linalg.pinv(dense(matrix), hermitian=True))
    return components, inverses, strengths


def laplacian_pseudoinverse(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """``L†``, the Moore-Penrose pseudoinverse of the combinatorial Laplacian (§11.4).

    Returns ``(matrix, nodes)``. §11.4 explains why the dagger and not an inverse: *"the
    Laplacian is a singular matrix, which means it cannot be inverted"* -- its rows sum to zero,
    so the vector of ones is in its kernel -- and the pseudoinverse is what stands in, built from
    the SVD by *"[reciprocating] L's singular values"* and satisfying ``L L† L = L``.

    The book writes the object it needs as ``Γ = (L + 1/|V| · 1)†``, adding ``1/|V|`` to every
    entry first. That matrix is genuinely invertible for a connected graph, and ``Γ = L† + J/|V|``
    exactly; the difference cancels in :func:`effective_resistance`, where the three terms
    contribute ``1/|V| + 1/|V| - 2/|V| = 0``. Both routes are computed and compared in the tests,
    because that identity is a known answer rather than an implementation detail.

    Computed per connected component and assembled, so that a component's ``L†`` never carries
    another component's nodes. Cross-component entries are 0. A directed graph is refused, as
    §8.4 refuses to give it a Laplacian.
    """
    order = node_order(graph, nodes)
    blocks, inverses, _ = _pseudoinverses(graph, order, weight, max_nodes)
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    for block, inverse in zip(blocks, inverses, strict=True):
        positions = np.array(block, dtype=np.int64)
        matrix[np.ix_(positions, positions)] = inverse
    return matrix, order


def hitting_times(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """The hitting time matrix ``H`` (§11.3): ``H[i][j]`` is the expected number of steps a
    random walker starting at ``nodes[i]`` takes before it first stands on ``nodes[j]``.

    Returns ``(matrix, nodes)``. The diagonal is 0 by definition -- §11.3's footnote is emphatic
    that ``H[u][u]`` is *"the hitting time of the origin from the origin"* and not the time to go
    away and come back. **``H`` is not symmetric**, which is the point of having it: §11.3 notes
    that the formula depends on the degree of the destination and not of the origin, so a hub is
    hit quickly and left slowly. The three-node chain is the section's own example -- one step
    from an end node to the middle, three steps back -- and this function reproduces it.

    **The route taken here, and why it is not the book's.** §11.3 derives ``H`` from the
    eigendecomposition of ``N = D^-1/2 A D^-1/2`` and then warns, in exercise 3, that *"a naive
    implementation in python using numpy and scipy might lead to the wrong result"* and suggests
    Octave. The warning is about that formula: it subtracts nearly equal quantities, so it loses
    precision badly whenever an eigenvalue sits close to 1. This computes the same matrix from
    the Laplacian pseudoinverse instead, which §11.4 introduces two pages later and which is
    stable:

        ``H[u][v] = Σ_w k_w (L†[v][v] - L†[v][w] - L†[u][v] + L†[u][w])``

    The two agree analytically -- their sum ``H[u][v] + H[v][u]`` collapses to ``2|E| Ω[u][v]``,
    which is exactly §11.4's ``C = 2|E|Ω`` -- and this one agrees with the section's worked
    example to machine precision, which the eigen route does not. :func:`random_target_time`
    keeps the book's eigen formula, where the same cancellation does not arise.

    **Undefined between components, and reported as ``inf``.** A walker cannot reach what it
    cannot reach, so the expected number of steps is infinite rather than large. Inside each
    component the numbers are finite and use *that component's* ``2|E|``, which is §11.1's
    "two different networks" applied consistently: do not compare a hitting time in one component
    with a hitting time in another.

    On a weighted graph ``k_w`` is the strength, and the walker crosses heavy edges more often.
    A directed graph is refused: its hitting times need the fundamental matrix of the directed
    chain, not a Laplacian, and §11.3 does not cover that case.
    """
    order = node_order(graph, nodes)
    blocks, inverses, strengths = _pseudoinverses(graph, order, weight, max_nodes)
    matrix = np.full((len(order), len(order)), np.inf, dtype=np.float64)
    for block, inverse in zip(blocks, inverses, strict=True):
        positions = np.array(block, dtype=np.int64)
        degrees = strengths[positions]
        diagonal = np.diag(inverse)
        # H[u][v] = Σ_w k_w (L†vv - L†vw - L†uv + L†uw), written as one outer-product expression.
        weighted = inverse @ degrees
        total = float(degrees.sum())
        rows = weighted[:, np.newaxis] - inverse * total
        columns = diagonal[np.newaxis, :] * total - weighted[np.newaxis, :]
        block_times = rows + columns
        np.fill_diagonal(block_times, 0.0)
        matrix[np.ix_(positions, positions)] = block_times
    return matrix, order


def hitting_time(
    graph: nx.Graph,
    source: Node,
    target: Node,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> float:
    """``H[source][target]`` (§11.3): expected steps for a walker starting at ``source`` to first
    stand on ``target``.

    ``0.0`` when the two are the same node, ``inf`` when no walk joins them. Asymmetric: this is
    not ``hitting_time(graph, target, source)``, and on the three-node chain of §11.3 the two
    differ by a factor of three. See :func:`hitting_times` for the formula, the precision note
    and the component rule; this computes the whole matrix and reads one cell, because the
    pseudoinverse the cell needs is the pseudoinverse the matrix needs.
    """
    order = _pair_order(graph, source, target)
    matrix, _ = hitting_times(graph, order, weight=weight, max_nodes=max_nodes)
    return float(matrix[order.index(source), order.index(target)])


def commute_time(
    graph: nx.Graph,
    u: Node,
    v: Node,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> float:
    """The commute time ``C[u][v] = H[u][v] + H[v][u]`` (§11.3): expected steps to walk from
    ``u`` to ``v`` *and back*.

    Symmetric *"trivially"*, as §11.3 puts it, since it is a sum of the two directions, and tied
    to §11.4 by ``C = 2|E| Ω``. ``0.0`` for a node with itself, ``inf`` across components.

    A commute time is not a distance, even though it is symmetric and zero on the diagonal: it
    grows with the size of the graph through the ``2|E|`` factor, so the same pair in a larger
    network commutes more slowly without being any further apart. :func:`effective_resistance`
    is the size-free version and :func:`commute_distance` is the metric built from this one.
    """
    order = _pair_order(graph, u, v)
    matrix, _ = hitting_times(graph, order, weight=weight, max_nodes=max_nodes)
    i, j = order.index(u), order.index(v)
    return float(matrix[i, j] + matrix[j, i])


def _pair_order(graph: nx.Graph, u: Node, v: Node) -> list[Node]:
    """The node order for a two-node question: the whole graph, with both endpoints checked.

    The quantities of §11.3 and §11.4 are properties of the *whole* network -- every path between
    the two nodes contributes -- so a pair cannot be answered from a subgraph. This only exists
    to turn an unknown node id into a clear error rather than a ``KeyError`` from inside numpy.
    """
    order = node_order(graph)
    missing = [str(node) for node in (u, v) if node not in graph]
    if missing:
        msg = f"not in the network: {', '.join(missing)}"
        raise ValueError(msg)
    return order


def random_target_time(
    graph: nx.Graph, nodes: Sequence[Node] | None = None, weight: str | None = "weight"
) -> float:
    """The Random Target Lemma's constant (§11.3): the expected number of steps to hit a
    destination drawn from the stationary distribution, which does not depend on where you start.

    §11.3: *"suppose you start from node u and you pick destinations v at random. What's the
    expected hitting time?"* Averaging ``H[u][v]`` over ``v`` weighted by π gives

        ``Σ_v π_v H[u][v] = Σ_{n=2}^{|V|} 1 / (1 - λ_n)``

    and *"the right hand side has no trace of u"*. The λ are the eigenvalues of
    ``N = D^-1/2 A D^-1/2``, descending, with the leading one (always 1, since ``N`` is similar
    to the stochastic matrix) skipped -- it is the term that would divide by zero.

    This keeps the book's eigen route, unlike :func:`hitting_times`: here the eigenvalues are
    summed rather than differenced, so the cancellation the book's exercise 3 warns about does
    not arise. The tests check the two against each other on a graph whose hitting times are
    known by hand, which is the strongest statement either formula can make.

    Undefined for a disconnected graph -- the second eigenvalue is also 1 there, the sum divides
    by zero, and §11.1 has already said the components are two networks -- and for a graph with
    no edges. Both raise.

    **Stays dense regardless of size (ATL-ENT-3).** The formula sums over every eigenvalue but
    the leading one, so it needs the *whole* spectrum, not a handful of extremes -- exactly the
    case :func:`~graphrag.sna.matrices.eigenpairs`'s sparse path is not for. This is the one
    walk quantity in this module that a large network makes slower rather than merely bigger.
    """
    _undirected_or_raise(graph, "the random target time")
    order = node_order(graph, nodes)
    view = graph.subgraph(order)
    if not order or view.number_of_edges() == 0:
        msg = "the random target time is undefined on a network with no edges (§11.3)"
        raise ValueError(msg)
    if not nx.is_connected(view):
        msg = (
            "the random target time is undefined on a disconnected network: the eigenvalue 1 of "
            "N is repeated once per component, so the 1/(1 - λ) term is infinite, and §11.1 "
            "reads the components as separate networks anyway. Run it per component"
        )
        raise ValueError(msg)
    matrix, _ = adjacency(view, order, weight=weight)
    strengths = _strengths(view, order, weight)
    scale = 1.0 / np.sqrt(strengths)
    normalised = dense(matrix) * scale[:, np.newaxis] * scale[np.newaxis, :]
    values, _ = eigenpairs(normalised, largest=True)
    return float(np.sum(1.0 / (1.0 - values[1:])))


def resistance_matrix(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """The effective resistance matrix Ω (§11.4), *"one of the awesomest matrices in network
    science"*.

    Returns ``(matrix, nodes)``. ``Ω[i][j]`` is what an ohmmeter would read between the two nodes
    if every edge of weight ``w`` were a resistor of ``1/w`` ohms -- §11.4's original definition,
    with the book's unit-ohm resistor being the ``weight=None`` case. §11.4's formula, computed
    here from the Laplacian pseudoinverse:

        ``Ω[u][v] = L†[u][u] + L†[v][v] - 2 L†[u][v]``

    **What the number means.** It is the commute time with the size of the graph divided out:
    ``C = 2|E| Ω``, so Ω is *"a sort of normalized commute time"*. Low means many short parallel
    routes; high means the two nodes hang off each other through few, long ones. Two nodes joined
    by a single edge and nothing else are at resistance 1 from each other; add a second
    independent route and the resistance falls below 1, the way parallel resistors do. That is
    the whole difference from shortest-path distance, which counts one route and ignores the
    rest -- and in a **tree**, where there is only ever one route, Ω is exactly the hop count and
    tells you nothing a shortest path did not.

    **Why §11.4 prefers it to shortest paths.** It is a *proper metric*, which shortest-path
    distance on a weighted graph is not, so the operations of ch. 47 that need one are defined on
    it. And it is stable: *"the removal or introduction of a single edge can radically change the
    shortest path distance between two nodes, but Ω will change less abruptly"*. The section's
    own simulation: adding one edge cut a shortest path by a factor of 7 and the resistance
    between the same pair by less than 3. On noisy, error-prone corpus data (ch. 28) that is the
    difference between a measurement and an artefact.

    ``inf`` between two components, where no current flows; ``0`` on the diagonal. Computed per
    component for the same reason :func:`hitting_times` is. A directed graph is refused: a
    resistor has no direction, and §8.4 will not give a directed graph a Laplacian.
    """
    order = node_order(graph, nodes)
    blocks, inverses, _ = _pseudoinverses(graph, order, weight, max_nodes)
    matrix = np.full((len(order), len(order)), np.inf, dtype=np.float64)
    for block, inverse in zip(blocks, inverses, strict=True):
        positions = np.array(block, dtype=np.int64)
        diagonal = np.diag(inverse)
        resistances = diagonal[:, np.newaxis] + diagonal[np.newaxis, :] - 2.0 * inverse
        np.fill_diagonal(resistances, 0.0)
        matrix[np.ix_(positions, positions)] = resistances
    return matrix, order


def effective_resistance(
    graph: nx.Graph,
    u: Node,
    v: Node,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> float:
    """``Ω[u][v]`` (§11.4): the effective resistance between two nodes, in ohms if every edge of
    weight ``w`` is a ``1/w``-ohm resistor.

    ``0.0`` for a node with itself and ``inf`` when no path joins the two -- an open circuit.
    See :func:`resistance_matrix` for what the number means, why §11.4 prefers it to a shortest
    path, and the one case (a tree) where it is a shortest path.
    """
    order = _pair_order(graph, u, v)
    matrix, _ = resistance_matrix(graph, order, weight=weight, max_nodes=max_nodes)
    return float(matrix[order.index(u), order.index(v)])


def commute_matrix(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """The commute time matrix ``C = 2|E| Ω`` (§11.3, §11.4): expected steps to walk from one
    node to another and back.

    Returns ``(matrix, nodes)``. Symmetric, zero on the diagonal, ``inf`` across components.
    Equal entry by entry to ``H + H.T`` from :func:`hitting_times`, which is the identity the
    tests check in both directions.

    Each component is scaled by **its own** ``2|E|``, because that is the graph the walk is
    actually running on: a walker inside one component never learns that the other exists. On a
    connected graph this is the ``2|E|`` of the whole network, and the identity reads exactly as
    §11.4 writes it.
    """
    order = node_order(graph, nodes)
    blocks, inverses, strengths = _pseudoinverses(graph, order, weight, max_nodes)
    matrix = np.full((len(order), len(order)), np.inf, dtype=np.float64)
    for block, inverse in zip(blocks, inverses, strict=True):
        positions = np.array(block, dtype=np.int64)
        diagonal = np.diag(inverse)
        resistances = diagonal[:, np.newaxis] + diagonal[np.newaxis, :] - 2.0 * inverse
        np.fill_diagonal(resistances, 0.0)
        matrix[np.ix_(positions, positions)] = float(strengths[positions].sum()) * resistances
    return matrix, order


def commute_distance(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    max_nodes: int = MAX_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """The commute-time **distance**, ``sqrt(C)`` (§11.3, §11.4), for the measures that need one.

    Returns ``(matrix, nodes)``. The square root is what makes it a Euclidean distance: the
    nodes can be embedded in a space where this is their straight-line separation, which
    ``C`` itself cannot claim and which is the property node similarity (ATL-15) and the
    random-walk projection (ATL-26, §26.5) reach for.

    §11.4's own metric is the resistance, ``Ω = C / 2|E|``, and it is a metric without a square
    root; this differs from ``sqrt(Ω)`` only by the constant ``sqrt(2|E|)``, so a *ranking* by
    either is the same ranking. Which to hand a clustering algorithm depends on whether it wants
    a squared distance or a distance -- state which one the report used.

    ``inf`` across components, where the walk never commutes.
    """
    matrix, order = commute_matrix(graph, nodes, weight=weight, max_nodes=max_nodes)
    return np.sqrt(matrix), order


# ------------------------------------------------------------------------- §11.5 the mincut


@dataclass(frozen=True)
class Cut:
    """One partition of a network into two sides, and the weight crossing between them."""

    value: float
    """The total weight of the edges with one endpoint on each side -- the number the cut
    minimises. On an unweighted network it is a count of edges."""
    source_side: frozenset[Node]
    target_side: frozenset[Node]
    method: str
    """How the cut was found, in the report's words."""
    section: str
    """The section of the Atlas this cut implements."""

    def __len__(self) -> int:
        """How many nodes the cut covers."""
        return len(self.source_side) + len(self.target_side)


def _capacitated(graph: nx.Graph, weight: str | None) -> nx.Graph:
    """A copy of ``graph`` with an explicit ``capacity`` on every edge.

    Not cosmetic: ``networkx``'s flow algorithms read a *missing* capacity attribute as
    **infinite**, so an unweighted graph handed straight to ``minimum_cut`` raises
    ``NetworkXUnbounded`` rather than returning the cut of size 1 that is obviously there. Every
    edge gets a number here, 1.0 when the graph carries no weights, so the flow that comes back
    is the flow the network has.
    """
    copy = nx.Graph()
    copy.add_nodes_from(graph.nodes)
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        value = 1.0 if weight is None else float(data.get(weight, 1.0))
        copy.add_edge(u, v, capacity=copy.get_edge_data(u, v, {}).get("capacity", 0.0) + value)
    return copy


def mincut(graph: nx.Graph, s: Node, t: Node, weight: str | None = "weight") -> Cut:
    """The minimum ``s``-``t`` cut (§11.5): the cheapest set of edges whose removal leaves no
    path from ``s`` to ``t``, and the two sides it leaves behind.

    ``Cut.value`` is the total weight of those edges, and by the max-flow min-cut theorem it is
    also the maximum flow from ``s`` to ``t`` -- the tests assert the two are equal because that
    equality is the known answer here. Solved exactly by ``networkx``; nothing is approximated.

    **The random-walk reading, which is what puts this in chapter 11.** The cut and the effective
    resistance of §11.4 measure the same thinness from opposite sides. A cut of total weight
    ``c`` is a set of resistors in parallel carrying every route between the two nodes, so the
    conductance between them is at most ``c`` and therefore

        ``Ω[s][t] >= 1 / mincut(graph, s, t).value``

    with equality when the cut edges are the *only* thing between them, as in a barbell joined by
    one edge. A network that is expensive to cut is a network that is cheap to walk across: the
    walker has many routes, the resistance is low, and §11.6's consensus spreads quickly. That
    inequality is a known answer and the tests hold this to it.

    **What §11.5 actually warns about.** The section's own conclusion is that the minimum cut is
    usually not the structure you were looking for: *"most of the times, the best way to solve
    the 2-cut problem is to put in one group a node with degree equal to one and put all other
    nodes of the network in the other group"*. A cut is a statement about a pair you chose, not a
    discovery of groups. *"If you want to find non-trivial k-cuts of the network that are
    meaningful for humans... well... you have to do community discovery"* -- Part X, which in
    this package is :func:`graphrag.sna.cluster.louvain`.

    Undefined when ``s`` and ``t`` are the same node, and when either is not in the network;
    both raise. When the two sit in different components the cut is empty and the value is 0,
    which is the correct answer rather than an error: they are already separated.
    """
    if s == t:
        msg = "mincut needs two different nodes: there is nothing to cut between a node and itself"
        raise ValueError(msg)
    _pair_order(graph, s, t)
    _undirected_or_raise(graph, "the minimum cut", because="cut")
    flow = _capacitated(graph, weight)
    if not nx.has_path(flow, s, t):
        reachable = frozenset(nx.node_connected_component(flow, s))
        return Cut(
            value=0.0,
            source_side=reachable,
            target_side=frozenset(flow.nodes) - reachable,
            method=f"minimum {s}-{t} cut (max-flow); they are already in different components",
            section="§11.5",
        )
    value, (near, far) = nx.minimum_cut(flow, s, t, capacity="capacity")
    return Cut(
        value=float(value),
        source_side=frozenset(near),
        target_side=frozenset(far),
        method=f"minimum {s}-{t} cut, solved exactly by max-flow (networkx)",
        section="§11.5",
    )


def global_mincut(graph: nx.Graph, weight: str | None = "weight") -> Cut:
    """The global minimum cut (§11.5): the cheapest way to split the network in two, over every
    pair of sides rather than a chosen ``s`` and ``t``.

    This is the problem §11.5 poses -- *"how to divide nodes in two disjoint groups such that the
    number of edges running across groups is minimized"* -- solved exactly by Stoer-Wagner rather
    than approximated by the Fiedler vector. Which makes it the direct demonstration of the
    section's warning: on a real network the answer is almost always one pendant node on its own,
    with a cut of weight equal to its single edge. Read the answer, see the pendant, and go to
    community discovery (Part X). :func:`fiedler_cut` is the spectral approximation §11.5 spends
    its pages on, and it does *not* return this -- that disagreement is the section's lesson, not
    a bug in either.

    0 for a disconnected network, where the split costs nothing and the two sides are one
    component against the rest. Undefined for a network with fewer than two nodes, which raises.
    """
    _undirected_or_raise(graph, "the global minimum cut", because="cut")
    flow = _capacitated(graph, weight)
    if flow.number_of_nodes() < 2:
        msg = "the global minimum cut needs at least two nodes"
        raise ValueError(msg)
    components = list(nx.connected_components(flow))
    if len(components) > 1:
        first = frozenset(components[0])
        return Cut(
            value=0.0,
            source_side=first,
            target_side=frozenset(flow.nodes) - first,
            method="global minimum cut; the network is already disconnected, so it costs nothing",
            section="§11.5",
        )
    value, (near, far) = nx.stoer_wagner(flow, weight="capacity")
    return Cut(
        value=float(value),
        source_side=frozenset(near),
        target_side=frozenset(far),
        method="global minimum cut, solved exactly by Stoer-Wagner (networkx)",
        section="§11.5",
    )


def fiedler_vector(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    sparse_above: int = SPARSE_ABOVE_NODES,
) -> tuple[np.ndarray, list[Node]]:
    """The Fiedler vector (§11.5): the eigenvector of the *second* smallest eigenvalue of the
    combinatorial Laplacian, and the node order it follows.

    §11.5 says why the second and not the first: the smallest eigenvector *"isn't so special
    after all ... it is a vector of constant, telling us to which component the node belongs"*.
    The second carries the cut. Its **sign** says which side of the 2-cut a node falls on, and
    its **absolute value** says *"how embedded the node is in the group, or how far from the cut
    it is"* -- the section's node 5 sits next to the cut, its node 9 as far from it as possible.
    §11.6 adds the dynamic reading: a node far from the cut takes longer to be pulled away from
    its own side's opinion, which is *why* the sign works.

    Returned raw, signs and all. The sign of an eigenvector is arbitrary -- ``-v`` is as valid as
    ``v`` -- so which side is positive carries no meaning; only the split does.
    ``graphrag.sna.cluster.spectral_embedding`` is the same idea taken to ``k`` dimensions, which
    is how §11.5 says to solve the k-cut.

    Undefined for a graph with fewer than two nodes, and meaningless on a disconnected one, where
    the second eigenvalue is also 0 and its eigenvector separates components rather than cutting
    anything; that case raises, with §11.5's own parenthesis -- read *"the eigenvector associated
    to the smallest non-zero eigenvalue"* per component instead.

    **ATL-ENT-3 (scale).** Above ``sparse_above`` nodes this solves for only the two smallest
    eigenpairs, sparsely (``matrices.eigenpairs(..., sparse=True)``), rather than diagonalising
    the whole dense Laplacian to read off column 1: the vector this function returns is
    numerically close to the dense answer but not bit identical, since the sparse solver is
    iterative where the dense one is exact.
    """
    order = node_order(graph, nodes)
    view = graph.subgraph(order)
    if len(order) < 2:
        msg = "the Fiedler vector needs at least two nodes (§11.5)"
        raise ValueError(msg)
    if not nx.is_connected(view):
        msg = (
            "the Fiedler vector is undefined on a disconnected network: the eigenvalue 0 is "
            "repeated once per component (§8.4), so the second smallest eigenvector separates "
            "components rather than cutting one. Run it per component"
        )
        raise ValueError(msg)
    use_sparse = len(order) > sparse_above
    matrix, _ = laplacian(view, order, kind="combinatorial", weight=weight, sparse=use_sparse)
    _, vectors = eigenpairs(matrix, k=2 if use_sparse else None, largest=False, sparse=use_sparse)
    return np.ascontiguousarray(vectors[:, 1]), order


def fiedler_cut(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
) -> Cut:
    """The 2-cut §11.5 approximates with the Fiedler vector: positive entries on one side,
    negative on the other.

    *"By looking at the sign of the value of a node in its second smallest eigenvector we know in
    which group the node has to be to solve the 2-cut problem!"* -- with the emphasis on
    *approximation*: Fiedler's bound makes this a good split, not the cheapest one.
    :func:`global_mincut` finds the cheapest one exactly, and on a real network the two disagree,
    because the cheapest one is a pendant node alone (§11.5) while this one is balanced by
    construction. Both numbers belong in a report that claims either.

    ``Cut.value`` is the weight actually crossing this split, so it can be compared directly with
    ``global_mincut(graph).value``: it is never smaller. A node whose entry is exactly 0 is put
    on the positive side, which is a tie-break and not a finding.
    """
    vector, order = fiedler_vector(graph, nodes, weight=weight)
    positive = frozenset(node for node, value in zip(order, vector, strict=True) if value >= 0.0)
    negative = frozenset(order) - positive
    view = graph.subgraph(order)
    crossing = sum(
        1.0 if weight is None else float(data.get(weight, 1.0))
        for u, v, data in view.edges(data=True)
        if (u in positive) != (v in positive)
    )
    return Cut(
        value=crossing,
        source_side=positive,
        target_side=negative,
        method="spectral 2-cut by the sign of the Fiedler vector; an approximation, not the "
        "minimum (compare global_mincut)",
        section="§11.5",
    )


# ------------------------------------------------------------------------- §11.6 consensus


@dataclass(frozen=True, eq=False)
class Consensus:
    """The end state of a consensus dynamics (§11.6), and whether it is one opinion or several."""

    nodes: list[Node]
    values: np.ndarray
    """The opinion of each node, in ``nodes`` order, when the iteration stopped."""
    dynamics: str
    """``random-walk`` (discrete, driven by ``P``) or ``laplacian`` (continuous, driven by
    ``L``). They converge to different means; see :func:`consensus`."""
    steps: int
    converged: bool
    """Whether every connected component ended internally agreed, to within ``tolerance``, which
    is what §11.6 says a consensus dynamics does. ``False`` means the opinions were still apart
    inside at least one component when the step budget ran out, which for the discrete dynamics
    usually means they are oscillating rather than crawling -- see ``note``."""
    agreed: bool
    """Whether *every* node ended on the same value: a single consensus for the whole network,
    rather than one per component. Equal to ``converged`` on a connected network."""
    limit: float | None
    """The value a connected network must converge to, computed from the *initial* opinions and
    the conserved quantity of the dynamics. ``None`` when the network is disconnected, because
    §11.6 says each component then reaches its own and there is no single answer."""
    note: str
    """Why the run did not end in one agreed value, in the book's terms. Empty when it did."""

    def by_node(self) -> dict[Node, float]:
        """The final opinions keyed by node id."""
        return {node: float(value) for node, value in zip(self.nodes, self.values, strict=True)}


def consensus(
    graph: nx.Graph,
    values: Mapping[Node, float] | Sequence[float] | np.ndarray,
    dynamics: Dynamics = "random-walk",
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    steps: int | None = None,
    tolerance: float = 1e-9,
    max_steps: int = 10_000,
    lazy: bool = False,
    rate: float | None = None,
) -> Consensus:
    """Run §11.6's consensus dynamics: every node repeatedly averages its neighbours' opinions
    until nobody's opinion is moving.

    ``values`` is the starting opinion of each node -- a mapping keyed by node id, or a sequence
    aligned with the node order. §11.6's example draws them uniformly from ``[0, 1]``, but
    nothing here requires that range.

    **§11.6 describes two dynamics, and they converge to different numbers.** The section uses
    the stochastic adjacency and then says *"what I'm saying also holds for the Laplacian"*, with
    the difference being that ``P`` *"describes the discrete diffusion over a network"* -- a
    clock ticking, nothing happening between ticks -- while ``L`` describes *"continuous
    diffusion: time flows without ticks"*. Both are implemented, and which one is asked for
    changes the answer:

    ``dynamics="random-walk"``
        ``x <- P x``: each node replaces its opinion with the weighted average of its
        neighbours'. This is DeGroot's original model (§11.6 cites him for it). The section
        writes the update with its indices the other way round, ``x'_v = sum_u x_u A_uv``, and
        describes it in prose as averaging the neighbours' opinions "because the rows sum to
        1"; the code follows the prose and DeGroot, since the transposed product converges to
        the stationary distribution times the opinion sum rather than to a consensus. The
        quantity it conserves is ``π · x``, so it converges to the **degree-weighted mean**
        ``Σ_v k_v x_v / 2|E|`` -- a hub's starting opinion counts for more, in exact proportion
        to its degree, because the walk visits it more (§11.1).
    ``dynamics="laplacian"``
        ``x <- x - rate · L x``, the explicit discretisation of ``dx/dt = -L x``. The quantity it
        conserves is the plain sum, so it converges to the **unweighted mean**
        ``Σ_v x_v / |V|``, where every node's starting opinion counts the same.

    §11.6 says the limit is *"the average of the initial x, in this case 0.5 since values were
    extracted uniformly at random between 0 and 1"*. On a regular graph the two means coincide
    and the sentence is exactly right; off one they differ, and with random starting values both
    land near 0.5 anyway, which is why the difference does not show up in the figure. Ask for the
    dynamics whose mean is the one you meant, and say which in the report.

    ``rate`` is the step size of the continuous dynamics; left alone it is ``1 / 2k_max``, which
    Gershgorin's bound makes stable on every graph. A larger one converges faster until it
    overshoots and diverges.

    **When it does not converge, and what the result says instead.**

    * *Disconnected network.* Each component settles on its own value and no exchange crosses
      between them: §11.6's *"if a network has separate connected components, it cannot reach a
      unique consensus"*. ``converged`` is ``True``, ``agreed`` is ``False``, ``limit`` is
      ``None``, and ``note`` says so.
    * *Bipartite network, discrete dynamics.* ``P`` has an eigenvalue of ``-1`` there, so the
      opinions flip between the two sides forever instead of settling -- the periodicity §11.1
      warns about, in its dynamic form. ``converged`` is ``False`` and ``note`` names it. Pass
      ``lazy=True`` to iterate ``(I + P)/2`` instead, which has the same limit and cannot
      oscillate, or use the Laplacian dynamics, which cannot either.
    * *Isolated node.* It has no neighbour to average with, so it keeps its opinion forever. Its
      row of ``P`` is made a self-loop rather than left at zero, because a zero row would
      silently drag its opinion to 0 (§8.2 leaves such a row sub-stochastic).

    §11.6's other claim -- that the *second* eigenvector says how fast this happens, and that
    nodes bundle with their community before the network agrees -- is :func:`consensus_time`.
    """
    if dynamics not in DYNAMICS:
        msg = f"dynamics must be one of {', '.join(DYNAMICS)}, got {dynamics!r}"
        raise ValueError(msg)
    _undirected_or_raise(graph, "consensus", because="consensus")
    order = node_order(graph, nodes)
    start = _opinions(values, order)
    operator, limit = _consensus_operator(graph, order, dynamics, weight, lazy, rate, start)
    within = _component_spread(graph, order)
    current = start.copy()
    taken = 0
    settled = bool(order) and within(current) < tolerance
    budget = 0 if not order else (max_steps if steps is None else steps)
    for taken in range(1, budget + 1):  # noqa: B007 -- taken is read after the loop
        current = operator @ current
        settled = bool(within(current) < tolerance)
        if settled and steps is None:
            break
    agreed = bool(current.size > 0 and float(np.max(current) - np.min(current)) < tolerance)
    return Consensus(
        nodes=order,
        values=current,
        dynamics=dynamics,
        steps=taken,
        converged=settled,
        agreed=agreed,
        limit=limit,
        note=_consensus_note(graph, order, dynamics, lazy, settled, agreed),
    )


def _component_spread(graph: nx.Graph, order: Sequence[Node]) -> Callable[[np.ndarray], float]:
    """A function giving the widest disagreement **inside** any one connected component.

    That, and not the change from one step to the next, is what :func:`consensus` iterates until:
    §11.6's statement is that each component reaches its own consensus, so "has it happened yet"
    is a question about spread within a component. A step-size criterion stops too early -- the
    step is a fraction of the remaining gap, so the gap is still several tolerances wide when the
    step is one -- and a whole-network spread would never be satisfied by a disconnected graph
    that has in fact finished.
    """
    if not order:
        return lambda _: 0.0
    view = graph.subgraph(order)
    position = {node: index for index, node in enumerate(order)}
    blocks = [sorted(position[node] for node in group) for group in nx.connected_components(view)]
    permutation = np.array([index for block in blocks for index in block], dtype=np.int64)
    starts = np.cumsum([0] + [len(block) for block in blocks[:-1]], dtype=np.int64)

    def spread(vector: np.ndarray) -> float:
        ordered = vector[permutation]
        high = np.maximum.reduceat(ordered, starts)
        low = np.minimum.reduceat(ordered, starts)
        return float(np.max(high - low))

    return spread


def _opinions(
    values: Mapping[Node, float] | Sequence[float] | np.ndarray, order: Sequence[Node]
) -> np.ndarray:
    """``values`` as an array in ``order``, however the caller expressed them."""
    if isinstance(values, Mapping):
        missing = [str(node) for node in order if node not in values]
        if missing:
            msg = f"no starting opinion for: {', '.join(missing[:5])}"
            raise ValueError(msg)
        return np.array([float(values[node]) for node in order], dtype=np.float64)
    given = np.asarray(values, dtype=np.float64).ravel()
    if given.shape[0] != len(order):
        msg = f"consensus needs one opinion per node: got {given.shape[0]} for {len(order)} nodes"
        raise ValueError(msg)
    return given


def _consensus_operator(
    graph: nx.Graph,
    order: Sequence[Node],
    dynamics: str,
    weight: str | None,
    lazy: bool,
    rate: float | None,
    start: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    """The one-step update matrix and the value a connected network converges to under it."""
    connected = nx.is_connected(graph.subgraph(order)) if order else False
    strengths = _strengths(graph, order, weight)
    if dynamics == "random-walk":
        matrix, _ = stochastic(graph, order, orientation="row", weight=weight)
        operator = dense(matrix).copy()
        for position, total in enumerate(operator.sum(axis=1)):
            if total <= 0.0:
                operator[position, position] = 1.0
        if lazy:
            operator = 0.5 * (operator + np.eye(len(order)))
        total_strength = float(strengths.sum())
        limit = float(strengths @ start / total_strength) if total_strength > 0 else None
    else:
        matrix, _ = laplacian(graph, order, kind="combinatorial", weight=weight)
        largest = float(strengths.max()) if strengths.size else 0.0
        step = rate if rate is not None else (1.0 / (2.0 * largest) if largest > 0 else 0.0)
        operator = np.eye(len(order)) - step * dense(matrix)
        limit = float(start.mean()) if start.size else None
    return operator, limit if connected else None


def _consensus_note(
    graph: nx.Graph,
    order: Sequence[Node],
    dynamics: str,
    lazy: bool,
    settled: bool,
    agreed: bool,
) -> str:
    """Why this run did not end in one agreed opinion, in §11.6's terms. Empty when it did."""
    if agreed and settled:
        return ""
    view = graph.subgraph(order)
    components = nx.number_connected_components(view) if order else 0
    if not settled and dynamics == "random-walk" and not lazy and nx.is_bipartite(view):
        return (
            "did not settle: this network is bipartite, so P has an eigenvalue of -1 and the "
            "opinions flip between the two sides forever rather than converging -- the "
            "periodicity of §11.1 in its dynamic form. Pass lazy=True, which iterates (I + P)/2 "
            "to the same limit, or use dynamics='laplacian'"
        )
    if not settled:
        return (
            "did not settle within the step budget: the opinions were still moving when the "
            "iteration stopped, so nothing here is a limit"
        )
    if components > 1:
        return (
            f"settled, but on {components} different values: §11.6 says a network with separate "
            "components 'cannot reach a unique consensus; every connected component will reach "
            "its own consensus independently because there's no exchange of information across "
            "components'. Read one component at a time"
        )
    return "settled without full agreement, which on a connected network means tolerance is tight"


def consensus_time(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    sparse_above: int = SPARSE_ABOVE_NODES,
) -> float:
    """``1 / λ2`` (§11.6): roughly how many steps the network takes to settle, from the second
    smallest eigenvalue of the combinatorial Laplacian.

    §11.6 computes exactly this for its example network -- *"this is described by the inverse of
    the second eigenvalue of the Laplacian. For that network, it is 1/λ2 ~ 5.13. This tells you
    that the nodes are expected to converge to their community's opinion between step 5 and 6"*.
    λ2 is the algebraic connectivity: small when the network has a thin cut, so a network that is
    easy to cut (§11.5) is slow to agree, and the two sides of the cut hold their own opinions
    for a while before the network as a whole does. That is the mechanism §11.6 offers for why
    the Fiedler vector solves the mincut at all.

    An order-of-magnitude reading, not a step count: it is the slowest mode's time constant, so a
    report should say "of this order" rather than quote it to two decimals. ``inf`` for a
    disconnected network, where λ2 is 0 and the components never agree at all.

    **ATL-ENT-3 (scale).** Above ``sparse_above`` nodes only the two smallest eigenvalues are
    solved for, sparsely, the same routing :func:`fiedler_vector` uses and for the same reason --
    λ2 alone is wanted, not the rest of the spectrum a dense diagonalisation would also compute.
    """
    order = node_order(graph, nodes)
    view = graph.subgraph(order)
    if len(order) < 2:
        msg = "the consensus time needs at least two nodes (§11.6)"
        raise ValueError(msg)
    _undirected_or_raise(graph, "the consensus time")
    if not nx.is_connected(view):
        return math.inf
    use_sparse = len(order) > sparse_above
    matrix, _ = laplacian(view, order, kind="combinatorial", weight=weight, sparse=use_sparse)
    values, _ = eigenpairs(matrix, k=2 if use_sparse else None, largest=False, sparse=use_sparse)
    second = float(values[1])
    return math.inf if second <= 0.0 else 1.0 / second


# --------------------------------------------------------------- `sna walks` report (ATL-F1)


@dataclass(frozen=True)
class WalksReport:
    """Everything one ``sna walks``/``sna_walks`` call reports about one pair of nodes: the six
    numbers of §11.3-11.5, next to the query that produced them.

    Built once by :func:`walks_report`, so the CLI's `rich` table, its `--out`/`--json` and the
    `sna_walks` MCP tool's `markdown`/`payload` all read the same values rather than each calling
    `stationary_distribution`/`hitting_time`/`commute_time`/`effective_resistance`/`mincut` a
    second time.
    """

    persona_id: str
    network: str
    source_node: Node
    target_node: Node
    unweighted: bool
    frame: str
    n: int
    m: int
    volume: float
    flattened_note: str
    source_probability: float
    target_probability: float
    hitting_out: float
    hitting_back: float
    commute_time: float
    effective_resistance: float
    cut: Cut
    connected: bool


def walks_report(
    graph: nx.Graph,
    source_node: Node,
    target_node: Node,
    *,
    persona_id: str,
    network: str,
    flattened_note: str = "",
    unweighted: bool = False,
    max_nodes: int = MAX_NODES,
) -> WalksReport:
    """§11.3 (times), §11.4 (resistance) and §11.5 (cut) for one pair, in one pass (ATL-F1).

    ``graph`` must already be undirected (:func:`graphrag.sna.measures.undirected_view`'s first
    return) and contain both nodes -- the caller checks membership and that the two nodes differ
    before calling this, the same way both `sna walks` and `sna_walks` always did, so that a
    caller-specific refusal (`typer.Exit(2)` vs. a structured error) stays in the caller rather
    than being decided here. ``flattened_note`` is whatever :func:`undirected_view` returned for
    ``graph``, carried through unchanged for :func:`render_walks` to print.

    Raises ``ValueError`` exactly where `hitting_time`/`commute_time`/`effective_resistance` do:
    a component above ``max_nodes`` (§11.4's pseudoinverse is a dense ``O(n^3)`` factorisation).
    """
    weight = None if unweighted else "weight"
    distribution, order = stationary_distribution(graph, weight=weight)
    index = {node: position for position, node in enumerate(order)}
    forward = hitting_time(graph, source_node, target_node, weight=weight, max_nodes=max_nodes)
    backward = hitting_time(graph, target_node, source_node, weight=weight, max_nodes=max_nodes)
    commute = commute_time(graph, source_node, target_node, weight=weight, max_nodes=max_nodes)
    resistance = effective_resistance(
        graph, source_node, target_node, weight=weight, max_nodes=max_nodes
    )
    cut = mincut(graph, source_node, target_node, weight=weight)
    return WalksReport(
        persona_id=persona_id,
        network=network,
        source_node=source_node,
        target_node=target_node,
        unweighted=unweighted,
        frame=str(graph.graph.get("frame", "")),
        n=graph.number_of_nodes(),
        m=graph.number_of_edges(),
        volume=volume(graph, weight=weight),
        flattened_note=flattened_note,
        source_probability=float(distribution[index[source_node]]),
        target_probability=float(distribution[index[target_node]]),
        hitting_out=forward,
        hitting_back=backward,
        commute_time=commute,
        effective_resistance=resistance,
        cut=cut,
        connected=cut.value > 0 and math.isfinite(resistance),
    )


def volume_label(report: WalksReport) -> str:
    """``report.volume`` named for what it is on this run: ``2|E|`` counts edges only when the
    walk ignored weights; otherwise it is twice the total edge *weight* (see :func:`volume`), and
    printing it as ``2|E|`` beside ``m`` edges reads as an arithmetic error."""
    if report.unweighted:
        return f"2|E|={report.volume:,.0f}"
    return f"2W={report.volume:,.0f} (twice the total edge weight)"


def render_walks(report: WalksReport) -> list[str]:
    """``report`` as the markdown lines `sna walks --out`/`sna_walks`'s ``markdown`` both write --
    the same numbers the CLI's console table renders (ATL-F1: one function both surfaces call)."""
    lines = [
        f"{report.persona_id} / {report.network}: {report.source_node} -> {report.target_node}",
        report.frame,
        (
            f"Atlas ch. 11 (random walks), §11.3 times, §11.4 resistance, §11.5 cut. "
            f"n={report.n:,} nodes, m={report.m:,} edges, "
            f"{volume_label(report)}"
            + (f". {report.flattened_note}" if report.flattened_note else "")
        ),
        "Null model: none. These are exact quantities of the network as built, not tests "
        "against a null, so there is no p-value to withhold them behind.",
        f"stationary probability: {report.source_probability:.5f} / "
        f"{report.target_probability:.5f}",
        f"hitting time out: {report.hitting_out:,.2f}",
        f"hitting time back: {report.hitting_back:,.2f}",
        f"commute time: {report.commute_time:,.2f}",
        f"effective resistance: {report.effective_resistance:,.4f}",
        f"minimum cut (= max flow): {report.cut.value:,.2f} "
        f"({len(report.cut.source_side):,} nodes on one side, "
        f"{len(report.cut.target_side):,} on the other)",
    ]
    if report.connected:
        lines.append(
            f"§11.4-11.5 check: resistance {report.effective_resistance:.4f} >= 1 / cut "
            f"{1 / report.cut.value:.4f} — a network that is expensive to cut is cheap to walk "
            "across."
        )
    else:
        lines.append(
            "These two are in different components: the walker never arrives, so the times "
            "are infinite and §11.1 reads the two sides as two networks rather than one."
        )
    return lines


def walks_payload(report: WalksReport) -> dict[str, Any]:
    """``report`` as the JSON `sna walks` writes with ``--json`` and the ``sna_walks`` MCP tool
    returns as ``payload`` (ATL-F1: one function both surfaces call)."""
    return {
        "persona_id": report.persona_id,
        "network": report.network,
        "source": report.source_node,
        "target": report.target_node,
        "unweighted": report.unweighted,
        "n": report.n,
        "m": report.m,
        "stationary_probability": {
            "source": report.source_probability,
            "target": report.target_probability,
        },
        "hitting_time": {"out": report.hitting_out, "back": report.hitting_back},
        "commute_time": report.commute_time,
        "effective_resistance": report.effective_resistance,
        "min_cut": {
            "value": report.cut.value,
            "source_side": sorted(report.cut.source_side),
            "target_side": sorted(report.cut.target_side),
        },
        "connected": report.connected,
    }
