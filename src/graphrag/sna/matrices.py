"""The matrix representations of a network (ch. 8) and the linear algebra that reads them (§5.5,
§5.6). Returns arrays; builds no graphs and prints nothing.

**The node-order contract.** A matrix is only meaningful next to the list of nodes its rows and
columns follow, so every builder here returns ``(matrix, nodes)`` -- or ``(matrix, left, right)``
for the two-mode case -- and every builder takes an optional ``nodes`` sequence to align with.
Left to itself it sorts the node ids. That default is not cosmetic: ``networkx`` lays its own
matrices out in *insertion* order, so the same data arriving in a different order gives a
different matrix, and two consumers that each called ``nx.to_numpy_array`` separately could not
safely compare their results row by row. Sorting makes the layout a function of the data alone.

The one case that must not be sorted is a consumer that already published an alignment: both
``measures._eigenvector`` and ``cluster.spectral_embedding`` hand back one row per node in
``list(graph.nodes)`` order, and their callers zip against that list, so they pass that order in
explicitly rather than take the default.

**Which matrix answers what.** ``adjacency`` is the network itself (§8.1). ``stochastic`` is the
same network read as transition probabilities, and the row-oriented one is the random walk's
transition matrix (§8.2). ``incidence`` is the two-mode network, whose products are the bipartite
projections (§8.3, §26.1). ``laplacian`` is the matrix whose spectrum counts components and cuts
the graph (§8.4). ``eigenpairs``, ``svd`` and ``nmf`` are the solvers the rest of the package
reaches for (§5.5, §5.6).

Sparse forms come from ``scipy.sparse``, which ``scikit-learn`` already requires; the dense form
stays the default because every solver here is dense and the networks this package builds have
thousands of nodes, not millions.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.decomposition import NMF

type Matrix = np.ndarray | sp.csr_array
"""A dense ``numpy`` array or a ``scipy.sparse`` CSR array, depending on ``sparse=``."""

Node = Any
"""A node id. This package's own networks use ``str``; the book's legendary graphs use ``int``.
Nothing here needs more than ids that hash and sort."""

Orientation = Literal["row", "column"]
LaplacianKind = Literal["combinatorial", "symmetric", "random-walk"]
Side = Literal["left", "right"]

LAPLACIANS: tuple[str, ...] = ("combinatorial", "symmetric", "random-walk")

#: ATL-ENT-3 (scale). Above this many nodes, a caller that only needs a handful of eigenpairs --
#: :func:`graphrag.sna.walks.fiedler_vector`'s second-smallest, :func:`graphrag.sna.walks.
#: consensus_time`'s -- asks :func:`eigenpairs` for ``sparse=True`` instead of densifying the
#: matrix first and diagonalising the whole thing. Below it the dense solver is fast enough
#: (well under a second on this package's own networks) that the difference does not matter, and
#: it stays the default everywhere: it is exact and seed-free, where the sparse path is an
#: iterative approximation (see ``eigenpairs``). Chosen to sit below the entity network of the
#: largest persona this repository ships (13k chunks, several thousand entities), which is the
#: network the ATL-ENT-3 benchmark exercises.
SPARSE_ABOVE_NODES: int = 2_000


# ----------------------------------------------------------------------------- the contract


def node_order(graph: nx.Graph, nodes: Sequence[Node] | None = None) -> list[Node]:
    """The row/column order for ``graph``: ``nodes`` if given, otherwise the sorted node ids.

    A given order may be a subset of the graph -- the matrix is then the one of the induced
    subgraph -- but it may not name a node the graph does not have, and may not repeat one,
    because either would make row ``i`` mean something other than node ``i``.
    """
    if nodes is None:
        return sorted(graph.nodes)
    return _checked(nodes, set(graph.nodes), "the graph")


def _checked(nodes: Sequence[Node], known: set[Node], where: str) -> list[Node]:
    """``nodes`` as a list, refused if it repeats an id or names one ``known`` does not hold.

    Either mistake makes row ``i`` stand for something other than node ``i``: a repeat gives two
    rows the same name, and a stranger gives a row that no data can ever fill. Both are quiet
    failures -- a duplicated or empty row looks like an isolated node -- so they are raised on
    here, once, for every axis of every matrix in this module.
    """
    order = list(nodes)
    if len(set(order)) != len(order):
        msg = f"nodes must not repeat: each row or column stands for exactly one node ({where})"
        raise ValueError(msg)
    missing = [node for node in order if node not in known]
    if missing:
        msg = f"nodes not in {where}: {', '.join(str(node) for node in missing[:5])}"
        raise ValueError(msg)
    return order


def dense(matrix: Matrix) -> np.ndarray:
    """``matrix`` as a dense array, whichever form it arrived in."""
    if isinstance(matrix, np.ndarray):
        return matrix
    out: np.ndarray = np.asarray(matrix.toarray(), dtype=np.float64)
    return out


def _sums(matrix: Matrix, axis: int) -> np.ndarray:
    """Row (``axis=1``) or column (``axis=0``) sums, without densifying a sparse matrix."""
    return np.asarray(matrix.sum(axis=axis), dtype=np.float64).ravel()


def _inverse(totals: np.ndarray, root: bool = False) -> np.ndarray:
    """``1/x`` (or ``1/sqrt(x)``) per entry, with zero where ``x`` is zero.

    A node with no edges has no inverse degree. Taking it as zero is the usual convention and
    the one that keeps the normalised Laplacian's zero-eigenvalue count equal to the number of
    components: the isolated node's row stays empty instead of gaining a spurious 1.
    """
    base = np.sqrt(totals) if root else totals
    return np.divide(1.0, base, out=np.zeros_like(base), where=totals > 0)


def scale_to_unit_sums(matrix: Matrix, axis: int) -> Matrix:
    """Divide every row (``axis=1``) or column (``axis=0``) by its own sum, dense or sparse alike.

    One half-step of Sinkhorn-Knopp doubly stochastic normalisation (§27.2): repeat this for
    both axes in turn and every row and column sum approaches 1. ``backbone.doubly_stochastic_
    scores`` is the caller, and calls this on a dense array below :data:`SPARSE_ABOVE_NODES`
    nodes and on a sparse one above it (ATL-F2) -- diagonal scaling adds no fill-in, so the
    sparse path costs one sparse matrix-vector-diagonal product per half-step rather than a
    dense ``n x n`` one.

    A row or column that already sums to zero is left at zero rather than divided by it: an
    isolated node has no edges to normalise, which the caller reads as the matrix staying
    sub-stochastic in that row or column rather than as a division failure.
    """
    totals = _sums(matrix, axis)
    inverse = np.divide(1.0, totals, out=np.zeros_like(totals), where=totals > 0)
    if isinstance(matrix, np.ndarray):
        return matrix * (inverse[:, None] if axis == 1 else inverse[None, :])
    diagonal = sp.diags(inverse)
    scaled: Matrix = (diagonal @ matrix) if axis == 1 else (matrix @ diagonal)
    return scaled


# ----------------------------------------------------------------------------- §8.1 adjacency


def adjacency(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    weight: str | None = "weight",
    sparse: bool = False,
) -> tuple[Matrix, list[Node]]:
    """The adjacency matrix ``A`` (§8.1): ``A[u][v]`` is the weight of the edge ``u-v``, 0 if none.

    For an undirected graph this is square and symmetric, with a zero diagonal as long as the
    graph has no self-loops -- §8.1 keeps the diagonal free deliberately, because the Laplacian
    puts the degree there. Row ``u`` sums to ``u``'s weighted degree.

    ``weight`` names the edge attribute to read; ``weight=None`` gives the book's binary matrix,
    where every edge counts 1. Our edge weights are affinities (a weight of 6 means "shared six
    documents"), so a heavier entry means a stronger tie, never a longer distance.

    A directed graph is laid out the way ``networkx`` lays it out -- row ``u`` holds ``u``'s
    out-edges, so ``A[u][v]`` is the ``u -> v`` edge, which §8.1 warns is one of two conventions
    in the literature -- and the matrix is then asymmetric, as it should be. Not built here:
    §8.1's **multilayer** representations, the three-dimensional tensor and the supra-adjacency
    matrix, which belong with the multilayer network itself: see
    :func:`graphrag.sna.layers.supra_adjacency`.
    """
    order = node_order(graph, nodes)
    if sparse:
        matrix: Matrix = sp.csr_array(
            nx.to_scipy_sparse_array(graph, nodelist=order, weight=weight, dtype=np.float64)
        )
        return matrix, order
    return nx.to_numpy_array(graph, nodelist=order, weight=weight, dtype=np.float64), order


# ----------------------------------------------------------------------------- §8.2 stochastic


def stochastic(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    orientation: Orientation = "row",
    weight: str | None = "weight",
    sparse: bool = False,
) -> tuple[Matrix, list[Node]]:
    """The stochastic adjacency matrix (§8.2): the adjacency divided by its row (or column) sums.

    **The row-oriented matrix is the random walk's transition matrix.** Entry ``(u, v)`` is the
    probability that a walker standing on ``u`` -- the *row* -- crosses to ``v`` in one step, so
    each row sums to 1. The book's Pⁿ -- ``numpy.linalg.matrix_power(P, n)``, the repeated matrix
    product and never the elementwise ``P ** n`` -- then gives the transition probabilities of a
    walk of length ``n`` (§8.2).

    The column-oriented matrix is the transpose convention: each column sums to 1, and it is what
    a distribution written as a column vector is multiplied by. Pick one and say which: a
    stochastic matrix is **not** symmetric even when the graph is undirected, because ``u`` and
    ``v`` normalise by different degrees, so transposing it changes what the numbers mean.

    Undefined for a node with no edges: its row (or column) sums to 0 and cannot be scaled to 1,
    because a walker on an isolated node has nowhere to go. Those rows are left at zero rather
    than filled in, which makes the matrix sub-stochastic and is the honest statement of it.
    """
    if orientation not in ("row", "column"):
        msg = f"orientation must be 'row' or 'column', got {orientation!r}"
        raise ValueError(msg)
    matrix, order = adjacency(graph, nodes, weight=weight, sparse=sparse)
    axis = 1 if orientation == "row" else 0
    scale = _inverse(_sums(matrix, axis))
    if sparse:
        diagonal = sp.diags_array(scale, format="csr")
        scaled: Matrix = diagonal @ matrix if orientation == "row" else matrix @ diagonal
        return sp.csr_array(scaled), order
    dense_matrix = dense(matrix)
    if orientation == "row":
        return dense_matrix * scale[:, np.newaxis], order
    return dense_matrix * scale[np.newaxis, :], order


# ----------------------------------------------------------------------------- §8.3 incidence


def incidence(
    pairs: Iterable[tuple[Node, Node]],
    left: Sequence[Node] | None = None,
    right: Sequence[Node] | None = None,
    sparse: bool = False,
) -> tuple[Matrix, list[Node], list[Node]]:
    """The incidence matrix ``B`` of a two-mode edge list (§8.3): left nodes on the rows, right
    nodes on the columns, 1 where the two are connected.

    ``pairs`` are ``(left, right)`` memberships -- (speaker, document), (entity, passage) -- the
    same input ``export.bipartite_projection`` takes, so the two can be compared entry by entry.
    §8.3 notes that the adjacency matrix of a bipartite network *is* an incidence matrix; this is
    that one. (For the node-by-edge matrix that owns the term in the literature, see
    ``edge_incidence``.)

    The matrix is **binary**: a pair repeated in ``pairs`` still sets its cell to 1, so row sums
    count distinct memberships. That is what makes ``project`` agree with the shared-neighbour
    count of §26.1, which is a count of distinct common neighbours. Edge weights on the two-mode
    network are therefore dropped; a weighted projection is a different weighting scheme.

    ``left`` and ``right`` align the axes with an order a caller already published; left to
    themselves they are the sorted ids seen in ``pairs``. Either may be a subset -- the matrix is
    then the one of those memberships alone -- but, exactly as for ``node_order``, neither may
    repeat an id or name one that appears in no pair, because a duplicated or empty row is
    indistinguishable from a node that happens to have no memberships.

    Not built here: §8.2's *stochastic* reading of a two-mode network, where ``B`` is row- or
    column-normalised first so that ``B @ B.T`` holds probabilities of walking from one left node
    to another rather than counts. That is a choice of projection weighting, and which weighting
    a projection should carry is ATL-26's question, not this module's.
    """
    memberships = {(a, b) for a, b in pairs}
    seen_left = {a for a, _ in memberships}
    seen_right = {b for _, b in memberships}
    rows = sorted(seen_left) if left is None else _checked(left, seen_left, "the left of pairs")
    columns = (
        sorted(seen_right) if right is None else _checked(right, seen_right, "the right of pairs")
    )
    row_index = {node: index for index, node in enumerate(rows)}
    column_index = {node: index for index, node in enumerate(columns)}
    coordinates = [
        (row_index[a], column_index[b])
        for a, b in memberships
        if a in row_index and b in column_index
    ]
    if sparse:
        data = np.ones(len(coordinates), dtype=np.float64)
        ii = np.array([r for r, _ in coordinates], dtype=np.int64)
        jj = np.array([c for _, c in coordinates], dtype=np.int64)
        built = sp.coo_array((data, (ii, jj)), shape=(len(rows), len(columns))).tocsr()
        return sp.csr_array(built), rows, columns
    matrix = np.zeros((len(rows), len(columns)), dtype=np.float64)
    for row, column in coordinates:
        matrix[row, column] = 1.0
    return matrix, rows, columns


def edge_incidence(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    oriented: bool = False,
    sparse: bool = False,
) -> tuple[Matrix, list[Node], list[tuple[Node, Node]]]:
    """The ``|V| x |E|`` incidence matrix of §8.3: nodes on the rows, edges on the columns.

    This is the object a paper means when it says "the incidence matrix". Every column has two
    non-zero entries, one per endpoint, so an unoriented column sums to 2 -- only a hypergraph's
    column can sum to more.

    With ``oriented=True`` one endpoint of each edge is ``+1`` and the other ``-1``, so every
    column sums to zero. §8.3 says which endpoint gets which sign does not matter as long as the
    columns sum to zero; here the earlier node in the row order is ``+1``, which makes the matrix
    reproducible. The oriented matrix reconstructs the combinatorial Laplacian as ``B @ B.T``,
    and ``B.T @ B - 2I`` is the adjacency of the line graph.

    Unweighted by construction: an entry says an edge touches a node, not how strongly. A
    self-loop is dropped rather than represented, because it has one endpoint where the matrix
    assumes two, and its column would break both the sum-to-2 and the sum-to-0 property.
    """
    order = node_order(graph, nodes)
    index = {node: position for position, node in enumerate(order)}
    edges = [
        (u, v) if index[u] < index[v] else (v, u)
        for u, v in graph.edges
        if u in index and v in index and u != v
    ]
    matrix = np.zeros((len(order), len(edges)), dtype=np.float64)
    for column, (u, v) in enumerate(edges):
        matrix[index[u], column] = 1.0
        matrix[index[v], column] = -1.0 if oriented else 1.0
    if sparse:
        return sp.csr_array(matrix), order, edges
    return matrix, order, edges


def project(matrix: Matrix, side: Side = "left") -> Matrix:
    """The simple-weight bipartite projection of an incidence matrix (§8.3, §26.1).

    ``B @ B.T`` for ``side="left"`` and ``B.T @ B`` for ``side="right"``. With a binary ``B``,
    cell ``(u, v)`` of the product is the number of opposite-side nodes ``u`` and ``v`` share,
    which is exactly the simple weight ``w(u,v) = |N(u) ∩ N(v)|`` of §26.1. The diagonal of the
    product is each node's own degree rather than a self-tie, so §26.1 sets it to zero, and so
    does this.

    A weight here is a count of shared neighbours, not evidence of a tie: §26.1 warns that one
    high-degree node on the opposite side connects everything it touches to everything else it
    touches. Thresholding or backboning the result is a separate decision.

    This is the *simple weight* and only that. §8.2's probabilistic reading -- normalise ``B`` by
    its row sums and its transpose by its, so the product holds the probability of walking from
    one left node to another via a right node -- and every other scheme in chapter 26 are
    weighting choices that belong to the projection ticket (ATL-26), not to the matrix algebra.
    """
    if side not in ("left", "right"):
        msg = f"side must be 'left' or 'right', got {side!r}"
        raise ValueError(msg)
    if isinstance(matrix, np.ndarray):
        product = matrix @ matrix.T if side == "left" else matrix.T @ matrix
        np.fill_diagonal(product, 0.0)
        return product
    sparse_product = matrix @ matrix.T if side == "left" else matrix.T @ matrix
    without_diagonal: Matrix = sparse_product - sp.diags_array(sparse_product.diagonal())
    return sp.csr_array(without_diagonal)


# ----------------------------------------------------------------------------- §8.4 Laplacian


def laplacian(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    kind: LaplacianKind = "combinatorial",
    weight: str | None = "weight",
    sparse: bool = False,
) -> tuple[Matrix, list[Node]]:
    """A graph Laplacian (§8.4). ``kind`` picks which one, and they are not interchangeable.

    ``combinatorial``
        ``L = D - A``: the degree on the diagonal, ``-w`` for every edge, zero elsewhere, so
        every row and column sums to zero (§8.4). Positive semi-definite for an undirected
        graph, so no eigenvalue is negative, the smallest is always ``0``, and **the multiplicity
        of that zero is the number of connected components** -- the property §8.4 points at
        §10.4 for. On a weighted network ``D`` holds the weighted degree (the strength), which is
        the generalisation that keeps the row sums at zero; ``weight=None`` gives the book's
        unweighted ``D``.
    ``symmetric``
        ``D^-1/2 L D^-1/2``: the same matrix with the degrees divided out, so a hub and a
        pendant are read on the same scale. Still symmetric, so its spectrum is real, and every
        eigenvalue lies in ``[0, 2]``; the upper end is reached only by a bipartite component.
        This is the normalised cut's matrix, and what ``cluster.spectral_embedding`` clusters.
    ``random-walk``
        ``D^-1 L = I - P``, with ``P`` the row-stochastic matrix of §8.2. **Not symmetric**, for
        the same reason ``P`` is not; its eigenvalues are those of the symmetric form, but its
        eigenvectors are not, so do not mix the two.

    Both normalised forms need ``D^-1``, which an isolated node does not have. Its inverse degree
    is taken as zero, which leaves that node's whole row and column at zero: it then contributes
    its own zero eigenvalue and so still counts as the component it is.

    **A directed graph is refused rather than answered.** §8.4 says a directed network has two
    Laplacians, because a node has two degrees: put the indegree on the diagonal and you get one
    matrix, the outdegree and you get another, and neither is more correct than the other. Left
    to itself this function would silently build the outdegree one -- ``nx.to_numpy_array`` fills
    row ``u`` with ``u``'s out-edges -- which is asymmetric, has non-zero column sums, and whose
    eigenvalues §8.4 warns are complex as soon as a single directed edge exists. Choosing between
    the two belongs to the ticket that first needs a directed spectrum (ATL-37, directed
    communities), so this raises and says so.

    §8.4's other special case is deliberately absent too: a **signed** graph needs ``D`` built
    from ``|A|`` (the signed Laplacian) or from ``A`` itself (the unsigned one, which can leave a
    zero on the diagonal), and neither is built here because this package has no signed network.
    """
    if kind not in LAPLACIANS:
        msg = f"kind must be one of {', '.join(LAPLACIANS)}, got {kind!r}"
        raise ValueError(msg)
    if graph.is_directed():
        msg = (
            "laplacian is undefined for a directed graph: §8.4 gives it two Laplacians, one with "
            "the indegree on the diagonal and one with the outdegree, and this cannot pick for "
            "you. Take graph.to_undirected() if that is what you meant, or wait for the directed "
            "Laplacians (ATL-37)"
        )
        raise ValueError(msg)
    matrix, order = adjacency(graph, nodes, weight=weight, sparse=sparse)
    degrees = _sums(matrix, axis=1)
    if kind == "combinatorial":
        scale = np.ones_like(degrees)
    else:
        scale = _inverse(degrees, root=kind == "symmetric")
    if sparse:
        combinatorial = sp.diags_array(degrees, format="csr") - matrix
        left = sp.diags_array(scale, format="csr")
        if kind == "combinatorial":
            return sp.csr_array(combinatorial), order
        if kind == "random-walk":
            return sp.csr_array(left @ combinatorial), order
        return sp.csr_array(left @ combinatorial @ left), order
    combinatorial_dense = np.diag(degrees) - dense(matrix)
    if kind == "combinatorial":
        return combinatorial_dense, order
    if kind == "random-walk":
        return combinatorial_dense * scale[:, np.newaxis], order
    return combinatorial_dense * scale[:, np.newaxis] * scale[np.newaxis, :], order


# ----------------------------------------------------------------- §5.5-5.6 the solvers


def eigenpairs(
    matrix: Matrix, k: int | None = None, largest: bool = True, sparse: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The eigenvalues and eigenvectors of a square matrix, sorted (§5.5).

    Returns ``(values, vectors)`` where ``values[i]`` pairs with the *column* ``vectors[:, i]``,
    and the rows of ``vectors`` follow whatever node order built the matrix. ``largest=True``
    sorts descending -- the convention for the adjacency and the stochastic matrix, whose leading
    eigenvalue carries the meaning (§8.2) -- and ``largest=False`` ascending, which is the
    convention for the Laplacian, whose *smallest* eigenvalue is the interesting one and is
    always 0 (§8.4). ``k`` keeps only that many. The multiplicity §5.5 defines is read off the
    returned values: count how many of them are equal.

    **The dense path (``sparse=False``, the default).** Symmetric input is solved with
    ``numpy.linalg.eigh``, and symmetry is checked rather than assumed. This is deliberate, and
    is the same choice ``measures._eigenvector`` documents: a dense, exact, seed-free solver
    always gives the same answer for the same graph, where the iterative solvers below start
    from a vector and can land differently between two runs. Undefined, and raised on, for a
    non-square matrix -- use ``svd`` instead -- and for a non-symmetric matrix whose eigenvalues
    turn out to be complex, which §8.4 says is what a single directed edge produces. The
    non-symmetric-but-real case (the stochastic matrix of an undirected graph, §8.2, whose
    leading eigenvalue is always 1) is solved with ``numpy.linalg.eig``, reduced to a real basis
    and sorted the same way. Always returns the **full spectrum** before ``k`` truncates it, so
    this is the only path for a caller that needs more than a handful of eigenpairs -- ``svd``'s
    counterpart and :func:`graphrag.sna.walks.random_target_time`, which sums over every
    eigenvalue but the leading one, are why the full spectrum stays available at all.

    **The sparse path (``sparse=True``, ATL-ENT-3).** For a large, sparse, symmetric matrix and a
    *small* ``k`` -- :func:`graphrag.sna.walks.fiedler_vector`'s second-smallest eigenpair,
    :func:`graphrag.sna.walks.consensus_time`'s -- densifying an otherwise-sparse matrix just to
    diagonalise it whole is the expensive step, and it is also the one this path skips:
    ``scipy.sparse.linalg.eigsh`` (implicitly restarted Lanczos, ARPACK) finds the ``k`` extreme
    eigenpairs directly from the sparse matrix, without ever holding the ``n x n`` dense array.
    ``which='LA'``/``'SA'`` picks the largest or smallest **algebraic** eigenvalues to match
    ``largest``; this repository's own callers only ever route a symmetric Laplacian through it
    (never the plain adjacency or the stochastic matrix, which are exactly where §8.4 warns an
    extreme-eigenvalue solver without a shift struggles), and the empty-graph, singular-Laplacian
    case that motivates a spectral gap near zero converges in practice without one -- tested on
    both a well-connected and a near-disconnected 3,000-node graph. Requires ``k`` (there is no
    "sparse full spectrum": that is what the dense path is for) and ``k < n - 1``, both raised on
    when violated; raises if ARPACK fails to converge, naming the dense path as the fallback,
    rather than returning a partly-converged answer silently. **Not exact**: consumers that
    assert dense and sparse agree do so to a numerical tolerance, never with ``==``, and
    :data:`SPARSE_ABOVE_NODES` is the threshold this repository's own callers use to decide when
    the tolerance is worth it.
    """
    if sparse:
        return _sparse_eigenpairs(matrix, k, largest)
    values, vectors = _eigen(dense(matrix))
    if largest:
        values, vectors = values[::-1], vectors[:, ::-1]
    if k is not None:
        values, vectors = values[:k], vectors[:, :k]
    return np.ascontiguousarray(values), np.ascontiguousarray(vectors)


def _sparse_eigenpairs(
    matrix: Matrix, k: int | None, largest: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest or smallest eigenpairs of a large sparse symmetric matrix by Lanczos
    iteration. See :func:`eigenpairs`'s "sparse path" for what this is and is not a substitute
    for."""
    square = matrix if isinstance(matrix, sp.csr_array) else sp.csr_array(sp.csr_matrix(matrix))
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        msg = f"eigenpairs needs a square matrix, got shape {square.shape}; use svd instead"
        raise ValueError(msg)
    size = square.shape[0]
    if k is None:
        msg = (
            "eigenpairs(sparse=True) needs k: scipy.sparse.linalg.eigsh solves for k eigenpairs "
            "at a time and never the full spectrum a dense call would -- pass sparse=False for that"
        )
        raise ValueError(msg)
    if size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    if k < 1 or k >= size - 1:
        msg = f"eigenpairs(sparse=True) needs 1 <= k < n - 1 (n={size}), got k={k}"
        raise ValueError(msg)
    difference = square - square.T
    if difference.nnz and float(np.abs(difference.data).max()) > 1e-8:
        msg = (
            "eigenpairs(sparse=True) needs a symmetric matrix: scipy.sparse.linalg.eigsh solves "
            "the symmetric eigenproblem only, and this module's asymmetric matrices (a directed "
            "adjacency, the stochastic matrix) are not one -- densify and use numpy.linalg.eig "
            "through eigenpairs(sparse=False) for those"
        )
        raise ValueError(msg)
    which = "LA" if largest else "SA"
    try:
        values, vectors = spla.eigsh(square.astype(np.float64), k=k, which=which)
    except spla.ArpackNoConvergence as exc:
        msg = (
            f"the sparse eigensolver (ARPACK, which={which!r}) did not converge for k={k} on a "
            f"{size}-node matrix; eigenpairs(sparse=False) solves it exactly, at the cost of the "
            "dense O(n^3) factorisation this path exists to avoid"
        )
        raise ValueError(msg) from exc
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    return np.ascontiguousarray(values[order]), np.ascontiguousarray(vectors[:, order])


def _eigen(square: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues ascending, and the matching eigenvector columns. See ``eigenpairs``."""
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        msg = f"eigenpairs needs a square matrix, got shape {square.shape}; use svd instead"
        raise ValueError(msg)
    if square.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    if np.allclose(square, square.T):
        symmetric = np.linalg.eigh(square)  # ascending by construction, and exact
        return symmetric.eigenvalues, symmetric.eigenvectors
    raw_values, raw_vectors = np.linalg.eig(square)
    if not np.allclose(raw_values.imag, 0.0):
        msg = (
            "this matrix has complex eigenvalues, which §8.4 says is what a directed graph "
            "gives; decide what the imaginary part means before dropping it"
        )
        raise ValueError(msg)
    values, vectors = raw_values.real, _real_basis(raw_vectors)
    order = np.argsort(values)
    return values[order], vectors[:, order]


def _real_basis(vectors: np.ndarray) -> np.ndarray:
    """Real unit eigenvectors from what ``numpy.linalg.eig`` returned, for a real spectrum.

    LAPACK hands back a complex basis whenever an eigenvalue is repeated, even when every
    eigenvalue is real -- which the row-stochastic matrix of an undirected graph is (§8.2). For
    a real eigenvalue, the real and the imaginary part of a complex eigenvector are each
    themselves eigenvectors, so the larger of the two is taken and rescaled to unit length. What
    this cannot recover is a *choice* inside a repeated eigenvalue's eigenspace, because there
    is none to make: read the multiplicity there, not the individual vector (§5.5).
    """
    real = np.where(
        np.linalg.norm(vectors.real, axis=0) >= np.linalg.norm(vectors.imag, axis=0),
        vectors.real,
        vectors.imag,
    )
    norms = np.linalg.norm(real, axis=0)
    return np.divide(real, norms, out=np.zeros_like(real), where=norms > 0)


def svd(matrix: Matrix, k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The singular value decomposition ``M = U Σ V^T``, truncated to ``k`` components (§5.6).

    Returns ``(u, singular_values, vt)`` with the singular values descending, which is the one
    part of the factorisation that is unique -- §5.6 notes there are many valid ``U`` and ``V``,
    so the *sign* of a column is not meaningful on its own and two runs on differently ordered
    input may flip one. Unlike eigendecomposition this is defined for a non-square matrix, which
    is why it, not ``eigenpairs``, is what an incidence matrix goes through.

    ``k=None`` keeps every component. ``k`` larger than ``min(rows, columns)`` is clamped to it.
    """
    grid = dense(matrix)
    u, singular_values, vt = np.linalg.svd(grid, full_matrices=False)
    if k is not None:
        keep = max(0, min(k, singular_values.shape[0]))
        u, singular_values, vt = u[:, :keep], singular_values[:keep], vt[:keep, :]
    return u, singular_values, vt


def nmf(
    matrix: Matrix, k: int, seed: int | None = None, max_iter: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorisation: ``M ≈ W @ H`` with no negative entry (§5.6).

    Returns ``(w, h)``, where ``w`` has one row per row of ``M`` and ``k`` columns -- a node's
    loading on each of ``k`` parts -- and ``h`` has one row per part. §5.6 makes the case for it
    over PCA: the components of a PCA may be negative, and a negative amount of a thing is hard
    to read, while NMF's parts only ever add up, which is why it is the factorisation used for
    overlapping communities (Part X).

    Deterministic: the ``nndsvda`` initialisation is computed from the matrix rather than drawn,
    and ``seed`` pins what remains. It is still a local optimum of a non-convex problem, so the
    reconstruction is an approximation, not a decomposition.

    Undefined for a matrix with a negative entry -- a Laplacian, for instance, whose off-diagonal
    entries are ``-w`` by construction (§8.4) -- and for ``k`` above ``min(rows, columns)``; both
    raise rather than return a number.

    The rest of §5.6 is deliberately not built: **PCA**, because §5.6 introduces it only to
    motivate NMF and nothing in the programme consumes principal components, and the **tensor
    decompositions** (rank/PARAFAC, Tucker), because they factorise a multilayer tensor this
    package does not build -- §5.6 itself points the reader elsewhere for them.
    """
    grid = dense(matrix)
    if grid.size and float(grid.min()) < 0.0:
        msg = "nmf needs a non-negative matrix; this one has negative entries"
        raise ValueError(msg)
    if k < 1 or k > min(grid.shape):
        msg = f"k must be between 1 and min(rows, columns) = {min(grid.shape)}, got {k}"
        raise ValueError(msg)
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=max_iter)
    w: np.ndarray = model.fit_transform(grid)
    h: np.ndarray = model.components_
    return w, h
