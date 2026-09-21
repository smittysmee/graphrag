"""Node and graph embeddings (ch. 42): the embedding contract, spectral embeddings, and pooling.
Returns arrays and an :class:`Embedding`; builds no graphs and prints nothing.

**The contract (§42.1).** A node embedding is a vector per node that should be *small* (fewer
entries than the graph has nodes), *dense*, *permutation invariant* (the same graph gives the
same vectors however the node ids happen to sort), and *faithful* to some stated notion of node
similarity. A sliced row of the adjacency matrix fails the first three -- it has one entry per
node, is mostly zero, and reorders with the node ids -- which is the chapter's own worked
counter-example for why embeddings are built rather than read off ``A`` directly. §42.1 is also
explicit that there is no single "the" embedding of a graph: Figures 42.1 and 42.4 embed the same
nine-node graph two different, equally valid ways, because they optimise for different notions of
similarity (community membership versus structural equivalence). :class:`Embedding` is this
package's answer: a vector per node plus the node order, the method that built it, and the
provenance a report needs to say which question that method answers.

**Spectral embeddings (§42.3).** :func:`spectral` is the family's one member built here: the
smallest non-trivial eigenvectors of the symmetric normalised Laplacian, moved onto
:mod:`graphrag.sna.matrices` so this and :func:`graphrag.sna.cluster.spectral_embedding` agree
about what that matrix is (ATL-08). Laplacian Eigenmaps and Locally Linear Embedding, the
chapter's two other named members, are the same family on a different matrix -- see
:func:`spectral`'s docstring for which one and how to build it from the primitives already here.

**Pooling (§42.4).** A node embedding answers a question about one node; a graph classification
or comparison question (ATL-48) needs one vector for the whole graph. :func:`pool` is the
chapter's "flat pooling" family -- mean, sum and max over every node's vector -- which does not
look at the graph's structure at all, unlike the node-clustering and node-drop families §42.4
also describes; see :func:`pool` for why those are not built here.

**Random-walk embeddings (§43.1-43.2).** Where §42.3 decomposes a matrix, this family samples:
"the embedding of node v is created by starting a bunch of random walks from v and noting down
which nodes appear in these random walks" (p. 622), the same idea Word2Vec applies to sentences.
:func:`deepwalk` is that idea with an unbiased walk; :func:`node2vec` is the same walk biased by
§43.2's ``p`` (how much a step hates to backtrack) and ``q`` (how much it hates to leave what it
has already seen), with ``p = q = 1`` collapsing exactly to :func:`deepwalk`. Both fit a numpy
skip-gram with negative sampling (:func:`_numpy_skipgram`) rather than the exact softmax p.624
writes down, which is the approximation the chapter itself recommends: "we can approximate the
softmax normalization via negative sampling ... While using negative sampling technically gives
you a function that is not softmax, it approximates it well enough to be considered the same."
Negative samples are drawn proportional to node degree, per the chapter's own suggestion (p. 624).
``gensim``, never imported unless asked for, is the "optional accelerator" the ticket names: the
same walks, trained by compiled code instead of the pure-numpy loop here, behind
``backend="gensim"``.

**Heterogeneous graphs: metapath2vec (§43.3).** An unbiased walk on a two-mode network spends
almost its whole budget on whichever side has the most edges -- the chapter's own example is a
physics paper with 5,154 co-authors, "every pair of co-authors is a valid path in the
co-authorship network" -- so :func:`metapath2vec` instead walks a stated schema, here
``speaker -> entity -> speaker`` repeated, which is what keeps every step of the walk answering
"who wrote about what" rather than drowning in one side's degree.

**Applications and limitations (§43.4-43.5).** §43.4's own menu -- feed the vectors to a
clusterer, or score a candidate edge by the cosine of its two endpoints' vectors -- is
:func:`node2vec_scores` here and ``sna/analysis.py``'s ``--features node2vec|metapath2vec``
routing into the same K-means/GMM clustering :func:`spectral` already feeds (ATL-43's own two
uses); graph summarisation (ATL-46) and 2-D visualisation (ATL-49/51) are the same idea again and
are not repeated in this module. §43.5's caution that this whole family is transductive -- "the
presence/absence of new nodes will change the random walk content" -- and reads no node or edge
attribute is carried into :mod:`graphrag.sna.guide` as reading rules rather than restated in every
docstring below.
"""

from __future__ import annotations

import importlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.experiment import ScoreTable
from graphrag.sna.matrices import Node, eigenpairs, laplacian
from graphrag.sna.measures import undirected_view

PoolingKind = Literal["mean", "sum", "max"]
POOLINGS: tuple[str, ...] = ("mean", "sum", "max")

# --------------------------------------------------------------- random-walk embeddings (§43.1)

WALKS_PER_NODE = 10
"""How many random walks :func:`node2vec`, :func:`deepwalk` and :func:`metapath2vec` start from
each node (§43.1: "you simply perform a bunch of random walks -- normally a certain number of
random walks starting at a given node, with a given length, and so on"). §43.1 also names this as
one of the parameters the family "require[s] you to set a lot of"; it is a module constant rather
than a value derived from the graph so a report can print exactly what produced its vectors
(:attr:`Embedding.provenance` carries it), and so raising it is a stated choice rather than a
silent one."""

WALK_LENGTH = 40
"""How many nodes one random walk visits, including the node it started from."""

EMBED_WINDOW = 5
"""The skip-gram's window (§43.1): how many walk positions on each side of a node count as
"co-appearing" with it for the loss the embedding fits."""

EMBED_NEGATIVE = 5
"""Negative samples drawn per positive (node, context) pair -- §43.1's approximation of the
softmax normaliser, "estimating the similarity of two nodes that co-appear in a random walk with
a sample of the similarities of the nodes that don't co-appear with them"."""

EMBED_EPOCHS = 5
"""Full passes the skip-gram makes over every walk before it stops."""

EMBED_LEARNING_RATE = 0.025
"""The skip-gram's starting step size, linearly annealed toward :data:`EMBED_MIN_LR_FRACTION` of
itself over training, the way the original Word2Vec implementation anneals its own. Never printed
in a report: unlike :data:`WALKS_PER_NODE` and its neighbours, a caller does not choose this."""

EMBED_MIN_LR_FRACTION = 1e-4
"""The floor :data:`EMBED_LEARNING_RATE` anneals to, as a fraction of its starting value, so the
last updates of a long run still move the vectors instead of multiplying by (near) zero."""

NEGATIVE_SAMPLING_SMOOTHING = 1e-3
"""Added to every node's degree before it is normalised into the negative-sampling distribution
(§43.1, p. 624: "pick nodes in the negative sample with a probability proportional to their
degree"), so an isolated node -- degree 0, and never the *positive* half of a pair since no walk
passes through it -- can still be drawn as a negative example instead of being assigned a literal
zero that ``numpy.random.Generator.choice`` would refuse to normalise."""

EMBED_MAX_NODES = 2_000
"""The largest network :func:`node2vec`, :func:`deepwalk` and :func:`metapath2vec` will build
walks and train a skip-gram for with ``backend="numpy"``. §43.1 puts no number on "a lot of
parameters", but the cost is real: :data:`WALKS_PER_NODE` walks of length :data:`WALK_LENGTH`
from every node, each token touched by :data:`EMBED_EPOCHS` passes of a pure-Python/numpy
training loop. Past this bound the honest answer is to sample the network first (``sna sample``)
or train with ``backend="gensim"``, the "optional accelerator" that runs the same walks through
compiled code instead of raising the limit."""

EIGENGAP_RTOL = 1e-6
"""How close two consecutive eigenvalues have to be, relative to their own size, to count as the
same degenerate block for :func:`_eigengap_cut`. Loose enough to catch the exact ties floating
point gives an exactly-regular subgraph (a clique's internal eigenvalues, a pair of nodes with
identical neighbourhoods), tight enough not to fold together two genuinely different eigenvalues
that a real, irregular corpus network usually produces."""
EIGENGAP_ATOL = 1e-9
"""The absolute floor :data:`EIGENGAP_RTOL` needs beside it: several of a disconnected graph's
own eigenvalues are exactly 0, where a purely relative tolerance divides by nothing."""


@dataclass(frozen=True, eq=False)
class Embedding:
    """A node embedding under the §42.1 contract: one row per node, in a stated order, built by
    a named method, with the parameters that produced it recorded rather than implied.

    This dataclass enforces three of the chapter's four properties directly. **Small**: every
    builder here refuses a ``dims`` the graph cannot support rather than silently truncating (see
    :func:`spectral`). **Dense** and **permutation invariant**: ``__post_init__`` refuses a shape
    that does not have exactly one row per node, which is what makes ``vectors[i]`` mean
    ``nodes[i]`` regardless of how the caller happened to order the graph's own nodes. The fourth,
    *faithfulness* -- that similar nodes land at a low distance -- is a property of the method
    that built the vectors, not of the container, and is asserted by that method's own tests
    (:func:`spectral`'s two-clique known answer, for instance) rather than checked here.

    ``method`` and ``provenance`` exist because §42.1 is explicit that no single embedding is
    "the" embedding of a graph: two methods (or the same method with different parameters) give
    different, equally valid vectors, so a report comparing two Embeddings has to say whether they
    answer the same question before comparing them row by row.
    """

    vectors: np.ndarray
    """One row per node: ``vectors.shape == (len(nodes), dims)``."""
    nodes: list[Node]
    """The row order that ``vectors`` follows. Two embeddings of the same graph are only
    comparable row by row when this order agrees, which is why every builder here takes an
    explicit ``nodes`` rather than trusting a graph's own iteration order."""
    method: str
    """Which function built this, e.g. ``"spectral"``."""
    provenance: dict[str, object]
    """What a report needs next to the numbers to say which question this embedding answers:
    at minimum the chapter section and the matrix or loss the method used, plus anything that
    was randomised."""
    _index: dict[Node, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            msg = (
                "an embedding's vectors must be 2-D (one row per node), got shape "
                f"{self.vectors.shape}"
            )
            raise ValueError(msg)
        if self.vectors.shape[0] != len(self.nodes):
            msg = (
                "an embedding needs exactly one row per node (§42.1's permutation invariance "
                f"depends on it): {self.vectors.shape[0]} rows for {len(self.nodes)} nodes"
            )
            raise ValueError(msg)
        if len(set(self.nodes)) != len(self.nodes):
            msg = "an embedding's node order must not repeat a node: row i must mean node i alone"
            raise ValueError(msg)
        object.__setattr__(self, "_index", {node: i for i, node in enumerate(self.nodes)})

    @property
    def dims(self) -> int:
        """The embedding's dimensionality: ``vectors.shape[1]``."""
        return int(self.vectors.shape[1])

    def vector(self, node: Node) -> np.ndarray:
        """The row for ``node``. Raises ``KeyError`` for a node this embedding does not have."""
        if node not in self._index:
            raise KeyError(node)
        row: np.ndarray = self.vectors[self._index[node]]
        return row


def _canonical_sign(vectors: np.ndarray) -> np.ndarray:
    """Flip each column's sign so its largest-magnitude entry is positive.

    An eigenvector and its negation solve the same eigenproblem equally well, so a solver is free
    to hand back either one; nothing here corrects a mistake, it removes a coin flip that §42.1's
    permutation-invariance property would otherwise leave in place. The rule commutes with a node
    relabelling -- flipping is decided by each column's own values, not by row position -- so a
    graph and a relabelled copy of it embed to rows that agree, not rows that agree up to a
    per-dimension sign a solver chose arbitrarily.
    """
    peak_rows = np.argmax(np.abs(vectors), axis=0)
    peak_values = vectors[peak_rows, np.arange(vectors.shape[1])]
    flips = np.where(peak_values < 0, -1.0, 1.0)
    return vectors * flips


def _tied(a: float, b: float) -> bool:
    """Whether two consecutive eigenvalues are the same degenerate block, within
    :data:`EIGENGAP_RTOL`/:data:`EIGENGAP_ATOL`."""
    return bool(np.isclose(a, b, rtol=EIGENGAP_RTOL, atol=EIGENGAP_ATOL))


def _eigengap_cut(values: np.ndarray, dims: int, canonical: int = 0) -> tuple[int, float | None]:
    """How many of the ``dims`` requested eigenvectors to actually keep, and the eigenvalue of
    the block that was cut into, if any.

    ``values`` is the full ascending spectrum :func:`graphrag.sna.matrices.eigenpairs` returned,
    including the trivial eigenvalue at index 0 that :func:`spectral` always discards; ``dims``
    counts eigenvectors *after* that discard, so ``values[dims]`` is the last one a caller asked
    to keep and ``values[dims + 1]`` the first one dropped.

    If those two agree within tolerance, ``dims`` cuts into a run of mutually tied eigenvalues --
    a degenerate eigenspace with no canonical basis (see :func:`spectral`'s own docstring on why
    a dense solver's choice within one is arbitrary) -- and every dimension in that run is
    dropped, not an arbitrary few of them: this walks back to the last dimension before the run
    started. A clean cut (no tie at the boundary) returns ``dims`` unchanged. Never returns less
    than 1, however large the tied run: a caller that asked for at least one dimension gets at
    least one back, even from a graph with no internal structure to offer it.

    ``canonical`` is how many of the leading kept dimensions already have a canonical basis --
    the zero eigenspace of a disconnected graph, which :func:`_canonical_null_space` replaces with
    one indicator per component. Those are tied (all exactly 0) but not arbitrary, so a cut
    inside them is honoured as asked and the walk back never goes below them: without this, a
    network in more components than ``dims + 1`` collapsed to a single, arbitrary dimension.
    """
    if dims <= canonical or dims + 1 >= len(values):
        return dims, None
    if not _tied(values[dims], values[dims + 1]):
        return dims, None
    cut = dims
    while cut > canonical + 1 and _tied(values[cut], values[cut - 1]):
        cut -= 1
    return max(cut - 1, canonical, 1), float(values[cut])


def _canonical_null_space(
    graph: nx.Graph, order: Sequence[Node], vectors: np.ndarray
) -> tuple[np.ndarray, int]:
    """Replace the solver's basis of the Laplacian's zero eigenspace with one vector per
    connected component, and return how many components there are.

    On a graph in ``c`` components the zero eigenvalue repeats ``c`` times (§8.4), and like any
    tied block its eigenvectors come back from ``numpy.linalg.eigh`` as an arbitrary rotation.
    This block does have a canonical basis, though: each component's own null vector, supported
    on that component alone. Each one is found by projecting the component's indicator onto the
    solver's zero block and keeping only the component's own rows, so it is exact whichever
    degree the Laplacian was normalised by (an isolated node gets its own indicator). Components
    are ordered largest first, ties by first appearance in ``order``, so the largest one's vector
    lands at index 0 and is the one :func:`spectral` discards as the trivial eigenvector: every
    kept component dimension then reads "in this smaller component", and the largest component
    sits at the origin of all of them.

    A graph with one component, or one whose projection degenerates (a node joined only to
    itself, say), is returned unchanged with ``c = 1``, which is the behaviour before this basis
    existed.
    """
    index = {node: i for i, node in enumerate(order)}
    parts = sorted(
        ([index[node] for node in part] for part in nx.connected_components(graph.subgraph(order))),
        key=lambda rows: (-len(rows), min(rows)),
    )
    count = len(parts)
    if count < 2 or vectors.shape[1] < count:
        return vectors, 1
    null = vectors[:, :count]
    basis = np.zeros((len(order), count), dtype=vectors.dtype)
    for column, rows in enumerate(parts):
        indicator = np.zeros(len(order), dtype=vectors.dtype)
        indicator[rows] = 1.0
        projected = null @ (null.T @ indicator)
        own = np.zeros_like(projected)
        own[rows] = projected[rows]
        norm = float(np.linalg.norm(own))
        if norm < EIGENGAP_ATOL:
            return vectors, 1
        basis[:, column] = own / norm
    out = vectors.copy()
    out[:, :count] = basis
    return out, count


def spectral(
    graph: nx.Graph,
    dims: int,
    *,
    seed: int | None = None,
    nodes: Sequence[Node] | None = None,
) -> Embedding:
    """The spectral node embedding of §42.3: the smallest non-trivial eigenvectors of the
    symmetric normalised Laplacian ``D^-1/2 L D^-1/2`` (§8.4), one embedding dimension per
    eigenvector, aligned to ``nodes`` (``list(graph.nodes)`` if not given).

    §42.3 frames spectral embedding as an eigenvector problem on "a transformation of A ... but
    most often of the Laplacian L", with different transformations answering different loss
    functions and "this is the main thing differentiating spectral embedding approaches" (p. 614).
    This function picks one member of that family: the symmetric normalised Laplacian, the same
    matrix :func:`graphrag.sna.cluster.spectral_embedding` clusters and the one the chapter's own
    min-cut aside points at (§11.5). The other two the chapter names are the same family on a
    different matrix, and are one call to :func:`graphrag.sna.matrices.eigenpairs` away rather
    than separately exposed here: **Locally Linear Embedding** takes the smallest eigenvectors of
    ``(I - A)^T(I - A)`` after discarding the very smallest, and **Laplacian Eigenmaps** takes the
    smallest eigenvectors of this same ``D^-1/2 L D^-1/2``, which is why picking this one matrix
    already gives a caller both readings the chapter offers under that name.

    The Laplacian's smallest eigenvalue is always 0 and its eigenvector is the trivial "every node
    looks the same" answer (§8.4) that §42.1's faithfulness property rules out, so it is always
    discarded; ``dims`` counts the eigenvectors kept **after** that discard, and each column's sign
    is canonicalised (see :func:`_canonical_sign`) so that discard is the only place a sign choice
    is made. On a graph with ``c`` connected components the zero eigenspace has dimension ``c``
    (§8.4), so the first ``c - 1`` kept dimensions of a disconnected graph separate its components
    from each other before anything about the structure inside one appears -- which is what the
    two-disjoint-cliques known answer below is. That block is tied at exactly 0, so the solver's
    basis for it is arbitrary; :func:`_canonical_null_space` swaps in one indicator per component
    (largest discarded, the rest by size), and ``provenance`` records ``components`` and
    ``component_dims``, how many kept dimensions do nothing but name a component.

    ``dims`` must leave at least one eigenvector after the discard: an ``n``-node graph offers
    ``n - 1``, so ``dims > n - 1`` is refused rather than silently clamped, which is §42.1's first
    property (an embedding needs fewer entries than the graph has nodes) read as a hard bound.
    :func:`graphrag.sna.cluster.spectral_embedding`, used to seed K-means and the blockmodel
    restarts, clamps instead of raising, because those callers already decided how many
    coordinates they want to cluster and asking them to catch this error would not change what
    they do next; this function is the contract a caller states ``dims`` against directly.

    **A requested ``dims`` that cuts into a degenerate eigenspace is truncated, not honoured.**
    An exactly regular subgraph -- a clique, or two nodes with identical neighbourhoods -- has a
    block of mutually tied eigenvalues with no internal structure to distinguish, so *any*
    orthonormal basis of that block is an equally valid answer; a dense, deterministic solver
    still has to pick one, and what it picks can concentrate almost all of a basis vector's
    weight on one or two nodes rather than spreading it out (empirically true of
    ``numpy.linalg.eigh`` on a large tied block). Feeding that to a distance-based consumer --
    K-means chief among them -- reads a solver artefact as if it were a structural finding.
    :func:`_eigengap_cut` catches this: if keeping ``dims`` eigenvectors would end partway
    through a tied block, every dimension in that block is dropped instead, down to the last one
    before it started, never below 1. ``provenance`` then carries ``dims_requested``,
    ``dims_kept`` and (when they differ) ``eigengap_eigenvalue``, the tied block's own value, so
    a report can say why fewer columns came back than were asked for; :func:`Embedding.dims`
    already reflects the count actually kept.

    A directed graph is flattened first (§6.2), for the reason
    :func:`graphrag.sna.matrices.laplacian` refuses a directed one outright: the Laplacian needs a
    symmetric adjacency. ``seed`` is accepted because every caller in this package's clustering
    code carries one, and it is recorded in ``provenance`` for a report to repeat, but it changes
    nothing: unlike the ``scikit-learn`` implementation this replaces, ``numpy.linalg.eigh`` is an
    exact dense solver with no random initialisation, so the same graph gives the same embedding
    on every call.
    """
    flat, note = undirected_view(graph)
    order = list(nodes) if nodes is not None else list(flat.nodes)
    n = len(order)
    if dims < 1:
        msg = f"dims must be at least 1, got {dims}"
        raise ValueError(msg)
    max_dims = max(n - 1, 0)
    if dims > max_dims:
        msg = (
            f"spectral needs dims <= {max_dims} for a {n}-node graph, once the trivial "
            "(constant) eigenvector is discarded -- §42.1 asks an embedding to have fewer "
            f"entries than the graph has nodes -- but dims={dims} was asked for"
        )
        raise ValueError(msg)
    sym_laplacian, lap_order = laplacian(flat, nodes=order, kind="symmetric")
    values, vectors_all = eigenpairs(sym_laplacian, largest=False)
    vectors_all, components = _canonical_null_space(flat, lap_order, vectors_all)
    kept, eigengap_eigenvalue = _eigengap_cut(values, dims, canonical=components - 1)
    selected = _canonical_sign(vectors_all[:, 1 : 1 + kept])
    provenance: dict[str, object] = {
        "chapter": "§42.3",
        "matrix": "symmetric normalised Laplacian (D^-1/2 L D^-1/2, §8.4)",
        "discarded_eigenvalue": float(values[0]),
        "eigenvalues": values[1 : 1 + kept].tolist(),
        "seed": seed,
        "n_nodes": n,
        "directed_flattened_note": note,
        "dims_requested": dims,
        "dims_kept": kept,
        "eigengap_eigenvalue": eigengap_eigenvalue,
        "components": components,
        "component_dims": min(components - 1, kept),
    }
    return Embedding(vectors=selected, nodes=lap_order, method="spectral", provenance=provenance)


def pool(embedding: Embedding, kind: PoolingKind = "mean") -> np.ndarray:
    """Flat graph-level pooling (§42.4): reduce every node vector of ``embedding`` to one vector
    describing the whole graph, for a task -- graph classification, or comparing two graphs'
    embeddings the way ATL-48 does -- that needs a single row rather than one per node.

    §42.4 calls this "flat pooling": apply the same reduction to every node vector without looking
    at the graph's structure at all. ``mean``, ``sum`` and ``max`` are the three the section
    describes that need nothing trained; the attention-weighted and hashed variants it also names
    need a learned model this package does not build. The two structure-aware families it
    describes are named here rather than built: **node clustering pooling** replaces the flat
    reduction with a fixed number of "virtual nodes", each the pooled vector of one community, and
    needs a community-discovery call this function is not in a position to choose on the caller's
    behalf (``sna community`` already exposes every method §42.4 would draw virtual nodes from);
    **node drop pooling** scores nodes for relevance and coarsens the graph by removing the least
    relevant, which needs a scoring rule (the chapter's own example is betweenness centrality) the
    same way. Both stay documented rather than built until a consumer names which scoring or
    community method it wants.

    ``mean`` is the only one of the three whose scale does not grow with the number of nodes, so
    it is the one comparable across two graphs of different order; ``sum`` says something about
    the graph's size on top of its structure, and ``max`` about its most extreme node in each
    dimension. Every kind inherits the permutation invariance :class:`Embedding` already
    guarantees at the node level (§42.1's third property, now asked of the graph vector): two
    isomorphic graphs embed to the same set of rows up to a permutation, so they pool to the same
    vector, while a graph with one more edge generally embeds -- and so pools -- differently.

    Undefined for an embedding with no nodes: there is nothing to reduce, and each kind's identity
    element (0 for a sum, nothing at all for a mean or a max) would look like a real answer to a
    caller that did not check for it first.
    """
    if kind not in POOLINGS:
        msg = f"kind must be one of {POOLINGS}, got {kind!r}"
        raise ValueError(msg)
    if embedding.vectors.shape[0] == 0:
        msg = "pool needs at least one node vector; this embedding has none"
        raise ValueError(msg)
    if kind == "mean":
        return np.asarray(embedding.vectors.mean(axis=0), dtype=np.float64)
    if kind == "sum":
        return np.asarray(embedding.vectors.sum(axis=0), dtype=np.float64)
    return np.asarray(embedding.vectors.max(axis=0), dtype=np.float64)


# ------------------------------------------------------------------ random-walk embeddings (ch. 43)


def _weighted_neighbours(
    graph: nx.Graph, order: Sequence[Node], index: Mapping[Node, int]
) -> list[list[tuple[int, float]]]:
    """Every node's neighbours as ``(row index, edge weight)`` pairs, in ``order``'s own indexing.

    Computed once per embedding call rather than read off ``graph`` at every step of every walk:
    a single call here can take :data:`WALKS_PER_NODE` times :data:`WALK_LENGTH` steps *per
    node*, and ``graph[node]`` is a dict view this would otherwise rebuild and re-weight on every
    one of them.
    """
    return [
        [
            (index[neighbour], float(graph.edges[node, neighbour].get("weight", 1.0)))
            for neighbour in graph.neighbors(node)
            if neighbour in index
        ]
        for node in order
    ]


def _weighted_choice(candidates: Sequence[tuple[int, float]], rng: random.Random) -> int:
    """One row index from ``candidates``, with probability proportional to its weight.

    Falls back to a uniform pick when every weight is zero or negative (a node whose only
    neighbours are joined by a zero-weight edge, which the walk should still be free to take)
    rather than raising on an all-zero distribution.
    """
    total = sum(weight for _, weight in candidates)
    if total <= 0:
        return candidates[rng.randrange(len(candidates))][0]
    draw = rng.random() * total
    upto = 0.0
    for node, weight in candidates:
        upto += weight
        if draw <= upto:
            return node
    return candidates[-1][0]


def _random_walk(
    neighbours: Sequence[Sequence[tuple[int, float]]],
    start: int,
    length: int,
    p: float,
    q: float,
    rng: random.Random,
) -> list[int]:
    """One second-order biased walk of up to ``length`` nodes from ``start`` (§43.2, Figure 43.3).

    The first step has no "node we just came from" to bias against, so it is a plain weighted
    draw over ``start``'s neighbours (§43.2 is silent on this edge case; every published node2vec
    implementation resolves it the same way). Every step after that reweights each neighbour
    ``x`` of the current node ``v`` by where ``x`` stands relative to the node ``t`` the walk just
    left: ``1/p`` if ``x == t`` (backtracking), ``1`` if ``x`` is also a neighbour of ``t`` (a
    shared neighbour -- BFS-like, biased up by a high ``q``), ``1/q`` otherwise (unseen from
    ``t`` -- DFS-like, biased up by a low ``q``), each on top of the edge's own weight. ``p = q =
    1`` makes every factor 1, which is exactly :func:`deepwalk`'s unbiased walk. Stops short of
    ``length`` at a dead end (a node with no neighbours) rather than teleporting elsewhere, which
    would be a different, unstated walk.
    """
    walk = [start]
    if not neighbours[start]:
        return walk
    walk.append(_weighted_choice(neighbours[start], rng))
    while len(walk) < length:
        prev, curr = walk[-2], walk[-1]
        candidates = neighbours[curr]
        if not candidates:
            break
        prev_neighbours = {node for node, _ in neighbours[prev]}
        biased = []
        for node, weight in candidates:
            if node == prev:
                factor = 1.0 / p
            elif node in prev_neighbours:
                factor = 1.0
            else:
                factor = 1.0 / q
            biased.append((node, weight * factor))
        walk.append(_weighted_choice(biased, rng))
    return walk


def _metapath_walk(
    neighbours: Sequence[Sequence[tuple[int, float]]],
    node_type: Sequence[int],
    start: int,
    length: int,
    cycle: Sequence[int],
    rng: random.Random,
) -> list[int]:
    """One metapath-constrained walk of up to ``length`` nodes from ``start`` (§43.3).

    At walk position ``i`` (``i >= 1``) the next node must carry ``cycle[(i - 1) % len(cycle)]``
    -- the metapath's own types repeated, since :func:`metapath2vec` requires a metapath that
    starts and ends with the same type. Stops short of ``length`` when the current node has no
    neighbour of the required type, which is the honest answer rather than relaxing the schema
    for one step.
    """
    walk = [start]
    step = 0
    while len(walk) < length:
        wanted = cycle[step % len(cycle)]
        candidates = [
            (node, weight) for node, weight in neighbours[walk[-1]] if node_type[node] == wanted
        ]
        if not candidates:
            break
        walk.append(_weighted_choice(candidates, rng))
        step += 1
    return walk


def _sigmoid(score: float) -> float:
    """A numerically stable logistic, used only on the scalar dot product the positive half of
    a pair trains."""
    if score >= 0:
        return float(1.0 / (1.0 + np.exp(-score)))
    exp_score = np.exp(score)
    return float(exp_score / (1.0 + exp_score))


def _sigmoid_array(scores: np.ndarray) -> np.ndarray:
    """The same logistic over a whole batch of dot products at once -- the negative samples of one
    pair -- clipped before the exponential so a still-large early-training dot product cannot
    overflow it."""
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0))), dtype=np.float64)


def _sgd_step(
    w_in: np.ndarray, w_out: np.ndarray, center: int, context: int, negatives: np.ndarray, lr: float
) -> None:
    """One SGNS training step for the pair ``(center, context)`` against its ``negatives`` (§43.1),
    updating ``w_in[center]`` and every touched row of ``w_out`` in place.

    The positive half is one dot product; the negative half is ``len(negatives)`` of them done as
    one batched matrix-vector product rather than a Python loop over each sample, which is what
    makes this function fast enough to call once per (walk position, window offset) pair on a
    real network rather than only on the small graphs a test builds. Every gradient here is
    computed from ``w_out``'s values *before* this call's own update to it -- one joint step, not
    several sequential ones reading each other's already-moved state -- and a repeated index in
    ``negatives`` (certain on a graph smaller than a few hundred nodes) is accumulated with
    :func:`numpy.add.at` rather than plain fancy-index ``+=``, which silently drops all but the
    last write to a repeated index.
    """
    center_vector = w_in[center]
    context_vector = w_out[context].copy()
    positive_gradient = (1.0 - _sigmoid(float(np.dot(center_vector, context_vector)))) * lr
    grad_in = positive_gradient * context_vector
    w_out[context] += positive_gradient * center_vector

    negative_vectors = w_out[negatives].copy()
    negative_gradient = (-_sigmoid_array(negative_vectors @ center_vector) * lr)[:, np.newaxis]
    grad_in += (negative_gradient * negative_vectors).sum(axis=0)
    np.add.at(w_out, negatives, negative_gradient * center_vector[np.newaxis, :])

    w_in[center] += grad_in


def _numpy_skipgram(
    walks: Sequence[Sequence[int]],
    n: int,
    dims: int,
    *,
    window: int,
    negative: int,
    epochs: int,
    degree: np.ndarray,
    seed: int | None,
) -> np.ndarray:
    """Word2Vec's skip-gram with negative sampling (§43.1), built on nothing but numpy.

    For every walk position, every context within ``window`` on either side is one positive pair,
    and ``negative`` nodes drawn proportional to ``degree`` (p. 624) are the negative pairs; each
    pair takes one :func:`_sgd_step`, annealing the learning rate linearly from
    :data:`EMBED_LEARNING_RATE` toward :data:`EMBED_MIN_LR_FRACTION` of itself over the run. The
    returned matrix is the *target* (input) vectors, ``w_in``: the ones every downstream reader of
    an :class:`Embedding` -- K-means, cosine similarity, a report -- reads as "the" embedding, the
    same convention Word2Vec itself uses.

    Every pair's negative samples are drawn in one batched call to ``numpy.random.Generator.choice``
    before training starts, rather than one call per pair: building the cumulative distribution
    ``choice`` samples from is the expensive part of that call and it does not depend on the pair,
    so paying for it once instead of once per pair is the difference between training being
    usable on a real network and not.

    Walks of length 1 (an isolated node) contribute no pair and are skipped; if none of ``walks``
    is longer than that, or ``negative`` is 0 (nothing to contrast a positive against), the random
    initialisation is returned unchanged rather than training on nothing.
    """
    rng = np.random.default_rng(seed)
    w_in = (rng.random((n, dims)) - 0.5) / dims
    w_out = np.zeros((n, dims))
    usable = [list(walk) for walk in walks if len(walk) > 1]
    if not usable or negative < 1:
        return w_in
    weights = degree + NEGATIVE_SAMPLING_SMOOTHING
    probabilities = weights / weights.sum()
    total_pairs = max(
        epochs * sum(2 * min(window, len(walk) - 1) * len(walk) for walk in usable), 1
    )
    negative_samples = rng.choice(n, size=(total_pairs, negative), p=probabilities)
    step = 0
    for _ in range(epochs):
        for walk in usable:
            length = len(walk)
            for i, center in enumerate(walk):
                lo, hi = max(0, i - window), min(length, i + window + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    context = walk[j]
                    lr = EMBED_LEARNING_RATE * max(1.0 - step / total_pairs, EMBED_MIN_LR_FRACTION)
                    _sgd_step(w_in, w_out, center, context, negative_samples[step], lr)
                    step += 1
    return w_in


def _optional_gensim() -> Any:
    """Import ``gensim`` on demand, or explain how to install it and stop.

    Same bridge convention as :func:`graphrag.sna.export._optional` (``to_igraph``,
    ``to_graph_tool``): never a dependency of this package, never declared as an extra of it,
    imported only when a caller asks for ``backend="gensim"``. A dynamic ``importlib`` import
    rather than a static ``import gensim`` statement, so mypy --strict does not need a stub for a
    module this package never requires.
    """
    try:
        return importlib.import_module("gensim")
    except ImportError as exc:
        msg = (
            "gensim is not installed. It is an optional accelerator for the skip-gram behind "
            "node2vec/deepwalk/metapath2vec (§43.1), never a dependency of this package: install "
            "it with `pip install gensim`, then call this again with backend='gensim'."
        )
        raise ImportError(msg) from exc


def _gensim_skipgram(
    walks: Sequence[Sequence[int]],
    n: int,
    dims: int,
    *,
    window: int,
    negative: int,
    epochs: int,
    seed: int | None,
) -> np.ndarray:
    """The same walks, trained by ``gensim.models.Word2Vec`` instead of :func:`_numpy_skipgram`.

    Node indices become string tokens for the duration of this call only; gensim's own vocabulary
    and negative-sampling machinery (its own unigram-frequency table, not §43.1's degree-based
    one) replace this module's, which is the accuracy/speed trade the "optional accelerator" is
    for. A node no walk ever reached gets no vocabulary entry and is filled with the same small
    random initialisation :func:`_numpy_skipgram` starts from, seeded the same way.
    """
    gensim = _optional_gensim()
    sentences = [[str(node) for node in walk] for walk in walks if len(walk) > 1]
    rng = np.random.default_rng(seed)
    vectors = (rng.random((n, dims)) - 0.5) / dims
    if not sentences:
        return vectors
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=dims,
        window=window,
        min_count=0,
        sg=1,
        negative=negative,
        epochs=epochs,
        seed=seed if seed is not None else 0,
        workers=1,
    )
    for i in range(n):
        key = str(i)
        if key in model.wv:
            vectors[i] = model.wv[key]
    return vectors


def _skipgram(
    walks: Sequence[Sequence[int]],
    n: int,
    dims: int,
    *,
    window: int,
    negative: int,
    epochs: int,
    degree: np.ndarray,
    seed: int | None,
    backend: str,
) -> np.ndarray:
    if backend == "gensim":
        return _gensim_skipgram(
            walks, n, dims, window=window, negative=negative, epochs=epochs, seed=seed
        )
    if backend != "numpy":
        msg = f"backend must be 'numpy' or 'gensim', got {backend!r}"
        raise ValueError(msg)
    return _numpy_skipgram(
        walks, n, dims, window=window, negative=negative, epochs=epochs, degree=degree, seed=seed
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, or 0.0 -- an honest "neither similar nor dissimilar" -- for a zero
    vector, which a node no walk ever reached can produce no other reading from."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _refuse_oversized(n: int, max_nodes: int, method: str) -> None:
    if n > max_nodes:
        msg = (
            f"{method} needs at most {max_nodes:,} nodes to keep its skip-gram tractable "
            f"(§43.1); this graph has {n:,}. Sample it first (`sna sample`), narrow the network, "
            "or pass a higher max_nodes if the run is expected to take a while."
        )
        raise ValueError(msg)


def _build_random_walk_embedding(
    graph: nx.Graph,
    dims: int,
    *,
    p: float,
    q: float,
    walks_per_node: int,
    walk_length: int,
    window: int,
    negative: int,
    epochs: int,
    seed: int | None,
    nodes: Sequence[Node] | None,
    max_nodes: int,
    method: str,
    section: str,
    backend: str,
) -> Embedding:
    """The shared body of :func:`node2vec` and :func:`deepwalk`: generate the walks, train the
    skip-gram, and wrap the result under the §42.1 contract with §43.1-43.2's own provenance."""
    if dims < 1:
        msg = f"dims must be at least 1, got {dims}"
        raise ValueError(msg)
    if p <= 0 or q <= 0:
        msg = f"p and q must be positive (§43.2's weights are 1/p and 1/q): got p={p}, q={q}"
        raise ValueError(msg)
    flat, note = undirected_view(graph)
    order = list(nodes) if nodes is not None else list(flat.nodes)
    n = len(order)
    _refuse_oversized(n, max_nodes, method)
    if n == 0:
        return Embedding(
            vectors=np.zeros((0, dims)), nodes=[], method=method, provenance={"chapter": section}
        )
    index = {node: i for i, node in enumerate(order)}
    neighbours = _weighted_neighbours(flat, order, index)
    degree = np.array([len(row) for row in neighbours], dtype=np.float64)
    py_rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    walks = [
        _random_walk(neighbours, start, walk_length, p, q, py_rng)
        for start in range(n)
        for _ in range(walks_per_node)
    ]
    vectors = _skipgram(
        walks,
        n,
        dims,
        window=window,
        negative=negative,
        epochs=epochs,
        degree=degree,
        seed=seed,
        backend=backend,
    )
    visited = {node for walk in walks for node in walk}
    provenance: dict[str, object] = {
        "chapter": section,
        "p": p,
        "q": q,
        "walks_per_node": walks_per_node,
        "walk_length": walk_length,
        "window": window,
        "negative": negative,
        "epochs": epochs,
        "seed": seed,
        "n_nodes": n,
        "unvisited_nodes": n - len(visited),
        "directed_flattened_note": note,
        "backend": backend,
    }
    return Embedding(vectors=vectors, nodes=order, method=method, provenance=provenance)


def node2vec(
    graph: nx.Graph,
    dims: int,
    *,
    p: float = 1.0,
    q: float = 1.0,
    walks_per_node: int = WALKS_PER_NODE,
    walk_length: int = WALK_LENGTH,
    window: int = EMBED_WINDOW,
    negative: int = EMBED_NEGATIVE,
    epochs: int = EMBED_EPOCHS,
    seed: int | None = None,
    nodes: Sequence[Node] | None = None,
    max_nodes: int = EMBED_MAX_NODES,
    backend: str = "numpy",
) -> Embedding:
    """node2vec (§43.2): :func:`deepwalk`'s walk, biased by ``p`` (1/p to backtrack to the node
    just left) and ``q`` (1/q to a neighbour that is not also a neighbour of the node just left).
    Low ``q`` pushes the walk away from where it has been (DFS-like, favouring structural roles
    over locality); high ``q`` pulls it back toward shared neighbours (BFS-like, favouring tight
    local communities); ``p = q = 1`` is exactly :func:`deepwalk`. ``dims`` is the embedding width
    fit by the skip-gram, not a spectral cut, so there is no eigengap to truncate at and no upper
    bound on it beyond what the training data can support.

    A directed graph is flattened first (§6.2), the same convention :func:`spectral` uses. Raises
    ``ValueError`` above ``max_nodes`` nodes (:data:`EMBED_MAX_NODES` by default) rather than
    running an unbounded training loop, and for a non-positive ``p`` or ``q``, which §43.2's
    weights (``1/p``, ``1/q``) cannot be taken of. ``seed`` fixes both the walk and the skip-gram's
    initialisation and negative sampling, so the same graph and seed reproduce the same vectors;
    two different seeds do not, unlike :func:`spectral`'s exact eigensolver.

    ``backend="gensim"`` trains the same walks with ``gensim.models.Word2Vec`` instead of this
    module's own :func:`_numpy_skipgram`, for a network too slow to train in pure Python; see
    :func:`_gensim_skipgram` for what changes. ``gensim`` is never imported unless this is asked
    for.
    """
    return _build_random_walk_embedding(
        graph,
        dims,
        p=p,
        q=q,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        window=window,
        negative=negative,
        epochs=epochs,
        seed=seed,
        nodes=nodes,
        max_nodes=max_nodes,
        method="node2vec",
        section="§43.2",
        backend=backend,
    )


def deepwalk(
    graph: nx.Graph,
    dims: int,
    *,
    walks_per_node: int = WALKS_PER_NODE,
    walk_length: int = WALK_LENGTH,
    window: int = EMBED_WINDOW,
    negative: int = EMBED_NEGATIVE,
    epochs: int = EMBED_EPOCHS,
    seed: int | None = None,
    nodes: Sequence[Node] | None = None,
    max_nodes: int = EMBED_MAX_NODES,
    backend: str = "numpy",
) -> Embedding:
    """DeepWalk (§43.1): unbiased random walks fit with the same skip-gram §43.2 gives node2vec.
    "DeepWalk is equal to Node2Vec when we set p = q = 1, so each neighbor ... is treated equally"
    (p. 625) -- this is exactly that call, kept as its own function because it is the family's
    plain baseline and the one the chapter introduces first, not because it does anything
    :func:`node2vec` cannot already do with its default ``p`` and ``q``.

    See :func:`node2vec` for every parameter, the bound this shares (:data:`EMBED_MAX_NODES`),
    and what ``backend="gensim"`` changes.
    """
    return _build_random_walk_embedding(
        graph,
        dims,
        p=1.0,
        q=1.0,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        window=window,
        negative=negative,
        epochs=epochs,
        seed=seed,
        nodes=nodes,
        max_nodes=max_nodes,
        method="deepwalk",
        section="§43.1",
        backend=backend,
    )


def metapath2vec(
    graph: nx.Graph,
    dims: int,
    *,
    metapath: Sequence[str] = ("speaker", "entity", "speaker"),
    mode_key: str = "mode",
    walks_per_node: int = WALKS_PER_NODE,
    walk_length: int = WALK_LENGTH,
    window: int = EMBED_WINDOW,
    negative: int = EMBED_NEGATIVE,
    epochs: int = EMBED_EPOCHS,
    seed: int | None = None,
    nodes: Sequence[Node] | None = None,
    max_nodes: int = EMBED_MAX_NODES,
    backend: str = "numpy",
) -> Embedding:
    """metapath2vec (§43.3): a walk constrained to repeat ``metapath``'s node types, on a
    heterogeneous graph whose nodes each carry ``mode_key`` (this package's own two-mode label,
    written by :func:`graphrag.sna.export.speaker_entity_bipartite` and read by
    :func:`graphrag.sna.coreperiphery.two_mode_sides`). The default metapath walks
    speaker-entity-speaker-entity-... on the corpus's own two-mode network, so a speaker's vector
    reflects the entities its passages mention rather than every other node degree alone would
    put it near (§43.3's own example: a paper's 5,154 co-authors, every pair a valid length-2
    path, drowning a plain walk's signal).

    ``metapath`` must have at least two entries and start and end with the same type -- the
    walk's schema has to close into a cycle to be repeated past the metapath's own length,
    exactly as the default ``speaker, entity, speaker`` does. A node with none of
    ``metapath[0]``'s type to start from, or a graph where some node carries no ``mode_key`` at
    all, raises rather than silently walking a smaller graph than asked for.

    Every node in the graph gets a row -- ``dims`` wide, aligned to ``nodes`` (``list(graph.nodes)``
    if not given) -- even one no walk reaches because it sits on a side the metapath never visits
    from any seed; :attr:`Embedding.provenance`'s ``unvisited_nodes`` says how many that was, and
    such a row is nothing but its random initialisation. See :func:`node2vec` for ``max_nodes``,
    ``seed`` and ``backend``, which all mean the same thing here.
    """
    if len(metapath) < 2 or metapath[0] != metapath[-1]:
        msg = (
            "metapath2vec needs a metapath that starts and ends with the same type, so the walk "
            f"can repeat it without stopping (§43.3); got {tuple(metapath)}."
        )
        raise ValueError(msg)
    if dims < 1:
        msg = f"dims must be at least 1, got {dims}"
        raise ValueError(msg)
    flat, note = undirected_view(graph)
    order = list(nodes) if nodes is not None else list(flat.nodes)
    n = len(order)
    _refuse_oversized(n, max_nodes, "metapath2vec")
    if n == 0:
        return Embedding(
            vectors=np.zeros((0, dims)),
            nodes=[],
            method="metapath2vec",
            provenance={"chapter": "§43.3"},
        )
    index = {node: i for i, node in enumerate(order)}
    missing = [node for node in order if mode_key not in flat.nodes[node]]
    if missing:
        msg = (
            f"metapath2vec needs every node to carry a {mode_key!r} attribute (§43.3): this is "
            "the network's own two-mode label. Build --network speakers-entities without "
            f"--project, or pass mode_key= for a differently-labelled two-mode graph. "
            f"{len(missing):,} of {n:,} node(s) carry none, e.g. {missing[0]!r}."
        )
        raise ValueError(msg)
    raw_types = [str(flat.nodes[node][mode_key]) for node in order]
    names = sorted(set(raw_types) | set(metapath))
    codes = {name: i for i, name in enumerate(names)}
    node_type = [codes[t] for t in raw_types]
    cycle = [codes[t] for t in metapath[1:]]
    seeds = [i for i, t in enumerate(raw_types) if t == metapath[0]]
    if not seeds:
        msg = f"no node carries mode={metapath[0]!r} to start the metapath walk from"
        raise ValueError(msg)
    neighbours = _weighted_neighbours(flat, order, index)
    degree = np.array([len(row) for row in neighbours], dtype=np.float64)
    py_rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    walks = [
        _metapath_walk(neighbours, node_type, start, walk_length, cycle, py_rng)
        for start in seeds
        for _ in range(walks_per_node)
    ]
    vectors = _skipgram(
        walks,
        n,
        dims,
        window=window,
        negative=negative,
        epochs=epochs,
        degree=degree,
        seed=seed,
        backend=backend,
    )
    visited = {node for walk in walks for node in walk}
    provenance: dict[str, object] = {
        "chapter": "§43.3",
        "metapath": list(metapath),
        "walks_per_node": walks_per_node,
        "walk_length": walk_length,
        "window": window,
        "negative": negative,
        "epochs": epochs,
        "seed": seed,
        "n_nodes": n,
        "seed_nodes": len(seeds),
        "unvisited_nodes": n - len(visited),
        "directed_flattened_note": note,
        "backend": backend,
    }
    return Embedding(vectors=vectors, nodes=order, method="metapath2vec", provenance=provenance)


def node2vec_scores(
    graph: nx.Graph,
    *,
    dims: int = 8,
    p: float = 1.0,
    q: float = 1.0,
    walks_per_node: int = WALKS_PER_NODE,
    walk_length: int = WALK_LENGTH,
    window: int = EMBED_WINDOW,
    negative: int = EMBED_NEGATIVE,
    epochs: int = EMBED_EPOCHS,
    seed: int | None = None,
    max_nodes: int = EMBED_MAX_NODES,
    backend: str = "numpy",
) -> ScoreTable:
    """A link-prediction score (§43.4, p. 635's exercise 4): ``score(u, v)`` is the cosine of
    ``u`` and ``v``'s :func:`node2vec` vectors, trained fresh on whatever graph this is called
    with -- :func:`graphrag.sna.experiment.evaluate_predictor` calls it with the training graph of
    a holdout, never the full network, which is what makes the AUC it returns a measurement
    rather than a foregone conclusion (§25.1).

    ``p = q = 1`` by default, i.e. the embedding is :func:`deepwalk`'s unless told otherwise. Every
    other keyword is :func:`node2vec`'s own; see it for what each means and for ``max_nodes``.
    """

    embedding = node2vec(
        graph,
        dims,
        p=p,
        q=q,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        window=window,
        negative=negative,
        epochs=epochs,
        seed=seed,
        max_nodes=max_nodes,
        backend=backend,
    )

    def score(u: str, v: str) -> float:
        return _cosine(embedding.vector(u), embedding.vector(v))

    return ScoreTable(name="node2vec cosine", section="§43.4", graph=graph, score=score)
