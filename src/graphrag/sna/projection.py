"""Bipartite projections (ch. 26): the eleven ways this package turns a two-mode network into a
one-mode one, and the report that compares them.

Every network this package builds except ``relations`` is a *projection*. What was observed is a
set of memberships -- this speaker in that document, this entity in that passage -- and the graph
is what you get by joining two members of one mode because they share a member of the other.
Chapter 26 is about the fact that **the weight on the resulting edge is a choice, not a
measurement**: "In the following sections we explore the different ways in which one can project
a bipartite network. They all boil down to the same strategy: we use a different criterion to
give the projected edges a weight, we establish a threshold, and drop the edges below this
minimum acceptable weight" (p. 370).

The schemes, in the order the chapter introduces them:

``simple`` (§26.1, p. 371)
    ``w(u,v) = |N(u) ∩ N(v)|``, the number of shared opposite-mode nodes. The default here and
    the reason the chapter exists: one power user "who watched everything" joins every movie to
    every other movie, so the projection is a hairball and its weights are combinatorial.
``jaccard`` (§26.1, p. 371)
    the same count "normalized with the size of the union of the neighbor sets",
    ``|N(u) ∩ N(v)| / |N(u) U N(v)|``. Between 0 and 1, and blind to how much was shared in
    absolute terms: two nodes with one membership each, shared, score 1.0.
``cosine``, ``pearson``, ``euclidean`` (§26.2, pp. 372-373)
    the *vectorized* projection: read each row of the incidence matrix as a vector and use a
    standard vector similarity. Figure 26.6's worked example gives all three next to the simple
    weight. ``pearson`` is stored as the book plots it, ``r + 1``, because a projection weight
    has to be non-negative; ``euclidean`` as ``1 / (d + 1)``, for the same reason. §26.2 warns
    twice about exactly this step: "it's not always immediately obvious how to translate a
    distance into a similarity while preserving its properties", and these measures "were not
    really developed with network data in mind", so they do not fix the hub problem.
``hyperbolic`` (§26.3, p. 374)
    ``w(u,v) = Σ_{z ∈ N(u) ∩ N(v)} 1/k_z``: each shared node contributes less the more nodes it
    touches, because "bandwidth is finite" -- a paper with hundreds of coauthors is weak evidence
    that any two of them know each other. The result "is similar to simple weight, but it
    exaggerates the differences, so that thresholding becomes easier".
``probs`` / ``resource`` (§26.4, p. 375)
    resource allocation, Zhou, Ren, Medo and Zhang, *Bipartite network projection and personal
    recommendation*, Physical Review E 76(4):046115, 2007: ``w(u,v) = Σ_z 1/(k_u k_z)``, two
    steps of discounting instead of one. ``resource`` is an alias, since §26.4 names the family
    "resource allocation" and the variant "ProbS". **Asymmetric**: see below.
``heats`` (§26.4, p. 376)
    the ProbS variant of Zhou, Kuscsik, Liu, Medo, Wakeling and Zhang (PNAS 107(10):4511, 2010)
    that normalises by the *destination*: ``w(u,v) = Σ_z 1/(k_v k_z)``. "HeatS is the transpose
    of ProbS."
``hybrid`` (§26.4, p. 376)
    Lü and Liu (Physical Review E 83(6):066119, 2011) interpolate the two:
    ``w(u,v) = Σ_z 1/(k_u^λ k_v^(1-λ) k_z)``. λ=1 is ProbS, λ=0 is HeatS, λ=0.5 the middle.
``randomwalk`` (§26.5, p. 376)
    "we take the resource allocation to the extreme. Rather than looking at 2-step walks, we look
    at infinite length random walks": ``w(u,v) = π_v A(u,v)``, where ``A`` is the two-step
    transition matrix (which is ProbS with its diagonal kept, and is row-stochastic) and ``π`` is
    its stationary distribution. Yildirim and Coscia, *Using random walks to generate associations
    between objects*, PLoS One 9(8):e104813, 2014.

**Asymmetry, and what this package does with it.** ProbS, HeatS, the hybrid and the random walk
are directed in the book: "u's score for v would be (k_u k_z)^-1, while v's score would be
(k_v k_z)^-1" (p. 375). The networks here are undirected, so p. 376's own remedy is used -- "you
can make the result of resource allocation symmetric by always choosing the minimum or maximum
between w(u,v) and w(v,u), or simply their average" -- and the average is taken. Nothing is
thrown away: each edge also carries ``weight_uv`` and ``weight_vu``, the two directed scores in
node-sorted order. This is **not** a directed network: ``--network relations`` is the only
directed network this package builds, and its direction is a claim somebody wrote down, not an
artefact of a normalisation. :data:`ASYMMETRIC` names the four schemes this applies to.

**What the book defines and what this adds.**

*The edge set.* Every scheme here puts an edge exactly where the simple count is non-zero -- two
nodes that share at least one opposite-mode node, which is §26.7's definition of a projection.
The book does not say this outright but observes it in §26.6 ("almost all these methods return
the same set of non-zero weighted edges"), and it is what makes the rank agreement of
:func:`compare_schemes` well defined: the schemes are compared over one edge list.

*The diagonal.* Zeroed everywhere. §26.1 says so for the simple product and §26.4 says the
resource-allocation diagonal is well defined but optional ("also self-loops can be annoying
sometimes... you can manually set W's diagonal to zero"); nothing downstream here reads a
self-loop, so there is one convention rather than one per scheme. The random-walk scheme needs
the diagonal on the way through -- it is what makes the transition matrix stochastic -- so it is
zeroed only at the end.

*The stationary distribution.* §26.5 says to multiply the ProbS matrix by its stationary
distribution and leaves it there. A two-step walk on one mode of a bipartite network is
reversible with ``π_u ∝ k_u`` (§14.4's π for the induced chain), so π is taken in that closed
form: it is exact, it is defined on a disconnected projection -- where the eigenvector is not
unique and any solver's answer is arbitrary -- and ``test_sna_projection.py`` checks it against
the dominant left eigenvector on a connected example.

*The memberships this package hands it are binary.* ``project_incidence`` takes any incidence
matrix and §26.4's weighted form (``Σ_z B_uv/(k_u k_z)``) falls out of the same arithmetic, but
:func:`graphrag.sna.matrices.incidence` builds a 0/1 matrix, so the corpus path never uses that
generality: a speaker who wrote about an entity in nine documents and one who wrote about it in
one are the same membership here, and
:func:`graphrag.sna.export.speaker_entity_bipartite` collapses that multiplicity before
projecting. That is exactly the "saturation problem" §26.2 opens with -- the hundredth
collaboration counting as much as the second -- and it is the motivation the chapter gives for
the vectorized schemes, which cannot act on it as long as the matrix is binary. Feeding the
document counts through as weights is a change to what a membership *is*, not to this module,
and it would move the default projection's numbers, so it is left to a ticket that can say what
the new weight means.

*The hyperbolic formula.* p. 374 displays ``w(u,v) = Σ_z 1/(k_z - 1)`` and justifies the minus
one, but its own figure 26.7 and its own matrix description in §26.4 ("in hyperbolic, you
multiply the adjacency matrix A with its degree-normalized transpose") both compute ``Σ_z
1/k_z``: the figure's .46 is ``1/3 + 1/8`` and its .79 is ``1/3 + 1/3 + 1/8``. The figure's
arithmetic is implemented, because it is also what makes ProbS "the same as the hyperbolic
projection, but you normalize differently" (p. 376) come out right.

**The projection is not the end.** §26.6 closes on the thing that matters more than the choice of
scheme: every scheme returns "extremely dense projections, as a single common node is enough to
create an edge", and "the process to get rid of hairballs has two steps: first one performs the
bipartite projection, and then she applies a threshold to throw away low-weighted edges". The
threshold in :func:`compare_schemes` is a plain weight cut, named as one -- proper backboning is
chapter 27.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.matrices import Side, incidence
from graphrag.sna.stats import spearman

__all__ = [
    "ASYMMETRIC",
    "DEFAULT_KEEP_TOP",
    "DEFAULT_LAMBDA",
    "PAIRS",
    "PROJECTION",
    "PROJECTION_LAMBDA",
    "SCHEMES",
    "SCHEME_NOTES",
    "ProjectionComparison",
    "Scheme",
    "SchemeWeights",
    "compare_schemes",
    "graph_from_incidence",
    "project",
    "project_incidence",
    "projections_payload",
    "render_projections",
    "symmetrise",
]

Scheme = Literal[
    "simple",
    "jaccard",
    "cosine",
    "pearson",
    "euclidean",
    "hyperbolic",
    "resource",
    "probs",
    "heats",
    "hybrid",
    "randomwalk",
]

#: Every weighting scheme chapter 26 describes, in the order the chapter introduces them.
SCHEMES: tuple[Scheme, ...] = (
    "simple",
    "jaccard",
    "cosine",
    "pearson",
    "euclidean",
    "hyperbolic",
    "resource",
    "probs",
    "heats",
    "hybrid",
    "randomwalk",
)

#: The schemes whose matrix is not symmetric in the book (§26.4 p. 375, §26.5 p. 377). The graph
#: builders average the two directions and record both on the edge; see the module docstring.
ASYMMETRIC: frozenset[str] = frozenset({"resource", "probs", "heats", "hybrid", "randomwalk"})

#: ``resource`` is §26.4's name for the family and ``probs`` the name of this member of it, so
#: the two are one scheme under two names rather than two schemes that happen to agree.
_ALIASES: dict[str, Scheme] = {"resource": "probs"}

#: The hybrid's interpolation between HeatS (0) and ProbS (1), §26.4 p. 376. The book calls 1/2
#: "the middle point between HeatS and ProbS" and states no default, so this is this package's.
DEFAULT_LAMBDA = 0.5

#: The share of edges :func:`compare_schemes` keeps when no explicit threshold is given. A
#: convention, not the book's: §26.6 thresholds without naming a level.
DEFAULT_KEEP_TOP = 0.1

#: Where a projected graph keeps the two-mode memberships it was projected from, so that
#: :func:`graphrag.sna.null.bipartite_preserving` can rewire those rather than the projection.
PAIRS = "pairs"

#: Where it records which of :data:`SCHEMES` produced its weights, and the hybrid's λ. A null
#: model, a report and a reader all need it: a weight of 0.46 means nothing until the scheme is
#: named, and a re-projection under a different scheme is not a null for this network.
PROJECTION = "projection"
PROJECTION_LAMBDA = "projection_lambda"

#: One line per scheme, for the report's table and for ``--projection``'s help. Each says what
#: the weight *is*, because the number alone is not readable.
SCHEME_NOTES: dict[str, str] = {
    "simple": "shared opposite-mode nodes, counted (§26.1)",
    "jaccard": "shared over the union of both neighbourhoods, 0 to 1 (§26.1)",
    "cosine": "cosine similarity of the two incidence rows (§26.2)",
    "pearson": "Pearson correlation of the two rows, plus 1 so it cannot be negative (§26.2)",
    "euclidean": "1/(Euclidean distance between the rows + 1) (§26.2)",
    "hyperbolic": "sum of 1/k over the shared nodes: a hub contributes less (§26.3)",
    "resource": "alias of probs, §26.4's name for the family",
    "probs": "resource allocation ProbS, sum of 1/(k_u k_z); asymmetric, averaged (§26.4)",
    "heats": "HeatS, the transpose of ProbS: sum of 1/(k_v k_z); asymmetric, averaged (§26.4)",
    "hybrid": "ProbS and HeatS interpolated by --lambda; asymmetric, averaged (§26.4)",
    "randomwalk": "stationary probability of the two-step walk; asymmetric, averaged (§26.5)",
}

#: What the comparison report is measuring, printed above the numbers.
COMPARISON_FRAME = (
    "One two-mode observation projected {count} ways. The nodes and the edges are the same in "
    "every row -- two nodes are joined when they share at least one {unit} -- and only the "
    "weights differ, so the rows are {count} readings of one network rather than {count} "
    "networks. Nothing here is compared against chance: the question §26.6 asks is whether the "
    "schemes disagree with each other, not whether any of them beats a null."
)


def _array(values: Any) -> np.ndarray:
    """A float array, whatever numpy handed back. ``numpy``'s stubs widen the result of an
    elementwise expression to ``Any``, and this package's type checking is strict, so every
    formula above returns through here rather than carrying an ``Any`` into a public signature."""
    return np.asarray(values, dtype=np.float64)


def _resolve(scheme: str) -> Scheme:
    """The canonical name of a scheme, having refused one this module does not implement."""
    if scheme not in SCHEMES:
        msg = f"projection scheme must be one of {', '.join(SCHEMES)}, got {scheme!r}"
        raise ValueError(msg)
    return _ALIASES.get(scheme, scheme)


def _lambda_or_refuse(lam: float) -> float:
    """§26.4: λ "should be a number between zero and one". Outside that it is not an
    interpolation between HeatS and ProbS and the weight has no reading at all."""
    if not 0.0 <= lam <= 1.0:
        msg = f"--lambda must be between 0 and 1 (0 is HeatS, 1 is ProbS), got {lam}"
        raise ValueError(msg)
    return lam


# ----------------------------------------------------------------------------- the matrices


def project_incidence(
    matrix: np.ndarray,
    scheme: str = "simple",
    *,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """One chapter-26 projection of an incidence matrix, **onto its rows**.

    ``matrix`` is the ``|V1| x |V2|`` incidence matrix of :func:`graphrag.sna.matrices.incidence`:
    left nodes on the rows, right nodes on the columns. To project onto the columns instead, pass
    ``matrix.T`` -- the arithmetic is identical with the modes swapped, which is §26.1's "you need
    to pay attention to the dimension onto which you're projecting".

    Returns a dense ``|V1| x |V1|`` array with a zero diagonal. Cell ``(u, v)`` is zero wherever
    ``u`` and ``v`` share no column, for every scheme: a projection joins nodes with common
    neighbours (§26.7), and a Pearson correlation between two rows that share nothing is a
    statement about the zeroes rather than a tie. The matrix is **not** symmetric for the schemes
    in :data:`ASYMMETRIC`; :func:`symmetrise` and :func:`graph_from_incidence` are where that is
    resolved, so that a caller who wants the book's directed W can have it.

    A node with no membership at all -- an empty row, or an empty column of the kind figure
    26.6's two-row inset has -- is not an error: it shares nothing with anybody, so its ``1/k``
    is never multiplied by anything and is taken as zero rather than as a division by zero.
    Raises ``ValueError`` when ``matrix`` is not two-dimensional, when ``scheme`` names no
    scheme, or when ``lam`` is outside ``[0, 1]``.
    """
    name = _resolve(scheme)
    lam = _lambda_or_refuse(lam)
    left = np.asarray(matrix, dtype=np.float64)
    if left.ndim != 2:
        msg = f"project_incidence() needs a 2-D incidence matrix, got shape {left.shape}"
        raise ValueError(msg)
    if left.size == 0:
        return np.zeros((left.shape[0], left.shape[0]), dtype=np.float64)
    row_degree = left.sum(axis=1)
    column_degree = left.sum(axis=0)

    shared = left @ left.T
    weights = _weights(name, left, shared, row_degree, column_degree, lam)
    weights = np.where(shared > 0, weights, 0.0)
    np.fill_diagonal(weights, 0.0)
    return weights


def _weights(
    name: Scheme,
    left: np.ndarray,
    shared: np.ndarray,
    row_degree: np.ndarray,
    column_degree: np.ndarray,
    lam: float,
) -> np.ndarray:
    """The scheme's raw matrix, before the support mask and the diagonal. One branch per formula
    in the module docstring; every one of them is a whole-matrix product, never a pairwise loop,
    which is §26.1's "this is equivalent to multiplying the bipartite adjacency matrix with its
    transpose"."""
    if name == "simple":
        return _array(shared)
    row_inverse = _reciprocal(row_degree)
    column_inverse = _reciprocal(column_degree)
    if name == "jaccard":
        union = row_degree[:, None] + row_degree[None, :] - shared
        return _array(np.divide(shared, union, out=np.zeros_like(shared), where=union > 0))
    if name == "cosine":
        norms = np.sqrt(np.einsum("ij,ij->i", left, left))
        outer = norms[:, None] * norms[None, :]
        return _array(np.divide(shared, outer, out=np.zeros_like(shared), where=outer > 0))
    if name == "pearson":
        return _array(_pearson_rows(left) + 1.0)
    if name == "euclidean":
        squared = (
            np.einsum("ij,ij->i", left, left)[:, None]
            + np.einsum("ij,ij->i", left, left)[None, :]
            - 2.0 * shared
        )
        return _array(1.0 / (np.sqrt(np.maximum(squared, 0.0)) + 1.0))
    # Everything below is the hyperbolic sum, normalised differently (§26.4 p. 376).
    hyperbolic = (left * column_inverse[None, :]) @ left.T
    if name == "hyperbolic":
        return _array(hyperbolic)
    if name == "probs":
        return _array(hyperbolic * row_inverse[:, None])
    if name == "heats":
        return _array(hyperbolic * row_inverse[None, :])
    if name == "hybrid":
        origin = row_inverse[:, None] ** lam
        destination = row_inverse[None, :] ** (1.0 - lam)
        return _array(hyperbolic * origin * destination)
    return _random_walk(hyperbolic, row_degree, row_inverse)


def _pearson_rows(left: np.ndarray) -> np.ndarray:
    """The Pearson correlation between every pair of rows (§26.2).

    A constant row -- a node connected to every column, or to none -- has no variance, so its
    correlation with anything is undefined; it is returned as 0, which the caller shifts to a
    weight of 1, the value that says "no information either way". ``numpy.corrcoef`` would
    return ``nan`` there and poison every downstream measure instead.
    """
    centred = left - left.mean(axis=1, keepdims=True)
    spread = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    outer = spread[:, None] * spread[None, :]
    return _array(
        np.divide(
            centred @ centred.T,
            outer,
            out=np.zeros((left.shape[0], left.shape[0])),
            where=outer > 0,
        )
    )


def _reciprocal(degrees: np.ndarray) -> np.ndarray:
    """``1/k``, and zero where ``k`` is zero.

    A node with no membership contributes to no sum, so the zero never reaches a weight; it is
    here so that a hand-built matrix with an empty row or column -- the book's own figure 26.6
    inset has seven empty columns -- projects rather than raising.
    """
    return _array(np.divide(1.0, degrees, out=np.zeros_like(degrees), where=degrees > 0))


def _random_walk(
    hyperbolic: np.ndarray, row_degree: np.ndarray, row_inverse: np.ndarray
) -> np.ndarray:
    """§26.5's ``w(u,v) = π_v A(u,v)``, with ``A`` the two-step transition matrix.

    ``A`` is ProbS *with its diagonal*, which is what makes it row-stochastic: the walk leaves
    ``u`` for one of its ``k_u`` opposite-mode neighbours and comes back to one of that
    neighbour's ``k_z``, and landing back on ``u`` is one of the outcomes. ``π_u ∝ k_u``, as the
    module docstring explains. The diagonal is dropped by the caller, after the multiplication.
    """
    transition = hyperbolic * row_inverse[:, None]
    total = float(row_degree.sum())
    stationary = row_degree / total if total > 0 else np.zeros_like(row_degree)
    return _array(stationary[None, :] * transition)


def symmetrise(matrix: np.ndarray) -> np.ndarray:
    """``(W + W.T) / 2``: §26.4 p. 376's third remedy for an asymmetric projection.

    The book offers the minimum, the maximum or the average; the average is taken because it is
    the only one of the three that preserves the total weight of the two directions, so the
    symmetrised network carries the same amount of "resource" the directed one did.
    """
    square = np.asarray(matrix, dtype=np.float64)
    return (square + square.T) / 2.0


# ----------------------------------------------------------------------------- the graphs


def graph_from_incidence(
    matrix: np.ndarray,
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    side: Side = "left",
    scheme: str = "simple",
    lam: float = DEFAULT_LAMBDA,
) -> nx.Graph:
    """The projected graph of an incidence matrix already built, with its provenance on it.

    Split out from :func:`project` so that :func:`graphrag.sna.null.bipartite_preserving` can
    project a *rewired* incidence matrix through exactly the same arithmetic as the observation:
    a null re-projected under a different scheme is not a null for this network.

    Isolated nodes are kept -- a speaker who shares no document with anybody is still a speaker --
    and the node set is the whole of ``left`` (or ``right``), added in the order given, which
    :func:`graphrag.sna.matrices.incidence` sorts. Insertion order decides matrix order
    everywhere in ``networkx``, so a graph that is not built identically twice cannot be seeded.

    ``weight`` is an ``int`` for the simple scheme and a ``float`` for every other one, because a
    count is a count: the default projection's numbers, and the file formats' types, are exactly
    what they were before this module existed.
    """
    name = _resolve(scheme)
    lam = _lambda_or_refuse(lam)
    oriented = np.asarray(matrix, dtype=np.float64)
    if side == "right":
        oriented = oriented.T
    labels = list(left if side == "left" else right)
    weights = project_incidence(oriented, name, lam=lam)
    directed = weights if name in ASYMMETRIC else None
    if directed is not None:
        weights = symmetrise(weights)

    graph = nx.Graph()
    graph.add_nodes_from(labels)
    rows, columns = np.nonzero(np.triu(weights, k=1) > 0)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        value = float(weights[row, column])
        data: dict[str, Any] = {"weight": round(value) if name == "simple" else value}
        if directed is not None:
            # Both directions of the book's W, in node order, so the averaging above is
            # reversible and a reader can see how far apart the two were (§26.4 p. 375).
            data["weight_uv"] = float(directed[row, column])
            data["weight_vu"] = float(directed[column, row])
        graph.add_edge(labels[row], labels[column], **data)
    graph.graph[PROJECTION] = scheme
    graph.graph[PROJECTION_LAMBDA] = lam
    return graph


def project(
    pairs: Iterable[tuple[Any, Any]],
    side: Side = "left",
    *,
    scheme: str = "simple",
    lam: float = DEFAULT_LAMBDA,
) -> nx.Graph:
    """Project a two-mode edge list onto one side, weighted by ``scheme`` (ch. 26).

    ``pairs`` are ``(left, right)`` memberships -- (speaker, document), (entity, passage) -- and
    ``side`` picks the mode to keep: ``left`` joins two left nodes that share a right one,
    ``right`` the other way round. A repeated pair is one membership, as in
    :func:`graphrag.sna.matrices.incidence`, so the simple weight counts *distinct* shared
    neighbours, which is what ``|N(u) ∩ N(v)|`` means.

    The memberships are kept on the graph under :data:`PAIRS` and the scheme under
    :data:`PROJECTION`: a projection is not an observation -- the memberships are -- so a null
    model that rewires the projected edges invents graphs no membership table could produce, and
    one that re-projects under another scheme measures a different network.
    ``graphrag.sna.export.bipartite_projection`` is the corpus-facing wrapper around this.
    """
    memberships = sorted(set(pairs))
    if not memberships:
        empty = nx.Graph()
        empty.graph[PAIRS] = memberships
        empty.graph[PROJECTION] = scheme
        empty.graph[PROJECTION_LAMBDA] = _lambda_or_refuse(lam)
        return empty
    matrix, rows, columns = incidence(memberships)
    graph = graph_from_incidence(
        np.asarray(matrix, dtype=np.float64), rows, columns, side=side, scheme=scheme, lam=lam
    )
    graph.graph[PAIRS] = memberships
    return graph


# ----------------------------------------------------------------------------- §26.6 comparison


@dataclass(frozen=True)
class SchemeWeights:
    """What one scheme did to one network's edge weights: §26.6's figure 26.10, in numbers.

    ``readings`` is how many numbers the row describes: one per *ordered* pair of joined nodes,
    so twice the undirected edge count. §26.6 compares the schemes in the form the chapter
    defines them, and four of them are directed there, so the two directions of an edge are two
    readings. On the symmetric schemes the pair of readings is simply equal.

    ``entropy`` is the Shannon entropy in bits of those weights read as a distribution
    (``p_e = w_e / Σw``), which is this package's summary of the histograms the book plots rather
    than anything §26.6 defines. It is at most ``max_entropy = log2(readings)``, reached when
    every reading is equal; the further below it, the more of the network's total weight sits on
    a handful of edges -- the "one edge with weight 2,252" of p. 377.

    ``cut`` is the weight at which ``kept`` readings survive, and ``kept_share`` is
    ``kept / readings``. ``agreement`` is how far this scheme's surviving set is the same set the
    first scheme in the comparison keeps (Jaccard of the two), which is §26.6's real question:
    not whether the weights differ but whether the *network* does once thresholded.
    """

    scheme: str
    note: str
    readings: int
    minimum: float
    median: float
    maximum: float
    mean: float
    entropy: float
    max_entropy: float
    cut: float
    kept: int
    kept_share: float
    agreement: float


@dataclass(frozen=True)
class ProjectionComparison:
    """Every scheme run over one set of memberships, as §26.6 does it on its Twitter network."""

    persona_id: str
    network: str
    side: Side
    unit: str
    frame: str
    lam: float
    left_nodes: int
    right_nodes: int
    memberships: int
    nodes: int
    edges: int
    """Undirected edges of the projection: pairs sharing at least one opposite-mode node."""
    readings: int
    """Ordered pairs compared, which is twice ``edges``. See :class:`SchemeWeights`."""
    schemes: tuple[SchemeWeights, ...]
    correlations: dict[tuple[str, str], float | None]
    threshold: float | None
    keep_top: float

    @property
    def names(self) -> tuple[str, ...]:
        """The scheme names, in the order they were run."""
        return tuple(summary.scheme for summary in self.schemes)


def compare_schemes(
    pairs: Sequence[tuple[str, str]],
    side: Side = "left",
    *,
    schemes: Sequence[str] = SCHEMES,
    lam: float = DEFAULT_LAMBDA,
    keep_top: float = DEFAULT_KEEP_TOP,
    threshold: float | None = None,
    persona_id: str = "",
    network: str = "",
    unit: str = "opposite-mode node",
    frame: str = "",
) -> ProjectionComparison:
    """Run every scheme over one two-mode observation and compare them (§26.6).

    §26.6 asks three questions of a real projection and this answers all three: what the space of
    edge weights looks like under each scheme (figure 26.10), whether the schemes disagree about
    *specific* edges rather than only about the shape of the distribution (figure 26.11, answered
    here with Spearman rather than the book's log-log Pearson, because the schemes are on
    incomparable scales and only their order can be compared), and what survives a threshold --
    since "these techniques, while necessary, are not usually sufficient" and the hairball goes
    only after the second step.

    ``keep_top`` is the share of readings each scheme keeps, so every scheme keeps the same
    *number* and the comparison is of *which* ones. ``threshold`` overrides it with an absolute
    weight cut applied to every scheme, which is only readable when the schemes are on one scale
    -- they are not -- and is there for asking what one scheme keeps at a stated weight.

    **Directed, unlike the graphs.** Every scheme is read here in the form chapter 26 defines it,
    so ProbS, HeatS, the hybrid and the random walk are compared as the asymmetric matrices they
    are, over ordered pairs. That is what makes figure 26.11(c) reproducible: the graph builders
    average the two directions (§26.4 p. 376), and averaging a matrix with its transpose gives
    the same answer for a matrix and for its transpose -- so as *undirected networks* ProbS and
    HeatS are identical, and a comparison of the averaged forms would report a correlation of 1
    where the chapter reports -0.34. The difference between them is real and lives in the
    direction, which is why every asymmetric edge keeps both of its scores.

    Raises ``ValueError`` when there are no memberships, when ``schemes`` is empty or names one
    that does not exist, or when ``keep_top`` is not in ``(0, 1]``.
    """
    memberships = sorted(set(pairs))
    if not memberships:
        msg = "compare_schemes() needs at least one membership: there is nothing to project"
        raise ValueError(msg)
    names = [_named(scheme) for scheme in schemes]
    if not names:
        msg = f"compare_schemes() needs at least one scheme of {', '.join(SCHEMES)}"
        raise ValueError(msg)
    if not 0.0 < keep_top <= 1.0:
        msg = f"--keep-top is a share of the edges and must be in (0, 1], got {keep_top}"
        raise ValueError(msg)

    matrix, rows, columns = incidence(memberships)
    dense = np.asarray(matrix, dtype=np.float64)
    labels = rows if side == "left" else columns
    oriented = dense if side == "left" else dense.T
    support = oriented @ oriented.T
    np.fill_diagonal(support, 0.0)
    # Both directions of every joined pair, because four of the schemes are directed (see above).
    edge_rows, edge_columns = np.nonzero(support > 0)
    readings = [
        (int(r), int(c)) for r, c in zip(edge_rows.tolist(), edge_columns.tolist(), strict=True)
    ]

    weights: dict[str, list[float]] = {}
    for name in names:
        matrix_for = project_incidence(oriented, name, lam=lam)
        weights[name] = [float(matrix_for[r, c]) for r, c in readings]

    kept: dict[str, set[int]] = {}
    cuts: dict[str, float] = {}
    for name in names:
        cut, surviving = _cut(weights[name], keep_top, threshold)
        cuts[name] = cut
        kept[name] = surviving
    reference = kept[names[0]]

    summaries = tuple(
        _summarise(name, weights[name], cuts[name], kept[name], reference, len(readings))
        for name in names
    )
    return ProjectionComparison(
        persona_id=persona_id,
        network=network,
        side=side,
        unit=unit,
        frame=frame,
        lam=lam,
        left_nodes=len(rows),
        right_nodes=len(columns),
        memberships=len(memberships),
        nodes=len(labels),
        edges=len(readings) // 2,
        readings=len(readings),
        schemes=summaries,
        correlations=_correlations(names, weights),
        threshold=threshold,
        keep_top=keep_top,
    )


def _named(scheme: str) -> str:
    """A scheme name as given, refused if unknown. Aliases keep their own name here so a report
    that was asked for ``resource`` says ``resource``."""
    _resolve(scheme)
    return scheme


def _cut(
    values: Sequence[float], keep_top: float, threshold: float | None
) -> tuple[float, set[int]]:
    """The weight each scheme is thresholded at, and the indices of the readings that survive it.

    With ``threshold`` the cut is that number. Without it the cut is the quantile that leaves
    ``keep_top`` of the readings, so every scheme keeps the same count and the comparison is
    about which edges rather than how many. Ties at the cut are kept, so a scheme with many equal
    weights -- ``simple`` on a sparse corpus, where most edges weigh 1 -- keeps more than the
    share asks for, which is itself the finding: there is no threshold that separates them.
    """
    if not values:
        return (float(threshold or 0.0), set())
    cut = float(threshold) if threshold is not None else float(np.quantile(values, 1.0 - keep_top))
    return cut, {index for index, value in enumerate(values) if value >= cut}


def _summarise(
    name: str,
    values: Sequence[float],
    cut: float,
    kept: set[int],
    reference: set[int],
    readings: int,
) -> SchemeWeights:
    """One row of the distribution table."""
    array = np.asarray(values, dtype=np.float64)
    union = kept | reference
    return SchemeWeights(
        scheme=name,
        note=SCHEME_NOTES[name],
        readings=readings,
        minimum=float(array.min()) if array.size else 0.0,
        median=float(np.median(array)) if array.size else 0.0,
        maximum=float(array.max()) if array.size else 0.0,
        mean=float(array.mean()) if array.size else 0.0,
        entropy=_weight_entropy(array),
        max_entropy=math.log2(readings) if readings > 1 else 0.0,
        cut=cut,
        kept=len(kept),
        kept_share=len(kept) / readings if readings else 0.0,
        agreement=len(kept & reference) / len(union) if union else 1.0,
    )


def _weight_entropy(values: np.ndarray) -> float:
    """Shannon entropy in bits of the weights read as a distribution over the readings.

    Zero when one reading carries all the weight, ``log2(n)`` when they are all equal. Zero for
    an empty or all-zero list, which has no distribution to describe.
    """
    total = float(values.sum())
    if values.size == 0 or total <= 0.0:
        return 0.0
    shares = values[values > 0] / total
    return float(-(shares * np.log2(shares)).sum())


def _correlations(
    names: Sequence[str], weights: Mapping[str, Sequence[float]]
) -> dict[tuple[str, str], float | None]:
    """Spearman between every pair of schemes over the shared reading list (§26.6, fig. 26.11).

    Rank rather than value, because the schemes are on scales that have nothing to do with each
    other -- a simple weight of 3 and a random-walk weight of 0.049 are the same edge -- and the
    book's own comparison is of the *shape* of the scattergram. ``None`` where the coefficient is
    undefined: fewer than three readings, or a scheme whose weights are all equal (``simple`` on
    a network where every pair shares exactly one node), which has no order to agree with.
    """
    scores: dict[tuple[str, str], float | None] = {}
    for first in names:
        for second in names:
            if first == second:
                scores[(first, second)] = 1.0
                continue
            if (second, first) in scores:
                scores[(first, second)] = scores[(second, first)]
                continue
            try:
                scores[(first, second)] = spearman(weights[first], weights[second]).coefficient
            except ValueError:
                scores[(first, second)] = None
    return scores


# ----------------------------------------------------------------------------- the report


def render_projections(comparison: ProjectionComparison) -> str:
    """The §26.6 comparison as markdown, frame and null first."""
    lines = [
        f"# Projection schemes — {comparison.persona_id or 'network'}",
        "",
        "**Implements.** Atlas §26.6, *Comparison in a Practical Scenario*: one two-mode "
        "network projected by every scheme of chapter 26, compared on its edge-weight "
        "distributions (figure 26.10), on whether the schemes rank the same edges the same way "
        "(figure 26.11) and on what survives a threshold.",
        "",
        f"**Sampling frame.** {comparison.frame}"
        if comparison.frame
        else "**Sampling frame.** Not recorded.",
        "",
        COMPARISON_FRAME.format(count=len(comparison.schemes), unit=comparison.unit),
        "",
        f"**n.** {comparison.left_nodes:,} left nodes and {comparison.right_nodes:,} right "
        f"nodes holding {comparison.memberships:,} memberships; projected onto the "
        f"{comparison.side} mode, {comparison.nodes:,} nodes and {comparison.edges:,} edges, "
        f"read as {comparison.readings:,} ordered pairs -- four of chapter 26's schemes give u a "
        "different score for v than v gives u (§26.4), so both directions are compared. The "
        "graphs the builders return average the two, which makes ProbS and HeatS the *same* "
        "undirected network; their difference is the direction, and it is visible only here.",
        "",
        "**Null model.** None, and none is possible here: a scheme is a definition, not a "
        "measurement, so there is nothing for it to be unlikely against. `sna analyze --null "
        "bipartite` is where a projected number meets a null, and it re-projects under this "
        "same scheme.",
        "",
    ]
    lines += _distribution_table(comparison)
    lines += _correlation_table(comparison)
    lines += _threshold_section(comparison)
    return "\n".join(lines)


def _distribution_table(comparison: ProjectionComparison) -> list[str]:
    """Figure 26.10 as a table: what each scheme did to the space of edge weights."""
    lines = [
        "## Edge-weight distributions",
        "",
        "Every row describes the same "
        f"{comparison.readings:,} ordered pairs. Entropy is of the weights read as a "
        "distribution over them, in bits, against a maximum of "
        f"{comparison.schemes[0].max_entropy:.2f} when they all weigh the same; the lower it is, "
        "the more of the weight sits on a handful of edges.",
        "",
        "| scheme | what the weight is | min | median | max | mean | entropy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines += [
        f"| `{row.scheme}` | {row.note} | {row.minimum:.4g} | {row.median:.4g} | "
        f"{row.maximum:.4g} | {row.mean:.4g} | {row.entropy:.2f} |"
        for row in comparison.schemes
    ]
    lines.append("")
    return lines


def _correlation_table(comparison: ProjectionComparison) -> list[str]:
    """Figure 26.11 as a matrix: Spearman between every pair of schemes."""
    names = comparison.names
    lines = [
        "## Rank agreement between schemes",
        "",
        "Spearman's rank correlation between the weights two schemes give the same ordered "
        f"pairs (n = {comparison.readings:,}). Rank, not value: the scales are unrelated, so "
        "only the order can agree. `—` means the coefficient is undefined, which happens when a "
        "scheme gives every pair the same weight. §26.6's own warning: schemes you would expect "
        "to agree need not, and on the book's Twitter data HeatS and ProbS came out at -0.34.",
        "",
        "| | " + " | ".join(f"`{name}`" for name in names) + " |",
        "| --- |" + " ---: |" * len(names),
    ]
    for first in names:
        cells = []
        for second in names:
            value = comparison.correlations.get((first, second))
            cells.append("—" if value is None else f"{value:.2f}")
        lines.append(f"| `{first}` | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _threshold_section(comparison: ProjectionComparison) -> list[str]:
    """What survives the second step: §26.6's threshold, named as the plain cut it is."""
    reference = comparison.schemes[0].scheme
    how = (
        f"an absolute weight cut of {comparison.threshold:g} applied to every scheme, which is "
        "only readable for schemes on the same scale"
        if comparison.threshold is not None
        else f"the heaviest {comparison.keep_top:.0%} of ordered pairs under each scheme"
    )
    return [
        "## What survives a threshold",
        "",
        f"Keeping {how}. This is a plain weight threshold, **not** a backbone: extracting the "
        "statistically significant edges is chapter 27's subject and a different operation. "
        f"`agreement` is how far a scheme's survivors are the ones `{reference}` keeps "
        "(Jaccard of the two sets), which is §26.6's question -- the schemes can rank "
        "differently and still leave the same network standing, or agree on the ranking and "
        "disagree on the tail.",
        "",
        f"| scheme | cut | kept | share | agreement with `{reference}` |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| `{row.scheme}` | {row.cut:.4g} | {row.kept:,} | {row.kept_share:.1%} | "
            f"{row.agreement:.2f} |"
            for row in comparison.schemes
        ],
        "",
        "**Reading it.** §26.6 closes on the point this table is for: every scheme returns "
        '"extremely dense projections, as a single common node is enough to create an edge", '
        "so the choice of scheme decides which edges a threshold leaves, and the threshold "
        "decides whether there is a network to read at all.",
    ]


def projections_payload(comparison: ProjectionComparison) -> dict[str, Any]:
    """The same comparison as plain JSON-able data."""
    return {
        "persona_id": comparison.persona_id,
        "network": comparison.network,
        "side": comparison.side,
        "unit": comparison.unit,
        "frame": comparison.frame,
        "implements": "26.6",
        "lambda": comparison.lam,
        "left_nodes": comparison.left_nodes,
        "right_nodes": comparison.right_nodes,
        "memberships": comparison.memberships,
        "nodes": comparison.nodes,
        "edges": comparison.edges,
        "readings": comparison.readings,
        "keep_top": comparison.keep_top,
        "threshold": comparison.threshold,
        "schemes": [
            {
                "scheme": row.scheme,
                "note": row.note,
                "readings": row.readings,
                "min": row.minimum,
                "median": row.median,
                "max": row.maximum,
                "mean": row.mean,
                "entropy": row.entropy,
                "max_entropy": row.max_entropy,
                "cut": row.cut,
                "kept": row.kept,
                "kept_share": row.kept_share,
                "agreement": row.agreement,
            }
            for row in comparison.schemes
        ],
        "correlations": [
            {"a": first, "b": second, "spearman": value}
            for (first, second), value in comparison.correlations.items()
        ],
    }
