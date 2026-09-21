"""Core-periphery structure (Atlas ch. 32): one dense core and a periphery hanging off it.

Chapter 32 is about the other mesoscale organisation, the one community discovery cannot see.
"Many large scale networks have a common topology: a very densely connected set of core nodes,
and a bunch of casual nodes attaching only to few neighbors" (p. 450). Four things live here.

*The discrete model* (§32.1, pp. 451-452). Nodes are either core or periphery, the ideal pattern
is "core-core and core-periphery yes, periphery-periphery no", and the quality to maximise is
``sum_uv A_uv D_uv`` with ``D_uv = 1`` when either node is in the core. That sum is not
comparable between networks -- "you cannot compare the 'coreness' of two different networks
unless they have the same number of nodes and edges" -- so the number reported is the book's
second option, "the Pearson correlation coefficient between A and D" (p. 452).

*The continuous model* (§32.1, pp. 452-453). Same quality function with ``D_uv = c_u c_v`` for a
coreness vector ``c``, which lets a semi-periphery exist without anyone having to decide how many
classes there are. The book leaves how to build ``c`` open and lists the options; this module
takes the leading eigenvector of the adjacency, which is the exact maximiser of the book's own
quality function over unit-length vectors (§5.5), and also reports the k-core numbers as the
chapter's "a priori approach" so the two can be seen not to agree.

*The tension with communities* (§32.2, pp. 454-455). A network cannot be both: "in CP there isn't
space for communities, given that there's only one dense area and everything connects to it".
Every Louvain report therefore fits both ideal patterns to the same adjacency, scores both
against the same degree-preserving null, and says which of the two explains the edges better --
because a partition of a core-periphery network is the periphery cut into slices.

*Nestedness* (§32.4, pp. 458-460), which is the two-mode version of the same structure: "Core-
periphery structures are a generalization of a specific meso-scale organization of complex
systems that is relevant in multiple fields: nestedness."

Everything here reads the **binary** adjacency of the undirected, self-loop-free view of the
network. The discrete model is a statement about which pairs are connected at all, and the ideal
pattern it is correlated against holds only ones and zeros; correlating a count of shared
documents against a 0/1 mask would measure how heavy the core's edges are rather than whether
there is a core. The report says so next to the numbers.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp

from graphrag.sna.matrices import Node, adjacency, eigenpairs, incidence
from graphrag.sna.measures import core_shells, coreness, undirected_view
from graphrag.sna.null import (
    HOLDS_FIXED,
    Significance,
    bipartite_preserving,
    configuration,
    pairs_of,
    significance,
)
from graphrag.sna.stats import mean, pearson, std

__all__ = [
    "MAX_NESTEDNESS_NODES",
    "NULL_SAMPLES",
    "RESTARTS",
    "RICH_CLUB_ROWS",
    "RICH_CLUB_TAIL",
    "ContinuousCoreness",
    "CorePeripheryReport",
    "DiscreteCore",
    "Nestedness",
    "RichClub",
    "RichClubRow",
    "Tension",
    "community_correlation",
    "continuous_coreness",
    "core_periphery_payload",
    "core_periphery_report",
    "core_periphery_tension",
    "discrete_core",
    "nestedness",
    "render_core_periphery",
    "rich_club",
    "two_mode_sides",
]

#: How many random restarts the discrete search adds to its degree-seeded start. The book leaves
#: the search open -- "genetic algorithms, simulated annealing, or basin hopping" (p. 452) -- so
#: this is a convention of this package, not the chapter's answer. Raising it costs one hill
#: climb each and is the first thing to try when the correlation looks low for a network that
#: plainly has a core.
RESTARTS = 8

#: Sweeps the hill climb will make before it stops even if it is still improving. A sweep that
#: changes nothing ends the climb anyway; this only bounds the pathological case.
MAX_SWEEPS = 50

#: Null networks drawn for the rich club and for the tension check, matching the ``--samples``
#: default the rest of the package uses.
NULL_SAMPLES = 50

#: The share of the k range counted as "the high k's" when deciding whether a rich club is
#: sustained. Zhou and Mondragon's claim is about the top of the degree range, not about one k.
RICH_CLUB_TAIL = 0.5

#: How many rows of the rich-club curve the markdown prints. The full curve is in the payload.
RICH_CLUB_ROWS = 12

#: Above this many nodes on one side of a two-mode network, NODF's paired overlaps -- one per
#: pair of nodes on that side -- stop fitting in memory as a matrix. ``nestedness`` refuses
#: instead of thrashing, exactly as :func:`graphrag.sna.null.ergm` refuses above its own cap.
MAX_NESTEDNESS_NODES = 3_000

#: Below this many nodes above the threshold, phi(k) is a ratio over one or two pairs and says
#: nothing. Two is where it stops being defined at all; the curve is cut there.
MIN_RICH_CLUB_NODES = 2

#: The node attributes a builder may have used to say which mode a node is in. ``mode`` is what
#: :func:`graphrag.sna.export.speaker_entity_bipartite` writes; ``bipartite`` is what
#: ``networkx``'s own two-mode generators write, which is how the legendary Southern women
#: network arrives.
MODE_KEYS: tuple[str, ...] = ("mode", "bipartite")


# ----------------------------------------------------------------- the view and the correlation


def _simple_view(graph: nx.Graph) -> tuple[nx.Graph, str]:
    """The undirected, self-loop-free view chapter 32 draws its adjacency matrices on.

    Every figure in the chapter from 32.1 to 32.10 is a symmetric 0/1 matrix with a free
    diagonal, and the ideal patterns are defined over unordered pairs, so a directed network is
    flattened first (§6.2) and self loops are dropped: a node is neither its own core-core edge
    nor its own periphery-periphery one. The note this returns is what the report prints.
    """
    flat, note = undirected_view(graph)
    loops = list(nx.selfloop_edges(flat))
    if loops:
        flat = flat.copy()
        flat.remove_edges_from(loops)
    return flat, note


def _phi(*, pairs: int, edges: int, ideal_pairs: int, ideal_edges: int) -> float:
    """The correlation between two binary vectors, from counts alone (§32.1, p. 452).

    The book's quality measure is the Pearson correlation between the adjacency ``A`` and the
    ideal pattern ``D``, both read as one value per unordered pair of distinct nodes. When both
    are binary, that correlation is the phi coefficient and needs only four counts: ``pairs``
    (how many pairs there are, ``n(n-1)/2``), ``edges`` (how many of them carry an edge),
    ``ideal_pairs`` (how many the ideal pattern marks with a one) and ``ideal_edges`` (how many
    pairs are marked *and* carry an edge). Computing it this way is what lets the search below
    evaluate a candidate core in constant time instead of building an n-by-n mask for each one;
    a unit test holds it to :func:`graphrag.sna.stats.pearson` over the explicit matrices.

    Undefined -- and returned as ``nan`` -- when either vector is constant: a network with no
    edges or with every pair joined, an ideal that marks no pair or every pair. The last case is
    the one §32.2 turns on: the sum ``sum_uv A_uv D_uv`` is maximised by putting every node in
    the core, and it is the correlation, not the sum, that refuses to call that a core.
    """
    marked, total = float(ideal_pairs), float(pairs)
    hit, seen = float(ideal_edges), float(edges)
    spread = marked * (total - marked) * seen * (total - seen)
    if total <= 0 or spread <= 0:
        return math.nan
    return float((total * hit - marked * seen) / math.sqrt(spread))


def _pattern_correlation(matrix: np.ndarray, ideal: np.ndarray) -> float:
    """Pearson between the off-diagonal entries of the adjacency and of a real-valued ideal.

    The counterpart of :func:`_phi` for the continuous model, where ``D_uv = c_u c_v`` is not
    binary and no closed form in four counts exists. Both matrices are symmetric, so each pair
    is read twice; duplicating every pair changes neither correlation nor its reading, and it
    keeps the code honest about what it is comparing. The diagonal is excluded because §8.1
    leaves it free and a node is not connected to itself.

    ``nan`` when either side is constant, for the reason :func:`_phi` gives.
    """
    size = matrix.shape[0]
    if size < 3:
        return math.nan
    mask = ~np.eye(size, dtype=bool)
    left, right = matrix[mask], ideal[mask]
    if float(np.ptp(left)) == 0.0 or float(np.ptp(right)) == 0.0:
        return math.nan
    return pearson(left.tolist(), right.tolist()).coefficient


# ----------------------------------------------------------------------- §32.1 discrete model


@dataclass(frozen=True)
class DiscreteCore:
    """Borgatti and Everett's discrete partition, with the correlation that scored it."""

    core: list[Node]
    """The core, sorted. Empty when the network was too small or too sparse to fit one."""
    periphery: list[Node]
    correlation: float
    """The Pearson correlation between the adjacency and the ideal pattern (§32.1, p. 452).
    1.0 is the perfect discrete model of Figure 32.1: every core-core and core-periphery pair
    joined, no periphery-periphery pair joined. ``nan`` when it is undefined -- a network with
    no edges, or one where every pair is joined, neither of which has a core to find."""
    core_edges: int
    """Edges inside the core: the block the model says should be dense."""
    boundary_edges: int
    """Edges between core and periphery: the block the model allows."""
    periphery_edges: int
    """Edges inside the periphery: the block Figure 32.1 shows empty. What the search minimises,
    at a given core size."""
    restarts: int
    seed: int | None
    method: str
    note: str = ""

    @property
    def size(self) -> int:
        return len(self.core)


@dataclass(frozen=True)
class _Arena:
    """The arrays one discrete fit works on, built once and reused by every restart.

    A single-node flip changes the quality by a fixed amount (see :func:`_quality`), so the
    search never needs the adjacency matrix -- only each node's edges to the periphery. Holding
    the endpoints and the neighbour lists as arrays is what lets a whole sweep of candidate
    flips be scored in one ``numpy`` expression, which matters because the tension check of
    §32.2 refits this model on every null network.
    """

    nodes: list[Node]
    matrix: sp.csr_array
    """The binary adjacency, sparse, in ``nodes`` order (§8.1)."""
    source: np.ndarray
    """Endpoint indices of every edge, paired with ``target``."""
    target: np.ndarray
    degree: np.ndarray

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def m(self) -> int:
        return int(self.source.size)

    def neighbours(self, index: int) -> np.ndarray:
        matrix = self.matrix
        return np.asarray(matrix.indices[matrix.indptr[index] : matrix.indptr[index + 1]])

    def in_core(self, core: np.ndarray) -> np.ndarray:
        """How many of each node's neighbours are in ``core``, by one sparse product."""
        return np.asarray(self.matrix @ core.astype(np.int64)).ravel()


def _arena(graph: nx.Graph) -> _Arena:
    order = sorted(graph.nodes, key=str)
    matrix, nodes = adjacency(graph, nodes=order, weight=None, sparse=True)
    index = {node: position for position, node in enumerate(nodes)}
    pairs = np.array([(index[u], index[v]) for u, v in graph.edges()], dtype=np.int64).reshape(
        -1, 2
    )
    return _Arena(
        nodes=nodes,
        matrix=matrix,
        source=pairs[:, 0],
        target=pairs[:, 1],
        degree=np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64),
    )


def _quality(arena: _Arena, size: np.ndarray, periphery_edges: np.ndarray) -> np.ndarray:
    """:func:`_phi` for many candidate cores at once, from their sizes and periphery edges.

    The ideal pattern marks every pair with at least one core node, which is every pair except
    the periphery's own: ``ideal_pairs = P - (n-k)(n-k-1)/2``. The marked pairs that carry an
    edge are every edge except the periphery-periphery ones. So the quality of a candidate core
    is a function of two integers -- how big it is and how many edges it leaves inside the
    periphery -- and a whole sweep of single-node flips can be scored in one expression.
    ``nan`` wherever the correlation is undefined, which :func:`_phi` explains.
    """
    total = float(arena.n * (arena.n - 1) // 2)
    seen = float(arena.m)
    outside = arena.n - np.asarray(size, dtype=np.float64)
    marked = total - outside * (outside - 1.0) / 2.0
    hit = seen - np.asarray(periphery_edges, dtype=np.float64)
    spread = marked * (total - marked) * seen * (total - seen)
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = (total * hit - marked * seen) / np.sqrt(spread)
    return np.where(spread > 0, scores, np.nan)


def _counts(arena: _Arena, core: np.ndarray) -> tuple[int, int, int]:
    """(core-core, core-periphery, periphery-periphery) edge counts for one candidate core."""
    both = core[arena.source] & core[arena.target]
    neither = ~core[arena.source] & ~core[arena.target]
    inside, outside = int(both.sum()), int(neither.sum())
    return inside, arena.m - inside - outside, outside


def _ascend(arena: _Arena, core: np.ndarray, periphery_edges: int) -> tuple[np.ndarray, float]:
    """Steepest-ascent hill climbing: flip the one node that helps most, until none does.

    ``outside[u]`` is how many of ``u``'s neighbours are currently in the periphery, which is
    exactly how much the periphery-periphery edge count moves when ``u`` changes side, so every
    candidate flip is scored at once and the best one taken. Steepest ascent rather than
    first-improvement so that the climb is deterministic given its starting point: the only
    randomness in the search is where it starts from, which the seed fixes.
    """
    core = core.copy()
    outside = arena.degree - arena.in_core(core)
    size = int(core.sum())
    best = float(_quality(arena, np.array([size]), np.array([periphery_edges]))[0])
    for _ in range(MAX_SWEEPS):
        step = np.where(core, 1, -1)
        scores = _quality(arena, size - step, periphery_edges + step * outside)
        if np.all(np.isnan(scores)):
            break
        pick = int(np.nanargmax(scores))
        if not (scores[pick] > best):
            break
        best = float(scores[pick])
        periphery_edges += int(step[pick]) * int(outside[pick])
        size -= int(step[pick])
        outside[arena.neighbours(pick)] += int(step[pick])
        core[pick] = not core[pick]
    return core, best


def _seeded_start(arena: _Arena) -> tuple[np.ndarray, int, float]:
    """The best prefix of the degree ranking, with its periphery edge count and quality.

    Degree is what a core is made of -- "the core nodes are the one with a high degree of
    interconnectedness" (p. 451) -- so the ``k`` highest-degree nodes are the natural family of
    candidates, and all ``n - 1`` of them are scored at once: an edge leaves the periphery at
    the step its *later*-ranked endpoint joins the core, so one histogram over the edges gives
    the periphery edge count for every ``k``.
    """
    ranking = np.lexsort((np.arange(arena.n), -arena.degree))
    rank = np.empty(arena.n, dtype=np.int64)
    rank[ranking] = np.arange(arena.n)
    leaves = np.maximum(rank[arena.source], rank[arena.target])
    left = arena.m - np.cumsum(np.bincount(leaves, minlength=arena.n))
    sizes = np.arange(1, arena.n)
    scores = _quality(arena, sizes, left[:-1])
    if np.all(np.isnan(scores)):
        return np.zeros(arena.n, dtype=bool), arena.m, -math.inf
    pick = int(np.nanargmax(scores))
    core = np.zeros(arena.n, dtype=bool)
    core[ranking[: pick + 1]] = True
    return core, int(left[pick]), float(scores[pick])


def discrete_core(
    graph: nx.Graph, *, seed: int | None = None, restarts: int = RESTARTS
) -> DiscreteCore:
    """Fit the discrete core-periphery model of §32.1 and score it by correlation (p. 452).

    The ideal pattern is Figure 32.1: ones wherever either node is in the core, and "in the main
    diagonal in the peripheral area there are no entries larger than zero". The fit maximises
    the Pearson correlation between that pattern and the observed adjacency, which is the
    comparable form of the book's quality measure and the one that refuses the degenerate answer
    §32.2 warns about -- an all-core partition makes the pattern constant, and a constant has no
    correlation with anything.

    **The search.** The book does not name an algorithm: "without going into details of
    specialized algorithms, one can find the best D using classical randomization algorithms.
    Options are genetic algorithms, simulated annealing, or basin hopping" (p. 452). This is the
    cheap end of that list, stated plainly so a reader knows what they have: the ``k`` highest-
    degree nodes are scored for every ``k`` -- degree is what the core is made of, so the
    nested prefixes of the degree ranking are a good place to start and cost one pass -- and the
    best of those, plus ``restarts`` random subsets, are each improved by single-node hill
    climbing. The best correlation found wins. It is a local search: it is not guaranteed to
    find the global optimum, and a different ``seed`` may find a different core of equal quality.

    **What the correlation means.** 1.0 is the perfect discrete model, 0.0 is a core that
    explains the edges no better than the pair count alone, and it is ``nan`` when the network
    has no edges or when every pair is joined. It is *not* a p-value: a network with a broad
    degree distribution scores well here for free, because "if you create a network with a
    configuration model and you have a broad degree distribution, the high degree nodes have a
    high probability of connecting to each other" (p. 450). :func:`core_periphery_tension` is
    where that is scored against a degree-preserving null.
    """
    view, note = _simple_view(graph)
    nodes, edges = view.number_of_nodes(), view.number_of_edges()
    empty = DiscreteCore(
        core=[],
        periphery=sorted(view.nodes),
        correlation=math.nan,
        core_edges=0,
        boundary_edges=0,
        periphery_edges=edges,
        restarts=restarts,
        seed=seed,
        method="none: too few nodes or edges to fit a core",
        note=note,
    )
    if nodes < 3 or edges == 0 or edges == nodes * (nodes - 1) // 2:
        return empty
    rng = random.Random(seed)  # noqa: S311  # reproducibility, not secrecy
    arena = _arena(view)
    seeded, seeded_left, seeded_score = _seeded_start(arena)
    starts: list[tuple[np.ndarray, int]] = [(seeded, seeded_left)]
    for _ in range(max(restarts, 0)):
        drawn = np.zeros(nodes, dtype=bool)
        drawn[rng.sample(range(nodes), rng.randrange(1, nodes))] = True
        starts.append((drawn, _counts(arena, drawn)[2]))

    best, quality = seeded, seeded_score
    for start, start_left in starts:
        found, score = _ascend(arena, start, start_left)
        if score > quality and 0 < int(found.sum()) < nodes:
            best, quality = found, score
    if not int(best.sum()) or math.isinf(quality) or math.isnan(quality):
        return empty
    core_edges, boundary, periphery_edges = _counts(arena, best)
    core = [node for node, inside in zip(arena.nodes, best, strict=True) if inside]
    return DiscreteCore(
        core=core,
        periphery=[node for node, inside in zip(arena.nodes, best, strict=True) if not inside],
        correlation=quality,
        core_edges=core_edges,
        boundary_edges=boundary,
        periphery_edges=periphery_edges,
        restarts=restarts,
        seed=seed,
        method=(
            f"degree-seeded greedy, then single-node hill climbing from {max(restarts, 0)} "
            "random restarts (§32.1 leaves the search open)"
        ),
        note=note,
    )


# --------------------------------------------------------------------- §32.1 continuous model


@dataclass(frozen=True)
class ContinuousCoreness:
    """A coreness value per node, and how well ``c_u c_v`` reproduces the adjacency."""

    coreness: dict[Node, float]
    """One value per node, scaled so the largest is 1.0. Scaling is cosmetic: multiplying ``c``
    by a constant multiplies the ideal pattern by its square and leaves the correlation alone."""
    correlation: float
    """Pearson between the adjacency and ``c c^T`` off the diagonal. Comparable with
    :attr:`DiscreteCore.correlation`, which is the same coefficient over a 0/1 ideal, so the two
    say which of the two models of §32.1 fits this network better."""
    method: str
    kcore: dict[Node, int]
    """The k-core number of each node (:func:`graphrag.sna.measures.coreness`, §14.7): §32.1's
    "a priori approach", the one that "fixes your D, which may or may not fit your data well"."""
    shells: dict[int, int]
    """How many nodes sit in each k-shell (:func:`graphrag.sna.measures.core_shells`). Printed
    beside the agreement because a coreness is a floor rather than a rank: a network with four
    shells has four distinct values, and "the deepest shell" can be a third of the network."""
    kcore_correlation: float
    """The same correlation with ``D_uv = c_u c_v`` built from the normalised k-core numbers
    instead. Printed beside the fitted one so the two can be seen to disagree."""
    kcore_agreement: float
    """Jaccard between the nodes of the deepest k-shell and the discrete core, or ``nan`` when
    there is no discrete core to compare with. §32.1 keeps k-core out of this chapter because it
    "finds a fundamentally different type of core-periphery structure" (p. 450); this number is
    how much of that difference showed up here."""
    note: str = ""


def _coreness_correlation(matrix: np.ndarray, values: np.ndarray) -> float:
    """Correlation of the adjacency with the outer product of a coreness vector (§32.1)."""
    return _pattern_correlation(matrix, np.outer(values, values))


def continuous_coreness(graph: nx.Graph, core: Sequence[Node] | None = None) -> ContinuousCoreness:
    """Fit §32.1's continuous model: a coreness value per node, scored the same way (p. 453).

    The quality function does not change -- "the quality measure we want to optimize is still
    ``sum_uv A_uv D_uv``" -- only the ideal does: ``D_uv = c_u c_v``, so a node can be
    semi-peripheral instead of having to be one thing or the other. "If you force nodes to have
    a coreness of either 1 or 0, you're back in the discrete model."

    **Which ``c``.** The book leaves this open and lists three families: the k-core numbers as an
    a priori choice, Rombach et al.'s parametrised vector, and "a technique similar to the ones I
    mentioned before, to build your D via the c vector in such a way that your quality measure is
    maximized" (p. 453). This takes the last one in its closed form. With ``D_uv = c_u c_v`` the
    quality is ``c^T A c``, and §5.5's Rayleigh quotient says the unit-length vector that
    maximises it is the leading eigenvector of ``A``; its entries are non-negative because ``A``
    is. So the vector returned is not a proxy for the book's answer, it is the book's quality
    function solved exactly -- under the one constraint the book does not state, that ``c`` has
    unit length, without which the maximum is unbounded.

    Two consequences to read with it. On a disconnected network the leading eigenvector
    concentrates on the densest component and gives every other component a coreness near zero,
    which is a property of the eigenvector and not a finding about the corpus. And the vector is
    solved densely with ``numpy.linalg.eigh`` through :func:`graphrag.sna.matrices.eigenpairs`,
    for the reasons ``measures._eigenvector`` gives: the iterative solvers do not converge on the
    fragmented graphs a corpus produces, and they are not reproducible between runs.

    ``core`` is the discrete core, when one has been fitted, and is used only for
    :attr:`ContinuousCoreness.kcore_agreement`.

    **Not built:** Rombach et al.'s ``alpha``/``beta`` vector and the p-normalised ideals of
    Figure 32.3 (``D_uv = c_u + c_v``, ``D_uv = (c_u^p + c_v^p)^(1/p)``). The chapter presents
    them as "another freedom you can take" rather than as the model, and each is a second
    parametrisation to explain in every report that prints it.
    """
    view, note = _simple_view(graph)
    order = sorted(view.nodes, key=str)
    kcore = {node: int(value) for node, value in coreness(view).items()}
    shells = core_shells(view)
    if view.number_of_nodes() < 3 or view.number_of_edges() == 0:
        return ContinuousCoreness(
            coreness=dict.fromkeys(order, 0.0),
            correlation=math.nan,
            method="none: too few nodes or edges to fit a coreness vector",
            kcore={node: int(kcore.get(node, 0)) for node in order},
            shells=shells,
            kcore_correlation=math.nan,
            kcore_agreement=math.nan,
            note=note,
        )
    matrix, nodes = adjacency(view, nodes=order, weight=None)
    dense = np.asarray(matrix, dtype=np.float64)
    _, vectors = eigenpairs(dense, k=1, largest=True)
    principal = np.abs(vectors[:, 0])
    scaled = principal / (float(principal.max()) or 1.0)
    numbers = np.array([float(kcore.get(node, 0)) for node in nodes])
    deepest = max(kcore.values()) if kcore else 0
    top_shell = {node for node, number in kcore.items() if number == deepest}
    return ContinuousCoreness(
        coreness={node: float(value) for node, value in zip(nodes, scaled, strict=True)},
        correlation=_coreness_correlation(dense, scaled),
        method="leading eigenvector of the binary adjacency (§5.5's maximiser of c^T A c)",
        kcore={node: int(kcore.get(node, 0)) for node in nodes},
        shells=shells,
        kcore_correlation=_coreness_correlation(dense, numbers / (numbers.max() or 1.0)),
        kcore_agreement=_jaccard(top_shell, set(core)) if core is not None else math.nan,
        note=note,
    )


def _jaccard(left: set[Node], right: set[Node]) -> float:
    """Intersection over union, or ``nan`` when both sides are empty."""
    union = left | right
    return len(left & right) / len(union) if union else math.nan


# ------------------------------------------------------------------------- §32.1 the rich club


@dataclass(frozen=True)
class RichClubRow:
    """One point of the rich-club curve: the subgraph of nodes with degree above ``k``."""

    k: int
    nodes: int
    """``N_>k``: how many nodes have degree strictly greater than ``k``."""
    edges: int
    """``E_>k``: how many edges run between them."""
    phi: float
    """``2 E_>k / (N_>k (N_>k - 1))``: the density of the club, between 0 and 1."""
    null_mean: float
    null_std: float
    rho: float
    """``phi / mean(phi_null)``: the normalised coefficient. Above 1 means the club is denser
    than the degree sequence alone accounts for; ``nan`` when the null was not built or was
    empty at this ``k``."""


@dataclass(frozen=True)
class RichClub:
    """The rich-club curve and its normalisation against a degree-preserving null (§32.1)."""

    rows: list[RichClubRow]
    samples: int
    seed: int | None
    sustained: bool
    """Whether ``rho(k) > 1`` at every defined ``k`` in the top :data:`RICH_CLUB_TAIL` of the
    range. One ``k`` above 1 is noise; the claim is about the top of the degree range."""
    verdict: str
    note: str = ""


def _club_curve(graph: nx.Graph) -> list[tuple[int, int, int]]:
    """``(k, N_>k, E_>k)`` at every threshold where the club actually changes.

    Walks ``k`` upwards, peeling off the nodes whose degree has just been passed and subtracting
    the edges they take with them, so the whole curve costs one pass over the edges rather than
    one subgraph per ``k``. A ``k`` at which no node was peeled has the same club as ``k - 1``
    and is left out: a degree sequence with gaps in it would otherwise repeat a row for every
    integer in the gap, and the null would be compared against the same point several times.
    The curve stops where the club falls below :data:`MIN_RICH_CLUB_NODES`, since a density
    needs two nodes to be defined at all.
    """
    degrees = dict(graph.degree())
    if not degrees:
        return []
    remaining = set(graph.nodes)
    edges = graph.number_of_edges()
    by_degree: dict[int, list[Node]] = {}
    for node, degree in degrees.items():
        by_degree.setdefault(int(degree), []).append(node)
    curve: list[tuple[int, int, int]] = []
    for k in range(max(degrees.values())):
        peeled = by_degree.get(k, [])
        for node in peeled:
            edges -= sum(1 for other in graph.neighbors(node) if other in remaining)
            remaining.discard(node)
        if (peeled or not curve) and len(remaining) >= MIN_RICH_CLUB_NODES:
            curve.append((k, len(remaining), edges))
    return curve


def _phi_of(curve: Sequence[tuple[int, int, int]]) -> dict[int, float]:
    return {k: 2.0 * e / (n * (n - 1)) for k, n, e in curve}


def rich_club(
    graph: nx.Graph,
    *,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
    rewirings: Iterable[nx.Graph] | None = None,
) -> RichClub:
    """The rich-club coefficient at every degree threshold, normalised by a configuration null.

    §32.1 names the structure -- the core "sometimes dubbed as 'rich club'" (p. 450, citing Zhou
    and Mondragon 2004 and Colizza et al. 2006) -- and states in words what makes it a finding:
    "the surprising part is that the cores of some empirical networks are even denser than what
    you'd anticipate by looking at the degree distribution of the network". **What the book does
    not give is the formula**, so the definition here is those papers': ``phi(k) = 2 E_>k /
    (N_>k (N_>k - 1))``, the density among the nodes of degree greater than ``k``, and
    ``rho(k) = phi(k) / mean(phi_null(k))`` against degree-preserving rewirings of the same
    network (:func:`graphrag.sna.null.configuration`).

    The normalisation is the whole point and is the sentence above turned into a number. Raw
    ``phi(k)`` rises with ``k`` on *every* network, rich club or not, because high-degree nodes
    have many edges and few places to put them; only ``rho`` says whether they are denser than
    the degrees alone force. ``N_>k`` is identical in every null sample, since the null holds
    every degree fixed, so the two curves are compared threshold by threshold over the same node
    counts and only ``E_>k`` moves.

    **Reading it.** ``rho(k) > 1`` sustained over the high ``k`` is a rich club; a single ``k``
    above 1, especially the last one where the club is three nodes, is noise. ``sustained`` is
    that reading applied to the top :data:`RICH_CLUB_TAIL` of the ``k`` range, and the last rows
    of the curve are the least reliable whatever it says, because the denominator is a handful
    of pairs. The curve stops where the club falls below :data:`MIN_RICH_CLUB_NODES` nodes,
    below which the density is not defined at all.

    Degrees are unweighted here, as everywhere in this module: the club is the set of nodes with
    many distinct neighbours, not the set with the heaviest ties.

    ``rewirings`` is for the caller who has already drawn this null family, which
    :func:`core_periphery_report` has: §32.2's comparison needs the same rewirings, at the same
    count and from the same seed, so the report draws them once and spends them on both sections
    rather than paying §19.1's edge swaps twice. Passing them in skips the draw; passing ``None``
    draws ``samples`` of them here. Nothing is accumulated: each rewiring is measured and
    dropped, which is what keeps a fifty-sample null on a large network out of memory.
    """
    view, note = _simple_view(graph)
    curve = _club_curve(view)
    observed = _phi_of(curve)
    rng = random.Random(seed)  # noqa: S311  # reproducibility, not secrecy
    null: dict[int, list[float]] = {k: [] for k in observed}
    drawn = 0
    family = (
        configuration(view, max(samples, 0), seed=rng, weights=False)
        if rewirings is None
        else rewirings
    )
    for rewired in family:
        drawn += 1
        for k, value in _phi_of(_club_curve(rewired)).items():
            if k in null:
                null[k].append(value)
    rows = [
        RichClubRow(
            k=k,
            nodes=n,
            edges=e,
            phi=observed[k],
            null_mean=mean(null[k]) if null[k] else math.nan,
            null_std=std(null[k]) if null[k] else math.nan,
            rho=(observed[k] / mean(null[k]) if null[k] and mean(null[k]) > 0 else math.nan),
        )
        for k, n, e in curve
    ]
    high = rows[int(len(rows) * (1.0 - RICH_CLUB_TAIL)) :]
    tail = [row for row in high if not math.isnan(row.rho)]
    sustained = bool(tail) and all(row.rho > 1.0 for row in tail)
    return RichClub(
        rows=rows,
        samples=drawn,
        seed=seed,
        sustained=sustained,
        verdict=_club_verdict(rows, tail, sustained, drawn),
        note=note,
    )


def _club_verdict(
    rows: Sequence[RichClubRow], tail: Sequence[RichClubRow], sustained: bool, samples: int
) -> str:
    """What the curve says, in the terms §32.1 states the phenomenon in."""
    if not rows:
        return "No rich-club curve: the network has fewer than two nodes above any threshold."
    if samples == 0 or not tail:
        return (
            "Not testable: no degree-preserving rewiring of this network was possible, so phi(k) "
            "is a raw density and nothing here is normalised. A raw phi rises with k on every "
            "network (§32.1, p. 450); report the curve as a description and stop."
        )
    if sustained:
        return (
            f"Rich club: rho(k) stays above 1 at every one of the top {len(tail)} threshold(s), "
            "so the high-degree nodes are denser among themselves than their own degrees "
            "require -- §32.1's \"even denser than what you'd anticipate by looking at the "
            'degree distribution".'
        )
    return (
        f"No rich club: rho(k) falls to 1 or below somewhere in the top {len(tail)} "
        "threshold(s). The hubs do connect to each other, but no more than a network with this "
        "degree sequence does by chance (§32.1, p. 450)."
    )


# ---------------------------------------------------------- §32.2 the tension with communities


def community_correlation(graph: nx.Graph, partition: Sequence[Iterable[Node]]) -> float:
    """Correlation of the adjacency with the ideal *community* pattern (§32.2, Figure 32.4b).

    The community ideal is the other archetype of Figure 32.4: ``D_uv = 1`` when ``u`` and ``v``
    are in the same group and 0 otherwise -- blocks down the diagonal, "nothing here" everywhere
    else, which is the mirror image of Figure 32.1. Scoring it with the same coefficient as the
    core-periphery ideal is what makes the two comparable: one adjacency, two ideal patterns,
    one correlation each.

    ``nan`` when the pattern is constant, which is a partition with one group (every pair
    marked) or a partition into singletons (no pair marked). Nodes the partition does not cover
    are counted as singletons: a group has to be stated to mark a pair.
    """
    view, _ = _simple_view(graph)
    nodes, edges = view.number_of_nodes(), view.number_of_edges()
    membership: dict[Node, int] = {}
    for index, group in enumerate(partition):
        for node in group:
            if node in view:
                membership[node] = index
    sizes: dict[int, int] = {}
    for index in membership.values():
        sizes[index] = sizes.get(index, 0) + 1
    within = sum(size * (size - 1) // 2 for size in sizes.values())
    inside = sum(
        1
        for u, v in view.edges()
        if u in membership and v in membership and membership[u] == membership[v]
    )
    return _phi(
        pairs=nodes * (nodes - 1) // 2,
        edges=edges,
        ideal_pairs=within,
        ideal_edges=inside,
    )


@dataclass(frozen=True)
class Tension:
    """§32.2's comparison: which of the two mesoscale ideals explains these edges better."""

    core_correlation: float
    community_correlation: float
    core: Significance
    """The core-periphery correlation against the degree-preserving null, with the model refitted
    on every rewired network."""
    community: Significance
    """The same for the community correlation, with Louvain re-run on every rewired network."""
    communities: int
    core_size: int
    verdict: str
    """``core-periphery``, ``communities`` or ``not testable``: which ideal pattern of Figure
    32.4 the adjacency is closer to."""
    reading: str
    samples: int
    seed: int | None

    @property
    def core_excess(self) -> float:
        """How much better than its own null the core-periphery fit did. Not the verdict, and
        :func:`core_periphery_tension` says at length why: it answers the other question, which
        is whether this structure is more than the degree sequence."""
        return self.core_correlation - self.core.null_mean

    @property
    def community_excess(self) -> float:
        """The same for the community ideal. This one *is* the usual modularity question in
        another currency: whether the groups are more than the degrees force."""
        return self.community_correlation - self.community.null_mean


def core_periphery_tension(
    graph: nx.Graph,
    partition: Sequence[Iterable[Node]],
    *,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
    resolution: float = 1.0,
    restarts: int = RESTARTS,
    scores: tuple[Sequence[float], Sequence[float]] | None = None,
) -> Tension:
    """Fit both mesoscale ideals to the same adjacency and say which one the edges support.

    §32.2 states the problem: "in CP there isn't space for communities, given that there's only
    one dense area and everything connects to it. In CD, there's little space for peripheries,
    and there are multiple cores." A partition handed to a core-periphery network will still
    come back with communities, because the periphery has to be cut somewhere, and its
    modularity will not look absurd. So the report asks the question the chapter asks: which
    ideal pattern of Figure 32.4 is closer to the observed adjacency.

    **The comparison, stated.** One adjacency; two ideal patterns over the same pairs -- the
    core-periphery one from :func:`discrete_core` (Figure 32.1), the community one from
    :func:`community_correlation` (Figure 32.4b); one Pearson correlation each, so the two
    numbers are the same quantity on the same data and may be compared directly. The verdict is
    that comparison and nothing else: ``core-periphery`` means the core-periphery ideal
    reproduces the adjacency at least as well as the partition does, and then the communities
    are the periphery's tail -- slices of the one region the core is not in -- and naming them
    as groups says more than the network does.

    **Why the verdict is on the raw coefficients, and what the null is still for.** Both
    coefficients are also scored against the same degree-preserving null
    (:func:`graphrag.sna.null.configuration`), each rewired network having its core refitted and
    Louvain re-run on it, so each model is compared with the best that model could have done on
    a network with these degrees. Those two excesses are printed, and they are deliberately
    *not* the verdict. The reason is §32.1's own observation (p. 450): with a broad degree
    distribution "the high degree nodes have a high probability of connecting to each other", so
    the degree-preserving null of a core-periphery network is itself core-periphery and the
    core's excess over it is near zero for exactly the networks this chapter is about. A verdict
    on the excess could therefore never return "core-periphery", which would be a test that
    cannot fail in one direction. The excesses answer the other question, and the report prints
    them for it: whether either structure is more than the degree sequence. A core with no
    excess is still a core; it is just not a surprise, and :func:`rich_club` is where that
    surprise is measured properly.

    A third thing the null does here is fail informatively. A dense core leaves an edge swap
    almost nowhere to land, so a strongly core-periphery network can have *no* degree-preserving
    null available inside the sampler's budget (§19.1, :data:`graphrag.sna.null.TRIES_PER_EDGE`);
    the sample count then reads 0, the excesses are not printed as if they were measurements,
    and the reading says so.

    **Cost.** One Louvain run and one core fit per sample, which is the same order as
    :func:`graphrag.sna.cluster.null_model_modularity` and is why the null count is the
    report's ``--samples`` rather than something larger. The core is refitted on the rewired
    networks with the same search and the same ``restarts`` as the observation, so neither side
    of the comparison is handicapped. ``scores`` -- the two null distributions, core first --
    is how :func:`core_periphery_report` avoids drawing a second family of rewirings for the
    rich club, exactly as ``null_model_modularity`` takes its own.
    """
    view, _ = _simple_view(graph)
    fitted = discrete_core(view, seed=seed, restarts=restarts)
    observed_core = fitted.correlation
    observed_community = community_correlation(view, partition)
    if scores is not None:
        core_null, community_null = list(scores[0]), list(scores[1])
    else:
        rng = random.Random(seed)  # noqa: S311  # reproducibility, not secrecy
        core_null, community_null = [], []
        for rewired in configuration(view, max(samples, 0), seed=rng, weights=False):
            _score_rewiring(
                rewired,
                resolution=resolution,
                restarts=restarts,
                rng=rng,
                core=core_null,
                community=community_null,
            )
    core_result = significance(
        observed_core, [v for v in core_null if not math.isnan(v)], null="configuration"
    )
    community_result = significance(
        observed_community,
        [v for v in community_null if not math.isnan(v)],
        null="configuration",
    )
    groups = len([group for group in partition if list(group)])
    verdict = _tension_verdict(observed_core, observed_community)
    return Tension(
        core_correlation=observed_core,
        community_correlation=observed_community,
        core=core_result,
        community=community_result,
        communities=groups,
        core_size=fitted.size,
        verdict=verdict,
        reading=(
            _tension_reading(verdict, fitted.size, groups)
            + " "
            + _null_sentence(observed_core, observed_community, core_result, community_result)
        ),
        samples=min(core_result.samples, community_result.samples),
        seed=seed,
    )


def _score_rewiring(
    rewired: nx.Graph,
    *,
    resolution: float,
    restarts: int,
    rng: random.Random,
    core: list[float],
    community: list[float],
) -> None:
    """Refit both models of Figure 32.4 on one rewired network, appending both correlations.

    Refitting rather than carrying the observed core and the observed partition across: a
    partition fitted to one graph and scored on another always transfers badly, which would make
    every network look structured. :func:`graphrag.sna.cluster.null_model_modularity` says the
    same thing at length and for the same reason.
    """
    core.append(
        discrete_core(rewired, seed=rng.randrange(1_000_000), restarts=restarts).correlation
    )
    found = nx.community.louvain_communities(
        rewired, weight=None, resolution=resolution, seed=rng.randrange(1_000_000)
    )
    community.append(community_correlation(rewired, [sorted(c, key=str) for c in found]))


def _tension_verdict(core: float, community: float) -> str:
    """Which of the two ideals of Figure 32.4 fits the adjacency better (§32.2).

    "At least as well" is the book's asymmetry and this keeps it: a tie goes to core-periphery,
    because a partition of a network that a single core explains equally well has told you
    nothing the core did not.
    """
    if math.isnan(core) and math.isnan(community):
        return "not testable"
    if math.isnan(community):
        return "core-periphery"
    if math.isnan(core):
        return "communities"
    return "core-periphery" if core >= community else "communities"


def _tension_reading(verdict: str, core_size: int, groups: int) -> str:
    """The sentence §32.2 asks the report to print beside the two coefficients."""
    if verdict == "core-periphery":
        return (
            f"This network reads as core-periphery: a core of {core_size} node(s) reproduces the "
            f"adjacency at least as well as the {groups} communities do. §32.2's warning "
            "applies -- there is one dense area and everything connects to it, so a partition "
            "has to cut the periphery somewhere and the communities below are that cut, not "
            "groups the corpus put there. Read the core membership and the periphery's size; do "
            "not name the communities."
        )
    if verdict == "communities":
        return (
            f"This network reads as communities: the {groups} groups reproduce the adjacency "
            "better than any single core does. §32.2's continuum still applies -- \"the blend is "
            'always different" -- so a core-periphery correlation above zero is expected here '
            "and is not a contradiction."
        )
    return (
        "Neither model could be scored: both ideal patterns are constant on this network, which "
        "has no edges, or no pair without one (§32.2). Report the coefficients as a description "
        "and draw no verdict."
    )


def _null_sentence(
    core: float, community: float, core_null: Significance, community_null: Significance
) -> str:
    """What the two degree-preserving nulls add to the verdict, which is a different question."""
    if not (core_null.testable and community_null.testable):
        return (
            "No degree-preserving rewiring of this network was possible inside the sampler's "
            "budget, so neither coefficient is normalised: the verdict above is a comparison of "
            "fits and not a significance test. That failure is itself informative -- a dense "
            "core leaves an edge swap almost nowhere to land -- but it means nothing here has "
            "been scored against chance (§19.1)."
        )
    core_beats = not math.isnan(core) and core > core_null.null_mean
    community_beats = not math.isnan(community) and community > community_null.null_mean
    tail = (
        "Neither is more than the degree sequence already implies."
        if not core_beats and not community_beats
        else (
            "The core is no more than the degree sequence implies, which §32.1 (p. 450) says is "
            "the usual case: hubs connect to hubs by chance."
            if not core_beats
            else "The communities are no more than the degree sequence implies."
            if not community_beats
            else "Both beat that null."
        )
    )
    return (
        f"Against {core_null.samples} degree-preserving rewiring(s), with each model refitted on "
        f"each of them: the core fit is {_num(core - core_null.null_mean)} above its null and "
        f"the community fit {_num(community - community_null.null_mean)} above its own. {tail}"
    )


# ------------------------------------------------------------------------- §32.4 nestedness


@dataclass(frozen=True)
class Nestedness:
    """NODF on a two-mode network, with the fixed-fixed null that says whether it is a finding."""

    nodf: float
    """Nestedness metric based on Overlap and Decreasing Fill, 0 to 100. 100 is the perfectly
    nested matrix of Figure 32.9, where every smaller row is a subset of every larger one."""
    row_nodf: float
    column_nodf: float
    rows: int
    columns: int
    row_mode: str
    column_mode: str
    fill: float
    """Share of cells that hold a 1. NODF is not independent of it, which is why the null holds
    both margins fixed."""
    discrepancy: int
    """§32.4's "how many mistakes you made": the number of ones lying outside the packed shape
    the non-parametric isocline draws. 0 for a perfectly nested matrix."""
    significance: Significance | None
    samples: int
    seed: int | None
    note: str = ""


def two_mode_sides(graph: nx.Graph) -> tuple[str, str, list[Node], list[Node]] | None:
    """The two modes of a two-mode network and their nodes, or ``None`` if it has only one.

    The modes are read off the nodes, never inferred: :func:`graphrag.sna.export.describe` gives
    the reason, which is that plenty of one-mode networks happen to be 2-colourable without that
    meaning the nodes are of two *kinds* (§6.4). ``mode`` is what this package's builders write,
    ``bipartite`` what ``networkx``'s own two-mode graphs carry; the two mode names are sorted so
    the rows of the matrix are a function of the data alone, and both are returned so the report
    can say which side is which.

    ``None`` when there are not exactly two modes, when a node carries no mode at all, or when
    some edge does not cross between them -- a graph with a within-mode edge is not the two-mode
    network §6.4 defines, and no incidence matrix represents it.
    """
    for key in MODE_KEYS:
        labels = {node: str(data[key]) for node, data in graph.nodes(data=True) if key in data}
        if len(labels) != graph.number_of_nodes() or len(set(labels.values())) != 2:
            continue
        first, second = sorted(set(labels.values()))
        if any(labels[u] == labels[v] for u, v in graph.edges()):
            continue
        left = sorted((n for n, mode in labels.items() if mode == first), key=str)
        right = sorted((n for n, mode in labels.items() if mode == second), key=str)
        return first, second, left, right
    return None


def _nodf(matrix: np.ndarray) -> tuple[float, float, float]:
    """(NODF, row NODF, column NODF) of a binary matrix (Almeida-Neto et al. 2008).

    For each pair of rows: if the lower total is not strictly smaller there is no decreasing
    fill and the pair contributes 0; otherwise it contributes the percentage of the smaller
    row's ones that fall in columns the larger row also has ("paired overlap"). Columns the same
    way. NODF is the mean over all row pairs and all column pairs together, which is the
    published normalisation -- the sum of the paired scores over ``(r(r-1) + c(c-1))/2``.
    """
    rows, row_pairs = _paired_overlap(matrix)
    columns, column_pairs = _paired_overlap(matrix.T)
    pairs = row_pairs + column_pairs
    return (
        (rows + columns) / pairs if pairs else math.nan,
        rows / row_pairs if row_pairs else math.nan,
        columns / column_pairs if column_pairs else math.nan,
    )


def _paired_overlap(matrix: np.ndarray) -> tuple[float, int]:
    """The total paired-overlap score over the rows of ``matrix``, and how many pairs it covers.

    Written as one matrix product rather than a loop over pairs: ``matrix @ matrix.T`` holds
    every shared-column count at once, and decreasing fill is a comparison of the row totals
    against each other. A pair with equal totals scores zero, which is why a checkerboard -- and
    any matrix with a flat margin -- has no nestedness at all.
    """
    if matrix.shape[0] < 2:
        return 0.0, 0
    totals = matrix.sum(axis=1)
    shared = matrix @ matrix.T
    smaller = np.minimum(totals[:, None], totals[None, :])
    decreasing = totals[:, None] != totals[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(decreasing & (smaller > 0), 100.0 * shared / np.maximum(smaller, 1), 0.0)
    rows = matrix.shape[0]
    return float(np.triu(scores, k=1).sum()), rows * (rows - 1) // 2


def _discrepancy(matrix: np.ndarray) -> int:
    """§32.4's count of mistakes against the non-parametric isocline (p. 459).

    "If your row (or column) sums to 50, then you expect a perfectly nested matrix to have 50
    ones followed only by zeros. So your isocline should pass through that point." The packed
    matrix is exactly that, once the columns are sorted by their totals; this counts the ones
    that fall outside it, which is Brualdi and Sanderson's discrepancy. Zero for a perfectly
    nested matrix and large for a random one.
    """
    if matrix.size == 0:
        return 0
    columns = np.argsort(-matrix.sum(axis=0), kind="stable")
    sorted_matrix = matrix[:, columns]
    index = np.arange(sorted_matrix.shape[1])
    packed = index[None, :] < sorted_matrix.sum(axis=1, keepdims=True)
    return int(np.sum((sorted_matrix > 0) & ~packed))


def nestedness(
    graph: nx.Graph, *, samples: int = NULL_SAMPLES, seed: int | None = None
) -> Nestedness:
    """NODF on a two-mode network, against the fixed-fixed null (§32.4).

    "A nested system is one where the elements containing few items only contain a subset of the
    items of elements with more items" (p. 458): the second island's species are a subset of the
    first's, the less diversified economy exports a subset of what the more diversified one
    does, and the matrix sorted by row and column totals is "upper-triangular" (Figure 32.9).
    §32.4 calls this the two-mode form of the same structure the rest of the chapter is about --
    "core-periphery structures are a generalization of ... nestedness" -- so it is fitted on the
    two-mode network rather than on a projection of it, which would destroy exactly the
    asymmetry being measured.

    **Which measure.** The chapter describes *temperature* (pp. 459-460): fit an isocline,
    count the ones on the wrong side of it, 0 degrees for perfectly nested and 100 for random.
    What it does not give is a formula, and the normalisation Atmar and Patterson's temperature
    needs -- the distance of each misplaced cell from the isocline, divided by what the maximally
    unexpected matrix would give -- is a second set of conventions to explain in a report. So
    the number here is **NODF** (Almeida-Neto, Guimaraes, Guimaraes, Loyola and Ulrich, "A
    consistent metric for nestedness analysis in ecological systems: reconciling concept and
    measurement", *Oikos* 117:1227-1239, 2008), the measure that replaced temperature in the
    field the chapter takes the idea from: 100 when every smaller row is a subset of every larger
    one, 0 when no row is. The chapter's own procedure is reported beside it as
    :attr:`Nestedness.discrepancy` -- the count of ones outside the non-parametric isocline,
    which is the "how many mistakes you made" of p. 459 and needs no convention at all.

    **The null.** :func:`graphrag.sna.null.bipartite_preserving`, the curveball algorithm, which
    holds **both** margins fixed: every row total and every column total of the incidence matrix.
    That is the null this measure needs and the one §32.4's own caveat demands -- other authors
    "suggest that nestedness could arise simply from the degree distribution ... and that there
    are fewer nested system than we originally thought" (p. 458). A high NODF against a
    fixed-fixed null is a finding; a high NODF on its own is mostly a statement about how
    unequal the row totals are. (The sampler returns a *projection* of each rewired membership
    table, and this reads the memberships back off it with
    :func:`graphrag.sna.null.pairs_of`; the projection itself is not used.)

    Raises ``ValueError`` on a one-mode network, which has no incidence matrix and therefore no
    nestedness: the answer there is the rest of this module. And on a network with more than
    :data:`MAX_NESTEDNESS_NODES` on a side, because every paired overlap of that side is
    computed at once and the matrix of them would not fit; take a component or raise
    ``--min-weight`` first, as :func:`graphrag.sna.null.ergm` says for the same reason.
    """
    sides = two_mode_sides(graph)
    if sides is None:
        msg = (
            "nestedness() needs a two-mode network: NODF is defined on the incidence matrix of "
            "two kinds of node (§32.4, and §6.4 on what makes a network two-mode), and this "
            "graph has one kind, or an edge that does not cross between two. Build the "
            "speakers-entities network without --project, or ask the one-mode question with "
            "discrete_core()/continuous_coreness() instead."
        )
        raise ValueError(msg)
    row_mode, column_mode, rows, columns = sides
    side = max(len(rows), len(columns))
    if side > MAX_NESTEDNESS_NODES:
        msg = (
            f"nestedness() would compare every pair of {side:,} nodes on one side, which is "
            f"more than MAX_NESTEDNESS_NODES ({MAX_NESTEDNESS_NODES:,}) and does not fit in "
            "memory as a matrix. Raise --min-weight, filter the network, or take one component "
            "first, and say in the report which matrix the NODF describes."
        )
        raise ValueError(msg)
    left = set(rows)
    pairs = [(u, v) if u in left else (v, u) for u, v in graph.edges()]
    # Isolated nodes have no memberships, so they are not rows or columns of the matrix at all:
    # ``incidence`` refuses an axis naming a node that appears in no pair, and an all-zero row
    # would in any case contribute nothing but a denominator.
    seen_rows = {u for u, _ in pairs}
    seen_columns = {v for _, v in pairs}
    matrix, row_order, column_order = incidence(
        pairs,
        left=[node for node in rows if node in seen_rows],
        right=[node for node in columns if node in seen_columns],
    )
    dense = np.asarray(matrix, dtype=np.float64)
    nodf, row_nodf, column_nodf = _nodf(dense)
    rng = random.Random(seed)  # noqa: S311  # reproducibility, not secrecy
    null: list[float] = []
    for sample in bipartite_preserving(pairs, max(samples, 0), seed=rng):
        drawn = pairs_of(sample)
        if not drawn:
            continue
        rewired, _, _ = incidence(drawn, left=row_order, right=column_order)
        null.append(_nodf(np.asarray(rewired, dtype=np.float64))[0])
    return Nestedness(
        nodf=nodf,
        row_nodf=row_nodf,
        column_nodf=column_nodf,
        rows=len(row_order),
        columns=len(column_order),
        row_mode=row_mode,
        column_mode=column_mode,
        fill=float(dense.mean()) if dense.size else math.nan,
        discrepancy=_discrepancy(dense),
        significance=(significance(nodf, null, null="bipartite_preserving") if null else None),
        samples=len(null),
        seed=seed,
    )


# ----------------------------------------------------------------------------- the report


@dataclass
class CorePeripheryReport:
    """Chapter 32 over one network: both models, the rich club, the tension, the nestedness."""

    frame: str
    nodes: int
    edges: int
    discrete: DiscreteCore
    continuous: ContinuousCoreness
    club: RichClub
    tension: Tension | None = None
    nested: Nestedness | None = None
    samples: int = NULL_SAMPLES
    seed: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def better_model(self) -> str:
        """Which of §32.1's two models fits this network's edges better, by correlation."""
        discrete, continuous = self.discrete.correlation, self.continuous.correlation
        if math.isnan(discrete) and math.isnan(continuous):
            return "neither"
        if math.isnan(continuous) or (not math.isnan(discrete) and discrete > continuous):
            return "discrete"
        return "continuous"


def core_periphery_report(
    graph: nx.Graph,
    partition: Sequence[Iterable[Node]] | None = None,
    *,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
    resolution: float = 1.0,
    restarts: int = RESTARTS,
) -> CorePeripheryReport:
    """Everything chapter 32 has to say about one network, in one object.

    ``partition`` is the grouping the rest of the report found. Given one, §32.2's tension check
    runs and the report can say whether those groups are the periphery's tail; without one, the
    two models and the rich club are still fitted, since they are properties of the network and
    not of any method.

    The nestedness section appears only on a two-mode network (§32.4) and the note says why it
    is absent otherwise. On a two-mode network the *one*-mode numbers are still computed and are
    still readable, but with a caveat this adds to the report: a bipartite network has no
    core-core block at all, so the discrete model can only ever describe one side's hubs.

    **One family of rewirings, not two.** The rich club and §32.2's comparison want the same
    degree-preserving null, at the same count and from the same seed, and drawing it is the
    expensive half. So the family is drawn once here: each rewiring is scored for the comparison
    on its way past and then handed to the rich club, which measures it and drops it. Only the
    two score lists are kept, for the reason ``analysis._shared_rewirings`` gives -- holding
    fifty rewirings of a large network in a list costs about a gigabyte.
    """
    view, flattened = _simple_view(graph)
    notes = [flattened] if flattened else []
    discrete = discrete_core(view, seed=seed, restarts=restarts)
    continuous = continuous_coreness(view, discrete.core)
    core_scores: list[float] = []
    community_scores: list[float] = []
    club = rich_club(
        view,
        samples=samples,
        seed=seed,
        rewirings=_scored_rewirings(
            view,
            partition,
            samples=samples,
            seed=seed,
            resolution=resolution,
            restarts=restarts,
            core=core_scores,
            community=community_scores,
        ),
    )
    sides = two_mode_sides(view)
    too_wide = sides is not None and max(len(sides[2]), len(sides[3])) > MAX_NESTEDNESS_NODES
    if too_wide and sides is not None:
        notes.append(
            f"The nestedness section is not here: one side of this two-mode network has "
            f"{max(len(sides[2]), len(sides[3])):,} nodes, above MAX_NESTEDNESS_NODES "
            f"({MAX_NESTEDNESS_NODES:,}), and NODF compares every pair of them. Filter the "
            "network or raise --min-weight and ask for it again (§32.4)."
        )
    if sides is not None:
        notes.append(
            f"This network is two-mode (mode `{sides[0]}` against mode `{sides[1]}`), so every "
            "periphery-periphery pair the discrete model penalises includes every pair within a "
            "mode, which no two-mode network can ever join. Read the Nestedness section, which "
            "§32.4 gives as the two-mode form of this structure, and read the core above as "
            '"the hubs of both modes" rather than as Figure 32.1.'
        )
    if any(data.get("weight") is not None for _, _, data in view.edges(data=True)):
        notes.append(
            "The edges carry weights and nothing here reads them: the ideal patterns of "
            "chapter 32 hold ones and zeros, so the correlations are against the binary "
            "adjacency and say whether pairs are connected, not how heavily."
        )
    return CorePeripheryReport(
        frame=str(graph.graph.get("frame", "")),
        nodes=view.number_of_nodes(),
        edges=view.number_of_edges(),
        discrete=discrete,
        continuous=continuous,
        club=club,
        tension=(
            core_periphery_tension(
                view,
                partition,
                samples=samples,
                seed=seed,
                resolution=resolution,
                restarts=restarts,
                scores=(core_scores, community_scores),
            )
            if partition
            else None
        ),
        nested=(
            nestedness(view, samples=samples, seed=seed)
            if sides is not None and not too_wide
            else None
        ),
        samples=samples,
        seed=seed,
        notes=notes,
    )


def _scored_rewirings(
    view: nx.Graph,
    partition: Sequence[Iterable[Node]] | None,
    *,
    samples: int,
    seed: int | None,
    resolution: float,
    restarts: int,
    core: list[float],
    community: list[float],
) -> Iterator[nx.Graph]:
    """One family of degree-preserving rewirings, scored for §32.2 on the way past to the club.

    The seeding order is :func:`core_periphery_tension`'s own -- draw a sample, refit the core
    from the same stream, then take a Louvain seed from it -- so the comparison comes out with
    the numbers it would have had if it had drawn the family itself. When there is no partition
    there is no comparison to score, and this is only the draw.
    """
    rng = random.Random(seed)  # noqa: S311  # reproducibility, not secrecy
    for rewired in configuration(view, max(samples, 0), seed=rng, weights=False):
        if partition is not None:
            _score_rewiring(
                rewired,
                resolution=resolution,
                restarts=restarts,
                rng=rng,
                core=core,
                community=community,
            )
        yield rewired


def _num(value: float, places: int = 4) -> str:
    if math.isnan(value):
        return "undefined"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _label(graph: nx.Graph, node: Node) -> str:
    data = graph.nodes.get(node, {})
    return str(data.get("label") or data.get("name") or node)


def _club_rows(club: RichClub) -> list[RichClubRow]:
    """At most :data:`RICH_CLUB_ROWS` points of the curve, evenly spaced and keeping the ends."""
    if len(club.rows) <= RICH_CLUB_ROWS:
        return list(club.rows)
    picks = np.linspace(0, len(club.rows) - 1, RICH_CLUB_ROWS).round().astype(int)
    return [club.rows[index] for index in sorted(set(picks.tolist()))]


def render_core_periphery(report: CorePeripheryReport, graph: nx.Graph | None = None) -> list[str]:
    """The core-periphery section as markdown lines, frame and null beside every number.

    ``graph`` is only used to print node labels instead of ids; without it the ids are printed.
    """
    labels = graph if graph is not None else nx.Graph()
    core = report.discrete
    lines = [
        "## Core-periphery",
        "",
        f"**Sampling frame.** {report.frame} Read here as the binary, undirected, self-loop-free "
        "adjacency the chapter's matrix figures are drawn on.",
        "",
        f"**n.** {report.nodes:,} node(s), {report.edges:,} edge(s), "
        f"{report.nodes * (report.nodes - 1) // 2:,} pair(s) scored.",
        "",
        f"**Null model.** `configuration`, which holds fixed {HOLDS_FIXED['configuration']}. "
        f"{report.club.samples} rewiring(s) for the rich club"
        + (
            f" and {report.tension.samples} for the model comparison."
            if report.tension is not None
            else "; the model comparison did not run, because no grouping was passed to it."
        ),
        "",
        "**Implements.** §32.1 (the discrete and the continuous model, and the rich club), "
        "§32.2 (the tension with communities)"
        + (", §32.4 (nestedness)" if report.nested is not None else "")
        + ".",
        "",
        "### The two models",
        "",
    ]
    lines += _table(
        ["model", "ideal pattern", "correlation with A"],
        [
            [
                f"discrete, core of {core.size:,}",
                "1 where either node is in the core (Figure 32.1)",
                _num(core.correlation),
            ],
            [
                "continuous",
                "c_u c_v, c the leading eigenvector (§32.1, p. 453)",
                _num(report.continuous.correlation),
            ],
            [
                "continuous, k-core",
                "c_u c_v, c the normalised k-core number (a priori)",
                _num(report.continuous.kcore_correlation),
            ],
        ],
    )
    lines += [
        f"- core edges {core.core_edges:,}, core-periphery edges {core.boundary_edges:,}, "
        f"periphery-periphery edges {core.periphery_edges:,} (Figure 32.1 has none of the last).",
        f"- search: {core.method}; seed {core.seed}.",
        f"- the {report.better_model} model fits this network's edges better.",
        "",
    ]
    if core.core:
        shown = sorted(core.core, key=lambda n: (-labels.degree(n) if n in labels else 0, str(n)))
        names = ", ".join(_label(labels, node) for node in shown[:20])
        lines += [
            f"**The core.** {names}"
            + (f", +{len(core.core) - 20} more" if len(core.core) > 20 else ""),
            "",
        ]
    lines += [
        "The k-core row is the chapter's *a priori* option and is printed because it disagrees: "
        '§32.1 keeps k-core centrality out of this chapter because it "finds a fundamentally '
        "different type of core-periphery structure, which is more similar to a hierarchical "
        'decomposition of the network" (p. 450). Jaccard between the deepest k-shell and the '
        f"core above: {_num(report.continuous.kcore_agreement)}. Shells (core number: nodes): "
        + ", ".join(f"{k}: {n:,}" for k, n in report.continuous.shells.items())
        + ".",
        "",
    ]
    lines += _render_club(report.club)
    if report.tension is not None:
        lines += _render_tension(report.tension)
    for note in report.notes:
        lines += [f"> {note}", ""]
    if report.nested is not None:
        lines += render_nestedness(report.nested)
    return lines


def _render_club(club: RichClub) -> list[str]:
    lines = [
        "### Rich club",
        "",
        "phi(k) is the density among the nodes of degree above k; rho(k) is phi(k) divided by "
        "its mean over the degree-preserving rewirings above. Raw phi rises with k on every "
        "network, so rho is the column to read (§32.1, p. 450).",
        "",
    ]
    if not club.rows:
        return [*lines, club.verdict, ""]
    lines += _table(
        ["k", "N>k", "E>k", "phi(k)", "null mean", "rho(k)"],
        [
            [
                f"{row.k:,}",
                f"{row.nodes:,}",
                f"{row.edges:,}",
                _num(row.phi),
                _num(row.null_mean),
                _num(row.rho, 2),
            ]
            for row in _club_rows(club)
        ],
    )
    if len(club.rows) > RICH_CLUB_ROWS:
        lines += [
            f"{RICH_CLUB_ROWS} of {len(club.rows)} thresholds shown, evenly spaced; the payload "
            "carries the whole curve.",
            "",
        ]
    return [*lines, club.verdict, ""]


def _render_tension(tension: Tension) -> list[str]:
    """§32.2's comparison, with both nulls named beside both coefficients."""
    lines = [
        "### Core-periphery against communities",
        "",
        "**Implements.** §32.2. One adjacency, two ideal patterns (Figure 32.4), the same "
        "correlation each, and each against a null in which its own model was refitted on a "
        "degree-preserving rewiring.",
        "",
    ]
    lines += _table(
        ["model", "correlation", "null mean", "excess", "z", "samples"],
        [
            [
                f"core-periphery (core of {tension.core_size:,})",
                _num(tension.core_correlation),
                *_null_columns(tension.core, tension.core_excess),
            ],
            [
                f"communities ({tension.communities:,})",
                _num(tension.community_correlation),
                *_null_columns(tension.community, tension.community_excess),
            ],
        ],
    )
    lines += [tension.reading, ""]
    # Only the caveats of a null that ran: the "no null sample" one is already covered, and more
    # precisely, by the last sentence of ``reading``, which knows it was the swap budget rather
    # than the network's size that ran out. One copy, not two, when both sides say the same.
    spoken = [r.caveat for r in (tension.core, tension.community) if r.testable and r.caveat]
    for caveat in dict.fromkeys(spoken):
        lines += [f"> {caveat}", ""]
    return lines


def _null_columns(result: Significance, excess: float) -> list[str]:
    """The null half of a tension row, dashed out when no null network could be built.

    A z-score of 0.00 beside a sample count of 0 is the one thing this section must not print:
    it reads as "exactly average" when it means "never measured" (§19.1).
    """
    if not result.testable:
        return ["-", "-", "-", "0"]
    return [_num(result.null_mean), _num(excess), _num(result.z, 2), f"{result.samples:,}"]


def render_nestedness(nested: Nestedness) -> list[str]:
    """§32.4's section: NODF, the isocline's own count of mistakes, and the fixed-fixed null."""
    lines = [
        "## Nestedness",
        "",
        f"**Sampling frame.** The incidence matrix of the two-mode network: {nested.rows:,} "
        f"node(s) of mode `{nested.row_mode}` on the rows, {nested.columns:,} of mode "
        f"`{nested.column_mode}` on the columns, {_num(100.0 * nested.fill, 2)}% of cells "
        "filled. Nodes with no membership at all are not rows or columns of a matrix and are "
        "not counted here.",
        "",
        f"**n.** {nested.rows * nested.columns:,} cell(s); "
        f"{round(nested.fill * nested.rows * nested.columns):,} membership(s).",
        "",
        f"**Null model.** `bipartite_preserving`, which holds fixed "
        f"{HOLDS_FIXED['bipartite_preserving']} -- the fixed-fixed null, both margins. "
        f"{nested.samples} sample(s).",
        "",
        "**Implements.** §32.4 (nestedness as the two-mode core-periphery structure), with NODF "
        "(Almeida-Neto et al. 2008) in place of the chapter's temperature.",
        "",
    ]
    lines += _table(
        ["measure", "value"],
        [
            ["NODF", _num(nested.nodf, 2)],
            [f"NODF over the rows (`{nested.row_mode}`)", _num(nested.row_nodf, 2)],
            [f"NODF over the columns (`{nested.column_mode}`)", _num(nested.column_nodf, 2)],
            ["ones outside the isocline (§32.4, p. 459)", f"{nested.discrepancy:,}"],
        ],
    )
    result = nested.significance
    if result is None or not result.testable:
        lines += [
            "No fixed-fixed null could be built for this matrix, so the NODF above is a "
            "description. §32.4's own caveat is that nestedness "
            '"could arise simply from the degree distribution", which is exactly what this '
            "null would have held fixed; do not call it a finding.",
            "",
        ]
        return lines
    lines += [
        f"- observed {_num(nested.nodf, 2)}; null mean {_num(result.null_mean, 2)}, "
        f"sd {_num(result.null_std, 2)}",
        f"- z {_num(result.z, 2)}; empirical p {_num(result.p_value)}",
        "",
        _nestedness_reading(nested, result),
        "",
    ]
    if result.caveat:
        lines += [f"> {result.caveat}", ""]
    return lines


def _nestedness_reading(nested: Nestedness, result: Significance) -> str:
    """What NODF and its null say together, which is more than either says alone."""
    if result.z >= 2.0:
        return (
            f"Nested beyond the margins: NODF {_num(nested.nodf, 2)} sits {_num(result.z, 2)} "
            "null standard deviations above what matrices with these row and column totals "
            "produce. The smaller rows hold subsets of the larger ones more often than the "
            "totals alone force, which is §32.4's structure."
        )
    return (
        f"NODF {_num(nested.nodf, 2)} is not above the fixed-fixed null "
        f"(z {_num(result.z, 2)}). §32.4 reports exactly this possibility -- that nestedness "
        '"could arise simply from the degree distribution ... and that there are fewer nested '
        'system than we originally thought" -- so the pattern in the sorted matrix is what the '
        "margins already imply and is not a finding about the corpus."
    )


def core_periphery_payload(report: CorePeripheryReport) -> dict[str, Any]:
    """The same report as plain JSON-able data."""
    core = report.discrete
    payload: dict[str, Any] = {
        "frame": report.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "samples": report.samples,
        "seed": report.seed,
        "better_model": report.better_model,
        "discrete": {
            "core": [str(node) for node in core.core],
            "size": core.size,
            "correlation": core.correlation,
            "core_edges": core.core_edges,
            "boundary_edges": core.boundary_edges,
            "periphery_edges": core.periphery_edges,
            "restarts": core.restarts,
            "method": core.method,
        },
        "continuous": {
            "correlation": report.continuous.correlation,
            "method": report.continuous.method,
            "kcore_correlation": report.continuous.kcore_correlation,
            "kcore_agreement": report.continuous.kcore_agreement,
            "coreness": {str(node): value for node, value in report.continuous.coreness.items()},
            "kcore": {str(node): value for node, value in report.continuous.kcore.items()},
            "shells": {str(number): count for number, count in report.continuous.shells.items()},
        },
        "rich_club": {
            "samples": report.club.samples,
            "sustained": report.club.sustained,
            "verdict": report.club.verdict,
            "rows": [
                {
                    "k": row.k,
                    "nodes": row.nodes,
                    "edges": row.edges,
                    "phi": row.phi,
                    "null_mean": row.null_mean,
                    "null_std": row.null_std,
                    "rho": row.rho,
                }
                for row in report.club.rows
            ],
        },
        "notes": report.notes,
    }
    if report.tension is not None:
        tension = report.tension
        payload["tension"] = {
            "core_correlation": tension.core_correlation,
            "community_correlation": tension.community_correlation,
            "core_excess": tension.core_excess,
            "community_excess": tension.community_excess,
            "core_z": tension.core.z,
            "community_z": tension.community.z,
            "core_null_mean": tension.core.null_mean,
            "community_null_mean": tension.community.null_mean,
            "samples": tension.samples,
            "communities": tension.communities,
            "core_size": tension.core_size,
            "verdict": tension.verdict,
            "reading": tension.reading,
        }
    if report.nested is not None:
        nested = report.nested
        payload["nestedness"] = {
            "nodf": nested.nodf,
            "row_nodf": nested.row_nodf,
            "column_nodf": nested.column_nodf,
            "rows": nested.rows,
            "columns": nested.columns,
            "row_mode": nested.row_mode,
            "column_mode": nested.column_mode,
            "fill": nested.fill,
            "discrepancy": nested.discrepancy,
            "samples": nested.samples,
            "z": nested.significance.z if nested.significance else math.nan,
            "p_value": nested.significance.p_value if nested.significance else math.nan,
            "null_mean": nested.significance.null_mean if nested.significance else math.nan,
        }
    return payload
