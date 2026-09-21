"""The strength of weak ties: which edges bridge, and whether the heavy ones are the embedded ones.

Atlas §30.3. Granovetter's argument is about *where* an edge sits, not about how much traffic it
carries: *"A weak tie is established between individuals whose social circles do not overlap much.
A strong tie is the opposite: an edge between nodes well embedded in the same community"*
(p. 436). So the quantity this module measures per edge is the overlap of the two endpoints'
neighbourhoods, and the quantity it plots it against is the edge weight, because the chapter is
explicit that the two are different things and only usually agree: *"We can, of course, expect a
correlation between being a weak tie and having a low weight. However, we can construct equally
valid scenarios in which there is an anti-correlation instead"* (p. 436, which then builds one out
of edge betweenness).

That is why there is a curve here and not a single number. A network where overlap rises with
weight has the structure Granovetter described -- the heavy edges are inside the groups, the light
ones are the bridges between them, and a message travels between communities only over the light
ones. A network where it does not has weak ties too, in the categorical sense of §30.3, but its
weights are not telling you which they are, and a report that says "our weak ties bridge" on that
evidence has assumed the correlation the chapter refuses to assume.

**What the book defines and what this adds.** The chapter defines overlap in words and gives no
formula. The formula used here is the standard one, from the mobile-phone study that put the
curve on the map (J.-P. Onnela, J. Saramäki, J. Hyvönen, G. Szabó, D. Lazer, K. Kaski, J. Kertész
and A.-L. Barabási, *Structure and tie strengths in mobile communication networks*, PNAS
104(18):7332-7336, 2007): the shared neighbours of the two endpoints over the neighbours either
of them has, excluding the two themselves. The quantile binning and the rank correlation are this
package's conventions for any curve (§3.4), not the paper's.

**What a weight means here matters.** In this package an edge weight counts shared documents or
shared passages, so "strong" means "written down together often", which is a claim about the
corpus rather than about intimacy. Two entities named together in fifty passages are a strong tie
in the only sense the data support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np

from graphrag.sna.export import describe
from graphrag.sna.measures import undirected_view
from graphrag.sna.stats import Correlation, spearman

__all__ = [
    "COLOURABLE",
    "TWO_MODE",
    "OverlapBin",
    "TieReport",
    "neighbourhood_overlap",
    "render_ties",
    "tie_payload",
    "tie_strength_curve",
    "two_mode_kind",
]

#: How many quantile bins the overlap-against-weight curve uses by default. Four because corpus
#: weights are mostly small integers with long ties, and asking for more bins than the weights
#: have distinct quantiles collapses them anyway (see :func:`tie_strength_curve`).
BINS = 4

#: An edge whose endpoints share none of their neighbours: §30.3's weak tie in the structural
#: sense, and a *local bridge* in Granovetter's own terminology -- remove it and its two endpoints
#: have no common acquaintance left.
BRIDGE = 0.0

#: The p-value below which the rank correlation between weight and overlap is read as a
#: relationship rather than as noise. The package's usual 0.05, stated here because §30.3's
#: reading turns on it.
SIGNIFICANT = 0.05

#: The network says so itself: its builders marked its nodes with two ``mode``s, which is what
#: :func:`graphrag.sna.export.describe` reads (§6.4).
TWO_MODE = "declared"
#: Nobody marked the modes, but the network is 2-colourable, so it has no triangle either.
COLOURABLE = "colourable"


def two_mode_kind(graph: nx.Graph) -> str:
    """Whether no edge of this network can sit in a triangle, and how we know (§6.4).

    This matters to every number in this module and to the brokerage in
    :mod:`graphrag.sna.ego`, because both are built out of *shared neighbours*. On a two-mode
    network every edge joins a speaker to an entity, so two adjacent nodes are never of the same
    kind and can share nothing: the neighbourhood overlap is 0 on every edge and the local
    clustering is 0 on every node, **by construction**. Printed without that, it reads as "every
    tie in this corpus is a bridge and every speaker is a perfect broker", which is a fact about
    the data model and not about the corpus.

    Returns :data:`TWO_MODE` when the builders declared the modes, :data:`COLOURABLE` when
    nothing declared them but the graph is 2-colourable anyway -- a star, a tree, any
    triangle-free network -- and ``""`` otherwise. The two are kept apart because the remedy
    differs: a declared two-mode network should be projected (ch. 26), while a one-mode network
    that happens to have no triangles has nothing to project and simply has no §30.3 structure
    to find.
    """
    if describe(graph).bipartite:
        return TWO_MODE
    if graph.number_of_edges() and nx.is_bipartite(graph):
        return COLOURABLE
    return ""


def two_mode_sentence(kind: str, quantity: str, *, section: str = "§30.3") -> str:
    """The sentence a section prints instead of a reading, when §6.4 has decided the answer.

    ``section`` is what is being withheld, because the same artefact hits two of them: §30.3's
    weak and strong ties here, and §30.1's brokerage in :mod:`graphrag.sna.ego`.
    """
    if kind == TWO_MODE:
        return (
            f"This is a **two-mode network**: every edge joins the two modes (§6.4), so two "
            f"adjacent nodes are never of the same kind and share no neighbour. {quantity} "
            f"That is the data model, not a finding about this corpus, and {section}'s reading "
            "is withheld rather than printed over it. Project onto one mode first -- `--project "
            "speakers` or `--project entities` (ch. 26) -- and ask again there, remembering that "
            "a projection closes triangles by construction too."
        )
    return (
        f"This network is **2-colourable**: no edge of it sits in a triangle. {quantity} "
        f"That follows from the shape of the network rather than from how its ties were formed, "
        f"so {section}'s reading is withheld: nothing here declared two modes, so there is "
        "nothing to project and no structure of that kind to find."
    )


@dataclass(frozen=True)
class OverlapBin:
    """One quantile bin of edge weight, and how embedded the edges in it are."""

    label: str
    low: float
    high: float
    edges: int
    mean_overlap: float
    median_overlap: float
    bridges: int
    """Edges in this bin whose endpoints share no neighbour at all."""


@dataclass(frozen=True)
class TieReport:
    """§30.3 over one network: the overlap per edge, binned by weight, with its correlation."""

    nodes: int
    edges: int
    measured: int
    """Edges whose overlap is defined. The undefined ones are the isolated pairs: see
    :func:`neighbourhood_overlap`."""
    weighted: bool
    """Whether the weights vary at all. When they do not there is no curve to draw, only the mean
    overlap, and the report says so instead of binning one value into four bins."""
    mean_overlap: float
    median_overlap: float
    bridges: int
    """Edges with an overlap of exactly 0 (:data:`BRIDGE`)."""
    bins: tuple[OverlapBin, ...] = ()
    correlation: Correlation | None = None
    """Spearman's rho between edge weight and overlap, or ``None`` when it is undefined: fewer
    than three edges, a constant weight, or a constant overlap."""
    weakest: tuple[tuple[str, str, float, float], ...] = ()
    """The heaviest edges that are still bridges, as ``(u, v, weight, overlap)``: the ties this
    network would lose most by losing."""
    two_mode: str = ""
    """Empty, :data:`TWO_MODE` or :data:`COLOURABLE`, from :func:`two_mode_kind`. When it is
    set, every overlap below is 0 because no edge of this network can sit in a triangle, and the
    §30.3 reading is withheld."""
    flattened: str = ""
    frame: str = ""
    notes: tuple[str, ...] = ()

    @property
    def undefined(self) -> int:
        return max(self.edges - self.measured, 0)

    @property
    def bridge_share(self) -> float:
        return self.bridges / self.measured if self.measured else 0.0

    @property
    def reading(self) -> str:
        """What the curve supports, in the chapter's own terms (§30.3)."""
        if not self.measured:
            return (
                "No edge in this network has a defined neighbourhood overlap, so there is "
                "nothing here to read: every edge joins two nodes with no other neighbour."
            )
        if self.two_mode:
            return two_mode_sentence(
                self.two_mode,
                f"All {self.measured:,} of its edges therefore have an overlap of 0 and are "
                "counted as bridges above.",
            )
        if not self.weighted:
            return (
                "Every edge here carries the same weight, so there is no strength to correlate "
                "overlap with. The overlap itself still divides the edges into the embedded and "
                "the bridging ones -- §30.3's weak and strong are categorical, and the chapter "
                "says outright that both exist in an unweighted network -- but which of them are "
                "the heavy ties is a question this network cannot answer."
            )
        result = self.correlation
        if result is None:
            return (
                "The rank correlation between weight and overlap is undefined here (too few "
                "edges, or one of the two does not vary), so the curve above is a description of "
                "these edges and not evidence of a relationship."
            )
        if result.p_value >= SIGNIFICANT:
            return (
                f"Overlap and weight are not monotonically related (rho {result.coefficient:.2f}, "
                f"p {result.p_value:.3f} over {result.n:,} edges): in this network a heavy edge is "
                "no more embedded than a light one, so weight is not a proxy for tie strength in "
                "Granovetter's sense and the bridges have to be read off the overlap column."
            )
        if result.coefficient > 0:
            return (
                f"Overlap rises with weight (rho {result.coefficient:.2f}, p "
                f"{result.p_value:.3f}, {result.n:,} edges): this network has the structure §30.3 "
                "describes. The heavy edges sit inside groups whose members already know each "
                f"other, and the {self.bridges:,} edge(s) with no overlap at all are the ones "
                "carrying anything between groups. A finding that travels only over those is a "
                "finding about the bridges, not about the bulk of the network."
            )
        return (
            f"Overlap *falls* as weight rises (rho {result.coefficient:.2f}, p "
            f"{result.p_value:.3f}, {result.n:,} edges), which is the anti-correlation §30.3 says "
            "can be constructed just as validly as the correlation. Here the heaviest edges are "
            "the ones spanning between groups, so weight and embeddedness point opposite ways and "
            "neither may be called tie strength without saying which one is meant."
        )


def neighbourhood_overlap(graph: nx.Graph) -> dict[tuple[str, str], float]:
    """Per edge, the share of the two endpoints' neighbours that both of them have (§30.3).

    ``O(u, v) = n_uv / ((k_u - 1) + (k_v - 1) - n_uv)`` where ``n_uv`` is the number of nodes
    adjacent to both: 0 when the two share nobody, which is Granovetter's local bridge, and 1
    when neither has a neighbour the other lacks, which is an edge buried inside a clique. It is
    the Jaccard coefficient of the two neighbourhoods with the endpoints themselves taken out,
    so an edge is not counted as evidence that its own endpoints know each other.

    **Undefined**, and therefore absent from the returned mapping rather than reported as 0, when
    both endpoints have degree 1: an isolated pair shares no neighbours because neither has any,
    which is not the same thing as a bridge between two crowded neighbourhoods.
    :class:`TieReport` counts how many edges that was.

    Keys are ``(u, v)`` sorted, so an edge is looked up the same way whichever end you start
    from. A directed network is flattened first (§6.2): a shared neighbour is a shared neighbour
    in either direction, and reading only out-edges would call every sink a bridge.
    """
    flat, _ = undirected_view(graph)
    neighbours = {node: set(flat.neighbors(node)) - {node} for node in flat}
    overlap: dict[tuple[str, str], float] = {}
    for u, v in flat.edges():
        shared = neighbours[u] & neighbours[v]
        union = (neighbours[u] | neighbours[v]) - {u, v}
        if not union:
            continue
        overlap[_pair(u, v)] = len(shared) / len(union)
    return overlap


def _pair(u: str, v: str) -> tuple[str, str]:
    """One edge's key, ordered so that both directions find it."""
    return (u, v) if str(u) <= str(v) else (v, u)


def tie_strength_curve(graph: nx.Graph, *, bins: int = BINS) -> TieReport:
    """Neighbourhood overlap against edge weight, binned by weight quantile (§30.3).

    The book draws this as a curve and reads its slope: rising means the heavy ties are the
    embedded ones and the light ties do the bridging, which is Granovetter's claim; flat or
    falling means the weights are not telling you where the edges sit, which the chapter says is
    an equally constructible network. So the slope is reported as Spearman's rho (§3.4, rank
    rather than linear because nothing here says the relationship is a straight line) with the
    bins printed beside it, because a monotone coefficient hides a U shape and the bins do not.

    ``bins`` is a request, not a promise: the boundaries are weight quantiles, and corpus weights
    are small integers with long ties, so four requested bins over weights that are all 1 or 2
    collapse into two. The report prints the bins it actually made.

    Undefined and said so rather than guessed: a network whose weights are all equal has no curve
    at all (``weighted=False``), and a rank correlation needs at least three edges and variation
    in both columns.
    """
    flat, flattened = undirected_view(graph)
    overlap = neighbourhood_overlap(flat)
    weights = {edge: float(flat[edge[0]][edge[1]].get("weight", 1.0) or 1.0) for edge in overlap}
    values = np.array([overlap[edge] for edge in overlap], dtype=float)
    strengths = np.array([weights[edge] for edge in overlap], dtype=float)
    varies = bool(strengths.size) and float(strengths.min()) != float(strengths.max())
    report = TieReport(
        two_mode=two_mode_kind(flat),
        nodes=flat.number_of_nodes(),
        edges=flat.number_of_edges(),
        measured=int(values.size),
        weighted=varies,
        mean_overlap=float(values.mean()) if values.size else 0.0,
        median_overlap=float(np.median(values)) if values.size else 0.0,
        bridges=int(np.count_nonzero(values == BRIDGE)),
        weakest=_weakest(overlap, weights),
        flattened=flattened,
        frame=str(flat.graph.get("frame", "")),
        notes=_notes(flat, overlap),
    )
    if not values.size:
        return report
    return replace(
        report,
        bins=_bins(strengths, values, bins) if varies else (),
        correlation=_correlation(strengths, values) if varies else None,
    )


def _correlation(strengths: np.ndarray, values: np.ndarray) -> Correlation | None:
    """Spearman's rho, or ``None`` in the cases §3.4 calls undefined rather than zero."""
    try:
        return spearman(strengths.tolist(), values.tolist())
    except ValueError:
        return None


def _bins(strengths: np.ndarray, values: np.ndarray, bins: int) -> tuple[OverlapBin, ...]:
    """The curve: mean overlap inside each weight quantile, with duplicate boundaries collapsed."""
    edges = np.unique(np.quantile(strengths, np.linspace(0.0, 1.0, max(bins, 1) + 1)))
    if edges.size < 2:  # a single boundary is no interval at all
        return ()
    rows: list[OverlapBin] = []
    for index in range(edges.size - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        last = index == edges.size - 2
        inside = (strengths >= low) & (strengths <= high if last else strengths < high)
        chosen = values[inside]
        if not chosen.size:
            continue
        rows.append(
            OverlapBin(
                label=f"{low:g}" if low == high else f"{low:g}-{high:g}",
                low=low,
                high=high,
                edges=int(chosen.size),
                mean_overlap=float(chosen.mean()),
                median_overlap=float(np.median(chosen)),
                bridges=int(np.count_nonzero(chosen == BRIDGE)),
            )
        )
    return tuple(rows)


def _weakest(
    overlap: Mapping[tuple[str, str], float], weights: Mapping[tuple[str, str], float], n: int = 10
) -> tuple[tuple[str, str, float, float], ...]:
    """The heaviest bridges: edges of zero overlap, ranked by weight.

    These are the edges §30.3 is about. They are heavy in the corpus -- the two things were
    written down together often -- and yet their neighbourhoods do not meet, so they are the only
    route between two parts of the network that otherwise share nobody.
    """
    bridges = [
        (u, v, weights[(u, v)], value) for (u, v), value in overlap.items() if value == BRIDGE
    ]
    bridges.sort(key=lambda row: (-row[2], row[0], row[1]))
    return tuple(bridges[:n])


def _notes(graph: nx.Graph, overlap: Mapping[tuple[str, str], float]) -> tuple[str, ...]:
    """What was left out of the curve, and what §6.4 decided before it ran."""
    notes: list[str] = []
    kind = two_mode_kind(graph)
    if kind == TWO_MODE:
        notes.append(
            "Every edge here joins the two modes, so the bridge count above is the edge count "
            "and the mean overlap is 0 whatever this corpus looks like (§6.4). Read it after "
            "`--project`, not here."
        )
    elif kind == COLOURABLE:
        notes.append(
            "This network is 2-colourable and therefore triangle-free, so the bridge count "
            "above is the edge count by construction. Nothing declared two modes, so there is "
            "nothing to project: this network simply has no embedded ties."
        )
    undefined = graph.number_of_edges() - len(overlap)
    if undefined:
        notes.append(
            f"{undefined:,} edge(s) join two nodes that have no other neighbour between them, so "
            "their overlap is undefined (an empty union, not an empty intersection) and they are "
            "in none of the numbers above. An isolated pair is not a bridge."
        )
    return tuple(notes)


# ----------------------------------------------------------------------------- rendering


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_ties(report: TieReport) -> list[str]:
    """The ``## Weak ties`` section: the frame, the n, the absent null, and the chapter."""
    frame = f" Frame: {report.frame}" if report.frame else ""
    lines = [
        "## Weak ties",
        "",
        f"**Sampling frame.** Every edge of this network: {report.edges:,} edge(s) over "
        f"{report.nodes:,} node(s), of which {report.measured:,} have a defined neighbourhood "
        f"overlap.{frame}",
        "",
        f"**n.** {report.measured:,} edge(s)."
        + (f" {report.undefined:,} left out as undefined." if report.undefined else ""),
        "",
        "**Null model.** None. An overlap is an exact property of the network as it was built, "
        "not a claim about a population, and the rank correlation below carries its own p-value "
        "under the null that weight and overlap are independent (§3.4).",
        "",
        "**Implements.** §30.3 (strength of weak ties), with the overlap formula of Onnela et "
        "al. (2007), which the section describes in words.",
        "",
        f"- **Mean overlap** {report.mean_overlap:.4f}, median {report.median_overlap:.4f}.",
        f"- **Bridges** {report.bridges:,} edge(s) ({report.bridge_share:.0%} of those measured) "
        "join two nodes with no neighbour in common"
        + (
            ", which on this network is every edge and means nothing about tie strength -- see "
            "below."
            if report.two_mode
            else ": the weak ties in §30.3's structural sense."
        ),
    ]
    if report.correlation is not None:
        lines.append(
            f"- **Weight against overlap** Spearman rho {report.correlation.coefficient:.4f}, "
            f"p {report.correlation.p_value:.4f} over {report.correlation.n:,} edges."
        )
    lines.append("")
    if report.bins:
        lines += _table(
            ["weight bin", "edges", "mean overlap", "median overlap", "bridges"],
            [
                [
                    row.label,
                    f"{row.edges:,}",
                    f"{row.mean_overlap:.4f}",
                    f"{row.median_overlap:.4f}",
                    f"{row.bridges:,}",
                ]
                for row in report.bins
            ],
        )
    lines += [report.reading, ""]
    if report.weakest and not report.two_mode:
        lines += [
            "The heaviest edges that are still bridges -- often written down together, yet "
            "sharing no neighbour:",
            "",
        ]
        lines += _table(
            ["u", "v", "weight", "overlap"],
            [
                [str(u), str(v), f"{weight:g}", f"{value:.2f}"]
                for u, v, weight, value in report.weakest
            ],
        )
    if report.flattened:
        lines += [
            f"Overlap is computed {report.flattened}, because a shared neighbour is shared in "
            "either direction and reading only out-edges would call every sink a bridge (§6.2).",
            "",
        ]
    lines += [f"- {note}" for note in report.notes]
    if report.notes:
        lines += [""]
    return lines


def tie_payload(report: TieReport) -> dict[str, object]:
    """The same section as plain JSON-able data, for ``--json``."""
    result = report.correlation
    return {
        "nodes": report.nodes,
        "edges": report.edges,
        "measured": report.measured,
        "undefined": report.undefined,
        "weighted": report.weighted,
        "mean_overlap": report.mean_overlap,
        "median_overlap": report.median_overlap,
        "bridges": report.bridges,
        "bridge_share": report.bridge_share,
        "two_mode": report.two_mode,
        "bins": [
            {
                "label": row.label,
                "low": row.low,
                "high": row.high,
                "edges": row.edges,
                "mean_overlap": row.mean_overlap,
                "median_overlap": row.median_overlap,
                "bridges": row.bridges,
            }
            for row in report.bins
        ],
        "correlation": None
        if result is None
        else {
            "method": result.method,
            "coefficient": result.coefficient,
            "p_value": result.p_value,
            "n": result.n,
        },
        "weakest_bridges": [
            {"u": u, "v": v, "weight": weight, "overlap": value}
            for u, v, weight, value in report.weakest
        ],
        "reading": report.reading,
        "frame": report.frame,
        "notes": list(report.notes),
    }
