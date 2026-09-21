"""Graph summarization (ch. 46): reducing a graph to a smaller graph that stands in for it.

The chapter names four techniques and this module builds one function per section:

*§46.1, aggregation* (:func:`aggregate`). Collapse a grouping -- communities, or the nodes that
share an attribute value -- into super nodes, and read off each super node's internal density and
each super edge's weight the way Figure 46.2 does: *"if two nodes u and v are in the same super
node a, then their expected connection probability is |E_a| / (|V_a|(|V_a|-1)), [...] Vice versa,
if u and v are in different super nodes a and b, then their connection probability is
|E_ab| / (|V_a||V_b|)"* (p. 671). This module does not go looking for the grouping that best
compresses the graph -- Grass's mutual-information search (p. 671) -- because ``--by`` already
names one: a community partition or an attribute is what this package groups by everywhere else,
and summarization should aggregate that same answer rather than a second, unrelated one.

*§46.2, compression* (:func:`compress`). The same grouping, this time as a model M whose
description length is charged: which super-node pairs the model declares "connected" costs bits,
and every edge that disagrees with that declaration -- present where the model assumed none, or
missing where it assumed a clique or a complete bipartite block -- is a correction Figure 46.4
charges for by name: *"we add these two rules to the model M and now the summary is a perfect
reconstruction"* (p. 673), each one a named pair -- ``"+(1,5)"``, ``"-(4,8)"`` -- not a member of
an unordered set. ``min L(G, M) = L(M) + L(G|M)`` (p. 673) is the chapter's own objective; the book
states it and not a formula for it (Navlakha et al. 2008, the chapter's own citation for exactly
this two-step scheme), so this module charges each correction ``log2(possible)`` bits, the cost of
naming *which* one of a block's possible pairs it is. That choice is deliberate and not the
obvious one: a combinatorial code that names the whole *set* of exceptions at once,
``log2(C(possible, wrong))``, is symmetric under complementation (``C(n, k) == C(n, n-k)``), so it
would price a block identically whichever way the model declared it and the declaration would do
no work at all -- an earlier version of this module made exactly that mistake. Charging per
correction is what makes "declare this block connected" and "declare it empty" cost different
numbers of bits, which is what lets choosing the cheaper declaration mean something.

*§46.3, simplification* (:func:`simplify_by_importance`). Rank nodes by one centrality (ch. 14)
and keep the top of it, dropping the rest -- no super nodes, the original nodes that survive keep
their own identity, which is what separates this from the first two techniques. The chapter is
explicit that nothing here is chosen to preserve a property of the original graph, unlike sampling
(ch. 29): *"in graph simplification, we are not really interested in preserving any specific
property of the original graph. This is, instead, a core focus of network sampling"* (p. 674-675).

*§46.4, influence-based* (:func:`influence_summary`). SPINE's idea (p. 677, Fig 46.8): run a
spreading process and keep only the edges it actually used. This module simulates the independent
cascade (§21.2, spread.py's ``limited`` model) from a pool of candidate seeds and keeps the union
of edges any of them used, ranking the candidates by how quickly each one's cascade would saturate
its own component -- the fastest spreader is the seed the book's SPINE argument would keep the most
edges around. It does not reuse :func:`graphrag.sna.spread.simulate`, which returns aggregate
state counts over many runs and never records which edge a state change travelled over; that is
exactly the trace SPINE needs, so a small edge-tracking version of the same model lives here.

**What this is not.** The chapter draws its own line, twice. Against network sampling (ch. 29):
*"in network sampling, you explore a part of the network and you operate on the observed structure
directly. Neither is true in graph summarization[...] you want to analyze and understand the whole
structure, and you do so indirectly by manipulating it"* (p. 669). Against backboning (ch. 27,
p. 381-382, quoted in full in :data:`FRAME`): backboning drops edges and keeps every node it can;
summarization is the opposite, built to merge nodes into super nodes and willing to lose sight of
the individual ones for a meso-level view. :func:`aggregate` and :func:`compress` merge nodes on
purpose; :mod:`graphrag.sna.backbone` never does.

**What the chapter cites that this module does not build, and why.** Grass (p. 671), the
mutual-information-guided search for the aggregation itself rather than a grouping named by the
caller -- out of scope for the reason above. The "compressor" edge-aggregation of Figure 46.3,
which collapses a dense bipartite block into a virtual node rather than merging the nodes at either
end -- a second aggregation primitive the ticket's brief does not ask for beside
community/attribute aggregation. Graph OLAP's slice and dice (p. 675) -- ``--where`` already slices
a network by an attribute before any of this module runs, which is the same operation under
another name. The Laplacian-preserving reduction of Purohit et al. (p. 676), which needs the
eigendecomposition of a *smaller* Laplacian with the *same* spectrum, a harder problem than
:func:`graphrag.sna.matrices.eigenpairs` solves here. The community-level "tribes" reading of
influence (GuruMine, p. 676), which needs timestamped adoption events per node -- this corpus
does not have them, the same gap :data:`graphrag.sna.attributes.CONTAGION_CAVEAT` names for
homophily. SPINE's own paper scores edges against *observed* influence events (p. 677); this
corpus has none, so :func:`influence_summary` simulates a plausible one instead and says so with
:data:`graphrag.sna.spread.SPREAD_FRAME`, the same what-if disclaimer every other spreading number
in this package carries.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graphrag.sna.attributes import attribute_labels, attribute_sources
from graphrag.sna.evaluate import (
    PartitionScores,
    evaluate_partition,
    evaluation_payload,
    render_evaluation,
)
from graphrag.sna.export import OWN
from graphrag.sna.matrices import Node
from graphrag.sna.measures import centralities_for, centrality, centrality_meaning, undirected_view
from graphrag.sna.spread import DEFAULT_ATTEMPTS, SPREAD_FRAME

__all__ = [
    "CHAPTER",
    "FRAME",
    "INFLUENCE_CANDIDATES",
    "SIMPLIFY_DEFAULT_SHARE",
    "Aggregation",
    "Compression",
    "InfluenceSummary",
    "SeedRank",
    "Simplification",
    "SimplificationRow",
    "SummaryReport",
    "SuperEdge",
    "SuperNode",
    "aggregate",
    "compress",
    "groups_for",
    "influence_summary",
    "render_summary",
    "simplify_by_importance",
    "summarize",
    "summary_payload",
]

type Edge = tuple[Node, Node]

#: What the section header says it implements, so a reader can open the book at the right page.
CHAPTER = "§46.1-46.4"

#: The chapter's own distinction from backboning, in its own words (p. 381-382), printed in every
#: report this module writes so a reader who reaches for `sna backbone` first, or reads one report
#: after the other, is told which task they are looking at.
FRAME = (
    "This is not backboning (ch. 27). Backboning removes edges and keeps as many nodes as it can, "
    '"because you want to let the strong connections emerge, but you want to preserve all the '
    'entities in your data" (p. 382). Graph summarization does the opposite on purpose: '
    "aggregation and compression merge nodes into super nodes, trading the identity of the "
    'individual node for a meso-level view -- "graph summarization only focuses on empowering '
    'meso-level analysis [...] where you lose sight of the single individual nodes" (p. 382). '
    "Simplification and the influence summary keep the original nodes but, unlike backboning, name "
    "no null model over the edges they drop: nothing here is a significance test."
)

#: How many nodes `--simplify-method` keeps by default, when the caller gives neither `keep` nor
#: `share`. Half is a convention of this module: the book gives no default (p. 674-675).
SIMPLIFY_DEFAULT_SHARE = 0.5

#: How many of the highest-degree nodes `influence_summary` tries as sole seeds when the caller
#: names no candidates. Every candidate needs its own simulated run, so this bounds the cost on a
#: large network the way `spread.py`'s own defaults bound theirs; a convention, not the book's.
INFLUENCE_CANDIDATES = 15


# ----------------------------------------------------------------------------- shared plumbing


def _membership(groups: Sequence[Sequence[Node]]) -> tuple[list[list[Node]], dict[Node, int]]:
    """Groups, with any empty one dropped, and the node -> group index they imply.

    Raises when a node is claimed by two groups: every measure in this module reads a disjoint
    partition, the same requirement :func:`graphrag.sna.evaluate.evaluate_partition` states for
    the chapter 36 battery every grouping here is also checked against.
    """
    ordered = [sorted(set(group), key=str) for group in groups]
    ordered = [group for group in ordered if group]
    membership: dict[Node, int] = {}
    for index, group in enumerate(ordered):
        for node in group:
            if node in membership:
                msg = (
                    f"summarize() needs disjoint groups: node {node!r} is in groups "
                    f"{membership[node] + 1} and {index + 1}"
                )
                raise ValueError(msg)
            membership[node] = index
    return ordered, membership


def _prepare(
    graph: nx.Graph, groups: Sequence[Sequence[Node]]
) -> tuple[nx.Graph, str, list[list[Node]], dict[Node, int]]:
    """Flatten a directed graph and validate `groups`, shared by :func:`aggregate` and
    :func:`compress`.

    Both read only the block structure a grouping imposes and never an edge's direction: Figure
    46.2's reconstruction probability and §46.2's model M are both stated on an undirected
    reading, so a directed network is flattened first, as every other meso-level measure in this
    package is (§6.2).
    """
    view, flattened = undirected_view(graph)
    ordered, membership = _membership(groups)
    unknown = [node for node in membership if node not in view]
    if unknown:
        msg = (
            f"summarize() was given {len(unknown)} node(s) that are not in the graph, starting "
            f"with {unknown[0]!r}"
        )
        raise ValueError(msg)
    return view, flattened, ordered, membership


def _block_counts(
    view: nx.Graph, ordered: Sequence[Sequence[Node]], membership: dict[Node, int]
) -> tuple[list[int], dict[tuple[int, int], int]]:
    """Edges inside every group, and edges between every pair of groups, in one pass."""
    internal = [0] * len(ordered)
    between: dict[tuple[int, int], int] = defaultdict(int)
    for u, v in view.edges():
        if u == v:
            continue
        cu, cv = membership.get(u), membership.get(v)
        if cu is None or cv is None:
            continue
        if cu == cv:
            internal[cu] += 1
        else:
            key = (cu, cv) if cu < cv else (cv, cu)
            between[key] += 1
    return internal, between


def _groups_by_attribute(
    graph: nx.Graph, key: str
) -> tuple[list[list[Node]], list[str], int, float]:
    """Node groups by one attribute's value (`--by attr:<key>`), dropping unlabelled nodes.

    Reads :func:`graphrag.sna.attributes.attribute_labels`, the same function `sna analyze --by`
    reads, so the same key groups the same way everywhere this package groups by an attribute.
    Also returns how many nodes carried no value, and the share of labelled nodes whose value was
    *borrowed* from a document rather than recorded about the node itself
    (:func:`graphrag.sna.attributes.attribute_sources`): a group built from borrowed labels is
    tautologically dense, because the edges and the label both come from the same documents
    (CLAUDE.md, node attributes; :mod:`graphrag.sna.attributes`), and the report prints the share
    rather than trusting the density.
    """
    labels = attribute_labels(graph, key)
    sources = attribute_sources(graph, key)
    by_value: dict[str, list[Node]] = defaultdict(list)
    for node, value in labels.items():
        by_value[value].append(node)
    values = sorted(by_value)
    groups = [sorted(by_value[value], key=str) for value in values]
    left_out = graph.number_of_nodes() - len(labels)
    borrowed = sum(1 for source in sources.values() if source != OWN)
    borrowed_share = borrowed / len(sources) if sources else 0.0
    return groups, values, left_out, borrowed_share


def groups_for(
    graph: nx.Graph,
    by: str,
    *,
    resolution: float = 1.0,
    runs: int = 10,
    seed: int | None = None,
) -> tuple[list[list[Node]], list[str], int, float | None, str]:
    """The grouping `--by` names: `community` (Louvain) or `attr:<key>` (an attribute's values).

    Returns the groups, one label per group, how many nodes were left out (0 for `community`,
    since Louvain covers the graph it partitioned), the share of borrowed attribute labels
    (`None` for `community`, where the question does not apply), and Louvain's own note about
    flattening a directed network, when there is one.
    """
    if by == "community":
        from graphrag.sna.cluster import louvain

        result = louvain(graph, resolution=resolution, seed=seed, runs=runs)
        groups = result.communities
        labels = [f"community #{index + 1}" for index in range(len(groups))]
        return groups, labels, 0, None, result.note
    if by.startswith("attr:"):
        key = by.removeprefix("attr:").strip()
        if not key:
            msg = "--by attr:<key> needs a key after 'attr:'"
            raise ValueError(msg)
        groups, labels, left_out, borrowed_share = _groups_by_attribute(graph, key)
        return groups, labels, left_out, borrowed_share, ""
    msg = f"--by must be 'community' or 'attr:<key>', got {by!r}"
    raise ValueError(msg)


def _num(value: float, places: int = 4) -> str:
    """A number, or a dash where the measure is undefined (nan) or unmeasured (None)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


# ----------------------------------------------------------------------------- §46.1 aggregation


@dataclass(frozen=True)
class SuperNode:
    """One group, collapsed into a super node (§46.1, Fig 46.2(b))."""

    index: int
    label: str
    members: list[Node]
    size: int
    internal_edges: int
    """|E_a|: edges with both ends in this group."""
    possible_edges: int
    """|V_a|(|V_a|-1)/2: the pairs this group could hold."""
    internal_density: float
    """|E_a| / possible_edges (Fig 46.2's "expected connection probability" inside a super node).
    `nan` for a group of fewer than two members, where no pair exists to have a density."""


@dataclass(frozen=True)
class SuperEdge:
    """One pair of groups with at least one edge between them (§46.1, Fig 46.2(b))."""

    source: int
    target: int
    edges: int
    """|E_ab|: edges collapsed into this super edge -- the weight Figure 46.2(b) labels it with."""
    possible: int
    """|V_a||V_b|: the pairs that could join the two groups."""
    density: float
    """|E_ab| / possible (Fig 46.2's reconstructed connection probability between two super
    nodes)."""


@dataclass(frozen=True)
class Aggregation:
    """The meta-network §46.1 collapses `groups` into."""

    frame: str
    by: str
    directed: bool
    nodes: int
    edges: int
    left_out: int
    borrowed_share: float | None
    supernodes: list[SuperNode] = field(default_factory=list)
    superedges: list[SuperEdge] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def aggregate(
    graph: nx.Graph,
    groups: Sequence[Sequence[Node]],
    labels: Sequence[str],
    *,
    by: str,
    left_out: int = 0,
    borrowed_share: float | None = None,
) -> Aggregation:
    """§46.1: collapse `groups` into super nodes, with edge weights and within-group density.

    Figure 46.2 is the worked example this follows to the letter: a super node's internal density
    is |E_a| / (|V_a|(|V_a|-1)/2), the share of the pairs it could hold that are actual edges, and
    a super edge's weight is |E_ab|, the count of original edges collapsed between the two groups,
    printed beside |E_ab| / (|V_a||V_b|), the share of the possible pairs it represents -- exactly
    the reconstructed adjacency matrix of Figure 46.2(c).

    `groups` need not cover the network: an attribute grouping drops the nodes nobody labelled,
    and an edge with an unlabelled endpoint is outside every count here, the convention
    :func:`graphrag.sna.evaluate.evaluate_partition` uses for a partial partition too. `labels`
    names each group in `groups`, in the same order, after any empty group is dropped it is
    re-zipped against the surviving ones.

    Raises on two groups sharing a node (`labels` and `groups` must then be the same length before
    filtering), or on a node not in `graph`.
    """
    view, flattened, ordered, membership = _prepare(graph, groups)
    kept_labels = _labels_for(groups, labels, ordered)
    internal, between = _block_counts(view, ordered, membership)
    supernodes = [
        SuperNode(
            index=index,
            label=kept_labels[index],
            members=members,
            size=len(members),
            internal_edges=internal[index],
            possible_edges=(len(members) * (len(members) - 1)) // 2,
            internal_density=(
                internal[index] / ((len(members) * (len(members) - 1)) // 2)
                if len(members) > 1
                else math.nan
            ),
        )
        for index, members in enumerate(ordered)
    ]
    superedges = [
        SuperEdge(
            source=source,
            target=target,
            edges=count,
            possible=len(ordered[source]) * len(ordered[target]),
            density=count / (len(ordered[source]) * len(ordered[target])),
        )
        for (source, target), count in sorted(between.items())
    ]
    notes: list[str] = []
    if flattened:
        notes.append(f"Aggregation was computed {flattened} (§6.2, §46.1).")
    if left_out:
        notes.append(
            f"{left_out:,} node(s) carried no value for `{by}` and are outside every super node."
        )
    if borrowed_share:
        notes.append(
            f"{borrowed_share:.0%} of the labelled nodes' values were *borrowed* from a document "
            "rather than recorded about the node itself, so a super node's density partly "
            "measures how the label was assigned, not the entities it groups (CLAUDE.md, node "
            "attributes; graphrag.sna.attributes)."
        )
    return Aggregation(
        frame=str(view.graph.get("frame", "")),
        by=by,
        directed=graph.is_directed(),
        nodes=sum(len(group) for group in ordered),
        edges=sum(internal) + sum(between.values()),
        left_out=left_out,
        borrowed_share=borrowed_share,
        supernodes=supernodes,
        superedges=superedges,
        notes=notes,
    )


def _labels_for(
    groups: Sequence[Sequence[Node]], labels: Sequence[str], ordered: Sequence[Sequence[Node]]
) -> list[str]:
    """`labels`, re-aligned to `ordered` after `_membership` has dropped any empty group."""
    if len(groups) != len(labels):
        msg = f"summarize() needs one label per group: {len(groups)} groups, {len(labels)} labels"
        raise ValueError(msg)
    kept = [label for group, label in zip(groups, labels, strict=True) if set(group)]
    if len(kept) != len(ordered):  # pragma: no cover -- defensive; _membership dedupes by set()
        msg = "summarize() could not align labels to groups after dropping empty ones"
        raise ValueError(msg)
    return kept


# ----------------------------------------------------------------------------- §46.2 compression


def _log2_choose(n: int, k: int) -> float:
    """log2(C(n, k)): the combinatorial code that says which k of n things happened.

    Written with the log-gamma function rather than `math.comb`, which returns an exact integer
    that can overflow a float's range long before a real network's edge count does. 0.0 at either
    end of the range (k=0 or k=n), where there is nothing left to choose.
    """
    if k <= 0 or k >= n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2.0)


def _log2_multinomial(sizes: Sequence[int]) -> float:
    """log2(N! / prod(n_i!)): the bits to say which of N nodes went into which labelled group."""
    total = sum(sizes)
    if total <= 0:
        return 0.0
    return (math.lgamma(total + 1) - sum(math.lgamma(size + 1) for size in sizes)) / math.log(2.0)


def _list_bits(possible: int, wrong: int) -> float:
    """The cost of writing down `wrong` explicit exceptions to one group-pair's declaration.

    Figure 46.4's own scheme (p. 673) lists each correction by name -- ``"+(1,5)"``,
    ``"-(4,8)"`` -- one specific pair at a time, not a member of an unordered set. Naming one of
    `possible` candidate pairs costs log2(possible) bits, so `wrong` of them cost
    `wrong * log2(possible)`.

    This is deliberately **not** the combinatorial code `log2(C(possible, wrong))` that names the
    whole *set* of exceptions in one code, because that code is symmetric under complementation --
    `C(n, k) == C(n, n-k)` -- so it prices a block identically whichever way the model declared
    it, and the declaration `compress` makes (§46.2's "connected" bit) would cost `model_bits`
    for no benefit at all. Charging per correction breaks that symmetry: declaring a block
    "connected" when it is a clique costs 0 corrections, and declaring the same block "empty"
    would cost `possible * log2(possible)` -- a real difference that makes choosing the cheaper
    declaration a real saving, not just an accounting fiction.

    0.0 when there is nothing wrong, or nothing to name (`possible <= 1`, where there is only one
    candidate pair and so no bits are needed to say which one it is).
    """
    if wrong <= 0 or possible <= 1:
        return 0.0
    return wrong * math.log2(possible)


@dataclass(frozen=True)
class Compression:
    """§46.2's `min L(G, M) = L(M) + L(G|M)` (p. 673), for one grouping of one graph."""

    frame: str
    by: str
    nodes: int
    edges: int
    groups: int
    declared_pairs: int
    """q: how many of the possible group-pairs the model M declares "connected"."""
    possible_pairs: int
    """P = k(k+1)/2: every group-pair the model could declare connected, including a group with
    itself."""
    partition_bits: float
    """The cost of stating which node is in which group -- log2(N! / prod(n_i!))."""
    model_bits: float
    """L(M): the cost of stating which q of P group-pairs are "connected" -- log2(C(P, q))."""
    correction_bits: float
    """L(G|M): `log2(possible)` bits per edge that disagrees with what M declared, summed over
    every group-pair -- Figure 46.4's "+" and "-" rules, each one named individually
    (:func:`_list_bits`) rather than folded into a single set-valued code."""
    mistakes: int
    """The correction count in edges, not bits: how many "+"/"-" rules Figure 46.4's scheme would
    need to write down."""
    total_bits: float
    """partition_bits + model_bits + correction_bits: the whole cost of G given this grouping."""
    raw_bits: float
    """log2(C(N(N-1)/2, |E|)): the same combinatorial code applied to the plain edge list, with no
    grouping at all -- the baseline this grouping's cost is judged against."""
    ratio: float
    """total_bits / raw_bits. Below 1, the grouping compresses; above 1, stating the grouping and
    its corrections costs more than just listing the edges would."""
    notes: list[str] = field(default_factory=list)


def compress(
    graph: nx.Graph,
    groups: Sequence[Sequence[Node]],
    labels: Sequence[str],
    *,
    by: str,
) -> Compression:
    """§46.2: the description length of `graph` given `groups` as the model's blocks.

    Follows the two-step scheme Figure 46.4 works by hand: for every pair of groups (including a
    group paired with itself), the model declares "connected" when that costs fewer correction
    bits than declaring it empty would -- equivalently, when at least half of the pairs it could
    hold are actual edges, since :func:`_list_bits` charges each correction the same
    `log2(possible)`, so minimising the *count* of corrections minimises their cost too -- and
    every edge that disagrees with the declaration is a correction: a missing edge inside a
    "connected" pair, or a present edge inside one that is not. `mistakes` counts them;
    `correction_bits` is `log2(possible)` per correction, summed over every group-pair
    (:func:`_list_bits`, and *not* the combinatorial code that names the whole set of exceptions
    in one number -- that code is blind to which way a block was declared, which would make the
    declaration cost bits for nothing). `model_bits` is the combinatorial code that says which
    pairs were declared connected at all, and `partition_bits` the code that says which node went
    into which group -- a cost the raw edge list, with no grouping, never has to pay.

    A directed network is flattened first, for the same reason :func:`aggregate` flattens one
    (§6.2): the model Figure 46.4 describes reads no direction.

    Two groups, both cliques, joined by one edge cost far fewer bits this way than their edge
    list, because both diagonal group-pairs need zero corrections and the one off-diagonal pair
    needs exactly one; a network with no block structure at that grouping -- an Erdos-Renyi graph
    of the same size, say -- pays a `log2(possible)` correction for close to half the pairs of
    every block and gets no benefit from stating the grouping at all, so its total is the larger
    of the two.
    """
    view, flattened, ordered, membership = _prepare(graph, groups)
    _labels_for(groups, labels, ordered)  # validated for its own sake; compress prints no labels
    internal, between = _block_counts(view, ordered, membership)
    sizes = [len(group) for group in ordered]
    k = len(ordered)
    possible_pairs = k * (k + 1) // 2
    declared = 0
    mistakes = 0
    correction_bits = 0.0
    for i in range(k):
        possible = (sizes[i] * (sizes[i] - 1)) // 2
        actual = internal[i]
        connected = possible > 0 and actual * 2 >= possible
        declared += int(connected)
        wrong = (possible - actual) if connected else actual
        mistakes += wrong
        correction_bits += _list_bits(possible, wrong)
    for i in range(k):
        for j in range(i + 1, k):
            possible = sizes[i] * sizes[j]
            actual = between.get((i, j), 0)
            connected = possible > 0 and actual * 2 >= possible
            declared += int(connected)
            wrong = (possible - actual) if connected else actual
            mistakes += wrong
            correction_bits += _list_bits(possible, wrong)
    partition_bits = _log2_multinomial(sizes)
    model_bits = _log2_choose(possible_pairs, declared)
    total_bits = partition_bits + model_bits + correction_bits
    total_nodes = sum(sizes)
    total_possible = (total_nodes * (total_nodes - 1)) // 2
    total_edges = sum(internal) + sum(between.values())
    raw_bits = _log2_choose(total_possible, total_edges)
    notes: list[str] = []
    if flattened:
        notes.append(f"Compression was computed {flattened} (§6.2, §46.2).")
    return Compression(
        frame=str(view.graph.get("frame", "")),
        by=by,
        nodes=total_nodes,
        edges=total_edges,
        groups=k,
        declared_pairs=declared,
        possible_pairs=possible_pairs,
        partition_bits=partition_bits,
        model_bits=model_bits,
        correction_bits=correction_bits,
        mistakes=mistakes,
        total_bits=total_bits,
        raw_bits=raw_bits,
        ratio=(total_bits / raw_bits if raw_bits else math.nan),
        notes=notes,
    )


# ----------------------------------------------------------------------------- §46.3 simplification


@dataclass(frozen=True)
class SimplificationRow:
    rank: int
    node: Node
    value: float
    kept: bool


@dataclass(frozen=True)
class Simplification:
    """§46.3: the graph with only its most important nodes, ranked by one centrality (ch. 14)."""

    frame: str
    method: str
    meaning: str
    nodes_before: int
    edges_before: int
    nodes_after: int
    edges_after: int
    keep: int
    ranked: list[SimplificationRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def simplify_by_importance(
    graph: nx.Graph,
    *,
    method: str = "degree",
    keep: int | None = None,
    share: float | None = None,
) -> Simplification:
    """§46.3: drop the least important nodes, ranked by `method` (one of ch. 14's centralities).

    Exactly one of `keep` (a node count) or `share` (a fraction of the network) says how many
    survive; neither given defaults to :data:`SIMPLIFY_DEFAULT_SHARE`. Ties in the centrality are
    broken by node id, so the ranking -- and therefore the result -- is reproducible: on a star,
    every leaf ties the others at the lowest degree, and the id order is what decides which ones
    the report lists first among the dropped.

    Nothing here targets a property of the original graph, unlike sampling (ch. 29): the chapter
    says so explicitly -- "we are not really interested in preserving any specific property of the
    original graph. This is, instead, a core focus of network sampling" (p. 674-675) -- so no bias
    table is printed the way `sna sample`'s is; the ranking itself is the whole method.

    Raises on `method` not in :func:`graphrag.sna.measures.centralities_for` for this graph, or on
    `keep` and `share` both given at once.
    """
    if keep is not None and share is not None:
        msg = "simplify_by_importance() takes keep or share, not both"
        raise ValueError(msg)
    valid = centralities_for(graph)
    if method not in valid:
        msg = f"method must be one of {', '.join(valid)}, got {method!r}"
        raise ValueError(msg)
    scores = centrality(graph, method)
    total = graph.number_of_nodes()
    if keep is not None:
        target = max(0, min(keep, total))
    else:
        target = max(
            0, min(total, round((SIMPLIFY_DEFAULT_SHARE if share is None else share) * total))
        )
    order = sorted(scores, key=lambda node: (-scores[node], str(node)))
    kept_nodes = set(order[:target])
    ranked = [
        SimplificationRow(rank=index + 1, node=node, value=scores[node], kept=node in kept_nodes)
        for index, node in enumerate(order)
    ]
    result = graph.subgraph(kept_nodes)
    notes = [
        f"share = keep / n = {target}/{total} = {target / total:.0%}" if total else "n = 0",
    ]
    return Simplification(
        frame=str(graph.graph.get("frame", "")),
        method=method,
        meaning=centrality_meaning(method, graph),
        nodes_before=total,
        edges_before=graph.number_of_edges(),
        nodes_after=result.number_of_nodes(),
        edges_after=result.number_of_edges(),
        keep=target,
        ranked=ranked,
        notes=notes,
    )


# ----------------------------------------------------------------------------- §46.4 influence


@dataclass(frozen=True)
class SeedRank:
    """One candidate seed, and how its own simulated cascade did (§46.4)."""

    rank: int
    node: Node
    steps_to_saturate: float | None
    """Mean number of steps until every node reachable from this seed was infected, averaged over
    the runs that got there. `None` when no run saturated its component."""
    reach_share: float
    """Mean share of the seed's own component ever infected, over all runs."""
    saturated_share: float
    """Share of runs that reached every node of the seed's component."""


@dataclass(frozen=True)
class InfluenceSummary:
    """§46.4: the SPINE-style summary -- edges a simulated cascade used, and who spread fastest."""

    frame: str
    model: str
    beta: float
    attempts: int
    runs: int
    seed: int | None
    candidates: int
    ranked: list[SeedRank] = field(default_factory=list)
    kept_edges: int = 0
    total_edges: int = 0
    edge_share: float = 0.0
    caveat: str = SPREAD_FRAME
    notes: list[str] = field(default_factory=list)


def _cascade(
    graph: nx.Graph, start: Node, *, beta: float, attempts: int, rng: random.Random
) -> tuple[dict[Node, int], list[Edge]]:
    """One run of the independent cascade (§21.2, `spread.py`'s `limited` model) from `start`,
    tracking the step and the edge each infection travelled over.

    Not `graph.sna.spread.simulate`: that function reports aggregate state counts over many runs
    at once and never records which edge did the work, which is exactly the trace SPINE needs
    (§46.4, Fig 46.8). A node tries each susceptible neighbour once per step, for up to `attempts`
    steps after its own infection, and then never tries that neighbour again --
    `graphrag.sna.spread.MODEL_NOTES["limited"]`'s own description. `attempts=1`, this module's
    default and `spread.py`'s own :data:`graphrag.sna.spread.DEFAULT_ATTEMPTS`, is one shot per
    neighbour, at the step right after infection.
    """
    infected_at: dict[Node, int] = {start: 0}
    tried: set[Edge] = set()
    used: list[Edge] = []
    live = {start}
    step = 0
    limit = graph.number_of_nodes() * max(attempts, 1) + 1
    while live and step < limit:
        step += 1
        exhausted: set[Node] = set()
        newly: set[Node] = set()
        for node in live:
            born = infected_at[node]
            if step > born + attempts:
                exhausted.add(node)
                continue
            for neighbor in graph.neighbors(node):
                if neighbor in infected_at or neighbor in newly:
                    continue
                key = (node, neighbor)
                if key in tried:
                    continue
                tried.add(key)
                if rng.random() < beta:
                    newly.add(neighbor)
                    used.append(key)
            if step >= born + attempts:
                exhausted.add(node)
        for node in newly:
            infected_at[node] = step
        live = (live - exhausted) | newly
    return infected_at, used


def _seed_run(
    graph: nx.Graph, start: Node, *, beta: float, attempts: int, rng: random.Random
) -> tuple[float | None, float, list[Edge]]:
    """One cascade from `start`, read against the component it could possibly reach."""
    infected_at, used = _cascade(graph, start, beta=beta, attempts=attempts, rng=rng)
    if graph.is_directed():
        component = nx.descendants(graph, start) | {start}
    else:
        component = nx.node_connected_component(graph, start)
    reach_share = len(infected_at) / len(component) if component else 0.0
    saturated = len(infected_at) >= len(component)
    steps = float(max(infected_at.values())) if saturated and infected_at else None
    return steps, reach_share, used


def _top_candidates(graph: nx.Graph, top: int) -> list[Node]:
    """The `top` highest-degree nodes, id-ordered on a tie -- the pool `influence_summary` tries
    as sole seeds when the caller names none."""
    scores = centrality(graph, "degree")
    order = sorted(scores, key=lambda node: (-scores[node], str(node)))
    return order[: max(top, 0)]


def influence_summary(
    graph: nx.Graph,
    *,
    candidates: Sequence[Node] | None = None,
    top: int = INFLUENCE_CANDIDATES,
    beta: float = 1.0,
    attempts: int = DEFAULT_ATTEMPTS,
    runs: int = 1,
    seed: int | None = None,
) -> InfluenceSummary:
    """§46.4: simulate an independent cascade from every candidate seed and keep the edges it used.

    `candidates` names the seeds to try; without it, the :data:`INFLUENCE_CANDIDATES`
    highest-degree nodes are tried, one at a time, each as the sole seed of its own run. Every
    candidate is ranked by `steps_to_saturate`: the mean number of steps its cascade took to
    infect every node it could possibly reach (its own connected component, or -- on a directed
    network -- its descendants), over the runs that got there at all; a candidate whose cascade
    never saturates in any run sorts last. `reach_share` is printed beside it because a fast
    partial spreader and a slow complete one are different findings even when the step count ties.

    `beta=1.0`, this module's default, makes the cascade deterministic (`runs=1` is then enough):
    every reachable node is infected eventually, and the only thing that varies between seeds is
    how many steps it takes -- which is what makes a hub's cascade provably faster than a leaf's,
    rather than a coin flip that happens to land that way. A `beta` below 1 needs `runs` above 1 to
    average over, and the report says what share of them saturated at all.

    `kept_edges` is the union of every edge any candidate's cascade used, across every run: Figure
    46.8 shows one spreading event; pooling several plausible ones is this module's way of asking
    the same question of a whole network rather than one hypothetical run, and the choice is
    stated here rather than left for a reader to assume the figure's single-event reading.

    :data:`graphrag.sna.spread.SPREAD_FRAME` is carried as `caveat` on every result: this
    network's edges are corpus co-occurrences, so nothing here is a record of what happened, only
    a simulation of what an idea moving along these edges would do.
    """
    frame = str(graph.graph.get("frame", ""))
    if graph.number_of_nodes() == 0:
        return InfluenceSummary(
            frame=frame,
            model="limited (§21.2)",
            beta=beta,
            attempts=attempts,
            runs=runs,
            seed=seed,
            candidates=0,
            notes=["Nothing to summarise: the network has no nodes."],
        )
    pool = (
        list(dict.fromkeys(candidates)) if candidates is not None else _top_candidates(graph, top)
    )
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    directed = graph.is_directed()
    used_edges: set[Edge] = set()
    unranked: list[SeedRank] = []
    for node in pool:
        step_samples: list[float] = []
        reach_samples: list[float] = []
        saturations = 0
        for _ in range(max(runs, 1)):
            steps, reach_share, used = _seed_run(graph, node, beta=beta, attempts=attempts, rng=rng)
            reach_samples.append(reach_share)
            # On an undirected graph the same edge can be the channel of a cascade travelling
            # either way -- hub->leaf in one seed's run, leaf->hub in another's -- and it is one
            # edge of the network either time, so it is canonicalised before joining the kept set.
            used_edges.update(
                used if directed else {(min(u, v, key=str), max(u, v, key=str)) for u, v in used}
            )
            if steps is not None:
                step_samples.append(steps)
                saturations += 1
        unranked.append(
            SeedRank(
                rank=0,
                node=node,
                steps_to_saturate=(sum(step_samples) / len(step_samples) if step_samples else None),
                reach_share=sum(reach_samples) / len(reach_samples) if reach_samples else 0.0,
                saturated_share=saturations / max(runs, 1),
            )
        )
    unranked.sort(
        key=lambda row: (
            row.steps_to_saturate is None,
            row.steps_to_saturate if row.steps_to_saturate is not None else math.inf,
            -row.reach_share,
            str(row.node),
        )
    )
    ranked = [
        SeedRank(
            rank=index + 1,
            node=row.node,
            steps_to_saturate=row.steps_to_saturate,
            reach_share=row.reach_share,
            saturated_share=row.saturated_share,
        )
        for index, row in enumerate(unranked)
    ]
    total_edges = graph.number_of_edges()
    kept = len(used_edges)
    notes: list[str] = []
    if candidates is None and top < graph.number_of_nodes():
        notes.append(
            f"Candidates limited to the {top} highest-degree nodes; pass explicit seeds to try "
            "your own."
        )
    return InfluenceSummary(
        frame=frame,
        model="limited (§21.2)",
        beta=beta,
        attempts=attempts,
        runs=runs,
        seed=seed,
        candidates=len(pool),
        ranked=ranked,
        kept_edges=kept,
        total_edges=total_edges,
        edge_share=(kept / total_edges if total_edges else 0.0),
        notes=notes,
    )


# ----------------------------------------------------------------------------- the whole chapter


@dataclass(frozen=True)
class SummaryReport:
    """All four techniques of chapter 46, over one network and one grouping."""

    frame: str
    by: str
    nodes: int
    edges: int
    aggregation: Aggregation
    compression: Compression
    simplification: Simplification
    influence: InfluenceSummary
    evaluation: PartitionScores | None = None
    notes: list[str] = field(default_factory=list)


def summarize(
    graph: nx.Graph,
    *,
    by: str,
    resolution: float = 1.0,
    community_runs: int = 10,
    simplify_method: str = "degree",
    simplify_share: float | None = None,
    influence_top: int = INFLUENCE_CANDIDATES,
    influence_beta: float = 1.0,
    influence_attempts: int = DEFAULT_ATTEMPTS,
    influence_runs: int = 1,
    seed: int | None = None,
) -> SummaryReport:
    """Chapter 46, end to end: group `graph` by `by`, then run every one of §46.1-46.4 over it.

    `by` is `"community"` (Louvain, §35.1-35.2, at `resolution` over `community_runs` seeds for
    stability) or `"attr:<key>"` (the nodes that share one value of an attribute, the same
    grouping `sna analyze --by` reads). The grouping is checked against the chapter 36 battery
    (`evaluation`) whenever it has at least two groups and the network has an edge, the same rule
    every other partition this package produces is held to -- `resolution` is passed through to
    that check too, since modularity at one resolution is a different function from modularity at
    another and comparing a partition found at 0.5 against the battery's default of 1.0 would
    score a partition the search never optimised for (§36.1, p. 517; the same reason
    `graphrag.sna.multilayer` threads its own resolution through).
    """
    groups, labels, left_out, borrowed_share, community_note = groups_for(
        graph, by, resolution=resolution, runs=community_runs, seed=seed
    )
    if not groups:
        msg = f"--by {by} found no groups: nothing carries a value, or the network has no edges"
        raise ValueError(msg)
    aggregation = aggregate(
        graph, groups, labels, by=by, left_out=left_out, borrowed_share=borrowed_share
    )
    compression = compress(graph, groups, labels, by=by)
    simplification = simplify_by_importance(graph, method=simplify_method, share=simplify_share)
    influence = influence_summary(
        graph,
        top=influence_top,
        beta=influence_beta,
        attempts=influence_attempts,
        runs=influence_runs,
        seed=seed,
    )
    evaluation: PartitionScores | None = None
    if len(groups) >= 2 and graph.number_of_edges():
        evaluation = evaluate_partition(graph, groups, resolution=resolution, seed=seed)
    notes: list[str] = []
    if community_note:
        notes.append(community_note)
    return SummaryReport(
        frame=str(graph.graph.get("frame", "")),
        by=by,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        aggregation=aggregation,
        compression=compression,
        simplification=simplification,
        influence=influence,
        evaluation=evaluation,
        notes=notes,
    )


# ----------------------------------------------------------------------------- rendering


def render_summary(report: SummaryReport) -> list[str]:
    """The whole chapter as markdown: one `##` section per technique, in the book's own order."""
    lines: list[str] = [
        "# Graph summarization",
        "",
        f"**Implements.** {CHAPTER} — aggregation, compression, simplification and an "
        "influence-based summary.",
        "",
        f"> {FRAME}",
        "",
        f"**Sampling frame.** {report.frame or 'not recorded'}",
        "",
        f"n = {report.nodes:,} nodes, {report.edges:,} edges, grouped by `{report.by}`.",
        "",
    ]
    if report.notes:
        lines += [f"> {note}" for note in report.notes]
        lines.append("")
    lines += _render_aggregation(report.aggregation)
    if report.evaluation is not None:
        lines += render_evaluation(report.evaluation)
    lines += _render_compression(report.compression)
    lines += _render_simplification(report.simplification)
    lines += _render_influence(report.influence)
    return lines


def _render_aggregation(aggregation: Aggregation) -> list[str]:
    lines = [
        "## Aggregation (§46.1)",
        "",
        "**Null model.** none: aggregation is descriptive, a reading of the grouping already "
        "given, never a test against chance.",
        "",
        f"{len(aggregation.supernodes)} super node(s) from {aggregation.nodes:,} node(s), "
        f"{aggregation.edges:,} edge(s) accounted for.",
        "",
    ]
    lines += _table(
        ["#", "label", "size", "internal edges", "possible", "internal density"],
        [
            [
                str(node.index + 1),
                node.label,
                f"{node.size:,}",
                f"{node.internal_edges:,}",
                f"{node.possible_edges:,}",
                _num(node.internal_density, 3),
            ]
            for node in aggregation.supernodes
        ],
    )
    if aggregation.superedges:
        lines += ["Super edges: the collapsed weight between two super nodes, and its density.", ""]
        lines += _table(
            ["from", "to", "weight", "possible", "density"],
            [
                [
                    aggregation.supernodes[edge.source].label,
                    aggregation.supernodes[edge.target].label,
                    f"{edge.edges:,}",
                    f"{edge.possible:,}",
                    _num(edge.density, 3),
                ]
                for edge in aggregation.superedges
            ],
        )
    else:
        lines += ["No edges cross between super nodes: every surviving edge is internal.", ""]
    if aggregation.notes:
        lines += [f"> {note}" for note in aggregation.notes] + [""]
    return lines


def _render_compression(compression: Compression) -> list[str]:
    lines = [
        "## Compression (§46.2)",
        "",
        "**Null model.** none: `raw_bits` is the baseline this grouping's cost is judged "
        "against, not a random-graph null.",
        "",
        f"{compression.groups} group(s) over {compression.nodes:,} node(s) and "
        f"{compression.edges:,} edge(s). Model M declares {compression.declared_pairs} of "
        f'{compression.possible_pairs} group-pair(s) "connected"; {compression.mistakes:,} '
        "edge(s) disagree with that declaration and are corrected individually.",
        "",
    ]
    lines += _table(
        ["quantity", "bits"],
        [
            ["partition (which node, which group)", _num(compression.partition_bits, 2)],
            ["model M (which group-pairs are connected)", _num(compression.model_bits, 2)],
            ["corrections L(G|M)", _num(compression.correction_bits, 2)],
            ["total L(partition) + L(M) + L(G|M)", _num(compression.total_bits, 2)],
            ["raw edge list, same n and |E|, no grouping", _num(compression.raw_bits, 2)],
            ["ratio: total / raw", _num(compression.ratio, 3)],
        ],
    )
    lines.append(
        "Below 1, this grouping compresses the network; above 1, stating the grouping and its "
        "corrections costs more than the plain edge list would."
    )
    lines.append("")
    if compression.notes:
        lines += [f"> {note}" for note in compression.notes] + [""]
    return lines


def _render_simplification(simplification: Simplification) -> list[str]:
    lines = [
        "## Simplification (§46.3)",
        "",
        "**Null model.** none, and deliberately: the chapter is explicit that nothing here is "
        "chosen to preserve a property of the original graph, unlike sampling (ch. 29, p. "
        "674-675).",
        "",
        f"Ranked by **{simplification.method}** ({simplification.meaning}). Kept "
        f"{simplification.keep:,} of {simplification.nodes_before:,} nodes "
        f"({simplification.nodes_after:,} nodes and {simplification.edges_after:,} edges survive, "
        f"from {simplification.edges_before:,} edges before).",
        "",
    ]
    kept_rows = [row for row in simplification.ranked if row.kept]
    dropped_rows = [row for row in simplification.ranked if not row.kept]
    lines += ["Kept, highest first:", ""]
    lines += _table(
        ["rank", "node", simplification.method],
        [[str(row.rank), str(row.node), _num(row.value, 4)] for row in kept_rows[:20]],
    )
    if len(kept_rows) > 20:
        lines.append(f"... {len(kept_rows) - 20:,} more kept.")
        lines.append("")
    if dropped_rows:
        lines += ["Dropped, in the order the ranking gives (ties broken by node id):", ""]
        lines += _table(
            ["rank", "node", simplification.method],
            [[str(row.rank), str(row.node), _num(row.value, 4)] for row in dropped_rows[:10]],
        )
        if len(dropped_rows) > 10:
            lines.append(f"... {len(dropped_rows) - 10:,} more dropped.")
            lines.append("")
    if simplification.notes:
        lines += [f"> {note}" for note in simplification.notes] + [""]
    return lines


def _render_influence(influence: InfluenceSummary) -> list[str]:
    lines = [
        "## Influence-based summary (§46.4)",
        "",
        "**Null model.** none: this is a simulation of a what-if, not a significance test.",
        "",
        f"Model: {influence.model}, beta={_num(influence.beta, 2)}, "
        f"attempts={influence.attempts}, runs={influence.runs}, seed="
        f"{influence.seed if influence.seed is not None else 'not fixed'}, "
        f"{influence.candidates} candidate seed(s) tried.",
        "",
        f"Edges kept: {influence.kept_edges:,} of {influence.total_edges:,} "
        f"({influence.edge_share:.1%}) -- the union of every edge any candidate's simulated "
        "cascade used.",
        "",
        "Seeds ranked by how fast their own cascade saturated the nodes it could reach:",
        "",
    ]
    lines += _table(
        ["rank", "node", "steps to saturate", "reach", "saturated"],
        [
            [
                str(row.rank),
                str(row.node),
                _num(row.steps_to_saturate, 2) if row.steps_to_saturate is not None else "-",
                f"{row.reach_share:.0%}",
                f"{row.saturated_share:.0%}",
            ]
            for row in influence.ranked
        ],
    )
    lines += [f"> {influence.caveat}", ""]
    if influence.notes:
        lines += [f"> {note}" for note in influence.notes] + [""]
    return lines


def _jsonable(value: float | None) -> float | None:
    """`nan` and `None` both travel as JSON `null`; "undefined" is never 0.0."""
    if value is None:
        return None
    return None if math.isnan(value) else value


def summary_payload(report: SummaryReport) -> dict[str, Any]:
    """The same report as plain JSON-able data."""
    payload: dict[str, Any] = {
        "implements": f"Atlas {CHAPTER} (graph summarization)",
        "frame": report.frame,
        "by": report.by,
        "nodes": report.nodes,
        "edges": report.edges,
        "notes": report.notes,
        "aggregation": {
            "left_out": report.aggregation.left_out,
            "borrowed_share": _jsonable(report.aggregation.borrowed_share),
            "supernodes": [
                {
                    "index": node.index + 1,
                    "label": node.label,
                    "size": node.size,
                    "internal_edges": node.internal_edges,
                    "possible_edges": node.possible_edges,
                    "internal_density": _jsonable(node.internal_density),
                }
                for node in report.aggregation.supernodes
            ],
            "superedges": [
                {
                    "source": report.aggregation.supernodes[edge.source].label,
                    "target": report.aggregation.supernodes[edge.target].label,
                    "edges": edge.edges,
                    "possible": edge.possible,
                    "density": _jsonable(edge.density),
                }
                for edge in report.aggregation.superedges
            ],
            "notes": report.aggregation.notes,
        },
        "compression": {
            "groups": report.compression.groups,
            "declared_pairs": report.compression.declared_pairs,
            "possible_pairs": report.compression.possible_pairs,
            "partition_bits": _jsonable(report.compression.partition_bits),
            "model_bits": _jsonable(report.compression.model_bits),
            "correction_bits": _jsonable(report.compression.correction_bits),
            "mistakes": report.compression.mistakes,
            "total_bits": _jsonable(report.compression.total_bits),
            "raw_bits": _jsonable(report.compression.raw_bits),
            "ratio": _jsonable(report.compression.ratio),
            "notes": report.compression.notes,
        },
        "simplification": {
            "method": report.simplification.method,
            "keep": report.simplification.keep,
            "nodes_before": report.simplification.nodes_before,
            "nodes_after": report.simplification.nodes_after,
            "edges_before": report.simplification.edges_before,
            "edges_after": report.simplification.edges_after,
            "ranked": [
                {
                    "rank": row.rank,
                    "node": str(row.node),
                    "value": _jsonable(row.value),
                    "kept": row.kept,
                }
                for row in report.simplification.ranked
            ],
            "notes": report.simplification.notes,
        },
        "influence": {
            "model": report.influence.model,
            "beta": report.influence.beta,
            "attempts": report.influence.attempts,
            "runs": report.influence.runs,
            "seed": report.influence.seed,
            "candidates": report.influence.candidates,
            "kept_edges": report.influence.kept_edges,
            "total_edges": report.influence.total_edges,
            "edge_share": report.influence.edge_share,
            "caveat": report.influence.caveat,
            "ranked": [
                {
                    "rank": row.rank,
                    "node": str(row.node),
                    "steps_to_saturate": _jsonable(row.steps_to_saturate),
                    "reach_share": row.reach_share,
                    "saturated_share": row.saturated_share,
                }
                for row in report.influence.ranked
            ],
            "notes": report.influence.notes,
        },
    }
    if report.evaluation is not None:
        payload["evaluation"] = evaluation_payload(report.evaluation)
    return payload
