"""Does this network sort by something we already know about its nodes?

Community detection asks the graph where its groups are. An attribute asks a different question:
here is a property somebody recorded about each node -- which region a speaker is in, which half
of the corpus a document belongs to -- does the network divide along it? The two questions are
easy to confuse and the confusion is expensive, because a partition handed to a graph always
produces a number, and the number always looks like a finding.

So every measure here comes with what it would have been by chance:

*Assortativity* is the correlation between the labels at the two ends of an edge: 1 when every
edge joins like to like, 0 when the labels are spread as a coin would spread them, negative when
the network joins unlike to unlike. Its null is a permutation: the same graph, the same labels,
shuffled between the nodes, many times. That null holds the network and the label distribution
fixed and varies only which node got which label, which is exactly the claim under test.

*Modularity of the attribute partition* asks the same thing in the currency Louvain reports, so
the two can be read side by side: how much better does grouping by this attribute explain the
edges than a degree-preserving random graph, and how far short of the best grouping Louvain can
find does it fall. A partition that scores near Louvain's best *is* the community structure; one
that scores well above its own null but far below Louvain has found something real that is not
the main division in the network.

The degree-preserving null here scores the *same* fixed partition on each rewired graph, unlike
:func:`graphrag.sna.cluster.null_model_modularity`, which re-runs Louvain on every sample. The
difference is not an oversight. Louvain's partition is fitted to the graph, so transferring it to
another graph measures nothing but transfer; an attribute partition is fitted to nothing, so the
honest question is whether these labels explain these edges better than the same labels would
explain a random graph with the same degrees.

*Agreement with the clustering* (adjusted Rand index and normalised mutual information) says
whether the groups the run found are the attribute in disguise. The adjusted index is corrected
for chance; the mutual information is not, so a random baseline is computed for both by shuffling
the cluster labels.

*Where the labels came from* decides whether any of the above is evidence about the attribute.
Every network in this package draws its edges from co-membership in documents: two entities are
joined because a passage names both, two topics because a document carries both. A label the
node carries in its own right -- a speaker an attribution pass tagged, an entity an extraction
pass tagged -- is independent of that, so correlating the two is a real question. A label
*borrowed* from those same documents is not: the edge and both its endpoints' labels have one
source, so the edges join like to like because of how the label was assigned, and the number
that comes out measures the projection.

The permutation null does not catch this, and it is worth being precise about why. The null
shuffles labels across a fixed graph, so it destroys exactly the correlation the borrowing
creates; the observed value keeps it. A confounded attribute therefore produces a large
z-score -- it produces the largest z-scores in the corpus -- and looks like the strongest finding
on the page. So the share of borrowed labels is counted, reported next to the assortativity, and
said out loud in the verdict rather than left for a reader to infer.

*The null that does catch it.* ``null="bipartite"`` goes back to what was actually observed --
the memberships, not the projection -- rewires them so that every node keeps its number of
documents and every document keeps its size (§18.1, via
:func:`graphrag.sna.null.bipartite_preserving`), projects again, and then **derives the borrowed
labels again from the rewired corpus**, by the same majority rule
:func:`graphrag.sna.export._label_attributes` used on the real one. Labels a node owns stay
pinned, because those really are properties of the thing.

That puts the circularity inside the null instead of inside the observation. In a sample, two
entities that share a document are still handed that document's value, so like still joins like
by construction -- and the observed network only stands out if it sorts harder than the borrowing
mechanism does on its own. The permutation null prices that mechanism at zero, which is why a
confounded attribute scores its largest z-score there; this one prices it at whatever it is
worth. It also prices the corpus's shape: the clique a single long document creates, and the fact
that the nodes recorded everywhere meet each other constantly, so an attribute that is really
"was recorded a lot" lands inside it too.

What it is not: a test of whether the tag is *true*. It answers "would a label distribution like
this one arise from rewired documents?", which is a question about how the labels were made, and
a borrowed label that survives it is still a label nobody recorded about the node. The two
z-scores are therefore printed side by side rather than one replacing the other, and the
borrowed-label warning stays either way.

One caveat runs through all of it and is printed with the numbers: an attribute is a hypothesis
about structure, not a finding. Nodes nobody tagged are not a third group -- they are outside
every measure here, and the report says how many they were.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import networkx as nx

from graphrag.sna.cluster import louvain
from graphrag.sna.export import (
    INHERITED,
    OWN,
    RIGHT_ATTRIBUTES,
    attr_count_key,
    attr_key,
    attr_source_key,
)
from graphrag.sna.matrices import Side
from graphrag.sna.null import (
    HOLDS_FIXED,
    Significance,
    bipartite_preserving,
    configuration,
    label_permutation,
    pairs_of,
    projected_side,
    significance,
)
from graphrag.sna.projection import DEFAULT_LAMBDA, PROJECTION, PROJECTION_LAMBDA
from graphrag.sna.stats import compare_partitions, mean, std, z_score

__all__ = [
    "ATTRIBUTE_NULLS",
    "CONTAGION_CAVEAT",
    "AttributeReport",
    "HomophilyIndices",
    "ValueHomophily",
    "analyse_attribute",
    "attribute_labels",
    "attribute_payload",
    "attribute_sources",
    "coleman_index",
    "ei_index",
    "ei_ratio",
    "homophily_indices",
    "render_attribute",
]

#: How many times the labels are shuffled for the assortativity null.
PERMUTATIONS = 200
#: How many degree-preserving rewirings the fixed-partition null uses.
NULL_SAMPLES = 50
#: How many Louvain seeds the comparison partition is the best of.
LOUVAIN_RUNS = 5
#: Below this many labelled nodes nothing here is worth reporting as a measurement.
MIN_NODES = 4
#: Which null the assortativity is measured against. ``permutation`` shuffles the labels over a
#: fixed graph; ``bipartite`` rewires the two-mode memberships the network was projected from,
#: projects again, and lets the labels follow their provenance -- owned values pinned, borrowed
#: ones derived again from the rewired corpus. They hold different things fixed and therefore test
#: different claims -- see :func:`analyse_attribute` -- and the default stays the shuffle because
#: every network can be shuffled and only a projection has memberships to rewire.
ATTRIBUTE_NULLS: tuple[str, ...] = ("permutation", "bipartite")

#: Above this share of borrowed labels, the assortativity is reported as confounded rather than
#: as a reading of the attribute. Set at half because that is the point where the shuffle null
#: is measuring the borrowing rather than the attribute for most of the edges; below it the
#: share is still reported, because the confound is a matter of degree and never of kind.
CONFOUNDED_SHARE = 0.5

#: How negative an EI index has to be, and how near zero the same value's Coleman index, before
#: the report says the value looks homophilous only because of its size (§30.2, p. 434). Both are
#: conventions of this package: the book states the failure and gives no threshold for it.
SIZE_DRIVEN_EI = 0.2
SIZE_DRIVEN_COLEMAN = 0.1

#: What §30.4 says about every number in this module, printed with them. The chapter's own
#: sentence is the one that makes it unavoidable: *"if we observe a strong homophily, it could be
#: because our social connections are influencing us into adopting behaviors we would not
#: otherwise"* (p. 437). Selection and influence leave the same cross-section, and the evidence
#: the chapter itself cites for influence is longitudinal -- Christakis and Fowler *"looked at
#: health indicators from thousands of people in a community over 32 years"* (p. 438).
CONTAGION_CAVEAT = (
    "Homophily or contagion: this number cannot tell them apart. §30.4's own sentence is that "
    '"if we observe a strong homophily, it could be because our social connections are '
    'influencing us into adopting behaviors we would not otherwise" (p. 437) -- nodes that were '
    "already alike coming together, and nodes that came together becoming alike, leave the same "
    "snapshot behind. Separating them needs what the chapter's own evidence uses, thousands of "
    "people followed over 32 years (p. 438): a dated edge list with each node's value as it was "
    "*before* the tie formed. Half of that exists here -- `--window` and "
    "`graphrag.sna.layers.dynamic_edges` date the edges -- and the other half does not: an "
    "attribution or extraction sidecar records what a node is now, with no history, so the "
    "attribute is re-read as constant over every window. Until it is dated, report this as "
    "association and not as either mechanism."
)


@dataclass(frozen=True)
class ValueHomophily:
    """One value of one attribute, counted against its own size (§30.2)."""

    value: str
    nodes: int
    share: float
    """The value's share of the labelled nodes: what Coleman's index compares the ties with."""
    internal: int
    """Edges with both ends in this value."""
    external: int
    """Edges with exactly one end in it."""
    ei: float | None
    """``(E - I) / (E + I)`` for this value: -1 when it only ties to itself, +1 when it never
    does, ``None`` when the value has no edge at all."""
    observed: float
    """The share of this value's tie-ends that stay inside it, ``2I / (2I + E)``."""
    expected: float
    """The share a random graph on these nodes would give it, ``(n_i - 1) / (n - 1)``: the
    population share of the partners available to it, itself excluded."""
    coleman: float | None
    """Coleman's homophily index for this value: see :func:`coleman_index`."""


@dataclass(frozen=True)
class HomophilyIndices:
    """§30.2's count-based indices for one attribute: EI overall and per value, and Coleman's.

    These are counts, not correlations, and that is the point of printing them next to the
    assortativity: the chapter warns that a count-based reading *"breaks down if you have more
    than two possible values for your attribute, and also if some values are more popular than
    others"* (p. 434), and Coleman's index is the one here that prices the popularity in. A
    report that carries EI alone will call a majority group homophilous for being large.
    """

    key: str
    nodes: int
    edges: int
    internal: int
    external: int
    ei: float | None
    permutations: int = 0
    permuted_mean: float = 0.0
    permuted_std: float = 0.0
    ei_z: float = 0.0
    values: tuple[ValueHomophily, ...] = ()

    @property
    def size_driven(self) -> tuple[str, ...]:
        """Values whose EI reads as homophily while Coleman's index says it is size (§30.2).

        The book's warning made into a check: a value with a strongly negative EI and a Coleman
        index near zero is tying inside itself exactly as often as a random graph would, and the
        EI is negative only because most of the nodes available to it are its own.
        """
        return tuple(
            row.value
            for row in self.values
            if row.ei is not None
            and row.coleman is not None
            and row.ei < -SIZE_DRIVEN_EI
            and abs(row.coleman) < SIZE_DRIVEN_COLEMAN
        )

    @property
    def reading(self) -> str:
        """What the two indices say together, which is more than either says alone."""
        if self.ei is None:
            return (
                "No edge joins two labelled nodes, so there is nothing to count inside or "
                "between values and the EI index is undefined."
            )
        direction = (
            "inside values"
            if self.ei < 0
            else (
                "between values, which is the **heterophily** §30.4 names -- the love of the "
                "different -- and what the chapter's summary calls disassortativity: the values "
                "are connected through each other rather than within themselves, as a dating "
                "network is by gender"
            )
        )
        base = (
            f"EI {self.ei:+.4f} over {self.edges:,} edge(s) between labelled nodes: most edges "
            f"run {direction}. The label shuffle puts it at {self.permuted_mean:+.4f} "
            f"(sd {self.permuted_std:.4f}, {self.permutations} shuffles), so the observed value "
            f"is {self.ei_z:+.2f} null standard deviations from it -- and that z, not the index, "
            "is the part that is about the attribute."
        )
        if self.size_driven:
            return base + (
                " Read the per-value table before quoting the overall index: "
                + ", ".join(f"`{value}`" for value in self.size_driven)
                + " has a negative EI and a Coleman index near zero, which is §30.2's warning "
                "exactly -- a majority looks homophilous by size alone, because most of the "
                "nodes available to it are its own kind."
            )
        return base


@dataclass(frozen=True)
class AttributeReport:
    """One attribute measured against one network, with the nulls that make it readable."""

    key: str
    counts: dict[str, int] = field(default_factory=dict)
    labelled: int = 0
    total: int = 0
    mixed: int = 0
    edges_within: int = 0
    edges_between: int = 0
    weight_within: float = 0.0
    weight_between: float = 0.0
    assortativity: float | None = None
    permutations: int = 0
    permuted_mean: float = 0.0
    permuted_std: float = 0.0
    assortativity_z: float = 0.0
    modularity: float = 0.0
    louvain_modularity: float | None = None
    null_samples: int = 0
    null_mean: float = 0.0
    null_std: float = 0.0
    modularity_z: float = 0.0
    group_noun: str = "cluster"
    agreement: dict[str, float] = field(default_factory=dict)
    #: How many labelled nodes carry the attribute in their own right, and how many borrowed it
    #: from the documents their passages sit in. Keyed by :data:`graphrag.sna.export.OWN` and
    #: :data:`graphrag.sna.export.INHERITED`.
    sources: dict[str, int] = field(default_factory=dict)
    #: Of the borrowed labels, how many came from a node with exactly one document. Those are
    #: the fully circular ones: every edge such a node has sits inside the document that also
    #: handed it its label.
    single_document: int = 0
    #: Which null was asked for, one of :data:`ATTRIBUTE_NULLS`. The permutation null is computed
    #: either way, because it is the one every network can produce.
    null: str = "permutation"
    #: The same observed assortativity against the bipartite-preserving null, when it was asked
    #: for and the network carried the memberships it was projected from.
    bipartite: Significance | None = None
    #: How many two-mode memberships that null rewired: the sampling frame of the section.
    bipartite_pairs: int = 0
    #: Whether that null re-derived the borrowed labels from each rewired corpus rather than
    #: carrying them across. True only when some label was borrowed *and* the builder recorded
    #: what the opposite-mode nodes were tagged with, which is what makes the null able to price
    #: the borrowing in. When it is false with borrowed labels present, the section says so.
    bipartite_rederived: bool = False
    #: §30.2's two count-based indices beside the coefficient: the EI index overall and per
    #: value, and Coleman's index, which is the one that knows how common each value is. ``None``
    #: when there was too little to measure -- the same condition that leaves ``assortativity``
    #: unset.
    homophily: HomophilyIndices | None = None
    notes: tuple[str, ...] = ()

    @property
    def values(self) -> int:
        return len(self.counts)

    @property
    def own(self) -> int:
        """Labelled nodes whose value was recorded about the node itself."""
        return self.sources.get(OWN, 0)

    @property
    def inherited(self) -> int:
        """Labelled nodes whose value was borrowed from their documents."""
        return self.sources.get(INHERITED, 0)

    @property
    def inherited_share(self) -> float:
        return self.inherited / self.labelled if self.labelled else 0.0

    @property
    def confounded(self) -> bool:
        """Whether enough labels were borrowed to make the correlation circular.

        The edges of every network here come from documents. A borrowed label comes from the
        same documents, so past this share the assortativity is largely a statement about how
        the labels were assigned. See this module's docstring.
        """
        return self.inherited_share > CONFOUNDED_SHARE

    @property
    def group_plural(self) -> str:
        return "communities" if self.group_noun == "community" else f"{self.group_noun}s"

    @property
    def unlabelled(self) -> int:
        return max(self.total - self.labelled, 0)

    @property
    def within_share(self) -> float:
        total = self.edges_within + self.edges_between
        return self.edges_within / total if total else 0.0

    @property
    def verdict(self) -> str:
        """What the assortativity and its permutation null together support, in one sentence."""
        if self.assortativity is None:
            return (
                "Not measurable: fewer than two values, or no edges between labelled nodes. "
                "Report the counts and stop."
            )
        if self.confounded:
            return (
                f"Not a reading of this attribute: {self.inherited:,} of {self.labelled:,} "
                f"labelled nodes ({self.inherited_share:.0%}) took their value from the very "
                "documents the edges were drawn from, so like joins like by construction and "
                "the shuffle null cannot subtract it. Record the attribute on the nodes "
                "themselves before reading anything off this number."
            )
        if abs(self.assortativity_z) < 2:
            return (
                "The labels are spread across the edges about as a shuffle would spread them, "
                "so this network does not divide along this attribute."
            )
        if self.assortativity > 0:
            return (
                "Edges join like to like far more than a shuffle of the same labels does, so "
                "the attribute tracks a real division in the network."
            )
        return (
            "Edges join unlike to unlike far more than a shuffle does: the two values are "
            "connected through each other rather than within themselves."
        )


def attribute_labels(graph: nx.Graph, key: str) -> dict[str, str]:
    """Each node's value for one attribute, leaving out the nodes that carry none."""
    data = attr_key(key)
    return {
        node: str(values[data])
        for node, values in graph.nodes(data=True)
        if str(values.get(data, "")).strip()
    }


def attribute_sources(graph: nx.Graph, key: str) -> dict[str, str]:
    """Where each labelled node's value came from: :data:`OWN` or :data:`INHERITED`.

    An unmarked label is read as the node's own. Borrowing is the case that has to be declared,
    and the only thing that borrows is :func:`graphrag.sna.export._label_attributes`, which
    always marks what it borrowed; a graph arrives here either from that builder or from a
    caller who put the value on the node deliberately. Reading silence as borrowed instead would
    make every hand-built network report itself as confounded, which would teach a reader to
    scroll past the one warning on the page that is never decorative.
    """
    data = attr_source_key(key)
    return {
        node: INHERITED if str(values.get(data, "")).strip() == INHERITED else OWN
        for node, values in graph.nodes(data=True)
        if str(values.get(attr_key(key), "")).strip()
    }


# ------------------------------------------------------- §30.2: counting edges instead of
# ------------------------------------------------------- correlating labels


def ei_ratio(internal: float, external: float) -> float | None:
    """The EI index of one pair of counts: ``(E - I) / (E + I)``.

    -1 when every tie counted stays inside the group, +1 when every one leaves it, 0 at an even
    split. ``None`` when there is no tie at all, which is undefined and not zero: a group with no
    edges is not evenly mixed.

    **What the book defines and what this adds.** §30.2 works in these counts -- its worked
    example is *"20 edges connecting nodes with the same color over 22 total edges"* (p. 434) --
    but names no index over them. This is Krackhardt and Stern's E-I index (*Informal networks
    and organizational crises*, Social Psychology Quarterly 51(2):123-140, 1988), which is that
    ratio with a sign: **negative means homophily**, which is the opposite sign convention from
    the assortativity r that §30.2 goes on to define, so the two are never printed without saying
    which is which.
    """
    total = internal + external
    return (external - internal) / total if total else None


def _split(graph: nx.Graph, labels: Mapping[str, str], value: str | None) -> tuple[int, int]:
    """Edges inside and edges leaving, over the labelled nodes only.

    With ``value``, the counts are that value's: ``I`` is edges with both ends in it and ``E``
    edges with exactly one. Without, they are the whole network's: ``I`` is edges joining two
    nodes of the same value, whichever, and ``E`` edges joining two different ones. Unlabelled
    nodes are in neither, as everywhere else in this module.

    Counts, not weights. The book counts edges, and a weighted EI would answer a different
    question (how much of what was written down stayed inside a value); the weight split is in
    :class:`AttributeReport` already, for whoever wants it.
    """
    internal = external = 0
    for u, v in graph.edges():
        left, right = labels.get(u), labels.get(v)
        if left is None or right is None:
            continue
        if value is None:
            internal, external = (
                (internal + 1, external) if left == right else (internal, external + 1)
            )
            continue
        if left == value and right == value:
            internal += 1
        elif left == value or right == value:
            external += 1
    return internal, external


def ei_index(
    graph: nx.Graph, labels: Mapping[str, str], *, value: str | None = None
) -> float | None:
    """§30.2's edge counts as one number, for the network or for one value.

    Without ``value``: ``(between - within) / (between + within)`` over every edge joining two
    labelled nodes. With one: the same over the edges that touch that value, so ``I`` is the
    edges inside it and ``E`` the edges leaving it.

    **Negative is homophily.** The index is a ratio of counts and knows nothing about how common
    each value is, which is precisely the failure §30.2 names: *"This approach breaks down if you
    have more than two possible values for your attribute, and also if some values are more
    popular than others. In these cases, you might conclude that there is assortativity in a
    non-assortative network, simply because you're assuming the incorrect null model of equal
    attribute value popularity"* (p. 434). A group holding 90% of the nodes has 89 in-group
    partners for every 10 out-group ones before anybody chooses anything, so its EI is strongly
    negative in a graph wired at random. :func:`coleman_index` is the companion that divides that
    out, and :func:`homophily_indices` computes the label-shuffle null that does the same job
    from the other side.

    ``None`` when no edge joins two labelled nodes, or none touches ``value``.
    """
    internal, external = _split(graph, labels, value)
    return ei_ratio(internal, external)


def coleman_index(
    graph: nx.Graph, labels: Mapping[str, str], *, value: str | None = None
) -> dict[str, float]:
    """Each value's in-group ties against the share of the population it could have tied to.

    For a value *i*, with ``s_i`` the share of the tie-ends of *i*'s nodes that land on other
    *i* nodes and ``w_i`` the share of the available partners that are *i*:

    ``H_i = (s_i - w_i) / (1 - w_i)`` when ``s_i >= w_i``, and ``(s_i - w_i) / w_i`` otherwise.

    It runs from -1 (never ties inside the group) through 0 (ties inside it exactly as often as
    its size implies) to +1 (ties only inside it). That zero is what makes it the answer to
    §30.2's popularity warning: EI calls a 90% majority homophilous in a random graph, Coleman
    calls it nothing.

    **What the book defines and what this adds.** The chapter's own fix for unequal popularity is
    Newman's assortativity r, which this module already reports. This is the sociological
    alternative it does not name (James S. Coleman, *Relational analysis: the study of social
    organizations with survey methods*, Human Organization 17(4):28-36, 1958), kept because it is
    per value rather than per network -- r says whether the network sorts, Coleman says which
    group does the sorting. ``w_i`` is taken as ``(n_i - 1) / (n - 1)``, excluding the node
    itself from the partners available to it, which is the value that makes a random graph score
    0 exactly rather than nearly.

    Returns one entry per value that has at least one edge, or just ``value``'s when it is given.
    A value with no edge at all is absent rather than 0.0: nothing was chosen, so nothing is
    measured. Undirected counting throughout -- an edge inside the value is two tie-ends of it,
    an edge leaving it is one.
    """
    counts = Counter(labels[node] for node in graph if node in labels)
    total = sum(counts.values())
    wanted = [value] if value is not None else sorted(counts)
    indices: dict[str, float] = {}
    for name in wanted:
        size = counts.get(name, 0)
        internal, external = _split(graph, labels, name)
        ends = 2 * internal + external
        if not size or not ends or total <= 1:
            continue
        observed = 2 * internal / ends
        expected = (size - 1) / (total - 1)
        if expected >= 1.0:
            continue
        indices[name] = (
            (observed - expected) / (1 - expected)
            if observed >= expected
            else (observed - expected) / expected
        )
    return indices


def homophily_indices(
    graph: nx.Graph,
    key: str,
    *,
    permutations: int = PERMUTATIONS,
    seed: int | random.Random | None = None,
) -> HomophilyIndices:
    """§30.2's two count-based indices for one attribute, with the label shuffle behind the EI.

    The EI index is a ratio of counts, so on its own it says as much about the sizes of the
    groups as about the network (p. 434). Two things are printed beside it to stop that being
    read as a finding: Coleman's index per value, which divides the size out arithmetically, and
    the same label shuffle the assortativity uses (:func:`graphrag.sna.null.label_permutation`,
    §19.1), which divides it out empirically -- the shuffled EI carries the group sizes and not
    the sorting, so the gap between the observed EI and the shuffled one is the sorting alone.

    The shuffles here are drawn independently of the assortativity's, from the same generator and
    the same null family, so the two z-scores are comparable in kind but not paired sample by
    sample.
    """
    labels = attribute_labels(graph, key)
    counts = Counter(labels.values())
    total = sum(counts.values())
    internal, external = _split(graph, labels, None)
    colemans = coleman_index(graph, labels)
    observed = ei_ratio(internal, external)
    permuted: list[float] = []
    if observed is not None:
        data = attr_key(key)
        for sample in label_permutation(graph, max(permutations, 0), key=data, seed=seed):
            score = ei_index(sample, attribute_labels(sample, key))
            if score is not None:
                permuted.append(score)
    rows = []
    for name, size in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        inside, leaving = _split(graph, labels, name)
        ends = 2 * inside + leaving
        rows.append(
            ValueHomophily(
                value=name,
                nodes=size,
                share=size / total if total else 0.0,
                internal=inside,
                external=leaving,
                ei=ei_ratio(inside, leaving),
                observed=2 * inside / ends if ends else 0.0,
                expected=(size - 1) / (total - 1) if total > 1 else 0.0,
                coleman=colemans.get(name),
            )
        )
    return HomophilyIndices(
        key=key,
        nodes=total,
        edges=internal + external,
        internal=internal,
        external=external,
        ei=observed,
        permutations=len(permuted),
        permuted_mean=mean(permuted),
        permuted_std=std(permuted),
        ei_z=z_score(observed, permuted) if observed is not None else 0.0,
        values=tuple(rows),
    )


def analyse_attribute(
    graph: nx.Graph,
    key: str,
    *,
    groups: Sequence[Sequence[str]] = (),
    group_noun: str = "cluster",
    resolution: float = 1.0,
    permutations: int = PERMUTATIONS,
    samples: int = NULL_SAMPLES,
    runs: int = LOUVAIN_RUNS,
    null: str = "permutation",
    seed: int | None = None,
) -> AttributeReport:
    """Measure how far one node attribute explains this network's edges.

    Everything is computed over the nodes that carry a value, never over the whole network: a
    node nobody tagged has no label to correlate, and quietly treating it as a value of its own
    would make "untagged" the biggest community in most corpora.

    ``null`` picks the *second* null for the assortativity; the label permutation is computed
    either way, because it is the only null a graph without memberships can produce.

    ``permutation``
        the shuffle of §19.1 applied to the labels: the network and the mix of values are held
        fixed and only which node holds which value moves. It tests whether this pairing of
        labels with positions is one the same labels could have made by accident.
    ``bipartite``
        the two-mode memberships the network was projected from are rewired instead, keeping
        every node's number of documents and every document's size (curveball, §18.1), and the
        network is projected again. It tests the same claim against a family of *plausible
        corpora* rather than against one fixed graph, so the cliques a single document creates,
        the sizes of the documents, and how much each node was recorded are inside the null
        instead of inside the observation.

        Labels move with the corpus, by provenance. A label the node owns is pinned -- it is a
        property of the thing, so it travels. A **borrowed** label is derived again from the
        rewired corpus, by the same majority rule the builder used on the real one, so the
        borrowing mechanism itself is inside the null: in a sample, two nodes that share a
        document are still handed that document's value. That is what a label shuffle cannot do,
        and it is why a confounded attribute scores its *largest* z-score against the shuffle and
        an ordinary one here.

        Two things it is still not. It is not a test of whether the tag is true -- it asks
        whether a label distribution like this one would arise from rewired documents, which is a
        question about how the labels were made -- so the borrowed-label warning stays whatever
        it says. And re-deriving needs the table of what the opposite-mode nodes were tagged
        with, which only the entity network records (``export.RIGHT_ATTRIBUTES``); elsewhere the
        labels stay pinned and the section says so.

        It needs the memberships, which only a projected network carries
        (:func:`graphrag.sna.null.pairs_of`), and raises ``ValueError`` when they are absent --
        the topic network reads stored edges and never went through a projection, and a graph
        built by hand has no corpus behind it.
    """
    if null not in ATTRIBUTE_NULLS:
        msg = f"null must be one of {', '.join(ATTRIBUTE_NULLS)}, got {null!r}"
        raise ValueError(msg)
    memberships = _memberships_or_refuse(graph, null)
    labels = attribute_labels(graph, key)
    counts = Counter(labels.values())
    sources = attribute_sources(graph, key)
    notes: list[str] = []
    report = AttributeReport(
        key=key,
        counts=dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        labelled=len(labels),
        total=graph.number_of_nodes(),
        mixed=sum(
            1 for node in labels if int(graph.nodes[node].get(attr_count_key(key), 1) or 1) > 1
        ),
        sources=dict(Counter(sources[node] for node in labels)),
        single_document=sum(
            1
            for node in labels
            if sources[node] == INHERITED and int(graph.nodes[node].get("documents", 0) or 0) == 1
        ),
        group_noun=group_noun,
        null=null,
        bipartite_pairs=len(memberships[0]) if memberships else 0,
        notes=tuple(notes),
    )
    if not labels:
        return _with_notes(
            report,
            [
                f"No node in this network carries '{key}'. Either the tagging pass has not run "
                "or the key is spelled differently in the sidecars."
            ],
        )
    sub: nx.Graph = graph.subgraph(labels).copy()
    within, between, w_within, w_between = _edge_split(sub, labels)
    report = _replace(
        report,
        edges_within=within,
        edges_between=between,
        weight_within=w_within,
        weight_between=w_between,
    )
    if len(labels) < MIN_NODES or len(counts) < 2 or sub.number_of_edges() == 0:
        notes.append(
            f"{len(labels)} labelled node(s) over {len(counts)} value(s) with "
            f"{sub.number_of_edges()} edge(s) between them: too little to measure. The value "
            "counts above are the whole finding."
        )
        return _with_notes(report, notes)

    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    observed, permuted = _assortativity(sub, key, permutations, rng)
    groups_by_value = [
        sorted(n for n, v in labels.items() if v == value) for value in sorted(counts)
    ]
    modularity = float(
        nx.community.modularity(sub, groups_by_value, weight="weight", resolution=resolution)
    )
    best = louvain(sub, resolution=resolution, seed=seed, runs=runs)
    fixed = _fixed_partition_null(sub, groups_by_value, samples, resolution, rng)
    report = _replace(
        report,
        assortativity=observed,
        permutations=len(permuted),
        permuted_mean=mean(permuted),
        permuted_std=std(permuted),
        assortativity_z=z_score(observed, permuted) if observed is not None else 0.0,
        modularity=modularity,
        louvain_modularity=best.modularity,
        null_samples=len(fixed),
        null_mean=mean(fixed),
        null_std=std(fixed),
        modularity_z=z_score(modularity, fixed),
        agreement=_agreement(groups, labels, rng),
        # §30.2's count-based pair, beside the correlation-based one: the EI index with the same
        # shuffle null, and Coleman's index, which is what stops a large value being read as a
        # homophilous one (p. 434). Computed over the same labelled subgraph as everything above.
        homophily=homophily_indices(sub, key, permutations=permutations, seed=rng),
    )
    if memberships is not None and observed is not None:
        rederived = bool(memberships.right_attributes) and report.inherited > 0
        report = _replace(
            report,
            bipartite=significance(
                observed,
                _bipartite_null(memberships, key, labels, sources, samples, rng),
                null="bipartite_preserving",
                # Two-sided, because an attribute can divide a network either way and the
                # verdict above already reads both directions: joining like to like far more
                # than the null, and far less, are both findings about the attribute.
                tail="two",
            ),
            bipartite_rederived=rederived,
        )
        notes.append(_bipartite_note(report))
        notes.extend(_pruned_note(graph))
    if report.inherited:
        notes.append(
            f"{report.inherited} of {report.labelled} labelled node(s) "
            f"({report.inherited_share:.0%}) carry no value of their own for '{key}' and were "
            "given the one most of their documents carry. Those labels come from the same "
            "documents as the edges, so they raise the assortativity whether or not the "
            "attribute divides anything"
            + (
                f"; {report.single_document} of them sit in a single document, where every "
                "edge is inside the document that supplied the label."
                if report.single_document
                else "."
            )
        )
    if report.mixed:
        notes.append(
            f"{report.mixed} of {report.labelled} labelled node(s) had passages in documents "
            "that disagreed about this attribute; each was given the value most of its "
            "documents carry. Read those nodes as a majority, not as a fact."
        )
    if report.unlabelled:
        notes.append(
            f"{report.unlabelled} of {report.total} node(s) carry no value for '{key}' and are "
            "in none of these measures. They are untagged, not a third value."
        )
    return _with_notes(report, notes)


class Memberships(NamedTuple):
    """What the bipartite null needs from a projected network to rewire the corpus behind it."""

    pairs: list[tuple[str, str]]
    side: Side
    """Which mode the network is the projection of, so the null projects the same way."""
    right_attributes: dict[str, dict[str, str]]
    """What each opposite-mode node was tagged with, empty when the builder recorded none. This
    is what turns a borrowed label from something the null carries along into something the null
    re-derives: see :func:`_sample_labels`."""
    scheme: str = "simple"
    """Which of chapter 26's weighting schemes the observed projection used, so the null
    re-projects the same way. Re-projecting under another would compare the observation against
    a different definition of an edge weight rather than against a rewired corpus."""
    lam: float = DEFAULT_LAMBDA
    """The hybrid scheme's λ, for the same reason."""


def _memberships_or_refuse(graph: nx.Graph, null: str) -> Memberships | None:
    """The two-mode memberships the bipartite null needs, or ``None`` when it was not asked for.

    Refused loudly rather than quietly downgraded to the permutation null: a report that said
    "bipartite" and printed a shuffle would be worse than no report. What is missing in each case
    is named, because the fix differs -- a topic network has no projection behind it at all, while
    a hand-built graph simply never recorded one.
    """
    if null != "bipartite":
        return None
    pairs = pairs_of(graph)
    if pairs is None:
        msg = (
            "null='bipartite' needs the two-mode memberships this network was projected from, "
            "and this graph carries none. Only a projected network has them (speakers, entities, "
            "or speakers-entities with --project); the topic network reads stored co-occurrence "
            "edges and was never projected. Use null='permutation' there."
        )
        raise ValueError(msg)
    side = projected_side(graph, pairs)
    if side is None:
        msg = (
            "null='bipartite' could not tell which mode this graph is the projection of: its "
            "nodes are in neither side of the memberships it carries. Rebuild the network rather "
            "than relabelling a projected one."
        )
        raise ValueError(msg)
    table = graph.graph.get(RIGHT_ATTRIBUTES) or {}
    return Memberships(
        pairs=pairs,
        side=side,
        right_attributes={
            str(node): {str(k): str(v) for k, v in values.items()} for node, values in table.items()
        },
        scheme=str(graph.graph.get(PROJECTION) or "simple"),
        lam=float(graph.graph.get(PROJECTION_LAMBDA, DEFAULT_LAMBDA)),
    )


def _bipartite_null(
    memberships: Memberships,
    key: str,
    labels: Mapping[str, str],
    sources: Mapping[str, str],
    samples: int,
    rng: random.Random,
) -> list[float]:
    """The same assortativity, measured on re-projections of rewired memberships (§18.1).

    Each sample rewires who was recorded in what, keeping every node's number of documents and
    every document's size, and projects again. Two things then happen to the labels, and which
    one happens to a given node is decided by its provenance and by nothing else:

    *A label the node owns stays pinned.* It was recorded about the node, so it is a property of
    the thing and travels with it into any corpus. The null asks of those nodes exactly what the
    permutation null asks, against a family of corpora rather than one fixed graph.

    *A borrowed label is derived again, from the rewired corpus.* It was never a property of the
    node: it is what most of the node's documents happened to be tagged with, and in a rewired
    corpus the node is in different documents, so it is a different label. Re-deriving it is what
    makes this the null for the borrowed-label confound: under it, "like joins like because both
    endpoints copied their value from the document that joined them" is *inside* the null, and
    survives as a finding only when the observed network sorts harder than that mechanism does on
    its own. The permutation null cannot do this -- shuffling breaks the copying, so it prices the
    confound at zero and the z-score goes up with the circularity.

    The derivation is :func:`graphrag.sna.export._label_attributes`'s, so the observed labels and
    the sampled ones are made the same way: the value most of the node's opposite-mode nodes
    carry, ties broken alphabetically. It needs the table of what those nodes were tagged with,
    which only a builder whose opposite mode *is* the source of the borrowing records -- the
    entity network. Without it every label stays pinned and the report says so.
    """
    data = attr_key(key)
    scores: list[float] = []
    for sample in bipartite_preserving(
        memberships.pairs,
        max(samples, 0),
        side=memberships.side,
        scheme=memberships.scheme,
        lam=memberships.lam,
        seed=rng,
    ):
        values = _sample_labels(sample, memberships, key, labels, sources)
        for node, value in values.items():
            sample.nodes[node][data] = value
        score = _coefficient(sample.subgraph(values), data)
        if score is not None:
            scores.append(score)
    return scores


def _sample_labels(
    sample: nx.Graph,
    memberships: Memberships,
    key: str,
    labels: Mapping[str, str],
    sources: Mapping[str, str],
) -> dict[str, str]:
    """Every labelled node's value in one null corpus: own pinned, borrowed derived again.

    One vote per membership, exactly as :func:`graphrag.sna.export._label_attributes` counts them
    -- with one difference worth knowing: the observed derivation votes once per *mention*, so an
    entity named twice in one passage votes twice there, while a membership table holds that
    passage once. The two agree except for entities named more than once in the same passage.

    A borrowed label whose new documents carry no value for ``key`` keeps the one it has: the
    alternative is dropping the node out of the sample, which would compare an assortativity over
    one node set with an assortativity over another.
    """
    values = dict(labels)
    if not memberships.right_attributes:
        return values
    votes: dict[str, Counter[str]] = defaultdict(Counter)
    for left, right in pairs_of(sample) or []:
        node, other = (left, right) if memberships.side == "left" else (right, left)
        if labels.get(node) is None or sources.get(node) != INHERITED:
            continue
        value = memberships.right_attributes.get(other, {}).get(key, "")
        if value:
            votes[node][value] += 1
    for node, counts in votes.items():
        values[node] = min(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return values


def _pruned_note(graph: nx.Graph) -> list[str]:
    """Say so when the observed network was pruned but the null's re-projections were not.

    ``--min-weight`` drops edges *after* the projection, and the memberships the graph carries are
    the ones it was projected from, so a re-projection of them arrives unpruned. The observation
    and the null are then not built the same way, which is the one thing a null may never do
    quietly. The comparison is still the best available -- pruning each sample would threshold a
    random network against a number chosen for the real one -- so it is reported rather than
    refused.
    """
    min_weight = int(graph.graph.get("min_weight", 1) or 1)
    if min_weight <= 1:
        return []
    return [
        f"This network was pruned at --min-weight {min_weight}, but the bipartite null's "
        "re-projections are not pruned: the memberships it rewires are the ones the network was "
        "projected from, before the threshold. The null therefore has edges the observed network "
        "no longer has, which flattens it. Re-run with --min-weight 1 to compare like with like."
    ]


def _bipartite_note(report: AttributeReport) -> str:
    """What the two nulls each hold fixed, in the report, next to the two z-scores."""
    result = report.bipartite
    if result is None or not result.testable:
        return (
            "The bipartite-preserving null produced no sample -- the two-mode network behind "
            "this projection has fewer than two nodes to trade between -- so only the label "
            "shuffle is reported."
        )
    labels = (
        "re-derives every borrowed label from each rewired corpus, so the copying that makes a "
        "borrowed label circular happens in the null too"
        if report.bipartite_rederived
        else "keeps every label where it is"
    )
    return (
        f"Two nulls, two claims. The label shuffle (z {report.assortativity_z:.2f}) holds this "
        f"projection fixed and moves the labels; the bipartite-preserving null (z {result.z:.2f}, "
        f"empirical p {result.p_value:.4f} over {result.samples} sample(s)) holds every left and "
        f"every right degree of the two-mode network fixed (§18.1) and {labels}. The second "
        "prices in the cliques a single document creates and how much each node was recorded, "
        "which the first cannot: where the two disagree, the gap is the corpus's own shape."
    )


def _replace(report: AttributeReport, **changes: Any) -> AttributeReport:
    """The report with some fields changed. It is frozen, so every step builds a new one."""
    return replace(report, **changes)


def _with_notes(report: AttributeReport, notes: Sequence[str]) -> AttributeReport:
    return _replace(report, notes=tuple(notes))


def _edge_split(graph: nx.Graph, labels: Mapping[str, str]) -> tuple[int, int, float, float]:
    """Edges and edge weight inside one value against edges crossing between two."""
    within = between = 0
    w_within = w_between = 0.0
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0) or 1.0)
        if labels[u] == labels[v]:
            within += 1
            w_within += weight
        else:
            between += 1
            w_between += weight
    return within, between, w_within, w_between


def _assortativity(
    graph: nx.Graph,
    key: str,
    permutations: int,
    rng: random.Random,
) -> tuple[float | None, list[float]]:
    """The observed attribute assortativity, and the same measure over shuffled labels.

    The permutation keeps the graph and the label counts exactly as they are and moves only
    which node holds which label (:func:`graphrag.sna.null.label_permutation`). That is the null a
    claim about sorting needs: a network with two dense clumps and labels sprinkled at random will
    score near zero here, however obvious the clumps are, because the clumps are not what the
    attribute is about. What it cannot do is price the clumps themselves, which is why a label
    borrowed from the documents that drew the edges scores *higher* here the more circular it is.
    """
    data = attr_key(key)
    observed = _coefficient(graph, data)
    if observed is None:
        return None, []
    scores: list[float] = []
    for sample in label_permutation(graph, max(permutations, 0), key=data, seed=rng):
        score = _coefficient(sample, data)
        if score is not None:
            scores.append(score)
    return observed, scores


def _coefficient(graph: nx.Graph, data: str) -> float | None:
    """``nx.attribute_assortativity_coefficient``, with its undefined cases turned into None."""
    if graph.number_of_edges() == 0:
        return None
    try:
        value = float(nx.attribute_assortativity_coefficient(graph, data))
    except (ValueError, ZeroDivisionError, nx.NetworkXError):
        return None
    return None if math.isnan(value) else value


def _fixed_partition_null(
    graph: nx.Graph,
    partition: Sequence[Sequence[str]],
    samples: int,
    resolution: float,
    rng: random.Random,
) -> list[float]:
    """Score this exact partition on degree-preserving rewirings of the same graph.

    The partition is held fixed because it was never fitted to the graph in the first place.
    Re-running Louvain on each sample, as the community null does, would answer a different
    question: how good a partition the rewired graph admits, which is not what an attribute
    claims.

    The rewiring is :func:`graphrag.sna.null.configuration`, the same sampler the community null
    draws from, so the two differ in what they score and in nothing else.
    """
    return [
        float(nx.community.modularity(rewired, partition, weight="weight", resolution=resolution))
        for rewired in configuration(graph, max(samples, 0), seed=rng)
    ]


def _agreement(
    groups: Sequence[Sequence[str]], labels: Mapping[str, str], rng: random.Random
) -> dict[str, float]:
    """How far the run's groups and the attribute labels are the same partition.

    Computed over the nodes that are in both, with a baseline from shuffling the group labels:
    the adjusted Rand index is already chance-corrected and should land near zero, and the
    mutual information is not, so its baseline is the only thing that makes it readable.
    """
    membership = {node: index for index, group in enumerate(groups) for node in group}
    shared = sorted(set(membership) & set(labels))
    if len(shared) < MIN_NODES or len({labels[n] for n in shared}) < 2:
        return {}
    values = sorted({labels[node] for node in shared})
    left = [membership[node] for node in shared]
    right = [values.index(labels[node]) for node in shared]
    scores = compare_partitions(left, right)
    baseline_ari: list[float] = []
    baseline_nmi: list[float] = []
    shuffled = list(left)
    for _ in range(PERMUTATIONS):
        rng.shuffle(shuffled)
        random_scores = compare_partitions(shuffled, right)
        baseline_ari.append(random_scores["adjusted_rand_index"])
        baseline_nmi.append(random_scores["normalized_mutual_information"])
    return {
        "adjusted_rand_index": scores["adjusted_rand_index"],
        "normalized_mutual_information": scores["normalized_mutual_information"],
        "baseline_adjusted_rand_index": mean(baseline_ari),
        "baseline_normalized_mutual_information": mean(baseline_nmi),
        "nodes": float(len(shared)),
    }


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_attribute(report: AttributeReport) -> list[str]:
    """The attribute section of a report: the counts first, then what they explain."""
    lines = [
        f"## By attribute: {report.key}",
        "",
        f"{report.labelled:,} of {report.total:,} nodes carry a value for `{report.key}`, over "
        f"{report.values} value(s). An attribute is a hypothesis about where this network "
        "divides, not a finding: the numbers below say how much of the network's shape it "
        "accounts for, and every one of them is computed over the labelled nodes only.",
        "",
    ]
    lines += _table(
        ["value", "nodes", "share of labelled"],
        [
            [value, f"{count:,}", f"{count / max(report.labelled, 1):.0%}"]
            for value, count in report.counts.items()
        ],
    )
    lines += [
        f"Recorded about the node itself: {report.own:,}. Borrowed from the node's documents: "
        f"{report.inherited:,}"
        + (
            f", {report.single_document:,} of them from a single document"
            if report.single_document
            else ""
        )
        + ".",
        "",
        "### Edges within and between values",
        "",
        f"- edges inside one value: {report.edges_within:,} "
        f"({report.within_share:.0%} of edges between labelled nodes)",
        f"- edges between two values: {report.edges_between:,}",
        f"- weight inside: {_num(report.weight_within)}; "
        f"weight between: {_num(report.weight_between)}",
        "",
    ]
    if report.assortativity is not None:
        lines += [
            "### Assortativity",
            "",
            f"- observed: {_num(report.assortativity)}",
            f"- {report.permutations} label shuffles: mean {_num(report.permuted_mean)}, "
            f"sd {_num(report.permuted_std)}",
            f"- z-score: {_num(report.assortativity_z, 2)}",
            "",
            report.verdict,
            "",
        ]
    lines += _render_homophily(report.homophily)
    lines += _render_bipartite(report)
    if report.louvain_modularity is not None:
        lines += [
            "### Modularity of the attribute partition",
            "",
            f"- grouping by `{report.key}`: {_num(report.modularity)}",
            f"- best Louvain grouping of the same nodes: {_num(report.louvain_modularity)}",
            f"- {report.null_samples} degree-preserving rewirings scoring the same partition: "
            f"mean {_num(report.null_mean)}, sd {_num(report.null_std)}, "
            f"z {_num(report.modularity_z, 2)}",
            "",
            _modularity_reading(report),
            "",
        ]
    if report.agreement:
        lines += [
            f"### The {report.group_plural} against the attribute",
            "",
        ]
        lines += _table(
            ["measure", "observed", "random baseline"],
            [
                [
                    "adjusted Rand index",
                    _num(report.agreement["adjusted_rand_index"]),
                    _num(report.agreement["baseline_adjusted_rand_index"]),
                ],
                [
                    "normalised mutual information",
                    _num(report.agreement["normalized_mutual_information"]),
                    _num(report.agreement["baseline_normalized_mutual_information"]),
                ],
            ],
        )
        lines += [
            f"Over the {int(report.agreement['nodes']):,} node(s) that carry both a "
            f"{report.group_noun} and a value. The baseline is the same comparison with the "
            f"{report.group_noun} labels shuffled, which is what 'no relationship' scores.",
            "",
        ]
    if report.notes:
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return lines


def _render_homophily(indices: HomophilyIndices | None) -> list[str]:
    """§30.2's count-based block: EI overall and per value, Coleman's index, then §30.4's caveat.

    The per-value table is the point of the block, not decoration. EI alone reproduces the
    chapter's warning rather than answering it, so every row carries the group's share of the
    nodes and the share of its tie-ends that stayed inside it, which is what Coleman's index is
    computed from -- a reader can check the index by hand from the row that prints it.
    """
    if indices is None or indices.ei is None:
        return []
    lines = [
        "### Homophily by counting edges (EI and Coleman)",
        "",
        f"**Sampling frame.** The {indices.nodes:,} labelled node(s) of this network and the "
        f"{indices.edges:,} edge(s) between two of them: {indices.internal:,} inside a value, "
        f"{indices.external:,} between two. Counts of edges, not weights.",
        "",
        f"**n.** {indices.edges:,} edge(s), {len(indices.values)} value(s).",
        "",
        f"**Null model.** `label_permutation`, {indices.permutations} shuffle(s) of the same "
        "labels over the same graph (§19.1). It holds the group sizes fixed, which is exactly "
        "what the EI index cannot do for itself.",
        "",
        "**Implements.** §30.2 (quantifying homophily), with Krackhardt and Stern's E-I index "
        "(1988) and Coleman's homophily index (1958) over the counts the section works in; the "
        "chapter's own answer to the same problem, Newman's assortativity, is above.",
        "",
        f"- **EI index** {indices.ei:+.4f} — **negative is homophily**, the opposite sign to the "
        "assortativity above.",
        f"- **Against the shuffle** mean {indices.permuted_mean:+.4f}, sd "
        f"{indices.permuted_std:.4f}, z {indices.ei_z:+.2f}.",
        "",
    ]
    lines += _table(
        ["value", "nodes", "share", "edges inside", "edges out", "EI", "in-group ties", "Coleman"],
        [
            [
                row.value,
                f"{row.nodes:,}",
                f"{row.share:.0%}",
                f"{row.internal:,}",
                f"{row.external:,}",
                "—" if row.ei is None else f"{row.ei:+.3f}",
                f"{row.observed:.0%} vs {row.expected:.0%} expected",
                "—" if row.coleman is None else f"{row.coleman:+.3f}",
            ]
            for row in indices.values
        ],
    )
    lines += [
        "`in-group ties` is the share of this value's tie-ends that stayed inside it against the "
        "share its size alone would give it; Coleman's index is the gap between those two, "
        "scaled so that 0 is 'exactly as often as size implies' and +1 is 'only inside'. EI does "
        "not make that comparison, which is why the two columns can point opposite ways.",
        "",
        indices.reading,
        "",
        CONTAGION_CAVEAT,
        "",
    ]
    return lines


def _homophily_payload(indices: HomophilyIndices | None) -> dict[str, object] | None:
    """§30.2's indices as plain data, or ``None`` when there was too little to count."""
    if indices is None:
        return None
    return {
        "key": indices.key,
        "nodes": indices.nodes,
        "edges": indices.edges,
        "internal": indices.internal,
        "external": indices.external,
        "ei": indices.ei,
        "permutations": indices.permutations,
        "permuted_mean": indices.permuted_mean,
        "permuted_std": indices.permuted_std,
        "ei_z": indices.ei_z,
        "size_driven": list(indices.size_driven),
        "reading": indices.reading,
        "contagion_caveat": CONTAGION_CAVEAT,
        "values": [
            {
                "value": row.value,
                "nodes": row.nodes,
                "share": row.share,
                "internal": row.internal,
                "external": row.external,
                "ei": row.ei,
                "observed": row.observed,
                "expected": row.expected,
                "coleman": row.coleman,
            }
            for row in indices.values
        ],
    }


def _tail_words(tail: str) -> str:
    """Which test a p-value came from, in words, because the number alone does not say."""
    return "two-sided" if tail == "two" else f"{tail} tail"


def _render_bipartite(report: AttributeReport) -> list[str]:
    """The bipartite-null block: the frame, the n, the null and the chapter, beside the numbers."""
    result = report.bipartite
    if result is None:
        return []
    lines = [
        "### The same assortativity against a bipartite-preserving null",
        "",
        f"**Sampling frame.** The {report.bipartite_pairs:,} two-mode membership(s) this network "
        f"was projected from -- who was recorded in what -- restricted afterwards to the "
        f"{report.labelled:,} labelled node(s) the observed number was computed over.",
        "",
        f"**n.** {result.samples} null network(s), each one a re-projection of a rewired "
        "membership table.",
        "",
        "**Null model.** `bipartite_preserving`, which holds fixed "
        f"{HOLDS_FIXED['bipartite_preserving']}. {_label_rule(report)}",
        "",
        "**Implements.** §18.1 (fixed degree sequences, on the two-mode network) and §19.1 "
        "(shuffling for significance), via the curveball algorithm of Strona et al. (2014).",
        "",
    ]
    if not result.testable:
        return [*lines, result.caveat, ""]
    lines += [
        f"- observed: {_num(result.observed)}",
        f"- null: mean {_num(result.null_mean)}, sd {_num(result.null_std)}",
        f"- z-score: {_num(result.z, 2)}; empirical p ({_tail_words(result.tail)}): "
        f"{_num(result.p_value)}",
        "",
    ]
    if result.null_std == 0.0 and result.samples:
        lines += [
            f"Every one of the {result.samples} samples produced the same value to the digit, so "
            "there is no spread to score against and the z-score above is 0.0 by convention "
            "rather than by measurement. Read the empirical p: this corpus produces this "
            "assortativity whenever it is rewired, which makes the observation a property of how "
            "the network was built.",
            "",
        ]
    if result.caveat:
        lines += [result.caveat, ""]
    return lines


def _label_rule(report: AttributeReport) -> str:
    """Which labels the null moved, which is half of what it holds fixed."""
    if report.bipartite_rederived:
        return (
            "Labels move with the corpus: a value the node owns is pinned to it, and a borrowed "
            "one is derived again from the documents each sample put the node in, by the same "
            "majority rule the network was built with."
        )
    if report.inherited:
        return (
            "The labels do not move. This network records no table of what its opposite-mode "
            "nodes were tagged with, so the borrowed labels could not be derived again and were "
            "carried across instead; against this null a borrowed label is still circular. Only "
            "the entity network can re-derive them."
        )
    return "The labels do not move: every one of them was recorded about its own node."


def _modularity_reading(report: AttributeReport) -> str:
    """What the three modularity numbers say together, which is more than any one of them."""
    best = report.louvain_modularity or 0.0
    if report.confounded:
        # The same circularity the verdict refuses on. Saying "the communities are largely this
        # attribute" two paragraphs under "not a reading of this attribute" would hand a reader
        # the sentence they wanted and let them quote it on its own.
        return (
            "Borrowed labels, so this says nothing the assortativity did not: the partition "
            "groups nodes by the documents whose co-membership drew the edges, and it would "
            "score well here however the values were named."
        )
    if report.modularity_z < 2:
        return (
            "The attribute partition explains the edges no better than the same partition would "
            "explain a random graph with these degrees. Whatever divides this network, it is "
            "not this."
        )
    if best and report.modularity >= 0.8 * best:
        return (
            "The attribute partition is close to the best grouping Louvain found, so the "
            "communities in this network are largely this attribute."
        )
    return (
        "The attribute partition beats its null but falls well short of Louvain's best, so it "
        "is a real division and not the main one. Name what the Louvain groups have in common "
        "before treating this attribute as the explanation."
    )


def attribute_payload(report: AttributeReport) -> dict[str, object]:
    """The same section as plain JSON-able data, for the ``--json`` output."""
    return {
        "key": report.key,
        "counts": report.counts,
        "labelled": report.labelled,
        "total": report.total,
        "unlabelled": report.unlabelled,
        "mixed": report.mixed,
        "sources": report.sources,
        "own": report.own,
        "inherited": report.inherited,
        "inherited_share": report.inherited_share,
        "single_document": report.single_document,
        "confounded": report.confounded,
        "edges_within": report.edges_within,
        "edges_between": report.edges_between,
        "weight_within": report.weight_within,
        "weight_between": report.weight_between,
        "assortativity": report.assortativity,
        "permutations": report.permutations,
        "permuted_mean": report.permuted_mean,
        "permuted_std": report.permuted_std,
        "assortativity_z": report.assortativity_z,
        "homophily": _homophily_payload(report.homophily),
        "null": report.null,
        "bipartite_pairs": report.bipartite_pairs,
        "bipartite_rederived": report.bipartite_rederived,
        "bipartite": _significance_payload(report.bipartite),
        "verdict": report.verdict,
        "modularity": report.modularity,
        "louvain_modularity": report.louvain_modularity,
        "null_samples": report.null_samples,
        "null_mean": report.null_mean,
        "null_std": report.null_std,
        "modularity_z": report.modularity_z,
        "agreement": report.agreement,
        "notes": list(report.notes),
    }


def _significance_payload(result: Significance | None) -> dict[str, object] | None:
    """One :class:`graphrag.sna.null.Significance` as plain data, or ``None`` when it was not run.

    The sample count and the null's name travel with the numbers rather than in a header, because
    a z-score without them is not readable: 3.0 over 20 samples and 3.0 over 2,000 are different
    claims, and neither means anything until the reader knows what the null held fixed.
    """
    if result is None:
        return None
    return {
        "null": result.null,
        "holds_fixed": HOLDS_FIXED.get(result.null, ""),
        "observed": result.observed,
        "samples": result.samples,
        "null_mean": result.null_mean,
        "null_std": result.null_std,
        "z": result.z,
        "p_value": result.p_value,
        "tail": result.tail,
        "heavy_tailed": bool(result.summary and result.summary.heavy_tailed),
        "caveat": result.caveat,
    }
