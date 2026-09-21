"""Is this partition any good? Chapter 36 of the Atlas, as a battery rather than a number.

The chapter opens by refusing the question it was asked (p. 510): *"Why a 'battery' of functions?
Doesn't 'best' imply that there is some sort of ideal partition? Not really. What's 'best'
depends on what you want to use your communities for. Different functions privilege different
applications."* So nothing here returns a verdict. Every measure is printed with what it wants
(minimised or maximised), which direction community *size* pushes it, and the section of the book
that defines it, because a quality function that favours large communities and one that favours
small ones will disagree about the same partition and both be right.

Four questions, one per section of the chapter:

*§36.1, modularity.* The observed edges inside the communities against a configuration model's
expectation. Its domain is -0.5 to +1 (p. 512), and a negative value is a real reading -- nodes
grouped together that connect less than chance would -- not a bug. Its famous defect is the
resolution limit (p. 516-517): *"modularity has a preferred community size, relative to the size
of the graph… it accepts partitions only of a comparable size with the size of the network"*, so
:func:`resolution_limit` flags every community small enough to be at risk of being absorbed.

*§36.2, the other topological measures.* Conductance, internal density, the cut ratio and its
normalised form, and the three out-degree fractions of Flake et al., each worked through by hand
in the book and each reproduced here to the digit. Their point is the one the chapter makes on
p. 521: *"Neither conductance nor internal density fully capture the classical definition of
community discovery… Each of the two measures only satisfies one of the two requirements."*

*§36.3, link prediction.* A partition is a claim about where the next edge will appear, so it can
be scored as a link predictor: hold edges out, score every pair by whether it is inside a
community, and read the AUC (p. 523-524).

*§36.4, mutual information against ground truth.* NMI, and beside it the chance-corrected AMI and
ARI, because *"the problem of mutual information is that it is always non-zero… there will be
always a little mutual information between two vectors, even if they are both completely random"*
(p. 526). The chapter's own example scores two independently drawn vectors at NMI 0.09 and AMI
-0.22.

**What the book defines and what this adds.** Coverage, performance, expansion and the triangle
participation ratio are not worked through in chapter 36; they come from the review the chapter
sends you to for *"a battery of other quality measures, conveniently grouped in a review paper"*
(p. 519, Leskovec, Lang and Mahoney 2010) and from Yang and Leskovec 2015, which p. 524 cites.
They are computed here because the report needs a measure that favours large communities
(coverage) beside one that favours small ones (performance), which is the chapter's own argument
made visible. Each carries its source in :data:`MEASURES`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score

from graphrag.sna.experiment import auc, holdout
from graphrag.sna.matrices import Node
from graphrag.sna.measures import undirected_view
from graphrag.sna.null import Significance, significance
from graphrag.sna.stats import (
    compare_partitions,
    normalized_mutual_information,
    variation_of_information,
)

#: What the section header says it implements, so a reader can open the book at the right page.
CHAPTER = "§36.1-36.4"

#: The share of edges §36.3's test holds out by default. Ten per cent is the usual split of
#: chapter 25 and is a convention, not the book's number: §36.3 says to "design the experiment
#: (Chapter 25)" and leaves the split to that chapter.
HOLDOUT_SHARE = 0.1

#: How many random partitions of the same community sizes the link-prediction AUC is scored
#: against. Twenty gives an empirical p-value no finer than 0.048, which is all this baseline is
#: for: it says whether "same community" beat dealing the same labels out at random.
NULL_PARTITIONS = 20

#: Below this many edges there is nothing to hold out: one edge in ten of a nine-edge graph is
#: one edge, and an AUC over one positive pair is not a measurement.
MIN_HOLDOUT_EDGES = 20


@dataclass(frozen=True)
class Measure:
    """One quality function, with what it wants and what community size does to it.

    ``favours`` is the sentence the report prints beside the number. It is the chapter's central
    warning made concrete: a partition that looks excellent under conductance and terrible under
    performance has not been measured twice, it has been measured by two definitions of
    community (p. 521).
    """

    key: str
    name: str
    section: str
    """Where the book defines it, with the page, or the citation when it does not."""
    wants: str
    """``minimise`` or ``maximise``: which direction is "better" for this measure."""
    favours: str
    """Which community size the measure is biased towards, and why."""
    meaning: str


#: Every measure the battery computes, in the order the report prints them. The partition-level
#: ones come first; the rest are per community and are also averaged over them.
MEASURES: tuple[Measure, ...] = (
    Measure(
        key="modularity",
        name="modularity",
        section="§36.1, p. 512",
        wants="maximise",
        favours=(
            "communities of a size comparable with the network: below the resolution limit it "
            "prefers to merge two genuinely distinct small communities (p. 516)"
        ),
        meaning=(
            "edges inside the communities minus what a configuration model with the same degrees "
            "would put there. 0 is the partition that puts every node in one community; the "
            "domain is -0.5 to +1"
        ),
    ),
    Measure(
        key="coverage",
        name="coverage",
        section="Fortunato 2010 (the review §36.2 sends you to, p. 519)",
        wants="maximise",
        favours=(
            "large communities, without limit: put every node in one community and coverage is "
            "exactly 1"
        ),
        meaning="the share of all edges that fall inside a community",
    ),
    Measure(
        key="performance",
        name="performance",
        section="Fortunato 2010 (the review §36.2 sends you to, p. 519)",
        wants="maximise",
        favours=(
            "small communities on a sparse network: most node pairs are non-edges, and every "
            "one of them scores as long as the two sit in different communities"
        ),
        meaning=(
            "the share of node pairs the partition gets right: edges inside a community plus "
            "non-edges across communities, over all pairs"
        ),
    ),
    Measure(
        key="conductance",
        name="conductance",
        section="§36.2, p. 519",
        wants="minimise",
        favours=(
            "large communities: the whole network is one community with no boundary at all, so "
            "conductance 0, which is why the book says you cannot maximise it blindly"
        ),
        meaning=(
            "boundary edges over the community's volume, |E_B| / (2|E_C| + |E_B|): how easily "
            "something inside the community gets out. It measures external sparsity and says "
            "nothing about internal density"
        ),
    ),
    Measure(
        key="internal_density",
        name="internal density",
        section="§36.2, p. 520",
        wants="maximise",
        favours=(
            "small communities: the maximum is a clique, so maximising it blindly \"you're only "
            'going to find cliques in your networks" (p. 521)'
        ),
        meaning=(
            "edges inside over edges the community could hold, |E_C| / (|C|(|C|-1)/2). The other "
            "side of conductance's coin: internal density with no view of the outside"
        ),
    ),
    Measure(
        key="expansion",
        name="expansion",
        section="Leskovec, Lang and Mahoney 2010 (the review of p. 519)",
        wants="minimise",
        favours="large communities: the same boundary spread over more nodes",
        meaning="boundary edges per node of the community, |E_B| / |C|",
    ),
    Measure(
        key="cut_ratio",
        name="cut ratio",
        section="§36.2, p. 521",
        wants="minimise",
        favours=(
            "nothing consistently, which is the problem: the denominator counts every pair that "
            "crosses the boundary, so on a sparse network every community scores near zero and "
            'the measure separates partitions barely at all. "A measure easy to game"'
        ),
        meaning="boundary edges over all possible boundary edges, |E_B| / (|C|(|V|-|C|))",
    ),
    Measure(
        key="normalized_cut",
        name="normalised cut",
        section="§36.2, p. 521",
        wants="minimise",
        favours=(
            "balanced partitions: the second term is the conductance of the rest of the network, "
            "so a community that is nearly everything is penalised where plain conductance "
            "rewards it"
        ),
        meaning=(
            "conductance plus the conductance of the complement, |E_B| / (2|E_C| + |E_B|) + "
            "|E_B| / (2(|E| - |E_C|) + |E_B|)"
        ),
    ),
    Measure(
        key="max_odf",
        name="maximum out-degree fraction",
        section="§36.2, p. 522",
        wants="minimise",
        favours=(
            "communities with no low-degree node on the boundary: one node decides the number, "
            'so it is "a blunt tool which disregards lots of information" (p. 523)'
        ),
        meaning=(
            "the largest share of any one node's edges that leave the community. A hub may point "
            "out a lot; a degree-3 node pointing out twice is what this catches"
        ),
    ),
    Measure(
        key="average_odf",
        name="average out-degree fraction",
        section="§36.2, p. 522",
        wants="minimise",
        favours=(
            "the same sizes conductance does, and lands close to it: the book's worked example "
            "gives 0.319 against a conductance of 0.316 for the same community (p. 523)"
        ),
        meaning=(
            "the mean over nodes of the share of that node's edges leaving the community. "
            "Normalised node by node, where conductance normalises over the whole community"
        ),
    ),
    Measure(
        key="flake_odf",
        name="Flake out-degree fraction",
        section="§36.2, p. 522",
        wants="minimise",
        favours=(
            "communities whose members are each individually inside them, at any size: it counts "
            "nodes rather than edges, so one enormous hub costs no more than one leaf"
        ),
        meaning=(
            "the share of the community's nodes that have more edges leaving it than staying "
            'in. "A node shouldn\'t do that!"'
        ),
    ),
    Measure(
        key="triangle_participation",
        name="triangle participation ratio",
        section="Yang and Leskovec 2015 (cited at p. 524)",
        wants="maximise",
        favours=(
            "small dense communities, for the same reason internal density does: a clique scores "
            "1 and a tree scores 0 whatever its size"
        ),
        meaning=(
            "the share of the community's nodes that sit in at least one triangle *inside* the "
            "community"
        ),
    ),
)

#: The per-community measures, in report order: everything in :data:`MEASURES` that
#: :class:`CommunityScores` carries a field for.
PER_COMMUNITY: tuple[str, ...] = (
    "conductance",
    "internal_density",
    "expansion",
    "cut_ratio",
    "normalized_cut",
    "max_odf",
    "average_odf",
    "flake_odf",
    "triangle_participation",
)

_BY_KEY: dict[str, Measure] = {measure.key: measure for measure in MEASURES}


# ----------------------------------------------------------------------------- §36.1, §36.2


@dataclass(frozen=True)
class CommunityScores:
    """One community, measured on its own. Every field is a number §36.2 defines.

    A field is ``nan`` where the measure is undefined for this community rather than zero:
    internal density needs two nodes, the cut ratio needs something outside the community, and
    the normalised cut needs an edge outside it. The report prints a dash and the payload a
    ``null``, so a reader never reads "0.0" for "could not be computed".
    """

    index: int
    size: int
    internal_edges: int
    """|E_C| in the book's notation: edges with both ends in the community."""
    boundary_edges: int
    """|E_B,C|: edges with exactly one end in the community."""
    conductance: float
    internal_density: float
    expansion: float
    cut_ratio: float
    normalized_cut: float
    max_odf: float
    average_odf: float
    flake_odf: float
    triangle_participation: float
    below_resolution_limit: bool
    """Whether |E_C| is under sqrt(2|E|), the size at which modularity maximisation may absorb
    this community into a neighbour (§36.1, p. 516-517)."""

    def value(self, key: str) -> float:
        """The measure named by ``key``, for the report's table and the payload."""
        return float(getattr(self, key))


@dataclass(frozen=True)
class TruthComparison:
    """§36.4: the partition against a labelling somebody claims is the true one.

    All four numbers compare the same two labellings of the same nodes. ``nmi`` is not corrected
    for chance and ``adjusted_mutual_information`` and ``adjusted_rand_index`` are, which is the
    whole point of the section: the book's own pair of independently drawn random vectors scores
    NMI 0.09 and AMI -0.22 (p. 526).
    """

    nodes: int
    found_communities: int
    truth_communities: int
    nmi: float
    adjusted_mutual_information: float
    adjusted_rand_index: float
    variation_of_information: float
    caveat: str
    """Why a high number here is not proof, in the chapter's own terms (p. 527-528)."""


@dataclass(frozen=True)
class LinkPrediction:
    """§36.3: the partition scored as a link predictor over a held-out share of the edges.

    ``auc`` is the probability that a held-out edge is ranked above a sampled non-edge by the
    binary "are these two in the same community" score of p. 524, with ties -- every pair the
    partition scores identically -- counted as half, which is what makes 0.5 the value of a
    predictor that knows nothing.
    """

    auc: float
    share: float
    held_out: int
    """How many edges were removed and used as the positive class."""
    non_edges: int
    """How many unconnected pairs were sampled as the negative class."""
    seed: int | None
    null: Significance
    """The same AUC over :data:`NULL_PARTITIONS` random relabellings that keep the community
    sizes exactly, so the baseline is a partition of this shape and not of any shape."""
    caveat: str


@dataclass
class PartitionScores:
    """The whole battery over one partition of one network (§36.1-36.4).

    ``notes`` carries everything the numbers cannot: that the graph was flattened, that the
    partition did not cover the network, that modularity read the weights while §36.2's counts
    did not. A report prints them beside the table.
    """

    frame: str
    nodes: int
    edges: int
    communities: int
    sizes: list[int]
    modularity: float
    resolution: float
    weighted: bool
    """Whether modularity was computed with edge weights. §36.2's measures never are: the
    chapter's formulas count edges (p. 519-522)."""
    resolution_limit: float
    """sqrt(2|E|): the internal-edge count below which §36.1's resolution limit bites."""
    flagged: list[int] = field(default_factory=list)
    """The indices of the communities under that limit."""
    per_community: list[CommunityScores] = field(default_factory=list)
    averages: dict[str, float] = field(default_factory=dict)
    """Every key of :data:`MEASURES`: the partition-level measures as themselves, the
    per-community ones averaged over the communities they are defined on."""
    truth: TruthComparison | None = None
    link_prediction: LinkPrediction | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def max_modularity(self) -> float:
        """The ceiling this many communities could reach: 1 - 1/k.

        A partition into k equal, mutually disconnected groups scores 1 - 1/k, which is the
        highest modularity k communities can have -- the book gives the domain (+1) rather than
        this bound, so it is the standard result and not the chapter's. Printed beside the
        observed value because "modularity 0.42" says nothing until you know the ceiling.
        """
        return 1.0 - 1.0 / self.communities if self.communities else 0.0


def resolution_limit(graph: nx.Graph) -> float:
    """sqrt(2|E|): the community size modularity starts absorbing (§36.1, p. 516-517).

    The chapter states the limit in words -- *"when we have small communities relative to the
    number of the edges of the network, for modularity it is better to merge them, even if they
    are clearly and intuitively distinct"* -- and cites Fortunato and Barthelemy for it. This
    number is that paper's threshold: a community with fewer than sqrt(2|E|) internal edges can
    raise modularity by merging with a neighbour, whatever a human thinks of the result. It is a
    property of the *network*, not of any partition, so it is the same for every partition
    scored on the same graph.

    Returns 0.0 for a graph with no edges, where nothing can be absorbed because nothing is
    connected.
    """
    return math.sqrt(2.0 * graph.number_of_edges())


def adjusted_mutual_information(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Mutual information with chance subtracted: 0 for random labellings (§36.4, p. 526-527).

    *"We subtract from mutual information the amount of bits we would expect to obtain about a
    vector by pure chance… AMI and ARI are defined to be equal to zero when you get nothing more
    than you'd expect by just tossing coins. At this point, any positive value starts getting
    interesting. These indexes can be negative… A negative AMI means that your clustering isn't
    good."*

    It lives here rather than beside :func:`graphrag.sna.stats.normalized_mutual_information`
    because the book introduces it in this chapter, as the repair for the NMI that chapter 3
    defines, and nothing outside community evaluation asks for it. 1.0 when the two labellings
    are the same partition; undefined for empty or unequal-length sequences, which raise
    ``ValueError``.
    """
    if not a or not b:
        msg = "adjusted_mutual_information() needs two non-empty labellings"
        raise ValueError(msg)
    if len(a) != len(b):
        msg = (
            "adjusted_mutual_information() needs two labellings of the same elements, got "
            f"{len(a)} and {len(b)}"
        )
        raise ValueError(msg)
    return float(adjusted_mutual_info_score(list(a), list(b)))


def _membership(communities: Sequence[Iterable[Node]]) -> dict[Node, int]:
    """Node to community index, rejecting a node claimed by two communities.

    Modularity and every measure here are defined on *disjoint* partitions: "the standard
    definition of modularity works exclusively with undirected, unweighted, disjoint partitions"
    (p. 515). Overlapping communities are chapter 38 and need their own measures, so this raises
    rather than counting a node twice.
    """
    membership: dict[Node, int] = {}
    for index, community in enumerate(communities):
        for node in community:
            if node in membership:
                msg = (
                    f"evaluate_partition() needs disjoint communities (§36.1 p. 515): node "
                    f"{node!r} is in communities {membership[node] + 1} and {index + 1}. "
                    "Overlapping partitions are Atlas ch. 38 and need their own measures."
                )
                raise ValueError(msg)
            membership[node] = index
    return membership


def _community_scores(
    graph: nx.Graph, communities: Sequence[Sequence[Node]], membership: Mapping[Node, int]
) -> list[CommunityScores]:
    """Every §36.2 measure for every community, in one pass over the edges plus one per triangle.

    Edges are *counted*, never weighted: the chapter's formulas are counts of edges (p. 519-522),
    and a conductance computed over weights is a different quantity that none of its worked
    examples would reproduce. Self-loops are skipped, as §12.1's density does, because a loop is
    neither inside a community in the sense of joining two of its members nor on its boundary.
    """
    count = len(communities)
    internal = [0] * count
    boundary = [0] * count
    out_degree: dict[Node, int] = dict.fromkeys(membership, 0)
    in_degree: dict[Node, int] = dict.fromkeys(membership, 0)
    for u, v in graph.edges():
        if u == v:
            continue
        cu, cv = membership[u], membership[v]
        if cu == cv:
            internal[cu] += 1
            in_degree[u] += 1
            in_degree[v] += 1
        else:
            boundary[cu] += 1
            boundary[cv] += 1
            out_degree[u] += 1
            out_degree[v] += 1

    total_nodes = graph.number_of_nodes()
    total_edges = sum(1 for u, v in graph.edges() if u != v)
    limit = math.sqrt(2.0 * total_edges)
    rows: list[CommunityScores] = []
    for index, community in enumerate(communities):
        size, inside, cut = len(community), internal[index], boundary[index]
        volume = 2 * inside + cut
        fractions = [
            out_degree[n] / (out_degree[n] + in_degree[n])
            if out_degree[n] + in_degree[n]
            else 0.0  # an isolated node has no edges pointing anywhere, so none point out
            for n in community
        ]
        bad = sum(1 for n in community if in_degree[n] < (out_degree[n] + in_degree[n]) / 2)
        outside_edges = total_edges - inside
        rows.append(
            CommunityScores(
                index=index,
                size=size,
                internal_edges=inside,
                boundary_edges=cut,
                conductance=cut / volume if volume else math.nan,
                internal_density=(inside / (size * (size - 1) / 2) if size > 1 else math.nan),
                expansion=cut / size if size else math.nan,
                cut_ratio=(
                    cut / (size * (total_nodes - size)) if size and total_nodes > size else math.nan
                ),
                normalized_cut=(
                    cut / volume + cut / (2 * outside_edges + cut)
                    if volume and (2 * outside_edges + cut)
                    else math.nan
                ),
                max_odf=max(fractions) if fractions else math.nan,
                average_odf=sum(fractions) / size if size else math.nan,
                flake_odf=bad / size if size else math.nan,
                triangle_participation=_triangle_participation(graph, community),
                below_resolution_limit=inside < limit,
            )
        )
    return rows


def _triangle_participation(graph: nx.Graph, community: Sequence[Node]) -> float:
    """The share of the community's nodes in at least one triangle whose three nodes are in it.

    Yang and Leskovec's measure, cited by §36.3 (p. 524) rather than defined by the chapter. The
    triangles are counted on the induced subgraph, so a node in a triangle with two outsiders
    does not count: the question is whether the community is internally cohesive, not whether its
    members are.
    """
    if len(community) < 3:
        return 0.0
    induced = graph.subgraph(community)
    triangles = nx.triangles(induced)
    return sum(1 for node in community if triangles.get(node, 0)) / len(community)


def _mean(values: Sequence[float]) -> float:
    """The mean of the defined values, or ``nan`` when none of them is.

    A measure undefined for one community -- a singleton has no internal density -- must not
    drag the average to zero, and must not turn it into ``nan`` either. The report prints how
    many communities each average covers.
    """
    defined = [value for value in values if not math.isnan(value)]
    return sum(defined) / len(defined) if defined else math.nan


# ----------------------------------------------------------------------------- §36.3


def _same_community(
    membership: Mapping[Node, int], pairs: Iterable[tuple[Node, Node]]
) -> dict[tuple[Node, Node], float]:
    """§36.3's classifier: *"1 if u and v are in the same community, 0 otherwise"* (p. 524).

    A node the partition never placed scores 0 against everything, including another unplaced
    node: "not in a community" is not a community.
    """
    scores: dict[tuple[Node, Node], float] = {}
    for u, v in pairs:
        here, there = membership.get(u), membership.get(v)
        scores[u, v] = float(here is not None and here == there)
    return scores


def link_prediction_score(
    graph: nx.Graph,
    communities: Sequence[Iterable[Node]],
    *,
    share: float = HOLDOUT_SHARE,
    nulls: int = NULL_PARTITIONS,
    seed: int | None = None,
) -> LinkPrediction | None:
    """Score the partition as a link predictor over a held-out share of the edges (§36.3).

    *"Communities are dense areas in the network, thus they tell you something about where you
    expect to find new links… you can use your communities to have a prior about where the new
    links will appear… The higher your AUC, the better looking your ROC curve, the better your
    partition is"* (p. 523-524).

    The score is the chapter's own binary classifier: 1 if the two nodes are in the same
    community, 0 otherwise. The baseline beside it is the same AUC over ``nulls`` random
    relabellings that keep the community *sizes* exactly, so what is tested is the placement of
    the nodes and not the shape of the partition, and a partition that predicts nothing reads
    0.5 against a baseline of 0.5.

    The split and the AUC are :func:`graphrag.sna.experiment.holdout` and
    :func:`graphrag.sna.experiment.auc` (ATL-25): one holdout in this package, not two. That
    module needs string node ids (its own error names the fix), so the graph and the community
    labels are relabelled with ``str()`` before either is called; every id compared below is a
    string for exactly that reason.

    **Where this departs from the chapter.** §36.3 is a test for a *temporal* network, where the
    held-out edges are the future; on a static one it points at chapter 25's k-fold cross
    validation. This is a single seeded split, not k folds, and -- more importantly -- the
    partition it scores was found on the whole network, held-out edges included. That leakage
    makes the AUC optimistic and it is why the shuffled-label baseline is printed beside it
    rather than the AUC alone: the honest version refits the partition on the training network,
    which needs the community-discovery method as an argument that nothing here supplies.
    "same community" reads the partition and not the edges, so the training graph
    :func:`graphrag.sna.experiment.holdout` builds is handed to nothing; it exists so that a
    predictor which does read the edges scores the same split.

    Returns ``None`` when the network has fewer than :data:`MIN_HOLDOUT_EDGES` edges, where a ten
    per cent holdout is one or two pairs and the AUC is noise, or when the split itself has
    nothing to hold out (:func:`graphrag.sna.experiment.holdout` raises in that case; this
    function turns the refusal into the same ``None`` rather than letting it propagate).
    """
    graph, _ = undirected_view(graph)
    if graph.number_of_edges() < MIN_HOLDOUT_EDGES or not communities:
        return None
    membership = {str(node): index for node, index in _membership(communities).items()}
    relabelled = nx.relabel_nodes(graph, str)
    try:
        split = holdout(relabelled, share=share, seed=seed, negatives=1.0)
    except ValueError:
        return None
    if not split.positives or not split.negatives:
        return None
    pairs = split.pairs
    observed = auc(_same_community(membership, pairs), split.positives, split.negatives)

    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    nodes = sorted(membership, key=str)
    labels = [membership[node] for node in nodes]
    samples: list[float] = []
    for _ in range(max(nulls, 0)):
        rng.shuffle(labels)
        shuffled = dict(zip(nodes, labels, strict=True))
        samples.append(auc(_same_community(shuffled, pairs), split.positives, split.negatives))
    return LinkPrediction(
        auc=observed,
        share=share,
        held_out=len(split.positives),
        non_edges=len(split.negatives),
        seed=seed,
        null=significance(
            observed,
            samples,
            null="random partition of the same community sizes (labels dealt out again)",
        ),
        caveat=(
            "One seeded split, not chapter 25's k folds, and the partition was found on the "
            "whole network including the held-out edges, so this AUC is optimistic. The "
            "shuffled-label baseline beside it is what makes it readable."
        ),
    )


# ----------------------------------------------------------------------------- the battery


def _truth_labels(
    truth: Mapping[Node, Hashable] | Sequence[Iterable[Node]], nodes: Sequence[Node]
) -> tuple[list[Hashable], list[Node]]:
    """The ground-truth label of every node that has one, with the nodes it covers.

    ``truth`` is either a mapping from node to label -- what node metadata looks like (§36.4
    calls the attributes "node metadata") -- or a list of communities, which is what another
    partition looks like. Nodes with no label are left out of the comparison entirely rather than
    given a label of their own: an untagged node is not a value.
    """
    if isinstance(truth, Mapping):
        labelled = [node for node in nodes if node in truth]
        return [truth[node] for node in labelled], labelled
    membership = _membership(list(truth))
    labelled = [node for node in nodes if node in membership]
    return [membership[node] for node in labelled], labelled


def _compare_to_truth(
    communities: Sequence[Sequence[Node]],
    truth: Mapping[Node, Hashable] | Sequence[Iterable[Node]],
    nodes: Sequence[Node],
) -> TruthComparison | None:
    """§36.4 over the nodes that carry a ground-truth label, or ``None`` when too few do."""
    labels, covered = _truth_labels(truth, nodes)
    if len(covered) < 2:
        return None
    membership = _membership(communities)
    found = [membership[node] for node in covered]
    codes = {label: index for index, label in enumerate(dict.fromkeys(labels))}
    agreement = compare_partitions(found, [codes[label] for label in labels])
    return TruthComparison(
        nodes=len(covered),
        found_communities=len(set(found)),
        truth_communities=len(set(labels)),
        nmi=normalized_mutual_information(found, labels),
        adjusted_mutual_information=adjusted_mutual_information(found, labels),
        adjusted_rand_index=agreement["adjusted_rand_index"],
        variation_of_information=variation_of_information(found, labels),
        caveat=(
            "§36.4 rests on an assumption the chapter then takes apart: that node metadata go "
            'hand in hand with the network structure. "Node metadata and structural network '
            'communities are rarely the same thing" (p. 527), and a network can be wired by '
            "several attributes at once, so a low agreement is as likely to be about the labels "
            "as about the partition."
        ),
    )


def evaluate_partition(
    graph: nx.Graph,
    communities: Sequence[Iterable[Node]],
    *,
    truth: Mapping[Node, Hashable] | Sequence[Iterable[Node]] | None = None,
    resolution: float = 1.0,
    link_prediction: bool = True,
    share: float = HOLDOUT_SHARE,
    nulls: int = NULL_PARTITIONS,
    seed: int | None = None,
) -> PartitionScores:
    """The whole of chapter 36 over one partition: §36.1 to §36.4, with no verdict.

    ``communities`` is a disjoint partition, as a list of node collections. It does not have to
    cover the network -- a clustering over features drops the nodes that had none -- and when it
    does not, everything here is computed on the induced subgraph of the nodes it does cover, and
    a note says how many were left out. Modularity has no meaning on a partition of part of a
    graph, so the alternative would be to print nothing.

    ``resolution`` must be the one the partition was found at: modularity at another resolution
    is a different function and comparing the two says nothing (p. 517). ``truth`` adds §36.4 and
    is either node metadata (a mapping from node to label) or another partition. Weights are read
    for modularity, because that is the number the partition was optimised for, and ignored by
    every §36.2 measure, whose formulas count edges; the note says so when the graph is weighted.

    A directed network is flattened first (§6.2), because the standard modularity of p. 515 and
    every measure in §36.2 are defined on undirected graphs, and the note says so.
    """
    view, flattened = undirected_view(graph)
    partition = [sorted(set(community), key=str) for community in communities]
    partition = [community for community in partition if community]
    membership = _membership(partition)
    notes: list[str] = []
    if flattened:
        notes.append(f"Every number here was computed {flattened} (§6.2, §36.1 p. 515).")

    unknown = [node for node in membership if node not in view]
    if unknown:
        msg = (
            f"evaluate_partition() was given {len(unknown)} node(s) that are not in the graph, "
            f"starting with {unknown[0]!r}"
        )
        raise ValueError(msg)
    if len(membership) < view.number_of_nodes():
        left_out = view.number_of_nodes() - len(membership)
        view = view.subgraph(membership).copy()
        notes.append(
            f"The partition covers {len(membership)} of the network's "
            f"{len(membership) + left_out} nodes, so every number here is computed on the "
            f"subgraph those nodes induce; {left_out} node(s) and their edges are outside it."
        )

    weighted = any("weight" in data for _, _, data in view.edges(data=True))
    if weighted:
        notes.append(
            "Modularity reads the edge weights, because that is the quantity the partition was "
            "optimised for; §36.2's measures count edges, as the chapter's formulas do "
            "(p. 519-522). The two therefore answer slightly different questions about the same "
            "communities."
        )
    scores = PartitionScores(
        frame=str(view.graph.get("frame", "")),
        nodes=view.number_of_nodes(),
        edges=view.number_of_edges(),
        communities=len(partition),
        sizes=[len(community) for community in partition],
        modularity=(
            float(nx.community.modularity(view, partition, weight="weight", resolution=resolution))
            if partition and view.number_of_edges()
            else 0.0
        ),
        resolution=resolution,
        weighted=weighted,
        resolution_limit=resolution_limit(view),
        notes=notes,
    )
    if not partition or not view.number_of_edges():
        notes.append(
            "There is nothing to evaluate: the partition is empty or the network has no edges."
        )
        return scores

    scores.per_community = _community_scores(view, partition, membership)
    scores.flagged = [row.index for row in scores.per_community if row.below_resolution_limit]
    coverage, performance = nx.community.partition_quality(view, [set(c) for c in partition])
    scores.averages = {
        "modularity": scores.modularity,
        "coverage": float(coverage),
        "performance": float(performance),
        **{key: _mean([row.value(key) for row in scores.per_community]) for key in PER_COMMUNITY},
    }
    if truth is not None:
        scores.truth = _compare_to_truth(partition, truth, sorted(view.nodes, key=str))
    if link_prediction:
        scores.link_prediction = link_prediction_score(
            view, partition, share=share, nulls=nulls, seed=seed
        )
    return scores


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    """A number, or a dash where the measure was undefined for this community."""
    if math.isnan(value):
        return "-"
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _resolution_sentence(scores: PartitionScores) -> str:
    """What the resolution limit found, in the chapter's terms (§36.1, p. 516-517)."""
    limit = scores.resolution_limit
    if not scores.flagged:
        return (
            f"No community has fewer than sqrt(2|E|) = {limit:.1f} internal edges, so none of "
            "them is at the size where modularity maximisation prefers to merge it into a "
            'neighbour (p. 516: it "accepts partitions only of a comparable size with the size '
            'of the network").'
        )
    named = ", ".join(
        f"#{scores.per_community[i].index + 1} ({scores.per_community[i].internal_edges} edges)"
        for i in scores.flagged[:10]
    )
    more = f", and {len(scores.flagged) - 10} more" if len(scores.flagged) > 10 else ""
    return (
        f"{len(scores.flagged)} of {scores.communities} communities have fewer than "
        f"sqrt(2|E|) = {limit:.1f} internal edges: {named}{more}. At that size modularity "
        "maximisation prefers to merge them into a neighbour even when they are clearly "
        "distinct, so their boundaries are the algorithm's resolution and not a finding about "
        "the network."
    )


def _modularity_sentence(scores: PartitionScores) -> str:
    """The reading of the modularity value itself, including the negative case (p. 512)."""
    if scores.modularity < 0:
        return (
            "A negative modularity is a reading, not a bug: these communities hold fewer edges "
            "than a configuration model with the same degrees would put in them. The domain runs "
            "to -0.5, and the book's own example of it is a disassortative partition, where "
            "nodes of the same group deliberately avoid each other (p. 512, §30.2)."
        )
    if scores.modularity < 0.05:
        return (
            "A modularity this close to zero is what the partition that puts every node in one "
            "community scores (p. 513): the communities hold about as many edges as chance would "
            "give them."
        )
    return (
        f"The ceiling for {scores.communities} communities is 1 - 1/k = "
        f"{scores.max_modularity:.4f}, which only {scores.communities} equal and mutually "
        "disconnected groups reach; the domain runs from -0.5 to +1 (p. 512)."
    )


def render_evaluation(scores: PartitionScores) -> list[str]:
    """The ``## Community evaluation`` section: the battery, with what each measure favours."""
    lines: list[str] = [
        "## Community evaluation",
        "",
        f"**Implements.** {CHAPTER} — modularity and its resolution limit, the other topological "
        "measures, the partition as a link predictor, and mutual information against a ground "
        "truth. Chapter 36 is a *battery*: \"what's 'best' depends on what you want to use your "
        'communities for" (p. 510), so every measure below is printed with what it wants and '
        "which community size pushes it that way, and none of them is a verdict.",
        "",
        f"**Sampling frame.** {scores.frame or 'not recorded'}",
        "",
        f"n = {scores.nodes:,} nodes and {scores.edges:,} edges in "
        f"{scores.communities} communities (sizes "
        f"{', '.join(str(size) for size in scores.sizes[:12])}"
        f"{', …' if len(scores.sizes) > 12 else ''}).",
        "",
        "**Null models.** Modularity carries one inside it — a configuration model with the same "
        "degrees (p. 511) — and whether this partition beats what that null actually achieves is "
        "the `### Null model` check elsewhere in this report. §36.2's measures have **no** null "
        "and are descriptions of this partition, which is why each is printed with the community "
        "size that flatters it. §36.3's AUC is tested against the same community sizes with the "
        "labels dealt out at random.",
        "",
    ]
    lines += [
        "### Modularity (§36.1)",
        "",
        f"- modularity {_num(scores.modularity)} at resolution {_num(scores.resolution, 2)}"
        f"{', on the weighted graph' if scores.weighted else ', unweighted'}",
        f"- {_modularity_sentence(scores)}",
        f"- resolution limit: {_resolution_sentence(scores)}",
        "",
        "### What each measure wants (§36.2)",
        "",
    ]
    lines += _table(
        ["measure", "value", "wants", "size pushes it", "defined in"],
        [
            [
                measure.name,
                _num(scores.averages.get(measure.key, math.nan)),
                measure.wants,
                measure.favours,
                measure.section,
            ]
            for measure in MEASURES
        ],
    )
    lines += [
        "Per-community values follow -- `inside` is the book's |E_C| and `boundary` its "
        "|E_B,C| -- and the column above is their mean, over the communities each measure is "
        "defined on. Conductance and coverage are the pair to read together: "
        "conductance falls as communities grow and coverage rises with them, so a partition that "
        "improves both has improved, and a partition that improves one has merged.",
        "",
    ]
    lines += _table(
        ["#", "size", "inside", "boundary", *(_BY_KEY[key].name for key in PER_COMMUNITY)],
        [
            [
                str(row.index + 1),
                f"{row.size:,}",
                f"{row.internal_edges:,}",
                f"{row.boundary_edges:,}",
                *(_num(row.value(key), 3) for key in PER_COMMUNITY),
            ]
            for row in scores.per_community
        ],
    )
    lines += _render_link_prediction(scores)
    lines += _render_truth(scores)
    if scores.notes:
        lines += [f"> {note}" for note in scores.notes] + [""]
    return lines


def _render_link_prediction(scores: PartitionScores) -> list[str]:
    """§36.3, or the sentence that says why the test did not run."""
    lines = ["### The partition as a link predictor (§36.3)", ""]
    prediction = scores.link_prediction
    if prediction is None:
        return [
            *lines,
            f"Not run: §36.3 holds a share of the edges out and this network has "
            f"{scores.edges:,}, below the {MIN_HOLDOUT_EDGES} at which a "
            f"{HOLDOUT_SHARE:.0%} holdout is more than a couple of pairs.",
            "",
        ]
    null = prediction.null
    return [
        *lines,
        f"The partition predicts held-out edges with AUC {_num(prediction.auc, 3)} against "
        f"{_num(null.null_mean, 3)} for shuffled labels.",
        "",
        f"- {prediction.held_out:,} edges held out ({prediction.share:.0%} of them), "
        f"{prediction.non_edges:,} non-edges sampled as the negative class, seed "
        f"{prediction.seed if prediction.seed is not None else 'not fixed'}",
        "- score: 1 if the two nodes share a community, 0 otherwise (p. 524); ties count as "
        "half, which is what makes 0.5 the value of knowing nothing",
        f"- null model: {null.null}, {null.samples} of them — z = {_num(null.z, 2)}, "
        f"p = {_num(null.p_value, 3)}",
        "",
        f"> {prediction.caveat}",
        "",
    ]


def _render_truth(scores: PartitionScores) -> list[str]:
    """§36.4, or the sentence that says there is no ground truth to compare with."""
    lines = ["### Against a ground truth (§36.4)", ""]
    truth = scores.truth
    if truth is None:
        return [
            *lines,
            "No ground truth was given, so nothing is compared. §36.4 needs node metadata that "
            "somebody is willing to call the true communities, and a corpus network has none: "
            "`analyze --by <attribute>` is where an attribute the corpus recorded is put beside "
            'the grouping, and it is deliberately not called truth — "in real observed networks '
            'metadata is just data" (p. 527).',
            "",
        ]
    return [
        *lines,
        f"n = {truth.nodes:,} nodes carrying a label, in {truth.truth_communities} "
        f"ground-truth group(s), against this run's {truth.found_communities} communities.",
        "",
        f"- normalised mutual information: {_num(truth.nmi, 3)}",
        f"- adjusted mutual information: {_num(truth.adjusted_mutual_information, 3)}",
        f"- adjusted Rand index: {_num(truth.adjusted_rand_index, 3)}",
        f"- variation of information: {_num(truth.variation_of_information, 3)} bits",
        "",
        "NMI is not corrected for chance and grows with the number of communities — the book's "
        "own pair of independently drawn random vectors scores NMI 0.09 and AMI -0.22 (p. 526) — "
        "so the adjusted pair is what says whether there is agreement at all, and the NMI is "
        "only comparable with another NMI over the same number of groups.",
        "",
        f"> {truth.caveat}",
        "",
    ]


def evaluation_payload(scores: PartitionScores) -> dict[str, Any]:
    """The same battery as plain JSON-able data. ``nan`` becomes ``null``, never 0."""
    payload: dict[str, Any] = {
        "implements": f"Atlas {CHAPTER} (community evaluation)",
        "frame": scores.frame,
        "nodes": scores.nodes,
        "edges": scores.edges,
        "communities": scores.communities,
        "sizes": scores.sizes,
        "modularity": scores.modularity,
        "max_modularity": scores.max_modularity,
        "resolution": scores.resolution,
        "weighted": scores.weighted,
        "resolution_limit": scores.resolution_limit,
        "flagged_by_resolution_limit": [index + 1 for index in scores.flagged],
        "averages": {key: _jsonable(value) for key, value in scores.averages.items()},
        "measures": [
            {
                "key": measure.key,
                "name": measure.name,
                "section": measure.section,
                "wants": measure.wants,
                "favours": measure.favours,
                "meaning": measure.meaning,
            }
            for measure in MEASURES
        ],
        "per_community": [
            {
                "index": row.index + 1,
                "size": row.size,
                "internal_edges": row.internal_edges,
                "boundary_edges": row.boundary_edges,
                "below_resolution_limit": row.below_resolution_limit,
                **{key: _jsonable(row.value(key)) for key in PER_COMMUNITY},
            }
            for row in scores.per_community
        ],
        "notes": scores.notes,
    }
    prediction = scores.link_prediction
    if prediction is not None:
        payload["link_prediction"] = {
            "auc": _jsonable(prediction.auc),
            "share": prediction.share,
            "held_out": prediction.held_out,
            "non_edges": prediction.non_edges,
            "seed": prediction.seed,
            "null": prediction.null.null,
            "null_mean": prediction.null.null_mean,
            "samples": prediction.null.samples,
            "z_score": prediction.null.z,
            "p_value": _jsonable(prediction.null.p_value),
            "caveat": prediction.caveat,
        }
    truth = scores.truth
    if truth is not None:
        payload["truth"] = {
            "nodes": truth.nodes,
            "found_communities": truth.found_communities,
            "truth_communities": truth.truth_communities,
            "normalized_mutual_information": truth.nmi,
            "adjusted_mutual_information": truth.adjusted_mutual_information,
            "adjusted_rand_index": truth.adjusted_rand_index,
            "variation_of_information": truth.variation_of_information,
            "caveat": truth.caveat,
        }
    return payload


def _jsonable(value: float) -> float | None:
    """``nan`` is not JSON, and "undefined" is not 0.0, so it travels as ``null``."""
    return None if math.isnan(value) else value
