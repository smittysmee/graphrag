"""Is this directed network an organisation chart, and how badly is it not (ch. 33).

Chapter 33 is about *flow* hierarchies, and it is explicit that the other two senses of the word
live elsewhere in the book: an **order** hierarchy "is nothing more than a different point of
view of node centrality" and belongs to chapter 14, a **nested** hierarchy "is equivalent to
performing hierarchical community discovery" and belongs to chapter 37 (§33.1, pp. 463-464).
What is left is the one this module measures: *"nodes in a higher level connect to nodes at the
level directly beneath it, and can be seen as managers spreading information or messages to the
lower levels"* (p. 464). The chapter then says the sentence every function here inherits: *"From
now on, we always assume that the network we're analyzing is directed"* (p. 465). Handed an
undirected graph, each one refuses and says which chapter to read instead.

**Four measures, because no one of them is right.** The chapter's own argument is that each
score is wrong in a way the next one fixes and the next one breaks again, so the report prints
all four side by side rather than picking one:

``cycle_share`` / flow hierarchy (§33.2)
    The share of edges that lie on a cycle, which is the share of edges inside a strongly
    connected component. *"The fewer edges are part of a cycle in a network, the more
    hierarchical it is"* (p. 466). **Too lenient:** it "says that any and all directed acyclic
    graphs are perfect hierarchies", including Figure 33.5's wheel with one edge flipped and its
    chart with more bosses than workers (p. 467).
``global_reach_centrality`` (§33.3)
    How far the best-connected node's reach stands above everybody else's. **Too strict:** "the
    only perfect hierarchy is a star", so a tree of depth three scores 0.898 and not 1 (p. 468).
``arborescence`` (§33.4)
    Chop the network down to a tree in which every node has exactly one boss and count what
    survived. *"Arborescence is a very punitive measure, much more than cycle-based flow
    hierarchy, but less so than GRC"* (p. 470).
``agony`` (§33.5)
    Put the nodes on levels and pay ``l_u - l_v + 1`` for every arrow that points up. Zero on
    any DAG, so it inherits §33.2's leniency, but unlike §33.2 it *grades* the violations: a
    back-edge across three levels costs more than one across a single level (Figure 33.8).

``hierarchy_type`` (§33.1) names which of the chapter's shapes the network is, with the checks
that decided it, and ``layered_layout`` (§33.6) turns the layering into coordinates so a drawing
command can place the levels without re-deriving them.

**Everything here is compared with the configuration model**, not with an absolute scale. None
of the four scores has a value that means "hierarchical" on its own: the chapter's exercises ask
for exactly this comparison -- *"generate 25 versions of the network with the same degree
distributions [...] calculate how many standard deviations the observed value is above or below
the average"* (§33.8, exercises 1 and 4) -- and ``graphrag.sna.null.configuration`` rewires a
digraph with the three-arc swap that holds in- and out-degree fixed.

**Where the local reach comes from.** §33.3 is built on §14.3's reach centrality, which lives in
:func:`graphrag.sna.measures.reach`; :func:`local_reach` here is that function with ``hops=None``
pinned, since the GRC is defined on the *unbounded* reach and chapter 14 defaults to a bounded
one. Nothing in this module re-implements it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_array

from graphrag.sna.matrices import Node
from graphrag.sna.measures import reach, reciprocity
from graphrag.sna.null import Significance, configuration, significance
from graphrag.sna.paths import components

#: The shapes §33.1-33.4 distinguish, from the perfect hierarchy to the one the book says has "a
#: hierarchicalness of zero by definition". :func:`hierarchy_type` returns one of these.
HIERARCHY_TYPES: tuple[str, ...] = (
    "empty",
    "arborescence",
    "arborescence forest",
    "directed acyclic",
    "cyclic",
    "strongly connected",
)

#: How many degree-preserving rewirings the section tests against when the caller does not say.
#: The same 50 ``graphrag sna analyze --samples`` defaults to, so the two sections in one report
#: are tested against null samples of the same size.
NULL_SAMPLES: int = 50

#: Above this many arcs, :func:`agony` stops solving the linear program of §33.5 exactly and
#: descends from a layering instead. The program has ``|V| + |E|`` variables and ``|E|``
#: constraints, which HiGHS solves in seconds at five thousand arcs and minutes well before a
#: corpus entity network's hundred thousand -- and the null model solves it once per rewiring.
AGONY_EXACT_EDGES: int = 5_000

#: How many improvement passes the agony descent makes before it stops. It stops early when a
#: pass changes nothing, which is the usual case after two or three.
AGONY_PASSES: int = 20

#: How many sweeps the layered drawing makes over the layers when it orders them (§33.6).
LAYOUT_SWEEPS: int = 4

#: How many nodes or edges a list in the report prints before it is truncated.
TOP_EDGES: int = 10


def _directed_or_refuse(graph: nx.Graph) -> nx.DiGraph:
    """The graph if it is directed, and the chapter's own refusal if it is not (§33.1, p. 465).

    Every measure in this chapter is degenerate on an undirected network rather than merely
    inapplicable, which is why this raises instead of flattening: reach is component membership,
    so every node in a component ties and GRC is 0; every edge lies on a cycle of length two, so
    the cycle share is 1; and agony is the edge count, since no level assignment can point both
    ends of an edge downward. Those are four numbers that look like an answer.
    """
    if not graph.is_directed():
        msg = (
            "chapter 33 measures flow hierarchies and assumes a directed network: 'From now on, "
            "we always assume that the network we're analyzing is directed' (§33.1, p. 465). "
            "This network is undirected, where every one of these measures is degenerate rather "
            "than merely inapplicable -- GRC is 0, every edge is on a two-cycle, agony is the "
            "edge count. The order hierarchy of an undirected network is its centrality ranking "
            "(ch. 14: sna analyze) and the nested one is hierarchical community discovery "
            "(ch. 37); for the flow hierarchy build --network relations."
        )
        raise ValueError(msg)
    return graph


def _sorted(nodes: Iterable[Node]) -> list[Node]:
    """Node ids in a reproducible order, whatever they are. ``paths._sorted``'s rule."""
    listed = list(nodes)
    try:
        return sorted(listed)
    except TypeError:
        return sorted(listed, key=str)


def _arcs(graph: nx.DiGraph) -> list[tuple[Node, Node]]:
    """The edges every measure here is defined on: the self loops are dropped.

    A self loop is a cycle of length one under §33.2 (it lies inside its own node's strongly
    connected component) and :func:`cycle_share` counts it as such. It is useless everywhere
    else: no ranking can give it zero agony, since ``l_u - l_u + 1 = 1`` whatever the level, and
    no arborescence can hold it, since the root is the node with in-degree zero and a loop gives
    itself one. So the optimisations drop it and the report says how many were dropped rather
    than letting a self-related entity add a fixed unavoidable cost to the agony of the network.
    """
    return [(u, v) for u, v in graph.edges() if u != v]


# -------------------------------------------------------------------------- §33.2 cycles


@dataclass(frozen=True)
class Cycles:
    """How much of the network lies on a cycle: §33.2's flow hierarchy, and its complement."""

    edges: int
    #: Edges whose two endpoints are in the same strongly connected component, which is exactly
    #: the set of edges lying on a cycle (§10.4 for the equivalence, §33.2 for the reading).
    on_cycles: int
    #: Strongly connected components holding more than one node -- the blue blobs of Figure
    #: 33.4(b), each one a knot of nodes that all take orders from each other.
    knots: int
    #: The size of the largest of them, in nodes.
    largest: int
    #: Self loops, counted in ``on_cycles`` and excluded from everything else here.
    self_loops: int

    @property
    def share(self) -> float:
        """The share of edges on a cycle. 0 on a DAG, 1 on a strongly connected network."""
        return self.on_cycles / self.edges if self.edges else 0.0

    @property
    def flow_hierarchy(self) -> float:
        """§33.2's flow hierarchy: *"the ratio between the sum of the edge weights in the
        condensed graph and the number of edges in the original graph"* (p. 466).

        Which is ``1 - share``, and the book's own worked example says why: condensing Figure
        33.4(a)'s 20 edges leaves 11 edges of total weight 12, "because of the edge of weight
        two coming out of node 2", for 12/20 = 0.6. The weights are there to keep the count of
        *original* edges that survived the condensation, and an original edge survives exactly
        when its endpoints ended up in two different components. So counting the edges inside
        components and subtracting is the same number with no bookkeeping.

        Undefined for a network with no edges; 1.0 is returned, because a network with nothing
        pointing the wrong way is the vacuous perfect hierarchy rather than the worst one.
        """
        return 1.0 - self.share if self.edges else 1.0


def cycle_share(graph: nx.Graph) -> Cycles:
    """The share of edges that lie on a cycle (§33.2).

    *"Cycles are natural enemies of hierarchies. There is no way to have a perfect hierarchy if
    you have a cycle in your network. In a perfect hierarchy, you can always tell who's your
    boss. Your boss will never take orders from you, nor from your peer, and even less so from
    any of your underlings"* (pp. 465-466). So the simplest hierarchicalness score counts the
    edges that break that: *"the simplest way to estimate the hierarchicalness of a directed
    network is to count the number of edges involved in a cycle"* (p. 466).

    An edge is on a cycle if and only if its two endpoints lie in the same strongly connected
    component, which is what makes this one pass over §10.4's machinery rather than a search for
    cycles: :func:`graphrag.sna.paths.components` supplies the component counts this dataclass
    prints, and the condensation supplies the per-node component map the edge test needs
    (``ComponentSet`` keeps the sizes, not the map, which is the only reason both are called).

    **What the book says about this number and this module repeats.** It is too lenient: every
    DAG scores a perfect 1, including both counter-examples of Figure 33.5, and *"one should not
    be able to go from a perfect hierarchy to less than 50% hierarchicalness by simply flipping
    one direction"* (p. 467). Print it beside the GRC and the agony, never alone.
    """
    directed = _directed_or_refuse(graph)
    report = components(directed)
    strong = report.strong
    sizes = strong.sizes if strong is not None else ()
    mapping: dict[Node, int] = dict(nx.condensation(directed).graph["mapping"])
    inside = sum(1 for u, v in directed.edges() if mapping[u] == mapping[v])
    return Cycles(
        edges=directed.number_of_edges(),
        on_cycles=inside,
        knots=sum(1 for size in sizes if size > 1),
        largest=max(sizes) if sizes else 0,
        self_loops=int(nx.number_of_selfloops(directed)),
    )


# ------------------------------------------------------------- §33.3 global reach centrality


@dataclass(frozen=True)
class GlobalReach:
    """§33.3's GRC, with the local reach centralities it averages the differences of."""

    nodes: int
    #: ``GRC = (1/(|V|-1)) * sum_v (LRC_max - LRC_v)``, in [0, 1]. ``nan`` on a network with
    #: fewer than two nodes, where the denominator is zero and the quantity is undefined.
    grc: float
    #: The local reach centrality of every node: the share of the other ``|V| - 1`` nodes it can
    #: reach by a directed path (§14.3, §33.3).
    local: dict[Node, float] = field(default_factory=dict)
    maximum: float = 0.0
    mean: float = 0.0
    #: The nodes holding that maximum -- the *"overseer that sees all and knows all"* (p. 467) --
    #: truncated to :data:`TOP_EDGES` names.
    heads: tuple[Node, ...] = ()
    #: How many nodes tie for it, which is exercise 2's question: "Is there a single head of the
    #: hierarchy or multiple? How many?"
    head_count: int = 0


def local_reach(graph: nx.Graph) -> dict[Node, float]:
    """The local reach centrality of every node (§14.3, as §33.3 uses it).

    *"The reach centrality of node v in a directed network is the fraction of nodes it can reach
    using directed paths originating from itself"* (p. 467). The fraction is over the other
    ``|V| - 1`` nodes: a node does not reach itself, a sink scores 0, and a node that reaches
    everybody scores 1.

    This is :func:`graphrag.sna.measures.reach` with ``hops=None`` -- chapter 14's measure, whose
    unbounded case is the one §33.3 averages the differences of -- and it is a wrapper rather
    than a second implementation for exactly that reason. What it adds is the refusal: §14.3
    offers a bounded *m*-reach and defaults to it, because reach without a radius says nothing
    on an undirected network, while §33.3's GRC is defined on the unbounded reach of a directed
    network and on nothing else. Pinning ``hops=None`` here means a change to that default
    cannot silently redefine the GRC.

    ``{}`` for an empty graph, and all-zero for a single node, which reaches nothing because
    there is nothing to reach.
    """
    directed = _directed_or_refuse(graph)
    return dict(reach(directed, hops=None))


def global_reach_centrality(graph: nx.Graph) -> GlobalReach:
    """How far the most far-seeing node stands above the rest (§33.3).

    *"A network has a strong hierarchy if there is a node which has an overwhelming reaching
    power compared to the average of all other nodes. Or, to put it in other words, if there is
    an overseer that sees all and knows all"* (p. 467). The book's formula, with ``LRC_MAX`` the
    largest local reach centrality in the network:

        ``GRC = (1 / (|V| - 1)) * sum_{v in V} (LRC_MAX - LRC_v)``

    It sits in [0, 1] because the sum has ``|V| - 1`` non-trivial terms, each at most
    ``LRC_MAX <= 1``. **GRC = 1** is the star -- one node reaching everything, nothing else
    reaching anything -- and the book is blunt that this is the measure's flaw rather than its
    virtue: *"we will never get a perfect GRC score if there is more than one node with non-zero
    local reach centrality [...] if for cycle-based flow hierarchy the perfect hierarchy is a
    DAG, for GRC the only perfect hierarchy is a star"* (p. 468). A perfect binary tree of depth
    three scores 0.898, which is the 0.89 the book reads off Figure 33.6. **GRC = 0** is every
    node reaching equally far, which includes the two cases that could not be less alike: a
    single directed cycle, where everybody reaches everybody, and a network with no edges at
    all, where nobody reaches anybody.

    Undefined for fewer than two nodes: ``grc`` is ``nan`` and the report says so rather than
    printing a zero that would read as "perfectly flat".
    """
    directed = _directed_or_refuse(graph)
    scores = local_reach(directed)
    total = len(scores)
    if total < 2:
        return GlobalReach(nodes=total, grc=math.nan, local=scores)
    largest = max(scores.values())
    heads = [node for node in _sorted(scores) if scores[node] >= largest]
    grc = sum(largest - value for value in scores.values()) / (total - 1)
    return GlobalReach(
        nodes=total,
        grc=grc,
        local=scores,
        maximum=largest,
        mean=sum(scores.values()) / total,
        heads=tuple(heads[:TOP_EDGES]),
        head_count=len(heads),
    )


# ------------------------------------------------------------------- §33.1 types of hierarchy


@dataclass(frozen=True)
class HierarchyType:
    """Which of the chapter's shapes this network is, and the checks that decided it."""

    #: One of :data:`HIERARCHY_TYPES`.
    name: str
    #: One sentence per check, in the order they were made: cycles, roots, bosses, reciprocity,
    #: pieces. Printed under the name, because the name alone is a label and these are evidence.
    reasons: tuple[str, ...]
    acyclic: bool
    #: Nodes with in-degree 0: candidates for §33.4's CEO, *"the head of the hierarchy"*.
    roots: int
    #: Nodes with more than one boss -- in-degree above one -- which is what separates an
    #: arborescence from a plain DAG (§33.4).
    many_bosses: int
    #: §10.3's reciprocity: how often the corpus states a relation in both directions. A perfect
    #: hierarchy has none, since *"your boss will never take orders from you"* (p. 465).
    reciprocity: float
    weak_components: int

    @property
    def sentence(self) -> str:
        """The name with what the chapter says about networks of that shape."""
        return _TYPE_SENTENCES[self.name]


#: What each shape means for a reader, in the chapter's own terms. Keyed by
#: :data:`HIERARCHY_TYPES`.
_TYPE_SENTENCES: dict[str, str] = {
    "empty": "The network has no nodes, so there is nothing to be a hierarchy of.",
    "arborescence": (
        "An arborescence: one node with no boss, every other node with exactly one, and no "
        "cycles. §33.4's perfect hierarchy -- 'every arborescence is a perfect hierarchy: all "
        "nodes have a single boss, there are no cycles, and there is one node with no bosses -- "
        "the CEO' (p. 468). Note that GRC still refuses to score it 1 unless it is a star, which "
        "is §33.3's flaw and not this network's."
    ),
    "arborescence forest": (
        "An arborescence forest: every node has at most one boss and there are no cycles, but "
        "there is more than one head. §33.4 ends up here too -- 'technically speaking, this "
        "technique reduces an arbitrary directed network into an arborescence forest, not an "
        "arborescence' (p. 469) -- so this is a perfect hierarchy per piece with no single CEO "
        "over them."
    ),
    "directed acyclic": (
        "A DAG, but not a tree: some nodes take orders from more than one boss. §33.2's "
        "cycle-based flow hierarchy calls this a perfect hierarchy, and §33.2 says that verdict "
        "is too lenient: 'it is very easy to construct toy examples of less-than-ideal "
        "structures that this cycle-based flow hierarchy will consider perfect' (p. 467), "
        "Figure 33.5's wheel with one flipped edge and its chart with more bosses than workers "
        "among them. Read the arborescence score and the GRC, which are not fooled."
    ),
    "cyclic": (
        "Cyclic: some edges lie on a cycle, so for those nodes there is no answer to 'who is "
        "your boss'. §33.2: 'a cycle means that your boss gives you an order, you pass it down "
        "to one of your underlings and, somehow, they give it back to your boss' (p. 466). How "
        "much of the network that is, is the cycle share below; a cycle in a corpus network is "
        "usually two entities each described in terms of the other, not an error."
    ),
    "strongly connected": (
        "One strongly connected component holding every node: everything reaches everything. "
        "§33.2 says what that scores -- 'any directed network composed by a single strongly "
        "connected component has a hierarchicalness of zero by definition' (p. 466) -- and GRC "
        "agrees, because every node reaches equally far."
    ),
}


def hierarchy_type(graph: nx.Graph) -> HierarchyType:
    """Name the shape of this directed network, with the checks that decided it (§33.1, §33.4).

    §33.1 sorts hierarchies into order, nested and flow, and sends the first two elsewhere: the
    order hierarchy *"is nothing more than a different point of view of node centrality"*
    (ch. 14), the nested one *"is equivalent to performing hierarchical community discovery"*
    (ch. 37). This function classifies within the third, along the four questions the rest of
    the chapter turns into scores:

    - **Is it acyclic?** §33.2's question. A cycle is the one thing a perfect hierarchy cannot
      hold.
    - **Is there a single root?** §33.4's CEO, the node with in-degree zero.
    - **Does every other node have exactly one boss?** *"An arborescence is a directed tree in
      which all nodes have in-degree of one, except the root, which has in-degree of zero"*
      (p. 468).
    - **Is reciprocity zero?** §10.3's ratio, which is the pairwise form of the same question:
      *"your boss will never take orders from you"* (p. 465). It cannot be positive on an
      acyclic network -- a reciprocated pair is a cycle of length two -- so it is reported as
      evidence rather than used as a test.

    The verdict is a shape, not a score. Two networks can both be "cyclic" with 1% and 99% of
    their edges on those cycles, which is why :func:`cycle_share` is printed next to it.
    """
    directed = _directed_or_refuse(graph)
    total = directed.number_of_nodes()
    if total == 0:
        return HierarchyType(
            name="empty",
            reasons=("The network has no nodes.",),
            acyclic=True,
            roots=0,
            many_bosses=0,
            reciprocity=0.0,
            weak_components=0,
        )
    acyclic = bool(nx.is_directed_acyclic_graph(directed))
    in_degrees = {node: int(directed.in_degree(node)) for node in directed.nodes()}
    roots = [node for node, degree in in_degrees.items() if degree == 0]
    many = [node for node, degree in in_degrees.items() if degree > 1]
    weak = int(nx.number_weakly_connected_components(directed))
    strong_sizes = [len(group) for group in nx.strongly_connected_components(directed)]
    reciprocal = reciprocity(directed)
    name = _type_name(
        acyclic=acyclic,
        roots=len(roots),
        many=len(many),
        weak=weak,
        biggest_knot=max(strong_sizes) if strong_sizes else 0,
        nodes=total,
    )
    reasons = (
        (
            "No edge lies on a cycle."
            if acyclic
            else f"The largest strongly connected component holds {max(strong_sizes):,} of "
            f"{total:,} nodes, so those nodes lie on cycles."
        ),
        f"{len(roots):,} node(s) have in-degree 0: candidates for the head of the hierarchy.",
        (
            "Every node with a boss has exactly one."
            if not many
            else f"{len(many):,} node(s) have more than one incoming edge, so they take orders "
            "from more than one boss."
        ),
        f"Reciprocity {reciprocal:.4f} (§10.3): the share of connected pairs stated both ways.",
        f"{weak:,} weakly connected piece(s).",
    )
    return HierarchyType(
        name=name,
        reasons=reasons,
        acyclic=acyclic,
        roots=len(roots),
        many_bosses=len(many),
        reciprocity=reciprocal,
        weak_components=weak,
    )


def _type_name(
    *, acyclic: bool, roots: int, many: int, weak: int, biggest_knot: int, nodes: int
) -> str:
    """The shape the five counts imply. See :func:`hierarchy_type`."""
    if not acyclic:
        return "strongly connected" if biggest_knot == nodes else "cyclic"
    if many:
        return "directed acyclic"
    return "arborescence" if roots == 1 and weak == 1 else "arborescence forest"


# -------------------------------------------------------------------- §33.4 arborescences


@dataclass(frozen=True)
class Arborescence:
    """The tree of one boss per node that §33.4 cuts the network down to, and what it cost."""

    #: The surviving tree. An arborescence when ``spanning``; otherwise the *arborescence
    #: forest* of p. 469, one tree per piece.
    tree: nx.DiGraph
    #: The nodes with in-degree 0 in that tree -- §33.4's CEO, one per tree.
    roots: tuple[Node, ...]
    spanning: bool
    #: Why no single tree could span the network, empty when one did. The refusal is a value and
    #: not an exception because it is a fact about the network rather than about the call: the
    #: chapter's own procedure ends in a forest on its own Figure 33.4 example (p. 469).
    refusal: str
    #: Arcs in the network, self loops excluded (they can never be in a tree).
    edges: int
    kept: int
    weight_kept: float
    total_weight: float
    #: The heaviest arcs the tree dropped, worst first, truncated to :data:`TOP_EDGES`.
    dropped: tuple[tuple[Node, Node, float], ...]
    self_loops: int

    @property
    def score(self) -> float:
        """§33.4's arborescence score: *"we can count how many connections survived [...] The
        more edges we needed to remove to obtain an arborescence, the less the original network
        was resembling a perfect hierarchy"* (p. 469). Nine of Figure 33.4's 20 edges survive,
        for 9/20 = 0.45. 1.0 for a network with no arcs to lose."""
        return self.kept / self.edges if self.edges else 1.0

    @property
    def weight_share(self) -> float:
        """The same ratio in weight rather than in edges: how much of the evidence behind the
        network the one-boss-per-node reading keeps. Not the book's number -- §33.4 says of its
        own condensation step that "here we ignore edge weights" (p. 469) -- but the one to
        quote when the weights are counts of passages, because dropping fifty arcs each stated
        once is not the same loss as dropping one stated fifty times."""
        return self.weight_kept / self.total_weight if self.total_weight else 1.0

    @property
    def dropped_edges(self) -> int:
        return self.edges - self.kept


def arborescence(graph: nx.Graph, *, weight: str = "weight") -> Arborescence:
    """Cut the network down to one boss per node, keeping as much weight as possible (§33.4).

    *"An arborescence is a directed tree in which all nodes have in-degree of one, except the
    root, which has in-degree of zero [...] Every arborescence is a perfect hierarchy: all nodes
    have a single boss, there are no cycles, and there is one node with no bosses -- the CEO"*
    (p. 468). So the hierarchicalness score is what is left after forcing the network to be one:
    *"once all edges breaking the arborescence requirements are eliminated, we can count how
    many connections survived"* (p. 469).

    **Which edges to drop is where this departs from the chapter, deliberately.** §33.4 gets
    there in two steps -- condense the strongly connected components away, then break every
    remaining in-degree above one by keeping the edge *"coming from the node with the lowest
    closeness centrality"* (p. 469) -- and notes in the same breath that *"there are alternative
    methods to reduce a generic directed network to a DAG, which can preserve more edges"*.
    This uses the optimal one: Chu-Liu/Edmonds' maximum spanning arborescence,
    ``networkx.maximum_spanning_arborescence``, which maximises the total weight kept over every
    possible choice of one incoming edge per node, resolving the cycles as part of the same
    optimisation rather than by condensing them away first. Three consequences, all in the
    network's favour and none of them silent: the score here is never below the score the
    chapter's heuristic would give; the tree keeps original nodes, so the "boss" of an entity
    inside a knot of mutual references is a named entity and not a blob; and the answer does not
    depend on a closeness tie-break. ``weight`` names the edge attribute maximised, defaulting
    to the count of passages stating each relation, so each node's one boss is the relation the
    corpus states most often. Missing weights count as 1.

    **When no single tree spans the network**, ``spanning`` is ``False``, ``refusal`` says why
    -- most often that more than one node has in-degree zero, so all but one of them would have
    to take a boss from somewhere, or that the network comes in several pieces -- and the result
    is the maximum spanning *branching*: the forest of arborescences p. 469 says the technique
    generally produces. Its score is still ``kept / edges``, and a forest is a weaker claim than
    a tree, so the report prints the refusal above the number rather than below it.
    """
    directed = _directed_or_refuse(graph)
    simple = nx.DiGraph()
    simple.add_nodes_from(directed.nodes())
    for u, v, data in directed.edges(data=True):
        if u != v:
            simple.add_edge(u, v, **{weight: float(data.get(weight, 1.0))})
    arcs = simple.number_of_edges()
    total_weight = sum(float(data[weight]) for _, _, data in simple.edges(data=True))
    spanning = True
    refusal = ""
    try:
        tree = nx.maximum_spanning_arborescence(simple, attr=weight, default=1.0)
    except nx.NetworkXException:
        spanning = False
        refusal = _why_not_spanning(simple)
        tree = nx.maximum_branching(simple, attr=weight, default=1.0)
    tree.add_nodes_from(simple.nodes())
    kept = {(u, v) for u, v in tree.edges()}
    dropped = sorted(
        (
            (u, v, float(data[weight]))
            for u, v, data in simple.edges(data=True)
            if (u, v) not in kept
        ),
        key=lambda row: (-row[2], str(row[0]), str(row[1])),
    )
    return Arborescence(
        tree=tree,
        roots=tuple(_sorted(n for n in tree.nodes() if tree.in_degree(n) == 0)),
        spanning=spanning,
        refusal=refusal,
        edges=arcs,
        kept=tree.number_of_edges(),
        weight_kept=sum(float(simple[u][v][weight]) for u, v in kept),
        total_weight=total_weight,
        dropped=tuple(dropped[:TOP_EDGES]),
        self_loops=int(nx.number_of_selfloops(directed)),
    )


def _why_not_spanning(graph: nx.DiGraph) -> str:
    """Why no arborescence covers this network. See :func:`arborescence`.

    A spanning arborescence exists exactly when some node reaches every other, so that is the
    sentence; the two symptoms a reader can see for themselves are named beside it.
    """
    sources = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    pieces = int(nx.number_weakly_connected_components(graph))
    detail = (
        f"the network comes in {pieces:,} weakly connected pieces and a tree is one piece"
        if pieces > 1
        else (
            f"{len(sources):,} nodes have in-degree 0, and only one of them can be the root -- "
            "the others would each need a boss from somewhere"
            if len(sources) > 1
            else "every node has a boss, so no node is left to be the root of a tree over all "
            "of them"
        )
    )
    return (
        "No spanning arborescence exists: no single node reaches every other one, so no tree "
        f"rooted anywhere covers the network ({detail}). What is below is the maximum spanning "
        "*branching* -- the 'arborescence forest' of §33.4 (p. 469), one tree per head -- and "
        "the score counts its edges the same way."
    )


# -------------------------------------------------------------------------- §33.5 agony


@dataclass(frozen=True)
class Agony:
    """§33.5's agony: the price of every arrow that points up, minimised over the levels."""

    #: ``A(G, l) = sum_{(u,v) in E} max(l_u - l_v + 1, 0)`` at the levels below. An integer in
    #: every case, carried as a float because the null-model comparison averages it.
    total: float
    #: The level of each node, 0 at the top. Not unique: many rankings reach the same minimum,
    #: and the report says so rather than naming a node's level as a finding.
    ranks: dict[Node, int]
    #: How many distinct levels those ranks use.
    levels: int
    #: Whether ``total`` is the true minimum (the linear program was solved) or an upper bound
    #: on it (the descent was used because the network was too big).
    exact: bool
    #: The linear program's objective: a lower bound on the agony of the network, equal to
    #: ``total`` when ``exact``. ``nan`` when the program was not solved.
    lower_bound: float
    method: str
    #: The arcs charged something, worst first, as ``(u, v, cost)``; truncated to
    #: :data:`TOP_EDGES`.
    backward: tuple[tuple[Node, Node, int], ...]
    backward_edges: int
    edges: int
    self_loops: int
    note: str

    @property
    def per_edge(self) -> float:
        """Agony per arc. Not the book's -- §33.5 defines no normalisation -- but the total is
        proportional to the number of arcs, so this is what two networks of different sizes can
        be compared on. 0 on a DAG, 1 when every arc is flat (all nodes on one level), above 1
        when the violations span levels."""
        return self.total / self.edges if self.edges else 0.0


def agony(graph: nx.Graph, *, max_exact_edges: int = AGONY_EXACT_EDGES) -> Agony:
    """Rank the nodes so that as few arrows as possible point upwards, and price the rest (§33.5).

    *"In the agony measure we start from the assumption that we can partially order nodes into
    levels. The CEO lives at the top of the hierarchy (level 1), its immediate executive are at
    level 2 [...] If l_u < l_v then a u -> v edge is ordinary and expected. On the other hand, a
    u <- v edge will cause 'agony'"*, and that agony is *"proportional to the level difference
    between the nodes [...] the agony of the u <- v edge as: l_v - l_u + 1"* (p. 470). Summed
    over the arcs of the network, with ``(u, v)`` written as the arc from ``u`` to ``v``:

        ``A(G, l) = sum_{(u,v) in E} max(l_u - l_v + 1, 0)``

    **A note on the book's subscripts.** p. 470 prints that sum as ``max(l_v - l_u + 1, 0)``,
    which contradicts its own sentence two lines below -- *"every time l_u < l_v, we contribute
    zero to the sum"* -- and its own per-edge definition, which is about the arc ``u <- v``,
    i.e. the arc whose *source* is ``v``. Written per arc source-to-target, as above, the three
    agree: an arc from a higher level to a lower one is free, a flat arc costs 1, and an arc
    that climbs ``k`` levels costs ``k + 1``. That is also Gupte et al.'s definition, which the
    chapter cites. The ``+1`` is load-bearing, and the book says why: *"if we were to exclude
    it, we could put all nodes in the same level and obtain zero agony, which would defeat the
    purpose of the measure"*.

    **The minimisation is exact, and it is a linear program.** The chapter says only that
    *"there are efficient algorithms to estimate the agony of a directed graph"*, citing Gupte
    et al. (2011) and Tatti (2015), and both of those are built on the observation that this
    problem is the dual of a minimum-cost flow. Rather than reimplement Tatti's primal-dual
    algorithm, this solves the program itself with HiGHS through ``scipy.optimize.linprog``:
    minimise ``sum_e z_e`` subject to ``r_u - r_v - z_e <= -1`` for each arc and
    ``0 <= r <= |V| - 1``. Its constraint matrix is a network matrix with an identity block
    appended, so it is totally unimodular and every vertex of the feasible region is integral --
    which is why a solver over the reals returns whole levels and why the objective is the true
    integer minimum rather than a relaxation of it. The result is then run through the same
    coordinate descent the large-network path uses, as a cheap guard against a solver returning
    a non-vertex solution; it never improves an exact solve, and the report prints both the
    achieved total and the program's bound so a disagreement would be visible.

    Above ``max_exact_edges`` arcs the program is skipped and the ranks come from the descent
    alone, started from the longest-path layering of the condensation. ``exact`` is then
    ``False`` and the total is an **upper bound** on the true agony -- the report says so, and
    the null-model comparison is then between two upper bounds, which is a comparison worth
    less than the numbers look.

    Self loops are excluded (see :func:`_arcs`); an acyclic network has agony 0, since its
    topological order already sends every arc downward.
    """
    directed = _directed_or_refuse(graph)
    nodes = _sorted(directed.nodes())
    arcs = _arcs(directed)
    loops = int(nx.number_of_selfloops(directed))
    if not arcs:
        return Agony(
            total=0.0,
            ranks=dict.fromkeys(nodes, 0),
            levels=1 if nodes else 0,
            exact=True,
            lower_bound=0.0,
            method="none needed: the network has no arcs to point the wrong way",
            backward=(),
            backward_edges=0,
            edges=0,
            self_loops=loops,
            note="",
        )
    exact = len(arcs) <= max_exact_edges
    if exact:
        ranks, bound = _agony_program(nodes, arcs)
        method = (
            "exact: the linear program of §33.5 (Gupte et al. 2011; Tatti 2015), solved by "
            "HiGHS through scipy.optimize.linprog, whose constraint matrix is totally "
            "unimodular so its optimum is integral"
        )
        note = ""
    else:
        ranks, bound = _layer_ranks(directed), math.nan
        method = (
            "heuristic: coordinate descent from the longest-path layering of the condensation, "
            "each node moved to its own best level until nothing moves"
        )
        note = (
            f"The network has {len(arcs):,} arcs, above the {max_exact_edges:,} at which the "
            "exact program is solved, so this total is an upper bound on the agony of the "
            "network and not the agony. Two upper bounds are not a comparison: read the "
            "null-model line below with that in mind."
        )
    ranks = _descend(nodes, arcs, ranks)
    charges = [(u, v, max(ranks[u] - ranks[v] + 1, 0)) for u, v in arcs]
    backward = sorted(
        ((u, v, cost) for u, v, cost in charges if cost > 0),
        key=lambda row: (-row[2], str(row[0]), str(row[1])),
    )
    return Agony(
        total=float(sum(cost for _, _, cost in charges)),
        ranks=ranks,
        levels=len(set(ranks.values())),
        exact=exact,
        lower_bound=bound,
        method=method,
        backward=tuple(backward[:TOP_EDGES]),
        backward_edges=len(backward),
        edges=len(arcs),
        self_loops=loops,
        note=note,
    )


def _agony_program(
    nodes: Sequence[Node], arcs: Sequence[tuple[Node, Node]]
) -> tuple[dict[Node, int], float]:
    """Solve §33.5's minimisation as a linear program. See :func:`agony`.

    The variables are the ``|V|`` levels followed by the ``|E|`` per-arc costs; the objective
    counts only the costs; each arc contributes one row ``r_u - r_v - z_e <= -1``. The level
    bounds ``[0, |V| - 1]`` are what makes the feasible region a polytope at all: without them
    the program is invariant under adding a constant to every level and has no vertices, and no
    optimum is lost by them, since shifting an optimal ranking down to 0 leaves it inside.
    """
    index = {node: position for position, node in enumerate(nodes)}
    count, arc_count = len(nodes), len(arcs)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for position, (u, v) in enumerate(arcs):
        rows += [position, position, position]
        columns += [index[u], index[v], count + position]
        values += [1.0, -1.0, -1.0]
    constraints = coo_array(
        (values, (rows, columns)), shape=(arc_count, count + arc_count), dtype=np.float64
    )
    objective = np.concatenate([np.zeros(count), np.ones(arc_count)])
    bounds = [(0.0, float(max(count - 1, 0)))] * count + [(0.0, None)] * arc_count
    solution = linprog(
        objective,
        A_ub=constraints,
        b_ub=-np.ones(arc_count),
        bounds=bounds,
        method="highs",
    )
    if not solution.success:  # pragma: no cover - HiGHS does not fail on a bounded feasible LP
        return _layer_ranks_from(nodes, arcs), math.nan
    levels = np.rint(np.asarray(solution.x[:count], dtype=np.float64)).astype(int)
    return {node: int(levels[index[node]]) for node in nodes}, float(solution.fun)


def _agony_of(arcs: Iterable[tuple[Node, Node]], ranks: Mapping[Node, int]) -> int:
    """``A(G, l)`` at one ranking (§33.5, p. 470)."""
    return sum(max(ranks[u] - ranks[v] + 1, 0) for u, v in arcs)


def _descend(
    nodes: Sequence[Node],
    arcs: Sequence[tuple[Node, Node]],
    ranks: Mapping[Node, int],
    passes: int = AGONY_PASSES,
) -> dict[Node, int]:
    """Move one node at a time to its own best level until none of them moves (§33.5).

    The agony of a single node, with every other level held fixed, is a convex piecewise-linear
    function of its level whose breakpoints are at ``rank(boss) + 1`` for each incoming arc and
    ``rank(underling) - 1`` for each outgoing one, so the minimum is attained at one of those
    and the candidate set is small. This is a local method: it is the fallback when the network
    is too large for the exact program, and a no-op safety net after it.
    """
    current = {node: int(ranks[node]) for node in nodes}
    into: dict[Node, list[Node]] = {node: [] for node in nodes}
    out_of: dict[Node, list[Node]] = {node: [] for node in nodes}
    for u, v in arcs:
        out_of[u].append(v)
        into[v].append(u)
    ceiling = max(len(nodes) - 1, 0)
    for _ in range(passes):
        moved = False
        for node in nodes:
            candidates = {current[node], 0}
            candidates |= {min(current[u] + 1, ceiling) for u in into[node]}
            candidates |= {max(current[v] - 1, 0) for v in out_of[node]}
            best, cost = current[node], _node_agony(node, current, into, out_of, current[node])
            for candidate in sorted(candidates):
                here = _node_agony(node, current, into, out_of, candidate)
                if here < cost:
                    best, cost = candidate, here
            if best != current[node]:
                current[node] = best
                moved = True
        if not moved:
            break
    floor = min(current.values(), default=0)
    return {node: level - floor for node, level in current.items()}


def _node_agony(
    node: Node,
    ranks: Mapping[Node, int],
    into: Mapping[Node, Sequence[Node]],
    out_of: Mapping[Node, Sequence[Node]],
    level: int,
) -> int:
    """The agony of the arcs touching ``node`` if it sat at ``level``. See :func:`_descend`."""
    cost = sum(max(ranks[u] - level + 1, 0) for u in into[node])
    cost += sum(max(level - ranks[v] + 1, 0) for v in out_of[node])
    return cost


def _layer_ranks(graph: nx.DiGraph) -> dict[Node, int]:
    """The longest-path layering of the condensation, as a starting ranking. See §33.6."""
    return _condensation_layers(graph)


def _layer_ranks_from(nodes: Sequence[Node], arcs: Sequence[tuple[Node, Node]]) -> dict[Node, int]:
    """The same starting ranking, rebuilt from nodes and arcs when no graph is to hand."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(arcs)
    return _condensation_layers(graph)


# ------------------------------------------------------------------ §33.6 drawing hierarchies


@dataclass(frozen=True)
class Dummy:
    """A waypoint on an arc that crosses more than one layer (§33.6, the Sugiyama convention).

    A layered drawing has no place to put an arc that skips a layer: drawn straight it cuts
    through whatever is on the layer between. The standard answer, which the chapter's figures
    show without naming, is a placeholder node on each crossed layer, so the arc becomes a chain
    of one-layer segments. They are not nodes of the network and carry no measurement.
    """

    source: Node
    target: Node
    layer: int
    x: float
    y: float


@dataclass(frozen=True)
class Layout:
    """Coordinates for a layered drawing of the hierarchy (§33.6). Nothing here draws anything."""

    #: Which layering produced it, for the caption.
    layering: str
    #: The nodes of each layer, top layer first, ordered left to right as ``positions`` places
    #: them.
    layers: tuple[tuple[Node, ...], ...]
    #: ``node -> (x, y)``. ``x`` is the position within the layer, ``y`` is ``-layer``, so a
    #: plot with the y axis pointing up draws layer 0 -- the roots -- at the top.
    positions: dict[Node, tuple[float, float]]
    dummies: tuple[Dummy, ...]
    #: The arcs the layering could not send downward: ``(u, v)`` with ``layer(v) <= layer(u)``.
    #: Truncated to :data:`TOP_EDGES`.
    upward: tuple[tuple[Node, Node], ...]
    upward_edges: int
    note: str

    @property
    def depth(self) -> int:
        """How many layers the drawing has."""
        return len(self.layers)

    @property
    def widest(self) -> int:
        """The widest layer, in nodes: the chapter's *"more bosses than workers"* check (p. 467)
        is read off the shape of this list, not off any single number."""
        return max((len(layer) for layer in self.layers), default=0)


def layered_layout(
    graph: nx.Graph,
    *,
    levels: Mapping[Node, int] | None = None,
    sweeps: int = LAYOUT_SWEEPS,
) -> Layout:
    """Place every node on a level and every level on a row, for a drawing (§33.6).

    *"The arborescence approach is a further step up. Since it reduces the network to an
    arborescence, one can draw the resulting condensed graph, identifying not only the root of
    the hierarchy, but at which level each node lies. The same can be said for agony: it assigns
    each node to a level, thus you can plot the network by layering nodes vertically according
    to their assigned rank"* (p. 472).

    **The default layering is the condensation's longest path.** Collapse every strongly
    connected component to a point -- the DAG §33.2 builds -- and put each component one layer
    below the lowest of its bosses. That is the standard first step of a Sugiyama layered
    drawing and it has the property the chapter wants: every arc between two different
    components points strictly downward, so the only arcs that do not are the ones inside a
    knot, which no layering can fix. Every member of a component sits on its component's layer.

    ``levels`` overrides it with any ranking, which is how Figure 33.9(c) is drawn: pass
    :attr:`Agony.ranks` and the arcs drawn upward are exactly the arcs agony charged for. The
    mapping must cover every node; a partial one is refused rather than filled in, because a
    node placed by default would silently acquire a level the caller did not mean. Levels are
    compressed to consecutive layers, which cannot change which arcs point downward.

    **What the book defines and what this adds.** The chapter gives the layering and stops; the
    two steps that make a layered drawing readable are conventions from the Sugiyama framework
    it is describing, and both are here in their simplest form. Arcs crossing more than one
    layer get :class:`Dummy` waypoints. Nodes within a layer are ordered by the barycentre
    heuristic -- repeatedly, each node moves to the average position of its neighbours on the
    layer above, then on the layer below -- for ``sweeps`` passes, which reduces crossings
    without minimising them (minimising them is NP-hard). The dummies are placed on the straight
    line between their arc's endpoints rather than ordered with the real nodes, so a long arc
    does not bend; a drawing command that wants proper splines should order them too.

    Returns coordinates and nothing else: no figure, no file, no dependency on a plotting
    library. ATL-49 is the ticket that draws them.
    """
    directed = _directed_or_refuse(graph)
    if levels is None:
        raw = _condensation_layers(directed)
        layering = (
            "longest path on the condensation (§33.2, §33.6): every strongly connected "
            "component one layer below its lowest boss"
        )
    else:
        missing = [node for node in directed.nodes() if node not in levels]
        if missing:
            msg = (
                f"levels must cover every node; {len(missing):,} are missing, starting with "
                f"{', '.join(str(node) for node in _sorted(missing)[:3])}. A node placed on a "
                "default layer would carry a level its caller never assigned."
            )
            raise ValueError(msg)
        raw = {node: int(levels[node]) for node in directed.nodes()}
        layering = "the levels supplied by the caller (§33.6: agony ranks draw Figure 33.9(c))"
    layer_of = _compress(raw)
    order = _order_layers(directed, layer_of, sweeps)
    positions = {
        node: (float(position), float(-layer))
        for layer, row in enumerate(order)
        for position, node in enumerate(row)
    }
    upward = [(u, v) for u, v in directed.edges() if layer_of[v] <= layer_of[u]]
    return Layout(
        layering=layering,
        layers=tuple(tuple(row) for row in order),
        positions=positions,
        dummies=_dummies(directed, layer_of, positions),
        upward=tuple(_sorted(upward)[:TOP_EDGES]),
        upward_edges=len(upward),
        note=(
            f"{len(upward):,} of {directed.number_of_edges():,} arcs could not be drawn "
            "downward. Under this layering those are exactly the arcs inside a strongly "
            "connected component, which is what §33.2 says a cycle costs a drawing."
            if levels is None
            else f"{len(upward):,} of {directed.number_of_edges():,} arcs point upward or "
            "sideways under these levels: the ones §33.5 charges agony for."
        ),
    )


def _condensation_layers(graph: nx.DiGraph) -> dict[Node, int]:
    """Longest-path layering of the condensation, expanded back onto the nodes. See §33.6."""
    if graph.number_of_nodes() == 0:
        return {}
    dag = nx.condensation(graph)
    layer: dict[int, int] = {}
    for component in nx.topological_sort(dag):
        predecessors = list(dag.predecessors(component))
        layer[component] = 1 + max((layer[p] for p in predecessors), default=-1)
    return {
        node: layer[component]
        for component in dag.nodes()
        for node in dag.nodes[component]["members"]
    }


def _compress(levels: Mapping[Node, int]) -> dict[Node, int]:
    """Renumber levels to consecutive layers from 0, keeping their order."""
    ordered = sorted(set(levels.values()))
    rank = {level: position for position, level in enumerate(ordered)}
    return {node: rank[level] for node, level in levels.items()}


def _order_layers(graph: nx.DiGraph, layer_of: Mapping[Node, int], sweeps: int) -> list[list[Node]]:
    """Order each layer left to right by the barycentre heuristic. See :func:`layered_layout`."""
    depth = max(layer_of.values(), default=-1) + 1
    rows: list[list[Node]] = [[] for _ in range(depth)]
    for node in _sorted(graph.nodes()):
        rows[layer_of[node]].append(node)
    for sweep in range(max(sweeps, 0)):
        downward = sweep % 2 == 0
        indices = range(1, depth) if downward else range(depth - 2, -1, -1)
        for layer in indices:
            place = {node: position for row in rows for position, node in enumerate(row)}
            neighbours = graph.predecessors if downward else graph.successors
            fixed = layer - 1 if downward else layer + 1
            rows[layer].sort(
                key=lambda node: (
                    _barycentre(
                        [n for n in neighbours(node) if layer_of[n] == fixed], place, place[node]
                    ),
                    str(node),
                )
            )
    return rows


def _barycentre(neighbours: Sequence[Node], place: Mapping[Node, int], fallback: int) -> float:
    """The average position of ``neighbours``, or ``fallback`` when a node has none there."""
    if not neighbours:
        return float(fallback)
    return sum(place[node] for node in neighbours) / len(neighbours)


def _dummies(
    graph: nx.DiGraph,
    layer_of: Mapping[Node, int],
    positions: Mapping[Node, tuple[float, float]],
) -> tuple[Dummy, ...]:
    """One waypoint per crossed layer, on every arc that skips one. See :class:`Dummy`."""
    out: list[Dummy] = []
    for u, v in _sorted(graph.edges()):
        span = layer_of[v] - layer_of[u]
        if span <= 1:
            continue
        start, end = positions[u][0], positions[v][0]
        for step in range(1, span):
            out.append(
                Dummy(
                    source=u,
                    target=v,
                    layer=layer_of[u] + step,
                    x=start + (end - start) * step / span,
                    y=float(-(layer_of[u] + step)),
                )
            )
    return tuple(out)


# ------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class HierarchyReport:
    """Everything the ``Hierarchy`` section of a report prints (ch. 33)."""

    frame: str
    nodes: int
    edges: int
    kind: HierarchyType
    cycles: Cycles
    reach: GlobalReach
    tree: Arborescence
    levels: Agony
    layout: Layout
    samples: int
    seed: int | None
    #: The four scores against the configuration model, or ``None`` where the measure has no
    #: null to compare against (an empty network) -- each one a :class:`Significance` carrying
    #: its own sample count, so a null that could not be built reports itself.
    flow_null: Significance | None = None
    grc_null: Significance | None = None
    agony_null: Significance | None = None
    arborescence_null: Significance | None = None
    notes: tuple[str, ...] = ()


def analyse_hierarchy(
    graph: nx.Graph,
    *,
    samples: int = NULL_SAMPLES,
    seed: int | None = None,
    max_exact_edges: int = AGONY_EXACT_EDGES,
) -> HierarchyReport:
    """Run all of chapter 33 over one directed network, with its null model.

    The four scores are computed on the observation and again on ``samples`` degree-preserving
    rewirings, which is the chapter's own exercise: *"generate 25 versions of the network with
    the same degree distributions of the observed one (use the directed configuration model) and
    calculate how many standard deviations the observed value is above or below the average
    value you obtain from the null model"* (§33.8, exercise 1, repeated for GRC and the
    arborescence score in exercise 4). :func:`graphrag.sna.null.configuration` does the rewiring
    with §19.1's three-arc swap, which holds every node's in-degree *and* out-degree fixed -- so
    the comparison is with a network that has the same bosses-per-node and underlings-per-node
    distribution and no hierarchy beyond what those force.

    The tails differ by measure and are set here rather than guessed by a reader: flow
    hierarchy, GRC and the arborescence score are hierarchicalness, so the question is whether
    the observation is *above* the null; agony is its opposite, so the question is whether it is
    *below*.

    Refuses an undirected network (see :func:`_directed_or_refuse`). Costs one condensation, one
    Edmonds and one linear program per sample, so ``samples`` is the knob that decides how long
    this section takes.
    """
    directed = _directed_or_refuse(graph)
    cycles = cycle_share(directed)
    reach = global_reach_centrality(directed)
    tree = arborescence(directed)
    levels = agony(directed, max_exact_edges=max_exact_edges)
    layout = layered_layout(directed)
    flow_samples: list[float] = []
    grc_samples: list[float] = []
    agony_samples: list[float] = []
    tree_samples: list[float] = []
    for rewired in configuration(directed, samples, seed=seed):
        flow_samples.append(cycle_share(rewired).flow_hierarchy)
        grc_samples.append(global_reach_centrality(rewired).grc)
        agony_samples.append(agony(rewired, max_exact_edges=max_exact_edges).total)
        tree_samples.append(arborescence(rewired).score)
    null = "configuration (directed three-arc swaps: every in- and out-degree held fixed)"
    notes: list[str] = []
    if cycles.self_loops:
        notes.append(
            f"{cycles.self_loops:,} self loop(s) count as edges on a cycle (§33.2) and are "
            "excluded from the agony, the arborescence and the drawing, where a relation from "
            "an entity to itself has no reading: it can be given no level that makes it free "
            "and no tree can hold it."
        )
    if levels.note:
        notes.append(levels.note)
    if not flow_samples:
        notes.append(
            "No null model could be built: §19.1's directed swap rotates three arcs, so a "
            "network with fewer than three of them, or one whose degree sequence admits no "
            "rotation, has no degree-preserving rewiring at all. The four scores above are "
            "descriptions of this network and not comparisons."
        )
    return HierarchyReport(
        frame=str(directed.graph.get("frame", "")),
        nodes=directed.number_of_nodes(),
        edges=directed.number_of_edges(),
        kind=hierarchy_type(directed),
        cycles=cycles,
        reach=reach,
        tree=tree,
        levels=levels,
        layout=layout,
        samples=len(flow_samples),
        seed=seed,
        flow_null=significance(cycles.flow_hierarchy, flow_samples, null=null, tail="right"),
        grc_null=significance(reach.grc, grc_samples, null=null, tail="right"),
        agony_null=significance(levels.total, agony_samples, null=null, tail="left"),
        arborescence_null=significance(tree.score, tree_samples, null=null, tail="right"),
        notes=tuple(notes),
    )


# ------------------------------------------------------------------------- rendering


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


def _names(nodes: Sequence[Node], limit: int = TOP_EDGES) -> str:
    if not nodes:
        return "none"
    shown = ", ".join(str(node) for node in nodes[:limit])
    return shown + (f", +{len(nodes) - limit:,} more" if len(nodes) > limit else "")


def render_hierarchy(report: HierarchyReport) -> list[str]:
    """The ``Hierarchy`` section as markdown lines.

    Prints the four things a reader needs beside every number -- the sampling frame, the n, the
    null model and the chapter each subsection implements -- and then the four scores, which are
    printed together because the chapter's whole argument is that each one is wrong where the
    next one is right.
    """
    lines: list[str] = [
        "## Hierarchy",
        "",
        "**Chapter.** Atlas ch. 33 — §33.1 types of hierarchy, §33.2 cycles, §33.3 global reach "
        "centrality, §33.4 arborescences, §33.5 agony, §33.6 drawing hierarchies.",
        "",
        "**Sampling frame.** "
        + (f"{report.frame} " if report.frame else "")
        + "Every number below is about the direction of the arcs, so it describes what the "
        "corpus stated about which entity, not what is true of either.",
        "",
        f"**n.** {report.nodes:,} nodes and {report.edges:,} arcs, directed.",
        "",
        "**Null model.** "
        + (
            f"The configuration model: {report.samples:,} directed rewirings (§19.1's three-arc "
            "swap, which holds every node's in-degree and out-degree fixed), with the seed "
            f"{report.seed if report.seed is not None else 'unset'}."
            if report.samples
            else "None could be built — see the notes at the end of this section."
        ),
        "",
        "**Implements.** §33.1 (type), §33.2 (cycles), §33.3 (global reach centrality), §33.4 "
        "(arborescence), §33.5 (agony), §33.6 (layers).",
        "",
    ]
    lines += _render_type(report)
    lines += _render_scores(report)
    lines += _render_tree(report.tree)
    lines += _render_agony(report.levels)
    lines += _render_layout(report.layout)
    lines += _render_nulls(report)
    if report.notes:
        lines += ["### Notes on this section", ""]
        lines += [f"- {note}" for note in report.notes]
        lines += [""]
    return lines


def _render_type(report: HierarchyReport) -> list[str]:
    """§33.1's classification, with the checks under it."""
    kind = report.kind
    out = [f"### Type: {kind.name} (§33.1)", "", kind.sentence, ""]
    out += [f"- {reason}" for reason in kind.reasons]
    out += [
        "",
        "§33.1 sorts hierarchies into three kinds and sends two of them elsewhere: an *order* "
        "hierarchy is a centrality ranking (ch. 14 — the Centrality section of this report), a "
        "*nested* hierarchy is hierarchical community discovery (ch. 37, not built). This "
        "section is the third, the **flow** hierarchy.",
        "",
    ]
    return out


def _render_scores(report: HierarchyReport) -> list[str]:
    """The four hierarchicalness scores in one table, with what each one is blind to."""
    cycles, reach, tree, levels = report.cycles, report.reach, report.tree, report.levels
    out = ["### How hierarchical, by four measures that disagree (§33.2-33.5)", ""]
    out += _table(
        ["measure", "value", "perfect score means", "what it is blind to"],
        [
            [
                "flow hierarchy (§33.2)",
                _num(cycles.flow_hierarchy),
                "1: no edge lies on a cycle",
                "too lenient: every DAG scores 1, however odd its shape",
            ],
            [
                "global reach centrality (§33.3)",
                _num(reach.grc),
                "1: one node reaches all, nobody else reaches anything",
                "too strict: only a star scores 1; a tree of depth 3 scores 0.898",
            ],
            [
                "arborescence score (§33.4)",
                _num(tree.score),
                "1: every node already had exactly one boss",
                "punitive: it counts edges removed, not how wrong they were",
            ],
            [
                "agony (§33.5)",
                _num(levels.total),
                "0: every arc runs from a level to a lower one",
                "like §33.2, it scores every DAG perfectly — but it grades the violations",
            ],
        ],
    )
    out += [
        f"**Cycles (§33.2).** {cycles.on_cycles:,} of {cycles.edges:,} arcs "
        f"({cycles.share:.1%}) lie on a cycle, which is to say inside a strongly connected "
        f"component; {cycles.knots:,} such component(s) hold more than one node, the "
        f"largest holding {cycles.largest:,}. Flow hierarchy is one minus that share, which is "
        "the book's ratio of condensed weight to original edges with the bookkeeping removed.",
        "",
        f"**Reach (§33.3).** The furthest-seeing node reaches {reach.maximum:.1%} of the "
        f"network against a mean of {reach.mean:.1%}; {reach.head_count:,} node(s) tie for that "
        f"maximum ({_names(reach.heads)}). GRC averages the gap between the maximum and every "
        "node, so it is one number about one root: a network with two equal heads over two "
        "halves scores it as if each head were half-blind.",
        "",
    ]
    return out


def _render_tree(tree: Arborescence) -> list[str]:
    """§33.4: the tree, its root, and what it threw away."""
    out = ["### One boss per node (§33.4)", ""]
    if tree.refusal:
        out += [f"> {tree.refusal}", ""]
    out += [
        f"The maximum spanning {'arborescence' if tree.spanning else 'branching'} keeps "
        f"{tree.kept:,} of {tree.edges:,} arcs (score {_num(tree.score)}) and "
        f"{tree.weight_share:.1%} of the weight. "
        + (
            f"Its root is {_names(tree.roots, 1)}."
            if len(tree.roots) == 1
            else f"It has {len(tree.roots):,} roots: {_names(tree.roots)}."
        ),
        "",
        "Chu-Liu/Edmonds picks, for every node, the single incoming arc that keeps the most "
        "weight overall — here, the relation the most passages state. §33.4's own procedure "
        "condenses the cycles away first and breaks ties by closeness centrality; this keeps "
        "at least as many arcs, over named nodes rather than condensed blobs.",
        "",
    ]
    if tree.dropped:
        out += ["What one boss per node throws away, heaviest first:", ""]
        out += _table(
            ["from", "to", "weight"],
            [[str(u), str(v), _num(weight)] for u, v, weight in tree.dropped],
        )
        out += [
            f"{tree.dropped_edges:,} arc(s) in total. Every one of them is a relation the corpus "
            "stated: the tree is a reading of the network, not a correction of it.",
            "",
        ]
    return out


def _render_agony(levels: Agony) -> list[str]:
    """§33.5: the levels, the total, and the arcs that pointed up."""
    out = [
        "### Agony: the arrows that point up (§33.5)",
        "",
        f"Total agony {_num(levels.total)} over {levels.edges:,} arc(s) "
        f"({_num(levels.per_edge)} per arc), on {levels.levels:,} levels. "
        f"{levels.backward_edges:,} arc(s) are charged anything at all; the rest run downward "
        "and are free.",
        "",
        f"**Method.** {levels.method}."
        + (
            ""
            if levels.exact
            else " The number above is therefore an upper bound on the agony of this network."
        )
        + (
            f" The program's own bound is {_num(levels.lower_bound)}."
            if not math.isnan(levels.lower_bound)
            else ""
        ),
        "",
        "An arc from level `a` to level `b` costs `max(a - b + 1, 0)`: free downward, 1 between "
        "two nodes on the same level, and `k + 1` for one that climbs `k` levels. The `+1` is "
        "why the measure cannot be gamed by putting everybody on one level. Many rankings reach "
        "the same minimum, so a node's level is not a finding about that node; the total is.",
        "",
        "*The book's own printed formula (p. 470) writes this sum with its levels the other way "
        "round, which contradicts its own surrounding sentence and its own worked example; the "
        "cost above follows that sentence and example instead. See `agony()`'s docstring for "
        "the full argument.*",
        "",
    ]
    if levels.backward:
        out += ["The arcs that cost the most:", ""]
        out += _table(
            ["from", "to", "agony"],
            [[str(u), str(v), f"{cost:,}"] for u, v, cost in levels.backward],
        )
    return out


def _render_layout(layout: Layout) -> list[str]:
    """§33.6: the layering, for a drawing that is not made here."""
    return [
        "### Layers, for a drawing (§33.6)",
        "",
        f"{layout.depth:,} layer(s), the widest holding {layout.widest:,} node(s), with "
        f"{len(layout.dummies):,} waypoint(s) on arcs that cross more than one layer. "
        f"Layering: {layout.layering}.",
        "",
        layout.note,
        "",
        "Coordinates only — nothing here draws anything. `x` is the position within the layer "
        "after a barycentre ordering, `y` is minus the layer, so the roots sit at the top.",
        "",
    ]


def _render_nulls(report: HierarchyReport) -> list[str]:
    """The four scores against the configuration model (ch. 19, §33.8 exercises 1 and 4)."""
    rows = [
        ("flow hierarchy (§33.2)", report.flow_null, "above"),
        ("global reach centrality (§33.3)", report.grc_null, "above"),
        ("arborescence score (§33.4)", report.arborescence_null, "above"),
        ("agony (§33.5)", report.agony_null, "below"),
    ]
    testable = [(name, test, side) for name, test, side in rows if test is not None]
    if not testable or not report.samples:
        return [
            "### Against the configuration model (§33.8)",
            "",
            "Not tested: no degree-preserving rewiring of this network could be built, so there "
            "is nothing to compare the four scores with. They describe this network and make no "
            "claim that its shape is more hierarchical than its degrees force it to be.",
            "",
        ]
    out = [
        "### Against the configuration model (§33.8)",
        "",
        f"Each score against {report.samples:,} rewirings that hold every in- and out-degree "
        "fixed. The question for the first three is whether the observation sits *above* the "
        "null, and for agony whether it sits *below* it — the p-values are one-tailed in that "
        "direction.",
        "",
    ]
    out += _table(
        ["measure", "observed", "null mean", "null sd", "z", "p", "hierarchical if"],
        [
            [
                name,
                _num(test.observed),
                _num(test.null_mean),
                _num(test.null_std),
                _num(test.z, 2),
                f"{test.p_value:.3f}",
                side,
            ]
            for name, test, side in testable
        ],
    )
    caveats = sorted({test.caveat for _, test, _ in testable if test.caveat})
    out += [f"> {caveat}" for caveat in caveats]
    out += [
        "",
        "A network can be more hierarchical than chance and still be a poor hierarchy: the "
        "z-score says the degrees do not explain the shape, not that the shape is an "
        "organisation chart. And a z of 0 here can mean the null had no room rather than that "
        "the network is ordinary: rewiring holds every in-degree exactly, the arborescence "
        "score keeps one arc per node that has any boss at all, and on a network where no node "
        "has two bosses every rewiring is another arborescence forest — so the last three rows "
        "can be frozen by construction while the first still moves.",
        "",
    ]
    return out


def hierarchy_payload(report: HierarchyReport) -> dict[str, Any]:
    """The same section as plain JSON-able data."""
    payload: dict[str, Any] = {
        "frame": report.frame,
        "nodes": report.nodes,
        "edges": report.edges,
        "samples": report.samples,
        "seed": report.seed,
        "type": {
            "name": report.kind.name,
            "reasons": list(report.kind.reasons),
            "acyclic": report.kind.acyclic,
            "roots": report.kind.roots,
            "many_bosses": report.kind.many_bosses,
            "reciprocity": report.kind.reciprocity,
            "weak_components": report.kind.weak_components,
            "sentence": report.kind.sentence,
        },
        "cycles": {
            "edges": report.cycles.edges,
            "on_cycles": report.cycles.on_cycles,
            "share": report.cycles.share,
            "flow_hierarchy": report.cycles.flow_hierarchy,
            "knots": report.cycles.knots,
            "largest": report.cycles.largest,
            "self_loops": report.cycles.self_loops,
        },
        "global_reach": {
            "grc": report.reach.grc,
            "maximum": report.reach.maximum,
            "mean": report.reach.mean,
            "heads": [str(node) for node in report.reach.heads],
            "head_count": report.reach.head_count,
        },
        "arborescence": {
            "score": report.tree.score,
            "kept": report.tree.kept,
            "edges": report.tree.edges,
            "weight_share": report.tree.weight_share,
            "spanning": report.tree.spanning,
            "refusal": report.tree.refusal,
            "roots": [str(node) for node in report.tree.roots],
            "dropped": [
                {"source": str(u), "target": str(v), "weight": weight}
                for u, v, weight in report.tree.dropped
            ],
        },
        "agony": {
            "total": report.levels.total,
            "per_edge": report.levels.per_edge,
            "levels": report.levels.levels,
            "exact": report.levels.exact,
            "lower_bound": report.levels.lower_bound,
            "method": report.levels.method,
            "backward_edges": report.levels.backward_edges,
            "backward": [
                {"source": str(u), "target": str(v), "agony": cost}
                for u, v, cost in report.levels.backward
            ],
            "ranks": {str(node): level for node, level in report.levels.ranks.items()},
        },
        "layout": {
            "layering": report.layout.layering,
            "depth": report.layout.depth,
            "widest": report.layout.widest,
            "layers": [[str(node) for node in row] for row in report.layout.layers],
            "positions": {str(node): list(xy) for node, xy in report.layout.positions.items()},
            "dummies": [
                {
                    "source": str(dummy.source),
                    "target": str(dummy.target),
                    "layer": dummy.layer,
                    "x": dummy.x,
                    "y": dummy.y,
                }
                for dummy in report.layout.dummies
            ],
            "upward_edges": report.layout.upward_edges,
        },
        "notes": list(report.notes),
    }
    payload["null_model"] = {
        name: {
            "observed": test.observed,
            "null": test.null,
            "samples": test.samples,
            "null_mean": test.null_mean,
            "null_std": test.null_std,
            "z": test.z,
            "p_value": test.p_value,
            "tail": test.tail,
            "caveat": test.caveat,
        }
        for name, test in (
            ("flow_hierarchy", report.flow_null),
            ("global_reach_centrality", report.grc_null),
            ("agony", report.agony_null),
            ("arborescence", report.arborescence_null),
        )
        if test is not None
    }
    return payload
