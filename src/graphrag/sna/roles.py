"""Node roles: what a node *is* in a network, not how much of it there is (Atlas ch. 15).

Chapter 14 ranked nodes. This chapter refuses to: *"Sometimes you cannot put a number to what
you're trying to describe. What the person is doing in the social network does not have a
quantity, but a quality: she is playing a specific role"* (p. 225). The role is not a property of
the person either -- *"rather than being a characteristic of the person by itself, the node role
is determined by her position in the network"* -- which is why everything here is computed from
the topology and nothing here reads a node attribute.

**§15.1, the four roles, and what this module adds.** The chapter names four roles in prose and
draws them (pp. 226-227): a **broker** sits between two social circles and belongs to neither, a
**gatekeeper** belongs to one and manages its traffic with the outside, a **core** member has many
ties and all of them inside, a **periphery** member has few ties and all of them inside. It gives
those four no formula in the body; the operational definitions are in its own exercises (§15.5,
p. 235), which separate brokers from gatekeepers by whether a high-betweenness node's ties are
spread *equally* over the communities or prefer its own, and core from periphery by degree among
the nodes whose ties stay at home. Both readings need the same two numbers per node: how much of
a node's connection is inside its own module, and how evenly the rest is spread over the others.
This module computes those two -- :func:`within_module_degree` and :func:`participation`, the
latter being the coefficient :func:`graphrag.sna.measures.brokers` already ranks by -- and cuts
the plane they span with the seven regions of Guimerà and Amaral's functional cartography, a
paper the Atlas cites in its own bibliography (ch. 36, n. 13: *Roger Guimera and Luis A Nunes
Amaral. Functional cartography of complex metabolic networks. Nature, 433(7028):895, 2005*) but
whose thresholds it never prints. Those thresholds are therefore **this module's addition**, not
the book's, and :data:`GA_THRESHOLDS` is a parameter rather than a constant for exactly that
reason. :data:`ATLAS_ROLE` maps each of the seven back onto the four words the chapter uses, so
a report can answer the chapter's question in the chapter's vocabulary.

**§15.2, node similarity.** Structural equivalence is the strict test: *"For two nodes to be
structurally equivalent they have to be connected to the same neighbors"* (p. 229), measured on
the **rows of the adjacency matrix** -- *"If you sort the nodes consistently, each node can be
represented as a vector of zeros and ones"* (pp. 229-230). :func:`structural_similarity` is that,
in the three forms the book names: Jaccard on the neighbour sets, cosine, and the Pearson
correlation of the rows. The chapter's own worked example is the test: its Figure 15.6 pair has
*"three common neighbors out of four possible, thus their structural equivalence is 0.75"*, and
*"the Pearson correlation coefficient of nodes 1 and 2 in Figure 15.6 is around 0.7"*.

Regular equivalence relaxes it -- *"nodes can be equivalent to each other if they have
connections to equivalent nodes"* (p. 231) -- and the book gives the recursion for it,
``sigma = alpha A sigma A + I`` (p. 232), which :func:`regular_similarity` runs. :func:`simrank`
runs the sibling recursion the chapter cites here and spells out later, at p. 338:
``score(u, u) = 1`` and
``score(u, v) = gamma Σ_{a∈Nu} Σ_{b∈Nv} score(a, b) / (k_u k_v)``. Clustering either matrix into
positions and reporting the density of edges between them -- :func:`blockmodel`,
:func:`regular_equivalence` -- is the standard blockmodelling step, which the chapter describes
in words (its three classes of Figure 15.8) without naming an algorithm; the algorithm used here
is stated in the report and in the docstring rather than left implied.

**§15.3, node embeddings, is not here.** The section is a pointer forward -- *"the most common way
to discover such roles has become the use of graph neural networks [...] I will explain more in
detail how they work much later in the book"* (p. 233) -- and this repository honours the pointer
in the same direction: the embedding-based route to roles belongs to ATL-42/43 (shallow graph
learning, random-walk embeddings) and the message-passing one to ATL-44. Nothing in this module
learns anything.

**What a role is not.** Two nodes with the same role need not be connected, need not be near each
other, and need not be alike in any way a corpus would recognise. The chapter is explicit that
structural equivalence is about *neighbours* -- its Figure 15.5(a) pair share every neighbour and
no edge with each other -- and :func:`same_role_never_cooccur` is that observation turned into a
report section: the entities this corpus talks about in the same way without ever talking about
them together.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import networkx as nx
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from graphrag.sna.matrices import Node, adjacency, eigenpairs, node_order
from graphrag.sna.measures import brokers, undirected_view

#: The similarity matrices :func:`similarity` can build, in the order the report offers them.
#: The first three are §15.2's structural equivalence on adjacency rows; ``simrank`` and
#: ``regular`` are its two recursive relaxations.
SIMILARITIES: tuple[str, ...] = ("jaccard", "cosine", "pearson", "simrank", "regular")

#: How many pairs a report lists. Deliberately the same number as
#: :data:`graphrag.sna.analysis.TOP_N`, repeated rather than imported because ``analysis``
#: imports this module and the dependency may not run the other way.
TOP_PAIRS: int = 20

#: Above this many nodes the similarity matrices are refused rather than built. Every one of them
#: is ``n x n`` dense and :func:`simrank` multiplies three of those per iteration, so 2,000 nodes
#: is a 32 MB matrix and 20,000 would be 3.2 GB. The same guard, and the same number, as
#: :data:`graphrag.sna.walks.MAX_NODES`, which refuses a dense ``O(n^3)`` inverse for the same
#: reason. Narrow the network with a filter or a backbone first.
MAX_SIMILARITY_NODES: int = 2_000

#: The default decay of :func:`simrank`. The book calls it *"a parameter you can tune"* (p. 338)
#: and gives it no value, so it is a choice this package makes and every report prints.
SIMRANK_DECAY: float = 0.8


@dataclass(frozen=True)
class RoleThresholds:
    """Where the ``(P, z)`` plane is cut into the seven regions, and by whom.

    Not the Atlas's numbers: the chapter defines its four roles in prose and leaves the cut
    points to the paper it cites (Guimerà and Amaral 2005, in the Atlas's bibliography at ch. 36
    n. 13). They are a dataclass rather than module constants so a report can say which cut
    points produced its counts and a caller can move them -- on a network whose modules are all
    of one size, a hub threshold of 2.5 standard deviations may name nobody at all.
    """

    hub: float = 2.5
    """``z`` at or above which a node is a hub of its own module."""
    ultra_peripheral: float = 0.05
    """Non-hubs at or below this ``P`` are R1: essentially all their ties are at home."""
    peripheral: float = 0.62
    """Non-hubs at or below this ``P`` are R2, above it R3."""
    connector: float = 0.80
    """Non-hubs above this ``P`` are R4, kinless: no module holds a majority of their ties."""
    provincial: float = 0.30
    """Hubs at or below this ``P`` are R5, provincial: a hub of one module only."""
    hub_connector: float = 0.75
    """Hubs at or below this ``P`` are R6, above it R7, kinless."""


#: The cut points every report here uses unless it is handed others.
GA_THRESHOLDS = RoleThresholds()

#: The seven regions, in order, with the name each is known by.
ROLE_NAMES: dict[str, str] = {
    "R1": "ultra-peripheral",
    "R2": "peripheral",
    "R3": "non-hub connector",
    "R4": "non-hub kinless",
    "R5": "provincial hub",
    "R6": "connector hub",
    "R7": "kinless hub",
}

#: Each region translated back into the four words chapter 15 actually uses (pp. 226-227). The
#: mapping is this module's reading, not the book's, and it is lossy in three places. The
#: chapter's *broker* is a node "not part of either community", which R3 and R4 describe by
#: their ties without being able to confirm the belonging, and its *gatekeeper* is a community
#: member "managing how the community relates to society", which is what R6 is. And the
#: chapter's *core* / *periphery* split is a matter of degree -- a core member "has many
#: connections", a peripheral one "does not have many" -- which here is drawn at the hub
#: threshold and nowhere else: *core* is the hubs whose ties stay at home (R5), so *periphery*
#: has to cover **every** non-hub whose ties stay at home, from a node with one tie to one that
#: fell just short of being a hub. Read the ``z`` beside the word rather than the word alone.
#:
#: One word in this table is a false friend, and §15.1 says so where it names the four (p. 226):
#: "core, and periphery (the latter two not to be confused with the core-periphery mesoscale
#: structure that we will see in Chapter 32)". A *core* here is a hub of one module of one
#: partition, so a network with six communities has six of them; chapter 32's core is a single
#: set defined without any partition at all, and the two can disagree completely.
ATLAS_ROLE: dict[str, str] = {
    "R1": "periphery",
    "R2": "periphery",
    "R3": "broker",
    "R4": "broker",
    "R5": "core",
    "R6": "gatekeeper",
    "R7": "broker",
}

#: What each region means, in the report's words.
ROLE_MEANING: dict[str, str] = {
    "R1": (
        "every tie it has is inside its own module, or so nearly every one that P rounds to "
        "nothing. It says nothing about how many ties that is: read z beside it."
    ),
    "R2": "most of its ties are at home; it is an ordinary member of one module.",
    "R3": "not a hub, but its ties are spread over several modules: a broker.",
    "R4": "not a hub, and no module holds even a plurality of its ties.",
    "R5": "a hub of its own module and of nothing else: the core of one circle.",
    "R6": "a hub that also reaches other modules: the gatekeeper of its circle.",
    "R7": "a hub whose ties are spread over so many modules that it has no home.",
}


@dataclass(frozen=True)
class NodeRole:
    """One node's position in the ``(P, z)`` plane, and the role that position falls in."""

    node: Node
    module: int
    """Index of the community the partition put it in, or ``-1`` if the partition missed it."""
    within_module_degree: float
    """``z``: how far this node's inside-the-module connection sits from its module's mean, in
    that module's standard deviations. 0.0 when the module has no spread to measure."""
    participation: float
    """``P``: how evenly the node's ties are spread over the modules. 0 when they all land in
    one, at most ``1 - 1/m`` when they are spread perfectly over ``m`` of them."""
    role: str
    """``R1``-``R7``."""

    @property
    def role_name(self) -> str:
        return ROLE_NAMES[self.role]

    @property
    def atlas_role(self) -> str:
        """The same node in chapter 15's own vocabulary: broker, gatekeeper, core, periphery."""
        return ATLAS_ROLE[self.role]

    @property
    def is_hub(self) -> bool:
        return self.role in {"R5", "R6", "R7"}


@dataclass(frozen=True)
class RoleTable:
    """Every node's role, with what the roles were computed against."""

    rows: list[NodeRole]
    modules: int
    """How many communities the partition had. The report needs it because it caps ``P``."""
    thresholds: RoleThresholds
    note: str = ""
    """What a directed network cost this table, or "" when there was nothing to say."""

    @property
    def ceiling(self) -> float:
        """The largest ``P`` this partition can produce: ``1 - 1/m`` (see :func:`participation`).

        Worth printing next to the counts: with two communities it is 0.5, which is below the
        0.62 that separates R2 from R3, so on a two-community network *no node can be classed a
        connector however evenly its ties are spread*. That is a property of the partition, not
        a finding about the network.
        """
        return participation_ceiling(self.modules)

    def counts(self) -> dict[str, int]:
        """How many nodes fell in each region, R1 through R7, zeros included."""
        counts = dict.fromkeys(ROLE_NAMES, 0)
        for row in self.rows:
            counts[row.role] += 1
        return counts

    def by_role(self, role: str) -> list[NodeRole]:
        """The nodes in one region, the hub-most first."""
        chosen = [row for row in self.rows if row.role == role]
        return sorted(chosen, key=lambda r: (-r.within_module_degree, str(r.node)))

    def hubs(self) -> list[NodeRole]:
        """Every node above the hub threshold, highest ``z`` first."""
        chosen = [row for row in self.rows if row.is_hub]
        return sorted(chosen, key=lambda r: (-r.within_module_degree, str(r.node)))

    def role_of(self, node: Node) -> str:
        return self._index[node].role

    @property
    def _index(self) -> dict[Node, NodeRole]:
        return {row.node: row for row in self.rows}


@dataclass(frozen=True)
class Similarity:
    """One node-similarity matrix, with the node order it follows and how it was made.

    The rest of this package returns ``(matrix, nodes)`` pairs (the node-order contract of
    :mod:`graphrag.sna.matrices`). This one carries a record instead because §15.2's measures
    are not interchangeable and two of them have parameters: a SimRank matrix without its decay
    beside it, or a regular-equivalence matrix without its ``alpha``, is a number a reader cannot
    reproduce. ``nodes`` keeps the same contract, so ``matrix[i][j]`` is the similarity of
    ``nodes[i]`` and ``nodes[j]``.
    """

    matrix: np.ndarray
    nodes: list[Node]
    method: str
    section: str
    """The section of the Atlas this measure implements."""
    meaning: str
    """What a high value means, for the caption above the table."""
    parameters: dict[str, float] = field(default_factory=dict)
    """Every choice that changes the numbers -- the decay, the alpha -- for the report to
    print."""

    def between(self, u: Node, v: Node) -> float:
        """The similarity of two nodes. Raises ``KeyError`` for a node the matrix does not hold."""
        index = self._index
        return float(self.matrix[index[u], index[v]])

    def top_pairs(
        self, n: int | None = TOP_PAIRS, *, graph: nx.Graph | None = None, adjacent: bool = True
    ) -> list[tuple[Node, Node, float]]:
        """The most similar distinct pairs, highest first, ties broken by node id.

        ``adjacent=False`` with a ``graph`` drops the pairs that are joined by an edge, which is
        the §15.2 reading: two nodes with the same neighbours need not be neighbours, and on a
        co-occurrence network the pairs that are *not* joined are the interesting ones.
        ``n=None`` returns every pair, for a caller that filters before it truncates.
        """
        pairs: list[tuple[Node, Node, float]] = []
        for i, u in enumerate(self.nodes):
            for j in range(i + 1, len(self.nodes)):
                v = self.nodes[j]
                if not adjacent and graph is not None and graph.has_edge(u, v):
                    continue
                pairs.append((u, v, float(self.matrix[i, j])))
        pairs.sort(key=lambda p: (-p[2], str(p[0]), str(p[1])))
        return pairs if n is None else pairs[:n]

    @property
    def _index(self) -> dict[Node, int]:
        return {node: i for i, node in enumerate(self.nodes)}


@dataclass(frozen=True)
class Blockmodel:
    """Positions (blocks of equivalent nodes) and the density of edges between them.

    ``image[i][j]`` is the share of the possible edges between position ``i`` and position ``j``
    that the network actually has -- the *image matrix* of classical blockmodelling. On the
    diagonal it is the density inside a position. A position is a cluster of a similarity
    matrix, so it inherits that matrix's instability: run it at two ``k`` before quoting one.
    """

    positions: list[list[Node]]
    image: np.ndarray
    method: str
    """The similarity the positions were clustered from, and the clustering that did it."""
    k: int

    @property
    def sizes(self) -> list[int]:
        return [len(position) for position in self.positions]


@dataclass(frozen=True)
class RolePair:
    """Two nodes that play the same role and share no edge (§15.2's own observation)."""

    u: Node
    v: Node
    role: str
    similarity: float
    same_module: bool
    """Whether the partition put both in the same community. A same-role pair in two different
    communities is the interesting one: the corpus treats them alike without ever joining them."""


@dataclass
class RolesReport:
    """Everything one ``graphrag sna roles`` run produced."""

    persona_id: str
    network: str
    graph: nx.Graph
    table: RoleTable
    similarity: Similarity | None
    pairs: list[RolePair]
    blockmodel: Blockmodel | None
    partition_source: str
    """Where the modules came from, in the report's words -- a role is relative to them."""
    minimum: float = 0.5
    """The similarity floor the same-role pairs were listed above. A reporting threshold with no
    null model behind it, printed beside the pairs so a reader knows what was withheld."""
    generated_at: str = ""
    notes: list[str] = field(default_factory=list)

    def label(self, node: Node) -> str:
        data = self.graph.nodes.get(node, {})
        text = data.get("label") or data.get("name") or node
        return str(text)


# --------------------------------------------------------------- §15.1 the two coordinates


def participation_ceiling(modules: int) -> float:
    """The largest participation coefficient a partition into ``modules`` modules can produce.

    ``P = 1 - Σ_s (k_is / k_i)^2`` is minimised at 1 when every tie lands in one module, so it is
    maximised when the ties are spread perfectly evenly: ``1 - m (1/m)^2 = 1 - 1/m``. With two
    communities that is 0.5, which sits **below** the 0.62 that separates an ordinary node from a
    connector, so a two-community network has no connectors by construction. Reports print this
    next to the role counts rather than let a reader take a ceiling for a finding.

    0.0 for fewer than two modules, where every tie is at home by definition.
    """
    return 0.0 if modules < 2 else 1.0 - 1.0 / modules


def _membership(communities: Sequence[Iterable[Node]]) -> dict[Node, int]:
    return {node: index for index, community in enumerate(communities) for node in community}


def within_module_degree(
    graph: nx.Graph, communities: Sequence[Iterable[Node]]
) -> dict[Node, float]:
    """``z``: how well connected a node is *inside its own module*, in that module's own units.

    ``z_i = (k_i - mean(k_s)) / sd(k_s)``, where ``k_i`` is the connection node ``i`` has to
    other members of its own module ``s`` and the mean and standard deviation are taken over all
    the members of ``s``. It is the vertical axis of the plane §15.1's roles are read off: the
    chapter's **core** member is *"very embedded"* in her circle and its **periphery** member
    *"does not have many connections in the community"* (p. 227), and this is that distinction
    with the module's own size and density divided out, so a small dense circle and a large
    sparse one can be read on one axis.

    What the book defines and what this adds: the chapter gives the distinction in words and its
    exercise (§15.5) operationalises it as plain degree among the nodes whose ties stay at home.
    Standardising per module is Guimerà and Amaral's step, not the Atlas's, and it is the reason
    a hub of a six-node community and a hub of a six-hundred-node one can appear in one table.

    Connection is measured in **edge weight**, which on an unweighted graph is a count of edges
    -- the same currency :func:`participation` uses, because a node placed in the plane by a
    weighted P and an unweighted z is in neither of the two places it belongs. Hand in a graph
    with no weights (``tests.legendary.unweighted``) for the book's plain counts.

    **Undefined**, and returned as 0.0, when a module has no spread to measure: every member with
    the same internal degree, which includes every module of one node. 0.0 is "exactly average",
    which is what such a node is, but it also means no member of a uniform module can ever be
    called a hub -- say so rather than reporting an empty hub list as a finding.

    A directed network is flattened first (§6.2): a tie inside a module is a tie whichever way it
    was written down. :func:`guimera_amaral` carries the sentence that says so.
    """
    graph, _ = undirected_view(graph)
    membership = _membership(communities)
    internal: dict[Node, float] = {}
    for node in graph.nodes:
        mine = membership.get(node, -1)
        internal[node] = sum(
            float(graph[node][other].get("weight", 1.0))
            for other in graph.neighbors(node)
            if membership.get(other, -1) == mine
        )
    per_module: dict[int, list[float]] = {}
    for node, value in internal.items():
        per_module.setdefault(membership.get(node, -1), []).append(value)
    statistics = {
        module: (float(np.mean(values)), float(np.std(values)))
        for module, values in per_module.items()
    }
    scores: dict[Node, float] = {}
    for node, value in internal.items():
        mean, deviation = statistics[membership.get(node, -1)]
        scores[node] = (value - mean) / deviation if deviation > 0 else 0.0
    return scores


def participation(graph: nx.Graph, communities: Sequence[Iterable[Node]]) -> dict[Node, float]:
    """``P``: how evenly a node's ties are spread over the modules -- the horizontal axis.

    ``P_i = 1 - Σ_s (k_is / k_i)^2``: 0 when every neighbour sits in one module, rising towards
    :func:`participation_ceiling` as the ties spread evenly over many. It is what separates
    §15.1's **broker** -- *"people who are not part of either community [...] but they still have
    friends in both"* (p. 226) -- from its core and periphery members, whose ties stay at home,
    and it is the number :func:`graphrag.sna.measures.brokers` already ranks a report by.

    This is that same function, asked for every node instead of the top twenty, so the two halves
    of a report cannot disagree about what a broker score is. A node with no edges at all scores
    0.0 there and here: it has no ties to spread.
    """
    return dict(brokers(graph, communities, graph.number_of_nodes()))


def classify(participation_score: float, z: float, thresholds: RoleThresholds) -> str:
    """Which of the seven regions a ``(P, z)`` point falls in. See :data:`GA_THRESHOLDS`."""
    if z < thresholds.hub:
        if participation_score <= thresholds.ultra_peripheral:
            return "R1"
        if participation_score <= thresholds.peripheral:
            return "R2"
        return "R3" if participation_score <= thresholds.connector else "R4"
    if participation_score <= thresholds.provincial:
        return "R5"
    return "R6" if participation_score <= thresholds.hub_connector else "R7"


def guimera_amaral(
    graph: nx.Graph,
    communities: Sequence[Iterable[Node]],
    *,
    thresholds: RoleThresholds = GA_THRESHOLDS,
) -> RoleTable:
    """Every node's role, from its within-module degree and its participation coefficient.

    The plane of §15.1 cut into seven regions: R1 ultra-peripheral, R2 peripheral, R3 non-hub
    connector, R4 non-hub kinless, R5 provincial hub, R6 connector hub, R7 kinless hub.
    :data:`ATLAS_ROLE` translates them back into the chapter's own four words.

    **A role is relative to the partition it was computed on.** Both coordinates are defined
    against the modules: change the resolution, change the seed, change the community method, and
    a connector hub becomes a provincial one without a single edge moving. The partition is part
    of the finding and the report prints where it came from.

    **The cut points are not the Atlas's** -- see the module docstring -- and two of their
    consequences have to be read off the partition rather than the network. A partition into
    ``m`` modules caps ``P`` at ``1 - 1/m`` (:func:`participation_ceiling`), so with two
    communities nobody can be a connector; and a module whose members all have the same internal
    degree has no ``z`` spread, so nobody in it can be a hub.

    A directed network is flattened first (§6.2) and ``note`` says so: a role here is about who a
    node is tied to, never about which way the tie was written down.
    """
    _, flattened = undirected_view(graph)
    z = within_module_degree(graph, communities)
    p = participation(graph, communities)
    membership = _membership(communities)
    rows = [
        NodeRole(
            node=node,
            module=membership.get(node, -1),
            within_module_degree=z[node],
            participation=p[node],
            role=classify(p[node], z[node], thresholds),
        )
        for node in sorted(z, key=str)
    ]
    note = (
        f"Roles were computed {flattened}, so a tie inside a module is a tie in either direction."
        if flattened
        else ""
    )
    return RoleTable(rows=rows, modules=len(list(communities)), thresholds=thresholds, note=note)


# ------------------------------------------------------------------- §15.2 node similarity


def _guard(graph: nx.Graph, nodes: Sequence[Node] | None, max_nodes: int) -> list[Node]:
    """The node order for a similarity matrix, refused when it would be too big to hold."""
    order = node_order(graph, nodes)
    if len(order) > max_nodes:
        msg = (
            f"a similarity matrix over {len(order):,} nodes is {len(order) ** 2:,} cells; the "
            f"guard is max_nodes={max_nodes:,}. Narrow the network (--min-weight, --types, a "
            "backbone) or raise max_nodes knowing the matrix is dense."
        )
        raise ValueError(msg)
    return order


def structural_similarity(
    graph: nx.Graph,
    method: str = "jaccard",
    nodes: Sequence[Node] | None = None,
    *,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> Similarity:
    """Structural equivalence on the rows of the adjacency matrix (§15.2, pp. 229-230).

    *"For two nodes to be structurally equivalent they have to be connected to the same
    neighbors. If they do, they are indistinguishable from one another, therefore they cannot be
    any more similar"* (p. 229). The vectors compared are §15.2's own: *"If you sort the nodes
    consistently, each node can be represented as a vector of zeros and ones [...] These are the
    rows in the adjacency matrix corresponding to the nodes"* (p. 230). Three of the measures the
    section names are offered:

    ``jaccard``
        The Jaccard similarity of the two neighbour sets: ``|N(u) ∩ N(v)|`` over the size of
        their union. The book's worked example is the Figure 15.6 pair, who have *"three common
        neighbors out of four possible, thus their structural equivalence is 0.75"*.
    ``cosine``
        The cosine of the angle between the two rows, ``|N(u) ∩ N(v)| / sqrt(k_u k_v)``.
    ``pearson``
        The Pearson correlation of the two rows, which the book puts at *"around 0.7"* for the
        same pair. Unlike the other two it can be **negative**: two nodes whose neighbourhoods
        avoid each other correlate below zero, which no count of shared neighbours can express.

    **Edge weights are ignored, deliberately.** The book's vector is of zeros and ones, and a
    weighted Jaccard would need a definition the chapter does not give. Structural equivalence
    here therefore asks whether two nodes were written down beside the same others, not how often.

    **The diagonal of the adjacency is zero**, so ``u`` is not its own neighbour and an edge
    between ``u`` and ``v`` does not make them similar -- it makes each a neighbour the other
    lacks. That is the book's own Figure 15.5(a), whose two structurally equivalent nodes share
    every neighbour and no edge with each other. Two isolated nodes score 0.0 rather than 1.0:
    their neighbour sets are equal and empty, and calling that "indistinguishable" would put
    every node nobody wrote about at the top of the table. The self-similarity on the diagonal is
    1.0 by convention, matching :func:`simrank`'s ``score(u, u) = 1``.
    """
    if method not in {"jaccard", "cosine", "pearson"}:
        msg = f"structural_similarity takes jaccard, cosine or pearson, got {method!r}"
        raise ValueError(msg)
    graph, _ = undirected_view(graph)
    order = _guard(graph, nodes, max_nodes)
    matrix, _ = adjacency(graph, order, weight=None)
    rows = np.asarray(matrix, dtype=np.float64)
    shared = rows @ rows.T
    degrees = rows.sum(axis=1)
    if method == "jaccard":
        union = degrees[:, np.newaxis] + degrees[np.newaxis, :] - shared
        scores = np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)
        meaning = "the share of the two neighbour sets' union that both nodes are joined to"
    elif method == "cosine":
        norms = np.sqrt(np.outer(degrees, degrees))
        scores = np.divide(shared, norms, out=np.zeros_like(shared), where=norms > 0)
        meaning = "the cosine of the angle between the two adjacency rows"
    else:
        scores = _correlation(rows)
        meaning = "the correlation of the two adjacency rows; negative means they avoid each other"
    np.fill_diagonal(scores, 1.0)
    return Similarity(
        matrix=scores,
        nodes=order,
        method=method,
        section="§15.2 (structural equivalence)",
        meaning=meaning,
    )


def _correlation(rows: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of rows, with constant rows scored 0.0.

    ``numpy.corrcoef`` divides by a zero standard deviation for a constant row -- an isolated
    node, or one joined to everything -- and returns ``nan``. A correlation with a constant is
    undefined rather than zero, but a table cannot hold a NaN, so 0.0 ("no relationship
    measurable") is reported and this is the one place it is done.
    """
    if rows.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    centred = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    denominator = np.outer(norms, norms)
    scores = np.divide(
        centred @ centred.T, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return np.clip(scores, -1.0, 1.0)


def simrank(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    *,
    decay: float = SIMRANK_DECAY,
    iterations: int = 100,
    tolerance: float = 1e-8,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> Similarity:
    """SimRank: *"two nodes are similar if they are connected to similar neighbors"* (p. 337).

    §15.2 introduces the recursion as the way to quantify regular equivalence and cites Jeh and
    Widom for it; the formula itself is printed later in the book, at p. 338, where the chapter
    on link prediction reaches for the same idea: ``score(u, u) = 1`` and

        ``score(u, v) = gamma Σ_{a∈Nu} Σ_{b∈Nv} score(a, b) / (k_u k_v)``.

    That is what this iterates, in matrix form -- ``S ← gamma W S Wᵀ`` with ``W = D⁻¹A`` the
    row-normalised adjacency, the diagonal reset to 1 after every sweep -- until no entry moves
    by more than ``tolerance`` or ``iterations`` sweeps have run. The matrix form is exactly the
    double sum: ``(W S Wᵀ)[u][v] = Σ_{a,b} A[u][a] A[v][b] S[a][b] / (k_u k_v)``.

    **The decay is a choice, and it changes the answer.** The book calls ``gamma`` *"a parameter you
    can tune"* and never fixes it; ``networkx`` calls the same parameter ``importance_factor``
    and defaults it to 0.9, this package defaults it to 0.8, and neither default is derived from
    anything. The report prints it. What it controls is how far the recursion looks: p. 338
    notes that the expected score is ``gamma^l`` for a walk of average length ``l``, so a smaller
    decay confines the similarity to short walks and a larger one lets distant structure in.

    Nodes with no edges have no neighbours to be similar through and score 0 against everything,
    which is the recursion's own answer rather than a special case.

    Undirected only: a directed network is flattened first (§6.2). SimRank is defined on a
    digraph over *in*-neighbours, which is a different question -- who points at these two -- and
    asking it would need the ``relations`` network read in one direction, which this does not do.
    """
    if not 0.0 < decay < 1.0:
        msg = f"simrank's decay must sit strictly between 0 and 1, got {decay}"
        raise ValueError(msg)
    graph, _ = undirected_view(graph)
    order = _guard(graph, nodes, max_nodes)
    matrix, _ = adjacency(graph, order, weight=None)
    rows = np.asarray(matrix, dtype=np.float64)
    degrees = rows.sum(axis=1)
    transition = np.divide(
        rows, degrees[:, np.newaxis], out=np.zeros_like(rows), where=degrees[:, np.newaxis] > 0
    )
    scores = np.eye(len(order), dtype=np.float64)
    sweeps, moved = 0, 0.0
    for sweeps in range(1, iterations + 1):  # noqa: B007 -- the count is reported
        updated = decay * (transition @ scores @ transition.T)
        np.fill_diagonal(updated, 1.0)
        moved = float(np.abs(updated - scores).max()) if scores.size else 0.0
        scores = updated
        if moved <= tolerance:
            break
    return Similarity(
        matrix=scores,
        nodes=order,
        method="simrank",
        section="§15.2 (regular equivalence), formula at p. 338",
        meaning="how similar the neighbours of the two nodes are, recursively",
        parameters={
            "decay": decay,
            "iterations": float(sweeps),
            "converged": float(moved <= tolerance),
        },
    )


def regular_similarity(
    graph: nx.Graph,
    nodes: Sequence[Node] | None = None,
    *,
    alpha: float | None = None,
    iterations: int = 200,
    tolerance: float = 1e-10,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> Similarity:
    """The book's own regular-equivalence recursion, ``sigma = alpha A sigma A + I`` (§15.2,
    p. 232).

    *"In regular equivalence, nodes can be equivalent to each other if they have connections to
    equivalent nodes [...] similar nodes connect to similar nodes"* (p. 231-232). The book writes
    the recursion twice: ``sigma_uv = alpha Σ_ij A_ui A_vj sigma_ij``, and then in matrix form
    with the identity added to *"boost the similarity of a node with itself"*. This iterates the
    second form from ``sigma = I`` until it stops moving.

    **Where this departs from the text.** The book motivates the decay with *"If alpha < 1, the
    alpha^l < alpha^(l-1) and you get smaller contributions from nodes that are farther away"*,
    but ``alpha < 1`` is not enough for the series to converge: the sum is
    ``Σ_l alpha^l A^2l``, so it needs ``alpha * λ₁² < 1``, where ``λ₁`` is the adjacency's
    leading eigenvalue -- the same condition Katz centrality carries and for the same reason.
    ``alpha=None`` therefore sets it to half that radius, ``0.5 / λ₁²``, which is inside the
    radius by a wide enough margin to settle in a few dozen sweeps; a larger share of the radius
    lets longer walks contribute and takes proportionally longer to converge. The value used, the
    radius, the sweeps taken and whether it converged are all reported as parameters, and an
    ``alpha`` above the radius is refused rather than iterated into overflow.

    The returned matrix is cosine-normalised -- ``sigma_uv / sqrt(sigma_uu sigma_vv)`` -- which
    is not in the book. Two reasons: the raw matrix has an arbitrary scale that depends entirely
    on ``alpha``, so two runs are not comparable; and normalising puts the numbers on [0, 1] with
    1 on the diagonal, which is where the other measures in this module live. The normalisation
    is valid because ``sigma`` is a sum of ``A^2l = (A^l)(A^l)ᵀ`` and so positive semi-definite.

    **The book's own warning applies to the output** (p. 232): *"you should be careful because
    this measure as written here reduces to something similar to structural equivalence. If you
    run this formula on the network in Figure 15.8 you get that nodes 1, 4, 5 and 6 are all
    similar -- in intuitive terms, you'd get that CEOs are similar to interns because they both
    connect to middle managers."* Everything at an even number of hops in a hierarchy collapses
    together, and the chapter offers no fix beyond *"you need to ensure that your initial classes
    are respected somehow"* -- which needs classes this network does not have. Read the positions
    it produces as "connected into the structure in the same way", never as a rank.
    """
    graph, _ = undirected_view(graph)
    order = _guard(graph, nodes, max_nodes)
    matrix, _ = adjacency(graph, order, weight=None)
    rows = np.asarray(matrix, dtype=np.float64)
    radius = float(np.abs(eigenpairs(rows)[0]).max()) if rows.size else 0.0
    limit = 1.0 / radius**2 if radius > 0 else float("inf")
    if alpha is None:
        alpha = 0.5 * limit if radius > 0 else 0.5
    elif alpha <= 0 or alpha >= limit:
        msg = (
            f"sigma = alpha A sigma A + I converges only for alpha < 1/λ₁² = {limit:.6g} on "
            f"this network (§15.2 gives the weaker alpha < 1); got {alpha}"
        )
        raise ValueError(msg)
    scores = np.eye(len(order), dtype=np.float64)
    identity = np.eye(len(order), dtype=np.float64)
    sweeps, moved = 0, 0.0
    for sweeps in range(1, iterations + 1):  # noqa: B007 -- the count is reported
        updated = alpha * (rows @ scores @ rows) + identity
        moved = float(np.abs(updated - scores).max()) if scores.size else 0.0
        scores = updated
        if moved <= tolerance:
            break
    diagonal = np.sqrt(np.clip(np.diag(scores), 0.0, None))
    denominator = np.outer(diagonal, diagonal)
    normalised = np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)
    return Similarity(
        matrix=np.clip(normalised, -1.0, 1.0),
        nodes=order,
        method="regular",
        section="§15.2 (regular equivalence), p. 232",
        meaning=(
            "how alike the two nodes' whole neighbourhoods are, at every even number of hops, "
            "cosine-normalised onto [0, 1] -- a step this package adds, because the book's raw "
            "matrix is on a scale that alpha alone decides"
        ),
        parameters={
            "alpha": float(alpha),
            "convergence_limit": limit,
            "iterations": float(sweeps),
            "converged": float(moved <= tolerance),
        },
    )


def similarity(
    graph: nx.Graph,
    method: str = "jaccard",
    nodes: Sequence[Node] | None = None,
    *,
    decay: float = SIMRANK_DECAY,
    alpha: float | None = None,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> Similarity:
    """One of :data:`SIMILARITIES` by name, so a command can take ``--method`` and pass it on."""
    if method in {"jaccard", "cosine", "pearson"}:
        return structural_similarity(graph, method, nodes, max_nodes=max_nodes)
    if method == "simrank":
        return simrank(graph, nodes, decay=decay, max_nodes=max_nodes)
    if method == "regular":
        return regular_similarity(graph, nodes, alpha=alpha, max_nodes=max_nodes)
    msg = f"similarity must be one of {', '.join(SIMILARITIES)}, got {method!r}"
    raise ValueError(msg)


# ------------------------------------------------------------------ §15.2 blockmodel positions


def image_matrix(graph: nx.Graph, positions: Sequence[Sequence[Node]]) -> np.ndarray:
    """The ``k x k`` density of edges between the positions: the blockmodel's image matrix.

    ``image[i][j]`` is how many of the possible edges between position ``i`` and position ``j``
    the network actually has -- ``edges / (n_i n_j)`` off the diagonal, and
    ``edges / (n_i (n_i - 1) / 2)`` on it, the density of the induced subgraph. A block of one
    node has no inside density and scores 0.0.

    This is the step the chapter describes without naming: its Figure 15.8 has three classes and
    the text reads them off *"the third class is defined by those nodes connected to nodes of
    class two but without connections to class one"* (p. 231), which is a statement about this
    matrix. A row of the image says how a position relates to every position including itself;
    two positions with the same row are regularly equivalent as blocks.

    Edge weights are ignored, as they are throughout §15.2: a density is a share of possible
    edges, and a weight is not a share of anything.
    """
    graph, _ = undirected_view(graph)
    blocks = [list(position) for position in positions]
    membership = _membership(blocks)
    k = len(blocks)
    counts = np.zeros((k, k), dtype=np.float64)
    for u, v in graph.edges():
        i, j = membership.get(u, -1), membership.get(v, -1)
        if i < 0 or j < 0 or u == v:
            continue
        counts[i, j] += 1.0
        if i != j:
            counts[j, i] += 1.0
    image = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            size_i, size_j = len(blocks[i]), len(blocks[j])
            possible = size_i * (size_i - 1) / 2 if i == j else size_i * size_j
            if possible > 0:
                image[i, j] = counts[i, j] / possible
    return image


def blockmodel(graph: nx.Graph, scores: Similarity, k: int) -> Blockmodel:
    """Positions from a similarity matrix, and the density between them.

    **How the positions are found, stated rather than implied.** The similarity is turned into a
    distance (``1 - s``, clipped at 0 for the measures that can go negative) and handed to
    average-linkage agglomerative clustering with that distance precomputed. Agglomerative rather
    than K-means for one reason: it is deterministic, so a position is a function of the network
    and ``k`` alone, with no seed to report. It is still a clustering of a similarity matrix, and
    a *position inherits that matrix's instability*: at ``k`` and ``k+1`` the blocks can be
    entirely different, and nothing here tells you which ``k`` is right.

    The Atlas describes positions (the three classes of its Figure 15.8, p. 231) without giving
    an algorithm for them, and names neither REGE nor CATREGE; this is therefore a standard
    convention laid on top of the book's similarity, not the book's own procedure.

    ``k`` must be between 1 and the number of nodes.
    """
    order = scores.nodes
    if not 1 <= k <= len(order):
        msg = f"k must be between 1 and the {len(order)} nodes being clustered, got {k}"
        raise ValueError(msg)
    if k == 1 or len(order) == 1:
        positions = [list(order)]
    else:
        distance = np.clip(1.0 - scores.matrix, 0.0, None)
        np.fill_diagonal(distance, 0.0)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        labels = model.fit_predict(distance)
        buckets: dict[int, list[Node]] = {}
        for node, label in zip(order, labels, strict=True):
            buckets.setdefault(int(label), []).append(node)
        positions = [sorted(buckets[key], key=str) for key in sorted(buckets)]
    return Blockmodel(
        positions=positions,
        image=image_matrix(graph, positions),
        method=f"{scores.method} similarity, average-linkage agglomerative clustering",
        k=len(positions),
    )


def regular_equivalence(
    graph: nx.Graph,
    k: int,
    *,
    alpha: float | None = None,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> Blockmodel:
    """Positions by regular equivalence: :func:`regular_similarity`, then :func:`blockmodel`.

    The chapter's own route to *"three equivalence classes"* (p. 231): compute how alike the
    nodes' neighbourhoods are with the recursion of p. 232, then group the nodes whose rows agree
    and report the density between the groups. The warning on :func:`regular_similarity` is the
    warning on this: in a hierarchy, everything an even number of hops apart collapses together,
    so a "position" here is a structural class, never a rank.
    """
    return blockmodel(graph, regular_similarity(graph, alpha=alpha, max_nodes=max_nodes), k)


# ---------------------------------------------------------- §15.2 same role, never together


def same_role_never_cooccur(
    graph: nx.Graph,
    table: RoleTable,
    scores: Similarity,
    *,
    minimum: float = 0.5,
    limit: int = TOP_PAIRS,
) -> list[RolePair]:
    """Pairs that play the same role, are at least ``minimum`` similar, and share no edge.

    This is §15.2's structural point turned into a report section. The chapter's Figure 15.5(a)
    pair are *"indistinguishable from one another"* and are not joined to each other: structural
    equivalence is about neighbours, not about being connected. On a co-occurrence network that
    is the useful direction of the observation -- two entities the corpus discusses in the same
    company, in the same structural position, and never in the same passage. Whether that is a
    gap in the corpus or a fact about the subject is not something the network can say.

    Sorted by similarity, highest first, ties broken by node id. ``minimum`` is a reporting
    threshold and nothing more: no null model stands behind it, and it is printed beside the
    pairs so a reader knows what was withheld.
    """
    roles = {row.node: row for row in table.rows}
    pairs: list[RolePair] = []
    for u, v, score in scores.top_pairs(None, graph=graph, adjacent=False):
        if score < minimum:
            break  # the pairs arrive sorted, so the first one below the floor ends the list
        left, right = roles.get(u), roles.get(v)
        if left is None or right is None or left.role != right.role:
            continue
        pairs.append(
            RolePair(
                u=u,
                v=v,
                role=left.role,
                similarity=score,
                same_module=left.module == right.module,
            )
        )
        if len(pairs) == limit:
            break
    return pairs


# ----------------------------------------------------------------------------- the report


#: How many members of a position the report names before it says "+n more". The same number
#: :data:`graphrag.sna.analysis.TOP_MEMBERS` uses, repeated for the same reason as
#: :data:`TOP_PAIRS`.
TOP_MEMBERS: int = 8


def build_roles_report(
    graph: nx.Graph,
    *,
    persona_id: str,
    network: str,
    communities: Sequence[Iterable[Node]] | None = None,
    partition_source: str = "",
    method: str = "jaccard",
    k: int | None = None,
    decay: float = SIMRANK_DECAY,
    alpha: float | None = None,
    minimum: float = 0.5,
    thresholds: RoleThresholds = GA_THRESHOLDS,
    resolution: float = 1.0,
    runs: int = 10,
    seed: int | None = None,
    max_nodes: int = MAX_SIMILARITY_NODES,
) -> RolesReport:
    """One ``sna roles`` run: the role table, one similarity matrix, and what the two agree on.

    ``communities`` is the partition the roles are relative to. Left out, Louvain finds one --
    the same Louvain ``sna analyze`` runs, with the same defaults -- and ``partition_source``
    records which, because §15.1's roles are a statement about a partition first and about the
    network second.

    ``k`` adds the blockmodel: §15.2's positions, clustered out of the same similarity matrix,
    with the density between them. Left out, no positions are claimed.
    """
    notes: list[str] = []
    if communities is None:
        from graphrag.sna.cluster import louvain  # local: cluster must not import this module

        found = louvain(graph, resolution=resolution, seed=seed, runs=runs)
        communities = found.communities
        partition_source = (
            f"Louvain at resolution {found.resolution:g} over {len(found.seeds)} seeds "
            f"(modularity {found.modularity:.4f}, stability ARI {found.stability:.3f})"
        )
        if found.note:
            notes.append(found.note)
        if found.stability < 0.6:
            notes.append(
                "The partition changes with the seed (stability ARI below 0.6) and every role "
                "below is defined against it, so read the role counts rather than any node's "
                "role."
            )
    partition = [list(community) for community in communities]
    table = guimera_amaral(graph, partition, thresholds=thresholds)
    if table.note:
        notes.append(table.note)
    scores: Similarity | None = None
    pairs: list[RolePair] = []
    block: Blockmodel | None = None
    try:
        scores = similarity(graph, method, decay=decay, alpha=alpha, max_nodes=max_nodes)
    except ValueError as exc:
        notes.append(f"No similarity matrix was built: {exc}")
    if scores is not None:
        pairs = same_role_never_cooccur(graph, table, scores, minimum=minimum)
        if k is not None:
            block = blockmodel(graph, scores, k)
    if 0.0 < table.ceiling < thresholds.peripheral:
        notes.append(
            f"With {table.modules} modules the participation coefficient cannot exceed "
            f"{table.ceiling:.2f}, which is below the {thresholds.peripheral:g} that separates an "
            "ordinary node from a connector: no node in this run can be classed a connector, "
            "whatever its ties do. That is the partition talking, not the network."
        )
    return RolesReport(
        persona_id=persona_id,
        network=network,
        graph=graph,
        table=table,
        similarity=scores,
        pairs=pairs,
        blockmodel=block,
        partition_source=partition_source or "a partition supplied by the caller",
        minimum=minimum,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=notes,
    )


def _rows(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _parameters(scores: Similarity) -> str:
    """The parameters of a similarity, for the line under its table. "" when it has none."""
    return ", ".join(f"{name}={value:g}" for name, value in scores.parameters.items())


def _similarity_section(report: RolesReport, scores: Similarity) -> list[str]:
    """The similarity method, what it means, and the pairs it ranks highest."""
    lines = [
        "## Node similarity (§15.2)",
        "",
        f"Method `{scores.method}` -- {scores.section}: {scores.meaning}."
        + (f" Parameters: {_parameters(scores)}." if scores.parameters else "")
        + " Edge weights are ignored throughout §15.2, because the vectors the section compares "
        "are its own rows of zeros and ones: this asks whether two nodes were written down "
        "beside the same others, never how often.",
        "",
        "### The most similar pairs",
        "",
    ]
    lines += _rows(
        ["node", "node", "similarity", "joined?"],
        [
            [
                report.label(u),
                report.label(v),
                f"{score:.3f}",
                "yes" if report.graph.has_edge(u, v) else "no",
            ]
            for u, v, score in scores.top_pairs(TOP_PAIRS)
        ],
    )
    lines += [
        "### Same role, never together",
        "",
        f"Pairs that fall in the same region of §15.1's plane, are at least {report.minimum:g} "
        "similar by the method above, and share no edge. On a co-occurrence network this is the "
        "corpus discussing two things in the same company without ever putting them in one "
        "passage -- which may be a gap in the corpus or a fact about the subject, and the "
        "network cannot say which. No null model stands behind the threshold; it is a reporting "
        "floor and nothing more.",
        "",
    ]
    if not report.pairs:
        return [*lines, "No pair cleared the threshold.", ""]
    lines += _rows(
        ["node", "node", "role", "similarity", "same module?"],
        [
            [
                report.label(pair.u),
                report.label(pair.v),
                f"{pair.role} {ROLE_NAMES[pair.role]}",
                f"{pair.similarity:.3f}",
                "yes" if pair.same_module else "no",
            ]
            for pair in report.pairs
        ],
    )
    return lines


def _blockmodel_section(report: RolesReport, block: Blockmodel) -> list[str]:
    """The positions and the image matrix, with the caveat that a position is a cluster."""
    lines = [
        "## Positions and the image matrix (§15.2)",
        "",
        f"{block.k} position(s) of sizes {', '.join(str(size) for size in block.sizes)}, from "
        f"{block.method}. The image matrix is the density of edges between two positions: 1.0 "
        "means every possible edge between them is there, 0.0 that none is, and the diagonal is "
        "the density inside a position. A position is a cluster of a similarity matrix and "
        "inherits its instability -- at k and k+1 the blocks can be entirely different, and "
        "nothing here says which k is right.",
        "",
    ]
    lines += _rows(
        ["position", "size", *[f"to {i + 1}" for i in range(block.k)]],
        [
            [str(i + 1), str(len(position)), *[f"{block.image[i, j]:.3f}" for j in range(block.k)]]
            for i, position in enumerate(block.positions)
        ],
    )
    members = "; ".join(
        f"**{i + 1}** "
        + ", ".join(report.label(node) for node in position[:TOP_MEMBERS])
        + (f", +{len(position) - TOP_MEMBERS} more" if len(position) > TOP_MEMBERS else "")
        for i, position in enumerate(block.positions)
    )
    return [*lines, f"Members: {members}", ""]


def render_roles(report: RolesReport) -> str:
    """The roles report as markdown: who plays what, next to the partition that decided it."""
    graph = report.graph
    table = report.table
    counts = table.counts()
    lines = [
        f"# Node roles: {report.persona_id} / {report.network}",
        "",
        "**Implements:** *The Atlas for the Aspiring Network Scientist*, chapter 15 (node "
        "roles) -- §15.1's classic node classification and §15.2's node similarity. §15.3 (node "
        "embeddings) is deliberately absent: that section is a pointer forward to graph neural "
        "networks, and so is this package.",
        "",
        f"**Sampling frame:** {graph.graph.get('frame', 'not recorded')}",
        "",
        f"**n:** {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges, "
        f"{table.modules} module(s). **Partition:** {report.partition_source}.",
        "",
        "**Null model:** none. A role here is a deterministic function of the network and the "
        "partition, not a claim about a population, so there is no p-value to print. What "
        "carries a null is the partition, and `sna analyze` is where it is tested.",
        "",
        "## Roles (§15.1)",
        "",
        "Every node placed by two numbers: `z`, how well connected it is inside its own module, "
        "and `P`, how evenly its ties are spread across the modules. The seven regions are "
        "Guimera and Amaral's cut points, which chapter 15 cites in its bibliography without "
        "printing them; the third column is the same node in the chapter's own vocabulary "
        "(pp. 226-227).",
        "",
    ]
    lines += _rows(
        ["role", "name", "chapter 15 calls it", "nodes", "what it means"],
        [
            [f"`{role}`", name, ATLAS_ROLE[role], f"{counts[role]:,}", ROLE_MEANING[role]]
            for role, name in ROLE_NAMES.items()
        ],
    )
    lines += [
        f"Cut points: hub at z >= {table.thresholds.hub:g}; non-hubs split at P = "
        f"{table.thresholds.ultra_peripheral:g}, {table.thresholds.peripheral:g} and "
        f"{table.thresholds.connector:g}; hubs at P = {table.thresholds.provincial:g} and "
        f"{table.thresholds.hub_connector:g}. With {table.modules} module(s) P cannot exceed "
        f"{table.ceiling:.3f}.",
        "",
    ]
    hubs = table.hubs()
    if hubs:
        lines += ["### The hubs, by role", ""]
        lines += _rows(
            ["node", "module", "z", "P", "role"],
            [
                [
                    report.label(row.node),
                    str(row.module),
                    f"{row.within_module_degree:.2f}",
                    f"{row.participation:.3f}",
                    f"{row.role} {row.role_name}",
                ]
                for row in hubs[:TOP_PAIRS]
            ],
        )
    else:
        lines += [
            f"No node reached z >= {table.thresholds.hub:g}, so this run names no hubs. Where "
            "the modules are small or uniform that is a property of the modules and not of the "
            "network: a module whose members all have the same internal degree has no spread "
            "for a z-score to measure.",
            "",
        ]
    if report.similarity is not None:
        lines += _similarity_section(report, report.similarity)
    if report.blockmodel is not None:
        lines += _blockmodel_section(report, report.blockmodel)
    lines += [
        "## Reading this",
        "",
        "- A role is relative to the partition it was computed on. Both coordinates are defined "
        "against the modules, so another resolution or another seed can turn a connector hub "
        "into a provincial one without a single edge moving.",
        "- Structural equivalence is about neighbours, not about being connected. The chapter's "
        "own pair of indistinguishable nodes (Figure 15.5(a), p. 229) have no edge between "
        "them, which is what makes the section above worth reading.",
        "- The `core` and `periphery` in the third column are not chapter 32's, and §15.1 says "
        'so where it names them: "core, and periphery (the latter two not to be confused with '
        'the core-periphery mesoscale structure that we will see in Chapter 32)" (p. 226). A '
        "core here is a hub of one module of the partition above, so this network has as many "
        "cores as it has modules; chapter 32's core is one set of nodes and needs no partition "
        "at all. The two can disagree completely.",
        "- §15.3 (node embeddings) is not implemented and is not missing: the section defers to "
        "the graph-neural-network chapters, and so does this package. Nor is automorphic "
        "equivalence, the middle of §15.2's three (pp. 230-231): deciding it needs the graph "
        "automorphisms of §41.3, so the strict rung and the loose rung are here and the middle "
        "one is not.",
        "",
    ]
    lines += [f"- {note}" for note in report.notes]
    return "\n".join(lines).rstrip() + "\n"


def roles_payload(report: RolesReport) -> dict[str, Any]:
    """The same report as plain JSON-able data."""
    table = report.table
    scores = report.similarity
    block = report.blockmodel
    payload: dict[str, Any] = {
        "persona_id": report.persona_id,
        "network": report.network,
        "generated_at": report.generated_at,
        "implements": "Atlas ch. 15 (node roles), §15.1 and §15.2",
        "frame": report.graph.graph.get("frame", ""),
        "nodes": report.graph.number_of_nodes(),
        "edges": report.graph.number_of_edges(),
        "modules": table.modules,
        "partition_source": report.partition_source,
        "null_model": "none: a role is a deterministic function of the network and the partition",
        "thresholds": {
            "hub": table.thresholds.hub,
            "ultra_peripheral": table.thresholds.ultra_peripheral,
            "peripheral": table.thresholds.peripheral,
            "connector": table.thresholds.connector,
            "provincial": table.thresholds.provincial,
            "hub_connector": table.thresholds.hub_connector,
            "participation_ceiling": table.ceiling,
        },
        "counts": table.counts(),
        "roles": [
            {
                "node": row.node,
                "label": report.label(row.node),
                "module": row.module,
                "within_module_degree": row.within_module_degree,
                "participation": row.participation,
                "role": row.role,
                "role_name": row.role_name,
                "atlas_role": row.atlas_role,
            }
            for row in table.rows
        ],
        "same_role_never_cooccur": {
            "minimum_similarity": report.minimum,
            "pairs": [
                {
                    "nodes": [pair.u, pair.v],
                    "labels": [report.label(pair.u), report.label(pair.v)],
                    "role": pair.role,
                    "similarity": pair.similarity,
                    "same_module": pair.same_module,
                }
                for pair in report.pairs
            ],
        },
        "notes": report.notes,
    }
    if scores is not None:
        payload["similarity"] = {
            "method": scores.method,
            "section": scores.section,
            "parameters": scores.parameters,
            "top_pairs": [
                {"nodes": [u, v], "similarity": score}
                for u, v, score in scores.top_pairs(TOP_PAIRS)
            ],
        }
    if block is not None:
        payload["blockmodel"] = {
            "k": block.k,
            "method": block.method,
            "positions": [list(position) for position in block.positions],
            "image": [[float(value) for value in row] for row in block.image],
        }
    return payload
