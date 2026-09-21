"""The legendary graphs and their published answers (Atlas §53.4).

*"There are some graphs that are so widely used that you don't really need to look for them in
an online repository. These are the pillars on which the entire cathedral of network science is
founded."* Every one of them has been measured in print, which is what makes them usable as
fixtures: a test that runs Louvain on a fabricated graph can only assert that something came
back, while a test that runs it on the karate club can assert the number Newman got.

Each loader returns the graph **and** a :class:`KnownAnswers` record holding what the literature
says about it, so a test never hard-codes a number without its source next to it. Only graphs
``networkx`` itself ships are here -- nothing is downloaded, and nothing is vendored -- and no
number appears in a record unless it can be checked against a citation or recomputed from the
graph by a reader of this file.

What these graphs are *not*: a sample of anything this package analyses. They are small, old,
hand-collected, and every one of them has a ground truth its authors knew before they drew the
network. A corpus network has none of that, which is exactly why these are the fixtures --
they are the only place where "the method found the right answer" is a statement that can be
made at all.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import networkx as nx

Node = str | int


@dataclass(frozen=True)
class KnownAnswers:
    """What is published about one legendary graph, with the citation that says it.

    ``nodes`` and ``edges`` are the counts of the graph as ``networkx`` ships it; where the
    literature disagrees with that copy -- the karate club has two circulating versions -- the
    disagreement is in ``caveats`` rather than silently resolved.
    """

    name: str
    source: str
    """The citation, as the Atlas gives it in §53.4 or its own bibliography. Where this file
    disagrees with the Atlas -- a publisher, an edition -- the disagreement is written into the
    citation rather than settled silently, so a reader holding the book open can see which of
    the two they are reading."""
    frame: str
    """What a node is, what an edge is, and what the network was a record of. Printed by a
    report exactly as a corpus network's frame is."""
    nodes: int
    edges: int
    weighted: bool
    groups: dict[str, tuple[Node, ...]] = field(default_factory=dict)
    """The published grouping of the nodes -- the two karate factions, the two modes of the
    Southern women network -- keyed by its name. Empty when the graph has no published one."""
    modularity: float | None = None
    """The best modularity published for this graph, or ``None`` when nobody has fixed one."""
    modularity_communities: int | None = None
    """How many communities that modularity was reached with."""
    modularity_note: str = ""
    """The conditions the published modularity holds under: which copy of the graph, weighted
    or not. A modularity quoted without them is not comparable with a computed one."""
    top_betweenness: Node | None = None
    top_degree: Node | None = None
    caveats: tuple[str, ...] = ()
    """What a test must not conclude from this graph, in the words the source uses."""


def karate_club() -> tuple[nx.Graph, KnownAnswers]:
    """Zachary's karate club: the network every method is tried on first (§53.4).

    Thirty-four members of a university karate club, joined when they interacted outside club
    activities, recorded by Wayne Zachary over two years in the early 1970s. The club then split
    in a dispute between the instructor (node 0, "Mr. Hi") and the president (node 33,
    "Officer"), and each member's side is on the node as ``club`` -- a ground truth collected
    independently of the edges, which is the rare property that makes this graph a test and not
    a demonstration.

    The Atlas tells the fun fact that matters here: Zachary's own adjacency matrix has one
    asymmetric entry, so two karate clubs circulate, one with 77 edges and one with 78. This is
    the 78-edge one, the version Newman and Girvan used. Edge weights are Zachary's count of the
    contexts in which two members met; the published community-detection numbers are computed on
    the graph *without* them, which is what ``unweighted`` is for.
    """
    graph = nx.karate_club_graph()
    factions: dict[str, tuple[Node, ...]] = {
        club: tuple(n for n, data in graph.nodes(data=True) if data["club"] == club)
        for club in ("Mr. Hi", "Officer")
    }
    known = KnownAnswers(
        name="karate club",
        source=(
            "Wayne W. Zachary. An information flow model for conflict and fission in small "
            "groups. Journal of Anthropological Research, 33(4):452-473, 1977."
        ),
        frame=(
            "34 members of one university karate club, 1970-1972; an edge is an interaction "
            "outside club activities, weighted by how many contexts it happened in. It records "
            "one club that split, not clubs in general."
        ),
        nodes=34,
        edges=78,
        weighted=True,
        groups=factions,
        modularity=0.4198,
        modularity_communities=4,
        modularity_note=(
            "the maximum modularity of the unweighted 78-edge karate club, 0.4198 over four "
            "communities, found by exhaustive optimisation (Ulrik Brandes, Daniel Delling, "
            "Marco Gaertler, Robert Goerke, Martin Hoefer, Zoran Nikoloski and Dorothea "
            "Wagner. On modularity clustering. IEEE TKDE, 20(2):172-188, 2008); Newman reports "
            "0.419 for the same graph (M. E. J. Newman. Modularity and community structure in "
            "networks. PNAS, 103(23):8577-8582, 2006). It does not hold for the weighted graph, "
            "which scores higher, nor for the 77-edge copy."
        ),
        top_degree=33,
        caveats=(
            "Two copies of this graph circulate, with 77 and 78 edges, because one entry of "
            "Zachary's published matrix is asymmetric (Atlas §53.4). Any number quoted for it "
            "belongs to one copy; this is the 78-edge one.",
            "The four communities modularity maximisation finds are not the two factions. The "
            "factions are the ground truth; the partition is a method's answer, and reporting "
            "the first as if it were the second is the oldest mistake made with this network.",
            "Zachary explains member 9 (node 8 here) himself: he sparred with the officers' "
            "side but joined the instructor's club, because he was three weeks from a black "
            "belt test he could only take with the instructor. A structural method cannot get "
            "that node right, and one that does has been tuned until it did.",
        ),
    )
    graph.graph.update(network=known.name, frame=known.frame, source=known.source)
    return graph, known


def les_miserables() -> tuple[nx.Graph, KnownAnswers]:
    """The Les Misérables co-appearance network (§53.4).

    A node is a character in Hugo's novel; two characters are joined when they appear in the
    same chapter, and the weight is how many chapters they share. Knuth compiled it for the
    Stanford GraphBase, which is why it turns up far outside network science -- it is the graph
    most data-visualisation tutorials draw.

    It is a co-appearance network, so it has the defect every co-occurrence network has,
    including the ones this package builds: an edge means two names were written down close
    together, not that the people met.
    """
    graph = nx.les_miserables_graph()
    known = KnownAnswers(
        name="les miserables",
        source=(
            "Donald E. Knuth. The Stanford GraphBase: a platform for combinatorial computing. "
            "ACM Press, New York, 1993."
        ),
        frame=(
            "77 characters of Victor Hugo's novel; an edge is a shared chapter, weighted by how "
            "many chapters the two share. It records one editor's reading of one book."
        ),
        nodes=77,
        edges=254,
        weighted=True,
        top_degree="Valjean",
        caveats=(
            "Co-appearance is not interaction: two characters named in one chapter need never "
            "meet in it.",
            "The community structure everyone shows for this graph follows the novel's parts, "
            "so it is a check that a method recovers a known grouping, never a discovery.",
        ),
    )
    graph.graph.update(network=known.name, frame=known.frame, source=known.source)
    return graph, known


def florentine_families() -> tuple[nx.Graph, KnownAnswers]:
    """Marriage ties between Renaissance Florentine families (§53.4).

    Fifteen families, joined by a marriage between them, from Padgett's reading of Kent's
    archival study. It is the standard example of centrality mattering more than size: the
    Medici hold the highest betweenness by a wide margin -- more than twice the next family --
    while never having the most wealth, which is the argument the network is always used to make.

    The copy ``networkx`` ships has 15 families, not the 16 the source tabulates: the Pucci
    married nobody in the sample and are dropped as an isolate. That is worth stating whenever a
    count from this graph is compared with one in print.
    """
    graph = nx.florentine_families_graph()
    known = KnownAnswers(
        name="florentine families",
        source=(
            "Ronald L. Breiger and Philippa E. Pattison. Cumulated social roles: the duality of "
            "persons and their algebras. Social Networks, 8(3):215-256, 1986, after John F. "
            "Padgett's coding of D. Kent's Florentine archival data."
        ),
        frame=(
            "15 elite families of Florence around 1430; an edge is a marriage tie between two "
            "of them. It records the families one historian tabulated, and marriages only -- "
            "not the business ties recorded in the same source."
        ),
        nodes=15,
        edges=20,
        weighted=False,
        top_betweenness="Medici",
        top_degree="Medici",
        caveats=(
            "The Pucci, the sixteenth family in the source, have no marriage tie in the sample "
            "and are absent from this copy of the graph entirely.",
            "Marriage ties are one layer of a multilayer network; the business ties are a "
            "second, and a claim about Medici power rests on both.",
        ),
    )
    graph.graph.update(network=known.name, frame=known.frame, source=known.source)
    return graph, known


def southern_women() -> tuple[nx.Graph, KnownAnswers]:
    """The Davis Southern women network: the canonical two-mode graph (§53.4).

    Eighteen women in Natchez, Mississippi, in the 1930s, connected to the 14 informal social
    events they were recorded as attending: 89 attendances in all. It is the bipartite network,
    the one every projection scheme (ch. 26) and every bipartite community method (ch. 39) is
    tried on, because the ethnographers named two overlapping groups of women before anybody
    computed anything.

    Which women belong to which group is *not* settled, and that is the useful part. Freeman's
    meta-analysis ran 21 published methods over these data and found them agreeing on the cores
    of two groups and disagreeing about the women at the edges, so a method that produces a
    clean two-way split here has resolved an ambiguity the data do not contain. The record
    therefore carries the modes, which are certain, and not a partition of the women, which is
    not.
    """
    graph = nx.davis_southern_women_graph()
    women = tuple(sorted(n for n, data in graph.nodes(data=True) if data["bipartite"] == 0))
    events = tuple(sorted(n for n, data in graph.nodes(data=True) if data["bipartite"] == 1))
    known = KnownAnswers(
        name="southern women",
        source=(
            "Allison Davis, Burleigh B. Gardner and Mary R. Gardner. Deep South: a social "
            "anthropological study of caste and class. University of Chicago Press, 1941. The "
            "Atlas's own footnote (§53.4, n. 73) gives the publisher as the University of "
            "South Carolina Press, which is a later reissue; the 1941 edition this network "
            "comes from is Chicago's."
        ),
        frame=(
            "18 women and the 14 informal social events they were observed attending in one "
            "Mississippi town in the 1930s; an edge is an attendance, and the observation was "
            "made by reading a local newspaper's society column and by interview. It records "
            "who was written down as attending, not who attended."
        ),
        nodes=32,
        edges=89,
        weighted=False,
        groups={"women": women, "events": events},
        caveats=(
            "It is two-mode: women are joined to events, never to each other. Any woman-woman "
            "edge is a projection, and the projection scheme is a choice that changes the "
            "answer (Atlas ch. 26).",
            "The two groups the ethnographers described overlap and their boundaries are "
            "disputed: Linton C. Freeman. Finding social groups: a meta-analysis of the "
            "Southern women data. In Dynamic Social Network Modeling and Analysis, National "
            "Academies Press, 2003, compares 21 analyses and finds agreement on the cores "
            "only. A method that splits these 18 women cleanly in two has decided something "
            "the data leave open.",
        ),
    )
    graph.graph.update(network=known.name, frame=known.frame, source=known.source)
    return graph, known


#: Every legendary graph this repository has, by name, for a test that wants to sweep them.
LEGENDARY: dict[str, Callable[[], tuple[nx.Graph, KnownAnswers]]] = {
    "karate club": karate_club,
    "les miserables": les_miserables,
    "florentine families": florentine_families,
    "southern women": southern_women,
}


def unweighted(graph: nx.Graph) -> nx.Graph:
    """The same graph with every edge weight dropped, keeping nodes, edges and node data.

    The published community-detection numbers for the karate club and for Les Misérables are
    computed on the unweighted graph -- modularity with a ``weight`` attribute present is a
    different quantity, and scores higher here -- so a test that compares against them has to
    drop the weights first, and say that it did.
    """
    plain = nx.DiGraph() if graph.is_directed() else nx.Graph()
    plain.graph.update(graph.graph)
    plain.add_nodes_from(graph.nodes(data=True))
    plain.add_edges_from(graph.edges())
    return plain
