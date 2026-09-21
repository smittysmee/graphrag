"""One node's neighbourhood as a network of its own: the ego network of Atlas §30.1.

*"You first have to identify your 'ego': the node on which the ego network is centered. Then, you
select all of its neighbors and the connections among them"* (p. 433). That is the whole
construction, and it is the mesoscale in miniature -- not a property of the network as a whole and
not a property of one node, but a description of the local neighbourhood a node sits in.

The chapter's own warning is the reason this module reports two densities rather than one:

    *"A consequence of this procedure is that we know that an ego node is connected to all nodes
    in its ego network. This is unfortunate in some cases, depending on our analytic needs. For
    instance, all ego networks have a single connected component and will have a diameter of two.
    If those forced properties are undesirable, one can extract an ego network and then remove the
    ego and all its connections."* (§30.1, p. 433)

So an ego network's connectedness, its diameter and most of its density are facts about the
procedure rather than about the node. What survives the ego's removal is not: the density of the
alters **without** the ego is exactly the ego's local clustering coefficient (§12.2) -- the
triangles the node sits in over the pairs of neighbours it could close -- and the pairs it leaves
unjoined are the structural holes the node brokers. This report prints the two side by side and
computes the clustering independently of the density, so the identity is a check rather than an
assertion.

**What the book defines and what this adds.** §30.1 defines the construction, names the removal of
the ego, and points at social capital (Borgatti, Jones and Everett, *Network measures of social
capital*, 1998) without giving a formula. The brokerage count here -- alter pairs joined only
through the ego -- is the structural-hole reading of that pointer, and it is `1 - C_local` in
share terms rather than a new quantity. The attribute composition is §30.2's question asked of one
neighbourhood, and it is the calculation the chapter's own exercise asks for (§30.6, exercise 1:
*"For each ego network, calculate the share of right-leaning nodes"*), together with §30.4's
majority illusion: a value that is a minority in the network as a whole and a majority among this
ego's alters is what Lerman, Yan and Wu's *majority illusion* looks like from inside one node's
neighbourhood (§30.4, p. 437).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import networkx as nx

from graphrag.sna.attributes import attribute_labels, attribute_sources, ei_ratio
from graphrag.sna.export import INHERITED, OWN, ego
from graphrag.sna.measures import undirected_view
from graphrag.sna.ties import TWO_MODE, two_mode_kind, two_mode_sentence

__all__ = [
    "AlterRow",
    "EgoComposition",
    "EgoReport",
    "ego_payload",
    "ego_report",
    "render_ego",
]

#: How many alters the rendered section lists by name. An ego network of a corpus hub has
#: hundreds; the section is a reading of one neighbourhood, not a dump of it.
MAX_LISTED = 20

#: More than half: what "a majority" means in :attr:`EgoComposition.illusion` and in §30.4's
#: majority illusion, where the question is whether a node's neighbours make a minority value
#: look like the common one.
MAJORITY = 0.5

#: Above this local clustering a neighbourhood is read as closed, below the other one as open
#: (§30.1 into §30.3). Conventions of this package -- the chapter gives no thresholds -- and the
#: numbers themselves are printed beside the words.
CLOSED = 0.6
OPEN = 0.2


@dataclass(frozen=True)
class AlterRow:
    """One alter, as the section lists it: how tied it is to the ego and to the other alters."""

    node: str
    label: str
    weight_to_ego: float
    """The weight of the edge joining it to the ego, which is how many documents or passages the
    two share. 0.0 for an alter that is in the ego network only because ``radius`` reached it."""
    degree_in_ego: int
    """How many of the other alters it is joined to, the ego excluded. 0 means the ego is its
    only route into this neighbourhood."""
    value: str = ""
    """Its value for the attribute the composition was asked about, empty when it carries none."""


@dataclass(frozen=True)
class EgoComposition:
    """What the alters are, against what the network is: §30.2 asked of one neighbourhood."""

    key: str
    ego_value: str = ""
    ego_source: str = ""
    """:data:`graphrag.sna.export.OWN` or :data:`graphrag.sna.export.INHERITED`, empty when the
    ego carries no value. A borrowed value makes every number below circular in the way
    :mod:`graphrag.sna.attributes` describes, and the report says so."""
    alters: int = 0
    labelled_alters: int = 0
    borrowed_alters: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    """How many alters carry each value, most common first."""
    network_counts: dict[str, int] = field(default_factory=dict)
    """The same over every labelled node in the whole network, which is what makes the alters'
    mix readable: a neighbourhood that is 70% one value in a network that is 70% that value has
    no composition to report."""
    internal: int = 0
    """Labelled alters carrying the ego's own value: the ``I`` of the ego's EI index."""
    external: int = 0
    """Labelled alters carrying a different one: the ``E``."""
    ei: float | None = None
    """This ego's own EI index over its alters, ``(E - I) / (E + I)``: -1 when every alter is
    like it, +1 when none is. ``None`` when the ego carries no value or no alter does."""

    @property
    def shares(self) -> dict[str, float]:
        """Each value's share of the *labelled* alters, which is §30.6's exercise 1."""
        total = sum(self.counts.values())
        return {value: count / total for value, count in self.counts.items()} if total else {}

    @property
    def network_shares(self) -> dict[str, float]:
        total = sum(self.network_counts.values())
        return (
            {value: count / total for value, count in self.network_counts.items()} if total else {}
        )

    @property
    def illusion(self) -> tuple[str, ...]:
        """Values in a minority network-wide and a majority among these alters (§30.4).

        The majority illusion: *"Even if the majority of people do not use drugs, we can draw a
        network in which everybody thinks that the opposite is true"* (p. 437). One ego cannot
        show it -- the effect is a property of the network, and Lerman et al. define it over every
        node -- so this names the values for which *this* neighbourhood is misleading, which is
        the per-node half of it.
        """
        network = self.network_shares
        return tuple(
            value
            for value, share in sorted(self.shares.items())
            if share > MAJORITY and network.get(value, 0.0) < MAJORITY
        )


@dataclass(frozen=True)
class EgoReport:
    """One ego network, measured with and without its ego (§30.1)."""

    node: str
    label: str
    radius: int
    with_ego: bool
    """Which of the two views the section leads with. Both are always computed: the book offers
    the removal of the ego as the remedy for the procedure's forced properties, not as a
    different analysis."""
    network_nodes: int
    network_edges: int
    alters: int
    """Nodes in the ego network other than the ego. Equal to the ego's degree at radius 1."""
    neighbours: int
    """The ego's degree in the whole network, which is how many alters radius 1 would give."""
    edges_with_ego: int
    edges_without_ego: int
    density_with_ego: float
    density_without_ego: float
    local_clustering: float
    """The ego's local clustering coefficient in the *whole* network (§12.2), computed
    independently of the densities above so that the §30.1 identity can be checked rather than
    assumed. Equal to ``density_without_ego`` at radius 1 and not in general beyond it."""
    degree_share: float
    """The share of the network's other nodes that are this ego's alters: the ego's normalised
    degree centrality (§14.1), and the reason a hub's ego network is not a neighbourhood."""
    brokerage_pairs: int
    """Pairs of the ego's *neighbours* that no edge joins, so the ego is their only route to each
    other in this neighbourhood: the structural holes it sits across."""
    brokerage: float
    """The same as a share of all neighbour pairs, which is ``1 - local_clustering``."""
    components_without_ego: int
    """How many pieces the alters fall into once the ego is removed. §30.1's point: with the ego
    in place this is 1 by construction, so it is only informative after the removal."""
    weight_to_ego: float
    """The total weight of the ego's own edges: its strength, in shared documents or passages."""
    isolated_alters: int
    """Alters with no edge to any other alter. Every one of them reaches this neighbourhood only
    through the ego."""
    rows: tuple[AlterRow, ...] = ()
    composition: EgoComposition | None = None
    two_mode: str = ""
    """Empty, or what :func:`graphrag.sna.ties.two_mode_kind` found: on a network where no edge
    can sit in a triangle, the density without the ego is 0, the local clustering is 0 and the
    brokerage is 1.0 for *every* node, by construction. The reading is withheld rather than
    printed as if this node were a broker."""
    flattened: str = ""
    frame: str = ""
    notes: tuple[str, ...] = ()

    @property
    def density(self) -> float:
        """The density of the view this report leads with."""
        return self.density_with_ego if self.with_ego else self.density_without_ego

    @property
    def identity_holds(self) -> bool:
        """Whether the §30.1 identity was checkable here, and held.

        Only at radius 1: beyond it the alters include nodes that are not the ego's neighbours,
        so the density of the ego-less view is no longer the ego's clustering coefficient.
        """
        return self.radius == 1 and abs(self.density_without_ego - self.local_clustering) < 1e-9

    @property
    def reading(self) -> str:
        """What this neighbourhood is, in §30.1's terms, in one sentence."""
        if self.two_mode and self.alters > 1:
            return two_mode_sentence(
                self.two_mode,
                f"This node's {self.alters:,} alters are all of the other kind, so none of them "
                f"can be joined to another: the density without the ego is 0, the local "
                f"clustering is 0 and all {self.brokerage_pairs:,} pair(s) of neighbours count "
                "as brokered, for every node of this network alike.",
                section="§30.1's brokerage",
            )
        if self.alters == 0:
            return (
                "This node has no neighbour at all, so it has no ego network: the corpus records "
                "it alone. Nothing below is a measurement."
            )
        if self.alters == 1:
            return (
                "One alter, so there is no pair of neighbours to close and no brokerage to read: "
                "the density without the ego is undefined and reported as 0.0 by convention "
                "(§12.2), which is not the same as its neighbours being strangers."
            )
        if self.local_clustering >= CLOSED:
            return (
                f"A closed neighbourhood: {self.local_clustering:.0%} of the pairs of this node's "
                "neighbours are joined to each other, so it sits inside a group that already "
                "knows itself and brokers little. §30.3's reading is that its ties are the strong "
                "ones and that what reaches it reaches its neighbours anyway."
            )
        if self.local_clustering <= OPEN:
            return (
                f"An open neighbourhood: only {self.local_clustering:.0%} of the pairs of this "
                f"node's neighbours are joined, leaving {self.brokerage_pairs:,} pair(s) whose "
                "only route to each other here is through this node. That is the brokerage "
                "position of §30.1 and the weak-tie position of §30.3 -- and on a corpus network "
                "it is also what a node looks like when it was written about in unrelated "
                "contexts, which is not the same thing."
            )
        return (
            f"A mixed neighbourhood: {self.local_clustering:.0%} of the pairs of this node's "
            f"neighbours are joined and {self.brokerage_pairs:,} pair(s) are not, so it is partly "
            "inside a group and partly between groups."
        )


def ego_report(
    graph: nx.Graph,
    node: str,
    *,
    radius: int = 1,
    with_ego: bool = True,
    by: str | None = None,
) -> EgoReport:
    """The ego network of one node, measured both ways §30.1 describes it.

    ``radius`` is how many hops out the neighbourhood reaches; 1 is the chapter's definition and
    the only radius at which the identity below holds. ``with_ego`` decides which view the report
    leads with, never what is computed: both densities, both edge counts and the ego's clustering
    are always there, because the section's point is the difference between them.

    What comes back:

    *Two densities.* With the ego, the density is inflated by the ego's own edges -- it is joined
    to every alter by construction, which is why *"all ego networks have a single connected
    component and will have a diameter of two"* (p. 433). Without the ego, the density is the
    ego's **local clustering coefficient** (§12.2): the same ratio of the same triangles over the
    same pairs. The two are computed separately here, from the subgraph and from the whole
    network, so that agreeing is evidence rather than tautology -- and beyond radius 1 they part
    company, which :attr:`EgoReport.identity_holds` records.

    *Brokerage.* How many pairs of the ego's neighbours are joined by no edge, so that the ego is
    their only route to each other. It is the count behind ``1 - C_local``, and on this package's
    networks a high one has two readings that the number cannot separate: a node that genuinely
    connects unrelated groups, and a node that was written about in unrelated contexts.

    *Composition*, when ``by`` names an attribute. What the alters are, against what the whole
    network is (§30.2 over one neighbourhood), with the provenance of every label carried through
    (:func:`graphrag.sna.attributes.attribute_sources`) because a borrowed label makes a
    neighbourhood look homophilous for the reason the whole network does: the edges and the
    labels came from the same documents.

    Raises ``KeyError`` when the node is not in the network, which is
    :func:`graphrag.sna.export.ego`'s contract; raises ``ValueError`` for a radius below 1.
    """
    if radius < 1:
        msg = f"radius must be at least 1 hop, got {radius}"
        raise ValueError(msg)
    flat, flattened = undirected_view(graph)
    if node not in flat:
        msg = f"{node!r} is not in this network"
        raise KeyError(msg)
    view: nx.Graph = ego(flat, node, radius=radius).copy()
    alters = [n for n in view if n != node]
    without: nx.Graph = view.subgraph(alters).copy()
    neighbours = sorted(flat.neighbors(node))
    pairs = len(neighbours) * (len(neighbours) - 1) // 2
    joined = without.subgraph(neighbours).number_of_edges()
    clustering = float(nx.clustering(flat, node))
    others = flat.number_of_nodes() - 1
    notes = _notes(flat, node, radius, view, without)
    return EgoReport(
        node=node,
        label=str(flat.nodes[node].get("label") or flat.nodes[node].get("name") or node),
        radius=radius,
        with_ego=with_ego,
        network_nodes=flat.number_of_nodes(),
        network_edges=flat.number_of_edges(),
        alters=len(alters),
        neighbours=len(neighbours),
        edges_with_ego=view.number_of_edges(),
        edges_without_ego=without.number_of_edges(),
        density_with_ego=float(nx.density(view)),
        density_without_ego=float(nx.density(without)) if without.number_of_nodes() > 1 else 0.0,
        local_clustering=clustering,
        degree_share=len(neighbours) / others if others > 0 else 0.0,
        brokerage_pairs=pairs - joined,
        brokerage=(pairs - joined) / pairs if pairs else 0.0,
        components_without_ego=nx.number_connected_components(without),
        weight_to_ego=float(flat.degree(node, weight="weight")),
        isolated_alters=sum(1 for n in alters if without.degree(n) == 0),
        rows=_rows(flat, view, without, node, alters, by),
        composition=_composition(flat, node, alters, by) if by else None,
        two_mode=two_mode_kind(flat),
        flattened=flattened,
        frame=str(flat.graph.get("frame", "")),
        notes=notes,
    )


def _rows(
    graph: nx.Graph,
    view: nx.Graph,
    without: nx.Graph,
    node: str,
    alters: Sequence[str],
    key: str | None,
) -> tuple[AlterRow, ...]:
    """The alters the section lists, heaviest tie to the ego first, then most connected."""
    labels = attribute_labels(graph, key) if key else {}
    rows = [
        AlterRow(
            node=alter,
            label=str(graph.nodes[alter].get("label") or graph.nodes[alter].get("name") or alter),
            weight_to_ego=float(view[node][alter].get("weight", 1.0) or 1.0)
            if view.has_edge(node, alter)
            else 0.0,
            degree_in_ego=int(without.degree(alter)),
            value=labels.get(alter, ""),
        )
        for alter in alters
    ]
    rows.sort(key=lambda row: (-row.weight_to_ego, -row.degree_in_ego, row.node))
    return tuple(rows[:MAX_LISTED])


def _composition(graph: nx.Graph, node: str, alters: Sequence[str], key: str) -> EgoComposition:
    """What the alters carry for one attribute, against what the whole network carries (§30.2).

    The ego's own EI index is computed from the same counts the network-wide one uses
    (:func:`graphrag.sna.attributes.ei_index`), so a neighbourhood and its network are read in the
    same currency. Alters with no value are outside every number, exactly as they are in the
    attribute section: an untagged node is not a third value.
    """
    labels = attribute_labels(graph, key)
    sources = attribute_sources(graph, key)
    among = Counter(labels[a] for a in alters if a in labels)
    value = labels.get(node, "")
    internal = among.get(value, 0) if value else 0
    external = sum(among.values()) - internal if value else 0
    return EgoComposition(
        key=key,
        ego_value=value,
        ego_source=sources.get(node, ""),
        alters=len(alters),
        labelled_alters=sum(among.values()),
        borrowed_alters=sum(1 for a in alters if sources.get(a) == INHERITED),
        counts=dict(sorted(among.items(), key=lambda item: (-item[1], item[0]))),
        network_counts=dict(
            sorted(Counter(labels.values()).items(), key=lambda item: (-item[1], item[0]))
        ),
        internal=internal,
        external=external,
        ei=ei_ratio(internal, external) if value else None,
    )


def _notes(
    graph: nx.Graph, node: str, radius: int, view: nx.Graph, without: nx.Graph
) -> tuple[str, ...]:
    """What the procedure forced, so that none of it is read as a property of this node."""
    notes = [
        "The ego is joined to every alter by construction, so the with-ego view has one "
        "component and a diameter of 2 whatever the node is like (§30.1). Neither is a finding; "
        "the view without the ego is where the readings are."
    ]
    if radius > 1:
        notes.append(
            f"At radius {radius} the alters include nodes the ego is not joined to, so the "
            "density without the ego is no longer the ego's local clustering coefficient and the "
            "two numbers below are different quantities. Radius 1 is §30.1's definition."
        )
    if without.number_of_nodes() > 1 and nx.number_connected_components(without) > 1:
        notes.append(
            f"Removing the ego breaks its neighbourhood into "
            f"{nx.number_connected_components(without):,} pieces, which is what §30.1 means by "
            "the ego's connectedness being an artefact of the construction."
        )
    kind = two_mode_kind(graph)
    if kind == TWO_MODE:
        notes.append(
            "Brokerage and local clustering are built out of shared neighbours, and this network "
            "is two-mode (§6.4): no two alters of any node are of the same kind, so none of them "
            "can be adjacent and every node scores a clustering of 0 and a brokerage of 1.0. "
            "Project onto one mode -- `--project speakers` or `--project entities` (ch. 26) -- "
            "before reading brokerage off this node."
        )
    elif kind:
        notes.append(
            "Brokerage and local clustering are built out of shared neighbours, and this network "
            "is 2-colourable: it has no triangle anywhere, so every node's clustering is 0 "
            "whatever its position. There is nothing to project -- the brokerage above is the "
            "shape of the network, not of this node."
        )
    if view.number_of_nodes() == graph.number_of_nodes():
        notes.append(
            "This ego network is the whole network: every node is within "
            f"{radius} hop(s) of {node}. A mesoscale reading needs a neighbourhood smaller than "
            "the network it sits in."
        )
    return tuple(notes)


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_ego(report: EgoReport) -> list[str]:
    """The ego section: the frame, the n, the absent null, and the chapter it implements."""
    frame = f" Frame: {report.frame}" if report.frame else ""
    view = "with the ego" if report.with_ego else "without the ego"
    lines = [
        f"## Ego network: {report.label}",
        "",
        f"**Sampling frame.** The {report.alters:,} node(s) within {report.radius} hop(s) of "
        f"`{report.node}` in a network of {report.network_nodes:,} nodes and "
        f"{report.network_edges:,} edges, and the edges among them.{frame}",
        "",
        f"**n.** {report.alters:,} alter(s), {report.edges_with_ego:,} edge(s) with the ego and "
        f"{report.edges_without_ego:,} among the alters alone. Reported {view}.",
        "",
        "**Null model.** None. An ego network is a view of the network as it was built, not a "
        "claim about a population; the section that does carry nulls is `--by`, which asks "
        "whether the whole network sorts by an attribute (§30.2).",
        "",
        "**Implements.** §30.1 (ego networks), with §12.2's local clustering as the identity "
        "below and §30.2's composition when `--by` is given.",
        "",
        f"- **Alters** {report.alters:,} of {report.network_nodes - 1:,} other nodes "
        f"({report.degree_share:.1%} of the network); the ego's own edges weigh "
        f"{_num(report.weight_to_ego)} in total.",
        f"- **Density with the ego** {report.density_with_ego:.4f} over "
        f"{report.alters + 1:,} node(s).",
        f"- **Density without the ego** {report.density_without_ego:.4f} — which is the ego's "
        f"**local clustering coefficient**, {report.local_clustering:.4f}, computed from the "
        "whole network rather than from this subgraph"
        + (". The two agree, as §30.1 says they must." if report.identity_holds else "."),
        f"- **Brokerage** {report.brokerage_pairs:,} of "
        f"{report.neighbours * (report.neighbours - 1) // 2:,} pair(s) of neighbours are joined "
        f"only through this node ({report.brokerage:.0%}, which is 1 - the clustering above).",
        f"- **Without the ego** the alters fall into {report.components_without_ego:,} "
        f"component(s), {report.isolated_alters:,} of them alone.",
        "",
        report.reading,
        "",
    ]
    if report.rows:
        lines += [
            f"The alters, heaviest tie first (up to {MAX_LISTED}). `ties among alters` counts "
            "edges to the other alters only, so a 0 there is a node the ego is the only route "
            "to:",
            "",
        ]
        header = ["alter", "weight to ego", "ties among alters"]
        rows = [[row.label, _num(row.weight_to_ego), str(row.degree_in_ego)] for row in report.rows]
        if report.composition is not None:
            header.append(report.composition.key)
            for line, row in zip(rows, report.rows, strict=True):
                line.append(row.value or "—")
        lines += _table(header, rows)
    if report.composition is not None:
        lines += _render_composition(report.composition)
    lines += [f"- {note}" for note in report.notes]
    return [*lines, ""]


def _provenance(source: str) -> str:
    """Where the ego's own value came from, in words, because the two are not equal evidence."""
    return "recorded about the node itself" if source == OWN else "borrowed from its documents"


def _render_composition(composition: EgoComposition) -> list[str]:
    """The alters' attribute mix against the network's, and what it does and does not show."""
    ego_value = composition.ego_value or "no value"
    lines = [
        f"### What the alters are: `{composition.key}`",
        "",
        f"The ego carries **{ego_value}**"
        + (f" ({_provenance(composition.ego_source)})" if composition.ego_source else "")
        + f". {composition.labelled_alters:,} of {composition.alters:,} alter(s) carry a value; "
        "the rest are untagged, not a third value. This is §30.2 asked of one neighbourhood and "
        "§30.6's first exercise: the share of each value among a node's alters.",
        "",
    ]
    network = composition.network_shares
    lines += _table(
        ["value", "alters", "share of labelled alters", "share of the network"],
        [
            [
                value,
                f"{count:,}",
                f"{composition.shares.get(value, 0.0):.0%}",
                f"{network.get(value, 0.0):.0%}",
            ]
            for value, count in composition.counts.items()
        ],
    )
    if composition.ei is not None:
        lines += [
            f"- **This ego's EI index** {composition.ei:+.4f} over {composition.internal:,} "
            f"alter(s) like it and {composition.external:,} unlike it: -1 is a neighbourhood "
            "entirely of its own kind, +1 entirely of others'. It is a count of this node's "
            "alters and nothing else — no null, and none of §30.2's correction for how common "
            "each value is.",
            "",
        ]
    if composition.illusion:
        lines += [
            "- **Majority illusion (§30.4).** "
            + ", ".join(f"`{value}`" for value in composition.illusion)
            + " is a minority in this network and a majority among these alters, so from this "
            "node the corpus looks like something it is not. Lerman, Yan and Wu's point is that "
            "this can be true of *every* node at once (p. 437); one neighbourhood only shows "
            "that it is true here.",
            "",
        ]
    if composition.borrowed_alters:
        lines += [
            f"- {composition.borrowed_alters:,} of {composition.alters:,} alter(s) carry no value "
            "of their own and were given the one most of their documents carry. Those labels come "
            "from the same documents as the edges that make them alters, so a neighbourhood full "
            "of them looks homophilous by construction — see the `--by` section for what that "
            "does to the coefficient.",
            "",
        ]
    return lines


def ego_payload(report: EgoReport) -> dict[str, object]:
    """The same section as plain JSON-able data, for ``--json``."""
    composition = report.composition
    payload: dict[str, object] = {
        "node": report.node,
        "label": report.label,
        "radius": report.radius,
        "with_ego": report.with_ego,
        "network_nodes": report.network_nodes,
        "network_edges": report.network_edges,
        "alters": report.alters,
        "neighbours": report.neighbours,
        "edges_with_ego": report.edges_with_ego,
        "edges_without_ego": report.edges_without_ego,
        "density_with_ego": report.density_with_ego,
        "density_without_ego": report.density_without_ego,
        "local_clustering": report.local_clustering,
        "identity_holds": report.identity_holds,
        "degree_share": report.degree_share,
        "brokerage_pairs": report.brokerage_pairs,
        "brokerage": report.brokerage,
        "components_without_ego": report.components_without_ego,
        "two_mode": report.two_mode,
        "isolated_alters": report.isolated_alters,
        "weight_to_ego": report.weight_to_ego,
        "alter_rows": [
            {
                "node": row.node,
                "label": row.label,
                "weight_to_ego": row.weight_to_ego,
                "degree_in_ego": row.degree_in_ego,
                "value": row.value,
            }
            for row in report.rows
        ],
        "reading": report.reading,
        "frame": report.frame,
        "notes": list(report.notes),
    }
    if composition is not None:
        payload["composition"] = {
            "key": composition.key,
            "ego_value": composition.ego_value,
            "ego_source": composition.ego_source,
            "alters": composition.alters,
            "labelled_alters": composition.labelled_alters,
            "borrowed_alters": composition.borrowed_alters,
            "counts": composition.counts,
            "shares": composition.shares,
            "network_counts": composition.network_counts,
            "network_shares": composition.network_shares,
            "internal": composition.internal,
            "external": composition.external,
            "ei": composition.ei,
            "illusion": list(composition.illusion),
        }
    return payload
