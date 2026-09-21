"""The same network over two time windows, and an honest account of what changed.

A before-and-after is the easiest analysis to get wrong. Two Louvain partitions of two graphs
carry community numbers that mean nothing across runs, so "community 2 grew" is meaningless;
what can be compared is how much the *partitioning* agrees on the nodes both windows contain,
which is what the adjusted Rand index and normalised mutual information measure. And a node
that rose twenty places in a centrality ranking may simply have stayed put while half the
network left, which is why the node sets are reported before the ranks are.

So this module reports, in this order: n for each build, who entered and who left, how far the
two partitions agree over the nodes in common, and the largest rank changes among those nodes
only. Nothing is compared across the whole of both builds, because the two whole networks are
not two measurements of one thing.

The two builds are usually two time windows. They can be two attribute values instead -- one
population of a corpus against another, with ``--where`` and ``--where2`` -- and every caution
above holds unchanged, including the one that matters most: two populations that share no nodes
produce no similarity score, and saying so is the finding.

``--topology`` adds a second, independent reading (ch. 48, ATL-48): where the sections above ask
whether the *nodes* both builds share moved, :mod:`graphrag.sna.topodist` asks whether the two
builds' *whole topologies* look alike, node overlap or none at all. A before-and-after with no
shared nodes still has a topological distance -- two node sets with nothing in common can still
be two facsimiles of the same shape -- which is why the flag exists rather than folding into the
node-based sections above.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from graphrag import __version__
from graphrag.graph.store import GraphStore
from graphrag.sna.backbone import DEFAULT_ALPHA
from graphrag.sna.cluster import LouvainResult, compare_partitions, louvain
from graphrag.sna.export import Project, describe, where_text
from graphrag.sna.guide import ALWAYS
from graphrag.sna.layers import window_graph, windows
from graphrag.sna.measures import (
    centralities_for,
    centrality,
    centrality_meaning,
    directed_notes,
    summary,
)
from graphrag.sna.projection import DEFAULT_LAMBDA
from graphrag.sna.topodist import TopologicalDistances, compare_topology, render_topology

__all__ = [
    "Comparison",
    "RankChange",
    "Window",
    "compare_payload",
    "compare_windows",
    "render_comparison",
]

#: How many rank movers the report names.
TOP_CHANGES = 10
#: Below this many shared nodes, a similarity score is noise and the report says so.
MIN_SHARED = 10
#: Accepted on any network, directed or not: the total degree these rank tables have always
#: used. ``measures.centrality`` documents what it is on a digraph and the caption repeats it.
CENTRALITY_ALWAYS = ("degree", "weighted_degree")


@dataclass(frozen=True)
class Window:
    """One window's graph and what was found in it."""

    since: str
    until: str
    graph: nx.Graph
    summary: dict[str, float]
    where: str = ""
    louvain_result: LouvainResult | None = None

    @property
    def label(self) -> str:
        """What this build was cut to, in words: its window, its attribute filter, or both."""
        window = ""
        if self.since and self.until:
            window = f"{self.since} to {self.until}"
        elif self.since:
            window = f"{self.since} onward"
        elif self.until:
            window = f"up to {self.until}"
        if window and self.where:
            return f"{window}, {self.where}"
        return window or self.where or "no window"

    @property
    def communities(self) -> list[list[str]]:
        return self.louvain_result.communities if self.louvain_result else []


@dataclass(frozen=True)
class RankChange:
    """One node's movement in a centrality ranking between the two windows."""

    node: str
    label: str
    rank_a: int
    rank_b: int

    @property
    def change(self) -> int:
        """Positive means it climbed: a smaller rank number in the second window."""
        return self.rank_a - self.rank_b


@dataclass(frozen=True)
class Comparison:
    """Everything ``graphrag sna compare`` produced."""

    persona_id: str
    network: str
    centrality: str
    a: Window
    b: Window
    shared: tuple[str, ...] = ()
    entered: tuple[str, ...] = ()
    left: tuple[str, ...] = ()
    similarity: dict[str, float] = field(default_factory=dict)
    changes: tuple[RankChange, ...] = ()
    topology: TopologicalDistances | None = None
    seed: int | None = None
    generated_at: str = ""
    notes: tuple[str, ...] = ()


def _membership(communities: Sequence[Sequence[str]], nodes: Sequence[str]) -> list[int]:
    """A label per node, ``-1`` for a node no community claims."""
    index = {node: i for i, group in enumerate(communities) for node in group}
    return [index.get(node, -1) for node in nodes]


def _ranks(graph: nx.Graph, kind: str, nodes: Sequence[str]) -> dict[str, int]:
    """Rank 1 is the highest score. Ties break on the node id, as everywhere else here."""
    scores = centrality(graph, kind)
    order = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranking = {node: position for position, (node, _) in enumerate(order, start=1)}
    return {node: ranking[node] for node in nodes if node in ranking}


def _label(graph: nx.Graph, node: str) -> str:
    data = graph.nodes.get(node, {})
    return str(data.get("label") or data.get("name") or node)


def compare_windows(
    store: GraphStore,
    persona_id: str,
    network: str,
    *,
    since: str | None = None,
    until: str | None = None,
    since2: str | None = None,
    until2: str | None = None,
    window: str | int | None = None,
    step: str | int | None = None,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
    where2: Mapping[str, str] | None = None,
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
    project: Project | None = None,
    centrality_kind: str = "weighted_degree",
    resolution: float = 1.0,
    runs: int = 10,
    seed: int | None = None,
    topology: bool = False,
) -> Comparison:
    """Build one network twice, and compare the two results.

    The two builds usually differ by their window, which is what the module is named for. They
    may differ by their attribute filter instead: ``where`` and ``where2`` build the network
    over two populations of the same corpus -- one value of an attribute against another -- and
    everything below reads the same way, because a population is a sampling frame exactly as a
    window is. Two attribute values are not two measurements of one thing any more than two
    windows are, which is why n comes first and the partitions are compared only over the nodes
    both builds hold.

    ``backbone`` applies the same chapter 27 filter to both builds, and it has to be the same
    one: two backbones are two networks, so a comparison across them would be reporting the
    filter rather than the corpus. It runs inside :func:`graphrag.sna.layers.window_graph`, so a
    window here and a snapshot there are filtered identically. ``projection`` is the same story
    for chapter 26: both builds are weighted by the one scheme, because a weight of 3 under
    ``simple`` and one of 0.46 under ``hyperbolic`` are not two measurements of the same thing,
    and a comparison across them would report the scheme. Both builds' frames name it.

    ``window`` and ``step`` are the general form of the two date pairs (Atlas §7.4): they cut
    ``since``..``until`` into a grid of windows and compare the **first against the last**, which
    is what a before-and-after over a dated corpus is. Every window in between is named in the
    notes rather than silently dropped -- a grid of nine windows compared at its ends is a
    comparison of two of them. Use :func:`graphrag.sna.layers.snapshots` when the whole sequence
    is the question.

    ``topology`` runs :func:`graphrag.sna.topodist.compare_topology` (ch. 48, ATL-48) over the
    two builds' whole graphs and adds it as :attr:`Comparison.topology`. It needs no shared nodes
    at all -- unlike every section above, it is not about the nodes both builds hold -- and is
    skipped, with a note, when either build has fewer than two nodes, which is the smallest a
    spectral distance can read a Laplacian from.
    """
    grid: list[tuple[str, str]] = []
    if window is not None:
        if since2 is not None or until2 is not None:
            msg = "--window builds both date pairs, so it cannot be given with --since2/--until2"
            raise ValueError(msg)
        if not since or not until:
            msg = "--window needs --since and --until to bound the grid it cuts"
            raise ValueError(msg)
        grid = windows(since, until, window, step)
        if len(grid) < 2:
            msg = (
                f"a window of {window} over {since}..{until} makes one window; a before-and-after "
                "needs two -- widen the range or narrow --window"
            )
            raise ValueError(msg)
        (since, until), (since2, until2) = grid[0], grid[-1]

    def build(start: str | None, end: str | None, filters: Mapping[str, str] | None) -> Window:
        graph = window_graph(
            store,
            persona_id,
            network,
            since=start,
            until=end,
            source_id=source_id,
            min_weight=min_weight,
            types=types,
            stances=stances,
            facets=facets,
            where=filters,
            projection=projection,
            lam=lam,
            backbone=backbone,
            alpha=alpha,
            correction=correction,
            threshold=threshold,
            project=project,
        )
        result = (
            louvain(graph, resolution=resolution, seed=seed, runs=runs)
            if graph.number_of_edges()
            else None
        )
        return Window(
            since=start or "",
            until=end or "",
            where=where_text(filters),
            graph=graph,
            summary=summary(graph),
            louvain_result=result,
        )

    first = build(since, until, where)
    # Checked against the first build rather than against a fixed list: which centralities a
    # network has is a property of the network, and on a directed one in_degree and out_degree
    # are two of them (§6.2). Both builds are the same network kind, so one of them settles it.
    allowed = centralities_for(first.graph)
    if centrality_kind not in allowed and centrality_kind not in CENTRALITY_ALWAYS:
        msg = f"centrality must be one of {', '.join(allowed)}, got {centrality_kind!r}"
        raise ValueError(msg)
    second = build(since2, until2, where2 if where2 is not None else where)
    nodes_a, nodes_b = set(first.graph.nodes), set(second.graph.nodes)
    shared = sorted(nodes_a & nodes_b)

    similarity: dict[str, float] = {}
    if shared and first.communities and second.communities:
        similarity = compare_partitions(
            _membership(first.communities, shared), _membership(second.communities, shared)
        )

    ranks_a = _ranks(first.graph, centrality_kind, shared)
    ranks_b = _ranks(second.graph, centrality_kind, shared)
    changes = [
        RankChange(
            node=node,
            label=_label(second.graph, node),
            rank_a=ranks_a[node],
            rank_b=ranks_b[node],
        )
        for node in shared
        if node in ranks_a and node in ranks_b
    ]
    changes.sort(key=lambda c: (-abs(c.change), c.node))

    topology_result: TopologicalDistances | None = None
    topology_note = ""
    if topology:
        if first.graph.number_of_nodes() < 2 or second.graph.number_of_nodes() < 2:
            topology_note = (
                "--topology was asked for but skipped: a spectral distance needs at least two "
                "nodes on each side, and at least one build has fewer than that."
            )
        else:
            topology_result = compare_topology(first.graph, second.graph)

    return Comparison(
        persona_id=persona_id,
        network=network,
        centrality=centrality_kind,
        a=first,
        b=second,
        shared=tuple(shared),
        entered=tuple(sorted(nodes_b - nodes_a)),
        left=tuple(sorted(nodes_a - nodes_b)),
        similarity=similarity,
        changes=tuple(changes[:TOP_CHANGES]),
        topology=topology_result,
        seed=seed,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=(*_notes(first, second, shared, grid), *([topology_note] if topology_note else ())),
    )


def _notes(
    a: Window, b: Window, shared: Sequence[str], grid: Sequence[tuple[str, str]] = ()
) -> tuple[str, ...]:
    """The things that make this particular comparison weaker than it looks.

    A directed network says here what it says in an analysis report: which measures could not
    read the direction, which summary numbers changed definition with it, and that Louvain
    flattened the graph before partitioning it. Both builds produce the same sentences, so they
    are said once.
    """
    notes: list[str] = list(directed_notes(a.graph))
    if len(grid) > 2:
        labels = ", ".join(f"{opens} to {closes}" for opens, closes in grid[1:-1])
        notes.append(
            f"The window grid holds {len(grid)} windows and only its ends are compared here. "
            f"Not compared: {labels}. A history is not a before-and-after (Atlas §7.4); read the "
            "whole sequence with sna layers or snapshots() before treating these two as a trend."
        )
    for window in (a, b):
        note = window.louvain_result.note if window.louvain_result else ""
        if note and note not in notes:
            notes.append(note)
    for window in (a, b):
        nodes = int(window.summary["nodes"])
        if nodes == 0:
            why = (
                "nothing carries those attribute values, or the tagging pass has not covered "
                "the documents that would"
                if window.where
                else "nothing was dated into it, or the attribution pass has not run on the "
                "documents that fall in it"
            )
            notes.append(f"The build {window.label} is empty. Either {why}.")
        elif nodes < 30:
            notes.append(
                f"The build {window.label} holds {nodes} nodes, which is small enough that its "
                "communities and rankings describe those nodes and estimate nothing."
            )
    if 0 < len(shared) < MIN_SHARED:
        notes.append(
            f"Only {len(shared)} node(s) appear in both windows, so the similarity scores are "
            "computed over almost nothing and the rank changes are arithmetic, not findings."
        )
    if not shared:
        notes.append(
            "The two builds share no nodes at all, so there is nothing to compare: report them "
            "as two separate networks rather than as a change. Two attribute values that share "
            "no nodes are the expected case, not a failure -- the comparison is then the two "
            "node counts and the two structures, side by side."
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


def _listing(graph: nx.Graph, nodes: Sequence[str], limit: int = 20) -> str:
    if not nodes:
        return "none"
    shown = [_label(graph, node) for node in nodes[:limit]]
    rest = len(nodes) - len(shown)
    return ", ".join(shown) + (f", +{rest} more" if rest else "")


def render_comparison(comparison: Comparison) -> str:
    """The comparison as a markdown note."""
    a, b = comparison.a, comparison.b
    seed = f", seed {comparison.seed}." if comparison.seed is not None else ", no seed fixed."
    lines = [
        f"# Network comparison: {comparison.persona_id} / {comparison.network}",
        "",
        f"Generated {comparison.generated_at} by graphrag {__version__}{seed}",
        "",
        f"**Network type.** {describe(a.graph).sentence}",
        "",
        f"**Sampling frame.** {a.graph.graph.get('frame', '')}",
        "",
        *(
            [f"**Sampling frame, {b.label}.** {b.graph.graph.get('frame', '')}", ""]
            if b.graph.graph.get("frame") != a.graph.graph.get("frame")
            else []
        ),
        f"**What is being compared.** Two builds of the same network: {a.label} against "
        f"{b.label}. Community numbers are not comparable between two Louvain runs, so the "
        "partitions are compared by how much they agree on the nodes both builds contain, and "
        "the rankings are compared over those same nodes only.",
        "",
        "## Each build",
        "",
    ]
    lines += _table(
        ["build", "nodes", "edges", "components", "communities", "modularity"],
        [
            [
                window.label,
                _num(window.summary["nodes"]),
                _num(window.summary["edges"]),
                _num(window.summary["components"]),
                str(len(window.communities)) if window.communities else "-",
                _num(window.louvain_result.modularity) if window.louvain_result else "-",
            ]
            for window in (a, b)
        ],
    )

    if comparison.topology is not None:
        lines += render_topology(comparison.topology)

    lines += [
        "## Who entered and who left",
        "",
        f"- in both windows: {len(comparison.shared):,}",
        f"- only in {b.label}: {len(comparison.entered):,} — "
        f"{_listing(b.graph, comparison.entered)}",
        f"- only in {a.label}: {len(comparison.left):,} — {_listing(a.graph, comparison.left)}",
        "",
        "A node absent from a build was not necessarily quiet in it: it is absent when nothing "
        "it appears in carries a date inside the window, or a value the attribute filter asked "
        "for. Absence is a property of the tagging and the dating, before it is anything else.",
        "",
    ]

    lines += ["## Partition similarity", ""]
    if comparison.similarity:
        lines += [
            f"- adjusted Rand index: "
            f"{_num(comparison.similarity['adjusted_rand_index'])} — chance-corrected, so 0 is "
            "what two unrelated partitions score and 1 is identical",
            f"- normalised mutual information: "
            f"{_num(comparison.similarity['normalized_mutual_information'])} — not "
            "chance-corrected, and drifts up as the number of communities grows, so read it "
            "beside the index above and never instead of it",
            f"- computed over the {len(comparison.shared):,} node(s) in both builds",
            "",
        ]
    else:
        lines += [
            "Not computed: one of the builds has no communities, or the two share no nodes.",
            "",
        ]

    lines += [
        f"## Largest rank changes ({comparison.centrality})",
        "",
        centrality_meaning(comparison.centrality, a.graph),
        "",
        "Rank 1 is the highest score. Ranks are computed inside each build, so a climb can mean "
        "the node rose or that the nodes above it left.",
        "",
    ]
    if comparison.changes:
        lines += _table(
            ["node", f"rank in {a.label}", f"rank in {b.label}", "change"],
            [
                [
                    change.label,
                    str(change.rank_a),
                    str(change.rank_b),
                    f"+{change.change}" if change.change > 0 else str(change.change),
                ]
                for change in comparison.changes
            ],
        )
    else:
        lines += ["No node appears in both builds with a rank in each.", ""]

    lines += ["## Caveats", ""]
    lines += [f"- {item}" for item in ALWAYS]
    lines += [f"- {note}" for note in comparison.notes]
    return "\n".join(lines).rstrip() + "\n"


def _window_payload(w: Window) -> dict[str, Any]:
    return {
        "since": w.since,
        "until": w.until,
        "where": w.where,
        "label": w.label,
        "summary": w.summary,
    }


def compare_payload(comparison: Comparison) -> dict[str, Any]:
    """``comparison`` as the JSON `sna compare` writes with ``--json`` and the ``sna_compare``
    MCP tool returns as ``payload`` -- the same :class:`Comparison` fields :func:`render_comparison`
    reads, field for field (ATL-F1: one function both surfaces call, instead of each building
    this dict inline). Each :class:`Window`'s graph itself is not included -- it is not
    JSON-serialisable and every number about it is already in ``summary``."""
    topo = comparison.topology
    return {
        "persona_id": comparison.persona_id,
        "network": comparison.network,
        "centrality": comparison.centrality,
        "a": _window_payload(comparison.a),
        "b": _window_payload(comparison.b),
        "shared": list(comparison.shared),
        "entered": list(comparison.entered),
        "left": list(comparison.left),
        "similarity": comparison.similarity,
        "changes": [
            {
                "node": c.node,
                "label": c.label,
                "rank_a": c.rank_a,
                "rank_b": c.rank_b,
                "change": c.change,
            }
            for c in comparison.changes
        ],
        "topology": None
        if topo is None
        else {
            "spectral_distance": topo.spectral.distance,
            "netsimile_distance": topo.netsimile.distance,
            "delta_con_similarity": topo.delta_con.similarity,
            "portrait_divergence": topo.portrait.divergence,
        },
        "seed": comparison.seed,
        "generated_at": comparison.generated_at,
        "notes": list(comparison.notes),
    }
