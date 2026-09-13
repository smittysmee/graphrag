"""The same network over two time windows, and an honest account of what changed.

A before-and-after is the easiest analysis to get wrong. Two Louvain partitions of two graphs
carry community numbers that mean nothing across runs, so "community 2 grew" is meaningless;
what can be compared is how much the *partitioning* agrees on the nodes both windows contain,
which is what the adjusted Rand index and normalised mutual information measure. And a node
that rose twenty places in a centrality ranking may simply have stayed put while half the
network left, which is why the node sets are reported before the ranks are.

So this module reports, in this order: n for each window, who entered and who left, how far the
two partitions agree over the nodes in common, and the largest rank changes among those nodes
only. Nothing is compared across the whole of both windows, because the two whole networks are
not two measurements of one thing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import networkx as nx

from graphrag import __version__
from graphrag.graph.store import GraphStore
from graphrag.sna.cluster import LouvainResult, compare_partitions, louvain
from graphrag.sna.export import Project, build_network
from graphrag.sna.guide import ALWAYS
from graphrag.sna.measures import CENTRALITIES, centrality, summary

__all__ = [
    "Comparison",
    "RankChange",
    "Window",
    "compare_windows",
    "render_comparison",
]

#: How many rank movers the report names.
TOP_CHANGES = 10
#: Below this many shared nodes, a similarity score is noise and the report says so.
MIN_SHARED = 10


@dataclass(frozen=True)
class Window:
    """One window's graph and what was found in it."""

    since: str
    until: str
    graph: nx.Graph
    summary: dict[str, float]
    louvain_result: LouvainResult | None = None

    @property
    def label(self) -> str:
        if self.since and self.until:
            return f"{self.since} to {self.until}"
        if self.since:
            return f"{self.since} onward"
        if self.until:
            return f"up to {self.until}"
        return "no window"

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
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    project: Project | None = None,
    centrality_kind: str = "weighted_degree",
    resolution: float = 1.0,
    runs: int = 10,
    seed: int | None = None,
) -> Comparison:
    """Build one network twice, on two windows, and compare the two results."""
    if centrality_kind not in CENTRALITIES:
        msg = f"centrality must be one of {', '.join(CENTRALITIES)}, got {centrality_kind!r}"
        raise ValueError(msg)

    def window(start: str | None, end: str | None) -> Window:
        graph = build_network(
            store,
            network,
            persona_id,
            source_id=source_id,
            min_weight=min_weight,
            types=types,
            stances=stances,
            facets=facets,
            since=start,
            until=end,
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
            graph=graph,
            summary=summary(graph),
            louvain_result=result,
        )

    first, second = window(since, until), window(since2, until2)
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
        seed=seed,
        generated_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        notes=_notes(first, second, shared),
    )


def _notes(a: Window, b: Window, shared: Sequence[str]) -> tuple[str, ...]:
    """The things that make this particular comparison weaker than it looks."""
    notes: list[str] = []
    for window in (a, b):
        nodes = int(window.summary["nodes"])
        if nodes == 0:
            notes.append(
                f"The window {window.label} is empty. Either nothing was dated into it, or the "
                "attribution pass has not run on the documents that fall in it."
            )
        elif nodes < 30:
            notes.append(
                f"The window {window.label} holds {nodes} nodes, which is small enough that its "
                "communities and rankings describe those nodes and estimate nothing."
            )
    if 0 < len(shared) < MIN_SHARED:
        notes.append(
            f"Only {len(shared)} node(s) appear in both windows, so the similarity scores are "
            "computed over almost nothing and the rank changes are arithmetic, not findings."
        )
    if not shared:
        notes.append(
            "The two windows share no nodes at all, so there is nothing to compare: report them "
            "as two separate networks rather than as a change."
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
        f"**Sampling frame.** {a.graph.graph.get('frame', '')}",
        "",
        "**What is being compared.** Two builds of the same network over two windows. Community "
        "numbers are not comparable between two Louvain runs, so the partitions are compared by "
        "how much they agree on the nodes both windows contain, and the rankings are compared "
        "over those same nodes only.",
        "",
        "## Each window",
        "",
    ]
    lines += _table(
        ["window", "nodes", "edges", "components", "communities", "modularity"],
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

    lines += [
        "## Who entered and who left",
        "",
        f"- in both windows: {len(comparison.shared):,}",
        f"- only in {b.label}: {len(comparison.entered):,} — "
        f"{_listing(b.graph, comparison.entered)}",
        f"- only in {a.label}: {len(comparison.left):,} — {_listing(a.graph, comparison.left)}",
        "",
        "A node absent from a window was not necessarily quiet in it: it is absent when nothing "
        "it appears in carries a date inside the window.",
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
            f"- computed over the {len(comparison.shared):,} node(s) in both windows",
            "",
        ]
    else:
        lines += [
            "Not computed: one of the windows has no communities, or the windows share no nodes.",
            "",
        ]

    lines += [
        f"## Largest rank changes ({comparison.centrality})",
        "",
        "Rank 1 is the highest score. Ranks are computed inside each window, so a climb can mean "
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
        lines += ["No node appears in both windows with a rank in each.", ""]

    lines += ["## Caveats", ""]
    lines += [f"- {item}" for item in ALWAYS]
    lines += [f"- {note}" for note in comparison.notes]
    return "\n".join(lines).rstrip() + "\n"
