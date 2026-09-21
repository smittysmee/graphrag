"""Node visual attributes, edge visual attributes and network layouts -- Atlas ch. 49-51.

*"[N]odes aren't dots, and edges aren't lines. The dot-line diagram is a map, not the territory"*
(p. 713). This module turns a network into that map: coordinates (ch. 51), a radius and a fill
per node (ch. 49), a width and a colour per edge (ch. 50), written as a self-contained SVG and an
HTML page that adds hover labels, drag and a legend with no external script.

**What "self-contained" buys and costs.** Nothing here fetches a font, a colour table or a
layout engine from anywhere; the SVG is built as plain strings and the HTML embeds it verbatim.
The price is everything Cytoscape and Gephi do that a hand-rolled renderer does not: real edge
bundling (Holten 2006; Holten and Van Wijk 2009), organic/orthogonal edge routing against ghost
edges (§51.3, Dwyer et al. 2006), hive plots and graph thumbnails (§51.4). Also not built, from
the same subsection: probabilistic layout (§51.4, needs repeated force layouts over a sample and
a spatial-probability contour per node -- the data-level version of "approximate a large network"
is what `sna sample`'s real samplers already do); revealing matrices (§51.4, one bipartite
sub-network per cell, a matrix-of-networks drawing rather than one network's matrix); and
timelines (§51.4, needs per-edge timestamps at a granularity a passage-level corpus does not
carry -- a passage can name several co-occurring edges at once, not one edge per moment). Those
stay documented rather than built; see the module's "not built" note beside each one below.

**The book's cautions, made mechanical rather than left as style advice.**

* Node size encodes *area*, never radius (§49.1, Figure 49.8): a value that doubles must not
  quadruple what the eye perceives, so :func:`size_scale` places the (quasi-logged) value
  linearly in area and takes the square root for the radius.
* No more than nine colours, node and edge together (§49.2 p.719, §50.1 p.729): the categorical
  palette stops at eight named, colour-blind-safe hues (Okabe and Ito, 2008 -- the book's own
  suggestion, Color Brewer, is a paletted lookup table this package does not vendor; Okabe-Ito is
  the same idea in eight hex codes) and a ninth, neutral grey absorbs everything else.
* No rainbow scale for a quantity (§49.2 p.721, Figure 49.12-49.13): :func:`sequential_palette`
  is one hue, monotone in lightness only, so a colour-blind reader loses nothing a sighted one
  has.
* A hairball is a layout failure, not evidence the network has no structure (Figure 49.2, ch. 51
  intro): every layout function's docstring says what it is for and what a bad-looking result
  under it should make a reader try next, not conclude.
* A long straight edge under a force layout can look like it threads through an unrelated node
  -- a *ghost edge* (§51.3, Figure 51.9) -- purely because that node's position is a coincidence
  of the physics, not a claim about a path. Nothing here routes edges around nodes to avoid this
  (the book's own fix needs orthogonal/organic routing this module does not implement); the guide
  rule below says why to read the edge list before trusting a crossing.

These are :data:`graphrag.sna.guide.DRAW_RULES`, quoted verbatim by ``docs/SNA.md`` and the SNA
skill, not restated here.

**§49.3's other node features, not built.** Node borders as a second encoding (colour plus
thickness): this module already spends node size and node colour on one quantitative and one
qualitative attribute, and a third visual channel risks exactly the overload the chapter warns
against on its own next page (p.725) -- nothing in `sna draw`'s flags names a third attribute to
put there. The xenographic node-shape distinction (circle vs square for a bipartite network's two
modes, or a symbol per type): needs an icon or a second drawn shape per node kind, which this
hand-rolled SVG renderer does not have and every node it draws is a circle. The alpha-channel
invisible-node trick (Figure 49.17): useful only for a network whose nodes carry nothing worth
naming so the edges alone can be read; every node this package draws is exactly what a report
already names, so hiding it would remove information rather than add legibility.

**Network lifting (§50.3).** The chapter's own order for building up a readable picture --
1. edge transparency, 2. edge width, 3. edge colour, 4. node size, 5. node colour -- is not a
sequence of flags here because there is only one drawing to make, not five: :func:`draw` applies
a fixed edge transparency (:data:`DEFAULT_EDGE_ALPHA`), scales edge width from the edge weight
(or, paired with ``--size betweenness``, from edge betweenness -- the chapter's own pairing,
p.727: *"This works well when used in conjunction with nodes sizes following the same
semantics"*), colours the edges on the same measure that set their width (Figure 50.3(a):
*"the same quantitative attribute for both thickness and colour, so that the two can work in
concert"*), then applies ``--size`` and ``--color`` to the nodes last, on top of the edges,
exactly the chapter's stated order.

**§51.5 case studies are documented, not built.** The Product Space's stretched, uncircularised
force layout and the Cathedral's two-level functional scatter are bespoke, hand-tuned pictures
the book itself calls "custom... no one should really follow your workflow" (p.751); building a
generic "stretch this force layout along an axis" flag would either force that axis onto every
corpus network or do nothing useful for the ones that have none. `docs/SNA.md` documents the
recipe -- `sna draw --layout force --seed <n> --json out.json` and then reading `positions` back
against `sna analyze`'s own centrality columns -- rather than a flag that assumes the shape.
"""

from __future__ import annotations

import colorsys
import math
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from html import escape as _esc
from typing import Any

import networkx as nx

from graphrag.sna.cluster import louvain
from graphrag.sna.evaluate import evaluate_partition, render_evaluation
from graphrag.sna.export import attr_key, describe
from graphrag.sna.hierarchy import Dummy, layered_layout
from graphrag.sna.hierarchy import Layout as HierarchyLayout
from graphrag.sna.matrices import Node
from graphrag.sna.measures import centrality

LAYOUTS: tuple[str, ...] = ("force", "circular", "arc", "matrix", "layered")

CANVAS_WIDTH = 960.0
CANVAS_HEIGHT = 720.0
PADDING = 32.0
BEND_PX = 18.0
"""How far a reciprocal directed pair's two arcs are pushed apart (§50.2, Figure 50.5(a))."""

MIN_RADIUS = 6.0
MAX_RADIUS = 26.0
DEFAULT_RADIUS = 9.0

MIN_EDGE_WIDTH = 0.6
MAX_EDGE_WIDTH = 5.0
DEFAULT_EDGE_WIDTH = 1.2

#: Fixed regardless of any attribute (§50.1, Figure 50.4(b)): "even if I added literally zero
#: bits of information by removing some edge opacity" a fixed transparency still declutters a
#: dense network by letting genuinely overlapping edges read as darker.
DEFAULT_EDGE_ALPHA = 0.45

DEFAULT_NODE_COLOR = "#4C72B0"
DEFAULT_EDGE_COLOR = "#8C8C8C"
#: The neutral colour for a category beyond the top eight, or a node with no value at all.
OTHER_CATEGORY_COLOR = "#999999"

#: Okabe and Ito's (2008) eight colour-blind-safe categorical hues, standing in for the Color
#: Brewer lookup table the book recommends (p.720-721) and this package does not vendor. The
#: book's own worked example (Figure 49.11) uses nine; the ninth slot here is reserved for
#: "other" (:data:`OTHER_CATEGORY_COLOR`) rather than an eighth named hue, so the total a viewer
#: has to tell apart never exceeds nine (§49.2 p.719) no matter how many categories exist.
CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)
MAX_CATEGORIES = len(CATEGORICAL_PALETTE)


# --------------------------------------------------------------------------------- §49.1 size


def _normalize(values: Mapping[Node, float]) -> dict[Node, float]:
    """``values`` onto ``[0, 1]``, monotone, after the quasi-logarithmic taming §49.1 asks for.

    Every step -- the shift to non-negative, ``log1p``, and the min-max scaling -- is monotone
    non-decreasing, so a value that was larger before this function runs is never smaller after
    it: the property :func:`size_scale` and :func:`sequential_colors` both depend on.
    """
    if not values:
        return {}
    numbers = list(values.values())
    if min(numbers) == max(numbers):
        return dict.fromkeys(values, 0.5)
    floor = min(0.0, min(numbers))
    transformed = {node: math.log1p(value - floor) for node, value in values.items()}
    lo, hi = min(transformed.values()), max(transformed.values())
    return {node: (t - lo) / (hi - lo) for node, t in transformed.items()}


def size_scale(
    values: Mapping[Node, float],
    *,
    min_radius: float = MIN_RADIUS,
    max_radius: float = MAX_RADIUS,
) -> dict[Node, float]:
    """Map a centrality onto node radius so *area*, not radius, is linear in the value (§49.1).

    A viewer perceives area. A radius that grows linearly with the value quadruples its apparent
    size for every doubling (Figure 49.8: *"double degree, but four times as large!"*). This
    places the (quasi-logged, via :func:`_normalize`) value linearly between the two radii'
    *areas* and takes the square root, so the drawn quantity is the honest one. The node with the
    largest value always gets the largest radius, monotonically, and a network where every value
    ties -- including one node -- gets the mid-point radius throughout: there is no "largest" to
    draw larger.
    """
    if not values:
        return {}
    normalized = _normalize(values)
    min_area, max_area = min_radius * min_radius, max_radius * max_radius
    return {node: math.sqrt(min_area + t * (max_area - min_area)) for node, t in normalized.items()}


def edge_width_scale(
    values: Mapping[Any, float],
    *,
    min_width: float = MIN_EDGE_WIDTH,
    max_width: float = MAX_EDGE_WIDTH,
) -> dict[Any, float]:
    """The edge equivalent of :func:`size_scale`, linear rather than square-rooted (§50.1).

    Lines are one-dimensional, so the area correction that node size needs does not apply here:
    *"you shouldn't worry too much about the square area problem... it will start to be a
    problem only if your edges are so large that your eyes start interpreting lines as
    rectangles"* (p.727) -- not a risk at these widths.
    """
    if not values:
        return {}
    normalized = _normalize(values)
    return {edge: min_width + t * (max_width - min_width) for edge, t in normalized.items()}


# -------------------------------------------------------------------------------- §49.2 colour


def _hex(r: float, g: float, b: float) -> str:
    def channel(value: float) -> str:
        return f"{round(max(0.0, min(1.0, value)) * 255):02x}"

    return f"#{channel(r)}{channel(g)}{channel(b)}"


def sequential_palette(
    n: int,
    *,
    hue: float = 0.58,
    light_range: tuple[float, float] = (0.88, 0.25),
    saturation: float = 0.62,
) -> tuple[str, ...]:
    """``n`` colours of one hue, strictly monotone in lightness (§49.2's "intensity gradient").

    One hue rather than a hue sweep: a rainbow scale is not perceptually uniform -- equal steps
    in hue do not read as equal steps in the eye (Figure 49.12-49.13, p.721-722) -- and a reader
    who cannot distinguish hues can still read lightness. ``light_range`` runs light-to-dark by
    default (low value = pale, high value = dark), the zero-to-maximum reading of Figure 49.9
    (bottom); the diverging, meaningful-midpoint gradient the same figure shows (top) is not
    built here, because nothing this package reads has a midpoint that is not itself a choice a
    caller would have to supply -- see the module docstring.
    """
    if n <= 0:
        return ()
    hi, lo = light_range
    lightness = [(hi + lo) / 2.0] if n == 1 else [hi + (lo - hi) * i / (n - 1) for i in range(n)]
    return tuple(_hex(*colorsys.hls_to_rgb(hue, level, saturation)) for level in lightness)


def sequential_colors(values: Mapping[Node, float], *, steps: int = 16) -> dict[Node, str]:
    """A numeric measure read onto :func:`sequential_palette`, quasi-log normalised first."""
    if not values:
        return {}
    palette = sequential_palette(steps)
    normalized = _normalize(values)
    return {
        node: palette[min(len(palette) - 1, int(t * len(palette)))]
        for node, t in normalized.items()
    }


@dataclass(frozen=True)
class CategoryColors:
    """One colour per node from a label, with the book's nine-colour ceiling enforced."""

    color_of: dict[Node, str]
    legend: tuple[tuple[str, str], ...]
    """``(label, colour)``, largest category first; ``"other (n more)"`` last when truncated."""
    collapsed: int
    """How many distinct label *values* were folded into "other"."""
    unlabelled: int
    """How many nodes carried no label at all."""


def categorical_colors(
    labels: Mapping[Node, Hashable], *, all_nodes: Sequence[Node]
) -> CategoryColors:
    """Up to :data:`MAX_CATEGORIES` labels get their own colour; the rest share "other" (§49.2).

    *"[N]ever use more than nine colours"* (p.719). The categories kept are the largest by node
    count, ties broken by their string form for a reproducible ordering, so the categories most
    worth telling apart are the ones a viewer can. A node the labels never mention is drawn in
    :data:`OTHER_CATEGORY_COLOR` too and counted in :attr:`CategoryColors.unlabelled`, exactly
    the way :func:`graphrag.sna.export.matches` treats a node with no value for a ``--where`` key:
    absent, not a group of its own.
    """
    counts: Counter[Hashable] = Counter(labels.values())
    ranked = sorted(counts, key=lambda label: (-counts[label], str(label)))
    kept = ranked[:MAX_CATEGORIES]
    palette = dict(zip(kept, CATEGORICAL_PALETTE, strict=False))
    color_of: dict[Node, str] = {}
    unlabelled = 0
    for node in all_nodes:
        label = labels.get(node)
        if label is None:
            unlabelled += 1
            color_of[node] = OTHER_CATEGORY_COLOR
        else:
            color_of[node] = palette.get(label, OTHER_CATEGORY_COLOR)
    legend = [(str(label), palette[label]) for label in kept]
    collapsed = len(ranked) - len(kept)
    if collapsed:
        legend.append((f"other ({collapsed} more)", OTHER_CATEGORY_COLOR))
    return CategoryColors(
        color_of=color_of, legend=tuple(legend), collapsed=collapsed, unlabelled=unlabelled
    )


# ------------------------------------------------------------------------------ §51.1-51.2 layouts


@dataclass(frozen=True)
class NodeLink:
    """Node coordinates for a node-link drawing, in an abstract plane before canvas scaling."""

    kind: str
    positions: dict[Node, tuple[float, float]]
    order: tuple[Node, ...]
    curved: bool
    """Whether edges under this layout are drawn as curves rather than straight lines."""
    note: str


def force_layout(graph: nx.Graph, *, seed: int | None = None) -> NodeLink:
    """Fruchterman-Reingold's spring model (§51.1): same-sign charges repel, edges are springs.

    Seeded, because the chapter's own argument for hive plots (§51.4) is that force-directed
    coordinates are otherwise not reproducible: *"a small change in the initial conditions... "
    will result in a completely different layout"* (p.745-746). ``seed=None`` still runs -- the
    layout is not refused -- but the report says the coordinates cannot be reproduced, because a
    layout whose seed was never recorded cannot be checked against its own report.

    Good for sparse to medium-sparse networks with real clusters (p.738-739); a dense network
    that still looks like a hairball under this layout is this layout's own cue to try
    ``--layout circular`` or ``--layout matrix`` next, not evidence the network has no structure.
    """
    weight = "weight" if nx.get_edge_attributes(graph, "weight") else None
    raw = nx.spring_layout(graph, seed=seed, weight=weight)
    positions = {node: (float(x), float(y)) for node, (x, y) in raw.items()}
    return NodeLink(
        kind="force",
        positions=positions,
        order=tuple(sorted(graph.nodes(), key=str)),
        curved=False,
        note=(
            f"force-directed (§51.1), seed {seed}"
            if seed is not None
            else "force-directed (§51.1), unseeded: these coordinates cannot be reproduced -- "
            "pass --seed to make them checkable."
        ),
    )


def circular_layout(graph: nx.Graph, *, order: Sequence[Node] | None = None) -> NodeLink:
    """``n`` nodes at equal angles around a circle, in the order given (§51.2).

    *"[T]he first part of the trick in using circular layouts is not to display the nodes in a
    random order"* (p.740): the order is the caller's decision -- communities or an attribute
    put nodes that share neighbours next to each other -- and this function only places them.
    With no order it falls back to descending degree, which at least keeps the best-connected
    nodes from scattering.
    """
    nodes = (
        tuple(order)
        if order is not None
        else tuple(sorted(graph.nodes(), key=lambda n: (-graph.degree(n), str(n))))
    )
    n = len(nodes)
    positions = {
        node: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) if n else (0.0, 0.0)
        for i, node in enumerate(nodes)
    }
    return NodeLink(
        kind="circular",
        positions=positions,
        order=nodes,
        curved=True,
        note=(
            "circular (§51.2): edges are bundled toward the centre so long chords do not "
            "obscure the ring; good for a network too dense for force-directed to pull apart."
        ),
    )


def arc_layout(graph: nx.Graph, *, order: Sequence[Node] | None = None) -> NodeLink:
    """Every node on one line, ordered; every edge drawn as an arc above it (§51.2's cousin).

    The book presents the arc diagram as circular layout's straight-line relative rather than
    under its own heading; it is offered here as ``--layout arc`` because it reads better than
    the circle for anything the caller has a genuine linear order for -- degree rank, a numeric
    attribute -- where a circle's wraparound would put the two ends artificially far apart.
    """
    nodes = (
        tuple(order)
        if order is not None
        else tuple(sorted(graph.nodes(), key=lambda n: (-graph.degree(n), str(n))))
    )
    positions = {node: (float(i), 0.0) for i, node in enumerate(nodes)}
    return NodeLink(
        kind="arc",
        positions=positions,
        order=nodes,
        curved=True,
        note="arc: every node on one line, ordered; every edge an arc above it.",
    )


@dataclass(frozen=True)
class MatrixCell:
    row: int
    col: int
    weight: float


@dataclass(frozen=True)
class MatrixLayout:
    """An adjacency-matrix drawing's data: an order and the cells an edge actually occupies."""

    order: tuple[Node, ...]
    cells: tuple[MatrixCell, ...]
    note: str


def matrix_layout(graph: nx.Graph, *, order: Sequence[Node] | None = None) -> MatrixLayout:
    """The network as an adjacency matrix (§51.4), ordered so structure becomes shape.

    *"[T]he order of the rows/columns you choose is the most important thing"* (p.745): with no
    order given, nodes are grouped by descending degree; a caller ordering by community or an
    attribute is what turns two disjoint cliques into two blocks on the diagonal rather than a
    scatter (Figure 51.11-51.12). Directed networks keep one cell per stated arc, ``(row=source,
    col=target)``; undirected ones fill both ``(i, j)`` and ``(j, i)`` so the matrix reads the
    same across the diagonal either way.
    """
    nodes = (
        tuple(order)
        if order is not None
        else tuple(sorted(graph.nodes(), key=lambda n: (-graph.degree(n), str(n))))
    )
    index = {node: i for i, node in enumerate(nodes)}
    cells: list[MatrixCell] = []
    for u, v, data in graph.edges(data=True):
        if u not in index or v not in index:
            continue
        weight = float(data.get("weight", 1.0))
        cells.append(MatrixCell(index[u], index[v], weight))
        if not graph.is_directed() and u != v:
            cells.append(MatrixCell(index[v], index[u], weight))
    return MatrixLayout(
        order=nodes,
        cells=tuple(cells),
        note=(
            "matrix (§51.4): rows and columns share the order given; a network with no "
            "structure worth the order looks like scatter, one with blocks of tightly connected "
            "nodes in that order looks block-diagonal."
        ),
    )


def _order_for(graph: nx.Graph, hint: Mapping[Node, Hashable] | None) -> tuple[Node, ...]:
    """The node order a circular, arc or matrix layout should use, from a colour rule if any.

    Grouped by label first -- largest group first, so the most populous category is not split
    across the wraparound of a circle -- then by degree within a group, both ties broken by the
    node's own string form so the order (and so the layout) is reproducible.
    """
    if hint is None:
        return tuple(sorted(graph.nodes(), key=lambda n: (-graph.degree(n), str(n))))
    counts: Counter[Hashable] = Counter(hint.values())

    def key(node: Node) -> tuple[int, int, str, int, str]:
        label = hint.get(node)
        if label is None:
            return (1, 0, "", -graph.degree(node), str(node))
        return (0, -counts[label], str(label), -graph.degree(node), str(node))

    return tuple(sorted(graph.nodes(), key=key))


# ----------------------------------------------------------------------- colour rule dispatch


@dataclass(frozen=True)
class ColorResult:
    color_of: dict[Node, str]
    legend: tuple[tuple[str, str], ...]
    rule: str
    notes: tuple[str, ...]
    order_by: dict[Node, Hashable] | None
    community_evaluation: tuple[str, ...] = ()


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _color_by_community(
    graph: nx.Graph, *, seed: int | None, resolution: float, runs: int
) -> ColorResult:
    result = louvain(graph, resolution=resolution, seed=seed, runs=runs)
    labels: dict[Node, Hashable] = {
        node: index for index, community in enumerate(result.communities) for node in community
    }
    colors = categorical_colors(labels, all_nodes=list(graph.nodes()))
    notes = [
        f"Louvain (§35.6): modularity {result.modularity:.4f}, stability {result.stability:.2f} "
        f"across {len(result.seeds)} seed(s) at resolution {result.resolution:g}.",
    ]
    if result.note:
        notes.append(result.note)
    if colors.collapsed:
        notes.append(
            f"{colors.collapsed} communities beyond the largest {MAX_CATEGORIES} share 'other' "
            "(§49.2's nine-colour ceiling)."
        )
    evaluation: tuple[str, ...] = ()
    if result.communities and graph.number_of_edges():
        scores = evaluate_partition(
            graph, result.communities, resolution=result.resolution, seed=seed
        )
        evaluation = tuple(render_evaluation(scores))
    return ColorResult(
        color_of=colors.color_of,
        legend=colors.legend,
        rule=f"community (Louvain, §35.6), {len(result.communities)} communities",
        notes=tuple(notes),
        order_by=labels,
        community_evaluation=evaluation,
    )


def _color_by_attribute(graph: nx.Graph, key: str) -> ColorResult:
    """§49.2: categorical for a qualitative attribute, sequential for a numeric one.

    A node the attribution/extraction sidecars never valued for ``key`` is drawn in the same
    neutral grey the categorical overflow uses and counted, rather than dropped from the picture:
    this is a drawing, and a network still has that node whether or not anyone tagged it.
    """
    field = attr_key(key)
    raw = {node: str(data[field]) for node, data in graph.nodes(data=True) if field in data}
    if not raw:
        msg = f"no node in this network carries an attribute {key!r}"
        raise ValueError(msg)
    missing = graph.number_of_nodes() - len(raw)
    notes: list[str] = []
    if missing:
        notes.append(f"{missing:,} node(s) carry no value for {key!r}; drawn in neutral grey.")
    if all(_is_number(value) for value in raw.values()):
        numeric = {node: float(value) for node, value in raw.items()}
        palette = sequential_palette(6)
        colored = sequential_colors(numeric, steps=len(palette))
        color_of = {node: colored.get(node, OTHER_CATEGORY_COLOR) for node in graph.nodes()}
        lo, hi = min(numeric.values()), max(numeric.values())
        legend = (
            (f"{key}: {lo:g} (low)", palette[0]),
            (f"{key}: {hi:g} (high)", palette[-1]),
        )
        notes.insert(0, f"`attr:{key}` read as numeric: a sequential, single-hue palette (§49.2).")
        return ColorResult(
            color_of=color_of,
            legend=legend,
            rule=f"attribute {key!r} (numeric)",
            notes=tuple(notes),
            order_by=numeric,  # type: ignore[arg-type]
        )
    colors = categorical_colors(raw, all_nodes=list(graph.nodes()))
    notes.insert(
        0, f"`attr:{key}` read as categorical: at most {MAX_CATEGORIES} colours plus 'other'."
    )
    if colors.collapsed:
        notes.append(f"{colors.collapsed} value(s) beyond the top {MAX_CATEGORIES} share 'other'.")
    return ColorResult(
        color_of=colors.color_of,
        legend=colors.legend,
        rule=f"attribute {key!r} (categorical)",
        notes=tuple(notes),
        order_by=raw,  # type: ignore[arg-type]
    )


def _resolve_color(graph: nx.Graph, color: str | None, *, seed: int | None) -> ColorResult:
    if color is None:
        return ColorResult(
            color_of=dict.fromkeys(graph.nodes(), DEFAULT_NODE_COLOR),
            legend=(),
            rule="none (uniform fill)",
            notes=(),
            order_by=None,
        )
    if color == "community":
        return _color_by_community(graph, seed=seed, resolution=1.0, runs=10)
    if color.startswith("attr:"):
        key = color.removeprefix("attr:")
        if not key:
            msg = "--color attr:<key> needs a key after the colon"
            raise ValueError(msg)
        return _color_by_attribute(graph, key)
    msg = f"--color must be 'community' or 'attr:<key>', not {color!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------------- SVG / HTML


def _svg_header(width: float, height: float, *, directed: bool) -> list[str]:
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.1f} {height:.1f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="sans-serif">'
    ]
    if directed:
        lines.append(
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#666666" /></marker></defs>'
        )
    return lines


def _transform_for(
    positions: Mapping[Any, tuple[float, float]],
    radii: Mapping[Any, float],
    *,
    width: float,
    height: float,
    padding: float,
) -> Callable[[float, float], tuple[float, float]]:
    """An affine data-space -> canvas-space map that keeps every node's circle on the canvas."""
    if not positions:
        return lambda x, y: (width / 2.0, height / 2.0)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    pad = padding + (max(radii.values()) if radii else DEFAULT_RADIUS)
    span_x, span_y = max(max_x - min_x, 1e-9), max(max_y - min_y, 1e-9)
    scale = max(min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y), 1e-9)
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    def transform(x: float, y: float) -> tuple[float, float]:
        return (width / 2.0 + (x - cx) * scale, height / 2.0 + (y - cy) * scale)

    return transform


def _canvas_positions(
    positions: Mapping[Node, tuple[float, float]],
    radii: Mapping[Node, float],
    *,
    width: float,
    height: float,
    padding: float,
) -> dict[Node, tuple[float, float]]:
    transform = _transform_for(positions, radii, width=width, height=height, padding=padding)
    return {node: transform(x, y) for node, (x, y) in positions.items()}


def _render_nodes(
    canvas: Mapping[Node, tuple[float, float]],
    radii: Mapping[Node, float],
    colors: Mapping[Node, str],
    ids: Mapping[Node, str],
) -> str:
    parts = ['<g class="nodes">']
    for node, (x, y) in canvas.items():
        r = radii.get(node, DEFAULT_RADIUS)
        fill = colors.get(node, DEFAULT_NODE_COLOR)
        parts.append(
            f'<g class="node" data-id="{ids[node]}" transform="translate({x:.2f},{y:.2f})">'
            f'<circle r="{r:.2f}" fill="{fill}" stroke="#ffffff" stroke-width="1">'
            f"<title>{_esc(str(node))}</title></circle></g>"
        )
    parts.append("</g>")
    return "".join(parts)


def _straight_or_offset_edge(
    u: Node,
    v: Node,
    canvas: Mapping[Node, tuple[float, float]],
    ids: Mapping[Node, str],
    width_px: float,
    stroke: str,
    *,
    bend: float,
    directed: bool,
) -> str:
    x1, y1 = canvas[u]
    x2, y2 = canvas[v]
    marker = ' marker-end="url(#arrow)"' if directed else ""
    if bend == 0.0:
        return (
            f'<line class="edge" data-source="{ids[u]}" data-target="{ids[v]}" '
            f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width_px:.2f}" '
            f'stroke-opacity="{DEFAULT_EDGE_ALPHA}"{marker} />'
        )
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx_, ny_ = -dy / length, dx / length
    mx, my = (x1 + x2) / 2 + nx_ * bend, (y1 + y2) / 2 + ny_ * bend
    return (
        f'<path class="edge-curve" d="M{x1:.2f},{y1:.2f} Q{mx:.2f},{my:.2f} {x2:.2f},{y2:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{width_px:.2f}" '
        f'stroke-opacity="{DEFAULT_EDGE_ALPHA}"{marker} />'
    )


def _arc_edge(
    u: Node,
    v: Node,
    canvas: Mapping[Node, tuple[float, float]],
    width_px: float,
    stroke: str,
    *,
    directed: bool,
) -> str:
    x1, y1 = canvas[u]
    x2, y2 = canvas[v]
    mx = (x1 + x2) / 2
    bow = min(abs(x2 - x1) * 0.5, 180.0)
    cy = min(y1, y2) - bow
    marker = ' marker-end="url(#arrow)"' if directed else ""
    return (
        f'<path class="edge-curve" d="M{x1:.2f},{y1:.2f} Q{mx:.2f},{cy:.2f} {x2:.2f},{y2:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{width_px:.2f}" '
        f'stroke-opacity="{DEFAULT_EDGE_ALPHA}"{marker} />'
    )


def _bundle_edge(
    u: Node,
    v: Node,
    canvas: Mapping[Node, tuple[float, float]],
    width_px: float,
    stroke: str,
    *,
    center: tuple[float, float],
    directed: bool,
) -> str:
    x1, y1 = canvas[u]
    x2, y2 = canvas[v]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    tx, ty = center
    cx, cy = mx + (tx - mx) * 0.35, my + (ty - my) * 0.35
    marker = ' marker-end="url(#arrow)"' if directed else ""
    return (
        f'<path class="edge-curve" d="M{x1:.2f},{y1:.2f} Q{cx:.2f},{cy:.2f} {x2:.2f},{y2:.2f}" '
        f'fill="none" stroke="{stroke}" stroke-width="{width_px:.2f}" '
        f'stroke-opacity="{DEFAULT_EDGE_ALPHA}"{marker} />'
    )


def _reciprocal_pairs(graph: nx.Graph) -> set[tuple[Node, Node]]:
    pairs: set[tuple[Node, Node]] = set()
    for u, v in graph.edges():
        if u != v and graph.has_edge(v, u):
            pairs.add((u, v) if str(u) < str(v) else (v, u))
    return pairs


def _render_nodelink_svg(
    graph: nx.Graph,
    canvas: Mapping[Node, tuple[float, float]],
    radii: Mapping[Node, float],
    colors: Mapping[Node, str],
    edge_widths: Mapping[tuple[Node, Node], float],
    edge_colors: Mapping[tuple[Node, Node], str],
    ids: Mapping[Node, str],
    *,
    width: float,
    height: float,
    layout: str,
) -> str:
    directed = bool(graph.is_directed())
    reciprocal = _reciprocal_pairs(graph) if directed else set()
    lines = _svg_header(width, height, directed=directed)
    lines.append('<g class="edges">')
    for u, v, _data in graph.edges(data=True):
        if u == v or u not in canvas or v not in canvas:
            continue
        width_px = edge_widths.get((u, v), DEFAULT_EDGE_WIDTH)
        stroke = edge_colors.get((u, v), DEFAULT_EDGE_COLOR)
        if layout == "arc":
            lines.append(_arc_edge(u, v, canvas, width_px, stroke, directed=directed))
        elif layout == "circular":
            lines.append(
                _bundle_edge(
                    u,
                    v,
                    canvas,
                    width_px,
                    stroke,
                    center=(width / 2, height / 2),
                    directed=directed,
                )
            )
        else:
            canon = (u, v) if str(u) < str(v) else (v, u)
            bend = 0.0
            if directed and canon in reciprocal:
                bend = BEND_PX if (u, v) == canon else -BEND_PX
            lines.append(
                _straight_or_offset_edge(
                    u, v, canvas, ids, width_px, stroke, bend=bend, directed=directed
                )
            )
    lines.append("</g>")
    lines.append(_render_nodes(canvas, radii, colors, ids))
    lines.append("</svg>")
    return "\n".join(lines)


def _render_matrix_svg(
    matrix: MatrixLayout, colors: Mapping[Node, str], *, width: float, height: float, directed: bool
) -> str:
    n = len(matrix.order)
    lines = _svg_header(width, height, directed=directed)
    if n == 0:
        lines.append("</svg>")
        return "\n".join(lines)
    margin = 90.0
    size_px = max(2.0, min(width - margin, height - margin) / n)
    max_weight = max((cell.weight for cell in matrix.cells), default=1.0) or 1.0
    palette = sequential_palette(16)
    lines.append(f'<g transform="translate({margin:.1f},{margin:.1f})">')
    for i, node in enumerate(matrix.order):
        fill = colors.get(node, DEFAULT_NODE_COLOR)
        lines.append(
            f'<rect x="-14" y="{i * size_px:.2f}" width="10" height="{size_px:.2f}" fill="{fill}">'
            f"<title>{_esc(str(node))}</title></rect>"
        )
        lines.append(
            f'<rect x="{i * size_px:.2f}" y="-14" width="{size_px:.2f}" height="10" fill="{fill}">'
            f"<title>{_esc(str(node))}</title></rect>"
        )
    for cell in matrix.cells:
        intensity = cell.weight / max_weight
        fill = palette[min(len(palette) - 1, int(intensity * len(palette)))]
        source, target = matrix.order[cell.row], matrix.order[cell.col]
        lines.append(
            f'<rect class="cell" x="{cell.col * size_px:.2f}" y="{cell.row * size_px:.2f}" '
            f'width="{size_px:.2f}" height="{size_px:.2f}" fill="{fill}">'
            f"<title>{_esc(str(source))} -- {_esc(str(target))}: {cell.weight:g}</title></rect>"
        )
    lines.append(
        f'<rect x="0" y="0" width="{n * size_px:.2f}" height="{n * size_px:.2f}" '
        'fill="none" stroke="#333333" stroke-width="1" />'
    )
    lines.append("</g></svg>")
    return "\n".join(lines)


def _render_layered_svg(
    graph: nx.DiGraph,
    hlayout: HierarchyLayout,
    radii: Mapping[Node, float],
    colors: Mapping[Node, str],
    edge_widths: Mapping[tuple[Node, Node], float],
    edge_colors: Mapping[tuple[Node, Node], str],
    ids: Mapping[Node, str],
    *,
    width: float,
    height: float,
) -> str:
    """§33.6's layering, drawn: straight arcs within a layer gap, polylines through dummies.

    An arc §33.6 could not send downward is dashed (:data:`ReadingRule` "Layers (§33.6)" in the
    guide already explains why; this only marks it) rather than left indistinguishable from a
    normal, downward one.
    """
    transform = _transform_for(
        hlayout.positions, radii, width=width, height=height, padding=PADDING
    )
    canvas = {node: transform(x, y) for node, (x, y) in hlayout.positions.items()}
    layer_of = {node: -y for node, (_, y) in hlayout.positions.items()}
    by_edge: dict[tuple[Node, Node], list[Dummy]] = {}
    for dummy in hlayout.dummies:
        by_edge.setdefault((dummy.source, dummy.target), []).append(dummy)
    lines = _svg_header(width, height, directed=True)
    lines.append('<g class="edges">')
    for u, v, _data in graph.edges(data=True):
        if u == v or u not in canvas or v not in canvas:
            continue
        width_px = edge_widths.get((u, v), DEFAULT_EDGE_WIDTH)
        stroke = edge_colors.get((u, v), DEFAULT_EDGE_COLOR)
        upward = layer_of.get(v, 0.0) <= layer_of.get(u, 0.0)
        dash = ' stroke-dasharray="6,4"' if upward else ""
        waypoints = by_edge.get((u, v))
        if waypoints:
            ordered = sorted(waypoints, key=lambda d: d.layer)
            points = [canvas[u], *(transform(d.x, d.y) for d in ordered), canvas[v]]
            path = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points)
            lines.append(
                f'<path class="edge-curve" d="{path}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width_px:.2f}" stroke-opacity="{DEFAULT_EDGE_ALPHA}"{dash} '
                'marker-end="url(#arrow)" />'
            )
        else:
            x1, y1 = canvas[u]
            x2, y2 = canvas[v]
            lines.append(
                f'<line class="edge" data-source="{ids[u]}" data-target="{ids[v]}" '
                f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{stroke}" stroke-width="{width_px:.2f}" '
                f'stroke-opacity="{DEFAULT_EDGE_ALPHA}"{dash} marker-end="url(#arrow)" />'
            )
    lines.append("</g>")
    lines.append(_render_nodes(canvas, radii, colors, ids))
    lines.append("</svg>")
    return "\n".join(lines)


def _wrap_html(
    svg: str,
    *,
    title: str,
    frame: str,
    legend: Sequence[tuple[str, str]],
    notes: Sequence[str],
) -> str:
    """A self-contained page: the SVG verbatim, hover labels via ``<title>``, drag, a legend.

    Nothing here is fetched: no ``<script src>``, no stylesheet link, no web font. Drag moves a
    node's own ``<g>`` and follows straight-line edges (``data-source``/``data-target``) live;
    the curved edges of ``--layout arc``/``circular``/the bundled portions of ``--layout
    layered`` do not re-route on drag, a limitation of building this without a routing library.
    """
    legend_html = "".join(
        f'<li><span class="swatch" style="background:{color}"></span>{_esc(label)}</li>'
        for label, color in legend
    )
    notes_html = "".join(f"<li>{_esc(note)}</li>" for note in notes)
    return (
        "<!doctype html>\n"
        f'<html><head><meta charset="utf-8"><title>{_esc(title)}</title>'
        "<style>"
        "body{font-family:sans-serif;margin:0;display:flex;gap:1.5rem;padding:1rem;"
        "align-items:flex-start;}"
        ".node{cursor:grab;} .node:active{cursor:grabbing;}"
        "ul.legend{list-style:none;padding:0;margin:0;}"
        "ul.legend li{display:flex;align-items:center;gap:.4rem;margin:.2rem 0;font-size:.85rem;}"
        ".swatch{width:12px;height:12px;border-radius:2px;display:inline-block;}"
        "ul.notes{font-size:.8rem;color:#444;max-width:26rem;padding-left:1rem;}"
        "p.frame{font-size:.8rem;color:#444;max-width:26rem;}"
        "</style></head><body>"
        f'<div class="canvas">{svg}</div>'
        "<aside>"
        f"<h2>{_esc(title)}</h2>"
        + (f'<p class="frame">{_esc(frame)}</p>' if frame else "")
        + (f'<ul class="legend">{legend_html}</ul>' if legend else "")
        + (f'<ul class="notes">{notes_html}</ul>' if notes else "")
        + "</aside>"
        "<script>\n"
        "(function () {\n"
        "  var svg = document.querySelector('svg');\n"
        "  if (!svg) { return; }\n"
        "  document.querySelectorAll('.node').forEach(function (g) {\n"
        "    g.addEventListener('pointerdown', function (ev) {\n"
        "      ev.preventDefault();\n"
        "      var id = g.getAttribute('data-id');\n"
        "      function move(e2) {\n"
        "        var pt = svg.createSVGPoint();\n"
        "        pt.x = e2.clientX; pt.y = e2.clientY;\n"
        "        var loc = pt.matrixTransform(svg.getScreenCTM().inverse());\n"
        "        g.setAttribute('transform', 'translate(' + loc.x + ',' + loc.y + ')');\n"
        "        document.querySelectorAll('[data-source=\"' + id + '\"]').forEach(function (e) {\n"
        "          e.setAttribute('x1', loc.x); e.setAttribute('y1', loc.y);\n"
        "        });\n"
        "        document.querySelectorAll('[data-target=\"' + id + '\"]').forEach(function (e) {\n"
        "          e.setAttribute('x2', loc.x); e.setAttribute('y2', loc.y);\n"
        "        });\n"
        "      }\n"
        "      function up() {\n"
        "        document.removeEventListener('pointermove', move);\n"
        "        document.removeEventListener('pointerup', up);\n"
        "      }\n"
        "      document.addEventListener('pointermove', move);\n"
        "      document.addEventListener('pointerup', up);\n"
        "    });\n"
        "  });\n"
        "})();\n"
        "</script>"
        "</body></html>"
    )


# --------------------------------------------------------------------------------- orchestration


@dataclass(frozen=True)
class DrawResult:
    layout: str
    svg: str
    html: str
    frame: str
    nodes: int
    edges: int
    size_rule: str
    color_rule: str
    legend: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    community_evaluation: tuple[str, ...]
    seed: int | None
    positions: dict[Node, tuple[float, float]]
    node_size: dict[Node, float]
    node_color: dict[Node, str]


def draw(
    graph: nx.Graph,
    *,
    layout: str = "force",
    size: str | None = None,
    color: str | None = None,
    seed: int | None = None,
    width: float = CANVAS_WIDTH,
    height: float = CANVAS_HEIGHT,
    title: str | None = None,
) -> DrawResult:
    """Lay out and style one network -- Atlas ch. 49 (nodes), ch. 50 (edges), ch. 51 (layout).

    ``title`` names the HTML page (its ``<title>`` and heading). It defaults to the layout and
    the node count; the sampling frame is printed in the page body under it rather than used as
    the name, since a frame is a paragraph and a browser tab shows a few words.

    ``layout`` is one of :data:`LAYOUTS`. ``size`` is a centrality name from
    :func:`graphrag.sna.measures.centralities_for` (raises the same message that function's
    caller would get for an unsupported one); ``None`` draws every node at
    :data:`DEFAULT_RADIUS`. ``color`` is ``"community"`` (Louvain, §35.6, imported rather than
    reimplemented) or ``"attr:<key>"`` (a node attribute written by an attribution/extraction
    sidecar, §7.5); ``None`` draws every node in :data:`DEFAULT_NODE_COLOR`.

    Edges are styled from the same measure, reinforcing rather than adding a second one (§50.1,
    Figure 50.3(a)): edge weight by default, or edge betweenness when ``size="betweenness"``,
    the chapter's own pairing (p.727). A fixed transparency (:data:`DEFAULT_EDGE_ALPHA`) applies
    regardless, the chapter's own "even with zero bits of information" trick (Figure 50.4(b)).

    Raises :class:`ValueError` on an empty network, an unknown ``layout``/``size``/``color``, or
    -- only for ``layout="layered"`` -- an undirected network, which
    :func:`graphrag.sna.hierarchy.layered_layout` refuses itself (§33.6 assumes direction
    throughout).
    """
    if layout not in LAYOUTS:
        msg = f"--layout must be one of {', '.join(LAYOUTS)}, not {layout!r}"
        raise ValueError(msg)
    if graph.number_of_nodes() == 0:
        msg = "cannot draw a network with no nodes"
        raise ValueError(msg)

    frame = str(graph.graph.get("frame", "")) or describe(graph).text
    ids = {node: f"n{i}" for i, node in enumerate(graph.nodes())}
    notes: list[str] = []

    color_result = _resolve_color(graph, color, seed=seed)
    notes.extend(color_result.notes)

    size_values: dict[Node, float] = {}
    size_rule = "none (uniform radius)"
    if size is not None:
        size_values = centrality(graph, size)
        size_rule = f"{size} (ch. 14, area-scaled per §49.1)"
    radii = size_scale(size_values) if size_values else dict.fromkeys(graph.nodes(), DEFAULT_RADIUS)

    if size == "betweenness" and graph.number_of_edges():
        edge_measure: dict[tuple[Node, Node], float] = dict(
            nx.edge_betweenness_centrality(graph, weight="weight").items()
        )
        edge_note = "edge betweenness (§50.1's pairing with node size)"
    else:
        edge_measure = {(u, v): float(d.get("weight", 1.0)) for u, v, d in graph.edges(data=True)}
        edge_note = "edge weight"
    edge_widths = edge_width_scale(edge_measure)
    edge_colors = sequential_colors(edge_measure)
    if edge_measure:
        notes.append(f"Edges sized and coloured by {edge_note}, both from the same values (§50.1).")

    order_hint = color_result.order_by

    if layout == "matrix":
        matrix = matrix_layout(graph, order=_order_for(graph, order_hint))
        svg = _render_matrix_svg(
            matrix,
            color_result.color_of,
            width=width,
            height=height,
            directed=bool(graph.is_directed()),
        )
        positions = {node: (float(i), 0.0) for i, node in enumerate(matrix.order)}
        notes.append(matrix.note)
    elif layout == "layered":
        hlayout = layered_layout(graph)
        svg = _render_layered_svg(
            graph,
            hlayout,
            radii,
            color_result.color_of,
            edge_widths,
            edge_colors,
            ids,
            width=width,
            height=height,
        )
        positions = hlayout.positions
        notes.append(hlayout.note)
    else:
        node_link = {
            "force": force_layout(graph, seed=seed),
            "circular": circular_layout(graph, order=_order_for(graph, order_hint)),
            "arc": arc_layout(graph, order=_order_for(graph, order_hint)),
        }[layout]
        canvas = _canvas_positions(
            node_link.positions, radii, width=width, height=height, padding=PADDING
        )
        svg = _render_nodelink_svg(
            graph,
            canvas,
            radii,
            color_result.color_of,
            edge_widths,
            edge_colors,
            ids,
            width=width,
            height=height,
            layout=layout,
        )
        positions = node_link.positions
        notes.append(node_link.note)

    page_title = title or f"{layout} drawing, {graph.number_of_nodes():,} nodes"
    html = _wrap_html(svg, title=page_title, frame=frame, legend=color_result.legend, notes=notes)

    return DrawResult(
        layout=layout,
        svg=svg,
        html=html,
        frame=frame,
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        size_rule=size_rule,
        color_rule=color_result.rule,
        legend=color_result.legend,
        notes=tuple(notes),
        community_evaluation=color_result.community_evaluation,
        seed=seed,
        positions=positions,
        node_size=size_values,
        node_color=color_result.color_of,
    )


def render_draw(result: DrawResult) -> list[str]:
    """The ``## Visualization`` section printed beside the SVG/HTML files."""
    lines = [
        "## Visualization",
        "",
        "**Implements.** Atlas ch. 49 (node visual attributes), ch. 50 (edge visual attributes), "
        "ch. 51 (network layouts).",
        "",
        f"**Frame.** {result.frame}. n = {result.nodes:,} nodes, {result.edges:,} edges.",
        f"**Layout.** `{result.layout}`"
        + (f", seed {result.seed}" if result.seed is not None else " (unseeded)"),
        f"**Size.** {result.size_rule}",
        f"**Colour.** {result.color_rule}",
    ]
    if result.legend:
        lines += ["", "| category | colour |", "|---|---|"]
        lines += [f"| {label} | `{color}` |" for label, color in result.legend]
    if result.notes:
        lines += ["", "**Notes.**"]
        lines += [f"- {note}" for note in result.notes]
    if result.community_evaluation:
        lines += ["", *result.community_evaluation]
    return lines


def draw_payload(result: DrawResult) -> dict[str, Any]:
    """The JSON twin of :func:`render_draw`: positions included, for the §51.5 recipes."""
    return {
        "layout": result.layout,
        "frame": result.frame,
        "nodes": result.nodes,
        "edges": result.edges,
        "size_rule": result.size_rule,
        "color_rule": result.color_rule,
        "seed": result.seed,
        "legend": [{"label": label, "color": color} for label, color in result.legend],
        "positions": {str(node): list(pos) for node, pos in result.positions.items()},
        "node_size": {str(node): value for node, value in result.node_size.items()},
        "node_color": {str(node): color for node, color in result.node_color.items()},
        "notes": list(result.notes),
    }
