"""Chapter 7's extended graphs over a corpus: layers, hyperedges and time.

The five networks in :mod:`graphrag.sna.export` are each one simple graph. Chapter 7 says that is
often not what the data are, and gives three ways out. This module builds all three from the same
store, and keeps the book's vocabulary rather than inventing one.

**Multilayer (§7.2).** An edge can carry a *type* as well as a weight: ``(u, v, l)`` where ``l``
is qualitative. Every layer is a simple graph over the same nodes, which is the book's *multiplex*
case -- our node mapping is one-to-one, because an entity is the same entity whichever stance was
annotated on the passage that named it. :func:`multilayer` builds one layer per value of a
*layering* (``stance``, ``facet``, ``source`` or ``relation-type``); :func:`flatten` sums them
into the single weighted graph the rest of the package can measure (§40.1); :func:`supra_adjacency`
gives §8.1's supra-adjacency matrix, the layers' own adjacencies down the diagonal and ``omega``
times the identity in every off-diagonal block, which is the interlayer coupling drawn as a matrix.

Three things about that are easy to get wrong and are therefore stated next to the numbers.
Flattening is not free: §40.1 defines it as "collapsing a qualitative information -- in which
layer a connection appears -- into a quantitative one -- an edge weight", and says outright that
this "assumes that every edge type is equally important". Nothing in a corpus makes praise and
complaint equally important, so the assumption is made visibly, in the frame, and the per-layer
table is printed above whatever was measured. The layers also do not have to sum to the unlayered
network: a facet layering partitions *passages*, so it does, while a stance layering partitions
*mentions*, so a pair named once approvingly and once disapprovingly in the same passage is in no
layer at all. And ``omega`` is a parameter somebody chose, not something the corpus states (§7.2,
"Many-to-Many").

**Hypergraphs (§7.3).** A passage naming four entities relates all four at once; splitting it into
six pairs is a modelling choice, not the data. :func:`hypergraph` keeps the passage as a hyperedge
with four members, and :func:`clique_expansion` performs the first of the two simplification
strategies §7.3 gives -- turn the hyperedge into the simple edges it stands for; Figure 7.8 draws
that one and the bipartite one side by side. The book stops there. That the expansion is
*numerically identical* to this package's bipartite projection of the same memberships is this
package's own claim, not the section's, which is why it is asserted in the tests on the fixture
and on the Southern women data rather than merely cited. The second strategy, the bipartite
network, is what ``--network speakers-entities`` already is.

**Dynamic (§7.4).** :func:`dynamic_edges` is the edge-level form -- ``(u, v, t)``, one row per
occurrence -- and :func:`snapshots` is the static-view form, cutting that history into windows.
§7.4 lists four ways to cut it and stresses that they produce radically different histories from
identical activation times, so :func:`windows` takes the width, the step and whether the windows
accumulate, and every snapshot says which it was. A document the attribution pass could not date
is in no window at all; the count of those is printed rather than left out.

**What chapter 7 describes and this module deliberately does not build.** Each one is a decision,
not an omission, so each is written down with the reason:

- *Many-to-many interlayer coupling, the set ``C`` of ``G = (V, E, L, C)`` (§7.2).* Our node
  mapping is one-to-one -- an entity is the same entity whichever stance annotated the passage
  that named it -- so the couplings are the identity and there is nothing for ``C`` to hold.
- *Actors (§7.2).* An actor is the connected component of the coupling edges alone. With a
  one-to-one mapping every component is a single node, so the term would name nothing.
- *The clique, chain and star coupling flavours (§7.2, Figure 7.6).* They are ways of drawing and
  of cheapening ``C`` when there are hundreds of layers; with an identity coupling and a handful
  of layers all three coincide.
- *Aspects (§7.2).* A second aspect -- layers at ``t`` and at ``t + 1`` -- is
  :func:`multilayer` called inside a window. Building the product explicitly multiplies the layer
  count and has no consumer until multilayer community discovery (ch. 40).
- *Signed networks (§7.2).* They already exist as the annotation layer:
  :mod:`graphrag.sna.stances` and ``--stance`` are the positive and negative relationships, and
  structural balance is the book's own forward reference to §24.1.
- *Simplicial complexes (§7.3).* §7.3 defers them itself -- "we're going to look at them
  extensively in Chapter 34" -- and so does this module.
- *The uniform-hypergraph constraint (§7.3).* A uniform hypergraph fixes every hyperedge's size,
  the way a football team has eleven players. Nothing about a passage fixes how many entities it
  names, so the constraint could only ever be violated.
- *§7.3's second simplification strategy, the bipartite conversion.* It is already
  ``--network speakers-entities``, and it is what :attr:`Hypergraph.incidence` is. Rebuilding it
  here would be a second copy of an existing network.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations
from typing import Any, Literal, cast

import networkx as nx
import numpy as np

from graphrag.graph.store import GraphStore
from graphrag.models import Document, EntityMention, RelationRow
from graphrag.sna import matrices
from graphrag.sna.backbone import DEFAULT_ALPHA
from graphrag.sna.export import (
    NETWORKS,
    STANCES,
    Network,
    Project,
    build_network,
    entity_passage_rows,
    matches,
    resolve,
)
from graphrag.sna.projection import DEFAULT_LAMBDA
from graphrag.sna.stats import Summary, describe

__all__ = [
    "DynamicEdges",
    "Hypergraph",
    "LayerSummary",
    "Membership",
    "Multilayer",
    "TimedEdge",
    "build_hypergraph",
    "clique_expansion",
    "document_dates",
    "dynamic_edges",
    "flatten",
    "hyperedge_sizes",
    "hypergraph",
    "layer_table",
    "multilayer",
    "render_layers",
    "snapshots",
    "supra_adjacency",
    "window_graph",
    "windows",
]

Layering = Literal["stance", "facet", "source", "relation-type"]
LAYERINGS: tuple[Layering, ...] = ("stance", "facet", "source", "relation-type")

#: Which key in ``graph.graph`` a layering occupies, so the flattened graph can blank it: a layer
#: was built with one value of that filter, and the flattening holds every value.
LAYER_META: dict[Layering, str] = {
    "stance": "stances",
    "facet": "facets",
    "source": "source_id",
    "relation-type": "relation_types",
}

#: Enough to cover any persona's documents or any document's passages; the store pages.
_ALL = 1_000_000
#: An ISO date no post can exceed. Asking the store for a window that spans everything is how it
#: is asked for the *dated* documents alone, since a windowed read drops anything undated.
_ANY_DATE = "9999-12-31"
#: A window grid longer than this is a mistake in the width or the step, not an analysis.
MAX_WINDOWS = 500
#: The default interlayer coupling. 1.0 says "a node in one layer is exactly as tied to itself in
#: another as a unit edge ties two nodes", which is a convention and nothing more.
DEFAULT_OMEGA = 1.0

MULTILAYER_FRAME = (
    "Read as a multilayer network (Atlas §7.2): the same network built once per {layering}, over "
    "one node set, so an edge carries a type -- the layer that stated it -- beside its weight. "
    "Layers ({count}): {names}. Every caution about the unlayered network holds inside every "
    "layer, and each layer is additionally as small as its own value is rare. The node mapping "
    "is one-to-one (the book's multiplex case) and the interlayer coupling omega={omega} is a "
    "parameter chosen here, not something the corpus states."
)
FLATTENED_FRAME = (
    "Flattened (Atlas §7.2, §40.1): the layers summed into one weighted graph, so every measure "
    "below is blind to which layer an edge came from. §40.1 puts the cost plainly -- flattening "
    "is 'collapsing a qualitative information -- in which layer a connection appears -- into a "
    "quantitative one -- an edge weight', and it 'assumes that every edge type is equally "
    "important'. Nothing about this corpus says its layers are. Each edge records the layers "
    "that contributed and their weights, but no measure reads them; the per-layer table above "
    "is what says how the total divides."
)
HYPERGRAPH_FRAME = (
    "Read as a hypergraph (Atlas §7.3): one hyperedge per {edge_noun}, whose members are the "
    "{member_noun} it holds, so a {edge_noun} naming four of them is one relation between four "
    "and not six separate pairs. {edges} hyperedge(s) over {nodes} node(s). A clique expansion "
    "turns a hyperedge of size k into k(k-1)/2 edges, so one large {edge_noun} contributes more "
    "of the expanded network than many small ones do: read the hyperedge-size distribution "
    "before reading the expansion."
)
DYNAMIC_FRAME = (
    "Read as a dynamic network (Atlas §7.4): {occurrences} dated edge occurrence(s) between "
    "{first} and {last}, one per {unit} that put the two nodes together. {undated} document(s) "
    "carry no dated passage and are therefore in no window at all, taking {dropped} occurrence(s) "
    "with them. A date here is the earliest dated passage of the document the edge came from, "
    "because that is the only date the attribution pass writes."
)
SNAPSHOT_FRAME = (
    "One snapshot of a dynamic network (Atlas §7.4): {mode}. Width {window}, advancing every "
    "{step}, covering {since} to {until}; this window is {label}, number {index} of {total}. "
    "§7.4 is explicit that the four ways of cutting a history -- single snapshot, disjoint, "
    "sliding and cumulative windows -- give radically different histories from identical "
    "activation times, so the shape below is a property of this choice as much as of the corpus. "
    "{clipping}{undated} document(s) carry no dated passage and are in none of these windows."
)
#: Appended to a snapshot's frame when the last window was cut short by the end of the range.
CLIPPED_FRAME = (
    "The last window is clipped at {until} rather than running past it, so it is shorter than "
    "the others and holds fewer edges for a reason that has nothing to do with the corpus. "
)
#: Appended when the step exceeds the width, which is none of §7.4's four strategies.
GAPPED_FRAME = (
    "The step is wider than the window, so the {gap} between one window and the next is "
    "discarded entirely: those edges are in no window of this sequence, and the totals across "
    "it do not add up to the corpus. "
)


# ----------------------------------------------------------------------------- §7.2 multilayer


@dataclass(frozen=True)
class Multilayer:
    """One network built once per layer, over a single node set (§7.2).

    ``G = (V, E, L)`` in the book's notation: ``names`` is ``L``, in the order the report prints
    them, ``graphs`` holds one simple graph per layer, and ``nodes`` is the union ``V``, sorted,
    which every layer is understood to share. A node absent from a layer has no edges there; it
    is still the same node, which is what makes the coupling one-to-one.

    ``omega`` is the interlayer coupling weight. It is a choice, recorded here so that any number
    computed from :func:`supra_adjacency` can be reported with the value it was computed at.
    """

    persona_id: str
    network: Network
    layering: Layering
    names: tuple[str, ...]
    graphs: Mapping[str, nx.Graph]
    nodes: tuple[str, ...]
    omega: float = DEFAULT_OMEGA
    frame: str = ""

    def __len__(self) -> int:
        return len(self.names)

    @property
    def directed(self) -> bool:
        """Whether the layers are directed, which they all are or none are."""
        return any(graph.is_directed() for graph in self.graphs.values())


@dataclass(frozen=True)
class LayerSummary:
    """One row of the per-layer table: how big this layer is and how much weight it carries."""

    name: str
    nodes: int
    edges: int
    weight: float
    isolated: int


def _layer_values(
    store: GraphStore,
    persona_id: str,
    layering: Layering,
    *,
    source_id: str | None,
    types: Sequence[str] | None,
    stances: Sequence[str] | None,
    facets: Sequence[str] | None,
    relation_types: Sequence[str] | None,
) -> list[str]:
    """Which values this layering has in the corpus, in the order the layers are printed.

    Only values something actually carries become layers: an empty layer is not a finding about
    the network, it is a value nobody used, and putting it in the table invites the reader to
    read its zero as an absence of ties rather than an absence of data.
    """
    if layering == "stance":
        rows = store.entity_mention_rows(persona_id, source_id, types)
        annotated = {str(row.stance) for row in rows if row.stance}
        wanted = {str(s) for s in (stances or STANCES)}
        return [s for s in STANCES if s in annotated and s in wanted]
    if layering == "facet":
        seen = {facet for row in store.chunk_facets(persona_id, source_id) for facet in row.facets}
    elif layering == "source":
        seen = {d.source_id for d in store.list_documents(persona_id, limit=_ALL)}
        return sorted({source_id} & seen) if source_id else sorted(seen)
    else:
        seen = {row.type for row in store.relation_rows(persona_id, source_id)}
    asked = facets if layering == "facet" else relation_types
    return sorted(seen & {v.strip() for v in asked}) if asked else sorted(seen)


def multilayer(
    store: GraphStore,
    persona_id: str,
    network: str = "entities",
    layers: Layering = "stance",
    *,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    project: Project | None = None,
    omega: float = DEFAULT_OMEGA,
) -> Multilayer:
    """Build one network once per value of ``layers`` and hold them as a multilayer network (§7.2).

    ``layers`` names the qualitative property that becomes the edge type ``l``:

    ``stance``
        one layer per stance the annotation layer wrote, so praise and complaint are two layers
        of one network rather than two unrelated networks.
    ``facet``
        one layer per annotated facet: what the passage was about.
    ``source``
        one layer per source of the persona, which is the closest thing a corpus has to the
        book's "one layer per platform".
    ``relation-type``
        one layer per stated relation type, on the ``relations`` network only. This is the
        multigraph §6.3 refused to fake: two entities related in two ways carry one edge per
        layer here instead of one folded edge.

    Every other filter narrows every layer identically and is passed straight through -- including
    ``projection``, chapter 26's weighting, so every layer is weighted the same way and the
    flattening sums like with like. A layer is the network the same call would have built with
    that one extra value set. ``min_weight``
    is therefore applied *inside* each layer: an edge too light to survive its own layer cannot
    reappear in the flattening, which is why the flattened graph of pruned layers is not the
    pruned unlayered graph. Leave it at 1 when the layers are meant to decompose a whole.

    Raises ``ValueError`` when the layering names no value the corpus carries -- an empty
    multilayer network is a missing annotation pass, not a result.
    """
    if network not in NETWORKS:
        msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
        raise ValueError(msg)
    if layers not in LAYERINGS:
        msg = f"--layers must be one of {', '.join(LAYERINGS)}, got {layers!r}"
        raise ValueError(msg)
    if layers == "relation-type" and network != "relations":
        msg = "--layers relation-type applies to --network relations, the only typed network"
        raise ValueError(msg)
    if layers == "stance" and network not in {"entities", "speakers-entities"}:
        msg = (
            "--layers stance reads the annotation on a mention, which only the entities and "
            "speakers-entities networks hold. Use --layers facet or --layers source."
        )
        raise ValueError(msg)

    values = _layer_values(
        store,
        persona_id,
        layers,
        source_id=source_id,
        types=types,
        stances=stances,
        facets=facets,
        relation_types=relation_types,
    )
    if not values:
        msg = (
            f"nothing in this corpus carries a {layers} value, so there are no layers to build. "
            "Run the pass that writes them, or choose another --layers."
        )
        raise ValueError(msg)

    graphs: dict[str, nx.Graph] = {}
    for value in values:
        graphs[value] = build_network(
            store,
            network,
            persona_id,
            source_id=value if layers == "source" else source_id,
            min_weight=min_weight,
            types=types,
            stances=[value] if layers == "stance" else stances,
            facets=[value] if layers == "facet" else facets,
            relation_types=[value] if layers == "relation-type" else relation_types,
            since=since,
            until=until,
            where=where,
            projection=projection,
            lam=lam,
            project=project,
        )
        graphs[value].graph.update(layer=value, layering=layers)
    nodes = tuple(sorted({node for graph in graphs.values() for node in graph.nodes}))
    base = str(graphs[values[0]].graph.get("frame", ""))
    frame = (
        base
        + " "
        + MULTILAYER_FRAME.format(
            layering=layers, count=len(values), names=", ".join(values), omega=omega
        )
    )
    return Multilayer(
        persona_id=persona_id,
        network=network,
        layering=layers,
        names=tuple(values),
        graphs=graphs,
        nodes=nodes,
        omega=omega,
        frame=frame,
    )


def flatten(ml: Multilayer) -> nx.Graph:
    """Sum the layers into one weighted graph, the book's *flattening* (§7.2, §40.1).

    §40.1 is where the operation is named and where its cost is stated: flattening is
    "collapsing a qualitative information -- in which layer a connection appears -- into a
    quantitative one -- an edge weight", and doing so "assumes that every edge type is equally
    important". That assumption is almost never true of a corpus -- a praise edge and a
    complaint edge are not interchangeable units -- so it is written into the frame rather than
    left implicit, and the per-layer table is printed above anything measured here.

    The weight of an edge is the sum of its weights across the layers that hold it, the weighted
    generalisation of §40.1's simplest scheme (which counts the layers an edge appears in; on
    binary layers the two coincide); the chapter's other choices (common neighbours per layer,
    differential flattening) are weighting schemes for community discovery and belong with it.
    Every node of every layer is present. Each edge additionally records ``layers`` and
    ``layer_weights`` -- which layers contributed and how much -- because dropping that silently
    is the whole hazard of flattening; but no measure in this package reads them, so the
    flattened graph really is blind to the type and the per-layer table has to be reported
    beside it.

    The result carries ``graph.graph["layers"]``, which is what makes
    :func:`graphrag.sna.export.describe` call it multilayer.

    Equality with the unlayered network is *not* guaranteed and is not meant to be. A layering
    that partitions the unit an edge is drawn from (facets partition passages) sums back to the
    whole; one that partitions something finer (stances partition mentions inside a passage)
    does not, because a pair whose two mentions carry different stances is in no layer.
    """
    first = ml.graphs[ml.names[0]]
    graph: nx.Graph = nx.DiGraph() if first.is_directed() else nx.Graph()
    graph.add_nodes_from(ml.nodes)
    for name in ml.names:
        layer = ml.graphs[name]
        for node in layer.nodes:
            if not graph.nodes[node]:
                graph.nodes[node].update(layer.nodes[node])
        for u, v, data in layer.edges(data=True):
            weight = float(data.get("weight", 1))
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += weight
                graph[u][v]["layers"] += f"; {name}"
                graph[u][v]["layer_weights"] += f"; {_plain(weight)}"
            else:
                graph.add_edge(u, v, weight=weight, layers=name, layer_weights=str(_plain(weight)))
    for _, _, data in graph.edges(data=True):
        data["weight"] = _plain(data["weight"])
    graph.graph.update(first.graph)
    graph.graph.update(
        layer="",
        # The filter the layering owns held one value in the layer it was copied from; the
        # flattened graph holds all of them, so leaving it would misreport the whole as a slice.
        **{LAYER_META[ml.layering]: ""},
        layers=",".join(ml.names),
        layering=ml.layering,
        omega=ml.omega,
        frame=f"{ml.frame} {FLATTENED_FRAME}",
    )
    return graph


def _plain(weight: float) -> float | int:
    """A whole weight as an ``int``, so a count keeps printing and exporting as a count."""
    return int(weight) if float(weight).is_integer() else weight


def supra_adjacency(
    ml: Multilayer, omega: float | None = None
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """The supra-adjacency matrix of a multilayer network (§8.1), and its ``(node, layer)`` order.

    Each layer's own adjacency matrix goes in a diagonal block, in ``ml.names`` order and over
    ``ml.nodes`` in every block, so row ``i`` of block ``l`` always stands for the same node. The
    off-diagonal blocks hold the interlayer couplings: ``omega`` times the identity, which is the
    one-to-one ("multiplex") coupling of §7.2 -- node ``u`` in layer ``l`` is coupled to node
    ``u`` in layer ``l'`` and to nothing else. The result is ``(n·L, n·L)``.

    §8.1 offers two representations and sorts them the other way round: a *tensor* for the
    one-to-one coupled case, where "we are assuming that the nodes are sorted in the same way
    across the third dimension, thus the inter-layer couplings are implicit", and the supra
    adjacency matrix for many-to-many couplings, where they cannot be. Ours is the one-to-one
    case, so the book would reach for the tensor. The supra-adjacency is built instead because
    it is a matrix -- every solver in :mod:`graphrag.sna.matrices` and every method in ch. 40
    takes one -- and writing ``omega·I`` into its off-diagonal blocks is exactly the multiplex
    convention laid into §8.1's block form, with the coupling made explicit and adjustable
    rather than implicit at 1.

    ``omega`` defaults to the value the multilayer network was built with. It is a parameter, not
    a measurement: at 0 the layers are independent graphs stacked in one matrix, and as it grows
    every diffusion, walk or partition computed on the matrix is pulled toward treating the
    layers as one. Report the value beside any number read off this matrix.

    A node with no edges in a layer still has a row and a column there, all zero except its
    couplings. That is the price of the ``n·L`` form and it is the right price: the actor exists
    in the layer, it simply did nothing in it, and dropping the row would silently renumber every
    block.

    For a directed multilayer network the diagonal blocks are asymmetric, as they should be, while
    the couplings stay symmetric: "these two rows are the same actor" has no direction.
    """
    coupling = ml.omega if omega is None else omega
    count = len(ml.nodes)
    order = [(node, name) for name in ml.names for node in ml.nodes]
    size = count * len(ml.names)
    matrix = np.zeros((size, size), dtype=np.float64)
    for index, name in enumerate(ml.names):
        padded = ml.graphs[name].copy()
        padded.add_nodes_from(ml.nodes)
        block, _ = matrices.adjacency(padded, nodes=list(ml.nodes))
        start = index * count
        matrix[start : start + count, start : start + count] = matrices.dense(block)
    identity = np.eye(count, dtype=np.float64) * float(coupling)
    for row in range(len(ml.names)):
        for column in range(len(ml.names)):
            if row == column:
                continue
            matrix[row * count : (row + 1) * count, column * count : (column + 1) * count] = (
                identity
            )
    return matrix, order


def layer_table(ml: Multilayer) -> list[LayerSummary]:
    """One row per layer: its n, its m, the weight it carries and how many nodes it isolates.

    ``isolated`` counts the nodes of the multilayer network that hold no edge in this layer,
    which is the number a reader needs before comparing two layers: a layer over four nodes and
    a layer over forty are not two measurements of one thing, exactly as two windows are not.
    """
    rows: list[LayerSummary] = []
    for name in ml.names:
        graph = ml.graphs[name]
        joined = {node for edge in graph.edges for node in edge}
        rows.append(
            LayerSummary(
                name=name,
                nodes=graph.number_of_nodes(),
                edges=graph.number_of_edges(),
                weight=float(sum(w or 1 for _, _, w in graph.edges(data="weight"))),
                isolated=len(ml.nodes) - len(joined),
            )
        )
    return rows


def render_layers(ml: Multilayer) -> list[str]:
    """The per-layer table as markdown lines, for a report that ran on the flattened graph."""
    rows = layer_table(ml)
    matrix, order = supra_adjacency(ml)
    lines = [
        f"## Layers ({len(rows)})",
        "",
        "**Chapter.** Atlas §7.2 (multilayer graphs) and §8.1 (the supra-adjacency matrix).",
        "",
        f"**Sampling frame.** {ml.frame}",
        "",
        f"**n.** {len(ml.nodes):,} node(s) across {len(rows)} layer(s); the supra-adjacency "
        f"matrix is {matrix.shape[0]:,} by {matrix.shape[1]:,} over {len(order):,} "
        f"(node, layer) "
        f"pairs, at omega={ml.omega:g}.",
        "",
        "**Null model.** None here: this table describes what was built, it tests nothing. The "
        "null models belong to the measures below, which ran on the flattened graph.",
        "",
        "| layer | nodes | edges | total weight | nodes with no edge here |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row.name} | {row.nodes:,} | {row.edges:,} | {row.weight:,.0f} | {row.isolated:,} |"
        for row in rows
    ]
    lines += [
        "",
        "Everything below this table was computed on the flattened graph, so it cannot see which "
        "layer an edge came from. Read a finding against this table before attributing it to a "
        "layer.",
        "",
    ]
    return lines


def layers_payload(persona_id: str, network: str, layering: str, ml: Multilayer) -> dict[str, Any]:
    """``ml`` (built by :func:`multilayer`) as the JSON `sna layers` writes with ``--json`` and
    the ``sna_layers`` MCP tool returns as ``payload`` -- the same :func:`layer_table` rows,
    :func:`supra_adjacency` shape and :func:`flatten` size :func:`render_layers` reads (ATL-F1:
    one function both surfaces call, instead of each building this dict inline)."""
    rows = layer_table(ml)
    matrix, order = supra_adjacency(ml)
    flattened = flatten(ml)
    return {
        "persona_id": persona_id,
        "network": network,
        "layering": layering,
        "frame": ml.frame,
        "omega": ml.omega,
        "layers": [
            {
                "name": r.name,
                "nodes": r.nodes,
                "edges": r.edges,
                "weight": r.weight,
                "isolated": r.isolated,
            }
            for r in rows
        ],
        "supra_adjacency": {
            "shape": list(matrix.shape),
            "node_layer_pairs": len(order),
            "nodes": len(ml.nodes),
            "layers": len(ml),
        },
        "flattened": {
            "nodes": flattened.number_of_nodes(),
            "edges": flattened.number_of_edges(),
        },
    }


# ----------------------------------------------------------------------------- §7.3 hypergraph


@dataclass(frozen=True)
class Membership:
    """One group of nodes that a single passage or document put together.

    The thing every network in this package is projected from, named once so chapter 7 can read
    it three ways: as a hyperedge (§7.3), as one side of a bipartite network (§7.1), or as a
    dated occurrence (§7.4). ``ordered`` marks a membership whose two members are a subject and
    an object -- a stated relation -- so it must not be sorted into a set.
    """

    group: str
    doc_id: str
    members: tuple[str, ...]
    ordered: bool = False

    def pairs(self) -> list[tuple[str, str]]:
        """The simple edges this membership stands for: its clique, or its ordered pair."""
        if self.ordered:
            return [(self.members[0], self.members[1])] if len(self.members) == 2 else []
        return list(combinations(self.members, 2))


@dataclass(frozen=True)
class Hypergraph:
    """A set of hyperedges, each joining any number of nodes at once (§7.3).

    ``E`` is not a set of tuples any more: ``members[e]`` may hold two ids or twenty. The
    incidence matrix is the binary node-by-hyperedge matrix of §8.3, rows in ``nodes`` order and
    columns in ``hyperedges`` order, which is the same matrix the bipartite conversion -- the
    book's second simplification strategy -- would give.
    """

    persona_id: str
    member_noun: str
    edge_noun: str
    nodes: tuple[str, ...]
    hyperedges: tuple[str, ...]
    members: Mapping[str, tuple[str, ...]]
    incidence: np.ndarray
    labels: Mapping[str, str] = field(default_factory=dict)
    frame: str = ""

    @property
    def sizes(self) -> tuple[int, ...]:
        """How many nodes each hyperedge joins, in ``hyperedges`` order."""
        return tuple(len(self.members[edge]) for edge in self.hyperedges)


def build_hypergraph(
    memberships: Iterable[tuple[str, str]],
    *,
    persona_id: str = "",
    member_noun: str = "node",
    edge_noun: str = "hyperedge",
    labels: Mapping[str, str] | None = None,
) -> Hypergraph:
    """A hypergraph from ``(member, hyperedge)`` pairs (§7.3).

    The same input :func:`graphrag.sna.matrices.incidence` takes, and the incidence matrix is
    built by it, so the axes are sorted and the matrix is binary: a member listed twice in one
    hyperedge is one membership, because a hyperedge is a set.
    """
    pairs = sorted({(member, edge) for member, edge in memberships})
    matrix, nodes, edges = matrices.incidence(pairs)
    held: dict[str, list[str]] = {edge: [] for edge in edges}
    for member, edge in pairs:
        held[edge].append(member)
    frame = HYPERGRAPH_FRAME.format(
        edge_noun=edge_noun, member_noun=member_noun, edges=len(edges), nodes=len(nodes)
    )
    return Hypergraph(
        persona_id=persona_id,
        member_noun=member_noun,
        edge_noun=edge_noun,
        nodes=tuple(nodes),
        hyperedges=tuple(edges),
        members={edge: tuple(sorted(names)) for edge, names in held.items()},
        incidence=np.asarray(matrices.dense(matrix)),
        labels=dict(labels or {}),
        frame=frame,
    )


def hypergraph(
    store: GraphStore,
    persona_id: str,
    *,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
) -> Hypergraph:
    """The corpus as a hypergraph: one hyperedge per passage, its members the entities named (§7.3).

    This is the network ``--network entities`` is a projection of, before the projection. A
    passage naming Alpha, Beta and Gamma is *one* relation between three things here; the entity
    network turns it into three edges and can no longer tell that trio from three separate
    passages that each named a pair. Which of the two is right depends on the question, which is
    why both exist.

    Every filter means what it means everywhere else, because the rows come from the same reader
    the entity network uses.
    """
    rows = entity_passage_rows(
        store,
        persona_id,
        source_id,
        types,
        stances=stances,
        facets=facets,
        since=since,
        until=until,
        where=where,
    )
    return build_hypergraph(
        ((row.entity_id, row.chunk_id) for row in rows),
        persona_id=persona_id,
        member_noun="entities",
        edge_noun="passage",
        labels={row.entity_id: row.name for row in rows},
    )


def clique_expansion(hg: Hypergraph) -> nx.Graph:
    """Turn every hyperedge into the simple edges it stands for (§7.3, first strategy).

    A hyperedge over ``k`` nodes becomes the ``k(k-1)/2`` edges of a clique between them, and an
    edge's weight counts how many hyperedges join that pair. Computed as ``B @ Bᵀ`` with the
    diagonal zeroed (§8.3), which is :func:`graphrag.sna.matrices.project`.

    §7.3 gives the strategy and Figure 7.8 draws it; it does not go on to say that a clique
    expansion and a bipartite projection are the same arithmetic. They are, on a binary
    incidence matrix, and that is this package's assertion rather than the book's: it is what
    makes :func:`graphrag.sna.export.entity_co_mention` a clique expansion of the passages
    whether or not anybody called it one, and ``tests/unit/test_sna_layers.py`` asserts the two
    weight for weight on the fixture and on the Southern women data instead of taking it on
    trust.

    The cost of the strategy, in the book's words: the full membership is lost. Nothing in the
    expanded graph says those three edges came from one passage naming three things rather than
    from three passages naming two each, and a single large hyperedge contributes quadratically
    more of the result than a small one. :func:`hyperedge_sizes` is the distribution to read
    first.
    """
    product = matrices.dense(matrices.project(hg.incidence, side="left"))
    graph = nx.Graph()
    graph.add_nodes_from(hg.nodes)
    rows, columns = np.nonzero(np.triu(product, 1))
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        graph.add_edge(hg.nodes[row], hg.nodes[column], weight=_plain(float(product[row, column])))
    for index, node in enumerate(hg.nodes):
        graph.nodes[node]["label"] = hg.labels.get(node, node)
        graph.nodes[node]["mode"] = hg.member_noun
        graph.nodes[node]["hyperedges"] = int(hg.incidence[index].sum())
    graph.graph.update(
        network="entities",
        persona_id=hg.persona_id,
        unit=f"{hg.edge_noun}s shared",
        frame=f"{hg.frame} This is its clique expansion, one edge per pair of members.",
    )
    return graph


def hyperedge_sizes(hg: Hypergraph) -> Summary:
    """How many members the hyperedges hold, described per §3.1.

    The distribution that decides whether a clique expansion is a reasonable simplification: a
    corpus of two-member passages expands losslessly, while one heavy hyperedge dominates the
    expansion on its own. Undefined for a hypergraph with no hyperedges, which raises
    ``ValueError`` -- there is no typical size of nothing.
    """
    if not hg.hyperedges:
        msg = "a hypergraph with no hyperedges has no size distribution"
        raise ValueError(msg)
    return describe([float(size) for size in hg.sizes])


# ----------------------------------------------------------------------------- §7.4 dynamic


@dataclass(frozen=True)
class TimedEdge:
    """``(u, v, t)``: two nodes and when the corpus put them together (§7.4)."""

    source: str
    target: str
    at: str
    doc_id: str


@dataclass(frozen=True)
class DynamicEdges:
    """A network as its edge activations rather than as one static graph (§7.4).

    One row per occurrence, so the same pair appearing in three documents is three rows with
    three dates. That is the form §7.4 draws before it cuts anything into windows, and it is the
    only form in which "these two were together *then*" is a statement the data can make.
    """

    persona_id: str
    network: Network
    edges: tuple[TimedEdge, ...]
    undated_documents: int
    undated_edges: int
    frame: str = ""

    @property
    def dates(self) -> tuple[str, ...]:
        """Every distinct date, sorted."""
        return tuple(sorted({edge.at for edge in self.edges}))

    @property
    def span(self) -> tuple[str, str]:
        """The first and last date, or two empty strings when nothing is dated."""
        dates = self.dates
        return (dates[0], dates[-1]) if dates else ("", "")


def document_dates(
    store: GraphStore, persona_id: str, source_id: str | None = None
) -> dict[str, str]:
    """When each document was posted: the earliest dated passage it holds, ISO ``YYYY-MM-DD``.

    A document with no dated passage is absent from the mapping rather than present with a
    blank, because "undated" is not a date and every window in this package treats it as outside.
    The earliest rather than the latest, so a thread that ran for a week is dated when it
    started, which is what the corpus's own ``--since`` filter would have matched it on.

    One read per document, which is why the window machinery does *not* use this: a windowed
    read of the store answers "which documents fall inside" in one query, and only the
    edge-level view of §7.4 needs the timestamps themselves.
    """
    dates: dict[str, str] = {}
    for document in store.list_documents(persona_id, limit=_ALL):
        if source_id is not None and document.source_id != source_id:
            continue
        posted = [
            post.posted_at
            for chunk in store.document_chunks(document.id, 0, _ALL)
            for post in chunk.speaker_posts
            if post.posted_at
        ]
        if posted:
            dates[document.id] = min(posted)
    return dates


def _facet_passages(
    store: GraphStore, persona_id: str, source_id: str | None, facets: Sequence[str]
) -> tuple[set[str], set[str]]:
    """The passages carrying any of ``facets``, and the documents holding them."""
    wanted = {f.strip() for f in facets if f.strip()}
    rows = [r for r in store.chunk_facets(persona_id, source_id) if wanted & set(r.facets)]
    return {r.chunk_id for r in rows}, {r.doc_id for r in rows}


def _memberships(
    store: GraphStore,
    persona_id: str,
    network: Network,
    *,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
) -> list[Membership]:
    """What each network is projected from, as groups rather than as edges.

    One membership is one passage or one document, and the members are the nodes it put together
    -- which is a hyperedge (§7.3) and, once dated, a set of edge activations (§7.4). Reading the
    networks this way once means the two chapters do not each grow their own copy of the filters.
    """
    if network == "entities":
        rows = entity_passage_rows(
            store, persona_id, source_id, types, stances=stances, facets=facets, where=where
        )
        held: dict[str, tuple[str, list[str]]] = {}
        for row in rows:
            held.setdefault(row.chunk_id, (row.doc_id, []))[1].append(row.entity_id)
        return [
            Membership(group=chunk, doc_id=doc, members=tuple(sorted(set(names))))
            for chunk, (doc, names) in sorted(held.items())
        ]
    if network == "speakers":
        pairs = store.speaker_document_pairs(persona_id, source_id)
        if facets:
            _, faceted = _facet_passages(store, persona_id, source_id, facets)
            pairs = [p for p in pairs if p.doc_id in faceted]
        pairs = [p for p in pairs if matches(p.speaker_attributes, where)]
        by_document: dict[str, list[str]] = {}
        for pair in pairs:
            by_document.setdefault(pair.doc_id, []).append(pair.speaker)
        return [
            Membership(group=doc, doc_id=doc, members=tuple(sorted(set(names))))
            for doc, names in sorted(by_document.items())
        ]
    if network == "topics":
        documents = _documents_for(store, persona_id, source_id, facets, where)
        return [
            Membership(group=doc.id, doc_id=doc.id, members=tuple(sorted(set(doc.topics))))
            for doc in documents
        ]
    if network == "relations":
        return _relation_memberships(store, persona_id, source_id, relation_types, facets, where)
    return _two_mode_memberships(store, persona_id, source_id, types, stances, facets, where)


def _documents_for(
    store: GraphStore,
    persona_id: str,
    source_id: str | None,
    facets: Sequence[str] | None,
    where: Mapping[str, str] | None,
) -> list[Document]:
    """The persona's documents after the facet and attribute filters, in id order."""
    attributes = store.document_attributes(persona_id)
    faceted = _facet_passages(store, persona_id, source_id, facets)[1] if facets else None
    documents = [
        d
        for d in store.list_documents(persona_id, limit=_ALL)
        if (source_id is None or d.source_id == source_id)
        and (faceted is None or d.id in faceted)
        and matches(attributes.get(d.id, {}), where)
    ]
    return sorted(documents, key=lambda d: d.id)


def _relation_memberships(
    store: GraphStore,
    persona_id: str,
    source_id: str | None,
    relation_types: Sequence[str] | None,
    facets: Sequence[str] | None,
    where: Mapping[str, str] | None,
) -> list[Membership]:
    """One membership per stated relation: an ordered pair, because ``(u, v)`` is not ``(v, u)``."""
    rows: list[RelationRow] = store.relation_rows(persona_id, source_id)
    if relation_types:
        wanted = {t.strip() for t in relation_types if t.strip()}
        rows = [r for r in rows if r.type in wanted]
    if facets:
        passages, _ = _facet_passages(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.chunk_id in passages]
    if where:
        documents = store.document_attributes(persona_id)
        entities = store.entity_attributes(persona_id)
        rows = [
            r
            for r in rows
            if all(
                matches(resolve(entities.get(node, {}), documents.get(r.doc_id, {})), where)
                for node in (r.source_id, r.target_id)
            )
        ]
    return [
        Membership(
            group=f"{row.chunk_id}|{row.source_id}|{row.target_id}|{row.type}",
            doc_id=row.doc_id,
            members=(row.source_id, row.target_id),
            ordered=True,
        )
        for row in sorted(rows, key=lambda r: (r.chunk_id, r.source_id, r.target_id, r.type))
    ]


def _two_mode_memberships(
    store: GraphStore,
    persona_id: str,
    source_id: str | None,
    types: Sequence[str] | None,
    stances: Sequence[str] | None,
    facets: Sequence[str] | None,
    where: Mapping[str, str] | None,
) -> list[Membership]:
    """One membership per (document, speaker, entity): the two-mode edge, deduplicated per document.

    Per document rather than per passage, for the reason
    :func:`graphrag.sna.export.speaker_entity_bipartite` counts documents: chunk boundaries are
    an artefact of the chunker, and a long post split in three would otherwise look like three
    occasions on which the speaker named the thing.
    """
    rows: list[EntityMention] = store.entity_mention_rows(persona_id, source_id, types)
    if stances:
        wanted = set(stances)
        rows = [r for r in rows if r.stance in wanted]
    if facets:
        passages, _ = _facet_passages(store, persona_id, source_id, facets)
        rows = [r for r in rows if r.chunk_id in passages]
    speaker_attrs = store.speaker_attributes(persona_id)
    entity_attrs = store.entity_attributes(persona_id)
    if where:
        rows = [
            r
            for r in rows
            if matches(resolve(entity_attrs.get(r.entity_id, {}), r.document_attributes), where)
        ]
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        for speaker in row.speakers:
            if matches(speaker_attrs.get(speaker, {}), where):
                seen.add((row.doc_id, speaker, row.entity_id))
    return [
        Membership(group=f"{doc}|{speaker}", doc_id=doc, members=(speaker, entity))
        for doc, speaker, entity in sorted(seen)
    ]


def dynamic_edges(
    store: GraphStore,
    persona_id: str,
    network: str = "speakers",
    *,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    where: Mapping[str, str] | None = None,
) -> DynamicEdges:
    """The network as a timestamped edge list: one row per activation (§7.4).

    Every edge in this package comes from a document or from one of its passages, so a date on
    the document dates the edge. That date is the earliest dated passage the attribution pass
    wrote (:func:`document_dates`); a document nobody dated contributes nothing and is counted
    instead, which is the same rule ``--since``/``--until`` already apply and the reason a
    snapshot of a half-dated corpus is not a sample of it.

    ``since`` and ``until`` filter the *timestamps* here rather than the store's windowed read,
    which is the same set of documents by construction and one read cheaper.

    A pair that turns up twice in one document with two different passages is two rows for the
    entity network (two passages named them together) and one row for the two-mode network (one
    document, per that network's own convention).
    """
    if network not in NETWORKS:
        msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
        raise ValueError(msg)
    dated = document_dates(store, persona_id, source_id)
    memberships = _memberships(
        store,
        persona_id,
        network,
        source_id=source_id,
        types=types,
        stances=stances,
        facets=facets,
        relation_types=relation_types,
        where=where,
    )
    edges: list[TimedEdge] = []
    dropped = 0
    undated: set[str] = set()
    for membership in memberships:
        pairs = membership.pairs()
        at = dated.get(membership.doc_id)
        if at is None:
            undated.add(membership.doc_id)
            dropped += len(pairs)
            continue
        if (since is not None and at < since) or (until is not None and at > until):
            continue
        edges += [TimedEdge(source=u, target=v, at=at, doc_id=membership.doc_id) for u, v in pairs]
    edges.sort(key=lambda e: (e.at, e.source, e.target, e.doc_id))
    first, last = (edges[0].at, edges[-1].at) if edges else ("none", "none")
    unit = "document" if network in {"speakers", "topics", "speakers-entities"} else "passage"
    return DynamicEdges(
        persona_id=persona_id,
        network=network,
        edges=tuple(edges),
        undated_documents=len(undated),
        undated_edges=dropped,
        frame=DYNAMIC_FRAME.format(
            occurrences=len(edges),
            first=first,
            last=last,
            unit=unit,
            undated=len(undated),
            dropped=dropped,
        ),
    )


_SPEC = re.compile(r"^(\d+)\s*([dwmy])?$", re.IGNORECASE)
_UNITS = {"D": "day", "W": "week", "M": "month", "Y": "year"}


def _parse_spec(value: str | int, option: str) -> tuple[int, str]:
    """``6M``, ``90d``, ``2y``, ``3W`` or a bare number of days, as ``(amount, unit)``."""
    text = str(value).strip()
    match = _SPEC.match(text)
    if match is None:
        msg = (
            f"{option} must be a count of days or a count with a unit -- 90, 90D, 6M, 2Y -- "
            f"got {value!r}"
        )
        raise ValueError(msg)
    amount = int(match.group(1))
    if amount < 1:
        msg = f"{option} must be at least 1, got {value!r}"
        raise ValueError(msg)
    return amount, (match.group(2) or "D").upper()


def _spec_text(spec: tuple[int, str]) -> str:
    """``(6, 'M')`` as ``6 months``, for the frame."""
    amount, unit = spec
    return f"{amount} {_UNITS[unit]}{'s' if amount != 1 else ''}"


def _iso(value: str, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{option} must be an ISO date, YYYY-MM-DD, got {value!r}"
        raise ValueError(msg) from exc


def _shift(start: date, amount: int, unit: str) -> date:
    """``start`` advanced by ``amount`` of ``unit``, clamping a month end onto a shorter month."""
    if unit == "D":
        return start + timedelta(days=amount)
    if unit == "W":
        return start + timedelta(weeks=amount)
    months = amount * 12 if unit == "Y" else amount
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1]))


def windows(
    since: str,
    until: str,
    window: str | int = "6M",
    step: str | int | None = None,
    cumulative: bool = False,
) -> list[tuple[str, str]]:
    """The window grid, as inclusive ``(since, until)`` ISO pairs (§7.4).

    ``window`` is the width and ``step`` how far each window starts after the one before, both
    as a count of days or a count with a unit (``90``, ``90D``, ``6M``, ``2Y``). Together they
    cover three of the four strategies §7.4 lists, and ``cumulative`` is the fourth:

    - ``step == window`` gives **disjoint windows**: no overlap, nothing discarded.
    - ``step < window`` gives **sliding windows**: the next one starts before the last ended, so
      continuity is preserved and consecutive windows share edges -- which is exactly why their
      measurements are not independent of each other.
    - a ``window`` of one day is the **single snapshot** of §7.4, one instant at a time.
    - ``cumulative=True`` fixes every window's start at ``since``, so information only ever
      accumulates and each window contains all the ones before it.

    The last window is clipped at ``until`` rather than running past it, so it may be shorter
    than the others -- worth saying in a report, because a short window has fewer edges for a
    reason that is nothing to do with the corpus.

    Raises ``ValueError`` if the range runs backwards or if the grid would exceed
    :data:`MAX_WINDOWS`, which at that point is a mistake in the width rather than an analysis.
    """
    start, last = _iso(since, "--since"), _iso(until, "--until")
    if last < start:
        msg = f"the window range runs backwards: {since} is after {until}"
        raise ValueError(msg)
    width = _parse_spec(window, "--window")
    stride = _parse_spec(step, "--step") if step is not None else width
    grid: list[tuple[str, str]] = []
    cursor = start
    while cursor <= last:
        end = min(_shift(cursor, *width) - timedelta(days=1), last)
        grid.append((since if cumulative else cursor.isoformat(), end.isoformat()))
        cursor = _shift(cursor, *stride)
        if len(grid) > MAX_WINDOWS:
            msg = (
                f"a window of {_spec_text(width)} stepping every {_spec_text(stride)} over "
                f"{since} to {until} makes more than {MAX_WINDOWS} windows; widen the window"
            )
            raise ValueError(msg)
    return grid


def window_graph(
    store: GraphStore,
    persona_id: str,
    network: str,
    *,
    since: str | None = None,
    until: str | None = None,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
    projection: str = "simple",
    lam: float = DEFAULT_LAMBDA,
    backbone: str | None = None,
    alpha: float = DEFAULT_ALPHA,
    correction: str = "none",
    threshold: float | None = None,
    project: Project | None = None,
) -> nx.Graph:
    """One window's network: :func:`graphrag.sna.export.build_network` with a date range.

    The single place a window is turned into a graph, so the two builds of
    :func:`graphrag.sna.compare.compare_windows` and the sequence :func:`snapshots` yields are
    the same object built the same way, and a change to one cannot quietly stop applying to the
    other. It adds nothing of its own: a graph from here is indistinguishable from the one
    ``--since``/``--until`` on the command line has always produced -- including ``projection``,
    chapter 26's weighting, which has to be the same in every window of a comparison for the same
    reason ``backbone`` does: two schemes are two definitions of an edge weight, so a difference
    across them would be reporting the definition rather than the corpus.
    """
    return build_network(
        store,
        network,
        persona_id,
        source_id=source_id,
        min_weight=min_weight,
        types=types,
        stances=stances,
        facets=facets,
        relation_types=relation_types,
        since=since,
        until=until,
        where=where,
        projection=projection,
        lam=lam,
        backbone=backbone,
        alpha=alpha,
        correction=correction,
        threshold=threshold,
        project=project,
    )


#: How many days a unit is worth, for comparing a width with a step. Months and years are not a
#: fixed number of days, so these are the usual averages and are used only to order two specs
#: against each other -- never to build a window, which :func:`_shift` does on the calendar.
_DAYS = {"D": 1.0, "W": 7.0, "M": 30.436875, "Y": 365.2425}


def _mode(width: tuple[int, str], stride: tuple[int, str], cumulative: bool) -> tuple[str, bool]:
    """Which of §7.4's strategies this grid is, and whether it is none of them.

    The book names four: a single snapshot (an instant), disjoint windows (step equals width),
    sliding windows (step narrower than width) and cumulative windows (every window starts at the
    beginning). A step *wider* than the width is a fifth thing the section does not describe --
    it discards the time between one window and the next -- so it is named for what it is rather
    than filed under sliding, and the frame says what was thrown away.
    """
    if cumulative:
        return "cumulative windows", False
    if width == (1, "D"):
        return "a single snapshot", False
    if stride == width:
        return "disjoint windows", False
    if _days(stride) < _days(width):
        return "sliding windows", False
    return (
        "gapped windows, which is none of §7.4's four: the time between windows is discarded",
        True,
    )


def _days(spec: tuple[int, str]) -> float:
    """A width or step in days, for ordering two specs. Approximate for months and years."""
    amount, unit = spec
    return amount * _DAYS[unit]


def _gap_text(width: tuple[int, str], stride: tuple[int, str]) -> str:
    """How much of the calendar falls between two consecutive windows, in words.

    Approximate, because a month is not a fixed number of days. The windows themselves are cut on
    the calendar by :func:`_shift`; this only has to tell a reader roughly how much is gone.
    """
    missing = _days(stride) - _days(width)
    if missing < 60:
        return f"roughly {missing:.0f} days"
    return f"roughly {missing / _DAYS['M']:.0f} months"


def _undated_count(store: GraphStore, persona_id: str, source_id: str | None) -> int:
    """How many of the persona's documents carry no dated passage, in two reads.

    A windowed read of the store drops anything undated, so asking for a window that ends after
    any conceivable post gives the dated documents; the rest are the undated ones. That is the
    cheap form of the count :func:`document_dates` computes exactly and expensively.
    """
    every = store.document_ids(persona_id, source_id)
    dated = {
        row.doc_id for row in store.speaker_document_pairs(persona_id, source_id, until=_ANY_DATE)
    }
    return len(every - dated)


def snapshots(
    source: GraphStore | DynamicEdges,
    persona_id: str = "",
    network: str = "speakers",
    *,
    window: str | int = "6M",
    step: str | int | None = None,
    since: str | None = None,
    until: str | None = None,
    cumulative: bool = False,
    source_id: str | None = None,
    min_weight: int | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
    project: Project | None = None,
) -> list[tuple[str, nx.Graph]]:
    """Cut a dynamic network into static views, one per window: ``G = (G1, G2, ..., Gn)`` (§7.4).

    ``source`` is either the store -- each window is then built by :func:`window_graph`, exactly
    as ``--since``/``--until`` would have built it, filters and node attributes and all -- or a
    :class:`DynamicEdges` list, in which case each window's graph is assembled from the
    activations that fall inside it and carries weights and nothing else. The first is what a
    report should use; the second is the cheap form for a sweep over many windows, and it says
    so in its frame rather than pretending to be the first.

    ``since`` and ``until`` bound the grid. Left out, they are the span of the corpus's own dates,
    which costs one read per document (:func:`document_dates`); pass them when you know them.

    Every graph carries ``graph.graph["dynamic"]``, which is what makes
    :func:`graphrag.sna.export.describe` call it dynamic, plus the window, the step and the count
    of documents that are in no window because nothing dated them. An empty window is yielded as
    an empty graph rather than skipped: a gap in a history is a finding, and dropping it would
    make the sequence look continuous.

    The frame also names the two ways this grid can be less than it looks. A ``step`` wider than
    the ``window`` makes **gapped windows**, which is none of §7.4's four strategies: the time
    between one window and the next is thrown away, so the sequence is a sample of the history
    rather than a cut of it. And the last window is clipped at ``until`` rather than running past
    it, so it can be shorter than the others and hold fewer edges for a reason that is nothing
    to do with the corpus.
    """
    history = source if isinstance(source, DynamicEdges) else None
    store = None if history is not None else cast(GraphStore, source)
    if store is not None and network not in NETWORKS:
        msg = f"network must be one of {', '.join(NETWORKS)}, got {network!r}"
        raise ValueError(msg)
    if history is not None:
        persona_id, network = history.persona_id, history.network
        undated = history.undated_documents
        first, last = history.span
    else:
        store = cast(GraphStore, source)
        undated = _undated_count(store, persona_id, source_id)
        first, last = "", ""
        if since is None or until is None:
            dates = sorted(document_dates(store, persona_id, source_id).values())
            first, last = (dates[0], dates[-1]) if dates else ("", "")
    start, end = since or first, until or last
    if not start or not end:
        msg = (
            "nothing in this corpus is dated, so it has no windows. Run the attribution pass "
            "that writes post dates, or pass --since and --until explicitly."
        )
        raise ValueError(msg)

    grid = windows(start, end, window, step, cumulative)
    width = _parse_spec(window, "--window")
    stride = _parse_spec(step, "--step") if step is not None else width
    mode, gapped = _mode(width, stride, cumulative)
    # The grid is clipped at ``end``, so the last window is short unless it happens to land on it.
    # Read off the last window's start, so this covers the non-cumulative grids only: a cumulative
    # grid pins every start to ``since``, and its last window is the longest, not the shortest.
    clipped = len(grid) > 1 and _shift(_iso(grid[-1][0], "--since"), *width) - timedelta(
        days=1
    ) > _iso(end, "--until")
    notice = CLIPPED_FRAME.format(until=end) if clipped else ""
    if gapped:
        notice = GAPPED_FRAME.format(gap=_gap_text(width, stride)) + notice
    out: list[tuple[str, nx.Graph]] = []
    for index, (opens, closes) in enumerate(grid, start=1):
        label = f"{opens} to {closes}"
        if history is not None:
            graph = _graph_from_edges(history, opens, closes)
        else:
            graph = window_graph(
                cast(GraphStore, source),
                persona_id,
                network,
                since=opens,
                until=closes,
                source_id=source_id,
                min_weight=min_weight,
                types=types,
                stances=stances,
                facets=facets,
                relation_types=relation_types,
                where=where,
                project=project,
            )
        frame = SNAPSHOT_FRAME.format(
            mode=mode,
            window=_spec_text(width),
            step=_spec_text(stride),
            since=start,
            until=end,
            label=label,
            index=index,
            total=len(grid),
            clipping=notice,
            undated=undated,
        )
        graph.graph.update(
            dynamic=label,
            window=_spec_text(width),
            step=_spec_text(stride),
            windows=len(grid),
            window_index=index,
            undated=undated,
            frame=f"{graph.graph.get('frame', '')} {frame}".strip(),
        )
        out.append((label, graph))
    return out


def _graph_from_edges(edges: DynamicEdges, since: str, until: str) -> nx.Graph:
    """The activations inside one window, as a weighted graph of how often each pair fired."""
    directed = edges.network == "relations"
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    inside = [e for e in edges.edges if since <= e.at <= until]
    graph.add_nodes_from(sorted({node for e in inside for node in (e.source, e.target)}))
    for edge in inside:
        u, v = (edge.source, edge.target) if directed else tuple(sorted((edge.source, edge.target)))
        if graph.has_edge(u, v):
            graph[u][v]["weight"] += 1
        else:
            graph.add_edge(u, v, weight=1)
    graph.graph.update(
        network=edges.network,
        persona_id=edges.persona_id,
        since=since,
        until=until,
        unit="activations in this window",
        frame=(
            f"{edges.frame} Assembled from the timestamped edge list rather than rebuilt from "
            "the store, so the nodes carry no attributes and the weights count activations "
            "inside the window, not the units the unlayered network counts."
        ),
    )
    return graph
