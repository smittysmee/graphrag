"""A notebook-facing facade over the rest of this package (entitlement ticket ATL-ENT-2).

Everything under ``sna/`` is written to run from ``graphrag sna <command>``: a persona id and a
handful of ``--flag``s in, a rendered report or a written file out. That is the right shape for a
terminal and the wrong one for a notebook, where a cell wants an object back -- something whose
attributes can be poked at, plotted, or handed to the next cell -- not a markdown string to
re-parse. :class:`Network` is that object: it holds one built ``nx.Graph`` and forwards every
question about it to the exact function ``graphrag sna`` already calls, so the numbers a notebook
prints are never a second implementation of the numbers a report prints. ``docs/SNA_NOTEBOOK.md``
is the page-length tour; this module is where its five calls live.

:meth:`Network.build` is the only thing here that touches a store, and it touches it for exactly
as long as building the network takes -- open through :class:`graphrag.app.AppContext`, call
:func:`graphrag.sna.export.build_network`, close, return. Every other method reads the graph this
already built and needs no connection at all, with one named exception in :meth:`Network.analyze`
(clustering by a stored entity embedding). :meth:`Network.from_graph` and the package's own
:func:`graphrag.sna.export.read_graph` are the other way in, for a graph that came from a file
or from a chain of calls this module never anticipated -- chapter 7's flattening, a sampler, a
hand-built synthetic network -- rather than from a persona's store at all.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import networkx as nx

from graphrag.app import AppContext
from graphrag.config import Settings
from graphrag.embed.base import Embedder
from graphrag.graph.store import GraphStore
from graphrag.sna.analysis import Analysis, Method, run_analysis
from graphrag.sna.backbone import backbone as apply_backbone
from graphrag.sna.draw import DrawResult
from graphrag.sna.draw import draw as draw_network
from graphrag.sna.evaluate import PartitionScores, evaluate_partition
from graphrag.sna.experiment import ScoreTable, preferential_attachment
from graphrag.sna.export import NetworkDescription, build_network
from graphrag.sna.export import describe as describe_network
from graphrag.sna.matrices import Node
from graphrag.sna.measures import (
    Centralization,
    DensityReport,
    centralization_table,
    density_report,
)
from graphrag.sna.measures import summary as network_summary
from graphrag.sna.predict import adamic_adar, common_neighbors, jaccard, katz, resource_allocation

#: The graph-only members of chapter 23's family (§23.1-§23.4, §23.7). ``hrg`` and ``rules`` are
#: not offered by :meth:`Network.predict`: HRG needs the same MCMC restarts `sna predict
#: --method hrg` pays for, and association rules need
#: :func:`graphrag.sna.motifs.document_graphs`, which reads the documents behind an edge -- a
#: network this module holds has already thrown those away. Both stay reachable through
#: `graphrag sna predict` and through calling :mod:`graphrag.sna.predict` directly with a store.
_PREDICT_METHODS: tuple[str, ...] = ("pa", "cn", "jaccard", "aa", "ra", "katz")


class _NoStore:
    """Stands in for a store a :class:`Network` never opened (:meth:`Network.from_graph`,
    :func:`graphrag.sna.export.read_graph`), so :meth:`Network.analyze` can hand ``run_analysis``
    something without asking every caller for a store up front. It raises the moment anything
    actually reads an attribute off it, naming that attribute, which is the one honest way to
    say "this needed a store and none was given" without duplicating ``run_analysis``'s own
    branching on which combination of ``method``/``features`` reads one.
    """

    def __getattr__(self, name: str) -> Any:
        msg = (
            "this Network holds no store connection (built with Network.from_graph or "
            "read_graph, which open none): pass store=<GraphStore> to analyze() -- needed only "
            f"because the requested method just tried to read {name!r} off it."
        )
        raise RuntimeError(msg)


@dataclass(frozen=True)
class Measures:
    """Chapter 12's density block and chapter 14's rankings, read once and bundled (§12.1-§12.4,
    §14.8). Every field is exactly what :mod:`graphrag.sna.measures` already returns for its own
    function -- this dataclass only saves a notebook cell three imports and three calls.

    ``summary`` is :func:`graphrag.sna.measures.summary`'s graph-level counts (nodes, edges,
    density, mean degree, ...). ``density`` is
    :func:`graphrag.sna.measures.density_report`'s chapter-12 section -- density read against n,
    the two clustering coefficients, cliques, an independent set -- and is ``None`` only for an
    empty network, which has none of them. ``centralization`` is one
    :class:`graphrag.sna.measures.Centralization` row per centrality this network supports
    (§14.8), each naming why a row is undefined rather than dropping it.
    """

    summary: dict[str, float]
    density: DensityReport | None
    centralization: list[Centralization]


@dataclass
class Network:
    """One built network, held as a plain object for a notebook cell to poke at.

    ``graph`` is the ``nx.Graph`` (or ``nx.DiGraph``, for ``--network relations``) every method
    below reads. ``persona_id`` and ``network`` name what it is a network *of*, the same two
    values every :mod:`graphrag.sna.analysis` report carries, and ``filters`` is the keyword
    arguments :meth:`build` (or a caller of :meth:`from_graph`) was given -- both are read-only
    bookkeeping, not re-applied by anything here.

    Every method that measures, groups, filters or draws the graph -- :meth:`analyze`,
    :meth:`backbone`, :meth:`measures`, :meth:`evaluate`, :meth:`predict`, :meth:`draw` -- calls
    the exact function ``graphrag sna <command>`` calls with the exact same arguments, and returns
    the exact dataclass that function already returns. Nothing here is a second implementation of
    a number the CLI prints; a test in ``tests/unit/test_sna_facade.py`` asserts the two paths
    agree field for field on the same fixture. ``describe()`` is the one exception in name only --
    it is :func:`graphrag.sna.export.describe` itself, called with no arguments to read.
    """

    graph: nx.Graph
    persona_id: str
    network: str
    filters: Mapping[str, Any] = field(default_factory=dict)
    #: The store :meth:`build` opened, already closed by the time this object exists. Kept only
    #: for :meth:`analyze`'s one path that still needs a connection (clustering by a stored
    #: entity embedding); every other method never reads it. ``None`` on a :class:`Network` made
    #: by :meth:`from_graph` or :func:`graphrag.sna.export.read_graph`, which never opened one.
    _store: GraphStore | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------------------- building

    @classmethod
    def build(
        cls,
        persona: str,
        network: str,
        *,
        settings: Settings | None = None,
        store: GraphStore | None = None,
        embedder: Embedder | None = None,
        **filters: Any,
    ) -> Network:
        """Open a store, build one network from it, close the store, return the result.

        ``persona`` and ``network`` are exactly ``graphrag sna export``'s persona argument and
        ``--network``. Every other keyword is one of
        :func:`graphrag.sna.export.build_network`'s own filters (``source_id``, ``min_weight``,
        ``types``, ``stances``, ``facets``, ``relation_types``, ``since``, ``until``, ``where``,
        ``projection``, ``lam``, ``backbone``, ``alpha``, ``correction``, ``threshold``,
        ``project``), passed straight through: a notebook already has native Python values --
        ``where={"region": "north"}``, not ``--where region=north`` -- so nothing here re-parses
        a string the CLI would have had to.

        ``settings``/``store``/``embedder`` are the exact overrides
        :meth:`graphrag.app.AppContext.build` takes, for the same reason that method takes them:
        a test (or a notebook already holding an open connection) injects one rather than reading
        :class:`graphrag.config.Settings` from the environment. What :meth:`AppContext.build`
        does when none are given -- read ``Settings`` from the environment, open the configured
        Neo4j store -- is exactly what this does too.

        Layering (``--layers``), sampling (``--sample``) and the multilayer stack CLI's
        ``sna analyze`` composes around ``build_network`` are not built here: build the layered
        or sampled graph with :mod:`graphrag.sna.layers`/:mod:`graphrag.sna.sampling` directly and
        hand the result to :meth:`from_graph`.
        """
        ctx = AppContext.build(settings, store=store, embedder=embedder)
        try:
            ctx.registry.get(persona)
            graph = build_network(ctx.store, network, persona, **filters)
        finally:
            ctx.close()
        return cls(
            graph=graph,
            persona_id=persona,
            network=network,
            filters=dict(filters),
            _store=ctx.store,
        )

    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        *,
        persona_id: str = "",
        network: str = "",
        store: GraphStore | None = None,
    ) -> Network:
        """Wrap an already-built graph -- from :func:`graphrag.sna.export.read_graph`, from a
        layering or a sample, or built by hand -- without opening anything.

        ``persona_id``/``network`` are bookkeeping only, printed by :meth:`analyze` and
        :meth:`draw`'s reports; leave them blank for a graph that has neither, such as one of
        ``tests/legendary.py``'s. ``store`` is optional and read only by :meth:`analyze`'s
        stored-embedding path (see :meth:`build`'s docstring); every other method needs none.
        """
        return cls(graph=graph, persona_id=persona_id, network=network, filters={}, _store=store)

    # ------------------------------------------------------------------------------- reading

    def describe(self) -> NetworkDescription:
        """This network's type in the Atlas's own vocabulary (ch. 6): directed, weighted,
        bipartite, ... -- :func:`graphrag.sna.export.describe`, read fresh from ``self.graph``
        every call, since a method such as :meth:`backbone` can change what it would say.
        """
        return describe_network(self.graph)

    def to_pandas(self) -> tuple[Any, Any]:
        """Node and edge tables, one row per node and per edge, as ``pandas.DataFrame``s.

        pandas is not a dependency of this package (see ``pyproject.toml``), so this degrades
        instead of failing it: with pandas installed, it returns two ``DataFrame``s built from
        the same rows ``sna export --network ... out.csv`` would write (one row per node keyed
        by ``id`` plus every node attribute, one row per edge keyed by ``source``/``target`` plus
        every edge attribute, ``weight`` among them); without it, the same two lists of plain
        ``dict``s, which ``pandas.DataFrame(records)`` builds identically once pandas is
        installed, and which ``csv.DictWriter`` writes without it.
        """
        nodes = [{"id": node, **data} for node, data in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
        try:
            # A dynamic import, not ``import pandas as pd``: pandas carries no stubs this
            # package declares, and a static import would make every ``mypy --strict`` run
            # depend on a library nothing else here needs (see :func:`graphrag.sna.export.
            # to_igraph` for the same trick, for the same reason, with a bridge library).
            pd = importlib.import_module("pandas")
        except ImportError:
            return nodes, edges
        return pd.DataFrame(nodes), pd.DataFrame(edges)

    # ------------------------------------------------------------------------------- analysis

    def analyze(
        self, *, method: Method = "louvain", store: GraphStore | None = None, **kwargs: Any
    ) -> Analysis:
        """Measure, group and check this network: :func:`graphrag.sna.analysis.run_analysis`,
        called with this network's graph, persona and network name, and every keyword this method
        was given -- ``method``, and the rest of ``run_analysis``'s own keywords (``by``,
        ``resolution``, ``features``, ``seed``, ``null``, ``uncertain``, ...) unchanged.

        ``store`` defaults to the (already closed, see :meth:`build`) store this network was
        built with, which is enough for every method: the default ``features="spectral"`` never
        reads it. Only ``method="kmeans"``/``"gmm"`` with ``features="embedding"`` -- clustering
        by a stored mean entity embedding -- needs a live connection, and reads it the moment it
        tries to. A :class:`Network` made by :meth:`from_graph` or
        :func:`graphrag.sna.export.read_graph`, which never opened a store at all, raises
        :class:`RuntimeError` at that same moment, naming the attribute it tried to read, unless
        ``store`` is given here -- and never raises at all for a method that reads no store.
        """
        resolved = store if store is not None else self._store
        store_arg = resolved if resolved is not None else cast(GraphStore, _NoStore())
        return run_analysis(
            store_arg,
            self.graph,
            persona_id=self.persona_id,
            network=self.network,
            method=method,
            **kwargs,
        )

    def backbone(self, method: str = "noise-corrected", **kwargs: Any) -> Network:
        """Chapter 27's filter (:func:`graphrag.sna.backbone.backbone`), returned as a new
        :class:`Network` over the filtered graph so it chains: ``net.backbone("mst").analyze()``.
        ``method`` and every keyword (``alpha``, ``correction``, ``threshold``, ``keep_isolates``,
        ``seed``) are exactly that function's own; see its docstring for what each backbone reads.
        The new network keeps this one's persona, network name, filters and store reference --
        backboning narrows edges, not what the network is a network *of*.
        """
        filtered = apply_backbone(self.graph, method, **kwargs)
        return Network(
            graph=filtered,
            persona_id=self.persona_id,
            network=self.network,
            filters=self.filters,
            _store=self._store,
        )

    def measures(self) -> Measures:
        """Chapter 12's density block and chapter 14's rankings over this graph, bundled as
        :class:`Measures`. Unlike :meth:`analyze`, nothing here groups the network or checks a
        partition -- these three are the measures every one of those needs but none of them
        assumes: the graph's own shape, before any clustering is asked of it.
        """
        graph = self.graph
        return Measures(
            summary=network_summary(graph),
            density=density_report(graph) if graph.number_of_nodes() else None,
            centralization=centralization_table(graph),
        )

    def evaluate(self, communities: Sequence[Iterable[Node]], **kwargs: Any) -> PartitionScores:
        """Chapter 36's whole battery over one partition of this graph
        (:func:`graphrag.sna.evaluate.evaluate_partition`): modularity and its resolution-limit
        check, the other topological measures, the partition read as a link predictor, and --
        with ``truth=``, one of this function's own keywords -- NMI and AMI against a ground
        truth. ``communities`` does not have to be :meth:`analyze`'s own ``Analysis.groups``; any
        disjoint partition of (a subset of) ``self.graph``'s nodes is scored the same way.
        """
        return evaluate_partition(self.graph, communities, **kwargs)

    def predict(self, method: str = "cn", **kwargs: Any) -> ScoreTable:
        """One of chapter 23's graph-only link-prediction scorers over this network
        (§23.1-§23.4, ``pa`` from §25.2's own baseline, §23.7's Katz): ``pa`` (preferential
        attachment), ``cn`` (common neighbours), ``jaccard``, ``aa`` (Adamic-Adar), ``ra``
        (resource allocation) or ``katz`` (``beta``, ``max_nodes`` keywords). ``hrg`` and
        ``rules``, the other two members of `graphrag sna predict`'s ``--method``, are not
        offered here: HRG needs the same MCMC restarts `sna predict --method hrg` pays for, and
        association rules need :func:`graphrag.sna.motifs.document_graphs`, which reads the
        documents behind an edge -- a network this class holds has already thrown those away.

        Returns a :class:`graphrag.sna.experiment.ScoreTable`: ``table[u, v]`` scores one pair,
        iterating or measuring it against a holdout the way :func:`graphrag.sna.experiment.
        evaluate_predictor` does. It never writes anything -- a score is a claim about what the
        corpus would say if somebody looked, exactly as the chapter frames it.
        """
        if method == "pa":
            return preferential_attachment(self.graph, **kwargs)
        if method == "cn":
            return common_neighbors(self.graph, **kwargs)
        if method == "jaccard":
            return jaccard(self.graph, **kwargs)
        if method == "aa":
            return adamic_adar(self.graph, **kwargs)
        if method == "ra":
            return resource_allocation(self.graph, **kwargs)
        if method == "katz":
            return katz(self.graph, **kwargs)
        msg = (
            f"method must be one of {', '.join(_PREDICT_METHODS)}, got {method!r}. 'hrg' and "
            "'rules' are not offered by this method; run `graphrag sna predict --method "
            "hrg|rules` for either."
        )
        raise ValueError(msg)

    def draw(self, **kwargs: Any) -> DrawResult:
        """Lay out and style this network for one SVG (:func:`graphrag.sna.draw.draw`, ch.
        49-51): ``layout``, ``size``, ``color``, ``seed``, ``width``, ``height`` are exactly that
        function's own keywords. ``result.svg`` is what a notebook cell displays (``IPython.
        display.SVG(result.svg)``, or ``HTML(result.html)`` for the version with a legend).
        """
        if self.persona_id or self.network:
            layout = kwargs.get("layout", "force")
            name = " · ".join(p for p in (self.persona_id, self.network, layout) if p)
            kwargs.setdefault("title", name)
        return draw_network(self.graph, **kwargs)
