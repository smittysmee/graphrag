"""Sampling a network that is too big to look at, and the honesty about what that costs.

Chapter 29 of *The Atlas for the Aspiring Network Scientist* is about the act of observation.
Every network in this package is already a sample -- of a corpus, not of a population, which is
what ``FRAMES`` in :mod:`graphrag.sna.export` says next to every number. This module adds the
second sampling step: taking a smaller network out of one we hold, for when the whole thing is
too large to measure, and reporting what the sampler did to the answer.

The chapter's own framing is the one to keep in mind while reading the code. *"By
'representative' we mean that the property you're interested in studying should be more or less
the same in the sample as in the network at large"* -- which is a property of a sampler and a
measure **together**, never of a sampler alone. A crawl that gives you the right degree
distribution can give you the wrong clustering coefficient, and the book names, per method,
which one it distorts and in which direction. Those directions are in :data:`BIASES`, in the
book's own terms, and :func:`bias_report` checks the observed direction against the predicted
one so that a report says "the book expects this sampler to oversample hubs, and here it did"
rather than printing a mean degree on its own.

The families, and the sections they come from:

``induced``, ``edge`` and ``ties`` (§29.1)
    Pick the elements first, collect what hangs off them. Uniform random nodes are unbiased in
    the mean but, on a heavy tail, unlikely to catch a hub, and the induced sample of a
    connected network falls apart into components. Uniform random edges pull the other way:
    *"most edges are attached to large hubs. Thus, if you pick an edge at random, it is likely
    that a hub will be attached to it"* -- the same arithmetic as §21.3's vaccination strategy.
    ``edge`` is the section's plain random edge induced sampling, which *"select[s] edges at
    random and you collect all their direct neighbors"*; ``ties`` is Totally Induced Edge
    Sampling, the *"more sophisticated"* variant it names, which stops at the endpoints and
    induces. Edge sampling is also the family the book says you cannot do through an API.

``bfs``, ``snowball``, ``forest-fire`` (§29.2)
    Start anywhere and explore. A plain breadth-first crawl reaches hubs early and explores a
    neighbourhood exhaustively, which *"overestimates"* clustering. Snowball caps the crawl at
    ``k`` connections per node -- the respondent names only ``k`` friends -- which keeps the hub
    but throws away its degree, *"generat[ing] weird degree distributions with a sharp cutoff,
    which aren't very realistic"*. Forest fire burns each neighbour with probability ``p``
    instead of taking them all, and is the BFS variant the book links to *"a proper estimation
    of the clustering coefficient"*.

``random-walk``, ``metropolis-hastings`` (§29.3)
    Walk instead of crawling. The stationary distribution of a random walk has a one-to-one
    correspondence with degree (§11.1), so a vanilla walk oversamples hubs **by construction**;
    teleportation stops it getting trapped but does nothing about that. The
    Metropolis-Hastings walk accepts a step from ``v`` to ``u`` with probability ``k_v / k_u``,
    which makes the stationary distribution uniform and is the book's fix for the bias itself.

``neighbor-reservoir`` (§29.3)
    The method the section itself says is *"one of those methods blending between the two
    families"*: a random walk builds a core, and then most of the budget goes on swapping a node
    of the core for a neighbour of it, keeping the sample the same size and refusing any swap
    that would break it into components. That refusal is the whole point -- a node whose
    neighbours are connected to each other is likelier to be removable, so high-clustering nodes
    churn and *"NRS ensures a realistic clustering coefficient distribution"*.

The re-weighting of §29.3 is here as well, as :func:`reweighted_degree_distribution`: not a
sampler but a correction applied to a sample after the fact, and the one piece of arithmetic
§29.5's completion estimate is built on.

**What §29.4 is about, and why none of it is code here.** The chapter's "Sampling Issues"
section is about the cost of crawling somebody else's system, and it is worth reading before
trusting any crawled dataset -- including the ones this package ingests.

*Pagination.* An API *"will rarely give you all connections of a user when you ask for them"*.
It returns ``k`` at a time, chosen *"with some criterion that is opaque to you (likely in the
order they are stored in their internal database)"*, and you pay a wait for each page. That
opacity is why every crawl here shuffles a node's neighbours before reading them: an order
nobody can justify is better modelled as no order at all.

*Throughput is not speed.* The book's own worked comparison: policy A returns 100 edges per page
with a two-second wait (50 edges/second on paper), policy B ten edges per page with a one-second
wait (10 edges/second). On a broad degree distribution B wins anyway -- *"out of 500k nodes,
492k have degree of 10 or less"*, so almost every node is one query for B and one query for A
alike, and B's wait is half. The nominally five-times-slower policy crawls the network in half
the time. The lesson generalises past APIs: a sampler's cost is set by the degree distribution
it is run against, not by its headline rate. Below that there is a floor of network latency, and
a server *"which gets hit too frequently with too many requests will also naturally slow
down"*, so no API is really one node per request with no wait.

*Hostility and privacy.* Some nodes *"will try to lie about their connections and it's your duty
to reconstruct the true underlying structure. Or not: there are reasonable and legit reasons to
lie about one's connection, for instance to protect one own privacy."*

*Node-centric questions.* When the question is about the local properties of one or a few nodes
rather than the whole topology, specialised node-centric strategies exist and a whole-network
sampler is the wrong tool.

None of the four is implemented, because none of them can be: this module samples a network we
already hold in memory, where there is no page size, no wait, no adversary and no query budget.
They are documented because most of the networks in this package came from somebody else's crawl
before we ever saw them, and §29.4 is what a reader needs in order to distrust them correctly.

**Named in the chapter and deliberately not built.** PIES (§29.1) is a *streaming* sampler,
built for the case where edges arrive one at a time and must be summarised in bounded space; the
book names it in one clause, gives no procedure, and our networks are static and small enough to
hold whole, so the space it saves buys nothing. Sample Edge Counts (§29.3) and m-dependent
Random Walk (§29.3) are both described as one-sentence variations on the walk -- SEC explores
the neighbour with the highest edge count toward the explored set, MRW runs ``m`` dependent
walkers chosen by the degree of where they stand -- and both are offered by the book as ways to
be *more* hub-biased on purpose (*"if we have to be biased, at least let's oversample the
important nodes"*), which is the opposite of what a sampler is for here; neither would change a
report's answer, only its tilt. ε-wgx (§29.5) is named as *"a more recent alternative"* to
MaxReach with no description at all.

Two things this module refuses to pretend. First, a sample's own measurements are always
compared against the population here, because we own the population -- we are sampling a network
we already hold. In the book's setting, crawling somebody else's API, that comparison does not
exist, which is why :func:`bias_report` labels the population column as known rather than
estimated. Second, :func:`completion` (§29.5) can only ever see the edges with at least one
endpoint in the sample; a region of the network the crawl never touched is invisible to any
estimate made from the crawl, so its completeness figure is an upper bound on how complete the
sample is.
"""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graphrag.sna.measures import summary
from graphrag.sna.stats import Correlation, Summary, describe, spearman

__all__ = [
    "BIASES",
    "CRAWL_DEGREE",
    "SAMPLERS",
    "SAMPLE_METHODS",
    "BiasCheck",
    "Completion",
    "ProbeTarget",
    "SampleBias",
    "SamplerBias",
    "bias_report",
    "completion",
    "render_sample",
    "reweighted_degree_distribution",
    "sample",
    "sample_payload",
    "sample_record",
]

#: Node attribute holding the degree a node had **in the population** at the moment the crawl
#: asked it for its connections. Only the nodes the sampler actually probed carry it: that is the
#: distinction §29.5 turns on, between a node whose neighbour list you have seen and a node you
#: merely know exists. :func:`completion` reads it back and estimates the rest.
CRAWL_DEGREE = "crawl_degree"

#: How far an observed value may sit from the population's before :func:`bias_report` calls the
#: direction anything other than "none", as a share of the population value. A sample is a random
#: object; a 3% difference in mean degree is not a bias, it is a sample.
DIRECTION_TOLERANCE = 0.05

#: Steps a walk may take without reaching a new node before the sampler gives up on that region
#: and restarts somewhere unsampled. Without it, ``--size`` could not be honoured on a network
#: whose walk-reachable part is smaller than the sample asked for.
STALL_LIMIT = 100

#: How many probe targets :func:`completion` lists by default.
TOP_PROBES = 5


@dataclass(frozen=True)
class SamplerBias:
    """What the book predicts one sampler will do to a measurement, and where it says so.

    ``degree``, ``clustering`` and ``connectivity`` are each ``"up"``, ``"down"``, ``"none"`` or
    ``""``. The empty string means the chapter makes no prediction for that measure and that
    sampler, which is different from predicting no effect: :func:`bias_report` still prints the
    number and simply does not grade it.
    """

    method: str
    section: str
    degree: str
    clustering: str
    connectivity: str
    sentence: str


#: The book's prediction per sampler, quoted closely enough to argue with. Read the direction as
#: a statement about the sample compared with the population it came from.
BIASES: dict[str, SamplerBias] = {
    "induced": SamplerBias(
        method="induced",
        section="§29.1",
        degree="none",
        clustering="",
        connectivity="down",
        sentence=(
            "Uniform random nodes. The selection is unbiased in the mean, but the degree "
            "distribution is 'emphatically not distributed normally', so the sample is 'unlikely "
            "to fairly represent the hubs', and the induced graph 'breaks down in multiple "
            "components' where the population was connected (§29.1). Read the degree row with "
            "that in mind: the fairness is fairness *over draws*, and one draw of a heavy tail "
            "moves a sample mean a long way in either direction, so a single induced sample "
            "often reads up or down and only the average over seeds sits on the population's "
            "value"
        ),
    ),
    "edge": SamplerBias(
        method="edge",
        section="§29.1",
        degree="",
        clustering="",
        connectivity="",
        sentence=(
            "Random edge induced sampling: uniformly random edges, and then 'you collect all "
            "their direct neighbors'. 'Most edges are attached to large hubs', so picking an "
            "edge at random is likely to pick up a hub: edge sampling counteracts the downward "
            "degree bias of node sampling and replaces it with an upward one. That prediction "
            "is about the endpoints the draw lands on, and this sampler then adds every "
            "neighbour of them -- the neighbours of a hub being overwhelmingly low-degree -- so "
            "the returned set mixes the two and its degree row is printed rather than graded. "
            "`ties` is the same draw without the neighbourhood, and is where the prediction is "
            "tested. Edge sampling is also the family you cannot use against an API, which "
            "rarely lets you start from an edge (§29.1)"
        ),
    ),
    "ties": SamplerBias(
        method="ties",
        section="§29.1",
        degree="up",
        clustering="",
        connectivity="",
        sentence=(
            "Totally Induced Edge Sampling, the first of the 'more sophisticated approaches' "
            "§29.1 names: uniformly random edges, their endpoints, and then every population "
            "edge between two of those endpoints added back. It carries the same upward degree "
            "bias as any edge-based selection -- 'most edges are attached to large hubs' -- and "
            "differs from plain edge induction in stopping at the endpoints, so the sample is "
            "the size that was asked for rather than the size the endpoints' degrees happen to "
            "make it (§29.1)"
        ),
    ),
    "bfs": SamplerBias(
        method="bfs",
        section="§29.2",
        degree="up",
        clustering="up",
        connectivity="none",
        sentence=(
            "A breadth-first crawl reaches the hubs early and explores each neighbourhood in "
            "full, which is why 'with a BFS we would overestimate' the clustering coefficient "
            "(§29.2). The sample is connected by construction, so it can say nothing about "
            "whether the population was"
        ),
    ),
    "snowball": SamplerBias(
        method="snowball",
        section="§29.2",
        degree="up",
        clustering="up",
        connectivity="none",
        sentence=(
            "Snowball is BFS with a cap of k connections per node: the hub is still found, but "
            "'their degree is somewhat capped, since they can only name k of their friends'. "
            "That 'generates weird degree distributions with a sharp cutoff, which aren't very "
            "realistic', and a snowball sample never learns any node's true degree (§29.2)"
        ),
    ),
    "forest-fire": SamplerBias(
        method="forest-fire",
        section="§29.2",
        degree="up",
        clustering="",
        connectivity="none",
        sentence=(
            "Forest fire is BFS that burns each neighbour with probability p rather than taking "
            "them all, so the neighbourhood is not explored exhaustively and the advantage 'is "
            "usually linked with a proper estimation of the clustering coefficient' (§29.2). "
            "That claim is comparative -- better than the BFS it is a variant of, which "
            "overestimates -- and not a claim that the sample's clustering equals the "
            "population's, so the clustering row is printed here and not graded. With p low the "
            "fire can die before the budget is spent"
        ),
    ),
    "random-walk": SamplerBias(
        method="random-walk",
        section="§29.3",
        degree="up",
        clustering="",
        connectivity="none",
        sentence=(
            "A vanilla random walk has a degree bias that is not an accident of the data: the "
            "stationary distribution 'has a 1-to-1 correspondence to the degree' (§11.1), so "
            "'high degree nodes are very likely to be sampled, while low degree nodes not so "
            "much'. Teleportation keeps the walk from getting trapped; it does not remove the "
            "bias (§29.3)"
        ),
    ),
    "neighbor-reservoir": SamplerBias(
        method="neighbor-reservoir",
        section="§29.3",
        degree="",
        clustering="",
        connectivity="none",
        sentence=(
            "Neighbor Reservoir Sampling, which the section calls 'one of those methods blending "
            "between the two families': a random walk builds a core, and then most of the budget "
            "goes on swapping a node of the core for a neighbour of it, refusing any swap that "
            "would 'break the graph into multiple components'. That refusal is what shapes the "
            "sample -- 'the v with higher clustering have higher probability to be replaced, "
            "because by removing them it is more likely that the graph will stay connected' -- "
            "and it is why 'NRS ensures a realistic clustering coefficient distribution'. That "
            "claim is about the shape of a distribution rather than about a mean moving in a "
            "direction, so the clustering row is printed and not graded; what is checked "
            "instead is the condition the claim rests on, that no swap may leave the sample in "
            "more components than it started in. Its core comes from a walk, so §29.3's degree "
            "bias is in it and the chapter predicts no direction of its own (§29.3)"
        ),
    ),
    "metropolis-hastings": SamplerBias(
        method="metropolis-hastings",
        section="§29.3",
        degree="none",
        clustering="",
        connectivity="none",
        sentence=(
            "The Metropolis-Hastings walk accepts a step from v to u with probability k_v / k_u, "
            "refusing a neighbour the more its degree exceeds the current node's. 'A random walk "
            "with this rule will generate a uniform stationary distribution', which is the "
            "book's fix for the vanilla walk's degree bias rather than a correction applied "
            "afterwards (§29.3). That uniformity is a property of where the walk *spends its "
            "steps*; the set of distinct nodes a finite crawl has discovered is a different "
            "object, and it still reaches hubs first, so a short MHRW sample keeps part of the "
            "bias and this row can come out 'up' against the prediction"
        ),
    ),
}


@dataclass
class _Crawl:
    """What a sampler produced: the nodes it took, in discovery order, and what it learned.

    ``degrees`` holds the population degree of every node the crawl *probed* -- asked for its
    connections -- which is what separates §29.5's "you already explored all its neighbors" from
    "you only know how many neighbors it has in your sample". A snowball crawl leaves it empty on
    purpose: a respondent who names k friends has not told you their degree.
    """

    nodes: list[Any] = field(default_factory=list)
    seen: set[Any] = field(default_factory=set)
    degrees: dict[Any, int] = field(default_factory=dict)
    seeds: int = 0
    #: Restarts forced by a walk or a fire that ran out of new nodes before ``size`` was reached.
    forced: int = 0
    #: Accepted node-for-node exchanges, for the one sampler that makes them (§29.3's NRS).
    swaps: int = 0

    def take(self, node: Any) -> None:
        if node not in self.seen:
            self.seen.add(node)
            self.nodes.append(node)

    def probed(self, graph: nx.Graph, node: Any) -> None:
        """Record that the crawl saw this node's full neighbour list."""
        self.degrees[node] = int(graph.degree(node))


Sampler = Callable[[nx.Graph, int, random.Random, Mapping[str, Any]], _Crawl]


def _accept(params: Mapping[str, Any], method: str, **defaults: Any) -> dict[str, Any]:
    """The parameters this sampler takes, coerced to the type of their default.

    An unknown parameter is refused rather than ignored: ``--param p=0.2`` on a sampler with no
    burning probability is a question about a different method, and silently dropping it would
    print a forest-fire caveat over a breadth-first sample.
    """
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        accepted = ", ".join(sorted(defaults)) or "no parameters"
        msg = f"{method} takes {accepted}; got {', '.join(unknown)}"
        raise ValueError(msg)
    out = dict(defaults)
    for key, value in params.items():
        kind = type(defaults[key])
        out[key] = value if isinstance(value, kind) else kind(value)
    return out


def _unsampled(graph: nx.Graph, crawl: _Crawl, rng: random.Random) -> Any:
    """A uniformly random node the crawl has not taken, or ``None`` when there is none left."""
    remaining = [node for node in graph if node not in crawl.seen]
    return rng.choice(remaining) if remaining else None


def _induced(graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]) -> _Crawl:
    """Node-induced sampling: uniformly random node ids, and what hangs off them (§29.1).

    The book's node-induced sample specifies a set of node ids and then *"collect[s] all
    information that is connected to the elements you selected"*, including their immediate
    neighbours. That is what ``neighbors=True`` does, and it is why §29.1 distinguishes an
    induced *sample* from an induced *subgraph* (§27.4): the sample gains nodes, the subgraph
    does not. It is off by default because it makes the returned node count a consequence of the
    seeds' degrees rather than the number asked for, and because the neighbours of random nodes
    are hubs (the friendship paradox, §31.4), which is a second bias on top of the one being
    measured.

    Either way the seeds are probed: we hold the population, so selecting a node means seeing its
    connections, which is exactly the "collect everything attached" the section describes.
    """
    opts = _accept(params, "induced", neighbors=False)
    crawl = _Crawl()
    seeds = rng.sample(list(graph), k=min(size, graph.number_of_nodes()))
    for node in seeds:
        crawl.take(node)
        crawl.probed(graph, node)
    crawl.seeds = len(seeds)
    if opts["neighbors"]:
        for node in seeds:
            for other in graph.neighbors(node):
                crawl.take(other)
    return crawl


def _edge(graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]) -> _Crawl:
    """Random edge induced sampling: uniformly random edges, and their neighbourhood (§29.1).

    *"You select edges at random and you collect all their direct neighbors."* ``size`` counts
    the endpoints drawn, and the neighbourhood collected around them is extra, so this sampler
    returns **more** than ``size`` nodes -- the same contract as ``induced`` with
    ``neighbors=True``, and the reason the section goes on to name the variants that do not do
    it. :func:`_ties` is the one of them implemented here.

    Every drawn endpoint is probed, because collecting all of its direct neighbours is exactly
    what knowing its degree means.
    """
    _accept(params, "edge")
    crawl = _draw_endpoints(graph, size, rng)
    for node in list(crawl.nodes):
        crawl.probed(graph, node)
        for other in graph.neighbors(node):
            crawl.take(other)
    return crawl


def _ties(graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]) -> _Crawl:
    """Totally Induced Edge Sampling: random edges, their endpoints, and the induction (§29.1).

    TIES is the first of the *"more sophisticated approaches"* the section names. It draws edges
    uniformly until it holds ``size`` distinct endpoints and stops there; the "totally induced"
    half of the name is the step :func:`sample` performs for every method here, adding back every
    population edge that runs between two sampled nodes. Against plain edge induction it trades
    the neighbourhood -- and with it the friendship-paradox helping of hubs, and a node count
    nobody chose -- for a sample of exactly the size asked for.

    Nothing is probed: drawing an edge tells you that edge exists, not how many others its
    endpoints have. That is the price of the budget being spent on edges instead of on nodes,
    and :func:`completion` reports it as an estimate rather than a count.
    """
    _accept(params, "ties")
    return _draw_endpoints(graph, size, rng)


def _draw_endpoints(graph: nx.Graph, size: int, rng: random.Random) -> _Crawl:
    """Uniformly random edges, kept for their endpoints, until ``size`` nodes are held.

    Edges are drawn without replacement and the last edge contributes only one endpoint when
    taking both would overshoot, so the node count is exact. An isolated node can never be drawn
    this way, so on a network with isolates the draw stops short of ``size`` -- a property of
    the method rather than a failure, and the record says ``truncated``.
    """
    crawl = _Crawl()
    edges = list(graph.edges())
    rng.shuffle(edges)
    for u, v in edges:
        for node in (u, v):
            if len(crawl.nodes) >= size:
                break
            crawl.take(node)
        if len(crawl.nodes) >= size:
            break
    crawl.seeds = len(crawl.nodes)
    return crawl


def _bfs(graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]) -> _Crawl:
    """Breadth-first crawl from a random seed (§29.2, §13.1).

    The neighbours of a node are visited in a shuffled order. Real crawls get them in whatever
    order the provider's database holds them, *"chosen with some criterion that is opaque to
    you"* (§29.4); shuffling is the neutral stand-in for an order we cannot know, and it stops
    the sample being a function of node insertion order.

    When the component runs out before ``size`` nodes are held, the crawl restarts at a random
    unsampled node. A single BFS would simply stop -- which is the book's picture of a crawl
    with a budget -- but a sampler that silently returns a tenth of what was asked for cannot be
    compared with one that does not, so the restarts are counted in the record instead.
    """
    _accept(params, "bfs")
    return _breadth_first(graph, size, rng, cap=None, burn=1.0)


def _snowball(graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]) -> _Crawl:
    """BFS that takes only ``k`` connections per node (§29.2).

    *"We start by taking an individual and asking her to reveal k of her connections. She might
    have more than k friends, but we only take k."* The k are drawn at random from the
    respondent's neighbours, for the same reason BFS shuffles them.

    Nothing here is probed, and that is the method rather than an omission: a node that named k
    friends has not revealed its degree. It is also why ``k`` is not the maximum degree of the
    sample -- *"a node might be mentioned by more than k neighbors"* -- and why
    :func:`completion` reports that it cannot estimate anything from a snowball sample.
    """
    opts = _accept(params, "snowball", k=3)
    cap = int(opts["k"])
    if cap < 1:
        msg = f"snowball needs k >= 1 connection per node, got {cap}"
        raise ValueError(msg)
    return _breadth_first(graph, size, rng, cap=cap, burn=1.0)


def _forest_fire(
    graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]
) -> _Crawl:
    """BFS that burns each neighbour with probability ``p`` (§29.2).

    *"Once we get all neighbors of a node, we do not explore them all. Instead, for each of them,
    we flip a coin and we explore the node only with probability p."* A neighbour that fails the
    test is skipped: not sampled, not queued, and not offered again, which is the book's default
    (*"usually, one won't try to visit again the neighbors that have been skipped"*) and the
    choice it says to make when *"the sample's topological properties are paramount"*.

    The node that was asked **is** probed: the fire asks for all of a node's connections and then
    declines to follow some of them, so its degree is known even though its neighbours are not in
    the sample. When the fire dies out the crawl restarts at a fresh unsampled seed rather than
    reviving a skipped node, so the budget is spent on new territory instead of re-treading.
    """
    opts = _accept(params, "forest-fire", p=0.5)
    burn = float(opts["p"])
    if not 0.0 < burn <= 1.0:
        msg = f"forest-fire needs a burning probability in (0, 1], got {burn}"
        raise ValueError(msg)
    return _breadth_first(graph, size, rng, cap=None, burn=burn)


def _breadth_first(
    graph: nx.Graph, size: int, rng: random.Random, *, cap: int | None, burn: float
) -> _Crawl:
    """The shared body of BFS, snowball and forest fire; they differ only in ``cap`` and ``burn``.

    ``cap`` limits how many neighbours a node reveals (snowball's k), ``burn`` is the probability
    that a revealed neighbour is followed (forest fire's p). Plain BFS is no cap and p = 1.
    A node that revealed its whole neighbour list is probed; a capped one is not.
    """
    crawl = _Crawl()
    queue: deque[Any] = deque()
    skipped: set[Any] = set()
    while len(crawl.nodes) < size:
        if not queue:
            start = _unsampled(graph, crawl, rng)
            if start is None:
                break
            crawl.seeds += 1
            crawl.take(start)
            queue.append(start)
            continue
        node = queue.popleft()
        if cap is None:
            crawl.probed(graph, node)
        neighbours = list(graph.neighbors(node))
        rng.shuffle(neighbours)
        revealed = neighbours if cap is None else neighbours[:cap]
        for other in revealed:
            if len(crawl.nodes) >= size:
                break
            if other in crawl.seen or other in skipped:
                continue
            if burn < 1.0 and rng.random() >= burn:
                skipped.add(other)
                continue
            crawl.take(other)
            queue.append(other)
    crawl.forced = max(0, crawl.seeds - 1)
    return crawl


def _random_walk(
    graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]
) -> _Crawl:
    """Vanilla random walk with teleportation (§29.3).

    *"We take an individual and we ask them to name one of their friends at random. Then we do
    the same with her and so on."* At every step the walker teleports to a uniformly random node
    with probability ``restart``, which is the section's answer to *"you might end up trapped in
    an area of the network where you already explored all nodes"*, *"just like PageRank does"*
    (§14.4). It is not an answer to the degree bias, which is structural: see :data:`BIASES`.

    Every visited node is probed -- the walker asks for all of a node's connections and then
    picks one -- so a walk sample carries the true degree of everything in it, which is what
    makes §29.5's estimate work on it.
    """
    opts = _accept(params, "random-walk", restart=0.15)
    restart = float(opts["restart"])
    if not 0.0 <= restart < 1.0:
        msg = f"random-walk needs a restart probability in [0, 1), got {restart}"
        raise ValueError(msg)
    return _walk(graph, size, rng, restart=restart, metropolis=False)


def _metropolis_hastings(
    graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]
) -> _Crawl:
    """Metropolis-Hastings random walk: the degree-corrected walk of §29.3.

    From ``v`` the walker picks a neighbour ``u`` uniformly and moves there with probability
    ``p = k_v / k_u``, refusing (and staying at ``v`` for that step) otherwise. *"If we were,
    instead, to transition from u to v, we would always accept the move, because 100/3 > 1"* --
    hence the ``min(1, ...)``. The stationary distribution is uniform, so this walk has no degree
    bias to report; what it has instead is rejected steps, many of them near a hub.

    Two departures from the clean chain, both recorded rather than hidden. It takes the **first
    visit** to each node rather than a thinned sample of the chain, because the object being
    built is a set of ``size`` nodes; the first-visit distribution is only uniform in the limit,
    so a short MHRW sample still carries a little of the walk's bias. And a walker that stalls
    is restarted at a uniformly random unsampled node -- a uniform restart, which is the
    distribution this walk is already aiming at.
    """
    _accept(params, "metropolis-hastings")
    return _walk(graph, size, rng, restart=0.0, metropolis=True)


def _walk(
    graph: nx.Graph, size: int, rng: random.Random, *, restart: float, metropolis: bool
) -> _Crawl:
    """The shared body of the two walks. ``metropolis`` turns on the ``k_v / k_u`` rejection."""
    crawl = _Crawl()
    nodes = list(graph)
    current = rng.choice(nodes)
    crawl.seeds = 1
    crawl.take(current)
    crawl.probed(graph, current)
    stalled = 0
    while len(crawl.nodes) < size:
        if stalled > STALL_LIMIT:
            # The walk has spent STALL_LIMIT steps without reaching anything new, so the budget
            # is going nowhere. Restart it at a uniformly random unsampled node rather than
            # bending the rule that got it stuck: a Metropolis-Hastings walk that accepted a
            # step it had just refused would not be a Metropolis-Hastings walk any more.
            forced = _unsampled(graph, crawl, rng)
            if forced is None:
                break
            crawl.forced += 1
            crawl.seeds += 1
            crawl.take(forced)
            crawl.probed(graph, forced)
            current = forced
            stalled = 0
            continue
        neighbours = list(graph.neighbors(current))
        if not neighbours or (restart > 0.0 and rng.random() < restart):
            current = rng.choice(nodes)
        else:
            candidate = rng.choice(neighbours)
            if metropolis:
                accept = min(1.0, graph.degree(current) / max(1, graph.degree(candidate)))
                if rng.random() >= accept:
                    stalled += 1
                    continue
            current = candidate
        if current in crawl.seen:
            stalled += 1
        else:
            stalled = 0
            crawl.take(current)
        crawl.probed(graph, current)
    return crawl


def _neighbor_reservoir(
    graph: nx.Graph, size: int, rng: random.Random, params: Mapping[str, Any]
) -> _Crawl:
    """Neighbor Reservoir Sampling: a walked core, then swaps that keep it connected (§29.3).

    Two phases, as the section gives them. *"Starting from the seed we provide as an input, NRS
    performs a normal random walk, including in the sample all nodes and edges it finds during
    this exploration"* -- that is the core, and it is :func:`_walk` with no teleport. Then *"the
    majority of NRS's budget is spent in the second phase"*: at each iteration pick a ``v``
    inside the sample and a ``u`` outside it but adjacent to it -- the reservoir -- and exchange
    them, so ``|V'|`` never changes.

    Both draws are uniform, *"you pick two nodes at random: u and v"*, and the reservoir is the
    **set** of nodes adjacent to the core: a node reachable from six core nodes is no likelier to
    be drawn than one reachable from one. Reading the adjacency lists as they come would instead
    draw in proportion to edges into the core, which on a heavy tail is a hub draw by another
    name -- on a 500-node preferential-attachment network it lifts the mean degree of the node
    drawn from 5.99 to 8.85. The reservoir is therefore de-duplicated, in the order the core was
    walked, which keeps the draw uniform over the set and the run reproducible.

    A swap happens only if both of the section's conditions hold. The first is connectivity:
    *"we cannot remove v if that would break the graph into multiple components"*, and ``u`` must
    still hang off the sample once ``v`` has gone. The second is the random test: draw
    ``0 < alpha < 1`` and swap only if ``alpha < |V'| / i``, where ``i`` starts at ``|V'|`` and
    rises by one each attempt, so *"at the beginning… all initial attempts to modify V' succeed.
    As we progress, the chances of accepting a new node in the set vanish."* ``i`` is raised
    before the first test rather than after it, so that first attempt is accepted with
    probability ``|V'| / (|V'| + 1)`` -- 0.98 on a 60-node core -- rather than with certainty,
    which is inside what the section means by all of them succeeding.

    ``rounds`` sets how long the second phase runs: there are ``rounds * |V'|`` attempts, so
    ``i`` ends at ``(1 + rounds) * |V'|`` and the last attempt is accepted with probability
    ``1 / (rounds + 1)``, one in eleven at the default. The book fixes no budget -- it says only
    that most of it goes here -- so this is the one number in this sampler that is ours, and the
    record carries the swaps that were accepted.

    Why the chapter argues about it: the connectivity condition does not fall equally on every
    node. *"The v with higher clustering have higher probability to be replaced, because by
    removing them it is more likely that the graph will stay connected"* -- a node whose
    neighbours are joined to each other leaves alternative paths behind it -- so high-clustering
    nodes churn out of the sample and *"NRS ensures a realistic clustering coefficient
    distribution"*.

    One departure, forced by the contract that every sampler returns ``size`` nodes: if the walk
    that builds the core had to restart, the core is in more than one component and the book's
    connectivity test would then refuse every swap forever. The test used is therefore that a
    swap may not *increase* the number of components, which is the same rule whenever the core is
    connected, and is the section's rule in every case the section considers.
    """
    opts = _accept(params, "neighbor-reservoir", rounds=10)
    rounds = int(opts["rounds"])
    if rounds < 1:
        msg = f"neighbor-reservoir needs rounds >= 1 pass over the core, got {rounds}"
        raise ValueError(msg)

    crawl = _walk(graph, size, rng, restart=0.0, metropolis=False)
    core = list(crawl.nodes)
    members = set(core)
    if len(members) < 2:
        return crawl
    components = _component_count(graph, members)
    i = len(members)
    for _ in range(rounds * len(members)):
        i += 1
        if rng.random() >= len(members) / i:
            continue
        # ``dict.fromkeys`` over the core's own order de-duplicates without sorting, so the
        # draw is uniform over the reservoir *set* and does not depend on how the nodes hash.
        reservoir = list(
            dict.fromkeys(
                other for node in core for other in graph.neighbors(node) if other not in members
            )
        )
        if not reservoir:
            break
        u = rng.choice(reservoir)
        index = rng.randrange(len(core))
        v = core[index]
        if not any(w != v and w in members for w in graph.neighbors(u)):
            continue
        after = members - {v} | {u}
        if _component_count(graph, after) > components:
            continue
        members = after
        core[index] = u
        crawl.take(u)
        crawl.swaps += 1
    # ``nodes`` is the discovery order and holds everything the crawl ever touched; the sample is
    # what survived the swaps.
    crawl.nodes = core
    crawl.seen = members
    return crawl


def _component_count(graph: nx.Graph, nodes: set[Any]) -> int:
    """How many components the induced subgraph on ``nodes`` falls into."""
    return int(nx.number_connected_components(graph.subgraph(nodes)))


#: Every sampler the chapter describes that can be run on a network we hold, by the name the
#: CLI takes. Re-weighted random walk (§29.3) is not here because it is not a sampler: it is a
#: correction applied to a vanilla walk's sample, and it lives in
#: :func:`reweighted_degree_distribution`.
SAMPLERS: dict[str, Sampler] = {
    "induced": _induced,
    "edge": _edge,
    "ties": _ties,
    "bfs": _bfs,
    "snowball": _snowball,
    "forest-fire": _forest_fire,
    "random-walk": _random_walk,
    "metropolis-hastings": _metropolis_hastings,
    "neighbor-reservoir": _neighbor_reservoir,
}

SAMPLE_METHODS: tuple[str, ...] = tuple(SAMPLERS)

#: Appended to the network's sampling frame by :func:`sample`, so every report that prints the
#: frame -- ``sna analyze`` included -- says what the numbers underneath it are numbers about.
SAMPLE_FRAME = (
    "Sampled with {method} to {nodes} of {population_nodes} nodes ({share:.1%}) and {edges} of "
    "{population_edges} edges, seed {seed}. {bias}. {completion} A measurement taken here is a "
    "measurement of the sampler as much as of the network (§29.4); `graphrag sna sample` prints "
    "the full bias table behind this sentence."
)


def sample(
    graph: nx.Graph, method: str, size: int, *, seed: int | None = None, **params: Any
) -> nx.Graph:
    """Take a smaller network out of this one, by one of the chapter's methods (ch. 29).

    Returns the **induced subgraph** on the sampled nodes: every population edge between two
    sampled nodes is in the result, which is what makes the sample a network rather than a node
    list, and which means the edges an edge-based sampler drew are in it by construction. Node
    attributes ride along; the nodes the crawl probed additionally carry :data:`CRAWL_DEGREE`,
    the degree they had in the population, which is what :func:`completion` reads.

    ``size`` is how many nodes to return, and every method returns exactly that many, with three
    stated exceptions recorded as ``truncated`` in the sample record: a population smaller than
    ``size``; edge-based sampling on a network with isolated nodes, which it can never reach; and
    a crawl that runs out of graph. Two samplers return *more* instead, because the book defines
    them that way: ``edge``, which collects the drawn endpoints' neighbours (§29.1), and
    ``induced`` with ``neighbors=True``. For both, ``size`` counts the seeds and ``seeds`` and
    ``nodes`` in the record are different numbers.

    ``params`` are the method's own: ``neighbors`` for ``induced``, ``k`` for ``snowball``
    (default 3, the book's figure), ``p`` for ``forest-fire`` (default 0.5, likewise),
    ``restart`` for ``random-walk`` (default 0.15, PageRank's) and ``rounds`` for
    ``neighbor-reservoir`` (default 10). An unknown one raises.

    ``seed`` makes a run reproducible: the same graph, method, size, seed and parameters give
    the same sample, node for node.

    The record lands on ``graph.graph["sample"]`` **as a JSON string**, not as a mapping. It has
    to: GraphML is the interchange format that carries ``graph.graph`` (§53.2) and its writer
    refuses nested values, so a sample that recorded its provenance as a dict would lose exactly
    that provenance on being written to the file it exists to be written to. :func:`sample_record`
    reads it back.

    Raises ``ValueError`` on an unknown method, a non-positive ``size``, or an empty graph --
    there is no sample of nothing.
    """
    if method not in SAMPLERS:
        msg = f"method must be one of {', '.join(SAMPLE_METHODS)}, got {method!r}"
        raise ValueError(msg)
    if size < 1:
        msg = f"--size must be at least 1 node, got {size}"
        raise ValueError(msg)
    if graph.number_of_nodes() == 0:
        msg = "cannot sample an empty network: there is nothing to take a sample of"
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    wanted = min(size, graph.number_of_nodes())
    crawl = SAMPLERS[method](graph, wanted, rng, params)
    sampled: nx.Graph = graph.subgraph(crawl.nodes).copy()
    for node, degree in crawl.degrees.items():
        if node in sampled:
            sampled.nodes[node][CRAWL_DEGREE] = degree

    bias = BIASES[method]
    record: dict[str, Any] = {
        "method": method,
        "section": bias.section,
        "chapter": 29,
        "size": size,
        "seed": seed,
        "params": {key: params[key] for key in sorted(params)},
        "population_nodes": graph.number_of_nodes(),
        "population_edges": graph.number_of_edges(),
        "nodes": sampled.number_of_nodes(),
        "edges": sampled.number_of_edges(),
        "seeds": crawl.seeds,
        "probed": len(crawl.degrees),
        "restarts": crawl.forced,
        "swaps": crawl.swaps,
        "truncated": len(crawl.nodes) < size,
        "bias": bias.sentence,
    }
    sampled.graph["sample"] = json.dumps(record, sort_keys=True)
    estimate = completion(sampled)
    sampled.graph["frame"] = " ".join(
        part
        for part in (
            str(graph.graph.get("frame", "")).strip(),
            SAMPLE_FRAME.format(
                method=method,
                nodes=record["nodes"],
                population_nodes=record["population_nodes"],
                share=record["nodes"] / max(1, record["population_nodes"]),
                edges=record["edges"],
                population_edges=record["population_edges"],
                seed="unset" if seed is None else seed,
                bias=bias.sentence,
                completion=estimate.sentence,
            ),
        )
        if part
    )
    return sampled


def sample_record(graph: nx.Graph) -> dict[str, Any] | None:
    """The sampling record :func:`sample` wrote, or ``None`` when this network is not a sample.

    Reads ``graph.graph["sample"]`` back from the JSON string it is stored as -- see
    :func:`sample` for why it is a string -- and tolerates a graph that never went through a
    sampler, because every caller here has to handle the unsampled case anyway.
    """
    raw = graph.graph.get("sample")
    if not raw:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    loaded = json.loads(str(raw))
    return dict(loaded) if isinstance(loaded, dict) else None


# ------------------------------------------------------------------ §29.3 re-weighting


def reweighted_degree_distribution(degrees: Iterable[int]) -> dict[int, float]:
    """The Re-Weighted Random Walk correction of a sampled degree distribution (§29.3).

    A vanilla random walk lands on a node with probability proportional to its degree, so the
    degree distribution you measure on the sample is the population's, tilted by one factor of
    the degree. RWRW untilts it after the fact, node by node:

        ``p_i = sum over the sampled nodes of degree i of 1/i, over the same sum of 1/k for every
        sampled node``

    The book's worked example (p. 422): of 100 sampled nodes, 50 have degree 1, 20 degree 2, 10
    degree 3, 8 degree 4, 7 degree 5 and 5 degree 6. The denominator is
    50/1 + 20/2 + 10/3 + 8/4 + 7/5 + 5/6 = 67.56..., the numerator for i = 2 is 20 x 1/2 = 10, and
    so ``p_2`` is 0.148 rather than the 0.20 the raw sample showed. Every method here is a
    *correction of the numbers*, not of the sample: *"if what you need was the sample rather than
    the estimation of a simple measure, you're out of luck"*.

    Returns a mapping from degree to corrected probability, summing to 1. Degree-zero nodes are
    dropped, since 1/0 is where the formula stops being defined -- an isolated node cannot be
    reached by a walk in the first place, so it is not in the sample the formula is written for.
    Applied to a sample that was not taken by a random walk, this is arithmetic rather than a
    correction: it is the right estimator only for the degree-proportional sampler §29.3
    describes. Undefined, and returns ``{}``, when no sampled node has a positive degree.
    """
    counts = Counter(int(d) for d in degrees if int(d) > 0)
    denominator = sum(count / degree for degree, count in counts.items())
    if denominator <= 0:
        return {}
    return {degree: (count / degree) / denominator for degree, count in sorted(counts.items())}


# ------------------------------------------------------------------ §29.4 sampling bias


@dataclass(frozen=True)
class BiasCheck:
    """One measure, in the population and in the sample, against what the book predicted."""

    measure: str
    population: float
    sample: float
    #: ``up``, ``down``, ``none``, or ``""`` when the chapter predicts nothing here.
    predicted: str
    observed: str
    #: ``None`` when there was no prediction to agree with.
    agrees: bool | None
    section: str
    note: str


@dataclass(frozen=True)
class SampleBias:
    """What one sample did to the network's measurements, next to what §29.1-29.3 said it would.

    ``selected_degrees`` is the part that isolates the *sampler*: the degrees the sampled nodes
    have **in the population**, which is what "oversamples hubs" means. ``sample_degrees`` is
    what somebody measuring the sample would see, and it is always lower, because every edge to
    an unsampled node is gone -- that is arithmetic, not bias, and it is graded as such.
    """

    method: str
    section: str
    prediction: str
    population_nodes: int
    population_edges: int
    sample_nodes: int
    sample_edges: int
    population_degrees: Summary
    selected_degrees: Summary
    sample_degrees: Summary
    checks: tuple[BiasCheck, ...]
    rank_agreement: Correlation | None
    frame: str

    @property
    def share(self) -> float:
        """n / N: the share of the population's nodes this sample holds."""
        return self.sample_nodes / self.population_nodes if self.population_nodes else 0.0

    @property
    def disagreements(self) -> tuple[BiasCheck, ...]:
        """The checks where the book predicted a direction and this sample went another way."""
        return tuple(check for check in self.checks if check.agrees is False)


def _direction(population: float, observed: float, tolerance: float) -> str:
    """Which way the sample moved, with a band around the population value for "not at all"."""
    if population == 0:
        return "none" if observed == 0 else "up"
    change = (observed - population) / abs(population)
    if abs(change) < tolerance:
        return "none"
    return "up" if change > 0 else "down"


def bias_report(
    population: nx.Graph, sampled: nx.Graph, *, tolerance: float = DIRECTION_TOLERANCE
) -> SampleBias:
    """Compare a sample with the network it came from, per the directions the book predicts.

    The comparison is possible here only because we sampled a network we hold. In the chapter's
    own setting -- crawling somebody else's API -- the population column does not exist, which is
    why the report labels it as known rather than estimated and why there is no null model in
    this section: the reference is the population itself, not a random graph.

    Four measures are graded against :data:`BIASES`:

    *Mean degree of the sampled nodes, measured in the population.* The one that isolates the
    sampler's choice of nodes from the arithmetic of cutting edges. This is what §29.1's
    "unlikely to fairly represent the hubs" and §29.3's degree-proportional stationary
    distribution are statements about.

    *Average clustering*, which §29.2 says BFS overestimates and forest fire does not.

    *Largest component share*, which §29.1 says node induction destroys -- *"the resulting
    node-induced graph breaks down in multiple components"* -- and which every crawl keeps at 1
    by construction, so a crawl's connectivity is evidence about the crawler and none at all
    about the network.

    *Mean degree inside the sample* and *density* are reported without a grade: the first is
    always lower than the population's because every edge leaving the sample is gone, and the
    chapter makes no prediction about the second. Forest fire's clustering row is ungraded for a
    third reason -- §29.2's claim for it is comparative, better than the BFS it varies, not equal
    to the population -- and every row carries the ``note`` that says which of these it is, into
    the rendered table as well as the JSON.

    ``tolerance`` is the band, as a share of the population value, inside which a difference
    counts as no direction at all. ``rank_agreement`` is Spearman's correlation (§3.4) between
    each sampled node's degree in the sample and its degree in the population: how far the
    sample preserves the *ranking* even where it distorts the values, and ``None`` when fewer
    than three nodes make that undefined.

    Raises ``ValueError`` if the sample holds a node the population does not, or if it carries no
    sampling record to name a method.
    """
    record = sample_record(sampled)
    if record is None:
        msg = "bias_report() needs a sample built by sample(); this network carries no record"
        raise ValueError(msg)
    method = str(record["method"])
    missing = [node for node in sampled if node not in population]
    if missing:
        msg = (
            f"the sample holds {len(missing)} node(s) the population does not, so it is not a "
            f"sample of this network (first: {missing[0]!r})"
        )
        raise ValueError(msg)

    bias = BIASES[method]
    population_stats = summary(population)
    sample_stats = summary(sampled)
    population_degrees = [int(d) for _, d in population.degree()]
    selected = [int(population.degree(node)) for node in sampled]
    inside = [int(d) for _, d in sampled.degree()]
    mean_population = sum(population_degrees) / len(population_degrees) if population_degrees else 0
    mean_selected = sum(selected) / len(selected) if selected else 0.0
    mean_inside = sum(inside) / len(inside) if inside else 0.0

    checks = (
        _check(
            "mean degree of the sampled nodes, in the population",
            mean_population,
            mean_selected,
            bias.degree,
            bias.section,
            tolerance,
            "which nodes the sampler chose, with the edges it cut left out of it.",
        ),
        _check(
            "mean degree inside the sample",
            mean_population,
            mean_inside,
            "",
            "§29.1",
            tolerance,
            "what a reader of the sample alone would measure. Lower by construction: every "
            "edge to an unsampled node is gone, so this is not graded.",
        ),
        _check(
            "density",
            population_stats["density"],
            sample_stats["density"],
            "",
            "§12.1",
            tolerance,
            "density falls with size on its own (§12.1), so a sample's density is not "
            "comparable with a population's and the chapter predicts nothing about it.",
        ),
        _check(
            "average clustering",
            population_stats["average_clustering"],
            sample_stats["average_clustering"],
            bias.clustering,
            "§29.2",
            tolerance,
            "the measure §29.2 says a full-neighbourhood crawl overestimates and forest fire "
            "was designed to get right.",
        ),
        _check(
            "largest component share",
            population_stats["largest_component_share"],
            sample_stats["largest_component_share"],
            bias.connectivity,
            "§29.1",
            tolerance,
            "a crawl's sample is connected because a crawl walks edges, so 1.0 here is a fact "
            "about the sampler; induced sampling shatters it instead.",
        ),
    )
    agreement = (
        spearman(inside, selected) if len(selected) >= 3 and len(set(selected)) > 1 else None
    )
    return SampleBias(
        method=method,
        section=bias.section,
        prediction=bias.sentence,
        population_nodes=population.number_of_nodes(),
        population_edges=population.number_of_edges(),
        sample_nodes=sampled.number_of_nodes(),
        sample_edges=sampled.number_of_edges(),
        population_degrees=describe(population_degrees) if population_degrees else describe([0]),
        selected_degrees=describe(selected) if selected else describe([0]),
        sample_degrees=describe(inside) if inside else describe([0]),
        checks=checks,
        rank_agreement=agreement,
        frame=str(sampled.graph.get("frame", "")),
    )


def _check(
    measure: str,
    population: float,
    observed: float,
    predicted: str,
    section: str,
    tolerance: float,
    note: str,
) -> BiasCheck:
    """One row of the bias table, graded only where the book made a prediction."""
    direction = _direction(population, observed, tolerance)
    return BiasCheck(
        measure=measure,
        population=float(population),
        sample=float(observed),
        predicted=predicted,
        observed=direction,
        agrees=None if not predicted else direction == predicted,
        section=section,
        note=note,
    )


# ------------------------------------------------------------------ §29.5 completion


@dataclass(frozen=True)
class ProbeTarget:
    """One node worth asking about next, and how much new the ask is expected to return."""

    node: str
    sample_degree: int
    estimated_degree: float
    #: MaxReach's score: estimated true degree minus degree in the sample.
    score: float


@dataclass(frozen=True)
class Completion:
    """How much of the network the sample did not see, and where to look next (§29.5).

    ``known_missing_edges`` is not an estimate: it is the number of edges the crawl *saw* leaving
    the sample, from the probed nodes' true degrees. ``estimated_missing_edges`` adds the
    estimate for the nodes that were never probed. Both count only edges with an endpoint in the
    sample; a part of the network the crawl never touched is invisible here, so ``completeness``
    is an upper bound on how complete the sample is.
    """

    method: str
    section: str
    nodes: int
    edges: int
    probed: int
    unprobed: int
    known_missing_edges: int
    estimated_missing_edges: float
    #: ``None`` when nothing in the sample was probed, so nothing can be scaled against a known
    #: degree -- a snowball sample, in other words.
    completeness: float | None
    #: The factor a probed node's sample degree is multiplied by to estimate a true degree.
    ratio: float | None
    observed_mean_degree: float
    corrected_mean_degree: float
    corrected_degree_distribution: dict[int, float]
    probes: tuple[ProbeTarget, ...]
    sentence: str


def completion(
    sampled: nx.Graph, method: str | None = None, *, top: int = TOP_PROBES
) -> Completion:
    """Estimate what the sample missed, and rank what to probe next (§29.5).

    §29.5 is a survey, not a formula: it names MaxReach and ε-wgx and describes the first as
    estimating *"the true degree of a node and its clustering coefficient using the information
    gathered so far… with a technique similar to Re-Weighted Random Walk"*, scoring each node by
    *"the difference between its estimated degree and its degree in the sample"*, and probing the
    highest scores first. The formula is in the cited paper rather than in the book, so what is
    implemented here is the simplest estimator the text itself supports, and it is stated rather
    than dressed up:

    - A node the crawl **probed** carries its true degree (:data:`CRAWL_DEGREE`). For it, the
      section's own two objections do not apply -- we know its degree, and we know we have seen
      all of its neighbours -- so its score is zero: *"since the node has a high degree in your
      sample, there's some chance you already explored all its neighbors, thus probing it won't
      help you"*. The edges it has outside the sample are counted exactly.
    - A node the crawl only **discovered** has a sample degree and no true one. Its true degree
      is estimated as its sample degree times the ratio, over the probed nodes, of true degree to
      sample degree. That is a ratio estimator, not MaxReach's model; it produces the same
      ordering as "the highest-degree node you have not probed yet", which is §29.5's naive
      strategy restricted to the unprobed nodes -- the half of the section's warning the text
      supports without its citation.

    ``corrected_degree_distribution`` is §29.3's RWRW correction of the sample's own degrees,
    reported beside the raw mean because on a walk-based sample the two differ and the corrected
    one is the estimate of the population's.

    When nothing was probed -- snowball, which never learns a degree, or edge sampling without
    ``neighbors`` -- there is no ratio and no completeness, and the sentence says so instead of
    printing a number with nothing behind it.
    """
    record = sample_record(sampled)
    name = method or (str(record["method"]) if record else "unknown")
    degrees = {node: int(d) for node, d in sampled.degree()}
    probed = {
        node: int(value) for node, value in sampled.nodes(data=CRAWL_DEGREE) if value is not None
    }
    known_missing = sum(probed[node] - degrees[node] for node in probed)
    inside = sum(degrees[node] for node in probed)
    ratio = (sum(probed.values()) / inside) if probed and inside > 0 else None

    unprobed = [node for node in sampled if node not in probed]
    probes: list[ProbeTarget] = []
    estimated_missing = float(known_missing)
    for node in unprobed:
        estimate = degrees[node] * ratio if ratio is not None else float(degrees[node])
        score = max(0.0, estimate - degrees[node])
        estimated_missing += score
        probes.append(ProbeTarget(str(node), degrees[node], float(estimate), score))
    probes.sort(key=lambda p: (-p.score, str(p.node)))

    edges = sampled.number_of_edges()
    completeness = (
        edges / (edges + estimated_missing)
        if ratio is not None and edges + estimated_missing > 0
        else None
    )
    observed_mean = sum(degrees.values()) / len(degrees) if degrees else 0.0
    corrected = reweighted_degree_distribution(degrees.values())
    corrected_mean = sum(degree * p for degree, p in corrected.items())
    return Completion(
        method=name,
        section="§29.5",
        nodes=sampled.number_of_nodes(),
        edges=edges,
        probed=len(probed),
        unprobed=len(unprobed),
        known_missing_edges=known_missing,
        estimated_missing_edges=estimated_missing,
        completeness=completeness,
        ratio=ratio,
        observed_mean_degree=observed_mean,
        corrected_mean_degree=corrected_mean,
        corrected_degree_distribution=corrected,
        probes=tuple(probes[:top]),
        sentence=_completion_sentence(name, completeness, known_missing, estimated_missing),
    )


def _completion_sentence(
    method: str, completeness: float | None, known: int, estimated: float
) -> str:
    """The completion estimate in one sentence, for the frame every sampled report prints."""
    if completeness is None:
        return (
            f"How much the sample missed cannot be estimated from it: {method} never learns any "
            "node's true degree, so there is nothing in the sample to scale an estimate against "
            "(§29.5)."
        )
    return (
        f"It holds an estimated {completeness:.0%} of the edges that touch it -- {known} more "
        f"were seen leaving it and about {estimated - known:.0f} more are estimated from the "
        "nodes the crawl never probed; edges between two unsampled nodes are invisible to this "
        "estimate, so it is an upper bound (§29.5)."
    )


# ------------------------------------------------------------------ rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _verdict(check: BiasCheck) -> str:
    if check.agrees is None:
        return "not predicted"
    return "as the book predicts" if check.agrees else "against the book's prediction"


def render_sample(bias: SampleBias, estimate: Completion) -> str:
    """The sampling sections of a report: what the sampler did, and what it did not see.

    Prints, next to the numbers and in this order, the four things a reader needs in order to
    disagree: the sampling frame, n against N, the null model (there is none, and why), and the
    chapter the section implements.
    """
    lines: list[str] = [
        f"## Sampling ({bias.method}, {bias.section})",
        "",
        f"**Sampling frame.** {bias.frame}",
        "",
        f"**n.** {bias.sample_nodes:,} of {bias.population_nodes:,} nodes "
        f"({bias.share:.1%}) and {bias.sample_edges:,} of {bias.population_edges:,} edges.",
        "",
        "**Null model.** None: every row below is compared against the population this sample "
        "was drawn from, which is known here because the network being sampled is one we hold. "
        "In the chapter's own setting -- crawling an API -- that column does not exist.",
        "",
        "**Chapter.** Atlas ch. 29, *Network Sampling*: the samplers of §29.1-29.3, the biases "
        f"of §29.4, the completion estimate of §29.5. This sampler: {bias.section}.",
        "",
        f"**What the book predicts.** {bias.prediction}.",
        "",
    ]
    lines += _table(
        ["measure", "population", "sample", "predicted", "observed", "verdict", "§", "note"],
        [
            [
                check.measure,
                _num(check.population),
                _num(check.sample),
                check.predicted or "-",
                check.observed,
                _verdict(check),
                check.section,
                check.note,
            ]
            for check in bias.checks
        ],
    )
    if bias.rank_agreement is not None:
        lines += [
            f"**Degree ranking.** Spearman {bias.rank_agreement.coefficient:.3f} "
            f"(p={bias.rank_agreement.p_value:.3g}, n={bias.rank_agreement.n}) between each "
            "sampled node's degree inside the sample and its degree in the population: how far "
            "the sample keeps the order even where it moves the values (§3.4).",
            "",
        ]
    lines += [
        "**Degree distributions.** Population: "
        f"{bias.population_degrees.caveat} Sampled nodes, in the population: mean "
        f"{bias.selected_degrees.mean:.3f}, median {bias.selected_degrees.median:.1f}, max "
        f"{bias.selected_degrees.maximum:.0f}. Inside the sample: mean "
        f"{bias.sample_degrees.mean:.3f}, median {bias.sample_degrees.median:.1f}, max "
        f"{bias.sample_degrees.maximum:.0f}.",
        "",
        f"### Completion estimate ({estimate.section})",
        "",
        estimate.sentence,
        "",
        f"**n.** {estimate.probed:,} of {estimate.nodes:,} sampled nodes were probed (their true "
        f"degree is known); {estimate.unprobed:,} were only discovered.",
        "",
    ]
    if estimate.ratio is not None:
        lines += [
            f"A probed node holds {1 / estimate.ratio:.1%} of its true degree inside the sample, "
            f"so an unprobed node's true degree is estimated at {estimate.ratio:.2f} times its "
            "degree here. Mean degree in the sample "
            f"{estimate.observed_mean_degree:.3f}; re-weighted per §29.3, "
            f"{estimate.corrected_mean_degree:.3f} -- the estimate of the population's, and the "
            "right one to quote only if this sample came from a random walk.",
            "",
        ]
    if estimate.probes:
        lines += ["**Probe these next (MaxReach order, §29.5).**", ""]
        lines += _table(
            ["node", "degree here", "estimated true degree", "score"],
            [
                [
                    probe.node,
                    str(probe.sample_degree),
                    f"{probe.estimated_degree:.2f}",
                    f"{probe.score:.2f}",
                ]
                for probe in estimate.probes
            ],
        )
    return "\n".join(lines)


def sample_payload(bias: SampleBias, estimate: Completion) -> dict[str, Any]:
    """The same two sections as JSON, for a caller that would rather read numbers than prose."""
    return {
        "chapter": 29,
        "method": bias.method,
        "section": bias.section,
        "frame": bias.frame,
        "null_model": (
            "none: the reference is the population the sample was drawn from, which is known "
            "only because we hold the network being sampled"
        ),
        "population": {"nodes": bias.population_nodes, "edges": bias.population_edges},
        "sample": {"nodes": bias.sample_nodes, "edges": bias.sample_edges, "share": bias.share},
        "prediction": bias.prediction,
        "checks": [
            {
                "measure": check.measure,
                "population": check.population,
                "sample": check.sample,
                "predicted": check.predicted,
                "observed": check.observed,
                "agrees": check.agrees,
                "section": check.section,
                "note": check.note,
            }
            for check in bias.checks
        ],
        "degree_rank_spearman": (
            None
            if bias.rank_agreement is None
            else {
                "coefficient": bias.rank_agreement.coefficient,
                "p_value": bias.rank_agreement.p_value,
                "n": bias.rank_agreement.n,
            }
        ),
        "completion": {
            "section": estimate.section,
            "probed": estimate.probed,
            "unprobed": estimate.unprobed,
            "known_missing_edges": estimate.known_missing_edges,
            "estimated_missing_edges": estimate.estimated_missing_edges,
            "completeness": estimate.completeness,
            "ratio": estimate.ratio,
            "observed_mean_degree": estimate.observed_mean_degree,
            "corrected_mean_degree": estimate.corrected_mean_degree,
            "corrected_degree_distribution": {
                str(degree): p for degree, p in estimate.corrected_degree_distribution.items()
            },
            "probes": [
                {
                    "node": probe.node,
                    "sample_degree": probe.sample_degree,
                    "estimated_degree": probe.estimated_degree,
                    "score": probe.score,
                }
                for probe in estimate.probes
            ],
            "sentence": estimate.sentence,
        },
    }
