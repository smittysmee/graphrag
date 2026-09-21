"""Synthetic networks (ch. 16-18), and the expectations they set for an observed one (ch. 17).

Two halves, and the second is the reason for the first.

**The models.** Chapter 16 opens with the observation that a network analysis needs a network,
and that when there was no data "some of the most brilliant mathematical minds in the field
determined different ways to create synthetic network data by defining network models". The book
gives three reasons to keep generating them: to *explain* how a property might arise (ch. 17), to
*test* an algorithm against data whose answer is known (ch. 18), and to *describe* -- to ask
whether an observed property "is something you'd expect any network to have, given other
characteristics on its topology" (ch. 19). This module holds one thin, seeded wrapper per model
the book names: G(n,p) and G(n,m) (§16.1), the cavemen graph (§17.1), Watts-Strogatz (§17.2),
Barabasi-Albert preferential attachment (§17.3), the Molloy-Reed configuration model (§18.1),
the planted partition / stochastic block model and the LFR benchmark (§18.2), and the random
geometric graph (§18.3). :data:`GENERATORS` says in one line what each one reproduces and what
it fails to reproduce, because a model is chosen for what it gets wrong as much as for what it
gets right.

Every generated graph is stamped with its model, its parameters, its seed and a ``frame`` saying
it is synthetic. That is not decoration: every report in this package prints the sampling frame
next to the numbers, and a generated network's honest frame is "nothing here was observed".

**The expectations.** Chapter 17's whole argument is that real networks are *clustered, short and
broad* only by comparison: clustering is high compared with p (§16.5, §17.1), the diameter is
small compared with ln|V| / ln k̄ (§16.4, §17.2), and the degree distribution is broad compared
with a Poisson (§16.2, §17.3). :func:`expected_properties` computes those closed forms for the
observed n and m, and :func:`against_random` puts them beside the observed numbers and beside a
degree-preserving null sample, so that "this network is clustered" is never printed as a number
on its own. :func:`render_against_random` is the report section, and every ``sna analyze`` report
carries it.

That section is not free, and the cost is the null sample rather than the closed forms: per
rewiring it measures one average clustering coefficient and one breadth-first search from
:data:`PATH_SOURCES` sources, and the rewiring itself -- §19.1's edge swap, once per edge -- costs
about as much again. The report therefore draws **one** family of rewirings and spends it on both
this section and Louvain's modularity null (:func:`graphrag.sna.cluster.rewired_modularity`),
which is what ``against_random(..., rewirings=...)`` is for; on a network of a few thousand nodes
and tens of thousands of edges, expect the section to add tens of seconds at the default
``--samples``, and lower ``--samples`` if that is too slow to iterate with.

What this module does **not** do is fit anything. §17.3's power-law exponent is a claim about a
distribution, and the book is careful that the preferential attachment model *produces* alpha = 3
rather than that a broad distribution *implies* a power law. "Broad" here means "wider than the
Poisson with the same mean", measured by the variance-to-mean ratio; whether it is a power law
is chapter 9's question and ``sna degree``'s answer.

Three models the chapters name are documented rather than built, each for the same reason --
nothing in this package consumes one, and an unused generator is an untested one. §17.3's **link
selection** and **copying** models are the two alternatives to preferential attachment ("no
matter which link you select, the central hub is connected to it"); they are dynamic growth
models with no ``networkx`` generator, and they produce the same broad distribution
:func:`barabasi_albert` already provides for a test, differing in the mechanism rather than in
the result. §18.2's **Kronecker graphs** could be built from ``nx.tensor_product`` applied
repeatedly to a seed adjacency matrix, so the obstacle is not the library; it is that §18.2 puts
LFR ahead of them anyway ("LFR is preferred because it leaves space for the randomness of real
world noise"), and this package has one consumer of a benchmark -- community discovery -- which
:func:`lfr_benchmark` already serves. §18.2's mixed-membership and degree-corrected stochastic
block models are named in :func:`planted_partition`'s docstring and belong with the community
tickets that would use them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import networkx as nx
import numpy as np
from scipy import stats as scipy_stats
from scipy.sparse import csgraph

from graphrag.sna.matrices import adjacency
from graphrag.sna.measures import undirected_view
from graphrag.sna.null import HOLDS_FIXED, Significance, configuration, significance
from graphrag.sna.stats import MIN_TAIL_SAMPLE, Summary, describe

__all__ = [
    "BROAD_DISPERSION",
    "CLUSTERED_RATIO",
    "GENERATORS",
    "PATH_SOURCES",
    "SHORT_RATIO",
    "SIGNIFICANT_Z",
    "AgainstRandom",
    "ExpectedProperties",
    "against_random",
    "against_random_payload",
    "barabasi_albert",
    "caveman",
    "configuration_model",
    "erdos_renyi_gnm",
    "erdos_renyi_gnp",
    "expected_properties",
    "lfr_benchmark",
    "planted_partition",
    "poisson_degrees",
    "random_geometric",
    "render_against_random",
    "watts_strogatz",
]


#: What each model reproduces, and what it does not, in one line -- the sentence a report prints
#: when it names the model it compared something against. The failures are the point: §16.5 sums
#: the random graph up as getting giant components and short paths right and "three of the most
#: crucial tests" wrong, and every later model in the chapter exists because of one of those.
GENERATORS: dict[str, str] = {
    "erdos_renyi_gnp": (
        "every pair of n nodes tossed independently at probability p (§16.1): a binomial degree "
        "distribution, clustering equal to p, short paths, a giant component above k̄ = 1 -- and "
        "no hubs, no clustering worth the name, no communities"
    ),
    "erdos_renyi_gnm": (
        "one graph drawn uniformly from all graphs with n nodes and m edges (§16.1): the same "
        "properties as G(n,p) at p = 2m/(n(n-1)), with the edge count fixed rather than expected"
    ),
    "caveman": (
        "a ring of cliques joined by one rewired edge each (§17.1): the clustering and the "
        "communities G(n,p) cannot produce, bought with a long diameter and a degree "
        "distribution with two values in it"
    ),
    "watts_strogatz": (
        "a ring lattice with each edge rewired at probability p (§17.2): high clustering *and* "
        "short paths at small p, because 'even a tiny bridge probability can connect parts of "
        "the network that are very far away' -- but no hubs and no communities"
    ),
    "barabasi_albert": (
        "growth by preferential attachment, m edges per new node (§17.3): a power-law degree "
        "distribution with exponent 3 in the limit and short paths through the hubs, with "
        "clustering 'a far cry from' real networks and an age-degree correlation real networks "
        "do not have"
    ),
    "configuration_model": (
        "a graph forced to have a given degree sequence (§18.1): the degree distribution exactly "
        "-- that is the input -- and everything else random, so the triangles fall where they "
        "fall and the clustering tends to zero for the usual exponents"
    ),
    "planted_partition": (
        "a stochastic block model with equal blocks (§18.2): communities by construction, at "
        "p_in inside a block and p_out between them, so a community-discovery method can be "
        "scored against an answer known before it ran"
    ),
    "lfr_benchmark": (
        "the LFR benchmark (§18.2): a power-law degree distribution *and* power-law community "
        "sizes *and* a tunable mixing parameter mu, which makes it 'the most realistic model we "
        "have' for testing community discovery"
    ),
    "random_geometric": (
        "n points dropped uniformly in a d-dimensional space, joined when they are within radius "
        "r (§18.3): the high clustering that space forces on a network, with long paths, because "
        "a neighbour of a neighbour is nearby too"
    ),
}


def _stamp(graph: nx.Graph, model: str, parameters: dict[str, Any], seed: int | None) -> nx.Graph:
    """Record what made this graph, and that it is a model rather than a record of anything.

    Every report in this package prints ``frame`` next to its numbers, so a synthetic graph that
    carried a corpus-shaped frame -- or none -- could be read as data by a reader two steps
    downstream. The honest frame is the one written here.
    """
    detail = ", ".join(f"{key}={value}" for key, value in parameters.items())
    graph.graph.update(
        model=model,
        parameters=dict(parameters),
        seed=seed,
        network=model,
        min_weight=1,
        frame=(
            f"A synthetic network generated by the `{model}` model ({detail}, seed={seed}). "
            "Nothing in it was observed: it records a process somebody specified, so no number "
            "computed on it describes a corpus, a population or a person (Atlas ch. 16)."
        ),
    )
    return graph


# --------------------------------------------------------------------- §16.1 G(n,p), G(n,m)


def erdos_renyi_gnp(n: int, p: float, *, seed: int | None = None) -> nx.Graph:
    """The G(n,p) random graph: n nodes, every pair connected at probability p (§16.1).

    "Given a pair of nodes we toss the coin and if it lands on heads we connect them. Another
    pair, same procedure. The two tosses are independent." The properties the chapter derives
    from that, and which :func:`expected_properties` computes for an observed network:

    - mean degree ``p(n-1)``, binomial and, at large n, Poisson (§16.2);
    - clustering equal to ``p`` for every node, hence also on average (§16.5) -- "much lower than
      the one of real world networks";
    - a giant component once the mean degree passes 1, i.e. ``p > 1/n``, and no node outside it
      once ``p > ln n / n`` (§16.3);
    - average path length ``ln n / ln(np)`` (§16.4).

    The edge count is not fixed: it varies around ``p·n(n-1)/2``, which is the chapter's own
    conversion to :func:`erdos_renyi_gnm`. Raises ``ValueError`` for a ``p`` outside [0, 1].
    """
    if not 0.0 <= p <= 1.0:
        msg = f"p must be a probability in [0, 1], got {p!r}"
        raise ValueError(msg)
    graph: nx.Graph = nx.gnp_random_graph(n, p, seed=seed)
    return _stamp(graph, "erdos_renyi_gnp", {"n": n, "p": p}, seed)


def erdos_renyi_gnm(n: int, m: int, *, seed: int | None = None) -> nx.Graph:
    """The G(n,m) random graph: one graph drawn uniformly from all graphs with n nodes, m edges.

    §16.1's "Big Box of Graphs": "you go and take all possible graphs with n nodes and m edges,
    and you pick one at random". Mathematically equivalent to :func:`erdos_renyi_gnp` at
    ``p = 2m/(n(n-1))`` -- "the difference is there simply for convenience in what you want to
    fix" -- and this is the one to use as a baseline for an observed network, because it matches
    both counts exactly.

    Sampling a *null* for an observed network goes through
    :func:`graphrag.sna.null.erdos_renyi` instead, which draws the same way but keeps the
    observed node ids and can deal the observed weights over the result. This one is for
    building a synthetic network from parameters, where there is no observation to keep.
    """
    graph: nx.Graph = nx.gnm_random_graph(n, m, seed=seed)
    return _stamp(graph, "erdos_renyi_gnm", {"n": n, "m": m}, seed)


# ------------------------------------------------------------------- §17.1-17.3 explanations


def caveman(cliques: int, size: int) -> nx.Graph:
    """The connected cavemen graph: a ring of cliques, one edge rewired between neighbours (§17.1).

    Watts's tribes: "everybody knew everyone else in their cave, but between caves there was
    almost no communication". Each cave is a clique of ``size`` members and each elects a member
    to connect to the next cave along the ring, so the graph has one component by construction.

    What it is *for* is the thing G(n,p) cannot give: clustering and communities. "You can't get
    anything clearer than a clique with just two edges pointing outwards", which is why it is the
    graph a community-discovery method is sanity-checked on. What it gets wrong is everything
    else -- the diameter is long ("worse than Gn,p in approximating realistic diameters") and the
    degree distribution has two values in it, one for cave members and one for the emissaries,
    "needless to say... isn't found anywhere in natural networks".

    Deterministic: ``networkx``'s construction rewires a fixed edge per clique, so there is no
    seed to fix and two calls with the same parameters are the same graph.
    """
    graph: nx.Graph = nx.connected_caveman_graph(cliques, size)
    return _stamp(graph, "caveman", {"cliques": cliques, "size": size}, None)


def watts_strogatz(n: int, k: int, p: float, *, seed: int | None = None) -> nx.Graph:
    """The small-world model: a ring lattice of degree k, each edge rewired at probability p.

    §17.2's two steps: "you put nodes into a single dimensional space and connect them with their
    k nearest neighbours", then "for each edge you toss a coin: if it lands on heads you delete
    the edge and you rewire it by picking a random destination".

    The point of the model is that the two properties move at different speeds. Clustering
    survives a small p, because "each connection that we don't rewire will create triangles with
    some of the neighbours of the connected nodes"; path length collapses at the same small p,
    because "even a tiny bridge probability can connect parts of the network that are very far
    away. Thousands of shortest paths will use it". At p near 1 "randomness overcomes every other
    feature of the model, and the result would be almost indistinguishable from a Gn,p model".

    Two things it still does not have: hubs (the degree distribution is "very weird", nearly
    constant at low p) and communities -- "the triangles are distributed everywhere uniformly in
    the network. There are no discontinuities in the density".

    ``k`` is the number of nearest neighbours, so it must be even and less than ``n``; k = 2
    leaves no triangles at all and the clustering is zero (§17.2, n. 4). ``networkx`` raises
    ``NetworkXError`` when k is out of range.
    """
    graph: nx.Graph = nx.watts_strogatz_graph(n, k, p, seed=seed)
    return _stamp(graph, "watts_strogatz", {"n": n, "k": k, "p": p}, seed)


def barabasi_albert(n: int, m: int, *, seed: int | None = None) -> nx.Graph:
    """Preferential attachment: grow to n nodes, each newcomer bringing m edges (§17.3).

    "Each time you add a new node, it will connect to m of the old ones... It will connect at
    random, but preferentially to nodes with higher degree", with probability proportional to a
    node's degree over twice the number of edges. The rich get richer, and the chapter's promise
    is a power-law degree distribution with **exponent 3** in the limit: "this is independent
    from the m parameter, meaning that you cannot tune it to obtain different exponents... as
    |V| -> infinity, then alpha -> 3. If you want to reproduce a real world network with
    alpha = 2, you cannot use
    the basic preferential attachment model."

    Paths are short -- the diameter grows as ``log n / log log n``, slower than a random graph's
    -- because "hubs with thousands of connections can be used as shortcuts". Clustering is
    "higher than a random Gn,p one, but not by much... still a far cry from the clustering levels
    you see in real world networks", and there are no communities, "because everything connects
    to hubs which make up a single core". The model also plants an artefact: a positive
    correlation between a node's age and its degree, which §17.3 notes real systems do not show.

    The seed is the m-node clique ``networkx`` starts from; the chapter leaves the initial
    condition open and says "after enough steps, what you started from doesn't matter", with the
    one constraint that it must hold at least m nodes.
    """
    graph: nx.Graph = nx.barabasi_albert_graph(n, m, seed=seed)
    return _stamp(graph, "barabasi_albert", {"n": n, "m": m}, seed)


# ----------------------------------------------------------------- §18.1-18.3 realistic data


def configuration_model(degrees: Sequence[int], *, seed: int | None = None) -> nx.Graph:
    """A graph forced to have the given degree sequence, as a simple graph (§18.1, Molloy-Reed).

    "You first establish a degree sequence and then you force each node to pick a value from the
    sequence as its degree": each node gets as many stubs as its degree, and pairs of stubs are
    drawn at random and joined.

    **What happens to self-loops and parallel edges**, which is the part of §18.1 a caller has to
    know. The chapter's algorithm rejects a draw that would make either: "if the identifiers are
    different and the two nodes are not already connected to each other, you connect them. The
    two conditions are necessary to avoid the creation of self loops and parallel edges". It then
    admits the cost -- "you might end up with a few unassigned edges, but usually these are an
    insignificant number which will not affect the degree distribution too much". ``networkx``
    makes the multigraph and this collapses it: parallel edges become one edge and self-loops are
    dropped. So **the degree sequence is respected only approximately**, always downward, and how
    far depends on how heavy the sequence's tail is. The number of edges lost is recorded on the
    graph as ``dropped_edges`` and the realised sequence is on the nodes; check them before
    quoting this graph as having a given degree distribution.

    That approximation is also why this is not the null model for an observed network. §18.1 is
    explicit that for "asymptotic mathematics" Molloy-Reed is good enough, but "if you want to
    compare a random graph to data, the differences are crucial", and the edge-swap null of §19.1
    is the one to use -- :func:`graphrag.sna.null.configuration`, which stays inside the family
    of simple graphs and holds every degree exactly.

    The sequence has to sum to an even number to be graphic; ``networkx`` raises
    ``NetworkXError`` if it does not.
    """
    multigraph = nx.configuration_model(list(degrees), seed=seed)
    graph = nx.Graph()
    graph.add_nodes_from(multigraph.nodes)
    graph.add_edges_from((u, v) for u, v in multigraph.edges() if u != v)
    dropped = int(multigraph.number_of_edges() - graph.number_of_edges())
    _stamp(graph, "configuration_model", {"nodes": len(degrees), "sum_degrees": sum(degrees)}, seed)
    graph.graph["dropped_edges"] = dropped
    return graph


def planted_partition(
    communities: int, size: int, p_in: float, p_out: float, *, seed: int | None = None
) -> nx.Graph:
    """A stochastic block model with equal blocks: the communities are known before anything runs.

    §18.2's "easiest way to create communities in your graph model": nodes have "two different
    probabilities to connect to each other. One probability, say pin, determines the probability
    of a node u to connect to another node inside the same community. Another probability, pout,
    determines the likelihood of connecting to a node from a different community." With
    ``p_out < p_in`` "the resulting adjacency matrix will be block diagonal"; with
    ``p_in == p_out`` "the SBM is fully equivalent to a Gn,p model", which is the degenerate case
    worth remembering when a planted partition stops being recoverable.

    Every node carries its planted block as ``community`` (an ``int``), so a discovered partition
    can be scored against it with :func:`graphrag.sna.stats.compare_partitions`.

    **What the book defines and what this adds.** The chapter also describes the GN benchmark as
    a fixed parameterisation of this idea -- four caves of 32 nodes, expected degree 16, and a
    mixing parameter mu for the share of a node's edges that leave its cave. That is this
    function at ``communities=4, size=32, p_in=16(1-mu)/31, p_out=16·mu/96``; it is not given a
    name of its own here because it is one call, and because §18.2's own verdict is that the GN
    benchmark "isn't particularly realistic" beside :func:`lfr_benchmark`. Real community sizes
    "distribute broadly, just like the degree", and equal blocks do not.
    """
    graph: nx.Graph = nx.planted_partition_graph(communities, size, p_in, p_out, seed=seed)
    for node in graph.nodes:
        graph.nodes[node]["community"] = int(node) // size
    return _stamp(
        graph,
        "planted_partition",
        {"communities": communities, "size": size, "p_in": p_in, "p_out": p_out},
        seed,
    )


def lfr_benchmark(
    n: int,
    *,
    mu: float,
    degree_exponent: float = 3.0,
    community_exponent: float = 1.5,
    average_degree: float | None = None,
    min_degree: int | None = None,
    max_degree: int | None = None,
    min_community: int | None = None,
    max_community: int | None = None,
    seed: int | None = None,
) -> nx.Graph:
    """The LFR benchmark: power-law degrees, power-law community sizes, tunable mixing (§18.2).

    The parameters are the chapter's list, under names that say what they are (the book's greek
    letters in brackets): ``degree_exponent`` is alpha, "the exponent of the power law degree
    distribution of the graph"; ``community_exponent`` is beta, "the exponent of the power law
    size distribution of the communities"; ``mu`` is the mixing parameter, "regulating the
    fraction of edges going outside their planted communities"; and ``average_degree`` /
    ``min_degree`` / ``max_degree`` / ``min_community`` / ``max_community`` are k̄, kmin, kmax,
    smin and smax. ``networkx`` wants exactly one of ``average_degree`` and ``min_degree``.

    mu is what makes a benchmark hard: "if mu = 0 all edges run between nodes part of the same
    community, i.e. each community becomes a distinct connected component... if mu = 1 then nodes
    in the same community do not connect at all. Usually, you want to set mu to a reasonably low
    non-zero value", and the chapter's own figures show mu = 0.2 already producing communities
    that are hard to tell apart.

    Two constraints from step 4 of the algorithm, both of which turn into exceptions rather than
    into a quietly wrong graph: ``min_degree < min_community`` and ``max_degree < max_community``,
    "otherwise the nodes with minimum and maximum degree will never belong to any community". And
    even inside them, "sometimes the combination of parameters will require the generation of an
    impossible graph": ``networkx`` gives up with ``ExceededMaxIterations``, which is allowed to
    propagate here because the honest answer to an impossible request is that it was impossible.
    A different seed with the same parameters often succeeds where one failed.

    Why it beats the alternatives: "since we're plugging in a power law degree distribution and
    communities, it is obvious that LFR benchmarks will reproduce these characteristics of real
    world networks well -- although now you're actually forced to have a power degree
    distribution, which in some cases you might not want. They also respect clustering and small
    diameters, making them the most realistic model we have."

    **What the book defines and what this adds.** Step 5 is "a modified configuration model", so
    it inherits §18.1's self-loops; they are dropped here, as in :func:`configuration_model`, and
    counted in ``dropped_edges``. ``networkx`` records each node's planted community as the set
    of its members; that set is replaced here by an ``int`` index under the same ``community``
    key, so a planted partition reads the same way whichever generator made it.
    """
    built = nx.LFR_benchmark_graph(
        n,
        degree_exponent,
        community_exponent,
        mu,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
    )
    graph = nx.Graph()
    graph.add_nodes_from(built.nodes)
    graph.add_edges_from((u, v) for u, v in built.edges() if u != v)
    dropped = int(built.number_of_edges() - graph.number_of_edges())
    planted = sorted(
        {frozenset(built.nodes[node]["community"]) for node in built}, key=lambda c: sorted(c)
    )
    for index, block in enumerate(planted):
        for node in block:
            graph.nodes[node]["community"] = index
    _stamp(
        graph,
        "lfr_benchmark",
        {
            "n": n,
            "mu": mu,
            "degree_exponent": degree_exponent,
            "community_exponent": community_exponent,
            "average_degree": average_degree,
            "min_degree": min_degree,
            "min_community": min_community,
        },
        seed,
    )
    graph.graph["dropped_edges"] = dropped
    graph.graph["communities"] = len(planted)
    return graph


def random_geometric(
    n: int, radius: float, *, dimensions: int = 2, seed: int | None = None
) -> nx.Graph:
    """n points dropped uniformly in a space, joined when they are within ``radius`` (§18.3).

    "You first decide the dimensionality of your space... Then you generate |V| points in this
    space, by extracting them uniformly at random. Finally, you connect two points if they are at
    r distance -- or less -- from each other." Self-loops are ignored, as the chapter says.

    What it is for is the property none of the other models has: "many networks live on a
    physical space, and this physical space constraints the edge generating process. If two nodes
    are too far way from each other, they cannot connect." The shape that follows is high
    clustering -- two nodes near a third are near each other -- with *long* paths, because there
    are no shortcuts: distance in the graph tracks distance in the space. That makes it the
    counter-example to hold beside Watts-Strogatz, where the same clustering comes with short
    paths.

    Each node keeps its coordinates under ``pos``, which is what makes this graph drawable at its
    real positions. The distance is Euclidean here; the chapter notes "you're not forced to use
    the Euclidean", and ``networkx`` takes a ``p`` for other Minkowski norms, which this wrapper
    does not expose because nothing in this package asks for one yet.
    """
    graph: nx.Graph = nx.random_geometric_graph(n, radius, dim=dimensions, seed=seed)
    return _stamp(
        graph, "random_geometric", {"n": n, "radius": radius, "dimensions": dimensions}, seed
    )


# ----------------------------------------------------------------------------- ch. 16 closed forms


def poisson_degrees(mean_degree: float, max_k: int) -> list[float]:
    """The Poisson degree distribution at k̄: ``P(k) = e^-k̄ k̄^k / k!`` for k = 0..``max_k`` (§16.2).

    The chapter's own substitution: a G(n,p) degree distribution "is a binomial", but "many
    papers studying the degree distributions of random graphs use a Poisson distribution instead.
    They are practically identical... We use a Poisson because the parameters regulating it make
    it easier to calculate the things that interest us -- they all depend on a single parameter:
    k̄, the average degree."

    Multiply each entry by n to read it as "how many nodes of this network a random graph would
    put at that degree". Returns an empty list for a negative ``max_k``; at ``mean_degree`` 0 it
    is 1.0 at k = 0 and 0.0 above, which is the correct answer for a network with no edges.
    """
    if max_k < 0:
        return []
    ks = np.arange(max_k + 1)
    return [float(v) for v in scipy_stats.poisson.pmf(ks, max(mean_degree, 0.0))]


@dataclass(frozen=True)
class ExpectedProperties:
    """What chapter 16 says a random graph with this n and this m would look like.

    Every field is a closed form evaluated at the observed counts, never a sample: there is no
    n and no standard deviation here, which is exactly why :class:`AgainstRandom` also carries a
    degree-preserving null sample. A closed form says what the average random graph looks like; a
    sample says how much random graphs vary.
    """

    nodes: int
    edges: int
    p: float
    """The connection probability, ``2m/(n(n-1))`` -- which §16.1 notes is also the density: "one
    divided by the other gives you p"."""
    mean_degree: float
    """``p(n-1)`` = ``2m/n``: "p times the number of nodes minus one -- because we avoid creating
    self loops" (§16.2)."""
    clustering: float
    """Equal to ``p``, for every node and hence on average (§16.5): "the clustering coefficient
    doesn't depend on any node characteristic, and it's expected to be the same -- equal to p --
    for all nodes"."""
    degree_variance: float
    """Equal to the mean, because a Poisson's variance is its rate (§16.2). This is the number
    that makes the variance-to-mean ratio a test: 1 for a random graph, above 1 for a broad
    distribution."""
    max_degree: float
    """The degree that fewer than one node in n is expected to exceed. See
    :func:`expected_properties` for why this is a convention and not the book's formula."""
    path_length: float | None
    """``ln n / ln k̄`` (§16.4), or ``None`` when k̄ <= 1, where the derivation divides by a
    logarithm that is zero or negative: a graph that thin has no expected path length because it
    has no connected bulk to take paths across."""
    giant_component_share: float
    """The share of nodes a random graph with this mean degree would put in its largest
    component. 0 below the k̄ = 1 threshold. See :func:`expected_properties`."""
    giant_threshold_p: float
    """``1/n``: "the magical value of p" above which the giant component appears, because that is
    where the mean degree passes 1 (§16.3)."""
    connected_threshold_p: float
    """``ln n / n``: the p above which fewer than one node is left outside the giant component,
    i.e. the graph is expected to be connected (§16.3, n. 11)."""
    degree_pmf: tuple[float, ...]
    """The Poisson probability of each degree from 0 to the observed maximum (§16.2). Multiply by
    ``nodes`` for expected node counts."""

    @property
    def above_giant_threshold(self) -> bool:
        """Whether k̄ > 1, the phase transition of §16.3."""
        return self.mean_degree > 1.0

    @property
    def expected_connected(self) -> bool:
        """Whether p is past ``ln n / n``, where a random graph has no node outside the giant."""
        return self.nodes > 1 and self.p >= self.connected_threshold_p


def expected_properties(graph: nx.Graph) -> ExpectedProperties:
    """The chapter 16 closed forms, evaluated at this network's own n and m.

    This is the "observed versus random" baseline: not a random graph, but what the *average*
    random graph with these counts would have. §16.1's conversion ``p·n(n-1)/2 = m`` fixes p from
    the observed counts, and every other field follows from it -- clustering = p (§16.5), mean
    degree = p(n-1) (§16.2), path length = ln n / ln k̄ (§16.4), the two thresholds of §16.3.

    A directed network is flattened first (§6.2): p is defined over *unordered* pairs, and every
    formula here counts a mutual pair once. The flattening is stated by whatever prints the
    result -- :func:`render_against_random` does.

    **What the book defines and what this adds.** Two fields are conventions, marked as such
    because a reader checking them against the chapter will not find them:

    ``giant_component_share`` -- §16.3 gives the *threshold* ("when the average degree is higher
    than one, the giant component appears") and the point where nothing is left outside it, but
    not the share in between. The standard fixed point ``S = 1 - e^(-k̄S)``, iterated here, fills
    that gap; it is the branching-process answer the threshold is derived from, and it agrees
    with the chapter at both ends.

    ``max_degree`` -- §16.2 says only that real hubs "have a much higher degree than the highest
    degrees you'll find in a random network". To make that comparable, this is the degree that a
    Poisson puts fewer than one node of n above, which is §16.3's own style of argument ("this is
    equivalent to say that we want to have fewer than one node outside GCC") applied to the top
    of the distribution rather than to the bottom.

    Undefined for a graph with fewer than two nodes: p, the mean degree and everything derived
    from them are 0.0 there, and the path length is ``None``, rather than raising -- an empty
    network is a thing a report has to be able to print.
    """
    flat, _ = undirected_view(graph)
    nodes = flat.number_of_nodes()
    edges = flat.number_of_edges()
    pairs = nodes * (nodes - 1) / 2
    p = edges / pairs if pairs > 0 else 0.0
    mean_degree = 2 * edges / nodes if nodes else 0.0
    observed_max = max((d for _, d in flat.degree()), default=0)
    path_length = (
        math.log(nodes) / math.log(mean_degree) if nodes > 1 and mean_degree > 1.0 else None
    )
    return ExpectedProperties(
        nodes=nodes,
        edges=edges,
        p=p,
        mean_degree=mean_degree,
        clustering=p,
        degree_variance=mean_degree,
        max_degree=_poisson_max_degree(mean_degree, nodes),
        path_length=path_length,
        giant_component_share=_giant_component_share(mean_degree),
        giant_threshold_p=1 / nodes if nodes else 0.0,
        connected_threshold_p=math.log(nodes) / nodes if nodes > 1 else 0.0,
        degree_pmf=tuple(poisson_degrees(mean_degree, int(observed_max))),
    )


def _giant_component_share(mean_degree: float) -> float:
    """The share of nodes in the giant component of a random graph with this mean degree.

    The fixed point ``S = 1 - e^(-k̄S)``, iterated to convergence. Below k̄ = 1 the only solution
    is S = 0, which is §16.3's phase transition: "if the average degree is less than one, many
    nodes won't have edges. Thus they cannot be part of the largest connected component."
    """
    if mean_degree <= 1.0:
        return 0.0
    share = 0.5
    for _ in range(200):
        nxt = 1.0 - math.exp(-mean_degree * share)
        if abs(nxt - share) < 1e-12:
            return nxt
        share = nxt
    return share


def _poisson_max_degree(mean_degree: float, nodes: int) -> float:
    """The degree fewer than one node of ``nodes`` is expected to exceed, under Poisson(k̄)."""
    if nodes < 1 or mean_degree <= 0:
        return 0.0
    return float(scipy_stats.poisson.isf(1 / nodes, mean_degree))


# ----------------------------------------------------------------------------- ch. 17 comparison


#: Above how many nodes the average path length is estimated from a sample of sources rather than
#: computed from every one. §16.4 defines the average over all pairs; computing it is O(n·m) per
#: graph and this report computes it on the observed network *and* on every null sample, so above
#: this size it is estimated from :data:`PATH_SOURCES` randomly chosen sources and the report says
#: that it was. At or below it, every node is a source and the number is exact.
PATH_SOURCES = 64

#: How many times the G(n,p) expectation the observed clustering has to be before the report says
#: the word "clustered". The book gives no threshold -- §17.1 only argues that real networks are
#: clustered *compared with p* -- so this is a convention: twice the expectation, together with a
#: z of :data:`SIGNIFICANT_Z` against the degree-preserving null, which is the stricter of the two
#: on any network of interesting size.
CLUSTERED_RATIO = 2.0

#: How far above ``ln n / ln k̄`` the observed path length may sit and still be called short
#: (§16.4, §17.2). Also a convention: the chapter's claim is that real networks are short *like*
#: random graphs, so the test is "the same order as the random expectation", not "shorter".
#:
#: Twice, rather than one and a half times, because §17.2 exists to separate two graphs and the
#: tighter threshold gave them the same word. The chapter's own exercise (§17.5, ex. 1) contrasts
#: a small world of 100 nodes at k = 8 and rewiring probability 0.05 with a caveman graph of ten
#: ten-node caves: the small world comes out at 1.71x this expectation and the caveman graph at
#: 2.84x, which is the difference the model was built to demonstrate -- "differently from
#: cavemen, this time we have short paths". Rewiring moves it exactly as §17.2 says it does
#: (3.57x at p = 0.01, 1.96x at 0.05, 1.54x at 0.1 on 500 nodes), so the threshold has to sit
#: above a lightly rewired small world and below a lattice. At 2.0 the random geometric graph
#: (2.59x) and the caveman graph stay long, and G(n,m), preferential attachment and the karate
#: club stay short at roughly 1.0x.
SHORT_RATIO = 2.0

#: The variance-to-mean ratio above which a degree distribution is called broad. A Poisson has
#: exactly 1 (§16.2), so this asks for twice the spread a random graph would produce. It is not
#: a power-law test and this package never calls a distribution scale-free on the strength of it.
BROAD_DISPERSION = 2.0

#: The z at which a null-model comparison is worth quoting, matching the threshold the rest of
#: this package prints its verdicts at.
SIGNIFICANT_Z = 2.0


@dataclass(frozen=True)
class AgainstRandom:
    """Chapter 17's three comparisons for one network: clustered, short and broad, or not.

    Everything here is computed on the **unweighted, undirected** graph, because the closed forms
    of chapter 16 are about a simple graph: clustering counts triangles, degree counts
    neighbours, and a path length counts hops. The report's summary table prints the *weighted*
    clustering, so the two numbers differ by construction and the section says so.
    """

    nodes: int
    edges: int
    expected: ExpectedProperties
    component_expected: ExpectedProperties | None
    """The same closed forms recomputed on the largest component, when the network is in more
    than one piece and the observed path length therefore describes only that piece. ``None``
    when the network is connected, where it would be the same object."""
    clustering: float
    clustering_null: Significance
    path_length: float | None
    path_nodes: int
    """How many nodes the observed path length was averaged over: the largest component's."""
    path_sources: int
    """How many nodes it was measured from. Equal to ``path_nodes`` when exact."""
    path_estimated: bool
    path_null: Significance
    degrees: Summary
    samples: int
    """How many degree-preserving rewirings the null columns are averaged over. 0 means the
    network was too small to rewire and there is no null column at all."""
    flattened: str
    """The sentence saying this ran on a flattened view, or "" on an undirected network."""
    filters: str
    """The filters that built this network, for the section's own frame line."""
    frame: str
    """What the network is a network of, from the graph itself."""
    verdict: str
    notes: tuple[str, ...]

    @property
    def dispersion(self) -> float:
        """The degree distribution's variance-to-mean ratio: 1 for a Poisson, above 1 for broad.

        ``inf`` when every node has degree 0 and something still varies, 0.0 for an empty
        network. The one number that separates "broad" from "random" without fitting anything.
        """
        if self.degrees.mean <= 0:
            return 0.0
        return self.degrees.variance / self.degrees.mean

    @property
    def clustering_ratio(self) -> float:
        """Observed clustering over the G(n,p) expectation (= p). 0.0 when p is 0."""
        return self.clustering / self.expected.clustering if self.expected.clustering > 0 else 0.0

    @property
    def path_ratio(self) -> float:
        """Observed path length over ``ln n / ln k̄``, on the frame the observation was taken on."""
        expected = (self.component_expected or self.expected).path_length
        if self.path_length is None or not expected:
            return 0.0
        return self.path_length / expected

    @property
    def clustered(self) -> bool:
        """Clustered in ch. 17's sense: far above p, and not explained by the degrees alone."""
        if self.clustering_ratio < CLUSTERED_RATIO:
            return False
        return not self.clustering_null.testable or self.clustering_null.z >= SIGNIFICANT_Z

    @property
    def short(self) -> bool:
        """Short in §17.2's sense: at most :data:`SHORT_RATIO` times ``ln n / ln k̄`` (2.0x).

        False when there is no path length at all, and when the mean degree is too low for the
        expectation to exist -- ``path_ratio`` is 0.0 in both cases and neither is "short".
        """
        ratio = self.path_ratio
        return 0.0 < ratio <= SHORT_RATIO

    @property
    def broad(self) -> bool:
        """Broader than the Poisson with the same mean -- *not* a claim that it is a power law.

        False on a network with no edges, where every degree is 0 and there is no distribution
        to be broad or narrow, as well as on one whose degrees sit at the Poisson's own spread.
        """
        if self.edges == 0 or self.degrees.count < MIN_TAIL_SAMPLE:
            return False
        return self.dispersion >= BROAD_DISPERSION


def against_random(
    graph: nx.Graph,
    *,
    samples: int = 50,
    seed: int | None = None,
    filters: str = "",
    rewirings: Iterable[nx.Graph] | None = None,
) -> AgainstRandom:
    """Measure the three properties chapter 17 compares, against two kinds of random (ch. 16-19).

    The chapter's argument in one function. A real network is interesting because it is
    *clustered* (§17.1: clustering far above the p a random graph would have), *short* (§17.2:
    path lengths near ``ln n / ln k̄``, which random graphs already achieve) and *broad* (§17.3:
    a degree distribution wider than the Poisson, with hubs "much higher than the highest degrees
    you'll find in a random network"). None of the three is a property of a number; each is a
    comparison, and this returns both sides of all three.

    **Two nulls, deliberately.** The G(n,p) closed forms of :func:`expected_properties` hold only
    n and m fixed, which is why §16.5 calls them the weakest baseline there is: they flatter
    almost any observation. The degree-preserving sample from
    :func:`graphrag.sna.null.configuration` holds every node's degree fixed as well, so a
    clustering that beats *it* is not explained by "some nodes are simply busier". Where the two
    disagree, the degrees are the explanation.

    **The degree distribution has no configuration column, and cannot have one.** That null is
    built by rewiring the observed edges, so every sample has the observed degree sequence
    exactly -- mean, variance, maximum and all (§18.1: the configuration model reproduces the
    degree distribution because the distribution is its input). Testing broadness against it
    would be testing a number against itself, so the degree rows are compared with the Poisson
    only, and the section says why.

    ``samples`` is the number of rewirings, ``seed`` fixes them and the path-length source sample,
    and ``filters`` is the filter line the report prints in this section's frame. A directed
    network is flattened first and the flattening is stated, exactly as the ``--by`` section does
    (§6.2). Edge weights are ignored throughout: chapter 16's formulas count triangles, neighbours
    and hops.

    ``rewirings`` is the null family to measure, for a caller that already has one: the full
    report's modularity null (:func:`graphrag.sna.cluster.null_model_modularity`) draws the same
    family from the same seed, so ``run_analysis`` draws it once and passes it here rather than
    paying §19.1's edge swaps twice. It is consumed lazily and one sample is alive at a time,
    because a list of fifty rewirings of a 74,000-edge network is a gigabyte. When it is given,
    ``samples`` is ignored and the count reported is however many samples arrived; when it is
    ``None`` this draws its own, which is what a standalone ``against_random(graph)`` does.
    Whatever arrives is measured unweighted, so a family drawn with weights costs nothing here.

    **What this costs.** Per sample: one average clustering coefficient and one breadth-first
    search from :data:`PATH_SOURCES` sources over the sparse adjacency, which is a few hundred
    milliseconds on a network of a few thousand nodes and tens of thousands of edges, times
    ``samples``. Drawing the family costs about as much again, which is why it is shared.
    """
    flat, flattened = undirected_view(graph)
    rng = random.Random(seed)  # noqa: S311 -- shuffling a null, not a secret
    expected = expected_properties(flat)
    clustering = float(nx.average_clustering(flat)) if flat.number_of_nodes() > 2 else 0.0
    path_length, path_nodes, path_sources = _path_length(flat, rng=rng)
    degrees = [float(d) for _, d in flat.degree()]
    summary = describe(degrees) if degrees else describe([0.0])
    component_expected = _component_expected(flat, path_nodes)

    clustering_samples: list[float] = []
    path_samples: list[float] = []
    family = (
        configuration(flat, max(samples, 0), seed=rng, weights=False)
        if rewirings is None
        else rewirings
    )
    for sample in family:
        if sample.number_of_nodes() > 2:
            clustering_samples.append(float(nx.average_clustering(sample)))
        sampled_path, _, _ = _path_length(sample, rng=rng)
        if sampled_path is not None:
            path_samples.append(sampled_path)
    clustering_result = significance(
        clustering, clustering_samples, null="configuration", tail="right"
    )
    path_result = significance(
        path_length if path_length is not None else 0.0,
        path_samples if path_length is not None else [],
        null="configuration",
        tail="left",
    )
    report = AgainstRandom(
        nodes=flat.number_of_nodes(),
        edges=flat.number_of_edges(),
        expected=expected,
        component_expected=component_expected,
        clustering=clustering,
        clustering_null=clustering_result,
        path_length=path_length,
        path_nodes=path_nodes,
        path_sources=path_sources,
        path_estimated=path_sources < path_nodes,
        path_null=path_result,
        degrees=summary,
        samples=clustering_result.samples,
        flattened=flattened,
        filters=filters,
        frame=str(graph.graph.get("frame", "")).strip(),
        verdict="",
        notes=(),
    )
    return replace(report, verdict=_verdict(report), notes=_notes(report))


def _path_length(graph: nx.Graph, *, rng: random.Random) -> tuple[float | None, int, int]:
    """§16.4's average path length in hops, over the largest component, and what it covers.

    Returns the average, the number of nodes it was averaged over, and the number of sources it
    was measured from. Above :data:`PATH_SOURCES` nodes the sources are a random sample, which
    makes this an estimate with a sampling error of its own; below it, every node is a source and
    the result is what ``networkx.average_shortest_path_length`` would return for that component.

    Hops, not weights: §16.4 counts "the number of nodes at l hops away", and the weights in this
    package's networks are affinities rather than distances anyway (see
    :mod:`graphrag.sna.measures`). The breadth-first search runs over the sparse adjacency matrix
    in ``scipy`` rather than in ``networkx``, because this is computed once for the observation
    and again for every null sample.

    ``None`` when the largest component has one node or none: a path length needs two nodes to
    run between, and an isolated node has no distance to anything. A disconnected network gets
    its largest component and nothing else, because the distance between two components is
    infinite and averaging it is not defined -- the caller has to say which component the number
    describes.
    """
    if graph.number_of_nodes() < 2:
        return None, graph.number_of_nodes(), 0
    largest = max(nx.connected_components(graph), key=len)
    if len(largest) < 2:
        return None, len(largest), 0
    matrix, order = adjacency(graph.subgraph(largest), weight=None, sparse=True)
    rows = list(range(len(order)))
    sources = rows if len(rows) <= PATH_SOURCES else sorted(rng.sample(rows, PATH_SOURCES))
    distances = csgraph.shortest_path(matrix, method="D", unweighted=True, indices=sources)
    reachable = np.isfinite(distances) & (distances > 0)
    pairs = int(reachable.sum())
    total = float(distances[reachable].sum())
    return (total / pairs if pairs else None), len(largest), len(sources)


def _component_expected(graph: nx.Graph, path_nodes: int) -> ExpectedProperties | None:
    """The closed forms recomputed on the largest component, when that is what was measured.

    The observed path length is an average over one component, so comparing it with ``ln n / ln
    k̄`` for the *whole* network compares two different frames -- and on a fragmented network the
    whole-network k̄ is the smaller of the two, which biases the comparison towards "short".
    ``None`` when the network is connected and the two are the same thing.
    """
    if path_nodes >= graph.number_of_nodes() or path_nodes < 2:
        return None
    largest = max(nx.connected_components(graph), key=len)
    return expected_properties(graph.subgraph(largest))


def _verdict(report: AgainstRandom) -> str:
    """The one sentence chapter 17 is an argument for, with the claims it cannot make named.

    "Clustered, short and broad" is three comparisons, not one verdict, and a network can be any
    combination of them -- a Watts-Strogatz graph is clustered and short and not broad, a
    Barabasi-Albert graph is broad and short and not clustered, and a random geometric graph is
    clustered and neither of the other two.
    """
    can: list[str] = []
    cannot: list[str] = []
    if report.expected.clustering > 0 and report.nodes > 2:
        can.append(
            f"{'clustered' if report.clustered else 'not clustered'} "
            f"({report.clustering:.4f} against a G(n,p) expectation of "
            f"{report.expected.clustering:.4f}, {report.clustering_ratio:.1f}x"
            + (
                f", z = {report.clustering_null.z:.2f} against {report.samples} rewirings)"
                if report.clustering_null.testable
                else ", with no null sample)"
            )
        )
    else:
        cannot.append(
            "clustered: the network has no edges to close triangles with, so the G(n,p) "
            "expectation is 0 and the ratio is undefined (§16.5)"
        )
    expected_path = (report.component_expected or report.expected).path_length
    if report.path_length is not None and expected_path:
        frame = (
            f"the largest component, {report.path_nodes:,} of {report.nodes:,} nodes"
            if report.component_expected is not None
            else "the whole network"
        )
        can.append(
            f"{'short' if report.short else 'not short'} ({report.path_length:.3f} hops on "
            f"{frame}, against ln n / ln k̄ = {expected_path:.3f}, {report.path_ratio:.1f}x)"
        )
    elif report.path_length is None:
        cannot.append(
            "short: no two nodes of this network are connected, so there is no path length to "
            "average (§16.4)"
        )
    else:
        cannot.append(
            "short: the mean degree is at or below 1, where ln n / ln k̄ divides by a "
            "non-positive logarithm and §16.4's derivation gives no expectation to compare with"
        )
    if report.edges == 0:
        cannot.append(
            "broad: the network has no edges, so every degree is 0 and there is no distribution "
            "to hold beside a Poisson (§16.2)"
        )
    elif report.degrees.count >= MIN_TAIL_SAMPLE:
        can.append(
            f"{'broad' if report.broad else 'not broad'} (degree variance-to-mean ratio "
            f"{report.dispersion:.2f} against the Poisson's 1.00, maximum degree "
            f"{report.degrees.maximum:.0f} against {report.expected.max_degree:.0f})"
        )
    else:
        cannot.append(
            f"broad: {report.degrees.count} node(s) is too few for the shape of a degree "
            f"distribution to be a measurement (§3.1 wants at least {MIN_TAIL_SAMPLE})"
        )
    made = "Relative to a random graph with the same n and m, this network is " + _join(can) + "."
    if not can:
        made = "None of chapter 17's three comparisons can be made on this network."
    missing = (
        " It cannot say whether the network is " + _join(cannot) + "." if cannot and can else ""
    )
    if cannot and not can:
        sentence = _join(cannot)
        missing = " " + sentence[0].upper() + sentence[1:] + "."
    return made + missing


def _join(parts: Sequence[str]) -> str:
    """The parts as an English list, because a report sentence is read, not parsed."""
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _notes(report: AgainstRandom) -> tuple[str, ...]:
    """The conditions this section's numbers hold under, printed under them rather than implied."""
    notes: list[str] = [
        "Clustering, degree and path length here are unweighted and undirected, because chapter "
        "16's formulas count triangles, neighbours and hops. The summary table above prints the "
        "*weighted* average clustering, so the two numbers are different quantities and are "
        "expected to differ.",
        "Broad is not scale-free. §17.3's exponent belongs to a model that produces one; "
        "whether *this* distribution is a power law is a separate claim needing a fit and a "
        "comparison with the alternatives (ch. 9), and nothing here fits anything.",
    ]
    if report.flattened:
        notes.append(f"This section ran {report.flattened}")
    if report.path_estimated:
        notes.append(
            f"The average path length is an estimate: it was measured from {report.path_sources} "
            f"randomly chosen sources of the {report.path_nodes:,} in the largest component, "
            "rather than from all of them, and the same estimator was used on every null sample "
            "so the comparison is like for like. Re-run with a different seed to see how much it "
            "moves."
        )
    if report.component_expected is not None:
        notes.append(
            f"The network is in more than one piece, so the path length describes its largest "
            f"component ({report.path_nodes:,} of {report.nodes:,} nodes) and is compared with "
            "the closed form recomputed on that component. Nothing here is a statement about the "
            "distance between components, which is infinite."
        )
    if not report.expected.above_giant_threshold:
        notes.append(
            f"The mean degree is {report.expected.mean_degree:.2f}, at or below the k̄ = 1 phase "
            "transition of §16.3, so a random graph of this size would have no giant component "
            "at all. Read every number above as describing a scattering of small pieces."
        )
    for result in (report.clustering_null, report.path_null):
        if result.caveat and result.caveat not in notes:
            notes.append(result.caveat)
    return tuple(notes)


# ----------------------------------------------------------------------------- rendering


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _null_cell(result: Significance) -> str:
    """One null column: the rewirings' mean and spread and the z, or why there is none."""
    if not result.testable:
        return "no sample"
    return (
        f"mean {_num(result.null_mean)}, sd {_num(result.null_std)}, "
        f"z {result.z:.2f}, p {_num(result.p_value)}"
    )


def render_against_random(report: AgainstRandom) -> list[str]:
    """The "Against random" section: chapter 17's comparison, with both nulls named beside it."""
    expected = report.expected
    component = report.component_expected
    path_expected = (component or expected).path_length
    variance_ratio = (
        report.degrees.variance / expected.degree_variance if expected.degree_variance else 0.0
    )
    max_ratio = report.degrees.maximum / expected.max_degree if expected.max_degree else 0.0
    lines = [
        "## Against random",
        "",
        f"**Sampling frame.** The {report.nodes:,} nodes and {report.edges:,} edges of this "
        "network, unweighted and undirected"
        + (f", {report.flattened}" if report.flattened else "")
        + (f". Filters: {report.filters}." if report.filters else ".")
        + (f" {report.frame}" if report.frame else ""),
        "",
        "**n.** "
        + (
            f"{report.samples} degree-preserving rewirings for the null columns"
            if report.samples
            else "no null sample: this network is too small to rewire"
        )
        + f"; {report.degrees.count:,} degrees for the distribution rows"
        + (
            f"; the path length averaged over {report.path_nodes:,} node(s) from "
            f"{report.path_sources} source(s)"
            if report.path_length is not None
            else "; no path length"
        )
        + ". The G(n,p) column is a closed form evaluated at this n and m, not a sample, so it "
        "has no n and no spread of its own.",
        "",
        "**Null models.** Two. `erdos_renyi` holds fixed "
        + HOLDS_FIXED["erdos_renyi"]
        + " -- the weakest null there is, and the one chapter 17 argues against. `configuration` "
        "holds fixed "
        + HOLDS_FIXED["configuration"]
        + ", so a property that beats it is not explained by the degrees.",
        "",
        "**Implements.** §16.2-16.5 (the closed forms of a random graph) and §17.1-17.3 (whether "
        "this network is clustered, short and broad by comparison with them), with the "
        "degree-preserving sample of §18.1 and §19.1.",
        "",
        "| property | observed | G(n,p) expectation | ratio | degree-preserving null |",
        "|---|---|---|---|---|",
        f"| average clustering (§17.1) | {_num(report.clustering)} | p = "
        f"{_num(expected.clustering)} | {report.clustering_ratio:.1f}x | "
        f"{_null_cell(report.clustering_null)} |",
        f"| average path length, hops (§17.2) | "
        f"{_num(report.path_length) if report.path_length is not None else 'none'} | "
        f"{'ln n / ln k̄ = ' + _num(path_expected) if path_expected else 'undefined at k̄ <= 1'} | "
        f"{report.path_ratio:.1f}x | {_null_cell(report.path_null)} |",
        f"| mean degree (§16.2) | {_num(report.degrees.mean)} | 2m/n = "
        f"{_num(expected.mean_degree)} | 1.0x | held fixed by construction |",
        f"| degree variance (§16.2) | {_num(report.degrees.variance)} | = k̄ = "
        f"{_num(expected.degree_variance)} | "
        f"{variance_ratio:.1f}x | held fixed by construction |",
        f"| maximum degree (§17.3) | {_num(report.degrees.maximum)} | "
        f"{_num(expected.max_degree)} | {max_ratio:.1f}x | held fixed by construction |",
        f"| variance-to-mean ratio (§17.3) | {report.dispersion:.2f} | 1.00 (Poisson) | "
        f"{report.dispersion:.1f}x | held fixed by construction |",
        "",
        report.verdict,
        "",
        _giant_sentence(report),
        "",
        "The z-scores read in the direction of the property: clustering is tested on the right "
        "tail, because clustered means *more* triangles than the null, and path length on the "
        "left, because short means *fewer* hops. A positive z on the path-length row therefore "
        "says this network's paths are longer than rewiring its own degrees would produce, and "
        "its p-value is the share of rewirings that came out at or below the observation.",
        "",
        "The degree rows have no null column because they cannot have one: the degree-preserving "
        "null is built by rewiring the observed edges, so every sample carries the observed "
        "degree sequence exactly (§18.1). Against that null the degree distribution is not a "
        "finding, it is the input.",
        "",
    ]
    lines += [f"- {note}" for note in report.notes if note]
    lines += [""]
    return lines


def _giant_sentence(report: AgainstRandom) -> str:
    """§16.3's phase transition, read against this network's own largest component."""
    expected = report.expected
    observed_share = report.path_nodes / report.nodes if report.nodes else 0.0
    if not expected.above_giant_threshold:
        return (
            f"**Giant component (§16.3).** Mean degree {expected.mean_degree:.2f} is at or below "
            "the k̄ = 1 threshold, so a random graph with these counts would have no giant "
            f"component; this network holds {observed_share:.0%} of its nodes in its largest one."
        )
    connected = (
        "past ln n / n, so a random graph this dense would have every node in it"
        if expected.expected_connected
        else f"below ln n / n = {expected.connected_threshold_p:.4f}, so a random graph this "
        "dense would still leave nodes outside it"
    )
    return (
        f"**Giant component (§16.3).** Mean degree {expected.mean_degree:.2f} is above the k̄ = 1 "
        f"threshold, where the giant component appears; p = {expected.p:.4f} is {connected}. A "
        f"random graph would put about {expected.giant_component_share:.0%} of nodes in its "
        f"largest component and this network puts {observed_share:.0%} there."
    )


def against_random_payload(report: AgainstRandom) -> dict[str, Any]:
    """The same section as plain JSON-able data, under a stable set of keys."""
    expected = report.expected
    component = report.component_expected
    return {
        "nodes": report.nodes,
        "edges": report.edges,
        "samples": report.samples,
        "frame": report.frame,
        "filters": report.filters,
        "flattened": report.flattened,
        "verdict": report.verdict,
        "notes": list(report.notes),
        "clustered": report.clustered,
        "short": report.short,
        "broad": report.broad,
        "observed": {
            "average_clustering": report.clustering,
            "average_path_length": report.path_length,
            "path_length_nodes": report.path_nodes,
            "path_length_sources": report.path_sources,
            "path_length_estimated": report.path_estimated,
            "degree_mean": report.degrees.mean,
            "degree_variance": report.degrees.variance,
            "degree_maximum": report.degrees.maximum,
            "degree_dispersion": report.dispersion,
            "degree_caveat": report.degrees.caveat,
        },
        "erdos_renyi": _expected_payload(expected),
        "erdos_renyi_largest_component": _expected_payload(component) if component else None,
        "ratios": {
            "clustering": report.clustering_ratio,
            "path_length": report.path_ratio,
            "dispersion": report.dispersion,
        },
        "configuration_null": {
            "average_clustering": _significance_payload(report.clustering_null),
            "average_path_length": _significance_payload(report.path_null),
        },
    }


def _expected_payload(expected: ExpectedProperties) -> dict[str, Any]:
    return {
        "nodes": expected.nodes,
        "edges": expected.edges,
        "p": expected.p,
        "mean_degree": expected.mean_degree,
        "average_clustering": expected.clustering,
        "degree_variance": expected.degree_variance,
        "maximum_degree": expected.max_degree,
        "average_path_length": expected.path_length,
        "giant_component_share": expected.giant_component_share,
        "giant_threshold_p": expected.giant_threshold_p,
        "connected_threshold_p": expected.connected_threshold_p,
        "degree_pmf": list(expected.degree_pmf),
    }


def _significance_payload(result: Significance) -> dict[str, Any]:
    return {
        "observed": result.observed,
        "null": result.null,
        "samples": result.samples,
        "null_mean": result.null_mean,
        "null_std": result.null_std,
        "z": result.z,
        "p_value": result.p_value,
        "tail": result.tail,
        "caveat": result.caveat,
    }
