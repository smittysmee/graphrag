"""Edges we are not sure about: probabilistic networks and measurement error (Atlas ch. 28).

Every other module in this package measures the network as if its edges were facts. Chapter 28
is the chapter that says they are not, and it is blunt about how the field handles that: *"[in
network analysis] the practice of ignoring measurement error is still mainstream"* (p. 395). Its
opening example is the graph everything is tested on -- *"we actually don't know whether it has
77 or 78 edges. The bewildering thing about this graph is that almost no one even mentions this
problem!"* (p. 396).

A corpus network has the same problem twice over. An edge here is not an observation of a tie;
it is a **claim with evidence behind it**: an extraction pass named an entity, a matcher found a
passage for it, a projection turned two mentions in that passage into an edge. Each of those
steps can be wrong in the ways §28.1 lists -- missing edges, spurious edges, duplicated nodes,
misestimated weights -- and the package already holds the record of how well each step went. The
mention tier is that record (:data:`graphrag.models.MentionTier`), and this module turns it into
the object §28.2 defines: a probabilistic network ``G = (V, E, Pi)``, where ``Pi`` gives each
observed edge a probability ``p`` of existing, and a measure's value is its expectation over the
possible worlds.

**What the book gives and what this adds.** §28.1's own methods for *estimating* the error need
either the network measured several times (*"it is very very unlikely you're going to measure
your network multiple times"*, p. 399) or a generative model good enough to say which edges are
spurious -- and the chapter warns that a bad model, *"say a Gn,p model"*, hands you *"a 50-50
chance of having or not having the edge, which is pretty useless"* (p. 400). Neither is
available here: one corpus, read once, with no model of how a podcast archive generates entity
co-mentions. What *is* available is the provenance of every edge, which is the case §28.2 is
written for: *"the edges in your network come already with their own probabilities, and you don't
know the model that generated them."* The mapping from tier to probability is therefore this
package's, not the book's; it is stated in :data:`TIER_PROBABILITY`, it is monotone in the
evidence, and no number computed from it is worth more than that mapping is.

**The rule, stated so it can be argued with.**

* One passage supporting one edge is worth :data:`TIER_PROBABILITY` of the weaker of the two
  mentions behind it. The weaker one, not the product of the two: the same matcher read the same
  passage for both, so their errors are not independent and the product's independence
  assumption would understate a pair that is really one measurement. Within a passage, the
  weakest link decides.
* ``k`` passages supporting one edge are combined as ``1 - prod(1 - p_i)``: the edge exists if
  any one of them is right. Across passages the independence assumption of §28.2 (*"Normally,
  the probabilities in Pi are all independent from each other"*, p. 400) is kept, because two
  passages really are two readings.
* An edge whose evidence is not mention-based is certain, ``p = 1.0``, and says so: the speaker
  network's edges come from a document's own speaker list, and the topic network's from a count
  written at ingestion. Certainty here means "this pipeline cannot put a number on it", never
  "this is true of the world" -- the sampling frame still says what the network is a record of.

**What is exact and what is sampled.** §28.3 gives closed forms for some questions and says
others have none. :func:`expected_degree` (``E[k_u] = sum_v p_uv``), :func:`expected_edges` and
:func:`expected_density` are sums of probabilities and are computed exactly, with no sampling and
no seed. :func:`degree_distribution` is exact too, and it is the form the book actually asks for:
*"that would be a bit silly: the degree is a count, it shouldn't be a continuous number... It is
much better to give each node a probability distribution for each possible degree value"*
(p. 403, Figure 28.7). Everything else -- betweenness, clustering, modularity, whether two nodes
are in the same component -- goes through :func:`realisations` and :func:`expectation`, which is
the Monte Carlo of §28.2: the ``2^m`` possible worlds *"can (and will) get out of hand pretty
quickly"* (p. 401), so a seeded sample of them stands in for all of them, and the interval is a
percentile of that sample rather than a standard-error formula.

**A stated budget for the sampling itself (ATL-F2, not the book's).** Re-running every centrality
on every sampled world is what made ``analyze --uncertain`` slow on a large network: the book's
own remedy for a Monte Carlo that has gotten out of hand is to sparsify the network first (p.
402), which changes the sampling frame; this package instead prices the *measures*. The four
shortest-path-search centralities -- betweenness, closeness, harmonic, reach
(:data:`graphrag.sna.measures.EXPENSIVE_CENTRALITIES`) -- cost one more traversal per node per
realisation, so they are not sampled by default; ``--uncertain-expensive`` opts them in. And
:func:`expectation`/:func:`node_expectation`/:func:`uncertainty_report` all take ``max_seconds``:
a wall-clock budget, shared across every measure in one ``--uncertain`` run rather than reset per
measure, that stops the realisations once it is spent and reports the interval over whatever
finished -- fewer than the requested sample count, with :class:`Expectation`'s own caveat saying
so next to the numbers, rather than a command that hangs.

**What the budget cannot do.** It stops the *next* realisation from starting; it cannot pre-empt
one already running. On a network large enough that a single call to an expensive centrality
costs more than the whole budget -- exact betweenness is O(n(m + n log n)) and has no early exit
-- the realisation in progress when the clock runs out still finishes, so the wall time actually
spent can exceed ``max_seconds`` by up to one measure's own cost. The budget's guarantee is on
the *count*: no realisation starts once the clock has run out, never on the total time.

**Named in §28.3 and deliberately not built.** The section deep-dives five classical problems.
Three are here -- node degree, connected components (:func:`reliability`) and, through
:func:`expectation`, anything else a measure can be run on. The other two, and one that is here
by a different route:

- *Ego networks (pp. 404-405, Figures 28.8-28.9).* The section's point is that there are two
  orders -- realise the possible worlds and take each one's ego network, or take a probabilistic
  ego network and realise *that* -- and that they give correlated but different answers, because
  the second can produce an ego not connected to its own neighbours. Both orders already compose
  out of what exists (:func:`graphrag.sna.export.ego` with :func:`realisations` either way
  round), so nothing is missing arithmetically. What is not built is the command that picks one
  and reports the gap, and it is not built here because the ego network itself is §30.1's object
  and ATL-30's ticket: this module would have to define the thing before it could say which
  order to extract it in.
- *Densest subgraph (p. 407).* The probabilistic form is the expected density of a subgraph
  across the possible worlds, and :func:`expected_density` is exactly that for any subgraph a
  caller hands it. What is missing is the *search* -- finding the densest one -- and that is
  missing in the deterministic case too: its home is k-core decomposition (§14.7, ATL-14).
  Building the uncertain version first would leave it with nothing to be compared against.
- *Betweenness centrality (pp. 405-406).* The section gives a closed-ish form -- weight each
  path by the product of its edge probabilities, and truncate at some length because the
  contributions keep halving -- and this module samples instead. Two reasons. The truncation
  length is the parameter the book itself declines to fix (*"Finding the right balance is
  something that probably depends on the probability distributions on your edges and there is no
  silver bullet"*), so a report would be quoting a threshold nobody could defend, which is the
  objection chapter 27 makes to naive backboning. And our betweenness is weighted -- it runs over
  ``distance = 1/weight`` (:func:`graphrag.sna.measures.centrality`) -- while the path-product
  form is written for the unweighted shortest-path count, so grafting one onto the other would
  produce a number that is neither. Sampling costs more and answers the same question, with an
  interval rather than a point estimate.

**What this is not.** It is not backboning. §28.2 is explicit about the difference: *"in the
latter we want to remove spurious edges, here we want to use all the information we have and
integrate it in the analysis, no matter how unlikely an edge is to exist"* (p. 401). No function
here deletes an edge. And the interval is over the edges the corpus holds: a missing edge has no
``p``, is in no realisation, and cannot widen any interval -- which is the one sentence §28.1
most needs a reader to carry away.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np

from graphrag.models import MENTION_TIERS, MentionTier
from graphrag.sna.stats import describe

__all__ = [
    "CERTAIN_RULE",
    "EDGE_PROBABILITY",
    "PROBABILITY_RULE",
    "RELATION_RULE",
    "SAMPLES",
    "STATED_PROBABILITY",
    "TIERS",
    "TIER_PROBABILITY",
    "TIER_RULE",
    "Evidence",
    "Expectation",
    "Uncertainty",
    "combine",
    "degree_distribution",
    "edge_probabilities",
    "evidence_report",
    "expectation",
    "expected_degree",
    "expected_density",
    "expected_edges",
    "format_tier_counts",
    "node_expectation",
    "realisations",
    "reliability",
    "render_evidence",
    "render_uncertain",
    "support",
    "tier_counts",
    "tier_probability",
    "uncertain_payload",
    "uncertainty_report",
]

#: Where an edge keeps its probability of existing. One key, used by every network builder and
#: read back by everything here; it survives into every interchange format that carries edge
#: attributes, and an edge without it is read as certain.
EDGE_PROBABILITY = "p"

#: How much one supporting passage is worth, by the tier that found it (§28.1). These are this
#: package's numbers, not the book's -- the book has no corpus and no matcher -- and they are
#: named here so a reader can disagree with one number rather than with the whole idea:
#:
#: ``exact``
#:     the passage contains the entity's name verbatim. Strong, and deliberately short of 1.0:
#:     a verbatim match can still be the wrong sense of an ambiguous name, and extraction can
#:     name a thing the passage does not discuss. Two such passages take the edge to 0.9975.
#: ``unknown``
#:     no tier was recorded -- every mention in a snapshot committed before tiers existed. Set
#:     between ``loose`` and ``exact`` because such a mention *is* one of the two and nobody
#:     wrote down which. It is a placeholder for a missing measurement, so every report says how
#:     many edges rest on one instead of letting the number pass as evidence.
#: ``loose``
#:     the passage contains a significant token of the name and not the name. "Lenny" for "Lenny
#:     Rachitsky" is usually right; "Apple" for "Apple Health" is usually not.
#: ``none``
#:     no passage names the entity at all and the mention was anchored to the document's first
#:     passage so the extraction would not be lost. The document is evidence; *that passage* is
#:     barely any, and an edge drawn through it is close to an artefact of the fallback.
TIER_PROBABILITY: dict[MentionTier, float] = {
    "exact": 0.95,
    "unknown": 0.8,
    "loose": 0.6,
    "none": 0.3,
}

#: The rule in one line, recorded on every network built from mentions so a report can print it.
TIER_RULE = (
    "p per supporting passage from the mention tier (exact "
    f"{TIER_PROBABILITY['exact']}, unknown {TIER_PROBABILITY['unknown']}, loose "
    f"{TIER_PROBABILITY['loose']}, fallback {TIER_PROBABILITY['none']}), the weaker of the two "
    "mentions within a passage, combined across passages as 1 - prod(1 - p) (Atlas §28.1-28.2)"
)

#: What one passage stating a relation is worth. An extraction pass wrote the relation *from*
#: that passage and quoted its evidence, so the passage is the claim rather than a match to it;
#: it is priced as an exact mention is, and k passages combine the same way. Short of 1.0 for the
#: same reason: an extraction pass can misread a sentence, and §28.1's spurious edges are exactly
#: that. What it cannot price is the anchor -- a relation whose quoted evidence matched no passage
#: is anchored to the document's first one, and the graph keeps no record of which those were.
STATED_PROBABILITY = TIER_PROBABILITY["exact"]

#: The relation network's rule in one line.
RELATION_RULE = (
    f"p = {STATED_PROBABILITY} per passage stating the relation, combined across passages as "
    "1 - prod(1 - p) (Atlas §28.1-28.2)"
)

#: And the rule for a network whose edges carry no mention evidence.
CERTAIN_RULE = (
    "p = 1.0 on every edge: this network's edges rest on records the pipeline does not match "
    "fuzzily, so nothing here can put a number below one on them. It is a statement about this "
    "pipeline, not about the world -- the sampling frame still says what was recorded"
)

#: Where a network records which rule priced its edges, for the report's frame.
PROBABILITY_RULE = "p_rule"

#: Where a network built from mentions records how many of them came in at each tier, as
#: ``exact=120; loose=8; unknown=0; none=2``. A string rather than a mapping because a graph
#: attribute has to survive GraphML and igraph, which take scalars only.
TIERS = "tiers"

#: Realisations drawn when a measure has no closed form. Enough for a 95% interval to be stable
#: to about a percentile on the networks here; the cost is this many evaluations of the measure,
#: which is why the number is a parameter of every function that samples.
SAMPLES = 200

#: The interval a report prints unless asked otherwise.
LEVEL = 0.95

#: Below this many realisations an interval is arithmetic on a handful of numbers, not an
#: estimate, and :class:`Expectation` says so instead of printing it as one.
MIN_SAMPLES = 20


def tier_probability(
    tier: MentionTier, probabilities: Mapping[MentionTier, float] = TIER_PROBABILITY
) -> float:
    """What one passage found at this tier is worth. An unrecognised tier reads as ``unknown``."""
    return probabilities.get(tier, probabilities.get("unknown", 1.0))


def support(*probabilities: float) -> float:
    """The probability of one piece of evidence that needs several things to be right at once.

    The minimum, not the product. A co-mention edge needs both mentions in that passage to be
    right, but the same matcher produced both from the same text, so they fail together far more
    often than independence would predict; the product would price a pair of exact matches below
    either of them, which is worse than wrong, it is confidently wrong. The minimum is the upper
    Frechet bound for a conjunction under unknown dependence, and taking the weakest link is the
    reading a caption can defend.

    Returns 1.0 for no arguments: nothing required is nothing that can fail.
    """
    return min(probabilities, default=1.0)


def combine(probabilities: Iterable[float]) -> float:
    """``1 - prod(1 - p)``: the edge exists if any one of its supporting passages is right.

    The independence assumption of §28.2 (p. 400), applied *across* passages, where it is
    defensible: two passages are two readings of the corpus. Empty evidence gives 0.0 -- an edge
    nothing supports is not an edge -- and any ``p = 1`` takes the result to 1.
    """
    absent = 1.0
    seen = False
    for value in probabilities:
        seen = True
        absent *= 1.0 - float(value)
    return 1.0 - absent if seen else 0.0


def edge_probabilities(graph: nx.Graph) -> dict[tuple[str, str], float]:
    """Every edge's ``p``, keyed by the edge as the graph holds it.

    An edge with no recorded probability reads as 1.0. That is the honest default for a network
    that arrived from a file, a flattening or anything else that dropped the attribute: the
    alternative is to invent an uncertainty nobody measured.
    """
    return {(u, v): float(data.get(EDGE_PROBABILITY, 1.0)) for u, v, data in graph.edges(data=True)}


def tier_counts(graph: nx.Graph) -> dict[str, int]:
    """How many mentions of each tier this network was built from, from :data:`TIERS`.

    Empty for a network whose builder recorded none -- the speaker and topic networks, and
    anything read back from a file, which is a different statement from "all of them exact".
    """
    counts: dict[str, int] = {}
    for part in str(graph.graph.get(TIERS, "")).split(";"):
        key, _, value = part.strip().partition("=")
        if key and value.strip().isdigit():
            counts[key] = int(value)
    return counts


def format_tier_counts(counts: Mapping[str, int]) -> str:
    """The counts as the string :data:`TIERS` holds, in tier order."""
    return "; ".join(f"{tier}={counts[tier]}" for tier in MENTION_TIERS if tier in counts)


# ----------------------------------------------------------------- §28.3 the closed forms


def expected_degree(graph: nx.Graph, *, weight: str | None = None) -> dict[str, float]:
    """``E[k_u] = sum_v p_uv``: each node's expected degree, exactly (§28.3, p. 403).

    The book reaches this by averaging over the possible worlds -- *"go over all the possible
    worlds and take the average of the degrees of the node, weighted by how likely the world is
    to exist"* -- and linearity of expectation makes that sum of probabilities the same number
    without enumerating anything. On a directed network it is the total (in + out), the
    convention :func:`graphrag.sna.measures.centrality` uses for ``degree``.

    ``weight`` multiplies each term by that edge attribute, giving the expected weighted degree;
    left as ``None`` it is the plain count.

    On a directed network the sum runs over the node's **incident directed edges**, in and out
    both, so a reciprocated pair contributes twice -- once for ``u -> v`` and once for ``v -> u``
    -- exactly as the classical total degree counts it twice. Undirecting first would merge the
    two arcs into one edge and quietly halve a reciprocated node's expectation.

    **Read it with the book's caveat, which is the point of the section.** *"However, that would
    be a bit silly: the degree is a count, it shouldn't be a continuous number. The
    interpretability of what you'd do by averaging would go out of the window"* (p. 403): two
    nodes with radically different topologies can both come out at 3.4, and no node ever has
    degree 3.4. :func:`degree_distribution` is the answer the section actually recommends.
    Undefined for a node not in the graph, which raises ``KeyError``.
    """
    scores: dict[str, float] = {}
    for node in graph.nodes:
        total = 0.0
        for other, data in _incident(graph, node):
            probability = float(data.get(EDGE_PROBABILITY, 1.0))
            total += probability * (float(data.get(weight, 1.0)) if weight else 1.0)
            if other == node:  # a self loop contributes twice to a degree, as it does classically
                total += probability * (float(data.get(weight, 1.0)) if weight else 1.0)
        scores[node] = total
    return scores


def _incident(graph: nx.Graph, node: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Every edge that counts toward ``node``'s degree, as ``(other endpoint, edge data)``.

    On an undirected graph that is the node's edges. On a directed one it is its out-edges *and*
    its in-edges, because the total degree of §6.2 is in plus out and a reciprocated pair is two
    edges, not one: ``a -> b`` and ``b -> a`` are two claims with two probabilities. Reading the
    undirected view here would collapse them into one and halve the expectation of every node in
    a reciprocated pair.
    """
    if not graph.is_directed():
        return [(other, data) for _, other, data in graph.edges(node, data=True)]
    out = [(other, data) for _, other, data in graph.out_edges(node, data=True)]
    return out + [(other, data) for other, _, data in graph.in_edges(node, data=True)]


def degree_distribution(graph: nx.Graph, node: str) -> dict[int, float]:
    """The probability of each possible degree for one node, exactly (§28.3, Figure 28.7a).

    The distribution the section says to report instead of the expected degree: *"It is much
    better to give each node a probability distribution for each possible degree value which
    allows you to differentiate between the nodes in Figure 28.6"* (p. 403). Node 3 of the
    book's Figure 28.4 comes back here as its Figure 28.7(a).

    This is the Poisson-binomial distribution of the node's incident edge probabilities, summed
    by convolution rather than by enumerating the ``2^k`` worlds of its neighbourhood: exact, and
    linear in the degree. Keys are every degree with non-zero probability, ``0`` included; the
    values sum to 1. A node with no edges gives ``{0: 1.0}``.

    The incident edges are the ones :func:`expected_degree` sums, so on a directed network this
    is the distribution of the **total** degree and a reciprocated pair contributes two
    Bernoullis rather than one. The mean of the distribution returned here is therefore
    ``expected_degree(graph)[node]``, on any network, which is the property that makes the two
    readable in one table.

    Raises ``KeyError`` for a node the graph does not hold.
    """
    if node not in graph:
        msg = f"{node!r} is not in this network"
        raise KeyError(msg)
    distribution = [1.0]
    for _, data in _incident(graph, node):
        probability = float(data.get(EDGE_PROBABILITY, 1.0))
        shifted = [0.0, *distribution]
        distribution = [
            value * (1.0 - probability) + shifted[index] * probability
            for index, value in enumerate([*distribution, 0.0])
        ]
    return {degree: value for degree, value in enumerate(distribution) if value > 0.0}


def expected_edges(graph: nx.Graph) -> float:
    """``sum_e p_e``: how many of this network's edges exist, in expectation (§28.3).

    The same linearity argument as :func:`expected_degree`, one level up. On the karate club
    with the one disputed edge of p. 396 at ``p = 0.5`` it is 77.5 -- which is the chapter's
    opening complaint stated as a number rather than as a footnote.
    """
    return float(sum(edge_probabilities(graph).values()))


def expected_density(graph: nx.Graph) -> float:
    """Expected edges over possible pairs: density with the uncertainty priced in.

    Undefined on a network with fewer than two nodes, where there are no pairs to be dense in;
    returns 0.0 there, as ``networkx.density`` does.
    """
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    pairs = float(n * (n - 1)) if graph.is_directed() else n * (n - 1) / 2.0
    return expected_edges(graph) / pairs


# ----------------------------------------------------------------- §28.2 possible worlds


def realisations(
    graph: nx.Graph, samples: int = SAMPLES, seed: int | None = None
) -> Iterator[nx.Graph]:
    """Sampled possible worlds: each edge kept with probability ``p``, independently (§28.2).

    One *possible world* is a classical graph, and the network's own measure is a distribution
    over all ``2^m`` of them, weighted by how likely each is (p. 401). Enumerating them is
    hopeless past a handful of edges -- the book's four-edge example already has sixteen -- so
    these are drawn instead, and every world comes back with **all** the nodes: an edge failing
    isolates a node, it does not delete it, and a measure that silently changed its ``n`` between
    realisations would be measuring two different networks.

    Node and edge attributes ride along, so a measure that reads ``weight`` reads the same
    weights it would on the observed network. Deterministic given ``seed``.
    """
    if samples < 1:
        msg = f"samples must be at least 1, got {samples}"
        raise ValueError(msg)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    edges = [(u, v, data) for u, v, data in graph.edges(data=True)]
    for _ in range(samples):
        world: nx.Graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
        world.graph.update(graph.graph)
        world.add_nodes_from(graph.nodes(data=True))
        world.add_edges_from(
            (u, v, data)
            for u, v, data in edges
            if rng.random() < float(data.get(EDGE_PROBABILITY, 1.0))
        )
        yield world


@dataclass(frozen=True)
class Expectation:
    """One measure over the sampled possible worlds: its mean, its spread and its interval.

    ``observed`` is the measure on the network as it stands -- every edge present, which is the
    world the rest of the package reports. It is not the mean: the mean is pulled down by the
    worlds where an edge is missing, and the gap between the two is the uncertainty priced in.

    ``low`` and ``high`` are percentiles of the sample, not ``mean ± 1.96 sd``. A measure over
    realisations is rarely symmetric -- betweenness and modularity least of all -- and a
    percentile interval does not assume it is.

    ``certain`` is set when every edge of the network had ``p = 1``: the realisations are all the
    observed network, the interval collapses onto it, and the report says the network carried no
    uncertainty rather than implying it was tested for some.
    """

    name: str
    observed: float
    mean: float
    sd: float
    low: float
    high: float
    level: float = LEVEL
    samples: int = 0
    certain: bool = False
    caveat: str = ""

    @property
    def width(self) -> float:
        """How wide the interval is, in the measure's own units."""
        return self.high - self.low

    def overlaps(self, other: Expectation) -> bool:
        """Whether two intervals overlap, i.e. whether these two values can be ranked.

        Two nodes whose intervals overlap are not in a known order: some possible worlds put one
        first and some the other, and printing them one above the other is a presentation choice
        rather than a finding.
        """
        return self.low <= other.high and other.low <= self.high

    @property
    def text(self) -> str:
        """``0.4120 (mean 0.3987, 95% [0.3100, 0.4800])``, as the report prints it."""
        if self.certain:
            return f"{self.observed:.4f} (certain)"
        return (
            f"{self.observed:.4f} (mean {self.mean:.4f}, "
            f"{self.level:.0%} [{self.low:.4f}, {self.high:.4f}])"
        )


def _interval(values: Sequence[float], level: float) -> tuple[float, float]:
    """The central ``level`` percentile interval of a sample."""
    if not values:
        return (0.0, 0.0)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [tail, 100.0 - tail])
    return (float(low), float(high))


def _certain(graph: nx.Graph) -> bool:
    """Whether every edge of this network is certain, so sampling has nothing to vary."""
    return all(p >= 1.0 for p in edge_probabilities(graph).values())


def _deadline(max_seconds: float | None) -> float | None:
    """An absolute ``time.monotonic()`` timestamp ``max_seconds`` from now, or ``None`` for no
    budget at all (ATL-F2). A duration in, a fixed point in time out, so a caller that threads
    the same value through several measures in a row (:func:`uncertainty_report` does, over every
    centrality plus clustering plus modularity) shares one wall-clock budget rather than handing
    each measure a fresh one -- see :func:`_remaining`.
    """
    return None if max_seconds is None else time.monotonic() + max_seconds


def _expired(deadline: float | None) -> bool:
    """Whether the wall clock has reached ``deadline``, an absolute ``time.monotonic()`` stamp."""
    return deadline is not None and time.monotonic() >= deadline


def _remaining(deadline: float | None) -> float | None:
    """Seconds left before ``deadline``, floored at 0.0, or ``None`` when there is no budget."""
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _summarise(
    values: Sequence[float], observed: float, level: float
) -> tuple[float, float, float, float]:
    """``(mean, sd, low, high)`` from a sample that may be empty.

    An empty sample -- every realisation stopped by ``max_seconds`` before this measure got to
    run one (ATL-F2) -- collapses onto the observed value rather than onto ``(0.0, 0.0)``:
    "we don't know" is not the same statement as "zero", and :func:`_caveat` is what tells a
    reader which one this is.
    """
    if not values:
        return observed, 0.0, observed, observed
    low, high = _interval(values, level)
    return float(np.mean(values)), float(np.std(values)), low, high


def _caveat(samples: int, certain: bool, *, requested: int = 0) -> str:
    """The sentence an interval needs beside it, or empty when it needs none.

    ``requested`` is the sample count that was asked for; ``samples < requested`` can only
    happen when a ``max_seconds`` budget stopped the realisations early (ATL-F2, §28.2), since
    without one every measure here always runs exactly ``requested`` of them.
    """
    if certain:
        return (
            "Every edge of this network has p = 1, so every realisation is the observed network "
            "and the interval is a point. That is not a finding about the data: it means this "
            "pipeline records no uncertainty for these edges (§28.1 lists the errors it still "
            "cannot see)."
        )
    if requested and samples < requested:
        return (
            f"--max-seconds stopped this measure after {samples:,} of the {requested:,} "
            "requested realisations; the interval above is over what finished, not the full "
            "sample, and can be wider than it would be at the full count. Raise the budget, or "
            "drop --uncertain-expensive, to finish the rest."
        )
    if samples < MIN_SAMPLES:
        return (
            f"{samples} realisation(s): too few for a percentile interval to mean much. It is "
            f"reported because it was asked for; raise the sample count above {MIN_SAMPLES}."
        )
    return ""


def _measured(
    graph: nx.Graph,
    measure: Callable[[nx.Graph], float],
    samples: int,
    seed: int | None,
    *,
    deadline: float | None = None,
) -> list[float]:
    """One graph-level measure over each sampled world, stopping at ``deadline`` if given.

    Checked before the first world (so a deadline already passed by the time this measure's turn
    comes stops it before it starts) and after scoring each one (so the realisation that crosses
    the budget still finishes and counts, and the loop stops there rather than starting another,
    per ATL-F2's rule).
    """
    if _expired(deadline):
        return []
    values: list[float] = []
    for world in realisations(graph, samples, seed):
        values.append(float(measure(world)))
        if _expired(deadline):
            break
    return values


def expectation(
    graph: nx.Graph,
    measure: Callable[[nx.Graph], float],
    *,
    name: str = "",
    samples: int = SAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
    max_seconds: float | None = None,
) -> Expectation:
    """A whole-network measure's expectation and interval over the possible worlds (§28.2).

    ``measure`` is any callable from a graph to a number -- ``nx.average_clustering``, the
    modularity of a fixed partition, :func:`graphrag.sna.measures.summary` read for one key. It
    is called once per realisation, so the cost of this function is ``samples`` times the cost of
    the measure, which is the whole reason the book reaches for sparsification (p. 402) and this
    function takes a sample count rather than assuming one.

    ``max_seconds``, when given, stops drawing realisations once that many seconds have passed
    and reports the interval over whatever finished (ATL-F2's own addition, not the book's): the
    stated budget the ticket's docstring calls for, rather than a command that hangs on a large
    network. ``Expectation.samples`` is then the actual count, short of ``samples``, and
    ``Expectation.caveat`` says so.

    The measure must be defined on every possible world, including ones in which an edge is
    missing and the network has fallen apart; if it raises on a disconnected graph, wrap it.
    :func:`node_expectation` is the same contract for a measure that returns one number per node.
    """
    values = _measured(graph, measure, samples, seed, deadline=_deadline(max_seconds))
    observed = float(measure(graph))
    certain = _certain(graph)
    mean, sd, low, high = _summarise(values, observed, level)
    return Expectation(
        name=name or getattr(measure, "__name__", "measure"),
        observed=observed,
        mean=mean,
        sd=sd,
        low=low,
        high=high,
        level=level,
        samples=len(values),
        certain=certain,
        caveat=_caveat(len(values), certain, requested=samples),
    )


def node_expectation(
    graph: nx.Graph,
    measure: Callable[[nx.Graph], Mapping[str, float]],
    *,
    name: str = "",
    samples: int = SAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
    max_seconds: float | None = None,
) -> dict[str, Expectation]:
    """The same as :func:`expectation`, for a measure that scores every node.

    One realisation is one call, so all ``n`` nodes come out of the same possible world and their
    intervals are comparable with each other -- which is what makes
    :meth:`Expectation.overlaps` a statement about a ranking rather than about two unrelated
    samples. A node the measure omits in some world scores 0.0 there rather than being dropped:
    the node is in the network, its score in that world is simply nothing.

    ``max_seconds`` is the same budget :func:`expectation` takes, checked once before the loop
    and once after each realisation, so every node's :class:`Expectation` here shares the same
    (possibly short of ``samples``) realisation count (ATL-F2).
    """
    deadline = _deadline(max_seconds)
    per_node: dict[str, list[float]] = {node: [] for node in graph.nodes}
    ran = 0
    if not _expired(deadline):
        for world in realisations(graph, samples, seed):
            scores = measure(world)
            for node in per_node:
                per_node[node].append(float(scores.get(node, 0.0)))
            ran += 1
            if _expired(deadline):
                break
    observed = measure(graph)
    certain = _certain(graph)
    caveat = _caveat(ran, certain, requested=samples)
    label = name or getattr(measure, "__name__", "measure")
    out: dict[str, Expectation] = {}
    for node, values in per_node.items():
        obs = float(observed.get(node, 0.0))
        mean, sd, low, high = _summarise(values, obs, level)
        out[node] = Expectation(
            name=label,
            observed=obs,
            mean=mean,
            sd=sd,
            low=low,
            high=high,
            level=level,
            samples=ran,
            certain=certain,
            caveat=caveat,
        )
    return out


def reliability(
    graph: nx.Graph,
    source: str,
    target: str,
    *,
    samples: int = SAMPLES,
    seed: int | None = None,
) -> float:
    """The probability two nodes end up in the same component: their *reliability* (§28.3).

    *"In probabilistic networks, we can estimate a probability that two nodes are reachable --
    i.e. they are part of the same connected component. We call this probability 'reliability'.
    The reliability of node pairs (1, 2) is simply the sum of the probabilities of all possible
    worlds in which they are in the same connected component"* (p. 406). Two nodes joined by an
    edge can still be unreachable in most worlds, which is the section's Figure 28.10.

    That sum is over ``2^m`` worlds and has no closed form in general -- computing it exactly is
    the classical network-reliability problem, which is #P-hard -- so this is the Monte Carlo
    estimate: the share of sampled worlds in which the two are connected. Its own uncertainty is
    about ``sqrt(r(1-r)/samples)``; at the default that is under two points.

    Direction is ignored, because "same connected component" is the weakly connected question
    (§10.4). Raises ``KeyError`` for a node the graph does not hold; a node paired with itself is
    reliable with probability 1.
    """
    for node in (source, target):
        if node not in graph:
            msg = f"{node!r} is not in this network"
            raise KeyError(msg)
    if source == target:
        return 1.0
    hits = 0
    for world in realisations(graph, samples, seed):
        view = world.to_undirected(as_view=True) if world.is_directed() else world
        if nx.has_path(view, source, target):
            hits += 1
    return hits / samples


# ----------------------------------------------------------------- §28.1 error estimation


@dataclass(frozen=True)
class Evidence:
    """What this network's edges rest on, and how thin the thinnest of them are (§28.1).

    Always computed, never behind a flag: an edge's evidence is a property of the network, not an
    extra analysis, and the counts here are what decide whether the rest of a report is worth
    reading. ``unit`` is the network's own unit of evidence -- passages shared, documents shared,
    passages stating the relation -- so "rests on one" means one of those.
    """

    network: str
    unit: str
    rule: str
    #: The sampling frame carried on the graph -- persona, what a node and an edge are, and every
    #: filter that narrowed it -- repeated here so this section prints it rather than assuming a
    #: reader scrolled up. Empty for a graph nobody built here.
    frame: str
    nodes: int
    edges: int
    #: Edges whose ``p`` is 1.0. On a mention network, an edge several exact passages support.
    certain: int
    #: Edges resting on a single unit of evidence, and the share of total weight they carry.
    single: int
    single_weight_share: float
    #: Of those, the ones whose single piece of evidence was a loose match or weaker.
    single_weak: int
    #: How many of the mentions behind this network came in at each tier, when it recorded them.
    tiers: dict[str, int] = field(default_factory=dict)
    mean_p: float = 1.0
    median_p: float = 1.0
    minimum_p: float = 1.0
    #: Set when no edge carries a probability at all, which is a network that lost them rather
    #: than one whose edges are certain.
    unpriced: bool = False

    @property
    def unknown_tier(self) -> int:
        """Mentions with no recorded tier: the measurement §28.1 would want and does not have."""
        return self.tiers.get("unknown", 0)

    @property
    def single_share(self) -> float:
        """The share of edges resting on one unit of evidence."""
        return self.single / self.edges if self.edges else 0.0


def evidence_report(graph: nx.Graph) -> Evidence:
    """§28.1 for one network: what its edges rest on, counted.

    The chapter's own way of putting the question is which edges are *spurious* -- false
    positives that should not be there -- and which are *missing*. The second half is
    unanswerable from inside the network and stays unanswerable (see :func:`render_evidence`);
    the first half is exactly what an edge on one thin piece of evidence is a candidate for, so
    they are counted rather than described.

    ``weight`` is read as the number of units of evidence behind an edge, which is what every
    builder in :mod:`graphrag.sna.export` makes it. An edge of weight 1 rests on one passage, one
    document or one stated relation, and if that one is wrong the edge is not there.
    """
    probabilities = edge_probabilities(graph)
    weights = {(u, v): float(data.get("weight", 1.0)) for u, v, data in graph.edges(data=True)}
    total_weight = sum(weights.values())
    single = [edge for edge, weight in weights.items() if weight <= 1.0]
    weak = TIER_PROBABILITY["loose"]
    values = list(probabilities.values())
    summary = describe(values) if values else None
    return Evidence(
        network=str(graph.graph.get("network", "")),
        unit=str(graph.graph.get("unit", "units of evidence")),
        rule=str(graph.graph.get(PROBABILITY_RULE, "")),
        frame=str(graph.graph.get("frame", "")),
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        certain=sum(1 for p in values if p >= 1.0),
        single=len(single),
        single_weight_share=(
            sum(weights[edge] for edge in single) / total_weight if total_weight else 0.0
        ),
        single_weak=sum(1 for edge in single if probabilities[edge] <= weak),
        tiers=tier_counts(graph),
        mean_p=summary.mean if summary else 1.0,
        median_p=summary.median if summary else 1.0,
        minimum_p=summary.minimum if summary else 1.0,
        unpriced=bool(values)
        and all(EDGE_PROBABILITY not in data for _, _, data in graph.edges(data=True)),
    )


#: What the book says a measurement error of any size does to the numbers above it, and what it
#: says about the half of the problem no report can measure. Printed under every evidence table.
ERROR_SENTENCES: tuple[str, ...] = (
    "An edge resting on one passage is the edge a spurious-edge rate acts on first: if that "
    "passage was matched wrongly, the edge is not there, and every measure that walks through "
    "it -- betweenness, closeness, any path -- is wrong by more than one edge's worth, because "
    "removing one edge can split a component.",
    "The other half of §28.1 is missing edges, and nothing here can count them. A tie the corpus "
    "never wrote down has no row, no probability and no place in any realisation, so the "
    "interval below is over the edges we have and not over the edges we missed. A report that "
    "reads it as total uncertainty has made the chapter's own mistake.",
    "Correcting either half needs something this pipeline does not have: the network measured "
    "several times (p. 398), or a generative model good enough to say which edges are spurious "
    "-- and a poor model, the book warns, gives every edge 'a 50-50 chance of having or not "
    "having the edge, which is pretty useless' (p. 400).",
)


# ----------------------------------------------------------------- the report


@dataclass(frozen=True)
class Uncertainty:
    """Everything ``analyze --uncertain`` computed, ready to render (§28.2-28.3).

    ``centralities`` holds one entry per centrality the report ranks, each a list of
    ``(node, Expectation)`` in the observed ranking's order, so the table can be read beside the
    certain one. ``ties`` counts the adjacent pairs in those rankings whose intervals overlap,
    which is the number that decides whether a ranking is a ranking.
    """

    samples: int
    seed: int | None
    level: float
    evidence: Evidence
    expected_edges: float
    expected_density: float
    observed_edges: int
    observed_density: float
    degrees: list[tuple[str, float, float, int]] = field(default_factory=list)
    """``(node, observed degree, expected degree, most likely degree)`` for the top nodes."""
    centralities: dict[str, list[tuple[str, Expectation]]] = field(default_factory=dict)
    ties: dict[str, int] = field(default_factory=dict)
    #: Average clustering over the sampled worlds. Sampled and not closed form: chapter 28 gives
    #: none for it, and §28.1's own argument for doing it this way is that a measure like the
    #: clustering coefficient is better read as a distribution of potential values than as the
    #: number the single likeliest network happens to have (p. 398).
    clustering: Expectation | None = None
    modularity: Expectation | None = None
    notes: list[str] = field(default_factory=list)
    #: The ``--max-seconds`` budget this run was given, or ``None`` for no budget (ATL-F2). One
    #: wall-clock allowance shared across every measure below, not a fresh one each; see
    #: :func:`uncertainty_report`. Recorded here so the report's frame can name it even when
    #: nothing was actually truncated.
    max_seconds: float | None = None


def _most_likely_degree(graph: nx.Graph, node: str) -> int:
    """The mode of a node's exact degree distribution: a degree it could actually have.

    A tie -- two degrees equally likely, which is what one uncertain edge on an otherwise certain
    node gives -- goes to the smaller, so the number never overstates how connected a node is.
    """
    distribution = degree_distribution(graph, node)
    return max(distribution.items(), key=lambda item: (item[1], -item[0]))[0]


def uncertainty_report(
    graph: nx.Graph,
    centralities: Mapping[str, Sequence[tuple[str, float]]],
    *,
    measure_for: Callable[[str], Callable[[nx.Graph], Mapping[str, float]]],
    modularity: Callable[[nx.Graph], float] | None = None,
    samples: int = SAMPLES,
    seed: int | None = None,
    level: float = LEVEL,
    max_seconds: float | None = None,
) -> Uncertainty:
    """Run chapter 28 over one network: the closed forms, then the sampled ones.

    ``centralities`` is the report's own ranking per measure -- the nodes it printed and in which
    order -- and ``measure_for`` maps a measure's name to the callable that scores every node, so
    this module never learns which centralities exist (the caller decides which are worth the
    cost, e.g. by dropping the expensive ones -- :data:`graphrag.sna.measures.
    EXPENSIVE_CENTRALITIES` -- unless asked to keep them). ``modularity`` is optional and is the
    partition's modularity as a function of a graph, evaluated on each realisation with the
    partition held fixed: the question is what the uncertainty does to *this* grouping's score,
    not what a fresh Louvain run would find in each possible world.

    ``max_seconds``, when given, is one wall-clock budget shared across everything this function
    computes -- every centrality's realisations, then clustering, then modularity -- rather than
    a fresh one per measure (ATL-F2): a budget spent ranking betweenness is not handed back for
    ranking closeness. Whichever measure is running when the clock runs out finishes its current
    realisation and stops there; every measure after it gets none, and each one's
    ``Expectation.caveat`` says so next to its numbers, per measure, not only once at the top.

    One pass of realisations per measure, seeded identically, so the tables are all drawn from
    the same set of worlds and can be read against each other.
    """
    deadline = _deadline(max_seconds)
    evidence = evidence_report(graph)
    ranked: dict[str, list[tuple[str, Expectation]]] = {}
    ties: dict[str, int] = {}
    for kind, scores in centralities.items():
        if not scores:
            continue
        estimates = node_expectation(
            graph,
            measure_for(kind),
            name=kind,
            samples=samples,
            seed=seed,
            level=level,
            max_seconds=_remaining(deadline),
        )
        rows = [(node, estimates[node]) for node, _ in scores if node in estimates]
        ranked[kind] = rows
        ties[kind] = sum(1 for (_, a), (_, b) in pairwise(rows) if a.overlaps(b))
    top = [node for node, _ in next(iter(centralities.values()), [])] or list(graph.nodes)[:20]
    expected = expected_degree(graph)
    degrees = [
        (node, float(graph.degree(node)), expected[node], _most_likely_degree(graph, node))
        for node in top
        if node in expected
    ]
    return Uncertainty(
        samples=samples,
        seed=seed,
        level=level,
        evidence=evidence,
        clustering=expectation(
            graph,
            lambda world: float(nx.average_clustering(world, weight="weight")),
            name="average clustering",
            samples=samples,
            seed=seed,
            level=level,
            max_seconds=_remaining(deadline),
        ),
        expected_edges=expected_edges(graph),
        expected_density=expected_density(graph),
        observed_edges=graph.number_of_edges(),
        observed_density=float(nx.density(graph)),
        degrees=degrees,
        centralities=ranked,
        ties=ties,
        max_seconds=max_seconds,
        modularity=(
            expectation(
                graph,
                modularity,
                name="modularity",
                samples=samples,
                seed=seed,
                level=level,
                max_seconds=_remaining(deadline),
            )
            if modularity is not None
            else None
        ),
    )


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def render_evidence(evidence: Evidence) -> list[str]:
    """The §28.1 section: what the edges rest on, always printed."""
    lines = [
        "## Evidence behind the edges (Atlas §28.1)",
        "",
        f"**Frame.** n = {evidence.nodes:,} nodes and {evidence.edges:,} edges of the "
        f"{evidence.network or 'network'} network, whose unit of evidence is {evidence.unit}."
        + (f" {evidence.frame}" if evidence.frame else "")
        + " **Null model.** None: this section counts what the corpus recorded, it does not test "
        "it against anything. "
        f"**Rule.** {evidence.rule or 'no probability rule was recorded for this network'}.",
        "",
    ]
    if evidence.edges == 0:
        return [*lines, "No edges, so nothing rests on anything.", ""]
    rows = [
        ["edges", f"{evidence.edges:,}"],
        ["certain (p = 1)", f"{evidence.certain:,}"],
        [
            f"resting on one piece of evidence ({evidence.unit})",
            f"{evidence.single:,} ({evidence.single_share:.1%} of edges, "
            f"{evidence.single_weight_share:.1%} of total weight)",
        ],
        ["of those, on a loose match or weaker", f"{evidence.single_weak:,}"],
        [
            "p: mean / median / minimum",
            f"{evidence.mean_p:.3f} / {evidence.median_p:.3f} / {evidence.minimum_p:.3f}",
        ],
    ]
    if evidence.tiers:
        rows.append(["mentions by tier", format_tier_counts(evidence.tiers)])
    lines += _table(["what", "how many"], rows)
    if evidence.unpriced:
        lines += [
            "> No edge of this network carries a probability. Something between the builder and "
            "here dropped it -- a flattening of layers, a read back from a format that keeps no "
            "edge attributes -- so every edge below is treated as certain, which is a statement "
            "about what survived rather than about the evidence. Rebuild the network to price "
            "it.",
            "",
        ]
    if evidence.unknown_tier:
        lines += [
            f"> {evidence.unknown_tier:,} of the mentions behind this network carry no tier. "
            "They were written before the tier was recorded -- every mention in a snapshot "
            f"committed before it existed -- and are priced at {TIER_PROBABILITY['unknown']}, "
            "which is a placeholder between an exact and a loose match rather than a "
            "measurement. Re-importing the extraction files records the real tiers.",
            "",
        ]
    lines += [f"- {sentence}" for sentence in ERROR_SENTENCES]
    return [*lines, ""]


def render_uncertain(report: Uncertainty) -> list[str]:
    """The §28.2-28.3 section: the closed forms, the intervals, and which ranks are ties."""
    evidence = report.evidence
    lines = [
        *render_evidence(evidence),
        "## Under uncertainty (Atlas §28.2-28.3)",
        "",
        f"**Frame.** The same n = {evidence.nodes:,} nodes and {evidence.edges:,} edges of the "
        f"{evidence.network or 'network'} network"
        + (f" -- {evidence.frame}" if evidence.frame else "")
        + " -- read as a probabilistic network G = (V, E, Pi) (§28.2): each edge exists with its "
        f"own p, and every number below is an expectation over {report.samples:,} sampled "
        "possible worlds"
        + (f", seed {report.seed}" if report.seed is not None else ", unseeded")
        + f". **Interval.** The central {report.level:.0%} of those samples, a percentile rather "
        "than a standard error. **Null model.** None, and none is wanted: this is not a test "
        "against chance, it is the spread of one measure over the worlds the evidence allows. "
        "The interval is over the edges we have, never over the edges we missed (§28.1)."
        + (
            f" **Budget.** --max-seconds {report.max_seconds:g}s, shared across every measure "
            "below rather than given afresh to each one; a measure whose actual realisation "
            "count is short of the number above says so next to its own numbers."
            if report.max_seconds is not None
            else ""
        ),
        "",
        "### Size, in closed form (§28.3)",
        "",
    ]
    lines += _table(
        ["measure", "observed", "expected"],
        [
            ["edges", f"{report.observed_edges:,}", _num(report.expected_edges)],
            ["density", _num(report.observed_density), _num(report.expected_density)],
        ],
    )
    if report.degrees:
        lines += [
            "### Degree (§28.3, p. 403)",
            "",
            "`E[k] = sum_v p_uv`, computed exactly rather than sampled. The book's own warning "
            "sits in this table: an expected degree is a continuous number and *"
            '"the degree is a count, it shouldn\'t be a continuous number"* -- so the last '
            "column is the mode of the node's exact degree distribution (Figure 28.7a), which "
            "is a degree the node could actually have.",
            "",
        ]
        lines += _table(
            ["node", "observed", "expected", "most likely"],
            [
                [node, _num(observed), _num(expected), str(likely)]
                for node, observed, expected, likely in report.degrees
            ],
        )
    for kind, rows in report.centralities.items():
        overlaps = report.ties.get(kind, 0)
        # The realisation count actually run, which a --max-seconds budget can leave short of
        # report.samples (ATL-F2); every node's Expectation in one node_expectation() call
        # shares one realisation loop, so the first row's count speaks for the whole table.
        actual = rows[0][1].samples if rows else report.samples
        caveat = rows[0][1].caveat if rows else ""
        lines += [
            f"### {kind} under uncertainty",
            "",
            f"Ranked as observed; the interval is over {actual:,} realisation(s). "
            + (
                f"{overlaps} of the {max(len(rows) - 1, 0)} adjacent pairs have overlapping "
                "intervals: those neighbours are not ranked, they are tied."
                if overlaps
                else "No two adjacent intervals overlap, so this ordering survives the "
                "uncertainty in the edges."
            ),
            "",
        ]
        if caveat:
            lines += [f"> {caveat}", ""]
        lines += _table(
            ["rank", "node", "observed", f"mean ± {report.level:.0%} interval", "separated"],
            [
                [
                    str(index),
                    node,
                    _num(estimate.observed),
                    f"{_num(estimate.mean)} ± [{_num(estimate.low)}, {_num(estimate.high)}]",
                    # Separated *from the rank below*, so the last row has nothing to be
                    # separated from and says so rather than claiming a gap it cannot have.
                    "-"
                    if index == len(rows)
                    else ("no" if estimate.overlaps(rows[index][1]) else "yes"),
                ]
                for index, (node, estimate) in enumerate(rows, start=1)
            ],
        )
    if report.clustering is not None:
        lines += [
            "### Clustering under uncertainty",
            "",
            "Sampled rather than closed form: chapter 28 gives no closed form for the clustering "
            "coefficient, and §28.1 argues that the distribution is the better answer anyway -- "
            'rather than the coefficient of the single likeliest network, *"you get a '
            "distribution of potential values, each weighted by how likely their corresponding A "
            'is"* (p. 398), because that likeliest network *"could still be pretty unlikely in '
            'absolute terms, and thus its clustering coefficient might be very wrong"*.',
            "",
            f"- average clustering (weighted): {report.clustering.text}",
            "",
        ]
        if report.clustering.caveat:
            lines += [f"> {report.clustering.caveat}", ""]
    if report.modularity is not None:
        modularity = report.modularity
        lines += [
            "### Modularity under uncertainty",
            "",
            f"- the partition is held fixed and re-scored on each realisation: {modularity.text}",
            "",
        ]
        if modularity.caveat:
            lines += [f"> {modularity.caveat}", ""]
    lines += [f"- {note}" for note in report.notes]
    return [*lines, ""]


def uncertain_payload(report: Uncertainty) -> dict[str, object]:
    """The same section as plain JSON-able data."""
    evidence = report.evidence
    return {
        "samples": report.samples,
        "seed": report.seed,
        "level": report.level,
        "max_seconds": report.max_seconds,
        "evidence": {
            "network": evidence.network,
            "unit": evidence.unit,
            "rule": evidence.rule,
            "frame": evidence.frame,
            "nodes": evidence.nodes,
            "edges": evidence.edges,
            "certain": evidence.certain,
            "single": evidence.single,
            "single_share": evidence.single_share,
            "single_weight_share": evidence.single_weight_share,
            "single_weak": evidence.single_weak,
            "tiers": dict(evidence.tiers),
            "mean_p": evidence.mean_p,
            "median_p": evidence.median_p,
            "minimum_p": evidence.minimum_p,
            "caveats": list(ERROR_SENTENCES),
        },
        "expected_edges": report.expected_edges,
        "expected_density": report.expected_density,
        "observed_edges": report.observed_edges,
        "observed_density": report.observed_density,
        "degree": [
            {"node": node, "observed": observed, "expected": expected, "most_likely": likely}
            for node, observed, expected, likely in report.degrees
        ],
        "centrality": {
            kind: [
                {
                    "node": node,
                    "observed": estimate.observed,
                    "mean": estimate.mean,
                    "sd": estimate.sd,
                    "low": estimate.low,
                    "high": estimate.high,
                }
                for node, estimate in rows
            ]
            for kind, rows in report.centralities.items()
        },
        # The realisation count actually run per centrality, which --max-seconds can leave short
        # of "samples" above (ATL-F2); every node in one centrality's table shares one count.
        "centrality_samples": {
            kind: rows[0][1].samples for kind, rows in report.centralities.items() if rows
        },
        "ties": dict(report.ties),
        "clustering": (
            {
                "observed": report.clustering.observed,
                "mean": report.clustering.mean,
                "sd": report.clustering.sd,
                "low": report.clustering.low,
                "high": report.clustering.high,
            }
            if report.clustering is not None
            else None
        ),
        "modularity": (
            {
                "observed": report.modularity.observed,
                "mean": report.modularity.mean,
                "sd": report.modularity.sd,
                "low": report.modularity.low,
                "high": report.modularity.high,
            }
            if report.modularity is not None
            else None
        ),
        "notes": list(report.notes),
    }
