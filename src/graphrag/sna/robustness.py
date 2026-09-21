"""What it takes to break a network: random failure, deliberate attack, cascade, coupling.

Chapter 22 of *The Atlas for the Aspiring Network Scientist* (pp. 314-326) changes the
perspective of the epidemic chapters one more time: *"Rather than propagating a disease, we
propagate failures"* (p. 314). Nodes stop working, for accidental or deliberate reasons, and the
question is whether what is left still holds together. The chapter's own criterion is the one
every number here is built on: *"Our criterion to say whether a network is still fulfilling its
purpose is the share of nodes part of its largest component. If there still is a path between
all or most nodes in the network, even if it becomes longer, the network still works. However,
when nodes start breaking down in multiple components and getting isolated ... then the network
is failing"* (p. 315).

The four sections, and what each is here:

**§22.1 Random Failures.** Remove nodes uniformly at random and watch the giant component. A
G(n,p) network *"will withstand small failures"* and then goes over a cliff -- *"we reach a
critical value of |R| beyond which the GCC disappears and the network effectively breaks down
... At some point there is a phase transition"* (p. 316) -- while a heavy-tailed network shows
*"no trace of the phase transition"*, because *"when you pick a node at random and you make it
fail, you're overwhelmingly more likely to pick one of the peripheral low degree ones"* (p.
317). :func:`removal_curve` is that plot, with the second signal of p. 315 beside it: the mean
size of the components that are *not* the giant one, which rises as the network shatters.
:func:`molloy_reed` is the closed form for where the cliff is.

**§22.2 Targeted Attacks.** *"an attacker would not target nodes at random. They would go after
the nodes allowing them to maximize the amount of damage while minimizing the effort required.
This translates into prioritizing attacks to the nodes with the highest degree"* (p. 318). The
same :func:`removal_curve` with a ranking instead of a shuffle, and :func:`attack_summary` puts
the two side by side, which is the comparison figures 22.5 and 22.6 make. Degree is the
chapter's attack; the other rankings are this package's centralities (§14) asked the same
question, and they are not all the same attack.

**§22.3 Chain Effects.** Failures propagate: *"whatever power that generator was providing has
to come from somewhere else"* (p. 320). :func:`cascade` is the chapter's load-capacity model
(the Failure Propagation model it cites at footnote 12, p. 320): every node has a load and a
capacity, a failed node's load is redistributed over its neighbours, and *"If the new load
exceeds the capacity of the node, also this node shuts down"* (p. 320).

**§22.4 Interdependent Networks.** Two networks whose nodes need each other, where *"even if
the two networks are resilient to random failures in isolation, the inter-dependencies cause
them to be fragile to failures propagating back and forth between them"* (p. 322).
:func:`interdependent` runs that back-and-forth to its fixed point and returns the mutual giant
component; :func:`interdependent_curve` sweeps the initial damage, because the collapse is
abrupt and an endpoint does not show it.

**What the book defines and what this adds.**

*The critical fraction.* The chapter states the phase transition and points at Cohen, Erez,
Ben-Avraham and Havlin's *Resilience of the internet to random breakdowns* (footnote 6, p. 318)
rather than writing the criterion down. :func:`molloy_reed` implements that citation's closed
form, kappa = <k^2>/<k> and f_c = 1 - 1/(kappa - 1), and says so where it is printed. Note the
name collision with the book's own vocabulary: §18.1 (p. 258) calls the *configuration-model
algorithm* "the Molloy-Reed approach", which is a different object from the criterion named
after the same two authors, and :func:`graphrag.sna.generators.configuration_model` is the one
the book means there.

*The area under the curve.* One number per strategy, so that a table of curves can be sorted:
the trapezoidal area under the giant-component curve over the removal fractions. It is not in
the chapter; it is Schneider et al.'s robustness index in all but the normalisation, and its
ceiling is 0.5 rather than 1.0 because a share of the *original* n cannot exceed 1 - f.

*The collapse fraction.* The chapter's "critical value of |R|" read off a simulated curve: the
first fraction at which the giant component no longer holds half of the nodes still standing.
Measured against the survivors and not against the original n, because the second denominator
makes any curve cross one half at f = 0.5 by arithmetic rather than by fragmentation.

*Capacity.* Figure 22.8's nodes carry capacities given by the figure. A corpus network carries
no such number, so :func:`cascade` uses the convention of the cascading-failure literature --
capacity = (1 + tolerance) x the load the node started with -- which makes ``tolerance`` the one
free parameter and 0 the fragile end of it.

*Coupling by degree.* §22.4 closes on what happens when the coupling is not random: *"what if
hubs in one layer tend to connect to hubs in the other? If such correlations were perfect, we'd
obtain again the robustness of power law networks to random failures"* (p. 324). That perfect
correlation is ``coupling="degree"``, so the sentence can be checked rather than quoted.

**Named in the chapter and deliberately not built.**

*Edge failures* (p. 318). The book raises them and drops them in the same breath: *"the
underlying math is rather similar and the functions describing the failures are not so different
than the ones I've been showing you so far. For this reason we keep looking at node failures."*
So do we.

*The alpha dependence* (figures 22.4, 22.7, 22.12). Every one of those figures is a family of
curves indexed by the exponent of a power-law degree distribution, and drawing it needs a
generator that takes alpha as an argument. :mod:`graphrag.sna.generators` has preferential
attachment, whose exponent is fixed at 3, and the configuration model, which takes a degree
sequence; the comparison is therefore available to a caller who builds the sequence, and is not
wrapped here because the chapter's claim is about a model family and not about any network this
package holds.

*The cascade size distribution* (p. 322): *"for alpha < 3 then the cascade size will grow with
exponent alpha/(alpha - 1)"*. That is a claim about the distribution of cascade sizes over an
ensemble of runs on a model network, not a measurement of one network, and it is chapter 9's
machinery (:mod:`graphrag.sna.degree`) that would test it.

*The threshold form of failure propagation* (p. 320): *"having a node V failing if a fraction
f_v of its neighbors are failing"*. That is Granovetter's threshold model of §21.1, and it
belongs to the spreading chapter rather than to a second copy here.

**What none of this is.** The chapter's own warning, in its introduction, is the one to keep
beside any number this module prints: *"real failures in power lines are neither [linear nor
local], due to the underlying laws governing electrical flows. You can use these methods in
other scenarios, but only if you're sure that the assumptions made here are respected in the
phenomenon you're studying"* (p. 315). A corpus network is not an infrastructure. Nothing flows
through it, nothing fails in it, and a node cannot be attacked: removing a speaker from a
co-appearance network is removing a *record*, not a person. What these curves measure is how
much of the co-occurrence structure survives losing its most central records -- a statement
about how concentrated the evidence is, which is a real question about a corpus, and not a
statement about anything failing.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import networkx as nx

from graphrag.sna.measures import centralities_for, centrality, undirected_view

__all__ = [
    "CASCADE_TOLERANCES",
    "COUPLINGS",
    "DEFAULT_RUNS",
    "DEFAULT_STEPS",
    "LOADS",
    "REPORT_FRACTIONS",
    "TARGETED_STRATEGIES",
    "AttackRow",
    "AttackSummary",
    "Cascade",
    "Interdependent",
    "InterdependentCurve",
    "InterdependentPoint",
    "MolloyReed",
    "RemovalCurve",
    "RemovalPoint",
    "RobustnessReport",
    "attack_summary",
    "cascade",
    "cascade_profile",
    "couple",
    "heaviest_node",
    "interdependent",
    "interdependent_curve",
    "molloy_reed",
    "removal_curve",
    "render_robustness",
    "robustness_payload",
    "robustness_report",
    "strategies_for",
]

Node = Any

#: The attacks :func:`attack_summary` runs beside the random null unless told otherwise. Degree
#: is §22.2's own attack; the other three ask the same question with a different definition of
#: "the node worth taking down", and the whole point of printing them together is that they
#: disagree.
TARGETED_STRATEGIES: tuple[str, ...] = ("degree", "betweenness", "pagerank", "coreness")

#: How many removal fractions a curve is measured at, from 0 to 1 inclusive. Twenty steps is a
#: 5% resolution, which is the grain at which a collapse fraction is reported.
DEFAULT_STEPS = 20

#: How many independent removal orders the random curve averages over. The book's own exercise
#: (§22.6.1) says *"Repeat 10 times and plot the average result"*; twenty is that, rounded up,
#: and the quantile band across the runs is what says whether the mean means anything.
DEFAULT_RUNS = 20

#: The removal fractions :func:`attack_summary` tabulates, chosen small: §22.2's damage is done
#: early, and a table at 5% and 10% is where a targeted curve separates from the random one.
REPORT_FRACTIONS: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.5)

#: The tolerances the cascade section sweeps. 0.0 is a network with no slack at all, where any
#: redistributed load is fatal; 1.0 is a network built with twice the capacity it uses, which is
#: the *"some slack in mind"* of p. 320.
CASCADE_TOLERANCES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)

#: What a node's load can be. ``betweenness`` is the load of a network that carries something
#: between pairs -- the traffic reading §22.3 opens with -- and ``degree`` is the load of a
#: network whose demand sits on its ties. Neither is measured; both are stated.
LOADS: tuple[str, ...] = ("betweenness", "degree")

#: How the two layers of §22.4 are matched one to one. ``same-id`` couples the nodes the two
#: networks share by name, which is what makes two of one persona's networks interdependent at
#: all; ``random`` is the chapter's own assumption (p. 324, *"hubs in one layer will pick a
#: random node to couple to in the other"*); ``degree`` is the perfect degree correlation the
#: same page says would restore the robustness.
COUPLINGS: tuple[str, ...] = ("same-id", "random", "degree")

#: Below this share of the survivors, the largest component is no longer holding the network
#: together and the curve is past the chapter's "critical value of |R|" (p. 316).
COLLAPSE_SHARE = 0.5


def strategies_for(graph: nx.Graph) -> tuple[str, ...]:
    """The removal orders this graph supports: the random null plus every centrality it has.

    Read off the undirected view, because that is where every curve here is measured: a
    directed network's in- and out-degree rankings are not defined on the flattened graph whose
    components are being counted, and an attack ranking that cannot be recomputed on what is
    left is not an attack ranking (§22.2).
    """
    return ("random", *centralities_for(undirected_view(graph)[0]))


# --------------------------------------------------------------- §22.1 the critical fraction


@dataclass(frozen=True)
class MolloyReed:
    """Where the giant component disappears, from the degree distribution alone (§22.1).

    ``kappa`` is <k^2>/<k>. The criterion is ``kappa > 2``: below it the network has no giant
    component to lose, and ``critical_fraction`` is then ``None`` rather than a negative number.
    ``critical_fraction`` is the share of nodes that has to fail at random before the giant
    component goes, ``1 - 1/(kappa - 1)``; it is a statement about a network drawn at random
    with this degree distribution, not about this particular wiring, so a real network's
    simulated collapse sits near it rather than on it.
    """

    nodes: int
    edges: int
    mean_degree: float
    mean_square_degree: float
    kappa: float
    critical_fraction: float | None
    has_giant_criterion: bool
    reading: str


def molloy_reed(graph: nx.Graph) -> MolloyReed:
    """kappa = <k^2>/<k> and the critical fraction f_c = 1 - 1/(kappa - 1) (§22.1).

    §22.1 states the phase transition -- *"we reach a critical value of |R| beyond which the GCC
    disappears"* (p. 316) -- and hands the closed form to its footnote 6 (p. 318), Cohen, Erez,
    Ben-Avraham and Havlin 2000. This is that form. A network keeps a giant component while the
    average excess degree of a node reached along an edge is at least one, which is kappa > 2;
    remove a share f of the nodes at random and kappa falls to ``1 + (1 - f)(kappa - 1)``, so
    the network crosses the criterion at f_c = 1 - 1/(kappa - 1).

    Why it says something about heavy tails: kappa carries the *second* moment, so a handful of
    enormous hubs pushes it up without moving <k> much, and f_c goes to 1. That is §22.1's
    robustness of a skewed network to random failure, in one number.

    Undefined, and returned as ``None``, when the network has no edges at all (there is no <k>
    to divide by) or when kappa <= 2, where the network is already below the criterion and
    "the fraction that breaks it" is not a fraction. Directed networks are read on the flattened
    undirected view, because the criterion is defined on undirected degrees.
    """
    flat, _ = undirected_view(graph)
    degrees = [float(d) for _, d in flat.degree()]
    n = len(degrees)
    mean = sum(degrees) / n if n else 0.0
    mean_square = sum(d * d for d in degrees) / n if n else 0.0
    kappa = mean_square / mean if mean > 0 else 0.0
    has_giant = kappa > 2.0
    critical = 1.0 - 1.0 / (kappa - 1.0) if kappa > 2.0 else None
    return MolloyReed(
        nodes=n,
        edges=flat.number_of_edges(),
        mean_degree=mean,
        mean_square_degree=mean_square,
        kappa=kappa,
        critical_fraction=critical,
        has_giant_criterion=has_giant,
        reading=_molloy_reading(mean, kappa, critical),
    )


def _molloy_reading(mean: float, kappa: float, critical: float | None) -> str:
    """The criterion in words, which is the half of it a reader needs (§22.1)."""
    if mean <= 0:
        return (
            "No edges, so there is no degree distribution to read: kappa is undefined and the "
            "network has nothing to lose (§22.1)."
        )
    if critical is None:
        return (
            f"kappa = {kappa:.4f} is at or below 2, so this network is already under the "
            "criterion for holding a giant component at all: there is no critical fraction to "
            "report, because random failure does not have to remove anything to break what is "
            "already broken (§22.1, after Cohen et al. 2000)."
        )
    return (
        f"kappa = {kappa:.4f} (> 2), so random failure has to take out {critical:.1%} of the "
        "nodes before the giant component goes. The second moment is what puts it there: a "
        "heavier degree tail raises kappa without raising the mean degree much, which is why "
        "§22.1 finds skewed networks robust to accidents -- 'it is extremely unlikely to pick "
        "the hub' (p. 317). It is the threshold for a *random* network with this degree "
        "distribution, so read it beside the simulated curve rather than instead of it."
    )


# --------------------------------------------------------------- §22.1-22.2 the removal curves


@dataclass(frozen=True)
class RemovalPoint:
    """One point of a removal curve: what was left after removing this share of the nodes.

    ``giant`` is the chapter's y axis, *"% nodes in LCC"* (figure 22.2), as a share of the
    **original** node count, so it cannot exceed ``1 - fraction``. ``giant_of_survivors`` is the
    same component as a share of what is still standing, which is the one that says whether the
    survivors are still one network. ``mean_other_size`` is p. 315's second signal -- the mean
    size of the components that are not the giant one, isolated nodes counted as components of
    size one -- and it rises as the network shatters and falls again once there is nothing left
    to shatter.

    ``low`` and ``high`` bracket ``giant`` over the runs (the 10th and 90th percentile); on a
    single-run curve they equal it.
    """

    fraction: float
    removed: int
    giant: float
    giant_of_survivors: float
    mean_other_size: float
    components: float
    low: float
    high: float


@dataclass(frozen=True)
class RemovalCurve:
    """A whole removal experiment: the strategy, how it was run, and what it did (§22.1-22.2)."""

    strategy: str
    section: str
    recompute: bool
    runs: int
    steps: int
    seed: int | None
    nodes: int
    edges: int
    points: tuple[RemovalPoint, ...]
    #: Trapezoidal area under ``giant`` over the removal fractions. Ceiling 0.5; see the module
    #: docstring. Higher is more robust, and it is the only single number here that orders
    #: strategies.
    area: float
    #: The first fraction at which the giant component holds less than half of the *survivors*,
    #: or ``None`` when it never does.
    collapse_fraction: float | None
    #: The fraction at which ``mean_other_size`` peaks: where the network is coming apart
    #: fastest (p. 315). ``None`` when nothing ever fragments.
    shatter_fraction: float | None
    note: str

    def at(self, fraction: float) -> RemovalPoint:
        """The measured point nearest this removal fraction."""
        if not self.points:
            msg = "this curve has no points"
            raise ValueError(msg)
        return min(self.points, key=lambda p: (abs(p.fraction - fraction), p.fraction))


def removal_curve(
    graph: nx.Graph,
    strategy: str = "random",
    *,
    recompute: bool = False,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
) -> RemovalCurve:
    """Remove nodes and watch the largest component: §22.1 at random, §22.2 by ranking.

    ``strategy="random"`` is §22.1's accident -- *"nodes can spontaneously break for
    uncorrelated and not deliberate reasons"* (p. 315) -- run ``runs`` times with independent
    orders and averaged, which is what the chapter's own exercise asks for (*"Repeat 10 times
    and plot the average result"*, §22.6.1). Any other strategy is §22.2's attacker, removing
    nodes from the highest score down; ``degree`` is the chapter's own (*"This translates into
    prioritizing attacks to the nodes with the highest degree"*, p. 318) and the rest are the
    centralities of chapter 14 asked the same question.

    ``recompute`` decides whether the attacker is told what their last blow did. With it off the
    ranking is computed once, on the whole network, and followed to the end. With it on the
    ranking is computed again at the start of every step, on what is still standing, which is
    the stronger attack and usually the shorter path to collapse -- a node that was second most
    central often is not, once the first is gone. It is recomputed once per *step* and not once
    per node: ``steps`` is the resolution of the curve anyway, and ``steps=n`` makes it the
    node-by-node attack exactly. The report prints which of the two was done, because the two
    give different curves and a curve without that label cannot be compared with anything.

    Ties are broken at random, seeded, rather than by node id: ``coreness`` puts most of a
    network on the same integer and ``degree`` ties heavily too, so a deterministic tie-break
    would report one arbitrary ordering of a set of equal candidates as though it were the
    attack. When the scores have no ties at all there is nothing to average and a single run is
    kept, which the note says.

    Every measurement is on the undirected view: components here are the connected ones, or the
    weakly connected ones of a directed network, which is the chapter's *"can still reach each
    other"* reading either way.

    Raises ``ValueError`` on an unknown strategy, on ``steps < 1`` or ``runs < 1``.
    """
    allowed = strategies_for(graph)
    if strategy not in allowed:
        msg = f"strategy must be one of {', '.join(allowed)}, got {strategy!r}"
        raise ValueError(msg)
    if steps < 1:
        msg = f"steps must be at least 1, got {steps}"
        raise ValueError(msg)
    if runs < 1:
        msg = f"runs must be at least 1, got {runs}"
        raise ValueError(msg)

    flat, flattened = undirected_view(graph)
    n = flat.number_of_nodes()
    if n == 0:
        msg = "an empty network has nothing to remove"
        raise ValueError(msg)

    scores = None if strategy == "random" else centrality(flat, strategy)
    tied = scores is not None and len(set(scores.values())) < n
    effective_runs = runs if strategy == "random" or tied else 1
    targets = [round(n * step / steps) for step in range(steps + 1)]

    measured: list[list[tuple[float, float, float, float]]] = []
    for run in range(effective_runs):
        rng = random.Random(None if seed is None else seed + run)  # noqa: S311 -- reproducible
        measured.append(_one_removal(flat, strategy, scores, targets, recompute, rng))

    points = tuple(
        _point(targets[index] / n, targets[index], [run[index] for run in measured])
        for index in range(len(targets))
    )
    return RemovalCurve(
        strategy=strategy,
        section="§22.1" if strategy == "random" else "§22.2",
        recompute=recompute,
        runs=effective_runs,
        steps=steps,
        seed=seed,
        nodes=n,
        edges=flat.number_of_edges(),
        points=points,
        area=_area(points),
        collapse_fraction=_collapse_fraction(points),
        shatter_fraction=_shatter_fraction(points),
        note=_curve_note(strategy, recompute, effective_runs, runs, tied, flattened),
    )


def _one_removal(
    graph: nx.Graph,
    strategy: str,
    scores: Mapping[Node, float] | None,
    targets: Sequence[int],
    recompute: bool,
    rng: random.Random,
) -> list[tuple[float, float, float, float]]:
    """One removal order run to the end, measured at each target count.

    Returns, per target: the giant share of the original n, the giant share of the survivors,
    the mean size of the other components, and how many components there were.
    """
    remaining = graph.copy()
    total = graph.number_of_nodes()
    order = [] if recompute else _order(graph, strategy, scores, rng)
    removed = 0
    readings: list[tuple[float, float, float, float]] = []
    for target in targets:
        while removed < target and remaining.number_of_nodes() > 0:
            if recompute:
                order = _order(
                    remaining,
                    strategy,
                    None if strategy == "random" else centrality(remaining, strategy),
                    rng,
                )
                take = min(target - removed, len(order))
                remaining.remove_nodes_from(order[:take])
                removed += take
            else:
                remaining.remove_node(order[removed])
                removed += 1
        readings.append(_measure(remaining, total))
    return readings


def _order(
    graph: nx.Graph,
    strategy: str,
    scores: Mapping[Node, float] | None,
    rng: random.Random,
) -> list[Node]:
    """The nodes in the order this strategy removes them, ties broken by the seeded rng."""
    nodes = list(graph)
    rng.shuffle(nodes)
    if strategy == "random" or scores is None:
        return nodes
    return sorted(nodes, key=lambda node: -float(scores.get(node, 0.0)))


def _measure(graph: nx.Graph, total: int) -> tuple[float, float, float, float]:
    """The four readings of one surviving network (§22.1, p. 315)."""
    sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    if not sizes:
        return (0.0, 0.0, 0.0, 0.0)
    survivors = sum(sizes)
    others = sizes[1:]
    mean_other = sum(others) / len(others) if others else 0.0
    return (sizes[0] / total, sizes[0] / survivors, mean_other, float(len(sizes)))


def _point(
    fraction: float, removed: int, readings: Sequence[tuple[float, float, float, float]]
) -> RemovalPoint:
    """One curve point, averaged over the runs, with a 10-90 band on the giant share."""
    giants = sorted(reading[0] for reading in readings)
    return RemovalPoint(
        fraction=fraction,
        removed=removed,
        giant=sum(giants) / len(giants),
        giant_of_survivors=sum(r[1] for r in readings) / len(readings),
        mean_other_size=sum(r[2] for r in readings) / len(readings),
        components=sum(r[3] for r in readings) / len(readings),
        low=_quantile(giants, 0.1),
        high=_quantile(giants, 0.9),
    )


def _quantile(values: Sequence[float], q: float) -> float:
    """The q-quantile of already sorted values, by linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _area(points: Sequence[RemovalPoint]) -> float:
    """Trapezoidal area under the giant-component curve; 0.5 is the ceiling. Not in the book."""
    total = 0.0
    for first, second in pairwise(points):
        total += (second.fraction - first.fraction) * (first.giant + second.giant) / 2
    return total


def _collapse_fraction(points: Sequence[RemovalPoint]) -> float | None:
    """The first fraction where the giant component holds under half of the survivors."""
    for point in points:
        if point.fraction > 0 and point.giant_of_survivors < COLLAPSE_SHARE:
            return point.fraction
    return None


def _shatter_fraction(points: Sequence[RemovalPoint]) -> float | None:
    """Where the mean non-giant component is largest: the network coming apart (p. 315)."""
    candidates = [point for point in points if point.mean_other_size > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda point: (point.mean_other_size, -point.fraction)).fraction


def _curve_note(
    strategy: str, recompute: bool, effective: int, asked: int, tied: bool, flattened: str
) -> str:
    """What was actually done, in the words the report has to print next to the curve."""
    parts: list[str] = []
    if strategy == "random":
        parts.append(
            f"Uniformly random removal, {effective} independent orders averaged, with the "
            "10th-90th percentile band across them (§22.1)."
        )
    else:
        parts.append(
            f"Removal by {strategy}, highest first, ranking "
            + (
                "recomputed on the survivors at the start of every step"
                if recompute
                else "computed once on the whole network and followed to the end"
            )
            + " (§22.2)."
        )
        if tied:
            parts.append(
                f"The scores tie, so the order within a tie is random and {effective} of them "
                "are averaged."
            )
        elif effective < asked:
            parts.append(
                f"The scores do not tie, so the order is deterministic and one run is the same "
                f"as the {asked} that were asked for."
            )
    if flattened:
        parts.append(f"Measured {flattened}")
    return " ".join(parts)


# --------------------------------------------------------------- §22.2 the comparison


@dataclass(frozen=True)
class AttackRow:
    """One strategy in the attack table, beside what random removal did at the same fractions."""

    strategy: str
    area: float
    collapse_fraction: float | None
    #: Giant-component share (of the original n) at each of :data:`REPORT_FRACTIONS`.
    giant_at: tuple[tuple[float, float], ...]
    #: How much lower the giant component is than under random removal, at the same fractions.
    damage_over_random: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AttackSummary:
    """Random versus every targeted strategy: figures 22.5 and 22.6 as a table (§22.2)."""

    random_curve: RemovalCurve
    curves: tuple[RemovalCurve, ...]
    rows: tuple[AttackRow, ...]
    fractions: tuple[float, ...]
    recompute: bool
    #: The strategy with the smallest area under the curve: the cheapest way to break this
    #: network out of the ones that were tried.
    worst: str
    verdict: str


def attack_summary(
    graph: nx.Graph,
    *,
    strategies: Sequence[str] = TARGETED_STRATEGIES,
    recompute: bool = False,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
    fractions: Sequence[float] = REPORT_FRACTIONS,
) -> AttackSummary:
    """Random removal against each targeted one, at a few fractions and as one area each (§22.2).

    This is the comparison the chapter draws twice, once for a G(n,p) network (figure 22.5,
    where *"targeted attacks don't change the scenario much"*) and once for a degree-skewed one
    (figure 22.6, where *"Removing even a single node brings down the GCC size by almost 20%"*).
    Random removal is the null the other rows are read against -- not a statistical null, a
    *behavioural* one: the damage an accident does, so that the damage an attacker does can be
    quoted as a difference rather than as a number.

    The strategies that are not ``degree`` are this package's addition: §22.2 argues only about
    degree, on the ground that *"Taking down the node with most connections is guaranteed to
    cause the maximum possible amount of damage"*, which is true of the *effort* and not of the
    damage. ``betweenness`` frequently beats it -- the node that holds two halves together can
    have an ordinary degree -- and printing them together is what shows that.

    Raises ``ValueError`` if a strategy is not defined on this graph, or if ``strategies`` is
    empty.
    """
    if not strategies:
        msg = "attack_summary needs at least one targeted strategy"
        raise ValueError(msg)
    baseline = removal_curve(graph, "random", steps=steps, runs=runs, seed=seed)
    curves = tuple(
        removal_curve(graph, strategy, recompute=recompute, steps=steps, runs=runs, seed=seed)
        for strategy in strategies
    )
    rows = tuple(_attack_row(curve, baseline, fractions) for curve in curves)
    worst = min([baseline, *curves], key=lambda curve: curve.area).strategy
    return AttackSummary(
        random_curve=baseline,
        curves=curves,
        rows=rows,
        fractions=tuple(fractions),
        recompute=recompute,
        worst=worst,
        verdict=_attack_verdict(baseline, curves, worst),
    )


def _attack_row(
    curve: RemovalCurve, baseline: RemovalCurve, fractions: Sequence[float]
) -> AttackRow:
    """One row of the attack table: the curve at a few fractions, and the damage over random."""
    at = tuple((fraction, curve.at(fraction).giant) for fraction in fractions)
    over = tuple(
        (fraction, baseline.at(fraction).giant - curve.at(fraction).giant) for fraction in fractions
    )
    return AttackRow(
        strategy=curve.strategy,
        area=curve.area,
        collapse_fraction=curve.collapse_fraction,
        giant_at=at,
        damage_over_random=over,
    )


def _attack_verdict(baseline: RemovalCurve, curves: Sequence[RemovalCurve], worst: str) -> str:
    """§22.2's reading of the gap between the random curve and the worst targeted one."""
    if worst == "random":
        return (
            "No targeted strategy did more damage than random removal, which happens when the "
            "degrees are all alike -- §22.2's G(n,p) case, where 'hubs are less common and "
            "their degree isn't much different from the average degree of all other nodes' (p. "
            "318). A network with nothing to target is a network with nothing to protect."
        )
    worst_curve = next(curve for curve in curves if curve.strategy == worst)
    gap = baseline.area - worst_curve.area
    first = worst_curve.at(REPORT_FRACTIONS[0])
    random_first = baseline.at(REPORT_FRACTIONS[0])
    collapse = (
        f"it is past the point where the survivors are no longer one network at "
        f"{worst_curve.collapse_fraction:.0%} removed, against "
        + (
            f"{baseline.collapse_fraction:.0%} for random removal"
            if baseline.collapse_fraction is not None
            else "never, for random removal"
        )
        if worst_curve.collapse_fraction is not None
        else "it never breaks the survivors into pieces at this resolution"
    )
    return (
        f"The cheapest attack here is by {worst}: it leaves "
        f"{first.giant:.1%} of the nodes in the giant component after removing "
        f"{REPORT_FRACTIONS[0]:.0%} of them, where random removal leaves "
        f"{random_first.giant:.1%}, and over the whole curve it costs "
        f"{gap:.4f} of area against the accident. {collapse[0].upper()}{collapse[1:]}. "
        "§22.2's reading of that gap is the one worth keeping: robustness to random failure and "
        "fragility to attack are the same property looked at twice -- a network that survives "
        "accidents because its structure rests on a few nodes is a network an attacker only has "
        "to find those nodes in."
    )


# --------------------------------------------------------------- §22.3 chain effects


@dataclass(frozen=True)
class Cascade:
    """What one initial failure did once the load it was carrying had to go somewhere (§22.3).

    ``surviving`` is the share of nodes still standing, ``giant_share`` what is left of the
    largest component. ``branching`` is the mean number of further failures each failure caused,
    the average out-degree of the cascade tree of figure 22.9: the book's critical value is 1,
    *"If, on average, the failure of a node generates another node failure -- or more -- the
    cascade will propagate indefinitely"* (p. 321), and a finite realised cascade is therefore
    always slightly below 1 when it swallowed the network and near 0 when it died out.
    """

    tolerance: float
    load: str
    seeds: tuple[str, ...]
    nodes: int
    failed: int
    surviving: float
    giant_share: float
    rounds: int
    branching: float
    #: Load that went nowhere, because the node holding it failed with no surviving neighbour.
    lost_load: float
    total_load: float


def cascade(
    graph: nx.Graph,
    *,
    tolerance: float = 0.1,
    seeds: int | Sequence[Node] = 1,
    load: str = "degree",
    seed: int | None = None,
) -> Cascade:
    """The load-capacity cascade of §22.3: fail one node, let its load find somewhere else.

    The model is the chapter's own, the Failure Propagation model it cites at footnote 12 (p.
    320). *"All nodes start in the state S. They are characterized by a current load and by a
    total capacity"*; at t = 1 the seeds shut down, and *"we have to redistribute the load ...
    through alternative routes: the neighbors of that node. However, that means that their load
    will increase. If the new load exceeds the capacity of the node, also this node shuts down
    due to congestion. So its load has also to be redistributed to its neighbors and so on and
    so forth"* (p. 320). §22.6's fourth exercise fixes the two details the prose leaves open:
    a failing node *"equally distribute[s] all of its current load to its neighbors"*, and the
    comparison is ``load > capacity`` (figure 22.8's caption: *"If load > capacity, the node
    will fail at the next time step"*). Failures are resolved in rounds, as the figure's time
    steps are; a node that fails with no surviving neighbour left drops its load, which is what
    happens to the 37 units figure 22.8 has accumulated in one node at t = 6.

    ``load`` is where the initial load comes from: ``degree`` or ``betweenness`` (§14.4), the
    second being the traffic reading the section opens with. ``tolerance`` is the slack: a
    node's capacity is ``(1 + tolerance)`` times the load it started with, which is the
    convention of the cascading-failure literature rather than the book's -- figure 22.8 reads
    its capacities off the figure, and a corpus network has no such number. At ``tolerance=0``
    any redistributed load at all is fatal and the cascade takes everything it can reach; at
    ``tolerance=1`` every node was built with twice the capacity it uses.

    ``seeds`` is either the nodes to shut down or a count, in which case that many are drawn
    uniformly at random with ``seed`` -- §22.1's accident, lit at one point. Pass the nodes
    explicitly for §22.2's attacker.

    Raises ``ValueError`` on an unknown load, a negative tolerance, a seed count larger than the
    network, or a named seed that is not in it.
    """
    if load not in LOADS:
        msg = f"load must be one of {', '.join(LOADS)}, got {load!r}"
        raise ValueError(msg)
    if tolerance < 0:
        msg = f"tolerance must be at least 0, got {tolerance}"
        raise ValueError(msg)
    flat, _ = undirected_view(graph)
    total = flat.number_of_nodes()
    if total == 0:
        msg = "an empty network has nothing to fail"
        raise ValueError(msg)

    if isinstance(seeds, int):
        if not 0 < seeds <= total:
            msg = f"seeds must be between 1 and {total}, got {seeds}"
            raise ValueError(msg)
        rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
        chosen: list[Node] = rng.sample(sorted(flat, key=str), k=seeds)
    else:
        chosen = list(seeds)
        missing = [node for node in chosen if node not in flat]
        if missing:
            msg = f"these seeds are not in the network: {', '.join(str(m) for m in missing)}"
            raise ValueError(msg)
        if not chosen:
            msg = "cascade needs at least one seed"
            raise ValueError(msg)

    loads = _initial_load(flat, load)
    capacity = {node: (1.0 + tolerance) * value for node, value in loads.items()}
    failed = set(chosen)
    wave = list(chosen)
    rounds = 0
    lost = 0.0
    while wave:
        rounds += 1
        incoming: dict[Node, float] = {}
        for node in wave:
            alive = [other for other in flat.neighbors(node) if other not in failed]
            carried = loads[node]
            loads[node] = 0.0
            if carried <= 0:
                continue
            if not alive:
                lost += carried
                continue
            share = carried / len(alive)
            for other in alive:
                incoming[other] = incoming.get(other, 0.0) + share
        for node, extra in incoming.items():
            loads[node] += extra
        wave = sorted(
            (node for node in incoming if node not in failed and loads[node] > capacity[node]),
            key=str,
        )
        failed.update(wave)

    standing = flat.subgraph([node for node in flat if node not in failed])
    sizes = [len(component) for component in nx.connected_components(standing)]
    return Cascade(
        tolerance=tolerance,
        load=load,
        seeds=tuple(sorted(str(node) for node in chosen)),
        nodes=total,
        failed=len(failed),
        surviving=(total - len(failed)) / total,
        giant_share=max(sizes, default=0) / total,
        rounds=rounds,
        branching=(len(failed) - len(chosen)) / len(failed) if failed else 0.0,
        lost_load=lost,
        total_load=sum(capacity[node] / (1.0 + tolerance) for node in flat),
    )


def _initial_load(graph: nx.Graph, load: str) -> dict[Node, float]:
    """Every node's load at t = 0, from the measure named (§22.3, §14.4)."""
    if load == "degree":
        return {node: float(degree) for node, degree in graph.degree()}
    scores = centrality(graph, "betweenness")
    return {node: float(scores.get(node, 0.0)) for node in graph}


def cascade_profile(
    graph: nx.Graph,
    *,
    tolerances: Sequence[float] = CASCADE_TOLERANCES,
    seeds: int | Sequence[Node] = 1,
    load: str = "degree",
    seed: int | None = None,
) -> tuple[Cascade, ...]:
    """The same cascade at each tolerance: how much slack this network needs to survive itself.

    One run is a number; the sweep is the finding, because §22.3's whole claim is that the
    outcome is a step rather than a slope -- below some slack the snowball takes everything,
    above it nothing moves. Same seeds at every tolerance, so the rows differ only by the
    capacity.
    """
    return tuple(
        cascade(graph, tolerance=tolerance, seeds=seeds, load=load, seed=seed)
        for tolerance in tolerances
    )


def heaviest_node(graph: nx.Graph, load: str = "degree") -> Node:
    """The node an attacker would shut down first under this load measure (§22.2 meets §22.3)."""
    flat, _ = undirected_view(graph)
    if flat.number_of_nodes() == 0:
        msg = "an empty network has no heaviest node"
        raise ValueError(msg)
    loads = _initial_load(flat, load)
    return max(sorted(flat, key=str), key=lambda node: loads[node])


# --------------------------------------------------------------- §22.4 interdependent networks


@dataclass(frozen=True)
class Interdependent:
    """The mutual giant component of two coupled networks after a shock (§22.4).

    ``mutual_share`` is the share of the coupled nodes that end up in a component that survives
    in *both* layers at once, after the failures have finished bouncing between them.
    ``alone_a`` and ``alone_b`` are what each layer's giant component would have been had the
    same nodes failed and nothing propagated, which is the comparison §22.4 asks for: *"if you
    were to calculate the critical |R| value for each layer separately you would obtain a result
    much higher than the one for the interdependent network as a whole"* (p. 323).
    """

    coupling: str
    fraction_removed: float
    coupled: int
    #: Nodes of each layer that no coupling could match, and are therefore outside the analysis.
    uncoupled_a: int
    uncoupled_b: int
    removed: int
    mutual_share: float
    alone_a: float
    alone_b: float
    iterations: int


def couple(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    coupling: str = "same-id",
    *,
    seed: int | None = None,
) -> dict[Node, Node]:
    """The one-to-one dependency between two layers' nodes (§22.4).

    *"A power station needs the coupled computer to work and vice versa"* (p. 322), which is a
    bijection: every node has exactly one counterpart. ``same-id`` matches the nodes the two
    networks share by name, which is what makes two of one persona's networks interdependent --
    the same speaker in the co-appearance network and in the two-mode one is one person.
    ``random`` is the chapter's own assumption for a model (p. 324) and ``degree`` its perfect
    positive correlation, hubs to hubs.

    Nodes with no counterpart are left out of the mapping rather than coupled to nothing: they
    are outside §22.4's model, and the report prints how many there were, because a coupling
    that matched a tenth of the network is a statement about that tenth.
    """
    if coupling not in COUPLINGS:
        msg = f"coupling must be one of {', '.join(COUPLINGS)}, got {coupling!r}"
        raise ValueError(msg)
    nodes_a = sorted(graph_a, key=str)
    nodes_b = sorted(graph_b, key=str)
    if coupling == "same-id":
        shared = [node for node in nodes_a if node in graph_b]
        return {node: node for node in shared}
    size = min(len(nodes_a), len(nodes_b))
    if coupling == "degree":
        left = sorted(nodes_a, key=lambda node: (-graph_a.degree(node), str(node)))[:size]
        right = sorted(nodes_b, key=lambda node: (-graph_b.degree(node), str(node)))[:size]
        return dict(zip(left, right, strict=True))
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    left = nodes_a[:]
    right = nodes_b[:]
    rng.shuffle(left)
    rng.shuffle(right)
    return dict(zip(left[:size], right[:size], strict=True))


def interdependent(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    coupling: str = "same-id",
    *,
    fraction_removed: float,
    seed: int | None = None,
    pairs: Mapping[Node, Node] | None = None,
) -> Interdependent:
    """Fail a fraction of the coupled nodes and let the two layers take each other down (§22.4).

    The process is the one figure 22.11 draws. Remove the initial nodes from both layers at
    once, since a node needs its counterpart to work. Then: whatever is not in the largest
    component of layer A stops working, because *"A power station needs to know the statuses of
    its neighboring stations: if they are on a different computer component it cannot know it,
    so their links deactivate"* (p. 322); the counterparts of those nodes stop working in layer
    B; whatever that leaves outside B's largest component stops working too; and back to A.
    Iterate to the fixed point -- *"These failures propagate in a chain reaction until we end up
    in a situation where practically every node in both layers is isolated"* (p. 322-323) -- and
    what is left is the mutual giant component.

    The number to read beside it is ``alone_a`` / ``alone_b``, the same shock with no coupling.
    §22.4's finding is the distance between them, and the shape of that distance over
    ``fraction_removed`` is the finding rather than any single value: the mutual component does
    not decay, it holds and then goes, which is why :func:`interdependent_curve` exists.

    ``pairs`` overrides the coupling with an explicit bijection, so a caller who already knows
    which node depends on which can say so.

    Raises ``ValueError`` if the fraction is outside [0, 1] or the coupling matches nothing.
    """
    if not 0.0 <= fraction_removed <= 1.0:
        msg = f"fraction_removed must be between 0 and 1, got {fraction_removed}"
        raise ValueError(msg)
    flat_a, _ = undirected_view(graph_a)
    flat_b, _ = undirected_view(graph_b)
    mapping = dict(pairs) if pairs is not None else couple(flat_a, flat_b, coupling, seed=seed)
    if not mapping:
        msg = (
            f"the {coupling} coupling matched no node of the first network to a node of the "
            "second, so there is no interdependent network to analyse"
        )
        raise ValueError(msg)

    left = sorted(mapping, key=str)
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    order = left[:]
    rng.shuffle(order)
    knocked = round(len(order) * fraction_removed)
    alive = set(order[knocked:])

    iterations = 0
    while True:
        iterations += 1
        kept_a = _largest_component(flat_a.subgraph(alive))
        kept_b = _largest_component(flat_b.subgraph({mapping[node] for node in kept_a}))
        survivors = {node for node in kept_a if mapping[node] in kept_b}
        if survivors == alive:
            break
        alive = survivors
        if not alive:
            break

    coupled = len(mapping)
    return Interdependent(
        coupling=coupling,
        fraction_removed=fraction_removed,
        coupled=coupled,
        uncoupled_a=flat_a.number_of_nodes() - coupled,
        uncoupled_b=flat_b.number_of_nodes() - coupled,
        removed=knocked,
        mutual_share=len(alive) / coupled,
        alone_a=len(_largest_component(flat_a.subgraph(set(order[knocked:])))) / coupled,
        alone_b=len(_largest_component(flat_b.subgraph({mapping[n] for n in order[knocked:]})))
        / coupled,
        iterations=iterations,
    )


def _largest_component(graph: nx.Graph) -> set[Node]:
    """The biggest connected component's nodes, or an empty set when there is nothing left."""
    components = list(nx.connected_components(graph))
    return max(components, key=len) if components else set()


@dataclass(frozen=True)
class InterdependentPoint:
    """One point of the collapse curve: the mutual giant component against each layer alone.

    ``mutual``, ``alone_a`` and ``alone_b`` are shares of *all* the coupled nodes, which is the
    y axis of figure 22.12 and cannot exceed ``1 - fraction``. ``of_survivors`` divides the same
    counts by the nodes still standing instead, which is the pair the collapse fractions are
    read off: a share of the original count crosses one half at f = 0.5 by arithmetic, and the
    whole question here is which of the two structures crosses it *first*.
    """

    fraction: float
    mutual: float
    alone_a: float
    alone_b: float
    mutual_of_survivors: float
    alone_of_survivors: float
    iterations: float


@dataclass(frozen=True)
class InterdependentCurve:
    """The collapse of two coupled networks as the initial damage grows (§22.4)."""

    coupling: str
    coupled: int
    uncoupled_a: int
    uncoupled_b: int
    runs: int
    points: tuple[InterdependentPoint, ...]
    #: First fraction at which the mutual giant component holds under half of the coupled nodes
    #: *still standing*.
    mutual_collapse: float | None
    #: The same for the better of the two layers measured on its own, which §22.4 says is the
    #: later fraction of the two.
    alone_collapse: float | None
    #: The largest one-step drop in the mutual component: how abrupt the transition is.
    largest_drop: float
    verdict: str


def interdependent_curve(
    graph_a: nx.Graph,
    graph_b: nx.Graph,
    coupling: str = "same-id",
    *,
    steps: int = 10,
    runs: int = 5,
    seed: int | None = None,
) -> InterdependentCurve:
    """Sweep the initial damage and watch the mutual giant component go (§22.4).

    The sweep is the point. §22.4's transition is first-order -- the coupled system holds a
    respectable mutual component and then, over one step of the x axis, has none -- so a single
    ``fraction_removed`` reports a number from one side of a cliff without saying there is one.
    ``largest_drop`` measures the cliff, and the verdict names it.

    Each fraction is run ``runs`` times with different initial victims and averaged.
    """
    flat_a, _ = undirected_view(graph_a)
    flat_b, _ = undirected_view(graph_b)
    pairs = couple(flat_a, flat_b, coupling, seed=seed)
    fractions = [step / steps for step in range(steps + 1)]
    points: list[InterdependentPoint] = []
    coupled = uncoupled_a = uncoupled_b = 0
    for fraction in fractions:
        results = [
            interdependent(
                graph_a,
                graph_b,
                coupling,
                fraction_removed=fraction,
                seed=None if seed is None else seed + run,
                pairs=pairs,
            )
            for run in range(runs)
        ]
        coupled = results[0].coupled
        uncoupled_a, uncoupled_b = results[0].uncoupled_a, results[0].uncoupled_b
        mutual = sum(r.mutual_share for r in results) / len(results)
        alone_a = sum(r.alone_a for r in results) / len(results)
        alone_b = sum(r.alone_b for r in results) / len(results)
        standing = 1.0 - fraction
        points.append(
            InterdependentPoint(
                fraction=fraction,
                mutual=mutual,
                alone_a=alone_a,
                alone_b=alone_b,
                mutual_of_survivors=mutual / standing if standing > 0 else 0.0,
                alone_of_survivors=max(alone_a, alone_b) / standing if standing > 0 else 0.0,
                iterations=sum(r.iterations for r in results) / len(results),
            )
        )
    mutual_collapse = next(
        (
            point.fraction
            for point in points
            if point.fraction > 0 and point.mutual_of_survivors < COLLAPSE_SHARE
        ),
        None,
    )
    alone_collapse = next(
        (
            point.fraction
            for point in points
            if point.fraction > 0 and point.alone_of_survivors < COLLAPSE_SHARE
        ),
        None,
    )
    drop = max(
        (first.mutual - second.mutual for first, second in pairwise(points)),
        default=0.0,
    )
    return InterdependentCurve(
        coupling=coupling,
        coupled=coupled,
        uncoupled_a=uncoupled_a,
        uncoupled_b=uncoupled_b,
        runs=runs,
        points=tuple(points),
        mutual_collapse=mutual_collapse,
        alone_collapse=alone_collapse,
        largest_drop=drop,
        verdict=_interdependent_verdict(
            mutual_collapse,
            alone_collapse,
            drop,
            coupled_at_zero=points[0].mutual if points else 0.0,
        ),
    )


def _interdependent_verdict(
    mutual: float | None, alone: float | None, drop: float, *, coupled_at_zero: float = 1.0
) -> str:
    """§22.4's claim, checked against these two networks rather than quoted at them."""
    if coupled_at_zero < COLLAPSE_SHARE:
        return (
            f"With nothing removed at all the mutual giant component already holds only "
            f"{coupled_at_zero:.1%} of the coupled nodes, so there is no collapse here to "
            "report: this coupling joins two structures that do not share one component to "
            "begin with, and §22.4's model assumes they do. Read it as a statement about the "
            "coupling -- which pairs were matched -- and not about either network's robustness."
        )
    if mutual is None:
        return (
            "The mutual giant component never fell below half of the coupled nodes still "
            "standing: the coupling did not break this pair. §22.4's warning is about what "
            "*can* happen, and here it did not."
        )
    order = (
        f"{mutual:.0%} against {alone:.0%}"
        if alone is not None
        else f"{mutual:.0%}, where neither layer alone ever falls that far"
    )
    lead = (
        "The coupled pair breaks before either layer would on its own"
        if alone is None or mutual < alone
        else "The coupled pair breaks no earlier than the layers do on their own"
    )
    return (
        f"{lead}: {order} of the coupled nodes removed. That is §22.4's finding -- 'if you were "
        "to calculate the critical |R| value for each layer separately you would obtain a "
        "result much higher than the one for the interdependent network as a whole' (p. 323) -- "
        f"and the transition is abrupt: the largest single-step fall is {drop:.1%} of the "
        "coupled nodes. Read the curve, not the endpoint: a mutual component measured at one "
        "fraction is a number from one side of a cliff and says nothing about where the cliff "
        "is."
    )


# --------------------------------------------------------------- the report


@dataclass(frozen=True)
class RobustnessReport:
    """Everything chapter 22 has to say about one network, with the frame it was measured in."""

    frame: str
    network: str
    persona_id: str
    nodes: int
    edges: int
    directed: bool
    criterion: MolloyReed
    attacks: AttackSummary
    cascades: tuple[Cascade, ...]
    cascade_seeds: tuple[str, ...]
    coupled: InterdependentCurve | None
    coupled_network: str
    seed: int | None
    notes: tuple[str, ...]


def robustness_report(
    graph: nx.Graph,
    *,
    network: str = "",
    persona_id: str = "",
    strategies: Sequence[str] = TARGETED_STRATEGIES,
    recompute: bool = False,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
    cascades: bool = False,
    tolerances: Sequence[float] = CASCADE_TOLERANCES,
    load: str = "degree",
    other: nx.Graph | None = None,
    coupling: str = "same-id",
    coupled_network: str = "",
) -> RobustnessReport:
    """Chapter 22 over one network: criterion, curves, and optionally cascades and coupling.

    The cascade section is opt-in because a cascade needs a load, and a load on a corpus network
    is a modelling choice rather than a measurement -- nothing is carried between these nodes.
    The interdependent section is opt-in for the same reason and one more: it needs a second
    network and a coupling between the two, both of which are arguments and not facts.

    The cascade seeds are the heaviest node under the load measure, deterministically, so the
    tolerances sweep is comparable row to row: it is §22.2's attacker choosing where to start
    §22.3's chain, which is the worst case rather than the typical one.
    """
    flat, flattened = undirected_view(graph)
    summary = attack_summary(
        graph, strategies=strategies, recompute=recompute, steps=steps, runs=runs, seed=seed
    )
    seeds: tuple[str, ...] = ()
    profile: tuple[Cascade, ...] = ()
    if cascades and flat.number_of_nodes() > 0:
        worst = heaviest_node(flat, load)
        seeds = (str(worst),)
        profile = cascade_profile(flat, tolerances=tolerances, seeds=[worst], load=load, seed=seed)
    curve = (
        interdependent_curve(graph, other, coupling, runs=max(1, runs // 4), seed=seed)
        if other is not None
        else None
    )
    notes = [flattened] if flattened else []
    return RobustnessReport(
        frame=str(graph.graph.get("frame", "")),
        network=network,
        persona_id=persona_id,
        nodes=flat.number_of_nodes(),
        edges=flat.number_of_edges(),
        directed=graph.is_directed(),
        criterion=molloy_reed(graph),
        attacks=summary,
        cascades=profile,
        cascade_seeds=seeds,
        coupled=curve,
        coupled_network=coupled_network,
        seed=seed,
        notes=tuple(notes),
    )


def _num(value: float, places: int = 4) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _fraction(value: float | None) -> str:
    return "never" if value is None else f"{value:.0%}"


def render_robustness(report: RobustnessReport) -> str:
    """The robustness sections as markdown: frame, n, null, chapter, then the numbers.

    The four things a reader needs in order to disagree come before anything else, which for
    this report means one unusual line: the null model of a targeted curve is the *random* curve
    beside it, because the question §22.2 asks is comparative from the start.
    """
    title = " — ".join(part for part in (report.persona_id, report.network) if part)
    lines: list[str] = [
        f"# Robustness{' — ' + title if title else ''}",
        "",
        "**Implements.** Atlas ch. 22, *Catastrophic Failures*: random failure and the critical "
        "fraction (§22.1), targeted attack (§22.2), load-redistribution cascades (§22.3) and "
        "interdependent networks (§22.4).",
        "",
        f"**Sampling frame.** {report.frame or 'Not recorded.'}",
        "",
        f"**n.** {report.nodes:,} nodes and {report.edges:,} edges"
        + (", on the flattened undirected view of a directed network" if report.directed else "")
        + f". Every curve below removes a share of those {report.nodes:,} nodes and measures "
        "what is left; the giant-component share is a share of the original n, so it cannot "
        "exceed one minus the fraction removed.",
        "",
        "**Null model.** Random removal is the null the targeted curves are read against — a "
        "behavioural null, not a statistical one: it is what an accident does to this network, "
        "so what an attacker does can be quoted as a difference. It is averaged over "
        f"{report.attacks.random_curve.runs} independent orders"
        + (f", seeded at {report.seed}" if report.seed is not None else "")
        + ". Nothing here is a test and no p-value is computed.",
        "",
        "**What a removal is here.** Not a failure. Nothing flows through a corpus network and "
        "no node in it can be attacked; removing a node removes a *record*. These curves say "
        "how concentrated the evidence is — how much of the co-occurrence structure survives "
        "losing its best-connected records — which is a question about the corpus, and not "
        "about anything breaking (§22, p. 315).",
        "",
    ]
    lines += _criterion_section(report.criterion)
    lines += _attack_section(report.attacks)
    if report.cascades:
        lines += _cascade_section(report)
    if report.coupled is not None:
        lines += _coupled_section(report)
    if report.notes:
        lines += [f"**Note.** {note}" for note in report.notes] + [""]
    return "\n".join(lines)


def _criterion_section(criterion: MolloyReed) -> list[str]:
    """§22.1's closed form, and where it came from."""
    return [
        "## Critical fraction (§22.1)",
        "",
        f"⟨k⟩ = {criterion.mean_degree:.4f}, ⟨k²⟩ = {criterion.mean_square_degree:.4f}, "
        f"κ = ⟨k²⟩/⟨k⟩ = {criterion.kappa:.4f}, f_c = "
        + (
            f"{criterion.critical_fraction:.4f}"
            if criterion.critical_fraction is not None
            else "undefined"
        )
        + ".",
        "",
        criterion.reading,
        "",
        "The criterion kappa > 2 and the form f_c = 1 - 1/(kappa - 1) are not written in chapter "
        "22; it states the phase transition and cites Cohen et al. 2000 for it (footnote 6, p. "
        "318). "
        "Note that the book's own 'Molloy-Reed approach' (§18.1, p. 258) is the "
        "configuration-model algorithm, a different object with the same two names on it.",
        "",
    ]


def _attack_section(summary: AttackSummary) -> list[str]:
    """Figures 22.5 and 22.6 as two tables: the curves, then the comparison."""
    lines = [
        "## Removal curves (§22.1 random, §22.2 targeted)",
        "",
        "Ranking "
        + (
            "recomputed on the survivors at the start of every step"
            if summary.recompute
            else "computed once on the whole network and followed to the end"
        )
        + ".",
        "",
    ]
    curves = [summary.random_curve, *summary.curves]
    header = ["removed", *[curve.strategy for curve in curves]]
    rows: list[list[str]] = []
    for index, point in enumerate(summary.random_curve.points):
        rows.append(
            [
                f"{point.fraction:.0%}",
                *[f"{curve.points[index].giant:.3f}" for curve in curves],
            ]
        )
    lines += _table(header, rows)
    lines += [
        "Giant-component share of the original n at each removed fraction. The random column is "
        f"the mean of {summary.random_curve.runs} orders; its 10-90 band at the halfway point is "
        f"[{summary.random_curve.at(0.5).low:.3f}, {summary.random_curve.at(0.5).high:.3f}].",
        "",
        "### Where each strategy leaves the network",
        "",
    ]
    lines += _table(
        [
            "strategy",
            "§",
            "area under the curve",
            "collapse fraction",
            "shatter fraction",
            *[f"giant at {fraction:.0%}" for fraction in summary.fractions],
        ],
        [
            [
                curve.strategy,
                curve.section,
                _num(curve.area),
                _fraction(curve.collapse_fraction),
                _fraction(curve.shatter_fraction),
                *[f"{curve.at(fraction).giant:.3f}" for fraction in summary.fractions],
            ]
            for curve in curves
        ],
    )
    lines += [
        "*Area under the curve* is the trapezoidal area under the giant-component share over the "
        "removal fractions, higher being more robust; its ceiling is 0.5, because the share is "
        "of the original n. It is this package's summary, not the book's. *Collapse fraction* is "
        "the first fraction at which the largest component no longer holds half of the nodes "
        "still standing — the chapter's 'critical value of |R|' (p. 316) read off the curve. "
        "*Shatter fraction* is where the mean non-giant component is largest, which is p. 315's "
        f"second signal of a network coming apart. The last {len(summary.fractions)} columns are "
        "read at the nearest fraction the curve was actually measured at, which is one step of "
        f"{1 / summary.random_curve.steps:.0%}.",
        "",
        summary.verdict,
        "",
    ]
    for curve in curves:
        lines += [f"- **{curve.strategy}** — {curve.note}"]
    lines += [""]
    return lines


def _cascade_section(report: RobustnessReport) -> list[str]:
    """§22.3's sweep: what slack this network needs to survive one node going down."""
    first = report.cascades[0]
    lines = [
        "## Cascade (§22.3)",
        "",
        f"Load = {first.load}; capacity = (1 + tolerance) * the load the node started with; the "
        f"cascade is lit at {', '.join(report.cascade_seeds)}, the heaviest node, and a node "
        "fails when its load exceeds its capacity, dropping all of it equally on the neighbours "
        "still standing (§22.3, p. 320; §22.6 exercise 4). Total load "
        f"{first.total_load:,.2f} over {first.nodes:,} nodes.",
        "",
    ]
    lines += _table(
        ["tolerance", "failed", "surviving", "giant after", "rounds", "branching", "load lost"],
        [
            [
                f"{run.tolerance:.2f}",
                f"{run.failed:,}",
                f"{run.surviving:.3f}",
                f"{run.giant_share:.3f}",
                str(run.rounds),
                f"{run.branching:.3f}",
                f"{run.lost_load:,.2f}",
            ]
            for run in report.cascades
        ],
    )
    lines += [
        "*Branching* is the mean number of further failures each failure caused: the average "
        "out-degree of figure 22.9's cascade tree, whose critical value is 1 — 'If, on average, "
        "the failure of a node generates another node failure -- or more -- the cascade will "
        "propagate indefinitely' (p. 321). A finite cascade that swallowed the network reads "
        "just under 1 and one that died out reads near 0. *Load lost* is load whose node failed "
        "with no surviving neighbour to take it, which is figure 22.8's last step.",
        "",
    ]
    return lines


def _coupled_section(report: RobustnessReport) -> list[str]:
    """§22.4: the mutual giant component, against what each layer would have done alone."""
    curve = report.coupled
    if curve is None:  # pragma: no cover -- guarded by the caller
        return []
    lines = [
        "## Interdependent networks (§22.4)",
        "",
        f"Coupled to the {report.coupled_network or 'second'} network by `{curve.coupling}`: "
        f"{curve.coupled:,} nodes matched one to one, {curve.uncoupled_a:,} left over in this "
        f"network and {curve.uncoupled_b:,} in the other. Those are outside the analysis — "
        "§22.4's model is a bijection — so every share below is of the "
        f"{curve.coupled:,} coupled nodes, which is a narrower frame than the rest of this "
        f"report. Each fraction is {curve.runs} run{'' if curve.runs == 1 else 's'} averaged.",
        "",
    ]
    lines += _table(
        [
            "removed",
            "mutual giant",
            "this layer alone",
            "other layer alone",
            "mutual, of survivors",
            "iterations",
        ],
        [
            [
                f"{point.fraction:.0%}",
                f"{point.mutual:.3f}",
                f"{point.alone_a:.3f}",
                f"{point.alone_b:.3f}",
                f"{point.mutual_of_survivors:.3f}",
                f"{point.iterations:.1f}",
            ]
            for point in curve.points
        ],
    )
    lines += [curve.verdict, ""]
    return lines


def robustness_payload(report: RobustnessReport) -> dict[str, Any]:
    """The same report as JSON, for a caller that would rather read numbers than prose."""
    return {
        "chapter": 22,
        "network": report.network,
        "persona_id": report.persona_id,
        "frame": report.frame,
        "null_model": (
            "random removal, averaged over independent orders: the damage an accident does, "
            "which is what the targeted curves are quoted against (§22.1-22.2)"
        ),
        "nodes": report.nodes,
        "edges": report.edges,
        "seed": report.seed,
        "criterion": {
            "section": "§22.1",
            "mean_degree": report.criterion.mean_degree,
            "mean_square_degree": report.criterion.mean_square_degree,
            "kappa": report.criterion.kappa,
            "critical_fraction": report.criterion.critical_fraction,
            "has_giant_criterion": report.criterion.has_giant_criterion,
            "reading": report.criterion.reading,
        },
        "curves": [_curve_payload(curve) for curve in _all_curves(report.attacks)],
        "attack": {
            "section": "§22.2",
            "recompute": report.attacks.recompute,
            "worst": report.attacks.worst,
            "verdict": report.attacks.verdict,
            "rows": [
                {
                    "strategy": row.strategy,
                    "area": row.area,
                    "collapse_fraction": row.collapse_fraction,
                    "giant_at": {f"{f:.2f}": value for f, value in row.giant_at},
                    "damage_over_random": {
                        f"{f:.2f}": value for f, value in row.damage_over_random
                    },
                }
                for row in report.attacks.rows
            ],
        },
        "cascades": [
            {
                "section": "§22.3",
                "tolerance": run.tolerance,
                "load": run.load,
                "seeds": list(run.seeds),
                "failed": run.failed,
                "surviving": run.surviving,
                "giant_share": run.giant_share,
                "rounds": run.rounds,
                "branching": run.branching,
                "lost_load": run.lost_load,
                "total_load": run.total_load,
            }
            for run in report.cascades
        ],
        "interdependent": None
        if report.coupled is None
        else {
            "section": "§22.4",
            "network": report.coupled_network,
            "coupling": report.coupled.coupling,
            "coupled": report.coupled.coupled,
            "uncoupled_a": report.coupled.uncoupled_a,
            "uncoupled_b": report.coupled.uncoupled_b,
            "runs": report.coupled.runs,
            "mutual_collapse": report.coupled.mutual_collapse,
            "alone_collapse": report.coupled.alone_collapse,
            "largest_drop": report.coupled.largest_drop,
            "verdict": report.coupled.verdict,
            "points": [
                {
                    "fraction": point.fraction,
                    "mutual": point.mutual,
                    "alone_a": point.alone_a,
                    "alone_b": point.alone_b,
                    "mutual_of_survivors": point.mutual_of_survivors,
                    "alone_of_survivors": point.alone_of_survivors,
                    "iterations": point.iterations,
                }
                for point in report.coupled.points
            ],
        },
        "notes": list(report.notes),
    }


def _all_curves(summary: AttackSummary) -> tuple[RemovalCurve, ...]:
    return (summary.random_curve, *summary.curves)


def _curve_payload(curve: RemovalCurve) -> dict[str, Any]:
    return {
        "strategy": curve.strategy,
        "section": curve.section,
        "recompute": curve.recompute,
        "runs": curve.runs,
        "steps": curve.steps,
        "area": curve.area,
        "collapse_fraction": curve.collapse_fraction,
        "shatter_fraction": curve.shatter_fraction,
        "note": curve.note,
        "points": [
            {
                "fraction": point.fraction,
                "removed": point.removed,
                "giant": point.giant,
                "giant_of_survivors": point.giant_of_survivors,
                "mean_other_size": point.mean_other_size,
                "components": point.components,
                "low": point.low,
                "high": point.high,
            }
            for point in curve.points
        ],
    }
