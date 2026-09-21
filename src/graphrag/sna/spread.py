"""Spreading processes on a network (ch. 20) and the complications that make them realistic
(ch. 21): the compartmental models, the epidemic threshold, immunisation and controllability.

**Nothing spread here.** Every other module in this package measures something the corpus did.
This one does not: chapter 20's models are *dynamics you embed in a network* -- "Here, edges
don't change, but nodes can transition into different states" (p. 286) -- and the network they
are embedded in is a co-mention graph. Two entities are joined because a passage named them
together, never because anything passed between them. So a run of :func:`simulate` is a what-if
about structure: *if* something moved along the lines this corpus drew, at a per-contact rate β,
this is how far it would get and how fast. :data:`SPREAD_FRAME` is that sentence, and every
report prints it above the numbers. A final size of 92% is a statement about how well connected
the network is, not a forecast, and not a claim that any idea in the corpus actually travelled.

**The models.**

``si`` (§20.1, pp. 287-291)
    Susceptible and Infected, one transition, no recovery. "All SI Models, no matter the value
    of β will end up with a complete infection" -- on a connected network; on a corpus network,
    which is usually in pieces, the outbreak stops at its seed's component and the report says
    what that component was. β "is the probability that you will contract the disease after
    meeting an infected individual".
``sis`` (§20.2, pp. 292-295)
    Recovery puts you back in Susceptible at rate µ, so the run settles into an *endemic state*
    -- a constant share infected, "the number of people recovering is perfectly balanced by the
    new infected" -- or dies out. Which of the two happens is the threshold below.
``sir`` (§20.3, pp. 295-296)
    Recovery removes you: "Either you die, or you survive", and either way you are out of the
    outbreak. No endemic state is possible, so the readable numbers are the *final size* (how
    many were ever infected) and the *peak* (when the infected share was highest).
``threshold`` (§21.1, pp. 301-303)
    Complex contagion: "in the latter, you require reinforcement". A susceptible node counts its
    infected neighbours and transitions only when they clear a bar. ``threshold >= 1`` is
    Granovetter's absolute κ ("if κ = 4, you need four infected friends"); ``0 < threshold < 1``
    is Watts' cascade model, where the bar is a *fraction* of your neighbours (p. 303) -- and
    the two behave oppositely on hubs, which is why the chapter separates them: κ is easy for a
    hub to clear and a fraction is nearly impossible.
``limited`` (§21.2, pp. 304-306)
    The independent cascade. Chapter 20's models assume infinite chances -- "The next time step
    represents a new occasion for them to contract the disease. And so on, ad infinitum" (p. 299)
    -- and this one drops that: a node that has just been infected tries each susceptible
    neighbour, and then stops trying forever. "You will be in I forever, this is the reason why
    this is an SI model, but you won't propagate the disease" (p. 305). Its outbreak does not
    fill the network, so which seeds you pick is the whole question (the influence-maximisation
    problem, p. 306; the greedy solver Kempe et al. give for it is **not** built here).

**Which coin gets tossed.** §21.1 distinguishes the "simple" SI, where a susceptible node with
any infected neighbour tosses one β coin per step, from the "classical" reinforcement, where it
tosses one coin *per infected neighbour* and is therefore infected with probability
``1 - (1 - β)^n``. What is implemented for ``si``, ``sis``, ``sir`` and ``limited`` is the
classical form, per edge: it is the standard network implementation, it is what makes a hub a
super-spreader (p. 301: "it's easy to infect hubs: they have more neighbours. More neighbours
mean that they toss their coins much more often"), and the mean-field formulas this module
compares against assume exactly that many contacts per step. For ``threshold`` β keeps the other
meaning the chapter gives it -- "once you clear the κ threshold, you have a chance β < 1 to
contract the disease" (p. 301) -- so ``--beta 1`` is the deterministic Granovetter model.

**What the book defines and what this adds.**

*The spectral threshold.* §20.2 gives two thresholds and no third: on a Gn,p network with
homogeneous mixing the disease is endemic when ``λ = β/µ > 1/(k̄ + 1)``, and on a preferential
attachment network when ``λ > k̄/⟨k²⟩``, which "tends to zero" as the hubs grow, so "any disease,
no matter β and µ, will be endemic in a network with a power law degree distribution" (p. 295).
The exact form for *this* network -- ``λ > 1/λ₁``, with λ₁ the leading eigenvalue of the
adjacency matrix -- is Wang, Chakrabarti, Wang and Faloutsos, *Epidemic spreading in real
networks: an eigenvalue viewpoint* (2003), which chapter 20 cites (p. 297, n. 18) without
stating. It is the one this report reads, because the two the chapter states are properties of a
*model* of the network (a Poisson degree distribution, a power law) and λ₁ is a property of the
network in hand. All three are printed together, and :func:`epidemic_threshold` returns all
three: where they disagree, the disagreement is the finding.

*R₀.* Chapter 20 never writes it. What it defines is ``λ = β/µ``, "the ratio [that] depends
exclusively [on] the pathogen's characteristics" (p. 294). :func:`reproduction` returns that
ratio and, next to it, the expected number of secondary infections one infected node causes in a
fully susceptible network, ``β⟨k² - k⟩/(µ k̄)`` -- the standard heterogeneous mean-field R₀,
which is a convention added here, not the book's. The endemic/dies-out verdict is taken from λ
against the thresholds, which is the chapter's own test.

*Discrete time, synchronous update.* The book's formulas are differential equations it tells you
to integrate (p. 289, n. 7) and its own exercises run in steps. A step here updates every node
from the state at the start of the step: the recoveries of step t use the infected set of step t,
not the set left after that step's infections. That is the usual convention and it is a
convention, not a result -- with large β a step is a coarse approximation of a rate.

*Edge weights are ignored.* β is a per-contact probability and an edge here is a contact. An edge
seen in nine passages is not nine contacts, because the nine passages are nine sentences about
one pair, so the adjacency is read binary. Filter with ``--min-weight`` if a single co-mention is
too thin a contact to run a spread along.

*Immunisation.* §21.3 wants nodes flipped "directly from the S to the R state, without passing by
I", which is what ``immune=`` does: the node starts Removed and stays there for the whole run,
under every model including SIS. The four strategies are the chapter's -- random, by degree, by
betweenness (the chapter names hubs; betweenness is offered because a corpus network's brokers
are not always its hubs), and ``acquaintance``, which is "pick a node at random in the network
and vaccinate one of its friends", Cohen, Havlin and Ben-Avraham (2003), and works because of
the friendship paradox (§31.2). Both of §21.3's success criteria are reported: the drop in final
size (figure 21.9) and the delay, as the area between the two mean curves (figure 21.10).

*Controllability.* §21.4 states the result and explicitly declines to derive it ("the
mathematical details to reach this conclusion are beyond the scope of this book"), so
:func:`driver_nodes` implements the structural-controllability result it points at: Liu, Slotine
and Barabási, *Controllability of complex networks*, Nature 473:167 (2011), where the minimum
driver set is ``N_D = max(1, N - |M*|)`` for ``M*`` a maximum matching of the bipartite graph
whose left side is the nodes as sources and whose right side is the nodes as targets. On a
*directed* network that is the book's number. Our networks are undirected except ``relations``,
and an undirected edge is read as both directions -- stated in the report, because it makes the
answer a statement about a channel that runs both ways rather than about the original result.

**Not built.** The Kempe-Kleinberg-Tardos greedy influence maximisation and the inferred
influence graphs of §21.2 (they need an action log with timestamps, which is a different
observation than this corpus provides); the universality parameter ϕ of p. 304 (the book gives
no estimator); simplicial contagion (p. 287 defers it to §34.1, and so does this package); SEIR,
SIRS, MSEIRS and the rest of the alphabet (p. 297: "the math becomes fiendishly complicated and
it's just not worth delving into that").
"""

from __future__ import annotations

import math
import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np

from graphrag.sna.matrices import adjacency, eigenpairs, node_order
from graphrag.sna.measures import centrality

__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_BETA",
    "DEFAULT_MU",
    "DEFAULT_RUNS",
    "DEFAULT_SHARE",
    "DEFAULT_STEPS",
    "DEFAULT_THRESHOLD",
    "ENDEMIC_TAIL",
    "IMMUNISATION_STRATEGIES",
    "MAX_SPREAD_NODES",
    "MODEL_NOTES",
    "SPREAD_BAND",
    "SPREAD_FRAME",
    "SPREAD_MODELS",
    "STRATEGY_NOTES",
    "DriverNodes",
    "EpidemicThreshold",
    "ImmunisationStrategy",
    "Intervention",
    "Reproduction",
    "SpreadModel",
    "SpreadReport",
    "SpreadResult",
    "SpreadStep",
    "build_spread_report",
    "driver_nodes",
    "epidemic_threshold",
    "immunise",
    "interventions",
    "logistic_curve",
    "mean_field_curve",
    "render_spread",
    "reproduction",
    "simulate",
    "spread_payload",
]

Node = Any
"""A node id, as everywhere else in this package: anything that hashes and sorts."""

SpreadModel = Literal["si", "sis", "sir", "threshold", "limited"]
SPREAD_MODELS: tuple[SpreadModel, ...] = ("si", "sis", "sir", "threshold", "limited")

ImmunisationStrategy = Literal["random", "degree", "betweenness", "acquaintance"]
IMMUNISATION_STRATEGIES: tuple[ImmunisationStrategy, ...] = (
    "random",
    "degree",
    "betweenness",
    "acquaintance",
)

#: One line per model, printed next to its numbers: what the states are and what ends the run.
MODEL_NOTES: dict[str, str] = {
    "si": (
        "Susceptible to Infected, no recovery (§20.1). Everything reachable from the seeds is "
        "infected eventually; beta only sets how fast."
    ),
    "sis": (
        "Infected nodes recover to Susceptible at rate mu and can catch it again (§20.2). The "
        "run settles at an endemic share or dies out, and which one is the threshold's question."
    ),
    "sir": (
        "Infected nodes are Removed at rate mu and never return (§20.3). No endemic state is "
        "possible: the readable numbers are the final size and the peak."
    ),
    "threshold": (
        "Complex contagion (§21.1): a node needs several infected neighbours at once, not one. "
        "A threshold of 1 or more is Granovetter's absolute kappa; below 1 it is Watts' cascade "
        "fraction of the neighbourhood."
    ),
    "limited": (
        "The independent cascade (§21.2): a newly infected node tries each susceptible "
        "neighbour for --attempts steps and then stops trying forever. The outbreak stalls, so "
        "the seeds decide the answer."
    ),
}

#: Which parameters each model reads. A report that prints mu for an SI run is lying about the
#: model it ran.
PARAMETERS: dict[str, tuple[str, ...]] = {
    "si": ("beta",),
    "sis": ("beta", "mu"),
    "sir": ("beta", "mu"),
    "threshold": ("beta", "threshold"),
    "limited": ("beta", "attempts"),
}

#: What each immunisation strategy does and where the chapter puts it.
STRATEGY_NOTES: dict[str, str] = {
    "random": "immunise uniformly at random: §21.3's baseline, and what a blind campaign gets.",
    "degree": (
        "immunise the highest-degree nodes: §21.3's obvious answer -- 'who should we vaccinate? "
        "the hubs' -- and it needs the whole topology to be known."
    ),
    "betweenness": (
        "immunise the highest-betweenness nodes: the brokers rather than the hubs. Not the "
        "chapter's strategy; offered because a corpus network's brokers need not be its hubs."
    ),
    "acquaintance": (
        "pick a node at random and immunise one of its neighbours (Cohen, Havlin and "
        "Ben-Avraham 2003, §21.3). It needs no topology at all and still finds hubs, because a "
        "random neighbour of a random node is a hub more often than a random node is -- the "
        "friendship paradox (§31.2)."
    ),
}

#: The what-if sentence. Every spread report prints it before any number.
SPREAD_FRAME = (
    "Nothing spread here. The edges of this network are co-occurrences in a corpus -- two nodes "
    "are joined because the same document or passage named them together -- so no disease, idea "
    "or message ever travelled along one of them. What follows is a what-if over the structure: "
    "*if* something moved along the lines this corpus drew, at the stated rate, this is how far "
    "it would get and how fast. Read the final size as a statement about how well connected the "
    "network is, never as a forecast and never as evidence that anything did spread. The run "
    "also ignores edge weights: an edge is one contact, whether the corpus drew it from one "
    "passage or from nine."
)

DEFAULT_BETA = 0.2
"""The per-contact infection probability the chapter's own exercises use (§20.5, §21.6)."""

DEFAULT_MU = 0.1
"""The recovery probability per step. The book varies it and fixes no default; this is one."""

DEFAULT_THRESHOLD = 2.0
"""kappa = 2, the value §21.6's first exercise sets. Below 1 the same number is a fraction."""

DEFAULT_ATTEMPTS = 1
"""Steps a newly infected node stays contagious in ``limited``. 1 is the independent cascade of
figure 21.5, where a node infected at t tries its neighbours at t+1 and then gives up."""

DEFAULT_STEPS = 50
"""Steps per run: the length §21.6's exercises use."""

DEFAULT_RUNS = 20
"""Simulations per report. The models are stochastic, so a single run is an anecdote."""

DEFAULT_SHARE = 0.1
"""Share of nodes an intervention immunises when none is given."""

SPREAD_BAND = 0.95
"""Width of the band printed around every mean: the 2.5th and 97.5th percentiles over the runs.
It describes the stochasticity of the process on this fixed network, and nothing else."""

ENDEMIC_TAIL = 0.25
"""Share of the run, at the end, whose mean infected share is reported as the endemic level
(§20.2's t -> infinity, which a finite run can only approach)."""

MAX_SPREAD_NODES = 3000
"""The largest network a spread runs on. Both halves of this module are dense: a step multiplies
the ``runs x n`` state matrix by the ``n x n`` adjacency, and the threshold takes the full
eigendecomposition of that same matrix (the exact solver, for the reasons
:func:`graphrag.sna.matrices.eigenpairs` gives). Above this the answer is a smaller network --
``--min-weight``, a backbone (ch. 27) or a sample (ch. 29) -- not a slower run, so it refuses."""

#: Rows in the curve table. The trajectory is kept in full in the payload; the table is a
#: readable sample of it.
CURVE_ROWS = 12


# ----------------------------------------------------------------------------- the results


@dataclass(frozen=True)
class SpreadStep:
    """One time step of the simulation, averaged over the runs, as shares of all nodes.

    ``infected_low`` and ``infected_high`` are the percentiles of :data:`SPREAD_BAND` over the
    runs, so they say how much the *process* varies on this one network -- not how uncertain the
    network is. ``ever`` is the share that has been infected at least once by this step, the only
    readable infection count for SIS (where a node can be infected, recover and be infected
    again) and equal to ``infected + removed`` for the other models.
    """

    step: int
    susceptible: float
    infected: float
    removed: float
    ever: float
    infected_low: float
    infected_high: float


@dataclass(frozen=True)
class SpreadResult:
    """What ``runs`` simulations of one model on one network did.

    ``final_size`` is the share of nodes ever infected by the last step -- §21.3's first success
    criterion -- with its band. ``peak_infected`` is the highest simultaneous infected share
    (mean over runs of each run's own peak) and ``peak_step`` the mean step at which each run
    reached it, which is undefined (``None``) when nothing was ever infected. ``endemic`` is the
    mean infected share over the last :data:`ENDEMIC_TAIL` of the run, which is the endemic level
    of §20.2 for SIS and an artefact of the last few steps for anything else; ``died_out`` is the
    share of runs with nobody infected at the last step.

    ``trajectories`` holds one row per run per step (``runs x (steps+1) x 4``: susceptible,
    infected, removed, ever, as counts) when the caller asked for it, and is ``None`` otherwise.
    """

    model: SpreadModel
    nodes: int
    edges: int
    seeds: tuple[Node, ...]
    seed_count: int
    immune: tuple[Node, ...]
    beta: float
    mu: float
    threshold: float
    attempts: int
    steps: tuple[SpreadStep, ...]
    runs: int
    seed: int | None
    final_size: float
    final_low: float
    final_high: float
    peak_infected: float
    peak_step: float | None
    endemic: float
    died_out: float
    trajectories: tuple[tuple[tuple[int, int, int, int], ...], ...] | None = None

    @property
    def parameters(self) -> dict[str, float]:
        """Only the parameters this model actually reads (:data:`PARAMETERS`)."""
        values = {
            "beta": self.beta,
            "mu": self.mu,
            "threshold": self.threshold,
            "attempts": float(self.attempts),
        }
        return {name: values[name] for name in PARAMETERS[self.model]}


@dataclass(frozen=True)
class EpidemicThreshold:
    """The three epidemic thresholds of §20.2, for one network.

    Each is the value ``λ = β/µ`` has to exceed for the disease to be endemic, under a different
    assumption about the network. ``spectral`` is ``1/λ₁`` and is the one to read (see the module
    docstring); ``homogeneous`` is ``1/(k̄ + 1)``, what §20.2 gives for a Gn,p graph; and
    ``heterogeneous`` is ``k̄/⟨k²⟩``, what it gives for a preferential-attachment one, the form
    that "tends to zero" on a heavy tail and makes every disease endemic.

    Undefined, and ``None``, on a network with no edges: there is nothing for a spread to use.
    """

    nodes: int
    edges: int
    mean_degree: float
    mean_square_degree: float
    leading_eigenvalue: float | None
    spectral: float | None
    homogeneous: float
    heterogeneous: float | None

    @property
    def reading(self) -> str:
        """What the three numbers, held together, say about this network."""
        if self.spectral is None or self.heterogeneous is None:
            return (
                "No edges, so no threshold: nothing can spread on this network at any beta and "
                "the simulation below infects only its seeds."
            )
        gap = self.homogeneous / self.spectral if self.spectral > 0 else math.inf
        heavy = (
            f"The heterogeneous form is {self.homogeneous / self.heterogeneous:.1f} times lower "
            "than the homogeneous one"
            if self.heterogeneous > 0
            else "The heterogeneous form is zero"
        )
        return (
            f"A hub lowers the bar. {heavy}, because squaring the degrees lets the hubs eclipse "
            "everyone else, and the spectral threshold -- the exact one for this network -- sits "
            f"{gap:.1f} times below the homogeneous-mixing value a Gn,p graph of the same mean "
            "degree would have. The more unequal the degrees, the smaller the beta/mu that keeps "
            "a spread alive."
        )


@dataclass(frozen=True)
class Reproduction:
    """β and µ read against this network's thresholds (§20.2), plus an R₀ the book does not give.

    ``ratio`` is the book's ``λ = β/µ``. ``spectral``, ``homogeneous`` and ``heterogeneous`` are
    ``λ`` divided by the matching threshold: above 1 means endemic under that reading. ``r0`` is
    the expected number of secondary infections one infected node causes in an otherwise
    susceptible network, ``β⟨k² - k⟩/(µ k̄)``; it is the standard heterogeneous mean-field
    quantity and a convention this module adds, so it is labelled as such wherever it prints.

    Undefined, and raising, for ``mu <= 0``: with no recovery there is no ratio to take, which is
    why §20.1's SI model has no threshold at all -- it always saturates.
    """

    beta: float
    mu: float
    ratio: float
    r0: float | None
    spectral: float | None
    homogeneous: float
    heterogeneous: float | None

    @property
    def endemic(self) -> bool | None:
        """Whether §20.2 expects the disease to persist, by the spectral threshold. ``None`` when
        the network has no edges and the threshold does not exist."""
        return None if self.spectral is None else self.spectral > 1.0


@dataclass(frozen=True)
class Intervention:
    """One immunisation strategy run against the same outbreak with nobody immunised (§21.3).

    ``reduction`` is the first success criterion, figure 21.9: the final size without the
    immunisation minus the final size with it. ``delay`` is the second, figure 21.10: the area
    between the two mean infected-ever curves, in node-shares x steps, which is positive when the
    immunised run got everywhere later even if it got everywhere in the end.

    ``expected_neighbour_degree`` is ``⟨k²⟩/⟨k⟩``, the expected degree of the node at the end of
    a randomly chosen edge -- the exact form of the friendship paradox
    (:func:`graphrag.sna.assortativity.friendship_paradox`, §31.2) and therefore what one
    acquaintance draw is worth before the duplicates are dropped. It is the analytic comparison
    for the acquaintance strategy's mean degree, and the reason §21.3 says the strategy works.

    ``immunised_mean_degree`` beside ``network_mean_degree`` is what makes the acquaintance
    strategy legible: it immunises no more nodes than the random one and catches a far higher
    mean degree. ``seeds_immunised`` counts seeds the strategy happened to vaccinate, which is a
    real outcome of a blind strategy rather than a bug -- but it flatters the comparison, so it
    is printed.
    """

    strategy: ImmunisationStrategy
    share: float
    requested: int
    immunised: tuple[Node, ...]
    immunised_mean_degree: float
    network_mean_degree: float
    expected_neighbour_degree: float
    seeds_immunised: int
    baseline: SpreadResult
    treated: SpreadResult
    reduction: float
    delay: float


@dataclass(frozen=True)
class DriverNodes:
    """The minimum set of nodes that has to be steered to steer the whole network (§21.4).

    ``count`` is ``N_D = max(1, N - |M*|)`` of Liu, Slotine and Barabási (2011) and ``share`` is
    ``N_D/N``, the number the paper reads: near 0 the network is easy to control from a handful
    of nodes, near 1 almost every node has to be driven. ``nodes`` is *one* minimum set -- the
    nodes left unmatched by the matching that was found. The **size** is unique; the set is not,
    which is the book's own figure 21.11 point that some nodes "could or could not" be chosen, so
    read the count and treat the list as an example.

    ``directed`` says whether this was computed on a directed network. On an undirected one every
    edge is read as running both ways, which is a choice this module makes and the report states.
    """

    count: int
    share: float
    nodes: tuple[Node, ...]
    matching: int
    directed: bool
    total: int


@dataclass(frozen=True)
class SpreadReport:
    """Everything ``sna spread`` prints: one simulation, its thresholds, and the two chapter-21
    sections that do not depend on it."""

    persona_id: str
    network: str
    frame: str
    result: SpreadResult
    threshold: EpidemicThreshold
    reproduction: Reproduction | None
    mean_field: tuple[float, ...] | None
    components: int
    largest_component: int
    intervention: Intervention | None
    drivers: DriverNodes


# ----------------------------------------------------------------------------- the simulation


def _model_or_refuse(model: str) -> SpreadModel:
    if model not in SPREAD_MODELS:
        msg = f"--model must be one of {', '.join(SPREAD_MODELS)}, got {model!r}"
        raise ValueError(msg)
    return model


def _strategy_or_refuse(strategy: str) -> ImmunisationStrategy:
    if strategy not in IMMUNISATION_STRATEGIES:
        msg = f"--strategy must be one of {', '.join(IMMUNISATION_STRATEGIES)}, got {strategy!r}"
        raise ValueError(msg)
    return strategy


def _probability(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} is a probability and must be between 0 and 1, got {value}"
        raise ValueError(msg)
    return float(value)


def _small_enough(graph: nx.Graph, what: str) -> None:
    """Refuse a network too large for a dense run, naming the ways to make it smaller."""
    if graph.number_of_nodes() > MAX_SPREAD_NODES:
        msg = (
            f"{what} is dense and this network has {graph.number_of_nodes():,} nodes, above the "
            f"{MAX_SPREAD_NODES:,} ceiling. Cut it down first -- --min-weight, a backbone (`sna "
            "backbone`) or a sample (`sna sample`) -- rather than waiting for an n x n matrix."
        )
        raise ValueError(msg)


def _binary_adjacency(graph: nx.Graph, order: Sequence[Node]) -> np.ndarray:
    """The adjacency matrix with every edge counted once, whatever the corpus weighted it.

    ``weight=None`` is the book's matrix: β is a per-contact probability and an edge is one
    contact (see the module docstring). Row ``u`` holds ``u``'s out-edges, so the number of
    infected *in*-neighbours of ``v`` is column ``v`` -- which is why every transmission product
    below is ``infected @ A`` and not ``A @ infected``, and why a directed network spreads the
    way its edges point.
    """
    matrix, _ = adjacency(graph, list(order), weight=None)
    return np.asarray(matrix, dtype=np.float64)


def _seed_sets(
    order: Sequence[Node],
    seeds: Sequence[Node] | int | None,
    immune_index: set[int],
    runs: int,
    rng: random.Random,
) -> tuple[list[list[int]], tuple[Node, ...], int]:
    """One seed set per run, plus what to print: the fixed seeds, or how many were drawn.

    An explicit list is used unchanged in every run, which is what makes two runs comparable.
    An integer (or nothing, meaning one) draws that many nodes uniformly per run, the way §20.5
    asks -- "pick a random node and place it in the Infected state" -- and the report then names
    the count rather than the nodes, because they differ per run.
    """
    index = {node: i for i, node in enumerate(order)}
    if seeds is not None and not isinstance(seeds, int):
        chosen = list(seeds)
        if not chosen:
            msg = "simulate() needs at least one seed node: nothing can spread from nobody"
            raise ValueError(msg)
        missing = [node for node in chosen if node not in index]
        if missing:
            msg = f"seed node(s) not in this network: {', '.join(str(m) for m in missing[:5])}"
            raise ValueError(msg)
        rows = [index[node] for node in chosen]
        return [rows for _ in range(runs)], tuple(chosen), len(rows)
    count = 1 if seeds is None else int(seeds)
    pool = [i for i in range(len(order)) if i not in immune_index]
    if count < 1:
        msg = f"the number of seeds must be at least 1, got {count}"
        raise ValueError(msg)
    if count > len(pool):
        msg = f"asked for {count} random seeds but only {len(pool)} nodes can be seeded"
        raise ValueError(msg)
    return [sorted(rng.sample(pool, count)) for _ in range(runs)], (), count


def simulate(
    graph: nx.Graph,
    model: str = "si",
    *,
    seeds: Sequence[Node] | int | None = None,
    beta: float = DEFAULT_BETA,
    mu: float = DEFAULT_MU,
    threshold: float = DEFAULT_THRESHOLD,
    attempts: int = DEFAULT_ATTEMPTS,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
    immune: Collection[Node] = (),
    keep_trajectories: bool = False,
) -> SpreadResult:
    """Run one of chapter 20 and 21's models on ``graph`` ``runs`` times and band the result.

    The models are listed in :data:`MODEL_NOTES` and described in the module docstring. Every
    step updates every node from the state at the start of that step, and the runs are
    independent draws of the same process on the same network, so the band around each step is
    the stochasticity of the process and nothing else -- there is no null model here, because a
    simulation is not a measurement that could have come out by chance.

    ``seeds`` is the patient-zero set: a list of nodes, used unchanged in every run, or an
    integer, which draws that many nodes uniformly at random per run (§20.5's own instruction),
    or ``None`` for one random node. ``immune`` are nodes that start Removed and stay there --
    §21.3's vaccination, "flip some people directly from the S to the R state, without passing
    by I" -- and they are excluded from a random seed draw.

    ``beta`` is the per-contact infection probability; with several infected neighbours the
    chance of escaping them all is ``(1 - β)^n``, which is §21.1's classical reinforcement. For
    ``threshold`` it is instead the chance of transitioning *once the trigger is cleared*
    (p. 301), so ``beta=1`` is the deterministic model. ``mu`` is the per-step recovery
    probability, read only by ``sis`` and ``sir``. ``threshold`` is κ when it is 1 or more and a
    fraction of the neighbourhood when it is below 1. ``attempts`` is how many steps a newly
    infected node keeps trying its neighbours in ``limited``.

    Raises ``ValueError`` for an empty graph, an unknown model, a probability outside [0, 1], a
    non-positive threshold, fewer than one step, run or attempt, or a seed the graph does not
    have.
    """
    name = _model_or_refuse(model)
    if graph.number_of_nodes() == 0:
        msg = "simulate() needs a network with at least one node"
        raise ValueError(msg)
    _small_enough(graph, "the simulation")
    beta = _probability("beta", beta)
    mu = _probability("mu", mu)
    if threshold <= 0:
        msg = f"--threshold must be positive: kappa >= 1, or a fraction in (0, 1), got {threshold}"
        raise ValueError(msg)
    for label, value in (("--steps", steps), ("--runs", runs), ("--attempts", attempts)):
        if value < 1:
            msg = f"{label} must be at least 1, got {value}"
            raise ValueError(msg)

    order = node_order(graph)
    matrix = _binary_adjacency(graph, order)
    position = {node: i for i, node in enumerate(order)}
    immune_nodes = tuple(node for node in order if node in set(immune))
    immune_index = {position[node] for node in immune_nodes}
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    seed_rows, named_seeds, seed_count = _seed_sets(order, seeds, immune_index, runs, rng)
    generator = np.random.default_rng(seed)
    history = _run(
        matrix,
        name,
        seed_rows=seed_rows,
        immune_index=sorted(immune_index),
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        steps=steps,
        generator=generator,
    )
    return _summarise(
        history,
        model=name,
        graph=graph,
        seeds=named_seeds,
        seed_count=seed_count,
        immune=immune_nodes,
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        runs=runs,
        seed=seed,
        keep_trajectories=keep_trajectories,
    )


def _run(
    matrix: np.ndarray,
    model: SpreadModel,
    *,
    seed_rows: Sequence[Sequence[int]],
    immune_index: Sequence[int],
    beta: float,
    mu: float,
    threshold: float,
    attempts: int,
    steps: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Every run of the model at once: a ``runs x (steps+1) x 4`` array of counts.

    All runs advance together as rows of a ``runs x n`` state matrix, so one step of the whole
    ensemble is one matrix product ``T @ A`` -- ``T[r, u]`` says node ``u`` is transmitting in
    run ``r``, and the product counts, for every run and every node, how many of its
    in-neighbours are transmitting. The columns of the result are the counts of susceptible,
    infected, removed and ever-infected nodes.
    """
    runs, nodes = len(seed_rows), matrix.shape[0]
    # 0 susceptible, 1 infected, 2 removed. Immunised nodes start removed and never move (§21.3).
    state = np.zeros((runs, nodes), dtype=np.int8)
    if immune_index:
        state[:, list(immune_index)] = 2
    ever = np.zeros((runs, nodes), dtype=bool)
    for run, rows in enumerate(seed_rows):
        for row in rows:
            if state[run, row] == 0:
                state[run, row] = 1
                ever[run, row] = True
    # How many more steps each infected node will keep trying its neighbours (``limited`` only).
    remaining = np.where(state == 1, attempts, 0).astype(np.int32)
    in_degree = matrix.sum(axis=0)

    history = np.zeros((runs, steps + 1, 4), dtype=np.int64)
    history[:, 0, :] = _counts(state, ever)
    for step in range(1, steps + 1):
        infected = state == 1
        transmitting = infected & (remaining > 0) if model == "limited" else infected
        exposure = transmitting.astype(np.float64) @ matrix
        chance = _transition_chance(model, exposure, in_degree, beta, threshold)
        caught = (state == 0) & (generator.random((runs, nodes)) < chance)
        if model in {"sis", "sir"}:
            healed = infected & (generator.random((runs, nodes)) < mu)
            state = np.where(healed, 0 if model == "sis" else 2, state)
        if model == "limited":
            remaining = np.where(transmitting, remaining - 1, remaining)
            state = np.where(infected & (remaining <= 0), 2, state)
        state = np.where(caught, 1, state)
        remaining = np.where(caught, attempts, remaining)
        ever = ever | caught
        history[:, step, :] = _counts(state, ever)
    return history


def _transition_chance(
    model: SpreadModel,
    exposure: np.ndarray,
    in_degree: np.ndarray,
    beta: float,
    threshold: float,
) -> np.ndarray:
    """The probability each node transitions this step, given how many neighbours are infectious.

    For the simple models this is §21.1's classical reinforcement, ``1 - (1 - β)^n``: one
    independent coin per infected neighbour, so a hub tosses more of them. For ``threshold`` it
    is β above the trigger and 0 below it -- the trigger being κ infected neighbours when
    ``threshold >= 1`` (p. 301) and that fraction of the neighbourhood when it is below 1
    (p. 303) -- and a node with no infected neighbour never transitions whatever the arithmetic
    of a zero threshold would say.
    """
    if model == "threshold":
        needed = (
            np.full_like(in_degree, threshold)
            if threshold >= 1.0
            else np.maximum(threshold * in_degree, 1.0)
        )
        triggered = (exposure > 0) & (exposure >= needed)
        return np.where(triggered, beta, 0.0)
    return np.asarray(1.0 - (1.0 - beta) ** exposure, dtype=np.float64)


def _counts(state: np.ndarray, ever: np.ndarray) -> np.ndarray:
    """Susceptible, infected, removed and ever-infected counts per run."""
    return np.stack(
        [
            (state == 0).sum(axis=1),
            (state == 1).sum(axis=1),
            (state == 2).sum(axis=1),
            ever.sum(axis=1),
        ],
        axis=1,
    )


def _summarise(
    history: np.ndarray,
    *,
    model: SpreadModel,
    graph: nx.Graph,
    seeds: tuple[Node, ...],
    seed_count: int,
    immune: tuple[Node, ...],
    beta: float,
    mu: float,
    threshold: float,
    attempts: int,
    runs: int,
    seed: int | None,
    keep_trajectories: bool,
) -> SpreadResult:
    """Turn the raw per-run counts into the shares, bands and summary numbers a report prints."""
    nodes = graph.number_of_nodes()
    shares = history.astype(np.float64) / float(nodes)
    low, high = (1.0 - SPREAD_BAND) / 2.0, (1.0 + SPREAD_BAND) / 2.0
    infected = shares[:, :, 1]
    bands_low = np.quantile(infected, low, axis=0)
    bands_high = np.quantile(infected, high, axis=0)
    means = shares.mean(axis=0)
    steps = tuple(
        SpreadStep(
            step=index,
            susceptible=float(means[index, 0]),
            infected=float(means[index, 1]),
            removed=float(means[index, 2]),
            ever=float(means[index, 3]),
            infected_low=float(bands_low[index]),
            infected_high=float(bands_high[index]),
        )
        for index in range(history.shape[1])
    )
    final = shares[:, -1, 3]
    peaks = infected.max(axis=1)
    peak_step = float(np.argmax(infected, axis=1).mean()) if float(peaks.max()) > 0 else None
    tail = max(1, round(ENDEMIC_TAIL * (history.shape[1] - 1)))
    return SpreadResult(
        model=model,
        nodes=nodes,
        edges=graph.number_of_edges(),
        seeds=seeds,
        seed_count=seed_count,
        immune=immune,
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        steps=steps,
        runs=runs,
        seed=seed,
        final_size=float(final.mean()),
        final_low=float(np.quantile(final, low)),
        final_high=float(np.quantile(final, high)),
        peak_infected=float(peaks.mean()),
        peak_step=peak_step,
        endemic=float(infected[:, -tail:].mean()),
        died_out=float((history[:, -1, 1] == 0).mean()),
        trajectories=(
            tuple(tuple((int(a), int(b), int(c), int(d)) for a, b, c, d in run) for run in history)
            if keep_trajectories
            else None
        ),
    )


# ----------------------------------------------------------------------------- §20.2 thresholds


def epidemic_threshold(graph: nx.Graph) -> EpidemicThreshold:
    """The three epidemic thresholds of §20.2 for this network, and the mean degrees behind them.

    Each is a value of ``λ = β/µ``: above it the SIS model settles into an endemic state, below
    it the disease dies out. ``spectral = 1/λ₁`` is exact for this network (Wang et al. 2003, the
    reference §20.2 points at); ``homogeneous = 1/(k̄ + 1)`` is what §20.2 gives for a Gn,p graph
    (p. 294) and ``heterogeneous = k̄/⟨k²⟩`` what it gives for a preferential-attachment one
    (p. 294), the form that collapses towards zero on a heavy tail -- "any disease, no matter β
    and µ, will be endemic in a network with a power law degree distribution".

    λ₁ is read from the *binary* adjacency matrix, because β is a per-contact probability and an
    edge is one contact. On a directed network the matrix is not symmetric and its leading
    eigenvalue can be complex; the real part is taken and the report says the network was
    directed. Undefined on a network with no edges: ``leading_eigenvalue``, ``spectral`` and
    ``heterogeneous`` are then ``None``, because nothing can spread at any β.
    """
    order = node_order(graph)
    degrees = np.array([float(d) for _, d in graph.degree()], dtype=np.float64)
    mean_degree = float(degrees.mean()) if degrees.size else 0.0
    mean_square = float((degrees**2).mean()) if degrees.size else 0.0
    homogeneous = 1.0 / (mean_degree + 1.0)
    if graph.number_of_edges() == 0:
        return EpidemicThreshold(
            nodes=graph.number_of_nodes(),
            edges=0,
            mean_degree=mean_degree,
            mean_square_degree=mean_square,
            leading_eigenvalue=None,
            spectral=None,
            homogeneous=homogeneous,
            heterogeneous=None,
        )
    _small_enough(graph, "the spectral threshold")
    matrix = _binary_adjacency(graph, order)
    values, _ = eigenpairs(matrix, k=1, largest=True)
    leading = float(np.real(values[0]))
    return EpidemicThreshold(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        mean_degree=mean_degree,
        mean_square_degree=mean_square,
        leading_eigenvalue=leading,
        spectral=1.0 / leading if leading > 0 else None,
        homogeneous=homogeneous,
        heterogeneous=mean_degree / mean_square if mean_square > 0 else None,
    )


def reproduction(beta: float, mu: float, graph: nx.Graph) -> Reproduction:
    """``λ = β/µ`` against this network's thresholds, plus the mean-field R₀ (§20.2, §20.3).

    §20.2's test is ``λ`` against a threshold, and the three ratios returned here are ``λ``
    divided by each of :func:`epidemic_threshold`'s three: above 1 the disease is endemic under
    that reading, below 1 it dies out. ``r0 = β⟨k² - k⟩/(µ k̄)`` is the expected number of
    secondary infections one infected node causes in an otherwise susceptible network; chapter 20
    does not define it, and it is here because it is the number a reader of any other epidemic
    text will look for. It is ``None`` when the network has no edges.

    Raises ``ValueError`` for ``mu <= 0``: an SI model has no recovery, so ``λ`` does not exist
    and there is no threshold to be above -- which is exactly why §20.1's SI always saturates.
    """
    if mu <= 0:
        msg = (
            "beta/mu is undefined for mu = 0: the SI model has no recovery and no threshold, so "
            "it always saturates (§20.1). Use --model sis or sir to ask this question."
        )
        raise ValueError(msg)
    beta = _probability("beta", beta)
    mu = _probability("mu", mu)
    limits = epidemic_threshold(graph)
    ratio = beta / mu
    excess = limits.mean_square_degree - limits.mean_degree
    return Reproduction(
        beta=beta,
        mu=mu,
        ratio=ratio,
        r0=(beta * excess / (mu * limits.mean_degree)) if limits.mean_degree > 0 else None,
        spectral=(ratio / limits.spectral) if limits.spectral else None,
        homogeneous=ratio / limits.homogeneous,
        heterogeneous=(ratio / limits.heterogeneous) if limits.heterogeneous else None,
    )


def logistic_curve(beta: float, mean_degree: float, *, i0: float, steps: int) -> tuple[float, ...]:
    """§20.1's closed-form SI curve under homogeneous mixing (p. 289).

    ``i(t) = i₀ e^(β k̄ t) / (1 - i₀ + i₀ e^(β k̄ t))``: the S-shape of figure 20.2, exponential
    while ``i₀`` is small and flat once there is nobody left to infect. It assumes every
    susceptible is equally likely to meet every infected -- "in homogenous mixing, the global
    social network is a lattice" (p. 289) -- which no corpus network is, so it is printed as the
    curve this run *would* have had, not as a prediction of it.

    Returns ``steps + 1`` shares, starting at ``i₀``.
    """
    rate = beta * mean_degree
    curve = []
    for step in range(steps + 1):
        growth = math.exp(min(rate * step, 700.0))
        curve.append(i0 * growth / (1.0 - i0 + i0 * growth))
    return tuple(curve)


def mean_field_curve(
    model: str, *, beta: float, mu: float, mean_degree: float, i0: float, steps: int
) -> tuple[float, ...]:
    """The infected share step by step under homogeneous mixing, by the book's own recursion.

    §20.1 gives ``i_{t+1} = i_t + β k̄ i_t (1 - i_t)`` (p. 289, n. 7, which is where the book
    writes the full form rather than only the increment); §20.2 subtracts the recoveries,
    ``- µ i_t``, and that is the SIS equation whose fixed point is the endemic state; §20.3 keeps
    the same infection term but sends the recoveries to R, so the susceptible pool shrinks by
    more than the infected pool grows (``r_{t+1} = µ i_t``, p. 296).

    This is the mean field, not this network: it knows only ``k̄``, so it cannot see hubs, and
    §20.1's whole point is that hubs change the answer. It is the line a report plots beside the
    simulation to show the distance between the two. ``threshold`` and ``limited`` have no
    closed form in the book and raise here.
    """
    name = _model_or_refuse(model)
    if name in {"threshold", "limited"}:
        msg = (
            f"chapter 21 gives no mean-field equation for the {name} model: complex contagion "
            "depends on the neighbourhood, which a homogeneous-mixing formula cannot see"
        )
        raise ValueError(msg)
    rate = beta * mean_degree
    infected, susceptible = i0, 1.0 - i0
    curve = [i0]
    for _ in range(steps):
        caught = rate * infected * susceptible
        healed = 0.0 if name == "si" else mu * infected
        susceptible = susceptible - caught + (healed if name == "sis" else 0.0)
        infected = min(max(infected + caught - healed, 0.0), 1.0)
        susceptible = min(max(susceptible, 0.0), 1.0)
        curve.append(infected)
    return tuple(curve)


# ----------------------------------------------------------------------------- §21.3 immunisation


def immunise(
    graph: nx.Graph,
    share: float = DEFAULT_SHARE,
    strategy: str = "random",
    *,
    seed: int | None = None,
) -> tuple[Node, ...]:
    """Pick the nodes an intervention vaccinates (§21.3). Returns them in node order.

    ``share`` is the share of the network to immunise, rounded to a whole number of nodes and at
    least one. The strategies are :data:`STRATEGY_NOTES`. ``acquaintance`` is the one the chapter
    argues for: it picks a node uniformly at random and immunises **one of its neighbours**,
    which needs no knowledge of the topology and still finds hubs, because a random neighbour is
    reached through every one of its edges. It can fall short of its target on a network of
    isolated nodes -- there is nobody to be a friend of -- and then returns what it found.

    Raises ``ValueError`` for an unknown strategy or a share outside ``(0, 1]``.
    """
    _strategy_or_refuse(strategy)
    if not 0.0 < share <= 1.0:
        msg = f"--share is a share of the nodes and must be in (0, 1], got {share}"
        raise ValueError(msg)
    order = node_order(graph)
    if not order:
        return ()
    target = max(1, round(share * len(order)))
    rng = random.Random(seed)  # noqa: S311 -- reproducibility, not secrecy
    if strategy == "random":
        chosen = set(rng.sample(order, min(target, len(order))))
    elif strategy == "acquaintance":
        chosen = _acquaintances(graph, order, target, rng)
    else:
        kind = "degree" if strategy == "degree" else "betweenness"
        scores = centrality(graph, kind)
        ranked = sorted(order, key=lambda node: (-scores.get(node, 0.0), str(node)))
        chosen = set(ranked[:target])
    return tuple(node for node in order if node in chosen)


def _acquaintances(
    graph: nx.Graph, order: Sequence[Node], target: int, rng: random.Random
) -> set[Node]:
    """§21.3's "pick a node at random in the network and vaccinate one of its friends".

    Draws with replacement, as the strategy does: the same hub will be named by several of its
    neighbours, which is the point -- it is reached with probability proportional to its degree.
    Gives up after a bounded number of draws so that a network of isolated nodes, where the
    strategy has nothing to vaccinate, cannot spin forever.
    """
    chosen: set[Node] = set()
    budget = 50 * target + 100
    for _ in range(budget):
        if len(chosen) >= target:
            break
        neighbours = list(graph.neighbors(rng.choice(list(order))))
        if neighbours:
            chosen.add(rng.choice(neighbours))
    return chosen


def interventions(
    graph: nx.Graph,
    share: float = DEFAULT_SHARE,
    strategy: str = "random",
    *,
    model: str = "sir",
    seeds: Sequence[Node] | int | None = None,
    beta: float = DEFAULT_BETA,
    mu: float = DEFAULT_MU,
    threshold: float = DEFAULT_THRESHOLD,
    attempts: int = DEFAULT_ATTEMPTS,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    seed: int | None = None,
) -> Intervention:
    """Run the same outbreak twice, once with ``share`` of the network immunised (§21.3).

    Both runs use the same seeds and the same random seed, so the difference between them is the
    immunisation and not the draw. §21.3's two success criteria come back on the result: the drop
    in final size (figure 21.9) and the delay, as the area between the two mean curves (figure
    21.10) -- "we might want to either calculate the time t at which the system reaches
    saturation, or compute the area between the two curves as a more precise sense of the delay
    we imposed".

    When ``seeds`` is an integer or ``None`` the seed nodes are drawn once here and reused, which
    is what makes the two runs comparable. A strategy that happens to immunise a seed keeps it
    immune -- a vaccinated patient zero never starts an outbreak, which is a real outcome of a
    blind campaign -- and the count of those is on the result, because it flatters the strategy.
    """
    order = node_order(graph)
    if not order:
        msg = "interventions() needs a network with at least one node"
        raise ValueError(msg)
    immunised = immunise(graph, share, strategy, seed=seed)
    fixed: Sequence[Node]
    if seeds is not None and not isinstance(seeds, int):
        fixed = list(seeds)
    else:
        count = 1 if seeds is None else int(seeds)
        if count > len(order):
            msg = f"asked for {count} random seeds but the network has {len(order)} nodes"
            raise ValueError(msg)
        fixed = sorted(random.Random(seed).sample(order, count), key=str)  # noqa: S311
    baseline = simulate(
        graph,
        model,
        seeds=fixed,
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        steps=steps,
        runs=runs,
        seed=seed,
    )
    treated = simulate(
        graph,
        model,
        seeds=fixed,
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        steps=steps,
        runs=runs,
        seed=seed,
        immune=immunised,
    )
    degrees = dict(graph.degree())
    picked = [float(degrees[node]) for node in immunised]
    every = np.array([float(d) for d in degrees.values()], dtype=np.float64)
    mean_degree = float(every.mean()) if every.size else 0.0
    delay = sum(
        before.ever - after.ever
        for before, after in zip(baseline.steps, treated.steps, strict=True)
    )
    return Intervention(
        strategy=_strategy_or_refuse(strategy),
        share=share,
        requested=max(1, round(share * len(order))),
        immunised=immunised,
        immunised_mean_degree=float(np.mean(picked)) if picked else 0.0,
        network_mean_degree=mean_degree,
        expected_neighbour_degree=(
            float((every**2).mean() / mean_degree) if mean_degree > 0 else 0.0
        ),
        seeds_immunised=len(set(fixed) & set(immunised)),
        baseline=baseline,
        treated=treated,
        reduction=baseline.final_size - treated.final_size,
        delay=float(delay),
    )


# ----------------------------------------------------------------------------- §21.4 control


def driver_nodes(graph: nx.Graph) -> DriverNodes:
    """The minimum driver set by maximum matching (§21.4; Liu, Slotine and Barabási 2011).

    §21.4 asks "the smallest possible subset of nodes we have to manipulate so that they will
    influence the other nodes to switch to the state we want them to assume" and then declines
    to derive it. The structural-controllability answer it points at is a matching problem: split
    every node into a source copy and a target copy, join source ``u`` to target ``v`` for every
    edge ``u -> v``, take a maximum matching ``M*`` of that bipartite graph, and the unmatched
    targets are the nodes that have to be driven from outside -- ``N_D = max(1, N - |M*|)``.

    ``share = N_D/N`` is the number to read: the paper's finding is that sparse, heavy-tailed
    networks need many drivers, and that "driver nodes tend not to be hubs". The count is unique;
    the *set* is not, which is figure 21.11's point that some nodes "could or could not" be
    chosen -- one valid set is returned and the report says so.

    An **undirected** network is read as though every edge ran both ways. Controllability is
    defined on directed systems, and a co-mention edge states no direction, so this is the
    charitable reading (influence can travel either way) rather than the book's case; the report
    prints which it was. ``N_D`` is at least 1 even on a fully matched network: something has to
    be driven.
    """
    order = node_order(graph)
    total = len(order)
    if total == 0:
        return DriverNodes(count=0, share=0.0, nodes=(), matching=0, directed=False, total=0)
    bipartite = nx.Graph()
    sources = [("out", node) for node in order]
    bipartite.add_nodes_from(sources)
    bipartite.add_nodes_from(("in", node) for node in order)
    for u, v in graph.edges():
        bipartite.add_edge(("out", u), ("in", v))
        if not graph.is_directed():
            bipartite.add_edge(("out", v), ("in", u))
    matched: dict[Any, Any] = (
        nx.algorithms.bipartite.hopcroft_karp_matching(bipartite, top_nodes=sources)
        if bipartite.number_of_edges()
        else {}
    )
    unmatched = tuple(node for node in order if ("in", node) not in matched)
    size = len(matched) // 2
    count = max(1, total - size)
    return DriverNodes(
        count=count,
        share=count / total,
        nodes=unmatched if unmatched else (order[0],),
        matching=size,
        directed=bool(graph.is_directed()),
        total=total,
    )


# ----------------------------------------------------------------------------- the report


def build_spread_report(
    graph: nx.Graph,
    *,
    persona_id: str = "",
    network: str = "",
    model: str = "si",
    seeds: Sequence[Node] | int | None = None,
    beta: float = DEFAULT_BETA,
    mu: float = DEFAULT_MU,
    threshold: float = DEFAULT_THRESHOLD,
    attempts: int = DEFAULT_ATTEMPTS,
    steps: int = DEFAULT_STEPS,
    runs: int = DEFAULT_RUNS,
    share: float | None = None,
    strategy: str = "random",
    seed: int | None = None,
) -> SpreadReport:
    """One simulation, its thresholds, an optional intervention, and the driver-node count.

    ``share`` turns §21.3's comparison on: the same outbreak is then run twice, with and without
    ``strategy``'s immunisation. The driver-node section (§21.4) is always computed -- it depends
    on the network alone, not on the model -- and the mean-field curve is computed for the three
    simple models only, because chapter 21 gives none for the other two.
    """
    name = _model_or_refuse(model)
    result = simulate(
        graph,
        name,
        seeds=seeds,
        beta=beta,
        mu=mu,
        threshold=threshold,
        attempts=attempts,
        steps=steps,
        runs=runs,
        seed=seed,
    )
    limits = epidemic_threshold(graph)
    ratio = reproduction(beta, mu, graph) if mu > 0 and name in {"sis", "sir"} else None
    curve = (
        mean_field_curve(
            name,
            beta=beta,
            mu=mu,
            mean_degree=limits.mean_degree,
            i0=result.steps[0].infected,
            steps=steps,
        )
        if name in {"si", "sis", "sir"}
        else None
    )
    intervention = (
        interventions(
            graph,
            share,
            strategy,
            model=name,
            seeds=seeds,
            beta=beta,
            mu=mu,
            threshold=threshold,
            attempts=attempts,
            steps=steps,
            runs=runs,
            seed=seed,
        )
        if share is not None
        else None
    )
    pieces = _components(graph)
    return SpreadReport(
        persona_id=persona_id,
        network=network,
        frame=str(graph.graph.get("frame", "")),
        result=result,
        threshold=limits,
        reproduction=ratio,
        mean_field=curve,
        components=len(pieces),
        largest_component=max(pieces) if pieces else 0,
        intervention=intervention,
        drivers=driver_nodes(graph),
    )


def _components(graph: nx.Graph) -> list[int]:
    """Component sizes, weakly for a directed network: a spread stops at the seed's component."""
    finder = nx.weakly_connected_components if graph.is_directed() else nx.connected_components
    return [len(piece) for piece in finder(graph)]


def _curve_rows(result: SpreadResult, mean_field: Sequence[float] | None) -> list[list[str]]:
    """A readable sample of the trajectory: first step, last step and evenly spaced steps."""
    total = len(result.steps) - 1
    if total < CURVE_ROWS:
        indices = list(range(total + 1))
    else:
        indices = sorted({round(i * total / (CURVE_ROWS - 1)) for i in range(CURVE_ROWS)})
    rows = []
    for index in indices:
        step = result.steps[index]
        row = [
            str(step.step),
            f"{step.susceptible:.3f}",
            f"{step.infected:.3f}",
            f"{step.infected_low:.3f}-{step.infected_high:.3f}",
            f"{step.removed:.3f}",
            f"{step.ever:.3f}",
        ]
        if mean_field is not None:
            row.append(f"{mean_field[index]:.3f}")
        rows.append(row)
    return rows


def _table(
    header: Sequence[str], rows: Sequence[Sequence[str]], *, prose_last: bool = False
) -> list[str]:
    """A markdown table, numbers right-aligned after the first column. ``prose_last`` leaves the
    final column left-aligned, for the ones that hold a sentence rather than a number."""
    columns = len(header) - (2 if prose_last else 1)
    align = "| --- |" + " ---: |" * columns + (" --- |" if prose_last else "")
    return [
        "| " + " | ".join(header) + " |",
        align,
        *["| " + " | ".join(row) + " |" for row in rows],
        "",
    ]


def _threshold_section(report: SpreadReport) -> list[str]:
    """§20.2's three thresholds, with where this run's β/µ falls against them."""
    limits = report.threshold
    lines = [
        "## The epidemic threshold (§20.2)",
        "",
        "The value `lambda = beta/mu` has to exceed for a spread to persist rather than die out. "
        "Three readings of the same question, on the same network: the first is exact for this "
        "network, the other two are what §20.2 states for two *models* of a network.",
        "",
    ]
    if limits.spectral is None or limits.heterogeneous is None:
        lines += [limits.reading, ""]
        return lines
    lines += _table(
        ["threshold", "value", "what it assumes"],
        [
            [
                "spectral, `1/lambda_1`",
                f"{limits.spectral:.4f}",
                f"nothing beyond this network; lambda_1 = {limits.leading_eigenvalue:.4f} of the "
                "binary adjacency matrix (Wang et al. 2003, cited at p. 297 n. 18)",
            ],
            [
                "homogeneous, `1/(k+1)`",
                f"{limits.homogeneous:.4f}",
                f"a Gn,p graph with the same mean degree, k = {limits.mean_degree:.3f} (p. 294)",
            ],
            [
                "heterogeneous, `k/k^2`",
                f"{limits.heterogeneous:.4f}",
                "a preferential-attachment graph with the same first two degree moments, "
                f"k^2 = {limits.mean_square_degree:.3f} (p. 294)",
            ],
        ],
        prose_last=True,
    )
    lines += [limits.reading, ""]
    ratio = report.reproduction
    if ratio is None:
        lines += [
            f"This run has no `beta/mu` to compare: the `{report.result.model}` model has no "
            "recovery rate, so there is no lambda to put against the thresholds. §20.1's SI "
            "always saturates whatever beta is, and chapter 21's triggers change the rule for "
            "catching it rather than the rule for losing it.",
            "",
        ]
        return lines
    verdict = (
        "**endemic**: above the spectral threshold, so §20.2 expects the spread to settle at a "
        "non-zero share rather than disappear"
        if ratio.endemic
        else "**dies out**: below the spectral threshold, so §20.2 expects the spread to "
        "disappear, whatever this run's first few steps look like"
    )
    r0 = "undefined (no edges)" if ratio.r0 is None else f"{ratio.r0:.3f}"
    lines += [
        f"This run: `lambda = beta/mu = {ratio.beta:g}/{ratio.mu:g} = {ratio.ratio:.4f}`, which "
        f"is {ratio.ratio / (report.threshold.spectral or 1.0):.2f}x the spectral threshold. "
        f"{verdict}.",
        "",
        f"Beside it, and **not** from chapter 20: R0 = `beta (k^2 - k) / (mu k)` = {r0}, the "
        "expected secondary infections from one infected node in an otherwise susceptible "
        "network. The chapter defines lambda and the thresholds, never R0; it is printed because "
        "it is the number a reader of any other epidemic text will look for.",
        "",
    ]
    return lines


def _curve_section(report: SpreadReport) -> list[str]:
    """The simulated trajectory, its band, and the mean-field line beside it."""
    result = report.result
    chapter = {
        "si": "§20.1",
        "sis": "§20.2",
        "sir": "§20.3",
        "threshold": "§21.1",
        "limited": "§21.2",
    }[result.model]
    seeded = (
        ", ".join(str(node) for node in result.seeds[:6])
        + (f", +{len(result.seeds) - 6} more" if len(result.seeds) > 6 else "")
        if result.seeds
        else f"{result.seed_count} node(s) drawn uniformly at random in each run (§20.5)"
    )
    parameters = ", ".join(f"`{k} = {v:g}`" for k, v in result.parameters.items())
    header = ["step", "S", "I", f"I {SPREAD_BAND:.0%} band", "R", "ever infected"]
    if report.mean_field is not None:
        header.append("mean field")
    lines = [
        f"## The curve ({chapter})",
        "",
        f"**Model.** `{result.model}` -- {MODEL_NOTES[result.model]}",
        "",
        f"**Parameters.** {parameters}. **Seeds.** {seeded}. **Runs.** {result.runs} "
        f"simulations of {len(result.steps) - 1} steps"
        + (f", seeded with {result.seed}." if result.seed is not None else ", unseeded."),
        "",
        "Shares of all nodes, averaged over the runs. `ever infected` is the share infected at "
        "least once, which is the only readable count for SIS and equals `I + R` elsewhere.",
        "",
    ]
    lines += _table(header, _curve_rows(result, report.mean_field))
    if report.mean_field is not None:
        lines += [
            "The `mean field` column is the book's own equation under homogeneous mixing "
            "(p. 289 n. 7), which knows only the mean degree and therefore cannot see a hub. "
            "§20.1's whole argument is that the distance between those two columns is what the "
            "network's degree distribution does to a spread.",
            "",
        ]
    peak = "never (nothing was infected)" if result.peak_step is None else f"{result.peak_step:.1f}"
    lines += [
        f"**Final size.** {result.final_size:.1%} of the network was infected at some point "
        f"(band {result.final_low:.1%}-{result.final_high:.1%} over {result.runs} runs). "
        f"**Peak.** {result.peak_infected:.1%} infected at once, on average at step {peak}. "
        f"**Last quarter of the run.** {result.endemic:.1%} infected on average; "
        f"{result.died_out:.0%} of runs ended with nobody infected.",
        "",
    ]
    if report.components > 1:
        share = report.largest_component / max(result.nodes, 1)
        lines += [
            f"**The ceiling is not 100%.** This network is in {report.components:,} pieces and "
            f"the largest holds {report.largest_component:,} nodes ({share:.1%}). A spread never "
            "leaves the component it started in, so a final size below 100% here is a fact about "
            "the network's fragmentation before it is a fact about the model.",
            "",
        ]
    return lines


def _intervention_section(intervention: Intervention) -> list[str]:
    """§21.3's two success criteria, with the strategy's own mean degree beside them."""
    seeds = (
        f" {intervention.seeds_immunised} of the seed nodes were themselves immunised, which "
        "stops the outbreak before it starts and flatters the comparison."
        if intervention.seeds_immunised
        else ""
    )
    # Both rows are "without minus with", which is §21.3's own arithmetic: "subtract the
    # predicted infected share without immunization with the one with immunization. The higher
    # the difference the better."
    peak_change = intervention.treated.peak_infected - intervention.baseline.peak_infected
    return [
        "## Immunisation (§21.3)",
        "",
        f"**Strategy.** `{intervention.strategy}` -- {STRATEGY_NOTES[intervention.strategy]}",
        "",
        f"**n.** {len(intervention.immunised):,} nodes immunised of "
        f"{intervention.requested:,} asked for ({intervention.share:.0%} of the network), mean "
        f"degree {intervention.immunised_mean_degree:.2f} against the network's "
        f"{intervention.network_mean_degree:.2f}, and against the "
        f"{intervention.expected_neighbour_degree:.2f} a *randomly chosen neighbour* has in "
        "expectation (`<k^2>/<k>`, the exact form of the friendship paradox, §31.2). That last "
        "number is what §21.3's acquaintance strategy is buying, and it is the comparison to "
        "read this table's first line against. The same seeds and the same random seed run both "
        f"arms, so the difference is the immunisation and not the draw.{seeds}",
        "",
        *_table(
            ["criterion", "without", "with", "without - with"],
            [
                [
                    "final size (figure 21.9)",
                    f"{intervention.baseline.final_size:.1%}",
                    f"{intervention.treated.final_size:.1%}",
                    f"{intervention.reduction:+.1%}",
                ],
                [
                    "peak infected at once",
                    f"{intervention.baseline.peak_infected:.1%}",
                    f"{intervention.treated.peak_infected:.1%}",
                    f"{-peak_change:+.1%}",
                ],
            ],
        ),
        f"**Delay (figure 21.10).** {intervention.delay:.2f} node-shares x steps of area between "
        "the two `ever infected` curves. Positive means the immunised run reached the same place "
        "later; §21.3 counts that as a success on its own, because the time buys a vaccine -- or, "
        "read the other way round for viral marketing, because arriving sooner is the whole point.",
        "",
    ]


def _driver_section(drivers: DriverNodes) -> list[str]:
    """§21.4's driver-node count, with the caveat about what it is not."""
    read = (
        "every edge read as running both ways, because a co-mention states no direction. "
        "Controllability is defined on directed systems, so this is a charitable reading of an "
        "undirected network rather than the book's own case"
        if not drivers.directed
        else "the edges read as they point"
    )
    example = ", ".join(str(node) for node in drivers.nodes[:8])
    return [
        "## Driver nodes (§21.4)",
        "",
        f"**n.** {drivers.count:,} driver nodes of {drivers.total:,} "
        f"({drivers.share:.1%}), from a maximum matching of {drivers.matching:,} edges, with "
        f"{read}.",
        "",
        "`N_D = max(1, N - |M*|)` (Liu, Slotine and Barabási, *Controllability of complex "
        "networks*, Nature 473:167, 2011 -- §21.4 states the result and says the derivation is "
        "beyond the book). The share is the reading: near 0 the whole network can be steered "
        "from a handful of nodes, near 1 almost every node has to be steered by hand. The "
        "chapter's counter-intuitive finding is that these are usually **not** the hubs, so this "
        "is not another centrality: it is a count of how much of the network is out of reach of "
        "any small set of levers.",
        "",
        f"One valid driver set (there are others of the same size; figure 21.11 makes exactly "
        f"this point): {example}"
        + (f", +{len(drivers.nodes) - 8} more." if len(drivers.nodes) > 8 else "."),
        "",
        "**Null model.** None. A matching is a deterministic function of the network, so there "
        "is no p-value here; what would carry one is a comparison against a rewired network of "
        "the same degree sequence, which `sna analyze --null configuration` is for.",
        "",
    ]


def render_spread(report: SpreadReport) -> str:
    """The spread report as markdown: the what-if sentence first, then the numbers."""
    result = report.result
    lines = [
        f"# Spreading: {report.persona_id or 'network'} / {report.network or 'graph'} "
        f"— {result.model}",
        "",
        "**Implements.** *The Atlas for the Aspiring Network Scientist*, chapter 20 (epidemics: "
        "§20.1 SI, §20.2 SIS and the epidemic threshold, §20.3 SIR) and chapter 21 (complex "
        "contagion: §21.1 triggers, §21.2 limited infection chances, §21.3 interventions, §21.4 "
        "controllability).",
        "",
        f"**Sampling frame.** {report.frame or 'Not recorded.'}",
        "",
        f"**What a spread means here.** {SPREAD_FRAME}",
        "",
        f"**n.** {result.nodes:,} nodes and {result.edges:,} edges, mean degree "
        f"{report.threshold.mean_degree:.3f}, mean squared degree "
        f"{report.threshold.mean_square_degree:.3f}, in {report.components:,} component(s) of "
        f"which the largest holds {report.largest_component:,}. Edge weights are ignored: an "
        "edge is one contact.",
        "",
        f"**Null model.** None, and none is possible: a simulation is not a measurement that "
        f"could have come out differently by chance. The bands are {result.runs} seeded runs of "
        "the same process on the same network, so they report the stochasticity of the process "
        "and nothing about the network's uncertainty. The analytic comparison -- the thing that "
        "plays the role a null would -- is the epidemic threshold below: it says what §20.2 "
        "expects before any run happens.",
        "",
    ]
    lines += _threshold_section(report)
    lines += _curve_section(report)
    if report.intervention is not None:
        lines += _intervention_section(report.intervention)
    lines += _driver_section(report.drivers)
    return "\n".join(lines)


def spread_payload(report: SpreadReport) -> dict[str, Any]:
    """The same report as plain JSON-able data."""
    result = report.result
    limits = report.threshold
    payload: dict[str, Any] = {
        "persona_id": report.persona_id,
        "network": report.network,
        "frame": report.frame,
        "implements": "Atlas ch. 20 (§20.1-20.3) and ch. 21 (§21.1-21.4)",
        "what_a_spread_means": SPREAD_FRAME,
        "null_model": (
            f"none; {result.runs} seeded simulations of the same process, banded at "
            f"{SPREAD_BAND:.0%}, with the epidemic threshold as the analytic comparison"
        ),
        "nodes": result.nodes,
        "edges": result.edges,
        "components": report.components,
        "largest_component": report.largest_component,
        "model": result.model,
        "model_note": MODEL_NOTES[result.model],
        "parameters": result.parameters,
        "seeds": [str(node) for node in result.seeds],
        "seed_count": result.seed_count,
        "runs": result.runs,
        "seed": result.seed,
        "threshold": {
            "spectral": limits.spectral,
            "homogeneous": limits.homogeneous,
            "heterogeneous": limits.heterogeneous,
            "leading_eigenvalue": limits.leading_eigenvalue,
            "mean_degree": limits.mean_degree,
            "mean_square_degree": limits.mean_square_degree,
            "reading": limits.reading,
        },
        "curve": [
            {
                "step": step.step,
                "susceptible": step.susceptible,
                "infected": step.infected,
                "removed": step.removed,
                "ever": step.ever,
                "infected_low": step.infected_low,
                "infected_high": step.infected_high,
            }
            for step in result.steps
        ],
        "mean_field": list(report.mean_field) if report.mean_field is not None else None,
        "final_size": result.final_size,
        "final_low": result.final_low,
        "final_high": result.final_high,
        "peak_infected": result.peak_infected,
        "peak_step": result.peak_step,
        "endemic": result.endemic,
        "died_out": result.died_out,
        "drivers": {
            "count": report.drivers.count,
            "share": report.drivers.share,
            "matching": report.drivers.matching,
            "directed": report.drivers.directed,
            "nodes": [str(node) for node in report.drivers.nodes],
        },
    }
    if report.reproduction is not None:
        ratio = report.reproduction
        payload["reproduction"] = {
            "ratio": ratio.ratio,
            "r0": ratio.r0,
            "r0_note": "not from chapter 20; the heterogeneous mean-field convention",
            "over_spectral": ratio.spectral,
            "over_homogeneous": ratio.homogeneous,
            "over_heterogeneous": ratio.heterogeneous,
            "endemic": ratio.endemic,
        }
    if report.intervention is not None:
        intervention = report.intervention
        payload["intervention"] = {
            "strategy": intervention.strategy,
            "note": STRATEGY_NOTES[intervention.strategy],
            "share": intervention.share,
            "requested": intervention.requested,
            "immunised": [str(node) for node in intervention.immunised],
            "immunised_mean_degree": intervention.immunised_mean_degree,
            "network_mean_degree": intervention.network_mean_degree,
            "expected_neighbour_degree": intervention.expected_neighbour_degree,
            "seeds_immunised": intervention.seeds_immunised,
            "final_size_without": intervention.baseline.final_size,
            "final_size_with": intervention.treated.final_size,
            "reduction": intervention.reduction,
            "delay": intervention.delay,
        }
    return payload
