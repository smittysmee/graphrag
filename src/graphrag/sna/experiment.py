"""Designing a link-prediction experiment, and refusing to print a prediction without one.

Chapter 25 of *The Atlas for the Aspiring Network Scientist* is not about predicting links. It
is about the fact that *"evaluating the performance of an oracle in getting things right is
harder than it might seem. There are surprising ways to get it wrong"* (p. 356). This module
holds the experiment, so that every command in this package that ranks pairs of nodes reports
through it and no score is ever printed on its own.

Two halves, the chapter's own.

**§25.1, the split.** *"You can never ever ever use the data that trained your model to test
it"* (p. 357), and the consequence peculiar to link prediction: *"you cannot claim to have
predicted a link that was already in your data. You have to focus only on those pairs of nodes
that were not connected in the training set."* :class:`Holdout` enforces exactly that -- a
positive that is still an edge of the training graph is a construction error and is refused
rather than scored. The chapter offers two ways to build the split (Figure 25.2, p. 358):

``temporal`` (:func:`temporal_holdout`)
    *"If you have temporal information on your edges you can use earlier edges to predict the
    later ones."* Train is what the corpus had joined before a date, test is what it first
    joined on or after it. Nothing is deleted, so the structure the predictor reads is the
    structure that actually existed on that date. This is the honest split and the one to use
    whenever the documents are dated; ``sna predict-eval --temporal`` is it.

``random`` (:func:`holdout`, :func:`kfold`)
    *"If you don't have the luxury of time data, you have to do k-fold cross validation: divide
    your dataset in a train and test set (say 90% of edges in train and 10% in test) and then
    perform multiple runs of train-test by rotating the test set so that each edge appears in it
    at least once."* The caveat the chapter does not spell out but that every frame printed here
    does: a deleted edge is missing from the evidence as well as from the answer key, so a
    common-neighbour scorer is asked to rediscover a link whose supporting triangle was removed
    along with it. On a corpus network, where an edge *is* a passage, that is worse than usual --
    removing the edge removes the sentence.

**Negative sampling.** Real networks are sparse (§12.1), so the pair space dwarfs the edge set:
the book's Internet backbone has 18,478,781,646 possible edges and 609,066 real ones, which
makes computing every score *"an unreasonable burden, both for computation time and memory
storage"* (p. 358) and makes the always-negative predictor 99.999% accurate and useless. *"The
usual fix for this problem is building your test set in a balanced way. Rather than asking about
all possible new edges, you create a smaller test set. Half of the edges in the test set is an
actual new edge, and then you sample an equal number of non-edges"* (p. 359). ``negatives=1``
is that ratio, and it is a *parameter* here because the balanced set is a convenience, not a
truth: every report prints the imbalance of the full pair space next to the balanced n, so a
reader can see what was sampled away.

**§25.2, the measurement.** The four counts (TP, FP, TN, FN) as :class:`ConfusionMatrix`, the
rates built on them, the ROC curve and its area, the precision-recall curve and its area, and
precision@k -- *"defining n as the number of predictions we want to make"* (p. 363). The
chapter's warnings travel with the numbers rather than in a footnote: AUC is *"unaffected if you
sample your test set randomly"* but is read against a floor of 0.5, an AUC of 1 *"you'll never
see and, if you do, it means you did something wrong"*, accuracy *"hides the difference between
type I and type II errors ... and thus it should be handled with care"*, and a confusion matrix
on an unbalanced test set is worthless because *"the vast majority of your observations will end
up in the true negative cell, obliterating all the rest"* (p. 361).

**What the book defines and what this adds.** The chapter defines the curves and names their
areas; it does not fix an interpolation, a tie rule or an estimator. Three conventions are this
module's, and each is stated where it is used: the area under the ROC curve is the rank
(Mann-Whitney) statistic with tied scores counted as half a win, which is the trapezoid area
under the step curve :func:`roc` returns and is what makes ties in a degree-product score
harmless; the area under the precision-recall curve is the step sum ``Σ (R_i - R_{i-1}) P_i``
rather than a trapezoid, because interpolating precision between thresholds invents pairs;
and precision@k is computed over the *test set*, not over the whole pair space, which is where
it departs from p. 363 and is discussed at :func:`precision_at_k`.

**What is not here.** No predictor. :func:`preferential_attachment` is in this module only
because §25.2 asks for a baseline to compare against and a report without one is not a report;
the predictors themselves are chapter 23's. Validation sets are named in §25.1 and pointed
straight back at chapter 4 (*"there is nothing special about the validation set for link
prediction"*), so there is none here either: with two-parameter baselines there is nothing to
overfit, and a predictor that needs one brings it with itself.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from graphrag.graph.store import GraphStore
from graphrag.sna.layers import dynamic_edges

__all__ = [
    "DEFAULT_FOLDS",
    "DEFAULT_K",
    "DEFAULT_NEGATIVES",
    "DEFAULT_SHARE",
    "HOLDOUT_METHODS",
    "MAX_ENUMERATED_PAIRS",
    "RANDOM_AUC",
    "ConfusionMatrix",
    "EvaluationReport",
    "Holdout",
    "Pair",
    "PrecisionRecallPoint",
    "RocPoint",
    "ScoreTable",
    "auc",
    "average_precision",
    "confusion",
    "evaluate_predictor",
    "experiment_payload",
    "holdout",
    "kfold",
    "precision_at_k",
    "precision_recall",
    "prediction_power",
    "preferential_attachment",
    "random_scores",
    "render_experiment",
    "roc",
    "temporal_holdout",
]

#: A candidate link, as a pair of node ids. Sorted for an undirected network, because ``(u, v)``
#: and ``(v, u)`` are one link there; left in the stated order for a directed one, where they
#: are two.
Pair = tuple[str, str]

#: How the train/test split was made. The book's two (Figure 25.2, p. 358), plus the fold a
#: k-fold rotation produces, which is a random split that knows which rotation it came from.
HOLDOUT_METHODS: tuple[str, ...] = ("random", "temporal", "fold")

#: The book's own worked share: *"say 90% of edges in train and 10% in test"* (p. 358).
DEFAULT_SHARE = 0.1

#: The book's own number of blocks: *"divide the data in ten blocks and rotate one block as test
#: set using the other nine as training"* (§25.3).
DEFAULT_FOLDS = 10

#: Negatives per positive. 1.0 is §25.1's balanced test set -- *"half of the edges in the test
#: set is an actual new edge, and then you sample an equal number of non-edges"* (p. 359).
DEFAULT_NEGATIVES = 1.0

#: How many predictions precision@k scores by default. The book's example is Precision@100
#: (p. 363); ten is the length of a list a person reads, and every report prints the k it used.
DEFAULT_K = 10

#: The AUC of the random guess: *"that's the area under the 45 degree line"* (p. 363). Printed
#: as the null model of every evaluation, and measured rather than assumed by
#: :func:`random_scores`.
RANDOM_AUC = 0.5

#: Above this many possible pairs, nothing in this module will enumerate the pair space; it
#: samples or refuses instead. p. 358's arithmetic is the reason: the Internet backbone's 18.4
#: billion candidate pairs are *"an unreasonable burden, both for computation time and memory
#: storage"*.
MAX_ENUMERATED_PAIRS = 2_000_000

#: Rejection draws allowed per negative asked for, before the sampler gives up on rejection and
#: enumerates the non-edges instead. Twenty is the expected cost of one draw when a twentieth of
#: the pair space is free, which is far denser than any network here.
_REJECTION_TRIES = 200

RANDOM_FRAME = (
    "A random edge holdout (Atlas §25.1): {removed:,} of {edges:,} edge(s) ({share:.1%}) were "
    "deleted at seed {seed} and are the positives; the training graph is what is left, with "
    "every node kept so that the pair space does not move. The chapter's fundamental tenet is "
    "enforced -- no positive is an edge of the training graph -- but its cost is not avoidable: "
    "a deleted edge is missing from the evidence as well as from the answer key, so a scorer "
    "that reads common neighbours is asked to rediscover a link whose supporting structure was "
    "removed with it. On a corpus network an edge is a passage, so this deletes the sentence "
    "and then asks for it back. {negatives:,} negative(s) were sampled uniformly from the "
    "{available:,} pair(s) this network never joined."
)
CONNECTED_FRAME = (
    " {skipped:,} candidate edge(s) were put back because deleting them would have cut their "
    "endpoints apart, so the achieved share is below the one asked for: keeping the training "
    "graph in one piece is a choice about the experiment, not a property of the corpus."
)
TEMPORAL_FRAME = (
    "A temporal holdout at {split_at} (Atlas §25.1, the split the book prefers -- 'if you have "
    "temporal information on your edges you can use earlier edges to predict the later ones'): "
    "the training graph is the {train_edges:,} pair(s) the {network} network had joined before "
    "that date, weighted by how often they were joined, and the positives are the "
    "{positives:,} pair(s) first joined on or after it. Nothing was deleted, so the structure "
    "the predictor reads is the structure the corpus had on {split_at}. {unseen:,} later "
    "pair(s) were dropped because at least one endpoint had not appeared before the split: a "
    "predictor that reads topology cannot score a node it has never seen, and counting those "
    "as misses would score it on the corpus's growth rather than on its own structure. "
    "{negatives:,} negative(s) were sampled uniformly from the {available:,} pair(s) that were "
    "never joined at any date. {undated:,} document(s) carry no dated passage and are on "
    "neither side of the split."
)
FOLD_FRAME = (
    "Fold {index} of {folds} in a {folds}-fold cross validation (Atlas §25.1): the edges were "
    "cut into {folds} blocks at seed {seed} and this block is the test set, the other "
    "{others} the training graph. Every edge is a positive in exactly one fold. The caveat of "
    "the random holdout applies to each fold -- the block's edges are missing from the evidence "
    "as well as from the answer key. {negatives:,} negative(s) were sampled uniformly from the "
    "{available:,} pair(s) this network never joined."
)


# ------------------------------------------------------------------- the pair space (§25.1)


def _directed(graph: nx.Graph) -> bool:
    return bool(graph.is_directed())


def _pair(graph: nx.Graph, u: str, v: str) -> Pair:
    """``(u, v)`` in the canonical order for this network's kind."""
    if _directed(graph):
        return (str(u), str(v))
    return (str(u), str(v)) if str(u) <= str(v) else (str(v), str(u))


def _require_string_nodes(graph: nx.Graph) -> None:
    """Refuse a graph whose node ids are not strings, rather than quietly mismatching them.

    A :data:`Pair` is two strings, and every network :mod:`graphrag.sna.export` builds has string
    ids. A graph from elsewhere -- ``nx.karate_club_graph()`` numbers its members -- would have
    its pairs stringified and then fail to match its own edges, which is a wrong answer rather
    than an error, so it is an error here instead.
    """
    wrong = [node for node in graph.nodes if not isinstance(node, str)]
    if wrong:
        msg = (
            f"node ids must be strings, and {len(wrong):,} of {graph.number_of_nodes():,} are "
            f"not (e.g. {wrong[0]!r}): relabel with nx.relabel_nodes(graph, str) first"
        )
        raise ValueError(msg)


def _modes(graph: nx.Graph) -> dict[str, str]:
    """The two-mode labels, when every node carries one and there are exactly two of them.

    A pair of speakers is not a candidate link in the ``speakers-entities`` network: no passage
    could ever produce it, so drawing it as a negative would be sampling from a space the corpus
    cannot reach and would flatter every scorer. Any other network returns an empty mapping and
    the pair space is the plain one.
    """
    labels = {
        str(node): str(data["mode"]) for node, data in graph.nodes(data=True) if "mode" in data
    }
    if len(labels) != graph.number_of_nodes() or len(set(labels.values())) != 2:
        return {}
    return labels


def _possible_pairs(graph: nx.Graph, modes: Mapping[str, str]) -> int:
    """How many pairs this network could conceivably join, the ``|V|(|V|-1)/2`` of p. 358."""
    nodes = int(graph.number_of_nodes())
    if modes:
        left, right = (count for _, count in sorted(Counter(modes.values()).items()))
        return left * right
    if _directed(graph):
        return nodes * (nodes - 1)
    return nodes * (nodes - 1) // 2


def _allowed(pair: Pair, modes: Mapping[str, str]) -> bool:
    u, v = pair
    if u == v:
        return False
    return not modes or modes.get(u) != modes.get(v)


def _enumerate_pairs(graph: nx.Graph, modes: Mapping[str, str]) -> Iterator[Pair]:
    """Every pair this network could join, edges included, in node order."""
    nodes = sorted(str(node) for node in graph.nodes)
    for index, u in enumerate(nodes):
        others = nodes if _directed(graph) else nodes[index + 1 :]
        for v in others:
            pair = _pair(graph, u, v)
            if _allowed(pair, modes):
                yield pair


def _non_edges(graph: nx.Graph, modes: Mapping[str, str]) -> Iterator[Pair]:
    for pair in _enumerate_pairs(graph, modes):
        if not graph.has_edge(*pair):
            yield pair


def _sample_non_edges(
    graph: nx.Graph, count: int, rng: random.Random, *, exclude: frozenset[Pair] = frozenset()
) -> tuple[list[Pair], int]:
    """``count`` pairs drawn uniformly from the pairs ``graph`` does not join (§25.1).

    Returns the sample and how many pairs it was drawn from, which is the number the report
    prints as the size of the negative class before balancing. ``exclude`` holds pairs that are
    real edges elsewhere -- the positives of a holdout, which are absent from the training graph
    precisely because they are the answer -- and they are neither drawn nor counted as available.

    Rejection sampling while the space is mostly free, which it is for every sparse network, and
    a full enumeration when it is not and the space is small enough to hold. Above
    :data:`MAX_ENUMERATED_PAIRS` it refuses instead of allocating, for p. 358's reason.
    """
    modes = _modes(graph)
    total = _possible_pairs(graph, modes)
    blocked = {pair for pair in exclude if not graph.has_edge(*pair)}
    available = total - graph.number_of_edges() - len(blocked)
    if count > available:
        msg = (
            f"cannot sample {count:,} negative pair(s): this network has only {available:,} "
            "pair(s) it never joined. Ask for fewer negatives, or a smaller share."
        )
        raise ValueError(msg)
    if count == 0 or available == 0:
        return [], available

    nodes = sorted(str(node) for node in graph.nodes)
    by_mode: dict[str, list[str]] = {}
    for node in nodes:
        by_mode.setdefault(modes.get(node, ""), []).append(node)
    drawn: set[Pair] = set()
    if available / total >= 1 / 20:
        for _ in range(_REJECTION_TRIES * count):
            if len(drawn) == count:
                break
            if modes:
                left, right = sorted(by_mode)
                u, v = rng.choice(by_mode[left]), rng.choice(by_mode[right])
            else:
                u, v = rng.sample(nodes, 2)
            pair = _pair(graph, u, v)
            if graph.has_edge(*pair) or pair in blocked:
                continue
            drawn.add(pair)
    if len(drawn) == count:
        return sorted(drawn), available
    if total > MAX_ENUMERATED_PAIRS:
        msg = (
            f"this network is too dense to draw {count:,} negative pair(s) by rejection and too "
            f"large to enumerate ({total:,} possible pairs, over the {MAX_ENUMERATED_PAIRS:,} "
            "this module will hold: §25.1 calls that 'an unreasonable burden, both for "
            "computation time and memory storage'). Narrow the network first."
        )
        raise ValueError(msg)
    candidates = [pair for pair in _non_edges(graph, modes) if pair not in blocked]
    return sorted(rng.sample(candidates, count)), available


# ----------------------------------------------------------------------- the split (§25.1)


@dataclass(frozen=True)
class Holdout:
    """One train/test split of a network, built so that §25.1's tenet cannot be broken.

    ``train`` is the graph a predictor may read. ``positives`` are the pairs that are really
    joined and are absent from it -- the answer key. ``negatives`` are pairs sampled from those
    the network never joined, in the ratio §25.1's balanced test set asks for. The three are
    disjoint by construction and the constructor refuses a split where they are not, because a
    positive that is still an edge of ``train`` is the one mistake the chapter says will *"grossly
    overestimate your actual performance in the real world"*.

    ``possible_pairs`` and ``observed_edges`` describe the pair space the test set was cut out
    of, so :attr:`imbalance` can say what the balancing hid. Read that number before reading any
    AUC.
    """

    train: nx.Graph
    positives: tuple[Pair, ...]
    negatives: tuple[Pair, ...]
    method: str = "random"
    seed: int | None = None
    share: float = 0.0
    """The share of the observed edges that ended up as positives, as achieved rather than as
    asked for: ``keep_connected`` and a network with bridges can lower it."""
    requested_share: float = 0.0
    split_at: str = ""
    """The date the temporal split was made at, ISO ``YYYY-MM-DD``; empty for a random one."""
    possible_pairs: int = 0
    observed_edges: int = 0
    """Edges of the network *before* the split: the training graph plus the positives."""
    unseen_positives: int = 0
    """Later pairs dropped because an endpoint is not in the training graph (temporal only)."""
    available_negatives: int = 0
    """How many pairs the negatives were drawn from, before balancing cut them to a ratio."""
    fold: tuple[int, int] = (0, 0)
    """``(index, folds)`` for a fold of a cross validation, ``(0, 0)`` otherwise."""
    frame: str = ""

    def __post_init__(self) -> None:
        positives, negatives = set(self.positives), set(self.negatives)
        if overlap := positives & negatives:
            msg = (
                f"{len(overlap)} pair(s) are both positive and negative, e.g. {sorted(overlap)[0]}"
            )
            raise ValueError(msg)
        for label, pairs in (("positive", positives), ("negative", negatives)):
            still = [pair for pair in sorted(pairs) if self.train.has_edge(*pair)]
            if still:
                msg = (
                    f"{len(still)} {label} pair(s) are still edges of the training graph, e.g. "
                    f"{still[0]}: §25.1 requires the train and test sets to be disjoint, because "
                    "'you cannot claim to have predicted a link that was already in your data'"
                )
                raise ValueError(msg)

    @property
    def pairs(self) -> tuple[Pair, ...]:
        """The test set: the positives followed by the negatives."""
        return (*self.positives, *self.negatives)

    @property
    def labels(self) -> dict[Pair, bool]:
        """The answer key, as ``pair -> did it really appear``."""
        return {**dict.fromkeys(self.positives, True), **dict.fromkeys(self.negatives, False)}

    @property
    def non_edges(self) -> int:
        """Pairs in the whole space that the network never joined: the class that was sampled."""
        return max(self.possible_pairs - self.observed_edges, 0)

    @property
    def imbalance(self) -> float:
        """Non-edges per positive in the *unsampled* space -- p. 359's 18B against 30k.

        Undefined, and returned as ``inf``, when there are no positives. This is the number the
        balanced test set hides and the reason §25.2 warns that a confusion matrix built without
        balancing ends up with everything in the true-negative cell.
        """
        if not self.positives:
            return math.inf
        return self.non_edges / len(self.positives)

    @property
    def section(self) -> str:
        return "§25.1"


def holdout(
    graph: nx.Graph,
    *,
    share: float = DEFAULT_SHARE,
    seed: int | None = None,
    negatives: float = DEFAULT_NEGATIVES,
    keep_connected: bool = False,
) -> Holdout:
    """Delete a random share of the edges and call them the future (§25.1).

    The split of Figure 25.2's right-hand branch, for a network whose edges carry no date:
    *"divide your dataset in a train and test set (say 90% of edges in train and 10% in test)"*.
    ``share`` is that 10%, ``negatives`` is how many sampled non-edges to put in the test set per
    positive (1.0 is the book's balanced set), and ``seed`` fixes both draws.

    Every node stays in the training graph even when all its edges left, so the pair space a
    scorer is judged on is the same one before and after. That is a convention, not the book's:
    it keeps the imbalance figure comparable across splits of the same network.

    ``keep_connected`` refuses to delete an edge whose endpoints would then be in different
    components. It is off by default because switching it on biases the test set towards the
    redundant edges -- the ones inside triangles, which are exactly the ones a common-neighbour
    scorer finds easiest -- and the frame says so when it is used.

    What the number cannot be, whatever comes out: a measurement of how well this predictor will
    do on the corpus's future. The removed edges are missing from the structure the predictor
    reads, so the experiment scores it on a network that never existed. :func:`temporal_holdout`
    is the one that does not have this problem.
    """
    _require_string_nodes(graph)
    if graph.number_of_edges() == 0:
        msg = "cannot hold out edges from a network that has none"
        raise ValueError(msg)
    if not 0 < share < 1:
        msg = f"share must be strictly between 0 and 1, got {share}"
        raise ValueError(msg)
    if negatives < 0:
        msg = f"negatives must not be negative, got {negatives}"
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311 -- an experiment, not a secret
    edges = [_pair(graph, u, v) for u, v in graph.edges()]
    edges.sort()
    total = len(edges)
    wanted = max(1, round(share * total))
    if wanted >= total:
        msg = (
            f"a share of {share} would hold out {wanted:,} of {total:,} edge(s) and leave the "
            "training graph with nothing to learn from"
        )
        raise ValueError(msg)

    order = list(edges)
    rng.shuffle(order)
    train = graph.copy()
    held: list[Pair] = []
    skipped = 0
    for pair in order:
        if len(held) == wanted:
            break
        if keep_connected:
            train.remove_edge(*pair)
            if nx.has_path(_undirected(train), *pair):
                held.append(pair)
            else:
                train.add_edge(*pair, **graph.get_edge_data(*pair))
                skipped += 1
            continue
        train.remove_edge(*pair)
        held.append(pair)

    positives = tuple(sorted(held))
    sampled, available = _sample_non_edges(
        train, round(negatives * len(positives)), rng, exclude=frozenset(positives)
    )
    frame = RANDOM_FRAME.format(
        removed=len(positives),
        edges=total,
        share=len(positives) / total,
        seed=seed,
        negatives=len(sampled),
        available=available,
    )
    if skipped:
        frame += CONNECTED_FRAME.format(skipped=skipped)
    return Holdout(
        train=train,
        positives=positives,
        negatives=tuple(sampled),
        method="random",
        seed=seed,
        share=len(positives) / total,
        requested_share=share,
        possible_pairs=_possible_pairs(graph, _modes(graph)),
        observed_edges=total,
        available_negatives=available,
        frame=f"{str(graph.graph.get('frame', '')).strip()} {frame}".strip(),
    )


def _undirected(graph: nx.Graph) -> nx.Graph:
    """The connectivity check of ``keep_connected`` ignores direction, as §6.2 flattening does."""
    return graph.to_undirected(as_view=True) if graph.is_directed() else graph


def kfold(
    graph: nx.Graph,
    k: int = DEFAULT_FOLDS,
    *,
    seed: int | None = None,
    negatives: float = DEFAULT_NEGATIVES,
) -> tuple[Holdout, ...]:
    """The k-fold cross validation of §25.1: every edge is a positive in exactly one fold.

    *"Perform multiple runs of train-test by rotating the test set so that each edge appears in
    it at least once"* (p. 358). The folds partition the edge set -- their sizes differ by at
    most one -- so summing their positives gives the network back, which is the property that
    distinguishes a rotation from ``k`` independent random holdouts.

    Each fold's negatives are drawn separately, so a pair the network never joined can appear as
    a negative in more than one fold. That is deliberate: the negative class is a sample of a
    space that does not change between folds, and rotating it too would make ``k`` smaller
    samples of it rather than ``k`` views of the same one.
    """
    _require_string_nodes(graph)
    if k < 2:
        msg = f"k must be at least 2, got {k}"
        raise ValueError(msg)
    if graph.number_of_edges() < k:
        msg = (
            f"cannot cut {graph.number_of_edges():,} edge(s) into {k} folds: a fold with no "
            "positive in it cannot be evaluated"
        )
        raise ValueError(msg)

    rng = random.Random(seed)  # noqa: S311 -- an experiment, not a secret
    edges = sorted(_pair(graph, u, v) for u, v in graph.edges())
    rng.shuffle(edges)
    blocks: list[list[Pair]] = [[] for _ in range(k)]
    for index, pair in enumerate(edges):
        blocks[index % k].append(pair)

    folds: list[Holdout] = []
    for index, block in enumerate(blocks, start=1):
        positives = tuple(sorted(block))
        train = graph.copy()
        train.remove_edges_from(positives)
        sampled, available = _sample_non_edges(
            train,
            round(negatives * len(positives)),
            random.Random(f"{seed}|{index}"),  # noqa: S311 -- one draw per fold
            exclude=frozenset(positives),
        )
        frame = FOLD_FRAME.format(
            index=index,
            folds=k,
            others=k - 1,
            seed=seed,
            negatives=len(sampled),
            available=available,
        )
        folds.append(
            Holdout(
                train=train,
                positives=positives,
                negatives=tuple(sampled),
                method="fold",
                seed=seed,
                share=len(positives) / len(edges),
                requested_share=1 / k,
                possible_pairs=_possible_pairs(graph, _modes(graph)),
                observed_edges=len(edges),
                available_negatives=available,
                fold=(index, k),
                frame=f"{str(graph.graph.get('frame', '')).strip()} {frame}".strip(),
            )
        )
    return tuple(folds)


def temporal_holdout(
    store: GraphStore,
    persona_id: str,
    network: str = "entities",
    *,
    split_at: str,
    seed: int | None = None,
    negatives: float = DEFAULT_NEGATIVES,
    source_id: str | None = None,
    types: Sequence[str] | None = None,
    stances: Sequence[str] | None = None,
    facets: Sequence[str] | None = None,
    relation_types: Sequence[str] | None = None,
    where: Mapping[str, str] | None = None,
) -> Holdout:
    """Train on what the corpus had joined before a date, test on what it joined after (§25.1).

    The left-hand branch of Figure 25.2 and the one the chapter prefers: *"if you have temporal
    information on your edges you can use earlier edges to predict the later ones. Meaning that
    your train set only contains links up to time t, and the test set only contains links from
    time t + 1 on."* ``split_at`` is the first day of the test period, so the interval is
    half-open: an activation dated ``split_at`` itself is in the test set.

    Built on §7.4's timestamped edge list (:func:`graphrag.sna.layers.dynamic_edges`), which
    dates an edge by the earliest dated passage of the document that produced it. A document
    nobody dated is on neither side and is counted in the frame -- a corpus that is half-dated
    gives a split of the dated half and nothing else.

    A pair joined before the split and joined again after it is **not** a positive: it was
    already in the training data, and §25.1 is explicit that *"we need to throw away the (1, 2)
    edge prediction"* for exactly that reason. A later pair with an endpoint the training graph
    has never seen is dropped and counted, because no topological scorer can rank a node with no
    edges; that convention is this module's, not the book's, and the frame prints how many it
    cost.

    The training graph's weights count activations before the split, so it is the same object
    ``--until`` would have built, assembled from the edge list rather than rebuilt from the
    store: the nodes carry no attributes.
    """
    if not split_at:
        msg = "split_at must be an ISO date (YYYY-MM-DD): the temporal split needs a cut point"
        raise ValueError(msg)

    edges = dynamic_edges(
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
    directed = network == "relations"
    train: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    later: list[Pair] = []
    for edge in edges.edges:
        u, v = (edge.source, edge.target) if directed else tuple(sorted((edge.source, edge.target)))
        if edge.at < split_at:
            if train.has_edge(u, v):
                train[u][v]["weight"] += 1
            else:
                train.add_nodes_from((u, v))
                train.add_edge(u, v, weight=1)
        else:
            later.append((u, v))

    unseen = sorted({pair for pair in later if pair[0] not in train or pair[1] not in train})
    positives = tuple(
        sorted({pair for pair in later if pair not in unseen and not train.has_edge(*pair)})
    )
    train.graph.update(
        network=network,
        persona_id=persona_id,
        until=split_at,
        unit="activations before the split",
    )
    ever = frozenset(positives) | {pair for pair in later if pair not in unseen}
    sampled, available = _sample_non_edges(
        train,
        round(negatives * len(positives)),
        random.Random(seed),  # noqa: S311 -- an experiment, not a secret
        exclude=ever,
    )
    observed = train.number_of_edges() + len(positives)
    frame = TEMPORAL_FRAME.format(
        split_at=split_at,
        network=network,
        train_edges=train.number_of_edges(),
        positives=len(positives),
        unseen=len(unseen),
        negatives=len(sampled),
        available=available,
        undated=edges.undated_documents,
    )
    return Holdout(
        train=train,
        positives=positives,
        negatives=tuple(sampled),
        method="temporal",
        seed=seed,
        share=len(positives) / observed if observed else 0.0,
        split_at=split_at,
        possible_pairs=_possible_pairs(train, _modes(train)),
        observed_edges=observed,
        unseen_positives=len(unseen),
        available_negatives=available,
        frame=f"{edges.frame} {frame}".strip(),
    )


# ------------------------------------------------------------------------ the scores (§25.1)


@dataclass(frozen=True)
class ScoreTable(Mapping[Pair, float]):
    """``score(u, v)`` for every pair the training graph has not already joined, computed lazily.

    This is Figure 25.1(b) with p. 358's arithmetic taken seriously. The score table of a
    192,244-node network has 18 billion rows, which is *"an unreasonable burden, both for
    computation time and memory storage"*, so nothing is computed until a pair is asked for and
    an evaluation only ever asks for the pairs in its test set. Iterating one is possible and
    refused above :data:`MAX_ENUMERATED_PAIRS`, which is the honest form of the same refusal.

    A pair that is already an edge of ``graph`` raises :class:`KeyError` rather than scoring:
    §25.1 is explicit that a link already in the training data is not a prediction, and this is
    where that is enforced for anything reading a table rather than a holdout.
    """

    name: str
    section: str
    graph: nx.Graph = field(repr=False)
    score: Callable[[str, str], float] = field(repr=False)

    def __post_init__(self) -> None:
        _require_string_nodes(self.graph)

    def _modes(self) -> dict[str, str]:
        return _modes(self.graph)

    def __getitem__(self, key: Pair) -> float:
        pair = _pair(self.graph, *key)
        if pair[0] not in self.graph or pair[1] not in self.graph:
            raise KeyError(key)
        if not _allowed(pair, self._modes()) or self.graph.has_edge(*pair):
            raise KeyError(key)
        return float(self.score(*pair))

    def __iter__(self) -> Iterator[Pair]:
        modes = self._modes()
        total = _possible_pairs(self.graph, modes)
        if total > MAX_ENUMERATED_PAIRS:
            msg = (
                f"this score table has {total:,} pair(s), over the {MAX_ENUMERATED_PAIRS:,} this "
                "module will enumerate (§25.1, p. 358). Score the pairs of a holdout instead: "
                "that is what sampling the test set is for."
            )
            raise ValueError(msg)
        yield from _non_edges(self.graph, modes)

    def __len__(self) -> int:
        modes = self._modes()
        return _possible_pairs(self.graph, modes) - int(self.graph.number_of_edges())


def preferential_attachment(graph: nx.Graph) -> ScoreTable:
    """The degree product, as the baseline every evaluation is printed against (§25.2).

    ``score(u, v) = k_u * k_v``: the simplest predictor there is, needing nothing but the degree
    sequence, and the one §25.2 holds a new method to -- a predictor that does not beat it has
    not been shown to have learned anything about structure. The predictors proper are chapter
    23's; this one is here because a report with no baseline column is not a report.

    Degree here is the plain count of distinct neighbours, not the weighted strength, and on a
    directed network it ignores direction while the pair space does not -- so ``(u, v)`` and
    ``(v, u)`` score alike and are still two different candidate links.
    """
    degrees = {str(node): int(degree) for node, degree in graph.degree()}
    return ScoreTable(
        name="preferential attachment",
        section="§25.2",
        graph=graph,
        score=lambda u, v: float(degrees.get(u, 0) * degrees.get(v, 0)),
    )


def random_scores(graph: nx.Graph, *, seed: int | None = None) -> ScoreTable:
    """A score that ignores the network entirely: §25.2's 45-degree line, measured not assumed.

    *"In a ROC curve, the 45 degree line corresponds to the random guess"* (p. 362) and its AUC
    is 0.5. Printing that constant would be an assertion; running this scorer through the same
    evaluation puts a measured number in the same table, which also shows how far from 0.5 a
    test set of this size lets chance wander.

    The value is a deterministic function of the seed and the pair, so the table can be read
    twice and give the same answer -- a scorer whose ranking changed between two reads would
    make precision@k meaningless.
    """

    def score(u: str, v: str) -> float:
        return random.Random(f"{seed}|{u}|{v}").random()  # noqa: S311 -- the 45-degree line

    return ScoreTable(name="random", section="§25.2", graph=graph, score=score)


# -------------------------------------------------------------------- the measurement (§25.2)


@dataclass(frozen=True)
class ConfusionMatrix:
    """The four counts of §25.2 and the rates built on them, at one threshold.

    *"A confusion matrix is simply a grid of four cells ... Confusion matrices are nice because
    they don't attempt to reduce complexity"* (p. 360). Every rate here is a combination of the
    four, and every one of them is undefined when its denominator is empty, which is returned as
    ``nan`` rather than as a zero that would read like a measurement.

    ``true_negatives`` counts only the *sampled* negatives, never the whole pair space: on an
    unbalanced test set *"the vast majority of your observations will end up in the true negative
    cell, obliterating all the rest"* (p. 361), which is what balancing the test set avoids and
    what makes this matrix readable at all.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    threshold: float
    """The score at or above which a pair was predicted to be a link."""

    @property
    def precision(self) -> float:
        """TP/(TP+FP): *"when we predict that a link exists, it exists"*. ``nan`` if nothing was
        predicted."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else math.nan

    @property
    def recall(self) -> float:
        """TP/(TP+FN), the true positive rate: what share of the real links were found."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else math.nan

    @property
    def false_positive_rate(self) -> float:
        """FP/(FP+TN): the share of the non-links that were predicted anyway."""
        actual = self.false_positives + self.true_negatives
        return self.false_positives / actual if actual else math.nan

    @property
    def f1(self) -> float:
        """The harmonic mean of precision and recall (p. 364), ``nan`` when either is undefined."""
        precision, recall = self.precision, self.recall
        if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
            return math.nan
        return 2 * precision * recall / (precision + recall)

    @property
    def accuracy(self) -> float:
        """(TP+TN)/all. *"Its lure is its straightforward intuition. However, it hides the
        difference between type I and type II errors ... and thus it should be handled with
        care"* (p. 365), which is why it is printed last and never alone."""
        total = (
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        )
        return (self.true_positives + self.true_negatives) / total if total else math.nan


@dataclass(frozen=True)
class PrecisionRecallPoint:
    """One threshold of the precision-recall curve of Figure 25.9 (p. 364)."""

    threshold: float
    precision: float
    recall: float


@dataclass(frozen=True)
class RocPoint:
    """One threshold of the ROC curve of Figure 25.6 (p. 362): FPR on x, TPR on y."""

    threshold: float
    false_positive_rate: float
    true_positive_rate: float


def _lookup(scores: Mapping[Pair, float], pair: Pair) -> float | None:
    """The score of one pair, or ``None`` when the scorer did not offer one.

    Both orders are tried, so a caller's plain dictionary does not have to know whether the
    network it came from was directed.
    """
    for key in (pair, (pair[1], pair[0])):
        try:
            return float(scores[key])
        except KeyError:
            continue
    return None


def _ranked(
    scores: Mapping[Pair, float], positives: Sequence[Pair], negatives: Sequence[Pair]
) -> list[tuple[float, bool]]:
    """The test set as ``(score, is a real link)``, highest score first.

    A pair the scorer left out scores 0.0, which is what every neighbourhood predictor means by
    omitting it: no common neighbours, no resource to allocate, no evidence at all. The count of
    those is reported separately so that "the scorer had nothing to say about 40% of the test
    set" is never hidden inside a 0.
    """
    rows = [(_lookup(scores, pair) or 0.0, True) for pair in positives]
    rows += [(_lookup(scores, pair) or 0.0, False) for pair in negatives]
    rows.sort(key=lambda row: (-row[0], not row[1]))
    return rows


def _require_both_classes(positives: Sequence[Pair], negatives: Sequence[Pair]) -> None:
    if not positives or not negatives:
        msg = (
            f"a threshold curve needs both classes: got {len(positives):,} positive(s) and "
            f"{len(negatives):,} negative(s). §25.1's balanced test set is half of each."
        )
        raise ValueError(msg)


def _groups(rows: Sequence[tuple[float, bool]]) -> Iterator[tuple[float, int, int]]:
    """Cumulative ``(threshold, true positives, false positives)``, one per distinct score.

    Grouping by score rather than by row is what makes ties harmless: a predictor that gives the
    same score to a positive and a negative gets neither the credit nor the blame for the order
    they happen to sit in.
    """
    seen_positive = seen_negative = 0
    current: float | None = None
    for score, is_positive in rows:
        if current is not None and score != current:
            yield current, seen_positive, seen_negative
        current = score
        seen_positive += int(is_positive)
        seen_negative += int(not is_positive)
    if current is not None:
        yield current, seen_positive, seen_negative


def auc(
    scores: Mapping[Pair, float], positives: Sequence[Pair], negatives: Sequence[Pair]
) -> float:
    """The area under the ROC curve (§25.2): *"the AUC is 0.5 for the random guess"*.

    Read as the probability that a randomly chosen real link outranks a randomly chosen non-link,
    with a tie counting half. That is the rank statistic rather than a numerical integration, and
    it is exactly the trapezoid area under the step curve :func:`roc` returns.

    1.0 is a perfect ranking and, in the chapter's words, one *"you'll never see and, if you do,
    it means you did something wrong"* -- on a corpus network the usual something is a scorer
    that can see the held-out edge, or a negative class drawn from pairs that could not exist.
    0.5 is chance and below 0.5 is a ranking that would be better reversed.

    Undefined, and refused, when either class is empty: with no negatives there is no false
    positive rate to plot against.
    """
    _require_both_classes(positives, negatives)
    rows = _ranked(scores, positives, negatives)
    ascending = sorted(row[0] for row in rows)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(ascending):
        stop = index
        while stop + 1 < len(ascending) and ascending[stop + 1] == ascending[index]:
            stop += 1
        ranks[ascending[index]] = (index + stop) / 2 + 1
        index = stop + 1
    positive_ranks = sum(ranks[score] for score, is_positive in rows if is_positive)
    count = len(positives)
    return (positive_ranks - count * (count + 1) / 2) / (count * len(negatives))


def roc(
    scores: Mapping[Pair, float], positives: Sequence[Pair], negatives: Sequence[Pair]
) -> tuple[RocPoint, ...]:
    """The ROC curve of Figure 25.6: *"we sort all our predictions by their score ... then we
    keep track of the evolution of TPR and FPR"* (p. 362).

    One point per distinct score, preceded by the origin, so the curve can be plotted or
    integrated. The 45-degree line between ``(0, 0)`` and ``(1, 1)`` is the random guess, and the
    area between the curve and that line is what :func:`auc` reduces to one number.
    """
    _require_both_classes(positives, negatives)
    rows = _ranked(scores, positives, negatives)
    total_positive, total_negative = len(positives), len(negatives)
    points = [RocPoint(threshold=math.inf, false_positive_rate=0.0, true_positive_rate=0.0)]
    points += [
        RocPoint(
            threshold=threshold,
            false_positive_rate=false / total_negative,
            true_positive_rate=true / total_positive,
        )
        for threshold, true, false in _groups(rows)
    ]
    return tuple(points)


def precision_recall(
    scores: Mapping[Pair, float], positives: Sequence[Pair], negatives: Sequence[Pair]
) -> tuple[PrecisionRecallPoint, ...]:
    """The precision-recall curve of Figure 25.9 (p. 364), one point per distinct score.

    *"They tell you how much your precision suffers as you want to recover more and more of the
    actual new edges in the network."* The random classifier is the horizontal line at
    ``P / (P + N)`` -- on a balanced test set, 0.5 -- which is the baseline
    :func:`average_precision` has to be read against and which :func:`render_experiment` prints.
    """
    _require_both_classes(positives, negatives)
    rows = _ranked(scores, positives, negatives)
    total_positive = len(positives)
    return tuple(
        PrecisionRecallPoint(
            threshold=threshold,
            precision=true / (true + false),
            recall=true / total_positive,
        )
        for threshold, true, false in _groups(rows)
    )


def average_precision(
    scores: Mapping[Pair, float], positives: Sequence[Pair], negatives: Sequence[Pair]
) -> float:
    """The area under the precision-recall curve (§25.3's *"again with the objective of
    maximizing their AUC"*).

    **What the book defines and what this adds.** The chapter names the area and leaves the
    estimator open. This is the step sum ``Σ (R_i - R_{i-1}) P_i`` over the distinct thresholds:
    no interpolation between them, because interpolating precision between two thresholds invents
    pairs that were never scored. On a balanced test set its floor is the share of positives,
    ``P / (P + N)``, not 0.5 -- which is the whole reason this is printed beside the AUC rather
    than instead of it.
    """
    points = precision_recall(scores, positives, negatives)
    total = 0.0
    previous = 0.0
    for point in points:
        total += (point.recall - previous) * point.precision
        previous = point.recall
    return total


def precision_at_k(
    scores: Mapping[Pair, float],
    positives: Sequence[Pair],
    negatives: Sequence[Pair],
    k: int = DEFAULT_K,
) -> float:
    """Precision over the ``k`` highest-scoring pairs: §25.2's Precision@n (p. 363).

    *"In Precision@100 we only consider as an actual prediction the 100 pairs of nodes that have
    the highest scores. Everything else is classified as 'no link'."*

    **Where this departs from the book.** The book's Precision@n ranks *every* candidate pair;
    this ranks the pairs of the test set, because those are the pairs a sampled experiment has
    scores for and enumerating the rest is p. 358's unreasonable burden. The consequence is that
    the number is inflated by the balancing -- half the ranked pairs are real links to begin
    with -- so it is only readable against its own random baseline, ``P / (P + N)``, which
    :func:`render_experiment` prints beside it and :func:`prediction_power` turns into a ratio.

    ``k`` is clamped to the size of the test set. Ties at the boundary are broken by putting
    positives first, which is the optimistic reading; the report prints how many pairs shared the
    boundary score so that an optimistic number can be recognised as one.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)
    rows = _ranked(scores, positives, negatives)
    if not rows:
        return math.nan
    top = rows[: min(k, len(rows))]
    return sum(1 for _, is_positive in top if is_positive) / len(top)


def confusion(
    scores: Mapping[Pair, float],
    positives: Sequence[Pair],
    negatives: Sequence[Pair],
    *,
    k: int | None = None,
    threshold: float | None = None,
) -> ConfusionMatrix:
    """The four counts at one cut of the score (§25.2, p. 360).

    Exactly one of ``k`` (predict the k highest-scoring pairs) and ``threshold`` (predict every
    pair scoring at or above it). The chapter's own complaint about this measure is that *"you
    have to pick a threshold in your score ... This is in itself a problematic choice"* (p. 361),
    which is why the curves above exist and why a report prints a matrix beside them rather than
    instead of them.
    """
    if (k is None) == (threshold is None):
        msg = "pass exactly one of k and threshold: a confusion matrix is one cut of the score"
        raise ValueError(msg)
    rows = _ranked(scores, positives, negatives)
    if k is not None:
        if k < 1:
            msg = f"k must be at least 1, got {k}"
            raise ValueError(msg)
        cut = min(k, len(rows))
        predicted = rows[:cut]
        boundary = predicted[-1][0] if predicted else math.inf
    else:
        boundary = float(threshold if threshold is not None else math.inf)
        predicted = [row for row in rows if row[0] >= boundary]
    true_positive = sum(1 for _, is_positive in predicted if is_positive)
    false_positive = len(predicted) - true_positive
    return ConfusionMatrix(
        true_positives=true_positive,
        false_positives=false_positive,
        true_negatives=len(negatives) - false_positive,
        false_negatives=len(positives) - true_positive,
        threshold=boundary,
    )


def prediction_power(precision: float, random_precision: float) -> float:
    """§25.2's prediction power: how many times better than chance this precision is, in decibels.

    *"If we say that your precision is P and the random precision is Pr, then the prediction
    power PP is PP = 10 log10 (P / Pr)"* (p. 365).

    **Where this departs from the book.** The formula and the reading the chapter gives it do not
    agree: it says *"a PP = 1 implies your predictor is ten times better than random, while
    PP = 2 means you are one hundred times better"*, which is ``log10``, while ``10 log10`` -- a
    decibel rather than a bel -- puts those at 10 and 20. The formula as written is what is
    implemented, so read a PP of 10 as ten times better than chance and a PP of 0 as no better,
    which is the horizontal line the chapter's PP-curves are drawn against either way.

    ``-inf`` when the predictor found nothing (a precision of zero is infinitely worse than
    chance on this scale); ``nan`` when either number is undefined.
    """
    if math.isnan(precision) or math.isnan(random_precision):
        return math.nan
    if random_precision <= 0:
        msg = "the random precision must be positive: it is the share of positives in the test set"
        raise ValueError(msg)
    if precision <= 0:
        return -math.inf
    return 10 * math.log10(precision / random_precision)


# ---------------------------------------------------------------------------- the report


@dataclass(frozen=True)
class EvaluationReport:
    """What one scorer did on one holdout, as §25.2 asks it to be reported.

    Never a single number: the chapter's warning about *"fixed threshold metrics, i.e. everything
    that boils down a complex phenomenon to a single number"* (p. 360) is the reason this carries
    the confusion matrix, both areas, precision@k and the curves together.
    """

    method: str
    section: str
    auc: float
    average_precision: float
    precision_at_k: float
    k: int
    random_precision: float
    """``P / (P + N)``: the horizontal line of the precision-recall plot and the precision a
    coin would reach on this test set. Every precision here is read against it."""
    prediction_power: float
    matrix: ConfusionMatrix
    curve: tuple[PrecisionRecallPoint, ...]
    roc_points: tuple[RocPoint, ...]
    scored: int
    """Pairs of the test set the scorer gave a number for."""
    unscored: int
    """Pairs it did not, which were counted as 0.0 -- the meaning a neighbourhood scorer gives
    an omission."""
    tied_at_k: int
    """How many test pairs share the score at the precision@k boundary. A large number means the
    k chosen is inside a block of ties and the ordering within it is arbitrary."""

    @property
    def beats_random(self) -> bool:
        """Whether the AUC is above the 45-degree line at all (§25.2). Not a significance test."""
        return self.auc > RANDOM_AUC


def evaluate_predictor(
    split: Holdout,
    scorer: Callable[[nx.Graph], Mapping[Pair, float]],
    *,
    name: str = "",
    k: int = DEFAULT_K,
) -> EvaluationReport:
    """Run one scorer on one holdout and measure it the way §25.2 asks.

    ``scorer`` takes the **training graph only** -- it never sees the positives, which is the
    whole point of §25.1 -- and returns something that maps a pair to a score. A
    :class:`ScoreTable` is the lazy form and the one the baselines return; a plain dictionary
    works too, and any pair missing from it is read as 0.0.

    The name is taken from ``name``, or from the score table, or from the callable, in that
    order, because the method column of the report is what makes two rows comparable.
    """
    scores = scorer(split.train)
    label = name or getattr(scores, "name", "") or getattr(scorer, "__name__", "scorer")
    section = str(getattr(scores, "section", "§25.2"))
    pairs = split.pairs
    if not split.positives or not split.negatives:
        msg = (
            f"this holdout has {len(split.positives):,} positive(s) and "
            f"{len(split.negatives):,} negative(s); §25.2's measures need both classes"
        )
        raise ValueError(msg)

    found = [pair for pair in pairs if _lookup(scores, pair) is not None]
    rows = _ranked(scores, split.positives, split.negatives)
    effective_k = min(k, len(rows))
    boundary = rows[effective_k - 1][0]
    matrix = confusion(scores, split.positives, split.negatives, k=effective_k)
    random_precision = len(split.positives) / len(pairs)
    at_k = precision_at_k(scores, split.positives, split.negatives, effective_k)
    return EvaluationReport(
        method=str(label),
        section=section,
        auc=auc(scores, split.positives, split.negatives),
        average_precision=average_precision(scores, split.positives, split.negatives),
        precision_at_k=at_k,
        k=effective_k,
        random_precision=random_precision,
        prediction_power=prediction_power(at_k, random_precision),
        matrix=matrix,
        curve=precision_recall(scores, split.positives, split.negatives),
        roc_points=roc(scores, split.positives, split.negatives),
        scored=len(found),
        unscored=len(pairs) - len(found),
        tied_at_k=sum(1 for score, _ in rows if score == boundary),
    )


def _num(value: float, places: int = 3) -> str:
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _n_line(split: Holdout) -> str:
    return (
        f"**n.** {len(split.pairs):,} pair(s) scored: {len(split.positives):,} positive(s) "
        f"(links the network really has and the training graph does not) and "
        f"{len(split.negatives):,} negative(s) sampled from the {split.non_edges:,} pair(s) it "
        f"never joined, out of {split.possible_pairs:,} possible. Imbalance in the unsampled "
        f"space: {split.imbalance:,.1f} non-edge(s) per positive -- the test set is balanced and "
        "the network is not."
    )


def render_experiment(
    split: Holdout, reports: Sequence[EvaluationReport], *, title: str = ""
) -> str:
    """The evaluation as markdown: the split, the imbalance, the table, and what not to read.

    Prints, next to the numbers, the four things a reader needs in order to disagree -- the
    sampling frame, the n with its imbalance ratio, the null model, and the chapter -- and then
    the chapter's own cautions about the numbers immediately above them.
    """
    heading = title or f"Link prediction evaluation ({split.method} holdout, §25.1-25.2)"
    lines: list[str] = [
        f"## {heading}",
        "",
        f"**Sampling frame.** {split.frame}",
        "",
        _n_line(split),
        "",
        "**Null model.** Two, both measured rather than assumed: a scorer that ignores the "
        f"network, whose AUC is {RANDOM_AUC} by construction (§25.2, the 45-degree line of "
        "Figure 25.6) and whose precision is the share of positives in the test set; and "
        "preferential attachment, the degree product, which is the baseline §25.2 holds a new "
        "predictor to. A method that does not beat both rows has not been shown to have learned "
        "anything about this network's structure.",
        "",
        "**Implements.** Atlas ch. 25, *Designing an Experiment*: §25.1 for the split and the "
        f"negative sampling, §25.2 for everything measured below. This split: {split.section}, "
        f"{split.method}" + (f" at {split.split_at}" if split.split_at else "") + ".",
        "",
    ]
    if split.unseen_positives:
        lines += [
            f"**Dropped.** {split.unseen_positives:,} later pair(s) had an endpoint the training "
            "graph never saw and are in neither class: a topological scorer cannot rank a node "
            "with no edges, so scoring them would measure the corpus's growth rather than the "
            "predictor.",
            "",
        ]
    lines += _table(
        [
            "method",
            "AUC",
            "avg precision",
            f"P@{reports[0].k if reports else DEFAULT_K}",
            "random P",
            "PP (dB)",
            "TP",
            "FP",
            "TN",
            "FN",
            "F1",
            "accuracy",
            "§",
        ],
        [
            [
                report.method,
                _num(report.auc),
                _num(report.average_precision),
                _num(report.precision_at_k),
                _num(report.random_precision),
                _num(report.prediction_power, 2),
                f"{report.matrix.true_positives:,}",
                f"{report.matrix.false_positives:,}",
                f"{report.matrix.true_negatives:,}",
                f"{report.matrix.false_negatives:,}",
                _num(report.matrix.f1),
                _num(report.matrix.accuracy),
                report.section,
            ]
            for report in reports
        ],
    )
    lines += [
        "The confusion matrix columns are cut at the top-k threshold, not at a score, because "
        '§25.2 says picking one *"is in itself a problematic choice"*; the two areas are the '
        "threshold-free readings and are what a comparison between methods should use.",
        "",
        "**How to read this.** An AUC is flattered by imbalance -- on the full pair space almost "
        "every candidate is a non-link, and the easy non-links inflate it -- so precision@k and "
        "the average precision, both read against the random-P column, are the numbers to quote. "
        'Accuracy is printed last because it *"hides the difference between type I and type II '
        'errors"* (p. 365). An AUC of 1.0 is not a triumph: the chapter says you *"never"* see '
        "one and that seeing one means something is wrong, usually a scorer that can see the "
        "answer.",
        "",
    ]
    notes = [
        f"- {report.method}: {report.unscored:,} of {len(split.pairs):,} test pair(s) got no "
        f"score and were read as 0.0; {report.tied_at_k:,} pair(s) share the score at the "
        f"P@{report.k} boundary."
        for report in reports
        if report.unscored or report.tied_at_k > report.k
    ]
    if notes:
        lines += ["**Ties and silences.**", "", *notes, ""]
    return "\n".join(lines)


def experiment_payload(split: Holdout, reports: Sequence[EvaluationReport]) -> dict[str, Any]:
    """The same evaluation as JSON, curves included, for a caller that would rather plot it."""
    return {
        "chapter": 25,
        "section": "§25.1-25.2",
        "holdout": {
            "method": split.method,
            "seed": split.seed,
            "share": split.share,
            "requested_share": split.requested_share,
            "split_at": split.split_at,
            "fold": list(split.fold),
            "frame": split.frame,
            "train_nodes": split.train.number_of_nodes(),
            "train_edges": split.train.number_of_edges(),
            "positives": len(split.positives),
            "negatives": len(split.negatives),
            "possible_pairs": split.possible_pairs,
            "non_edges": split.non_edges,
            "imbalance": split.imbalance,
            "unseen_positives": split.unseen_positives,
            "available_negatives": split.available_negatives,
        },
        "null_model": (
            "a scorer that ignores the network (AUC 0.5, §25.2) and preferential attachment, the "
            "degree-product baseline a predictor has to beat"
        ),
        "methods": [
            {
                "method": report.method,
                "section": report.section,
                "auc": report.auc,
                "average_precision": report.average_precision,
                "precision_at_k": report.precision_at_k,
                "k": report.k,
                "random_precision": report.random_precision,
                "prediction_power_db": report.prediction_power,
                "beats_random": report.beats_random,
                "confusion": {
                    "true_positives": report.matrix.true_positives,
                    "false_positives": report.matrix.false_positives,
                    "true_negatives": report.matrix.true_negatives,
                    "false_negatives": report.matrix.false_negatives,
                    "threshold": report.matrix.threshold,
                    "precision": report.matrix.precision,
                    "recall": report.matrix.recall,
                    "f1": report.matrix.f1,
                    "accuracy": report.matrix.accuracy,
                },
                "precision_recall": [
                    {"threshold": p.threshold, "precision": p.precision, "recall": p.recall}
                    for p in report.curve
                ],
                "roc": [
                    {
                        "threshold": None if math.isinf(p.threshold) else p.threshold,
                        "false_positive_rate": p.false_positive_rate,
                        "true_positive_rate": p.true_positive_rate,
                    }
                    for p in report.roc_points
                ],
                "scored": report.scored,
                "unscored": report.unscored,
                "tied_at_k": report.tied_at_k,
            }
            for report in reports
        ],
    }
