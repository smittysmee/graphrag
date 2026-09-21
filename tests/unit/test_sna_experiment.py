"""Chapter 25's experiment, held to answers that are known before the code runs.

Three kinds of fixture, and each one exists because a link-prediction evaluation is exactly the
sort of code that can return a plausible number while being wrong.

*Planted score tables.* A perfect ranking must give AUC 1.0 and precision@k 1.0, its reverse
0.0, and a scorer that ignores the network must land near 0.5 -- the book's own three landmarks
(*"the AUC is 0.5 for the random guess ... an AUC of 1 ... means a perfect classifier"*, p. 363).
Those are asserted on hand-written tables whose confusion matrices are written out in the test.

*A planted preferential-attachment network.* ``nx.barabasi_albert_graph(500, 3, seed=1)``: 500
nodes, 1,491 edges. The degree-product baseline should beat chance on a network *grown* by the
degree product, and the measured values over ten seeds are in the test beside the threshold they
motivate.

*A planted dated corpus.* ``DATED_PASSAGES`` below is a corpus whose temporal split is worked
out by hand in its docstring: which pairs are in the training graph, which are the positives,
which later pair is dropped because its endpoint is new, and which later pair is not a positive
because it was already joined before the split (§25.1's *"we need to throw away the (1, 2) edge
prediction"*).
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import PREDICT_K, PREDICT_NEGATIVES, PREDICT_SHARE
from graphrag.cli import app as cli_app
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import (
    Chunk,
    Document,
    Enrichment,
    Entity,
    Mention,
    PersonaSpec,
    SourceSpec,
    SpeakerPost,
)
from graphrag.personas.registry import PersonaRegistry
from graphrag.sna.experiment import (
    DEFAULT_K,
    DEFAULT_NEGATIVES,
    DEFAULT_SHARE,
    RANDOM_AUC,
    Holdout,
    auc,
    average_precision,
    confusion,
    evaluate_predictor,
    experiment_payload,
    holdout,
    kfold,
    precision_at_k,
    precision_recall,
    prediction_power,
    preferential_attachment,
    random_scores,
    render_experiment,
    roc,
    temporal_holdout,
)
from graphrag.sna.guide import READING_RULES
from tests.legendary import karate_club

runner = CliRunner()

#: Seeds every "at every seed" assertion runs over. Ten is enough for a mean to settle and few
#: enough to keep the file quick.
SEEDS = tuple(range(1, 11))


@pytest.fixture(scope="module")
def grown() -> nx.Graph:
    """A preferential-attachment network, relabelled to string ids as every network here has."""
    return nx.relabel_nodes(nx.barabasi_albert_graph(500, 3, seed=1), lambda node: f"n{node}")


# ------------------------------------------------------------------ the split (§25.1)


def test_a_random_holdout_removes_the_share_it_was_asked_for_and_keeps_the_nodes(
    grown: nx.Graph,
) -> None:
    split = holdout(grown, share=0.1, seed=1)
    assert len(split.positives) == round(0.1 * 1491) == 149
    assert split.train.number_of_edges() == 1491 - 149
    # Every node stays, so the pair space is the same before and after: 500 * 499 / 2.
    assert split.train.number_of_nodes() == 500
    assert split.possible_pairs == 124_750
    assert split.observed_edges == 1491
    assert split.non_edges == 124_750 - 1491
    # No positive survives in the training graph: §25.1's fundamental tenet, enforced.
    assert not [pair for pair in split.positives if split.train.has_edge(*pair)]
    assert not set(split.positives) & set(split.negatives)


def test_the_negatives_are_pairs_the_network_never_joined_at_the_ratio_asked_for(
    grown: nx.Graph,
) -> None:
    balanced = holdout(grown, share=0.1, seed=2)
    assert len(balanced.negatives) == len(balanced.positives)
    assert not [pair for pair in balanced.negatives if grown.has_edge(*pair)]

    unbalanced = holdout(grown, share=0.1, seed=2, negatives=3)
    assert len(unbalanced.negatives) == 3 * len(unbalanced.positives)
    assert not [pair for pair in unbalanced.negatives if grown.has_edge(*pair)]


def test_the_holdout_is_reproducible_at_a_seed_and_moves_without_one(grown: nx.Graph) -> None:
    first = holdout(grown, share=0.05, seed=7)
    again = holdout(grown, share=0.05, seed=7)
    other = holdout(grown, share=0.05, seed=8)
    assert first.positives == again.positives
    assert first.negatives == again.negatives
    assert first.positives != other.positives


def test_keeping_the_training_graph_connected_costs_share_and_says_so() -> None:
    """A star has no removable edge: every one of them is a bridge (§25.1's cost, made visible)."""
    star = nx.relabel_nodes(nx.star_graph(20), lambda node: f"n{node}")
    free = holdout(star, share=0.5, seed=1)
    assert len(free.positives) == 10
    assert nx.number_connected_components(free.train) == 11  # ten leaves cut loose, plus the star

    careful = holdout(star, share=0.5, seed=1, keep_connected=True)
    assert careful.positives == ()
    assert nx.number_connected_components(careful.train) == 1
    assert "were put back because deleting them would have cut their endpoints apart" in (
        careful.frame
    )


def test_a_holdout_refuses_what_it_cannot_split() -> None:
    empty: nx.Graph = nx.Graph()
    with pytest.raises(ValueError, match="network that has none"):
        holdout(empty, seed=1)
    path = nx.relabel_nodes(nx.path_graph(10), lambda node: f"n{node}")
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        holdout(path, share=1.5, seed=1)
    with pytest.raises(ValueError, match="nothing to learn from"):
        holdout(path, share=0.99, seed=1)
    with pytest.raises(ValueError, match="node ids must be strings"):
        holdout(nx.path_graph(10), seed=1)


def test_a_holdout_refuses_a_test_set_that_is_still_in_the_training_graph() -> None:
    """The one mistake §25.1 says would 'grossly overestimate your actual performance'."""
    triangle = nx.Graph([("a", "b"), ("b", "c"), ("a", "c")])
    with pytest.raises(ValueError, match="still edges of the training graph"):
        Holdout(train=triangle, positives=(("a", "b"),), negatives=(("a", "z"),))


def test_more_negatives_than_the_network_has_non_edges_is_refused() -> None:
    """A complete graph has no non-edge at all, so it has no negative class to sample."""
    complete = nx.relabel_nodes(nx.complete_graph(6), lambda node: f"n{node}")
    with pytest.raises(ValueError, match="pair\\(s\\) it never joined"):
        holdout(complete, share=0.2, seed=1)


def test_negatives_of_a_two_mode_network_never_join_two_nodes_of_one_mode() -> None:
    """A pair of speakers is not a candidate link: no passage could ever produce it."""
    graph = nx.Graph()
    speakers, entities = [f"s{i}" for i in range(4)], [f"e{i}" for i in range(4)]
    graph.add_nodes_from(speakers, mode="speaker")
    graph.add_nodes_from(entities, mode="entity")
    graph.add_edges_from((s, e) for s in speakers for e in entities[:2])
    split = holdout(graph, share=0.2, seed=1, negatives=4)
    assert split.possible_pairs == 16  # 4 speakers x 4 entities, not 8 * 7 / 2 = 28
    # 16 possible pairs, 8 of them joined, so the 8 left are every candidate there is.
    assert len(split.negatives) == 4 * len(split.positives) == 8
    assert split.available_negatives == 8
    for left, right in split.negatives:
        assert graph.nodes[left]["mode"] != graph.nodes[right]["mode"]


def test_kfold_partitions_the_edges_exactly(grown: nx.Graph) -> None:
    """*"Rotating the test set so that each edge appears in it at least once"* (p. 358)."""
    folds = kfold(grown, 10, seed=1)
    assert len(folds) == 10
    positives = [pair for fold in folds for pair in fold.positives]
    edges = {tuple(sorted(edge)) for edge in grown.edges()}
    assert len(positives) == len(edges) == 1491  # every edge once, and no edge twice
    assert set(positives) == edges
    assert {len(fold.positives) for fold in folds} == {149, 150}
    for index, fold in enumerate(folds, start=1):
        assert fold.fold == (index, 10)
        assert fold.train.number_of_edges() == 1491 - len(fold.positives)
        assert not [pair for pair in fold.positives if fold.train.has_edge(*pair)]


def test_kfold_refuses_a_k_it_cannot_honour() -> None:
    triangle = nx.Graph([("a", "b"), ("b", "c"), ("a", "c")])
    with pytest.raises(ValueError, match="k must be at least 2"):
        kfold(triangle, 1, seed=1)
    with pytest.raises(ValueError, match="cannot cut 3 edge"):
        kfold(triangle, 5, seed=1)


# ------------------------------------------------------------- the measurement (§25.2)

#: Four pairs with a perfect ranking: both positives above both negatives.
PERFECT: dict[tuple[str, str], float] = {
    ("a", "b"): 0.9,
    ("c", "d"): 0.8,
    ("e", "f"): 0.2,
    ("g", "h"): 0.1,
}
POSITIVES = (("a", "b"), ("c", "d"))
NEGATIVES = (("e", "f"), ("g", "h"))


def test_a_perfect_ranking_scores_one_and_its_reverse_scores_zero() -> None:
    assert auc(PERFECT, POSITIVES, NEGATIVES) == 1.0
    assert average_precision(PERFECT, POSITIVES, NEGATIVES) == 1.0
    assert precision_at_k(PERFECT, POSITIVES, NEGATIVES, 2) == 1.0

    reversed_scores = {pair: 1 - score for pair, score in PERFECT.items()}
    assert auc(reversed_scores, POSITIVES, NEGATIVES) == 0.0
    assert precision_at_k(reversed_scores, POSITIVES, NEGATIVES, 2) == 0.0
    # Precision@4 takes every pair, so it is the share of positives whatever the order.
    assert precision_at_k(reversed_scores, POSITIVES, NEGATIVES, 4) == 0.5


def test_a_score_that_ties_every_pair_is_the_random_guess() -> None:
    """Ties count half a win each, which is the only reading that puts a flat score at 0.5."""
    flat = dict.fromkeys(PERFECT, 1.0)
    assert auc(flat, POSITIVES, NEGATIVES) == 0.5
    # One positive tied with one negative, the other positive on top: 0.5 + 0.25 + 0.5 * 0.25.
    partial = {("a", "b"): 0.9, ("c", "d"): 0.5, ("e", "f"): 0.5, ("g", "h"): 0.1}
    assert auc(partial, POSITIVES, NEGATIVES) == 0.875


def test_an_unscored_pair_is_read_as_zero_and_counted() -> None:
    """What a neighbourhood predictor means by leaving a pair out: no common neighbours at all."""
    partial = {("a", "b"): 0.9}
    assert auc(partial, POSITIVES, NEGATIVES) == 0.75  # a,b beats both; c,d ties both at 0.0
    split = Holdout(train=nx.Graph(), positives=POSITIVES, negatives=NEGATIVES)
    report = evaluate_predictor(split, lambda _: partial, name="partial", k=2)
    assert report.scored == 1
    assert report.unscored == 3


def test_the_roc_curve_starts_at_the_origin_and_integrates_to_the_auc(grown: nx.Graph) -> None:
    split = holdout(grown, share=0.1, seed=3)
    scores = preferential_attachment(split.train)
    points = roc(scores, split.positives, split.negatives)
    assert (points[0].false_positive_rate, points[0].true_positive_rate) == (0.0, 0.0)
    assert (points[-1].false_positive_rate, points[-1].true_positive_rate) == (1.0, 1.0)
    area = sum(
        (right.false_positive_rate - left.false_positive_rate)
        * (right.true_positive_rate + left.true_positive_rate)
        / 2
        for left, right in itertools.pairwise(points)
    )
    # The rank statistic and the trapezoid area under the step curve are the same number.
    assert area == pytest.approx(auc(scores, split.positives, split.negatives), abs=1e-12)


def test_the_precision_recall_curve_ends_at_full_recall_and_the_random_share() -> None:
    points = precision_recall(PERFECT, POSITIVES, NEGATIVES)
    assert [(p.precision, p.recall) for p in points] == [
        (1.0, 0.5),  # the top pair is a positive
        (1.0, 1.0),  # so is the second: precision still 1, recall complete
        (2 / 3, 1.0),
        (0.5, 1.0),  # every pair predicted: precision falls to the share of positives
    ]
    assert points[-1].precision == len(POSITIVES) / (len(POSITIVES) + len(NEGATIVES))


def test_a_curve_needs_both_classes() -> None:
    for call in (auc, average_precision, precision_recall, roc):
        with pytest.raises(ValueError, match="needs both classes"):
            call(PERFECT, POSITIVES, ())


def test_the_confusion_matrix_counts_the_four_cells_at_one_cut() -> None:
    matrix = confusion(PERFECT, POSITIVES, NEGATIVES, k=2)
    assert (matrix.true_positives, matrix.false_positives) == (2, 0)
    assert (matrix.true_negatives, matrix.false_negatives) == (2, 0)
    assert matrix.precision == matrix.recall == matrix.f1 == matrix.accuracy == 1.0
    assert matrix.false_positive_rate == 0.0

    # Predict everything: perfect recall, precision down to the share of positives (p. 364).
    everything = confusion(PERFECT, POSITIVES, NEGATIVES, threshold=0.0)
    assert (everything.true_positives, everything.false_positives) == (2, 2)
    assert everything.recall == 1.0
    assert everything.precision == 0.5
    assert everything.f1 == pytest.approx(2 / 3)

    with pytest.raises(ValueError, match="exactly one of k and threshold"):
        confusion(PERFECT, POSITIVES, NEGATIVES)


def test_prediction_power_is_the_books_formula_and_zero_at_chance() -> None:
    """PP = 10 log10(P / Pr), p. 365 -- the formula as written, not the prose's reading of it."""
    assert prediction_power(0.5, 0.5) == 0.0
    assert prediction_power(1.0, 0.5) == pytest.approx(10 * math.log10(2))
    assert prediction_power(0.0, 0.5) == -math.inf
    assert math.isnan(prediction_power(math.nan, 0.5))
    with pytest.raises(ValueError, match="random precision must be positive"):
        prediction_power(0.5, 0.0)


# ------------------------------------------------------- the baselines on a planted network


def test_preferential_attachment_is_the_degree_product_and_refuses_an_existing_edge() -> None:
    graph = nx.Graph([("a", "b"), ("b", "c"), ("b", "d"), ("c", "e")])
    scores = preferential_attachment(graph)
    # degrees: a 1, b 3, c 2, d 1, e 1.
    assert scores[("a", "c")] == 2.0
    assert scores[("a", "e")] == 1.0
    assert scores[("d", "c")] == 2.0
    with pytest.raises(KeyError):
        scores[("a", "b")]  # already an edge: §25.1 says that is not a prediction
    with pytest.raises(KeyError):
        scores[("a", "zzz")]
    assert len(scores) == 10 - 4  # 5 nodes, 10 possible pairs, 4 already joined


def test_preferential_attachment_beats_chance_on_a_network_grown_by_it(grown: nx.Graph) -> None:
    """The planted case: a BA network is built by the degree product, so the degree product ranks.

    Measured over the ten seeds below: AUC 0.598 to 0.708, mean 0.654; the random scorer 0.462 to
    0.561, mean 0.509. The thresholds are set under the observed minimum, not at it. 0.654 rather
    than the 0.8 a reader might expect is a fact about ties: half the nodes of BA(500, 3) have
    degree 3 or 4, so a great many pairs share a score and each of those ties counts half a win.
    """
    baseline = []
    chance = []
    for seed in SEEDS:
        split = holdout(grown, share=0.1, seed=seed)
        baseline.append(evaluate_predictor(split, preferential_attachment).auc)
        chance.append(evaluate_predictor(split, lambda train: random_scores(train, seed=1)).auc)
    assert min(baseline) > 0.58
    assert statistics.mean(baseline) > 0.64
    assert all(pa > rand for pa, rand in zip(baseline, chance, strict=True))
    # The random scorer is the 45-degree line, and a 298-pair test set lets it wander about 0.05.
    assert statistics.mean(chance) == pytest.approx(RANDOM_AUC, abs=0.03)


def test_the_random_scorer_ranks_the_same_way_twice(grown: nx.Graph) -> None:
    """A scorer whose ranking moved between two reads would make precision@k meaningless."""
    split = holdout(grown, share=0.05, seed=4)
    first = random_scores(split.train, seed=11)
    again = random_scores(split.train, seed=11)
    assert [first[pair] for pair in split.pairs] == [again[pair] for pair in split.pairs]
    assert first[split.pairs[0]] != random_scores(split.train, seed=12)[split.pairs[0]]


def test_a_holdout_of_the_karate_club_keeps_its_published_edge_count() -> None:
    """The legendary graph, as a check that the arithmetic holds on a network with a citation."""
    graph, known = karate_club()
    graph = nx.relabel_nodes(graph, str)
    assert known.edges == 78
    split = holdout(graph, share=0.1, seed=1)
    assert len(split.positives) == 8  # round(0.1 * 78)
    assert split.train.number_of_edges() == 70
    assert split.possible_pairs == 34 * 33 // 2
    assert split.imbalance == pytest.approx((561 - 78) / 8)
    report = evaluate_predictor(split, preferential_attachment, k=5)
    assert 0.0 <= report.auc <= 1.0
    assert report.random_precision == 0.5


# ------------------------------------------------------------ the temporal split (§25.1)

DATED_PERSONA = "test-dated"
DATED_SOURCE = "notes"
#: ``(slug, date, entity ids)``: one passage per document, naming the entities it holds.
#:
#: Read the entity network off the table. Before 2025-06-01 the corpus joins e0-e1, e1-e2,
#: e2-e3, e0-e2, e4-e5, e5-e6 and e4-e6: two triangles, seven nodes, seven edges. On and after
#: it, four pairs are named: (e0, e3) and (e1, e3) are new and are the **positives**; (e0, e1)
#: was already joined and is thrown away, which is §25.1's own instruction about the (1, 2)
#: edge; (e4, e7) names a node that had never appeared, so it is **dropped** rather than counted
#: as a miss. One document carries no date at all and is on neither side.
DATED_PASSAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("d1", "2025-01-05", ("e0", "e1")),
    ("d2", "2025-01-12", ("e1", "e2")),
    ("d3", "2025-02-03", ("e2", "e3")),
    ("d4", "2025-02-14", ("e0", "e2")),
    ("d5", "2025-03-01", ("e4", "e5")),
    ("d6", "2025-03-20", ("e5", "e6")),
    ("d7", "2025-04-02", ("e4", "e6")),
    ("d8", "2025-07-01", ("e0", "e3")),
    ("d9", "2025-07-08", ("e1", "e3")),
    ("d10", "2025-07-15", ("e0", "e1")),
    ("d11", "2025-08-01", ("e4", "e7")),
    ("d12", "", ("e2", "e6")),
)
SPLIT_AT = "2025-06-01"


@pytest.fixture
def dated(
    memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder, registry: PersonaRegistry
) -> InMemoryGraphStore:
    """The planted corpus above, in the shared in-memory store and the shared registry."""
    registry.save(
        PersonaSpec(
            id=DATED_PERSONA,
            name="Test Dated",
            role_prompt="You answer from dated notes.",
            sources=[SourceSpec(id=DATED_SOURCE, kind="local", path="notes", loader="documents")],
        )
    )
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for ordinal, (slug, posted, entities) in enumerate(DATED_PASSAGES):
        doc_id = f"{DATED_PERSONA}:{DATED_SOURCE}:{slug}"
        text = f"A note about {' and '.join(entities)}, written for the experiment."
        documents.append(
            Document(
                id=doc_id,
                persona_id=DATED_PERSONA,
                source_id=DATED_SOURCE,
                title=slug,
                path=f"notes/{slug}.md",
                speakers=["writer"],
                topics=["notes"],
                word_count=len(text.split()),
            )
        )
        chunks.append(
            Chunk(
                id=f"{doc_id}#0",
                doc_id=doc_id,
                persona_id=DATED_PERSONA,
                ordinal=ordinal,
                text=text,
                speaker="writer",
                speakers=["writer"],
                speaker_posts=[SpeakerPost(speaker="writer", posted_at=posted or None)],
                word_count=len(text.split()),
            )
        )
    memory_store.upsert_documents(documents)
    memory_store.upsert_chunks(chunks, hash_embedder.embed_documents([c.text for c in chunks]))
    memory_store.upsert_enrichment(
        Enrichment(
            entities=[Entity(id=f"e{i}", name=f"Entity {i}", type="concept") for i in range(8)],
            mentions=[
                Mention(chunk_id=f"{DATED_PERSONA}:{DATED_SOURCE}:{slug}#0", entity_id=entity)
                for slug, _, entities in DATED_PASSAGES
                for entity in entities
            ],
        )
    )
    return memory_store


def test_the_temporal_split_puts_every_training_edge_before_it_and_every_positive_after(
    dated: InMemoryGraphStore,
) -> None:
    split = temporal_holdout(dated, DATED_PERSONA, "entities", split_at=SPLIT_AT, seed=1)
    assert split.method == "temporal"
    assert split.split_at == SPLIT_AT

    # The training graph is exactly the pairs the corpus joined before the split.
    assert sorted(tuple(sorted(edge)) for edge in split.train.edges()) == [
        ("e0", "e1"),
        ("e0", "e2"),
        ("e1", "e2"),
        ("e2", "e3"),
        ("e4", "e5"),
        ("e4", "e6"),
        ("e5", "e6"),
    ]
    assert set(split.train.nodes) == {f"e{i}" for i in range(7)}  # e7 is never seen before it
    # The positives are the pairs first joined on or after it, and only those.
    assert split.positives == (("e0", "e3"), ("e1", "e3"))
    # (e0, e1) is joined on both sides of the split, so it is not a prediction (§25.1).
    assert split.train.has_edge("e0", "e1")
    assert ("e0", "e1") not in split.positives
    # (e4, e7) is dropped: e7 has no edge in the training graph, so nothing can rank it.
    assert split.unseen_positives == 1
    assert not [pair for pair in split.positives if "e7" in pair]
    # The undated document is on neither side and is counted in the frame.
    assert "1 document(s) carry no dated passage" in split.frame


def test_the_temporal_negatives_are_pairs_the_corpus_never_joined(
    dated: InMemoryGraphStore,
) -> None:
    split = temporal_holdout(
        dated, DATED_PERSONA, "entities", split_at=SPLIT_AT, seed=1, negatives=4
    )
    assert len(split.negatives) == 4 * len(split.positives) == 8
    joined = {tuple(sorted(pair)) for _, _, entities in DATED_PASSAGES for pair in [entities]}
    for pair in split.negatives:
        assert pair not in joined
        assert not split.train.has_edge(*pair)
    # 7 nodes -> 21 pairs, 7 joined before, 2 positives and the undated (e2, e6) pair excluded.
    assert split.possible_pairs == 21
    assert split.available_negatives == 21 - 7 - 2


def test_the_temporal_split_needs_a_date(dated: InMemoryGraphStore) -> None:
    with pytest.raises(ValueError, match="must be an ISO date"):
        temporal_holdout(dated, DATED_PERSONA, "entities", split_at="")


def test_a_split_after_the_corpus_ends_has_no_positives(dated: InMemoryGraphStore) -> None:
    """Which is not an error: it is a corpus that stopped, and the report has to say so."""
    split = temporal_holdout(dated, DATED_PERSONA, "entities", split_at="2099-01-01", seed=1)
    assert split.positives == ()
    assert split.negatives == ()
    assert split.imbalance == math.inf
    with pytest.raises(ValueError, match="measures need both classes"):
        evaluate_predictor(split, preferential_attachment)


# ---------------------------------------------------------------------- the report


def test_the_report_prints_its_frame_its_n_its_null_model_and_its_chapter(
    grown: nx.Graph,
) -> None:
    split = holdout(grown, share=0.1, seed=5)
    reports = [
        evaluate_predictor(split, preferential_attachment, k=10),
        evaluate_predictor(split, lambda train: random_scores(train, seed=5), k=10),
    ]
    text = render_experiment(split, reports)
    assert "**Sampling frame.** A random edge holdout (Atlas §25.1)" in text
    assert "**n.** 298 pair(s) scored: 149 positive(s)" in text
    assert "827.2 non-edge(s) per positive" in text
    assert "**Null model.** Two, both measured rather than assumed" in text
    assert "**Implements.** Atlas ch. 25" in text
    assert "| preferential attachment |" in text and "| random |" in text
    assert "An AUC is flattered by imbalance" in text
    assert "precision@k and the average precision" in text


def test_the_payload_carries_the_curves_and_is_json(grown: nx.Graph) -> None:
    split = holdout(grown, share=0.05, seed=6)
    report = evaluate_predictor(split, preferential_attachment, k=5)
    payload = experiment_payload(split, [report])
    assert payload["chapter"] == 25
    assert payload["holdout"]["method"] == "random"
    assert payload["holdout"]["positives"] == len(split.positives)
    assert payload["holdout"]["imbalance"] == pytest.approx(split.imbalance)
    assert payload["methods"][0]["method"] == "preferential attachment"
    assert payload["methods"][0]["k"] == 5
    assert len(payload["methods"][0]["roc"]) == len(report.roc_points)
    assert payload["methods"][0]["roc"][0]["threshold"] is None  # +inf is not JSON
    assert json.dumps(payload)


def test_the_guide_carries_the_link_prediction_rules() -> None:
    headings = [heading for heading, _ in READING_RULES]
    assert "Link prediction (sna predict-eval)" in headings
    names = [rule.name for rule in dict(READING_RULES)["Link prediction (sna predict-eval)"]]
    assert "A random holdout changes the network it tests" in names
    assert "AUC is flattered by imbalance: report precision@k" in names
    assert "The temporal split is the honest one" in names
    assert "Beat preferential attachment before claiming a predictor" in names


def test_the_cli_defaults_repeat_the_modules(grown: nx.Graph) -> None:
    """The CLI cannot import them at signature time, so a test holds the copies together."""
    assert (PREDICT_SHARE, PREDICT_NEGATIVES, PREDICT_K) == (
        DEFAULT_SHARE,
        DEFAULT_NEGATIVES,
        DEFAULT_K,
    )


# ---------------------------------------------------------------------------- the CLI


@pytest.fixture
def dated_cli(cli_context: AppContext, dated: InMemoryGraphStore) -> AppContext:
    """The planted dated corpus behind the CLI."""
    return cli_context


def test_sna_predict_eval_runs_the_temporal_split(dated_cli: AppContext, tmp_path: Path) -> None:
    target = tmp_path / "predict.md"
    as_json = tmp_path / "predict.json"
    result = runner.invoke(
        cli_app,
        [
            "sna",
            "predict-eval",
            DATED_PERSONA,
            "--network",
            "entities",
            "--temporal",
            SPLIT_AT,
            "--negatives",
            "3",
            "--k",
            "2",
            "--seed",
            "1",
            "--out",
            str(target),
            "--json",
            str(as_json),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "temporal holdout" in result.output
    assert "A temporal holdout at 2025-06-01" in result.output
    assert "**n.** 8 pair(s) scored: 2 positive(s)" in result.output
    assert "later pair(s) had an endpoint the training graph never saw" in result.output
    assert "Implements." in result.output

    written = target.read_text(encoding="utf-8")
    assert written.startswith("## Link prediction evaluation (temporal holdout, §25.1-25.2)")
    payload = json.loads(as_json.read_text())
    assert payload["holdout"]["split_at"] == SPLIT_AT
    assert payload["holdout"]["unseen_positives"] == 1
    assert [method["method"] for method in payload["methods"]] == [
        "preferential attachment",
        "random",
    ]


def test_sna_predict_eval_runs_the_random_split(dated_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        [
            "sna",
            "predict-eval",
            DATED_PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--share",
            "0.2",
            "--k",
            "2",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    # 8 entities, 10 co-mention pairs: 2 positives at --share 0.2, and 2 sampled negatives.
    assert "A random edge holdout (Atlas §25.1)" in result.output
    assert "**n.** 4 pair(s) scored: 2 positive(s)" in result.output
    assert "out of 28 possible" in result.output


def test_sna_predict_eval_refuses_a_split_with_no_class_to_measure(dated_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        ["sna", "predict-eval", DATED_PERSONA, "--temporal", "2099-01-01", "--seed", "1"],
    )
    assert result.exit_code == 2
    assert "need both classes" in result.output

    unknown = runner.invoke(cli_app, ["sna", "predict-eval", DATED_PERSONA, "--network", "vibes"])
    assert unknown.exit_code == 2
    assert "--network must be one of" in unknown.output
