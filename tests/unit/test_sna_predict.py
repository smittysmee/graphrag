"""Chapter 23's link prediction, held to answers worked out by hand before the code runs.

*The neighbourhood family (§23.2-23.4).* Zachary's karate club, node "0" and node "33" -- never
an edge, four common neighbours of known degree -- gives common neighbours, Adamic-Adar, resource
allocation and Jaccard their published inputs; all four are computed by hand in
:func:`test_common_neighbours_adamic_adar_resource_allocation_and_jaccard_match_a_hand_count`'s
own docstring. Preferential attachment (already in :mod:`graphrag.sna.experiment`) ranks the same
club's two hubs, node "33" (degree 17) and node "0" (degree 16), above every other non-edge.

*Katz (§23.7).* A 3-node path, the book's own single length-2 connection: Katz(0, 2) at a small
beta is beta**2 up to the higher-order walk terms the book's "paths" language glosses over (see
:mod:`graphrag.sna.predict`'s own docstring on that departure).

*HRG (§23.5).* Two 5-cliques joined by one bridge edge, one intra-clique edge deleted: the deleted
edge outranks every cross-clique pair once the score looks at the community one level above the
degenerate two-leaf grouping (see the comment in :func:`graphrag.sna.predict.hrg_scores`).

*Association rules (§23.6).* Four small document graphs -- two open wedges, two closed triangles,
all on the same three entities A, B, C -- give the triad-closure rule a confidence of 2/4 = 0.5 by
hand, and the two wedge documents are exactly where :func:`association_rule_scores` proposes the
missing edge (A, C).

*The sweep.* A planted two-community graph, held out at random: every neighbourhood, Katz and HRG
predictor beats the random baseline and beats preferential attachment, which does not (§23.1's own
counterexample -- Einstein and Curie never collaborated -- generalises: degree alone carries no
community signal here, and the report says so rather than only asserting the six that do win).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import (
    PREDICT_HRG_RESTARTS,
    PREDICT_MIN_SUPPORT,
    PREDICT_SIGN_SHARE,
    PREDICT_TOP,
)
from graphrag.cli import app as cli_app
from graphrag.embed.hashing import HashEmbedder
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import Chunk, Document, Enrichment, Entity, Mention, PersonaSpec, SourceSpec
from graphrag.personas.registry import PersonaRegistry
from graphrag.sna.cluster import hrg_fit
from graphrag.sna.experiment import Holdout, evaluate_predictor, holdout, preferential_attachment
from graphrag.sna.guide import READING_RULES
from graphrag.sna.layers import Multilayer
from graphrag.sna.predict import (
    DEFAULT_ASSOCIATION_MIN_SUPPORT,
    DEFAULT_SIGN_SHARE,
    DEFAULT_TOP,
    SignEvaluation,
    SignHoldout,
    adamic_adar,
    association_rule_scores,
    balance_score,
    classify_triangle,
    common_neighbors,
    confirming_evidence,
    cross_layer_common_neighbours,
    documents_without,
    evaluate_sign_predictor,
    hrg_scores,
    jaccard,
    katz,
    layer_correlation,
    passages_by_entity,
    predicted_sign,
    random_sign_baseline,
    rank_hypotheses,
    render_hypotheses,
    render_predicted_signs,
    render_sign_evaluation,
    resource_allocation,
    sign_holdout,
    sign_payload,
    signed_graph,
    triad_closure_rule,
)
from graphrag.sna.stances import SignedPair
from tests.legendary import karate_club

runner = CliRunner()


def _karate() -> nx.Graph:
    graph, _ = karate_club()
    return nx.relabel_nodes(graph, str)


# ------------------------------------------------------------- §23.2-23.4 the neighbourhood family


def test_common_neighbours_adamic_adar_resource_allocation_and_jaccard_match_a_hand_count() -> None:
    """Node "0" (degree 16) and node "33" (degree 17) share exactly four common neighbours:
    "8" (degree 5), "13" (degree 5), "19" (degree 3) and "31" (degree 6). By hand:

    - common neighbours: 4.
    - Adamic-Adar: 2/ln(5) + 1/ln(3) + 1/ln(6) = 2(0.621335) + 0.910239 + 0.558111 = 2.711020.
    - resource allocation: 2/5 + 1/3 + 1/6 = 0.4 + 0.333333 + 0.166667 = 0.9.
    - Jaccard: 4 / (16 + 17 - 4) = 4/29 = 0.137931.
    """
    graph = _karate()
    assert not graph.has_edge("0", "33")
    assert set(graph.neighbors("0")) & set(graph.neighbors("33")) == {"8", "13", "19", "31"}
    assert dict(graph.degree())["0"] == 16
    assert dict(graph.degree())["33"] == 17

    assert common_neighbors(graph)["0", "33"] == 4.0
    assert adamic_adar(graph)["0", "33"] == pytest.approx(2.711020, abs=1e-5)
    assert resource_allocation(graph)["0", "33"] == pytest.approx(0.9, abs=1e-9)
    assert jaccard(graph)["0", "33"] == pytest.approx(4 / 29, abs=1e-9)


def test_common_neighbour_scores_are_zero_with_no_shared_neighbour() -> None:
    path = nx.relabel_nodes(nx.path_graph(4), str)  # 0-1-2-3
    assert common_neighbors(path)["0", "3"] == 0.0
    assert adamic_adar(path)["0", "3"] == 0.0
    assert resource_allocation(path)["0", "3"] == 0.0
    assert jaccard(path)["0", "3"] == 0.0


def test_preferential_attachment_ranks_the_two_karate_hubs_first() -> None:
    """§23.1's own worked example, on real data: node "33" (17) and node "0" (16) are the two
    highest-degree members, never joined, and no other non-edge beats their degree product."""
    graph = _karate()
    scores = preferential_attachment(graph)
    best = max(iter(scores), key=lambda pair: scores[pair])
    assert best == ("0", "33")
    assert scores[best] == 16 * 17 == 272


# --------------------------------------------------------------------------------- §23.7 Katz


def test_katz_of_a_single_length_two_connection_is_beta_squared_to_leading_order() -> None:
    """A 3-node path 0-1-2: exactly one path of length 2 between the ends, none of length 1.
    lambda_max = sqrt(2) (an eigenvalue of the path's adjacency matrix), so beta = 0.05 is
    comfortably under the 1/sqrt(2) = 0.7071 convergence bound. The closed form sums *walks*,
    not simple paths (see the module docstring), so the exact value is 0.00251256..., about
    0.5% over beta**2 = 0.0025 -- within 1% relative, which is what is asserted.
    """
    path = nx.relabel_nodes(nx.path_graph(3), str)
    scores = katz(path, beta=0.05)
    assert scores["0", "2"] == pytest.approx(0.05**2, rel=0.01)
    # "0"-"1" is already an edge, so it is not a candidate score at all (§25.1's own tenet).
    with pytest.raises(KeyError):
        scores["0", "1"]


def test_katz_refuses_a_beta_at_or_above_the_convergence_bound() -> None:
    path = nx.relabel_nodes(nx.path_graph(3), str)
    with pytest.raises(ValueError, match="must be below 1/lambda_max"):
        katz(path, beta=0.75)  # bound is 1/sqrt(2) = 0.7071...
    with pytest.raises(ValueError, match="must be positive"):
        katz(path, beta=0.0)


def test_katz_default_beta_is_half_the_convergence_bound() -> None:
    path = nx.relabel_nodes(nx.path_graph(3), str)
    scores = katz(path)  # default beta
    # Should not raise, and should differ from the beta=0.05 table above.
    assert scores["0", "2"] != pytest.approx(0.05**2, rel=0.01)


# ----------------------------------------------------------------------------------- §23.5 HRG


@pytest.fixture(scope="module")
def two_cliques() -> nx.Graph:
    """Two 5-cliques joined by one bridge edge, with one intra-clique edge of the first missing:
    "a0"-"a1" is the hypothesis, and every "a*"-"b*" pair is the cross-clique alternative."""
    graph = nx.Graph()
    a = [f"a{i}" for i in range(5)]
    b = [f"b{i}" for i in range(5)]
    graph.add_nodes_from(a + b)
    for i, u in enumerate(a):
        for v in a[i + 1 :]:
            if {u, v} != {"a0", "a1"}:
                graph.add_edge(u, v)
    for i, u in enumerate(b):
        for v in b[i + 1 :]:
            graph.add_edge(u, v)
    graph.add_edge("a4", "b0")
    return graph


def test_hrg_ranks_the_missing_intra_clique_edge_above_every_cross_clique_pair(
    two_cliques: nx.Graph,
) -> None:
    fit = hrg_fit(two_cliques, seed=1, restarts=8, samples=6000)
    scores = hrg_scores(two_cliques, fit=fit)
    missing = scores["a0", "a1"]
    a_nodes = [f"a{i}" for i in range(5)]
    b_nodes = [f"b{i}" for i in range(5)]
    cross = [scores[u, v] for u in a_nodes for v in b_nodes if not two_cliques.has_edge(u, v)]
    assert missing > max(cross)
    # Every cross-clique pair's lowest common ancestor is the whole network (root), whose
    # observed-over-expected ratio is always exactly 1 (see hrg_scores's own docstring).
    assert cross == pytest.approx([1.0] * len(cross))


def test_hrg_refuses_a_fit_from_a_different_node_set(two_cliques: nx.Graph) -> None:
    other = nx.relabel_nodes(nx.path_graph(4), str)
    fit = hrg_fit(other, seed=1, restarts=1, samples=200)
    with pytest.raises(ValueError, match="node set this graph does not hold"):
        hrg_scores(two_cliques, fit=fit)


# --------------------------------------------------------------------- §23.6 association rules


def _wedge(center: str, left: str, right: str) -> nx.Graph:
    return nx.Graph([(center, left), (center, right)])


def _triangle(a: str, b: str, c: str) -> nx.Graph:
    return nx.Graph([(a, b), (b, c), (a, c)])


def test_the_triad_closure_confidence_is_exactly_the_hand_computed_fraction() -> None:
    """Documents: two open wedges A-B-C (B is the centre) and two closed triangles A-B-C.

    Wedge support: all four documents hold the two-edge path (a triangle contains it too), so
    support(wedge) = 4. Triangle support: only the two closed documents, so support(triangle) =
    2. Confidence = 2/4 = 0.5 by hand -- "2 of the 4 documents where {A, B} and {B, C} are both
    named also name {A, C}".
    """
    documents = [
        _wedge("B", "A", "C"),
        _wedge("B", "A", "C"),
        _triangle("A", "B", "C"),
        _triangle("A", "B", "C"),
    ]
    pooled = nx.Graph([("A", "B"), ("B", "C")])  # A-C is not (yet) an edge of the pooled network
    rule = triad_closure_rule(documents, pooled, min_support=2)
    assert rule.support_antecedent == 4
    assert rule.support_consequent == 2
    assert rule.confidence == pytest.approx(0.5)
    assert rule.transitivity == 0.0  # the pooled network (a bare wedge) has no triangle at all
    assert rule.beats_transitivity


def test_association_rule_scores_proposes_the_missing_edge_from_the_open_wedges_only() -> None:
    documents = [
        _wedge("B", "A", "C"),
        _wedge("B", "A", "C"),
        _triangle("A", "B", "C"),
        _triangle("A", "B", "C"),
    ]
    pooled = nx.Graph([("A", "B"), ("B", "C")])
    scores = association_rule_scores(pooled, documents, min_support=2)
    assert scores["A", "C"] == pytest.approx(0.5)


def test_association_rule_scores_is_undefined_confidence_when_nothing_clears_min_support() -> None:
    documents = [_wedge("B", "A", "C")]  # a single document: nothing to generalise a rule from
    pooled = nx.Graph([("A", "B"), ("B", "C")])
    rule = triad_closure_rule(documents, pooled, min_support=2)
    assert math.isnan(rule.confidence)
    scores = association_rule_scores(pooled, documents, min_support=2)
    assert scores["A", "C"] == 0.0


def test_documents_without_strips_a_held_out_pair_from_every_document_that_has_it() -> None:
    documents = [_triangle("A", "B", "C"), nx.Graph([("D", "E")])]
    trimmed = documents_without(documents, frozenset({("A", "C")}))
    assert not trimmed[0].has_edge("A", "C")
    assert trimmed[0].has_edge("A", "B")  # untouched
    assert trimmed[1].has_edge("D", "E")  # untouched, no such pair here
    assert documents_without(documents, frozenset()) == documents


# --------------------------------------------------------------------- the sweep, planted graph


@pytest.fixture(scope="module")
def planted_communities() -> nx.Graph:
    """Two dense 15-node communities, sparsely cross-joined: community, not degree, predicts
    an edge here, which is exactly the case §23.1 says defeats preferential attachment."""
    return nx.relabel_nodes(nx.planted_partition_graph(2, 15, 0.6, 0.05, seed=3), lambda n: f"n{n}")


def test_every_neighbourhood_katz_and_hrg_predictor_beats_random_and_beats_pa(
    planted_communities: nx.Graph,
) -> None:
    split = holdout(planted_communities, share=0.1, seed=1)
    pa_report = evaluate_predictor(split, preferential_attachment, k=10)
    scorers = {
        "cn": common_neighbors,
        "aa": adamic_adar,
        "ra": resource_allocation,
        "jaccard": jaccard,
        "katz": lambda g: katz(g),
        "hrg": lambda g: hrg_scores(g, restarts=3, samples=3000, seed=1),
    }
    for name, scorer in scorers.items():
        report = evaluate_predictor(split, scorer, name=name, k=10)
        assert report.auc > 0.5, f"{name} did not beat the random baseline"
        assert report.auc > pa_report.auc, f"{name} did not beat preferential attachment"
    # Preferential attachment itself is the counterexample the chapter predicts: degree carries
    # no community signal on this graph, so it need not (and here does not) beat random.
    assert pa_report.auc < 0.5


# ---------------------------------------------------------------------- confirming passages


def test_confirming_evidence_prefers_a_shared_passage_over_a_bridge_over_nothing() -> None:
    graph = nx.Graph([("u", "z"), ("z", "v")])
    passages = {
        "u": {"c-uv": "d1", "c-uz": "d2"},
        "v": {"c-uv": "d1", "c-zv": "d3"},
        "z": {"c-uz": "d2", "c-zv": "d3"},
    }
    shared = confirming_evidence(graph, passages, "u", "v")
    assert shared.kind == "co-mentioned"
    assert shared.passages[0].chunk_id == "c-uv"

    bridge_only = confirming_evidence(
        graph, {"u": {"c-uz": "d2"}, "v": {"c-zv": "d3"}, "z": passages["z"]}, "u", "v"
    )
    assert bridge_only.kind == "bridge"
    assert bridge_only.bridge == "z"
    assert {ref.chunk_id for ref in bridge_only.passages} == {"c-uz", "c-zv"}

    lonely = nx.Graph()
    lonely.add_nodes_from(["u", "v"])
    nothing = confirming_evidence(lonely, {"u": {"c1": "d1"}, "v": {"c2": "d2"}}, "u", "v")
    assert nothing.kind == "none"
    assert nothing.bridge is None
    assert {ref.chunk_id for ref in nothing.passages} == {"c1", "c2"}


def test_passages_by_entity_groups_chunk_ids_by_entity() -> None:
    from graphrag.models import EntityChunk

    rows = [
        EntityChunk(entity_id="e1", name="E1", chunk_id="c1", doc_id="d1"),
        EntityChunk(entity_id="e1", name="E1", chunk_id="c2", doc_id="d1"),
        EntityChunk(entity_id="e2", name="E2", chunk_id="c1", doc_id="d1"),
    ]
    grouped = passages_by_entity(rows)
    assert grouped == {"e1": {"c1": "d1", "c2": "d1"}, "e2": {"c1": "d1"}}


# ------------------------------------------------------------------------------ rank_hypotheses


def test_rank_hypotheses_returns_the_top_n_above_zero_with_deterministic_ties() -> None:
    graph = nx.Graph([("a", "z"), ("z", "b"), ("a", "y"), ("y", "c")])
    scores = common_neighbors(graph)
    passages: dict[str, dict[str, str]] = {}
    ranked = rank_hypotheses(
        scores, graph, passages, method="common neighbours", section="§23.2", top=1
    )
    assert len(ranked) == 1
    assert ranked[0].score > 0
    assert ranked[0].method == "common neighbours"
    assert ranked[0].section == "§23.2"


def test_render_hypotheses_says_nothing_scored_when_the_list_is_empty() -> None:
    text = render_hypotheses((), frame="a frame", null_model="a null")
    assert "No pair scored above the threshold" in text
    assert "**Sampling frame.** a frame" in text
    assert "**Null model.** a null" in text


# ------------------------------------------------------------------------------------- the guide


def test_the_guide_carries_the_chapter_23_rules() -> None:
    headings = [heading for heading, _ in READING_RULES]
    assert "Link prediction, simple graphs (sna predict)" in headings
    names = [
        rule.name for rule in dict(READING_RULES)["Link prediction, simple graphs (sna predict)"]
    ]
    assert "Every score here predicts an old-old link" in names
    assert "Katz's beta must stay under 1/lambda_max" in names
    assert "HRG scores are a Monte Carlo estimate, refit every call" in names
    assert "An association rule's confidence needs a null: the network's own transitivity" in names


def test_the_guide_carries_the_chapter_24_rules() -> None:
    heading = "Signed and multilayer prediction (sna predict --signed / --layers)"
    headings = [h for h, _ in READING_RULES]
    assert heading in headings
    names = [rule.name for rule in dict(READING_RULES)[heading]]
    assert "A sign prediction and a link prediction are different questions" in names
    assert "Balance theory abstains; that is not a neutral reading" in names
    assert "Status theory needs a direction this corpus does not have" in names
    assert "The all-negative triangle reads as unbalanced here" in names
    assert (
        "The cross-layer weighting is an inter-layer correlation, not §9.1's layer relevance"
        in (names)
    )


def test_the_cli_defaults_repeat_the_module() -> None:
    assert PREDICT_TOP == DEFAULT_TOP
    assert PREDICT_MIN_SUPPORT == DEFAULT_ASSOCIATION_MIN_SUPPORT
    assert PREDICT_SIGN_SHARE == DEFAULT_SIGN_SHARE
    import inspect

    assert inspect.signature(hrg_fit).parameters["restarts"].default == PREDICT_HRG_RESTARTS


# ------------------------------------------------------------------------------------- the CLI

PREDICT_PERSONA = "test-predict"
PREDICT_SOURCE = "notes"

#: A small graph on 8 entities with every edge repeated in two documents, so it clears the
#: entity network's default --min-weight of 2. e0 and e2 share no edge and no document, but each
#: shares a document with e1, which is the bridge common-neighbour evidence should point at.
PREDICT_EDGES: tuple[tuple[str, str], ...] = (
    ("e0", "e1"),
    ("e1", "e2"),
    ("e2", "e3"),
    ("e3", "e4"),
    ("e4", "e5"),
    ("e5", "e6"),
    ("e6", "e7"),
)

#: Both mentions of a tagged edge's chunk carry the same stance, which is what makes the pair a
#: signed co-mention (§24.1): (e0, e1) and (e2, e3) praised, (e1, e2) and (e3, e4) complained
#: about, an alternating-sign path long enough for `--signed`'s holdout to have something to hide.
PREDICT_STANCES: dict[tuple[str, str], str] = {
    ("e0", "e1"): "praise",
    ("e1", "e2"): "complaint",
    ("e2", "e3"): "praise",
    ("e3", "e4"): "complaint",
}


@pytest.fixture
def predict_store(
    memory_store: InMemoryGraphStore, hash_embedder: HashEmbedder, registry: PersonaRegistry
) -> InMemoryGraphStore:
    registry.save(
        PersonaSpec(
            id=PREDICT_PERSONA,
            name="Test Predict",
            role_prompt="You answer from test notes.",
            sources=[SourceSpec(id=PREDICT_SOURCE, kind="local", path="notes", loader="documents")],
        )
    )
    documents: list[Document] = []
    chunks: list[Chunk] = []
    mentions: list[Mention] = []
    entities: dict[str, Entity] = {}
    ordinal = 0
    for u, v in PREDICT_EDGES:
        for copy in range(2):  # two passages per edge, to clear --min-weight 2
            doc_id = f"{PREDICT_PERSONA}:{PREDICT_SOURCE}:{u}-{v}-{copy}"
            text = f"A note naming {u} and {v}."
            documents.append(
                Document(
                    id=doc_id,
                    persona_id=PREDICT_PERSONA,
                    source_id=PREDICT_SOURCE,
                    title=doc_id,
                    path=f"notes/{doc_id}.md",
                )
            )
            chunk_id = f"{doc_id}#0"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    persona_id=PREDICT_PERSONA,
                    ordinal=ordinal,
                    text=text,
                )
            )
            ordinal += 1
            stance = PREDICT_STANCES.get((u, v))
            for entity_id in (u, v):
                entities.setdefault(entity_id, Entity(id=entity_id, name=entity_id, type="concept"))
                mentions.append(
                    Mention(chunk_id=chunk_id, entity_id=entity_id, stance=stance)  # type: ignore[arg-type]
                )
    memory_store.upsert_documents(documents)
    memory_store.upsert_chunks(chunks, hash_embedder.embed_documents([c.text for c in chunks]))
    memory_store.upsert_enrichment(Enrichment(entities=list(entities.values()), mentions=mentions))
    return memory_store


@pytest.fixture
def predict_cli(cli_context: AppContext, predict_store: InMemoryGraphStore) -> AppContext:
    return cli_context


def test_sna_predict_runs_common_neighbours_end_to_end(predict_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        ["sna", "predict", PREDICT_PERSONA, "--method", "cn", "--top", "3", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Link prediction evaluation (common neighbours, ch. 25)" in result.output
    assert "Link prediction hypotheses (common neighbours, ch. 23)" in result.output
    assert "Nothing here is written to the graph" in result.output


def test_sna_predict_rejects_an_unknown_method(predict_cli: AppContext) -> None:
    result = runner.invoke(cli_app, ["sna", "predict", PREDICT_PERSONA, "--method", "bogus"])
    assert result.exit_code == 2
    assert "--method must be one of" in result.output


def test_sna_predict_writes_json_hypotheses(predict_cli: AppContext, tmp_path: Path) -> None:
    out_json = tmp_path / "hyp.json"
    result = runner.invoke(
        cli_app,
        [
            "sna",
            "predict",
            PREDICT_PERSONA,
            "--method",
            "aa",
            "--top",
            "2",
            "--seed",
            "1",
            "--json",
            str(out_json),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_json.read_text())
    # ATL-ENT-4: wrapped under "hypotheses" so the payload has a "provenance" key like every
    # other command's, since `hypotheses_payload` on its own is a list, not a dict.
    assert "provenance" in payload
    hypotheses = payload["hypotheses"]
    assert isinstance(hypotheses, list)
    if hypotheses:
        assert "pair" in hypotheses[0]
        assert "evidence" in hypotheses[0]


# ================================================================ ch. 24: signed and multilayer


def _ml(graphs: dict[str, nx.Graph]) -> Multilayer:
    """A :class:`Multilayer` built by hand, in the same spirit as ``test_sna_multilayer.py``'s
    own helper: every answer here is worked out with a pencil before the code runs."""
    nodes = tuple(sorted({node for graph in graphs.values() for node in graph.nodes}))
    return Multilayer(
        persona_id="test",
        network="entities",
        layering="stance",
        names=tuple(graphs),
        graphs=graphs,
        nodes=nodes,
        omega=1.0,
        frame="a hand-planted network for testing",
    )


# --------------------------------------------------------------- §24.1 social balance theory


def test_classify_triangle_covers_all_four_of_figure_24_1s_types() -> None:
    """(a) all positive and (b) one positive are balanced (an odd count of positive edges, or
    equivalently an even count of negative ones); (d) two positive is not; (c) all negative is
    not, under the classical rule this module implements (p. 345-346 notes real corpora sometimes
    read it as neutral or balanced instead -- not modelled here, see the guide's own caveat)."""
    a = classify_triangle((1, 1, 1))
    assert a.negatives == 0 and a.balanced

    b = classify_triangle((1, -1, -1))
    assert b.negatives == 2 and b.balanced

    d = classify_triangle((1, 1, -1))
    assert d.negatives == 1 and not d.balanced

    c = classify_triangle((-1, -1, -1))
    assert c.negatives == 3 and not c.balanced


def test_classify_triangle_refuses_anything_but_three_signed_edges() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        classify_triangle((1, 1))
    with pytest.raises(ValueError, match="\\+1 or -1"):
        classify_triangle((1, 1, 0))


def test_balance_score_always_proposes_the_sign_that_would_balance_the_triangle() -> None:
    """§24.1's own three worked cases: two positive legs predict positive, mixed legs predict
    negative, two negative legs predict positive (the book's stated preference over an
    all-negative close, p. 347)."""
    two_positive = nx.Graph()
    two_positive.add_edge("u", "z", sign=1)
    two_positive.add_edge("z", "v", sign=1)
    assert predicted_sign(balance_score(two_positive, "u", "v")) == 1

    mixed = nx.Graph()
    mixed.add_edge("u", "z", sign=1)
    mixed.add_edge("z", "v", sign=-1)
    assert predicted_sign(balance_score(mixed, "u", "v")) == -1

    two_negative = nx.Graph()
    two_negative.add_edge("u", "z", sign=-1)
    two_negative.add_edge("z", "v", sign=-1)
    assert predicted_sign(balance_score(two_negative, "u", "v")) == 1


def test_balance_score_abstains_with_no_signed_common_neighbour() -> None:
    lonely = nx.Graph([("u", "v")])  # no common neighbour at all
    assert balance_score(lonely, "u", "v") == 0.0
    assert predicted_sign(balance_score(lonely, "u", "v")) is None
    assert balance_score(lonely, "u", "nowhere") == 0.0  # not even in the graph


def test_signed_graph_folds_dominant_sign_and_reports_the_ties_it_dropped() -> None:
    pairs = (
        SignedPair(left="a", right="b", shared={"praise": 2, "complaint": 0}),
        SignedPair(left="b", right="c", shared={"praise": 1, "complaint": 1}),  # tie: ambiguous
        SignedPair(left="c", right="d", shared={"complaint": 3}),
    )
    graph = signed_graph(pairs)
    assert graph["a"]["b"]["sign"] == 1
    assert not graph.has_edge("b", "c")
    assert graph["c"]["d"]["sign"] == -1
    assert graph.graph["ambiguous"] == 1


def test_random_sign_baseline_converges_to_one_half_over_many_hidden_edges() -> None:
    """A coin flip's own expectation, measured (§25.2's convention) rather than assumed: over
    400 hidden edges the empirical accuracy sits close to 0.5 whatever their true signs are."""
    hidden = tuple((f"n{i}", f"m{i}", 1 if i % 2 == 0 else -1) for i in range(400))
    baseline = random_sign_baseline(hidden, seed=7)
    assert abs(baseline - 0.5) < 0.1


def test_sign_holdout_on_a_fully_balanced_planted_network_scores_perfect_accuracy() -> None:
    """Two 6-node factions, complete graph, positive within a faction and negative across --
    Harary's own clusterable construction, whose every triangle is balanced by hand: two same-
    faction edges are positive-positive (closes type (a)); a same/cross pair is positive-negative
    (closes type (b)); two cross-faction edges are negative-negative (closes type (b) too, not
    the all-negative type (c), since two same-faction nodes are never on the "far" side of both).
    With this much redundancy, holding out even a third of the signs leaves enough visible
    triangles that balance theory recovers every one of them.
    """
    faction_a = [f"a{i}" for i in range(6)]
    faction_b = [f"b{i}" for i in range(6)]
    nodes = faction_a + faction_b
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            same_faction = (u in faction_a) == (v in faction_a)
            graph.add_edge(u, v, sign=1 if same_faction else -1)

    split = sign_holdout(graph, share=0.3, seed=1)
    evaluation = evaluate_sign_predictor(split, seed=1)
    assert evaluation.abstained == 0
    assert evaluation.accuracy == pytest.approx(1.0)
    assert evaluation.random_accuracy < 0.9  # a coin flip, not a second balance-theory reading


def test_evaluate_sign_predictor_counts_an_abstention_separately_from_a_wrong_answer() -> None:
    train = nx.Graph()
    train.add_edge("u", "v")  # the sign was hidden; the edge itself was never removed
    train.add_edge("u", "w", sign=1)  # signed, but shares no path back to v
    split = SignHoldout(train=train, hidden=(("u", "v", 1),), share=0.5, seed=None, frame="test")
    evaluation = evaluate_sign_predictor(split)
    assert evaluation.abstained == 1
    assert evaluation.correct == 0
    assert evaluation.incorrect == 0
    assert math.isnan(evaluation.accuracy)
    assert evaluation.coverage == 0.0


def test_sign_holdout_refuses_fewer_than_two_signed_edges() -> None:
    with pytest.raises(ValueError, match="at least 2 signed edges"):
        sign_holdout(nx.Graph([("u", "v", {"sign": 1})]))


def test_render_sign_evaluation_and_payload_carry_frame_null_model_and_chapter() -> None:
    split = SignHoldout(
        train=nx.Graph(),
        hidden=(("a", "b", 1),),
        share=0.5,
        seed=1,
        frame="a hand-built sign frame",
    )
    evaluation = SignEvaluation(
        method="balance theory",
        section="§24.1",
        correct=1,
        incorrect=0,
        abstained=0,
        total=1,
        accuracy=1.0,
        coverage=1.0,
        random_accuracy=0.5,
    )
    text = render_sign_evaluation(split, evaluation)
    assert "**Sampling frame.** a hand-built sign frame" in text
    assert "**Null model.**" in text
    assert "Atlas §24.1" in text
    payload = sign_payload(evaluation)
    assert payload["chapter"] == 24
    assert payload["accuracy"] == 1.0
    assert payload["beats_random"] is True


def test_render_predicted_signs_labels_each_hypothesis() -> None:
    scores = common_neighbors(nx.Graph([("a", "z"), ("z", "b")]))
    passages: dict[str, dict[str, str]] = {}
    hypotheses = rank_hypotheses(scores, scores.graph, passages, method="cn", section="§23.2")
    signed = nx.Graph()
    signed.add_edge("a", "z", sign=1)
    signed.add_edge("z", "b", sign=1)
    text = render_predicted_signs(hypotheses, signed)
    assert "a -- b" in text
    assert "positive (praise-like)" in text


# --------------------------------------------------- §24.2 generalized multilayer prediction


def test_layer_correlation_matches_a_hand_computed_pearson_coefficient() -> None:
    """l1 = {u-z, z-v, a-b, b-c, a-c} (5 edges), l2 = {u-v, a-b, b-c, a-c} (4 edges), over the 15
    possible pairs of 6 nodes: 3 edges shared, so E[XY]=3/15, E[X]=5/15, E[Y]=4/15, giving
    cov=1/9, var(l1)=2/9, var(l2)=44/225 and r = (1/9) / sqrt(2/9 * 44/225) = 0.5330.
    """
    l1 = nx.Graph([("u", "z"), ("z", "v"), ("a", "b"), ("b", "c"), ("a", "c")])
    l2 = nx.Graph([("u", "v"), ("a", "b"), ("b", "c"), ("a", "c")])
    nodes = ("u", "v", "z", "a", "b", "c")
    assert layer_correlation(l2, l1, nodes) == pytest.approx(0.5330, abs=1e-3)


def test_layer_correlation_is_undefined_for_an_empty_or_a_complete_layer() -> None:
    nodes = ("a", "b", "c")
    empty = nx.Graph()
    empty.add_nodes_from(nodes)
    some = nx.Graph([("a", "b")])
    assert math.isnan(layer_correlation(empty, some, nodes))


def test_cross_layer_common_neighbours_recovers_what_the_single_layer_score_misses() -> None:
    """l1 bridges u and v through z but never states the edge itself; l2 states it directly and
    agrees with l1 on an unrelated a-b-c triangle. Holding out l2's (u, v) leaves u and v
    isolated in l2's own training graph -- the single-layer scorer has nothing -- while l1 still
    has the bridge, so the cross-layer score is positive there and exactly 0.0 for every one of
    the 11 other candidate pairs (worked out by hand: none of them share a common neighbour in
    either layer), making the contrast exact rather than a matter of which negatives were drawn.
    """
    l1 = nx.Graph([("u", "z"), ("z", "v"), ("a", "b"), ("b", "c"), ("a", "c")])
    l2 = nx.Graph([("u", "v"), ("a", "b"), ("b", "c"), ("a", "c")])
    ml = _ml({"l1": l1, "l2": l2})

    train = l2.copy()
    train.remove_edge("u", "v")
    negatives = tuple(
        sorted(
            (u, v) if u <= v else (v, u)
            for u in ml.nodes
            for v in ml.nodes
            if u < v and not train.has_edge(u, v) and {u, v} != {"u", "v"}
        )
    )
    assert len(negatives) == 11
    split = Holdout(
        train=train,
        positives=(("u", "v"),),
        negatives=negatives,
        method="random",
        seed=1,
        share=0.25,
        requested_share=0.25,
        possible_pairs=15,
        observed_edges=4,
        frame="hand-built for the cross-layer test",
    )

    cross_report = evaluate_predictor(
        split, lambda t: cross_layer_common_neighbours(ml, "l2", t), k=1
    )
    single_report = evaluate_predictor(split, common_neighbors, k=1)
    assert cross_report.auc == pytest.approx(1.0)
    assert single_report.auc == pytest.approx(0.5)


def test_cross_layer_scoring_falls_back_to_the_single_layer_score_when_uncorrelated() -> None:
    """§24.2's own two-sided caution made concrete: a layer with no information about the target
    (here, no edges at all) contributes nothing, so the blended score equals the plain one."""
    target = nx.Graph([("a", "b"), ("b", "c")])
    empty = nx.Graph()
    empty.add_nodes_from(["a", "b", "c"])
    ml = _ml({"target": target, "empty": empty})
    single = common_neighbors(target)
    cross = cross_layer_common_neighbours(ml, "target")
    assert cross["a", "c"] == single["a", "c"] == 1.0


def test_cross_layer_common_neighbours_refuses_an_unknown_layer() -> None:
    ml = _ml({"only": nx.Graph([("a", "b")])})
    with pytest.raises(ValueError, match="not a layer"):
        cross_layer_common_neighbours(ml, "missing")


# ------------------------------------------------------------------------------------- the CLI


def test_sna_predict_signed_prints_the_balance_theory_section(predict_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        ["sna", "predict", PREDICT_PERSONA, "--method", "cn", "--signed", "--seed", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Signed link prediction (balance theory)" in result.output
    assert "Predicted signs for the ranked hypotheses" in result.output


def test_sna_predict_layers_runs_the_multilayer_report(predict_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        [
            "sna",
            "predict",
            PREDICT_PERSONA,
            "--layers",
            "stance",
            "--target-layer",
            "praise",
            "--seed",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Multilayer link prediction (praise, Atlas §24.2)" in result.output


def test_sna_predict_layers_rejects_an_unknown_target(predict_cli: AppContext) -> None:
    result = runner.invoke(
        cli_app,
        ["sna", "predict", PREDICT_PERSONA, "--layers", "stance", "--target-layer", "bogus"],
    )
    assert result.exit_code == 2
    assert "--target-layer must be one of" in result.output
