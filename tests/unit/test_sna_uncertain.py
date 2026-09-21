"""Chapter 28 over the corpus networks: edge probabilities, possible worlds, and what breaks.

The known answers here come from three places. The chapter's own opening example -- *"we
actually don't know whether it has 77 or 78 edges"* (p. 396) -- is a number: the karate club with
its one disputed edge at ``p = 0.5`` has 77.5 edges in expectation and the two endpoints of that
edge have an exactly two-valued degree distribution. The book's Figure 28.4 network is four nodes
with hand-written probabilities, so its expected degrees and its degree PMF can be read off the
page. And the pricing rule itself is arithmetic stated in :mod:`graphrag.sna.uncertain`, so a
planted store whose mention tiers are known produces edge probabilities that can be written down
before the code runs.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import networkx as nx
import pytest
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import UNCERTAIN_SAMPLES, app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.models import MENTION_TIERS, Chunk, Document, Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport
from graphrag.sna.analysis import run_analysis
from graphrag.sna.export import build_network, entity_co_mention, speaker_co_participation
from graphrag.sna.measures import CHEAP_CENTRALITIES, centrality
from graphrag.sna.uncertain import (
    EDGE_PROBABILITY,
    SAMPLES,
    STATED_PROBABILITY,
    TIER_PROBABILITY,
    combine,
    degree_distribution,
    edge_probabilities,
    evidence_report,
    expectation,
    expected_degree,
    expected_density,
    expected_edges,
    node_expectation,
    realisations,
    reliability,
    render_evidence,
    render_uncertain,
    support,
    tier_probability,
    uncertain_payload,
    uncertainty_report,
)
from tests.legendary import karate_club

runner = CliRunner()

#: A row of one of the report's ranked-with-interval tables, which every centrality gets.
RANKED_ROW = re.compile(r"^\| \d+ \| .+ \| .+ \| .+ ± \[.+\] \| (yes|no|-) \|$")

PERSONA = "test-uncertain"
SOURCE = "posts"


def _plant(store: InMemoryGraphStore, mentions: dict[str, list[tuple[str, str]]]) -> None:
    """One document per key, one passage each, with the mentions and tiers the test names.

    Written straight into the store rather than imported, for the reason the ``layered`` fixture
    gives: a test about how a tier is *priced* must not also depend on what the matcher would
    have made of some invented prose.
    """
    documents = []
    chunks = []
    rows: list[Mention] = []
    entities: dict[str, Entity] = {}
    for ordinal, (slug, pairs) in enumerate(sorted(mentions.items())):
        doc_id = f"{PERSONA}:{SOURCE}:{slug}"
        documents.append(
            Document(
                id=doc_id,
                persona_id=PERSONA,
                source_id=SOURCE,
                title=slug,
                path=f"posts/{slug}.md",
                speakers=["ana"],
            )
        )
        chunks.append(
            Chunk(
                id=f"{doc_id}#0",
                doc_id=doc_id,
                persona_id=PERSONA,
                ordinal=ordinal,
                text=f"passage {slug}",
                speaker="ana",
                speakers=["ana"],
            )
        )
        for entity_id, tier in pairs:
            entities.setdefault(
                entity_id,
                Entity(id=entity_id, name=entity_id.split(":")[-1].title(), type="product"),
            )
            rows.append(
                Mention(chunk_id=f"{doc_id}#0", entity_id=entity_id, tier=tier)  # type: ignore[arg-type]
            )
    store.upsert_documents(documents)
    store.upsert_chunks(chunks, [[0.0] * 64 for _ in chunks])
    store.upsert_enrichment(Enrichment(entities=list(entities.values()), mentions=rows))


# --------------------------------------------------------------------------- the rule


def test_the_tiers_are_ordered_strongest_first_and_priced_that_way() -> None:
    """The ordering is a contract: a Cypher statement in the Neo4j store depends on it.

    ``merge_entities`` picks the stronger of two tiers with ``m.tier < n.tier``, which works only
    because the four tiers sort strongest-first alphabetically. If a fifth tier is ever added
    out of that order, this fails before the store silently starts keeping the weaker one.
    """
    assert MENTION_TIERS == ("exact", "loose", "none", "unknown")
    assert list(MENTION_TIERS) == sorted(MENTION_TIERS)
    priced = [TIER_PROBABILITY[tier] for tier in ("exact", "unknown", "loose", "none")]
    assert priced == sorted(priced, reverse=True)
    assert all(0.0 < value <= 1.0 for value in priced)


def test_evidence_combines_the_way_the_chapter_assumes() -> None:
    """Within a passage the weakest link; across passages independent (§28.2, p. 400)."""
    assert support(0.95, 0.6) == 0.6
    assert support() == 1.0
    assert combine([]) == 0.0
    assert combine([0.6]) == pytest.approx(0.6)
    assert combine([0.5, 0.5]) == pytest.approx(0.75)
    assert combine([0.95, 0.95, 1.0]) == 1.0
    assert tier_probability("nonsense") == TIER_PROBABILITY["unknown"]  # type: ignore[arg-type]


def test_a_loose_edge_is_priced_below_an_exact_one_and_two_passages_above_both(
    memory_store: InMemoryGraphStore,
) -> None:
    """The ticket's own acceptance test, with the arithmetic written out.

    Alpha-Beta is named in two passages that both matched verbatim, Alpha-Gamma in one that
    matched verbatim, and Alpha-Delta in one that matched only on a token.
    """
    exact = TIER_PROBABILITY["exact"]
    loose = TIER_PROBABILITY["loose"]
    _plant(
        memory_store,
        {
            "post-1": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-2": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-3": [("p:alpha", "exact"), ("p:gamma", "exact")],
            "post-4": [("p:alpha", "exact"), ("p:delta", "loose")],
        },
    )
    graph = entity_co_mention(memory_store, PERSONA, min_weight=1)
    p = edge_probabilities(graph)

    assert graph["p:alpha"]["p:delta"][EDGE_PROBABILITY] == pytest.approx(loose)
    assert graph["p:alpha"]["p:gamma"][EDGE_PROBABILITY] == pytest.approx(exact)
    # two exact passages: 1 - (1 - p_exact)^2, which is the rule and is 0.9975
    assert graph["p:alpha"]["p:beta"][EDGE_PROBABILITY] == pytest.approx(1 - (1 - exact) ** 2)
    assert p[("p:alpha", "p:delta")] < p[("p:alpha", "p:gamma")] < p[("p:alpha", "p:beta")]
    # and the weights are exactly what they always were: pricing changed no other number
    assert graph["p:alpha"]["p:beta"]["weight"] == 2
    assert graph["p:alpha"]["p:delta"]["weight"] == 1


def test_an_untiered_corpus_is_priced_at_the_documented_placeholder(
    layered: InMemoryGraphStore,
) -> None:
    """Every committed snapshot predates the tier, so this is the case that matters in practice."""
    graph = entity_co_mention(layered, "test-layers", min_weight=1)
    unknown = TIER_PROBABILITY["unknown"]
    assert graph["product:alpha"]["product:gamma"][EDGE_PROBABILITY] == pytest.approx(
        1 - (1 - unknown) ** 2  # post-2 and post-4 both name them
    )
    evidence = evidence_report(graph)
    assert evidence.tiers == {"unknown": 8}
    assert evidence.unknown_tier == 8
    assert "unknown=8" in graph.graph["tiers"]


def test_the_networks_without_matched_evidence_are_certain_and_say_so(
    ingested: IngestReport, memory_store: InMemoryGraphStore, layered: InMemoryGraphStore
) -> None:
    """A speaker is on a document because a record says so, not because a string matched.

    The speaker network comes from the ingested transcripts, where the host shares a document
    with every guest; the topic network is asked for one source, which is the path that
    recomputes co-occurrence over the documents rather than reading the stored aggregate.
    """
    speakers = speaker_co_participation(memory_store, "test-pm")
    topics = build_network(layered, "topics", "test-layers", min_weight=1, source_id="posts")
    for graph in (speakers, topics):
        assert graph.number_of_edges() > 0
        assert all(value == 1.0 for value in edge_probabilities(graph).values())
        assert "p = 1.0 on every edge" in graph.graph["p_rule"]
        assert graph.graph["tiers"] == ""


def test_a_relation_is_priced_by_how_many_passages_state_it(related: InMemoryGraphStore) -> None:
    """Alpha->Beta is stated twice, Beta->Alpha once (§28.1: more evidence, surer edge)."""
    graph = build_network(related, "relations", "test-layers", min_weight=1)
    assert graph["product:alpha"]["product:beta"][EDGE_PROBABILITY] == pytest.approx(
        1 - (1 - STATED_PROBABILITY) ** 2
    )
    assert graph["product:beta"]["product:alpha"][EDGE_PROBABILITY] == pytest.approx(
        STATED_PROBABILITY
    )


def test_the_two_mode_network_and_its_projections_carry_p(layered: InMemoryGraphStore) -> None:
    """The speaker side is a record and the entity side a match, so the match decides."""
    unknown = TIER_PROBABILITY["unknown"]
    two_mode = build_network(layered, "speakers-entities", "test-layers", min_weight=1)
    # ana names Beta in post-1 and post-3, so two passages support the edge
    assert two_mode["ana"]["product:beta"][EDGE_PROBABILITY] == pytest.approx(
        1 - (1 - unknown) ** 2
    )
    assert two_mode["bo"]["product:alpha"][EDGE_PROBABILITY] == pytest.approx(unknown)
    projected = build_network(
        layered, "speakers-entities", "test-layers", min_weight=1, project="entities"
    )
    # Alpha and Gamma share three speakers -- bo (post-2), cy (post-4) and ana, who named Alpha
    # in post-1 and Gamma in post-3 -- and each is worth the weaker of the two memberships
    # through it, which is `unknown` on both sides.
    assert projected["product:alpha"]["product:gamma"]["weight"] == 3
    assert projected["product:alpha"]["product:gamma"][EDGE_PROBABILITY] == pytest.approx(
        combine([unknown] * 3)
    )


# --------------------------------------------------------------------------- §28.3 closed forms


def test_expected_degree_is_the_sum_of_the_probabilities(layered: InMemoryGraphStore) -> None:
    """§28.3, p. 403: E[k_u] = sum_v p_uv, and here that sum is taken by hand."""
    graph = entity_co_mention(layered, "test-layers", min_weight=1)
    expected = expected_degree(graph)
    for node in graph.nodes:
        assert expected[node] == pytest.approx(
            sum(float(graph[node][other][EDGE_PROBABILITY]) for other in graph[node])
        )
        assert expected[node] <= graph.degree(node)


def test_the_book_s_own_figure_28_4_comes_back_off_the_page() -> None:
    """The four-node network of Figure 28.4 (p. 401), whose answers are drawn in Figure 28.7.

    Edges as the figure labels them: 1-2 at 0.7, 1-3 at 0.9, 2-3 at 0.5, 3-4 at 0.3. Node 3
    touches three edges, so its degree distribution is the convolution of those three Bernoullis
    -- the book's Figure 28.7(a) -- and its expected degree is their sum, 1.7.
    """
    graph = nx.Graph()
    graph.add_edge("1", "2", weight=1, p=0.7)
    graph.add_edge("1", "3", weight=1, p=0.9)
    graph.add_edge("2", "3", weight=1, p=0.5)
    graph.add_edge("3", "4", weight=1, p=0.3)

    assert expected_degree(graph)["3"] == pytest.approx(0.9 + 0.5 + 0.3)
    distribution = degree_distribution(graph, "3")
    assert distribution[0] == pytest.approx(0.1 * 0.5 * 0.7)
    assert distribution[3] == pytest.approx(0.9 * 0.5 * 0.3)
    assert sum(distribution.values()) == pytest.approx(1.0)
    # 0.9*0.5*0.7 + 0.9*0.5*0.3 + 0.1*0.5*0.3: two of the three edges is the likeliest case
    assert distribution[2] == pytest.approx(0.465)
    assert max(distribution, key=lambda k: distribution[k]) == 2
    assert expected_edges(graph) == pytest.approx(0.7 + 0.9 + 0.5 + 0.3)
    assert expected_density(graph) == pytest.approx(2.4 / 6)
    assert degree_distribution(graph, "4") == {0: pytest.approx(0.7), 1: pytest.approx(0.3)}
    with pytest.raises(KeyError):
        degree_distribution(graph, "nobody")


def test_the_closed_forms_count_both_directions_of_a_reciprocated_pair() -> None:
    """On a digraph the total degree is in + out (§6.2), so a reciprocated pair counts twice.

    a->b at 0.95, b->a at 0.95, b->c at 0.6. Node a touches two directed edges, not one: its
    classical total degree is 2 and its expectation is 1.90, with 2 the likeliest value at
    0.9025. Undirecting first would merge the two arcs and print 0.95 beside an observed 2,
    which is the report contradicting itself.
    """
    graph = nx.DiGraph()
    graph.add_edge("a", "b", weight=1, p=0.95)
    graph.add_edge("b", "a", weight=1, p=0.95)
    graph.add_edge("b", "c", weight=1, p=0.6)

    expected = expected_degree(graph)
    assert graph.degree("a") == 2
    assert expected["a"] == pytest.approx(1.90)
    assert expected["b"] == pytest.approx(0.95 + 0.95 + 0.6)
    assert expected["c"] == pytest.approx(0.6)

    distribution = degree_distribution(graph, "a")
    assert distribution == {
        0: pytest.approx(0.05**2),
        1: pytest.approx(2 * 0.95 * 0.05),
        2: pytest.approx(0.95**2),
    }
    assert max(distribution, key=lambda k: distribution[k]) == 2
    # the distribution's mean is the expectation, on any network: that is what puts the two in
    # one table
    for node in graph:
        assert sum(
            degree * value for degree, value in degree_distribution(graph, node).items()
        ) == pytest.approx(expected[node])
    # and the weighted form sums the same terms
    assert expected_degree(graph, weight="weight")["a"] == pytest.approx(1.90)


def test_the_karate_club_s_disputed_edge_is_half_an_edge() -> None:
    """The chapter's opening complaint as a number (p. 396): 77 edges or 78, so 77.5.

    *"The fun thing about this graph is that we actually don't know whether it has 77 or 78
    edges."* Model that as one edge at p = 0.5 and everything else certain: the expectation is
    exactly half way between the two copies of the graph, and each endpoint of the disputed edge
    has a degree that is one of two values with equal probability -- which is the chapter's point
    that a point estimate hides.
    """
    graph, known = karate_club()
    assert known.edges == 78
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 1.0
    disputed = (0, 31)  # any one edge; the literature does not say which entry is asymmetric
    graph[disputed[0]][disputed[1]][EDGE_PROBABILITY] = 0.5

    assert expected_edges(graph) == pytest.approx(77.5)
    for node in disputed:
        classical = graph.degree(node)
        assert degree_distribution(graph, node) == {
            classical - 1: pytest.approx(0.5),
            classical: pytest.approx(0.5),
        }
        assert expected_degree(graph)[node] == pytest.approx(classical - 0.5)


# --------------------------------------------------------------------------- §28.2 sampling


def test_realisations_keep_every_node_and_are_seeded() -> None:
    """A world in which an edge failed has an isolated node, not a missing one."""
    graph = nx.path_graph(6)
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 0.5
    worlds = list(realisations(graph, 25, seed=3))
    assert len(worlds) == 25
    assert all(set(w.nodes) == set(graph.nodes) for w in worlds)
    assert {w.number_of_edges() for w in worlds} != {graph.number_of_edges()}
    again = [w.number_of_edges() for w in realisations(graph, 25, seed=3)]
    assert again == [w.number_of_edges() for w in worlds]
    assert again != [w.number_of_edges() for w in realisations(graph, 25, seed=4)]
    with pytest.raises(ValueError, match="samples must be at least 1"):
        list(realisations(graph, 0))


def test_monte_carlo_degree_converges_on_the_closed_form() -> None:
    """The sampled mean has to agree with `sum p`, or one of the two is wrong (§28.2 vs §28.3)."""
    graph = nx.karate_club_graph()
    for index, (_u, _v, data) in enumerate(graph.edges(data=True)):
        data[EDGE_PROBABILITY] = 0.4 + 0.1 * (index % 6)
    exact = expected_degree(graph)
    sampled = node_expectation(
        graph, lambda world: dict(world.degree()), samples=500, seed=11, level=0.95
    )
    for node, estimate in sampled.items():
        assert estimate.mean == pytest.approx(exact[node], abs=0.35)
        assert estimate.low <= estimate.observed
        assert estimate.samples == 500
        assert not estimate.certain


def test_a_certain_network_collapses_every_interval_onto_the_observed_value() -> None:
    """With every p = 1 there is one possible world, and it is the network (§28.2)."""
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 1.0
    clustering = expectation(graph, nx.average_clustering, samples=25, seed=5)
    assert clustering.certain
    assert clustering.low == clustering.high == pytest.approx(clustering.observed)
    assert clustering.sd == pytest.approx(0.0)
    assert clustering.width == pytest.approx(0.0)
    assert "no uncertainty" in clustering.caveat or "p = 1" in clustering.caveat
    assert "(certain)" in clustering.text

    degrees = node_expectation(graph, lambda world: dict(world.degree()), samples=25, seed=5)
    assert all(e.low == e.high == e.observed for e in degrees.values())


def test_overlapping_intervals_are_a_tie_and_separate_ones_are_not() -> None:
    """The rule the report prints: a rank whose interval overlaps its neighbour's is not one."""
    graph = nx.Graph()
    # A star plus a triangle: the hub's degree dwarfs everything, and two triangle nodes are
    # indistinguishable by construction, so one pair must separate and the other must not.
    for spoke in range(1, 9):
        graph.add_edge("hub", f"s{spoke}", weight=1, p=0.9)
    graph.add_edge("t1", "t2", weight=1, p=0.9)
    graph.add_edge("t2", "t3", weight=1, p=0.9)
    graph.add_edge("t1", "t3", weight=1, p=0.9)
    estimates = node_expectation(
        graph, lambda world: dict(world.degree()), samples=300, seed=2, level=0.95
    )
    assert not estimates["hub"].overlaps(estimates["t1"])
    assert estimates["t1"].overlaps(estimates["t2"])
    assert estimates["s1"].overlaps(estimates["s2"])


# --------------------------------------------------------------------- ATL-F2: --max-seconds


def test_a_zero_second_budget_runs_no_realisations_and_says_so() -> None:
    """§28.2's Monte Carlo, budgeted (ATL-F2): a budget that is already spent before the first
    realisation collapses the interval onto the observed value, the same shape a certain network
    gets, but for a different, stated reason."""
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 0.5
    result = expectation(graph, nx.average_clustering, samples=200, seed=1, max_seconds=0.0)
    assert result.samples == 0
    assert result.low == result.high == pytest.approx(result.observed)
    assert not result.certain
    assert "--max-seconds" in result.caveat
    assert "0 of the 200" in result.caveat

    degrees = node_expectation(
        graph, lambda world: dict(world.degree()), samples=200, seed=1, max_seconds=0.0
    )
    for estimate in degrees.values():
        assert estimate.samples == 0
        assert estimate.low == estimate.high == pytest.approx(estimate.observed)
        assert "--max-seconds" in estimate.caveat


def test_a_generous_budget_changes_nothing() -> None:
    """Rule: identical numbers where nothing was sampled (or, here, budgeted away). A budget with
    ample room finishes every requested realisation, same as no budget at all."""
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 0.5
    unbudgeted = expectation(graph, nx.average_clustering, samples=40, seed=7)
    budgeted = expectation(graph, nx.average_clustering, samples=40, seed=7, max_seconds=60.0)
    assert budgeted.samples == unbudgeted.samples == 40
    assert budgeted.mean == pytest.approx(unbudgeted.mean)
    assert budgeted.caveat == unbudgeted.caveat == ""


def test_max_seconds_stops_mid_run_and_reports_what_finished() -> None:
    """The realisation that crosses the budget still finishes and counts; nothing after it
    starts. Timed against a deliberately slow measure so the cutoff falls predictably between
    the first and the last of a handful of realisations, without pinning an exact count."""
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 0.9

    def slow_clustering(world: nx.Graph) -> float:
        time.sleep(0.05)
        return float(nx.average_clustering(world))

    result = expectation(graph, slow_clustering, samples=20, seed=1, max_seconds=0.12)
    assert 0 < result.samples < 20
    assert "--max-seconds" in result.caveat
    assert f"{result.samples} of the 20" in result.caveat


def test_uncertainty_report_shares_one_budget_across_every_measure() -> None:
    """One wall-clock allowance for the whole ``--uncertain`` section (ATL-F2), not a fresh one
    per centrality: a slow first measure that spends the whole budget leaves nothing for the
    ones after it, and each says so on its own numbers."""
    # Relabelled to str node ids: this package's own networks always carry them (the karate
    # club's int ids are a legendary-graph-only convention -- see backbone.Node), and
    # render_uncertain's degree table assumes a node id joins straight into a markdown row.
    graph = nx.relabel_nodes(nx.karate_club_graph(), str)
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 0.9
    scores = {node: float(degree) for node, degree in graph.degree()}
    ranking = sorted(scores.items(), key=lambda item: -item[1])

    def slow(_kind: str) -> object:
        def measure(world: nx.Graph) -> dict[str, float]:
            time.sleep(0.05)
            return dict(world.degree())

        return measure

    report = uncertainty_report(
        graph,
        {"first": ranking, "second": ranking},
        measure_for=slow,
        samples=20,
        seed=1,
        max_seconds=0.08,
    )
    first_ran = report.centralities["first"][0][1].samples
    second_ran = report.centralities["second"][0][1].samples
    assert 0 < first_ran < 20
    assert second_ran == 0  # the budget the first measure spent is not handed back
    assert "--max-seconds" in report.centralities["second"][0][1].caveat
    rendered = "\n".join(render_uncertain(report))
    assert "--max-seconds" in rendered
    payload = uncertain_payload(report)
    assert payload["max_seconds"] == 0.08
    assert payload["centrality_samples"]["second"] == 0


def test_reliability_is_a_probability_of_reaching_not_of_being_joined() -> None:
    """§28.3, p. 406: two nodes can be adjacent and still unreachable in most worlds."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1, p=0.25)
    graph.add_edge("b", "c", weight=1, p=1.0)
    graph.add_edge("c", "d", weight=1, p=1.0)
    assert reliability(graph, "b", "d", samples=200, seed=1) == 1.0
    # a hangs off the network by one edge at p=0.25, so it reaches anything a quarter of the time
    assert reliability(graph, "a", "d", samples=400, seed=1) == pytest.approx(0.25, abs=0.06)
    assert reliability(graph, "a", "a", samples=10, seed=1) == 1.0
    with pytest.raises(KeyError):
        reliability(graph, "a", "zzz", samples=5, seed=1)


# --------------------------------------------------------------------------- §28.1 the report


def test_the_evidence_section_counts_the_thin_edges(memory_store: InMemoryGraphStore) -> None:
    """§28.1 is about spurious edges, and a single loose match is the first candidate."""
    _plant(
        memory_store,
        {
            "post-1": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-2": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-3": [("p:alpha", "loose"), ("p:gamma", "loose")],
        },
    )
    graph = entity_co_mention(memory_store, PERSONA, min_weight=1)
    evidence = evidence_report(graph)
    assert evidence.edges == 2
    assert evidence.certain == 0
    assert evidence.single == 1  # only Alpha-Gamma rests on one passage
    assert evidence.single_weak == 1
    assert evidence.single_weight_share == pytest.approx(1 / 3)
    assert evidence.tiers == {"exact": 4, "loose": 2}
    assert evidence.unit == "passages shared"
    assert "§28.1" in evidence.rule


def test_both_frames_name_n_and_the_sampling_frame(layered: InMemoryGraphStore) -> None:
    """Both chapter-28 sections obey "report n", and say what a node is while they do it."""
    graph = entity_co_mention(layered, "test-layers", min_weight=1)
    evidence = evidence_report(graph)
    assert evidence.nodes == graph.number_of_nodes()
    assert evidence.frame == graph.graph["frame"]
    assert "Entities mentioned in the same passage" in evidence.frame

    first = "\n".join(render_evidence(evidence))
    assert f"n = {evidence.nodes:,} nodes" in first
    assert evidence.frame in first

    report = uncertainty_report(
        graph,
        {"degree": [(n, 1.0) for n in sorted(graph.nodes)]},
        measure_for=lambda kind: lambda world: centrality(world, kind),
        samples=10,
        seed=1,
    )
    second = "\n".join(render_uncertain(report)).split("## Under uncertainty")[1]
    assert f"n = {evidence.nodes:,} nodes" in second
    assert evidence.frame in second


def test_a_graph_that_lost_its_probabilities_reads_as_certain_rather_than_as_measured() -> None:
    """A network from a file, or a flattening, has no p: inventing one would be worse."""
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=3)
    graph.add_edge("b", "c", weight=1)
    evidence = evidence_report(graph)
    assert evidence.unpriced
    assert evidence.certain == 2
    assert evidence.single == 1
    assert evidence.tiers == {}
    assert edge_probabilities(graph) == {("a", "b"): 1.0, ("b", "c"): 1.0}


def test_a_flattened_multilayer_network_says_its_probabilities_did_not_survive(
    layered: InMemoryGraphStore,
) -> None:
    """`flatten` builds new edges, so the evidence section has to say it is not measuring any."""
    from graphrag.sna.layers import flatten, multilayer

    graph = flatten(multilayer(layered, "test-layers", "entities", "facet", min_weight=1))
    evidence = evidence_report(graph)
    assert evidence.unpriced
    assert "\n".join(render_evidence(evidence)).count("No edge of this network carries") == 1


def test_a_probability_and_a_backbone_s_p_value_are_two_different_numbers() -> None:
    """Chapter 27 and chapter 28 both want to write ``p``; only one of them may.

    A statistical backbone leaves the significance it survived on under ``p_value``, and the
    probability of existing stays under ``p``. They point opposite ways -- a small ``p_value``
    says the edge is probably real, a small ``p`` that it probably is not -- so an edge carrying
    both must keep them apart, or ``--backbone noise-corrected --uncertain`` would sample
    Bernoulli draws from a column of p-values.
    """
    from graphrag.sna.backbone import backbone

    graph = nx.karate_club_graph()
    for index, (_u, _v, data) in enumerate(graph.edges(data=True)):
        data["weight"] = 1 + index % 4
        data[EDGE_PROBABILITY] = 0.5 + 0.05 * (index % 10)
    filtered = backbone(graph, "noise-corrected", alpha=0.5)

    assert filtered.number_of_edges() > 0
    for u, v, data in filtered.edges(data=True):
        assert data[EDGE_PROBABILITY] == pytest.approx(graph[u][v][EDGE_PROBABILITY])
        assert "p_value" in data
    # and the sampler carries the probabilities into the sample it takes
    from graphrag.sna.sampling import sample

    taken = sample(graph, "bfs", 12, seed=1)
    assert taken.number_of_edges() > 0
    assert all(EDGE_PROBABILITY in data for _, _, data in taken.edges(data=True))


def test_the_analysis_always_reports_the_evidence_and_only_samples_when_asked(
    memory_store: InMemoryGraphStore,
) -> None:
    _plant(
        memory_store,
        {
            "post-1": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-2": [("p:alpha", "exact"), ("p:gamma", "loose")],
            "post-3": [("p:beta", "exact"), ("p:gamma", "exact")],
        },
    )
    graph = entity_co_mention(memory_store, PERSONA, min_weight=1)
    plain = run_analysis(
        memory_store, graph, persona_id=PERSONA, network="entities", method="louvain", seed=1
    )
    assert plain.evidence is not None
    assert plain.uncertainty is None

    priced = run_analysis(
        memory_store,
        graph,
        persona_id=PERSONA,
        network="entities",
        method="louvain",
        seed=1,
        samples=5,
        uncertain=True,
        uncertain_samples=40,
    )
    assert priced.uncertainty is not None
    assert priced.uncertainty.samples == 40
    assert priced.uncertainty.expected_edges < graph.number_of_edges()
    assert priced.uncertainty.modularity is not None
    # ATL-F2: only the cheap centralities are sampled by default -- a strict subset of what the
    # plain report ranks, not the same set -- and the report says which were left out.
    assert set(priced.uncertainty.centralities) < set(plain.centralities)
    assert set(priced.uncertainty.centralities) == set(CHEAP_CENTRALITIES) & set(plain.centralities)
    assert any("uncertain-expensive" in note for note in priced.uncertainty.notes)

    expensive = run_analysis(
        memory_store,
        graph,
        persona_id=PERSONA,
        network="entities",
        method="louvain",
        seed=1,
        samples=5,
        uncertain=True,
        uncertain_samples=40,
        uncertain_expensive=True,
    )
    assert expensive.uncertainty is not None
    assert set(expensive.uncertainty.centralities) == set(plain.centralities)


def test_the_cli_default_for_uncertain_samples_matches_the_module() -> None:
    """The CLI repeats the default rather than importing it, so it must be held to the number."""
    assert UNCERTAIN_SAMPLES == SAMPLES


# --------------------------------------------------------------------------- the store


def test_the_tier_is_stored_read_back_and_never_erased_by_a_re_import(
    memory_store: InMemoryGraphStore,
) -> None:
    """An old file carries no tier; re-importing it must not cost the evidence a later pass got."""
    _plant(memory_store, {"post-1": [("p:alpha", "loose"), ("p:beta", "exact")]})
    chunk_id = f"{PERSONA}:{SOURCE}:post-1#0"
    assert {r.entity_id: r.tier for r in memory_store.entity_chunk_pairs(PERSONA)} == {
        "p:alpha": "loose",
        "p:beta": "exact",
    }
    assert {r.entity_id: r.tier for r in memory_store.entity_mention_rows(PERSONA)} == {
        "p:alpha": "loose",
        "p:beta": "exact",
    }

    memory_store.upsert_enrichment(
        Enrichment(mentions=[Mention(chunk_id=chunk_id, entity_id="p:alpha")])
    )
    assert {r.entity_id: r.tier for r in memory_store.entity_chunk_pairs(PERSONA)}[
        "p:alpha"
    ] == "loose"

    memory_store.upsert_enrichment(
        Enrichment(mentions=[Mention(chunk_id=chunk_id, entity_id="p:alpha", tier="exact")])
    )
    assert {r.entity_id: r.tier for r in memory_store.entity_chunk_pairs(PERSONA)}[
        "p:alpha"
    ] == "exact"
    # the snapshot's own object carries it, which is how it survives an export/load round trip
    assert {m.entity_id: m.tier for m in memory_store.enrichment_for_persona(PERSONA).mentions} == {
        "p:alpha": "exact",
        "p:beta": "exact",
    }


def test_a_snapshot_written_before_tiers_existed_still_loads(tmp_path: Path) -> None:
    """The committed snapshots have no tier on any mention, and must go on loading."""
    payload = {
        "entities": [{"id": "p:alpha", "name": "Alpha", "type": "product"}],
        "mentions": [{"chunk_id": "c1", "entity_id": "p:alpha"}],
        "relations": [],
    }
    path = tmp_path / "enrichment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    enrichment = Enrichment.model_validate_json(path.read_text(encoding="utf-8"))
    assert enrichment.mentions[0].tier == "unknown"


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def priced(cli_context: AppContext) -> AppContext:
    """The planted corpus of this module, in the CLI's own context and registry."""
    from graphrag.models import PersonaSpec, SourceSpec

    cli_context.registry.save(
        PersonaSpec(
            id=PERSONA,
            name="Uncertain",
            role_prompt="You answer from planted posts.",
            sources=[SourceSpec(id=SOURCE, kind="local", path="posts", loader="documents")],
        )
    )
    store = cli_context.store
    assert isinstance(store, InMemoryGraphStore)
    _plant(
        store,
        {
            "post-1": [("p:alpha", "exact"), ("p:beta", "exact")],
            "post-2": [("p:alpha", "exact"), ("p:beta", "exact"), ("p:gamma", "loose")],
            "post-3": [("p:beta", "exact"), ("p:gamma", "exact"), ("p:delta", "none")],
            "post-4": [("p:alpha", "loose"), ("p:delta", "loose")],
        },
    )
    return cli_context


@pytest.mark.parametrize(
    ("flag", "budget"),
    [([], 300.0), (["--max-seconds", "0"], None), (["--max-seconds", "45"], 45.0)],
    ids=["default", "zero-is-none", "named"],
)
def test_sna_analyze_uncertain_has_a_default_budget_that_zero_turns_off(
    priced: AppContext, tmp_path: Path, flag: list[str], budget: float | None
) -> None:
    """Without a budget, fifty realisations on a 1,300-node entity network ran past fifteen
    minutes, so the command a person types now bounds itself by default; `0` is the way out."""
    payload = tmp_path / "note.json"
    report = tmp_path / "note.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--uncertain",
            "--uncertain-samples",
            "5",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "3",
            "--out",
            str(report),
            "--json",
            str(payload),
            *flag,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(payload.read_text())["uncertainty"]["max_seconds"] == budget
    budget_line = "**Budget.** --max-seconds"
    assert (budget_line in report.read_text()) == (budget is not None)


def test_sna_analyze_uncertain_prints_an_interval_on_every_centrality_row(
    priced: AppContext, tmp_path: Path
) -> None:
    report = tmp_path / "note.md"
    payload = tmp_path / "note.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--uncertain",
            "--uncertain-samples",
            "60",
            # ATL-F2: betweenness and closeness are the expensive, shortest-path-search
            # centralities, off by default; this is ATL-28's own acceptance test -- every
            # centrality row carries an interval -- so it opts them back in.
            "--uncertain-expensive",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "3",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "## Evidence behind the edges (Atlas §28.1)" in text
    assert "## Under uncertainty (Atlas §28.2-28.3)" in text
    assert "60 sampled possible worlds, seed 3" in text
    assert "### Size, in closed form (§28.3)" in text
    assert "### Degree (§28.3, p. 403)" in text
    # every centrality row carries an interval, which is the ticket's acceptance test
    for kind in ("degree", "betweenness", "closeness", "eigenvector", "pagerank"):
        assert f"### {kind} under uncertainty" in text
    # every row of every centrality table under the heading carries an interval, which is the
    # ticket's acceptance test; the ranked rows are the ones ending in the separated column
    section = text.split("## Under uncertainty")[1].split("\n## ")[0]
    ranked = [
        line for line in section.splitlines() if RANKED_ROW.match(line) and "observed" not in line
    ]
    assert len(ranked) >= 5
    assert all("±" in row for row in ranked)

    data = json.loads(payload.read_text())
    assert data["uncertainty"]["samples"] == 60
    assert data["uncertainty"]["evidence"]["tiers"]["loose"] == 3
    assert data["uncertainty"]["evidence"]["single"] >= 1
    assert data["uncertainty"]["expected_edges"] < data["uncertainty"]["observed_edges"]
    assert set(data["uncertainty"]["centrality"]) == set(data["centrality"])
    assert data["uncertainty"]["modularity"]["low"] <= data["uncertainty"]["modularity"]["high"]
    # clustering is sampled, not closed form, and the section says which and why (p. 398)
    assert "### Clustering under uncertainty" in text
    assert "no closed form for the clustering coefficient" in text
    assert data["uncertainty"]["clustering"]["low"] <= data["uncertainty"]["clustering"]["high"]


def test_sna_analyze_uncertain_skips_the_expensive_centralities_by_default(
    priced: AppContext, tmp_path: Path
) -> None:
    """ATL-F2: without ``--uncertain-expensive`` the shortest-path centralities are left out of
    the sampling, and the report names them rather than silently shrinking the table."""
    report = tmp_path / "note.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--uncertain",
            "--uncertain-samples",
            "20",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "3",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "### degree under uncertainty" in text
    assert "### pagerank under uncertainty" in text
    for kind in ("betweenness", "closeness", "harmonic", "reach"):
        assert f"### {kind} under uncertainty" not in text
    assert "were not sampled under uncertainty" in text
    assert "--uncertain-expensive" in text


def test_sna_analyze_max_seconds_reaches_the_cli(priced: AppContext, tmp_path: Path) -> None:
    """``--max-seconds`` is wired end to end; a generous budget on a tiny network changes
    nothing about the numbers, only that the frame says a budget was given."""
    report = tmp_path / "note.md"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--uncertain",
            "--uncertain-samples",
            "20",
            "--max-seconds",
            "30",
            "--samples",
            "5",
            "--seed",
            "3",
            "--out",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "**Budget.** --max-seconds 30" in text
    assert "### degree under uncertainty" in text
    assert "the interval is over 20 realisation(s)" in text  # the full count, budget not spent


def test_sna_analyze_reports_the_evidence_even_without_the_flag(
    priced: AppContext, tmp_path: Path
) -> None:
    report = tmp_path / "plain.md"
    payload = tmp_path / "plain.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--samples",
            "5",
            "--runs",
            "2",
            "--seed",
            "3",
            "--out",
            str(report),
            "--json",
            str(payload),
        ],
    )
    assert result.exit_code == 0, result.output
    text = report.read_text()
    assert "## Evidence behind the edges (Atlas §28.1)" in text
    assert "## Under uncertainty" not in text
    assert "edges rest on one piece of evidence" in result.output
    data = json.loads(payload.read_text())
    assert "uncertainty" not in data
    assert data["evidence"]["edges"] > 0
    assert data["evidence"]["caveats"]


def test_sna_export_writes_p_into_the_formats_that_carry_attributes(
    priced: AppContext, tmp_path: Path
) -> None:
    target = tmp_path / "entities.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "export",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--out",
            str(target),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(target.read_text())
    assert payload["links"]
    assert all(0.0 < link["p"] <= 1.0 for link in payload["links"])
    assert any(link["p"] < 1.0 for link in payload["links"])
    assert "§28.1" in payload["graph"]["p_rule"]


def test_centrality_under_uncertainty_is_the_same_measure_on_a_sampled_world() -> None:
    """The contract `analyze --uncertain` relies on: the measure is not re-implemented here."""
    graph = nx.karate_club_graph()
    for _, _, data in graph.edges(data=True):
        data[EDGE_PROBABILITY] = 1.0
    estimates = node_expectation(
        graph, lambda world: centrality(world, "betweenness"), samples=3, seed=1
    )
    observed = centrality(graph, "betweenness")
    for node, estimate in estimates.items():
        assert estimate.observed == pytest.approx(observed[node])
        assert estimate.mean == pytest.approx(observed[node])
