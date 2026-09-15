"""The stance report and the two-window comparison, over the planted corpus in ``conftest``."""

from __future__ import annotations

import pytest

from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.sna.compare import compare_windows, render_comparison
from graphrag.sna.stances import build_stance_report, render_stances


def test_the_stance_report_counts_annotations_per_entity_and_names_who_wrote_them(
    layered: InMemoryGraphStore,
) -> None:
    report = build_stance_report(layered, "test-layers")
    assert report.annotations == 8
    by_name = {entity.name: entity for entity in report.entities}
    assert set(by_name) == {"Alpha", "Beta", "Gamma"}

    alpha = by_name["Alpha"]
    assert alpha.counts == {"praise": 1, "complaint": 2}
    assert alpha.documents == 3
    assert alpha.speakers["complaint"] == [("bo", 1), ("cy", 1)]
    assert alpha.speakers["praise"] == [("ana", 1)]
    assert alpha.facets["complaint"] == [("outage", 2)]
    assert alpha.facets["praise"] == [("quoting", 1)]

    # The quotes are the passages themselves, so a count can be checked against text.
    assert len(alpha.quotes["complaint"]) == 2
    assert "busiest morning" in alpha.quotes["complaint"][0]
    assert report.stances_present == ("praise", "complaint", "neutral")


def test_the_report_quotes_at_most_the_number_of_passages_asked_for(
    layered: InMemoryGraphStore,
) -> None:
    report = build_stance_report(layered, "test-layers", quotes_per_stance=1)
    alpha = next(e for e in report.entities if e.name == "Alpha")
    assert alpha.counts["complaint"] == 2  # still counted
    assert len(alpha.quotes["complaint"]) == 1  # but quoted once
    assert build_stance_report(layered, "test-layers", quotes_per_stance=0).entities[0].quotes == {}


def test_the_signed_co_mention_table_separates_praise_from_complaint(
    layered: InMemoryGraphStore,
) -> None:
    """The point of the table: a pair heavy in one column and empty in the other."""
    report = build_stance_report(layered, "test-layers")
    pairs = {(p.left, p.right): p for p in report.pairs}
    assert pairs[("Alpha", "Beta")].shared == {"praise": 1}
    assert pairs[("Alpha", "Gamma")].shared == {"complaint": 2}
    assert ("Beta", "Gamma") not in pairs  # post 3 marks them with different stances
    assert pairs[("Alpha", "Gamma")].count("praise") == 0


def test_the_report_can_be_narrowed_to_named_entities_and_to_a_facet(
    layered: InMemoryGraphStore,
) -> None:
    named = build_stance_report(layered, "test-layers", entities=["alpha"])
    assert [e.name for e in named.entities] == ["Alpha"]
    assert named.annotations == 3
    assert named.missing == ()

    outage = build_stance_report(layered, "test-layers", facets=["outage"])
    assert {e.name for e in outage.entities} == {"Alpha", "Gamma"}
    assert all("praise" not in e.counts for e in outage.entities)

    typo = build_stance_report(layered, "test-layers", entities=["Alpah"])
    assert typo.missing == ("Alpah",)
    assert typo.entities == ()


def test_the_rendered_report_carries_the_frame_the_quotes_and_the_caveats(
    layered: InMemoryGraphStore,
) -> None:
    text = render_stances(build_stance_report(layered, "test-layers"))
    assert "# Stances: test-layers" in text
    assert "a count of annotations, not of mentions" in text
    assert "## Signed co-mention" in text
    assert "> Alpha saved me an afternoon" in text
    assert "Absence of a stance is absence of an annotation" in text


def test_an_empty_report_still_renders_and_says_so(layered: InMemoryGraphStore) -> None:
    text = render_stances(build_stance_report(layered, "test-layers", facets=["nothing"]))
    assert "No annotated mentions matched these filters." in text


# ----------------------------------------------------------------------------- compare


def test_comparing_two_windows_reports_n_the_roster_and_the_partition_agreement(
    layered: InMemoryGraphStore,
) -> None:
    comparison = compare_windows(
        layered,
        "test-layers",
        "speakers",
        until="2025-03-01",
        since2="2025-06-01",
        seed=1,
        runs=2,
    )
    assert list(comparison.a.graph.nodes) == ["ana", "bo"]
    assert list(comparison.b.graph.nodes) == ["ana", "cy"]
    assert comparison.shared == ("ana",)
    assert comparison.entered == ("cy",)
    assert comparison.left == ("bo",)
    assert comparison.a.label == "up to 2025-03-01"
    assert comparison.b.label == "2025-06-01 onward"
    # One shared node is far too few for a similarity score, and the report has to say so.
    assert any("appear in both windows" in note for note in comparison.notes)


def test_comparing_two_windows_ranks_the_movers_among_the_shared_nodes(
    layered: InMemoryGraphStore,
) -> None:
    comparison = compare_windows(
        layered,
        "test-layers",
        "entities",
        until="2025-03-01",
        since2="2025-06-01",
        min_weight=1,
        seed=1,
        runs=2,
    )
    # Both halves of the year name all three, so nothing enters and nothing leaves; what
    # changes is which of them is the hub. Alpha holds two edges before March, Gamma two after.
    every = {"product:alpha", "product:beta", "product:gamma"}
    assert set(comparison.a.graph.nodes) == every
    assert set(comparison.b.graph.nodes) == every
    assert comparison.entered == () and comparison.left == ()
    assert set(comparison.shared) == every

    moved = {change.label: change for change in comparison.changes}
    assert (moved["Gamma"].rank_a, moved["Gamma"].rank_b, moved["Gamma"].change) == (3, 1, 2)
    assert moved["Alpha"].change == -1
    assert comparison.changes[0].label == "Gamma"  # the largest mover is reported first
    assert comparison.centrality == "weighted_degree"


def test_a_comparison_rejects_a_centrality_it_does_not_compute(
    layered: InMemoryGraphStore,
) -> None:
    with pytest.raises(ValueError, match="centrality must be one of"):
        compare_windows(layered, "test-layers", "speakers", centrality_kind="charisma")


def test_the_rendered_comparison_explains_what_is_and_is_not_comparable(
    layered: InMemoryGraphStore,
) -> None:
    text = render_comparison(
        compare_windows(
            layered,
            "test-layers",
            "speakers-entities",
            until="2025-03-01",
            since2="2025-06-01",
            project="speakers",
            seed=1,
            runs=2,
        )
    )
    assert "# Network comparison: test-layers / speakers-entities" in text
    assert "Community numbers are not comparable" in text
    assert "## Who entered and who left" in text
    assert "## Partition similarity" in text
    assert "Largest rank changes (weighted_degree)" in text


def test_an_empty_window_is_reported_rather_than_rendered_as_a_change(
    layered: InMemoryGraphStore,
) -> None:
    comparison = compare_windows(
        layered, "test-layers", "speakers", until="2020-01-01", since2="2025-01-01", seed=1, runs=2
    )
    assert comparison.a.summary["nodes"] == 0
    assert any("is empty" in note for note in comparison.notes)
    assert any("share no nodes at all" in note for note in comparison.notes)
    assert "Not computed" in render_comparison(comparison)
