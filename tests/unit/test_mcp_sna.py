"""`graphrag sna` exposed as MCP tools (ATL-ENT-1).

Every ``sna_*`` tool is checked against the exact command it wraps: the CLI is driven with
``typer.testing.CliRunner`` against the same in-memory store (``cli_context``), and the tool's
``markdown``/``payload`` are asserted equal to what the CLI wrote to ``--out``/``--json`` for the
same arguments. A handful of reports carry a ``generated_at`` timestamp with second precision;
``_mask`` blanks it out so two calls a moment apart never flake.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import typer
from fastmcp import Client
from typer.testing import CliRunner

from graphrag.app import AppContext
from graphrag.cli import app, sna_app
from graphrag.graph.memory_store import InMemoryGraphStore
from graphrag.mcp_server import ServerState, create_server
from graphrag.models import Enrichment, Entity, Mention
from graphrag.pipeline import IngestReport

runner = CliRunner()

PERSONA = "test-layers"


@pytest.fixture
def enriched(cli_context: AppContext, ingested: IngestReport) -> AppContext:
    """``test-pm``'s three episodes, each carrying a different three of four entities (mirrors
    ``tests/unit/test_sna_cli.py``'s own fixture): a non-clique entity network with an actual
    non-edge, which the planted ``test-layers`` corpus (a full triangle either way) does not
    have -- needed for `sna_predict`/`sna_predict_eval`, which sample a *negative* pair."""
    store = cli_context.store
    entities = [
        Entity(id="metric:retention", name="Retention", type="metric"),
        Entity(id="concept:onboarding", name="Onboarding", type="concept"),
        Entity(id="concept:pricing", name="Pricing", type="concept"),
        Entity(id="concept:roadmap", name="Roadmap", type="concept"),
    ]
    per_document = [
        ["metric:retention", "concept:onboarding", "concept:pricing"],
        ["metric:retention", "concept:onboarding", "concept:roadmap"],
        ["metric:retention", "concept:pricing", "concept:roadmap"],
    ]
    mentions: list[Mention] = []
    for doc_id, entity_ids in zip(sorted(store.document_ids("test-pm")), per_document, strict=True):
        for chunk in store.document_chunks(doc_id, 0, 100):
            mentions += [Mention(chunk_id=chunk.id, entity_id=e) for e in entity_ids]
    store.upsert_enrichment(Enrichment(entities=entities, mentions=mentions))
    return cli_context


def _call(server: Any, tool: str, **args: Any) -> Any:
    async def run() -> Any:
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            return result.data if result.data is not None else result.content

    return asyncio.run(run())


def _tools(server: Any) -> set[str]:
    async def run() -> set[str]:
        async with Client(server) as client:
            return {t.name for t in await client.list_tools()}

    return asyncio.run(run())


def _mask(value: Any) -> Any:
    """Blank ``generated_at`` (and the markdown line quoting it) so a real timestamp never
    breaks equality between a CLI run and a tool call a moment apart; blank a float ``nan``
    (``nan != nan``, so an untouched one would fail equality against itself) and ``None`` to the
    same sentinel, since a handful of null-model sections on a network this small (attempted
    degree-preserving rewirings within a fixed budget) report either "not testable" (``None``) or
    an empty-sample ``nan`` depending on process-wide random state neither the CLI nor this tool
    controls -- a pre-existing property of `graphrag.sna.analysis`/`coreperiphery`, not something
    either call path gets wrong relative to the other.

    ATL-ENT-4's own `## Provenance` block (every report now carries one) adds a second timestamp
    line, ``- generated: <iso8601>``, blanked the same way; `sna export`'s payload carries a
    third form again, a JSON-encoded ``"generated_at": "<iso8601>"`` string (GraphML/GEXF cannot
    hold a nested attribute, so `graph.graph["provenance"]` is a string there, not a dict this
    function's own dict branch would reach into)."""
    if isinstance(value, dict):
        return {k: ("<ts>" if k == "generated_at" else _mask(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<none-or-nan>"
    if isinstance(value, str):
        masked = re.sub(r"Generated \S+ by graphrag \S+", "Generated <ts> by graphrag <ver>", value)
        masked = re.sub(r"- generated: \S+", "- generated: <ts>", masked)
        return re.sub(r'"generated_at": "[^"]*"', '"generated_at": "<ts>"', masked)
    return value


def test_tool_registry_covers_every_sna_command(cli_context: AppContext) -> None:
    """A future `sna_app` command cannot be forgotten: every CLI command name, hyphens turned to
    underscores and prefixed `sna_`, must be a registered tool.

    Command groups are left out: `sna cache` (ATL-ENT-3) holds maintenance -- `clear` deletes the
    on-disk network cache -- and the server exposes analyses, never a write to the machine.
    """
    server = create_server(ServerState(context=cli_context))
    cli_commands = typer.main.get_command(sna_app).commands  # type: ignore[attr-defined]
    expected = {
        f"sna_{name.replace('-', '_')}"
        for name, command in cli_commands.items()
        if not hasattr(command, "commands")
    }
    assert expected <= _tools(server)


#: Commands that write a primary artifact rather than a report -- a sampled graph, an
#: interchange file, an SVG -- and keep their own `--out` contract (required, one path, one
#: format); `guide` prints static text and takes neither. Every other `sna_app` command is a
#: report and, since ATL-F1, takes both `--out <md>` and `--json <path>`, both optional.
NOT_A_REPORT_COMMAND = {"sample", "export", "draw", "guide"}


def test_every_report_command_has_optional_out_and_json() -> None:
    """ATL-F1: `--out`/`--json` are optional on every report command -- the one contract this
    ticket gives them all, replacing `backbone`/`layers` (no `--json`), `walks`/`distance` (no
    `--out` or `--json`), `stances`/`compare` (no `--json`, `--out` required) and `analyze`/
    `multilayer-communities` (`--out` required).
    """
    cli_commands = typer.main.get_command(sna_app).commands  # type: ignore[attr-defined]
    for name, command in cli_commands.items():
        if hasattr(command, "commands") or name in NOT_A_REPORT_COMMAND:
            continue
        by_opt = {opt: param for param in command.params for opt in param.opts}
        assert "--out" in by_opt, f"sna {name} has no --out"
        assert "--json" in by_opt, f"sna {name} has no --json"
        assert not by_opt["--out"].required, f"sna {name}'s --out is required"
        assert not by_opt["--json"].required, f"sna {name}'s --json is required"


def test_sna_guide_matches_cli(cli_context: AppContext) -> None:
    result = runner.invoke(app, ["sna", "guide"])
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_guide")
    assert out["markdown"].strip() == result.stdout.strip()
    assert out["payload"] == {}


def test_sna_analyze_matches_cli(
    cli_context: AppContext, ingested: IngestReport, tmp_path: Path
) -> None:
    report = tmp_path / "note.md"
    payload_path = tmp_path / "note.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "analyze",
            "test-pm",
            "--network",
            "speakers",
            "--method",
            "louvain",
            "--seed",
            "1",
            "--runs",
            "3",
            "--samples",
            "10",
            "--out",
            str(report),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_analyze",
        persona="test-pm",
        network="speakers",
        method="louvain",
        seed=1,
        runs=3,
        samples=10,
    )
    assert _mask(out["markdown"]) == _mask(report.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


@pytest.mark.parametrize(("given", "budget"), [({}, 300.0), ({"uncertain_max_seconds": 0}, None)])
def test_sna_analyze_uncertain_has_the_cli_default_budget(
    cli_context: AppContext, ingested: IngestReport, given: dict[str, float], budget: float | None
) -> None:
    """The tool bounds `uncertain` by the same five minutes the CLI does, and `0` turns it off."""
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_analyze",
        persona="test-pm",
        network="speakers",
        seed=1,
        runs=2,
        samples=3,
        uncertain=True,
        uncertain_samples=3,
        **given,
    )
    assert out["payload"]["uncertainty"]["max_seconds"] == budget


def test_sna_analyze_unknown_method_is_a_structured_error(
    cli_context: AppContext, ingested: IngestReport
) -> None:
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_analyze", persona="test-pm", method="not-a-method")
    assert "error" in out


def test_sna_export_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    target = tmp_path / "entities.json"
    result = runner.invoke(
        app, ["sna", "export", PERSONA, "--network", "entities", "--out", str(target)]
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_export", persona=PERSONA, network="entities")
    assert _mask(out["payload"]) == _mask(json.loads(target.read_text()))
    assert str(out["markdown"]).endswith(result.stdout.strip().rsplit(": ", 1)[-1])


def test_sna_sample_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    graph_out = tmp_path / "sample.json"
    bias_out = tmp_path / "bias.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "sample",
            PERSONA,
            "--out",
            str(graph_out),
            "--size",
            "2",
            "--network",
            "entities",
            "--method",
            "induced",
            "--seed",
            "3",
            "--json",
            str(bias_out),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_sample",
        persona=PERSONA,
        size=2,
        network="entities",
        method="induced",
        seed=3,
    )
    # The CLI's first line confirms the write ("wrote .../sample.json: ..."), which the tool
    # never does (it does not write the sampled graph); the rest is `render_sample`'s own text,
    # up to the CLI's own final "wrote .../bias.json" confirmation of --json, which the tool
    # also never prints.
    body = result.stdout.split("\n", 1)[1]
    body = body.rsplit("\nwrote ", 1)[0]
    assert _mask(out["markdown"]) == _mask(body)
    assert _mask(out["payload"]) == _mask(json.loads(bias_out.read_text()))


def test_sna_degree_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    section = tmp_path / "degree.md"
    payload_path = tmp_path / "degree.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            PERSONA,
            "--network",
            "entities",
            "--bootstrap",
            "5",
            "--seed",
            "2",
            "--out",
            str(section),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_degree", persona=PERSONA, network="entities", bootstrap=5, seed=2)
    assert _mask(out["markdown"]) == _mask(section.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_filter_arguments_agree_between_cli_and_mcp(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """ATL-F1: `graphrag.sna.arguments.parse_types`/`parse_where`/`parse_sample_params` give the
    CLI's `--types a,b` (one comma string) and `--where k=v` (repeated) the same canonical shape
    (`list[str]`, `dict[str, str]`) a tool caller's `types`/`where` already have, so the same
    query's `provenance.parameters` agrees byte for byte no matter which surface asked -- the
    inconsistency this ticket's `ATLAS_PLAN.md` entry names.
    """
    payload_path = tmp_path / "degree.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "degree",
            PERSONA,
            "--network",
            "entities",
            "--types",
            " product, product ",
            "--where",
            "region=north",
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_degree",
        persona=PERSONA,
        network="entities",
        types=["product", "product", ""],
        where={"region": "north"},
    )
    cli_params = json.loads(payload_path.read_text())["provenance"]["parameters"]
    tool_params = out["payload"]["provenance"]["parameters"]
    assert cli_params["types"] == tool_params["types"] == ["product", "product"]
    assert cli_params["where"] == tool_params["where"] == {"region": "north"}

    # `sna sample`'s `--param name=value` (repeated) against a tool's `params` dict, the other
    # shape the ticket names: a list of `"k=v"` strings versus a mapping.
    bias_path = tmp_path / "bias.json"
    sample_result = runner.invoke(
        app,
        [
            "sna",
            "sample",
            PERSONA,
            "--out",
            str(tmp_path / "sample.json"),
            "--size",
            "2",
            "--network",
            "entities",
            "--method",
            "induced",
            "--param",
            "neighbors=true",
            "--seed",
            "1",
            "--json",
            str(bias_path),
        ],
    )
    assert sample_result.exit_code == 0, sample_result.output
    sample_out = _call(
        server,
        "sna_sample",
        persona=PERSONA,
        size=2,
        network="entities",
        method="induced",
        params={"neighbors": True},
        seed=1,
    )
    cli_sample_params = json.loads(bias_path.read_text())["provenance"]["parameters"]
    tool_sample_params = sample_out["payload"]["provenance"]["parameters"]
    assert cli_sample_params["param"] == tool_sample_params["param"] == {"neighbors": True}


def test_sna_roles_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    section = tmp_path / "roles.md"
    payload_path = tmp_path / "roles.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "roles",
            PERSONA,
            "--network",
            "entities",
            "--method",
            "jaccard",
            "--seed",
            "1",
            "--out",
            str(section),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_roles", persona=PERSONA, network="entities", method="jaccard", seed=1)
    assert _mask(out["markdown"]) == _mask(section.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_ego_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    section = tmp_path / "ego.md"
    payload_path = tmp_path / "ego.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "ego",
            PERSONA,
            "product:alpha",
            "--network",
            "entities",
            "--out",
            str(section),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_ego", persona=PERSONA, node="product:alpha", network="entities")
    assert _mask(out["markdown"]) == _mask(section.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_ego_unknown_node_is_a_structured_error(
    cli_context: AppContext, related: InMemoryGraphStore
) -> None:
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_ego", persona=PERSONA, node="product:not-a-thing", network="entities")
    assert "error" in out


def test_sna_backbone_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "backbone.md"
    payload_path = tmp_path / "backbone.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "backbone",
            PERSONA,
            "--network",
            "entities",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_backbone", persona=PERSONA, network="entities")
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    from graphrag.sna.backbone import BACKBONES

    assert len(out["payload"]["rows"]) == len(BACKBONES)
    # ATL-F1: `--json` (new on this command) writes the same payload the tool returns.
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_layers_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "layers.md"
    payload_path = tmp_path / "layers.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "layers",
            PERSONA,
            "--network",
            "entities",
            "--layers",
            "stance",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_layers", persona=PERSONA, network="entities", layers="stance")
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    payload = out["payload"]
    assert payload["layering"] == "stance"
    assert len(payload["layers"]) == payload["supra_adjacency"]["layers"]
    # ATL-F1: `--json` (new on this command) writes the same payload the tool returns.
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_stances_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "stances.md"
    payload_path = tmp_path / "stances.json"
    result = runner.invoke(
        app,
        ["sna", "stances", PERSONA, "--out", str(out_path), "--json", str(payload_path)],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_stances", persona=PERSONA)
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert out["payload"]["persona_id"] == PERSONA
    assert out["payload"]["annotations"] > 0
    # ATL-F1: `--json` (new on this command) writes the same payload the tool returns.
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_compare_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "compare.md"
    payload_path = tmp_path / "compare.json"
    args = [
        "sna",
        "compare",
        PERSONA,
        "--network",
        "speakers",
        "--since",
        "2025-01-01",
        "--until",
        "2025-03-01",
        "--since2",
        "2025-05-01",
        "--until2",
        "2025-07-01",
        "--out",
        str(out_path),
        "--json",
        str(payload_path),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_compare",
        persona=PERSONA,
        network="speakers",
        since="2025-01-01",
        until="2025-03-01",
        since2="2025-05-01",
        until2="2025-07-01",
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert out["payload"]["a"]["since"] == "2025-01-01"
    assert out["payload"]["b"]["since"] == "2025-05-01"
    # ATL-F1: `--json` (new on this command) writes the same payload the tool returns.
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_multilayer_communities_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "mc.md"
    payload_path = tmp_path / "mc.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "multilayer-communities",
            PERSONA,
            "--network",
            "entities",
            "--layers",
            "stance",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_multilayer_communities",
        persona=PERSONA,
        network="entities",
        layers="stance",
        seed=1,
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_projections_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "proj.md"
    payload_path = tmp_path / "proj.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "projections",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_projections", persona=PERSONA, network="entities", min_weight=1)
    assert _mask(out["markdown"]).rstrip() == _mask(out_path.read_text()).rstrip()
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_walks_matches_cli_console(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "walks.md"
    payload_path = tmp_path / "walks.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "walks",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--from",
            "product:alpha",
            "--to",
            "product:beta",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_walks",
        persona=PERSONA,
        network="entities",
        min_weight=1,
        source_node="product:alpha",
        target_node="product:beta",
    )
    # `sna walks` renders a `rich` table to the console; this tool renders the same numbers as
    # plain lines instead (see the tool's own docstring), so markdown is not asserted byte-equal
    # to `result.stdout` here -- only that the same numbers appear in both.
    payload = out["payload"]
    assert f"{payload['commute_time']:,.2f}" in result.stdout
    assert f"{payload['effective_resistance']:,.4f}" in result.stdout
    assert f"{payload['min_cut']['value']:,.2f}" in result.stdout
    # ATL-F1: `--out`/`--json` (new on this command) write the same markdown/payload the tool
    # returns.
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_distance_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "distance.md"
    payload_path = tmp_path / "distance.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "distance",
            PERSONA,
            "--network",
            "entities",
            "--vectors",
            "attr:degree",
            "attr:mentions",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_distance",
        persona=PERSONA,
        network="entities",
        vector_a="attr:degree",
        vector_b="attr:mentions",
    )
    # `--out`/`--json` (new on this command, ATL-F1) each add their own trailing "wrote ..."
    # confirmation line to the console, which the tool's `markdown` never prints.
    console = result.stdout.rsplit("\nwrote ", 1)[0].rsplit("\nwrote ", 1)[0]
    assert _mask(out["markdown"]) == _mask(console.rstrip("\n"))
    # `--out` writes the full markdown, with the trailing newline the console/`markdown` value
    # above does not carry.
    assert _mask(out["markdown"]) == _mask(out_path.read_text().rstrip("\n"))
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_summarize_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "summarize.md"
    payload_path = tmp_path / "summarize.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "summarize",
            PERSONA,
            "--network",
            "entities",
            "--by",
            "community",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server, "sna_summarize", persona=PERSONA, network="entities", by="community", seed=1
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_overlap_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "overlap.md"
    payload_path = tmp_path / "overlap.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "overlap",
            PERSONA,
            "--network",
            "entities",
            "--method",
            "clique",
            "-k",
            "2",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server, "sna_overlap", persona=PERSONA, network="entities", method="clique", k=2, seed=1
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_hierarchy_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "hierarchy.md"
    payload_path = tmp_path / "hierarchy.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "hierarchy",
            PERSONA,
            "--samples",
            "5",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_hierarchy", persona=PERSONA, samples=5, seed=1)
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_highorder_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    """The planted corpus has one chunk per document, so §34.2's walk has no first-order
    transition to read and both the CLI and the tool refuse for the same reason -- there is no
    bigger fixture in this suite to ask for a network the memory walk is actually defined on."""
    out_path = tmp_path / "highorder.md"
    result = runner.invoke(app, ["sna", "highorder", PERSONA, "--out", str(out_path)])
    assert result.exit_code == 2

    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_highorder", persona=PERSONA)
    assert "error" in out


def test_sna_highorder_refuses_a_non_entity_network(
    cli_context: AppContext, related: InMemoryGraphStore
) -> None:
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_highorder", persona=PERSONA, network="speakers")
    assert "error" in out


def test_sna_community_local_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "community.md"
    payload_path = tmp_path / "community.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "community",
            PERSONA,
            "--network",
            "entities",
            "--seed",
            "product:alpha",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server, "sna_community", persona=PERSONA, network="entities", seed_node="product:alpha"
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_community_needs_exactly_one_mode(
    cli_context: AppContext, related: InMemoryGraphStore
) -> None:
    server = create_server(ServerState(context=cli_context))
    assert "error" in _call(server, "sna_community", persona=PERSONA)
    assert "error" in _call(
        server, "sna_community", persona=PERSONA, seed_node="product:alpha", temporal=True
    )


def test_sna_predict_eval_matches_cli(
    cli_context: AppContext, enriched: AppContext, tmp_path: Path
) -> None:
    out_path = tmp_path / "predict-eval.md"
    payload_path = tmp_path / "predict-eval.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "predict-eval",
            "test-pm",
            "--network",
            "entities",
            "--share",
            "0.34",
            "--seed",
            "7",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_predict_eval",
        persona="test-pm",
        network="entities",
        share=0.34,
        seed=7,
    )
    # The file gets `render_experiment`'s own text; the console (and this tool's markdown) leads
    # with a title line the file does not carry, and the console has two trailing "wrote ..."
    # confirmations (--out, --json) this tool never prints.
    console = result.stdout.rsplit("\nwrote ", 1)[0].rsplit("\nwrote ", 1)[0]
    assert _mask(out["markdown"]) == _mask(console)
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_predict_matches_cli(
    cli_context: AppContext, enriched: AppContext, tmp_path: Path
) -> None:
    payload_path = tmp_path / "predict.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "predict",
            "test-pm",
            "--method",
            "cn",
            "--share",
            "0.34",
            "--seed",
            "7",
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_predict", persona="test-pm", method="cn", share=0.34, seed=7)
    console = result.stdout.rsplit("\nwrote ", 1)[0]
    assert _mask(out["markdown"]) == _mask(console)
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_motifs_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "motifs.md"
    payload_path = tmp_path / "motifs.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "motifs",
            PERSONA,
            "--network",
            "entities",
            "--min-weight",
            "1",
            "--samples",
            "5",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server, "sna_motifs", persona=PERSONA, network="entities", min_weight=1, samples=5, seed=1
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_spread_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "spread.md"
    payload_path = tmp_path / "spread.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "spread",
            PERSONA,
            "--network",
            "entities",
            "--seeds",
            "product:alpha",
            "--steps",
            "5",
            "--runs",
            "3",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server,
        "sna_spread",
        persona=PERSONA,
        network="entities",
        seeds="product:alpha",
        steps=5,
        runs=3,
        seed=1,
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_robustness_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    out_path = tmp_path / "robustness.md"
    payload_path = tmp_path / "robustness.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "robustness",
            PERSONA,
            "--network",
            "entities",
            "--steps",
            "3",
            "--runs",
            "2",
            "--seed",
            "1",
            "--out",
            str(out_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(
        server, "sna_robustness", persona=PERSONA, network="entities", steps=3, runs=2, seed=1
    )
    assert _mask(out["markdown"]) == _mask(out_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_draw_matches_cli(
    cli_context: AppContext, related: InMemoryGraphStore, tmp_path: Path
) -> None:
    svg_path = tmp_path / "draw.svg"
    report_path = tmp_path / "draw.md"
    payload_path = tmp_path / "draw.json"
    result = runner.invoke(
        app,
        [
            "sna",
            "draw",
            PERSONA,
            "--network",
            "entities",
            "--seed",
            "1",
            "--out",
            str(svg_path),
            "--report",
            str(report_path),
            "--json",
            str(payload_path),
        ],
    )
    assert result.exit_code == 0, result.output
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_draw", persona=PERSONA, network="entities", seed=1)
    assert _mask(out["markdown"]) == _mask(report_path.read_text())
    assert _mask(out["payload"]) == _mask(json.loads(payload_path.read_text()))


def test_sna_complete_reports_missing_gnn_extra(
    cli_context: AppContext, related: InMemoryGraphStore
) -> None:
    """`torch` is not installed in this image, so both the CLI and the tool refuse before
    touching the persona -- the CLI with exit 2, the tool with a structured error."""
    from graphrag.sna import gnn

    assert not gnn.available()
    server = create_server(ServerState(context=cli_context))
    out = _call(server, "sna_complete", persona=PERSONA, key="region")
    assert "error" in out
    assert "graphrag[gnn]" in out["error"]


def test_every_report_command_carries_a_provenance_block(
    cli_context: AppContext, related: InMemoryGraphStore, enriched: AppContext
) -> None:
    """ATL-ENT-4: every report-producing `sna` command's payload has a `"provenance"` key, and
    every one of its markdown carries `## Provenance` -- so a future command that forgets to
    call `make_provenance` fails here rather than shipping silently. `sna_guide` and `sna_cache
    *` are not reports (`test_tool_registry_covers_every_sna_command` excludes the same two for
    the same reason) and are not in this table.

    Two commands that are real `sna_app` commands are deliberately not called here:
    `sna_highorder` (this suite's fixtures give it no document with more than one chunk, so
    §34.2's memory walk is never defined -- see `test_sna_highorder_matches_cli`'s own docstring
    -- and it always returns a structured error instead of a report) and `sna_complete` (`torch`
    is not installed in this image, so it always returns the missing-extra error). Both still go
    through `make_provenance` on their success path in `cli.py`/`mcp_server.py`; neither can
    reach it from a unit test's fixtures.
    """
    server = create_server(ServerState(context=cli_context))
    calls: list[tuple[str, dict[str, Any]]] = [
        ("sna_sample", {"persona": PERSONA, "size": 2, "network": "entities", "method": "induced"}),
        ("sna_analyze", {"persona": "test-pm", "network": "speakers", "runs": 2, "samples": 3}),
        ("sna_stances", {"persona": PERSONA}),
        (
            "sna_compare",
            {
                "persona": PERSONA,
                "network": "speakers",
                "since": "2025-01-01",
                "until": "2025-03-01",
                "since2": "2025-05-01",
                "until2": "2025-07-01",
            },
        ),
        ("sna_layers", {"persona": PERSONA, "network": "entities", "layers": "stance"}),
        (
            "sna_multilayer_communities",
            {"persona": PERSONA, "network": "entities", "layers": "stance", "seed": 1},
        ),
        ("sna_projections", {"persona": PERSONA, "network": "entities", "min_weight": 1}),
        (
            "sna_walks",
            {
                "persona": PERSONA,
                "network": "entities",
                "min_weight": 1,
                "source_node": "product:alpha",
                "target_node": "product:beta",
            },
        ),
        (
            "sna_distance",
            {
                "persona": PERSONA,
                "network": "entities",
                "vector_a": "attr:degree",
                "vector_b": "attr:mentions",
            },
        ),
        ("sna_backbone", {"persona": PERSONA, "network": "entities"}),
        ("sna_summarize", {"persona": PERSONA, "network": "entities", "by": "community"}),
        ("sna_degree", {"persona": PERSONA, "network": "entities", "bootstrap": 5}),
        ("sna_roles", {"persona": PERSONA, "network": "entities", "method": "jaccard"}),
        ("sna_ego", {"persona": PERSONA, "node": "product:alpha", "network": "entities"}),
        (
            "sna_community",
            {"persona": PERSONA, "network": "entities", "seed_node": "product:alpha"},
        ),
        ("sna_overlap", {"persona": PERSONA, "network": "entities", "method": "clique", "k": 2}),
        ("sna_hierarchy", {"persona": PERSONA, "samples": 5}),
        (
            "sna_predict_eval",
            {"persona": "test-pm", "network": "entities", "share": 0.34, "seed": 7},
        ),
        ("sna_predict", {"persona": "test-pm", "method": "cn", "share": 0.34, "seed": 7}),
        (
            "sna_robustness",
            {"persona": PERSONA, "network": "entities", "steps": 3, "runs": 2},
        ),
        (
            "sna_motifs",
            {"persona": PERSONA, "network": "entities", "min_weight": 1, "samples": 5},
        ),
        (
            "sna_spread",
            {
                "persona": PERSONA,
                "network": "entities",
                "seeds": "product:alpha",
                "steps": 5,
                "runs": 3,
            },
        ),
        ("sna_draw", {"persona": PERSONA, "network": "entities"}),
    ]
    seen = set()
    for tool, kwargs in calls:
        seen.add(tool)
        out = _call(server, tool, **kwargs)
        assert "error" not in out, (tool, out)
        assert "## Provenance" in out["markdown"], tool
        assert "provenance" in out["payload"], tool

    # `sna_export`'s payload is `nx.node_link_data`, not a report; the block rides on
    # `graph.graph`, which lands under `payload["graph"]` -- and, because GraphML/GEXF cannot
    # hold a nested attribute, as a JSON string rather than a dict (see `graphrag.sna.provenance`
    # and `_mask`'s own docstring).
    export_out = _call(server, "sna_export", persona=PERSONA, network="entities")
    seen.add("sna_export")
    assert json.loads(export_out["payload"]["graph"]["provenance"])["method"] == "export"

    cli_commands = typer.main.get_command(sna_app).commands  # type: ignore[attr-defined]
    every_report_command = {
        f"sna_{name.replace('-', '_')}"
        for name, command in cli_commands.items()
        if not hasattr(command, "commands") and name != "guide"
    }
    not_reachable_here = {"sna_highorder", "sna_complete"}
    assert seen == every_report_command - not_reachable_here


def test_unknown_persona_is_a_structured_error_everywhere(cli_context: AppContext) -> None:
    server = create_server(ServerState(context=cli_context))
    for tool, kwargs in (
        ("sna_analyze", {"persona": "nope"}),
        ("sna_degree", {"persona": "nope"}),
        ("sna_export", {"persona": "nope"}),
        ("sna_hierarchy", {"persona": "nope"}),
        ("sna_walks", {"persona": "nope", "source_node": "a", "target_node": "b"}),
    ):
        assert "error" in _call(server, tool, **kwargs), tool


def test_unknown_method_is_a_structured_error(
    cli_context: AppContext, related: InMemoryGraphStore
) -> None:
    server = create_server(ServerState(context=cli_context))
    for tool, kwargs in (
        ("sna_roles", {"persona": PERSONA, "method": "not-a-method"}),
        ("sna_overlap", {"persona": PERSONA, "method": "not-a-method"}),
        ("sna_predict", {"persona": PERSONA, "method": "not-a-method"}),
        ("sna_spread", {"persona": PERSONA, "model": "not-a-model"}),
    ):
        assert "error" in _call(server, tool, **kwargs), tool
