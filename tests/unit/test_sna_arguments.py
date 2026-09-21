"""``graphrag.sna.arguments`` (ATL-F1): the CLI's strings and a tool caller's native types both
parse to one canonical shape.

The "known answer" here is not a network -- there is nothing to build one over -- it is the two
input shapes each function must treat alike: a CLI's comma string / repeated ``"k=v"`` list
against an MCP tool's already-typed list / mapping. Every assertion below checks that both inputs
produce the identical canonical value, which is the property `tests/unit/test_mcp_sna.py::
test_filter_arguments_agree_between_cli_and_mcp` then checks end to end through a real command.
"""

from __future__ import annotations

import pytest

from graphrag.sna.arguments import (
    DEFAULT_MAX_SECONDS,
    parse_max_seconds,
    parse_sample_params,
    parse_types,
    parse_where,
)


class TestParseTypes:
    def test_none_stays_none(self) -> None:
        assert parse_types(None) is None

    def test_a_cli_comma_string_and_a_tool_list_agree(self) -> None:
        assert parse_types("a, b ,c") == parse_types(["a", "b", "c"]) == ["a", "b", "c"]

    def test_blanks_are_dropped_either_way(self) -> None:
        assert parse_types(" , a,, ") == parse_types(["", "a", "", " "]) == ["a"]

    def test_an_all_blank_input_is_none_not_an_empty_list(self) -> None:
        assert parse_types("") is None
        assert parse_types(",  ,") is None
        assert parse_types([]) is None
        assert parse_types(["", " "]) is None


class TestParseWhere:
    def test_none_stays_none(self) -> None:
        assert parse_where(None) is None

    def test_a_cli_key_value_list_and_a_tool_mapping_agree(self) -> None:
        assert (
            parse_where(["region=north", "line=life"])
            == parse_where({"region": "north", "line": "life"})
            == {"region": "north", "line": "life"}
        )

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert parse_where([" region = north "]) == {"region": "north"}

    def test_an_empty_input_is_none_not_an_empty_mapping(self) -> None:
        assert parse_where([]) is None
        assert parse_where({}) is None

    def test_a_malformed_pair_is_refused_with_the_names_option(self) -> None:
        with pytest.raises(ValueError, match=r"--where2 must look like key=value, got 'nope'"):
            parse_where(["nope"], "--where2")

    def test_a_key_given_twice_is_refused_rather_than_resolved(self) -> None:
        """A node holds one value per key, so the last-write-wins reading a plain dict update
        would give is refused instead: ``--where region=north --where region=south`` can only
        ever match nothing, and resolving it silently would answer a question nobody asked."""
        with pytest.raises(ValueError, match="region given twice"):
            parse_where(["region=north", "region=south"])

    def test_a_mapping_input_cannot_carry_a_duplicate_key_so_it_is_never_refused(self) -> None:
        """The CLI's repeated-string form is the only one that can name a key twice; a caller's
        own ``dict`` already enforces uniqueness before this function ever sees it."""
        assert parse_where({"region": "north"}) == {"region": "north"}


class TestParseSampleParams:
    def test_none_or_empty_is_an_empty_dict(self) -> None:
        assert parse_sample_params(None) == {}
        assert parse_sample_params([]) == {}
        assert parse_sample_params({}) == {}

    def test_a_cli_name_value_list_and_a_tool_mapping_agree_on_types(self) -> None:
        """``true``/``false`` become booleans and digits become numbers on the CLI's string
        side, so it lands on the same value a tool's already-typed mapping does."""
        assert (
            parse_sample_params(["k=3", "p=0.3", "neighbors=true"])
            == parse_sample_params({"k": 3, "p": 0.3, "neighbors": True})
            == {"k": 3, "p": 0.3, "neighbors": True}
        )

    def test_a_mapping_input_is_returned_unexamined(self) -> None:
        """Coercing an already-typed value back through string parsing would be the one way to
        *lose* information the CLI's own strings never carried in the first place -- a tool
        caller's ``3`` stays an ``int``, not a value re-guessed from ``str(3)``."""
        params = {"k": 3, "restart": 0.15}
        assert parse_sample_params(params) == params
        assert parse_sample_params(params) is not params  # a copy, not the caller's own dict

    def test_a_malformed_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"--param must look like name=value, got 'nope'"):
            parse_sample_params(["nope"])

    def test_an_unparseable_value_stays_a_string(self) -> None:
        assert parse_sample_params(["mode=snowball"]) == {"mode": "snowball"}


@pytest.mark.parametrize(
    ("given", "expected"), [(None, None), (0, None), (0.0, None), (-5.0, None), (45.0, 45.0)]
)
def test_parse_max_seconds_reads_zero_and_below_as_no_budget(
    given: float | None, expected: float | None
) -> None:
    """At the CLI and the MCP tool, `0` means no budget, the usual reading of a limit flag; the
    library's own `max_seconds=0.0` is a zero budget, and this is the one place that translates."""
    assert parse_max_seconds(given) == expected


def test_the_default_budget_is_five_minutes() -> None:
    assert DEFAULT_MAX_SECONDS == 300.0
