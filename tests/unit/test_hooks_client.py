"""The host-side MCP client, exercised against a fake transport (no socket, no patching)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from graphrag.hooks.mcp_client import (
    DEFAULT_PORT,
    McpClient,
    McpUnavailable,
    connect,
    default_url,
)

SESSION = "sess-abc123"
URL = "http://localhost:8765/mcp"


class FakeHeaders:
    """Case-insensitive header lookup, like ``http.client.HTTPMessage``."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = {k.lower(): v for k, v in values.items()}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._values.get(name.lower(), default)


class FakeResponse:
    def __init__(self, body: bytes = b"", status: int = 200, **headers: str) -> None:
        self.status = status
        self.headers = FakeHeaders(headers)
        self._body = body

    def read(self) -> bytes:
        return self._body


class Recorded:
    def __init__(self, request: urllib.request.Request, timeout: float) -> None:
        self.timeout = timeout
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.body = json.loads(request.data or b"{}")

    @property
    def method(self) -> str:
        return str(self.body.get("method", ""))

    @property
    def session(self) -> str | None:
        return self.headers.get("mcp-session-id")


def sse(payload: dict[str, Any]) -> bytes:
    return f"event: message\ndata: {json.dumps(payload)}\n\n".encode()


def rpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def tool_text(value: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value)}]}


class FakeTransport:
    """Replays a handshake and hands each ``tools/call`` the next canned response."""

    def __init__(self, *tool_responses: FakeResponse, session: str = SESSION) -> None:
        self.requests: list[Recorded] = []
        self._tool_responses = list(tool_responses)
        self._session = session

    def __call__(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        recorded = Recorded(request, timeout)
        self.requests.append(recorded)
        if recorded.method == "initialize":
            body = sse(rpc_result(recorded.body["id"], {"protocolVersion": "2025-06-18"}))
            return FakeResponse(body, **{"mcp-session-id": self._session})
        if recorded.method == "notifications/initialized":
            return FakeResponse(b"", status=202)
        if not self._tool_responses:
            msg = "the fake transport ran out of canned tool responses"
            raise AssertionError(msg)
        return self._tool_responses.pop(0)

    @property
    def methods(self) -> list[str]:
        return [r.method for r in self.requests]


def sse_tool(request_id: int, value: Any) -> FakeResponse:
    body = sse(rpc_result(request_id, tool_text(value)))
    return FakeResponse(body, **{"content-type": "text/event-stream"})


def json_tool(request_id: int, result: Any) -> FakeResponse:
    body = json.dumps(rpc_result(request_id, result)).encode()
    return FakeResponse(body, **{"content-type": "application/json"})


# ----------------------------------------------------------------------------- handshake


def test_handshake_runs_once_and_echoes_the_session_id() -> None:
    stats = {"graph": {"documents": 291}}
    transport = FakeTransport(sse_tool(2, stats), sse_tool(3, stats))
    client = McpClient(URL, transport=transport)

    assert client.call_tool("stats", {}) == stats
    assert client.call_tool("stats", {}) == stats

    assert transport.methods == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
    ]
    assert transport.requests[0].session is None
    assert all(r.session == SESSION for r in transport.requests[1:])
    assert client.session_id == SESSION


def test_request_headers_and_arguments() -> None:
    transport = FakeTransport(sse_tool(2, []))
    McpClient(URL, transport=transport, timeout=0.5).call_tool("topics", {"limit": 60})

    first, call = transport.requests[0], transport.requests[-1]
    assert first.headers["content-type"] == "application/json"
    assert first.headers["accept"] == "application/json, text/event-stream"
    assert call.timeout == 0.5
    assert call.body["params"] == {"name": "topics", "arguments": {"limit": 60}}
    assert call.body["id"] != first.body["id"]


# ----------------------------------------------------------------------------- bodies


def test_plain_json_body_is_parsed() -> None:
    transport = FakeTransport(json_tool(2, tool_text({"ok": True})))
    assert McpClient(URL, transport=transport).call_tool("stats", {}) == {"ok": True}


def test_sse_body_is_parsed() -> None:
    transport = FakeTransport(sse_tool(2, [{"topic": "growth", "count": 271}]))
    result = McpClient(URL, transport=transport).call_tool("topics", {})
    assert result == [{"topic": "growth", "count": 271}]


def test_structured_content_is_used_when_there_is_no_text_content() -> None:
    transport = FakeTransport(json_tool(2, {"structuredContent": {"documents": 291}}))
    assert McpClient(URL, transport=transport).call_tool("stats", {}) == {"documents": 291}


def test_structured_content_result_wrapper_is_unwrapped() -> None:
    transport = FakeTransport(json_tool(2, {"structuredContent": {"result": ["a", "b"]}}))
    assert McpClient(URL, transport=transport).call_tool("topics", {}) == ["a", "b"]


def test_non_json_text_content_comes_back_as_a_string() -> None:
    transport = FakeTransport(json_tool(2, {"content": [{"type": "text", "text": "a brief"}]}))
    assert McpClient(URL, transport=transport).call_tool("persona_brief", {}) == "a brief"


def test_sse_skips_non_result_events() -> None:
    body = b"event: ping\ndata: {}\n\n" + sse(rpc_result(2, tool_text({"n": 1})))
    transport = FakeTransport(FakeResponse(body, **{"content-type": "text/event-stream"}))
    assert McpClient(URL, transport=transport).call_tool("stats", {}) == {"n": 1}


# ----------------------------------------------------------------------------- failures


def test_jsonrpc_error_raises() -> None:
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 2, "error": {"code": -32602, "message": "unknown tool"}}
    ).encode()
    transport = FakeTransport(FakeResponse(body, **{"content-type": "application/json"}))
    with pytest.raises(McpUnavailable, match="unknown tool"):
        McpClient(URL, transport=transport).call_tool("nope", {})


def test_tool_error_raises() -> None:
    result = {"isError": True, "content": [{"type": "text", "text": "persona not found"}]}
    transport = FakeTransport(json_tool(2, result))
    with pytest.raises(McpUnavailable, match="persona not found"):
        McpClient(URL, transport=transport).call_tool("topics", {})


def test_non_200_raises() -> None:
    transport = FakeTransport(FakeResponse(b"", status=503))
    with pytest.raises(McpUnavailable, match="503"):
        McpClient(URL, transport=transport).call_tool("stats", {})


def test_missing_result_raises() -> None:
    transport = FakeTransport(FakeResponse(json.dumps({"jsonrpc": "2.0", "id": 2}).encode()))
    with pytest.raises(McpUnavailable):
        McpClient(URL, transport=transport).call_tool("stats", {})


def test_unparseable_body_raises() -> None:
    transport = FakeTransport(FakeResponse(b"<html>nope</html>"))
    with pytest.raises(McpUnavailable, match="non-JSON"):
        McpClient(URL, transport=transport).call_tool("stats", {})


def dead_transport(request: urllib.request.Request, timeout: float) -> FakeResponse:
    raise urllib.error.URLError("connection refused")


def http_error_transport(request: urllib.request.Request, timeout: float) -> FakeResponse:
    raise urllib.error.HTTPError(URL, 500, "boom", {}, None)  # type: ignore[arg-type]


def test_connection_error_raises_mcp_unavailable() -> None:
    with pytest.raises(McpUnavailable, match="unreachable"):
        McpClient(URL, transport=dead_transport).call_tool("stats", {})


def test_http_error_raises_mcp_unavailable() -> None:
    with pytest.raises(McpUnavailable, match="HTTP 500"):
        McpClient(URL, transport=http_error_transport).call_tool("stats", {})


def test_ping_and_connect_report_a_down_server() -> None:
    assert McpClient(URL, transport=dead_transport).ping() is False
    assert connect(URL, transport=dead_transport) is None


def test_connect_returns_a_ready_client() -> None:
    transport = FakeTransport(sse_tool(2, {"documents": 291}))
    client = connect(URL, transport=transport)
    assert client is not None
    assert client.call_tool("stats", {}) == {"documents": 291}
    assert transport.methods.count("initialize") == 1


# ----------------------------------------------------------------------------- url


def test_default_url_uses_the_port_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHRAG_MCP_PORT", "9999")
    assert default_url() == "http://localhost:9999/mcp"


@pytest.mark.parametrize("value", ["", "  ", "not-a-port"])
def test_default_url_falls_back_to_the_default_port(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("GRAPHRAG_MCP_PORT", value)
    assert default_url() == f"http://localhost:{DEFAULT_PORT}/mcp"


def test_default_url_without_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRAPHRAG_MCP_PORT", raising=False)
    assert default_url() == f"http://localhost:{DEFAULT_PORT}/mcp"
