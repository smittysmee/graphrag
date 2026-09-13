"""A tiny standard-library client for the graphrag FastMCP streamable-HTTP endpoint.

Hooks run on the host, where neither ``fastmcp`` nor ``httpx`` is installed, so this speaks
the protocol directly over ``urllib``:

1. ``POST`` a JSON-RPC ``initialize``; the response carries an ``mcp-session-id`` header that
   every later request must echo back;
2. ``POST`` the ``notifications/initialized`` notification (no id, no result);
3. ``POST`` ``tools/call`` with ``{"name": ..., "arguments": {...}}``.

Bodies come back either as plain JSON or as ``text/event-stream`` (``event:``/``data:`` lines);
both are parsed. The HTTP round trip is injectable so tests never touch a socket.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

__all__ = [
    "DEFAULT_PORT",
    "HttpResponse",
    "McpClient",
    "McpUnavailable",
    "Transport",
    "connect",
    "default_url",
    "urllib_transport",
]

DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 2.0
PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "graphrag-hooks"
ACCEPT = "application/json, text/event-stream"
_SESSION_HEADER = "mcp-session-id"


class McpUnavailable(RuntimeError):  # noqa: N818 - waves 2/3 import this name from the brief
    """The MCP server could not be reached, or answered with an error."""


class HttpResponse(Protocol):
    """The part of ``http.client.HTTPResponse`` this client relies on."""

    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> Any: ...

    def read(self) -> bytes: ...


class Transport(Protocol):
    """Performs one HTTP round trip. Injected so tests can replay a canned handshake."""

    def __call__(self, request: urllib.request.Request, timeout: float) -> HttpResponse: ...


def default_url() -> str:
    """``http://localhost:<GRAPHRAG_MCP_PORT or 8765>/mcp``.

    Hooks run on the host, outside the container that ``graphrag.config.Settings`` describes,
    so this is the one place in the package that looks at the environment directly.
    """
    port = os.environ.get("GRAPHRAG_MCP_PORT", "").strip()
    if not port.isdigit():
        port = str(DEFAULT_PORT)
    return f"http://localhost:{port}/mcp"


def urllib_transport(request: urllib.request.Request, timeout: float) -> HttpResponse:
    """The real transport: ``urllib.request.urlopen`` with a hard timeout."""
    if request.type not in ("http", "https"):
        msg = f"refusing to open a {request.type!r} url"
        raise McpUnavailable(msg)
    # The URL is built from a fixed scheme/host and a port we validated; S310 does not apply.
    response: HttpResponse = urllib.request.urlopen(request, timeout=timeout)  # noqa: S310
    return response


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _parse_sse(body: str) -> dict[str, Any]:
    """Return the first ``data:`` payload that carries a JSON-RPC result or error."""
    fallback: dict[str, Any] = {}
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            message = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(message, dict):
            continue
        if "result" in message or "error" in message:
            return message
        if not fallback:
            fallback = message
    return fallback


def _parse_body(content_type: str, body: str) -> dict[str, Any]:
    text = body.strip()
    if not text:
        return {}
    if "text/event-stream" in content_type.lower() or text.startswith(("event:", "data:")):
        return _parse_sse(text)
    try:
        message = json.loads(text)
    except ValueError as exc:
        msg = f"MCP server returned a non-JSON body: {text[:120]!r}"
        raise McpUnavailable(msg) from exc
    if not isinstance(message, dict):
        msg = f"MCP server returned {type(message).__name__}, expected an object"
        raise McpUnavailable(msg)
    return message


def _header(response: HttpResponse, name: str) -> str:
    headers = response.headers
    if headers is None:
        return ""
    getter = getattr(headers, "get", None)
    value = getter(name) if callable(getter) else None
    return str(value) if value else ""


def _tool_payload(result: Any) -> Any:
    """Unwrap a ``tools/call`` result into the value the tool returned."""
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        raise McpUnavailable(_error_text(result))
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            text = first["text"]
            try:
                return json.loads(text)
            except ValueError:
                return text
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        # FastMCP wraps a non-object return value as ``{"result": ...}``.
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    return result


def _error_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return f"MCP tool error: {item['text'][:200]}"
    return "MCP tool error"


class McpClient:
    """Calls graphrag MCP tools over streamable HTTP. Not thread-safe; one per hook run."""

    def __init__(
        self,
        url: str,
        transport: Transport | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self._transport: Transport = urllib_transport if transport is None else transport
        self._session_id: str | None = None
        self._ready = False
        self._next_id = 0

    @property
    def session_id(self) -> str | None:
        """The ``mcp-session-id`` handed out by the server, once the handshake has run."""
        return self._session_id

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one MCP tool and return its decoded return value.

        Raises ``McpUnavailable`` on a connection error, a timeout, a non-200 response, a
        JSON-RPC error, or a tool that reported failure.
        """
        self._handshake()
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        return _tool_payload(result)

    def ping(self) -> bool:
        """True when the handshake succeeds. Never raises."""
        try:
            self._handshake()
        except McpUnavailable:
            return False
        return True

    # ------------------------------------------------------------------ internals

    def _handshake(self) -> None:
        if self._ready:
            return
        self._call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": "1"},
            },
        )
        self._call("notifications/initialized", {}, notification=True)
        self._ready = True

    def _call(
        self, method: str, params: dict[str, Any], *, notification: bool = False
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            self._next_id += 1
            body["id"] = self._next_id
        message = self._post(body)
        if notification:
            return message
        error = message.get("error")
        if error is not None:
            raise McpUnavailable(f"MCP error from {method}: {_jsonrpc_error(error)}")
        result = message.get("result")
        if result is None:
            msg = f"MCP server returned no result for {method}"
            raise McpUnavailable(msg)
        return result if isinstance(result, dict) else {"result": result}

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": ACCEPT,
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers[_SESSION_HEADER] = self._session_id
        request = urllib.request.Request(  # noqa: S310 - scheme checked in the transport
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            response = self._transport(request, self.timeout)
        except McpUnavailable:
            raise
        except urllib.error.HTTPError as exc:
            raise McpUnavailable(f"MCP server returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise McpUnavailable(f"MCP server unreachable at {self.url}: {exc}") from exc
        return self._read(response)

    def _read(self, response: HttpResponse) -> dict[str, Any]:
        status = int(response.status)
        if status >= 300 or status < 200:
            msg = f"MCP server returned HTTP {status}"
            raise McpUnavailable(msg)
        session = _header(response, _SESSION_HEADER)
        if session:
            self._session_id = session
        try:
            raw = response.read()
        except OSError as exc:
            raise McpUnavailable(f"MCP response could not be read: {exc}") from exc
        return _parse_body(_header(response, "content-type"), _decode(raw))


def _jsonrpc_error(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error)[:200]
    return str(error)[:200]


def connect(
    url: str | None = None,
    transport: Transport | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> McpClient | None:
    """A handshaken client, or ``None`` when the server is down.

    The convenience every hook wants: one call, no exception handling, a fast fail when
    nothing is listening.
    """
    client = McpClient(url or default_url(), transport=transport, timeout=timeout)
    return client if client.ping() else None
