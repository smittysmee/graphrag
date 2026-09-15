"""Host-side helpers for Claude Code hooks.

Everything under this package runs *outside* Docker, with whatever ``python3`` is on the
user's PATH, so it is held to a stricter contract than the rest of ``graphrag``:

* standard library only (``yaml`` may be imported behind ``try``/``except`` with a fallback);
* Python 3.10 compatible syntax, always with ``from __future__ import annotations``;
* no ``graphrag.config.Settings`` -- the container's settings do not describe the host, so the
  few environment variables a hook needs are read here directly (see ``mcp_client.default_url``);
* fail silent: a hook that cannot answer prints nothing and exits 0.
"""

from __future__ import annotations

__all__ = ["mcp_client", "project", "session_card"]
