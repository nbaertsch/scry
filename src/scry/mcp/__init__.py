"""scry MCP server package (workstream W2i).

Exports the public server API: :class:`MCPServer`, :class:`MCPContext`,
and :class:`MCPServerError`.

UAT-R5-1 fix: ``MCPServer`` is loaded LAZILY via ``__getattr__`` because
its module pulls in ``fastmcp`` (~2.4s import) which in turn pulls in
``mcp`` (~0.9s).  CLI commands that don't touch the MCP layer (the
common case — search, status, link, check, …) should not pay that
3.3-second tax.  ``MCPContext`` and ``MCPServerError`` are cheap and
remain eagerly imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scry.mcp.handlers import MCPContext, MCPServerError

if TYPE_CHECKING:
    from scry.mcp.server import MCPServer

__all__ = [
    "MCPContext",
    "MCPServer",
    "MCPServerError",
]


def __getattr__(name: str) -> Any:
    """Lazy-load :class:`MCPServer` to avoid the eager fastmcp import."""
    if name == "MCPServer":
        from scry.mcp.server import MCPServer as _MCPServer

        return _MCPServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
