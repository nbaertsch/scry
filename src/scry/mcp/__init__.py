"""scry MCP server package (workstream W2i).

Exports the public server API: :class:`MCPServer`, :class:`MCPContext`,
and :class:`MCPServerError`.
"""

from scry.mcp.handlers import MCPContext, MCPServerError
from scry.mcp.server import MCPServer

__all__ = [
    "MCPContext",
    "MCPServer",
    "MCPServerError",
]
