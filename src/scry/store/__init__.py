"""scry.store — SQLite persistence layer (DESIGN.md §7).

Exports the public API surface used by the indexer (W2l), retrieval (W2d),
and MCP server (W2i).
"""

from scry.store.db import IntegrityError, LockTimeout, SchemaError, ScryDB, WriteLock

__all__ = [
    "IntegrityError",
    "LockTimeout",
    "SchemaError",
    "ScryDB",
    "WriteLock",
]
