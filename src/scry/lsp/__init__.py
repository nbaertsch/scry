"""LSP subprocess infrastructure for scry.

Exposes three submodules:

* ``proto``   — JSON-RPC 2.0 wire encoding/decoding (DESIGN.md §11)
* ``manager`` — :class:`~scry.lsp.manager.LSPManager` lifecycle, allowlist
                enforcement, and per-language session pool (DESIGN.md §6.2)
* ``closure`` — transitive call-closure walker via callHierarchy (DESIGN.md §5.3)

See Also
--------
DESIGN.md §5.3  — transitive drift via callHierarchy
DESIGN.md §6.2  — LSP binary allowlist (security)
DESIGN.md §10.5 — Windows .exe/.cmd shim spawning
DESIGN.md §11   — JSON-RPC over stdio tech stack
"""

from scry.lsp.adapters import (  # W3c — added
    ADAPTERS as LSP_ADAPTERS,
)
from scry.lsp.adapters import (
    AdapterProtocol,
    get_adapter,
)
from scry.lsp.closure import (
    CalleeRef,
    ClosureResult,
    ClosureStatus,
    compute_closure,
)
from scry.lsp.manager import (
    LSP_ALLOWLIST,
    LSPAllowlistViolation,
    LSPInitializeError,
    LSPLaunchError,
    LSPLaunchSpec,
    LSPManager,
    LSPSession,
)
from scry.lsp.proto import (
    LSPCodec,
    LSPMessage,
    LSPProtocolError,
    LSPStreamReader,
    LSPStreamWriter,
)

__all__ = [
    "LSP_ADAPTERS",
    "LSP_ALLOWLIST",
    "AdapterProtocol",
    "CalleeRef",
    "ClosureResult",
    "ClosureStatus",
    "LSPAllowlistViolation",
    "LSPCodec",
    "LSPInitializeError",
    "LSPLaunchError",
    "LSPLaunchSpec",
    "LSPManager",
    "LSPMessage",
    "LSPProtocolError",
    "LSPSession",
    "LSPStreamReader",
    "LSPStreamWriter",
    "compute_closure",
    "get_adapter",
]
