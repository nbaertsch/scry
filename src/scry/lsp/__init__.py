"""LSP subprocess infrastructure for scry.

Exposes five submodules:

* ``proto``            — JSON-RPC 2.0 wire encoding/decoding (DESIGN.md §11)
* ``manager``          — :class:`~scry.lsp.manager.LSPManager` lifecycle, allowlist
                         enforcement, and per-language session pool (DESIGN.md §6.2)
* ``closure``          — transitive call-closure walker via callHierarchy (DESIGN.md §5.3)
* ``full_resolution``  — full-mode symbol resolution (W6d)
* ``reverse``          — reverse-link queries: get_callers / get_subclasses (W6e)

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
from scry.lsp.full_resolution import (
    compute_closure_full,
    supports_full_mode,
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
from scry.lsp.reverse import (
    CallerRef,
    SubclassRef,
    get_callers,
    get_subclasses,
)

__all__ = [
    "LSP_ADAPTERS",
    "LSP_ALLOWLIST",
    "AdapterProtocol",
    "CalleeRef",
    "CallerRef",
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
    "SubclassRef",
    "compute_closure",
    "compute_closure_full",
    "get_adapter",
    "get_callers",
    "get_subclasses",
    "supports_full_mode",
]
