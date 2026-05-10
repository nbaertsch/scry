"""Per-language LSP adapter registry for scry.

Each adapter module exports a single class providing three ``@staticmethod``
methods:

* ``prepare_initialize_params(repo_root, allow_untrusted)`` — LSP initialize params
* ``language_id_for_uri(uri)`` — LSP ``languageId`` for textDocument/didOpen
* ``initial_workspace_settings()`` — post-initialize ``workspace/didChangeConfiguration`` payload

Usage
-----
::

    from scry.lsp.adapters import get_adapter

    adapter = get_adapter("python")
    if adapter is not None:
        params = adapter.prepare_initialize_params(repo_root, allow_untrusted)
        settings = adapter.initial_workspace_settings()

Language-ID convention
-----------------------
``ADAPTERS`` keys are **scry-side language IDs** (e.g. ``"tsx"``), which
mirror the keys in ``LSP_ALLOWLIST``.  These differ from LSP-side
``languageId`` strings returned by ``language_id_for_uri``
(e.g. ``"typescriptreact"``).

Therefore ``get_adapter("typescriptreact")`` returns ``None`` — use the
scry-side ID ``"tsx"`` instead.  The distinction is documented here to
answer the question the spec raises: *"scry-side vs LSP-side IDs?"* — the
registry uses scry-side IDs exclusively.

Design: why a Protocol?
------------------------
``AdapterProtocol`` is defined for static-analysis documentation and
``isinstance``-free structural checking.  The runtime dict uses the
concrete Union alias ``_AnyAdapter`` so that mypy --strict can verify
that all adapter types carry the required methods without the ambiguity
of ``type[Protocol]`` semantics.

References
----------
DESIGN.md §5.3  — transitive code drift via callHierarchy
DESIGN.md §6.2  — LSP binary allowlist; language IDs align with allowlist keys
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeAlias

from scry.lsp.adapters.pyright import PyrightAdapter
from scry.lsp.adapters.typescript_ls import TypeScriptLSAdapter
from scry.lsp.adapters.uri import is_local_file_uri
from scry.lsp.adapters.zls import ZlsAdapter

__all__ = [
    "ADAPTERS",
    "AdapterProtocol",
    "PyrightAdapter",
    "TypeScriptLSAdapter",
    "ZlsAdapter",
    "get_adapter",
    "is_local_file_uri",
]


# ─── Protocol (structural contract) ──────────────────────────────────


class AdapterProtocol(Protocol):
    """Structural protocol describing a stateless per-language LSP adapter.

    Concrete adapter classes (``PyrightAdapter``, ``TypeScriptLSAdapter``,
    ``ZlsAdapter``) satisfy this protocol structurally; they are never
    instantiated.  The protocol exists for documentation and to allow
    downstream consumers to type-hint adapter objects without importing
    concrete classes.
    """

    LANGUAGES: ClassVar[tuple[str, ...]]

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return the LSP ``languageId`` for *uri*, or ``None`` if unhandled."""
        ...

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Return a complete LSP ``initialize`` params dict."""
        ...

    @staticmethod
    def initial_workspace_settings() -> dict[str, Any]:
        """Return settings for ``workspace/didChangeConfiguration`` (may be empty)."""
        ...


# ─── Runtime union type ───────────────────────────────────────────────

# Union of all concrete adapter *classes* (not instances).  Using an explicit
# Union rather than ``type[AdapterProtocol]`` avoids mypy --strict ambiguity
# around Protocol metaclass checking while preserving full static typing.
_AnyAdapter: TypeAlias = type[PyrightAdapter] | type[TypeScriptLSAdapter] | type[ZlsAdapter]


# ─── Registry ─────────────────────────────────────────────────────────

#: Maps **scry-side language IDs** → adapter class.
#: Keys mirror the :data:`~scry.lsp.manager.LSP_ALLOWLIST` entries for
#: languages that have a dedicated W3c adapter.  Languages in the allowlist
#: without an adapter (``go``, ``rust``) fall back to minimal inline params
#: in :class:`~scry.lsp.manager.LSPSession`.
ADAPTERS: dict[str, _AnyAdapter] = {
    "python": PyrightAdapter,
    "typescript": TypeScriptLSAdapter,
    "tsx": TypeScriptLSAdapter,
    "javascript": TypeScriptLSAdapter,
    "jsx": TypeScriptLSAdapter,
    "zig": ZlsAdapter,
}


# ─── Public helper ────────────────────────────────────────────────────


def get_adapter(language: str) -> _AnyAdapter | None:
    """Return the adapter class for *language*, or ``None`` if unknown.

    Parameters
    ----------
    language:
        A **scry-side** language ID (e.g. ``"python"``, ``"tsx"``).
        LSP-side ``languageId`` strings (e.g. ``"typescriptreact"``)
        are **not** valid keys here — see module docstring.

    Returns
    -------
    type[PyrightAdapter] | type[TypeScriptLSAdapter] | type[ZlsAdapter] | None
        The adapter class, or ``None`` when no adapter is registered.
    """
    return ADAPTERS.get(language)
