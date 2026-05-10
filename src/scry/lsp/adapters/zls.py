"""Zig Language Server (ZLS) adapter for scry.

Handles ``.zig`` and ``.zon`` files via ``zls``.

ZLS callHierarchy support is in flight upstream but not yet shipped on
the master branch as of this writing.  We advertise the client
capability unconditionally so that ``session.supports("callHierarchyProvider")``
correctly reflects whatever the server announces in its initialize
response — when ZLS does ship the provider, scry picks it up
automatically; until then the manager treats Zig anchors as
``transitive_unsupported`` per DESIGN.md §5.3.

References
----------
DESIGN.md §5.3  — callHierarchy-based transitive drift
DESIGN.md §6.2  — binary allowlist (zls)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scry.lsp.adapters.uri import is_local_file_uri as _is_local_file_uri

# ─── Extension → languageId mapping ──────────────────────────────────

_ZIG_EXTS: frozenset[str] = frozenset({".zig", ".zon"})


class ZlsAdapter:
    """Adapter for ZLS (Zig Language Server).

    All methods are ``@staticmethod``; the class is never instantiated.

    ZLS capability note
    -------------------
    ZLS callHierarchy support is in flight upstream but not yet on the
    master branch.  scry advertises the client capability so that when
    ZLS does ship ``callHierarchyProvider``, ``session.supports()``
    reflects it without a code change.  Until then, Zig anchors are
    treated as ``transitive_unsupported`` (DESIGN.md §5.3).

    ``.zon`` note
    -------------
    ``.zon`` (Zig Object Notation) files are Zig-flavored data files parsed
    by the Zig compiler.  ZLS treats them as Zig for syntax purposes, so
    the same ``languageId = "zig"`` applies.
    """

    LANGUAGES: tuple[str, ...] = ("zig",)

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return ``"zig"`` for local ``.zig`` / ``.zon`` URIs, else ``None``.

        Only ``file://`` URIs are accepted (review-w3c LOW: URI scheme
        validation).
        """
        if not _is_local_file_uri(uri):
            return None
        path_part = uri.split("?")[0].split("#")[0]
        ext = Path(path_part).suffix.lower()
        return "zig" if ext in _ZIG_EXTS else None

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Build LSP ``initialize`` params for a ZLS session.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root.
        allow_untrusted:
            Reserved for future use (DESIGN.md §6.2).
        """
        root_uri = repo_root.as_uri()
        return {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": repo_root.name}],
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "openClose": True,
                        "change": 1,  # TextDocumentSyncKind.Full
                    },
                    "callHierarchy": {"dynamicRegistration": False},
                },
            },
            "clientInfo": {"name": "scry", "version": "0"},
        }

    @staticmethod
    def initial_workspace_settings() -> dict[str, Any]:
        """Return ZLS workspace settings for ``workspace/didChangeConfiguration``.

        ZLS does not require any post-initialize settings for scry's use
        case (callHierarchy + textDocument open/close).  Returns an empty
        dict; the manager skips the ``workspace/didChangeConfiguration``
        notification when settings are empty.
        """
        return {}


__all__ = ["ZlsAdapter"]
