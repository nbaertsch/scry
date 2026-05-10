"""gopls LSP adapter for scry.

Handles ``.go`` files via ``gopls``, the official Go language server.

References
----------
DESIGN.md §5.3  — callHierarchy-based transitive drift
DESIGN.md §6.2  — binary allowlist (gopls)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scry.lsp.adapters.uri import is_local_file_uri as _is_local_file_uri

# ─── Extension → languageId mapping ──────────────────────────────────

_GO_EXTS: frozenset[str] = frozenset({".go"})


class GoplsAdapter:
    """Adapter for gopls (Go language server).

    All methods are ``@staticmethod``; the class is never instantiated.

    Design notes
    ------------
    * ``processId`` is set to ``os.getpid()`` so that gopls's watchdog can
      clean up the server after abnormal scry exit.
    * ``initializationOptions`` disables features that scry never requests
      (semantic tokens, hover documentation) and sets memory caps where
      gopls exposes them.  gopls reads tuneable knobs from initialization
      options at startup and does not re-read them via
      ``workspace/didChangeConfiguration``.
    * ``initial_workspace_settings()`` returns ``{}`` because no gopls
      setting meaningful to scry's callHierarchy / documentSymbol use case
      has a runtime-tunable knob.
    """

    LANGUAGES: tuple[str, ...] = ("go",)

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return ``"go"`` for local ``.go`` URIs, else ``None``.

        Only ``file://`` URIs are accepted (review-w3c LOW: URI scheme
        validation).
        """
        if not _is_local_file_uri(uri):
            return None
        # Strip query/fragment before extension check.
        path_part = uri.split("?")[0].split("#")[0]
        ext = Path(path_part).suffix.lower()
        return "go" if ext in _GO_EXTS else None

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Build LSP ``initialize`` params for a gopls session.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root; used as ``rootUri`` and
            the sole workspace folder.
        allow_untrusted:
            Reserved for future use (DESIGN.md §6.2).

        initializationOptions notes
        ---------------------------
        gopls reads several settings from ``initializationOptions`` at startup:

        * ``build.directoryFilters`` — exclude vendored / dependency trees
          recursively.  The ``-**/`` glob prefix is required to match
          nested ``vendor/`` or ``node_modules/`` (review-w6a MEDIUM fix:
          previously used flat ``-vendor`` which only matched at root).
        * ``ui.semanticTokens: false`` — scry never requests semantic tokens;
          disabling them prevents gopls from computing and buffering token
          data for every opened file.
        * ``ui.documentation.hoverKind: "NoDocumentation"`` — scry does not
          request hover responses; suppressing doc extraction reduces per-
          package load time on large workspaces.
        * ``maxFileCacheBytes`` (1 GiB) — gopls's only documented memory
          cap; bounds in-memory file cache so large monorepos don't OOM
          the analyser.
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
                    "documentSymbol": {"dynamicRegistration": False},
                },
            },
            "clientInfo": {"name": "scry", "version": "0"},
            "initializationOptions": {
                "build.directoryFilters": ["-**/vendor", "-**/node_modules"],
                "ui.semanticTokens": False,
                "ui.documentation.hoverKind": "NoDocumentation",
                "maxFileCacheBytes": 1024 * 1024 * 1024,  # 1 GiB
            },
        }

    @staticmethod
    def initial_workspace_settings() -> dict[str, Any]:
        """Return gopls workspace settings for ``workspace/didChangeConfiguration``.

        gopls reads all performance-critical knobs from
        ``initializationOptions`` at startup; no runtime setting is needed
        for scry's callHierarchy / documentSymbol use case.  Returns an
        empty dict; the manager skips the ``workspace/didChangeConfiguration``
        notification when settings are empty.
        """
        return {}


__all__ = ["GoplsAdapter"]
