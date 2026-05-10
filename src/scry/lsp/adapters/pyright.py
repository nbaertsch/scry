"""Pyright LSP adapter for scry.

Handles ``.py`` and ``.pyi`` files; assumes pyright-langserver (or
basedpyright-langserver) as the server binary.

References
----------
DESIGN.md §5.3  — callHierarchy-based transitive drift
DESIGN.md §6.2  — binary allowlist (pyright-langserver, basedpyright-langserver, pylsp)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scry.lsp.adapters.uri import is_local_file_uri as _is_local_file_uri

# ─── Extension → languageId mapping ──────────────────────────────────

_PYTHON_EXTS: frozenset[str] = frozenset({".py", ".pyi"})


class PyrightAdapter:
    """Adapter for Pyright / BasedPyright language server.

    All methods are ``@staticmethod``; the class is never instantiated.

    Design notes
    ------------
    * ``processId`` is set to ``os.getpid()`` so that Pyright's
      ``vscode-languageserver/node`` watchdog can clean up the server
      after abnormal scry exit (review-w3c HIGH fix).
    * ``diagnosticMode: "openFilesOnly"`` keeps server memory bounded: Pyright
      will not crawl the full workspace unless explicitly asked.
    """

    LANGUAGES: tuple[str, ...] = ("python",)

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return ``"python"`` for local ``.py`` / ``.pyi`` URIs, else ``None``.

        Only ``file://`` URIs are accepted (review-w3c LOW: URI scheme
        validation).
        """
        if not _is_local_file_uri(uri):
            return None
        # Strip query/fragment components before checking the extension.
        path_part = uri.split("?")[0].split("#")[0]
        ext = Path(path_part).suffix.lower()
        return "python" if ext in _PYTHON_EXTS else None

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Build LSP ``initialize`` params for a Pyright session.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root; used as ``rootUri`` and
            the sole workspace folder.
        allow_untrusted:
            Propagated from the CLI flag (DESIGN.md §6.2); not currently
            used to alter Pyright params but reserved for future adapters
            that may enable experimental capabilities.
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
        }

    @staticmethod
    def initial_workspace_settings() -> dict[str, Any]:
        """Return Pyright workspace settings for ``workspace/didChangeConfiguration``.

        ``diagnosticMode: "openFilesOnly"`` prevents Pyright from crawling
        the entire workspace eagerly, keeping memory usage bounded — important
        when scry opens only the anchored files (per §5.3).
        """
        return {
            "python": {
                "analysis": {
                    "diagnosticMode": "openFilesOnly",
                },
            },
        }


__all__ = ["PyrightAdapter"]
