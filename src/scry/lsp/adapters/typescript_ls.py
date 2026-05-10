"""TypeScript / JavaScript LSP adapter for scry.

Handles ``.ts``, ``.tsx``, ``.js``, ``.jsx``, ``.mts``, ``.cts``,
``.mjs``, ``.cjs`` files via ``typescript-language-server``.

On Windows, ``typescript-language-server`` is typically npm-installed as a
``.cmd`` shim.  The W3a ``_build_spawn_cmd`` already handles the wrapping;
this adapter just provides the initialize params and workspace settings.

References
----------
DESIGN.md §5.3  — callHierarchy-based transitive drift
DESIGN.md §6.2  — binary allowlist (typescript-language-server)
DESIGN.md §10.5 — Windows .cmd shim spawning
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scry.lsp.adapters.uri import is_local_file_uri as _is_local_file_uri

# ─── Extension → languageId mapping ──────────────────────────────────

# typescript-language-server uses LSP languageId strings that differ from
# scry-side language IDs (e.g. scry "tsx" → LSP "typescriptreact").
_EXT_TO_LANGUAGE_ID: dict[str, str] = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascriptreact",
}


class TypeScriptLSAdapter:
    """Adapter for ``typescript-language-server`` (TypeScript, JavaScript).

    All methods are ``@staticmethod``; the class is never instantiated.

    Scry-side language IDs handled
    --------------------------------
    ``"typescript"``, ``"tsx"``, ``"javascript"``, ``"jsx"`` — all routed
    to the same server binary (typescript-language-server supports all four).

    Windows note
    ------------
    ``typescript-language-server`` is commonly npm-installed as a ``.cmd``
    shim (DESIGN.md §10.5).  The W3a ``_build_spawn_cmd`` wraps it in
    ``cmd.exe /C`` transparently; args must pass the metachar-safety check.

    Memory cap
    ----------
    ``tsserver.maxTsServerMemory`` is set to 2 048 MB to prevent tsserver
    from consuming unlimited heap in large monorepos.
    """

    LANGUAGES: tuple[str, ...] = ("typescript", "tsx", "javascript", "jsx")

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return the LSP ``languageId`` for *uri*, or ``None`` if not handled.

        Only ``file://`` URIs are accepted — virtual schemes (``http``,
        ``https``, ``data``, ``scry``, ``untitled``, etc.) return None to
        prevent accidentally routing non-local documents into a local LSP
        session (review-w3c LOW: URI scheme validation).

        Note: LSP ``languageId`` strings differ from scry-side language IDs.

        +------------+------------------+
        | Extension  | ``languageId``   |
        +============+==================+
        | .ts .mts   | typescript       |
        | .cts       | typescript       |
        | .tsx       | typescriptreact  |
        | .js .mjs   | javascript       |
        | .cjs       | javascript       |
        | .jsx       | javascriptreact  |
        +------------+------------------+
        """
        if not _is_local_file_uri(uri):
            return None
        path_part = uri.split("?")[0].split("#")[0]
        ext = Path(path_part).suffix.lower()
        return _EXT_TO_LANGUAGE_ID.get(ext)

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Build LSP ``initialize`` params for a typescript-language-server session.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root.
        allow_untrusted:
            Reserved for future use (DESIGN.md §6.2).

        Memory cap
        ----------
        ``maxTsServerMemory`` is sent via ``initializationOptions`` (NOT
        via ``workspace/didChangeConfiguration``) because
        typescript-language-server reads the value from initialization
        options at tsserver-spawn time and ignores later updates
        (review-w3c MEDIUM fix).
        """
        root_uri = repo_root.as_uri()
        return {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": repo_root.name}],
            "initializationOptions": {
                "maxTsServerMemory": 2048,
            },
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
        """Return typescript-language-server settings for ``workspace/didChangeConfiguration``.

        Note: ``maxTsServerMemory`` is NOT here — it is sent via
        ``initializationOptions`` in :meth:`prepare_initialize_params`
        because typescript-language-server only reads it at startup.

        * ``preferences.includeCompletionsForModuleExports``: disabled to skip
          expensive module-export completions that scry never requests.  Both
          the ``typescript`` and ``javascript`` sections are set so the
          setting applies regardless of the open file type.
        """
        return {
            "typescript": {
                "preferences": {
                    "includeCompletionsForModuleExports": False,
                },
            },
            "javascript": {
                "preferences": {
                    "includeCompletionsForModuleExports": False,
                },
            },
        }


__all__ = ["TypeScriptLSAdapter"]
