"""rust-analyzer LSP adapter for scry.

Handles ``.rs`` files via ``rust-analyzer``, the official Rust language
server.

References
----------
DESIGN.md §5.3  — callHierarchy-based transitive drift
DESIGN.md §6.2  — binary allowlist (rust-analyzer)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from scry.lsp.adapters.uri import is_local_file_uri as _is_local_file_uri

# ─── Extension → languageId mapping ──────────────────────────────────

_RUST_EXTS: frozenset[str] = frozenset({".rs"})


class RustAnalyzerAdapter:
    """Adapter for rust-analyzer (Rust language server).

    All methods are ``@staticmethod``; the class is never instantiated.

    Design notes
    ------------
    * ``processId`` is set to ``os.getpid()`` so that rust-analyzer's
      watchdog can clean up the server after abnormal scry exit.
    * ``initializationOptions`` disables expensive features scry never
      requests (cargo check on save, diagnostics) while keeping build scripts
      and proc-macro expansion enabled for accurate symbol resolution.
    * ``initial_workspace_settings()`` returns ``{}`` because no
      rust-analyzer setting meaningful to scry's use case has a separate
      runtime-tunable knob beyond what is set at initialization.
    """

    LANGUAGES: tuple[str, ...] = ("rust",)

    @staticmethod
    def language_id_for_uri(uri: str) -> str | None:
        """Return ``"rust"`` for local ``.rs`` URIs, else ``None``.

        Only ``file://`` URIs are accepted (review-w3c LOW: URI scheme
        validation).
        """
        if not _is_local_file_uri(uri):
            return None
        # Strip query/fragment before extension check.
        path_part = uri.split("?")[0].split("#")[0]
        ext = Path(path_part).suffix.lower()
        return "rust" if ext in _RUST_EXTS else None

    @staticmethod
    def prepare_initialize_params(repo_root: Path, allow_untrusted: bool) -> dict[str, Any]:
        """Build LSP ``initialize`` params for a rust-analyzer session.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root; used as ``rootUri`` and
            the sole workspace folder.
        allow_untrusted:
            Reserved for future use (DESIGN.md §6.2).

        initializationOptions notes
        ---------------------------
        rust-analyzer is configured to minimize overhead while preserving
        accurate symbol resolution for callHierarchy:

        * ``checkOnSave.enable: false`` — disables the ``cargo check`` run
          triggered on every save.  scry does not surface diagnostics, so
          check output is pure overhead.
        * ``cargo.buildScripts.enable: true`` — build scripts are required
          for accurate type and symbol resolution in crates that generate
          code (e.g. ``prost``, ``bindgen``).  Disabling them produces
          phantom symbol-resolution failures in common Rust crates.
        * ``procMacro.enable: true`` — proc macros (``#[derive(...)]``,
          ``tokio::main``, ``async_trait``, etc.) must be expanded for
          accurate callHierarchy results; disabling them produces incomplete
          or missing call graphs for idiomatic async Rust.
        * ``diagnostics.enable: false`` — scry does not request
          ``textDocument/publishDiagnostics``; suppressing them avoids
          rust-analyzer buffering per-file diagnostic sets and sending
          unsolicited notification traffic.

        Settings shape note (review-w6a BLOCKING fix): rust-analyzer reads
        ``initializationOptions`` as **nested JSON**, NOT flat dotted
        keys.  Sending ``"checkOnSave.enable": false`` as a top-level
        flat key is silently ignored; the correct path is
        ``checkOnSave: false`` (or the nested ``check.enable`` for newer
        versions).  This adapter sends both the legacy and modern shapes
        so it works across rust-analyzer 2024.* and 2025.* releases.
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
                # Modern shape (rust-analyzer 2024.10+):
                "check": {"enable": False},
                # Legacy shape (older rust-analyzer): both forms accepted.
                "checkOnSave": False,
                "cargo": {"buildScripts": {"enable": True}},
                "procMacro": {"enable": True},
                "diagnostics": {"enable": False},
            },
        }

    @staticmethod
    def initial_workspace_settings() -> dict[str, Any]:
        """Return rust-analyzer settings for ``workspace/didChangeConfiguration``.

        rust-analyzer reads all performance-critical settings from
        ``initializationOptions`` at startup; no runtime setting is needed
        for scry's callHierarchy / documentSymbol use case.  Returns an
        empty dict; the manager skips the ``workspace/didChangeConfiguration``
        notification when settings are empty.
        """
        return {}


__all__ = ["RustAnalyzerAdapter"]
