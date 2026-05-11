"""Tests for the RustAnalyzerAdapter (W6a).

Unit tests (no subprocess):
* ``language_id_for_uri`` returns ``"rust"`` for ``.rs`` URIs
* ``language_id_for_uri`` returns ``None`` for foreign extensions
* ``language_id_for_uri`` returns ``None`` for non-file URI schemes
* ``language_id_for_uri`` is case-insensitive on the extension
* ``prepare_initialize_params`` returns all required LSP keys
* ``capabilities.textDocument.callHierarchy`` is advertised
* ``capabilities.textDocument.documentSymbol`` is advertised
* ``initializationOptions`` contains expected rust-analyzer settings
* ``initial_workspace_settings()`` returns a dict (empty)
* ``get_adapter("rust")`` returns ``RustAnalyzerAdapter``
* ``RustAnalyzerAdapter.LANGUAGES`` includes ``"rust"``

Integration tests (spawn fake_lsp.py):
* ``LSPSession.start()`` with language="rust" succeeds and reports capabilities
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scry.lsp.adapters import get_adapter
from scry.lsp.adapters.rust_analyzer import RustAnalyzerAdapter

FAKE_LSP = Path(__file__).parent / "fixtures" / "fake_lsp.py"

# ─── language_id_for_uri ──────────────────────────────────────────────


def test_rust_analyzer_uri_rs() -> None:
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/src/main.rs") == "rust"


def test_rust_analyzer_uri_rs_upper_case() -> None:
    """Extension check is case-insensitive."""
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/Foo.RS") == "rust"


def test_rust_analyzer_uri_foreign_extensions() -> None:
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/foo.go") is None
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/foo.py") is None
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/foo.ts") is None
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/foo.zig") is None
    assert RustAnalyzerAdapter.language_id_for_uri("file:///repo/Cargo.toml") is None


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/main.rs",
        "http://example.com/main.rs",
        "data:text/plain,main.rs",
        "scry://repo/main.rs",
        "untitled:Untitled-1.rs",
        "",
        "/abs/path/to/main.rs",
        "relative/path.rs",
        "ftp://example.com/main.rs",
    ],
)
def test_rust_analyzer_rejects_non_file_uris(uri: str) -> None:
    """Regression (review-w3c LOW): non-file URIs must return None."""
    assert RustAnalyzerAdapter.language_id_for_uri(uri) is None


def test_rust_analyzer_accepts_localhost_file_uri() -> None:
    """RFC 8089: file://localhost/ is a valid local file URI."""
    assert RustAnalyzerAdapter.language_id_for_uri("file://localhost/repo/main.rs") == "rust"


def test_rust_analyzer_rejects_remote_host_file_uri() -> None:
    """file://remotehost/ is not a local URI (review-w3c LOW)."""
    assert RustAnalyzerAdapter.language_id_for_uri("file://nas/repo/main.rs") is None


# ─── prepare_initialize_params — required keys ───────────────────────

_REQUIRED_KEYS = {
    "processId",
    "rootUri",
    "capabilities",
    "workspaceFolders",
    "initializationOptions",
}


def test_rust_analyzer_init_params_has_required_keys(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    missing = _REQUIRED_KEYS - params.keys()
    assert not missing, f"RustAnalyzerAdapter.prepare_initialize_params missing keys: {missing}"


def test_rust_analyzer_process_id_is_pid(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["processId"] == os.getpid()


def test_rust_analyzer_root_uri_matches_repo(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["rootUri"] == tmp_path.as_uri()


def test_rust_analyzer_workspace_folders_non_empty(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    folders = params["workspaceFolders"]
    assert isinstance(folders, list) and len(folders) >= 1
    assert folders[0]["uri"] == tmp_path.as_uri()
    assert folders[0]["name"] == tmp_path.name


def test_rust_analyzer_client_info(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["clientInfo"]["name"] == "scry"


# ─── capabilities ─────────────────────────────────────────────────────


def test_rust_analyzer_call_hierarchy_capability_advertised(tmp_path: Path) -> None:
    """rust-analyzer adapter MUST advertise textDocument.callHierarchy (DESIGN.md §5.3)."""
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert "callHierarchy" in params["capabilities"].get("textDocument", {}), (
        "RustAnalyzerAdapter did not advertise textDocument.callHierarchy"
    )


def test_rust_analyzer_document_symbol_capability_advertised(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert "documentSymbol" in params["capabilities"].get("textDocument", {})


def test_rust_analyzer_synchronization_capability(tmp_path: Path) -> None:
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    sync = params["capabilities"]["textDocument"]["synchronization"]
    assert sync["openClose"] is True
    assert sync["change"] == 1  # TextDocumentSyncKind.Full


# ─── initializationOptions ────────────────────────────────────────────


def test_rust_analyzer_check_on_save_disabled(tmp_path: Path) -> None:
    """cargo check on save is expensive; scry does not surface diagnostics.

    Updated for review-w6a BLOCKING fix: rust-analyzer expects nested
    JSON, not flat dotted keys.  We send both the modern nested
    ``check.enable`` and the legacy ``checkOnSave`` for compatibility
    across versions.
    """
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    init = params["initializationOptions"]
    # Modern nested form (rust-analyzer 2024.10+):
    assert init["check"]["enable"] is False
    # Legacy form (older rust-analyzer):
    assert init["checkOnSave"] is False


def test_rust_analyzer_build_scripts_enabled(tmp_path: Path) -> None:
    """Build scripts are required for accurate symbol resolution (prost, bindgen, etc.).

    Settings shape is nested per review-w6a BLOCKING fix.
    """
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["cargo"]["buildScripts"]["enable"] is True


def test_rust_analyzer_proc_macro_enabled(tmp_path: Path) -> None:
    """Proc macros must be expanded for accurate callHierarchy on async Rust."""
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["procMacro"]["enable"] is True


def test_rust_analyzer_diagnostics_disabled(tmp_path: Path) -> None:
    """scry does not request publishDiagnostics; suppressing avoids buffering overhead."""
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["diagnostics"]["enable"] is False


def test_rust_analyzer_settings_shape_is_nested_not_flat(tmp_path: Path) -> None:
    """Regression (review-w6a BLOCKING): rust-analyzer initializationOptions
    must use nested JSON paths; flat dotted keys (``cargo.buildScripts.enable``)
    are silently ignored by rust-analyzer, leaving the intended performance
    controls disabled.
    """
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, False)
    init = params["initializationOptions"]
    # No flat dotted keys at the top level.
    flat_keys = [k for k in init if "." in k]
    assert flat_keys == [], (
        f"rust-analyzer initializationOptions must be nested, "
        f"not flat dotted; found flat keys: {flat_keys}"
    )


# ─── initial_workspace_settings ──────────────────────────────────────


def test_rust_analyzer_initial_workspace_settings_is_dict() -> None:
    settings = RustAnalyzerAdapter.initial_workspace_settings()
    assert isinstance(settings, dict)


def test_rust_analyzer_initial_workspace_settings_is_empty() -> None:
    """All rust-analyzer knobs relevant to scry are sent via initializationOptions."""
    assert RustAnalyzerAdapter.initial_workspace_settings() == {}


# ─── Registry ─────────────────────────────────────────────────────────


def test_get_adapter_rust_returns_rust_analyzer() -> None:
    assert get_adapter("rust") is RustAnalyzerAdapter


def test_rust_analyzer_languages_constant_includes_rust() -> None:
    assert "rust" in RustAnalyzerAdapter.LANGUAGES


# ─── allow_untrusted param is accepted ───────────────────────────────


def test_rust_analyzer_prepare_accepts_allow_untrusted_true(tmp_path: Path) -> None:
    """allow_untrusted=True must not raise; it is reserved for future use."""
    params = RustAnalyzerAdapter.prepare_initialize_params(tmp_path, True)
    assert "processId" in params


# ─── Integration ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_rust_analyzer_adapter_params_reach_fake_lsp(tmp_path: Path) -> None:
    """Integration: LSPSession.start() with language='rust' uses RustAnalyzerAdapter params.

    The fake LSP responds with canned capabilities regardless of init params,
    so this test verifies the full adapter → session lifecycle without
    requiring a real rust-analyzer installation.
    """
    from scry.lsp.manager import LSPLaunchSpec, LSPSession

    spec = LSPLaunchSpec(
        language="rust",
        command=sys.executable,
        args=[str(FAKE_LSP)],
        cwd=tmp_path,
    )
    session = LSPSession(language="rust", spec=spec, allow_untrusted=False)
    await session.start()

    assert session.is_alive
    assert session.supports("callHierarchyProvider")
    assert session.supports("textDocumentSync.openClose")
    assert isinstance(session.capabilities, dict) and session.capabilities

    await session.shutdown()


# uat-r5-5 pr-d noise
