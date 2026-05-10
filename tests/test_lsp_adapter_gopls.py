"""Tests for the GoplsAdapter (W6a).

Unit tests (no subprocess):
* ``language_id_for_uri`` returns ``"go"`` for ``.go`` URIs
* ``language_id_for_uri`` returns ``None`` for foreign extensions
* ``language_id_for_uri`` returns ``None`` for non-file URI schemes
* ``language_id_for_uri`` is case-insensitive on the extension
* ``prepare_initialize_params`` returns all required LSP keys
* ``capabilities.textDocument.callHierarchy`` is advertised
* ``capabilities.textDocument.documentSymbol`` is advertised
* ``initializationOptions`` contains expected gopls settings
* ``initial_workspace_settings()`` returns a dict (empty)
* ``get_adapter("go")`` returns ``GoplsAdapter``
* ``GoplsAdapter.LANGUAGES`` includes ``"go"``

Integration tests (spawn fake_lsp.py):
* ``LSPSession.start()`` with language="go" succeeds and reports capabilities
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scry.lsp.adapters import get_adapter
from scry.lsp.adapters.gopls import GoplsAdapter

FAKE_LSP = Path(__file__).parent / "fixtures" / "fake_lsp.py"

# ─── language_id_for_uri ──────────────────────────────────────────────


def test_gopls_uri_go() -> None:
    assert GoplsAdapter.language_id_for_uri("file:///repo/src/main.go") == "go"


def test_gopls_uri_go_upper_case() -> None:
    """Extension check is case-insensitive."""
    assert GoplsAdapter.language_id_for_uri("file:///repo/Foo.GO") == "go"


def test_gopls_uri_foreign_extensions() -> None:
    assert GoplsAdapter.language_id_for_uri("file:///repo/foo.rs") is None
    assert GoplsAdapter.language_id_for_uri("file:///repo/foo.py") is None
    assert GoplsAdapter.language_id_for_uri("file:///repo/foo.ts") is None
    assert GoplsAdapter.language_id_for_uri("file:///repo/foo.zig") is None
    assert GoplsAdapter.language_id_for_uri("file:///repo/foo.c") is None


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/main.go",
        "http://example.com/main.go",
        "data:text/plain,main.go",
        "scry://repo/main.go",
        "untitled:Untitled-1.go",
        "",
        "/abs/path/to/main.go",
        "relative/path.go",
        "ftp://example.com/main.go",
    ],
)
def test_gopls_rejects_non_file_uris(uri: str) -> None:
    """Regression (review-w3c LOW): non-file URIs must return None."""
    assert GoplsAdapter.language_id_for_uri(uri) is None


def test_gopls_accepts_localhost_file_uri() -> None:
    """RFC 8089: file://localhost/ is a valid local file URI."""
    assert GoplsAdapter.language_id_for_uri("file://localhost/repo/main.go") == "go"


def test_gopls_rejects_remote_host_file_uri() -> None:
    """file://remotehost/ is not a local URI (review-w3c LOW)."""
    assert GoplsAdapter.language_id_for_uri("file://nas/repo/main.go") is None


# ─── prepare_initialize_params — required keys ───────────────────────

_REQUIRED_KEYS = {
    "processId",
    "rootUri",
    "capabilities",
    "workspaceFolders",
    "initializationOptions",
}


def test_gopls_init_params_has_required_keys(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    missing = _REQUIRED_KEYS - params.keys()
    assert not missing, f"GoplsAdapter.prepare_initialize_params missing keys: {missing}"


def test_gopls_process_id_is_pid(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["processId"] == os.getpid()


def test_gopls_root_uri_matches_repo(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["rootUri"] == tmp_path.as_uri()


def test_gopls_workspace_folders_non_empty(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    folders = params["workspaceFolders"]
    assert isinstance(folders, list) and len(folders) >= 1
    assert folders[0]["uri"] == tmp_path.as_uri()
    assert folders[0]["name"] == tmp_path.name


def test_gopls_client_info(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["clientInfo"]["name"] == "scry"


# ─── capabilities ─────────────────────────────────────────────────────


def test_gopls_call_hierarchy_capability_advertised(tmp_path: Path) -> None:
    """gopls adapter MUST advertise textDocument.callHierarchy (DESIGN.md §5.3)."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert "callHierarchy" in params["capabilities"].get("textDocument", {}), (
        "GoplsAdapter did not advertise textDocument.callHierarchy"
    )


def test_gopls_document_symbol_capability_advertised(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert "documentSymbol" in params["capabilities"].get("textDocument", {})


def test_gopls_synchronization_capability(tmp_path: Path) -> None:
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    sync = params["capabilities"]["textDocument"]["synchronization"]
    assert sync["openClose"] is True
    assert sync["change"] == 1  # TextDocumentSyncKind.Full


# ─── initializationOptions ────────────────────────────────────────────


def test_gopls_directory_filters_exclude_vendor(tmp_path: Path) -> None:
    """``vendor/`` excluded recursively (review-w6a MEDIUM: needs ``-**/`` prefix)."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    filters: list[str] = params["initializationOptions"]["build.directoryFilters"]
    assert "-**/vendor" in filters


def test_gopls_directory_filters_exclude_node_modules(tmp_path: Path) -> None:
    """``node_modules/`` excluded recursively."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    filters: list[str] = params["initializationOptions"]["build.directoryFilters"]
    assert "-**/node_modules" in filters


def test_gopls_max_file_cache_bytes_set(tmp_path: Path) -> None:
    """gopls memory cap (review-w6a LOW): bound the in-memory file cache
    so huge monorepos don't OOM the analyser.
    """
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    cap = params["initializationOptions"]["maxFileCacheBytes"]
    assert isinstance(cap, int)
    assert cap > 0


def test_gopls_semantic_tokens_disabled(tmp_path: Path) -> None:
    """gopls must not be asked to compute semantic tokens (scry never uses them)."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["ui.semanticTokens"] is False


def test_gopls_hover_kind_no_documentation(tmp_path: Path) -> None:
    """Hover doc extraction is disabled to reduce per-package load time."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["ui.documentation.hoverKind"] == "NoDocumentation"


# ─── initial_workspace_settings ──────────────────────────────────────


def test_gopls_initial_workspace_settings_is_dict() -> None:
    settings = GoplsAdapter.initial_workspace_settings()
    assert isinstance(settings, dict)


def test_gopls_initial_workspace_settings_is_empty() -> None:
    """All gopls knobs relevant to scry are sent via initializationOptions."""
    assert GoplsAdapter.initial_workspace_settings() == {}


# ─── Registry ─────────────────────────────────────────────────────────


def test_get_adapter_go_returns_gopls() -> None:
    assert get_adapter("go") is GoplsAdapter


def test_gopls_languages_constant_includes_go() -> None:
    assert "go" in GoplsAdapter.LANGUAGES


# ─── allow_untrusted param is accepted ───────────────────────────────


def test_gopls_prepare_accepts_allow_untrusted_true(tmp_path: Path) -> None:
    """allow_untrusted=True must not raise; it is reserved for future use."""
    params = GoplsAdapter.prepare_initialize_params(tmp_path, True)
    assert "processId" in params


# ─── Integration ─────────────────────────────────────────────────────


@pytest.mark.integration
async def test_gopls_adapter_params_reach_fake_lsp(tmp_path: Path) -> None:
    """Integration: LSPSession.start() with language='go' uses GoplsAdapter params.

    The fake LSP responds with canned capabilities regardless of init params,
    so this test verifies the full adapter → session lifecycle without
    requiring a real gopls installation.
    """
    from scry.lsp.manager import LSPLaunchSpec, LSPSession

    spec = LSPLaunchSpec(
        language="go",
        command=sys.executable,
        args=[str(FAKE_LSP)],
        cwd=tmp_path,
    )
    session = LSPSession(language="go", spec=spec, allow_untrusted=False)
    await session.start()

    assert session.is_alive
    assert session.supports("callHierarchyProvider")
    assert session.supports("textDocumentSync.openClose")
    assert isinstance(session.capabilities, dict) and session.capabilities

    await session.shutdown()
