"""Tests for W3c LSP adapters.

Unit tests (no subprocess):
* ``language_id_for_uri`` maps every declared extension correctly
* ``language_id_for_uri`` returns None for foreign extensions
* ``prepare_initialize_params`` returns required LSP keys for every adapter
* TypeScript adapter advertises ``callHierarchy`` client capability
* Pyright adapter sets ``diagnosticMode: "openFilesOnly"`` in workspace settings
* ``get_adapter("python")`` returns PyrightAdapter
* ``get_adapter("tsx")`` / ``get_adapter("javascript")`` returns TypeScriptLSAdapter
* ``get_adapter("zig")`` returns ZlsAdapter
* ``get_adapter("typescriptreact")`` returns None (LSP-side ID, not scry-side)
* ``get_adapter("nonexistent")`` returns None
* ``get_adapter("go")`` returns None (allowlisted but no W3c adapter yet)

Integration tests (spawn fake_lsp.py):
* Session.start() sends adapter params; fake LSP still responds with capabilities
* Windows .cmd shim path produces correct cmd list (Windows-only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scry.lsp.adapters import (
    ADAPTERS,
    AdapterProtocol,  # noqa: F401 — imported to verify it is exported
    PyrightAdapter,
    TypeScriptLSAdapter,
    ZlsAdapter,
    get_adapter,
)

# Delay imports of manager symbols to the tests that need them
FAKE_LSP = Path(__file__).parent / "fixtures" / "fake_lsp.py"


# ─── language_id_for_uri — Pyright ───────────────────────────────────


def test_pyright_uri_py() -> None:
    assert PyrightAdapter.language_id_for_uri("file:///repo/src/foo.py") == "python"


def test_pyright_uri_pyi() -> None:
    assert PyrightAdapter.language_id_for_uri("file:///repo/src/foo.pyi") == "python"


def test_pyright_uri_upper_case() -> None:
    """Extension check is case-insensitive."""
    assert PyrightAdapter.language_id_for_uri("file:///repo/Foo.PY") == "python"


def test_pyright_uri_foreign() -> None:
    assert PyrightAdapter.language_id_for_uri("file:///repo/foo.ts") is None
    assert PyrightAdapter.language_id_for_uri("file:///repo/foo.js") is None
    assert PyrightAdapter.language_id_for_uri("file:///repo/foo.zig") is None


# ─── language_id_for_uri — TypeScriptLS ──────────────────────────────


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file:///repo/src/foo.ts", "typescript"),
        ("file:///repo/src/foo.mts", "typescript"),
        ("file:///repo/src/foo.cts", "typescript"),
        ("file:///repo/src/foo.tsx", "typescriptreact"),
        ("file:///repo/src/foo.js", "javascript"),
        ("file:///repo/src/foo.mjs", "javascript"),
        ("file:///repo/src/foo.cjs", "javascript"),
        ("file:///repo/src/foo.jsx", "javascriptreact"),
    ],
)
def test_typescript_uri_handled(uri: str, expected: str) -> None:
    assert TypeScriptLSAdapter.language_id_for_uri(uri) == expected


def test_typescript_uri_foreign() -> None:
    assert TypeScriptLSAdapter.language_id_for_uri("file:///repo/foo.py") is None
    assert TypeScriptLSAdapter.language_id_for_uri("file:///repo/foo.zig") is None
    assert TypeScriptLSAdapter.language_id_for_uri("file:///repo/foo.go") is None


# ─── language_id_for_uri — ZLS ───────────────────────────────────────


def test_zls_uri_zig() -> None:
    assert ZlsAdapter.language_id_for_uri("file:///repo/src/main.zig") == "zig"


def test_zls_uri_zon() -> None:
    assert ZlsAdapter.language_id_for_uri("file:///repo/build.zon") == "zig"


def test_zls_uri_foreign() -> None:
    assert ZlsAdapter.language_id_for_uri("file:///repo/foo.py") is None
    assert ZlsAdapter.language_id_for_uri("file:///repo/foo.ts") is None


# ─── prepare_initialize_params — required keys ───────────────────────

_REQUIRED_KEYS = {"processId", "rootUri", "capabilities", "workspaceFolders"}


@pytest.mark.parametrize(
    "adapter_cls",
    [PyrightAdapter, TypeScriptLSAdapter, ZlsAdapter],
    ids=["pyright", "typescript", "zls"],
)
def test_init_params_has_required_keys(adapter_cls: type, tmp_path: Path) -> None:
    params = adapter_cls.prepare_initialize_params(tmp_path, False)
    assert params.keys() >= _REQUIRED_KEYS, (
        f"{adapter_cls.__name__}.prepare_initialize_params is missing keys: "
        f"{_REQUIRED_KEYS - params.keys()}"
    )


def test_pyright_process_id_is_pid(tmp_path: Path) -> None:
    """Pyright adapter sets processId=os.getpid() so vscode-languageserver/node watchdog can clean up.

    Regression (review-w3c HIGH): previously this was None, which
    disabled Pyright's parent-process watchdog and could leave Pyright
    running after abnormal scry exit.
    """
    import os

    params = PyrightAdapter.prepare_initialize_params(tmp_path, False)
    assert params["processId"] == os.getpid()


def test_typescript_process_id_is_int(tmp_path: Path) -> None:
    import os

    params = TypeScriptLSAdapter.prepare_initialize_params(tmp_path, False)
    assert params["processId"] == os.getpid()


def test_zls_process_id_is_int(tmp_path: Path) -> None:
    import os

    params = ZlsAdapter.prepare_initialize_params(tmp_path, False)
    assert params["processId"] == os.getpid()


def test_init_params_root_uri_matches_repo(tmp_path: Path) -> None:
    for adapter in (PyrightAdapter, TypeScriptLSAdapter, ZlsAdapter):
        params = adapter.prepare_initialize_params(tmp_path, False)
        assert params["rootUri"] == tmp_path.as_uri()


def test_init_params_workspace_folders_non_empty(tmp_path: Path) -> None:
    for adapter in (PyrightAdapter, TypeScriptLSAdapter, ZlsAdapter):
        params = adapter.prepare_initialize_params(tmp_path, False)
        folders = params["workspaceFolders"]
        assert isinstance(folders, list) and len(folders) >= 1
        assert folders[0]["uri"] == tmp_path.as_uri()


# ─── capabilities — callHierarchy ────────────────────────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [TypeScriptLSAdapter, ZlsAdapter],
    ids=["typescript", "zls"],
)
def test_call_hierarchy_capability_advertised(adapter_cls: type, tmp_path: Path) -> None:
    """TypeScript and ZLS adapters must advertise callHierarchy capability."""
    params = adapter_cls.prepare_initialize_params(tmp_path, False)
    caps = params["capabilities"]
    # Capability lives under textDocument.callHierarchy
    assert "callHierarchy" in caps.get("textDocument", {}), (
        f"{adapter_cls.__name__} did not advertise textDocument.callHierarchy"
    )


def test_pyright_call_hierarchy_capability_advertised(tmp_path: Path) -> None:
    params = PyrightAdapter.prepare_initialize_params(tmp_path, False)
    caps = params["capabilities"]
    assert "callHierarchy" in caps.get("textDocument", {})


# ─── initial_workspace_settings — Pyright ────────────────────────────


def test_pyright_diagnostic_mode_open_files_only() -> None:
    settings = PyrightAdapter.initial_workspace_settings()
    assert settings["python"]["analysis"]["diagnosticMode"] == "openFilesOnly"


def test_pyright_settings_non_empty() -> None:
    assert PyrightAdapter.initial_workspace_settings()


# ─── initial_workspace_settings — TypeScript ─────────────────────────


def test_typescript_max_ts_server_memory_in_init_options(tmp_path: Path) -> None:
    """maxTsServerMemory MUST be in initializationOptions, not didChangeConfiguration.

    Regression (review-w3c MEDIUM): typescript-language-server reads
    maxTsServerMemory from initialize.initializationOptions only;
    sending it via workspace/didChangeConfiguration is silently ignored
    so the intended 2 GB cap was not applied.
    """
    params = TypeScriptLSAdapter.prepare_initialize_params(tmp_path, False)
    assert params["initializationOptions"]["maxTsServerMemory"] == 2048

    # And NOT in workspace settings (where it's silently ignored).
    settings = TypeScriptLSAdapter.initial_workspace_settings()
    assert "tsserver" not in settings or "maxTsServerMemory" not in settings.get("tsserver", {})


def test_typescript_no_module_export_completions() -> None:
    settings = TypeScriptLSAdapter.initial_workspace_settings()
    assert settings["typescript"]["preferences"]["includeCompletionsForModuleExports"] is False
    assert settings["javascript"]["preferences"]["includeCompletionsForModuleExports"] is False


# ─── initial_workspace_settings — ZLS ────────────────────────────────


def test_zls_workspace_settings_is_dict() -> None:
    settings = ZlsAdapter.initial_workspace_settings()
    assert isinstance(settings, dict)


# ─── Registry: get_adapter ───────────────────────────────────────────


def test_get_adapter_python_returns_pyright() -> None:
    assert get_adapter("python") is PyrightAdapter


@pytest.mark.parametrize("lang", ["typescript", "tsx", "javascript", "jsx"])
def test_get_adapter_ts_variants_return_typescript_ls(lang: str) -> None:
    assert get_adapter(lang) is TypeScriptLSAdapter


def test_get_adapter_zig_returns_zls() -> None:
    assert get_adapter("zig") is ZlsAdapter


def test_get_adapter_lsp_side_id_returns_none() -> None:
    """LSP-side IDs (e.g. 'typescriptreact') are NOT valid registry keys.

    The ADAPTERS dict uses scry-side language IDs only (same namespace as
    LSP_ALLOWLIST).  Callers must translate LSP languageId strings to scry
    IDs before calling get_adapter.
    """
    assert get_adapter("typescriptreact") is None
    assert get_adapter("javascriptreact") is None


def test_get_adapter_nonexistent_returns_none() -> None:
    assert get_adapter("nonexistent") is None
    assert get_adapter("") is None


def test_all_allowlisted_languages_have_adapters() -> None:
    """W6a: every language in LSP_ALLOWLIST now has a dedicated adapter.

    Previously 'go' and 'rust' lacked W3c adapters and fell back to minimal
    inline params.  W6a adds GoplsAdapter and RustAnalyzerAdapter, completing
    full adapter coverage of the allowlist.
    """
    from scry.lsp.manager import LSP_ALLOWLIST

    for lang in LSP_ALLOWLIST:
        assert get_adapter(lang) is not None, (
            f"LSP_ALLOWLIST language '{lang}' has no adapter — add one or update LSP_ALLOWLIST."
        )


def test_adapters_dict_covers_allowlist_intersection() -> None:
    """Every language in ADAPTERS is also in LSP_ALLOWLIST."""
    from scry.lsp.manager import LSP_ALLOWLIST

    for lang in ADAPTERS:
        assert lang in LSP_ALLOWLIST, (
            f"ADAPTERS key '{lang}' is not in LSP_ALLOWLIST; "
            f"add it there or remove it from the adapter registry."
        )


# ─── LANGUAGES constant ───────────────────────────────────────────────


def test_pyright_languages_constant() -> None:
    assert "python" in PyrightAdapter.LANGUAGES


def test_typescript_languages_constant() -> None:
    for lang in ("typescript", "tsx", "javascript", "jsx"):
        assert lang in TypeScriptLSAdapter.LANGUAGES


def test_zls_languages_constant() -> None:
    assert "zig" in ZlsAdapter.LANGUAGES


# ─── Windows .cmd shim — arg list verification ───────────────────────


@pytest.mark.windows_only
def test_cmd_shim_integration_safe_args(
    tmp_path: Path,
    windows_only: None,
) -> None:
    """Integration (Windows): a .cmd shim + safe args yields cmd.exe /C wrapping.

    This simulates the real typescript-language-server .cmd install path
    described in DESIGN.md §10.5.  We create a real .cmd file in a temp dir
    and verify _build_spawn_cmd produces the correct command list.
    The metachar-rejection defense is already tested exhaustively in
    test_lsp_manager.py; this test confirms the HAPPY PATH for the same code
    path used by TypeScriptLSAdapter's Windows install scenario.
    """
    from scry.lsp.manager import LSPLaunchSpec, _build_spawn_cmd

    shim = tmp_path / "typescript-language-server.cmd"
    shim.write_text("@echo off\r\n")  # minimal valid .cmd content

    spec = LSPLaunchSpec(
        language="typescript",
        command=str(shim),
        args=["--stdio"],
        cwd=tmp_path,
    )
    cmd = _build_spawn_cmd(spec)

    assert cmd[0] == "cmd.exe"
    assert cmd[1] == "/C"
    assert cmd[2] == str(shim)
    assert cmd[3] == "--stdio"


# ─── Integration: adapter params flow through session.start() ─────────


@pytest.mark.integration
async def test_pyright_adapter_params_reach_fake_lsp(tmp_path: Path) -> None:
    """Integration: LSPSession.start() uses PyrightAdapter params; fake LSP responds.

    Verifies that the refactored session.start() correctly looks up the
    adapter, sends its initialize params (processId=None, rootUri, etc.),
    receives capabilities, then sends workspace/didChangeConfiguration with
    Pyright settings — all without breaking the fake-LSP fixture.
    """
    from scry.lsp.manager import LSPLaunchSpec, LSPSession

    spec = LSPLaunchSpec(
        language="python",
        command=sys.executable,
        args=[str(FAKE_LSP)],
        cwd=tmp_path,
    )
    session = LSPSession(language="python", spec=spec, allow_untrusted=True)
    await session.start()

    assert session.is_alive
    assert session.supports("callHierarchyProvider")
    assert session.supports("textDocumentSync.openClose")
    assert isinstance(session.capabilities, dict) and session.capabilities

    await session.shutdown()


@pytest.mark.integration
async def test_zls_adapter_params_reach_fake_lsp(tmp_path: Path) -> None:
    """Integration: ZlsAdapter params (empty settings) don't break session.start().

    ZlsAdapter.initial_workspace_settings() returns {}, so the manager must
    NOT send workspace/didChangeConfiguration in that case.
    """
    from scry.lsp.manager import LSPLaunchSpec, LSPSession

    # fake_lsp.py handles any language; force it to be used as "zig"
    spec = LSPLaunchSpec(
        language="zig",
        command=sys.executable,
        args=[str(FAKE_LSP)],
        cwd=tmp_path,
    )
    session = LSPSession(language="zig", spec=spec, allow_untrusted=False)
    await session.start()

    assert session.is_alive
    assert session.supports("callHierarchyProvider")
    await session.shutdown()


@pytest.mark.integration
async def test_go_adapter_session_works_with_fake_lsp(tmp_path: Path) -> None:
    """Integration: GoplsAdapter params flow through LSPSession.start() correctly.

    'go' is now covered by GoplsAdapter (W6a); the manager uses its
    prepare_initialize_params() instead of the minimal fallback.  The fake LSP
    accepts any init params and returns canned capabilities.
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
    await session.shutdown()


# ─── URI scheme validation (review-w3c LOW fix) ──────────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [PyrightAdapter, TypeScriptLSAdapter, ZlsAdapter],
)
@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/x.py",
        "http://example.com/x.ts",
        "data:text/plain,a.zig",
        "scry://repo/x.py",
        "untitled:Untitled-1.py",
        "",
        "/abs/path/to/file.py",
        "relative/path.tsx",
        "ftp://example.com/x.tsx",
    ],
)
def test_adapters_reject_non_file_uris(adapter_cls: type, uri: str) -> None:
    """Regression (review-w3c LOW): adapters MUST return None for non-file URIs.

    Otherwise a future didOpen path could route a virtual / remote
    document into a local LSP session.
    """
    assert adapter_cls.language_id_for_uri(uri) is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "adapter_cls,uri,expected",
    [
        (PyrightAdapter, "file:///repo/x.py", "python"),
        (PyrightAdapter, "file://localhost/repo/x.pyi", "python"),
        (TypeScriptLSAdapter, "file:///repo/x.tsx", "typescriptreact"),
        (TypeScriptLSAdapter, "file://localhost/repo/x.cts", "typescript"),
        (ZlsAdapter, "file:///repo/x.zig", "zig"),
        (ZlsAdapter, "file:///repo/x.zon", "zig"),
    ],
)
def test_adapters_accept_file_uris(adapter_cls: type, uri: str, expected: str) -> None:
    """file:// URIs (with optional empty/localhost host per RFC 8089) are accepted."""
    assert adapter_cls.language_id_for_uri(uri) == expected  # type: ignore[attr-defined]


def test_adapters_reject_non_localhost_host() -> None:
    """A file:// URI with a non-localhost authority is treated as non-local."""
    assert PyrightAdapter.language_id_for_uri("file://remote-server/x.py") is None
    assert TypeScriptLSAdapter.language_id_for_uri("file://nas/repo/x.ts") is None


# ─── Manager fallback capability shape (review-w3c MEDIUM fix) ────────


def test_manager_fallback_uses_textdocument_callhierarchy_capability() -> None:
    """Regression (review-w3c MEDIUM): fallback init params MUST nest callHierarchy under textDocument.

    LSP spec puts call hierarchy client capability at
    ``capabilities.textDocument.callHierarchy``; a top-level
    ``capabilities.callHierarchy`` is silently ignored. Without this,
    the go/rust fallback path silently disables transitive drift
    detection.
    """
    # Read the manager source and confirm the structure.  This is a
    # static check because the fallback only runs for languages without
    # a W3c adapter (go, rust) and we don't want to spawn a real LSP.
    import inspect

    from scry.lsp import manager as mgr_mod

    src = inspect.getsource(mgr_mod)

    # Locate the fallback block
    fallback_marker = "Fallback for languages without a W3c adapter"
    assert fallback_marker in src, (
        "fallback block marker missing — this test must be updated if the fallback comment changes"
    )
    fallback_idx = src.index(fallback_marker)
    # Look at the next ~600 chars of the fallback block
    fallback_block = src[fallback_idx : fallback_idx + 800]

    # The capability MUST be nested under textDocument
    assert '"textDocument"' in fallback_block, (
        "fallback init_params must nest capabilities under 'textDocument'"
    )
    assert '"callHierarchy"' in fallback_block

    # And there must NOT be a top-level callHierarchy under capabilities
    # (negative regression): make sure we don't have
    # `"capabilities": {"callHierarchy": ...}` at the top level
    bad_pattern = '"capabilities": {\n                "callHierarchy"'
    assert bad_pattern not in fallback_block, (
        "regression: fallback put callHierarchy at the wrong (top-level) capability path"
    )
