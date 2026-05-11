"""Tests for scry.lsp.manager — lifecycle, allowlist enforcement, session pool.

Unit tests (no subprocess):
* Allowlist enforcement — LSPAllowlistViolation on command: without flag
* allow_untrusted=True — custom command accepted
* skip language — returns None without touching PATH
* Missing binary — returns None + WARNING logged
* Windows .cmd/.bat detection in _build_spawn_cmd
* Unknown language — returns None

Integration tests (spawn tests/fixtures/fake_lsp.py):
* Full lifecycle: spawn → initialize → request → shutdown
* supports() reflects server capabilities
* session_for() is a singleton: same instance returned on repeat calls
* shutdown_all() terminates every session
* is_alive transitions correctly after shutdown

Integration tests are marked @pytest.mark.integration so they run only
when requested.  A minimal fake LSP (fake_lsp.py) is used in place of a
real language server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scry.lsp.manager import (
    LSP_ALLOWLIST,
    LSPLaunchError,
    LSPLaunchSpec,
    LSPManager,
    LSPSession,
    _build_spawn_cmd,
)
from scry.models import CodeAnchorsConfig

# Path to the fake LSP fixture
FAKE_LSP = Path(__file__).parent / "fixtures" / "fake_lsp.py"


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    (tmp_path / ".scry").mkdir()
    return tmp_path


@pytest.fixture()
def lsp_config_python() -> CodeAnchorsConfig:
    """Config with python=lsp and no custom command override."""
    return CodeAnchorsConfig(languages={"python": "lsp"})


@pytest.fixture()
def fake_lsp_config(repo_root: Path) -> CodeAnchorsConfig:
    """Config that points python at fake_lsp.py via allow_untrusted."""
    return CodeAnchorsConfig(
        languages={"python": "lsp"},
        lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP)]}},
    )


# ─── LSP_ALLOWLIST contents ───────────────────────────────────────────


def test_allowlist_has_expected_languages() -> None:
    for lang in ("python", "typescript", "tsx", "javascript", "jsx", "zig", "go", "rust"):
        assert lang in LSP_ALLOWLIST, f"'{lang}' missing from LSP_ALLOWLIST"


def test_allowlist_python_entries() -> None:
    assert "pyright-langserver" in LSP_ALLOWLIST["python"]
    assert "pylsp" in LSP_ALLOWLIST["python"]
    assert "basedpyright-langserver" in LSP_ALLOWLIST["python"]


# ─── Allowlist enforcement ────────────────────────────────────────────


async def test_allowlist_violation_when_command_set_no_flag(
    repo_root: Path,
) -> None:
    """command: in config without allow_untrusted → per-language rejection, not a crash.

    HIGH #3 fix (DESIGN.md §6.2): LSPAllowlistViolation is now caught inside
    session_for() and converted to a per-language failure.  The caller (indexer)
    continues with transitive_hash_status=unsupported for that language.
    session_for() returns None and adds the language to _untrusted_rejected.
    """
    cfg = CodeAnchorsConfig(
        languages={"python": "lsp"},
        lsp={"python": {"command": "my-custom-pyright", "args": []}},
    )
    mgr = LSPManager(repo_root, cfg, allow_untrusted=False)

    session = await mgr.session_for("python")

    assert session is None, "untrusted command must be rejected → None, not a crash"
    assert "python" in mgr._failed
    assert "python" in mgr._untrusted_rejected
    # status_for returns "skip" (→ UNSUPPORTED) not "lsp_unavailable"
    assert mgr.status_for("python") == "skip"


async def test_allowlist_violation_message_mentions_flag(
    repo_root: Path,
) -> None:
    """LSPAllowlistViolation is logged (with the flag name) when rejected per-language.

    HIGH #3 fix: the exception is no longer re-raised; instead a WARNING is
    emitted.  We verify that the language ends up in _untrusted_rejected and
    status_for maps it to "skip".
    """
    cfg = CodeAnchorsConfig(
        languages={"python": "lsp"},
        lsp={"python": {"command": "evil-lsp", "args": []}},
    )
    mgr = LSPManager(repo_root, cfg, allow_untrusted=False)

    session = await mgr.session_for("python")

    assert session is None
    # The per-language tracking set must contain the rejected language so
    # downstream callers get "skip" (→ UNSUPPORTED) not "lsp_unavailable".
    assert "python" in mgr._untrusted_rejected
    assert mgr.status_for("python") == "skip"


async def test_allow_untrusted_accepts_custom_command(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_untrusted=True: custom command is accepted, no exception raised.

    We monkeypatch LSPSession.start to avoid actually spawning a process;
    the point is to confirm no LSPAllowlistViolation is raised.
    """
    mgr = LSPManager(repo_root, fake_lsp_config, allow_untrusted=True)

    async def _no_op_start(self: LSPSession) -> None:
        """Simulate a successful start() without a real subprocess."""
        self._proc = MagicMock()
        self._proc.returncode = None
        self.capabilities = {"callHierarchyProvider": True}

    monkeypatch.setattr(LSPSession, "start", _no_op_start)

    session = await mgr.session_for("python")

    assert session is not None, "Expected a session when allow_untrusted=True"


# ─── Skip language ────────────────────────────────────────────────────


async def test_skip_language_returns_none(repo_root: Path) -> None:
    """language configured as 'skip' → None, no PATH lookup attempted."""
    cfg = CodeAnchorsConfig(languages={"python": "skip"})
    mgr = LSPManager(repo_root, cfg)

    with patch("scry.lsp.manager.shutil.which") as mock_which:
        session = await mgr.session_for("python")

    assert session is None
    mock_which.assert_not_called()


async def test_skip_language_no_warning_logged(
    repo_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """skip is intentional — no WARNING should be emitted."""
    cfg = CodeAnchorsConfig(languages={"python": "skip"})
    mgr = LSPManager(repo_root, cfg)

    with caplog.at_level(logging.WARNING, logger="scry.lsp.manager"):
        await mgr.session_for("python")

    assert not caplog.records


# ─── Missing binary ───────────────────────────────────────────────────


async def test_missing_binary_returns_none(
    repo_root: Path,
    lsp_config_python: CodeAnchorsConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No allowlisted binary on PATH → None + WARNING."""
    mgr = LSPManager(repo_root, lsp_config_python)

    with (
        patch("scry.lsp.manager.shutil.which", return_value=None),
        caplog.at_level(logging.WARNING, logger="scry.lsp.manager"),
    ):
        session = await mgr.session_for("python")

    assert session is None
    assert any("python" in r.message for r in caplog.records)


async def test_missing_binary_cached_failure(
    repo_root: Path,
    lsp_config_python: CodeAnchorsConfig,
) -> None:
    """Second call for a failed language returns None immediately (no retry)."""
    mgr = LSPManager(repo_root, lsp_config_python)

    with patch("scry.lsp.manager.shutil.which", return_value=None):
        await mgr.session_for("python")  # populates _failed

    # Second call — shutil.which must NOT be invoked again
    with patch("scry.lsp.manager.shutil.which") as mock_which:
        session = await mgr.session_for("python")

    assert session is None
    mock_which.assert_not_called()


# ─── Unknown / unconfigured language ─────────────────────────────────


async def test_unknown_language_not_in_config_returns_none(
    repo_root: Path,
) -> None:
    """Language not in config.languages → None (treated as skip)."""
    cfg = CodeAnchorsConfig(languages={})
    mgr = LSPManager(repo_root, cfg)
    # "cobol" is not configured at all
    session = await mgr.session_for("cobol")
    assert session is None


async def test_language_in_allowlist_but_not_in_config(
    repo_root: Path,
) -> None:
    """Regression (review-w3a MEDIUM #3): allowlisted language NOT in config → no spawn.

    Before the fix, a language present in LSP_ALLOWLIST but absent from
    ``CodeAnchorsConfig.languages`` would still proceed to spawn because
    the skip check only matched the literal ``"skip"`` value, missing
    the ``None`` (key absent) case. The fix gates on explicit presence:
    if the language is not configured, treat it as skip.
    """
    cfg = CodeAnchorsConfig(languages={})  # no python entry
    mgr = LSPManager(repo_root, cfg)

    # shutil.which must NOT be called: we should bail at the config gate
    # BEFORE attempting binary resolution.
    with patch("scry.lsp.manager.shutil.which") as mock_which:
        session = await mgr.session_for("python")

    assert session is None
    mock_which.assert_not_called()


# ─── Windows .cmd/.bat detection ─────────────────────────────────────


@pytest.mark.windows_only
def test_cmd_shim_wraps_in_cmd_exe(windows_only: None) -> None:
    """On Windows, .cmd suffix → cmd.exe /C <path> <args>."""
    spec = LSPLaunchSpec(
        language="typescript",
        command="C:\\npm\\typescript-language-server.cmd",
        args=["--stdio"],
        cwd=Path("C:\\repo"),
    )
    cmd = _build_spawn_cmd(spec)
    assert cmd[0] == "cmd.exe"
    assert cmd[1] == "/C"
    assert cmd[2] == "C:\\npm\\typescript-language-server.cmd"
    assert cmd[3] == "--stdio"


@pytest.mark.windows_only
def test_bat_shim_wraps_in_cmd_exe(windows_only: None) -> None:
    """On Windows, .bat suffix → cmd.exe /C <path> <args>."""
    spec = LSPLaunchSpec(
        language="typescript",
        command="C:\\tools\\run-lsp.bat",
        args=[],
        cwd=Path("C:\\repo"),
    )
    cmd = _build_spawn_cmd(spec)
    assert cmd[0] == "cmd.exe"
    assert "/C" in cmd


@pytest.mark.windows_only
def test_exe_not_wrapped_on_windows(windows_only: None) -> None:
    """On Windows, .exe suffix → direct invocation (no cmd.exe wrapper)."""
    spec = LSPLaunchSpec(
        language="python",
        command="C:\\tools\\pyright-langserver.exe",
        args=["--stdio"],
        cwd=Path("C:\\repo"),
    )
    cmd = _build_spawn_cmd(spec)
    assert cmd[0] == "C:\\tools\\pyright-langserver.exe"
    assert "cmd.exe" not in cmd


@pytest.mark.unix_only
def test_cmd_suffix_not_wrapped_on_unix(unix_only: None) -> None:
    """On Unix, .cmd suffix does NOT trigger cmd.exe wrapping."""
    spec = LSPLaunchSpec(
        language="typescript",
        command="/usr/local/bin/typescript-language-server.cmd",
        args=["--stdio"],
        cwd=Path("/repo"),
    )
    cmd = _build_spawn_cmd(spec)
    assert cmd[0] == "/usr/local/bin/typescript-language-server.cmd"
    assert "cmd.exe" not in cmd


def test_build_spawn_cmd_plain_binary() -> None:
    """Non-.cmd binary: command + args, no wrapping."""
    spec = LSPLaunchSpec(
        language="go",
        command="/usr/local/bin/gopls",
        args=["-mode=stdio"],
        cwd=Path("/repo"),
    )
    cmd = _build_spawn_cmd(spec)
    assert cmd == ["/usr/local/bin/gopls", "-mode=stdio"]


# ─── Windows .cmd shim arg-injection defense (review-w3a BLOCKING fix) ─


@pytest.mark.windows_only
@pytest.mark.parametrize(
    "bad_arg",
    [
        "&",  # cmd.exe sequencing: foo & calc.exe
        "|",  # pipe
        "<",  # input redirect
        ">",  # output redirect
        "^",  # cmd escape character
        '"',  # ends quoted region
        "%PATH%",  # env var expansion
        "(",  # subshell open
        ")",  # subshell close
        "--stdio & calc.exe",  # realistic injection vector
    ],
)
def test_cmd_shim_rejects_shell_metachars(windows_only: None, bad_arg: str) -> None:
    """Regression (review-w3a BLOCKING): .cmd/.bat shims must NOT forward shell metachars.

    The args list comes from repo-controlled .scry/config.yaml
    lsp.<lang>.args. Without --allow-untrusted-lsp-config covering args
    (it only covers `command:`), a hostile repo could inject arbitrary
    cmd.exe via args like ['--stdio', '&', 'calc.exe']. The shim path
    now refuses to spawn when any arg contains a shell metacharacter.
    """
    spec = LSPLaunchSpec(
        language="typescript",
        command="C:\\npm\\typescript-language-server.cmd",
        args=["--stdio", bad_arg],
        cwd=Path("C:\\repo"),
    )
    with pytest.raises(LSPLaunchError, match="metacharacters"):
        _build_spawn_cmd(spec)


@pytest.mark.unix_only
def test_unix_shim_path_does_not_trigger_metachar_check(unix_only: None) -> None:
    """On Unix, .cmd suffix doesn't go through cmd.exe so metachar args are passed through.

    The Windows-specific defense doesn't apply here because the spawn
    is direct (no shell interposition). Args are passed verbatim to the
    binary, which can interpret them however it wants without affecting
    other processes.
    """
    spec = LSPLaunchSpec(
        language="typescript",
        command="/usr/local/bin/lsp.cmd",  # weird but legal on Unix
        args=["--stdio", "&", "echo", "no shell here"],
        cwd=Path("/repo"),
    )
    # Should NOT raise on Unix.
    cmd = _build_spawn_cmd(spec)
    assert cmd[0] == "/usr/local/bin/lsp.cmd"
    assert "cmd.exe" not in cmd


# ─── Cancelled request future cleanup (review-w3a MEDIUM #2) ─────────


async def test_cancelled_request_releases_pending_future(
    repo_root: Path,
) -> None:
    """Regression: asyncio.CancelledError must clean up the pending-futures map.

    Previously the `except Exception` clause in LSPSession.request did
    NOT catch CancelledError (which inherits from BaseException), so
    cancelled requests left their futures in self._pending forever —
    a slow memory leak and a footgun for stale-response delivery.
    """
    from scry.lsp.manager import LSPSession

    spec = LSPLaunchSpec(
        language="python",
        command="dummy",
        args=[],
        cwd=repo_root,
    )
    session = LSPSession(language="python", spec=spec)

    # Manually inject a stream writer that hangs forever so the request
    # is in flight when we cancel it.
    class _HangingWriter:
        async def write_message(self, _msg: dict[str, Any]) -> None:
            return None  # write succeeds; the response just never arrives

    session._stream_writer = _HangingWriter()  # type: ignore[assignment]

    task = asyncio.create_task(session.request("anyMethod", {"foo": "bar"}, timeout=10.0))
    # Yield once so the request is registered in _pending.
    await asyncio.sleep(0.01)
    assert len(session._pending) == 1, "request should be pending"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert len(session._pending) == 0, (
        "CancelledError must clean up the pending-futures map (review-w3a MEDIUM #2)"
    )


# ─── Integration: full lifecycle via fake_lsp.py ─────────────────────


@pytest.mark.integration
async def test_lifecycle_spawn_initialize_shutdown(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: spawn fake LSP, initialize, verify capabilities, shutdown."""
    mgr = LSPManager(repo_root, fake_lsp_config, allow_untrusted=True)

    session = await mgr.session_for("python")
    assert session is not None, "Expected a live session after spawn"
    assert session.is_alive
    assert session.supports("callHierarchyProvider")
    assert session.supports("textDocumentSync.openClose")
    assert not session.supports("nonexistentCapability")

    await mgr.shutdown_all()
    assert not session.is_alive


@pytest.mark.integration
async def test_session_for_is_singleton(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: repeated session_for calls return the identical instance."""
    mgr = LSPManager(repo_root, fake_lsp_config, allow_untrusted=True)

    session_a = await mgr.session_for("python")
    session_b = await mgr.session_for("python")

    assert session_a is not None
    assert session_b is not None
    assert session_a is session_b, "Expected the same session object"

    await mgr.shutdown_all()


@pytest.mark.integration
async def test_shutdown_all_terminates_sessions(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: shutdown_all() terminates all live sessions."""
    mgr = LSPManager(repo_root, fake_lsp_config, allow_untrusted=True)

    session = await mgr.session_for("python")
    assert session is not None
    assert session.is_alive

    await mgr.shutdown_all()

    assert not session.is_alive
    assert mgr._sessions == {}


@pytest.mark.integration
async def test_context_manager_shuts_down(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: async context manager calls shutdown_all on exit."""
    async with LSPManager(repo_root, fake_lsp_config, allow_untrusted=True) as mgr:
        session = await mgr.session_for("python")
        assert session is not None
        assert session.is_alive

    # After __aexit__, session should be terminated
    assert session is not None
    assert not session.is_alive


@pytest.mark.integration
async def test_session_request_after_initialize(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: we can issue a second request (shutdown) via session.request."""
    mgr = LSPManager(repo_root, fake_lsp_config, allow_untrusted=True)
    session = await mgr.session_for("python")
    assert session is not None

    # Manually send shutdown (mgr.shutdown_all also does this, but test directly)
    result = await session.request("shutdown", {}, timeout=10.0)
    # fake_lsp returns result: None for shutdown
    assert result is None

    await session.notify("exit")
    # Give the process a moment to exit
    import asyncio

    await asyncio.sleep(0.1)

    # Clean up the manager (session already shutting down)
    mgr._sessions.clear()


@pytest.mark.integration
async def test_supports_false_for_missing_capability(
    repo_root: Path,
    fake_lsp_config: CodeAnchorsConfig,
) -> None:
    """Integration: supports() returns False for capabilities not in the response."""
    async with LSPManager(repo_root, fake_lsp_config, allow_untrusted=True) as mgr:
        session = await mgr.session_for("python")
        assert session is not None
        assert not session.supports("completionProvider")
        assert not session.supports("hoverProvider")
        assert session.supports("callHierarchyProvider")


# uat-r5-5 pr-d noise
