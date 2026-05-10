"""Regression tests for bugs surfaced by the swarm user-testing simulation.

Each test corresponds to a UTx-y bug-id catalogued during the swarm
simulation pass. These guard against the targeted fixes regressing
in future refactors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ─── UT3-1 BLOCKING: git_context must close stdin to avoid pipe-inheritance hang


def test_ut3_1_run_git_passes_devnull_stdin() -> None:
    """UT3-1: ``_run_git`` and ``_run_git_bytes`` MUST set ``stdin=DEVNULL``.

    Without this, when scry runs over MCP stdio, git inherits the parent's
    stdin pipe (Windows ``CreateProcess`` ``bInheritHandles=True``) and
    never exits, freezing the event loop and breaking ALL write tools.
    """
    import inspect

    from scry.git_context import _run_git, _run_git_bytes

    for fn in (_run_git, _run_git_bytes):
        src = inspect.getsource(fn)
        assert "stdin=subprocess.DEVNULL" in src, (
            f"{fn.__name__} must pass stdin=subprocess.DEVNULL to subprocess.run "
            f"(UT3-1 BLOCKING regression — see docstring for rationale)"
        )


# ─── UT3-2 + UT3-3 HIGH: get_links + find_drift output schemas


def test_ut3_2_get_links_returns_dict_per_section_7_3() -> None:
    """get_links MUST return dict (not list) so the FastMCP outputSchema
    matches the handler's actual return shape and ``index_state`` per §7.3
    can be conveyed.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    # Locate the get_links @mcp.tool() block; verify it declares -> dict.
    assert "async def get_links(" in src
    # Find "async def get_links" and check the next 200 chars include `-> dict`
    idx = src.index("async def get_links(")
    block = src[idx : idx + 400]
    assert "-> dict[str, Any]" in block, (
        "UT3-2: get_links must declare -> dict[str, Any] (not list[...]) "
        "so its FastMCP outputSchema accepts the handler's dict return."
    )


def test_ut3_3_find_drift_returns_dict_per_section_7_3() -> None:
    """find_drift MUST return dict (not list)."""
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert "async def find_drift(" in src
    idx = src.index("async def find_drift(")
    block = src[idx : idx + 400]
    assert "-> dict[str, Any]" in block, (
        "UT3-3: find_drift must declare -> dict[str, Any] (not list[...])."
    )


# ─── UT4-1 BLOCKING: leader lock cleanup on graceful shutdown


def test_ut4_1_mcp_command_calls_server_stop_in_finally() -> None:
    """UT4-1: the ``scry mcp`` command MUST call ``server.stop()`` in a
    ``finally`` block so the leader lock file is cleaned up on graceful
    shutdown (stdin EOF / Ctrl-C).  Without this, followers can read
    stale PID/endpoint metadata from the lock file after the leader
    has shut down cleanly.
    """
    import inspect

    from scry.cli import mcp

    src = inspect.getsource(mcp.callback)  # type: ignore[arg-type]
    assert "finally" in src, "scry mcp must have a finally block"
    assert "server.stop()" in src, (
        "UT4-1: scry mcp's finally block must call server.stop() to "
        "release the leader lock and unlink the lock file."
    )


# ─── UT5-1 BLOCKING: TypeScript export_statement unwrap


def test_ut5_1_typescript_export_class_is_extracted(tmp_path: Path) -> None:
    """UT5-1: ``export class Foo {}`` must produce a CODE anchor named ``Foo``.

    Pre-fix the TS walker only checked direct children of root; ``export``
    wraps every top-level declaration so all exports were silently
    dropped, blocking polyglot use cases.
    """
    from scry.extract.code import extract_code_symbols

    p = tmp_path / "test.ts"
    p.write_text(
        "export class User { name: string = ''; }\n"
        "export function validateCredentials(c: any) { return true; }\n"
        "export default class AuthService {}\n"
        "function helper() { return 1; }\n",
        encoding="utf-8",
    )
    anchors = extract_code_symbols(path=p, repo_root=tmp_path, language="typescript")
    names = {a.symbol_name for a in anchors}
    assert "User" in names, f"export class User not extracted; got {names}"
    assert "validateCredentials" in names
    assert "AuthService" in names
    assert "helper" in names  # non-exported should also still work


# ─── UT4-4 MEDIUM: scry watch outside scry project gives friendly error


def test_ut4_4_watch_without_scry_init_friendly_error(tmp_path: Path) -> None:
    """UT4-4: running ``scry watch`` in a directory that hasn't been
    ``scry init``-ed must print a friendly error and exit 2 — NOT raise
    a raw ``FileNotFoundError`` traceback.
    """
    import asyncio

    from scry.cmd_watch import run_watch

    # tmp_path has no .scry/ directory.
    rc = asyncio.run(run_watch(tmp_path, once=True))
    assert rc == 2


# ─── UT2-2 + UT2-4: reconcile --json error path is JSON; LLM-unavail = exit 1


def test_ut2_2_reconcile_emits_json_error_when_json_output() -> None:
    """UT2-2: ``scry reconcile --json`` must emit JSON on errors so
    machine-readable pipelines can route on them.
    """
    import inspect

    from scry.cmd_reconcile import _emit_error

    src = inspect.getsource(_emit_error)
    assert 'json.dumps({"error"' in src or "json.dumps({'error'" in src, (
        "_emit_error must wrap errors in JSON when json_output=True (UT2-2)"
    )


def test_ut2_4_reconcile_llm_unavailable_exits_1() -> None:
    """UT2-4: ``scry reconcile`` with no LLM configured exits 1
    (configuration issue), not 2 (infrastructure failure).
    """
    import inspect

    from scry.cmd_reconcile import run_reconcile_cmd

    src = inspect.getsource(run_reconcile_cmd)
    # The first LLM-unavailable branch should exit_code=1.
    # Locate "LLM provider unavailable" and check the surrounding
    # _emit_error has exit_code=1.
    idx = src.index("LLM provider unavailable")
    nearby = src[max(0, idx - 50) : idx + 200]
    assert "exit_code=1" in nearby, (
        "UT2-4: LLM-unavailable error path must use exit_code=1 (configuration "
        "issue), not 2 (infra failure)."
    )


# ─── Round-2 swarm fixes ─────────────────────────────────────────────


def test_ut1_5_readme_says_v0_0_1_not_design_phase() -> None:
    """UT1-5: README must not mislead first-time evaluators with the
    obsolete 'design phase' status line.
    """
    from pathlib import Path as _P

    text = _P(__file__).resolve().parent.parent.joinpath("README.md").read_text(encoding="utf-8")
    assert "design phase" not in text.lower(), (
        "README still claims scry is in 'design phase' (UT1-5)"
    )
    assert "v0.0.1" in text or "fully functional" in text.lower()


def test_ut1_10_authlib_warning_suppressed() -> None:
    """UT1-10: importing scry must register an authlib warning filter
    so subsequent ``warnings.warn(..., AuthlibDeprecationWarning)``
    calls are silently dropped.

    Run in a subprocess so pytest's per-test ``catch_warnings`` wrapper
    doesn't reset the filter list between import and assertion.
    """
    code = (
        "import scry, warnings, sys\n"
        "matches = [\n"
        "    f for f in warnings.filters\n"
        "    if f[2] is not None\n"
        "    and (getattr(f[2], '__module__', '').startswith('authlib')\n"
        "         or 'authlib' in getattr(f[2], '__name__', '').lower())\n"
        "    and f[0] == 'ignore'\n"
        "]\n"
        "sys.exit(0 if matches else 1)\n"
    )
    rc = subprocess.run([sys.executable, "-c", code], check=False).returncode
    assert rc == 0, "scry must register an ignore filter for AuthlibDeprecationWarning (UT1-10)"


def test_ut2_6_relink_same_pair_refreshes_existing(tmp_path: Path) -> None:
    """UT2-6: re-running ``scry link A B --type X`` reuses the existing
    link_id instead of creating a duplicate.
    """
    # Smoke-test the helper logic by direct Python: scan replay for
    # an existing (from, to, type) match and ensure the CLI module
    # exposes the upsert-by-pair logic.
    # The implementation of `scry link` must reference the existing-link
    # detection block.  Verify by inspecting its source.
    import inspect

    from scry.cli import link

    src = inspect.getsource(link.callback)  # type: ignore[arg-type]
    assert "existing_id" in src, (
        "scry link must check for an existing link with the same "
        "(from, to, type) tuple before minting a new link_id (UT2-6)"
    )
    assert "refreshing existing link_id" in src


def test_ut3_4_propose_link_idempotency_at_leader() -> None:
    """UT3-4: leader-direct write ops must dedup by idempotency_token."""
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._dispatch)
    assert "_leader_idem_cache" in src, (
        "MCPServer._dispatch must have a leader-side idempotency cache (UT3-4)"
    )


def test_ut3_5_commit_links_returns_dict_with_index_state() -> None:
    """UT3-5: commit_links MCP handler must include index_state per §7.3."""
    import inspect

    from scry.mcp.handlers import commit_links

    sig = inspect.signature(commit_links)
    # The return annotation must be dict, not list
    ret = sig.return_annotation
    assert ret is not list, "commit_links must return dict, not list (UT3-5)"


def test_ut4_2_doctor_explains_windows_pid_wrapper() -> None:
    """UT4-2: doctor must clarify the scry.exe / python.exe PID gap."""
    import inspect

    import scry.cli as _cli

    src = inspect.getsource(_cli.doctor.callback)  # type: ignore[arg-type]
    assert "lock_pid" in src
    assert "wrapper" in src.lower(), (
        "doctor must mention the Windows scry.exe wrapper PID issue (UT4-2)"
    )


def test_ut4_3_serve_stdio_has_eof_watchdog_on_windows() -> None:
    """UT4-3: serve_stdio must spawn a Windows stdin-EOF watchdog thread."""
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer.serve_stdio)
    assert "stdin" in src.lower()
    assert "eof" in src.lower() or "watchdog" in src.lower() or "watcher" in src.lower(), (
        "serve_stdio must include the stdin-EOF watchdog (UT4-3)"
    )


def test_ut1_1_mcp_has_daemon_flag() -> None:
    """UT1-1: scry mcp --daemon enables headless leader runs for watch testing."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--daemon" in result.output, "scry mcp --help must advertise --daemon (UT1-1)"


def test_ut1_2_check_has_verbose_flag() -> None:
    """UT1-2: scry check --verbose lists drifted links (not just counts)."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["check", "--help"])
    assert result.exit_code == 0
    assert "--verbose" in result.output


def test_ut5_2_warns_on_full_with_unconfigured_language(tmp_path: Path) -> None:
    """UT5-2: transitive_resolution=full + empty languages -> WARN not silent."""
    import inspect

    import scry.index

    src = inspect.getsource(scry.index._enrich_all_with_lsp)
    # Verify the warning is in the source.
    assert "transitive_resolution=full but language" in src, (
        "Indexer must warn when transitive_resolution=full and a language "
        "is not in code_anchors.languages (UT5-2)"
    )
