"""Regression tests for bugs surfaced by the swarm user-testing simulation.

Each test corresponds to a UTx-y bug-id catalogued during the swarm
simulation pass. These guard against the targeted fixes regressing
in future refactors.
"""

from __future__ import annotations

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
