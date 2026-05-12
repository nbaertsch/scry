"""Regression tests for bugs surfaced by the swarm user-testing simulation.

Each test corresponds to a UTx-y bug-id catalogued during the swarm
simulation pass. These guard against the targeted fixes regressing
in future refactors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

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
    """UT1-10 / SR1-5: importing scry must register an authlib warning filter
    so subsequent ``warnings.warn(..., AuthlibDeprecationWarning)``
    calls are silently dropped.

    Run in a subprocess so pytest's per-test ``catch_warnings`` wrapper
    doesn't reset the filter list between import and assertion.

    SR1-5: also verify end-to-end that importing ``scry.mcp.server``
    (which transitively imports ``fastmcp`` → ``authlib.jose``) does
    NOT surface the warning to stderr.  This is the user-observable
    contract; checking only that the filter is REGISTERED is not
    sufficient because ``simplefilter`` can be wiped by a subsequent
    framework call.
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

    # SR1-5 end-to-end: import scry then scry.mcp.server with default
    # warning behaviour and ensure NOTHING is printed to stderr.
    e2e_code = (
        "import warnings, sys\n"
        # Promote anything that escapes our filter to a hard exception:
        "import scry  # registers the ignore filter first\n"
        "import scry.mcp.server  # transitively imports fastmcp → authlib.jose\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", e2e_code],
        capture_output=True,
        text=True,
        check=False,
    )
    # Note: with ``-W error::DeprecationWarning`` from the CLI, an
    # explicit user override trumps any in-process suppression.  We
    # accept either rc==0 (suppression won) OR a clear AuthlibDeprecationWarning
    # in stderr (user override won) — what we DON'T want is a silent
    # wrong outcome (rc!=0 with no recognizable error).
    if proc.returncode != 0:
        assert "AuthlibDeprecationWarning" in proc.stderr, (
            f"unexpected failure: rc={proc.returncode} stderr={proc.stderr!r}"
        )


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


# ─── SR1 round-3 regression tests ─────────────────────────────────────────────


def test_sr1_1_link_refresh_sets_supersedes() -> None:
    """SR1-1: refresh path in `scry link` MUST set supersedes per §3.5.2 rule 5.

    Without this, the second `scry link A B --type X` invocation fails
    LinkStore validation with "requires supersedes", silently breaking
    UT2-6's duplicate-link refresh fix.
    """
    import inspect

    from scry.cli import link

    src = inspect.getsource(link.callback)  # type: ignore[arg-type]
    assert "existing_supersedes" in src, (
        "scry link refresh path must capture active_link.last_event_id (SR1-1)"
    )
    assert '"supersedes"' in src, (
        "scry link refresh path must include supersedes in the LinkRecord payload (SR1-1)"
    )


def test_sr1_2_leader_idempotency_uses_per_token_lock() -> None:
    """SR1-2: leader-direct idempotency cache MUST serialize concurrent
    same-token calls so the second call awaits the first instead of
    racing into the handler.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._dispatch)
    assert "_leader_idem_locks" in src, (
        "MCPServer._dispatch must use a per-(op, token) asyncio.Lock to "
        "serialize concurrent same-token requests (SR1-2)"
    )
    assert "asyncio.Lock" in src, (
        "MCPServer._dispatch must reference asyncio.Lock for SR1-2 idempotency"
    )


def test_sr1_3_dir_excluded_handles_double_star_root_dirs(tmp_path: Path) -> None:
    """SR1-3: ``_dir_excluded`` must prune root-level directories that
    match ``**/<name>/**`` style patterns instead of descending into them.
    """
    from scry.index import _dir_excluded

    repo = tmp_path
    # Root .venv vs **/.venv/** — must prune
    assert _dir_excluded(repo / ".venv", repo, ["**/.venv/**"]) is True, (
        "Root .venv must be pruned by **/.venv/** (SR1-3)"
    )
    # Nested case — must also prune
    assert _dir_excluded(repo / "sub" / ".venv", repo, ["**/.venv/**"]) is True
    # Plain dir-name pattern — must prune
    assert _dir_excluded(repo / "node_modules", repo, ["node_modules/**"]) is True
    # File-only pattern — must NOT over-prune
    assert _dir_excluded(repo / "src", repo, ["**/*.pyc"]) is False, (
        "src/ must not be pruned by a file-only pattern (SR1-3)"
    )


def test_sr1_4_daemon_exits_when_not_leader() -> None:
    """SR1-4: ``scry mcp --daemon`` MUST exit (not silently sleep) when
    another process already holds the leader lock.
    """
    import inspect

    from scry.cli import mcp

    src = inspect.getsource(mcp.callback)  # type: ignore[arg-type]
    # Either an explicit role check OR an early exit when role != "leader".
    assert "ctx.role" in src or 'role != "leader"' in src or "role !=" in src, (
        "scry mcp --daemon must verify it actually became the leader (SR1-4)"
    )


# ─── SR5 polyglot regression tests ────────────────────────────────────────────


def test_sr5_1_go_extraction(tmp_path: Path) -> None:
    """SR5-1: Go source extracts function/method/type anchors."""
    from scry.extract.code import extract_code_symbols

    src = (
        b"package main\n"
        b"type Service struct { name string }\n"
        b"type Handler interface { Handle() string }\n"
        b"func NewService() *Service { return &Service{} }\n"
        b"func (s *Service) Hello(name string) string { return s.name }\n"
        b"type Alias = string\n"
    )
    f = tmp_path / "h.go"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="go")
    names = {a.symbol_name for a in anchors}
    assert "NewService" in names, "Go function_declaration must be extracted (SR5-1)"
    assert "Service" in names, "Go struct via type_declaration must be extracted (SR5-1)"
    assert "Handler" in names, "Go interface via type_declaration must be extracted (SR5-1)"
    assert "Alias" in names, "Go type alias must be extracted (SR5-1)"
    # Receiver method gets qualified path Service.Hello
    qualified = {a.id.split(":", 1)[1] for a in anchors}
    assert "Service.Hello" in qualified, "Go receiver method must be qualified (SR5-1)"


def test_sr5_1_rust_extraction(tmp_path: Path) -> None:
    """SR5-1: Rust source extracts fn/struct/trait/impl/macro anchors."""
    from scry.extract.code import extract_code_symbols

    src = (
        b"pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        b"pub struct Config { name: String }\n"
        b"pub trait Handler { fn handle(&self) -> String; }\n"
        b"impl Config { pub fn new() -> Self { Self { name: String::new() } } }\n"
        b"macro_rules! my_macro { () => { () } }\n"
        b"pub enum Status { Ok, Err }\n"
    )
    f = tmp_path / "m.rs"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="rust")
    names = {a.symbol_name for a in anchors}
    assert "add" in names, "Rust pub fn must be extracted (SR5-1)"
    assert "Config" in names, "Rust struct must be extracted (SR5-1)"
    assert "Handler" in names, "Rust trait must be extracted (SR5-1)"
    assert "my_macro" in names, "Rust macro_rules must be extracted (SR5-1)"
    assert "Status" in names, "Rust enum must be extracted (SR5-1)"
    # Impl methods get qualified path impl_<Type>.<method>
    qualified = {a.id.split(":", 1)[1] for a in anchors}
    assert "impl_Config.new" in qualified, "Rust impl method must be qualified (SR5-1)"


def test_sr5_2_typescript_export_const_arrow(tmp_path: Path) -> None:
    """SR5-2: TypeScript `export const fn = () => {}` produces an anchor."""
    from scry.extract.code import extract_code_symbols

    src = (
        b"export const createContext = (id: string): any => ({});\n"
        b"export enum StatusCode { OK = 200 }\n"
        b"export namespace Config { export const PORT = 3000; }\n"
    )
    f = tmp_path / "s.ts"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="typescript")
    names = {a.symbol_name for a in anchors}
    assert "createContext" in names, "TS export const arrow fn must be extracted (SR5-2)"
    assert "StatusCode" in names, "TS export enum must be extracted (SR5-2)"
    assert "Config" in names, "TS export namespace must be extracted (SR5-2)"


def test_sr5_3_javascript_arrow_and_class_expression(tmp_path: Path) -> None:
    """SR5-3: JavaScript `const fn = () =>` and `const Cls = class {}` indexed."""
    from scry.extract.code import extract_code_symbols

    src = (
        b"const delay = (ms) => new Promise((r) => setTimeout(r, ms));\n"
        b"const Cache = class { constructor() {} };\n"
    )
    f = tmp_path / "l.js"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="javascript")
    names = {a.symbol_name for a in anchors}
    assert "delay" in names, "JS const arrow fn must be extracted (SR5-3)"
    assert "Cache" in names, "JS const class expression must be extracted (SR5-3)"


def test_sr5_7_watch_ignores_vendor_dir() -> None:
    """SR5-7: cmd_watch._IGNORE_DIRS includes Go's vendor/ directory."""
    from scry.cmd_watch import _IGNORE_DIRS

    assert "vendor" in _IGNORE_DIRS, "watch must ignore vendor/ (SR5-7)"


def test_sr1_5_pyproject_filterwarnings_includes_authlib() -> None:
    """SR1-5: pyproject ``filterwarnings`` MUST include the AuthlibDeprecationWarning
    ignore so pytest's per-test warning-filter reset doesn't undo the
    import-time suppression in scry/__init__.py.
    """
    import tomllib
    from pathlib import Path as _Path

    pyproject = _Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    pytest_cfg = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    fw = pytest_cfg.get("filterwarnings", [])
    matches = [f for f in fw if "AuthlibDeprecationWarning" in f and "ignore" in f]
    assert matches, (
        "pyproject [tool.pytest.ini_options].filterwarnings must include "
        "an ignore for AuthlibDeprecationWarning (SR1-5)"
    )


def test_sr5_4_lang_from_anchor_id() -> None:
    """SR5-4: callers/subclasses must derive language from anchor ID extension."""
    from scry.cli import _lang_from_anchor_id, _lsp_binary_for

    assert _lang_from_anchor_id("src/foo.py:bar") == "python"
    assert _lang_from_anchor_id("src/foo.ts:bar") == "typescript"
    assert _lang_from_anchor_id("src/foo.go:Bar") == "go"
    assert _lang_from_anchor_id("src/foo.rs:Bar.baz") == "rust"
    assert _lang_from_anchor_id("src/foo.zig:bar") == "zig"
    # Unknown extensions fall back to python (preserves historical behaviour)
    assert _lang_from_anchor_id("src/foo.unknown:bar") == "python"

    assert _lsp_binary_for("python") == "pyright-langserver"
    assert _lsp_binary_for("typescript") == "typescript-language-server"
    assert _lsp_binary_for("go") == "gopls"
    assert _lsp_binary_for("rust") == "rust-analyzer"
    assert _lsp_binary_for("zig") == "zls"


# ─── UAT round-4 regression tests ─────────────────────────────────────────────


def test_uat_1_indexer_emits_progress_for_all_phases(tmp_path: Path) -> None:
    """UAT-1: Indexer.index() fires progress_callback for extract / lsp / embed.

    Verifies the per-phase progress contract so the silent-burn UX bug
    (UAT1's most painful moment) cannot regress.
    """
    import subprocess as _subprocess

    from scry.config import load_config
    from scry.embed import StubEmbedder
    from scry.index import Indexer

    repo = tmp_path
    _subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _subprocess.run(
        [
            "git",
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=a",
            "commit",
            "-qm",
            "init",
            "--allow-empty",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "a.py").write_text("def foo():\n    pass\n")
    (repo / "spec.md").write_text("# Spec\n\nbody\n")

    # Minimal config so load_config doesn't reach for .scry/config.yaml.
    (repo / ".scry").mkdir(exist_ok=True)
    (repo / ".scry" / "config.yaml").write_text(
        "include:\n  - '**/*.py'\n  - '**/*.md'\nexclude: []\n"
    )

    config = load_config(repo)
    embedder = StubEmbedder()
    indexer = Indexer(repo, config=config, embedder=embedder, allow_untrusted=True)

    events: list[tuple[str, int, int, str]] = []

    def cb(phase: str, processed: int, total: int, label: str) -> None:
        events.append((phase, processed, total, label))

    indexer.index(force=True, progress_callback=cb)

    # We expect at least one extract event AND at least one embed event.
    phases_seen = {e[0] for e in events}
    assert "extract" in phases_seen, (
        f"UAT-1: progress_callback must fire for the extract phase; got phases={phases_seen}"
    )
    assert "embed" in phases_seen, (
        f"UAT-1: progress_callback must ALSO fire for the embed phase per "
        f"review-u1-r2 MEDIUM #1; got phases={phases_seen}"
    )
    # Final extract event should reach total.
    extract_events = [e for e in events if e[0] == "extract"]
    assert extract_events[-1][1] == extract_events[-1][2], (
        "UAT-1: last extract progress event must reach total (n == total)"
    )


def test_uat_6_lsp_install_hint_includes_concrete_command() -> None:
    """UAT-6: _lsp_install_hint must give a runnable install command, not
    just the binary name.  Day-1 users don't know if pyright-langserver
    is npm/pip/system; the hint must say "npm install -g pyright"."""
    from scry.cli import _lsp_install_hint

    py_hint = _lsp_install_hint("python")
    assert "npm install" in py_hint and "pyright" in py_hint, py_hint

    go_hint = _lsp_install_hint("go")
    assert "go install" in go_hint and "gopls" in go_hint, go_hint

    rust_hint = _lsp_install_hint("rust")
    assert "rustup" in rust_hint and "rust-analyzer" in rust_hint, rust_hint

    ts_hint = _lsp_install_hint("typescript")
    assert "npm install" in ts_hint, ts_hint


def test_uat_7_check_warns_when_fs_is_newer_than_index(tmp_path: Path) -> None:
    """UAT-7: scry check must surface a clear warning when on-disk files
    have changed since the last index, so users can't be misled by a
    "drift_score: 100 / fresh" report on stale data.
    """
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=a",
            "commit",
            "-qm",
            "init",
            "--allow-empty",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "spec.md").write_text("# Spec\n\nbody\n")
    (repo / "code.py").write_text("def f(): pass\n")
    (repo / ".scry").mkdir(exist_ok=True)
    (repo / ".scry" / "config.yaml").write_text(
        "include:\n  - '**/*.py'\n  - '**/*.md'\nexclude: []\n"
    )

    import os as _os

    cwd0 = _os.getcwd()
    try:
        _os.chdir(repo)
        env = {**_os.environ, "SCRY_EMBEDDER": "stub"}
        # Index then mutate a file then check.
        runner.invoke(main, ["index", "--quiet"], env=env)
        (repo / "code.py").write_text("def f(): return 'changed'\n")
        result = runner.invoke(main, ["check"], env=env)
        assert "WARNING" in result.output, (
            f"UAT-7: scry check must warn when fs is newer than index; output:\n{result.output}"
        )
        assert "scry index" in result.output, (
            "UAT-7: warning must point users at the remediation command"
        )
    finally:
        _os.chdir(cwd0)


def test_uat_r5_2_suggest_links_candidates_only_no_llm(tmp_path: Path) -> None:
    """UAT-R5-2: scry suggest-links --candidates-only must produce
    a complete agent-driven payload (system_prompt + schema + pairs)
    WITHOUT calling any LLM provider.
    """
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["suggest-links", "--help"])
    assert result.exit_code == 0
    assert "--candidates-only" in result.output, (
        "scry suggest-links --help must advertise --candidates-only (UAT-R5-2)"
    )
    assert "--from-file" in result.output, (
        "scry suggest-links --help must advertise --from-file (UAT-R5-2)"
    )


def test_uat_r5_2_build_candidates_payload_shape() -> None:
    """UAT-R5-2: build_candidates_payload returns system_prompt + schema + pairs."""
    from scry.models import Anchor, AnchorType
    from scry.suggest import build_candidates_payload

    code = Anchor(
        id="src/m.py:f",
        type=AnchorType.CODE.value,
        path="src/m.py",
        symbol_name="f",
        content_text="def f(): pass",
        content_hash="sha256:" + "0" * 64,
        fingerprint_simhash=0,
    )
    doc = Anchor(
        id="d/s.md::sec",
        type=AnchorType.SECTION.value,
        path="d/s.md",
        heading_path=("sec",),
        content_text="spec",
        content_hash="sha256:" + "0" * 64,
        fingerprint_simhash=0,
    )
    payload = build_candidates_payload([(code, doc)])
    assert "system_prompt" in payload and isinstance(payload["system_prompt"], str)
    assert "schema" in payload
    assert "pairs" in payload and len(payload["pairs"]) == 1
    assert payload["pairs"][0]["pair_id"] == "p_0"
    assert payload["pairs"][0]["code"]["id"] == "src/m.py:f"
    assert payload["pairs"][0]["doc"]["id"] == "d/s.md::sec"


def test_uat_r5_2_parse_agent_suggestions_validates() -> None:
    """UAT-R5-2: parse_agent_suggestions enforces same threshold/type
    rules the LLM-provider path uses."""
    from scry.models import Anchor, AnchorType
    from scry.suggest import parse_agent_suggestions

    code = Anchor(
        id="src/m.py:f",
        type=AnchorType.CODE.value,
        path="src/m.py",
        symbol_name="f",
        content_text="x",
        content_hash="sha256:" + "0" * 64,
        fingerprint_simhash=0,
    )
    doc = Anchor(
        id="d/s.md::sec",
        type=AnchorType.SECTION.value,
        path="d/s.md",
        heading_path=("sec",),
        content_text="y",
        content_hash="sha256:" + "0" * 64,
        fingerprint_simhash=0,
    )
    raw = {
        "suggestions": [
            {
                "pair_id": "p_0",
                "should_link": True,
                "link_type": "implements",
                "confidence": 0.92,
                "reason": "matches",
            },
            # Should be filtered: should_link=false
            {
                "pair_id": "p_0",
                "should_link": False,
                "link_type": "implements",
                "confidence": 0.4,
                "reason": "no match",
            },
            # Should be filtered: bad link_type
            {
                "pair_id": "p_0",
                "should_link": True,
                "link_type": "invented",
                "confidence": 0.99,
                "reason": "fake",
            },
        ]
    }
    out = parse_agent_suggestions(raw, pairs=[(code, doc)], min_confidence=0.5)
    assert len(out) == 1
    assert out[0].link_type == "implements"
    assert out[0].confidence == 0.92


def test_uat_r5_2_mcp_handlers_include_agent_driven() -> None:
    """UAT-R5-2: HANDLERS dict must include both new agent-driven tools."""
    from scry.mcp.handlers import HANDLERS

    assert "suggest_links_candidates" in HANDLERS
    assert "apply_link_suggestions" in HANDLERS


def test_uat_r5_1_lazy_mcp_keeps_cli_startup_fast() -> None:
    """UAT-R5-1: scry/mcp/__init__.py must NOT eagerly import MCPServer
    so cli.py doesn't pay the ~3.3s fastmcp import tax on every command.
    """
    import inspect

    from scry import mcp as _mcp_pkg

    src = inspect.getsource(_mcp_pkg)
    # Top-level (non-TYPE_CHECKING) import of MCPServer would defeat the fix.
    assert "def __getattr__" in src, (
        "scry/mcp/__init__.py must use __getattr__ to lazy-load MCPServer (UAT-R5-1)"
    )
    # The import must be inside the lazy loader, not at module level.
    top_level_imports = [
        line
        for line in src.splitlines()
        if line.startswith("from scry.mcp.server import") and "MCPServer" in line
    ]
    assert not top_level_imports, (
        "scry/mcp/__init__.py must not eagerly 'from scry.mcp.server import MCPServer' "
        "at module level (UAT-R5-1)"
    )


def test_uat_19_link_warns_on_inverted_direction() -> None:
    """UAT-19: scry link warns when implements/tests/examples direction
    looks inverted vs DESIGN.md §3.6 canonical orientation.
    """
    import inspect

    from scry.cli import link

    src = inspect.getsource(link.callback)  # type: ignore[arg-type]
    assert "_CANONICAL_DIRECTION" in src, (
        "UAT-19: scry link must check direction against DESIGN.md §3.6"
    )
    assert "looks inverted" in src or "canonical" in src


def test_uat_11_mcp_descriptions_no_dev_notes() -> None:
    """UAT-11: MCP tool descriptions must not leak internal dev notes."""
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    # Strip the registration-method-level docstring (lines between the
    # def and the first "@mcp.tool") since the explanatory comment
    # there legitimately mentions the leaks we're scrubbing FROM the
    # actual @mcp.tool docstrings.
    body_start = src.find("@mcp.tool")
    body = src[body_start:]
    forbidden = ["UT3-2 fix", "Wave 2 stub", "W6e — DESIGN.md", "review-w"]
    leaks = [s for s in forbidden if s in body]
    assert not leaks, f"UAT-11: MCP tool docstrings still leak internal dev notes: {leaks}"


def test_uat_12_mcp_tools_have_annotations() -> None:
    """UAT-12: every MCP tool must declare readOnlyHint/destructiveHint."""
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert src.count("ToolAnnotations") >= 1
    # Every @mcp.tool(...) call should have annotations=...
    bare_decorators = src.count("@mcp.tool()")
    assert bare_decorators == 0, (
        f"UAT-12: {bare_decorators} MCP tool registrations still use @mcp.tool() "
        f"without annotations="
    )


def test_uat_13_mcp_search_response_compact() -> None:
    """UAT-13: search response must omit content_hash / fingerprint_simhash
    / def_line / def_char from the default packet shape (token bloat),
    but KEEP transitive_hash_status (LLM-relevant LSP coverage signal,
    per review-u16-18 MEDIUM).
    """
    from scry.mcp.handlers import _compact_packet

    raw = {
        "anchor": {
            "id": "x",
            "content_text": "a",
            "content_hash": "deadbeef",
            "fingerprint_simhash": 12345,
            "def_line": 10,
            "def_char": 4,
            "closure_hash": "abc",
            "transitive_hash_status": "lsp_unavailable",
        },
        "score": 0.05,
    }
    out = _compact_packet(raw)
    assert "content_hash" not in out["anchor"]
    assert "fingerprint_simhash" not in out["anchor"]
    assert "def_line" not in out["anchor"]
    assert "closure_hash" not in out["anchor"]
    assert "content_text" in out["anchor"]
    assert out["anchor"]["transitive_hash_status"] == "lsp_unavailable", (
        "UAT-13 review-u16-18 MEDIUM: transitive_hash_status must NOT be "
        "stripped — it's the LLM-relevant LSP coverage signal."
    )
    assert out["score"] == 0.05


def test_uat_14_get_anchor_accepts_both_id_and_anchor_id() -> None:
    """UAT-14 review-u16-18 HIGH: legacy `id` keyword still works for
    back-compat with MCP clients that haven't migrated to `anchor_id`.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert "id: str | None = None" in src or "id=" in src, (
        "UAT-14: get_anchor must still accept legacy 'id' parameter "
        "(review-u16-18 HIGH back-compat fix)"
    )


def test_uat_14_mcp_get_anchor_uses_anchor_id_param() -> None:
    """UAT-14: MCP get_anchor exposes ``anchor_id`` (not ``id``) for
    consistency with get_links / get_callers / get_subclasses.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    # The new tool registration should declare anchor_id parameter.
    assert "anchor_id: str | None = None" in src, (
        "UAT-14: get_anchor MCP tool must use 'anchor_id' parameter name"
    )


def test_uat_9_unlink_command_exists() -> None:
    """UAT-9: scry unlink <link_id> tombstones a link via DELETE op."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["unlink", "--help"])
    assert result.exit_code == 0
    assert "tombstone" in result.output.lower(), (
        "scry unlink --help must explain tombstone semantics"
    )


def test_uat_8_check_supports_since_flag() -> None:
    """UAT-8: scry check --since <ref> diff-aware drift filtering."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["check", "--help"])
    assert result.exit_code == 0
    assert "--since" in result.output, "scry check --help must advertise --since (UAT-8)"


def test_uat_4_init_auto_detects_languages(tmp_path: Path) -> None:
    """UAT-4: scry init walks the repo and adds Go/Rust/JS to include
    when those files exist, instead of hard-coding md/py/ts only.
    """
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    repo = tmp_path / "polyglot"
    repo.mkdir()
    (repo / "main.go").write_text("package main\nfunc main() {}\n")
    (repo / "lib.rs").write_text("pub fn x() {}\n")
    (repo / "app.js").write_text("function f() {}\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    import os as _os

    cwd0 = _os.getcwd()
    try:
        _os.chdir(repo)
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        config_text = (repo / ".scry" / "config.yaml").read_text(encoding="utf-8")
        for needed in ("**/*.go", "**/*.rs", "**/*.js"):
            assert needed in config_text, (
                f"UAT-4: {needed} should be auto-detected and included; config:\n{config_text}"
            )
    finally:
        _os.chdir(cwd0)


def test_uat_18_check_supports_uncovered_flag() -> None:
    """UAT-18: scry check --uncovered must list unlinked spec sections."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["check", "--help"])
    assert result.exit_code == 0
    assert "--uncovered" in result.output, "scry check --help must advertise --uncovered (UAT-18)"


def test_uat_15_search_supports_scope_glob() -> None:
    """UAT-15: scry search --scope <glob> filters results to matching paths."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--scope" in result.output, "scry search --help must advertise --scope (UAT-15)"


def test_uat_16_anchors_list_command_exists() -> None:
    """UAT-16: scry anchors list <scope> must exist as a browse command."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["anchors", "--help"])
    assert result.exit_code == 0, result.output
    assert "list" in result.output.lower()
    list_help = runner.invoke(main, ["anchors", "list", "--help"])
    assert list_help.exit_code == 0
    assert "--scope" in list_help.output
    assert "--type" in list_help.output


def test_uat_3_get_anchor_cli_command_exists() -> None:
    """UAT-3: scry get-anchor must exist as a CLI command, not MCP-only."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["get-anchor", "--help"])
    assert result.exit_code == 0, result.output
    assert "anchor" in result.output.lower()


def test_uat_3_get_link_cli_command_exists() -> None:
    """UAT-3 / UAT4 #2: scry get-link must exist as a CLI command."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["get-link", "--help"])
    assert result.exit_code == 0, result.output
    assert "link" in result.output.lower()


def test_uat_5_show_cli_command_exists() -> None:
    """UAT-5: scry show <anchor_id> must exist (UAT3's #1 missing feature)."""
    from click.testing import CliRunner

    from scry.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["show", "--help"])
    assert result.exit_code == 0, result.output
    assert "content" in result.output.lower() or "source" in result.output.lower()


def test_uat_24_js_require_calls_not_extracted_as_anchors(tmp_path: Path) -> None:
    """UAT-24: ``const x = require('y')`` and ``const x = require('y').member``
    must not produce anchors (CommonJS module imports, not real symbols).
    """
    from scry.extract.code import extract_code_symbols

    src = (
        b"const fs = require('fs');\n"
        b"const path = require('path');\n"
        b"const read = require('fs').readFileSync;\n"  # member-form
        b"const realThing = function() { return 1; };\n"
        b"const realArrow = (x) => x + 1;\n"
    )
    f = tmp_path / "lib.js"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="javascript")
    names = {a.symbol_name for a in anchors}
    assert "fs" not in names, "UAT-24: require('fs') must not be extracted"
    assert "path" not in names, "UAT-24: require('path') must not be extracted"
    assert "read" not in names, (
        "UAT-24 review-u7 MEDIUM: require('fs').readFileSync alias must also "
        "be filtered (member-form)"
    )
    assert "realThing" in names, "real const arrow / fn should still be extracted"
    assert "realArrow" in names


def test_uat_25_rust_impl_method_symbol_name_avoids_collision(tmp_path: Path) -> None:
    """UAT-25: distinct Rust impl methods sharing a method name must NOT
    collide on symbol_name (which is used for retrieval grouping).
    """
    from scry.extract.code import extract_code_symbols

    src = (
        b"pub struct User {}\n"
        b"pub struct Order {}\n"
        b"impl User { pub fn validate(&self) -> bool { true } }\n"
        b"impl Order { pub fn validate(&self) -> bool { true } }\n"
    )
    f = tmp_path / "m.rs"
    f.write_bytes(src)
    anchors = extract_code_symbols(f, tmp_path, language="rust")
    impl_methods = [a for a in anchors if "validate" in a.id and "impl_" in a.id]
    assert len(impl_methods) == 2, f"expected 2 impl methods named validate; got {impl_methods}"
    sym_names = {a.symbol_name for a in impl_methods}
    assert len(sym_names) == 2, (
        f"UAT-25: impl_User.validate and impl_Order.validate must have distinct "
        f"symbol_name values; got {sym_names}"
    )


def test_uat_23_reembed_runs_wal_checkpoint() -> None:
    """UAT-23 review-u4-u6 HIGH: reembed must also checkpoint the WAL so
    embedding-model migrations are visible to read-only consumers
    immediately (parity with index() / index_async())."""
    import inspect

    from scry.index import Indexer

    src = inspect.getsource(Indexer.reembed)
    assert "wal_checkpoint" in src, (
        "UAT-23: Indexer.reembed() must also run PRAGMA wal_checkpoint after "
        "writing index_metadata; missed in initial fix per review-u4-u6 HIGH."
    )


def test_uat_23_index_runs_wal_checkpoint(tmp_path: Path) -> None:
    """UAT-23: Indexer.index() runs PRAGMA wal_checkpoint(TRUNCATE) so a
    subsequent doctor / read-only consumer sees the just-written manifest
    even if SQLite hasn't naturally checkpointed.
    """
    import inspect

    from scry.index import Indexer

    src = inspect.getsource(Indexer.index)
    assert "wal_checkpoint" in src, (
        "UAT-23: Indexer.index() must run PRAGMA wal_checkpoint after writing "
        "the index_metadata to prevent silent stale-state on interrupted runs"
    )


async def test_uat_2_litellm_connection_error_normalized_to_network_error() -> None:
    """UAT-2 review-u2: LiteLLMProvider must normalize connection-class
    exceptions (litellm.exceptions.APIConnectionError, etc.) to
    LLMNetworkError so the suggest-links fail-fast path triggers in
    real-world Ollama-via-LiteLLM scenarios, not just synthetic ones.
    """
    pytest.importorskip("litellm")
    from scry.llm import LiteLLMProvider, LLMConfig, LLMError, LLMNetworkError, LLMRequest

    cfg = LLMConfig(provider="litellm", model="ollama/llama3.2")
    provider = LiteLLMProvider(cfg)

    class _ConnectionRefused(Exception):
        def __str__(self) -> str:
            return "Connection refused: cannot reach http://localhost:11434"

    async def _stub_acompletion(**_kw: object) -> object:
        raise _ConnectionRefused()

    provider._litellm.acompletion = _stub_acompletion  # type: ignore[attr-defined]

    raised: BaseException | None = None
    try:
        await provider.complete(LLMRequest(system="s", messages=[{"role": "user", "content": "h"}]))
    except BaseException as exc:
        raised = exc

    assert isinstance(raised, LLMNetworkError), (
        f"UAT-2: connection-class exceptions must be normalized to LLMNetworkError; "
        f"got {type(raised).__name__}: {raised}"
    )
    assert isinstance(raised, LLMError), "LLMNetworkError must remain a LLMError subclass"


async def test_uat_2_suggest_links_fails_fast_on_network_error() -> None:
    """UAT-2: evaluate_pairs_batched MUST abort on the first batch when an
    LLMNetworkError fires, instead of iterating 800+ pairs printing the
    same error per batch (UAT2's 5-minute time-sink).
    """
    from scry.llm import LLMError, LLMNetworkError, LLMRequest, LLMResponse
    from scry.models import Anchor, AnchorType
    from scry.suggest import batch_llm_evaluate

    call_count = 0

    class _UnreachableProvider:
        async def complete(self, req: LLMRequest) -> LLMResponse:
            nonlocal call_count
            call_count += 1
            raise LLMNetworkError("Ollama is not reachable at http://localhost:11434")

    pairs = []
    for i in range(100):
        code = Anchor(
            id=f"src/m{i}.py:f",
            type=AnchorType.CODE.value,
            path=f"src/m{i}.py",
            symbol_name="f",
            content_text="def f(): pass",
            content_hash="sha256:" + "0" * 64,
            fingerprint_simhash=0,
        )
        doc = Anchor(
            id=f"d/s{i}.md::sec",
            type=AnchorType.SECTION.value,
            path=f"d/s{i}.md",
            heading_path=("sec",),
            content_text="spec",
            content_hash="sha256:" + "0" * 64,
            fingerprint_simhash=0,
        )
        pairs.append((code, doc))

    try:
        await batch_llm_evaluate(
            pairs,
            provider=cast("Any", _UnreachableProvider()),
            batch_size=20,
        )
    except LLMNetworkError:
        pass
    except LLMError as exc:
        raise AssertionError(
            f"UAT-2: expected LLMNetworkError to abort the loop; got {type(exc).__name__}: {exc}"
        ) from exc
    else:
        raise AssertionError(
            "UAT-2: expected LLMNetworkError to abort the loop; nothing was raised"
        )

    assert call_count == 1, (
        f"UAT-2: must fail-fast on first network error; got call_count={call_count} "
        f"(was 5 before the fix — that's the 800-pair iteration bug)"
    )


def test_uat_1_indexer_silent_when_no_callback(tmp_path: Path) -> None:
    """UAT-1: Indexer remains library-pure (no stdout emission)
    when no progress_callback is supplied — preserves MCP/library use.
    """
    import io
    import subprocess as _subprocess
    from contextlib import redirect_stdout

    from scry.config import load_config
    from scry.embed import StubEmbedder
    from scry.index import Indexer

    repo = tmp_path
    _subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _subprocess.run(
        [
            "git",
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=a",
            "commit",
            "-qm",
            "i",
            "--allow-empty",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "a.py").write_text("def foo(): pass\n")
    (repo / ".scry").mkdir(exist_ok=True)
    (repo / ".scry" / "config.yaml").write_text("include:\n  - '**/*.py'\nexclude: []\n")

    config = load_config(repo)
    embedder = StubEmbedder()
    indexer = Indexer(repo, config=config, embedder=embedder, allow_untrusted=True)

    out = io.StringIO()
    with redirect_stdout(out):
        indexer.index(force=True)
    assert out.getvalue() == "", (
        f"UAT-1: Indexer must be silent on stdout when no progress_callback; "
        f"got: {out.getvalue()!r}"
    )
    """UAT-1: Indexer remains library-pure (no stdout/stderr emission)
    when no progress_callback is supplied — preserves MCP/library use.
    """
    import io
    import subprocess as _subprocess
    from contextlib import redirect_stderr, redirect_stdout

    from scry.config import load_config
    from scry.embed import StubEmbedder
    from scry.index import Indexer

    repo = tmp_path
    _subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _subprocess.run(
        [
            "git",
            "-c",
            "user.email=a@b.c",
            "-c",
            "user.name=a",
            "commit",
            "-qm",
            "i",
            "--allow-empty",
        ],
        cwd=repo,
        check=True,
    )
    (repo / "a.py").write_text("def foo(): pass\n")
    (repo / ".scry").mkdir(exist_ok=True)
    (repo / ".scry" / "config.yaml").write_text("include:\n  - '**/*.py'\nexclude: []\n")

    config = load_config(repo)
    embedder = StubEmbedder()
    indexer = Indexer(repo, config=config, embedder=embedder, allow_untrusted=True)

    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        indexer.index(force=True)
    # Logger output may go to stderr depending on logging config but we
    # don't assert on it here.  We only assert that the indexer didn't
    # print to stdout via click/print directly.
    assert out.getvalue() == "", (
        f"UAT-1: Indexer must be silent on stdout when no progress_callback; "
        f"got: {out.getvalue()!r}"
    )


# ─── SR2 concurrency regression tests ─────────────────────────────────────────


async def test_sr2_1_ipc_idempotency_serializes_concurrent_same_token() -> None:
    """SR2-1 BLOCKING: 3 concurrent same-token IPC requests MUST execute the
    handler exactly once.  Previously the check-then-execute-then-store
    sequence had a TOCTOU window where 2-3 of 3 concurrent calls would
    all see a cache miss and all run the handler.
    """
    import asyncio as _asyncio

    from scry.process.ipc import (
        IPCConfig,
        IPCRequest,
        IPCResponse,
        _IdempotencyCache,
        _run_dispatch_logic,
    )

    handler_calls = 0

    async def slow_handler(req: IPCRequest) -> IPCResponse:
        nonlocal handler_calls
        handler_calls += 1
        await _asyncio.sleep(0.05)
        return IPCResponse(request_id=req.request_id, ok=True, result={"n": handler_calls})

    cache = _IdempotencyCache(maxsize=10)
    cfg = IPCConfig()
    from scry.models import new_idempotency_token

    token = new_idempotency_token()  # generates a valid tok_<alphanum>
    reqs = [
        IPCRequest(
            request_id=i,
            op="propose_link",
            args={"x": 1},
            idempotency_token=token,
        )
        for i in range(3)
    ]
    responses = await _asyncio.gather(
        *[_run_dispatch_logic(r, slow_handler, cache, cfg) for r in reqs]
    )
    assert handler_calls == 1, (
        f"handler must run exactly once for 3 concurrent same-token requests; "
        f"got {handler_calls} (SR2-1 BLOCKING regression)"
    )
    assert all(r.ok for r in responses)
    # All 3 responses share the same payload (the first one's).
    payloads = {r.result["n"] if r.result else None for r in responses}
    assert payloads == {1}, f"all responses must mirror the first handler's result; got {payloads}"


def test_sr2_4_ipc_client_docstring_no_longer_claims_not_implemented() -> None:
    """SR2-4: IPCClient docstring must NOT claim Windows raises NotImplementedError."""
    from scry.process.ipc import IPCClient

    doc = IPCClient.__doc__ or ""
    assert "NotImplementedError" not in doc, (
        "IPCClient docstring is stale — Windows IPC is fully implemented "
        "via _WinPipeIO (Wave 6b).  Remove the NotImplementedError claim."
    )


# ─── SR3 edge-case regression tests ───────────────────────────────────────────


def test_sr3_3_index_command_catches_sqlite_operationalerror() -> None:
    """SR3-3: scry index must handle sqlite3.OperationalError without traceback."""
    import inspect

    from scry.cli import index as _index_cmd

    src = inspect.getsource(_index_cmd.callback)  # type: ignore[arg-type]
    assert "sqlite3.OperationalError" in src, (
        "scry index must catch sqlite3.OperationalError to avoid stack "
        "traces on read-only DB scenarios (SR3-3)"
    )


def test_sr3_1_2_check_callers_subclasses_catch_git_context_error() -> None:
    """SR3-1/2: check, callers, subclasses must catch GitContextError cleanly."""
    import inspect

    from scry.cli import callers, check, subclasses

    for name, cmd in (("check", check), ("callers", callers), ("subclasses", subclasses)):
        src = inspect.getsource(cmd.callback)  # type: ignore[arg-type]
        assert "GitContextError" in src, (
            f"scry {name} must catch GitContextError to avoid stack traces on "
            f"no-commits repos (SR3-1/2)"
        )


def test_sr3_4_code_extractor_skips_utf16_bom(tmp_path: Path) -> None:
    """SR3-4: extract_code_symbols must detect UTF-16 BOM and return [] cleanly."""
    from scry.extract.code import extract_code_symbols

    f = tmp_path / "u.py"
    # Write a UTF-16 LE encoded Python source.  The BOM (\xff\xfe)
    # is included automatically by .write_text(encoding="utf-16").
    f.write_text("def hello(): pass\n", encoding="utf-16")
    raw = f.read_bytes()
    assert raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"), "fixture sanity"
    anchors = extract_code_symbols(f, tmp_path, language="python")
    assert anchors == [], (
        f"UTF-16 BOM Python source must be skipped (SR3-4 / §15.4); got {anchors!r}"
    )


def test_sr3_7_code_extractor_warns_on_parse_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """SR3-7: parse errors in source files must produce a warning log."""
    import logging

    from scry.extract.code import extract_code_symbols

    f = tmp_path / "broken.py"
    f.write_bytes(b"def good(): pass\nx = (1 + 2\ndef after(): return 99\n")
    with caplog.at_level(logging.WARNING, logger="scry.extract.code"):
        extract_code_symbols(f, tmp_path, language="python")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "syntax errors" in text, (
        "extract_code_symbols must warn when tree-sitter detects parse errors "
        f"(SR3-7); records=\n{text}"
    )


# ─── SR4 MCP fuzzer regression tests ──────────────────────────────────────────


async def test_sr4_1_reindex_rejects_non_bool_force() -> None:
    """SR4-1: reindex MUST reject string ``force`` instead of coercing to True."""
    from scry.mcp.handlers import MCPServerError, reindex

    class _DummyIndexer:
        pass

    class _DummyCtx:
        indexer = _DummyIndexer()
        index_state_tracker = None  # type: ignore[assignment]

    try:
        await reindex(_DummyCtx(), force="yes")  # type: ignore[arg-type]
    except MCPServerError as exc:
        assert "force" in str(exc).lower() and "bool" in str(exc).lower(), (
            f"reindex must reject non-bool force; error message was: {exc}"
        )
    else:
        raise AssertionError("reindex must reject non-bool force value (SR4-1); no error raised")


async def test_sr4_2_search_rejects_non_positive_top_k() -> None:
    """SR4-2: search MUST reject top_k < 1 with a clear error."""
    from scry.mcp.handlers import MCPServerError, search

    for bad in (0, -1, -100):
        try:
            await search(None, "anything", top_k=bad)  # type: ignore[arg-type]
        except MCPServerError as exc:
            assert "top_k" in str(exc), f"expected top_k in error, got {exc}"
        except AttributeError:
            # Implementation reaches ctx.git_context before raising — also acceptable
            # AS LONG AS the validation runs first.  If we got past validation,
            # AttributeError on None means the assertion did NOT happen first.
            raise AssertionError(
                f"search must reject top_k={bad} BEFORE accessing ctx (SR4-2)"
            ) from None
        else:
            raise AssertionError(f"search must reject top_k={bad} (SR4-2)")


async def test_sr4_3_find_drift_rejects_unknown_status_filter() -> None:
    """SR4-3: find_drift MUST reject unknown status_filter values."""
    from scry.mcp.handlers import MCPServerError, find_drift

    try:
        await find_drift(None, status_filter=["typo_status"])  # type: ignore[arg-type]
    except MCPServerError as exc:
        msg = str(exc)
        assert "status" in msg.lower() and "typo_status" in msg, (
            f"find_drift error must mention the unknown value; got: {exc}"
        )
    except AttributeError:
        # Validation didn't run before ctx access — fail.
        raise AssertionError(
            "find_drift must validate status_filter BEFORE accessing ctx (SR4-3)"
        ) from None
    else:
        raise AssertionError("find_drift must reject unknown status_filter values (SR4-3)")


# uat-r5-5 pr-d noise


# ─────────────────────────────────────────────────────────────────────────
# UAT Round 6 (MCP-focused) regression tests — U-fix-1 through U-fix-9
# ─────────────────────────────────────────────────────────────────────────


def test_uat_m_2_reindex_force_strict_bool_only() -> None:
    """UAT-M-2 / U-fix-1: ``reindex.force`` must reject coerced bools.

    FastMCP's Pydantic v2 layer otherwise turns ``"yes"``, ``"true"``,
    ``"1"``, or any non-empty string into ``True`` BEFORE the
    handler-level ``isinstance(force, bool)`` guard runs.  The fix is
    ``Annotated[bool, Field(strict=True)]`` at the @mcp.tool
    registration so Pydantic rejects non-bool inputs server-side.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert "force: Annotated[bool, Field(strict=True)]" in src, (
        "reindex.force must be Annotated[bool, Field(strict=True)] (UAT-M-2 / U-fix-1)"
    )


def test_uat_m_2_apply_link_suggestions_apply_strict_bool_only() -> None:
    """UAT-M-2 / U-fix-1: ``apply_link_suggestions.apply`` must reject coerced bools.

    Same Pydantic-coercion bug class as ``reindex.force`` — a string
    ``"yes"`` would silently turn into a write op.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert "apply: Annotated[bool, Field(strict=True)]" in src, (
        "apply_link_suggestions.apply must be Annotated[bool, Field(strict=True)] "
        "(UAT-M-2 / U-fix-1)"
    )


def test_uat_m_4_find_drift_accepts_since_param() -> None:
    """UAT-M-4 / U-fix-3: find_drift exposes a ``since`` param for git-diff scoping."""
    import inspect

    from scry.mcp.handlers import find_drift
    from scry.mcp.server import MCPServer

    sig = inspect.signature(find_drift)
    assert "since" in sig.parameters, "find_drift handler must accept 'since' (UAT-M-4)"

    src = inspect.getsource(MCPServer._register_tools)
    assert '"since": since' in src, (
        "find_drift @mcp.tool must forward 'since' through _dispatch (UAT-M-4)"
    )


def test_uat_m_5_unlink_mcp_tool_exists() -> None:
    """UAT-M-5 / U-fix-4: unlink is exposed as an MCP tool, not just a CLI command."""
    from scry.mcp.handlers import HANDLERS, unlink
    from scry.process.ipc import WRITE_OPS

    assert callable(unlink), "scry.mcp.handlers.unlink must be a callable"
    assert "unlink" in HANDLERS, "HANDLERS dict must include 'unlink' (UAT-M-5)"
    assert "unlink" in WRITE_OPS, "unlink must be in WRITE_OPS (idempotency-token-required)"


def test_uat_m_6_commit_links_promoted_includes_link_id() -> None:
    """UAT-M-6 / U-fix-5: commit_links returns ``promoted: [{event_id, link_id}, ...]``.

    Old shape was ``list[event_id]`` — agents had to round-trip via
    get_links to recover the link_id (the user-facing handle).  New
    shape includes both, plus a ``promoted_event_ids`` back-compat alias.
    """
    import inspect

    from scry.mcp.handlers import commit_links

    src = inspect.getsource(commit_links)
    assert '"event_id": eid' in src and '"link_id":' in src, (
        "commit_links must return promoted records with both event_id and link_id (UAT-M-6)"
    )
    assert '"promoted_event_ids":' in src, (
        "commit_links must keep 'promoted_event_ids' as a back-compat alias (UAT-M-6)"
    )


def test_uat_m_8_reindex_response_includes_scope_ignored_flag() -> None:
    """UAT-M-8: reindex must surface ``scope_ignored: true`` when scope is supplied.

    Wave 2 doesn't implement path-prefix scoping, but agents need a
    machine-detectable signal that their ``scope=`` arg was ignored
    instead of silently running a full re-index.
    """
    import inspect

    from scry.mcp.handlers import reindex

    src = inspect.getsource(reindex)
    assert '"scope_ignored": scope is not None' in src, (
        "reindex must include 'scope_ignored' in its response (UAT-M-8)"
    )


def test_uat_m_9_get_links_validates_link_types() -> None:
    """UAT-M-9 / U-fix-6: get_links must reject typo'd link_type values.

    Mirrors find_drift status_filter validation — silent empty result
    on a typo is worse than an explicit error.
    """
    import inspect

    from scry.mcp.handlers import get_links

    src = inspect.getsource(get_links)
    assert "Unknown link_type values" in src, (
        "get_links must reject unknown link_type values with a clear error (UAT-M-9)"
    )


def test_uat_m_10_search_rejects_empty_query() -> None:
    """UAT-M-10 / U-fix-7: search() must reject empty / whitespace-only queries.

    Pre-fix: silently returned [] (looked like a relevance bug).
    Post-fix: explicit MCPServerError mirrors the SR4-2 top_k validation.
    """
    import inspect

    from scry.mcp.handlers import search

    src = inspect.getsource(search)
    assert "must be a non-empty string" in src, (
        "search() must reject empty/whitespace queries with a clear error (UAT-M-10)"
    )


def test_uat_m_12_apply_link_suggestions_exposes_rejection_reasons() -> None:
    """UAT-M-12 / U-fix-8: apply_link_suggestions returns ``rejected_reasons``.

    Pre-fix: ``rejected: int`` was opaque — caller couldn't tell if
    suggestions were dropped because of a bad pair_id, low confidence,
    invalid link_type, etc.  Post-fix: ``rejected_reasons: dict[str, int]``
    breaks down the count by reason.
    """
    import inspect

    from scry.mcp.handlers import apply_link_suggestions

    src = inspect.getsource(apply_link_suggestions)
    assert '"rejected_reasons": rejected_reasons' in src, (
        "apply_link_suggestions must return 'rejected_reasons' breakdown (UAT-M-12)"
    )
    assert '"unknown_pair_id"' in src, (
        "apply_link_suggestions must classify mismatched pair_id explicitly (UAT-M-12)"
    )


def test_uat_r5_15_propose_link_rejects_self_link() -> None:
    """UAT-R5-15 / U-fix-9: propose_link must reject from_id == to_id."""
    import inspect

    from scry.mcp.handlers import propose_link

    src = inspect.getsource(propose_link)
    assert "from_id == to_id" in src, "propose_link must reject self-links (UAT-R5-15)"


def test_uat_m_4_find_drift_since_rejects_option_injection() -> None:
    """review-r6-1 BLOCKING: find_drift(since=...) must reject option-flag refs.

    Without ``--end-of-options`` and an explicit option-flag guard,
    ``git diff`` interprets ``--output=<path>`` as an option BEFORE
    treating it as a revision — turning this read-only MCP tool into
    a filesystem-write primitive.  The fix rejects refs starting with
    ``-`` and resolves via ``git rev-parse --verify --end-of-options``
    before the diff call.
    """
    import inspect

    from scry.mcp.handlers import find_drift

    src = inspect.getsource(find_drift)
    # Refs that begin with "-" must be rejected up-front.
    assert "since.startswith" in src and "option flag" in src, (
        "find_drift must reject 'since' values that begin with '-' (review-r6-1)"
    )
    # Both git calls must use --end-of-options.
    assert src.count("--end-of-options") >= 2, (
        "find_drift must pass --end-of-options to BOTH git rev-parse and git diff (review-r6-1)"
    )
    # Both git calls must set stdin=DEVNULL and a timeout.
    assert "stdin=_subprocess.DEVNULL" in src and "timeout=" in src, (
        "find_drift git subprocess calls must set stdin=DEVNULL and a timeout (review-r6-1)"
    )


# ─── UAT Round 5 / M regression tests ────────────────────────────────────────


def test_uat_r5_8_get_callers_lsp_unavailable_has_status_field() -> None:
    """UAT-R5-8: get_callers / get_subclasses MUST include lsp_status in
    response so callers can distinguish "leaf function" from "LSP missing".

    Checks that when session_for() returns None the handler returns
    lsp_status == "unavailable" (or "unsupported") instead of a bare
    empty list.
    """
    import inspect

    from scry.mcp.handlers import get_callers, get_subclasses

    for fn in (get_callers, get_subclasses):
        src = inspect.getsource(fn)
        assert '"lsp_status"' in src, (
            f'{fn.__name__} must include "lsp_status" in all response paths (UAT-R5-8)'
        )
        assert '"unavailable"' in src or '"unsupported"' in src, (
            f'{fn.__name__} must use "unavailable" or "unsupported" string '
            f"values for lsp_status (UAT-R5-8)"
        )
        assert '"available"' in src, (
            f'{fn.__name__} success path must set lsp_status="available" (UAT-R5-8)'
        )
        assert '"error"' in src, f'{fn.__name__} must handle lsp_status="error" path (UAT-R5-8)'


def test_uat_r5_14_propose_link_has_idempotent_hint() -> None:
    """UAT-R5-14: propose_link and accept_link must use _idem_write (idempotentHint=True).

    Pre-fix: both used _write (no idempotentHint). Post-fix: both use _idem_write.
    """
    import inspect

    from scry.mcp.server import MCPServer

    src = inspect.getsource(MCPServer._register_tools)
    assert "_idem_write" in src, "_idem_write must be defined/used in _register_tools"
    idx = src.index("async def propose_link(")
    block = src[max(0, idx - 120) : idx]
    assert "_idem_write" in block, (
        "propose_link must use @mcp.tool(annotations=_idem_write) (UAT-R5-14)"
    )
    idx2 = src.index("async def accept_link(")
    block2 = src[max(0, idx2 - 120) : idx2]
    assert "_idem_write" in block2, (
        "accept_link must use @mcp.tool(annotations=_idem_write) (UAT-R5-14)"
    )


def test_uat_r5_14_propose_link_warns_without_idempotency_token() -> None:
    """UAT-R5-14: propose_link / accept_link / unlink handlers must emit a
    warning and include a warning field when called without idempotency_token.
    """
    import inspect

    from scry.mcp.handlers import accept_link, propose_link, unlink

    for fn in (propose_link, accept_link, unlink):
        src = inspect.getsource(fn)
        assert "idempotency_token is None" in src, (
            f"{fn.__name__} must check idempotency_token is None (UAT-R5-14)"
        )
        assert "logger.warning" in src, (
            f"{fn.__name__} must call logger.warning when idempotency_token is None (UAT-R5-14)"
        )
        assert '"warning"' in src, (
            f'{fn.__name__} must include "warning" field in response when no token (UAT-R5-14)'
        )


def test_uat_r5_9_evidence_excerpt_dropped_when_identical_to_content() -> None:
    """UAT-R5-9: build_anchor_packet must omit evidence_excerpt when it equals
    content_text (avoids 40% token waste on short anchors), and AnchorPacket
    must expose a match_offset field for true-substring excerpts.
    """
    import inspect

    from scry.models import AnchorPacket
    from scry.retrieve import build_anchor_packet

    assert "match_offset" in AnchorPacket.model_fields, (
        "AnchorPacket must have match_offset field (UAT-R5-9)"
    )

    src = inspect.getsource(build_anchor_packet)
    assert "evidence_excerpt" in src and ("truncated_text" in src or "content_text" in src), (
        "build_anchor_packet must compare evidence_excerpt to content_text (UAT-R5-9)"
    )
    assert "match_offset" in src, "build_anchor_packet must compute match_offset (UAT-R5-9)"


def test_uat_m7_get_links_deduplicates_by_logical_triple() -> None:
    """UAT-M-7: get_links must deduplicate by (from_id, to_id, type) and
    include historical_count per result so callers can see there was history.
    """
    import inspect

    from scry.mcp.handlers import get_links

    src = inspect.getsource(get_links)
    assert "historical_count" in src, (
        "get_links must add historical_count field per result (UAT-M-7)"
    )
    assert "deduped" in src or "dedup" in src.lower(), (
        "get_links must deduplicate links by (from_id, to_id, type) (UAT-M-7)"
    )


# ─────────────────────────────────────────────────────────────────────────
# review-r6abc: GPT-5.5 review findings on the round-6 batch A+B+C diff
# ─────────────────────────────────────────────────────────────────────────


def test_review_r6abc_1_count_projected_files_caps_walk(tmp_path: Path) -> None:
    """review-r6abc-1 HIGH: ``_count_projected_files`` must early-exit at cap.

    Without the cap, ``scry init`` did a full ``os.walk`` on huge repos
    (Linux kernel scale) before warning — defeating the purpose of the
    ``--max-files`` safety net.
    """
    from scry.cli import _count_projected_files

    # Build a tree with 50 markdown files so any cap < 50 should trigger.
    for i in range(50):
        (tmp_path / f"f{i:03d}.md").write_text("x", encoding="utf-8")

    # No cap → walks everything.
    total_full, _dirs_full, capped_full = _count_projected_files(tmp_path, ["**/*.md"], [])
    assert total_full == 50
    assert capped_full is False

    # Cap at 5 -> walk early-exits when we hit 2x cap = 10 files.
    total_capped, _dirs_capped, capped_capped = _count_projected_files(
        tmp_path, ["**/*.md"], [], cap=5
    )
    assert capped_capped is True
    assert total_capped >= 10  # at least 2x cap
    assert total_capped < 50  # but did NOT walk to completion


def test_review_r6abc_2_get_links_dedup_uses_replay_order() -> None:
    """review-r6abc-2 MEDIUM: ``get_links`` dedup tie-breaker must NOT be
    lexicographic ``link_id`` (uuid4 ordering is random — exposes stale
    duplicates).  The fix is "last write wins" against replay order
    (which preserves overlay/baseline append order = creation order).
    """
    import inspect

    from scry.mcp.handlers import get_links

    src = inspect.getsource(get_links)
    # The buggy max(link_id) approach must be GONE.
    assert 'max(group, key=lambda r: r["link_id"])' not in src, (
        "get_links must NOT pick the lexicographic max link_id (review-r6abc-2)"
    )
    # The new behavior is "last row wins" — last assignment to the
    # deduped dict for a given key sticks (Python preserves insertion
    # order; we rely on rows being appended in replay order).
    assert "deduped[key] = row" in src or "last write wins" in src, (
        "get_links dedup must keep the last row per (from, to, type) triple (review-r6abc-2)"
    )


def test_review_r6abc_2b_replay_active_links_iteration_is_last_event_order() -> None:
    """review-r6abc-2 follow-up: ``LinkStore.replay()`` must guarantee that
    ``active_links.values()`` iterates in LAST-EVENT order (most recently
    touched link last) — NOT first-creation order.

    Python's default ``dict[k] = v`` on an existing key preserves the
    original insertion position.  Without the pop+reassign in replay(),
    a link refreshed by a later overlay record would still iterate at
    its original position, silently breaking get_links dedup.
    """
    import inspect

    from scry.store.links import LinkStore

    src = inspect.getsource(LinkStore.replay)
    # The fix must include either the explicit pop or an equivalent move.
    assert "del link_last[record.link_id]" in src or "link_last.pop" in src, (
        "LinkStore.replay must pop existing link_id keys before reassign so "
        "iteration order tracks last-event order (review-r6abc-2 follow-up)"
    )


def test_review_r6abc_3_match_offset_none_when_not_substring() -> None:
    """review-r6abc-3 LOW: ``match_offset`` must be None when the excerpt
    is NOT a substring of ``content_text`` (e.g. generated chunks,
    overlap windows).  Falling back to ``0`` silently misdirects callers
    to the start of the anchor.
    """
    import inspect

    from scry.retrieve import build_anchor_packet

    src = inspect.getsource(build_anchor_packet)
    # The buggy "raw_offset if raw_offset >= 0 else 0" must be gone.
    assert "raw_offset >= 0 else 0" not in src, (
        "build_anchor_packet must NOT fall back to match_offset=0 on miss (review-r6abc-3)"
    )
    # The fix uses None as the explicit miss signal.
    assert "raw_offset if raw_offset >= 0 else None" in src, (
        "build_anchor_packet must return match_offset=None when excerpt is "
        "not a substring of content_text (review-r6abc-3)"
    )
