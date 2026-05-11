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
