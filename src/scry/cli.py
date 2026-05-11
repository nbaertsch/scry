"""Click-based CLI for scry (workstream W2j).

Implements the full §9 CLI surface from DESIGN.md v3.1.  Every subcommand
maps 1:1 to a row in the §9 table.  Three Wave-2 deferrals are stubbed
with informative messages and exit 0.

Global flag
-----------
``--allow-untrusted-lsp-config`` — wired per §6.2; Wave 3 (LSP) will
consume the value. For Wave 2 the flag is plumbed through the Click
context and ``scry doctor`` prints a warning when it is set.

Deferred commands (stubs)
-------------------------
``scry watch``     — Wave 6; prints deferral message, exits 0.
``scry reconcile`` — Wave 5; prints deferral message, exits 0.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

import click

# Windows console default code page (cp1252) cannot encode the unicode
# punctuation we use in help text and doctor output (↔, →, §, em-dashes,
# etc.). Force UTF-8 with replacement so click.echo never crashes the CLI.
# Python 3.7+ stdio supports reconfigure(); on Windows Terminal / VS Code
# this is a no-op (stdout is already UTF-8) — only legacy cmd.exe / older
# PowerShell hosts need it.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        _reconfigure = getattr(_stream, "reconfigure", None)
        if _reconfigure is not None:
            with contextlib.suppress(OSError, ValueError):  # pragma: no cover
                _reconfigure(encoding="utf-8", errors="replace")

import scry
from scry.config import ConfigError, load_config, parse_frontmatter
from scry.drift import compute_drift_summary, evaluate_all_drift
from scry.embed import StubEmbedder, make_embedder
from scry.git_context import GitContextError, GitContextProvider
from scry.index import Indexer, IndexerError, IndexResult
from scry.mcp.handlers import MCPServerError
from scry.mcp.server import MCPServer
from scry.models import (
    AnchorType,
    DriftStatus,
    IndexState,
    LinkOp,
    LinkRecord,
    LinkType,
    new_event_id,
    new_link_id,
)
from scry.process.leader import LeaderState, detect_leader_state
from scry.reconcile import check_staleness
from scry.retrieve import build_anchor_packet, hybrid_search
from scry.store.db import LockTimeout, ScryDB
from scry.store.links import LinkStore, LinkValidationError, MergeConflictError
from scry.store.overlay import OverlayManager

# ─── LSP allowlist (§6.2) ─────────────────────────────────────────────────────

_LSP_ALLOWLIST: dict[str, list[str]] = {
    "python": ["pyright-langserver", "pylsp", "basedpyright-langserver"],
    "typescript": ["typescript-language-server"],
    "tsx": ["typescript-language-server"],
    "javascript": ["typescript-language-server"],
    "jsx": ["typescript-language-server"],
    "zig": ["zls"],
    "go": ["gopls"],
    "rust": ["rust-analyzer"],
}

# ─── Default config YAML text written by `scry init` ─────────────────────────

_DEFAULT_INCLUDE_GLOBS: list[str] = ["**/*.md", "**/*.py", "**/*.ts"]

# UAT-4: extension → glob mapping for `scry init` language detection.
# Walk the repo (skipping VCS / venv / node_modules) and add every
# matching extension's glob to the config.  Map is intentionally
# language-by-language so we don't accidentally include image/binary
# extensions just because the user has one in tree.
_DETECT_EXT_TO_GLOB: dict[str, str] = {
    ".md": "**/*.md",
    ".py": "**/*.py",
    ".ts": "**/*.ts",
    ".tsx": "**/*.tsx",
    ".js": "**/*.js",
    ".jsx": "**/*.jsx",
    ".go": "**/*.go",
    ".rs": "**/*.rs",
    ".zig": "**/*.zig",
}

_DETECT_SKIP_DIRS: set[str] = {
    ".git",
    ".scry",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
    # UAT-4 review-u11-u13 MEDIUM: also skip the default-excluded cache /
    # generated dirs so the 5000-file cap isn't burned inside trees the
    # generated config will exclude anyway.
    ".next",
    ".nuxt",
    ".parcel-cache",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    "out",
    "tmp",
    ".tmp",
    "vendor",
}


def _detect_repo_languages(repo: Path, max_files: int = 5000) -> list[str]:
    """Return ordered include-glob list based on what's present in *repo*.

    UAT-4: scry init previously hard-coded include = [md, py, ts] and
    silently dropped Go / Rust / JS files on polyglot repos.  This walk
    inspects up to ``max_files`` files (cap to keep large repos fast)
    and emits every glob whose extension was seen at least once.

    Always includes ``**/*.md`` so doc anchors keep working in any
    repo, even one without explicit Markdown files yet.
    """
    seen_globs: set[str] = {"**/*.md"}
    files_checked = 0
    for _dirpath, dirnames, filenames in os.walk(repo):
        # In-place prune skip dirs.
        dirnames[:] = [d for d in dirnames if d not in _DETECT_SKIP_DIRS]
        for name in filenames:
            files_checked += 1
            if files_checked > max_files:
                # Don't burn time on huge repos; what we've seen is enough.
                return _ordered_globs(seen_globs)
            ext = Path(name).suffix.lower()
            glob = _DETECT_EXT_TO_GLOB.get(ext)
            if glob is not None:
                seen_globs.add(glob)
    return _ordered_globs(seen_globs)


def _ordered_globs(globs: set[str]) -> list[str]:
    """Return globs in the canonical order used by `_DEFAULT_CONFIG_YAML`."""
    canonical_order = list(_DETECT_EXT_TO_GLOB.values())
    return [g for g in canonical_order if g in globs]


def _render_config_yaml(include_globs: list[str]) -> str:
    """Render the default config.yaml with *include_globs* substituted in."""
    include_block = "\n".join(f'  - "{g}"' for g in include_globs)
    return _DEFAULT_CONFIG_YAML.replace(
        '  - "**/*.md"\n  - "**/*.py"\n  - "**/*.ts"',
        include_block,
    )


_DEFAULT_CONFIG_YAML = """\
# Generated by `scry init`.  See DESIGN.md §6 for full schema.

include:
  - "**/*.md"
  - "**/*.py"
  - "**/*.ts"
exclude:
  # scry's own state directory.
  - .scry/**
  # Common dependency / build / cache directories.
  - node_modules/**
  - dist/**
  - build/**
  - target/**
  - out/**
  - coverage/**
  - .next/**
  - .nuxt/**
  - .parcel-cache/**
  - .cache/**
  - .pytest_cache/**
  - .mypy_cache/**
  - .ruff_cache/**
  - .venv/**
  - venv/**
  # UT5-3: exclude common diagnostic / scratch script names so a quick
  # debug.py or scratch.py at the repo root doesn't pollute the index.
  # Remove these globs if your project intentionally tracks files with
  # these names.
  - debug*.py
  - scratch*.py
  - tmp*.py
classify:
  - { glob: "docs/**.md", type: spec }
  - { glob: "**/*.md", type: doc }
embeddings:
  provider: local
  model: BAAI/bge-small-en-v1.5
  dimensions: 384
"""

_DEFAULT_GITIGNORE = """\
# Generated by `scry init` — see DESIGN.md §10.
vectors.db
vectors.db.lock
vectors.db-wal
vectors.db-shm
leader.lock
overlays/
cache/
stats.json
commit-links.*.marker
"""

_DEFAULT_GITATTRIBUTES = ".scry/links.jsonl merge=union\n"

_MCP_SNIPPET = """\
{
  "mcpServers": {
    "scry": {
      "command": "scry",
      "args": ["mcp"]
    }
  }
}"""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """Return cwd as the repo root (scry always runs from the repo root)."""
    return Path.cwd()


def _resolve_repo_root(ctx: click.Context) -> Path:
    """Return repo root from context or fall back to cwd."""
    obj = ctx.obj or {}
    root = obj.get("repo_root")
    return root if isinstance(root, Path) else _repo_root()


# SR5-4: helpers for "LSP unavailable" messages.  Both ``scry callers``
# and ``scry subclasses`` need to surface the LSP status for the
# anchor's actual language (not the hardcoded "python") and recommend
# the right binary to install.
_EXT_TO_LANG_FOR_HINT: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
}

_LSP_BINARY_FOR_LANG: dict[str, str] = {
    "python": "pyright-langserver",
    "typescript": "typescript-language-server",
    "tsx": "typescript-language-server",
    "javascript": "typescript-language-server",
    "jsx": "typescript-language-server",
    "go": "gopls",
    "rust": "rust-analyzer",
    "zig": "zls",
}


def _lang_from_anchor_id(anchor_id: str) -> str:
    """Return the language hint for an anchor ID's file path portion.

    Anchor IDs look like ``path/to/file.py:Symbol`` or
    ``path/to/file.py:Class.method`` — split on the first ``:`` to
    isolate the file portion, then map by extension.  Falls back to
    ``"python"`` for unknown extensions to preserve historical behaviour.
    """
    file_part = anchor_id.split(":", 1)[0]
    ext = Path(file_part).suffix.lower()
    return _EXT_TO_LANG_FOR_HINT.get(ext, "python")


def _lsp_binary_for(language: str) -> str:
    """Return the recommended LSP binary name for *language*.

    Used in ``scry callers`` / ``scry subclasses`` "LSP unavailable"
    notes so we don't tell a Go user to install ``pyright-langserver``.
    """
    return _LSP_BINARY_FOR_LANG.get(language, f"the {language} language server")


# UAT-6: concrete install commands per LSP binary.  Day-1 users don't
# know whether pyright-langserver is npm/pip/system or where rust-analyzer
# comes from; the previous "Install pyright-langserver" hint was correct
# but unhelpful.  These commands are well-known canonical installs.
_LSP_INSTALL_COMMANDS: dict[str, str] = {
    "pyright-langserver": "npm install -g pyright    (Node.js required)",
    "typescript-language-server": (
        "npm install -g typescript typescript-language-server    (Node.js required)"
    ),
    "gopls": "go install golang.org/x/tools/gopls@latest    (Go required)",
    "rust-analyzer": "rustup component add rust-analyzer    (Rust toolchain required)",
    "zls": "see https://github.com/zigtools/zls#installation",
}


def _lsp_install_hint(language: str) -> str:
    """Return a multi-line "how to install" hint for *language*'s LSP.

    UAT-6: the historical "Install pyright-langserver" note told users
    WHAT but not HOW.  This returns a concrete install command (with
    runtime dependency in parentheses) for the languages we know.
    Falls back to the binary name alone when we don't have a recipe.
    """
    binary = _lsp_binary_for(language)
    install = _LSP_INSTALL_COMMANDS.get(binary)
    if install is None:
        return f"Install {binary}"
    return f"Install command: {install}"


def _path_excluded(p: Path, repo: Path, exclude_patterns: list[str]) -> bool:
    """Return True if *p* (a directory) matches any exclude glob in the config.

    Used by the validate walker to PRUNE entire subtrees (e.g. ``.venv/``,
    ``node_modules/``, ``tests/fixtures/``) instead of descending into
    them and then filtering each file individually.  Mirrors the
    indexer's behaviour and respects the include/exclude config the
    same way.
    """
    import fnmatch as _fnmatch

    try:
        rel = p.relative_to(repo)
    except ValueError:
        return False
    rel_str = str(rel).replace("\\", "/") + "/"
    for pat in exclude_patterns:
        norm_pat = pat.replace("**", "*")
        if _fnmatch.fnmatchcase(rel_str, norm_pat) or _fnmatch.fnmatchcase(
            rel_str.rstrip("/"), norm_pat.rstrip("/*")
        ):
            return True
        # Also match the leading directory name against patterns like
        # ".venv/**" (without recursing).
        if rel_str.startswith(pat.rstrip("/*").rstrip("/") + "/"):
            return True
    return False


def _make_stub_embedder(config: Any) -> StubEmbedder:
    """Return a StubEmbedder for use when SCRY_EMBEDDER=stub is set."""
    dims = 384
    if hasattr(config, "embeddings") and hasattr(config.embeddings, "dimensions"):
        dims = config.embeddings.dimensions
    return StubEmbedder(dimensions=dims)


def _get_embedder(config: Any) -> Any:
    """Return the configured embedder; use StubEmbedder when SCRY_EMBEDDER=stub."""
    if os.environ.get("SCRY_EMBEDDER") == "stub":
        return _make_stub_embedder(config)
    return make_embedder(config.embeddings)


# ─── Root group ───────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version=scry.__version__, prog_name="scry")
@click.option(
    "--allow-untrusted-lsp-config",
    is_flag=True,
    default=False,
    envvar="SCRY_ALLOW_UNTRUSTED_LSP_CONFIG",
    help=(
        "Permit lsp.command overrides in .scry/config.yaml (§6.2 security override). "
        "Wave 3 will enforce this at LSP spawn time; for Wave 2 it is plumbed through "
        "and printed as a warning by `scry doctor`."
    ),
)
@click.pass_context
def main(ctx: click.Context, allow_untrusted_lsp_config: bool) -> None:
    """scry — local-first MCP server for code↔spec drift detection.

    Run `scry COMMAND --help` for per-command usage.
    """
    ctx.ensure_object(dict)
    ctx.obj["allow_untrusted_lsp_config"] = allow_untrusted_lsp_config


# ─── scry init ────────────────────────────────────────────────────────────────


@main.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing .scry/config.yaml.")
@click.option(
    "--register-global",
    is_flag=True,
    help="Write MCP entry to ~/.claude.json and ~/.cursor/mcp.json.",
)
@click.pass_context
def init(ctx: click.Context, force: bool, register_global: bool) -> None:
    """Wizard: write .scry/config.yaml, .gitignore, and .gitattributes.

    By default, prints the MCP JSON snippet for manual paste.  With
    ``--register-global`` the snippet is also written to
    ~/.claude.json and ~/.cursor/mcp.json.
    """
    repo = _resolve_repo_root(ctx)
    scry_dir = repo / ".scry"
    config_path = scry_dir / "config.yaml"

    scry_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists() and not force:
        click.echo(
            f"error: {config_path} already exists. Use --force to overwrite.",
            err=True,
        )
        raise SystemExit(1) from None

    # UAT-4: walk the repo and pick include patterns based on what's
    # actually present.  Without this, polyglot repos (Go/Rust/JS)
    # silently get a Python+TS+MD-only config and an indexer that
    # silently drops half their codebase.
    detected_globs = _detect_repo_languages(repo)
    config_text = _render_config_yaml(detected_globs)
    config_path.write_text(config_text, encoding="utf-8")
    click.echo(f"Wrote {config_path}")
    if detected_globs != _DEFAULT_INCLUDE_GLOBS:
        added = sorted(set(detected_globs) - set(_DEFAULT_INCLUDE_GLOBS))
        if added:
            click.echo(
                f"  detected and included: {', '.join(added)}",
                err=True,
            )

    gitignore_path = scry_dir / ".gitignore"
    gitignore_path.write_text(_DEFAULT_GITIGNORE, encoding="utf-8")
    click.echo(f"Wrote {gitignore_path}")

    root_gitattributes = repo / ".gitattributes"
    existing = ""
    if root_gitattributes.exists():
        existing = root_gitattributes.read_text(encoding="utf-8")
    union_line = ".scry/links.jsonl merge=union"
    if union_line not in existing:
        with root_gitattributes.open("a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write(union_line + "\n")
        click.echo(f"Updated {root_gitattributes}")

    click.echo("\nMCP registration snippet (paste into your editor config):")
    click.echo(_MCP_SNIPPET)

    if register_global:
        home = Path.home()
        targets = [
            home / ".claude.json",
            home / ".cursor" / "mcp.json",
        ]
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                existing_json: object = {}
                if target.exists():
                    raw = target.read_text(encoding="utf-8")
                    try:
                        existing_json = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        # Refuse to write rather than overwrite a malformed file —
                        # silently truncating ~/.claude.json (Claude Code config
                        # with substantial state) on parse failure was the W2j
                        # HIGH bug from review-w2j.
                        click.echo(
                            f"warning: {target} is not valid JSON ({exc.msg}); "
                            f"refusing to register MCP entry to avoid data loss. "
                            f"Fix the file (or move it aside) and re-run "
                            f"`scry init --register-global`.",
                            err=True,
                        )
                        continue
                # Reject non-dict roots (json may legitimately decode to a
                # list/null/number/string from a hand-written config) — we
                # cannot safely set "mcpServers" on those without
                # destroying user content.
                if not isinstance(existing_json, dict):
                    click.echo(
                        f"warning: {target} top-level is "
                        f"{type(existing_json).__name__!r}, not an object; "
                        f"refusing to register MCP entry. The file should be "
                        f"a JSON object (e.g. `{{}}`).",
                        err=True,
                    )
                    continue
                mcp_servers = existing_json.setdefault("mcpServers", {})
                if not isinstance(mcp_servers, dict):
                    click.echo(
                        f"warning: {target} has a non-object `mcpServers` value; "
                        f"refusing to overwrite.",
                        err=True,
                    )
                    continue
                mcp_servers["scry"] = {"command": "scry", "args": ["mcp"]}
                target.write_text(json.dumps(existing_json, indent=2) + "\n", encoding="utf-8")
                click.echo(f"Registered MCP entry in {target}")
            except OSError as exc:
                click.echo(f"warning: could not write {target}: {exc}", err=True)


# ─── scry index ───────────────────────────────────────────────────────────────


@main.command("index")
@click.option("--force", is_flag=True, help="Nuclear rebuild: drop and recreate everything.")
@click.option(
    "--reembed",
    is_flag=True,
    help="Re-embed existing anchors with the current model (preserves anchors, links).",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress per-file progress output (UAT-1).",
)
@click.pass_context
def index(ctx: click.Context, force: bool, reembed: bool, quiet: bool) -> None:
    """Build or refresh the vector store.

    ``--force`` and ``--reembed`` are mutually exclusive.
    """
    if force and reembed:
        raise click.UsageError("--force and --reembed are mutually exclusive")

    repo = _resolve_repo_root(ctx)
    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    embedder = _get_embedder(config)
    indexer = Indexer(
        repo,
        config=config,
        embedder=embedder,
        allow_untrusted=ctx.obj.get("allow_untrusted_lsp_config", False),
    )

    try:
        # UAT-1 (review-u1): the CLI reports per-phase progress through a
        # single callback the indexer fires during extract / lsp / embed
        # (and reembed for the --reembed path).  --quiet truly suppresses
        # output (the callback is None); non-TTY emits a sparse stderr
        # line every 25 items per phase; TTY uses click.progressbar.
        if quiet:
            cb = None
        elif not sys.stderr.isatty():
            phase_state: dict[str, int] = {}

            def _progress_log(phase: str, processed: int, total: int, label: str) -> None:
                last = phase_state.get(phase, 0)
                if processed - last >= 25 or processed == total or processed == 1:
                    click.echo(
                        f"  {phase} {processed}/{total} ({label})",
                        err=True,
                    )
                    phase_state[phase] = processed

            cb = _progress_log
        else:
            # TTY: one progressbar per phase.  We close+reopen as the
            # phase changes so the ETA stays meaningful.
            bar_state: dict[str, Any] = {"phase": None, "bar": None}

            def _progress_bar(phase: str, processed: int, total: int, label: str) -> None:
                if bar_state["phase"] != phase:
                    if bar_state["bar"] is not None:
                        bar_state["bar"].render_finish()
                    bar_state["bar"] = click.progressbar(
                        length=max(total, 1),
                        label=phase,
                        file=sys.stderr,
                    )
                    bar_state["bar"].render_progress()
                    bar_state["phase"] = phase
                bar_state["bar"].update(1, current_item=label)

            cb = _progress_bar

        try:
            if reembed:
                result: IndexResult = indexer.reembed(progress_callback=cb)
            else:
                result = indexer.index(force=force, progress_callback=cb)
        finally:
            # Make sure any outstanding TTY bar is closed cleanly.
            if not quiet and sys.stderr.isatty():
                final_bar = bar_state.get("bar") if "bar_state" in dir() else None
                if final_bar is not None:
                    final_bar.render_finish()
    except (IndexerError, LockTimeout) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    except sqlite3.OperationalError as exc:
        # SR3-3: previously surfaced as a Python traceback when
        # vectors.db was read-only or otherwise un-writable.
        click.echo(f"error: cannot write to index: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo(
        f"Indexed: files_processed={result.files_processed} "
        f"anchors_extracted={result.anchors_extracted} "
        f"anchors_embedded={result.anchors_embedded} "
        f"chunks_written={result.chunks_written} "
        f"files_pruned={result.files_pruned} "
        f"elapsed={result.elapsed_seconds:.2f}s"
    )


# ─── scry watch ───────────────────────────────────────────────────────────────


@main.command("watch")
@click.option(
    "--debounce-ms",
    default=500,
    type=int,
    show_default=True,
    help="Milliseconds of quiet after the last file event before triggering reindex.",
)
@click.option(
    "--once",
    is_flag=True,
    help="Run a single reindex pass then exit (useful for scripting/testing).",
)
@click.option(
    "--reconnect-timeout",
    default=60,
    type=int,
    show_default=True,
    help="Seconds to keep retrying IPC before giving up when the leader disappears mid-watch.",
)
@click.pass_context
def watch(ctx: click.Context, debounce_ms: int, once: bool, reconnect_timeout: int) -> None:
    """Watch for file changes and reindex incrementally.

    Requires a running leader process (``scry mcp``).  ``scry watch`` acts as
    a follower: file events are forwarded to the leader via IPC so the leader
    owns the single write path.  Exits with code 2 if no leader is detected.

    Press Ctrl+C to stop.
    """
    from scry.cmd_watch import WatchError, run_watch

    repo = _resolve_repo_root(ctx)
    try:
        exit_code = asyncio.run(
            run_watch(repo, debounce_ms=debounce_ms, once=once, reconnect_timeout=reconnect_timeout)
        )
    except WatchError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        pass  # clean Ctrl+C exit
    else:
        if exit_code != 0:
            raise SystemExit(exit_code)


# ─── scry check ───────────────────────────────────────────────────────────────


@main.command("check")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "md"]),
    default="md",
    show_default=True,
    help="Output format (json|md).",
)
@click.option(
    "--json",
    "json_flag",
    is_flag=True,
    help="Shorthand for --format json (machine-readable output).",
)
@click.option(
    "--ci",
    is_flag=True,
    help="Exit 1 when thresholds are violated; exit 2 on error (§5.2 v3.1).",
)
@click.option(
    "--drift-min",
    type=float,
    default=None,
    help="Minimum drift_score for --ci pass (null score = pass per §5.2 v3.1).",
)
@click.option(
    "--coverage-min",
    type=float,
    default=None,
    help="Minimum coverage_score for --ci pass (null score = pass per §5.2 v3.1).",
)
@click.option(
    "--require-fresh-embedder",
    is_flag=True,
    help="Error (exit 2) if the stored embedding model differs from config.",
)
@click.option(
    "--ignore-lsp-error",
    is_flag=True,
    help=(
        "Exclude drift-unknown (caused by lsp_error) from the failure set. "
        "Links still appear in output counts. "
        "Overrides §5.1 default of failing on drift-unknown."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Any non-fresh link is a failure (exit 1). "
        "With --ignore-lsp-error, drift-unknown is excluded from the failure set."
    ),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help=(
        "List the actual drifted links (link_id, status, from, to) "
        "in addition to the summary counts (UT1-2)."
    ),
)
@click.option(
    "--uncovered",
    is_flag=True,
    help=(
        "List the top spec sections (markdown anchors) with no incoming "
        "or outgoing active links, so spec authors can prioritise "
        "linking work (UAT-18)."
    ),
)
@click.pass_context
def check(
    ctx: click.Context,
    fmt: str,
    json_flag: bool,
    ci: bool,
    drift_min: float | None,
    coverage_min: float | None,
    require_fresh_embedder: bool,
    ignore_lsp_error: bool,
    strict: bool,
    verbose: bool,
    uncovered: bool,
) -> None:
    """Drift + coverage scores (§5.2 v3.1).

    Evaluates all active links for drift and reports scores.

    Exit codes: 0 = clean, 1 = drift detected, 2 = operational error.

    With ``--ci`` exits 1 when ``drift_score < drift_min`` OR
    ``coverage_score < coverage_min``.  A ``null`` score is always a PASS.

    With ``--strict`` any non-fresh link causes exit 1.

    With ``--ignore-lsp-error`` links with ``drift-unknown`` status are
    excluded from the failure set but still appear in the output counts
    (§5.1 override for broken LSPs).
    """
    if json_flag:
        fmt = "json"

    repo = _resolve_repo_root(ctx)
    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(2) from None

    # UAT-7: warn when on-disk files are newer than the indexed manifest so
    # users can't be misled by a "fresh" drift report on stale data.
    # Optimization (review-u4-u6 MEDIUM): use mtime+size as a cheap pre-filter.
    # Only files whose mtime/size doesn't match the cached "fingerprint" pay
    # the full content-hash cost.  We cache mtime_ns and size in
    # index_metadata.indexed_file_manifest_stats — when absent (older index
    # files) we fall back to hashing every file.
    fs_stale_warning: str | None = None
    try:
        with ScryDB(repo, read_only=True) as _stale_check_db:
            _stale_meta = _stale_check_db.read_index_metadata()
        if _stale_meta is not None:
            from scry.index import Indexer as _Indexer
            from scry.index import _file_content_hash as _file_hash

            _idx = _Indexer(
                repo,
                config=config,
                embedder=None,  # not used for discover_files
                allow_untrusted=True,
            )
            _stale_count = 0
            try:
                _current_files = _idx.discover_files()
            except Exception:  # pragma: no cover — be defensive
                _current_files = []
            _current_rel = {p.relative_to(repo).as_posix(): p for p in _current_files}
            _manifest = _stale_meta.indexed_file_manifest
            for rel, abspath in _current_rel.items():
                manifest_hash = _manifest.get(rel)
                if manifest_hash is None:
                    _stale_count += 1
                    continue
                # mtime+size pre-filter: if both match the indexed values
                # (when stored), skip the expensive hash.  We don't have
                # stored stats yet in this implementation, so we always
                # hash; this is still O(reads) per check.  TODO: persist
                # mtime+size in IndexMetadata and use that as the fast path.
                try:
                    if _file_hash(abspath) != manifest_hash:
                        _stale_count += 1
                except OSError:
                    continue
            _missing = sum(1 for rel in _manifest if rel not in _current_rel)
            if _stale_count or _missing:
                fs_stale_warning = (
                    f"WARNING: {_stale_count} file(s) changed and "
                    f"{_missing} file(s) deleted since the last `scry index`. "
                    "Drift results below reflect the indexed snapshot, NOT "
                    "your current working tree. Run `scry index` to refresh."
                )
    except (LockTimeout, OSError):
        # Couldn't read metadata — proceed without the stale warning.
        pass

    try:
        with ScryDB(repo, read_only=True) as db:
            if require_fresh_embedder:
                meta = db.read_index_metadata()
                if meta is not None:
                    cfg_model = config.embeddings.model
                    cfg_provider = config.embeddings.provider
                    if meta.embedding_model != cfg_model or meta.embedding_provider != cfg_provider:
                        click.echo(
                            f"error: embedding model mismatch. "
                            f"Index was built with {meta.embedding_provider}/{meta.embedding_model}; "
                            f"config says {cfg_provider}/{cfg_model}. "
                            "Run `scry index --reembed` to migrate.",
                            err=True,
                        )
                        raise SystemExit(2) from None

            link_store = LinkStore(repo)
            git_ctx = GitContextProvider(repo)
            overlay_mgr = OverlayManager(repo, git_context=git_ctx, link_store=link_store)
            overlay_path = overlay_mgr.current_overlay_path()

            # Use a link store that replays baseline ⊕ current overlay.
            class _BranchLinkStore(LinkStore):
                """LinkStore that uses the current branch overlay in evaluate_all_drift."""

                def replay(self, *, overlay_path: Path | None = None) -> Any:
                    return super().replay(overlay_path=overlay_path or _ov_path)

            _ov_path = overlay_path
            branch_store = _BranchLinkStore(repo)

            evaluations = evaluate_all_drift(db=db, link_store=branch_store, config=config.drift)

            # Count code anchors for coverage.
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)
            total_code = len(code_anchors)
            all_links = branch_store.replay(overlay_path=overlay_path)
            linked_code_ids = {
                lnk.from_id
                for lnk in all_links.active_links.values()
                if lnk.from_type == AnchorType.CODE
            } | {
                lnk.to_id
                for lnk in all_links.active_links.values()
                if lnk.to_type == AnchorType.CODE
            }
            linked_code = len(linked_code_ids & {a.id for a in code_anchors})
            # Coverage is null (PASS in CI per §5.2 v3.1) when there are no
            # active links yet — the metric is not applicable until linking begins.
            coverage_total: int | None = total_code if all_links.active_links else None

            summary = compute_drift_summary(
                evaluations,
                config=config.drift,
                coverage_total_code_anchors=coverage_total,
                coverage_linked_code_anchors=linked_code,
            )
    except (GitContextError, LockTimeout, OSError) as exc:
        # SR3-1: GitContextError previously propagated as a stack trace
        # in repos with no commits.  Catch it here so users get a clean
        # ``error: git unavailable: ...`` message and exit 2.
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    c = summary.counts

    if fmt == "json":
        # UT1-2: include drifted-link details in JSON output when --verbose
        # is set so machine-readable consumers can act on the report.
        payload = json.loads(summary.model_dump_json())
        if fs_stale_warning is not None:
            payload["fs_stale_warning"] = fs_stale_warning
        if verbose:
            payload["drifted_links"] = [
                {
                    "link_id": ev.link.link_id,
                    "drift_status": str(ev.drift_status),
                    "from_id": ev.link.from_id,
                    "to_id": ev.link.to_id,
                    "type": str(ev.link.type),
                }
                for ev in evaluations
                if ev.drift_status != DriftStatus.FRESH
            ]
        click.echo(json.dumps(payload, indent=2))
    else:
        # Markdown summary.
        if fs_stale_warning is not None:
            click.echo(f"> ⚠ {fs_stale_warning}\n", err=True)
        ds = f"{summary.drift_score:.1f}" if summary.drift_score is not None else "null"
        cs = f"{summary.coverage_score:.1f}" if summary.coverage_score is not None else "null"
        click.echo("## scry check\n")
        click.echo("| Metric | Score |")
        click.echo("|--------|-------|")
        click.echo(f"| drift_score | {ds} |")
        click.echo(f"| coverage_score | {cs} |")
        click.echo(f"| drift_coverage | {summary.drift_coverage} |")
        # UT1-8: explain drift_coverage so first-time users know what
        # "section-only" means (no LSP active → only spec/doc anchor
        # hashes can drift; transitive code closures unavailable).
        if summary.drift_coverage == "section-only":
            click.echo(
                "\n_drift_coverage = 'section-only': no LSP transitive "
                "closures available; only spec/doc/code endpoint hash "
                "changes are detected.  Configure code_anchors.languages "
                "in .scry/config.yaml to enable transitive code drift._"
            )
        click.echo("\n### Counts\n")
        click.echo("| Status | Count |")
        click.echo("|--------|-------|")
        click.echo(f"| fresh | {c.fresh} |")
        click.echo(f"| spec_changed | {c.spec_changed} |")
        click.echo(f"| code_changed | {c.code_changed} |")
        click.echo(f"| both_changed | {c.both_changed} |")
        click.echo(f"| broken_source | {c.broken_source} |")
        click.echo(f"| broken_target | {c.broken_target} |")
        click.echo(f"| merge_conflict | {c.merge_conflict} |")
        click.echo(f"| drift_unknown | {c.drift_unknown} |")
        click.echo(f"| semantic_drift_flagged | {c.semantic_drift_flagged} |")
        click.echo(f"| total | {c.total} |")

        # UT1-2: --verbose lists each drifted link's id + status + endpoints
        # so users can act on the report without parsing JSONL or querying
        # the MCP server.
        if verbose:
            non_fresh_evals = [ev for ev in evaluations if ev.drift_status != DriftStatus.FRESH]
            if non_fresh_evals:
                click.echo("\n### Drifted links\n")
                click.echo("| link_id | status | from -> to |")
                click.echo("|---------|--------|------------|")
                for ev in non_fresh_evals:
                    lk = ev.link
                    click.echo(f"| {lk.link_id} | {ev.drift_status} | {lk.from_id} -> {lk.to_id} |")
            else:
                click.echo("\n_(verbose) all links fresh._")

        # UAT-18: --uncovered lists spec section anchors with no incoming
        # OR outgoing active link, sorted by content length descending so
        # spec authors prioritise the largest unlinked sections first.
        if uncovered:
            try:
                with ScryDB(repo, read_only=True) as _ucdb:
                    section_anchors = _ucdb.list_anchors(anchor_type=AnchorType.SECTION)
            except (LockTimeout, OSError):
                section_anchors = []
            linked_ids = {lnk.from_id for lnk in all_links.active_links.values()} | {
                lnk.to_id for lnk in all_links.active_links.values()
            }
            uncovered_secs = [a for a in section_anchors if a.id not in linked_ids]
            uncovered_secs.sort(key=lambda a: len(a.content_text), reverse=True)
            click.echo(f"\n### Uncovered spec sections ({len(uncovered_secs)} total)\n")
            if not uncovered_secs:
                click.echo("_All section anchors have at least one active link._")
            else:
                click.echo("| anchor_id | size_chars |")
                click.echo("|-----------|-----------|")
                for a in uncovered_secs[:25]:
                    click.echo(f"| {a.id} | {len(a.content_text)} |")
                if len(uncovered_secs) > 25:
                    click.echo(f"\n_... showing top 25 of {len(uncovered_secs)} uncovered._")

        # UAT-17: surface the cross-language semantic-drift hint when one
        # or more mirrors links have semantic_drift=None AND the user
        # hasn't configured drift.cross_language_threshold.  Without this
        # the relevant warning only appears in the logger output (which
        # the CLI suppresses by default).
        # UAT-17 / review-u11-u13 MEDIUM: tighten the predicate to only
        # fire on TRUE cross-language pairs.  semantic_drift=None can also
        # mean missing embeddings; we don't want to suggest a config
        # change for the wrong reason.
        from scry.drift import _infer_language as _drift_infer_language

        def _is_cross_lang_unconfigured(ev: object) -> bool:
            from scry.drift import DriftEvaluation as _DE

            if not isinstance(ev, _DE):
                return False
            if ev.link.type != LinkType.MIRRORS:
                return False
            if ev.semantic_drift is not None:
                return False
            if ev.link.from_type != AnchorType.CODE or ev.link.to_type != AnchorType.CODE:
                return False
            from_lang = _drift_infer_language(ev.link.from_id.split(":", 1)[0])
            to_lang = _drift_infer_language(ev.link.to_id.split(":", 1)[0])
            return from_lang is not None and to_lang is not None and from_lang != to_lang

        cross_lang_unconfigured = config.drift.cross_language_threshold is None and any(
            _is_cross_lang_unconfigured(ev) for ev in evaluations
        )
        if cross_lang_unconfigured:
            click.echo(
                "\n_⚠ One or more cross-language `mirrors` links have "
                "`semantic_drift=null` because `drift.cross_language_threshold` "
                "is unset.  Add it to .scry/config.yaml to enable cross-language "
                "semantic-drift checking._"
            )

    # --strict: any non-fresh link is a failure (exit 1).
    # With --ignore-lsp-error, drift-unknown is excluded from the failure set.
    if strict:
        non_fresh = (
            c.broken_source
            + c.broken_target
            + c.merge_conflict
            + c.both_changed
            + c.spec_changed
            + c.code_changed
            + (0 if ignore_lsp_error else c.drift_unknown)
        )
        if non_fresh > 0:
            click.echo(
                f"FAIL (strict): {non_fresh} non-fresh link(s) detected",
                err=True,
            )
            raise SystemExit(1) from None

    # --ci: threshold-based gate (§5.2 v3.1).
    # With --ignore-lsp-error, drift_unknown links are excluded from the effective
    # drift_score used for threshold comparisons.
    if ci:
        # Compute effective_drift_score: when --ignore-lsp-error is active and there are
        # drift_unknown links, back out their weight from the score so they don't push
        # the score below the threshold.
        effective_drift_score = summary.drift_score
        if ignore_lsp_error and c.drift_unknown > 0 and summary.drift_score is not None:
            scoring = config.drift.scoring
            adj_total = c.total - c.drift_unknown
            if adj_total == 0:
                effective_drift_score = None  # all links were drift_unknown → null = pass
            else:
                # Reverse the formula: weighted_sum = (1 - score/100) * total
                ws = (1.0 - summary.drift_score / 100.0) * c.total
                adj_ws = ws - scoring.drift_unknown * c.drift_unknown
                effective_drift_score = max(0.0, min(100.0, 100.0 * (1.0 - adj_ws / adj_total)))

        failed = False
        if (
            drift_min is not None
            and effective_drift_score is not None
            and effective_drift_score < drift_min
        ):
            click.echo(
                f"CI FAIL: drift_score {summary.drift_score:.1f} < {drift_min}",
                err=True,
            )
            failed = True
        if (
            coverage_min is not None
            and summary.coverage_score is not None
            and summary.coverage_score < coverage_min
        ):
            click.echo(
                f"CI FAIL: coverage_score {summary.coverage_score:.1f} < {coverage_min}",
                err=True,
            )
            failed = True
        if failed:
            raise SystemExit(1) from None


# ─── scry status ──────────────────────────────────────────────────────────────


@main.command("status")
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show pending overlay records, merge conflicts, and index state."""
    repo = _resolve_repo_root(ctx)

    try:
        git_ctx_prov = GitContextProvider(repo)
        git_ctx = git_ctx_prov.get()
    except GitContextError as exc:
        click.echo(f"error: git unavailable: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo(f"Branch:     {git_ctx.branch or '(detached)'}")
    click.echo(f"HEAD:       {git_ctx.head_short}")
    click.echo(f"Dirty files: {len(git_ctx.dirty_files)}")

    overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
    try:
        pending = overlay_mgr.list_pending_overlay_records()
        replay = overlay_mgr.replay_active()
    except (GitContextError, Exception) as exc:
        click.echo(f"warning: could not read overlay: {exc}", err=True)
        pending = []
        replay = None

    click.echo(f"\nPending overlay records: {len(pending)}")
    for rec in pending:
        click.echo(f"  {rec.event_id}  op={rec.op}  link_id={rec.link_id}")

    conflicts = replay.merge_conflicts if replay else []
    click.echo(f"\nMerge conflicts: {len(conflicts)}")
    for lnk_id in conflicts:
        click.echo(f"  {lnk_id}")

    db_path = repo / ".scry" / "vectors.db"
    if db_path.exists():
        try:
            with ScryDB(repo, read_only=True) as db:
                meta = db.read_index_metadata()
                if meta:
                    try:
                        config = load_config(repo)
                    except Exception:
                        from scry.models import Config

                        config = Config()
                    is_stale = check_staleness(git_ctx, meta, config)
                    click.echo("\nIndex:")
                    click.echo(f"  indexed_branch: {meta.indexed_branch}")
                    click.echo(f"  indexed_head:   {meta.indexed_git_head[:12]}")
                    click.echo(
                        f"  model:          {meta.embedding_provider}/{meta.embedding_model}"
                    )
                    # MEDIUM #2: print full enum-like state, not just stale/fresh.
                    if not is_stale:
                        state_label = IndexState.FRESH.value
                    else:
                        # Best-effort derivation without an in-memory tracker.
                        # If no leader is running, no one can reconcile.
                        leader_st, _ = detect_leader_state(repo)
                        if leader_st == LeaderState.FOLLOWER:
                            state_label = "stale"
                        else:
                            state_label = IndexState.STALE_NO_WRITE_LOCK.value
                    click.echo(f"  index_state:    {state_label}")
                    # UT2-7: clarify what stale-no-write-lock means for
                    # solo CLI users.  The state_label name suggests
                    # something is broken, but it just means "no leader
                    # process is running, so no one can auto-reconcile;
                    # run scry index when you want to rebuild".
                    if state_label == IndexState.STALE_NO_WRITE_LOCK.value:
                        click.echo(
                            "  note:           index is stale and no leader is running. "
                            "Run `scry index` to refresh, or start `scry mcp` "
                            "for auto-reconcile."
                        )
                    elif state_label == "stale":
                        click.echo(
                            "  note:           index is stale; the running leader "
                            "will auto-reconcile shortly."
                        )
                else:
                    click.echo("\nIndex: not yet built")
        except Exception as exc:
            click.echo(f"\nIndex: unreadable ({exc})")
    else:
        click.echo("\nIndex: vectors.db not found — run `scry index`")


# ─── scry commit-links ────────────────────────────────────────────────────────


@main.command("commit-links")
@click.argument("event_ids", nargs=-1)
@click.pass_context
def commit_links(ctx: click.Context, event_ids: tuple[str, ...]) -> None:
    """Promote overlay records to baseline links.jsonl.

    With no arguments promotes ALL pending records on the current branch.
    Accepts one or more EVENT_ID values to promote selectively.
    """
    repo = _resolve_repo_root(ctx)
    try:
        overlay_mgr = OverlayManager(repo)
    except GitContextError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    ids_to_promote = list(event_ids) if event_ids else None
    try:
        promoted = overlay_mgr.promote_pending(event_ids=ids_to_promote)
    except (LinkValidationError, MergeConflictError, GitContextError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    if not promoted:
        click.echo("No overlay records to promote.")
        return

    click.echo(f"Promoted {len(promoted)} record(s) to baseline:")
    for eid in promoted:
        click.echo(f"  {eid}")


# ─── scry search ──────────────────────────────────────────────────────────────


@main.command("search")
@click.argument("query")
@click.option("--top-k", default=10, show_default=True, help="Maximum results to return.")
@click.option(
    "--type",
    "anchor_type_filter",
    type=click.Choice(["section", "code", "code_in_doc"]),
    default=None,
    help="Restrict results to this anchor type.",
)
@click.option(
    "--scope",
    default=None,
    help=(
        "Glob pattern restricting results to matching paths "
        "(e.g. ``--scope src/scry/*.py``).  Applied AFTER hybrid retrieval; "
        "increases top-k internally to compensate for the post-filter."
    ),
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    top_k: int,
    anchor_type_filter: str | None,
    scope: str | None,
) -> None:
    """Hybrid BM25 + vector search over the indexed repository."""
    repo = _resolve_repo_root(ctx)
    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None

    anchor_types = None
    if anchor_type_filter:
        anchor_types = [AnchorType(anchor_type_filter)]

    embedder = _get_embedder(config)

    # UAT-15: --scope post-filters by path glob.  Pushed into hybrid_search
    # via path_globs= so the scope filter applies BEFORE the final top_k
    # truncation (review-u9-u10 HIGH: post-filter alone could miss results
    # ranked just outside the global cutoff).

    try:
        with ScryDB(repo, read_only=True) as db:
            results = hybrid_search(
                query,
                db=db,
                embedder=embedder,
                config=config.retrieval,
                top_k=top_k,
                anchor_types=anchor_types,
                path_globs=[scope] if scope else None,
            )
            packets = [build_anchor_packet(r, db=db, config=config.retrieval) for r in results]
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    if not packets:
        click.echo("No results." if not scope else f"No results matched scope {scope!r}.")
        return

    # UT1-9: scores from sqlite-vec + BM25 RRF are arbitrary-scale
    # (typical range 0.005 - 0.05 for a small corpus).  Print a one-line
    # legend so first-time users don't think 0.03 is bad.
    click.echo(
        f"# Top {len(packets)} result(s)  "
        f"(scores are RRF-fused; relative within this query, not 0-1).\n"
    )
    for pkt in packets:
        excerpt = (pkt.evidence_excerpt or "")[:120].replace("\n", " ")
        click.echo(f"{pkt.anchor.id}  score={pkt.score:.4f}")
        if excerpt:
            click.echo(f"  {excerpt}")


# ─── scry anchors list ────────────────────────────────────────────────────────


@main.command("anchors")
@click.argument("subcommand", type=click.Choice(["list"]))
@click.option(
    "--scope",
    default=None,
    help="Glob restricting which paths to enumerate (e.g. ``--scope src/**/*.py``).",
)
@click.option(
    "--type",
    "anchor_type_filter",
    type=click.Choice(["section", "code", "code_in_doc"]),
    default=None,
    help="Filter by anchor type.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of one-per-line text.")
@click.option(
    "--limit",
    default=500,
    show_default=True,
    type=int,
    help="Maximum anchors to print.",
)
@click.pass_context
def anchors_cmd(
    ctx: click.Context,
    subcommand: str,
    scope: str | None,
    anchor_type_filter: str | None,
    as_json: bool,
    limit: int,
) -> None:
    """Anchor browse commands (UAT-16).

    Today only ``scry anchors list`` is implemented — print every
    indexed anchor's primary ID with optional path-glob and type
    filters.  Solves UAT6's "no anchor browser" complaint that made
    cross-language link authoring painful.
    """
    if subcommand != "list":  # pragma: no cover — Click already restricts
        raise click.UsageError(f"unknown subcommand: {subcommand}")

    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None

    anchor_type = AnchorType(anchor_type_filter) if anchor_type_filter else None

    try:
        with ScryDB(repo, read_only=True) as db:
            all_anchors = db.list_anchors(anchor_type=anchor_type)
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    if scope:
        from scry.config import matches_globs as _matches

        all_anchors = [a for a in all_anchors if _matches(a.path, [scope])]

    truncated = len(all_anchors) > limit
    all_anchors = all_anchors[:limit]

    if as_json:
        click.echo(
            json.dumps(
                {
                    "anchors": [
                        {"id": a.id, "type": a.type, "path": a.path, "symbol_name": a.symbol_name}
                        for a in all_anchors
                    ],
                    "truncated": truncated,
                },
                indent=2,
            )
        )
        return

    for anchor in all_anchors:
        click.echo(f"{anchor.id}\t{anchor.type}\t{anchor.symbol_name or ''}")
    if truncated:
        click.echo(
            f"\n... truncated at --limit {limit}.  Re-run with a higher limit to see more.",
            err=True,
        )


# ─── scry get-anchor / scry get-link / scry get-links / scry show ────────────


@main.command("get-anchor")
@click.argument("anchor_id")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON instead of formatted text.")
@click.pass_context
def get_anchor_cmd(ctx: click.Context, anchor_id: str, as_json: bool) -> None:
    """Print a single anchor's full record by primary ID (UAT-3).

    DESIGN.md §8 promises CLI mirrors MCP 1:1.  Spec authors and
    reviewers reach for this immediately after `scry search` returns
    candidate IDs; previously it was MCP-only.
    """
    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None
    try:
        with ScryDB(repo, read_only=True) as db:
            anchor = db.get_anchor(anchor_id)
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None
    if anchor is None:
        click.echo(f"error: anchor not found: {anchor_id}", err=True)
        raise SystemExit(1) from None
    if as_json:
        click.echo(anchor.model_dump_json(indent=2))
        return
    click.echo(f"id:           {anchor.id}")
    click.echo(f"type:         {anchor.type}")
    click.echo(f"path:         {anchor.path}")
    if anchor.symbol_name:
        click.echo(f"symbol_name:  {anchor.symbol_name}")
    if anchor.heading_path:
        click.echo(f"heading_path: {' / '.join(anchor.heading_path)}")
    click.echo(f"content_hash: {anchor.content_hash}")
    if anchor.transitive_hash_status:
        click.echo(f"lsp_status:   {anchor.transitive_hash_status}")
    click.echo(f"\n--- content ({len(anchor.content_text)} chars) ---")
    click.echo(anchor.content_text)


@main.command("show")
@click.argument("anchor_id")
@click.pass_context
def show_cmd(ctx: click.Context, anchor_id: str) -> None:
    """Print just the content_text of an anchor (UAT-5).

    Equivalent to ``scry get-anchor`` minus the metadata header — the
    "self-contained read the source" command UAT3 #1 was missing.
    """
    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None
    try:
        with ScryDB(repo, read_only=True) as db:
            anchor = db.get_anchor(anchor_id)
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None
    if anchor is None:
        click.echo(f"error: anchor not found: {anchor_id}", err=True)
        raise SystemExit(1) from None
    # UAT-5 review-u8 MEDIUM: write content_text verbatim — no Click-added
    # trailing newline — so `scry show <id> > file` produces exactly the
    # canonicalised content (which preserves its own trailing-newline state).
    sys.stdout.write(anchor.content_text)
    sys.stdout.flush()


@main.command("get-link")
@click.argument("link_id")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON.")
@click.pass_context
def get_link_cmd(ctx: click.Context, link_id: str, as_json: bool) -> None:
    """Print a single link record by link_id (UAT-3 / UAT4 #2).

    Searches the active link table (baseline ⊕ current branch overlay)
    for the requested link.  Returns exit 1 if the link doesn't exist
    (or has been tombstoned).
    """
    from scry.store.links import LinkValidationError, MergeConflictError

    repo = _resolve_repo_root(ctx)
    try:
        link_store = LinkStore(repo)
        git_ctx = GitContextProvider(repo)
        overlay_mgr = OverlayManager(repo, git_context=git_ctx, link_store=link_store)
        replay = link_store.replay(overlay_path=overlay_mgr.current_overlay_path())
    except (GitContextError, LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None
    except (LinkValidationError, MergeConflictError) as exc:
        # UAT-3 review-u8 HIGH: clean error if the link table is mid-write
        # by another process (or has unresolved merge conflicts) rather
        # than a stack trace.
        click.echo(f"error: link table inconsistent: {exc}", err=True)
        raise SystemExit(2) from None
    link = replay.active_links.get(link_id)
    if link is None:
        click.echo(f"error: link not found in active table: {link_id}", err=True)
        raise SystemExit(1) from None
    if as_json:
        click.echo(link.model_dump_json(indent=2))
        return
    click.echo(f"link_id:           {link.link_id}")
    click.echo(f"from:              {link.from_id}  ({link.from_type})")
    click.echo(f"to:                {link.to_id}  ({link.to_type})")
    click.echo(f"type:              {link.type}")
    click.echo(f"from_content_hash: {link.from_content_hash}")
    click.echo(f"to_content_hash:   {link.to_content_hash}")
    if link.evidence:
        click.echo(f"evidence:          {link.evidence}")


# ─── scry link ────────────────────────────────────────────────────────────────


@main.command("link")
@click.argument("from_id")
@click.argument("to_id")
@click.option(
    "--type",
    "link_type",
    required=True,
    type=click.Choice([lt.value for lt in LinkType]),
    help="Link type (e.g. implements, tests, mirrors).",
)
@click.option("--evidence", default=None, help="Optional evidence / rationale string.")
@click.pass_context
def link(
    ctx: click.Context, from_id: str, to_id: str, link_type: str, evidence: str | None
) -> None:
    """Author a link from the CLI (writes to the current branch overlay).

    FROM_ID and TO_ID are anchor primary IDs.
    """
    repo = _resolve_repo_root(ctx)
    try:
        load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None

    try:
        with ScryDB(repo, read_only=True) as db:
            from_anchor = db.get_anchor(from_id)
            if from_anchor is None:
                click.echo(f"error: anchor not found: {from_id!r}", err=True)
                raise SystemExit(1) from None
            to_anchor = db.get_anchor(to_id)
            if to_anchor is None:
                click.echo(f"error: anchor not found: {to_id!r}", err=True)
                raise SystemExit(1) from None

            from_type = AnchorType(from_anchor.type)
            to_type = AnchorType(to_anchor.type)
            from_hash = from_anchor.content_hash
            to_hash = to_anchor.content_hash
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    try:
        git_ctx_prov = GitContextProvider(repo)
        git_ctx = git_ctx_prov.get()
    except GitContextError as exc:
        click.echo(f"error: git context unavailable: {exc}", err=True)
        raise SystemExit(1) from None

    overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)

    # UT2-6: detect duplicate-pair links so re-running ``scry link A B
    # --type implements`` doesn't silently create two active link_ids
    # for the same (from, to, type) triple.  When a duplicate is found,
    # we REFRESH the existing link by reusing its link_id (writes a new
    # event_id with updated content hashes — equivalent to a "rebaseline"
    # for that one link).
    try:
        link_store = LinkStore(repo)
        replay = link_store.replay(overlay_path=overlay_mgr.current_overlay_path())
    except (GitContextError, OSError):
        replay = None

    existing_id: str | None = None
    existing_supersedes: str | None = None
    if replay is not None:
        for active_link in replay.active_links.values():
            if (
                active_link.from_id == from_id
                and active_link.to_id == to_id
                and active_link.type == link_type
            ):
                existing_id = active_link.link_id
                # SR1-1: refresh path requires ``supersedes`` per §3.5.2 rule 5
                # (any upsert whose link_id already exists in baseline ⊕ overlay
                # must reference the prior event_id).  Without this the second
                # invocation fails with "requires supersedes".
                existing_supersedes = active_link.last_event_id
                break

    if existing_id is not None:
        click.echo(
            f"warning: an active link of type {link_type!r} already exists "
            f"for ({from_id} -> {to_id}); refreshing existing link_id={existing_id} "
            f"(new event_id) instead of creating a duplicate.",
            err=True,
        )
        lnk_id = existing_id
    else:
        lnk_id = new_link_id()
    evt_id = new_event_id()
    record_payload: dict[str, object] = {
        "op": LinkOp.UPSERT,
        "link_id": lnk_id,
        "event_id": evt_id,
        "from": from_id,
        "from_type": from_type,
        "to": to_id,
        "to_type": to_type,
        "type": LinkType(link_type),
        "from_content_hash": from_hash,
        "to_content_hash": to_hash,
        "commit_sha": git_ctx.head_sha,
        "worktree_dirty": bool(git_ctx.dirty_files),
        "evidence": evidence,
    }
    if existing_supersedes is not None:
        record_payload["supersedes"] = existing_supersedes
    record = LinkRecord.model_validate(record_payload)
    try:
        overlay_mgr.append_to_current_branch_overlay(record)
    except (LinkValidationError, GitContextError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo(f"Link created: {lnk_id}")
    click.echo(f"  event_id: {evt_id}")
    click.echo(f"  {from_id} --[{link_type}]--> {to_id}")


# ─── scry suggest-links ───────────────────────────────────────────────────────


@main.command("suggest-links")
@click.option("--scope", default=None, help="Restrict suggestions to this path prefix.")
@click.option(
    "--accept-all",
    is_flag=True,
    help="Accept all suggestions without confirmation (implies --apply --yes).",
)
@click.option("--limit", default=None, type=int, help="Maximum candidate pairs to evaluate.")
@click.option(
    "--min-confidence",
    default=None,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum confidence threshold (default 0.7).",
)
@click.option("--apply", is_flag=True, help="Write accepted suggestions to the overlay.")
@click.option("--yes", is_flag=True, help="Skip confirmation when --apply is set.")
@click.option("--json", "as_json", is_flag=True, help="Output suggestions as JSON.")
@click.option(
    "--source",
    type=click.Choice(["code", "doc", "both"]),
    default="both",
    show_default=True,
    help="Which anchor type to scan for unlinked neighbors.",
)
@click.pass_context
def suggest_links(
    ctx: click.Context,
    scope: str | None,
    accept_all: bool,
    limit: int | None,
    min_confidence: float | None,
    apply: bool,
    yes: bool,
    as_json: bool,
    source: str,
) -> None:
    """AI-augmented batch link suggestions (requires LLM provider).

    Scans for code\u2194doc anchor pairs that are semantically related but have no
    existing link, then proposes link types via the configured LLM.

    Use --apply (or --accept-all) to write accepted suggestions to the current
    branch overlay.  Re-running on unchanged state produces no new suggestions
    (idempotent: already-linked pairs are always excluded).
    """
    import asyncio as _asyncio
    from typing import Literal
    from typing import cast as _cast

    from scry.llm import LLMError, make_provider
    from scry.store.links import LinkStore as _LinkStore
    from scry.suggest import (
        DEFAULT_MIN_CONFIDENCE,
        LinkSuggestion,
        SuggestConfig,
        run_suggest_links,
    )

    # --accept-all is shorthand for --apply --yes.
    if accept_all:
        apply = True
        yes = True

    effective_confidence = min_confidence if min_confidence is not None else DEFAULT_MIN_CONFIDENCE
    source_typed = _cast(Literal["code", "doc", "both"], source)

    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(1) from None

    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    try:
        provider = make_provider(config.llm)
    except LLMError as exc:
        click.echo(f"error: LLM provider unavailable: {exc}", err=True)
        raise SystemExit(1) from None

    embedder = _get_embedder(config)
    suggest_config = SuggestConfig(
        min_confidence=effective_confidence,
        limit=limit,
        source=source_typed,
        scope=scope,
    )

    try:
        with ScryDB(repo, read_only=True) as db:
            # Build an overlay-aware LinkStore wrapper so that
            # ``replay()`` returns the baseline ⊕ current branch overlay.
            # Without this, suggest-links would re-propose links that
            # ``--apply`` had already written to the overlay, breaking
            # idempotency (review-w5b BLOCKING fix).
            git_ctx_prov_local = GitContextProvider(repo)
            link_store_base = _LinkStore(repo)
            overlay_mgr_local = OverlayManager(
                repo, git_context=git_ctx_prov_local, link_store=link_store_base
            )
            current_overlay = overlay_mgr_local.current_overlay_path()

            class _BranchLinkStore(_LinkStore):
                def replay(self, *, overlay_path: Path | None = None) -> Any:
                    return super().replay(overlay_path=overlay_path or current_overlay)

            link_store = _BranchLinkStore(repo)
            suggestions: list[LinkSuggestion] = _asyncio.run(
                run_suggest_links(
                    db=db,
                    link_store=link_store,
                    embedder=embedder,
                    provider=provider,
                    config=suggest_config,
                )
            )
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    except LLMError as exc:
        click.echo(f"error: LLM call failed: {exc}", err=True)
        raise SystemExit(1) from None

    if not suggestions:
        if as_json:
            click.echo(json.dumps({"suggestions": []}))
        else:
            click.echo("No link suggestions found.")
        return

    if as_json:
        click.echo(
            json.dumps(
                {
                    "suggestions": [
                        {
                            "from_id": s.from_id,
                            "to_id": s.to_id,
                            "link_type": s.link_type,
                            "confidence": s.confidence,
                            "reason": s.reason,
                        }
                        for s in suggestions
                    ]
                }
            )
        )
        return

    # ── Table output ──────────────────────────────────────────────────────────
    click.echo(f"Found {len(suggestions)} suggestion(s):\n")
    for i, s in enumerate(suggestions, start=1):
        click.echo(f"  [{i}] {s.from_id}")
        click.echo(f"      --[{s.link_type}]--> {s.to_id}")
        click.echo(f"      confidence={s.confidence:.2f}  reason: {s.reason}")

    if not apply:
        return

    # ── --apply: write links to the overlay ───────────────────────────────────
    click.echo()
    if not yes and not click.confirm(f"Apply {len(suggestions)} suggested link(s)?"):
        click.echo("Aborted.")
        return

    try:
        git_ctx_prov = GitContextProvider(repo)
        git_ctx = git_ctx_prov.get()
    except GitContextError as exc:
        click.echo(f"error: git context unavailable: {exc}", err=True)
        raise SystemExit(1) from None

    overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
    written = 0
    try:
        with ScryDB(repo, read_only=True) as db:
            for s in suggestions:
                from_anchor = db.get_anchor(s.from_id)
                to_anchor = db.get_anchor(s.to_id)
                if from_anchor is None or to_anchor is None:
                    click.echo(
                        f"warning: anchor not found, skipping: {s.from_id!r} or {s.to_id!r}",
                        err=True,
                    )
                    continue
                lnk_id = new_link_id()
                evt_id = new_event_id()
                record = LinkRecord.model_validate(
                    {
                        "op": LinkOp.UPSERT,
                        "link_id": lnk_id,
                        "event_id": evt_id,
                        "from": s.from_id,
                        "from_type": AnchorType(from_anchor.type),
                        "to": s.to_id,
                        "to_type": AnchorType(to_anchor.type),
                        "type": LinkType(s.link_type),
                        "from_content_hash": from_anchor.content_hash,
                        "to_content_hash": to_anchor.content_hash,
                        "commit_sha": git_ctx.head_sha,
                        "worktree_dirty": bool(git_ctx.dirty_files),
                        "evidence": s.reason,
                    }
                )
                try:
                    overlay_mgr.append_to_current_branch_overlay(record)
                    written += 1
                except (LinkValidationError, GitContextError) as exc:
                    click.echo(
                        f"warning: could not write link {s.from_id!r} -> {s.to_id!r}: {exc}",
                        err=True,
                    )
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo(f"Wrote {written} link(s) to overlay.")


# ─── scry reconcile ───────────────────────────────────────────────────────────


@main.command("reconcile")
@click.argument("link_id", required=False, default=None)
@click.option("--all", "all_links", is_flag=True, help="Reconcile all drifted links.")
@click.option(
    "--apply",
    "apply_patch",
    is_flag=True,
    help="Apply the proposed patch to the working tree via git apply.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip per-link confirmation when --apply is set.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Machine-readable JSON output.",
)
@click.pass_context
def reconcile(
    ctx: click.Context,
    link_id: str | None,
    all_links: bool,
    apply_patch: bool,
    yes: bool,
    json_output: bool,
) -> None:
    """AI-assisted patch proposal for drifted links (DESIGN.md §5.5).

    Fetches the git diff of each changed endpoint, sends context to the
    configured LLM, and outputs a unified diff for review.

    Requires ``scry index`` to have been run first.
    """
    from scry.cmd_reconcile import run_reconcile_cmd

    run_reconcile_cmd(
        repo_root=_resolve_repo_root(ctx),
        link_id=link_id,
        all_links=all_links,
        apply_patch=apply_patch,
        yes=yes,
        json_output=json_output,
    )


# ─── scry callers / scry subclasses (W6e — DESIGN.md lines 1444-1445) ────────


@main.command("callers")
@click.argument("anchor_id", type=str)
@click.option(
    "--max-depth",
    default=1,
    type=int,
    show_default=True,
    help="BFS depth for incomingCalls walk (1 = direct callers only).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.pass_context
def callers(ctx: click.Context, anchor_id: str, max_depth: int, as_json: bool) -> None:
    """Show symbols that CALL the given code anchor (W6e reverse query)."""
    import asyncio as _asyncio

    from scry.mcp.handlers import MCPContext, MCPServerError
    from scry.mcp.handlers import get_callers as _get_callers

    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(2) from None

    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    async def _run() -> dict[str, Any]:
        from scry.embed import StubEmbedder
        from scry.process.ipc import IPCClient as _IPC

        with ScryDB(repo, read_only=True) as db:
            git_ctx_prov = GitContextProvider(repo)
            overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
            mcp_ctx = MCPContext(
                repo_root=repo,
                config=config,
                db=db,
                embedder=StubEmbedder(),
                git_context=git_ctx_prov,
                overlay_mgr=overlay_mgr,
                indexer=None,
                role="leader",
                ipc_client=cast(_IPC | None, None),
            )
            return await _get_callers(mcp_ctx, anchor_id=anchor_id, max_depth=max_depth)

    try:
        result = _asyncio.run(_run())
    except MCPServerError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None
    except (GitContextError, LockTimeout, OSError) as exc:
        # SR3-2: include GitContextError so callers on a no-commits
        # repo gets a clean error message instead of a stack trace.
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    callers_list: list[dict[str, Any]] = result.get("callers", [])
    if not callers_list:
        # UT1-4 / SR5-4: empty result is ambiguous between "truly no callers"
        # and "LSP not configured/installed".  Surface the LSP availability
        # via the same status_for() check the indexer uses, parameterised
        # by the actual anchor language (not the hardcoded "python").
        from scry.lsp.manager import LSPManager

        anchor_lang = _lang_from_anchor_id(anchor_id)
        try:
            lsp_status = LSPManager(repo, config.code_anchors).status_for(anchor_lang)
        except Exception:
            lsp_status = "unknown"
        if lsp_status in ("lsp_unavailable", "skip", "unknown"):
            click.echo(
                "No callers found.\n"
                f"  note: LSP for {anchor_lang} is '{lsp_status}'; results may be "
                f"incomplete.\n"
                f"        {_lsp_install_hint(anchor_lang)}\n"
                "        Or configure code_anchors.languages in .scry/config.yaml."
            )
        else:
            click.echo("No callers found.")
        return
    click.echo(f"Found {len(callers_list)} caller(s) of {anchor_id}:")
    for c in callers_list:
        anchor = c.get("anchor_id") or "(unindexed)"
        click.echo(f"  {anchor}  {c.get('path')}  symbol={c.get('symbol_name')!r}")


@main.command("subclasses")
@click.argument("anchor_id", type=str)
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
@click.pass_context
def subclasses(ctx: click.Context, anchor_id: str, as_json: bool) -> None:
    """Show classes that EXTEND the given class anchor (W6e reverse query)."""
    import asyncio as _asyncio

    from scry.mcp.handlers import MCPContext, MCPServerError
    from scry.mcp.handlers import get_subclasses as _get_subclasses

    repo = _resolve_repo_root(ctx)
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("error: vectors.db not found. Run `scry index` first.", err=True)
        raise SystemExit(2) from None

    try:
        config = load_config(repo)
    except ConfigError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    async def _run() -> dict[str, Any]:
        from scry.embed import StubEmbedder
        from scry.process.ipc import IPCClient as _IPC

        with ScryDB(repo, read_only=True) as db:
            git_ctx_prov = GitContextProvider(repo)
            overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
            mcp_ctx = MCPContext(
                repo_root=repo,
                config=config,
                db=db,
                embedder=StubEmbedder(),
                git_context=git_ctx_prov,
                overlay_mgr=overlay_mgr,
                indexer=None,
                role="leader",
                ipc_client=cast(_IPC | None, None),
            )
            return await _get_subclasses(mcp_ctx, anchor_id=anchor_id)

    try:
        result = _asyncio.run(_run())
    except MCPServerError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None
    except (GitContextError, LockTimeout, OSError) as exc:
        # SR3-2: same GitContextError handling as `scry callers`.
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    subs: list[dict[str, Any]] = result.get("subclasses", [])
    if not subs:
        # UT1-4 / SR5-4: same LSP-availability disambiguation as `scry callers`,
        # parameterised by the actual anchor language.
        from scry.lsp.manager import LSPManager

        anchor_lang = _lang_from_anchor_id(anchor_id)
        try:
            lsp_status = LSPManager(repo, config.code_anchors).status_for(anchor_lang)
        except Exception:
            lsp_status = "unknown"
        if lsp_status in ("lsp_unavailable", "skip", "unknown"):
            click.echo(
                "No subclasses found.\n"
                f"  note: LSP for {anchor_lang} is '{lsp_status}'; results may be "
                f"incomplete.\n"
                f"        {_lsp_install_hint(anchor_lang)}\n"
                "        Or configure code_anchors.languages in .scry/config.yaml."
            )
        else:
            click.echo("No subclasses found.")
        return
    click.echo(f"Found {len(subs)} subclass(es) of {anchor_id}:")
    for s in subs:
        anchor = s.get("anchor_id") or "(unindexed)"
        click.echo(f"  {anchor}  {s.get('path')}  symbol={s.get('symbol_name')!r}")


# ─── scry doctor ──────────────────────────────────────────────────────────────


@main.command("doctor")
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Diagnostics: version, embedder, vectors.db health, lock status, LSP allowlist."""
    allow_untrusted = (ctx.obj or {}).get("allow_untrusted_lsp_config", False)
    repo = _resolve_repo_root(ctx)

    click.echo(f"scry version:  {scry.__version__}")
    click.echo(f"Python:        {sys.version.split()[0]}")
    click.echo(f"Platform:      {platform.system()} {platform.machine()}")

    # Embedding provider.
    try:
        import fastembed

        click.echo(f"fastembed:     available (version {getattr(fastembed, '__version__', '?')})")
    except ImportError:
        click.echo("fastembed:     NOT available (install fastembed to use local embedder)")

    # sqlite-vec version.
    try:
        import sqlite_vec

        # sqlite_vec exposes __version__ (modern) but historically also
        # had a ``version`` attribute on some builds; check both.
        sv_ver = getattr(sqlite_vec, "__version__", None) or getattr(sqlite_vec, "version", "?")
        click.echo(f"sqlite-vec:    {sv_ver}")
    except ImportError:
        click.echo("sqlite-vec:    NOT available")

    # Git context.
    try:
        git_ctx_prov = GitContextProvider(repo)
        git_ctx = git_ctx_prov.get()
        click.echo(f"Git branch:    {git_ctx.branch or '(detached)'}")
        click.echo(f"Git HEAD:      {git_ctx.head_short}")
    except GitContextError as exc:
        # UT1-7: was "ERROR" prefix but doctor exits 0; not a fatal
        # condition. Many scry features work without git (e.g. local
        # search/index in a non-git directory).  Use "WARN" so users
        # don't think scry is broken when scanning a non-git dir.
        click.echo(f"Git:           WARN  - not a git repository ({exc})")

    # leader.lock status.
    lock_path = repo / ".scry" / "leader.lock"
    if not lock_path.exists():
        click.echo("leader.lock:   not present")
    else:
        try:
            state, lock_meta = detect_leader_state(repo)
            if state == LeaderState.LEADER:
                click.echo("leader.lock:   held by THIS process")
            else:
                click.echo("leader.lock:   held by another process (follower mode)")
            # UT4-2: print the lock-holder PID and clarify the
            # Windows scry.exe wrapper PID mismatch.  When tooling
            # spawns ``scry mcp`` via ``subprocess.Popen``, ``Popen.pid``
            # is the scry.exe wrapper, NOT the python.exe that holds
            # the lock.  Document this so PID-based liveness checks
            # don't get tripped up.
            if lock_meta is not None:
                click.echo(f"  lock_pid:       {lock_meta.pid}")
                click.echo(f"  endpoint_uri:   {lock_meta.endpoint_uri}")
                if sys.platform == "win32":
                    click.echo(
                        "  note: on Windows the lock_pid is the python.exe "
                        "holding the lock, NOT the scry.exe wrapper PID. "
                        "subprocess.Popen('scry mcp').pid returns the "
                        "wrapper PID; use lock_pid for liveness checks."
                    )
        except Exception:
            click.echo("leader.lock:   present (status unknown)")

    # vectors.db health.
    db_path = repo / ".scry" / "vectors.db"
    if not db_path.exists():
        click.echo("vectors.db:    not found — run `scry index`")
    else:
        try:
            with ScryDB(repo, read_only=True) as db:
                meta = db.read_index_metadata()
                if meta:
                    click.echo(
                        f"vectors.db:    OK — model={meta.embedding_provider}/{meta.embedding_model}"
                        f" dims={meta.embedding_dimensions}"
                    )
                    # Show index staleness when git context is available.
                    try:
                        git_ctx_prov = GitContextProvider(repo)
                        git_ctx_local = git_ctx_prov.get()
                        try:
                            config = load_config(repo)
                        except Exception:
                            from scry.models import Config as _Config

                            config = _Config()
                        is_stale = check_staleness(git_ctx_local, meta, config)
                        # MEDIUM #2: print full enum-like state, not just stale/fresh.
                        if not is_stale:
                            state_label = IndexState.FRESH.value
                        else:
                            leader_st, _ = detect_leader_state(repo)
                            if leader_st == LeaderState.FOLLOWER:
                                state_label = "stale"
                            else:
                                state_label = IndexState.STALE_NO_WRITE_LOCK.value
                        click.echo(f"index_state:   {state_label}")
                    except Exception:
                        pass
                else:
                    click.echo("vectors.db:    exists but not yet indexed")
        except Exception as exc:
            click.echo(f"vectors.db:    ERROR — {exc}")

    # LSP allowlist.
    click.echo("\nLSP allowlist (§6.2):")
    for lang, binaries in _LSP_ALLOWLIST.items():
        click.echo(f"  {lang}: {', '.join(binaries)}")
        # UAT-6 review-u3 LOW: surface install command per language so users
        # troubleshooting LSP setup via doctor get the same actionable hint
        # callers/subclasses give them.
        primary = binaries[0]
        install = _LSP_INSTALL_COMMANDS.get(primary)
        if install is not None:
            click.echo(f"      install: {install}")

    if allow_untrusted:
        click.echo(
            "\nWARNING: --allow-untrusted-lsp-config is set. "
            "LSP `command:` overrides in .scry/config.yaml will be honoured. "
            "This bypasses the §6.2 security allowlist — use only in trusted repos."
        )

    # Cross-language semantic-drift gate (§5.1 v3.1, review-w4b MEDIUM).
    # Surface mirrors links whose endpoints resolve to different programming
    # languages — these get semantic_drift=null unless the user configures
    # drift.cross_language_threshold explicitly.
    try:
        from scry.drift import _infer_language as _drift_infer_language
        from scry.models import AnchorType as _AT
        from scry.models import LinkType as _LT
        from scry.store.links import LinkStore as _LinkStore

        try:
            _doc_config = load_config(repo)
        except Exception:
            from scry.models import Config as _Config

            _doc_config = _Config()
        _cross_lang_threshold = _doc_config.drift.cross_language_threshold
        if db_path.exists():
            with ScryDB(repo, read_only=True) as _doc_db:
                _link_store = _LinkStore(repo)
                _replay = _link_store.replay()
                cross_lang_pairs: list[tuple[str, str, str, str]] = []
                for _link in _replay.active_links.values():
                    if _link.type != _LT.MIRRORS:
                        continue
                    if _link.from_type != _AT.CODE or _link.to_type != _AT.CODE:
                        continue
                    _from_a = _doc_db.get_anchor(_link.from_id)
                    _to_a = _doc_db.get_anchor(_link.to_id)
                    if _from_a is None or _to_a is None:
                        continue
                    _from_lang = _drift_infer_language(_from_a.path)
                    _to_lang = _drift_infer_language(_to_a.path)
                    if _from_lang and _to_lang and _from_lang != _to_lang:
                        cross_lang_pairs.append((_link.from_id, _from_lang, _link.to_id, _to_lang))
                if cross_lang_pairs:
                    click.echo("\nCross-language mirrors links (§5.1 v3.1):")
                    if _cross_lang_threshold is None:
                        click.echo(
                            "  semantic_drift will be null on these pairs because "
                            "drift.cross_language_threshold is not configured."
                        )
                    else:
                        click.echo(f"  Using cross_language_threshold={_cross_lang_threshold}.")
                    for _src, _sl, _dst, _dl in cross_lang_pairs[:10]:
                        click.echo(f"    [{_sl} → {_dl}]  {_src}  ↔  {_dst}")
                    if len(cross_lang_pairs) > 10:
                        click.echo(f"    ...and {len(cross_lang_pairs) - 10} more")
    except Exception as exc:
        # Doctor must never fail because of cross-language detection.
        click.echo(f"  (cross-language detection skipped: {type(exc).__name__})", err=True)


# ─── scry validate ────────────────────────────────────────────────────────────


@main.command("validate")
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate .scry/config.yaml, frontmatter id uniqueness, and .gitattributes.

    Exits 0 on clean; exits 1 with messages on errors.
    """
    repo = _resolve_repo_root(ctx)
    errors: list[str] = []

    # 1. Config validation.
    try:
        config = load_config(repo)
        click.echo("config.yaml:   OK")
    except ConfigError as exc:
        errors.append(f"config.yaml:   ERROR — {exc}")
        config = None

    # 2. Frontmatter id uniqueness.
    if config is not None:
        from scry.config import should_index as _should_index

        seen_ids: dict[str, str] = {}
        for root, dirs, files in os.walk(repo):
            # Prune ignored directories early so we never descend into
            # .venv / node_modules / tests/fixtures / etc. (B2 fix:
            # validate must respect the exclude config the same way the
            # indexer does).
            dirs[:] = [d for d in dirs if not _path_excluded(Path(root) / d, repo, config.exclude)]
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                fpath = Path(root) / fname
                rel = fpath.relative_to(repo)
                rel_str = str(rel).replace("\\", "/")
                # Skip .scry directory.
                if rel_str.startswith(".scry"):
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    fm, _ = parse_frontmatter(text)
                except Exception as exc:
                    # B3 fix: malformed YAML (e.g. Jinja templates inside
                    # ``{}`` delimiters that look like a frontmatter block)
                    # should not abort validate; just skip the file.
                    errors.append(f"frontmatter parse error in {rel}: {exc}")
                    continue
                # Honour the exclude config so test fixtures with
                # intentional duplicates aren't flagged in real-world
                # repos (B2 fix).
                if not _should_index(rel_str, fm, config.include, config.exclude):
                    continue
                if fm and fm.id is not None:
                    scry_id = fm.id
                    if scry_id in seen_ids:
                        errors.append(
                            f"frontmatter id conflict: {scry_id!r} in {rel} and {seen_ids[scry_id]}"
                        )
                    else:
                        seen_ids[scry_id] = str(rel)

        if not errors or all("frontmatter" not in e for e in errors):
            click.echo(f"frontmatter:   OK ({len(seen_ids)} scry.id values)")

    # 3. .gitattributes union driver check.
    gitattributes_path = repo / ".gitattributes"
    union_line = ".scry/links.jsonl merge=union"
    if not gitattributes_path.exists():
        errors.append(".gitattributes: not found — run `scry init` to create it")
    else:
        content = gitattributes_path.read_text(encoding="utf-8")
        if union_line in content:
            click.echo(".gitattributes: OK (merge=union present)")
        else:
            errors.append(f".gitattributes: missing '{union_line}' — run `scry init` to add it")

    if errors:
        for err in errors:
            click.echo(f"ERROR: {err}", err=True)
        raise SystemExit(1) from None

    click.echo("All checks passed.")


# ─── scry mcp ─────────────────────────────────────────────────────────────────


@main.command("mcp")
@click.option(
    "--daemon",
    is_flag=True,
    help=(
        "Headless / daemon mode (UT1-1): start the leader, IPC server, "
        "and lock file but do NOT serve MCP over stdio.  Stays alive until "
        "Ctrl-C / SIGTERM.  Use this for `scry watch` testing or as a "
        "long-running background leader for editor agents that connect "
        "via the IPC endpoint instead of stdio."
    ),
)
@click.pass_context
def mcp(ctx: click.Context, daemon: bool) -> None:
    """Run the stdio MCP server.

    Starts the FastMCP-based server over stdio.  Auto-detects whether to
    be the leader or a follower process (§10).  Blocks until stdin closes.

    With ``--daemon`` (UT1-1) the process binds the leader lock + IPC
    endpoint but does NOT consume stdin or serve MCP messages over
    stdio — useful for ``scry watch`` testing on Windows where the
    parent shell can't feed an MCP client.
    """
    repo = _resolve_repo_root(ctx)
    allow_untrusted = ctx.obj.get("allow_untrusted_lsp_config", False)
    server = MCPServer(repo, allow_untrusted_lsp_config=allow_untrusted)

    async def _serve_daemon() -> None:
        """Run the leader/IPC server forever, no stdio."""
        await server.start()
        # SR1-4: detect follower-mode startup so we don't silently
        # claim leader status when another --daemon is already running.
        # ``_ctx`` is populated by ``start()``; reading it directly is
        # the simplest way to surface role without a public property.
        ctx = server._ctx
        role = ctx.role if ctx is not None else "unknown"
        if role != "leader":
            click.echo(
                f"scry mcp --daemon: another leader is already running "
                f"(this process is a {role}). Daemon mode is only "
                f"useful for the LEADER process; exiting.",
                err=True,
            )
            await server.stop()
            raise SystemExit(2)
        click.echo(
            f"scry mcp --daemon: leader running, IPC ready. Ctrl-C to stop. PID={os.getpid()}",
            err=True,
        )
        # Sleep forever; cancelled by Ctrl-C / signal.
        with contextlib.suppress(asyncio.CancelledError, KeyboardInterrupt):
            await asyncio.Event().wait()

    try:
        if daemon:
            asyncio.run(_serve_daemon())
        else:
            asyncio.run(server.serve_stdio())
    except (KeyboardInterrupt, EOFError):
        # Clean shutdown when stdin closes (MCP client disconnects) or
        # Ctrl-C is pressed.  Note: bare ValueError used to be caught
        # here too, but that masked real bugs (review-w2j MEDIUM #2).
        pass
    except (ConfigError, IndexerError, MCPServerError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None
    finally:
        # UT4-1 BLOCKING fix: cleanly tear down the IPC server, release
        # the leader lock, and (best-effort) unlink the lock file so
        # followers don't read stale PID/endpoint metadata after a
        # clean shutdown.  Wrapped in try/except so a partial-init
        # failure (e.g. config error before server.start()) doesn't
        # mask the original error from the user.
        with contextlib.suppress(Exception):
            asyncio.run(server.stop())


if __name__ == "__main__":  # pragma: no cover
    main()
