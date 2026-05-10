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
``scry watch``          — Wave 6; prints deferral message, exits 0.
``scry suggest-links``  — Wave 5; prints deferral message, exits 0.
``scry reconcile``      — Wave 5; prints deferral message, exits 0.
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import click

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

_DEFAULT_CONFIG_YAML = """\
include:
  - "**/*.md"
  - "**/*.py"
  - "**/*.ts"
exclude:
  - .scry/**
  - node_modules/**
  - dist/**
  - build/**
  - .venv/**
  - venv/**
classify:
  - { glob: "docs/**.md", type: spec }
  - { glob: "**/*.md", type: doc }
embeddings:
  provider: local
  model: BAAI/bge-small-en-v1.5
  dimensions: 384
"""

_DEFAULT_GITIGNORE = """\
vectors.db
vectors.db.lock
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

    config_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
    click.echo(f"Wrote {config_path}")

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
@click.pass_context
def index(ctx: click.Context, force: bool, reembed: bool) -> None:
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
        if reembed:
            result: IndexResult = indexer.reembed()
        else:
            result = indexer.index(force=force)
    except (IndexerError, LockTimeout) as exc:
        click.echo(f"error: {exc}", err=True)
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
@click.pass_context
def watch(ctx: click.Context) -> None:
    """File watcher — deferred to Wave 6.

    .. note::
        ``scry watch`` is deferred to Wave 6. It will reindex on file change
        and coordinate with the leader process via IPC.
    """
    click.echo("scry watch is deferred to Wave 6")


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
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    c = summary.counts

    if fmt == "json":
        click.echo(summary.model_dump_json(indent=2))
    else:
        # Markdown summary.
        ds = f"{summary.drift_score:.1f}" if summary.drift_score is not None else "null"
        cs = f"{summary.coverage_score:.1f}" if summary.coverage_score is not None else "null"
        click.echo("## scry check\n")
        click.echo("| Metric | Score |")
        click.echo("|--------|-------|")
        click.echo(f"| drift_score | {ds} |")
        click.echo(f"| coverage_score | {cs} |")
        click.echo(f"| drift_coverage | {summary.drift_coverage} |")
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
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, anchor_type_filter: str | None) -> None:
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

    try:
        with ScryDB(repo, read_only=True) as db:
            results = hybrid_search(
                query,
                db=db,
                embedder=embedder,
                config=config.retrieval,
                top_k=top_k,
                anchor_types=anchor_types,
            )
            packets = [build_anchor_packet(r, db=db, config=config.retrieval) for r in results]
    except (LockTimeout, OSError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None

    if not packets:
        click.echo("No results.")
        return

    for pkt in packets:
        excerpt = (pkt.evidence_excerpt or "")[:120].replace("\n", " ")
        click.echo(f"{pkt.anchor.id}  score={pkt.score:.4f}")
        if excerpt:
            click.echo(f"  {excerpt}")


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

    lnk_id = new_link_id()
    evt_id = new_event_id()
    record = LinkRecord.model_validate(
        {
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
    )

    overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
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
@click.option("--accept-all", is_flag=True, help="Automatically accept all suggestions.")
@click.pass_context
def suggest_links(ctx: click.Context, scope: str | None, accept_all: bool) -> None:
    """AI-augmented batch link suggestions — deferred to Wave 5.

    .. note::
        ``scry suggest-links`` is deferred to Wave 5 (requires LLM provider).
    """
    click.echo("scry suggest-links is deferred to Wave 5")


# ─── scry reconcile ───────────────────────────────────────────────────────────


@main.command("reconcile")
@click.argument("link_id")
@click.pass_context
def reconcile(ctx: click.Context, link_id: str) -> None:
    """AI-assisted patch proposal for drifted links — deferred to Wave 5.

    .. note::
        ``scry reconcile`` is deferred to Wave 5 (requires LLM provider).
    """
    click.echo("scry reconcile is deferred to Wave 5")


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

        click.echo(f"sqlite-vec:    {getattr(sqlite_vec, 'version', '?')}")
    except ImportError:
        click.echo("sqlite-vec:    NOT available")

    # Git context.
    try:
        git_ctx_prov = GitContextProvider(repo)
        git_ctx = git_ctx_prov.get()
        click.echo(f"Git branch:    {git_ctx.branch or '(detached)'}")
        click.echo(f"Git HEAD:      {git_ctx.head_short}")
    except GitContextError as exc:
        click.echo(f"Git:           ERROR — {exc}")

    # leader.lock status.
    lock_path = repo / ".scry" / "leader.lock"
    if not lock_path.exists():
        click.echo("leader.lock:   not present")
    else:
        try:
            state, _ = detect_leader_state(repo)
            if state == LeaderState.LEADER:
                click.echo("leader.lock:   held by THIS process")
            else:
                click.echo("leader.lock:   held by another process (follower mode)")
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
        seen_ids: dict[str, str] = {}
        for root, _dirs, files in os.walk(repo):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                fpath = Path(root) / fname
                rel = fpath.relative_to(repo)
                # Skip .scry directory.
                if str(rel).startswith(".scry"):
                    continue
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    fm, _ = parse_frontmatter(text)
                    if fm and fm.id is not None:
                        scry_id = fm.id
                        if scry_id in seen_ids:
                            errors.append(
                                f"frontmatter id conflict: {scry_id!r} "
                                f"in {rel} and {seen_ids[scry_id]}"
                            )
                        else:
                            seen_ids[scry_id] = str(rel)
                except Exception as exc:
                    errors.append(f"frontmatter parse error in {rel}: {exc}")

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
@click.pass_context
def mcp(ctx: click.Context) -> None:
    """Run the stdio MCP server.

    Starts the FastMCP-based server over stdio.  Auto-detects whether to
    be the leader or a follower process (§10).  Blocks until stdin closes.
    """
    repo = _resolve_repo_root(ctx)
    allow_untrusted = ctx.obj.get("allow_untrusted_lsp_config", False)
    server = MCPServer(repo, allow_untrusted_lsp_config=allow_untrusted)
    try:
        asyncio.run(server.serve_stdio())
    except (KeyboardInterrupt, EOFError):
        # Clean shutdown when stdin closes (MCP client disconnects) or
        # Ctrl-C is pressed.  Note: bare ValueError used to be caught
        # here too, but that masked real bugs (review-w2j MEDIUM #2).
        pass
    except (ConfigError, IndexerError, MCPServerError) as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from None


if __name__ == "__main__":  # pragma: no cover
    main()
