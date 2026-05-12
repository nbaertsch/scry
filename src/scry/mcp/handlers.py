"""Per-tool handler functions for the scry MCP server (workstream W2i).

These are PURE functions of ``MCPContext + args → result``. They are
deliberately separated from FastMCP wiring (``server.py``) so they can be
unit-tested directly without going through the MCP protocol layer or IPC
transport.

Wave 2 simplifications (documented here so Wave 4 knows where to land)
-----------------------------------------------------------------------
* **IndexState** — every response carries ``index_state: IndexState.FRESH``.
  The ``STALE_RECONCILING``, ``STALE_NO_WRITE_LOCK``, and ``STALE_WARNED``
  states are reserved for Wave 4 when the auto-reconcile loop and write-lock
  polling are introduced.
* **find_drift scope** — section-level drift only (DESIGN.md §5.1). LSP
  code-level closure drift (``code-changed`` via transitive hash) is deferred
  to Wave 4 (W4a). ``repo_summary`` returns ``drift_coverage='section-only'``
  to signal the partial scope.
* **accept_link** — Wave 2 has no "accepted" status field on overlay records.
  ``accept_link`` looks up the proposed link in the active overlay and returns
  its current state as a confirmation; a real "accepted" stage lands in Wave 3.
* **reindex scope** — ``Indexer.index()`` does not support a path-prefix scope
  in Wave 2; the ``scope`` argument to the MCP ``reindex`` tool is accepted but
  silently ignored. Wave 4 (W4b) adds scoped incremental indexing.
* **propose_link supersedes** — always creates a new ``link_id``; duplicate
  ``from → to → type`` triples are allowed in Wave 2 (dedup logic is Wave 3).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scry.drift import DriftEvaluation, compute_drift_summary, evaluate_link_drift
from scry.embed import Embedder
from scry.git_context import GitContextProvider
from scry.index import Indexer
from scry.models import (
    AnchorLinkProjection,
    AnchorType,
    Config,
    DriftConfig,
    DriftStatus,
    IndexState,
    LinkOp,
    LinkRecord,
    LinkType,
    new_event_id,
    new_link_id,
)
from scry.process.ipc import IPCClient
from scry.reconcile import IndexStateTracker
from scry.retrieve import build_anchor_packet, hybrid_search
from scry.store.db import ScryDB
from scry.store.links import ReplayResult
from scry.store.overlay import OverlayManager

__all__ = [
    "HANDLERS",
    "MCPContext",
    "MCPServerError",
    "accept_link",
    "apply_link_suggestions",
    "commit_links",
    "find_drift",
    "get_anchor",
    "get_callers",
    "get_links",
    "get_subclasses",
    "propose_link",
    "reindex",
    "repo_summary",
    "search",
    "status",
    "suggest_links_candidates",
    "unlink",
]

logger = logging.getLogger(__name__)

# Mapping from lowercase file extension → LSP language name (mirrors index.py).
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".zig": "zig",
    ".go": "go",
    ".rs": "rust",
}


# ─── Exceptions ───────────────────────────────────────────────────────────────


class MCPServerError(Exception):
    """Raised for expected tool errors (validation, auth, not-found, etc.)."""


# ─── Context ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MCPContext:
    """All long-lived per-process state injected into handlers.

    Composed once at startup by :class:`~scry.mcp.server.MCPServer` and passed
    to every tool handler.  All fields are read at handler call time (the
    context is frozen — references are stable, but the objects they point to
    are mutable and may evolve between calls).

    Attributes:
        repo_root:   Absolute path to the repository root.
        config:      Loaded and validated ``Config`` from ``.scry/config.yaml``.
        db:          Live ``ScryDB`` connection (read-write for leader, read-only
                     for follower).
        embedder:    Embedding backend (lazy-loaded on first search call).
        git_context: Cached git state provider.
        overlay_mgr: Branch-aware link overlay façade (W2c).
        indexer:     ``Indexer`` instance for the leader; ``None`` for a follower
                     (followers never call :meth:`Indexer.index` directly —
                     writes go via IPC to the leader).
        role:        ``"leader"`` or ``"follower"``.
        ipc_client:  Connected :class:`~scry.process.ipc.IPCClient` for a
                     follower; ``None`` on the leader.
        index_state_tracker: Mutable state machine for DESIGN.md §7.2 v3.1
                     auto-reconcile logic (W4d).
    """

    repo_root: Path
    config: Config
    db: ScryDB
    embedder: Embedder
    git_context: GitContextProvider
    overlay_mgr: OverlayManager
    indexer: Indexer | None
    role: Literal["leader", "follower"]
    ipc_client: IPCClient | None
    index_state_tracker: IndexStateTracker = field(default_factory=IndexStateTracker)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _get_current_content_hash(db: ScryDB, anchor_id: str) -> str | None:
    """Return the current ``content_hash`` for *anchor_id*, or ``None`` if absent."""
    anchor = db.get_anchor(anchor_id)
    return anchor.content_hash if anchor is not None else None


def _replay_active(overlay_mgr: OverlayManager) -> ReplayResult:
    """Convenience wrapper — call ``overlay_mgr.replay_active()``."""
    return overlay_mgr.replay_active()


def _build_link_projections(
    anchor_id: str,
    replay: ReplayResult,
    db: ScryDB,
    *,
    direction: Literal["outgoing", "incoming", "both"] = "both",
    link_types: list[str] | None = None,
    max_out: int = 5,
    max_in: int = 5,
    drift_config: DriftConfig | None = None,
) -> list[AnchorLinkProjection]:
    """Project active links into ``AnchorLinkProjection`` instances.

    Evaluates drift for each link so the projection carries live drift status.

    Args:
        anchor_id:    Anchor whose links to project.
        replay:       Active link table (baseline ⊕ overlay).
        db:           Live database connection for drift evaluation.
        direction:    Which direction(s) to include.
        link_types:   Optional allow-list of :class:`~scry.models.LinkType` strings.
        max_out:      Maximum number of outgoing projections.
        max_in:       Maximum number of incoming projections.
        drift_config: Optional ``DriftConfig`` so user-tuned semantic-drift
                      thresholds apply to projection drift evaluation
                      (review-w4b BLOCKING fix).

    Returns:
        List of :class:`~scry.models.AnchorLinkProjection` sorted outgoing-first.
    """
    conflicts = set(replay.merge_conflicts)
    allowed_types: set[str] | None = set(link_types) if link_types is not None else None

    projections: list[AnchorLinkProjection] = []
    count_out = 0
    count_in = 0

    for link in replay.active_links.values():
        is_out = link.from_id == anchor_id
        is_in = link.to_id == anchor_id

        include = False
        if direction in ("outgoing", "both") and is_out and count_out < max_out:
            include = True
        if direction in ("incoming", "both") and is_in and count_in < max_in:
            include = True
        if not include:
            continue

        if allowed_types is not None and link.type not in allowed_types:
            continue

        evaluation = evaluate_link_drift(
            link, db=db, merge_conflicts=conflicts, config=drift_config
        )

        proj = AnchorLinkProjection(
            to=link.to_id if is_out else link.from_id,
            to_type=AnchorType(link.to_type if is_out else link.from_type),
            type=LinkType(link.type),
            drift_status=evaluation.drift_status,
            semantic_drift=evaluation.semantic_drift,
        )
        projections.append(proj)
        if is_out:
            count_out += 1
        if is_in:
            count_in += 1

    return projections


# ─── Handlers ────────────────────────────────────────────────────────────────


async def search(
    ctx: MCPContext,
    query: str,
    *,
    types: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Hybrid BM25 + vector retrieval; returns ranked :class:`~scry.models.AnchorPacket` dicts.

    Args:
        ctx:    Injected :class:`MCPContext`.
        query:  Natural-language search string.
        types:  Optional allow-list of :class:`~scry.models.AnchorType` strings
                to restrict the candidate pool.
        top_k:  Maximum number of results to return.

    Returns:
        List of :class:`~scry.models.AnchorPacket` dicts, sorted by descending
        relevance score.  Each dict includes a populated ``links`` field sourced
        from the active overlay (Wave 2 section-level drift only).
    """
    # UAT-M-10 / U-fix-7: reject empty/whitespace-only queries early.
    # Was silently reaching the DB and returning unranked results,
    # inconsistent with the top_k<1 guard in the same handler.
    if not query or not query.strip():
        raise MCPServerError("'query' must be a non-empty string (got empty or whitespace-only)")

    anchor_types: list[AnchorType] | None = None
    if types is not None:
        try:
            anchor_types = [AnchorType(t) for t in types]
        except ValueError as exc:
            raise MCPServerError(f"Invalid anchor type in 'types': {exc}") from exc

    # SR4-2: reject non-positive top_k explicitly.  Negative values
    # otherwise leak Python list-slicing semantics; zero silently
    # returns an empty result set without indicating a caller error.
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise MCPServerError(f"'top_k' must be a positive integer, got {top_k!r}")

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    # UAT-M-3 / U-fix-2: hybrid_search is synchronous and shares the
    # process-local SQLite connection.  We previously wrapped it in
    # ``asyncio.to_thread`` to keep the event loop responsive — but
    # that violates SQLite's "connection used in a single thread"
    # constraint and crashes with ``ProgrammingError``.
    #
    # The original "20s hang" reported by UAT-M-3 was first-time
    # fastembed model load on cold disk + Windows Defender scanning,
    # NOT a search-time bug.  Once the model is warm, hybrid_search
    # returns in ~10ms.  Embedder warm-up should happen at server
    # startup (handled in MCPServer._start_embedder); this handler
    # therefore calls hybrid_search directly.  If you re-introduce
    # backgrounding, switch the SQLite layer to a connection pool
    # first (see DESIGN.md §10.4 future work).
    results = hybrid_search(
        query,
        db=ctx.db,
        embedder=ctx.embedder,
        config=ctx.config.retrieval,
        top_k=top_k,
        anchor_types=anchor_types,
    )

    replay = _replay_active(ctx.overlay_mgr)
    cfg = ctx.config.retrieval
    packets: list[dict[str, Any]] = []

    for result in results:
        packet = build_anchor_packet(
            result,
            db=ctx.db,
            config=cfg,
            index_state=index_state,
        )
        links = _build_link_projections(
            result.parent_anchor_id,
            replay,
            ctx.db,
            direction="both",
            max_out=cfg.links_per_result.outgoing,
            max_in=cfg.links_per_result.incoming,
            drift_config=ctx.config.drift,
        )
        packet = packet.model_copy(update={"links": links})
        packets.append(_compact_packet(packet.model_dump()))

    return packets


# UAT-13: keys to strip from default MCP responses to reduce token bloat
# (5-result search ≈ 4-8k tokens otherwise; UAT5 finding).  These are
# internal-only fields useful for storage / debugging but not for LLM
# decision-making.  The CLI surface still has access to the full record
# via ``scry get-anchor``.
_COMPACT_DROP_KEYS: tuple[str, ...] = (
    "content_hash",
    "fingerprint_simhash",
    "def_line",
    "def_char",
    "closure_hash",
    "overview_embedding",
    # NOTE: transitive_hash_status is INTENTIONALLY kept (review-u16-18 MEDIUM):
    # it's the small LLM-relevant signal that tells agents when LSP coverage
    # is incomplete (lsp_unavailable / lsp_error / partial).  Bulky hashes
    # are dropped; this single status field stays.
)


def _compact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return *packet* with internal-only fields removed (UAT-13).

    Recursively strips :data:`_COMPACT_DROP_KEYS` from the packet and
    its nested ``anchor`` / ``links`` entries.  Token-bloat reduction
    only — semantics unchanged.
    """
    if not isinstance(packet, dict):
        return packet
    out: dict[str, Any] = {}
    for k, v in packet.items():
        if k in _COMPACT_DROP_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = _compact_packet(v)
        elif isinstance(v, list):
            out[k] = [_compact_packet(item) if isinstance(item, dict) else item for item in v]
        else:
            out[k] = v
    return out


async def get_anchor(ctx: MCPContext, id: str) -> dict[str, Any] | None:
    """Load a single anchor by its primary ID.

    Args:
        ctx: Injected :class:`MCPContext`.
        id:  Anchor primary key (e.g. ``"docs/spec.md::intro"``).

    Returns:
        The anchor as a dict (full ``content_text`` + ``index_state``)
        or ``None`` if the anchor is not in the database.

    UAT-13: per-anchor internal fields (``content_hash``,
    ``fingerprint_simhash``, ``def_line``/``def_char``, etc.) are
    stripped from the response to reduce token bloat in LLM contexts.
    Use the CLI ``scry get-anchor`` command for the full record.
    """
    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )
    anchor = ctx.db.get_anchor(id)
    if anchor is None:
        return None
    result = _compact_packet(anchor.model_dump())
    result["index_state"] = index_state
    return result


async def get_links(
    ctx: MCPContext,
    anchor_id: str,
    *,
    link_types: list[str] | None = None,
    direction: str = "outgoing",
) -> dict[str, Any]:
    """Return active links for *anchor_id* from the baseline ⊕ overlay table.

    Args:
        ctx:        Injected :class:`MCPContext`.
        anchor_id:  Anchor to query.
        link_types: Optional allow-list of :class:`~scry.models.LinkType` strings.
        direction:  ``"outgoing"`` (default), ``"incoming"``, or ``"both"``.

    Returns:
        Dict with top-level keys ``links`` (list of link dicts) and
        ``index_state``.  Each link dict contains: ``link_id``, ``from_id``,
        ``to_id``, ``type``, ``drift_status``, ``semantic_drift``,
        ``direction``, and ``from_content_hash`` / ``to_content_hash``.

    Raises:
        :class:`MCPServerError`: If *direction* is not a recognised value.
    """
    if direction not in ("outgoing", "incoming", "both"):
        raise MCPServerError(
            f"Invalid direction {direction!r}; expected 'outgoing', 'incoming', or 'both'"
        )

    # UAT-M-9 / U-fix-6: validate link_types against the LinkType enum
    # so callers don't silently get an empty result list (previously
    # a typo'd link_type just returned no matches).  Mirrors the
    # find_drift status_filter validation pattern.
    if link_types is not None:
        try:
            valid_link_types = {lt.value for lt in LinkType}
        except Exception:
            valid_link_types = set()
        unknown = set(link_types) - valid_link_types
        if unknown and valid_link_types:
            raise MCPServerError(
                f"Unknown link_type values: {sorted(unknown)!r}. "
                f"Valid values: {sorted(valid_link_types)!r}"
            )

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    replay = _replay_active(ctx.overlay_mgr)
    conflicts = set(replay.merge_conflicts)

    allowed_types: set[str] | None = set(link_types) if link_types is not None else None

    rows: list[dict[str, Any]] = []
    for link in replay.active_links.values():
        is_out = link.from_id == anchor_id
        is_in = link.to_id == anchor_id

        if direction == "outgoing" and not is_out:
            continue
        if direction == "incoming" and not is_in:
            continue
        if direction == "both" and not (is_out or is_in):
            continue

        if allowed_types is not None and link.type not in allowed_types:
            continue

        evaluation = evaluate_link_drift(
            link, db=ctx.db, merge_conflicts=conflicts, config=ctx.config.drift
        )
        link_direction = "outgoing" if is_out else "incoming"

        rows.append(
            {
                "link_id": link.link_id,
                "from_id": link.from_id,
                "to_id": link.to_id,
                "type": link.type,
                "drift_status": evaluation.drift_status,
                "semantic_drift": evaluation.semantic_drift,
                "direction": link_direction,
                "from_content_hash": link.from_content_hash,
                "to_content_hash": link.to_content_hash,
            }
        )

    # UAT-M-7: Wave 2 allows multiple link_ids for the same (from, to, type)
    # triple.  Deduplicate to one logical link per triple, keeping the LAST
    # row encountered — replay processes records in append order, so the
    # last row for a triple corresponds to the most recently created/updated
    # logical link.  (review-r6abc-2: the previous tie-breaker was
    # ``max(link_id)`` which is random for uuid4-shaped IDs.)
    # Report ``historical_count`` so callers can see history existed.
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["from_id"], row["to_id"], row["type"])
        deduped[key] = row  # last write wins (replay-order = creation-order)
        counts[key] = counts.get(key, 0) + 1
    final_rows: list[dict[str, Any]] = []
    for key, best in deduped.items():
        best["historical_count"] = counts[key]
        final_rows.append(best)

    return {"links": final_rows, "index_state": index_state}


async def find_drift(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    status_filter: list[str] | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """Evaluate section-level drift for active links (DESIGN.md §5.1).

    Wave 2: section-level only.  LSP code-level closure drift is deferred to
    Wave 4 (W4a).

    Args:
        ctx:           Injected :class:`MCPContext`.
        scope:         Optional path-prefix filter applied to ``from_id``.
                       Only links whose ``from_id`` starts with *scope* are
                       included.
        status_filter: Optional allow-list of :class:`~scry.models.DriftStatus`
                       strings.  When supplied only matching rows are returned.
        since:         UAT-M-4 / U-fix-3: optional git ref (e.g. ``"main"``,
                       ``"HEAD~1"``).  When supplied only links whose endpoint
                       files appear in ``git diff --name-only <since>..HEAD``
                       are returned — the diff-aware drift gate the CLI
                       exposes as ``scry check --since`` (UAT-8).

    Returns:
        Dict with top-level keys ``entries`` (list of drift dicts) and
        ``index_state``.  Each entry dict has keys: ``link_id``, ``from_id``,
        ``to_id``, ``link_type``, ``drift_status``, ``semantic_drift``,
        ``drift_coverage`` (always ``"section-only"`` in Wave 2).
    """
    # SR4-3: validate status_filter EARLY so callers don't silently get
    # an empty entries list (which is indistinguishable from "no drift").
    # Mirrors the search() contract for `types`.
    if status_filter is not None:
        try:
            valid_statuses = {s.value for s in DriftStatus}
        except Exception:
            valid_statuses = set()
        unknown = set(status_filter) - valid_statuses
        if unknown and valid_statuses:
            raise MCPServerError(
                f"Unknown drift status values: {sorted(unknown)!r}. "
                f"Valid values: {sorted(valid_statuses)!r}"
            )

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    replay = _replay_active(ctx.overlay_mgr)

    # Build a minimal LinkStore-compatible adapter: evaluate_all_drift needs a
    # LinkStore but we already have the replayed active links via overlay_mgr.
    # Rather than re-running replay through a separate LinkStore, we call
    # evaluate_link_drift directly on each active link.
    conflicts = set(replay.merge_conflicts)

    evaluations: list[DriftEvaluation] = [
        evaluate_link_drift(link, db=ctx.db, merge_conflicts=conflicts, config=ctx.config.drift)
        for link in replay.active_links.values()
    ]

    allowed_statuses: set[str] | None = set(status_filter) if status_filter is not None else None

    # UAT-M-4 / U-fix-3: --since diff-aware drift.  Resolve the git diff
    # ONCE up-front (rather than per-link) and filter evaluations to
    # links whose endpoint files appear in the touched set.
    touched_paths: set[str] | None = None
    if since is not None:
        import subprocess as _subprocess

        # SECURITY (review-r6-1 BLOCKING): refuse anything that looks
        # like a git option *before* shelling out.  ``git diff`` would
        # otherwise interpret ``--output=<path>`` as an option and
        # write to that path, turning this read-only MCP tool into a
        # filesystem-write primitive.  We additionally resolve the
        # ref to a SHA via ``rev-parse --verify --end-of-options``
        # so any remaining ambiguity (refs starting with ``-``,
        # whitespace, glob chars) is caught before the diff call.
        if not isinstance(since, str) or not since.strip():
            raise MCPServerError("'since' must be a non-empty string")
        if since.startswith("-"):
            raise MCPServerError(
                f"'since' must be a git ref or commit SHA, not an option flag: {since!r}"
            )
        if any(c in since for c in ("\n", "\r", "\0")):
            raise MCPServerError(f"'since' contains illegal whitespace/NUL: {since!r}")

        try:
            resolved = _subprocess.run(
                [
                    "git",
                    "-C",
                    str(ctx.repo_root),
                    "rev-parse",
                    "--verify",
                    "--end-of-options",
                    f"{since}^{{commit}}",
                ],
                check=True,
                capture_output=True,
                text=True,
                stdin=_subprocess.DEVNULL,
                timeout=10,
            )
        except _subprocess.CalledProcessError as exc:
            raise MCPServerError(
                f"--since {since!r} could not be resolved to a commit (git: {exc.stderr.strip()})"
            ) from None
        except _subprocess.TimeoutExpired as exc:
            raise MCPServerError(f"--since {since!r} resolution timed out after 10s") from exc
        except FileNotFoundError as exc:
            raise MCPServerError("--since requires git on PATH") from exc

        resolved_sha = resolved.stdout.strip()
        # Belt + suspenders: ensure the resolved SHA is a hex string.
        if not resolved_sha or any(c not in "0123456789abcdef" for c in resolved_sha.lower()):
            raise MCPServerError(f"--since {since!r} resolved to non-SHA value: {resolved_sha!r}")

        try:
            diff_out = _subprocess.run(
                [
                    "git",
                    "-C",
                    str(ctx.repo_root),
                    "diff",
                    "--name-only",
                    "--end-of-options",
                    resolved_sha,
                    "HEAD",
                    "--",
                ],
                check=True,
                capture_output=True,
                text=True,
                stdin=_subprocess.DEVNULL,
                timeout=30,
            )
        except _subprocess.CalledProcessError as exc:
            raise MCPServerError(
                f"--since {since!r} diff failed (git: {exc.stderr.strip()})"
            ) from None
        except _subprocess.TimeoutExpired as exc:
            raise MCPServerError(f"--since {since!r} diff timed out after 30s") from exc
        touched_paths = {ln.strip() for ln in diff_out.stdout.splitlines() if ln.strip()}

    rows: list[dict[str, Any]] = []
    for ev in evaluations:
        link = ev.link

        if scope is not None and not link.from_id.startswith(scope):
            continue

        if allowed_statuses is not None and ev.drift_status not in allowed_statuses:
            continue

        # UAT-M-4 / U-fix-3: --since restricts to links touching files
        # in the diff range.  Anchor IDs encode the path before the first
        # ``:``; we match on either endpoint's path.
        if touched_paths is not None:
            from_path = link.from_id.split(":", 1)[0]
            to_path = link.to_id.split(":", 1)[0]
            if from_path not in touched_paths and to_path not in touched_paths:
                continue

        rows.append(
            {
                "link_id": link.link_id,
                "from_id": link.from_id,
                "to_id": link.to_id,
                "link_type": link.type,
                "drift_status": ev.drift_status,
                "semantic_drift": ev.semantic_drift,
                "drift_coverage": "section-only",
            }
        )

    return {"entries": rows, "index_state": index_state}


async def propose_link(
    ctx: MCPContext,
    from_id: str,
    to_id: str,
    link_type: str,
    *,
    evidence: str | None = None,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Stage a new typed link in the current branch overlay.

    Wave 2: always generates a fresh ``link_id`` (no de-duplication of
    ``from → to → type`` triples — Wave 3 adds idempotent upsert).

    Args:
        ctx:               Injected :class:`MCPContext`.
        from_id:           Source anchor ID.
        to_id:             Target anchor ID.
        link_type:         One of the :class:`~scry.models.LinkType` values.
        evidence:          Optional free-text justification for the link.
        idempotency_token: Unused by the handler directly; carried through IPC
                           for leader-side deduplication.

    Returns:
        Dict with ``link_id``, ``from_id``, ``to_id``, ``link_type``,
        ``status`` (``"staged"``), and ``index_state``.

    Raises:
        :class:`MCPServerError`: If *link_type* is invalid, or if either anchor
            is not in the database.
    """
    try:
        lt = LinkType(link_type)
    except ValueError as exc:
        raise MCPServerError(f"Invalid link_type {link_type!r}: {exc}") from exc

    # UAT-R5-15 / U-fix-9: reject self-links.  X→X is semantically
    # nonsensical (would always be both-changed if X changes) and was
    # silently allowed before.
    if from_id == to_id:
        raise MCPServerError(f"Self-links are not allowed: from_id == to_id == {from_id!r}")

    from_anchor = ctx.db.get_anchor(from_id)
    if from_anchor is None:
        raise MCPServerError(f"Source anchor not found: {from_id!r}")
    to_anchor = ctx.db.get_anchor(to_id)
    if to_anchor is None:
        raise MCPServerError(f"Target anchor not found: {to_id!r}")

    git_ctx = ctx.git_context.get()
    lid = new_link_id()

    record = LinkRecord.model_validate(
        {
            "op": LinkOp.UPSERT,
            "link_id": lid,
            "from": from_id,
            "from_type": from_anchor.type,
            "to": to_id,
            "to_type": to_anchor.type,
            "type": lt,
            "from_content_hash": from_anchor.content_hash,
            "to_content_hash": to_anchor.content_hash,
            # W3d §5.3: persist closure hashes so drift can detect callee-body
            # changes even when the caller's own AST is unchanged.  None for
            # non-CODE anchors or when LSP enrichment was unavailable.
            "from_closure_hash": from_anchor.closure_hash,
            "to_closure_hash": to_anchor.closure_hash,
            "evidence": evidence,
            "commit_sha": git_ctx.head_sha,
            "worktree_dirty": bool(git_ctx.dirty_files),
        }
    )

    ctx.overlay_mgr.append_to_current_branch_overlay(record)
    ctx.git_context.invalidate()

    result: dict[str, Any] = {
        "link_id": lid,
        "from_id": from_id,
        "to_id": to_id,
        "link_type": link_type,
        "status": "staged",
        "index_state": ctx.index_state_tracker.current_state,
    }
    # UAT-R5-14: warn when no idempotency_token supplied — retries will
    # mint a new link_id each time, silently creating duplicates.
    if idempotency_token is None:
        logger.warning(
            "propose_link called without idempotency_token; retries will create duplicate links"
        )
        result["warning"] = "called without idempotency_token; retries will create duplicates"
    return result


async def accept_link(
    ctx: MCPContext,
    proposed_id: str,
    *,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Confirm an overlay-staged link proposal (Wave 2 stub).

    Wave 2 does not persist an "accepted" status on overlay records.  This
    handler looks up *proposed_id* (a ``link_id``) in the active overlay and
    returns its current state as a confirmation.  A proper accepted/rejected
    workflow with status persistence lands in Wave 3.

    Args:
        ctx:               Injected :class:`MCPContext`.
        proposed_id:       ``link_id`` returned by :func:`propose_link`.
        idempotency_token: Carried through IPC for leader-side deduplication.

    Returns:
        Dict with ``link_id``, ``from_id``, ``to_id``, ``link_type``,
        ``status`` (``"accepted"``), and ``index_state``.

    Raises:
        :class:`MCPServerError`: If *proposed_id* is not found in the active
            overlay.
    """
    replay = _replay_active(ctx.overlay_mgr)
    link = replay.active_links.get(proposed_id)
    if link is None:
        raise MCPServerError(f"Proposed link not found in active overlay: {proposed_id!r}")

    result: dict[str, Any] = {
        "link_id": link.link_id,
        "from_id": link.from_id,
        "to_id": link.to_id,
        "link_type": link.type,
        "status": "accepted",
        "index_state": ctx.index_state_tracker.current_state,
    }
    # UAT-R5-14: warn when no idempotency_token — retries are safe (Wave 2
    # accept is a read-confirm), but callers should supply a token for
    # consistency with the idempotency contract.
    if idempotency_token is None:
        logger.warning(
            "accept_link called without idempotency_token; retries will create duplicates"
        )
        result["warning"] = "called without idempotency_token; retries will create duplicates"
    return result


async def commit_links(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Promote pending overlay records to the baseline link store.

    Wave 2: *scope* is accepted but not used to filter promoted records —
    ``OverlayManager.promote_pending()`` promotes all pending records for the
    current branch.  Wave 4 adds fine-grained scoped promotion.

    Args:
        ctx:               Injected :class:`MCPContext`.
        scope:             Optional path-prefix hint (accepted, ignored in Wave 2).
        idempotency_token: Carried through IPC for leader-side deduplication.

    Returns:
        Dict with keys ``promoted`` (list of event_id strings for promoted
        records) and ``index_state`` (UT3-5 fix: §7.3 requires index_state
        on every response).  Pre-fix returned bare ``list[str]``.
    """
    if scope is not None:
        logger.debug("commit_links: scope=%r is accepted but ignored in Wave 2", scope)

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    promoted_event_ids = ctx.overlay_mgr.promote_pending()
    ctx.git_context.invalidate()

    # UAT-M-6 / U-fix-5: enrich the promoted list so callers can verify
    # by link_id (the stable user-facing handle) instead of event_id
    # (an implementation-internal identifier).  Cross-reference the
    # promoted event IDs against the active link table to recover
    # link_ids.
    replay_after = ctx.overlay_mgr.replay_active()
    event_to_link: dict[str, str] = {}
    for lk in replay_after.active_links.values():
        event_to_link[lk.last_event_id] = lk.link_id
    promoted_records = [
        {"event_id": eid, "link_id": event_to_link.get(eid)} for eid in promoted_event_ids
    ]
    return {
        "promoted": promoted_records,
        # Backwards-compatible alias for callers using the old shape.
        "promoted_event_ids": list(promoted_event_ids),
        "index_state": index_state,
    }


async def unlink(
    ctx: MCPContext,
    link_id: str,
    *,
    reason: str | None = None,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Tombstone a link by appending a DELETE record (UAT-M-5 / U-fix-4).

    Mirrors the ``scry unlink`` CLI command.  Appends a DELETE record
    to the current branch overlay; the link no longer appears in
    ``get_links`` / ``find_drift`` output.  If the link is in baseline,
    the DELETE is promoted on next ``commit_links``.

    Per DESIGN.md §3.5: a tombstoned link's ``link_id`` is reserved
    permanently.  To re-create a logically equivalent link, call
    :func:`propose_link` (which mints a fresh ``link_id``).

    Args:
        ctx:               Injected :class:`MCPContext`.
        link_id:           The ``link_id`` of the link to tombstone.
        reason:            Optional rationale stored with the DELETE record.
        idempotency_token: Carried through IPC for leader-side dedup.

    Returns:
        Dict with ``link_id``, ``event_id`` (the DELETE event), ``reason``,
        and ``index_state``.

    Raises:
        :class:`MCPServerError`: If *link_id* is not in the active link
            table (already tombstoned or never existed).
    """
    if not isinstance(link_id, str) or not link_id.strip():
        raise MCPServerError("unlink: link_id must be a non-empty string.")

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    replay = _replay_active(ctx.overlay_mgr)
    active = replay.active_links.get(link_id)
    if active is None:
        raise MCPServerError(
            f"link_id {link_id!r} not found in active table "
            "(may already be tombstoned, or never existed)."
        )

    evt_id = new_event_id()
    record = LinkRecord.model_validate(
        {
            "op": LinkOp.DELETE,
            "link_id": link_id,
            "event_id": evt_id,
            # supersedes is required on DELETE per §3.5 rule 5.
            "supersedes": active.last_event_id,
            "reason": reason or "scry unlink (MCP)",
        }
    )
    try:
        ctx.overlay_mgr.append_to_current_branch_overlay(record)
    except Exception as exc:
        raise MCPServerError(f"unlink: failed to append DELETE record: {exc}") from exc

    result: dict[str, Any] = {
        "link_id": link_id,
        "event_id": evt_id,
        "reason": record.reason,
        "index_state": index_state,
    }
    # UAT-R5-14: warn when no idempotency_token — a retry without a token
    # will fail gracefully (link already tombstoned), but token-aware
    # callers should supply one for correct IPC idempotency semantics.
    if idempotency_token is None:
        logger.warning("unlink called without idempotency_token; retries will create duplicates")
        result["warning"] = "called without idempotency_token; retries will create duplicates"
    return result


async def status(ctx: MCPContext) -> dict[str, Any]:
    """Return current server + overlay status.

    Args:
        ctx: Injected :class:`MCPContext`.

    Returns:
        Dict with keys: ``role``, ``branch``, ``head_sha``, ``pending_count``,
        ``merge_conflict_count``, ``index_state``.
    """
    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )
    replay = _replay_active(ctx.overlay_mgr)
    pending = ctx.overlay_mgr.list_pending_overlay_records()

    return {
        "role": ctx.role,
        "branch": git_ctx.branch,
        "head_sha": git_ctx.head_sha,
        "pending_count": len(pending),
        "merge_conflict_count": len(replay.merge_conflicts),
        "index_state": index_state,
    }


async def repo_summary(ctx: MCPContext) -> dict[str, Any]:
    """Return a high-level repository summary with drift score.

    Uses Wave 2 partial-drift output (``drift_coverage='section-only'``).

    Args:
        ctx: Injected :class:`MCPContext`.

    Returns:
        Dict with keys: ``total_anchors``, ``anchor_counts`` (by type),
        ``drift_score``, ``coverage_score``, ``drift_counts``,
        ``drift_coverage`` (always ``"section-only"`` in Wave 2),
        ``index_state``, ``branch``.
    """
    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    # Anchor counts.
    anchors = ctx.db.list_anchors()
    total = len(anchors)
    counts_by_type: dict[str, int] = {}
    code_anchor_ids: list[str] = []
    for anchor in anchors:
        t = str(anchor.type)
        counts_by_type[t] = counts_by_type.get(t, 0) + 1
        if anchor.type == AnchorType.CODE:
            code_anchor_ids.append(anchor.id)

    # Drift evaluation.
    replay = _replay_active(ctx.overlay_mgr)
    conflicts = set(replay.merge_conflicts)
    evaluations = [
        evaluate_link_drift(link, db=ctx.db, merge_conflicts=conflicts, config=ctx.config.drift)
        for link in replay.active_links.values()
    ]

    # Only CODE anchors with at least one link count as "linked".
    linked_code_ids = {
        ev.link.from_id for ev in evaluations if str(ev.link.from_type) == AnchorType.CODE
    } | {ev.link.to_id for ev in evaluations if str(ev.link.to_type) == AnchorType.CODE}
    linked_code_anchors = len(linked_code_ids & set(code_anchor_ids))

    summary = compute_drift_summary(
        evaluations,
        config=ctx.config.drift,
        coverage_total_code_anchors=len(code_anchor_ids),
        coverage_linked_code_anchors=linked_code_anchors,
    )

    return {
        "total_anchors": total,
        "anchor_counts": counts_by_type,
        "drift_score": summary.drift_score,
        "coverage_score": summary.coverage_score,
        "drift_counts": summary.counts.model_dump(),
        "drift_coverage": "section-only",
        "index_state": index_state,
        "branch": git_ctx.branch,
    }


async def reindex(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    force: bool = False,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Trigger an incremental (or forced) re-index of the repository.

    Wave 2: *scope* is accepted but not used — ``Indexer.index()`` does not
    support path-prefix scoping.  Wave 4 (W4b) adds scoped incremental
    indexing.

    Args:
        ctx:               Injected :class:`MCPContext`.
        scope:             Optional path-prefix hint (accepted, ignored in Wave 2).
        force:             When ``True``, forces a full rebuild from scratch.
        idempotency_token: Carried through IPC for leader-side deduplication.

    Returns:
        Dict with keys: ``anchors_indexed``, ``files_visited``, ``force``,
        ``scope`` (echoed back), ``index_state``.

    Raises:
        :class:`MCPServerError`: If no :class:`~scry.index.Indexer` is available
            (i.e. the calling process is a follower — write calls should have
            been routed to the leader via IPC).
    """
    if ctx.indexer is None:
        raise MCPServerError(
            "reindex is a write operation — this process is a follower and "
            "has no Indexer. The request should be forwarded to the leader via IPC."
        )

    # SR4-1: reject non-bool ``force`` values explicitly.  FastMCP's
    # Pydantic layer otherwise coerces strings like "yes" / "true" /
    # any non-empty string to True, leading to silent surprise full
    # re-indexes.
    if not isinstance(force, bool):
        raise MCPServerError(f"'force' must be a boolean, got {type(force).__name__}: {force!r}")

    if scope is not None:
        # UAT-M-8: surface the limitation in the response (used to be
        # debug-logged only — agents had no way to detect that their
        # ``scope=`` arg was silently ignored).
        logger.warning(
            "reindex: scope=%r is accepted but currently ignored — "
            "running a full repository re-index. Set scope_ignored=true "
            "in the response so callers can detect the limitation.",
            scope,
        )

    result = await ctx.index_state_tracker.run_leader_reindex(ctx.indexer, force=force)
    await ctx.index_state_tracker.mark_fresh()

    return {
        "anchors_extracted": result.anchors_extracted,
        "anchors_embedded": result.anchors_embedded,
        "files_processed": result.files_processed,
        "files_pruned": result.files_pruned,
        "force": force,
        "scope": scope,
        "scope_ignored": scope is not None,
        "index_state": IndexState.FRESH,
    }


async def _run_index(indexer: Indexer, *, force: bool) -> Any:
    """Run :meth:`Indexer.index_async` from within an async context.

    Wave 3 note: uses the async variant so the LSP enrichment coroutine is
    awaited directly instead of being dispatched via asyncio.run() (which
    raises RuntimeError when an event loop is already running).
    """
    return await indexer.index_async(force=force)


# ─── LSP reverse-link handlers (W6e) ─────────────────────────────────────────


async def get_callers(
    ctx: MCPContext,
    anchor_id: str,
    *,
    max_depth: int = 1,
) -> dict[str, Any]:
    """Return symbols that CALL the given code anchor.

    Leverages ``callHierarchy/incomingCalls`` — the inverse of the transitive
    outgoing-call closure built in W3b.

    Args:
        ctx:        Injected :class:`MCPContext`.
        anchor_id:  Primary ID of the target anchor (must be ``CODE`` type).
        max_depth:  Number of incomingCalls hops to walk.  Default ``1``
                    returns direct callers only; values ``> 1`` return
                    transitive callers via BFS.

    Returns:
        Dict with keys:

        * ``callers`` — list of caller dicts, each with:
          - ``anchor_id``: scry anchor ID if found in the index, else ``None``
          - ``path``: repo-relative path inferred from the URI
          - ``symbol_name``: caller's symbol name from LSP
          - ``uri``: raw LSP file URI
          - ``range_start_line``, ``range_start_char``, ``range_end_line``,
            ``range_end_char``: position in the caller's file
        * ``index_state``: current index state string

    Raises:
        :class:`MCPServerError`: If *anchor_id* is not found, is not CODE
            type, or has no persisted LSP position (def_line / def_char).
    """
    from pathlib import Path as _Path

    from scry.lsp.manager import LSPManager as _LSPManager
    from scry.lsp.reverse import get_callers as _get_callers

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    anchor = ctx.db.get_anchor(anchor_id)
    if anchor is None:
        raise MCPServerError(f"Anchor not found: {anchor_id!r}")
    if anchor.type != AnchorType.CODE:
        raise MCPServerError(
            f"Anchor {anchor_id!r} is not CODE type (got {anchor.type!r}); "
            "get_callers requires a CODE anchor"
        )
    if anchor.def_line is None or anchor.def_char is None:
        raise MCPServerError(
            f"Anchor {anchor_id!r} has no persisted LSP position "
            "(def_line / def_char are None — run `scry index` to populate them)"
        )

    suffix = _Path(anchor.path).suffix.lower()
    lang = _EXT_TO_LANG.get(suffix)
    if lang is None:
        # UAT-R5-8: extension has no LSP integration in scry — return with
        # lsp_status instead of raising so callers can distinguish "leaf
        # function" from "no LSP for this file type".
        return {
            "callers": [],
            "lsp_status": "unsupported",
            "index_state": index_state,
        }

    file_uri = (ctx.repo_root / anchor.path).as_uri()
    def_line: int = anchor.def_line
    def_char: int = anchor.def_char

    # Build a position-to-anchor-id lookup from indexed CODE anchors.
    pos_to_id: dict[tuple[str, int, int], str] = {}
    for a in ctx.db.list_anchors(anchor_type=AnchorType.CODE):
        if a.def_line is not None and a.def_char is not None:
            a_uri = (ctx.repo_root / a.path).as_uri()
            pos_to_id[(a_uri, a.def_line, a.def_char)] = a.id

    async with _LSPManager(ctx.repo_root, ctx.config.code_anchors) as mgr:
        session = await mgr.session_for(lang)
        if session is None:
            # UAT-R5-8: report why the list is empty so callers can
            # distinguish "leaf function" from "LSP unavailable".
            mgr_status = mgr.status_for(lang)
            lsp_status_val: str = "unsupported" if mgr_status == "skip" else "unavailable"
            return {
                "callers": [],
                "lsp_status": lsp_status_val,
                "index_state": index_state,
            }

        try:
            caller_refs = await _get_callers(
                session,
                file_uri,
                def_line,
                def_char,
                max_depth=max_depth,
            )
        except Exception as exc:
            logger.warning("get_callers LSP error for %r: %s", anchor_id, exc)
            return {
                "callers": [],
                "lsp_status": "error",
                "lsp_error": str(exc),
                "index_state": index_state,
            }

    def _uri_to_path(uri: str) -> str:
        """Convert file:// URI to a repo-relative path best-effort."""
        try:
            from urllib.parse import unquote, urlparse

            parsed = urlparse(uri)
            if parsed.scheme.lower() != "file":
                return uri
            raw = unquote(parsed.path)
            if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
                raw = raw[1:]
            from pathlib import Path as _P

            full = _P(raw)
            try:
                return str(full.relative_to(ctx.repo_root)).replace("\\", "/")
            except ValueError:
                return str(full)
        except Exception:
            return uri

    callers_out: list[dict[str, Any]] = []
    for ref in caller_refs:
        matched_id = pos_to_id.get((ref.uri, ref.range_start_line, ref.range_start_char))
        callers_out.append(
            {
                "anchor_id": matched_id,
                "path": _uri_to_path(ref.uri),
                "symbol_name": ref.name,
                "uri": ref.uri,
                "range_start_line": ref.range_start_line,
                "range_start_char": ref.range_start_char,
                "range_end_line": ref.range_end_line,
                "range_end_char": ref.range_end_char,
            }
        )

    return {"callers": callers_out, "lsp_status": "available", "index_state": index_state}


async def get_subclasses(
    ctx: MCPContext,
    anchor_id: str,
) -> dict[str, Any]:
    """Return classes that extend / implement the given code anchor.

    Uses ``textDocument/implementation`` (the LSP standard for subclass
    discovery).  The queried anchor should represent a class or interface.

    Args:
        ctx:        Injected :class:`MCPContext`.
        anchor_id:  Primary ID of the class anchor (must be ``CODE`` type).

    Returns:
        Dict with keys:

        * ``subclasses`` — list of subclass dicts, each with:
          - ``anchor_id``: scry anchor ID if found in the index, else ``None``
          - ``path``: repo-relative path inferred from the URI
          - ``symbol_name``: best-available name (anchor name or URI stem)
          - ``uri``: raw LSP file URI
          - ``range_start_line``, ``range_start_char``, ``range_end_line``,
            ``range_end_char``: position in the subclass's file
        * ``index_state``: current index state string

    Raises:
        :class:`MCPServerError`: If *anchor_id* is not found, is not CODE
            type, or has no persisted LSP position.
    """
    from pathlib import Path as _Path

    from scry.lsp.manager import LSPManager as _LSPManager
    from scry.lsp.reverse import get_subclasses as _get_subclasses

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

    anchor = ctx.db.get_anchor(anchor_id)
    if anchor is None:
        raise MCPServerError(f"Anchor not found: {anchor_id!r}")
    if anchor.type != AnchorType.CODE:
        raise MCPServerError(
            f"Anchor {anchor_id!r} is not CODE type (got {anchor.type!r}); "
            "get_subclasses requires a CODE anchor"
        )
    if anchor.def_line is None or anchor.def_char is None:
        raise MCPServerError(
            f"Anchor {anchor_id!r} has no persisted LSP position "
            "(def_line / def_char are None — run `scry index` to populate them)"
        )

    suffix = _Path(anchor.path).suffix.lower()
    lang = _EXT_TO_LANG.get(suffix)
    if lang is None:
        # UAT-R5-8: unsupported extension — return with lsp_status rather than
        # raising so callers can distinguish "no subclasses" from "no LSP here".
        return {
            "subclasses": [],
            "lsp_status": "unsupported",
            "index_state": index_state,
        }

    file_uri = (ctx.repo_root / anchor.path).as_uri()
    def_line_s: int = anchor.def_line
    def_char_s: int = anchor.def_char

    # Build position → anchor_id lookup from indexed CODE anchors.
    pos_to_id: dict[tuple[str, int, int], str] = {}
    for a in ctx.db.list_anchors(anchor_type=AnchorType.CODE):
        if a.def_line is not None and a.def_char is not None:
            a_uri = (ctx.repo_root / a.path).as_uri()
            pos_to_id[(a_uri, a.def_line, a.def_char)] = a.id

    async with _LSPManager(ctx.repo_root, ctx.config.code_anchors) as mgr:
        session = await mgr.session_for(lang)
        if session is None:
            # UAT-R5-8: report why the list is empty.
            mgr_status = mgr.status_for(lang)
            lsp_status_val: str = "unsupported" if mgr_status == "skip" else "unavailable"
            return {
                "subclasses": [],
                "lsp_status": lsp_status_val,
                "index_state": index_state,
            }

        try:
            subclass_refs = await _get_subclasses(
                session,
                file_uri,
                def_line_s,
                def_char_s,
            )
        except Exception as exc:
            logger.warning("get_subclasses LSP error for %r: %s", anchor_id, exc)
            return {
                "subclasses": [],
                "lsp_status": "error",
                "lsp_error": str(exc),
                "index_state": index_state,
            }

    def _uri_to_path(uri: str) -> str:
        try:
            from urllib.parse import unquote, urlparse

            parsed = urlparse(uri)
            if parsed.scheme.lower() != "file":
                return uri
            raw = unquote(parsed.path)
            if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
                raw = raw[1:]
            from pathlib import Path as _P

            full = _P(raw)
            try:
                return str(full.relative_to(ctx.repo_root)).replace("\\", "/")
            except ValueError:
                return str(full)
        except Exception:
            return uri

    subclasses_out: list[dict[str, Any]] = []
    for ref in subclass_refs:
        matched_id = pos_to_id.get((ref.uri, ref.range_start_line, ref.range_start_char))
        # Prefer indexed anchor's symbol_name over URI stem when available.
        if matched_id is not None:
            matched_anchor = ctx.db.get_anchor(matched_id)
            name = (
                matched_anchor.symbol_name
                if matched_anchor is not None and matched_anchor.symbol_name
                else ref.name
            )
        else:
            name = ref.name
        subclasses_out.append(
            {
                "anchor_id": matched_id,
                "path": _uri_to_path(ref.uri),
                "symbol_name": name,
                "uri": ref.uri,
                "range_start_line": ref.range_start_line,
                "range_start_char": ref.range_start_char,
                "range_end_line": ref.range_end_line,
                "range_end_char": ref.range_end_char,
            }
        )

    return {"subclasses": subclasses_out, "lsp_status": "available", "index_state": index_state}


# ─── Agent-driven suggest-links (UAT-R5-2) ───────────────────────────────────


async def suggest_links_candidates(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    source: str = "both",
    limit: int | None = 25,
) -> dict[str, Any]:
    """Return a candidate-pair payload for an LLM-powered MCP client to classify.

    UAT-R5-2: when an LLM-powered MCP client (Claude/Copilot/Cursor) is
    calling scry, requiring scry to ALSO have its own LLM provider
    configured is wasteful and a real adoption barrier.  This tool
    surfaces just the candidate (code, doc) pairs plus the system
    prompt + JSON schema; the calling agent runs the classifier
    itself, then feeds the result to :func:`apply_link_suggestions`.

    No LLM call is made by scry in this path.

    Args:
        ctx:    Injected :class:`MCPContext`.
        scope:  Optional path-prefix filter applied to both sides.
        source: ``"code"`` (scan from code → docs), ``"doc"`` (scan
                from docs → code), or ``"both"`` (default).
        limit:  Maximum candidate pairs to return.  Defaults to 25 to
                fit within typical LLM agent context budgets.

    Returns:
        Dict with ``system_prompt`` (string), ``schema`` (JSON-output
        contract the agent must follow), and ``pairs`` (list of
        ``{pair_id, code, doc}`` records).
    """
    from scry.suggest import (
        DEFAULT_MIN_CONFIDENCE,
        SuggestConfig,
        build_candidates_payload,
        select_candidate_pairs,
    )

    if source not in ("code", "doc", "both"):
        raise MCPServerError(f"'source' must be one of 'code' / 'doc' / 'both', got {source!r}")

    # UAT-R5-2 review-r5-1-2 HIGH: replay MUST include the current branch
    # overlay so already-applied suggestions are excluded from candidate
    # selection.  Without this, an agent that runs candidates → apply →
    # candidates again would re-surface the same pairs.
    replay = ctx.overlay_mgr.replay_active()
    cfg = SuggestConfig(
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        limit=limit,
        source=source,  # type: ignore[arg-type]
        scope=scope,
    )
    pairs = select_candidate_pairs(
        db=ctx.db,
        active_links=replay.active_links,
        embedder=ctx.embedder,
        config=cfg,
    )
    return build_candidates_payload(pairs)


async def apply_link_suggestions(
    ctx: MCPContext,
    *,
    suggestions: list[dict[str, Any]],
    pair_payloads: list[dict[str, Any]] | None = None,
    min_confidence: float = 0.7,
    apply: bool = False,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Apply (or preview) agent-classified link suggestions.

    Companion to :func:`suggest_links_candidates`: takes the agent's
    structured classification of (code, doc) pairs and turns it into
    real overlay link records.

    Args:
        ctx:               Injected :class:`MCPContext`.
        suggestions:       Agent's classification output, matching the
                           ``schema`` returned by ``suggest_links_candidates``
                           (one entry per ``pair_id``).
        pair_payloads:     The ``pairs`` list from the prior
                           ``suggest_links_candidates`` call, used to
                           resolve ``pair_id`` → anchor IDs.  Required
                           because the candidate selection isn't
                           deterministic across separate MCP turns.
        min_confidence:    Threshold below which suggestions are dropped.
        apply:             When ``False`` (default), returns the
                           filtered, validated suggestion list WITHOUT
                           writing.  When ``True``, writes each surviving
                           suggestion to the current branch overlay via
                           the same code-path as ``propose_link``.
        idempotency_token: Optional token for retry-safety.

    Returns:
        ``{"suggestions": [...], "applied": <count|0>, "rejected": <count>}``.
    """
    from scry.models import LinkOp, LinkRecord, new_event_id, new_link_id
    from scry.store.links import LinkStore
    from scry.store.overlay import OverlayManager
    from scry.suggest import LinkSuggestion

    if pair_payloads is None:
        raise MCPServerError(
            "apply_link_suggestions requires 'pair_payloads' from the "
            "preceding suggest_links_candidates call (used to resolve "
            "pair_id back to anchor IDs)."
        )

    # Build pair_id → (from_id, to_id) lookup from the supplied payload.
    pair_id_to_anchors: dict[str, tuple[str, str]] = {}
    for pp in pair_payloads:
        pid = pp.get("pair_id")
        code = pp.get("code", {}).get("id")
        doc = pp.get("doc", {}).get("id")
        if isinstance(pid, str) and isinstance(code, str) and isinstance(doc, str):
            pair_id_to_anchors[pid] = (code, doc)

    # Validate + filter using the same parser the LLM-provider path uses.
    validated: list[LinkSuggestion] = []
    rejected = 0
    # UAT-M-12 / U-fix-8: track WHY each suggestion was rejected so the
    # caller can distinguish "agent sent garbage" from "model below
    # threshold".  Previously a mismatched pair_id was silently
    # dropped — the count went into ``rejected`` but with no signal
    # that anything was wrong with the input.
    rejected_reasons: dict[str, int] = {}

    def _reject(reason: str) -> None:
        nonlocal rejected
        rejected += 1
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for item in suggestions:
        if not isinstance(item, dict):
            _reject("not_a_dict")
            continue
        pid = str(item.get("pair_id", ""))
        if pid not in pair_id_to_anchors:
            _reject("unknown_pair_id")
            continue
        if not bool(item.get("should_link", False)):
            _reject("should_link_false")
            continue
        link_type = str(item.get("link_type", ""))
        if link_type not in ("mirrors", "implements", "references"):
            _reject("invalid_link_type")
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            _reject("invalid_confidence")
            continue
        if not (0.0 <= confidence <= 1.0):
            _reject("confidence_out_of_range")
            continue
        if confidence < min_confidence:
            _reject("below_min_confidence")
            continue
        from_id, to_id = pair_id_to_anchors[pid]
        validated.append(
            LinkSuggestion(
                from_id=from_id,
                to_id=to_id,
                link_type=link_type,
                confidence=confidence,
                reason=str(item.get("reason", "")),
            )
        )

    validated.sort(key=lambda s: s.confidence, reverse=True)

    out: dict[str, Any] = {
        "suggestions": [
            {
                "from_id": s.from_id,
                "to_id": s.to_id,
                "link_type": s.link_type,
                "confidence": s.confidence,
                "reason": s.reason,
            }
            for s in validated
        ],
        "applied": 0,
        "rejected": rejected,
        "rejected_reasons": rejected_reasons,
    }

    if not apply:
        return out

    # ── Write path ──────────────────────────────────────────────────────────
    if ctx.indexer is None:
        raise MCPServerError(
            "apply_link_suggestions(apply=true) is a write op — this process "
            "is a follower and has no Indexer.  The request should be "
            "forwarded to the leader via IPC."
        )

    link_store = LinkStore(ctx.repo_root)
    overlay_mgr = OverlayManager(ctx.repo_root, git_context=ctx.git_context, link_store=link_store)
    git_ctx = ctx.git_context.get()
    written = 0
    for s in validated:
        # Look up endpoint anchors for type + content_hash.
        from_anchor = ctx.db.get_anchor(s.from_id)
        to_anchor = ctx.db.get_anchor(s.to_id)
        if from_anchor is None or to_anchor is None:
            continue
        record = LinkRecord.model_validate(
            {
                "op": LinkOp.UPSERT,
                "link_id": new_link_id(),
                "event_id": new_event_id(),
                "from": s.from_id,
                "from_type": from_anchor.type,
                "to": s.to_id,
                "to_type": to_anchor.type,
                "type": s.link_type,
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
        except Exception as exc:
            logger.warning(
                "apply_link_suggestions: failed to write %s -> %s: %s", s.from_id, s.to_id, exc
            )
    out["applied"] = written
    return out


# ─── Dispatch table ───────────────────────────────────────────────────────────

#: Maps MCP tool names to their handler functions.
#: Used by the leader IPC handler and by ``_dispatch`` in ``server.py``.
HANDLERS: dict[str, Callable[..., Any]] = {
    "search": search,
    "get_anchor": get_anchor,
    "get_links": get_links,
    "find_drift": find_drift,
    "propose_link": propose_link,
    "accept_link": accept_link,
    "commit_links": commit_links,
    "unlink": unlink,
    "status": status,
    "repo_summary": repo_summary,
    "reindex": reindex,
    "get_callers": get_callers,
    "get_subclasses": get_subclasses,
    # UAT-R5-2: agent-driven suggest-links (no scry-side LLM required).
    "suggest_links_candidates": suggest_links_candidates,
    "apply_link_suggestions": apply_link_suggestions,
}
