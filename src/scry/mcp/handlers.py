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
    IndexState,
    LinkOp,
    LinkRecord,
    LinkType,
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
    "commit_links",
    "find_drift",
    "get_anchor",
    "get_links",
    "propose_link",
    "reindex",
    "repo_summary",
    "search",
    "status",
]

logger = logging.getLogger(__name__)


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
    anchor_types: list[AnchorType] | None = None
    if types is not None:
        try:
            anchor_types = [AnchorType(t) for t in types]
        except ValueError as exc:
            raise MCPServerError(f"Invalid anchor type in 'types': {exc}") from exc

    git_ctx = ctx.git_context.get()
    index_state = await ctx.index_state_tracker.poll_and_maybe_reconcile(
        git_ctx,
        ctx.db,
        ctx.config,
        indexer=ctx.indexer,
        ipc_client=ctx.ipc_client,
        repo_root=ctx.repo_root,
    )

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
        packets.append(packet.model_dump())

    return packets


async def get_anchor(ctx: MCPContext, id: str) -> dict[str, Any] | None:
    """Load a single anchor by its primary ID.

    Args:
        ctx: Injected :class:`MCPContext`.
        id:  Anchor primary key (e.g. ``"docs/spec.md::intro"``).

    Returns:
        The anchor as a dict (including full ``content_text`` and top-level
        ``index_state``), or ``None`` if the anchor is not in the database.
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
    result = anchor.model_dump()
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

    return {"links": rows, "index_state": index_state}


async def find_drift(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    status_filter: list[str] | None = None,
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

    Returns:
        Dict with top-level keys ``entries`` (list of drift dicts) and
        ``index_state``.  Each entry dict has keys: ``link_id``, ``from_id``,
        ``to_id``, ``link_type``, ``drift_status``, ``semantic_drift``,
        ``drift_coverage`` (always ``"section-only"`` in Wave 2).
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

    rows: list[dict[str, Any]] = []
    for ev in evaluations:
        link = ev.link

        if scope is not None and not link.from_id.startswith(scope):
            continue

        if allowed_statuses is not None and ev.drift_status not in allowed_statuses:
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

    return {
        "link_id": lid,
        "from_id": from_id,
        "to_id": to_id,
        "link_type": link_type,
        "status": "staged",
        "index_state": ctx.index_state_tracker.current_state,
    }


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

    return {
        "link_id": link.link_id,
        "from_id": link.from_id,
        "to_id": link.to_id,
        "link_type": link.type,
        "status": "accepted",
        "index_state": ctx.index_state_tracker.current_state,
    }


async def commit_links(
    ctx: MCPContext,
    *,
    scope: str | None = None,
    idempotency_token: str | None = None,
) -> list[str]:
    """Promote pending overlay records to the baseline link store.

    Wave 2: *scope* is accepted but not used to filter promoted records —
    ``OverlayManager.promote_pending()`` promotes all pending records for the
    current branch.  Wave 4 adds fine-grained scoped promotion.

    Args:
        ctx:               Injected :class:`MCPContext`.
        scope:             Optional path-prefix hint (accepted, ignored in Wave 2).
        idempotency_token: Carried through IPC for leader-side deduplication.

    Returns:
        List of ``event_id`` strings for the promoted records.
    """
    if scope is not None:
        logger.debug("commit_links: scope=%r is accepted but ignored in Wave 2", scope)

    promoted_event_ids = ctx.overlay_mgr.promote_pending()
    ctx.git_context.invalidate()
    return list(promoted_event_ids)


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

    if scope is not None:
        logger.debug("reindex: scope=%r is accepted but ignored in Wave 2", scope)

    result = await ctx.index_state_tracker.run_leader_reindex(ctx.indexer, force=force)
    await ctx.index_state_tracker.mark_fresh()

    return {
        "anchors_extracted": result.anchors_extracted,
        "anchors_embedded": result.anchors_embedded,
        "files_processed": result.files_processed,
        "files_pruned": result.files_pruned,
        "force": force,
        "scope": scope,
        "index_state": IndexState.FRESH,
    }


async def _run_index(indexer: Indexer, *, force: bool) -> Any:
    """Run :meth:`Indexer.index_async` from within an async context.

    Wave 3 note: uses the async variant so the LSP enrichment coroutine is
    awaited directly instead of being dispatched via asyncio.run() (which
    raises RuntimeError when an event loop is already running).
    """
    return await indexer.index_async(force=force)


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
    "status": status,
    "repo_summary": repo_summary,
    "reindex": reindex,
}
