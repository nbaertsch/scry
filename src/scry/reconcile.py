"""Auto-reconcile non-fresh index states (DESIGN.md §7.2 v3.1, W4d).

State machine::

    fresh ──[stale detected]──► stale-reconciling
                                    ├── success ──────► fresh
                                    └── failure ───────► stale-warned
          ──[too many files]────────────────────────────► stale-warned
          ──[no leader / no IPC]──────────────────────── ► stale-no-write-lock

:class:`IndexStateTracker` is held on
:class:`~scry.mcp.handlers.MCPContext` (one instance per process).  Read
handlers call :meth:`~IndexStateTracker.poll_and_maybe_reconcile`; write
handlers and all other callers use :attr:`~IndexStateTracker.current_state`.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scry.config import compute_config_hash
from scry.git_context import GitContext
from scry.models import Config, IndexMetadata, IndexState, new_idempotency_token

if TYPE_CHECKING:
    from scry.index import Indexer
    from scry.process.ipc import IPCClient
    from scry.store.db import ScryDB

__all__ = [
    "IndexStateTracker",
    "check_staleness",
    "estimate_changed_count",
]

logger = logging.getLogger(__name__)


def _staleness_reason(
    git_ctx: GitContext,
    meta: IndexMetadata,
    config: Config,
) -> str | None:
    """Return the primary staleness reason, or ``None`` if the index is fresh.

    Priority: head changed → dirty indexed files → branch changed → config changed.

    Args:
        git_ctx: Current git snapshot.
        meta:    Index provenance metadata (must not be ``None``).
        config:  Validated repo config.

    Returns:
        One of ``"head_changed"``, ``"dirty_files"``, ``"branch_changed"``,
        ``"config_changed"``, or ``None`` when the index is up-to-date.
    """
    if meta.indexed_git_head != git_ctx.head_sha:
        return "head_changed"

    if git_ctx.dirty_files:
        manifest = meta.indexed_file_manifest
        if any(f in manifest for f in git_ctx.dirty_files):
            return "dirty_files"

    # HIGH #1: compare branch / detached-HEAD identity against stored provenance.
    current_branch = git_ctx.branch or f"detached-{git_ctx.head_short}"
    if current_branch != meta.indexed_branch:
        return "branch_changed"

    if compute_config_hash(config) != meta.config_hash:
        return "config_changed"

    return None


def check_staleness(
    git_ctx: GitContext,
    meta: IndexMetadata | None,
    config: Config,
) -> bool:
    """Return ``True`` iff the index is out of date relative to the current worktree.

    Returns ``False`` (not stale) when *meta* is ``None`` — the unbuilt-index
    case is handled by leader startup (initial ``index_async(force=False)``),
    not by the auto-reconcile loop.  DESIGN.md §7.2 v3.1.

    Staleness triggers (in priority order):

    1. HEAD SHA changed.
    2. A dirty tracked file is in the manifest.
    3. Branch or detached-HEAD identity changed (HIGH #1).
    4. Config hash changed.

    Args:
        git_ctx: Current git snapshot.
        meta:    Index provenance metadata from ``ScryDB.read_index_metadata()``.
        config:  Validated repo config (used for config-hash comparison).

    Returns:
        ``True`` when HEAD moved, a dirty tracked file is in the manifest,
        the active branch changed, or the config hash changed.
    """
    if meta is None:
        return False
    return _staleness_reason(git_ctx, meta, config) is not None


def estimate_changed_count(
    git_ctx: GitContext,
    meta: IndexMetadata | None,
    repo_root: Path,
) -> int:
    """Estimate the number of indexed files that need re-indexing.

    - Dirty-worktree: count dirty files present in the manifest.
    - HEAD-change: count files in ``git diff --name-only <old> <new>`` that
      appear in the manifest; falls back to the full manifest size on failure.
    - Returns ``999_999`` as a sentinel when *meta* is ``None``.

    Args:
        git_ctx:   Current git snapshot.
        meta:      Index provenance metadata (``None`` → sentinel).
        repo_root: Repo root for running git commands.
    """
    if meta is None:
        return 999_999

    manifest = meta.indexed_file_manifest
    dirty_in_manifest = sum(1 for f in git_ctx.dirty_files if f in manifest)

    head_changed = 0
    if meta.indexed_git_head != git_ctx.head_sha:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "diff",
                    "--name-only",
                    meta.indexed_git_head,
                    git_ctx.head_sha,
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            if result.returncode == 0:
                changed_files = {
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                }
                head_changed = sum(1 for f in changed_files if f in manifest)
            else:
                head_changed = len(manifest)
        except Exception:  # broad catch: git not found, timeout, etc.
            head_changed = len(manifest)

    return dirty_in_manifest + head_changed


class IndexStateTracker:
    """Mutable state machine tracking whether the scry index is fresh or stale.

    One instance is held on :class:`~scry.mcp.handlers.MCPContext` (created at
    server startup).  All method calls originate from a single asyncio event
    loop, so no internal locking is required except for the reindex-serialization
    lock documented on :meth:`run_leader_reindex`.

    The :attr:`current_state` property drains any completed reconcile task on
    every access so callers always see up-to-date state without an explicit
    ``await``.
    """

    def __init__(self) -> None:
        self._state: IndexState = IndexState.FRESH
        self._reconcile_task: asyncio.Task[None] | None = None
        # Explicit-reindex coalescing (HIGH #3 + MEDIUM #1).
        # Set to a Future while an explicit run_leader_reindex() is in progress;
        # concurrent callers wait on this future instead of starting a second op.
        self._in_flight_reindex: asyncio.Future[Any] | None = None

    @property
    def current_state(self) -> IndexState:
        """Current index state (drains a finished reconcile task on access)."""
        self._drain_task()
        return self._state

    def _drain_task(self) -> None:
        """If the background reconcile task has finished, update state from its outcome."""
        task = self._reconcile_task
        if task is None or not task.done():
            return

        self._reconcile_task = None
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            # Task was cancelled (e.g. by mark_fresh()); leave state as-is.
            return

        if exc is None:
            logger.info("scry: auto-reconcile succeeded → fresh")
            self._state = IndexState.FRESH
        else:
            logger.warning("scry: auto-reconcile failed: %s → stale-warned", exc)
            self._state = IndexState.STALE_WARNED

    async def poll_and_maybe_reconcile(
        self,
        git_ctx: GitContext,
        db: ScryDB,
        config: Config,
        *,
        indexer: Indexer | None,
        ipc_client: IPCClient | None,
        repo_root: Path,
    ) -> IndexState:
        """Detect staleness and, if needed, launch a background reconcile.

        Called by read-path handlers (``search``, ``status``, ``repo_summary``).
        Returns the current :class:`~scry.models.IndexState` immediately — the
        reconcile (if any) runs as a background asyncio Task.

        DESIGN.md §7.2 v3.1 transition rules:

        - Already reconciling → no-op (return ``stale-reconciling``).
        - Not stale → ensure state is ``fresh`` (unless ``stale-warned``
          persists from a previous manual-action-required decision).
        - Changed file count > ``auto_reconcile_max_changed_files`` → ``stale-warned``.
        - Leader available → launch background ``index_async`` → ``stale-reconciling``.
        - Follower + IPC available → ping leader → ``stale-reconciling``.
        - No leader → ``stale-no-write-lock``.

        Args:
            git_ctx:    Current git snapshot.
            db:         ScryDB connection for reading index metadata.
            config:     Validated repo config.
            indexer:    ``Indexer`` instance if this process is the leader; else
                        ``None``.
            ipc_client: ``IPCClient`` for follower → leader delegation; ``None``
                        on the leader or when IPC is unavailable.
            repo_root:  Repo root, passed to :func:`estimate_changed_count`.
        """
        self._drain_task()

        # A reconcile is already in flight — do not start a second pass.
        if self._state == IndexState.STALE_RECONCILING:
            return self._state

        meta = db.read_index_metadata()
        reason = None if meta is None else _staleness_reason(git_ctx, meta, config)
        if reason is None:
            # Index is fresh.  BLOCKING #1: always clear to FRESH — this also
            # recovers from STALE_WARNED when the user has run `scry index`
            # and the on-disk metadata is now current.  Trust the fresh signal.
            self._state = IndexState.FRESH
            return self._state

        # --- Index is stale ---
        max_changed = config.index.auto_reconcile_max_changed_files

        # BLOCKING #2 + HIGH #1: config-only and branch-only changes mean every
        # indexed file is potentially invalidated — the git-diff estimate would
        # return 0 (same HEAD, no dirty files).  Use the full manifest size so
        # the threshold gate triggers STALE_WARNED for large repos.
        if reason in ("config_changed", "branch_changed"):
            estimated = len(meta.indexed_file_manifest) if meta is not None else 999_999
        else:
            estimated = estimate_changed_count(git_ctx, meta, repo_root)
        if estimated > max_changed:
            logger.warning(
                "scry: ~%d changed files exceeds auto-reconcile limit %d"
                " → stale-warned (run `scry index` manually)",
                estimated,
                max_changed,
            )
            self._state = IndexState.STALE_WARNED
            return self._state

        # Launch a background reconcile.
        if indexer is not None:
            logger.info("scry: index stale — leader launching background reconcile")
            self._state = IndexState.STALE_RECONCILING
            self._reconcile_task = asyncio.create_task(
                self._leader_reconcile(indexer), name="scry-reconcile"
            )
        elif ipc_client is not None:
            logger.info("scry: index stale — follower pinging leader to reconcile")
            self._state = IndexState.STALE_RECONCILING
            self._reconcile_task = asyncio.create_task(
                self._follower_reconcile(ipc_client), name="scry-reconcile"
            )
        else:
            logger.warning("scry: index stale — no leader available → stale-no-write-lock")
            self._state = IndexState.STALE_NO_WRITE_LOCK

        return self._state

    async def _leader_reconcile(self, indexer: Indexer) -> None:
        """Run an incremental re-index as the leader."""
        await indexer.index_async(force=False)

    async def _follower_reconcile(self, ipc_client: IPCClient) -> None:
        """Ask the leader to re-index via IPC."""
        token = new_idempotency_token()
        await ipc_client.call(
            "reindex",
            {"force": False, "scope": None},
            idempotency_token=token,
        )

    async def run_leader_reindex(self, indexer: Indexer, *, force: bool = False) -> Any:
        """Serialize explicit reindexes; concurrent calls share the in-flight result.

        Implements HIGH #3 (concurrent follower coalescing) and MEDIUM #1
        (prevent overlap with background reconcile):

        * The check and future-registration happen atomically with respect to
          asyncio (no ``await`` between them), so two concurrent callers cannot
          both decide to start a new reindex.
        * Any in-flight background reconcile is cancelled before the explicit
          reindex starts so the two never run concurrently against the DB.

        Args:
            indexer: The leader's :class:`~scry.index.Indexer` instance.
            force:   Passed through to :meth:`~scry.index.Indexer.index_async`.

        Returns:
            The result returned by :meth:`~scry.index.Indexer.index_async`.
        """
        # HIGH #3: if an explicit reindex is already in flight, join it.
        # This check + the assignment below are atomic in asyncio (no awaits
        # between them), so two concurrent callers cannot both see None.
        if self._in_flight_reindex is not None and not self._in_flight_reindex.done():
            # Cancel background reconcile while we wait (best effort).
            if self._reconcile_task is not None and not self._reconcile_task.done():
                self._reconcile_task.cancel()
            return await asyncio.shield(self._in_flight_reindex)

        # Register the in-flight future BEFORE the first await so any
        # subsequent concurrent call (arriving after our first yield) sees it.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._in_flight_reindex = future

        try:
            # MEDIUM #1: cancel background reconcile before running explicit reindex.
            if self._reconcile_task is not None and not self._reconcile_task.done():
                self._reconcile_task.cancel()
                await asyncio.gather(self._reconcile_task, return_exceptions=True)
            self._reconcile_task = None

            result = await indexer.index_async(force=force)
            future.set_result(result)
            return result
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            # Clear the slot so the next call starts a fresh reindex.
            if self._in_flight_reindex is future:
                self._in_flight_reindex = None

    async def mark_fresh(self) -> None:
        """Force the state to ``fresh`` after a successful explicit reindex.

        Cancels and *awaits* any in-flight background reconcile task (MEDIUM #1)
        so the explicit ``scry index`` / MCP ``reindex`` result is never
        overwritten by a concurrent background run finishing late.
        """
        if self._reconcile_task is not None and not self._reconcile_task.done():
            self._reconcile_task.cancel()
            await asyncio.gather(self._reconcile_task, return_exceptions=True)
        self._reconcile_task = None
        self._state = IndexState.FRESH
