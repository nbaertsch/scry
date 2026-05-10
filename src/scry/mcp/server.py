"""FastMCP-based MCP server with leader/follower coordination (workstream W2i).

This module wires together all Wave 2 workstreams into a single FastMCP
server process.  It implements the cold-start sequence described in
DESIGN.md §10.4 v3.1 and the leader/follower IPC dispatch from §10.3.

Cold-start sequence (``MCPServer.start()``)
-------------------------------------------
1. Load ``.scry/config.yaml``; raise :class:`~scry.index.IndexerError` if absent.
2. Try to acquire :class:`~scry.process.leader.LeaderLock`.
3. **Leader path:**
   a. Bind the IPC endpoint (:class:`~scry.process.ipc.IPCServer`); on Windows
      ``IPCServer.start()`` raises ``NotImplementedError`` — the leader
      continues in single-process mode (no IPC server, no metadata write).
   b. Once IPC is ready, write lock metadata
      (:meth:`~scry.process.leader.LeaderLock.write_metadata`).
   c. Open :class:`~scry.store.db.ScryDB` read-write.
   d. Construct :class:`~scry.index.Indexer` and run
      :meth:`~scry.index.Indexer.index` (no-op when everything is fresh).
   e. Call :meth:`~scry.store.overlay.OverlayManager.recover_pending` for
      crash recovery.
   f. Build :class:`~scry.mcp.handlers.MCPContext` with ``role="leader"``.
4. **Follower path:**
   a. Read leader metadata via
      :func:`~scry.process.leader.read_leader_metadata_if_present`.
   b. Parse the endpoint URI; construct
      :class:`~scry.process.ipc.IPCClient`.
   c. Open :class:`~scry.store.db.ScryDB` read-only.
   d. Build :class:`~scry.mcp.handlers.MCPContext` with ``role="follower"``.
5. Register all tool handlers with :attr:`_mcp` and run the FastMCP stdio loop.

Wave 2 deferrals
----------------
* Embedder lazy-loading: the embedder is initialised at startup via
  :func:`~scry.embed.make_embedder`; true lazy-loading (first search call)
  is a Wave 4 optimisation.
* Heartbeat for long IPC ops is stubbed; real heartbeat lands in Wave 6
  (*scry watch*).
* Windows IPC: followers on Windows cannot forward write ops via IPC (named
  pipe support is deferred to Wave 6).  In single-leader Windows mode write
  tools are available only on the leader process.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP

import scry
from scry.config import load_config
from scry.embed import make_embedder
from scry.git_context import GitContextProvider
from scry.index import Indexer
from scry.mcp.handlers import (
    HANDLERS,
    MCPContext,
    MCPServerError,
)
from scry.models import Config, new_idempotency_token
from scry.process.ipc import (
    WRITE_OPS,
    EndpointSpec,
    IPCClient,
    IPCHandler,
    IPCRequest,
    IPCResponse,
    IPCServer,
    derive_endpoint_uri,
    parse_endpoint_uri,
)
from scry.process.leader import LeaderLock, read_leader_metadata_if_present
from scry.reconcile import IndexStateTracker
from scry.store.db import ScryDB
from scry.store.overlay import OverlayManager

logger = logging.getLogger(__name__)

# scry semantic version exposed in leader lock metadata.
_SCRY_VERSION = "0.2.0"


class MCPServer:
    """FastMCP server with leader-follower coordination.

    On startup:
        1. Load config + git context.
        2. Try to acquire :class:`~scry.process.leader.LeaderLock`.
        3. If leader: bind IPC endpoint; init/reconcile DB; run crash recovery.
        4. If follower: open DB read-only; connect to leader's IPC endpoint;
           writes are forwarded via :class:`~scry.process.ipc.IPCClient`.
        5. Register all tool handlers and run FastMCP over stdio.

    On shutdown: release leader lock, close DB, stop IPC.

    Args:
        repo_root: Absolute path to the repository root.
        config:    Optional pre-loaded :class:`~scry.models.Config`.  When
                   ``None`` the config is loaded from ``.scry/config.yaml``
                   during :meth:`start`.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        config: Config | None = None,
        allow_untrusted_lsp_config: bool = False,
    ) -> None:
        self._repo_root = repo_root
        self._preloaded_config = config
        self._allow_untrusted_lsp_config = allow_untrusted_lsp_config
        self._ctx: MCPContext | None = None
        self._leader_lock: LeaderLock | None = None
        self._ipc_server: IPCServer | None = None
        self._mcp: FastMCP = FastMCP(name="scry", version=scry.__version__)
        self._started = False
        self._register_tools()

    # ─── Tool registration ────────────────────────────────────────────────────

    def _register_tools(self) -> None:
        """Register all 10 MCP tools with the FastMCP instance.

        Each tool is a thin async closure that resolves the current
        :class:`~scry.mcp.handlers.MCPContext` and delegates to
        :func:`_dispatch`.
        """
        mcp = self._mcp

        @mcp.tool()
        async def search(
            query: str,
            types: list[str] | None = None,
            top_k: int = 10,
        ) -> list[dict[str, Any]]:
            """Hybrid BM25 + vector retrieval over the indexed repository."""
            return cast(
                list[dict[str, Any]],
                await self._dispatch("search", {"query": query, "types": types, "top_k": top_k}),
            )

        @mcp.tool()
        async def get_anchor(id: str) -> dict[str, Any] | None:
            """Load a single anchor by primary ID (full content, no truncation)."""
            return cast(dict[str, Any] | None, await self._dispatch("get_anchor", {"id": id}))

        @mcp.tool()
        async def get_links(
            anchor_id: str,
            link_types: list[str] | None = None,
            direction: str = "outgoing",
        ) -> list[dict[str, Any]]:
            """Return active links for an anchor from the baseline ⊕ overlay table."""
            return cast(
                list[dict[str, Any]],
                await self._dispatch(
                    "get_links",
                    {"anchor_id": anchor_id, "link_types": link_types, "direction": direction},
                ),
            )

        @mcp.tool()
        async def find_drift(
            scope: str | None = None,
            status_filter: list[str] | None = None,
        ) -> list[dict[str, Any]]:
            """Evaluate section-level drift for active links (Wave 2: section-only)."""
            return cast(
                list[dict[str, Any]],
                await self._dispatch(
                    "find_drift", {"scope": scope, "status_filter": status_filter}
                ),
            )

        @mcp.tool()
        async def propose_link(
            from_id: str,
            to_id: str,
            link_type: str,
            evidence: str | None = None,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Stage a new typed link in the current branch overlay."""
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "propose_link",
                    {
                        "from_id": from_id,
                        "to_id": to_id,
                        "link_type": link_type,
                        "evidence": evidence,
                        "idempotency_token": idempotency_token,
                    },
                ),
            )

        @mcp.tool()
        async def accept_link(
            proposed_id: str,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Confirm an overlay-staged link proposal (Wave 2 stub — no status persistence)."""
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "accept_link",
                    {"proposed_id": proposed_id, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool()
        async def commit_links(
            scope: str | None = None,
            idempotency_token: str | None = None,
        ) -> list[str]:
            """Promote pending overlay records to the baseline link store."""
            return cast(
                list[str],
                await self._dispatch(
                    "commit_links",
                    {"scope": scope, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool()
        async def status() -> dict[str, Any]:
            """Return current server + overlay status."""
            return cast(dict[str, Any], await self._dispatch("status", {}))

        @mcp.tool()
        async def repo_summary() -> dict[str, Any]:
            """Return a high-level repository summary with drift score (section-only)."""
            return cast(dict[str, Any], await self._dispatch("repo_summary", {}))

        @mcp.tool()
        async def reindex(
            scope: str | None = None,
            force: bool = False,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Trigger an incremental (or forced) re-index of the repository."""
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "reindex",
                    {"scope": scope, "force": force, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool()
        async def get_callers(anchor_id: str, max_depth: int = 1) -> dict[str, Any]:
            """Return symbols that CALL the given code anchor (W6e — DESIGN.md line 1444)."""
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "get_callers",
                    {"anchor_id": anchor_id, "max_depth": max_depth},
                ),
            )

        @mcp.tool()
        async def get_subclasses(anchor_id: str) -> dict[str, Any]:
            """Return classes that EXTEND the given class anchor (W6e — DESIGN.md line 1445)."""
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "get_subclasses",
                    {"anchor_id": anchor_id},
                ),
            )

    # ─── Dispatch ────────────────────────────────────────────────────────────

    async def _dispatch(self, op: str, args: dict[str, Any]) -> Any:
        """Route a tool call to the appropriate handler.

        For the **leader** role: call the handler directly on ``self._ctx``.
        For the **follower** role:

        * *Read tools* — call the handler directly on the read-only DB.
        * *Write tools* — forward via :class:`~scry.process.ipc.IPCClient`
          to the leader's IPC endpoint.

        Args:
            op:   Tool name (must be a key in :data:`~scry.mcp.handlers.HANDLERS`).
            args: Keyword-argument dict to unpack into the handler.

        Returns:
            The handler's return value.

        Raises:
            :class:`~scry.mcp.handlers.MCPServerError`: On handler errors or if
                the follower has no IPC client but a write op was requested.
        """
        ctx = self._ctx
        if ctx is None:
            raise MCPServerError("MCPServer has not been started — call start() first")

        if op in WRITE_OPS and ctx.role == "follower":
            # Forward write ops to the leader via IPC.
            if ctx.ipc_client is None:
                raise MCPServerError(
                    f"Write op '{op}' cannot be forwarded: this follower has no IPC "
                    "connection to the leader (Windows single-process mode or leader "
                    "not advertising an endpoint)."
                )
            token: str = args.get("idempotency_token") or new_idempotency_token()
            return await ctx.ipc_client.call(op, args, idempotency_token=token)

        handler = HANDLERS.get(op)
        if handler is None:
            raise MCPServerError(f"Unknown op: {op!r}")

        try:
            return await handler(ctx, **args)
        except MCPServerError:
            raise
        except Exception as exc:
            logger.exception("Handler raised for op=%s", op)
            raise MCPServerError(f"Internal error in '{op}': {exc}") from exc

    # ─── Leader IPC handler ───────────────────────────────────────────────────

    def _make_ipc_handler(self, ctx: MCPContext) -> IPCHandler:
        """Return an :data:`~scry.process.ipc.IPCHandler` for the leader's IPC server.

        The returned callable receives :class:`~scry.process.ipc.IPCRequest`
        objects from follower processes, dispatches them to local handlers, and
        wraps the result in an :class:`~scry.process.ipc.IPCResponse`.
        """

        async def _handle(req: IPCRequest) -> IPCResponse:
            handler = HANDLERS.get(req.op)
            if handler is None:
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Unknown op: {req.op!r}",
                    error_type="validation",
                )
            try:
                result = await handler(ctx, **req.args)
                return IPCResponse(request_id=req.request_id, ok=True, result=result)
            except MCPServerError as exc:
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=str(exc),
                    error_type="validation",
                )
            except Exception as exc:
                logger.exception("Leader IPC handler raised for op=%s", req.op)
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=str(exc),
                    error_type="internal",
                )

        return _handle

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Acquire the leader/follower role and initialise everything.

        Cold-start sequence (DESIGN.md §10.4 v3.1):
        1. Load ``.scry/config.yaml`` (raise if missing).
        2. Try to acquire the leader lock.
        3. Leader: open DB; build context; build IPC handler; bind IPC with
           the REAL handler from the start; write metadata; reconcile;
           recover pending.
        4. Follower: read leader metadata; open DB read-only; connect IPC client.

        ``self._started`` is set only after the entire sequence completes
        successfully — partial failures leave the flag False so a
        subsequent ``start()`` call can retry cleanly (review-w2i MEDIUM
        #3 fix).

        Raises:
            :class:`~scry.config.ConfigError`: If ``.scry/config.yaml`` is
                absent or malformed (review-w2i MEDIUM #4 docstring fix —
                previously claimed IndexerError).
        """
        if self._started:
            return

        # Step 1: Load config.
        cfg = self._preloaded_config or load_config(self._repo_root)

        # Step 2: Try to acquire the leader lock.
        leader_lock = LeaderLock.try_acquire(self._repo_root)

        if leader_lock is not None:
            # ── Leader path ────────────────────────────────────────────────
            self._leader_lock = leader_lock
            logger.info("scry: acquired leader lock; starting as leader")

            # Step 3a: Open DB read-write FIRST (build all the deps the IPC
            # handler will need before binding the socket).
            db = ScryDB(self._repo_root)

            # Step 3b: Build git context + overlay manager + embedder.
            git_context = GitContextProvider.from_config(self._repo_root, cfg.index)
            overlay_mgr = OverlayManager(self._repo_root, git_context=git_context)
            embedder = make_embedder(cfg.embeddings)

            # Step 3c: Build indexer + run initial reconciliation.
            indexer = Indexer(
                repo_root=self._repo_root,
                config=cfg,
                db=db,
                embedder=embedder,
                git_context=git_context,
                allow_untrusted=self._allow_untrusted_lsp_config,
            )
            await indexer.index_async(force=False)

            # Step 3d: Crash recovery for any prior commit-links transaction
            # interrupted before its marker was deleted.
            overlay_mgr.recover_pending()

            # Step 3e: Build context — must be ready BEFORE the IPC handler
            # is constructed because the handler closes over it.
            ctx = MCPContext(
                repo_root=self._repo_root,
                config=cfg,
                db=db,
                embedder=embedder,
                git_context=git_context,
                overlay_mgr=overlay_mgr,
                indexer=indexer,
                role="leader",
                ipc_client=None,
                index_state_tracker=IndexStateTracker(),
            )

            # Step 3f: Build the REAL IPC handler with ctx already wired in.
            # Earlier versions used a placeholder handler at IPCServer.start()
            # time and tried to mutate self._ipc_server._handler afterwards;
            # that mutation was silently ignored on Linux/macOS because
            # _start_unix captures the handler into a local closure variable
            # that the connection callback pins for its lifetime. The
            # corrected ordering builds the real handler first, then binds.
            # (review-w2i BLOCKING #1 fix.)
            ipc_handler = self._make_ipc_handler(ctx)

            # Step 3g: Bind IPC endpoint with the real handler from the
            # start (Windows: NotImplementedError → single-leader fallback).
            endpoint_uri: str | None = None
            try:
                endpoint_uri = derive_endpoint_uri(self._repo_root)
                ipc_server_inst = IPCServer(
                    self._repo_root,
                    handler=ipc_handler,
                    config=cfg.ipc,
                )
                await ipc_server_inst.start()
                self._ipc_server = ipc_server_inst
            except NotImplementedError:
                logger.info(
                    "scry: IPC server not available on this platform; running in single-leader mode"
                )
                endpoint_uri = None

            # Step 3h: Write leader metadata (only when IPC is available
            # AND the listener is accepting per §10.2 v3.1 ordering).
            if endpoint_uri is not None:
                leader_lock.write_metadata(
                    endpoint_uri=endpoint_uri,
                    scry_version=_SCRY_VERSION,
                )

            self._ctx = ctx

        else:
            # ── Follower path ──────────────────────────────────────────────
            logger.info("scry: leader lock held by another process; starting as follower")

            # Step 4a: Read leader metadata.
            leader_metadata = read_leader_metadata_if_present(self._repo_root)

            # Step 4b: Connect IPC client.
            ipc_client: IPCClient | None = None
            if leader_metadata is not None and leader_metadata.endpoint_uri is not None:
                try:
                    spec: EndpointSpec = parse_endpoint_uri(
                        leader_metadata.endpoint_uri, self._repo_root
                    )
                    ipc_client = IPCClient(spec, config=cfg.ipc)
                except (ValueError, NotImplementedError) as exc:
                    logger.warning("scry: follower cannot connect to leader IPC: %s", exc)

            # Step 4c: Open DB read-only.
            db = ScryDB(self._repo_root, read_only=True)

            # Step 4d: Build supporting objects (no indexer for followers).
            git_context = GitContextProvider.from_config(self._repo_root, cfg.index)
            overlay_mgr = OverlayManager(self._repo_root, git_context=git_context)
            embedder = make_embedder(cfg.embeddings)

            ctx = MCPContext(
                repo_root=self._repo_root,
                config=cfg,
                db=db,
                embedder=embedder,
                git_context=git_context,
                overlay_mgr=overlay_mgr,
                indexer=None,
                role="follower",
                ipc_client=ipc_client,
                index_state_tracker=IndexStateTracker(),
            )
            self._ctx = ctx

        # Mark the server started ONLY after the entire cold-start sequence
        # has succeeded — failures above leave _started=False so retries
        # can proceed cleanly (review-w2i MEDIUM #3 fix).
        self._started = True

    async def stop(self) -> None:
        """Stop the server cleanly.

        Releases the leader lock (if held), stops the IPC server, and closes
        the database connection.

        Each cleanup step is wrapped in its own ``try`` so a failure in one
        does not prevent the others — important when ``MCPServer`` is
        embedded in a longer-running host (Wave 6 ``scry watch``) where a
        leaked leader lock would block clean handoff (review-w2i
        MEDIUM #2 fix).
        """
        if self._ipc_server is not None:
            ipc_server = self._ipc_server
            self._ipc_server = None
            try:
                await ipc_server.stop()
            except Exception:
                logger.exception("scry: IPC server stop raised; continuing cleanup")

        if self._leader_lock is not None:
            leader_lock = self._leader_lock
            self._leader_lock = None
            try:
                leader_lock.release()
            except Exception:
                logger.exception("scry: LeaderLock release raised; continuing cleanup")

        if self._ctx is not None and self._ctx.db is not None:
            with contextlib.suppress(Exception):
                self._ctx.db.close()

        self._ctx = None
        self._started = False

    async def serve_stdio(self) -> None:
        """Run the FastMCP stdio loop.

        Blocks until stdin closes (i.e. the MCP client disconnects).
        Calls :meth:`start` first if not yet started.
        """
        if not self._started:
            await self.start()
        await self._mcp.run_stdio_async(show_banner=False)
