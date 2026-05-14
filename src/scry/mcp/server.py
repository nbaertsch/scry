"""FastMCP-based MCP server with leader/follower coordination (workstream W2i).

This module wires together all Wave 2 workstreams into a single FastMCP
server process.  It implements the cold-start sequence described in
DESIGN.md §10.4 v3.1 and the leader/follower IPC dispatch from §10.3.

Cold-start sequence (``MCPServer.start()``)
-------------------------------------------
1. Load ``.scry/config.yaml``; raise :class:`~scry.index.IndexerError` if absent.
2. Try to acquire :class:`~scry.process.leader.LeaderLock`.
3. **Leader path:**
   a. Bind the IPC endpoint (:class:`~scry.process.ipc.IPCServer`).  Both
      Unix domain sockets and Windows named pipes are now supported (W6b);
      the leader fails fast on bind errors instead of silently falling
      back to single-process mode.
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

import asyncio
import contextlib
import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated, Any, cast

from fastmcp import FastMCP
from pydantic import Field

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


# ─── SR4-4: strip noisy tracebacks for expected validation errors ────────
#
# Background: the MCP host (Claude Desktop / Copilot CLI) renders the
# scry stdio process's stderr verbatim.  Any ``logger.exception(...)``
# call that fires for an EXPECTED validation error (MCPServerError,
# pydantic.ValidationError) emits a multi-line Python traceback that
# looks alarming to users even though the protocol response is clean.
#
# Fix: install a logging.Filter on the root handlers that strips
# ``exc_info`` AND ``exc_text`` from records whose exception is a
# known-expected type AND whose logger name lives in the scry / mcp /
# fastmcp namespaces.  The MESSAGE survives (operators still see e.g.
# "validation: ..."); only the noisy stack is suppressed.
#
# Per code-review feedback (sr4-4-plan):
#   * handler-level filter (not named-logger filter) — Python logging
#     does NOT apply ancestor filters to descendant emissions, so
#     hooking the named loggers wouldn't catch fastmcp.server.server.
#   * narrow scope by both exception type AND logger name prefix.
#   * strip both exc_info and exc_text (the latter can be cached).
def _expected_validation_exception_types() -> tuple[type[BaseException], ...]:
    """Resolve the expected-error exception classes.

    Lazy import + best-effort fallback so the filter still installs even
    when ``pydantic_core`` is unavailable in some odd environment.
    """
    types_: list[type[BaseException]] = [MCPServerError]
    try:
        import pydantic

        types_.append(pydantic.ValidationError)
    except Exception:
        pass
    try:
        import pydantic_core

        types_.append(pydantic_core.ValidationError)
    except Exception:
        pass
    return tuple(types_)


_SCRY_LOGGER_NAME_PREFIXES: tuple[str, ...] = ("scry", "mcp", "fastmcp")


class _ExpectedErrorTracebackFilter(logging.Filter):
    """SR4-4: strip exc_info / exc_text for expected validation errors.

    Only fires when BOTH conditions hold:
      1. The record's logger name starts with ``scry``, ``mcp`` or
         ``fastmcp`` (so non-MCP code paths are untouched).
      2. The exception type is in
         :func:`_expected_validation_exception_types`.

    Returns True (don't drop the record) — only the stack is removed,
    so operators still see the message text.
    """

    def __init__(self) -> None:
        super().__init__()
        self._expected_types = _expected_validation_exception_types()

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(_SCRY_LOGGER_NAME_PREFIXES):
            return True
        exc_info = record.exc_info
        if exc_info is None:
            return True
        exc_type, _exc, _tb = exc_info
        if exc_type is None or not issubclass(exc_type, self._expected_types):
            return True
        # Strip both — exc_text is sometimes cached from a prior format()
        # call on the same record (rare but real).
        record.exc_info = None
        record.exc_text = None
        return True


def _install_traceback_filter() -> None:
    """Attach :class:`_ExpectedErrorTracebackFilter` to every root
    handler exactly once.  Idempotent so repeated MCPServer starts
    don't stack duplicate filters.

    Per code-review (sr4-4-code-review): when the root logger has NO
    handlers we attach the filter to ``logging.lastResort`` instead of
    creating a fresh ``StreamHandler``.  The latter would silently
    no-op the embedding app's later ``logging.basicConfig(...)``
    (basicConfig only configures root when it has no handlers).
    """
    sentinel_attr = "_scry_sr4_4_filter_installed"
    root_logger = logging.getLogger()
    filt = _ExpectedErrorTracebackFilter()
    for handler in root_logger.handlers:
        if not getattr(handler, sentinel_attr, False):
            handler.addFilter(filt)
            setattr(handler, sentinel_attr, True)
    last_resort = logging.lastResort
    if last_resort is not None and not getattr(last_resort, sentinel_attr, False):
        last_resort.addFilter(filt)
        setattr(last_resort, sentinel_attr, True)


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
        # UT3-4: in-process LRU cache for leader-direct write ops keyed by
        # (op, idempotency_token).  Mirrors IPCServer's _IdempotencyCache
        # so duplicate MCP calls via the leader's stdio path are
        # deduplicated the same way IPC follower→leader forwards are.
        self._leader_idem_cache: dict[tuple[str, str, str], Any] = {}
        # SR1-2: per-(op, token, args_hash) lock so concurrent leader-direct
        # calls with the same idempotency_token + args serialize and the
        # second call
        # observes the cached result of the first instead of also
        # executing the handler.  Locks are evicted alongside cache
        # entries to bound memory.
        self._leader_idem_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._mcp: FastMCP = FastMCP(name="scry", version=scry.__version__)
        self._started = False
        # SR4-4: install the expected-error traceback filter so MCP host
        # stderr stays clean for routine validation errors.
        _install_traceback_filter()
        self._register_tools()

    # ─── Tool registration ────────────────────────────────────────────────────

    def _register_tools(self) -> None:
        """Register all 12 MCP tools with the FastMCP instance.

        Each tool is a thin async closure that resolves the current
        :class:`~scry.mcp.handlers.MCPContext` and delegates to
        :func:`_dispatch`.

        UAT-11: tool descriptions are end-user-facing — visible to LLM
        agents via ``tools/list``.  Internal dev notes ("UT3-2 fix",
        "Wave 2 stub", "W6e — DESIGN.md line ...") have been removed
        from the docstrings; rationale is preserved in code comments
        below where appropriate.

        UAT-12: every tool carries explicit ``readOnlyHint`` /
        ``destructiveHint`` annotations so MCP clients (and LLM agents)
        can distinguish safe queries from mutations without parsing
        descriptions.
        """
        from mcp.types import ToolAnnotations

        mcp = self._mcp
        _read = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
        _write = ToolAnnotations(readOnlyHint=False, destructiveHint=True)
        _idem_write = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True)

        @mcp.tool(annotations=_read)
        async def search(
            query: str,
            types: list[str] | None = None,
            top_k: int = 10,
            exclude_tests: bool = False,
        ) -> list[dict[str, Any]]:
            """Hybrid BM25 + vector retrieval over the indexed repository.

            Returns ranked anchor packets.  Use ``types`` to restrict to
            ``["section"]``, ``["code"]``, or ``["code_in_doc"]``.
            SR5-6: pass ``exclude_tests=True`` to suppress anchors from
            test files (filename heuristic) AND test-framework anchors
            (Jest-style ``describe`` / ``it`` / hooks).
            """
            return cast(
                list[dict[str, Any]],
                await self._dispatch(
                    "search",
                    {
                        "query": query,
                        "types": types,
                        "top_k": top_k,
                        "exclude_tests": exclude_tests,
                    },
                ),
            )

        @mcp.tool(annotations=_read)
        async def get_anchor(
            anchor_id: str | None = None,
            id: str | None = None,
        ) -> dict[str, Any] | None:
            """Load a single anchor by its primary ID (full content, no truncation).

            UAT-14: parameter renamed from ``id`` → ``anchor_id`` for
            consistency with ``get_links`` / ``get_callers`` / etc.
            The legacy ``id`` keyword is still accepted (review-u16-18
            HIGH back-compat fix); supply EITHER ``anchor_id`` (preferred)
            OR ``id``.
            """
            resolved = anchor_id if anchor_id is not None else id
            if resolved is None:
                raise MCPServerError("get_anchor requires 'anchor_id' (or legacy 'id')")
            return cast(
                dict[str, Any] | None,
                await self._dispatch("get_anchor", {"id": resolved}),
            )

        @mcp.tool(annotations=_read)
        async def get_links(
            anchor_id: str,
            link_types: list[str] | None = None,
            direction: str = "outgoing",
        ) -> dict[str, Any]:
            """Return active links for an anchor from the baseline ⊕ overlay table.

            Returns ``{"links": [...], "index_state": "..."}`` per §7.3.
            Each link entry includes ``link_type``, drift status, and
            content hashes.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "get_links",
                    {"anchor_id": anchor_id, "link_types": link_types, "direction": direction},
                ),
            )

        @mcp.tool(annotations=_read)
        async def find_drift(
            scope: str | None = None,
            status_filter: list[str] | None = None,
            since: str | None = None,
        ) -> dict[str, Any]:
            """Evaluate section-level drift for active links.

            Returns ``{"entries": [...], "index_state": "..."}`` per §7.3.
            ``scope`` accepts a path-prefix glob; ``status_filter`` is
            an allow-list of drift status values
            (``"fresh"``, ``"code-changed"``, ``"spec-changed"``, etc.).
            ``since`` is a git ref (commit / branch / tag) — only links
            whose endpoints touch files changed in
            ``git diff --name-only <since>..HEAD`` are returned.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "find_drift",
                    {
                        "scope": scope,
                        "status_filter": status_filter,
                        "since": since,
                    },
                ),
            )

        @mcp.tool(annotations=_idem_write)
        async def propose_link(
            from_id: str,
            to_id: str,
            link_type: str,
            evidence: str | None = None,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Stage a new typed link in the current branch overlay.

            Each invocation mints a new link_id; supply
            ``idempotency_token`` to make retries safe (a duplicate
            call with the same token returns the cached response
            without re-executing).
            """
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

        @mcp.tool(annotations=_idem_write)
        async def accept_link(
            proposed_id: str,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Confirm an overlay-staged link proposal.

            Acknowledges a proposal and returns its current state.
            Status persistence is currently a no-op; the call returns
            the proposal record for client-side bookkeeping.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "accept_link",
                    {"proposed_id": proposed_id, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool(annotations=_idem_write)
        async def commit_links(
            scope: str | None = None,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Promote pending overlay records to the baseline link store.

            Returns ``{"promoted": [...], "index_state": "..."}`` per §7.3.
            Use ``scope`` to restrict the promotion to a path prefix.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "commit_links",
                    {"scope": scope, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool(annotations=_idem_write)
        async def unlink(
            link_id: str,
            reason: str | None = None,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Tombstone a link by its ``link_id`` (UAT-M-5 / U-fix-4).

            Appends a DELETE record to the current branch overlay; the
            link no longer appears in ``get_links`` / ``find_drift``.
            Use ``reason`` to record the rationale for the deletion.
            Per DESIGN.md §3.5, a tombstoned ``link_id`` is permanently
            reserved — call ``propose_link`` to re-create a logically
            equivalent link with a fresh ``link_id``.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "unlink",
                    {
                        "link_id": link_id,
                        "reason": reason,
                        "idempotency_token": idempotency_token,
                    },
                ),
            )

        @mcp.tool(annotations=_read)
        async def status() -> dict[str, Any]:
            """Return current server + overlay status (branch, HEAD, pending records)."""
            return cast(dict[str, Any], await self._dispatch("status", {}))

        @mcp.tool(annotations=_read)
        async def repo_summary() -> dict[str, Any]:
            """Return a high-level repository summary with section-level drift score."""
            return cast(dict[str, Any], await self._dispatch("repo_summary", {}))

        @mcp.tool(annotations=_idem_write)
        async def reindex(
            scope: str | None = None,
            force: Annotated[bool, Field(strict=True)] = False,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Trigger an incremental (or forced) re-index of the repository.

            ``force=True`` drops and rebuilds the entire index; use
            sparingly.  ``scope`` is accepted but currently ignored —
            the server logs a warning and surfaces ``scope_ignored: true``
            in the response so callers can detect the limitation.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "reindex",
                    {"scope": scope, "force": force, "idempotency_token": idempotency_token},
                ),
            )

        @mcp.tool(annotations=_read)
        async def get_callers(anchor_id: str, max_depth: int = 1) -> dict[str, Any]:
            """Return symbols that CALL the given code anchor.

            Always returns a ``lsp_status`` field:

            * ``"available"``   — LSP responded successfully.
            * ``"unavailable"`` — LSP not configured / binary not on PATH.
            * ``"unsupported"`` — file's language has no LSP integration in scry.
            * ``"error"``       — LSP responded with an error (see ``lsp_error``).

            An empty ``callers`` list with ``lsp_status != "available"`` means
            the LSP is absent, NOT that the function has no callers.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "get_callers",
                    {"anchor_id": anchor_id, "max_depth": max_depth},
                ),
            )

        @mcp.tool(annotations=_read)
        async def get_subclasses(anchor_id: str) -> dict[str, Any]:
            """Return classes that EXTEND the given class anchor.

            Always returns a ``lsp_status`` field:

            * ``"available"``   — LSP responded successfully.
            * ``"unavailable"`` — LSP not configured / binary not on PATH.
            * ``"unsupported"`` — file's language has no LSP integration in scry.
            * ``"error"``       — LSP responded with an error (see ``lsp_error``).

            An empty ``subclasses`` list with ``lsp_status != "available"``
            means the LSP is absent, NOT that the class has no subclasses.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "get_subclasses",
                    {"anchor_id": anchor_id},
                ),
            )

        @mcp.tool(annotations=_read)
        async def suggest_links_candidates(
            scope: str | None = None,
            source: str = "both",
            limit: int = 25,
        ) -> dict[str, Any]:
            """Surface (code, doc) pair candidates plus a classifier prompt.

            UAT-R5-2: when an LLM-powered MCP client is calling scry,
            requiring scry to ALSO have its own LLM provider is wasteful.
            This tool returns the candidate pairs + system prompt + JSON
            output schema so the calling agent can run the classifier
            itself, then feed the result to ``apply_link_suggestions``.

            No LLM call is made by scry in this path.  Cap ``limit`` to
            keep the payload within the agent's context budget.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "suggest_links_candidates",
                    {"scope": scope, "source": source, "limit": limit},
                ),
            )

        @mcp.tool(annotations=_idem_write)
        async def apply_link_suggestions(
            suggestions: list[dict[str, Any]],
            pair_payloads: list[dict[str, Any]],
            min_confidence: float = 0.7,
            apply: Annotated[bool, Field(strict=True)] = False,
            idempotency_token: str | None = None,
        ) -> dict[str, Any]:
            """Apply (or preview) agent-classified link suggestions.

            Companion to ``suggest_links_candidates``.  When ``apply=False``
            (default) returns the validated, threshold-filtered suggestion
            list WITHOUT writing.  When ``apply=true`` writes each
            surviving suggestion to the current branch overlay.

            ``pair_payloads`` is the ``pairs`` list from the prior
            ``suggest_links_candidates`` call — required to resolve
            ``pair_id`` back to anchor IDs.
            """
            return cast(
                dict[str, Any],
                await self._dispatch(
                    "apply_link_suggestions",
                    {
                        "suggestions": suggestions,
                        "pair_payloads": pair_payloads,
                        "min_confidence": min_confidence,
                        "apply": apply,
                        "idempotency_token": idempotency_token,
                    },
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

        # UT3-4 fix: idempotency for direct MCP calls on the leader.
        # Followers route through IPC which has its own LRU cache; leader
        # write ops invoked directly bypass that cache, so a duplicate
        # ``propose_link`` request with the same idempotency_token used
        # to create two overlay records.  Cache the response in-process
        # keyed by (op, token) so duplicate calls return the cached
        # result (matching IPC semantics).
        #
        # SR1-2 fix: serialize concurrent same-token requests on a
        # per-(op, token) asyncio.Lock so the second request awaits
        # the first instead of racing into the handler.
        #
        # UAT-R5-2 review-r5-1-2 HIGH: include a stable args fingerprint
        # in the cache key so two retries with the same token but
        # different payloads don't share a cached response (a preview
        # apply=false call could otherwise poison a later apply=true
        # call with the same token, returning the preview's empty
        # write count).  We hash args with idempotency_token excluded
        # since the token is already in the key.
        cache_key: tuple[str, str, str] | None = None
        token_arg: str | None = None
        if op in WRITE_OPS and ctx.role == "leader":
            token_arg = args.get("idempotency_token")
            if token_arg:
                args_for_hash = {k: v for k, v in args.items() if k != "idempotency_token"}
                try:
                    args_hash = hashlib.sha256(
                        json.dumps(args_for_hash, sort_keys=True, default=str).encode("utf-8")
                    ).hexdigest()[:16]
                except Exception:
                    # Defensive: an args dict that isn't JSON-serialisable falls
                    # back to no-args-fingerprint behaviour (matches pre-fix).
                    args_hash = ""
                cache_key = (op, token_arg, args_hash)
                lock = self._leader_idem_locks.setdefault(cache_key, asyncio.Lock())
                await lock.acquire()
        try:
            if cache_key is not None:
                cached = self._leader_idem_cache.get(cache_key)
                if cached is not None:
                    logger.debug(
                        "MCP: duplicate token %s for op=%r; returning cached response",
                        token_arg,
                        op,
                    )
                    return cached

            handler = HANDLERS.get(op)
            if handler is None:
                raise MCPServerError(f"Unknown op: {op!r}")

            try:
                result = await handler(ctx, **args)
            except MCPServerError:
                raise
            except Exception as exc:
                logger.exception("Handler raised for op=%s", op)
                raise MCPServerError(f"Internal error in '{op}': {exc}") from exc

            # Cache successful responses for idempotency replay (UT3-4).
            if cache_key is not None:
                self._leader_idem_cache[cache_key] = result
                # Bound the cache to prevent unbounded growth (10k entries
                # mirrors the IPCConfig default).
                if len(self._leader_idem_cache) > 10_000:
                    # Drop oldest 100 entries (cheap dict-based LRU eviction).
                    for key in list(self._leader_idem_cache)[:100]:
                        del self._leader_idem_cache[key]
                        # Drop the matching lock too, but only if no
                        # one is waiting on it (defensive).
                        existing_lock = self._leader_idem_locks.get(key)
                        if existing_lock is not None and not existing_lock.locked():
                            del self._leader_idem_locks[key]

            return result
        finally:
            if cache_key is not None:
                lock_to_release = self._leader_idem_locks.get(cache_key)
                if lock_to_release is not None and lock_to_release.locked():
                    lock_to_release.release()

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

        Known limitation (UT4-3, Windows-only): FastMCP's anyio-backed
        stdio transport does not reliably surface ``stdin`` EOF on
        Windows ``ProactorEventLoop`` — the process can hang for up
        to several seconds after the client disconnects.  POSIX
        platforms detect EOF natively.  Workarounds attempted (a
        secondary watchdog thread reading ``sys.stdin``) failed
        because the watchdog races FastMCP for the underlying handle.
        Real-world impact is bounded: the lock and IPC pipe are
        released by the OS on process exit.
        """
        if not self._started:
            await self.start()
        await self._mcp.run_stdio_async(show_banner=False)
