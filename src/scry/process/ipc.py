"""IPC transport — Unix socket (Linux/macOS) or Windows named pipe (stub).

Implements the leader/follower JSON-over-stream protocol described in
DESIGN.md §10.3 v3.1 and §10.5 for scry's multi-process coordination layer.

Public API summary::

    EndpointSpec          - platform-specific parsed connection address
    derive_endpoint_uri   - compute canonical URI for this repo's leader
    parse_endpoint_uri    - parse a URI from leader lock-file metadata
    IPCRequest            - follower -> leader wire message
    IPCResponse           - leader -> follower wire message
    IPCHandler            - Callable[[IPCRequest], Awaitable[IPCResponse]]
    WRITE_OPS             - ops that require an idempotency_token
    IPCServer             - leader-side listener (Unix; NotImplementedError on Windows)
    IPCClient             - follower-side connector

Idempotency (DESIGN.md §10.3 v3.1):
    Write ops (WRITE_OPS) carry a ``tok_<...>`` idempotency token. The leader
    maintains an LRU cache (default 10 000 entries) keyed by token; duplicate
    tokens receive the cached response without re-executing the handler.

Per-op timeouts (DESIGN.md §10.3 v3.1):
    Short ops (propose_link, accept_link, status): default 5 s.
    Long ops (commit_links, reindex): no timeout; heartbeat every 10 s keeps
    the connection alive (Wave 2 tests verify the timeout path; full heartbeat
    loop lands in Wave 6 scry watch).

Windows IPC (DESIGN.md §10.5):
    Named-pipe support (pywin32 + restrictive DACL) is **deferred to Wave 6**
    (scry watch).  ``IPCServer.start()`` and ``IPCClient.call()`` raise
    ``NotImplementedError`` on Windows.  The single-leader mode (no followers)
    remains fully functional on Windows without IPC.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import struct
import sys
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from scry.models import IPCConfig

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────

#: Connections sending a single JSON line larger than this are rejected (DoS guard).
MAX_MESSAGE_BYTES: int = 1_048_576  # 1 MiB

#: asyncio StreamReader internal buffer size.
RECV_BUFFER_SIZE: int = 65_536

_IS_WINDOWS: bool = sys.platform == "win32"

# Compiled pattern matching the tok_<...> idempotency token format (models.py).
_TOKEN_RE: re.Pattern[str] = re.compile(r"^tok_[A-Za-z0-9_-]+$")

# ─── Write-op set ─────────────────────────────────────────────────────

#: Operations that require an ``idempotency_token`` and benefit from the LRU
#: cache.  Read ops bypass the cache entirely (DESIGN.md §10.3 v3.1).
WRITE_OPS: frozenset[str] = frozenset({"propose_link", "accept_link", "commit_links", "reindex"})

# ─── Endpoint URI helpers ─────────────────────────────────────────────


@dataclass(frozen=True)
class EndpointSpec:
    """Platform-specific connection address parsed from a leader URI.

    Attributes:
        scheme: ``"unix"`` (Linux/macOS) or ``"pipe"`` (Windows).
        address: Absolute socket path (unix) or ``\\\\.\\pipe\\<name>`` (pipe).
    """

    scheme: Literal["unix", "pipe"]
    address: str


def derive_endpoint_uri(repo_root: Path) -> str:
    """Compute the canonical endpoint URI for this repo's leader process.

    On Linux/macOS: ``unix:<abs-path-to-.scry/scry.sock>``
    On Windows:     ``pipe:scry-<sha256[:16] of resolved repo path>``

    The URI is written to the leader lock-file by the leader process (W2g)
    and parsed back by followers via :func:`parse_endpoint_uri`.

    Args:
        repo_root: Repository root directory (used to locate the socket file).

    Returns:
        URI string, e.g. ``"unix:/home/user/proj/.scry/scry.sock"`` or
        ``"pipe:scry-a1b2c3d4e5f67890"``.
    """
    if _IS_WINDOWS:
        digest = hashlib.sha256(str(repo_root.resolve()).encode()).hexdigest()
        return f"pipe:scry-{digest[:16]}"
    sock_path = repo_root / ".scry" / "scry.sock"
    return f"unix:{sock_path}"


def parse_endpoint_uri(uri: str, repo_root: Path) -> EndpointSpec:
    """Parse a leader URI (from lock-file metadata) into an :class:`EndpointSpec`.

    Supported schemes:

    * ``unix:<path>`` — path is resolved relative to *repo_root* when relative.
    * ``pipe:<name>`` — the ``\\\\.\\pipe\\`` prefix is prepended automatically.

    Args:
        uri: Raw URI string from the leader metadata file (W2g).
        repo_root: Used to resolve relative unix paths.

    Returns:
        :class:`EndpointSpec` ready to pass to :class:`IPCClient`.

    Raises:
        ValueError: If the scheme is unknown or the URI is otherwise malformed.
    """
    if uri.startswith("unix:"):
        raw = uri[len("unix:") :]
        path = Path(raw)
        if not path.is_absolute():
            path = repo_root / path
        return EndpointSpec(scheme="unix", address=str(path))
    if uri.startswith("pipe:"):
        name = uri[len("pipe:") :]
        if not name:
            raise ValueError(f"Empty pipe name in URI: {uri!r}")
        return EndpointSpec(scheme="pipe", address=rf"\\.\pipe\{name}")
    raise ValueError(f"Unknown IPC URI scheme (expected 'unix:' or 'pipe:'): {uri!r}")


# ─── Wire protocol types ──────────────────────────────────────────────


@dataclass(frozen=True)
class IPCRequest:
    """Follower → leader request envelope (DESIGN.md §10.3).

    Attributes:
        request_id: Per-connection monotonic sequence number assigned by the
            client; echoed back in the response.
        op: Tool name — ``propose_link``, ``accept_link``, ``commit_links``,
            ``reindex``, ``status``, ``search``, etc.
        args: Tool-specific keyword arguments (mirrors the MCP tool surface).
        idempotency_token: Required for :data:`WRITE_OPS`; ``None`` for reads.
            Must match ``tok_[A-Za-z0-9_-]+``.
        protocol_version: Wire format version — ``1`` for this release.
    """

    request_id: int
    op: str
    args: dict[str, Any]
    idempotency_token: str | None = None
    protocol_version: int = 1


@dataclass(frozen=True)
class IPCResponse:
    """Leader → follower response envelope (DESIGN.md §10.3).

    Attributes:
        request_id: Echoed from the originating :class:`IPCRequest`.
        ok: ``True`` on success, ``False`` on any error.
        result: Arbitrary payload on success; ``None`` on error.
        error: Human-readable description; ``None`` on success.
        error_type: Machine-readable category:
            ``"validation"`` | ``"timeout"`` | ``"auth"`` |
            ``"internal"`` | ``"oversized"``.
    """

    request_id: int
    ok: bool
    result: Any = None
    error: str | None = None
    error_type: str | None = None


#: Type alias for the handler callable passed to :class:`IPCServer`.
IPCHandler: TypeAlias = Callable[[IPCRequest], Awaitable[IPCResponse]]

# ─── Idempotency LRU cache ────────────────────────────────────────────


class _IdempotencyCache:
    """Bounded LRU cache: ``idempotency_token → IPCResponse``.

    Single-threaded (asyncio event loop only); no locking required.
    Capacity is :attr:`IPCConfig.idempotency_cache_size` (default 10 000).
    Evicts the least-recently-used entry when at capacity.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, IPCResponse] = OrderedDict()

    def get(self, key: str) -> IPCResponse | None:
        """Return cached response for *key*, promoting it to MRU. ``None`` on miss."""
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: IPCResponse) -> None:
        """Store *value* for *key*, evicting the LRU entry when at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


# ─── Framing helpers ──────────────────────────────────────────────────


def _encode_response(resp: IPCResponse) -> bytes:
    """Serialize *resp* to a compact newline-terminated JSON line."""
    payload: dict[str, Any] = {"id": resp.request_id, "ok": resp.ok}
    if resp.result is not None:
        payload["result"] = resp.result
    if resp.error is not None:
        payload["error"] = resp.error
    if resp.error_type is not None:
        payload["error_type"] = resp.error_type
    return json.dumps(payload, separators=(",", ":")).encode() + b"\n"


def _decode_request(raw: bytes) -> IPCRequest | None:
    """Parse a JSON line into an :class:`IPCRequest`. Returns ``None`` on error."""
    try:
        d: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        return None
    try:
        token = d.get("idempotency_token")
        if token is not None and not isinstance(token, str):
            return None
        return IPCRequest(
            request_id=int(d["id"]),
            op=str(d["op"]),
            args=dict(d.get("args") or {}),
            idempotency_token=token,
            protocol_version=int(d.get("protocol_version", 1)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _validate_token(token: str) -> bool:
    """Return ``True`` iff *token* matches the ``tok_<alphanum>`` format."""
    return bool(_TOKEN_RE.match(token))


# ─── Per-connection handler ───────────────────────────────────────────


class _ConnectionHandler:
    """Handles one accepted connection on the leader side.

    Verifies peer UID (Linux: SO_PEERCRED), reads newline-framed JSON
    requests, dispatches to the :class:`IPCHandler` with idempotency and
    timeout logic, then writes responses back.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        handler: IPCHandler,
        cache: _IdempotencyCache,
        overlay_locks: dict[str, asyncio.Lock],
        config: IPCConfig,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._handler = handler
        self._cache = cache
        self._overlay_locks = overlay_locks
        self._config = config

    def _verify_peer_uid(self) -> bool:
        """Return ``False`` (and log) if the peer UID differs from ours.

        Uses ``SO_PEERCRED`` on Linux (DESIGN.md §10.3 security requirement).
        On macOS and other Unix systems, the mode-0600 socket file is the
        primary access guard; this method returns ``True`` unconditionally.

        On Linux: fails CLOSED if SO_PEERCRED itself errors. The previous
        fail-open behaviour (review-w2h LOW finding) defeated the
        "leader rejects any connection whose SO_PEERCRED UID does not
        match its own UID" guarantee whenever the syscall failed for
        any reason — kernel quirks, getsockopt failure, etc. Failing
        closed is the correct interpretation of the §10.3 contract.
        """
        if platform.system() != "Linux":
            return True
        try:
            import socket as _socket

            sock: _socket.socket | None = self._writer.get_extra_info("socket")
            if sock is None:
                # No underlying socket — cannot verify; reject conservatively.
                log.warning("IPC: cannot verify peer UID (no socket); rejecting")
                return False
            # SO_PEERCRED: struct { pid_t pid; uid_t uid; gid_t gid; }
            raw = sock.getsockopt(
                _socket.SOL_SOCKET,
                _socket.SO_PEERCRED,  # type: ignore[attr-defined]
                struct.calcsize("3i"),
            )
            _pid, uid, _gid = struct.unpack("3i", raw)
            our_uid: int = os.getuid()  # type: ignore[attr-defined]
            if uid != our_uid:
                log.warning(
                    "IPC: rejecting connection from uid=%d (expected %d)",
                    uid,
                    our_uid,
                )
                return False
        except Exception:
            # SO_PEERCRED failure on Linux: fail closed.
            log.warning("IPC: SO_PEERCRED check failed; rejecting connection", exc_info=True)
            return False
        return True

    async def run(self) -> None:
        """Read / dispatch / write loop. Cleans up the connection on exit."""
        if not self._verify_peer_uid():
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            return
        try:
            while True:
                try:
                    line = await self._reader.readline()
                except asyncio.LimitOverrunError:
                    log.warning("IPC: message exceeds buffer limit, closing")
                    await self._send_error(-1, "Message exceeds 1 MiB limit", "oversized")
                    break
                except (asyncio.IncompleteReadError, ConnectionError, OSError):
                    break

                if not line:
                    break  # clean EOF

                if len(line) > MAX_MESSAGE_BYTES:
                    log.warning("IPC: oversized message (%d bytes), closing", len(line))
                    await self._send_error(-1, "Message exceeds 1 MiB limit", "oversized")
                    break

                req = _decode_request(line)
                if req is None:
                    log.warning("IPC: malformed request, closing connection")
                    break

                resp = await self._dispatch(req)
                try:
                    self._writer.write(_encode_response(resp))
                    await self._writer.drain()
                except (ConnectionError, OSError):
                    break
        finally:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

    async def _send_error(self, request_id: int, error: str, error_type: str) -> None:
        """Best-effort: write an error response then let the caller break."""
        resp = IPCResponse(request_id=request_id, ok=False, error=error, error_type=error_type)
        try:
            self._writer.write(_encode_response(resp))
            await self._writer.drain()
        except Exception:
            pass

    async def _dispatch(self, req: IPCRequest) -> IPCResponse:
        """Apply idempotency / validation / timeout, then call the handler."""
        is_write = req.op in WRITE_OPS

        if is_write:
            token = req.idempotency_token
            if token is None:
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Write op '{req.op}' requires idempotency_token",
                    error_type="validation",
                )
            if not _validate_token(token):
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Invalid idempotency_token format: {token!r}",
                    error_type="validation",
                )
            cached = self._cache.get(token)
            if cached is not None:
                log.debug("IPC: idempotency cache hit token=%s", token)
                # Return cached payload; update request_id to the current one.
                return IPCResponse(
                    request_id=req.request_id,
                    ok=cached.ok,
                    result=cached.result,
                    error=cached.error,
                    error_type=cached.error_type,
                )

        is_long_op = req.op in {"commit_links", "reindex"}
        server_timeout: float | None = None if is_long_op else self._config.timeouts.short

        try:
            if server_timeout is not None:
                resp = await asyncio.wait_for(self._handler(req), timeout=server_timeout)
            else:
                resp = await self._handler(req)
        except TimeoutError:
            return IPCResponse(
                request_id=req.request_id,
                ok=False,
                error=f"Op '{req.op}' timed out on server after {server_timeout}s",
                error_type="timeout",
            )
        except asyncio.CancelledError:
            raise  # propagate cancellation (stop() in progress)
        except Exception as exc:
            log.exception("IPC: handler raised for op=%s", req.op)
            return IPCResponse(
                request_id=req.request_id,
                ok=False,
                error=str(exc),
                error_type="internal",
            )

        # Cache the result so repeat tokens get the same response.
        if is_write and req.idempotency_token is not None:
            self._cache.put(req.idempotency_token, resp)

        return resp


# ─── IPCServer ────────────────────────────────────────────────────────


class IPCServer:
    """Leader-side JSON-over-stream listener (DESIGN.md §10.3 v3.1).

    Owns the Unix socket at ``.scry/scry.sock`` (Linux/macOS) or the Windows
    named pipe (stubbed — deferred to Wave 6). Also owns:

    * The **idempotency LRU cache** bounded by
      :attr:`IPCConfig.idempotency_cache_size` (§10.3).
    * **Per-overlay asyncio locks** for write serialization (§10.1) — the
      handler acquires ``get_overlay_lock(path)`` before appending to a JSONL
      file so concurrent IPC requests cannot interleave bytes.

    Lifecycle::

        srv = IPCServer(repo_root, handler=my_handler)
        await srv.start()   # binds; must complete BEFORE write_metadata() (§10.2)
        ...
        await srv.stop()    # stops accepting, cancels in-flight handlers, removes socket

    Windows:
        :meth:`start` raises :exc:`NotImplementedError` — named-pipe support
        (pywin32 + restrictive DACL, §10.5) is deferred to Wave 6 (scry watch).
        scry operates in single-leader mode on Windows without IPC.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        handler: IPCHandler,
        config: IPCConfig | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._handler = handler
        self._config = config or IPCConfig()
        self._cache = _IdempotencyCache(self._config.idempotency_cache_size)
        self._overlay_locks: dict[str, asyncio.Lock] = {}
        self._server: asyncio.AbstractServer | None = None
        self._connection_tasks: set[asyncio.Task[Any]] = set()
        self._uri: str = derive_endpoint_uri(repo_root)

    @property
    def endpoint_uri(self) -> str:
        """URI to advertise in the leader lock-file metadata (W2g format)."""
        return self._uri

    def get_overlay_lock(self, overlay_path: str) -> asyncio.Lock:
        """Return the asyncio.Lock for *overlay_path*, creating it if needed.

        The handler acquires this lock before writing to a JSONL overlay file
        so concurrent IPC requests cannot interleave bytes (§10.1).

        Args:
            overlay_path: Absolute or canonical path to the overlay file.

        Returns:
            A per-path :class:`asyncio.Lock` scoped to this leader's lifetime.
        """
        if overlay_path not in self._overlay_locks:
            self._overlay_locks[overlay_path] = asyncio.Lock()
        return self._overlay_locks[overlay_path]

    async def start(self) -> None:
        """Bind the endpoint and begin accepting connections.

        The endpoint is ready to accept before this coroutine returns, satisfying
        the ordering requirement in DESIGN.md §10.2 v3.1: the leader must bind
        its IPC endpoint *before* writing the metadata file.

        Raises:
            NotImplementedError: Always on Windows — named-pipe support is
                deferred to Wave 6.  See module docstring for details.
        """
        if _IS_WINDOWS:
            raise NotImplementedError(
                "Windows IPC requires pywin32 and a restrictive DACL — "
                "install via `pip install pywin32` and rerun. "
                "Full Windows named-pipe support is deferred to Wave 6 "
                "(scry watch). scry operates in single-leader mode on Windows."
            )
        await self._start_unix()

    async def _start_unix(self) -> None:
        """Bind the Unix socket and start the accept loop."""
        spec = parse_endpoint_uri(self._uri, self._repo_root)
        sock_path = spec.address

        # Remove stale socket file from a previous crashed leader.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(sock_path)

        Path(sock_path).parent.mkdir(parents=True, exist_ok=True)

        # Capture instance state for the closure (avoids holding self).
        handler = self._handler
        cache = self._cache
        overlay_locks = self._overlay_locks
        config = self._config
        connection_tasks = self._connection_tasks

        async def _client_cb(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task: asyncio.Task[Any] | None = asyncio.current_task()
            if task is not None:
                connection_tasks.add(task)
            try:
                conn = _ConnectionHandler(reader, writer, handler, cache, overlay_locks, config)
                await conn.run()
            finally:
                if task is not None:
                    connection_tasks.discard(task)

        # Restrictive umask wraps start_unix_server so the socket is
        # created with mode 0600 from the first instant — closes the
        # TOCTOU window on macOS where SO_PEERCRED is unavailable and
        # socket mode is the only access guard (review-w2h MEDIUM fix).
        prior_umask = os.umask(0o077)
        try:
            self._server = await asyncio.start_unix_server(  # type: ignore[attr-defined]
                _client_cb,
                path=sock_path,
                limit=MAX_MESSAGE_BYTES + 1,
            )
            # Belt-and-braces: explicit chmod even though umask should
            # have handled it, since asyncio's bind path may differ
            # from a vanilla socket() + bind().
            os.chmod(sock_path, 0o600)
        finally:
            os.umask(prior_umask)
        log.info("IPC: listening at %s", sock_path)

    async def stop(self) -> None:
        """Stop accepting, cancel in-flight handlers, and remove the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Cancel all in-flight connection tasks and wait for them to exit.
        tasks = list(self._connection_tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._connection_tasks.clear()

        if not _IS_WINDOWS:
            spec = parse_endpoint_uri(self._uri, self._repo_root)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(spec.address)

        log.info("IPC: server stopped")


# ─── IPCClient ────────────────────────────────────────────────────────


class IPCClient:
    """Follower-side connector to the leader's IPC endpoint (DESIGN.md §10.3 v3.1).

    Maintains a persistent connection to the leader. For write operations
    (:data:`WRITE_OPS`) the caller **must** supply an *idempotency_token*
    so the leader can deduplicate retries across transient failures.

    Concurrency model:
        :meth:`call` is safe to invoke from multiple coroutines on the SAME
        instance — an internal :class:`asyncio.Lock` serialises the
        send-then-receive critical section so concurrent callers cannot
        race on the shared stream and silently swap each other's responses.
        The lock also guards the response-id check that defends against
        any unexpected wire-protocol drift.

    Timeouts:

    * Short ops (``propose_link``, ``accept_link``, ``status``): default to
      :attr:`IPCConfig.timeouts.short` (5 s) when ``timeout_seconds=None``.
    * Long ops (``commit_links``, ``reindex``): pass ``timeout_seconds=None``
      explicitly; the connection stays open for the duration and the heartbeat
      mechanism keeps it alive (Wave 6 scry watch).

    Windows:
        :meth:`call` raises :exc:`NotImplementedError` (deferred to Wave 6).
        Followers on Windows serve all read tools directly from the read-only DB.
    """

    def __init__(
        self,
        endpoint_spec: EndpointSpec,
        *,
        config: IPCConfig | None = None,
    ) -> None:
        self._spec = endpoint_spec
        self._config = config or IPCConfig()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._request_id: int = 0
        # Serializes the send-then-receive critical section so concurrent
        # call() invocations on the same instance cannot interleave writes
        # OR receive each other's responses (review-w2h HIGH fix).
        self._call_lock = asyncio.Lock()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open the transport connection if not already open."""
        if self._reader is not None and self._writer is not None:
            return self._reader, self._writer
        if self._spec.scheme == "pipe":
            raise NotImplementedError(
                "Windows IPC requires pywin32 — deferred to Wave 6 (scry watch)."
            )
        reader, writer = await asyncio.open_unix_connection(  # type: ignore[attr-defined]
            self._spec.address,
            limit=MAX_MESSAGE_BYTES + 1,
        )
        self._reader, self._writer = reader, writer
        return reader, writer

    async def call(
        self,
        op: str,
        args: dict[str, Any],
        *,
        idempotency_token: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        """Send an IPC request and return the leader's ``result`` payload.

        ``timeout_seconds=None`` applies the auto-default: short ops receive
        :attr:`IPCConfig.timeouts.short`; long ops (``commit_links``,
        ``reindex``) receive no timeout.

        Args:
            op: Tool name — e.g. ``"propose_link"``, ``"status"``.
            args: Tool-specific arguments (mirrors the MCP tool surface).
            idempotency_token: Required for write ops (:data:`WRITE_OPS`).
                Must match ``tok_[A-Za-z0-9_-]+``.
            timeout_seconds: Override per-op timeout. Pass ``None`` to use the
                auto-default (see above); pass a float to override explicitly.

        Returns:
            The ``result`` field from the leader's successful response.

        Raises:
            asyncio.TimeoutError: If the per-op timeout elapses waiting for the
                response. The connection is closed on timeout.
            RuntimeError: If the leader returns ``ok: false`` (the exception
                message contains the leader's ``error`` field) OR if the
                response's ``id`` does not match the request's (wire-protocol
                drift defense).
            NotImplementedError: On Windows (deferred to Wave 6).
        """
        async with self._call_lock:
            self._request_id += 1
            req_id = self._request_id

            payload: dict[str, Any] = {
                "id": req_id,
                "op": op,
                "args": args,
                "protocol_version": 1,
            }
            if idempotency_token is not None:
                payload["idempotency_token"] = idempotency_token

            raw = json.dumps(payload, separators=(",", ":")).encode() + b"\n"

            is_long_op = op in {"commit_links", "reindex"}
            effective_timeout: float | None = (
                timeout_seconds
                if timeout_seconds is not None
                else (None if is_long_op else self._config.timeouts.short)
            )

            reader, writer = await self._connect()
            writer.write(raw)
            await writer.drain()

            try:
                if effective_timeout is not None:
                    line = await asyncio.wait_for(reader.readline(), timeout=effective_timeout)
                else:
                    line = await reader.readline()
            except TimeoutError:
                # Close connection: stream state is unknown after a timeout.
                await self.close()
                raise

            if not line:
                await self.close()
                raise RuntimeError("IPC: server closed connection unexpectedly")

            try:
                resp_d: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                await self.close()
                raise RuntimeError(f"IPC: invalid JSON in leader response: {exc}") from exc

            # Wire-protocol invariant: the leader echoes our request id.
            # If it doesn't, our stream state is corrupt — close and raise
            # rather than silently returning someone else's response.
            resp_id = resp_d.get("id")
            if resp_id != req_id:
                await self.close()
                raise RuntimeError(
                    f"IPC: response id mismatch (expected {req_id}, got {resp_id!r}); "
                    "stream state is corrupt"
                )

            if not resp_d.get("ok"):
                raise RuntimeError(str(resp_d.get("error") or "IPC error"))

            return resp_d.get("result")

    async def close(self) -> None:
        """Close the underlying transport connection gracefully."""
        writer = self._writer
        self._writer = None
        self._reader = None
        if writer is not None:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()


# ─── Public re-exports ────────────────────────────────────────────────

__all__ = [
    "MAX_MESSAGE_BYTES",
    "RECV_BUFFER_SIZE",
    "WRITE_OPS",
    "EndpointSpec",
    "IPCClient",
    "IPCHandler",
    "IPCRequest",
    "IPCResponse",
    "IPCServer",
    "derive_endpoint_uri",
    "parse_endpoint_uri",
]
