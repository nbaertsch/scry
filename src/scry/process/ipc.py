"""IPC transport — Unix socket (Linux/macOS) or Windows named pipe.

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
    IPCServer             - leader-side listener (Unix socket or Windows named pipe)
    IPCClient             - follower-side connector

Idempotency (DESIGN.md §10.3 v3.1):
    Write ops (WRITE_OPS) carry a ``tok_<...>`` idempotency token. The leader
    maintains an LRU cache (default 10 000 entries) keyed by token; duplicate
    tokens receive the cached response without re-executing the handler.

Per-op timeouts (DESIGN.md §10.3 v3.1):
    Short ops (propose_link, accept_link, status): default 5 s.
    Long ops (commit_links, reindex): heartbeat every 10 s keeps the connection
    alive; the client enforces a lapse timeout (30 s by default) between
    successive heartbeats/responses to detect a hung server.

Windows IPC (DESIGN.md §10.5, §10.7):
    Named-pipe support via pywin32 with a restrictive DACL (current-user-only)
    is implemented in Wave 6.  The DACL grants GENERIC_READ|GENERIC_WRITE
    exclusively to the current user's SID; cross-user connections are rejected
    at accept time via impersonation + SID comparison.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import platform
import re
import struct
import sys
import threading
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


# ─── Windows pipe concurrency limits (SR2-2) ─────────────────────────
#
# The Windows IPC path wraps every blocking ReadFile/WriteFile in
# ``loop.run_in_executor(_win_pipe_executor, ...)`` so the asyncio
# event loop stays responsive.  Each connected follower holds at
# minimum one executor thread blocked in ``readline()``.
#
# Without a cap this starves the executor — the leader stops accepting
# both new connections AND new request bytes from existing followers
# once concurrent connections exceed pool size.  Originally observed
# at ~12 concurrent followers with the default ThreadPoolExecutor.
#
# Mitigation (pragmatic, not a full IOCP refactor):
#   1. Use a DEDICATED executor sized to ``MAX_WIN_PIPE_IO_WORKERS`` so
#      Windows pipe I/O cannot starve unrelated ``asyncio.to_thread``
#      callers (git diff, file hashing, model load, etc.).
#   2. Cap concurrent connections at ``MAX_WIN_CONNECTIONS``.  When at
#      cap the leader sends a ``connection_rejected`` connection-level
#      frame (see :func:`IPCClient._recv_response`) and closes the
#      pipe, so clients see a clean ``too_many_connections`` error
#      instead of hanging or seeing OSError.
#
# A proper IOCP/FILE_FLAG_OVERLAPPED rewrite remains future work.
def _win_default_cpu_count() -> int:
    # Python 3.13+ exposes process_cpu_count which honors CPU affinity;
    # fall back to os.cpu_count for 3.11/3.12.
    proc_count = getattr(os, "process_cpu_count", None)
    n = proc_count() if proc_count is not None else os.cpu_count()
    return n or 4


_WIN_CPUS: int = _win_default_cpu_count()

#: Hard cap on concurrent followers on Windows.  Bounded to a safe
#: range (4..28) so small machines aren't penalized and very large
#: machines don't keep an absurd number of pipe threads alive.
MAX_WIN_CONNECTIONS: int = max(4, min(28, _WIN_CPUS * 2))

#: Slack threads reserved for the accept loop's ConnectNamedPipe call,
#: per-connection write/SID-check tasks, and the rejection-frame writes
#: themselves.  Without slack a fully-loaded connection set could still
#: starve writes.
WIN_PIPE_RESERVED_THREADS: int = 4

#: Worker count for the dedicated Windows pipe executor.
MAX_WIN_PIPE_IO_WORKERS: int = MAX_WIN_CONNECTIONS + WIN_PIPE_RESERVED_THREADS

# Compiled pattern matching the tok_<...> idempotency token format (models.py).
_TOKEN_RE: re.Pattern[str] = re.compile(r"^tok_[A-Za-z0-9_-]+$")

# ─── Write-op set ─────────────────────────────────────────────────────

#: Operations that require an ``idempotency_token`` and benefit from the LRU
#: cache.  Read ops bypass the cache entirely (DESIGN.md §10.3 v3.1).
WRITE_OPS: frozenset[str] = frozenset(
    {
        "propose_link",
        "accept_link",
        "commit_links",
        # UAT-M-5 / U-fix-4: unlink appends a DELETE record to the
        # current branch overlay — must hit the leader.
        "unlink",
        "reindex",
        # UAT-R5-2: agent-driven suggest-links — apply_link_suggestions is
        # the write half of the two-phase API (suggest_links_candidates is
        # read-only and stays out of WRITE_OPS).
        "apply_link_suggestions",
    }
)

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


class IPCConnectionRejected(RuntimeError):
    """Raised by :class:`IPCClient` when the leader sends a
    ``connection_rejected`` connection-level frame.

    Surfaces the leader's machine-readable ``error_type`` (e.g.
    ``"too_many_connections"``) and a hint for how long to wait
    before retrying.  See SR2-2 in the issue tracker for context.
    """

    def __init__(self, message: str, *, error_type: str, retry_after_ms: int) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.retry_after_ms = retry_after_ms


class _IdempotencyCache:
    """Bounded LRU cache: ``idempotency_token → IPCResponse``.

    Single-threaded (asyncio event loop only); no locking required for
    cache state itself.  However, write handlers can take arbitrary time
    and the event loop will yield while they ``await``, so concurrent
    requests with the SAME idempotency_token can race past the
    ``cache.get`` miss check.  We therefore additionally maintain a
    ``token → asyncio.Lock`` map and require callers (see
    :func:`_run_dispatch_logic`) to acquire the lock around their
    check-then-execute-then-store sequence.

    Capacity is :attr:`IPCConfig.idempotency_cache_size` (default 10 000).
    Evicts the least-recently-used entry when at capacity.

    SR2-1 BLOCKING fix: the per-token lock plugs the TOCTOU window that
    previously let 2-3 concurrent same-token requests all execute the
    handler and then have the second/third writer's response overwrite
    the first via :meth:`put`.
    """

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict[str, IPCResponse] = OrderedDict()
        # SR2-1: per-token serialization locks (separate from cache eviction).
        self._locks: dict[str, asyncio.Lock] = {}

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
            evicted_key, _ = self._store.popitem(last=False)
            # Drop the matching lock IFF nobody is waiting on it.  Defensive:
            # an in-flight handler holding the lock will keep it alive via
            # the dict reference even after eviction.
            existing_lock = self._locks.get(evicted_key)
            if existing_lock is not None and not existing_lock.locked():
                del self._locks[evicted_key]

    def lock_for(self, key: str) -> asyncio.Lock:
        """Return (creating if necessary) the per-token serialization lock."""
        existing = self._locks.get(key)
        if existing is not None:
            return existing
        new_lock = asyncio.Lock()
        self._locks[key] = new_lock
        return new_lock

    def __len__(self) -> int:
        return len(self._store)


# ─── Framing helpers ──────────────────────────────────────────────────


def _encode_response(resp: IPCResponse) -> bytes:
    """Serialize *resp* to a compact newline-terminated JSON line.

    SR2-3: enforces ``MAX_MESSAGE_BYTES`` on the WRITE side so a
    misbehaving handler cannot ship oversized result payloads to
    followers (the read side has the same cap).  When the encoded
    response exceeds the cap, it is replaced with an
    ``error_type="oversized"`` response that fits cleanly.  Reuses
    the request_id so the client can correlate.
    """
    payload: dict[str, Any] = {"id": resp.request_id, "ok": resp.ok}
    if resp.result is not None:
        payload["result"] = resp.result
    if resp.error is not None:
        payload["error"] = resp.error
    if resp.error_type is not None:
        payload["error_type"] = resp.error_type
    encoded = json.dumps(payload, separators=(",", ":")).encode() + b"\n"
    if len(encoded) > MAX_MESSAGE_BYTES:
        log.warning(
            "IPC: response for request_id=%s exceeds %d bytes (%d); "
            "replacing with oversized error response",
            resp.request_id,
            MAX_MESSAGE_BYTES,
            len(encoded),
        )
        oversized = {
            "id": resp.request_id,
            "ok": False,
            "error": (
                f"Response exceeds {MAX_MESSAGE_BYTES} bytes ({len(encoded)} bytes); "
                "narrow the query or paginate the result."
            ),
            "error_type": "oversized",
        }
        encoded = json.dumps(oversized, separators=(",", ":")).encode() + b"\n"
    return encoded


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

                is_long_op = req.op in {"commit_links", "reindex"}
                if is_long_op:
                    hb_task: asyncio.Task[None] = asyncio.create_task(
                        self._heartbeat(self._config.timeouts.long_heartbeat_interval)
                    )
                    try:
                        resp = await self._dispatch(req)
                    finally:
                        hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await hb_task
                else:
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

    async def _heartbeat(self, interval: float) -> None:
        """Send ``{"type":"heartbeat"}`` lines every *interval* seconds."""
        hb_line = json.dumps({"type": "heartbeat"}, separators=(",", ":")).encode() + b"\n"
        while True:
            await asyncio.sleep(interval)
            try:
                self._writer.write(hb_line)
                await self._writer.drain()
            except Exception:
                return

    async def _dispatch(self, req: IPCRequest) -> IPCResponse:
        """Apply idempotency / validation / timeout, then call the handler."""
        return await _run_dispatch_logic(req, self._handler, self._cache, self._config)


# ─── Shared dispatch logic ────────────────────────────────────────────


async def _run_dispatch_logic(
    req: IPCRequest,
    handler: IPCHandler,
    cache: _IdempotencyCache,
    config: IPCConfig,
) -> IPCResponse:
    """Apply idempotency / validation / timeout, then invoke the handler.

    Shared by :class:`_ConnectionHandler` (Unix) and
    :class:`_WinConnectionHandler` (Windows) so business logic is not
    duplicated across transports.
    """
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
        # SR2-1 BLOCKING fix: serialize concurrent same-token requests
        # via the per-token lock so the second request observes the
        # first's cached response instead of also entering the handler.
        # The lock is held for the entire check → execute → store
        # sequence, so the TOCTOU window that previously allowed
        # handler_ran=2-3 is closed.
        async with cache.lock_for(token):
            cached = cache.get(token)
            if cached is not None:
                log.debug("IPC: idempotency cache hit token=%s", token)
                return IPCResponse(
                    request_id=req.request_id,
                    ok=cached.ok,
                    result=cached.result,
                    error=cached.error,
                    error_type=cached.error_type,
                )

            is_long_op = req.op in {"commit_links", "reindex"}
            server_timeout: float | None = None if is_long_op else config.timeouts.short

            try:
                if server_timeout is not None:
                    resp = await asyncio.wait_for(handler(req), timeout=server_timeout)
                else:
                    resp = await handler(req)
            except TimeoutError:
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=f"Op '{req.op}' timed out on server after {server_timeout}s",
                    error_type="timeout",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("IPC: handler raised for op=%s", req.op)
                return IPCResponse(
                    request_id=req.request_id,
                    ok=False,
                    error=str(exc),
                    error_type="internal",
                )

            cache.put(token, resp)
            return resp

    # Non-write path: no idempotency, no lock.
    is_long_op = req.op in {"commit_links", "reindex"}
    server_timeout = None if is_long_op else config.timeouts.short

    try:
        if server_timeout is not None:
            resp = await asyncio.wait_for(handler(req), timeout=server_timeout)
        else:
            resp = await handler(req)
    except TimeoutError:
        return IPCResponse(
            request_id=req.request_id,
            ok=False,
            error=f"Op '{req.op}' timed out on server after {server_timeout}s",
            error_type="timeout",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.exception("IPC: handler raised for op=%s", req.op)
        return IPCResponse(
            request_id=req.request_id,
            ok=False,
            error=str(exc),
            error_type="internal",
        )

    return resp


# ─── Windows pipe I/O helpers ─────────────────────────────────────────


def _win_get_current_user_sid() -> Any:
    """Return the SID for the current user's account (Windows only)."""
    import win32api
    import win32security

    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32security.TOKEN_QUERY,
    )
    user_info = win32security.GetTokenInformation(token, win32security.TokenUser)
    return user_info[0]


def _win_build_pipe_sa() -> Any:
    """Build a SECURITY_ATTRIBUTES that grants current-user-only pipe access.

    The resulting DACL allows GENERIC_READ | GENERIC_WRITE exclusively for the
    current user's SID — no other accounts can open the pipe handle.
    """
    import pywintypes
    import win32file
    import win32security

    sid = _win_get_current_user_sid()

    dacl = win32security.ACL()
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        sid,
    )

    sd = win32security.SECURITY_DESCRIPTOR()
    sd.SetSecurityDescriptorDacl(True, dacl, False)

    sa = pywintypes.SECURITY_ATTRIBUTES()
    sa.SECURITY_DESCRIPTOR = sd
    return sa


class _OversizedMessageError(RuntimeError):
    """Raised when a single line on the IPC wire exceeds ``MAX_MESSAGE_BYTES``.

    Distinct from ``OSError`` so it doesn't get swallowed by the
    ``except OSError`` arms that treat broken pipes as EOF
    (the original SR2-3 bug — Windows callers couldn't distinguish
    "stream ended" from "stream over budget").

    Carries the byte count of the over-budget buffer for diagnostics.
    """

    def __init__(self, byte_count: int) -> None:
        super().__init__(f"IPC message exceeds {MAX_MESSAGE_BYTES} bytes (saw {byte_count} bytes)")
        self.byte_count = byte_count


class _WinPipeIO:
    """Async-compatible I/O wrapper around a blocking Windows named-pipe HANDLE.

    All blocking calls (ReadFile, WriteFile) are dispatched to the
    *executor* passed at construction time, or to the default thread
    pool via :func:`asyncio.to_thread` when *executor* is ``None``.
    Server-side connections receive the leader's dedicated pipe
    executor (see :data:`MAX_WIN_PIPE_IO_WORKERS`) so a stuck pipe
    cannot starve unrelated ``asyncio.to_thread`` callers (SR2-2).

    A :class:`threading.Lock` serialises concurrent writes so heartbeat
    and response bytes never interleave in the pipe's write buffer.
    """

    def __init__(
        self,
        handle: Any,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self._handle = handle
        self._read_buf = b""
        self._write_lock = threading.Lock()
        self._executor = executor

    async def _run_in_executor(self, fn: Callable[..., Any], *args: Any) -> Any:
        if self._executor is None:
            return await asyncio.to_thread(fn, *args)
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _readline_sync(self) -> bytes:
        """Read bytes from the pipe until a newline is found (blocking).

        Enforces ``MAX_MESSAGE_BYTES`` while accumulating chunks: a
        same-user client could otherwise send an unterminated oversized
        line and grow memory + thread-pool usage unbounded.  When the
        buffered length exceeds the cap we raise
        :class:`_OversizedMessageError` so the caller can emit a
        protocol-level ``error_type="oversized"`` response instead of
        treating the truncated stream as a generic EOF (SR2-3).
        """
        import pywintypes
        import win32file

        while True:
            nl = self._read_buf.find(b"\n")
            if nl >= 0:
                # SR2-3 (review-r6sr2-3): even a newline-terminated frame
                # must be rejected if it exceeds the cap, so a peer that
                # sends one giant line with a trailing \n cannot bypass
                # the size guard.  Pre-fix this branch returned the
                # oversized line which then sailed past the (Unix-only)
                # post-read length check on the client side.
                if nl + 1 > MAX_MESSAGE_BYTES:
                    line_len = nl + 1
                    log.warning(
                        "IPC(win): newline-terminated message exceeded %d bytes "
                        "(line=%d); raising _OversizedMessageError",
                        MAX_MESSAGE_BYTES,
                        line_len,
                    )
                    self._read_buf = self._read_buf[nl + 1 :]
                    raise _OversizedMessageError(line_len)
                line = self._read_buf[: nl + 1]
                self._read_buf = self._read_buf[nl + 1 :]
                return line
            if len(self._read_buf) > MAX_MESSAGE_BYTES:
                # SR2-3: oversized unterminated line.  Raise a typed
                # exception so the caller can tell this apart from
                # plain EOF (the original bug — Windows callers got
                # silent ``b""`` and surfaced it as broken pipe).
                buf_len = len(self._read_buf)
                log.warning(
                    "IPC(win): unterminated message exceeded %d bytes; raising _OversizedMessageError",
                    MAX_MESSAGE_BYTES,
                )
                self._read_buf = b""
                raise _OversizedMessageError(buf_len)
            try:
                _, data = win32file.ReadFile(self._handle, RECV_BUFFER_SIZE)
                self._read_buf += data
            except pywintypes.error as exc:
                # ERROR_BROKEN_PIPE, ERROR_NO_DATA, ERROR_PIPE_NOT_CONNECTED,
                # ERROR_OPERATION_ABORTED
                if exc.args[0] in (109, 232, 233, 995):
                    return b""
                raise OSError(exc.args[0], exc.args[2]) from exc

    async def readline(self) -> bytes:
        """Async readline: delegates to the configured executor."""
        result: bytes = await self._run_in_executor(self._readline_sync)
        return result

    def _write_sync(self, data: bytes) -> None:
        """Write *data* to the pipe (blocking, serialised by write lock)."""
        import pywintypes
        import win32file

        if not data:
            return
        with self._write_lock:
            try:
                win32file.WriteFile(self._handle, data)
            except pywintypes.error as exc:
                raise OSError(exc.args[0], exc.args[2]) from exc

    async def write_all(self, data: bytes) -> None:
        """Async write: delegates to the configured executor."""
        await self._run_in_executor(self._write_sync, data)

    def close(self) -> None:
        """Close the underlying pipe handle (idempotent).

        Calls ``CancelIoEx`` first to abort any blocking ``ReadFile``/``WriteFile``
        in a thread-pool thread, ensuring the thread returns promptly and the
        asyncio executor can shut down without hanging.
        """
        import ctypes

        import pywintypes
        import win32file

        handle, self._handle = self._handle, None
        if handle is not None:
            with contextlib.suppress(Exception):
                ctypes.windll.kernel32.CancelIoEx(int(handle), None)
            with contextlib.suppress(pywintypes.error, OSError):
                win32file.CloseHandle(handle)


class _WinConnectionHandler:
    """Windows named-pipe per-connection handler (DESIGN.md §10.5, §10.7).

    Verifies that the connected client's SID matches the current user's SID
    (cross-user rejection per §10.7), then runs the same JSON-over-stream
    request/response loop as :class:`_ConnectionHandler`, reusing
    :func:`_run_dispatch_logic` for all business logic.
    """

    def __init__(
        self,
        io: _WinPipeIO,
        handler: IPCHandler,
        cache: _IdempotencyCache,
        overlay_locks: dict[str, asyncio.Lock],
        config: IPCConfig,
        our_sid: Any,
    ) -> None:
        self._io = io
        self._handler = handler
        self._cache = cache
        self._overlay_locks = overlay_locks
        self._config = config
        self._our_sid = our_sid

    async def _verify_client_sid(self) -> bool:
        """Impersonate the client and compare its SID to ours (§10.7).

        Returns ``False`` (and logs) if the SIDs differ or impersonation fails.
        Fails CLOSED on any exception — same conservative policy as SO_PEERCRED.
        """
        import pywintypes
        import win32api
        import win32security

        try:

            def _check_sync() -> bool:
                win32security.ImpersonateNamedPipeClient(self._io._handle)
                try:
                    thread_token = win32security.OpenThreadToken(
                        win32api.GetCurrentThread(),
                        win32security.TOKEN_QUERY,
                        True,
                    )
                    client_user = win32security.GetTokenInformation(
                        thread_token, win32security.TokenUser
                    )
                    client_sid = client_user[0]
                finally:
                    win32security.RevertToSelf()
                return str(client_sid) == str(self._our_sid)

            result: bool = await self._io._run_in_executor(_check_sync)
            return result
        except pywintypes.error:
            log.warning("IPC(win): SID check failed; rejecting connection", exc_info=True)
            return False
        except Exception:
            log.warning("IPC(win): SID check failed; rejecting connection", exc_info=True)
            return False

    async def _heartbeat(self, interval: float) -> None:
        """Send ``{"type":"heartbeat"}`` lines every *interval* seconds."""
        hb_line = json.dumps({"type": "heartbeat"}, separators=(",", ":")).encode() + b"\n"
        while True:
            await asyncio.sleep(interval)
            try:
                await self._io.write_all(hb_line)
            except Exception:
                return

    async def _send_error(self, request_id: int, error: str, error_type: str) -> None:
        resp = IPCResponse(request_id=request_id, ok=False, error=error, error_type=error_type)
        with contextlib.suppress(Exception):
            await self._io.write_all(_encode_response(resp))

    async def run(self) -> None:
        """Read / dispatch / write loop for the Windows pipe connection.

        Windows requires that at least one pipe read has completed before
        ``ImpersonateNamedPipeClient`` can succeed (error 1368).  The first
        request line is therefore read unconditionally; only then is the client
        SID verified.  If the SID check fails, the connection is closed without
        processing the request.
        """
        sid_verified = False
        try:
            while True:
                try:
                    line = await self._io.readline()
                except _OversizedMessageError as exc:
                    # SR2-3: emit the documented protocol-level error
                    # (Unix path already does this).  Send-then-close.
                    log.warning(
                        "IPC(win): oversized message (%d bytes), sending oversized error",
                        exc.byte_count,
                    )
                    await self._send_error(-1, "Message exceeds 1 MiB limit", "oversized")
                    break
                except OSError:
                    break

                if not line:
                    break

                # Defense-in-depth: a line at or below the cap that
                # nonetheless exceeds it (e.g. via a future framing
                # change) is still rejected with the same protocol
                # error so behaviour stays consistent with Unix.
                if len(line) > MAX_MESSAGE_BYTES:
                    log.warning("IPC(win): oversized message (%d bytes), closing", len(line))
                    await self._send_error(-1, "Message exceeds 1 MiB limit", "oversized")
                    break

                # Verify client SID on first request (requires at least one read).
                if not sid_verified:
                    if not await self._verify_client_sid():
                        return
                    sid_verified = True

                req = _decode_request(line)
                if req is None:
                    log.warning("IPC(win): malformed request, closing connection")
                    break

                is_long_op = req.op in {"commit_links", "reindex"}
                if is_long_op:
                    hb_task: asyncio.Task[None] = asyncio.create_task(
                        self._heartbeat(self._config.timeouts.long_heartbeat_interval)
                    )
                    try:
                        resp = await _run_dispatch_logic(
                            req, self._handler, self._cache, self._config
                        )
                    finally:
                        hb_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await hb_task
                else:
                    resp = await _run_dispatch_logic(req, self._handler, self._cache, self._config)

                try:
                    await self._io.write_all(_encode_response(resp))
                except OSError:
                    break
        finally:
            self._io.close()


# ─── IPCServer ────────────────────────────────────────────────────────


class IPCServer:
    """Leader-side JSON-over-stream listener (DESIGN.md §10.3 v3.1).

    Owns the Unix socket at ``.scry/scry.sock`` (Linux/macOS) or the Windows
    named pipe at ``\\\\.\\pipe\\scry-<hash>`` with a restrictive DACL. Also owns:

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
        # Windows-only state
        self._win_accept_task: asyncio.Task[None] | None = None
        self._win_stop_event: threading.Event = threading.Event()
        self._win_pipe_name: str = ""
        # SR2-2: dedicated executor for Windows blocking pipe I/O.
        # Lazily created in _start_win so non-Windows / never-started
        # servers don't pay for thread creation.
        self._win_pipe_executor: concurrent.futures.ThreadPoolExecutor | None = None

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
        """
        if _IS_WINDOWS:
            await self._start_windows()
        else:
            await self._start_unix()

    async def _start_windows(self) -> None:
        """Create the Windows named pipe with a restrictive DACL and start accepting.

        The pipe is created with ``FILE_FLAG_FIRST_PIPE_INSTANCE`` on the first
        instance to prevent pipe-squatting by another process.  Subsequent pipe
        instances (for new connections after the first client disconnects) do NOT
        use that flag, but the DACL on all instances grants only the current user.

        The accept loop runs as a background asyncio task.  Stopping it uses a
        threading.Event to signal the blocking ``ConnectNamedPipe`` thread, plus
        a short-circuit dummy ``CreateFile`` to unblock any pending wait.
        """
        import pywintypes
        import win32file
        import win32pipe

        spec = parse_endpoint_uri(self._uri, self._repo_root)
        self._win_pipe_name = spec.address
        self._win_stop_event.clear()

        # SR2-2: dedicated executor for Windows blocking pipe I/O so a
        # stuck pipe cannot starve unrelated ``asyncio.to_thread`` callers
        # (git diff, file hashing, model load, etc.).  Sized to
        # MAX_WIN_PIPE_IO_WORKERS = MAX_WIN_CONNECTIONS + reserved slack.
        if self._win_pipe_executor is None:
            self._win_pipe_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WIN_PIPE_IO_WORKERS,
                thread_name_prefix="scry-ipc-win-pipe",
            )
        win_executor = self._win_pipe_executor

        sa = _win_build_pipe_sa()
        our_sid = _win_get_current_user_sid()

        handler = self._handler
        cache = self._cache
        overlay_locks = self._overlay_locks
        config = self._config
        connection_tasks = self._connection_tasks
        pipe_name = self._win_pipe_name
        stop_event = self._win_stop_event

        first_instance_flag = win32pipe.FILE_FLAG_FIRST_PIPE_INSTANCE

        # Bind synchronously BEFORE returning so a squatting / pre-existing
        # pipe causes a clean failure that the caller surfaces, instead of
        # silently advertising an endpoint we don't actually own
        # (review-w6b BLOCKING fix).
        # SR2-2: cap concurrent pipe instances at MAX_WIN_CONNECTIONS+1
        # so Windows itself applies backpressure once we hit the limit
        # (extra +1 is the always-present "currently being created" slot).
        max_pipe_instances = MAX_WIN_CONNECTIONS + 1
        try:
            first_handle = win32pipe.CreateNamedPipe(
                pipe_name,
                win32pipe.PIPE_ACCESS_DUPLEX | first_instance_flag,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                max_pipe_instances,
                RECV_BUFFER_SIZE,
                RECV_BUFFER_SIZE,
                0,
                sa,
            )
        except pywintypes.error as exc:
            # 231 (ERROR_PIPE_BUSY) / 183 (ERROR_ALREADY_EXISTS) /
            # 5 (ERROR_ACCESS_DENIED) all indicate the pipe name is
            # taken or otherwise unbindable.  Surface to the caller.
            raise OSError(
                f"IPC(win): cannot bind named pipe {pipe_name!r}: {exc} — "
                "another process may already own this endpoint, or the "
                "name is squatted by a hostile process.  Refusing to start."
            ) from exc

        async def _accept_loop(initial_handle: Any) -> None:
            handle = initial_handle  # use the pre-bound first instance
            while not stop_event.is_set():
                if handle is None:
                    # Subsequent instances — first-instance flag is OFF.
                    try:
                        handle = win32pipe.CreateNamedPipe(
                            pipe_name,
                            win32pipe.PIPE_ACCESS_DUPLEX,
                            win32pipe.PIPE_TYPE_BYTE
                            | win32pipe.PIPE_READMODE_BYTE
                            | win32pipe.PIPE_WAIT,
                            max_pipe_instances,
                            RECV_BUFFER_SIZE,
                            RECV_BUFFER_SIZE,
                            0,
                            sa,
                        )
                    except pywintypes.error as exc:
                        log.error("IPC(win): CreateNamedPipe failed: %s", exc)
                        break

                # ConnectNamedPipe blocks until a client connects.  We run it
                # in the dedicated pipe executor so the event loop stays
                # responsive AND pipe accepts cannot starve the default pool.
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        win_executor, win32pipe.ConnectNamedPipe, handle, None
                    )
                except asyncio.CancelledError:
                    win32file.CloseHandle(handle)
                    return
                except pywintypes.error as exc:
                    # ERROR_PIPE_CONNECTED (535) is a benign race: a client
                    # connected between CreateNamedPipe and ConnectNamedPipe
                    # so the pipe is ALREADY connected.  Treat as success
                    # (review-w6b HIGH fix).
                    if exc.args[0] == 535:
                        pass  # fall through to dispatch
                    else:
                        win32file.CloseHandle(handle)
                        # 995 = ERROR_OPERATION_ABORTED (stop in progress)
                        if exc.args[0] == 995:
                            return
                        log.warning("IPC(win): ConnectNamedPipe error: %s", exc)
                        handle = None
                        continue
                except Exception as exc:
                    win32file.CloseHandle(handle)
                    log.warning("IPC(win): ConnectNamedPipe unexpected error: %s", exc)
                    handle = None
                    continue

                if stop_event.is_set():
                    # Dummy client connected during shutdown — discard.
                    win32file.CloseHandle(handle)
                    return

                # SR2-2: prune done tasks BEFORE counting so the
                # add_done_callback lag (one event-loop turn) doesn't
                # cause false rejections.
                for done_task in tuple(connection_tasks):
                    if done_task.done():
                        connection_tasks.discard(done_task)

                if len(connection_tasks) >= MAX_WIN_CONNECTIONS:
                    # SR2-2: at cap — send a connection-level rejection
                    # frame and close.  Clients see a clean
                    # ``too_many_connections`` error instead of hanging.
                    log.warning(
                        "IPC(win): rejecting connection — at MAX_WIN_CONNECTIONS=%d",
                        MAX_WIN_CONNECTIONS,
                    )
                    rejected_io = _WinPipeIO(handle, executor=win_executor)
                    try:
                        reject_frame = (
                            json.dumps(
                                {
                                    "type": "connection_rejected",
                                    "ok": False,
                                    "error": (
                                        f"IPC server is at MAX_WIN_CONNECTIONS"
                                        f"={MAX_WIN_CONNECTIONS}; retry shortly."
                                    ),
                                    "error_type": "too_many_connections",
                                    "retry_after_ms": 100,
                                },
                                separators=(",", ":"),
                            ).encode()
                            + b"\n"
                        )
                        with contextlib.suppress(OSError):
                            await rejected_io.write_all(reject_frame)
                    finally:
                        rejected_io.close()
                    handle = None
                    continue

                io = _WinPipeIO(handle, executor=win_executor)
                conn = _WinConnectionHandler(io, handler, cache, overlay_locks, config, our_sid)

                async def _conn_task(c: _WinConnectionHandler = conn) -> None:
                    await c.run()

                ct = asyncio.create_task(_conn_task())
                connection_tasks.add(ct)
                ct.add_done_callback(connection_tasks.discard)
                handle = None  # next loop iteration creates a fresh instance

        # Yield once so the task starts before this coroutine returns
        # (satisfies §10.2 ordering requirement).
        self._win_accept_task = asyncio.create_task(_accept_loop(first_handle))
        await asyncio.sleep(0)
        log.info("IPC: listening at %s", self._win_pipe_name)

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
        if _IS_WINDOWS:
            # Signal the accept loop to exit and unblock its ConnectNamedPipe wait.
            self._win_stop_event.set()
            if self._win_accept_task is not None:
                import pywintypes
                import win32file

                # Open a dummy client to unblock the pending ConnectNamedPipe.
                if self._win_pipe_name:
                    with contextlib.suppress(pywintypes.error, OSError):
                        h = win32file.CreateFile(
                            self._win_pipe_name,
                            win32file.GENERIC_WRITE,
                            0,
                            None,
                            win32file.OPEN_EXISTING,
                            0,
                            None,
                        )
                        with contextlib.suppress(pywintypes.error):
                            win32file.CloseHandle(h)

                self._win_accept_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._win_accept_task
                self._win_accept_task = None

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

        # SR2-2: shut down the dedicated Windows pipe executor AFTER all
        # connection tasks/handles are closed so we don't strand workers.
        if self._win_pipe_executor is not None:
            with contextlib.suppress(Exception):
                self._win_pipe_executor.shutdown(wait=False, cancel_futures=True)
            self._win_pipe_executor = None

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
        :meth:`call` is fully supported via the ``_WinPipeIO`` path
        (Wave 6b): named pipes with a current-user-restricted DACL,
        framed line-delimited JSON, and overlapped I/O via
        ``asyncio.to_thread`` for the blocking ``ReadFile`` /
        ``WriteFile`` calls.
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
        self._win_io: _WinPipeIO | None = None
        self._request_id: int = 0
        # Serializes the send-then-receive critical section so concurrent
        # call() invocations on the same instance cannot interleave writes
        # OR receive each other's responses (review-w2h HIGH fix).
        self._call_lock = asyncio.Lock()

    async def _connect(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open the Unix socket connection if not already open."""
        if self._reader is not None and self._writer is not None:
            return self._reader, self._writer
        reader, writer = await asyncio.open_unix_connection(  # type: ignore[attr-defined]
            self._spec.address,
            limit=MAX_MESSAGE_BYTES + 1,
        )
        self._reader, self._writer = reader, writer
        return reader, writer

    async def _connect_win(self) -> _WinPipeIO:
        """Open a Windows named-pipe connection if not already open."""
        if self._win_io is not None:
            return self._win_io
        import pywintypes
        import win32file

        pipe_path = self._spec.address
        # Retry briefly on ERROR_FILE_NOT_FOUND (2) or ERROR_PIPE_BUSY (231).
        for attempt in range(10):
            try:
                handle = win32file.CreateFile(
                    pipe_path,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )
                self._win_io = _WinPipeIO(handle)
                return self._win_io
            except pywintypes.error as exc:
                if exc.args[0] in (2, 231) and attempt < 9:
                    await asyncio.sleep(0.05 * (attempt + 1))
                    continue
                raise OSError(exc.args[0], exc.args[2]) from exc
        raise OSError("IPC(win): could not connect to named pipe")

    async def _recv_response(
        self,
        req_id: int,
        readline: Any,
        is_long_op: bool,
        effective_timeout: float | None,
    ) -> Any:
        """Read lines from the server, skipping heartbeats, until a real response arrives.

        For long ops, each individual readline is guarded by the lapse timeout
        (``long_heartbeat_max_lapse``) so a hung server is detected within that
        window even while heartbeats are flowing.
        """
        lapse_timeout = self._config.timeouts.long_heartbeat_max_lapse

        while True:
            try:
                if is_long_op:
                    line = await asyncio.wait_for(readline(), timeout=lapse_timeout)
                elif effective_timeout is not None:
                    line = await asyncio.wait_for(readline(), timeout=effective_timeout)
                else:
                    line = await readline()
            except TimeoutError:
                await self.close()
                raise
            except _OversizedMessageError as exc:
                # SR2-3: server sent (or attempted to send) a response
                # exceeding MAX_MESSAGE_BYTES.  The pipe is poisoned —
                # close the client connection so subsequent .call()s
                # don't reuse a half-read buffer.  Surface as a clean
                # RuntimeError so callers don't see a generic OSError.
                await self.close()
                raise RuntimeError(
                    f"IPC: server response exceeds {MAX_MESSAGE_BYTES} bytes "
                    f"({exc.byte_count} bytes seen) — connection closed"
                ) from exc

            if not line:
                await self.close()
                raise RuntimeError("IPC: server closed connection unexpectedly")

            try:
                resp_d: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                await self.close()
                raise RuntimeError(f"IPC: invalid JSON in leader response: {exc}") from exc

            # Skip heartbeat messages transparently.
            if resp_d.get("type") == "heartbeat":
                continue

            # SR2-2: ``connection_rejected`` is a connection-level
            # frame sent by the leader when at MAX_WIN_CONNECTIONS.
            # It has no ``id`` field; surface it as a clean
            # too_many_connections error before the response-id check
            # so callers don't see a confusing "stream state is corrupt".
            if resp_d.get("type") == "connection_rejected":
                await self.close()
                error_msg = (
                    str(resp_d.get("error"))
                    or "IPC server is busy (too_many_connections); retry shortly"
                )
                raise IPCConnectionRejected(
                    error_msg,
                    error_type=str(resp_d.get("error_type") or "too_many_connections"),
                    retry_after_ms=int(resp_d.get("retry_after_ms") or 100),
                )

            # Wire-protocol invariant: the leader echoes our request id.
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
        ``reindex``) receive no timeout (heartbeat lapse timeout applies instead).

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

            if self._spec.scheme == "pipe":
                win_io = await self._connect_win()
                await win_io.write_all(raw)
                return await self._recv_response(
                    req_id, win_io.readline, is_long_op, effective_timeout
                )
            else:
                reader, writer = await self._connect()
                writer.write(raw)
                await writer.drain()
                return await self._recv_response(
                    req_id, reader.readline, is_long_op, effective_timeout
                )

    async def close(self) -> None:
        """Close the underlying transport connection gracefully."""
        win_io, self._win_io = self._win_io, None
        if win_io is not None:
            win_io.close()

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
    "MAX_WIN_CONNECTIONS",
    "MAX_WIN_PIPE_IO_WORKERS",
    "RECV_BUFFER_SIZE",
    "WIN_PIPE_RESERVED_THREADS",
    "WRITE_OPS",
    "EndpointSpec",
    "IPCClient",
    "IPCConnectionRejected",
    "IPCHandler",
    "IPCRequest",
    "IPCResponse",
    "IPCServer",
    "derive_endpoint_uri",
    "parse_endpoint_uri",
]
