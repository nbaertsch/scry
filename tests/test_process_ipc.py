"""Tests for `scry.process.ipc` — W2h: IPC transport.

Covers:
  - derive_endpoint_uri / parse_endpoint_uri round-trip (Unix + pipe schemes)
  - IPCServer.start binds; IPCClient.call returns handler result
  - Concurrent calls serialized via per-overlay asyncio.Lock
  - Idempotency: same token → handler runs once; cache hit on repeat
  - Different tokens for same op → handler runs twice
  - Read ops bypass idempotency cache
  - Write op missing token → leader rejects
  - Bad token format → leader rejects with error
  - Per-op timeout: short op exceeds client timeout → asyncio.TimeoutError
  - Connection-close mid-call → leader handles gracefully
  - Message > 1 MiB → leader rejects and closes connection
  - Unix socket mode 0600 after start (unix_only)
  - SO_PEERCRED code path verification (unix_only)
  - _IdempotencyCache LRU eviction (unit test, no server needed)
  - Windows stubs raise NotImplementedError (windows_only)
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scry.models import IPCConfig, IPCTimeoutsConfig, new_idempotency_token
from scry.process.ipc import (
    MAX_MESSAGE_BYTES,
    WRITE_OPS,
    EndpointSpec,
    IPCClient,
    IPCRequest,
    IPCResponse,
    IPCServer,
    _ConnectionHandler,
    _IdempotencyCache,
    _validate_token,
    derive_endpoint_uri,
    parse_endpoint_uri,
)

_IS_WINDOWS = sys.platform == "win32"


# ─── Helpers ──────────────────────────────────────────────────────────


def _echo_handler(req: IPCRequest) -> IPCResponse:
    """Sync helper — tests wrap this in an async lambda."""
    return IPCResponse(request_id=req.request_id, ok=True, result={"op": req.op})


async def _async_echo(req: IPCRequest) -> IPCResponse:
    return IPCResponse(request_id=req.request_id, ok=True, result={"op": req.op})


# ─── Token validation (unit, no server) ───────────────────────────────


def test_validate_token_valid() -> None:
    assert _validate_token("tok_abc123")
    assert _validate_token("tok_ABC-_XYZ")
    assert _validate_token("tok_" + "a" * 64)


def test_validate_token_invalid_no_suffix() -> None:
    # "tok_" with nothing after underscore — + quantifier requires ≥1 char
    assert not _validate_token("tok_")


def test_validate_token_invalid_wrong_prefix() -> None:
    assert not _validate_token("bad_abc")
    assert not _validate_token("token_abc")


def test_validate_token_invalid_spaces() -> None:
    assert not _validate_token("tok abc")
    assert not _validate_token(" tok_abc")


def test_validate_token_empty() -> None:
    assert not _validate_token("")


# ─── Idempotency cache (unit, no server) ──────────────────────────────


def test_idempotency_cache_miss_returns_none() -> None:
    cache = _IdempotencyCache(maxsize=10)
    assert cache.get("tok_x") is None


def test_idempotency_cache_put_and_get() -> None:
    cache = _IdempotencyCache(maxsize=10)
    resp = IPCResponse(request_id=1, ok=True, result="hello")
    cache.put("tok_a", resp)
    assert cache.get("tok_a") is resp


def test_idempotency_cache_lru_eviction() -> None:
    """LRU eviction: oldest entry is dropped when at capacity."""
    cache = _IdempotencyCache(maxsize=2)
    r1 = IPCResponse(request_id=1, ok=True, result="r1")
    r2 = IPCResponse(request_id=2, ok=True, result="r2")
    r3 = IPCResponse(request_id=3, ok=True, result="r3")

    cache.put("tok_a", r1)
    cache.put("tok_b", r2)
    cache.put("tok_c", r3)  # evicts tok_a (LRU)

    assert cache.get("tok_a") is None  # evicted
    assert cache.get("tok_b") is not None
    assert cache.get("tok_c") is not None


def test_idempotency_cache_access_promotes_to_mru() -> None:
    """Accessing tok_a makes it MRU; tok_b becomes LRU and is evicted next."""
    cache = _IdempotencyCache(maxsize=2)
    r1 = IPCResponse(request_id=1, ok=True)
    r2 = IPCResponse(request_id=2, ok=True)
    r3 = IPCResponse(request_id=3, ok=True)

    cache.put("tok_a", r1)
    cache.put("tok_b", r2)
    cache.get("tok_a")  # promote tok_a → tok_b is now LRU
    cache.put("tok_c", r3)  # should evict tok_b

    assert cache.get("tok_a") is not None
    assert cache.get("tok_b") is None  # evicted
    assert cache.get("tok_c") is not None


def test_idempotency_cache_update_existing_key() -> None:
    cache = _IdempotencyCache(maxsize=5)
    r1 = IPCResponse(request_id=1, ok=True, result="old")
    r2 = IPCResponse(request_id=1, ok=True, result="new")
    cache.put("tok_a", r1)
    cache.put("tok_a", r2)  # update same key
    assert cache.get("tok_a") is r2


def test_idempotency_cache_len() -> None:
    cache = _IdempotencyCache(maxsize=5)
    assert len(cache) == 0
    cache.put("tok_a", IPCResponse(request_id=1, ok=True))
    assert len(cache) == 1


# ─── URI helpers ──────────────────────────────────────────────────────


def test_derive_endpoint_uri_unix(tmp_path: Path) -> None:
    if _IS_WINDOWS:
        pytest.skip("Unix-specific")
    uri = derive_endpoint_uri(tmp_path)
    assert uri.startswith("unix:")
    assert "scry.sock" in uri
    assert ".scry" in uri


def test_derive_endpoint_uri_windows(tmp_path: Path) -> None:
    if not _IS_WINDOWS:
        pytest.skip("Windows-specific")
    uri = derive_endpoint_uri(tmp_path)
    assert uri.startswith("pipe:scry-")
    # SHA-256[:16] is 16 hex chars; "scry-" adds 5 → total name length 21
    name = uri[len("pipe:") :]
    assert name.startswith("scry-")
    assert len(name) == len("scry-") + 16


def test_derive_endpoint_uri_windows_deterministic(tmp_path: Path) -> None:
    """Same repo_root produces the same URI (hash is deterministic)."""
    if not _IS_WINDOWS:
        pytest.skip("Windows-specific")
    assert derive_endpoint_uri(tmp_path) == derive_endpoint_uri(tmp_path)


def test_derive_endpoint_uri_unix_different_repos(tmp_path: Path) -> None:
    if _IS_WINDOWS:
        pytest.skip("Unix-specific")
    a = tmp_path / "repo_a"
    b = tmp_path / "repo_b"
    a.mkdir()
    b.mkdir()
    assert derive_endpoint_uri(a) != derive_endpoint_uri(b)


def test_parse_endpoint_uri_unix_relative(tmp_path: Path) -> None:
    spec = parse_endpoint_uri("unix:.scry/scry.sock", tmp_path)
    assert spec.scheme == "unix"
    assert spec.address == str(tmp_path / ".scry" / "scry.sock")


def test_parse_endpoint_uri_unix_absolute(tmp_path: Path) -> None:
    sock = str(tmp_path / "scry.sock")
    spec = parse_endpoint_uri(f"unix:{sock}", tmp_path)
    assert spec.scheme == "unix"
    assert spec.address == sock


def test_parse_endpoint_uri_pipe(tmp_path: Path) -> None:
    spec = parse_endpoint_uri("pipe:scry-abc123def456", tmp_path)
    assert spec.scheme == "pipe"
    assert spec.address == r"\\.\pipe\scry-abc123def456"


def test_parse_endpoint_uri_empty_pipe_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Empty pipe name"):
        parse_endpoint_uri("pipe:", tmp_path)


def test_parse_endpoint_uri_unknown_scheme_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown IPC URI scheme"):
        parse_endpoint_uri("tcp:localhost:1234", tmp_path)


def test_derive_parse_roundtrip_unix(tmp_path: Path) -> None:
    if _IS_WINDOWS:
        pytest.skip("Unix-specific")
    uri = derive_endpoint_uri(tmp_path)
    spec = parse_endpoint_uri(uri, tmp_path)
    assert spec.scheme == "unix"
    assert spec.address.endswith("scry.sock")
    # Re-derive must match
    assert uri == derive_endpoint_uri(tmp_path)


def test_endpoint_spec_is_frozen(tmp_path: Path) -> None:
    spec = parse_endpoint_uri("unix:.scry/scry.sock", tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        spec.address = "other"  # type: ignore[misc]


# ─── Server + client integration (unix_only) ──────────────────────────


@pytest.mark.unix_only
async def test_ipc_server_start_binds(tmp_repo: Path, unix_only: None) -> None:
    """IPCServer.start() creates the socket file."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        assert Path(spec.address).exists()
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_ipc_server_stop_removes_socket(tmp_repo: Path, unix_only: None) -> None:
    """IPCServer.stop() removes the socket file."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
    sock = Path(spec.address)
    assert sock.exists()
    await srv.stop()
    assert not sock.exists()


@pytest.mark.unix_only
async def test_ipc_client_call_returns_result(tmp_repo: Path, unix_only: None) -> None:
    """IPCClient.call returns the handler's result payload."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        result = await client.call("status", {})
        assert result == {"op": "status"}
        await client.close()
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_ipc_client_multiple_calls_same_connection(tmp_repo: Path, unix_only: None) -> None:
    """The persistent connection handles multiple sequential calls correctly."""
    call_count = 0

    async def _handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result=call_count)

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        r1 = await client.call("status", {})
        r2 = await client.call("status", {})
        await client.close()
        assert r1 == 1
        assert r2 == 2
        assert call_count == 2
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_ipc_client_concurrent_calls_routed_correctly(
    tmp_repo: Path, unix_only: None
) -> None:
    """Regression (review-w2h HIGH): concurrent IPCClient.call MUST NOT swap responses.

    Without the per-instance asyncio.Lock + response-id check, two
    concurrent calls would race on the shared stream. Caller A writes,
    caller B writes, then whichever wakes first reads — silently
    receiving the OTHER caller's response.

    Each call passes a unique sentinel in args; the handler echoes it
    back. Mixed-up responses produce mismatched (echo, expected) pairs.
    """
    handler_call_count = 0

    async def _handler(req: IPCRequest) -> IPCResponse:
        nonlocal handler_call_count
        handler_call_count += 1
        # Tiny sleep encourages interleaving in the absence of the lock.
        await asyncio.sleep(0.01)
        return IPCResponse(request_id=req.request_id, ok=True, result=req.args.get("sentinel"))

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        sentinels = [f"unique-{i}" for i in range(20)]
        results = await asyncio.gather(*(client.call("status", {"sentinel": s}) for s in sentinels))
        await client.close()
        # Each call must receive its own sentinel back, not someone else's.
        assert results == sentinels, (
            f"response-routing bug: results={results} sentinels={sentinels}"
        )
        assert handler_call_count == 20
    finally:
        await srv.stop()


# ─── Per-overlay lock (concurrent serialization) ──────────────────────


@pytest.mark.unix_only
async def test_concurrent_calls_serialized_by_overlay_lock(tmp_repo: Path, unix_only: None) -> None:
    """Concurrent IPC requests that acquire the same overlay lock are serialized.

    The handler acquires the overlay lock before its work window, so concurrent
    calls on separate connections must queue behind it.  The execution windows
    must not overlap.
    """
    overlay_key = "test_overlay_path"
    execution_windows: list[tuple[float, float]] = []
    srv_holder: list[IPCServer] = []

    async def _handler(req: IPCRequest) -> IPCResponse:
        srv = srv_holder[0]
        lock = srv.get_overlay_lock(overlay_key)
        async with lock:
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.06)  # simulate write work
            end = asyncio.get_event_loop().time()
            execution_windows.append((start, end))
        return IPCResponse(request_id=req.request_id, ok=True)

    srv = IPCServer(tmp_repo, handler=_handler)
    srv_holder.append(srv)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)

        async def _one_call() -> Any:
            c = IPCClient(spec)
            res = await c.call("status", {})
            await c.close()
            return res

        await asyncio.gather(*[_one_call() for _ in range(3)])

        assert len(execution_windows) == 3
        # Sort by start time and check non-overlapping.
        windows = sorted(execution_windows, key=lambda w: w[0])
        for i in range(len(windows) - 1):
            _, end_i = windows[i]
            start_next, _ = windows[i + 1]
            # Allow a tiny float tolerance (1 ms) for scheduling jitter.
            assert end_i <= start_next + 0.001, (
                f"Windows overlapped: {windows[i]} and {windows[i + 1]}"
            )
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_overlay_lock_distinct_keys_independent(tmp_repo: Path, unix_only: None) -> None:
    """Different overlay paths use independent locks (no spurious blocking)."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        lock_a = srv.get_overlay_lock("path/a.jsonl")
        lock_b = srv.get_overlay_lock("path/b.jsonl")
        assert lock_a is not lock_b
        # Same path returns the same lock object.
        assert srv.get_overlay_lock("path/a.jsonl") is lock_a
    finally:
        await srv.stop()


# ─── Idempotency (unix_only) ──────────────────────────────────────────


@pytest.mark.unix_only
async def test_idempotency_same_token_handler_runs_once(tmp_repo: Path, unix_only: None) -> None:
    """Same idempotency token → handler runs exactly once; second call returns cache."""
    call_count = 0

    async def _handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result=call_count)

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        tok = new_idempotency_token()
        client = IPCClient(spec)
        r1 = await client.call("propose_link", {"from": "a", "to": "b"}, idempotency_token=tok)
        r2 = await client.call("propose_link", {"from": "a", "to": "b"}, idempotency_token=tok)
        await client.close()

        assert call_count == 1, "handler must run only once for the same token"
        assert r1 == r2 == 1
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_idempotency_different_tokens_handler_runs_twice(
    tmp_repo: Path, unix_only: None
) -> None:
    """Different tokens for the same op → handler runs for each."""
    call_count = 0

    async def _handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result=call_count)

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        await client.call("propose_link", {}, idempotency_token=new_idempotency_token())
        await client.call("propose_link", {}, idempotency_token=new_idempotency_token())
        await client.close()
        assert call_count == 2
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_read_ops_bypass_idempotency_cache(tmp_repo: Path, unix_only: None) -> None:
    """Read ops (e.g. 'search') have no token and are not cached."""
    call_count = 0

    async def _handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result=call_count)

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        await client.call("search", {"q": "foo"})
        await client.call("search", {"q": "foo"})
        await client.close()
        assert call_count == 2  # both must have run
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_write_op_missing_token_rejected(tmp_repo: Path, unix_only: None) -> None:
    """Write op without an idempotency_token → leader returns validation error."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        with pytest.raises(RuntimeError, match="idempotency_token"):
            await client.call("propose_link", {"from": "a", "to": "b"})
        await client.close()
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_write_ops_frozenset_contents() -> None:
    """WRITE_OPS contains exactly the expected operations."""
    assert {"propose_link", "accept_link", "commit_links", "reindex"} == WRITE_OPS


# ─── Validation (unix_only) ───────────────────────────────────────────


@pytest.mark.unix_only
async def test_bad_idempotency_token_rejected(tmp_repo: Path, unix_only: None) -> None:
    """Malformed idempotency_token → leader rejects with validation error."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        # Bypass the client's normal call() and inject a bad token manually.
        reader, writer = await asyncio.open_unix_connection(spec.address)
        bad_msg = (
            json.dumps(
                {
                    "id": 1,
                    "op": "propose_link",
                    "args": {},
                    "idempotency_token": "BAD TOKEN!",
                    "protocol_version": 1,
                }
            ).encode()
            + b"\n"
        )
        writer.write(bad_msg)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=2.0)
        resp = json.loads(line)
        assert resp["ok"] is False
        assert "Invalid idempotency_token" in resp["error"]
        writer.close()
        await writer.wait_closed()
    finally:
        await srv.stop()


# ─── Per-op timeouts (unix_only) ──────────────────────────────────────


@pytest.mark.unix_only
async def test_short_op_client_timeout(tmp_repo: Path, unix_only: None) -> None:
    """Short op: client times out waiting for leader response → asyncio.TimeoutError."""

    async def _slow_handler(req: IPCRequest) -> IPCResponse:
        await asyncio.sleep(100)  # effectively infinite
        return IPCResponse(request_id=req.request_id, ok=True)

    # Give server a generous timeout so only the client fires.
    srv_cfg = IPCConfig(timeouts=IPCTimeoutsConfig(short=30.0))
    client_cfg = IPCConfig(timeouts=IPCTimeoutsConfig(short=0.05))

    srv = IPCServer(tmp_repo, handler=_slow_handler, config=srv_cfg)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec, config=client_cfg)
        with pytest.raises(asyncio.TimeoutError):
            await client.call("status", {})
        # Connection should be closed after timeout.
        assert client._writer is None
    finally:
        await srv.stop()


@pytest.mark.unix_only
async def test_long_op_not_subject_to_short_timeout(tmp_repo: Path, unix_only: None) -> None:
    """Long ops (commit_links, reindex) are not limited by timeouts.short."""
    completed = asyncio.Event()

    async def _handler(req: IPCRequest) -> IPCResponse:
        await asyncio.sleep(0.1)  # longer than short timeout
        completed.set()
        return IPCResponse(request_id=req.request_id, ok=True, result="done")

    # Very short short-timeout — must NOT affect commit_links.
    cfg = IPCConfig(timeouts=IPCTimeoutsConfig(short=0.01))
    srv = IPCServer(tmp_repo, handler=_handler, config=cfg)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec, config=cfg)
        tok = new_idempotency_token()
        # commit_links is a long op → no timeout applied
        result = await client.call("commit_links", {}, idempotency_token=tok)
        assert result == "done"
        assert completed.is_set()
        await client.close()
    finally:
        await srv.stop()


# ─── Connection-close mid-call ────────────────────────────────────────


@pytest.mark.unix_only
async def test_connection_close_mid_call_leader_handles(tmp_repo: Path, unix_only: None) -> None:
    """Abrupt client close during an in-flight call → server handles gracefully."""
    handler_started = asyncio.Event()

    async def _handler(req: IPCRequest) -> IPCResponse:
        handler_started.set()
        await asyncio.sleep(2)  # long enough for client to close first
        return IPCResponse(request_id=req.request_id, ok=True)

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        call_task = asyncio.create_task(client.call("status", {}))

        # Wait for the handler to start before closing.
        await asyncio.wait_for(handler_started.wait(), timeout=2.0)
        await client.close()

        # The call fails because the connection was closed.
        with pytest.raises((RuntimeError, asyncio.TimeoutError, OSError, EOFError)):
            await asyncio.wait_for(call_task, timeout=1.0)

        # Server itself must still be alive.
        await asyncio.sleep(0.05)
        assert srv._server is not None
    finally:
        await srv.stop()


# ─── Oversized message ────────────────────────────────────────────────


@pytest.mark.unix_only
async def test_oversized_message_rejected(tmp_repo: Path, unix_only: None) -> None:
    """Message > 1 MiB → leader sends error response and closes connection."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        # Connect directly and send a > 1 MiB line.
        reader, writer = await asyncio.open_unix_connection(
            spec.address, limit=MAX_MESSAGE_BYTES * 3
        )
        huge_args = "a" * (2 * 1024 * 1024)
        huge_msg = (
            json.dumps(
                {"id": 1, "op": "status", "args": {"x": huge_args}, "protocol_version": 1}
            ).encode()
            + b"\n"
        )
        writer.write(huge_msg)
        await writer.drain()

        # Server MUST send a structured oversized error response per spec
        # §10.3 v3.1 ("Connection rejected if message > 1 MB (DoS guard)").
        # Allowing silent EOF would let a regression that drops the error
        # response slip through (review-w2h LOW finding).
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=3.0)
        except TimeoutError:
            pytest.fail("server did not respond to oversized message within 3s")

        assert line, "server closed connection without sending the oversized error"
        resp = json.loads(line)
        assert resp["ok"] is False
        assert resp.get("error_type") == "oversized", (
            f"expected error_type=oversized, got {resp.get('error_type')!r}"
        )

        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    finally:
        await srv.stop()


# ─── Unix socket mode 0600 ────────────────────────────────────────────


@pytest.mark.unix_only
async def test_unix_socket_mode_0600(tmp_repo: Path, unix_only: None) -> None:
    """Unix socket file must have mode 0600 immediately after start()."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        mode = Path(spec.address).stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600 but got {oct(mode)}"
    finally:
        await srv.stop()


# ─── SO_PEERCRED code path ────────────────────────────────────────────


@pytest.mark.unix_only
def test_so_peercred_code_path_present(unix_only: None) -> None:
    """Verify the SO_PEERCRED rejection code path exists in _ConnectionHandler."""
    src = inspect.getsource(_ConnectionHandler._verify_peer_uid)
    # The implementation must reference SO_PEERCRED and getuid.
    assert "SO_PEERCRED" in src
    assert "getuid" in src
    assert "uid" in src


@pytest.mark.unix_only
async def test_so_peercred_same_user_allowed(tmp_repo: Path, unix_only: None) -> None:
    """A connection from the same UID as the leader is accepted (Linux)."""
    if sys.platform != "linux":
        pytest.skip("SO_PEERCRED only checked on Linux")

    reached_handler = asyncio.Event()

    async def _handler(req: IPCRequest) -> IPCResponse:
        reached_handler.set()
        return IPCResponse(request_id=req.request_id, ok=True, result="ok")

    srv = IPCServer(tmp_repo, handler=_handler)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        client = IPCClient(spec)
        result = await client.call("status", {})
        assert result == "ok"
        assert reached_handler.is_set()
        await client.close()
    finally:
        await srv.stop()


# ─── Windows stubs ────────────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_ipc_server_raises_not_implemented(
    tmp_repo: Path, windows_only: None
) -> None:
    """On Windows, IPCServer.start() raises NotImplementedError (Wave 6 gap)."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    with pytest.raises(NotImplementedError, match="pywin32"):
        await srv.start()


@pytest.mark.windows_only
async def test_windows_ipc_client_raises_not_implemented(
    tmp_repo: Path, windows_only: None
) -> None:
    """On Windows, IPCClient.call() raises NotImplementedError (Wave 6 gap)."""
    spec = parse_endpoint_uri("pipe:scry-abc123", tmp_repo)
    client = IPCClient(spec)
    with pytest.raises(NotImplementedError, match="pywin32"):
        await client.call("status", {})


# ─── WRITE_OPS constant ───────────────────────────────────────────────


def test_write_ops_is_frozenset() -> None:
    assert isinstance(WRITE_OPS, frozenset)


def test_write_ops_contains_expected() -> None:
    assert "propose_link" in WRITE_OPS
    assert "accept_link" in WRITE_OPS
    assert "commit_links" in WRITE_OPS
    assert "reindex" in WRITE_OPS


def test_write_ops_excludes_reads() -> None:
    assert "search" not in WRITE_OPS
    assert "status" not in WRITE_OPS
    assert "get_anchor" not in WRITE_OPS


# ─── EndpointSpec ─────────────────────────────────────────────────────


def test_endpoint_spec_unix(tmp_path: Path) -> None:
    spec = EndpointSpec(scheme="unix", address="/tmp/scry.sock")
    assert spec.scheme == "unix"
    assert spec.address == "/tmp/scry.sock"


def test_endpoint_spec_pipe(tmp_path: Path) -> None:
    spec = EndpointSpec(scheme="pipe", address=r"\\.\pipe\scry-abc")
    assert spec.scheme == "pipe"


# ─── Malformed JSON ───────────────────────────────────────────────────


@pytest.mark.unix_only
async def test_malformed_json_closes_connection(tmp_repo: Path, unix_only: None) -> None:
    """Sending malformed JSON to the leader closes the connection gracefully."""
    srv = IPCServer(tmp_repo, handler=_async_echo)
    await srv.start()
    try:
        spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
        reader, writer = await asyncio.open_unix_connection(spec.address)
        writer.write(b"not json at all\n")
        await writer.drain()

        # Server closes the connection; we get EOF.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(reader.read(1024), timeout=2.0)

        # Either empty (EOF) or nothing meaningful.
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        # No assertion on data — the key check is that the server didn't crash.
        assert srv._server is not None
    finally:
        await srv.stop()
