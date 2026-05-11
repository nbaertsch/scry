"""Windows named-pipe IPC tests — W6b.

All tests are marked ``windows_only`` and are automatically skipped on
Linux/macOS by the ``pytest_collection_modifyitems`` hook in conftest.py.

Covers:
  - Server starts; URI has the correct ``pipe:`` scheme
  - Client connects and a call returns the handler result
  - Multiple sequential calls on a single client connection
  - Idempotency: same token → handler runs exactly once
  - DACL: current-user connection is allowed (positive)
  - Cross-user SID rejection: mock ``_verify_client_sid`` → False, verify
    the connection is rejected and the handler is never called
  - Heartbeat: long-op handler; client receives heartbeats and skips them,
    returning the final response
  - Heartbeat lapse timeout: long-op handler stalls; client raises TimeoutError
    within the configured lapse window
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from scry.models import IPCConfig, IPCTimeoutsConfig, new_idempotency_token
from scry.process.ipc import (
    IPCClient,
    IPCRequest,
    IPCResponse,
    IPCServer,
    _WinConnectionHandler,  # type: ignore[attr-defined]
    parse_endpoint_uri,
)

# ─── Shared helpers ───────────────────────────────────────────────────


async def _echo_handler(req: IPCRequest) -> IPCResponse:
    return IPCResponse(request_id=req.request_id, ok=True, result={"op": req.op})


# ─── Server lifecycle ─────────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_pipe_server_starts(tmp_repo: Path, windows_only: None) -> None:
    """IPCServer.start() creates a named pipe; URI has pipe: scheme."""
    srv = IPCServer(tmp_repo, handler=_echo_handler)
    await srv.start()
    try:
        assert srv.endpoint_uri.startswith("pipe:")
    finally:
        await srv.stop()


# ─── Basic call / response ────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_pipe_client_call_returns_result(tmp_repo: Path, windows_only: None) -> None:
    """A client call to the Windows pipe server returns the handler result."""
    srv = IPCServer(tmp_repo, handler=_echo_handler)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
    client = IPCClient(spec)
    try:
        result = await client.call("status", {})
        assert result == {"op": "status"}
    finally:
        await client.close()
        await srv.stop()


@pytest.mark.windows_only
async def test_windows_pipe_multiple_sequential_calls(tmp_repo: Path, windows_only: None) -> None:
    """Multiple sequential calls from one client all succeed."""
    call_count = 0

    async def _counting_handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result={"n": call_count})

    srv = IPCServer(tmp_repo, handler=_counting_handler)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)

    # Each call opens its own connection (Windows client reconnects each call).
    results = []
    for _ in range(3):
        client = IPCClient(spec)
        try:
            r = await client.call("status", {})
            results.append(r)
        finally:
            await client.close()

    await srv.stop()
    assert len(results) == 3
    assert call_count == 3


# ─── Idempotency ──────────────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_pipe_idempotency_same_token(tmp_repo: Path, windows_only: None) -> None:
    """Same idempotency token → handler runs exactly once; second call is cached."""
    call_count = 0

    async def _once_handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result={"count": call_count})

    srv = IPCServer(tmp_repo, handler=_once_handler)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)

    token = new_idempotency_token()

    client1 = IPCClient(spec)
    r1 = await client1.call("propose_link", {"src": "a", "dst": "b"}, idempotency_token=token)
    await client1.close()

    client2 = IPCClient(spec)
    r2 = await client2.call("propose_link", {"src": "a", "dst": "b"}, idempotency_token=token)
    await client2.close()

    await srv.stop()

    assert call_count == 1, "handler should be called exactly once for the same token"
    assert r1 == r2


# ─── DACL / SID checks ────────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_pipe_same_user_connection_succeeds(
    tmp_repo: Path, windows_only: None
) -> None:
    """Current user's own process can connect and receive a response (positive DACL test)."""
    srv = IPCServer(tmp_repo, handler=_echo_handler)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
    client = IPCClient(spec)
    try:
        result = await client.call("search", {"query": "x"})
        assert result == {"op": "search"}
    finally:
        await client.close()
        await srv.stop()


@pytest.mark.windows_only
async def test_windows_pipe_cross_user_sid_rejected(tmp_repo: Path, windows_only: None) -> None:
    """_WinConnectionHandler.run() rejects connections whose SID differs from the server's."""
    call_count = 0

    async def _tracking_handler(req: IPCRequest) -> IPCResponse:
        nonlocal call_count
        call_count += 1
        return IPCResponse(request_id=req.request_id, ok=True, result={})

    srv = IPCServer(tmp_repo, handler=_tracking_handler)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)

    # Patch _verify_client_sid to simulate a cross-user connection.
    with patch.object(
        _WinConnectionHandler,
        "_verify_client_sid",
        new_callable=lambda: lambda self: AsyncMock(return_value=False)(),  # type: ignore[arg-type]
    ):
        # The server should close the connection immediately.
        client = IPCClient(spec, config=IPCConfig(timeouts=IPCTimeoutsConfig(short=2.0)))
        try:
            with pytest.raises((RuntimeError, OSError, asyncio.TimeoutError)):
                await client.call("status", {})
        finally:
            await client.close()

    # Allow the server task to complete the rejection.
    await asyncio.sleep(0.05)
    await srv.stop()

    assert call_count == 0, "handler must not be called for a rejected connection"


# ─── Heartbeat ────────────────────────────────────────────────────────


@pytest.mark.windows_only
async def test_windows_heartbeat_client_skips_and_gets_response(
    tmp_repo: Path, windows_only: None
) -> None:
    """Client transparently skips heartbeat messages and returns the real response."""
    config = IPCConfig(
        timeouts=IPCTimeoutsConfig(
            long_heartbeat_interval=0.02,  # 20ms heartbeats
            long_heartbeat_max_lapse=5.0,
        )
    )

    async def _slow_reindex(req: IPCRequest) -> IPCResponse:
        # 80ms delay → ~4 heartbeats at 20ms interval
        await asyncio.sleep(0.08)
        return IPCResponse(request_id=req.request_id, ok=True, result={"indexed": 42})

    srv = IPCServer(tmp_repo, handler=_slow_reindex, config=config)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
    client = IPCClient(spec, config=config)
    try:
        result = await client.call("reindex", {}, idempotency_token=new_idempotency_token())
        assert result == {"indexed": 42}
    finally:
        await client.close()
        await srv.stop()


@pytest.mark.windows_only
async def test_windows_heartbeat_lapse_timeout(tmp_repo: Path, windows_only: None) -> None:
    """Client raises asyncio.TimeoutError when heartbeat lapse timeout elapses."""
    stall_event: asyncio.Event = asyncio.Event()

    config = IPCConfig(
        timeouts=IPCTimeoutsConfig(
            long_heartbeat_interval=60.0,  # no heartbeats sent
            long_heartbeat_max_lapse=0.10,  # 100ms lapse timeout
        )
    )

    async def _stalling_handler(req: IPCRequest) -> IPCResponse:
        # Never returns — simulates a hung reindex.
        await stall_event.wait()  # pragma: no cover
        return IPCResponse(request_id=req.request_id, ok=True, result={})  # pragma: no cover

    srv = IPCServer(tmp_repo, handler=_stalling_handler, config=config)
    await srv.start()
    spec = parse_endpoint_uri(srv.endpoint_uri, tmp_repo)
    client = IPCClient(spec, config=config)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await client.call("reindex", {}, idempotency_token=new_idempotency_token())
    finally:
        stall_event.set()
        await client.close()
        await srv.stop()

# uat-r5-5 pr-d noise
