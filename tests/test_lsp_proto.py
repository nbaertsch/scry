"""Tests for scry.lsp.proto — JSON-RPC 2.0 wire protocol.

Covers:
* LSPCodec.encode / decode_one round-trips (request, response, notification)
* decode_one returns None for incomplete messages
* decode_one raises LSPProtocolError for corrupt framing
* LSPStreamReader handles partial reads (split mid-header, mid-body)
* LSPStreamWriter emits correctly framed bytes (verified via round-trip)
* LSPMessage is immutable (frozen dataclass)

All tests are pure unit tests — no real LSP process is required.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import pytest

from scry.lsp.proto import (
    LSPCodec,
    LSPMessage,
    LSPProtocolError,
    LSPStreamReader,
    LSPStreamWriter,
)

# ─── Helpers ──────────────────────────────────────────────────────────


def _request(req_id: int, method: str, params: dict[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


def _response(req_id: int, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _notification(method: str, params: dict[str, object]) -> dict[str, object]:
    return {"jsonrpc": "2.0", "method": method, "params": params}


# ─── LSPCodec.encode ──────────────────────────────────────────────────


def test_encode_produces_content_length_header() -> None:
    """encode() prefixes Content-Length: N\\r\\n\\r\\n."""
    msg = _request(1, "initialize", {"rootUri": "file:///repo"})
    encoded = LSPCodec.encode(msg)

    sep = encoded.find(b"\r\n\r\n")
    assert sep != -1, "No \\r\\n\\r\\n separator found"

    header_line = encoded[:sep].decode("ascii")
    assert header_line.startswith("Content-Length: ")

    content_length = int(header_line.split(": ")[1])
    body_bytes = encoded[sep + 4 :]
    assert len(body_bytes) == content_length


def test_encode_body_is_valid_json() -> None:
    msg = _request(99, "shutdown", {})
    encoded = LSPCodec.encode(msg)
    sep = encoded.find(b"\r\n\r\n")
    body = json.loads(encoded[sep + 4 :])
    assert body["id"] == 99
    assert body["method"] == "shutdown"


def test_encode_no_content_type_header() -> None:
    """encode() must NOT emit Content-Type (saves bytes, spec says optional)."""
    msg = _notification("initialized", {})
    encoded = LSPCodec.encode(msg)
    sep = encoded.find(b"\r\n\r\n")
    header_block = encoded[:sep].decode("ascii")
    assert "content-type" not in header_block.lower()


# ─── LSPCodec.decode_one — happy paths ────────────────────────────────


def test_decode_request_roundtrip() -> None:
    msg = _request(1, "textDocument/definition", {"uri": "file:///test.py", "pos": 0})
    encoded = LSPCodec.encode(msg)
    result = LSPCodec.decode_one(encoded)
    assert result is not None
    decoded, consumed = result
    assert consumed == len(encoded)
    assert decoded.id == 1
    assert decoded.method == "textDocument/definition"
    assert decoded.params == {"uri": "file:///test.py", "pos": 0}
    assert decoded.result is None
    assert decoded.error is None


def test_decode_response_roundtrip() -> None:
    msg = _response(2, {"capabilities": {"callHierarchyProvider": True}})
    encoded = LSPCodec.encode(msg)
    result = LSPCodec.decode_one(encoded)
    assert result is not None
    decoded, consumed = result
    assert consumed == len(encoded)
    assert decoded.id == 2
    assert decoded.method is None
    assert decoded.result == {"capabilities": {"callHierarchyProvider": True}}
    assert decoded.error is None


def test_decode_notification_roundtrip() -> None:
    msg = _notification("initialized", {})
    encoded = LSPCodec.encode(msg)
    result = LSPCodec.decode_one(encoded)
    assert result is not None
    decoded, consumed = result
    assert consumed == len(encoded)
    assert decoded.id is None
    assert decoded.method == "initialized"
    assert decoded.params == {}


def test_decode_string_id() -> None:
    """LSP spec allows string IDs as well as integer IDs."""
    msg: dict[str, object] = {"jsonrpc": "2.0", "id": "req-abc", "method": "ping", "params": {}}
    encoded = LSPCodec.encode(msg)
    result = LSPCodec.decode_one(encoded)
    assert result is not None
    decoded, _ = result
    assert decoded.id == "req-abc"


def test_decode_ignores_content_type_header() -> None:
    """Optional Content-Type header is accepted and ignored."""
    body_dict = {"jsonrpc": "2.0", "method": "ping", "params": {}}
    body_bytes = json.dumps(body_dict).encode("utf-8")
    header = (
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n"
        f"\r\n"
    ).encode("ascii")
    result = LSPCodec.decode_one(header + body_bytes)
    assert result is not None
    msg, _ = result
    assert msg.method == "ping"


def test_decode_one_consumes_exactly_one_message() -> None:
    """Concatenated messages: only the first is consumed; bytes_consumed is exact."""
    m1 = LSPCodec.encode(_request(1, "first", {}))
    m2 = LSPCodec.encode(_request(2, "second", {}))
    buf = m1 + m2

    result = LSPCodec.decode_one(buf)
    assert result is not None
    msg, consumed = result
    assert consumed == len(m1)
    assert msg.id == 1

    result2 = LSPCodec.decode_one(buf[consumed:])
    assert result2 is not None
    msg2, _ = result2
    assert msg2.id == 2


# ─── LSPCodec.decode_one — incomplete inputs ──────────────────────────


def test_decode_empty_buffer_returns_none() -> None:
    assert LSPCodec.decode_one(b"") is None


def test_decode_partial_header_returns_none() -> None:
    """Separator not yet arrived → None (wait for more data)."""
    full = LSPCodec.encode(_request(1, "test", {}))
    # Only the first 5 bytes (inside "Content-Length: ...")
    assert LSPCodec.decode_one(full[:5]) is None


def test_decode_header_only_no_body_returns_none() -> None:
    """Header + separator present but body not started → None."""
    full = LSPCodec.encode(_request(1, "test", {}))
    sep = full.find(b"\r\n\r\n") + 4
    assert LSPCodec.decode_one(full[:sep]) is None


def test_decode_body_one_byte_short_returns_none() -> None:
    """One byte missing from the body → None."""
    full = LSPCodec.encode(_request(1, "test", {}))
    assert LSPCodec.decode_one(full[: len(full) - 1]) is None


# ─── LSPCodec.decode_one — corrupt framing ────────────────────────────


def test_decode_raises_no_content_length() -> None:
    """Header separator present but no Content-Length → LSPProtocolError."""
    bad = b"Content-Type: application/vscode-jsonrpc\r\n\r\n{}"
    with pytest.raises(LSPProtocolError, match="No Content-Length"):
        LSPCodec.decode_one(bad)


def test_decode_raises_malformed_content_length() -> None:
    """Non-integer Content-Length value → LSPProtocolError."""
    bad = b"Content-Length: abc\r\n\r\n{}"
    with pytest.raises(LSPProtocolError, match="Malformed Content-Length"):
        LSPCodec.decode_one(bad)


def test_decode_raises_malformed_json_body() -> None:
    """Body is not valid JSON → LSPProtocolError."""
    body = b"not-json!!!"
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    with pytest.raises(LSPProtocolError, match="Malformed JSON"):
        LSPCodec.decode_one(header + body)


def test_decode_raises_non_ascii_header() -> None:
    """Non-ASCII bytes in the header block → LSPProtocolError."""
    bad = b"Content-Length: \xff\r\n\r\n{}"
    with pytest.raises(LSPProtocolError):
        LSPCodec.decode_one(bad)


def test_decode_raises_json_body_not_object() -> None:
    """JSON body that is an array → LSPProtocolError."""
    body = b"[1, 2, 3]"
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    with pytest.raises(LSPProtocolError, match="object"):
        LSPCodec.decode_one(header + body)


# ─── LSPStreamReader ──────────────────────────────────────────────────


async def test_stream_reader_full_message() -> None:
    """Reader returns a complete message when the buffer holds one."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    msg_dict = _request(1, "initialize", {"rootUri": "file:///repo"})
    reader.feed_data(LSPCodec.encode(msg_dict))

    msg = await lsp.read_message()
    assert msg.id == 1
    assert msg.method == "initialize"


async def test_stream_reader_partial_header() -> None:
    """Reader assembles a message whose header arrives in two chunks."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    data = LSPCodec.encode(_request(42, "test/partial", {}))
    split = 7  # inside "Content-Length: ..."

    read_task = asyncio.create_task(lsp.read_message())
    await asyncio.sleep(0)
    reader.feed_data(data[:split])
    await asyncio.sleep(0)
    reader.feed_data(data[split:])

    msg = await read_task
    assert msg.id == 42
    assert msg.method == "test/partial"


async def test_stream_reader_partial_body() -> None:
    """Reader assembles a message whose body arrives in two chunks."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    data = LSPCodec.encode(_response(7, {"capabilities": {"callHierarchyProvider": True}}))
    sep = data.find(b"\r\n\r\n") + 4  # body_start
    mid = sep + 3  # 3 bytes into the body

    read_task = asyncio.create_task(lsp.read_message())
    await asyncio.sleep(0)
    reader.feed_data(data[:mid])
    await asyncio.sleep(0)
    reader.feed_data(data[mid:])

    msg = await read_task
    assert msg.id == 7
    assert isinstance(msg.result, dict)


async def test_stream_reader_sequential_messages() -> None:
    """Reader correctly separates back-to-back messages in the stream."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    reader.feed_data(
        LSPCodec.encode(_request(1, "first", {})) + LSPCodec.encode(_request(2, "second", {}))
    )

    msg1 = await lsp.read_message()
    msg2 = await lsp.read_message()
    assert msg1.id == 1
    assert msg2.id == 2


async def test_stream_reader_raises_on_eof_mid_message() -> None:
    """Reader raises LSPProtocolError when stream closes without a full message."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    reader.feed_data(b"Content-Length: 1000\r\n\r\n")  # body will never arrive
    reader.feed_eof()

    with pytest.raises(LSPProtocolError, match="closed"):
        await lsp.read_message()


async def test_stream_reader_many_small_chunks() -> None:
    """Reader reassembles a message delivered one byte at a time."""
    reader = asyncio.StreamReader()
    lsp = LSPStreamReader(reader)

    data = LSPCodec.encode(_notification("$/progress", {"value": 42}))

    read_task = asyncio.create_task(lsp.read_message())
    await asyncio.sleep(0)
    for byte in data:
        reader.feed_data(bytes([byte]))
        await asyncio.sleep(0)

    msg = await read_task
    assert msg.method == "$/progress"


# ─── LSPStreamWriter ──────────────────────────────────────────────────


async def test_stream_writer_framing() -> None:
    """LSPStreamWriter produces valid Content-Length framing (round-trip check)."""
    server_sock, client_sock = socket.socketpair()
    try:
        _, client_writer = await asyncio.open_connection(sock=client_sock)
        server_reader, server_writer = await asyncio.open_connection(sock=server_sock)

        lsp_writer = LSPStreamWriter(client_writer)
        lsp_reader = LSPStreamReader(server_reader)

        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        await lsp_writer.write_message(msg)

        received = await lsp_reader.read_message()
        assert received.method == "initialize"
        assert received.id == 1

        client_writer.close()
        server_writer.close()
        with contextlib.suppress(Exception):
            await client_writer.wait_closed()
        with contextlib.suppress(Exception):
            await server_writer.wait_closed()
    finally:
        server_sock.close()
        client_sock.close()


async def test_stream_writer_multiple_messages() -> None:
    """Writer can send several messages; reader receives them in order."""
    server_sock, client_sock = socket.socketpair()
    try:
        _, client_writer = await asyncio.open_connection(sock=client_sock)
        server_reader, server_writer = await asyncio.open_connection(sock=server_sock)

        lsp_writer = LSPStreamWriter(client_writer)
        lsp_reader = LSPStreamReader(server_reader)

        for i in range(3):
            await lsp_writer.write_message(
                {"jsonrpc": "2.0", "id": i, "method": "ping", "params": {}}
            )

        for i in range(3):
            msg = await lsp_reader.read_message()
            assert msg.id == i

        client_writer.close()
        server_writer.close()
        with contextlib.suppress(Exception):
            await client_writer.wait_closed()
        with contextlib.suppress(Exception):
            await server_writer.wait_closed()
    finally:
        server_sock.close()
        client_sock.close()


# ─── LSPMessage ───────────────────────────────────────────────────────


def test_lsp_message_is_frozen() -> None:
    """LSPMessage is a frozen dataclass — attribute assignment must fail."""
    msg = LSPMessage(id=1, method="test", params={}, result=None, error=None)
    with pytest.raises((TypeError, AttributeError)):
        msg.id = 2  # type: ignore[misc]


def test_lsp_message_fields() -> None:
    msg = LSPMessage(
        id=None,
        method="initialized",
        params={},
        result=None,
        error=None,
    )
    assert msg.id is None
    assert msg.method == "initialized"
    assert msg.error is None


# uat-r5-5 pr-d noise
