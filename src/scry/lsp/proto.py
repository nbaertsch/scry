"""LSP wire protocol: JSON-RPC 2.0 framing over stdio.

Implements the *base protocol* layer of the Language Server Protocol:
Content-Length header framing, stateless byte-level codec, and async
stream reader/writer wrappers.

References
----------
DESIGN.md §11 — tech stack: existing LSP servers as subprocesses, JSON-RPC
                over stdio
https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#baseProtocol

Wire format::

    Content-Length: 256\\r\\n
    \\r\\n
    { "jsonrpc": "2.0", "id": 1, "method": "...", "params": {...} }

The ``Content-Type`` header is *optional* per spec; it is ignored on read
and never emitted on write (saving ~45 bytes per message).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "LSPCodec",
    "LSPMessage",
    "LSPProtocolError",
    "LSPStreamReader",
    "LSPStreamWriter",
]


# ─── Exceptions ───────────────────────────────────────────────────────


class LSPProtocolError(Exception):
    """Raised when LSP framing is corrupt or the JSON body is malformed.

    Callers should treat this as a fatal session error and terminate
    the LSP subprocess rather than attempting to resume.
    """


# ─── Message ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LSPMessage:
    """One JSON-RPC 2.0 message — request, response, or notification.

    Shapes
    ------
    Request:
        ``id=<int|str>``, ``method=<str>``, ``params=<dict|None>``
    Response (success):
        ``id=<int|str>``, ``result=<any>``
    Response (error):
        ``id=<int|str>``, ``error=<dict>``
    Notification:
        ``id=None``, ``method=<str>``, ``params=<dict|None>``
    """

    id: int | str | None
    method: str | None
    params: dict[str, Any] | None
    result: Any
    error: dict[str, Any] | None


# ─── Codec ────────────────────────────────────────────────────────────


class LSPCodec:
    """LSP wire format: ``Content-Length: N\\r\\n\\r\\n`` + JSON body.

    Completely stateless — no IO, no buffering.  All methods are static.
    """

    @staticmethod
    def encode(msg: dict[str, Any]) -> bytes:
        """Encode *msg* to LSP wire bytes.

        The body is UTF-8 JSON; the header uses ASCII.  No Content-Type
        header is emitted (spec: optional, servers must accept its absence).
        """
        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        return header + body

    @staticmethod
    def decode_one(buf: bytes) -> tuple[LSPMessage, int] | None:
        """Decode the first complete LSP message from *buf*.

        Returns ``(msg, bytes_consumed)`` when a complete message is
        present, or ``None`` if *buf* does not yet contain a full message.

        Raises
        ------
        LSPProtocolError
            When the header section is present but malformed (non-ASCII
            bytes, missing ``Content-Length``, non-integer length value,
            or invalid JSON body).  A ``None`` return is never mixed with
            a protocol error — if the separator is absent the method
            returns ``None``; once the separator is found, any subsequent
            parse failure raises.
        """
        sep = buf.find(b"\r\n\r\n")
        if sep == -1:
            return None

        # Decode and parse headers
        try:
            header_str = buf[:sep].decode("ascii")
        except UnicodeDecodeError as exc:
            raise LSPProtocolError(f"Non-ASCII bytes in LSP header block: {buf[:sep]!r}") from exc

        content_length: int | None = None
        for line in header_str.split("\r\n"):
            if line.lower().startswith("content-length:"):
                value = line.split(":", 1)[1].strip()
                try:
                    content_length = int(value)
                except ValueError as exc:
                    raise LSPProtocolError(f"Malformed Content-Length value: {value!r}") from exc
                break  # first wins; ignore duplicates

        if content_length is None:
            raise LSPProtocolError("No Content-Length header found in LSP message frame")

        body_start = sep + 4  # skip \r\n\r\n
        if len(buf) < body_start + content_length:
            return None  # body not fully received yet

        body_bytes = buf[body_start : body_start + content_length]
        try:
            data: Any = json.loads(body_bytes)
        except json.JSONDecodeError as exc:
            raise LSPProtocolError(f"Malformed JSON body: {exc}") from exc

        if not isinstance(data, dict):
            raise LSPProtocolError(f"JSON-RPC body must be an object, got {type(data).__name__}")

        raw_params = data.get("params")
        params: dict[str, Any] | None = raw_params if isinstance(raw_params, dict) else None

        raw_error = data.get("error")
        error: dict[str, Any] | None = raw_error if isinstance(raw_error, dict) else None

        msg = LSPMessage(
            id=data.get("id"),
            method=data.get("method"),
            params=params,
            result=data.get("result"),
            error=error,
        )
        return msg, body_start + content_length


# ─── Async stream wrappers ────────────────────────────────────────────


class LSPStreamReader:
    """Async reader that pulls one :class:`LSPMessage` at a time from a stream.

    Maintains an internal byte buffer to handle partial reads (the kernel
    may deliver data in arbitrary chunks).
    """

    def __init__(self, reader: asyncio.StreamReader) -> None:
        self._reader = reader
        self._buf = bytearray()

    async def read_message(self) -> LSPMessage:
        """Read and return the next complete :class:`LSPMessage`.

        Keeps reading from the underlying stream until a full message is
        buffered.

        Raises
        ------
        LSPProtocolError
            If the stream closes before a complete message arrives, or if
            the framing / JSON body is corrupt.
        """
        while True:
            result = LSPCodec.decode_one(bytes(self._buf))
            if result is not None:
                msg, consumed = result
                del self._buf[:consumed]
                return msg

            chunk = await self._reader.read(4096)
            if not chunk:
                raise LSPProtocolError("LSP stream closed before a complete message was received")
            self._buf.extend(chunk)


class LSPStreamWriter:
    """Async writer that frames :class:`LSPMessage` dicts on a stream."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer

    async def write_message(self, msg: dict[str, Any]) -> None:
        """Encode *msg* and write it to the stream, flushing immediately."""
        data = LSPCodec.encode(msg)
        self._writer.write(data)
        await self._writer.drain()
