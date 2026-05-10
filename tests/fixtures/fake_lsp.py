#!/usr/bin/env python3
"""Minimal fake LSP server for integration testing.

Implements just enough of the Language Server Protocol (JSON-RPC 2.0 over
stdio) to exercise LSPSession lifecycle:

* ``initialize``   → responds with ``callHierarchyProvider: true``
* ``initialized``  → notification, no response needed
* ``shutdown``     → responds with ``result: null``
* ``exit``         → exits with code 0
* Everything else → silently ignored

Spawned by tests as::

    [sys.executable, str(FAKE_LSP_PATH)]

This avoids any dependency on real LSP installations.
"""

from __future__ import annotations

import json
import sys


def _read_one() -> dict[str, object] | None:
    """Read one Content-Length-framed JSON-RPC message from stdin.

    Returns the parsed message dict, or ``None`` on EOF / parse failure.
    """
    headers: dict[str, str] = {}
    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            return None  # EOF
        line = raw.rstrip(b"\r\n")
        if not line:
            break  # blank line = end of headers
        if b":" in line:
            key, _, value = line.partition(b":")
            headers[key.strip().lower().decode("ascii", errors="replace")] = value.strip().decode(
                "ascii", errors="replace"
            )

    content_length_str = headers.get("content-length", "0")
    try:
        content_length = int(content_length_str)
    except ValueError:
        return None

    if content_length == 0:
        return None

    body = sys.stdin.buffer.read(content_length)
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    return data  # type: ignore[return-value]


def _write_one(msg: dict[str, object]) -> None:
    """Write one Content-Length-framed JSON-RPC message to stdout."""
    body = json.dumps(msg).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header + body)
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        msg = _read_one()
        if msg is None:
            break  # EOF — parent closed stdin

        method = msg.get("method")
        msg_id = msg.get("id")

        if method == "initialize":
            _write_one(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "capabilities": {
                            "callHierarchyProvider": True,
                            "textDocumentSync": {"openClose": True},
                        },
                        "serverInfo": {
                            "name": "fake-lsp",
                            "version": "0.0.1",
                        },
                    },
                }
            )
        elif method == "shutdown":
            _write_one({"jsonrpc": "2.0", "id": msg_id, "result": None})
        elif method == "exit":
            sys.exit(0)
        # initialized + all other notifications: no response


if __name__ == "__main__":
    main()
