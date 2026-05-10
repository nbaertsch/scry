#!/usr/bin/env python3
"""Enhanced fake LSP server for testing LSP call-hierarchy enrichment.

Extends the minimal fake_lsp.py by also handling:

* ``textDocument/prepareCallHierarchy`` → returns one dummy
  ``CallHierarchyItem`` so the BFS walk has something to start from.
* ``callHierarchy/outgoingCalls`` → returns an empty list, making the
  starting item a leaf function.  This produces ``status == "complete"``
  and a non-empty ``closure_hash`` from ``compute_closure``.
* ``textDocument/didOpen`` / ``textDocument/didClose`` → silently accepted
  (notifications, no response needed).

Expected closure_hash value
---------------------------
Because ``outgoingCalls`` returns an empty list, the walker visits exactly
the starting item.  Its content hash is computed from the *content* read at
the LSP-reported range - but for a fake LSP the range ``(0, 0)-(0, 0)``
maps to an empty string, giving:

    closure_hash = sha256("") = "e3b0c44298fc1c149afbf4c8996fb924..."

The actual hash depends on the file content under the reported range, so
tests should check ``closure_hash is not None`` rather than a specific hex.

Spawned by tests as::

    [sys.executable, str(FAKE_LSP_CALLS_PATH)]
"""

from __future__ import annotations

import json
import sys


def _read_one() -> dict[str, object] | None:
    """Read one Content-Length-framed JSON-RPC message from stdin."""
    headers: dict[str, str] = {}
    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            return None  # EOF
        line = raw.rstrip(b"\r\n")
        if not line:
            break
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
            break

        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

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
                            "name": "fake-lsp-calls",
                            "version": "0.0.1",
                        },
                    },
                }
            )

        elif method == "textDocument/prepareCallHierarchy":
            # Return a single dummy CallHierarchyItem pointing to (0,0)-(0,0).
            # The URI comes from the request so the walker can read the file.
            text_document = params.get("textDocument", {})
            uri = text_document.get("uri", "file:///unknown")
            position = params.get("position", {"line": 0, "character": 0})
            _write_one(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": [
                        {
                            "name": "fake_function",
                            "kind": 12,  # Function
                            "uri": uri,
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 0},
                            },
                            "selectionRange": {
                                "start": position,
                                "end": position,
                            },
                            "detail": "",
                        }
                    ],
                }
            )

        elif method == "callHierarchy/outgoingCalls":
            # No outgoing calls — this is a leaf function.
            # Produces status="complete" from compute_closure.
            _write_one({"jsonrpc": "2.0", "id": msg_id, "result": []})

        elif method == "shutdown":
            _write_one({"jsonrpc": "2.0", "id": msg_id, "result": None})

        elif method == "exit":
            sys.exit(0)

        # initialized, textDocument/didOpen, textDocument/didClose, and all
        # other notifications: no response.


if __name__ == "__main__":
    main()
