"""Interactive web dashboard for scry (``scry dashboard``).

Serves a self-contained single-page app over HTTP with two views:

* **Drift Overview** — drift score gauge, status breakdown, and a
  force-directed graph of anchors + links colored by drift status.
* **Anchor Explorer** — searchable, filterable table of all anchors
  with drill-down to linked neighbours and content preview.

No external dependencies beyond the Python stdlib — the frontend uses
D3.js loaded from a CDN.

**Read-only / follower-only:**  The dashboard never acquires the leader
lock and never writes to the DB or overlay files.  All data is gathered
via ``ScryDB(read_only=True)`` and ``LinkStore.replay()`` (read-only
JSONL replay).  This ensures the dashboard can run safely alongside a
leader MCP server or when no leader is running.

Usage::

    from scry.dashboard import serve_dashboard
    serve_dashboard(repo_root, port=5555)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data gathering
# ──────────────────────────────────────────────────────────────────────


def gather_dashboard_data(repo_root: Path) -> dict[str, Any]:
    """Read all anchors, links, and drift evaluations from the local index.

    Opens :class:`~scry.store.db.ScryDB` in read-only mode and replays
    links from the baseline + current branch overlay.  Returns a single
    JSON-serialisable dict suitable for the ``/api/data`` endpoint.
    """
    from scry.config import load_config
    from scry.drift import (
        DriftEvaluation,
        compute_drift_summary,
        evaluate_all_drift,
    )
    from scry.git_context import GitContextProvider
    from scry.models import AnchorType, Config
    from scry.store.db import ScryDB
    from scry.store.links import LinkStore
    from scry.store.overlay import OverlayManager

    try:
        config = load_config(repo_root)
    except Exception:
        config = Config()

    db = ScryDB(repo_root, read_only=True)
    try:
        anchors = db.list_anchors()

        link_store = LinkStore(repo_root)
        git_ctx_provider = GitContextProvider(repo_root)
        overlay_mgr = OverlayManager(
            repo_root, git_context=git_ctx_provider, link_store=link_store
        )
        overlay_path = overlay_mgr.current_overlay_path()

        class _BranchLinkStore(LinkStore):
            def replay(self, *, overlay_path: Path | None = None) -> Any:
                return super().replay(overlay_path=overlay_path or _ov_path)

        _ov_path = overlay_path
        branch_store = _BranchLinkStore(repo_root)

        evaluations: list[DriftEvaluation] = evaluate_all_drift(
            db=db, link_store=branch_store, config=config.drift
        )

        replay = branch_store.replay()

        # Drift summary
        code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)
        total_code = len(code_anchors)
        linked_code_ids = {
            lnk.from_id
            for lnk in replay.active_links.values()
            if lnk.from_type == AnchorType.CODE
        } | {
            lnk.to_id
            for lnk in replay.active_links.values()
            if lnk.to_type == AnchorType.CODE
        }
        linked_code = len(linked_code_ids & {a.id for a in code_anchors})
        coverage_total: int | None = total_code if replay.active_links else None

        summary = compute_drift_summary(
            evaluations,
            config=config.drift,
            coverage_total_code_anchors=coverage_total,
            coverage_linked_code_anchors=linked_code,
        )

        # Serialize anchors (lightweight — omit content_text for graph view)
        anchor_list = [
            {
                "id": a.id,
                "type": a.type,
                "path": a.path,
                "heading_path": a.heading_path,
                "symbol_name": a.symbol_name,
                "is_test": a.is_test,
                "language": _lang_from_path(a.path),
                "content_preview": (
                    a.content_text[:300] + "…" if len(a.content_text) > 300 else a.content_text
                ),
            }
            for a in anchors
        ]

        # Serialize evaluations (links + drift)
        eval_list = [
            {
                "link_id": ev.link.link_id,
                "from_id": ev.link.from_id,
                "from_type": str(ev.link.from_type),
                "to_id": ev.link.to_id,
                "to_type": str(ev.link.to_type),
                "link_type": str(ev.link.type),
                "drift_status": str(ev.drift_status),
                "semantic_drift": ev.semantic_drift,
                "evidence": ev.link.evidence,
            }
            for ev in evaluations
        ]

        # Summary counts
        counts = {
            k.replace("_", "-"): v for k, v in summary.counts.model_dump().items()
        }

        return {
            "repo_root": str(repo_root),
            "branch": _safe_branch(repo_root),
            "anchors": anchor_list,
            "evaluations": eval_list,
            "summary": {
                "drift_score": summary.drift_score,
                "coverage_score": summary.coverage_score,
                "counts": counts,
                "drift_coverage": summary.drift_coverage,
            },
            "anchor_type_counts": {
                "section": sum(1 for a in anchors if a.type == AnchorType.SECTION),
                "code": sum(1 for a in anchors if a.type == AnchorType.CODE),
                "code_in_doc": sum(1 for a in anchors if a.type == AnchorType.CODE_IN_DOC),
            },
        }
    finally:
        db.close()


def _safe_branch(repo_root: Path) -> str | None:
    """Best-effort current branch name."""
    try:
        from scry.git_context import GitContextProvider

        return GitContextProvider(repo_root).get().branch
    except Exception:
        return None


_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "tsx",
    ".js": "javascript", ".jsx": "jsx",
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
    ".md": "markdown", ".markdown": "markdown",
    ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".cs": "csharp", ".swift": "swift",
}


def _lang_from_path(path: str) -> str | None:
    """Derive a language tag from a file path's extension."""
    ext = Path(path).suffix.lower()
    return _EXT_TO_LANGUAGE.get(ext)


# ──────────────────────────────────────────────────────────────────────
# HTTP + WebSocket server
# ──────────────────────────────────────────────────────────────────────


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new thread.

    Required for WebSocket support: WS connections hold their handler
    thread open for the lifetime of the connection, so a single-threaded
    server would be unable to serve any other requests.
    """

    daemon_threads = True


class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves the SPA, ``/api/data`` JSON endpoint, and WebSocket upgrades.

    WebSocket connections are handled inline: after the 101 handshake,
    the handler blocks in a read loop until the client disconnects.
    ``handle()`` is overridden to skip the ``finish()`` cleanup when
    the connection has been upgraded — ``BaseHTTPRequestHandler``
    normally closes ``rfile``/``wfile`` (and the underlying socket)
    after ``handle()`` returns, which kills the WS connection.
    """

    protocol_version = "HTTP/1.1"

    repo_root: Path
    _ws_upgraded: bool = False  # set True when connection upgrades to WS

    def handle(self) -> None:
        """Override to skip finish() socket teardown for WS connections."""
        self._ws_upgraded = False
        try:
            super().handle()
        finally:
            # If WS, the socket is already managed by our read loop
            # and was closed there. Prevent finish() from double-closing.
            if self._ws_upgraded:
                # Neuter rfile/wfile so finish() doesn't close the socket
                import io

                self.rfile = io.BytesIO()
                self.wfile = io.BytesIO()

    def do_GET(self) -> None:  # noqa: N802
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self._handle_ws_upgrade()
            return
        if self.path == "/api/data":
            self._serve_api()
        elif self.path in ("/", "/index.html"):
            self._serve_html()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_api(self) -> None:
        try:
            data = gather_dashboard_data(self.repo_root)
            body = json.dumps(data).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            logger.exception("Dashboard API error")
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _serve_html(self) -> None:
        body = DASHBOARD_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def _handle_ws_upgrade(self) -> None:
        """WebSocket handshake + read loop (blocks until client disconnects)."""
        import base64
        import hashlib as _hl
        import struct

        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(
            _hl.sha1((key + "258EAFA5-E914-47DA-95CA-5AB9CF64AD42").encode()).digest()
        ).decode()

        # Send 101 directly on the raw socket to avoid any handler buffering.
        sock = self.request
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        sock.sendall(resp.encode())
        self._ws_upgraded = True
        self.close_connection = True

        # Detach rfile/wfile from the socket BEFORE entering the WS
        # read loop. The BufferedReader (rfile) may interfere with
        # raw socket reads if it holds the file descriptor.
        import io
        self.rfile = io.BytesIO()
        self.wfile = io.BytesIO()

        with _ws_lock:
            _ws_clients.append(sock)
        try:
            while True:
                h = _recv_exact(sock, 2)
                if h is None:
                    break
                opcode = h[0] & 0x0F
                if opcode == 0x8:
                    break
                length = h[1] & 0x7F
                if length == 126:
                    raw = _recv_exact(sock, 2)
                    if raw is None:
                        break
                    length = struct.unpack(">H", raw)[0]
                elif length == 127:
                    raw = _recv_exact(sock, 8)
                    if raw is None:
                        break
                    length = struct.unpack(">Q", raw)[0]
                if h[1] & 0x80:
                    if _recv_exact(sock, 4) is None:
                        break
                if length > 0:
                    if _recv_exact(sock, length) is None:
                        break
        except Exception:
            pass
        finally:
            with _ws_lock:
                if sock in _ws_clients:
                    _ws_clients.remove(sock)
            with contextlib.suppress(Exception):
                sock.close()

    def log_message(self, format: str, *args: Any) -> None:
        """Log requests via the module logger instead of stdout."""
        logger.debug("HTTP: %s", format % args)


def _ws_send_text(sock: Any, text: str) -> bool:
    """Send a WebSocket text frame. Returns False on failure."""
    import struct

    data = text.encode("utf-8")
    frame = bytearray()
    frame.append(0x81)  # FIN + text opcode
    if len(data) < 126:
        frame.append(len(data))
    elif len(data) < 65536:
        frame.append(126)
        frame.extend(struct.pack(">H", len(data)))
    else:
        frame.append(127)
        frame.extend(struct.pack(">Q", len(data)))
    frame.extend(data)
    try:
        sock.sendall(bytes(frame))
        return True
    except Exception:
        return False


def make_handler(repo_root: Path) -> type[_DashboardHandler]:
    """Create a handler class bound to a specific repo root."""

    class BoundHandler(_DashboardHandler):
        pass

    BoundHandler.repo_root = repo_root  # type: ignore[attr-defined]
    return BoundHandler


_ws_clients: list[Any] = []
_ws_lock = threading.Lock()


def _recv_exact(sock: Any, n: int) -> bytes | None:
    """Read exactly n bytes. Returns None on EOF/error."""
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except Exception:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


def _ws_push_loop(
    repo_root: Path,
    interval: float = 5.0,
) -> None:
    """Background thread: gather data and push to all WS clients when data changes."""
    import time

    last_hash = ""
    last_client_count = 0
    while True:
        time.sleep(interval)
        with _ws_lock:
            clients = list(_ws_clients)
        if not clients:
            last_client_count = 0
            continue
        try:
            data = gather_dashboard_data(repo_root)
            payload = json.dumps(data)
            data_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
            # Always push when new clients join (they need initial data),
            # or when data has changed.
            new_clients = len(clients) > last_client_count
            last_client_count = len(clients)
            if data_hash == last_hash and not new_clients:
                continue
            last_hash = data_hash
        except Exception:
            logger.exception("scry dashboard: push loop gather error")
            continue
        dead: list[Any] = []
        for sock in clients:
            if not _ws_send_text(sock, payload):
                dead.append(sock)
        if dead:
            with _ws_lock:
                for s in dead:
                    if s in _ws_clients:
                        _ws_clients.remove(s)


def serve_dashboard(
    repo_root: Path,
    *,
    port: int = 5555,
    open_browser: bool = True,
) -> None:
    """Start the dashboard HTTP server (blocks until interrupted).

    Args:
        repo_root:    Absolute path to the repository root.
        port:         TCP port to listen on (WS upgrades on same port).
        open_browser: Open the dashboard URL in the default browser.
    """
    handler_cls = make_handler(repo_root)
    server = _ThreadedHTTPServer(("127.0.0.1", port), handler_cls)
    url = f"http://127.0.0.1:{port}"
    logger.info("scry dashboard: serving at %s", url)
    print(f"scry dashboard → {url}  (Ctrl-C to stop)")  # noqa: T201

    # Start WS push thread.
    push_thread = threading.Thread(
        target=_ws_push_loop, args=(repo_root, 5.0),
        daemon=True, name="scry-ws-push",
    )
    push_thread.start()

    if open_browser:
        import webbrowser

        threading.Timer(0.5, partial(webbrowser.open, url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        print("\nscry dashboard stopped.")  # noqa: T201


# ──────────────────────────────────────────────────────────────────────
# Embedded HTML/JS/CSS frontend
# ──────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>scry dashboard</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {
  --bg: #0d1117; --bg2: #161b22; --fg: #c9d1d9; --fg2: #8b949e;
  --accent: #58a6ff; --green: #3fb950; --yellow: #d29922;
  --orange: #db6d28; --red: #f85149; --gray: #484f58;
  --border: #30363d; --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       background: var(--bg); color: var(--fg); line-height: 1.5; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header { background: var(--bg2); border-bottom: 1px solid var(--border);
          padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 18px; font-weight: 600; }
.header h1 span { color: var(--accent); }
.header .branch { color: var(--fg2); font-size: 14px; }

/* Tabs */
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border);
        background: var(--bg2); padding: 0 24px; }
.tab { padding: 10px 20px; cursor: pointer; color: var(--fg2);
       border-bottom: 2px solid transparent; font-size: 14px; font-weight: 500;
       transition: all 0.15s; }
.tab:hover { color: var(--fg); }
.tab.active { color: var(--fg); border-bottom-color: var(--accent); }

/* Panels */
.panel { display: none; padding: 24px; }
.panel.active { display: block; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
         gap: 16px; margin-bottom: 24px; }
.card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
        padding: 16px; }
.card .label { font-size: 12px; color: var(--fg2); text-transform: uppercase;
               letter-spacing: 0.05em; margin-bottom: 4px; }
.card .value { font-size: 28px; font-weight: 700; }
.card .sub { font-size: 12px; color: var(--fg2); margin-top: 4px; }

/* Drift gauge */
.gauge-wrap { display: flex; align-items: center; gap: 8px; }
.gauge-bar { flex: 1; height: 8px; background: var(--border); border-radius: 4px;
             overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }

/* Status pills */
.status-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px;
        border-radius: 999px; font-size: 13px; font-weight: 500;
        background: var(--bg2); border: 1px solid var(--border); }
.pill .dot { width: 8px; height: 8px; border-radius: 50%; }

/* Graph */
#graph-container { background: var(--bg2); border: 1px solid var(--border);
                   border-radius: var(--radius); position: relative; overflow: hidden; }
#graph-container svg { width: 100%; display: block; }

/* Explorer table */
.search-bar { display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }
.search-bar input { padding: 8px 12px; background: var(--bg2); border: 1px solid var(--border);
                    border-radius: var(--radius); color: var(--fg); font-size: 14px;
                    flex: 1; min-width: 200px; outline: none; }
.search-bar input:focus { border-color: var(--accent); }
.search-bar select { padding: 8px 12px; background: var(--bg2); border: 1px solid var(--border);
                     border-radius: var(--radius); color: var(--fg); font-size: 14px; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th { text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--border);
           color: var(--fg2); font-weight: 600; position: sticky; top: 0;
           background: var(--bg); }
tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border);
           max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
tbody tr:hover { background: var(--bg2); }
.table-wrap { max-height: 600px; overflow-y: auto; border: 1px solid var(--border);
              border-radius: var(--radius); }

/* Anchor detail overlay */
.detail-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6);
                  display: none; z-index: 100; justify-content: center; align-items: start;
                  padding: 60px 24px; overflow-y: auto; }
.detail-overlay.open { display: flex; }
.detail-card { background: var(--bg2); border: 1px solid var(--border);
               border-radius: var(--radius); padding: 24px; max-width: 800px;
               width: 100%; position: relative; }
.detail-card .close-btn { position: absolute; top: 12px; right: 16px; cursor: pointer;
                          font-size: 20px; color: var(--fg2); background: none; border: none; }
.detail-card .close-btn:hover { color: var(--fg); }
.detail-card h2 { font-size: 16px; margin-bottom: 12px; word-break: break-all; }
.detail-card pre { background: var(--bg); border: 1px solid var(--border);
                   border-radius: var(--radius); padding: 12px; font-size: 12px;
                   overflow-x: auto; white-space: pre-wrap; max-height: 300px;
                   overflow-y: auto; margin: 12px 0; }
.detail-card .links-list { margin-top: 12px; }
.detail-card .link-item { padding: 8px; border-bottom: 1px solid var(--border);
                          font-size: 13px; display: flex; justify-content: space-between; }

/* Tooltip */
.tooltip { position: absolute; padding: 8px 12px; background: var(--bg2);
           border: 1px solid var(--border); border-radius: var(--radius);
           font-size: 12px; pointer-events: none; z-index: 50;
           max-width: 350px; word-break: break-all; }

/* Loading */
.loading { text-align: center; padding: 60px; color: var(--fg2); }
.loading .spinner { display: inline-block; width: 32px; height: 32px;
                    border: 3px solid var(--border); border-top-color: var(--accent);
                    border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Graph legend */
.legend { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0;
          font-size: 12px; color: var(--fg2); }
.legend-item { display: flex; align-items: center; gap: 4px; }
.legend-swatch { width: 12px; height: 3px; border-radius: 2px; }

/* Graph physics controls */
.physics-controls { display: flex; flex-wrap: wrap; gap: 16px; padding: 12px 16px;
                    background: var(--bg); border: 1px solid var(--border);
                    border-radius: var(--radius); margin-bottom: 12px;
                    align-items: center; }
.physics-controls .ctrl { display: flex; align-items: center; gap: 6px; font-size: 12px;
                          color: var(--fg2); min-width: 0; }
.physics-controls .ctrl label { white-space: nowrap; min-width: 70px; }
.physics-controls .ctrl input[type=range] { width: 100px; accent-color: var(--accent);
                                            cursor: pointer; }
.physics-controls .ctrl .val { font-variant-numeric: tabular-nums; min-width: 36px;
                               text-align: right; color: var(--fg); font-size: 11px; }
</style>
</head>
<body>

<div class="header">
  <h1>🔮 <span>scry</span> dashboard</h1>
  <span class="branch" id="branch-label"></span>
  <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
    <span id="ws-status" style="font-size:11px;color:var(--fg2)">⏳ connecting…</span>
    <span id="last-updated" style="font-size:12px;color:var(--fg2)"></span>
    <button id="refresh-btn" title="Refresh data now" style="padding:4px 12px;background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);color:var(--fg);cursor:pointer;font-size:13px">↻ Refresh</button>
    <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--fg2);cursor:pointer" title="When enabled, dashboard auto-updates via WebSocket when scry data changes">
      <input type="checkbox" id="auto-refresh-toggle" checked> Live
    </label>
  </div>
</div>

<div class="tabs">
  <div class="tab active" data-panel="overview">Drift Overview</div>
  <div class="tab" data-panel="explorer">Anchor Explorer</div>
</div>

<div id="loading" class="loading"><div class="spinner"></div><p>Loading index…</p></div>

<div id="overview" class="panel active">
</div>

<div id="explorer" class="panel">
  <div class="search-bar">
    <input type="text" id="search-input" placeholder="Search anchors by ID, path, or symbol…">
    <select id="type-filter">
      <option value="">All types</option>
      <option value="section">section</option>
      <option value="code">code</option>
      <option value="code_in_doc">code_in_doc</option>
    </select>
    <select id="drift-filter">
      <option value="">All drift statuses</option>
      <option value="fresh">fresh</option>
      <option value="code-changed">code-changed</option>
      <option value="spec-changed">spec-changed</option>
      <option value="both-changed">both-changed</option>
      <option value="broken-source">broken-source</option>
      <option value="broken-target">broken-target</option>
    </select>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr>
        <th>Anchor ID</th><th>Type</th><th>Path</th><th>Drift</th><th>Links</th>
      </tr></thead>
      <tbody id="anchor-tbody"></tbody>
    </table>
  </div>
</div>

<div class="detail-overlay" id="detail-overlay">
  <div class="detail-card">
    <button class="close-btn" id="detail-close">&times;</button>
    <h2 id="detail-title"></h2>
    <div id="detail-meta"></div>
    <pre id="detail-content"></pre>
    <div class="links-list" id="detail-links"></div>
  </div>
</div>

<div class="tooltip" id="tooltip" style="display:none;"></div>

<script>
const DRIFT_COLORS = {
  'fresh': '#3fb950', 'code-changed': '#db6d28', 'spec-changed': '#d29922',
  'both-changed': '#f85149', 'broken-source': '#484f58', 'broken-target': '#484f58',
  'merge-conflict': '#f85149', 'drift-unknown': '#8b949e'
};
const TYPE_COLORS = { 'section': '#58a6ff', 'code': '#3fb950', 'code_in_doc': '#d29922' };

let DATA = null;
let anchorIndex = {};  // id -> anchor
let anchorDrift = {};  // anchor_id -> worst drift status
let anchorLinks = {};  // anchor_id -> [evaluation]

console.log('scry dashboard: script loaded, v2');

// ── Tab switching ──
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.panel).classList.add('active');
  });
});

// ── HTML escaping ──
function esc(s) {
  if (s == null) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

// ── Data loading + WebSocket live updates ──
let _ws = null;
let _currentSim = null;  // track active D3 simulation for cleanup

function applyData(data) {
  DATA = data;
  document.getElementById('loading').style.display = 'none';
  document.getElementById('branch-label').textContent = DATA.branch ? `branch: ${DATA.branch}` : '';
  document.getElementById('last-updated').textContent = `updated ${new Date().toLocaleTimeString()}`;

  // Reset indices
  anchorIndex = {}; anchorDrift = {}; anchorLinks = {};
  DATA.anchors.forEach(a => anchorIndex[a.id] = a);
  DATA.evaluations.forEach(ev => {
    [ev.from_id, ev.to_id].forEach(aid => {
      if (!anchorLinks[aid]) anchorLinks[aid] = [];
      anchorLinks[aid].push(ev);
    });
    [ev.from_id, ev.to_id].forEach(aid => {
      const cur = anchorDrift[aid];
      if (!cur || driftSeverity(ev.drift_status) > driftSeverity(cur))
        anchorDrift[aid] = ev.drift_status;
    });
  });

  // Clean up old simulation before rebuilding
  if (_currentSim) { _currentSim.stop(); _currentSim = null; }

  renderOverview();
  renderExplorer();
}

function loadData() {
  const btn = document.getElementById('refresh-btn');
  btn.disabled = true; btn.textContent = '↻ …';
  return fetch('/api/data')
    .then(r => {
      if (!r.ok) return r.json().then(d => { throw new Error(d.error || r.statusText); });
      return r.json();
    })
    .then(data => {
      if (!data.anchors || !data.summary) throw new Error('Unexpected response');
      applyData(data);
    })
    .catch(err => {
      document.getElementById('loading').style.display = 'block';
      document.getElementById('loading').innerHTML =
        `<p style="color:#f85149">Failed to load: ${esc(err.message)}</p>`;
    })
    .finally(() => { btn.disabled = false; btn.textContent = '↻ Refresh'; });
}

function connectWS() {
  if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) return;
  const url = `ws://${location.host}/ws`;
  console.log('scry: connecting WebSocket to', url);
  const ws = new WebSocket(url);
  _ws = ws;
  const statusEl = document.getElementById('ws-status');

  ws.onopen = () => {
    console.log('scry: WS connected');
    statusEl.textContent = '🟢 live';
    statusEl.style.color = 'var(--green)';
  };
  ws.onclose = (e) => {
    console.log('scry: WS closed', e.code, e.reason);
    // Only update UI if this is still the active socket
    if (_ws === ws) {
      _ws = null;
      statusEl.textContent = '🔴 disconnected';
      statusEl.style.color = 'var(--red)';
      if (document.getElementById('auto-refresh-toggle').checked)
        setTimeout(connectWS, 3000);
    }
  };
  ws.onerror = (e) => {
    console.error('scry: WS error', e);
    // onclose will fire after onerror — let it handle cleanup
  };
  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      if (data.anchors && data.summary) {
        console.log('scry: WS push received, anchors:', data.anchors.length);
        applyData(data);
      }
    } catch(e) { console.error('scry: WS message parse error', e); }
  };
}

function disconnectWS() {
  const ws = _ws;
  _ws = null;  // clear first so onclose doesn't reconnect
  if (ws) { ws.onclose = null; ws.close(); }
  document.getElementById('ws-status').textContent = '⏸ paused';
  document.getElementById('ws-status').style.color = 'var(--fg2)';
}

document.getElementById('refresh-btn').addEventListener('click', loadData);
document.getElementById('auto-refresh-toggle').addEventListener('change', e => {
  if (e.target.checked) connectWS(); else disconnectWS();
});

// Initial load via HTTP, then connect WS for live updates.
console.log('scry: calling loadData + connectWS');
loadData();
if (document.getElementById('auto-refresh-toggle').checked) {
  connectWS();
}

function driftSeverity(status) {
  const order = ['fresh','drift-unknown','code-changed','spec-changed',
                 'both-changed','broken-source','broken-target','merge-conflict'];
  return order.indexOf(status);
}

// ── Overview panel ──
function renderOverview() {
  const s = DATA.summary;
  const c = s.counts;
  const panel = document.getElementById('overview');

  // Score cards
  const score = s.drift_score != null ? s.drift_score.toFixed(1) : 'N/A';
  const scoreColor = s.drift_score != null
    ? (s.drift_score >= 80 ? 'var(--green)' : s.drift_score >= 50 ? 'var(--yellow)' : 'var(--red)')
    : 'var(--fg2)';
  const coverage = s.coverage_score != null ? s.coverage_score.toFixed(1) + '%' : 'N/A';
  const total = c.total || 0;

  let html = `<div class="cards">
    <div class="card">
      <div class="label">Drift Score</div>
      <div class="value" style="color:${scoreColor}">${score}</div>
      <div class="gauge-wrap"><div class="gauge-bar">
        <div class="gauge-fill" style="width:${s.drift_score||0}%;background:${scoreColor}"></div>
      </div></div>
      <div class="sub">100 = all fresh, 0 = all drifted</div>
    </div>
    <div class="card">
      <div class="label">Total Anchors</div>
      <div class="value">${DATA.anchors.length.toLocaleString()}</div>
      <div class="sub">${DATA.anchor_type_counts.section} sections · ${DATA.anchor_type_counts.code} code · ${DATA.anchor_type_counts.code_in_doc} code-in-doc</div>
    </div>
    <div class="card">
      <div class="label">Active Links</div>
      <div class="value">${total}</div>
      <div class="sub">${c.fresh||0} fresh · ${total - (c.fresh||0)} drifted</div>
    </div>
    <div class="card">
      <div class="label">Coverage</div>
      <div class="value">${coverage}</div>
      <div class="sub">code anchors with ≥1 link</div>
    </div>
  </div>`;

  // Status pills
  html += '<div class="status-grid">';
  for (const [status, color] of Object.entries(DRIFT_COLORS)) {
    const count = c[status] || 0;
    if (count > 0 || status === 'fresh')
      html += `<div class="pill"><span class="dot" style="background:${color}"></span>${status}: ${count}</div>`;
  }
  html += '</div>';

  // Legend
  const usedLangs = new Set(DATA.anchors.filter(a => a.language).map(a => a.language));
  html += `<div class="legend">
    <strong style="color:var(--fg)">Edges:</strong>
    ${Object.entries(DRIFT_COLORS).map(([s,c]) =>
      `<span class="legend-item"><span class="legend-swatch" style="background:${c}"></span>${s}</span>`
    ).join('')}
  </div>
  <div class="legend">
    <strong style="color:var(--fg)">Nodes:</strong>
    ${Object.entries(TYPE_COLORS).map(([t,c]) =>
      `<span class="legend-item"><span class="legend-swatch" style="background:${c};height:8px;width:8px;border-radius:50%"></span>${t}</span>`
    ).join('')}
    &nbsp;·&nbsp;
    <strong style="color:var(--fg)">Language badges:</strong>
    ${[...usedLangs].sort().map(l =>
      `<span class="legend-item"><span class="legend-swatch" style="background:${LANG_COLORS[l]||'#8b949e'};height:6px;width:6px;border-radius:50%"></span>${l}</span>`
    ).join('')}
  </div>`;

  // Graph container
  html += '<div id="graph-controls"></div>';
  html += '<div id="graph-container"></div>';
  panel.innerHTML = html;

  renderGraph();
}

// ── Force-directed graph ──
const LANG_COLORS = {
  'python':'#3572A5','typescript':'#3178c6','javascript':'#f1e05a','tsx':'#3178c6',
  'jsx':'#f1e05a','go':'#00ADD8','rust':'#dea584','zig':'#f7a41d',
  'markdown':'#083fa1','java':'#b07219','kotlin':'#A97BFF','c':'#555555',
  'cpp':'#f34b7d','ruby':'#701516','csharp':'#178600','swift':'#F05138',
};

function renderGraph() {
  const container = document.getElementById('graph-container');
  if (!container) return;
  // Clear previous graph content
  container.innerHTML = '';
  const width = container.clientWidth || 900;
  const height = 600;
  container.style.height = height + 'px';

  // Only show anchors that have links
  const linkedIds = new Set();
  DATA.evaluations.forEach(ev => { linkedIds.add(ev.from_id); linkedIds.add(ev.to_id); });

  const nodes = DATA.anchors
    .filter(a => linkedIds.has(a.id))
    .map(a => ({
      id: a.id, type: a.type, path: a.path, symbol_name: a.symbol_name,
      language: a.language,
      label: a.symbol_name || a.id.split('::').pop() || a.id
    }));
  const nodeSet = new Set(nodes.map(n => n.id));

  const links = DATA.evaluations
    .filter(ev => nodeSet.has(ev.from_id) && nodeSet.has(ev.to_id))
    .map(ev => ({
      source: ev.from_id, target: ev.to_id,
      drift_status: ev.drift_status, link_type: ev.link_type
    }));

  if (nodes.length === 0) {
    container.innerHTML = '<div style="padding:40px;text-align:center;color:var(--fg2)">No links to display. Create links with <code>scry link</code> or the MCP <code>propose_link</code> tool.</div>';
    return;
  }

  const MAX_GRAPH_NODES = 300;
  if (nodes.length > MAX_GRAPH_NODES) {
    container.innerHTML = `<div style="padding:40px;text-align:center;color:var(--fg2)">
      Graph disabled: ${nodes.length} linked anchors exceeds the ${MAX_GRAPH_NODES}-node rendering limit.<br>
      Use the <strong>Anchor Explorer</strong> tab to browse anchors and links.
    </div>`;
    return;
  }

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', [0, 0, width, height]);

  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.1, 8]).on('zoom', e => g.attr('transform', e.transform)));

  // Arrow markers
  svg.append('defs').selectAll('marker')
    .data(Object.keys(DRIFT_COLORS)).enter().append('marker')
    .attr('id', d => 'arrow-' + d).attr('viewBox', '0 -5 10 10')
    .attr('refX', 18).attr('refY', 0).attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', d => DRIFT_COLORS[d]);

  // ── Physics controls ──
  const cx = width / 2, cy = height / 2;
  const defaults = {
    charge: -40, linkDist: 50, linkStrength: 0.7, collide: 12,
    gravity: 0.15, alpha: 0.5
  };
  const P = { ...defaults };
  const ctrlBox = document.getElementById('graph-controls');
  ctrlBox.innerHTML = `<div class="physics-controls">
    <div class="ctrl"><label>Repulsion</label><input type="range" id="p-charge" min="-300" max="0" step="5" value="${P.charge}"><span class="val" id="v-charge">${P.charge}</span></div>
    <div class="ctrl"><label>Link dist</label><input type="range" id="p-linkDist" min="10" max="200" step="5" value="${P.linkDist}"><span class="val" id="v-linkDist">${P.linkDist}</span></div>
    <div class="ctrl"><label>Link pull</label><input type="range" id="p-linkStr" min="0" max="1" step="0.05" value="${P.linkStrength}"><span class="val" id="v-linkStr">${P.linkStrength}</span></div>
    <div class="ctrl"><label>Collision</label><input type="range" id="p-collide" min="0" max="40" step="1" value="${P.collide}"><span class="val" id="v-collide">${P.collide}</span></div>
    <div class="ctrl"><label>Gravity</label><input type="range" id="p-gravity" min="0" max="1" step="0.01" value="${P.gravity}"><span class="val" id="v-gravity">${P.gravity}</span></div>
    <div class="ctrl"><label>Reheat</label><input type="range" id="p-alpha" min="0.05" max="1" step="0.05" value="${P.alpha}"><span class="val" id="v-alpha">${P.alpha}</span></div>
    <button style="padding:4px 12px;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);color:var(--fg);cursor:pointer;font-size:12px" id="p-reset">Reset</button>
  </div>`;

  // Gravity: forceX + forceY pull all nodes toward center
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(P.linkDist).strength(P.linkStrength))
    .force('charge', d3.forceManyBody().strength(P.charge))
    .force('x', d3.forceX(cx).strength(P.gravity))
    .force('y', d3.forceY(cy).strength(P.gravity))
    .force('collide', d3.forceCollide(P.collide))
    .alphaDecay(0.02)
    .velocityDecay(0.4);
  _currentSim = sim;

  function reheat() { sim.alpha(P.alpha).restart(); }

  function bindSlider(sid, vid, fn) {
    const s = document.getElementById(sid), v = document.getElementById(vid);
    s.addEventListener('input', () => { const n = parseFloat(s.value); v.textContent = n; fn(n); reheat(); });
  }
  bindSlider('p-charge', 'v-charge', v => { P.charge = v; sim.force('charge').strength(v); });
  bindSlider('p-linkDist', 'v-linkDist', v => { P.linkDist = v; sim.force('link').distance(v); });
  bindSlider('p-linkStr', 'v-linkStr', v => { P.linkStrength = v; sim.force('link').strength(v); });
  bindSlider('p-collide', 'v-collide', v => { P.collide = v; sim.force('collide').radius(v); });
  bindSlider('p-gravity', 'v-gravity', v => { P.gravity = v; sim.force('x').strength(v); sim.force('y').strength(v); });
  bindSlider('p-alpha', 'v-alpha', v => { P.alpha = v; });
  document.getElementById('p-reset').addEventListener('click', () => {
    Object.assign(P, defaults);
    sim.force('charge').strength(P.charge);
    sim.force('link').distance(P.linkDist).strength(P.linkStrength);
    sim.force('collide').radius(P.collide);
    sim.force('x').strength(P.gravity); sim.force('y').strength(P.gravity);
    ['charge','linkDist','linkStr','collide','gravity','alpha'].forEach((k,i) => {
      const keys = ['charge','linkDist','linkStrength','collide','gravity','alpha'];
      document.getElementById('p-' + k).value = P[keys[i]];
      document.getElementById('v-' + k).textContent = P[keys[i]];
    });
    reheat();
  });

  // ── Edges ──
  const linkEl = g.append('g').selectAll('line').data(links).enter().append('line')
    .attr('stroke', d => DRIFT_COLORS[d.drift_status] || '#484f58')
    .attr('stroke-width', 1.5).attr('stroke-opacity', 0.6)
    .attr('marker-end', d => `url(#arrow-${d.drift_status})`);

  // ── Node groups (circle + language badge + label) ──
  const nodeG = g.append('g').selectAll('g').data(nodes).enter().append('g')
    .style('cursor', 'pointer')
    .call(d3.drag().on('start', dragStart).on('drag', dragging).on('end', dragEnd));

  // Main circle
  nodeG.append('circle')
    .attr('r', d => d.type === 'section' ? 7 : 5)
    .attr('fill', d => TYPE_COLORS[d.type] || '#8b949e')
    .attr('stroke', d => anchorDrift[d.id] ? DRIFT_COLORS[anchorDrift[d.id]] : '#30363d')
    .attr('stroke-width', 1.5);

  // Language badge (small colored dot below-right)
  nodeG.filter(d => d.language && LANG_COLORS[d.language])
    .append('circle')
    .attr('r', 3).attr('cx', 6).attr('cy', 6)
    .attr('fill', d => LANG_COLORS[d.language])
    .attr('stroke', 'var(--bg)').attr('stroke-width', 0.5);

  // Labels — always shown, truncated
  nodeG.append('text')
    .text(d => { const l = d.label; return l.length > 20 ? l.slice(0,17) + '…' : l; })
    .attr('font-size', 8).attr('fill', '#8b949e')
    .attr('dx', 10).attr('dy', 3)
    .style('pointer-events', 'none');

  // Tooltip
  const tooltip = document.getElementById('tooltip');
  nodeG.on('mouseover', (e, d) => {
    tooltip.style.display = 'block';
    const lang = d.language ? `<span style="color:${LANG_COLORS[d.language] || 'var(--fg2)'}">${esc(d.language)}</span> · ` : '';
    tooltip.innerHTML = `<strong>${esc(d.label)}</strong><br>
      ${lang}<span style="color:${TYPE_COLORS[d.type]}">${esc(d.type)}</span> · ${esc(d.path)}
      ${anchorDrift[d.id] ? `<br>drift: <span style="color:${DRIFT_COLORS[anchorDrift[d.id]]}">${anchorDrift[d.id]}</span>` : ''}`;
  }).on('mousemove', e => {
    tooltip.style.left = (e.pageX + 12) + 'px';
    tooltip.style.top = (e.pageY - 10) + 'px';
  }).on('mouseout', () => { tooltip.style.display = 'none'; })
    .on('click', (e, d) => showDetail(d.id));

  // Tick
  sim.on('tick', () => {
    linkEl.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
           .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    nodeG.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Auto-fit after simulation settles
  sim.on('end', () => {
    const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
    const pad = 40;
    const x0 = Math.min(...xs) - pad, y0 = Math.min(...ys) - pad;
    const x1 = Math.max(...xs) + pad + 80, y1 = Math.max(...ys) + pad;
    const bw = x1 - x0, bh = y1 - y0;
    const scale = Math.min(width / bw, height / bh, 2);
    const tx = (width - bw * scale) / 2 - x0 * scale;
    const ty = (height - bh * scale) / 2 - y0 * scale;
    svg.transition().duration(750).call(
      d3.zoom().transform, d3.zoomIdentity.translate(tx, ty).scale(scale)
    );
  });

  function dragStart(e, d) { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
  function dragging(e, d) { d.fx = e.x; d.fy = e.y; }
  function dragEnd(e, d) { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }
}

// ── Explorer panel ──
function renderExplorer() {
  const tbody = document.getElementById('anchor-tbody');
  const input = document.getElementById('search-input');
  const typeFilter = document.getElementById('type-filter');
  const driftFilter = document.getElementById('drift-filter');

  function render() {
    const q = input.value.toLowerCase();
    const tf = typeFilter.value;
    const df = driftFilter.value;

    let filtered = DATA.anchors;
    if (q) filtered = filtered.filter(a =>
      a.id.toLowerCase().includes(q) ||
      a.path.toLowerCase().includes(q) ||
      (a.symbol_name && a.symbol_name.toLowerCase().includes(q))
    );
    if (tf) filtered = filtered.filter(a => a.type === tf);
    if (df) filtered = filtered.filter(a => anchorDrift[a.id] === df);

    // Limit to 500 rows for performance
    const shown = filtered.slice(0, 500);
    tbody.innerHTML = shown.map(a => {
      const drift = anchorDrift[a.id] || '';
      const driftColor = DRIFT_COLORS[drift] || 'transparent';
      const linkCount = (anchorLinks[a.id] || []).length;
      const safeId = esc(a.id);
      const escapedId = a.id.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
      return `<tr style="cursor:pointer" onclick="showDetail('${escapedId}')">
        <td title="${safeId}">${safeId}</td>
        <td><span style="color:${TYPE_COLORS[a.type]}">${esc(a.type)}</span></td>
        <td>${esc(a.path)}</td>
        <td>${drift ? `<span class="pill" style="padding:2px 8px"><span class="dot" style="background:${driftColor}"></span>${esc(drift)}</span>` : '—'}</td>
        <td>${linkCount || '—'}</td>
      </tr>`;
    }).join('');
    if (filtered.length > 500) {
      tbody.innerHTML += `<tr><td colspan="5" style="color:var(--fg2);text-align:center">
        Showing 500 of ${filtered.length.toLocaleString()} results. Refine your search.</td></tr>`;
    }
  }

  input.addEventListener('input', render);
  typeFilter.addEventListener('change', render);
  driftFilter.addEventListener('change', render);
  render();
}

// ── Anchor detail overlay ──
function showDetail(anchorId) {
  const overlay = document.getElementById('detail-overlay');
  const a = anchorIndex[anchorId];
  document.getElementById('detail-title').textContent = anchorId;

  if (a) {
    document.getElementById('detail-meta').innerHTML =
      `<span style="color:${TYPE_COLORS[a.type]}">${esc(a.type)}</span> · ${esc(a.path)}` +
      (a.symbol_name ? ` · <code>${esc(a.symbol_name)}</code>` : '') +
      (a.is_test ? ' · <span style="color:var(--yellow)">test</span>' : '');
    document.getElementById('detail-content').textContent = a.content_preview || '(no content)';
  } else {
    document.getElementById('detail-meta').innerHTML = '<span style="color:var(--fg2)">Anchor not in index (broken link endpoint)</span>';
    document.getElementById('detail-content').textContent = '';
  }

  const links = anchorLinks[anchorId] || [];
  const linksHtml = links.length
    ? links.map(ev => {
        const other = ev.from_id === anchorId ? ev.to_id : ev.from_id;
        const dir = ev.from_id === anchorId ? '→' : '←';
        const escapedOther = other.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        return `<div class="link-item">
          <span>${dir} <a href="#" onclick="event.preventDefault();showDetail('${escapedOther}')">${esc(other)}</a></span>
          <span><span class="pill" style="padding:2px 8px"><span class="dot" style="background:${DRIFT_COLORS[ev.drift_status]}"></span>${esc(ev.drift_status)}</span>
          <span style="color:var(--fg2)">${esc(ev.link_type)}</span></span>
        </div>`;
      }).join('')
    : '<div style="color:var(--fg2);padding:8px">No links</div>';
  document.getElementById('detail-links').innerHTML =
    `<div class="label" style="margin-bottom:8px">Links (${links.length})</div>` + linksHtml;

  overlay.classList.add('open');
}
// Make showDetail available globally for inline onclick handlers
window.showDetail = showDetail;

document.getElementById('detail-close').addEventListener('click', () => {
  document.getElementById('detail-overlay').classList.remove('open');
});
document.getElementById('detail-overlay').addEventListener('click', e => {
  if (e.target === e.currentTarget)
    document.getElementById('detail-overlay').classList.remove('open');
});
</script>
</body>
</html>"""
