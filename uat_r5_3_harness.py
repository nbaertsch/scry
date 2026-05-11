"""UAT-R5-3 — MCP Integration After Round-4 Fixes.

Run as:  uv run python uat_r5_3_harness.py
Output: uat_r5_3_results.json  (raw logs) + printed narrative
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ─── Config ──────────────────────────────────────────────────────────────────

SCRY_REPO = Path(r"C:\Users\noahbaertsch\Projects\scry")
# Run scry from inside the repo so cwd-based repo root detection works.
SCRY_CMD = ["uv", "run", "scry", "mcp"]
LOG_FILE = SCRY_REPO / "uat_r5_3_results.json"

# ─── MCP client (minimal, synchronous) ───────────────────────────────────────

class MCPClient:
    """Minimal synchronous JSON-RPC 2.0 / MCP client over subprocess stdio."""

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:  # type: ignore[type-arg]
        self._proc = proc
        self._seq = 0
        self.log: list[dict[str, Any]] = []

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    def _send(self, payload: dict[str, Any]) -> None:
        raw = (json.dumps(payload) + "\n").encode()
        print(f"\n→ SEND  {payload.get('method', '?')} id={payload.get('id', '-')}", file=sys.stderr)
        assert self._proc.stdin is not None
        self._proc.stdin.write(raw)
        self._proc.stdin.flush()

    def _recv(self, timeout: float = 30.0) -> dict[str, Any]:
        """Read one JSON line from stdout."""
        assert self._proc.stdout is not None
        start = time.time()
        buf = b""
        while time.time() - start < timeout:
            ch = self._proc.stdout.read(1)
            if ch == b"":
                raise EOFError("scry MCP process closed stdout")
            buf += ch
            if ch == b"\n":
                text = buf.decode().strip()
                if text:
                    obj: dict[str, Any] = json.loads(text)
                    print(f"← RECV  id={obj.get('id', '-')}", file=sys.stderr)
                    return obj
                buf = b""
        raise TimeoutError(f"No response within {timeout}s")

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        rid = self._next_id()
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        while True:
            resp = self._recv(timeout=timeout)
            if resp.get("id") == rid:
                entry = {"request": payload, "response": resp, "ts": time.time()}
                self.log.append(entry)
                return resp
            # Discard notifications (id is absent or None)

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def tool_call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )


def start_server() -> tuple[subprocess.Popen[bytes], MCPClient]:  # type: ignore[type-arg]
    env = os.environ.copy()
    # Suppress authlib deprecation noise that would corrupt the stdio stream.
    env.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
    proc = subprocess.Popen(
        SCRY_CMD,
        cwd=str(SCRY_REPO),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    client = MCPClient(proc)
    return proc, client


# ─── Result helpers ───────────────────────────────────────────────────────────

def result_content(resp: dict[str, Any]) -> Any:
    """Extract the tool result payload (unwrap FastMCP's text-content wrapping)."""
    result = resp.get("result", {})
    if isinstance(result, dict):
        content = result.get("content", [])
        if content and isinstance(content, list):
            # FastMCP wraps results in [{type: "text", text: "<json>"}]
            first = content[0]
            if isinstance(first, dict) and first.get("type") == "text":
                try:
                    return json.loads(first["text"])
                except (json.JSONDecodeError, KeyError):
                    return first.get("text", first)
        return result
    return result


def is_error(resp: dict[str, Any]) -> bool:
    return "error" in resp or (
        isinstance(resp.get("result"), dict)
        and resp["result"].get("isError") is True
    )


def error_msg(resp: dict[str, Any]) -> str:
    if "error" in resp:
        err = resp["error"]
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    result = resp.get("result", {})
    if isinstance(result, dict) and result.get("isError"):
        for c in result.get("content", []):
            if isinstance(c, dict) and c.get("type") == "text":
                return c["text"]
    return str(result)


def count_tokens_approx(obj: Any) -> int:
    """Rough token estimate: len(json_text) / 4."""
    return len(json.dumps(obj)) // 4


# ─── Observations ─────────────────────────────────────────────────────────────

obs: list[str] = []  # Ordered narrative observations from "Claude's POV"


def note(tag: str, msg: str) -> None:
    obs.append(f"[{tag}] {msg}")
    print(f"  OBS [{tag}] {msg}", file=sys.stderr)


# ─── Main harness ─────────────────────────────────────────────────────────────

def run() -> None:
    print("=== UAT-R5-3: Starting scry MCP server ===", file=sys.stderr)
    proc, client = start_server()

    try:
        _run_workflow(client)
    finally:
        # Write log FIRST before any blocking wait
        with open(LOG_FILE, "w") as f:
            json.dump(client.log, f, indent=2, default=str)
        print(f"\n[LOG] Full request/response log written to {LOG_FILE}", file=sys.stderr)
        try:
            proc.stdin.close()  # type: ignore[union-attr]
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()  # Force-kill if graceful shutdown times out


def _run_workflow(client: MCPClient) -> None:

    # ── Step 1: Initialize ────────────────────────────────────────────────────
    print("\n=== Step 1: initialize ===", file=sys.stderr)
    init_resp = client.request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "uat-r5-3-claude-sim", "version": "1.0"},
        },
        timeout=120,  # cold-start indexing can be slow
    )
    if is_error(init_resp):
        print(f"FATAL: initialize failed: {error_msg(init_resp)}", file=sys.stderr)
        return

    server_info = init_resp.get("result", {})
    note("INIT", f"serverInfo={server_info.get('serverInfo', {})} capabilities={list(server_info.get('capabilities', {}).keys())}")

    client.notify("notifications/initialized")

    # ── Step 2: tools/list ────────────────────────────────────────────────────
    print("\n=== Step 2: tools/list ===", file=sys.stderr)
    tl_resp = client.request("tools/list")
    tools_raw = tl_resp.get("result", {}).get("tools", [])

    note("TOOLS_COUNT", f"{len(tools_raw)} tools registered")

    # Inspect each tool for LLM-relevance
    tool_issues: list[str] = []
    for t in tools_raw:
        name = t.get("name", "?")
        desc = t.get("description", "")
        ann = t.get("annotations", {})
        ro = ann.get("readOnlyHint")
        dest = ann.get("destructiveHint")
        idem = ann.get("idempotentHint")

        issues = []
        if not desc:
            issues.append("MISSING description")
        elif len(desc) < 20:
            issues.append(f"SHORT description ({len(desc)} chars)")
        if ro is None:
            issues.append("MISSING readOnlyHint")
        if dest is None:
            issues.append("MISSING destructiveHint")

        status = "OK" if not issues else f"WARN: {', '.join(issues)}"
        note("TOOL", f"  {name:20s}  ro={ro}  dest={dest}  idem={idem}  → {status}")
        if issues:
            tool_issues.extend([f"{name}: {i}" for i in issues])

    if not tool_issues:
        note("TOOLS_QUALITY", "All tools have non-empty descriptions and ro/dest annotations. ✓")
    else:
        note("TOOLS_QUALITY", f"Issues found: {tool_issues}")

    # ── Step 3: status / repo_summary ────────────────────────────────────────
    print("\n=== Step 3: status + repo_summary ===", file=sys.stderr)
    status_resp = client.tool_call("status")
    status_data = result_content(status_resp)
    note("STATUS", f"branch={status_data.get('branch','?')} pending={status_data.get('pending_count','?')} index_state={status_data.get('index_state','?')}")

    rs_resp = client.tool_call("repo_summary")
    rs_data = result_content(rs_resp)
    note("REPO_SUMMARY", f"total_anchors={rs_data.get('total_anchors','?')} drift_score={rs_data.get('drift_score','?')} coverage_score={rs_data.get('coverage_score','?')} drift_coverage={rs_data.get('drift_coverage','?')}")

    # ── Step 4: Discovery — search for drift ─────────────────────────────────
    print("\n=== Step 4: search 'how does scry handle drift?' ===", file=sys.stderr)
    search_resp = client.tool_call("search", {"query": "how does scry handle drift", "top_k": 5})
    search_data = result_content(search_resp)

    if is_error(search_resp):
        note("SEARCH_ERR", error_msg(search_resp))
        search_results: list[dict[str, Any]] = []
    else:
        search_results = search_data if isinstance(search_data, list) else []

    tok_search = count_tokens_approx(search_data)
    note("SEARCH_TOKENS", f"5-result 'drift' search ≈ {tok_search} tokens (json bytes/4)")
    note("SEARCH_COUNT", f"returned {len(search_results)} results")

    # Inspect first result for compact fields
    if search_results:
        first = search_results[0]
        anchor_part = first.get("anchor", first)
        compact_dropped = [k for k in ("content_hash", "fingerprint_simhash", "def_line", "def_char", "closure_hash", "overview_embedding") if k in anchor_part]
        compact_kept = [k for k in ("transitive_hash_status",) if k in anchor_part]
        note("COMPACT_DROPPED", f"Compact-strip dropped from result[0]: {compact_dropped or 'none visible at this level'}")
        note("COMPACT_KEPT", f"transitive_hash_status present: {bool(compact_kept)} → value={anchor_part.get('transitive_hash_status','NOT PRESENT')}")
        note("RESULT_SHAPE", f"result[0] top-level keys: {list(first.keys())}")
        note("ANCHOR_SHAPE", f"result[0].anchor keys: {list(anchor_part.keys())}")

    # ── Step 5: Drill-down — get_anchor with BOTH anchor_id and legacy id ────
    print("\n=== Step 5: get_anchor (anchor_id + legacy id) ===", file=sys.stderr)
    first_id: str | None = None
    if search_results:
        anchor_part = search_results[0].get("anchor", search_results[0])
        first_id = anchor_part.get("id") or anchor_part.get("anchor_id")

    if first_id:
        # Test with anchor_id (preferred)
        ga_new = client.tool_call("get_anchor", {"anchor_id": first_id})
        ga_new_data = result_content(ga_new)
        if is_error(ga_new):
            note("GET_ANCHOR_NEW", f"FAIL with anchor_id=: {error_msg(ga_new)}")
        else:
            note("GET_ANCHOR_NEW", f"anchor_id= succeeded. id={ga_new_data.get('id','?') if isinstance(ga_new_data, dict) else '?'}")

        # Test with legacy id
        ga_legacy = client.tool_call("get_anchor", {"id": first_id})
        ga_legacy_data = result_content(ga_legacy)
        if is_error(ga_legacy):
            note("GET_ANCHOR_LEGACY", f"FAIL with legacy id=: {error_msg(ga_legacy)}")
        else:
            note("GET_ANCHOR_LEGACY", f"legacy id= succeeded. back-compat ✓")

        # Verify both return the same data
        if (not is_error(ga_new)) and (not is_error(ga_legacy)):
            same = ga_new_data == ga_legacy_data
            note("GET_ANCHOR_BACKCOMPAT", f"Both responses identical: {same}")

        # Check full content is present (not truncated)
        if isinstance(ga_new_data, dict):
            has_content = bool(ga_new_data.get("content_text"))
            note("GET_ANCHOR_CONTENT", f"content_text present (no truncation): {has_content}")
    else:
        note("GET_ANCHOR_SKIP", "No first_id available — skipping get_anchor test")

    # ── Step 6: get_links on the first anchor ─────────────────────────────────
    print("\n=== Step 6: get_links ===", file=sys.stderr)
    if first_id:
        gl_resp = client.tool_call("get_links", {"anchor_id": first_id, "direction": "both"})
        gl_data = result_content(gl_resp)
        if is_error(gl_resp):
            note("GET_LINKS_ERR", error_msg(gl_resp))
        else:
            links = gl_data.get("links", []) if isinstance(gl_data, dict) else []
            note("GET_LINKS_COUNT", f"anchor={first_id!r} → {len(links)} links, index_state={gl_data.get('index_state','?')}")
            if links:
                note("GET_LINKS_SCHEMA", f"link[0] keys: {list(links[0].keys())}")
                note("GET_LINKS_DRIFT", f"link[0] drift_status={links[0].get('drift_status','?')} semantic_drift={links[0].get('semantic_drift','?')}")
    else:
        note("GET_LINKS_SKIP", "Skipping get_links — no first_id")

    # ── Step 7: Cross-language search ─────────────────────────────────────────
    print("\n=== Step 7: search 'extract' (cross-language) ===", file=sys.stderr)
    xl_resp = client.tool_call("search", {"query": "extract", "top_k": 10})
    xl_data = result_content(xl_resp)
    xl_results: list[dict[str, Any]] = xl_data if isinstance(xl_data, list) else []

    py_results = []
    rs_results = []
    for r in xl_results:
        a = r.get("anchor", r)
        path = a.get("path", "")
        if path.endswith(".py"):
            py_results.append(a)
        elif path.endswith(".rs"):
            rs_results.append(a)

    note("CROSS_LANG", f"'extract' search: {len(xl_results)} total, {len(py_results)} .py, {len(rs_results)} .rs")
    if xl_results:
        paths = [r.get("anchor", r).get("path", "?") for r in xl_results[:5]]
        note("CROSS_LANG_PATHS", f"top-5 paths: {paths}")

    # ── Step 8: Drift workflow — propose_link + commit_links + find_drift ─────
    print("\n=== Step 8: drift workflow ===", file=sys.stderr)

    # We need two anchor IDs for propose_link. Use any two from search results.
    from_id: str | None = None
    to_id: str | None = None

    # Build a search to find a SECTION anchor (spec) and a CODE anchor
    sec_resp = client.tool_call("search", {"query": "drift detection", "types": ["section"], "top_k": 3})
    sec_data = result_content(sec_resp)
    sec_results: list[dict[str, Any]] = sec_data if isinstance(sec_data, list) else []

    code_resp = client.tool_call("search", {"query": "drift evaluation", "types": ["code"], "top_k": 3})
    code_data = result_content(code_resp)
    code_results: list[dict[str, Any]] = code_data if isinstance(code_data, list) else []

    if sec_results and code_results:
        from_id = sec_results[0].get("anchor", sec_results[0]).get("id")
        to_id = code_results[0].get("anchor", code_results[0]).get("id")

    idem_token = str(uuid.uuid4())

    if from_id and to_id:
        note("PROPOSE_IDS", f"from={from_id!r} to={to_id!r}")

        # First propose with idempotency token
        pl_resp1 = client.tool_call(
            "propose_link",
            {
                "from_id": from_id,
                "to_id": to_id,
                "link_type": "implements",
                "evidence": "UAT-R5-3 test link",
                "idempotency_token": idem_token,
            },
        )
        pl1_data = result_content(pl_resp1)
        if is_error(pl_resp1):
            note("PROPOSE_1_ERR", error_msg(pl_resp1))
        else:
            link_id_1 = pl1_data.get("link_id", "?") if isinstance(pl1_data, dict) else "?"
            note("PROPOSE_1_OK", f"link_id={link_id_1} status={pl1_data.get('status','?')}")

        # Second propose with SAME idempotency token (must be cache hit)
        pl_resp2 = client.tool_call(
            "propose_link",
            {
                "from_id": from_id,
                "to_id": to_id,
                "link_type": "implements",
                "evidence": "UAT-R5-3 test link",
                "idempotency_token": idem_token,
            },
        )
        pl2_data = result_content(pl_resp2)
        if is_error(pl_resp2):
            note("IDEM_SECOND_ERR", error_msg(pl_resp2))
        else:
            link_id_2 = pl2_data.get("link_id", "?") if isinstance(pl2_data, dict) else "?"
            same_link_id = (link_id_2 == link_id_1) if isinstance(pl1_data, dict) else False
            note("IDEM_CACHE_HIT", f"Both calls returned same link_id: {same_link_id} (link_id={link_id_2})")

        # Mutation safety: propose WITHOUT idempotency_token
        pl_no_idem = client.tool_call(
            "propose_link",
            {
                "from_id": from_id,
                "to_id": to_id,
                "link_type": "implements",
                "evidence": "UAT-R5-3 mutation safety test — no idempotency_token",
            },
        )
        if is_error(pl_no_idem):
            note("NO_IDEM_ERR", f"propose without token → ERROR: {error_msg(pl_no_idem)}")
        else:
            no_idem_data = result_content(pl_no_idem)
            note("NO_IDEM_OK", f"propose without token → succeeded (link_id={no_idem_data.get('link_id','?') if isinstance(no_idem_data, dict) else '?'}). No protocol-level refusal — risk implicit.")
            # Check if annotations on the tool hint at destructive nature
            propose_tool = next((t for t in tools_raw if t["name"] == "propose_link"), None)
            if propose_tool:
                ann = propose_tool.get("annotations", {})
                note("NO_IDEM_ANNOTATION", f"propose_link annotations: readOnly={ann.get('readOnlyHint')} destructive={ann.get('destructiveHint')} idempotent={ann.get('idempotentHint')}")

        # commit_links
        commit_idem = str(uuid.uuid4())
        cl_resp = client.tool_call(
            "commit_links",
            {"idempotency_token": commit_idem},
        )
        if is_error(cl_resp):
            note("COMMIT_ERR", error_msg(cl_resp))
        else:
            cl_data = result_content(cl_resp)
            promoted = cl_data.get("promoted", []) if isinstance(cl_data, dict) else []
            note("COMMIT_OK", f"promoted={len(promoted)} records, index_state={cl_data.get('index_state','?')}")

        # find_drift
        fd_resp = client.tool_call("find_drift", {})
        if is_error(fd_resp):
            note("FIND_DRIFT_ERR", error_msg(fd_resp))
        else:
            fd_data = result_content(fd_resp)
            entries = fd_data.get("entries", []) if isinstance(fd_data, dict) else []
            note("FIND_DRIFT_OK", f"{len(entries)} drift entries, coverage={fd_data.get('drift_coverage','?')}, index_state={fd_data.get('index_state','?')}")
            if entries:
                note("FIND_DRIFT_SCHEMA", f"entry[0] keys: {list(entries[0].keys())}")
    else:
        note("DRIFT_WORKFLOW_SKIP", f"Could not find anchor pair (from={from_id}, to={to_id}) — skipping drift workflow")

    # ── Step 9: get_callers (expect lsp_unavailable) ──────────────────────────
    print("\n=== Step 9: get_callers (expect LSP signal) ===", file=sys.stderr)
    code_anchor_id: str | None = None
    if code_results:
        code_anchor_id = code_results[0].get("anchor", code_results[0]).get("id")

    if code_anchor_id:
        gc_resp = client.tool_call("get_callers", {"anchor_id": code_anchor_id, "max_depth": 1})
        if is_error(gc_resp):
            err = error_msg(gc_resp)
            note("GET_CALLERS_ERR", f"error: {err}")
            # Check if error is about missing LSP position
            if "def_line" in err or "lsp" in err.lower() or "lsp_unavailable" in err.lower():
                note("GET_CALLERS_LSP_SIGNAL", "Error clearly identifies LSP/def_line constraint. ✓ Claude can recover.")
            else:
                note("GET_CALLERS_LSP_SIGNAL", "Error text does NOT mention LSP — unclear recovery path.")
        else:
            gc_data = result_content(gc_resp)
            callers = gc_data.get("callers", []) if isinstance(gc_data, dict) else []
            index_state = gc_data.get("index_state", "?") if isinstance(gc_data, dict) else "?"
            note("GET_CALLERS_OK", f"{len(callers)} callers returned, index_state={index_state}")
            if len(callers) == 0:
                note("GET_CALLERS_LSP_SIGNAL", "Empty callers list — LSP likely unavailable (no error, just empty). Ambiguous: is it 'no callers' or 'LSP missing'?")
    else:
        note("GET_CALLERS_SKIP", "No code anchor found — skipping get_callers")

    # ── Step 10: Token efficiency ──────────────────────────────────────────────
    print("\n=== Step 10: Token counting ===", file=sys.stderr)
    tok_5 = count_tokens_approx(xl_data)  # 10-result 'extract' search
    note("TOKEN_COUNT", f"10-result 'extract' search ≈ {tok_5} tokens. 5-result 'drift' search ≈ {tok_search} tokens.")

    # Estimate what it would be without compaction (rough: add back ~4 fields × 32 bytes each × results)
    BLOAT_PER_RESULT = (4 * 32) // 4  # tokens per result if hashes were present
    est_without_compact_5 = tok_search + (5 * BLOAT_PER_RESULT)
    est_without_compact_10 = tok_5 + (10 * BLOAT_PER_RESULT)
    note("TOKEN_ESTIMATE_SAVED", f"Estimated tokens saved by compaction: ~{5 * BLOAT_PER_RESULT} (5-result), ~{10 * BLOAT_PER_RESULT} (10-result)")

    # ── Final log dump ─────────────────────────────────────────────────────────
    print("\n=== Observations summary ===", file=sys.stderr)
    for o in obs:
        print(f"  {o}", file=sys.stderr)


if __name__ == "__main__":
    run()
    print("\n[UAT-R5-3] Harness complete.", file=sys.stderr)
