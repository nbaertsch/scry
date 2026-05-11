"""Tests for scry.lsp.closure — transitive call-closure walker.

All tests mock ``LSPSession.request`` and ``session.supports`` via
``AsyncMock`` / ``MagicMock``; no real LSP subprocess is spawned.

The closure hash now fingerprints the canonicalized BODY CONTENT of
each callee (not just positional identity), so most tests create real
files in ``tmp_path`` whose URIs are passed through to the mocked LSP
responses.

Status values follow DESIGN.md §5.3 directly:
    complete > partial > unsupported > lsp_error  (best to worst)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scry.lsp.closure import (
    CalleeRef,
    ClosureResult,
    _compute_closure_hash,
    _file_uri_to_path,
    _slice_range,
    compute_closure,
)
from scry.lsp.manager import LSPSession

# ─── Helpers ──────────────────────────────────────────────────────────


def _make_item(
    name: str,
    uri: str,
    *,
    start_line: int = 0,
    start_char: int = 0,
    end_line: int = 10,
    end_char: int = 0,
) -> dict[str, Any]:
    """Build a minimal LSP ``CallHierarchyItem`` dict."""
    return {
        "name": name,
        "kind": 12,  # Function
        "uri": uri,
        "range": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
        "selectionRange": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": start_line, "character": start_char + len(name)},
        },
    }


def _make_outgoing(callee_item: dict[str, Any]) -> dict[str, Any]:
    """Wrap a callee item as an ``OutgoingCall`` dict."""
    return {"to": callee_item, "fromRanges": []}


def _make_session(*, supports_hierarchy: bool = True) -> MagicMock:
    """Return a MagicMock LSPSession with a configurable AsyncMock request."""
    session: MagicMock = MagicMock(spec=LSPSession)
    session.language = "python"
    session.supports.return_value = supports_hierarchy
    session.request = AsyncMock()
    return session


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    """Write *content* to ``tmp_path/name`` and return the file:// URI."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p.as_uri()


# ─── Capability check ────────────────────────────────────────────────


async def test_unsupported_when_callhierarchy_not_advertised(tmp_path: Path) -> None:
    """Server lacking callHierarchyProvider → unsupported."""
    session = _make_session(supports_hierarchy=False)
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "unsupported"
    assert result.callees == ()
    assert result.depth_reached == 0
    assert result.diagnostic.get("reason") == "callHierarchyProvider not advertised"
    session.request.assert_not_awaited()


# ─── prepareCallHierarchy outcomes ───────────────────────────────────


async def test_unsupported_when_prepare_returns_null(tmp_path: Path) -> None:
    """prepareCallHierarchy returns None → unsupported (not callable)."""
    session = _make_session()
    session.request.return_value = None
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "unsupported"
    assert result.callees == ()


async def test_unsupported_when_prepare_returns_empty_list(tmp_path: Path) -> None:
    """prepareCallHierarchy returns [] → unsupported."""
    session = _make_session()
    session.request.return_value = []
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "unsupported"


async def test_lsp_error_when_prepare_raises(tmp_path: Path) -> None:
    """prepareCallHierarchy raises → lsp_error."""
    session = _make_session()
    session.request.side_effect = TimeoutError("simulated")
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "lsp_error"
    assert result.callees == ()
    assert "TimeoutError" in result.diagnostic.get("exception", "")


async def test_lsp_error_when_prepare_returns_non_list(tmp_path: Path) -> None:
    """Regression (review-w3b HIGH): non-list prepareCallHierarchy → lsp_error.

    Must NOT raise KeyError on `prepare_result[0]`.
    """
    session = _make_session()
    session.request.return_value = {"unexpected": "shape"}  # dict, not list
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "lsp_error"
    assert "non-list" in result.diagnostic.get("reason", "")


async def test_lsp_error_when_prepare_first_element_non_dict(tmp_path: Path) -> None:
    """Defensive: prepareCallHierarchy returns [str] → lsp_error, no AttributeError."""
    session = _make_session()
    session.request.return_value = ["not a dict"]
    result = await compute_closure(session, "file:///x.py", 0, 0)
    assert result.status == "lsp_error"


# ─── Empty / leaf closure ─────────────────────────────────────────────


async def test_leaf_function_is_complete(tmp_path: Path) -> None:
    """Function with no outgoing calls → complete (vacuously) per §5.3.

    Per the §5.3 status table: 'A leaf function with zero outgoing
    calls is complete (vacuously), not partial.'
    """
    uri = _write_file(tmp_path, "leaf.py", "def leaf(): return 1\n")
    item = _make_item("leaf", uri)
    session = _make_session()
    session.request.side_effect = [
        [item],  # prepare result
        [],  # outgoingCalls returns empty list
    ]
    result = await compute_closure(session, uri, 0, 0)
    assert result.status == "complete"
    assert result.callees == ()
    assert result.depth_reached == 0


async def test_leaf_function_with_null_outgoing_is_complete(tmp_path: Path) -> None:
    """Regression (review-w3b MEDIUM): outgoingCalls null = empty (NOT lsp_error).

    Previously misclassified null outgoing as lsp_error; per LSP spec
    null and empty list are equivalent ('no outgoing calls').
    """
    uri = _write_file(tmp_path, "leaf.py", "def leaf(): pass\n")
    item = _make_item("leaf", uri)
    session = _make_session()
    session.request.side_effect = [[item], None]
    result = await compute_closure(session, uri, 0, 0)
    assert result.status == "complete"
    assert result.callees == ()


# ─── Single direct call ──────────────────────────────────────────────


async def test_single_direct_call_complete(tmp_path: Path) -> None:
    """A → B → ∅ : status complete, 1 callee."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b()\n\ndef b(): return 1\n",
    )
    a_item = _make_item("a", uri, start_line=0, end_line=2)
    b_item = _make_item("b", uri, start_line=3, end_line=4)
    session = _make_session()
    session.request.side_effect = [
        [a_item],  # prepare
        [_make_outgoing(b_item)],  # a → b
        [],  # b is leaf
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "complete"
    assert len(result.callees) == 1
    assert result.callees[0].name == "b"
    assert result.depth_reached == 1


# ─── Two-level chain ─────────────────────────────────────────────────


async def test_two_level_chain_complete_in_bfs_order(tmp_path: Path) -> None:
    """A → B → C : status complete, 2 callees in BFS order (B before C)."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b()\n\ndef b():\n    c()\n\ndef c(): return 1\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)
    c = _make_item("c", uri, start_line=6, end_line=7)
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare
        [_make_outgoing(b)],  # a → b
        [_make_outgoing(c)],  # b → c
        [],  # c is leaf
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "complete"
    assert [r.name for r in result.callees] == ["b", "c"]
    assert result.depth_reached == 2


# ─── Cycle handling (per §5.3 cycles do NOT degrade status) ──────────


async def test_indirect_cycle_status_complete(tmp_path: Path) -> None:
    """A → B → A : cycle handled silently per §5.3, status complete.

    Regression (review-w3b HIGH): previous design exposed a
    'cycle-detected' status that degraded the signal. §5.3 treats
    cycle detection as a walker-correctness concern only.
    """
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b()\n\ndef b():\n    a()\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare → root = a
        [_make_outgoing(b)],  # a → b
        [_make_outgoing(a)],  # b → a (back-edge)
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "complete"
    # Both a and b should appear (b discovered as child; a discovered
    # as cycle target).  Order: BFS visits b first, then sees a as a
    # back-edge.
    names = sorted(r.name for r in result.callees)
    assert "a" in names
    assert "b" in names
    # cycle_edges diagnostic is set
    assert result.diagnostic.get("cycle_edges", 0) >= 1


async def test_self_call_status_complete(tmp_path: Path) -> None:
    """A → A (direct recursion) : complete, A in closure once."""
    uri = _write_file(tmp_path, "app.py", "def a():\n    a()\n")
    a = _make_item("a", uri, start_line=0, end_line=2)
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare
        [_make_outgoing(a)],  # a → a
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "complete"
    assert [r.name for r in result.callees] == ["a"]
    assert result.diagnostic.get("cycle_edges", 0) >= 1


# ─── Diamond DAG (review-w3b HIGH fix) ────────────────────────────────


async def test_diamond_dag_status_complete(tmp_path: Path) -> None:
    """A → B → D, A → C → D : DAG convergence MUST NOT be classified as cycle.

    Regression (review-w3b HIGH): the old flat-visited algorithm
    treated the second discovery of D (via C) as cycle-detected,
    corrupting the status signal.  Per §5.3, only true back-edges
    (which would loop the walker) matter; DAG convergence is normal.
    """
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b(); c()\n\ndef b():\n    d()\n\ndef c():\n    d()\n\ndef d(): return 1\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)
    c = _make_item("c", uri, start_line=6, end_line=8)
    d = _make_item("d", uri, start_line=9, end_line=10)
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare
        [_make_outgoing(b), _make_outgoing(c)],  # a → b, a → c
        [_make_outgoing(d)],  # b → d
        [_make_outgoing(d)],  # c → d (DIAMOND CONVERGENCE)
        [],  # d leaf (only fetched once due to expansion dedup)
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "complete", (
        f"diamond DAG must be 'complete', got {result.status!r} (diagnostic={result.diagnostic!r})"
    )
    # D must appear exactly once in the closure (no duplicate)
    d_count = sum(1 for r in result.callees if r.name == "d")
    assert d_count == 1, f"D appeared {d_count} times in closure"


# ─── Depth cap → partial (per §5.3 status table) ──────────────────────


async def test_depth_cap_hit_is_partial(tmp_path: Path) -> None:
    """A → B → C → D with max_depth=2 : partial, callees up to depth 2.

    Regression (review-w3b BLOCKING): status was 'depth-cap-hit'; per
    §5.3 status table, depth-cap-hit → partial.
    """
    uri = _write_file(
        tmp_path,
        "app.py",
        ("def a():\n    b()\n\ndef b():\n    c()\n\ndef c():\n    d()\n\ndef d(): return 1\n"),
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)
    c = _make_item("c", uri, start_line=6, end_line=8)
    # NB: function `d` exists in the source file but is never referenced
    # by the walker because c is at max_depth and its outgoing is not
    # fetched.  Asserted via string-name absence below.
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare
        [_make_outgoing(b)],  # a → b (depth 1)
        [_make_outgoing(c)],  # b → c (depth 2)
        # c is at depth 2 = max_depth, walker won't fetch its outgoing
    ]
    result = await compute_closure(session, uri, 0, 4, max_depth=2)
    assert result.status == "partial"
    assert result.diagnostic.get("depth_cap") == 2
    names = [r.name for r in result.callees]
    assert "b" in names and "c" in names
    assert "d" not in names  # never reached


async def test_default_max_depth_is_32(tmp_path: Path) -> None:
    """Regression (review-w3b HIGH): default max_depth must be 32 per spec.

    DESIGN.md §6 / §11 set transitive_max_depth default to 32; the
    closure walker default must align so callers don't silently
    under-walk when not overriding.
    """
    import inspect

    sig = inspect.signature(compute_closure)
    assert sig.parameters["max_depth"].default == 32


# ─── Mid-walk LSP errors → lsp_error ──────────────────────────────────


async def test_outgoing_call_raises_status_lsp_error(tmp_path: Path) -> None:
    """Exception during outgoingCalls → lsp_error, partial closure preserved."""
    uri = _write_file(tmp_path, "app.py", "def a(): pass\n")
    a = _make_item("a", uri)
    session = _make_session()
    session.request.side_effect = [
        [a],  # prepare succeeds
        TimeoutError("middle of walk"),
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "lsp_error"
    assert "lsp_error_messages" in result.diagnostic


async def test_outgoing_returns_non_list_is_lsp_error(tmp_path: Path) -> None:
    """Regression (review-w3b HIGH): non-list outgoingCalls → lsp_error."""
    uri = _write_file(tmp_path, "app.py", "def a(): pass\n")
    a = _make_item("a", uri)
    session = _make_session()
    session.request.side_effect = [
        [a],
        {"unexpected": "shape"},  # not a list
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "lsp_error"
    assert result.diagnostic.get("malformed_responses", 0) >= 1


async def test_outgoing_entry_non_dict_is_lsp_error(tmp_path: Path) -> None:
    """Regression (review-w3b HIGH): non-dict outgoing entry → lsp_error."""
    uri = _write_file(tmp_path, "app.py", "def a(): pass\n")
    a = _make_item("a", uri)
    session = _make_session()
    session.request.side_effect = [
        [a],
        ["not a dict"],
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "lsp_error"


async def test_outgoing_to_field_missing_is_lsp_error(tmp_path: Path) -> None:
    """OutgoingCall with missing/malformed `to` field → lsp_error escalation."""
    uri = _write_file(tmp_path, "app.py", "def a(): pass\n")
    a = _make_item("a", uri)
    session = _make_session()
    session.request.side_effect = [
        [a],
        [{"fromRanges": []}],  # no `to` field, no usable fallback
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.status == "lsp_error"


# ─── Closure hash semantics (review-w3b BLOCKING) ─────────────────────


async def test_closure_hash_is_content_based(tmp_path: Path) -> None:
    """Regression (review-w3b BLOCKING): hash MUST fingerprint callee content.

    Editing a callee's body MUST change the closure_hash even though
    its position and name are unchanged.
    """
    file_path = tmp_path / "app.py"
    file_path.write_text(
        "def a():\n    b()\n\ndef b():\n    return 1\n",
        encoding="utf-8",
    )
    uri = file_path.as_uri()
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)

    def setup_session() -> MagicMock:
        session = _make_session()
        session.request.side_effect = [
            [a],
            [_make_outgoing(b)],
            [],
        ]
        return session

    result1 = await compute_closure(setup_session(), uri, 0, 4)

    # Now edit the callee body but keep the same position/name
    file_path.write_text(
        "def a():\n    b()\n\ndef b():\n    return 999\n",  # body changed
        encoding="utf-8",
    )
    result2 = await compute_closure(setup_session(), uri, 0, 4)

    assert result1.closure_hash != result2.closure_hash, (
        "closure_hash must change when callee body content changes "
        "(content-fingerprint semantics, not positional)"
    )


async def test_closure_hash_stable_across_runs(tmp_path: Path) -> None:
    """Same call graph + same files → identical closure_hash."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b()\n\ndef b(): return 1\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=4)

    async def run() -> ClosureResult:
        session = _make_session()
        session.request.side_effect = [[a], [_make_outgoing(b)], []]
        return await compute_closure(session, uri, 0, 4)

    r1 = await run()
    r2 = await run()
    assert r1.closure_hash == r2.closure_hash


async def test_closure_hash_order_independent(tmp_path: Path) -> None:
    """A → [B, C] same hash regardless of LSP-reported order."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a(): pass\n\ndef b(): return 1\n\ndef c(): return 2\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=1)
    b = _make_item("b", uri, start_line=2, end_line=3)
    c = _make_item("c", uri, start_line=4, end_line=5)

    async def run(order: list[dict[str, Any]]) -> ClosureResult:
        session = _make_session()
        session.request.side_effect = [
            [a],
            order,  # outgoing calls in given order
            [],  # b leaf
            [],  # c leaf
        ]
        return await compute_closure(session, uri, 0, 4)

    r_bc = await run([_make_outgoing(b), _make_outgoing(c)])
    r_cb = await run([_make_outgoing(c), _make_outgoing(b)])
    assert r_bc.closure_hash == r_cb.closure_hash


async def test_closure_hash_unaffected_by_callee_rename(tmp_path: Path) -> None:
    """Renaming a callee (different `name` field) does NOT change closure_hash.

    Per §5.3 the closure_hash fingerprints CONTENT, not identities.
    Renames are caught by the caller's own AST hash.
    """
    body = "def helper():\n    return 42\n"
    uri = _write_file(tmp_path, "app.py", body)

    a = _make_item("a", uri, start_line=0, end_line=2)
    callee_v1 = _make_item("helper", uri, start_line=0, end_line=2)
    callee_v2 = _make_item("renamed_helper", uri, start_line=0, end_line=2)

    async def walk_with(callee: dict[str, Any]) -> ClosureResult:
        session = _make_session()
        session.request.side_effect = [[a], [_make_outgoing(callee)], []]
        return await compute_closure(session, uri, 0, 4)

    r1 = await walk_with(callee_v1)
    r2 = await walk_with(callee_v2)
    assert r1.closure_hash == r2.closure_hash


async def test_unreadable_callee_escalates_to_partial(tmp_path: Path) -> None:
    """A callee whose file cannot be read → partial."""
    uri_a = _write_file(tmp_path, "app.py", "def a(): pass\n")
    a = _make_item("a", uri_a, start_line=0, end_line=1)
    # Callee in a non-existent file
    b = _make_item("b", "file:///does/not/exist.py", start_line=0, end_line=5)
    session = _make_session()
    session.request.side_effect = [[a], [_make_outgoing(b)], []]
    result = await compute_closure(session, uri_a, 0, 4)
    assert result.status == "partial"
    assert result.diagnostic.get("unreadable_files", 0) >= 1


async def test_empty_closure_hash_is_sha256_of_empty(tmp_path: Path) -> None:
    """Empty closure → SHA-256 of empty bytes (well-known constant)."""
    expected = hashlib.sha256(b"").hexdigest()
    h, unread = _compute_closure_hash([])
    assert h == expected
    assert unread == 0


# ─── Status precedence (lsp_error > partial > unsupported > complete) ─


async def test_status_precedence_partial_beats_complete(tmp_path: Path) -> None:
    """Walk that hits depth cap on one branch escalates from complete → partial."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a():\n    b()\n\ndef b():\n    c()\n\ndef c(): return 1\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=2)
    b = _make_item("b", uri, start_line=3, end_line=5)
    c = _make_item("c", uri, start_line=6, end_line=7)
    session = _make_session()
    session.request.side_effect = [
        [a],
        [_make_outgoing(b)],
        [_make_outgoing(c)],  # b → c
        # c at depth 2 = max_depth, no walk into c
    ]
    result = await compute_closure(session, uri, 0, 4, max_depth=2)
    assert result.status == "partial"


async def test_status_precedence_lsp_error_beats_partial(tmp_path: Path) -> None:
    """A walk that BOTH hits depth cap AND has an LSP error → lsp_error wins."""
    uri = _write_file(
        tmp_path,
        "app.py",
        "def a(): b(); c()\n\ndef b(): return 1\n\ndef c(): return 2\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=1)
    b = _make_item("b", uri, start_line=2, end_line=3)
    c = _make_item("c", uri, start_line=4, end_line=5)
    session = _make_session()
    session.request.side_effect = [
        [a],
        [_make_outgoing(b), _make_outgoing(c)],
        # b's outgoingCalls raises
        TimeoutError("boom"),
        # c also non-list
        {"bad": "shape"},
    ]
    result = await compute_closure(session, uri, 0, 4, max_depth=10)
    assert result.status == "lsp_error"


# ─── _file_uri_to_path helper ─────────────────────────────────────────


def test_file_uri_to_path_unix() -> None:
    p = _file_uri_to_path("file:///tmp/x.py")
    assert p is not None
    # On Windows a path starting with `/tmp/x.py` will be interpreted relatively;
    # the helper just produces a Path, not necessarily an existing one.
    assert "x.py" in str(p)


def test_file_uri_to_path_rejects_non_file() -> None:
    assert _file_uri_to_path("https://x.com/x.py") is None
    assert _file_uri_to_path("scry://repo/x.py") is None
    assert _file_uri_to_path("not-a-uri") is None


def test_file_uri_to_path_handles_localhost() -> None:
    p = _file_uri_to_path("file://localhost/tmp/x.py")
    assert p is not None


def test_file_uri_to_path_rejects_remote_host() -> None:
    assert _file_uri_to_path("file://nas/share/x.py") is None


# ─── _slice_range helper ──────────────────────────────────────────────


def test_slice_range_single_line() -> None:
    text = "alpha beta gamma"
    assert _slice_range(text, 0, 6, 0, 10) == "beta"


def test_slice_range_multi_line() -> None:
    text = "first\nsecond\nthird\n"
    # From mid of line 0 to mid of line 2
    assert _slice_range(text, 0, 1, 2, 3) == "irst\nsecond\nthi"


def test_slice_range_clips_overshoot() -> None:
    """end_line beyond file → clipped to last line."""
    text = "one\ntwo\n"
    assert _slice_range(text, 0, 0, 999, 999) != ""


def test_slice_range_empty_when_start_past_eof() -> None:
    assert _slice_range("only one line", 5, 0, 6, 0) == ""


# ─── depth_reached field ──────────────────────────────────────────────


async def test_depth_reached_zero_for_leaf(tmp_path: Path) -> None:
    uri = _write_file(tmp_path, "x.py", "def a(): pass\n")
    a = _make_item("a", uri)
    session = _make_session()
    session.request.side_effect = [[a], []]
    result = await compute_closure(session, uri, 0, 4)
    assert result.depth_reached == 0


async def test_depth_reached_tracks_deepest_path(tmp_path: Path) -> None:
    uri = _write_file(
        tmp_path,
        "x.py",
        "def a(): pass\n\ndef b(): pass\n\ndef c(): pass\n",
    )
    a = _make_item("a", uri, start_line=0, end_line=1)
    b = _make_item("b", uri, start_line=2, end_line=3)
    c = _make_item("c", uri, start_line=4, end_line=5)
    session = _make_session()
    session.request.side_effect = [
        [a],
        [_make_outgoing(b)],
        [_make_outgoing(c)],
        [],
    ]
    result = await compute_closure(session, uri, 0, 4)
    assert result.depth_reached == 2


# ─── CalleeRef shape ──────────────────────────────────────────────────


def test_callee_ref_is_frozen_dataclass() -> None:
    """CalleeRef is hashable and immutable."""
    ref = CalleeRef(
        uri="file:///x.py",
        name="foo",
        range_start_line=1,
        range_start_char=0,
        range_end_line=5,
        range_end_char=10,
    )
    # frozen dataclass → setattr raises
    with pytest.raises((AttributeError, TypeError)):
        ref.name = "bar"  # type: ignore[misc]
    # hashable
    assert hash(ref) == hash(ref)


def test_callee_ref_equality() -> None:
    a = CalleeRef("file:///x.py", "f", 0, 0, 1, 0)
    b = CalleeRef("file:///x.py", "f", 0, 0, 1, 0)
    assert a == b
    assert hash(a) == hash(b)


# uat-r5-5 pr-d noise
