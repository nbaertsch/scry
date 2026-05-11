"""Tests for scry.lsp.reverse — get_callers / get_subclasses (W6e).

All tests mock ``LSPSession.request`` and ``session.supports`` via
``AsyncMock`` / ``MagicMock``; no real LSP subprocess is spawned.

Test strategy
-------------
* get_callers: capability check, prepareCallHierarchy outcomes, BFS walk,
  max_depth=2 transitive walk, defensive shape validation.
* get_subclasses: capability check, all three LSP response shapes (Location,
  Location[], LocationLink[]), defensive shape validation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from scry.lsp.manager import LSPSession
from scry.lsp.reverse import CallerRef, SubclassRef, get_callers, get_subclasses

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_session(
    *, supports_call_hierarchy: bool = True, supports_implementation: bool = True
) -> MagicMock:
    """Return a MagicMock LSPSession with configurable capability support."""
    session: MagicMock = MagicMock(spec=LSPSession)
    session.language = "python"

    def _supports(cap: str) -> bool:
        if cap == "callHierarchyProvider":
            return supports_call_hierarchy
        if cap == "implementationProvider":
            return supports_implementation
        return False

    session.supports.side_effect = _supports
    session.request = AsyncMock()
    return session


def _make_item(
    name: str,
    uri: str,
    *,
    start_line: int = 0,
    start_char: int = 0,
    end_line: int = 5,
    end_char: int = 0,
) -> dict[str, Any]:
    """Build a minimal LSP CallHierarchyItem."""
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


def _make_incoming_call(caller_item: dict[str, Any]) -> dict[str, Any]:
    """Wrap a caller item as an IncomingCall dict."""
    return {"from": caller_item, "fromRanges": []}


def _make_location(
    uri: str, *, sl: int = 0, sc: int = 0, el: int = 10, ec: int = 0
) -> dict[str, Any]:
    """Build a minimal LSP Location dict."""
    return {
        "uri": uri,
        "range": {
            "start": {"line": sl, "character": sc},
            "end": {"line": el, "character": ec},
        },
    }


def _make_location_link(
    target_uri: str, *, sl: int = 0, sc: int = 0, el: int = 10, ec: int = 0
) -> dict[str, Any]:
    """Build a minimal LSP LocationLink dict."""
    return {
        "targetUri": target_uri,
        "targetRange": {
            "start": {"line": sl, "character": sc},
            "end": {"line": el, "character": ec},
        },
        "targetSelectionRange": {
            "start": {"line": sl, "character": sc},
            "end": {"line": sl, "character": sc + 5},
        },
    }


# ─── get_callers: capability check ────────────────────────────────────────────


async def test_get_callers_unsupported_when_no_capability() -> None:
    """Server lacking callHierarchyProvider → returns empty tuple, no raise."""
    session = _make_session(supports_call_hierarchy=False)
    result = await get_callers(session, "file:///a.py", 0, 0)
    assert result == ()
    session.request.assert_not_awaited()


# ─── get_callers: prepareCallHierarchy outcomes ───────────────────────────────


async def test_get_callers_empty_when_prepare_returns_null() -> None:
    """prepareCallHierarchy returning null → empty (not an error)."""
    session = _make_session()
    session.request.return_value = None
    result = await get_callers(session, "file:///a.py", 0, 0)
    assert result == ()


async def test_get_callers_empty_when_prepare_returns_empty_list() -> None:
    """prepareCallHierarchy returning [] → empty."""
    session = _make_session()
    session.request.return_value = []
    result = await get_callers(session, "file:///a.py", 0, 0)
    assert result == ()


async def test_get_callers_empty_when_prepare_returns_non_list() -> None:
    """prepareCallHierarchy returning non-list → WARNING logged, empty returned."""
    session = _make_session()
    session.request.return_value = {"uri": "file:///a.py"}  # non-list
    result = await get_callers(session, "file:///a.py", 0, 0)
    assert result == ()


async def test_get_callers_empty_when_prepare_raises() -> None:
    """prepareCallHierarchy raising → WARNING logged, empty returned (never raises)."""
    session = _make_session()
    session.request.side_effect = RuntimeError("timeout")
    result = await get_callers(session, "file:///a.py", 0, 0)
    assert result == ()


# ─── get_callers: correct CallerRef extraction ────────────────────────────────


async def test_get_callers_single_direct_caller() -> None:
    """Single caller returned by incomingCalls → one CallerRef."""
    target_uri = "file:///target.py"
    caller_uri = "file:///caller.py"

    target_item = _make_item("target_fn", target_uri, start_line=10, start_char=4)
    caller_item = _make_item("caller_fn", caller_uri, start_line=20, start_char=0)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        if method == "callHierarchy/incomingCalls":
            return [_make_incoming_call(caller_item)]
        return None

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 10, 4)
    assert len(result) == 1
    ref = result[0]
    assert isinstance(ref, CallerRef)
    assert ref.uri == caller_uri
    assert ref.name == "caller_fn"
    assert ref.range_start_line == 20
    assert ref.range_start_char == 0


async def test_get_callers_multiple_callers_dedup() -> None:
    """Duplicate callers (same URI+name+position) are de-duplicated."""
    target_uri = "file:///target.py"
    caller_uri = "file:///caller.py"

    target_item = _make_item("target_fn", target_uri)
    caller_item = _make_item("caller_fn", caller_uri, start_line=5)
    duplicate_caller_item = _make_item("caller_fn", caller_uri, start_line=5)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        if method == "callHierarchy/incomingCalls":
            return [_make_incoming_call(caller_item), _make_incoming_call(duplicate_caller_item)]
        return None

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0)
    assert len(result) == 1


async def test_get_callers_max_depth_1_stops_at_direct() -> None:
    """max_depth=1 fetches incomingCalls for root only; does NOT recurse into callers."""
    target_uri = "file:///target.py"
    caller_uri = "file:///caller.py"
    transitive_uri = "file:///transitive.py"

    target_item = _make_item("target_fn", target_uri)
    caller_item = _make_item("caller_fn", caller_uri, start_line=1)
    transitive_item = _make_item("transitive_fn", transitive_uri, start_line=2)

    call_counts: dict[str, int] = {"incoming": 0}

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        if method == "callHierarchy/incomingCalls":
            call_counts["incoming"] += 1
            item = params.get("item", {})
            if item.get("name") == "target_fn":
                return [_make_incoming_call(caller_item)]
            return [_make_incoming_call(transitive_item)]

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0, max_depth=1)
    assert len(result) == 1
    assert result[0].name == "caller_fn"
    # Only one incomingCalls call (for the root), not for caller_fn.
    assert call_counts["incoming"] == 1


async def test_get_callers_max_depth_2_transitive() -> None:
    """max_depth=2 recurses into direct callers to find transitive callers."""
    target_uri = "file:///target.py"
    caller_uri = "file:///caller.py"
    transitive_uri = "file:///transitive.py"

    target_item = _make_item("target_fn", target_uri)
    caller_item = _make_item("caller_fn", caller_uri, start_line=1)
    transitive_item = _make_item("transitive_fn", transitive_uri, start_line=2)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        if method == "callHierarchy/incomingCalls":
            item = params.get("item", {})
            if item.get("name") == "target_fn":
                return [_make_incoming_call(caller_item)]
            if item.get("name") == "caller_fn":
                return [_make_incoming_call(transitive_item)]
            return []

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0, max_depth=2)
    names = {r.name for r in result}
    assert "caller_fn" in names
    assert "transitive_fn" in names


# ─── get_callers: defensive shape validation ──────────────────────────────────


async def test_get_callers_non_list_incoming_calls_response() -> None:
    """Non-list incomingCalls response → WARNING, continues without raising."""
    target_uri = "file:///target.py"
    target_item = _make_item("target_fn", target_uri)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        return "bad-response"  # not a list

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0)
    assert result == ()


async def test_get_callers_missing_from_field_in_incoming_call() -> None:
    """IncomingCall missing 'from' field → WARNING, skipped without raising."""
    target_uri = "file:///target.py"
    target_item = _make_item("target_fn", target_uri)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        # Missing 'from' key.
        return [{"fromRanges": []}]

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0)
    assert result == ()


async def test_get_callers_null_incoming_calls_is_not_error() -> None:
    """incomingCalls returning null → treated as no callers (leaf node)."""
    target_uri = "file:///target.py"
    target_item = _make_item("target_fn", target_uri)

    async def _request(method: str, params: dict[str, Any], *, timeout: float = 5.0) -> Any:
        if method == "textDocument/prepareCallHierarchy":
            return [target_item]
        return None  # null → no callers

    session = _make_session()
    session.request.side_effect = _request

    result = await get_callers(session, target_uri, 0, 0)
    assert result == ()


# ─── get_subclasses: capability check ─────────────────────────────────────────


async def test_get_subclasses_unsupported_when_no_capability() -> None:
    """Server lacking implementationProvider → empty tuple, no raise."""
    session = _make_session(supports_implementation=False)
    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert result == ()
    session.request.assert_not_awaited()


# ─── get_subclasses: LSP response shape handling ──────────────────────────────


async def test_get_subclasses_location_array_shape() -> None:
    """Location[] response (most common shape) → SubclassRefs created correctly."""
    session = _make_session()
    sub_uri = "file:///sub.py"
    session.request.return_value = [_make_location(sub_uri, sl=5, sc=0, el=30, ec=0)]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1
    ref = result[0]
    assert isinstance(ref, SubclassRef)
    assert ref.uri == sub_uri
    assert ref.range_start_line == 5
    assert ref.range_start_char == 0
    assert ref.range_end_line == 30


async def test_get_subclasses_single_location_shape() -> None:
    """Bare Location (not wrapped in array) → handled as single subclass."""
    session = _make_session()
    sub_uri = "file:///sub.py"
    session.request.return_value = _make_location(sub_uri, sl=10)

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1
    assert result[0].uri == sub_uri
    assert result[0].range_start_line == 10


async def test_get_subclasses_location_link_array_shape() -> None:
    """LocationLink[] response → targetUri / targetRange used for SubclassRef."""
    session = _make_session()
    sub_uri = "file:///impl.py"
    session.request.return_value = [_make_location_link(sub_uri, sl=7, sc=4, el=20, ec=0)]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1
    ref = result[0]
    assert ref.uri == sub_uri
    assert ref.range_start_line == 7
    assert ref.range_start_char == 4


async def test_get_subclasses_null_response_returns_empty() -> None:
    """null response → empty tuple (no implementations found)."""
    session = _make_session()
    session.request.return_value = None
    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert result == ()


async def test_get_subclasses_raises_returns_empty() -> None:
    """Exception during implementation request → WARNING, empty tuple returned."""
    session = _make_session()
    session.request.side_effect = RuntimeError("connection reset")
    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert result == ()


async def test_get_subclasses_dedup_by_uri_and_position() -> None:
    """Duplicate locations (same URI + start position) are de-duplicated."""
    session = _make_session()
    sub_uri = "file:///sub.py"
    loc = _make_location(sub_uri, sl=5)
    session.request.return_value = [loc, loc]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1


async def test_get_subclasses_multiple_subclasses() -> None:
    """Multiple distinct subclass locations → all returned."""
    session = _make_session()
    uri_a = "file:///sub_a.py"
    uri_b = "file:///sub_b.py"
    session.request.return_value = [
        _make_location(uri_a, sl=1),
        _make_location(uri_b, sl=2),
    ]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 2
    uris = {r.uri for r in result}
    assert uri_a in uris
    assert uri_b in uris


async def test_get_subclasses_malformed_entry_skipped() -> None:
    """Malformed entry in implementation response → WARNING, rest processed."""
    session = _make_session()
    sub_uri = "file:///valid.py"
    session.request.return_value = [
        {"bad": "shape"},  # malformed — missing uri/range
        _make_location(sub_uri, sl=3),
    ]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1
    assert result[0].uri == sub_uri


async def test_get_subclasses_unexpected_type_returns_empty() -> None:
    """Completely unexpected response type → WARNING, empty tuple returned."""
    session = _make_session()
    session.request.return_value = 42  # invalid type

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert result == ()


# ─── name population from URI ─────────────────────────────────────────────────


async def test_get_subclasses_name_from_uri_stem() -> None:
    """SubclassRef.name is populated from the URI stem when no other source."""
    session = _make_session()
    sub_uri = "file:///path/to/my_subclass.py"
    session.request.return_value = [_make_location(sub_uri)]

    result = await get_subclasses(session, "file:///base.py", 0, 0)
    assert len(result) == 1
    assert result[0].name == "my_subclass"


# uat-r5-5 pr-d noise
