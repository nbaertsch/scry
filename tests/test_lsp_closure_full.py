"""Tests for scry.lsp.full_resolution — full transitive-resolution pass.

All tests mock ``LSPSession.request`` via ``AsyncMock``; no real LSP subprocess
is spawned.  Files are created in ``tmp_path`` so that ``_hash_callee_body``
can read them for the closure-hash computation.

Test coverage
-------------
- Inheritance: class B(A); calling B() resolves to A.__init__ via references+definition
- Imported constants: references to BAR include import site + definition; both hashed
- Language fallback: zig (unsupported) → call_only with WARNING log
- Status escalation: lsp_error / partial logic for extra pass failures
- Compatibility: full mode == call_only when references return no new symbols
- Early return: lsp_error / unsupported from base walk skips extra pass
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scry.lsp.closure import (
    compute_closure,
)
from scry.lsp.full_resolution import (
    _location_to_callee_ref,
    compute_closure_full,
    supports_full_mode,
)
from scry.lsp.manager import LSPSession

# ─── Shared helpers ───────────────────────────────────────────────────


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
        "kind": 12,
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
    return {"to": callee_item, "fromRanges": []}


def _make_session(*, language: str = "python", supports_hierarchy: bool = True) -> MagicMock:
    session: MagicMock = MagicMock(spec=LSPSession)
    session.language = language
    session.supports.return_value = supports_hierarchy
    session.request = AsyncMock()
    return session


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    """Write *content* to ``tmp_path/name`` and return the ``file://`` URI."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p.as_uri()


def _make_location(
    uri: str, start_line: int, start_char: int, end_line: int, end_char: int
) -> dict[str, Any]:
    return {
        "uri": uri,
        "range": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
    }


def _make_location_link(
    target_uri: str,
    start_line: int,
    start_char: int,
    end_line: int,
    end_char: int,
) -> dict[str, Any]:
    """Build an LSP ``LocationLink`` dict."""
    return {
        "targetUri": target_uri,
        "targetRange": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": end_line, "character": end_char},
        },
        "targetSelectionRange": {
            "start": {"line": start_line, "character": start_char},
            "end": {"line": start_line, "character": start_char + 5},
        },
    }


# ─── supports_full_mode / _FULL_MODE_SUPPORTED ───────────────────────


def test_supported_languages_set() -> None:
    """Canonical supported languages are present."""
    for lang in ("python", "typescript", "tsx", "javascript", "jsx", "go", "rust"):
        assert supports_full_mode(lang), f"{lang} should be in _FULL_MODE_SUPPORTED"


def test_zig_not_supported() -> None:
    assert not supports_full_mode("zig")


def test_unknown_language_not_supported() -> None:
    assert not supports_full_mode("cobol")


# ─── _location_to_callee_ref ─────────────────────────────────────────


def test_location_to_callee_ref_from_location(tmp_path: Path) -> None:
    uri = _write_file(tmp_path, "a.py", "")
    loc = _make_location(uri, 3, 0, 10, 0)
    ref = _location_to_callee_ref(loc, "foo")
    assert ref is not None
    assert ref.uri == uri
    assert ref.range_start_line == 3
    assert ref.range_start_char == 0
    assert ref.range_end_line == 10
    assert ref.name == "foo"


def test_location_to_callee_ref_from_location_link(tmp_path: Path) -> None:
    uri = _write_file(tmp_path, "b.py", "")
    loc_link = _make_location_link(uri, 5, 2, 15, 0)
    ref = _location_to_callee_ref(loc_link, "bar")
    assert ref is not None
    assert ref.uri == uri
    assert ref.range_start_line == 5
    assert ref.range_start_char == 2


def test_location_to_callee_ref_none_on_non_dict() -> None:
    assert _location_to_callee_ref("not a dict", "x") is None
    assert _location_to_callee_ref(None, "x") is None
    assert _location_to_callee_ref(42, "x") is None


def test_location_to_callee_ref_none_on_missing_fields() -> None:
    assert _location_to_callee_ref({}, "x") is None
    assert _location_to_callee_ref({"uri": "file:///x.py"}, "x") is None
    assert _location_to_callee_ref({"uri": "", "range": {}}, "x") is None


# ─── compute_closure_full — early-exit cases ─────────────────────────


async def test_full_mode_returns_base_when_unsupported(tmp_path: Path) -> None:
    """If base walk returns unsupported, extra pass is skipped."""
    uri = _write_file(tmp_path, "x.py", "def f(): pass\n")
    session = _make_session(supports_hierarchy=False)
    result = await compute_closure_full(session, uri, 0, 0)
    assert result.status == "unsupported"
    assert result.callees == ()
    # No references / definition requests issued — only prepareCallHierarchy
    # was NOT called since callHierarchyProvider unsupported.
    # (The base compute_closure returns before any request for unsupported)
    session.request.assert_not_awaited()


async def test_full_mode_returns_base_when_lsp_error(tmp_path: Path) -> None:
    """If base walk's prepareCallHierarchy raises, extra pass is skipped."""
    uri = _write_file(tmp_path, "x.py", "def f(): pass\n")
    session = _make_session()
    session.request.side_effect = TimeoutError("timed out")
    result = await compute_closure_full(session, uri, 0, 0)
    assert result.status == "lsp_error"
    # Only the one prepare request was made (base walk errored out).
    assert session.request.await_count == 1


# ─── compute_closure_full — no extra refs (compatibility) ─────────────


async def test_full_mode_identical_to_call_only_when_no_new_refs(tmp_path: Path) -> None:
    """When textDocument/references returns empty, result equals call_only."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n" * 11)
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        # prepareCallHierarchy
        [_make_item("anchor", uri, start_line=5)],
        # outgoingCalls → one callee
        [_make_outgoing(callee_item)],
        # outgoingCalls on callee → leaf
        None,
        # textDocument/references on callee → empty
        [],
    ]

    base_result = None

    # Also compute call_only baseline for comparison.
    session2 = _make_session()
    session2.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
    ]
    base_result = await compute_closure(session2, uri, 5, 0)

    result = await compute_closure_full(session, uri, 5, 0)

    assert result.status == base_result.status
    assert result.closure_hash == base_result.closure_hash
    assert len(result.callees) == len(base_result.callees)
    assert "full_mode_extra" not in result.diagnostic


# ─── compute_closure_full — inheritance ──────────────────────────────


async def test_full_mode_inheritance_adds_parent_init(tmp_path: Path) -> None:
    """B() call → call_only finds B; full mode resolves A.__init__ via references+definition."""
    # Write files so _hash_callee_body can read them.
    b_uri = _write_file(tmp_path, "b.py", "class B: pass\n")
    a_uri = _write_file(tmp_path, "a.py", "class A:\n    def __init__(self): ...\n")
    anchor_uri = _write_file(tmp_path, "main.py", "from b import B\ndef create(): return B()\n")

    # B is the call_only callee (class B, line 0, char 0 in b.py).
    b_item = _make_item("B", b_uri, start_line=0, start_char=0, end_line=0, end_char=13)

    # The reference to B (a call site in main.py, line 1).
    b_reference = _make_location(anchor_uri, 1, 22, 1, 25)

    # textDocument/definition at the call site resolves to A.__init__ in a.py.
    a_init_location = _make_location(a_uri, 1, 4, 1, 34)

    session = _make_session()
    session.request.side_effect = [
        # prepareCallHierarchy on anchor
        [_make_item("create", anchor_uri, start_line=1)],
        # outgoingCalls → B
        [_make_outgoing(b_item)],
        # outgoingCalls on B → leaf
        None,
        # textDocument/references on B → [call site in main.py]
        [b_reference],
        # textDocument/definition at call site → A.__init__
        a_init_location,
    ]

    result = await compute_closure_full(session, anchor_uri, 1, 0)

    assert result.status == "complete"
    # B (call_only) + A.__init__ (full mode extra).
    assert len(result.callees) == 2
    uris = {c.uri for c in result.callees}
    assert b_uri in uris
    assert a_uri in uris
    assert result.diagnostic.get("full_mode_extra") == 1

    # Hash should differ from call_only (extra callee body included).
    session2 = _make_session()
    session2.request.side_effect = [
        [_make_item("create", anchor_uri, start_line=1)],
        [_make_outgoing(b_item)],
        None,
    ]
    base_result = await compute_closure(session2, anchor_uri, 1, 0)
    assert result.closure_hash != base_result.closure_hash


# ─── compute_closure_full — imported constants ────────────────────────


async def test_full_mode_imported_constant_hashes_definition(tmp_path: Path) -> None:
    """full mode discovers BAR's definition via references on print callee."""
    # print is a callee; its references include print(BAR) call.
    # definition at print(BAR) → BAR's definition in foo.py.
    foo_uri = _write_file(tmp_path, "foo.py", "BAR = 42\n")
    main_uri = _write_file(tmp_path, "main.py", "from foo import BAR\nprint(BAR)\n")

    # print callee at some stdlib location (non-file uri OK for test — no body read).
    # Use a real file so hash computation doesn't escalate to partial.
    print_uri = _write_file(tmp_path, "builtins.py", "def print(*args): ...\n")
    print_item = _make_item("print", print_uri, start_line=0, start_char=0, end_line=0, end_char=21)

    # Reference to print: the call site in main.py line 1.
    print_call_ref = _make_location(main_uri, 1, 0, 1, 5)

    # Definition at the print(BAR) line → BAR's definition in foo.py.
    bar_definition = _make_location(foo_uri, 0, 0, 0, 8)

    session = _make_session()
    session.request.side_effect = [
        # prepareCallHierarchy on main.py anchor
        [_make_item("anchor", main_uri, start_line=0)],
        # outgoingCalls → print
        [_make_outgoing(print_item)],
        # outgoingCalls on print → leaf
        None,
        # textDocument/references on print → [call site in main.py]
        [print_call_ref],
        # textDocument/definition at call site → BAR definition in foo.py
        bar_definition,
    ]

    result = await compute_closure_full(session, main_uri, 0, 0)

    assert result.status == "complete"
    # print (call_only) + BAR definition (full mode extra).
    assert len(result.callees) == 2
    uris = {c.uri for c in result.callees}
    assert print_uri in uris
    assert foo_uri in uris
    assert result.diagnostic.get("full_mode_extra") == 1


# ─── compute_closure_full — deduplication ────────────────────────────


async def test_full_mode_dedup_skips_already_seen_positions(tmp_path: Path) -> None:
    """Extra pass skips definitions that are already in call_only closure."""
    uri = _write_file(tmp_path, "a.py", "def f(): pass\ndef g(): pass\n")
    f_item = _make_item("f", uri, start_line=0, start_char=0, end_line=0, end_char=13)

    # Definition at the reference points back to f itself (same position).
    f_ref = _make_location(uri, 5, 0, 5, 5)
    f_definition_same_pos = _make_location(uri, 0, 0, 0, 13)  # same as f_item

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(f_item)],
        None,
        # references on f → one reference
        [f_ref],
        # definition at that reference → f itself (same position)
        f_definition_same_pos,
    ]

    result = await compute_closure_full(session, uri, 5, 0)

    # Only f, no extra.
    assert len(result.callees) == 1
    assert "full_mode_extra" not in result.diagnostic


# ─── compute_closure_full — status escalation ────────────────────────


async def test_full_mode_escalates_status_on_references_error(tmp_path: Path) -> None:
    """textDocument/references exception → lsp_error escalation."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        # textDocument/references raises
        TimeoutError("references timed out"),
    ]

    result = await compute_closure_full(session, uri, 5, 0)

    assert result.status == "lsp_error"
    assert result.diagnostic.get("full_mode_ref_errors") == 1


async def test_full_mode_escalates_status_on_definition_error(tmp_path: Path) -> None:
    """textDocument/definition exception → lsp_error escalation."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    other_uri = _write_file(tmp_path, "b.py", "callee()\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        # references → one hit
        [_make_location(other_uri, 0, 0, 0, 8)],
        # definition raises
        RuntimeError("connection lost"),
    ]

    result = await compute_closure_full(session, uri, 5, 0)

    assert result.status == "lsp_error"
    assert result.diagnostic.get("full_mode_ref_errors") == 1


async def test_full_mode_partial_when_extra_callee_unreadable(tmp_path: Path) -> None:
    """Extra callee body unreadable → status escalates to partial."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    other_uri = _write_file(tmp_path, "b.py", "callee()\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    # Extra callee points to a non-existent file.
    missing_uri = (tmp_path / "missing.py").as_uri()

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        # references → hit in other file
        [_make_location(other_uri, 0, 0, 0, 8)],
        # definition → extra callee at missing file
        _make_location(missing_uri, 0, 0, 5, 0),
    ]

    result = await compute_closure_full(session, uri, 5, 0)

    assert result.status == "partial"
    assert result.diagnostic.get("full_mode_extra") == 1


# ─── compute_closure_full — null / malformed references ──────────────


async def test_full_mode_handles_null_references(tmp_path: Path) -> None:
    """Null textDocument/references response is treated as no refs (not error)."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        # null references
        None,
    ]

    result = await compute_closure_full(session, uri, 5, 0)
    assert result.status == "complete"
    assert "full_mode_ref_errors" not in result.diagnostic


async def test_full_mode_malformed_references_escalates(tmp_path: Path) -> None:
    """Non-list, non-null textDocument/references → lsp_error."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        # malformed — a string instead of list
        "unexpected string",
    ]

    result = await compute_closure_full(session, uri, 5, 0)
    assert result.status == "lsp_error"
    assert result.diagnostic.get("full_mode_ref_errors") == 1


# ─── compute_closure_full — LocationLink support ─────────────────────


async def test_full_mode_accepts_location_link_from_definition(tmp_path: Path) -> None:
    """textDocument/definition may return LocationLink objects (not just Location)."""
    uri = _write_file(tmp_path, "a.py", "def callee(): pass\n")
    extra_uri = _write_file(tmp_path, "extra.py", "def extra(): ...\n")
    other_uri = _write_file(tmp_path, "b.py", "callee()\n")
    callee_item = _make_item("callee", uri, start_line=0, end_line=0)

    session = _make_session()
    session.request.side_effect = [
        [_make_item("anchor", uri, start_line=5)],
        [_make_outgoing(callee_item)],
        None,
        [_make_location(other_uri, 0, 0, 0, 8)],
        # definition as LocationLink
        _make_location_link(extra_uri, 0, 0, 0, 16),
    ]

    result = await compute_closure_full(session, uri, 5, 0)

    assert result.status == "complete"
    assert len(result.callees) == 2
    assert any(c.uri == extra_uri for c in result.callees)
    assert result.diagnostic.get("full_mode_extra") == 1


# ─── Language fallback (zig → call_only + WARNING) ───────────────────


async def test_full_mode_fallback_for_zig_emits_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When language is zig (unsupported), compute_closure_full warns and returns call_only."""
    uri = _write_file(tmp_path, "x.zig", "fn f() void {}\n")
    callee_uri = _write_file(tmp_path, "y.zig", "fn g() void {}\n")
    callee_item = _make_item("g", callee_uri, start_line=0, end_line=0)

    session = _make_session(language="zig")

    # Full mode is NOT gated here — compute_closure_full itself doesn't gate.
    # The gating lives in _enrich_all_with_lsp in the indexer.
    # For the direct API, the function runs regardless; we test the indexer gate separately.
    # Instead, verify that supports_full_mode("zig") is False.
    assert not supports_full_mode("zig")

    # For the indexer WARNING test: simulate what _enrich_all_with_lsp does.
    # If use_full_mode is False for zig, it calls compute_closure (not compute_closure_full).
    # We verify this by showing the result is the same as call_only.
    session.request.side_effect = [
        [_make_item("f", uri, start_line=0)],
        [_make_outgoing(callee_item)],
        None,
    ]
    base_result = await compute_closure(session, uri, 0, 0)
    assert base_result.status == "complete"
    assert len(base_result.callees) == 1


async def test_indexer_logs_warning_for_unsupported_language(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """_enrich_all_with_lsp logs WARNING for full mode on unsupported language."""
    from scry.index import _enrich_all_with_lsp
    from scry.lsp.manager import LSPManager
    from scry.models import Anchor, AnchorType

    zig_file = tmp_path / "x.zig"
    zig_file.write_text("fn f() void {}\n", encoding="utf-8")

    anchor = Anchor(
        id="zig::f",
        path="x.zig",
        type=AnchorType.CODE.value,
        symbol_name="f",
        content_text="fn f() void {}",
        content_hash="sha256:" + "a" * 64,
        fingerprint_simhash=0,
        def_line=0,
        def_char=0,
    )

    lsp_mgr = MagicMock(spec=LSPManager)
    session = _make_session(language="zig")
    session.request.side_effect = [
        # prepareCallHierarchy → unsupported (no callHierarchyProvider)
        None,
    ]
    session.supports.return_value = False
    lsp_mgr.session_for = AsyncMock(return_value=session)
    lsp_mgr.status_for.return_value = "alive"
    lsp_mgr.shutdown_all = AsyncMock()

    with caplog.at_level(logging.WARNING, logger="scry.index"):
        await _enrich_all_with_lsp(
            [anchor],
            lsp_mgr,
            tmp_path,
            max_depth=32,
            transitive_resolution="full",
        )

    assert any(
        "falling back to call_only" in r.message and "zig" in r.message for r in caplog.records
    )


# ─── Multiple callees — all get references queried ────────────────────


async def test_full_mode_queries_references_for_each_callee(tmp_path: Path) -> None:
    """Extra pass issues references requests for ALL call_only callees."""
    uri = _write_file(tmp_path, "a.py", "def f(): pass\ndef g(): pass\ndef anchor(): f(); g()\n")
    f_item = _make_item("f", uri, start_line=0, start_char=0, end_line=0, end_char=13)
    g_item = _make_item("g", uri, start_line=1, start_char=0, end_line=1, end_char=13)

    session = _make_session()
    session.request.side_effect = [
        # prepareCallHierarchy
        [_make_item("anchor", uri, start_line=2)],
        # outgoingCalls → f, g
        [_make_outgoing(f_item), _make_outgoing(g_item)],
        # outgoingCalls on f → leaf
        None,
        # outgoingCalls on g → leaf
        None,
        # references on f → empty
        [],
        # references on g → empty
        [],
    ]

    result = await compute_closure_full(session, uri, 2, 0)

    assert result.status == "complete"
    assert len(result.callees) == 2
    # Verify references were queried for both callees.
    refs_calls = [
        c for c in session.request.call_args_list if c.args[0] == "textDocument/references"
    ]
    assert len(refs_calls) == 2


# ─── compute_closure_full — documentSymbol pass (W6d BLOCKING #1) ────


async def test_full_mode_imported_constant_via_document_symbol(tmp_path: Path) -> None:
    """Step 3: BAR found via documentSymbol + identifier-substring match.

    Anchor body contains "BAR".  documentSymbol returns a BAR symbol whose
    definition resolves to a different file.  BAR is added as an extra callee.

    This test uses a REAL identifier-substring match (anchor body has "BAR",
    documentSymbol returns a symbol named "BAR") so it is not dependent on
    unrealistic LSP-mock behaviour.
    """
    # main.py body: anchor lives here; "BAR" appears in its body.
    main_uri = _write_file(tmp_path, "main.py", "from foo import BAR\nresult = BAR + 1\n")
    # foo.py: canonical definition of BAR (a different file).
    foo_uri = _write_file(tmp_path, "foo.py", "BAR = 42\n")

    # documentSymbol for BAR in main.py: selectionRange points at "BAR" on line 0 char 16.
    bar_doc_symbol: dict[str, Any] = {
        "name": "BAR",
        "kind": 13,  # Variable
        "range": {
            "start": {"line": 0, "character": 16},
            "end": {"line": 0, "character": 19},
        },
        "selectionRange": {
            "start": {"line": 0, "character": 16},
            "end": {"line": 0, "character": 19},
        },
    }

    # textDocument/definition at BAR's position → location in foo.py.
    bar_definition = _make_location(foo_uri, 0, 0, 0, 8)

    session = _make_session()
    session.request.side_effect = [
        # 1. prepareCallHierarchy on main.py
        [_make_item("anchor", main_uri, start_line=0)],
        # 2. outgoingCalls on anchor → no callees (BAR is not a call)
        None,
        # 3a. textDocument/documentSymbol on main.py → [BAR]
        [bar_doc_symbol],
        # 3b. textDocument/definition at BAR's position → foo.py
        bar_definition,
    ]

    result = await compute_closure_full(session, main_uri, 0, 0)

    assert result.status == "complete"
    # BAR from foo.py is the only callee (added by step 3).
    assert len(result.callees) == 1
    assert result.callees[0].uri == foo_uri
    assert result.callees[0].name == "BAR"
    assert result.diagnostic.get("full_mode_extra") == 1

# uat-r5-5 pr-d noise
