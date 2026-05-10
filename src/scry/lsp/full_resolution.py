"""Full transitive-closure pass (DESIGN.md §5.3 opt-in mode).

Setting ``code_anchors.transitive_resolution: full`` enables this module's
:func:`compute_closure_full` on top of the standard ``call_only`` walk in
:mod:`scry.lsp.closure`.

Algorithm (three passes beyond the base call_only walk)
---------------------------------------------------------
1. Run :func:`~scry.lsp.closure.compute_closure` (the call_only walk) to
   obtain the base callee set.

2. **References pass — inheritance chains** (one level deep for each call_only callee):

   a. Issue ``textDocument/references`` at *C*'s definition-start position
      (``C.uri``, ``C.range_start_line``, ``C.range_start_char``) with
      ``includeDeclaration: false``.  This returns every location that
      references (calls, imports, subclasses) the symbol.
   b. For each reference *R* returned:

      - Issue ``textDocument/definition`` at *R*'s start position.
      - Each returned ``Location`` / ``LocationLink`` is converted to an
        extra :class:`~scry.lsp.closure.CalleeRef`.
      - Duplicates are suppressed via a ``(uri, start_line, start_char)``
        position-key set shared with the base callees.

3. **documentSymbol pass — imported constants / module-level identifiers**:

   a. Issue ``textDocument/documentSymbol`` on the anchor's *own* source file
      (``file_uri``).  This returns the module's top-level symbol table.
   b. For each returned ``DocumentSymbol`` *S*:

      - Skip if ``len(S.name) < 2`` (too short, high noise).
      - Skip if ``S.name`` is already in the call_only callee set (dedup by name).
      - Skip if ``S.name`` does not appear as a substring in the anchor's body
        content (read via :func:`_read_anchor_body`).
      - Issue ``textDocument/definition`` at ``S.selectionRange.start``
        (falling back to ``S.range.start``) to resolve the canonical definition.
      - If the definition resolves to a **different file** than ``file_uri``,
        add it as an extra :class:`~scry.lsp.closure.CalleeRef`.  This
        captures ``from foo import BAR`` patterns where ``BAR`` is used in the
        anchor body but does not appear in callHierarchy.

4. Merge all extra refs into the combined callee list, recompute the closure
   hash, and return a :class:`~scry.lsp.closure.ClosureResult` whose
   ``diagnostic`` includes ``full_mode_extra`` and optionally
   ``full_mode_ref_errors``.

How this catches inheritance and imported constants
----------------------------------------------------
**Inheritance** (step 2): When class ``B(A)`` is called as ``B()``,
callHierarchy may only resolve to ``B`` (the class) if the LSP doesn't
propagate through ``__init__`` inheritance.  The extra pass issues
``textDocument/references(B.uri, B.start)`` → returns all call sites of
``B``, then ``textDocument/definition`` at each call site resolves the
*actual* constructor, which a type-aware LSP will resolve to ``A.__init__``.
That definition is added to the closure.

**Imported constants / module-level identifiers** (step 3): A constant
``BAR`` imported via ``from foo import BAR`` and used in the anchor body
(e.g. ``result = BAR + 1``) doesn't appear as an outgoing *call* in
callHierarchy.  Step 3 discovers ``BAR`` via ``textDocument/documentSymbol``
on the anchor's file, confirms ``"BAR"`` appears in the anchor's body text,
then issues ``textDocument/definition`` at ``BAR``'s position.  If the
definition is in a different file (``foo.py``), it is added to the closure.

Status handling
---------------
The extra passes use the same :class:`~scry.lsp.closure.ClosureStatus`
escalation rules:

* A request exception or malformed response escalates to ``lsp_error``
  (only in step 2; step 3 documentSymbol failures are debug-logged and
  silently skipped so a missing documentSymbol capability doesn't pollute
  the status).
* Unreadable callee bodies escalate to ``partial``.
* The base status is *never* upgraded (call_only result governs the floor).

Per-language gating
-------------------
Not all LSPs implement ``textDocument/references`` reliably.  Languages
listed in :data:`_FULL_MODE_SUPPORTED` get the extra pass; others fall
back to ``call_only`` with a one-time WARNING log.  Use
:func:`supports_full_mode` to query the gate programmatically.

``extra_callees`` disposition
------------------------------
Extra callees are **merged into the main** ``callees`` tuple in the
returned :class:`~scry.lsp.closure.ClosureResult`.  No separate field is
added to ``ClosureResult`` — the ``diagnostic["full_mode_extra"]`` key
documents how many new refs the pass contributed.  This keeps the public
API stable and avoids a parallel field that callers would need to union
themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from scry.lsp.closure import (
    CalleeRef,
    ClosureResult,
    ClosureStatus,
    _compute_closure_hash,
    _escalate,
    compute_closure,
)
from scry.lsp.manager import LSPSession

logger = logging.getLogger(__name__)

__all__ = ["compute_closure_full", "supports_full_mode"]

# Languages with sufficiently reliable ``textDocument/references`` support.
# ZLS (Zig) doesn't expose references consistently → falls back to call_only.
_FULL_MODE_SUPPORTED: frozenset[str] = frozenset(
    {"python", "typescript", "tsx", "javascript", "jsx", "go", "rust"}
)


def supports_full_mode(language: str) -> bool:
    """Return ``True`` if *language* supports the full transitive-resolution pass.

    Languages not in :data:`_FULL_MODE_SUPPORTED` (e.g. ``"zig"``) will be
    warned and automatically fall back to ``call_only`` when
    :func:`compute_closure_full` is called for them.
    """
    return language in _FULL_MODE_SUPPORTED


def _read_anchor_body(file_uri: str, start_line: int) -> str:
    """Return the body of *file_uri* starting at *start_line*.

    Used to substring-match ``DocumentSymbol`` names against the anchor's
    body without a real LSP hover/range request.  Returns ``""`` on any IO
    or parse error — safe to use in ``in`` checks.
    """
    try:
        parsed = urlparse(file_uri)
        raw_path = unquote(parsed.path)
        # On Windows file:///C:/foo/bar → raw_path is /C:/foo/bar.
        # Strip the leading slash when followed by a drive-letter colon.
        if len(raw_path) > 2 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[start_line:])
    except Exception:
        return ""


def _location_to_callee_ref(location: Any, hint_name: str) -> CalleeRef | None:
    """Convert an LSP ``Location`` or ``LocationLink`` dict to a :class:`CalleeRef`.

    ``hint_name`` is propagated as the ``name`` field (for diagnostics only;
    it is the name of the originating callee whose reference chain produced
    this definition).  Returns ``None`` for malformed inputs.
    """
    if not isinstance(location, dict):
        return None
    # LocationLink uses targetUri / targetRange; Location uses uri / range.
    uri: Any = location.get("targetUri") or location.get("uri")
    rng: Any = location.get("targetRange") or location.get("range")
    if not isinstance(uri, str) or not uri:
        return None
    if not isinstance(rng, dict):
        return None
    start: Any = rng.get("start")
    end: Any = rng.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    s_line: Any = start.get("line")
    s_char: Any = start.get("character")
    e_line: Any = end.get("line")
    e_char: Any = end.get("character")
    if not all(isinstance(v, int) for v in (s_line, s_char, e_line, e_char)):
        return None
    return CalleeRef(
        uri=uri,
        name=hint_name,
        range_start_line=s_line,
        range_start_char=s_char,
        range_end_line=e_line,
        range_end_char=e_char,
    )


async def compute_closure_full(
    session: LSPSession,
    file_uri: str,
    line: int,
    character: int,
    *,
    max_depth: int = 32,
    timeout_per_call: float = 5.0,
) -> ClosureResult:
    """Compute the transitive closure using both callHierarchy and references.

    Runs :func:`~scry.lsp.closure.compute_closure` (call_only) first, then
    adds an extra pass over ``textDocument/references`` +
    ``textDocument/definition`` to capture inheritance chains and imported
    constants that callHierarchy alone would miss.

    For languages where ``textDocument/references`` is unreliable, call
    :func:`supports_full_mode` before invoking this function and fall back to
    :func:`~scry.lsp.closure.compute_closure` with a warning.

    Parameters
    ----------
    session:
        A live, initialized :class:`~scry.lsp.manager.LSPSession`.
    file_uri:
        Absolute ``file://`` URI of the document containing the anchor.
    line:
        0-based line index of the anchor's definition position.
    character:
        0-based character offset within ``line``.
    max_depth:
        Forwarded to the underlying call_only walk.
        Default 32 per ``code_anchors.transitive_max_depth``.
    timeout_per_call:
        Per-LSP-request timeout in seconds.  Applied to every request in
        both the call_only walk and the extra references pass.

    Returns
    -------
    ClosureResult
        Always returns a result — never raises.  The ``callees`` tuple
        contains both call_only and extra callees merged and deduplicated.
        ``diagnostic["full_mode_extra"]`` reports how many new symbols the
        extra pass contributed (absent when zero).

    Notes
    -----
    The extra pass runs **one level deep**: references are queried for each
    call_only callee, and definitions are queried for each reference.  The
    resulting definitions are added to the closure but are **not** recursively
    expanded via outgoingCalls.  This is intentional — the extra pass is
    narrow by design (inheritance parent, import source), not a second BFS.

    When the base walk returns ``lsp_error`` or ``unsupported``, the extra
    pass is skipped entirely and the base result is returned as-is.
    """
    # ── 1. call_only base walk ─────────────────────────────────────────
    base = await compute_closure(
        session,
        file_uri,
        line,
        character,
        max_depth=max_depth,
        timeout_per_call=timeout_per_call,
    )

    # For lsp_error / unsupported the extra pass cannot add signal.
    if base.status in ("lsp_error", "unsupported"):
        return base

    # ── 2. Extra references pass ───────────────────────────────────────
    extra_callees: list[CalleeRef] = []
    status: ClosureStatus = base.status
    diagnostic: dict[str, Any] = dict(base.diagnostic)

    # Deduplication across call_only and extra callees keyed by
    # (uri, start_line, start_char) — name is irrelevant for dedup.
    seen_positions: set[tuple[str, int, int]] = {
        (c.uri, c.range_start_line, c.range_start_char) for c in base.callees
    }

    extra_count: int = 0
    ref_errors: int = 0

    for callee in base.callees:
        # 2a. All reference locations for this callee's definition position.
        try:
            refs: Any = await session.request(
                "textDocument/references",
                {
                    "textDocument": {"uri": callee.uri},
                    "position": {
                        "line": callee.range_start_line,
                        "character": callee.range_start_char,
                    },
                    "context": {"includeDeclaration": False},
                },
                timeout=timeout_per_call,
            )
        except Exception as exc:
            logger.debug(
                "LSP [%s] textDocument/references failed for '%s' at %s:%d:%d — %s",
                session.language,
                callee.name,
                callee.uri,
                callee.range_start_line,
                callee.range_start_char,
                exc,
            )
            status = _escalate(status, "lsp_error")
            ref_errors += 1
            continue

        if refs is None:
            # Null is a valid "no references" response per LSP spec.
            continue
        if not isinstance(refs, list):
            logger.debug(
                "LSP [%s] textDocument/references returned %s for '%s'",
                session.language,
                type(refs).__name__,
                callee.name,
            )
            status = _escalate(status, "lsp_error")
            ref_errors += 1
            continue

        # 2b. For each reference, resolve the actual definition.
        for ref_loc in refs:
            if not isinstance(ref_loc, dict):
                continue
            ref_uri: Any = ref_loc.get("uri")
            ref_range: Any = ref_loc.get("range")
            if not isinstance(ref_uri, str) or not isinstance(ref_range, dict):
                continue
            ref_start: Any = ref_range.get("start")
            if not isinstance(ref_start, dict):
                continue
            ref_line: Any = ref_start.get("line")
            ref_char: Any = ref_start.get("character")
            if not isinstance(ref_line, int) or not isinstance(ref_char, int):
                continue

            try:
                defs: Any = await session.request(
                    "textDocument/definition",
                    {
                        "textDocument": {"uri": ref_uri},
                        "position": {"line": ref_line, "character": ref_char},
                    },
                    timeout=timeout_per_call,
                )
            except Exception as exc:
                logger.debug(
                    "LSP [%s] textDocument/definition failed at %s:%d:%d — %s",
                    session.language,
                    ref_uri,
                    ref_line,
                    ref_char,
                    exc,
                )
                status = _escalate(status, "lsp_error")
                ref_errors += 1
                continue

            if defs is None:
                continue

            # defs may be a single Location, list[Location], or list[LocationLink].
            def_list: list[Any]
            if isinstance(defs, dict):
                def_list = [defs]
            elif isinstance(defs, list):
                def_list = defs
            else:
                continue

            for def_item in def_list:
                extra_ref = _location_to_callee_ref(def_item, callee.name)
                if extra_ref is None:
                    continue
                pos_key = (
                    extra_ref.uri,
                    extra_ref.range_start_line,
                    extra_ref.range_start_char,
                )
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)
                extra_callees.append(extra_ref)
                extra_count += 1

    if extra_count > 0:
        diagnostic["full_mode_extra"] = extra_count
    if ref_errors > 0:
        diagnostic["full_mode_ref_errors"] = ref_errors

    # ── 3. documentSymbol pass: imported constants / module-level identifiers ─
    # Issue textDocument/documentSymbol on the anchor's own source file and
    # cross-reference each symbol name against the anchor's body text.
    # Symbols whose definitions resolve to a *different* file are imported
    # identifiers (e.g. ``from foo import BAR``) that callHierarchy misses.
    doc_symbols: list[Any] | None = None
    try:
        raw_symbols: Any = await session.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": file_uri}},
            timeout=timeout_per_call,
        )
        if isinstance(raw_symbols, list):
            doc_symbols = raw_symbols
    except Exception as exc:
        # documentSymbol is optional: many LSPs don't implement it or return
        # errors. Treat all failures as debug noise rather than status errors.
        logger.debug(
            "LSP [%s] textDocument/documentSymbol failed for %s — %s",
            session.language,
            file_uri,
            exc,
        )

    if doc_symbols is not None:
        anchor_body = _read_anchor_body(file_uri, line)
        seen_callee_names: set[str] = {c.name for c in base.callees}

        for sym in doc_symbols:
            if not isinstance(sym, dict):
                continue
            sym_name: Any = sym.get("name")
            if not isinstance(sym_name, str) or len(sym_name) < 2:
                continue
            if sym_name in seen_callee_names:
                continue
            if sym_name not in anchor_body:
                continue

            # Use selectionRange (tight around the identifier) or fall back to range.
            sym_range: Any = sym.get("selectionRange") or sym.get("range")
            if not isinstance(sym_range, dict):
                continue
            sym_start: Any = sym_range.get("start")
            if not isinstance(sym_start, dict):
                continue
            sym_line: Any = sym_start.get("line")
            sym_char: Any = sym_start.get("character")
            if not isinstance(sym_line, int) or not isinstance(sym_char, int):
                continue

            try:
                sym_defs: Any = await session.request(
                    "textDocument/definition",
                    {
                        "textDocument": {"uri": file_uri},
                        "position": {"line": sym_line, "character": sym_char},
                    },
                    timeout=timeout_per_call,
                )
            except Exception as exc:
                logger.debug(
                    "LSP [%s] textDocument/definition failed for symbol '%s' — %s",
                    session.language,
                    sym_name,
                    exc,
                )
                status = _escalate(status, "lsp_error")
                ref_errors += 1
                continue

            if sym_defs is None:
                continue

            sym_def_list: list[Any]
            if isinstance(sym_defs, dict):
                sym_def_list = [sym_defs]
            elif isinstance(sym_defs, list):
                sym_def_list = sym_defs
            else:
                continue

            for def_item in sym_def_list:
                extra_ref = _location_to_callee_ref(def_item, sym_name)
                if extra_ref is None:
                    continue
                # Only add definitions that live in a DIFFERENT file; that's
                # what makes the symbol "imported" rather than locally defined.
                if extra_ref.uri == file_uri:
                    continue
                pos_key = (
                    extra_ref.uri,
                    extra_ref.range_start_line,
                    extra_ref.range_start_char,
                )
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)
                extra_callees.append(extra_ref)
                extra_count += 1
                seen_callee_names.add(sym_name)  # prevent re-querying same name

    # Update counters in diagnostic after step 3 may have added more.
    if extra_count > 0:
        diagnostic["full_mode_extra"] = extra_count
    if ref_errors > 0:
        diagnostic["full_mode_ref_errors"] = ref_errors

    # ── 4. Merge and recompute hash ────────────────────────────────────
    if not extra_callees:
        # No extra refs found — return base result (possibly with updated status
        # and diagnostic if the extra pass encountered errors).
        if status != base.status or diagnostic != dict(base.diagnostic):
            return ClosureResult(
                status=status,
                closure_hash=base.closure_hash,
                callees=base.callees,
                depth_reached=base.depth_reached,
                diagnostic=diagnostic,
            )
        return base

    all_callees: list[CalleeRef] = list(base.callees) + extra_callees
    closure_hash, unreadable = _compute_closure_hash(all_callees)
    if unreadable > 0:
        status = _escalate(status, "partial")
        diagnostic["unreadable_files"] = diagnostic.get("unreadable_files", 0) + unreadable

    return ClosureResult(
        status=status,
        closure_hash=closure_hash,
        callees=tuple(all_callees),
        depth_reached=base.depth_reached,
        diagnostic=diagnostic,
    )
