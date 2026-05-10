"""Transitive call-closure walker using LSP callHierarchy protocol.

Implements DESIGN.md §5.3: builds the transitive outgoing-call closure
for a single code anchor by walking ``callHierarchy/outgoingCalls`` edges
from a starting file position, then computes a content-fingerprint hash
over the canonicalized definitions of every reachable callee.

Status values
-------------
The walker returns a :class:`ClosureResult` whose ``status`` is one of
the spec values from §5.3 (no internal-only synonyms):

* ``complete`` — every reachable in-repo callee was resolved.  A leaf
  function with zero outgoing calls is ``complete`` vacuously (per the
  §5.3 status table).
* ``partial`` — at least one symbol in the closure was unresolvable, the
  recursion-depth cap was hit, or a callee resolved outside the
  include set.  The closure hash incorporates whatever WAS resolved.
* ``unsupported`` — the LSP did not advertise ``callHierarchyProvider``
  or ``textDocument/prepareCallHierarchy`` returned null/empty for this
  symbol.  Only the anchor's own AST text contributes to the
  ``content_hash`` upstream; closure_hash is the empty-content sentinel.
* ``lsp_error`` — runtime failure (timeout, crash, malformed response)
  during the walk.  The closure hash incorporates the partial walk.

(``lsp_unavailable`` — binary not installed — is handled by
:class:`~scry.lsp.manager.LSPManager` BEFORE compute_closure is called.)

Cycle handling
--------------
Cycles are silently handled by walker dedup: every callee key is added
to ``expanded`` after its outgoingCalls are fetched, so back-edges
terminate naturally without re-walking.  Per §5.3, cycle detection is a
correctness mechanism — it does NOT degrade ``status``.  A diamond DAG
(A→B, A→C, B→C) is therefore ``complete``, not falsely ``partial``.

Closure hash semantics (§3.5, §5.3 lines 622-625)
-------------------------------------------------
The closure hash is SHA-256 over the **sorted concatenation of
canonicalized callee body content hashes**.  For each callee, the
walker reads its file, slices the LSP-reported range, canonicalizes
per :func:`scry.anchor_id.canonicalize_content`, and hashes.  The
per-callee hashes are then sorted lexicographically and combined.

This means:

* Renaming a callee does NOT change the hash (the caller's own AST
  hash does — that's where renames are caught).
* Editing a callee's body DOES change the hash (the drift signal §5.3
  promises).
* Reordering / re-sorting outgoing calls does NOT change the hash
  (BFS vs DFS is irrelevant).
* A callee in another file that we cannot read (permission, missing
  file) → escalates to ``partial`` (its hash is omitted from the
  combination).

Default depth cap
-----------------
``max_depth=32`` per ``code_anchors.transitive_max_depth`` default
(DESIGN.md §6 / §11 config schema, also reflected in
:class:`scry.models.CodeAnchorsConfig`).

References
----------
DESIGN.md §5.3  — transitive code drift via callHierarchy
DESIGN.md §3.5  — code_anchor fields: closure_hash, transitive_hash_status
DESIGN.md §5.4  — content canonicalization (delegated to anchor_id)
DESIGN.md §5.1  — section-level drift signals consume transitive_hash_status
"""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from scry.anchor_id import canonicalize_content
from scry.lsp.manager import LSPSession

logger = logging.getLogger(__name__)

__all__ = ["CalleeRef", "ClosureResult", "ClosureStatus", "compute_closure"]

# ─── Status enum (spec values; §5.3) ──────────────────────────────────

ClosureStatus = Literal["complete", "partial", "unsupported", "lsp_error"]

# Status precedence per §5.3 lines 678-681:
#   complete > partial > unsupported > lsp_error  (best to worst)
# When walking, we keep track of the *worst* observed status.
_STATUS_RANK: dict[str, int] = {
    "complete": 0,
    "partial": 1,
    "unsupported": 2,
    "lsp_error": 3,
}


def _escalate(current: ClosureStatus, candidate: ClosureStatus) -> ClosureStatus:
    """Return whichever status represents the worse quality signal."""
    return candidate if _STATUS_RANK[candidate] > _STATUS_RANK[current] else current


# ─── Public data-classes ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CalleeRef:
    """Reference to a single callee in the transitive closure.

    Records the LSP-reported identity (URI + name + range) so that the
    walker output remains useful for diagnostics, dedup, and downstream
    inspection.  The name and range are NOT inputs to ``closure_hash``;
    only the callee's canonicalized body content is hashed.
    """

    uri: str
    name: str
    range_start_line: int
    range_start_char: int
    range_end_line: int
    range_end_char: int


@dataclass(frozen=True, slots=True)
class ClosureResult:
    """Outcome of a transitive call-closure walk.

    Attributes
    ----------
    status:
        Spec-aligned quality signal (§5.3 status table).
    closure_hash:
        Hex-encoded SHA-256 over the sorted callee-content hashes.  An
        empty closure (no resolved callees) returns the SHA-256 of the
        empty byte sequence:
        ``e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855``.
    callees:
        Ordered, de-duplicated callees in BFS discovery order.  Useful
        for diagnostics; not part of the hash.
    depth_reached:
        Deepest BFS level whose callees were added to the closure
        (0 = no callees discovered).
    diagnostic:
        Free-form structured detail about WHY the status escalated.
        Keys may include ``"cycle"``, ``"depth_cap"``,
        ``"unreadable_files"``, ``"malformed_responses"``.  Empty when
        ``status == "complete"``.
    """

    status: ClosureStatus
    closure_hash: str
    callees: tuple[CalleeRef, ...]
    depth_reached: int
    diagnostic: dict[str, Any]


# ─── Internal helpers ─────────────────────────────────────────────────

# Deduplication key: (uri, name, (start_line, start_char))
_VisitKey = tuple[str, str, tuple[int, int]]

# SHA-256 of zero bytes — used as the empty-closure sentinel.
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _visit_key_for(item_or_ref: dict[str, Any] | CalleeRef) -> _VisitKey | None:
    """Build a deduplication key from a CallHierarchyItem dict or CalleeRef.

    Returns ``None`` if the input is malformed (missing required fields).
    """
    if isinstance(item_or_ref, CalleeRef):
        return (
            item_or_ref.uri,
            item_or_ref.name,
            (item_or_ref.range_start_line, item_or_ref.range_start_char),
        )
    if not isinstance(item_or_ref, dict):
        return None
    uri = item_or_ref.get("uri")
    name = item_or_ref.get("name")
    rng = item_or_ref.get("range")
    if not isinstance(uri, str) or not isinstance(name, str) or not isinstance(rng, dict):
        return None
    start = rng.get("start")
    if not isinstance(start, dict):
        return None
    line = start.get("line")
    char = start.get("character")
    if not isinstance(line, int) or not isinstance(char, int):
        return None
    return (uri, name, (line, char))


def _item_to_ref(item: dict[str, Any]) -> CalleeRef | None:
    """Convert an LSP ``CallHierarchyItem`` dict to a :class:`CalleeRef`.

    Returns ``None`` when required fields are missing or malformed.
    """
    if not isinstance(item, dict):
        return None
    uri = item.get("uri")
    name = item.get("name")
    rng = item.get("range")
    if not isinstance(uri, str) or not uri:
        return None
    if not isinstance(name, str):
        return None
    if not isinstance(rng, dict):
        return None
    start = rng.get("start")
    end = rng.get("end")
    if not isinstance(start, dict) or not isinstance(end, dict):
        return None
    s_line = start.get("line")
    s_char = start.get("character")
    e_line = end.get("line")
    e_char = end.get("character")
    if not all(isinstance(v, int) for v in (s_line, s_char, e_line, e_char)):
        return None
    return CalleeRef(
        uri=uri,
        name=name,
        range_start_line=s_line,  # type: ignore[arg-type]
        range_start_char=s_char,  # type: ignore[arg-type]
        range_end_line=e_line,  # type: ignore[arg-type]
        range_end_char=e_char,  # type: ignore[arg-type]
    )


def _file_uri_to_path(uri: str) -> Path | None:
    """Convert a ``file://`` URI to a local :class:`Path`, or ``None``.

    Non-file schemes return None.  Handles Windows drive-letter URIs
    (``file:///C:/x``) correctly.
    """
    try:
        parsed = urlparse(uri)
    except ValueError:
        return None
    if parsed.scheme.lower() != "file":
        return None
    if parsed.netloc and parsed.netloc.lower() != "localhost":
        return None
    raw = unquote(parsed.path)
    # On Windows, urllib leaves a leading slash before the drive: ``/C:/x``
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    try:
        return Path(raw)
    except (OSError, ValueError):
        return None


def _slice_range(text: str, start_line: int, start_char: int, end_line: int, end_char: int) -> str:
    """Extract the substring spanning the LSP range from *text*.

    LSP ranges are 0-indexed (line, character) and end-exclusive on the
    character offset.  Lines are split on ``\\n``; carriage returns are
    preserved if present (canonicalization normalizes them upstream).
    """
    lines = text.split("\n")
    if start_line >= len(lines):
        return ""
    if end_line >= len(lines):
        end_line = len(lines) - 1
        end_char = len(lines[end_line])
    if start_line == end_line:
        return lines[start_line][start_char:end_char]
    parts = [lines[start_line][start_char:]]
    parts.extend(lines[start_line + 1 : end_line])
    parts.append(lines[end_line][:end_char])
    return "\n".join(parts)


def _hash_callee_body(
    ref: CalleeRef,
    file_cache: dict[str, str | None],
) -> bytes | None:
    """Return SHA-256 (raw bytes) of the canonicalized body of *ref*.

    Returns ``None`` when the file cannot be read (missing, permission,
    non-file URI), in which case the caller escalates to ``partial``.
    The *file_cache* is keyed by URI and stores either the file text or
    ``None`` if reading already failed (avoids retrying on every callee
    in the same file).
    """
    cached = file_cache.get(ref.uri, _SENTINEL)
    if cached is _SENTINEL:
        path = _file_uri_to_path(ref.uri)
        if path is None:
            file_cache[ref.uri] = None
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("closure: cannot read %s — %s", ref.uri, exc)
            file_cache[ref.uri] = None
            return None
        file_cache[ref.uri] = text
        cached = text
    if cached is None:
        return None
    body = _slice_range(
        cached,
        ref.range_start_line,
        ref.range_start_char,
        ref.range_end_line,
        ref.range_end_char,
    )
    return hashlib.sha256(canonicalize_content(body).encode("utf-8")).digest()


_SENTINEL: Any = object()


def _compute_closure_hash(
    callees: list[CalleeRef],
) -> tuple[str, int]:
    """Hash the closure as sorted concatenation of per-callee body hashes.

    Returns ``(hex_digest, unreadable_count)``.  ``unreadable_count``
    is the number of callees whose body could not be read; the caller
    uses this to escalate ``status`` to ``partial``.
    """
    file_cache: dict[str, str | None] = {}
    per_callee_hashes: list[bytes] = []
    unreadable = 0
    for ref in callees:
        h = _hash_callee_body(ref, file_cache)
        if h is None:
            unreadable += 1
            continue
        per_callee_hashes.append(h)
    if not per_callee_hashes:
        return _EMPTY_SHA256, unreadable
    per_callee_hashes.sort()
    combiner = hashlib.sha256()
    for h in per_callee_hashes:
        combiner.update(h)
    return combiner.hexdigest(), unreadable


# ─── Public API ───────────────────────────────────────────────────────


async def compute_closure(
    session: LSPSession,
    file_uri: str,
    line: int,
    character: int,
    *,
    max_depth: int = 32,
    timeout_per_call: float = 5.0,
) -> ClosureResult:
    """Compute the transitive outgoing-call closure from a file position.

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
        Maximum BFS depth to recurse.  Nodes discovered AT ``max_depth``
        are added to the closure, but their children are not fetched.
        Hitting the cap escalates ``status`` to ``partial`` per §5.3
        status table.  Default 32 per
        ``code_anchors.transitive_max_depth`` (DESIGN.md §6 / §11).
    timeout_per_call:
        Per-LSP-request timeout in seconds (applies to both
        ``textDocument/prepareCallHierarchy`` and each
        ``callHierarchy/outgoingCalls`` call).

    Returns
    -------
    ClosureResult
        Always returns a result — never raises.  Errors during the walk
        produce ``status == "lsp_error"`` and any partial closure
        collected so far.

    Notes
    -----
    The walker assumes the caller has already opened the document via
    ``textDocument/didOpen`` (or that the LSP server reads files lazily).
    Per LSP spec ``prepareCallHierarchy`` may require the document to be
    open; this is the caller's responsibility.
    """
    diagnostic: dict[str, Any] = {}

    # ── 1. Capability check ─────────────────────────────────────────
    if not session.supports("callHierarchyProvider"):
        logger.debug(
            "LSP [%s] does not advertise callHierarchyProvider — skipping closure walk",
            session.language,
        )
        return ClosureResult(
            status="unsupported",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "callHierarchyProvider not advertised"},
        )

    # ── 2. prepareCallHierarchy ──────────────────────────────────────
    try:
        prepare_result: Any = await session.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character},
            },
            timeout=timeout_per_call,
        )
    except Exception as exc:
        logger.debug(
            "LSP [%s] textDocument/prepareCallHierarchy failed at %s:%d:%d — %s",
            session.language,
            file_uri,
            line,
            character,
            exc,
        )
        return ClosureResult(
            status="lsp_error",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "prepareCallHierarchy raised", "exception": repr(exc)},
        )

    # Null OR empty list → position is not callable; treat as
    # ``unsupported`` per §5.3 status table ("LSP doesn't implement
    # callHierarchy for this symbol's language" — same observable
    # outcome from the closure's perspective).
    if prepare_result is None or (isinstance(prepare_result, list) and not prepare_result):
        logger.debug(
            "LSP [%s] prepareCallHierarchy returned null/empty at %s:%d:%d — not callable",
            session.language,
            file_uri,
            line,
            character,
        )
        return ClosureResult(
            status="unsupported",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "prepareCallHierarchy returned null/empty"},
        )

    # Defensive shape check (review-w3b HIGH fix): server MUST return a list.
    if not isinstance(prepare_result, list):
        logger.warning(
            "LSP [%s] prepareCallHierarchy returned %s instead of list",
            session.language,
            type(prepare_result).__name__,
        )
        return ClosureResult(
            status="lsp_error",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "prepareCallHierarchy non-list result"},
        )

    root_item = prepare_result[0]
    if not isinstance(root_item, dict):
        return ClosureResult(
            status="lsp_error",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "prepareCallHierarchy first element non-dict"},
        )
    root_key = _visit_key_for(root_item)
    if root_key is None:
        return ClosureResult(
            status="lsp_error",
            closure_hash=_EMPTY_SHA256,
            callees=(),
            depth_reached=0,
            diagnostic={"reason": "prepareCallHierarchy root malformed"},
        )

    # ── 3. BFS walk ──────────────────────────────────────────────────
    # ``expanded`` holds keys whose outgoingCalls have already been fetched.
    # This is purely for dedup / cycle avoidance; per §5.3 cycles are
    # algorithmic-correctness concerns and do NOT degrade ``status``
    # (review-w3b HIGH: diamond DAG fix).
    expanded: set[_VisitKey] = {root_key}

    closure_list: list[CalleeRef] = []
    in_closure: set[_VisitKey] = set()

    status: ClosureStatus = "complete"
    depth_reached: int = 0
    cycle_count: int = 0
    malformed_count: int = 0

    queue: deque[tuple[dict[str, Any], int]] = deque([(root_item, 0)])

    while queue:
        current_item, depth = queue.popleft()

        # Hitting the cap means we have callees beyond what we'll walk
        # into — that's the §5.3 "partial" case.
        if depth >= max_depth:
            status = _escalate(status, "partial")
            diagnostic.setdefault("depth_cap", max_depth)
            logger.debug(
                "LSP [%s] depth cap (%d) reached at '%s'",
                session.language,
                max_depth,
                current_item.get("name") if isinstance(current_item, dict) else "?",
            )
            continue

        # ── outgoingCalls request ────────────────────────────────────
        try:
            outgoing: Any = await session.request(
                "callHierarchy/outgoingCalls",
                {"item": current_item},
                timeout=timeout_per_call,
            )
        except Exception as exc:
            logger.debug(
                "LSP [%s] callHierarchy/outgoingCalls failed for '%s' — %s",
                session.language,
                current_item.get("name") if isinstance(current_item, dict) else "?",
                exc,
            )
            status = _escalate(status, "lsp_error")
            diagnostic.setdefault("lsp_error_messages", []).append(repr(exc))
            continue

        # outgoingCalls null → leaf function (no outgoing calls).
        # Per LSP spec both null and [] are equivalent (review-w3b
        # MEDIUM fix: was previously misclassified as lsp_error).
        if outgoing is None:
            continue

        if not isinstance(outgoing, list):
            logger.warning(
                "LSP [%s] callHierarchy/outgoingCalls returned %s instead of list",
                session.language,
                type(outgoing).__name__,
            )
            status = _escalate(status, "lsp_error")
            diagnostic.setdefault("malformed_responses", 0)
            diagnostic["malformed_responses"] = diagnostic.get("malformed_responses", 0) + 1
            continue

        # ── process each outgoing edge ──────────────────────────────
        for outgoing_call in outgoing:
            if not isinstance(outgoing_call, dict):
                malformed_count += 1
                status = _escalate(status, "lsp_error")
                continue

            callee_item = outgoing_call.get("to")
            if not isinstance(callee_item, dict):
                # Legacy / non-spec servers sometimes flatten the item
                # directly into the OutgoingCall object.  Try that as a
                # fallback before giving up.
                callee_item = outgoing_call
            callee_ref = _item_to_ref(callee_item)
            if callee_ref is None:
                malformed_count += 1
                status = _escalate(status, "lsp_error")
                continue

            callee_key = _visit_key_for(callee_ref)
            if callee_key is None:
                malformed_count += 1
                status = _escalate(status, "lsp_error")
                continue

            # Add to closure (dedup) regardless of whether it's been expanded.
            if callee_key not in in_closure:
                closure_list.append(callee_ref)
                in_closure.add(callee_key)

            if callee_key in expanded:
                # Already walked (diamond convergence or back-edge).
                # Per §5.3, cycles do NOT degrade status — they're a
                # walker-correctness concern only.  Track in diagnostic
                # for observability (review-w3b HIGH fix).
                if callee_key == root_key or callee_key in expanded:
                    cycle_count += 1
                continue

            expanded.add(callee_key)
            next_depth = depth + 1
            depth_reached = max(depth_reached, next_depth)
            queue.append((callee_item, next_depth))

    # ── 4. Compute content-fingerprint hash ──────────────────────────
    closure_hash, unreadable = _compute_closure_hash(closure_list)
    if unreadable > 0:
        # Some callee bodies couldn't be read — closure is incomplete.
        status = _escalate(status, "partial")
        diagnostic["unreadable_files"] = unreadable

    if cycle_count > 0:
        diagnostic["cycle_edges"] = cycle_count
    if malformed_count > 0:
        diagnostic["malformed_outgoing_entries"] = malformed_count

    return ClosureResult(
        status=status,
        closure_hash=closure_hash,
        callees=tuple(closure_list),
        depth_reached=depth_reached,
        diagnostic=diagnostic,
    )
