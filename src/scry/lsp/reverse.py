"""Reverse-link LSP queries: get_callers and get_subclasses (workstream W6e).

Implements DESIGN.md §5.3 / Wave 6 spec block (lines 1437-1447):
"Reverse-link queries leveraging the LSP-built call graph".

* :func:`get_callers`    — who CALLS this function (callHierarchy/incomingCalls)
* :func:`get_subclasses` — which classes EXTEND this class (textDocument/implementation)

Both functions:
- Perform capability checks before issuing any LSP requests.
- Apply defensive shape validation on every response (mirroring W3b closure.py).
- Never raise — return an empty tuple on error and log a WARNING.

References
----------
DESIGN.md §5.3  — callHierarchy semantics
DESIGN.md §11   — Wave 6 spec block, lines 1437-1447
LSP spec §3.16.5 — callHierarchy/incomingCalls
LSP spec §3.6.14 — textDocument/implementation
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

from scry.lsp.manager import LSPSession

logger = logging.getLogger(__name__)

__all__ = ["CallerRef", "SubclassRef", "get_callers", "get_subclasses"]


# ─── Public data-classes ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CallerRef:
    """A caller of the queried symbol.

    Records the LSP-reported identity (URI + name + range) of each function
    that calls the queried symbol.  Mirrors the structure of
    :class:`~scry.lsp.closure.CalleeRef` for the inverse direction.
    """

    uri: str
    name: str
    range_start_line: int
    range_start_char: int
    range_end_line: int
    range_end_char: int


@dataclass(frozen=True, slots=True)
class SubclassRef:
    """A subclass (implementation) of the queried class.

    Records the LSP-reported location of each class that implements or extends
    the queried class.  The ``name`` field is the best available identifier:
    the anchor's ``symbol_name`` if found in the index, else the source
    filename stem.
    """

    uri: str
    name: str
    range_start_line: int
    range_start_char: int
    range_end_line: int
    range_end_char: int


# ─── Internal helpers ─────────────────────────────────────────────────


def _item_to_caller_ref(item: Any) -> CallerRef | None:
    """Convert an LSP ``CallHierarchyItem`` dict to a :class:`CallerRef`.

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
    return CallerRef(
        uri=uri,
        name=name,
        range_start_line=s_line,  # type: ignore[arg-type]
        range_start_char=s_char,  # type: ignore[arg-type]
        range_end_line=e_line,  # type: ignore[arg-type]
        range_end_char=e_char,  # type: ignore[arg-type]
    )


def _caller_dedup_key(item: Any) -> tuple[str, str, int, int] | None:
    """Build a deduplication key from a CallHierarchyItem dict."""
    if not isinstance(item, dict):
        return None
    uri = item.get("uri")
    name = item.get("name")
    rng = item.get("range")
    if not isinstance(uri, str) or not isinstance(name, str) or not isinstance(rng, dict):
        return None
    start = rng.get("start")
    if not isinstance(start, dict):
        return None
    line = start.get("line")
    char = start.get("character")
    if not isinstance(line, int) or not isinstance(char, int):
        return None
    return (uri, name, line, char)


def _uri_stem(uri: str) -> str:
    """Extract the last path segment (without extension) from a URI."""
    try:
        parsed = urlparse(uri)
        raw = unquote(parsed.path)
        from pathlib import PurePosixPath

        return PurePosixPath(raw).stem or raw
    except Exception:
        return uri


def _location_to_ref_parts(
    loc: Any,
) -> tuple[str, int, int, int, int] | None:
    """Extract (uri, start_line, start_char, end_line, end_char) from a Location dict.

    Returns ``None`` when the shape is malformed.
    """
    if not isinstance(loc, dict):
        return None
    uri = loc.get("uri")
    rng = loc.get("range")
    if not isinstance(uri, str) or not uri or not isinstance(rng, dict):
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
    return uri, s_line, s_char, e_line, e_char  # type: ignore[return-value]


def _location_link_to_ref_parts(
    link: Any,
) -> tuple[str, int, int, int, int] | None:
    """Extract ref parts from a ``LocationLink`` dict (targetUri / targetRange).

    Returns ``None`` when the shape is malformed.
    """
    if not isinstance(link, dict):
        return None
    uri = link.get("targetUri")
    rng = link.get("targetRange")
    if not isinstance(uri, str) or not uri or not isinstance(rng, dict):
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
    return uri, s_line, s_char, e_line, e_char  # type: ignore[return-value]


# ─── Public API ───────────────────────────────────────────────────────


async def get_callers(
    session: LSPSession,
    file_uri: str,
    line: int,
    character: int,
    *,
    max_depth: int = 1,
    timeout_per_call: float = 5.0,
) -> tuple[CallerRef, ...]:
    """Return symbols that CALL the symbol at *(file_uri, line, character)*.

    Uses ``callHierarchy/incomingCalls`` — the inverse of
    :func:`~scry.lsp.closure.compute_closure`'s outgoing-calls walk.

    Parameters
    ----------
    session:
        A live, initialized :class:`~scry.lsp.manager.LSPSession`.
    file_uri:
        Absolute ``file://`` URI of the document containing the target symbol.
    line:
        0-based line index of the symbol's definition position.
    character:
        0-based character offset within *line*.
    max_depth:
        Number of incoming-call hops to walk.  The default of ``1`` returns
        only direct callers.  Values ``> 1`` produce transitive callers via BFS
        (uncommon in practice; callers typically care about direct callers only).
    timeout_per_call:
        Per-LSP-request timeout in seconds.

    Returns
    -------
    tuple[CallerRef, ...]
        De-duplicated callers in BFS discovery order (root item excluded).
        Empty tuple when unsupported, the prepare result is null/empty, or any
        error occurs (a WARNING is logged in error cases).

    Notes
    -----
    Never raises — all exceptions are caught and converted to an empty result
    with a logged WARNING.
    """
    # ── 1. Capability check ─────────────────────────────────────────────
    if not session.supports("callHierarchyProvider"):
        logger.debug(
            "LSP [%s] does not advertise callHierarchyProvider — get_callers skipped",
            session.language,
        )
        return ()

    # ── 2. prepareCallHierarchy ─────────────────────────────────────────
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
        logger.warning(
            "LSP [%s] textDocument/prepareCallHierarchy failed at %s:%d:%d — %s",
            session.language,
            file_uri,
            line,
            character,
            exc,
        )
        return ()

    if prepare_result is None or (isinstance(prepare_result, list) and not prepare_result):
        logger.debug(
            "LSP [%s] prepareCallHierarchy returned null/empty at %s:%d:%d",
            session.language,
            file_uri,
            line,
            character,
        )
        return ()

    if not isinstance(prepare_result, list):
        logger.warning(
            "LSP [%s] prepareCallHierarchy returned %s instead of list",
            session.language,
            type(prepare_result).__name__,
        )
        return ()

    root_item = prepare_result[0]
    if not isinstance(root_item, dict):
        logger.warning(
            "LSP [%s] prepareCallHierarchy first element is %s, not dict",
            session.language,
            type(root_item).__name__,
        )
        return ()

    # ── 3. BFS walk ─────────────────────────────────────────────────────
    # ``expanded`` tracks items whose incomingCalls have been fetched (dedup).
    # The root item is in expanded but NOT in results — it is the target.
    root_key = _caller_dedup_key(root_item)
    if root_key is None:
        logger.warning(
            "LSP [%s] prepareCallHierarchy root item is malformed",
            session.language,
        )
        return ()

    expanded: set[tuple[str, str, int, int]] = {root_key}
    callers: list[CallerRef] = []
    in_callers: set[tuple[str, str, int, int]] = set()

    queue: deque[tuple[dict[str, Any], int]] = deque([(root_item, 0)])

    while queue:
        current_item, depth = queue.popleft()

        if depth >= max_depth:
            continue

        try:
            incoming: Any = await session.request(
                "callHierarchy/incomingCalls",
                {"item": current_item},
                timeout=timeout_per_call,
            )
        except Exception as exc:
            logger.warning(
                "LSP [%s] callHierarchy/incomingCalls failed for %r — %s",
                session.language,
                current_item.get("name") if isinstance(current_item, dict) else "?",
                exc,
            )
            continue

        # null / [] → no callers for this item (not an error).
        if incoming is None:
            continue

        if not isinstance(incoming, list):
            logger.warning(
                "LSP [%s] callHierarchy/incomingCalls returned %s instead of list",
                session.language,
                type(incoming).__name__,
            )
            continue

        for call in incoming:
            if not isinstance(call, dict):
                logger.warning(
                    "LSP [%s] incomingCalls entry is not a dict: %r",
                    session.language,
                    type(call).__name__,
                )
                continue

            # Per LSP spec, IncomingCall.from is the caller item.
            caller_item = call.get("from")
            if not isinstance(caller_item, dict):
                logger.warning(
                    "LSP [%s] incomingCalls entry missing 'from' dict",
                    session.language,
                )
                continue

            key = _caller_dedup_key(caller_item)
            if key is None:
                logger.warning(
                    "LSP [%s] incomingCalls 'from' item malformed",
                    session.language,
                )
                continue

            ref = _item_to_caller_ref(caller_item)
            if ref is None:
                continue

            if key not in in_callers:
                in_callers.add(key)
                callers.append(ref)

            # Enqueue for deeper walk if not yet expanded and within depth budget.
            if key not in expanded and depth + 1 < max_depth:
                expanded.add(key)
                queue.append((caller_item, depth + 1))

    return tuple(callers)


async def get_subclasses(
    session: LSPSession,
    file_uri: str,
    line: int,
    character: int,
    *,
    timeout_per_call: float = 5.0,
) -> tuple[SubclassRef, ...]:
    """Return classes that EXTEND / implement the class at *(file_uri, line, character)*.

    Uses ``textDocument/implementation`` (the LSP standard for subclass
    discovery, per DESIGN.md §5.3 Wave 6 spec).  Handles all three response
    shapes defined by the LSP spec:

    * ``null``          — no implementations found.
    * ``Location``      — a single location object.
    * ``Location[]``    — an array of location objects.
    * ``LocationLink[]``— an array of location-link objects (``targetUri``
                          / ``targetRange`` fields).

    Parameters
    ----------
    session:
        A live, initialized :class:`~scry.lsp.manager.LSPSession`.
    file_uri:
        Absolute ``file://`` URI of the document containing the class.
    line:
        0-based line index of the class's definition position.
    character:
        0-based character offset within *line*.
    timeout_per_call:
        Per-LSP-request timeout in seconds.

    Returns
    -------
    tuple[SubclassRef, ...]
        Subclass locations (de-duplicated by URI + start position).
        Empty tuple when unsupported, null response, or an error occurs
        (WARNING logged for errors).

    Notes
    -----
    Never raises.  The ``name`` field on each :class:`SubclassRef` is
    populated from the URI stem when the symbol name is unavailable from the
    raw LSP response (``textDocument/implementation`` returns Locations, not
    CallHierarchyItems — there is no name field).
    """
    # ── 1. Capability check ─────────────────────────────────────────────
    if not session.supports("implementationProvider"):
        logger.debug(
            "LSP [%s] does not advertise implementationProvider — get_subclasses skipped",
            session.language,
        )
        return ()

    # ── 2. textDocument/implementation ──────────────────────────────────
    try:
        result: Any = await session.request(
            "textDocument/implementation",
            {
                "textDocument": {"uri": file_uri},
                "position": {"line": line, "character": character},
            },
            timeout=timeout_per_call,
        )
    except Exception as exc:
        logger.warning(
            "LSP [%s] textDocument/implementation failed at %s:%d:%d — %s",
            session.language,
            file_uri,
            line,
            character,
            exc,
        )
        return ()

    if result is None:
        return ()

    # ── 3. Normalise the three possible response shapes ──────────────────
    # Shape 1: Location[]  (most common)
    # Shape 2: Location    (single — some servers return bare Location)
    # Shape 3: LocationLink[] (targetUri / targetRange)
    locations: list[Any]
    if isinstance(result, list):
        locations = result
    elif isinstance(result, dict):
        # Single Location (bare object, not wrapped in array).
        locations = [result]
    else:
        logger.warning(
            "LSP [%s] textDocument/implementation returned unexpected type %s",
            session.language,
            type(result).__name__,
        )
        return ()

    subclasses: list[SubclassRef] = []
    seen: set[tuple[str, int, int]] = set()

    for loc in locations:
        if not isinstance(loc, dict):
            logger.warning(
                "LSP [%s] implementation result entry is not a dict: %r",
                session.language,
                type(loc).__name__,
            )
            continue

        # Detect shape: LocationLink has "targetUri"; Location has "uri".
        if "targetUri" in loc:
            parts = _location_link_to_ref_parts(loc)
        else:
            parts = _location_to_ref_parts(loc)

        if parts is None:
            logger.warning(
                "LSP [%s] implementation result entry malformed: %r",
                session.language,
                loc,
            )
            continue

        uri, s_line, s_char, e_line, e_char = parts
        dedup_key = (uri, s_line, s_char)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        name = _uri_stem(uri)
        subclasses.append(
            SubclassRef(
                uri=uri,
                name=name,
                range_start_line=s_line,
                range_start_char=s_char,
                range_end_line=e_line,
                range_end_char=e_char,
            )
        )

    return tuple(subclasses)
