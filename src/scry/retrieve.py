"""Hybrid BM25 + vector retrieval engine for scry (DESIGN.md §4, §4.1, §4.2).

Implements the §4.1 v3.1 hybrid retrieval algorithm:

  1. Embed the query via ``embedder.encode([query])``.
  2. Run vector ANN (``ScryDB.query_vector``) and BM25/FTS5
     (``ScryDB.query_bm25``) retrievals over the chunk pool.
  3. PROMOTE: for each parent anchor take its best-ranked chunk in each
     list so every parent appears exactly once per list.  **After dedup,
     each list is re-ranked 1..N before RRF** (v3.1 critical fix — without
     this step a long parent's many sub-chunks crowd out single-chunk parents
     in the lower rank positions, re-introducing length bias in inverse form).
  4. RRF-fuse the two parent-level lists:
         parent_score = Σ_{list} 1 / (k + parent_rank_in_list)
     where ``k = config.fusion_rrf_k`` (default 60).
  5. Sort parents by ``parent_score`` descending; take ``top_k``.
  6. Assemble ``AnchorPacket`` objects (``build_anchor_packet``).

No graph-traversal influence on ranking (§4.1 v3.1).  Graph context is
added by the caller via ``build_anchor_packet``.

Public API
----------
SearchResult        — one ranked result before AnchorPacket assembly
hybrid_search       — §4.1 v3.1 hybrid retrieval algorithm
build_anchor_packet — §4.2 turn a SearchResult into an AnchorPacket
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scry.embed import Embedder
from scry.models import (
    Anchor,
    AnchorLinkProjection,
    AnchorPacket,
    AnchorType,
    IndexState,
    RetrievalConfig,
)
from scry.store.db import ScryDB

__all__ = [
    "SearchResult",
    "build_anchor_packet",
    "hybrid_search",
]

# Singleton default config — avoids re-constructing on every call.
_DEFAULT_CONFIG = RetrievalConfig()


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One ranked result before AnchorPacket assembly (DESIGN.md §4.1 v3.1).

    Produced by :func:`hybrid_search`; consumed by :func:`build_anchor_packet`.
    All rank fields are 1-indexed post-promotion positions (never the raw
    chunk-level ranks from the ANN or FTS5 indices).
    """

    parent_anchor_id: str
    """Primary key of the parent :class:`~scry.models.Anchor`."""

    score: float
    """RRF-fused relevance score (higher is better)."""

    best_chunk_rowid_vec: int | None
    """Integer rowid of the vector-list best chunk for this parent.

    ``None`` when this parent had no hits in the vector list.
    """

    best_chunk_rowid_bm25: int | None
    """Integer rowid of the BM25-list best chunk for this parent.

    ``None`` when this parent had no hits in the BM25 list.
    """

    parent_rank_in_vec: int | None
    """1-indexed post-promotion rank in the vector list.

    ``None`` when this parent had no hits in the vector list.
    """

    parent_rank_in_bm25: int | None
    """1-indexed post-promotion rank in the BM25 list.

    ``None`` when this parent had no hits in the BM25 list.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chunk_rowids_to_parents(
    rowids: Sequence[int],
    *,
    db: ScryDB,
) -> dict[int, str]:
    """Map chunk integer rowids to their parent anchor IDs.

    Executes a single ``SELECT … WHERE id IN (…)`` query against the
    ``chunks`` table.  ``ScryDB`` does not expose a batch rowid-to-parent
    method, so we go directly to ``db._conn`` (per the W2d spec guidance:
    "Don't modify db.py").

    Args:
        rowids: Chunk primary keys (``chunks.id`` values) to resolve.
        db:     Live database connection.

    Returns:
        ``{chunk_rowid: parent_anchor_id}``; rowids absent from the table
        are silently omitted (handles stale FTS5/vec index entries).
    """
    if not rowids:
        return {}
    placeholders = ",".join("?" * len(rowids))
    rows = db._conn.execute(
        f"SELECT id, parent_id FROM chunks WHERE id IN ({placeholders})",
        list(rowids),
    ).fetchall()
    return {int(r[0]): str(r[1]) for r in rows}


def _get_parent_anchor_types(
    parent_ids: Sequence[str],
    *,
    db: ScryDB,
) -> dict[str, AnchorType]:
    """Return the :class:`~scry.models.AnchorType` for each parent anchor.

    Args:
        parent_ids: Anchor primary keys to look up.
        db:         Live database connection.

    Returns:
        ``{anchor_id: AnchorType}``; IDs absent from the table are omitted.
    """
    if not parent_ids:
        return {}
    placeholders = ",".join("?" * len(parent_ids))
    rows = db._conn.execute(
        f"SELECT id, type FROM anchors WHERE id IN ({placeholders})",
        list(parent_ids),
    ).fetchall()
    return {str(r[0]): AnchorType(str(r[1])) for r in rows}


def _get_parent_anchor_paths(
    parent_ids: Sequence[str],
    *,
    db: ScryDB,
) -> dict[str, str]:
    """Return the ``path`` for each parent anchor (UAT-15 review-u9-u10 HIGH).

    Used by :func:`hybrid_search` to push a path-glob filter down so
    ``--scope`` filtering happens BEFORE the final ``top_k`` truncation
    (otherwise scoped results that rank just outside the global top
    are silently dropped).
    """
    if not parent_ids:
        return {}
    placeholders = ",".join("?" * len(parent_ids))
    rows = db._conn.execute(
        f"SELECT id, path FROM anchors WHERE id IN ({placeholders})",
        list(parent_ids),
    ).fetchall()
    return {str(r[0]): str(r[1]) for r in rows}


def _promote(
    ranked: list[tuple[int, int]],
    rowid_to_parent: dict[int, str],
) -> list[tuple[int, str, int]]:
    """Best-chunk-per-parent promotion with post-dedup re-ranking (§4.1 v3.1).

    For each parent anchor, keep only the entry with the lowest original rank
    (its best-matching chunk).  After deduplication the survivors are
    re-ranked 1..N so that a single-chunk parent at original rank 6 is
    promoted to post-dedup rank 2 (not 6) when there is only one other
    parent — eliminating the length-bias that would otherwise re-emerge in
    the RRF denominator.

    Args:
        ranked:          ``[(original_rank, chunk_rowid), …]`` sorted by
                         original_rank ascending (1-indexed).
        rowid_to_parent: Chunk rowid → parent anchor ID.  Rowids absent
                         from this mapping (filtered-out type, stale index)
                         are silently skipped.

    Returns:
        ``[(post_dedup_rank, parent_anchor_id, best_chunk_rowid), …]``
        sorted by ``post_dedup_rank`` ascending (1-indexed).
    """
    # Track the best (lowest) original rank and its rowid per parent.
    seen: dict[str, tuple[int, int]] = {}  # parent_id → (best_rank, best_rowid)
    for original_rank, rowid in ranked:
        parent_id = rowid_to_parent.get(rowid)
        if parent_id is None:
            continue
        if parent_id not in seen or original_rank < seen[parent_id][0]:
            seen[parent_id] = (original_rank, rowid)

    # Sort by best original rank, then assign new 1-indexed positions.
    seen_sorted = sorted(seen.items(), key=lambda kv: kv[1][0])
    result: list[tuple[int, str, int]] = []
    for new_rank, (parent_id, best_pair) in enumerate(seen_sorted, start=1):
        result.append((new_rank, parent_id, best_pair[1]))
    return result


def _get_chunk_text(rowid: int, *, db: ScryDB) -> str | None:
    """Fetch the ``text`` column of a single chunk by its integer rowid.

    Args:
        rowid: ``chunks.id`` primary key.
        db:    Live database connection.

    Returns:
        The chunk text, or ``None`` if the rowid does not exist.
    """
    row = db._conn.execute("SELECT text FROM chunks WHERE id = ?", (rowid,)).fetchone()
    return str(row[0]) if row is not None else None


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate *text* to approximately *max_tokens* tokens (word-split method).

    Approximates token count as ``word_count * 1.4``, consistent with the
    ``split_section_text`` token budget used by the W2k chunking helper.

    Args:
        text:       Source text to possibly truncate.
        max_tokens: Maximum approximate token budget.

    Returns:
        ``(output_text, was_truncated)`` pair.  ``was_truncated`` is
        ``True`` only when characters were actually removed.
    """
    words = text.split()
    # word_count * 1.4 <= max_tokens  =>  word_count <= max_tokens / 1.4
    max_words = max(1, int(max_tokens / 1.4))
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def _sanitize_fts5_query(query: str) -> str:
    """Wrap user input in FTS5-safe quoted phrases.

    FTS5's MATCH expression treats many ordinary punctuation characters as
    syntax: ``-`` is NOT, ``+`` is PLUS, ``:`` is column-qualifier, ``/``,
    ``.``, ``,``, ``!``, ``(`` all trigger ``OperationalError: fts5: syntax
    error``.  Forwarding raw user queries (e.g.  ``"How does C++ work?"``,
    ``"file/path.py"``, ``"hello-world"``) crashes the server (review-w2d
    HIGH finding).

    Sanitization strategy: split the query on whitespace, then wrap each
    surviving non-empty token in double quotes after escaping any internal
    ``"`` (FTS5 syntax: ``""`` inside a quoted string represents a literal
    ``"``).  Quoted tokens are matched as exact terms regardless of any
    embedded operator characters.  Multiple tokens are space-joined; FTS5
    treats this as an implicit AND over the terms.

    Returns ``""`` when the query has no usable tokens (all whitespace);
    callers must short-circuit on that to avoid an empty-MATCH error.

    Examples:
        ``How does C++ work?``      → ``"How" "does" "C++" "work?"``
        ``file/path.py``            → ``"file/path.py"``
        ``unbalanced "quote``       → ``"unbalanced" "" + chr(34)*3 + "quote"`` -- internal " becomes "" pair
    """
    tokens = query.split()
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens if tok]
    return " ".join(quoted)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hybrid_search(
    query: str,
    *,
    db: ScryDB,
    embedder: Embedder,
    config: RetrievalConfig | None = None,
    top_k: int = 10,
    anchor_types: Sequence[AnchorType] | None = None,
    path_globs: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Run the §4.1 v3.1 hybrid BM25 + vector retrieval algorithm.

    Steps (per DESIGN.md §4.1 v3.1):

    1. Embed *query* via ``embedder.encode([query])``.
    2. Run two retrievals over the chunk pool:

       - Vector ANN via ``ScryDB.query_vector`` (parent overview +
         sub-chunk + code anchor embeddings).
       - BM25/FTS5 via ``ScryDB.query_bm25`` over the same set.

    3. **PROMOTE**: for each parent anchor keep only its best-ranked chunk in
       each list (one entry per parent per list).  **After dedup, re-rank
       each list 1..N** (v3.1 critical fix — prevents long parents from
       occupying lower slots and penalising single-chunk parents).

    4. **RRF-fuse** the two parent-level lists:

       .. code-block:: text

           parent_score = Σ_{list ∈ {vec, bm25}} 1 / (k + parent_rank_in_list)

       where ``k = config.fusion_rrf_k`` (default 60) and
       *parent_rank_in_list* is the post-promotion 1..N rank.

    5. Sort parents by ``parent_score`` descending; return the top *top_k*.

    No graph-traversal influence on ranking (§4.1 v3.1).  Graph context is
    added by the caller (``build_anchor_packet``).

    Args:
        query:        Natural-language search string.
        db:           Live ``ScryDB`` connection.
        embedder:     Embedding backend (any :class:`~scry.embed.Embedder`).
        config:       Retrieval configuration; defaults to
                      ``RetrievalConfig()`` when ``None``.
        top_k:        Maximum number of results to return.
        anchor_types: When given, restrict the candidate pool to parents
                      whose ``anchor_type`` is in this set.

    Returns:
        List of :class:`SearchResult` sorted by ``score`` descending,
        length ≤ *top_k*.  Returns ``[]`` on an empty/whitespace-only
        query or when the index contains no matching chunks.
    """
    if not query.strip():
        return []

    cfg = config if config is not None else _DEFAULT_CONFIG

    # Step 1: Embed the query.
    query_embedding: bytes = embedder.encode([query])[0]

    # Step 2: Retrieve a large candidate set from both indices so that
    # promotion always yields ≥ top_k unique parents after dedup.
    candidate_k = max(top_k * 20, 200)
    try:
        vec_raw: list[tuple[int, float]] = db.query_vector(query_embedding, top_k=candidate_k)
    except Exception as exc:
        # UAT-M-11: sqlite-vec raises an error that includes the internal
        # oversampled candidate_k (top_k * 20) rather than the user-supplied
        # top_k.  Re-raise with the user-facing value so the error message is
        # actionable ("k value 100000" not "k value 2000000").
        msg = str(exc)
        if "k value" in msg:
            raise ValueError(
                f"top_k {top_k} is too large for the vector index. Use a smaller value."
            ) from exc
        raise
    # Sanitize the query for FTS5 (review-w2d HIGH fix): natural-language
    # queries with punctuation crash MATCH otherwise.  An empty sanitized
    # query short-circuits BM25 (vector retrieval still runs).
    bm25_query = _sanitize_fts5_query(query)
    bm25_raw: list[tuple[int, float]] = (
        db.query_bm25(bm25_query, top_k=candidate_k) if bm25_query else []
    )

    if not vec_raw and not bm25_raw:
        return []

    # Resolve all candidate chunk rowids to parent anchor IDs in one query.
    all_rowids: list[int] = list({rowid for rowid, _ in vec_raw} | {rowid for rowid, _ in bm25_raw})
    rowid_to_parent = _chunk_rowids_to_parents(all_rowids, db=db)

    # Optional anchor-type filter: remove entries for disallowed parent types.
    if anchor_types is not None:
        type_set = set(anchor_types)
        all_parent_ids: list[str] = list(set(rowid_to_parent.values()))
        parent_type_map = _get_parent_anchor_types(all_parent_ids, db=db)
        allowed_parents = {pid for pid, atype in parent_type_map.items() if atype in type_set}
        rowid_to_parent = {
            rowid: pid for rowid, pid in rowid_to_parent.items() if pid in allowed_parents
        }

    # UAT-15 review-u9-u10 HIGH: push the path-glob filter down so it
    # happens BEFORE the final top_k truncation; otherwise scoped
    # results that rank just outside the global top are silently dropped.
    if path_globs:
        from scry.config import matches_globs as _matches

        path_parent_ids: list[str] = list(set(rowid_to_parent.values()))
        parent_path_map = _get_parent_anchor_paths(path_parent_ids, db=db)
        allowed_by_path = {
            pid for pid, p in parent_path_map.items() if _matches(p, list(path_globs))
        }
        rowid_to_parent = {
            rowid: pid for rowid, pid in rowid_to_parent.items() if pid in allowed_by_path
        }

    # Step 3: Assign 1-indexed original ranks to each chunk list, then promote
    # to parent level (best chunk per parent + post-dedup re-ranking).
    vec_ranked: list[tuple[int, int]] = [
        (rank, rowid) for rank, (rowid, _) in enumerate(vec_raw, start=1)
    ]
    bm25_ranked: list[tuple[int, int]] = [
        (rank, rowid) for rank, (rowid, _) in enumerate(bm25_raw, start=1)
    ]
    vec_promoted = _promote(vec_ranked, rowid_to_parent)
    bm25_promoted = _promote(bm25_ranked, rowid_to_parent)

    # Build parent-level rank and best-chunk-rowid lookup tables.
    vec_parent_rank: dict[str, int] = {}
    vec_parent_chunk: dict[str, int] = {}
    for new_rank, parent_id, best_rowid in vec_promoted:
        vec_parent_rank[parent_id] = new_rank
        vec_parent_chunk[parent_id] = best_rowid

    bm25_parent_rank: dict[str, int] = {}
    bm25_parent_chunk: dict[str, int] = {}
    for new_rank, parent_id, best_rowid in bm25_promoted:
        bm25_parent_rank[parent_id] = new_rank
        bm25_parent_chunk[parent_id] = best_rowid

    # Step 4: RRF fusion over all parents that appear in at least one list.
    k = cfg.fusion_rrf_k
    all_parents: set[str] = set(vec_parent_rank) | set(bm25_parent_rank)
    results: list[SearchResult] = []
    for parent_id in all_parents:
        score = 0.0
        v_rank = vec_parent_rank.get(parent_id)
        b_rank = bm25_parent_rank.get(parent_id)
        if v_rank is not None:
            score += 1.0 / (k + v_rank)
        if b_rank is not None:
            score += 1.0 / (k + b_rank)
        results.append(
            SearchResult(
                parent_anchor_id=parent_id,
                score=score,
                best_chunk_rowid_vec=vec_parent_chunk.get(parent_id),
                best_chunk_rowid_bm25=bm25_parent_chunk.get(parent_id),
                parent_rank_in_vec=v_rank,
                parent_rank_in_bm25=b_rank,
            )
        )

    # Step 5: Sort descending by score, return top_k.
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def build_anchor_packet(
    result: SearchResult,
    *,
    db: ScryDB,
    config: RetrievalConfig | None = None,
    index_state: IndexState = IndexState.FRESH,
) -> AnchorPacket:
    """§4.2: turn a :class:`SearchResult` into an :class:`~scry.models.AnchorPacket`.

    Steps (per DESIGN.md §4.2):

    1. Load the parent :class:`~scry.models.Anchor` from the database.
    2. Compute the evidence excerpt: prefer the **vector-list** best chunk
       (§4.1 v3.1 — when BM25 and vector best-chunks differ, the vector
       chunk wins because it captures semantic relevance; the BM25 chunk
       is available via ``get_anchor(id)?evidence_for=bm25``).
    3. Truncate the anchor's ``content_text`` to
       ``config.content_preview_tokens`` tokens; set
       ``content_truncated=True`` if truncation occurred.
    4. Pull 1-hop neighbors from the link store — **deferred** (returns
       ``links=[]`` for now; W2i will wire link enumeration once the link
       store workstreams W2b/W2c stabilise).
    5. Compute ``drift_status`` for each link — **deferred to W2e**
       (defaults to ``DriftStatus.FRESH`` on any ``AnchorLinkProjection``
       objects created later).
    6. Set ``index_state`` on the packet.

    Args:
        result:      :class:`SearchResult` from :func:`hybrid_search`.
        db:          Live ``ScryDB`` connection.
        config:      Retrieval configuration; defaults to
                     ``RetrievalConfig()`` when ``None``.
        index_state: Index freshness to attach to the returned packet.

    Returns:
        Populated :class:`~scry.models.AnchorPacket`.

    Raises:
        KeyError: If the parent anchor ID is not present in the database.
    """
    cfg = config if config is not None else _DEFAULT_CONFIG

    # Step 1: Load the parent anchor.
    anchor = db.get_anchor(result.parent_anchor_id)
    if anchor is None:
        raise KeyError(f"Anchor not found in database: {result.parent_anchor_id!r}")

    # Step 2: Evidence excerpt — prefer vector-list best chunk; fall back to
    # BM25 best chunk when the parent had no vector hits.
    preferred_rowid: int | None = (
        result.best_chunk_rowid_vec
        if result.best_chunk_rowid_vec is not None
        else result.best_chunk_rowid_bm25
    )
    evidence_excerpt: str | None = (
        _get_chunk_text(preferred_rowid, db=db) if preferred_rowid is not None else None
    )

    # Step 3: Truncate content to config.content_preview_tokens.
    truncated_text, content_truncated = _truncate_to_tokens(
        anchor.content_text, cfg.content_preview_tokens
    )
    # Create a new (frozen) Anchor copy with the (possibly truncated) content.
    display_anchor: Anchor = anchor.model_copy(update={"content_text": truncated_text})

    # UAT-R5-9: deduplicate evidence_excerpt vs displayed content.
    # When the excerpt is byte-identical to the truncated content, it adds 0
    # information and roughly doubles the token payload.  Drop it.
    # When it IS a true substring (long anchor, specific chunk), cap at 200
    # chars and record the match offset so callers can reconstruct context.
    #
    # NOTE (review-r6abc-3): ``match_offset`` is a CHARACTER offset (Python
    # ``str.find``), not a byte offset.  This is consistent with the rest
    # of the AnchorPacket which exposes character-based positions.  We
    # explicitly return ``None`` when the excerpt is NOT a substring of
    # ``content_text`` (e.g. generated chunks, overlap windows) — falling
    # back to ``0`` would silently misdirect callers to the start of the
    # anchor.
    match_offset: int | None = None
    if evidence_excerpt is not None:
        if evidence_excerpt == truncated_text:
            # Exact match — omit the duplicate.
            evidence_excerpt = None
        else:
            # True substring — record offset into the ORIGINAL content, then
            # cap to 200 chars to avoid per-result token waste.  Offset is
            # a CHARACTER offset; ``None`` if the excerpt is not a substring.
            raw_offset = anchor.content_text.find(evidence_excerpt)
            match_offset = raw_offset if raw_offset >= 0 else None
            if len(evidence_excerpt) > 200:
                evidence_excerpt = evidence_excerpt[:200]

    # Steps 4 & 5: Links — deferred to W2i / W2e.
    # TODO(W2i): populate outgoing/incoming links from the link store once
    # W2b/W2c link-store workstreams are stable.
    links: list[AnchorLinkProjection] = []

    return AnchorPacket(
        anchor=display_anchor,
        score=result.score,
        evidence_excerpt=evidence_excerpt,
        match_offset=match_offset,
        links=links,
        index_state=index_state,
        content_truncated=content_truncated,
    )
