"""Tests for scry.retrieve — hybrid BM25 + vector retrieval engine.

Covers the §4.1 v3.1 algorithm:
  - best-chunk-per-parent promotion
  - post-dedup re-ranking (the v3.1 critical fix)
  - RRF fusion math
  - anchor-type filtering
  - build_anchor_packet (§4.2) round-trips
  - content_preview_tokens truncation
  - evidence_excerpt vector-preference rule
  - empty-query / empty-index edge cases
"""

from __future__ import annotations

import struct
from collections.abc import Generator
from pathlib import Path

import pytest

from scry.models import (
    Anchor,
    AnchorPacket,
    AnchorType,
    IndexState,
    RetrievalConfig,
    SubChunk,
    TransitiveHashStatus,
)
from scry.retrieve import (
    SearchResult,
    _promote,
    _sanitize_fts5_query,
    _truncate_to_tokens,
    build_anchor_packet,
    hybrid_search,
)
from scry.store.db import ScryDB

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMS = 4  # Low dimensionality keeps tests fast and deterministic.
_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pack_vec(*values: float) -> bytes:
    """Pack floats into a little-endian float32 blob for sqlite-vec."""
    return struct.pack(f"<{len(values)}f", *values)


def _make_anchor(
    anchor_id: str,
    *,
    content_text: str = "hello world",
    anchor_type: AnchorType = AnchorType.SECTION,
    content_hash: str = _HASH_A,
    path: str | None = None,
) -> Anchor:
    """Build a minimal valid :class:`~scry.models.Anchor` for testing."""
    if path is None:
        path = f"docs/{anchor_id.replace('::', '_')}.md"
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        content_text=content_text,
        content_hash=content_hash,
        fingerprint_simhash=0,
    )


def _make_chunk(
    parent_id: str,
    chunk_index: int,
    text: str,
    *,
    parent_hash: str = _HASH_A,
) -> SubChunk:
    """Build a minimal :class:`~scry.models.SubChunk` for testing."""
    return SubChunk(
        parent_id=parent_id,
        chunk_index=chunk_index,
        text=text,
        parent_content_hash=parent_hash,
    )


# ---------------------------------------------------------------------------
# FixedEmbedder — returns a constant blob for every query (no fastembed needed)
# ---------------------------------------------------------------------------


class FixedEmbedder:
    """Embedder that always returns the same blob regardless of input text.

    Allows tests to control the exact query vector sent to sqlite-vec, making
    ANN ranking predictable when chunk embeddings are inserted manually.
    """

    def __init__(self, embedding: bytes, dims: int) -> None:
        self._embedding = embedding
        self._dims = dims

    @property
    def model_name(self) -> str:
        return "fixed-stub"

    @property
    def dimensions(self) -> int:
        return self._dims

    @property
    def provider(self) -> str:
        return "stub"

    @property
    def tokenizer_version(self) -> str | None:
        return None

    def encode(self, texts: list[str]) -> list[bytes]:
        return [self._embedding] * len(texts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_repo: Path) -> Generator[ScryDB, None, None]:
    """Fresh ScryDB at DIMS=4 dimensionality."""
    d = ScryDB(tmp_repo)
    d.init_schema(embedding_dimensions=DIMS)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Unit tests for _promote
# ---------------------------------------------------------------------------


class TestPromote:
    """Unit tests for the internal _promote helper."""

    def test_basic_single_parent(self) -> None:
        """Single parent with one chunk → post-dedup rank 1."""
        ranked = [(1, 10)]
        rtp = {10: "parent_a"}
        result = _promote(ranked, rtp)
        assert result == [(1, "parent_a", 10)]

    def test_best_chunk_per_parent_keeps_lowest_rank(self) -> None:
        """Parent with two chunks at ranks 3 and 7 → best chunk rowid is rank-3 chunk."""
        ranked = [(3, 30), (7, 70)]
        rtp = {30: "parent_a", 70: "parent_a"}
        result = _promote(ranked, rtp)
        assert len(result) == 1
        _new_rank, parent_id, best_rowid = result[0]
        assert parent_id == "parent_a"
        assert best_rowid == 30  # rowid at rank 3, not 7

    def test_reranking_single_chunk_parent_gets_rank_2(self) -> None:
        """Key re-ranking invariant: B at original rank 6 → post-dedup rank 2.

        With a long parent A consuming original ranks 1-5 and a single-chunk
        parent B at rank 6, after promotion B should be re-ranked to position
        2 (not left at 6).  This test FAILS if _promote returns original ranks
        instead of re-ranked positions.
        """
        # A has 5 chunks at original ranks 1-5; B has 1 chunk at rank 6.
        ranked = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        rtp = {1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "B"}
        result = _promote(ranked, rtp)
        assert len(result) == 2
        # A: best chunk is rowid 1 (original rank 1) → post-dedup rank 1
        a_entry = next(e for e in result if e[1] == "A")
        assert a_entry[0] == 1
        assert a_entry[2] == 1  # rowid at rank 1
        # B: best chunk is rowid 6 (original rank 6) → post-dedup rank 2 (NOT 6!)
        b_entry = next(e for e in result if e[1] == "B")
        assert b_entry[0] == 2, (
            f"Expected B to be re-ranked to post-dedup rank 2, got {b_entry[0]}. "
            "Post-dedup re-ranking (§4.1 v3.1) is not being applied."
        )
        assert b_entry[2] == 6  # rowid 6

    def test_missing_rowid_is_skipped(self) -> None:
        """Rowids absent from rowid_to_parent (stale FTS/vec entries) are skipped."""
        ranked = [(1, 99), (2, 10)]  # rowid 99 is stale
        rtp = {10: "parent_a"}
        result = _promote(ranked, rtp)
        assert len(result) == 1
        assert result[0][1] == "parent_a"

    def test_empty_ranked_returns_empty(self) -> None:
        """Empty ranked list → empty result."""
        result = _promote([], {})
        assert result == []

    def test_multiple_parents_preserve_relative_order(self) -> None:
        """Three parents interleaved: ordering must follow best original rank."""
        # A's best chunk at rank 1, C's at rank 2, B's at rank 3
        ranked = [(1, 10), (2, 20), (3, 30), (4, 11), (5, 21)]
        rtp = {10: "A", 11: "A", 20: "C", 21: "C", 30: "B"}
        result = _promote(ranked, rtp)
        assert len(result) == 3
        ranks = {e[1]: e[0] for e in result}
        assert ranks["A"] == 1
        assert ranks["C"] == 2
        assert ranks["B"] == 3


# ---------------------------------------------------------------------------
# Unit tests for _truncate_to_tokens
# ---------------------------------------------------------------------------


class TestTruncateToTokens:
    """Unit tests for the content-truncation helper."""

    def test_short_text_not_truncated(self) -> None:
        """Text within budget → returned unchanged, flag False."""
        text, truncated = _truncate_to_tokens("one two three", 10)
        assert text == "one two three"
        assert not truncated

    def test_long_text_truncated(self) -> None:
        """Text exceeding budget → truncated, flag True."""
        # 10 tokens budget → max_words = int(10/1.4) = 7
        text, truncated = _truncate_to_tokens("a b c d e f g h i j", 10)
        assert truncated
        words = text.split()
        assert len(words) == 7

    def test_exact_budget_not_truncated(self) -> None:
        """Exactly max_words words → no truncation."""
        # int(10 / 1.4) = 7 words
        _text, truncated = _truncate_to_tokens("a b c d e f g", 10)
        assert not truncated

    def test_max_tokens_1_returns_at_least_one_word(self) -> None:
        """max_tokens=1 never returns an empty string."""
        text, truncated = _truncate_to_tokens("hello world", 1)
        assert text  # non-empty
        assert truncated


# ---------------------------------------------------------------------------
# Hybrid search tests — use a real (in-memory-path) ScryDB
# ---------------------------------------------------------------------------


class TestSanitizeFTS5Query:
    """Regression for review-w2d HIGH bug — natural-language queries must not crash FTS5."""

    @pytest.mark.parametrize(
        ("raw", "expected_tokens"),
        [
            ("hello world", ['"hello"', '"world"']),
            ("How does C++ work?", ['"How"', '"does"', '"C++"', '"work?"']),
            ("file/path.py", ['"file/path.py"']),
            ("hello-world", ['"hello-world"']),
            ("foo:bar", ['"foo:bar"']),
            ("hello world!", ['"hello"', '"world!"']),
            ("a.b,c", ['"a.b,c"']),
            ('unbalanced "quote', ['"unbalanced"', '"""quote"']),  # internal " escaped
            ("", []),
            ("   ", []),
        ],
    )
    def test_sanitize_examples(self, raw: str, expected_tokens: list[str]) -> None:
        result = _sanitize_fts5_query(raw)
        assert result == " ".join(expected_tokens), (
            f"sanitize({raw!r}) returned {result!r}, expected {' '.join(expected_tokens)!r}"
        )


# ---------------------------------------------------------------------------
# Hybrid search tests — use a real (in-memory-path) ScryDB
# ---------------------------------------------------------------------------


class TestHybridSearch:
    """End-to-end tests for hybrid_search."""

    # ------------------------------------------------------------------ helpers

    def _insert_anchor_with_chunks(
        self,
        db: ScryDB,
        anchor_id: str,
        chunks_text: list[str],
        embeddings: list[bytes] | None = None,
        *,
        anchor_type: AnchorType = AnchorType.SECTION,
        content_hash: str = _HASH_A,
    ) -> Anchor:
        anchor = _make_anchor(anchor_id, anchor_type=anchor_type, content_hash=content_hash)
        db.upsert_anchor(anchor)
        sub_chunks = [
            _make_chunk(anchor_id, i, text, parent_hash=content_hash)
            for i, text in enumerate(chunks_text)
        ]
        db.replace_chunks(anchor_id, content_hash, sub_chunks, embeddings)
        return anchor

    # ------------------------------------------------------------------ tests

    def test_empty_query_returns_empty(self, db: ScryDB) -> None:
        """Empty / whitespace-only query → []."""
        emb = FixedEmbedder(_pack_vec(1, 0, 0, 0), DIMS)
        assert hybrid_search("", db=db, embedder=emb) == []
        assert hybrid_search("   ", db=db, embedder=emb) == []

    def test_empty_index_returns_empty(self, db: ScryDB) -> None:
        """No chunks in the index → []."""
        emb = FixedEmbedder(_pack_vec(1, 0, 0, 0), DIMS)
        assert hybrid_search("anything", db=db, embedder=emb) == []

    @pytest.mark.parametrize(
        "query",
        [
            "C++",  # plus sign — FTS5 PLUS operator
            "hello-world",  # dash — FTS5 NOT operator
            "How does the policy engine work?",  # natural-language sentence with ?
            "src/scry/retrieve.py",  # slashes + dots
            "foo:bar",  # colon — FTS5 column qualifier
            "hello.world",  # period
            "hello,world",  # comma
            "hello world!",  # exclamation
            'unbalanced "quote',  # internal quote needs escaping
            "(group)",  # parens
        ],
    )
    def test_natural_language_queries_do_not_crash(self, db: ScryDB, query: str) -> None:
        """Regression (review-w2d HIGH): raw user queries must NOT crash FTS5.

        Each of these queries previously raised ``sqlite3.OperationalError:
        fts5: syntax error near …`` because the user input was forwarded
        to FTS5 MATCH unsanitized. The fix routes the query through
        ``_sanitize_fts5_query`` which wraps each whitespace-split token
        in double quotes, treating punctuation as part of the term
        rather than as FTS5 syntax.
        """
        # Index any anchor so the search has something to query against.
        self._insert_anchor_with_chunks(db, "docs/a.md::a", ["body content here"])
        emb = FixedEmbedder(_pack_vec(1, 0, 0, 0), DIMS)
        # The assertion is simply that the call returns without raising.
        result = hybrid_search(query, db=db, embedder=emb)
        assert isinstance(result, list)

    def test_bm25_only_basic_ranking(self, db: ScryDB) -> None:
        """BM25-only scenario (no vec embeddings): keyword match drives ranking."""
        # Insert two anchors with no vector embeddings.
        self._insert_anchor_with_chunks(db, "docs/a.md::a", ["unique_keyword content"])
        self._insert_anchor_with_chunks(db, "docs/b.md::b", ["unrelated material"])

        emb = FixedEmbedder(_pack_vec(1, 0, 0, 0), DIMS)
        results = hybrid_search("unique_keyword", db=db, embedder=emb)

        ids = [r.parent_anchor_id for r in results]
        assert "docs/a.md::a" in ids
        # A should rank first because it contains the keyword.
        assert results[0].parent_anchor_id == "docs/a.md::a"

    def test_vec_only_basic_ranking(self, db: ScryDB) -> None:
        """Vec-only scenario: closest embedding wins, BM25 misses both."""
        # Query embedding: [1, 0, 0, 0]
        # A's chunk closer to query than B's chunk.
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        a_emb = _pack_vec(0.9, 0.0, 0.0, 0.0)  # L2 dist 0.1 → rank 1
        b_emb = _pack_vec(0.5, 0.0, 0.0, 0.0)  # L2 dist 0.5 → rank 2

        self._insert_anchor_with_chunks(db, "docs/a.md::a", ["zzz xqq bbb"], [a_emb])
        self._insert_anchor_with_chunks(db, "docs/b.md::b", ["zzz xqq bbb"], [b_emb])

        emb = FixedEmbedder(query_emb, DIMS)
        # Use a query that won't match FTS5 so only vec contributes.
        results = hybrid_search("nomatchwhatsoever_xyz", db=db, embedder=emb)

        assert results, "Expected at least one result from vec search"
        assert results[0].parent_anchor_id == "docs/a.md::a"

    def test_rrf_fusion_both_lists(self, db: ScryDB) -> None:
        """Parent appearing in both vec and BM25 scores higher than single-list parent."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        # A appears in both lists (has embedding + keyword).
        a_emb = _pack_vec(0.95, 0.0, 0.0, 0.0)
        # B appears only in vec list (no keyword match).
        b_emb = _pack_vec(0.90, 0.0, 0.0, 0.0)

        self._insert_anchor_with_chunks(db, "docs/a.md::a", ["special_term alpha"], [a_emb])
        self._insert_anchor_with_chunks(db, "docs/b.md::b", ["zzz xqq nnn"], [b_emb])

        emb = FixedEmbedder(query_emb, DIMS)
        results = hybrid_search("special_term", db=db, embedder=emb)
        assert results[0].parent_anchor_id == "docs/a.md::a"
        # A gets contributions from both lists; B only from vec.
        a_res = next(r for r in results if r.parent_anchor_id == "docs/a.md::a")
        b_res = next((r for r in results if r.parent_anchor_id == "docs/b.md::b"), None)
        assert a_res.parent_rank_in_bm25 is not None
        if b_res is not None:
            assert b_res.parent_rank_in_bm25 is None

    def test_best_chunk_per_parent_appears_exactly_once(self, db: ScryDB) -> None:
        """A parent with 5 chunks must appear exactly once in the result list."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        # Insert parent A with 5 chunks, each with a distinct embedding.
        embs = [_pack_vec(0.9 - i * 0.1, 0.0, 0.0, 0.0) for i in range(5)]
        texts = [f"chunk {i} content" for i in range(5)]
        self._insert_anchor_with_chunks(db, "docs/a.md::a", texts, embs)

        emb = FixedEmbedder(query_emb, DIMS)
        results = hybrid_search("nomatchwhatsoever_xyz", db=db, embedder=emb)

        parent_ids = [r.parent_anchor_id for r in results]
        assert parent_ids.count("docs/a.md::a") == 1

    def test_reranking_eliminates_length_bias(self, db: ScryDB) -> None:
        """§4.1 v3.1 re-ranking: single-chunk parent gets post-dedup rank 2 not 6.

        Setup:
          - Long parent A: 5 chunks at vec-ranks 1-5
          - Short parent B: 1 chunk at vec-rank 6

        After best-chunk-per-parent promotion there are 2 unique parents.
        With post-dedup re-ranking B's rank must be 2 (NOT 6).

        This test FAILS without post-dedup re-ranking, which would leave B
        at its original rank 6, artificially lowering its RRF score.
        """
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        # A's 5 chunks at vec-ranks 1-5 (distances 0.1, 0.2, …, 0.5)
        a_embs = [_pack_vec(0.9 - i * 0.1, 0.0, 0.0, 0.0) for i in range(5)]
        a_texts = [f"a chunk {i}" for i in range(5)]
        # B's 1 chunk at vec-rank 6 (distance 0.6)
        b_embs = [_pack_vec(0.4, 0.0, 0.0, 0.0)]
        b_texts = ["b chunk 0"]

        self._insert_anchor_with_chunks(db, "docs/a.md::a", a_texts, a_embs, content_hash=_HASH_A)
        self._insert_anchor_with_chunks(db, "docs/b.md::b", b_texts, b_embs, content_hash=_HASH_B)

        emb = FixedEmbedder(query_emb, DIMS)
        # Use a query that won't hit FTS5 so only vec contributes.
        results = hybrid_search("nomatchwhatsoever_xyz", db=db, embedder=emb)

        a_result = next((r for r in results if r.parent_anchor_id == "docs/a.md::a"), None)
        b_result = next((r for r in results if r.parent_anchor_id == "docs/b.md::b"), None)

        assert a_result is not None, "Long parent A must appear in results"
        assert b_result is not None, "Short parent B must appear in results"

        assert a_result.parent_rank_in_vec == 1, (
            f"A's best chunk was at rank 1; expected post-dedup rank 1, "
            f"got {a_result.parent_rank_in_vec}"
        )
        assert b_result.parent_rank_in_vec == 2, (
            f"Expected B to be re-ranked to post-dedup rank 2 (not original rank 6), "
            f"got {b_result.parent_rank_in_vec}.  "
            "Post-dedup re-ranking (§4.1 v3.1) is not being applied."
        )

    def test_rrf_math_contribution(self, db: ScryDB) -> None:
        """RRF formula: k=60, rank=1 → contribution = 1/(60+1) = 1/61."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        a_emb = _pack_vec(0.95, 0.0, 0.0, 0.0)

        self._insert_anchor_with_chunks(db, "docs/a.md::a", ["alpha beta gamma"], [a_emb])

        # Config: k=60 (default), query matches BM25 too.
        cfg = RetrievalConfig(fusion_rrf_k=60)
        emb = FixedEmbedder(query_emb, DIMS)
        results = hybrid_search("alpha", db=db, embedder=emb, config=cfg)

        assert results, "Expected at least one result"
        a_res = next(r for r in results if r.parent_anchor_id == "docs/a.md::a")

        # If A is rank 1 in vec and rank 1 in BM25, score = 1/61 + 1/61 = 2/61.
        # If A is rank 1 in vec only, score = 1/61.
        expected_min = 1.0 / 61.0
        assert a_res.score >= expected_min - 1e-9
        # Verify the formula: each contributing list adds 1/(60+rank).
        computed = 0.0
        if a_res.parent_rank_in_vec is not None:
            computed += 1.0 / (60 + a_res.parent_rank_in_vec)
        if a_res.parent_rank_in_bm25 is not None:
            computed += 1.0 / (60 + a_res.parent_rank_in_bm25)
        assert abs(a_res.score - computed) < 1e-9

    def test_top_k_respected(self, db: ScryDB) -> None:
        """Result count must not exceed top_k."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        # Insert 5 anchors.
        for i in range(5):
            emb_vec = _pack_vec(0.9 - i * 0.1, 0.0, 0.0, 0.0)
            self._insert_anchor_with_chunks(db, f"docs/{i}.md::anc", [f"content {i}"], [emb_vec])
        emb = FixedEmbedder(query_emb, DIMS)
        results = hybrid_search("nomatchwhatsoever_xyz", db=db, embedder=emb, top_k=2)
        assert len(results) <= 2

    def test_anchor_types_filter(self, db: ScryDB) -> None:
        """anchor_types filter must exclude parents of non-matching types."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        sec_emb = _pack_vec(0.95, 0.0, 0.0, 0.0)
        code_emb = _pack_vec(0.90, 0.0, 0.0, 0.0)

        # SECTION anchor A
        a = _make_anchor(
            "docs/a.md::a",
            anchor_type=AnchorType.SECTION,
            path="docs/a.md",
        )
        db.upsert_anchor(a)
        db.replace_chunks(
            a.id,
            _HASH_A,
            [_make_chunk(a.id, 0, "section text")],
            [sec_emb],
        )

        # CODE anchor B
        b = Anchor(
            id="src/b.py:func",
            type=AnchorType.CODE,
            path="src/b.py",
            content_text="def func(): pass",
            content_hash=_HASH_B,
            fingerprint_simhash=0,
            transitive_hash_status=TransitiveHashStatus.COMPLETE,
        )
        db.upsert_anchor(b)
        db.replace_chunks(
            b.id,
            _HASH_B,
            [_make_chunk(b.id, 0, "code text", parent_hash=_HASH_B)],
            [code_emb],
        )

        emb = FixedEmbedder(query_emb, DIMS)

        # Filter to SECTION only — CODE anchor must be absent.
        results = hybrid_search(
            "nomatchwhatsoever_xyz",
            db=db,
            embedder=emb,
            anchor_types=[AnchorType.SECTION],
        )
        ids = [r.parent_anchor_id for r in results]
        assert "docs/a.md::a" in ids
        assert "src/b.py:func" not in ids

        # Filter to CODE only — SECTION anchor must be absent.
        results = hybrid_search(
            "nomatchwhatsoever_xyz",
            db=db,
            embedder=emb,
            anchor_types=[AnchorType.CODE],
        )
        ids = [r.parent_anchor_id for r in results]
        assert "src/b.py:func" in ids
        assert "docs/a.md::a" not in ids

    def test_custom_rrf_k(self, db: ScryDB) -> None:
        """Custom fusion_rrf_k is used in the RRF formula."""
        query_emb = _pack_vec(1.0, 0.0, 0.0, 0.0)
        self._insert_anchor_with_chunks(
            db, "docs/a.md::a", ["alpha test"], [_pack_vec(0.9, 0, 0, 0)]
        )
        emb = FixedEmbedder(query_emb, DIMS)

        cfg_k1 = RetrievalConfig(fusion_rrf_k=1)
        cfg_k60 = RetrievalConfig(fusion_rrf_k=60)

        res_k1 = hybrid_search("alpha", db=db, embedder=emb, config=cfg_k1)
        res_k60 = hybrid_search("alpha", db=db, embedder=emb, config=cfg_k60)

        assert res_k1 and res_k60
        # With k=1 and rank=1: 1/(1+1)=0.5; with k=60: 1/61≈0.016.
        # k=1 score must be higher.
        assert res_k1[0].score > res_k60[0].score


# ---------------------------------------------------------------------------
# build_anchor_packet tests
# ---------------------------------------------------------------------------


class TestBuildAnchorPacket:
    """Tests for the §4.2 build_anchor_packet function."""

    def _setup_anchor_and_chunks(
        self,
        db: ScryDB,
        anchor_id: str = "docs/a.md::a",
        content_text: str = "hello world content",
        chunks_text: list[str] | None = None,
    ) -> tuple[Anchor, list[int]]:
        """Insert an anchor + chunks, return the anchor and chunk rowids."""
        if chunks_text is None:
            chunks_text = ["chunk zero", "chunk one"]
        anchor = _make_anchor(anchor_id, content_text=content_text)
        db.upsert_anchor(anchor)
        sub_chunks = [_make_chunk(anchor_id, i, text) for i, text in enumerate(chunks_text)]
        db.replace_chunks(anchor_id, _HASH_A, sub_chunks)
        # Retrieve the auto-assigned rowids.
        rows = db._conn.execute(
            "SELECT id FROM chunks WHERE parent_id = ? ORDER BY chunk_index",
            (anchor_id,),
        ).fetchall()
        rowids = [int(r[0]) for r in rows]
        return anchor, rowids

    def _make_result(
        self,
        parent_anchor_id: str,
        *,
        score: float = 0.5,
        vec_rowid: int | None = None,
        bm25_rowid: int | None = None,
        vec_rank: int | None = 1,
        bm25_rank: int | None = None,
    ) -> SearchResult:
        return SearchResult(
            parent_anchor_id=parent_anchor_id,
            score=score,
            best_chunk_rowid_vec=vec_rowid,
            best_chunk_rowid_bm25=bm25_rowid,
            parent_rank_in_vec=vec_rank,
            parent_rank_in_bm25=bm25_rank,
        )

    def test_roundtrip_returns_anchor_packet(self, db: ScryDB) -> None:
        """build_anchor_packet returns a well-formed AnchorPacket."""
        anchor, rowids = self._setup_anchor_and_chunks(db)
        result = self._make_result(anchor.id, vec_rowid=rowids[0])
        packet = build_anchor_packet(result, db=db)
        assert isinstance(packet, AnchorPacket)
        assert packet.anchor.id == anchor.id
        assert packet.score == result.score
        assert packet.index_state == IndexState.FRESH

    def test_index_state_propagated(self, db: ScryDB) -> None:
        """index_state argument is reflected on the returned packet."""
        anchor, rowids = self._setup_anchor_and_chunks(db)
        result = self._make_result(anchor.id, vec_rowid=rowids[0])
        packet = build_anchor_packet(result, db=db, index_state=IndexState.STALE_NO_WRITE_LOCK)
        assert packet.index_state == IndexState.STALE_NO_WRITE_LOCK

    def test_links_empty_deferred(self, db: ScryDB) -> None:
        """Links list is empty (deferred to W2i)."""
        anchor, rowids = self._setup_anchor_and_chunks(db)
        result = self._make_result(anchor.id, vec_rowid=rowids[0])
        packet = build_anchor_packet(result, db=db)
        assert packet.links == []

    def test_missing_anchor_raises_key_error(self, db: ScryDB) -> None:
        """KeyError when the parent anchor ID is not in the database."""
        result = self._make_result("docs/nonexistent.md::ghost")
        with pytest.raises(KeyError, match="nonexistent"):
            build_anchor_packet(result, db=db)

    def test_content_preview_tokens_caps_content(self, db: ScryDB) -> None:
        """content_preview_tokens truncates long content; content_truncated=True."""
        # 100 words → well above any small token budget.
        long_text = " ".join(["word"] * 100)
        anchor, rowids = self._setup_anchor_and_chunks(db, content_text=long_text)
        # max_words = int(10 / 1.4) = 7
        cfg = RetrievalConfig(content_preview_tokens=10)
        result = self._make_result(anchor.id, vec_rowid=rowids[0])
        packet = build_anchor_packet(result, db=db, config=cfg)

        assert packet.content_truncated is True
        assert len(packet.anchor.content_text.split()) == 7

    def test_short_content_not_truncated(self, db: ScryDB) -> None:
        """Content within token budget is returned unchanged; content_truncated=False."""
        anchor, rowids = self._setup_anchor_and_chunks(db, content_text="short text")
        cfg = RetrievalConfig(content_preview_tokens=500)
        result = self._make_result(anchor.id, vec_rowid=rowids[0])
        packet = build_anchor_packet(result, db=db, config=cfg)

        assert packet.content_truncated is False
        assert packet.anchor.content_text == "short text"

    def test_evidence_excerpt_prefers_vector_chunk(self, db: ScryDB) -> None:
        """evidence_excerpt uses the vector-list best chunk, not BM25 chunk."""
        anchor, rowids = self._setup_anchor_and_chunks(
            db,
            chunks_text=["vec chunk text", "bm25 chunk text"],
        )
        vec_rowid = rowids[0]
        bm25_rowid = rowids[1]
        result = self._make_result(
            anchor.id,
            vec_rowid=vec_rowid,
            bm25_rowid=bm25_rowid,
            vec_rank=1,
            bm25_rank=1,
        )
        packet = build_anchor_packet(result, db=db)
        assert packet.evidence_excerpt == "vec chunk text"

    def test_evidence_excerpt_falls_back_to_bm25(self, db: ScryDB) -> None:
        """When there is no vector hit, evidence_excerpt uses the BM25 chunk."""
        anchor, rowids = self._setup_anchor_and_chunks(
            db,
            chunks_text=["bm25 only text"],
        )
        bm25_rowid = rowids[0]
        result = self._make_result(
            anchor.id,
            vec_rowid=None,
            bm25_rowid=bm25_rowid,
            vec_rank=None,
            bm25_rank=1,
        )
        packet = build_anchor_packet(result, db=db)
        assert packet.evidence_excerpt == "bm25 only text"

    def test_evidence_excerpt_none_when_no_chunks(self, db: ScryDB) -> None:
        """No rowid available → evidence_excerpt is None."""
        anchor, _ = self._setup_anchor_and_chunks(db)
        result = self._make_result(
            anchor.id,
            vec_rowid=None,
            bm25_rowid=None,
            vec_rank=None,
            bm25_rank=None,
        )
        packet = build_anchor_packet(result, db=db)
        assert packet.evidence_excerpt is None


# uat-r5-5 pr-d noise
