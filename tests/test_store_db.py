"""Tests for scry.store.db — SQLite persistence layer.

Covers all tables, FTS5 + sqlite-vec virtual tables, advisory write lock,
read-only mode, and public API contracts per DESIGN.md §3.4, §7.1, §7.2.1,
§7.3.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from scry.models import Anchor, AnchorType, IndexMetadata, SubChunk, TransitiveHashStatus
from scry.store.db import IntegrityError, LockTimeout, SchemaError, ScryDB

# ─── Fixtures and helpers ──────────────────────────────────────────────────────

DIMS = 4  # Small, deterministic dimensionality for tests.

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64


def _make_anchor(
    anchor_id: str = "docs/test.md::intro",
    *,
    anchor_type: AnchorType = AnchorType.SECTION,
    path: str = "docs/test.md",
    content_text: str = "Hello world",
    content_hash: str = _HASH_A,
    fingerprint_simhash: int = 0xDEADBEEF,
    heading_path: list[str] | None = None,
    symbol_name: str | None = None,
    transitive_hash_status: TransitiveHashStatus | None = None,
) -> Anchor:
    """Build a minimal valid ``Anchor`` for testing."""
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        content_text=content_text,
        content_hash=content_hash,
        fingerprint_simhash=fingerprint_simhash,
        heading_path=heading_path,
        symbol_name=symbol_name,
        transitive_hash_status=transitive_hash_status,
    )


def _make_code_anchor(anchor_id: str = "src/app.py:main") -> Anchor:
    """Build a minimal valid ``CODE`` anchor for testing."""
    return _make_anchor(
        anchor_id=anchor_id,
        anchor_type=AnchorType.CODE,
        path="src/app.py",
        transitive_hash_status=TransitiveHashStatus.COMPLETE,
    )


def _float_vec(*values: float) -> bytes:
    """Pack floats into a float32 blob (little-endian) for sqlite-vec."""
    return struct.pack(f"{len(values)}f", *values)


def _make_meta(embedding_dimensions: int = DIMS) -> IndexMetadata:
    """Build a minimal valid ``IndexMetadata`` for testing."""
    return IndexMetadata(
        indexed_git_head="abc123" * 6 + "abcd",
        indexed_git_tree_hash=None,
        indexed_branch="main",
        indexed_file_manifest={"docs/test.md": _HASH_A},
        config_hash=_HASH_B,
        embedding_provider="local",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dimensions=embedding_dimensions,
        tokenizer_version=None,
    )


@pytest.fixture
def db(tmp_repo: Path) -> ScryDB:
    """A fresh ``ScryDB`` with schema initialised at DIMS dimensions."""
    d = ScryDB(tmp_repo)
    d.init_schema(embedding_dimensions=DIMS)
    return d


# ─── Schema creation ──────────────────────────────────────────────────────────


def test_init_schema_creates_all_tables(tmp_repo: Path) -> None:
    """init_schema must create all expected tables and virtual tables."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
        tables = {
            row[0]
            for row in d._conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
            ).fetchall()
        }
        for expected in ("index_metadata", "anchors", "chunks", "chunks_fts", "chunks_vec"):
            assert expected in tables, f"Missing table: {expected}"


def test_init_schema_creates_indexes(tmp_repo: Path) -> None:
    """init_schema must create the declared indexes."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
        indexes = {
            row[0]
            for row in d._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        for expected in ("idx_anchors_path", "idx_anchors_type", "idx_chunks_parent_id"):
            assert expected in indexes, f"Missing index: {expected}"


def test_init_schema_creates_fts5_triggers(tmp_repo: Path) -> None:
    """init_schema must create all four FTS5 sync triggers."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
        triggers = {
            row[0]
            for row in d._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        for expected in ("chunks_ai", "chunks_bd", "chunks_bu", "chunks_au"):
            assert expected in triggers, f"Missing trigger: {expected}"


def test_init_schema_idempotent(tmp_repo: Path) -> None:
    """Calling init_schema twice at the same dimensions must not raise."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
        d.init_schema(embedding_dimensions=DIMS)  # second call — must be a no-op


def test_init_schema_idempotent_with_data(db: ScryDB, tmp_repo: Path) -> None:
    """Idempotency must be preserved even when anchors and chunks exist."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(
        parent_id=anchor.id,
        chunk_index=0,
        text="some text",
        parent_content_hash=_HASH_A,
    )
    db.replace_chunks(anchor.id, _HASH_A, [chunk])

    # Re-init at the SAME dimension — must not wipe existing data.
    db.init_schema(embedding_dimensions=DIMS)
    assert db.get_anchor(anchor.id) is not None
    assert len(db.list_chunks(anchor.id)) == 1


# ─── Vec virtual table dimension management ───────────────────────────────────


def test_init_schema_recreates_vec_on_dim_change(db: ScryDB) -> None:
    """init_schema with a different dimension must drop+recreate chunks_vec."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(
        parent_id=anchor.id,
        chunk_index=0,
        text="some text",
        parent_content_hash=_HASH_A,
    )
    db.replace_chunks(anchor.id, _HASH_A, [chunk])

    # Change dimensions — vec table must be recreated.
    new_dims = 8
    db.init_schema(embedding_dimensions=new_dims)

    # Anchor row must be preserved.
    assert db.get_anchor(anchor.id) is not None
    # Chunk must still exist (only embeddings are cleared, not rows).
    chunks = db.list_chunks(anchor.id)
    assert len(chunks) == 1


def test_recreate_vector_table_changes_dim(db: ScryDB) -> None:
    """recreate_vector_table must create a new table with the requested dims."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)

    db.recreate_vector_table(8)

    # Anchor row must be preserved.
    assert db.get_anchor(anchor.id) is not None
    # New vec table must accept 8-float vectors.
    v8 = _float_vec(1, 0, 0, 0, 0, 0, 0, 0)
    chunk = SubChunk(parent_id=anchor.id, chunk_index=0, text="t", parent_content_hash=_HASH_A)
    db.replace_chunks(anchor.id, _HASH_A, [chunk], embeddings=[v8])
    results = db.query_vector(v8, top_k=1)
    assert len(results) == 1


def test_recreate_vector_table_nulls_chunk_embeddings(db: ScryDB) -> None:
    """recreate_vector_table must NULL out the embedding column in chunks."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(parent_id=anchor.id, chunk_index=0, text="t", parent_content_hash=_HASH_A)
    v = _float_vec(1, 0, 0, 0)
    db.replace_chunks(anchor.id, _HASH_A, [chunk], embeddings=[v])

    db.recreate_vector_table(4)

    row = db._conn.execute(
        "SELECT embedding FROM chunks WHERE parent_id = ?", (anchor.id,)
    ).fetchone()
    assert row is not None
    assert row[0] is None


# ─── IndexMetadata round-trip ─────────────────────────────────────────────────


def test_read_index_metadata_empty(db: ScryDB) -> None:
    """read_index_metadata returns None on a fresh DB."""
    assert db.read_index_metadata() is None


def test_write_read_index_metadata_round_trip(db: ScryDB) -> None:
    """write then read must produce an equivalent IndexMetadata."""
    meta = _make_meta()
    db.write_index_metadata(meta)
    got = db.read_index_metadata()
    assert got is not None
    assert got.indexed_git_head == meta.indexed_git_head
    assert got.indexed_branch == meta.indexed_branch
    assert got.indexed_file_manifest == meta.indexed_file_manifest
    assert got.config_hash == meta.config_hash
    assert got.embedding_provider == meta.embedding_provider
    assert got.embedding_model == meta.embedding_model
    assert got.embedding_dimensions == meta.embedding_dimensions
    assert got.tokenizer_version == meta.tokenizer_version


def test_write_index_metadata_upsert(db: ScryDB) -> None:
    """Writing twice must update the singleton row (not create a second)."""
    db.write_index_metadata(_make_meta())
    meta2 = _make_meta(embedding_dimensions=768)
    db.write_index_metadata(meta2)
    got = db.read_index_metadata()
    assert got is not None
    assert got.embedding_dimensions == 768
    # Confirm only one row exists.
    count = db._conn.execute("SELECT COUNT(*) FROM index_metadata").fetchone()[0]
    assert count == 1


# ─── Anchor CRUD ──────────────────────────────────────────────────────────────


def test_upsert_anchor_get_anchor_round_trip(db: ScryDB) -> None:
    """upsert_anchor → get_anchor must return an equivalent Anchor."""
    anchor = _make_anchor(heading_path=["Intro", "Overview"], symbol_name=None)
    db.upsert_anchor(anchor)
    got = db.get_anchor(anchor.id)
    assert got is not None
    assert got.id == anchor.id
    assert got.type == anchor.type
    assert got.path == anchor.path
    assert got.heading_path == anchor.heading_path
    assert got.content_text == anchor.content_text
    assert got.content_hash == anchor.content_hash
    assert got.fingerprint_simhash == anchor.fingerprint_simhash


def test_upsert_anchor_code_type_with_status(db: ScryDB) -> None:
    """CODE anchor with transitive_hash_status must round-trip correctly."""
    anchor = _make_code_anchor()
    db.upsert_anchor(anchor)
    got = db.get_anchor(anchor.id)
    assert got is not None
    assert str(got.transitive_hash_status) == str(TransitiveHashStatus.COMPLETE)


def test_upsert_anchor_preserves_created_at(db: ScryDB) -> None:
    """Upserting with changed content must preserve the original created_at."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    created_at_orig = db._conn.execute(
        "SELECT created_at FROM anchors WHERE id = ?", (anchor.id,)
    ).fetchone()[0]

    time.sleep(0.01)  # Ensure a different timestamp is possible.

    updated = _make_anchor(content_text="new text", content_hash=_HASH_B)
    db.upsert_anchor(updated)
    created_at_after = db._conn.execute(
        "SELECT created_at FROM anchors WHERE id = ?", (anchor.id,)
    ).fetchone()[0]
    assert created_at_orig == created_at_after


def test_upsert_anchor_nulls_embedding_on_hash_change(db: ScryDB) -> None:
    """Upserting with a new content_hash must NULL the stored overview_embedding."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    # Simulate the indexer filling in the embedding.
    db._conn.execute(
        "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
        (b"\x00" * 16, anchor.id),
    )
    db._conn.commit()

    # Upsert with a different hash.
    updated = _make_anchor(content_text="changed text", content_hash=_HASH_B)
    db.upsert_anchor(updated)
    row = db._conn.execute(
        "SELECT overview_embedding FROM anchors WHERE id = ?", (anchor.id,)
    ).fetchone()
    assert row is not None
    assert row[0] is None


def test_upsert_anchor_preserves_embedding_on_same_hash(db: ScryDB) -> None:
    """Upserting with the same content_hash must preserve overview_embedding."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    fake_embedding = b"\x01" * 16
    db._conn.execute(
        "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
        (fake_embedding, anchor.id),
    )
    db._conn.commit()

    # Upsert with the SAME hash — embedding should survive.
    db.upsert_anchor(_make_anchor())
    row = db._conn.execute(
        "SELECT overview_embedding FROM anchors WHERE id = ?", (anchor.id,)
    ).fetchone()
    assert row is not None
    assert row[0] == fake_embedding


def test_get_anchor_missing(db: ScryDB) -> None:
    """get_anchor returns None for unknown IDs."""
    assert db.get_anchor("does/not::exist") is None


def test_list_anchors_all(db: ScryDB) -> None:
    """list_anchors with no filters returns all anchors."""
    a1 = _make_anchor("docs/a.md::s1", path="docs/a.md")
    a2 = _make_anchor("docs/b.md::s1", path="docs/b.md")
    db.upsert_anchors([a1, a2])
    result = db.list_anchors()
    ids = {a.id for a in result}
    assert ids == {a1.id, a2.id}


def test_list_anchors_by_type(db: ScryDB) -> None:
    """list_anchors(anchor_type=...) filters correctly."""
    section = _make_anchor("docs/s.md::s", anchor_type=AnchorType.SECTION, path="docs/s.md")
    code = _make_code_anchor()
    db.upsert_anchors([section, code])

    sections = db.list_anchors(anchor_type=AnchorType.SECTION)
    codes = db.list_anchors(anchor_type=AnchorType.CODE)

    assert all(str(a.type) == "section" for a in sections)
    assert all(str(a.type) == "code" for a in codes)
    assert {a.id for a in sections} == {section.id}
    assert {a.id for a in codes} == {code.id}


def test_list_anchors_by_path(db: ScryDB) -> None:
    """list_anchors(path=...) returns only anchors for that file."""
    a1 = _make_anchor("docs/a.md::s1", path="docs/a.md")
    a2 = _make_anchor("docs/b.md::s1", path="docs/b.md")
    db.upsert_anchors([a1, a2])

    result = db.list_anchors(path="docs/a.md")
    assert {r.id for r in result} == {a1.id}


def test_upsert_anchors_batch_atomicity(db: ScryDB) -> None:
    """A failure mid-batch must roll back the entire batch."""

    def _bad_iter() -> Iterable[Anchor]:
        yield _make_anchor("docs/ok1.md::s1", path="docs/ok1.md")
        yield _make_anchor("docs/ok2.md::s1", path="docs/ok2.md")
        raise RuntimeError("Simulated failure mid-batch")

    with pytest.raises(RuntimeError, match="Simulated failure"):
        db.upsert_anchors(_bad_iter())

    # Nothing should have been committed.
    assert db.list_anchors() == []


def test_delete_anchor_removes_row(db: ScryDB) -> None:
    """delete_anchor removes the anchor row."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    db.delete_anchor(anchor.id)
    assert db.get_anchor(anchor.id) is None


def test_delete_anchor_cascades_to_chunks(db: ScryDB) -> None:
    """delete_anchor cascades via FK to remove all child chunks."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunks = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"chunk {i}", parent_content_hash=_HASH_A)
        for i in range(3)
    ]
    db.replace_chunks(anchor.id, _HASH_A, chunks)

    db.delete_anchor(anchor.id)

    remaining = db.list_chunks(anchor.id)
    assert remaining == []
    # FTS5 entries must also be gone.
    fts_count = db._conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE text MATCH 'chunk'"
    ).fetchone()[0]
    assert fts_count == 0


def test_delete_anchor_removes_chunks_vec_entries(db: ScryDB) -> None:
    """delete_anchor must purge the associated chunks_vec rows."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    v = _float_vec(1, 0, 0, 0)
    chunk = SubChunk(parent_id=anchor.id, chunk_index=0, text="t", parent_content_hash=_HASH_A)
    db.replace_chunks(anchor.id, _HASH_A, [chunk], embeddings=[v])

    db.delete_anchor(anchor.id)

    vec_count = db._conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    assert vec_count == 0


# ─── Sub-chunk CRUD ───────────────────────────────────────────────────────────


def test_replace_chunks_inserts_new(db: ScryDB) -> None:
    """replace_chunks must insert all provided chunks."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunks = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"text {i}", parent_content_hash=_HASH_A)
        for i in range(3)
    ]
    db.replace_chunks(anchor.id, _HASH_A, chunks)
    result = db.list_chunks(anchor.id)
    assert len(result) == 3
    assert [c.chunk_index for c in result] == [0, 1, 2]


def test_replace_chunks_replaces_old(db: ScryDB) -> None:
    """replace_chunks must atomically delete old chunks and insert new ones."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    old = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"old {i}", parent_content_hash=_HASH_A)
        for i in range(5)
    ]
    db.replace_chunks(anchor.id, _HASH_A, old)

    new = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"new {i}", parent_content_hash=_HASH_B)
        for i in range(2)
    ]
    db.replace_chunks(anchor.id, _HASH_B, new)

    result = db.list_chunks(anchor.id)
    assert len(result) == 2
    assert all(c.parent_content_hash == _HASH_B for c in result)
    assert all(c.text.startswith("new") for c in result)


def test_replace_chunks_stores_parent_content_hash(db: ScryDB) -> None:
    """Each chunk row must carry the parent_content_hash passed to replace_chunks."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(parent_id=anchor.id, chunk_index=0, text="t", parent_content_hash=_HASH_A)
    db.replace_chunks(anchor.id, _HASH_A, [chunk])
    result = db.list_chunks(anchor.id)
    assert result[0].parent_content_hash == _HASH_A


def test_replace_chunks_stale_hash_filter(db: ScryDB) -> None:
    """After re-indexing with a new hash, old chunks are gone (§7.3 inv 4)."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    old_chunk = SubChunk(
        parent_id=anchor.id, chunk_index=0, text="old text", parent_content_hash=_HASH_A
    )
    db.replace_chunks(anchor.id, _HASH_A, [old_chunk])

    # Now re-index with new hash.
    new_chunk = SubChunk(
        parent_id=anchor.id, chunk_index=0, text="new text", parent_content_hash=_HASH_B
    )
    db.replace_chunks(anchor.id, _HASH_B, [new_chunk])

    chunks = db.list_chunks(anchor.id)
    assert len(chunks) == 1
    assert chunks[0].parent_content_hash == _HASH_B
    assert chunks[0].text == "new text"


def test_replace_chunks_atomicity(db: ScryDB) -> None:
    """replace_chunks is atomic: exception during insert must leave old chunks intact."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    old = [
        SubChunk(parent_id=anchor.id, chunk_index=0, text="original", parent_content_hash=_HASH_A)
    ]
    db.replace_chunks(anchor.id, _HASH_A, old)

    # Trigger a DB constraint violation by inserting a chunk with a parent that
    # doesn't exist — this causes the FK check to fire inside replace_chunks.
    bad = [
        SubChunk(
            parent_id="nonexistent::anchor",
            chunk_index=0,
            text="bad",
            parent_content_hash=_HASH_A,
        )
    ]
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_chunks("nonexistent::anchor", _HASH_A, bad)

    # Original anchor's chunks must be untouched.
    result = db.list_chunks(anchor.id)
    assert len(result) == 1
    assert result[0].text == "original"


def test_reindex_anchor_with_chunks_atomic(db: ScryDB) -> None:
    """Regression (review-w2a HIGH): per-parent reindex is a SINGLE transaction.

    Per §7.3 invariant 2 the parent's content_hash and its chunks'
    parent_content_hash MUST advance together.  This test simulates a
    partial failure during the chunk-insert phase and confirms the
    parent's content_hash also rolls back to the old value (so no
    stale-orphan window can be observed mid-reindex).
    """
    anchor_v1 = _make_anchor(content_hash=_HASH_A)
    db.reindex_anchor_with_chunks(
        anchor_v1,
        [
            SubChunk(
                parent_id=anchor_v1.id,
                chunk_index=0,
                text="v1 chunk",
                parent_content_hash=_HASH_A,
            )
        ],
    )

    anchor_v2 = _make_anchor(content_hash=_HASH_B)
    bad_chunk = SubChunk(
        parent_id="some::other::parent",  # FK violation → mid-transaction error
        chunk_index=0,
        text="will not commit",
        parent_content_hash=_HASH_B,
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.reindex_anchor_with_chunks(anchor_v2, [bad_chunk])

    # The parent's content_hash MUST still be the v1 value — neither the
    # parent upsert nor the chunk replacement persisted.
    fetched = db.get_anchor(anchor_v1.id)
    assert fetched is not None
    assert fetched.content_hash == _HASH_A, (
        "stale-orphan window: parent advanced to v2 but chunks rolled back"
    )
    chunks = db.list_chunks(anchor_v1.id)
    assert len(chunks) == 1
    assert chunks[0].parent_content_hash == _HASH_A
    assert chunks[0].text == "v1 chunk"


def test_transaction_context_manager_rolls_back_on_exception(db: ScryDB) -> None:
    """db.transaction() commits on clean exit, rolls back on exception."""
    anchor = _make_anchor(content_hash=_HASH_A)
    db.upsert_anchor(anchor)
    with pytest.raises(RuntimeError, match="boom"), db.transaction():
        db.upsert_anchor(_make_anchor(anchor_id="boom::other", content_hash=_HASH_B))
        raise RuntimeError("boom")
    # The bogus anchor must NOT be persisted.
    assert db.get_anchor("boom::other") is None
    # The original anchor is still present.
    assert db.get_anchor(anchor.id) is not None


def test_transaction_context_manager_nesting_forbidden(db: ScryDB) -> None:
    """Nested transactions raise IntegrityError (sqlite3 has no save-points by default)."""
    with (
        db.transaction(),
        pytest.raises(IntegrityError, match="nested transactions"),
        db.transaction(),
    ):
        pass


def test_replace_chunks_updates_fts5(db: ScryDB) -> None:
    """After replace_chunks, query_bm25 must find the new chunk text."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)

    old = [
        SubChunk(
            parent_id=anchor.id, chunk_index=0, text="apple banana", parent_content_hash=_HASH_A
        )
    ]
    db.replace_chunks(anchor.id, _HASH_A, old)

    # FTS5 should find old text.
    hits = db.query_bm25("apple", top_k=5)
    assert len(hits) == 1

    # Replace with entirely different text.
    new = [
        SubChunk(
            parent_id=anchor.id, chunk_index=0, text="zephyr quantum", parent_content_hash=_HASH_B
        )
    ]
    db.replace_chunks(anchor.id, _HASH_B, new)

    # Old text must be gone.
    old_hits = db.query_bm25("apple", top_k=5)
    assert len(old_hits) == 0

    # New text must be findable.
    new_hits = db.query_bm25("zephyr", top_k=5)
    assert len(new_hits) == 1


def test_replace_chunks_with_embeddings_populates_vec(db: ScryDB) -> None:
    """replace_chunks with embeddings must insert rows into chunks_vec."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    v = _float_vec(1, 0, 0, 0)
    chunk = SubChunk(parent_id=anchor.id, chunk_index=0, text="t", parent_content_hash=_HASH_A)
    db.replace_chunks(anchor.id, _HASH_A, [chunk], embeddings=[v])

    vec_count = db._conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    assert vec_count == 1


def test_replace_chunks_embedding_count_mismatch(db: ScryDB) -> None:
    """replace_chunks must raise ValueError when embedding count != chunk count."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunks = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"t{i}", parent_content_hash=_HASH_A)
        for i in range(3)
    ]
    embs = [_float_vec(1, 0, 0, 0), _float_vec(0, 1, 0, 0)]  # Only 2 for 3 chunks.

    with pytest.raises(ValueError, match="embeddings count"):
        db.replace_chunks(anchor.id, _HASH_A, chunks, embeddings=embs)


# ─── Vector retrieval ─────────────────────────────────────────────────────────


def test_query_vector_returns_correct_ordering(db: ScryDB) -> None:
    """query_vector must return results ordered by ascending distance."""
    a1 = _make_anchor("docs/a.md::s1", path="docs/a.md", content_hash=_HASH_A)
    a2 = _make_anchor("docs/b.md::s1", path="docs/b.md", content_hash=_HASH_B)
    db.upsert_anchors([a1, a2])

    # v1 ≈ [1,0,0,0]; v2 ≈ [0,1,0,0]; query ≈ [0.9,0.1,0,0] — closer to v1.
    v1 = _float_vec(1.0, 0.0, 0.0, 0.0)
    v2 = _float_vec(0.0, 1.0, 0.0, 0.0)
    query = _float_vec(0.9, 0.1, 0.0, 0.0)

    c1 = SubChunk(parent_id=a1.id, chunk_index=0, text="near", parent_content_hash=_HASH_A)
    c2 = SubChunk(parent_id=a2.id, chunk_index=0, text="far", parent_content_hash=_HASH_B)
    db.replace_chunks(a1.id, _HASH_A, [c1], embeddings=[v1])
    db.replace_chunks(a2.id, _HASH_B, [c2], embeddings=[v2])

    results = db.query_vector(query, top_k=2)
    assert len(results) == 2
    row0_id, row0_dist = results[0]
    _row1_id, row1_dist = results[1]
    # First result must be closer (smaller distance).
    assert row0_dist < row1_dist
    # The near chunk's rowid must come first.
    near_rowid = db._conn.execute("SELECT id FROM chunks WHERE parent_id = ?", (a1.id,)).fetchone()[
        0
    ]
    assert row0_id == near_rowid


def test_query_vector_top_k(db: ScryDB) -> None:
    """query_vector must respect the top_k limit."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    vecs = [_float_vec(float(i), 0, 0, 0) for i in range(5)]
    chunks = [
        SubChunk(parent_id=anchor.id, chunk_index=i, text=f"t{i}", parent_content_hash=_HASH_A)
        for i in range(5)
    ]
    db.replace_chunks(anchor.id, _HASH_A, chunks, embeddings=vecs)

    results = db.query_vector(_float_vec(1, 0, 0, 0), top_k=3)
    assert len(results) == 3


# ─── BM25 retrieval ───────────────────────────────────────────────────────────


def test_query_bm25_finds_matching_text(db: ScryDB) -> None:
    """query_bm25 must return chunks containing the queried keyword."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(
        parent_id=anchor.id,
        chunk_index=0,
        text="scry is a hybrid retrieval tool",
        parent_content_hash=_HASH_A,
    )
    db.replace_chunks(anchor.id, _HASH_A, [chunk])

    hits = db.query_bm25("retrieval", top_k=5)
    assert len(hits) == 1
    rowid, score = hits[0]
    assert isinstance(rowid, int)
    assert isinstance(score, float)


def test_query_bm25_no_match(db: ScryDB) -> None:
    """query_bm25 must return an empty list when no chunks match."""
    anchor = _make_anchor()
    db.upsert_anchor(anchor)
    chunk = SubChunk(
        parent_id=anchor.id, chunk_index=0, text="hello world", parent_content_hash=_HASH_A
    )
    db.replace_chunks(anchor.id, _HASH_A, [chunk])

    hits = db.query_bm25("zzznomatch", top_k=5)
    assert hits == []


def test_query_bm25_top_k(db: ScryDB) -> None:
    """query_bm25 must respect the top_k limit."""
    for i in range(5):
        anchor = _make_anchor(
            f"docs/a{i}.md::s", path=f"docs/a{i}.md", content_hash="sha256:" + hex(i)[2:].zfill(64)
        )
        db.upsert_anchor(anchor)
        chunk = SubChunk(
            parent_id=anchor.id,
            chunk_index=0,
            text="common keyword here",
            parent_content_hash=anchor.content_hash,
        )
        db.replace_chunks(anchor.id, anchor.content_hash, [chunk])

    hits = db.query_bm25("keyword", top_k=3)
    assert len(hits) == 3


# ─── Advisory write lock ──────────────────────────────────────────────────────


@pytest.mark.unix_only
def test_write_lock_acquired_and_released(tmp_repo: Path) -> None:
    """WriteLock context manager acquires and releases on Unix."""
    db = ScryDB(tmp_repo)
    db.init_schema(embedding_dimensions=DIMS)
    with db.acquire_write_lock(timeout_seconds=1.0):
        assert (tmp_repo / ".scry" / "vectors.db.lock").exists()
    # After exit, another acquisition must succeed.
    with db.acquire_write_lock(timeout_seconds=1.0):
        pass


@pytest.mark.unix_only
def test_write_lock_blocks_second_writer(tmp_repo: Path) -> None:
    """A second acquire_write_lock must time out while the first is held."""
    db1 = ScryDB(tmp_repo)
    db1.init_schema(embedding_dimensions=DIMS)
    db2 = ScryDB(tmp_repo)

    locked_event = threading.Event()
    release_event = threading.Event()

    def _hold_lock() -> None:
        with db1.acquire_write_lock(timeout_seconds=5.0):
            locked_event.set()
            release_event.wait(timeout=5.0)

    t = threading.Thread(target=_hold_lock, daemon=True)
    t.start()
    locked_event.wait(timeout=5.0)

    try:
        with pytest.raises(LockTimeout), db2.acquire_write_lock(timeout_seconds=0.15):
            pass
    finally:
        release_event.set()
        t.join(timeout=5.0)


@pytest.mark.unix_only
def test_write_lock_released_on_context_exit(tmp_repo: Path) -> None:
    """After the context exits, a second lock acquisition must succeed."""
    db = ScryDB(tmp_repo)
    db.init_schema(embedding_dimensions=DIMS)

    with db.acquire_write_lock(timeout_seconds=1.0):
        pass  # Lock released here.

    # Must be acquirable again immediately.
    with db.acquire_write_lock(timeout_seconds=1.0):
        pass


@pytest.mark.windows_only
def test_write_lock_acquired_and_released_windows(tmp_repo: Path) -> None:
    """WriteLock context manager acquires and releases on Windows."""
    db = ScryDB(tmp_repo)
    db.init_schema(embedding_dimensions=DIMS)
    with db.acquire_write_lock(timeout_seconds=1.0):
        assert (tmp_repo / ".scry" / "vectors.db.lock").exists()
    with db.acquire_write_lock(timeout_seconds=1.0):
        pass


@pytest.mark.windows_only
def test_write_lock_blocks_second_writer_windows(tmp_repo: Path) -> None:
    """A second acquire_write_lock must time out while the first is held (Windows)."""
    db1 = ScryDB(tmp_repo)
    db1.init_schema(embedding_dimensions=DIMS)
    db2 = ScryDB(tmp_repo)

    locked_event = threading.Event()
    release_event = threading.Event()

    def _hold_lock() -> None:
        with db1.acquire_write_lock(timeout_seconds=5.0):
            locked_event.set()
            release_event.wait(timeout=5.0)

    t = threading.Thread(target=_hold_lock, daemon=True)
    t.start()
    locked_event.wait(timeout=5.0)

    try:
        with pytest.raises(LockTimeout), db2.acquire_write_lock(timeout_seconds=0.15):
            pass
    finally:
        release_event.set()
        t.join(timeout=5.0)


# ─── Read-only mode ───────────────────────────────────────────────────────────


def test_read_only_mode_rejects_write_metadata(tmp_repo: Path) -> None:
    """write_index_metadata must raise IntegrityError in read-only mode."""
    # Create and populate the DB in write mode first.
    with ScryDB(tmp_repo) as rw:
        rw.init_schema(embedding_dimensions=DIMS)

    with ScryDB(tmp_repo, read_only=True) as ro, pytest.raises(IntegrityError):
        ro.write_index_metadata(_make_meta())


def test_read_only_mode_rejects_upsert_anchor(tmp_repo: Path) -> None:
    """upsert_anchor must raise IntegrityError in read-only mode."""
    with ScryDB(tmp_repo) as rw:
        rw.init_schema(embedding_dimensions=DIMS)

    with ScryDB(tmp_repo, read_only=True) as ro, pytest.raises(IntegrityError):
        ro.upsert_anchor(_make_anchor())


def test_read_only_mode_rejects_acquire_write_lock(tmp_repo: Path) -> None:
    """acquire_write_lock must raise IntegrityError in read-only mode."""
    with ScryDB(tmp_repo) as rw:
        rw.init_schema(embedding_dimensions=DIMS)

    with ScryDB(tmp_repo, read_only=True) as ro, pytest.raises(IntegrityError):
        ro.acquire_write_lock()


def test_read_only_mode_allows_reads(tmp_repo: Path) -> None:
    """A read-only connection must be able to read previously written data."""
    anchor = _make_anchor()
    with ScryDB(tmp_repo) as rw:
        rw.init_schema(embedding_dimensions=DIMS)
        rw.upsert_anchor(anchor)
        rw.write_index_metadata(_make_meta())

    with ScryDB(tmp_repo, read_only=True) as ro:
        got = ro.get_anchor(anchor.id)
        assert got is not None
        assert got.id == anchor.id
        meta = ro.read_index_metadata()
        assert meta is not None
        assert meta.embedding_dimensions == DIMS


# ─── Error handling ───────────────────────────────────────────────────────────


def test_schema_error_on_unparseable_vec_sql(tmp_repo: Path) -> None:
    """init_schema raises SchemaError when chunks_vec SQL lacks float[N]."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
        # Corrupt the sqlite_master shadow entry to remove the dimension info.
        # We can't directly modify sqlite_master, but we can test the SchemaError
        # branch by monkeypatching the regex to never match.
        import scry.store.db as db_mod

        original_re = db_mod._RE_VEC_DIM
        try:
            import re

            db_mod._RE_VEC_DIM = re.compile(r"NEVERMATCH")
            with pytest.raises(SchemaError, match="Cannot determine"):
                d.init_schema(embedding_dimensions=DIMS + 1)
        finally:
            db_mod._RE_VEC_DIM = original_re


# ─── Context manager protocol ─────────────────────────────────────────────────


def test_scrydb_context_manager(tmp_repo: Path) -> None:
    """ScryDB used as a context manager must close the connection on exit."""
    with ScryDB(tmp_repo) as d:
        d.init_schema(embedding_dimensions=DIMS)
    # After exit the connection is closed; further operations should raise.
    with pytest.raises(sqlite3.ProgrammingError):
        d.get_anchor("any::id")

# uat-r5-5 pr-d noise
