"""SQLite persistence layer for scry (DESIGN.md §3.4, §7.1, §7.2.1, §7.3).

Provides ``ScryDB`` — a connection wrapper around ``.scry/vectors.db`` that
owns the sqlite-vec extension load, WAL mode, schema creation, advisory
write-lock acquisition, and CRUD for anchors / chunks / index_metadata.

Schema tables:
    - ``index_metadata``  §7.1   singleton row tracking what produced the index
    - ``anchors``         §3.1   one row per anchor; ``content_text`` persisted
                                  for ``--reembed`` without source re-reads (§7.2.1)
    - ``chunks``          §3.4   sub-chunks of each anchor for retrieval
    - ``chunks_fts``      §11    FTS5 virtual table (BM25 keyword search)
    - ``chunks_vec``      §7.2   sqlite-vec vec0 virtual table (ANN vector search)
      Dimension is config-dependent; drop+recreate on dimension mismatch.

Concurrency model (DESIGN.md §7.3):
    Single writer (the leader, §10) holds the advisory write lock at
    ``.scry/vectors.db.lock`` via ``fcntl.flock`` (Unix) or
    ``msvcrt.locking`` (Windows) for the duration of any write transaction.
    Multiple concurrent readers open the DB in read-only mode and wrap each
    tool call in a fresh short read transaction.  WAL mode ensures readers
    never block writers and vice-versa.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Generator, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlite_vec

from scry.models import Anchor, AnchorType, IndexMetadata, SubChunk

# ─── Exceptions ───────────────────────────────────────────────────────────────


class SchemaError(Exception):
    """Raised for schema-level issues (e.g., unexpected table shapes)."""


class LockTimeout(Exception):
    """Raised when the advisory write lock cannot be acquired within the timeout."""


class IntegrityError(Exception):
    """Raised for integrity or access violations (e.g., writes in read-only mode)."""


# ─── Advisory write lock ──────────────────────────────────────────────────────


class WriteLock:
    """OS-level advisory write lock on ``.scry/vectors.db.lock``.

    The lock is OS-level, so it is automatically released on process death
    (kernel reclaims open file descriptors), satisfying the crash-safety
    requirement of DESIGN.md §7.3 invariant 6.

    Usage::

        with db.acquire_write_lock():
            db.upsert_anchors(anchors)

    Args:
        lock_path: Path to the lock file (created if absent).
        timeout_seconds: Maximum wall-clock seconds to wait for the lock.

    Raises:
        LockTimeout: If the lock cannot be acquired within ``timeout_seconds``.
    """

    def __init__(self, lock_path: Path, *, timeout_seconds: float) -> None:
        self._lock_path = lock_path
        self._timeout = timeout_seconds
        self._fd: int | None = None

    def __enter__(self) -> WriteLock:
        """Acquire the lock; raise ``LockTimeout`` on failure."""
        self._acquire()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Release the lock unconditionally."""
        self._release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _acquire(self) -> None:
        """Open the lock file and acquire an exclusive OS-level lock."""
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + self._timeout
        try:
            if sys.platform == "win32":
                _acquire_win(fd, deadline)
            else:
                _acquire_unix(fd, deadline)
        except LockTimeout:
            os.close(fd)
            raise
        self._fd = fd

    def _release(self) -> None:
        """Release and close the lock file descriptor."""
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if sys.platform == "win32":
                import msvcrt

                with contextlib.suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _acquire_unix(fd: int, deadline: float) -> None:
    """Poll ``fcntl.flock`` with LOCK_NB until the deadline."""
    if sys.platform == "win32":
        return  # pragma: no cover
    import fcntl

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise LockTimeout("Could not acquire write lock within the timeout") from None
            time.sleep(0.01)


def _acquire_win(fd: int, deadline: float) -> None:
    """Poll ``msvcrt.locking`` with LK_NBLCK until the deadline."""
    if sys.platform != "win32":
        return  # pragma: no cover
    import msvcrt

    while True:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise LockTimeout("Could not acquire write lock within the timeout") from None
            time.sleep(0.01)


# ─── DDL strings ──────────────────────────────────────────────────────────────

_DDL_INDEX_METADATA = """
CREATE TABLE IF NOT EXISTS index_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    indexed_git_head       TEXT    NOT NULL,
    indexed_git_tree_hash  TEXT,
    indexed_branch         TEXT    NOT NULL,
    indexed_file_manifest  TEXT    NOT NULL,
    config_hash            TEXT    NOT NULL,
    embedding_provider     TEXT    NOT NULL,
    embedding_model        TEXT    NOT NULL,
    embedding_dimensions   INTEGER NOT NULL,
    tokenizer_version      TEXT,
    updated_at             TEXT    NOT NULL
)
"""

_DDL_ANCHORS = """
CREATE TABLE IF NOT EXISTS anchors (
    id                    TEXT    PRIMARY KEY,
    type                  TEXT    NOT NULL,
    path                  TEXT    NOT NULL,
    heading_path_json     TEXT,
    symbol_name           TEXT,
    content_text          TEXT    NOT NULL,
    content_hash          TEXT    NOT NULL,
    fingerprint_simhash   INTEGER NOT NULL,
    transitive_hash_status TEXT,
    closure_hash          TEXT,
    def_line              INTEGER,
    def_char              INTEGER,
    overview_embedding    BLOB,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL
)
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id            TEXT    NOT NULL,
    chunk_index          INTEGER NOT NULL,
    text                 TEXT    NOT NULL,
    parent_content_hash  TEXT    NOT NULL,
    embedding            BLOB,
    overlap_with_prev    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (parent_id, chunk_index),
    FOREIGN KEY (parent_id) REFERENCES anchors(id) ON DELETE CASCADE
)
"""

_DDL_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='unicode61'
)
"""

# FTS5 sync triggers — BEFORE DELETE/UPDATE to capture old text, AFTER
# INSERT/UPDATE to register new text.
_DDL_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai
    AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_bd
    BEFORE DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_bu
    BEFORE UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text)
        VALUES ('delete', old.id, old.text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au
    AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END
    """,
]

_RE_VEC_DIM = re.compile(r"float\[(\d+)\]", re.IGNORECASE)


# ─── ScryDB ───────────────────────────────────────────────────────────────────


class ScryDB:
    """Connection wrapper around ``.scry/vectors.db``.

    Owns: sqlite-vec extension load, WAL mode, schema creation, advisory
    write lock acquisition, and CRUD for anchors / chunks / index_metadata.

    Concurrency model (DESIGN.md §7.3):
        Single writer (the leader, §10) holds the advisory write lock at
        ``.scry/vectors.db.lock`` for the duration of any write transaction.
        Multiple concurrent readers (followers, CLI, MCP queries) open the DB
        in read-only mode; they wrap each tool call in a fresh short read
        transaction.

    Args:
        repo_root: Root of the git repository (must contain ``.scry/``).
        read_only: Open in read-only mode (URI ``?mode=ro``).  Write methods
            raise ``IntegrityError`` when ``read_only=True``.
    """

    def __init__(self, repo_root: Path, *, read_only: bool = False) -> None:
        self._repo_root = repo_root
        self._read_only = read_only
        db_path = repo_root / ".scry" / "vectors.db"
        self._lock_path = repo_root / ".scry" / "vectors.db.lock"

        if read_only:
            uri = db_path.as_uri() + "?mode=ro"
            self._conn: sqlite3.Connection = sqlite3.connect(uri, uri=True)
        else:
            self._conn = sqlite3.connect(str(db_path))

        # Load the sqlite-vec extension before any schema operations.
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        # Core PRAGMAs — must run outside any explicit transaction.
        if not read_only:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Detect optional columns added by later migrations.  Read-only
        # callers (W2j search, MCP follower) cannot run ``ALTER TABLE``
        # so they must tolerate pre-migration databases by emitting
        # ``NULL AS <col>`` instead of selecting a column that doesn't
        # exist (review-w6e HIGH fix).
        self._has_def_position_columns = self._detect_def_position_columns()

    def _detect_def_position_columns(self) -> bool:
        """Return True if the ``anchors`` table already has ``def_line``/``def_char``.

        Tolerates a missing ``anchors`` table (fresh DB before
        ``init_schema``); returns False so the conservative SELECT shape
        is used until the schema is initialised.
        """
        try:
            rows = self._conn.execute("PRAGMA table_info(anchors)").fetchall()
        except sqlite3.OperationalError:
            return False
        names = {row[1] for row in rows}
        return "def_line" in names and "def_char" in names

    def _anchor_select_columns(self) -> str:
        """Return the comma-separated SELECT clause for the ``anchors`` table.

        When ``def_line`` / ``def_char`` are not present (read-only opens
        of pre-W6e databases), substitute ``NULL`` so ``_row_to_anchor``
        still receives 12 values and renders ``def_line=None``.
        """
        if self._has_def_position_columns:
            return (
                "id, type, path, heading_path_json, symbol_name, "
                "content_text, content_hash, fingerprint_simhash, "
                "transitive_hash_status, closure_hash, def_line, def_char"
            )
        return (
            "id, type, path, heading_path_json, symbol_name, "
            "content_text, content_hash, fingerprint_simhash, "
            "transitive_hash_status, closure_hash, "
            "NULL AS def_line, NULL AS def_char"
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> ScryDB:
        """Support ``with ScryDB(...) as db:`` for guaranteed cleanup."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Close the connection on exit."""
        self.close()

    # ------------------------------------------------------------------
    # Schema lifecycle
    # ------------------------------------------------------------------

    def init_schema(self, *, embedding_dimensions: int) -> None:
        """Create all tables, indexes, FTS5, and sqlite-vec virtual tables.

        Idempotent (``CREATE … IF NOT EXISTS``).  Drops and recreates the
        ``chunks_vec`` virtual table if the stored dimensionality differs from
        ``embedding_dimensions``, per DESIGN.md §7.2.1.

        Args:
            embedding_dimensions: Width of the embedding vectors (e.g. 384
                for BAAI/bge-small-en-v1.5).

        Raises:
            IntegrityError: If the DB is in read-only mode.
            SchemaError: If the existing ``chunks_vec`` SQL cannot be parsed.
        """
        self._check_writable()
        with self._conn:
            self._conn.execute(_DDL_INDEX_METADATA)
            self._conn.execute(_DDL_ANCHORS)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_anchors_path ON anchors(path)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_anchors_type ON anchors(type)")
            self._conn.execute(_DDL_CHUNKS)
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_parent_hash "
                "ON chunks(parent_id, parent_content_hash)"
            )
            self._conn.execute(_DDL_CHUNKS_FTS)
            for ddl in _DDL_TRIGGERS:
                self._conn.execute(ddl)

            # W3d migration: add closure_hash column if the anchors table was
            # created before this column was introduced.  ALTER TABLE … ADD
            # COLUMN raises OperationalError when the column already exists, so
            # we suppress that specific error (idempotent).
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ALTER TABLE anchors ADD COLUMN closure_hash TEXT")

            # W6e migration: add def_line / def_char columns for LSP position
            # lookup.  Idempotent — suppresses OperationalError when the columns
            # already exist (e.g. fresh DB created from the updated _DDL_ANCHORS).
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ALTER TABLE anchors ADD COLUMN def_line INTEGER")
            with contextlib.suppress(sqlite3.OperationalError):
                self._conn.execute("ALTER TABLE anchors ADD COLUMN def_char INTEGER")
            # Refresh the cached column-presence flag now that the migration
            # has (idempotently) run.
            self._has_def_position_columns = self._detect_def_position_columns()

            # Decide whether to create or recreate the vec virtual table.
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
            ).fetchone()

            if row is None:
                self._conn.execute(
                    f"CREATE VIRTUAL TABLE chunks_vec "
                    f"USING vec0(embedding float[{embedding_dimensions}])"
                )
            else:
                existing_sql: str = row[0] or ""
                m = _RE_VEC_DIM.search(existing_sql)
                if m is None:
                    raise SchemaError(
                        f"Cannot determine dimensionality of existing chunks_vec table "
                        f"from SQL: {existing_sql!r}"
                    )
                existing_dim = int(m.group(1))
                if existing_dim != embedding_dimensions:
                    # Dimension mismatch — drop and recreate (§7.2.1).
                    self._conn.execute("DROP TABLE IF EXISTS chunks_vec")
                    self._conn.execute(
                        f"CREATE VIRTUAL TABLE chunks_vec "
                        f"USING vec0(embedding float[{embedding_dimensions}])"
                    )
                    # Null out stored embeddings so the indexer knows to refill.
                    self._conn.execute("UPDATE chunks SET embedding = NULL")

    def recreate_vector_table(self, new_dimensions: int) -> None:
        """Drop and recreate ``chunks_vec`` with new dimensionality.

        Used by ``scry index --reembed`` (W4d) when the embedding model
        changes.  Anchor rows, fingerprints, FTS5 index, and chunk text are
        preserved.  Chunk ``embedding`` columns are set to NULL so the indexer
        knows to refill them.

        The operation is a single SQLite transaction together with the
        ``index_metadata.embedding_dimensions`` update when called from the
        indexer, satisfying the atomicity requirement of DESIGN.md §7.2.1.

        Args:
            new_dimensions: New vector width.

        Raises:
            IntegrityError: If the DB is in read-only mode.
        """
        self._check_writable()
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS chunks_vec")
            self._conn.execute(
                f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{new_dimensions}])"
            )
            self._conn.execute("UPDATE chunks SET embedding = NULL")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Index metadata (singleton row)
    # ------------------------------------------------------------------

    def read_index_metadata(self) -> IndexMetadata | None:
        """Read the singleton ``index_metadata`` row.

        Returns:
            The stored ``IndexMetadata``, or ``None`` if the table is empty.
        """
        row = self._conn.execute(
            """
            SELECT indexed_git_head, indexed_git_tree_hash, indexed_branch,
                   indexed_file_manifest, config_hash, embedding_provider,
                   embedding_model, embedding_dimensions, tokenizer_version
            FROM index_metadata WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return None
        manifest: dict[str, str] = json.loads(row[3]) if row[3] else {}
        return IndexMetadata(
            indexed_git_head=row[0],
            indexed_git_tree_hash=row[1],
            indexed_branch=row[2],
            indexed_file_manifest=manifest,
            config_hash=row[4],
            embedding_provider=row[5],
            embedding_model=row[6],
            embedding_dimensions=row[7],
            tokenizer_version=row[8],
        )

    def write_index_metadata(self, meta: IndexMetadata) -> None:
        """Upsert the singleton ``index_metadata`` row.

        Args:
            meta: Metadata to persist.

        Raises:
            IntegrityError: If the DB is in read-only mode.
        """
        self._check_writable()
        now = _now_iso()
        manifest_json = json.dumps(meta.indexed_file_manifest)
        with self._maybe_txn():
            self._conn.execute(
                """
                INSERT INTO index_metadata
                    (id, indexed_git_head, indexed_git_tree_hash, indexed_branch,
                     indexed_file_manifest, config_hash, embedding_provider,
                     embedding_model, embedding_dimensions, tokenizer_version,
                     updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    indexed_git_head      = excluded.indexed_git_head,
                    indexed_git_tree_hash = excluded.indexed_git_tree_hash,
                    indexed_branch        = excluded.indexed_branch,
                    indexed_file_manifest = excluded.indexed_file_manifest,
                    config_hash           = excluded.config_hash,
                    embedding_provider    = excluded.embedding_provider,
                    embedding_model       = excluded.embedding_model,
                    embedding_dimensions  = excluded.embedding_dimensions,
                    tokenizer_version     = excluded.tokenizer_version,
                    updated_at            = excluded.updated_at
                """,
                (
                    meta.indexed_git_head,
                    meta.indexed_git_tree_hash,
                    meta.indexed_branch,
                    manifest_json,
                    meta.config_hash,
                    meta.embedding_provider,
                    meta.embedding_model,
                    meta.embedding_dimensions,
                    meta.tokenizer_version,
                    now,
                ),
            )

    # ------------------------------------------------------------------
    # Anchors
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _maybe_txn(self) -> Generator[None, None, None]:
        """Open an implicit transaction unless we're already in one.

        Used by single-write methods (``upsert_anchor``, ``replace_chunks``,
        ``write_index_metadata``) so they remain atomic when called
        standalone, but participate in the outer transaction (without
        auto-committing) when called from within a :meth:`transaction`
        block.
        """
        if self._conn.in_transaction:
            yield
            return
        with self._conn:
            yield

    def upsert_anchor(self, anchor: Anchor) -> None:
        """Insert or update a single anchor row.

        On content-hash change the ``overview_embedding`` is NULLed so the
        indexer knows to re-embed.  ``created_at`` is preserved on update.

        Args:
            anchor: Anchor to persist.

        Raises:
            IntegrityError: If the DB is in read-only mode.

        Note:
            For per-parent reindex (delete-all chunks + insert-all + update
            parent in one transaction per §7.3 invariant 2), use
            :meth:`reindex_anchor_with_chunks` instead, OR use
            :meth:`transaction` to group ``upsert_anchor`` and
            ``replace_chunks`` calls into one atomic unit.  Calling these
            two methods sequentially without an outer transaction creates
            a stale-orphan window if a crash interleaves.
        """
        self._check_writable()
        now = _now_iso()
        with self._maybe_txn():
            _upsert_anchor_in_txn(self._conn, anchor, now)

    @contextlib.contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that groups multiple write methods into ONE
        SQLite transaction (DESIGN.md §7.3 invariant 2 / §7.2.1 reembed).

        Use this whenever a write operation must be atomic across multiple
        method calls — e.g. updating a parent anchor AND replacing its
        chunks (the canonical per-parent reindex path):

        .. code-block:: python

            with db.transaction():
                db.upsert_anchor(anchor)
                db.replace_chunks(anchor.id, anchor.content_hash, chunks, embeddings)

        Implementation: opens an explicit ``BEGIN IMMEDIATE`` and either
        commits (on clean exit) or rolls back (on exception).  Yields the
        underlying ``sqlite3.Connection`` so callers can issue raw SQL
        within the same transaction if needed.

        Nesting is NOT supported (sqlite3 doesn't support nested
        transactions natively); calling :meth:`transaction` from within
        another transaction raises ``IntegrityError``.

        Raises:
            IntegrityError: If the DB is in read-only mode OR if called
                while another transaction is already in progress.
        """
        self._check_writable()
        if self._conn.in_transaction:
            raise IntegrityError(
                "transaction() called while another transaction is in progress; "
                "nested transactions are not supported"
            )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            with contextlib.suppress(Exception):
                self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def reindex_anchor_with_chunks(
        self,
        anchor: Anchor,
        chunks: Iterable[SubChunk],
        embeddings: Iterable[bytes] | None = None,
    ) -> None:
        """Atomic per-parent reindex (DESIGN.md §7.3 invariant 2).

        Performs ``upsert_anchor(anchor)`` AND
        ``replace_chunks(anchor.id, anchor.content_hash, chunks, embeddings)``
        inside a SINGLE SQLite transaction so that on crash the parent's
        ``content_hash`` and its chunks' ``parent_content_hash`` either
        BOTH reflect the new state or BOTH reflect the old — never a mix
        that would leave stale orphans visible to the read-side hash
        equality filter (§7.3 invariant 4).

        This is the recommended path for the indexer (W2l).  Direct
        sequential calls to ``upsert_anchor`` then ``replace_chunks``
        without an enclosing :meth:`transaction` MUST NOT be used in
        production code paths.

        Args:
            anchor: Parent anchor (its ``content_hash`` is the value
                stamped on every new chunk row).
            chunks: New sub-chunks to insert (consumed once).
            embeddings: Optional parallel iterable of embedding blobs
                (one per chunk; same wire format as
                :func:`scry.embed.serialize_embedding`).

        Raises:
            IntegrityError: If the DB is in read-only mode.
            ValueError: If ``embeddings`` length does not match ``chunks``.
        """
        self._check_writable()
        chunks_list = list(chunks)
        emb_list: list[bytes] | None = list(embeddings) if embeddings is not None else None
        if emb_list is not None and len(emb_list) != len(chunks_list):
            raise ValueError(
                f"embeddings count ({len(emb_list)}) != chunks count ({len(chunks_list)})"
            )
        now = _now_iso()
        with self.transaction():
            _upsert_anchor_in_txn(self._conn, anchor, now)
            _replace_chunks_in_txn(
                self._conn, anchor.id, anchor.content_hash, chunks_list, emb_list
            )

    def upsert_anchors(self, anchors: Iterable[Anchor]) -> None:
        """Batch-upsert anchors in a single transaction.

        All-or-nothing: if any upsert raises, the whole batch is rolled back.

        Args:
            anchors: Anchors to persist (consumed once).

        Raises:
            IntegrityError: If the DB is in read-only mode.
        """
        self._check_writable()
        now = _now_iso()
        with self._maybe_txn():
            for anchor in anchors:
                _upsert_anchor_in_txn(self._conn, anchor, now)

    def get_anchor(self, anchor_id: str) -> Anchor | None:
        """Fetch a single anchor by primary ID.

        Args:
            anchor_id: The anchor's primary ID (e.g.
                ``"docs/SPEC.md::section::sub"``).

        Returns:
            The matching ``Anchor``, or ``None`` if not found.
        """
        row = self._conn.execute(
            f"""
            SELECT {self._anchor_select_columns()}
            FROM anchors WHERE id = ?
            """,
            (anchor_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_anchor(row)

    def list_anchors(
        self,
        *,
        anchor_type: AnchorType | None = None,
        path: str | None = None,
    ) -> list[Anchor]:
        """Return anchors matching optional filters.

        Args:
            anchor_type: Only return anchors of this type (``None`` = all).
            path: Only return anchors from this repo-relative path
                (exact match; ``None`` = all).

        Returns:
            List of matching ``Anchor`` objects (unordered).
        """
        query = f"SELECT {self._anchor_select_columns()} FROM anchors"
        conditions: list[str] = []
        params: list[Any] = []
        if anchor_type is not None:
            conditions.append("type = ?")
            params.append(str(anchor_type))
        if path is not None:
            conditions.append("path = ?")
            params.append(path)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_anchor(r) for r in rows]

    def delete_anchor(self, anchor_id: str) -> None:
        """Delete an anchor and cascade to its chunks.

        The FK ``ON DELETE CASCADE`` removes child ``chunks`` rows; the
        ``chunks_bd`` trigger removes their FTS5 entries; and this method
        explicitly removes the corresponding ``chunks_vec`` rows before
        deleting the anchor.

        Args:
            anchor_id: ID of the anchor to delete.

        Raises:
            IntegrityError: If the DB is in read-only mode.
        """
        self._check_writable()
        with self._maybe_txn():
            cur = self._conn.cursor()
            # Fetch chunk rowids before the cascade deletes them.
            rowids = [
                r[0]
                for r in cur.execute(
                    "SELECT id FROM chunks WHERE parent_id = ?", (anchor_id,)
                ).fetchall()
            ]
            for rowid in rowids:
                cur.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
            cur.execute("DELETE FROM anchors WHERE id = ?", (anchor_id,))

    # ------------------------------------------------------------------
    # Sub-chunks
    # ------------------------------------------------------------------

    def replace_chunks(
        self,
        parent_id: str,
        parent_content_hash: str,
        chunks: Iterable[SubChunk],
        embeddings: Iterable[bytes] | None = None,
    ) -> None:
        """Atomic per-parent chunk replacement: delete old chunks, insert new ones.

        Satisfies DESIGN.md §7.3 invariants 1, 3, 4 (within this method):
        - All new chunk rows carry ``parent_content_hash`` so the read-side
          filter can drop stale orphans.
        - Delete + insert happen in a single transaction; on crash the parent
          either has its old chunks or its new chunks, never a mixture.
        - FTS5 is kept in sync via the ``chunks_bd``/``chunks_ai`` triggers.
        - ``chunks_vec`` rows are managed explicitly (no FK to the vec table).

        Args:
            parent_id: ID of the parent anchor.
            parent_content_hash: Current ``content_hash`` of the parent anchor.
                Stamped on every new chunk row.
            chunks: New sub-chunks to insert (consumed once).
            embeddings: Optional parallel iterable of float32-blob embeddings
                (one per chunk, in ``struct.pack("<Nf", ...)`` format).  When
                provided, each embedding is stored in ``chunks.embedding`` and
                inserted into ``chunks_vec``.

        Raises:
            IntegrityError: If the DB is in read-only mode.
            ValueError: If ``embeddings`` length does not match ``chunks``.

        Note:
            This method ALONE does not satisfy §7.3 invariant 2 (atomic
            parent + chunks reindex).  When updating both the parent
            anchor's content_hash AND its chunks, use
            :meth:`reindex_anchor_with_chunks` (or wrap the two calls in
            an explicit :meth:`transaction`).
        """
        self._check_writable()
        chunks_list = list(chunks)
        emb_list: list[bytes] | None = list(embeddings) if embeddings is not None else None
        if emb_list is not None and len(emb_list) != len(chunks_list):
            raise ValueError(
                f"embeddings count ({len(emb_list)}) != chunks count ({len(chunks_list)})"
            )

        with self._maybe_txn():
            _replace_chunks_in_txn(
                self._conn, parent_id, parent_content_hash, chunks_list, emb_list
            )

    def list_chunks(self, parent_id: str) -> list[SubChunk]:
        """Return all sub-chunks for a given parent anchor, ordered by index.

        Args:
            parent_id: ID of the parent anchor.

        Returns:
            List of ``SubChunk`` objects sorted by ``chunk_index``.
        """
        rows = self._conn.execute(
            """
            SELECT parent_id, chunk_index, text, parent_content_hash,
                   overlap_with_prev
            FROM chunks
            WHERE parent_id = ?
            ORDER BY chunk_index
            """,
            (parent_id,),
        ).fetchall()
        return [
            SubChunk(
                parent_id=r[0],
                chunk_index=r[1],
                text=r[2],
                parent_content_hash=r[3],
                overlap_with_prev=r[4],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Vector + BM25 retrieval hooks
    # ------------------------------------------------------------------

    def query_vector(self, query_embedding: bytes, *, top_k: int) -> list[tuple[int, float]]:
        """Return nearest-neighbour chunk rowids via sqlite-vec ANN.

        Args:
            query_embedding: Float32 blob in ``struct.pack("<Nf", ...)`` format.
            top_k: Maximum number of results.

        Returns:
            List of ``(chunk_rowid, distance)`` tuples, ordered
            ascending by distance (smallest = most similar).
        """
        rows = self._conn.execute(
            """
            SELECT rowid, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_embedding, top_k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def query_bm25(self, query: str, *, top_k: int) -> list[tuple[int, float]]:
        """Return keyword-matched chunk rowids via FTS5 BM25.

        Args:
            query: FTS5 MATCH expression (e.g. ``"policy engine"``).
            top_k: Maximum number of results.

        Returns:
            List of ``(chunk_rowid, bm25_score)`` tuples, ordered by
            ``rank`` (FTS5 canonical ordering; more-negative scores are better
            matches — negate for a conventional descending relevance sort).
        """
        rows = self._conn.execute(
            """
            SELECT rowid, bm25(chunks_fts)
            FROM chunks_fts
            WHERE text MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, top_k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    # ------------------------------------------------------------------
    # Advisory write lock
    # ------------------------------------------------------------------

    def acquire_write_lock(self, *, timeout_seconds: float = 5.0) -> WriteLock:
        """Return a context manager that holds the advisory write lock.

        The lock file is ``.scry/vectors.db.lock``.  Uses ``fcntl.flock``
        on Unix and ``msvcrt.locking`` on Windows.  The lock is OS-level and
        is released automatically on process death.

        Usage::

            with db.acquire_write_lock():
                db.upsert_anchors(...)

        Args:
            timeout_seconds: Maximum seconds to wait for lock acquisition.

        Returns:
            A ``WriteLock`` context manager.

        Raises:
            IntegrityError: If the DB is in read-only mode.
            LockTimeout: If the lock cannot be acquired within the timeout.
        """
        self._check_writable()
        return WriteLock(self._lock_path, timeout_seconds=timeout_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_writable(self) -> None:
        """Raise ``IntegrityError`` if this connection is read-only."""
        if self._read_only:
            raise IntegrityError("Database opened in read-only mode; writes are not allowed")


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _to_signed_int64(value: int) -> int:
    """Convert an unsigned 64-bit integer to a signed 64-bit two's-complement.

    SQLite ``INTEGER`` is signed 64-bit (``-2^63`` to ``2^63 - 1``), but
    ``simhash.Simhash.value`` returns an unsigned 64-bit fingerprint (``0``
    to ``2^64 - 1``).  Passing a value with the high bit set directly
    raises ``sqlite3.OverflowError`` ≈ 50 % of the time on real-world
    fingerprints.  This helper reinterprets the bit pattern as
    two's-complement so any value in [0, 2^64) round-trips through SQLite
    without OverflowError.

    Inverse: :func:`_from_signed_int64`.
    """
    if value < 0 or value >= (1 << 64):
        raise ValueError(f"_to_signed_int64 requires an unsigned 64-bit input, got {value}")
    return value if value < (1 << 63) else value - (1 << 64)


def _from_signed_int64(value: int) -> int:
    """Inverse of :func:`_to_signed_int64`.

    Reinterprets a signed 64-bit two's-complement integer as the unsigned
    64-bit value the extractor originally produced.  Required so that
    ``Anchor.fingerprint_simhash`` round-trips losslessly through the
    SQLite store and downstream code (W4 fuzzy-match Jaccard, §3.3 inline
    rebase) compares fresh extractor output against stored fingerprints
    using the same sign convention.
    """
    return value if value >= 0 else value + (1 << 64)


def _upsert_anchor_in_txn(conn: sqlite3.Connection, anchor: Anchor, now: str) -> None:
    """Execute a single anchor upsert inside the caller's transaction.

    ``created_at`` is preserved on conflict.  ``overview_embedding`` is
    NULLed when ``content_hash`` changes (indexer must re-embed).

    Args:
        conn: Active SQLite connection (transaction must be open).
        anchor: Anchor to upsert.
        now: ISO-8601 timestamp to use for ``updated_at`` (and ``created_at``
            on first insert).
    """
    heading_path_json: str | None = (
        json.dumps(anchor.heading_path) if anchor.heading_path is not None else None
    )
    # Convert the unsigned 64-bit simhash to signed two's-complement so
    # SQLite (signed INTEGER) accepts the high-bit-set values.  The read
    # side (_row_to_anchor) applies the inverse so callers always see
    # the original unsigned value.
    simhash_signed = _to_signed_int64(anchor.fingerprint_simhash)
    conn.execute(
        """
        INSERT INTO anchors
            (id, type, path, heading_path_json, symbol_name, content_text,
             content_hash, fingerprint_simhash, transitive_hash_status,
             closure_hash, def_line, def_char, overview_embedding,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            type                  = excluded.type,
            path                  = excluded.path,
            heading_path_json     = excluded.heading_path_json,
            symbol_name           = excluded.symbol_name,
            content_text          = excluded.content_text,
            content_hash          = excluded.content_hash,
            fingerprint_simhash   = excluded.fingerprint_simhash,
            transitive_hash_status = excluded.transitive_hash_status,
            closure_hash          = excluded.closure_hash,
            def_line              = excluded.def_line,
            def_char              = excluded.def_char,
            overview_embedding    = CASE
                WHEN content_hash != excluded.content_hash THEN NULL
                ELSE overview_embedding
            END,
            updated_at            = excluded.updated_at
        """,
        (
            anchor.id,
            str(anchor.type),
            anchor.path,
            heading_path_json,
            anchor.symbol_name,
            anchor.content_text,
            anchor.content_hash,
            simhash_signed,
            str(anchor.transitive_hash_status)
            if anchor.transitive_hash_status is not None
            else None,
            anchor.closure_hash,
            anchor.def_line,
            anchor.def_char,
            now,
            now,
        ),
    )


def _replace_chunks_in_txn(
    conn: sqlite3.Connection,
    parent_id: str,
    parent_content_hash: str,
    chunks_list: list[SubChunk],
    emb_list: list[bytes] | None,
) -> None:
    """Inner per-parent chunk replacement (called from within a transaction).

    The caller is responsible for opening the transaction.  Used by
    :meth:`ScryDB.replace_chunks` (which opens its own implicit
    transaction) and by :meth:`ScryDB.reindex_anchor_with_chunks` (which
    needs to share a transaction with the parent anchor's upsert per
    DESIGN.md §7.3 invariant 2).
    """
    cur = conn.cursor()
    # Collect existing chunk rowids to purge from chunks_vec.
    old_rowids = [
        r[0]
        for r in cur.execute("SELECT id FROM chunks WHERE parent_id = ?", (parent_id,)).fetchall()
    ]
    for rowid in old_rowids:
        cur.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
    # Delete from chunks — the BD trigger removes FTS5 entries.
    cur.execute("DELETE FROM chunks WHERE parent_id = ?", (parent_id,))

    # Insert new chunks.
    for i, chunk in enumerate(chunks_list):
        emb_blob: bytes | None = emb_list[i] if emb_list is not None else None
        cur.execute(
            """
            INSERT INTO chunks
                (parent_id, chunk_index, text, parent_content_hash,
                 embedding, overlap_with_prev)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.parent_id,
                chunk.chunk_index,
                chunk.text,
                parent_content_hash,
                emb_blob,
                chunk.overlap_with_prev,
            ),
        )
        new_rowid: int = cur.lastrowid  # type: ignore[assignment]
        if emb_blob is not None:
            cur.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (new_rowid, emb_blob),
            )


def _row_to_anchor(row: Any) -> Anchor:
    """Reconstruct an ``Anchor`` from a SELECT row (12 columns, no embedding).

    Column order: id, type, path, heading_path_json, symbol_name,
    content_text, content_hash, fingerprint_simhash, transitive_hash_status,
    closure_hash, def_line, def_char.

    Args:
        row: A tuple (or sqlite3.Row) from an anchors SELECT query.

    Returns:
        Reconstructed ``Anchor`` (Pydantic validates field values).
    """
    (
        id_,
        type_,
        path,
        heading_path_json,
        symbol_name,
        content_text,
        content_hash,
        fingerprint_simhash,
        transitive_hash_status,
        closure_hash,
        def_line,
        def_char,
    ) = row
    heading_path: list[str] | None = (
        json.loads(heading_path_json) if heading_path_json is not None else None
    )
    return Anchor(
        id=id_,
        type=type_,
        path=path,
        heading_path=heading_path,
        symbol_name=symbol_name,
        content_text=content_text,
        content_hash=content_hash,
        # Reverse the signed→unsigned conversion applied at write time so
        # the round-trip preserves the extractor's original unsigned value
        # (downstream W4 fuzzy-match Jaccard + §3.3 inline rebase compare
        # against this verbatim).
        fingerprint_simhash=_from_signed_int64(int(fingerprint_simhash)),
        transitive_hash_status=transitive_hash_status,
        closure_hash=closure_hash,
        def_line=int(def_line) if def_line is not None else None,
        def_char=int(def_char) if def_char is not None else None,
    )
