"""Indexer orchestrator for scry (workstream W2l).

Implements the ``scry index`` runtime per DESIGN.md:
    §3.4   Two-tier embedding (parent overview + sub-chunks)
    §6     Configuration (include/exclude/classify/code_anchors)
    §7.1   Index provenance metadata
    §7.2   Auto-reconcile via polling (incremental change detection)
    §7.2.1 --reembed semantics (surgical embedding-model migration)
    §7.3   Two-tier consistency invariants (per-parent atomic transactions)

§7.3 Note on per-parent atomicity
----------------------------------
The spec requires anchor upsert + chunk replace to be a **single SQLite
transaction**.  ScryDB.upsert_anchor() and ScryDB.replace_chunks() each open
their own ``with self._conn:`` transaction, making them unsuitable for combined
atomicity via the public API.

The indexer therefore accesses ``db._conn`` (the underlying
``sqlite3.Connection``) directly and opens a single ``with conn:`` block per
parent anchor.  This is an intentional, documented coupling to ScryDB internals
and is the sole location in the codebase that does so.  The companion private
helpers ``_upsert_anchor_in_txn`` and ``_now_iso`` are imported from
``scry.store.db`` to keep the SQL in one place.

Advisory write lock
-------------------
``db.acquire_write_lock()`` (OS-level ``fcntl.flock`` / ``msvcrt.locking``)
is held for the full duration of every ``index()`` or ``reembed()`` call.
For Wave 2 the indexer runs as a one-shot CLI command (W2j), so the simpler
advisory lock from W2a is sufficient; the leader-election mechanism (W2g) is
not wired here.

Public API
----------
ExtractionTarget   — file classification result (kind + optional language)
IndexResult        — summary dataclass returned by index() / reembed()
IndexerError       — fatal indexer failure
Indexer            — main orchestrator class
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from scry.anchor_id import canonicalize_content
from scry.anchor_id import content_hash as compute_content_hash
from scry.config import (
    classify_path,
    compute_config_hash,
    load_config,
    parse_frontmatter,
    should_index,
)
from scry.embed import Embedder, chunk_anchor, make_embedder
from scry.extract.code import extract_code_symbols
from scry.extract.markdown import extract_markdown
from scry.git_context import GitContextError, GitContextProvider
from scry.models import Anchor, AnchorType, Config, Frontmatter, IndexMetadata
from scry.store.db import ScryDB, _now_iso, _upsert_anchor_in_txn

__all__ = [
    "ExtractionTarget",
    "IndexResult",
    "Indexer",
    "IndexerError",
]

logger = logging.getLogger(__name__)

# Approximate characters per token for the §3.4 overview text budget.
# ~4 chars/token is a standard approximation for English prose; 200 tokens ≈ 800 chars.
_OVERVIEW_CHARS: int = 800

# Mapping from lowercase file extension → tree-sitter language name.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "jsx",
    ".zig": "zig",
}

# Number of anchors processed per reembed batch transaction.
_REEMBED_BATCH_SIZE: int = 50


# ─── Public data classes ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionTarget:
    """Classification of a single file for extraction dispatch.

    Attributes:
        kind:     ``'markdown'``, ``'code'``, or ``'skip'``.
        language: Resolved language name (e.g. ``'python'``), set only when
                  ``kind == 'code'``.  Always ``None`` for other kinds.
    """

    kind: Literal["markdown", "code", "skip"]
    language: str | None = None


@dataclass(frozen=True)
class IndexResult:
    """Summary of what an index() or reembed() pass produced.

    Attributes:
        files_processed:  Files for which extraction was attempted.
        anchors_extracted: Total anchors returned by the extractors.
        anchors_embedded:  Anchors for which embeddings were stored.
        chunks_written:    Total sub-chunk rows written (including overview chunks).
        files_pruned:      Distinct source files deleted since the last index whose
                           anchors were removed from vectors.db.
        elapsed_seconds:   Wall-clock time for the full pass.
    """

    files_processed: int
    anchors_extracted: int
    anchors_embedded: int
    chunks_written: int
    files_pruned: int
    elapsed_seconds: float


# ─── Exception ────────────────────────────────────────────────────────────────


class IndexerError(Exception):
    """Raised for fatal indexer failures (missing config, git error, etc.)."""


# ─── Indexer ──────────────────────────────────────────────────────────────────


class Indexer:
    """The ``scry index`` runtime.

    Wires together extractors (W1a/W1b), embedder (W2k), DB (W2a), and
    config (W1d) to produce a fully-indexed ``vectors.db`` state.

    Per §7.3 invariants:
        - Per-parent reindex is a single SQLite transaction (see module
          docstring for the ``db._conn`` rationale).
        - ``index_metadata`` updates are committed atomically with the final
          chunk-write batch so partial runs do not leave stale provenance.
        - The search-side hash-equality filter (``chunk.parent_content_hash ==
          parent.content_hash``) drops stale orphans automatically.

    All constructor arguments are optional.  Unset ones are resolved lazily
    from ``repo_root`` on first use, which is convenient for tests that inject
    a :class:`~scry.embed.StubEmbedder` and an in-memory (or temp-file) DB.

    Args:
        repo_root:    Absolute path to the repository root (must contain
                      ``.scry/config.yaml``).
        config:       Pre-loaded :class:`~scry.models.Config`; loaded from
                      ``repo_root`` on first use when ``None``.
        db:           Open :class:`~scry.store.db.ScryDB`; created and
                      schema-initialised on first use when ``None``.
        embedder:     :class:`~scry.embed.Embedder` backend; created from
                      config on first use when ``None``.
        git_context:  :class:`~scry.git_context.GitContextProvider`; created
                      from config on first use when ``None``.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        config: Config | None = None,
        db: ScryDB | None = None,
        embedder: Embedder | None = None,
        git_context: GitContextProvider | None = None,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._config = config
        self._db = db
        self._embedder = embedder
        self._git_context = git_context

    # ------------------------------------------------------------------
    # Lazy resource accessors
    # ------------------------------------------------------------------

    def _ensure_config(self) -> Config:
        """Return the config, loading from disk if not yet provided."""
        if self._config is None:
            self._config = load_config(self._repo_root)
        return self._config

    def _ensure_db(self, config: Config) -> ScryDB:
        """Return the DB, creating and schema-initialising it if needed.

        ``init_schema()`` is idempotent (``CREATE … IF NOT EXISTS``), so
        calling it on an already-initialised DB is safe.
        """
        if self._db is None:
            self._db = ScryDB(self._repo_root)
        self._db.init_schema(embedding_dimensions=config.embeddings.dimensions)
        return self._db

    def _ensure_embedder(self, config: Config) -> Embedder:
        """Return the embedder, creating it from config if not yet provided."""
        if self._embedder is None:
            self._embedder = make_embedder(config.embeddings)
        return self._embedder

    def _ensure_git_context(self, config: Config) -> GitContextProvider:
        """Return the git context provider, creating it from config if needed."""
        if self._git_context is None:
            self._git_context = GitContextProvider.from_config(self._repo_root, config.index)
        return self._git_context

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def discover_files(self) -> list[Path]:
        """Walk the repo applying include/exclude globs from config.

        Reads frontmatter from markdown files (lightweight) to honour
        per-file ``skip: true`` directives.  Results are returned in
        deterministic (sorted) order so that two identical invocations
        produce identical manifests.

        Returns:
            Absolute paths for every file that passes ``should_index()``,
            sorted lexicographically.
        """
        config = self._ensure_config()
        results: list[Path] = []

        for path in sorted(self._repo_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel_str = path.relative_to(self._repo_root).as_posix()
            except ValueError:
                continue

            # Read frontmatter only for markdown files (cheap; ~1 KB typical).
            frontmatter: Frontmatter | None = None
            if path.suffix.lower() in (".md", ".markdown"):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    frontmatter, _ = parse_frontmatter(text)
                except OSError:
                    pass

            if should_index(rel_str, frontmatter, config.include, config.exclude):
                results.append(path)

        return results

    def classify_for_extraction(self, path: Path) -> ExtractionTarget:
        """Classify *path* as markdown, code, or skip.

        Decision logic:
          1. If the repo-relative path matches the ``classify:`` list → markdown.
          2. Else if the extension maps to a configured code language → code.
          3. Otherwise → skip.

        Args:
            path: Absolute path to the file.

        Returns:
            An :class:`ExtractionTarget` with the resolved kind and (for code)
            the language name.
        """
        config = self._ensure_config()
        try:
            rel_str = path.relative_to(self._repo_root).as_posix()
        except ValueError:
            return ExtractionTarget(kind="skip")

        # Markdown first: classify list is ordered, first-match-wins (§6).
        if classify_path(rel_str, config.classify) is not None:
            return ExtractionTarget(kind="markdown")

        # Code: check extension against language map.
        suffix = path.suffix.lower()
        lang = _EXT_TO_LANG.get(suffix)
        if lang is not None:
            lang_mode = config.code_anchors.languages.get(lang, "lsp")
            if lang_mode != "skip":
                return ExtractionTarget(kind="code", language=lang)

        return ExtractionTarget(kind="skip")

    def index(self, *, force: bool = False) -> IndexResult:
        """Build or refresh the vector store.

        Incremental (``force=False``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Compares the current file manifest against
        ``IndexMetadata.indexed_file_manifest``.  Only files whose content hash
        has changed (or which are new) are re-extracted and re-embedded.  Files
        deleted since the last index have their anchors pruned.  If the config
        hash has changed, a full reindex is forced regardless.

        Full rebuild (``force=True``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Drops **all** existing anchors (and their chunks), then re-extracts
        and re-embeds everything from scratch.  Does NOT touch ``links.jsonl``
        or overlay files.

        Returns:
            An :class:`IndexResult` summarising what was done.

        Raises:
            IndexerError: If the config is missing/invalid, git is unavailable,
                or a fatal I/O error occurs.
        """
        start = time.monotonic()
        try:
            config = self._ensure_config()
        except Exception as exc:
            raise IndexerError(f"Failed to load config: {exc}") from exc

        db = self._ensure_db(config)
        embedder = self._ensure_embedder(config)
        git_provider = self._ensure_git_context(config)

        files_processed = 0
        anchors_extracted = 0
        anchors_embedded = 0
        chunks_written = 0
        files_pruned = 0

        with db.acquire_write_lock():
            try:
                git_ctx = git_provider.get(force_refresh=True)
            except GitContextError as exc:
                raise IndexerError(f"Git context unavailable: {exc}") from exc

            prior_meta = db.read_index_metadata()
            current_config_hash = compute_config_hash(config)
            all_files = self.discover_files()
            all_files_rel = {p.relative_to(self._repo_root).as_posix() for p in all_files}

            # ── Decide which files to process ───────────────────────────
            files_to_process: list[Path]
            if (
                not force
                and prior_meta is not None
                and prior_meta.config_hash == current_config_hash
            ):
                # Incremental: only changed / new files.
                prior_manifest = prior_meta.indexed_file_manifest
                files_to_process = []
                for path in all_files:
                    rel = path.relative_to(self._repo_root).as_posix()
                    current_hash = _file_content_hash(path)
                    if prior_manifest.get(rel) != current_hash:
                        files_to_process.append(path)

                # Prune: anchors for files no longer in the indexed set.
                removed_paths = set(prior_manifest.keys()) - all_files_rel
                pruned_file_paths: set[str] = set()
                for removed_rel in removed_paths:
                    for anchor in db.list_anchors(path=removed_rel):
                        db.delete_anchor(anchor.id)
                        pruned_file_paths.add(removed_rel)
                files_pruned = len(pruned_file_paths)
            else:
                # Full reindex (force=True, first run, or config changed).
                if force or prior_meta is not None:
                    # Delete all existing anchors.
                    for existing in db.list_anchors():
                        db.delete_anchor(existing.id)
                files_to_process = list(all_files)

            # ── Build the new manifest from current on-disk state ────────
            new_manifest: dict[str, str] = {
                path.relative_to(self._repo_root).as_posix(): _file_content_hash(path)
                for path in all_files
            }

            # ── Extract and embed ────────────────────────────────────────
            for path in files_to_process:
                target = self.classify_for_extraction(path)
                if target.kind == "skip":
                    continue

                rel = path.relative_to(self._repo_root).as_posix()

                # Delete stale anchors for this file before re-inserting,
                # so renamed sections don't leave orphan IDs behind.
                for old_anchor in db.list_anchors(path=rel):
                    db.delete_anchor(old_anchor.id)

                anchors = _extract_file_anchors(path, target, config, self._repo_root)
                files_processed += 1
                anchors_extracted += len(anchors)

                for anchor in anchors:
                    n = _process_anchor(
                        anchor,
                        db,
                        embedder,
                        max_tokens=config.sections.max_tokens,
                        overlap_tokens=config.sections.overlap_tokens,
                    )
                    anchors_embedded += 1
                    chunks_written += n

            # ── Write index_metadata ─────────────────────────────────────
            indexed_branch = git_ctx.branch or f"detached-{git_ctx.head_short}"
            new_meta = IndexMetadata(
                indexed_git_head=git_ctx.head_sha,
                indexed_git_tree_hash=None,
                indexed_branch=indexed_branch,
                indexed_file_manifest=new_manifest,
                config_hash=current_config_hash,
                embedding_provider=embedder.provider,
                embedding_model=embedder.model_name,
                embedding_dimensions=embedder.dimensions,
                tokenizer_version=embedder.tokenizer_version,
            )
            db.write_index_metadata(new_meta)

        elapsed = time.monotonic() - start
        return IndexResult(
            files_processed=files_processed,
            anchors_extracted=anchors_extracted,
            anchors_embedded=anchors_embedded,
            chunks_written=chunks_written,
            files_pruned=files_pruned,
            elapsed_seconds=elapsed,
        )

    def reembed(self) -> IndexResult:
        """Surgical embedding-model migration (DESIGN.md §7.2.1).

        Re-embeds every anchor that still has a source file using the current
        configured embedding model without touching anchors, fingerprints, FTS5
        entries, ``links.jsonl``, or overlays.

        Algorithm
        ~~~~~~~~~
        1. Prune anchors whose source files no longer exist in the working tree.
        2. If the new embedding dimensions differ from the stored dimensions,
           drop and recreate the ``chunks_vec`` virtual table (preserving all
           ``chunks`` rows; their ``embedding`` columns are set to ``NULL``).
        3. Re-embed all remaining anchors in batches of :const:`_REEMBED_BATCH_SIZE`.
           Each batch is its own SQLite transaction for crash safety.
        4. Update ``index_metadata.embedding_*`` fields atomically with the
           **final** batch so that a crash mid-reembed leaves the old model
           fields intact and the mismatch detector triggers on next startup
           (causing the user to re-run ``--reembed``, which resumes).

        Returns:
            An :class:`IndexResult` with ``files_processed=0`` (no extraction)
            and ``anchors_embedded`` / ``chunks_written`` populated.

        Raises:
            IndexerError: If the config is missing/invalid or git is unavailable.
        """
        start = time.monotonic()
        try:
            config = self._ensure_config()
        except Exception as exc:
            raise IndexerError(f"Failed to load config: {exc}") from exc

        db = self._ensure_db(config)
        embedder = self._ensure_embedder(config)
        git_provider = self._ensure_git_context(config)

        anchors_embedded = 0
        chunks_written = 0
        pruned_paths: set[str] = set()

        with db.acquire_write_lock():
            try:
                git_ctx = git_provider.get(force_refresh=True)
            except GitContextError as exc:
                raise IndexerError(f"Git context unavailable: {exc}") from exc

            prior_meta = db.read_index_metadata()
            all_files_rel = {
                p.relative_to(self._repo_root).as_posix() for p in self.discover_files()
            }

            # ── Step 1: Prune anchors with missing source files ──────────
            all_anchors = db.list_anchors()
            remaining: list[Anchor] = []
            for anchor in all_anchors:
                if anchor.path not in all_files_rel:
                    db.delete_anchor(anchor.id)
                    pruned_paths.add(anchor.path)
                else:
                    remaining.append(anchor)

            # ── Step 2: Vector-table dimension reconciliation ────────────
            #
            # init_schema is idempotent + dim-aware: it only drops+recreates
            # chunks_vec when the on-disk dimensionality differs from the
            # supplied value (db.py §7.2.1).  Calling it here ensures the
            # vec table matches new_dims regardless of how the db was
            # constructed (Indexer.__init__ may receive a pre-initialized
            # db whose vec table is still at old_dims).  Calling
            # recreate_vector_table unconditionally instead would NULL
            # out batch progress committed on a prior crashed reembed
            # retry — review-w2l HIGH bug #1.
            new_dims = embedder.dimensions
            db.init_schema(embedding_dimensions=new_dims)

            # ── Step 3 + 4: Re-embed in batches; metadata in final batch ─
            conn = db._conn
            total = len(remaining)

            # Determine what the updated metadata will look like.
            indexed_branch = git_ctx.branch or f"detached-{git_ctx.head_short}"
            updated_meta_kwargs: dict[str, object] = {
                "indexed_git_head": prior_meta.indexed_git_head if prior_meta else git_ctx.head_sha,
                "indexed_git_tree_hash": prior_meta.indexed_git_tree_hash if prior_meta else None,
                "indexed_branch": indexed_branch,
                "indexed_file_manifest": prior_meta.indexed_file_manifest if prior_meta else {},
                "config_hash": prior_meta.config_hash
                if prior_meta
                else compute_config_hash(config),
                "embedding_provider": embedder.provider,
                "embedding_model": embedder.model_name,
                "embedding_dimensions": embedder.dimensions,
                "tokenizer_version": embedder.tokenizer_version,
            }

            for batch_start in range(0, max(total, 1), _REEMBED_BATCH_SIZE):
                batch = remaining[batch_start : batch_start + _REEMBED_BATCH_SIZE]
                is_final = (batch_start + _REEMBED_BATCH_SIZE) >= total

                with conn:
                    for anchor in batch:
                        # Load existing chunk texts from DB (avoids re-reading source files).
                        existing_chunks = db.list_chunks(anchor.id)
                        overview_text = _overview_text(anchor)
                        all_texts = [overview_text] + [c.text for c in existing_chunks]
                        all_embeddings = embedder.encode(all_texts)
                        overview_emb = all_embeddings[0]
                        chunk_embs = all_embeddings[1:]

                        # Update overview embedding on the anchor row.
                        conn.execute(
                            "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
                            (overview_emb, anchor.id),
                        )

                        # Re-insert chunk embeddings into chunks_vec.
                        # vec0 doesn't support INSERT OR REPLACE; use DELETE + INSERT.
                        for chunk, emb_blob in zip(existing_chunks, chunk_embs, strict=False):
                            rowid_row = conn.execute(
                                "SELECT id FROM chunks WHERE parent_id = ? AND chunk_index = ?",
                                (anchor.id, chunk.chunk_index),
                            ).fetchone()
                            if rowid_row is not None:
                                rowid: int = rowid_row[0]
                                conn.execute(
                                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                                    (emb_blob, rowid),
                                )
                                conn.execute(
                                    "DELETE FROM chunks_vec WHERE rowid = ?",
                                    (rowid,),
                                )
                                conn.execute(
                                    "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                                    (rowid, emb_blob),
                                )
                                chunks_written += 1
                        anchors_embedded += 1

                    # Commit updated index_metadata only in the final batch.
                    if is_final:
                        _write_meta_in_txn(conn, updated_meta_kwargs)

            # Edge case: no anchors remaining — write metadata now (loop skips it).
            if total == 0:
                with conn:
                    _write_meta_in_txn(conn, updated_meta_kwargs)

        elapsed = time.monotonic() - start
        return IndexResult(
            files_processed=0,
            anchors_extracted=0,
            anchors_embedded=anchors_embedded,
            chunks_written=chunks_written,
            files_pruned=len(pruned_paths),
            elapsed_seconds=elapsed,
        )


# ─── Module-level helpers (not part of the public API) ───────────────────────


def _file_content_hash(path: Path) -> str:
    """Return ``sha256:<hex>`` over the canonicalized content of *path*.

    Uses the same §5.4 canonicalization steps as anchor content hashes so
    that CRLF-only changes do not trigger a spurious incremental reindex.

    Returns a zero-hash on I/O error (the file will be re-processed next run
    once the error clears).
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return compute_content_hash(canonicalize_content(raw))
    except OSError:
        return "sha256:" + "0" * 64


def _overview_text(anchor: Anchor) -> str:
    """Build the §3.4 overview text for the parent anchor's overview embedding.

    SECTION anchors: ``"<heading path>\\n<first ~200 tokens of content_text>"``
    CODE / CODE_IN_DOC anchors: the full ``content_text`` (they are atomic).

    Args:
        anchor: The parent anchor.

    Returns:
        A string suitable for embedding as the anchor's overview vector.
    """
    if anchor.type == AnchorType.SECTION and anchor.heading_path:
        heading = " ".join(anchor.heading_path)
        body = anchor.content_text[:_OVERVIEW_CHARS]
        return f"{heading}\n{body}"
    return anchor.content_text


def _extract_file_anchors(
    path: Path,
    target: ExtractionTarget,
    config: Config,
    repo_root: Path,
) -> list[Anchor]:
    """Dispatch to the appropriate extractor and return anchors.

    Logs a warning and returns an empty list on I/O errors so a single bad
    file does not abort the entire index run.

    Args:
        path:      Absolute path to the file.
        target:    Classification from :meth:`~Indexer.classify_for_extraction`.
        config:    Validated repo config.
        repo_root: Repository root used for repo-relative path computation.

    Returns:
        List of extracted :class:`~scry.models.Anchor` objects.
    """
    if target.kind == "markdown":
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Skipping %s (read error): %s", path, exc)
            return []
        frontmatter, _ = parse_frontmatter(text)
        return extract_markdown(
            path,
            repo_root,
            frontmatter=frontmatter,
            config=config.sections,
            max_file_size_bytes=config.index.max_file_size_bytes,
        )

    if target.kind == "code" and target.language is not None:
        return extract_code_symbols(
            path,
            repo_root,
            language=target.language,
            config=config.code_anchors,
            max_file_size_bytes=config.index.max_file_size_bytes,
        )

    return []  # kind == 'skip'


def _process_anchor(
    anchor: Anchor,
    db: ScryDB,
    embedder: Embedder,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> int:
    """Embed a single anchor and write it atomically (§7.3 invariant 2).

    Opens **one** SQLite transaction that:
      1. Upserts the anchor row (via ``_upsert_anchor_in_txn``).
      2. Stores the overview embedding on the anchor row.
      3. Deletes old chunk rows (triggers FTS5 cleanup via ``chunks_bd``).
      4. Deletes old ``chunks_vec`` entries.
      5. Inserts new chunk rows (triggers FTS5 registration via ``chunks_ai``).
      6. Inserts new ``chunks_vec`` entries.

    This is the only place in the codebase that accesses ``ScryDB._conn``
    directly; see the module docstring for the rationale.

    Args:
        anchor:        Parent anchor to embed and write.
        db:            Open writable :class:`~scry.store.db.ScryDB`.
        embedder:      Backend that produces embedding vectors.
        max_tokens:    Sub-chunk size budget (§3.4).
        overlap_tokens: Sub-chunk overlap (§3.4).

    Returns:
        Number of sub-chunk rows written.
    """
    sub_chunks = chunk_anchor(anchor, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    overview = _overview_text(anchor)
    all_texts = [overview] + [c.text for c in sub_chunks]
    all_embeddings = embedder.encode(all_texts)
    overview_emb = all_embeddings[0]
    chunk_embeddings = all_embeddings[1:]

    now = _now_iso()
    conn = db._conn

    with conn:
        # 1. Upsert anchor.  db.py now applies the unsigned→signed simhash
        # conversion internally (review-w2l HIGH fix #2 — was previously
        # only applied here, leaving db.upsert_anchor unsafe for any
        # other caller and corrupting fingerprint round-trips since
        # _row_to_anchor had no inverse).  No anchor copy needed.
        _upsert_anchor_in_txn(conn, anchor, now)

        # 2. Store the overview embedding computed above.
        conn.execute(
            "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
            (overview_emb, anchor.id),
        )

        # 3 & 4. Clean up stale chunks (FTS5 via trigger; vec manually).
        old_rowids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM chunks WHERE parent_id = ?", (anchor.id,)
            ).fetchall()
        ]
        for rowid in old_rowids:
            conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (rowid,))
        conn.execute("DELETE FROM chunks WHERE parent_id = ?", (anchor.id,))

        # 5 & 6. Insert new chunks + their vectors.
        for chunk, emb_blob in zip(sub_chunks, chunk_embeddings, strict=True):
            conn.execute(
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
                    chunk.parent_content_hash,
                    emb_blob,
                    chunk.overlap_with_prev,
                ),
            )
            new_rowid: int = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (new_rowid, emb_blob),
            )

    return len(sub_chunks)


def _write_meta_in_txn(
    conn: sqlite3.Connection,
    kwargs: dict[str, object],
) -> None:
    """Write (upsert) ``index_metadata`` inside an already-open transaction.

    Called from within the final reembed batch's ``with conn:`` block so the
    metadata update is atomic with the last batch of chunk writes (§7.2.1).

    Args:
        conn:   Open ``sqlite3.Connection`` (transaction already started by
                the caller's ``with conn:``).
        kwargs: Keyword arguments matching :class:`~scry.models.IndexMetadata`
                field names.
    """
    now_str = datetime.now(UTC).isoformat()
    manifest_json = json.dumps(kwargs.get("indexed_file_manifest", {}))
    conn.execute(
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
            kwargs["indexed_git_head"],
            kwargs.get("indexed_git_tree_hash"),
            kwargs["indexed_branch"],
            manifest_json,
            kwargs["config_hash"],
            kwargs["embedding_provider"],
            kwargs["embedding_model"],
            kwargs["embedding_dimensions"],
            kwargs.get("tokenizer_version"),
            now_str,
        ),
    )
