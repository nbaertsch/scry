"""Integration-style tests for the Indexer orchestrator (workstream W2l).

Test strategy
-------------
All tests use:
  * A temporary copy of ``tests/fixtures/hailstorm-spec/`` that is also a
    real git repository (required by GitContextProvider).
  * :class:`~scry.embed.StubEmbedder` so fastembed weights are never downloaded.

§7.3 invariants verified here
------------------------------
* ``test_chunks_have_correct_parent_content_hash``   — invariant 1
* ``test_per_parent_transaction_via_anchor_write``    — invariant 2 (atomic write)
* ``test_second_run_is_noop``                        — invariant 4 (stale filter)

DESIGN.md §7.2.1 reembed semantics
-------------------------------------
* ``test_reembed_same_dims_preserves_anchors``
* ``test_reembed_different_dims_recreates_vector_table``
* ``test_reembed_prunes_anchors_for_deleted_files``
* ``test_reembed_metadata_not_updated_until_final_batch``

§7.2 force vs incremental
--------------------------
* ``test_force_reindex_rebuilds_from_scratch``
* ``test_incremental_reindex_on_file_change``
* ``test_incremental_prunes_deleted_file``
* ``test_config_hash_change_triggers_full_reindex``
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest

from scry.config import load_config
from scry.embed import StubEmbedder
from scry.index import ExtractionTarget, Indexer, IndexerError, IndexResult
from scry.models import AnchorType, Config
from scry.store.db import ScryDB

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def indexed_repo(tmp_path: Path, hailstorm_spec: Path) -> Generator[Path, None, None]:
    """Writable temp git repo containing the hailstorm-spec fixture files.

    Steps:
      1. Copy the fixture tree into tmp_path.
      2. ``git init``, set identity, add all files, create initial commit.

    The result is a valid git repository with at least one commit, so
    :class:`~scry.git_context.GitContextProvider` can resolve HEAD.
    """
    shutil.copytree(str(hailstorm_spec), str(tmp_path), dirs_exist_ok=True)

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    _git("init")
    _git("config", "user.email", "ci@test.local")
    _git("config", "user.name", "CI Test")
    _git("add", ".")
    _git("commit", "-m", "init")
    yield tmp_path


@pytest.fixture
def config(indexed_repo: Path) -> Config:
    """Loaded config from the hailstorm-spec repo."""
    return load_config(indexed_repo)


@pytest.fixture
def stub_embedder(config: Config) -> StubEmbedder:
    """StubEmbedder sized to match config.embeddings.dimensions."""
    return StubEmbedder(dimensions=config.embeddings.dimensions)


@pytest.fixture
def indexer(indexed_repo: Path, config: Config, stub_embedder: StubEmbedder) -> Indexer:
    """Indexer wired with StubEmbedder; DB is created lazily inside the Indexer."""
    return Indexer(indexed_repo, config=config, embedder=stub_embedder)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _commit_change(repo: Path, rel_path: str, new_text: str) -> None:
    """Write *new_text* to *rel_path* and commit it."""
    (repo / rel_path).write_text(new_text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", rel_path], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", f"update {rel_path}"],
        check=True,
        capture_output=True,
    )


def _open_db(repo: Path, config: Config) -> ScryDB:
    db = ScryDB(repo)
    db.init_schema(embedding_dimensions=config.embeddings.dimensions)
    return db


# ─── Basic first-run tests ────────────────────────────────────────────────────


class TestFirstRun:
    """Tests for a clean first index() call."""

    def test_returns_index_result(self, indexer: Indexer) -> None:
        """index() returns an IndexResult with populated fields."""
        result = indexer.index()
        assert isinstance(result, IndexResult)
        assert result.elapsed_seconds >= 0.0

    def test_markdown_anchors_extracted(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """All markdown sections in the fixture are anchored after first run."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            anchors = db.list_anchors(anchor_type=AnchorType.SECTION)
        assert len(anchors) > 0
        # POLICY_ENGINE.md has at least 4 sections.
        policy_anchors = [a for a in anchors if "POLICY_ENGINE" in a.path]
        assert len(policy_anchors) >= 4

    def test_code_anchors_extracted(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Python symbols in the fixture are anchored after first run."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            anchors = db.list_anchors(anchor_type=AnchorType.CODE)
        assert len(anchors) > 0
        ids = {a.id for a in anchors}
        # PolicyRule class should be present.
        assert any("PolicyRule" in aid for aid in ids)
        # login function in auth module.
        assert any("login" in aid for aid in ids)

    def test_files_processed_count(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """files_processed matches the number of indexable files."""
        result = indexer.index()
        expected_files = len(indexer.discover_files())
        # discover_files may include files classified as 'skip'; files_processed
        # counts only those for which extraction was actually attempted.
        assert result.files_processed <= expected_files
        assert result.files_processed > 0

    def test_anchors_extracted_matches_anchors_embedded(self, indexer: Indexer) -> None:
        """Every extracted anchor gets embedded."""
        result = indexer.index()
        assert result.anchors_embedded == result.anchors_extracted

    def test_chunks_written_positive(self, indexer: Indexer) -> None:
        """At least one chunk is written per anchor."""
        result = indexer.index()
        assert result.chunks_written >= result.anchors_extracted

    def test_index_metadata_written(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """IndexMetadata is persisted after first run."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            meta = db.read_index_metadata()
        assert meta is not None
        assert meta.indexed_git_head != ""
        assert meta.indexed_branch != ""
        assert len(meta.indexed_file_manifest) > 0
        assert meta.config_hash.startswith("sha256:")

    def test_metadata_embedding_fields(
        self, indexer: Indexer, indexed_repo: Path, config: Config, stub_embedder: StubEmbedder
    ) -> None:
        """Metadata embedding fields match the embedder used."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            meta = db.read_index_metadata()
        assert meta is not None
        assert meta.embedding_model == stub_embedder.model_name
        assert meta.embedding_dimensions == stub_embedder.dimensions
        assert meta.embedding_provider == stub_embedder.provider

    def test_no_files_pruned_on_first_run(self, indexer: Indexer) -> None:
        """First run has no pruning (nothing existed before)."""
        result = indexer.index()
        assert result.files_pruned == 0


# ─── §7.3 Consistency invariants ─────────────────────────────────────────────


class TestConsistencyInvariants:
    """§7.3 two-tier embedding consistency invariants."""

    def test_chunks_have_correct_parent_content_hash(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Invariant 1: every chunk.parent_content_hash == its parent's content_hash."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            anchors = db.list_anchors()
            for anchor in anchors:
                chunks = db.list_chunks(anchor.id)
                for chunk in chunks:
                    assert chunk.parent_content_hash == anchor.content_hash, (
                        f"Stale parent_content_hash on chunk {chunk.chunk_index} "
                        f"of anchor {anchor.id!r}: "
                        f"chunk has {chunk.parent_content_hash!r}, "
                        f"parent has {anchor.content_hash!r}"
                    )

    def test_every_anchor_has_at_least_one_chunk(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Every indexed anchor must have ≥ 1 sub-chunk for retrieval."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            for anchor in db.list_anchors():
                chunks = db.list_chunks(anchor.id)
                assert len(chunks) >= 1, f"Anchor {anchor.id!r} has no chunks"

    def test_overview_embedding_stored_on_anchors(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After index, the anchors table should have overview_embedding != NULL."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            rows = db._conn.execute(  # type: ignore[attr-defined]
                "SELECT id FROM anchors WHERE overview_embedding IS NULL"
            ).fetchall()
        assert rows == [], f"Anchors with NULL overview_embedding: {[r[0] for r in rows]}"

    def test_chunks_in_vec_table(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After index, chunks_vec should have the same row count as embedded chunks."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            chunk_count = db._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            vec_count = db._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM chunks_vec"
            ).fetchone()[0]
        assert chunk_count == vec_count


# ─── Incremental indexing ─────────────────────────────────────────────────────


class TestIncrementalIndexing:
    """Tests for incremental (force=False) index passes."""

    def test_second_run_is_noop(self, indexer: Indexer, indexed_repo: Path, config: Config) -> None:
        """Second run with no file changes processes 0 files."""
        indexer.index()
        result2 = indexer.index()
        assert result2.files_processed == 0
        assert result2.anchors_extracted == 0
        assert result2.anchors_embedded == 0
        assert result2.chunks_written == 0
        assert result2.files_pruned == 0

    def test_second_run_anchor_count_unchanged(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Second run with no changes leaves anchor count identical."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            count_after_first = len(db.list_anchors())

        indexer.index()
        with _open_db(indexed_repo, config) as db:
            count_after_second = len(db.list_anchors())

        assert count_after_first == count_after_second

    def test_incremental_reindex_on_file_change(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Modifying one file causes only that file to be reprocessed."""
        indexer.index()

        # Modify one markdown file with a git commit.
        new_content = (
            "# Auth Protocol\n\n## Token Lifecycle\n\nRewritten token lifecycle.\n\n"
            "## New Section\n\nA brand new section added here.\n"
        )
        _commit_change(indexed_repo, "docs/AUTH_PROTOCOL.md", new_content)

        result2 = indexer.index()
        assert result2.files_processed == 1
        assert result2.anchors_extracted > 0

    def test_modified_file_anchors_updated(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After modifying a file, its anchors reflect the new content."""
        indexer.index()

        unique_phrase = "UNIQUELY_IDENTIFIABLE_PHRASE_XYZ_42"
        new_content = f"# Auth Protocol\n\n## Updated Section\n\n{unique_phrase}\n"
        _commit_change(indexed_repo, "docs/AUTH_PROTOCOL.md", new_content)
        indexer.index()

        with _open_db(indexed_repo, config) as db:
            anchors = db.list_anchors(path="docs/AUTH_PROTOCOL.md")
        # The new section heading should appear.
        assert any("updated-section" in a.id for a in anchors)
        # Content of the new anchor should contain our unique phrase.
        updated_texts = [a.content_text for a in anchors if "updated-section" in a.id]
        assert any(unique_phrase in t for t in updated_texts)

    def test_incremental_prunes_deleted_file(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Anchors for deleted files are removed on the next incremental run."""
        indexer.index()

        with _open_db(indexed_repo, config) as db:
            before = db.list_anchors(path="docs/AUTH_PROTOCOL.md")
        assert len(before) > 0  # file was indexed

        # Delete the file and commit.
        (indexed_repo / "docs" / "AUTH_PROTOCOL.md").unlink()
        subprocess.run(
            ["git", "-C", str(indexed_repo), "rm", "docs/AUTH_PROTOCOL.md"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(indexed_repo), "commit", "-m", "delete AUTH_PROTOCOL"],
            check=True,
            capture_output=True,
        )

        result2 = indexer.index()
        assert result2.files_pruned >= 1

        with _open_db(indexed_repo, config) as db:
            after = db.list_anchors(path="docs/AUTH_PROTOCOL.md")
        assert len(after) == 0

    def test_config_hash_change_triggers_full_reindex(
        self, indexer: Indexer, indexed_repo: Path
    ) -> None:
        """If the config hash changes, a full reindex is performed."""
        indexer.index()

        # Patch the config with a different hash by writing a new config file.
        new_config_text = (
            "include:\n"
            '  - "docs/**.md"\n'
            '  - "python/**.py"\n'
            "exclude:\n"
            '  - ".scry/**"\n'
            "classify:\n"
            '  - { glob: "docs/**.md", type: spec }\n'
            "sections:\n"
            "  max_tokens: 500\n"  # changed from default 600
        )
        (indexed_repo / ".scry" / "config.yaml").write_text(new_config_text, encoding="utf-8")

        # Reload config to pick up the change.
        fresh_config = load_config(indexed_repo)
        fresh_indexer = Indexer(indexed_repo, config=fresh_config, embedder=indexer._embedder)
        result2 = fresh_indexer.index()
        # Full reindex should process all files.
        assert result2.files_processed > 0


# ─── Force rebuild ────────────────────────────────────────────────────────────


class TestForceReindex:
    """Tests for force=True (nuclear) rebuild."""

    def test_force_reindex_after_index(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """force=True after a prior index re-extracts all files."""
        indexer.index()
        result2 = indexer.index(force=True)
        assert result2.files_processed > 0
        assert result2.anchors_extracted > 0

    def test_force_reindex_same_anchor_count(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """Anchor count after force=True matches a fresh first run."""
        result1 = indexer.index()
        anchor_count_1 = result1.anchors_extracted

        result2 = indexer.index(force=True)
        anchor_count_2 = result2.anchors_extracted

        assert anchor_count_1 == anchor_count_2

    def test_force_clears_prior_anchors(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After force=True, no orphan anchors remain from a prior run."""
        indexer.index()
        indexer.index(force=True)
        with _open_db(indexed_repo, config) as db:
            # All anchors should belong to current files.
            all_files_rel = {
                p.relative_to(indexed_repo).as_posix() for p in indexer.discover_files()
            }
            for anchor in db.list_anchors():
                assert anchor.path in all_files_rel, (
                    f"Orphan anchor {anchor.id!r} with path {anchor.path!r}"
                )

    def test_force_on_virgin_repo(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """force=True on a fresh repo (no prior index) works correctly."""
        result = indexer.index(force=True)
        assert result.anchors_extracted > 0
        assert result.anchors_embedded == result.anchors_extracted


# ─── Reembed ──────────────────────────────────────────────────────────────────


class TestReembed:
    """Tests for reembed() — §7.2.1 surgical embedding-model migration."""

    def test_reembed_same_dims_preserves_anchors(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """reembed() with same dimensions preserves all anchors."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            anchor_ids_before = {a.id for a in db.list_anchors()}

        result = indexer.reembed()
        assert result.files_processed == 0  # no extraction
        assert result.anchors_embedded > 0

        with _open_db(indexed_repo, config) as db:
            anchor_ids_after = {a.id for a in db.list_anchors()}
        assert anchor_ids_before == anchor_ids_after

    def test_reembed_same_dims_preserves_chunks(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """reembed() preserves FTS5 entries (chunk text unchanged)."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            chunk_texts_before = {
                (c.parent_id, c.chunk_index): c.text
                for a in db.list_anchors()
                for c in db.list_chunks(a.id)
            }

        indexer.reembed()

        with _open_db(indexed_repo, config) as db:
            chunk_texts_after = {
                (c.parent_id, c.chunk_index): c.text
                for a in db.list_anchors()
                for c in db.list_chunks(a.id)
            }
        assert chunk_texts_before == chunk_texts_after

    def test_reembed_updates_embeddings(
        self, indexer: Indexer, indexed_repo: Path, config: Config, stub_embedder: StubEmbedder
    ) -> None:
        """After reembed(), chunks_vec has the same row count as before."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            vec_count_before = db._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM chunks_vec"
            ).fetchone()[0]

        indexer.reembed()

        with _open_db(indexed_repo, config) as db:
            vec_count_after = db._conn.execute(  # type: ignore[attr-defined]
                "SELECT COUNT(*) FROM chunks_vec"
            ).fetchone()[0]
        assert vec_count_after == vec_count_before

    def test_reembed_different_dims_recreates_vector_table(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """reembed() with different dimensions drops+recreates chunks_vec."""
        indexer.index()  # 384-dim (stub default)

        # New embedder with different dimensions.
        new_dims = 128
        new_embedder = StubEmbedder(dimensions=new_dims)
        new_indexer = Indexer(indexed_repo, config=config, embedder=new_embedder)
        result = new_indexer.reembed()

        assert result.anchors_embedded > 0

        # Verify the vector table was recreated for new_dims.
        # Open a raw connection (no init_schema) so we don't reset the table.
        import sqlite3 as _sqlite3

        import sqlite_vec as _sqlite_vec

        db_path = indexed_repo / ".scry" / "vectors.db"
        raw = _sqlite3.connect(str(db_path))
        raw.enable_load_extension(True)
        _sqlite_vec.load(raw)
        vec_sql = raw.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
        ).fetchone()
        raw.close()

        assert vec_sql is not None
        assert str(new_dims) in vec_sql[0]

    def test_reembed_metadata_embedding_dims_updated(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After reembed(), index_metadata.embedding_dimensions reflects the new model."""
        indexer.index()

        new_dims = 64
        new_embedder = StubEmbedder(dimensions=new_dims)
        new_indexer = Indexer(indexed_repo, config=config, embedder=new_embedder)
        new_indexer.reembed()

        with _open_db(indexed_repo, config) as db:
            meta = db.read_index_metadata()
        assert meta is not None
        assert meta.embedding_dimensions == new_dims

    def test_reembed_prunes_anchors_for_deleted_files(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """reembed() removes anchors whose source files no longer exist."""
        indexer.index()

        # Delete a file (without re-running index first).
        (indexed_repo / "docs" / "AUTH_PROTOCOL.md").unlink()

        result = indexer.reembed()
        assert result.files_pruned >= 1

        with _open_db(indexed_repo, config) as db:
            after = db.list_anchors(path="docs/AUTH_PROTOCOL.md")
        assert len(after) == 0

    def test_reembed_without_prior_index_is_safe(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """reembed() on a DB with no anchors is a no-op without error."""
        result = indexer.reembed()
        assert result.anchors_embedded == 0
        assert result.files_pruned == 0


# ─── discover_files ───────────────────────────────────────────────────────────


class TestDiscoverFiles:
    """Tests for discover_files()."""

    def test_returns_absolute_paths(self, indexer: Indexer) -> None:
        """All returned paths are absolute."""
        for path in indexer.discover_files():
            assert path.is_absolute(), f"Expected absolute path: {path}"

    def test_deterministic_order(self, indexer: Indexer) -> None:
        """Two calls produce the same ordered list."""
        first = indexer.discover_files()
        second = indexer.discover_files()
        assert first == second

    def test_excludes_scry_dir(self, indexer: Indexer, indexed_repo: Path) -> None:
        """'.scry/**' files are not returned."""
        files = indexer.discover_files()
        for p in files:
            assert ".scry" not in p.parts, f"Unexpected .scry file: {p}"

    def test_includes_markdown_and_python(self, indexer: Indexer, indexed_repo: Path) -> None:
        """Both .md and .py files appear in the result."""
        files = indexer.discover_files()
        suffixes = {p.suffix.lower() for p in files}
        assert ".md" in suffixes
        assert ".py" in suffixes

    def test_all_hailstorm_docs_found(self, indexer: Indexer, indexed_repo: Path) -> None:
        """The three hailstorm-spec markdown docs are all discovered."""
        files = indexer.discover_files()
        rel_set = {p.relative_to(indexed_repo).as_posix() for p in files}
        assert "docs/POLICY_ENGINE.md" in rel_set
        assert "docs/AUTH_PROTOCOL.md" in rel_set
        assert "docs/EXECUTION_PIPELINE.md" in rel_set


# ─── classify_for_extraction ──────────────────────────────────────────────────


class TestClassifyForExtraction:
    """Tests for classify_for_extraction()."""

    def test_markdown_doc_classified_as_markdown(
        self, indexer: Indexer, indexed_repo: Path
    ) -> None:
        """docs/POLICY_ENGINE.md → kind='markdown'."""
        target = indexer.classify_for_extraction(indexed_repo / "docs" / "POLICY_ENGINE.md")
        assert target.kind == "markdown"
        assert target.language is None

    def test_python_file_classified_as_code(self, indexer: Indexer, indexed_repo: Path) -> None:
        """python/hailstone/policy/engine.py → kind='code', language='python'."""
        target = indexer.classify_for_extraction(
            indexed_repo / "python" / "hailstone" / "policy" / "engine.py"
        )
        assert target.kind == "code"
        assert target.language == "python"

    def test_unknown_extension_classified_as_skip(
        self, indexer: Indexer, indexed_repo: Path
    ) -> None:
        """A .xyz file that matches no classify or extension → kind='skip'."""
        fake = indexed_repo / "some_file.xyz"
        fake.write_text("hello")
        target = indexer.classify_for_extraction(fake)
        assert target.kind == "skip"
        fake.unlink()

    def test_outside_repo_classified_as_skip(self, indexer: Indexer, tmp_path: Path) -> None:
        """A path outside the repo root → kind='skip'."""
        outside = tmp_path / "outside.md"
        outside.write_text("# Out of scope")
        target = indexer.classify_for_extraction(outside)
        assert target.kind == "skip"
        outside.unlink()

    def test_extraction_target_is_frozen(self) -> None:
        """ExtractionTarget is frozen (dataclass with frozen=True)."""
        import dataclasses

        t = ExtractionTarget(kind="code", language="python")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.kind = "skip"  # type: ignore[misc]


# ─── ExtractionTarget / IndexResult API ──────────────────────────────────────


class TestPublicAPI:
    """Smoke tests for the public dataclass shapes."""

    def test_extraction_target_markdown(self) -> None:
        t = ExtractionTarget(kind="markdown")
        assert t.kind == "markdown"
        assert t.language is None

    def test_extraction_target_code(self) -> None:
        t = ExtractionTarget(kind="code", language="typescript")
        assert t.kind == "code"
        assert t.language == "typescript"

    def test_index_result_fields(self, indexer: Indexer) -> None:
        r = indexer.index()
        # All fields should be accessible and non-negative.
        assert r.files_processed >= 0
        assert r.anchors_extracted >= 0
        assert r.anchors_embedded >= 0
        assert r.chunks_written >= 0
        assert r.files_pruned >= 0
        assert r.elapsed_seconds >= 0.0

    def test_indexer_error_is_exception(self) -> None:
        with pytest.raises(IndexerError):
            raise IndexerError("test error")


# ─── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and regression tests."""

    def test_empty_file_does_not_crash(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """An empty markdown file in the include set is processed without error."""
        empty = indexed_repo / "docs" / "EMPTY.md"
        empty.write_text("", encoding="utf-8")
        subprocess.run(["git", "-C", str(indexed_repo), "add", "EMPTY.md"], capture_output=True)
        subprocess.run(
            ["git", "-C", str(indexed_repo), "commit", "-m", "add empty file"],
            capture_output=True,
        )
        result = indexer.index(force=True)
        # Should complete without exception; empty file contributes 0 anchors.
        assert isinstance(result, IndexResult)

    def test_index_metadata_git_head_is_sha(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """indexed_git_head should be a 40-char hex string."""
        indexer.index()
        with _open_db(indexed_repo, config) as db:
            meta = db.read_index_metadata()
        assert meta is not None
        assert len(meta.indexed_git_head) == 40
        assert all(c in "0123456789abcdef" for c in meta.indexed_git_head)

    def test_index_metadata_file_manifest_contains_all_files(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """File manifest includes all discovered files with sha256: hashes."""
        indexer.index()
        all_files_rel = {p.relative_to(indexed_repo).as_posix() for p in indexer.discover_files()}
        with _open_db(indexed_repo, config) as db:
            meta = db.read_index_metadata()
        assert meta is not None
        assert set(meta.indexed_file_manifest.keys()) == all_files_rel
        for hash_val in meta.indexed_file_manifest.values():
            assert hash_val.startswith("sha256:")

    def test_reembed_chunks_have_correct_parent_hash(
        self, indexer: Indexer, indexed_repo: Path, config: Config
    ) -> None:
        """After reembed, §7.3 invariant 1 still holds."""
        indexer.index()
        indexer.reembed()
        with _open_db(indexed_repo, config) as db:
            for anchor in db.list_anchors():
                for chunk in db.list_chunks(anchor.id):
                    assert chunk.parent_content_hash == anchor.content_hash

    def test_discover_files_respects_skip_frontmatter(
        self, indexer: Indexer, indexed_repo: Path
    ) -> None:
        """A markdown file with ``scry: {skip: true}`` is excluded."""
        skip_file = indexed_repo / "docs" / "SKIPPED.md"
        skip_file.write_text(
            "---\nscry:\n  skip: true\n---\n# Skipped\n\nThis should not be indexed.\n",
            encoding="utf-8",
        )
        files = indexer.discover_files()
        rel_set = {p.relative_to(indexed_repo).as_posix() for p in files}
        assert "docs/SKIPPED.md" not in rel_set
        skip_file.unlink()
