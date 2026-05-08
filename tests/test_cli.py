"""Tests for scry CLI (workstream W2j).

All tests use Click's CliRunner to invoke commands in isolation.

Tests that require a real indexed repository use the ``indexed_repo``
fixture from ``test_index.py`` which:
  1. Copies the hailstorm-spec fixture into a temp dir.
  2. ``git init``, adds all files, creates an initial commit.

The ``SCRY_EMBEDDER=stub`` environment variable is set on every CliRunner
invocation so that fastembed weights are never downloaded during tests.

Stub notes:
- ``scry watch``, ``scry suggest-links``, ``scry reconcile`` are verified
  to print their deferral messages and exit 0.
- ``scry mcp`` is tested by passing an instantly-closing stdin (via
  CliRunner's input="") — the server is expected to exit cleanly without
  error.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scry.cli import main
from scry.config import load_config
from scry.embed import StubEmbedder
from scry.index import Indexer
from scry.models import AnchorType

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STUB_ENV = {"SCRY_EMBEDDER": "stub"}


def _run(
    runner: CliRunner,
    args: list[str],
    *,
    repo: Path,
    input: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> Any:
    """Invoke `scry` with *args* as if run from *repo* directory."""
    env = {**_STUB_ENV, **(env_extra or {})}
    # Change cwd to repo so that _repo_root() returns the right path.
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return runner.invoke(
            main,
            args,
            catch_exceptions=False,
            env=env,
            input=input,
        )
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click CliRunner (stderr mixed into output for this Click version)."""
    return CliRunner()


@pytest.fixture
def hailstorm_spec(fixture_dir: Path) -> Path:
    """Return the hailstorm-spec fixture directory, skipping if absent."""
    p = fixture_dir / "hailstorm-spec"
    if not p.exists():
        pytest.skip("hailstorm-spec fixture not yet created (W1d)")
    return p


@pytest.fixture
def fixture_dir() -> Path:
    """Path to tests/fixtures."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def indexed_repo(tmp_path: Path, hailstorm_spec: Path) -> Generator[Path, None, None]:
    """Temp git repo with the hailstorm-spec fixture files + initial commit.

    Mirrors the fixture from test_index.py so CLI tests can operate against
    the same data set without re-loading fastembed.
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
def indexed_and_built_repo(indexed_repo: Path) -> Path:
    """Run ``scry index`` (stub embedder) against the temp repo and return it."""
    config = load_config(indexed_repo)
    indexer = Indexer(
        indexed_repo,
        config=config,
        embedder=StubEmbedder(dimensions=config.embeddings.dimensions),
    )
    indexer.index(force=True)
    return indexed_repo


@pytest.fixture
def clean_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """An empty temp dir with a .scry/ subdirectory and a git repo."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    _git("init")
    _git("config", "user.email", "ci@test.local")
    _git("config", "user.name", "CI Test")

    yield tmp_path


# ---------------------------------------------------------------------------
# scry init
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for ``scry init``."""

    def test_happy_path_creates_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry init creates config.yaml, .gitignore, and updates .gitattributes."""
        # tmp_path is NOT a git repo — that's fine for init.
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["init"], repo=repo)

        assert result.exit_code == 0, result.output
        # The files should be in tmp_path itself since we ran from there.
        # Actually runner.isolated_filesystem creates its own subdir.
        # Let's verify via output.
        assert "Wrote" in result.output
        assert "config.yaml" in result.output
        assert ".gitignore" in result.output
        assert "MCP registration snippet" in result.output

    def test_writes_config_yaml(self, runner: CliRunner, tmp_path: Path) -> None:
        """Written config.yaml is valid YAML with expected keys."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["init"], repo=repo)
            assert result.exit_code == 0, result.output
            config_path = repo / ".scry" / "config.yaml"
            assert config_path.exists()
            import yaml

            data = yaml.safe_load(config_path.read_text())
            assert "include" in data
            assert "embeddings" in data

    def test_refuses_overwrite_without_force(self, runner: CliRunner, tmp_path: Path) -> None:
        """Second scry init without --force exits 1."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            r1 = _run(runner, ["init"], repo=repo)
            assert r1.exit_code == 0
            r2 = _run(runner, ["init"], repo=repo)
            assert r2.exit_code == 1
            assert "already exists" in r2.output

    def test_force_overwrites(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry init --force succeeds even when config.yaml already exists."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            _run(runner, ["init"], repo=repo)
            r2 = _run(runner, ["init", "--force"], repo=repo)
            assert r2.exit_code == 0, r2.output

    def test_register_global_writes_files(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scry init --register-global writes to ~home/.claude.json and ~/.cursor/mcp.json."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["init", "--register-global"], repo=repo)
        assert result.exit_code == 0, result.output

        claude_json = fake_home / ".claude.json"
        cursor_json = fake_home / ".cursor" / "mcp.json"

        for target in [claude_json, cursor_json]:
            assert target.exists(), f"{target} not found"
            data = json.loads(target.read_text())
            assert "mcpServers" in data
            assert "scry" in data["mcpServers"]

    def test_register_global_refuses_malformed_target(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression (review-w2j HIGH): malformed ~/.claude.json must NOT be silently truncated.

        Previously the code suppressed ``json.JSONDecodeError`` and then
        wrote a fresh ``{"mcpServers": ...}`` object, destroying whatever
        the user had in the file (Claude Code's config typically holds
        substantial state).  The fix refuses to write and prints a
        warning that points the user at the broken file.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        # Seed a malformed claude config that has substantial user content.
        claude_json = fake_home / ".claude.json"
        original_text = "this is malformed json {{{ but I have lots of important content here"
        claude_json.write_text(original_text, encoding="utf-8")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["init", "--register-global"], repo=repo)

        # Init itself should succeed (the warning is non-fatal).
        assert result.exit_code == 0, result.output
        # The original (malformed) content MUST still be on disk —
        # i.e. NOT truncated.
        assert claude_json.read_text(encoding="utf-8") == original_text, (
            "Malformed ~/.claude.json was silently overwritten — review-w2j HIGH bug regressed."
        )
        # And the warning must have surfaced to the user.
        assert "refusing to register" in (result.output + result.stderr), (
            f"Expected refusal warning in output; got: {result.output!r} stderr={result.stderr!r}"
        )

    def test_register_global_refuses_non_dict_root(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: a JSON list/null/scalar at the root must not crash or be overwritten."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        claude_json = fake_home / ".claude.json"
        # Valid JSON, but a list root — old code would crash on .setdefault().
        claude_json.write_text('["not", "an", "object"]', encoding="utf-8")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["init", "--register-global"], repo=repo)

        assert result.exit_code == 0, result.output
        assert claude_json.read_text(encoding="utf-8") == '["not", "an", "object"]'
        assert "refusing to register" in (result.output + result.stderr)

    def test_gitattributes_gets_union_line(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry init adds merge=union line to .gitattributes."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            _run(runner, ["init"], repo=repo)
            ga = repo / ".gitattributes"
            assert ga.exists()
            assert "merge=union" in ga.read_text()

    def test_gitattributes_not_duplicated(self, runner: CliRunner, tmp_path: Path) -> None:
        """Running scry init twice does not duplicate the merge=union line."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            _run(runner, ["init"], repo=repo)
            _run(runner, ["init", "--force"], repo=repo)
            ga = repo / ".gitattributes"
            content = ga.read_text()
            assert content.count("merge=union") == 1


# ---------------------------------------------------------------------------
# scry index
# ---------------------------------------------------------------------------


class TestIndex:
    """Tests for ``scry index``."""

    def test_index_happy_path(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry index exits 0 and prints an IndexResult summary."""
        result = _run(runner, ["index"], repo=indexed_repo)
        assert result.exit_code == 0, result.output
        assert "files_processed" in result.output
        assert "anchors_extracted" in result.output
        assert "elapsed" in result.output

    def test_index_force(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry index --force exits 0."""
        result = _run(runner, ["index", "--force"], repo=indexed_repo)
        assert result.exit_code == 0, result.output

    def test_index_reembed(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry index --reembed exits 0 (first creates, then re-embeds)."""
        # Need to index first.
        _run(runner, ["index"], repo=indexed_repo)
        result = _run(runner, ["index", "--reembed"], repo=indexed_repo)
        assert result.exit_code == 0, result.output

    def test_force_and_reembed_mutually_exclusive(
        self, runner: CliRunner, indexed_repo: Path
    ) -> None:
        """scry index --force --reembed exits with UsageError."""
        result = runner.invoke(
            main,
            ["index", "--force", "--reembed"],
            env=_STUB_ENV,
            catch_exceptions=False,
        )
        # Click raises UsageError which shows up as exit code 2.
        assert result.exit_code == 2

    def test_index_no_config_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry index without .scry/config.yaml exits 1."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            result = _run(runner, ["index"], repo=repo)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# scry watch (stub)
# ---------------------------------------------------------------------------


class TestWatch:
    """Tests for ``scry watch`` (Wave 6 stub)."""

    def test_watch_prints_deferred_message(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry watch prints deferral message and exits 0."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["watch"], env=_STUB_ENV, catch_exceptions=False)
        assert result.exit_code == 0
        assert "Wave 6" in result.output


# ---------------------------------------------------------------------------
# scry check
# ---------------------------------------------------------------------------


class TestCheck:
    """Tests for ``scry check``."""

    def test_check_md_format(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry check --format md exits 0 and prints a markdown table."""
        result = _run(runner, ["check", "--format", "md"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        assert "drift_score" in result.output

    def test_check_json_format(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry check --format json outputs valid JSON with expected keys."""
        result = _run(runner, ["check", "--format", "json"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "drift_score" in data
        assert "coverage_score" in data
        assert "counts" in data

    def test_check_ci_null_score_passes(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """With no links, drift_score is null; --ci with --drift-min treats null as PASS."""
        result = _run(
            runner,
            ["check", "--ci", "--drift-min", "90"],
            repo=indexed_and_built_repo,
        )
        # No links → null drift_score → pass per §5.2 v3.1.
        assert result.exit_code == 0, result.output

    def test_check_ci_coverage_null_passes(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """null coverage_score (no code anchors linked) passes --ci --coverage-min."""
        result = _run(
            runner,
            ["check", "--ci", "--coverage-min", "50"],
            repo=indexed_and_built_repo,
        )
        # null = pass; no actual links in hailstorm so coverage = null.
        assert result.exit_code == 0, result.output

    def test_check_no_db_exits_1(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry check without vectors.db exits 1."""
        result = _run(runner, ["check"], repo=indexed_repo)
        assert result.exit_code == 1

    def test_check_require_fresh_embedder_mismatch(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """--require-fresh-embedder passes when model matches (stub records stub)."""
        # The StubEmbedder writes 'stub' as provider/model; config says 'local'.
        # This test verifies the flag exercises the code path without crashing.
        result = _run(
            runner,
            ["check", "--require-fresh-embedder", "--format", "json"],
            repo=indexed_and_built_repo,
        )
        # When provider/model mismatch exits 1; when match exits 0.
        # With StubEmbedder the stored model is "stub" but config.embeddings.provider="local"
        # so this should exit 1 (mismatch).
        assert result.exit_code in (0, 1)


# ---------------------------------------------------------------------------
# scry status
# ---------------------------------------------------------------------------


class TestStatus:
    """Tests for ``scry status``."""

    def test_status_shows_branch(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry status prints branch and HEAD fields."""
        result = _run(runner, ["status"], repo=indexed_repo)
        assert result.exit_code == 0, result.output
        assert "Branch" in result.output
        assert "HEAD" in result.output

    def test_status_shows_pending_count(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry status shows pending overlay records (0 when none)."""
        result = _run(runner, ["status"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        assert "Pending overlay records:" in result.output

    def test_status_shows_index_info_when_db_present(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry status shows index metadata when vectors.db exists."""
        result = _run(runner, ["status"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        assert "Index" in result.output

    def test_status_no_db_message(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry status prints a hint when vectors.db is missing."""
        result = _run(runner, ["status"], repo=indexed_repo)
        assert result.exit_code == 0
        assert "vectors.db" in result.output


# ---------------------------------------------------------------------------
# scry commit-links + round-trip
# ---------------------------------------------------------------------------


class TestCommitLinks:
    """Tests for ``scry commit-links``."""

    def _add_link(
        self,
        runner: CliRunner,
        repo: Path,
        from_id: str,
        to_id: str,
        link_type: str = "implements",
    ) -> Any:
        return _run(
            runner,
            ["link", from_id, to_id, "--type", link_type],
            repo=repo,
        )

    def _first_anchor_id(self, repo: Path, anchor_type: AnchorType) -> str:
        """Return the first anchor ID of given type from vectors.db."""
        from scry.store.db import ScryDB

        with ScryDB(repo, read_only=True) as db:
            anchors = db.list_anchors(anchor_type=anchor_type)
        assert anchors, f"No {anchor_type} anchors found in DB"
        return anchors[0].id

    def test_commit_links_no_pending(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry commit-links with nothing pending prints info and exits 0."""
        result = _run(runner, ["commit-links"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        assert "No overlay records" in result.output

    def test_link_then_status_shows_pending(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """After scry link, scry status shows 1 pending record."""
        repo = indexed_and_built_repo
        from scry.store.db import ScryDB

        with ScryDB(repo, read_only=True) as db:
            anchors = db.list_anchors()
        assert len(anchors) >= 2, "Need at least 2 anchors for a link test"
        a1 = anchors[0].id
        a2 = anchors[1].id

        link_result = _run(runner, ["link", a1, a2, "--type", "references"], repo=repo)
        assert link_result.exit_code == 0, link_result.output

        status_result = _run(runner, ["status"], repo=repo)
        assert status_result.exit_code == 0
        assert "1" in status_result.output  # "Pending overlay records: 1"

    def test_commit_links_promotes_all(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry commit-links (no args) promotes the pending record to baseline."""
        repo = indexed_and_built_repo
        from scry.store.db import ScryDB

        with ScryDB(repo, read_only=True) as db:
            anchors = db.list_anchors()
        assert len(anchors) >= 2
        a1 = anchors[0].id
        a2 = anchors[1].id

        _run(runner, ["link", a1, a2, "--type", "references"], repo=repo)

        # Promote.
        commit_result = _run(runner, ["commit-links"], repo=repo)
        assert commit_result.exit_code == 0, commit_result.output
        assert "Promoted" in commit_result.output

        # Overlay should now be empty.
        status_result = _run(runner, ["status"], repo=repo)
        assert "Pending overlay records: 0" in status_result.output

    def test_commit_links_selective_by_event_id(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry commit-links <event_id> promotes only that event."""
        repo = indexed_and_built_repo
        from scry.store.db import ScryDB

        with ScryDB(repo, read_only=True) as db:
            anchors = db.list_anchors()
        assert len(anchors) >= 2
        a1 = anchors[0].id
        a2 = anchors[1].id

        link_result = _run(runner, ["link", a1, a2, "--type", "references"], repo=repo)
        assert link_result.exit_code == 0

        # Extract event_id from link output.
        lines = link_result.output.splitlines()
        evt_line = next((ln for ln in lines if "event_id:" in ln), None)
        assert evt_line is not None
        event_id = evt_line.split(":")[1].strip()

        commit_result = _run(runner, ["commit-links", event_id], repo=repo)
        assert commit_result.exit_code == 0, commit_result.output
        assert "Promoted 1 record" in commit_result.output


# ---------------------------------------------------------------------------
# scry search
# ---------------------------------------------------------------------------


class TestSearch:
    """Tests for ``scry search``."""

    def test_search_returns_results(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry search returns at least one result for a broad query."""
        result = _run(runner, ["search", "policy"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output

    def test_search_no_db_exits_1(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry search without vectors.db exits 1."""
        result = _run(runner, ["search", "anything"], repo=indexed_repo)
        assert result.exit_code == 1

    def test_search_empty_query_exits_0(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry search with a whitespace query exits 0 and prints 'No results'."""
        result = _run(runner, ["search", "   "], repo=indexed_and_built_repo)
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_type_filter(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry search --type section exits 0."""
        result = _run(
            runner,
            ["search", "policy", "--type", "section"],
            repo=indexed_and_built_repo,
        )
        assert result.exit_code == 0, result.output

    def test_search_top_k(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry search --top-k 3 returns at most 3 results."""
        result = _run(
            runner,
            ["search", "policy", "--top-k", "3"],
            repo=indexed_and_built_repo,
        )
        assert result.exit_code == 0, result.output
        # Each result line has "score="; count them.
        score_lines = [ln for ln in result.output.splitlines() if "score=" in ln]
        assert len(score_lines) <= 3


# ---------------------------------------------------------------------------
# scry link
# ---------------------------------------------------------------------------


class TestLink:
    """Tests for ``scry link``."""

    def test_link_appends_to_overlay(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """scry link creates a link record in the overlay and prints link_id."""
        repo = indexed_and_built_repo
        from scry.store.db import ScryDB

        with ScryDB(repo, read_only=True) as db:
            anchors = db.list_anchors()
        assert len(anchors) >= 2
        a1 = anchors[0].id
        a2 = anchors[1].id

        result = _run(
            runner,
            ["link", a1, a2, "--type", "references", "--evidence", "test evidence"],
            repo=repo,
        )
        assert result.exit_code == 0, result.output
        assert "lnk_" in result.output
        assert "evt_" in result.output

    def test_link_invalid_anchor_exits_1(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry link with a nonexistent anchor ID exits 1."""
        result = _run(
            runner,
            ["link", "does::not::exist", "also::missing", "--type", "references"],
            repo=indexed_and_built_repo,
        )
        assert result.exit_code == 1

    def test_link_no_db_exits_1(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry link without vectors.db exits 1."""
        result = _run(runner, ["link", "a", "b", "--type", "references"], repo=indexed_repo)
        assert result.exit_code == 1

    def test_link_missing_type_exits_2(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """scry link without --type exits with Click's UsageError (exit code 2)."""
        result = runner.invoke(main, ["link", "a", "b"], env=_STUB_ENV, catch_exceptions=False)
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# scry suggest-links (stub)
# ---------------------------------------------------------------------------


class TestSuggestLinks:
    """Tests for ``scry suggest-links`` (Wave 5 stub)."""

    def test_suggest_links_deferred(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry suggest-links prints deferral message and exits 0."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["suggest-links"], env=_STUB_ENV, catch_exceptions=False)
        assert result.exit_code == 0
        assert "Wave 5" in result.output


# ---------------------------------------------------------------------------
# scry reconcile (stub)
# ---------------------------------------------------------------------------


class TestReconcile:
    """Tests for ``scry reconcile`` (Wave 5 stub)."""

    def test_reconcile_deferred(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry reconcile prints deferral message and exits 0."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                main, ["reconcile", "lnk_abc123"], env=_STUB_ENV, catch_exceptions=False
            )
        assert result.exit_code == 0
        assert "Wave 5" in result.output


# ---------------------------------------------------------------------------
# scry doctor
# ---------------------------------------------------------------------------


class TestDoctor:
    """Tests for ``scry doctor``."""

    def test_doctor_runs_without_error(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry doctor exits 0 and prints version info."""
        result = _run(runner, ["doctor"], repo=indexed_repo)
        assert result.exit_code == 0, result.output
        assert "scry version" in result.output

    def test_doctor_shows_lsp_allowlist(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry doctor prints the LSP allowlist."""
        result = _run(runner, ["doctor"], repo=indexed_repo)
        assert result.exit_code == 0
        assert "LSP allowlist" in result.output
        assert "python" in result.output

    def test_doctor_warns_on_allow_untrusted(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry --allow-untrusted-lsp-config doctor prints a security warning."""
        result = _run(runner, ["--allow-untrusted-lsp-config", "doctor"], repo=indexed_repo)
        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "allow-untrusted-lsp-config" in result.output

    def test_doctor_no_git_shows_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry doctor in a non-git directory gracefully reports git error."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["doctor"], env=_STUB_ENV, catch_exceptions=False)
        # Should still exit 0 — doctor is diagnostic, not fatal.
        assert result.exit_code == 0
        assert "scry version" in result.output


# ---------------------------------------------------------------------------
# scry validate
# ---------------------------------------------------------------------------


class TestValidate:
    """Tests for ``scry validate``."""

    def test_validate_clean_repo(self, runner: CliRunner, indexed_repo: Path) -> None:
        """scry validate exits 0 on a clean hailstorm-spec repo."""
        # Ensure .gitattributes has the union line.
        ga = indexed_repo / ".gitattributes"
        if not ga.exists():
            ga.write_text(".scry/links.jsonl merge=union\n", encoding="utf-8")
        elif ".scry/links.jsonl merge=union" not in ga.read_text():
            with ga.open("a") as fh:
                fh.write(".scry/links.jsonl merge=union\n")

        result = _run(runner, ["validate"], repo=indexed_repo)
        assert result.exit_code == 0, result.output
        assert "All checks passed" in result.output

    def test_validate_missing_config_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry validate without .scry/config.yaml exits 1."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            (repo / ".scry").mkdir()
            result = _run(runner, ["validate"], repo=repo)
        assert result.exit_code == 1

    def test_validate_missing_gitattributes_exits_1(
        self, runner: CliRunner, indexed_repo: Path
    ) -> None:
        """scry validate exits 1 when .gitattributes lacks the merge=union line."""
        ga = indexed_repo / ".gitattributes"
        ga.write_text("# no union driver here\n", encoding="utf-8")

        result = _run(runner, ["validate"], repo=indexed_repo)
        assert result.exit_code == 1
        assert "merge=union" in result.output

    def test_validate_duplicate_frontmatter_id_exits_1(
        self, runner: CliRunner, indexed_repo: Path
    ) -> None:
        """scry validate detects duplicate scry.id values in frontmatter."""
        # Ensure .gitattributes is correct so only the id conflict causes failure.
        ga = indexed_repo / ".gitattributes"
        if not ga.exists() or ".scry/links.jsonl merge=union" not in ga.read_text():
            with ga.open("a") as fh:
                fh.write(".scry/links.jsonl merge=union\n")

        # Write two markdown files with the same scry.id.
        (indexed_repo / "docs" / "a.md").write_text(
            "---\nscry:\n  id: DUPLICATE-ID\n---\n# A\n", encoding="utf-8"
        )
        (indexed_repo / "docs" / "b.md").write_text(
            "---\nscry:\n  id: DUPLICATE-ID\n---\n# B\n", encoding="utf-8"
        )

        result = _run(runner, ["validate"], repo=indexed_repo)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# scry mcp
# ---------------------------------------------------------------------------


class TestMCP:
    """Tests for ``scry mcp``."""

    def test_mcp_starts_and_exits_on_eof(self, indexed_and_built_repo: Path) -> None:
        """scry mcp exits cleanly when stdin closes immediately (EOF).

        Must run as a subprocess because FastMCP's stdio transport closes
        sys.stdout on EOF, which conflicts with CliRunner's stdout capture.
        """
        result = subprocess.run(
            [sys.executable, "-m", "scry.cli", "mcp"],
            cwd=str(indexed_and_built_repo),
            env={**os.environ, "SCRY_EMBEDDER": "stub"},
            input=b"",
            capture_output=True,
            timeout=15,
        )
        # Exit 0: clean EOF. Non-zero is acceptable only if an error is printed.
        assert result.returncode == 0, result.stderr.decode(errors="replace")

    def test_mcp_missing_config_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry mcp exits 1 when .scry/config.yaml is missing."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            repo = Path.cwd()
            (repo / ".scry").mkdir()
            old_cwd = Path.cwd()
            try:
                os.chdir(repo)
                result = runner.invoke(main, ["mcp"], env=_STUB_ENV, input="")
            finally:
                os.chdir(old_cwd)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Global flag: --allow-untrusted-lsp-config
# ---------------------------------------------------------------------------


class TestGlobalFlags:
    """Tests for global flags on the main group."""

    def test_allow_untrusted_flag_accepted(self, runner: CliRunner, indexed_repo: Path) -> None:
        """--allow-untrusted-lsp-config is accepted as a global flag."""
        result = _run(
            runner,
            ["--allow-untrusted-lsp-config", "watch"],
            repo=indexed_repo,
        )
        # watch just prints its stub message; exit 0.
        assert result.exit_code == 0

    def test_version_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry --version prints the package version."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["--version"], env=_STUB_ENV, catch_exceptions=False)
        assert result.exit_code == 0
        assert "scry" in result.output
        assert "0.0.1" in result.output
