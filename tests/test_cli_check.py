"""Tests for ``scry check`` CLI extensions (W4c).

Covers §5.2 v3.1 flags:
  - ``--ignore-lsp-error``: drift-unknown excluded from exit code; still in output
  - ``--strict``: any non-fresh link → exit 1
  - ``--json``: shorthand for --format json
  - Exit codes: 0 = clean, 1 = drift, 2 = operational error

All tests use Click CliRunner + a lightweight DB fixture that writes
anchors and links directly so we can inject specific drift statuses
without a full ``scry index`` run.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scry.cli import main
from scry.config import load_config
from scry.embed import StubEmbedder
from scry.index import Indexer
from scry.models import (
    AnchorType,
    LinkRecord,
    TransitiveHashStatus,
    new_event_id,
    new_link_id,
)
from scry.store.db import ScryDB
from scry.store.links import LinkStore

# ---------------------------------------------------------------------------
# Helpers shared with test_cli.py
# ---------------------------------------------------------------------------

_STUB_ENV = {"SCRY_EMBEDDER": "stub"}

_HA = "sha256:" + "a" * 64  # "original" hash — both endpoints unchanged
_DIMS = 4


def _run(
    runner: CliRunner,
    args: list[str],
    *,
    repo: Path,
    input: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> Any:
    """Invoke ``scry`` with *args* as if run from *repo*."""
    env = {**_STUB_ENV, **(env_extra or {})}
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return runner.invoke(main, args, catch_exceptions=False, env=env, input=input)
    finally:
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def hailstorm_spec(fixture_dir: Path) -> Path:
    p = fixture_dir / "hailstorm-spec"
    if not p.exists():
        pytest.skip("hailstorm-spec fixture not yet created (W1d)")
    return p


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def indexed_repo(tmp_path: Path, hailstorm_spec: Path) -> Path:
    """Temp git repo with hailstorm-spec files + initial commit (no vectors.db)."""
    import shutil

    shutil.copytree(str(hailstorm_spec), str(tmp_path), dirs_exist_ok=True)

    def _git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    _git("init")
    _git("config", "user.email", "ci@test.local")
    _git("config", "user.name", "CI Test")
    _git("add", ".")
    _git("commit", "-m", "init")
    return tmp_path


@pytest.fixture
def indexed_and_built_repo(indexed_repo: Path) -> Path:
    """Run ``scry index`` (stub embedder) against the temp repo."""
    config = load_config(indexed_repo)
    indexer = Indexer(
        indexed_repo,
        config=config,
        embedder=StubEmbedder(dimensions=config.embeddings.dimensions),
    )
    indexer.index(force=True)
    return indexed_repo


@pytest.fixture
def repo_with_drift_unknown(indexed_and_built_repo: Path) -> Path:
    """Extend the indexed repo with one drift-unknown link.

    Injects a SECTION anchor and a CODE anchor with
    ``transitive_hash_status=lsp_error`` (unchanged content hashes).
    Then writes a baseline link between them so ``evaluate_all_drift``
    will return ``drift-unknown``.
    """
    repo = indexed_and_built_repo
    from_id = "docs/spec.md::drift-unknown-test-section"
    to_id = "src/app.py::drift_unknown_fn"

    with ScryDB(repo) as db:
        # SECTION from-anchor.
        from scry.models import Anchor

        db.upsert_anchor(
            Anchor(
                id=from_id,
                type=AnchorType.SECTION,
                path="docs/spec.md",
                content_text="drift unknown test section",
                content_hash=_HA,
                fingerprint_simhash=0xDEAD,
            )
        )
        # CODE to-anchor with lsp_error.
        db.upsert_anchor(
            Anchor(
                id=to_id,
                type=AnchorType.CODE,
                path="src/app.py",
                content_text="def drift_unknown_fn(): pass",
                content_hash=_HA,
                fingerprint_simhash=0xBEEF,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )

    # Write a baseline link.
    store = LinkStore(repo)
    record = LinkRecord.model_validate(
        {
            "op": "upsert",
            "link_id": new_link_id(),
            "event_id": new_event_id(),
            "from": from_id,
            "from_type": "section",
            "to": to_id,
            "to_type": "code",
            "type": "implements",
            "from_content_hash": _HA,
            "to_content_hash": _HA,
        }
    )
    store.append_baseline(record)
    return repo


@pytest.fixture
def repo_with_spec_changed(indexed_and_built_repo: Path) -> Path:
    """Extend the indexed repo with one spec-changed link (non-fresh, not drift-unknown)."""
    repo = indexed_and_built_repo
    _HB = "sha256:" + "b" * 64  # changed hash
    from_id = "docs/spec.md::spec-changed-section"
    to_id = "src/app.py::unchanged_fn"

    with ScryDB(repo) as db:
        from scry.models import Anchor

        # from-anchor with CHANGED hash.
        db.upsert_anchor(
            Anchor(
                id=from_id,
                type=AnchorType.SECTION,
                path="docs/spec.md",
                content_text="changed spec section",
                content_hash=_HB,  # current hash differs from stored _HA below
                fingerprint_simhash=0xDEAD,
            )
        )
        db.upsert_anchor(
            Anchor(
                id=to_id,
                type=AnchorType.CODE,
                path="src/app.py",
                content_text="def unchanged_fn(): pass",
                content_hash=_HA,
                fingerprint_simhash=0xBEEF,
                transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
            )
        )

    store = LinkStore(repo)
    record = LinkRecord.model_validate(
        {
            "op": "upsert",
            "link_id": new_link_id(),
            "event_id": new_event_id(),
            "from": from_id,
            "from_type": "section",
            "to": to_id,
            "to_type": "code",
            "type": "implements",
            "from_content_hash": _HA,  # stored hash = _HA, but current = _HB → changed
            "to_content_hash": _HA,
        }
    )
    store.append_baseline(record)
    return repo


# ---------------------------------------------------------------------------
# Tests: --ignore-lsp-error
# ---------------------------------------------------------------------------


class TestIgnoreLspError:
    """§5.1 v3.1: drift-unknown from lsp_error excluded when --ignore-lsp-error."""

    def test_drift_unknown_causes_exit_1_in_strict_mode(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """Without --ignore-lsp-error, drift-unknown causes exit 1 under --strict."""
        result = _run(runner, ["check", "--strict"], repo=repo_with_drift_unknown)
        assert result.exit_code == 1, result.output

    def test_ignore_lsp_error_suppresses_exit_1_in_strict_mode(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """With --ignore-lsp-error, drift-unknown does NOT cause exit 1 under --strict."""
        result = _run(
            runner,
            ["check", "--strict", "--ignore-lsp-error"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 0, result.output

    def test_drift_unknown_still_in_output_when_ignored(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """drift_unknown count appears in JSON output even when --ignore-lsp-error is set."""
        result = _run(
            runner,
            ["check", "--json", "--ignore-lsp-error"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["counts"]["drift_unknown"] >= 1, "drift_unknown must be reported in counts"

    def test_drift_unknown_in_md_output_when_ignored(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """drift_unknown row appears in markdown table even with --ignore-lsp-error."""
        result = _run(
            runner,
            ["check", "--ignore-lsp-error"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 0, result.output
        assert "drift_unknown" in result.output

    def test_ignore_lsp_error_with_ci_and_perfect_score(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """--ignore-lsp-error removes drift_unknown weight so score passes --ci --drift-min."""
        # With only drift_unknown links (weight 0.3 each), the raw drift_score < 100.
        # --ignore-lsp-error should exclude that weight so effective score = 100 → pass.
        result = _run(
            runner,
            ["check", "--ci", "--drift-min", "99", "--ignore-lsp-error"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 0, result.output

    def test_without_ignore_lsp_error_ci_fails_on_drift_unknown(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """Without --ignore-lsp-error, drift_unknown lowers drift_score, failing --ci."""
        # drift_score < 100 because of drift_unknown weight → fails --drift-min 99.
        result = _run(
            runner,
            ["check", "--ci", "--drift-min", "99"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 1, result.output


# ---------------------------------------------------------------------------
# Tests: --strict
# ---------------------------------------------------------------------------


class TestStrict:
    """--strict: any non-fresh link → exit 1."""

    def test_strict_clean_repo_exits_0(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """--strict exits 0 when all links are fresh (no links → trivially clean)."""
        result = _run(runner, ["check", "--strict"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output

    def test_strict_exits_1_on_spec_changed(
        self, runner: CliRunner, repo_with_spec_changed: Path
    ) -> None:
        """--strict exits 1 when a non-fresh (spec-changed) link exists."""
        result = _run(runner, ["check", "--strict"], repo=repo_with_spec_changed)
        assert result.exit_code == 1, result.output

    def test_strict_exits_1_on_drift_unknown(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """--strict exits 1 when drift-unknown link exists (without --ignore-lsp-error)."""
        result = _run(runner, ["check", "--strict"], repo=repo_with_drift_unknown)
        assert result.exit_code == 1, result.output

    def test_strict_with_ignore_lsp_error_passes_on_drift_unknown_only(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """--strict --ignore-lsp-error exits 0 when only non-fresh is drift-unknown."""
        result = _run(
            runner,
            ["check", "--strict", "--ignore-lsp-error"],
            repo=repo_with_drift_unknown,
        )
        assert result.exit_code == 0, result.output

    def test_strict_error_message_on_stderr(
        self, runner: CliRunner, repo_with_spec_changed: Path
    ) -> None:
        """--strict prints a FAIL message (mixed into output by CliRunner) on failure."""
        result = _run(runner, ["check", "--strict"], repo=repo_with_spec_changed)
        assert result.exit_code == 1
        # CliRunner mixes stderr into output by default.
        assert "FAIL" in result.output


# ---------------------------------------------------------------------------
# Tests: --json flag (shorthand for --format json)
# ---------------------------------------------------------------------------


class TestJsonFlag:
    """--json is shorthand for --format json."""

    def test_json_flag_produces_valid_json(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """--json flag outputs valid JSON with required keys."""
        result = _run(runner, ["check", "--json"], repo=indexed_and_built_repo)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "drift_score" in data
        assert "coverage_score" in data
        assert "counts" in data

    def test_json_flag_same_as_format_json(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """--json and --format json produce identical output."""
        r1 = _run(runner, ["check", "--json"], repo=indexed_and_built_repo)
        r2 = _run(runner, ["check", "--format", "json"], repo=indexed_and_built_repo)
        assert r1.exit_code == r2.exit_code == 0
        assert json.loads(r1.output) == json.loads(r2.output)

    def test_json_flag_with_drift_unknown_shows_count(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """--json output includes drift_unknown in counts."""
        result = _run(runner, ["check", "--json"], repo=repo_with_drift_unknown)
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["counts"]["drift_unknown"] >= 1


# ---------------------------------------------------------------------------
# Tests: exit code convention (0=clean, 1=drift, 2=error)
# ---------------------------------------------------------------------------


class TestExitCodes:
    """Exit code semantics per §5.2 v3.1."""

    def test_missing_db_exits_2(self, runner: CliRunner, indexed_repo: Path) -> None:
        """Missing vectors.db is an operational error → exit 2."""
        result = _run(runner, ["check"], repo=indexed_repo)
        assert result.exit_code == 2

    def test_ci_threshold_violation_exits_1(
        self, runner: CliRunner, repo_with_spec_changed: Path
    ) -> None:
        """Drift threshold violation is drift → exit 1."""
        result = _run(
            runner,
            ["check", "--ci", "--drift-min", "99"],
            repo=repo_with_spec_changed,
        )
        assert result.exit_code == 1

    def test_clean_repo_exits_0(self, runner: CliRunner, indexed_and_built_repo: Path) -> None:
        """Clean repo (no links) → exit 0."""
        result = _run(runner, ["check"], repo=indexed_and_built_repo)
        assert result.exit_code == 0

    def test_drift_unknown_md_output_shows_count(
        self, runner: CliRunner, repo_with_drift_unknown: Path
    ) -> None:
        """Markdown output includes drift_unknown row."""
        result = _run(runner, ["check"], repo=repo_with_drift_unknown)
        assert result.exit_code == 0, result.output
        assert "drift_unknown" in result.output

# uat-r5-5 pr-d noise
