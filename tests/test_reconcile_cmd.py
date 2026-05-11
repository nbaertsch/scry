"""Tests for scry reconcile CLI command (workstream W5c).

Tests the ``scry reconcile <link_id>`` command implementation in
``src/scry/cmd_reconcile.py``.  Covers:

- Fresh-link short-circuit: exit 0 with "no drift" message
- Drifted link: LLM is called and diff is printed
- ``--all``: iterates over all drifted links
- ``--json``: machine-readable JSON output
- ``--apply --yes``: patch written to disk via git apply
- Missing LINK_ID (no --all): exit 2 with clear error
- Unknown link ID: exit 2 with "not found"
- LLM error: exit 2 with message surfaced gracefully
- Non-actionable drift status (broken-source): exit 2 with explanation

IMPORTANT: this is test_reconcile_cmd.py — NOT test_reconcile.py
(which covers W4d auto-reconcile state machine, a separate concept).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scry.cli import main
from scry.cmd_reconcile import (
    ReconcileError,
    _apply_git_patch,
    _compute_diff,
    _git_show,
    _read_current,
    _validate_diff_paths,
    propose_patch,
)
from scry.drift import DriftEvaluation
from scry.git_context import GitContextProvider
from scry.llm import LLMNetworkError, LLMRequest, LLMResponse
from scry.models import (
    Anchor,
    AnchorType,
    DriftStatus,
    Link,
    LinkType,
    TransitiveHashStatus,
    new_event_id,
    new_link_id,
)
from scry.store.db import ScryDB
from scry.store.links import LinkRecord, LinkStore
from scry.store.overlay import OverlayManager

# ─── Constants ────────────────────────────────────────────────────────────────

_STUB_ENV = {"SCRY_EMBEDDER": "stub"}

# Baseline hash (stored in link at creation time) and a different current hash.
_HA = "sha256:" + "a" * 64
_HB = "sha256:" + "b" * 64

_CANNED_DIFF = (
    "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old spec content\n+patched spec content\n"
)
_CANNED_JSON = json.dumps(
    {
        "target": "spec",
        "diff": _CANNED_DIFF,
        "explanation": "Update spec to match current code behaviour.",
    }
)

_MINIMAL_CONFIG = (
    "include:\n  - '**/*.md'\n  - '**/*.py'\nclassify:\n  - {glob: '**/*.md', type: spec}\n"
)

# ─── Mock providers ───────────────────────────────────────────────────────────


class _MockProvider:
    """Returns a canned reconcile JSON response."""

    name = "mock"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text=_CANNED_JSON,
            model="mock",
            provider="mock",
            usage=None,
            finish_reason="stop",
        )


class _ErrorProvider:
    """Raises LLMNetworkError on every call."""

    name = "error"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        raise LLMNetworkError("cannot connect to LLM provider")


# ─── Shared helpers ───────────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _run(
    runner: CliRunner,
    args: list[str],
    *,
    repo: Path,
    input: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> Any:
    """Invoke ``scry`` with *args* as if run from *repo* directory."""
    env = {**_STUB_ENV, **(env_extra or {})}
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return runner.invoke(main, args, catch_exceptions=False, env=env, input=input)
    finally:
        os.chdir(old_cwd)


def _write_anchor(db: ScryDB, anchor_id: str, path: str, content_hash: str) -> None:
    """Write a minimal SECTION anchor to the DB."""
    db.upsert_anchor(
        Anchor(
            id=anchor_id,
            type=AnchorType.SECTION,
            path=path,
            content_text="test content",
            content_hash=content_hash,
            fingerprint_simhash=0,
        )
    )


def _write_code_anchor(db: ScryDB, anchor_id: str, path: str, content_hash: str) -> None:
    """Write a minimal CODE anchor to the DB."""
    db.upsert_anchor(
        Anchor(
            id=anchor_id,
            type=AnchorType.CODE,
            path=path,
            content_text="def foo(): pass",
            content_hash=content_hash,
            fingerprint_simhash=0,
            transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
        )
    )


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mini_git_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Minimal git repo with spec.md committed (for git show tests)."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("old spec content\n", encoding="utf-8")

    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "config.yaml").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@test.local")
    _git(tmp_path, "config", "user.name", "CI Test")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")

    yield tmp_path


@pytest.fixture
def fresh_link_repo(mini_git_repo: Path) -> tuple[Path, str]:
    """Repo with one FRESH link (link hash == DB anchor hash)."""
    repo = mini_git_repo
    from_id = "spec.md::from-section"
    to_id = "spec.md::to-section"

    with ScryDB(repo) as db:
        db.init_schema(embedding_dimensions=4)
        _write_anchor(db, from_id, "spec.md", _HA)
        _write_anchor(db, to_id, "spec.md", _HA)

    link_id = new_link_id()
    store = LinkStore(repo)
    store.append_baseline(
        LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": link_id,
                "event_id": new_event_id(),
                "from": from_id,
                "from_type": "section",
                "to": to_id,
                "to_type": "section",
                "type": "implements",
                "from_content_hash": _HA,  # matches DB → fresh
                "to_content_hash": _HA,
            }
        )
    )
    return repo, link_id


@pytest.fixture
def drifted_link_repo(mini_git_repo: Path) -> tuple[Path, str]:
    """Repo with one spec-changed drifted link (link hash _HA, DB hash _HB)."""
    repo = mini_git_repo

    # Fetch the commit SHA for git-show tests.
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = result.stdout.strip()

    from_id = "spec.md::spec-section"
    to_id = "spec.md::code-section"

    with ScryDB(repo) as db:
        db.init_schema(embedding_dimensions=4)
        # DB has _HB (current, changed); link will store _HA (baseline) → drift
        _write_anchor(db, from_id, "spec.md", _HB)
        _write_anchor(db, to_id, "spec.md", _HA)

    link_id = new_link_id()
    store = LinkStore(repo)
    store.append_baseline(
        LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": link_id,
                "event_id": new_event_id(),
                "from": from_id,
                "from_type": "section",
                "to": to_id,
                "to_type": "section",
                "type": "implements",
                "from_content_hash": _HA,  # stored at link creation: _HA
                "to_content_hash": _HA,  # DB has _HA → unchanged
                "commit_sha": commit_sha,
            }
        )
    )
    return repo, link_id


@pytest.fixture
def two_drifted_links_repo(mini_git_repo: Path) -> tuple[Path, list[str]]:
    """Repo with TWO spec-changed drifted links for --all tests."""
    repo = mini_git_repo

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    commit_sha = result.stdout.strip()

    with ScryDB(repo) as db:
        db.init_schema(embedding_dimensions=4)
        for i in range(1, 3):
            _write_anchor(db, f"spec.md::section-{i}", "spec.md", _HB)
            _write_anchor(db, f"spec.md::target-{i}", "spec.md", _HA)

    link_ids = []
    store = LinkStore(repo)
    for i in range(1, 3):
        lnk_id = new_link_id()
        link_ids.append(lnk_id)
        store.append_baseline(
            LinkRecord.model_validate(
                {
                    "op": "upsert",
                    "link_id": lnk_id,
                    "event_id": new_event_id(),
                    "from": f"spec.md::section-{i}",
                    "from_type": "section",
                    "to": f"spec.md::target-{i}",
                    "to_type": "section",
                    "type": "implements",
                    "from_content_hash": _HA,
                    "to_content_hash": _HA,
                    "commit_sha": commit_sha,
                }
            )
        )
    return repo, link_ids


# ─── Unit tests for pure helpers ──────────────────────────────────────────────


class TestGitHelpers:
    """Unit tests for git utility functions."""

    def test_git_show_returns_content(self, mini_git_repo: Path) -> None:
        result = subprocess.run(
            ["git", "-C", str(mini_git_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        content = _git_show(mini_git_repo, sha, "spec.md")
        assert content == "old spec content\n"

    def test_git_show_returns_none_for_bad_commit(self, mini_git_repo: Path) -> None:
        content = _git_show(mini_git_repo, "deadbeef" * 5, "spec.md")
        assert content is None

    def test_git_show_returns_none_for_bad_path(self, mini_git_repo: Path) -> None:
        result = subprocess.run(
            ["git", "-C", str(mini_git_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        content = _git_show(mini_git_repo, sha, "nonexistent.md")
        assert content is None

    def test_read_current_happy_path(self, mini_git_repo: Path) -> None:
        content = _read_current(mini_git_repo, "spec.md")
        assert content == "old spec content\n"

    def test_read_current_missing_file(self, mini_git_repo: Path) -> None:
        content = _read_current(mini_git_repo, "nope.md")
        assert content is None

    def test_compute_diff_returns_diff(self) -> None:
        diff = _compute_diff("a.txt", "hello\n", "world\n")
        assert "--- a/a.txt" in diff
        assert "+++ b/a.txt" in diff
        assert "-hello" in diff
        assert "+world" in diff

    def test_compute_diff_empty_when_same(self) -> None:
        diff = _compute_diff("a.txt", "hello\n", "hello\n")
        assert diff == ""


class TestApplyGitPatch:
    """Tests for _apply_git_patch."""

    def test_apply_patch_modifies_file(self, mini_git_repo: Path) -> None:
        diff = (
            "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old spec content\n+patched spec content\n"
        )
        _apply_git_patch(mini_git_repo, diff)
        result = (mini_git_repo / "spec.md").read_text(encoding="utf-8")
        assert result == "patched spec content\n"

    def test_apply_bad_patch_raises(self, mini_git_repo: Path) -> None:
        bad_diff = (
            "--- a/spec.md\n"
            "+++ b/spec.md\n"
            "@@ -1 +1 @@\n"
            "-this line does not exist in the file\n"
            "+replacement\n"
        )
        with pytest.raises(ReconcileError, match="git apply failed"):
            _apply_git_patch(mini_git_repo, bad_diff)


# ─── Unit tests for propose_patch ─────────────────────────────────────────────


class TestProposePatch:
    """Unit tests for the async propose_patch function."""

    def _make_evaluation(
        self,
        *,
        from_hash: str = _HA,
        to_hash: str = _HA,
        drift: DriftStatus = DriftStatus.SPEC_CHANGED,
        commit_sha: str | None = None,
    ) -> DriftEvaluation:
        link = Link(
            link_id=new_link_id(),
            from_id="spec.md::from",
            from_type=AnchorType.SECTION,
            to_id="spec.md::to",
            to_type=AnchorType.SECTION,
            type=LinkType.IMPLEMENTS,
            from_content_hash=_HA,
            to_content_hash=_HA,
            commit_sha=commit_sha,
            last_event_id=new_event_id(),
        )
        return DriftEvaluation(link=link, drift_status=drift, semantic_drift=None)

    def _make_anchor(self, anchor_id: str, path: str, content: str = "test") -> Anchor:
        return Anchor(
            id=anchor_id,
            type=AnchorType.SECTION,
            path=path,
            content_text=content,
            content_hash=_HA,
            fingerprint_simhash=0,
        )

    def test_propose_patch_calls_provider(self, tmp_path: Path) -> None:
        """propose_patch returns a ReconcileResult from the LLM response."""
        ev = self._make_evaluation()
        from_anchor = self._make_anchor("spec.md::from", "spec.md")
        to_anchor = self._make_anchor("spec.md::to", "spec.md")

        import asyncio

        result = asyncio.run(
            propose_patch(
                ev,
                repo_root=tmp_path,
                from_anchor=from_anchor,
                to_anchor=to_anchor,
                provider=_MockProvider(),
            )
        )
        assert result.target == "spec"
        assert result.diff == _CANNED_DIFF
        assert "spec" in result.explanation.lower() or result.explanation

    def test_propose_patch_llm_error_raises(self, tmp_path: Path) -> None:
        """LLM failure raises ReconcileError."""
        ev = self._make_evaluation()
        from_anchor = self._make_anchor("spec.md::from", "spec.md")
        to_anchor = self._make_anchor("spec.md::to", "spec.md")

        import asyncio

        with pytest.raises(ReconcileError, match="LLM error"):
            asyncio.run(
                propose_patch(
                    ev,
                    repo_root=tmp_path,
                    from_anchor=from_anchor,
                    to_anchor=to_anchor,
                    provider=_ErrorProvider(),
                )
            )

    def test_propose_patch_invalid_target_raises(self, tmp_path: Path) -> None:
        """LLM response with invalid target raises ReconcileError."""

        class _BadTargetProvider:
            name = "bad"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    text=json.dumps({"target": "invalid", "diff": "", "explanation": "x"}),
                    model="bad",
                    provider="bad",
                    usage=None,
                    finish_reason="stop",
                )

        ev = self._make_evaluation()
        from_anchor = self._make_anchor("spec.md::from", "spec.md")
        to_anchor = self._make_anchor("spec.md::to", "spec.md")

        import asyncio

        with pytest.raises(ReconcileError, match="unexpected target"):
            asyncio.run(
                propose_patch(
                    ev,
                    repo_root=tmp_path,
                    from_anchor=from_anchor,
                    to_anchor=to_anchor,
                    provider=_BadTargetProvider(),
                )
            )


# ─── CLI integration tests ─────────────────────────────────────────────────────


class TestReconcileCmdFreshLink:
    """scry reconcile <link_id> on a FRESH link."""

    def test_fresh_link_exits_0_no_drift(
        self, runner: CliRunner, fresh_link_repo: tuple[Path, str]
    ) -> None:
        """Fresh link prints 'no drift' and exits 0 without calling the LLM."""
        repo, link_id = fresh_link_repo

        call_count = 0

        class _CountingProvider:
            name = "counting"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                return LLMResponse(
                    text=_CANNED_JSON,
                    model="x",
                    provider="x",
                    usage=None,
                    finish_reason="stop",
                )

        with patch("scry.cmd_reconcile.make_provider", return_value=_CountingProvider()):
            result = _run(runner, ["reconcile", link_id], repo=repo)

        assert result.exit_code == 0, result.output
        assert "fresh" in result.output.lower()
        assert call_count == 0  # LLM never called for fresh links


class TestReconcileCmdDrifted:
    """scry reconcile <link_id> on a DRIFTED link."""

    def test_drifted_link_calls_llm_prints_diff(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """Drifted link calls LLM and prints the proposed diff."""
        repo, link_id = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", link_id], repo=repo)

        assert result.exit_code == 0, result.output
        assert "spec" in result.output.lower()
        assert "old spec content" in result.output or "Proposed diff" in result.output

    def test_json_output_valid_json(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """--json flag outputs valid JSON with the expected fields."""
        repo, link_id = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", link_id, "--json"], repo=repo)

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        entry = parsed[0]
        assert entry["link_id"] == link_id
        assert entry["target"] == "spec"
        assert entry["drift_status"] == "spec-changed"
        assert "diff" in entry
        assert "explanation" in entry

    def test_apply_yes_writes_patch(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """--apply --yes applies the patch to the working tree."""
        repo, link_id = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", link_id, "--apply", "--yes"], repo=repo)

        assert result.exit_code == 0, result.output
        # The canned diff replaces "old spec content" with "patched spec content"
        updated = (repo / "spec.md").read_text(encoding="utf-8")
        assert updated == "patched spec content\n"

    def test_apply_without_yes_prompts(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """--apply without --yes shows a confirmation prompt."""
        repo, link_id = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            # Respond "n" to the confirmation prompt
            result = _run(runner, ["reconcile", link_id, "--apply"], repo=repo, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Apply patch" in result.output or "kipped" in result.output
        # File should NOT have been changed
        content = (repo / "spec.md").read_text(encoding="utf-8")
        assert content == "old spec content\n"


class TestReconcileCmdAll:
    """scry reconcile --all iterates over drifted links."""

    def test_all_reconciles_every_drifted_link(
        self, runner: CliRunner, two_drifted_links_repo: tuple[Path, list[str]]
    ) -> None:
        """--all calls LLM for each drifted link and prints all results."""
        repo, link_ids = two_drifted_links_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", "--all"], repo=repo)

        assert result.exit_code == 0, result.output
        # Both link IDs should appear in the output
        for lid in link_ids:
            assert lid in result.output

    def test_all_json_is_a_list(
        self, runner: CliRunner, two_drifted_links_repo: tuple[Path, list[str]]
    ) -> None:
        """--all --json outputs a JSON array with one entry per link."""
        repo, _ = two_drifted_links_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", "--all", "--json"], repo=repo)

        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_all_no_drifted_links_prints_message(
        self, runner: CliRunner, fresh_link_repo: tuple[Path, str]
    ) -> None:
        """--all with all links fresh prints 'No drifted links' and exits 0."""
        repo, _ = fresh_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", "--all"], repo=repo)

        assert result.exit_code == 0, result.output
        assert "No drifted links" in result.output


class TestReconcileCmdErrors:
    """Error handling for scry reconcile."""

    def test_no_link_id_no_all_exits_2(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """Missing LINK_ID without --all exits 2 with guidance."""
        repo, _ = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile"], repo=repo)

        assert result.exit_code == 2

    def test_unknown_link_id_exits_2(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """An unknown link_id exits 2 with 'not found' message."""
        repo, _ = drifted_link_repo
        fake_id = new_link_id()

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", fake_id], repo=repo)

        assert result.exit_code == 2
        assert "not found" in result.output.lower()

    def test_llm_error_exits_2(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """LLM network error exits 2 with a descriptive message."""
        repo, link_id = drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_ErrorProvider()):
            result = _run(runner, ["reconcile", link_id], repo=repo)

        assert result.exit_code == 2
        assert "error" in result.output.lower()

    def test_missing_db_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        """Missing vectors.db exits 2 with helpful message."""
        scry_dir = tmp_path / ".scry"
        scry_dir.mkdir()

        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "ci@test.local")
        _git(tmp_path, "config", "user.name", "CI Test")

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", new_link_id()], repo=tmp_path)

        assert result.exit_code == 2
        assert "vectors.db" in result.output.lower()

    def test_non_actionable_drift_exits_2(self, runner: CliRunner, mini_git_repo: Path) -> None:
        """A link with broken-source drift (non-actionable) exits 2."""
        repo = mini_git_repo
        # Write only the from anchor; to_anchor is missing → broken-target
        from_id = "spec.md::orphaned-from"
        to_id = "spec.md::missing-to"

        with ScryDB(repo) as db:
            db.init_schema(embedding_dimensions=4)
            _write_anchor(db, from_id, "spec.md", _HA)
            # to_anchor deliberately NOT written → broken-target

        link_id = new_link_id()
        store = LinkStore(repo)
        store.append_baseline(
            LinkRecord.model_validate(
                {
                    "op": "upsert",
                    "link_id": link_id,
                    "event_id": new_event_id(),
                    "from": from_id,
                    "from_type": "section",
                    "to": to_id,
                    "to_type": "section",
                    "type": "implements",
                    "from_content_hash": _HA,
                    "to_content_hash": _HA,
                }
            )
        )

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", link_id], repo=repo)

        # broken-target is not in _ACTIONABLE_STATUSES → exit 2
        assert result.exit_code == 2


# ─── W5c regression fixtures ─────────────────────────────────────────────────


@pytest.fixture
def overlay_drifted_link_repo(mini_git_repo: Path) -> tuple[Path, str]:
    """Repo with a drifted link stored ONLY in the branch overlay (not baseline).

    Used to verify BLOCKING #2: overlay links must be visible to --all and
    reconcileable by ID.
    """
    repo = mini_git_repo
    from_id = "spec.md::overlay-from"
    to_id = "spec.md::overlay-to"

    with ScryDB(repo) as db:
        db.init_schema(embedding_dimensions=4)
        # DB has _HB (current); link will record _HA (stale) → spec-changed drift
        _write_anchor(db, from_id, "spec.md", _HB)
        _write_anchor(db, to_id, "spec.md", _HA)

    link_id = new_link_id()
    git_ctx_prov = GitContextProvider(repo)
    overlay_mgr = OverlayManager(repo, git_context=git_ctx_prov)
    overlay_mgr.append_to_current_branch_overlay(
        LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": link_id,
                "event_id": new_event_id(),
                "from": from_id,
                "from_type": "section",
                "to": to_id,
                "to_type": "section",
                "type": "implements",
                "from_content_hash": _HA,  # stale → drift
                "to_content_hash": _HA,
            }
        )
    )
    return repo, link_id


# ─── BLOCKING #1 regression: idempotency after --apply ────────────────────────


class TestApplyIdempotency:
    """After --apply the link must be FRESH on the next reconcile invocation."""

    def test_apply_then_reconcile_reports_fresh(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """Idempotency: second `scry reconcile <id>` after --apply reports fresh."""
        repo, link_id = drifted_link_repo

        # First pass: apply the patch.
        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            r1 = _run(runner, ["reconcile", link_id, "--apply", "--yes"], repo=repo)
        assert r1.exit_code == 0, r1.output

        # Second pass (no LLM call needed — should short-circuit at FRESH check).
        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            r2 = _run(runner, ["reconcile", link_id], repo=repo)
        assert r2.exit_code == 0, r2.output
        assert "fresh" in r2.output.lower(), f"expected 'fresh' in: {r2.output!r}"


# ─── BLOCKING #2 regression: overlay links surfaced ───────────────────────────


class TestOverlayLinkSurfaced:
    """Links in the current-branch overlay must be visible to reconcile."""

    def test_overlay_link_surfaced_by_all(
        self,
        runner: CliRunner,
        overlay_drifted_link_repo: tuple[Path, str],
    ) -> None:
        """--all must surface a drifted link that lives only in the overlay."""
        repo, link_id = overlay_drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", "--all"], repo=repo)

        assert result.exit_code == 0, result.output
        assert link_id in result.output, (
            f"overlay link {link_id!r} not found in output: {result.output!r}"
        )

    def test_overlay_link_reconcileable_by_id(
        self,
        runner: CliRunner,
        overlay_drifted_link_repo: tuple[Path, str],
    ) -> None:
        """Single-link reconcile must work for a link that lives only in the overlay."""
        repo, link_id = overlay_drifted_link_repo

        with patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()):
            result = _run(runner, ["reconcile", link_id], repo=repo)

        assert result.exit_code == 0, result.output
        # Output should contain the proposed diff or explanation.
        assert "spec" in result.output.lower() or "explanation" in result.output.lower()


# ─── BLOCKING #3 regression: diff path validation ─────────────────────────────


class TestValidateDiffPaths:
    """Unit tests for _validate_diff_paths (defence-in-depth path guards)."""

    def _call(
        self,
        diff_text: str,
        repo_root: Path,
        allowed: set[str],
        link_id: str = "lnk_test",
    ) -> None:
        _validate_diff_paths(diff_text, repo_root, allowed, link_id)

    def test_valid_endpoint_path_passes(self, tmp_path: Path) -> None:
        diff = "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old\n+new\n"
        self._call(diff, tmp_path, {"spec.md"})  # no exception

    def test_absolute_unix_path_rejected(self, tmp_path: Path) -> None:
        diff = "--- a//etc/passwd\n+++ b//etc/passwd\n@@ -1 +1 @@\n-root\n+evil\n"
        with pytest.raises(ReconcileError, match="absolute path"):
            self._call(diff, tmp_path, {"/etc/passwd"})

    def test_absolute_path_without_a_prefix_rejected(self, tmp_path: Path) -> None:
        diff = "--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\n-root\n+evil\n"
        with pytest.raises(ReconcileError, match="absolute path"):
            self._call(diff, tmp_path, {"/etc/passwd"})

    def test_windows_drive_path_rejected(self, tmp_path: Path) -> None:
        diff = "--- a/C:/Windows/system32/evil.dll\n+++ b/C:/Windows/system32/evil.dll\n"
        with pytest.raises(ReconcileError, match="absolute path"):
            self._call(diff, tmp_path, {"C:/Windows/system32/evil.dll"})

    def test_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        diff = "--- a/../../../etc/shadow\n+++ b/../../../etc/shadow\n"
        with pytest.raises(ReconcileError, match="traversal"):
            self._call(diff, tmp_path, {"../../../etc/shadow"})

    def test_nested_dotdot_rejected(self, tmp_path: Path) -> None:
        diff = "--- a/subdir/../../secret.txt\n+++ b/subdir/../../secret.txt\n"
        with pytest.raises(ReconcileError, match="traversal"):
            self._call(diff, tmp_path, {"subdir/../../secret.txt"})

    def test_path_not_in_allowed_rejected(self, tmp_path: Path) -> None:
        diff = "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old\n+new\n"
        with pytest.raises(ReconcileError, match="not a link endpoint"):
            self._call(diff, tmp_path, {"other.py"})

    def test_dev_null_skipped(self, tmp_path: Path) -> None:
        """New-file diffs use /dev/null as the 'from' sentinel — must not be blocked."""
        diff = "--- /dev/null\n+++ b/spec.md\n@@ -0,0 +1 @@\n+new file\n"
        self._call(diff, tmp_path, {"spec.md"})  # no exception

    def test_both_endpoints_in_diff_both_must_be_allowed(self, tmp_path: Path) -> None:
        diff = (
            "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old\n+new\n"
            "--- a/code.py\n+++ b/code.py\n@@ -1 +1 @@\n-def f():\n+def g():\n"
        )
        # Both paths in allowed → OK
        self._call(diff, tmp_path, {"spec.md", "code.py"})

    def test_third_file_in_diff_rejected(self, tmp_path: Path) -> None:
        diff = (
            "--- a/spec.md\n+++ b/spec.md\n@@ -1 +1 @@\n-old\n+new\n"
            "--- a/extra.py\n+++ b/extra.py\n@@ -1 +1 @@\n-x\n+y\n"
        )
        with pytest.raises(ReconcileError, match="not a link endpoint"):
            self._call(diff, tmp_path, {"spec.md"})


class TestApplyPathValidationCLI:
    """CLI integration: a diff touching a bad path must be rejected, exit 1."""

    def test_diff_touching_non_endpoint_exits_1(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """An LLM diff that edits a file not in the link endpoints is rejected."""
        repo, link_id = drifted_link_repo

        class _BadPathProvider:
            name = "bad_path"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                # Produce a diff touching a third file.
                evil_diff = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-old\n+evil\n"
                return LLMResponse(
                    text=json.dumps({"target": "spec", "diff": evil_diff, "explanation": "evil"}),
                    model="bad",
                    provider="bad",
                    usage=None,
                    finish_reason="stop",
                )

        with patch("scry.cmd_reconcile.make_provider", return_value=_BadPathProvider()):
            result = _run(runner, ["reconcile", link_id, "--apply", "--yes"], repo=repo)

        assert result.exit_code == 1, result.output
        assert "rejected" in result.output.lower() or "rejected" in (result.output + "").lower()

    def test_diff_with_absolute_path_exits_1(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """A diff with an absolute path is rejected even if --yes is passed."""
        repo, link_id = drifted_link_repo

        class _AbsPathProvider:
            name = "abs_path"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                abs_diff = (
                    "--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\n-root:x:0:0\n+evil:x:0:0\n"
                )
                return LLMResponse(
                    text=json.dumps({"target": "spec", "diff": abs_diff, "explanation": "evil"}),
                    model="bad",
                    provider="bad",
                    usage=None,
                    finish_reason="stop",
                )

        with patch("scry.cmd_reconcile.make_provider", return_value=_AbsPathProvider()):
            result = _run(runner, ["reconcile", link_id, "--apply", "--yes"], repo=repo)

        assert result.exit_code == 1, result.output


# ─── HIGH #1 regression: apply failures exit 1 ────────────────────────────────


class TestApplyExitCodes:
    """--apply failure semantics: single-link → exit 1; --all → exit 1 with summary."""

    def test_single_link_apply_failure_exits_1(
        self, runner: CliRunner, drifted_link_repo: tuple[Path, str]
    ) -> None:
        """A patch that git apply rejects causes exit 1 (not 0)."""
        repo, link_id = drifted_link_repo

        class _BadDiffProvider:
            name = "bad_diff"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                bad = (
                    "--- a/spec.md\n+++ b/spec.md\n"
                    "@@ -1 +1 @@\n-this line does not exist\n+replacement\n"
                )
                return LLMResponse(
                    text=json.dumps({"target": "spec", "diff": bad, "explanation": "x"}),
                    model="bad",
                    provider="bad",
                    usage=None,
                    finish_reason="stop",
                )

        with patch("scry.cmd_reconcile.make_provider", return_value=_BadDiffProvider()):
            result = _run(runner, ["reconcile", link_id, "--apply", "--yes"], repo=repo)

        assert result.exit_code == 1, result.output

    def test_all_apply_mixed_failure_exits_1_with_summary(
        self, runner: CliRunner, two_drifted_links_repo: tuple[Path, list[str]]
    ) -> None:
        """--all --apply: one patch fails, one succeeds → exit 1 + summary printed."""
        repo, _link_ids = two_drifted_links_repo
        call_count = 0

        class _MixedProvider:
            name = "mixed"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First link: valid patch.
                    return LLMResponse(
                        text=_CANNED_JSON,
                        model="m",
                        provider="m",
                        usage=None,
                        finish_reason="stop",
                    )
                # Second link: bad patch (will fail git apply).
                bad = (
                    "--- a/spec.md\n+++ b/spec.md\n"
                    "@@ -1 +1 @@\n-line that does not exist anywhere\n+replacement\n"
                )
                return LLMResponse(
                    text=json.dumps({"target": "spec", "diff": bad, "explanation": "bad"}),
                    model="m",
                    provider="m",
                    usage=None,
                    finish_reason="stop",
                )

        with patch("scry.cmd_reconcile.make_provider", return_value=_MixedProvider()):
            result = _run(runner, ["reconcile", "--all", "--apply", "--yes"], repo=repo)

        assert result.exit_code == 1, result.output
        # Summary line should be printed when multiple links are processed.
        assert "summary" in result.output.lower() or "failed" in result.output.lower()

    def test_all_apply_all_succeed_exits_0(
        self, runner: CliRunner, two_drifted_links_repo: tuple[Path, list[str]]
    ) -> None:
        """--all --apply where all patches apply cleanly exits 0.

        We mock _apply_git_patch and _write_post_apply_overlay so that identical
        diffs for two links don't conflict on the same file — this test is purely
        about the exit-code logic, not actual git apply behaviour.
        """
        repo, _ = two_drifted_links_repo

        with (
            patch("scry.cmd_reconcile.make_provider", return_value=_MockProvider()),
            patch("scry.cmd_reconcile._apply_git_patch"),  # no-op
            patch("scry.cmd_reconcile._write_post_apply_overlay"),  # no-op
        ):
            result = _run(runner, ["reconcile", "--all", "--apply", "--yes"], repo=repo)

        assert result.exit_code == 0, result.output


# ─── MEDIUM #1 regression: LLM context truncation ────────────────────────────


class TestLLMContextTruncation:
    """Anchor text and git diffs are capped before reaching the LLM."""

    def test_long_anchor_text_truncated(self, tmp_path: Path) -> None:
        """A content_text longer than _MAX_ANCHOR_TEXT_CHARS is cut + warning logged."""
        from scry.cmd_reconcile import _MAX_ANCHOR_TEXT_CHARS, _build_user_message
        from scry.models import AnchorType, DriftStatus

        long_text = "x" * (_MAX_ANCHOR_TEXT_CHARS + 500)
        from_anchor = Anchor(
            id="spec.md::from",
            type=AnchorType.SECTION,
            path="spec.md",
            content_text=long_text,
            content_hash="sha256:" + "a" * 64,
            fingerprint_simhash=0,
        )
        to_anchor = Anchor(
            id="spec.md::to",
            type=AnchorType.SECTION,
            path="spec.md",
            content_text="short",
            content_hash="sha256:" + "b" * 64,
            fingerprint_simhash=0,
        )

        msg = _build_user_message(
            link_id="lnk_test",
            link_type="implements",
            drift_status=DriftStatus.SPEC_CHANGED,
            from_anchor=from_anchor,
            to_anchor=to_anchor,
            commit_sha=None,
            worktree_dirty=False,
            evidence=None,
            spec_diff=None,
            code_diff=None,
            unreachable_note=None,
        )
        # The truncation marker must appear in the message.
        assert "truncated" in msg
        # The full long text must NOT appear.
        assert long_text not in msg
        # The truncated portion fits.
        assert long_text[:_MAX_ANCHOR_TEXT_CHARS] in msg

    def test_long_anchor_text_warning_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Truncation emits a WARNING log entry."""
        from scry.cmd_reconcile import _MAX_ANCHOR_TEXT_CHARS, _build_user_message
        from scry.models import AnchorType, DriftStatus

        long_text = "y" * (_MAX_ANCHOR_TEXT_CHARS + 1)
        anchor = Anchor(
            id="a::b",
            type=AnchorType.SECTION,
            path="a.md",
            content_text=long_text,
            content_hash="sha256:" + "c" * 64,
            fingerprint_simhash=0,
        )
        short_anchor = Anchor(
            id="a::c",
            type=AnchorType.SECTION,
            path="a.md",
            content_text="short",
            content_hash="sha256:" + "d" * 64,
            fingerprint_simhash=0,
        )

        with caplog.at_level(logging.WARNING, logger="scry.cmd_reconcile"):
            _build_user_message(
                link_id="lnk_x",
                link_type="implements",
                drift_status=DriftStatus.SPEC_CHANGED,
                from_anchor=anchor,
                to_anchor=short_anchor,
                commit_sha=None,
                worktree_dirty=False,
                evidence=None,
                spec_diff=None,
                code_diff=None,
                unreachable_note=None,
            )

        assert any("Truncating" in r.message for r in caplog.records), (
            "Expected a WARNING about truncation in caplog"
        )

    def test_long_diff_truncated(self, tmp_path: Path) -> None:
        """A git diff longer than _MAX_DIFF_CHARS is truncated in the LLM message."""
        from scry.cmd_reconcile import _MAX_DIFF_CHARS, _build_user_message
        from scry.models import AnchorType, DriftStatus

        long_diff = "+" + "z" * (_MAX_DIFF_CHARS + 100) + "\n"
        short_anchor = Anchor(
            id="x::y",
            type=AnchorType.SECTION,
            path="x.md",
            content_text="short",
            content_hash="sha256:" + "e" * 64,
            fingerprint_simhash=0,
        )

        msg = _build_user_message(
            link_id="lnk_y",
            link_type="implements",
            drift_status=DriftStatus.SPEC_CHANGED,
            from_anchor=short_anchor,
            to_anchor=short_anchor,
            commit_sha=None,
            worktree_dirty=False,
            evidence=None,
            spec_diff=long_diff,
            code_diff=None,
            unreachable_note=None,
        )
        assert "truncated" in msg
        assert long_diff not in msg


# uat-r5-5 pr-d noise
