"""Tests for ``scry.dashboard`` — data gathering + HTTP server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scry.dashboard import (
    DASHBOARD_HTML,
    _DashboardHandler,
    gather_dashboard_data,
    make_handler,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture()
def indexed_repo(tmp_path: Path) -> Path:
    """Create a minimal repo with .scry/config.yaml, git init, and a fresh index."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    scry_dir = repo / ".scry"
    scry_dir.mkdir()
    (scry_dir / "config.yaml").write_text(
        "include:\n  - '**/*.md'\n  - '**/*.py'\n"
        "exclude:\n  - .scry/**\n"
        "classify:\n  - { glob: '**/*.md', type: doc }\n"
        "embeddings:\n  provider: local\n  model: BAAI/bge-small-en-v1.5\n  dimensions: 384\n",
        encoding="utf-8",
    )
    (scry_dir / ".gitignore").write_text("vectors.db\nvectors.db-*\nleader.lock\n")

    # Create some source files.
    (repo / "README.md").write_text("# My Project\n\nSome overview.\n\n## Features\n\nCool stuff.\n")
    src = repo / "src"
    src.mkdir()
    (src / "main.py").write_text("def hello():\n    return 'world'\n\nclass Greeter:\n    pass\n")

    # Git init so git_context works.
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, check=True, capture_output=True,
    )

    # Build the index (with stub embedder to avoid fastembed load time).
    from scry.config import load_config
    from scry.embed import StubEmbedder
    from scry.index import Indexer

    config = load_config(repo)
    embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
    indexer = Indexer(repo_root=repo, config=config, embedder=embedder)
    indexer.index(force=True)

    return repo


# ──────────────────────────────────────────────────────────────────────
# Data gathering tests
# ──────────────────────────────────────────────────────────────────────


class TestGatherDashboardData:
    """Tests for ``gather_dashboard_data``."""

    def test_returns_expected_keys(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        assert "anchors" in data
        assert "evaluations" in data
        assert "summary" in data
        assert "anchor_type_counts" in data
        assert "repo_root" in data
        assert "branch" in data

    def test_anchors_are_populated(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        assert len(data["anchors"]) > 0
        # Each anchor has the expected fields.
        anchor = data["anchors"][0]
        assert "id" in anchor
        assert "type" in anchor
        assert "path" in anchor
        assert "content_preview" in anchor

    def test_anchor_types_counted(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        counts = data["anchor_type_counts"]
        assert counts["section"] >= 1  # README.md headings
        assert counts["code"] >= 1  # main.py symbols

    def test_summary_structure(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        summary = data["summary"]
        assert "drift_score" in summary
        assert "coverage_score" in summary
        assert "counts" in summary
        assert "drift_coverage" in summary

    def test_counts_use_hyphenated_keys(self, indexed_repo: Path) -> None:
        """Drift counts keys should use hyphens, not underscores."""
        data = gather_dashboard_data(indexed_repo)
        counts = data["summary"]["counts"]
        for key in counts:
            assert "_" not in key or key == "semantic-drift-flagged", (
                f"Count key {key!r} uses underscores — should use hyphens"
            )

    def test_evaluations_empty_when_no_links(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        # No links created → no evaluations.
        assert data["evaluations"] == []

    def test_evaluations_populated_with_links(self, indexed_repo: Path) -> None:
        """Create a link and verify it appears in evaluations."""
        from scry.git_context import GitContextProvider
        from scry.models import LinkOp, LinkType
        from scry.store.links import LinkRecord
        from scry.store.overlay import OverlayManager

        # Find two anchors to link.
        data = gather_dashboard_data(indexed_repo)
        sections = [a for a in data["anchors"] if a["type"] == "section"]
        codes = [a for a in data["anchors"] if a["type"] == "code"]
        if not sections or not codes:
            pytest.skip("Need both section and code anchors")

        git_ctx = GitContextProvider(indexed_repo)
        overlay_mgr = OverlayManager(indexed_repo, git_context=git_ctx)

        from scry.store.db import ScryDB

        db = ScryDB(indexed_repo, read_only=True)
        try:
            from_anchor = db.get_anchor(sections[0]["id"])
            to_anchor = db.get_anchor(codes[0]["id"])
        finally:
            db.close()

        assert from_anchor is not None
        assert to_anchor is not None

        record = LinkRecord.model_validate({
            "op": LinkOp.UPSERT,
            "link_id": "lnk_test_dashboard_001",
            "from": from_anchor.id,
            "from_type": from_anchor.type,
            "to": to_anchor.id,
            "to_type": to_anchor.type,
            "type": LinkType.IMPLEMENTS,
            "from_content_hash": from_anchor.content_hash,
            "to_content_hash": to_anchor.content_hash,
            "commit_sha": "abc123",
        })
        overlay_mgr.append_to_current_branch_overlay(record)

        data2 = gather_dashboard_data(indexed_repo)
        assert len(data2["evaluations"]) >= 1
        ev = data2["evaluations"][0]
        assert "link_id" in ev
        assert "drift_status" in ev
        assert "from_id" in ev
        assert "to_id" in ev

    def test_content_preview_truncated(self, indexed_repo: Path) -> None:
        """Long content_text is truncated to 300 chars + ellipsis."""
        data = gather_dashboard_data(indexed_repo)
        for anchor in data["anchors"]:
            assert len(anchor["content_preview"]) <= 302  # 300 + "…" (2 bytes utf-8 len can be 1)

    def test_data_is_json_serializable(self, indexed_repo: Path) -> None:
        data = gather_dashboard_data(indexed_repo)
        # Should not raise.
        serialized = json.dumps(data)
        assert len(serialized) > 0


# ──────────────────────────────────────────────────────────────────────
# HTML template tests
# ──────────────────────────────────────────────────────────────────────


class TestDashboardHTML:
    """Sanity checks on the embedded HTML template."""

    def test_html_is_valid_string(self) -> None:
        assert isinstance(DASHBOARD_HTML, str)
        assert len(DASHBOARD_HTML) > 1000

    def test_html_contains_essential_elements(self) -> None:
        assert "<!DOCTYPE html>" in DASHBOARD_HTML
        assert "scry" in DASHBOARD_HTML.lower()
        assert "/api/data" in DASHBOARD_HTML
        assert "d3.v7" in DASHBOARD_HTML

    def test_html_has_both_panels(self) -> None:
        assert 'id="overview"' in DASHBOARD_HTML
        assert 'id="explorer"' in DASHBOARD_HTML

    def test_html_has_drift_colors(self) -> None:
        assert "fresh" in DASHBOARD_HTML
        assert "code-changed" in DASHBOARD_HTML
        assert "broken-source" in DASHBOARD_HTML

    def test_html_has_xss_escape_utility(self) -> None:
        """Frontend must define an ``esc()`` function for HTML-escaping."""
        assert "function esc(" in DASHBOARD_HTML

    def test_html_uses_esc_for_user_content(self) -> None:
        """User-controlled strings in innerHTML must go through esc()."""
        assert "esc(d.label)" in DASHBOARD_HTML
        assert "esc(a.path)" in DASHBOARD_HTML
        assert "esc(other)" in DASHBOARD_HTML

    def test_html_has_graph_size_guard(self) -> None:
        """Large graphs should be guarded to prevent browser freeze."""
        assert "MAX_GRAPH_NODES" in DASHBOARD_HTML


# ──────────────────────────────────────────────────────────────────────
# HTTP server tests
# ──────────────────────────────────────────────────────────────────────


class TestDashboardServer:
    """Test the HTTP server endpoints."""

    def test_make_handler_binds_repo_root(self, tmp_path: Path) -> None:
        handler_cls = make_handler(tmp_path)
        assert handler_cls.repo_root == tmp_path  # type: ignore[attr-defined]

    def test_serve_html_and_api(self, indexed_repo: Path) -> None:
        """Start the server in a thread, hit both endpoints, then shut down."""
        from http.server import HTTPServer

        handler_cls = make_handler(indexed_repo)
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            # Test HTML endpoint.
            url = f"http://127.0.0.1:{port}/"
            resp = urllib.request.urlopen(url, timeout=10)
            assert resp.status == 200
            html = resp.read()
            assert b"scry" in html
            assert b"d3.v7" in html

            # Test API endpoint.
            api_url = f"http://127.0.0.1:{port}/api/data"
            resp2 = urllib.request.urlopen(api_url, timeout=10)
            assert resp2.status == 200
            data = json.loads(resp2.read())
            assert "anchors" in data
            assert "evaluations" in data
            assert "summary" in data

            # Test 404.
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nonexistent", timeout=5)
                raise AssertionError("Expected 404")  # noqa: TRY301
            except urllib.error.HTTPError as e:
                assert e.code == 404
        finally:
            server.shutdown()
            thread.join(timeout=5)
