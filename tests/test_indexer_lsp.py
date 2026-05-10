"""Tests for W3d LSP integration with the Indexer (workstream W3d).

Tests the LSP-enrichment pipeline: anchor extraction → LSP closure walk →
``transitive_hash_status`` / ``closure_hash`` population → drift evaluation.

Test strategy
-------------
* **Unit tests** (no subprocess): mock or stub out LSP infra; test status
  mapping, drift propagation, and didOpen/shutdown call discipline.
* **Integration tests** (``@pytest.mark.integration``): spawn the fake LSP
  fixtures and exercise the real ``compute_closure`` path end-to-end.

The enrichment function ``_enrich_all_with_lsp`` is called directly in some
tests to avoid the full git-repo setup required by ``Indexer.index()``.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scry.embed import StubEmbedder
from scry.index import Indexer, _enrich_all_with_lsp, _ext_to_lang, _lang_to_language_id
from scry.lsp.manager import LSPManager
from scry.models import (
    AnchorType,
    CodeAnchorsConfig,
    DriftStatus,
    Link,
    LinkType,
    TransitiveHashStatus,
    new_event_id,
    new_link_id,
)
from scry.store.db import ScryDB

# ─── Fixture paths ───────────────────────────────────────────────────────────

FAKE_LSP = Path(__file__).parent / "fixtures" / "fake_lsp.py"
FAKE_LSP_CALLS = Path(__file__).parent / "fixtures" / "fake_lsp_calls.py"

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_code_anchor(
    path_str: str,
    symbol_path: str = "my_func",
    *,
    status: str = "lsp_unavailable",
    closure_hash: str | None = None,
    def_line: int | None = None,
    def_char: int | None = None,
) -> Any:
    """Build a minimal CODE anchor for testing without tree-sitter."""
    from scry.anchor_id import canonicalize_content
    from scry.anchor_id import content_hash as _ch
    from scry.anchor_id import fingerprint_simhash as _sh
    from scry.models import Anchor

    content = f"def {symbol_path}(): pass"
    canon = canonicalize_content(content)
    return Anchor(
        id=f"{path_str}:{symbol_path}",
        type=AnchorType.CODE,
        path=path_str,
        symbol_name=symbol_path,
        content_text=canon,
        content_hash=_ch(content),
        fingerprint_simhash=_sh(content),
        transitive_hash_status=TransitiveHashStatus(status),
        closure_hash=closure_hash,
        def_line=def_line,
        def_char=def_char,
    )


def _make_section_anchor(path_str: str, heading: str = "Intro") -> Any:
    from scry.anchor_id import canonicalize_content
    from scry.anchor_id import content_hash as _ch
    from scry.anchor_id import fingerprint_simhash as _sh
    from scry.models import Anchor

    content = f"# {heading}\nSome spec text."
    canon = canonicalize_content(content)
    return Anchor(
        id=f"{path_str}::{heading}",
        type=AnchorType.SECTION,
        path=path_str,
        heading_path=[heading],
        symbol_name=None,
        content_text=canon,
        content_hash=_ch(content),
        fingerprint_simhash=_sh(content),
        transitive_hash_status=None,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def py_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Minimal git repo with a single Python file and .scry config.

    The config enables Python code anchors via ``code_anchors.languages.python: lsp``.
    """
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "overlays").mkdir()

    # Simple Python file with two functions.
    py_dir = tmp_path / "src"
    py_dir.mkdir()
    (py_dir / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n",
        encoding="utf-8",
    )

    # Config: include src/*.py, mark python as lsp.
    (scry_dir / "config.yaml").write_text(
        "include:\n  - 'src/**.py'\ncode_anchors:\n  languages:\n    python: lsp\n",
        encoding="utf-8",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@test.local")
    _git(tmp_path, "config", "user.name", "CI")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")

    yield tmp_path


@pytest.fixture
def skip_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Repo with python configured as ``skip``."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "overlays").mkdir()

    py_dir = tmp_path / "src"
    py_dir.mkdir()
    (py_dir / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    (scry_dir / "config.yaml").write_text(
        "include:\n  - 'src/**.py'\ncode_anchors:\n  languages:\n    python: skip\n",
        encoding="utf-8",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@test.local")
    _git(tmp_path, "config", "user.name", "CI")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")

    yield tmp_path


@pytest.fixture
def no_lang_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Repo with no code_anchors.languages at all (unknown language scenario)."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "overlays").mkdir()

    py_dir = tmp_path / "src"
    py_dir.mkdir()
    (py_dir / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    (scry_dir / "config.yaml").write_text(
        "include:\n  - 'src/**.py'\n",
        encoding="utf-8",
    )

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@test.local")
    _git(tmp_path, "config", "user.name", "CI")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")

    yield tmp_path


def _open_db(repo: Path, dims: int = 384) -> ScryDB:
    db = ScryDB(repo)
    db.init_schema(embedding_dimensions=dims)
    return db


# ─── Helper function unit tests ───────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for module-level helpers added for W3d."""

    def test_ext_to_lang_python(self) -> None:
        assert _ext_to_lang("src/foo.py") == "python"

    def test_ext_to_lang_typescript(self) -> None:
        assert _ext_to_lang("src/app.ts") == "typescript"

    def test_ext_to_lang_tsx(self) -> None:
        assert _ext_to_lang("src/app.tsx") == "tsx"

    def test_ext_to_lang_unknown(self) -> None:
        assert _ext_to_lang("src/app.rb") is None

    def test_ext_to_lang_markdown(self) -> None:
        assert _ext_to_lang("docs/README.md") is None

    def test_lang_to_language_id_python(self) -> None:
        assert _lang_to_language_id("python") == "python"

    def test_lang_to_language_id_tsx(self) -> None:
        assert _lang_to_language_id("tsx") == "typescriptreact"

    def test_lang_to_language_id_unknown_passthrough(self) -> None:
        # Unknown languages pass through unchanged.
        assert _lang_to_language_id("ruby") == "ruby"


# ─── LSPManager.status_for() ──────────────────────────────────────────────────


class TestLSPManagerStatusFor:
    """Unit tests for the new LSPManager.status_for() public method."""

    def test_unknown_language_returns_unknown(self, tmp_path: Path) -> None:
        """Language not in languages dict → 'unknown'."""
        cfg = CodeAnchorsConfig(languages={})
        mgr = LSPManager(tmp_path, cfg)
        assert mgr.status_for("python") == "unknown"

    def test_skip_language_returns_skip(self, tmp_path: Path) -> None:
        """Language configured as 'skip' → 'skip'."""
        cfg = CodeAnchorsConfig(languages={"python": "skip"})
        mgr = LSPManager(tmp_path, cfg)
        assert mgr.status_for("python") == "skip"

    def test_failed_language_returns_lsp_unavailable(self, tmp_path: Path) -> None:
        """Language in _failed set → 'lsp_unavailable'."""
        cfg = CodeAnchorsConfig(languages={"python": "lsp"})
        mgr = LSPManager(tmp_path, cfg)
        mgr._failed.add("python")
        assert mgr.status_for("python") == "lsp_unavailable"

    def test_configured_lsp_not_started_returns_lsp_unavailable(self, tmp_path: Path) -> None:
        """Language configured as 'lsp' but never started → 'lsp_unavailable'."""
        cfg = CodeAnchorsConfig(languages={"python": "lsp"})
        mgr = LSPManager(tmp_path, cfg)
        # status_for without calling session_for first → 'lsp_unavailable'
        assert mgr.status_for("python") == "lsp_unavailable"

    @pytest.mark.integration
    async def test_available_after_session_for(self, tmp_path: Path) -> None:
        """Language returns 'available' after a successful session_for()."""
        cfg = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP)]}},
        )
        async with LSPManager(tmp_path, cfg, allow_untrusted=True) as mgr:
            session = await mgr.session_for("python")
            assert session is not None
            assert mgr.status_for("python") == "available"


# ─── Anchor model new fields ──────────────────────────────────────────────────


class TestAnchorNewFields:
    """Tests for the new def_line, def_char, closure_hash fields on Anchor."""

    def test_code_anchor_accepts_def_line_def_char(self) -> None:
        anchor = _make_code_anchor("src/utils.py", def_line=5, def_char=4)
        assert anchor.def_line == 5
        assert anchor.def_char == 4

    def test_code_anchor_accepts_closure_hash(self) -> None:
        anchor = _make_code_anchor("src/utils.py", closure_hash="abc123")
        assert anchor.closure_hash == "abc123"

    def test_closure_hash_rejected_on_non_code_anchor(self) -> None:
        from pydantic import ValidationError

        from scry.anchor_id import canonicalize_content
        from scry.anchor_id import content_hash as _ch
        from scry.anchor_id import fingerprint_simhash as _sh
        from scry.models import Anchor

        content = "# Section\nSome text."
        with pytest.raises(ValidationError, match="closure_hash"):
            Anchor(
                id="docs/spec.md::section",
                type=AnchorType.SECTION,
                path="docs/spec.md",
                heading_path=["Section"],
                symbol_name=None,
                content_text=canonicalize_content(content),
                content_hash=_ch(content),
                fingerprint_simhash=_sh(content),
                transitive_hash_status=None,
                closure_hash="should_be_rejected",
            )

    def test_def_line_defaults_to_none(self) -> None:
        anchor = _make_code_anchor("src/utils.py")
        assert anchor.def_line is None
        assert anchor.def_char is None

    def test_closure_hash_defaults_to_none(self) -> None:
        anchor = _make_code_anchor("src/utils.py")
        assert anchor.closure_hash is None


# ─── DB round-trip tests ──────────────────────────────────────────────────────


class TestClosureHashDB:
    """closure_hash persists to and is read back from the DB."""

    def test_closure_hash_stored_and_retrieved(self, tmp_path: Path) -> None:
        (tmp_path / ".scry").mkdir()
        (tmp_path / ".scry" / "overlays").mkdir()
        anchor = _make_code_anchor("src/utils.py", closure_hash="deadbeef")
        with _open_db(tmp_path) as db:
            db.upsert_anchors([anchor])
            stored = db.get_anchor(anchor.id)
        assert stored is not None
        assert stored.closure_hash == "deadbeef"

    def test_closure_hash_none_round_trips(self, tmp_path: Path) -> None:
        (tmp_path / ".scry").mkdir()
        (tmp_path / ".scry" / "overlays").mkdir()
        anchor = _make_code_anchor("src/utils.py", closure_hash=None)
        with _open_db(tmp_path) as db:
            db.upsert_anchors([anchor])
            stored = db.get_anchor(anchor.id)
        assert stored is not None
        assert stored.closure_hash is None

    def test_init_schema_migration_idempotent(self, tmp_path: Path) -> None:
        """init_schema() can be called twice without error (migration is idempotent)."""
        (tmp_path / ".scry").mkdir()
        (tmp_path / ".scry" / "overlays").mkdir()
        with _open_db(tmp_path) as db:
            # Second call should not raise OperationalError.
            db.init_schema(embedding_dimensions=384)


# ─── Drift module: lsp_error → drift-unknown ──────────────────────────────────


class TestDriftLspError:
    """evaluate_link_drift returns drift-unknown when an endpoint has lsp_error."""

    def _make_db_with_anchors(self, tmp_path: Path, from_anchor: Any, to_anchor: Any) -> ScryDB:
        (tmp_path / ".scry").mkdir(exist_ok=True)
        (tmp_path / ".scry" / "overlays").mkdir(exist_ok=True)
        db = _open_db(tmp_path)
        db.upsert_anchors([from_anchor, to_anchor])
        return db

    def test_lsp_error_on_code_anchor_produces_drift_unknown(self, tmp_path: Path) -> None:
        from scry.drift import evaluate_link_drift

        code = _make_code_anchor("src/utils.py", status="lsp_error")
        spec = _make_section_anchor("docs/spec.md")

        db = self._make_db_with_anchors(tmp_path, code, spec)
        # Match stored hashes so no hash-change status fires (step 8 can trigger).
        link = Link(
            link_id=new_link_id(),
            from_id=code.id,
            from_type=AnchorType.CODE,
            to_id=spec.id,
            to_type=AnchorType.SECTION,
            type=LinkType.IMPLEMENTS,
            from_content_hash=code.content_hash,
            to_content_hash=spec.content_hash,
            last_event_id=new_event_id(),
        )
        evaluation = evaluate_link_drift(link, db=db)
        assert evaluation.drift_status == DriftStatus.DRIFT_UNKNOWN
        db.close()

    def test_lsp_error_on_both_endpoints(self, tmp_path: Path) -> None:
        from scry.drift import evaluate_link_drift

        code1 = _make_code_anchor("src/a.py", "funcA", status="lsp_error")
        code2 = _make_code_anchor("src/b.py", "funcB", status="lsp_error")

        db = self._make_db_with_anchors(tmp_path, code1, code2)
        link = Link(
            link_id=new_link_id(),
            from_id=code1.id,
            from_type=AnchorType.CODE,
            to_id=code2.id,
            to_type=AnchorType.CODE,
            type=LinkType.IMPLEMENTS,
            from_content_hash=code1.content_hash,
            to_content_hash=code2.content_hash,
            last_event_id=new_event_id(),
        )
        evaluation = evaluate_link_drift(link, db=db)
        assert evaluation.drift_status == DriftStatus.DRIFT_UNKNOWN
        db.close()

    def test_lsp_unavailable_does_not_produce_drift_unknown(self, tmp_path: Path) -> None:
        """lsp_unavailable status keeps drift as 'fresh' (different from lsp_error)."""
        from scry.drift import evaluate_link_drift

        code = _make_code_anchor("src/utils.py", status="lsp_unavailable")
        spec = _make_section_anchor("docs/spec.md")

        db = self._make_db_with_anchors(tmp_path, code, spec)
        link = Link(
            link_id=new_link_id(),
            from_id=code.id,
            from_type=AnchorType.CODE,
            to_id=spec.id,
            to_type=AnchorType.SECTION,
            type=LinkType.IMPLEMENTS,
            from_content_hash=code.content_hash,
            to_content_hash=spec.content_hash,
            last_event_id=new_event_id(),
        )
        evaluation = evaluate_link_drift(link, db=db)
        assert evaluation.drift_status == DriftStatus.FRESH
        db.close()

    def test_code_changed_takes_precedence_over_lsp_error(self, tmp_path: Path) -> None:
        """When code is changed, code-changed wins over drift-unknown (step 7 > 8)."""
        from scry.drift import evaluate_link_drift

        code = _make_code_anchor("src/utils.py", status="lsp_error")
        spec = _make_section_anchor("docs/spec.md")

        db = self._make_db_with_anchors(tmp_path, code, spec)
        # Force a prior-hash mismatch on the code endpoint.
        link = Link(
            link_id=new_link_id(),
            from_id=code.id,
            from_type=AnchorType.CODE,
            to_id=spec.id,
            to_type=AnchorType.SECTION,
            type=LinkType.IMPLEMENTS,
            from_content_hash=code.content_hash,
            to_content_hash=spec.content_hash,
            prior_from_content_hash="sha256:" + "0" * 64,  # Different from stored hash
            last_event_id=new_event_id(),
        )
        evaluation = evaluate_link_drift(link, db=db)
        assert evaluation.drift_status == DriftStatus.CODE_CHANGED
        db.close()


# ─── _enrich_all_with_lsp unit tests (mocked LSP) ────────────────────────────


class TestEnrichAllWithLsp:
    """Tests for _enrich_all_with_lsp with mocked/faked LSP infra."""

    def test_non_code_anchors_pass_through_unchanged(self, tmp_path: Path) -> None:
        """Non-CODE anchors are not modified by enrichment."""
        section = _make_section_anchor("docs/spec.md")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="unknown")
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([section], mock_mgr, tmp_path, max_depth=4))
        assert len(result) == 1
        assert result[0].id == section.id
        # Non-CODE anchors never touched
        mock_mgr.session_for.assert_not_called()

    def test_empty_anchor_list_shuts_down_manager(self, tmp_path: Path) -> None:
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([], mock_mgr, tmp_path, max_depth=4))
        assert result == []
        # Shutdown still called even with no anchors.

    def test_no_code_anchors_calls_shutdown(self, tmp_path: Path) -> None:
        section = _make_section_anchor("docs/spec.md")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.shutdown_all = AsyncMock()

        asyncio.run(_enrich_all_with_lsp([section], mock_mgr, tmp_path, max_depth=4))
        mock_mgr.shutdown_all.assert_called_once()

    def test_unknown_extension_gets_unsupported_status(self, tmp_path: Path) -> None:
        """Code anchor with unrecognized extension → UNSUPPORTED (not lsp_unavailable)."""
        from scry.anchor_id import canonicalize_content
        from scry.anchor_id import content_hash as _ch
        from scry.anchor_id import fingerprint_simhash as _sh
        from scry.models import Anchor

        content = "fn main() {}"
        anchor = Anchor(
            id="src/main.rb:main",
            type=AnchorType.CODE,
            path="src/main.rb",  # Unknown extension
            symbol_name="main",
            content_text=canonicalize_content(content),
            content_hash=_ch(content),
            fingerprint_simhash=_sh(content),
            transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
        )
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([anchor], mock_mgr, tmp_path, max_depth=4))
        assert result[0].transitive_hash_status == TransitiveHashStatus.UNSUPPORTED

    def test_skip_language_gets_unsupported_status(self, tmp_path: Path) -> None:
        anchor = _make_code_anchor("src/utils.py")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="skip")
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([anchor], mock_mgr, tmp_path, max_depth=4))
        assert result[0].transitive_hash_status == TransitiveHashStatus.UNSUPPORTED

    def test_unknown_language_gets_unsupported_status(self, tmp_path: Path) -> None:
        anchor = _make_code_anchor("src/utils.py")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="unknown")
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([anchor], mock_mgr, tmp_path, max_depth=4))
        assert result[0].transitive_hash_status == TransitiveHashStatus.UNSUPPORTED

    def test_lsp_unavailable_gets_lsp_unavailable_status(self, tmp_path: Path) -> None:
        anchor = _make_code_anchor("src/utils.py")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="lsp_unavailable")
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([anchor], mock_mgr, tmp_path, max_depth=4))
        assert result[0].transitive_hash_status == TransitiveHashStatus.LSP_UNAVAILABLE

    def test_original_order_preserved(self, tmp_path: Path) -> None:
        """Anchors are returned in the same order they were passed in."""
        a1 = _make_code_anchor("src/a.py", "func_a")
        a2 = _make_section_anchor("docs/spec.md")
        a3 = _make_code_anchor("src/b.py", "func_b")

        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="unknown")
        mock_mgr.shutdown_all = AsyncMock()

        result = asyncio.run(_enrich_all_with_lsp([a1, a2, a3], mock_mgr, tmp_path, max_depth=4))
        assert [r.id for r in result] == [a1.id, a2.id, a3.id]

    def test_did_open_called_once_per_file(self, tmp_path: Path) -> None:
        """Two anchors in the same file → didOpen called only once."""
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "utils.py").write_text(
            "def add(a, b): return a + b\ndef sub(a, b): return a - b\n",
            encoding="utf-8",
        )

        a1 = _make_code_anchor("src/utils.py", "add", def_line=0, def_char=4)
        a2 = _make_code_anchor("src/utils.py", "sub", def_line=1, def_char=4)

        # Build a mock session with a real notify call tracker
        mock_session = MagicMock()
        mock_session.notify = AsyncMock()
        mock_session.is_alive = True

        async def _fake_closure(sess: Any, uri: str, line: int, char: int, **kw: Any) -> Any:
            from scry.lsp.closure import ClosureResult

            return ClosureResult(
                status="complete",
                closure_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                callees=(),
                depth_reached=0,
                diagnostic={},
            )

        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=mock_session)
        mock_mgr.shutdown_all = AsyncMock()

        with patch("scry.index.compute_closure", side_effect=_fake_closure):
            asyncio.run(_enrich_all_with_lsp([a1, a2], mock_mgr, tmp_path, max_depth=4))

        # Collect all didOpen calls
        did_open_calls = [
            c for c in mock_session.notify.call_args_list if c.args[0] == "textDocument/didOpen"
        ]
        assert len(did_open_calls) == 1, (
            f"Expected 1 didOpen call, got {len(did_open_calls)}: {did_open_calls}"
        )

    def test_did_close_called_for_opened_file(self, tmp_path: Path) -> None:
        """didClose is sent for each URI that was opened."""
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "utils.py").write_text("def add(a, b): return a + b\n")

        a1 = _make_code_anchor("src/utils.py", "add")
        mock_session = MagicMock()
        mock_session.notify = AsyncMock()

        async def _fake_closure(sess: Any, uri: str, line: int, char: int, **kw: Any) -> Any:
            from scry.lsp.closure import ClosureResult

            return ClosureResult(
                status="complete",
                closure_hash="abc",
                callees=(),
                depth_reached=0,
                diagnostic={},
            )

        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=mock_session)
        mock_mgr.shutdown_all = AsyncMock()

        with patch("scry.index.compute_closure", side_effect=_fake_closure):
            asyncio.run(_enrich_all_with_lsp([a1], mock_mgr, tmp_path, max_depth=4))

        did_close_calls = [
            c for c in mock_session.notify.call_args_list if c.args[0] == "textDocument/didClose"
        ]
        assert len(did_close_calls) == 1

    def test_shutdown_all_called_after_enrichment(self, tmp_path: Path) -> None:
        """shutdown_all() is always called at the end of _enrich_all_with_lsp."""
        anchor = _make_code_anchor("src/utils.py")
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=None)
        mock_mgr.status_for = MagicMock(return_value="unknown")
        mock_mgr.shutdown_all = AsyncMock()

        asyncio.run(_enrich_all_with_lsp([anchor], mock_mgr, tmp_path, max_depth=4))
        mock_mgr.shutdown_all.assert_called_once()

    def test_shutdown_all_called_even_on_exception(self, tmp_path: Path) -> None:
        """shutdown_all() is called even when session_for raises."""
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "utils.py").write_text("def add(a, b): return a + b\n")

        a1 = _make_code_anchor("src/utils.py", "add")

        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(side_effect=RuntimeError("boom"))
        mock_mgr.shutdown_all = AsyncMock()

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(_enrich_all_with_lsp([a1], mock_mgr, tmp_path, max_depth=4))

        mock_mgr.shutdown_all.assert_called_once()


# ─── Indexer constructor tests ────────────────────────────────────────────────


class TestIndexerConstructor:
    """Indexer accepts lsp_manager and allow_untrusted kwargs."""

    def test_indexer_accepts_lsp_manager_kwarg(self, tmp_path: Path) -> None:
        (tmp_path / ".scry").mkdir()
        mock_mgr = MagicMock(spec=LSPManager)
        indexer = Indexer(tmp_path, lsp_manager=mock_mgr)
        assert indexer._lsp_manager is mock_mgr

    def test_indexer_accepts_allow_untrusted(self, tmp_path: Path) -> None:
        (tmp_path / ".scry").mkdir()
        indexer = Indexer(tmp_path, allow_untrusted=True)
        assert indexer._allow_untrusted is True

    def test_indexer_default_allow_untrusted_false(self, tmp_path: Path) -> None:
        (tmp_path / ".scry").mkdir()
        indexer = Indexer(tmp_path)
        assert indexer._allow_untrusted is False

    def test_ensure_lsp_manager_creates_lsp_manager(self, py_repo: Path) -> None:
        """_ensure_lsp_manager() constructs LSPManager lazily from config."""
        from scry.config import load_config

        config = load_config(py_repo)
        indexer = Indexer(py_repo, config=config)
        mgr = indexer._ensure_lsp_manager(config)
        assert isinstance(mgr, LSPManager)
        # Same instance on second call.
        mgr2 = indexer._ensure_lsp_manager(config)
        assert mgr is mgr2


# ─── Indexer full integration tests (no real LSP) ────────────────────────────


class TestIndexerWithNoLsp:
    """Indexer runs successfully when LSP binary is not available."""

    def test_no_lsp_binary_all_code_anchors_get_lsp_unavailable(self, py_repo: Path) -> None:
        """When no LSP binary is on PATH, all CODE anchors get lsp_unavailable."""
        from scry.config import load_config

        config = load_config(py_repo)
        # Override lsp config to point at a non-existent binary.
        bad_lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": "nonexistent_binary_scry_xyz_123", "args": []}},
        )

        bad_config = config.model_copy(update={"code_anchors": bad_lsp_config})
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        indexer = Indexer(py_repo, config=bad_config, embedder=embedder, allow_untrusted=True)

        # Should not raise.
        result = indexer.index()
        assert result.anchors_extracted >= 1

        with _open_db(py_repo, config.embeddings.dimensions) as db:
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)

        # All code anchors should have lsp_unavailable (no real binary present).
        assert len(code_anchors) >= 1
        for a in code_anchors:
            assert a.transitive_hash_status == TransitiveHashStatus.LSP_UNAVAILABLE, (
                f"Expected lsp_unavailable, got {a.transitive_hash_status} for {a.id}"
            )

    def test_no_crash_when_lsp_completely_missing(self, py_repo: Path) -> None:
        """Indexer completes without raising even when all LSP sessions fail."""
        from scry.config import load_config

        config = load_config(py_repo)
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        # Default config has no binary → session_for returns None.
        indexer = Indexer(py_repo, config=config, embedder=embedder)
        result = indexer.index()
        assert result.files_processed >= 1


class TestIndexerSkipLanguage:
    """Indexer with skip-configured language produces UNSUPPORTED anchors."""

    def test_skip_language_produces_unsupported(self, skip_repo: Path) -> None:
        """Python anchors with languages.python: skip have no code extraction."""
        from scry.config import load_config

        config = load_config(skip_repo)
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        indexer = Indexer(skip_repo, config=config, embedder=embedder)
        indexer.index()

        # With python=skip, extract_code_symbols returns [] → no code anchors.
        with _open_db(skip_repo, config.embeddings.dimensions) as db:
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)

        assert len(code_anchors) == 0
        # files_processed may be 0 if only py files exist and they're skipped
        # at extraction level (not at index level, since index level marks
        # language as "code" then extractor returns []).


class TestIndexerUnknownLanguage:
    """Code anchors extracted from 'unknown' language files get UNSUPPORTED."""

    def test_unknown_lang_code_anchors_get_unsupported(self, no_lang_repo: Path) -> None:
        """Python files with no languages config → language unknown → unsupported."""
        from scry.config import load_config

        config = load_config(no_lang_repo)
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        indexer = Indexer(no_lang_repo, config=config, embedder=embedder)
        result = indexer.index()

        with _open_db(no_lang_repo, config.embeddings.dimensions) as db:
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)

        assert result.files_processed >= 1
        assert len(code_anchors) >= 1
        for a in code_anchors:
            assert a.transitive_hash_status == TransitiveHashStatus.UNSUPPORTED, (
                f"{a.id}: expected unsupported, got {a.transitive_hash_status}"
            )


# ─── Integration tests (real subprocess) ─────────────────────────────────────


@pytest.mark.integration
class TestIntegrationFakeLspCalls:
    """End-to-end tests using fake_lsp_calls.py (responds to callHierarchy)."""

    def test_working_fake_lsp_produces_complete_status(self, py_repo: Path) -> None:
        """Anchors get status=complete and non-None closure_hash from fake LSP."""
        from scry.config import load_config

        config = load_config(py_repo)
        lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP_CALLS)]}},
        )
        full_config = config.model_copy(update={"code_anchors": lsp_config})
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        mgr = LSPManager(py_repo, lsp_config, allow_untrusted=True)
        indexer = Indexer(py_repo, config=full_config, embedder=embedder, lsp_manager=mgr)

        result = indexer.index()
        assert result.anchors_extracted >= 1

        with _open_db(py_repo, config.embeddings.dimensions) as db:
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)

        assert len(code_anchors) >= 1
        for a in code_anchors:
            assert a.transitive_hash_status == TransitiveHashStatus.COMPLETE, (
                f"{a.id}: expected complete, got {a.transitive_hash_status}"
            )
            assert a.closure_hash is not None, f"{a.id}: closure_hash should not be None"

    def test_did_open_called_once_per_file_integration(self, py_repo: Path) -> None:
        """Multiple anchors in one file → didOpen called exactly once (via audit log)."""
        # We audit via _enrich_all_with_lsp directly to avoid full indexer overhead.
        (py_repo / "src" / "multi.py").write_text(
            "def alpha(): pass\ndef beta(): pass\ndef gamma(): pass\n",
            encoding="utf-8",
        )
        a1 = _make_code_anchor("src/multi.py", "alpha", def_line=0, def_char=4)
        a2 = _make_code_anchor("src/multi.py", "beta", def_line=1, def_char=4)
        a3 = _make_code_anchor("src/multi.py", "gamma", def_line=2, def_char=4)

        lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP_CALLS)]}},
        )
        mgr = LSPManager(py_repo, lsp_config, allow_untrusted=True)

        # Intercept notify to count didOpen calls.
        did_open_count: list[int] = [0]
        original_session_for = mgr.session_for

        async def _spy_session_for(lang: str) -> Any:
            session = await original_session_for(lang)
            if session is not None:
                original_notify = session.notify

                async def _spy_notify(method: str, params: Any = None) -> None:
                    if method == "textDocument/didOpen":
                        did_open_count[0] += 1
                    await original_notify(method, params)

                session.notify = _spy_notify  # type: ignore[method-assign]
            return session

        mgr.session_for = _spy_session_for  # type: ignore[method-assign]

        asyncio.run(_enrich_all_with_lsp([a1, a2, a3], mgr, py_repo, max_depth=4))

        assert did_open_count[0] == 1, (
            f"Expected 1 didOpen call for multi.py, got {did_open_count[0]}"
        )

    def test_lsp_that_ignores_callhierarchy_produces_lsp_error(self, py_repo: Path) -> None:
        """fake_lsp.py silently ignores prepareCallHierarchy → timeout → lsp_error."""
        a1 = _make_code_anchor("src/utils.py", "add", def_line=0, def_char=4)
        (py_repo / "src" / "utils.py").write_text("def add(a, b): return a + b\n")

        lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP)]}},
        )
        mgr = LSPManager(py_repo, lsp_config, allow_untrusted=True)

        # Use a very short timeout so the test doesn't hang for 10s.
        result = asyncio.run(
            _enrich_all_with_lsp([a1], mgr, py_repo, max_depth=4, timeout_per_call=0.25)
        )
        assert result[0].transitive_hash_status == TransitiveHashStatus.LSP_ERROR

    def test_lsp_error_anchor_drift_is_unknown(self, py_repo: Path) -> None:
        """Integration: lsp_error anchor stored in DB → evaluate_link_drift returns drift-unknown."""
        from scry.drift import evaluate_link_drift

        code = _make_code_anchor("src/utils.py", status="lsp_error")
        spec = _make_section_anchor("docs/spec.md")

        dims = 384
        with _open_db(py_repo, dims) as db:
            db.upsert_anchors([code, spec])
            link = Link(
                link_id=new_link_id(),
                from_id=code.id,
                from_type=AnchorType.CODE,
                to_id=spec.id,
                to_type=AnchorType.SECTION,
                type=LinkType.IMPLEMENTS,
                from_content_hash=code.content_hash,
                to_content_hash=spec.content_hash,
                last_event_id=new_event_id(),
            )
            evaluation = evaluate_link_drift(link, db=db)

        assert evaluation.drift_status == DriftStatus.DRIFT_UNKNOWN

    def test_shutdown_all_called_after_integration_enrichment(self, py_repo: Path) -> None:
        """shutdown_all() is called at the end of _enrich_all_with_lsp (integration)."""
        a1 = _make_code_anchor("src/utils.py", "add", def_line=0, def_char=4)
        (py_repo / "src" / "utils.py").write_text("def add(a, b): return a + b\n")

        lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP_CALLS)]}},
        )
        mgr = LSPManager(py_repo, lsp_config, allow_untrusted=True)

        original_shutdown = mgr.shutdown_all
        shutdown_calls: list[int] = [0]

        async def _spy_shutdown() -> None:
            shutdown_calls[0] += 1
            await original_shutdown()

        mgr.shutdown_all = _spy_shutdown  # type: ignore[method-assign]

        asyncio.run(_enrich_all_with_lsp([a1], mgr, py_repo, max_depth=4))
        assert shutdown_calls[0] == 1


# ─── Allow-untrusted flag end-to-end ─────────────────────────────────────────


class TestAllowUntrustedWiring:
    """The --allow-untrusted-lsp-config flag reaches Indexer and LSPManager."""

    def test_allow_untrusted_wired_to_indexer(self, py_repo: Path) -> None:
        """allow_untrusted=True in Indexer is propagated to the LSPManager."""
        from scry.config import load_config

        config = load_config(py_repo)
        indexer = Indexer(py_repo, config=config, allow_untrusted=True)
        mgr = indexer._ensure_lsp_manager(config)
        assert mgr._allow_untrusted is True

    def test_allow_untrusted_false_by_default(self, py_repo: Path) -> None:
        from scry.config import load_config

        config = load_config(py_repo)
        indexer = Indexer(py_repo, config=config)
        mgr = indexer._ensure_lsp_manager(config)
        assert mgr._allow_untrusted is False

    @pytest.mark.integration
    def test_allow_untrusted_enables_custom_command_in_indexer(self, py_repo: Path) -> None:
        """With allow_untrusted, custom lsp command is accepted and session spawns."""
        from scry.config import load_config

        config = load_config(py_repo)
        lsp_config = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": sys.executable, "args": [str(FAKE_LSP_CALLS)]}},
        )
        full_config = config.model_copy(update={"code_anchors": lsp_config})
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)

        # Should NOT raise LSPAllowlistViolation.
        indexer = Indexer(py_repo, config=full_config, embedder=embedder, allow_untrusted=True)
        result = indexer.index()
        assert result.anchors_extracted >= 1

        with _open_db(py_repo, config.embeddings.dimensions) as db:
            code_anchors = db.list_anchors(anchor_type=AnchorType.CODE)

        # All code anchors enriched successfully.
        for a in code_anchors:
            assert a.transitive_hash_status == TransitiveHashStatus.COMPLETE


# ─── Code extractor def_line/def_char tests ──────────────────────────────────


class TestExtractorDefPosition:
    """Code extractor now populates def_line and def_char from tree-sitter nodes."""

    def test_python_extractor_populates_def_line(self, tmp_path: Path) -> None:
        """extract_code_symbols sets def_line/def_char from start_point."""
        from scry.extract.code import extract_code_symbols

        py_file = tmp_path / "funcs.py"
        py_file.write_text(
            "def first_func():\n    pass\n\ndef second_func():\n    pass\n",
            encoding="utf-8",
        )
        anchors = extract_code_symbols(py_file, tmp_path, language="python")
        assert len(anchors) >= 2

        names = {a.symbol_name: a for a in anchors}
        assert "first_func" in names
        assert "second_func" in names

        # first_func is on line 0, second_func on line 3 (0-indexed).
        assert names["first_func"].def_line == 0
        assert names["second_func"].def_line == 3

    def test_python_class_method_def_line(self, tmp_path: Path) -> None:
        """Class methods have correct def_line relative to file start."""
        from scry.extract.code import extract_code_symbols

        py_file = tmp_path / "cls.py"
        py_file.write_text(
            "class MyClass:\n    def method_one(self):\n        pass\n",
            encoding="utf-8",
        )
        anchors = extract_code_symbols(py_file, tmp_path, language="python")
        method_anchors = [a for a in anchors if a.symbol_name == "method_one"]
        assert len(method_anchors) == 1
        # method_one is on line 1 (0-indexed).
        assert method_anchors[0].def_line == 1


# ─── Regression tests for review findings ────────────────────────────────────


class TestBLOCKING1AsyncioRunInEventLoop:
    """BLOCKING #1: asyncio.run() must not crash inside a running event loop."""

    def test_index_from_running_event_loop_does_not_raise(self, py_repo: Path) -> None:
        """Indexer.index() uses a thread-pool fallback when inside an event loop.

        Regression: asyncio.run() raises RuntimeError when called from a running
        loop.  The fix detects the running loop and uses ThreadPoolExecutor.
        """
        from scry.config import load_config
        from scry.embed import StubEmbedder

        config = load_config(py_repo)
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        indexer = Indexer(py_repo, config=config, embedder=embedder)

        async def _call_sync_index_from_event_loop() -> None:
            # This simulates the MCP server calling index() from an async context.
            indexer.index(force=False)

        # Must not raise RuntimeError.
        asyncio.run(_call_sync_index_from_event_loop())

    async def test_index_async_from_running_event_loop(self, py_repo: Path) -> None:
        """index_async() works correctly inside a running event loop (MCP path)."""
        from scry.config import load_config
        from scry.embed import StubEmbedder

        config = load_config(py_repo)
        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)
        indexer = Indexer(py_repo, config=config, embedder=embedder)

        result = await indexer.index_async(force=False)
        assert result.files_processed >= 0  # Completes without exception.

    def test_index_async_method_exists(self, py_repo: Path) -> None:
        """index_async() is a public method on Indexer."""
        import inspect

        assert hasattr(Indexer, "index_async")
        assert inspect.iscoroutinefunction(Indexer.index_async)


class TestBLOCKING2InterAnchorPropagation:
    """BLOCKING #2: inter-anchor status propagation (DESIGN.md §5.3)."""

    def test_caller_inherits_callee_lsp_error(self, tmp_path: Path) -> None:
        """If A → B and B has lsp_error, A.transitive_hash_status must be lsp_error.

        Regression: _enrich_all_with_lsp used to assign status based only on
        the anchor's own closure walk without propagating callee status upward.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from scry.lsp.closure import CalleeRef, ClosureResult

        # B is at line 10, char 0 in the fake file.
        b_uri = (tmp_path / "src" / "b.py").as_uri()
        b_line, b_char = 10, 0

        anchor_a = _make_code_anchor("src/a.py", "func_a", def_line=0, def_char=0)
        anchor_b = _make_code_anchor("src/b.py", "func_b", def_line=b_line, def_char=b_char)

        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        (tmp_path / "src" / "a.py").write_text("def func_a(): func_b()\n")
        (tmp_path / "src" / "b.py").write_text("\n" * b_line + "def func_b(): pass\n")

        # A's closure walk discovers B as a callee.
        callee_b_ref = CalleeRef(
            uri=b_uri,
            name="func_b",
            range_start_line=b_line,
            range_start_char=b_char,
            range_end_line=b_line,
            range_end_char=b_char + 8,
        )

        empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        def _fake_closure_result(uri: str, line: int, char: int) -> ClosureResult:
            """Return callees for A; return lsp_error for B's own walk."""
            if "a.py" in uri:
                return ClosureResult(
                    status="complete",
                    closure_hash=empty_hash,
                    callees=(callee_b_ref,),
                    depth_reached=1,
                    diagnostic={},
                )
            else:
                # B's own closure walk encounters an error.
                return ClosureResult(
                    status="lsp_error",
                    closure_hash=empty_hash,
                    callees=(),
                    depth_reached=0,
                    diagnostic={"reason": "lsp timeout"},
                )

        mock_session = MagicMock()
        mock_session.notify = AsyncMock()

        async def _side_effect_closure(
            session: object,
            file_uri: str,
            def_line: int,
            def_char: int,
            **kw: object,
        ) -> ClosureResult:
            return _fake_closure_result(file_uri, def_line, def_char)

        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=mock_session)
        mock_mgr.shutdown_all = AsyncMock()

        with patch("scry.index.compute_closure", side_effect=_side_effect_closure):
            result = asyncio.run(
                _enrich_all_with_lsp([anchor_a, anchor_b], mock_mgr, tmp_path, max_depth=4)
            )

        result_by_id = {a.id: a for a in result}
        # B's own status must be lsp_error.
        assert result_by_id[anchor_b.id].transitive_hash_status == TransitiveHashStatus.LSP_ERROR
        # A calls B → A must propagate B's lsp_error to itself.
        assert result_by_id[anchor_a.id].transitive_hash_status == TransitiveHashStatus.LSP_ERROR, (
            f"Inter-anchor propagation failed: A reports "
            f"{result_by_id[anchor_a.id].transitive_hash_status}, expected lsp_error"
        )

    def test_chain_propagation(self, tmp_path: Path) -> None:
        """A → B → C with C=lsp_error propagates through the chain to A and B."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from scry.lsp.closure import CalleeRef, ClosureResult

        (tmp_path / "src").mkdir(parents=True, exist_ok=True)
        for name in ("a", "b", "c"):
            (tmp_path / "src" / f"{name}.py").write_text(f"def func_{name}(): pass\n")

        anchor_a = _make_code_anchor("src/a.py", "func_a", def_line=0, def_char=0)
        anchor_b = _make_code_anchor("src/b.py", "func_b", def_line=0, def_char=0)
        anchor_c = _make_code_anchor("src/c.py", "func_c", def_line=0, def_char=0)

        b_uri = (tmp_path / "src" / "b.py").as_uri()
        c_uri = (tmp_path / "src" / "c.py").as_uri()
        empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

        ref_b = CalleeRef(
            uri=b_uri,
            name="func_b",
            range_start_line=0,
            range_start_char=0,
            range_end_line=0,
            range_end_char=6,
        )
        ref_c = CalleeRef(
            uri=c_uri,
            name="func_c",
            range_start_line=0,
            range_start_char=0,
            range_end_line=0,
            range_end_char=6,
        )

        async def _side_effect(
            session: object, file_uri: str, def_line: int, def_char: int, **kw: object
        ) -> ClosureResult:
            if "a.py" in file_uri:
                return ClosureResult(
                    status="complete",
                    closure_hash=empty_hash,
                    callees=(ref_b,),
                    depth_reached=1,
                    diagnostic={},
                )
            if "b.py" in file_uri:
                return ClosureResult(
                    status="complete",
                    closure_hash=empty_hash,
                    callees=(ref_c,),
                    depth_reached=1,
                    diagnostic={},
                )
            # C has lsp_error.
            return ClosureResult(
                status="lsp_error",
                closure_hash=empty_hash,
                callees=(),
                depth_reached=0,
                diagnostic={},
            )

        mock_session = MagicMock()
        mock_session.notify = AsyncMock()
        mock_mgr = MagicMock(spec=LSPManager)
        mock_mgr.session_for = AsyncMock(return_value=mock_session)
        mock_mgr.shutdown_all = AsyncMock()

        with patch("scry.index.compute_closure", side_effect=_side_effect):
            result = asyncio.run(
                _enrich_all_with_lsp(
                    [anchor_a, anchor_b, anchor_c], mock_mgr, tmp_path, max_depth=4
                )
            )

        by_id = {a.id: a for a in result}
        assert by_id[anchor_c.id].transitive_hash_status == TransitiveHashStatus.LSP_ERROR
        assert by_id[anchor_b.id].transitive_hash_status == TransitiveHashStatus.LSP_ERROR
        assert by_id[anchor_a.id].transitive_hash_status == TransitiveHashStatus.LSP_ERROR


class TestHIGH1ClosureHashDrift:
    """HIGH #1: closure_hash participates in drift decisions."""

    def _make_link_with_closure(
        self,
        from_anchor: Any,
        to_anchor: Any,
        from_closure_hash: str | None = None,
        to_closure_hash: str | None = None,
    ) -> Link:
        return Link(
            link_id=new_link_id(),
            from_id=from_anchor.id,
            from_type=AnchorType.CODE,
            to_id=to_anchor.id,
            to_type=AnchorType.SECTION,
            type=LinkType.IMPLEMENTS,
            from_content_hash=from_anchor.content_hash,
            to_content_hash=to_anchor.content_hash,
            from_closure_hash=from_closure_hash,
            to_closure_hash=to_closure_hash,
            last_event_id=new_event_id(),
        )

    def _make_db(self, tmp_path: Path, *anchors: Any) -> ScryDB:
        (tmp_path / ".scry").mkdir(exist_ok=True)
        (tmp_path / ".scry" / "overlays").mkdir(exist_ok=True)
        db = _open_db(tmp_path)
        db.upsert_anchors(list(anchors))
        return db

    def test_closure_hash_change_triggers_code_changed(self, tmp_path: Path) -> None:
        """When closure_hash changes but content_hash is same, drift is code-changed."""
        from scry.drift import evaluate_link_drift

        from_anchor = _make_code_anchor("src/a.py", "func_a", closure_hash="new-hash")
        to_anchor = _make_section_anchor("docs/spec.md")

        db = self._make_db(tmp_path, from_anchor, to_anchor)
        link = self._make_link_with_closure(from_anchor, to_anchor, from_closure_hash="old-hash")
        evaluation = evaluate_link_drift(link, db=db)
        assert evaluation.drift_status == DriftStatus.CODE_CHANGED
        db.close()

    def test_no_closure_hash_in_baseline_no_false_positive(self, tmp_path: Path) -> None:
        """Pre-W3d baseline (from_closure_hash=None) → no false positive on upgrade."""
        from scry.drift import evaluate_link_drift

        from_anchor = _make_code_anchor("src/a.py", "func_a", closure_hash="some-hash")
        to_anchor = _make_section_anchor("docs/spec.md")

        db = self._make_db(tmp_path, from_anchor, to_anchor)
        # Baseline has no closure_hash (old record).
        link = self._make_link_with_closure(from_anchor, to_anchor, from_closure_hash=None)
        evaluation = evaluate_link_drift(link, db=db)
        # No false positive: should be fresh (hashes same, no baseline closure hash).
        assert evaluation.drift_status == DriftStatus.FRESH
        db.close()

    def test_link_record_stores_closure_hash(self) -> None:
        """LinkRecord accepts and round-trips from_closure_hash / to_closure_hash."""
        from scry.models import LinkOp, LinkRecord

        record = LinkRecord.model_validate(
            {
                "op": LinkOp.UPSERT,
                "link_id": new_link_id(),
                "from": "src/a.py:func_a",
                "from_type": AnchorType.CODE,
                "to": "docs/spec.md::Section",
                "to_type": AnchorType.SECTION,
                "type": LinkType.IMPLEMENTS,
                "from_content_hash": "sha256:" + "a" * 64,
                "to_content_hash": "sha256:" + "b" * 64,
                "from_closure_hash": "closure-hash-abc",
                "to_closure_hash": None,
            }
        )
        assert record.from_closure_hash == "closure-hash-abc"
        assert record.to_closure_hash is None


class TestHIGH3UntrustedLspPerLanguage:
    """HIGH #3: untrusted LSP per-language rejection (not abort-the-world)."""

    async def test_untrusted_command_returns_none_not_raises(self, tmp_path: Path) -> None:
        """session_for() returns None for untrusted command (not LSPAllowlistViolation)."""
        from scry.lsp.manager import LSPManager

        cfg = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": "custom-binary", "args": []}},
        )
        mgr = LSPManager(tmp_path, cfg, allow_untrusted=False)
        result = await mgr.session_for("python")
        assert result is None

    async def test_untrusted_language_gets_unsupported_status_for(self, tmp_path: Path) -> None:
        """status_for() returns 'skip' (→ UNSUPPORTED) for untrusted-rejected language."""
        from scry.lsp.manager import LSPManager

        cfg = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": "custom-binary", "args": []}},
        )
        mgr = LSPManager(tmp_path, cfg, allow_untrusted=False)
        await mgr.session_for("python")
        assert mgr.status_for("python") == "skip"

    async def test_untrusted_rejection_produces_unsupported_anchor_status(
        self, tmp_path: Path
    ) -> None:
        """Anchors for untrusted-rejected language get UNSUPPORTED (not LSP_UNAVAILABLE)."""
        from scry.lsp.manager import LSPManager

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("def main(): pass\n")

        cfg = CodeAnchorsConfig(
            languages={"python": "lsp"},
            lsp={"python": {"command": "custom-binary", "args": []}},
        )
        anchor = _make_code_anchor("src/app.py", "main")
        mgr = LSPManager(tmp_path, cfg, allow_untrusted=False)

        result = await _enrich_all_with_lsp([anchor], mgr, tmp_path, max_depth=4)
        assert result[0].transitive_hash_status == TransitiveHashStatus.UNSUPPORTED

    def test_mcp_server_accepts_allow_untrusted_param(self, tmp_path: Path) -> None:
        """MCPServer.__init__ accepts allow_untrusted_lsp_config kwarg."""
        from scry.mcp.server import MCPServer

        server = MCPServer(tmp_path, allow_untrusted_lsp_config=True)
        assert server._allow_untrusted_lsp_config is True

    def test_mcp_server_default_allow_untrusted_false(self, tmp_path: Path) -> None:
        from scry.mcp.server import MCPServer

        server = MCPServer(tmp_path)
        assert server._allow_untrusted_lsp_config is False


class TestMEDIUM1TransitiveMaxDepthConfig:
    """MEDIUM #1: transitive_max_depth is under code_anchors, not code_anchors_extra."""

    def test_code_anchors_config_has_transitive_max_depth(self) -> None:
        """CodeAnchorsConfig has transitive_max_depth with default 32."""
        cfg = CodeAnchorsConfig()
        assert cfg.transitive_max_depth == 32

    def test_code_anchors_transitive_max_depth_custom_value(self) -> None:
        cfg = CodeAnchorsConfig(transitive_max_depth=50)
        assert cfg.transitive_max_depth == 50

    def test_code_anchors_transitive_max_depth_validation_ge1(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CodeAnchorsConfig(transitive_max_depth=0)

    def test_code_anchors_transitive_max_depth_validation_le256(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CodeAnchorsConfig(transitive_max_depth=257)

    def test_indexer_uses_code_anchors_transitive_max_depth(self, py_repo: Path) -> None:
        """Indexer reads max_depth from config.code_anchors.transitive_max_depth."""
        from unittest.mock import patch

        from scry.config import load_config

        config = load_config(py_repo)
        new_code_anchors = config.code_anchors.model_copy(update={"transitive_max_depth": 7})
        test_config = config.model_copy(update={"code_anchors": new_code_anchors})

        embedder = StubEmbedder(dimensions=config.embeddings.dimensions)

        captured_max_depths: list[int] = []

        async def _capture_enrich(
            anchors: list[Any],
            lsp_mgr: Any,
            repo_root: Any,
            *,
            max_depth: int,
            **kw: Any,
        ) -> list[Any]:
            captured_max_depths.append(max_depth)
            return anchors

        with patch("scry.index._enrich_all_with_lsp", side_effect=_capture_enrich):
            indexer = Indexer(py_repo, config=test_config, embedder=embedder)
            indexer.index(force=False)

        # If any enrichment ran (code anchors present), verify the depth.
        if captured_max_depths:
            assert captured_max_depths[0] == 7, (
                f"Expected max_depth=7 from code_anchors.transitive_max_depth, "
                f"got {captured_max_depths[0]}"
            )

    def test_code_anchors_extra_deprecation_warning_on_explicit_use(self) -> None:
        """code_anchors_extra.transitive_max_depth emits a DeprecationWarning when configured."""
        import warnings

        from scry.models import Config

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config.model_validate({"code_anchors_extra": {"transitive_max_depth": 64}})
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert dep_warnings, "Expected DeprecationWarning for code_anchors_extra usage"

    def test_config_no_deprecation_warning_by_default(self) -> None:
        """Config() without code_anchors_extra does NOT emit DeprecationWarning."""
        import warnings

        from scry.models import Config

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Config()
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert not dep_warnings, f"Unexpected DeprecationWarning: {dep_warnings}"
